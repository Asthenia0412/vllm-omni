#!/usr/bin/env python3
"""Helios AutoencoderKLWan VAE CUDA Graph capture and benchmark.

W5 diffusion conditioning encoder timing script for issue #7452.
NOT intended for merge — this is a standalone profiling helper.

Captures tiered CUDA graphs for Helios VAE encode:
  - Tier 1: (frames=9, 17, 33) x default spatial resolution
  - Each tier captures once, replays N times in the chunk loop

Usage:
    python helios_vae_cudagraph_benchmark.py \
        --model BestWishYsh/Helios-Base \
        --device cuda:0 \
        --num_warmup 5 --num_iters 20
"""

from __future__ import annotations

import argparse
import time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class CaptureTier:
    """One CUDA graph capture keyed by (temporal_frames, height, width)."""

    frames: int
    height: int
    width: int

    @property
    def key(self) -> str:
        return f"f{self.frames}_h{self.height}_w{self.width}"


class HeliosVaeCudaGraphCapture:
    """Capture and replay tiered CUDA graphs for AutoencoderKLWan.encode().

    Design notes (from source analysis on issue #7452):
      - VAE encode is deterministic (no random ops inside forward)
      - Per-chunk input shape is invariant: (B, C, min_frames, H, W)
      - Randomness (latent_dist.sample) happens AFTER encode returns
      - No multi-GPU communication in encode path
      - chunk_count varies per request but per-chunk shape does NOT
    """

    def __init__(
        self,
        vae: torch.nn.Module,
        device: torch.device,
        dtype: torch.dtype,
        batch_size: int = 1,
        capture_stream: Optional[torch.cuda.Stream] = None,
    ):
        self.vae = vae
        self.device = device
        self.dtype = dtype
        self.batch_size = batch_size
        self.capture_stream = capture_stream or torch.cuda.Stream(device=device)
        self.graphs: dict[str, tuple[torch.cuda.CUDAGraph, torch.Tensor, torch.Tensor]] = {}
        self._vae_channel_in = self._infer_in_channels()

    def _infer_in_channels(self) -> int:
        """AutoencoderKLWan takes RGB (C=3) but we verify from config."""
        cfg = getattr(self.vae, "config", None)
        if cfg is not None:
            return getattr(cfg, "in_channels", 3)
        return 3

    def capture(self, tier: CaptureTier, num_warmup: int = 5) -> torch.cuda.CUDAGraph:
        """Capture a CUDA graph for this tier.

        Returns the graph; also stores static input buffer and lazy output
        so replay() can inject real data without recapturing.
        """
        key = tier.key
        if key in self.graphs:
            return self.graphs[key][0]

        # Static input buffer — same zeros-shape every capture, replay
        # overwrites it with real video_chunk data.
        static_input = torch.zeros(
            self.batch_size,
            self._vae_channel_in,
            tier.frames,
            tier.height,
            tier.width,
            device=self.device,
            dtype=self.dtype,
        )

        # Warmup: run eager so cuDNN picks algorithms and cudnnFindConvolutionForward
        # results are valid for capture.
        self.vae.eval()
        with torch.no_grad():
            for _ in range(num_warmup):
                _ = self.vae(static_input)

        torch.cuda.synchronize(self.device)

        # Capture on dedicated stream so default stream is unaffected.
        self.capture_stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(self.capture_stream):
            self.capture_stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=self.capture_stream):
                static_output = self.vae(static_input)
        self.capture_stream.synchronize()
        torch.cuda.current_stream(self.device).wait_stream(self.capture_stream)

        self.graphs[key] = (graph, static_input, static_output)
        return graph

    def replay(self, tier: CaptureTier, real_input: torch.Tensor) -> torch.Tensor:
        """Copy real_input into static buffer and replay the captured graph."""
        key = tier.key
        if key not in self.graphs:
            raise KeyError(f"No captured graph for tier {key}")
        graph, static_input, _ = self.graphs[key]
        static_input.copy_(real_input)
        graph.replay()
        # Output aliases the static buffer — caller clones if retention needed.
        _, _, static_output = self.graphs[key]
        return static_output.sample() if hasattr(static_output, "sample") else static_output

    def replay_chunk_loop(self, tier: CaptureTier, chunks: list[torch.Tensor]) -> list[torch.Tensor]:
        """Replay N chunks through the same captured graph.

        This is the Helios V2V hot path — N identical-shape calls.
        """
        results = []
        for chunk in chunks:
            results.append(self.replay(tier, chunk))
        return results


def benchmark_eager(
    vae: torch.nn.Module,
    chunks: list[torch.Tensor],
    num_iters: int,
    device: torch.device,
) -> tuple[float, list[torch.Tensor]]:
    """Eager baseline: run vae.encode for each chunk."""
    vae.eval()
    # Warmup
    with torch.no_grad():
        for c in chunks:
            _ = vae(c)
    torch.cuda.synchronize(device)

    start = time.perf_counter()
    outputs = []
    for _ in range(num_iters):
        for c in chunks:
            with torch.no_grad():
                out = vae(c)
            outputs.append(out)
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    return elapsed, outputs


def benchmark_graph(
    capture: HeliosVaeCudaGraphCapture,
    tier: CaptureTier,
    chunks: list[torch.Tensor],
    num_iters: int,
    device: torch.device,
) -> float:
    """Graph replay baseline."""
    capture.capture(tier, num_warmup=3)
    torch.cuda.synchronize(device)

    start = time.perf_counter()
    for _ in range(num_iters):
        capture.replay_chunk_loop(tier, chunks)
    torch.cuda.synchronize(device)
    return time.perf_counter() - start


def run_timing_comparison(
    model_path: str,
    device: torch.device,
    num_warmup: int = 5,
    num_iters: int = 20,
) -> None:
    """Run eager vs CUDA graph for default Helios chunks."""
    try:
        from diffusers import AutoencoderKLWan
    except ImportError:
        print("[ERROR] diffusers not installed. pip install diffusers")
        return

    print(f"[Helios-VAE-CUDAGRAPH] Loading VAE from {model_path} ...")
    vae = AutoencoderKLWan.from_pretrained(
        model_path, subfolder="vae", torch_dtype=torch.float32
    ).to(device)
    vae.eval()

    # Typical Helios configs (from source analysis):
    # vae_scale_factor_temporal = 4, so min_frames = (npfpc - 1) * 4 + 1
    default_spatial_resolutions = [(720, 1280)]  # production default
    chunk_configs = [
        9,   # num_latent_frames_per_chunk -> 33 frames
        5,   # -> 17 frames
        3,   # -> 9 frames
    ]

    dtype = vae.dtype
    print(f"[Helios-VAE-CUDAGRAPH] VAE dtype={dtype}, device={device}")

    for npfpc in chunk_configs:
        min_frames = (npfpc - 1) * 4 + 1
        for h, w in default_spatial_resolutions:
            tier = CaptureTier(frames=min_frames, height=h, width=w)
            print(f"\n=== Tier {tier.key} (npfpc={npfpc}, min_frames={min_frames}, HxW={h}x{w}) ===")

            # Simulate V2V chunk loop with 3 chunks (typical)
            num_chunks = 3
            chunks = [
                torch.randn(1, 3, min_frames, h, w, device=device, dtype=dtype)
                for _ in range(num_chunks)
            ]

            # Eager
            eager_time, eager_outputs = benchmark_eager(vae, chunks, num_iters, device)
            eager_per_iter = eager_time / num_iters
            print(f"  Eager:  {eager_time:.4f}s total over {num_iters} iters "
                  f"({eager_per_iter*1000:.2f}ms/iter, {eager_per_iter/num_chunks*1000:.2f}ms/chunk)")

            # Graph
            capture = HeliosVaeCudaGraphCapture(vae, device, dtype)
            graph_time = benchmark_graph(capture, tier, chunks, num_iters, device)
            graph_per_iter = graph_time / num_iters
            speedup = eager_per_iter / graph_per_iter if graph_per_iter > 0 else float("inf")
            print(f"  Graph:  {graph_time:.4f}s total over {num_iters} iters "
                  f"({graph_per_iter*1000:.2f}ms/iter, {graph_per_iter/num_chunks*1000:.2f}ms/chunk)")
            print(f"  Speedup: {speedup:.2f}x")

            # Verify correctness
            with torch.no_grad():
                graph_out = capture.replay(tier, chunks[0])
                eager_out = vae(chunks[0])
            err = (graph_out.sample() - eager_out.sample()).abs().max().item()
            print(f"  Max abs error (replay vs eager): {err:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Helios VAE CUDA Graph benchmark")
    parser.add_argument("--model", default="BestWishYsh/Helios-Base", help="Helios HF model ID")
    parser.add_argument("--device", default="cuda:0", help="CUDA device")
    parser.add_argument("--num_warmup", type=int, default=5)
    parser.add_argument("--num_iters", type=int, default=20)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if "cuda" not in str(device):
        print("[WARN] No CUDA available — capture requires GPU. Exiting.")
    else:
        run_timing_comparison(args.model, device, args.num_warmup, args.num_iters)
