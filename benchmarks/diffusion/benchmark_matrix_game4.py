# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Bounded Matrix-Game qualification with real geometry frames and optional oracle.

Run the reference and Omni modes in separate processes on the same GPU/software
stack. See recipes/Skywork/Matrix-Game-4.0-SM120.md for the pinned reference.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

import numpy as np
import torch

from vllm_omni.platforms import current_omni_platform


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resources", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path, help="uint8 THWC .npy geometry frames at 864x480")
    parser.add_argument("--output", required=True, type=Path, help="output prefix for .npy frames and .json metrics")
    parser.add_argument("--mode", choices=("omni", "reference"), default="omni")
    parser.add_argument("--frames", type=int, default=241)
    parser.add_argument("--steps", type=int, choices=(1, 2), default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup-chunks", type=int, default=25)
    parser.add_argument("--compare", type=Path, help="reference output .npy to compare against")
    args = parser.parse_args()
    if args.frames < 1 or (args.frames - 1) % 4 or args.warmup_chunks < 0:
        parser.error("--frames must be 1+4k and --warmup-chunks must be nonnegative")
    return args


async def render_omni(args: argparse.Namespace, video: np.ndarray) -> tuple[np.ndarray, list[float]]:
    from vllm_omni.diffusion.data import OmniDiffusionConfig, VideoOutputTransportConfig
    from vllm_omni.diffusion.diffusion_engine import DiffusionEngine
    from vllm_omni.diffusion.models.matrix_game4.pipeline_matrix_game4 import END_KEY, SESSION_KEY, START_KEY
    from vllm_omni.diffusion.request import OmniDiffusionRequest
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    config = OmniDiffusionConfig(
        model=str(args.resources),
        model_class_name="MatrixGame4Pipeline",
        enforce_eager=True,
        num_gpus=1,
        max_num_seqs=1,
        distributed_executor_backend="uni",
        video_output_transport=VideoOutputTransportConfig(enable_device_postprocess=True),
    )
    engine = DiffusionEngine(config)
    outputs, timings = [], []
    try:
        for start in [0, *range(1, args.frames, 4)]:
            size = 1 if start == 0 else 4
            # The input owns writable host storage, independent of the mmap.
            pixels = torch.from_numpy(np.array(video[start : start + size]))
            request = OmniDiffusionRequest(
                prompt={"prompt": "", "multi_modal_data": {"video": pixels}},
                request_id=f"matrix-{start}",
                sampling_params=OmniDiffusionSamplingParams(
                    seed=args.seed,
                    num_inference_steps=args.steps,
                    output_type="np",
                    extra_args={SESSION_KEY: "qualification", START_KEY: start, END_KEY: start + size == args.frames},
                ),
            )
            current_omni_platform.synchronize()
            begin = time.perf_counter()
            chunk = None
            async for results in engine.step_streaming(request):
                for result in results:
                    if result.images:
                        if len(result.images) != 1:
                            raise RuntimeError("Expected one batched video payload")
                        chunk = result.images[0]
            current_omni_platform.synchronize()
            timings.append(time.perf_counter() - begin)
            if chunk is None or chunk.shape != (1, size, 480, 864, 3) or chunk.dtype != np.uint8:
                raise RuntimeError(
                    f"Invalid Omni output at frame {start}: "
                    f"type={type(chunk)}, shape={getattr(chunk, 'shape', None)}, "
                    f"dtype={getattr(chunk, 'dtype', None)}"
                )
            outputs.append(chunk[0])
    finally:
        engine.close()
    return np.concatenate(outputs), timings


def render_reference(args: argparse.Namespace, video: np.ndarray) -> tuple[np.ndarray, list[float]]:
    # Reference-only imports: install the pinned oracle separately for this mode.
    from matrix_game.inference.loader import ModelComponents
    from matrix_game.scenes import add_model_arguments, resolve_model_arguments

    parser = argparse.ArgumentParser()
    add_model_arguments(parser)
    options = parser.parse_args(
        [
            "--scene",
            "castle",
            "--resources",
            str(args.resources),
            "--preset",
            "rtx5090",
            "--steps",
            "1000" if args.steps == 1 else "1000,500",
        ]
    )
    resolve_model_arguments(options)
    components = ModelComponents(options)
    components.load()
    stream = components.stream(seed=args.seed)
    outputs, timings = [], []
    try:
        for start in [0, *range(1, args.frames, 4)]:
            size = 1 if start == 0 else 4
            pixels = torch.from_numpy(np.array(video[start : start + size])).permute(0, 3, 1, 2).float()
            pixels = pixels * (2.0 / 255.0) - 1.0
            current_omni_platform.synchronize()
            begin = time.perf_counter()
            stream.push(pixels, frame_start=start, on_rgb=outputs.append)
            current_omni_platform.synchronize()
            timings.append(time.perf_counter() - begin)
    finally:
        stream.close()
    return torch.cat(outputs).numpy(), timings


def main() -> None:
    import vllm

    args = arguments()
    video = np.load(args.input, mmap_mode="r")
    if video.dtype != np.uint8 or video.ndim != 4 or video.shape[1:] != (480, 864, 3) or len(video) < args.frames:
        raise ValueError("Input must contain enough uint8 RGB geometry frames at 864x480")
    frames, timings = asyncio.run(render_omni(args, video)) if args.mode == "omni" else render_reference(args, video)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output.with_suffix(".npy"), frames)
    measured = timings[args.warmup_chunks :]
    report = {
        "mode": args.mode,
        "frames": len(frames),
        "steps": args.steps,
        "seed": args.seed,
        "input": str(args.input),
        "resources": str(args.resources),
        "warmup_chunks": args.warmup_chunks,
        "chunk_seconds": timings,
        "torch": torch.__version__,
        "vllm": vllm.__version__,
        "cuda_runtime": torch.version.cuda,
        "timing_scope": "synchronized chunk execution; includes clean KV; Omni includes engine and postprocess",
    }
    if measured:
        report.update(
            steady_fps=4 * len(measured) / sum(measured),
            chunk_p50_ms=float(np.percentile(measured, 50) * 1000),
            chunk_p95_ms=float(np.percentile(measured, 95) * 1000),
        )
    if args.compare:
        reference = np.load(args.compare, mmap_mode="r")
        if reference.shape != frames.shape or reference.dtype != frames.dtype:
            raise ValueError("Reference output geometry/dtype differs")
        absolute = np.abs(frames.astype(np.float32) - reference.astype(np.float32))
        report.update(
            pixel_mae=float(absolute.mean()),
            pixel_max_error=float(absolute.max()),
            exact_match=bool(np.array_equal(frames, reference)),
        )
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "chunk_seconds"}, indent=2))


if __name__ == "__main__":
    main()
