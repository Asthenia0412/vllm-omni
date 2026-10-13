# SPDX-License-Identifier: Apache-2.0
"""Explicitly opt-in diagnostics; captured sessions are not timing benchmarks."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re

import numpy as np
import torch

_active_capture = None


def _rank() -> int:
    return torch.distributed.get_rank() if torch.distributed.is_initialized() else 0


def _save_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def _tensor_copy(value: torch.Tensor) -> torch.Tensor:
    return value.detach().to(device="cpu").contiguous()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class DiagnosticCapture:
    def __init__(self, destination: Path, session_id: str):
        self.session_id = session_id
        self.directory = destination / re.sub(r"[^a-zA-Z0-9_.-]", "_", session_id)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.chunk = -1
        self.probe = -1
        self.attention_seen = False
        self.records = []
        self.activation_records = []
        self.saved_inputs = set()
        self.activation_limit = int(os.environ.get("LINGBOT_CAPTURE_ACTIVATION_CHUNKS", "6"))
        self.capture_activations = os.environ.get("LINGBOT_CAPTURE_ACTIVATIONS", "0") == "1"
        self.payload_directory = self.directory
        spool = os.environ.get("LINGBOT_CAPTURE_PAYLOAD_SPOOL_DIR")
        if spool:
            root = Path(spool).expanduser().resolve()
            if self.capture_activations or not str(root).startswith("/tmp/lingbot-generated-"):
                raise ValueError("Generated-payload spooling requires a dedicated temporary directory and no activation/weight capture")
            self.payload_directory = root / self.directory.name
            self.payload_directory.mkdir(parents=True, exist_ok=False)
        self._publish()

    def _publish(self):
        _save_json(self.directory / "capture.json", {
            "session_id": self.session_id, "rank": _rank(),
            "timing_eligible": False, "reason": "synchronous diagnostic copies and file writes",
            "rgb_contract": "uint8 FHWC, before video encoding",
            "latent_contract": "original model-space dtype and shape, before VAE normalization",
            "chunks": self.records, "activations": self.activation_records,
            "payload_directory": str(self.payload_directory),
            "payload_storage_persistent": any(self.payload_directory.is_relative_to(root)
                                              for root in (Path("/mnt/data"), Path("/mnt/sfs_turbo"))),
        })

    def begin_probe(self, chunk: int, probe: int):
        self.chunk, self.probe = int(chunk), int(probe)
        self.attention_seen = False

    def attention(self, query, key, value, head_num, scale):
        if not self.capture_activations or self.attention_seen or not 0 <= self.chunk < self.activation_limit:
            return
        if self.probe not in range(4) or tuple(query.shape) != (1, 4680, 10, 128):
            return
        self.attention_seen = True
        path = self.directory / f"attention_chunk{self.chunk:03d}_probe{self.probe}.pt"
        temporary = path.with_suffix(".tmp")
        torch.save({"query": _tensor_copy(query), "key": _tensor_copy(key), "value": _tensor_copy(value),
                    "head_num": int(head_num), "scale": float(scale), "chunk": self.chunk,
                    "probe": self.probe, "rank": _rank()}, temporary)
        temporary.replace(path)
        self.activation_records.append({"kind": "first_layer_attention", "path": path.name,
                                        "chunk": self.chunk, "probe": self.probe,
                                        "query_shape": list(query.shape), "key_shape": list(key.shape)})

    def linear_input(self, name, module, arguments):
        if not self.capture_activations or not 0 <= self.chunk < self.activation_limit or self.probe not in range(4):
            return
        signature = (self.chunk, self.probe, name)
        if signature in self.saved_inputs or not arguments or not isinstance(arguments[0], torch.Tensor):
            return
        self.saved_inputs.add(signature)
        path = self.directory / f"input_chunk{self.chunk:03d}_probe{self.probe}_{name.replace('.', '_')}.pt"
        temporary = path.with_suffix(".tmp")
        torch.save({"input": _tensor_copy(arguments[0]), "module": name,
                    "chunk": self.chunk, "probe": self.probe}, temporary)
        temporary.replace(path)
        self.activation_records.append({"kind": "first_layer_input", "path": path.name,
                                        "module": name, "chunk": self.chunk, "probe": self.probe})

    def output(self, chunk: int, latents: torch.Tensor, rgb: np.ndarray | None):
        if chunk != len(self.records):
            raise RuntimeError(f"Diagnostic output ordering mismatch: expected {len(self.records)}, got {chunk}")
        stem = f"chunk{chunk:04d}"
        latent_path = self.payload_directory / (stem + ".latents.pt")
        temporary = latent_path.with_suffix(".tmp")
        torch.save(_tensor_copy(latents), temporary)
        temporary.replace(latent_path)
        record = {"chunk": chunk, "latent_path": latent_path.name,
                  "latent_shape": list(latents.shape), "latent_dtype": str(latents.dtype),
                  "latent_sha256": _file_hash(latent_path)}
        if rgb is not None:
            if rgb.dtype != np.uint8 or rgb.ndim != 4 or rgb.shape[-1] != 3:
                raise ValueError("Diagnostic RGB requires uint8 FHWC")
            rgb_path = self.payload_directory / (stem + ".rgb.npy")
            temporary = rgb_path.with_suffix(".tmp")
            with temporary.open("wb") as output:
                np.save(output, rgb, allow_pickle=False)
            temporary.replace(rgb_path)
            record.update(rgb_path=rgb_path.name, rgb_shape=list(rgb.shape),
                          rgb_file_sha256=_file_hash(rgb_path),
                          rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest())
        self.records.append(record)
        self._publish()


def begin_capture(pipeline, session_id: str):
    """Install first-layer input hooks once; switch the writer on session start."""
    global _active_capture
    selected = int(os.environ.get("LINGBOT_CAPTURE_RANK", "0"))
    if selected != -1 and _rank() != selected:
        _active_capture = None
        return
    destination = Path(os.environ["LINGBOT_CAPTURE_DIR"]).expanduser().resolve()
    if not any(destination.is_relative_to(root) for root in (Path("/mnt/data"), Path("/mnt/sfs_turbo"))):
        raise ValueError("Diagnostic captures must use persistent /mnt/data or /mnt/sfs_turbo storage")
    if selected == -1:
        destination = destination / f"rank{_rank()}"
    _active_capture = DiagnosticCapture(destination, session_id)
    pipeline._lingbot_diagnostic_capture = _active_capture
    if _active_capture.capture_activations:
        from .lingbot_fusion_capture import install_vae_capture

        install_vae_capture(pipeline.vae)
    if _active_capture.capture_activations and not getattr(pipeline, "_lingbot_capture_hooks", None):
        hooks = []
        names = {"blocks.0.self_attn.qkv", "blocks.0.self_attn.o", "blocks.0.cross_attn.q",
                 "blocks.0.cross_attn.o", "blocks.0.ffn.0", "blocks.0.ffn.2"}
        for name, module in pipeline.transformer.named_modules():
            if name in names:
                def callback(module, arguments, name=name):
                    if _active_capture is not None:
                        _active_capture.linear_input(name, module, arguments)
                hooks.append(module.register_forward_pre_hook(callback))
        pipeline._lingbot_capture_hooks = hooks


def begin_probe(pipeline, chunk: int, probe: int):
    writer = getattr(pipeline, "_lingbot_diagnostic_capture", None)
    if writer is not None:
        writer.begin_probe(chunk, probe)


def capture_attention(query, key, value, *, head_num: int, scale: float):
    if _active_capture is not None:
        _active_capture.attention(query, key, value, head_num, scale)


def capture_output(pipeline, chunk: int, latents: torch.Tensor, rgb: np.ndarray | None):
    writer = getattr(pipeline, "_lingbot_diagnostic_capture", None)
    if writer is not None:
        writer.output(int(chunk), latents, rgb)
