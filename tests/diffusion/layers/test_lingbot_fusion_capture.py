# SPDX-License-Identifier: Apache-2.0
"""CPU checks for the actual decoder operand capture hooks."""
import importlib.util
from pathlib import Path

import torch
from torch import nn

_path = Path(__file__).resolve().parents[3] / "vllm_omni/diffusion/layers/lingbot_fusion_capture.py"
_spec = importlib.util.spec_from_file_location("lingbot_fusion_capture_tests", _path)
capture = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(capture)


class RMSNormVAE(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(channels, 1, 1, 1))
        self.channel_first = True
        self.epsilon = 1e-6
        self.scale = 1.0

    def forward(self, value):
        return value


def test_real_vae_pre_hook_preserves_input_and_records_source_name(monkeypatch):
    monkeypatch.setenv("LINGBOT_CAPTURE_ACTIVATIONS", "1")
    vae = nn.Module()
    vae.decoder = nn.Module()
    vae.decoder.norm_out = RMSNormVAE(96)
    vae.decoder.norm1 = RMSNormVAE(192)
    vae.decoder.extra = RMSNormVAE(192)
    records = []

    # Match the real writer signature to catch collisions between its owner
    # argument and descriptive metadata before launching a model session.
    def write(module, kind, tensors, **metadata):
        records.append((module, kind, tensors, metadata))

    monkeypatch.setattr(capture, "capture_fusion_operands", write)
    capture.install_vae_capture(vae)
    capture.install_vae_capture(vae)
    assert len(vae._lingbot_vae_fusion_capture_hooks) == 2
    for channels, norm in ((96, vae.decoder.norm_out), (192, vae.decoder.norm1)):
        value = torch.randn(1, channels, 1, 2, 3)
        assert norm(value) is value
        owner, kind, tensors, metadata = records[-1]
        assert owner is norm
        assert kind == f"vae_rms_silu_C{channels}"
        assert tensors["value"] is value
        assert tensors["gamma"] is norm.gamma
        assert metadata["source_module"] in {"decoder.norm_out", "decoder.norm1"}
        assert metadata["source_contract"] == "npu_rms_norm"
        assert metadata["eps"] == norm.epsilon
        assert metadata["upcast"] is False
    assert not vae.decoder.extra._forward_pre_hooks


def test_disabled_vae_capture_attaches_no_hooks(monkeypatch):
    monkeypatch.setenv("LINGBOT_CAPTURE_ACTIVATIONS", "0")
    vae = nn.Module()
    vae.decoder = nn.Module()
    vae.decoder.norm_out = RMSNormVAE(96)
    capture.install_vae_capture(vae)
    assert not vae.decoder.norm_out._forward_pre_hooks
    assert not hasattr(vae, "_lingbot_vae_fusion_capture_hooks")
