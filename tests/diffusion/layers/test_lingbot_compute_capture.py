# SPDX-License-Identifier: Apache-2.0
"""CPU tests for actual-operator counting and diagnostic capture boundaries."""
import importlib.util
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

_path = Path(__file__).resolve().parents[3] / "vllm_omni/diffusion/layers/lingbot_compute_capture.py"
_spec = importlib.util.spec_from_file_location("lingbot_compute_capture_tests", _path)
capture = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(capture)


def test_real_linear_and_conv_operations_count_once_preserving_results():
    model = nn.Module()
    model.linear = nn.Linear(8, 12)
    model.conv = nn.Conv3d(4, 6, (1, 3, 3), groups=2)
    x, image = torch.randn(3, 8), torch.randn(1, 4, 2, 5, 7)
    expected_linear, expected_conv = model.linear(x), model.conv(image)
    recorder = capture.ComputeRecorder()
    recorder.attach_modules(model)
    recorder.set_phase("probe", 11, 0)
    with recorder:
        actual_linear = model.linear(x)
        actual_conv = model.conv(image)
    recorder.detach_modules()
    assert torch.equal(expected_linear, actual_linear)
    assert torch.equal(expected_conv, actual_conv)
    matrix = [row for row in recorder.records if row["kind"] == "matrix"]
    convolution = [row for row in recorder.records if row["kind"] == "convolution"]
    assert len(matrix) == len(convolution) == 1
    assert matrix[0]["operations_executed_tensor_estimate"] == 2*3*8*12
    assert convolution[0]["operations_executed_tensor_estimate"] == 2*actual_conv.numel()*2*1*3*3
    assert matrix[0]["module"] == "linear"
    assert convolution[0]["operations_useful_serial"] is None
    assert recorder.summary()["whole_model_mfu"] is None


def test_batched_matmul_counts_broadcast_tensor_work_once():
    recorder = capture.ComputeRecorder()
    with recorder:
        result = torch.randn(2, 3, 7) @ torch.randn(2, 7, 5)
    matrices = [row for row in recorder.records if row["kind"] == "matrix"]
    assert sum(row["operations_executed_tensor_estimate"] for row in matrices) == 2*2*3*7*5
    assert result.shape == (2, 3, 5)


def test_replication_label_and_truncation_cannot_be_complete_mfu():
    model = nn.Module()
    model.text_embedding = nn.Linear(4, 8)
    recorder = capture.ComputeRecorder(max_records=1)
    recorder.attach_modules(model)
    recorder.set_phase("startup", -1)
    with recorder:
        model.text_embedding(torch.ones(2, 4))
        model.text_embedding(torch.ones(2, 4))
    recorder.detach_modules()
    summary = recorder.summary()
    assert summary["truncated"]
    assert not summary["capture_complete"]
    assert summary["groups"][0]["replication"] == "SP_replicated_conditioning"
    assert summary["useful_complete_model_operations"] is None


def test_disabled_compute_capture_does_not_attach_hooks(monkeypatch):
    monkeypatch.setenv("LINGBOT_CAPTURE_COMPUTE", "0")
    model = nn.Linear(2, 3)
    capture.begin_compute_capture(model, phase="probe", chunk=11, probe=0)
    assert not hasattr(model, "_lingbot_compute_capture")
    assert not model._forward_pre_hooks


def test_opaque_private_linear_boundary_uses_real_leaf_shapes_once():
    from torch.utils._python_dispatch import _disable_current_modes

    class OpaqueLinear(nn.Linear):
        def forward(self, value):
            with _disable_current_modes():
                return F.linear(value, self.weight, self.bias)

    model = nn.Module()
    model.transformer = nn.Module()
    model.transformer.c2ws_hidden_states_layer1 = OpaqueLinear(8, 12)
    recorder = capture.ComputeRecorder()
    recorder.attach_modules(model)
    value = torch.randn(2, 3, 8)
    recorder.set_phase("probe", 11, 0)
    with recorder:
        result = model.transformer.c2ws_hidden_states_layer1(value)
    recorder.detach_modules()
    assert len(recorder.records) == 1
    row = recorder.records[0]
    assert row["operations_executed_tensor_estimate"] == 2*result.numel()*8
    assert row["operator"] == "modulehook::OpaqueLinear"
    assert row["inputs"][0]["shape"] == list(value.shape)
    assert row["replication"] == "SP_replicated_transformer_prefix"
    assert recorder.weighted_module_calls[0]["observation"] == "weighted_leaf_fallback"


def test_visible_dispatch_owns_leaf_call_and_embedding_is_not_matrix_work():
    model = nn.Module()
    model.linear = nn.Linear(8, 12)
    model.embedding = nn.Embedding(16, 8)
    recorder = capture.ComputeRecorder()
    recorder.attach_modules(model)
    with recorder:
        model.linear(torch.randn(2, 8))
        model.embedding(torch.tensor([1, 2]))
    recorder.detach_modules()
    assert len(recorder.records) == 1
    assert recorder.records[0]["operator"] == "aten::addmm"
    assert recorder.weighted_module_calls[0]["observation"] == "dispatch"
    assert recorder.weighted_module_calls[1]["observation"] == "unresolved"
    assert "embedding indices" in recorder.weighted_module_calls[1]["reason"]


def test_opaque_w8a8_leaf_marks_contract_without_inventing_payload_dtype():
    from torch.utils._python_dispatch import _disable_current_modes

    class OpaqueQuantLinear(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(8, 12, dtype=torch.int8), requires_grad=False)
            self._lingbot_quant_weight_layout = "kn"
            self._lingbot_quant_mode = "w8a8"

        def forward(self, value):
            with _disable_current_modes():
                return value @ self.weight.float()

    model = OpaqueQuantLinear()
    recorder = capture.ComputeRecorder()
    recorder.attach_modules(model)
    with recorder:
        model(torch.randn(3, 8))
    recorder.detach_modules()
    assert len(recorder.records) == 1
    row = recorder.records[0]
    assert row["precision"] == "int8"
    assert row["inputs"][0]["dtype"] == "torch.float32"
    assert row["precision_basis"] == "installed quantization contract"
    assert row["operations_executed_tensor_estimate"] == 2*3*8*12


def test_opaque_direct_convolution_shape_hook_and_composite_no_duplicate():
    from torch.utils._python_dispatch import _disable_current_modes

    # Preserve the audited direct Conv3d class contract while hiding dispatch.
    class Conv3d(nn.Conv3d):
        def forward(self, value):
            with _disable_current_modes():
                return super().forward(value)

    class Composite(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = Conv3d(4, 6, (1, 3, 3), groups=2)
            self.weight = self.conv.weight  # Parent alias must never add work.

        def forward(self, value):
            return self.conv(value)

    model = Composite()
    recorder = capture.ComputeRecorder()
    recorder.attach_modules(model)
    with recorder:
        result = model(torch.randn(1, 4, 2, 5, 7))
    recorder.detach_modules()
    assert len(recorder.records) == 1
    assert recorder.records[0]["operations_executed_tensor_estimate"] == 2*result.numel()*2*1*3*3
    assert recorder.records[0]["module"] == "conv"


def test_audited_spatial_causal_conv_counts_retained_output_without_claiming_pretrim_work():
    from torch.utils._python_dispatch import _disable_current_modes

    class WanDistCausalConv3d(nn.Conv3d):
        split_dim, halo_size = "width", 1

        def forward(self, value):
            with _disable_current_modes():
                output = F.conv3d(value, self.weight, self.bias, groups=self.groups)
                return output[..., 1:-1]

    model = nn.Module()
    model.vae = nn.Module()
    model.vae.decoder = WanDistCausalConv3d(4, 6, (1, 3, 3), groups=2)
    recorder = capture.ComputeRecorder()
    recorder.attach_modules(model)
    recorder.set_phase("decode", 11)
    with recorder:
        output = model.vae.decoder(torch.randn(1, 4, 2, 5, 9))
    recorder.detach_modules()
    row = recorder.records[0]
    assert row["operations_local_useful_dense_estimate"] == 2*output.numel()*2*9
    assert row["pre_trim_executed_operations"] is None
    assert row["split_dim"] == "width"
    assert recorder.summary()["coverage"]["vae_convolution_calls"] == 1
    assert not recorder.summary()["scheduled_compute_regions_coverage_complete"]


def test_opaque_camera_patch_linear_counts_reformatted_matrix_contract():
    from torch.utils._python_dispatch import _disable_current_modes

    class _LingBotCameraPatchEmbedding(nn.Module):
        def __init__(self):
            super().__init__()
            self.in_features = 8
            self.weight = nn.Parameter(torch.randn(12, 8))

        def forward(self, value):
            with _disable_current_modes():
                return F.linear(value.reshape(1, -1, 8), self.weight)

    model = _LingBotCameraPatchEmbedding()
    recorder = capture.ComputeRecorder()
    recorder.attach_modules(model)
    with recorder:
        result = model(torch.randn(1, 2, 2, 4, 6))
    recorder.detach_modules()
    row = recorder.records[0]
    assert row["input_reformatted_by_module"]
    assert row["logical_matrix_input_shape"] == [12, 8]
    assert row["operations_executed_tensor_estimate"] == 2*result.numel()*8


def test_direct_payload_call_can_keep_its_real_module_scope(monkeypatch):
    monkeypatch.setenv("LINGBOT_CAPTURE_COMPUTE", "1")
    model = nn.Module()
    model.ffn = nn.Sequential(nn.Linear(8, 12), nn.GELU(), nn.Linear(12, 8))
    recorder = capture.ComputeRecorder()
    recorder.attach_modules(model)
    monkeypatch.setattr(capture, "_active_compute_recorder", recorder)
    value = torch.randn(3, 12)
    with recorder:
        with capture.compute_module_scope(model.ffn[2], value) as diagnostic:
            diagnostic["output"] = F.linear(value, model.ffn[2].weight, model.ffn[2].bias)
    recorder.detach_modules()
    assert len(recorder.records) == 1
    assert recorder.records[0]["module"] == "ffn.2"
    assert recorder.records[0]["operations_executed_tensor_estimate"] == 2*3*12*8


def test_explicit_attention_owns_qk_pv_when_sdpa_decomposes_to_bmm():
    from torch.nn.attention import SDPBackend, sdpa_kernel
    recorder = capture.ComputeRecorder()
    q, k, v = torch.randn(1, 2, 3, 4), torch.randn(1, 2, 5, 4), torch.randn(1, 2, 5, 4)
    recorder.record_attention(q, k, v, input_layout="BNSD", attention_kind="cross")
    with recorder, sdpa_kernel(SDPBackend.MATH):
        F.scaled_dot_product_attention(q, k, v)
    assert len(recorder.records) == 1
    assert recorder.records[0]["operations_executed_tensor_estimate"] == 4*1*2*3*5*4


def test_explicit_attention_has_dense_useful_shape_and_deduplicates_visible_dispatch():
    recorder = capture.ComputeRecorder()
    recorder.set_phase("probe", 11, 0)
    q, k, v = torch.randn(1, 3, 2, 4), torch.randn(1, 5, 2, 4), torch.randn(1, 5, 2, 4)
    recorder.record_attention(q, k, v, attention_kind="self")
    with recorder:
        F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
    attention = [row for row in recorder.records if row["kind"] == "attention_qk_pv"]
    assert len(attention) == 1
    assert attention[0]["operations_executed_tensor_estimate"] == 4*1*3*5*2*4
    assert attention[0]["observation_source"].startswith("explicit")
    summary = recorder.summary()
    assert summary["record_buffer_complete"]
    assert not summary["capture_complete"]
    assert not summary["full_model_arithmetic_complete"]
    assert summary["coverage"]["self_attention_calls"] == 1


def test_vae_attention_uses_actual_qkv_geometry_and_counts_replication_once():
    class WanAttentionBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.to_qkv = nn.Conv2d(4, 12, 1)
            self.proj = nn.Conv2d(4, 4, 1)
            self._vllm_omni_spatial_shard_attention = True

        def forward(self, value):
            b, c, h, w = value.shape
            q, k, v = self.to_qkv(value).reshape(b, 3, c, h*w).permute(0, 1, 3, 2).unbind(1)
            result = F.scaled_dot_product_attention(q[:, None], k[:, None], v[:, None])
            return self.proj(result[:, 0].transpose(1, 2).reshape(b, c, h, w))

    model = nn.Module()
    model.vae = nn.Module()
    model.vae.decoder = nn.Module()
    model.vae.decoder.attention = WanAttentionBlock()
    model.vae.post_quant_conv = nn.Conv2d(4, 4, 1)
    recorder = capture.ComputeRecorder()
    recorder.attach_modules(model)
    recorder.set_phase("decode", 11)
    image = torch.randn(2, 4, 3, 5)
    expected = model.vae.decoder.attention(image)
    with recorder:
        actual = model.vae.decoder.attention(image)
        model.vae.post_quant_conv(image)
    recorder.detach_modules()
    assert torch.equal(actual, expected)
    attention = [row for row in recorder.records if row["kind"] == "attention_qk_pv"]
    assert len(attention) == 1
    assert attention[0]["attention_kind"] == "vae"
    assert attention[0]["inputs"][0]["shape"] == [2, 1, 15, 4]
    assert attention[0]["operations_executed_tensor_estimate"] == 4*2*15*15*4
    assert attention[0]["replication"] == "SP_replicated_VAE"
    convs = [row for row in recorder.records if row["kind"] == "convolution"]
    assert len(convs) == 3
    assert all(row["replication"] == "SP_replicated_VAE" for row in convs)
    assert recorder.summary()["coverage"]["vae_attention_calls"] == 1
