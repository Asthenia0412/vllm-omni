# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU numerical and loader-contract tests; no NPU is initialized or used."""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

pytestmark = [pytest.mark.cpu, pytest.mark.diffusion]

# Avoid the package __init__, which imports the distributed serving pipeline.
_path = Path(__file__).resolve().parents[4] / "vllm_omni/diffusion/models/lingbot_world/npu_quantization.py"
_spec = importlib.util.spec_from_file_location("lingbot_npu_quantization_cpu_tests", _path)
quant = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = quant
_spec.loader.exec_module(quant)


class FakeUnquantizedLinearMethod:
    def apply(self, layer, value, bias=None):
        return F.linear(value, layer.weight, bias)


class ParallelLinear(nn.Module):
    def __init__(self, k=32, n=48, *, skip_bias_add=False, return_bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n, k) / k**0.5)
        self.bias = nn.Parameter(torch.randn(n))
        self.tp_size = 1
        self.quant_method = FakeUnquantizedLinearMethod()
        self.custom_op = object()
        self.skip_bias_add = skip_bias_add
        self.return_bias = return_bias

    def forward(self, value):
        output = self.quant_method.apply(self, value, None if self.skip_bias_add else self.bias)
        return (output, self.bias if self.skip_bias_add else None) if self.return_bias else output


def _transformer():
    block = nn.Module()
    block.self_attn = nn.Module()
    block.self_attn.qkv = ParallelLinear(n=96)
    block.self_attn.o = ParallelLinear(n=32)
    block.cross_attn = nn.Module()
    for name in ("q", "k", "v", "o"):
        setattr(block.cross_attn, name, ParallelLinear(n=32))
    block.ffn = nn.Sequential(ParallelLinear(n=64), nn.GELU(), ParallelLinear(k=64, n=32))
    block.cam_injector_layer1 = ParallelLinear(n=32)
    block.cam_injector_layer2 = ParallelLinear(n=32)
    block.cam_scale_layer = nn.Linear(32, 32)
    block.cam_shift_layer = nn.Linear(32, 32)
    model = nn.Module()
    model.blocks = nn.ModuleList([block])
    model.head = nn.Linear(32, 8)
    model.time_embedding = nn.Linear(32, 32)
    return model


def _config(mode="w8a8", scope="hot", **kwargs):
    return quant.NpuQuantizationConfig(mode=mode, scope=scope, require_npu=False, **kwargs)


@pytest.mark.parametrize("initial", [False, True])
@pytest.mark.parametrize("failure", [None, "fallback", "payload", "cast"])
def test_one_time_nz_layout_restores_execution_policy_even_on_failure(initial, failure):
    from types import SimpleNamespace
    state = {"enabled": initial}

    class WriteOnlyConfig:
        def __getattribute__(self, name):
            if name == "allow_internal_format":
                raise AttributeError(name)
            return object.__getattribute__(self, name)

        def __setattr__(self, name, value):
            assert name == "allow_internal_format"
            state["enabled"] = bool(value)

    policy = WriteOnlyConfig()
    calls = []

    def cast(value, target):
        calls.append((target, state["enabled"]))
        assert state["enabled"]
        if failure == "cast":
            raise RuntimeError("cast failed")
        result = value.clone()
        if target == 2 and failure == "payload":
            result[0, 0] += 1
        return result

    ops = SimpleNamespace(npu=SimpleNamespace(config=policy),
                          _C=SimpleNamespace(_npu_getOption=lambda key: b"enable" if state["enabled"] else b"disable"),
                          npu_format_cast=cast,
                          get_npu_format=lambda unused: 2 if failure == "fallback" else 29)
    value = torch.ones(8, 12, dtype=torch.int8)
    if failure is None:
        output = quant._convert_weight_to_nz(value, ops)
        assert torch.equal(value, output)
        assert calls == [(29, True), (2, True)]
    else:
        with pytest.raises(RuntimeError):
            quant._convert_weight_to_nz(value, ops)
    assert state["enabled"] is initial


@pytest.mark.parametrize("unknown", [None, b"", b"unknown", "false", 0])
def test_nz_layout_rejects_unknown_execution_policy_before_conversion(unknown):
    from types import SimpleNamespace
    calls = []
    ops = SimpleNamespace(_C=SimpleNamespace(_npu_getOption=lambda key: unknown),
                          npu_format_cast=lambda *args: calls.append(args))
    with pytest.raises(RuntimeError, match="faithfully restore"):
        quant._convert_weight_to_nz(torch.ones(4, 4, dtype=torch.int8), ops)
    assert not calls


def test_default_is_exact_bf16_path():
    model = _transformer()
    before = {name: p.clone() for name, p in model.named_parameters()}
    result = quant.install_lingbot_npu_quantization(model, quant.NpuQuantizationConfig.from_env({}))
    assert result["selected_layers"] == 0
    for name, p in model.named_parameters():
        assert torch.equal(before[name], p)
    assert not hasattr(model, "_lingbot_npu_quantization_config")


@pytest.mark.parametrize("scope,count", [("ffn", 2), ("attention", 4), ("hot", 6), ("all", 12)])
def test_selection_and_checkpoint_parameter_namespace(scope, count):
    model = _transformer()
    before = set(dict(model.named_parameters()))
    result = quant.install_lingbot_npu_quantization(model, _config(scope=scope))
    assert result["selected_layers"] == count
    assert before == set(dict(model.named_parameters()))
    assert model.head.weight.dtype == torch.float32
    assert model.time_embedding.weight.dtype == torch.float32
    if scope != "all":
        assert model.blocks[0].cam_scale_layer.weight.dtype == torch.float32
        assert model.blocks[0].cross_attn.k.weight.dtype == torch.float32


@pytest.mark.parametrize("scope,count", [
    ("qkv", 1), ("self_out", 1), ("cross_q", 1), ("cross_out", 1),
    ("ffn_qkv", 3), ("ffn_self", 4), ("ffn_self_crossq", 5), ("ffn_self_crosso", 5),
])
def test_controlled_projection_stages_protect_conditioning(scope, count):
    model = _transformer()
    protected = {
        name: p.clone() for name, p in model.named_parameters()
        if any(part in name for part in ("cam_", "cross_attn.k", "cross_attn.v", "head", "embedding"))
    }
    result = quant.install_lingbot_npu_quantization(model, _config(scope=scope))
    assert result["selected_layers"] == count
    assert result["protected_conditioning_and_head"]
    for name, before in protected.items():
        assert torch.equal(dict(model.named_parameters())[name], before)


def test_selectors_allowlist_and_exclusion_only_narrow_protected_modules():
    config = _config(projections=("ffn", "self_qkv", "cross_out"),
                     allowlist=("blocks.*.*",), exclude=("blocks.*.ffn.2",))
    result = quant.install_lingbot_npu_quantization(_transformer(), config)
    assert result["selected_names"] == ["blocks.0.self_attn.qkv", "blocks.0.cross_attn.o", "blocks.0.ffn.0"]
    assert not quant.selected_linear("blocks.0.cross_attn.k", "hot", allowlist=("*",))
    assert not quant.selected_linear("blocks.0.cam_scale_layer", "hot", allowlist=("*",))
    assert not quant.selected_linear("head", "hot", allowlist=("*",))


def test_model_settings_override_environment_selectively_and_validate():
    config = quant.NpuQuantizationConfig.from_sources({
        "lingbot_npu_quant_projections": ["ffn", "self_qkv"],
        "lingbot_npu_quant_exclude": ["blocks.7.*"],
    }, {"LINGBOT_NPU_QUANT": "w8a8", "LINGBOT_NPU_QUANT_SCOPE": "hot"})
    assert config.mode == "w8a8"
    assert config.projections == ("ffn", "self_qkv")
    assert config.exclude == ("blocks.7.*",)
    with pytest.raises(ValueError, match="Projection selectors"):
        _config(projections=("camera",))
    with pytest.raises(ValueError, match="Legacy"):
        _config(scope="all", exclude=("blocks.0.cam*",))
    with pytest.raises(ValueError, match="must be a string"):
        quant.NpuQuantizationConfig.from_sources({"lingbot_npu_quant": True}, {})


@pytest.mark.parametrize("skip_bias_add,return_bias", [(False, False), (False, True), (True, True)])
def test_prequantized_payload_preserves_parallel_bias_contract(skip_bias_add, return_bias):
    layer = ParallelLinear(skip_bias_add=skip_bias_add, return_bias=return_bias).to(dtype=torch.bfloat16)
    quant._install_on_linear(layer, _config())
    x = torch.randn(2, 3, 32, dtype=torch.bfloat16)
    xf = x.float()
    maximum = xf.abs().amax(-1)
    scales = torch.where(maximum > 0, maximum/127, torch.ones_like(maximum))
    payload = torch.round(xf/scales[..., None]).clamp(-127, 127).to(torch.int8)
    actual = quant.quantized_linear_from_payload(layer, payload, scales, output_dtype=x.dtype)
    expected = layer(x)
    if return_bias:
        assert torch.equal(actual[0], expected[0])
        assert actual[1] is expected[1]
    else:
        assert torch.equal(actual, expected)


def test_useful_matrix_ledger_recomputes_precision_without_inflating_work():
    ffn = quant.useful_matrix_ledger(_config(scope="ffn"), served_seconds=1.798182)
    hot = quant.useful_matrix_ledger(_config(scope="hot"), served_seconds=.7, int8_peak_tops=727.4496)
    qkv = quant.useful_matrix_ledger(_config(scope="ffn_qkv"))
    assert ffn["operations_per_rank"] == {"bf16": 178320310272000, "int8": 52999225344000}
    assert hot["operations_per_rank"] == {"bf16": 119432282112000, "int8": 111887253504000}
    assert qkv["operations_per_rank"]["int8"] == ffn["operations_per_rank"]["int8"] + 29444014080000
    assert hot["useful_operations_total_global"] == ffn["useful_operations_total_global"] == 925278142464000
    assert ffn["self_attention_launches_per_rank_per_chunk"] == 160
    assert ffn["bf16_e2e_selected_proxy"] == pytest.approx(.27264293693387)
    assert ffn["whole_model_mfu"] is None
    assert hot["precision_weighted_selected_proxy"] is None
    assert hot["conditional_precision_weighted_selected_proxy"] == pytest.approx(.6888092664092665)
    excluded = quant.useful_matrix_ledger(_config(scope="hot", exclude=("blocks.0.self_attn.qkv",)))
    assert excluded["operations_per_rank"]["int8"] == hot["operations_per_rank"]["int8"] - 736100352000


@pytest.mark.parametrize("mode", ["w8a8", "w8a16"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_standalone_shape_dtype_bias_and_numerical_error(mode, dtype):
    torch.manual_seed(17)
    source = nn.Linear(128, 96).to(dtype=dtype)
    value = torch.randn(2, 3, 128, dtype=dtype)
    reference = F.linear(value.float(), source.weight.float(), source.bias.float()).to(dtype)
    converted = quant.NpuQuantLinear.from_linear(source, mode=mode)
    output = converted(value)
    assert output.shape == (2, 3, 96)
    assert output.dtype == dtype
    metrics = quant.tensor_error(reference, output)
    assert metrics["relative_rmse"] < 0.03
    assert metrics["cosine_similarity"] > 0.999
    # Conversion must not mutate the retained source Linear.
    assert source.weight.dtype == dtype
    assert converted.weight.dtype == torch.int8


def test_weight_zero_channels_and_symmetric_error_bound():
    source = torch.tensor([[0.0, 0.0, 0.0], [-2.0, 0.123, 1.1]])
    integers, scale, stats = quant.quantize_weight_per_channel(source, chunk_rows=1)
    assert integers.dtype == torch.int8
    assert torch.equal(integers[0], torch.zeros(3, dtype=torch.int8))
    assert scale[0] == 1
    error = (source - integers.float() * scale[:, None]).abs()
    assert bool((error <= scale[:, None] / 2 + 1e-6).all())
    assert stats["elements"] == 6
    assert torch.isfinite(scale).all()


def test_nonfinite_weight_rejected():
    with pytest.raises(ValueError, match="NaN or infinity"):
        quant.quantize_weight_per_channel(torch.tensor([[float("nan"), 1.0]]))


@pytest.mark.parametrize("mode", ["w8a8", "w8a16"])
def test_parallel_bias_contract_and_custom_dispatch_bypass(mode):
    model = _transformer()
    layer = model.blocks[0].self_attn.o
    layer.skip_bias_add = True
    layer.return_bias = True
    quant.install_lingbot_npu_quantization(model, _config(mode=mode))
    value = torch.zeros(2, 32)
    output, bias = layer(value)
    assert torch.equal(output, torch.zeros(2, 32))
    assert bias is layer.bias
    assert layer.custom_op is None
    if mode == "w8a16":
        assert layer._lingbot_quant_bias_fp32.dtype == torch.float32
        assert torch.equal(layer._lingbot_quant_bias_fp32, layer.bias.float())
    weight_before = layer.weight.clone()
    layer.quant_method.process_weights_after_loading(layer)
    assert torch.equal(weight_before, layer.weight)
    assert not layer.quant_method.uses_meta_device


def test_fp64_full_output_quality_metrics_and_empty_values():
    # More than one accumulation chunk, with BF16 values spanning a wide
    # dynamic range. Identical tensors must report cosine exactly one.
    value = torch.linspace(-20, 20, (1 << 20) + 17).to(torch.bfloat16)
    exact = quant.tensor_error(value, value.clone())
    assert exact == {"max_abs_error": 0.0, "relative_rmse": 0.0, "cosine_similarity": 1.0}
    perturbed = value.float() + 0.1
    actual = quant.tensor_error(value, perturbed)
    a, b = value.double(), perturbed.double()
    expected_cosine = (a * b).sum() / torch.sqrt(a.square().sum() * b.square().sum())
    expected_rmse = torch.sqrt((a - b).square().sum() / a.square().sum())
    assert actual["cosine_similarity"] == pytest.approx(expected_cosine.item(), abs=1e-13)
    assert actual["relative_rmse"] == pytest.approx(expected_rmse.item(), abs=1e-13)
    assert quant.tensor_error(torch.empty(0), torch.empty(0))["cosine_similarity"] == 1.0


def test_retained_cpu_bf16_restoration_is_exact_and_reinstallable():
    model = _transformer()
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    original_requires_grad = {name: p.requires_grad for name, p in model.named_parameters()}
    old_method = model.blocks[0].self_attn.o.quant_method
    old_custom = model.blocks[0].self_attn.o.custom_op
    config = _config(keep_bf16_cpu=True, scope="all")
    first = quant.install_lingbot_npu_quantization(model, config)
    assert quant.install_lingbot_npu_quantization(model, config) is first
    quant.restore_lingbot_bf16(model)
    assert model.blocks[0].self_attn.o.quant_method is old_method
    assert model.blocks[0].self_attn.o.custom_op is old_custom
    for name, p in model.named_parameters():
        assert torch.equal(before[name], p)
        assert p.requires_grad == original_requires_grad[name]
    assert not hasattr(model, "_lingbot_npu_quantization_config")
    assert quant.install_lingbot_npu_quantization(model, config)["selected_layers"] == 12


def test_restore_without_originals_requires_checkpoint_reload():
    model = _transformer()
    quant.install_lingbot_npu_quantization(model, _config())
    with pytest.raises(ValueError, match="reload the unchanged original checkpoint"):
        quant.restore_lingbot_bf16(model)


def test_tp_guard_precedes_mutation():
    model = _transformer()
    model.blocks[0].ffn[2].tp_size = 2
    with pytest.raises(ValueError, match="restricted to the existing FFN scope"):
        quant.install_lingbot_npu_quantization(model, _config())
    assert model.blocks[0].self_attn.qkv.weight.is_floating_point()


def test_conflicting_quant_method_rejected_before_mutation():
    model = _transformer()
    model.blocks[0].ffn[2].quant_method = object()
    with pytest.raises(ValueError, match="already uses quantization"):
        quant.install_lingbot_npu_quantization(model, _config())
    assert model.blocks[0].self_attn.qkv.weight.is_floating_point()


def test_vector_empty_input_and_config_validation():
    source = nn.Linear(32, 16)
    converted = quant.NpuQuantLinear.from_linear(source)
    assert converted(torch.zeros(32)).shape == (16,)
    assert torch.equal(converted(torch.zeros(32)), source.bias)
    assert converted(torch.zeros(0, 2, 32)).shape == (0, 2, 16)
    with pytest.raises(ValueError, match="must be off"):
        quant.NpuQuantizationConfig.from_env({"LINGBOT_NPU_QUANT": "fp8"})
    with pytest.raises(ValueError, match="transposed ND"):
        _config("w8a16", weight_format="nz")


def test_tp_shared_weight_scales_preserve_full_matrix_payload():
    weight = torch.tensor([[0.01, -0.03, 9.0, -7.0], [2.0, -1.0, 0.0, 0.0]])
    full, scale, _ = quant.quantize_weight_per_channel(weight)
    maximum = weight.abs().amax(dim=1)
    shards = [quant.quantize_weight_per_channel(part, shared_maximum=maximum)
              for part in weight.chunk(2, dim=1)]
    assert torch.equal(torch.cat([part[0] for part in shards], dim=1), full)
    assert all(torch.equal(part[1], scale) for part in shards)


def test_tp_row_reduces_integers_before_dequant_and_single_bias(monkeypatch):
    from types import SimpleNamespace
    import torch.distributed as dist

    activations = torch.tensor([[0.03, -0.1, 4.0, -2.0], [1.0, -1.0, 0.0, 0.0]])
    weights = torch.tensor([[0.1, -0.2, 3.0, -2.0], [2.0, 1.0, 0.5, -1.0]])
    qweight, weight_scale, _ = quant.quantize_weight_per_channel(weights)
    maximum = activations.abs().amax(dim=1)
    scale = maximum / 127.0
    integers = torch.round(activations / scale[:, None]).clamp(-127, 127).to(torch.int32)
    other = integers[:, 2:] @ qweight[:, 2:].to(torch.int32).T
    bias = torch.tensor([0.5, -0.25])
    expected = ((integers @ qweight.to(torch.int32).T).float() * scale[:, None] * weight_scale + bias).to(torch.bfloat16)
    layer = SimpleNamespace(weight=qweight[:, :2].T.contiguous(), bias=bias,
                            _lingbot_quant_weight_layout="kn", _lingbot_quant_weight_scale=weight_scale,
                            _lingbot_quant_out_features=2, skip_bias_add=False)

    def reduce(value, *, op, group):
        if op == dist.ReduceOp.MAX:
            value.copy_(maximum)
        else:
            assert op == dist.ReduceOp.SUM and value.dtype == torch.int32
            value.add_(other)

    monkeypatch.setattr(quant, "_tp_group", lambda: None)
    monkeypatch.setattr(dist, "all_reduce", reduce)
    actual = quant._tp_row_quantized_linear(layer, activations[:, :2], (2, 2), torch.bfloat16)
    assert torch.equal(actual, expected)
