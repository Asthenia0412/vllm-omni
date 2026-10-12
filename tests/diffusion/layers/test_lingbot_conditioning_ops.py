"""CPU proof of repeated-cast reuse, with no distributed package imports."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

path = Path(__file__).resolve().parents[3]/"vllm_omni/diffusion/layers/lingbot_conditioning_ops.py"
spec = importlib.util.spec_from_file_location("conditioning_ops_cpu", path)
ops = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ops)


def test_default_does_not_convert_or_change_identity():
    module=SimpleNamespace()
    summary=ops.configure_timestep_fp32_once(module, environ={})
    value=torch.randn(1,3,6,5120).bfloat16()
    assert not summary["timestep_fp32_once"]
    assert ops.prepare_timestep_projection(module,value) is value


def test_every_bf16_bit_pattern_cast_and_affine_reuse_is_identical():
    module=SimpleNamespace(_lingbot_timestep_fp32_once=True)
    # All encodings include signed zeros, normals, subnormals, infinities and
    # NaN payloads. Compare bits, since torch.equal treats NaNs as unequal.
    value=torch.arange(65536,dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
    original=value.clone()
    converted=ops.prepare_timestep_projection(module,value)
    for index in range(40):
        assert torch.equal(value.float().view(torch.int32),converted.view(torch.int32))
        modulation=torch.tensor(index/40.,dtype=torch.float32)
        assert torch.equal((modulation+value.float()).view(torch.int32),
                           (modulation+converted.float()).view(torch.int32))
    assert torch.equal(value.view(torch.int16),original.view(torch.int16))


def test_fp32_projection_remains_same_tensor():
    value=torch.randn(1,3,6,16)
    assert ops.prepare_timestep_projection(SimpleNamespace(_lingbot_timestep_fp32_once=True),value) is value


@pytest.mark.parametrize("override",[True,False])
def test_model_setting_overrides_environment(override):
    module=SimpleNamespace()
    ops.configure_timestep_fp32_once(module,{"lingbot_npu_timestep_fp32_once":override},
                                    environ={"LINGBOT_NPU_TIMESTEP_FP32_ONCE":"1" if not override else "0"})
    assert module._lingbot_timestep_fp32_once is override


@pytest.mark.parametrize("value",["yes",None,"",2,1.0,0.0])
def test_unknown_flag_rejected(value):
    with pytest.raises(ValueError):
        ops.configure_timestep_fp32_once(SimpleNamespace(),environ={"LINGBOT_NPU_TIMESTEP_FP32_ONCE":value})


def test_actual_capture_is_separately_gated_and_keeps_original_bf16(monkeypatch, tmp_path):
    import sys
    import types
    package = "conditioning_capture_cpu"
    monkeypatch.setattr(ops, "__package__", package)
    monkeypatch.setitem(sys.modules, package, types.ModuleType(package))
    writer = SimpleNamespace(chunk=1, probe=2, session_id="first", directory=tmp_path, activation_records=[])
    capture = types.ModuleType(package + ".lingbot_capture")
    capture._active_capture = writer
    monkeypatch.setitem(sys.modules, capture.__name__, capture)
    monkeypatch.setattr(ops, "_projection_capture_session", None)
    monkeypatch.setattr(ops, "_projection_captures", set())
    first = SimpleNamespace(modulation=torch.randn(1,6,16))
    transformer = SimpleNamespace(_lingbot_timestep_fp32_once=True, blocks=[first])
    original = torch.randn(1,3,6,16).bfloat16()
    monkeypatch.setenv("LINGBOT_CAPTURE_TIMESTEP", "0")
    monkeypatch.setenv("LINGBOT_CAPTURE_ACTIVATIONS", "0")
    ops.prepare_timestep_projection(transformer, original)
    assert not writer.activation_records
    monkeypatch.setenv("LINGBOT_CAPTURE_TIMESTEP", "1")
    result = ops.prepare_timestep_projection(transformer, original)
    assert result.dtype == torch.float32
    saved = torch.load(tmp_path/writer.activation_records[0]["path"], weights_only=True)
    assert saved["tensors"]["projection"].dtype == torch.bfloat16
    assert torch.equal(saved["tensors"]["projection"], original)
    assert torch.equal(saved["tensors"]["modulation"], first.modulation)
    ops.prepare_timestep_projection(transformer, original)
    assert len(writer.activation_records) == 1
    writer.chunk = 2
    ops.prepare_timestep_projection(transformer, original)
    assert len(writer.activation_records) == 1
    writer.chunk, writer.probe, writer.session_id = 1, 3, "another"
    ops.prepare_timestep_projection(transformer, original)
    assert len(writer.activation_records) == 1
