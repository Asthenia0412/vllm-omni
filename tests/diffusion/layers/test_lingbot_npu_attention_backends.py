# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU-only contracts for dense BF16 candidates, with every NPU op replaced."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

# Loading this leaf directly avoids importing platform discovery or serving
# dependencies. Actual operator bindings are tested separately on granted NPUs.
_SOURCE = Path(__file__).resolve().parents[3] / "vllm_omni/diffusion/layers/lingbot_npu_attention.py"
_SPEC = importlib.util.spec_from_file_location("lingbot_attention_contracts", _SOURCE)
attention = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(attention)


@pytest.fixture(autouse=True)
def clean_state(monkeypatch):
    attention.reset_attention_state()
    monkeypatch.delenv("VLLM_LINGBOT_NPU_ATTENTION_BACKEND", raising=False)
    monkeypatch.delenv("VLLM_LINGBOT_NPU_ATTENTION_STRICT", raising=False)
    monkeypatch.delenv("VLLM_LINGBOT_NPU_ATTENTION_SPLITS", raising=False)
    monkeypatch.setenv("VLLM_LINGBOT_NPU_INFER_ATTENTION", "1")
    yield
    attention.reset_attention_state()


def inputs(sk=7, batch=1):
    generator = torch.Generator().manual_seed(42)
    return tuple(
        torch.randn(shape, dtype=torch.bfloat16, generator=generator)
        for shape in ((batch, 3, 2, 16), (batch, sk, 2, 16), (batch, sk, 2, 16))
    )


def test_static_rejection_blocks_community_library_import(monkeypatch, tmp_path):
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git/HEAD").write_text(attention.FLASH_PIN + "\n")
    (tmp_path / "lingbot_variant.json").write_text(json.dumps({
        "status": "rejected_static_resource_violation", "launch_forbidden": True,
    }))
    monkeypatch.setenv("VLLM_LINGBOT_NPU_FLASH_SOURCE", str(tmp_path))

    def must_not_import(name):
        raise AssertionError(f"unsafe extension import attempted: {name}")

    monkeypatch.setattr(attention.importlib, "import_module", must_not_import)
    with pytest.raises(ImportError, match="static audit"):
        attention._community_module("v4")


@pytest.mark.parametrize("backend", ["fia_bsnd", "fia_tnd", "fia_v2_bsnd", "fia_v2_tnd"])
@pytest.mark.parametrize("batch,sk", [(1, 7), (2, 19)])
def test_native_layout_and_exact_cumulative_lengths(monkeypatch, backend, batch, sk):
    calls = []

    def operator(q, k, v, **kwargs):
        calls.append((q, k, v, kwargs))
        return q.clone(), None

    monkeypatch.setitem(
        sys.modules,
        "torch_npu",
        SimpleNamespace(npu_fused_infer_attention_score=operator, npu_fused_infer_attention_score_v2=operator),
    )
    q, k, v = inputs(sk, batch)
    result = attention.run_attention(q, k, v, head_num=2, scale=0.25, backend=backend, strict=True)
    qs, ks, vs, kw = calls[0]
    assert result.shape == q.shape and result.dtype == torch.bfloat16
    assert qs.data_ptr() == q.data_ptr() and ks.data_ptr() == k.data_ptr()
    assert vs.data_ptr() == v.data_ptr()  # TND is a view, with no conversion copy.
    assert kw["sparse_mode"] == kw["inner_precise"] == 0
    assert kw["input_layout"] == ("TND" if backend.endswith("tnd") else "BSND")
    if backend.endswith("tnd"):
        qname, kname = (
            ("actual_seq_qlen", "actual_seq_kvlen")
            if "v2" in backend
            else ("actual_seq_lengths", "actual_seq_lengths_kv")
        )
        assert kw[qname] == [3 * (i + 1) for i in range(batch)]
        assert kw[kname] == [sk * (i + 1) for i in range(batch)]
        assert qs.shape == (batch * 3, 2, 16) and ks.shape == (batch * sk, 2, 16)


@pytest.mark.parametrize("backend", ["fia_bnsd", "fia_v2_bnsd"])
def test_bnsd_full_call_conversion_is_exact_and_not_cached(monkeypatch, backend):
    calls = []

    def operator(q, k, v, **kwargs):
        assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
        assert kwargs["input_layout"] == "BNSD"
        calls.append((q, k, v))
        return q.clone(), None

    monkeypatch.setitem(
        sys.modules, "torch_npu",
        SimpleNamespace(npu_fused_infer_attention_score=operator, npu_fused_infer_attention_score_v2=operator),
    )
    q, k, v = inputs(19, batch=2)
    for offset in (0, 1):
        result = attention.run_attention(q + offset, k + offset, v + offset, head_num=2, scale=0.25,
                                         backend=backend, strict=True)
        assert result.is_contiguous()
        assert torch.equal(result, q + offset)
        assert torch.equal(calls[-1][1], (k + offset).transpose(1, 2))
        assert torch.equal(calls[-1][2], (v + offset).transpose(1, 2))
    assert calls[0][0].data_ptr() != calls[1][0].data_ptr()
    assert torch.equal(calls[0][1], k.transpose(1, 2))


@pytest.mark.parametrize("backend", ["flash_v2", "flash_v3", "flash_v4"])
def test_community_metadata_reused_for_four_probes_and_length_scale_changes(monkeypatch, backend):
    metadata_calls, forward_calls = [], []

    def get_metadata(**kwargs):
        metadata = object()
        metadata_calls.append((kwargs, metadata))
        return metadata

    def forward(q, k, v, **kwargs):
        forward_calls.append((q, k, v, kwargs))
        return (q.clone(), None) if backend == "flash_v4" else q.clone()

    module = SimpleNamespace(
        get_scheduler_metadata=get_metadata, flash_attn_with_kvcache=forward, _flash_attn_forward=forward
    )
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace())
    monkeypatch.setattr(attention, "_community_module", lambda version: module)

    def forbidden(*args, **kwargs):
        raise AssertionError("per-call length D2H must not occur")

    for name in ("cpu", "item", "tolist"):
        monkeypatch.setattr(torch.Tensor, name, forbidden)
    q, k, v = inputs()
    original_k, original_v = k.clone(), v.clone()
    for _ in range(4):
        attention.run_attention(q, k, v, head_num=2, scale=0.25, backend=backend, strict=True)
    assert len(metadata_calls) == 1
    assert attention.attention_state()["metadata_hits"] == 3
    for _, key, value, kwargs in forward_calls:
        assert key is k and value is v
        assert kwargs["scheduler_metadata"] is metadata_calls[0][1]
        assert kwargs["causal"] is False and kwargs["softcap"] == 0
        assert kwargs["num_splits"] == 1
        assert "k_new" not in kwargs and "v_new" not in kwargs and "page_table" not in kwargs
    assert torch.equal(original_k, k) and torch.equal(original_v, v)
    lengths = metadata_calls[0][0]["cache_seqlens"]
    assert torch.equal(lengths, torch.tensor([7], dtype=torch.int32))
    attention.run_attention(q, k, v, head_num=2, scale=0.5, backend=backend, strict=True)
    attention.run_attention(*inputs(19), head_num=2, scale=0.25, backend=backend, strict=True)
    assert len(metadata_calls) == 3  # Scale and visible history are distinct tilings.


def test_failure_is_shape_local_and_strict_screen_cannot_time_fallback(monkeypatch):
    calls = []

    def infer(q, k, v, **kwargs):
        calls.append(("infer", k.shape[1]))
        raise NotImplementedError("unsupported candidate")

    def fusion(q, k, v, **kwargs):
        calls.append(("fusion", k.shape[1]))
        return q, None

    monkeypatch.setitem(
        sys.modules, "torch_npu", SimpleNamespace(npu_fused_infer_attention_score=infer, npu_fusion_attention=fusion)
    )
    for sk in (7, 7, 19):
        attention.run_attention(*inputs(sk), head_num=2, scale=0.25)
    assert calls == [("infer", 7), ("fusion", 7), ("fusion", 7), ("infer", 19), ("fusion", 19)]
    with pytest.raises(NotImplementedError):
        attention.run_attention(*inputs(), head_num=2, scale=0.25, backend="fia_bsnd", strict=True)
    assert calls[-1] == ("infer", 7)


@pytest.mark.parametrize("backend", ["flash_v2", "flash_v3", "flash_v4"])
def test_dense_split_constraints_rejected_before_loading_npu(monkeypatch, backend):
    monkeypatch.delitem(sys.modules, "torch_npu", raising=False)
    with pytest.raises(ValueError, match="num_splits"):
        attention.run_attention(*inputs(), head_num=2, scale=0.25, backend=backend, num_splits=2)
    assert "torch_npu" not in sys.modules


def test_invalid_dtype_scale_and_backend_fail_early():
    q, k, v = inputs()
    with pytest.raises(ValueError, match="BF16"):
        attention.run_attention(q.float(), k.float(), v.float(), head_num=2, scale=0.25, backend="flash_v3")
    with pytest.raises(ValueError, match="finite"):
        attention.run_attention(q, k, v, head_num=2, scale=float("nan"))
    with pytest.raises(ValueError, match="unknown"):
        attention.run_attention(q, k, v, head_num=2, scale=0.25, backend="linear_attention")


def test_native_default_preserves_gate_off(monkeypatch):
    calls = []
    monkeypatch.setenv("VLLM_LINGBOT_NPU_INFER_ATTENTION", "0")
    monkeypatch.setitem(
        sys.modules,
        "torch_npu",
        SimpleNamespace(npu_fusion_attention=lambda q, k, v, **kw: (calls.append(kw) or q, None)),
    )
    attention.run_attention(*inputs(), head_num=2, scale=0.25)
    assert calls[0]["keep_prob"] == 1
    assert calls[0]["pre_tockens"] == calls[0]["next_tockens"] == 2147483647


def test_model_policy_overrides_env_without_cross_instance_leak(monkeypatch):
    monkeypatch.setenv("VLLM_LINGBOT_NPU_ATTENTION_BACKEND", "fusion")
    monkeypatch.setenv("VLLM_LINGBOT_NPU_ATTENTION_SPLITS", "0")
    monkeypatch.setenv("VLLM_LINGBOT_NPU_ATTENTION_STRICT", "1")
    first = SimpleNamespace(blocks=[SimpleNamespace(self_attn=SimpleNamespace())])
    second = SimpleNamespace(blocks=[SimpleNamespace(self_attn=SimpleNamespace())])
    result = attention.configure_lingbot_attention(
        first,
        {
            "lingbot_npu_attention_backend": "fia_bsnd",
            "lingbot_npu_attention_splits": 1,
            "lingbot_npu_attention_strict": False,
        },
    )
    attention.configure_lingbot_attention(second, {})
    assert result == {"configured_layers": 1, "backend": "fia_bsnd", "num_splits": 1, "strict": False}
    calls = []
    monkeypatch.setitem(
        sys.modules,
        "torch_npu",
        SimpleNamespace(
            npu_fused_infer_attention_score=lambda q, k, v, **kw: (calls.append("infer") or q, None),
            npu_fusion_attention=lambda q, k, v, **kw: (calls.append("fusion") or q, None),
        ),
    )
    for model in (first, second):
        layer = model.blocks[0].self_attn
        attention.run_attention(
            *inputs(),
            head_num=2,
            scale=0.25,
            backend=layer._lingbot_npu_attention_backend,
            num_splits=layer._lingbot_npu_attention_splits,
            strict=layer._lingbot_npu_attention_strict,
        )
    assert calls == ["infer", "fusion"]
    assert first.blocks[0].self_attn._lingbot_npu_attention_backend == "fia_bsnd"
    assert second.blocks[0].self_attn._lingbot_npu_attention_backend is None
    assert attention.os.environ["VLLM_LINGBOT_NPU_ATTENTION_BACKEND"] == "fusion"
    assert attention.os.environ["VLLM_LINGBOT_NPU_ATTENTION_SPLITS"] == "0"
    assert attention.os.environ["VLLM_LINGBOT_NPU_ATTENTION_STRICT"] == "1"


@pytest.mark.parametrize(
    "key,value",
    [
        ("lingbot_npu_attention_backend", "mla"),
        ("lingbot_npu_attention_splits", 2),
        ("lingbot_npu_attention_splits", True),
        ("lingbot_npu_attention_strict", "false"),
    ],
)
def test_invalid_model_policy_rejected_before_any_instance_mutation(key, value):
    layer = SimpleNamespace(_lingbot_npu_attention_backend="fia_bsnd")
    model = SimpleNamespace(blocks=[SimpleNamespace(self_attn=layer)])
    with pytest.raises(ValueError):
        attention.configure_lingbot_attention(model, {key: value})
    assert layer._lingbot_npu_attention_backend == "fia_bsnd"


def test_allocator_failure_is_not_hidden_by_fallback(monkeypatch):
    def infer(*args, **kwargs):
        raise RuntimeError("NPU out of memory")

    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace(npu_fused_infer_attention_score=infer))
    with pytest.raises(RuntimeError, match="out of memory"):
        attention.run_attention(*inputs(), head_num=2, scale=0.25)


def test_screen_numeric_reductions_detect_invalid_output_and_use_fp64():
    path = _SOURCE.parents[4] / "profiling/screen_lingbot_attention_backends.py"
    spec = importlib.util.spec_from_file_location("lingbot_attention_screen_contracts", path)
    screen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(screen)
    reference = torch.tensor([1.0, 2.0, 3.0, -4.0], dtype=torch.bfloat16)
    same = screen.numeric_errors(torch, reference, reference)
    assert same["finite"] and same["rmse"] == same["max_absolute_error"] == 0
    assert same["cosine_fp64"] == pytest.approx(1, abs=1e-15)
    changed = screen.numeric_errors(torch, reference, reference + 1)
    assert changed["rmse"] == 1 and changed["max_absolute_error"] == 1
    invalid = screen.numeric_errors(torch, reference, torch.full_like(reference, float("nan")))
    assert invalid["finite"] is False and invalid["rmse"] is None


def test_bounded_cpu_fp32_reference_matches_full_dense_attention():
    path = _SOURCE.parents[4] / "profiling/prepare_lingbot_attention_references.py"
    spec = importlib.util.spec_from_file_location("lingbot_attention_fp32_reference_contracts", path)
    reference_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference_module)
    q, k, v = inputs(19)
    result = reference_module.fp32_reference(torch, q, k, v, 0.25, query_block=2)
    qs, ks, vs = (tensor.transpose(1, 2).float() for tensor in (q, k, v))
    expected = (torch.softmax((qs @ ks.transpose(-1, -2)) * 0.25, dim=-1) @ vs).transpose(1, 2)
    assert result.dtype == torch.float32 and result.shape == q.shape
    torch.testing.assert_close(result, expected, atol=1e-7, rtol=1e-6)


def test_kernel_summary_requires_one_attention_kernel_per_probe(tmp_path, monkeypatch):
    import csv
    import re

    profiling = _SOURCE.parents[4] / "profiling"
    monkeypatch.syspath_prepend(str(profiling))
    spec = importlib.util.spec_from_file_location(
        "lingbot_attention_profile_contracts", profiling / "summarize_lingbot_attention_screen.py"
    )
    summary = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(summary)
    path = tmp_path / "kernel_details.csv"
    with path.open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Name", "Duration(us)", "Type", "aic_mac_ratio"])
        writer.writeheader()
        writer.writerows(
            [
                dict(Name="FAInfer", **{"Duration(us)": 100, "Type": "AI_CORE", "aic_mac_ratio": 0.5}),
                dict(Name="FAInfer", **{"Duration(us)": 200, "Type": "AI_CORE", "aic_mac_ratio": 0.8}),
                dict(Name="Fill", **{"Duration(us)": 10, "Type": "AI_VECTOR", "aic_mac_ratio": ""}),
            ]
        )
    result = summary.summarize_case(tmp_path, 2, 1_000_000, 100, re.compile("FAInfer"))
    assert result["mean_attention_kernel_us"] == 150
    assert result["all_kernel_duration_sum_us"] == 310
    assert result["duration_weighted_raw_PMU_fields"]["aic_mac_ratio"] == pytest.approx(0.7)
    inconsistent = summary.summarize_case(tmp_path, 3, 1_000_000, 100, re.compile("FAInfer"))
    assert inconsistent["one_attention_kernel_per_probe_verified"] is False
    assert inconsistent["modeled_internal_math_utilization_percent"] is None


def test_cli_actual_four_probe_schedule_and_missing_capture_rejected(monkeypatch, tmp_path):
    import csv
    from collections import Counter

    profiling = _SOURCE.parents[4] / "profiling"
    monkeypatch.syspath_prepend(str(profiling))
    spec = importlib.util.spec_from_file_location(
        "lingbot_attention_op_contracts", profiling / "profile_lingbot_attention_op.py"
    )
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    schedule = helper.launch_schedule()
    assert len(schedule["warmup_probes"]) == 80
    assert Counter(schedule["marked_probes"][schedule["skip_launches"] :]) == {0: 8, 1: 8, 2: 8, 3: 8}
    with pytest.raises(ValueError, match="whole-group"):
        helper.launch_schedule(skip_launches=21)
    fields = ["Device Id", "Task Duration(us)", "Op Name"]
    for device in (0, 2, 4, 6):
        directory = tmp_path / str(device)
        directory.mkdir()
        with (directory / "OpBasicInfo_1.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerow({"Device Id": device, "Task Duration(us)": 4200, "Op Name": "FAInfer"})
    assert helper.validate_capture(tmp_path, [0, 2, 4, 6], 1, 28080)["valid"]
    assert not helper.validate_capture(tmp_path, [0, 2, 4, 6], 1, 28080, "PipeUtilization")["valid"]
    for device in (0, 2, 4, 6):
        (tmp_path / str(device) / "PipeUtilization_1.csv").write_text("sub_block_id,aic_cube_ratio\ncube0,0.5\n")
    assert helper.validate_capture(tmp_path, [0, 2, 4, 6], 1, 28080, "PipeUtilization")["valid"]
    assert not helper.validate_capture(tmp_path, [0, 2, 4, 6], 32, 28080)["valid"]
    (tmp_path / "6/OpBasicInfo_1.csv").unlink()
    assert not helper.validate_capture(tmp_path, [0, 2, 4, 6], 1, 28080)["valid"]


def test_cli_counter_summary_adds_distinct_request_bytes_and_rejects_duplicate_cores(tmp_path):
    import csv

    path = _SOURCE.parents[4] / "profiling/summarize_lingbot_attention_op.py"
    spec = importlib.util.spec_from_file_location("lingbot_attention_op_summary_contracts", path)
    summary = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(summary)
    basic = tmp_path / "OpBasicInfo_1.csv"
    basic.write_text("Device Id,Task Duration(us),Current Freq,Rated Freq\n0,1000,1850,1850\n")
    fields = ["block_id", "sub_block_id", *summary.TRAFFIC_NAMES]
    rows = []
    for block in range(24):
        for kind in ("cube0", "vector0", "vector1"):
            row = dict.fromkeys(fields, "NA")
            row.update(block_id=block, sub_block_id=kind)
            row["read_main_memory_datas(KB)"] = block + 1
            row["write_main_memory_datas(KB)"] = 2
            rows.append(row)
    metric = tmp_path / "Memory_1.csv"

    def write_metric():
        with metric.open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    write_metric()
    result = summary.parse_launch(basic, "Memory")
    assert result["core_side_read_bytes"] == 3 * sum(range(1, 25)) * 1024
    assert result["core_side_read_logical_gbps"] == pytest.approx(0.9216)
    assert result["core_side_write_bytes"] == 72 * 2 * 1024
    rows[-1]["block_id"] = rows[-4]["block_id"]
    write_metric()
    with pytest.raises(ValueError, match="unique 24 Cube/48 Vector"):
        summary.parse_launch(basic, "Memory")


def test_page1560_screen_order_current_writes_and_metadata_capacity(monkeypatch):
    profiling = _SOURCE.parents[4] / "profiling"
    monkeypatch.syspath_prepend(str(profiling))
    spec = importlib.util.spec_from_file_location("lingbot_paged_screen_contracts", profiling / "screen_lingbot_paged_attention.py")
    screen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(screen)
    q, k, v = inputs(9)
    records = [dict(query=q + probe, key=torch.cat((k[:, :6], k[:, 6:] + probe), 1),
                    value=torch.cat((v[:, :6], v[:, 6:] + probe), 1)) for probe in range(4)]
    case = screen.PagedCase(torch, records, "cpu", page_size=3, table_capacity=19)
    assert case.ids_cpu != list(range(3))
    for probe, record in enumerate(records):
        case.write_current(probe)
        gathered = case.gather()
        assert torch.equal(gathered[0], record["key"])
        assert torch.equal(gathered[1], record["value"])
    metadata_calls, forwards = [], []
    module = SimpleNamespace(
        get_scheduler_metadata=lambda **kw: metadata_calls.append(kw) or object(),
        flash_attn_with_kvcache=lambda q, k, v, **kw: forwards.append(kw) or q,
    )
    leaf = SimpleNamespace(_community_module=lambda version: module)

    def forbidden(*unused_args, **unused_kwargs):
        raise AssertionError("length metadata must not perform per-call D2H")

    for name in ("cpu", "item", "tolist"):
        monkeypatch.setattr(torch.Tensor, name, forbidden)
    for probe in range(4):
        case.paged_v3(leaf, probe)
    assert len(metadata_calls) == case.metadata_builds == 1
    assert metadata_calls[0]["max_seqlen_k"] == 3 * 19
    assert metadata_calls[0]["page_size"] == 3
    assert torch.equal(metadata_calls[0]["cache_seqlens"], torch.tensor([9], dtype=torch.int32))
    assert all(call["page_table"] is case.table for call in forwards)
    assert all(call["scheduler_metadata"] is case.metadata for call in forwards)
    assert all("k_new" not in call and "v_new" not in call for call in forwards)
