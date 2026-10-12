# SPDX-License-Identifier: Apache-2.0
"""Opt-in actual tensor compute ledger; no timing or full-model MFU claim.

TorchDispatchMode observes real operator operands after dtype/layout selection.
It records logical tensor arithmetic, never profiler padding or hardware PMU
operation counters. Streamed/halo convolution outputs are an executed estimate;
their serial useful numerator remains unknown until crop geometry is attributed.
"""
from __future__ import annotations

import json
from contextlib import contextmanager
import math
import os
from pathlib import Path
from typing import Any

import torch
from torch.utils._python_dispatch import TorchDispatchMode

_active_compute_recorder = None


def _descriptor(value: torch.Tensor) -> dict:
    return {"shape": list(value.shape), "dtype": str(value.dtype), "device": str(value.device)}


def _tensor(value):
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        return next((item for item in value if isinstance(item, torch.Tensor)), None)
    return None


def _parameters(func, args, kwargs):
    values = {argument.name: argument.default_value for argument in func._schema.arguments}
    values.update({argument.name: value for argument, value in zip(func._schema.arguments, args)})
    values.update(kwargs)
    return values


class ComputeRecorder(TorchDispatchMode):
    def __init__(self, *, rank: int = 0, logical_ranks: int = 4, max_records: int = 30000):
        super().__init__()
        self.rank, self.logical_ranks, self.max_records = rank, logical_ranks, max_records
        self.phase, self.chunk, self.probe = "unknown", -1, None
        self.module_stack = []
        self.records = []
        self.truncated = False
        self.hooks = []
        self._pending_explicit_attention = None
        self._vae_attention_scopes = {}
        self.weighted_module_calls = []
        self.unrecognized_weighted_operator_names = {}
        self._recording_enabled = False
        self._module_names = {}
        self._pending_attention_matrix_stages = set()

    def __enter__(self):
        result = super().__enter__()
        self._recording_enabled = True
        return result

    def __exit__(self, exc_type, exc_value, traceback):
        self._recording_enabled = False
        return super().__exit__(exc_type, exc_value, traceback)

    def set_phase(self, phase: str, chunk: int, probe: int | None = None):
        self.phase, self.chunk, self.probe = str(phase), int(chunk), probe

    def attach_modules(self, model):
        for name, module in model.named_modules():
            self._module_names[id(module)] = name
            if module.__class__.__name__ == "WanAttentionBlock" and "decoder" in name:
                self._vae_attention_scopes[name] = bool(getattr(module, "_vllm_omni_spatial_shard_attention", False))
            weight = getattr(module, "weight", None)
            weighted_leaf = (isinstance(weight, torch.Tensor) and weight.ndim in (2, 4, 5)
                             and not any(module.children()))
            frames = []
            def before(mod, arguments, name=name, weighted_leaf=weighted_leaf, frames=frames):
                if not self._recording_enabled:
                    return
                self.module_stack.append(name)
                if weighted_leaf:
                    frames.append((len(self.records), _tensor(arguments)))
            def after(mod, arguments, result, name=name, weighted_leaf=weighted_leaf, frames=frames):
                if not self._recording_enabled:
                    return
                if weighted_leaf and frames:
                    start, value = frames.pop()
                    self.record_weighted_leaf(mod, name, value, _tensor(result), start)
                if self.module_stack and self.module_stack[-1] == name:
                    self.module_stack.pop()
                else:
                    self.module_stack.clear()
            self.hooks.extend((module.register_forward_pre_hook(before),
                               module.register_forward_hook(after, always_call=True)))

    def record_weighted_leaf(self, module, scope, value, output, start):
        """Fill opaque private linear/conv boundaries from REAL module operands.

        Native dispatch records own an observed call whenever they are visible.
        A fallback is emitted only for an unobserved weighted leaf with a known
        linear or direct-convolution shape contract. This is diagnostic tensor
        arithmetic, never a fabricated hardware operator or timing record.
        """
        existing = [row for row in self.records[start:] if row["module"] == scope
                    and row["kind"] in {"matrix", "convolution"}]
        call = {"module": scope, "module_class": type(module).__name__, "phase": self.phase,
                "chunk": self.chunk, "probe": self.probe, "dispatch_arithmetic_records": len(existing),
                "observation": "dispatch" if existing else "unresolved"}
        self.weighted_module_calls.append(call)
        if existing:
            if (type(module).__name__ in {"WanDistCausalConv3d", "WanDistConv2d"}
                    and output is not None and isinstance(getattr(module, "weight", None), torch.Tensor)):
                weight = module.weight
                retained = 2*output.numel()*weight.shape[1]*math.prod(weight.shape[2:])
                for row in existing:
                    row.update(operations_local_useful_dense_estimate=retained, retained_output=_descriptor(output),
                        pre_trim_executed_operations=row["operations_executed_tensor_estimate"],
                        operations_estimate_scope="visible pre-trim dispatch arithmetic with separately audited retained interior output")
            return  # One arithmetic call is counted once across both observers.
        weight = getattr(module, "weight", None)
        if value is None or output is None or not isinstance(weight, torch.Tensor):
            call["reason"] = "no completed tensor input/output"
            return
        call.update(input=_descriptor(value), output=_descriptor(output), weight=_descriptor(weight))
        if not (value.is_floating_point() or value.dtype == torch.int8):
            call["reason"] = "non arithmetic input (for example embedding indices)"
            return
        row = None
        if weight.ndim == 2:
            layout = getattr(module, "_lingbot_quant_weight_layout", "nk")
            n, k = (weight.shape[1], weight.shape[0]) if layout == "kn" else weight.shape
            patchifier = type(module).__name__ == "_LingBotCameraPatchEmbedding"
            if output.ndim and output.shape[-1] == n and (
                value.shape[-1] == k or (patchifier and getattr(module, "in_features", None) == k)
            ):
                mode = getattr(module, "_lingbot_quant_mode", None)
                precision = "int8" if mode == "w8a8" else (
                    "weight_only_mixed_unverified" if mode else str(value.dtype).removeprefix("torch.")
                )
                row = {"kind": "matrix", "operations_executed_tensor_estimate": 2*k*output.numel(),
                       "precision": precision, "inputs": [_descriptor(value), _descriptor(weight)],
                       "output": _descriptor(output), "weight_layout": layout,
                       "logical_matrix_input_shape": [output.numel()//n, k],
                       "precision_basis": "installed quantization contract" if mode else "actual module input dtype",
                       "input_reformatted_by_module": patchifier}
        elif weight.ndim in (4, 5):
            # Known direct convolution leaves. Arbitrary cropped, transposed,
            # or composite wrappers need a separate audited source contract.
            direct = type(module).__name__ in {"Conv2d", "Conv3d", "Conv2dLayer", "Conv3dLayer",
                "WanCausalConv3d", "WanDistCausalConv3d", "WanDistConv2d"}
            if direct and output.ndim == weight.ndim and output.shape[1] == weight.shape[0]:
                row = {"kind": "convolution", "operations_executed_tensor_estimate":
                       2*output.numel()*weight.shape[1]*math.prod(weight.shape[2:]),
                       "precision": str(value.dtype).removeprefix("torch."),
                       "inputs": [_descriptor(value), _descriptor(weight)], "output": _descriptor(output),
                       "groups": int(getattr(module, "groups", 1)), "transposed": False,
                       "stride": getattr(module, "stride", None), "padding": getattr(module, "padding", None),
                       "dilation": getattr(module, "dilation", None),
                       "useful_numerator_note": "Direct weighted leaf output; SP/halo/cache scope still requires attribution"}
                if type(module).__name__ in {"WanDistCausalConv3d", "WanDistConv2d"}:
                    row.update(operations_local_useful_dense_estimate=row["operations_executed_tensor_estimate"],
                        operations_estimate_scope="retained interior output arithmetic; pre-trim executed work is unknown",
                        pre_trim_executed_operations=None, split_dim=getattr(module, "split_dim", None),
                        halo_size=getattr(module, "halo_size", None),
                        source_contract="audited Wan spatial-shard convolution, then local trim/invalid-extent zeroing")
                    row["operations_executed_tensor_estimate"] = None
                elif type(module).__name__ == "WanCausalConv3d":
                    row["source_contract"] = "audited Wan causal cache concatenation/padding then direct Conv3d"
        if row is None:
            call["reason"] = "unsupported or inconsistent weighted leaf shape contract"
            return
        row.update(operator="modulehook::"+type(module).__name__, _scope_override=scope,
                   observation_source="actual weighted leaf input/output and weight contract; private dispatch not observed")
        self.append_record(row)
        call["observation"] = "weighted_leaf_fallback"
        if row["kind"] == "convolution":
            self.record_vae_attention_from_qkv(row)

    def detach_modules(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.module_stack.clear()
        self._module_names.clear()

    def append_record(self, row):
        if len(self.records) >= self.max_records:
            self.truncated = True
            return
        scope = row.pop("_scope_override", self.module_stack[-1] if self.module_stack else "unattributed")
        replicated = any(token in scope for token in ("text_encoder", "text_embedding", "time_embedding",
            "time_projection", "cross_attn.k", "cross_attn.v"))
        prefix_replicated = any(scope.endswith("."+name) for name in (
            "patch_embedding", "patch_embedding_wancamctrl", "c2ws_hidden_states_layer1", "c2ws_hidden_states_layer2"))
        vae_replicated = any(enabled and scope.startswith(parent+".")
                             for parent, enabled in self._vae_attention_scopes.items()) or scope == "vae.post_quant_conv"
        replication = ("SP_replicated_VAE" if vae_replicated else "SP_replicated_transformer_prefix" if prefix_replicated
                       else "SP_replicated_conditioning" if replicated
                       else "unresolved_or_sharded")
        row.update(module=scope, phase=self.phase, chunk=self.chunk, probe=self.probe, rank=self.rank,
                   replication=row.get("replication", replication),
                   operations_useful_serial=None,
                   numerator_scope=("retained interior output arithmetic; pre-trim execution unknown"
                       if row.get("operations_local_useful_dense_estimate") is not None
                       and row.get("pre_trim_executed_operations") is None
                       else "observed executed tensor arithmetic; not a full useful-model numerator"))
        self.records.append(row)

    def record_vae_attention_from_qkv(self, row):
        scope = row["module"]
        if not scope.endswith(".to_qkv") or scope.removesuffix(".to_qkv") not in self._vae_attention_scopes:
            return
        parent = scope.removesuffix(".to_qkv")
        output = row["output"]
        if len(output["shape"]) != 4 or output["shape"][1] % 3:
            return
        # Exact WanAttentionBlock source contract: one head, reshape its REAL
        # to_qkv output into [B*time,1,H*W,C], then dense SDPA. No tensor copy.
        b, tripled_channels, height, width = output["shape"]
        channels, tokens = tripled_channels//3, height*width
        descriptor = {"shape": [b, 1, tokens, channels], "dtype": output["dtype"], "device": output["device"]}
        self._pending_explicit_attention = (self.phase, self.chunk, self.probe, parent, b, tokens, tokens, 1, channels)
        self._pending_attention_matrix_stages.clear()
        self.append_record({"operator": "explicit::WanAttentionBlock_single_head_sdpa", "kind": "attention_qk_pv",
            "attention_kind": "vae", "operations_executed_tensor_estimate": 4*b*tokens*tokens*channels,
            "precision": output["dtype"].removeprefix("torch."), "inputs": [descriptor.copy() for _ in range(3)],
            "input_layout": "BNSD", "_scope_override": parent,
            "replication": "SP_replicated_VAE" if self._vae_attention_scopes[parent] else "unresolved_or_sharded",
            "observation_source": "WanAttentionBlock one-head contract and actual to_qkv convolution output shape",
            "attention_semantics": "Dense SDPA QK/PV; excludes softmax/vector math"})

    def record_attention(self, query, key, value, *, input_layout="BSND", attention_kind="self"):
        if query.ndim != 4 or key.ndim != 4 or value.shape != key.shape:
            raise ValueError("Explicit compute attention ledger needs matching4D Q/K/V")
        if input_layout == "BSND":
            b, qs, heads, dim = query.shape
            ks = key.shape[1]
        elif input_layout == "BNSD":
            b, heads, qs, dim = query.shape
            ks = key.shape[2]
        else:
            raise ValueError("Explicit attention compute layout must be BSND or BNSD")
        row = {"operator": "explicit::"+attention_kind+"_attention", "kind": "attention_qk_pv",
               "attention_kind": attention_kind, "operations_executed_tensor_estimate": 4*b*qs*ks*heads*dim,
               "precision": str(query.dtype).removeprefix("torch."), "input_layout": input_layout,
               "inputs": [_descriptor(query), _descriptor(key), _descriptor(value)],
               "attention_semantics": "Dense supplied visible QK/PV; excludes softmax/masks/vector math",
               "observation_source": "explicit gated model call before private CANN operator"}
        scope = self.module_stack[-1] if self.module_stack else "unattributed"
        self._pending_explicit_attention = (self.phase, self.chunk, self.probe, scope, b, qs, ks, heads, dim)
        self._pending_attention_matrix_stages.clear()
        self.append_record(row)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        result = func(*args, **kwargs)
        name = func._schema.name
        lower = name.lower()
        row = None
        matrices = {"aten::mm", "aten::bmm", "aten::matmul", "aten::addmm", "aten::linear"}
        if name in matrices or "quant_matmul" in lower or "weight_quant_batchmatmul" in lower:
            values = _parameters(func, args, kwargs)
            if name == "aten::addmm":
                a, b = values.get("mat1"), values.get("mat2")
            elif name == "aten::linear":
                a, b = values.get("input"), values.get("weight")
            elif name in matrices:
                a, b = args[:2]
            else:
                a, b = _tensor(args[0]), _tensor(args[1])
            output = _tensor(result)
            if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor) and output is not None:
                k = a.shape[-1]
                # Output elements account for broadcast batch/vector matmul;
                # no assumptions about transposed physical weight storage.
                operations = 2*k*output.numel()
                precision = "int8" if a.dtype == b.dtype == torch.int8 else str(a.dtype).removeprefix("torch.")
                if "weight_quant" in lower:
                    precision = "weight_only_mixed_unverified"
                row = {"kind": "matrix", "operations_executed_tensor_estimate": operations,
                       "precision": precision, "inputs": [_descriptor(a), _descriptor(b)],
                       "output": _descriptor(output)}
        elif name in {"aten::convolution", "aten::_convolution", "aten::conv2d", "aten::conv3d"}:
            values = _parameters(func, args, kwargs)
            x, weight, output = values.get("input"), values.get("weight"), _tensor(result)
            groups = int(values.get("groups", 1))
            if isinstance(x, torch.Tensor) and isinstance(weight, torch.Tensor) and output is not None:
                transposed = bool(values.get("transposed", False))
                # Transposed convolutions need actual scatter coverage; do not
                # assign the forward-convolution formula to that operator.
                operations = None if transposed else 2*output.numel()*weight.shape[1]*math.prod(weight.shape[2:])
                row = {"kind": "convolution", "operations_executed_tensor_estimate": operations,
                       "precision": str(x.dtype).removeprefix("torch."), "groups": groups,
                       "transposed": transposed, "inputs": [_descriptor(x), _descriptor(weight)],
                       "output": _descriptor(output), "stride": values.get("stride"),
                       "padding": values.get("padding"), "dilation": values.get("dilation"),
                       "useful_numerator_note": "Stream/cache/halo/crop redundancy not yet attributed"}
        elif any(token in lower for token in (
            "npu_fusion_attention", "npu_fused_infer_attention_score", "npu_prompt_flash_attention",
            "npu_incre_flash_attention", "scaled_dot_product_flash_attention", "scaled_dot_product_efficient_attention",
        )):
            values = _parameters(func, args, kwargs)
            q, k, v = (_tensor(values.get(name)) for name in ("query", "key", "value"))
            if q is None and len(args) >= 3:
                q, k, v = (_tensor(item) for item in args[:3])
            if q is not None and k is not None and v is not None and q.ndim == k.ndim == 4:
                layout = values.get("input_layout", "BNSD" if "scaled_dot_product" in lower else "unknown")
                if layout in {"BSND", "BNSD"}:
                    b, qs, heads, dim = q.shape if layout == "BSND" else (q.shape[0], q.shape[2], q.shape[1], q.shape[3])
                    ks = k.shape[1] if layout == "BSND" else k.shape[2]
                    row = {"kind": "attention_qk_pv", "operations_executed_tensor_estimate": 4*b*qs*ks*heads*dim,
                           "precision": str(q.dtype).removeprefix("torch."), "input_layout": layout,
                           "inputs": [_descriptor(q), _descriptor(k), _descriptor(v)],
                           "attention_semantics": "Dense supplied visible QK/PV; excludes softmax/masks/vector math"}
                    scope = self.module_stack[-1] if self.module_stack else "unattributed"
                    signature = (self.phase, self.chunk, self.probe, scope, b, qs, ks, heads, dim)
                    if signature == self._pending_explicit_attention:
                        row = None  # Explicit call already owns the useful QK/PV estimate.
                        self._pending_explicit_attention = None
                    elif row is not None:
                        row["attention_kind"] = "cross" if "cross_attn" in scope else "self"
        if row is not None:
            if row["kind"] == "matrix" and self._pending_explicit_attention is not None:
                phase, chunk, probe, scope, batch, qs, ks, heads, dim = self._pending_explicit_attention
                current_scope = self.module_stack[-1] if self.module_stack else "unattributed"
                shape = row["output"]["shape"]
                if (phase, chunk, probe, scope) == (self.phase, self.chunk, self.probe, current_scope) and len(shape) >= 2:
                    stage = None
                    if (shape[-2:] == [qs, ks] and math.prod(shape[:-2]) == batch*heads
                            and row["inputs"][0]["shape"][-1] == dim
                            and "qk" not in self._pending_attention_matrix_stages):
                        stage = "qk"
                    elif (shape[-2:] == [qs, dim] and math.prod(shape[:-2]) == batch*heads
                            and row["inputs"][0]["shape"][-1] == ks
                            and "pv" not in self._pending_attention_matrix_stages):
                        stage = "pv"
                    if stage is not None:
                        self._pending_attention_matrix_stages.add(stage)
                        if self._pending_attention_matrix_stages == {"qk", "pv"}:
                            self._pending_explicit_attention = None
                            self._pending_attention_matrix_stages.clear()
                        return result  # Explicit attention owns these QK/PV matrices.
            row["operator"] = name
            self.append_record(row)
            if row["kind"] == "convolution" and "module" in row:
                self.record_vae_attention_from_qkv(row)
        elif any(token in lower for token in ("linear", "matmul", "conv")):
            self.unrecognized_weighted_operator_names[name] = self.unrecognized_weighted_operator_names.get(name, 0)+1
        return result

    def summary(self) -> dict[str, Any]:
        groups = {}
        for row in self.records:
            key = (row["phase"], row["kind"], row["precision"], row["replication"])
            target = groups.setdefault(key, {"phase": key[0], "kind": key[1], "precision": key[2],
                                            "replication": key[3], "calls": 0, "operations_executed_tensor_estimate": 0,
                                            "unknown_operations_calls": 0})
            target["calls"] += 1
            ops = row["operations_executed_tensor_estimate"]
            if ops is None:
                target["unknown_operations_calls"] += 1
            else:
                target["operations_executed_tensor_estimate"] += ops
        probes = sorted({row["probe"] for row in self.records if row["phase"] == "probe" and row["probe"] is not None})
        attention_counts = {kind: sum(row["kind"] == "attention_qk_pv" and row.get("attention_kind") == kind
                                     for row in self.records) for kind in ("self", "cross")}
        camera_suffixes = (".cam_injector_layer1", ".cam_injector_layer2", ".cam_scale_layer", ".cam_shift_layer")
        matrix_suffixes = {"self_qkv": ".self_attn.qkv", "self_out": ".self_attn.o",
                           "cross_q": ".cross_attn.q", "cross_out": ".cross_attn.o",
                           "ffn_expand": ".ffn.0", "ffn_contract": ".ffn.2"}
        matrix_calls = {name: sum(row["kind"] == "matrix" and row["module"].endswith(suffix)
                                  for row in self.records) for name, suffix in matrix_suffixes.items()}
        camera_calls = sum(row["kind"] == "matrix" and row["module"].endswith(camera_suffixes) for row in self.records)
        prefix_suffixes = ("patch_embedding", "patch_embedding_wancamctrl", "c2ws_hidden_states_layer1",
                           "c2ws_hidden_states_layer2", "time_embedding.0", "time_embedding.2", "time_projection.1")
        prefix_calls = {name: sum(row["kind"] in {"matrix", "convolution"}
                                 and row["module"].endswith("."+name) for row in self.records)
                        for name in prefix_suffixes}
        coverage = {"observed_probe_indices": probes, "expected_probe_indices": [0, 1, 2, 3],
                    "self_attention_calls": attention_counts["self"], "cross_attention_calls": attention_counts["cross"],
                    "expected_self_attention_calls": 160, "expected_cross_attention_calls": 160,
                    "main_projection_ffn_calls": matrix_calls, "expected_each_main_projection_ffn_calls": 160,
                    "camera_calls": camera_calls, "expected_camera_calls_nonterminal": 160,
                    "transformer_prefix_calls": prefix_calls, "expected_each_transformer_prefix_calls": 4,
                    "vae_attention_calls": sum(row.get("attention_kind") == "vae" for row in self.records),
                    "expected_vae_attention_calls": 3,
                    "vae_convolution_observed": any(row["kind"] == "convolution" and row["phase"] == "decode"
                                                    for row in self.records),
                    "vae_convolution_calls": sum(row["kind"] == "convolution" and row["phase"] == "decode"
                                                  and row["module"].startswith("vae.") for row in self.records),
                    "expected_vae_convolution_calls": 117,
                    "private_attention_requires_explicit_hooks": True,
                    "vector_quantizer_arithmetic_included": False}
        complete = (not self.truncated and probes == [0, 1, 2, 3]
                    and attention_counts == {"self": 160, "cross": 160} and coverage["vae_convolution_calls"] == 117
                    and coverage["vae_attention_calls"] == 3 and camera_calls == 160
                    and all(value == 160 for value in matrix_calls.values())
                    and all(value == 4 for value in prefix_calls.values()))
        return {"rank": self.rank, "logical_ranks": self.logical_ranks, "timing_eligible": False,
                "record_buffer_complete": not self.truncated, "capture_complete": complete,
                "scheduled_compute_regions_coverage_complete": complete,
                "full_model_arithmetic_complete": False, "coverage": coverage,
                "weighted_module_calls": self.weighted_module_calls,
                "unrecognized_weighted_operator_names": self.unrecognized_weighted_operator_names,
                "truncated": self.truncated, "groups": list(groups.values()),
                "records": self.records, "whole_model_mfu": None, "useful_complete_model_operations": None,
                "limitations": ["Executed tensor estimate includes SP duplicates and convolution/cache/halo work",
                    "Useful serial convolution requires crop attribution", "Softmax/vector/quantizer arithmetic excluded",
                    "Normalized mixed MFU requires verified precision peaks and complete useful numerator"]}


def begin_compute_capture(pipeline, *, phase: str, chunk: int, probe: int | None = None) -> None:
    """Root calls before probe/decode; select one full steady chunk by env."""
    if os.environ.get("LINGBOT_CAPTURE_COMPUTE", "0") != "1":
        return
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    if rank != int(os.environ.get("LINGBOT_CAPTURE_RANK", "0")):
        return
    existing = getattr(pipeline, "_lingbot_compute_capture", None)
    if existing is not None:
        existing.set_phase(phase, chunk, probe)
        return
    if phase == "startup":
        if os.environ.get("LINGBOT_CAPTURE_COMPUTE_STARTUP", "0") != "1":
            return
    elif chunk != int(os.environ.get("LINGBOT_CAPTURE_COMPUTE_CHUNK", "11")):
        return
    key = "startup" if phase == "startup" else f"chunk{chunk}"
    finished = getattr(pipeline, "_lingbot_compute_capture_finished", set())
    if key in finished:
        return
    destination = Path(os.environ.get("LINGBOT_CAPTURE_COMPUTE_DIR", os.environ.get("LINGBOT_CAPTURE_DIR", ""))).resolve()
    if not str(destination).startswith("/mnt/data/"):
        raise ValueError("Compute diagnostic captures must use persistent /mnt/data storage")
    destination.mkdir(parents=True, exist_ok=True)
    recorder = ComputeRecorder(rank=rank)
    recorder.set_phase(phase, chunk, probe)
    recorder.attach_modules(pipeline)
    recorder.__enter__()
    global _active_compute_recorder
    _active_compute_recorder = recorder
    pipeline._lingbot_compute_capture = recorder
    pipeline._lingbot_compute_capture_destination = destination
    pipeline._lingbot_compute_capture_key = key


def finish_compute_capture(pipeline) -> None:
    recorder = getattr(pipeline, "_lingbot_compute_capture", None)
    if recorder is None:
        return
    recorder.__exit__(None, None, None)
    global _active_compute_recorder
    _active_compute_recorder = None
    recorder.detach_modules()
    destination = pipeline._lingbot_compute_capture_destination
    key = pipeline._lingbot_compute_capture_key
    path = destination / f"compute_{key}_rank{recorder.rank}.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(recorder.summary(), indent=2)+"\n")
    temporary.replace(path)
    finished = getattr(pipeline, "_lingbot_compute_capture_finished", set())
    finished.add(key)
    pipeline._lingbot_compute_capture_finished = finished
    pipeline._lingbot_compute_capture = None


def record_attention(query, key, value, *, input_layout="BSND", attention_kind="self") -> None:
    """No copies/device work; explicit coverage for private CANN attention APIs."""
    if os.environ.get("LINGBOT_CAPTURE_COMPUTE", "0") != "1" or _active_compute_recorder is None:
        return
    _active_compute_recorder.record_attention(query, key, value,
                                              input_layout=input_layout, attention_kind=attention_kind)


@contextmanager
def compute_module_scope(module, value):
    """Attribute direct payload GEMMs that intentionally bypass module.__call__."""
    recorder = _active_compute_recorder if os.environ.get("LINGBOT_CAPTURE_COMPUTE", "0") == "1" else None
    scope = recorder._module_names.get(id(module)) if recorder is not None else None
    if recorder is None or scope is None:
        yield
        return
    start = len(recorder.records)
    recorder.module_stack.append(scope)
    state = {}
    try:
        yield state
    finally:
        recorder.record_weighted_leaf(module, scope, value, _tensor(state.get("output")), start)
        if recorder.module_stack and recorder.module_stack[-1] == scope:
            recorder.module_stack.pop()
