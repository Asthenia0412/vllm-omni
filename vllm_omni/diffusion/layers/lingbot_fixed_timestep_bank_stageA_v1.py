# SPDX-License-Identifier: Apache-2.0
"""Default-off, frozen-worker StageA: retain original three-row BF16 results.

Handwritten experimental SOURCE. This file is not installed by its author.
No StageB modulation cache, changed timestep arithmetic, SP shortcut, graph,
new OPP, raw launch, or general hot-weight-mutation support is implemented.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import os
import struct
from types import MappingProxyType
import weakref

import torch


FLAG = "lingbot_npu_fixed_timestep_bank"
STATE = "_lingbot_fixed_timestep_stageA_state"
REQUESTED = "_lingbot_fixed_timestep_stageA_requested"
DIAGNOSTICS = "_lingbot_fixed_timestep_stageA_diagnostics"
DESCRIPTOR = {
    "path": "/mnt/data/zhoutaichang/project/lingbot-world/profiling/runs/20261002-fixed4-realtime-implementation/qkv_exchange_descriptor_cpu_build_v1/liblingbot_storage_descriptor.so",
    "bytes": 89376,
    "sha256": "23127093ca55147fddd0d961b397f3f1b0ec2fe6ea6c6b320b7bed9a61cbbdae",
}
FIELDS = ("npu_format", "origin_format", "base_sizes", "base_strides", "physical_sizes",
          "shape", "stride", "storage_offset", "storage_bytes", "storage_data_ptr", "data_ptr")
SESSION_BYTES = 4 * 3 * 5120 * (1 + 6) * 2 + 8 * 4096  # 892928; StageA only.
GRAPH_ENV = ("LINGBOT_REGION_CAPTURE_DIR", "LINGBOT_POINTWISE_CAPTURE_DIR",
             "LINGBOT_LEASED_GRAPH_CAPTURE_DIR", "LINGBOT_FFN_GRAPH_CAPTURE_DIR")


def need(value, message):
    if not value:
        raise RuntimeError("fixed timestep StageA: " + message)


def configure(pipeline, model_config):
    value = model_config.get(FLAG, False)
    if type(value) is not bool:
        raise ValueError("model_config." + FLAG + " must be a bool")
    pipeline.__dict__[REQUESTED] = value
    pipeline.__dict__[STATE] = None
    diagnostic = model_config.get("lingbot_npu_fixed_timestep_bank_diagnostics", False)
    if type(diagnostic) is not bool:
        raise ValueError("fixed timestep bank diagnostics must be a bool")
    pipeline.__dict__[DIAGNOSTICS] = diagnostic


def reservation_bytes(pipeline):
    return SESSION_BYTES if pipeline.__dict__.get(REQUESTED, False) else 0


def _version(tensor):
    try:
        return tensor._version
    except RuntimeError as error:
        if "Inference tensors do not track version counter" in str(error):
            return None  # Explicitly unavailable; frozen-worker contract is still required.
        raise


def _light(tensor):
    storage = tensor.untyped_storage()
    return (tensor._cdata, storage._cdata, tuple(tensor.shape), tuple(tensor.stride()),
            tensor.storage_offset(), str(tensor.dtype), str(tensor.device), tensor.data_ptr(),
            storage.data_ptr(), storage.nbytes(), _version(tensor))


def _function(value):
    return getattr(value, "__func__", value)


def _physical(describe, tensor):
    values = describe(tensor)
    need(len(values) == 11, "the existing eleven-field descriptor ABI is required")
    record = dict(zip(FIELDS, values))
    for name in ("base_sizes", "base_strides", "physical_sizes", "shape", "stride"):
        record[name] = tuple(record[name])
    storage = tensor.untyped_storage()
    need(record["shape"] == tuple(tensor.shape) and record["stride"] == tuple(tensor.stride())
         and record["storage_offset"] == tensor.storage_offset()
         and record["storage_bytes"] == storage.nbytes()
         and record["storage_data_ptr"] == storage.data_ptr() and record["data_ptr"] == tensor.data_ptr(),
         "framework and physical descriptor/extent disagree")
    need(record["storage_bytes"] > 0 and record["storage_data_ptr"] > 0
         and all(type(x) is int and x > 0 for x in record["physical_sizes"])
         and all(type(x) is int and x >= 0 for x in record["stride"]), "invalid physical metadata")
    highest = tensor.storage_offset() + sum((n - 1) * s for n, s in zip(tensor.shape, tensor.stride()))
    need(highest >= 0 and (highest + 1) * tensor.element_size() <= record["storage_bytes"],
         "logical view exceeds the complete allocation extent")
    return record


def _stream_key(stream):
    return str(stream.device), int(stream.npu_stream)


def _supported_policy(pipeline):
    config = pipeline.od_config.model_config or {}
    parallel = pipeline.od_config.parallel_config
    if not bool(getattr(pipeline.od_config, "enforce_eager", False)):
        return False
    if (getattr(parallel, "tensor_parallel_size", 1) != 1
            or getattr(parallel, "sequence_parallel_size", 1) != 8
            or getattr(parallel, "ulysses_degree", 1) != 8
            or getattr(parallel, "ring_degree", 1) != 1):
        return False
    if (not config.get("lingbot_npu_uniform_timestep", False)
            or not config.get("lingbot_reuse_last_step_kv", False)
            or config.get("lingbot_npu_timestep_fp32_once", False)
            or config.get("lingbot_npu_regional_graph", False)
            or config.get("lingbot_npu_pointwise_graph_regions", [])
            or config.get("lingbot_npu_leased_post_attention", False)
            or config.get("lingbot_npu_output_overlap_tiles", 0)
            or config.get("lingbot_npu_self_output_mc2", False)
            or config.get("lingbot_npu_fusion_backend", "off") != "off"
            or any(os.environ.get(name) for name in GRAPH_ENV)):
        return False
    transformer = pipeline.transformer
    if (pipeline.device.type != "npu" or transformer.dim != 5120
            or tuple(transformer.config.patch_size) != (1, 2, 2)
            or len(transformer.blocks) != 40):
        return False
    # StageA is limited to the current stepwise frozen worker. Other invocation
    # modes receive no capability token and keep the original model route.
    return bool(getattr(pipeline.od_config, "step_execution", False))


def _descriptor_at_startup(pipeline):
    """Consume only the already registered, accepted MI4 descriptor capability.

    MI4 selected_route validates retained construction configuration and the
    fresh startup manifest, then performs its existing route initialization.
    This moves that one-time setup to enabled worker load; its cost and the
    post-setup platform fence must be reported. It is not claimed to be a
    registration-only or zero-device-work operation. StageA adds no DSO loader.
    """
    config = pipeline.od_config.model_config or {}
    if config.get("lingbot_npu_qk_single_host_admission_v1") is True:
        from vllm_omni.diffusion.layers.qk_single_host_integration_v1 import selected_route
        from vllm_omni.platforms import current_omni_platform

        route = selected_route()
        need(route is not None, "enabled MI4 construction must initialize its original route")
        current_omni_platform.synchronize()
        return route.describe, route
    return getattr(torch.ops.lingbot_storage_descriptor, "describe", None), None


@dataclass(frozen=True, eq=False)
class _OwnedTensor:
    value: torch.Tensor
    storage: object
    metadata: tuple
    physical: object = None

    @classmethod
    def take(cls, value, physical=None):
        return cls(value, value.untyped_storage(), _light(value), physical)

    def check_light(self):
        need(_light(self.value) == self.metadata, "retained output owner/metadata changed")


class _Model:
    def __init__(self, pipeline, worker, describe, startup_owner):
        from vllm_omni.diffusion.distributed.parallel_state import get_sp_group

        self.pipeline = weakref.ref(pipeline)
        self.worker = weakref.ref(worker)
        self.transformer = weakref.ref(pipeline.transformer)
        self.generation = object()  # Published only by the actual successful worker load owner.
        self.describe = describe
        self.startup_owner = startup_owner
        self.group = get_sp_group()
        self.rank = int(self.group.ulysses_rank)
        self.device = pipeline.device
        self.stream = torch.npu.default_stream(self.device)
        self.stream_key = _stream_key(self.stream)
        self.sessions = {}
        self.poisoned = False
        self.retained_on_failure = []
        self.diagnostics = pipeline.__dict__.get(DIAGNOSTICS, False)
        self.counts = {key: 0 for key in ("original_calls", "hits", "fallbacks", "publishes", "retirements", "poisons")}
        transformer = pipeline.transformer
        self.time_embedding = transformer.time_embedding
        self.time_projection = transformer.time_projection
        self.embedding_function = _function(transformer._timestep_embeddings)
        self.providers = tuple((module, _function(module.forward)) for branch in
                               (self.time_embedding, self.time_projection) for module in branch.modules())
        self.weights = []
        for branch in (self.time_embedding, self.time_projection):
            for module in branch.modules():
                for name, value in module._parameters.items():
                    if value is not None:
                        need(value.dtype == torch.bfloat16 and value.device == self.device,
                             "StageA requires final BF16 time weights on the worker NPU")
                        self.weights.append((module, name, _OwnedTensor.take(value, _physical(describe, value))))
        need(len(self.weights) == 6 and 0 <= self.rank < 8, "exact current time-weight/rank geometry required")
        need(_stream_key(torch.npu.current_stream(self.device)) == self.stream_key,
             "seal must occur on the actual original default stream")

    def poison(self, *owners):
        if self.diagnostics and not self.poisoned:
            self.counts["poisons"] += 1
        self.poisoned = True
        self.retained_on_failure.append(tuple(owners))

    def check_light(self):
        from vllm_omni.diffusion.distributed.parallel_state import get_sp_group

        need(not self.poisoned, "worker is poisoned; root must abort and reclaim all eight workers")
        pipeline, worker, transformer = self.pipeline(), self.worker(), self.transformer()
        need(pipeline is not None and worker is not None and transformer is not None
             and worker.model_runner.pipeline is pipeline and pipeline.transformer is transformer
             and pipeline.__dict__.get(STATE) is self, "sealed model owner/generation is stale")
        need(transformer.time_embedding is self.time_embedding and transformer.time_projection is self.time_projection
             and _function(transformer._timestep_embeddings) is self.embedding_function,
             "sealed top-level conditioning provider changed")
        need(get_sp_group() is self.group and int(self.group.ulysses_rank) == self.rank,
             "sequence group/rank owner changed")
        for module, name, owned in self.weights:
            need(module._parameters.get(name) is owned.value and owned.value._cdata == owned.metadata[0]
                 and _version(owned.value) == owned.metadata[-1], "sealed time-weight owner/version changed")

    def check_publication(self):
        self.check_light()
        for module, function in self.providers:
            need(_function(module.forward) is function, "conditioning provider changed before publication")
        for _, _, owned in self.weights:
            owned.check_light()
            need(_physical(self.describe, owned.value) == owned.physical,
                 "sealed physical time-weight descriptor changed before publication")


def seal_after_worker_load(pipeline, worker):
    if not pipeline.__dict__.get(REQUESTED, False):
        return  # Disabled route performs no hash, registration, descriptor call or NPU fence.
    need(worker.model_runner.pipeline is pipeline, "only the actual successful worker-load owner may seal")
    need(pipeline.__dict__.get(STATE) is None, "a sealed model instance cannot be reloaded in place")
    if not _supported_policy(pipeline):
        pipeline._lingbot_fixed_timestep_stageA_unavailable = "unsupported frozen worker policy"
        return
    describe, startup_owner = _descriptor_at_startup(pipeline)
    if describe is None:
        pipeline._lingbot_fixed_timestep_stageA_unavailable = "existing descriptor capability absent"
        return
    model = _Model(pipeline, worker, describe, startup_owner)
    pipeline.__dict__[STATE] = model  # Final publication; construction failure does not publish a token.


def _schedule_key(schedule):
    if type(schedule) is not tuple or len(schedule) != 4:
        return None
    try:
        pairs = tuple((float(t), float(s)) for t, s in schedule)
        if any(not math.isfinite(t) or not math.isfinite(s) for t, s in pairs):
            return None
        return tuple((struct.pack("!d", t), struct.pack("!d", s), struct.pack("!f", t)) for t, s in pairs)
    except (TypeError, ValueError, OverflowError, struct.error):
        return None


class _Session:
    def __init__(self, model, session_id, key):
        self.model = model
        self.id = session_id
        self.generation = object()
        self.schedule = key
        self.building = {}
        self.published = None
        self.active = True
        self.cache_disabled = False
        self.used = False
        self.hits = 0
        self.original_calls = 0


@dataclass(frozen=True, eq=False)
class ProbeToken:
    session: _Session
    generation: object
    model_generation: object
    index: int
    fp32_timestep: bytes


@dataclass(frozen=True, eq=False)
class _CallToken:
    probe: ProbeToken
    timestep: _OwnedTensor


def begin_session(pipeline, session_id, *, schedule, flow_shift, height, width, block_frames):
    model = pipeline.__dict__.get(STATE)
    if model is None:
        return None
    model.check_light()
    key = _schedule_key(schedule)
    if (flow_shift != 5.0 or height != 480 or width != 832 or block_frames != 3
            or key is None or not _supported_policy(pipeline)):
        if model.diagnostics:
            model.counts["fallbacks"] += 1
        return None
    need(session_id not in model.sessions, "existing session must reset/close before a new generation")
    session = _Session(model, session_id, key)
    model.sessions[session_id] = session
    return session


def probe_token(session, *, schedule, index, timestep_value):
    if session is None:
        return None
    need(type(session) is _Session and session.active and not session.model.poisoned,
         "session capability is stale or poisoned")
    if (type(index) is not int or index not in range(4) or session.cache_disabled
            or _schedule_key(schedule) != session.schedule):
        if session.model.diagnostics:
            session.model.counts["fallbacks"] += 1
        return None
    try:
        bits = struct.pack("!f", float(timestep_value))
    except (ValueError, OverflowError, struct.error):
        return None
    if bits != session.schedule[index][2]:
        if session.model.diagnostics:
            session.model.counts["fallbacks"] += 1
        return None
    return ProbeToken(session, session.generation, session.model.generation, index, bits)


def bind_timestep(token, timestep, *, timestep_value, step_index):
    """Only the literal DMD producer binds its unchanged freshly created full.

    This binds actual Tensor/StorageImpl ownership and its descriptor to the
    already registered host scalar/index. It is not a tensor-value read or a
    proof for arbitrary foreign tensor construction; the frozen exact-source
    DMD torch.full followed immediately by this hook is the producer proof.
    """
    if token is None:
        return None
    need(type(token) is ProbeToken, "DMD producer requires the original host probe capability")
    session, model = token.session, token.session.model
    need(session.active and not model.poisoned and model.sessions.get(session.id) is session
         and session.generation is token.generation and model.generation is token.model_generation,
         "DMD producer received a stale host capability")
    need(type(step_index) is int and step_index == token.index
         and struct.pack("!f", float(timestep_value)) == token.fp32_timestep,
         "DMD full scalar/index differs from the registered host probe")
    need(type(timestep) is torch.Tensor and tuple(timestep.shape) == (1,)
         and timestep.dtype == torch.float32 and timestep.device == model.device,
         "DMD full owner/metadata differs from the trusted producer contract")
    return _CallToken(token, _OwnedTensor.take(timestep))


def _preflight(transformer, token, timestep, batch_size, frames, dtype, geometry):
    if token is None:
        return None
    need(type(token) is _CallToken, "transformer requires the DMD-bound call capability")
    need(timestep is token.timestep.value, "actual timestep Tensor owner differs from the DMD-bound full")
    token.timestep.check_light()
    probe = token.probe
    session, model = probe.session, probe.session.model
    model.check_light()
    need(session.active and model.sessions.get(session.id) is session
         and session.generation is probe.generation and model.generation is probe.model_generation
         and model.transformer() is transformer and probe.index in range(4)
         and probe.fp32_timestep == session.schedule[probe.index][2], "stale or invalid published capability")
    # Unsupported request metadata selects original arithmetic before cache use
    # or any candidate computation. No tensor values are copied to the host.
    if (session.cache_disabled or torch.is_grad_enabled() or torch.compiler.is_compiling()
            or batch_size != 1 or frames != 3
            or dtype != torch.bfloat16 or geometry != (30, 52)
            or timestep.ndim != 1 or tuple(timestep.shape) != (1,)
            or timestep.dtype != torch.float32 or timestep.device != model.device
            or _stream_key(torch.npu.current_stream(model.device)) != model.stream_key
            or not _supported_policy(model.pipeline())):
        if model.diagnostics:
            model.counts["fallbacks"] += 1
        return None
    return session


def embeddings(transformer, timestep, *, batch_size, frames, dtype, geometry, token=None):
    """Return originals or readonly cached originals; never combine probe GEMMs."""
    session = _preflight(transformer, token, timestep, batch_size, frames, dtype, geometry)
    if session is None:
        return transformer._timestep_embeddings(timestep, batch_size=batch_size, frames=frames, dtype=dtype)
    model = session.model
    try:
        if session.published is not None:
            embedding, projection = session.published[token.probe.index]
            embedding.check_light()
            projection.check_light()
            if model.diagnostics:
                session.hits += 1
                model.counts["hits"] += 1
            session.used = True
            return embedding.value, projection.value
        # The unchanged function retains original FP32-full input rounding,
        # sinusoid conversion and the native 3-row BF16 GEMM/provider choices.
        values = transformer._timestep_embeddings(timestep, batch_size=batch_size, frames=frames, dtype=dtype)
        if model.diagnostics:
            session.original_calls += 1
            model.counts["original_calls"] += 1
        session.used = True
        need(type(values) is tuple and len(values) == 2, "original embedding result contract changed")
        expected = ((1, 3, 5120), (1, 3, 6, 5120))
        for value, shape in zip(values, expected):
            need(tuple(value.shape) == shape and value.dtype == torch.bfloat16 and value.device == model.device,
                 "original three-row BF16 result metadata differs")
        if token.probe.index not in session.building:
            pair = tuple(_OwnedTensor.take(value) for value in values)
            retained = sum(owned.metadata[9] for row in session.building.values() for owned in row)
            if retained + sum(owned.metadata[9] for owned in pair) > SESSION_BYTES:
                session.cache_disabled = True  # Never retain beyond the declared persistent capacity.
                return values
            session.building[token.probe.index] = pair
        if len(session.building) == 4:
            model.check_publication()
            published = {}
            allocations = []
            for index in range(4):
                row = []
                for owned in session.building[index]:
                    owned.check_light()
                    physical = _physical(model.describe, owned.value)
                    row.append(_OwnedTensor.take(owned.value, physical))
                    allocations.append((physical["storage_data_ptr"], physical["storage_bytes"]))
                published[index] = tuple(row)
            for i, (start, length) in enumerate(allocations):
                need(all(start + length <= other or other + extent <= start
                         for other, extent in allocations[i + 1:]), "original bank outputs overlap allocations")
            session.published = MappingProxyType(published)  # Only a complete four-probe bank becomes usable.
            session.building = {}
            if model.diagnostics:
                model.counts["publishes"] += 1
        return values
    except BaseException:
        model.poison(session, timestep, transformer)
        raise  # No execution-started fallback. Root must terminate/reclaim/restart all eight workers.


def retire_session(pipeline, session_id):
    model = pipeline.__dict__.get(STATE)
    if model is None:
        return  # In particular: no unconditional fence on the original disabled reset path.
    model.check_light()
    session = model.sessions.get(session_id)
    if session is None:
        return
    session.active = False  # Prevent new uses before retirement/fence.
    try:
        if session.used:
            # Explicit first-version lifecycle boundary, outside per-call
            # inference. Only original-default-stream consumers are supported.
            # Teardown wall and this fence must be reported, not hidden.
            model.stream.synchronize()
        session.building = {}
        session.published = None
        del model.sessions[session_id]
        if model.diagnostics:
            model.counts["retirements"] += 1
    except BaseException:
        model.poison(session)
        raise  # Retain all banks and fail closed; caller must restart the workers.


def diagnostic_counts(pipeline, session_id):
    """Untimed/read-only diagnostic SOURCE API; never an admission by itself."""
    model = pipeline.__dict__.get(STATE)
    session = None if model is None else model.sessions.get(session_id)
    return dict(enabled=model is not None and model.diagnostics, sealed=model is not None,
                totals=None if model is None or not model.diagnostics else dict(model.counts),
                active=session is not None and session.active,
                hits=0 if session is None else session.hits,
                original_calls=0 if session is None else session.original_calls,
                published=session is not None and session.published is not None,
                lifecycle_proven=False, model_admitted=False)
