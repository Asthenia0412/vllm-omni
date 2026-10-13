# SPDX-License-Identifier: Apache-2.0
"""Independent, default-off cache of original FP32 block modulation and views.

StageA is a source-pinned provenance dependency, not inherited qualification.
Only frozen inference weights, SP8 eager/default-stream consumers and published
StageA banks are supported. Inference tensors may lack a version counter: this
contract does not support or claim detection of arbitrary hot weight mutation.
"""
from dataclasses import dataclass, fields, is_dataclass
from types import MappingProxyType
import sys
import threading
import weakref

FLAG = "lingbot_npu_fixed_modulation_stageB"
STATE = "_lingbot_fixed_modulation_stageB_state"
REQUESTED = "_lingbot_fixed_modulation_stageB_requested"
SESSION_BYTES = 24 * 1024**2
_QUARANTINE = []


def _stage_a():
    from . import lingbot_fixed_timestep_bank_stageA_v1
    return lingbot_fixed_timestep_bank_stageA_v1


def _report(error, owners):
    from .lingbot_sp8_fatal import report_failure
    report_failure("fixed_modulation_stageB", repr(error), owners)


def _need(condition, message):
    if not condition:
        raise RuntimeError("fixed modulation StageB: " + message)


def _function(value):
    return getattr(value, "__func__", value)


class _Callable:
    """Hold function/code owners; do not hash or compile on the forward path."""
    def __init__(self, value):
        self.owner = value
        function = _function(value)
        self.bound = getattr(value, "__self__", None)
        self.wrapper = None
        if not hasattr(function, "__code__"):
            self.wrapper = value
            function = _function(type(value).__call__)
        self.function, self.code = function, function.__code__

    def check(self, value):
        function = _function(value)
        if self.wrapper is not None:
            _need(value is self.wrapper, "callable wrapper owner changed")
            function = _function(type(value).__call__)
        _need(function is self.function and function.__code__ is self.code
              and getattr(value, "__self__", None) is self.bound,
              "sealed producer/provider function or code changed")


def _metadata(value):
    """Small static SP declarations only, never tensor values or dynamic caches."""
    if value is None or type(value) in (bool, int, float, str):
        return (type(value), value)
    if type(value) in (tuple, list):
        return (id(value), tuple(_metadata(v) for v in value))
    if type(value) is dict:
        return (id(value), tuple((k, _metadata(v)) for k, v in value.items()))
    if is_dataclass(value):
        return (id(value), tuple((f.name, _metadata(getattr(value, f.name))) for f in fields(value)))
    raise RuntimeError("fixed modulation StageB: unsupported static SP metadata type")


class _Producers:
    """Frozen SP/staging route. Hot hooks or policy mutation are unsupported."""
    def __init__(self, transformer):
        self.bindings, self.identities, self.values = [], [], []
        self.bind(transformer, "forward")
        self.bind(transformer, "_stage_timestep_projection")
        self.sp = transformer.sp_prepare
        self.identity(transformer, "sp_prepare")
        self.bind(self.sp, "forward")
        if hasattr(self.sp, "_omni_original_forward"):
            self.bind(self.sp, "_omni_original_forward")
        module = sys.modules[type(transformer).__module__]
        for name in ("prepare_uniform_timestep", "prepare_timestep_projection"):
            self.bind(module, name)
        for name in ("_lingbot_uniform_timestep", "_lingbot_timestep_fp32_once"):
            self.scalar(transformer, name)
        # Torch hook dictionaries are usually empty; retain and check any actual
        # callable owners instead of assuming their absence from source alone.
        for name in ("_forward_pre_hooks", "_forward_hooks"):
            hooks = getattr(self.sp, name, None)
            self.identity(self.sp, name)
            if hooks is not None:
                self.values.append((lambda h=hooks: tuple((k, id(v)) for k, v in h.items()),
                                    tuple((k, id(v)) for k, v in hooks.items())))
                for key, hook in hooks.items():
                    self.bindings.append((lambda h=hooks, k=key: h[k], _Callable(hook)))
        registry = getattr(self.sp, "_hook_registry", None)
        self.identity(self.sp, "_hook_registry")
        if registry is not None:
            self.identity(registry, "module")
            _need(registry.module is self.sp, "SP hook registry owns another module")
            self.bind(registry, "dispatch")
            hooks = registry._hooks
            self.identity(registry, "_hooks")
            self.values.append((lambda: tuple((k, id(v)) for k, v in registry._hooks.items()),
                                tuple((k, id(v)) for k, v in hooks.items())))
            self.values.append((lambda: tuple(id(v) for v in registry._sorted_hooks),
                                tuple(id(v) for v in registry._sorted_hooks)))
            self.identity(registry, "_new_fwd_impl_hook")
            for hook in hooks.values():
                for name in ("pre_forward", "post_forward", "new_forward"):
                    self.bind(hook, name)
                self.identity(hook, "metadata")
                self.values.append((lambda h=hook: _metadata(h.metadata), _metadata(hook.metadata)))
                self.identity(hook, "config")
                for name in ("ulysses_degree", "ring_degree", "allgather_degree", "convert_to_fp32"):
                    self.scalar(hook.config, name)
                self.identity(hook, "module_forward_metadata")
                if hasattr(hook, "fn_ref"):
                    self.identity(hook, "fn_ref")
                    self.bind(hook.fn_ref, "original_forward")

    def bind(self, owner, name):
        self.bindings.append((lambda o=owner, n=name: getattr(o, n), _Callable(getattr(owner, name))))

    def identity(self, owner, name):
        self.identities.append((owner, name, getattr(owner, name, None)))

    def scalar(self, owner, name):
        value = getattr(owner, name)
        _need(type(value) in (bool, int, str), "unsupported producer policy scalar")
        self.values.append((lambda o=owner, n=name: (type(getattr(o, n)), getattr(o, n)), (type(value), value)))

    def check(self):
        for owner, name, value in self.identities:
            _need(getattr(owner, name, None) is value, "sealed SP module/hook owner changed")
        for getter, expected in self.values:
            _need(getter() == expected, "sealed SP hook policy/order changed")
        for getter, expected in self.bindings:
            expected.check(getter())


def configure(pipeline, config):
    value = config.get(FLAG, False)
    _need(type(value) is bool, "flag must be an explicit bool")
    _need(not value or config.get("lingbot_npu_fixed_timestep_bank") is True,
          "StageB requires enabled StageA")
    pipeline.__dict__[REQUESTED] = value
    pipeline.__dict__[STATE] = None
    pipeline.transformer.__dict__[REQUESTED] = value
    pipeline.transformer.__dict__[STATE] = None


def reservation_bytes(pipeline):
    return SESSION_BYTES if pipeline.__dict__.get(REQUESTED, False) else 0


def _poison(model, error, *owners):
    retained = (model, error, *owners)
    _QUARANTINE.append(retained)
    if type(model) is _Model:
        model.poisoned = True
        model.retained.append(retained)
        try:
            model.stage_a_model.poison(*retained)
        except BaseException:
            pass  # Retain the B owners even if the dependency is already invalid.
    try:
        _report(error, retained)
    except BaseException:
        pass  # Keep the original exception; never retry or release quarantine.


@dataclass(frozen=True)
class _Parameter:
    block: object
    forward: object
    owned: object


@dataclass(frozen=True)
class _Entry:
    result: object
    views: tuple
    owned: tuple
    extent: int
    start: int

    def check(self):
        for owner in self.owned:
            owner.check_light()


class _Bank:
    def __init__(self, session):
        self.session = session
        self.generation = session.generation
        self.entries = {}
        self.published = None
        self.completed_probes = set()
        self.bytes = 0
        self.active = True
        self.retiring = False


class _Model:
    def __init__(self, pipeline, worker, a):
        self.a = a
        self.torch = a.torch
        self.stage_a_model = pipeline.__dict__.get(a.STATE)
        _need(type(self.stage_a_model) is a._Model, "seal requires the actual sealed StageA owner")
        self.stage_a_model.check_light()
        _need(a._supported_policy(pipeline), "seal requires eager SP8 fixed uniform policy")
        self.pipeline = weakref.ref(pipeline)
        self.worker = weakref.ref(worker)
        self.transformer = weakref.ref(pipeline.transformer)
        self.blocks_owner = pipeline.transformer.blocks
        self.blocks = tuple(self.blocks_owner)
        self.thread = threading.get_ident()
        self.device = pipeline.device
        self.stream = self.torch.npu.default_stream(self.device)
        self.describe = self.stage_a_model.describe
        self.generation = object()
        self.banks = {}
        self.current_token = None
        self.current_projection = None
        self.poisoned = False
        self.retained = []
        self.producers = _Producers(pipeline.transformer)
        self.parameters = []
        for block in self.blocks:
            value = block.modulation
            _need(tuple(value.shape) == (1, 6, 5120) and value.dtype == self.torch.bfloat16
                  and value.device == self.device, "exact BF16 modulation parameter required")
            owned = a._OwnedTensor.take(value, a._physical(self.describe, value))
            self.parameters.append(_Parameter(block, _Callable(block.forward), owned))
        _need(len(self.parameters) == 40, "exactly40 blocks required")
        _need(self.torch.npu.current_stream(self.device) == self.stream,
              "seal requires original default stream")

    def check_context(self, transformer):
        _need(not self.poisoned, "worker poisoned; restart all8 workers")
        _need(threading.get_ident() == self.thread, "foreign producer thread")
        pipeline, worker = self.pipeline(), self.worker()
        _need(pipeline is not None and worker is not None and self.transformer() is transformer
              and worker.model_runner.pipeline is pipeline and pipeline.transformer is transformer
              and pipeline.__dict__.get(STATE) is self and transformer.__dict__.get(STATE) is self
              and pipeline.__dict__.get(self.a.STATE) is self.stage_a_model
              and transformer.blocks is self.blocks_owner and len(transformer.blocks) == 40,
              "sealed pipeline/worker/transformer/StageA owner changed")
        _need(pipeline.__dict__.get(REQUESTED) is True and transformer.__dict__.get(REQUESTED) is True
              and pipeline.od_config.model_config.get(FLAG) is True,
              "StageB configuration changed")
        self.producers.check()
        self.stage_a_model.check_light()  # Once per forward, not once per layer.
        return pipeline


def seal_after_worker_load(pipeline, worker):
    if not pipeline.__dict__.get(REQUESTED, False):
        return
    model = None
    try:
        _need(pipeline.__dict__.get(STATE) is None, "model already sealed")
        model = _Model(pipeline, worker, _stage_a())
        pipeline.__dict__[STATE] = model
        pipeline.transformer.__dict__[STATE] = model
    except BaseException as error:
        _poison(model, error, pipeline, worker)
        raise


def prepare_forward(transformer, token, timestep, source_projection, projection, uniform_timestep):
    """Authorize only actual published StageA outputs after the original SP path.

    The trusted, independently pinned transformer hook supplies both the original
    pre-SP output and the actual post-SP/staged input. A reused staging address
    is never treated as proof of its value or probe identity.
    """
    if not transformer.__dict__.get(REQUESTED, False):
        return None
    model = transformer.__dict__.get(STATE)
    keep = (transformer, token, timestep, source_projection, projection)
    try:
        _need(type(model) is _Model, "enabled model was not sealed")
        pipeline = model.check_context(transformer)
        a, t = model.a, model.torch
        if token is None:
            return None
        _need(type(token) is a._CallToken and type(token.probe) is a.ProbeToken,
              "actual StageA DMD call capability required")
        probe = token.probe
        session = probe.session
        _need(type(session) is a._Session and session.model is model.stage_a_model
              and session.active
              and session.model.sessions.get(session.id) is session
              and session.generation is probe.generation and session.model.generation is probe.model_generation
              and type(probe.index) is int and probe.index in range(4)
              and probe.fp32_timestep == session.schedule[probe.index][2],
              "stale session/model/probe capability")
        _need(timestep is token.timestep.value, "actual timestep owner differs from DMD capability")
        token.timestep.check_light()
        if (session.cache_disabled or not a._supported_policy(pipeline) or uniform_timestep is not True
                or t.is_grad_enabled() or not t.is_inference_mode_enabled()
                or t.compiler.is_compiling() or t.npu.is_current_stream_capturing()
                or projection.device.type != "npu" or projection.dtype != t.bfloat16
                or tuple(projection.shape) != (1, 1, 6, 5120)
                or t.npu.current_stream(model.device) != model.stream):
            return None
        _need(projection.device == model.device and timestep.device == model.device
              and t.npu.current_device() == model.device.index, "current/operand device owner changed")
        if session.published is None:
            return None  # StageA cold/building remains exactly native; no B bank mutation.
        pair = session.published[probe.index]
        _need(source_projection is pair[1].value, "source is not this probe's original published projection")
        pair[0].check_light()
        pair[1].check_light()
        _need(tuple(source_projection.shape) == (1, 3, 6, 5120)
              and source_projection.dtype == t.bfloat16 and source_projection.device == model.device,
              "published projection ABI differs")
        bank = model.banks.get(session.id)
        if bank is None:
            _need(len(model.banks) < 2, "two-session persistent bank capacity exhausted")
            bank = _Bank(session)
            model.banks[session.id] = bank
        _need(bank.active and not bank.retiring and bank.session is session
              and bank.generation is session.generation, "retired/replaced StageB session")
        if model.current_token is token:
            _need(model.current_projection is projection, "same call replaced its staged projection")
        model.current_token, model.current_projection = token, projection
        return Forward(model, transformer, token, timestep, source_projection, projection, bank)
    except BaseException as error:
        _poison(model, error, *keep)
        raise


class Forward:
    """Per-invocation capability; hot checks do not rescan all40 parameters."""
    def __init__(self, model, transformer, token, timestep, source_projection, projection, bank):
        self.model, self.transformer, self.token = model, transformer, token
        self.timestep, self.source_projection, self.projection = timestep, source_projection, projection
        self.blocks, self.session, self.probe_index = model.blocks, bank.session, token.probe.index
        self.bank = bank
        self.projection_owner = model.a._OwnedTensor.take(projection)
        self.next_index = 0
        self.used_layers = []
        self.finished = False

    def _check(self):
        m, s = self.model, self.session
        _need(not self.finished and not m.poisoned and self.bank.active and not self.bank.retiring
              and s.active and m.stage_a_model.sessions.get(s.id) is s
              and m.banks.get(s.id) is self.bank and self.bank.generation is s.generation
              and self.token.probe.generation is s.generation
              and self.token.probe.model_generation is m.stage_a_model.generation
              and m.current_token is self.token and m.current_projection is self.projection,
              "stale/finished forward capability")
        _need(threading.get_ident() == m.thread, "foreign layer consumer thread")
        self.projection_owner.check_light()

    def get(self, index, block, projection):
        result = views = None
        try:
            self._check()
            _need(type(index) is int and index == self.next_index and index in range(40)
                  and block is self.blocks[index] and projection is self.projection,
                  "layer order/block/projection owner differs")
            parameter = self.model.parameters[index]
            _need(block.modulation is parameter.owned.value, "modulation parameter replaced")
            parameter.forward.check(block.forward)
            parameter.owned.check_light()
            key = (self.probe_index, index)
            if self.bank.published is not None:
                entry = self.bank.published[key]
                entry.check()
                views = entry.views
            else:
                # Literal original operation order and mixed BF16+FP32 boundary.
                result = block.modulation.unsqueeze(1) + projection.float()
                views = result.chunk(6, dim=2)
                if key not in self.bank.entries:
                    entry = self._entry(result, views)
                    _need(len(self.bank.entries) < 160 and self.bank.bytes + entry.extent <= SESSION_BYTES,
                          "bounded bank allocation budget exceeded after original computation")
                    _need(all(entry.start + entry.extent <= old.start or old.start + old.extent <= entry.start
                              for old in self.bank.entries.values()), "bank outputs overlap complete allocations")
                    self.bank.entries[key] = entry
                    self.bank.bytes += entry.extent
            self.used_layers.append(index)
            self.next_index += 1
            return views
        except BaseException as error:
            _poison(self.model, error, self, block, projection, result, views)
            raise

    def _entry(self, result, views):
        a, t, describe = self.model.a, self.model.torch, self.model.describe
        _need(type(views) is tuple and len(views) == 6 and result.dtype == t.float32
              and tuple(result.shape) == (1, 1, 6, 5120), "original modulation output ABI changed")
        owned = tuple(a._OwnedTensor.take(v, a._physical(describe, v)) for v in (result, *views))
        storage = result.untyped_storage()
        _need(all(v.dtype == t.float32 and v.device == result.device
                  and tuple(v.shape) == (1, 1, 1, 5120)
                  and v.untyped_storage()._cdata == storage._cdata
                  and v.storage_offset() == result.storage_offset() + i * result.stride(2)
                  and tuple(v.stride()) == tuple(result.stride()) for i, v in enumerate(views)),
              "original six views changed owner/layout/order")
        _need(result.device == self.model.device and storage.nbytes() > 0,
              "result allocation/device differs")
        return _Entry(result, views, owned, storage.nbytes(), storage.data_ptr())


def abort_forward(ctx, error):
    """Quarantine consumer failures from the pinned surrounding model hook."""
    if ctx is None:
        return
    model = getattr(ctx, "model", None)
    if type(model) is _Model and model.poisoned:
        retained = (model, ctx, error, error.__traceback__)
        _QUARANTINE.append(retained)
        model.retained.append(retained)
        return  # The first terminal failure already reported the worker restart.
    _poison(model, error, ctx, error.__traceback__)


def bank_published(ctx):
    """Read-only status for the separately pinned correctness-only observer."""
    return type(ctx) is Forward and ctx.bank.published is not None


def finish_forward(ctx):
    if ctx is None:
        return
    try:
        _need(type(ctx) is Forward, "actual forward capability required")
        ctx._check()
        _need(ctx.next_index == 40 and ctx.used_layers == list(range(40)), "forward did not complete all40 layers")
        ctx.bank.completed_probes.add(ctx.probe_index)
        if ctx.bank.published is None and len(ctx.bank.entries) == 160 and ctx.bank.completed_probes == set(range(4)):
            for entry in ctx.bank.entries.values():
                entry.check()
            ctx.bank.published = MappingProxyType(dict(ctx.bank.entries))
        ctx.finished = True  # Hook belongs after successful head/gather/unpatchify.
    except BaseException as error:
        _poison(getattr(ctx, "model", None), error, ctx)
        raise


def begin_retire(pipeline, session_id):
    model = pipeline.__dict__.get(STATE)
    if model is None:
        return
    try:
        model.check_context(pipeline.transformer)
        bank = model.banks.get(session_id)
        if bank is not None:
            _need(bank.active and not bank.retiring, "session retirement already started")
            _need(model.stage_a_model.sessions.get(session_id) is bank.session and bank.session.used,
                  "StageA drain owner missing or never used")
            bank.active = False
            bank.retiring = True
    except BaseException as error:
        _poison(model, error, pipeline, session_id)
        raise


def retire_after_stageA(pipeline, session_id):
    """Only after the unchanged StageA retire succeeded and removed this session."""
    model = pipeline.__dict__.get(STATE)
    if model is None:
        return
    try:
        model.check_context(pipeline.transformer)
        bank = model.banks.get(session_id)
        if bank is None:
            return
        _need(bank.retiring and not bank.active and not bank.session.active
              and bank.session.used and session_id not in model.stage_a_model.sessions,
              "StageA successful used-session drain/removal required before release")
        if model.current_token is not None and model.current_token.probe.session is bank.session:
            model.current_token = model.current_projection = None
        bank.published = None
        bank.entries.clear()
        bank.bytes = 0
        del model.banks[session_id]
    except BaseException as error:
        _poison(model, error, pipeline, session_id)
        raise
