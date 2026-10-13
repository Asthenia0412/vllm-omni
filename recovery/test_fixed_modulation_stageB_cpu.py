# SPDX-License-Identifier: Apache-2.0
"""CPU contract tests using explicit tensor/StageA doubles; no NPU proof.

The six represented scalar channels follow IEEE float32/BF16 rounding, while
logical shapes, storage extents and aliases retain the real 5120-channel ABI.
This does not replace actual BF16/FP32 device or model bitwise correctness.
"""
from dataclasses import dataclass
import importlib.util
from pathlib import Path
import struct
import sys
from types import SimpleNamespace as NS
import unittest

SOURCE = Path(__file__).resolve().parents[1] / 'vllm_omni/diffusion/layers/lingbot_fixed_modulation_stageB.py'
spec = importlib.util.spec_from_file_location('_stageB_cpu_under_test', SOURCE)
b = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = b
spec.loader.exec_module(b)


def fp32(v):
    return struct.unpack('!f', struct.pack('!f', v))[0]


def bf16(v):
    u = struct.unpack('!I', struct.pack('!f', v))[0]
    u = (u + 0x7fff + ((u >> 16) & 1)) & 0xffff0000
    return struct.unpack('!f', struct.pack('!I', u))[0]


@dataclass(frozen=True)
class Device:
    type: str = 'npu'
    index: int = 0


class Storage:
    address = 0x1000000
    def __init__(self, size):
        self.size = size
        self.ptr = Storage.address
        self._cdata = self.ptr
        Storage.address += size + 4096
    def data_ptr(self): return self.ptr
    def nbytes(self): return self.size


def strides(shape):
    out, n = [], 1
    for x in reversed(shape):
        out.append(n)
        n *= x
    return tuple(reversed(out))


class Tensor:
    ops = []
    allocation_extra = 0
    add_error = False
    def __init__(self, shape, values, dtype='bf16', *, storage=None, stride=None, offset=0, device=None):
        self.shape, self.dtype = tuple(shape), dtype
        self.device = device or Device()
        self._stride = tuple(stride or strides(shape))
        self.offset, self._version, self._cdata = offset, 0, id(self)
        n = 1
        for x in shape: n *= x
        self.storage = storage or Storage(n * self.element_size() + self.allocation_extra)
        self.values = tuple(bf16(v) if dtype == 'bf16' else fp32(v) for v in values)
    def untyped_storage(self): return self.storage
    def stride(self, dim=None): return self._stride if dim is None else self._stride[dim]
    def storage_offset(self): return self.offset
    def element_size(self): return 2 if self.dtype == 'bf16' else 4
    def data_ptr(self): return self.storage.ptr + self.offset * self.element_size()
    def unsqueeze(self, dim):
        Tensor.ops.append(('unsqueeze', self.dtype, dim))
        shape = list(self.shape); shape.insert(dim, 1)
        ss = list(self._stride); ss.insert(dim, self._stride[dim] * self.shape[dim])
        return Tensor(shape, self.values, self.dtype, storage=self.storage, stride=ss, offset=self.offset)
    def float(self):
        Tensor.ops.append(('float', self.dtype))
        return Tensor(self.shape, self.values, 'fp32')
    def __add__(self, rhs):
        Tensor.ops.append(('add', self.dtype, rhs.dtype))
        if self.add_error: raise RuntimeError('injected original add failure')
        return Tensor(self.shape, [x + y for x, y in zip(self.values, rhs.values)], 'fp32')
    def chunk(self, chunks, dim):
        Tensor.ops.append(('chunk', self.dtype, chunks, dim))
        return tuple(Tensor((1,1,1,5120), [v], self.dtype, storage=self.storage,
                            stride=self._stride, offset=self.offset + i*self._stride[dim])
                     for i, v in enumerate(self.values))


def light(v):
    s = v.untyped_storage()
    return (v._cdata, s._cdata, v.shape, v.stride(), v.storage_offset(), v.dtype,
            v.device, v.data_ptr(), s.data_ptr(), s.nbytes(), v._version)


class Owned:
    def __init__(self, value, physical):
        self.value, self.storage, self.metadata, self.physical = value, value.storage, light(value), physical
    @classmethod
    def take(cls, value, physical=None): return cls(value, physical)
    def check_light(self):
        if light(self.value) != self.metadata: raise RuntimeError('retained owner changed')


class AModel:
    def __init__(self):
        self.sessions, self.generation, self.poisoned = {}, object(), False
        self.checks, self.retained = 0, []
        self.describe = lambda v: light(v)
    def check_light(self):
        if self.poisoned: raise RuntimeError('StageA poisoned')
        self.checks += 1
    def poison(self, *owners): self.poisoned = True; self.retained.append(owners)


class Session:
    def __init__(self, model):
        self.id, self.model, self.generation = 'actual-session', model, object()
        self.active, self.cache_disabled, self.used = True, False, True
        self.schedule = tuple((None, None, struct.pack('!f', float(i))) for i in range(4))
        self.published = {i: (Owned.take(Tensor((1,3,5120), [float(i)])),
                              Owned.take(Tensor((1,3,6,5120), [i*.25+j/64 for j in range(6)])))
                          for i in range(4)}
        model.sessions[self.id] = self


@dataclass(frozen=True)
class Probe:
    session: object
    generation: object
    model_generation: object
    index: int
    fp32_timestep: bytes


@dataclass(frozen=True)
class Token:
    probe: object
    timestep: object


class Box: pass


class Block:
    def __init__(self): self.modulation = Tensor((1,6,5120), [1+j/32 for j in range(6)])
    def forward(self): pass


def prepare_uniform_timestep(*args): pass

def prepare_timestep_projection(*args): pass


class SPPrepare:
    def __init__(self):
        self._forward_pre_hooks, self._forward_hooks = {}, {}
    def forward(self, *args): pass


class Transformer:
    def __init__(self):
        self.sp_prepare = SPPrepare()
        self._lingbot_uniform_timestep = True
        self._lingbot_timestep_fp32_once = False
    def forward(self, *args): pass
    def _stage_timestep_projection(self, x): return x


class Env:
    def __init__(self):
        self.t = NS(bfloat16='bf16', float32='fp32',
                    is_grad_enabled=lambda: False, is_inference_mode_enabled=lambda: True,
                    compiler=NS(is_compiling=lambda: False))
        self.stream = object()
        self.t.npu = NS(default_stream=lambda d: self.stream, current_stream=lambda d: self.stream,
                        current_device=lambda: 0, is_current_stream_capturing=lambda: False)
        self.a = NS(STATE='_stageA', _Model=AModel, _Session=Session, ProbeToken=Probe,
                    _CallToken=Token, _OwnedTensor=Owned, torch=self.t,
                    _supported_policy=lambda p: getattr(p, 'supported', True),
                    _physical=lambda describe, value: describe(value))
        self.pipeline, self.worker, self.transformer = Box(), Box(), Transformer()
        self.pipeline.device = Device(); self.pipeline.transformer = self.transformer
        self.transformer.blocks = [Block() for _ in range(40)]
        self.pipeline.od_config = NS(model_config={b.FLAG: True, 'lingbot_npu_fixed_timestep_bank': True})
        self.worker.model_runner = NS(pipeline=self.pipeline)
        self.am = AModel(); self.pipeline.__dict__[self.a.STATE] = self.am
        self.session = Session(self.am)
        self.tokens = tuple(Token(Probe(self.session, self.session.generation, self.am.generation,
                                       i, self.session.schedule[i][2]),
                                  Owned.take(Tensor((1,), [i], 'fp32'))) for i in range(4))
        self.projected = tuple(Tensor((1,1,6,5120), self.session.published[i][1].value.values) for i in range(4))
        b._stage_a = lambda: self.a
        b._report = lambda *args: None
        b.configure(self.pipeline, self.pipeline.od_config.model_config)
        b.seal_after_worker_load(self.pipeline, self.worker)
        self.model = self.pipeline.__dict__[b.STATE]
    def prepare(self, i=0, **kwargs):
        token = kwargs.pop('token', self.tokens[i])
        return b.prepare_forward(self.transformer, token, kwargs.pop('timestep', token.timestep.value),
                                 kwargs.pop('source') if 'source' in kwargs else self.session.published[i][1].value,
                                 kwargs.pop('projection', self.projected[i]), kwargs.pop('uniform', True))
    def run(self, i=0, finish=True):
        ctx = self.prepare(i)
        values = [ctx.get(j, block, self.projected[i]) for j, block in enumerate(self.transformer.blocks)]
        if finish: b.finish_forward(ctx)
        return ctx, values
    def fill(self):
        for i in range(4): self.run(i)
        return self.model.banks[self.session.id]


class StageBTests(unittest.TestCase):
    def setUp(self):
        b._QUARANTINE.clear(); Tensor.ops.clear(); Tensor.allocation_extra = 0; Tensor.add_error = False
        self.e = Env()
        Tensor.ops.clear()
    def test_disabled_is_zero_work(self):
        p = Box(); p.transformer = Box()
        b.configure(p, {})
        self.assertEqual(b.reservation_bytes(p), 0)
        b.seal_after_worker_load(p, None)
        self.assertIsNone(b.prepare_forward(p.transformer, None, None, None, None, False))
        self.assertEqual(Tensor.ops, [])
    def test_config_requires_bool_and_stageA(self):
        for config in ({b.FLAG: 1}, {b.FLAG: True}):
            with self.assertRaises(RuntimeError): b.configure(Box(), config)
    def test_reserved_budget(self): self.assertEqual(b.reservation_bytes(self.e.pipeline), 24*1024**2)
    def test_building_has_no_bank_or_operations(self):
        source = self.e.session.published[0][1].value
        self.e.session.published = None
        self.assertIsNone(self.e.prepare(source=source))
        self.assertFalse(self.e.model.banks); self.assertEqual(Tensor.ops, [])
    def test_ordinary_unsupported_has_no_bank(self):
        for change in ('uniform', 'policy', 'disabled', 'grad', 'capture', 'stream', 'shape', 'dtype'):
            e = Env()
            if change == 'policy': e.pipeline.supported = False
            if change == 'disabled': e.session.cache_disabled = True
            if change == 'grad': e.t.is_grad_enabled = lambda: True
            if change == 'capture': e.t.npu.is_current_stream_capturing = lambda: True
            if change == 'stream': e.t.npu.current_stream = lambda d: object()
            if change == 'shape': e.projected[0].shape = (1,2,6,5120)
            if change == 'dtype': e.projected[0].dtype = 'fp32'
            self.assertIsNone(e.prepare(uniform=change != 'uniform'), change)
            self.assertFalse(e.model.banks); self.assertFalse(e.model.poisoned)
    def test_literal_order_preserves_fp32_add(self):
        e = self.e
        e.transformer.blocks[0].modulation.values = (1.0,)*6
        e.projected[0].values = (1/256,)*6
        ctx = e.prepare(); out = ctx.get(0, e.transformer.blocks[0], e.projected[0])
        self.assertEqual(Tensor.ops, [('unsqueeze','bf16',1), ('float','bf16'), ('add','bf16','fp32'), ('chunk','fp32',6,2)])
        self.assertEqual(out[0].values, (1+1/256,))
        self.assertNotEqual(out[0].values[0], bf16(1+1/256))
    def test_all160_and_four_successful_forwards_required(self):
        e = self.e
        for i in range(3): e.run(i)
        bank = e.model.banks[e.session.id]
        self.assertEqual(len(bank.entries), 120); self.assertIsNone(bank.published)
        ctx, values = e.run(3, finish=False)
        self.assertEqual(len(bank.entries), 160); self.assertIsNone(bank.published)
        b.finish_forward(ctx)
        self.assertTrue(b.bank_published(ctx)); self.assertEqual(bank.bytes, 19660800)
        self.assertNotEqual(bank.published[(0,0)].views[0].values, bank.published[(3,0)].views[0].values)
        with self.assertRaises(TypeError): bank.published[(0,0)] = None
    def test_partial_bank_never_hits(self):
        self.e.run(0); Tensor.ops.clear()
        self.e.run(0)
        self.assertEqual(sum(v[0]=='add' for v in Tensor.ops), 40)
        self.assertEqual(len(self.e.model.banks[self.e.session.id].entries), 40)
    def test_published_hit_returns_original_views_without_math(self):
        bank = self.e.fill(); Tensor.ops.clear(); before = self.e.am.checks
        _, values = self.e.run(2)
        self.assertEqual(Tensor.ops, []); self.assertIs(values[0], bank.published[(2,0)].views)
        self.assertEqual(self.e.am.checks-before, 1)
    def test_source_identity_not_shape_is_authority(self):
        with self.assertRaisesRegex(RuntimeError, 'original published'):
            self.e.prepare(source=Tensor((1,3,6,5120), [0]*6))
        self.assertTrue(self.e.model.poisoned); self.assertTrue(b._QUARANTINE)
    def test_timestep_owner_is_authority(self):
        with self.assertRaisesRegex(RuntimeError, 'timestep owner'):
            self.e.prepare(timestep=Tensor((1,), [0], 'fp32'))
    def test_stale_generations_sessions_and_probe_rejected(self):
        for change in ('sessiongen', 'modelgen', 'replacement', 'inactive', 'timestepbits'):
            e = Env(); probe = e.tokens[0].probe
            if change == 'sessiongen': e.session.generation = object()
            if change == 'modelgen': e.am.generation = object()
            if change == 'replacement': e.am.sessions[e.session.id] = object()
            if change == 'inactive': e.session.active = False
            if change == 'timestepbits':
                e.tokens = (Token(Probe(probe.session, probe.generation, probe.model_generation, 0, b'wrong'), e.tokens[0].timestep), *e.tokens[1:])
            with self.assertRaises(RuntimeError, msg=change): e.prepare()
    def test_layer_order_and_finished_reentry_rejected(self):
        ctx = self.e.prepare()
        with self.assertRaisesRegex(RuntimeError, 'layer order'): ctx.get(1, self.e.transformer.blocks[1], self.e.projected[0])
        e = Env(); ctx, _ = e.run()
        with self.assertRaisesRegex(RuntimeError, 'finished'): ctx.get(0, e.transformer.blocks[0], e.projected[0])
    def test_incomplete_finish_poison_retains_bank(self):
        ctx = self.e.prepare(); ctx.get(0, self.e.transformer.blocks[0], self.e.projected[0])
        with self.assertRaisesRegex(RuntimeError, 'all40'): b.finish_forward(ctx)
        self.assertEqual(len(ctx.bank.entries), 1); self.assertTrue(self.e.model.poisoned)
    def test_retained_weight_owner_and_version_changes_rejected(self):
        for change in ('replace', 'version', 'metadata', 'provider'):
            e = Env(); ctx = e.prepare(); block = e.transformer.blocks[0]
            if change == 'replace': block.modulation = Tensor((1,6,5120), [0]*6)
            if change == 'version': block.modulation._version += 1
            if change == 'metadata': block.modulation.storage.size += 32
            if change == 'provider': block.forward = lambda: None
            with self.assertRaises(RuntimeError, msg=change): ctx.get(0, block, e.projected[0])
    def test_output_view_version_mutation_rejected(self):
        bank = self.e.fill(); bank.published[(0,0)].views[0]._version += 1
        ctx = self.e.prepare()
        with self.assertRaisesRegex(RuntimeError, 'retained owner'): ctx.get(0, self.e.transformer.blocks[0], self.e.projected[0])
    def test_budget_is_actual_owner_extent_and_failure_retains(self):
        Tensor.allocation_extra = b.SESSION_BYTES
        ctx = self.e.prepare()
        with self.assertRaisesRegex(RuntimeError, 'budget'): ctx.get(0, self.e.transformer.blocks[0], self.e.projected[0])
        self.assertEqual(ctx.bank.bytes, 0); self.assertTrue(self.e.model.retained)
        self.assertTrue(any(isinstance(x, Tensor) for x in self.e.model.retained[-1]))
    def test_overlapping_allocations_rejected_after_original_math(self):
        ctx = self.e.prepare(); ctx.get(0, self.e.transformer.blocks[0], self.e.projected[0])
        old = ctx.bank.entries[(0,0)]
        Storage.address = old.start - (6*5120*4+4096) # float allocation then result aliases prior allocation
        with self.assertRaisesRegex(RuntimeError, 'overlap'): ctx.get(1, self.e.transformer.blocks[1], self.e.projected[0])
        self.assertEqual(len(ctx.bank.entries), 1)
    def test_native_failure_terminal_and_no_fallback(self):
        ctx = self.e.prepare(); Tensor.add_error = True
        with self.assertRaisesRegex(RuntimeError, 'injected'): ctx.get(0, self.e.transformer.blocks[0], self.e.projected[0])
        self.assertTrue(ctx.model.poisoned); self.assertEqual(sum(v[0]=='add' for v in Tensor.ops), 1)
    def test_stale_forward_after_other_probe_and_valid_reprepare(self):
        ctx = self.e.prepare(); self.e.prepare(1)
        # Correctness observer may restore the actual saved projection and re-authorize old token.
        self.e.prepare(0)
        ctx.get(0, self.e.transformer.blocks[0], self.e.projected[0])
        self.e.prepare(1)
        with self.assertRaisesRegex(RuntimeError, 'stale'): ctx.get(1, self.e.transformer.blocks[1], self.e.projected[0])
    def test_same_token_replaced_projection_rejected(self):
        self.e.prepare()
        with self.assertRaisesRegex(RuntimeError, 'same call'):
            self.e.prepare(projection=Tensor((1,1,6,5120), [0]*6))
    def test_wrong_device_and_thread_terminal(self):
        e = self.e; e.t.npu.current_device = lambda: 1
        with self.assertRaisesRegex(RuntimeError, 'device'): e.prepare()
        e = Env(); e.model.thread = -1
        with self.assertRaisesRegex(RuntimeError, 'thread'): e.prepare()
    def test_retirement_requires_begin_and_actual_A_removal(self):
        self.e.run()
        with self.assertRaisesRegex(RuntimeError, 'drain/removal'): b.retire_after_stageA(self.e.pipeline, self.e.session.id)
        self.assertTrue(self.e.model.banks)
        e = Env(); e.run(); b.begin_retire(e.pipeline, e.session.id)
        self.assertTrue(e.model.banks[e.session.id].retiring)
        with self.assertRaisesRegex(RuntimeError, 'drain/removal'): b.retire_after_stageA(e.pipeline, e.session.id)
        self.assertTrue(e.model.banks)
    def test_retirement_after_successful_A_drain_releases(self):
        self.e.fill(); bank = self.e.model.banks[self.e.session.id]
        b.begin_retire(self.e.pipeline, self.e.session.id)
        self.assertFalse(bank.active); self.assertTrue(bank.entries)
        # Simulate only the public postcondition of the unchanged A fence; no device fence is claimed.
        self.e.session.active = False; del self.e.am.sessions[self.e.session.id]
        b.retire_after_stageA(self.e.pipeline, self.e.session.id)
        self.assertFalse(self.e.model.banks); self.assertFalse(bank.entries); self.assertEqual(bank.bytes, 0)
        self.assertIsNone(self.e.model.current_token)
    def test_retiring_bank_prevents_reentry(self):
        self.e.run(); b.begin_retire(self.e.pipeline, self.e.session.id)
        with self.assertRaisesRegex(RuntimeError, 'retired'): self.e.prepare()
        self.assertTrue(self.e.model.banks[self.e.session.id].entries)
    def test_two_session_capacity_is_explicit(self):
        self.e.prepare()
        for number in (2, 3):
            e = self.e
            old_session = e.session
            second = Session(e.am)
            # Session test helper's default ID is rewritten; restore the first mapping.
            e.am.sessions[old_session.id] = old_session
            second.id = 'session-' + str(number)
            e.am.sessions[second.id] = second
            token = Token(Probe(second, second.generation, e.am.generation, 0, second.schedule[0][2]),
                          Owned.take(Tensor((1,), [0], 'fp32')))
            if number == 2:
                self.assertIsNotNone(e.prepare(token=token, source=second.published[0][1].value))
            else:
                with self.assertRaisesRegex(RuntimeError, 'two-session'):
                    e.prepare(token=token, source=second.published[0][1].value)
        self.assertEqual(len(self.e.model.banks), 2)

    def test_same_function_code_mutation_rejected(self):
        function = prepare_uniform_timestep
        original = function.__code__
        try:
            function.__code__ = (lambda *args: 'changed').__code__
            with self.assertRaisesRegex(RuntimeError, 'function or code'): self.e.prepare()
        finally:
            function.__code__ = original

    def test_current_layer_forward_code_mutation_rejected(self):
        ctx = self.e.prepare()
        function, original = Block.forward, Block.forward.__code__
        try:
            function.__code__ = (lambda self: 'changed').__code__
            with self.assertRaisesRegex(RuntimeError, 'function or code'):
                ctx.get(0, self.e.transformer.blocks[0], self.e.projected[0])
        finally:
            function.__code__ = original

    def test_producer_replacement_and_policy_mutation_rejected(self):
        for change in ('sp_module', 'sp_forward', 'stage_method', 'torch_hooks', 'policy'):
            e = Env()
            if change == 'sp_module': e.transformer.sp_prepare = SPPrepare()
            if change == 'sp_forward': e.transformer.sp_prepare.forward = lambda *args: None
            if change == 'stage_method': e.transformer._stage_timestep_projection = lambda x: x
            if change == 'torch_hooks': e.transformer.sp_prepare._forward_hooks[1] = lambda *args: None
            if change == 'policy': e.transformer._lingbot_uniform_timestep = False
            with self.assertRaises(RuntimeError, msg=change): e.prepare()

    def test_wrapped_sp_registry_owner_and_policy_sealed(self):
        class Wrapped:
            def __init__(self, module): self.module = module
            def __call__(self, *args): return self.module._omni_original_forward(*args)
        class Hook:
            def pre_forward(self, *args): pass
            def post_forward(self, *args): pass
            def new_forward(self, *args): pass
        class Registry:
            def dispatch(self, *args): pass
        e = self.e
        sp = e.transformer.sp_prepare
        sp._omni_original_forward = sp.forward
        sp.forward = Wrapped(sp)
        registry, hook = Registry(), Hook()
        registry.module = sp
        registry._hooks = {'sp': hook}; registry._sorted_hooks = [hook]; registry._new_fwd_impl_hook = None
        hook.metadata = {2: {'split_dim': 1, 'expected_dims': 4, 'split_output': True}}
        hook.config = NS(ulysses_degree=8, ring_degree=1, allgather_degree=1, convert_to_fp32=True)
        hook.module_forward_metadata = object()
        hook.fn_ref = NS(original_forward=sp._omni_original_forward)
        sp._hook_registry = registry
        # Construct the independent snapshot directly; actual startup seals once.
        snapshot = b._Producers(e.transformer)
        snapshot.check()
        hook.metadata[2]['split_dim'] = 0
        with self.assertRaisesRegex(RuntimeError, 'policy/order'): snapshot.check()

    def test_abort_native_consumer_retains_and_reports_once(self):
        reports = []
        b._report = lambda *args: reports.append(args)
        ctx = self.e.prepare(); ctx.get(0, self.e.transformer.blocks[0], self.e.projected[0])
        error = RuntimeError('native consumer failed after modulation')
        b.abort_forward(ctx, error)
        self.assertTrue(self.e.model.poisoned); self.assertTrue(self.e.am.poisoned)
        self.assertEqual(len(reports), 1); self.assertEqual(len(ctx.bank.entries), 1)
        count = len(self.e.model.retained)
        b.abort_forward(ctx, error)
        self.assertEqual(len(reports), 1); self.assertEqual(len(self.e.model.retained), count + 1)
        b.abort_forward(None, error)
        self.assertEqual(len(reports), 1)


    def test_stageA_poison_hook_cannot_mask_original(self):
        self.e.am.poison = lambda *x: (_ for _ in ()).throw(ValueError('secondary'))
        with self.assertRaisesRegex(RuntimeError, 'original published'):
            self.e.prepare(source=Tensor((1,3,6,5120), [0]*6))
        self.assertTrue(self.e.model.retained)


if __name__ == '__main__': unittest.main()
