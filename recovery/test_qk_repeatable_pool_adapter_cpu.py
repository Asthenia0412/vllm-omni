"""CPU-only adapter contracts using fake tensors/ops; no framework import/device."""
import ast
import importlib.util
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
NAME = 'vllm_omni.diffusion.layers.lingbot_qk_repeatable_pool'
spec = importlib.util.spec_from_file_location(NAME, ROOT/'vllm_omni/diffusion/layers/lingbot_qk_repeatable_pool.py')
qk = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qk)
NATIVE_NAME = 'vllm_omni.diffusion.layers.lingbot_native_bf16_qk'
native_spec = importlib.util.spec_from_file_location(NATIVE_NAME, ROOT/'vllm_omni/diffusion/layers/lingbot_native_bf16_qk.py')
native = importlib.util.module_from_spec(native_spec)
sys.modules[NATIVE_NAME] = native
native_spec.loader.exec_module(native)
FIELDS = ('producer_tid', 'initialized', 'workspace_initial_bytes', 'workspace_resident_bytes',
          'native_closed', 'closed', 'pending', 'poisoned', 'retired_count', 'retained_call_slots')


class Device:
    type, index = 'npu', 0
    def __str__(self): return 'npu:0'


class Tensor:
    serial = 1024
    def __init__(self, shape, *, device=None, pointer=None):
        self.shape = tuple(shape)
        self.device = device or DEVICE
        self.dtype = 'BF16'
        self.pointer = pointer if pointer is not None else Tensor.serial
        Tensor.serial += 1024
        self.records = []
    def is_contiguous(self): return True
    def data_ptr(self): return self.pointer
    def untyped_storage(self): return self
    def view(self, *shape): return Tensor(shape, device=self.device, pointer=self.pointer)
    def record_stream(self, stream): self.records.append(stream)


DEVICE = Device()
class Stream:
    @property
    def npu_stream(self):
        raise AssertionError('raw stream pointer getter may drain the queue')
    def __hash__(self):
        raise AssertionError('stream hash may read the raw pointer')


STREAM = Stream()


def args():
    return tuple(Tensor(shape) for shape in ((1,585,15360),(5120,),(5120,),(1,585,1,128),(1,585,1,128)))


def config():
    values = {qk.CONFIG_KEY:True, 'lingbot_npu_native_bf16_qk':True}
    return SimpleNamespace(model_config=values, model_class_name='LingBotWorldCausalDMDPipeline',
        enforce_eager=True, parallel_config=SimpleNamespace(tensor_parallel_size=1,
            sequence_parallel_size=8, ulysses_degree=8, ring_degree=1))


class Ops:
    def __init__(self):
        self.creates, self.runs, self.closed, self.finalized = [], [], [], []
        self.state = dict.fromkeys(FIELDS, 0)
    def create(self, *values):
        self.creates.append(values)
        self.state.update(producer_tid=threading.get_native_id(), initialized=24,
            workspace_initial_bytes=1898213376, workspace_resident_bytes=1898213376)
        return 7
    def run(self, handle, *values):
        self.runs.append((handle, values))
        owner = Tensor((8986112,)); send = Tensor((8,585,3,1,5,128), pointer=owner.pointer+512)
        self.last_outputs = (send, owner)
        return send, owner
    def stats(self, handle): return [self.state[key] for key in FIELDS]
    def close(self, handle): self.closed.append(handle); self.state['native_closed'] = 1
    def finalize_close(self, handle):
        self.finalized.append(handle)
        self.state.update(closed=1, workspace_resident_bytes=0, retained_call_slots=0, retired_count=0)


def provider():
    ops = Ops()
    torch = SimpleNamespace(bfloat16='BF16', is_grad_enabled=lambda:False,
        is_inference_mode_enabled=lambda:True, are_deterministic_algorithms_enabled=lambda:False,
        npu=SimpleNamespace(current_device=lambda:0, current_stream=lambda device:STREAM,
            default_stream=lambda device:STREAM, is_current_stream_capturing=lambda:False))
    backend = SimpleNamespace(npu=SimpleNamespace(is_jit_compile_false=lambda:True),
        _C=SimpleNamespace(_npu_getOption=lambda name:'enable'), get_npu_format=lambda value:2)
    reports = []
    def describe(value):
        shape = value.shape; strides=[]; size=1
        for dim in reversed(shape): strides.insert(0,size); size *= dim
        return (2,2,shape,tuple(strides),shape,shape,tuple(strides),0,size*2,value.pointer,value.pointer)
    base = SimpleNamespace(torch=torch, backend=backend, describe=describe, poisoned=False,
        fatal=SimpleNamespace(report_failure=lambda *values:reports.append(values)))
    guard_calls=[]
    p=qk.PoolProvider(base,ops,qk.LeaseGuard(lambda *v:guard_calls.append(v)),
                      dict(statistics_fields=list(FIELDS)),qk.policy())
    p.reports,p.guard_calls=reports,guard_calls
    return p


def route(p, arguments=None):
    arguments = arguments or args()
    shared = native.SharedCoefficients(cosine=object(), sine=object(),
        expanded_cosine=arguments[3], expanded_sine=arguments[4], generation=1)
    module=SimpleNamespace(ulysses_world_size=8,num_local_heads=40,head_dim=128,
        norm_q=SimpleNamespace(eps=1e-6,weight=arguments[1]),
        norm_k=SimpleNamespace(eps=1e-6,weight=arguments[2]),
        _lingbot_native_bf16_qk_coefficients=shared)
    r=qk.configure_operator_module(module,config());r._provider=p
    return r,arguments,shared


class PoolAdapterContracts(unittest.TestCase):
    def setUp(self):
        self.previous = qk._BOUND
        self.quarantine_length = len(qk._QUARANTINE)
    def tearDown(self):
        qk._BOUND=self.previous
        del qk._QUARANTINE[self.quarantine_length:]

    def test_import_has_no_torch_and_no_assert_or_combined_compute(self):
        self.assertNotIn('torch',sys.modules)
        tree=ast.parse(Path(qk.__file__).read_text())
        self.assertFalse(any(isinstance(n,ast.Assert) for n in ast.walk(tree)))
        self.assertFalse(any(isinstance(n,ast.Attribute) and n.attr in ('try_run','synchronize') for n in ast.walk(tree)))
        self.assertFalse(any(isinstance(n,ast.ImportFrom) and n.module and 'lingbot_qk_combined' in n.module for n in ast.walk(tree)))

    def test_model_default_off_and_enabled_rejected_without_loading(self):
        cfg=config();cfg.model_config[qk.CONFIG_KEY]=False
        fake=SimpleNamespace(get_current_diffusion_config_or_none=lambda:cfg)
        with patch.dict(sys.modules,{'vllm_omni.diffusion.config':fake}),patch.object(qk,'load',side_effect=AssertionError('no load')):
            self.assertIsNone(qk.configure_module(object()))
            cfg.model_config[qk.CONFIG_KEY]=True
            with self.assertRaisesRegex(RuntimeError,'model construction is not admitted'):
                qk.configure_module(object())

    def test_operator_config_mutual_exclusion_and_topology(self):
        for key in qk._OTHERS:
            cfg=config();cfg.model_config[key]=True
            with self.assertRaisesRegex(RuntimeError,'mutually exclusive'):
                qk.configure_operator_module(object(),cfg)
        cfg=config();cfg.parallel_config.tensor_parallel_size=2
        with self.assertRaisesRegex(RuntimeError,'SP8'):
            qk.configure_operator_module(object(),cfg)
        with self.assertRaisesRegex(RuntimeError,'explicit standalone'):
            qk.PoolRoute(object(),config(),{})

    def test_snapshot_miss_has_no_get_expansion_pool_or_submission(self):
        p=provider();r,a,shared=route(p)
        with patch.object(shared,'get',side_effect=AssertionError('must not expand')):
            self.assertIsNone(r.prepare(a[0],(object(),object()),1))
        self.assertEqual(shared.generation,1);self.assertEqual(p.pools,{})
        self.assertEqual(p.ops.creates,[]);self.assertEqual(p.ops.runs,[])

    def test_ordinary_cpu_dtype_stream_and_grad_refuse_before_create(self):
        for mode in ('cpu','dtype','stream','grad','format'):
            p=provider();r,a,shared=route(p)
            if mode=='cpu': a[0].device=SimpleNamespace(type='cpu',index=0)
            if mode=='dtype': a[0].dtype='FP32'
            if mode=='stream': p.torch.npu.default_stream=lambda device:object()
            if mode=='grad': p.torch.is_grad_enabled=lambda:True
            if mode=='format': p.backend.get_npu_format=lambda value:29
            self.assertIsNone(r.prepare(a[0],(shared.cosine,shared.sine),1),mode)
            self.assertFalse(p.base.poisoned);self.assertEqual(p.ops.creates,[])
            self.assertEqual(shared.generation,1)

    def test_forty_routes_share_one_pool_not_input_addresses(self):
        p=provider()
        for _ in range(40):
            r,a,s=route(p)
            lease=r.prepare(a[0],(s.cosine,s.sine),1)
            self.assertIsInstance(lease,qk.PoolLease)
        self.assertEqual(len(p.ops.creates),1)
        self.assertEqual(p.ops.creates[0][-2:],(24,32))
        self.assertEqual(qk.POOL_POLICY,dict(max_pools=1,plan_capacity=24,
            retirement_capacity=32,workspace_bytes_per_plan=79092224,
            workspace_budget_bytes=24*79092224))
        self.assertEqual(len(p.ops.runs),40)
        self.assertEqual(len(p.pools),1)
        self.assertEqual(len(p.cold_records),1)
        self.assertGreaterEqual(p.cold_records[0]["create_wall_ns"],0)

    def test_different_descriptor_is_terminal_no_growth_or_fallback(self):
        p=provider();p.ensure_pool(args())
        old=p.base.describe
        p.base.describe=lambda t:(*old(t)[:8],old(t)[8]+32,*old(t)[9:])
        with self.assertRaisesRegex(RuntimeError,'bounded descriptor pool'):
            p.ensure_pool(args())
        self.assertTrue(p.base.poisoned);self.assertEqual(len(p.ops.creates),1)
        self.assertEqual(len(p.pools),1)

    def test_current_device_metadata_and_cpp_failure_poison_before_more_work(self):
        for mode in ('device','descriptor','native'):
            p=provider()
            if mode=='device': p.torch.npu.current_device=lambda:1
            if mode=='descriptor': p.base.describe=lambda x:(0,)
            if mode=='native': p.ops.create=lambda *a:(_ for _ in ()).throw(RuntimeError('bad physical extent'))
            with self.assertRaises(RuntimeError): p.ensure_pool(args())
            self.assertTrue(p.base.poisoned);self.assertTrue(p.reports)
            with self.assertRaisesRegex(RuntimeError,'poisoned'): p.ensure_pool(args())

    def test_postsubmit_python_failure_retains_returned_send_and_owner(self):
        p=provider();r,a,s=route(p)
        with patch.object(qk,'PoolLease',side_effect=RuntimeError('lease constructor failed')):
            with self.assertRaisesRegex(RuntimeError,'constructor failed'):
                r.prepare(a[0],(s.cosine,s.sine),1)
        keep=qk._QUARANTINE[-1][1]
        self.assertIn(p.ops.last_outputs[0],keep);self.assertIn(p.ops.last_outputs[1],keep)
        self.assertTrue(p.base.poisoned);self.assertEqual(len(p.ops.runs),1)

    def test_original_collective_arguments_guard_order_and_result_record(self):
        p=provider();r,a,s=route(p);qk._BOUND=p
        lease=r.prepare(a[0],(s.cosine,s.sine),1); group=object(); calls=[]
        output=Tensor((1,4680,3,5,128))
        def collective(metadata,**kw):
            calls.append((metadata,kw));self.assertFalse(lease.used)
            qk.validate_padded_send(metadata,kw['_prepared_send'],kw['_qk_padded_owner'],8)
            self.assertTrue(lease.used)
            return output
        self.assertIs(lease.exchange(collective,group),output)
        self.assertEqual(len(calls),1);self.assertEqual(len(p.guard_calls),1)
        self.assertEqual(calls[0][1],dict(scatter_idx=3,gather_idx=1,group=group,
            _prepared_send=lease.send,_qk_padded_owner=lease))
        self.assertEqual(output.records,[STREAM]);self.assertEqual(lease.send.records,[])
        with self.assertRaisesRegex(RuntimeError,'lease reused'):lease.exchange(collective,group)
        self.assertTrue(p.base.poisoned);self.assertEqual(len(calls),1)

    def test_collective_exception_retains_lease_without_retry(self):
        p=provider();r,a,s=route(p);lease=r.prepare(a[0],(s.cosine,s.sine),1);calls=[]
        def collective(*a,**kw):calls.append(kw);raise RuntimeError('HCCL started failure')
        with self.assertRaisesRegex(RuntimeError,'HCCL started'):lease.exchange(collective,object())
        self.assertEqual(len(calls),1);self.assertFalse(lease.entering);self.assertTrue(p.base.poisoned)
        self.assertIs(qk._QUARANTINE[-1][1][0],lease)

    def test_guard_has_no_compute_entry_and_wrong_binding_fails(self):
        p=provider();self.assertFalse(hasattr(p.guard,'try_run'))
        with self.assertRaisesRegex(RuntimeError,'registered pool lease'):
            qk.validate_padded_send(object(),object(),object(),8)

    def test_close_two_phases_preserve_owners_until_finalize(self):
        p=provider();p.ensure_pool(args())
        self.assertEqual(p.close_all_drained(),(7,))
        self.assertEqual(p.snapshot(7)['workspace_resident_bytes'],1898213376)
        self.assertEqual(p.snapshot(7)['closed'],0)
        with self.assertRaisesRegex(RuntimeError,'closing/closed'):p.check_policy()
        p.finalize_all_drained()
        self.assertEqual(p.ops.closed,[7]);self.assertEqual(p.ops.finalized,[7])
        self.assertTrue(p._closed);self.assertEqual(p.snapshot(7)['workspace_resident_bytes'],0)

    def test_early_finalize_is_terminal_no_cleanup_retry(self):
        p=provider();p.ensure_pool(args())
        with self.assertRaisesRegex(RuntimeError,'successful drained close'):p.finalize_all_drained()
        self.assertTrue(p.base.poisoned);self.assertEqual(p.ops.closed,[]);self.assertEqual(p.ops.finalized,[])

    def test_relocated_base_source_uses_bytes_and_exact_ancestor(self):
        def item(name, digest=None):return dict(path='/p/'+name,bytes=1,sha256=digest or name)
        adapter_pin=item('adapter'); source_old=item('oldsource','same'); source_new=item('newsource','same')
        parent_pin=item('parent'); lib=item('base-lib'); kernels=item('kernels')
        old_base=dict(schema='lingbot_recovery_build_v1',source=source_old,library=lib,kernels=kernels)
        current_base=dict(old_base,source=source_new,registration_passed=True,
                          historical_admission_inherited=False,compilation_ancestor=parent_pin)
        ancestor=dict(schema='lingbot_qk_repeatable_event_pool_cpu_build_v2',registration=True,
            v1_qualification_inherited=False,source=item('pool-source'),library=item('pool-lib'),
            namespace=qk.NAMESPACE,dependencies=[item('sdk')],statistics_fields=list(FIELDS),
            ownership_policy='pool-v2',provider_manifest=parent_pin)
        old_guard=dict(schema='lingbot_qk_combined_build_v1',registration_passed=True,
            source=item('guard-source'),library=item('guard-lib'),dependencies=[kernels],abi_headers=[],
            provider_manifest=parent_pin)
        guard=dict(old_guard,compilation_ancestor=item('guard-ancestor'),namespace='lingbot_qk_combined_v1',
                   allowed_operations=['validate_lease'],resolution_only_operations=['dependencies'])
        binding=dict(ancestor,schema=qk.SCHEMA,binding_only=True,recompiled=False,
            historical_admission_inherited=False,device_admitted=False,model_admitted=False,
            performance_admitted=False,pool_policy=dict(qk.POOL_POLICY),compilation_ancestor=item('ancestor'),
            guard=guard,provider_manifest=item('base-current'),freeze_manifest=item('freeze'),
            binder=item('binder'),abi_headers=[],runtime_sources=[adapter_pin],test_sources=[])
        docs={'/p/parent':old_base,'/p/base-current':current_base,'/p/ancestor':ancestor,
              '/p/guard-ancestor':old_guard,'/p/freeze':{}}
        allpins=[v for d in [binding,guard,ancestor] for v in d.values() if type(v)is dict and 'path'in v]
        allpins += [*binding['dependencies'],*guard['dependencies'],adapter_pin]
        pins={v['path']:v for v in allpins};pins[qk.__file__]=adapter_pin
        with patch.object(qk,'read_pinned',side_effect=lambda p:docs[p['path']]),patch.object(qk,'pin',side_effect=lambda p:pins[str(p)]):
            qk.validate_binding(binding)
            current_base['source']=item('newsource','changed')
            with self.assertRaisesRegex(RuntimeError,'source bytes'):qk.validate_binding(binding)
            current_base['source']=source_new;current_base['compilation_ancestor']=item('wrong-parent')
            with self.assertRaisesRegex(RuntimeError,'compilation ancestry'):qk.validate_binding(binding)


if __name__=='__main__':unittest.main()
