"""Default-off experiment route. No framework import before explicit enable.

This is a source proposal, not a model admission. Startup requires independent
new192 numeric/dispatch, three actual leaf screens and padded-HCCL cold proof.
The new single-host normal queued adapter and original collective remain the only
execution paths. A started failure poisons the worker and retains all owners.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import threading

ENABLE = 'ROOT_LINGBOT_QK_SINGLE_HOST_INTEGRATION_V1'
CONFIG_KEY = 'lingbot_npu_qk_single_host_admission_v1'
MANIFEST = 'ROOT_LINGBOT_QK_SINGLE_HOST_INTEGRATION_MANIFEST'
MANIFEST_SHA = MANIFEST + '_SHA256'
ADAPTER_ENABLE = 'LINGBOT_QK_NATIVE_RMS_ROPE_SEND_V1_ENABLE'
SMOKE_ENABLE = 'ROOT_LINGBOT_QK_SINGLE_HOST_PADDED_HCCL_SMOKE_V1'
HOST_ENABLE = 'ROOT_LINGBOT_QK_SINGLE_HOST_ENABLE'
HOST_NAMESPACE = 'lingbot_qk_single_host_admission_v1'
HOST_ADAPTER = Path('/mnt/data/zhoutaichang/project/lingbot-world/profiling/runs/20261008-h200-gap-optimization/QK_native_rms_rope_send_v1/single_host_framework_policy_v2/adapter_v2.py')
HOST_ADAPTER_SHA = '44fb26c182aeeeadf4f76a2b7f102640f9230b158520dcd008046677a6dc1a54'
COLLECTOR = HOST_ADAPTER.parent.parent/'operator192_single_host_v2/review_three_round_timing_CPU_v2.py'
COLLECTOR_SHA = 'd5be5d8b2312f786ad42f78615efe973a64034033c0e7f21713ab81969c78c3d'
CONTROLLER = COLLECTOR.parent/'screen192_v2.py'
CONTROLLER_SHA = '73bb9d9a6ac1eb8def859f123617b7683d990be5f7939a323cc8aac8e367f0a0'
SHAPE = (8, 585, 3, 1, 5, 128)
STRIDE = (1123200, 1920, 640, 640, 128, 1)
FIELDS = ('npu_format', 'origin_format', 'base_sizes', 'base_strides', 'physical_sizes',
          'shape', 'stride', 'storage_offset', 'storage_bytes', 'storage_data_ptr', 'data_ptr')
_LOCK = threading.RLock()
_BOUND = None
_CONSTRUCTION_BINDING = None
_SMOKE_BOUND = None
_REGISTERED_ROUTES = set()
_QUARANTINE = []


def need(ok, message):
    if not ok:
        raise RuntimeError(message)


def pin(path):
    path = Path(path).resolve(strict=True)
    raw = path.read_bytes()
    return dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def read_pinned(item):
    need(type(item) is dict and set(item) == {'path', 'bytes', 'sha256'}
         and Path(item['path']).is_absolute() and pin(item['path']) == item,
         'Required actual proof/dependency pin missing or changed')
    return json.loads(Path(item['path']).read_text())


def load_host_adapter(manifest):
    item = manifest['adapter']
    need(item == pin(HOST_ADAPTER) and item['sha256'] == HOST_ADAPTER_SHA,
         'Exact immutable single-host adapter required')
    spec = importlib.util.spec_from_file_location('_qk_model_single_host_adapter_v1', item['path'])
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    return adapter


def validate_host_binding(manifest):
    need(manifest.get('host_namespace') == HOST_NAMESPACE,
         'New single-host namespace required; old K3 host cannot be substituted')
    runtime_pin = manifest['host_runtime_manifest']
    read_pinned(runtime_pin)
    adapter = load_host_adapter(manifest)
    runtime = adapter.validate_manifest(runtime_pin['path'], runtime_pin['sha256'])
    need(runtime['new_host_build'] == manifest['new_host_build']
         and runtime['namespace'] == manifest['host_namespace']
         and runtime['kernel_parent'] == manifest['build']
         and runtime['adapter'] == manifest['adapter']
         and runtime['fatal_source'] == manifest['fatal_source'],
         'Model/leaf host build, manifest, parent package or fatal provenance differs')
    validate_framework_binding(runtime, manifest)
    return runtime


def validate_framework_binding(runtime, manifest):
    need(runtime.get('schema') == 'lingbot_QK_single_host_framework_policy_runtime_manifest_v2'
         and runtime.get('new_CPP_or_DSO_compiled') is False
         and runtime.get('old_numeric_or_speed_PASS_inherited') is False,
         'Distinct H2 policy must preserve the actual H1 compilation without inheriting PASS')
    fields = ('parent_runtime_manifest', 'parent_adapter', 'framework_policy',
              'framework_import_probe', 'expected_search_path')
    need(all(manifest.get(key) == runtime.get(key) and runtime.get(key) is not None for key in fields),
         'Startup and actual H2 framework policy/probe ancestry differ')
    policy = read_pinned(runtime['framework_policy'])
    need(policy['ordered_entries'][0] == manifest['build']['vendor_path']
         and runtime['expected_search_path'] == ':'.join(policy['ordered_entries'])
         and policy['expected_search_path'] == runtime['expected_search_path'],
         'Exact K3-first lexical framework OPP search path required')


def load_exact_fatal(manifest, derived_fatal):
    """Startup only: preserve exact host source and actual persistent channel.

    The generated model retains its original fatal module. The host adapter also
    requires the immutable original source PATH. Load that unmodified file into
    an independent module; never relabel __file__ or replace the model module.
    Both reporters retain owners and notify the same root-owned channel. Their
    counters/quarantines are separate; original stderr fallback is preserved.
    """
    runtime = read_pinned(manifest['host_runtime_manifest'])
    source = runtime['fatal_source']
    need(source == manifest['fatal_source'] and pin(source['path']) == source,
         'Exact host fatal source provenance differs')
    current = pin(derived_fatal.__file__)
    need(all(current[key] == source[key] for key in ('bytes', 'sha256')),
         'Generated model fatal source bytes differ from original host source')
    fields = ('_DIRECTORY', '_NONCE', '_MODE', '_FAULT_FLAG', 'ENABLED')
    expected = (os.environ.get('ROOT_LINGBOT_SP8_FATAL_DIR', ''),
        os.environ.get('ROOT_LINGBOT_SP8_FATAL_NONCE', ''),
        os.environ.get('ROOT_LINGBOT_SP8_EXPERIMENT_MODE', ''),
        os.environ.get('ROOT_LINGBOT_SP8_FAULT_ONCE', '0'), True)
    need(tuple(getattr(derived_fatal, key) for key in fields) == expected
         and derived_fatal.ensure_protocol() is True,
         'Generated model and current root fatal channel differ')
    spec = importlib.util.spec_from_file_location('_qk_exact_host_fatal_v2', source['path'])
    need(spec is not None and spec.loader is not None, 'Exact fatal source loader unavailable')
    fatal = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fatal)
    need(pin(fatal.__file__) == source
         and tuple(getattr(fatal, key) for key in fields) == expected
         and fatal.ensure_protocol() is True,
         'Exact original host source/channel binding failed')
    return fatal


def same_host(record, manifest):
    return all(record.get(key) == manifest[key]
               for key in ('new_host_build', 'host_runtime_manifest', 'host_namespace'))


def validate_leaf_admission(manifest):
    """No old cold/after-norm references, worker flags or source-only pass."""
    validate_host_binding(manifest)
    numeric = read_pinned(manifest['operator_correctness_proof'])
    need(numeric.get('schema') == 'lingbot_QK_single_host_framework_policy_actual192_correctness_proof_v2'
         and numeric.get('passed') is True and numeric.get('case_count') == 192
         and all(numeric.get(k) is True for k in ('full192_numeric_proven',
             'source_fullarenas_and_canaries_verified', 'production_empty_allocator_verified',
             'actual_backend_dispatch_independently_proven', 'all8_exit0_joined_procabsent'))
         and numeric.get('build') == manifest['build'] and numeric.get('enable_cache') is False
         and same_host(numeric, manifest) and numeric.get('controller') == pin(CONTROLLER)
         and numeric['controller']['sha256'] == CONTROLLER_SHA
         and numeric.get('production_output_route') == 'empty8986112BF16_offset256_no_memset',
         'Actual new pre-RMS192 numeric/dispatch admission required')
    child_schemas = dict(correctness_complete='lingbot_QK_single_host_framework_policy_actual192_complete_v2',
        independent_numeric_review='lingbot_QK_single_host_framework_policy_actual192_numeric_CPU_review_v2',
        independent_dispatch_review='lingbot_QK_single_host_framework_policy_actual192_dispatch_CPU_review_v2')
    for key, schema in child_schemas.items():
        proof = read_pinned(numeric[key])
        need(proof.get('schema') == schema and proof.get('passed') is True
             and proof.get('case_count') == 192 and same_host(proof, manifest),
             'Independent NEW actual192 proof/host binding is incomplete')
        if key == 'independent_numeric_review':
            need(proof.get('FP32_reference_rebuilt') is True
                 and proof.get('source_bound_oracle_verified') is True,
                 'New source-bound fullwidth FP32 reference proof required')
        if key != 'correctness_complete':
            need(proof.get('correctness_complete') == numeric['correctness_complete'],
                 'Independent proof is from another new192 completion')
    screen = read_pinned(manifest['operator_three_round_review'])
    # The frozen independent collector distinguishes integrity from speed.
    need(screen.get('schema') == 'lingbot_QK_single_host_framework_policy_actual192_three_round_independent_timing_CPU_review_v2'
         and screen.get('passed') is True and screen.get('evidence_integrity_verified') is True
         and same_host(screen, manifest)
         and screen.get('correctness_proof') == manifest['operator_correctness_proof']
         and screen.get('three_round_operator_performance_passed') is True
         and screen.get('three_round_gate_rejected') is False
         and screen.get('actual_cases') == 576 and screen.get('actual_fresh_workers') == 24
         and screen.get('raw_wall_samples') == screen.get('raw_event_samples') == 34560
         and screen.get('raw_samples') == 69120 and len(screen.get('rounds', [])) == 3,
         'Three new actual leaf screens must pass integrity AND performance')
    reviewer = screen.get('reviewer')
    need(type(reviewer) is dict and set(reviewer) == {'path', 'bytes', 'sha256'}
         and reviewer['sha256'] == COLLECTOR_SHA and Path(reviewer['path']) == COLLECTOR
         and pin(reviewer['path']) == reviewer, 'Frozen NEW actual three-round collector source required')
    identities, nonces, outputs = [], [], []
    for number, row in enumerate(screen['rounds'], 1):
        summary = row.get('summary', {})
        wall = summary.get('wall', {})
        need(row.get('round') == number and row.get('correctness_proof') == manifest['operator_correctness_proof']
             and row.get('build') == manifest['build'] and same_host(row, manifest)
             and summary.get('case_count') == 192
             and all(summary.get(k) is True for k in ('operator_performance_passed',
                 'wall_mean_gain_passed', 'wall_worst_case_passed')),
             'Each round must bind the same new correctness/build and original wall gates')
        need(all(type(wall.get(k)) in (int, float) and math.isfinite(wall[k])
                 for k in ('native_ms', 'candidate_ms', 'ratio', 'gain_fraction', 'worst_case_ratio'))
             and wall['native_ms'] > 0 and wall['candidate_ms'] > 0
             and math.isclose(wall['ratio'], wall['candidate_ms']/wall['native_ms'], rel_tol=1e-12, abs_tol=1e-12)
             and math.isclose(wall['gain_fraction'], 1-wall['ratio'], rel_tol=1e-12, abs_tol=1e-12)
             and wall['ratio'] <= .90 and wall['worst_case_ratio'] <= 1.05,
             'Original >=10% mean gain and <=1.05 worst-case wall gate required')
        records = {key: read_pinned(row[key]) for key in ('root', 'prepared', 'complete', 'cleanup',
                                                        'root_terminal', 'root_live_identity')}
        complete = records['complete']
        need(complete.get('passed') is True and complete.get('complete') is True
             and complete.get('mode') == 'timing' and complete.get('round') == number
             and complete.get('case_count') == 192 and complete.get('worker_count') == 8
             and complete.get('warmup') == 20 and complete.get('event_samples') == complete.get('wall_samples') == 30
             and complete.get('all8_exit0_joined_procabsent') is True
             and complete.get('prepared') == row['prepared'] and complete.get('cleanup') == row['cleanup']
             and complete.get('correctness_proof') == manifest['operator_correctness_proof']
             and complete.get('build') == manifest['build'] and same_host(complete, manifest), 'Actual per-round 20/30/30 completion and fresh8 closure required')
        need(row.get('arm_order') == (['candidate', 'native'] if number == 2 else ['native', 'candidate'])
             and row.get('run_nonce') == records['prepared'].get('run_nonce') == complete.get('run_nonce')
             and row.get('output') == records['prepared'].get('output')
             and records['root'].get('exit_code') == 0 and records['root'].get('all8_reclaimed') is True,
             'Actual counterbalanced fresh-round identity required')
        group = row.get('identities', [])
        need(len(group) == 8, 'Actual eight worker identities required per round')
        for worker in group:
            ticks = worker.get('process_start_ticks', worker.get('start_ticks'))
            need(type(worker.get('pid')) is int and worker['pid'] > 0 and type(ticks) is int
                 and ticks > 0 and isinstance(worker.get('boot_id'), str) and bool(worker['boot_id']),
                 'Actual PID/startticks/boot identity required')
            identities.append((worker['pid'], ticks, worker['boot_id']))
        nonces.append(row['run_nonce']);outputs.append(row['output'])
    need(len(set(identities)) == 24 and len(set(nonces)) == len(set(outputs)) == 3,
         'Three distinct actual fresh8 groups required')


def validate_admission(manifest):
    validate_leaf_admission(manifest)
    cold = read_pinned(manifest['padded_hccl_cold_review'])
    need(cold.get('schema') == 'lingbot_QK_single_host_padded_send_HCCL_actual_cold_CPU_review_v1'
         and cold.get('passed') is True and cold.get('ranks') == 8
         and cold.get('original_collective_bytes_equal') is True
         and cold.get('actual_nonzero_offset256_proven') is True
         and cold.get('source_fullarenas_and_canaries_verified') is True
         and cold.get('all8_exit0_joined_procabsent') is True
         and cold.get('build') == manifest['build'] and same_host(cold, manifest)
         and cold.get('operator_correctness_proof') == manifest['operator_correctness_proof']
         and cold.get('operator_three_round_review') == manifest['operator_three_round_review']
         and cold.get('communication_source_sha256') == manifest['patched_source_sha256']['comm'],
         'Actual original HCCL nonzero-offset cold proof is still required')
    for key in ('prepared', 'complete', 'root_execution', 'cleanup'):
        read_pinned(cold[key])


def output_descriptor(value, owner, describe):
    fields = describe(value)
    need(len(fields) == 11, 'Exact read-only descriptor ABI required')
    record = dict(zip(FIELDS, fields))
    need(record['npu_format'] == record['origin_format'] == 2
         and tuple(record['base_sizes']) == tuple(record['physical_sizes']) == (8986112,)
         and tuple(record['base_strides']) == (1,) and tuple(record['shape']) == SHAPE
         and tuple(record['stride']) == STRIDE and record['storage_offset'] == 256
         and record['storage_bytes'] == 17972224
         and record['storage_data_ptr'] > 0
         and record['data_ptr'] == record['storage_data_ptr'] + 512,
         'K3 padded send physical owner/extent differs')
    need(tuple(owner.shape) == (8986112,) and tuple(owner.stride()) == (1,)
         and owner.storage_offset() == 0 and value.is_contiguous()
         and str(value.dtype) == str(owner.dtype) == 'torch.bfloat16'
         and value.device == owner.device and value.device.type == 'npu'
         and value.untyped_storage().data_ptr() == owner.untyped_storage().data_ptr() == record['storage_data_ptr']
         and value.untyped_storage().nbytes() == owner.untyped_storage().nbytes() == 17972224
         and value.data_ptr() == record['data_ptr'] and tuple(value.shape) == SHAPE
         and tuple(value.stride()) == STRIDE and value.storage_offset() == 256,
         'K3 padded view disagrees with its actual fresh allocation')
    return record


class _Send:
    def __init__(self, route, metadata, owner, send, stream, owners):
        self.route, self.metadata, self.owner, self.send = route, metadata, owner, send
        self.stream, self.owners = stream, owners
        self.description = output_descriptor(send, owner, route.describe)
        self.used = False

    def exchange(self, collective, group):
        """Catch validation, HCCL submission and original receive-path failures."""
        try:
            output = collective(self.metadata, scatter_idx=3, gather_idx=1, group=group,
                                _prepared_send=self.send, _qk_padded_owner=self)
            output.record_stream(self.stream)
            return output
        except BaseException as error:
            self.route.fail(error, (self, collective, group))
            raise


def validate_padded_send(metadata, send, lease, world_size):
    """Only the independently enabled K3 route may bypass the old offset0 ABI."""
    need(type(lease) is _Send and type(lease.route) is _Route
         and lease.route in _REGISTERED_ROUTES and not lease.used
         and not lease.route.poisoned and metadata is lease.metadata and send is lease.send
         and world_size == 8, 'Invalid/reused K3 prepared-send lease')
    route = lease.route
    need(route.torch.npu.current_stream(send.device) == lease.stream
         == route.torch.npu.default_stream(send.device), 'K3/HCCL current stream changed')
    need(tuple(metadata.shape) == (1, 585, 3, 40, 128)
         and metadata.dtype == send.dtype and metadata.device == send.device,
         'K3 shape-only metadata differs')
    need(output_descriptor(send, lease.owner, route.describe) == lease.description,
         'K3 padded owner changed before original collective')
    for tensor in (*lease.owners, metadata, send):
        if isinstance(tensor, route.torch.Tensor):
            tensor.record_stream(lease.stream)
    lease.used = True


class _Route:
    def __init__(self, torch, backend, adapter, operations, indices, describe, fatal, manifest, config, policy, kind):
        need(kind in ('model', 'hccl-smoke'), 'Explicit admitted route kind required')
        self.kind = kind
        self.torch, self.backend, self.adapter = torch, backend, adapter
        self.operations, self.indices, self.describe, self.fatal = operations, indices, describe, fatal
        self.manifest, self.config, self.policy = manifest, config, policy
        self.poisoned = False

    def fail(self, error, owners):
        self.poisoned = True
        retained = (self, owners, error, sys.exc_info())
        with _LOCK:
            _QUARANTINE.append(retained)
        try:
            self.fatal.report_failure('qk_single_host_integration_v1', repr(error), retained)
        except BaseException:
            pass

    def prepare(self, module, raw, rotary, ring_size):
        keep = (module, raw, rotary, self, self.indices)
        try:
            need(not self.poisoned and (self.kind == 'hccl-smoke' or self.config.get(CONFIG_KEY) is True)
                 and all(os.environ.get(k) == v for k, v in self.policy.items()),
                 'QK integration policy/provider poisoned or changed; restart8 required')
            # Ordinary scope mismatches refuse before any send_out or collective.
            if (module.ulysses_world_size != 8 or ring_size != 1 or module.num_local_heads != 40
                    or module.head_dim != 128 or module.norm_q.eps != module.norm_k.eps
                    or module.norm_q.eps != 1e-6 or tuple(raw.shape) != (1, 585, 15360)
                    or not raw.is_contiguous()):
                return None
            if (raw.device.type != 'npu' or raw.dtype != self.torch.bfloat16
                    or self.torch.is_grad_enabled() or not self.torch.is_inference_mode_enabled()
                    or self.torch.npu.is_current_stream_capturing()
                    or self.torch.npu.current_stream(raw.device) != self.torch.npu.default_stream(raw.device)):
                return None
            # Use the actual model's fresh shared coefficient snapshot; never
            # rebuild/reorder/retag it or change the native BF16 policy.
            cosine, sine = module._lingbot_native_bf16_qk_coefficients.get(*rotary)
            owner = self.backend.empty_with_format((8986112,), dtype=self.torch.bfloat16,
                                                   device=raw.device, acl_format=2)
            send = self.torch.as_strided(owner, SHAPE, STRIDE, 256)
            metadata = raw.view(1, 585, 3, 40, 128)
            keep = (*keep, module.norm_q.weight, module.norm_k.weight, cosine, sine, owner, send, metadata)
            stream = self.torch.npu.current_stream(raw.device)
            result = self.adapter.try_send(raw, module.norm_q.weight, module.norm_k.weight,
                cosine, sine, send, indices=self.indices, epsilon=1e-6, heads=40,
                sp_size=8, tp_size=1, ring_size=ring_size, enable_cache=False)
            if result is None:
                return None
            need(result is send, 'Started K3 adapter returned a different owner')
            return _Send(self, metadata, owner, send, stream, keep)
        except BaseException as error:
            self.fail(error, keep)
            raise


_CONSTRUCTION_POLICY_KEYS = (ENABLE, MANIFEST, MANIFEST_SHA, ADAPTER_ENABLE, HOST_ENABLE, SMOKE_ENABLE,
    'LINGBOT_NPU_SLOT_GRANTED', 'LINGBOT_CPU_SLOT_GRANTED', 'LINGBOT_ROOT_NPU_SLOT_GRANTED',
    'ROOT_LINGBOT_SP8_FATAL_DIR', 'ROOT_LINGBOT_SP8_FATAL_NONCE',
    'ROOT_LINGBOT_SP8_EXPERIMENT_MODE', 'ASCEND_CUSTOM_OPP_PATH')


def bind_construction_config(module):
    """Retain the actual init-context configuration; never configure a runtime here."""
    global _CONSTRUCTION_BINDING
    need(os.environ.get(ENABLE, '0') == '1' and os.environ.get(SMOKE_ENABLE, '0') == '0',
         'Only explicitly enabled model construction may bind QK configuration')
    from vllm_omni.diffusion.config import get_current_diffusion_config_or_none
    config = get_current_diffusion_config_or_none()
    model_config = getattr(config, 'model_config', None)
    need(type(model_config) is dict and model_config.get(CONFIG_KEY) is True
         and getattr(config, 'model_class_name', None) == 'LingBotWorldCausalDMDPipeline',
         'Actual canonical construction context with explicit new model bool is required')
    policy = tuple((key, os.environ.get(key)) for key in _CONSTRUCTION_POLICY_KEYS)
    with _LOCK:
        if _CONSTRUCTION_BINDING is None:
            need(_BOUND is None, 'Cannot bind a construction config after route start')
            _CONSTRUCTION_BINDING = (config, model_config, policy)
        else:
            owner, original, original_policy = _CONSTRUCTION_BINDING
            need(config is owner and model_config is original and policy == original_policy,
                 'Repeated attention construction requires identical actual config owner and policy')
        previous = module.__dict__.get('_lingbot_qk_single_host_construction_config')
        need(previous is None or previous is _CONSTRUCTION_BINDING,
             'Module construction binding changed')
        module.__dict__['_lingbot_qk_single_host_construction_config'] = _CONSTRUCTION_BINDING
        return config


def _construction_model_config():
    need(_CONSTRUCTION_BINDING is not None, 'QK model construction binding is missing; no fallback')
    config, model_config, policy = _CONSTRUCTION_BINDING
    need(getattr(config, 'model_config', None) is model_config and type(model_config) is dict
         and model_config.get(CONFIG_KEY) is True
         and getattr(config, 'model_class_name', None) == 'LingBotWorldCausalDMDPipeline'
         and tuple((key, os.environ.get(key)) for key in _CONSTRUCTION_POLICY_KEYS) == policy,
         'Actual bound construction config or startup policy changed')
    return config, model_config


def selected_route():
    global _BOUND
    enabled = os.environ.get(ENABLE, '0')
    need(enabled in ('0', '1'), 'QK integration flag must be 0/1')
    need(os.environ.get(SMOKE_ENABLE, '0') == '0', 'Nonmodel HCCL smoke cannot enter the model route')
    if enabled == '0':
        need(_BOUND is None, 'Configured QK integration cannot be disabled mid-worker')
        return None
    with _LOCK:
        config, model_config = _construction_model_config()
        if _BOUND is not None:
            need(_BOUND.config is model_config, 'Configured route construction owner changed')
            return _BOUND
        need(isinstance(model_config, dict) and model_config.get(CONFIG_KEY) is True
             and getattr(config, 'model_class_name', None) == 'LingBotWorldCausalDMDPipeline',
             'Explicit new LingBot model bool required; old configs cannot opt in')
        policy = {k: os.environ.get(k) for k in (ENABLE, MANIFEST, MANIFEST_SHA, ADAPTER_ENABLE, HOST_ENABLE,
            'LINGBOT_NPU_SLOT_GRANTED', 'LINGBOT_CPU_SLOT_GRANTED', 'LINGBOT_ROOT_NPU_SLOT_GRANTED', 'ROOT_LINGBOT_SP8_FATAL_DIR',
            'ROOT_LINGBOT_SP8_FATAL_NONCE', 'ROOT_LINGBOT_SP8_EXPERIMENT_MODE', 'ASCEND_CUSTOM_OPP_PATH')}
        need(policy[ADAPTER_ENABLE] == policy[HOST_ENABLE] == policy['LINGBOT_CPU_SLOT_GRANTED'] == policy['LINGBOT_NPU_SLOT_GRANTED']
             == policy['LINGBOT_ROOT_NPU_SLOT_GRANTED'] == '1'
             and policy['ROOT_LINGBOT_SP8_EXPERIMENT_MODE'] in ('correctness', 'timing'),
             'Both original root grants and isolated mapped experiment required')
        manifest = read_pinned(dict(path=policy[MANIFEST], bytes=Path(policy[MANIFEST]).stat().st_size,
                                    sha256=policy[MANIFEST_SHA]))
        need(manifest.get('schema') == 'lingbot_QK_single_host_model_integration_actual_startup_v4'
             and manifest.get('source_contract_verified') is True and manifest.get('model_admitted') is False
             and manifest.get('template_only') is False and manifest.get('runnable') is True,
             'Actual experiment startup manifest required')
        validate_admission(manifest)
        _BOUND = _configure_runtime(manifest, model_config, policy, 'model')
        return _BOUND

def _configure_runtime(manifest, model_config, policy, kind):
    need(pin(__file__)['sha256'] == manifest['helper']['sha256'], 'Actual helper differs')
    root = Path(__file__).resolve().parents[3]
    for name, relative in manifest['patched_source_relative'].items():
        need(pin(root/relative)['sha256'] == manifest['patched_source_sha256'][name],
             'Actual patched model/communication source differs')
    need(os.environ.get('ASCEND_CUSTOM_OPP_PATH', '') == manifest['expected_search_path'],
         'Exact validated K3-first framework OPP path must be seeded before canonical import')
    import torch
    import torch_npu
    need(torch_npu.__version__ == '2.10.0.post4' and torch.npu.is_initialized()
         and torch_npu._C._npu_get_soc_version() == 251
         and torch_npu._C._npu_getOption('ALLOW_INTERNAL_FORMAT') in (b'enable', 'enable')
         and torch_npu.npu.is_jit_compile_false() is True, 'Actual unchanged supported runtime required')
    from vllm_omni.diffusion.layers import lingbot_sp8_fatal as derived_fatal
    fatal = load_exact_fatal(manifest, derived_fatal)
    for item in manifest['build'].values():
        if type(item) is dict and set(item) == {'path', 'bytes', 'sha256'}:
            need(pin(item['path']) == item, 'Pinned K3 actual package/provider changed')
    need(pin(manifest['descriptor_library']['path']) == manifest['descriptor_library'],
         'Pinned read-only eleven-field descriptor provider changed')
    torch.ops.load_library(manifest['descriptor_library']['path'])
    new_build = read_pinned(manifest['new_host_build'])
    torch.ops.load_library(new_build['library']['path'])
    operations = getattr(torch.ops, HOST_NAMESPACE)
    provider = manifest['build']['provider']['path']
    need(operations.bind(provider, provider, provider, manifest['build']['cache_ELF']['path'],
         manifest['cache_function_owner']) == provider, 'Exact normal typed-provider binding failed')
    indices = operations.initialize_indices(torch.npu.current_device())
    adapter = load_host_adapter(manifest)
    channel_token = adapter.configure(manifest['host_runtime_manifest']['path'],
        manifest['host_runtime_manifest']['sha256'], operations=operations, fatal=fatal)
    route = _Route(torch, torch_npu, adapter, operations, indices,
                    torch.ops.lingbot_storage_descriptor.describe, fatal, manifest, model_config, policy, kind)
    route.channel_token = channel_token
    route.derived_fatal_owner = derived_fatal
    _REGISTERED_ROUTES.add(route)
    return route


def configure_hccl_smoke(manifest_pin):
    """Future explicit root-only nonmodel cold entry; never a model bypass.

    The future fresh8 smoke controller imports the SAME materialized helper,
    binds real fixture/gamma/coefficient owners, then calls route.prepare and
    lease.exchange with the original group. No HCCL proof is required to create
    its first proof; leaf three-round proof remains mandatory. This function
    launches nothing. The current source tool does not prepare that controller.
    """
    global _SMOKE_BOUND
    need(_BOUND is None and os.environ.get(ENABLE, '0') == '0'
         and os.environ.get(SMOKE_ENABLE) == '1', 'Explicit separate nonmodel smoke only')
    policy = {k: os.environ.get(k) for k in (ENABLE, SMOKE_ENABLE, ADAPTER_ENABLE, HOST_ENABLE,
        'LINGBOT_NPU_SLOT_GRANTED', 'LINGBOT_CPU_SLOT_GRANTED', 'LINGBOT_ROOT_NPU_SLOT_GRANTED', 'ROOT_LINGBOT_SP8_FATAL_DIR',
        'ROOT_LINGBOT_SP8_FATAL_NONCE', 'ROOT_LINGBOT_SP8_EXPERIMENT_MODE', 'ASCEND_CUSTOM_OPP_PATH')}
    need(policy[ADAPTER_ENABLE] == policy[HOST_ENABLE] == policy['LINGBOT_CPU_SLOT_GRANTED'] == policy['LINGBOT_NPU_SLOT_GRANTED']
         == policy['LINGBOT_ROOT_NPU_SLOT_GRANTED'] == '1'
         and policy['ROOT_LINGBOT_SP8_EXPERIMENT_MODE'] == 'correctness',
         'Original root grants/fatal protocol and correctness-only smoke required')
    with _LOCK:
        need(_SMOKE_BOUND is None, 'Fresh worker required for one immutable smoke route')
        manifest = read_pinned(manifest_pin)
        need(manifest.get('schema') == 'lingbot_QK_single_host_padded_send_HCCL_smoke_actual_startup_v3'
             and manifest.get('source_contract_verified') is True and manifest.get('model_admitted') is False
             and manifest.get('template_only') is False and manifest.get('runnable') is True
             and manifest.get('scope') == 'nonmodel_original_collective_single_host_cold8',
             'Fresh explicit nonmodel smoke startup manifest required')
        validate_leaf_admission(manifest)
        _SMOKE_BOUND = _configure_runtime(manifest, None, policy, 'hccl-smoke')
        return _SMOKE_BOUND


def try_prepared_send(module, raw, rotary, *, ring_size):
    if getattr(module, "_lingbot_recovery_qk", False):
        from .lingbot_recovery import prepare_send
        return prepare_send(module, raw, rotary, ring_size)
    route = selected_route()
    return None if route is None else route.prepare(module, raw, rotary, ring_size)
