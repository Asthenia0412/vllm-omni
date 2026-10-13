"""CPU-only source binding for independent private-retention v3 experiments.

No compilation, framework import, registration, device calls or admission. Keep
original build receipts immutable; write two new manifests into a fresh directory.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path

from qk_private_retention_contract import (REGISTRATION_POLICY, validate_source)

SCHEMA = 'lingbot_qk_private_retention_runtime_binding_v3'
FREEZE_SCHEMA = 'lingbot_qk_private_retention_source_freeze_v3'
BUILD_SCHEMA = 'lingbot_qk_private_retention_cpu_build_v3'
NAMESPACE = 'lingbot_qk_private_retention_v3'
OWNERSHIP = 'bounded_consumer_retirement_producer_reclaim_two_phase_close_v2'
VERSIONS = ('2.10.0+cpu', '2.10.0.post4', '5dd8ef3f9b375b5ae4a83538d5785754148c3302')
POOL_POLICY = dict(max_pools=1, plan_capacity=24, retirement_capacity=32,
                   workspace_bytes_per_plan=79092224, workspace_budget_bytes=1898213376)
BASE_RELATIVE = tuple('vllm_omni/diffusion/' + suffix for suffix in (
    'layers/lingbot_recovery.py', 'layers/lingbot_recv_kv_sp8.py',
    'layers/lingbot_qk_combined.py', 'layers/lingbot_real_fixture_capture.py',
    'layers/lingbot_output_capture.py', 'layers/lingbot_sp8_fatal.py',
    'layers/qk_single_host_integration_v1.py', 'layers/lingbot_fixed_timestep_bank_stageA_v1.py',
    'models/lingbot_world/transformer.py', 'attention/backends/cross_combined_integration_v1.py',
    'distributed/autoencoders/wan_npu_exact_pack.py'))
RUNTIME_RELATIVE = BASE_RELATIVE + (
    'vllm_omni/diffusion/layers/lingbot_qk_repeatable_pool.py',
    'vllm_omni/diffusion/layers/lingbot_qk_private_retention.py',
    'recovery/qk_private_retention_contract.py',
    'recovery/check_qk_repeatable_matrix.py',
    'vllm_omni/diffusion/layers/lingbot_native_bf16_qk.py',
    'vllm_omni/diffusion/distributed/comm.py', 'vllm_omni/diffusion/config.py',
    'recovery/load_real_fixtures.py', 'recovery/check_qk_real_matrix.py',
    'recovery/qk_replay_fixture.py')
HARNESS_RELATIVE = ('recovery/check_qk_private_retention_matrix.py',
                    'recovery/aggregate_qk_private_retention_matrix.py',
                    'recovery/run_qk_private_retention.sh')
TEST_RELATIVE = ('recovery/test_qk_private_retention_cpu.py',
                 'recovery/test_qk_private_retention_matrix.py',
                 'recovery/test_bind_qk_repeatable_runtime.py',
                 'recovery/test_qk_repeatable_pool_adapter_cpu.py',
                 'recovery/test_qk_repeatable_matrix.py')
PARITY_RELATIVE = 'recovery/validation/historical-infrastructure-source-parity-20261013.json'
PROTECTED = {'vllm_omni/engine': 58, 'vllm_omni/distributed': 38,
             'vllm_omni/entrypoints': 114, 'vllm_omni/worker': 18}


def need(value, message):
    if not value:
        raise ValueError('QK repeatable binding: ' + message)


def pin(path):
    path = Path(path).resolve(strict=True)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest.hexdigest())


class Pins:
    def __init__(self):
        self.checked = {}

    def verify(self, item):
        need(type(item) is dict and set(item) == {'path', 'bytes', 'sha256'}, 'invalid pin schema')
        need(type(item['bytes']) is int and item['bytes'] >= 0
             and isinstance(item['sha256'], str) and len(item['sha256']) == 64
             and all(x in '0123456789abcdef' for x in item['sha256']), 'invalid pin size/SHA')
        path = Path(item['path'])
        need(path.is_absolute() and str(path.resolve(strict=True)) == item['path'], 'pin path must be canonical absolute')
        actual = self.checked.get(item['path'])
        if actual is None:
            actual = pin(path)
            self.checked[item['path']] = actual
        need(actual == item, 'changed or inconsistent file: ' + item['path'])
        return actual

    def capture(self, path):
        value = pin(path)
        self.verify(value)
        return value

    def json(self, item):
        self.verify(item)
        return json.loads(Path(item['path']).read_text())

    def recheck(self):
        for expected in self.checked.values():
            need(pin(expected['path']) == expected, 'file changed during binding: ' + expected['path'])


def manifest_pin(path, sha, pins):
    item = pins.capture(path)
    need(item['sha256'] == sha, 'explicit manifest SHA differs: ' + item['path'])
    return item


def same_bytes(left, right, name):
    need((left['bytes'], left['sha256']) == (right['bytes'], right['sha256']),
         'current source differs from compiled ancestor: ' + name)


def unique_pins(items, pins):
    output, seen = [], set()
    for item in items:
        pins.verify(item)  # Repeated declarations must agree, not just first occurrence.
        if item['path'] not in seen:
            output.append(item)
            seen.add(item['path'])
    return output


def current_sources(root, entries, required, pins, label):
    selected = {}
    for item in entries:
        pins.verify(item)
        path = Path(item['path'])
        need(path.is_relative_to(root), label + ' source escaped selected worktree')
        name = str(path.relative_to(root))
        need(name not in selected, 'duplicate ' + label + ' source: ' + name)
        selected[name] = item
    need(set(required).issubset(selected), 'missing required frozen ' + label + ': ' +
         ','.join(sorted(set(required) - set(selected))))
    return selected


def validate_freeze(freeze, pins):
    need(freeze.get('schema') == FREEZE_SCHEMA and freeze.get('frozen') is True
         and freeze.get('scope') == 'actual192_operator_integration', 'explicit operator source freeze required')
    root = Path(freeze['source_root']).resolve(strict=True)
    need(str(root) == freeze['source_root'], 'freeze root must be canonical')
    need(freeze['binder'] == pins.capture(__file__), 'freeze belongs to another binder source')
    runtime = current_sources(root, freeze['runtime_sources'], RUNTIME_RELATIVE, pins, 'runtime')
    harness = current_sources(root, freeze['harness_sources'], HARNESS_RELATIVE, pins, 'harness')
    tests = current_sources(root, freeze['test_sources'], TEST_RELATIVE, pins, 'test')
    for selected in (runtime, harness, tests):
        need(all(not any(name == prefix or name.startswith(prefix + '/') for prefix in PROTECTED)
                 for name in selected), 'new binding must not modify protected infrastructure')
    need(freeze['infrastructure_parity']['path'] == str(root / PARITY_RELATIVE),
         'historical parity evidence must belong to selected worktree')
    parity = pins.json(freeze['infrastructure_parity'])
    need(parity.get('schema') == 'lingbot_historical_infrastructure_python_parity_v1'
         and parity.get('passed') is True and parity.get('checked_files') == 228,
         'historical protected-file authority missing')
    expected = {}
    for entry in parity['files']:
        name = entry['relative_path']
        need(name not in expected and entry['identical'] is True
             and entry['historical_sha256'] == entry['current_sha256'], 'invalid original parity entry')
        expected[name] = entry['historical_sha256']
    actual_names = set()
    for prefix, count in PROTECTED.items():
        paths = sorted((root / prefix).rglob('*.py'))
        need(len(paths) == count, 'protected Python file count changed: ' + prefix)
        actual_names.update(str(path.relative_to(root)) for path in paths)
    need(actual_names == set(expected) and len(expected) == 228, 'protected Python filename set differs')
    protected_pins = []
    for name in sorted(expected):
        value = pins.capture(root / name)
        need(value['sha256'] == expected[name], 'protected infrastructure source changed: ' + name)
        protected_pins.append(value)
    parity_digest = hashlib.sha256(json.dumps(protected_pins, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return root, runtime, harness, tests, dict(authority=freeze['infrastructure_parity'],
        checked_files=228, names_and_content_match=True, current_file_pins_sha256=parity_digest,
        scope='four protected top-level Python trees only; no performance implication')


def prepare(pool_pin, guard_pin, freeze_pin, output, pins):
    pool, guard, freeze = pins.json(pool_pin), pins.json(guard_pin), pins.json(freeze_pin)
    need(pool.get('schema') == BUILD_SCHEMA and pool.get('namespace') == NAMESPACE
         and pool.get('registration') is True and pool.get('npu_initialized') is False
         and pool.get('device_admitted') is False and pool.get('production_entrypoint') is False
         and pool.get('v1_qualification_inherited') is False and pool.get('ownership_policy') == OWNERSHIP,
         'unsupported private-retention v3 compilation ancestor')
    need(guard.get('schema') == 'lingbot_qk_combined_build_v1'
         and guard.get('registration_passed') is True and guard.get('npu_initialized') is False
         and guard.get('historical_admission_inherited') is False, 'unsupported guard compilation ancestor')
    for record in (pool, guard):
        need(tuple(record[k] for k in ('torch', 'torch_npu', 'torch_npu_git')) == VERSIONS,
             'compiled framework ABI differs')
    need(pool['provider_manifest'] == guard['provider_manifest'], 'pool/guard compilation bases differ')
    ancestor_pin = pool['provider_manifest']
    ancestor = pins.json(ancestor_pin)
    need(ancestor.get('schema') == 'lingbot_recovery_build_v1'
         and ancestor.get('registration_passed') is True and ancestor.get('historical_admission_inherited') is False
         and (ancestor['torch'], ancestor['torch_npu']) == VERSIONS[:2], 'invalid shared base compilation ancestor')
    need(pool.get('v2_qualification_inherited') is False
         and pool.get('registration_policy') == REGISTRATION_POLICY,
         'private6 build must have independent registration policy and no inherited qualification')
    parent = pins.json(pool['parent_v2_build'])
    need(parent.get('schema') == 'lingbot_qk_repeatable_event_pool_cpu_build_v2'
         and parent.get('namespace') == 'lingbot_qk_repeatable_pool_v2'
         and parent.get('registration') is True and parent.get('npu_initialized') is False
         and parent.get('v1_qualification_inherited') is False
         and parent['provider_manifest'] == pool['provider_manifest']
         and parent['source'] == pool['parent_v2_source'], 'frozen v2 compilation ancestry differs')
    pins.verify(pool['parent_v2_source'])
    pins.verify(pool['source_contract_helper'])
    expected_contract = validate_source(pool['parent_v2_source']['path'], pool['source']['path'])
    need(pool.get('source_contract') == expected_contract, 'private6 exact source delta differs')
    root, runtime, harness, tests, parity = validate_freeze(freeze, pins)
    need(pool['source_contract_helper'] == runtime['recovery/qk_private_retention_contract.py'],
         'builder and binder source-contract helper differs')
    for key in ('source', 'builder', 'design', 'original_nonrepeatable_source', 'parent_v1_source', 'library', 'kernel'):
        pins.verify(pool[key])
    for item in (*pool['headers'], *pool['dependencies']):
        pins.verify(item)
    for key in ('source', 'library', 'build_script'):
        pins.verify(guard[key])
    for item in (*guard['dependencies'], *guard['abi_headers']):
        pins.verify(item)
    for key in ('source', 'library', 'kernels'):
        pins.verify(ancestor[key])
    for item in (*ancestor['kernel_sources'], *ancestor['abi_headers']):
        pins.verify(item)
    need(pool['kernel'] == ancestor['kernels'] and guard['dependencies'] == [ancestor['kernels']],
         'shared exact pack kernel differs')
    need(len(pool['dependencies']) == 15 and pool['dependencies'][-1] == ancestor['kernels'],
         'v3 actual symbol-resolution list differs')
    # Compare source content in the new checkout with the actual compilation
    # inputs. Different absolute source paths are explicit, never relabelled.
    current_source = pins.capture(root / 'csrc/lingbot_recovery/host.cpp')
    same_bytes(current_source, ancestor['source'], 'base host')
    same_bytes(pins.capture(root / 'csrc/lingbot_qk_combined/host.cpp'), guard['source'], 'guard host')
    same_bytes(pins.capture(root / 'recovery/experiments/qk_private_retention_v3/host.cpp'), pool['source'], 'v3 host')
    same_bytes(pins.capture(root / 'recovery/experiments/qk_repeatable_pool_v2/host.cpp'), pool['parent_v2_source'], 'frozen v2 host')
    current_kernel_sources = []
    ancestor_kernel_root = Path(ancestor['source']['path']).parent
    for item in ancestor['kernel_sources']:
        parent_path = Path(item['path'])
        need(parent_path.parent == ancestor_kernel_root, 'unexpected ancestor kernel source location')
        current = pins.capture(root / 'csrc/lingbot_recovery' / parent_path.name)
        same_bytes(current, item, parent_path.name)
        current_kernel_sources.append(current)
    current_names = {p.name for p in (root / 'csrc/lingbot_recovery').iterdir() if p.is_file()}
    need(current_names == {Path(p['path']).name for p in current_kernel_sources}, 'base kernel source set changed')
    # Original collective and coefficient producer must be byte-identical to
    # the compiled guard's recorded runtime ancestry, unlike the declared hooks.
    for suffix in ('vllm_omni/diffusion/distributed/comm.py',
                   'vllm_omni/diffusion/layers/lingbot_native_bf16_qk.py'):
        items = [item for item in guard['runtime_sources'] if item['path'].endswith('/' + suffix)]
        need(len(items) == 1, 'missing immutable ancestor runtime source: ' + suffix)
        pins.verify(items[0])
        same_bytes(runtime[suffix], items[0], suffix)
    headers = unique_pins([*pool['headers'], *ancestor['abi_headers'], *guard['abi_headers']], pins)
    base = dict(schema='lingbot_recovery_build_v1', binding_schema='lingbot_recovery_source_binding_v1',
        source=current_source, library=ancestor['library'], kernels=ancestor['kernels'],
        kernel_sources=current_kernel_sources, runtime_sources=[*runtime.values(), *harness.values()],
        abi_headers=ancestor['abi_headers'], torch=ancestor['torch'], torch_npu=ancestor['torch_npu'],
        sdk=ancestor['sdk'], compilation_ancestor=ancestor_pin, compilation_command=ancestor['command'],
        registration_passed=True, registration_evidence='same exact binary; ancestor CPU registration only',
        binding_only=True, recompiled=False, device_validated=False, model_validated=False,
        device_admitted=False, model_admitted=False, performance_admitted=False,
        historical_admission_inherited=False, binder=freeze['binder'], freeze_manifest=freeze_pin,
        infrastructure_parity=parity)
    base_path = output / 'host_manifest.json'
    base_bytes = (json.dumps(base, indent=2) + '\n').encode()
    new_base_pin = dict(path=str(base_path), bytes=len(base_bytes), sha256=hashlib.sha256(base_bytes).hexdigest())
    binding = dict(schema=SCHEMA, compilation_ancestor=pool_pin, source=pool['source'], library=pool['library'],
        namespace=NAMESPACE, provider_manifest=new_base_pin, dependencies=pool['dependencies'], abi_headers=headers,
        guard=dict(compilation_ancestor=guard_pin, source=guard['source'], library=guard['library'],
            dependencies=guard['dependencies'], abi_headers=guard['abi_headers'], namespace='lingbot_qk_combined_v1',
            allowed_operations=['validate_lease'], resolution_only_operations=['dependencies']),
        runtime_sources=[*runtime.values(), *harness.values()], harness_sources=list(harness.values()),
        test_sources=list(tests.values()), tests_executed_by_binder=False,
        torch=pool['torch'], torch_npu=pool['torch_npu'], torch_npu_git=pool['torch_npu_git'],
        statistics_fields=pool['statistics_fields'], initialization_status_stride=pool['initialization_status_stride'],
        ownership_policy=OWNERSHIP, pool_policy=POOL_POLICY, freeze_manifest=freeze_pin, binder=freeze['binder'],
        binding_only=True, recompiled=False, registration_executed_by_binder=False, device_calls_by_binder=False,
        historical_admission_inherited=False, v1_qualification_inherited=False,
        device_admitted=False, model_admitted=False, performance_admitted=False,
        source_root=str(root), infrastructure_parity=parity, execution_scope='actual192_operator_integration',
        no_compute_guard_reuse=True, v2_qualification_inherited=False,
        registration_policy=REGISTRATION_POLICY, source_contract=expected_contract,
        parent_v2_build=pool['parent_v2_build'], parent_v2_source=pool['parent_v2_source'],
        original_operator_adapter=runtime['vllm_omni/diffusion/layers/lingbot_qk_repeatable_pool.py'],
        registry_bridge=dict(policy='install_validated_v3_into_empty_original_registry_only',
            route_type='lingbot_qk_repeatable_pool.PoolRoute',
            lease_type='lingbot_qk_repeatable_pool.PoolLease', unchanged_consumer=True))
    return base_bytes, (json.dumps(binding, indent=2) + '\n').encode(), binding


def bind(pool_path, pool_sha, guard_path, guard_sha, freeze_path, freeze_sha, output):
    pins = Pins()
    pool_pin = manifest_pin(pool_path, pool_sha, pins)
    guard_pin = manifest_pin(guard_path, guard_sha, pins)
    freeze_pin = manifest_pin(freeze_path, freeze_sha, pins)
    output = Path(output).resolve()
    need(not output.exists(), 'output directory already exists; immutable binding never overwritten')
    base, runtime, binding = prepare(pool_pin, guard_pin, freeze_pin, output, pins)
    pins.recheck()  # Includes all current sources, test pins and protected files.
    output.mkdir(parents=True, exist_ok=False)
    for name, payload in (('host_manifest.json', base), ('qk_private_retention_runtime_manifest.json', runtime)):
        with (output / name).open('xb') as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    # A racing edit fails this CPU action and also makes the runtime loader's
    # exact pins invalid. Preserve this directory for audit; never fix in place.
    pins.recheck()
    need(pin(output / 'host_manifest.json') == binding['provider_manifest'], 'written base manifest differs')
    return dict(base=pin(output / 'host_manifest.json'), runtime=pin(output / 'qk_private_retention_runtime_manifest.json'),
                binding_only=True, recompiled=False, device_admitted=False)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pool-build', required=True, type=Path)
    p.add_argument('--pool-build-sha256', required=True)
    p.add_argument('--guard-build', required=True, type=Path)
    p.add_argument('--guard-build-sha256', required=True)
    p.add_argument('--freeze-manifest', required=True, type=Path)
    p.add_argument('--freeze-sha256', required=True)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args(argv)
    print(json.dumps(bind(args.pool_build, args.pool_build_sha256, args.guard_build, args.guard_build_sha256,
                         args.freeze_manifest, args.freeze_sha256, args.output), indent=2))


if __name__ == '__main__':
    main()
