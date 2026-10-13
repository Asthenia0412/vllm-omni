"""CPU-only binding integrity checks with tiny synthetic files, no real DSO load."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import bind_qk_repeatable_runtime as binding


class Files:
    def __init__(self, directory):
        self.root = Path(directory) / 'current'
        self.old = Path(directory) / 'compiled'
        self.out = Path(directory) / 'binding'
        self.files = {}
        for name in (*binding.RUNTIME_RELATIVE, *binding.HARNESS_RELATIVE, *binding.TEST_RELATIVE):
            self.put(self.root / name, 'source:' + name)
        self.base_source = self.put(self.old / 'csrc/lingbot_recovery/host.cpp', 'base source')
        self.kernel_source = self.put(self.old / 'csrc/lingbot_recovery/copy.cpp', 'copy source')
        self.guard_source = self.put(self.old / 'csrc/lingbot_qk_combined/host.cpp', 'guard source')
        self.pool_source = self.put(self.old / 'recovery/experiments/qk_repeatable_pool_v2/host.cpp', 'pool source')
        for item, name in ((self.base_source, 'csrc/lingbot_recovery/host.cpp'),
                           (self.kernel_source, 'csrc/lingbot_recovery/copy.cpp'),
                           (self.guard_source, 'csrc/lingbot_qk_combined/host.cpp'),
                           (self.pool_source, 'recovery/experiments/qk_repeatable_pool_v2/host.cpp')):
            self.put(self.root / name, Path(item['path']).read_text())
        self.kernel = self.put(self.old / 'libcopy.so', 'kernel binary')
        self.base_library = self.put(self.old / 'libbase.so', 'base binary')
        self.header = self.put(self.old / 'sdk/include/acl_meta.h', 'header')
        self.sdk = self.put(self.old / 'sdk/libopapi.so', 'sdk binary')
        self.base = dict(schema='lingbot_recovery_build_v1', source=self.base_source,
            library=self.base_library, kernels=self.kernel, kernel_sources=[self.base_source, self.kernel_source],
            abi_headers=[self.header], runtime_sources=[], torch=binding.VERSIONS[0], torch_npu=binding.VERSIONS[1],
            sdk=str(self.old / 'sdk'), command=['original-compiler'], registration_passed=True,
            historical_admission_inherited=False)
        self.base_pin = self.write('base-build.json', self.base)
        self.pool = dict(schema=binding.BUILD_SCHEMA, namespace=binding.NAMESPACE,
            source=self.pool_source, library=self.put(self.old / 'libpool.so', 'pool binary'),
            builder=self.put(self.old / 'pool-build.py', 'builder'), design=self.put(self.old / 'design.txt', 'design'),
            original_nonrepeatable_source=self.put(self.old / 'old.cpp', 'old source'),
            parent_v1_source=self.put(self.old / 'v1.cpp', 'v1 source'),
            provider_manifest=self.base_pin, kernel=self.kernel, headers=[self.header],
            dependencies=[self.sdk] * 14 + [self.kernel],
            torch=binding.VERSIONS[0], torch_npu=binding.VERSIONS[1], torch_npu_git=binding.VERSIONS[2],
            registration=True, npu_initialized=False, device_admitted=False, production_entrypoint=False,
            v1_qualification_inherited=False, ownership_policy=binding.OWNERSHIP,
            statistics_fields=['capacity', 'native_closed'], initialization_status_stride=['plan', 'stage'])
        ancestor_runtime = []
        for suffix in ('vllm_omni/diffusion/distributed/comm.py',
                       'vllm_omni/diffusion/layers/lingbot_native_bf16_qk.py'):
            ancestor_runtime.append(self.put(self.old / suffix, (self.root / suffix).read_text()))
        self.guard = dict(schema='lingbot_qk_combined_build_v1', source=self.guard_source,
            library=self.put(self.old / 'libguard.so', 'guard binary'),
            build_script=self.put(self.old / 'guard-build.py', 'guard builder'),
            provider_manifest=self.base_pin, dependencies=[self.kernel], abi_headers=[self.header],
            runtime_sources=ancestor_runtime, torch=binding.VERSIONS[0], torch_npu=binding.VERSIONS[1],
            torch_npu_git=binding.VERSIONS[2], registration_passed=True, npu_initialized=False,
            historical_admission_inherited=False)
        parity_files = []
        for prefix, count in binding.PROTECTED.items():
            for number in range(count):
                name = prefix + '/file' + str(number) + '.py'
                item = self.put(self.root / name, 'protected ' + name)
                parity_files.append(dict(relative_path=name, historical_sha256=item['sha256'],
                                         current_sha256=item['sha256'], identical=True))
        parity = dict(schema='lingbot_historical_infrastructure_python_parity_v1', passed=True,
                      checked_files=228, files=parity_files)
        parity_path = self.root / binding.PARITY_RELATIVE
        self.put(parity_path, json.dumps(parity))
        self.freeze = dict(schema=binding.FREEZE_SCHEMA, source_root=str(self.root), frozen=True,
            scope='actual192_operator_integration', binder=binding.pin(binding.__file__),
            runtime_sources=[binding.pin(self.root / name) for name in binding.RUNTIME_RELATIVE],
            harness_sources=[binding.pin(self.root / name) for name in binding.HARNESS_RELATIVE],
            test_sources=[binding.pin(self.root / name) for name in binding.TEST_RELATIVE],
            infrastructure_parity=binding.pin(parity_path))

    def put(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value)
        return binding.pin(path)

    def write(self, name, value):
        return self.put(self.old / name, json.dumps(value, indent=2) + '\n')

    def bind(self):
        pool = self.write('pool-build.json', self.pool)
        guard = self.write('guard-build.json', self.guard)
        freeze = self.write('freeze.json', self.freeze)
        return binding.bind(pool['path'], pool['sha256'], guard['path'], guard['sha256'],
                            freeze['path'], freeze['sha256'], self.out)


class BindingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.f = Files(self.directory.name)

    def test_binding_keeps_ancestors_and_never_claims_new_registration(self):
        result = self.f.bind()
        self.assertNotIn('torch', sys.modules)
        base = json.loads((self.f.out / 'host_manifest.json').read_text())
        runtime = json.loads((self.f.out / 'qk_repeatable_runtime_manifest.json').read_text())
        self.assertEqual(base['compilation_ancestor'], self.f.base_pin)
        self.assertEqual(base['library'], self.f.base_library)
        self.assertEqual(base['source']['sha256'], self.f.base_source['sha256'])
        self.assertNotEqual(base['source']['path'], self.f.base_source['path'])
        self.assertFalse(runtime['recompiled'])
        self.assertFalse(runtime['registration_executed_by_binder'])
        self.assertFalse(runtime['device_admitted'])
        self.assertFalse(runtime['model_admitted'])
        self.assertEqual(runtime['pool_policy'], binding.POOL_POLICY)
        self.assertEqual(runtime['guard']['allowed_operations'], ['validate_lease'])
        self.assertEqual(runtime['provider_manifest'], result['base'])
        self.assertEqual(runtime['statistics_fields'], self.f.pool['statistics_fields'])
        self.assertEqual(json.loads(Path(self.f.base_pin['path']).read_text()), self.f.base)

    def test_mixed_compilation_base_rejected(self):
        self.f.guard['provider_manifest'] = self.f.write('different-base.json', self.f.base)
        with self.assertRaisesRegex(ValueError, 'compilation bases differ'):
            self.f.bind()
        self.assertFalse(self.f.out.exists())

    def test_changed_frozen_runtime_rejected(self):
        (self.f.root / binding.RUNTIME_RELATIVE[0]).write_text('changed after freeze')
        with self.assertRaisesRegex(ValueError, 'changed or inconsistent'):
            self.f.bind()
        self.assertFalse(self.f.out.exists())

    def test_different_current_guard_source_rejected(self):
        (self.f.root / 'csrc/lingbot_qk_combined/host.cpp').write_text('new arithmetic/guard')
        with self.assertRaisesRegex(ValueError, 'compiled ancestor: guard host'):
            self.f.bind()

    def test_refreezing_modified_collective_cannot_bypass_original_contract(self):
        suffix = 'vllm_omni/diffusion/distributed/comm.py'
        path = self.f.root / suffix
        path.write_text('modified collective')
        self.f.freeze['runtime_sources'] = [binding.pin(path) if item['path'] == str(path) else item
                                            for item in self.f.freeze['runtime_sources']]
        with self.assertRaisesRegex(ValueError, 'compiled ancestor: vllm_omni/diffusion/distributed/comm.py'):
            self.f.bind()

    def test_protected_228_source_change_rejected(self):
        (self.f.root / 'vllm_omni/engine/file0.py').write_text('new engine behavior')
        with self.assertRaisesRegex(ValueError, 'protected infrastructure source changed'):
            self.f.bind()

    def test_repeated_pin_with_different_declared_sha_is_rejected(self):
        bad = dict(self.f.header, sha256='0' * 64)
        self.f.pool['headers'].append(bad)
        with self.assertRaisesRegex(ValueError, 'changed or inconsistent'):
            self.f.bind()

    def test_source_freeze_must_include_new_harness_and_tests(self):
        self.f.freeze['harness_sources'] = []
        with self.assertRaisesRegex(ValueError, 'missing required frozen harness'):
            self.f.bind()

    def test_existing_output_never_overwritten(self):
        self.f.out.mkdir()
        marker = self.f.out / 'keep'
        marker.write_text('original')
        with self.assertRaisesRegex(ValueError, 'already exists'):
            self.f.bind()
        self.assertEqual(marker.read_text(), 'original')
        self.assertEqual(list(self.f.out.iterdir()), [marker])

    def test_declared_kernel_must_be_same_as_base(self):
        self.f.pool['kernel'] = self.f.put(self.f.old / 'otherkernel.so', 'different kernel')
        with self.assertRaisesRegex(ValueError, 'exact pack kernel differs'):
            self.f.bind()

    def test_v1_cannot_be_presented_as_v2(self):
        self.f.pool['schema'] = 'lingbot_qk_repeatable_event_pool_cpu_build_v1'
        with self.assertRaisesRegex(ValueError, 'unsupported v2 compilation ancestor'):
            self.f.bind()


if __name__ == '__main__':
    unittest.main()
