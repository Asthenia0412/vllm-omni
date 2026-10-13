"""CPU-only source/ancestry/bridge checks; never load a DSO or import torch."""
from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
import re
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import bind_qk_private_retention_runtime as binding
import qk_private_retention_contract as contract
import test_bind_qk_repeatable_runtime as old_binding_test
import test_qk_repeatable_pool_adapter_cpu as old_adapter_test

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / 'recovery/experiments/qk_repeatable_pool_v2/host.cpp'
NATIVE = ROOT / 'recovery/experiments/qk_private_retention_v3/host.cpp'
NAME = 'vllm_omni.diffusion.layers.lingbot_qk_private_retention'
packages = {}
for name in ('vllm_omni', 'vllm_omni.diffusion', 'vllm_omni.diffusion.layers'):
    module = ModuleType(name)
    module.__path__ = [str(ROOT / name.replace('.', '/'))]
    packages[name] = module
packages[old_adapter_test.NAME] = old_adapter_test.qk
spec = importlib.util.spec_from_file_location(
    NAME, ROOT / 'vllm_omni/diffusion/layers/lingbot_qk_private_retention.py')
private = importlib.util.module_from_spec(spec)
with patch.dict(sys.modules, packages):
    spec.loader.exec_module(private)


class SourceDelta(unittest.TestCase):
    def test_parent_sha_and_exact_scoped_delta(self):
        result = contract.validate_source(PARENT, NATIVE)
        self.assertEqual(result['omitted_owner_indices'], [5, 6, 7, 8, 9, 10])
        self.assertEqual(result['retained_owner_indices'], [0, 1, 2, 3, 4, 11, 12, 13])
        self.assertTrue(result['owners_retained'])
        source = NATIVE.read_text()
        self.assertIn('TORCH_LIBRARY(lingbot_qk_private_retention_v3,m)', source)
        self.assertNotIn('TORCH_LIBRARY(lingbot_qk_repeatable_pool_v2,m)', source)

    def test_actual_allocator_order_matches_checked_six(self):
        source = PARENT.read_text()
        allocation = source[source.index('std::shared_ptr<Call> allocate_call'):source.index('struct Stage')]
        names = [m.group(1) or m.group(2) for m in re.finditer(r'(?:c->(\w+)|(\w+))=alloc\(', allocation)]
        self.assertEqual(names, list(contract.OWNER_ORDER[5:11]))
        self.assertIn('c->owners.assign(inputs.begin(),inputs.end())', allocation)
        self.assertIn('c->owners.insert(c->owners.end(),{c->send,c->v})', allocation)
        source = NATIVE.read_text()
        start = source.index('// Private6 are retained')
        self.assertLess(source.index('owners.size()==14', start), source.index('reclaim_on_producer(p);', start))
        self.assertLess(source.index('unsafeGetStorageImpl()!=', start), source.index('recordStream(', start))

    def test_arithmetic_lifetime_or_registration_expansion_rejected(self):
        source = NATIVE.read_text()
        for before, after in (('if(i>=5 && i<11)continue;', 'if(i>=4 && i<11)continue;'),
                              ('1e-6', '2e-6'),
                              ('plan.recorded_generation==plan.generation', 'true'),
                              ('p.retired[index]=std::move(plan.active);', 'plan.active.reset();')):
            self.assertIn(before, source)
            with tempfile.TemporaryDirectory() as directory:
                candidate = Path(directory) / 'host.cpp'
                candidate.write_text(source.replace(before, after, 1))
                with self.subTest(change=before), self.assertRaisesRegex(ValueError, 'exact private6 delta'):
                    contract.validate_source(PARENT, candidate)

    def test_changed_parent_cannot_redefine_allowed_delta(self):
        with self.assertRaisesRegex(ValueError, 'frozen parent v2 source changed'):
            contract.derive_source(PARENT.read_bytes() + b'\n')


class PrivateFiles(old_binding_test.Files):
    def __init__(self, directory):
        with patch.object(old_binding_test, 'binding', binding):
            super().__init__(directory)
        # Keep tiny fake libraries, but use real v2/v3 source for the strict
        # arithmetic/lifetime delta. This remains metadata-only, no DSO load.
        parent_source = self.put(self.old / 'recovery/experiments/qk_repeatable_pool_v2/host.cpp',
                                 PARENT.read_text())
        self.put(self.root / 'recovery/experiments/qk_repeatable_pool_v2/host.cpp', PARENT.read_text())
        self.pool['source'] = self.put(self.old / 'recovery/experiments/qk_private_retention_v3/host.cpp',
                                       NATIVE.read_text())
        self.put(self.root / 'recovery/experiments/qk_private_retention_v3/host.cpp', NATIVE.read_text())
        helper = self.put(self.root / 'recovery/qk_private_retention_contract.py',
                          (ROOT / 'recovery/qk_private_retention_contract.py').read_text())
        self.freeze['runtime_sources'] = [helper if item['path'] == helper['path'] else item
                                          for item in self.freeze['runtime_sources']]
        parent = dict(self.pool, schema='lingbot_qk_repeatable_event_pool_cpu_build_v2',
                      namespace='lingbot_qk_repeatable_pool_v2', source=parent_source)
        self.pool.update(parent_v2_build=self.write('v2-build.json', parent),
                         parent_v2_source=parent_source, source_contract_helper=helper,
                         source_contract=contract.validate_source(PARENT, NATIVE),
                         registration_policy=contract.REGISTRATION_POLICY,
                         v2_qualification_inherited=False,
                         model_admitted=False, performance_admitted=False)

    def bind(self):
        with patch.object(old_binding_test, 'binding', binding):
            return super().bind()


class IndependentBinding(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.f = PrivateFiles(self.directory.name)

    def test_new_receipt_uses_private_namespace_and_exact_original_bridge(self):
        result = self.f.bind()
        runtime = json.loads(Path(result['runtime']['path']).read_text())
        self.assertEqual(runtime['schema'], private.SCHEMA)
        self.assertEqual(runtime['namespace'], contract.NAMESPACE)
        self.assertEqual(runtime['registry_bridge'], private.BRIDGE)
        self.assertEqual(runtime['source_contract'], contract.validate_source(PARENT, NATIVE))
        self.assertEqual(runtime['original_operator_adapter'],
                         binding.pin(self.f.root / 'vllm_omni/diffusion/layers/lingbot_qk_repeatable_pool.py'))
        self.assertFalse(runtime['device_admitted'])
        self.assertFalse(runtime['v2_qualification_inherited'])
        self.assertNotIn('torch', sys.modules)

    def test_v2_or_wrong_namespace_cannot_be_presented_as_private_v3(self):
        for field, value in (('schema', 'lingbot_qk_repeatable_event_pool_cpu_build_v2'),
                             ('namespace', 'lingbot_qk_repeatable_pool_v2'),
                             ('v2_qualification_inherited', True)):
            old = self.f.pool[field]
            self.f.pool[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.f.bind()
            self.f.pool[field] = old
            self.assertFalse(self.f.out.exists())

    def test_changed_private_source_or_parent_ancestry_rejected(self):
        self.f.pool['source_contract']['omitted_owner_indices'] = [4, 5, 6, 7, 8, 9, 10]
        with self.assertRaisesRegex(ValueError, 'exact source delta'):
            self.f.bind()
        self.assertFalse(self.f.out.exists())
        self.f.pool['source_contract'] = contract.validate_source(PARENT, NATIVE)
        self.f.pool['parent_v2_source'] = self.f.put(self.f.old / 'changed-parent.cpp', 'changed')
        with self.assertRaisesRegex(ValueError, 'v2 compilation ancestry differs'):
            self.f.bind()
        self.assertFalse(self.f.out.exists())

    def test_source_edit_after_freeze_and_protected_change_are_rejected(self):
        path = self.f.root / 'vllm_omni/diffusion/layers/lingbot_qk_private_retention.py'
        old = path.read_text()
        path.write_text('changed after freeze')
        with self.assertRaisesRegex(ValueError, 'changed or inconsistent'):
            self.f.bind()
        path.write_text(old)
        (self.f.root / 'vllm_omni/engine/file0.py').write_text('new engine')
        with self.assertRaisesRegex(ValueError, 'protected infrastructure source changed'):
            self.f.bind()

    def test_wrong_runtime_namespace_is_rejected_before_ancestor_reads(self):
        self.f.bind()
        manifest = json.loads((self.f.out / 'qk_private_retention_runtime_manifest.json').read_text())
        for field, value in (('namespace', 'lingbot_qk_repeatable_pool_v2'),
                             ('schema', 'lingbot_qk_repeatable_runtime_binding_v1')):
            wrong = dict(manifest, **{field: value})
            with patch.object(private, 'read_pinned', side_effect=AssertionError('must refuse before reads')):
                with self.assertRaisesRegex(RuntimeError, 'independent operator-only'):
                    private.validate_binding(wrong)

    def runtime_for_imported_adapter(self):
        self.f.bind()
        manifest = json.loads((self.f.out / 'qk_private_retention_runtime_manifest.json').read_text())
        # The metadata-only fixture lives in a temporary root. Loader guards
        # separately bind the three actual imported files rather than allowing
        # fixture placeholders to impersonate their absolute import paths.
        imported = [private.pin(private.__file__), private.pin(private.original.__file__),
                    private.pin(private.source_contract_module.__file__)]
        manifest['runtime_sources'].extend(imported)
        manifest['original_operator_adapter'] = imported[1]
        return manifest

    def test_runtime_checks_all_imported_sources_and_actual_v3_ancestry(self):
        manifest = self.runtime_for_imported_adapter()
        ancestor = private.validate_binding(manifest)
        self.assertEqual(ancestor['schema'], contract.BUILD_SCHEMA)
        wrong = dict(manifest, registry_bridge=dict(private.BRIDGE, unchanged_consumer=False))
        with self.assertRaisesRegex(RuntimeError, 'Route/Lease registry bridge'):
            private.validate_binding(wrong)
        wrong = dict(manifest, original_operator_adapter=manifest['runtime_sources'][0])
        with self.assertRaisesRegex(RuntimeError, 'original operator adapter is not pinned'):
            private.validate_binding(wrong)

    def test_runtime_refuses_changed_pin_and_inherited_device_admission(self):
        manifest = self.runtime_for_imported_adapter()
        path = self.f.root / 'recovery/load_real_fixtures.py'
        path.write_text('changed source after binding')
        with self.assertRaisesRegex(RuntimeError, 'source/dependency changed'):
            private.validate_binding(manifest)
        manifest['device_admitted'] = True
        with self.assertRaisesRegex(RuntimeError, 'independent operator-only'):
            private.validate_binding(manifest)


class ExactConsumerBridge(unittest.TestCase):
    def setUp(self):
        self.original = private.original
        self.previous = self.original._BOUND
        self.addCleanup(setattr, self.original, '_BOUND', self.previous)

    def test_dirty_registry_is_refused_without_loading_or_replacing_it(self):
        occupied = object()
        self.original._BOUND = occupied
        with patch.object(private, 'validate_binding', side_effect=AssertionError('must not load')):
            with self.assertRaisesRegex(RuntimeError, 'registry is already occupied'):
                private.load()
        self.assertIs(self.original._BOUND, occupied)

    def test_exact_original_route_lease_and_policy_used(self):
        p = old_adapter_test.provider()
        p = private.PrivateProvider(p.base, p.ops, p.guard, p.manifest, p.policy)
        arguments = old_adapter_test.args()
        original_route, arguments, shared = old_adapter_test.route(p, arguments)
        with patch.object(private, 'load', return_value=p):
            route = private.configure_operator_module(original_route.module, original_route.config)
        self.original._BOUND = p
        lease = route.prepare(arguments[0], (shared.cosine, shared.sine), 1)
        self.assertIs(type(route), self.original.PoolRoute)
        self.assertIs(type(lease), self.original.PoolLease)
        self.assertIs(private.PoolLease, self.original.PoolLease)
        self.assertIs(private.PrivateProvider.check_policy, self.original.PoolProvider.check_policy)
        self.assertIs(private.PrivateProvider.close_all_drained, self.original.PoolProvider.close_all_drained)
        self.assertIs(private.PrivateProvider.finalize_all_drained, self.original.PoolProvider.finalize_all_drained)
        self.assertEqual(len(p.ops.runs), 1)

    def test_production_remains_default_off_and_enabled_rejected(self):
        cfg = old_adapter_test.config()
        cfg.model_config[private.CONFIG_KEY] = False
        fake = SimpleNamespace(get_current_diffusion_config_or_none=lambda: cfg)
        with patch.dict(sys.modules, {'vllm_omni.diffusion.config': fake}), \
             patch.object(private, 'load', side_effect=AssertionError('no production load')):
            self.assertIsNone(private.configure_module(object()))
            cfg.model_config[private.CONFIG_KEY] = True
            with self.assertRaisesRegex(RuntimeError, 'model construction is not admitted'):
                private.configure_module(object())

    def test_adapter_does_not_replace_consumer_or_original_functions(self):
        tree = ast.parse(Path(private.__file__).read_text())
        writes = [n for n in ast.walk(tree) if isinstance(n, ast.Attribute)
                  and isinstance(n.ctx, ast.Store) and isinstance(n.value, ast.Name)
                  and n.value.id == 'original']
        self.assertEqual([n.attr for n in writes], ['_BOUND'])
        self.assertNotIn('torch', sys.modules)


if __name__ == '__main__':
    unittest.main()
