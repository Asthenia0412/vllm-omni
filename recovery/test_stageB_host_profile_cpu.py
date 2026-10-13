"""Stdlib tests of real cProfile and fail-closed diagnostic selection.

No torch import, tensor fixture, device, compiler or performance assertion.
"""
import cProfile
import gc
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch
import weakref

ROOT = Path(__file__).resolve().parents[1]
LAYERS = ROOT / 'vllm_omni/diffusion/layers'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


helper = load('host_profile_cpu', LAYERS / 'lingbot_fixed_modulation_host_profile.py')
screen = load('stageB_screen_cpu', LAYERS / 'lingbot_fixed_modulation_screen.py')


class ProfileTests(unittest.TestCase):
    def setUp(self):
        self.before = len(helper._QUARANTINE)

    def tearDown(self):
        self.assertFalse(helper.profiling_active())
        del helper._QUARANTINE[self.before:]

    def test_real_profile_counts_20_invocations_and_800_leaf_calls(self):
        sync_seen, created = [], []
        class Owner:
            pass
        def leaf():
            return 1
        def enqueue():
            self.assertTrue(helper.profiling_active())
            for _ in range(40):
                leaf()
            result = Owner()
            created.append(weakref.ref(result))
            return result
        def synchronize():
            self.assertFalse(helper.profiling_active())
            sync_seen.append(len(created))
            self.assertTrue(all(item() is not None for item in created))
        report, held = helper.profile_enqueue(enqueue, synchronize)
        self.assertEqual(sync_seen, [0, 20])
        self.assertEqual(len(held), 20)
        names = {row['function']['name']: row for row in report['methods']}
        self.assertEqual(names['enqueue']['total_calls'], 20)
        self.assertEqual(names['leaf']['total_calls'], 800)
        self.assertNotIn('synchronize', names)
        self.assertTrue(any(c['function']['name'] == 'enqueue' and c['total_calls'] == 800
                            for c in names['leaf']['callers']))
        self.assertEqual(report['enqueue_calls'], 20)
        self.assertEqual(report['actual_layers_per_call'], 40)
        del held
        gc.collect()
        self.assertTrue(all(item() is None for item in created))

    def test_recursive_primitive_and_total_counts_are_not_swapped(self):
        def recursive(depth):
            return recursive(depth - 1) if depth else None
        report, held = helper.profile_enqueue(lambda: recursive(3), lambda: None)
        row = next(r for r in report['methods'] if r['function']['name'] == 'recursive')
        self.assertEqual((row['primitive_calls'], row['total_calls']), (20, 80))
        self.assertTrue(all(c['total_calls'] >= c['primitive_calls'] for c in row['callers']))
        self.assertEqual(len(held), 20)

    def test_existing_profiler_is_not_overwritten(self):
        seen = []
        def original(frame, event, arg):
            pass
        sys.setprofile(original)
        try:
            with self.assertRaisesRegex(RuntimeError, 'existing thread profiler'):
                helper.profile_enqueue(lambda: seen.append('invoke'), lambda: seen.append('sync'))
            self.assertIs(sys.getprofile(), original)
        finally:
            sys.setprofile(None)
        self.assertEqual(seen, [])

    def test_exception_disables_profile_retains_prior_results_and_never_retries(self):
        calls, syncs = [], []
        marker = RuntimeError('injected enqueue failure')
        def invoke():
            calls.append(len(calls))
            if len(calls) == 3:
                raise marker
            return object()
        with self.assertRaises(RuntimeError) as error:
            helper.profile_enqueue(invoke, lambda: syncs.append(helper.profiling_active()))
        self.assertIs(error.exception, marker)
        self.assertEqual(calls, [0, 1, 2])
        self.assertEqual(syncs, [False])
        self.assertFalse(helper.profiling_active())
        self.assertEqual(len(helper._QUARANTINE[-1][0]), 2)

    def test_drain_failure_retains_entire_batch(self):
        calls = []
        def sync():
            self.assertFalse(helper.profiling_active())
            calls.append(1)
            if len(calls) == 2:
                raise RuntimeError('drain failed')
        with self.assertRaisesRegex(RuntimeError, 'drain failed'):
            helper.profile_enqueue(object, sync)
        self.assertEqual(len(helper._QUARANTINE[-1][0]), 20)

    def test_serialization_and_sync_are_outside_real_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'rank0.json'
            report, held = helper.profile_enqueue(object, lambda: self.assertFalse(helper.profiling_active()))
            helper.write_report(output, report)
            restored = json.loads(output.read_text())
            self.assertEqual(restored, report)
            names = [r['function']['name'] for r in report['methods']]
            self.assertNotIn('write_report', names)
            self.assertNotIn('dump', names)
            with self.assertRaisesRegex(RuntimeError, 'overwrite'):
                helper.write_report(output, report)
            self.assertEqual(len(held), 20)

    def test_writer_rejects_active_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'rank0.json'
            profiler = cProfile.Profile()
            profiler.enable()
            try:
                with self.assertRaisesRegex(RuntimeError, 'serialization'):
                    helper.write_report(output, {})
            finally:
                profiler.disable()
            self.assertFalse(output.exists())

    def test_reserved_monitoring_tool_with_no_global_events_is_rejected(self):
        monitoring = getattr(sys, 'monitoring', None)
        if monitoring is None:
            self.skipTest('sys.monitoring is unavailable before Python3.12')
        self.assertIsNone(monitoring.get_tool(5))
        monitoring.use_tool_id(5, 'stageB-cpu-external-tool')
        calls = []
        try:
            self.assertEqual(monitoring.get_events(5), 0)
            self.assertIsNone(sys.getprofile())
            self.assertTrue(helper.profiling_active())
            with self.assertRaisesRegex(RuntimeError, 'existing thread profiler'):
                helper.profile_enqueue(lambda: calls.append('invoke'), lambda: calls.append('sync'))
            self.assertEqual(monitoring.get_tool(5), 'stageB-cpu-external-tool')
            with tempfile.TemporaryDirectory() as directory:
                with self.assertRaisesRegex(RuntimeError, 'serialization'):
                    helper.write_report(Path(directory) / 'rank0.json', {})
        finally:
            monitoring.free_tool_id(5)
        self.assertEqual(calls, [])

    def test_every_reserved_tool_id_is_checked_and_absent_monitoring_is_supported(self):
        for reserved in range(6):
            fake = NS(get_tool=lambda index: 'external' if index == reserved else None)
            with self.subTest(tool=reserved), patch.object(sys, 'monitoring', fake, create=True):
                self.assertTrue(helper.profiling_active())
        with patch.object(sys, 'monitoring', None, create=True):
            self.assertFalse(helper.profiling_active())

    def test_real_cprofile_detected_and_released_on_target_interpreter(self):
        profiler = cProfile.Profile()
        self.assertFalse(helper.profiling_active())
        profiler.enable()
        try:
            self.assertTrue(helper.profiling_active())
            if hasattr(sys, 'monitoring'):
                self.assertIsNotNone(sys.monitoring.get_tool(2))
            else:
                self.assertIsNotNone(sys.getprofile())
        finally:
            profiler.disable()
        self.assertFalse(helper.profiling_active())


class SelectionTests(unittest.TestCase):
    def test_disabled_config_does_not_import_profiler_or_create_screen(self):
        code = '''import importlib.util, os, sys
from types import SimpleNamespace
for key in list(os.environ):
 if key.startswith(('LINGBOT_', 'VLLM_LINGBOT_', 'ROOT_LINGBOT_')): del os.environ[key]
spec=importlib.util.spec_from_file_location('isolated_screen', sys.argv[1])
module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
pipeline=SimpleNamespace(transformer=SimpleNamespace())
module.configure(pipeline,{})
if pipeline.transformer._lingbot_stageB_operator_screen is not None: raise RuntimeError('unexpected screen')
if any(name in sys.modules for name in ('cProfile','pstats','torch','torch_npu')): raise RuntimeError('unexpected runtime import')
'''
        subprocess.run([sys.executable, '-c', code, str(LAYERS / 'lingbot_fixed_modulation_screen.py')], check=True)

    def test_invalid_flag_or_missing_directory_refused(self):
        for value in ('true', '2', '', '1'):
            with self.subTest(value=value), patch.dict(os.environ, {'LINGBOT_STAGEB_HOST_PROFILE': value}, clear=True):
                with self.assertRaises(RuntimeError):
                    screen.configure(NS(transformer=NS()), {})

    def test_host_mode_rejects_v1_and_timing_service(self):
        env = {'LINGBOT_STAGEB_HOST_PROFILE': '1', 'LINGBOT_STAGEB_OPERATOR_SCREEN_DIR': '/not-used',
               'ROOT_LINGBOT_SP8_EXPERIMENT_MODE': 'profile', 'VLLM_LINGBOT_NPU_ASYNC_METADATA': '0'}
        with patch.dict(os.environ, env, clear=True):
            with self.assertRaisesRegex(RuntimeError, 'CPPv2'):
                screen.configure(NS(transformer=NS()), {'lingbot_npu_fixed_modulation_stageB': True})
        env['ROOT_LINGBOT_SP8_EXPERIMENT_MODE'] = 'timing'
        with patch.dict(os.environ, env, clear=True):
            with self.assertRaisesRegex(RuntimeError, 'diagnostic service'):
                screen.configure(NS(transformer=NS()), {'lingbot_npu_fixed_modulation_cpp_v2': True})

    def test_configured_mode_selects_only_host_branch(self):
        package = ModuleType('profile_screen_pkg')
        package.__path__ = []
        fake = ModuleType('profile_screen_pkg.lingbot_fixed_modulation_host_profile')
        calls = []
        fake.run = lambda *args: calls.append(args) or 'diagnostic-only'
        module = load('profile_screen_pkg.screen', LAYERS / 'lingbot_fixed_modulation_screen.py')
        with patch.dict(sys.modules, {package.__name__: package, fake.__name__: fake}), \
             patch.dict(os.environ, {'LINGBOT_STAGEB_HOST_PROFILE': '1'}, clear=True):
            current = module.Screen(Path('/unused'), host_profile=True)
            self.assertEqual(current.run('transformer', 'core', 'torch-marker'), 'diagnostic-only')
        self.assertEqual(calls, [(current, 'transformer', 'core', 'torch-marker', module.EXPECTED)])
        self.assertNotIn('torch', sys.modules)

    def test_mode_cannot_change_after_screen_construction(self):
        for configured, actual in ((True, '0'), (False, '1')):
            with patch.dict(os.environ, {'LINGBOT_STAGEB_HOST_PROFILE': actual}, clear=True):
                with self.assertRaisesRegex(RuntimeError, 'policy changed'):
                    screen.Screen(Path('/unused'), host_profile=configured).run(None, None, None)

    def test_helper_refuses_unqualified_mode_before_importing_runtime(self):
        fake = NS(host_profile=False)
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, 'explicit retained host profile'):
                helper.run(fake, None, None, None, ())
        self.assertNotIn('torch', sys.modules)


if __name__ == '__main__':
    unittest.main()
