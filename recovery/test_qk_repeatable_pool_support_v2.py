"""CPU-only v2 support admission/state contract tests; no torch or payload reads."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import check_qk_repeatable_pool_support_v2 as support


def cold_snapshot():
    stats = dict(capacity=1, initialized=1, retirement_capacity=1, accepting=1,
                 workspace_initial_bytes=100, workspace_resident_bytes=100,
                 producer_tid=1001, consumer_tid=0)
    for name in ('submitted', 'callbacks_done', 'generations', 'pending', 'poisoned', 'closed',
                 'native_closed', 'finalizing', 'retired_count', 'retired_total', 'retired_high_water',
                 'retirement_overflows', 'producer_reclaimed_calls', 'producer_reclaim_tid',
                 'finalize_released_calls', 'finalize_tid', 'retained_call_slots', 'pool_exhaustions',
                 'event_not_ready', 'plan_reuses'):
        stats[name] = 0
    rows = [dict(plan=0, stage=i, stage1_status=0, repeatable_status=0, workspace_bytes=25)
            for i in range(4)]
    return dict(handle=1, stats=stats, initialization=rows)


def lifecycle_snapshot(phase):
    item = cold_snapshot()
    s = item['stats']
    s.update(submitted=2, callbacks_done=2, generations=2, consumer_tid=2002,
             plan_reuses=1, retired_total=1, retired_high_water=1, retained_call_slots=1,
             retired_count=1)
    if phase != 'retired':
        s.update(retired_count=0, producer_reclaimed_calls=1, producer_reclaim_tid=1001)
    if phase in ('native_closed', 'finalized'):
        s.update(accepting=0, native_closed=1)
    if phase == 'finalized':
        s.update(closed=1, retained_call_slots=0, workspace_resident_bytes=0,
                 finalize_tid=1001, finalize_released_calls=2)
    return item


class SupportV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path('/mnt/sfs_turbo/lingbot-recovery-zhoutaichang-20261012/audits/capture-fixtures/launch-v1/full/fixtures/rank0')
        cls.cases = [json.loads((root / f'qk_kv4680_p{i}_l00.json').read_text()) for i in (0, 1)]

    def test_import_is_stdlib_and_two_real_metadata_cases(self):
        self.assertNotIn('torch', sys.modules)
        support.validate_pair(self.cases)

    def test_changed_key_or_history_rejected(self):
        for field in ('offset', 'history'):
            cases = copy.deepcopy(self.cases)
            if field == 'offset':
                cases[1]['inputs']['descriptors']['qkv']['storage_offset'] += 16
            else:
                cases[1]['kv_len'] = 9360
            with self.assertRaises(RuntimeError):
                support.validate_pair(cases)

    def test_exact_v2_build_required(self):
        manifest = dict(schema=support.BUILD_SCHEMA, namespace=support.NAMESPACE,
                        registration=True, npu_initialized=False, device_admitted=False,
                        production_entrypoint=False, v1_qualification_inherited=False,
                        ownership_policy=support.OWNERSHIP_POLICY)
        support.validate_manifest_abi(manifest)
        for name, bad in (('schema', 'lingbot_qk_repeatable_event_pool_cpu_build_v1'),
                          ('namespace', 'lingbot_qk_repeatable_pool_v1'),
                          ('v1_qualification_inherited', True), ('device_admitted', True)):
            changed = dict(manifest, **{name: bad})
            with self.assertRaisesRegex(RuntimeError, 'unsupported v2'):
                support.validate_manifest_abi(changed)

    def test_initialization_and_partial_failure(self):
        item = cold_snapshot()
        support.initialization_pass(item)
        item['initialization'][2]['repeatable_status'] = 561103
        item['initialization'][3]['stage1_status'] = -999
        with self.assertRaisesRegex(RuntimeError, 'terminal candidate failure'):
            support.initialization_pass(item)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'failure.json'
            support.save(path, dict(passed=False, failure_pool_snapshots=[item]))
            self.assertEqual(json.loads(path.read_text())['failure_pool_snapshots'][0], item)

    def test_all_four_lifecycle_phases(self):
        for phase in ('retired', 'reclaimed', 'native_closed', 'finalized'):
            support.lifecycle_pass(phase, lifecycle_snapshot(phase), cold_snapshot())

    def test_wrong_reclaim_thread_rejected(self):
        item = lifecycle_snapshot('reclaimed')
        item['stats']['producer_reclaim_tid'] = item['stats']['consumer_tid']
        with self.assertRaisesRegex(RuntimeError, 'original producer'):
            support.lifecycle_pass('reclaimed', item, cold_snapshot())

    def test_native_close_cannot_clear_owners_or_publish_closed(self):
        for key, value in (('workspace_resident_bytes', 0), ('retained_call_slots', 0), ('closed', 1)):
            item = lifecycle_snapshot('native_closed')
            item['stats'][key] = value
            with self.assertRaises(RuntimeError):
                support.lifecycle_pass('native_closed', item, cold_snapshot())

    def test_finalize_wrong_thread_or_incomplete_release_rejected(self):
        for key, value in (('finalize_tid', 2002), ('finalize_released_calls', 1),
                           ('retired_count', 1), ('workspace_resident_bytes', 100)):
            item = lifecycle_snapshot('finalized')
            item['stats'][key] = value
            with self.assertRaises(RuntimeError):
                support.lifecycle_pass('finalized', item, cold_snapshot())

    def test_pending_poison_overflow_fail(self):
        for key in ('pending', 'poisoned', 'retirement_overflows'):
            item = lifecycle_snapshot('retired')
            item['stats'][key] = 1
            with self.assertRaisesRegex(RuntimeError, 'healthy'):
                support.lifecycle_pass('retired', item, cold_snapshot())

    def test_schema_is_new_and_no_execute_does_not_import_torch(self):
        support.main(['--manifest', 'missing.json', '--manifest-sha256', '0'*64,
                      '--harness-sha256', '0'*64, '--case', 'missing-case.json',
                      '--case-sha256', '0'*64, '--output', 'unused.json'])
        self.assertTrue(support.SCHEMA.endswith('_v2'))
        self.assertNotIn('torch', sys.modules)

    def test_sha_failure_receipt_no_retry_or_device_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / 'receipt.json'
            args = ['--execute', '--manifest', 'missing.json', '--manifest-sha256', '0'*64,
                    '--harness-sha256', '0'*64, '--case', 'missing-case.json',
                    '--case-sha256', '0'*64, '--output', str(out)]
            with self.assertRaisesRegex(RuntimeError, 'harness SHA gate'):
                support.main(args)
            receipt = json.loads(out.read_text())
            self.assertEqual(receipt['schema'], support.SCHEMA)
            self.assertFalse(receipt['passed'])
            self.assertFalse(receipt['v1_qualification_inherited'])
            self.assertFalse(receipt['hot_run_auto_reclaim_tested'])
            self.assertFalse(receipt['two_phase_close_tested'])
            self.assertEqual(receipt['retries'], 0)
            self.assertIn('no retry/run/reclaim/close/finalize/device sync', receipt['failure_cleanup'])
            self.assertNotIn('torch', sys.modules)
            with self.assertRaisesRegex(RuntimeError, 'overwrite'):
                support.main(args)


if __name__ == '__main__':
    unittest.main()
