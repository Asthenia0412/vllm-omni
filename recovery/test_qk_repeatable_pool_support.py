"""Small CPU-only admission/receipt tests; never imports torch or loads payloads."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import check_qk_repeatable_pool_support as support


class SupportAdmissionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        directory = Path('/mnt/sfs_turbo/lingbot-recovery-zhoutaichang-20261012/audits/capture-fixtures/launch-v1/full/fixtures/rank0')
        cls.cases = [json.loads((directory / f'qk_kv4680_p{probe}_l00.json').read_text()) for probe in (0, 1)]

    def test_stdlib_import_and_actual_pair_key(self):
        self.assertNotIn('torch', sys.modules)
        support.validate_pair(self.cases)

    def test_key_mutation_rejected(self):
        cases = copy.deepcopy(self.cases)
        cases[1]['inputs']['descriptors']['qkv']['storage_offset'] += 16
        with self.assertRaisesRegex(RuntimeError, 'non-address pool key'):
            support.validate_pair(cases)

    def test_different_history_rejected(self):
        cases = copy.deepcopy(self.cases)
        cases[1]['kv_len'] = 9360
        with self.assertRaisesRegex(RuntimeError, 'same real request'):
            support.validate_pair(cases)

    def test_partial_repeatable_failure_is_not_pass(self):
        rows = [dict(plan=0, stage=i, stage1_status=0, repeatable_status=0, workspace_bytes=0) for i in range(4)]
        rows[2]['repeatable_status'] = 561103
        rows[3]['stage1_status'] = rows[3]['repeatable_status'] = -999
        stats = dict(capacity=1, initialized=0, accepting=0, submitted=0, pending=0, poisoned=1, closed=0)
        class FailedOps:
            def stat_names(self): return list(stats)
            def stats(self, handle): return list(stats.values())
            def initialization_status(self, handle): return [v for row in rows for v in row.values()]
        item = support.snapshot(FailedOps(), 7)
        self.assertEqual(item['initialization'][2]['repeatable_status'], 561103)
        self.assertEqual(item['initialization'][3]['stage1_status'], -999)
        with self.assertRaisesRegex(RuntimeError, 'terminal candidate failure'):
            support.initialization_pass(item)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'failure.json'
            support.save(path, dict(passed=False, failure_pool_snapshots=[item]))
            self.assertEqual(json.loads(path.read_text())['failure_pool_snapshots'][0], item)

    def test_sha_gate_failure_writes_failure_without_torch(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / 'receipt.json'
            args = ['--execute', '--manifest', 'missing.json', '--manifest-sha256', '0'*64,
                    '--harness-sha256', '0'*64, '--case', 'missing-case.json',
                    '--case-sha256', '0'*64, '--output', str(out)]
            with self.assertRaisesRegex(RuntimeError, 'harness SHA gate'):
                support.main(args)
            receipt = json.loads(out.read_text())
            self.assertFalse(receipt['passed'])
            self.assertEqual(receipt['stage'], 'preflight')
            self.assertEqual(receipt['retries'], 0)
            self.assertIn('harness SHA gate', receipt['error'])
            self.assertNotIn('torch', sys.modules)
            with self.assertRaisesRegex(RuntimeError, 'overwrite'):
                support.main(args)


if __name__ == '__main__':
    unittest.main()
