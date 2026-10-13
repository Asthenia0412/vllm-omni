"""CPU admission regressions for the independent pool replay; no device claims."""
import ast
import copy
import json
from pathlib import Path
import tempfile
import unittest

import check_qk_repeatable_matrix as checker


def closed():
    stats = dict(closed=1, native_closed=1, finalize_tid=12, producer_tid=12,
                 workspace_resident_bytes=0, pending=0, poisoned=0,
                 pool_exhaustions=0, retirement_overflows=0, retained_call_slots=0,
                 retired_count=0, accepting=0, finalizing=0)
    return [dict(handle=1, stats=stats)]


class MatrixTest(unittest.TestCase):
    def test_lifecycle_requires_this_case_reclaim_and_exact_burst(self):
        before = dict(producer_reclaimed_calls=3, producer_reclaim_tid=12, producer_tid=12,
                      submitted=5, callbacks_done=5, pending=0, poisoned=0,
                      pool_exhaustions=0, retirement_overflows=0)
        after = dict(before, producer_reclaimed_calls=4, submitted=6, callbacks_done=6)
        burst_after = dict(after, submitted=10, callbacks_done=10)
        record = dict(automatic_reclaim=dict(before=before, after=after),
                      burst=dict(before=after, after=burst_after, calls=4,
                                 per_call_device_drain=False, inflight_coverage_claim=False),
                      missing_snapshot=dict(before=burst_after, after=burst_after))
        checker.validate_lifecycle(record)
        for section, field, value in (
            ('automatic_reclaim', 'producer_reclaimed_calls', 3),
            ('burst', 'submitted', 9), ('burst', 'pending', 1),
            ('missing_snapshot', 'submitted', 11),
        ):
            # Deep-copy each side independently: observed JSON snapshots never alias.
            wrong = json.loads(json.dumps(record))
            wrong[section]['after'][field] = value
            with self.subTest(section=section, field=field), self.assertRaises(RuntimeError):
                checker.validate_lifecycle(wrong)

    def test_cleanup_rejects_unfinished_or_poisoned_pool(self):
        checker.validate_closed(closed())
        for field in ('workspace_resident_bytes', 'pending', 'poisoned',
                      'pool_exhaustions', 'retirement_overflows', 'retained_call_slots',
                      'retired_count', 'accepting', 'finalizing'):
            data = closed()
            data[0]['stats'][field] = 1
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                checker.validate_closed(data)
        for field in ('closed', 'native_closed', 'finalize_tid'):
            data = closed()
            data[0]['stats'][field] = 0
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                checker.validate_closed(data)
        for data in ([], closed() + closed()):
            with self.assertRaises(RuntimeError):
                checker.validate_closed(data)

    def test_old_correctness_cannot_admit_pool_timing(self):
        report = dict(schema='lingbot_qk_real_matrix_replay_v1', mode='correctness',
                      passed=True, rank=0)
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / 'rank0.json').write_text(json.dumps(report))
            with self.assertRaisesRegex(RuntimeError, 'successful same-rank'):
                checker.verify_correctness(directory, {}, {}, {})

    def test_timing_requires_pool_cleanup(self):
        report = dict(schema=checker.SCHEMA, mode='correctness', passed=True, rank=0,
                      matrix_sha256='matrix', manifest={}, sources={},
                      original_hccl_included=True, all40_payloads=False,
                      real_payload_layers=[0], cases=[])
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / 'rank0.json').write_text(json.dumps(report))
            with self.assertRaisesRegex(RuntimeError, 'one provider-wide pool'):
                checker.verify_correctness(directory, {'sha256': 'matrix'}, {}, {})

    def test_stat_schema_rejects_truncation_and_duplicate_names(self):
        class Ops:
            def stat_names(self):
                return ['a', 'a']
            def stats(self, handle):
                return [1, 2]
        with self.assertRaises(RuntimeError):
            checker.pool_stats(Ops(), 1)
        Ops.stat_names = lambda self: ['a']
        with self.assertRaises(RuntimeError):
            checker.pool_stats(Ops(), 1)

    def test_no_assert_or_sample_barrier(self):
        tree = ast.parse(Path(checker.__file__).read_text())
        self.assertFalse(any(isinstance(n, ast.Assert) for n in ast.walk(tree)))
        loops = [n for n in ast.walk(tree) if isinstance(n, ast.For)
                 and ast.unparse(n.iter) == 'range(30)']
        self.assertEqual(len(loops), 2)
        for loop in loops:
            self.assertFalse(any(isinstance(n, ast.Call) and (
                isinstance(n.func, ast.Name) and n.func.id == 'phase'
                or isinstance(n.func, ast.Attribute) and n.func.attr == 'barrier')
                for n in ast.walk(loop)))
        with self.assertRaises(RuntimeError):
            checker.need(False, 'active under optimized Python')

    def test_case_contract_remains_actual_layer_zero(self):
        cases = [dict(kv_len=n, probe=p, layer=0) for n, p in checker.EXPECTED]
        checker.validate_order(cases)
        for wrong in (cases[:-1], list(reversed(cases))):
            with self.assertRaises(RuntimeError):
                checker.validate_order(wrong)
        wrong = copy.deepcopy(cases)
        wrong[0]['layer'] = 1
        with self.assertRaises(RuntimeError):
            checker.validate_order(wrong)


if __name__ == '__main__':
    unittest.main()
