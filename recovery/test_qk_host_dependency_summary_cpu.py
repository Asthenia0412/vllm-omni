"""Scalar receipt checks only; no torch or device import."""
import copy
import sys
import unittest

import summarize_qk_host_dependencies as s


class SummaryTest(unittest.TestCase):
    def gc(self, data, *, dropped=0, errors=0):
        return dict(columns=["wall_ns", "native_thread_id", "phase_start0_stop1", "generation"],
                    data=data, count=len(data) // 4, capacity=100, dropped=dropped, errors=errors,
                    complete=dropped == errors == 0)

    def test_same_thread_pairing_clipping_and_union(self):
        intervals, status = s.gc_intervals(self.gc([10, 7, 0, 0, 20, 8, 0, 0, 30, 7, 1, 0, 40, 8, 1, 0]))
        self.assertTrue(status['observation_complete'])
        self.assertEqual(s.overlap(15, 25, intervals, 7), 10)
        self.assertEqual(s.overlap(15, 25, intervals, 8), 5)
        self.assertEqual(s.overlap(15, 25, intervals, 9), 0)
        self.assertEqual(s.overlap(15, 35, intervals + intervals, 7), 15)

    def test_unmatched_generation_or_dropped_stays_partial(self):
        for record in (self.gc([10, 7, 0, 0, 30, 7, 1, 1]),
                       self.gc([10, 7, 0, 0], dropped=2),
                       self.gc([], errors=1)):
            intervals, status = s.gc_intervals(record)
            self.assertFalse(status['observation_complete'])
        self.assertEqual(intervals, [])

    def test_duplicate_start_and_wrong_complete_refuse(self):
        _, status = s.gc_intervals(self.gc([10, 7, 0, 0, 11, 7, 0, 0, 20, 7, 1, 0]))
        self.assertFalse(status['observation_complete'])
        record = self.gc([], dropped=1)
        record['complete'] = True
        with self.assertRaises(ValueError):
            s.gc_intervals(record)

    def test_partitions_producer_and_cached_without_overlap(self):
        row = [[i * 100, i * 99, i, i] for i in range(14)]
        for producer in (True, False):
            selected = copy.deepcopy(row)
            if not producer:
                selected[2] = selected[3] = [-1] * 4
            parts = s.stages_for(selected, producer, [], 7)
            self.assertEqual(sum(x['wall_ns'] for x in parts.values()), 1300)
            self.assertEqual('producer' in parts, producer)
            self.assertEqual(parts['pre_call_gap']['wall_ns'], 100 if producer else 300)

    def test_signed_nonatomic_cpu_residual_is_not_clamped(self):
        value = s.interval([[100, 50, 0, 0], [200, 151, 1, 2]], 0, 1, [], 7)
        self.assertEqual(value['wall_minus_thread_cpu_ns'], -1)
        with self.assertRaises(ValueError):
            s.interval([[100, 50, 0, 0], [99, 151, 1, 2]], 0, 1, [], 7)

    def test_thresholds_strictly_greater(self):
        stats = s.Stats()
        for ns in (2_000_000, 2_000_001, 5_000_000, 5_000_001):
            stats.add(dict(wall_ns=ns, thread_cpu_ns=10, voluntary_switches=0,
                           involuntary_switches=0, same_thread_gc_overlap_ns=0,
                           wall_minus_thread_cpu_ns=ns - 10))
        result = stats.export()
        self.assertEqual((result['gt2ms'], result['gt5ms']), (3, 1))

    def test_peer_skew_uses_same_row_and_not_device_entry(self):
        rows = [[dict(marks=[[i * 100 + rank * 10, 0, 0, 0] for i in range(14)], sample=0)] for rank in range(8)]
        peers = s.peer_view(rows, 0)
        self.assertEqual(peers['collective_host_start_skew_ns'], 70)
        self.assertEqual(peers['latest_host_collective_start_rank'], 7)
        self.assertEqual(len(peers['peers']), 8)
        self.assertNotIn('device_entry', peers)

    def test_cpu_only_and_boolean_not_integer(self):
        self.assertNotIn('torch', sys.modules)
        self.assertFalse(s.integer(True))
        with self.assertRaises(ValueError):
            s.need(False, 'check remains active under -O')


if __name__ == '__main__':
    unittest.main()
