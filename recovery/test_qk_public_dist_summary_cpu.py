"""Small scalar/evidence tests; no torch/device/real profile import."""
import copy
import sys
import unittest

import summarize_qk_public_dist as s


def example(count=2):
    code = dict(source_path='/original/comm.py', source_sha256='a' * 64, code_sha256='b' * 64,
                code_fingerprint_method='python_typed_code_fields_v1', public_dist={'cpu_mock': True})
    enclosing = dict(schema=s.OBSERVER_SCHEMA, diagnostic_only=True, performance_gate=False, row_count=count,
        marks=list(s.OUTER_BOUNDARY_MARKS), columns=list(s.scalar.COLS),
        intervals=[list(x) for x in s.OUTER_BOUNDARY_INTERVALS],
        status_bits=s.STATUS, terminal_failure=False, all_rows_complete=True,
        pins={k: code[k] for k in ('source_path', 'source_sha256', 'code_sha256', 'code_fingerprint_method')},
        status=[0] * count, visited=[1] * count, restored_original_bindings=[1] * count, complete_rows=[True] * count,
        data=[v for row in range(count) for point in range(6) for v in (row * 1000 + point * 100, point * 90, point, point)],
        native_thread_id=7)
    public = dict(schema='lingbot_public_dist_cprofile_v3', method='thread_local_exact_body_c_call_boundaries_v1',
        pins=code['public_dist'], row_count=count, marks=list(s.MARKS), columns=list(s.scalar.COLS),
        intervals=[list(x) for x in s.INTERVALS], terminal_failure=False, all_rows_complete=True,
        no_f_locals_or_work_tensor_frame_retention=True, status=[0] * count, matched_events=[6] * count,
        restored_original_none=[1] * count, complete_rows=[True] * count, group_ids=[100] * count,
        body_frame_ids=list(range(200, 200 + count)), submit_callable_ids=[300] * count,
        wait_callable_id=400, profile_events=[35] * count, native_thread_id=7,
        data=[v for row in range(count) for point in range(6) for v in (row * 1000 + 310 + point * 10, 280 + point * 10, 3, 3)])
    return enclosing, public, code


class PublicSummaryTests(unittest.TestCase):
    def rows(self, public=None):
        enclosing, initial, code = example()
        return s.public_rows(initial if public is None else public, 2, code, s.boundary_rows(enclosing, 2, code))

    def test_old_v2_or_broad_pass_rejected(self):
        for schema, method in (('lingbot_qk_collective_boundaries_diagnostic_v2',
                                'frozen_per_call_sync_private_constructor_and_scoped_module_delegates_v2'),
                               (s.scalar.SCHEMA, s.scalar.METHOD)):
            with self.assertRaisesRegex(ValueError, 'new successful wrapper'):
                s.common_report(dict(schema=schema, method=method, mode='profile', rank=0, passed=True), 0, 'profile')

    def test_five_public_parts_partition_without_double_count(self):
        row = self.rows()[0]
        parts = [s.scalar.interval(row, a, b, [], 7) for _, a, b in s.INTERVALS]
        self.assertEqual(len(parts), 5)
        self.assertEqual(sum(x['wall_ns'] for x in parts), row[-1][0] - row[0][0])
        self.assertEqual([name for name, _, _ in s.INTERVALS],
                         ['python_before_alltoall', 'alltoall_binding', 'python_between_bindings', 'wait_binding', 'python_after_wait'])

    def test_missing_c_events_restoration_schema_or_pin_fail(self):
        _, original, _ = example()
        for key, value in (('schema', 'lingbot_prepared_collective_host_boundaries_v2'),
                           ('method', 'old'), ('pins', {}), ('matched_events', [6, 5]),
                           ('status', [0, 1]), ('restored_original_none', [1, 0]),
                           ('complete_rows', [True, False]), ('terminal_failure', True)):
            bad = copy.deepcopy(original)
            bad[key] = value
            with self.assertRaises(ValueError):
                self.rows(bad)

    def test_identity_and_integer_arrays_not_coerced(self):
        _, original, _ = example()
        for key, value in (('group_ids', [100, 101]), ('submit_callable_ids', [300, 301]),
                           ('wait_callable_id', 0), ('body_frame_ids', [True, 200]),
                           ('profile_events', [35, 5]), ('status', [False, False])):
            bad = copy.deepcopy(original)
            bad[key] = value
            with self.assertRaises(ValueError):
                self.rows(bad)

    def test_nonmonotonic_and_outside_envelope_rejected(self):
        _, original, _ = example()
        for index, value in ((4, 305), (0, 299), (20, 401), (0, False)):
            bad = copy.deepcopy(original)
            bad['data'][index] = value
            with self.assertRaises(ValueError):
                self.rows(bad)
        bad = copy.deepcopy(original)
        bad['data'].pop()
        with self.assertRaises(ValueError):
            self.rows(bad)

    def test_gc_clip_uses_same_thread_and_union(self):
        row = self.rows()[0]
        gc = [dict(thread=7, start_ns=322, end_ns=328, generation=0),
              dict(thread=7, start_ns=326, end_ns=332, generation=1),
              dict(thread=8, start_ns=310, end_ns=360, generation=0)]
        result = s.scalar.interval(row, 1, 2, gc, 7)
        self.assertEqual(result['same_thread_gc_overlap_ns'], 8)
        self.assertEqual(s.scalar.interval(row, 1, 2, gc, 9)['same_thread_gc_overlap_ns'], 0)

    def test_framework_not_imported_and_need_survives_O(self):
        self.assertNotIn('torch', sys.modules)
        self.assertNotIn('torch_npu', sys.modules)
        with self.assertRaises(ValueError):
            s.need(False, 'still active under -O')


if __name__ == '__main__':
    unittest.main()
