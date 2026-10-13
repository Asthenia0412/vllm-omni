"""Small scalar/evidence tests; no torch, device or real profiling data import."""
import copy
import sys
import unittest

import summarize_qk_collective_boundaries as s


def boundary(count=2):
    code=dict(source_path='/original/comm.py',source_sha256='a'*64,code_sha256='b'*64,code_fingerprint_method='python_typed_code_fields_v1')
    return dict(schema=s.OBSERVER_SCHEMA,diagnostic_only=True,performance_gate=False,row_count=count,
        marks=list(s.MARKS),columns=list(s.scalar.COLS),intervals=[list(x) for x in s.INTERVALS],
        status_bits=s.STATUS,terminal_failure=False,all_rows_complete=True,pins=code,
        status=[0]*count,visited=[1]*count,restored_original_bindings=[1]*count,complete_rows=[True]*count,
        data=[v for row in range(count) for point in range(6) for v in (row*1000+point*100,point*90,point,point)],
        native_thread_id=7),code


class BoundarySummaryTest(unittest.TestCase):
    def test_new_schema_only_not_old_broad_pass(self):
        with self.assertRaisesRegex(ValueError,'new successful wrapper'):
            s.common_report(dict(schema=s.scalar.SCHEMA,method=s.scalar.METHOD,mode='profile',rank=0,passed=True),0,'profile')

    def test_exact_six_mark_rows_and_five_disjoint_stages(self):
        data,code=boundary()
        rows=s.boundary_rows(data,2,code)
        parts=[s.scalar.interval(rows[0],a,b,[],7) for _,a,b in s.INTERVALS]
        self.assertEqual(len(parts),5)
        self.assertEqual(sum(x['wall_ns'] for x in parts),rows[0][-1][0]-rows[0][0][0])

    def test_failed_unrestored_or_incomplete_observer_rejected(self):
        source,code=boundary()
        for key,value in (('terminal_failure',True),('status',[0,1]),('visited',[1,0]),
                          ('restored_original_bindings',[1,0]),('complete_rows',[True,False])):
            data=copy.deepcopy(source);data[key]=value
            with self.assertRaises(ValueError):s.boundary_rows(data,2,code)

    def test_missing_or_nonmonotonic_or_bool_timestamps_rejected(self):
        source,code=boundary()
        data=copy.deepcopy(source);data['data'].pop()
        with self.assertRaises(ValueError):s.boundary_rows(data,2,code)
        data=copy.deepcopy(source);data['data'][8]=50  # Point2 wall precedes point1=100.
        with self.assertRaises(ValueError):s.boundary_rows(data,2,code)
        data=copy.deepcopy(source);data['data'][0]=False
        with self.assertRaises(ValueError):s.boundary_rows(data,2,code)

    def test_identity_hash_and_boolean_flag_types_not_coerced(self):
        data,code=boundary();wrong=dict(code,code_sha256='c'*64)
        with self.assertRaises(ValueError):s.boundary_rows(data,2,wrong)
        data['visited']=[True,True]
        with self.assertRaises(ValueError):s.boundary_rows(data,2,code)

    def test_gc_clips_only_same_native_thread(self):
        data,code=boundary();row=s.boundary_rows(data,2,code)[0]
        gc=[dict(thread=7,start_ns=150,end_ns=250,generation=0),dict(thread=8,start_ns=100,end_ns=300,generation=0)]
        result=s.scalar.interval(row,1,2,gc,7)
        self.assertEqual(result['same_thread_gc_overlap_ns'],50)
        self.assertEqual(s.scalar.interval(row,1,2,gc,9)['same_thread_gc_overlap_ns'],0)

    def test_scalar_helpers_do_not_import_framework(self):
        self.assertNotIn('torch',sys.modules)
        self.assertNotIn('torch_npu',sys.modules)
        with self.assertRaises(ValueError):s.need(False,'kept under -O')


if __name__=='__main__':unittest.main()
