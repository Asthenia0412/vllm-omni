"""Small synthetic export tests; no live export, device, or model access."""
from __future__ import annotations

import copy
import csv
from decimal import Decimal
import importlib.util
from pathlib import Path
import sqlite3
import tempfile
import unittest

spec=importlib.util.spec_from_file_location('eight_trace_analysis',Path(__file__).with_name('analyze_eight.py'))
analysis=importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)
BASE=1_791_889_000_000_000_000


def fixture(root,rank,marked=True):
    folder=Path(root)/f'diffusion_rank{rank}'/'capture'/'ASCEND_PROFILER_OUTPUT'
    folder.mkdir(parents=True)
    path=folder/f'ascend_pytorch_profiler_{rank}.db'
    db=sqlite3.connect(path)
    db.executescript('''
        CREATE TABLE STRING_IDS(id INTEGER,value TEXT);
        CREATE TABLE PYTORCH_API(startNs TEXT,endNs TEXT,globalTid INTEGER,connectionId INTEGER,name INTEGER,inputShapes INTEGER,inputDtypes INTEGER);
        CREATE TABLE COMMUNICATION_OP(startNs INTEGER,endNs INTEGER,connectionId INTEGER,opName INTEGER,groupName INTEGER,count NUMERIC,opId INTEGER);
        CREATE TABLE CANN_API(startNs INTEGER,endNs INTEGER,globalTid INTEGER,connectionId INTEGER,name INTEGER);
        CREATE TABLE CONNECTION_IDS(id INTEGER,connectionId INTEGER);
        CREATE TABLE TASK(startNs INTEGER,endNs INTEGER,streamId INTEGER,taskId INTEGER,globalTaskId INTEGER,connectionId INTEGER,taskType INTEGER);
    ''')
    names={1:'lingbot.rgb_copy',2:'lingbot.denoise_step',3:'lingbot_recovery_send',4:'lingbot_recovery_recv',
           5:'hcom_alltoall__current_123',6:'HcclAlltoAll',7:'c10d::alltoall_base_',8:'EVENT_WAIT',9:'EVENT_RECORD',
           10:'lingbot_recovery_v1::qk_send',11:'aten::_to_copy',12:'9,480,832,3;\r\n;\r\n',
           13:'12,480,832,3;\r\n;\r\n',14:'unsigned char;\r\nScalar',15:'current-group',16:'KERNEL',17:'aclrtStreamWaitEvent'}
    db.executemany('INSERT INTO STRING_IDS VALUES(?,?)',names.items())
    marker_name=1 if marked else 11
    for chunk in range(16):
        db.execute('INSERT INTO PYTORCH_API VALUES(?,?,?,?,?,?,?)',
            (str(BASE+chunk*1_000_000),str(BASE+(chunk+1)*1_000_000),123,None,marker_name,12 if chunk==0 else 13,14))
    start=BASE+11_000_000
    db.execute('INSERT INTO COMMUNICATION_OP VALUES(?,?,?,?,?,?,?)',(start+100+rank*10,start+700,10000,5,15,1024,42))
    db.execute('INSERT INTO CANN_API VALUES(?,?,?,?,?)',(start+40,start+45,123,10000,6))
    db.execute('INSERT INTO CONNECTION_IDS VALUES(?,?)',(555,10000))
    db.execute('INSERT INTO PYTORCH_API VALUES(?,?,?,?,?,?,?)',(str(start-1000),str(start+30),123,555,7,None,None))
    db.execute('INSERT INTO PYTORCH_API VALUES(?,?,?,?,?,?,?)',(str(start-2000),str(start+900),123,None,2,None,None))
    db.execute('INSERT INTO TASK VALUES(?,?,?,?,?,?,?)',(start+50,start+90,17,1,1,10001,16))
    db.execute('INSERT INTO TASK VALUES(?,?,?,?,?,?,?)',(start+710,start+800,17,2,2,10002,16))
    db.execute('INSERT INTO TASK VALUES(?,?,?,?,?,?,?)',(start+95,start+700,17,3,3,10003,8))
    db.execute('INSERT INTO CANN_API VALUES(?,?,?,?,?)',(start+46,start+49,123,10003,17))
    db.commit();db.close()
    with (folder/'kernel_details.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['Name','Type','Start Time(us)','Duration(us)','Stream ID','Task ID'])
        writer.writeheader()
        for name,begin,end,stream_id,task in (
            ('lingbot_recovery_send',start+50,start+90,17,1),
            ('hcom_alltoall__current_123',start+100+rank*10,start+700,31,4),
            ('lingbot_recovery_recv',start+710,start+800,17,2)):
            writer.writerow(dict(Name=name,Type='custom',**{'Start Time(us)':str(Decimal(begin)/1000),
                'Duration(us)':str(Decimal(end-begin)/1000),'Stream ID':stream_id,'Task ID':task}))
    return path


class IntervalAndPrecision(unittest.TestCase):
    def test_epoch_precision_uses_decimal_nanoseconds(self):
        self.assertEqual(analysis.us_to_ns('1791858225205903.750\t'),1791858225205903750)
        self.assertEqual(analysis.us_to_ns('0.001'),1)
        with self.assertRaises(ValueError):analysis.us_to_ns('0.0001')
        with self.assertRaises(ValueError):analysis.us_to_ns('NaN')

    def test_clipped_unions_do_not_double_count_duplicates_or_waits(self):
        kernels=[dict(start_ns=-10,end_ns=70,communication=False),
                 dict(start_ns=20,end_ns=70,communication=False),
                 dict(start_ns=50,end_ns=120,communication=True)]
        result=analysis.partition(kernels,0,100)
        self.assertEqual(result,dict(window_ns=100,compute_union_ns=70,communication_union_ns=50,
            overlap_ns=20,compute_only_ns=50,communication_only_ns=30,neither_ns=0,kernel_union_ns=100))
        self.assertEqual(analysis.union_ns([(1,2),(4,5)]),2)


class CollectiveIdentity(unittest.TestCase):
    def setUp(self):
        self.windows=[dict(chunk=11,start_ns=100,end_ns=200)]
        self.row=dict(group='group',name='alltoall__dynamic_9',count_key='1024',
                      start_ns=110,end_ns=150,boundary=False,connectionId=9,opId=1)
        self.ranks={rank:[dict(self.row,start_ns=110+rank)] for rank in range(8)}

    def test_duplicate_same_key_is_reported_and_excluded(self):
        self.ranks[3].append(dict(self.row,opId=2))
        result=analysis.match_collectives(self.ranks,self.windows)
        self.assertEqual(result['matched'],[])
        self.assertEqual(result['duplicate_keys'][0]['per_rank_counts'],{3:2})

    def test_missing_rank_is_not_matched_using_seven(self):
        del self.ranks[7]
        result=analysis.match_collectives(self.ranks,self.windows)
        self.assertFalse(result['matched'])
        self.assertEqual(result['missing_keys'][0]['missing_ranks'],[7])

    def test_boundary_or_unresolved_key_is_not_guessed(self):
        self.ranks[2][0].update(start_ns=95,boundary=True)
        result=analysis.match_collectives(self.ranks,self.windows)
        self.assertFalse(result['matched'])
        self.assertEqual(result['excluded_boundary_keys'][0]['boundary_ranks'],[2])
        self.ranks[2][0].update(start_ns=112,boundary=False,key_resolved=False)
        result=analysis.match_collectives(self.ranks,self.windows)
        self.assertFalse(result['matched'])
        self.assertEqual(len(result['unresolved_keys']),1)


class ExportAnalysis(unittest.TestCase):
    def test_real_schema_join_and_stream_derivation(self):
        with tempfile.TemporaryDirectory() as root:
            for rank in range(8):fixture(root,rank)
            result=analysis.analyze(root,top=1,sync_top=0)
            self.assertTrue(result['complete_eight_rank_evidence'])
            self.assertEqual(result['collective_matching']['matched_count'],1)
            target=result['top_exported_start_spreads'][0]
            self.assertEqual(target['target']['exported_start_spread_ns'],70)
            self.assertEqual(target['target']['chunk'],11)
            detail=target['per_rank_dependencies'][0]
            self.assertEqual(detail['derived_compute_streams'],[17])
            self.assertEqual(detail['cann_api_link']['status'],'unique')
            self.assertEqual(detail['pytorch_api_link']['candidates'][0]['name'],'c10d::alltoall_base_')
            self.assertEqual(detail['send_task_link']['status'],'unique_stream_task_interval')
            self.assertEqual(detail['recv_task_link']['status'],'unique_stream_task_interval')
            self.assertFalse(detail['event_handle_identity_proven'])
            self.assertFalse(detail['task_wait_durations_added_to_latency'])
            # TASK wait605ns is not added to the130ns noncomm CSV work.
            self.assertEqual(result['per_rank'][0]['per_chunk'][0]['partition']['compute_union_ns'],130)

    def test_missing_export_is_reported_without_partial_eight_rank_match(self):
        with tempfile.TemporaryDirectory() as root:
            for rank in range(7):fixture(root,rank)
            result=analysis.analyze(root,top=0,sync_top=0)
            self.assertFalse(result['complete_eight_rank_evidence'])
            self.assertEqual(result['collective_matching']['matched_count'],0)
            self.assertEqual(result['export_issues'][0]['rank'],7)

    def test_default_does_not_silently_reconstruct_missing_markers(self):
        with tempfile.TemporaryDirectory() as root:
            for rank in range(8):fixture(root,rank,marked=False)
            result=analysis.analyze(root,top=0,sync_top=0)
            self.assertIn('Insufficient RGB markers',result['window_error'])
            self.assertFalse(result['complete_eight_rank_evidence'])
            result=analysis.analyze(root,top=0,sync_top=0,boundary_mode='rgb-uint8-copy',expected_chunks=16)
            self.assertTrue(result['marker_evidence']['boundary_reconstructed'])
            self.assertFalse(result['marker_evidence']['direction_is_D2H_proven'])
            self.assertFalse(result['marker_evidence']['complete_chunk_boundaries_independently_proven'])

    def test_reconstruction_refuses_count_thread_or_frame_pattern_change(self):
        for field,value in (('globalTid',456),('inputShapes',12)):
            with tempfile.TemporaryDirectory() as root:
                path=fixture(root,0,marked=False)
                with analysis.open_db(path) as db:
                    strings=analysis.read_strings(db)
                    with self.assertRaises(ValueError):analysis.uint8_copy_windows(db,strings,11,14,15)
                db=sqlite3.connect(path)
                db.execute('UPDATE PYTORCH_API SET '+field+'=? WHERE rowid=2',(value,));db.commit();db.close()
                with analysis.open_db(path) as db:
                    with self.assertRaises(ValueError):analysis.uint8_copy_windows(db,analysis.read_strings(db),11,14,16)

    def test_expected_send_recv_coverage_is_enforced(self):
        with tempfile.TemporaryDirectory() as root:
            for rank in range(8):fixture(root,rank)
            result=analysis.analyze(root,first=11,last=11,top=0,sync_top=0,expected_layer_calls=1)
            self.assertTrue(result['complete_eight_rank_evidence'])
            result=analysis.analyze(root,first=11,last=11,top=0,sync_top=0,expected_layer_calls=160)
            self.assertFalse(result['complete_eight_rank_evidence'])
            self.assertFalse(result['per_rank'][0]['send_recv_coverage'][0]['count_gate_passed'])

    def test_exact_alltoall_count_rejects_missing_and_duplicate_collective(self):
        with tempfile.TemporaryDirectory() as root:
            for rank in range(8):fixture(root,rank)
            result=analysis.analyze(root,first=11,last=11,top=0,sync_top=0,expected_alltoall_calls=1)
            self.assertTrue(result['complete_eight_rank_evidence'])
            result=analysis.analyze(root,first=11,last=11,top=0,sync_top=0,expected_alltoall_calls=2)
            self.assertFalse(result['complete_eight_rank_evidence'])
            self.assertFalse(result['alltoall_count_gate_passed'])
            path=next((Path(root)/'diffusion_rank3').glob('*/ASCEND_PROFILER_OUTPUT/*.db'))
            db=sqlite3.connect(path)
            db.execute('INSERT INTO COMMUNICATION_OP SELECT * FROM COMMUNICATION_OP')
            db.commit();db.close()
            result=analysis.analyze(root,first=11,last=11,top=0,sync_top=0,expected_alltoall_calls=1)
            self.assertFalse(result['complete_eight_rank_evidence'])
            self.assertTrue(result['collective_matching']['duplicate_keys'])


if __name__=='__main__':unittest.main()
