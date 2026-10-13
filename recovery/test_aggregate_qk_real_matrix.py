"""CPU-only synthetic tests for predeclared correlated-rank QK timing metric."""
import copy
import importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('qk_matrix_aggregate_cpu',Path(__file__).with_name('aggregate_qk_real_matrix.py'))
agg=importlib.util.module_from_spec(spec);spec.loader.exec_module(agg)


def record(durations,scope,rank=0,event_ms=100.0):
    wall=[];events=[]
    for i,duration in enumerate(durations):
        start=1_000_000_000+i*1_000_000_000+rank*1000
        wall.append(dict(index=i,start_ns=start,return_ns=start+duration//2,end_ns=start+duration,wall_ms=duration/1e6))
        start=100_000_000_000+i*1_000_000_000+rank*1000
        events.append(dict(index=i,start_ns=start,return_ns=start+duration//2,end_ns=start+duration,event_ms=event_ms))
    return dict(warmup=20,events_are_separate_batch=True,generation_start=1,
        generation_end=81 if scope==agg.SCOPES[0] else 1,wall_samples=wall,event_samples=events,
        phase_boundaries=[dict(name='warm20',enter_ns=0,ready_ns=1),dict(name='wall30',enter_ns=2,ready_ns=3),
                          dict(name='event30_separate_batch',enter_ns=90_000_000_000,ready_ns=90_000_000_001)])


def case_records(baseline,combined):
    result=[]
    for rank in range(8):
        result.append(dict(records={scope:dict(baseline=record(baseline[rank],scope,rank),
                                              combined=record(combined[rank],scope,rank)) for scope in agg.SCOPES}))
    return result


def reports_for_case_ratios(ratios):
    reports=[dict(boot_id='one-boot',cases=[]) for _ in range(8)]
    for ratio in ratios:
        rows=case_records([[1_000_000]*30 for _ in range(8)],[[int(1_000_000*ratio)]*30 for _ in range(8)])
        for rank in range(8):reports[rank]['cases'].append(rows[rank])
    return reports


class PredefinedMetric(unittest.TestCase):
    def test_max_rank_mean_not_mean_of_per_sample_max(self):
        baseline=[[1_000_000]*30 for _ in range(8)]
        baseline[0]=[100_000_000,1_000_000]*15;baseline[1]=[1_000_000,100_000_000]*15
        combined=[[value//2 for value in values] for values in baseline]
        case,_,_=agg.summarize_case(case_records(baseline,combined),agg.SCOPES[0],4680,0)
        self.assertEqual(case['arms']['baseline']['case_metric_ms'],50.5)
        self.assertEqual(case['arms']['baseline']['wall_batch']['samples'][0]['slowest_rank_call_ms'],100)
        self.assertEqual(case['case_ratio'],.5)
        self.assertEqual(case['arms']['baseline']['event_batch']['per_rank_event_mean_ms'][0],100)
        self.assertEqual(case['arms']['baseline']['wall_batch']['correlated_collective_samples'],30)

    def test_exact_aggregate_boundary_and_all_outliers_retained(self):
        result=agg.summarize_round(reports_for_case_ratios([.9]*24),1)
        self.assertTrue(result['round_gate_passed'])
        result=agg.summarize_round(reports_for_case_ratios([.8]*23+[1.10]),1)
        scope=result['scopes'][agg.SCOPES[0]]
        self.assertTrue(scope['aggregate_10_percent_gate_passed'])
        self.assertFalse(scope['all24_case_5_percent_gate_passed'])
        self.assertFalse(result['round_gate_passed'])
        self.assertEqual(scope['worst_case']['ratio'],1.10)

    def test_per_rank_regression_report_is_separate_from_case_gate(self):
        rows=case_records([[2_000_000]*30]+[[1_000_000]*30 for _ in range(7)],
                          [[1_500_000]*30 for _ in range(8)])
        case,_,_=agg.summarize_case(rows,agg.SCOPES[1],4680,0)
        self.assertEqual(case['case_ratio'],.75)
        self.assertEqual(case['per_rank_ratios'][1],1.5)
        self.assertTrue(case['case_worst_regression_gate_passed'])

    def test_event_batch_never_added_to_complete_wall_call(self):
        rows=case_records([[1_000_000]*30 for _ in range(8)],[[900_000]*30 for _ in range(8)])
        case,_,_=agg.summarize_case(rows,agg.SCOPES[0],4680,0)
        self.assertEqual(case['arms']['combined']['case_metric_ms'],.9)
        self.assertEqual(case['arms']['combined']['event_batch']['per_rank_event_mean_ms'],[100.0]*8)

    def test_bad_samples_and_producer_lifecycle_fail(self):
        valid=record([1_000_000]*30,agg.SCOPES[0])
        for mutate in (lambda x:x['wall_samples'].pop(),
                       lambda x:x['wall_samples'][1].update(index=0),
                       lambda x:x['wall_samples'][0].update(wall_ms=float('nan')),
                       lambda x:x.update(generation_end=1),
                       lambda x:x.update(events_are_separate_batch=False)):
            candidate=copy.deepcopy(valid);mutate(candidate)
            with self.assertRaises(ValueError):agg.validate_batch(candidate,agg.SCOPES[0])

    def test_host_union_not_sum_or_envelope_with_gaps(self):
        self.assertEqual(agg.interval_union_ns([(1,4),(3,6)]),5)
        self.assertEqual(agg.interval_union_ns([(1,2),(4,5)]),2)

    def test_clock_domain_mismatch_rejected(self):
        reports=reports_for_case_ratios([.8]*24);reports[3]['boot_id']='different-boot'
        with self.assertRaisesRegex(ValueError,'same host boot'):agg.summarize_round(reports,1)


if __name__=='__main__':unittest.main()
