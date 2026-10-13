"""CPU-only synthetic tests for predeclared correlated-rank QK timing metric."""
import copy
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('qk_repeatable_matrix_aggregate_cpu',Path(__file__).with_name('aggregate_qk_repeatable_matrix.py'))
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


def case_records(baseline,pool):
    result=[]
    for rank in range(8):
        result.append(dict(records={scope:dict(baseline=record(baseline[rank],scope,rank),
                                              pool=record(pool[rank],scope,rank)) for scope in agg.SCOPES}))
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
        pool=[[value//2 for value in values] for values in baseline]
        case,_,_=agg.summarize_case(case_records(baseline,pool),agg.SCOPES[0],4680,0)
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
        self.assertEqual(case['arms']['pool']['case_metric_ms'],.9)
        self.assertEqual(case['arms']['pool']['event_batch']['per_rank_event_mean_ms'],[100.0]*8)

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


def pool_state(submitted=3):
    return dict(capacity=24,initialized=24,workspace_initial_bytes=1898213376,
        workspace_resident_bytes=1898213376,submitted=submitted,callbacks_done=submitted,
        pending=0,pending_peak=2,event_queries=submitted-1,event_not_ready=0,
        plan_reuses=submitted-1,used_plan_peak=1,generations=submitted,pool_exhaustions=0,
        producer_tid=100,consumer_tid=101,accepting=1,closed=0,poisoned=0,last_error=0,last_step=-1,
        retained_call_slots=1,retirement_capacity=32,retired_count=1,retired_high_water=1,
        retired_total=submitted-1,retirement_overflows=0,producer_reclaimed_calls=submitted-2,
        producer_reclaim_tid=100,native_closed=0,finalizing=0,finalize_released_calls=0,finalize_tid=0)


def lifecycle():
    three,four,eight=pool_state(3),pool_state(4),pool_state(8)
    return dict(automatic_reclaim=dict(before=three,after=four),
        burst=dict(before=copy.deepcopy(four),after=eight,calls=4,per_call_device_drain=False,
                   inflight_coverage_claim=False),
        missing_snapshot=dict(before=copy.deepcopy(eight),after=copy.deepcopy(eight)))


def closed_pool():
    stats=pool_state(8)
    stats.update(workspace_resident_bytes=0,closed=1,native_closed=1,accepting=0,
        retained_call_slots=0,retired_count=0,producer_reclaimed_calls=7,
        finalize_released_calls=3,finalize_tid=100)
    return [dict(handle=1,stats=stats)]


class PoolEvidence(unittest.TestCase):
    def test_actual_reclaim_and_four_call_burst_accept_zero_not_ready(self):
        # A successful finite burst is accepted without inventing inflight proof.
        agg.validate_lifecycle(lifecycle())
        agg.validate_closed(closed_pool())
        maximum=pool_state(24)
        maximum.update(used_plan_peak=24,retained_call_slots=24)
        agg.validate_pool_state(maximum)
        for field in ('used_plan_peak','retained_call_slots'):
            over_capacity=copy.deepcopy(maximum);over_capacity[field]=25
            with self.assertRaisesRegex(ValueError,'Inconsistent drained pool counters'):
                agg.validate_pool_state(over_capacity)

    def test_prior_reclaim_counter_cannot_stand_in_for_third_call(self):
        evidence=lifecycle()
        evidence['automatic_reclaim']['after']['producer_reclaimed_calls']=1
        with self.assertRaisesRegex(ValueError,'Third production call'):
            agg.validate_lifecycle(evidence)

    def test_false_lifecycle_claims_or_failed_pool_rejected(self):
        for mutate in (
            lambda x:x['burst'].update(calls=3),
            lambda x:x['burst'].update(per_call_device_drain=True),
            lambda x:x['burst'].update(inflight_coverage_claim=True),
            lambda x:x['burst']['after'].update(capacity=64),
            lambda x:x['burst']['after'].update(capacity=8,initialized=8),
            lambda x:x['burst']['after'].update(poisoned=1),
            lambda x:x['automatic_reclaim']['after'].update(producer_reclaim_tid=101),
            lambda x:x['burst']['after'].update(event_queries=-1),
            lambda x:x['burst']['after'].update(submitted=7),
        ):
            evidence=lifecycle();mutate(evidence)
            with self.assertRaises(ValueError):agg.validate_lifecycle(evidence)

    def test_missing_snapshot_may_not_submit(self):
        evidence=lifecycle();evidence['missing_snapshot']['after']=pool_state(9)
        with self.assertRaisesRegex(ValueError,'Missing coefficient snapshot'):
            agg.validate_lifecycle(evidence)

    def test_native_close_without_producer_finalize_rejected(self):
        for mutate in (
            lambda x:x.update(closed=0),
            lambda x:x.update(workspace_resident_bytes=1024),
            lambda x:x.update(finalize_tid=101),
            lambda x:x.update(retained_call_slots=1),
            lambda x:x.update(retired_count=1),
            lambda x:x.update(pool_exhaustions=1),
            lambda x:x.update(last_error=-2),
        ):
            evidence=closed_pool();mutate(evidence[0]['stats'])
            with self.assertRaises(ValueError):agg.validate_closed(evidence)
        with self.assertRaises(ValueError):agg.validate_closed(closed_pool()*2)

    def test_reports_must_bind_one_supervisor_nonce(self):
        reports=[dict(supervisor_nonce='launch-one') for _ in range(8)]
        agg.bind_launch(reports,dict(nonce='launch-one'))
        reports[4]['supervisor_nonce']='different-launch'
        with self.assertRaisesRegex(ValueError,'same supervisor launch'):
            agg.bind_launch(reports,dict(nonce='launch-one'))

    def test_each_scope_must_pass(self):
        reports=reports_for_case_ratios([.8]*24)
        # Change only the cached scope: averaging scopes must not conceal failure.
        for rank,report in enumerate(reports):
            for case in report['cases']:
                case['records'][agg.SCOPES[1]]['pool']=record([1_000_000]*30,agg.SCOPES[1],rank)
        summary=agg.summarize_round(reports,1)
        self.assertTrue(summary['scopes'][agg.SCOPES[0]]['scope_gate_passed'])
        self.assertFalse(summary['scopes'][agg.SCOPES[1]]['scope_gate_passed'])
        self.assertFalse(summary['round_gate_passed'])

    def test_three_distinct_rounds_all_required_no_cross_round_pooling(self):
        # Exercise orchestration with synthetic metadata, while separate tests
        # above exercise real integer timing formulas and lifecycle validators.
        fixture=dict(path='/fixture',bytes=1,sha256='0'*64)
        manifest=dict(path='/manifest',bytes=1,sha256='1'*64)
        source_pins=dict(harness=fixture)
        matrix=dict(sha256='matrix',root='/fixtures')
        correct=[];correct_pins=[];timing={}
        for rank in range(8):
            case=dict(identity=dict(rank=rank),fixture=fixture,captured_provider_sha256='capture')
            base=dict(rank=rank,round=1,matrix_rank=dict(rank=rank),matrix_sha256='matrix',
                      manifest=manifest,sources=source_pins,cases=[case])
            correct.append(dict(base,supervisor_nonce='correctness'))
            correct_pins.append(dict(path=f'/correct/rank{rank}.json',bytes=1,sha256='2'*64))
        for number in range(1,4):
            for rank in range(8):
                report=copy.deepcopy(correct[rank])
                report.update(round=number,correctness_receipts=correct_pins,supervisor_nonce=f'round{number}')
                timing[f'/timing{number}/rank{rank}.json']=report
        def read(path):
            path=str(path)
            if path.startswith('/correct/'):
                rank=int(Path(path).stem.removeprefix('rank'))
                return correct[rank],correct_pins[rank]
            return timing[path],fixture
        def supervise(run,*args,**kwargs):
            return dict(nonce='correctness' if str(run)=='/correct-run' else Path(run).name)
        with patch.object(agg,'read_pinned',side_effect=read), \
             patch.object(agg,'validate_report'), \
             patch.object(agg,'reconstruct_matrix',return_value=matrix), \
             patch.object(agg,'supervisor',side_effect=supervise), \
             patch.object(agg,'summarize_round',side_effect=[
                 dict(round=1,round_gate_passed=True),dict(round=2,round_gate_passed=False),
                 dict(round=3,round_gate_passed=True)]):
            result=agg.aggregate('/correct','/correct-run',
                                 ['/timing1','/timing2','/timing3'],['/round1','/round2','/round3'])
        self.assertFalse(result['strict_performance_screen_passed'])
        self.assertFalse(result['sample_interpretation']['cross_round_pooling'])
        self.assertFalse(result['model_admitted'])
        self.assertFalse(result['lifecycle_scope']['inflight_coverage_claim'])
        with self.assertRaises(ValueError):
            agg.aggregate('/correct','/correct-run',['/timing1']*3,['/round1','/round2','/round3'])


if __name__=='__main__':unittest.main()
