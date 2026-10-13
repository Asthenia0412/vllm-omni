"""CPU-only aggregation of three frozen eight-rank QK/HCCL timing rounds.

Predefined gate: per case/arm, max_rank(mean of30 wall calls); equal-weight
24-case aggregate. Both separate scopes in each of three rounds must satisfy
aggregate B/A<=0.90 and every case B/A<=1.05. Event batches and cross-rank
host envelopes are descriptive only. No statistical or model-speed claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

SCHEMA='lingbot_qk_repeatable_matrix_replay_v1'
METHOD='phase_barrier_then_sync_per_call_sync_v2'
LENGTHS=(4680,9360,14040,18720,23400,28080)
EXPECTED=tuple((length,probe) for length in LENGTHS for probe in range(4))
SCOPES=('producer_plus_layer0_and_original_consumer','cached_snapshot_and_original_consumer')
ARMS=('baseline','pool')
CHECKS=('expanded_coefficients','captured_reference_hccl','baseline_send_and_receive',
        'sentinel_send','full_arena_after_pack','sentinel_receive','full_arena_after_hccl',
        'fresh_owner_extent_offset_and_distinctness','fresh_sends_and_receives','first_owner_not_overwritten',
        'full_source_and_reference_owners','source_descriptors_unchanged',
        'missing_snapshot_refuses_without_producer_mutation','production_run_auto_reclaim',
        'burst_four_fresh_outputs_and_original_hccl')
SOURCE_KEYS={'harness','loader','capture_contract','original_collective','coefficient_producer',
             'baseline_wrapper','pool_wrapper','padded_consumer_hook'}


def need(value,message):
    if not value:raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def read_pinned(path):
    path=Path(path).resolve(strict=True);raw=path.read_bytes()
    return json.loads(raw),dict(path=str(path),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())


def validate_pin(pin):
    need(set(pin)=={'path','bytes','sha256'} and Path(pin['path']).is_absolute()
         and type(pin['bytes']) is int and pin['bytes']>0 and isinstance(pin['sha256'],str)
         and len(pin['sha256'])==64 and all(c in '0123456789abcdef' for c in pin['sha256']), 'Invalid artifact pin')


def key(case):return case['identity']['kv_len'],case['identity']['probe']


def validate_pool_state(stats,closed=False):
    """Bound the observed pool state; counters do not prove last destructor TID."""
    fields=('capacity','initialized','workspace_initial_bytes','workspace_resident_bytes',
            'submitted','callbacks_done','pending','pending_peak','event_queries','event_not_ready',
            'plan_reuses','used_plan_peak','generations','pool_exhaustions','producer_tid',
            'consumer_tid','accepting','closed','poisoned','retained_call_slots',
            'retirement_capacity','retired_count','retired_high_water','retired_total',
            'retirement_overflows','producer_reclaimed_calls','producer_reclaim_tid',
            'native_closed','finalizing','finalize_released_calls','finalize_tid')
    need(all(type(stats.get(name)) is int and stats[name]>=0 for name in fields),
         'Missing/negative/noninteger pool lifecycle counter')
    need(stats['capacity']==stats['initialized']==8 and stats['retirement_capacity']==32,
         'Pool capacity differs from frozen eight-plan/32-retirement contract')
    need(stats['workspace_initial_bytes']>0 and stats['producer_tid']>0 and stats['consumer_tid']>0,
         'Pool was not initialized/executed on identified threads')
    need(stats['submitted']==stats['callbacks_done']==stats['generations']>0
         and stats['event_not_ready']<=stats['event_queries']
         and 1<=stats['used_plan_peak']<=8 and stats['retained_call_slots']<=8
         and stats['retired_count']<=stats['retired_high_water']<=32
         and stats['producer_reclaimed_calls']<=stats['retired_total'],
         'Inconsistent drained pool counters')
    need(all(stats[name]==0 for name in ('pending','poisoned','pool_exhaustions','retirement_overflows','finalizing')),
         'Pool snapshot contains pending or failed work')
    need(stats.get('last_error')==0 and stats.get('last_step')==-1,
         'Pool recorded a native/lifecycle error')
    if closed:
        need(stats['closed']==stats['native_closed']==1
             and stats['finalize_tid']==stats['producer_tid']
             and stats['finalize_released_calls']>0
             and all(stats[name]==0 for name in ('workspace_resident_bytes','retained_call_slots',
                                                'retired_count','accepting')),
             'Pool cleanup retained owners/workspace or finalized on the wrong thread')
    else:
        need(stats['accepting']==1 and stats['workspace_resident_bytes']==stats['workspace_initial_bytes']
             and stats['closed']==stats['native_closed']==stats['finalize_tid']==0,
             'Live pool snapshot is closed or released workspace early')


def validate_closed(snapshots):
    need(isinstance(snapshots,list) and len(snapshots)==1,
         'Exactly one provider-wide closed pool required')
    item=snapshots[0]
    need(type(item.get('handle')) is int and item['handle']>0,'Invalid closed pool handle')
    validate_pool_state(item['stats'],closed=True)


def validate_lifecycle(lifecycle):
    need(set(lifecycle)=={'automatic_reclaim','burst','missing_snapshot'},
         'Missing independent pool lifecycle checks')
    for stage in lifecycle.values():
        before,after=stage['before'],stage['after']
        validate_pool_state(before);validate_pool_state(after)
        need(all(before[name]==after[name] for name in ('producer_tid','consumer_tid','workspace_initial_bytes')),
             'Pool identity changed across lifecycle stage')
        need(all(before[name]<=after[name] for name in ('submitted','callbacks_done','generations',
                'event_queries','event_not_ready','plan_reuses','used_plan_peak','retired_total',
                'producer_reclaimed_calls','retired_high_water','pending_peak')),
             'Pool lifecycle counters moved backwards')
    auto=lifecycle['automatic_reclaim'];before,after=auto['before'],auto['after']
    need(after['submitted']-before['submitted']==1
         and after['producer_reclaimed_calls']>before['producer_reclaimed_calls']
         and after['producer_reclaim_tid']==after['producer_tid'],
         'Third production call did not exercise automatic producer reference release')
    burst=lifecycle['burst'];before,after=burst['before'],burst['after']
    need(before==auto['after'] and burst.get('calls')==4
         and burst.get('per_call_device_drain') is False and burst.get('inflight_coverage_claim') is False
         and after['submitted']-before['submitted']==4,
         'Four-call original-consumer burst contract differs')
    missing=lifecycle['missing_snapshot']
    need(missing['before']==burst['after']
         and missing['before']['submitted']==missing['after']['submitted'],
         'Missing coefficient snapshot submitted native work')


def validate_report(report,rank,mode,round_number=None):
    need(report.get('schema')==SCHEMA and report.get('method')==METHOD
         and report.get('mode')==mode and report.get('rank')==rank and report.get('passed') is True,
         'Unsuccessful/unsupported rank report')
    need(isinstance(report.get('supervisor_nonce'),str) and bool(report['supervisor_nonce']),
         'Missing launch-owned supervisor nonce')
    if round_number is not None:need(report.get('round')==round_number,'Mixed timing round')
    need(report.get('original_hccl_included') is True and report.get('actual_payload_cases_per_rank')==24
         and report.get('real_payload_layers')==[0] and report.get('all40_payloads') is False
         and report.get('model_admitted') is False and report.get('gate_passed') is False,
         'Rank report scope/admission differs')
    need(set(report['sources'])==SOURCE_KEYS,'Missing frozen source pins')
    validate_pin(report['manifest'])
    for pin in report['sources'].values():validate_pin(pin)
    validate_closed(report.get('pool_after_close',[]))
    cases=report['cases'];need(len(cases)==24 and tuple(map(key,cases))==EXPECTED,'Missing/duplicate/unordered real24 cases')
    matrix_rank=report['matrix_rank']
    need(matrix_rank['rank']==rank and len(matrix_rank['cases'])==24,'Invalid bound matrix rank')
    validate_pin(matrix_rank['coverage'])
    for case,bound in zip(cases,matrix_rank['cases']):
        validate_pin(case['fixture'])
        need(case['identity']==bound['identity'] and case['fixture']==bound['file']
             and case['identity']['rank']==rank and case['identity']['layer']==0
             and case['identity']['coefficient_generation']>0 and case['passed'] is True
             and case['full_owner_sha256_verified'] is True,'Case differs from real matrix or failed correctness')
        if mode=='correctness':
            need(case['checks']==list(CHECKS),'Incomplete candidate correctness checks')
            validate_lifecycle(case.get('lifecycle',{}))
        else:
            need(set(case['records'])==set(SCOPES),'Missing complete-call scope')
            for scope in SCOPES:
                need(set(case['records'][scope])==set(ARMS),'Missing timing arm')
    if mode=='timing':
        need(report.get('actual192_candidate_correctness_global') is True,'Timing did not bind global correctness')
    need(isinstance(report.get('boot_id'),str) and report['boot_id']
         and report.get('host_clock')=='time.perf_counter_ns: same-host monotonic clock; no cross-host epoch claim'
         and report.get('physical_devices')=='8,9,10,11,12,13,14,15','Unsupported clock/device domain')


def reconstruct_matrix(reports):
    roots={str(Path(r['matrix_rank']['coverage']['path']).parent.parent) for r in reports}
    captures={case['captured_provider_sha256'] for r in reports for case in r['cases']}
    need(len(roots)==1 and len(captures)==1,'Mixed fixture root/captured build')
    matrix=dict(root=next(iter(roots)),ranks=[r['matrix_rank'] for r in reports],
                captured_provider_sha256=next(iter(captures)),actual_qk_payload_cases=192,
                real_payload_layers=[0],all40_payloads=False)
    sha=digest(matrix)
    need(all(r['matrix_sha256']==sha for r in reports),'Matrix digest does not bind all eight actual ranks')
    locations=[[(c['identity']['kv_len'],c['identity']['probe'],c['identity']['request_id'],c['identity']['chunk'])
                for c in r['cases']] for r in reports]
    need(all(row==locations[0] for row in locations),'Rank actual request/chunk/history/probe occurrences differ')
    return dict(sha256=sha,root=matrix['root'],captured_provider_sha256=matrix['captured_provider_sha256'])


def supervisor(run,directory,mode,round_number,matrix,manifest,correctness_directory=None):
    launch,launch_pin=read_pinned(Path(run)/'launch.json');result,result_pin=read_pinned(Path(run)/'result.json')
    need(result.get('exit_code')==0 and result.get('fatal') is None
         and result.get('remaining_owned_processes')==[],'Supervisor was not a clean completed launch')
    for name in ('nonce','leader','command','mode','manifest'):
        need(result.get(name)==launch.get(name),'Supervisor launch/result identity mismatch')
    need(launch['mode']==mode,'Supervisor mode mismatch')
    command=launch['command']
    indices=[i for i,item in enumerate(command) if Path(item).name=='check_qk_repeatable_matrix.py']
    need(len(indices)==1,'Supervisor did not directly launch the frozen real-matrix harness')
    arguments=command[indices[0]+1:]
    def argument(name,default=None):
        need(arguments.count(name)<=1,'Duplicate supervisor command argument '+name)
        return arguments[arguments.index(name)+1] if name in arguments else default
    need('--execute' in arguments and argument('--mode','correctness')==mode
         and int(argument('--round','1'))==round_number,'Wrong executed mode/round')
    for name,wanted in (('--output',directory),('--fixtures',matrix['root']),('--manifest',manifest['path'])):
        need(str(Path(argument(name,'')).resolve())==str(Path(wanted).resolve()),'Supervisor command does not bind '+name)
    if correctness_directory is not None:
        need(str(Path(argument('--correctness-dir','')).resolve())==str(Path(correctness_directory).resolve()),
             'Timing supervisor uses a different correctness receipt directory')
    candidate,candidate_pin=read_pinned(manifest['path'])
    need(candidate_pin==manifest,'Candidate build manifest differs from reports')
    provider,provider_pin=read_pinned(candidate['provider_manifest']['path'])
    need(provider_pin==candidate['provider_manifest']
         and str(Path(launch['manifest']).resolve())==provider_pin['path'], 'Supervisor/provider build mismatch')
    return dict(launch=launch_pin,result=result_pin,nonce=launch['nonce'],leader=launch['leader'],clean_exit=True)


def bind_launch(reports,supervisor_receipt):
    need(len(reports)==8 and all(report.get('supervisor_nonce')==supervisor_receipt['nonce'] for report in reports),
         'Rank reports do not belong to the same supervisor launch')


def validate_batch(record,scope):
    need(record.get('warmup')==20 and record.get('events_are_separate_batch') is True,'Wrong warmup/batch contract')
    boundaries=record['phase_boundaries']
    need([item['name'] for item in boundaries]==['warm20','wall30','event30_separate_batch'], 'Wrong phase boundaries')
    need(all(type(item[field]) is int for item in boundaries for field in ('enter_ns','ready_ns'))
         and all(item['enter_ns']<=item['ready_ns'] for item in boundaries),'Invalid phase timestamps')
    need(all(boundaries[i]['ready_ns']<=boundaries[i+1]['enter_ns'] for i in range(2)), 'Phase order differs')
    expected_generation=80 if scope==SCOPES[0] else 0
    need(type(record['generation_start']) is int and type(record['generation_end']) is int
         and record['generation_end']-record['generation_start']==expected_generation,
         'Coefficient producer lifecycle differs from declared scope')
    for batch,metric,boundary in (('wall_samples','wall_ms',boundaries[1]),('event_samples','event_ms',boundaries[2])):
        samples=record[batch]
        need(len(samples)==30 and [s['index'] for s in samples]==list(range(30)), 'Missing/duplicate sample index')
        previous=boundary['ready_ns']
        for sample in samples:
            need(all(type(sample[name]) is int for name in ('start_ns','return_ns','end_ns'))
                 and previous<=sample['start_ns']<=sample['return_ns']<=sample['end_ns'], 'Invalid/overlapping rank sample clock')
            previous=sample['end_ns']
            value=sample[metric]
            need(type(value) in (float,int) and math.isfinite(value) and value>=0,'Nonfinite/negative timing sample')
            if batch=='wall_samples':
                duration=sample['end_ns']-sample['start_ns']
                need(duration>0 and math.isclose(value,duration/1e6,rel_tol=1e-12,abs_tol=1e-12),
                     'Wall value disagrees with recorded complete-call timestamps')
        if batch=='wall_samples':
            need(samples[-1]['end_ns']<=boundaries[2]['enter_ns'],'Wall and event batches overlap')


def interval_union_ns(intervals):
    total=0;start=end=None
    for left,right in sorted(intervals):
        if start is None:start,end=left,right
        elif left<=end:end=max(end,right)
        else:total+=end-start;start,end=left,right
    return total+(end-start if start is not None else 0)


def summarize_batch(records,batch):
    """Join each correlated collective sample index across ranks, never flatten240."""
    samples=[];wall_sums=[0]*8;event_sums=[0.0]*8
    for index in range(30):
        rows=[record[batch][index] for record in records]
        starts=[s['start_ns'] for s in rows];returns=[s['return_ns'] for s in rows];ends=[s['end_ns'] for s in rows]
        durations=[end-start for start,end in zip(starts,ends)]
        for rank,duration in enumerate(durations):wall_sums[rank]+=duration
        event_values=[s['event_ms'] for s in rows] if batch=='event_samples' else None
        if event_values is not None:
            for rank,value in enumerate(event_values):event_sums[rank]+=value
        slow=max(durations)
        row=dict(index=index,host_arrival_skew_ms=(max(starts)-min(starts))/1e6,
                 host_return_skew_ms=(max(returns)-min(returns))/1e6,
                 host_end_skew_ms=(max(ends)-min(ends))/1e6,
                 host_wall_envelope_ms=(max(ends)-min(starts))/1e6,
                 host_wall_interval_union_ms=interval_union_ns(zip(starts,ends))/1e6,
                 slowest_rank_call_ms=slow/1e6,slowest_ranks=[r for r,d in enumerate(durations) if d==slow],
                 per_rank_wall_ms=[value/1e6 for value in durations])
        if event_values is not None:row['per_rank_event_ms']=event_values
        samples.append(row)
    result=dict(correlated_collective_samples=30,ranks_per_sample=8,
                independent_sample_claim=False,per_rank_wall_mean_ms=[value/30e6 for value in wall_sums],
                samples=samples,descriptive_only=batch=='event_samples')
    if batch=='event_samples':result['per_rank_event_mean_ms']=[value/30 for value in event_sums]
    return result,wall_sums


def summarize_case(case_records,scope,length,probe):
    """Case gate uses max of rank means, explicitly not mean of per-sample maxima."""
    result=dict(kv_len=length,probe=probe,scope=scope,arms={});sums={}
    for arm in ARMS:
        records=[case['records'][scope][arm] for case in case_records]
        for record in records:validate_batch(record,scope)
        wall,wall_sums=summarize_batch(records,'wall_samples');event,_=summarize_batch(records,'event_samples')
        sums[arm]=wall_sums;maximum=max(wall_sums)
        result['arms'][arm]=dict(case_metric_ms=maximum/30e6,slowest_mean_ranks=[r for r,v in enumerate(wall_sums) if v==maximum],
                                  wall_batch=wall,event_batch=event)
    baseline=max(sums['baseline']);pool=max(sums['pool'])
    need(baseline>0 and all(value>0 for value in sums['baseline']),'Zero complete-call baseline')
    result.update(case_ratio=pool/baseline,case_improvement_fraction=1-pool/baseline,
                  case_worst_regression_gate_passed=20*pool<=21*baseline,
                  per_rank_ratios=[b/a for a,b in zip(sums['baseline'],sums['pool'])])
    return result,baseline,pool


def summarize_round(reports,round_number):
    need(len({r['boot_id'] for r in reports})==1,'Rank timestamps are not from the same host boot')
    summary=dict(round=round_number,boot_id=reports[0]['boot_id'],scopes={})
    for scope in SCOPES:
        cases=[];baseline=pool=0;rank_ratios=[]
        for index,(length,probe) in enumerate(EXPECTED):
            case,a,b=summarize_case([report['cases'][index] for report in reports],scope,length,probe)
            baseline+=a;pool+=b;cases.append(case)
            rank_ratios.extend(dict(kv_len=length,probe=probe,rank=rank,ratio=ratio)
                               for rank,ratio in enumerate(case['per_rank_ratios']))
        aggregate_pass=10*pool<=9*baseline
        worst_pass=all(case['case_worst_regression_gate_passed'] for case in cases)
        worst=max(cases,key=lambda case:case['case_ratio'])
        summary['scopes'][scope]=dict(aggregate_baseline_ms=baseline/(24*30e6),
            aggregate_pool_ms=pool/(24*30e6),aggregate_ratio=pool/baseline,
            aggregate_improvement_fraction=1-pool/baseline,aggregate_10_percent_gate_passed=aggregate_pass,
            all24_case_5_percent_gate_passed=worst_pass,scope_gate_passed=aggregate_pass and worst_pass,
            worst_case=dict(kv_len=worst['kv_len'],probe=worst['probe'],ratio=worst['case_ratio']),
            worst_per_rank=max(rank_ratios,key=lambda row:row['ratio']),worst_per_rank_is_separate_diagnostic=True,
            equal_case_weight=1/24,cases=cases)
    summary['round_gate_passed']=all(s['scope_gate_passed'] for s in summary['scopes'].values())
    return summary


def aggregate(correctness_dir,correctness_run,timing_dirs,timing_runs):
    need(len(timing_dirs)==len(timing_runs)==3,'Exactly three independent timing launches required')
    need(len({str(Path(p).resolve()) for p in timing_dirs})==3 and len({str(Path(p).resolve()) for p in timing_runs})==3,
         'Timing launch directories cannot be reused')
    correct=[];correct_pins=[]
    for rank in range(8):
        report,pin=read_pinned(Path(correctness_dir)/f'rank{rank}.json')
        validate_report(report,rank,'correctness');correct.append(report);correct_pins.append(pin)
    matrix=reconstruct_matrix(correct);manifest=correct[0]['manifest'];sources=correct[0]['sources']
    need(all(r['manifest']==manifest and r['sources']==sources for r in correct),'Correctness build/source pins differ')
    correct_supervisor=supervisor(correctness_run,correctness_dir,'correctness',correct[0]['round'],matrix,manifest)
    bind_launch(correct,correct_supervisor)
    rounds=[];raw_reports=[];supervisors=[]
    for number,(directory,run) in enumerate(zip(timing_dirs,timing_runs),1):
        reports=[];pins=[]
        for rank in range(8):
            report,pin=read_pinned(Path(directory)/f'rank{rank}.json')
            validate_report(report,rank,'timing',number)
            need(report['matrix_sha256']==matrix['sha256'] and report['matrix_rank']==correct[rank]['matrix_rank']
                 and report['manifest']==manifest and report['sources']==sources
                 and report['correctness_receipts']==correct_pins,'Timing fixture/build/source/correctness evidence differs')
            for case,reference in zip(report['cases'],correct[rank]['cases']):
                need(case['identity']==reference['identity'] and case['fixture']==reference['fixture']
                     and case['captured_provider_sha256']==reference['captured_provider_sha256'], 'Timing case identity differs')
            reports.append(report);pins.append(pin)
        reconstruct_matrix(reports)
        supervisors.append(supervisor(run,directory,'timing',number,matrix,manifest,correctness_dir))
        bind_launch(reports,supervisors[-1])
        rounds.append(summarize_round(reports,number));raw_reports.append(dict(round=number,rank_reports=pins))
    need(len({s['nonce'] for s in [correct_supervisor,*supervisors]})==4,'Correctness/timing launch nonce reused')
    return dict(schema='lingbot_qk_repeatable_matrix_aggregate_v1',metric_definition=dict(
        case='max over eight ranks of that rank\'s arithmetic mean of all30 complete-call wall durations',
        round_scope='equal-weight arithmetic mean of24 real history/probe case metrics',
        aggregate_gate='pool/baseline<=0.90, compared exactly using integer nanosecond sums',
        case_gate='every one of24 case pool/baseline ratios<=1.05, exact integer comparison',
        scopes_must_pass_separately=list(SCOPES),all_three_rounds_required=True,
        worst_per_rank_ratio_is_not_case_gate=True,producer_not_multiplied_by40=True),
        strict_performance_screen_passed=all(r['round_gate_passed'] for r in rounds),
        actual192_candidate_correctness_global=True,fixture_matrix=matrix,manifest=manifest,sources=sources,
        correctness_report_pins=correct_pins,correctness_supervisor=correct_supervisor,
        timing_supervisors=supervisors,raw_timing_report_pins=raw_reports,rounds=rounds,
        sample_interpretation=dict(collective_samples_per_arm_scope_case_round=30,correlated_ranks_per_sample=8,
            flattened240_independent_samples=False,wall_and_event_are_separate_batches=True,
            timestamps_are_host_call_timestamps_not_device_collective_arrivals=True,
            envelope_and_interval_union_are_descriptive_not_transfer_or_speed_metrics=True,
            host_wait_and_device_event_time_added=False,outlier_removal=False,cross_round_pooling=False),
        statistical_significance_claim=False,end_to_end_improvement_claim=False,model_admitted=False,
        historical_single_host_identity_proven=False,real_payload_layers=[0],all40_payloads=False,
        lifecycle_scope=dict(production_allocator=True,automatic_pool_reference_reclaim=True,
            fixed_plan_capacity=8,fixed_retirement_capacity=32,provider_pool_count=1,
            burst_original_collectives=4,inflight_coverage_claim=False,forced_exhaustion_tested=False,
            last_tensor_destructor_thread_proven=False,two_phase_pool_close_verified=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--correctness-dir',type=Path,required=True)
    parser.add_argument('--correctness-run',type=Path,required=True)
    parser.add_argument('--timing-dirs',type=Path,nargs=3,required=True)
    parser.add_argument('--timing-runs',type=Path,nargs=3,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    need(not args.output.exists(),'Refuse to overwrite earlier aggregation')
    result=aggregate(args.correctness_dir,args.correctness_run,args.timing_dirs,args.timing_runs)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(args.output),strict_performance_screen_passed=result['strict_performance_screen_passed'],
        rounds=[dict(round=r['round'],passed=r['round_gate_passed']) for r in result['rounds']]),indent=2))
