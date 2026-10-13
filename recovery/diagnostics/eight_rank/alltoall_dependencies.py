"""Narrow read-only follow-up: two selected AlltoAlls and other common ops.

Reuses the frozen analyzer's joins. Does not rerun its12-target analysis or
modify its inputs. Communication duration includes waits, not transfer savings.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import csv
import importlib.util
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise ValueError('Refuse to overwrite earlier narrow evidence')
    original=json.loads(args.input.read_text())
    source=original['analyzer_source']
    spec=importlib.util.spec_from_file_location('frozen_eight_followup',source['path'])
    a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
    a.need(a.pin(source['path'])==source,'Frozen analyzer source differs')
    a.need(original['complete_eight_rank_evidence'] is True,'Complete parent eight-rank evidence required')
    common=original['collective_matching']['matched_collectives']
    candidates=[r for r in common if 'alltoall' in r['key']['operation'].lower()]
    targets=sorted(candidates,key=lambda r:-r['exported_start_spread_ns'])[:2]
    ranges=[(min(r['start_ns'] for r in t['per_rank'])-10_000_000,
             max(r['end_ns'] for r in t['per_rank'])+10_000_000) for t in targets]
    lo,hi=original['global_window']['start_ns'],original['global_window']['end_ns']
    by_key={};task_details=[[] for _ in targets];all_read_counts={}
    for rank in range(8):
        context=original['per_rank'][str(rank)]
        kernels=[];scanned=0
        with Path(context['kernel_csv']).open(newline='') as stream:
            for line,row in enumerate(csv.DictReader(stream),2):
                scanned+=1
                start=a.us_to_ns(row['Start Time(us)']);end=start+a.us_to_ns(row['Duration(us)'])
                if not any(start<b and end>left for left,b in ranges):continue
                text=(row['Name']+' '+row.get('Type','')).lower()
                kernels.append(dict(name=row['Name'],start_ns=start,end_ns=end,
                    stream=a.optional_int(row['Stream ID']),task=a.optional_int(row['Task ID']),
                    communication=any(token in text for token in a.COMM_TOKENS),csv_line=line,
                    send_anchor=row['Name']==context['compute_stream_derivation']['send_name'],
                    recv_anchor=row['Name']==context['compute_stream_derivation']['recv_name']))
        kernels.sort(key=lambda r:(r['start_ns'],r['end_ns']))
        all_read_counts[rank]=dict(csv_rows_scanned=scanned,kernel_rows_retained_in_two_neighborhoods=len(kernels))
        with a.open_db(context['db']) as db:
            strings=a.read_strings(db)
            dtype_names=dict(db.execute('SELECT id,name FROM ENUM_HCCL_DATA_TYPE'))
            for raw in a.read_comms(db,strings,lo,hi):
                key=a.collective_key(raw)
                by_key.setdefault(key,{})[rank]=dict(data_type=dtype_names.get(raw.get('dataType'),str(raw.get('dataType'))),
                    op_type=a.decode(strings,raw.get('opType')),start_ns=raw['start_ns'],end_ns=raw['end_ns'])
            for index,target in enumerate(targets):
                detail=a.collective_detail(db,strings,target['per_rank'][rank],kernels,
                                          set(context['derived_compute_streams']))
                detail['largest_preceding_host_ranges_by_duration']=sorted(
                    detail['long_host_ranges_before_submit'],key=lambda r:-r['duration_ns'])[:8]
                waits=[r for r in detail['nearby_event_tasks'] if r['task_type']=='EVENT_WAIT'
                       and r['on_derived_compute_stream']]
                detail['compute_stream_event_wait_diagnostic']=dict(count=len(waits),
                    interval_union_ns=a.union_ns((r['start_ns'],r['end_ns']) for r in waits),
                    longest_intervals=sorted(waits,key=lambda r:-(r['end_ns']-r['start_ns']))[:6],
                    not_additional_latency=True,event_identity_unproven=True)
                task_details[index].append(detail)
        print(f'rank {rank}: two AlltoAll contexts complete',flush=True)
    other_groups=Counter();per_chunk=defaultdict(Counter);dtype_inconsistencies=[]
    op_types={};other_ops=[]
    for record in common:
        key=(record['key']['group'],record['key']['operation'],record['key']['count'])
        metadata=by_key.get(key,{})
        a.need(len(metadata)==8,'Common operation missing in current eight databases')
        types={(r['op_type'],r['data_type']) for r in metadata.values()}
        if len(types)!=1:dtype_inconsistencies.append(dict(key=record['key'],per_rank=metadata))
        typ,dtype=next(iter(types)) if len(types)==1 else ('INCONSISTENT','INCONSISTENT')
        op_types[key]=(typ,dtype)
        if 'alltoall' not in record['key']['operation'].lower():
            category=(typ,dtype,record['key']['group'],record['key']['count'])
            other_groups[category]+=1;per_chunk[record['chunk']][category]+=1
            other_ops.append(dict(key=record['key'],chunk=record['chunk'],op_type=typ,data_type=dtype,
                exported_start_spread_ns=record['exported_start_spread_ns'],per_rank=record['per_rank']))
    top=[]
    for target,details in zip(targets,task_details):
        a.need(len(details)==8,'Incomplete selected AlltoAll context')
        entry=dict(target=target,per_rank_dependencies=details)
        for field,api_field in (('host_submit_start','pytorch_api_link'),('cann_launch_start','cann_api_link')):
            unique=[d[api_field]['candidates'][0]['start_ns'] if d[api_field]['status']=='unique' else None for d in details]
            entry[field+'_ns_by_rank']=unique
            entry[field+'_spread_ns']=max(unique)-min(unique) if all(v is not None for v in unique) else None
        top.append(entry)
    summary=[]
    for (typ,dtype,group,count),number in sorted(other_groups.items()):
        summary.append(dict(op_type=typ,data_type=dtype,group=group,count=count,common_calls=number,
            per_chunk={str(chunk):per_chunk[chunk][(typ,dtype,group,count)] for chunk in sorted(per_chunk)}))
    sources=[]
    repository=Path('/mnt/sfs_turbo/lingbot-recovery-zhoutaichang-20261012/worktree')
    for name in ('vllm_omni/diffusion/models/lingbot_world/transformer.py',
                 'vllm_omni/diffusion/distributed/autoencoders/wan_spatial_shard.py',
                 'vllm_omni/diffusion/layers/lingbot_pixel_output.py',
                 'vllm_omni/diffusion/distributed/comm.py'):
        sources.append(a.pin(repository/name))
    output=dict(schema='lingbot_two_alltoall_dependencies_v1',input_report=a.pin(args.input),
        frozen_analyzer=source,followup_source=a.pin(__file__),scope='Two highest exported-start-spread common AlltoAll operations; no full12-target rerun',
        reconstructed_windows=original['windows'],boundary_evidence=original['marker_evidence'],
        alltoall_common_count=len(candidates),other_common_count=len(other_ops),other_communication_types=summary,
        common_metadata_inconsistencies=dtype_inconsistencies,other_common_operations=other_ops,
        selected_alltoalls=top,read_scope=all_read_counts,source_context_pins=sources,
        limitations=original['limitations']+[
            'Source associations based on counts, dtype, topology and code are inferences unless a direct exported callstack links them.',
            'Communication op duration and event-wait interval union are diagnostic and are not a transfer-only or additive savings estimate.',
            'This preserves reconstructed RGB-copy boundaries; explicit stage/probe labels remain absent.'])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as stream:json.dump(output,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(output=str(args.output),alltoalls=len(candidates),other_calls=len(other_ops),metadata_inconsistencies=len(dtype_inconsistencies))))


if __name__=='__main__':main()
