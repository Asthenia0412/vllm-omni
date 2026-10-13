"""Attribute complete rank0 chunks between blocking RGB-to-host completions.

Requires a single 16-chunk request, four denoise probes per chunk, and all
40 layers represented by send/recv. Report kernel sums separately from union
time; these are profiled durations, never clean performance acceptance.
"""
import argparse
import collections
import csv
import json
from pathlib import Path


def union(intervals):
    total = 0.0
    end = float('-inf')
    for a,b in sorted(intervals):
        total += max(0.0, b-max(a,end))
        end = max(end,b)
    return total


def main():
    p=argparse.ArgumentParser()
    p.add_argument('root',type=Path)
    p.add_argument('--rank',type=int,choices=range(8),default=0)
    args=p.parse_args()
    folder=args.root/'diffusion_rank0'
    trace,=folder.glob('*/ASCEND_PROFILER_OUTPUT/trace_view.json')
    kernel_folder=args.root/f'diffusion_rank{args.rank}'
    kernel,=kernel_folder.glob('*/ASCEND_PROFILER_OUTPUT/kernel_details.csv')
    cache=args.root/'rank0-stage-markers.json'
    if cache.exists():
        saved=json.loads(cache.read_text())
        assert saved['trace_size']==trace.stat().st_size
        markers=saved['markers']
    else:
        print('Reading trace',flush=True)
        data=json.loads(trace.read_text())
        events=data['traceEvents'] if isinstance(data,dict) else data
        markers=[e for e in events if e.get('ph')=='X' and e.get('name','').startswith('lingbot.')]
        cache.write_text(json.dumps({'trace_size':trace.stat().st_size,'markers':markers},indent=2)+'\n')
        del data, events
    groups=collections.defaultdict(list)
    for e in markers:
        groups[e['name']].append((float(e['ts']),float(e['ts'])+float(e['dur'])))
    for events in groups.values():events.sort()
    assert len(groups['lingbot.rgb_copy'])==16
    assert len(groups['lingbot.denoise_step'])==64
    assert len(groups['lingbot.streaming_vae_decode'])==16
    kernels=[]
    with kernel.open() as f:
        for row in csv.DictReader(f):
            start=float(row['Start Time(us)']);dur=float(row['Duration(us)'])
            kernels.append((start,start+dur,row['Name'],row['Input Shapes'],dur))
    records=[]
    for chunk in range(11,15):
        lo=groups['lingbot.rgb_copy'][chunk-1][1]
        hi=groups['lingbot.rgb_copy'][chunk][1]
        selected=[x for x in kernels if x[0]>=lo and x[1]<=hi]
        crossing=[x for x in kernels if (x[0]<lo<x[1]) or (x[0]<hi<x[1])]
        assert not crossing, f'device work crosses chunk {chunk} boundary'
        probes=[x for x in groups['lingbot.denoise_step'] if lo<=x[0] and x[1]<=hi]
        assert len(probes)==4
        counts=collections.Counter(x[2] for x in selected)
        assert counts['lingbot_recovery_send']==160 and counts['lingbot_recovery_recv']==160
        totals=collections.defaultdict(float)
        compute=[];comm=[]
        for a,b,name,shape,duration in selected:
            totals[name]+=duration/1000
            (comm if any(s in name.lower() for s in ('hccl','alltoall','allgather','batch_send_recv','allreduce')) else compute).append((a,b))
        cu=union(compute);co=union(comm);both=union(compute+comm)
        records.append({'chunk':chunk,'window_ms':(hi-lo)/1000,'probes':len(probes),
          'send_calls':counts['lingbot_recovery_send'],'recv_calls':counts['lingbot_recovery_recv'],
          'copy_calls':counts['lingbot_recovery_copy_pair'],
          'compute_union_ms':cu/1000,'communication_union_ms':co/1000,
          'overlap_ms':(cu+co-both)/1000,'neither_ms':(hi-lo-both)/1000,
          'kernel_workload_ms':dict(sorted(totals.items(),key=lambda x:-x[1]))})
    receipt={'scope':f'rank{args.rank} chunks11–14 inside rank0 blocking RGB boundaries; profiled, not acceptance',
       'boundary_validation':'no device kernel crosses either edge; exactly160 send and160 recv per chunk',
       'communication_classification':'name heuristic; neither is not necessarily CPU idle',
       'kernel_details':str(kernel),'trace':str(trace),'chunks':records}
    (args.root/f'rank{args.rank}-steady-attribution.json').write_text(json.dumps(receipt,indent=2)+'\n')
    for r in records:
        print(r['chunk'],r['window_ms'],list(r['kernel_workload_ms'].items())[:5],flush=True)


if __name__=='__main__':main()
