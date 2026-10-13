"""One owned timing launch, warm40 then161/600; never a multi-run admission."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from supervise import identity
from summarize_timing import summarize


def main():
    p=argparse.ArgumentParser()
    for name in ('run','model','integrity','image'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--chunks',type=int,choices=(161,600),default=161)
    p.add_argument('--port',type=int,default=7816)
    args=p.parse_args()
    launch=json.loads((args.run/'launch.json').read_text())
    assert launch['mode']=='timing', 'Independent timing service required'
    assert not (args.run/'timing-protocol.json').exists(), 'Fresh timing controller required'
    assert not list(args.run.glob('rollout*.json')), 'Fresh service must not have previous requests'
    leader=launch['leader']
    assert identity(leader['pid'])==leader
    raw=(Path('/proc')/str(leader['pid'])/'environ').read_bytes()
    env=dict(x.decode().split('=',1) for x in raw.split(b'\0') if b'=' in x)
    forbidden={k:v for k,v in env.items() if k.startswith('LINGBOT_') and
               ('CAPTURE' in k or 'PROFILE' in k) and v not in ('','0')}
    assert not forbidden, f'Capture/profiler environment active: {list(forbidden)}'
    config=Path(launch['command'][launch['command'].index('--deploy-config')+1])
    import yaml
    def contains_profiler(value):
        if isinstance(value,dict):
            return any((k=='profiler_config' and bool(v)) or contains_profiler(v) for k,v in value.items())
        return isinstance(value,list) and any(contains_profiler(v) for v in value)
    assert not contains_profiler(yaml.safe_load(config.read_text())), 'Profiler configured'
    def check_exporters():
        for proc in Path('/proc').iterdir():
            if not proc.name.isdigit():continue
            try:
                if proc.stat().st_uid!=os.getuid():continue
                argv=(proc/'cmdline').read_bytes().split(b'\0')
                if any(Path(x.decode()).name in ('export_profile.py','msprof.py') for x in argv[:3] if x):
                    raise RuntimeError(f'Offline exporter still active: {proc.name}')
            except (FileNotFoundError,ProcessLookupError,PermissionError):pass
    check_exporters()
    manifest=Path(launch['manifest'])
    receipt={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
             'worktree_status':subprocess.check_output(['git','status','--porcelain'],text=True),
             'manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),
             'warmup_chunks':40,'measurement_chunks':args.chunks,'launch':launch,
             'config_sha256':hashlib.sha256(config.read_bytes()).hexdigest(),
             'environment':{k:v for k,v in env.items() if k.startswith(('LINGBOT_','VLLM_LINGBOT_','ROOT_LINGBOT_'))
                            or k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','ASCEND_RT_VISIBLE_DEVICES')},
             'completed':False,'final_acceptance':False}
    output=args.run/'timing-protocol.json'
    output.write_text(json.dumps(receipt,indent=2)+'\n')
    base=[sys.executable,'recovery/run_smoke.py','--run',str(args.run),'--model',str(args.model),
          '--integrity',str(args.integrity),'--image',str(args.image),'--port',str(args.port)]
    for count,label in ((40,'warm40'),(args.chunks,f'measure{args.chunks}')):
        check_exporters()
        assert identity(leader['pid'])==leader
        subprocess.run(base+['--chunks',str(count)],check=True)
        for name in ('smoke.log','smoke-command.json','smoke-exit.json'):
            shutil.copy2(args.run/name,args.run/f'{label}-{name}')
    check_exporters()
    result=args.run/f'rollout{args.chunks}.json'
    summary=summarize(json.loads(result.read_text()),args.chunks)
    summary['input_sha256']=hashlib.sha256(result.read_bytes()).hexdigest()
    (args.run/'main-window.json').write_text(json.dumps(summary,indent=2)+'\n')
    receipt['completed']=True
    output.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
