"""Root-invoked correctness controller for an already-owned eight-rank service.

Does not launch a service. Reuses run_smoke requests and supervise identity-based
retirement; all timing/model admission flags remain false.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from supervise import identity, retire
from load_real_fixtures import case_paths, load_case, contract
from check_real_fixtures import check as check_full


def need(value, reason):
    if not value:
        raise ValueError(reason)


def check_partial(root):
    """Validate every actually observed smoke state without calling it full192."""
    root=Path(root); all_locations=[]; totals={'qk':0,'cross':0}; manifests=set(); descriptors=0
    source_pins=set()
    for rank in range(8):
        directory=root/f'rank{rank}';coverage=json.loads((directory/'coverage.json').read_text())
        need(coverage['schema']=='lingbot_real_qk_cross_capture_v1' and coverage['rank']==rank
             and not coverage['failed'] and not coverage['complete'] and not coverage['timing_eligible'],
             'Smoke must be successful but partial coverage')
        source_pins.add(tuple((p['path'],p['sha256']) for p in coverage['source_pins']))
        locations={(v['kv_len'],v['probe']):(v['request_id'],v['chunk']) for v in coverage['locations']}
        lengths={key[0] for key in locations}
        need(lengths and lengths.issubset(contract.LENGTHS)
             and len(lengths)<len(contract.LENGTHS)
             and set(locations)=={(length,probe) for length in lengths for probe in range(4)},
             'Smoke lacks complete four-probe coverage for each observed history')
        all_locations.append(locations);manifests.add(coverage['build_manifest']['sha256'])
        expected={(kind,length,probe,layer) for kind in ('qk','cross')
                  for length,probe in locations for layer in range(40)}
        actual=set()
        for name in coverage['cases']:
            path=(directory/name).resolve(strict=True)
            need(path.is_relative_to(directory.resolve()),'Case escaped rank directory')
            case=json.loads(path.read_text());key=(case['kind'],case['kv_len'],case['probe'],case['layer'])
            need(key in expected and key not in actual and case['rank']==rank,'Missing/duplicate case identity')
            need((case['request_id'],case['chunk'])==locations[key[1:3]],'Case selected context differs')
            receipt=case['dispatch_receipt'];operation='qk_send' if case['kind']=='qk' else 'cross'
            need(receipt['provider_operation']==operation and receipt['dispatch_count']==1
                 and receipt['operands_identical'] and receipt['output_identical']
                 and receipt['build_manifest_sha256']==coverage['build_manifest']['sha256'],
                 'Actual backend receipt mismatch')
            need(case['source_descriptors_unchanged'] and case['payload_present']==(case['layer']==0)
                 and case['full_sources_unchanged'] is (True if case['layer']==0 else None),
                 'Source/payload coverage mislabeled')
            for section in ('inputs','outputs'):
                snap=case[section]
                for tensor,desc in snap['descriptors'].items():
                    contract.view_extent(desc)
                    owner=snap['owners'][snap['view_owner_ids'][tensor]]
                    need(all(owner[k]==desc[k] for k in ('storage_bytes','base_sizes','base_strides','npu_format')),
                         'Owner descriptor mismatch')
            actual.add(key)
        need(actual==expected,'Missing all40 descriptor coverage for smoke states')
        descriptors+=len(actual)
        for kind in ('qk','cross'):
            paths=case_paths(root,kind,rank,require_complete=False)
            need(len(paths)==len(locations),'Smoke lacks actual layer0 payloads for observed states')
            for path in paths:
                fixture=load_case(path)
                totals[kind]+=1
                del fixture
    need(len(manifests)==1 and len(source_pins)==1 and all(loc==all_locations[0] for loc in all_locations),
         'Ranks disagree on actual history/probe context or build')
    return dict(schema='lingbot_real_fixture_partial_audit_v1',partial_smoke_passed=True,
                complete_input_matrix=False,actual192_global_proven=False,
                qk_real_layer0_payload_cases=totals['qk'],cross_real_layer0_payload_cases=totals['cross'],
                all40_descriptor_cases=descriptors,all40_payloads=False,payload_sha256_verified=True,
                actual_kv_lengths=sorted({key[0] for key in all_locations[0]}),
                destination_canary_verified=False,kernel_correctness_passed=False,
                model_admitted=False,performance_validated=False,build_manifest_sha256=next(iter(manifests)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args();plan=json.loads(args.plan.read_text())
    launch=json.loads((args.run/'launch.json').read_text());leader=launch['leader']
    need(launch['mode']=='correctness' and identity(leader['pid'])==leader
         and leader['uid']==os.getuid(),'Fresh owned correctness service required')
    need(not list(args.run.glob('rollout*.json')) and not (args.run/'fixture-protocol.json').exists(),
         'Capture controller requires fresh service/request order')
    need(str(Path(launch['manifest']).resolve())==str(Path(plan['manifest']).resolve()),'Wrong base manifest')
    need(hashlib.sha256(Path(plan['manifest']).read_bytes()).hexdigest()==plan['manifest_sha256'],
         'Prepared source manifest changed')
    command=launch['command'];config=Path(command[command.index('--deploy-config')+1])
    need(int(command[command.index('--port')+1])==plan['port'],'Owned service port differs from plan')
    need(config.resolve()==Path(plan['config']).resolve(),'Wrong capture config')
    need('config_sha256' in plan and 'harness_pins' in plan,'Pinned config/harness launch plan required')
    need(hashlib.sha256(config.read_bytes()).hexdigest()==plan['config_sha256'],'Prepared config changed')
    wanted_harness={str(Path(__file__).resolve().with_name(name)) for name in
                    ('run_real_fixture_capture.py','check_real_fixtures.py','load_real_fixtures.py',
                     'run_smoke.py','supervise.py','env.sh')}
    need({item['path'] for item in plan['harness_pins']}==wanted_harness,'Complete capture harness pins required')
    for item in plan['harness_pins']:
        path=Path(item['path'])
        need(path.stat().st_size==item['bytes']
             and hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256'], 'Prepared capture harness changed')
    import yaml
    values=yaml.safe_load(config.read_text())['stages'][0]['model_config']
    expected=dict(lingbot_npu_recovery_qk=True,lingbot_npu_recovery_qk_combined=False,
                  lingbot_npu_recovery_cross=True,lingbot_npu_recovery_cross_combined=False,
                  lingbot_npu_qk_single_host_admission_v1=False,lingbot_npu_cross_combined_guard_cpp_v1=False)
    need(all(values.get(k) is v for k,v in expected.items()),'Capture baseline/combined flags differ')
    raw=(Path('/proc')/str(leader['pid'])/'environ').read_bytes()
    env=dict(x.decode().split('=',1) for x in raw.split(b'\0') if b'=' in x)
    need(all(env.get(k)==v for k,v in plan['environment'].items()),'Owned service capture environment differs')
    need(env.get('LINGBOT_REAL_FIXTURE_CAPTURE')=='1'
         and env.get('LINGBOT_REAL_FIXTURE_DIR')==plan['fixture_root']
         and env.get('LINGBOT_OUTPUT_CAPTURE_DIR')==plan['fixture_root'], 'Owned fixture root differs from plan')
    need(env.get('LINGBOT_SP8_OBSERVE_MODE','')=='','Old SP8 observer must be disabled')
    need(env.get('LINGBOT_RECOVERY_RECV')=='1' and env.get('VLLM_LINGBOT_NPU_HISTORY_STAGING')=='1',
         'Recovered recv/history staging missing')
    receipt=dict(schema='lingbot_real_fixture_controller_v1',plan=str(args.plan.resolve()),launch=launch,
                 plan_sha256=hashlib.sha256(args.plan.read_bytes()).hexdigest(),
                 config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
                 environment=plan['environment'],completed=False,timing_eligible=False,
                 model_admitted=False,performance_validated=False)
    output=args.run/'fixture-protocol.json';output.write_text(json.dumps(receipt,indent=2)+'\n')
    base=[sys.executable,'recovery/run_smoke.py','--run',str(args.run),'--model',plan['model'],
          '--integrity',plan['integrity'],'--image',plan['image'],'--port',str(plan['port'])]
    try:
        for count,label in ((plan['warmup_chunks'],'request1'),(plan['capture_chunks'],'request2')):
            need(identity(leader['pid'])==leader,'Owned service exited; no replacement will be used')
            subprocess.run(base+['--chunks',str(count)],check=True)
            for name in ('smoke.log','smoke-command.json','smoke-exit.json',f'rollout{count}.json'):
                shutil.copy2(args.run/name,args.run/f'{label}-{name}')
        result=(check_partial(plan['fixture_root']) if plan['phase']=='smoke'
                else check_full(plan['fixture_root'],hash_payloads=True))
        need(result['build_manifest_sha256']==plan['manifest_sha256'],'Fixture audit belongs to a different build')
        (args.run/'fixture-matrix-audit.json').write_text(json.dumps(result,indent=2)+'\n')
        receipt['completed']=True;receipt['matrix_audit']=result
        print(json.dumps(result,indent=2),flush=True)
    except BaseException as error:
        receipt['error']=repr(error)
        raise
    finally:
        if identity(leader['pid'])==leader:
            retire(leader)
        # Supervisor owns final process reclamation/result; do not signal by name.
        for _ in range(40):
            if (args.run/'result.json').exists():break
            time.sleep(.5)
        receipt['supervisor_result_present']=(args.run/'result.json').exists()
        output.write_text(json.dumps(receipt,indent=2)+'\n')


if __name__=='__main__':main()
