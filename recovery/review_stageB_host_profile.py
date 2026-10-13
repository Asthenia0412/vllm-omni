"""Independent, stdlib-only audit of the completed host diagnostic; no runtime import/device work."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path

ROOT = Path('/mnt/sfs_turbo/lingbot-recovery-zhoutaichang-20261012')
DIRECTORY = ROOT / 'audits/stageB-host-profile-v1'
EXEC = DIRECTORY / 'execution-v1'
NONCE = '590ea9e1e5db7fcbb04a67a8cfeadec4'
PLAN_SHA = 'd1bd75053a361a637177ad31191a65e15d5afc5de0bb6feb76cab12a909296c8'
LENGTHS = [4680, 9360, 14040, 18720, 23400, 28080]
EXPECTED = [(h, p) for h in LENGTHS for p in range(4)]
FORBIDDEN = {'round','rounds','warmups','wall_samples','event_samples','wall_measurements','event_measurements','wall_ns','event_ms','speed_gate'}
INELIGIBLE = ['performance_admitted','model_admitted','speed_gate_eligible','model_timing_eligible']


def need(ok, message):
    if not ok:
        raise ValueError(message)


class Audit:
    def __init__(self):
        self.actual = {}
    def pin(self, path):
        path = Path(path).resolve(strict=True)
        key = str(path)
        if key not in self.actual:
            digest, size = hashlib.sha256(), 0
            with path.open('rb') as stream:
                for raw in iter(lambda: stream.read(1024*1024), b''):
                    digest.update(raw); size += len(raw)
            self.actual[key] = dict(path=key, bytes=size, sha256=digest.hexdigest())
        return self.actual[key]
    def check(self, item):
        need(self.pin(item['path']) == item, 'File pin differs: '+item['path'])
        return item
    def read(self, path):
        self.pin(path)
        return json.loads(Path(path).read_text())
    def pinned(self, item):
        self.check(item)
        return self.read(item['path'])


def reject_formal(value):
    if isinstance(value, dict):
        need(not FORBIDDEN.intersection(value), 'Formal speed fields inside diagnostic')
        for v in value.values(): reject_formal(v)
    elif isinstance(value, list):
        for v in value: reject_formal(v)


def function(row):
    f = row['function']
    need(set(f)=={'filename','line','name'} and type(f['line']) is int and f['line']>=0
         and type(f['filename']) is str and type(f['name']) is str, 'Invalid method identity')
    need(type(row['primitive_calls']) is int and type(row['total_calls']) is int
         and 0<=row['primitive_calls']<=row['total_calls'], 'Invalid counts')
    need(all(type(row[k]) in (int,float) and math.isfinite(row[k]) and row[k]>=0
             for k in ['tottime_s','cumtime_s']), 'Invalid profile times')
    return (f['filename'], f['line'], f['name'])


def main():
    a = Audit(); plan = a.read(EXEC/'plan.json'); w=Path(plan['worktree']); stage=EXEC/'profile1'
    need(a.pin(EXEC/'plan.json')['sha256']==PLAN_SHA, 'Frozen plan SHA changed')
    need(plan['schema']=='lingbot_actual_stageB_host_profile_plan_v1' and plan['diagnostic_only'] is True
         and all(plan[k] is False for k in INELIGIBLE) and plan['final_acceptance'] is False, 'Diagnostic plan scope')
    reject_formal(plan)
    need((plan['ranks'],plan['probes'],plan['actual_layers'],plan['profile_iterations'],plan['histories'])
         ==(8,4,40,20,LENGTHS), 'Matrix protocol changed')
    controller=a.read(stage/'controller-result.json'); launch=a.pinned(controller['launch']); terminal=a.pinned(controller['result'])
    need(controller['completed'] is True and controller['fatal'] is None and controller['remaining_owned_processes']==[]
         and controller['intentional_retirement'] is True, 'Not terminal')
    need(launch['nonce']==NONCE and launch['leader']==dict(pid=2855707,uid=1003,pgid=2855707,start=20903192), 'Wrong launch')
    need(all(launch[k]==terminal[k] for k in ['nonce','leader','command','mode','manifest'])
         and terminal['fatal'] is None and terminal['remaining_owned_processes']==[]
         and terminal['exit_code']==controller['exit_code']==-9 and controller['supervisor_return_code']==247, 'Retirement identity mismatch')
    run=Path(controller['launch']['path']).parent
    need(run==stage/'services'/NONCE and launch['mode']=='profile' and launch['manifest']==plan['manifest']['path'], 'Launch binding mismatch')
    need(launch['command']==['vllm','serve',plan['model'],'--omni','--deploy-config',plan['configs']['B']['path'],
                            '--port',str(plan['port']),'--init-timeout','1800','--stage-init-timeout','1800'], 'Wrong service command')
    intent=a.read(stage/'launch-intent.json'); completed=a.read(stage/'checks-completed.json'); retired=a.read(stage/'retirement-intent.json')
    need(intent['plan']==completed['plan']==a.pin(EXEC/'plan.json') and completed['launch']==retired['launch']==launch
         and retired['request_completed'] is True and intent['begin_boot_ns']<retired['begin_boot_ns']<=controller['end_boot_ns'], 'Ordering differs')
    need(all(controller[k] is False and completed[k] is False for k in INELIGIBLE), 'Admission flags wrong')
    for key in ['parent_plan','parent_baseline_config','env_script','integrity','image','profile_source','screen_source']:
        a.check(plan[key])
    for group in ['harness','benchmark','protected','launch_scripts']:
        for item in plan[group]: a.check(item)
    need(set(plan['configs'])=={'B'}, 'Host service config scope wrong')
    a.check(plan['configs']['B'])
    native=a.pinned(plan['manifest']); cpp=a.pinned(plan['cpp_manifest']); build=a.pinned(plan['cpp_compilation'])
    for key in ['source','library','kernels']: a.check(native[key])
    for group in ['kernel_sources','runtime_sources','abi_headers']:
        for item in native[group]: a.check(item)
    ancestor=a.pinned(native['compilation_ancestor'])
    for key in ['source','library','kernels']:
        need(native[key]['sha256']==ancestor[key]['sha256'] and native[key]['bytes']==ancestor[key]['bytes'], 'Native compilation bytes differ')
    need(native['runtime_binding_only'] is True and native['recompiled'] is False and native['historical_admission_inherited'] is False, 'Native binding claims changed')
    need(cpp['schema']=='lingbot_stageB_guard_runtime_v2' and build['schema']=='lingbot_stageB_guard_cpu_build_v2'
         and cpp['namespace']==build['namespace']=='lingbot_stageB_guard_v2' and cpp['compilation_ancestor']==plan['cpp_compilation'], 'CPP ancestor wrong')
    for field in ['source','library','abi_headers','abi_libraries','cxx11_abi','torch','torch_npu','sdk']:
        need(cpp[field]==build[field], 'CPP runtime vs build differs: '+field)
    for record in [cpp,build]:
        need(all(record[k] is False for k in ['device_admitted','model_admitted','v1_qualification_inherited'])
             and record['guard_timing']=='per_layer_before_consumption', 'CPP scope changed')
    need(cpp['runtime_sources']==native['runtime_sources'] and build['registration'] is True
         and build['npu_initialized'] is False and build['performance_admitted'] is False
         and cpp['historical_admission_inherited'] is False, 'CPP qualification changed')
    for field in ['source','library','builder','provenance','source_contract_tests','compile_result']:
        a.check(build[field])
    for group in ['abi_headers','abi_libraries']:
        need(bool(build[group]), 'Missing ABI pins')
        for item in build[group]: a.check(item)
    compilation=a.pinned(build['compile_result'])
    need(compilation['compilation_succeeded'] is True and compilation['exit_code']==0 and compilation['error'] is None
         and compilation['compiler_command']==build['compiler_command'], 'Compile terminal changed')
    need(build['cpu_rejection_cases']==[{'name':'Frame rejects CPU before NPU StorageImpl access','passed':True},
                                       {'name':'Entry rejects CPU before NPU StorageImpl access','passed':True}], 'CPU refusal evidence incomplete')
    protected={item['path']:item for item in plan['protected']}; parity=a.read(w/'recovery/validation/historical-infrastructure-source-parity-20261013.json')
    need(len(protected)==228 and parity['passed'] is True and len(parity['files'])==228, 'Not228 historical files')
    for item in parity['files']:
        need(protected[str(w/item['relative_path'])]['sha256']==item['historical_sha256'], 'Historical parity differs')
    stage_a=a.pin(w/'vllm_omni/diffusion/layers/lingbot_fixed_timestep_bank_stageA_v1.py')
    need(stage_a['sha256']=='3ea53af8a7f4bdc3e0a78137d235c8019c20f262549f6acc0c951c4c44ad24cc'
         and stage_a in native['runtime_sources'], 'StageA changed')
    observed=a.read(stage/'service-environment.json'); env=observed['environment']
    dynamic={'ROOT_LINGBOT_SP8_FATAL_BASE','ROOT_LINGBOT_SP8_FATAL_DIR','ROOT_LINGBOT_SP8_FATAL_NONCE','ROOT_LINGBOT_SP8_EXPERIMENT_MODE'}
    need(observed['launch']==launch and observed['observed_alive'] is True
         and {k:v for k,v in env.items() if k not in dynamic}==dict(plan['base_environment'],LINGBOT_STAGEB_OPERATOR_SCREEN_DIR=str(stage/'checks'))
         and env['ROOT_LINGBOT_SP8_FATAL_NONCE']==NONCE and env['ROOT_LINGBOT_SP8_FATAL_DIR']==str(run)
         and env['ROOT_LINGBOT_SP8_EXPERIMENT_MODE']=='profile' and env['LINGBOT_STAGEB_HOST_PROFILE']=='1'
         and env['VLLM_LINGBOT_NPU_ASYNC_METADATA']=='0', 'Environment differs')
    need(a.read(run/'smoke-exit.json')['exit_code']==0, 'Request failed')
    rollout=a.read(run/'rollout8.json'); command=a.read(run/'smoke-command.json')
    need(rollout['model']==plan['model'] and rollout['workload']['num_chunks']==8 and rollout['workload']['num_frames']==93
         and rollout['workload']['fps']==12 and rollout['config']['mux_fps']==rollout['config']['target_fps']==12
         and len(rollout['sessions'])==1, 'Request frame count/FPS wrong')
    chunks=rollout['sessions'][0]['chunks']
    need([c['index'] for c in chunks]==list(range(8)) and [c['num_frames'] for c in chunks]==[9]+[12]*7, 'Frame continuity failed')
    cpu=a.read(DIRECTORY/'cpu-test-results-v2.json')
    need(cpu['interpreter']=='/home/zhoutaichang/venvs/vllm-omni/bin/python' and cpu['python'].startswith('3.12.14 '), 'Wrong CPU interpreter')
    for item,(file,count,optimized) in zip(cpu['runs'],[(name,count,opt) for opt in [False,True]
            for name,count in [('test_stageB_host_profile_cpu.py',16),('test_stageB_host_profile_controller_cpu.py',22),('test_stageB_screen_cpu.py',6)]]):
        expected=[cpu['interpreter']]+(['-O'] if optimized else [])+['recovery/'+file]
        need(item['command']==expected and item['exit']==0 and f'Ran {count} tests' in item['stderr']
             and item['stderr'].endswith('OK\n') and item['source_sha256']==a.pin(w/'recovery'/file)['sha256'], 'CPU receipt/source differs')
    need(len(cpu['runs'])==6, 'CPU receipt incomplete')
    core=str(w/'vllm_omni/diffusion/layers/lingbot_fixed_modulation_stageB_cpp.py')
    stageapath=stage_a['path']; summaries=[]; selected=None
    need(len(completed['checks'])==8, 'Eight reports required')
    for rank,pin in enumerate(completed['checks']):
        need(pin['path']==str(stage/f'checks/rank{rank}.json'), 'Rank path repeated/reordered')
        report=a.pinned(pin); reject_formal(report)
        need(report['schema']=='lingbot_actual_stageB_host_profile_v1' and report['passed'] is True
             and report['rank']==rank and report['nonce']==NONCE and report['operator_only'] is True
             and report['diagnostic_only'] is True and all(report[k] is False for k in INELIGIBLE)
             and (report['actual_layers'],report['actual_probes'],report['profile_iterations'],report['actual_histories'])==(40,4,20,LENGTHS), 'Rank scope wrong')
        need(report['source']==plan['profile_source']['path'] and report['source_sha256']==plan['profile_source']['sha256']
             and report['screen_source']==plan['screen_source'], 'Helper source binding differs')
        need([(c['history'],c['probe']) for c in report['cases']]==EXPECTED, 'Actual context matrix wrong')
        for case in report['cases']:
            need(case['correctness'] is True and set(case['arms'])=={'native','stageB'}, 'Logical correctness failed')
            row=dict(rank=rank,history=case['history'],probe=case['probe'],arms={})
            for arm,stats in case['arms'].items():
                need(stats['enqueue_calls']==20 and stats['actual_layers_per_call']==40
                     and math.isfinite(stats['cprofile_total_tottime_s']) and stats['cprofile_total_tottime_s']>=0, 'Profile stats invalid')
                seen=set(); methods=stats['methods']
                for method in methods:
                    key=function(method);need(key not in seen,'Duplicate method');seen.add(key)
                    callers=[function(c) for c in method['callers']];need(len(callers)==len(set(callers)), 'Duplicate caller')
                need(math.isclose(sum(m['tottime_s'] for m in methods),stats['cprofile_total_tottime_s'],rel_tol=1e-10,abs_tol=1e-12), 'Total self time inconsistent')
                def find(path,name,count):
                    rows=[m for m in methods if m['function']['filename']==path and m['function']['name']==name]
                    need(len(rows)==(1 if count else 0), 'Wrong method count presence: '+name)
                    if count:
                        need(rows[0]['primitive_calls']==rows[0]['total_calls']==count, 'Wrong actual scope calls:'+name)
                        return {k:rows[0][k] for k in ['primitive_calls','total_calls','tottime_s','cumtime_s']}
                    return None
                invoke=find(plan['screen_source']['path'],'invoke',20)
                prep=find(core,'prepare_forward',20 if arm=='stageB' else 0)
                get=find(core,'get',800 if arm=='stageB' else 0)
                need(not any(m['function']['name'] in ('synchronize','write_report','file_pin') for m in methods), 'Sync/file work profiled')
                row['arms'][arm]=dict(invoke=invoke,prepare_forward=prep,get=get,
                    total_profile_tottime_s=stats['cprofile_total_tottime_s'],scriptmethod_named_rows=sum('ScriptMethod' in str(m['function']) for m in methods))
                if arm=='native':
                    row['arms'][arm]['native_tensor_methods']={name:find('~', "<method '"+name+"' of 'torch._C.TensorBase' objects>",800) for name in ['unsqueeze','float','chunk']}
                if arm=='stageB':
                    row['arms'][arm]['stageA_light']=find(stageapath,'_light',80)
                    row['arms'][arm]['stageA_version']=find(stageapath,'_version',200)
            summaries.append(row)
            if rank==0 and case['history']==28080 and case['probe']==3: selected=row
    installed=Path('/home/zhoutaichang/venvs/vllm-omni/lib/python3.12/site-packages/torch')
    torch_sources=[a.pin(installed/x) for x in ['_classes.py','include/torch/csrc/jit/python/pybind_utils.h','include/torch/custom_class_detail.h']]
    result=dict(schema='lingbot_stageB_host_profile_independent_review_v1',review_passed=True,diagnostic_only=True,
       performance_admitted=False,model_admitted=False,speed_gate_eligible=False,model_timing_eligible=False,
       plan=a.pin(EXEC/'plan.json'),nonce=NONCE,controller=a.pin(stage/'controller-result.json'),cpu_tests=a.pin(DIRECTORY/'cpu-test-results-v2.json'),
       verification=dict(current_files_sha_verified=len(a.actual),protected228_historical=True,stageA=stage_a,cpp_compilation=plan['cpp_compilation'],
                         cpp_manifest=plan['cpp_manifest'],cpp_header_count=len(build['abi_headers']),cpp_library_count=len(build['abi_libraries']),
                         profile_source=plan['profile_source'],screen_source=plan['screen_source'],target_cpu_normal=[16,22,6],target_cpu_optimized=[16,22,6]),
       termination=dict(intentional=True,service_exit=-9,supervisor_exit=247,fatal=None,remaining_owned_processes=[],normal_shutdown_proven=False),
       scope=dict(actual_contexts=192,rank_count=8,histories=LENGTHS,probes=4,layers=40,calls_per_case_arm=20,
                  actual_candidate_prepare_calls=3840,actual_candidate_get_calls=153600,native_get_calls=0,
                  replay='actual retained active StageA capabilities, original source owners, post-SP snapshots; histories are context labels, not modulation input'),
       request=dict(chunks=8,frames=93,fps=12,performance_metrics_ignored=True),selected_rank0_H6P3=selected,
       cprofile_boundary=dict(scriptmethod_rows_across_matrix=sum(r['arms']['stageB']['scriptmethod_named_rows'] for r in summaries),
           nonadditive='prepare contains context/light/version; get contains _check and callable check. Do not sum nested cumtimes.',
           opaque_native='No ScriptMethod row. get tottime includes unobserved custom-class bridge/guard work and possible blocking; cannot call it pure Python or measure bridge-only overhead.',
           native='Native tensor + is also opaque inside genexpr; self-time is not Python-only.',torch_source_pins=torch_sources),
       cases=summaries,
       limitations=['cProfile perturbs dispatch and current-thread wall time; neither clean timing nor device phase attribution',
                    '20 outputs retained until unprofiled sync; differs from formal timing owner lifetime',
                    'logical output/source and shape/stride/offset checks do not prove full arena/canary or lifecycle acceptance',
                    'rollout simulated playback and diagnostic inter-arrival metrics are not playback/end-to-end qualification',
                    'same-process active capability replay is not standalone payload fixture export or arbitrary input mutation proof',
                    'StageB v2 formal round2 had failed10percent gate; this diagnostic does not change that decision'])
    result['evidence_pins']=list(a.actual.values())
    output=DIRECTORY/'INDEPENDENT_HOST_PROFILE_REVIEW.json'
    need(not output.exists(),'Refuse overwrite')
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(path=str(output),sha256=hashlib.sha256(output.read_bytes()).hexdigest(),passed=True,files=len(a.actual),cases=len(summaries),scriptmethod_rows=result['cprofile_boundary']['scriptmethod_rows_across_matrix']),indent=2))


if __name__=='__main__': main()
