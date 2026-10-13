"""CPU-only preparation: emit separate smoke2+2 and full warm40+16 launches.

Never starts a service or sends requests. Root controls device availability and
executes generated commands serially. Runtime sources are read/pinned, not edited.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex


def pin(path):
    path=Path(path).resolve(strict=True)
    return dict(path=str(path),bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def prepare(root,output,manifest,base_config,devices,image,port=7816):
    import yaml
    root=Path(root).resolve(strict=True);worktree=root/'worktree'
    output=Path(output).resolve();manifest=Path(manifest).resolve(strict=True)
    if not output.is_relative_to('/mnt/sfs_turbo'):
        raise ValueError('Persistent prepared launch root required')
    ids=devices.split(',')
    if len(ids)!=8 or len(set(ids))!=8 or any(not item.isdecimal() for item in ids):
        raise ValueError('Root must explicitly select eight distinct visible dies')
    build=json.loads(manifest.read_text())
    for item in (build['source'],build['library'],build['kernels'],*build['kernel_sources'],
                 *build['runtime_sources'],*build['abi_headers']):
        if pin(item['path']) != item:
            raise ValueError('Source/library/header pin differs: '+item['path'])
    required=[worktree/'vllm_omni/diffusion/layers'/name for name in
              ('lingbot_real_fixture_capture.py','lingbot_output_capture.py')]
    if not all(pin(path) in build['runtime_sources'] for path in required):
        raise ValueError('Build must pin real capture helper/install hook')
    image=Path(image).resolve(strict=True)
    value=yaml.safe_load(Path(base_config).read_text());stage=value['stages'][0]
    if len(value['stages'])!=1 or stage['parallel_config']!={
            'ring_degree':1,'sequence_parallel_size':8,'tensor_parallel_size':1,'ulysses_degree':8}:
        raise ValueError('Exactly the original single-stage SP8/TP1/ring1 configuration required')
    config=stage['model_config']
    config.update(lingbot_npu_recovery_qk=True,lingbot_npu_recovery_qk_combined=False,
                  lingbot_npu_recovery_cross=True,lingbot_npu_recovery_cross_combined=False,
                  lingbot_npu_qk_single_host_admission_v1=False,lingbot_npu_cross_combined_guard_cpp_v1=False)
    if (config['ar_diffusion_width'],config['ar_diffusion_height'],
        stage['default_sampling_params']['num_inference_steps'])!=(832,480,4):
        raise ValueError('Original resolution/four-probe model required')
    if not all(config.get(key) is True for key in ('lingbot_npu_recovery_vae_pack',
            'lingbot_npu_fixed_timestep_bank','lingbot_npu_pixel_device_interleave')):
        raise ValueError('Recovered VAE/StageA/device-interleave baseline required')
    output.mkdir(parents=True,exist_ok=False)
    harness=[pin(worktree/'recovery'/name) for name in
             ('env.sh','supervise.py','run_smoke.py','run_real_fixture_capture.py',
              'load_real_fixtures.py','check_real_fixtures.py')]
    paths=[]
    for phase,warm,chunks in (('smoke',2,2),('full',40,16)):
        directory=output/phase;directory.mkdir()
        config_path=directory/'config.yaml';config_path.write_text(yaml.safe_dump(value,sort_keys=False))
        env=dict(ASCEND_RT_VISIBLE_DEVICES=devices,VLLM_LINGBOT_NPU_ASYNC_METADATA='0',
                 LINGBOT_REAL_FIXTURE_CAPTURE='1',LINGBOT_OUTPUT_CAPTURE_DIR=str(directory/'fixtures'),
                 LINGBOT_REAL_FIXTURE_DIR=str(directory/'fixtures'))
        command=['python',str(worktree/'recovery/supervise.py'),'--runs',str(directory/'services'),
                 '--mode','correctness','--recv','--manifest',str(manifest),'--','vllm','serve',
                 str(root/'models/lingbot-world'),'--omni','--deploy-config',str(config_path),
                 '--port',str(port),'--init-timeout','1800','--stage-init-timeout','1800']
        shell='#!/usr/bin/env bash\nset -eo pipefail\ncd '+shlex.quote(str(worktree))+'\nsource recovery/env.sh\n'
        shell+='while IFS= read -r name; do\n  case "$name" in\n    *LINGBOT*CAPTURE*|*LINGBOT*PROFILE*|*LINGBOT*DIAGNOSTIC*) unset "$name" ;;\n  esac\ndone < <(compgen -e)\n'
        shell+='unset LINGBOT_SP8_OBSERVE_MODE LINGBOT_QK_COMBINED_MANIFEST LINGBOT_QK_COMBINED_MANIFEST_SHA256\n'
        for key,val in env.items():shell+='export '+key+'='+shlex.quote(val)+'\n'
        shell+='exec '+shlex.join(command)+'\n';(directory/'launch.sh').write_text(shell)
        plan=dict(schema='lingbot_real_fixture_launch_plan_v1',phase=phase,config=str(config_path),
                  config_sha256=pin(config_path)['sha256'],manifest=str(manifest),
                  manifest_sha256=pin(manifest)['sha256'],fixture_root=str(directory/'fixtures'),
                  environment=env,warmup_chunks=warm,capture_chunks=chunks,port=port,
                  model=str(root/'models/lingbot-world'),integrity=str(root/'models/lingbot-world-integrity.json'),
                  image=str(image),harness_pins=harness,image_pin=pin(image),
                  timing_eligible=False,complete_input_matrix=False,prepared_only=True,source_pinned=True)
        plan_path=directory/'plan.json';plan_path.write_text(json.dumps(plan,indent=2)+'\n')
        commands='bash '+shlex.quote(str(directory/'launch.sh'))+'\n'
        commands+='python recovery/run_real_fixture_capture.py --plan '+shlex.quote(str(plan_path))+' --run <owned-supervisor-directory>\n'
        (directory/'commands.txt').write_text(commands);paths.append(str(plan_path))
    return paths


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('root','output','manifest','base-config','image'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--devices',required=True)
    parser.add_argument('--port',type=int,default=7816)
    args=parser.parse_args()
    print(json.dumps(prepare(args.root,args.output,args.manifest,args.base_config,args.devices,args.image,args.port),indent=2))
