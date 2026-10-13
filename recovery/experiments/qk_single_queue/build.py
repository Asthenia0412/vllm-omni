from pathlib import Path
import json,hashlib,os,subprocess
import torch,torch_npu
from torch.utils.cpp_extension import include_paths,library_paths
import argparse
parser=argparse.ArgumentParser(description='Rejected diagnostic only; never a model provider')
parser.add_argument('--provider-manifest',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
source=Path(__file__).resolve().with_name('host.cpp')
root=args.output.resolve();root.mkdir(parents=True,exist_ok=False)
work=Path(__file__).resolve().parents[3]
sdk=Path(os.environ['ASCEND_HOME_PATH']).resolve()
npu=Path(torch_npu.__file__).resolve().parent
manifest_path=args.provider_manifest.resolve()
base=json.loads(manifest_path.read_text())
kernel=Path(base['kernels']['path'])
lib=root/'liblingbot_qk_single_queue_guard_v2.so'
includes=include_paths()+[str(npu/'include'),str(sdk/'include'),str(work/'csrc/lingbot_recovery')]
libraries=library_paths()+[str(npu/'lib'),str(sdk/'lib64'),str(kernel.parent)]
cmd=['g++','-std=c++17','-shared','-fPIC','-O2',f'-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}',str(source),'-o',str(lib)]
cmd += ['-I'+s for s in includes]+['-L'+s for s in libraries]+['-Wl,-rpath,'+s for s in libraries]
cmd += [str(kernel),'-ltorch_npu','-ltorch_cpu','-ltorch','-lc10','-lopapi','-lnnopbase','-lascendcl','-ldl']
assert not torch_npu.npu.is_initialized()
subprocess.run(cmd,check=True)
torch.ops.load_library(str(lib))
paths=torch.ops.lingbot_qk_single_queue_guard_v2.dependencies()
assert not torch_npu.npu.is_initialized()
def pin(p):
 p=Path(p).resolve();b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
receipt=dict(schema='lingbot_qk_single_queue_cpu_build_guard_v2',operator_namespace='lingbot_qk_single_queue_guard_v2',provider_manifest=pin(manifest_path),source=pin(source),library=pin(lib),kernel=pin(kernel),dependencies=[pin(p) for p in paths],command=cmd,torch=torch.__version__,torch_npu=torch_npu.__version__,torch_npu_git=torch_npu.version.git_version,registration=True,npu_initialized=False,device_admitted=False,model_admitted=False)
(root/'build-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
