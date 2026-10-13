# SPDX-License-Identifier: Apache-2.0
"""Explicit host-only build/registration; no device construction or operation."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess

SCHEMA = 'lingbot_stageB_guard_cpu_build_v2'
NAMESPACE = 'lingbot_stageB_guard_v2'


def need(ok, message):
    if not ok:
        raise RuntimeError(message)


def pin(path):
    path = Path(path).resolve(strict=True)
    payload = path.read_bytes()
    return dict(path=str(path), bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())


def durable_json(path, value):
    """Publish complete receipts only after file data and directory are durable."""
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def mapped_abi_libraries(roots):
    """Actual mapped framework/SDK DSOs after registration, with resolved paths."""
    paths = set()
    for line in Path('/proc/self/maps').read_text().splitlines():
        pieces = line.split(maxsplit=5)
        if len(pieces) != 6 or not pieces[5].startswith('/'):
            continue
        path = Path(pieces[5])
        if '.so' not in path.name or not path.is_file():
            continue
        resolved = path.resolve()
        if any(resolved.is_relative_to(root) for root in roots):
            paths.add(resolved)
    return [pin(path) for path in sorted(paths)]


def build(output):
    source = Path(__file__).resolve().with_name('host.cpp')
    provenance = source.with_name('PROVENANCE.txt')
    test = source.parents[2] / 'test_stageB_cpp_guard_source_cpu.py'
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    before = dict(source=pin(source), builder=pin(__file__), provenance=pin(provenance),
                  source_contract_tests=pin(test))
    durable_json(output / 'build-intent.json', dict(schema=SCHEMA, namespace=NAMESPACE, **before,
                 phase='before_framework_import', npu_work_authorized=False))

    import torch
    import torch_npu
    from torch.utils.cpp_extension import include_paths, library_paths
    need((torch.__version__, torch_npu.__version__) == ('2.10.0+cpu', '2.10.0.post4'),
         'Exact framework ABI required; no implicit rebuild for another version')
    need(torch_npu.version.git_version == '5dd8ef3f9b375b5ae4a83538d5785754148c3302',
         'Unexpected torch-npu source version')
    need(not torch_npu.npu.is_initialized(), 'Build process already initialized NPU')
    torch_root = Path(torch.__file__).resolve().parent
    npu_root = Path(torch_npu.__file__).resolve().parent
    sdk = Path(os.environ['ASCEND_HOME_PATH']).resolve(strict=True)
    includes = include_paths() + [str(npu_root / 'include'), str(sdk / 'include')]
    libraries = library_paths() + [str(npu_root / 'lib')]
    library = output / 'liblingbot_stageB_guard_v2.so'
    command = ['g++', '-std=c++17', '-shared', '-fPIC', '-O2', '-pthread',
               f'-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}',
               str(source), '-o', str(library)]
    command += ['-I' + path for path in includes] + ['-L' + path for path in libraries]
    command += ['-Wl,-rpath,' + path for path in libraries]
    command += ['-ltorch_npu', '-ltorch_cpu', '-ltorch', '-lc10']
    headers = [npu_root / 'include/torch_npu/csrc/core/NPUStorageImpl.h',
               npu_root / 'include/torch_npu/csrc/aten/CustomFunctions.h',
               npu_root / 'include/third_party/acl/inc/acl/acl_base.h',
               npu_root / 'include/third_party/acl/inc/acl/acl_rt.h',
               sdk / 'include/acl/acl_base.h', sdk / 'include/acl/acl_rt.h']
    headers += [torch_root / 'include' / p for p in (
        'torch/custom_class.h', 'torch/custom_class_detail.h', 'torch/library.h',
        'ATen/core/ivalue.h', 'ATen/core/TensorBase.h', 'c10/core/TensorImpl.h',
        'c10/core/Storage.h', 'c10/core/StorageImpl.h', 'c10/core/GradMode.h', 'c10/core/InferenceMode.h')]
    header_pins = [pin(path) for path in headers]
    primary_so = [pin(torch_root / 'lib' / name) for name in ('libtorch.so', 'libtorch_cpu.so', 'libc10.so')]
    primary_so += [pin(npu_root / 'lib/libtorch_npu.so')]
    durable_json(output / 'compile-intent.json', dict(schema=SCHEMA, compiler_command=command,
                 abi_headers=header_pins, abi_libraries=primary_so, **before))
    terminal = dict(schema=SCHEMA, compiler_command=command, exit_code=None, error=None,
                    compilation_succeeded=False, registration_attempted=False)
    with (output / 'compiler.log').open('x') as log:
        try:
            process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
            terminal['exit_code'] = process.returncode
            terminal['compilation_succeeded'] = process.returncode == 0
        except BaseException as error:
            terminal['error'] = repr(error)
            raise
        finally:
            log.flush()
            os.fsync(log.fileno())
            durable_json(output / 'compile-result.json', terminal)
    # The compiler terminal result is durable BEFORE load_library or class tests.
    need(terminal['compilation_succeeded'], 'Host compilation failed; see preserved compiler.log/result')
    with library.open('rb') as stream:
        os.fsync(stream.fileno())
    receipt = dict(schema=SCHEMA, namespace=NAMESPACE, **before,
                   library=pin(library), abi_headers=header_pins, compiler_command=command,
                   torch=torch.__version__, torch_npu=torch_npu.__version__,
                   torch_npu_git=torch_npu.version.git_version, torch_git=torch.version.git_version,
                   cxx11_abi=int(torch._C._GLIBCXX_USE_CXX11_ABI), sdk=str(sdk),
                   compile_result=pin(output / 'compile-result.json'), registration=False,
                   npu_initialized=False, cpu_rejection_cases=[], device_admitted=False,
                   model_admitted=False, performance_admitted=False,
                   v1_qualification_inherited=False, guard_timing='per_layer_before_consumption')
    try:
        torch.ops.load_library(str(library))
        classes = getattr(torch.classes, NAMESPACE)
        with torch.inference_mode():
            cpu = torch.empty((1,), device='cpu', dtype=torch.bfloat16)
            for name, invoke in (
                ('Frame rejects CPU before NPU StorageImpl access', lambda: classes.Frame(cpu)),
                ('Entry rejects CPU before NPU StorageImpl access', lambda: classes.Entry(cpu, cpu, [cpu]*6)),
            ):
                rejected = False
                try:
                    invoke()
                except RuntimeError as error:
                    need('StageB guard requires a defined NPU tensor' in str(error),
                         'CPU test failed for an unexpected reason: ' + str(error))
                    rejected = True
                need(rejected, 'CPU owner unexpectedly accepted')
                receipt['cpu_rejection_cases'].append(dict(name=name, passed=True))
        need(not torch_npu.npu.is_initialized(), 'Registration/CPU rejection initialized NPU')
        for item in (*before.values(), *header_pins, *primary_so):
            need(pin(item['path']) == item, 'Source/ABI changed during build: ' + item['path'])
        receipt['abi_libraries'] = mapped_abi_libraries((torch_root, npu_root, sdk))
        mapped = {x['path'] for x in receipt['abi_libraries']}
        need(all(x['path'] in mapped for x in primary_so), 'Primary ABI library was not actually mapped')
        receipt['registration'] = True
    except BaseException as error:
        receipt['error'] = repr(error)
        raise
    finally:
        receipt['npu_initialized'] = torch_npu.npu.is_initialized()
        durable_json(output / 'build-receipt.json', receipt)
    return output / 'build-receipt.json'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--execute', action='store_true', help='Explicitly authorize host compilation/registration only')
    args = parser.parse_args()
    if not args.execute:
        parser.error('--execute is required; this script does not run device work')
    print(build(args.output))
