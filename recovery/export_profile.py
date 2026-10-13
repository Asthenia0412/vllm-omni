"""Offline export with a scoped fix for torch_npu's first-'tools' path split.

Does not mutate installed torch_npu/CANN. Uses the msprof binary and analysis
script from the same SDK; validates all eight rank exports before success.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def sdk_script(msprof, relative):
    binary = Path(msprof).resolve(strict=True)
    # .../<sdk>/tools/profiler/bin/msprof
    if tuple(binary.parts[-4:]) != ('tools', 'profiler', 'bin', 'msprof'):
        raise ValueError(f'Unexpected msprof layout: {binary}')
    candidate = binary.parents[3] / relative
    return str(candidate) if candidate.is_file() else ''


def main():
    p = argparse.ArgumentParser()
    p.add_argument('root', type=Path)
    args = p.parse_args()
    from torch_npu.profiler.analysis.prof_view.cann_parse._cann_export import CANNExportParser
    from torch_npu.profiler.profiler import analyse
    binary = shutil.which('msprof')
    if not binary:
        raise RuntimeError('Source the matching SDK environment first')
    script = sdk_script(binary, CANNExportParser._MSPROF_PY_PATH)
    if not script:
        raise RuntimeError('Matching SDK analysis script missing')
    CANNExportParser._get_msprof_script_path = lambda self, rel: sdk_script(self.msprof_path, rel)
    receipt = {'msprof': str(Path(binary).resolve()), 'analysis_script': script,
               'analysis_script_sha256': hashlib.sha256(Path(script).read_bytes()).hexdigest(),
               'ranks': [], 'passed': False}
    destination = args.root / 'export-receipt.json'
    for rank in range(8):
        folder = args.root / f'diffusion_rank{rank}'
        if not folder.is_dir():
            raise RuntimeError(f'Missing rank {rank}')
        print(f'EXPORT_START rank={rank}', flush=True)
        analyse(str(folder), max_process_number=1, export_type='text')
        traces = list(folder.glob('*/ASCEND_PROFILER_OUTPUT/trace_view.json'))
        kernels = list(folder.glob('*/ASCEND_PROFILER_OUTPUT/kernel_details.csv'))
        item = {'rank': rank, 'traces': [str(x) for x in traces],
                'kernels': [str(x) for x in kernels],
                'files_present': len(traces) == 1 and len(kernels) == 1}
        receipt['ranks'].append(item)
        destination.write_text(json.dumps(receipt, indent=2)+'\n')
        if not item['files_present']:
            raise RuntimeError(f'Incomplete rank {rank} export; inspect export logs')
        print(f'EXPORT_DONE rank={rank}', flush=True)
    receipt['passed'] = True
    destination.write_text(json.dumps(receipt, indent=2)+'\n')


if __name__ == '__main__':
    main()
