"""Warm40 then capture a complete16-chunk request on an owned profile service."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.request

from supervise import identity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--integrity', type=Path, required=True)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--port', type=int, default=7816)
    args = parser.parse_args()
    launch = json.loads((args.run / 'launch.json').read_text())
    if launch['mode'] != 'profile':
        raise ValueError('A separate profile service is required')
    base = [sys.executable, 'recovery/run_smoke.py', '--run', str(args.run),
            '--model', str(args.model), '--integrity', str(args.integrity),
            '--image', str(args.image), '--port', str(args.port)]
    receipt = {'timing_eligible': False, 'attribution_window': [11, 14], 'warmup_chunks': 40,
               'capture_chunks': 16, 'events': []}

    def api(name):
        if identity(launch['leader']['pid']) != launch['leader']:
            raise RuntimeError('Owned service identity changed')
        start = time.monotonic_ns()
        request = urllib.request.Request(f'http://127.0.0.1:{args.port}/{name}',
                                         data=b'{"stages":[0]}', headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=600) as response:
            body = json.load(response)
        receipt['events'].append({'action': name, 'start_ns': start, 'end_ns': time.monotonic_ns(), 'body': body})
        (args.run / 'profile-controller.json').write_text(json.dumps(receipt, indent=2) + '\n')
        if body.get('status') != 'SUCCESS':
            raise RuntimeError(f'Profiler action failed: {body}')

    subprocess.run(base + ['--chunks', '40'], check=True)
    for name in ['smoke.log', 'smoke-command.json', 'smoke-exit.json']:
        shutil.copy2(args.run / name, args.run / ('warm40-' + name))
    api('start_profile')
    try:
        subprocess.run(base + ['--chunks', '16'], check=True)
    finally:
        api('stop_profile')
    receipt['capture_request_completed'] = True
    (args.run / 'profile-controller.json').write_text(json.dumps(receipt, indent=2) + '\n')


if __name__ == '__main__':
    main()
