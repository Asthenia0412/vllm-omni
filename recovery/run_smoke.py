"""Wait for an owned recovery service, then check a 16-chunk SP8 rollout."""
import argparse
import json
from pathlib import Path
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
    parser.add_argument('--chunks', type=int, default=16)
    args = parser.parse_args()
    launch = json.loads((args.run / 'launch.json').read_text())
    leader = launch['leader']
    for _ in range(360):
        if (args.run / 'result.json').exists() or identity(leader['pid']) != leader:
            raise SystemExit('Owned service exited; no replacement service will be tested')
        integrity = json.loads(args.integrity.read_text())
        if integrity['complete'] and not integrity['passed']:
            raise SystemExit('Model integrity failed')
        try:
            if integrity['complete']:
                with urllib.request.urlopen(f'http://127.0.0.1:{args.port}/health', timeout=2) as response:
                    if response.status == 200:
                        break
        except OSError:
            pass
        time.sleep(5)
    else:
        raise SystemExit('Service readiness timeout; inspect the existing launch')
    command = [sys.executable, 'benchmarks/lingbot_world/benchmark_lingbot_world_realtime.py',
               '--port', str(args.port), '--model', str(args.model), '--image', str(args.image),
               '--num-chunks', str(args.chunks), '--width', '832', '--height', '480', '--fps', '12',
               '--output-json', str(args.run / f'rollout{args.chunks}.json'), '--print-chunks']
    (args.run / 'smoke-command.json').write_text(json.dumps(command, indent=2) + '\n')
    with (args.run / 'smoke.log').open('w') as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    (args.run / 'smoke-exit.json').write_text(json.dumps({'exit_code': result.returncode}) + '\n')
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
