"""Recompute the agreed window from ordered media arrivals, without pooling runs."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


def summarize(data, count):
    if data['config']['mux_fps'] != 12 or len(data['sessions']) != 1:
        raise ValueError('Expected one measured session encoded at12FPS')
    chunks = data['sessions'][0]['chunks']
    if [row['index'] for row in chunks] != list(range(count)):
        raise ValueError('Incomplete, duplicate or out-of-order chunks')
    if [row['num_frames'] for row in chunks] != [9] + [12] * (count - 1):
        raise ValueError('Unexpected frame counts')
    previous = 0.0
    for row in chunks:
        actual = (row['arrival_s'] - previous) * 1000
        if not math.isfinite(actual) or actual <= 0 or not math.isclose(actual, row['inter_arrival_ms'], abs_tol=1e-6):
            raise ValueError('Invalid or inconsistent serialized arrival intervals')
        previous = row['arrival_s']
    end = count - 2
    values = [row['inter_arrival_ms'] for row in chunks[11:end + 1]]
    if not values:
        raise ValueError('No main-window samples')
    ordered = sorted(values)
    position = .99 * (len(values) - 1)
    lo, hi = math.floor(position), math.ceil(position)
    p99 = ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)
    longest = streak = 0
    for value in values:
        streak = streak + 1 if value > 1000 else 0
        longest = max(longest, streak)
    mean = statistics.mean(values)
    return {'window_inclusive': [11, end], 'sample_count': len(values), 'mean_ms': mean,
            'p99_ms': p99, 'max_ms': max(values), 'over_1000ms': sum(x > 1000 for x in values),
            'longest_over_1000ms': longest, 'ttfc_ms': chunks[0]['arrival_s'] * 1000,
            'last_media_arrival_s': chunks[-1]['arrival_s'], 'total_frames': sum(x['num_frames'] for x in chunks),
            'single_run_threshold_pass': mean <= 900 and p99 <= 950,
            'final_acceptance': False, 'metric': 'client media inter-arrival; linear-interpolated P99',
            'limits': 'Requires separate launch/warmup/profiler-off evidence; no browser playback or multi-run claim'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=Path)
    parser.add_argument('--chunks', type=int, default=161)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = summarize(json.loads(args.input.read_text()), args.chunks)
    result.update(input=str(args.input.resolve()), input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest())
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
