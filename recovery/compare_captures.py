"""Compare all-rank model latents and primary RGB from separate correctness runs."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch


def session(root, rank, chunks):
    manifests = list((root / f'rank{rank}').glob('*/capture.json'))
    if len(manifests) != 1:
        raise ValueError(f'{root}: expected exactly one session for rank {rank}')
    path = manifests[0]
    data = json.loads(path.read_text())
    if data['rank'] != rank or data['timing_eligible']:
        raise ValueError(f'{path}: invalid rank or diagnostic timing classification')
    if [x['chunk'] for x in data['chunks']] != list(range(chunks)):
        raise ValueError(f'{path}: missing, duplicate or out-of-order chunks')
    return Path(data['payload_directory']), data['chunks']


def compare(reference, candidate, chunks):
    checked = []
    for rank in range(8):
        left_root, left = session(reference, rank, chunks)
        right_root, right = session(candidate, rank, chunks)
        frames = 0
        for a, b in zip(left, right):
            x = torch.load(left_root / a['latent_path'], map_location='cpu', weights_only=True)
            y = torch.load(right_root / b['latent_path'], map_location='cpu', weights_only=True)
            if x.shape != y.shape or x.dtype != y.dtype or not torch.isfinite(x).all() or not torch.isfinite(y).all():
                raise ValueError(f'Invalid latent shape/dtype/finiteness: rank {rank}, chunk {a["chunk"]}')
            if not torch.equal(x.contiguous().view(torch.uint8), y.contiguous().view(torch.uint8)):
                raise ValueError(f'Latent bit mismatch: rank {rank}, chunk {a["chunk"]}')
            if rank == 0:
                rgb_a = np.load(left_root / a['rgb_path'], allow_pickle=False)
                rgb_b = np.load(right_root / b['rgb_path'], allow_pickle=False)
                expected_frames = 9 if a['chunk'] == 0 else 12
                expected_shape = (expected_frames, 480, 832, 3)
                if (rgb_a.shape != expected_shape or rgb_b.shape != expected_shape
                        or rgb_a.dtype != np.uint8 or rgb_b.dtype != np.uint8
                        or not np.array_equal(rgb_a, rgb_b)):
                    raise ValueError(f'RGB bit/shape mismatch: chunk {a["chunk"]}')
                frames += len(rgb_a)
            elif 'rgb_path' in a or 'rgb_path' in b:
                raise ValueError('Nonprimary rank unexpectedly produced RGB')
        checked.append({'rank': rank, 'latent_chunks': chunks, 'rgb_frames': frames})
    return {'passed': True, 'scope': 'paired model latent/RGB bit parity only; not arena/canary, quality or timing acceptance',
            'reference': str(reference), 'candidate': str(candidate), 'ranks': checked}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--chunks', type=int, default=80)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        result = compare(args.reference, args.candidate, args.chunks)
    except Exception as error:
        args.output.write_text(json.dumps({'passed': False, 'error': str(error)}, indent=2) + '\n')
        raise
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
