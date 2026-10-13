"""Evidence reader must reject missing chunks and bit-level differences."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

spec = importlib.util.spec_from_file_location('compare_captures', Path(__file__).with_name('compare_captures.py'))
compare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compare)


def make_capture(root):
    for rank in range(8):
        folder = root / f'rank{rank}' / 'session'
        folder.mkdir(parents=True)
        torch.save(torch.zeros(1, dtype=torch.bfloat16), folder / 'latent.pt')
        chunk = {'chunk': 0, 'latent_path': 'latent.pt'}
        if rank == 0:
            np.save(folder / 'rgb.npy', np.zeros((9, 480, 832, 3), dtype=np.uint8))
            chunk['rgb_path'] = 'rgb.npy'
        (folder / 'capture.json').write_text(json.dumps({'rank': rank, 'timing_eligible': False,
            'payload_directory': str(folder), 'chunks': [chunk]}))


def test_parity_and_negative_zero_are_distinguished(tmp_path):
    a, b = tmp_path / 'a', tmp_path / 'b'
    make_capture(a)
    make_capture(b)
    assert compare.compare(a, b, 1)['passed']
    torch.save(torch.tensor([-0.0], dtype=torch.bfloat16), b / 'rank7/session/latent.pt')
    with pytest.raises(ValueError, match='bit mismatch'):
        compare.compare(a, b, 1)


def test_missing_chunk_is_not_accepted(tmp_path):
    make_capture(tmp_path)
    with pytest.raises(ValueError, match='missing'):
        compare.compare(tmp_path, tmp_path, 80)
