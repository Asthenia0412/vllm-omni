import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('summarize_timing', Path(__file__).with_name('summarize_timing.py'))
reader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reader)


def data():
    arrival = 0
    rows = []
    for index in range(161):
        delta = 900 + index
        arrival += delta / 1000
        rows.append(dict(index=index, arrival_s=arrival, inter_arrival_ms=delta,
                         num_frames=9 if index == 0 else 12))
    return {'config': {'mux_fps': 12}, 'sessions': [{'chunks': rows}]}


def test_exact_window_and_linear_percentile():
    result = reader.summarize(data(), 161)
    assert result['window_inclusive'] == [11, 159]
    assert result['sample_count'] == 149
    assert result['mean_ms'] == 985
    assert result['p99_ms'] == pytest.approx(1057.52)
    assert result['longest_over_1000ms'] == 59
    assert not result['final_acceptance']


def test_missing_chunk_fails():
    source = data()
    source['sessions'][0]['chunks'].pop(20)
    with pytest.raises(ValueError, match='chunks'):
        reader.summarize(source, 161)


def test_inconsistent_arrival_is_rejected():
    source = data()
    source['sessions'][0]['chunks'][20]['inter_arrival_ms'] = 1
    with pytest.raises(ValueError, match='intervals'):
        reader.summarize(source, 161)
