"""Targeted six-run gate/evidence tests; execute only outside device timing."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import summarize_cross_model_screen as s


def runs(a=(940, 941, 942), b=(930, 931, 932), ap=(960, 961, 962), bp=(955, 956, 957)):
    table = {}
    for i in range(3):
        table[f'A{i+1}'] = dict(label=f'A{i+1}', mean_ms=a[i], p99_ms=ap[i])
        table[f'B{i+1}'] = dict(label=f'B{i+1}', mean_ms=b[i], p99_ms=bp[i])
    return [table[label] for label in s.ORDER]


class GateTest(unittest.TestCase):
    def test_three_pairs_pass_but_not_final(self):
        result = s.screen_gate(runs())
        self.assertTrue(result['screen_passed'])
        self.assertFalse(result['final_acceptance'])
        self.assertEqual(result['baseline_mean_raw_mad_ms'], 1)
        self.assertEqual(result['required_average_improvement_ms'], 5)

    def test_raw_mad_without_scaling_or_pooled_samples(self):
        result = s.screen_gate(runs(a=(920, 940, 960), b=(890, 910, 930)))
        self.assertEqual(result['baseline_mean_raw_mad_ms'], 20)
        self.assertEqual(result['required_average_improvement_ms'], 60)
        self.assertFalse(result['screen_passed'])

    def test_every_pair_must_improve_even_when_average_passes(self):
        self.assertFalse(s.screen_gate(runs(b=(940, 911, 912)))['screen_passed'])
        self.assertFalse(s.screen_gate(runs(b=(941, 911, 912)))['screen_passed'])

    def test_mean_threshold_inclusive_p99_tolerance_per_pair(self):
        self.assertTrue(s.screen_gate(runs(b=(935, 936, 937), bp=(965, 966, 967)))['screen_passed'])
        self.assertFalse(s.screen_gate(runs(b=(935.01, 936, 937)))['screen_passed'])
        self.assertFalse(s.screen_gate(runs(bp=(965.001, 950, 950)))['screen_passed'])

    def test_order_missing_and_duplicate_are_rejected(self):
        good = runs()
        for bad in (good[:-1], good[::-1], [good[0]] * 6):
            with self.assertRaises(ValueError):
                s.screen_gate(bad)

    def test_bad_scalar_not_silently_dropped(self):
        for value in (float('nan'), float('inf'), 0, -1, True):
            bad = copy.deepcopy(runs()); bad[1]['mean_ms'] = value
            with self.assertRaises(ValueError):
                s.screen_gate(bad)


class EvidenceTest(unittest.TestCase):
    def write(self, path, obj):
        path.write_text(json.dumps(obj))
        return s.read_pinned(path)[1]

    def test_config_pair_cannot_admit_qk_combined(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            raw = (b'lingbot_npu_recovery_cross: true\nlingbot_npu_recovery_cross_combined: false\n'
                   b'lingbot_npu_recovery_qk: true\nlingbot_npu_recovery_qk_combined: false\n')
            def pinbytes(path, data):
                path.write_bytes(data)
                return dict(path=str(path), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
            a = pinbytes(root / 'A.yaml', raw)
            b = pinbytes(root / 'B.yaml', raw.replace(b'cross: true', b'cross: false').replace(b'cross_combined: false', b'cross_combined: true'))
            freeze = dict(configs={'A': a, 'B': b}, manifest={}, image={}, integrity={}, model='model', source_config=a)
            plan = {k: freeze[k] for k in ('configs', 'manifest', 'image', 'integrity', 'model')}
            plan['config_freeze'] = self.write(root / 'freeze.json', freeze)
            s.validate_config_pair(plan)
            bad = (root / 'B.yaml').read_bytes().replace(b'qk_combined: false', b'qk_combined: true')
            freeze['configs']['B'] = pinbytes(root / 'B.yaml', bad)
            plan['config_freeze'] = self.write(root / 'freeze.json', freeze)
            with self.assertRaisesRegex(ValueError, 'more than cross'):
                s.validate_config_pair(plan)

    def fixture(self, root):
        plan = dict(port=7816, model='/model', image={'path': '/image'}, base_environment={'LINGBOT_VENV': '/testvenv'},
                    workload=dict(flow_shift=5.0, seed=42))
        command = ['/testvenv/bin/python', 'benchmarks/lingbot_world/benchmark_lingbot_world_realtime.py',
                   '--port', '7816', '--model', '/model', '--image', '/image', '--num-chunks', '40',
                   '--width', '832', '--height', '480', '--fps', '12', '--output-json', str(root/'rollout40.json'), '--print-chunks']
        document = dict(benchmark='lingbot_world_realtime', model='/model', endpoint='ws://127.0.0.1:7816/v1/realtime/video',
                        timestamp='2026-10-13T00:00:00+00:00',
                        workload=dict(flow_shift=5.0, fps=12, height=480, width=832, num_chunks=40, num_frames=477,
                                      prompt_updates=[], seed=42, video_seconds=477/12),
                        config=dict(camera_pattern='forward', chunk_slo_ms=None, mux_fps=12, playback_buffer_chunks=1,
                                    sessions=1, target_fps=12.0, warmup_chunks=6, warmup_sessions=0, workload_file=None),
                        sessions=[dict(request_id='fresh-request', chunks=[dict(index=i, arrival_s=float(i+1),
                                  inter_arrival_ms=1000.0, num_frames=9 if i == 0 else 12) for i in range(40)])])
        self.write(root/'rollout40.json', document)
        self.write(root/'smoke-command.json', command)
        self.write(root/'smoke-exit.json', {'exit_code': 0})
        return plan, document, command

    def test_rollout_recomputes_and_rejects_wrong_seed_or_image(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); plan, document, command = self.fixture(root)
            result = s.validate_rollout(plan, root, 40)
            self.assertEqual(result['summary']['mean_ms'], 1000)
            document['workload']['seed'] = 43
            self.write(root/'rollout40.json', document)
            with self.assertRaisesRegex(ValueError, 'seed'):
                s.validate_rollout(plan, root, 40)
            document['workload']['seed'] = 42; self.write(root/'rollout40.json', document)
            command[command.index('--image') + 1] = '/other-image'; self.write(root/'smoke-command.json', command)
            with self.assertRaisesRegex(ValueError, 'command'):
                s.validate_rollout(plan, root, 40)

    def test_rollout_bad_arrival_is_not_repaired(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); plan, document, _ = self.fixture(root)
            document['sessions'][0]['chunks'][20]['inter_arrival_ms'] = 1
            self.write(root/'rollout40.json', document)
            with self.assertRaises(ValueError):
                s.validate_rollout(plan, root, 40)

    def test_stage_from_old_plan_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); (root/'timing-A1').mkdir()
            self.write(root/'timing-A1/stage.json', dict(schema='lingbot_cross_model_stage_v1', completed=True,
                 stage='timing-A1', plan={'sha256': 'old-plan'}, manifest={}, final_acceptance=False, evidence=[]))
            with self.assertRaisesRegex(ValueError, 'old plan'):
                s.stage_record(dict(output=td, manifest={}), {'sha256': 'new-plan'}, 'timing-A1')


if __name__ == '__main__':
    unittest.main()
