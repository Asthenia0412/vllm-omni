"""Independent stdlib audit of one TERMINAL StageB CPP operator round.

Never launches, waits for, signals, imports the runtime, or queries a device.
Run only after root reports terminal completion. V1 contributes methodology,
not a correctness, performance or ownership qualification for v2.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path

LENGTHS = [4680, 9360, 14040, 18720, 23400, 28080]
EXPECTED = [(length, probe) for length in LENGTHS for probe in range(4)]
STAGE_A_SHA = '3ea53af8a7f4bdc3e0a78137d235c8019c20f262549f6acc0c951c4c44ad24cc'


def need(condition, message):
    if not condition:
        raise ValueError(message)


class Audit:
    def __init__(self):
        self.actual = {}
        self.evidence = []

    def pin(self, path):
        path = Path(path).resolve(strict=True)
        key = str(path)
        if key not in self.actual:
            digest, size = hashlib.sha256(), 0
            with path.open('rb') as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(chunk)
                    size += len(chunk)
            self.actual[key] = dict(path=key, bytes=size, sha256=digest.hexdigest())
        return self.actual[key]

    def check(self, item):
        # Compare every declaration, even when this same path was seen before.
        need(self.pin(item['path']) == item, 'File pin mismatch: ' + item['path'])
        return item

    def read(self, path, *, evidence=False):
        if evidence:
            self.evidence.append(self.pin(path))
        return json.loads(Path(path).read_text())

    def pinned_json(self, item):
        self.check(item)
        return self.read(item['path'])


def review(plan_path, plan_sha, number):
    a = Audit()
    plan_path = Path(plan_path).resolve(strict=True)
    need(a.pin(plan_path)['sha256'] == plan_sha, 'Explicit frozen plan SHA differs')
    plan = a.read(plan_path, evidence=True)
    need(plan['schema'] == 'lingbot_actual_stageB_cpp_operator_plan_v2', 'Not v2 plan')
    need((plan['ranks'], plan['probes'], plan['actual_layers'], plan['histories'],
          plan['warmups'], plan['wall_samples'], plan['event_samples'], plan['rounds'])
         == (8, 4, 40, LENGTHS, 20, 30, 30, 3), 'Predeclared scope differs')
    need(plan['speed_gate'] == {'mean_ratio_max': .90, 'worst_case_ratio_max': 1.05}, 'Gate changed')
    stage = Path(plan['output']) / f'operator{number}'
    # Require terminal data FIRST, before potentially large ABI hashing.
    controller = a.read(stage / 'controller-result.json', evidence=True)
    need(controller['completed'] is True and controller['fatal'] is None
         and controller['remaining_owned_processes'] == [] and controller['intentional_retirement'] is True,
         'Controller not successfully terminal')
    launch = a.pinned_json(controller['launch'])
    terminal = a.pinned_json(controller['result'])
    run = Path(controller['launch']['path']).parent
    need(run.parent.resolve() == (stage / 'services').resolve() and run.name == launch['nonce'], 'Wrong service directory')
    need(terminal['nonce'] == launch['nonce'] and terminal['leader'] == launch['leader']
         and terminal['fatal'] is None and terminal['remaining_owned_processes'] == []
         and terminal['exit_code'] == controller['exit_code'], 'Supervisor identity/cleanup mismatch')
    need(controller['supervisor_return_code'] == terminal['exit_code'] % 256,
         'Supervisor process status does not match intentional terminal status')
    intended = ['vllm', 'serve', plan['model'], '--omni', '--deploy-config', plan['configs']['B']['path'],
                '--port', str(plan['port']), '--init-timeout', '1800', '--stage-init-timeout', '1800']
    need(launch['command'] == intended and launch['mode'] == 'profile'
         and launch['manifest'] == plan['manifest']['path'], 'Launch workload/binding mismatch')
    intent = a.read(stage / 'launch-intent.json', evidence=True)
    retired = a.read(stage / 'retirement-intent.json', evidence=True)
    need(intent['plan'] == a.pin(plan_path) and intent['round'] == number
         and retired['launch'] == launch and retired['request_completed'] is True
         and intent['begin_boot_ns'] < retired['begin_boot_ns'] <= controller['end_boot_ns'],
         'Launch/request/retirement ordering differs')
    completion = a.read(stage / 'checks-completed.json', evidence=True)
    need(completion['plan'] == a.pin(plan_path) and completion['launch'] == launch, 'Checks belong to another plan/service')

    for field in ('parent_plan', 'parent_baseline_config', 'env_script', 'integrity', 'image'):
        a.check(plan[field])
    for group in ('protected', 'harness', 'benchmark', 'launch_scripts'):
        for item in plan[group]:
            a.check(item)
    for item in plan['configs'].values():
        a.check(item)
    # Generated YAMLs use the identical baseline and differ in this one line.
    ca, cb = [Path(plan['configs'][arm]['path']).read_text() for arm in ('A', 'B')]
    flag = 'lingbot_npu_fixed_modulation_cpp_v2: '
    need(ca.count(flag + 'false') == 1 and cb.count(flag + 'true') == 1
         and ca.replace(flag + 'false', flag + 'true', 1) == cb, 'A/B differ beyond candidate flag')
    native = a.pinned_json(plan['manifest'])
    for field in ('source', 'library', 'kernels'):
        a.check(native[field])
    for field in ('kernel_sources', 'runtime_sources', 'abi_headers'):
        for item in native[field]:
            a.check(item)
    need(native['runtime_binding_only'] is True and native['recompiled'] is False
         and native['historical_admission_inherited'] is False, 'Native binding implies inherited/new qualification')
    cpp = a.pinned_json(plan['cpp_manifest'])
    build = a.pinned_json(plan['cpp_compilation'])
    need(cpp['schema'] == 'lingbot_stageB_guard_runtime_v2'
         and build['schema'] == 'lingbot_stageB_guard_cpu_build_v2'
         and cpp['namespace'] == build['namespace'] == 'lingbot_stageB_guard_v2'
         and cpp['compilation_ancestor'] == plan['cpp_compilation'], 'CPP build ancestry differs')
    need(build['registration'] is True and build['npu_initialized'] is False, 'CPP CPU registration differs')
    for record in (cpp, build):
        need(all(record[k] is False for k in ('device_admitted', 'model_admitted', 'v1_qualification_inherited'))
             and record['guard_timing'] == 'per_layer_before_consumption', 'Qualification/check timing differs')
    need(cpp['historical_admission_inherited'] is False and build['performance_admitted'] is False,
         'Prior qualification cannot be inherited')
    for field in ('source', 'library', 'abi_headers', 'abi_libraries', 'cxx11_abi', 'torch', 'torch_npu', 'sdk'):
        need(cpp[field] == build[field], 'CPP runtime/build differ: ' + field)
    need(cpp['runtime_sources'] == native['runtime_sources'], 'CPP/native runtime source binding differs')
    for field in ('source', 'library', 'builder', 'provenance', 'source_contract_tests', 'compile_result'):
        a.check(build[field])
    for field in ('abi_headers', 'abi_libraries'):
        need(bool(build[field]), 'Empty ABI coverage')
        for item in build[field]:
            a.check(item)
    compilation = a.pinned_json(build['compile_result'])
    need(compilation['compilation_succeeded'] is True and compilation['exit_code'] == 0
         and compilation['error'] is None and compilation['compiler_command'] == build['compiler_command'],
         'CPP compile terminal differs')
    need(build['cpu_rejection_cases'] == [
        {'name': 'Frame rejects CPU before NPU StorageImpl access', 'passed': True},
        {'name': 'Entry rejects CPU before NPU StorageImpl access', 'passed': True}], 'CPU refusal evidence differs')
    protected = {item['path']: item for item in plan['protected']}
    need(len(protected) == 228, 'Protected list not exactly228 unique files')
    parity_path = Path(plan['worktree']) / 'recovery/validation/historical-infrastructure-source-parity-20261013.json'
    parity = a.read(parity_path, evidence=True)
    need(parity['passed'] and len(parity['files']) == 228, 'Historical parity receipt incomplete')
    for item in parity['files']:
        path = str(Path(plan['worktree']) / item['relative_path'])
        need(protected[path]['sha256'] == item['historical_sha256'], 'Historical protected bytes differ')
    stage_a = a.pin(Path(plan['worktree']) / 'vllm_omni/diffusion/layers/lingbot_fixed_timestep_bank_stageA_v1.py')
    need(stage_a['sha256'] == STAGE_A_SHA and stage_a in native['runtime_sources'], 'Frozen StageA changed')

    observed = a.read(stage / 'service-environment.json', evidence=True)
    need(observed['launch'] == launch and observed['observed_alive'] is True, 'No launch-bound live environment record')
    env = observed['environment']
    dynamic = {'ROOT_LINGBOT_SP8_FATAL_BASE', 'ROOT_LINGBOT_SP8_FATAL_DIR',
               'ROOT_LINGBOT_SP8_FATAL_NONCE', 'ROOT_LINGBOT_SP8_EXPERIMENT_MODE'}
    expected_env = dict(plan['base_environment'], LINGBOT_STAGEB_OPERATOR_SCREEN_DIR=str(stage / 'checks'))
    need({k: v for k, v in env.items() if k not in dynamic} == expected_env
         and env['ROOT_LINGBOT_SP8_FATAL_NONCE'] == launch['nonce']
         and env['ROOT_LINGBOT_SP8_FATAL_DIR'] == str(run)
         and env['ROOT_LINGBOT_SP8_EXPERIMENT_MODE'] == 'profile'
         and env['VLLM_LINGBOT_NPU_ASYNC_METADATA'] == '0', 'Captured environment/nonce differs')
    need(a.read(run / 'smoke-exit.json', evidence=True)['exit_code'] == 0, 'Request failed')
    rollout = a.read(run / 'rollout8.json', evidence=True)
    need(rollout['model'] == plan['model'] and rollout['workload']['num_chunks'] == 8
         and rollout['workload']['num_frames'] == 93 and rollout['workload']['fps'] == 12
         and rollout['config']['mux_fps'] == 12 and rollout['config']['target_fps'] == 12
         and len(rollout['sessions']) == 1, 'Diagnostic request shape/encoding differs')
    chunks = rollout['sessions'][0]['chunks']
    need([c['index'] for c in chunks] == list(range(8))
         and [c['num_frames'] for c in chunks] == [9] + [12] * 7, 'Incomplete/unordered frame output')

    checks = completion['checks']
    need(len(checks) == 8, 'Missing rank receipts')
    screen_pin = a.pin(Path(plan['worktree']) / 'vllm_omni/diffusion/layers/lingbot_fixed_modulation_screen.py')
    reports = []
    for rank, item in enumerate(checks):
        need(Path(item['path']) == stage / f'checks/rank{rank}.json', 'Wrong/duplicate rank receipt path')
        report = a.pinned_json(item)
        a.evidence.append(item)
        need(report['schema'] == 'lingbot_actual_stageB_full_region_screen_v1'
             and report['passed'] is True and report['rank'] == rank and report['nonce'] == launch['nonce']
             and report['actual_layers'] == 40 and report['actual_histories'] == LENGTHS
             and report['actual_probes'] == 4 and report['warmups'] == 20
             and report['wall_measurements'] == report['event_measurements'] == 30
             and report['operator_only'] is True and report['model_timing_eligible'] is False
             and report['model_admitted'] is False, 'Rank correctness/scope/identity differs')
        need(report['source'] == screen_pin['path'] and report['source_sha256'] == screen_pin['sha256'],
             'Rank report source differs from pinned screen')
        need([(c['history'], c['probe']) for c in report['cases']] == EXPECTED, 'Actual case matrix differs')
        for case in report['cases']:
            need(case['correctness'] is True and set(case['arms']) == {'native', 'stageB'}, 'Correctness/arms differ')
            for samples in case['arms'].values():
                need(len(samples['wall_ns']) == len(samples['event_ms']) == 30, 'Not30 samples')
                need(all(type(v) is int and v > 0 for v in samples['wall_ns']), 'Wall samples not positive integer ns')
                need(all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in samples['event_ms']),
                     'Nonfinite/nonpositive event sample')
        reports.append(report)
    cases, native_total, candidate_total = [], 0, 0
    for index, (history, probe) in enumerate(EXPECTED):
        totals = {arm: [sum(r['cases'][index]['arms'][arm]['wall_ns']) for r in reports]
                  for arm in ('native', 'stageB')}
        baseline, candidate = max(totals['native']), max(totals['stageB'])
        native_total += baseline
        candidate_total += candidate
        cases.append(dict(history=history, probe=probe, native_ms=baseline / 30e6,
                          stageB_ms=candidate / 30e6, ratio=candidate / baseline,
                          gate_pass=20 * candidate <= 21 * baseline,
                          native_slowest_rank=totals['native'].index(baseline),
                          stageB_slowest_rank=totals['stageB'].index(candidate),
                          per_rank_mean_ms={arm: [v / 30e6 for v in sums] for arm, sums in totals.items()},
                          per_rank_event_mean_ms={arm: [sum(r['cases'][index]['arms'][arm]['event_ms']) / 30
                                                       for r in reports] for arm in ('native', 'stageB')}))
    mean_pass = 10 * candidate_total <= 9 * native_total
    worst = max(cases, key=lambda c: c['ratio'])
    worst_pass = all(c['gate_pass'] for c in cases)
    return dict(schema='lingbot_stageB_cpp_round_independent_review_v2', review_passed=True,
                plan=a.pin(plan_path), round=number, nonce=launch['nonce'], controller=a.pin(stage / 'controller-result.json'),
                verification=dict(files_verified=len(a.actual), protected228_exact_historical=True, stageA=stage_a,
                                  cpp_compilation=plan['cpp_compilation'], cpp_manifest=plan['cpp_manifest'],
                                  fatal=None, remaining_owned_processes=[], intentional_service_exit_code=terminal['exit_code'],
                                  supervisor_return_code=controller['supervisor_return_code']),
                method=dict(case_metric='max_rank(sum(all30 wall_ns)/30)', aggregate='equal-weight24 case means',
                            aggregate_gate='10*candidate_total <= 9*native_total (integer ns)',
                            worst_gate='every case20*candidate <=21*native (integer ns)', all_samples_included=True,
                            wall_event_batches='independent; never added or paired', ranks_not_independent_samples=True),
                metrics=dict(native_ms=native_total / (24 * 30e6), stageB_ms=candidate_total / (24 * 30e6),
                             ratio=candidate_total / native_total, aggregate_gate_pass=mean_pass,
                             worst_case_ratio=worst['ratio'], worst_case=[worst['history'], worst['probe']],
                             worst_case_gate_pass=worst_pass, failed_case_count=sum(not c['gate_pass'] for c in cases)),
                cases=cases, operator_round_gate_pass=mean_pass and worst_pass, three_round_pass=False,
                model_admitted=False, end_to_end_performance_claim=False,
                request=dict(chunks=8, frames=93, encoding_fps=12, model_timing_eligible=False),
                scope=dict(actual_rank_history_probe_contexts=192, layers=40, logical_views_per_layer=6,
                           replay='actual active StageA tokens/source owners and post-SP snapshots under pinned source guards',
                           history='modulation does not depend on KV length; not192 independent attention workloads'),
                limitations=['logical output/source equality does not prove complete arenas or canaries',
                             'no standalone payload fixture export or arbitrary inference-content mutation proof',
                             'one operator round; no three-round/model80/model600/playback/lifecycle admission',
                             'CPU registrations and v1 results do not transfer qualification',
                             'source/ABI hashes are post-terminal evidence plus pinned pre/post controller enforcement'],
                evidence=a.evidence)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--round', type=int, choices=(1, 2, 3), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    need(not args.output.exists(), 'Refuse review output overwrite')
    result = review(args.plan, args.plan_sha256, args.round)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(result['metrics'], allow_nan=False))
