"""Isolated v2 rank0/probe0+1 repeatable API and producer-release support.

No production entry point, HCCL, inflight/exhaustion coverage, timing or model
admission. Import/default invocation does not import torch or touch a device.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import traceback
from types import SimpleNamespace

from load_real_fixtures import contract, load_case, validate_case_metadata
from check_qk_real_matrix import check_owners, equal, identity, SHAPE, STRIDE, OWNER_ELEMENTS
from qk_replay_fixture import Runtime

SCHEMA = 'lingbot_qk_repeatable_pool_api_support_v2'
METHOD = 'rank0_two_real_probes_completed_reuse_producer_reclaim_two_phase_close_v2'
NAMESPACE = 'lingbot_qk_repeatable_pool_v2'
BUILD_SCHEMA = 'lingbot_qk_repeatable_event_pool_cpu_build_v2'
OWNERSHIP_POLICY = 'bounded_consumer_retirement_producer_reclaim_two_phase_close_v2'
INPUTS = ('qkv', 'query_weight', 'key_weight', 'expanded_cosine', 'expanded_sine')
_QUARANTINE = []


def need(value, message):
    if not value:
        raise RuntimeError('QK repeatable support v2: ' + message)


def pin(path):
    p = Path(path).resolve(strict=True)
    raw = p.read_bytes()
    return dict(path=str(p), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def verify(item, cache=None):
    cache = {} if cache is None else cache
    path = str(Path(item['path']).resolve(strict=True))
    if path not in cache:
        cache[path] = pin(path)
    need(cache[path] == item, 'source/library/fixture pin mismatch: ' + path)
    return cache[path]


def key(desc):
    return {name: desc[name] for name in (*contract.FIELDS[:9], 'dtype', 'device')}


def validate_pair(cases):
    need(len(cases) == 2, 'exactly two distinct real cases required')
    for index, case in enumerate(cases):
        validate_case_metadata(case)
        need(case['kind'] == 'qk' and case['rank'] == 0 and case['layer'] == 0
             and case['probe'] == index, 'only rank0/layer0 ordered probes0,1')
        need(case['actual_backend'] == 'lingbot_recovery_v1::qk_send'
             and case['epsilon'] == 1e-6 and case['head_num'] == 40 and case['ring_size'] == 1,
             'captured native ABI differs')
    need(all(cases[0][name] == cases[1][name] for name in ('request_id', 'chunk', 'kv_len')),
         'probe pair must come from same real request/chunk/history')
    need(cases[0]['dispatch_receipt']['build_manifest_sha256'] ==
         cases[1]['dispatch_receipt']['build_manifest_sha256'], 'mixed captured builds')
    for name in INPUTS:
        need(key(cases[0]['inputs']['descriptors'][name]) == key(cases[1]['inputs']['descriptors'][name]),
             'non-address pool key differs for ' + name)


def snapshot(ops, handle):
    names, values = list(ops.stat_names()), list(ops.stats(handle))
    need(len(names) == len(values) and len(set(names)) == len(names), 'invalid statistics API')
    flat = list(ops.initialization_status(handle))
    fields = ('plan', 'stage', 'stage1_status', 'repeatable_status', 'workspace_bytes')
    need(len(flat) % len(fields) == 0, 'invalid initialization status API')
    return dict(handle=handle, stats=dict(zip(names, values)),
                initialization=[dict(zip(fields, flat[i:i+5])) for i in range(0, len(flat), 5)])


def initialization_pass(item):
    rows = item['initialization']
    need([(r['plan'], r['stage']) for r in rows] == [(0, i) for i in range(4)],
         'capacity1 requires all four initialized native stages')
    need(all(r['stage1_status'] == 0 and r['repeatable_status'] == 0 for r in rows),
         'stage1/SetRepeatable rejected; terminal candidate failure')
    stats = item['stats']
    need(stats['capacity'] == stats['initialized'] == stats['accepting'] == 1
         and stats['retirement_capacity'] == 1
         and all(stats[name] == 0 for name in ('submitted', 'pending', 'poisoned', 'closed',
             'native_closed', 'finalizing', 'retired_count', 'retired_total',
             'retirement_overflows', 'producer_reclaimed_calls', 'finalize_released_calls')),
         'cold initialization state differs')


def validate_manifest_abi(manifest):
    need(manifest['schema'] == BUILD_SCHEMA and manifest['namespace'] == NAMESPACE
         and manifest['registration'] is True and manifest['npu_initialized'] is False
         and manifest['device_admitted'] is False and manifest['production_entrypoint'] is False
         and manifest['v1_qualification_inherited'] is False
         and manifest['ownership_policy'] == OWNERSHIP_POLICY,
         'unsupported v2 build receipt or inherited device admission')


def lifecycle_pass(stage, item, cold):
    """Scalar-only v2 lifecycle assertions; no device calls or inherited PASS."""
    stats = item['stats']
    need(stats['capacity'] == stats['initialized'] == stats['retirement_capacity'] == 1,
         'lifecycle capacity differs')
    need(stats['submitted'] == stats['callbacks_done'] == stats['generations'] == 2
         and stats['plan_reuses'] == stats['retired_total'] == stats['retired_high_water'] == 1,
         'lifecycle must follow exactly two completed calls and one retirement')
    need(all(stats[name] == 0 for name in ('pending', 'pool_exhaustions', 'poisoned',
                                        'retirement_overflows', 'event_not_ready', 'finalizing')),
         'lifecycle must remain healthy and non-pending')
    need(stats['producer_tid'] == cold['stats']['producer_tid'] > 0
         and stats['consumer_tid'] > 0, 'lifecycle thread identity differs')
    if stage == 'retired':
        need(stats['accepting'] == stats['retired_count'] == stats['retained_call_slots'] == 1
             and stats['producer_reclaimed_calls'] == stats['producer_reclaim_tid'] == 0
             and stats['closed'] == stats['native_closed'] == stats['finalize_released_calls'] == 0,
             'completed prior call was not retained before explicit reclaim')
    elif stage in ('reclaimed', 'native_closed', 'finalized'):
        need(stats['retired_count'] == 0 and stats['producer_reclaimed_calls'] == 1
             and stats['producer_reclaim_tid'] == stats['producer_tid'],
             'completed call was not reclaimed by original producer')
        if stage == 'reclaimed':
            need(stats['accepting'] == stats['retained_call_slots'] == 1
                 and stats['closed'] == stats['native_closed'] == stats['finalize_released_calls'] == 0,
                 'reclaim unexpectedly closed or released active call')
        elif stage == 'native_closed':
            need(stats['native_closed'] == stats['retained_call_slots'] == 1
                 and stats['accepting'] == stats['closed'] == stats['finalize_released_calls'] == 0,
                 'SDK close released framework owners or declared final close early')
        else:
            need(stats['native_closed'] == stats['closed'] == 1
                 and stats['accepting'] == stats['retained_call_slots'] == 0
                 and stats['finalize_released_calls'] == 2
                 and stats['finalize_tid'] == stats['producer_tid'],
                 'producer finalization did not release active plus cold calls')
    else:
        raise RuntimeError('unknown lifecycle phase: ' + stage)
    expected_bytes = 0 if stage == 'finalized' else cold['stats']['workspace_initial_bytes']
    need(stats['workspace_initial_bytes'] == cold['stats']['workspace_initial_bytes'] > 0
         and stats['workspace_resident_bytes'] == expected_bytes,
         'workspace ownership changed before producer finalization or remained after it')


def save(path, report):
    # The parent supervisor owns this unique path. Update checkpoints atomically.
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w') as output:
        json.dump(report, output, indent=2, allow_nan=False)
        output.write('\n')
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def full_arena(torch, backend, device, reference):
    arena = backend.empty_with_format((OWNER_ELEMENTS,), dtype=torch.bfloat16,
                                      device=device, acl_format=2)
    arena.fill_(-7)
    expected = torch.full((OWNER_ELEMENTS,), -7, dtype=torch.bfloat16, device='cpu')
    expected[256:-256].copy_(reference.detach().cpu().reshape(-1))
    return arena, expected


def check_arena_layout(provider, arena, owner, send):
    need(owner.data_ptr() == arena.data_ptr() and owner.untyped_storage().data_ptr() == arena.data_ptr(),
         'provided independent target owner was replaced')
    need(tuple(owner.shape) == (OWNER_ELEMENTS,) and owner.storage_offset() == 0
         and owner.untyped_storage().nbytes() == OWNER_ELEMENTS * 2
         and tuple(send.shape) == SHAPE and tuple(send.stride()) == STRIDE
         and send.storage_offset() == 256
         and send.untyped_storage().data_ptr() == owner.data_ptr(), 'padded send owner/view ABI differs')
    desc = dict(zip(contract.FIELDS, provider.describe(owner)))
    need(desc['npu_format'] == desc['origin_format'] == 2
         and list(desc['base_sizes']) == list(desc['physical_sizes']) == [OWNER_ELEMENTS]
         and list(desc['base_strides']) == [1], 'padded physical owner differs')
    return {name: list(value) if isinstance(value, (list, tuple)) else value for name, value in desc.items()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--manifest', required=True, type=Path, help='isolated CPU build receipt')
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--harness-sha256', required=True)
    parser.add_argument('--case', required=True, type=Path, action='append', help='ordered probe0 then probe1')
    parser.add_argument('--case-sha256', required=True, action='append')
    parser.add_argument('--output', required=True, type=Path, help='new receipt file, never overwritten at startup')
    args = parser.parse_args(argv)
    if not args.execute:
        print('NOT EXECUTED: explicit --execute, exact SHA gates and root-owned correctness supervisor required')
        return
    need(not args.output.exists() and not args.output.with_name(args.output.name + '.tmp').exists(),
         'refuse to overwrite existing receipt or checkpoint')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(schema=SCHEMA, method=METHOD, passed=False, stage='preflight', checks=[], snapshots=[], cases=[],
        actual_payload_cases=2, rank=0, real_payload_layers=[0], all40_payloads=False,
        actual192_global_proven=False, original_hccl_included=False, inflight_tested=False,
        exhaustion_tested=False, production_run_allocator_tested=False, performance_admitted=False,
        model_admitted=False, historical_identity_proven=False, pool_capacity=1, retirement_capacity=1,
        v1_qualification_inherited=False, hot_run_auto_reclaim_tested=False,
        explicit_producer_reclaim_tested=False, two_phase_close_tested=False,
        correctness_sync=True, retries=0, pid=os.getpid(), created_ns=time.time_ns(),
        supervisor_nonce=os.environ.get('ROOT_LINGBOT_SP8_FATAL_NONCE'),
        physical_devices=os.environ.get('ASCEND_RT_VISIBLE_DEVICES'),
        scope='two real probes; test-arena; completed-event reuse; explicit producer reclaim; SDK close then producer finalize')
    provider = ops = None
    retained, loaded_cases, initial_handles = [], [], []
    try:
        need(pin(__file__)['sha256'] == args.harness_sha256, 'harness SHA gate failed')
        manifest_pin = pin(args.manifest)
        need(manifest_pin['sha256'] == args.manifest_sha256, 'build receipt SHA gate failed')
        manifest = json.loads(args.manifest.read_text())
        validate_manifest_abi(manifest)
        cache = {}
        for name in ('source', 'builder', 'design', 'original_nonrepeatable_source', 'parent_v1_source',
                     'provider_manifest', 'kernel', 'library'):
            verify(manifest[name], cache)
        for item in (*manifest['dependencies'], *manifest['headers']):
            verify(item, cache)
        need(len(args.case) == len(args.case_sha256) == 2, 'two pinned cases required')
        case_pins = [pin(path) for path in args.case]
        need([item['sha256'] for item in case_pins] == args.case_sha256, 'fixture SHA gate failed')
        metadata = [json.loads(path.read_text()) for path in args.case]
        validate_pair(metadata)
        report.update(manifest=manifest_pin, case_files=case_pins,
            sources={name: pin(Path(__file__).parent / name) for name in
                     ('check_qk_repeatable_pool_support_v2.py', 'load_real_fixtures.py',
                      'check_qk_real_matrix.py', 'qk_replay_fixture.py')},
            capture_contract=pin(contract.__file__), non_address_keys_match=True)
        need(os.environ.get('ASCEND_RT_VISIBLE_DEVICES') == '8,9,10,11,12,13,14,15', 'reserved die mapping differs')
        need(int(os.environ.get('RANK', '0')) == int(os.environ.get('LOCAL_RANK', '0')) == 0
             and int(os.environ.get('WORLD_SIZE', '1')) == 1, 'rank0 single-process support only')
        need(os.environ.get('ROOT_LINGBOT_SP8_EXPERIMENT_MODE') == 'correctness', 'correctness supervisor required')
        need(os.environ.get('LINGBOT_RECOVERY_MANIFEST') == manifest['provider_manifest']['path']
             and os.environ.get('LINGBOT_RECOVERY_MANIFEST_SHA256') == manifest['provider_manifest']['sha256'],
             'parent provider environment differs')
        from vllm_omni.diffusion.layers import lingbot_sp8_fatal as fatal
        need(fatal.ensure_protocol(), 'launch-owned failure protocol required')
        save(args.output, report)
        import torch
        import torch_npu as backend
        need(str(torch.__version__) == manifest['torch'] and str(backend.__version__) == manifest['torch_npu']
             and backend.version.git_version == manifest['torch_npu_git'], 'framework ABI differs')
        need(not torch.are_deterministic_algorithms_enabled(), 'frozen nondeterministic policy required')
        torch.npu.set_device(0)
        torch.npu.set_compile_mode(jit_compile=False)
        torch.npu.config.allow_internal_format = True
        need(torch.npu.get_device_name() == 'Ascend910_9392', 'arch22 only')
        from vllm_omni.diffusion.layers.lingbot_recovery import load
        provider = load()
        torch.ops.load_library(manifest['library']['path'])
        ops = getattr(torch.ops, NAMESPACE)
        actual_dependencies = [str(Path(path).resolve(strict=True)) for path in ops.dependencies()]
        need(actual_dependencies == [item['path'] for item in manifest['dependencies']],
             'actual native/kernel symbol dependencies differ')
        need(list(ops.stat_names()) == manifest['statistics_fields'], 'statistics ABI differs')
        initial_handles = list(ops.handles())
        need(not initial_handles, 'fresh process must have no prior repeatable pools')
        with torch.inference_mode():
            for path, expected_pin in zip(args.case, case_pins):
                need(pin(path) == expected_pin, 'fixture metadata changed after preflight')
                source = load_case(path)
                loaded = source.materialize(torch, 'npu:0', provider.describe,
                    lambda shape, dtype, device, fmt: backend.empty_with_format(
                        shape, dtype=dtype, device=device, acl_format=fmt))
                loaded_cases.append(loaded)
                retained.extend((source, loaded))
                inputs = loaded['inputs']
                runtime = SimpleNamespace(torch=torch, npu=backend, raw=inputs['qkv'],
                    qw=inputs['query_weight'], kw=inputs['key_weight'],
                    cos=inputs['expanded_cosine'], sin=inputs['expanded_sine'])
                # Reuse frozen public-native reference, preserving full5120 RMS
                # and BF16 boundaries; captured native_send remains the oracle.
                original, native = Runtime.native(runtime, check_original_rms=False)
                retained.extend((runtime, original, native))
                equal(torch, native, loaded['outputs']['native_send'], 'public native equals captured native_send')
                equal(torch, inputs['cosine'].repeat_interleave(2, dim=-1).view_as(inputs['expanded_cosine']),
                      inputs['expanded_cosine'], 'captured cosine expansion')
                equal(torch, inputs['sine'].repeat_interleave(2, dim=-1).view_as(inputs['expanded_sine']),
                      inputs['expanded_sine'], 'captured sine expansion')
                check_owners(torch, loaded, SimpleNamespace(base=provider))
                report['cases'].append(dict(identity=identity(source.metadata), full_owner_sha256_verified=True,
                    reference='captured native_send plus frozen Runtime.native public-native equality', passed=False))
            for section, name in (('inputs', 'qkv'), ('outputs', 'native_send')):
                first = loaded_cases[0][section][name].detach().cpu().contiguous().view(torch.uint8)
                second = loaded_cases[1][section][name].detach().cpu().contiguous().view(torch.uint8)
                need(not torch.equal(first, second), 'real pair must differ in actual ' + name + ' values')
            for name in INPUTS:
                need(loaded_cases[0]['inputs'][name].untyped_storage().data_ptr() !=
                     loaded_cases[1]['inputs'][name].untyped_storage().data_ptr(), 'fresh input owner address required: ' + name)
            report['checks'].extend(('two_real_probes_raw_and_send_values_differ', 'all_five_input_addresses_are_fresh',
                                     'public_native_equals_captured', 'full_sources_and_descriptors_before'))
            report['stage'] = 'create'
            save(args.output, report)
            arguments = [tuple(loaded['inputs'][name] for name in INPUTS) for loaded in loaded_cases]
            torch.npu.synchronize()
            handle = ops.create(*arguments[0], 1, 1)
            cold = snapshot(ops, handle)
            report['snapshots'].append(dict(stage='create', **cold))
            initialization_pass(cold)
            save(args.output, report)
            prior = []
            for index, (args5, loaded) in enumerate(zip(arguments, loaded_cases)):
                arena, expected = full_arena(torch, backend, 'npu:0', loaded['outputs']['native_send'])
                retained.extend((arena, expected))
                need(all(arena.data_ptr() != old[0].data_ptr() for old in prior), 'arena address reused')
                report['stage'] = 'run' + str(index)
                save(args.output, report)
                torch.npu.synchronize()  # Correctness only; never hidden in a timing claim.
                send, owner = ops.run_with_test_arena(handle, *args5, arena)
                retained.extend((send, owner))
                torch.npu.synchronize()
                desc = check_arena_layout(provider, arena, owner, send)
                equal(torch, send, loaded['outputs']['native_send'], 'candidate send for real probe' + str(index))
                equal(torch, arena, expected, 'entire arena including both canaries probe' + str(index))
                for old_arena, old_expected, old_send, old_reference in prior:
                    equal(torch, old_arena, old_expected, 'retained first full arena unchanged')
                    equal(torch, old_send, old_reference, 'retained first send unchanged')
                for item in loaded_cases:
                    check_owners(torch, item, SimpleNamespace(base=provider))
                prior.append((arena, expected, send, loaded['outputs']['native_send']))
                current = snapshot(ops, handle)
                report['snapshots'].append(dict(stage='run' + str(index), **current))
                stats = current['stats']
                need(stats['submitted'] == stats['callbacks_done'] == stats['generations'] == index + 1
                     and stats['pending'] == stats['pool_exhaustions'] == stats['poisoned'] == 0,
                     'run did not complete cleanly')
                if index:
                    need(stats['plan_reuses'] == 1 and stats['event_queries'] >= 1
                         and stats['event_not_ready'] == 0, 'second call did not reuse completed plan')
                report['cases'][index].update(passed=True, arena_descriptor=desc)
                save(args.output, report)
            report['checks'].extend(('all_four_repeatable_stages_supported', 'two_fresh_full_arena_canary_passes',
                                     'native_bitwise_both_probes', 'completed_event_plan_reuse',
                                     'retained_first_output_unchanged', 'all_full_sources_and_descriptors_after'))
            # Two calls yield one completed retired Call. Exercise the same
            # producer reclaim implementation explicitly, without a third run.
            lifecycle_pass('retired', current, cold)
            report['stage'] = 'reclaim_retired'
            save(args.output, report)
            ops.reclaim_retired(handle)
            reclaimed = snapshot(ops, handle)
            report['snapshots'].append(dict(stage='reclaim_retired', **reclaimed))
            lifecycle_pass('reclaimed', reclaimed, cold)
            report['checks'].append('completed_retired_call_released_by_original_producer')
            report['explicit_producer_reclaim_tested'] = True
            # Same outputs remain externally retained; a pool-owner counter
            # does not prove the final physical allocator free completed.
            for arena, expected, send, reference in prior:
                equal(torch, arena, expected, 'full arena retained after producer reclaim')
                equal(torch, send, reference, 'send retained after producer reclaim')
            for item in loaded_cases:
                check_owners(torch, item, SimpleNamespace(base=provider))
            report['stage'] = 'close_native'
            save(args.output, report)
            torch.npu.synchronize()
            ops.close(handle)
            torch.npu.synchronize()
            native_closed = snapshot(ops, handle)
            report['snapshots'].append(dict(stage='close_native', **native_closed))
            lifecycle_pass('native_closed', native_closed, cold)
            report['checks'].append('native_close_kept_framework_owners_and_workspace')
            report['stage'] = 'finalize_close'
            save(args.output, report)
            ops.finalize_close(handle)
            finalized = snapshot(ops, handle)
            report['snapshots'].append(dict(stage='finalize_close_before_final_drain', **finalized))
            lifecycle_pass('finalized', finalized, cold)
            # Producer releases can enqueue allocator events. Drain explicitly
            # after finalization, never hide this in a performance claim.
            torch.npu.synchronize()
            drained = snapshot(ops, handle)
            report['snapshots'].append(dict(stage='finalize_close_after_final_drain', **drained))
            lifecycle_pass('finalized', drained, cold)
            report['two_phase_close_tested'] = True
            for arena, expected, send, reference in prior:
                equal(torch, arena, expected, 'full output arena retained after close')
                equal(torch, send, reference, 'send retained after close')
            for item in loaded_cases:
                check_owners(torch, item, SimpleNamespace(base=provider))
            report['checks'].extend(('successful_consumer_sdk_close_and_producer_finalize',
                                     'final_external_drain_outputs_and_sources_retained'))
            report.update(passed=True, stage='complete', finished_ns=time.time_ns())
            save(args.output, report)
            print(json.dumps(dict(passed=True, scope=METHOD, output=str(args.output))), flush=True)
    except BaseException as error:
        _QUARANTINE.append((provider, ops, retained, loaded_cases, error))
        report.update(passed=False, error=repr(error), traceback=traceback.format_exc(), finished_ns=time.time_ns(),
                      failure_cleanup='no retry/run/reclaim/close/finalize/device sync; retained owners; supervisor retires process')
        if ops is not None:
            try:
                report['failure_pool_snapshots'] = [snapshot(ops, h) for h in ops.handles() if h not in initial_handles]
            except BaseException as status_error:
                report['failure_snapshot_error'] = repr(status_error)
        save(args.output, report)
        if provider is not None:
            provider.poisoned = True
            provider.fatal.report_failure('qk_repeatable_pool_support_v2', repr(error), _QUARANTINE[-1])
        raise


if __name__ == '__main__':
    main()
