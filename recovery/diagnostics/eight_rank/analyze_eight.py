"""Read-only, parameterized eight-rank exported CANN dependency analysis.

Kernel CSV intervals alone define utilization partitions. TASK and host wait
records are diagnostic evidence, never an additional latency contribution.
This program does not establish clock calibration or CPU causality.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import sqlite3
import statistics

EXPECTED_RANKS = tuple(range(8))
COMM_TOKENS = ('hccl', 'hcom', 'alltoall', 'all_to_all', 'allgather',
               'all_gather', 'allreduce', 'all_reduce', 'reducescatter',
               'reduce_scatter', 'batchsendrecv', 'batch_send_recv')
LIMITATIONS = [
    'Cross-rank timestamp comparisons assume exporter alignment; calibration error and clock alignment are unproven.',
    'Communication operation start is an exported interval boundary, not packet-transfer start or proven rank arrival.',
    'Kernel-name classification separates communication from other kernels, including copies; neither is not proven CPU idle.',
    'TASK intervals can include communication subtasks, runtime waits and overlapping records; they are diagnostic only.',
    'Host synchronize duration, communication duration, TASK waits and kernel workloads overlap and must not be added.',
    'Explicit event-handle record-to-wait identity and causal blocking events remain unproven.',
    'Nearest send/receive kernels and temporal overlap are context, not a proven producer-consumer pairing.',
    'CPU scheduling, allocation, GC and the cause of rank lateness are unproven without further evidence.',
    'Profiling changes timing; these windows do not qualify clean-service performance or explain all historical latency differences.',
]


def need(value, message):
    if not value:
        raise ValueError(message)


def integer(value):
    number = Decimal(str(value).strip())
    need(number.is_finite() and number == number.to_integral_value(),
         'Noninteger/nonfinite nanosecond or identifier: ' + str(value))
    return int(number)


def us_to_ns(value):
    number = Decimal(str(value).strip()) * 1000
    need(number.is_finite() and number == number.to_integral_value(),
         'Subnanosecond/nonfinite microsecond timestamp: ' + str(value))
    return int(number)


def optional_int(value):
    try:
        return integer(value)
    except (InvalidOperation, ValueError):
        return None


def pin(path):
    path = Path(path).resolve(strict=True)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest.hexdigest())


def union_ns(intervals):
    result = 0
    left = right = None
    for start, end in sorted(intervals):
        need(start <= end, 'Reversed interval')
        if end == start:
            continue
        if left is None:
            left, right = start, end
        elif start <= right:
            right = max(right, end)
        else:
            result += right - left
            left, right = start, end
    return result + (right - left if left is not None else 0)


def partition(kernels, lo, hi):
    need(lo < hi, 'Nonpositive measurement window')
    compute, comm = [], []
    for row in kernels:
        start, end = max(lo, row['start_ns']), min(hi, row['end_ns'])
        if start < end:
            (comm if row['communication'] else compute).append((start, end))
    a, b, both_union = union_ns(compute), union_ns(comm), union_ns(compute + comm)
    overlap = a + b - both_union
    result = dict(window_ns=hi-lo, compute_union_ns=a, communication_union_ns=b,
                  overlap_ns=overlap, compute_only_ns=a-overlap,
                  communication_only_ns=b-overlap, neither_ns=hi-lo-both_union,
                  kernel_union_ns=both_union)
    need(result['compute_only_ns'] + result['communication_only_ns'] + overlap + result['neither_ns'] == hi-lo,
         'Interval partition does not conserve time')
    return result


def table_exists(db, name):
    return db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone() is not None


def rows(db, sql, arguments=()):
    return [dict(row) for row in db.execute(sql, arguments)]


def intervals(row):
    result = dict(row)
    result['start_ns'], result['end_ns'] = integer(row['startNs']), integer(row['endNs'])
    need(result['start_ns'] <= result['end_ns'], 'Reversed database interval')
    return result


def decode(strings, value):
    return strings.get(value, '<unresolved:' + str(value) + '>')


def find_exports(root):
    exports, issues = {}, []
    for rank in EXPECTED_RANKS:
        directory = root / f'diffusion_rank{rank}'
        candidates = sorted(directory.glob('*/ASCEND_PROFILER_OUTPUT/ascend_pytorch_profiler_*.db'))
        if len(candidates) != 1:
            issues.append(dict(rank=rank, kind='missing_export' if not candidates else 'ambiguous_exports',
                               candidates=[str(p) for p in candidates]))
            continue
        csv_path = candidates[0].parent / 'kernel_details.csv'
        if not csv_path.is_file():
            issues.append(dict(rank=rank, kind='missing_kernel_csv', path=str(csv_path)))
            continue
        exports[rank] = (candidates[0], csv_path)
    return exports, issues


def open_db(path):
    db = sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True)
    db.row_factory = sqlite3.Row
    return db


def read_strings(db):
    need(table_exists(db, 'STRING_IDS'), 'Missing STRING_IDS')
    return dict(db.execute('SELECT id,value FROM STRING_IDS'))


def rgb_windows(db, strings, first, last):
    need(1 <= first <= last, 'Window requires 1 <= first <= last')
    ids = [key for key, value in strings.items() if value == 'lingbot.rgb_copy']
    need(len(ids) == 1, 'Missing/ambiguous lingbot.rgb_copy marker name')
    markers = [intervals(row) for row in rows(db, 'SELECT * FROM PYTORCH_API WHERE name=?', (ids[0],))]
    markers.sort(key=lambda row: (row['start_ns'], row['end_ns']))
    need(len(markers) > last, f'Insufficient RGB markers: need index {last}, found {len(markers)}')
    need(len({(r['start_ns'], r['end_ns']) for r in markers}) == len(markers), 'Duplicate RGB marker intervals')
    need(all(markers[i]['end_ns'] < markers[i+1]['end_ns'] for i in range(len(markers)-1)),
         'RGB completion order is not strictly increasing')
    return [dict(chunk=chunk, start_ns=markers[chunk-1]['end_ns'], end_ns=markers[chunk]['end_ns'])
            for chunk in range(first, last+1)], dict(source='rank0 PYTORCH_API lingbot.rgb_copy completion',
                marker_count=len(markers), chunk_index_is_ordered_marker_index=True,
                explicit_request_chunk_id_proven=False)


def uint8_copy_windows(db, strings, first, last, expected_chunks, height=480, width=832):
    """Explicit reconstruction only; never silently replace missing markers."""
    need(1 <= first <= last and expected_chunks is not None and expected_chunks > last,
         'RGB-copy reconstruction requires explicit expected chunk count > last index')
    ids = [key for key, value in strings.items() if value == 'aten::_to_copy']
    need(len(ids) == 1, 'Missing/ambiguous aten::_to_copy operator name')
    markers = []
    for row in rows(db, 'SELECT * FROM PYTORCH_API WHERE name=?', (ids[0],)):
        shape = decode(strings, row.get('inputShapes')).split(';', 1)[0].strip().strip('"')
        dtype = decode(strings, row.get('inputDtypes')).split(';', 1)[0].strip().lower()
        if shape not in (f'9,{height},{width},3', f'12,{height},{width},3') or dtype not in ('unsigned char', 'uint8', 'byte'):
            continue
        item = intervals(row)
        item['frame_count'] = int(shape.split(',')[0])
        markers.append(item)
    markers.sort(key=lambda row: (row['start_ns'],row['end_ns']))
    need(len(markers) == expected_chunks,
         f'Reconstructed RGB-copy count differs: expected {expected_chunks}, found {len(markers)}')
    need([row['frame_count'] for row in markers] == [9] + [12]*(expected_chunks-1),
         'RGB-copy frame pattern must be one initial9 followed by complete12-frame chunks')
    need(len({row['globalTid'] for row in markers}) == 1, 'RGB-copy reconstruction spans multiple host threads')
    need(all(markers[i]['end_ns'] <= markers[i+1]['start_ns'] for i in range(len(markers)-1)),
         'RGB-copy reconstruction contains overlapping or duplicate calls')
    return [dict(chunk=chunk,start_ns=markers[chunk-1]['end_ns'],end_ns=markers[chunk]['end_ns'])
            for chunk in range(first,last+1)], dict(
        source='EXPLICIT RECONSTRUCTION: rank0 shaped uint8 aten::_to_copy completions',
        marker_count=len(markers),expected_chunk_count=expected_chunks,
        frame_counts=[row['frame_count'] for row in markers],height=height,width=width,
        boundary_reconstructed=True, direction_is_D2H_proven=False,
        complete_chunk_boundaries_independently_proven=False,
        explicit_request_chunk_id_proven=False,
        assumption='Unique ordered RGB-shaped copies correspond to complete request outputs; source/destination and request identity need independent confirmation')


def read_kernels(path, lo, hi):
    result, issues = [], []
    with Path(path).open(newline='') as stream:
        reader = csv.DictReader(stream)
        required = {'Start Time(us)', 'Duration(us)', 'Name', 'Stream ID', 'Task ID'}
        need(required.issubset(reader.fieldnames or []), 'Unsupported kernel CSV columns')
        for number, row in enumerate(reader, 2):
            try:
                start, duration = us_to_ns(row['Start Time(us)']), us_to_ns(row['Duration(us)'])
                need(duration >= 0, 'Negative kernel duration')
            except (ValueError, InvalidOperation) as error:
                issues.append(dict(kind='invalid_kernel_timestamp', line=number, error=str(error)))
                continue
            end = start + duration
            if start < hi and end > lo:
                text = (row['Name'] + ' ' + row.get('Type', '')).lower()
                result.append(dict(name=row['Name'], start_ns=start, end_ns=end,
                    stream=optional_int(row['Stream ID']), task=optional_int(row['Task ID']),
                    communication=any(token in text for token in COMM_TOKENS), csv_line=number))
    result.sort(key=lambda row: (row['start_ns'], row['end_ns']))
    return result, issues


def read_comms(db, strings, lo, hi):
    result = []
    for raw in rows(db, 'SELECT * FROM COMMUNICATION_OP WHERE startNs<? AND endNs>?', (hi, lo)):
        row = intervals(raw)
        count = Decimal(str(raw['count']))
        need(count.is_finite(), 'Nonfinite communication count')
        row.update(name=decode(strings, raw['opName']), group=decode(strings, raw['groupName']),
                   count_key=format(count.normalize(), 'f'),
                   key_resolved=raw['opName'] in strings and raw['groupName'] in strings,
                   boundary=not (lo <= row['start_ns'] and row['end_ns'] <= hi))
        result.append(row)
    return result


def collective_key(row):
    return row['group'], row['name'], row['count_key']


def key_json(key):
    return dict(group=key[0], operation=key[1], count=key[2])


def match_collectives(per_rank, windows):
    by_rank, boundaries, unresolved = {}, [], []
    for rank, entries in per_rank.items():
        index = defaultdict(list)
        for entry in entries:
            if not entry.get('key_resolved',True):
                unresolved.append(dict(rank=rank,key=key_json(collective_key(entry))))
                continue
            index[collective_key(entry)].append(entry)
            if entry['boundary']:
                boundaries.append(dict(rank=rank, key=key_json(collective_key(entry)),
                                       start_ns=entry['start_ns'], end_ns=entry['end_ns']))
        by_rank[rank] = index
    all_keys = set().union(*(set(index) for index in by_rank.values())) if by_rank else set()
    matched, missing, duplicate, excluded = [], [], [], []
    for key in sorted(all_keys):
        absent = [rank for rank in EXPECTED_RANKS if not by_rank.get(rank, {}).get(key)]
        repeats = {rank: len(by_rank.get(rank, {}).get(key, [])) for rank in EXPECTED_RANKS
                   if len(by_rank.get(rank, {}).get(key, [])) > 1}
        if absent:
            missing.append(dict(key=key_json(key), missing_ranks=absent))
        if repeats:
            duplicate.append(dict(key=key_json(key), per_rank_counts=repeats))
        if absent or repeats:
            continue
        entries = [by_rank[rank][key][0] for rank in EXPECTED_RANKS]
        if any(row['boundary'] for row in entries):
            excluded.append(dict(key=key_json(key), boundary_ranks=[r for r, row in enumerate(entries) if row['boundary']]))
            continue
        starts, ends = [r['start_ns'] for r in entries], [r['end_ns'] for r in entries]
        chunks = [[w['chunk'] for w in windows if w['start_ns'] <= row['start_ns'] and row['end_ns'] <= w['end_ns']]
                  for row in entries]
        same_chunk = all(len(x) == 1 for x in chunks) and len({x[0] for x in chunks}) == 1
        matched.append(dict(key=key_json(key), chunk=chunks[0][0] if same_chunk else None,
            per_rank_containing_chunks=chunks, crosses_chunk_boundary_or_rank_assignment=not same_chunk,
            exported_start_spread_ns=max(starts)-min(starts), exported_end_spread_ns=max(ends)-min(ends),
            last_start_ranks=[r for r, value in enumerate(starts) if value == max(starts)],
            per_rank=[dict(rank=r, start_ns=row['start_ns'], end_ns=row['end_ns'],
                          duration_ns=row['end_ns']-row['start_ns'], connection_id=row['connectionId'],
                          op_id=row.get('opId')) for r, row in enumerate(entries)]))
    matched.sort(key=lambda row: (-row['exported_start_spread_ns'], str(row['key'])))
    return dict(matched=matched, missing_keys=missing, duplicate_keys=duplicate,
                unresolved_keys=unresolved, global_window_boundary_rows=boundaries, excluded_boundary_keys=excluded)


def named_api(raw, strings):
    row = intervals(raw)
    return dict(name=decode(strings, row['name']), start_ns=row['start_ns'], end_ns=row['end_ns'],
                duration_ns=row['end_ns']-row['start_ns'], global_tid=row.get('globalTid'),
                thread_low32=(row['globalTid'] % (2**32)) if row.get('globalTid') is not None else None,
                connection_id=row.get('connectionId'))


def kernel_context(kernel):
    return {k: kernel[k] for k in ('name', 'start_ns', 'end_ns', 'stream', 'task', 'csv_line')}


def task_rows(db, strings, lo, hi):
    if not table_exists(db, 'TASK'):
        return []
    result = []
    for row in rows(db, 'SELECT * FROM TASK WHERE startNs<? AND endNs>? ORDER BY startNs', (hi, lo)):
        item = intervals(row)
        result.append(dict(start_ns=item['start_ns'], end_ns=item['end_ns'],
            stream=item.get('streamId'), task=item.get('taskId'), global_task_id=item.get('globalTaskId'),
            connection_id=item.get('connectionId'), task_type=decode(strings, item.get('taskType'))))
    return result


def anchor_task(db, strings, kernel):
    if kernel is None or kernel['stream'] is None or kernel['task'] is None or not table_exists(db, 'TASK'):
        return dict(status='unavailable', candidates=[])
    candidates = [row for row in task_rows(db, strings, kernel['start_ns'], kernel['end_ns'])
                  if row['stream'] == kernel['stream'] and row['task'] == kernel['task']]
    return dict(status='unique_stream_task_interval' if len(candidates) == 1 else
                ('missing' if not candidates else 'ambiguous'), candidates=candidates)


def collective_detail(db, strings, op, kernels, compute_streams):
    cid = op['connection_id']
    cann = [named_api(row, strings) for row in rows(db, 'SELECT * FROM CANN_API WHERE connectionId=?', (cid,))]
    references = rows(db, 'SELECT id FROM CONNECTION_IDS WHERE connectionId=?', (cid,)) if table_exists(db, 'CONNECTION_IDS') else []
    ids = [integer(row['id']) for row in references]
    host = []
    if ids:
        placeholders = ','.join('?' for _ in set(ids))
        host = [named_api(row, strings) for row in rows(db,
            'SELECT * FROM PYTORCH_API WHERE connectionId IN (' + placeholders + ')', tuple(set(ids)))]
    host.sort(key=lambda row: (row['start_ns'], row['end_ns']))
    previous_send = [k for k in kernels if k.get('send_anchor') and k['end_ns'] <= op['start_ns']]
    next_recv = [k for k in kernels if k.get('recv_anchor') and k['start_ns'] >= op['end_ns']]
    send = max(previous_send, key=lambda k: k['end_ns']) if previous_send else None
    recv = min(next_recv, key=lambda k: k['start_ns']) if next_recv else None
    near_tasks = task_rows(db, strings, op['start_ns']-10_000_000, op['end_ns']+1_000_000)
    waits = [dict(row, on_derived_compute_stream=row['stream'] in compute_streams)
             for row in near_tasks if row['task_type'] in ('EVENT_WAIT', 'EVENT_RECORD')]
    for row in waits:
        linked = rows(db, 'SELECT * FROM CANN_API WHERE connectionId=?', (row['connection_id'],))
        row['cann_api_candidates'] = [named_api(item, strings) for item in linked]
    enclosing, before_host, qk_before = [], [], []
    if len(host) == 1:
        submit = host[0]
        nearby = [named_api(row, strings) for row in rows(db,
            'SELECT * FROM PYTORCH_API WHERE CAST(startNs AS INTEGER)>=? AND CAST(startNs AS INTEGER)<=?',
            (submit['start_ns']-2_000_000_000, submit['start_ns']))]
        enclosing = [row for row in nearby if row['global_tid'] == submit['global_tid']
                     and row['start_ns'] <= submit['start_ns'] and row['end_ns'] >= submit['end_ns']]
        enclosing.sort(key=lambda row: row['duration_ns'])
        probes = [row for row in enclosing if row['name'] == 'lingbot.denoise_step']
        lower = probes[0]['start_ns'] if len(probes) == 1 else submit['start_ns']-150_000_000
        before_host = [row for row in nearby if row['global_tid'] == submit['global_tid']
                       and lower <= row['start_ns'] and row['end_ns'] <= submit['start_ns']
                       and row['duration_ns'] >= 100_000]
        qk_before = [row for row in before_host if 'qk' in row['name'].lower() or 'rmsnorm' in row['name'].lower()]
    linked_comm_tasks = []
    if op.get('op_id') is not None and table_exists(db, 'COMMUNICATION_TASK_INFO'):
        query = ('SELECT t.*,ct.opId AS linkedOpId FROM TASK t JOIN COMMUNICATION_TASK_INFO ct '
                 'ON t.globalTaskId=ct.globalTaskId WHERE ct.opId=? AND t.startNs<? AND t.endNs>?')
        for row in rows(db, query, (op['op_id'], op['end_ns']+1_000_000, op['start_ns']-1_000_000)):
            item = intervals(row)
            linked_comm_tasks.append(dict(start_ns=item['start_ns'], end_ns=item['end_ns'],
                stream=item.get('streamId'), task=item.get('taskId'), global_task_id=item.get('globalTaskId'),
                connection_id=item.get('connectionId'), task_type=decode(strings, item.get('taskType'))))
    preceding = [k for k in kernels if k['stream'] in compute_streams
                 and op['start_ns']-10_000_000 <= k['start_ns'] < op['start_ns']][-16:]
    result = dict(rank=op['rank'], collective=op,
        cann_api_link=dict(status='unique' if len(cann) == 1 else ('missing' if not cann else 'ambiguous'), candidates=cann),
        pytorch_api_link=dict(status='unique' if len(host) == 1 else ('missing' if not host else 'ambiguous'),
            connection_ref_ids=ids, duplicate_ref_ids=[key for key, count in Counter(ids).items() if count>1], candidates=host),
        enclosing_host_ranges=enclosing, long_host_ranges_before_submit=before_host,
        qk_or_norm_host_ranges_before_submit=qk_before,
        temporal_send_before=None if send is None else kernel_context(send),
        temporal_recv_after=None if recv is None else kernel_context(recv),
        send_task_link=anchor_task(db, strings, send), recv_task_link=anchor_task(db, strings, recv),
        derived_compute_streams=sorted(compute_streams), nearby_event_tasks=waits,
        communication_tasks_by_explicit_op_id=linked_comm_tasks,
        preceding_compute_stream_kernels=[kernel_context(k) for k in preceding],
        event_handle_identity_proven=False, cpu_lateness_cause_proven=False,
        task_wait_durations_added_to_latency=False, temporal_neighbors_are_proven_dependencies=False)
    if len(cann) == len(host) == 1:
        result['exported_host_to_device_offsets_ns'] = dict(
            submit_to_cann_start=cann[0]['start_ns']-host[0]['start_ns'],
            cann_start_to_communication_start=op['start_ns']-cann[0]['start_ns'])
    return result


def sync_context(db, strings, kernels, lo, hi, top):
    names = [key for key, name in strings.items() if 'SynchronizeStream' in name]
    syncs = []
    for name in names:
        syncs.extend(named_api(row, strings) for row in rows(db,
            'SELECT * FROM CANN_API WHERE name=? AND startNs>=? AND endNs<=?', (name, lo, hi)))
    syncs.sort(key=lambda row: -row['duration_ns'])
    selected = []
    for sync in syncs[:top]:
        parents = [named_api(row, strings) for row in rows(db,
            'SELECT * FROM PYTORCH_API WHERE globalTid=? AND CAST(startNs AS INTEGER)<=? AND CAST(endNs AS INTEGER)>=?',
            (sync['global_tid'], sync['start_ns'], sync['end_ns']))]
        parents.sort(key=lambda row: row['duration_ns'])
        selected.append(dict(host_sync=sync, enclosing_host_ranges=parents,
            temporally_overlapping_kernel_partition=partition(kernels, sync['start_ns'], sync['end_ns']),
            synchronized_stream_proven=False, causal_blocking_task_proven=False))
    return dict(fully_contained_sync_count=len(syncs),
                host_call_duration_sum_ns=sum(row['duration_ns'] for row in syncs),
                duration_sum_is_not_additional_chunk_cost=True, longest_calls=selected)


def analyze(trace_root, first=11, last=14, top=12, sync_top=3,
            send_name='lingbot_recovery_send', recv_name='lingbot_recovery_recv',
            boundary_mode='marked-rgb', expected_chunks=None, expected_layer_calls=None,
            expected_alltoall_calls=None):
    root = Path(trace_root).resolve(strict=True)
    exports, issues = find_exports(root)
    result = dict(schema='lingbot_eight_rank_dependencies_v1', trace_root=str(root),
        requested_chunk_window=[first,last], expected_ranks=list(EXPECTED_RANKS),
        export_issues=issues, available_ranks=sorted(exports),
        complete_eight_rank_evidence=len(exports)==8, clock_alignment_proven=False,
        complete_eight_rank_evidence_definition='Unique DB and valid kernel CSV for all8 ranks; no kernel crosses a selected edge; requested send/recv and optional alltoall counts pass. Reconstructed boundaries and clock calibration remain separately qualified.',
        event_handle_identity_proven=False, cpu_lateness_cause_proven=False,
        limitations=LIMITATIONS, analyzer_source=pin(__file__))
    if 0 not in exports:
        result['window_error'] = 'Cannot derive complete chunk windows without a unique rank0 export'
        return result
    with open_db(exports[0][0]) as db:
        try:
            strings=read_strings(db)
            if boundary_mode == 'marked-rgb':
                windows, marker_evidence = rgb_windows(db, strings, first, last)
            else:
                need(boundary_mode == 'rgb-uint8-copy','Unknown boundary mode')
                windows, marker_evidence = uint8_copy_windows(db,strings,first,last,expected_chunks)
        except (ValueError, sqlite3.Error) as error:
            result['window_error'] = str(error)
            result['complete_eight_rank_evidence'] = False
            return result
    lo, hi = windows[0]['start_ns'], windows[-1]['end_ns']
    result.update(windows=windows, marker_evidence=marker_evidence,
                  global_window=dict(start_ns=lo,end_ns=hi), per_rank={})
    kernels_by_rank, comms_by_rank, streams_by_rank = {}, {}, {}
    for rank, (db_path, csv_path) in exports.items():
        kernels, csv_issues = read_kernels(csv_path, lo, hi)
        for row in kernels:
            row['send_anchor'], row['recv_anchor'] = row['name']==send_name, row['name']==recv_name
        streams = {row['stream'] for row in kernels if (row['send_anchor'] or row['recv_anchor']) and row['stream'] is not None}
        kernels_by_rank[rank], streams_by_rank[rank] = kernels, streams
        with open_db(db_path) as db:
            strings = read_strings(db)
            comms = read_comms(db, strings, lo, hi)
            comms_by_rank[rank] = comms
            result['per_rank'][rank] = dict(db=str(db_path), kernel_csv=str(csv_path),
                kernel_csv_issues=csv_issues, communication_rows=len(comms),
                kernel_rows_intersecting_window=len(kernels), derived_compute_streams=sorted(streams),
                compute_stream_derivation=dict(send_name=send_name,recv_name=recv_name,
                    send_anchors=sum(row['send_anchor'] for row in kernels), recv_anchors=sum(row['recv_anchor'] for row in kernels)),
                per_chunk=[dict(window, partition=partition(kernels,window['start_ns'],window['end_ns'])) for window in windows],
                complete_window_partition=partition(kernels,lo,hi),
                host_sync_diagnostics=sync_context(db, strings, kernels, lo, hi, sync_top))
        coverage=[]
        for window in windows:
            selected=[row for row in kernels if window['start_ns']<=row['start_ns'] and row['end_ns']<=window['end_ns']]
            send_count,recv_count=sum(row['send_anchor'] for row in selected),sum(row['recv_anchor'] for row in selected)
            crossing=[kernel_context(row) for row in kernels if
                      any(row['start_ns'] < edge < row['end_ns'] for edge in (window['start_ns'],window['end_ns']))]
            coverage.append(dict(chunk=window['chunk'],fully_contained_send_count=send_count,
                fully_contained_recv_count=recv_count,expected_each=expected_layer_calls,
                crossing_kernel_count=len(crossing),crossing_kernels=crossing,
                count_gate_passed=None if expected_layer_calls is None else send_count==recv_count==expected_layer_calls,
                explicit_layer_probe_identity_proven=False))
        result['per_rank'][rank]['send_recv_coverage']=coverage
    matching = match_collectives(comms_by_rank, windows)
    matched = matching.pop('matched')
    result['collective_matching'] = dict(matching, matched_count=len(matched),
        key_definition=['decoded_group_name','full_decoded_operation_name','normalized_count'],
        missing_or_duplicate_rows_are_not_guessed=True, matched_collectives=matched)
    result['per_chunk_arrival_summary'] = []
    for window in windows:
        chunk_rows = [r for r in matched if r['chunk']==window['chunk']]
        values = [r['exported_start_spread_ns'] for r in chunk_rows]
        result['per_chunk_arrival_summary'].append(dict(chunk=window['chunk'],
            matched_collectives_fully_in_chunk=len(values),
            matched_alltoall_fully_in_chunk=sum('alltoall' in r['key']['operation'].lower()
                or 'all_to_all' in r['key']['operation'].lower() for r in chunk_rows),
            mean_exported_start_spread_ns=statistics.mean(values) if values else None,
            max_exported_start_spread_ns=max(values) if values else None,
            last_start_rank_histogram=dict(Counter(rank for r in chunk_rows for rank in r['last_start_ranks']))))
    result['top_exported_start_spreads'] = []
    for target in matched[:top]:
        details = []
        for rank in EXPECTED_RANKS:
            with open_db(exports[rank][0]) as db:
                details.append(collective_detail(db,read_strings(db),target['per_rank'][rank],
                    kernels_by_rank[rank],streams_by_rank[rank]))
        result['top_exported_start_spreads'].append(dict(target=target,per_rank_dependencies=details))
    result['expected_alltoall_calls_per_chunk']=expected_alltoall_calls
    result['alltoall_count_gate_passed']=None if expected_alltoall_calls is None else all(
        row['matched_alltoall_fully_in_chunk']==expected_alltoall_calls for row in result['per_chunk_arrival_summary'])
    result['complete_eight_rank_evidence'] = len(exports)==8 and all(not r['kernel_csv_issues']
        and all(c['count_gate_passed'] is not False and c['crossing_kernel_count']==0
                for c in r['send_recv_coverage']) for r in result['per_rank'].values())
    result['complete_eight_rank_evidence'] &= result['alltoall_count_gate_passed'] is not False
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace-root',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--window',nargs=2,type=int,default=(11,14),metavar=('FIRST','LAST'))
    parser.add_argument('--top',type=int,default=12)
    parser.add_argument('--sync-top',type=int,default=3)
    parser.add_argument('--send-name',default='lingbot_recovery_send')
    parser.add_argument('--recv-name',default='lingbot_recovery_recv')
    parser.add_argument('--boundary-mode',choices=('marked-rgb','rgb-uint8-copy'),default='marked-rgb')
    parser.add_argument('--expected-chunks',type=int,
                        help='Required explicit full request count for opt-in unmarked RGB-copy reconstruction')
    parser.add_argument('--expected-layer-calls',type=int,
                        help='Optional per-rank/per-window complete send and recv count; e.g.160 for40layers×4probes, without proving labels')
    parser.add_argument('--expected-alltoall-calls',type=int,
                        help='Optional exact eight-rank matched All-to-All count per full chunk; current SP8 baseline requires320')
    args=parser.parse_args(argv)
    need(not args.out.exists(),'Refuse to overwrite an earlier dependency analysis')
    need(args.top>=0 and args.sync_top>=0 and (args.expected_layer_calls is None or args.expected_layer_calls>0)
         and (args.expected_alltoall_calls is None or args.expected_alltoall_calls>0),
         'Invalid diagnostic or coverage limit')
    result=analyze(args.trace_root,*args.window,args.top,args.sync_top,args.send_name,args.recv_name,
                   args.boundary_mode,args.expected_chunks,args.expected_layer_calls,args.expected_alltoall_calls)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open('x') as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(output=str(args.out.resolve()),
        complete_eight_rank_evidence=result['complete_eight_rank_evidence'],
        matched_collectives=result.get('collective_matching',{}).get('matched_count',0),
        window_error=result.get('window_error'))))
    return 0 if result['complete_eight_rank_evidence'] else 2


if __name__=='__main__':
    raise SystemExit(main())
