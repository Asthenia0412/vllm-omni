"""CPU-only matrix/owner audit. This never admits a candidate for model or timing."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

HELPER = Path(__file__).resolve().parents[1] / 'vllm_omni/diffusion/layers/lingbot_real_fixture_capture.py'
spec = importlib.util.spec_from_file_location('real_fixture_contract_cpu', HELPER)
contract = importlib.util.module_from_spec(spec)
spec.loader.exec_module(contract)


def need(value, reason):
    if not value:
        raise ValueError(reason)


def local_file(directory, relative):
    path = (directory / relative).resolve(strict=True)
    need(path.is_relative_to(directory.resolve()), 'Fixture path escapes rank directory')
    return path


def check(root, *, hash_payloads=False):
    root = Path(root)
    expected = {(kind, length, probe, layer) for kind in ('qk', 'cross')
                for length in contract.LENGTHS for probe in range(4) for layer in range(40)}
    total_owners, total_bytes = set(), 0
    owner_hashes = {}
    manifests, helper_pins, rank_locations = set(), set(), []
    for rank in range(8):
        directory = root / f'rank{rank}'
        coverage = json.loads((directory / 'coverage.json').read_text())
        need(coverage['schema'] == 'lingbot_real_qk_cross_capture_v1'
             and coverage['rank'] == rank and coverage['complete']
             and not coverage['failed'] and not coverage['timing_eligible'], 'Incomplete/invalid rank coverage')
        need(coverage['payload_count'] == 48 and coverage['descriptor_count'] == 1920,
             'Layer0 payload/all40 descriptor cardinality differs')
        manifests.add(coverage['build_manifest']['sha256'])
        helper_pins.add(tuple((p['path'], p['sha256']) for p in coverage['source_pins']))
        locations = {(v['kv_len'],v['probe']):(v['request_id'],v['chunk']) for v in coverage['locations']}
        need(len(locations) == 24, 'Missing actual history/probe locations')
        rank_locations.append(locations)
        keys = set()
        for name in coverage['cases']:
            case = json.loads(local_file(directory,name).read_text())
            need(case.get('schema') == 'lingbot_real_fixture_case_v1', 'Invalid case schema')
            key = (case['kind'],case['kv_len'],case['probe'],case['layer'])
            need(key in expected and key not in keys and case['rank'] == rank, 'Duplicate/invalid case identity')
            keys.add(key)
            need((case['request_id'],case['chunk']) == locations[key[1:3]], 'Case differs from actual selected occurrence')
            payload = case['layer'] == 0
            need(case['payload_present'] is payload and case['source_descriptors_unchanged'] is True,
                 'Payload/descriptor coverage is mislabeled')
            need(case['full_sources_unchanged'] is (True if payload else None), 'Source byte proof is mislabeled')
            receipt = case['dispatch_receipt']
            expected_op = 'qk_send' if key[0] == 'qk' else 'cross'
            wanted_inputs = ({'qkv','query_weight','key_weight','cosine','sine','expanded_cosine','expanded_sine'}
                             if key[0] == 'qk' else {'query','key','value'})
            wanted_outputs = {'native_send'} if key[0] == 'qk' else {'native_output'}
            need(set(case['inputs']['descriptors']) == wanted_inputs
                 and set(case['outputs']['descriptors']) == wanted_outputs, 'Missing actual operand/output')
            need(receipt['provider_operation'] == expected_op and receipt['dispatch_count'] == 1
                 and receipt['operands_identical'] and receipt['output_identical']
                 and receipt['build_manifest_sha256'] == coverage['build_manifest']['sha256'],
                 'Missing actual native backend receipt')
            for section in ('inputs','outputs'):
                snap = case[section]
                need(set(snap['descriptors']) == set(snap['view_owner_ids']), 'Missing view owner association')
                owner_ids, allocations = {}, {}
                for tensor,desc in snap['descriptors'].items():
                    contract.view_extent(desc)
                    oid=snap['view_owner_ids'][tensor]
                    owner=snap['owners'][oid]
                    allocation=(desc['device'],desc['storage_data_ptr'],desc['storage_bytes'])
                    identity=(*allocation,desc['dtype'])
                    need(oid not in owner_ids or owner_ids[oid]==identity,'Owner ID has inconsistent allocation')
                    need(allocation not in allocations or allocations[allocation]==oid,
                         'One captured allocation is split across multiple owner IDs')
                    owner_ids[oid]=identity;allocations[allocation]=oid
                    need(owner['storage_bytes'] == desc['storage_bytes']
                         and owner['base_sizes'] == desc['base_sizes']
                         and owner['base_strides'] == desc['base_strides']
                         and owner['npu_format'] == desc['npu_format'], 'Owner descriptor mismatch')
                need(set(owner_ids)==set(snap['owners']),'Unreferenced owner in fixture')
                for owner in snap['owners'].values():
                    if not payload:
                        need(owner['raw_sha256'] is None and owner['relative_path'] is None,
                             'Metadata-only case unexpectedly claims payload')
                        continue
                    digest=owner['raw_sha256']
                    need(isinstance(digest,str) and len(digest)==64, 'Missing owner content hash')
                    path=local_file(directory,owner['relative_path'])
                    need(path.stat().st_size==owner['storage_bytes'], 'Truncated owner payload')
                    if path not in total_owners:
                        if hash_payloads:
                            state=hashlib.sha256()
                            with path.open('rb') as handle:
                                for block in iter(lambda:handle.read(1024*1024),b''):state.update(block)
                            owner_hashes[path] = state.hexdigest()
                        total_owners.add(path);total_bytes+=path.stat().st_size
                    if hash_payloads:
                        need(owner_hashes[path]==digest, 'Owner payload hash mismatch')
        need(keys == expected, 'Missing real case descriptors')
    need(len(manifests)==1 and len(helper_pins)==1, 'Ranks use different source/build pins')
    need(all(loc == rank_locations[0] for loc in rank_locations), 'Rank actual history/probe locations differ')
    return dict(schema='lingbot_real_fixture_matrix_audit_v1', complete_input_matrix=True,
                actual192_global_proven=True, qk_real_layer0_payload_cases=192,
                cross_real_layer0_payload_cases=192, all40_descriptor_cases=15360,
                all40_payloads=False, owners=len(total_owners), owner_bytes=total_bytes,
                payload_sha256_verified=hash_payloads, destination_canary_verified=False,
                kernel_correctness_passed=False, model_admitted=False, performance_validated=False,
                build_manifest_sha256=next(iter(manifests)))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    parser.add_argument('--hash-payloads',action='store_true')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    result=check(args.root,hash_payloads=args.hash_payloads)
    text=json.dumps(result,indent=2)+'\n'
    if args.output:args.output.write_text(text)
    print(text,end='')
