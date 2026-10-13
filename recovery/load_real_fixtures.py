"""Load real layer0 QK/cross cases without collapsing full owners into views.

Importing this module uses only the standard library. Device materialization is
an explicit caller action; it does not admit a benchmark or a model candidate.
"""
from dataclasses import dataclass
import hashlib
import importlib.util
import json
from pathlib import Path

HELPER = Path(__file__).resolve().parents[1] / 'vllm_omni/diffusion/layers/lingbot_real_fixture_capture.py'
spec = importlib.util.spec_from_file_location('real_fixture_loader_contract', HELPER)
contract = importlib.util.module_from_spec(spec)
spec.loader.exec_module(contract)


def need(value, message):
    if not value:
        raise ValueError(message)


def local_file(directory, relative):
    path = (directory / relative).resolve(strict=True)
    need(path.is_relative_to(directory.resolve()), 'Owner path escapes rank directory')
    return path


def case_paths(root, kind, rank, *, require_complete=True):
    """Return24 actual layer0 files per role/rank; never synthesize missing cases."""
    need(kind in ('qk','cross') and rank in range(8), 'Eight-rank QK/cross role required')
    directory = Path(root) / f'rank{rank}'
    coverage = json.loads((directory / 'coverage.json').read_text())
    need(coverage['rank'] == rank and not coverage['failed'] and not coverage['timing_eligible'],
         'Invalid or failed correctness capture')
    paths = {}
    for relative in coverage['cases']:
        if not relative.startswith(kind+'_'):
            continue
        path = local_file(directory,relative)
        case = json.loads(path.read_text())
        if case['layer'] != 0:
            continue
        key = (case['kv_len'],case['probe'])
        need(case['kind'] == kind and case['rank'] == rank and case['payload_present']
             and key not in paths, 'Invalid/duplicate real payload case')
        paths[key] = path
    expected = {(length,probe) for length in contract.LENGTHS for probe in range(4)}
    need(set(paths).issubset(expected), 'Unexpected real history/probe case')
    if require_complete:
        need(coverage['complete'] and set(paths) == expected, 'Missing actual24 cases for rank/role')
    return [paths[key] for key in sorted(paths)]


def validate_case_metadata(case):
    need(case['schema'] == 'lingbot_real_fixture_case_v1' and case['kind'] in ('qk','cross')
         and case['rank'] in range(8) and case['layer'] == 0 and case['probe'] in range(4)
         and case['kv_len'] in contract.LENGTHS, 'Not a real layer0 fixture identity')
    need(case['payload_present'] and case['full_sources_unchanged'] is True
         and case['source_descriptors_unchanged'] is True and not case['timing_eligible'],
         'Missing full-source correctness evidence')
    receipt = case['dispatch_receipt']
    wanted = 'qk_send' if case['kind'] == 'qk' else 'cross'
    need(receipt['provider_operation'] == wanted and receipt['dispatch_count'] == 1
         and receipt['operands_identical'] and receipt['output_identical'], 'Actual backend receipt missing')
    expected_inputs = ({'qkv','query_weight','key_weight','cosine','sine','expanded_cosine','expanded_sine'}
                       if case['kind'] == 'qk' else {'query','key','value'})
    expected_outputs = {'native_send'} if case['kind'] == 'qk' else {'native_output'}
    need(set(case['inputs']['descriptors']) == expected_inputs
         and set(case['outputs']['descriptors']) == expected_outputs, 'Missing actual operand/output')


@dataclass
class LoadedCase:
    path: Path
    metadata: dict
    owner_bytes: dict

    def materialize(self, torch, device, describe=None, allocate_owner=None):
        """Explicit allocation, preserving full base extents and snapshot aliases.

        A non-CPU destination requires the original eleven-field descriptor
        callable and allocate_owner(base_sizes, dtype, device, npu_format).
        Use an explicit-format allocator, not a format-label mutation. Its
        physical contract must match before copying/invoking a candidate.
        Returned owners MUST remain live through all consumer completion events.
        Source and reference-output sections intentionally retain independent
        snapshots; identical content hashes are not used as owner identity.
        """
        cpu = str(device) == 'cpu'
        need(cpu or (describe is not None and allocate_owner is not None),
             'Device replay requires explicit-format allocation and descriptor validation')
        requested_device=torch.device(device)
        need(cpu or requested_device.index is not None,'Explicit device index required for fixture replay')
        dtype_map = {'torch.bfloat16':torch.bfloat16, 'torch.float32':torch.float32,
                     'torch.float16':torch.float16, 'torch.int64':torch.int64,
                     'torch.int32':torch.int32, 'torch.uint8':torch.uint8}
        values, owners, host_owners = {}, {}, {}
        device_allocations={}
        for section in ('inputs','outputs'):
            snapshot = self.metadata[section]
            values[section], owners[section], host_owners[section] = {}, {}, {}
            for name,desc in snapshot['descriptors'].items():
                oid = snapshot['view_owner_ids'][name]
                if oid not in owners[section]:
                    need(desc['dtype'] in dtype_map, 'Unsupported fixture dtype')
                    raw = self.owner_bytes[(section,oid)]
                    base = torch.frombuffer(bytearray(raw),dtype=dtype_map[desc['dtype']]).clone()
                    need(base.element_size() == desc['element_bytes'], 'Dtype element width differs')
                    base = base.reshape(desc['base_sizes'])
                    host_owners[section][oid] = base
                    owners[section][oid] = (base if cpu else allocate_owner(
                        desc['base_sizes'],dtype_map[desc['dtype']],device,desc['npu_format']))
                base = owners[section][oid]
                need(str(base.dtype) == desc['dtype'], 'Aliased views use different dtypes')
                view = base.as_strided(desc['shape'],desc['stride'],desc['storage_offset'])
                if not cpu:
                    need(base.device==requested_device,'Allocator returned the wrong device')
                    actual = dict(zip(contract.FIELDS,describe(view)))
                    need(len(actual) == len(contract.FIELDS), 'Incomplete device descriptor')
                    for field in ('base_sizes','base_strides','physical_sizes','shape','stride'):
                        actual[field] = list(actual[field])
                    for field in contract.FIELDS[:9]:
                        need(actual[field] == desc[field], f'Device fixture {name}: {field} differs')
                    need(actual['storage_data_ptr'] == base.data_ptr()
                         and actual['data_ptr'] == base.data_ptr()+desc['storage_offset']*desc['element_bytes'],
                         'Device owner/view pointer mismatch')
                    allocation=(str(base.device),actual['storage_data_ptr'])
                    need(allocation not in device_allocations or device_allocations[allocation]==(section,oid),
                         'Allocator collapsed distinct snapshot owners')
                    device_allocations[allocation]=(section,oid)
                values[section][name] = view
        if not cpu:
            # Validate every owner/view first. Retain CPU owners as well because
            # the caller, allocator and framework determine copy completion.
            for section in ('inputs','outputs'):
                for oid,base in owners[section].items():
                    base.copy_(host_owners[section][oid])
        return dict(inputs=values['inputs'], outputs=values['outputs'], owners=owners,
                    host_owners=host_owners,
                    metadata=self.metadata, real_payload_layers=[0], all40_payloads=False)


def load_case(path):
    """Verify hashes and full-owner layout, without importing torch or allocating NPU."""
    path = Path(path).resolve(strict=True)
    case = json.loads(path.read_text())
    validate_case_metadata(case)
    payloads = {}
    for section in ('inputs','outputs'):
        snapshot = case[section]
        need(set(snapshot['descriptors']) == set(snapshot['view_owner_ids']), 'Missing owner mapping')
        owner_descriptors, allocation_owners = {}, {}
        for name,desc in snapshot['descriptors'].items():
            contract.view_extent(desc)
            oid = snapshot['view_owner_ids'][name]
            owner = snapshot['owners'][oid]
            fields = ('storage_bytes','base_sizes','base_strides','npu_format')
            need(all(owner[field] == desc[field] for field in fields), 'Owner extent/format differs from view')
            allocation = (desc['device'],desc['storage_data_ptr'],desc['storage_bytes'])
            identity = (*allocation,desc['dtype'])
            need(oid not in owner_descriptors or owner_descriptors[oid] == identity, 'Invalid aliased owner identity')
            need(allocation not in allocation_owners or allocation_owners[allocation] == oid,
                 'One captured allocation is split across multiple owner IDs')
            owner_descriptors[oid] = identity
            allocation_owners[allocation] = oid
        need(set(owner_descriptors) == set(snapshot['owners']), 'Unreferenced owner in fixture')
        for oid,owner in snapshot['owners'].items():
            raw = local_file(path.parent,owner['relative_path']).read_bytes()
            need(len(raw) == owner['storage_bytes'], 'Incomplete owner bytes')
            need(hashlib.sha256(raw).hexdigest() == owner['raw_sha256'], 'Owner SHA mismatch')
            payloads[(section,oid)] = raw
    return LoadedCase(path,case,payloads)
