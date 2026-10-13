"""CPU tests with explicitly synthetic tiny byte owners, never model evidence."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

spec=importlib.util.spec_from_file_location('fixture_loader_cpu_tests',Path(__file__).with_name('load_real_fixtures.py'))
loader=importlib.util.module_from_spec(spec);sys.modules[spec.name]=loader;spec.loader.exec_module(loader)


def fixture(directory):
    raw=bytes(range(12));(directory/'owner.u8').write_bytes(raw)
    def snapshot(names):
        return dict(descriptors={name:dict(npu_format=2,origin_format=2,base_sizes=[3,4],
            base_strides=[4,1],physical_sizes=[3,4],shape=[2,2],stride=[4,1],storage_offset=1,
            storage_bytes=12,storage_data_ptr=4096,data_ptr=4097,dtype='torch.uint8',element_bytes=1,device='cpu')
            for name in names},view_owner_ids={name:'owner000' for name in names},
            owners=dict(owner000=dict(storage_bytes=12,base_sizes=[3,4],base_strides=[4,1],npu_format=2,
                                      raw_sha256=hashlib.sha256(raw).hexdigest(),relative_path='owner.u8')))
    case=dict(schema='lingbot_real_fixture_case_v1',kind='cross',rank=0,layer=0,probe=0,kv_len=4680,
              payload_present=True,full_sources_unchanged=True,source_descriptors_unchanged=True,
              timing_eligible=False,dispatch_receipt=dict(provider_operation='cross',dispatch_count=1,
                   operands_identical=True,output_identical=True),inputs=snapshot(('query','key','value')),
              outputs=snapshot(('native_output',)))
    path=directory/'cross_kv4680_p0_l00.json';path.write_text(json.dumps(case));return path,case


class FullOwnerLoader(unittest.TestCase):
    def test_noncontiguous_alias_and_full_prefix_suffix_survive_cpu_materialization(self):
        import torch
        with tempfile.TemporaryDirectory() as tmp:
            path,_=fixture(Path(tmp));loaded=loader.load_case(path);replay=loaded.materialize(torch,'cpu')
            self.assertEqual(replay['inputs']['query'].tolist(),[[1,2],[5,6]])
            self.assertEqual(replay['owners']['inputs']['owner000'].flatten().tolist(),list(range(12)))
            self.assertEqual(replay['inputs']['query'].untyped_storage().data_ptr(),
                             replay['inputs']['key'].untyped_storage().data_ptr())
            self.assertNotEqual(replay['inputs']['query'].untyped_storage().data_ptr(),
                                replay['outputs']['native_output'].untyped_storage().data_ptr())
            self.assertFalse(replay['all40_payloads'])
            with self.assertRaisesRegex(ValueError,'explicit-format'):
                loaded.materialize(torch,'npu:0')  # Rejects before any device allocation/API.

    def test_rejects_truncated_or_changed_full_owner(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp);path,_=fixture(directory)
            (directory/'owner.u8').write_bytes(bytes(12))
            with self.assertRaisesRegex(ValueError,'SHA'):loader.load_case(path)
            (directory/'owner.u8').write_bytes(bytes(11))
            with self.assertRaisesRegex(ValueError,'Incomplete'):loader.load_case(path)

    def test_reused_payload_path_with_conflicting_hash_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path,case=fixture(Path(tmp))
            case['outputs']['owners']['owner000']['raw_sha256']='0'*64
            path.write_text(json.dumps(case))
            with self.assertRaisesRegex(ValueError,'SHA'):loader.load_case(path)

    def test_partial_index_cannot_be_promoted_to24(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);directory=root/'rank0';directory.mkdir();path,case=fixture(directory)
            (directory/'coverage.json').write_text(json.dumps(dict(rank=0,failed=False,timing_eligible=False,
                complete=False,cases=[path.name])))
            self.assertEqual(loader.case_paths(root,'cross',0,require_complete=False),[path])
            with self.assertRaisesRegex(ValueError,'Missing actual24'):loader.case_paths(root,'cross',0)

    def test_split_alias_cannot_create_two_independent_allocations(self):
        with tempfile.TemporaryDirectory() as tmp:
            path,case=fixture(Path(tmp))
            case['inputs']['owners']['owner001']=copy.deepcopy(case['inputs']['owners']['owner000'])
            case['inputs']['view_owner_ids']['key']='owner001'
            path.write_text(json.dumps(case))
            with self.assertRaisesRegex(ValueError,'split across'):loader.load_case(path)


if __name__=='__main__':unittest.main()
