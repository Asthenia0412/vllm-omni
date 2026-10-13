"""CPU-only coverage/owner/dispatch tests; does not import torch or use devices."""
import importlib.util
from pathlib import Path
import tempfile
import types
import unittest

PATH = Path(__file__).resolve().parents[1] / 'vllm_omni/diffusion/layers/lingbot_real_fixture_capture.py'
spec = importlib.util.spec_from_file_location('real_fixture_capture_cpu', PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def descriptor():
    return dict(npu_format=2, origin_format=2, base_sizes=[4,8], base_strides=[8,1],
                physical_sizes=[4,8], shape=[2,3], stride=[8,1], storage_offset=9,
                element_bytes=2, storage_bytes=64, storage_data_ptr=4096, data_ptr=4114)


class OwnerContract(unittest.TestCase):
    def test_noncontiguous_view_preserves_full_owner(self):
        self.assertEqual(module.view_extent(descriptor()), 32)

    def test_rejects_extent_pointer_format_and_hidden_owner_mismatches(self):
        for update in (dict(storage_bytes=66), dict(storage_offset=31, data_ptr=4158),
                       dict(data_ptr=4096), dict(npu_format=29, origin_format=29),
                       dict(physical_sizes=[4,16]), dict(base_strides=[9,1]),
                       dict(stride=[-8,1]), dict(origin_format=0)):
            with self.subTest(update=update), self.assertRaises(RuntimeError):
                desc=descriptor();desc.update(update);module.view_extent(desc)


class CoverageContract(unittest.TestCase):
    def test_length_is_authoritative_and_repeated_chunk_does_not_add_case(self):
        cov=module.Coverage(3)
        self.assertFalse(cov.choose('request2',0,0,1))
        self.assertTrue(cov.choose('request2',13,0,9360))
        self.assertFalse(cov.choose('request2',14,0,9360))
        self.assertFalse(cov.choose('request3',13,0,9360))
        self.assertEqual(cov.report()['locations'][0]['chunk'],13)

    def test_complete_is_24_real_payload_cases_per_role_and_all40_descriptors(self):
        cov=module.Coverage(7)
        for length in module.LENGTHS:
            for probe in range(4):
                cov.choose('request2',length//4680+11,probe,length)
                for layer in range(40):
                    for kind in ('qk','cross'):
                        cov.record(kind,length,probe,layer,payload=layer==0)
        report=cov.report()
        self.assertTrue(report['complete'])
        self.assertEqual(report['payload_count'],48)
        self.assertEqual(report['descriptor_count'],1920)
        self.assertFalse(report['actual192_global_proven'])
        self.assertFalse(report['all40_payloads'])
        with self.assertRaisesRegex(RuntimeError,'Duplicate'):
            cov.record('qk',4680,0,0,payload=True)

    def test_missing_history_or_payload_cannot_pass(self):
        cov=module.Coverage(0)
        cov.record('qk',4680,0,0,payload=False)
        report=cov.report()
        self.assertFalse(report['complete'])
        self.assertIn(['qk',4680,0,0],report['payload_missing'])
        self.assertIn(['cross',28080,3,39],report['descriptor_missing'])
        with self.assertRaises(RuntimeError):module.Coverage(8)
        with self.assertRaises(RuntimeError):module.Coverage(0,payload_layers=(40,))


class DispatchContract(unittest.TestCase):
    def setUp(self):
        self.rec=object.__new__(module.Recorder)
        self.rec.expected_dispatch=None
        self.rec.provider=types.SimpleNamespace(policy=('manifest','abc'))

    def test_receipt_requires_exact_one_call_and_actual_inputs_output(self):
        q,k,v,result=object(),object(),object(),object()
        self.rec.expect_dispatch('cross',(q,k,v))
        calls=[]
        def original(name,*args):calls.append((name,args));return result
        self.assertIs(self.rec.observed_call(original,'cross',q,k,v,0.1),result)
        receipt=self.rec.finish_dispatch(result)
        self.assertEqual(len(calls),1)
        self.assertEqual(receipt['provider_operation'],'cross')
        self.assertIsNone(self.rec.expected_dispatch)

    def test_fallback_wrong_operand_duplicate_and_changed_output_rejected(self):
        q,r=object(),object()
        self.rec.expect_dispatch('qk_send',(q,))
        with self.assertRaisesRegex(RuntimeError,'no fallback'):self.rec.finish_dispatch(r)
        with self.assertRaisesRegex(RuntimeError,'operands'):
            self.rec.observed_call(lambda *args:r,'qk_send',object())
        self.rec.observed_call(lambda *args:r,'qk_send',q)
        with self.assertRaisesRegex(RuntimeError,'Duplicate'):
            self.rec.observed_call(lambda *args:r,'qk_send',q)
        with self.assertRaisesRegex(RuntimeError,'captured output'):self.rec.finish_dispatch(object())

    def test_bind_clears_layer_context_and_tracks_real_request_identity(self):
        self.rec.active=True;self.rec.pending={};self.rec.layer_context={39:'old'};self.rec.sessions=[]
        state=types.SimpleNamespace(request_id='fresh',chunk_index=11,step_in_chunk=3)
        self.rec.bind(state)
        self.assertEqual(self.rec.layer_context,{})
        self.assertEqual(self.rec.context,('fresh',11,3))
        self.assertEqual(self.rec.sessions,['fresh'])
        self.rec.pending[0]='not consumed'
        with self.assertRaisesRegex(RuntimeError,'incomplete'):self.rec.bind(state)

    def test_metadata_only_never_claims_source_bytes_or_destination_canary(self):
        self.rec.coverage=module.Coverage(0)
        self.rec.rank=0;self.rec.saved_cases=[]
        with tempfile.TemporaryDirectory() as tmp:
            self.rec.directory=Path(tmp)
            self.rec.finish('qk',dict(layer=1,kv_len=4680,probe=0),{}, {},False)
            import json
            record=json.loads(next(Path(tmp).glob('*.json')).read_text())
            self.assertIsNone(record['full_sources_unchanged'])
            self.assertFalse(record['destination_canary_verified'])

# Exercise the global checker with synthetic CPU metadata only. These fixtures
# test the reader contract and are never recorded as real activation evidence.
class MatrixAuditContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec=importlib.util.spec_from_file_location('fixture_matrix_check_cpu',Path(__file__).with_name('check_real_fixtures.py'))
        cls.checker=importlib.util.module_from_spec(spec);spec.loader.exec_module(cls.checker)

    def test_complete_eight_rank_matrix_and_reject_missing_payload(self):
        import hashlib
        import json
        from unittest.mock import patch
        blob=b'\x00\x00';digest=hashlib.sha256(blob).hexdigest()
        names=[f'{kind}_{length}_{probe}_{layer}.json' for kind in ('qk','cross')
               for length in module.LENGTHS for probe in range(4) for layer in range(40)]
        corrupted=[False]
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for rank in range(8):
                (root/f'rank{rank}').mkdir();(root/f'rank{rank}/owner.u8').write_bytes(blob)
            def read(path,*args,**kwargs):
                rank=int(path.parent.name[4:])
                if path.name=='coverage.json':
                    return json.dumps(dict(schema='lingbot_real_qk_cross_capture_v1',rank=rank,
                        complete=True,failed=False,timing_eligible=False,payload_count=48,descriptor_count=1920,
                        build_manifest=dict(sha256='build'),source_pins=[dict(path='helper',sha256='source')],
                        locations=[dict(kv_len=l,probe=p,request_id='r2',chunk=l//4680)
                                   for l in module.LENGTHS for p in range(4)],cases=names))
                kind,length,probe,layer=path.stem.split('_');length,probe,layer=map(int,(length,probe,layer))
                payload=layer==0
                desc=dict(npu_format=2,origin_format=2,base_sizes=[1],base_strides=[1],physical_sizes=[1],
                          shape=[1],stride=[1],storage_offset=0,element_bytes=2,storage_bytes=2,
                          storage_data_ptr=4096,data_ptr=4096)
                owner=dict(storage_bytes=2,base_sizes=[1],base_strides=[1],npu_format=2,
                           raw_sha256=digest if payload else None,relative_path='owner.u8' if payload else None)
                snap=dict(descriptors=dict(x=desc),owners=dict(owner000=owner),view_owner_ids=dict(x='owner000'))
                return json.dumps(dict(kind=kind,kv_len=length,probe=probe,layer=layer,rank=rank,request_id='r2',
                    chunk=length//4680,payload_present=payload and not corrupted[0],source_descriptors_unchanged=True,
                    full_sources_unchanged=True if payload else None,inputs=snap,outputs=snap,
                    dispatch_receipt=dict(provider_operation='qk_send' if kind=='qk' else 'cross',dispatch_count=1,
                                          operands_identical=True,output_identical=True,build_manifest_sha256='build')))
            with patch.object(Path,'read_text',read),patch.object(self.checker,'local_file',lambda d,n:d/n):
                result=self.checker.check(root,hash_payloads=True)
                self.assertEqual(result['qk_real_layer0_payload_cases'],192)
                self.assertEqual(result['cross_real_layer0_payload_cases'],192)
                self.assertTrue(result['payload_sha256_verified'])
                self.assertFalse(result['model_admitted'])
                corrupted[0]=True
                with self.assertRaisesRegex(ValueError,'mislabeled'):self.checker.check(root)


if __name__=='__main__':unittest.main()
