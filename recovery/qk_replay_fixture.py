"""Standalone audit helpers; do not import from or install into model runtime."""
from pathlib import Path
import hashlib, json, os
FIELDS = ('npu_format', 'origin_format', 'base_sizes', 'base_strides', 'physical_sizes', 'shape', 'stride', 'storage_offset', 'storage_bytes', 'storage_data_ptr', 'data_ptr')
SHAPE = (8, 585, 3, 1, 5, 128)
STRIDE = (1123200, 1920, 640, 640, 128, 1)
_QUARANTINE = []

def require(condition, message):
    if not condition:
        raise RuntimeError(message)

def pin(path):
    p = Path(path).resolve()
    raw = p.read_bytes()
    return dict(path=str(p), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())

def verify(item):
    actual = pin(item['path'])
    require(all((actual[k] == item[k] for k in ('path', 'bytes', 'sha256'))), item['path'])
    return Path(item['path'])

class Runtime:

    def __init__(self, rank, fixture_manifest):
        import torch, torch_npu
        (self.torch, self.npu) = (torch, torch_npu)
        self.rank = rank
        require(os.environ['ASCEND_RT_VISIBLE_DEVICES'] == '8,9,10,11,12,13,14,15', "audit contract: os.environ['ASCEND_RT_VISIBLE_DEVICES'] == '8,9,10,11,12,13,14,15'")
        torch.npu.set_device(rank)
        torch.npu.set_compile_mode(jit_compile=False)
        torch.npu.config.allow_internal_format = True
        require(torch.npu.get_device_name() == 'Ascend910_9392', "audit contract: torch.npu.get_device_name() == 'Ascend910_9392'")
        from vllm_omni.diffusion.layers.lingbot_recovery import load
        self.provider = load()
        self.fixture_path = Path(fixture_manifest)
        self.fixture = json.loads(self.fixture_path.read_text())
        require(self.fixture['scope'] == dict(request_ordinal=2, chunk=11, probe=0, layer=0, ranks=8), "audit contract: self.fixture['scope'] == dict(request_ordinal=2, chunk=11, probe=0, layer=0, ranks=8)")
        require(self.fixture['actual192'] is False, "audit contract: self.fixture['actual192'] is False")
        self.case = self.fixture['cases'][rank]
        require(self.case['rank'] == rank, "audit contract: self.case['rank'] == rank")
        verify(self.case['record'])
        self.checks = []
        self.owners = {}
        self.source_views = {}
        for (name, desc) in self.case['descriptors'].items():
            oid = self.case['owner_ids'][name]
            require(desc['dtype'] == 'torch.bfloat16' and desc['npu_format'] == desc['origin_format'] == 2, "audit contract: desc['dtype'] == 'torch.bfloat16' and desc['npu_format'] == desc['origin_format'] == 2")
            require(desc['physical_sizes'] == desc['base_sizes'], "audit contract: desc['physical_sizes'] == desc['base_sizes']")
            if oid not in self.owners:
                item = self.case['source_owners'][oid]
                data = bytearray(verify(item).read_bytes())
                cpu = torch.frombuffer(data, dtype=torch.uint8).view(torch.bfloat16).reshape(desc['base_sizes'])
                owner = torch_npu.empty_with_format(tuple(desc['base_sizes']), dtype=torch.bfloat16, device=f'npu:{rank}', acl_format=2)
                owner.copy_(cpu)
                self.owners[oid] = owner
            view = self.owners[oid].as_strided(tuple(desc['shape']), tuple(desc['stride']), desc['storage_offset'])
            got = dict(zip(FIELDS, self.provider.describe(view)))
            for key in FIELDS[:9]:
                require((list(got[key]) if isinstance(got[key], (tuple, list)) else got[key]) == desc[key], (name, key, got[key], desc[key]))
            self.source_views[name] = view
        torch.npu.synchronize()
        self.before = {k: v.cpu().clone() for (k, v) in self.owners.items()}
        self.raw = self.source_views['qkv']
        self.qw = self.source_views['query_weight']
        self.kw = self.source_views['key_weight']
        self.cos = self.source_views['expanded_cosine']
        self.sin = self.source_views['expanded_sine']
        self.arguments = (self.raw, self.qw, self.kw, self.cos, self.sin)

    def equal(self, actual, expected, name):
        t = self.torch
        t.npu.synchronize()
        a = actual.cpu().contiguous()
        b = expected.cpu().contiguous()
        require(a.shape == b.shape and a.dtype == b.dtype and t.equal(a.view(t.uint8), b.view(t.uint8)), name)
        self.checks.append(name)

    def check_sources(self):
        for (oid, owner) in self.owners.items():
            self.equal(owner, self.before[oid], 'source_full_' + oid)

    def native(self, check_original_rms=False):
        t = self.torch
        n = self.npu
        (q, k, v) = self.raw.split(5120, -1)
        q = n.npu_rms_norm(q, self.qw, epsilon=1e-06)[0].view(1, 585, 40, 128)
        k = n.npu_rms_norm(k, self.kw, epsilon=1e-06)[0].view(1, 585, 40, 128)
        if check_original_rms:
            for (name, value) in (('query', q), ('key', k)):
                item = self.case['references'][name]
                data = bytearray(verify(item).read_bytes())
                ref = t.frombuffer(data, dtype=t.uint8).view(t.bfloat16).reshape(value.shape)
                self.equal(value, ref, 'retained_historical_' + name + '_RMS')
            self.equal(self.source_views['cosine'].repeat_interleave(2, dim=-1).view_as(self.cos), self.cos, 'expanded_cosine_snapshot')
            self.equal(self.source_views['sine'].repeat_interleave(2, dim=-1).view_as(self.sin), self.sin, 'expanded_sine_snapshot')
        q = n.npu_rotary_mul(q, self.cos, self.sin, rotary_mode='interleave')
        k = n.npu_rotary_mul(k, self.cos, self.sin, rotary_mode='interleave')
        original = t.stack((q, k, v.view(1, 585, 40, 128)), 2)
        packed = original.view(1, 585, 3, 8, 5, 128).permute(3, 1, 2, 0, 4, 5).contiguous()
        return (original, packed)

    def report(self):
        return dict(rank=self.rank, checks=self.checks, fixture_scope=self.fixture['scope'], fixture_manifest_sha256=pin(self.fixture_path)['sha256'], actual192=False, all40=False, all_probes=False, historical_binary_restored=False, model_admitted=False)
