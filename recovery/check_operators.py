"""Independent correctness run; never use its timings as performance data."""
import argparse
import hashlib
import json
import faulthandler
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    faulthandler.dump_traceback_later(45, repeat=True)
    import torch
    import torch_npu

    manifest = json.loads(args.manifest.read_text())
    for name in ("library", "kernels"):
        pin = manifest[name]
        assert hashlib.sha256(Path(pin["path"]).read_bytes()).hexdigest() == pin["sha256"]
    torch.npu.set_device(args.device)
    torch.npu.set_compile_mode(jit_compile=False)
    torch.npu.config.allow_internal_format = True
    torch.ops.load_library(manifest["library"]["path"])
    ops = torch.ops.lingbot_recovery_v1
    torch.manual_seed(1327)
    device = f"npu:{args.device}"
    results = []

    def equal(actual, expected, label):
        torch.npu.synchronize()
        a, b = actual.cpu().contiguous(), expected.cpu().contiguous()
        assert torch.equal(a.view(torch.int16), b.view(torch.int16)), label
        results.append(label)
        print("PASS", label, flush=True)

    with torch.inference_mode():
        # Source/arena/canaries cover strided offsets, width tails, 4D and 5D.
        for shape in ((1,384,2,120,26), (1,8,3,7,13), (1,4,2,8,52),
                      (1,4,3,7,104), (1,8,7,26)):
            count = 1
            for size in shape:
                count *= size
            owner = torch.full((count+512,), -7, dtype=torch.bfloat16, device=device)
            base = owner[256:-256].view(shape)
            expected_owner = owner.clone()
            expected = expected_owner[256:-256].view(shape)
            split = shape[-1] // 2
            da, db = base[..., :split], base[..., split:]
            sa = torch.randn((*shape[:-1],split+2),dtype=torch.bfloat16,device=device)[...,1:-1]
            sb = torch.randn((*shape[:-1],shape[-1]-split+2),dtype=torch.bfloat16,device=device)[...,1:-1]
            sa_before, sb_before = sa.clone(), sb.clone()
            expected[..., :split].copy_(sa); expected[..., split:].copy_(sb)
            ops.copy_pair(base,da,sa,db,sb)
            equal(owner,expected_owner,f"copy_arena_canary_{shape}")
            equal(sa,sa_before,f"copy_source_a_{shape}"); equal(sb,sb_before,f"copy_source_b_{shape}")
            before = owner.clone()
            try:
                ops.copy_pair(base,da,sa,da,sa)
            except RuntimeError:
                pass
            else:
                raise AssertionError("Overlapping destination accepted")
            equal(owner,before,f"copy_reject_before_write_{shape}")

        for pages in (33,34,40):
            received = torch.randn((4680,3,5,128),dtype=torch.bfloat16,device=device)
            before = received.clone()
            q,k,v = received.unbind(1)
            # Two scratch pages beyond managed capacity are represented explicitly.
            rows=(pages+3)*1560
            kp=torch.full((rows,5,128),-3,dtype=torch.bfloat16,device=device)
            vp=kp.clone(); ks=torch.full((28080,5,128),-4,dtype=torch.bfloat16,device=device); vs=ks.clone()
            slots=torch.cat([torch.arange(page*1560,(page+1)*1560,device=device,dtype=torch.int64)
                             for page in (pages,pages+1,pages+2)])
            ep,ev,es,esv=kp.clone(),vp.clone(),ks.clone(),vs.clone()
            ep.index_copy_(0,slots,k); ev.index_copy_(0,slots,v)
            es[23400:28080].copy_(k); esv[23400:28080].copy_(v)
            dense=ops.ingress(q,k,v,kp,vp,ks,vs,slots,15)
            equal(dense,q,f"recv_q_{pages}")
            for name,a,b in (("kp",kp,ep),("vp",vp,ev),("ks",ks,es),("vs",vs,esv),("source",received,before)):
                equal(a,b,f"recv_{name}_{pages}")

        raw=torch.randn((1,585,15360),dtype=torch.bfloat16,device=device)
        qw=torch.randn(5120,dtype=torch.bfloat16,device=device); kw=qw.clone()
        c=torch.randn((1,585,1,128),dtype=torch.bfloat16,device=device)
        s=torch.randn_like(c)
        qr,kr,vr=raw.split(5120,-1)
        qr=torch_npu.npu_rms_norm(qr,qw,epsilon=1e-6)[0].view(1,585,40,128)
        kr=torch_npu.npu_rms_norm(kr,kw,epsilon=1e-6)[0].view(1,585,40,128)
        qr=torch_npu.npu_rotary_mul(qr,c,s,rotary_mode="interleave")
        kr=torch_npu.npu_rotary_mul(kr,c,s,rotary_mode="interleave")
        expected=torch.stack((qr,kr,vr.view(1,585,40,128)),2).view(1,585,3,8,5,128).permute(3,1,2,0,4,5).contiguous()
        equal(ops.qk_send(raw,qw,kw,c,s),expected,"qk_native_bf16_send")
        for length in (64,512):
            key=torch.randn((1,length,40,128),dtype=torch.bfloat16,device=device)
            value=torch.randn_like(key)
            expected=torch_npu.npu_fused_infer_attention_score(qr,key,value,num_heads=40,
                    num_key_value_heads=40,input_layout="BSND",scale=128**-.5)[0]
            equal(ops.cross(qr,key,value,128**-.5),expected,f"cross_native_{length}")
        desc=ops.describe(raw)
        assert tuple(desc[5])==tuple(raw.shape) and desc[8]==raw.untyped_storage().nbytes()
        assert desc[9]==raw.untyped_storage().data_ptr() and desc[10]==raw.data_ptr()
        results.append("descriptor_extent_owner")
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps({"passed":results,"device":args.device,
        "manifest_sha256":hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "model_validated":False,"performance_validated":False},indent=2)+"\n")
    print(f"PASS: {len(results)} checks on logical device {args.device}")
    faulthandler.cancel_dump_traceback_later()


if __name__ == "__main__":
    main()
