"""Eight-rank correctness only: rebuilt send through the original collective."""
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch_npu


def main():
    rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(rank)
    torch.npu.set_compile_mode(jit_compile=False)
    torch.npu.config.allow_internal_format = True
    dist.init_process_group("hccl")
    from vllm_omni.diffusion.distributed.comm import all_to_all_5D
    from vllm_omni.diffusion.layers.lingbot_recovery import load, Send
    provider = load()
    torch.manual_seed(900 + rank)
    with torch.inference_mode():
        raw=torch.randn((1,585,15360),dtype=torch.bfloat16,device=f"npu:{rank}")
        qw=torch.randn(5120,dtype=torch.bfloat16,device=raw.device); kw=qw.clone()
        c=torch.randn((1,585,1,128),dtype=torch.bfloat16,device=raw.device); s=torch.randn_like(c)
        q,k,v=raw.split(5120,-1)
        q=torch_npu.npu_rms_norm(q,qw,epsilon=1e-6)[0].view(1,585,40,128)
        k=torch_npu.npu_rms_norm(k,kw,epsilon=1e-6)[0].view(1,585,40,128)
        q=torch_npu.npu_rotary_mul(q,c,s,rotary_mode="interleave")
        k=torch_npu.npu_rotary_mul(k,c,s,rotary_mode="interleave")
        original=torch.stack((q,k,v.view(1,585,40,128)),2)
        expected=all_to_all_5D(original,group=dist.group.WORLD)
        packed=provider.call("qk_send",raw,qw,kw,c,s)
        lease=Send(provider,raw.view(1,585,3,40,128),packed,(raw,qw,kw,c,s))
        actual=lease.exchange(all_to_all_5D,dist.group.WORLD)
        torch.npu.synchronize()
        assert torch.equal(actual.cpu().view(torch.int16),expected.cpu().view(torch.int16))
        rejected=False
        try:
            lease.exchange(all_to_all_5D,dist.group.WORLD)
        except RuntimeError:
            rejected=True
        assert rejected
        Path(os.environ["ROOT_LINGBOT_SP8_FATAL_DIR"],f"hccl-rank{rank}.json").write_text(
            json.dumps(dict(rank=rank,bitwise_equal=True,reused_lease_rejected=True,model_validated=False))+"\n")
    dist.barrier()
    dist.destroy_process_group()
    print("PASS HCCL rank",rank,flush=True)


if __name__ == "__main__":
    main()
