# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import copy
from types import SimpleNamespace

import pytest
import torch
from diffusers.models.autoencoders.autoencoder_kl_wan import WanDecoder3d

from vllm_omni.diffusion.distributed.autoencoders.wan_npu_decoder_fusion import (
    NPUWanNormSiLU,
    install_wan_npu_decoder_fusion,
)
from vllm_omni.experimental.ar_diffusion.streaming_decode import WanStreamingDecoder
from diffusers.models.autoencoders.autoencoder_kl_wan import WanCausalConv3d

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def vae():
    decoder = WanDecoder3d(dim=8, z_dim=4, dim_mult=[1, 2, 4], num_res_blocks=1, attn_scales=[]).eval()
    return SimpleNamespace(
        decoder=decoder, post_quant_conv=WanCausalConv3d(4, 4, 1).eval(),
        _cached_conv_counts={"decoder": sum(m.__class__.__name__ == "WanCausalConv3d" for m in decoder.modules())},
    )


@pytest.mark.parametrize("autocast", [False, True])
def test_npu_adapter_keeps_cpu_streaming_reset_and_parameter_names(autocast):
    torch.manual_seed(42)
    original = vae()
    candidate = copy.deepcopy(original)
    keys = set(candidate.decoder.state_dict())
    counts = install_wan_npu_decoder_fusion(candidate)
    assert counts["norm_silu_regions"] > 0
    assert set(candidate.decoder.state_dict()) == keys
    assert install_wan_npu_decoder_fusion(candidate) is counts
    latents = [torch.randn(1, 4, frames, 6, 10) for frames in (2, 1, 2)]
    with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
        a, b = WanStreamingDecoder(original), WanStreamingDecoder(candidate)
        sa, sb = a.new_decode_state("one"), b.new_decode_state("one")
        for index, value in enumerate(latents):
            if index == 2:
                sa, sb = a.new_decode_state("two"), b.new_decode_state("two")
            assert torch.equal(a.decode_chunk(value, sa), b.decode_chunk(value, sb))


def test_install_rejects_invalid_chain_without_partial_mutation():
    candidate = vae()
    norms = [m for m in candidate.decoder.modules() if m.__class__.__name__ == "WanResidualBlock"]
    original = norms[0].norm1
    candidate.decoder.nonlinearity = torch.nn.SiLU(inplace=True)
    with pytest.raises(ValueError, match="non-inplace"):
        install_wan_npu_decoder_fusion(candidate)
    assert norms[0].norm1 is original
    assert not any(isinstance(module, NPUWanNormSiLU) for module in candidate.decoder.modules())
