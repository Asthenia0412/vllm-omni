# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Matrix session lifecycle for the shared Wan tiny codec."""

import torch
from torch import nn

from vllm_omni.diffusion.models.wan2_2.taew2_2 import TAEW2Decoder, TAEW2Encoder, TAEW2EncodeState


class MatrixCodec(nn.Module):
    def __init__(self, checkpoint_path: str):
        super().__init__()
        self.encoder = TAEW2Encoder(checkpoint_path, torch.bfloat16)
        self.decoder = TAEW2Decoder(checkpoint_path, torch.bfloat16)
        self.encoder_state = TAEW2EncodeState()
        self.decoder_state = self.decoder.new_decode_state("matrix")

    def clear_cache(self) -> None:
        self.encoder_state = TAEW2EncodeState()
        self.decoder.release(self.decoder_state)
        self.decoder_state = self.decoder.new_decode_state("matrix")

    def encode(self, pixels: torch.Tensor) -> torch.Tensor:
        return self.encoder.encode_chunk(pixels, TAEW2EncodeState())

    def encode_stream(self, pixels: torch.Tensor) -> torch.Tensor:
        return self.encoder.encode_chunk(pixels, self.encoder_state)

    def decode_stream(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder.decode_chunk(latents, self.decoder_state)
