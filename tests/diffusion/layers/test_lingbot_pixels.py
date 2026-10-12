# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import numpy as np
import pytest
import torch

from vllm_omni.diffusion.layers.lingbot_pixels import planar_uint8_frames


def _previous_device_order(video):
    # Preserved pre-optimization byte contract, including its decoder-dtype
    # rounding before promotion. Use generic NumPy contiguity as an oracle.
    frames = (video[0] / 2 + 0.5).clamp(0, 1).permute(1, 2, 3, 0)
    return np.ascontiguousarray(frames.float().mul_(255).round_().to(torch.uint8).numpy())


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_every_16_bit_input_pattern_preserves_pixel_bytes(dtype):
    patterns = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(dtype).reshape(1, 1, 1, 256, 256)
    video = patterns.expand(1, 3, 1, 256, 256)
    output = planar_uint8_frames(video)
    np.testing.assert_array_equal(output, _previous_device_order(video))
    assert output.flags.c_contiguous and output.dtype == np.uint8


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_pixel_order_and_rounding_for_strided_chunks(dtype):
    # Different channels/frames and a strided spatial view catch layout/order
    # errors; include values outside [-1,1] to exercise clipping.
    torch.manual_seed(37)
    video = torch.randn(1, 3, 5, 8, 18, dtype=dtype)[..., ::2]
    output = planar_uint8_frames(video)
    np.testing.assert_array_equal(output, _previous_device_order(video))
    assert output.shape == (5, 8, 9, 3) and output.flags.c_contiguous


def test_fp32_halfway_rounding_and_saturation_preserves_bytes():
    values = torch.arange(256, dtype=torch.float32).add(.5).div(255).sub(.5).mul(2)
    values = torch.cat((torch.tensor([-4., -1., 1., 4.]), values))
    video = values.reshape(1, 1, 1, 1, -1).repeat(1, 3, 1, 1, 1)
    np.testing.assert_array_equal(planar_uint8_frames(video), _previous_device_order(video))


@pytest.mark.parametrize("shape", [(2, 3, 1, 8, 8), (1, 4, 1, 8, 8), (3, 1, 8, 8)])
def test_invalid_chunk_geometry_is_rejected(shape):
    with pytest.raises(ValueError):
        planar_uint8_frames(torch.zeros(shape))
