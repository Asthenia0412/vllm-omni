# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Lazy Triton-Ascend A3 vector kernels; disabled until measured contract gates pass.

Every arithmetic intermediate is explicitly FP32 except source-prescribed BF16
boundaries. enable_fp_fusion=False prevents contraction of multiply/add pairs.
No CPU/NPU fallback or implicit precision downgrade is implemented here.
"""
import torch
import triton
import triton.language as tl
from triton.language.extra.cann import libdevice


@triton.jit
def _qkv_norm_rope(X, WQ, WK, C, S, YQ, YK, WIDTH: tl.constexpr, TOKENS: tl.constexpr,
                   HEAD_DIM: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    q_total = tl.full((), 0.0, tl.float32)
    k_total = tl.full((), 0.0, tl.float32)
    # Static Q/K pointers avoid a Triton3.2.2 lowering crash on pointer select
    # inside runtime projection control flow; no output arithmetic is changed.
    for start in range(tl.cdiv(WIDTH, BLOCK)):
        col = start * BLOCK + lane
        q = tl.load(X + row * (3 * WIDTH) + col, col < WIDTH, 0).to(tl.float32)
        k = tl.load(X + row * (3 * WIDTH) + WIDTH + col, col < WIDTH, 0).to(tl.float32)
        q_total += tl.sum(q * q, 0)
        k_total += tl.sum(k * k, 0)
    q_rms = tl.sqrt(q_total / WIDTH + EPS)
    k_rms = tl.sqrt(k_total / WIDTH + EPS)
    for start in range(tl.cdiv(WIDTH, BLOCK)):
        col = start * BLOCK + lane
        partner_col = col ^ 1
        q = tl.load(X + row * (3 * WIDTH) + col, col < WIDTH, 0).to(tl.float32)
        k = tl.load(X + row * (3 * WIDTH) + WIDTH + col, col < WIDTH, 0).to(tl.float32)
        qw = tl.load(WQ + col, col < WIDTH, 0).to(tl.float32)
        kw = tl.load(WK + col, col < WIDTH, 0).to(tl.float32)
        qnorm = (q / q_rms * qw).to(YQ.dtype.element_ty).to(tl.float32)
        knorm = (k / k_rms * kw).to(YK.dtype.element_ty).to(tl.float32)
        qp = tl.load(X + row * (3 * WIDTH) + partner_col, partner_col < WIDTH, 0).to(tl.float32)
        kp = tl.load(X + row * (3 * WIDTH) + WIDTH + partner_col, partner_col < WIDTH, 0).to(tl.float32)
        qwp = tl.load(WQ + partner_col, partner_col < WIDTH, 0).to(tl.float32)
        kwp = tl.load(WK + partner_col, partner_col < WIDTH, 0).to(tl.float32)
        qpartner = (qp / q_rms * qwp).to(YQ.dtype.element_ty).to(tl.float32)
        kpartner = (kp / k_rms * kwp).to(YK.dtype.element_ty).to(tl.float32)
        coeff = (row % TOKENS) * (HEAD_DIM // 2) + (col % HEAD_DIM) // 2
        cosine = tl.load(C + coeff, col < WIDTH, 0).to(tl.float32)
        sine = tl.load(S + coeff, col < WIDTH, 0).to(tl.float32)
        qout = tl.where(col % 2 == 0, qnorm * cosine - qpartner * sine, qnorm * cosine + qpartner * sine)
        kout = tl.where(col % 2 == 0, knorm * cosine - kpartner * sine, knorm * cosine + kpartner * sine)
        tl.store(YQ + row * WIDTH + col, qout, col < WIDTH)
        tl.store(YK + row * WIDTH + col, kout, col < WIDTH)


@triton.jit
def _layernorm_modulation(X, SCALE, SHIFT, Y, WIDTH: tl.constexpr, TOKENS: tl.constexpr,
                          FRAMES: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    x = tl.load(X + row * WIDTH + col, col < WIDTH, 0).to(tl.float32)
    mean = tl.sum(x, 0) / WIDTH
    centered = tl.where(col < WIDTH, x - mean, 0.0)
    variance = tl.sum(centered * centered, 0) / WIDTH
    normalized = centered / tl.sqrt(variance + EPS)
    frame = (row // TOKENS) * FRAMES + (row % TOKENS) // (TOKENS // FRAMES)
    scale = tl.load(SCALE + frame * WIDTH + col, col < WIDTH, 0).to(tl.float32)
    shift = tl.load(SHIFT + frame * WIDTH + col, col < WIDTH, 0).to(tl.float32)
    result = normalized * (1.0 + scale) + shift
    tl.store(Y + row * WIDTH + col, result, col < WIDTH)


@triton.jit
def _tanh_gelu_fp32(x):
    cubic = x * x * x
    argument = 0.7978845608028654 * (x + 0.044715 * cubic)
    return 0.5 * x * (1.0 + libdevice.tanh(argument))


@triton.jit
def _gelu_quant(X, Q, SCALE, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    # Whole 13824-wide tanh+banker's rounding exceeds A3's 192KiB UB. Two
    # bounded passes recompute GELU instead of materializing its BF16 output.
    maximum = tl.full((), 0.0, tl.float32)
    for start in range(tl.cdiv(WIDTH, BLOCK)):
        col = start * BLOCK + lane
        x = tl.load(X + row * WIDTH + col, col < WIDTH, 0).to(tl.float32)
        rounded = _tanh_gelu_fp32(x).to(X.dtype.element_ty).to(tl.float32)
        maximum = tl.maximum(maximum, tl.max(tl.where(col < WIDTH, tl.abs(rounded), 0.0), 0))
    scale = tl.where(maximum > 0.0, maximum / 127.0, 1.0)
    for start in range(tl.cdiv(WIDTH, BLOCK)):
        col = start * BLOCK + lane
        x = tl.load(X + row * WIDTH + col, col < WIDTH, 0).to(tl.float32)
        rounded = _tanh_gelu_fp32(x).to(X.dtype.element_ty).to(tl.float32)
        payload = libdevice.nearbyint(rounded / scale)
        payload = tl.minimum(127.0, tl.maximum(-127.0, payload)).to(tl.int8)
        tl.store(Q + row * WIDTH + col, payload, col < WIDTH)
    tl.store(SCALE + row, scale)


@triton.jit
def _vae_rms_silu(X, GAMMA, BIAS, Y, CHANNELS: tl.constexpr, SPATIAL: tl.constexpr,
                  EPS: tl.constexpr, SCALE: tl.constexpr, HAS_BIAS: tl.constexpr,
                  UPCAST: tl.constexpr, NPU_RMS: tl.constexpr,
                  CHANNEL_BLOCK: tl.constexpr, PIXEL_BLOCK: tl.constexpr):
    spatial = tl.program_id(0) * PIXEL_BLOCK + tl.arange(0, PIXEL_BLOCK)
    batch = tl.program_id(1)
    channel = tl.arange(0, CHANNEL_BLOCK)
    offset = (batch * CHANNELS + channel[:, None]) * SPATIAL + spatial[None, :]
    mask = (channel[:, None] < CHANNELS) & (spatial[None, :] < SPATIAL)
    x = tl.load(X + offset, mask, 0).to(tl.float32)
    # CPU reference and device validation gate non-upcast norm separately.
    gamma = tl.load(GAMMA + channel, channel < CHANNELS, 0).to(tl.float32)
    if NPU_RMS:
        rms = tl.sqrt(tl.sum(x * x, 0) / CHANNELS + EPS)
        affine = (x / rms[None, :] * gamma[:, None]).to(X.dtype.element_ty).to(tl.float32)
    else:
        norm = tl.sqrt(tl.sum(x * x, 0))
        if not UPCAST:
            norm = norm.to(X.dtype.element_ty).to(tl.float32)
        denom = tl.maximum(norm, EPS)
        if not UPCAST:
            denom = denom.to(X.dtype.element_ty).to(tl.float32)
        normalized = (x / denom[None, :]).to(X.dtype.element_ty).to(tl.float32)
        scaled = (normalized * SCALE).to(X.dtype.element_ty).to(tl.float32)
        affine = scaled * gamma[:, None]
        # Match affine promotion: non-FP32 gamma keeps the dtype boundary.
        if GAMMA.dtype.element_ty != tl.float32:
            affine = affine.to(X.dtype.element_ty).to(tl.float32)
    if HAS_BIAS:
        bias = tl.load(BIAS + channel, channel < CHANNELS, 0).to(tl.float32)
        affine = affine + bias[:, None]
        if Y.dtype.element_ty != tl.float32:
            affine = affine.to(Y.dtype.element_ty).to(tl.float32)
    output = affine / (1.0 + tl.exp(-affine))
    tl.store(Y + offset, output, mask)


_OPTIONS = {"enable_fp_fusion": False, "disable_fma": True}


def qkv_norm_rope(qkv, wq, wk, cosine, sine, heads, eps):
    x = qkv.contiguous()
    width = x.shape[-1] // 3
    query = torch.empty((*x.shape[:-1], width), device=x.device, dtype=x.dtype)
    key = torch.empty_like(query)
    _qkv_norm_rope[(x.shape[0] * x.shape[1],)](
        x, wq.contiguous(), wk.contiguous(), cosine.contiguous(), sine.contiguous(), query, key,
        width, x.shape[1], width // heads, eps, 512, **_OPTIONS,
    )
    # Match native Q/K's contiguous BSND layout and preserve V's original view,
    # avoiding an artificial V copy or later hidden Q/K contiguous conversion.
    value = qkv.split(width, -1)[2]
    return tuple(t.unflatten(-1, (heads, width // heads)) for t in (query, key, value))


def layernorm_modulation(value, scale, shift, eps):
    x = value.contiguous()
    output = torch.empty_like(x)
    _layernorm_modulation[(x.shape[0] * x.shape[1],)](
        x, scale.contiguous(), shift.contiguous(), output, x.shape[-1], x.shape[1], scale.shape[1],
        eps, triton.next_power_of_2(x.shape[-1]), **_OPTIONS,
    )
    return output


def gelu_quant(value):
    x = value.contiguous()
    output = torch.empty_like(x, dtype=torch.int8)
    scales = torch.empty(x.shape[:-1], device=x.device, dtype=torch.float32)
    _gelu_quant[(x.numel() // x.shape[-1],)](
        x, output, scales, x.shape[-1], 1024, **_OPTIONS,
    )
    return output, scales


def vae_rms_silu(value, gamma, bias, eps, scale, upcast, source_contract):
    x = value.contiguous()
    output_dtype = value.dtype if source_contract == "npu_rms_norm" else torch.promote_types(value.dtype, gamma.dtype)
    if bias is not None:
        output_dtype = torch.promote_types(output_dtype, bias.dtype)
    output = torch.empty_like(x, dtype=output_dtype)
    spatial = x.shape[2] * x.shape[3] * x.shape[4]
    # Bounded UB footprint: C×8 FP32 values, rather than a full spatial tile.
    _vae_rms_silu[(triton.cdiv(spatial, 8), x.shape[0])](
        x, gamma.contiguous().reshape(-1), gamma if bias is None else bias.contiguous().reshape(-1),
        output, x.shape[1], spatial, eps, scale, bias is not None, upcast, source_contract == "npu_rms_norm",
        triton.next_power_of_2(x.shape[1]), 8, **_OPTIONS,
    )
    return output
