"""MXFP8 input packing helpers for the standalone benchmark."""
from typing import Tuple

import torch

D_HEAD = 512
FP8_MAX = 448.0
KV_GROUP_SIZE = 64


def _quantize_groups(
    x: torch.Tensor, group_size: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert x.shape[-1] == D_HEAD
    grouped = x.float().reshape(*x.shape[:-1], D_HEAD // group_size, group_size)
    scale = grouped.abs().amax(dim=-1) / FP8_MAX
    scale = torch.pow(2.0, torch.ceil(torch.log2(scale.clamp_min(2.0**-126))))
    scale_e8m0 = scale.to(torch.float8_e8m0fnu)
    scale_fp32 = scale_e8m0.float()
    quantized = (grouped / scale_fp32.unsqueeze(-1)).clamp(-FP8_MAX, FP8_MAX)
    quantized = quantized.to(torch.float8_e4m3fn)
    dequantized = (quantized.float() * scale_fp32.unsqueeze(-1)).reshape_as(x.float())
    return (
        quantized.reshape_as(x).view(torch.uint8),
        scale_e8m0.view(torch.uint8),
        dequantized,
    )


def pack_dual_q64(q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pack Q with one scale for corresponding 32-D chunks in both halves."""
    assert q.shape[-1] == D_HEAD
    paired = (
        q.float()
        .reshape(*q.shape[:-1], 2, D_HEAD // 64, 32)
        .transpose(-3, -2)
        .reshape(*q.shape[:-1], D_HEAD // 64, 64)
    )
    scale = paired.abs().amax(dim=-1) / FP8_MAX
    scale = torch.pow(2.0, torch.ceil(torch.log2(scale.clamp_min(2.0**-126))))
    scale_e8m0 = scale.to(torch.float8_e8m0fnu)
    scale_fp32 = scale_e8m0.float()
    quantized = (paired / scale_fp32.unsqueeze(-1)).clamp(-FP8_MAX, FP8_MAX)
    quantized = quantized.to(torch.float8_e4m3fn)
    data = quantized.view(torch.uint8)
    scales = scale_e8m0.view(torch.uint8)
    dequantized = quantized.float() * scale_fp32.unsqueeze(-1)
    data = (
        data.reshape(*q.shape[:-1], D_HEAD // 64, 2, 32)
        .transpose(-3, -2)
        .reshape_as(q)
    )
    dequantized = (
        dequantized.reshape(*q.shape[:-1], D_HEAD // 64, 2, 32)
        .transpose(-3, -2)
        .reshape_as(q.float())
    )
    packed = torch.empty(
        (*q.shape[:-1], D_HEAD + 16), dtype=torch.uint8, device=q.device
    )
    packed[..., :D_HEAD] = data
    packed[..., D_HEAD:] = torch.cat((scales, scales), dim=-1)
    return packed, dequantized


def pack_dual_prefill_kv64(kv: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pack dual-MXFP8 KV with replicated 16B scale slots in the tail plane."""
    assert kv.ndim == 3
    data, scales, dequantized = _quantize_groups(kv, KV_GROUP_SIZE)
    scale_slots = torch.cat((scales, scales), dim=-1)
    num_tokens = kv.shape[0] * kv.shape[1]
    bytes_per_token = D_HEAD + scale_slots.shape[-1]
    storage = torch.empty(
        num_tokens * bytes_per_token, dtype=torch.uint8, device=kv.device
    )
    storage[: num_tokens * D_HEAD] = data.reshape(-1)
    storage[num_tokens * D_HEAD :] = scale_slots.reshape(-1)
    return storage.view(*kv.shape[:-1], bytes_per_token), dequantized


def pack_dual_decode_kv_pages(
    kv: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack decode pages with the same independent group quantization as prefill."""
    assert kv.ndim == 4 and kv.shape[2] == 1 and kv.shape[-1] == D_HEAD
    data, scales, dequantized = _quantize_groups(kv, KV_GROUP_SIZE)
    num_pages, page_size, h_kv, _ = kv.shape
    num_tokens = page_size * h_kv
    flat_data = data.reshape(num_pages, -1)
    # Decode addresses one replicated 16B slot per token from the page tail.
    # Keep the scale bytes adjacent to their token, unlike the legacy packed
    # representation whose tail was two contiguous copies of the whole page.
    scale_slots = torch.cat((scales, scales), dim=-1).reshape(num_pages, -1)
    storage = torch.empty(
        (num_pages, num_tokens * (D_HEAD + 16)), dtype=torch.uint8, device=kv.device
    )
    storage[:, : num_tokens * D_HEAD] = flat_data
    storage[:, num_tokens * D_HEAD :] = scale_slots
    return (
        storage.view(num_pages, page_size, h_kv, D_HEAD + 16),
        dequantized,
        scales.view(num_pages, page_size, h_kv, D_HEAD // KV_GROUP_SIZE),
        data.view(num_pages, page_size, h_kv, D_HEAD),
    )
