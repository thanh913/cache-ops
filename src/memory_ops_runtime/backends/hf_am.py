"""AM payload types and CacheBlock construction helpers."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from memory_ops import CacheBlock


@dataclass(frozen=True)
class AMPayload:
    layer_keys: tuple[torch.Tensor, ...]
    layer_values: tuple[torch.Tensor, ...]
    layer_beta: tuple[torch.Tensor | None, ...]


def build_am_block(
    compacted: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    positions: torch.Tensor,
) -> CacheBlock:
    """Build a CacheBlock from AM-compacted per-layer tensors."""
    if positions.ndim != 1:
        raise ValueError("positions must be a rank-1 tensor")
    seq_len = int(positions.shape[0])
    layer_keys: list[torch.Tensor] = []
    layer_values: list[torch.Tensor] = []
    layer_beta: list[torch.Tensor] = []
    for layer_idx, (keys, beta, values) in enumerate(compacted):
        if keys.ndim != 4 or values.ndim != 4:
            raise ValueError(
                f"layer {layer_idx} compacted keys/values must have shape (1, H, T, d); "
                f"got {tuple(keys.shape)} and {tuple(values.shape)}"
            )
        if beta.ndim != 2:
            raise ValueError(f"layer {layer_idx} beta must have shape (H, T); got rank {beta.ndim}")
        if keys.shape != values.shape:
            raise ValueError(
                f"layer {layer_idx} compacted keys and values must share shape; "
                f"got {tuple(keys.shape)} and {tuple(values.shape)}"
            )
        if keys.shape[2] != seq_len:
            raise ValueError(
                f"layer {layer_idx} compacted sequence length must match positions; "
                f"got {keys.shape[2]} and {seq_len}"
            )
        expected_beta = (keys.shape[1], keys.shape[2])
        if tuple(beta.shape) != expected_beta:
            raise ValueError(
                f"layer {layer_idx} beta must have shape {expected_beta}; got {tuple(beta.shape)}"
            )
        layer_keys.append(keys.clone())
        layer_values.append(values.clone())
        layer_beta.append(beta.clone())

    kv = AMPayload(
        layer_keys=tuple(layer_keys),
        layer_values=tuple(layer_values),
        layer_beta=tuple(layer_beta),
    )
    return CacheBlock(
        kv=kv,
        positions=positions.clone(),
        rope_positions=positions.clone(),
    )
