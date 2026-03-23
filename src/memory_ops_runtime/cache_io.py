"""Save and load CacheBlock directories."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file, load_file

from memory_ops import CacheBlock


def save_block(
    block: CacheBlock,
    path: str | Path,
    *,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Save a CacheBlock to a directory on disk."""
    from transformers.cache_utils import DynamicCache
    from memory_ops_runtime.backends.hf_am import AMPayload

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    tensors: dict[str, torch.Tensor] = {}
    positions = block.positions
    rope_positions = block.rope_positions
    if not isinstance(positions, torch.Tensor):
        positions = torch.tensor(positions, dtype=torch.long)
    if not isinstance(rope_positions, torch.Tensor):
        rope_positions = torch.tensor(rope_positions, dtype=torch.long)

    tensors["positions"] = positions.cpu().contiguous()
    tensors["rope_positions"] = rope_positions.cpu().contiguous()

    kv = block.kv
    if isinstance(kv, AMPayload):
        block_type = "am_payload"
        num_layers = len(kv.layer_keys)
        for i in range(num_layers):
            tensors[f"layer_{i:03d}_keys"] = kv.layer_keys[i].cpu().contiguous()
            tensors[f"layer_{i:03d}_values"] = kv.layer_values[i].cpu().contiguous()
            if kv.layer_beta[i] is not None:
                tensors[f"layer_{i:03d}_beta"] = kv.layer_beta[i].cpu().contiguous()
    elif isinstance(kv, DynamicCache):
        block_type = "dynamic_cache"
        num_layers = len(kv.layers)
        for i, layer in enumerate(kv.layers):
            tensors[f"layer_{i:03d}_keys"] = layer.keys.cpu().contiguous()
            tensors[f"layer_{i:03d}_values"] = layer.values.cpu().contiguous()
    else:
        raise TypeError(f"unsupported KV type: {type(kv)}")

    save_file(tensors, path / "tensors.safetensors")

    meta = {
        "block_type": block_type,
        "num_layers": num_layers,
        "seq_len": block.seq_len,
        **(metadata or {}),
    }
    (path / "meta.json").write_text(json.dumps(meta, indent=2))
    return path


def load_block(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> CacheBlock:
    """Load a CacheBlock from a directory on disk."""
    from transformers.cache_utils import DynamicCache
    from memory_ops_runtime.backends.hf_am import AMPayload

    path = Path(path)
    meta = json.loads((path / "meta.json").read_text())
    tensors = load_file(path / "tensors.safetensors", device=str(device))

    positions = tensors["positions"]
    rope_positions = tensors["rope_positions"]
    num_layers = meta["num_layers"]

    if meta["block_type"] == "am_payload":
        layer_keys = []
        layer_values = []
        layer_beta: list[torch.Tensor | None] = []
        for i in range(num_layers):
            layer_keys.append(tensors[f"layer_{i:03d}_keys"])
            layer_values.append(tensors[f"layer_{i:03d}_values"])
            beta_key = f"layer_{i:03d}_beta"
            layer_beta.append(tensors[beta_key] if beta_key in tensors else None)
        kv = AMPayload(
            layer_keys=tuple(layer_keys),
            layer_values=tuple(layer_values),
            layer_beta=tuple(layer_beta),
        )
    elif meta["block_type"] == "dynamic_cache":
        layer_data = []
        for i in range(num_layers):
            layer_data.append((
                tensors[f"layer_{i:03d}_keys"],
                tensors[f"layer_{i:03d}_values"],
            ))
        kv = DynamicCache(ddp_cache_data=layer_data)
    else:
        raise ValueError(f"unknown block_type: {meta['block_type']}")

    return CacheBlock(kv=kv, positions=positions, rope_positions=rope_positions)
