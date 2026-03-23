from __future__ import annotations

import json
from pathlib import Path

import torch

from memory_ops.compression import ratio_to_count


def _apply_max_ratio_cap(
    proportions: dict[tuple[int, int], float],
    *,
    target_ratio: float,
    total_heads: int,
    max_ratio_per_head: float,
) -> dict[tuple[int, int], float]:
    if max_ratio_per_head >= 1.0 and target_ratio * total_heads <= 1.0:
        return proportions

    if target_ratio * total_heads <= 0:
        return proportions

    max_proportion = max_ratio_per_head / (target_ratio * total_heads)
    if max(proportions.values(), default=0.0) <= max_proportion:
        return proportions

    result = dict(proportions)
    for _ in range(total_heads + 1):
        excess_total = 0.0
        uncapped_total = 0.0
        capped_keys: set[tuple[int, int]] = set()

        for key, prop in result.items():
            if prop > max_proportion:
                excess_total += prop - max_proportion
                capped_keys.add(key)
            else:
                uncapped_total += prop

        if excess_total == 0.0:
            break

        if uncapped_total == 0.0:
            for key in result:
                result[key] = min(result[key], max_proportion)
            break

        for key, prop in result.items():
            if key in capped_keys:
                result[key] = max_proportion
            else:
                result[key] = prop + (prop / uncapped_total) * excess_total

    return result


def load_head_budget_counts_from_json(
    path: str | Path,
    *,
    num_layers: int,
    num_heads: int,
    seq_len: int,
    target_ratio: float,
    max_ratio_per_head: float = 1.0,
) -> torch.Tensor:
    """Load a head-budget JSON file and convert it to integer counts."""
    budget_path = Path(path)
    with budget_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    proportions: dict[tuple[int, int], float] = {}
    for layer_idx in range(num_layers):
        for head_idx in range(num_heads):
            key = f"L{layer_idx}H{head_idx}"
            proportions[(layer_idx, head_idx)] = float(raw.get(key, 0.0))

    total = sum(proportions.values())
    if total <= 0:
        raise ValueError("budget JSON must contain a positive total proportion")
    proportions = {key: value / total for key, value in proportions.items()}
    proportions = _apply_max_ratio_cap(
        proportions,
        target_ratio=target_ratio,
        total_heads=num_layers * num_heads,
        max_ratio_per_head=max_ratio_per_head,
    )

    total_budget = ratio_to_count(seq_len, target_ratio) * num_layers * num_heads
    max_tokens_per_head = max(1, int(seq_len * max_ratio_per_head))

    fractional_items: list[tuple[float, tuple[int, int]]] = []
    counts = torch.zeros((num_layers, num_heads), dtype=torch.long)
    assigned = 0
    for layer_idx in range(num_layers):
        for head_idx in range(num_heads):
            desired = min(proportions[(layer_idx, head_idx)] * total_budget, float(max_tokens_per_head))
            base = max(1, min(int(desired), seq_len))
            counts[layer_idx, head_idx] = base
            assigned += base
            fractional_items.append((desired - int(desired), (layer_idx, head_idx)))

    if assigned < total_budget:
        fractional_items.sort(reverse=True)
        remaining = total_budget - assigned
        for _, (layer_idx, head_idx) in fractional_items:
            if remaining == 0:
                break
            if counts[layer_idx, head_idx] >= min(seq_len, max_tokens_per_head):
                continue
            counts[layer_idx, head_idx] += 1
            remaining -= 1
    elif assigned > total_budget:
        fractional_items.sort()
        overflow = assigned - total_budget
        for _, (layer_idx, head_idx) in fractional_items:
            if overflow == 0:
                break
            if counts[layer_idx, head_idx] <= 1:
                continue
            counts[layer_idx, head_idx] -= 1
            overflow -= 1

    return counts
