from __future__ import annotations

import math


def ratio_to_count(seq_len: int, ratio: float) -> int:
    if seq_len < 0:
        raise ValueError("seq_len must be non-negative")
    if not 0.0 < ratio <= 1.0:
        raise ValueError("target_ratio must be in (0, 1]")
    if seq_len == 0:
        return 0
    rounded = int(math.floor((seq_len * ratio) + 0.5))
    return max(1, min(seq_len, rounded))


def select_uniform_indices(seq_len: int, target_count: int) -> list[int]:
    if seq_len < 0:
        raise ValueError("seq_len must be non-negative")
    if not 0 <= target_count <= seq_len:
        raise ValueError("target_count must be between 0 and seq_len")
    if target_count == 0:
        return []
    if target_count == seq_len:
        return list(range(seq_len))
    if target_count == 1:
        return [0]

    step = (seq_len - 1) / (target_count - 1)
    indices = [int(math.floor(i * step)) for i in range(target_count - 1)]
    indices.append(seq_len - 1)
    return indices


def compression_indices(seq_len: int, ratio: float, strategy: str) -> list[int]:
    target_count = ratio_to_count(seq_len, ratio)
    if strategy == "uniform":
        return select_uniform_indices(seq_len, target_count)
    raise ValueError(f"unsupported compression strategy: {strategy}")
