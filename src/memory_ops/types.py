from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence, TypeAlias


Span: TypeAlias = tuple[int, int]


def to_int_list(values: object) -> list[int]:
    if hasattr(values, "tolist"):
        return [int(item) for item in values.tolist()]
    return [int(item) for item in values]  # type: ignore[arg-type]


@dataclass(frozen=True)
class CacheBlock:
    kv: Any
    positions: Any
    rope_positions: Any
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def seq_len(self) -> int:
        if hasattr(self.positions, "shape"):
            return int(self.positions.shape[-1])
        return len(self.positions)

    def with_positions(self, new_positions: Any) -> "CacheBlock":
        return CacheBlock(
            kv=self.kv,
            positions=new_positions,
            rope_positions=self.rope_positions,
            metadata=self.metadata,
        )

    def with_metadata(self, metadata: Mapping[str, Any]) -> "CacheBlock":
        return CacheBlock(
            kv=self.kv,
            positions=self.positions,
            rope_positions=self.rope_positions,
            metadata={**dict(self.metadata), **dict(metadata)},
        )


@dataclass(frozen=True, init=False)
class CacheContext:
    blocks: tuple[CacheBlock, ...]

    def __init__(self, blocks: Sequence[CacheBlock] | CacheBlock | "CacheContext") -> None:
        if isinstance(blocks, CacheContext):
            object.__setattr__(self, "blocks", blocks.blocks)
        elif isinstance(blocks, CacheBlock):
            object.__setattr__(self, "blocks", (blocks,))
        else:
            object.__setattr__(self, "blocks", tuple(blocks))

    def __iter__(self):
        return iter(self.blocks)

    def __len__(self) -> int:
        return len(self.blocks)

    def __getitem__(self, index: int) -> CacheBlock:
        return self.blocks[index]

    def validate(self) -> None:
        last_end: int | None = None
        for block in self.blocks:
            positions = to_int_list(block.positions)
            rope_positions = to_int_list(block.rope_positions)
            if len(positions) != len(rope_positions):
                raise ValueError("positions and rope_positions must have matching lengths")
            if positions != sorted(set(positions)):
                raise ValueError("block positions must be sorted and unique")
            if last_end is not None and positions and positions[0] < last_end:
                raise ValueError("context block positions must be ordered and non-overlapping")
            if positions:
                last_end = positions[-1] + 1

    @property
    def next_position(self) -> int:
        logical = 0
        for block in self.blocks:
            positions = to_int_list(block.positions)
            if positions:
                logical = max(logical, positions[-1] + 1)
        return logical

    @property
    def physical_tokens(self) -> int:
        return sum(block.seq_len for block in self.blocks)

    @property
    def requires_rope_correction(self) -> bool:
        for block in self.blocks:
            if to_int_list(block.positions) != to_int_list(block.rope_positions):
                return True
        return False
