from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from .types import CacheBlock


@runtime_checkable
class CacheBackend(Protocol):
    name: str

    def prefill(self, input_ids: Any) -> CacheBlock:
        ...

    def slice(self, block: CacheBlock, start: int, end: int) -> CacheBlock:
        ...

    def rope_correct(self, block: CacheBlock) -> CacheBlock:
        ...

    def evict(self, block: CacheBlock, ratio: float, strategy: str = "uniform") -> CacheBlock:
        ...
