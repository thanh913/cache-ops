from dataclasses import is_dataclass

import pytest

from memory_ops import CacheBlock, CacheContext


def test_cache_types_are_dataclasses() -> None:
    assert is_dataclass(CacheBlock)
    assert is_dataclass(CacheContext)


def test_cache_block_with_positions_only_changes_target_positions() -> None:
    block = CacheBlock(kv=(0, 1), positions=[0, 1], rope_positions=[0, 1], metadata={"source": "prefill"})
    moved = block.with_positions([10, 11])
    assert moved.positions == [10, 11]
    assert moved.rope_positions == [0, 1]
    assert moved.metadata == block.metadata


def test_cache_block_with_metadata_merges() -> None:
    block = CacheBlock(kv=(0, 1), positions=[0, 1], rope_positions=[0, 1], metadata={"source": "prefill"})
    annotated = block.with_metadata({"chunk_index": 0})
    assert annotated.metadata == {"source": "prefill", "chunk_index": 0}
    assert annotated.positions == block.positions


def test_cache_context_wraps_single_block_and_sequences() -> None:
    block = CacheBlock(kv=(0, 1), positions=[0, 1], rope_positions=[0, 1])
    single = CacheContext(block)
    pair = CacheContext([block, block])
    assert len(single) == 1
    assert len(pair) == 2


def test_validate_context_rejects_overlapping_blocks() -> None:
    left = CacheBlock(kv=(0, 1), positions=[0, 1], rope_positions=[0, 1])
    right = CacheBlock(kv=(2, 3), positions=[1, 2], rope_positions=[1, 2])
    with pytest.raises(ValueError):
        CacheContext([left, right]).validate()
