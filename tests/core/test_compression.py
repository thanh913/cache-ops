import pytest

from memory_ops import CacheContext
from memory_ops_runtime import GenerationConfig
from memory_ops_runtime.backends import DummyBackend


@pytest.fixture
def backend() -> DummyBackend:
    return DummyBackend()


@pytest.mark.logical
def test_compress_block_ratio_one_is_identity(backend: DummyBackend) -> None:
    block = backend.prefill(list(range(8)))
    compressed = backend.evict(block, ratio=1.0)
    assert compressed.positions == block.positions
    assert compressed.rope_positions == block.rope_positions
    assert compressed.kv == block.kv


@pytest.mark.logical
def test_compress_block_clamps_non_empty_block_to_one_token(backend: DummyBackend) -> None:
    block = backend.prefill(list(range(8)))
    compressed = backend.evict(block, ratio=0.01)
    assert compressed.seq_len == 1
    assert compressed.positions == [0]


@pytest.mark.logical
def test_compress_block_positions_stay_sorted_and_unique(backend: DummyBackend) -> None:
    block = backend.prefill(list(range(10)))
    compressed = backend.evict(block, ratio=0.5)
    assert compressed.seq_len == 5
    assert compressed.positions == sorted(set(compressed.positions))


@pytest.mark.logical
def test_compress_block_preserves_rope_positions(backend: DummyBackend) -> None:
    local_block = backend.prefill([4, 5, 6]).with_positions([4, 5, 6])
    compressed = backend.evict(local_block, ratio=0.5)
    assert compressed.rope_positions == [0, 2]


@pytest.mark.logical
def test_sparse_compressed_blocks_can_be_used_as_context_sequence(backend: DummyBackend) -> None:
    full = backend.prefill(list(range(14)))
    prefix = backend.slice(full, 0, 4)
    suffix = backend.slice(full, 10, 14)
    article_blocks = [
        backend.evict(backend.slice(full, 4, 8), ratio=0.5),
        backend.evict(backend.slice(full, 8, 10), ratio=1.0),
    ]
    ctx = CacheContext([prefix, *article_blocks, suffix])
    out = backend.generate(ctx, "pick", GenerationConfig())
    assert out.answer_text == "A"
    assert ctx.physical_tokens < 14
    assert ctx.next_position == 14
    assert ctx.requires_rope_correction is False
