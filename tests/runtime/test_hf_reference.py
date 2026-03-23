import pytest
import torch

from memory_ops import CacheContext
from memory_ops_runtime import GenerationConfig

from conftest import assert_cache_close


def _token_count(input_ids) -> int:
    return int(input_ids.shape[-1])


def _tokenize_document(tokenizer, *, prefix_text: str = "", article_text: str, suffix_text: str = ""):
    prefix_ids = tokenizer(prefix_text, return_tensors="pt", add_special_tokens=False)["input_ids"]
    article_ids = tokenizer(article_text, return_tensors="pt", add_special_tokens=False)["input_ids"]
    suffix_ids = tokenizer(suffix_text, return_tensors="pt", add_special_tokens=False)["input_ids"]
    input_ids = torch.cat([prefix_ids, article_ids, suffix_ids], dim=1)
    article_span = (int(prefix_ids.shape[1]), int(prefix_ids.shape[1] + article_ids.shape[1]))
    return input_ids, article_span


def _scaffold_spans(article_span, total_tokens):
    return (0, article_span[0]), (article_span[1], total_tokens)


def _partition_fixed(article_span, chunk_size):
    chunks = []
    cursor = article_span[0]
    while cursor < article_span[1]:
        end = min(article_span[1], cursor + chunk_size)
        chunks.append((cursor, end))
        cursor = end
    return chunks


@pytest.mark.integration
@pytest.mark.gpu
def test_hf_prefill_and_generate(hf_backend, document_inputs) -> None:
    input_ids, _, _ = document_inputs
    block = hf_backend.prefill(input_ids)
    out = hf_backend.generate(CacheContext(block), "Which option is correct? A, B, or C?", GenerationConfig(max_new_tokens=8, do_sample=False))
    assert block.seq_len == _token_count(input_ids)
    assert isinstance(out.answer_text, str)
    assert out.num_generated_tokens <= 8


@pytest.mark.integration
@pytest.mark.gpu
def test_hf_slice_block_matches_full_span(hf_backend, document_inputs) -> None:
    input_ids, article_span, _ = document_inputs
    full = hf_backend.prefill(input_ids)
    article = hf_backend.slice(full, article_span[0], article_span[1])
    expected_positions = torch.arange(article_span[0], article_span[1], device=article.positions.device)
    assert torch.equal(article.positions, expected_positions)
    assert_cache_close(article.kv, hf_backend.slice(full, article_span[0], article_span[1]).kv)


@pytest.mark.integration
@pytest.mark.gpu
def test_hf_slice_context_roundtrip_matches_full(hf_backend, document_inputs) -> None:
    input_ids, article_span, _ = document_inputs
    prefix_span, suffix_span = _scaffold_spans(article_span, _token_count(input_ids))
    full = hf_backend.prefill(input_ids)
    prefix = hf_backend.slice(full, prefix_span[0], prefix_span[1])
    article = hf_backend.slice(full, article_span[0], article_span[1])
    suffix = hf_backend.slice(full, suffix_span[0], suffix_span[1])
    prompt = "Which option is correct? A, B, or C?"
    full_logits = hf_backend.next_token_logits(CacheContext(full), prompt)
    sliced_logits = hf_backend.next_token_logits(CacheContext([prefix, article, suffix]), prompt)
    assert torch.allclose(full_logits, sliced_logits, atol=1e-5, rtol=1e-4)


@pytest.mark.integration
@pytest.mark.gpu
def test_hf_identity_pic_rope1_single_chunk_matches_baseline(hf_backend) -> None:
    input_ids, article_span = _tokenize_document(
        hf_backend.tokenizer,
        article_text="Patient has fever. Labs were stable. Plan: monitor and reassess.",
    )
    full = hf_backend.prefill(input_ids)
    whole_chunk = _partition_fixed(article_span, chunk_size=article_span[1] - article_span[0])[0]
    assembled = hf_backend.rope_correct(
        hf_backend.prefill(input_ids[:, whole_chunk[0] : whole_chunk[1]]).with_positions(
            torch.arange(whole_chunk[0], whole_chunk[1], device=full.positions.device, dtype=torch.long)
        )
    )
    assert_cache_close(assembled.kv, full.kv)
    prompt = "Which option is correct? A, B, or C?"
    full_logits = hf_backend.next_token_logits(CacheContext(full), prompt)
    pic_logits = hf_backend.next_token_logits(CacheContext(assembled), prompt)
    assert torch.allclose(full_logits, pic_logits, atol=1e-5, rtol=1e-4)


@pytest.mark.integration
@pytest.mark.gpu
def test_hf_identity_pic_alignment_flags(hf_backend, document_inputs) -> None:
    input_ids, article_span, _ = document_inputs
    prefix_span, suffix_span = _scaffold_spans(article_span, _token_count(input_ids))
    full = hf_backend.prefill(input_ids)
    chunks = _partition_fixed(article_span, chunk_size=max(1, (article_span[1] - article_span[0]) // 2))
    rope0_blocks = [
        hf_backend.prefill(input_ids[:, chunk[0] : chunk[1]]).with_positions(
            torch.arange(chunk[0], chunk[1], device=full.positions.device, dtype=torch.long)
        )
        for chunk in chunks
    ]
    rope1_blocks = [hf_backend.rope_correct(block) for block in rope0_blocks]
    prefix = hf_backend.slice(full, prefix_span[0], prefix_span[1])
    suffix = hf_backend.slice(full, suffix_span[0], suffix_span[1])

    assert CacheContext([prefix, *rope0_blocks, suffix]).requires_rope_correction is True
    assert CacheContext([prefix, *rope1_blocks, suffix]).requires_rope_correction is False

    prompt = "Which option is correct? A, B, or C?"
    outs = [
        hf_backend.generate(CacheContext([prefix, *rope0_blocks, suffix]), prompt, GenerationConfig(max_new_tokens=6, do_sample=False)),
        hf_backend.generate(CacheContext([prefix, *rope1_blocks, suffix]), prompt, GenerationConfig(max_new_tokens=6, do_sample=False)),
    ]
    assert all(isinstance(out.answer_text, str) for out in outs)


@pytest.mark.integration
@pytest.mark.gpu
def test_hf_compression_only_ratio_one_matches_full_path(hf_backend, document_inputs) -> None:
    input_ids, article_span, _ = document_inputs
    prefix_span, suffix_span = _scaffold_spans(article_span, _token_count(input_ids))
    full = hf_backend.prefill(input_ids)
    prefix = hf_backend.slice(full, prefix_span[0], prefix_span[1])
    article = hf_backend.slice(full, article_span[0], article_span[1])
    suffix = hf_backend.slice(full, suffix_span[0], suffix_span[1])
    compressed_article = hf_backend.evict(article, ratio=1.0)
    prompt = "Which option is correct? A, B, or C?"
    full_logits = hf_backend.next_token_logits(CacheContext(full), prompt)
    compressed_logits = hf_backend.next_token_logits(CacheContext([prefix, compressed_article, suffix]), prompt)
    assert torch.allclose(full_logits, compressed_logits, atol=1e-5, rtol=1e-4)


@pytest.mark.integration
@pytest.mark.gpu
def test_hf_pic_compression_ratio_one_matches_identity_pic_paths(hf_backend, document_inputs) -> None:
    input_ids, article_span, _ = document_inputs
    prefix_span, suffix_span = _scaffold_spans(article_span, _token_count(input_ids))
    full = hf_backend.prefill(input_ids)
    prefix = hf_backend.slice(full, prefix_span[0], prefix_span[1])
    suffix = hf_backend.slice(full, suffix_span[0], suffix_span[1])
    chunks = _partition_fixed(article_span, chunk_size=max(1, (article_span[1] - article_span[0]) // 2))

    rope0_blocks = [
        hf_backend.evict(
            hf_backend.prefill(input_ids[:, chunk[0] : chunk[1]]).with_positions(
                torch.arange(chunk[0], chunk[1], device=full.positions.device, dtype=torch.long)
            ),
            ratio=1.0,
        )
        for chunk in chunks
    ]
    rope1_blocks = [hf_backend.evict(hf_backend.rope_correct(block), ratio=1.0) for block in rope0_blocks]
    rope0_expected = [
        hf_backend.prefill(input_ids[:, chunk[0] : chunk[1]]).with_positions(
            torch.arange(chunk[0], chunk[1], device=full.positions.device, dtype=torch.long)
        )
        for chunk in chunks
    ]
    rope1_expected = [hf_backend.rope_correct(block) for block in rope0_expected]

    assert CacheContext([prefix, *rope0_blocks, suffix]).requires_rope_correction is True
    assert CacheContext([prefix, *rope1_blocks, suffix]).requires_rope_correction is False
    for actual, expected in zip(rope0_blocks, rope0_expected):
        assert_cache_close(actual.kv, expected.kv)
    for actual, expected in zip(rope1_blocks, rope1_expected):
        assert_cache_close(actual.kv, expected.kv)


@pytest.mark.integration
@pytest.mark.gpu
def test_hf_sparse_compressed_context_generates(hf_backend, document_inputs) -> None:
    input_ids, article_span, _ = document_inputs
    prefix_span, suffix_span = _scaffold_spans(article_span, _token_count(input_ids))
    full = hf_backend.prefill(input_ids)
    prefix = hf_backend.slice(full, prefix_span[0], prefix_span[1])
    article = hf_backend.slice(full, article_span[0], article_span[1])
    suffix = hf_backend.slice(full, suffix_span[0], suffix_span[1])
    compressed_article = hf_backend.evict(article, ratio=0.5)
    ctx = CacheContext([prefix, compressed_article, suffix])
    out = hf_backend.generate(ctx, "Which option is correct? A, B, or C?", GenerationConfig(max_new_tokens=8, do_sample=False))
    assert prefix.seq_len + compressed_article.seq_len + suffix.seq_len < full.seq_len
    assert isinstance(out.answer_text, str)
    assert out.num_generated_tokens <= 8
