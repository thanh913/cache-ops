"""Regression tests for generation with gapped cache positions."""
import pytest
import torch

from memory_ops import CacheContext, compact_kv_cache
from memory_ops_runtime import GenerationConfig
from memory_ops_runtime.backends.hf_am import build_am_block
from memory_ops_runtime.eval_helpers import kv_pairs


@pytest.mark.integration
@pytest.mark.gpu
def test_evicted_cache_has_position_gap(hf_backend) -> None:
    """K-uniform eviction creates physical < logical gap."""
    input_ids = hf_backend.tokenizer(
        "Patient has fever. Labs stable. Plan: monitor.",
        return_tensors="pt", add_special_tokens=False,
    )["input_ids"]
    full = hf_backend.prefill(input_ids)
    evicted = hf_backend.evict(full, ratio=0.5)
    ctx = CacheContext(evicted)
    assert ctx.physical_tokens < ctx.next_position


@pytest.mark.integration
@pytest.mark.gpu
def test_generate_with_evicted_cache(hf_backend) -> None:
    """Generation works correctly with gapped positions from eviction."""
    input_ids = hf_backend.tokenizer(
        "Patient has fever. Labs stable. Plan: monitor.",
        return_tensors="pt", add_special_tokens=False,
    )["input_ids"]
    full = hf_backend.prefill(input_ids)
    evicted = hf_backend.evict(full, ratio=0.5)
    ctx = CacheContext(evicted)
    config = GenerationConfig(max_new_tokens=8, do_sample=False)
    out = hf_backend.generate(ctx, "A or B?", config)
    assert isinstance(out.answer_text, str)
    assert out.num_generated_tokens <= 8


@pytest.mark.integration
@pytest.mark.gpu
def test_perplexity_with_evicted_cache(hf_backend) -> None:
    """Perplexity computation works with gapped positions."""
    input_ids = hf_backend.tokenizer(
        "Patient has fever. Labs stable. Plan: monitor.",
        return_tensors="pt", add_special_tokens=False,
    )["input_ids"]
    full = hf_backend.prefill(input_ids)
    config = GenerationConfig(max_new_tokens=8, do_sample=False)
    prompt = "A or B?"

    out_f = hf_backend.generate(CacheContext(full), prompt, config)
    ref_ids = hf_backend.tokenizer.encode(out_f.answer_text, add_special_tokens=False)
    if not ref_ids:
        pytest.skip("Model produced empty output")

    ppl_f = hf_backend.compute_perplexity(CacheContext(full), prompt, ref_ids)
    assert ppl_f < 2.0, f"F-arm ppl should be ~1.0, got {ppl_f:.2f}"

    evicted = hf_backend.evict(full, ratio=0.5)
    ppl_k = hf_backend.compute_perplexity(CacheContext(evicted), prompt, ref_ids)
    assert ppl_k < 1000, f"K-uniform ppl too high: {ppl_k:.2f}"


@pytest.mark.integration
@pytest.mark.gpu
def test_am_cache_has_position_gap(qwen_backend) -> None:
    """AM compaction creates physical < logical gap."""
    input_ids = qwen_backend.tokenizer(
        "Patient has fever. Labs stable. Plan: monitor.",
        return_tensors="pt", add_special_tokens=False,
    )["input_ids"]
    full = qwen_backend.prefill(input_ids)
    queries = qwen_backend.extract_queries(input_ids)
    compacted = compact_kv_cache(kv_pairs(full), queries, 0.5)
    t = compacted[0][0].shape[2]
    am_block = build_am_block(
        compacted,
        torch.linspace(0, full.seq_len - 1, t, device=full.positions.device).long(),
    )
    ctx = CacheContext(am_block)
    assert ctx.physical_tokens < ctx.next_position


@pytest.mark.integration
@pytest.mark.gpu
def test_generate_with_am_cache(qwen_backend) -> None:
    """Generation works correctly with gapped positions from AM compaction."""
    input_ids = qwen_backend.tokenizer(
        "Patient has fever. Labs stable. Plan: monitor.",
        return_tensors="pt", add_special_tokens=False,
    )["input_ids"]
    full = qwen_backend.prefill(input_ids)
    queries = qwen_backend.extract_queries(input_ids)
    compacted = compact_kv_cache(kv_pairs(full), queries, 0.5)
    t = compacted[0][0].shape[2]
    am_block = build_am_block(
        compacted,
        torch.linspace(0, full.seq_len - 1, t, device=full.positions.device).long(),
    )
    config = GenerationConfig(max_new_tokens=8, do_sample=False)
    out = qwen_backend.generate(CacheContext(am_block), "A or B?", config)
    assert isinstance(out.answer_text, str)
    assert out.num_generated_tokens <= 8
