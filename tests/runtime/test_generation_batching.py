import pytest
import torch

from memory_ops import CacheContext, compact_kv_cache
from memory_ops_runtime import GenerationConfig
from memory_ops_runtime.backends.hf_am import build_am_block


def _tokenize(tokenizer, text: str) -> torch.Tensor:
    return tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"]


def _assert_results_match(lhs, rhs) -> None:
    assert len(lhs) == len(rhs)
    for left, right in zip(lhs, rhs):
        assert left.answer_text == right.answer_text
        assert left.num_generated_tokens == right.num_generated_tokens
        assert left.stopped_early == right.stopped_early


@pytest.mark.integration
def test_generate_batch_empty_returns_empty(hf_backend) -> None:
    block = hf_backend.prefill(_tokenize(hf_backend.tokenizer, "Patient has fever."))
    results = hf_backend.generate_batch(CacheContext(block), [], GenerationConfig(max_new_tokens=2, do_sample=False))
    assert results == []


@pytest.mark.integration
def test_generate_batch_matches_sequential_for_same_length_prompts(hf_backend) -> None:
    block = hf_backend.prefill(
        _tokenize(hf_backend.tokenizer, "Patient has fever. Labs were stable. Plan: monitor and reassess.")
    )
    prompts = ["A or B?", "C or D?"]
    lengths = [_tokenize(hf_backend.tokenizer, prompt).shape[1] for prompt in prompts]
    assert len(set(lengths)) == 1

    config = GenerationConfig(max_new_tokens=2, do_sample=False)
    batched = hf_backend.generate_batch(CacheContext(block), prompts, config)
    sequential = [hf_backend.generate(CacheContext(block), prompt, config) for prompt in prompts]

    _assert_results_match(batched, sequential)
    assert all(result.diagnostics.get("batch_size") == 2 for result in batched)


@pytest.mark.integration
def test_generate_batch_preserves_order_across_prompt_length_buckets(hf_backend) -> None:
    block = hf_backend.prefill(
        _tokenize(hf_backend.tokenizer, "Patient has fever. Labs were stable. Plan: monitor and reassess.")
    )
    prompts = [
        "A or B?",
        "Please answer with the best option only.",
        "C or D?",
    ]
    lengths = [_tokenize(hf_backend.tokenizer, prompt).shape[1] for prompt in prompts]
    assert len(set(lengths)) >= 2

    config = GenerationConfig(max_new_tokens=2, do_sample=False)
    batched = hf_backend.generate_batch(CacheContext(block), prompts, config)
    sequential = [hf_backend.generate(CacheContext(block), prompt, config) for prompt in prompts]

    _assert_results_match(batched, sequential)


@pytest.mark.integration
def test_generate_batch_matches_sequential_for_am_context(qwen_backend) -> None:
    input_ids = _tokenize(qwen_backend.tokenizer, "Patient has fever. Labs were stable. Plan: monitor and reassess.")
    full = qwen_backend.prefill(input_ids)
    queries = qwen_backend.extract_queries(input_ids)
    compacted = compact_kv_cache(
        [(layer.keys, layer.values) for layer in full.kv.layers],
        queries,
        target_ratio=0.5,
        c2_method="direct",
    )
    positions = torch.linspace(0, full.seq_len - 1, compacted[0][0].shape[2], device=full.positions.device).long()
    am_block = build_am_block(compacted, positions)
    prompts = ["A or B?", "C or D?"]
    lengths = [_tokenize(qwen_backend.tokenizer, prompt).shape[1] for prompt in prompts]
    assert len(set(lengths)) == 1

    config = GenerationConfig(max_new_tokens=2, do_sample=False)
    batched = qwen_backend.generate_batch(CacheContext(am_block), prompts, config)
    sequential = [qwen_backend.generate(CacheContext(am_block), prompt, config) for prompt in prompts]

    _assert_results_match(batched, sequential)
    assert all(result.diagnostics.get("am_execution") is True for result in batched)


@pytest.mark.integration
def test_generate_batch_preserves_order_for_am_prompt_buckets(qwen_backend) -> None:
    input_ids = _tokenize(qwen_backend.tokenizer, "Patient has fever. Labs were stable. Plan: monitor and reassess.")
    full = qwen_backend.prefill(input_ids)
    queries = qwen_backend.extract_queries(input_ids)
    compacted = compact_kv_cache(
        [(layer.keys, layer.values) for layer in full.kv.layers],
        queries,
        target_ratio=0.5,
        c2_method="direct",
    )
    positions = torch.linspace(0, full.seq_len - 1, compacted[0][0].shape[2], device=full.positions.device).long()
    am_block = build_am_block(compacted, positions)
    prompts = [
        "A or B?",
        "Please answer with the best option only.",
        "C or D?",
    ]
    lengths = [_tokenize(qwen_backend.tokenizer, prompt).shape[1] for prompt in prompts]
    assert len(set(lengths)) >= 2

    config = GenerationConfig(max_new_tokens=2, do_sample=False)
    batched = qwen_backend.generate_batch(CacheContext(am_block), prompts, config)
    sequential = [qwen_backend.generate(CacheContext(am_block), prompt, config) for prompt in prompts]

    _assert_results_match(batched, sequential)
