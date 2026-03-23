import os

import pytest
import torch

from memory_ops_runtime.backends import HFBackend


@pytest.fixture(scope="module")
def hf_backend():
    if HFBackend is None:
        pytest.skip("HFBackend unavailable (missing torch/transformers)")
    model_name = os.getenv("MEMORY_OPS_TEST_MODEL", "sshleifer/tiny-gpt2")
    return HFBackend(model_name=model_name, device="cuda" if os.getenv("FORCE_CUDA", "0") == "1" else None)


@pytest.fixture(scope="module")
def qwen_backend():
    if HFBackend is None:
        pytest.skip("HFBackend unavailable (missing torch/transformers)")
    model_name = os.getenv("MEMORY_OPS_MATCHED_TEST_MODEL", "Qwen/Qwen3-1.7B")
    return HFBackend(model_name=model_name, device="cuda" if os.getenv("FORCE_CUDA", "0") == "1" else None)


@pytest.fixture
def document_inputs(hf_backend) -> tuple[object, tuple[int, int], dict]:
    prefix_text = "PRE:"
    article_text = "Patient has fever. Labs were stable. Plan: monitor and reassess."
    suffix_text = ":SUF"
    prefix_ids = hf_backend.tokenizer(prefix_text, return_tensors="pt", add_special_tokens=False)["input_ids"]
    article_ids = hf_backend.tokenizer(article_text, return_tensors="pt", add_special_tokens=False)["input_ids"]
    suffix_ids = hf_backend.tokenizer(suffix_text, return_tensors="pt", add_special_tokens=False)["input_ids"]
    input_ids = torch.cat([prefix_ids, article_ids, suffix_ids], dim=1)
    article_span = (int(prefix_ids.shape[1]), int(prefix_ids.shape[1] + article_ids.shape[1]))
    metadata = {
        "article_id": "a1",
        "article_title": "note-1",
        "prefix_text": prefix_text,
        "article_text": article_text,
        "suffix_text": suffix_text,
    }
    return input_ids, article_span, metadata


def iter_kv_tensors(past_key_values):
    if hasattr(past_key_values, "layers"):
        for layer in past_key_values.layers:
            yield layer.keys, layer.values
        return
    for layer in past_key_values:
        yield layer[0], layer[1]


def assert_cache_close(lhs, rhs, atol: float = 1e-5, rtol: float = 1e-4) -> None:
    lhs_layers = list(iter_kv_tensors(lhs))
    rhs_layers = list(iter_kv_tensors(rhs))
    assert len(lhs_layers) == len(rhs_layers)
    for (l_key, l_val), (r_key, r_val) in zip(lhs_layers, rhs_layers):
        assert l_key.shape == r_key.shape
        assert l_val.shape == r_val.shape
        assert torch.allclose(l_key.float(), r_key.float(), atol=atol, rtol=rtol)
        assert torch.allclose(l_val.float(), r_val.float(), atol=atol, rtol=rtol)
