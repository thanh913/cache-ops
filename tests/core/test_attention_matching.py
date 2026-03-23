import pytest
import torch

from memory_ops.am import (
    compact_head,
    compact_kv_cache,
    select_keys_by_attention,
    solve_beta,
    solve_c2,
)


@pytest.fixture
def kv_and_queries():
    torch.manual_seed(42)
    total_tokens, head_dim, num_queries = 32, 16, 8
    keys = torch.randn(total_tokens, head_dim)
    values = torch.randn(total_tokens, head_dim)
    queries = torch.randn(num_queries, head_dim)
    return keys, values, queries


@pytest.mark.logical
def test_select_keys_returns_correct_count(kv_and_queries) -> None:
    keys, _, queries = kv_and_queries
    compacted, indices = select_keys_by_attention(keys, queries, t=8)
    assert compacted.shape == (8, keys.shape[1])
    assert len(indices) == 8
    assert len(set(indices)) == 8


@pytest.mark.logical
def test_select_keys_full_is_identity(kv_and_queries) -> None:
    keys, _, queries = kv_and_queries
    compacted, _ = select_keys_by_attention(keys, queries, t=keys.shape[0])
    assert compacted.shape == keys.shape


@pytest.mark.logical
@pytest.mark.parametrize("method", ["rms", "max", "mean"])
def test_select_keys_score_methods(kv_and_queries, method) -> None:
    keys, _, queries = kv_and_queries
    compacted, _ = select_keys_by_attention(keys, queries, t=8, score_method=method)
    assert compacted.shape == (8, keys.shape[1])


@pytest.mark.logical
def test_solve_beta_shape_and_finite(kv_and_queries) -> None:
    keys, _, queries = kv_and_queries
    _, indices = select_keys_by_attention(keys, queries, t=8)
    beta = solve_beta(keys, queries, indices)
    assert beta.shape == (8,)
    assert beta.dtype == torch.float32
    assert torch.isfinite(beta).all()


@pytest.mark.logical
def test_solve_c2_shape_and_dtype(kv_and_queries) -> None:
    keys, values, queries = kv_and_queries
    c1, indices = select_keys_by_attention(keys, queries, t=8)
    beta = solve_beta(keys, queries, indices)
    c2 = solve_c2(c1, beta, keys, values, queries)
    assert c2.shape == (8, keys.shape[1])
    assert c2.dtype == keys.dtype


@pytest.mark.logical
def test_compact_head_full_pipeline(kv_and_queries) -> None:
    keys, values, queries = kv_and_queries
    c1, beta, c2, indices = compact_head(keys, values, queries, target_size=8)
    assert c1.shape == (8, keys.shape[1])
    assert beta.shape == (8,)
    assert c2.shape == (8, keys.shape[1])
    assert len(indices) == 8


@pytest.mark.logical
def test_compact_head_identity_when_t_equals_total_tokens(kv_and_queries) -> None:
    keys, values, queries = kv_and_queries
    c1, beta, c2, indices = compact_head(keys, values, queries, target_size=keys.shape[0])
    assert c1.shape == keys.shape
    assert (beta == 0).all()
    assert torch.equal(c2, values)
    assert indices == list(range(keys.shape[0]))


@pytest.mark.logical
def test_compact_head_direct_c2(kv_and_queries) -> None:
    keys, values, queries = kv_and_queries
    _, _, c2, indices = compact_head(keys, values, queries, target_size=8, c2_method="direct")
    index_tensor = torch.tensor(indices, dtype=torch.long)
    assert torch.equal(c2, values[index_tensor])


@pytest.mark.logical
def test_compact_head_approximation_quality(kv_and_queries) -> None:
    keys, values, queries = kv_and_queries
    inv_sqrt_d = keys.shape[1] ** -0.5
    c1, beta, c2, _ = compact_head(keys, values, queries, target_size=16)

    original = torch.softmax((queries @ keys.T).float() * inv_sqrt_d, dim=1) @ values.float()
    compacted = torch.softmax((queries @ c1.T).float() * inv_sqrt_d + beta.float(), dim=1) @ c2.float()
    cosine_similarity = torch.nn.functional.cosine_similarity(original, compacted, dim=1)
    assert cosine_similarity.mean() > 0.9


@pytest.mark.logical
def test_compact_kv_cache_shapes() -> None:
    torch.manual_seed(42)
    past_key_values = [
        (torch.randn(1, 2, 32, 16), torch.randn(1, 2, 32, 16))
        for _ in range(4)
    ]
    queries = torch.randn(4, 2, 8, 16)
    result = compact_kv_cache(past_key_values, queries, target_ratio=0.5)
    assert len(result) == 4
    for c1, beta, c2 in result:
        assert c1.shape == (1, 2, 16, 16)
        assert beta.shape == (2, 16)
        assert c2.shape == (1, 2, 16, 16)


@pytest.mark.logical
def test_compact_kv_cache_ratio_one_is_identity() -> None:
    torch.manual_seed(42)
    past_key_values = [
        (torch.randn(1, 2, 16, 8), torch.randn(1, 2, 16, 8))
        for _ in range(2)
    ]
    queries = torch.randn(2, 2, 4, 8)
    result = compact_kv_cache(past_key_values, queries, target_ratio=1.0)
    for c1, beta, _ in result:
        assert c1.shape[2] == 16
        assert (beta == 0).all()


@pytest.mark.logical
@pytest.mark.parametrize("ratio", [0.0, -0.5, 1.5])
def test_compact_kv_cache_rejects_invalid_ratios(ratio: float) -> None:
    torch.manual_seed(42)
    past_key_values = [(torch.randn(1, 2, 8, 4), torch.randn(1, 2, 8, 4))]
    queries = torch.randn(1, 2, 4, 4)
    with pytest.raises(ValueError, match="target_ratio"):
        compact_kv_cache(past_key_values, queries, target_ratio=ratio)
