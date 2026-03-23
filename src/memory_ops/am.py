"""Attention Matching compaction over frozen KV caches.

AM builds a compacted cache `(C1, beta, C2)` that tries to preserve the full
attention computation under a fixed set of training queries. In this file the
pipeline is: choose compacted keys, fit beta so the partition function matches,
then fit compacted values against the induced attention output.
"""
from __future__ import annotations

import math

import torch

from memory_ops.compression import ratio_to_count

# Default bounds on B where beta = log(B).
BETA_LOWER_DEFAULT = math.exp(-3)   # ~0.0498
BETA_UPPER_DEFAULT = math.exp(7)    # ~1096.6  (paper uses exp(7) for HighestAttnKeys)

def select_keys_by_attention(
    K: torch.Tensor,
    queries: torch.Tensor,
    t: int,
    *,
    score_method: str = "rms",
    query_chunk_size: int = 4096,
) -> tuple[torch.Tensor, list[int]]:
    """Return the top-t keys under aggregated attention scores."""
    n, d = queries.shape
    T = K.shape[0]
    inv_sqrt_d = d ** -0.5

    # Accumulate key scores across query chunks to avoid (n, T) OOM
    if score_method == "rms":
        accum = torch.zeros(T, device=K.device, dtype=torch.float32)
    elif score_method == "max":
        accum = torch.full((T,), -float("inf"), device=K.device, dtype=torch.float32)
    else:
        accum = torch.zeros(T, device=K.device, dtype=torch.float32)

    for i in range(0, n, query_chunk_size):
        q_chunk = queries[i : i + query_chunk_size]
        scores = (q_chunk @ K.T).float() * inv_sqrt_d
        scores -= scores.max(dim=1, keepdim=True).values
        attn = torch.softmax(scores, dim=1)

        if score_method == "rms":
            accum += (attn ** 2).sum(dim=0)
        elif score_method == "max":
            accum = torch.max(accum, attn.max(dim=0).values)
        else:
            accum += attn.sum(dim=0)

    if score_method == "rms":
        key_scores = (accum / n).sqrt()
    elif score_method == "max":
        key_scores = accum
    else:
        key_scores = accum / n

    top = torch.topk(key_scores, t, largest=True)
    idx = top.indices
    return K[idx], idx.cpu().tolist()


def select_keys_by_omp(
    K: torch.Tensor,
    queries: torch.Tensor,
    t: int,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Greedily choose keys that reconstruct the partition function.

    OMP works against `sum_j exp(q·k_j)` rather than final attention outputs.
    At each step it picks the unselected key that best explains the current
    residual and then re-fits the nonnegative weights over the selected set.
    """
    n, d = queries.shape
    T = K.shape[0]
    device = K.device

    # Materialize exp(q·k / sqrt(d)) for the NNLS target.
    inv_sqrt_d = d ** -0.5
    scores = (queries @ K.T).float() * inv_sqrt_d       # (n, T)
    scores -= scores.max(dim=1, keepdim=True).values     # numerical stability
    exp_scores = torch.exp(scores)                        # (n, T)

    target = exp_scores.sum(dim=1)                        # (n,)
    selected_indices: list[int] = []
    mask = torch.zeros(T, dtype=torch.bool, device=device)
    current = torch.zeros_like(target)
    B = torch.zeros(0, device=device, dtype=torch.float32)

    for _ in range(t):
        # Pick the unselected key most aligned with the residual.
        residual = target - current                       # (n,)
        corr = (exp_scores * residual.unsqueeze(1)).sum(dim=0)  # (T,)
        corr[mask] = -float("inf")

        idx = corr.argmax().item()
        selected_indices.append(idx)
        mask[idx] = True

        # Re-fit nonnegative weights over the selected keys.
        M = exp_scores[:, selected_indices]               # (n, i+1)
        try:
            B = torch.linalg.lstsq(M, target.unsqueeze(1)).solution.squeeze(1)
            if torch.isnan(B).any():
                raise RuntimeError("NaN in lstsq")
        except Exception:
            # Fall back to a tiny ridge solve when lstsq is unstable.
            i = len(selected_indices)
            MtM = M.T @ M
            MtM = 0.5 * (MtM + MtM.T)
            MtM.diagonal().add_(1e-6)
            B = torch.cholesky_solve(
                (M.T @ target).unsqueeze(1),
                torch.linalg.cholesky(MtM),
            ).squeeze(1)
        B = B.clamp(min=1e-12)
        current = M @ B

    C1 = K[torch.tensor(selected_indices, device=device)]
    beta = torch.log(B)
    return C1, beta, selected_indices

def solve_beta(
    K: torch.Tensor,
    queries: torch.Tensor,
    indices: list[int],
    *,
    lower_bound: float = BETA_LOWER_DEFAULT,
    upper_bound: float = BETA_UPPER_DEFAULT,
    pgd_iters: int = 2,
) -> torch.Tensor:
    """Fit beta so the compacted partition function matches the full one.

    This does not try to match values or logits directly. It only matches the
    normalization term induced by the selected keys, which is why beta can be
    solved once C1 is fixed.
    """
    d = K.shape[1]
    inv_sqrt_d = d ** -0.5

    scores = (queries @ K.T).float() * inv_sqrt_d           # (n, T)
    scores -= scores.max(dim=1, keepdim=True).values
    exp_scores = torch.exp(scores)                           # (n, T)

    target = exp_scores.sum(dim=1)                           # (n,)
    idx_t = torch.tensor(indices, device=K.device, dtype=torch.long)
    M = exp_scores[:, idx_t]                                 # (n, t)

    try:
        B = torch.linalg.lstsq(M, target.unsqueeze(1), driver="gels").solution.squeeze(1)
        if torch.isnan(B).any():
            raise RuntimeError("NaN in lstsq")
    except Exception:
        # Fall back to a tiny ridge solve when lstsq is unstable.
        n, t = M.shape
        lam = 1e-6
        if n < t:
            MMt = M @ M.T
            MMt = 0.5 * (MMt + MMt.T)
            MMt.diagonal().add_(lam)
            alpha = torch.cholesky_solve(target.unsqueeze(1), torch.linalg.cholesky(MMt)).squeeze(1)
            B = M.T @ alpha
        else:
            MtM = M.T @ M
            MtM = 0.5 * (MtM + MtM.T)
            MtM.diagonal().add_(lam)
            B = torch.cholesky_solve((M.T @ target).unsqueeze(1), torch.linalg.cholesky(MtM)).squeeze(1)

    B = B.clamp(min=lower_bound)
    if upper_bound is not None:
        B = B.clamp(max=upper_bound)

    if pgd_iters > 0:
        # Estimate a stable PGD step size from the spectral norm.
        u = torch.randn(M.shape[1], device=M.device, dtype=M.dtype)
        for _ in range(10):
            v = M @ u
            v = v / (v.norm() + 1e-12)
            u = M.T @ v
            u = u / (u.norm() + 1e-12)
        L = (M @ u).norm().item()
        eta = 1.0 / max(L * L, 1e-12)

        for _ in range(pgd_iters):
            grad = M.T @ (M @ B - target)
            B = B - eta * grad
            B = B.clamp(min=lower_bound)
            if upper_bound is not None:
                B = B.clamp(max=upper_bound)

    return torch.log(B)

def solve_c2(
    C1: torch.Tensor,
    beta: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    queries: torch.Tensor,
    *,
    ridge_lambda: float = 0,
) -> torch.Tensor:
    """Fit compacted values against the full attention output.

    Once `C1` and `beta` are fixed, the compacted attention weights are fixed
    too. `C2` is then the least-squares regression target that makes those
    weights reproduce the full-cache attention output as closely as possible.
    """
    dtype = K.dtype
    d = K.shape[1]
    inv_sqrt_d = d ** -0.5

    # Target attention output under the full cache.
    sK = (queries @ K.T).float() * inv_sqrt_d
    Y = torch.softmax(sK, dim=1) @ V.float()                # (n, d)

    # Design matrix induced by the compacted cache.
    sC = (queries @ C1.T).float() * inv_sqrt_d + beta.float()
    X = torch.softmax(sC, dim=1)                             # (n, t)

    n, t = X.shape

    if ridge_lambda > 0:
        try:
            lam = ridge_lambda * (torch.linalg.matrix_norm(X, ord=2) ** 2)
        except Exception:
            lam = ridge_lambda * ((torch.linalg.matrix_norm(X, ord="fro") ** 2) / t)
    else:
        lam = 0

    if lam == 0:
        try:
            C2 = torch.linalg.lstsq(X, Y, driver="gels").solution
            if torch.isnan(C2).any():
                raise RuntimeError("NaN in lstsq")
        except Exception:
            # Fall back to a tiny ridge solve when lstsq is unstable.
            lam = 1e-6
            if n < t:
                XXt = X @ X.T
                XXt = 0.5 * (XXt + XXt.T)
                XXt.diagonal().add_(lam)
                Z = torch.cholesky_solve(Y, torch.linalg.cholesky(XXt))
                C2 = X.T @ Z
            else:
                XtX = X.T @ X
                XtX = 0.5 * (XtX + XtX.T)
                XtX.diagonal().add_(lam)
                C2 = torch.cholesky_solve(X.T @ Y, torch.linalg.cholesky(XtX))
    else:
        if n < t:
            XXt = X @ X.T
            XXt = 0.5 * (XXt + XXt.T)
            XXt.diagonal().add_(lam)
            Z = torch.cholesky_solve(Y, torch.linalg.cholesky(XXt))
            C2 = X.T @ Z
        else:
            XtX = X.T @ X
            XtX = 0.5 * (XtX + XtX.T)
            XtX.diagonal().add_(lam)
            C2 = torch.cholesky_solve(X.T @ Y, torch.linalg.cholesky(XtX))

    return C2.to(dtype)

def compact_head(
    K: torch.Tensor,
    V: torch.Tensor,
    queries: torch.Tensor,
    target_size: int,
    *,
    key_selection: str = "highest_attention",
    score_method: str = "rms",
    c2_method: str = "lsq",
    beta_lower_bound: float = BETA_LOWER_DEFAULT,
    beta_upper_bound: float | None = BETA_UPPER_DEFAULT,
    pgd_iters: int = 2,
    c2_ridge_lambda: float = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
    """Run the AM pipeline for one layer/head slice."""
    T = K.shape[0]
    t = min(target_size, T)
    if t == T:
        return K, torch.zeros(T, device=K.device, dtype=torch.float32), V, list(range(T))

    if key_selection == "omp":
        C1, beta, indices = select_keys_by_omp(K, queries, t)
    else:
        C1, indices = select_keys_by_attention(K, queries, t, score_method=score_method)
        beta = solve_beta(
            K, queries, indices,
            lower_bound=beta_lower_bound,
            upper_bound=beta_upper_bound,
            pgd_iters=pgd_iters,
        )

    if c2_method == "direct":
        idx_t = torch.tensor(indices, device=V.device, dtype=torch.long)
        C2 = V[idx_t]
    else:
        C2 = solve_c2(C1, beta, K, V, queries, ridge_lambda=c2_ridge_lambda)

    return C1, beta, C2, indices

def compact_kv_cache(
    past_key_values: list[tuple[torch.Tensor, torch.Tensor]],
    queries: torch.Tensor,
    target_ratio: float,
    *,
    key_selection: str = "highest_attention",
    score_method: str = "rms",
    c2_method: str = "lsq",
    beta_lower_bound: float = BETA_LOWER_DEFAULT,
    beta_upper_bound: float | None = BETA_UPPER_DEFAULT,
    pgd_iters: int = 2,
    c2_ridge_lambda: float = 0,
    head_budgets: dict[tuple[int, int], int] | None = None,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Run AM compaction across every layer and KV head."""
    layers = list(past_key_values)
    num_layers = len(layers)
    result = []

    for layer_idx in range(num_layers):
        keys, values = layers[layer_idx]  # (1, H, T, d)
        num_heads = keys.shape[1]
        T = keys.shape[2]
        d = keys.shape[3]
        t_uniform = ratio_to_count(T, target_ratio)

        c1_heads = []
        beta_heads = []
        c2_heads = []

        for head_idx in range(num_heads):
            if head_budgets is not None and (layer_idx, head_idx) in head_budgets:
                t = max(1, min(T, head_budgets[(layer_idx, head_idx)]))
            else:
                t = t_uniform

            K_h = keys[0, head_idx]    # (T, d)
            V_h = values[0, head_idx]  # (T, d)
            Q_h = queries[layer_idx, head_idx]  # (n, d)

            C1_h, beta_h, C2_h, _ = compact_head(
                K_h, V_h, Q_h, t,
                key_selection=key_selection,
                score_method=score_method,
                c2_method=c2_method,
                beta_lower_bound=beta_lower_bound,
                beta_upper_bound=beta_upper_bound,
                pgd_iters=pgd_iters,
                c2_ridge_lambda=c2_ridge_lambda,
            )
            c1_heads.append(C1_h)
            beta_heads.append(beta_h)
            c2_heads.append(C2_h)

        # Pad to the layer-local max so the layer tensors can stack.
        t_max = max(c.shape[0] for c in c1_heads)
        for i in range(num_heads):
            t_i = c1_heads[i].shape[0]
            if t_i < t_max:
                pad_k = torch.zeros(t_max - t_i, d, device=keys.device, dtype=keys.dtype)
                pad_v = torch.zeros(t_max - t_i, d, device=values.device, dtype=values.dtype)
                pad_b = torch.zeros(t_max - t_i, device=beta_heads[i].device, dtype=beta_heads[i].dtype)
                c1_heads[i] = torch.cat([c1_heads[i], pad_k])
                c2_heads[i] = torch.cat([c2_heads[i], pad_v])
                beta_heads[i] = torch.cat([beta_heads[i], pad_b])

        C1_layer = torch.stack(c1_heads).unsqueeze(0)  # (1, H, t_max, d)
        beta_layer = torch.stack(beta_heads)  # (H, t_max)
        C2_layer = torch.stack(c2_heads).unsqueeze(0)
        result.append((C1_layer, beta_layer, C2_layer))

    return result
