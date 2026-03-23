# cache-ops

`cache-ops` is a small cache operation library for experiments on KV cache compression
and reuse over frozen decoder-only LLMs. 
The core idea is simply: Provide a simple way where you can operate on cache and run the model on that cache.

## What It Implements

- Attention Matching-style KV compaction
- Uniform eviction as a simple compression baseline
- RoPE correction for independently-prefilled chunks
- Save/load of cache artifacts for retrieval-time reuse
- HF/Qwen3 execution paths for ordinary KV blocks and AM payloads

## Reading Order

1. `README.md`
2. `docs/architecture.md`
3. `src/memory_ops/`
4. `src/memory_ops_runtime/`
5. `scripts/canonical_eval.py`

That order gets you from repo intent to core abstractions to the runnable eval
surface.

## Usage Sketch

### Example Usage

```python
from memory_ops import CacheBlock, CacheContext, compact_kv_cache
from memory_ops_runtime.backends import HFBackend
from memory_ops_runtime.backends.hf_am import build_am_block
from memory_ops_runtime.cache_io import save_block, load_block

backend = HFBackend(model_name="Qwen/Qwen3-4B-Instruct-2507", device="cuda")

block = backend.prefill(input_ids)
queries = backend.extract_queries(input_ids)
compacted = compact_kv_cache(kv_pairs(block), queries, target_ratio=0.1)
am_block = build_am_block(compacted, positions)

save_block(am_block, "cache/article_42_r0.1")
cached = load_block("cache/article_42_r0.1", device="cuda")
result = backend.generate(CacheContext(cached), prompt, config)
```

### Evaluation

`scripts/canonical_eval.py` is the authoritative evaluation runner for the
frozen arm matrix.

```bash
cd cache-ops
uv sync --extra dev --extra backend
uv run python scripts/canonical_eval.py quality
```

### Demo

`demo/prep_corpus.py` plus `demo/app.py` is the interactive "Compress & Ask"
path. It is useful for demos and smoke testing, but it is not the canonical
research interface.

```bash
uv run python demo/prep_corpus.py --n-articles 5
uv run python demo/app.py
```

## Research Notes

- The central open question is still the residual PIC gap after RoPE correction. That missing cross-chunk attention is the harder remaining error source.

## Tests

```bash
uv run pytest -q -m logical
```
