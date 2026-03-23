# Direction

> Reference papers are in `docs/papers/`. The code lives in `src/memory_ops/`
> and `src/memory_ops_runtime/`. The authoritative experiment runner is
> `scripts/canonical_eval.py`.

## Why This Repo Exists

In document-grounded LLM workloads, prefill is expensive and highly repetitive.
The same documents get read over and over across different queries, but the
resulting KV caches are normally thrown away after each request.

In principle, those caches should be reusable artifacts. In practice, two
things get in the way:

- they are too large to store naively
- they are position-dependent, so a cache prefetched in one prompt cannot be
  dropped into another prompt unchanged

`cache-ops` exists to make those two problems programmable. The repo is not
trying to present one grand unified system. It is trying to provide a small set
of cache primitives that can be recombined into experiments on compression,
position-independent reuse, or both together.

If you want to add a new compression rule, a new reuse rule, or a new assembly
pattern, you should be able to do it by plugging one new piece into the
existing surfaces instead of building a new stack around the model.

## The Two Research Problems

### 1. Compression

Raw KV caches are far too large to store at scale. On Qwen3-4B, a single
5K-token document is already on the order of hundreds of megabytes of KV state.

This repo currently focuses on Attention Matching (AM) (docs/papers/attn_matching.pdf): 
choose a smaller key set, fit a bias term so the partition function stays close to the
full cache, then fit compacted values by least squares. The current implementation
uses the simple variant that is fast enough to run in this repo:

- HighestAttnKeys selection
- context-prefill queries
- uniform head budgets

### 2. Position-Independent Caching

If documents are prefetched independently and assembled later, two issues
appear:

- their RoPE basis is wrong for the new global position
- they never attended to neighboring chunks during prefill

RoPE misalignment is fixable by rotating keys into their target positions at
assembly time. Missing cross-chunk attention is harder. Training-free methods
that repair it usually rely on partial recomputation, which does not compose
cleanly with compression because evicted tokens are gone. That is why this repo
leans on RoPE correction as the reusable training-free primitive.

## Research Questions

### Improving AM

There are three obvious next levers:

1. OMP key selection instead of HighestAttnKeys
2. on-policy queries instead of context-prefill queries
3. non-uniform head budgets instead of uniform budgets

All three are known from the paper and all three should improve quality. The
current code intentionally leaves them as future work rather than pretending the
baseline is already the best variant.

### Adding New Primitives

The current surfaces do not cover partial recomputation after assembly. Methods
like CacheBlend or ProphetKV live in a different part of the design space:
their repair step depends on the assembled context, not just independent blocks.

If this repo starts to cover that class of method, it probably needs a new
primitive rather than one more option bolted onto the existing backend methods.

## Suggested Reading Path

1. `README.md`
2. `docs/architecture.md`
3. `src/memory_ops/`
4. `src/memory_ops_runtime/`
5. `scripts/canonical_eval.py`
6. `docs/papers/`
