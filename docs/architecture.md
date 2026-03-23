# Architecture

This document is about the code boundary: what belongs in the reusable library,
what belongs in the runtime adapter, and what should stay in scripts/examples.

## Purpose

`cache-ops` is a reference cache-ops library for frozen decoder-only transformer
inference.

Its near-term target is practical: support the current ablations and runnable
cache-assembly experiments in this repo.

Its longer-term target is structural: define cache artifacts and execution
semantics cleanly enough that future adapters can lower them into inference
engines.

It is not trying to be a production runtime, a training framework, or a
universal abstraction over every model family.

## Layers

The codebase is easiest to read in three layers.

1. Core library
   `memory_ops`: cache types (`CacheBlock`, `CacheContext`), the `CacheBackend`
   protocol, compression index math, and AM algebra.

2. Engine adapter
   `memory_ops_runtime`: HF-based backend (`HFBackend`), AM execution paths,
   and generation helpers.

3. Example workflows
   `examples/`: readable client code, datasets, prompt formatting,
   chunking policy, and demos.

The dependency direction is one-way:

- examples depend on the core library and engine adapter
- the engine adapter depends on the core library
- engine adapters should depend on the core library
- the core library should not depend on a specific experiment matrix

The readability rule is:

- `src/` ends at the reusable library and adapter
- `examples/` are clients and may duplicate small task glue
- example code should not become a shared support layer for tests or sibling examples
- examples are intentionally not part of the maintained unit-test surface

## Core Objects

### `CacheBlock`

Leaf cache artifact.

Fields:

- `kv`: backend-owned cache payload
- `positions`: canonical logical positions where the block belongs
- `rope_positions`: positions reflected by the RoPE basis inside `kv`
- `metadata`: optional provenance and debug data

Methods:

- `with_positions(...)`: change logical placement only
- `with_metadata(...)`: merge additional metadata

Meaning:

- `positions` is the target placement
- `rope_positions` is the current physical basis
- if they differ, the block is not yet globally aligned

### `CacheContext`

Ordered sequence of `CacheBlock`s.

Constructor accepts a single block, a list of blocks, or an existing context.

Properties:

- `next_position`: logical position after the last block
- `physical_tokens`: total stored KV rows across all blocks
- `requires_rope_correction`: whether any block has mismatched positions

Methods:

- `validate()`: raises if blocks overlap or have inconsistent positions

### `CacheBackend` protocol

The minimum interface a cache-ops backend must implement:

- `prefill(input_ids) -> CacheBlock`
- `slice(block, start, end) -> CacheBlock`
- `rope_correct(block) -> CacheBlock`
- `evict(block, ratio, strategy) -> CacheBlock`

Generation (`generate`, `next_token_logits`) is an additional capability on
concrete backends, not part of the core protocol.

### Qwen3 AM execution

The only maintained AM execution path in the repo today is backend-specific and
Qwen3-oriented.

The important distinction is:

- ordinary execution is the shared backend capability
- AM execution is backend-owned and not part of the shared protocol
- AM payloads (`AMPayload`) may live inside `CacheBlock.kv`

Today this is implemented via two paths:
- `memory_ops_runtime.compacted_prefix`: `StaticCompactedPrefixCache` integrates
  with HF's cache protocol and monkey-patches Qwen3 attention to inject beta.
  This is the primary path used by `HFBackend.generate`.
- `memory_ops_runtime.backends.hf_am`: `Qwen3AMExecutor` uses compiled
  FlexAttention with pre-allocated KV buffers. Kept as a reference/debug path.

### Generation types

- `GenerationConfig`
- `GenerationResult`

These belong to `memory_ops_runtime`. They are useful for validation, but
they should not become the center of the cache algebra.

## Invariants

### `CacheBlock` is always a leaf

A `CacheBlock` is produced by one artifact operation such as:

- `prefill`
- `slice`
- `with_positions`
- `rope_correct`
- `evict`

Context composition is an ordered sequence of blocks, not a merged super-block.

### Positions are explicit

- `positions` must be ordered and non-overlapping across a context
- `rope_positions` must have the same length as `positions`
- `with_positions(...)` changes logical placement only
- `rope_correct(...)` changes KV basis to match logical placement

### Ordinary execution is shared; AM execution is backend-specific

- shared backend execution consumes `Sequence[CacheBlock]`
- backend-specific AM paths may also consume `Sequence[CacheBlock]`
- the shared protocol does not define a universal AM-execution interface

## Current Core Surface

The current core public shape is:

- cache types: `CacheBlock`, `CacheContext`, `Span`
- protocol: `CacheBackend` (prefill, slice, rope_correct, evict)
- compression math: `ratio_to_count`, `select_uniform_indices`, `compression_indices`
- AM algebra: `select_keys_by_attention`, `solve_beta`, `solve_c2`, `compact_head`, `compact_kv_cache`

Current built-in core transforms are intentionally narrow:

- sparse uniform compression over ordinary KV caches
- RoPE correction for independently prefetched chunks
- no training-time methods

## Backend Boundary

Backends own:

- concrete KV payload layout
- cache slicing and materialization
- execution over cache artifacts

The core library owns:

- what a block means
- how blocks compose logically
- what it means for a block to need RoPE correction
- what metadata is attached at execution time

This is the boundary that makes future engine adapters plausible.

## Design Standard

A new abstraction belongs in the core only if it passes both tests:

1. it is required to express at least one concrete experiment
2. it can be implemented by more than one backend without encoding one engine's
   internals into the abstraction itself

If either test fails, the code should stay in an example app or a specific backend.

## What Is Deliberately Out Of Scope

The following may matter for the broader project, but they are not core-library
responsibilities for `cache-ops`:

- training-based PIC methods
- model-architecture changes
- serving schedulers
- corpus storage/indexing systems
- engine-specific kernels baked into the core abstraction

Those can plug in around the core later.

## Recommended Reading Order

1. `src/memory_ops`
2. `src/memory_ops_runtime`
3. one example app under `examples/`
