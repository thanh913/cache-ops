"""Example-oriented baseline workflows for report-style experiments."""
from __future__ import annotations

from dataclasses import dataclass
import re
import time
import warnings
from typing import Any, Mapping, Sequence

import torch

from memory_ops import CacheBlock, CacheContext, Span
from memory_ops_runtime import GenerationConfig
from memory_ops_runtime.backends import HFBackend
from memory_ops_runtime.eval_helpers import format_question, parse_choice


@dataclass(frozen=True)
class BaselineSpec:
    arm: str
    label: str
    prefill: str
    rope_correct: bool
    compression_strategy: str | None
    target_ratio: float


_ARM_PARAMS: dict[str, tuple[str, bool]] = {
    "F": ("full", False),
    "K": ("full", False),
    "P1": ("independent", True),
    "P0": ("independent", False),
}


def token_count(input_ids: Any) -> int:
    if hasattr(input_ids, "shape"):
        return int(input_ids.shape[-1])
    return len(input_ids)


def _tokenize_text(tokenizer: Any, text: str) -> Any:
    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    return encoded["input_ids"]


def _concat_token_ids(parts: list[Any]) -> Any:
    if not parts:
        return []
    first = parts[0]
    if hasattr(first, "shape"):
        return torch.cat(parts, dim=1)
    merged: list[int] = []
    for part in parts:
        merged.extend(part)
    return merged


def tokenize_document(
    tokenizer: Any,
    *,
    article_text: str,
    article_id: str,
    article_title: str | None = None,
    prefix_text: str = "",
    suffix_text: str = "",
    metadata: Mapping[str, Any] | None = None,
) -> tuple[Any, Span, dict[str, Any]]:
    prefix_ids = _tokenize_text(tokenizer, prefix_text)
    article_ids = _tokenize_text(tokenizer, article_text)
    suffix_ids = _tokenize_text(tokenizer, suffix_text)
    input_ids = _concat_token_ids([prefix_ids, article_ids, suffix_ids])

    prefix_len = token_count(prefix_ids)
    article_len = token_count(article_ids)
    merged_metadata = {
        "article_id": article_id,
        "article_title": article_title,
        "prefix_text": prefix_text,
        "article_text": article_text,
        "suffix_text": suffix_text,
        **(dict(metadata) if metadata is not None else {}),
    }
    return input_ids, (prefix_len, prefix_len + article_len), merged_metadata


def plan_fixed_chunks(article_span: Span, *, chunk_size: int) -> list[Span]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    article_start, article_end = article_span
    chunks: list[Span] = []
    cursor = article_start
    while cursor < article_end:
        end = min(article_end, cursor + chunk_size)
        chunks.append((cursor, end))
        cursor = end
    return chunks


def plan_longhealth_fine(article_span: Span, article_text: str, tokenizer: Any) -> list[Span]:
    article_start, article_end = article_span
    if not article_text:
        return [article_span]

    pattern = re.compile(r"<[^>\n]+>\n.*?\n</[^>\n]+>", re.DOTALL)
    matches = list(pattern.finditer(article_text))
    if not matches:
        return [article_span]

    chunks: list[Span] = []
    token_cursor = article_start
    for idx, match in enumerate(matches):
        text_start = match.start()
        text_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(article_text)
        chunk_text = article_text[text_start:text_end]
        chunk_tokens = len(tokenizer(chunk_text, add_special_tokens=False)["input_ids"])
        next_cursor = min(article_end, token_cursor + chunk_tokens)
        chunks.append((token_cursor, next_cursor))
        token_cursor = next_cursor

    if token_cursor != article_end:
        warnings.warn(
            "plan_longhealth_fine token mismatch; falling back to a single article chunk",
            stacklevel=2,
        )
        return [article_span]
    return chunks


def plan_chunks(
    article_span: Span,
    *,
    article_text: str,
    tokenizer: Any,
    strategy: str,
    chunk_size: int = 1024,
) -> list[Span]:
    strategy_lower = strategy.lower()
    if strategy_lower == "fixed":
        return plan_fixed_chunks(article_span, chunk_size=chunk_size)
    if strategy_lower == "longhealth_fine":
        return plan_longhealth_fine(article_span, article_text, tokenizer)
    raise ValueError(f"unsupported chunking strategy: {strategy}")


def scaffold_context(
    backend: HFBackend,
    full_block: CacheBlock,
    article_span: Span,
    article_blocks: list[CacheBlock],
    total_tokens: int,
) -> CacheContext:
    blocks: list[CacheBlock] = []
    if article_span[0] > 0:
        blocks.append(backend.slice(full_block, 0, article_span[0]))
    blocks.extend(article_blocks)
    if article_span[1] < total_tokens:
        blocks.append(backend.slice(full_block, article_span[1], total_tokens))
    return CacheContext(blocks)


def build_full_context(backend: HFBackend, input_ids: Any) -> CacheContext:
    """Arm F: full prefill, no chunking or compression."""
    return CacheContext(backend.prefill(input_ids))


def build_kv_chunked_context(
    backend: HFBackend,
    input_ids: Any,
    article_span: Span,
    chunks: list[Span],
    ratio: float,
    strategy: str,
) -> CacheContext:
    """Arm K: full prefill, slice article into chunks, compress each."""
    full = backend.prefill(input_ids)
    article_blocks = [
        backend.evict(backend.slice(full, s, e), ratio=ratio, strategy=strategy)
        for s, e in chunks
    ]
    return scaffold_context(backend, full, article_span, article_blocks, token_count(input_ids))


def build_pic_context(
    backend: HFBackend,
    input_ids: Any,
    article_span: Span,
    chunks: list[Span],
    rope_correct: bool,
    ratio: float | None = None,
    strategy: str | None = None,
) -> CacheContext:
    """Build a P0 or P1 context from independently prefetched chunks."""
    full = backend.prefill(input_ids)
    device = full.positions.device
    article_blocks = [
        backend.prefill(input_ids[:, s:e]).with_positions(
            torch.arange(s, e, device=device, dtype=torch.long)
        )
        for s, e in chunks
    ]
    if rope_correct:
        article_blocks = [backend.rope_correct(b) for b in article_blocks]
    if strategy is not None and ratio is not None:
        article_blocks = [backend.evict(b, ratio=ratio, strategy=strategy) for b in article_blocks]
    return scaffold_context(backend, full, article_span, article_blocks, token_count(input_ids))


def build_context_for_spec(
    backend: HFBackend,
    input_ids: Any,
    article_span: Span,
    chunks: list[Span],
    spec: BaselineSpec,
) -> CacheContext:
    """Dispatch to the right per-arm builder based on spec."""
    if spec.prefill == "full" and spec.compression_strategy is None:
        return build_full_context(backend, input_ids)
    if spec.prefill == "full":
        return build_kv_chunked_context(
            backend, input_ids, article_span, chunks,
            ratio=spec.target_ratio, strategy=spec.compression_strategy,
        )
    return build_pic_context(
        backend, input_ids, article_span, chunks,
        rope_correct=spec.rope_correct,
        ratio=spec.target_ratio if spec.compression_strategy else None,
        strategy=spec.compression_strategy,
    )


def choice_parse_summary(answer_texts: Sequence[str], *, num_choices: int) -> dict[str, float]:
    total = len(answer_texts)
    parsed = sum(1 for text in answer_texts if parse_choice(text, num_choices) is not None)
    return {"parse_rate": (parsed / total) if total else 0.0}


def default_chunking(dataset_name: str) -> str:
    if dataset_name.startswith("longhealth"):
        return "longhealth_fine"
    return "fixed"


def normalize_compression_strategy(strategy: str | None) -> str | None:
    if strategy is None:
        return None
    lowered = strategy.lower()
    if lowered == "none":
        return None
    return lowered


def resolve_specs(
    *,
    arms: Sequence[str],
    compression_strategy: str | None,
    target_ratio: float,
) -> list[BaselineSpec]:
    specs: list[BaselineSpec] = []
    for arm in arms:
        prefill, rope_correct = _ARM_PARAMS[arm]
        if arm == "F":
            specs.append(
                BaselineSpec(
                    arm=arm,
                    label="F",
                    prefill=prefill,
                    rope_correct=rope_correct,
                    compression_strategy=None,
                    target_ratio=1.0,
                )
            )
            continue
        if compression_strategy is None:
            raise ValueError(f"arm {arm!r} requires a compression strategy; use 'F' for no compression")
        specs.append(
            BaselineSpec(
                arm=arm,
                label=f"{arm}_{compression_strategy}_{target_ratio:.2f}",
                prefill=prefill,
                rope_correct=rope_correct,
                compression_strategy=compression_strategy,
                target_ratio=target_ratio,
            )
        )
    return specs


def execute_prompt(
    *,
    backend: HFBackend,
    ctx: CacheContext,
    prompt_text: str,
    config: GenerationConfig,
) -> tuple[Any, float]:
    start = time.perf_counter()
    output = backend.generate(ctx, prompt_text, config)
    return output, time.perf_counter() - start
