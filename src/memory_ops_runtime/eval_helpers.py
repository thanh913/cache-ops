"""Shared evaluation helpers for scripts and examples."""
from __future__ import annotations

import re
from typing import Any, Sequence

import torch

from memory_ops import CacheContext, Span

def format_question(tokenizer: Any, question: str, options: Sequence[str], *, enable_thinking: bool = True) -> str:
    """Format a multiple-choice question using the tokenizer's chat template."""
    option_lines = "\n".join(f"{chr(65 + i)}. {o}" for i, o in enumerate(options))
    if len(options) == 1:
        letter_list = "A"
    else:
        letter_list = ", ".join(chr(65 + i) for i in range(len(options) - 1)) + f", or {chr(64 + len(options))}"
    content = (
        f"{question}\n\n"
        f"{option_lines}\n\n"
        f"Please think step by step very briefly and then, on a new line, "
        f"respond with the letter ({letter_list}) of the correct option. "
        f"If you are not sure, still make a guess."
    )
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )

def parse_choice(text: str, num_options: int) -> int | None:
    """Parse a 1-indexed choice letter from generated text."""
    if num_options <= 0:
        return None
    capped = min(num_options, 26)

    if "</think>" in text:
        text = text.split("</think>")[-1]
    elif "<think>" in text:
        return None
    text = text.strip()

    # Try explicit answer patterns first.
    text_lower = text.lower()
    for pattern in ("answer:", "answer is", "correct option is"):
        if pattern in text_lower:
            after = text_lower.split(pattern)[-1].strip().lstrip(":").strip().rstrip(".").strip()
            after = after.replace("*", "")
            if after and after[0].isalpha():
                value = ord(after[0].upper()) - ord("A") + 1
                if 1 <= value <= capped:
                    return value

    # Try a standalone letter such as "B" or "B.".
    clean = text_lower.split(".")[0].strip().replace("*", "")
    valid_letters = [chr(ord("a") + i) for i in range(capped)]
    if clean in valid_letters:
        return ord(clean) - ord("a") + 1

    # Fall back to the last isolated capital letter.
    valid_upper = "".join(chr(ord("A") + i) for i in range(capped))
    matches = re.findall(rf"(?<![a-zA-Z])([{valid_upper}])(?![a-zA-Z])", text)
    if matches:
        return ord(matches[-1]) - ord("A") + 1
    return None

def kv_pairs(block: Any) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Extract raw (K, V) tensor pairs from a CacheBlock's DynamicCache."""
    return [(layer.keys, layer.values) for layer in block.kv.layers]


def fixed_chunks(span: Span, chunk_size: int) -> list[Span]:
    """Split a span into fixed-size chunks."""
    s, e = span
    chunks: list[Span] = []
    while s < e:
        chunks.append((s, min(e, s + chunk_size)))
        s += chunk_size
    return chunks


def scaffold(backend: Any, full: Any, article_span: Span, article_blocks: list[Any], total: int) -> CacheContext:
    """Assemble prefix, article blocks, and suffix from a full-prefill block."""
    blocks: list[Any] = []
    if article_span[0] > 0:
        blocks.append(backend.slice(full, 0, article_span[0]))
    blocks.extend(article_blocks)
    if article_span[1] < total:
        blocks.append(backend.slice(full, article_span[1], total))
    return CacheContext(blocks)
