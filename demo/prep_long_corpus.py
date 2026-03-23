"""Build long demo documents from independently compressed articles.

The long-doc demo keeps the same high-level story as P1 assembly: compress
articles independently, shift them into global positions, then concatenate.
The wrinkle is that AM blocks do not keep per-token source indices per head, so
we cannot do exact per-token RoPE correction after compaction. Instead this
script applies one uniform RoPE offset per article, which preserves the
within-article basis while moving the whole compacted block to the right global
region.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from memory_ops import CacheBlock, CacheContext, compact_kv_cache
from memory_ops_runtime.backends import HFBackend
from memory_ops_runtime.backends.hf_am import AMPayload, build_am_block
from memory_ops_runtime.cache_io import save_block
from memory_ops_runtime.datasets import load_dataset
from memory_ops_runtime.eval_helpers import kv_pairs

MODEL = "Qwen/Qwen3-4B-Instruct-2507"
RATIOS = [0.2, 0.1, 0.05]


def compress_article(
    backend: HFBackend,
    input_ids: torch.Tensor,
    ratio: float,
    global_start: int,
    global_end: int,
) -> tuple[CacheBlock, int]:
    """Compress one article and shift it to global RoPE positions.

    This is intentionally approximate. After AM, different heads may have kept
    different source tokens, but `CacheBlock` still exposes one shared position
    vector. A uniform article-level RoPE offset is the clean compromise for the
    demo: it preserves relative structure inside the article and moves the whole
    block to the target global offset.
    """
    device = backend._runtime_device()

    block = backend.prefill(input_ids)
    queries = backend.extract_queries(input_ids)
    compacted = compact_kv_cache(kv_pairs(block), queries, ratio)
    t = compacted[0][0].shape[2]
    local_positions = torch.linspace(0, input_ids.shape[1] - 1, t, device=device).long()
    am_block = build_am_block(compacted, local_positions)

    # Apply one RoPE offset to every compacted key in the article.
    if global_start > 0:
        offset_ids = torch.tensor([[global_start]], device=device, dtype=torch.long)
        dummy = torch.zeros(
            1, 1,
            int(backend._rotary_emb.inv_freq.shape[0]) * 2,
            device=device,
            dtype=next(backend.model.parameters()).dtype,
        )
        cos_off, sin_off = backend._rotary_emb(dummy, offset_ids)
        cos_off = cos_off.to(dtype=next(backend.model.parameters()).dtype)
        sin_off = sin_off.to(dtype=next(backend.model.parameters()).dtype)

        from memory_ops_runtime.backends.hf import apply_rotary_pos_emb_to_cache
        shifted_keys = tuple(
            apply_rotary_pos_emb_to_cache(k, cos_off.to(k.dtype), sin_off.to(k.dtype))
            for k in am_block.kv.layer_keys
        )
        am_block = CacheBlock(
            kv=AMPayload(shifted_keys, am_block.kv.layer_values, am_block.kv.layer_beta),
            positions=am_block.positions,
            rope_positions=am_block.rope_positions,
        )

    # Set global positions for next_position and rope_base accounting.
    global_positions = torch.linspace(global_start, global_end - 1, t, device=device).long()
    am_block = CacheBlock(
        kv=am_block.kv,
        positions=global_positions,
        rope_positions=global_positions.clone(),
    )

    torch.cuda.empty_cache()
    return am_block, t


def concat_am_blocks(blocks: list[CacheBlock], num_layers: int) -> CacheBlock:
    """Concatenate multiple RoPE-corrected AM blocks into one."""
    all_keys = [[] for _ in range(num_layers)]
    all_values = [[] for _ in range(num_layers)]
    all_beta = [[] for _ in range(num_layers)]
    all_positions = []

    for block in blocks:
        kv = block.kv
        all_positions.append(block.positions)
        for li in range(num_layers):
            all_keys[li].append(kv.layer_keys[li])
            all_values[li].append(kv.layer_values[li])
            all_beta[li].append(kv.layer_beta[li])

    merged_kv = AMPayload(
        layer_keys=tuple(torch.cat(lk, dim=2) for lk in all_keys),
        layer_values=tuple(torch.cat(lv, dim=2) for lv in all_values),
        layer_beta=tuple(torch.cat(lb, dim=-1) for lb in all_beta),
    )
    merged_pos = torch.cat(all_positions)
    return CacheBlock(kv=merged_kv, positions=merged_pos, rope_positions=merged_pos.clone())


def prep_long_doc(
    backend: HFBackend,
    articles: list[dict],
    doc_id: str,
    doc_title: str,
    output_dir: Path,
    ratios: list[float],
) -> dict:
    device = backend._runtime_device()
    num_layers = backend.model.config.num_hidden_layers

    parts = []
    article_ranges = []
    cursor = 0
    all_questions = []

    for i, art in enumerate(articles):
        text = f"=== Story {i+1}: {art['title']} ===\n\n{art['article']}"
        parts.append(text)
        ids = backend.tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
        n = ids.shape[1]
        article_ranges.append((cursor, cursor + n, ids))
        cursor += n
        for q in art.get("questions", [])[:3]:
            tagged_q = dict(q)
            tagged_q["question"] = f"[About: {art['title']}] {q['question']}"
            all_questions.append(tagged_q)

    full_text = "\n\n\n".join(parts)
    total_tokens = cursor
    doc_dir = output_dir / doc_id

    full_ids = backend.tokenizer(full_text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
    actual_total = full_ids.shape[1]
    print(f"    {actual_total} tokens from {len(articles)} articles", flush=True)

    t0 = time.time()
    full_block = backend.prefill(full_ids)
    print(f"    Full prefill: {time.time() - t0:.1f}s", flush=True)
    save_block(full_block, doc_dir / "full", metadata={
        "model": MODEL, "strategy": "full", "ratio": 1.0,
        "original_seq_len": actual_total,
    })
    torch.cuda.empty_cache()

    for ratio in ratios:
        t0 = time.time()
        corrected_blocks = []
        total_compressed = 0

        for j, (gs, ge, ids) in enumerate(article_ranges):
            am_block, t_comp = compress_article(backend, ids, ratio, gs, ge)
            corrected_blocks.append(am_block)
            total_compressed += t_comp

        merged = concat_am_blocks(corrected_blocks, num_layers)
        elapsed = time.time() - t0

        save_block(merged, doc_dir / f"am_r{ratio}", metadata={
            "model": MODEL, "strategy": "am", "ratio": ratio,
            "original_seq_len": actual_total, "compressed_tokens": total_compressed,
            "num_articles": len(articles), "rope_corrected": True,
        })
        print(f"    AM r={ratio}: {total_compressed} tok, {elapsed:.1f}s", flush=True)

    return {
        "article_id": doc_id,
        "title": doc_title,
        "text": full_text,
        "num_tokens": actual_total,
        "questions": all_questions,
        "ratios": ratios,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="demo/cache_store")
    parser.add_argument("--group-size", type=int, default=5)
    parser.add_argument("--n-groups", type=int, default=2)
    parser.add_argument("--model", type=str, default=MODEL)
    args = parser.parse_args()

    output_dir = Path(args.output)
    backend = HFBackend(model_name=args.model, device="cuda")
    dataset = load_dataset("quality")
    available = dataset[5:]  # Use articles not in short-article cache

    entries = []
    for g in range(args.n_groups):
        start = g * args.group_size
        group = available[start : start + args.group_size]
        if len(group) < args.group_size:
            break
        doc_id = f"long_{g}"
        titles = ", ".join(a["title"][:15] for a in group[:3])
        doc_title = f"Long ({args.group_size} stories: {titles}...)"

        print(f"\n[{g+1}/{args.n_groups}] {doc_title}", flush=True)
        entry = prep_long_doc(backend, group, doc_id, doc_title, output_dir, RATIOS)
        entries.append(entry)

    idx_path = output_dir / "index.json"
    existing = json.loads(idx_path.read_text()) if idx_path.exists() else []
    existing = [e for e in existing if not e["article_id"].startswith("long_")]
    existing.extend(entries)
    idx_path.write_text(json.dumps(existing, indent=2))
    print(f"\nDone. Updated {idx_path}", flush=True)


if __name__ == "__main__":
    main()
