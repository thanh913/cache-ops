"""Precompute full and AM caches for the demo corpus."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from memory_ops import CacheContext, compact_kv_cache
from memory_ops_runtime.backends import HFBackend
from memory_ops_runtime.backends.hf_am import build_am_block
from memory_ops_runtime.cache_io import save_block
from memory_ops_runtime.datasets import load_dataset
from memory_ops_runtime.eval_helpers import kv_pairs

MODEL = "Qwen/Qwen3-4B-Instruct-2507"
RATIOS = [0.2, 0.1, 0.05]
DATASET = "quality"
MAX_QUESTIONS = 10


def prep_article(
    backend: HFBackend,
    article: dict,
    output_dir: Path,
    ratios: list[float],
) -> dict:
    """Precompute full and AM caches for one article."""
    text = article["article"]
    article_id = article["article_id"]
    art_dir = output_dir / article_id

    input_ids = backend.tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"]
    n = input_ids.shape[1]
    device = backend._runtime_device()

    t0 = time.time()
    full_block = backend.prefill(input_ids)
    prefill_time = time.time() - t0
    print(f"    prefill: {n} tok, {prefill_time:.1f}s", flush=True)

    save_block(full_block, art_dir / "full", metadata={
        "model": MODEL, "strategy": "full", "ratio": 1.0,
        "original_seq_len": n,
    })

    t0 = time.time()
    queries = backend.extract_queries(input_ids)
    print(f"    queries: {time.time() - t0:.1f}s", flush=True)

    for ratio in ratios:
        t0 = time.time()
        compacted = compact_kv_cache(kv_pairs(full_block), queries, ratio)
        t_compressed = compacted[0][0].shape[2]
        positions = torch.linspace(0, n - 1, t_compressed, device=device).long()
        am_block = build_am_block(compacted, positions)
        compress_time = time.time() - t0

        save_block(am_block, art_dir / f"am_r{ratio}", metadata={
            "model": MODEL, "strategy": "am", "ratio": ratio,
            "original_seq_len": n, "compressed_tokens": t_compressed,
        })
        print(f"    AM r={ratio}: {t_compressed} tok, {compress_time:.1f}s", flush=True)

    torch.cuda.empty_cache()

    return {
        "article_id": article_id,
        "title": article.get("title", article_id),
        "text": text,
        "num_tokens": n,
        "questions": article["questions"][:MAX_QUESTIONS],
        "ratios": ratios,
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare demo corpus")
    parser.add_argument("--n-articles", type=int, default=10)
    parser.add_argument("--output", type=str, default="demo/cache_store")
    parser.add_argument("--dataset", type=str, default=DATASET)
    parser.add_argument("--model", type=str, default=MODEL)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model: {args.model}", flush=True)
    print(f"Dataset: {args.dataset}", flush=True)
    print(f"Articles: {args.n_articles}", flush=True)
    print(f"Ratios: {RATIOS}", flush=True)
    print(f"Output: {output_dir}", flush=True)

    backend = HFBackend(model_name=args.model, device="cuda")
    dataset = load_dataset(args.dataset)
    if args.n_articles > 0:
        dataset = dataset[:args.n_articles]

    index = []
    total_t0 = time.time()

    for i, article in enumerate(dataset):
        print(f"\n[{i+1}/{len(dataset)}] {article.get('title', article['article_id'])}", flush=True)
        entry = prep_article(backend, article, output_dir, RATIOS)
        index.append(entry)

    (output_dir / "index.json").write_text(json.dumps(index, indent=2))
    total_time = time.time() - total_t0
    print(f"\nDone: {len(index)} articles in {total_time:.0f}s", flush=True)
    print(f"Index saved to {output_dir / 'index.json'}", flush=True)


if __name__ == "__main__":
    main()
