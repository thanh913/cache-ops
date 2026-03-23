"""Flexible report experiment runner for F/K/P1/P0-style ablations."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from memory_ops import CacheContext, compact_kv_cache
from memory_ops.compression import ratio_to_count
from memory_ops_runtime.backends import HFBackend
from memory_ops_runtime.backends.hf_am import build_am_block
from memory_ops_runtime import GenerationConfig

from memory_ops_runtime.datasets import load_dataset as load_eval_dataset
from memory_ops_runtime.eval_helpers import format_question, fixed_chunks, kv_pairs, parse_choice, scaffold

def arm_F(backend, input_ids, **_):
    """Full prefill baseline."""
    return CacheContext(backend.prefill(input_ids))


def arm_K_uniform(backend, input_ids, article_span, chunks, ratio, **_):
    """Full prefill → slice chunks → uniform eviction."""
    full = backend.prefill(input_ids)
    article_blocks = [backend.evict(backend.slice(full, s, e), ratio=ratio) for s, e in chunks]
    return scaffold(backend, full, article_span, article_blocks, input_ids.shape[1])


def arm_K_AM(backend, input_ids, article_span, ratio, **_):
    """Full prefill → AM compact entire cache."""
    full = backend.prefill(input_ids)
    queries = backend.extract_queries(input_ids)
    compacted = compact_kv_cache(kv_pairs(full), queries, ratio)
    n = input_ids.shape[1]
    t = compacted[0][0].shape[2]
    return CacheContext(build_am_block(compacted, torch.linspace(0, n - 1, t, device=full.positions.device).long()))


def arm_P1_uniform(backend, input_ids, article_span, chunks, ratio, **_):
    """Independent prefill → RoPE correct → uniform eviction."""
    full = backend.prefill(input_ids)
    device = full.positions.device
    article_blocks = [
        backend.evict(
            backend.rope_correct(
                backend.prefill(input_ids[:, s:e]).with_positions(
                    torch.arange(s, e, device=device, dtype=torch.long)
                )
            ),
            ratio=ratio,
        )
        for s, e in chunks
    ]
    return scaffold(backend, full, article_span, article_blocks, input_ids.shape[1])


def arm_P1_AM(backend, input_ids, article_span, chunks, ratio, **_):
    """Independent prefill, per-chunk AM, then implicit RoPE correction."""
    full = backend.prefill(input_ids)
    device = full.positions.device
    am_chunks = []
    for s, e in chunks:
        chunk_raw = backend.prefill(input_ids[:, s:e])
        # Extract queries before RoPE correction so Q and K share positions.
        chunk_queries = backend.extract_queries(input_ids[:, s:e])
        compacted = compact_kv_cache(kv_pairs(chunk_raw), chunk_queries, ratio)
        t = compacted[0][0].shape[2]
        # Linspaced target positions make the AM block logically global.
        am_chunks.append(build_am_block(compacted, torch.linspace(s, e - 1, t, device=device).long()))
    return scaffold(backend, full, article_span, am_chunks, input_ids.shape[1])


def arm_P0_uniform(backend, input_ids, article_span, chunks, ratio, **_):
    """Independent prefill → NO RoPE correction → uniform eviction (ablation)."""
    full = backend.prefill(input_ids)
    device = full.positions.device
    article_blocks = [
        backend.evict(
            backend.prefill(input_ids[:, s:e]).with_positions(
                torch.arange(s, e, device=device, dtype=torch.long)
            ),
            ratio=ratio,
        )
        for s, e in chunks
    ]
    return scaffold(backend, full, article_span, article_blocks, input_ids.shape[1])

ARMS = {
    "F": (arm_F, "backend"),
    "K-uniform": (arm_K_uniform, "backend"),
    "K-AM": (arm_K_AM, "backend"),
    "P1-uniform": (arm_P1_uniform, "backend"),
    "P1-AM": (arm_P1_AM, "backend"),
    "P0-uniform": (arm_P0_uniform, "backend"),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--n-articles", type=int, default=0,
                        help="Number of articles (0=all)")
    parser.add_argument("--max-questions", type=int, default=10)
    parser.add_argument("--ratios", nargs="+", type=float, default=[0.5, 0.2, 0.1])
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--arms", nargs="+", default=list(ARMS.keys()))
    parser.add_argument("--max-article-chars", type=int, default=0,
                        help="Truncate articles to this many chars (0=no truncation)")
    parser.add_argument("--dataset", default="quality", choices=["quality", "longhealth"],
                        help="Dataset to evaluate on")
    parser.add_argument("--perplexity", action="store_true",
                        help="Compute perplexity of reference (F-arm) answer under each compressed arm")
    parser.add_argument("--output", default="logs/report_experiments.json")
    args = parser.parse_args()

    backend = HFBackend(model_name=args.model_name, device="cuda")
    config = GenerationConfig(max_new_tokens=32, do_sample=False)

    dataset = load_eval_dataset(args.dataset)
    if args.n_articles > 0:
        dataset = dataset[:args.n_articles]
    results = []

    for art_idx, article in enumerate(dataset):
        text = article["article"][:args.max_article_chars] if args.max_article_chars > 0 else article["article"]
        input_ids = backend.tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"]
        n = input_ids.shape[1]
        article_span = (0, n)
        chunks = fixed_chunks(article_span, args.chunk_size)
        questions = article["questions"][:args.max_questions]

        print(f"\n[{art_idx+1}/{len(dataset)}] {article['title']} — {n} tok, {len(chunks)} chunks, {len(questions)} q", flush=True)

        ref_tokens = {}  # q_idx -> list[int]
        ref_prompts = {}  # q_idx -> str
        if args.perplexity and "F" in args.arms:
            print("  [perplexity] Collecting F-arm reference answers...", flush=True)
            ctx_f = arm_F(backend=backend, input_ids=input_ids,
                          article_span=article_span, chunks=chunks, ratio=1.0)
            for q_idx, q in enumerate(questions):
                prompt = format_question(backend.tokenizer, q["question"], q["options"])
                ref_prompts[q_idx] = prompt
                out = backend.generate(ctx_f, prompt, config)
                ref_ids = backend.tokenizer.encode(out.answer_text, add_special_tokens=False)
                ref_tokens[q_idx] = ref_ids
            torch.cuda.empty_cache()
            print(f"  [perplexity] Collected {len(ref_tokens)} reference answers", flush=True)

        for arm_name in args.arms:
            arm_fn, exec_type = ARMS[arm_name]
            ratios = [1.0] if arm_name == "F" else args.ratios

            for ratio in ratios:
                label = arm_name if arm_name == "F" else f"{arm_name}(r={ratio})"

                t0 = time.time()
                ctx = arm_fn(
                    backend=backend, input_ids=input_ids,
                    article_span=article_span, chunks=chunks, ratio=ratio,
                )
                build_time = time.time() - t0

                gen_fn = backend.generate

                for q_idx, q in enumerate(questions):
                    prompt = format_question(backend.tokenizer, q["question"], q["options"])
                    t0 = time.time()
                    out = gen_fn(ctx, prompt, config)
                    gen_time = time.time() - t0
                    pred = parse_choice(out.answer_text, len(q["options"]))
                    gold = q["gold_label"]
                    correct = pred == gold
                    marker = "✓" if correct else "✗"

                    ppl = None
                    if args.perplexity and q_idx in ref_tokens and ref_tokens[q_idx]:
                        try:
                            ppl = backend.compute_perplexity(ctx, ref_prompts[q_idx], ref_tokens[q_idx])
                        except Exception as e:
                            print(f"    [perplexity error] {e}", flush=True)

                    ppl_str = f" ppl={ppl:.2f}" if ppl is not None else ""
                    print(f"  {label:20s} q{q_idx}: pred={pred} gold={gold} {marker}{ppl_str} ({build_time:.1f}s+{gen_time:.1f}s)", flush=True)
                    row = {
                        "article_id": article["article_id"],
                        "arm": arm_name, "ratio": ratio,
                        "question_idx": q_idx, "pred": pred, "gold": gold,
                        "correct": correct,
                        "build_s": build_time, "gen_s": gen_time,
                    }
                    if ppl is not None:
                        row["perplexity"] = ppl
                    results.append(row)

                torch.cuda.empty_cache()

    print("\n" + "=" * 60, flush=True)
    for arm_name in args.arms:
        ratios = [1.0] if arm_name == "F" else args.ratios
        for ratio in ratios:
            rows = [r for r in results if r["arm"] == arm_name and r["ratio"] == ratio]
            if not rows:
                continue
            n_total = len(rows)
            n_correct = sum(r["correct"] for r in rows)
            n_parsed = sum(1 for r in rows if r["pred"] is not None)
            label = arm_name if arm_name == "F" else f"{arm_name}(r={ratio})"
            ppl_rows = [r["perplexity"] for r in rows if "perplexity" in r]
            ppl_str = f", ppl={sum(ppl_rows)/len(ppl_rows):.2f}" if ppl_rows else ""
            print(f"  {label:20s}: {n_correct}/{n_total} correct ({100*n_correct/n_total:.0f}%), {n_parsed}/{n_total} parsed{ppl_str}", flush=True)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps({"results": results}, indent=2))
    print(f"\nSaved to {args.output}", flush=True)


if __name__ == "__main__":
    main()
