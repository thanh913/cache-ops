"""Canonical evaluation runner for the frozen experiment matrix.

This file is the repo's source of truth for the standard report-style numbers.
If another script or example disagrees with it, this file wins. The maintained
arms are:
- `F`: full prefill baseline
- `K-uniform`: full prefill plus uniform eviction
- `K-AM`: full prefill plus AM compaction
- `K-AM-nonuniform`: AM with bundled non-uniform head budgets
"""
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

MODEL = "Qwen/Qwen3-4B-Instruct-2507"
MAX_NEW_TOKENS = 2048
DO_SAMPLE = True
SEED = 42

RATIOS = [0.2, 0.1, 0.05, 0.02]
CANONICAL_ARMS = ["F", "K-uniform", "K-AM", "K-AM-nonuniform"]

DATASET_CONFIG = {
    "longhealth": {"n_articles": 10, "max_questions": 10, "chunk_size": 4096},
    "quality": {"n_articles": 0, "max_questions": 10, "chunk_size": 999999},  # whole-article
}

# Keep the attention-sink prefix out of AM compaction.
N_SINK = 1

# Bundled non-uniform head budgets for Qwen3-4B.
HEAD_BUDGET_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "head_budgets"
    / "qwen3_4b_optimized_agnostic.json"
)


def load_head_budgets(path: Path, num_layers: int, num_heads: int, seq_len: int, target_ratio: float) -> dict[tuple[int, int], int]:
    """Load per-head budgets from the bundled sensitivity file."""
    import json
    raw = json.loads(path.read_text())
    target_size = ratio_to_count(seq_len, target_ratio)
    total_heads = num_layers * num_heads
    budgets = {}
    for key, proportion in raw.items():
        layer = int(key.split("H")[0][1:])
        head = int(key.split("H")[1])
        if layer < num_layers and head < num_heads:
            budgets[(layer, head)] = max(1, int(proportion * target_size * total_heads))
    return budgets


def arm_F(backend, input_ids, **_):
    return CacheContext(backend.prefill(input_ids))


def arm_K_uniform(backend, input_ids, article_span, chunks, ratio, **_):
    full = backend.prefill(input_ids)
    article_blocks = [backend.evict(backend.slice(full, s, e), ratio=ratio) for s, e in chunks]
    return scaffold(backend, full, article_span, article_blocks, input_ids.shape[1])


def arm_K_AM(backend, input_ids, article_span, chunks, ratio, **_):
    """Apply AM compaction per chunk, matching the paper setup."""
    full = backend.prefill(input_ids)
    n = input_ids.shape[1]
    device = full.positions.device

    article_blocks = []
    for s, e in chunks:
        chunk_block = backend.slice(full, s, e)
        chunk_queries = backend.extract_queries(input_ids[:, s:e])
        compacted = compact_kv_cache(kv_pairs(chunk_block), chunk_queries, ratio)
        t = compacted[0][0].shape[2]
        am_block = build_am_block(
            compacted,
            torch.linspace(s, e - 1, t, device=device).long(),
        )
        article_blocks.append(am_block)

    return scaffold(backend, full, article_span, article_blocks, n)


def arm_K_AM_nonuniform(backend, input_ids, article_span, chunks, ratio, **_):
    """Apply AM compaction with bundled non-uniform head budgets."""
    if not HEAD_BUDGET_PATH.exists():
        raise FileNotFoundError(f"Head budget file not found: {HEAD_BUDGET_PATH}")

    full = backend.prefill(input_ids)
    n = input_ids.shape[1]
    device = full.positions.device
    num_layers = backend.model.config.num_hidden_layers
    num_kv_heads = backend.model.config.num_key_value_heads

    article_blocks = []
    for s, e in chunks:
        chunk_block = backend.slice(full, s, e)
        chunk_queries = backend.extract_queries(input_ids[:, s:e])
        seq_len = e - s
        budgets = load_head_budgets(HEAD_BUDGET_PATH, num_layers, num_kv_heads, seq_len, ratio)
        compacted = compact_kv_cache(kv_pairs(chunk_block), chunk_queries, ratio, head_budgets=budgets)
        # Pad layers to the global max so build_am_block sees one sequence length.
        global_t_max = max(layer_k.shape[2] for layer_k, _, _ in compacted)
        d = compacted[0][0].shape[3]
        padded = []
        for layer_k, layer_b, layer_v in compacted:
            t_layer = layer_k.shape[2]
            if t_layer < global_t_max:
                pad = global_t_max - t_layer
                layer_k = torch.cat([layer_k, torch.zeros(1, layer_k.shape[1], pad, d, device=device, dtype=layer_k.dtype)], dim=2)
                layer_v = torch.cat([layer_v, torch.zeros(1, layer_v.shape[1], pad, d, device=device, dtype=layer_v.dtype)], dim=2)
                layer_b = torch.cat([layer_b, torch.zeros(layer_b.shape[0], pad, device=device, dtype=layer_b.dtype)], dim=1)
            padded.append((layer_k, layer_b, layer_v))
        am_block = build_am_block(
            padded,
            torch.linspace(s, e - 1, global_t_max, device=device).long(),
        )
        article_blocks.append(am_block)

    return scaffold(backend, full, article_span, article_blocks, n)


def arm_K_OMP(backend, input_ids, article_span, ratio, **_):
    full = backend.prefill(input_ids)
    queries = backend.extract_queries(input_ids)
    compacted = compact_kv_cache(kv_pairs(full), queries, ratio, key_selection="omp")
    n = input_ids.shape[1]
    t = compacted[0][0].shape[2]
    return CacheContext(build_am_block(compacted, torch.linspace(0, n - 1, t, device=full.positions.device).long()))


def arm_P1_uniform(backend, input_ids, article_span, chunks, ratio, **_):
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


def arm_P0_uniform(backend, input_ids, article_span, chunks, ratio, **_):
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
    "F": arm_F,
    "K-uniform": arm_K_uniform,
    "K-AM": arm_K_AM,
    "K-AM-nonuniform": arm_K_AM_nonuniform,
    "K-OMP": arm_K_OMP,
    "P1-uniform": arm_P1_uniform,
    "P0-uniform": arm_P0_uniform,
}


def run_eval(dataset_name: str, n_articles_override: int | None = None, max_questions_override: int | None = None):
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    cfg = dict(DATASET_CONFIG[dataset_name])
    if n_articles_override is not None:
        cfg["n_articles"] = n_articles_override
    if max_questions_override is not None:
        cfg["max_questions"] = max_questions_override
    output_path = Path(f"logs/canonical_{dataset_name}.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Canonical eval: {dataset_name}", flush=True)
    print(f"  Model: {MODEL}", flush=True)
    print(f"  Arms: {CANONICAL_ARMS}", flush=True)
    print(f"  Ratios: {RATIOS}", flush=True)
    print(f"  Output: {output_path}", flush=True)

    backend = HFBackend(model_name=MODEL, device="cuda")
    gen_config = GenerationConfig(max_new_tokens=MAX_NEW_TOKENS, do_sample=DO_SAMPLE)

    dataset = load_eval_dataset(dataset_name)
    if cfg["n_articles"] > 0:
        dataset = dataset[:cfg["n_articles"]]

    results = []
    total_questions = 0

    for art_idx, article in enumerate(dataset):
        text = article["article"]
        input_ids = backend.tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"]
        n = input_ids.shape[1]
        article_span = (0, n)
        chunks = fixed_chunks(article_span, cfg.get("chunk_size", 999999))
        questions = article["questions"][:cfg["max_questions"]]

        print(f"\n[{art_idx+1}/{len(dataset)}] {article['title']} -- {n} tok, {len(chunks)} chunks, {len(questions)} q", flush=True)

        all_prompts = []
        for q_idx, q in enumerate(questions):
            all_prompts.append(format_question(backend.tokenizer, q["question"], q["options"]))

        ref_tokens = {}
        ref_prompts = dict(enumerate(all_prompts))
        t_ref = time.time()
        ctx_f = arm_F(backend=backend, input_ids=input_ids,
                      article_span=article_span, chunks=chunks, ratio=1.0)
        f_outputs = backend.generate_batch(ctx_f, all_prompts, gen_config)
        for q_idx, out in enumerate(f_outputs):
            ref_ids = backend.tokenizer.encode(out.answer_text, add_special_tokens=False)
            ref_tokens[q_idx] = ref_ids
        print(f"  [phase] F-arm refs: {time.time()-t_ref:.1f}s", flush=True)
        torch.cuda.empty_cache()

        for arm_name in CANONICAL_ARMS:
            arm_fn = ARMS[arm_name]
            ratios = [1.0] if arm_name == "F" else RATIOS

            for ratio in ratios:
                label = arm_name if arm_name == "F" else f"{arm_name}(r={ratio})"

                t0 = time.time()
                ctx = arm_fn(
                    backend=backend, input_ids=input_ids,
                    article_span=article_span, chunks=chunks, ratio=ratio,
                )
                build_time = time.time() - t0

                t0 = time.time()
                batch_outputs = backend.generate_batch(ctx, all_prompts, gen_config)
                gen_time = time.time() - t0

                t0 = time.time()
                ppl_prompts = [ref_prompts[i] for i in range(len(questions)) if i in ref_tokens and ref_tokens[i]]
                ppl_answers = [ref_tokens[i] for i in range(len(questions)) if i in ref_tokens and ref_tokens[i]]
                ppl_indices = [i for i in range(len(questions)) if i in ref_tokens and ref_tokens[i]]
                ppl_values: dict[int, float] = {}
                if ppl_prompts:
                    try:
                        batch_ppls = backend.compute_perplexity_batch(ctx, ppl_prompts, ppl_answers)
                        for idx, ppl_val in zip(ppl_indices, batch_ppls):
                            ppl_values[idx] = ppl_val
                    except Exception as e:
                        print(f"    [ppl batch error] {e}", flush=True)
                ppl_time = time.time() - t0

                print(f"  [phase] {label}: {build_time:.1f}s (build) + {gen_time:.1f}s (gen) + {ppl_time:.1f}s (ppl)", flush=True)

                for q_idx, (q, out) in enumerate(zip(questions, batch_outputs)):
                    pred = parse_choice(out.answer_text, len(q["options"]))
                    gold = q["gold_label"]
                    correct = pred == gold
                    ppl = ppl_values.get(q_idx)

                    marker = "+" if correct else "-"
                    ppl_str = f" ppl={ppl:.2f}" if ppl is not None else ""
                    print(f"    q{q_idx}: pred={pred} gold={gold} {marker}{ppl_str}", flush=True)

                    row = {
                        "article_id": article["article_id"],
                        "arm": arm_name,
                        "ratio": ratio,
                        "question_idx": q_idx,
                        "pred": pred,
                        "gold": gold,
                        "correct": correct,
                        "parsed": pred is not None,
                        "build_s": round(build_time, 2),
                        "gen_s": round(gen_time / len(questions), 2),
                        "raw_answer": out.answer_text,
                    }
                    if ppl is not None:
                        row["perplexity"] = round(ppl, 4)
                    results.append(row)
                    total_questions += 1

                torch.cuda.empty_cache()

    print(f"\n{'='*70}", flush=True)
    print(f"Canonical eval: {dataset_name} ({total_questions} question-arm pairs)", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"{'Arm':25s} {'Acc':>8s} {'Parse':>8s} {'PPL':>10s}", flush=True)
    print(f"{'-'*25} {'-'*8} {'-'*8} {'-'*10}", flush=True)

    for arm_name in CANONICAL_ARMS:
        ratios = [1.0] if arm_name == "F" else RATIOS
        for ratio in ratios:
            rows = [r for r in results if r["arm"] == arm_name and r["ratio"] == ratio]
            if not rows:
                continue
            n_total = len(rows)
            n_correct = sum(r["correct"] for r in rows)
            n_parsed = sum(r["parsed"] for r in rows)
            label = arm_name if arm_name == "F" else f"{arm_name}(r={ratio})"
            acc = f"{100*n_correct/n_total:.0f}%"
            parse = f"{100*n_parsed/n_total:.0f}%"
            ppl_rows = [r["perplexity"] for r in rows if "perplexity" in r]
            ppl_str = f"{sum(ppl_rows)/len(ppl_rows):.2f}" if ppl_rows else "n/a"
            print(f"  {label:23s} {acc:>8s} {parse:>8s} {ppl_str:>10s}", flush=True)

    meta = {
        "contract": "CANONICAL_EVAL.md",
        "model": MODEL,
        "dataset": dataset_name,
        "max_new_tokens": MAX_NEW_TOKENS,
        "seed": SEED,
        "ratios": RATIOS,
        "arms": CANONICAL_ARMS,
        "chunk_size": cfg.get("chunk_size", 999999),
        "n_articles": len(dataset),
        "total_questions": total_questions,
    }
    output_path.write_text(json.dumps({"meta": meta, "results": results}, indent=2))
    print(f"\nSaved to {output_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Canonical evaluation (frozen contract)")
    parser.add_argument("dataset", choices=["longhealth", "quality"],
                        help="Dataset to evaluate")
    parser.add_argument("--n-articles", type=int, default=None,
                        help="Override number of articles (for smoke testing)")
    parser.add_argument("--max-questions", type=int, default=None,
                        help="Override max questions per article (for smoke testing)")
    args = parser.parse_args()
    run_eval(args.dataset, n_articles_override=args.n_articles, max_questions_override=args.max_questions)


if __name__ == "__main__":
    main()
