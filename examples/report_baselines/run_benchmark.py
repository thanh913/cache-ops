"""CLI wrapper around the example report-baseline workflow."""
from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence

import torch

from memory_ops_runtime import GenerationConfig
from memory_ops_runtime.backends import HFBackend

from .datasets import load_dataset
from .workflow import (
    build_context_for_spec,
    default_chunking,
    execute_prompt,
    format_question,
    normalize_compression_strategy,
    parse_choice,
    plan_chunks,
    resolve_specs,
    token_count,
    tokenize_document,
)


def run_benchmark(
    *,
    backend: HFBackend,
    dataset: List[Dict],
    arms: Sequence[str],
    compression_strategy: str | None,
    target_ratio: float,
    n_questions_per_article: int | None,
    gen_config: GenerationConfig,
    seed: int,
    chunking: str,
    chunk_size: int,
) -> dict:
    rng = random.Random(seed)
    specs = resolve_specs(
        arms=arms,
        compression_strategy=compression_strategy,
        target_ratio=target_ratio,
    )

    article_cases: list[dict] = []
    for article in dataset:
        questions = list(article["questions"])
        if n_questions_per_article is not None:
            rng.shuffle(questions)
            questions = questions[:n_questions_per_article]
        input_ids, article_span, metadata = tokenize_document(
            backend.tokenizer,
            article_text=str(article["article"]),
            article_id=article["article_id"],
            article_title=article.get("title"),
        )
        article_cases.append(
            {
                "article": article,
                "input_ids": input_ids,
                "article_span": article_span,
                "metadata": metadata,
                "questions": questions,
            }
        )

    results_by_run: dict[str, list[dict]] = {spec.label: [] for spec in specs}
    for spec in specs:
        for case in article_cases:
            chunks = plan_chunks(
                case["article_span"],
                article_text=str(case["article"]["article"]),
                tokenizer=backend.tokenizer,
                strategy=chunking,
                chunk_size=chunk_size,
            )
            ctx = build_context_for_spec(
                backend, case["input_ids"], case["article_span"], chunks, spec,
            )

            outputs = []
            samples = []
            correct = 0
            total_generate_s = 0.0
            for question in case["questions"]:
                prompt_text = format_question(backend.tokenizer, str(question["question"]), question["options"])
                out, elapsed = execute_prompt(
                    backend=backend,
                    ctx=ctx,
                    prompt_text=prompt_text,
                    config=gen_config,
                )
                total_generate_s += elapsed
                outputs.append(out)

                gold = int(question["gold_label"])
                pred_choice = parse_choice(out.answer_text, len(question["options"]))
                if pred_choice == gold:
                    correct += 1
                samples.append(
                    {
                        "question_id": question["question_unique_id"],
                        "gold_choice": gold,
                        "pred_choice": pred_choice,
                        "answer_text": out.answer_text,
                        "num_generated_tokens": out.num_generated_tokens,
                    }
                )

            total_questions = len(outputs)
            parsed = sum(1 for sample in samples if sample["pred_choice"] is not None)
            total_generated_tokens = sum(output.num_generated_tokens for output in outputs)
            results_by_run[spec.label].append(
                {
                    "run_label": spec.label,
                    "prefill": spec.prefill,
                    "rope_correct": spec.rope_correct,
                    "compression_strategy": spec.compression_strategy or "none",
                    "target_ratio": float(spec.target_ratio),
                    "article_id": case["article"]["article_id"],
                    "article_title": case["article"].get("title", ""),
                    "num_questions": total_questions,
                    "num_correct": correct,
                    "document_tokens": token_count(case["input_ids"]),
                    "context_tokens": ctx.physical_tokens,
                    "logical_context_tokens": ctx.next_position,
                    "context_requires_rope_correction": ctx.requires_rope_correction,
                    "parse_rate": (parsed / total_questions) if total_questions else 0.0,
                    "accuracy": (correct / total_questions) if total_questions else 0.0,
                    "generate_s": total_generate_s,
                    "total_generated_tokens": total_generated_tokens,
                    "tokens_per_second": (
                        total_generated_tokens / total_generate_s
                    ) if total_generate_s > 0 else 0.0,
                    "samples": samples,
                }
            )

    summary: dict[str, dict] = {}
    for label, rows in results_by_run.items():
        total_questions = sum(row["num_questions"] for row in rows)
        total_correct = sum(row["num_correct"] for row in rows)
        weighted_parse = sum(row["parse_rate"] * row["num_questions"] for row in rows)
        total_generate_s = sum(row["generate_s"] for row in rows)
        total_generated_tokens = sum(row["total_generated_tokens"] for row in rows)
        summary[label] = {
            "num_articles": len(rows),
            "total_questions": total_questions,
            "overall_parse_rate": (weighted_parse / total_questions) if total_questions else 0.0,
            "overall_accuracy": (total_correct / total_questions) if total_questions else 0.0,
            "total_generate_s": total_generate_s,
            "total_generated_tokens": total_generated_tokens,
            "avg_tokens_per_second": (
                total_generated_tokens / total_generate_s
            ) if total_generate_s > 0 else 0.0,
        }

    return {"summary": summary, "results": results_by_run}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run report-baseline benchmark.")
    parser.add_argument("--dataset", default="quality")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--model-name", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--device", default=None)
    parser.add_argument("--arms", nargs="+", choices=["F", "K", "P1", "P0"], default=["F"])
    parser.add_argument("--compression-strategy", default="uniform")
    parser.add_argument("--target-ratio", type=float, default=1.0)
    parser.add_argument("--n-articles", type=int, default=1)
    parser.add_argument("--start-article", type=int, default=0)
    parser.add_argument("--n-questions-per-article", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--do-sample", type=int, default=0)
    parser.add_argument("--chunking", choices=["fixed", "longhealth_fine"], default=None)
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--attn-implementation", type=str, default=None)
    parser.add_argument("--seed", type=int, default=67)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    dataset = load_dataset(args.dataset, args.data_root)
    if args.n_articles == -1:
        selected_articles = dataset[args.start_article:]
    else:
        selected_articles = dataset[args.start_article : args.start_article + args.n_articles]

    backend = HFBackend(
        model_name=args.model_name,
        device=args.device,
        attn_implementation=args.attn_implementation,
    )
    chunking = args.chunking or default_chunking(args.dataset)
    gen_config = GenerationConfig(max_new_tokens=args.max_new_tokens, do_sample=bool(args.do_sample))

    report = run_benchmark(
        backend=backend,
        dataset=selected_articles,
        arms=args.arms,
        compression_strategy=normalize_compression_strategy(args.compression_strategy),
        target_ratio=args.target_ratio,
        n_questions_per_article=args.n_questions_per_article,
        gen_config=gen_config,
        seed=args.seed,
        chunking=chunking,
        chunk_size=args.chunk_size,
    )

    payload = {
        "meta": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "dataset": args.dataset,
            "model_name": args.model_name,
            "arms": args.arms,
            "compression_strategy": args.compression_strategy,
            "target_ratio": args.target_ratio,
            "n_articles": args.n_articles,
            "start_article": args.start_article,
            "n_questions_per_article": args.n_questions_per_article,
            "max_new_tokens": args.max_new_tokens,
            "seed": args.seed,
            "chunking": chunking,
            "chunk_size": args.chunk_size,
            "attn_implementation": args.attn_implementation,
        },
        "report": report,
    }

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2))
        print(f"Wrote benchmark report to {output_path}")
    else:
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
