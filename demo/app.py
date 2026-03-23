"""Interactive demo for full-prefill versus pre-compressed generation."""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from multiprocessing import Process, Queue
from pathlib import Path

import gradio as gr

MODEL = "Qwen/Qwen3-4B-Instruct-2507"
RATIOS = [0.2, 0.1, 0.05]
MAX_NEW_TOKENS = 512

def _strip_think(text: str) -> str:
    """Remove visible reasoning tags from model output before displaying it."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if "<think>" in text:
        text = text[:text.index("<think>")].strip()
    return text


def full_worker(model_name: str, max_cache_len: int, warmup_text: str, req_q: Queue, res_q: Queue):
    """Worker that prefills from scratch before generation."""
    import torch
    from memory_ops import CacheContext
    from memory_ops_runtime.backends import HFBackend
    from memory_ops_runtime import GenerationConfig

    print("[full] Loading model...", flush=True)
    backend = HFBackend(model_name=model_name, device="cuda")
    backend._static_max_cache_len = max_cache_len

    gen_config = GenerationConfig(max_new_tokens=MAX_NEW_TOKENS, do_sample=True, temperature=0.7, top_p=0.9)

    # Warm up on the largest article so CUDA graphs capture the max shape.
    print(f"[full] Compiling with {len(warmup_text)} char article...", flush=True)
    warmup_ids = backend.tokenizer(warmup_text, return_tensors="pt", add_special_tokens=False)["input_ids"]
    warmup_ids = warmup_ids.to(backend._runtime_device())
    warmup_block = backend.prefill(warmup_ids)
    backend.generate(CacheContext(warmup_block), "What is this about?", gen_config)
    torch.cuda.synchronize()
    del warmup_block, warmup_ids
    torch.cuda.empty_cache()
    print("[full] Ready!", flush=True)

    while True:
        req = req_q.get()
        if req is None:
            break

        article_text = req["article_text"]
        prompt = req["prompt"]

        try:
            input_ids = backend.tokenizer(article_text, return_tensors="pt", add_special_tokens=False)["input_ids"]
            input_ids = input_ids.to(backend._runtime_device())
            n_tokens = int(input_ids.shape[1])

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            full_block = backend.prefill(input_ids)
            result = backend.generate(CacheContext(full_block), prompt, gen_config)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0

            del full_block, input_ids
            torch.cuda.empty_cache()

            res_q.put({
                "answer": _strip_think(result.answer_text),
                "time": elapsed,
                "gen_tokens": result.num_generated_tokens,
                "kv_tokens": n_tokens,
            })
        except Exception as e:
            res_q.put({"error": str(e)})


def comp_worker(model_name: str, cache_store: str, max_cache_len: int, req_q: Queue, res_q: Queue):
    """Worker that loads cached blocks before generation."""
    import torch
    from memory_ops import CacheContext
    from memory_ops_runtime.backends import HFBackend
    from memory_ops_runtime import GenerationConfig
    from memory_ops_runtime.cache_io import load_block

    print("[comp] Loading model...", flush=True)
    backend = HFBackend(model_name=model_name, device="cuda")
    backend._static_max_cache_len = max_cache_len
    backend._am_static_max_cache_len = max_cache_len

    store = Path(cache_store)
    index = json.loads((store / "index.json").read_text())
    caches: dict[str, dict[str, object]] = {}

    print("[comp] Loading caches...", flush=True)
    for art in index:
        aid = art["article_id"]
        caches[aid] = {}
        art_dir = store / aid
        for ratio in art.get("ratios", RATIOS):
            am_path = art_dir / f"am_r{ratio}"
            if am_path.exists():
                caches[aid][f"am_{ratio}"] = load_block(am_path, device="cuda")
        full_path = art_dir / "full"
        if full_path.exists():
            caches[aid]["full"] = load_block(full_path, device="cuda")

    gen_config = GenerationConfig(max_new_tokens=MAX_NEW_TOKENS, do_sample=True, temperature=0.7, top_p=0.9)

    # Warm up on the largest cached block so CUDA graphs capture the max shape.
    print("[comp] Compiling...", flush=True)
    sample = max(
        (block for c in caches.values() for k, block in c.items() if k.startswith("am_")),
        key=lambda b: CacheContext(b).physical_tokens,
        default=None,
    )
    if sample:
        backend.generate(CacheContext(sample), "What is this about?", gen_config)
        torch.cuda.synchronize()
    print("[comp] Ready!", flush=True)

    while True:
        req = req_q.get()
        if req is None:
            break

        aid = req["article_id"]
        prompt = req["prompt"]
        ratio = req["ratio"]
        strategy = req["strategy"]

        try:
            art_caches = caches.get(aid, {})

            if strategy == "AM (attention matching)":
                block = art_caches.get(f"am_{ratio}")
            elif strategy == "Uniform eviction":
                full = art_caches.get("full")
                block = backend.evict(full, ratio=ratio) if full else None
            else:
                block = None

            if block is None:
                res_q.put({"error": f"No cache for {aid} {strategy} r={ratio}"})
                continue

            ctx = CacheContext(block)
            kv_tokens = ctx.physical_tokens

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            result = backend.generate(ctx, prompt, gen_config)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0

            art_info = next((a for a in index if a["article_id"] == aid), None)
            original_comp = max(1, int(art_info["num_tokens"] * ratio + 0.5)) if art_info else kv_tokens

            res_q.put({
                "answer": _strip_think(result.answer_text),
                "time": elapsed,
                "gen_tokens": result.num_generated_tokens,
                "kv_tokens": original_comp,
            })
        except Exception as e:
            import traceback; traceback.print_exc()
            res_q.put({"error": str(e)})

full_req_q: Queue | None = None
full_res_q: Queue | None = None
comp_req_q: Queue | None = None
comp_res_q: Queue | None = None
articles: list[dict] = []
_tokenizer = None


def startup(store_path: str, model_name: str = MODEL):
    global full_req_q, full_res_q, comp_req_q, comp_res_q, articles, _tokenizer

    store = Path(store_path)
    articles = json.loads((store / "index.json").read_text())

    from transformers import AutoTokenizer
    _tokenizer = AutoTokenizer.from_pretrained(model_name)

    max_tokens = max(a["num_tokens"] for a in articles)
    max_cache_len = max_tokens + 300 + MAX_NEW_TOKENS
    warmup_article = max(articles, key=lambda a: a["num_tokens"])

    full_req_q, full_res_q = Queue(), Queue()
    comp_req_q, comp_res_q = Queue(), Queue()

    full_proc = Process(
        target=full_worker,
        args=(model_name, max_cache_len, warmup_article["text"], full_req_q, full_res_q),
        daemon=True,
    )
    comp_proc = Process(
        target=comp_worker,
        args=(model_name, store_path, max_cache_len, comp_req_q, comp_res_q),
        daemon=True,
    )

    print("Starting workers...", flush=True)
    full_proc.start()
    comp_proc.start()

    # Give both workers time to compile, then ping them.
    import time
    time.sleep(5)
    print("Waiting for workers to compile...", flush=True)
    comp_req_q.put({
        "article_id": articles[0]["article_id"],
        "prompt": "A",
        "ratio": 0.2,
        "strategy": "AM (attention matching)",
    })
    full_req_q.put({
        "article_text": "Hello world.",
        "prompt": "A",
    })
    r1 = comp_res_q.get(timeout=300)
    r2 = full_res_q.get(timeout=300)
    print(f"Workers ready! comp={r1.get('gen_tokens', 'err')} full={r2.get('gen_tokens', 'err')}", flush=True)


def on_article_change(article_title: str):
    art = next((a for a in articles if a["title"] == article_title), None)
    if art is None:
        return "", gr.update(choices=[], value=None), ""
    questions = art.get("questions", [])
    q_choices = [f"Q{i+1}: {q['question'][:100]}" for i, q in enumerate(questions)]
    preview = art["text"][:3000] + ("..." if len(art["text"]) > 3000 else "")
    info = f"{art['num_tokens']} tokens"
    return preview, gr.update(choices=q_choices, value=q_choices[0] if q_choices else None), info


def on_ask(article_title, question_dd, custom_q, ratio, strategy):
    from memory_ops_runtime.eval_helpers import format_question, parse_choice

    art = next((a for a in articles if a["title"] == article_title), None)
    if art is None:
        yield "Select an article", "", "*Select an article first*"
        return

    if custom_q and custom_q.strip():
        prompt = _tokenizer.apply_chat_template(
            [{"role": "user", "content": custom_q.strip()}],
            tokenize=False, add_generation_prompt=True, enable_thinking=True,
        )
        options, gold = None, None
    else:
        q_idx = 0
        if question_dd:
            try:
                q_idx = int(question_dd.split(":")[0][1:]) - 1
            except (ValueError, IndexError):
                q_idx = 0
        questions = art.get("questions", [])
        if q_idx >= len(questions):
            yield "No question selected", "", "*Select a question*"
            return
        q = questions[q_idx]
        prompt = format_question(_tokenizer, q["question"], q["options"])
        options, gold = q.get("options"), q.get("gold_label")

    yield "", "", "**Generating...** (compressed first, then full prefill)"

    comp_req_q.put({
        "article_id": art["article_id"],
        "prompt": prompt,
        "ratio": ratio,
        "strategy": strategy,
    })
    full_req_q.put({
        "article_text": art["text"],
        "prompt": prompt,
    })

    comp_res = comp_res_q.get(timeout=120)
    if "error" in comp_res:
        yield "", f"Error: {comp_res['error']}", f"*Compressed failed: {comp_res['error']}*"
        full_res_q.get(timeout=120)  # drain
        return

    comp_tps = comp_res["gen_tokens"] / comp_res["time"] if comp_res["time"] > 0 else 0
    yield (
        "",
        comp_res["answer"],
        f"**Compressed done** in {comp_res['time']:.1f}s ({comp_tps:.0f} tok/s) — waiting for full prefill...",
    )

    full_res = full_res_q.get(timeout=120)
    if "error" in full_res:
        yield f"Error: {full_res['error']}", comp_res["answer"], f"*Full failed: {full_res['error']}*"
        return

    full_tps = full_res["gen_tokens"] / full_res["time"] if full_res["time"] > 0 else 0
    savings_pct = (1 - comp_res["kv_tokens"] / full_res["kv_tokens"]) * 100 if full_res["kv_tokens"] > 0 else 0
    speedup = full_res["time"] / comp_res["time"] if comp_res["time"] > 0 else 0

    lines = [
        f"### Speed: prefill-from-scratch vs pre-compressed cache",
        f"| | What happens | KV tokens | Total time | Gen speed |",
        f"|---|---|---|---|---|",
        f"| **Full** | prefill {full_res['kv_tokens']:,} tok + generate | {full_res['kv_tokens']:,} | **{full_res['time']:.1f}s** | {full_tps:.0f} tok/s |",
        f"| **Compressed** | load cached KV + generate | {comp_res['kv_tokens']:,} | **{comp_res['time']:.1f}s** | {comp_tps:.0f} tok/s |",
        f"",
        f"**{speedup:.1f}x faster** — skipped prefill of {full_res['kv_tokens']:,} tokens, "
        f"compressed cache stores only {comp_res['kv_tokens']:,} tokens ({savings_pct:.0f}% less memory)",
    ]

    if options and gold:
        full_pred = parse_choice(full_res["answer"], len(options))
        comp_pred = parse_choice(comp_res["answer"], len(options))
        gold_letter = chr(64 + gold)
        full_letter = chr(64 + full_pred) if full_pred else "?"
        comp_letter = chr(64 + comp_pred) if comp_pred else "?"
        lines.append(f"\n### Correctness")
        lines.append(f"Gold: **{gold_letter}** | Full: **{full_letter}** | Compressed: **{comp_letter}**")
        if comp_pred == gold:
            lines.append(f"Compressed is **correct** despite {savings_pct:.0f}% less context!")

    yield full_res["answer"], comp_res["answer"], "\n".join(lines)


def build_ui():
    titles = [a["title"] for a in articles]

    with gr.Blocks(title="Compress & Ask") as demo:
        gr.Markdown(
            "# Compress & Ask\n"
            "Pre-compress a document's KV cache offline, then answer questions "
            "**faster** — no prefill needed, less memory used, same answers.\n\n"
            "*Full path: prefill all tokens from scratch + generate.  \n"
            "Compressed path: load pre-compressed cache + generate (skip prefill!).*"
        )

        with gr.Row():
            with gr.Column(scale=3):
                article_dd = gr.Dropdown(choices=titles, value=titles[0] if titles else None, label="Article")
                article_info = gr.Textbox(label="", interactive=False, lines=1, show_label=False)
                article_text = gr.Textbox(label="Article text", interactive=False, lines=12, max_lines=12)

            with gr.Column(scale=2):
                question_dd = gr.Dropdown(choices=[], label="Preloaded question")
                custom_q = gr.Textbox(label="Or type your own", placeholder="Ask anything about the article...", lines=2)
                with gr.Group():
                    strategy = gr.Radio(
                        choices=["AM (attention matching)", "Uniform eviction"],
                        value="AM (attention matching)",
                        label="Compression strategy",
                    )
                    ratio = gr.Dropdown(
                        choices=[("5x (keep 20%)", 0.2), ("10x (keep 10%)", 0.1), ("20x (keep 5%)", 0.05)],
                        value=0.1,
                        label="Compression level",
                    )
                ask_btn = gr.Button("Ask", variant="primary", size="lg")

        metrics_box = gr.Markdown(value="*Press Ask to compare full vs compressed generation*")

        with gr.Row(equal_height=True):
            full_answer = gr.Textbox(label="Full cache answer", lines=8, interactive=False)
            comp_answer = gr.Textbox(label="Compressed cache answer", lines=8, interactive=False)

        article_dd.change(on_article_change, [article_dd], [article_text, question_dd, article_info])
        ask_btn.click(on_ask, [article_dd, question_dd, custom_q, ratio, strategy], [full_answer, comp_answer, metrics_box])

        if titles:
            demo.load(on_article_change, [article_dd], [article_text, question_dd, article_info])

    return demo


def main():
    # CUDA requires spawn; fork breaks worker contexts.
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-store", default="demo/cache_store")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    startup(args.cache_store, args.model)
    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=args.port)


if __name__ == "__main__":
    main()
