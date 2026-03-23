"""Dataset loaders for QA evaluation."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_quality(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    combined: dict[tuple[str, str], dict] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            article = row.get("article", "").strip()
            original_id = str(row["article_id"])
            article_id = f"quality_{original_id}"
            key = (article_id, article)

            questions = []
            for i, q in enumerate(row.get("questions", [])):
                options = list(q.get("options", []))
                if not options:
                    continue
                questions.append(
                    {
                        "question_unique_id": q.get("question_unique_id", f"{article_id}_q{i}"),
                        "question": str(q.get("question", "")),
                        "options": options,
                        "gold_label": int(q.get("gold_label", 1)),
                    }
                )

            payload = {
                "article_id": article_id,
                "title": str(row.get("title", original_id)),
                "article": article,
                "questions": questions,
            }
            if key in combined:
                combined[key]["questions"].extend(questions)
            else:
                combined[key] = payload
    return list(combined.values())


def load_longhealth(path: str | Path, *, patients_per_article: int = 1) -> list[dict[str, Any]]:
    if patients_per_article <= 0:
        raise ValueError("patients_per_article must be positive")

    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    patient_ids = sorted(raw.keys())
    out: list[dict[str, Any]] = []
    for article_idx, start in enumerate(range(0, len(patient_ids), patients_per_article)):
        end = min(start + patients_per_article, len(patient_ids))
        group = patient_ids[start:end]

        article_parts: list[str] = []
        grouped_questions: list[dict] = []
        for patient_id in group:
            patient = raw[patient_id]
            for note_id, note_text in patient["texts"].items():
                article_parts.append(f"<{note_id}>\n{note_text}\n</{note_id}>")

            for q_idx, q in enumerate(patient.get("questions", [])):
                options = [
                    str(q.get("answer_a", "")),
                    str(q.get("answer_b", "")),
                    str(q.get("answer_c", "")),
                    str(q.get("answer_d", "")),
                    str(q.get("answer_e", "")),
                ]
                correct = str(q.get("correct", "")).strip()
                gold = 1
                for i, opt in enumerate(options):
                    if opt.strip() == correct:
                        gold = i + 1
                        break
                grouped_questions.append(
                    {
                        "question_unique_id": f"{patient_id}_q{q.get('No', q_idx)}",
                        "question": str(q.get("question", "")),
                        "options": options,
                        "gold_label": gold,
                    }
                )

        article = "\n\n".join(article_parts).strip()
        if patients_per_article == 1:
            pid = group[0]
            title = str(raw[pid].get("name", pid))
            article_id = f"longhealth_{pid}"
        else:
            title = f"Patients {start + 1}-{end}"
            article_id = f"longhealth_group_{article_idx:02d}_patients_{start + 1:02d}-{end:02d}"

        out.append(
            {
                "article_id": article_id,
                "title": title,
                "article": article,
                "questions": grouped_questions,
            }
        )
    return out


def _resolve_data_root(data_root: str | Path | None) -> Path:
    if data_root is not None:
        root = Path(data_root).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(f"data root not found: {root}")
        return root

    # Walk upward from this file until we find repo data/.
    here = Path(__file__).resolve()
    for ancestor in [here.parents[2], here.parents[3]]:  # src/memory_ops_runtime -> repo root
        candidate = ancestor / "data"
        if candidate.exists():
            return candidate
    raise FileNotFoundError("could not find data directory; pass data_root explicitly")


_DATASET_FILES = {
    "quality": "QuALITY.v1.0.1.htmlstripped.dev",
    "longhealth": "longhealth_benchmark_v5.json",
}


def load_dataset(dataset_name: str, data_root: str | Path | None = None) -> list[dict[str, Any]]:
    root = _resolve_data_root(data_root)
    if dataset_name == "quality":
        return load_quality(root / _DATASET_FILES["quality"])
    if dataset_name == "longhealth":
        return load_longhealth(root / _DATASET_FILES["longhealth"])
    if dataset_name.startswith("longhealth"):
        patients = int(dataset_name[len("longhealth"):])
        if patients <= 0:
            raise ValueError("patients_per_article must be positive")
        return load_longhealth(root / _DATASET_FILES["longhealth"], patients_per_article=patients)
    raise ValueError(f"unsupported dataset: {dataset_name!r}")
