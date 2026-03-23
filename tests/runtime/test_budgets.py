from __future__ import annotations

import json

import pytest
import torch

from memory_ops_runtime import load_head_budget_counts_from_json


@pytest.mark.logical
def test_load_head_budget_counts_from_json_is_deterministic_and_capped(tmp_path) -> None:
    budget_path = tmp_path / "head_budgets.json"
    budget_path.write_text(
        json.dumps(
            {
                "L0H0": 10.0,
                "L0H1": 1.0,
                "L1H0": 1.0,
                "L1H1": 1.0,
            }
        ),
        encoding="utf-8",
    )

    counts_a = load_head_budget_counts_from_json(
        budget_path,
        num_layers=2,
        num_heads=2,
        seq_len=16,
        target_ratio=0.5,
        max_ratio_per_head=0.5,
    )
    counts_b = load_head_budget_counts_from_json(
        budget_path,
        num_layers=2,
        num_heads=2,
        seq_len=16,
        target_ratio=0.5,
        max_ratio_per_head=0.5,
    )

    assert torch.equal(counts_a, counts_b)
    assert counts_a.shape == (2, 2)
    assert int(counts_a.sum().item()) == 32
    assert int(counts_a.max().item()) <= 8
    assert (counts_a >= 1).all()


@pytest.mark.logical
def test_load_head_budget_counts_from_json_rejects_empty_budget(tmp_path) -> None:
    budget_path = tmp_path / "head_budgets.json"
    budget_path.write_text(json.dumps({"L0H0": 0.0, "L0H1": 0.0}), encoding="utf-8")

    with pytest.raises(ValueError, match="positive total proportion"):
        load_head_budget_counts_from_json(
            budget_path,
            num_layers=1,
            num_heads=2,
            seq_len=8,
            target_ratio=0.5,
        )
