from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int = 2048
    do_sample: bool = False
    temperature: float = 0.7
    top_p: float = 0.9


@dataclass(frozen=True)
class GenerationResult:
    answer_text: str
    num_generated_tokens: int
    stopped_early: bool
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
