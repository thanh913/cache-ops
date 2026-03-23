from __future__ import annotations

from typing import Any

from memory_ops import CacheBlock, CacheContext
from memory_ops.compression import compression_indices
from memory_ops_runtime.execution import GenerationConfig, GenerationResult


class DummyBackend:
    name = "dummy"

    def prefill(self, input_ids: Any) -> CacheBlock:
        seq_len = len(input_ids)
        positions = list(range(seq_len))
        return CacheBlock(
            kv=tuple(positions),
            positions=positions,
            rope_positions=list(positions),
        )

    def slice(self, block: CacheBlock, start: int, end: int) -> CacheBlock:
        indices = [idx for idx, pos in enumerate(block.positions) if start <= pos < end]
        positions = [block.positions[idx] for idx in indices]
        rope_positions = [block.rope_positions[idx] for idx in indices]
        kv = tuple(block.kv[idx] for idx in indices)
        return CacheBlock(
            kv=kv,
            positions=positions,
            rope_positions=rope_positions,
            metadata={**dict(block.metadata), "source": "slice"},
        )

    def rope_correct(self, block: CacheBlock) -> CacheBlock:
        return CacheBlock(
            kv=block.kv,
            positions=list(block.positions),
            rope_positions=list(block.positions),
            metadata={**dict(block.metadata), "source": "rope_corrected"},
        )

    def evict(self, block: CacheBlock, ratio: float, strategy: str = "uniform") -> CacheBlock:
        indices = compression_indices(block.seq_len, ratio, strategy)
        positions = [block.positions[idx] for idx in indices]
        rope_positions = [block.rope_positions[idx] for idx in indices]
        kv = tuple(block.kv[idx] for idx in indices)
        return CacheBlock(
            kv=kv,
            positions=positions,
            rope_positions=rope_positions,
            metadata={
                **dict(block.metadata),
                "source": f"compressed_{strategy}",
                "compression_strategy": strategy,
                "target_ratio": float(ratio),
                "target_count": len(indices),
            },
        )

    def next_token_logits(self, ctx: CacheContext, prompt_text: str) -> Any:
        ctx.validate()
        return [0.1, 0.2, 0.7]

    def generate(
        self,
        ctx: CacheContext,
        prompt_text: str,
        config: GenerationConfig,
    ) -> GenerationResult:
        ctx.validate()
        return GenerationResult(
            answer_text="A",
            num_generated_tokens=1,
            stopped_early=True,
            diagnostics={"num_context_blocks": len(ctx)},
        )
