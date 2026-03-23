"""Hugging Face backend for cache artifact execution.

The main thing to keep straight in this file is the split between logical
positions and physical cache rows. `position_ids` and `ctx.next_position`
describe where tokens live in the assembled sequence for RoPE and masking
semantics. `cache_position` and `ctx.physical_tokens` describe where rows are
written in the concrete cache tensor. Compression and AM payloads let those
two notions diverge, so the generation paths here must thread both.
"""
from __future__ import annotations

from dataclasses import dataclass
import inspect
import os
from typing import Any, Dict, Mapping, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache, StaticCache
from transformers.models.qwen3 import modeling_qwen3

from memory_ops import CacheBlock, CacheContext
from memory_ops.compression import compression_indices
from memory_ops_runtime.compacted_prefix import (
    StaticCompactedPrefixCache,
    patch_qwen3_for_static_compacted_prefix,
)
from memory_ops_runtime.execution import GenerationConfig, GenerationResult

# AM payloads stay backend-specific, so import them lazily.
try:
    from memory_ops_runtime.backends.hf_am import AMPayload
except ImportError:
    AMPayload = None


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the last dimension by half, matching RoPE's complex view."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_to_cache(cache: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply a RoPE rotation delta directly to cached keys."""
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (cache * cos) + (rotate_half(cache) * sin)


@dataclass(frozen=True)
class _ContextForwardResult:
    blocks: CacheContext
    output: Any
    next_position: int


@dataclass
class HFBackend:
    """Concrete runtime backend used by tests, scripts, and the demo."""

    model_name: str = "Qwen/Qwen3-1.7B"
    device: str | None = None
    dtype: torch.dtype | None = None
    attn_implementation: str | None = None

    name = "hf_reference"

    def __post_init__(self) -> None:
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model_kwargs: Dict[str, Any] = {}
        if self.dtype is not None:
            model_kwargs["torch_dtype"] = self.dtype
        if self.attn_implementation is not None:
            model_kwargs["attn_implementation"] = self.attn_implementation

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
        self.model = self.model.to(self.device)
        self.model.eval()
        self._forward_params = set(inspect.signature(self.model.forward).parameters)
        self._rotary_emb = getattr(getattr(self.model, "model", None), "rotary_emb", None)

        # Compile lazily because warmup sizes depend on the workload.
        self._compiled_model: Any | None = None
        self._compiled_warmup_done: bool = False
        self._compiled_am_model: Any | None = None
        # Pin static cache sizes so CUDA graphs keep the same shapes.
        self._static_max_cache_len: int = 0
        self._am_static_max_cache_len: int = 0

    def _runtime_device(self) -> torch.device:
        return next(self.model.parameters()).device

    def _encode_text(self, text: str) -> torch.Tensor:
        enc = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
        return enc["input_ids"].to(self._runtime_device())

    def _encode_prompt_ids(self, text: str) -> torch.Tensor:
        return self._encode_text(text)[0]

    def _clone_past_key_values(self, past_key_values: Any) -> Any:
        if past_key_values is None:
            return None
        if isinstance(past_key_values, DynamicCache):
            layer_data = []
            for layer in past_key_values.layers:
                layer_data.append((layer.keys.clone(), layer.values.clone()))
            return DynamicCache(ddp_cache_data=layer_data)
        if AMPayload is not None and isinstance(past_key_values, AMPayload):
            return AMPayload(
                layer_keys=tuple(k.clone() for k in past_key_values.layer_keys),
                layer_values=tuple(v.clone() for v in past_key_values.layer_values),
                layer_beta=tuple(b.clone() if b is not None else None for b in past_key_values.layer_beta),
            )
        raise TypeError(f"unsupported past_key_values type: {type(past_key_values)}")

    def _positions_tensor(self, span: tuple[int, int]) -> torch.Tensor:
        start, end = span
        return torch.arange(start, end, device=self._runtime_device(), dtype=torch.long)

    def _slice_past_key_values(self, past_key_values: Any, indices: torch.Tensor) -> Any:
        if isinstance(past_key_values, DynamicCache):
            layer_data = []
            for layer in past_key_values.layers:
                layer_data.append(
                    (
                        layer.keys.index_select(2, indices).clone(),
                        layer.values.index_select(2, indices).clone(),
                    )
                )
            return DynamicCache(ddp_cache_data=layer_data)
        raise TypeError(f"unsupported past_key_values type: {type(past_key_values)}")

    def _concat_past_key_values(self, blocks: Sequence[CacheBlock]) -> Any:
        kvs = [self._clone_past_key_values(block.kv) for block in blocks]
        first = kvs[0]
        if isinstance(first, DynamicCache):
            layer_data = []
            for layer_idx in range(len(first.layers)):
                keys = [kv.layers[layer_idx].keys for kv in kvs]
                values = [kv.layers[layer_idx].values for kv in kvs]
                layer_data.append((torch.cat(keys, dim=2), torch.cat(values, dim=2)))
            return DynamicCache(ddp_cache_data=layer_data)
        raise TypeError(f"unsupported past_key_values type: {type(first)}")

    def _full_past_key_values(self, input_ids: torch.Tensor) -> Any:
        with torch.no_grad():
            out = self.model(input_ids=input_ids, use_cache=True)
        return out.past_key_values

    def _forward_with_cache_positions(
        self,
        *,
        input_ids: torch.Tensor,
        past_key_values: Any,
        start_position: int,
        attention_mask: torch.Tensor | None = None,
    ) -> Any:
        positions = torch.arange(
            start_position,
            start_position + input_ids.shape[1],
            device=input_ids.device,
            dtype=torch.long,
        )
        kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": True,
        }
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask
        if "position_ids" in self._forward_params:
            kwargs["position_ids"] = positions.unsqueeze(0)
        if "cache_position" in self._forward_params:
            kwargs["cache_position"] = positions
        with torch.no_grad():
            return self.model(**kwargs)

    def _rope_correct_past_key_values(
        self,
        past_key_values: Any,
        source_positions: torch.Tensor,
        target_positions: torch.Tensor,
    ) -> Any:
        # RoPE correction is a pure basis change on cached keys: reuse the same
        # values, but rotate keys from the positions they were prefetched at to
        # the positions they should occupy in the assembled sequence.
        if torch.equal(source_positions, target_positions):
            return self._clone_past_key_values(past_key_values)

        if self._rotary_emb is None:
            raise RuntimeError(f"{self.model_name} does not expose rotary_emb; cannot apply RoPE correction")

        corrected = self._clone_past_key_values(past_key_values)
        position_diff = (target_positions - source_positions).unsqueeze(0)

        # Most rotary modules return cos/sin for supplied positions.
        dummy = torch.zeros(
            1,
            1,
            int(self._rotary_emb.inv_freq.shape[0]) * 2,
            device=target_positions.device,
            dtype=next(self.model.parameters()).dtype,
        )
        cos_diff, sin_diff = self._rotary_emb(dummy, position_diff)
        cos_diff = cos_diff.to(dtype=next(self.model.parameters()).dtype)
        sin_diff = sin_diff.to(dtype=next(self.model.parameters()).dtype)

        if isinstance(corrected, DynamicCache):
            for layer in corrected.layers:
                layer.keys = apply_rotary_pos_emb_to_cache(layer.keys, cos_diff.to(layer.keys.dtype), sin_diff.to(layer.keys.dtype))
            return corrected

        if AMPayload is not None and isinstance(corrected, AMPayload):
            corrected_keys = tuple(
                apply_rotary_pos_emb_to_cache(k, cos_diff.to(k.dtype), sin_diff.to(k.dtype))
                for k in corrected.layer_keys
            )
            return AMPayload(corrected_keys, corrected.layer_values, corrected.layer_beta)

        raise TypeError(f"unsupported past_key_values type: {type(corrected)}")

    def extract_queries(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return Qwen3 queries grouped into KV-head space.

        Qwen3 uses GQA, so several attention heads share one KV head. AM is
        defined over KV heads, so we capture per-layer Q projections and then
        fold the Q heads belonging to the same KV head into one query set.
        """
        if not self._is_qwen3_model():
            raise NotImplementedError("extract_queries only implemented for Qwen3")

        input_ids = input_ids.to(self._runtime_device())
        num_layers = self.model.config.num_hidden_layers
        num_attention_heads = self.model.config.num_attention_heads
        num_kv_heads = self.model.config.num_key_value_heads
        head_dim = getattr(self.model.config, "head_dim",
                           self.model.config.hidden_size // num_attention_heads)
        heads_per_kv_head = num_attention_heads // num_kv_heads

        all_layer_queries: list[torch.Tensor] = []

        def make_hook(layer_idx: int):
            def hook_fn(module, args, kwargs):
                hidden_states = args[0] if args else kwargs.get("hidden_states")
                if hidden_states is None:
                    return
                position_embeddings = kwargs.get("position_embeddings")

                q = module.q_proj(hidden_states)
                batch_size, seq_len, _ = q.shape
                q = q.view(batch_size, seq_len, num_attention_heads, head_dim)
                q = module.q_norm(q)
                q = q.transpose(1, 2)  # (B, num_attention_heads, T, d)

                if position_embeddings is not None:
                    cos, sin = position_embeddings
                    cos = cos.unsqueeze(1)
                    sin = sin.unsqueeze(1)
                    q = (q * cos) + (rotate_half(q) * sin)

                all_layer_queries.append(q[0])  # (num_attention_heads, T, d)

            return hook_fn

        hooks = []
        try:
            for layer_idx in range(num_layers):
                target = self.model.model.layers[layer_idx].self_attn
                handle = target.register_forward_pre_hook(make_hook(layer_idx), with_kwargs=True)
                hooks.append(handle)

            with torch.no_grad():
                self.model(input_ids=input_ids, use_cache=False, return_dict=True)
        finally:
            for h in hooks:
                h.remove()

        queries = torch.stack(all_layer_queries, dim=0)

        # Merge Q heads that share a KV head under GQA.
        if heads_per_kv_head > 1:
            T = queries.shape[2]
            queries = queries.view(num_layers, num_kv_heads, heads_per_kv_head, T, head_dim)
            queries = queries.reshape(num_layers, num_kv_heads, heads_per_kv_head * T, head_dim)

        return queries

    def prefill(self, input_ids: torch.Tensor) -> CacheBlock:
        input_ids = input_ids.to(self._runtime_device())
        kv = self._full_past_key_values(input_ids)
        positions = self._positions_tensor((0, int(input_ids.shape[1])))
        return CacheBlock(
            kv=kv,
            positions=positions,
            rope_positions=positions.clone(),
        )

    def slice(self, block: CacheBlock, start: int, end: int) -> CacheBlock:
        positions = block.positions
        mask = (positions >= start) & (positions < end)
        indices = mask.nonzero(as_tuple=False).flatten()
        sliced_kv = self._slice_past_key_values(block.kv, indices)
        sliced_positions = positions.index_select(0, indices).clone()
        sliced_rope_positions = block.rope_positions.index_select(0, indices).clone()
        return CacheBlock(
            kv=sliced_kv,
            positions=sliced_positions,
            rope_positions=sliced_rope_positions,
            metadata={**dict(block.metadata), "source": "slice"},
        )

    def rope_correct(self, block: CacheBlock) -> CacheBlock:
        corrected_kv = self._rope_correct_past_key_values(block.kv, block.rope_positions, block.positions)
        return CacheBlock(
            kv=corrected_kv,
            positions=block.positions.clone(),
            rope_positions=block.positions.clone(),
            metadata={**dict(block.metadata), "source": "rope_corrected"},
        )

    def evict(self, block: CacheBlock, ratio: float, strategy: str = "uniform") -> CacheBlock:
        indices = compression_indices(block.seq_len, ratio, strategy)
        index_tensor = torch.tensor(indices, device=block.positions.device, dtype=torch.long)
        compressed_kv = self._slice_past_key_values(block.kv, index_tensor)
        compressed_positions = block.positions.index_select(0, index_tensor).clone()
        compressed_rope_positions = block.rope_positions.index_select(0, index_tensor).clone()
        return CacheBlock(
            kv=compressed_kv,
            positions=compressed_positions,
            rope_positions=compressed_rope_positions,
            metadata={
                **dict(block.metadata),
                "source": f"compressed_{strategy}",
                "compression_strategy": strategy,
                "target_ratio": float(ratio),
                "target_count": len(indices),
            },
        )

    def _is_qwen3_model(self) -> bool:
        return isinstance(getattr(self.model, "model", None), modeling_qwen3.Qwen3Model)

    def _prime_prompt_with_context(self, ctx: CacheContext, prompt_text: str) -> _ContextForwardResult:
        ctx.validate()
        q_ids = self._encode_text(prompt_text)
        attn_mask = torch.ones(
            (1, ctx.physical_tokens + q_ids.shape[1]),
            device=self._runtime_device(),
            dtype=torch.long,
        )
        start_position = ctx.next_position
        out = self._forward_with_cache_positions(
            input_ids=q_ids,
            attention_mask=attn_mask,
            past_key_values=self._concat_past_key_values(ctx),
            start_position=start_position,
        )
        return _ContextForwardResult(
            blocks=ctx,
            output=out,
            next_position=start_position + int(q_ids.shape[1]),
        )

    def next_token_logits(self, ctx: CacheContext, prompt_text: str) -> Any:
        if self._has_am_payload(ctx):
            return self._next_token_logits_am(ctx, prompt_text)
        return self._prime_prompt_with_context(ctx, prompt_text).output.logits[:, -1, :].float()

    def _has_am_payload(self, ctx: CacheContext) -> bool:
        """Return whether any block carries an AM payload."""
        if AMPayload is None:
            return False
        return any(isinstance(block.kv, AMPayload) for block in ctx)

    def generate(
        self,
        ctx: CacheContext,
        prompt_text: str,
        config: GenerationConfig,
    ):
        if self._has_am_payload(ctx):
            return self._generate_am_native(ctx, prompt_text, config)
        return self._generate_native(ctx, prompt_text, config)

    def generate_batch(
        self,
        ctx: CacheContext,
        prompts: list[str],
        config: GenerationConfig,
    ) -> list[GenerationResult]:
        if not prompts:
            return []

        buckets: dict[int, list[tuple[int, torch.Tensor]]] = {}
        for idx, prompt_text in enumerate(prompts):
            prompt_ids = self._encode_prompt_ids(prompt_text)
            buckets.setdefault(int(prompt_ids.shape[0]), []).append((idx, prompt_ids))

        results: list[GenerationResult | None] = [None] * len(prompts)
        has_am_payload = self._has_am_payload(ctx)
        for bucket in buckets.values():
            bucket_indices = [idx for idx, _ in bucket]
            q_ids = torch.stack([prompt_ids for _, prompt_ids in bucket], dim=0)
            if has_am_payload:
                bucket_results = self._generate_am_native_batch(ctx, q_ids, config)
            else:
                bucket_results = self._generate_native_batch(ctx, q_ids, config)
            for idx, result in zip(bucket_indices, bucket_results):
                results[idx] = result

        if any(result is None for result in results):
            raise RuntimeError("generate_batch failed to produce a result for every prompt")
        return [result for result in results if result is not None]

    def _repeat_batch_tensor(self, tensor: torch.Tensor, batch_size: int) -> torch.Tensor:
        if tensor.shape[0] == batch_size:
            return tensor.clone()
        if tensor.shape[0] != 1:
            raise ValueError(f"expected cache batch dimension 1, got {tensor.shape[0]}")
        expand_shape = (batch_size, *tensor.shape[1:])
        return tensor.expand(expand_shape).clone()

    def _repeat_dynamic_cache(self, dynamic_kv: DynamicCache, batch_size: int) -> DynamicCache:
        layer_data = []
        for layer in dynamic_kv.layers:
            layer_data.append(
                (
                    self._repeat_batch_tensor(layer.keys, batch_size),
                    self._repeat_batch_tensor(layer.values, batch_size),
                )
            )
        return DynamicCache(ddp_cache_data=layer_data)

    def _dynamic_to_static(self, dynamic_kv: DynamicCache, max_cache_len: int) -> StaticCache:
        """Copy a DynamicCache into a new StaticCache."""
        static = StaticCache(config=self.model.config, max_cache_len=max_cache_len)
        for layer_idx, layer in enumerate(dynamic_kv.layers):
            if not static.layers[layer_idx].is_initialized:
                static.layers[layer_idx].lazy_initialization(layer.keys, layer.values)
            seq_len = layer.keys.shape[2]
            static.layers[layer_idx].keys[:, :, :seq_len, :] = layer.keys
            static.layers[layer_idx].values[:, :, :seq_len, :] = layer.values
        return static

    def _dynamic_to_static_batched(
        self,
        dynamic_kv: DynamicCache,
        batch_size: int,
        max_cache_len: int,
    ) -> StaticCache:
        static = StaticCache(config=self.model.config, max_cache_len=max_cache_len)
        for layer_idx, layer in enumerate(dynamic_kv.layers):
            keys = self._repeat_batch_tensor(layer.keys, batch_size)
            values = self._repeat_batch_tensor(layer.values, batch_size)
            if not static.layers[layer_idx].is_initialized:
                static.layers[layer_idx].lazy_initialization(keys, values)
            seq_len = keys.shape[2]
            static.layers[layer_idx].keys[:, :, :seq_len, :] = keys
            static.layers[layer_idx].values[:, :, :seq_len, :] = values
        return static

    def _build_generation_results(
        self,
        outputs: torch.Tensor,
        *,
        prompt_len: int,
        config: GenerationConfig,
        diagnostics: Mapping[str, Any],
    ) -> list[GenerationResult]:
        eos_token_id = self.tokenizer.eos_token_id
        pad_token_id = self.tokenizer.pad_token_id
        generated = outputs[:, prompt_len:]
        results: list[GenerationResult] = []

        for row in generated:
            token_ids = row.tolist()
            stopped_early = False
            if eos_token_id is not None and eos_token_id in token_ids:
                token_ids = token_ids[: token_ids.index(eos_token_id) + 1]
                stopped_early = True
            elif pad_token_id is not None and pad_token_id in token_ids:
                token_ids = token_ids[: token_ids.index(pad_token_id)]
                stopped_early = True

            results.append(
                GenerationResult(
                    answer_text=self.tokenizer.decode(token_ids, skip_special_tokens=True).strip(),
                    num_generated_tokens=len(token_ids),
                    stopped_early=stopped_early or (len(token_ids) < config.max_new_tokens),
                    diagnostics=dict(diagnostics),
                )
            )

        return results

    def _ensure_compiled(self, warmup_ids: torch.Tensor | None = None, max_new: int = 256) -> Any:
        """Compile the model lazily and warm up CUDA graphs."""
        if self._compiled_model is not None:
            return self._compiled_model

        if self._runtime_device().type != "cuda":
            # CPU falls back to the eager model.
            return self.model

        if os.environ.get("EXPCORE_NO_COMPILE", ""):
            return self.model

        torch.set_float32_matmul_precision("high")
        self._compiled_model = torch.compile(self.model, mode="reduce-overhead", fullgraph=True)

        # Warm up with a short generation to trigger compilation.
        if warmup_ids is not None:
            dummy_cache = StaticCache(
                config=self.model.config,
                max_cache_len=warmup_ids.shape[1] + 64 + max_new,
            )
            with torch.no_grad():
                out = self.model(input_ids=warmup_ids, past_key_values=dummy_cache, use_cache=True)
            attn_mask = torch.ones((1, warmup_ids.shape[1] + 1), device=self._runtime_device(), dtype=torch.long)
            dummy_prompt = warmup_ids[:, :1]
            cache_pos = torch.arange(warmup_ids.shape[1], warmup_ids.shape[1] + 1, device=self._runtime_device())
            for _ in range(3):
                warm_cache = StaticCache(
                    config=self.model.config,
                    max_cache_len=warmup_ids.shape[1] + 64 + max_new,
                )
                with torch.no_grad():
                    self.model(input_ids=warmup_ids, past_key_values=warm_cache, use_cache=True)
                self._compiled_model.generate(
                    input_ids=dummy_prompt,
                    attention_mask=attn_mask,
                    past_key_values=warm_cache,
                    cache_position=cache_pos,
                    max_new_tokens=max_new,
                    do_sample=False,
                )
                torch.cuda.synchronize()
            self._compiled_warmup_done = True

        return self._compiled_model

    def _warmup_compiled_generate(self, compiled_model: Any, past_key_values: Any, batch_size: int = 1) -> None:
        """Run a short warmup generation to capture CUDA graphs."""
        device = self._runtime_device()
        seq_len = past_key_values.get_seq_length() if hasattr(past_key_values, 'get_seq_length') else 0
        dummy_ids = torch.zeros((batch_size, 1), dtype=torch.long, device=device)
        attn_mask = torch.ones((batch_size, seq_len + 1), dtype=torch.long, device=device)
        cache_pos = torch.arange(seq_len, seq_len + 1, device=device)
        with torch.no_grad():
            compiled_model.generate(
                input_ids=dummy_ids,
                attention_mask=attn_mask,
                past_key_values=past_key_values,
                cache_position=cache_pos,
                max_new_tokens=4,
                do_sample=False,
            )
        torch.cuda.synchronize()

    def _ensure_compiled_am(self, max_new: int = 256) -> Any:
        # Reuse one compiled model; compiling twice conflicts with Dynamo state.
        compiled = self._ensure_compiled(max_new=max_new)
        self._compiled_am_model = compiled
        return compiled

    def _generate_native(
        self,
        ctx: CacheContext,
        prompt_text: str,
        config: GenerationConfig,
    ) -> GenerationResult:
        """Generate against an ordinary KV context.

        This path is where logical and physical positions split apart most
        clearly. RoPE should continue from `ctx.next_position`, but writes into
        `StaticCache` should continue from `ctx.physical_tokens`, which may be
        smaller after eviction.
        """
        ctx.validate()
        q_ids = self._encode_text(prompt_text)
        dynamic_kv = self._concat_past_key_values(ctx)
        past_seen_tokens = ctx.physical_tokens
        logical_position = ctx.next_position
        # StaticCache tracks physical rows, not logical positions.
        needed_cache_len = past_seen_tokens + q_ids.shape[1] + config.max_new_tokens

        use_compiled = self._runtime_device().type == "cuda"

        if use_compiled:
            # Keep one fixed cache shape so CUDA graphs remain reusable.
            if needed_cache_len > self._static_max_cache_len:
                self._static_max_cache_len = needed_cache_len
                self._compiled_model = None
                self._compiled_warmup_done = False

            static_kv = self._dynamic_to_static(dynamic_kv, self._static_max_cache_len)
            gen_model = self._ensure_compiled(max_new=config.max_new_tokens)
            past_key_values = static_kv
        else:
            gen_model = self.model
            past_key_values = dynamic_kv

        attn_mask = torch.ones(
            (1, past_seen_tokens + q_ids.shape[1]),
            device=self._runtime_device(),
            dtype=torch.long,
        )
        # position_ids carry the logical RoPE positions.
        position_ids = torch.arange(
            logical_position,
            logical_position + q_ids.shape[1],
            device=self._runtime_device(),
            dtype=torch.long,
        ).unsqueeze(0)
        # cache_position carries the physical write indices.
        cache_position = torch.arange(
            past_seen_tokens,
            past_seen_tokens + q_ids.shape[1],
            device=self._runtime_device(),
            dtype=torch.long,
        )

        generate_kwargs: dict[str, Any] = {
            "input_ids": q_ids,
            "attention_mask": attn_mask,
            "past_key_values": past_key_values,
            "position_ids": position_ids,
            "cache_position": cache_position,
            "max_new_tokens": config.max_new_tokens,
            "do_sample": config.do_sample,
        }
        if config.do_sample:
            generate_kwargs["temperature"] = config.temperature
            generate_kwargs["top_p"] = config.top_p

        with torch.no_grad():
            outputs = gen_model.generate(**generate_kwargs)

        gen_ids = outputs[0, q_ids.shape[1]:]
        num_generated = gen_ids.shape[0]
        answer_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        return GenerationResult(
            answer_text=answer_text,
            num_generated_tokens=int(num_generated),
            stopped_early=num_generated < config.max_new_tokens,
            diagnostics={"backend": self.name, "num_context_blocks": len(ctx)},
        )

    def _generate_native_batch(
        self,
        ctx: CacheContext,
        q_ids: torch.Tensor,
        config: GenerationConfig,
    ) -> list[GenerationResult]:
        ctx.validate()
        batch_size, prompt_len = q_ids.shape
        dynamic_kv = self._concat_past_key_values(ctx)
        past_seen_tokens = ctx.physical_tokens
        logical_position = ctx.next_position
        needed_cache_len = past_seen_tokens + prompt_len + config.max_new_tokens

        use_compiled = self._runtime_device().type == "cuda"
        if use_compiled:
            if needed_cache_len > self._static_max_cache_len:
                self._static_max_cache_len = needed_cache_len
                self._compiled_model = None
                self._compiled_warmup_done = False

            past_key_values = self._dynamic_to_static_batched(dynamic_kv, batch_size, self._static_max_cache_len)
            gen_model = self._ensure_compiled(max_new=config.max_new_tokens)
            if not self._compiled_warmup_done:
                warmup_cache = self._dynamic_to_static_batched(dynamic_kv, batch_size, self._static_max_cache_len)
                self._warmup_compiled_generate(gen_model, warmup_cache, batch_size)
                self._compiled_warmup_done = True
        else:
            past_key_values = self._repeat_dynamic_cache(dynamic_kv, batch_size)
            gen_model = self.model

        attn_mask = torch.ones(
            (batch_size, past_seen_tokens + prompt_len),
            device=self._runtime_device(),
            dtype=torch.long,
        )
        position_ids = torch.arange(
            logical_position,
            logical_position + prompt_len,
            device=self._runtime_device(),
            dtype=torch.long,
        ).unsqueeze(0).expand(batch_size, -1)
        cache_position = torch.arange(
            past_seen_tokens,
            past_seen_tokens + prompt_len,
            device=self._runtime_device(),
            dtype=torch.long,
        )

        generate_kwargs: dict[str, Any] = {
            "input_ids": q_ids,
            "attention_mask": attn_mask,
            "past_key_values": past_key_values,
            "position_ids": position_ids,
            "cache_position": cache_position,
            "max_new_tokens": config.max_new_tokens,
            "do_sample": config.do_sample,
        }
        if config.do_sample:
            generate_kwargs["temperature"] = config.temperature
            generate_kwargs["top_p"] = config.top_p

        with torch.no_grad():
            outputs = gen_model.generate(**generate_kwargs)

        return self._build_generation_results(
            outputs,
            prompt_len=prompt_len,
            config=config,
            diagnostics={
                "backend": self.name,
                "num_context_blocks": len(ctx),
                "batch_size": batch_size,
                "bucket_size": batch_size,
            },
        )
    def _am_payload_to_compacted(self, blocks: Sequence[CacheBlock]) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...]:
        """Flatten mixed AM and ordinary blocks into per-layer tuples."""
        device = self._runtime_device()
        dtype = next(self.model.parameters()).dtype
        num_layers = self.model.config.num_hidden_layers

        all_keys: list[list[torch.Tensor]] = [[] for _ in range(num_layers)]
        all_values: list[list[torch.Tensor]] = [[] for _ in range(num_layers)]
        all_beta: list[list[torch.Tensor]] = [[] for _ in range(num_layers)]

        for block in blocks:
            kv = block.kv
            if AMPayload is not None and isinstance(kv, AMPayload):
                for li in range(num_layers):
                    k = kv.layer_keys[li].to(device=device, dtype=dtype)
                    v = kv.layer_values[li].to(device=device, dtype=dtype)
                    all_keys[li].append(k)
                    all_values[li].append(v)
                    b = kv.layer_beta[li]
                    if b is not None:
                        all_beta[li].append(b.to(device=device, dtype=dtype))
                    else:
                        seq_len = k.shape[2]
                        num_kv_heads = k.shape[1]
                        all_beta[li].append(torch.zeros(num_kv_heads, seq_len, device=device, dtype=dtype))
            elif isinstance(kv, DynamicCache):
                for li in range(num_layers):
                    layer = kv.layers[li]
                    k = layer.keys.to(device=device, dtype=dtype)
                    v = layer.values.to(device=device, dtype=dtype)
                    all_keys[li].append(k)
                    all_values[li].append(v)
                    seq_len = k.shape[2]
                    num_kv_heads = k.shape[1]
                    all_beta[li].append(torch.zeros(num_kv_heads, seq_len, device=device, dtype=dtype))
            else:
                raise TypeError(f"unsupported KV type in AM context: {type(kv)}")

        compacted_layers = []
        for li in range(num_layers):
            keys = torch.cat(all_keys[li], dim=2)
            values = torch.cat(all_values[li], dim=2)
            beta = torch.cat(all_beta[li], dim=-1)
            compacted_layers.append((keys, beta, values))

        return tuple(compacted_layers)

    def _am_to_static_cache(
        self,
        ctx: CacheContext,
        max_cache_len: int,
        *,
        batch_size: int = 1,
    ) -> StaticCompactedPrefixCache:
        compacted = self._am_payload_to_compacted(ctx)
        if batch_size > 1:
            compacted = tuple(
                (
                    self._repeat_batch_tensor(keys, batch_size),
                    self._repeat_batch_tensor(beta.unsqueeze(0), batch_size),
                    self._repeat_batch_tensor(values, batch_size),
                )
                for keys, beta, values in compacted
            )
        return StaticCompactedPrefixCache(
            compacted,
            original_seq_len=ctx.next_position,
            max_cache_len=max_cache_len,
        )

    def _generate_am_native(
        self,
        ctx: CacheContext,
        prompt_text: str,
        config: GenerationConfig,
    ) -> GenerationResult:
        """Generate against an AM context via StaticCompactedPrefixCache."""
        ctx.validate()
        if not self._is_qwen3_model():
            raise NotImplementedError("AM generation is only implemented for Qwen3 models")

        patch_qwen3_for_static_compacted_prefix()
        q_ids = self._encode_text(prompt_text)
        past_seen_tokens = ctx.physical_tokens
        needed_cache_len = past_seen_tokens + q_ids.shape[1] + config.max_new_tokens

        if self._runtime_device().type == "cuda":
            if needed_cache_len > self._am_static_max_cache_len:
                self._am_static_max_cache_len = needed_cache_len
                self._compiled_am_model = None
            past_key_values = self._am_to_static_cache(ctx, self._am_static_max_cache_len)
            gen_model = self._ensure_compiled_am(max_new=config.max_new_tokens)
            if not getattr(self, '_am_compiled_warmup_done', False):
                warmup_cache = self._am_to_static_cache(ctx, self._am_static_max_cache_len)
                self._warmup_compiled_generate(gen_model, warmup_cache)
                self._am_compiled_warmup_done = True
        else:
            past_key_values = self._am_to_static_cache(ctx, needed_cache_len)
            gen_model = self.model

        attn_mask = torch.ones(
            (1, past_seen_tokens + q_ids.shape[1]),
            device=self._runtime_device(),
            dtype=torch.long,
        )
        cache_position = torch.arange(
            past_seen_tokens,
            past_seen_tokens + q_ids.shape[1],
            device=self._runtime_device(),
            dtype=torch.long,
        )

        generate_kwargs: dict[str, Any] = {
            "input_ids": q_ids,
            "attention_mask": attn_mask,
            "past_key_values": past_key_values,
            "cache_position": cache_position,
            "max_new_tokens": config.max_new_tokens,
            "do_sample": config.do_sample,
        }
        if config.do_sample:
            generate_kwargs["temperature"] = config.temperature
            generate_kwargs["top_p"] = config.top_p

        with torch.no_grad():
            outputs = gen_model.generate(**generate_kwargs)

        gen_ids = outputs[0, q_ids.shape[1]:]
        num_generated = gen_ids.shape[0]
        answer_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        return GenerationResult(
            answer_text=answer_text,
            num_generated_tokens=int(num_generated),
            stopped_early=num_generated < config.max_new_tokens,
            diagnostics={"backend": self.name, "num_context_blocks": len(ctx), "am_execution": True},
        )

    def _generate_am_native_batch(
        self,
        ctx: CacheContext,
        q_ids: torch.Tensor,
        config: GenerationConfig,
    ) -> list[GenerationResult]:
        ctx.validate()
        if not self._is_qwen3_model():
            raise NotImplementedError("AM generation is only implemented for Qwen3 models")

        patch_qwen3_for_static_compacted_prefix()
        batch_size, prompt_len = q_ids.shape
        past_seen_tokens = ctx.physical_tokens
        needed_cache_len = past_seen_tokens + prompt_len + config.max_new_tokens

        if self._runtime_device().type == "cuda":
            if needed_cache_len > self._am_static_max_cache_len:
                self._am_static_max_cache_len = needed_cache_len
                self._compiled_am_model = None
            past_key_values = self._am_to_static_cache(ctx, self._am_static_max_cache_len, batch_size=batch_size)
            gen_model = self._ensure_compiled_am(max_new=config.max_new_tokens)
            if not getattr(self, '_am_compiled_warmup_done', False):
                warmup_cache = self._am_to_static_cache(ctx, self._am_static_max_cache_len, batch_size=batch_size)
                self._warmup_compiled_generate(gen_model, warmup_cache, batch_size)
                self._am_compiled_warmup_done = True
        else:
            past_key_values = self._am_to_static_cache(ctx, needed_cache_len, batch_size=batch_size)
            gen_model = self.model

        attn_mask = torch.ones(
            (batch_size, past_seen_tokens + prompt_len),
            device=self._runtime_device(),
            dtype=torch.long,
        )
        cache_position = torch.arange(
            past_seen_tokens,
            past_seen_tokens + prompt_len,
            device=self._runtime_device(),
            dtype=torch.long,
        )

        generate_kwargs: dict[str, Any] = {
            "input_ids": q_ids,
            "attention_mask": attn_mask,
            "past_key_values": past_key_values,
            "cache_position": cache_position,
            "max_new_tokens": config.max_new_tokens,
            "do_sample": config.do_sample,
        }
        if config.do_sample:
            generate_kwargs["temperature"] = config.temperature
            generate_kwargs["top_p"] = config.top_p

        with torch.no_grad():
            outputs = gen_model.generate(**generate_kwargs)

        return self._build_generation_results(
            outputs,
            prompt_len=prompt_len,
            config=config,
            diagnostics={
                "backend": self.name,
                "num_context_blocks": len(ctx),
                "am_execution": True,
                "batch_size": batch_size,
                "bucket_size": batch_size,
            },
        )

    def _next_token_logits_am(self, ctx: CacheContext, prompt_text: str) -> torch.Tensor:
        ctx.validate()
        if not self._is_qwen3_model():
            raise NotImplementedError("AM next_token_logits is only implemented for Qwen3")

        patch_qwen3_for_static_compacted_prefix()
        q_ids = self._encode_text(prompt_text)
        past_seen_tokens = ctx.physical_tokens
        past_key_values = self._am_to_static_cache(ctx, past_seen_tokens + q_ids.shape[1])
        attn_mask = torch.ones(
            (1, past_seen_tokens + q_ids.shape[1]),
            device=self._runtime_device(),
            dtype=torch.long,
        )
        cache_position = torch.arange(
            past_seen_tokens,
            past_seen_tokens + q_ids.shape[1],
            device=self._runtime_device(),
            dtype=torch.long,
        )
        with torch.no_grad():
            outputs = self.model(
                input_ids=q_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                attention_mask=attn_mask,
                use_cache=True,
                return_dict=True,
            )
        return outputs.logits[:, -1, :].float()
    def compute_perplexity(
        self,
        ctx: CacheContext,
        prompt_text: str,
        answer_token_ids: list[int],
    ) -> float:
        """Return answer-token perplexity conditioned on ctx and prompt."""
        ctx.validate()
        device = self._runtime_device()
        dtype = next(self.model.parameters()).dtype

        q_ids_list = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        full_ids = q_ids_list + answer_token_ids
        full_ids_tensor = torch.tensor([full_ids], dtype=torch.long, device=device)
        num_q_tokens = len(q_ids_list)

        if self._has_am_payload(ctx):
            if not self._is_qwen3_model():
                raise NotImplementedError("AM perplexity is only implemented for Qwen3")

            patch_qwen3_for_static_compacted_prefix()
            cache_pos = torch.arange(
                ctx.physical_tokens, ctx.physical_tokens + len(full_ids),
                device=device, dtype=torch.long,
            )
            attn_mask = torch.ones(
                (1, ctx.physical_tokens + len(full_ids)),
                device=device,
                dtype=torch.long,
            )
            past_kv = self._am_to_static_cache(ctx, ctx.physical_tokens + len(full_ids))
            with torch.no_grad():
                outputs = self.model(
                    input_ids=full_ids_tensor,
                    past_key_values=past_kv,
                    cache_position=cache_pos,
                    attention_mask=attn_mask,
                    use_cache=False,
                    return_dict=True,
                )
        else:
            past_kv = self._concat_past_key_values(ctx)
            start_position = ctx.next_position
            # position_ids carry the logical RoPE positions.
            position_ids = torch.arange(
                start_position, start_position + len(full_ids),
                device=device, dtype=torch.long,
            )
            # cache_position matches the actual cache rows.
            cache_pos = torch.arange(
                ctx.physical_tokens, ctx.physical_tokens + len(full_ids),
                device=device, dtype=torch.long,
            )
            attn_mask = torch.ones(
                (1, ctx.physical_tokens + len(full_ids)),
                device=device, dtype=torch.long,
            )
            with torch.no_grad():
                outputs = self.model(
                    input_ids=full_ids_tensor,
                    past_key_values=past_kv,
                    position_ids=position_ids.unsqueeze(0),
                    cache_position=cache_pos,
                    attention_mask=attn_mask,
                    use_cache=False,
                    return_dict=True,
                )

        logits = outputs.logits  # (1, seq_len, vocab_size)
        seq_len = full_ids_tensor.size(1)
        start_idx = max(num_q_tokens - 1, 0)

        answer_logits = logits[:, start_idx:seq_len - 1, :].contiguous()
        answer_labels = full_ids_tensor[:, start_idx + 1:seq_len].contiguous()

        if answer_labels.numel() == 0:
            return float("nan")

        loss = torch.nn.functional.cross_entropy(
            answer_logits.view(-1, answer_logits.size(-1)),
            answer_labels.view(-1),
            reduction="mean",
        )
        return torch.exp(loss).item()

    def compute_perplexity_batch(
        self,
        ctx: CacheContext,
        prompts: list[str],
        answer_token_ids_list: list[list[int]],
    ) -> list[float]:
        """Batch perplexity over multiple prompt/answer pairs."""
        if not prompts:
            return []

        ctx.validate()
        device = self._runtime_device()

        sequences: list[list[int]] = []
        prompt_lengths: list[int] = []
        for prompt_text, answer_ids in zip(prompts, answer_token_ids_list):
            q_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
            sequences.append(q_ids + answer_ids)
            prompt_lengths.append(len(q_ids))

        max_len = max(len(s) for s in sequences)
        batch_size = len(sequences)
        padded = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)
        seq_lengths = []
        for i, s in enumerate(sequences):
            padded[i, :len(s)] = torch.tensor(s, dtype=torch.long, device=device)
            seq_lengths.append(len(s))

        physical = ctx.physical_tokens
        cache_pos = torch.arange(physical, physical + max_len, device=device, dtype=torch.long)
        attn_base = torch.ones(batch_size, physical, device=device, dtype=torch.long)
        seq_mask = torch.zeros(batch_size, max_len, device=device, dtype=torch.long)
        for i, sl in enumerate(seq_lengths):
            seq_mask[i, :sl] = 1
        attn_mask = torch.cat([attn_base, seq_mask], dim=1)

        if self._has_am_payload(ctx):
            patch_qwen3_for_static_compacted_prefix()
            past_kv = self._am_to_static_cache(ctx, physical + max_len, batch_size=batch_size)
        else:
            dynamic_kv = self._concat_past_key_values(ctx)
            past_kv = self._repeat_dynamic_cache(dynamic_kv, batch_size)

        with torch.no_grad():
            outputs = self.model(
                input_ids=padded,
                past_key_values=past_kv,
                cache_position=cache_pos,
                attention_mask=attn_mask,
                use_cache=False,
                return_dict=True,
            )

        logits = outputs.logits  # (B, max_len, vocab)
        results: list[float] = []
        for i in range(batch_size):
            sl = seq_lengths[i]
            start_idx = max(prompt_lengths[i] - 1, 0)
            row_logits = logits[i, start_idx:sl - 1, :].contiguous()
            row_labels = padded[i, start_idx + 1:sl].contiguous()
            if row_labels.numel() == 0:
                results.append(float("nan"))
                continue
            loss = torch.nn.functional.cross_entropy(
                row_logits, row_labels, reduction="mean",
            )
            results.append(torch.exp(loss).item())

        return results
