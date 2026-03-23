"""Static compacted-prefix cache for Qwen3 AM execution.

This is a backend runtime shim, not part of the core cache algebra. It teaches
HF's Qwen3 stack how to run against compacted `(K, beta, V)` prefixes by
storing beta alongside the cache, patching attention to add that bias before
softmax, and separating physical cache positions from the logical RoPE basis.
"""
from __future__ import annotations

from typing import Any

import torch
from transformers.cache_utils import Cache, CacheLayerMixin
from transformers.cache_utils import is_torchdynamo_compiling
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.models.qwen3 import modeling_qwen3


def _repeat_kv_for_beta(beta: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat KV-head beta biases across attention groups."""
    batch, num_key_value_heads, seq_len = beta.shape
    if n_rep == 1:
        return beta
    expanded = beta[:, :, None, :].expand(batch, num_key_value_heads, n_rep, seq_len)
    return expanded.reshape(batch, num_key_value_heads * n_rep, seq_len)


class StaticCompactedPrefixLayer(CacheLayerMixin):
    """Compile-friendly static cache layer for compacted prefixes."""

    is_compileable = True
    is_sliding = False

    def __init__(
        self,
        keys: torch.Tensor,
        beta: torch.Tensor,
        values: torch.Tensor,
        *,
        max_cache_len: int,
        clone: bool = False,
    ) -> None:
        super().__init__()
        if keys.ndim != 4 or values.ndim != 4:
            raise ValueError("keys and values must have shape (B, H, T, d)")
        if beta.ndim != 3:
            raise ValueError("beta must have shape (B, H, T)")
        if keys.shape != values.shape:
            raise ValueError("keys and values must share shape")
        if beta.shape[:2] != keys.shape[:2] or beta.shape[2] != keys.shape[2]:
            raise ValueError("beta must align with keys along (B, H, T)")
        if max_cache_len < keys.shape[2]:
            raise ValueError("max_cache_len must be at least the compacted prefix length")

        if clone:
            keys = keys.clone()
            beta = beta.clone()
            values = values.clone()

        self.dtype = keys.dtype
        self.device = keys.device
        self.max_batch_size = keys.shape[0]
        self.num_heads = keys.shape[1]
        self.k_head_dim = keys.shape[-1]
        self.v_head_dim = values.shape[-1]
        self.base_len = int(keys.shape[2])
        self.max_cache_len = int(max_cache_len)
        self.current_length = self.base_len

        self.keys = torch.zeros(
            (self.max_batch_size, self.num_heads, self.max_cache_len, self.k_head_dim),
            dtype=self.dtype,
            device=self.device,
        )
        self.values = torch.zeros(
            (self.max_batch_size, self.num_heads, self.max_cache_len, self.v_head_dim),
            dtype=self.dtype,
            device=self.device,
        )
        self.beta = torch.zeros(
            (self.max_batch_size, self.num_heads, self.max_cache_len),
            dtype=beta.dtype,
            device=beta.device,
        )

        self.keys[:, :, : self.base_len, :] = keys
        self.values[:, :, : self.base_len, :] = values
        self.beta[:, :, : self.base_len] = beta

        if not is_torchdynamo_compiling():
            torch._dynamo.mark_static_address(self.keys)
            torch._dynamo.mark_static_address(self.values)
            torch._dynamo.mark_static_address(self.beta)

        self.is_initialized = True

    def lazy_initialization(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        return None

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cache_position = cache_kwargs.get("cache_position") if cache_kwargs is not None else None
        cache_position = (
            cache_position if cache_position is not None else torch.arange(key_states.shape[-2], device=self.device)
        )

        try:
            self.keys.index_copy_(2, cache_position, key_states)
            self.values.index_copy_(2, cache_position, value_states)
        except NotImplementedError:
            self.keys[:, :, cache_position] = key_states
            self.values[:, :, cache_position] = value_states

        # Avoid Python scalar extraction in compiled code.
        if not is_torchdynamo_compiling():
            self.current_length = max(self.current_length, int(cache_position[-1]) + 1)
        return self.keys, self.values

    def get_mask_sizes(self, cache_position: torch.Tensor) -> tuple[int, int]:
        return self.max_cache_len, 0

    def get_seq_length(self) -> int:
        return self.current_length

    def get_max_cache_shape(self) -> int:
        return self.max_cache_len


class StaticCompactedPrefixCache(Cache):
    """Static cache carrying compacted K/V prefixes plus per-position beta."""

    def __init__(
        self,
        compacted_cache: tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...],
        *,
        original_seq_len: int,
        max_cache_len: int,
        clone: bool = False,
    ) -> None:
        if not compacted_cache:
            raise ValueError("compacted_cache must contain at least one layer")

        prefix_lens = [int(keys.shape[2]) for keys, _, _ in compacted_cache]
        max_prefix_len = max(prefix_lens)
        if max_cache_len < max_prefix_len:
            raise ValueError("max_cache_len must be at least the maximum compacted prefix length")

        layers = []
        for keys, beta, values in compacted_cache:
            beta_batched = beta.unsqueeze(0) if beta.ndim == 2 else beta
            layers.append(
                StaticCompactedPrefixLayer(
                    keys,
                    beta_batched,
                    values,
                    max_cache_len=max_cache_len,
                    clone=clone,
                )
            )

        super().__init__(layers=layers)
        self._rope_base = int(original_seq_len) - int(max_prefix_len)

    def rope_base(self) -> int:
        return self._rope_base

    def beta_for_layer(self, layer_idx: int) -> torch.Tensor:
        layer = self.layers[layer_idx]
        return layer.beta

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if not self.layers:
            return 0
        return max(layer.get_seq_length() for layer in self.layers)


def patch_qwen3_for_static_compacted_prefix() -> None:
    """Patch Qwen3 to consume StaticCompactedPrefixCache.

    Two things change relative to the stock path:
    1. attention adds the per-position AM beta term before softmax
    2. masking still uses physical cache positions, while RoPE is shifted back
       into the logical sequence via `rope_base()`
    """

    if getattr(modeling_qwen3.Qwen3Attention, "_static_compacted_prefix_patched", False):
        return

    original_attention_forward = modeling_qwen3.Qwen3Attention.forward
    original_model_forward = modeling_qwen3.Qwen3Model.forward

    def patched_attention_forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if not isinstance(past_key_values, StaticCompactedPrefixCache):
            return original_attention_forward(
                self,
                hidden_states,
                position_embeddings,
                attention_mask,
                past_key_values=past_key_values,
                cache_position=cache_position,
                **kwargs,
            )

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = modeling_qwen3.apply_rotary_pos_emb(query_states, key_states, cos, sin)

        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        beta_kv = past_key_values.beta_for_layer(self.layer_idx)
        beta_heads = _repeat_kv_for_beta(beta_kv, self.num_key_value_groups)

        if attention_mask is not None:
            modified_attention_mask = attention_mask
            if modified_attention_mask.dtype == torch.bool:
                finfo = torch.finfo(query_states.dtype)
                modified_attention_mask = torch.where(
                    modified_attention_mask,
                    torch.zeros(1, device=modified_attention_mask.device, dtype=query_states.dtype),
                    torch.full((1,), finfo.min, device=modified_attention_mask.device, dtype=query_states.dtype),
                )
            else:
                modified_attention_mask = modified_attention_mask.to(dtype=query_states.dtype)

            if modified_attention_mask.shape[1] == 1 and beta_heads.shape[1] > 1:
                modified_attention_mask = modified_attention_mask.expand(
                    beta_heads.shape[0],
                    beta_heads.shape[1],
                    modified_attention_mask.shape[2],
                    modified_attention_mask.shape[3],
                )
            modified_attention_mask = modified_attention_mask.clone()
        else:
            modified_attention_mask = torch.zeros(
                beta_heads.shape[0],
                beta_heads.shape[1],
                query_states.shape[2],
                key_states.shape[2],
                dtype=query_states.dtype,
                device=query_states.device,
            )

        modified_attention_mask = modified_attention_mask + beta_heads.unsqueeze(2)

        attention_interface = modeling_qwen3.ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation,
            modeling_qwen3.eager_attention_forward,
        )
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            modified_attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    def patched_model_forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Any,
    ):
        if not isinstance(past_key_values, StaticCompactedPrefixCache):
            return original_model_forward(
                self,
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length()
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        # Keep physical positions for masking and shift RoPE back to logical space.
        position_ids = cache_position.unsqueeze(0)
        rope_position_ids = position_ids + past_key_values.rope_base()

        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

            for mask_type, mask in causal_mask_mapping.items():
                if mask is not None and mask.dtype == torch.bool:
                    float_mask = torch.zeros_like(mask, dtype=inputs_embeds.dtype)
                    float_mask.masked_fill_(~mask, float("-inf"))
                    causal_mask_mapping[mask_type] = float_mask

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, rope_position_ids)

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return modeling_qwen3.BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )

    modeling_qwen3.Qwen3Attention.forward = patched_attention_forward
    modeling_qwen3.Qwen3Model.forward = patched_model_forward
    modeling_qwen3.Qwen3Attention._static_compacted_prefix_patched = True
