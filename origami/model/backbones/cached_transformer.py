"""Transformer backbone with KV caching for efficient generation.

This module provides a transformer backbone that supports KV caching,
enabling O(1) per-token generation instead of O(n) when generating sequences.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from origami.config import ModelConfig

from .base import BackboneBase


class KVCache:
    """Dynamic KV cache that grows as tokens are generated.

    Stores key and value tensors for each layer, enabling efficient
    incremental generation without recomputing attention for past tokens.
    """

    def __init__(self) -> None:
        """Initialize empty cache."""
        self._cache: list[tuple[Tensor, Tensor]] = []

    def __len__(self) -> int:
        """Return number of layers in cache."""
        return len(self._cache)

    def __getitem__(self, layer_idx: int) -> tuple[Tensor, Tensor]:
        """Get cached K, V for a layer."""
        return self._cache[layer_idx]

    def update(self, layer_idx: int, key: Tensor, value: Tensor) -> tuple[Tensor, Tensor]:
        """Update cache for a layer, appending new K, V to existing.

        Args:
            layer_idx: Layer index
            key: New key states (batch, num_heads, new_seq_len, head_dim)
            value: New value states (batch, num_heads, new_seq_len, head_dim)

        Returns:
            Tuple of (full_key, full_value) including cached and new states
        """
        if layer_idx < len(self._cache):
            # Append to existing cache
            cached_key, cached_value = self._cache[layer_idx]
            key = torch.cat([cached_key, key], dim=2)
            value = torch.cat([cached_value, value], dim=2)
            self._cache[layer_idx] = (key, value)
        else:
            # First time for this layer
            self._cache.append((key, value))

        return key, value

    def get_seq_len(self) -> int:
        """Get current cached sequence length."""
        if not self._cache:
            return 0
        return self._cache[0][0].size(2)

    def clear(self) -> None:
        """Clear all cached values."""
        self._cache.clear()


class CachedMultiHeadAttention(nn.Module):
    """Multi-head attention with explicit Q, K, V projections and KV caching.

    Unlike nn.MultiheadAttention, this exposes K, V tensors for caching.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        """Initialize attention module.

        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Separate projections for Q, K, V
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states: Tensor,  # (batch, seq_len, d_model)
        attention_mask: Tensor | None = None,  # (batch, 1, seq_len, key_len) or None
        kv_cache: KVCache | None = None,
        layer_idx: int = 0,
        use_cache: bool = False,
    ) -> tuple[Tensor, KVCache | None]:
        """Apply multi-head attention with optional KV caching.

        Args:
            hidden_states: Input tensor (batch, seq_len, d_model)
            attention_mask: Attention mask (batch, 1, seq_len, key_len)
                where True/1 means attend, False/0 means mask out.
                If None, no masking is applied.
            kv_cache: Optional KV cache from previous steps
            layer_idx: Index of this layer (for cache lookup)
            use_cache: Whether to update and return cache

        Returns:
            Tuple of (output, updated_cache). Cache is None if use_cache=False.
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Project to Q, K, V
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        # Reshape to (batch, n_heads, seq_len, head_dim)
        query = query.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # Update cache if provided
        if kv_cache is not None and layer_idx < len(kv_cache):
            # Append new K, V to cached values
            key, value = kv_cache.update(layer_idx, key, value)
        elif use_cache and kv_cache is not None:
            # First time caching for this layer
            key, value = kv_cache.update(layer_idx, key, value)

        # Compute attention using scaled_dot_product_attention
        # Uses Flash Attention on CUDA, memory-efficient attention on other backends
        # query: (batch, n_heads, q_len, head_dim)
        # key: (batch, n_heads, kv_len, head_dim)
        # value: (batch, n_heads, kv_len, head_dim)

        # SDPA boolean mask convention: True = attend, False = mask out
        # (same as our convention, no inversion needed)
        attn_output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=False,  # Causality is already encoded in the mask
        )
        # attn_output: (batch, n_heads, q_len, head_dim)

        # Reshape back to (batch, seq_len, d_model)
        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        )

        # Output projection
        attn_output = self.out_proj(attn_output)

        return attn_output, kv_cache if use_cache else None


class CachedTransformerBlock(nn.Module):
    """Single transformer block with attention + FFN and KV caching support.

    Uses pre-norm architecture (layer norm before attention/FFN) for stability.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        """Initialize transformer block.

        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            d_ff: Feed-forward hidden dimension
            dropout: Dropout probability
        """
        super().__init__()

        # Pre-norm architecture
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Self-attention with caching
        self.self_attn = CachedMultiHeadAttention(d_model, n_heads, dropout)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
        kv_cache: KVCache | None = None,
        layer_idx: int = 0,
        use_cache: bool = False,
    ) -> tuple[Tensor, KVCache | None]:
        """Apply transformer block with optional KV caching.

        Args:
            hidden_states: Input tensor (batch, seq_len, d_model)
            attention_mask: Attention mask for causal + padding masking
            kv_cache: Optional KV cache
            layer_idx: Index of this layer
            use_cache: Whether to use/update cache

        Returns:
            Tuple of (output, cache)
        """
        # Pre-norm self-attention with residual
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states, kv_cache = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            kv_cache=kv_cache,
            layer_idx=layer_idx,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # Pre-norm FFN with residual
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = residual + self.ffn(hidden_states)

        return hidden_states, kv_cache


class CachedTransformerBackbone(BackboneBase):
    """Transformer backbone with optional KV caching for efficient generation.

    This backbone is functionally equivalent to TransformerBackbone but supports
    KV caching for O(1) per-token generation instead of O(n).

    When use_cache=False (default), behaves identically to TransformerBackbone.
    When use_cache=True, caches K,V projections and returns them for reuse.
    """

    def __init__(self, config: ModelConfig):
        """Initialize cached transformer backbone.

        Args:
            config: Model configuration
        """
        super().__init__()

        self.config = config

        # Stack of transformer blocks
        self.layers = nn.ModuleList(
            [
                CachedTransformerBlock(
                    d_model=config.d_model,
                    n_heads=config.n_heads,
                    d_ff=config.d_ff,
                    dropout=config.dropout,
                )
                for _ in range(config.n_layers)
            ]
        )

        # Final layer norm
        self.norm = nn.LayerNorm(config.d_model)

    def forward(
        self,
        hidden_states: Tensor,  # (batch, seq_len, d_model)
        attention_mask: Tensor | None = None,  # (batch, seq_len)
        past_key_values: KVCache | None = None,
        use_cache: bool = False,
    ) -> Tensor | tuple[Tensor, KVCache]:
        """Apply transformer with optional KV caching.

        Args:
            hidden_states: Input embeddings (batch, seq_len, d_model)
            attention_mask: Boolean mask where True = valid, False = padding.
                Shape (batch, seq_len).
            past_key_values: Cached K,V from previous generation steps
            use_cache: Whether to compute and return KV cache

        Returns:
            If use_cache=False: hidden_states tensor
            If use_cache=True: tuple of (hidden_states, kv_cache)
        """
        batch_size, seq_len, _ = hidden_states.shape
        device = hidden_states.device

        # Determine past sequence length from cache
        past_len = past_key_values.get_seq_len() if past_key_values else 0
        total_len = past_len + seq_len

        # Build causal attention mask
        # Shape: (batch, 1, seq_len, total_len)
        causal_mask = self._make_causal_mask(seq_len, total_len, device)

        # Apply padding mask if provided
        if attention_mask is not None:
            # For incremental generation, attention_mask should cover full sequence
            # If it only covers new tokens, we assume past tokens were all valid
            if attention_mask.size(1) == seq_len and past_len > 0:
                # Extend with True for past positions
                past_mask = torch.ones(batch_size, past_len, dtype=torch.bool, device=device)
                full_mask = torch.cat([past_mask, attention_mask], dim=1)
            else:
                full_mask = attention_mask

            # Expand to (batch, 1, 1, total_len) for broadcasting
            key_padding_mask = full_mask.unsqueeze(1).unsqueeze(2)
            # Combine with causal mask
            causal_mask = causal_mask & key_padding_mask

            # Critical fix for left-padding: Allow padding query positions to attend
            # to themselves to avoid softmax(all -inf) = NaN. For left-padded sequences,
            # padding positions at the start have no valid keys to attend to (all
            # previous positions are also padding). This causes NaN which corrupts
            # the output. The fix ensures padding positions can attend to at least
            # one position (themselves), making softmax well-defined.
            #
            # Note: This only applies when processing the full sequence (not incremental),
            # since incremental steps only have 1 query token at a valid position.
            if past_len == 0 and seq_len > 1:
                # Get padding positions for queries (current tokens being processed)
                # attention_mask shape: (batch, seq_len), True = valid
                query_padding = ~attention_mask  # (batch, seq_len), True for padding

                # For each padding query position i, allow attending to key position i
                # causal_mask shape: (1, 1, seq_len, total_len)
                # We need to unmask diagonal positions for padding queries
                # Expand to (batch, 1, seq_len, total_len) first
                causal_mask = causal_mask.expand(batch_size, -1, -1, -1).clone()

                # Set mask[b, 0, i, i] = True for padding positions
                diag_indices = torch.arange(seq_len, device=device)
                # query_padding: (batch, seq_len) -> need to set causal_mask[b, 0, i, i]
                # for positions where query_padding[b, i] is True
                causal_mask[:, 0, diag_indices, diag_indices] = (
                    causal_mask[:, 0, diag_indices, diag_indices] | query_padding
                )

        # Initialize cache if needed
        if use_cache and past_key_values is None:
            past_key_values = KVCache()

        # Apply transformer layers
        for layer_idx, layer in enumerate(self.layers):
            hidden_states, past_key_values = layer(
                hidden_states,
                attention_mask=causal_mask,
                kv_cache=past_key_values,
                layer_idx=layer_idx,
                use_cache=use_cache,
            )

        # Final layer norm
        hidden_states = self.norm(hidden_states)

        if use_cache:
            return hidden_states, past_key_values
        return hidden_states

    def _make_causal_mask(self, query_len: int, key_len: int, device: torch.device) -> Tensor:
        """Create causal attention mask.

        Args:
            query_len: Length of query sequence (new tokens)
            key_len: Length of key sequence (past + new tokens)
            device: Device for tensor

        Returns:
            Boolean mask of shape (1, 1, query_len, key_len)
            True = can attend, False = cannot attend
        """
        # For incremental generation:
        # - query positions are the new tokens (at the end)
        # - key positions include past cached tokens + new tokens
        # - Each query position can attend to all past tokens and itself

        # Create indices
        # query positions: [key_len - query_len, key_len - query_len + 1, ..., key_len - 1]
        query_pos = torch.arange(key_len - query_len, key_len, device=device)
        key_pos = torch.arange(key_len, device=device)

        # Causal: query[i] can attend to key[j] if j <= query_pos[i]
        # query_pos: (query_len,) -> (query_len, 1)
        # key_pos: (key_len,) -> (1, key_len)
        mask = key_pos.unsqueeze(0) <= query_pos.unsqueeze(1)

        # Reshape to (1, 1, query_len, key_len) for broadcasting
        return mask.unsqueeze(0).unsqueeze(0)
