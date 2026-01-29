"""Tests for KV caching in the transformer backbone.

Tests verify that:
1. CachedTransformerBackbone produces correct outputs
2. KV cache produces same results as full recomputation
3. Generation with cache matches generation without cache
"""

import pytest
import torch

from origami.config import ModelConfig
from origami.model.backbones import (
    CachedMultiHeadAttention,
    CachedTransformerBackbone,
    CachedTransformerBlock,
    KVCache,
)
from origami.tokenizer import JSONTokenizer


class TestKVCache:
    """Tests for KVCache data structure."""

    def test_init_empty(self):
        """Test cache initializes empty."""
        cache = KVCache()
        assert len(cache) == 0
        assert cache.get_seq_len() == 0

    def test_update_single_layer(self):
        """Test updating cache for a single layer."""
        cache = KVCache()
        batch_size, n_heads, seq_len, head_dim = 2, 4, 10, 16

        k = torch.randn(batch_size, n_heads, seq_len, head_dim)
        v = torch.randn(batch_size, n_heads, seq_len, head_dim)

        full_k, full_v = cache.update(0, k, v)

        assert len(cache) == 1
        assert cache.get_seq_len() == seq_len
        assert torch.equal(full_k, k)
        assert torch.equal(full_v, v)

    def test_update_appends(self):
        """Test that update appends new K, V to existing cache."""
        cache = KVCache()
        batch_size, n_heads, head_dim = 2, 4, 16

        # First update
        k1 = torch.randn(batch_size, n_heads, 10, head_dim)
        v1 = torch.randn(batch_size, n_heads, 10, head_dim)
        cache.update(0, k1, v1)

        # Second update (append)
        k2 = torch.randn(batch_size, n_heads, 1, head_dim)
        v2 = torch.randn(batch_size, n_heads, 1, head_dim)
        full_k, full_v = cache.update(0, k2, v2)

        assert cache.get_seq_len() == 11
        assert full_k.shape[2] == 11
        assert full_v.shape[2] == 11
        # Verify concatenation
        assert torch.equal(full_k[:, :, :10, :], k1)
        assert torch.equal(full_k[:, :, 10:, :], k2)

    def test_clear(self):
        """Test clearing the cache."""
        cache = KVCache()
        k = torch.randn(2, 4, 10, 16)
        v = torch.randn(2, 4, 10, 16)
        cache.update(0, k, v)

        cache.clear()
        assert len(cache) == 0
        assert cache.get_seq_len() == 0


class TestCachedMultiHeadAttention:
    """Tests for CachedMultiHeadAttention."""

    @pytest.fixture
    def attention(self):
        return CachedMultiHeadAttention(d_model=64, n_heads=4, dropout=0.0)

    def test_forward_shape(self, attention):
        """Test output shape without caching."""
        batch_size, seq_len = 2, 10
        hidden = torch.randn(batch_size, seq_len, 64)

        output, cache = attention(hidden, use_cache=False)

        assert output.shape == (batch_size, seq_len, 64)
        assert cache is None

    def test_forward_with_cache(self, attention):
        """Test output shape with caching enabled."""
        batch_size, seq_len = 2, 10
        hidden = torch.randn(batch_size, seq_len, 64)
        kv_cache = KVCache()

        output, cache = attention(hidden, kv_cache=kv_cache, layer_idx=0, use_cache=True)

        assert output.shape == (batch_size, seq_len, 64)
        assert cache is not None
        assert len(cache) == 1
        assert cache.get_seq_len() == seq_len

    def test_incremental_matches_full(self, attention):
        """Test that incremental computation matches full computation."""
        torch.manual_seed(42)
        batch_size, seq_len = 2, 10
        hidden = torch.randn(batch_size, seq_len, 64)

        # Full computation
        full_output, _ = attention(hidden, use_cache=False)

        # Incremental computation
        kv_cache = KVCache()
        # First: process prefix (seq_len - 1 tokens)
        prefix_hidden = hidden[:, :-1, :]
        prefix_output, kv_cache = attention(
            prefix_hidden, kv_cache=kv_cache, layer_idx=0, use_cache=True
        )

        # Then: process last token with cache
        last_hidden = hidden[:, -1:, :]
        last_output, _ = attention(last_hidden, kv_cache=kv_cache, layer_idx=0, use_cache=True)

        # Compare outputs for the last position
        assert torch.allclose(full_output[:, -1, :], last_output[:, 0, :], atol=1e-5)


class TestCachedTransformerBlock:
    """Tests for CachedTransformerBlock."""

    @pytest.fixture
    def block(self):
        return CachedTransformerBlock(d_model=64, n_heads=4, d_ff=256, dropout=0.0)

    def test_forward_shape(self, block):
        """Test output shape without caching."""
        batch_size, seq_len = 2, 10
        hidden = torch.randn(batch_size, seq_len, 64)

        output, cache = block(hidden, use_cache=False)

        assert output.shape == (batch_size, seq_len, 64)
        assert cache is None

    def test_forward_with_cache(self, block):
        """Test output shape with caching enabled."""
        batch_size, seq_len = 2, 10
        hidden = torch.randn(batch_size, seq_len, 64)
        kv_cache = KVCache()

        output, cache = block(hidden, kv_cache=kv_cache, layer_idx=0, use_cache=True)

        assert output.shape == (batch_size, seq_len, 64)
        assert cache is not None


class TestCachedTransformerBackbone:
    """Tests for CachedTransformerBackbone."""

    @pytest.fixture
    def config(self):
        return ModelConfig(d_model=64, n_heads=4, n_layers=2, d_ff=256, dropout=0.0)

    @pytest.fixture
    def backbone(self, config):
        return CachedTransformerBackbone(config)

    def test_forward_without_cache(self, backbone):
        """Test forward pass without caching (training mode)."""
        batch_size, seq_len = 2, 10
        hidden = torch.randn(batch_size, seq_len, 64)
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        output = backbone(hidden, attention_mask, use_cache=False)

        # Without cache, returns just tensor
        assert isinstance(output, torch.Tensor)
        assert output.shape == (batch_size, seq_len, 64)

    def test_forward_with_cache(self, backbone):
        """Test forward pass with caching (inference mode)."""
        batch_size, seq_len = 2, 10
        hidden = torch.randn(batch_size, seq_len, 64)
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        output, cache = backbone(hidden, attention_mask, use_cache=True)

        assert output.shape == (batch_size, seq_len, 64)
        assert cache is not None
        assert len(cache) == 2  # 2 layers
        assert cache.get_seq_len() == seq_len

    def test_incremental_generation(self, backbone):
        """Test that incremental generation produces consistent outputs."""
        torch.manual_seed(42)
        batch_size, prefix_len = 2, 10
        hidden = torch.randn(batch_size, prefix_len + 3, 64)
        attention_mask = torch.ones(batch_size, prefix_len + 3, dtype=torch.bool)

        # Full forward pass
        full_output = backbone(hidden, attention_mask, use_cache=False)

        # Incremental: first process prefix
        prefix_hidden = hidden[:, :prefix_len, :]
        prefix_mask = attention_mask[:, :prefix_len]
        _, cache = backbone(prefix_hidden, prefix_mask, use_cache=True)

        # Then add tokens one by one
        for i in range(3):
            new_hidden = hidden[:, prefix_len + i : prefix_len + i + 1, :]
            output, cache = backbone(new_hidden, past_key_values=cache, use_cache=True)

        # Compare last token output
        assert torch.allclose(full_output[:, -1, :], output[:, 0, :], atol=1e-4), (
            "Incremental output should match full computation"
        )

    def test_with_padding(self, backbone):
        """Test with left-padded sequences."""
        batch_size, seq_len = 2, 10
        hidden = torch.randn(batch_size, seq_len, 64)

        # Left-padded: first 3 positions are padding
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        attention_mask[:, :3] = False

        output, cache = backbone(hidden, attention_mask, use_cache=True)

        assert output.shape == (batch_size, seq_len, 64)
        assert cache is not None
        # Cache should have full sequence length (padding handled by attention mask)
        assert cache.get_seq_len() == seq_len


class TestKVCacheIntegration:
    """Integration tests for KV caching with full model."""

    @pytest.fixture
    def tokenizer(self):
        tokenizer = JSONTokenizer()
        tokenizer.fit([{"a": 1, "b": 2}, {"x": "hello", "y": "world"}])
        return tokenizer

    @pytest.fixture
    def model_config(self):
        return ModelConfig(
            d_model=64,
            n_heads=4,
            n_layers=2,
            d_ff=256,
            dropout=0.0,
            backbone="cached_transformer",
            # Note: Grammar constraints are now handled by trainer, not model config
        )

    def test_model_with_cached_backbone(self, tokenizer, model_config):
        """Test that model works with cached backbone."""
        from origami.model import OrigamiModel

        model = OrigamiModel(model_config, tokenizer.vocab)
        model.eval()

        # Create simple batch
        batch_size, seq_len = 2, 8
        input_ids = torch.randint(0, len(tokenizer.vocab), (batch_size, seq_len))
        path_types = torch.zeros(batch_size, seq_len, model_config.max_depth, dtype=torch.long)
        path_ids = torch.zeros(batch_size, seq_len, model_config.max_depth, dtype=torch.long)
        path_lengths = torch.zeros(batch_size, seq_len, dtype=torch.long)
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        # Forward without cache
        output_no_cache = model(
            input_ids=input_ids,
            path_types=path_types,
            path_ids=path_ids,
            path_lengths=path_lengths,
            attention_mask=attention_mask,
            use_cache=False,
        )

        assert output_no_cache.logits.shape == (batch_size, seq_len, len(tokenizer.vocab))
        assert output_no_cache.past_key_values is None

        # Forward with cache
        output_with_cache = model(
            input_ids=input_ids,
            path_types=path_types,
            path_ids=path_ids,
            path_lengths=path_lengths,
            attention_mask=attention_mask,
            use_cache=True,
        )

        assert output_with_cache.logits.shape == (batch_size, seq_len, len(tokenizer.vocab))
        assert output_with_cache.past_key_values is not None

        # Outputs should be the same
        assert torch.allclose(output_no_cache.logits, output_with_cache.logits, atol=1e-5), (
            "Cached and non-cached outputs should match"
        )

    def test_generator_kv_cache_detection(self, tokenizer, model_config):
        """Test that Generator correctly detects KV cache support."""
        from origami.inference.generator import OrigamiGenerator
        from origami.model import OrigamiModel

        # With cached backbone
        model_cached = OrigamiModel(model_config, tokenizer.vocab)
        generator_cached = OrigamiGenerator(model_cached, tokenizer)
        assert generator_cached._supports_kv_cache is True

        # With standard backbone
        config_standard = ModelConfig(
            d_model=64,
            n_heads=4,
            n_layers=2,
            backbone="transformer",
        )
        model_standard = OrigamiModel(config_standard, tokenizer.vocab)
        generator_standard = OrigamiGenerator(model_standard, tokenizer)
        assert generator_standard._supports_kv_cache is False

    def test_left_padding_no_nan(self, tokenizer, model_config):
        """Test that left-padded sequences don't produce NaN outputs.

        This is a regression test for the critical fix where padding positions
        at the start of left-padded sequences need to attend to themselves
        to avoid softmax(all -inf) = NaN.
        """
        from origami.model import OrigamiModel

        model = OrigamiModel(model_config, tokenizer.vocab)
        model.eval()

        # Create batch with significant left-padding
        batch_size, seq_len = 4, 20
        input_ids = torch.randint(0, len(tokenizer.vocab), (batch_size, seq_len))
        path_types = torch.zeros(batch_size, seq_len, model_config.max_depth, dtype=torch.long)
        path_ids = torch.zeros(batch_size, seq_len, model_config.max_depth, dtype=torch.long)
        path_lengths = torch.zeros(batch_size, seq_len, dtype=torch.long)

        # Create attention mask with variable padding per sequence
        # This mimics real batched prediction scenarios
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        attention_mask[0, :5] = False  # 5 padding tokens
        attention_mask[1, :10] = False  # 10 padding tokens
        attention_mask[2, :15] = False  # 15 padding tokens
        attention_mask[3, :] = True  # No padding

        output = model(
            input_ids=input_ids,
            path_types=path_types,
            path_ids=path_ids,
            path_lengths=path_lengths,
            attention_mask=attention_mask,
            use_cache=False,
        )

        # Check no NaN in logits
        assert not torch.isnan(output.logits).any(), "Logits should not contain NaN"
        assert not torch.isinf(output.logits).any(), "Logits should not contain Inf"

        # Also test with cache enabled (inference mode)
        output_cached = model(
            input_ids=input_ids,
            path_types=path_types,
            path_ids=path_ids,
            path_lengths=path_lengths,
            attention_mask=attention_mask,
            use_cache=True,
        )

        assert not torch.isnan(output_cached.logits).any(), "Cached logits should not contain NaN"
        assert not torch.isinf(output_cached.logits).any(), "Cached logits should not contain Inf"
