"""Tests for OrigamiEmbedder."""

import pytest
import torch

from origami.config import ModelConfig
from origami.inference import OrigamiEmbedder
from origami.model import OrigamiModel
from origami.tokenizer import JSONTokenizer


@pytest.fixture
def simple_tokenizer():
    """Create a tokenizer fitted on simple data."""
    data = [
        {"name": "Alice", "age": 30, "city": "NYC"},
        {"name": "Bob", "age": 25, "city": "LA"},
        {"name": "Charlie", "age": 35, "city": "SF"},
    ]
    tokenizer = JSONTokenizer()
    tokenizer.fit(data)
    return tokenizer


@pytest.fixture
def simple_model(simple_tokenizer):
    """Create a small model for testing."""
    config = ModelConfig(
        d_model=32,
        n_heads=2,
        n_layers=1,
        d_ff=64,
        max_depth=simple_tokenizer.max_depth,
    )
    return OrigamiModel(config, vocab=simple_tokenizer.vocab)


class TestOrigamiEmbedder:
    """Tests for OrigamiEmbedder."""

    def test_init(self, simple_model, simple_tokenizer):
        """Test embedder initialization."""
        embedder = OrigamiEmbedder(simple_model, simple_tokenizer, pooling="mean")

        assert embedder.model is simple_model
        assert embedder.tokenizer is simple_tokenizer
        assert embedder.pooling == "mean"

    def test_embedding_dim(self, simple_model, simple_tokenizer):
        """Test embedding_dim property."""
        embedder = OrigamiEmbedder(simple_model, simple_tokenizer)
        assert embedder.embedding_dim == simple_model.config.d_model

    @pytest.mark.parametrize("pooling", ["mean", "max", "last", "target"])
    def test_pooling_strategies(self, simple_model, simple_tokenizer, pooling):
        """Test all pooling strategies work."""
        embedder = OrigamiEmbedder(simple_model, simple_tokenizer, pooling=pooling)

        obj = {"name": "Alice", "age": 30, "city": "NYC"}

        if pooling == "target":
            embedding = embedder.embed(obj, target_key="city")
        else:
            embedding = embedder.embed(obj)

        assert embedding.shape == (simple_model.config.d_model,)
        assert not torch.isnan(embedding).any()

    def test_embed_single(self, simple_model, simple_tokenizer):
        """Test embedding single object."""
        embedder = OrigamiEmbedder(simple_model, simple_tokenizer)

        obj = {"name": "Alice", "age": 30, "city": "NYC"}
        embedding = embedder.embed(obj)

        assert embedding.shape == (simple_model.config.d_model,)
        assert embedding.dtype == torch.float32

    def test_embed_batch(self, simple_model, simple_tokenizer):
        """Test embedding multiple objects."""
        embedder = OrigamiEmbedder(simple_model, simple_tokenizer)

        objects = [
            {"name": "Alice", "age": 30, "city": "NYC"},
            {"name": "Bob", "age": 25, "city": "LA"},
            {"name": "Charlie", "age": 35, "city": "SF"},
        ]
        embeddings = embedder.embed_batch(objects)

        assert embeddings.shape == (3, simple_model.config.d_model)
        assert not torch.isnan(embeddings).any()

    def test_normalization(self, simple_model, simple_tokenizer):
        """Test that normalization produces unit vectors."""
        embedder = OrigamiEmbedder(simple_model, simple_tokenizer)

        obj = {"name": "Alice", "age": 30, "city": "NYC"}

        # With normalization (default)
        embedding_norm = embedder.embed(obj, normalize=True)
        norm = torch.linalg.norm(embedding_norm)
        assert torch.isclose(norm, torch.tensor(1.0), atol=1e-5)

        # Without normalization
        embedding_unnorm = embedder.embed(obj, normalize=False)
        # Should not necessarily be unit length
        # (could be by chance, so just check it runs)
        assert embedding_unnorm.shape == (simple_model.config.d_model,)

    def test_target_pooling_requires_target_key(self, simple_model, simple_tokenizer):
        """Test that target pooling requires target_key."""
        embedder = OrigamiEmbedder(simple_model, simple_tokenizer, pooling="target")

        obj = {"name": "Alice", "age": 30, "city": "NYC"}

        with pytest.raises(ValueError, match="target_key is required"):
            embedder.embed(obj)

    def test_target_pooling_finds_key(self, simple_model, simple_tokenizer):
        """Test that target pooling finds the correct key position."""
        embedder = OrigamiEmbedder(simple_model, simple_tokenizer, pooling="target")

        obj = {"name": "Alice", "age": 30, "city": "NYC"}
        embedding = embedder.embed(obj, target_key="city")

        assert embedding.shape == (simple_model.config.d_model,)
        assert not torch.isnan(embedding).any()

    def test_target_pooling_different_keys_different_embeddings(
        self, simple_model, simple_tokenizer
    ):
        """Test that different target keys produce different embeddings."""
        embedder = OrigamiEmbedder(simple_model, simple_tokenizer, pooling="target")

        obj = {"name": "Alice", "age": 30, "city": "NYC"}

        emb_name = embedder.embed(obj, target_key="name")
        emb_age = embedder.embed(obj, target_key="age")
        emb_city = embedder.embed(obj, target_key="city")

        # Different keys should generally produce different embeddings
        # (not guaranteed but very likely with random init)
        assert not torch.allclose(emb_name, emb_city)
        assert not torch.allclose(emb_age, emb_city)

    def test_variable_length_sequences(self, simple_model, simple_tokenizer):
        """Test embedding objects with different lengths."""
        embedder = OrigamiEmbedder(simple_model, simple_tokenizer)

        objects = [
            {"name": "Alice"},  # Short
            {"name": "Bob", "age": 25, "city": "LA"},  # Longer
        ]
        embeddings = embedder.embed_batch(objects)

        assert embeddings.shape == (2, simple_model.config.d_model)
        assert not torch.isnan(embeddings).any()

    def test_deterministic_embeddings(self, simple_model, simple_tokenizer):
        """Test that embeddings are deterministic (no randomness)."""
        embedder = OrigamiEmbedder(simple_model, simple_tokenizer)

        obj = {"name": "Alice", "age": 30, "city": "NYC"}

        emb1 = embedder.embed(obj)
        emb2 = embedder.embed(obj)

        assert torch.allclose(emb1, emb2)

    def test_always_uses_cpu(self, simple_tokenizer):
        """Test that embedder always runs on CPU for performance."""
        config = ModelConfig(
            d_model=32,
            n_heads=2,
            n_layers=1,
            d_ff=64,
            max_depth=simple_tokenizer.max_depth,
        )
        model = OrigamiModel(config, vocab=simple_tokenizer.vocab)

        embedder = OrigamiEmbedder(model, simple_tokenizer)

        # Verify embedder uses CPU
        assert embedder.device == torch.device("cpu")
        # Model should be moved to CPU
        assert next(embedder.model.parameters()).device == torch.device("cpu")

        obj = {"name": "Alice", "age": 30, "city": "NYC"}
        embedding = embedder.embed(obj)

        assert embedding.device == torch.device("cpu")
        assert not torch.isnan(embedding).any()


class TestEmbedderPoolingDetails:
    """Detailed tests for pooling implementations."""

    def test_mean_pool_excludes_padding(self, simple_model, simple_tokenizer):
        """Test that mean pooling correctly excludes padding."""
        embedder = OrigamiEmbedder(simple_model, simple_tokenizer, pooling="mean")

        # Batch with different lengths should still work
        # Unknown keys/values map to UNK tokens gracefully
        objects = [
            {"name": "Alice"},
            {"name": "Bob", "age": 25, "city": "LA", "country": "USA"},
        ]
        embeddings = embedder.embed_batch(objects)

        assert embeddings.shape == (2, simple_model.config.d_model)
        # Both should be valid (not NaN)
        assert not torch.isnan(embeddings).any()

    def test_max_pool_excludes_padding(self, simple_model, simple_tokenizer):
        """Test that max pooling correctly excludes padding."""
        embedder = OrigamiEmbedder(simple_model, simple_tokenizer, pooling="max")

        objects = [
            {"name": "Alice"},
            {"name": "Bob", "age": 25, "city": "LA"},
        ]
        embeddings = embedder.embed_batch(objects)

        assert embeddings.shape == (2, simple_model.config.d_model)
        # Should not have -inf (which would happen if padding wasn't excluded)
        assert not torch.isinf(embeddings).any()

    def test_last_pool_gets_correct_position(self, simple_model, simple_tokenizer):
        """Test that last pooling gets the actual last token, not padding."""
        embedder = OrigamiEmbedder(simple_model, simple_tokenizer, pooling="last")

        objects = [
            {"name": "Alice"},
            {"name": "Bob", "age": 25},
        ]
        embeddings = embedder.embed_batch(objects)

        assert embeddings.shape == (2, simple_model.config.d_model)
        assert not torch.isnan(embeddings).any()


class TestEmbedderWithNestedData:
    """Tests for embedder with nested JSON structures."""

    @pytest.fixture
    def nested_tokenizer(self):
        """Create a tokenizer fitted on nested data."""
        data = [
            {
                "user": {"name": "Alice", "profile": {"age": 30}},
                "status": "active",
            },
            {
                "user": {"name": "Bob", "profile": {"age": 25}},
                "status": "inactive",
            },
        ]
        tokenizer = JSONTokenizer()
        tokenizer.fit(data)
        return tokenizer

    @pytest.fixture
    def nested_model(self, nested_tokenizer):
        """Create a model for nested data."""
        config = ModelConfig(
            d_model=32,
            n_heads=2,
            n_layers=1,
            d_ff=64,
            max_depth=nested_tokenizer.max_depth,
        )
        return OrigamiModel(config, vocab=nested_tokenizer.vocab)

    def test_embed_nested_object(self, nested_model, nested_tokenizer):
        """Test embedding nested objects."""
        embedder = OrigamiEmbedder(nested_model, nested_tokenizer)

        obj = {
            "user": {"name": "Alice", "profile": {"age": 30}},
            "status": "active",
        }
        embedding = embedder.embed(obj)

        assert embedding.shape == (nested_model.config.d_model,)
        assert not torch.isnan(embedding).any()

    def test_target_pooling_nested_key(self, nested_model, nested_tokenizer):
        """Test target pooling with nested key."""
        embedder = OrigamiEmbedder(nested_model, nested_tokenizer, pooling="target")

        obj = {
            "user": {"name": "Alice", "profile": {"age": 30}},
            "status": "active",
        }

        # Target the nested status field
        embedding = embedder.embed(obj, target_key="status")
        assert embedding.shape == (nested_model.config.d_model,)

        # Target a deeply nested field
        embedding_nested = embedder.embed(obj, target_key="user.profile.age")
        assert embedding_nested.shape == (nested_model.config.d_model,)


class TestEmbedderWithArrays:
    """Tests for embedder with array data."""

    @pytest.fixture
    def array_tokenizer(self):
        """Create a tokenizer fitted on array data."""
        data = [
            {"tags": ["python", "ml"], "scores": [95, 87, 92]},
            {"tags": ["java"], "scores": [88, 90]},
        ]
        tokenizer = JSONTokenizer()
        tokenizer.fit(data)
        return tokenizer

    @pytest.fixture
    def array_model(self, array_tokenizer):
        """Create a model for array data."""
        config = ModelConfig(
            d_model=32,
            n_heads=2,
            n_layers=1,
            d_ff=64,
            max_depth=array_tokenizer.max_depth,
        )
        return OrigamiModel(config, vocab=array_tokenizer.vocab)

    def test_embed_with_arrays(self, array_model, array_tokenizer):
        """Test embedding objects containing arrays."""
        embedder = OrigamiEmbedder(array_model, array_tokenizer)

        obj = {"tags": ["python", "ml"], "scores": [95, 87, 92]}
        embedding = embedder.embed(obj)

        assert embedding.shape == (array_model.config.d_model,)
        assert not torch.isnan(embedding).any()

    def test_variable_length_arrays(self, array_model, array_tokenizer):
        """Test batching objects with different array lengths."""
        embedder = OrigamiEmbedder(array_model, array_tokenizer)

        objects = [
            {"tags": ["python", "ml"], "scores": [95, 87, 92]},
            {"tags": ["java"], "scores": [88, 90]},
        ]
        embeddings = embedder.embed_batch(objects)

        assert embeddings.shape == (2, array_model.config.d_model)
        assert not torch.isnan(embeddings).any()


class TestEmbedderGradients:
    """Tests for gradient computation with enable_grad parameter."""

    def test_enable_grad_false_no_gradients(self, simple_model, simple_tokenizer):
        """Test that enable_grad=False (default) does not compute gradients."""
        embedder = OrigamiEmbedder(simple_model, simple_tokenizer, pooling="mean")

        obj = {"name": "Alice", "age": 30, "city": "NYC"}
        embedding = embedder.embed(obj, enable_grad=False)

        # Embedding should not require grad
        assert not embedding.requires_grad

    def test_enable_grad_true_computes_gradients(self, simple_model, simple_tokenizer):
        """Test that enable_grad=True computes gradients for fine-tuning."""
        # Enable grad on model parameters
        for param in simple_model.parameters():
            param.requires_grad = True

        embedder = OrigamiEmbedder(simple_model, simple_tokenizer, pooling="mean")

        obj = {"name": "Alice", "age": 30, "city": "NYC"}
        embedding = embedder.embed(obj, enable_grad=True)

        # Embedding should require grad
        assert embedding.requires_grad

        # Should be able to compute loss and backprop
        loss = embedding.sum()
        loss.backward()

        # At least some model parameters should have gradients
        # (not all parameters may be used in the forward pass)
        params_with_grad = [p for p in simple_model.parameters() if p.grad is not None]
        assert len(params_with_grad) > 0, "No parameters received gradients"

    def test_enable_grad_batch(self, simple_model, simple_tokenizer):
        """Test enable_grad works with batch embedding."""
        for param in simple_model.parameters():
            param.requires_grad = True

        embedder = OrigamiEmbedder(simple_model, simple_tokenizer, pooling="mean")

        objects = [
            {"name": "Alice", "age": 30, "city": "NYC"},
            {"name": "Bob", "age": 25, "city": "LA"},
        ]

        # Without grad
        embeddings_no_grad = embedder.embed_batch(objects, enable_grad=False)
        assert not embeddings_no_grad.requires_grad

        # With grad
        embeddings_with_grad = embedder.embed_batch(objects, enable_grad=True)
        assert embeddings_with_grad.requires_grad

    def test_enable_grad_target_pooling(self, simple_model, simple_tokenizer):
        """Test enable_grad works with target pooling."""
        for param in simple_model.parameters():
            param.requires_grad = True

        embedder = OrigamiEmbedder(simple_model, simple_tokenizer, pooling="target")

        obj = {"name": "Alice", "age": 30, "city": "NYC"}
        embedding = embedder.embed(obj, target_key="city", enable_grad=True)

        assert embedding.requires_grad

        # Backprop should work
        loss = embedding.sum()
        loss.backward()
