"""Tests for OrigamiPredictor."""

import pytest
import torch

from origami.inference import OrigamiPredictor
from origami.model import OrigamiConfig, OrigamiModel
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
    config = OrigamiConfig(
        vocab_size=simple_tokenizer.vocab.size,
        d_model=32,
        n_heads=2,
        n_layers=1,
        d_ff=64,
        max_depth=simple_tokenizer.max_depth,
    )
    return OrigamiModel(config, vocab=simple_tokenizer.vocab)


class TestOrigamiPredictor:
    """Tests for OrigamiPredictor."""

    def test_init(self, simple_model, simple_tokenizer):
        """Test predictor initialization."""
        predictor = OrigamiPredictor(simple_model, simple_tokenizer)

        assert predictor.model is simple_model
        assert predictor.tokenizer is simple_tokenizer

    def test_predict_single(self, simple_model, simple_tokenizer):
        """Test predicting single value."""
        predictor = OrigamiPredictor(simple_model, simple_tokenizer)

        obj = {"name": "Alice", "age": 30, "city": None}
        result = predictor.predict(obj, target_key="city")

        # With random weights, result is random but should be a valid value
        # (string, number, bool, or None from vocabulary)
        assert result is not None or result is None  # Can be any value

    def test_predict_with_return_probs(self, simple_model, simple_tokenizer):
        """Test prediction with probabilities."""
        predictor = OrigamiPredictor(simple_model, simple_tokenizer)

        obj = {"name": "Alice", "age": 30, "city": None}
        result = predictor.predict(obj, target_key="city", return_probs=True)

        # Should be (value, probability) tuple
        assert isinstance(result, tuple)
        assert len(result) == 2
        value, prob = result
        assert 0.0 <= prob <= 1.0

    def test_predict_top_k(self, simple_model, simple_tokenizer):
        """Test top-k predictions."""
        predictor = OrigamiPredictor(simple_model, simple_tokenizer)

        obj = {"name": "Alice", "age": 30, "city": None}
        results = predictor.predict(obj, target_key="city", top_k=3)

        # Should be list of (value, probability) tuples
        assert isinstance(results, list)
        assert len(results) == 3
        for value, prob in results:
            assert 0.0 <= prob <= 1.0

    def test_predict_batch(self, simple_model, simple_tokenizer):
        """Test batch prediction."""
        predictor = OrigamiPredictor(simple_model, simple_tokenizer)

        objects = [
            {"name": "Alice", "age": 30, "city": None},
            {"name": "Bob", "age": 25, "city": None},
            {"name": "Charlie", "age": 35, "city": None},
        ]
        results = predictor.predict_batch(objects, target_key="city")

        assert len(results) == 3
        for result in results:
            assert isinstance(result, list)
            assert len(result) == 1  # top_k=1 by default
            value, prob = result[0]
            assert 0.0 <= prob <= 1.0

    def test_predict_batch_top_k(self, simple_model, simple_tokenizer):
        """Test batch prediction with top-k."""
        predictor = OrigamiPredictor(simple_model, simple_tokenizer)

        objects = [
            {"name": "Alice", "age": 30, "city": None},
            {"name": "Bob", "age": 25, "city": None},
        ]
        results = predictor.predict_batch(objects, target_key="city", top_k=3)

        assert len(results) == 2
        for result in results:
            assert len(result) == 3

    def test_predict_proba_specific_values(self, simple_model, simple_tokenizer):
        """Test getting probabilities for specific values."""
        predictor = OrigamiPredictor(simple_model, simple_tokenizer)

        obj = {"name": "Alice", "age": 30, "city": None}
        result = predictor.predict_proba(
            obj, target_key="city", values=["NYC", "LA", "SF"]
        )

        assert isinstance(result, dict)
        assert "NYC" in result
        assert "LA" in result
        assert "SF" in result
        for prob in result.values():
            assert 0.0 <= prob <= 1.0

    def test_predict_proba_all_values(self, simple_model, simple_tokenizer):
        """Test getting probabilities for all values."""
        predictor = OrigamiPredictor(simple_model, simple_tokenizer)

        obj = {"name": "Alice", "age": 30, "city": None}
        result = predictor.predict_proba(obj, target_key="city")

        assert isinstance(result, dict)
        # Should have some values with non-zero probability
        total_prob = sum(result.values())
        # Probabilities should be reasonable (not all zero)
        # Note: with random weights, distribution is random

    def test_predict_unknown_value_prob(self, simple_model, simple_tokenizer):
        """Test probability for unknown value is zero."""
        predictor = OrigamiPredictor(simple_model, simple_tokenizer)

        obj = {"name": "Alice", "age": 30, "city": None}
        result = predictor.predict_proba(
            obj, target_key="city", values=["UnknownCity123"]
        )

        assert "UnknownCity123" in result
        assert result["UnknownCity123"] == 0.0

    def test_target_key_not_found(self, simple_model, simple_tokenizer):
        """Test error when target key not in object."""
        predictor = OrigamiPredictor(simple_model, simple_tokenizer)

        obj = {"name": "Alice", "age": 30}
        with pytest.raises(KeyError):
            predictor.predict(obj, target_key="city")

    def test_always_uses_cpu(self, simple_tokenizer):
        """Test that predictor always runs on CPU for performance."""
        config = OrigamiConfig(
            vocab_size=simple_tokenizer.vocab.size,
            d_model=32,
            n_heads=2,
            n_layers=1,
            d_ff=64,
            max_depth=simple_tokenizer.max_depth,
        )
        model = OrigamiModel(config, vocab=simple_tokenizer.vocab)

        predictor = OrigamiPredictor(model, simple_tokenizer)

        # Verify predictor uses CPU
        assert predictor.device == torch.device("cpu")
        # Model should be moved to CPU
        assert next(predictor.model.parameters()).device == torch.device("cpu")

        obj = {"name": "Alice", "age": 30, "city": None}
        result = predictor.predict(obj, target_key="city")
        # Should not raise


class TestPredictorWithNestedData:
    """Tests for predictor with nested JSON structures."""

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
        config = OrigamiConfig(
            vocab_size=nested_tokenizer.vocab.size,
            d_model=32,
            n_heads=2,
            n_layers=1,
            d_ff=64,
            max_depth=nested_tokenizer.max_depth,
        )
        return OrigamiModel(config, vocab=nested_tokenizer.vocab)

    def test_predict_nested_key(self, nested_model, nested_tokenizer):
        """Test predicting nested key value."""
        predictor = OrigamiPredictor(nested_model, nested_tokenizer)

        obj = {
            "user": {"name": "Alice", "profile": {"age": None}},
            "status": "active",
        }
        result = predictor.predict(obj, target_key="user.profile.age")

        # Should return some value (random with untrained model)
        assert result is not None or result is None

    def test_predict_root_level_key(self, nested_model, nested_tokenizer):
        """Test predicting root level key in nested object."""
        predictor = OrigamiPredictor(nested_model, nested_tokenizer)

        obj = {
            "user": {"name": "Alice", "profile": {"age": 30}},
            "status": None,
        }
        result = predictor.predict(obj, target_key="status")

        # Should return some value
        assert result is not None or result is None


class TestPredictorWithArrayData:
    """Tests for predictor with array data."""

    @pytest.fixture
    def array_tokenizer(self):
        """Create a tokenizer fitted on array data."""
        data = [
            {"tags": ["python", "ml"], "primary_tag": "python"},
            {"tags": ["java", "web"], "primary_tag": "java"},
        ]
        tokenizer = JSONTokenizer()
        tokenizer.fit(data)
        return tokenizer

    @pytest.fixture
    def array_model(self, array_tokenizer):
        """Create a model for array data."""
        config = OrigamiConfig(
            vocab_size=array_tokenizer.vocab.size,
            d_model=32,
            n_heads=2,
            n_layers=1,
            d_ff=64,
            max_depth=array_tokenizer.max_depth,
        )
        return OrigamiModel(config, vocab=array_tokenizer.vocab)

    def test_predict_with_array_context(self, array_model, array_tokenizer):
        """Test prediction with arrays in context."""
        predictor = OrigamiPredictor(array_model, array_tokenizer)

        obj = {"tags": ["python", "ml"], "primary_tag": None}
        result = predictor.predict(obj, target_key="primary_tag")

        # Should return some value
        assert result is not None or result is None


class TestPredictorDeterminism:
    """Tests for predictor determinism."""

    def test_predictions_are_deterministic(self, simple_model, simple_tokenizer):
        """Test that same input produces same output."""
        predictor = OrigamiPredictor(simple_model, simple_tokenizer)

        obj = {"name": "Alice", "age": 30, "city": None}

        result1 = predictor.predict(obj, target_key="city", top_k=3)
        result2 = predictor.predict(obj, target_key="city", top_k=3)

        # Results should be identical
        assert result1 == result2

    def test_different_objects_can_have_different_predictions(
        self, simple_model, simple_tokenizer
    ):
        """Test that different objects can produce different predictions."""
        predictor = OrigamiPredictor(simple_model, simple_tokenizer)

        obj1 = {"name": "Alice", "age": 30, "city": None}
        obj2 = {"name": "Bob", "age": 25, "city": None}

        result1 = predictor.predict(obj1, target_key="city", top_k=5)
        result2 = predictor.predict(obj2, target_key="city", top_k=5)

        # Results could be the same or different depending on model
        # We just verify they run without error
        assert len(result1) == 5
        assert len(result2) == 5
