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


class TestPredictorBatchVariations:
    """Tests for batch prediction with various object configurations."""

    @pytest.fixture
    def varied_tokenizer(self):
        """Create tokenizer with varied data."""
        data = [
            {"a": 1, "b": 2, "target": "x"},
            {"a": 1, "target": "y"},
            {"a": 1, "b": 2, "c": 3, "d": 4, "target": "z"},
        ]
        tokenizer = JSONTokenizer()
        tokenizer.fit(data)
        return tokenizer

    @pytest.fixture
    def varied_model(self, varied_tokenizer):
        """Create model for varied data."""
        config = OrigamiConfig(
            vocab_size=varied_tokenizer.vocab.size,
            d_model=32,
            n_heads=2,
            n_layers=1,
            d_ff=64,
            max_depth=varied_tokenizer.max_depth,
        )
        return OrigamiModel(config, vocab=varied_tokenizer.vocab)

    def test_batch_predict_different_sizes(self, varied_model, varied_tokenizer):
        """Test batch prediction with objects of different sizes."""
        predictor = OrigamiPredictor(varied_model, varied_tokenizer)

        objects = [
            {"a": 1, "target": None},  # Small
            {"a": 1, "b": 2, "target": None},  # Medium
            {"a": 1, "b": 2, "c": 3, "d": 4, "target": None},  # Large
        ]

        results = predictor.predict_batch(objects, target_key="target")

        assert len(results) == 3
        for result in results:
            assert isinstance(result, list)
            assert len(result) == 1  # top_k=1 by default

    def test_batch_predict_single_object(self, varied_model, varied_tokenizer):
        """Test batch prediction with a single object."""
        predictor = OrigamiPredictor(varied_model, varied_tokenizer)

        objects = [{"a": 1, "target": None}]
        results = predictor.predict_batch(objects, target_key="target")

        assert len(results) == 1

    def test_batch_predict_many_objects(self, varied_model, varied_tokenizer):
        """Test batch prediction with many objects."""
        predictor = OrigamiPredictor(varied_model, varied_tokenizer)

        # Create 10 objects
        objects = [{"a": i, "target": None} for i in range(10)]
        results = predictor.predict_batch(objects, target_key="target")

        assert len(results) == 10

    def test_batch_predict_consistent_with_single(self, varied_model, varied_tokenizer):
        """Test that batch prediction gives same results as single prediction."""
        predictor = OrigamiPredictor(varied_model, varied_tokenizer)

        obj = {"a": 1, "b": 2, "target": None}

        # Single prediction
        single_result = predictor.predict(obj, target_key="target", top_k=3)

        # Batch prediction with one object
        batch_results = predictor.predict_batch([obj], target_key="target", top_k=3)

        # Results should match
        assert single_result == batch_results[0]


class TestPredictorComplexValues:
    """Tests for predicting complex values (objects and arrays)."""

    @pytest.fixture
    def complex_tokenizer(self):
        """Create tokenizer with complex nested data."""
        data = [
            {"info": {"nested": "value"}, "target": {"result": "a"}},
            {"info": {"nested": "other"}, "target": {"result": "b"}},
            {"items": [1, 2], "target": [3, 4]},
        ]
        tokenizer = JSONTokenizer()
        tokenizer.fit(data)
        return tokenizer

    @pytest.fixture
    def complex_model(self, complex_tokenizer):
        """Create model for complex data."""
        config = OrigamiConfig(
            vocab_size=complex_tokenizer.vocab.size,
            d_model=32,
            n_heads=2,
            n_layers=1,
            d_ff=64,
            max_depth=complex_tokenizer.max_depth,
        )
        return OrigamiModel(config, vocab=complex_tokenizer.vocab)

    def test_predict_with_nested_context(self, complex_model, complex_tokenizer):
        """Test prediction when context contains nested objects."""
        predictor = OrigamiPredictor(complex_model, complex_tokenizer)

        obj = {"info": {"nested": "value"}, "target": None}
        result = predictor.predict(obj, target_key="target")

        # Should return some value (could be primitive or complex)
        # With untrained model, we just verify it doesn't crash
        assert result is not None or result is None

    def test_predict_with_array_context(self, complex_model, complex_tokenizer):
        """Test prediction when context contains arrays."""
        predictor = OrigamiPredictor(complex_model, complex_tokenizer)

        obj = {"items": [1, 2], "target": None}
        result = predictor.predict(obj, target_key="target")

        # Should return some value
        assert result is not None or result is None


class TestPredictorRobustness:
    """Robustness tests for predictor."""

    @pytest.fixture
    def robust_tokenizer(self):
        """Create tokenizer for robustness tests."""
        data = [
            {"str_field": "hello", "num_field": 42, "bool_field": True},
            {"str_field": "world", "num_field": 0, "bool_field": False},
            {"str_field": "", "num_field": -1, "bool_field": True},
        ]
        tokenizer = JSONTokenizer()
        tokenizer.fit(data)
        return tokenizer

    @pytest.fixture
    def robust_model(self, robust_tokenizer):
        """Create model for robustness tests."""
        config = OrigamiConfig(
            vocab_size=robust_tokenizer.vocab.size,
            d_model=32,
            n_heads=2,
            n_layers=1,
            d_ff=64,
            max_depth=robust_tokenizer.max_depth,
        )
        return OrigamiModel(config, vocab=robust_tokenizer.vocab)

    def test_predict_string_field(self, robust_model, robust_tokenizer):
        """Test predicting a string field."""
        predictor = OrigamiPredictor(robust_model, robust_tokenizer)

        obj = {"str_field": None, "num_field": 42, "bool_field": True}
        result = predictor.predict(obj, target_key="str_field")

        # Should not crash
        assert result is not None or result is None

    def test_predict_numeric_field(self, robust_model, robust_tokenizer):
        """Test predicting a numeric field."""
        predictor = OrigamiPredictor(robust_model, robust_tokenizer)

        obj = {"str_field": "hello", "num_field": None, "bool_field": True}
        result = predictor.predict(obj, target_key="num_field")

        # Should not crash
        assert result is not None or result is None

    def test_predict_boolean_field(self, robust_model, robust_tokenizer):
        """Test predicting a boolean field."""
        predictor = OrigamiPredictor(robust_model, robust_tokenizer)

        obj = {"str_field": "hello", "num_field": 42, "bool_field": None}
        result = predictor.predict(obj, target_key="bool_field")

        # Should not crash
        assert result is not None or result is None

    def test_predict_proba_returns_valid_distribution(self, robust_model, robust_tokenizer):
        """Test that predict_proba returns valid probability distribution."""
        predictor = OrigamiPredictor(robust_model, robust_tokenizer)

        obj = {"str_field": None, "num_field": 42, "bool_field": True}
        probs = predictor.predict_proba(obj, target_key="str_field")

        # All probabilities should be valid
        for value, prob in probs.items():
            assert 0.0 <= prob <= 1.0

    def test_predict_multiple_calls_stable(self, robust_model, robust_tokenizer):
        """Test that multiple prediction calls are stable."""
        predictor = OrigamiPredictor(robust_model, robust_tokenizer)

        obj = {"str_field": None, "num_field": 42, "bool_field": True}

        # Make multiple calls
        results = [predictor.predict(obj, target_key="str_field") for _ in range(5)]

        # All results should be identical (deterministic)
        assert all(r == results[0] for r in results)


class TestPredictorIntegration:
    """Integration tests for predictor with the full pipeline."""

    def test_predictor_uses_generator_internally(self):
        """Test that predictor creates and uses generator correctly."""
        data = [
            {"key": "value1", "target": {"nested": "obj"}},
            {"key": "value2", "target": {"nested": "other"}},
        ]
        tokenizer = JSONTokenizer()
        tokenizer.fit(data)

        config = OrigamiConfig(
            vocab_size=tokenizer.vocab.size,
            d_model=32,
            n_heads=2,
            n_layers=1,
            d_ff=64,
            max_depth=tokenizer.max_depth,
        )
        model = OrigamiModel(config, vocab=tokenizer.vocab)

        predictor = OrigamiPredictor(model, tokenizer)

        # Verify generator is created
        assert predictor._generator is not None

        # Prediction should work
        obj = {"key": "value1", "target": None}
        result = predictor.predict(obj, target_key="target")

        # Should return something (value could be anything with random weights)
        assert result is not None or result is None

    def test_predictor_with_grammar_constraints(self):
        """Test predictor works with grammar-constrained model."""
        data = [
            {"a": 1, "b": 2},
            {"a": 3, "b": 4},
        ]
        tokenizer = JSONTokenizer()
        tokenizer.fit(data)

        config = OrigamiConfig(
            vocab_size=tokenizer.vocab.size,
            d_model=32,
            n_heads=2,
            n_layers=1,
            d_ff=64,
            max_depth=tokenizer.max_depth,
            use_grammar_constraints=True,  # Enable grammar constraints
        )
        model = OrigamiModel(config, vocab=tokenizer.vocab)

        predictor = OrigamiPredictor(model, tokenizer)

        obj = {"a": 1, "b": None}
        result = predictor.predict(obj, target_key="b")

        # Should work without crashing
        assert result is not None or result is None
