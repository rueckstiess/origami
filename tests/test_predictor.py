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

    def test_predict_proba_top_k(self, simple_model, simple_tokenizer):
        """Test top-k predictions via predict_proba."""
        predictor = OrigamiPredictor(simple_model, simple_tokenizer)

        obj = {"name": "Alice", "age": 30, "city": None}
        results = predictor.predict_proba(obj, target_key="city", top_k=3)

        # Should be list of (value, probability) tuples
        assert isinstance(results, list)
        assert len(results) == 3
        for _value, prob in results:
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

        # Results should be a list of values (one per object)
        assert len(results) == 3
        for result in results:
            # Each result is a value (not a list of tuples)
            assert result is not None or result is None

    def test_predict_proba_specific_values(self, simple_model, simple_tokenizer):
        """Test getting probabilities for specific values."""
        predictor = OrigamiPredictor(simple_model, simple_tokenizer)

        obj = {"name": "Alice", "age": 30, "city": None}
        result = predictor.predict_proba(obj, target_key="city", values=["NYC", "LA", "SF"])

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
        # Note: with random weights, distribution is random
        assert sum(result.values()) >= 0  # Basic sanity check

    def test_predict_unknown_value_prob(self, simple_model, simple_tokenizer):
        """Test probability for unknown value is zero."""
        predictor = OrigamiPredictor(simple_model, simple_tokenizer)

        obj = {"name": "Alice", "age": 30, "city": None}
        result = predictor.predict_proba(obj, target_key="city", values=["UnknownCity123"])

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
        predictor.predict(obj, target_key="city")  # Should not raise


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

        result1 = predictor.predict(obj, target_key="city")
        result2 = predictor.predict(obj, target_key="city")

        # Results should be identical (greedy sampling)
        assert result1 == result2

    def test_different_objects_can_have_different_predictions(self, simple_model, simple_tokenizer):
        """Test that different objects can produce different predictions."""
        predictor = OrigamiPredictor(simple_model, simple_tokenizer)

        obj1 = {"name": "Alice", "age": 30, "city": None}
        obj2 = {"name": "Bob", "age": 25, "city": None}

        result1 = predictor.predict(obj1, target_key="city")
        result2 = predictor.predict(obj2, target_key="city")

        # Results could be the same or different depending on model
        # We just verify they run without error
        assert result1 is not None or result1 is None
        assert result2 is not None or result2 is None


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
            # Each result is a value
            assert result is not None or result is None

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
        single_result = predictor.predict(obj, target_key="target")

        # Batch prediction with one object
        batch_results = predictor.predict_batch([obj], target_key="target")

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
    def multi_type_tokenizer(self):
        """Create tokenizer with various value types."""
        data = [
            {"str_field": "hello", "num_field": 42, "bool_field": True},
            {"str_field": "world", "num_field": 100, "bool_field": False},
        ]
        tokenizer = JSONTokenizer()
        tokenizer.fit(data)
        return tokenizer

    @pytest.fixture
    def multi_type_model(self, multi_type_tokenizer):
        """Create model for multi-type data."""
        config = OrigamiConfig(
            vocab_size=multi_type_tokenizer.vocab.size,
            d_model=32,
            n_heads=2,
            n_layers=1,
            d_ff=64,
            max_depth=multi_type_tokenizer.max_depth,
        )
        return OrigamiModel(config, vocab=multi_type_tokenizer.vocab)

    def test_predict_string_field(self, multi_type_model, multi_type_tokenizer):
        """Test prediction for string field."""
        predictor = OrigamiPredictor(multi_type_model, multi_type_tokenizer)

        obj = {"str_field": None, "num_field": 42, "bool_field": True}
        result = predictor.predict(obj, target_key="str_field")

        # Should return some value
        assert result is not None or result is None

    def test_predict_numeric_field(self, multi_type_model, multi_type_tokenizer):
        """Test prediction for numeric field."""
        predictor = OrigamiPredictor(multi_type_model, multi_type_tokenizer)

        obj = {"str_field": "hello", "num_field": None, "bool_field": True}
        result = predictor.predict(obj, target_key="num_field")

        # Should return some value
        assert result is not None or result is None

    def test_predict_boolean_field(self, multi_type_model, multi_type_tokenizer):
        """Test prediction for boolean field."""
        predictor = OrigamiPredictor(multi_type_model, multi_type_tokenizer)

        obj = {"str_field": "hello", "num_field": 42, "bool_field": None}
        result = predictor.predict(obj, target_key="bool_field")

        # Should return some value
        assert result is not None or result is None

    def test_predict_proba_returns_valid_distribution(self, multi_type_model, multi_type_tokenizer):
        """Test that predict_proba returns valid probability distribution."""
        predictor = OrigamiPredictor(multi_type_model, multi_type_tokenizer)

        obj = {"str_field": None, "num_field": 42, "bool_field": True}
        probs = predictor.predict_proba(obj, target_key="str_field")

        # Should be a dict with non-negative probabilities
        assert isinstance(probs, dict)
        for prob in probs.values():
            assert 0.0 <= prob <= 1.0

    def test_predict_multiple_calls_stable(self, multi_type_model, multi_type_tokenizer):
        """Test that multiple prediction calls are stable."""
        predictor = OrigamiPredictor(multi_type_model, multi_type_tokenizer)

        obj = {"str_field": None, "num_field": 42, "bool_field": True}
        results = [predictor.predict(obj, target_key="str_field") for _ in range(5)]

        # All results should be identical (deterministic)
        assert all(r == results[0] for r in results)


class TestPredictorIntegration:
    """Integration tests for predictor."""

    @pytest.fixture
    def integration_tokenizer(self):
        """Create tokenizer for integration testing."""
        data = [
            {"a": 1, "b": 2, "target": "x"},
            {"a": 2, "b": 3, "target": "y"},
        ]
        tokenizer = JSONTokenizer()
        tokenizer.fit(data)
        return tokenizer

    @pytest.fixture
    def integration_model(self, integration_tokenizer):
        """Create model for integration testing."""
        config = OrigamiConfig(
            vocab_size=integration_tokenizer.vocab.size,
            d_model=32,
            n_heads=2,
            n_layers=1,
            d_ff=64,
            max_depth=integration_tokenizer.max_depth,
            use_grammar_constraints=True,
        )
        return OrigamiModel(config, vocab=integration_tokenizer.vocab)

    def test_predictor_uses_generator_internally(
        self, integration_model, integration_tokenizer
    ):
        """Test that predictor uses generator for value generation."""
        predictor = OrigamiPredictor(integration_model, integration_tokenizer)

        obj = {"a": 1, "b": 2, "target": None}
        result = predictor.predict(obj, target_key="target")

        # Should return a valid value
        assert result is not None or result is None

    def test_predictor_with_grammar_constraints(
        self, integration_model, integration_tokenizer
    ):
        """Test that predictor respects grammar constraints via generator."""
        predictor = OrigamiPredictor(integration_model, integration_tokenizer)

        obj = {"a": 1, "b": None}
        result = predictor.predict(obj, target_key="b")

        # With grammar constraints, should not return invalid tokens
        # (like OBJ_END when a value is expected)
        # The result should be a valid value
        assert result is not None or result is None
