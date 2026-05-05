"""Tests for NumericScaler preprocessing component."""

import pytest

from origami.preprocessing import NumericScaler, ScaledNumeric


class TestScaledNumeric:
    """Tests for ScaledNumeric marker class."""

    def test_creation(self):
        """Test basic creation."""
        scaled = ScaledNumeric(0.5)
        assert scaled.value == 0.5

    def test_negative_value(self):
        """Test with negative value (valid for normalized data)."""
        scaled = ScaledNumeric(-1.5)
        assert scaled.value == -1.5

    def test_zero_value(self):
        """Test with zero value."""
        scaled = ScaledNumeric(0.0)
        assert scaled.value == 0.0


class TestNumericScalerInit:
    """Tests for NumericScaler initialization."""

    def test_default_params(self):
        """Test default initialization."""
        scaler = NumericScaler()
        assert scaler.cat_threshold == 100
        assert scaler.scalers == {}
        assert scaler.scaled_fields == set()
        assert scaler.passthrough_fields == set()

    def test_custom_threshold(self):
        """Test custom cat_threshold."""
        scaler = NumericScaler(cat_threshold=50)
        assert scaler.cat_threshold == 50

    def test_invalid_threshold(self):
        """Test invalid cat_threshold raises error."""
        with pytest.raises(ValueError, match="cat_threshold must be >= 1"):
            NumericScaler(cat_threshold=0)
        with pytest.raises(ValueError, match="cat_threshold must be >= 1"):
            NumericScaler(cat_threshold=-10)


class TestNumericScalerFit:
    """Tests for NumericScaler.fit()."""

    def test_fit_high_cardinality_field(self):
        """Test fitting on high cardinality numeric field."""
        # Create data with >100 unique values
        data = [{"price": float(i)} for i in range(150)]

        scaler = NumericScaler(cat_threshold=100)
        scaler.fit(data)

        assert "price" in scaler.scaled_fields
        assert "price" not in scaler.passthrough_fields
        assert "price" in scaler.scalers

    def test_fit_low_cardinality_field(self):
        """Test fitting on low cardinality field (passthrough)."""
        # Create data with <100 unique values
        data = [{"category": i % 10} for i in range(200)]

        scaler = NumericScaler(cat_threshold=100)
        scaler.fit(data)

        assert "category" in scaler.passthrough_fields
        assert "category" not in scaler.scaled_fields
        assert "category" not in scaler.scalers

    def test_fit_nested_fields(self):
        """Test fitting on nested fields."""
        data = [{"user": {"score": float(i)}} for i in range(150)]

        scaler = NumericScaler(cat_threshold=100)
        scaler.fit(data)

        assert "user.score" in scaler.scaled_fields
        assert "user.score" in scaler.scalers

    def test_fit_mixed_cardinality(self):
        """Test with mix of high and low cardinality."""
        data = [{"high": float(i), "low": i % 5} for i in range(200)]

        scaler = NumericScaler(cat_threshold=100)
        scaler.fit(data)

        assert "high" in scaler.scaled_fields
        assert "low" in scaler.passthrough_fields

    def test_fit_returns_self(self):
        """Test fit returns self for chaining."""
        scaler = NumericScaler()
        result = scaler.fit([{"x": 1.0}])
        assert result is scaler

    def test_fit_string_fields_ignored(self):
        """Test string fields are ignored."""
        data = [{"name": f"user_{i}", "value": float(i)} for i in range(150)]

        scaler = NumericScaler(cat_threshold=100)
        scaler.fit(data)

        assert "name" not in scaler.scaled_fields
        assert "name" not in scaler.passthrough_fields
        assert "value" in scaler.scaled_fields


class TestNumericScalerTransform:
    """Tests for NumericScaler.transform()."""

    def test_transform_without_fit_raises(self):
        """Test transform without fit raises error."""
        scaler = NumericScaler()
        with pytest.raises(RuntimeError, match="fit.*transform"):
            scaler.transform([{"x": 1.0}])

    def test_transform_high_cardinality(self):
        """Test transformation of high cardinality field."""
        data = [{"price": float(i * 100)} for i in range(150)]

        scaler = NumericScaler(cat_threshold=100)
        scaler.fit(data)
        transformed = scaler.transform(data)

        # Check all prices are ScaledNumeric
        for obj in transformed:
            assert isinstance(obj["price"], ScaledNumeric)

    def test_transform_low_cardinality_passthrough(self):
        """Test low cardinality fields pass through unchanged."""
        data = [{"category": i % 5} for i in range(50)]

        scaler = NumericScaler(cat_threshold=100)
        scaler.fit(data)
        transformed = scaler.transform(data)

        # Check categories are unchanged
        for i, obj in enumerate(transformed):
            assert obj["category"] == i % 5
            assert not isinstance(obj["category"], ScaledNumeric)

    def test_transform_preserves_structure(self):
        """Test transformation preserves JSON structure."""
        data = [
            {"user": {"name": "test", "score": float(i)}, "items": [1, 2, 3]} for i in range(150)
        ]

        scaler = NumericScaler(cat_threshold=100)
        scaler.fit(data)
        transformed = scaler.transform(data)

        for obj in transformed:
            assert "user" in obj
            assert "name" in obj["user"]
            assert obj["user"]["name"] == "test"
            assert "items" in obj
            assert obj["items"] == [1, 2, 3]
            assert isinstance(obj["user"]["score"], ScaledNumeric)

    def test_transform_values_are_normalized(self):
        """Test transformed values are normalized (approx mean 0, std 1)."""
        data = [{"value": float(i)} for i in range(1000)]

        scaler = NumericScaler(cat_threshold=100)
        scaler.fit(data)
        transformed = scaler.transform(data)

        values = [obj["value"].value for obj in transformed]
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        std = variance**0.5

        # Should be approximately normalized
        assert abs(mean) < 0.01
        assert abs(std - 1.0) < 0.01

    def test_transform_does_not_mutate_input(self):
        """Test transform does not mutate input data."""
        data = [{"price": 100.0}, {"price": 200.0}]
        original_prices = [obj["price"] for obj in data]

        scaler = NumericScaler(cat_threshold=1)  # Force scaling
        scaler.fit(data)
        scaler.transform(data)

        # Original data should be unchanged
        assert data[0]["price"] == original_prices[0]
        assert data[1]["price"] == original_prices[1]


class TestNumericScalerInverseTransform:
    """Tests for NumericScaler.inverse_transform_value()."""

    def test_inverse_transform_single_value(self):
        """Test inverse transform of single value."""
        data = [{"price": float(i * 100)} for i in range(150)]

        scaler = NumericScaler(cat_threshold=100)
        scaler.fit(data)
        transformed = scaler.transform(data)

        # Inverse transform should recover original value
        original = 5000.0  # 50 * 100
        scaled = transformed[50]["price"].value
        recovered = scaler.inverse_transform_value("price", scaled)

        assert abs(recovered - original) < 1.0  # Allow small float error

    def test_inverse_transform_nested_field(self):
        """Test inverse transform of nested field."""
        data = [{"user": {"score": float(i)}} for i in range(150)]

        scaler = NumericScaler(cat_threshold=100)
        scaler.fit(data)
        transformed = scaler.transform(data)

        original = 75.0
        scaled = transformed[75]["user"]["score"].value
        recovered = scaler.inverse_transform_value("user.score", scaled)

        assert abs(recovered - original) < 0.1

    def test_inverse_transform_unknown_field_raises(self):
        """Test inverse transform of unknown field raises error."""
        scaler = NumericScaler()
        scaler.fit([{"x": float(i)} for i in range(150)])

        with pytest.raises(KeyError, match="not a scaled field"):
            scaler.inverse_transform_value("unknown", 0.5)

    def test_roundtrip_transform(self):
        """Test complete roundtrip: value -> scaled -> original."""
        import random

        random.seed(42)

        # Generate random data
        data = [{"value": random.uniform(0, 1000)} for _ in range(200)]

        scaler = NumericScaler(cat_threshold=100)
        scaler.fit(data)
        transformed = scaler.transform(data)

        # Check roundtrip for each value
        for original, trans in zip(data, transformed, strict=True):
            scaled = trans["value"].value
            recovered = scaler.inverse_transform_value("value", scaled)
            assert abs(recovered - original["value"]) < 0.01


class TestNumericScalerEdgeCases:
    """Tests for edge cases in NumericScaler."""

    def test_empty_data(self):
        """Test with empty data list."""
        scaler = NumericScaler()
        scaler.fit([])
        assert scaler.scaled_fields == set()
        assert scaler.passthrough_fields == set()

    def test_single_value(self):
        """Test with single value (std=0).

        When all values are identical, unique_count=1 which is NOT > cat_threshold,
        so the field passes through unchanged (low cardinality).
        """
        data = [{"x": 5.0} for _ in range(150)]

        scaler = NumericScaler(cat_threshold=1)
        scaler.fit(data)

        # With only 1 unique value, it passes through (not scaled)
        transformed = scaler.transform(data)
        for obj in transformed:
            assert obj["x"] == 5.0
            assert not isinstance(obj["x"], ScaledNumeric)
        assert "x" in scaler.passthrough_fields

    def test_none_values_skipped(self):
        """Test None values are preserved."""
        data = [{"value": float(i) if i % 2 == 0 else None} for i in range(200)]

        scaler = NumericScaler(cat_threshold=50)
        scaler.fit(data)
        transformed = scaler.transform(data)

        for i, obj in enumerate(transformed):
            if i % 2 == 0:
                assert isinstance(obj["value"], ScaledNumeric)
            else:
                assert obj["value"] is None

    def test_array_values_scaled(self):
        """Test array values are scaled correctly."""
        data = [{"values": [float(i), float(i + 1)]} for i in range(200)]

        scaler = NumericScaler(cat_threshold=100)
        scaler.fit(data)
        transformed = scaler.transform(data)

        for obj in transformed:
            assert isinstance(obj["values"], list)
            for v in obj["values"]:
                assert isinstance(v, ScaledNumeric)


class TestNumericScalerIntegrationWithTokenizer:
    """Integration tests with JSONTokenizer."""

    def test_scaler_then_tokenizer(self):
        """Test NumericScaler output works with JSONTokenizer."""
        from origami.tokenizer import JSONTokenizer

        # Create high-cardinality data
        data = [{"price": float(i * 10), "name": "item"} for i in range(150)]

        # Scale
        scaler = NumericScaler(cat_threshold=100)
        scaler.fit(data)
        scaled_data = scaler.transform(data)

        # Tokenize
        tokenizer = JSONTokenizer()
        tokenizer.fit(scaled_data)

        # Check NUM token is in vocab
        from origami.tokenizer.vocabulary import NUM

        assert tokenizer.vocab.encode(NUM) == 9  # NUM token ID

        # Tokenize an instance
        instance = tokenizer.tokenize(scaled_data[0])

        # Check numeric_values is populated
        assert instance.numeric_values is not None
        assert any(v is not None for v in instance.numeric_values)
