"""Tests for NumericDiscretizer."""

import numpy as np
import pytest

from origami.preprocessing import NumericDiscretizer


class TestNumericDiscretizerInit:
    """Tests for NumericDiscretizer initialization."""

    def test_default_params(self):
        """Test default parameters."""
        d = NumericDiscretizer()
        assert d.cat_threshold == 100
        assert d.n_bins == 20
        assert d.strategy == "quantile"
        assert not d._fitted

    def test_custom_params(self):
        """Test custom parameters."""
        d = NumericDiscretizer(cat_threshold=50, n_bins=10, strategy="uniform")
        assert d.cat_threshold == 50
        assert d.n_bins == 10
        assert d.strategy == "uniform"

    def test_invalid_cat_threshold(self):
        """Test invalid cat_threshold raises error."""
        with pytest.raises(ValueError, match="cat_threshold must be >= 1"):
            NumericDiscretizer(cat_threshold=0)

    def test_invalid_n_bins(self):
        """Test invalid n_bins raises error."""
        with pytest.raises(ValueError, match="n_bins must be >= 2"):
            NumericDiscretizer(n_bins=1)

    def test_invalid_strategy(self):
        """Test invalid strategy raises error."""
        with pytest.raises(ValueError, match="strategy must be"):
            NumericDiscretizer(strategy="invalid")


class TestNumericDiscretizerFit:
    """Tests for fit method."""

    def test_fit_high_cardinality_field(self):
        """Test fitting on high cardinality numeric field."""
        # 200 distinct values > default threshold of 100
        data = [{"value": i} for i in range(200)]

        d = NumericDiscretizer(cat_threshold=100, n_bins=10)
        d.fit(data)

        assert d._fitted
        assert "value" in d.discretized_fields
        assert "value" not in d.passthrough_fields
        assert "value" in d.discretizers

    def test_fit_low_cardinality_field(self):
        """Test fitting on low cardinality numeric field."""
        # 10 distinct values <= threshold of 100
        data = [{"value": i % 10} for i in range(200)]

        d = NumericDiscretizer(cat_threshold=100, n_bins=10)
        d.fit(data)

        assert d._fitted
        assert "value" not in d.discretized_fields
        assert "value" in d.passthrough_fields
        assert "value" not in d.discretizers

    def test_fit_nested_fields(self):
        """Test fitting handles nested fields."""
        data = [{"stats": {"score": i, "level": i % 5}, "name": "item"} for i in range(200)]

        d = NumericDiscretizer(cat_threshold=50, n_bins=10)
        d.fit(data)

        # stats.score has 200 distinct values > 50 threshold
        assert "stats.score" in d.discretized_fields
        # stats.level has 5 distinct values <= 50 threshold
        assert "stats.level" in d.passthrough_fields

    def test_fit_array_fields(self):
        """Test fitting handles array fields."""
        data = [{"items": [{"price": i * 10 + j} for j in range(3)]} for i in range(100)]

        d = NumericDiscretizer(cat_threshold=50, n_bins=10)
        d.fit(data)

        # Each array position gets its own field path
        assert "items.0.price" in d.discretized_fields
        assert "items.1.price" in d.discretized_fields
        assert "items.2.price" in d.discretized_fields

    def test_fit_mixed_types(self):
        """Test fitting ignores non-numeric fields."""
        data = [
            {
                "name": f"item_{i}",
                "active": i % 2 == 0,
                "value": i,
                "tags": ["a", "b"],
            }
            for i in range(200)
        ]

        d = NumericDiscretizer(cat_threshold=100, n_bins=10)
        d.fit(data)

        # Only numeric field should be tracked
        assert "value" in d.discretized_fields
        assert "name" not in d.discretized_fields
        assert "name" not in d.passthrough_fields
        assert "active" not in d.discretized_fields  # bool is not numeric

    def test_fit_returns_self(self):
        """Test fit returns self for chaining."""
        data = [{"value": i} for i in range(200)]
        d = NumericDiscretizer()
        result = d.fit(data)
        assert result is d


class TestNumericDiscretizerTransform:
    """Tests for transform method."""

    def test_transform_without_fit_raises(self):
        """Test transform before fit raises error."""
        d = NumericDiscretizer()
        with pytest.raises(RuntimeError, match="must be fit before transform"):
            d.transform([{"value": 1}])

    def test_transform_high_cardinality(self):
        """Test transform converts high cardinality fields to bin centers."""
        data = [{"value": float(i)} for i in range(200)]

        d = NumericDiscretizer(cat_threshold=100, n_bins=10)
        transformed = d.fit_transform(data)

        # All values should be converted to bin center floats
        for obj in transformed:
            assert isinstance(obj["value"], float)

    def test_transform_low_cardinality_passthrough(self):
        """Test transform passes through low cardinality fields."""
        data = [{"category": i % 5, "value": float(i)} for i in range(200)]

        d = NumericDiscretizer(cat_threshold=100, n_bins=10)
        transformed = d.fit_transform(data)

        # category should pass through unchanged
        for i, obj in enumerate(transformed):
            assert obj["category"] == i % 5
            # value gets discretized to bin center
            assert isinstance(obj["value"], float)

    def test_transform_preserves_non_numeric(self):
        """Test transform preserves non-numeric fields."""
        data = [{"name": f"item_{i}", "value": float(i), "active": True} for i in range(200)]

        d = NumericDiscretizer(cat_threshold=100, n_bins=10)
        transformed = d.fit_transform(data)

        for i, obj in enumerate(transformed):
            assert obj["name"] == f"item_{i}"
            assert obj["active"] is True

    def test_transform_nested_fields(self):
        """Test transform handles nested fields."""
        data = [{"outer": {"inner": float(i)}} for i in range(200)]

        d = NumericDiscretizer(cat_threshold=100, n_bins=10)
        transformed = d.fit_transform(data)

        for obj in transformed:
            assert isinstance(obj["outer"]["inner"], float)

    def test_transform_arrays(self):
        """Test transform handles arrays."""
        data = [{"items": [float(i), float(i + 100), float(i + 200)]} for i in range(200)]

        d = NumericDiscretizer(cat_threshold=100, n_bins=10)
        transformed = d.fit_transform(data)

        for obj in transformed:
            assert len(obj["items"]) == 3
            for item in obj["items"]:
                assert isinstance(item, float)

    def test_transform_does_not_mutate_input(self):
        """Test transform creates new objects, doesn't mutate input."""
        data = [{"value": float(i)} for i in range(200)]
        original_values = [obj["value"] for obj in data]

        d = NumericDiscretizer(cat_threshold=100, n_bins=10)
        d.fit_transform(data)

        # Original data should be unchanged
        for i, obj in enumerate(data):
            assert obj["value"] == original_values[i]

    def test_transform_consistent_binning(self):
        """Test same values map to same bin centers."""
        data = [{"value": float(i % 50)} for i in range(200)]

        d = NumericDiscretizer(cat_threshold=100, n_bins=10)
        transformed = d.fit_transform(data)

        # Check values 0 and 50 (which equal 0 after mod) get same bin center
        assert transformed[0]["value"] == transformed[50]["value"]
        assert transformed[1]["value"] == transformed[51]["value"]


class TestNumericDiscretizerBinEdges:
    """Tests for bin edge methods."""

    def test_get_bin_edges(self):
        """Test getting bin edges."""
        data = [{"value": float(i)} for i in range(200)]

        d = NumericDiscretizer(cat_threshold=100, n_bins=10)
        d.fit(data)

        edges = d.get_bin_edges("value")
        assert edges is not None
        assert len(edges) == 11  # 10 bins = 11 edges
        assert edges[0] <= 0  # First value
        assert edges[-1] >= 199  # Last value

    def test_get_bin_edges_nonexistent_field(self):
        """Test getting bin edges for non-discretized field."""
        data = [{"value": float(i)} for i in range(200)]

        d = NumericDiscretizer(cat_threshold=100, n_bins=10)
        d.fit(data)

        assert d.get_bin_edges("nonexistent") is None

    def test_get_bin_label(self):
        """Test getting bin label."""
        data = [{"value": float(i)} for i in range(200)]

        d = NumericDiscretizer(cat_threshold=100, n_bins=10)
        d.fit(data)

        label = d.get_bin_label("value", 0)
        assert label is not None
        assert "[" in label and ")" in label  # Interval notation

    def test_get_bin_label_invalid_index(self):
        """Test getting bin label with invalid index."""
        data = [{"value": float(i)} for i in range(200)]

        d = NumericDiscretizer(cat_threshold=100, n_bins=10)
        d.fit(data)

        assert d.get_bin_label("value", -1) is None
        assert d.get_bin_label("value", 100) is None


class TestNumericDiscretizerStrategies:
    """Tests for different binning strategies."""

    def test_quantile_strategy(self):
        """Test quantile strategy creates equal-frequency bins."""
        # Skewed distribution
        data = [{"value": float(i**2)} for i in range(200)]

        d = NumericDiscretizer(cat_threshold=100, n_bins=10, strategy="quantile")
        transformed = d.fit_transform(data)

        # Count samples in each bin (by bin center value)
        bin_counts = {}
        for obj in transformed:
            bin_center = obj["value"]
            bin_counts[bin_center] = bin_counts.get(bin_center, 0) + 1

        # Quantile strategy should have roughly equal counts per bin
        counts = list(bin_counts.values())
        assert max(counts) - min(counts) <= 2  # Small variance

    def test_uniform_strategy(self):
        """Test uniform strategy creates equal-width bins."""
        data = [{"value": float(i)} for i in range(200)]

        d = NumericDiscretizer(cat_threshold=100, n_bins=10, strategy="uniform")
        d.fit(data)

        edges = d.get_bin_edges("value")
        # Uniform bins should have equal widths
        widths = np.diff(edges)
        assert np.allclose(widths, widths[0], rtol=0.01)

    def test_kmeans_strategy(self):
        """Test kmeans strategy works."""
        # Two clusters
        data = [{"value": float(i)} for i in range(100)]
        data += [{"value": float(i + 500)} for i in range(100)]

        d = NumericDiscretizer(cat_threshold=100, n_bins=5, strategy="kmeans")
        transformed = d.fit_transform(data)

        # Should successfully transform to bin center floats
        assert all(isinstance(obj["value"], float) for obj in transformed)


class TestNumericDiscretizerSummary:
    """Tests for summary method."""

    def test_summary_not_fitted(self):
        """Test summary before fitting."""
        d = NumericDiscretizer()
        summary = d.summary()
        assert "not fitted" in summary

    def test_summary_fitted(self):
        """Test summary after fitting."""
        data = [{"high_card": float(i), "low_card": i % 5} for i in range(200)]

        d = NumericDiscretizer(cat_threshold=100, n_bins=10)
        d.fit(data)

        summary = d.summary()
        assert "Discretized fields" in summary
        assert "high_card" in summary
        assert "Pass-through fields" in summary
        assert "low_card" in summary


class TestNumericDiscretizerEdgeCases:
    """Tests for edge cases."""

    def test_empty_data(self):
        """Test fitting on empty data."""
        d = NumericDiscretizer()
        d.fit([])

        assert d._fitted
        assert len(d.discretized_fields) == 0
        assert len(d.passthrough_fields) == 0

    def test_no_numeric_fields(self):
        """Test data with no numeric fields."""
        data = [{"name": f"item_{i}", "active": True} for i in range(100)]

        d = NumericDiscretizer()
        transformed = d.fit_transform(data)

        # Should pass through unchanged
        for i, obj in enumerate(transformed):
            assert obj["name"] == f"item_{i}"
            assert obj["active"] is True

    def test_single_value_field(self):
        """Test field with single unique value."""
        data = [{"value": 42.0} for _ in range(100)]

        d = NumericDiscretizer(cat_threshold=10)
        d.fit(data)

        # Single value < threshold, should pass through
        assert "value" in d.passthrough_fields

    def test_fewer_unique_than_bins(self):
        """Test when unique values < n_bins."""
        # 15 unique values, but requesting 20 bins
        # Use uniform strategy to avoid quantile-related warnings
        data = [{"value": float(i % 15)} for i in range(200)]

        d = NumericDiscretizer(cat_threshold=10, n_bins=20, strategy="uniform")
        d.fit(data)

        # Should reduce bins to match unique values
        edges = d.get_bin_edges("value")
        assert len(edges) <= 21  # At most 20 bins + 1

    def test_integer_values(self):
        """Test integer values are handled correctly."""
        data = [{"value": i} for i in range(200)]

        d = NumericDiscretizer(cat_threshold=100, n_bins=10)
        transformed = d.fit_transform(data)

        # Should convert integers to bin center floats
        for obj in transformed:
            assert isinstance(obj["value"], float)

    def test_negative_values(self):
        """Test negative values are handled correctly."""
        data = [{"value": float(i - 100)} for i in range(200)]

        d = NumericDiscretizer(cat_threshold=100, n_bins=10)
        d.fit_transform(data)

        edges = d.get_bin_edges("value")
        assert edges[0] <= -100  # Should include negative values

    def test_float_precision(self):
        """Test float values with high precision."""
        data = [{"value": i * 0.00001} for i in range(200)]

        d = NumericDiscretizer(cat_threshold=100, n_bins=10)
        transformed = d.fit_transform(data)

        # Should successfully discretize to bin center floats
        for obj in transformed:
            assert isinstance(obj["value"], float)


class TestNumericDiscretizerBinCenters:
    """Tests for bin center calculation."""

    def test_transform_returns_bin_center(self):
        """Test transform returns the center of the bin."""
        data = [{"value": float(i)} for i in range(200)]

        d = NumericDiscretizer(cat_threshold=100, n_bins=10)
        d.fit(data)

        edges = d.get_bin_edges("value")

        # Transform a value and verify it's the bin center
        result = d.transform([{"value": 50.0}])[0]

        # Find which bin 50.0 falls into
        bin_idx = None
        for i in range(len(edges) - 1):
            if edges[i] <= 50.0 < edges[i + 1]:
                bin_idx = i
                break
        # Handle edge case where value equals the last edge
        if bin_idx is None:
            bin_idx = len(edges) - 2

        expected_center = (edges[bin_idx] + edges[bin_idx + 1]) / 2
        assert abs(result["value"] - expected_center) < 0.001

    def test_all_bins_produce_valid_centers(self):
        """Test that all bins produce centers within expected range."""
        data = [{"value": float(i)} for i in range(200)]

        d = NumericDiscretizer(cat_threshold=100, n_bins=10)
        d.fit(data)

        edges = d.get_bin_edges("value")
        transformed = d.transform(data)

        # All transformed values should be valid bin centers
        valid_centers = set()
        for i in range(len(edges) - 1):
            center = (edges[i] + edges[i + 1]) / 2
            valid_centers.add(round(center, 6))  # Round to avoid float precision issues

        for obj in transformed:
            # Each value should be one of the valid bin centers
            rounded = round(obj["value"], 6)
            assert rounded in valid_centers, f"{obj['value']} not in valid centers"

    def test_bin_center_within_original_range(self):
        """Test bin centers are within the original data range."""
        data = [{"value": float(i)} for i in range(100, 200)]  # Range 100-199

        d = NumericDiscretizer(cat_threshold=50, n_bins=5)
        transformed = d.fit_transform(data)

        for obj in transformed:
            # Bin center should be within the data range (with some tolerance for bin edges)
            assert 99 <= obj["value"] <= 200, f"Value {obj['value']} outside expected range"

    def test_multiple_fields_get_correct_centers(self):
        """Test that multiple fields get their own correct bin centers."""
        data = [{"a": float(i), "b": float(i * 10)} for i in range(200)]

        d = NumericDiscretizer(cat_threshold=100, n_bins=5)
        d.fit(data)

        # Both fields should be discretized
        assert "a" in d.discretized_fields
        assert "b" in d.discretized_fields

        # Field 'a' ranges 0-199, field 'b' ranges 0-1990
        # Their bin centers should be different
        transformed = d.transform([{"a": 50.0, "b": 500.0}])[0]

        # Verify 'a' center is reasonable (around 0-199 range)
        assert 0 <= transformed["a"] <= 200

        # Verify 'b' center is reasonable (around 0-1990 range)
        assert 0 <= transformed["b"] <= 2000

        # They should not be equal
        assert transformed["a"] != transformed["b"]
