"""Tests for JSON value metrics."""

import pytest

from origami.training.metrics import (
    _values_equal,
    accuracy,
    array_f1,
    get_metric,
    list_metrics,
    resolve_metrics,
)


class TestValuesEqual:
    """Tests for _values_equal comparison function."""

    # --- Numeric type tolerance (int/float) ---

    def test_int_equals_float(self):
        """int and float with same numeric value should be equal."""
        assert _values_equal(0, 0.0)
        assert _values_equal(1, 1.0)
        assert _values_equal(-3, -3.0)
        assert _values_equal(42, 42.0)

    def test_float_equals_int(self):
        """Symmetric: float/int order shouldn't matter."""
        assert _values_equal(0.0, 0)
        assert _values_equal(1.0, 1)

    def test_int_float_different_values(self):
        """int and float with different values should not be equal."""
        assert not _values_equal(0, 1.0)
        assert not _values_equal(1, 0.0)
        assert not _values_equal(2, 3.0)

    def test_float_float(self):
        """float/float comparison should work normally."""
        assert _values_equal(0.0, 0.0)
        assert _values_equal(3.14, 3.14)
        assert not _values_equal(0.0, 0.1)

    def test_int_int(self):
        """int/int comparison should work normally."""
        assert _values_equal(0, 0)
        assert _values_equal(42, 42)
        assert not _values_equal(0, 1)

    # --- Bool stays strict (bool is subclass of int in Python) ---

    def test_bool_not_equal_to_int(self):
        """bool should NOT equal int, even though bool is subclass of int."""
        assert not _values_equal(True, 1)
        assert not _values_equal(False, 0)
        assert not _values_equal(1, True)
        assert not _values_equal(0, False)

    def test_bool_not_equal_to_float(self):
        """bool should NOT equal float."""
        assert not _values_equal(True, 1.0)
        assert not _values_equal(False, 0.0)

    def test_bool_equal_to_bool(self):
        """bool/bool comparison should work."""
        assert _values_equal(True, True)
        assert _values_equal(False, False)
        assert not _values_equal(True, False)

    # --- Other types ---

    def test_string(self):
        assert _values_equal("hello", "hello")
        assert not _values_equal("hello", "world")

    def test_string_not_equal_to_number(self):
        assert not _values_equal("0", 0)
        assert not _values_equal("1", 1)

    def test_none(self):
        assert _values_equal(None, None)
        assert not _values_equal(None, 0)
        assert not _values_equal(None, "")

    def test_dict(self):
        assert _values_equal({"a": 1}, {"a": 1})
        assert _values_equal({"a": 1, "b": 2}, {"b": 2, "a": 1})
        assert not _values_equal({"a": 1}, {"a": 2})

    def test_list(self):
        assert _values_equal([1, 2, 3], [1, 2, 3])
        assert not _values_equal([1, 2], [1, 3])
        assert not _values_equal([1, 2], [1, 2, 3])

    def test_nested_list_with_mixed_numeric(self):
        """Lists containing mixed int/float should compare element-wise."""
        assert _values_equal([0, 1], [0.0, 1.0])
        assert _values_equal([1.0, 2], [1, 2.0])


class TestAccuracy:
    """Tests for accuracy metric."""

    def test_perfect_accuracy(self):
        assert accuracy([1, 2, 3], [1, 2, 3]) == 1.0

    def test_zero_accuracy(self):
        assert accuracy([1, 2, 3], [4, 5, 6]) == 0.0

    def test_partial_accuracy(self):
        assert accuracy([1, 2, 3, 4], [1, 2, 0, 0]) == 0.5

    def test_empty(self):
        assert accuracy([], []) == 1.0

    def test_mixed_int_float_perfect(self):
        """Accuracy should be 1.0 when predictions differ only in int/float type."""
        y_true = [0, 1, 0, 1, 0]
        y_pred = [0.0, 1.0, 0.0, 1.0, 0.0]
        assert accuracy(y_true, y_pred) == 1.0

    def test_mixed_int_float_reversed(self):
        """float true vs int pred should also work."""
        y_true = [0.0, 1.0, 0.0]
        y_pred = [0, 1, 0]
        assert accuracy(y_true, y_pred) == 1.0

    def test_bool_not_conflated_with_int(self):
        """Bool predictions should not match int targets."""
        y_true = [1, 0, 1]
        y_pred = [True, False, True]
        assert accuracy(y_true, y_pred) == 0.0


class TestMetricRegistry:
    """Tests for metric registry functions."""

    def test_list_metrics(self):
        names = list_metrics()
        assert "accuracy" in names
        assert "array_f1" in names

    def test_get_metric(self):
        fn = get_metric("accuracy")
        assert fn is accuracy

    def test_unknown_metric_raises(self):
        with pytest.raises(ValueError, match="Unknown metric"):
            get_metric("nonexistent")

    def test_resolve_metrics(self):
        resolved = resolve_metrics({"acc": "accuracy", "f1": "array_f1"})
        assert resolved["acc"] is accuracy
        assert resolved["f1"] is array_f1

    def test_resolve_mixed_string_and_function(self):
        resolved = resolve_metrics({"acc": "accuracy", "fn": accuracy})
        assert resolved["acc"] is accuracy
        assert resolved["fn"] is accuracy
