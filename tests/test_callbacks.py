"""Tests for trainer callbacks and metrics."""

import pytest

from origami.training.callbacks import (
    CallbackHandler,
    ProgressCallback,
    TableLogCallback,
    TrainerCallback,
)
from origami.training.metrics import (
    accuracy,
    array_f1,
    array_jaccard,
    array_precision,
    array_recall,
    object_key_accuracy,
)


class TestMetrics:
    """Tests for metric functions."""

    def test_accuracy_simple_values(self):
        """Test accuracy with simple values."""
        y_true = ["a", "b", "c"]
        y_pred = ["a", "b", "c"]
        assert accuracy(y_true, y_pred) == 1.0

        y_pred = ["a", "x", "c"]
        assert accuracy(y_true, y_pred) == pytest.approx(2 / 3)

        y_pred = ["x", "y", "z"]
        assert accuracy(y_true, y_pred) == 0.0

    def test_accuracy_numbers(self):
        """Test accuracy with numeric values."""
        y_true = [1, 2, 3]
        y_pred = [1, 2, 3]
        assert accuracy(y_true, y_pred) == 1.0

        y_pred = [1, 2, 4]
        assert accuracy(y_true, y_pred) == pytest.approx(2 / 3)

    def test_accuracy_arrays(self):
        """Test accuracy with array values."""
        y_true = [[1, 2], [3, 4]]
        y_pred = [[1, 2], [3, 4]]
        assert accuracy(y_true, y_pred) == 1.0

        # Different order = not equal for accuracy
        y_pred = [[2, 1], [3, 4]]
        assert accuracy(y_true, y_pred) == 0.5

    def test_accuracy_objects(self):
        """Test accuracy with object values (order-independent)."""
        y_true = [{"a": 1, "b": 2}]
        y_pred = [{"b": 2, "a": 1}]  # Same keys/values, different order
        assert accuracy(y_true, y_pred) == 1.0

        y_pred = [{"a": 1, "b": 3}]
        assert accuracy(y_true, y_pred) == 0.0

    def test_accuracy_empty(self):
        """Test accuracy with empty lists."""
        assert accuracy([], []) == 1.0

    def test_array_f1_exact_sets(self):
        """Test array_f1 with exact set matches."""
        y_true = [["a", "b"], ["c", "d"]]
        y_pred = [["a", "b"], ["c", "d"]]
        assert array_f1(y_true, y_pred) == 1.0

    def test_array_f1_order_independent(self):
        """Test array_f1 is order-independent."""
        y_true = [["a", "b", "c"]]
        y_pred = [["c", "b", "a"]]  # Same elements, different order
        assert array_f1(y_true, y_pred) == 1.0

    def test_array_f1_partial_overlap(self):
        """Test array_f1 with partial overlap."""
        y_true = [["a", "b", "c"]]
        y_pred = [["a", "b", "d"]]  # 2/3 overlap
        # Precision = 2/3, Recall = 2/3, F1 = 2/3
        assert array_f1(y_true, y_pred) == pytest.approx(2 / 3)

    def test_array_f1_no_overlap(self):
        """Test array_f1 with no overlap."""
        y_true = [["a", "b"]]
        y_pred = [["c", "d"]]
        assert array_f1(y_true, y_pred) == 0.0

    def test_array_f1_empty_arrays(self):
        """Test array_f1 with empty arrays."""
        assert array_f1([[]], [[]]) == 1.0
        assert array_f1([["a"]], [[]]) == 0.0
        assert array_f1([[]], [["a"]]) == 0.0

    def test_array_f1_non_list_prediction(self):
        """Test array_f1 with non-list predictions."""
        y_true = [["a", "b"]]
        y_pred = ["wrong"]
        assert array_f1(y_true, y_pred) == 0.0

        # Equal non-lists return 1.0
        y_true = ["same"]
        y_pred = ["same"]
        assert array_f1(y_true, y_pred) == 1.0

    def test_array_precision_exact(self):
        """Test array_precision with exact matches."""
        y_true = [["a", "b", "c"]]
        y_pred = [["a", "b", "c"]]
        assert array_precision(y_true, y_pred) == 1.0

    def test_array_precision_order_independent(self):
        """Test array_precision is order-independent."""
        y_true = [["a", "b", "c"]]
        y_pred = [["c", "b", "a"]]
        assert array_precision(y_true, y_pred) == 1.0

    def test_array_precision_partial_overlap(self):
        """Test array_precision with partial overlap."""
        y_true = [["a", "b", "c"]]
        y_pred = [["a", "b", "d"]]  # 2/3 of predictions are correct
        assert array_precision(y_true, y_pred) == pytest.approx(2 / 3)

    def test_array_precision_subset_prediction(self):
        """Test array_precision when prediction is subset of true."""
        y_true = [["a", "b", "c", "d"]]
        y_pred = [["a", "b"]]  # All predictions correct, but missing some
        # Precision = 2/2 = 1.0 (all predictions are correct)
        assert array_precision(y_true, y_pred) == 1.0

    def test_array_precision_superset_prediction(self):
        """Test array_precision when prediction is superset of true."""
        y_true = [["a", "b"]]
        y_pred = [["a", "b", "c", "d"]]  # Extra wrong predictions
        # Precision = 2/4 = 0.5 (half of predictions are correct)
        assert array_precision(y_true, y_pred) == 0.5

    def test_array_precision_no_overlap(self):
        """Test array_precision with no overlap."""
        y_true = [["a", "b"]]
        y_pred = [["c", "d"]]
        assert array_precision(y_true, y_pred) == 0.0

    def test_array_precision_empty_arrays(self):
        """Test array_precision with empty arrays."""
        assert array_precision([[]], [[]]) == 1.0
        assert array_precision([["a"]], [[]]) == 0.0
        assert array_precision([[]], [["a"]]) == 0.0

    def test_array_precision_non_list_prediction(self):
        """Test array_precision with non-list predictions."""
        y_true = [["a", "b"]]
        y_pred = ["wrong"]
        assert array_precision(y_true, y_pred) == 0.0

        y_true = ["same"]
        y_pred = ["same"]
        assert array_precision(y_true, y_pred) == 1.0

    def test_array_recall_exact(self):
        """Test array_recall with exact matches."""
        y_true = [["a", "b", "c"]]
        y_pred = [["a", "b", "c"]]
        assert array_recall(y_true, y_pred) == 1.0

    def test_array_recall_order_independent(self):
        """Test array_recall is order-independent."""
        y_true = [["a", "b", "c"]]
        y_pred = [["c", "b", "a"]]
        assert array_recall(y_true, y_pred) == 1.0

    def test_array_recall_partial_overlap(self):
        """Test array_recall with partial overlap."""
        y_true = [["a", "b", "c"]]
        y_pred = [["a", "b", "d"]]  # 2/3 of true values found
        assert array_recall(y_true, y_pred) == pytest.approx(2 / 3)

    def test_array_recall_subset_prediction(self):
        """Test array_recall when prediction is subset of true."""
        y_true = [["a", "b", "c", "d"]]
        y_pred = [["a", "b"]]  # Missing some true values
        # Recall = 2/4 = 0.5 (only half of true values found)
        assert array_recall(y_true, y_pred) == 0.5

    def test_array_recall_superset_prediction(self):
        """Test array_recall when prediction is superset of true."""
        y_true = [["a", "b"]]
        y_pred = [["a", "b", "c", "d"]]  # All true values found plus extras
        # Recall = 2/2 = 1.0 (all true values are found)
        assert array_recall(y_true, y_pred) == 1.0

    def test_array_recall_no_overlap(self):
        """Test array_recall with no overlap."""
        y_true = [["a", "b"]]
        y_pred = [["c", "d"]]
        assert array_recall(y_true, y_pred) == 0.0

    def test_array_recall_empty_arrays(self):
        """Test array_recall with empty arrays."""
        assert array_recall([[]], [[]]) == 1.0
        assert array_recall([["a"]], [[]]) == 0.0
        assert array_recall([[]], [["a"]]) == 0.0

    def test_array_recall_non_list_prediction(self):
        """Test array_recall with non-list predictions."""
        y_true = [["a", "b"]]
        y_pred = ["wrong"]
        assert array_recall(y_true, y_pred) == 0.0

        y_true = ["same"]
        y_pred = ["same"]
        assert array_recall(y_true, y_pred) == 1.0

    def test_precision_recall_f1_relationship(self):
        """Test that F1 is the harmonic mean of precision and recall."""
        y_true = [["a", "b", "c", "d"]]
        y_pred = [["a", "b", "e", "f"]]  # 2 correct, 2 wrong, 2 missing

        p = array_precision(y_true, y_pred)  # 2/4 = 0.5
        r = array_recall(y_true, y_pred)  # 2/4 = 0.5
        f1 = array_f1(y_true, y_pred)

        expected_f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        assert f1 == pytest.approx(expected_f1)
        assert p == pytest.approx(0.5)
        assert r == pytest.approx(0.5)
        assert f1 == pytest.approx(0.5)

    def test_array_jaccard_exact(self):
        """Test array_jaccard with exact matches."""
        y_true = [["a", "b"]]
        y_pred = [["a", "b"]]
        assert array_jaccard(y_true, y_pred) == 1.0

    def test_array_jaccard_partial(self):
        """Test array_jaccard with partial overlap."""
        y_true = [["a", "b", "c"]]
        y_pred = [["a", "b", "d"]]
        # Intersection = {a, b}, Union = {a, b, c, d}
        # Jaccard = 2/4 = 0.5
        assert array_jaccard(y_true, y_pred) == 0.5

    def test_array_jaccard_no_overlap(self):
        """Test array_jaccard with no overlap."""
        y_true = [["a"]]
        y_pred = [["b"]]
        assert array_jaccard(y_true, y_pred) == 0.0

    def test_object_key_accuracy_all_correct(self):
        """Test object_key_accuracy with all keys correct."""
        y_true = [{"a": 1, "b": 2}]
        y_pred = [{"a": 1, "b": 2, "c": 3}]  # Extra key doesn't matter
        assert object_key_accuracy(y_true, y_pred) == 1.0

    def test_object_key_accuracy_partial(self):
        """Test object_key_accuracy with partial match."""
        y_true = [{"a": 1, "b": 2}]
        y_pred = [{"a": 1, "b": 3}]  # b is wrong
        assert object_key_accuracy(y_true, y_pred) == 0.5

    def test_object_key_accuracy_missing_key(self):
        """Test object_key_accuracy with missing keys."""
        y_true = [{"a": 1, "b": 2}]
        y_pred = [{"a": 1}]  # b is missing
        assert object_key_accuracy(y_true, y_pred) == 0.5

    def test_object_key_accuracy_empty_true(self):
        """Test object_key_accuracy with empty true object."""
        y_true = [{}]
        y_pred = [{"a": 1}]
        assert object_key_accuracy(y_true, y_pred) == 1.0  # No keys to check

        y_pred = [{}]
        assert object_key_accuracy(y_true, y_pred) == 1.0

    def test_object_key_accuracy_nested(self):
        """Test object_key_accuracy with nested objects."""
        y_true = [{"a": {"x": 1}}]
        y_pred = [{"a": {"x": 1}}]
        assert object_key_accuracy(y_true, y_pred) == 1.0

        y_pred = [{"a": {"x": 2}}]
        assert object_key_accuracy(y_true, y_pred) == 0.0


class TestCallbackHandler:
    """Tests for CallbackHandler."""

    def test_fire_event_calls_callbacks(self):
        """Test that fire_event calls all registered callbacks."""
        events = []

        class TestCallback(TrainerCallback):
            def on_train_begin(self, trainer, state, metrics):
                events.append("train_begin")

            def on_epoch_end(self, trainer, state, metrics):
                events.append("epoch_end")

        handler = CallbackHandler([TestCallback()])
        handler.fire_event("on_train_begin", None, None, None)
        handler.fire_event("on_epoch_end", None, None, None)

        assert events == ["train_begin", "epoch_end"]

    def test_multiple_callbacks(self):
        """Test that multiple callbacks are called in order."""
        events = []

        class Callback1(TrainerCallback):
            def on_train_begin(self, trainer, state, metrics):
                events.append("cb1")

        class Callback2(TrainerCallback):
            def on_train_begin(self, trainer, state, metrics):
                events.append("cb2")

        handler = CallbackHandler([Callback1(), Callback2()])
        handler.fire_event("on_train_begin", None, None, None)

        assert events == ["cb1", "cb2"]

    def test_on_best_callback_fires(self):
        """Test that on_best callback is fired correctly."""
        events = []

        class TestCallback(TrainerCallback):
            def on_best(self, trainer, state, payload):
                events.append(("best", payload))

        handler = CallbackHandler([TestCallback()])
        handler.fire_event("on_best", None, None, {"val_loss": 0.5})

        assert events == [("best", {"val_loss": 0.5})]


class TestProgressCallback:
    """Tests for ProgressCallback."""

    def test_progress_callback_instantiation(self):
        """Test that ProgressCallback can be instantiated."""
        callback = ProgressCallback()
        assert callback._pbar is None

    def test_progress_callback_has_all_hooks(self):
        """Test that ProgressCallback has all expected hooks."""
        callback = ProgressCallback()
        assert hasattr(callback, "on_train_begin")
        assert hasattr(callback, "on_epoch_begin")
        assert hasattr(callback, "on_batch_end")
        assert hasattr(callback, "on_epoch_end")
        assert hasattr(callback, "on_evaluate")


class TestTableLogCallback:
    """Tests for TableLogCallback."""

    def test_table_log_callback_instantiation(self):
        """Test that TableLogCallback can be instantiated with defaults."""
        callback = TableLogCallback()
        assert callback.print_every == 10

    def test_table_log_callback_custom_params(self):
        """Test TableLogCallback with custom parameters."""
        callback = TableLogCallback(print_every=5)
        assert callback.print_every == 5

    def test_table_log_callback_has_all_hooks(self):
        """Test that TableLogCallback has all expected hooks."""
        callback = TableLogCallback()
        assert hasattr(callback, "on_batch_begin")
        assert hasattr(callback, "on_batch_end")
        assert hasattr(callback, "on_evaluate")

    def test_table_log_callback_batch_timing(self):
        """Test that TableLogCallback tracks batch timing."""
        callback = TableLogCallback()

        # Simulate batch begin
        callback.on_batch_begin(None, None, None)

        # Check that start time was recorded
        assert callback._batch_start_time > 0
