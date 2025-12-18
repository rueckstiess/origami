"""Tests for trainer callbacks and metrics."""

import pytest

from origami.training.callbacks import (
    CallbackHandler,
    MetricsCallback,
    ProgressCallback,
    TableLogCallback,
    TrainerCallback,
)
from origami.training.metrics import (
    array_f1,
    array_jaccard,
    exact_match,
    object_key_accuracy,
)


class TestMetrics:
    """Tests for metric functions."""

    def test_exact_match_simple_values(self):
        """Test exact_match with simple values."""
        y_true = ["a", "b", "c"]
        y_pred = ["a", "b", "c"]
        assert exact_match(y_true, y_pred) == 1.0

        y_pred = ["a", "x", "c"]
        assert exact_match(y_true, y_pred) == pytest.approx(2 / 3)

        y_pred = ["x", "y", "z"]
        assert exact_match(y_true, y_pred) == 0.0

    def test_exact_match_numbers(self):
        """Test exact_match with numeric values."""
        y_true = [1, 2, 3]
        y_pred = [1, 2, 3]
        assert exact_match(y_true, y_pred) == 1.0

        y_pred = [1, 2, 4]
        assert exact_match(y_true, y_pred) == pytest.approx(2 / 3)

    def test_exact_match_arrays(self):
        """Test exact_match with array values."""
        y_true = [[1, 2], [3, 4]]
        y_pred = [[1, 2], [3, 4]]
        assert exact_match(y_true, y_pred) == 1.0

        # Different order = not equal for exact_match
        y_pred = [[2, 1], [3, 4]]
        assert exact_match(y_true, y_pred) == 0.5

    def test_exact_match_objects(self):
        """Test exact_match with object values (order-independent)."""
        y_true = [{"a": 1, "b": 2}]
        y_pred = [{"b": 2, "a": 1}]  # Same keys/values, different order
        assert exact_match(y_true, y_pred) == 1.0

        y_pred = [{"a": 1, "b": 3}]
        assert exact_match(y_true, y_pred) == 0.0

    def test_exact_match_empty(self):
        """Test exact_match with empty lists."""
        assert exact_match([], []) == 1.0

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

    def test_log_every_n_batches(self):
        """Test that batch callbacks respect log_every_n_batches."""
        batch_events = []

        class TestCallback(TrainerCallback):
            def on_batch_end(self, trainer, state, metrics):
                batch_events.append("batch_end")

        handler = CallbackHandler([TestCallback()], log_every_n_batches=2)

        # Simulate 5 batches
        for _ in range(5):
            handler.fire_event("on_batch_begin", None, None, None)
            handler.fire_event("on_batch_end", None, None, None)

        # Should only fire on batches 2 and 4 (every 2nd batch)
        assert len(batch_events) == 2

    def test_batch_count_resets_on_epoch(self):
        """Test that batch count resets at epoch start."""
        batch_events = []

        class TestCallback(TrainerCallback):
            def on_batch_end(self, trainer, state, metrics):
                batch_events.append("batch_end")

        handler = CallbackHandler([TestCallback()], log_every_n_batches=2)

        # Epoch 1: 3 batches
        for _ in range(3):
            handler.fire_event("on_batch_begin", None, None, None)
            handler.fire_event("on_batch_end", None, None, None)

        # Reset at epoch start
        handler.fire_event("on_epoch_begin", None, None, None)

        # Epoch 2: 3 batches
        for _ in range(3):
            handler.fire_event("on_batch_begin", None, None, None)
            handler.fire_event("on_batch_end", None, None, None)

        # Should fire on batch 2 of each epoch (2 total)
        assert len(batch_events) == 2


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


class TestMetricsCallback:
    """Tests for MetricsCallback."""

    def test_metrics_callback_instantiation(self):
        """Test that MetricsCallback can be instantiated."""
        callback = MetricsCallback(target_key="label")
        assert callback.target_key == "label"
        assert callback._predictor is None
        assert "train_accuracy" in callback.history
        assert "eval_accuracy" in callback.history

    def test_metrics_callback_custom_metrics(self):
        """Test MetricsCallback with custom metrics."""

        def custom_metric(y_true, y_pred):
            return 0.5

        callback = MetricsCallback(
            target_key="label",
            metrics={"custom": custom_metric, "exact": exact_match},
        )
        assert "train_custom" in callback.history
        assert "eval_custom" in callback.history
        assert "train_exact" in callback.history
        assert "eval_exact" in callback.history

    def test_metrics_callback_sample_sizes(self):
        """Test MetricsCallback respects sample sizes."""
        callback = MetricsCallback(
            target_key="label",
            train_sample_size=50,
            eval_sample_size=25,
        )
        assert callback.train_sample_size == 50
        assert callback.eval_sample_size == 25

    def test_metrics_callback_epoch_frequency(self):
        """Test MetricsCallback respects compute_every_n_epochs."""
        callback = MetricsCallback(
            target_key="label",
            compute_every_n_epochs=5,
        )
        assert callback.compute_every_n_epochs == 5


class TestTableLogCallback:
    """Tests for TableLogCallback."""

    def test_table_log_callback_instantiation(self):
        """Test that TableLogCallback can be instantiated."""
        callback = TableLogCallback()
        assert callback.print_every == 10
        assert callback.eval_every == 100
        assert callback.target_key is None

    def test_table_log_callback_custom_params(self):
        """Test TableLogCallback with custom parameters."""
        callback = TableLogCallback(
            print_every=5,
            eval_every=50,
            target_key="category",
            train_sample_size=200,
            eval_sample_size=150,
        )
        assert callback.print_every == 5
        assert callback.eval_every == 50
        assert callback.target_key == "category"
        assert callback.train_sample_size == 200
        assert callback.eval_sample_size == 150

    def test_table_log_callback_has_all_hooks(self):
        """Test that TableLogCallback has all expected hooks."""
        callback = TableLogCallback()
        assert hasattr(callback, "on_train_begin")
        assert hasattr(callback, "on_epoch_begin")
        assert hasattr(callback, "on_batch_begin")
        assert hasattr(callback, "on_batch_end")

    def test_table_log_callback_batch_timing(self):
        """Test that TableLogCallback tracks batch timing."""
        callback = TableLogCallback()

        # Simulate batch begin
        callback.on_batch_begin(None, None, None)

        # Check that start time was recorded
        assert callback._batch_start_time > 0

    def test_table_log_callback_reset_on_train_begin(self):
        """Test that TableLogCallback resets state on train begin."""
        callback = TableLogCallback()
        callback._last_train_acc = 0.9
        callback._last_val_loss = 0.5
        callback._last_val_acc = 0.85

        callback.on_train_begin(None, None, None)

        assert callback._last_train_acc is None
        assert callback._last_val_loss is None
        assert callback._last_val_acc is None
