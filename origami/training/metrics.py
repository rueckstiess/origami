"""Metrics for evaluating JSON value predictions.

All metrics follow sklearn convention: (y_true: list, y_pred: list) -> float
where y_true and y_pred are lists of JSON values (can be any type).

Metrics can be specified by name (string) using the METRIC_REGISTRY. This allows
configuration via CLI and config files where function references aren't possible.

Example:
    # Using string names (recommended for config files and CLI)
    metrics = {"acc": "accuracy", "f1": "array_f1"}
    resolved = resolve_metrics(metrics)

    # Direct function references (still works for Python API)
    metrics = {"acc": accuracy, "f1": array_f1}
"""

from collections.abc import Callable, Hashable
from typing import Any

from sklearn.metrics import mean_absolute_error, mean_squared_error


def root_mean_squared_error(y_true: list, y_pred: list) -> float:
    """Root Mean Squared Error (RMSE)."""
    return mean_squared_error(y_true, y_pred) ** 0.5


# Metrics that require allow_complex_values=True for correct predictions.
# These metrics expect arrays or objects as predictions and will return 0.0
# if the predictor only generates primitive values.
COMPLEX_VALUE_METRICS: frozenset[str] = frozenset(
    {
        "array_f1",
        "array_precision",
        "array_recall",
        "array_jaccard",
        "object_key_accuracy",
    }
)


# Type alias for metric functions
MetricFn = Callable[[list[Any], list[Any]], float]

# Metrics can be specified as strings or functions
MetricSpec = str | MetricFn


def _get_metric_name(metric: MetricSpec) -> str:
    """Get the canonical metric name from a string or function."""
    if isinstance(metric, str):
        return metric
    return getattr(metric, "__name__", "")


def metric_requires_complex_values(metric: MetricSpec) -> bool:
    """Check if a metric requires complex values (arrays/objects).

    Works with both string metric names and function references.

    Args:
        metric: A metric name (string) or function.

    Returns:
        True if the metric requires complex value predictions.

    Example:
        >>> metric_requires_complex_values("array_f1")
        True
        >>> metric_requires_complex_values(array_f1)
        True
        >>> metric_requires_complex_values("accuracy")
        False
    """
    return _get_metric_name(metric) in COMPLEX_VALUE_METRICS


def any_metric_requires_complex_values(metrics: dict[str, MetricSpec] | None) -> bool:
    """Check if any metric in a dict requires complex values.

    Works with both string metric names and function references.

    Args:
        metrics: Dict mapping prefixes to metric names/functions, or None.

    Returns:
        True if any metric requires complex value predictions.

    Example:
        >>> any_metric_requires_complex_values({"f1": "array_f1", "acc": "accuracy"})
        True
        >>> any_metric_requires_complex_values({"acc": accuracy})
        False
    """
    if not metrics:
        return False
    return any(metric_requires_complex_values(m) for m in metrics.values())


def accuracy(y_true: list[Any], y_pred: list[Any]) -> float:
    """Fraction of predictions that exactly match the true values.

    Works for any JSON value type (str, int, float, bool, None, list, dict).
    For dicts, comparison is order-independent (keys can be in any order).

    Args:
        y_true: List of true values.
        y_pred: List of predicted values.

    Returns:
        Accuracy score (0.0 to 1.0).
    """
    if len(y_true) == 0:
        return 1.0
    matches = sum(_values_equal(t, p) for t, p in zip(y_true, y_pred, strict=True))
    return matches / len(y_true)


def _values_equal(a: Any, b: Any) -> bool:
    """Check if two JSON values are equal (order-independent for dicts).

    Numeric types (int/float) are compared by value, not type, since JSON
    does not distinguish between them. For example, int(0) == float(0.0).
    Bool is excluded from this relaxation since Python's bool is a subclass
    of int but semantically distinct in JSON (true vs 1).
    """
    # Allow int/float comparison (JSON doesn't distinguish numeric types)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if isinstance(a, bool) or isinstance(b, bool):
            return type(a) is type(b) and a == b
        return a == b
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a == b  # Python dict equality is order-independent
    if isinstance(a, list):
        if len(a) != len(b):
            return False
        return all(_values_equal(x, y) for x, y in zip(a, b, strict=True))
    return a == b


def array_f1(y_true: list[list], y_pred: list[list]) -> float:
    """Average F1 score treating arrays as sets.

    For each (true_array, pred_array) pair, computes set-based F1:
    - Precision = |pred ∩ true| / |pred|
    - Recall = |pred ∩ true| / |true|
    - F1 = 2 * precision * recall / (precision + recall)

    Ignores element order and duplicates within arrays.

    Args:
        y_true: List of true arrays.
        y_pred: List of predicted arrays.

    Returns:
        Average F1 score across all pairs (0.0 to 1.0).
    """
    if len(y_true) == 0:
        return 1.0

    f1_scores = []
    for true_arr, pred_arr in zip(y_true, y_pred, strict=True):
        f1 = _set_f1(true_arr, pred_arr)
        f1_scores.append(f1)

    return sum(f1_scores) / len(f1_scores)


def array_precision(y_true: list[list], y_pred: list[list]) -> float:
    """Average precision treating arrays as sets.

    For each (true_array, pred_array) pair, computes set-based precision:
    - Precision = |pred ∩ true| / |pred|

    Ignores element order and duplicates within arrays.

    Args:
        y_true: List of true arrays.
        y_pred: List of predicted arrays.

    Returns:
        Average precision across all pairs (0.0 to 1.0).
    """
    if len(y_true) == 0:
        return 1.0

    precision_scores = []
    for true_arr, pred_arr in zip(y_true, y_pred, strict=True):
        precision = _set_precision(true_arr, pred_arr)
        precision_scores.append(precision)

    return sum(precision_scores) / len(precision_scores)


def array_recall(y_true: list[list], y_pred: list[list]) -> float:
    """Average recall treating arrays as sets.

    For each (true_array, pred_array) pair, computes set-based recall:
    - Recall = |pred ∩ true| / |true|

    Ignores element order and duplicates within arrays.

    Args:
        y_true: List of true arrays.
        y_pred: List of predicted arrays.

    Returns:
        Average recall across all pairs (0.0 to 1.0).
    """
    if len(y_true) == 0:
        return 1.0

    recall_scores = []
    for true_arr, pred_arr in zip(y_true, y_pred, strict=True):
        recall = _set_recall(true_arr, pred_arr)
        recall_scores.append(recall)

    return sum(recall_scores) / len(recall_scores)


def _set_f1(true_arr: list, pred_arr: list) -> float:
    """Compute F1 score between two arrays treated as sets."""
    # Handle non-list predictions (e.g., model predicted wrong type)
    if not isinstance(true_arr, list) or not isinstance(pred_arr, list):
        return 1.0 if true_arr == pred_arr else 0.0

    # Convert to comparable sets (handle unhashable elements)
    true_set = _to_comparable_set(true_arr)
    pred_set = _to_comparable_set(pred_arr)

    if len(pred_set) == 0 and len(true_set) == 0:
        return 1.0
    if len(pred_set) == 0 or len(true_set) == 0:
        return 0.0

    intersection = len(true_set & pred_set)
    precision = intersection / len(pred_set)
    recall = intersection / len(true_set)

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _set_precision(true_arr: list, pred_arr: list) -> float:
    """Compute precision between two arrays treated as sets."""
    # Handle non-list predictions (e.g., model predicted wrong type)
    if not isinstance(true_arr, list) or not isinstance(pred_arr, list):
        return 1.0 if true_arr == pred_arr else 0.0

    # Convert to comparable sets (handle unhashable elements)
    true_set = _to_comparable_set(true_arr)
    pred_set = _to_comparable_set(pred_arr)

    if len(pred_set) == 0 and len(true_set) == 0:
        return 1.0
    if len(pred_set) == 0:
        return 0.0

    intersection = len(true_set & pred_set)
    return intersection / len(pred_set)


def _set_recall(true_arr: list, pred_arr: list) -> float:
    """Compute recall between two arrays treated as sets."""
    # Handle non-list predictions (e.g., model predicted wrong type)
    if not isinstance(true_arr, list) or not isinstance(pred_arr, list):
        return 1.0 if true_arr == pred_arr else 0.0

    # Convert to comparable sets (handle unhashable elements)
    true_set = _to_comparable_set(true_arr)
    pred_set = _to_comparable_set(pred_arr)

    if len(pred_set) == 0 and len(true_set) == 0:
        return 1.0
    if len(true_set) == 0:
        return 0.0

    intersection = len(true_set & pred_set)
    return intersection / len(true_set)


def _to_comparable_set(arr: list) -> set:
    """Convert array to a set, handling unhashable elements."""
    result = set()
    for item in arr:
        if isinstance(item, Hashable):
            result.add(item)
        else:
            # For unhashable items (lists, dicts), use their string repr
            result.add(repr(item))
    return result


def array_jaccard(y_true: list[list], y_pred: list[list]) -> float:
    """Average Jaccard similarity for array predictions.

    Jaccard = |intersection| / |union|

    Ignores element order and duplicates within arrays.

    Args:
        y_true: List of true arrays.
        y_pred: List of predicted arrays.

    Returns:
        Average Jaccard similarity across all pairs (0.0 to 1.0).
    """
    if len(y_true) == 0:
        return 1.0

    jaccard_scores = []
    for true_arr, pred_arr in zip(y_true, y_pred, strict=True):
        jaccard = _jaccard(true_arr, pred_arr)
        jaccard_scores.append(jaccard)

    return sum(jaccard_scores) / len(jaccard_scores)


def _jaccard(true_arr: list, pred_arr: list) -> float:
    """Compute Jaccard similarity between two arrays treated as sets."""
    # Handle non-list predictions
    if not isinstance(true_arr, list) or not isinstance(pred_arr, list):
        return 1.0 if true_arr == pred_arr else 0.0

    true_set = _to_comparable_set(true_arr)
    pred_set = _to_comparable_set(pred_arr)

    if len(true_set) == 0 and len(pred_set) == 0:
        return 1.0

    intersection = len(true_set & pred_set)
    union = len(true_set | pred_set)

    return intersection / union if union > 0 else 0.0


def object_key_accuracy(y_true: list[dict], y_pred: list[dict]) -> float:
    """Average fraction of keys with correct values in object predictions.

    For each (true_obj, pred_obj) pair:
    - Score = (# keys with matching values) / (# keys in true object)

    Only considers keys present in the true object.

    Args:
        y_true: List of true objects (dicts).
        y_pred: List of predicted objects (dicts).

    Returns:
        Average key accuracy across all pairs (0.0 to 1.0).
    """
    if len(y_true) == 0:
        return 1.0

    accuracies = []
    for true_obj, pred_obj in zip(y_true, y_pred, strict=True):
        acc = _key_accuracy(true_obj, pred_obj)
        accuracies.append(acc)

    return sum(accuracies) / len(accuracies)


def _key_accuracy(true_obj: dict, pred_obj: dict) -> float:
    """Compute key-level accuracy between two objects."""
    # Handle non-dict predictions
    if not isinstance(true_obj, dict) or not isinstance(pred_obj, dict):
        return 1.0 if true_obj == pred_obj else 0.0

    # If no keys to check, accuracy is 1.0 (vacuously true)
    if len(true_obj) == 0:
        return 1.0

    correct = 0
    for key, true_val in true_obj.items():
        if key in pred_obj and _values_equal(true_val, pred_obj[key]):
            correct += 1

    return correct / len(true_obj)


# Registry mapping metric names to functions.
# This enables string-based metric specification in configs and CLI.
METRIC_REGISTRY: dict[str, MetricFn] = {
    "accuracy": accuracy,
    "array_f1": array_f1,
    "array_precision": array_precision,
    "array_recall": array_recall,
    "array_jaccard": array_jaccard,
    "object_key_accuracy": object_key_accuracy,
    # Regression metrics (sklearn)
    "mse": mean_squared_error,
    "mae": mean_absolute_error,
    "rmse": root_mean_squared_error,
}

# Optimization direction for metrics: "maximize" (higher is better) or "minimize" (lower is better).
# Used by Trainer to determine when on_best callback should fire.
METRIC_DIRECTION: dict[str, str] = {
    # Classification metrics (maximize)
    "accuracy": "maximize",
    "array_f1": "maximize",
    "array_precision": "maximize",
    "array_recall": "maximize",
    "array_jaccard": "maximize",
    "object_key_accuracy": "maximize",
    # Regression metrics (minimize)
    "mse": "minimize",
    "mae": "minimize",
    "rmse": "minimize",
    # Special (always computed by evaluator)
    "loss": "minimize",
}


def get_metric_direction(metric: str) -> str | None:
    """Get optimization direction for a metric.

    Args:
        metric: The metric name (e.g., "accuracy", "rmse", "loss").

    Returns:
        "maximize" if higher is better, "minimize" if lower is better,
        or None if the metric is not registered (requires explicit direction).

    Example:
        >>> get_metric_direction("accuracy")
        'maximize'
        >>> get_metric_direction("rmse")
        'minimize'
        >>> get_metric_direction("custom_metric")
        None
    """
    return METRIC_DIRECTION.get(metric)


def get_metric(name: str) -> MetricFn:
    """Get a metric function by name.

    Args:
        name: The metric name (e.g., "accuracy", "array_f1").

    Returns:
        The metric function.

    Raises:
        ValueError: If the metric name is not recognized.

    Example:
        >>> fn = get_metric("accuracy")
        >>> fn([1, 2, 3], [1, 2, 3])
        1.0
    """
    if name not in METRIC_REGISTRY:
        available = ", ".join(sorted(METRIC_REGISTRY.keys()))
        raise ValueError(f"Unknown metric: '{name}'. Available metrics: {available}")
    return METRIC_REGISTRY[name]


def resolve_metrics(metrics: dict[str, MetricSpec]) -> dict[str, MetricFn]:
    """Resolve a dict of metric specifications to functions.

    Accepts both string metric names and direct function references.
    This allows gradual migration from function-based to string-based configs.

    Args:
        metrics: Dict mapping prefixes to metric names or functions.
            Example: {"acc": "accuracy", "f1": array_f1}

    Returns:
        Dict mapping the same prefixes to metric functions.

    Raises:
        ValueError: If a string metric name is not recognized.

    Example:
        >>> metrics = {"acc": "accuracy", "f1": "array_f1"}
        >>> resolved = resolve_metrics(metrics)
        >>> resolved["acc"]([1, 2, 3], [1, 2, 3])
        1.0
    """
    result: dict[str, MetricFn] = {}
    for prefix, metric in metrics.items():
        if isinstance(metric, str):
            result[prefix] = get_metric(metric)
        else:
            result[prefix] = metric
    return result


def list_metrics() -> list[str]:
    """List all available metric names.

    Returns:
        Sorted list of metric names.

    Example:
        >>> list_metrics()
        ['accuracy', 'array_f1', 'array_jaccard', ...]
    """
    return sorted(METRIC_REGISTRY.keys())
