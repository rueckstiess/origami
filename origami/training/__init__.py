"""ORIGAMI training infrastructure.

Provides dataset wrappers, collation, training loop utilities, and callbacks.
"""

from .callbacks import (
    CallbackHandler,
    ProgressCallback,
    TableLogCallback,
    TrainerCallback,
)
from .collator import OrigamiDataCollator
from .dataset import OrigamiDataset
from .metrics import (
    COMPLEX_VALUE_METRICS,
    METRIC_REGISTRY,
    accuracy,
    any_metric_requires_complex_values,
    array_f1,
    array_jaccard,
    array_precision,
    array_recall,
    get_metric,
    list_metrics,
    metric_requires_complex_values,
    object_key_accuracy,
    resolve_metrics,
)
from .trainer import EpochStats, OrigamiTrainer, TrainResult

__all__ = [
    # Dataset
    "OrigamiDataset",
    # Collation
    "OrigamiDataCollator",
    # Trainer
    "OrigamiTrainer",
    "EpochStats",
    "TrainResult",
    # Callbacks
    "TrainerCallback",
    "CallbackHandler",
    "ProgressCallback",
    "TableLogCallback",
    # Metrics
    "COMPLEX_VALUE_METRICS",
    "METRIC_REGISTRY",
    "accuracy",
    "any_metric_requires_complex_values",
    "array_f1",
    "array_jaccard",
    "array_precision",
    "array_recall",
    "get_metric",
    "list_metrics",
    "metric_requires_complex_values",
    "object_key_accuracy",
    "resolve_metrics",
]
