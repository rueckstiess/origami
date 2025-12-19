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
from .dataset import EvalDataset, UpscaledDataset
from .metrics import (
    COMPLEX_VALUE_METRICS,
    accuracy,
    any_metric_requires_complex_values,
    array_f1,
    array_jaccard,
    metric_requires_complex_values,
    object_key_accuracy,
)
from .trainer import EpochStats, OrigamiTrainer, TrainResult

__all__ = [
    # Datasets
    "EvalDataset",
    "UpscaledDataset",
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
    "accuracy",
    "any_metric_requires_complex_values",
    "array_f1",
    "array_jaccard",
    "metric_requires_complex_values",
    "object_key_accuracy",
]
