"""ORIGAMI training infrastructure.

Provides dataset wrappers, collation, training loop utilities, and callbacks.
"""

from .callbacks import (
    CallbackHandler,
    MetricsCallback,
    ProgressCallback,
    TableLogCallback,
    TrainerCallback,
)
from .collator import OrigamiDataCollator
from .dataset import EvalDataset, UpscaledDataset
from .metrics import (
    array_f1,
    array_jaccard,
    exact_match,
    object_key_accuracy,
)
from .trainer import OrigamiTrainer, TrainMetrics, TrainState

__all__ = [
    # Datasets
    "EvalDataset",
    "UpscaledDataset",
    # Collation
    "OrigamiDataCollator",
    # Trainer
    "OrigamiTrainer",
    "TrainMetrics",
    "TrainState",
    # Callbacks
    "TrainerCallback",
    "CallbackHandler",
    "ProgressCallback",
    "MetricsCallback",
    "TableLogCallback",
    # Metrics
    "exact_match",
    "array_f1",
    "array_jaccard",
    "object_key_accuracy",
]
