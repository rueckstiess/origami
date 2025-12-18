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
    accuracy,
    array_f1,
    array_jaccard,
    object_key_accuracy,
)
from .trainer import EpochStats, OrigamiTrainer, TrainState

__all__ = [
    # Datasets
    "EvalDataset",
    "UpscaledDataset",
    # Collation
    "OrigamiDataCollator",
    # Trainer
    "OrigamiTrainer",
    "EpochStats",
    "TrainState",
    # Callbacks
    "TrainerCallback",
    "CallbackHandler",
    "ProgressCallback",
    "TableLogCallback",
    # Metrics
    "accuracy",
    "array_f1",
    "array_jaccard",
    "object_key_accuracy",
]
