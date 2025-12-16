"""ORIGAMI position encoding module.

This module provides Key-Value Position Encoding (KVPE) for encoding
hierarchical JSON paths into position embeddings.
"""

from .kvpe import (
    PATH_TYPE_INDEX,
    PATH_TYPE_KEY,
    PATH_TYPE_PAD,
    KeyValuePositionEncoding,
    PathEncoder,
)
from .pooling import (
    POOLING_CLASSES,
    GRUPooling,
    PathPooling,
    RotaryPooling,
    SumPooling,
    TransformerPooling,
    WeightedSumPooling,
    create_pooling,
    make_depth_mask,
)

__all__ = [
    # KVPE main module
    "KeyValuePositionEncoding",
    "PathEncoder",
    # Path type constants
    "PATH_TYPE_PAD",
    "PATH_TYPE_KEY",
    "PATH_TYPE_INDEX",
    # Pooling strategies
    "PathPooling",
    "SumPooling",
    "WeightedSumPooling",
    "RotaryPooling",
    "GRUPooling",
    "TransformerPooling",
    "POOLING_CLASSES",
    "create_pooling",
    # Utilities
    "make_depth_mask",
]
