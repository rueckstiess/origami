"""ORIGAMI model module.

This module provides the core model components:
- OrigamiConfig: Model configuration
- OrigamiModel: Main model class
- OrigamiOutput: Model output dataclass
- Backbone classes: TransformerBackbone, etc.
- Head classes: DiscreteHead, ContinuousHead
"""

from .backbone import (
    BACKBONE_CLASSES,
    BackboneBase,
    LSTMBackbone,
    MambaBackbone,
    TransformerBackbone,
    create_backbone,
)
from .config import OrigamiConfig, TrainingConfig
from .embeddings import OrigamiEmbeddings
from .heads import ContinuousHead, DiscreteHead
from .origami_model import OrigamiModel, OrigamiOutput

__all__ = [
    # Configuration
    "OrigamiConfig",
    "TrainingConfig",
    # Model
    "OrigamiModel",
    "OrigamiOutput",
    # Embeddings
    "OrigamiEmbeddings",
    # Backbones
    "BackboneBase",
    "TransformerBackbone",
    "LSTMBackbone",
    "MambaBackbone",
    "BACKBONE_CLASSES",
    "create_backbone",
    # Heads
    "DiscreteHead",
    "ContinuousHead",
]
