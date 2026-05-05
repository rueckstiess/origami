"""ORIGAMI model module.

This module provides the core model components:
- ModelConfig: Model architecture configuration
- OrigamiModel: Main model class
- OrigamiOutput: Model output dataclass
- Backbone classes: TransformerBackbone, CachedTransformerBackbone, etc.
- Head classes: DiscreteHead, ContinuousHead
"""

from origami.config import ModelConfig

from .backbones import (
    BACKBONE_CLASSES,
    BackboneBase,
    CachedMultiHeadAttention,
    CachedTransformerBackbone,
    CachedTransformerBlock,
    KVCache,
    LSTMBackbone,
    MambaBackbone,
    TransformerBackbone,
    create_backbone,
)
from .embeddings import OrigamiEmbeddings
from .heads import ContinuousHead, DiscreteHead
from .origami_model import OrigamiModel, OrigamiOutput

__all__ = [
    # Configuration
    "ModelConfig",
    # Model
    "OrigamiModel",
    "OrigamiOutput",
    # Embeddings
    "OrigamiEmbeddings",
    # Backbones
    "BackboneBase",
    "TransformerBackbone",
    "CachedTransformerBackbone",
    "CachedTransformerBlock",
    "CachedMultiHeadAttention",
    "KVCache",
    "LSTMBackbone",
    "MambaBackbone",
    "BACKBONE_CLASSES",
    "create_backbone",
    # Heads
    "DiscreteHead",
    "ContinuousHead",
]
