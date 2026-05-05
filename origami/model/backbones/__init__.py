"""Backbone modules for sequence modeling.

This module provides pluggable sequence modeling backends:
- TransformerBackbone: Standard decoder-only transformer
- CachedTransformerBackbone: Transformer with KV caching for efficient generation
- LSTMBackbone: LSTM-based backbone (not yet implemented)
- MambaBackbone: Mamba/S4 SSM backbone (not yet implemented)
"""

from origami.config import ModelConfig

from .base import BackboneBase, make_causal_mask
from .cached_transformer import (
    CachedMultiHeadAttention,
    CachedTransformerBackbone,
    CachedTransformerBlock,
    KVCache,
)
from .lstm import LSTMBackbone
from .mamba import MambaBackbone
from .transformer import TransformerBackbone

# Backbone registry
BACKBONE_CLASSES: dict[str, type[BackboneBase]] = {
    "transformer": TransformerBackbone,
    "cached_transformer": CachedTransformerBackbone,
    "lstm": LSTMBackbone,
    "mamba": MambaBackbone,
}


def create_backbone(config: ModelConfig) -> BackboneBase:
    """Create a backbone module based on configuration.

    Args:
        config: Model configuration with backbone type

    Returns:
        Backbone module instance

    Raises:
        ValueError: If backbone type is unknown
    """
    if config.backbone not in BACKBONE_CLASSES:
        raise ValueError(
            f"Unknown backbone type: {config.backbone}. "
            f"Valid options: {list(BACKBONE_CLASSES.keys())}"
        )
    return BACKBONE_CLASSES[config.backbone](config)


__all__ = [
    # Base
    "BackboneBase",
    "make_causal_mask",
    # Transformer
    "TransformerBackbone",
    # Cached Transformer
    "CachedTransformerBackbone",
    "CachedTransformerBlock",
    "CachedMultiHeadAttention",
    "KVCache",
    # Stubs
    "LSTMBackbone",
    "MambaBackbone",
    # Factory
    "BACKBONE_CLASSES",
    "create_backbone",
]
