"""ORIGAMI backbone modules.

Provides pluggable sequence modeling backends (Transformer, LSTM, Mamba).
MVP implements TransformerBackbone only.
"""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch import Tensor

from .config import OrigamiConfig


class BackboneBase(nn.Module, ABC):
    """Abstract base for sequence modeling backbones.

    All backbones take hidden states and optional attention mask,
    and return processed hidden states of the same shape.
    """

    @abstractmethod
    def forward(
        self,
        hidden_states: Tensor,  # (batch, seq_len, d_model)
        attention_mask: Tensor | None = None,  # (batch, seq_len)
    ) -> Tensor:  # (batch, seq_len, d_model)
        """Process sequence, return hidden states.

        Args:
            hidden_states: Input embeddings of shape (batch, seq_len, d_model)
            attention_mask: Boolean mask where True indicates valid positions,
                False indicates padding. Shape (batch, seq_len).

        Returns:
            Processed hidden states of shape (batch, seq_len, d_model)
        """
        ...


def _make_causal_mask(seq_len: int, device: torch.device) -> Tensor:
    """Create causal attention mask for autoregressive modeling.

    Returns a boolean mask where position i can only attend to positions <= i.
    True = masked (cannot attend), False = not masked (can attend).

    Args:
        seq_len: Sequence length
        device: Device for the mask tensor

    Returns:
        Causal mask of shape (seq_len, seq_len), boolean, True for masked positions
    """
    # Create upper triangular boolean matrix (True = cannot attend)
    mask = torch.triu(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device=device),
        diagonal=1,
    )
    return mask


class TransformerBackbone(BackboneBase):
    """Decoder-only transformer with causal attention.

    Uses PyTorch's built-in TransformerEncoderLayer for efficiency.
    Applies causal masking for autoregressive generation.

    Attributes:
        layers: Stack of TransformerEncoderLayers
        norm: Final layer normalization
    """

    def __init__(self, config: OrigamiConfig):
        """Initialize transformer backbone.

        Args:
            config: Model configuration
        """
        super().__init__()

        self.config = config

        # Use PyTorch's TransformerEncoderLayer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,  # (batch, seq, feature) format
            norm_first=True,  # Pre-norm architecture (more stable training)
        )

        self.layers = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.n_layers,
            enable_nested_tensor=False,  # Needed for attention masks
        )

        # Final layer norm (post-norm after all layers)
        self.norm = nn.LayerNorm(config.d_model)

    def forward(
        self,
        hidden_states: Tensor,  # (batch, seq_len, d_model)
        attention_mask: Tensor | None = None,  # (batch, seq_len)
    ) -> Tensor:
        """Apply transformer layers with causal attention.

        Args:
            hidden_states: Input embeddings of shape (batch, seq_len, d_model)
            attention_mask: Boolean mask where True indicates valid positions.
                Shape (batch, seq_len). If None, all positions are valid.

        Returns:
            Processed hidden states of shape (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = hidden_states.shape
        device = hidden_states.device

        # Create causal mask: (seq_len, seq_len)
        causal_mask = _make_causal_mask(seq_len, device)

        # Create key padding mask from attention_mask
        # PyTorch expects True for masked (padding) positions
        if attention_mask is not None:
            # Invert: True (valid) -> False (don't mask), False (pad) -> True (mask)
            key_padding_mask = ~attention_mask
        else:
            key_padding_mask = None

        # Apply transformer layers
        hidden_states = self.layers(
            hidden_states,
            mask=causal_mask,
            src_key_padding_mask=key_padding_mask,
        )

        # Final layer norm
        hidden_states = self.norm(hidden_states)

        return hidden_states


class LSTMBackbone(BackboneBase):
    """LSTM backbone for comparison with RNN-based approaches.

    Useful for ablation: does attention matter for JSON?
    Implemented in Phase 6.
    """

    def __init__(self, config: OrigamiConfig):
        """Initialize LSTM backbone.

        Args:
            config: Model configuration
        """
        super().__init__()
        raise NotImplementedError("LSTMBackbone will be implemented in Phase 6")

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        """Not implemented."""
        raise NotImplementedError("LSTMBackbone will be implemented in Phase 6")


class MambaBackbone(BackboneBase):
    """Mamba (S4/SSM) backbone for efficient long sequences.

    Requires mamba-ssm package.
    Implemented in Phase 6.
    """

    def __init__(self, config: OrigamiConfig):
        """Initialize Mamba backbone.

        Args:
            config: Model configuration
        """
        super().__init__()
        raise NotImplementedError("MambaBackbone will be implemented in Phase 6")

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        """Not implemented."""
        raise NotImplementedError("MambaBackbone will be implemented in Phase 6")


# Factory for backbone creation
BACKBONE_CLASSES = {
    "transformer": TransformerBackbone,
    "lstm": LSTMBackbone,
    "mamba": MambaBackbone,
}


def create_backbone(config: OrigamiConfig) -> BackboneBase:
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
