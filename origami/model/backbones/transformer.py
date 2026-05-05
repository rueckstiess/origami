"""Standard transformer backbone using PyTorch's TransformerEncoder."""

import torch
import torch.nn as nn
from torch import Tensor

from origami.config import ModelConfig

from .base import BackboneBase, make_causal_mask


class TransformerBackbone(BackboneBase):
    """Decoder-only transformer with causal attention.

    Uses PyTorch's built-in TransformerEncoderLayer for efficiency.
    Applies causal masking for autoregressive generation.

    Attributes:
        layers: Stack of TransformerEncoderLayers
        norm: Final layer normalization
    """

    def __init__(self, config: ModelConfig):
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

        # Create combined attention mask that handles both causal masking and padding
        # For left-padded sequences, we need to ensure padding positions don't cause NaN
        # by having all keys masked out in softmax.
        #
        # PyTorch's MHA with separate causal mask and key_padding_mask can cause NaN
        # when a query position is padding (at start of left-padded sequence) because:
        # - Causal mask allows attending to positions <= current
        # - But key_padding_mask masks all those positions as padding
        # - Result: softmax over all -inf = NaN
        #
        # Solution: Create a combined 4D attention mask where padding positions
        # can attend to at least one position (themselves) to avoid NaN.

        # Start with causal mask: (seq_len, seq_len), True = masked
        causal_mask = make_causal_mask(seq_len, device)

        if attention_mask is not None:
            # Create combined mask: (batch, seq_len, seq_len)
            # Start with causal mask broadcast to batch
            combined_mask = causal_mask.unsqueeze(0).expand(batch_size, -1, -1).clone()

            # Add key padding: mask out keys that are padding
            # key_padding: (batch, seq_len) -> (batch, 1, seq_len)
            key_padding = ~attention_mask  # True for padding
            combined_mask = combined_mask | key_padding.unsqueeze(1)

            # Critical fix: For padding query positions, allow them to attend to themselves
            # to avoid softmax(all -inf) = NaN. These positions' outputs don't matter
            # as they're padding, but they shouldn't produce NaN that corrupts
            # other positions through numerical instability.
            #
            # For each padding position i, ensure mask[i, i] = False (can attend to self)
            # Vectorized: unmask diagonal for padding positions
            padding_positions = ~attention_mask  # (batch, seq_len), True for padding
            # Create indices for diagonal positions
            diag_indices = torch.arange(seq_len, device=device)
            # combined_mask[b, i, i] = False where padding_positions[b, i] = True
            # Use advanced indexing: combined_mask[:, diag, diag] &= ~padding
            combined_mask[:, diag_indices, diag_indices] = (
                combined_mask[:, diag_indices, diag_indices] & ~padding_positions
            )

            # Convert to float mask for nn.TransformerEncoder
            # PyTorch expects: 0 = attend, -inf = don't attend (additive mask)
            # For 3D masks, shape must be (batch * n_heads, seq_len, seq_len)
            attn_mask = combined_mask.float().masked_fill(combined_mask, float("-inf"))

            # Expand for multi-head attention: (batch, seq, seq) -> (batch * n_heads, seq, seq)
            n_heads = self.config.n_heads
            attn_mask = attn_mask.unsqueeze(1).expand(-1, n_heads, -1, -1)
            attn_mask = attn_mask.reshape(batch_size * n_heads, seq_len, seq_len)

            # Apply transformer layers with combined mask (no separate key_padding_mask)
            hidden_states = self.layers(
                hidden_states,
                mask=attn_mask,
                src_key_padding_mask=None,
            )
        else:
            # No padding - use simple causal mask
            hidden_states = self.layers(
                hidden_states,
                mask=causal_mask,
                src_key_padding_mask=None,
            )

        # Final layer norm
        hidden_states = self.norm(hidden_states)

        return hidden_states
