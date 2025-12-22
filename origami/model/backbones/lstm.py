"""LSTM backbone for comparison with RNN-based approaches."""

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from origami.config import ModelConfig

from .base import BackboneBase


class LSTMBackbone(BackboneBase):
    """LSTM backbone for comparison with transformer-based approaches.

    Uses standard PyTorch LSTM with unidirectional (forward-only) processing
    for autoregressive sequence modeling. Handles left-padded sequences by
    shifting valid tokens to the start before packing.

    Attributes:
        lstm: Multi-layer LSTM module
        norm: Final layer normalization
    """

    def __init__(self, config: ModelConfig):
        """Initialize LSTM backbone.

        Args:
            config: Model configuration with d_model, lstm_num_layers, dropout
        """
        super().__init__()

        self.config = config

        # Multi-layer LSTM (unidirectional for autoregressive modeling)
        self.lstm = nn.LSTM(
            input_size=config.d_model,
            hidden_size=config.d_model,
            num_layers=config.lstm_num_layers,
            batch_first=True,
            dropout=config.dropout if config.lstm_num_layers > 1 else 0.0,
            bidirectional=False,  # Always unidirectional for autoregressive
        )

        # Final layer norm (like transformer backbone)
        self.norm = nn.LayerNorm(config.d_model)

    def forward(
        self,
        hidden_states: Tensor,  # (batch, seq_len, d_model)
        attention_mask: Tensor | None = None,  # (batch, seq_len)
    ) -> Tensor:
        """Process sequence through LSTM layers.

        Args:
            hidden_states: Input embeddings of shape (batch, seq_len, d_model)
            attention_mask: Boolean mask where True indicates valid positions,
                False indicates padding. Shape (batch, seq_len).

        Returns:
            Processed hidden states of shape (batch, seq_len, d_model)
        """
        batch_size, seq_len, d_model = hidden_states.shape
        device = hidden_states.device

        if attention_mask is not None:
            # Compute sequence lengths from attention mask
            lengths = attention_mask.sum(dim=1)  # (batch,)

            # Handle edge case: all-padding sequences get length 1 to avoid errors
            lengths = lengths.clamp(min=1)

            # CRITICAL: pack_padded_sequence assumes RIGHT-padding (valid tokens first)
            # but Origami uses LEFT-padding (valid tokens at end).
            # We must shift valid tokens to the start before packing.

            # Create indices to gather valid tokens at the start
            # For each position j in output, we want input position (seq_len - length + j)
            # But only for j < length; positions j >= length get zeros (padding)

            # Build gather indices: for each batch item, shift by (seq_len - length)
            positions = torch.arange(seq_len, device=device).unsqueeze(0)  # (1, seq_len)
            shifts = (seq_len - lengths).unsqueeze(1)  # (batch, 1)
            gather_indices = positions + shifts  # (batch, seq_len)

            # Clamp to valid range and create mask for positions beyond length
            gather_indices = gather_indices.clamp(0, seq_len - 1)
            valid_positions = positions < lengths.unsqueeze(1)  # (batch, seq_len)

            # Gather to shift valid tokens to start (right-padded format)
            gather_indices_expanded = gather_indices.unsqueeze(-1).expand(-1, -1, d_model)
            shifted_states = torch.gather(hidden_states, dim=1, index=gather_indices_expanded)

            # Zero out positions beyond each sequence's length
            shifted_states = shifted_states * valid_positions.unsqueeze(-1)

            # Now pack with right-padded data
            lengths_cpu = lengths.cpu()
            packed = pack_padded_sequence(
                shifted_states,
                lengths_cpu,
                batch_first=True,
                enforce_sorted=False,
            )

            # Process through LSTM
            packed_output, _ = self.lstm(packed)

            # Unpack back to right-padded tensor
            unpacked, _ = pad_packed_sequence(
                packed_output,
                batch_first=True,
                total_length=seq_len,
            )

            # Shift output back to left-padded format
            # For each position j in output, we want unpacked position (j - shift)
            # where shift = seq_len - length
            scatter_indices = positions - shifts  # (batch, seq_len)
            scatter_indices = scatter_indices.clamp(0, seq_len - 1)

            # Positions where we have valid output (j >= shift means j - shift >= 0)
            output_valid = positions >= shifts  # (batch, seq_len)

            scatter_indices_expanded = scatter_indices.unsqueeze(-1).expand(-1, -1, d_model)
            output = torch.gather(unpacked, dim=1, index=scatter_indices_expanded)

            # Zero out padding positions at the start
            output = output * output_valid.unsqueeze(-1)
        else:
            # No masking - process full sequences
            output, _ = self.lstm(hidden_states)

        # Apply final layer normalization
        return self.norm(output)
