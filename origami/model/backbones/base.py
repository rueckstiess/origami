"""Base classes and utilities for backbone modules."""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch import Tensor


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


def make_causal_mask(seq_len: int, device: torch.device) -> Tensor:
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
