"""Mamba (S4/SSM) backbone for efficient long sequences."""

from torch import Tensor

from origami.config import ModelConfig

from .base import BackboneBase


class MambaBackbone(BackboneBase):
    """Mamba (S4/SSM) backbone for efficient long sequences.

    Requires mamba-ssm package.
    """

    def __init__(self, config: ModelConfig):
        """Initialize Mamba backbone.

        Args:
            config: Model configuration
        """
        super().__init__()
        raise NotImplementedError("MambaBackbone not yet implemented")

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        """Not implemented."""
        raise NotImplementedError("MambaBackbone not yet implemented")
