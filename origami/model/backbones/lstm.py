"""LSTM backbone for comparison with RNN-based approaches."""

from torch import Tensor

from origami.config import ModelConfig

from .base import BackboneBase


class LSTMBackbone(BackboneBase):
    """LSTM backbone for comparison with RNN-based approaches.

    Useful for ablation: does attention matter for JSON?
    """

    def __init__(self, config: ModelConfig):
        """Initialize LSTM backbone.

        Args:
            config: Model configuration
        """
        super().__init__()
        raise NotImplementedError("LSTMBackbone not yet implemented")

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        """Not implemented."""
        raise NotImplementedError("LSTMBackbone not yet implemented")
