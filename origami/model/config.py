"""ORIGAMI model configuration.

Defines the configuration dataclass for all model hyperparameters.
"""

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class OrigamiConfig:
    """Configuration for ORIGAMI model.

    Attributes:
        vocab_size: Size of the vocabulary (required).
        max_depth: Maximum nesting depth for KVPE position encoding.
        max_array_position: Maximum array index for position embeddings.
        d_model: Model embedding dimension.
        n_heads: Number of attention heads.
        n_layers: Number of transformer layers.
        d_ff: Feed-forward hidden dimension.
        dropout: Dropout probability.
        backbone: Type of sequence modeling backbone.
        lstm_bidirectional: Whether LSTM backbone is bidirectional.
        lstm_num_layers: Number of LSTM layers.
        kvpe_pooling: Pooling strategy for KVPE.
        kvpe_pooling_kwargs: Additional kwargs for pooling strategy.
        use_continuous_head: Whether to use continuous (MoG) head.
        num_mixture_components: Number of mixture components for continuous head.
        max_seq_length: Maximum sequence length.
        use_grammar_constraints: Whether to apply grammar constraints to logits.
    """

    # Vocabulary (required)
    vocab_size: int

    # Maximum nesting depth and array position
    max_depth: int = 32
    max_array_position: int = 256

    # Architecture
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 1024
    dropout: float = 0.1

    # Backbone (pluggable)
    backbone: Literal["transformer", "lstm", "mamba"] = "transformer"

    # LSTM-specific
    lstm_bidirectional: bool = False
    lstm_num_layers: int = 2

    # Position encoding (pluggable)
    kvpe_pooling: Literal["sum", "weighted", "rotary", "gru", "transformer"] = "sum"
    kvpe_pooling_kwargs: dict[str, Any] = field(default_factory=dict)

    # Continuous head (optional, Phase 6)
    use_continuous_head: bool = False
    num_mixture_components: int = 5

    # Sequence limits
    max_seq_length: int = 512

    # Grammar constraints
    use_grammar_constraints: bool = True

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {self.vocab_size}")
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )
        if self.dropout < 0 or self.dropout > 1:
            raise ValueError(f"dropout must be in [0, 1], got {self.dropout}")


@dataclass
class TrainingConfig:
    """Configuration for ORIGAMI training.

    Attributes:
        learning_rate: Learning rate for optimizer.
        batch_size: Training batch size.
        num_epochs: Number of training epochs.
        warmup_steps: Number of warmup steps for LR scheduler.
        weight_decay: Weight decay for optimizer.
        shuffle_keys: Whether to shuffle key order during tokenization.
        upscale_factor: Upscaling factor for data augmentation.
        continuous_loss_weight: Weight for continuous head loss.
        save_every_n_epochs: Save checkpoint every N epochs.
        eval_every_n_steps: Evaluate every N steps.
    """

    # Optimization
    learning_rate: float = 1e-3
    batch_size: int = 32
    num_epochs: int = 100
    warmup_steps: int = 1000
    weight_decay: float = 0.01

    # Shuffling and upscaling
    shuffle_keys: bool = True
    upscale_factor: int = 1

    # Continuous loss weight (if using continuous head)
    continuous_loss_weight: float = 1.0

    # Checkpointing
    save_every_n_epochs: int = 10
    eval_every_n_steps: int = 500
