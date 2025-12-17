"""Pipeline configuration for ORIGAMI.

Unified configuration that combines model architecture, training parameters,
and preprocessing options into a single dataclass.
"""

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class PipelineConfig:
    """Unified configuration for ORIGAMI pipeline.

    Combines model architecture, training parameters, and preprocessing options
    into a single configuration object. Uses sensible defaults that work well
    for most use cases.

    Example:
        ```python
        # Use defaults
        config = PipelineConfig()

        # Customize for continuous numerics
        config = PipelineConfig(
            numeric_mode="scale",
            d_model=128,
            n_layers=4,
        )

        # Customize for discrete bins
        config = PipelineConfig(
            numeric_mode="discretize",
            n_bins=20,
            bin_strategy="quantile",
        )
        ```

    Attributes:
        d_model: Model hidden dimension
        n_heads: Number of attention heads
        n_layers: Number of transformer layers
        d_ff: Feed-forward dimension
        dropout: Dropout rate
        max_depth: Maximum nesting depth for JSON paths
        max_array_position: Maximum array index to encode
        kvpe_pooling: KVPE pooling strategy

        numeric_mode: How to handle high-cardinality numeric fields
            - "none": Pass through as categorical tokens
            - "discretize": Bin into discrete categories
            - "scale": Use continuous head with Mixture of Gaussians
        cat_threshold: Fields with more unique values than this are processed
        n_bins: Number of bins for discretization (numeric_mode="discretize")
        bin_strategy: Binning strategy for discretization
        num_mixture_components: MoG components (numeric_mode="scale")

        batch_size: Training batch size
        learning_rate: Initial learning rate
        num_epochs: Number of training epochs
        warmup_steps: Linear warmup steps for learning rate
        weight_decay: AdamW weight decay

        shuffle_keys: Whether to shuffle key order during training
        upscale_factor: Data augmentation factor for key shuffling

        use_grammar_constraints: Whether to enforce valid JSON syntax
    """

    # Model architecture
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 512
    dropout: float = 0.1
    max_depth: int = 8
    max_array_position: int = 64
    kvpe_pooling: Literal["sum", "weighted", "rotary", "gru", "transformer"] = "sum"

    # Numeric handling
    numeric_mode: Literal["none", "discretize", "scale"] = "none"
    cat_threshold: int = 100
    n_bins: int = 20
    bin_strategy: Literal["quantile", "uniform", "kmeans"] = "quantile"
    num_mixture_components: int = 5

    # Training
    batch_size: int = 32
    learning_rate: float = 1e-3
    num_epochs: int = 10
    warmup_steps: int = 1000
    weight_decay: float = 0.01
    save_every_n_epochs: int = 5

    # Data augmentation
    shuffle_keys: bool = True
    upscale_factor: int = 1

    # Grammar
    use_grammar_constraints: bool = True

    # Internal - set during fit
    _vocab_size: int = field(default=0, repr=False)
    _max_seq_length: int = field(default=2048, repr=False)

    def __post_init__(self):
        """Validate configuration values."""
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )
        if self.n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {self.n_layers}")
        if self.cat_threshold < 1:
            raise ValueError(f"cat_threshold must be >= 1, got {self.cat_threshold}")
        if self.numeric_mode == "discretize" and self.n_bins < 2:
            raise ValueError(f"n_bins must be >= 2, got {self.n_bins}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate}")
