"""ORIGAMI configuration classes.

Provides a nested configuration structure for model, training, data preprocessing,
and runtime settings. Designed for composability with yanex experiment tracking.

Example:
    ```python
    from origami import OrigamiConfig, ModelConfig, TrainingConfig, DataConfig

    # Simple usage with defaults
    config = OrigamiConfig()

    # Customize specific sections
    config = OrigamiConfig(
        model=ModelConfig(d_model=256, n_layers=8),
        training=TrainingConfig(batch_size=64, num_epochs=20),
        data=DataConfig(numeric_mode="scale"),
    )

    # From yanex YAML dict
    config = OrigamiConfig(
        model=ModelConfig(**yaml_dict["model"]),
        training=TrainingConfig(**yaml_dict["training"]),
        data=DataConfig(**yaml_dict.get("data", {})),
        device=yaml_dict.get("device", "auto"),
    )
    ```
"""

from collections.abc import Callable
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Literal

import yaml


class PrettyReprMixin:
    """Mixin that provides pretty-printed __repr__ and YAML output for dataclasses."""

    def __repr__(self) -> str:
        """Return a nicely formatted multi-line representation."""
        return self._format_repr(indent=0)

    def _format_repr(self, indent: int = 0) -> str:
        """Format with proper indentation for nested configs."""
        class_name = self.__class__.__name__
        field_strs = []
        base_indent = "    " * indent
        field_indent = "    " * (indent + 1)

        for f in fields(self):
            value = getattr(self, f.name)
            # Check if value is also a PrettyReprMixin (nested config)
            if isinstance(value, PrettyReprMixin):
                formatted_value = value._format_repr(indent + 1)
            else:
                formatted_value = repr(value)
            field_strs.append(f"{field_indent}{f.name}={formatted_value},")

        if not field_strs:
            return f"{class_name}()"

        fields_block = "\n".join(field_strs)
        return f"{class_name}(\n{fields_block}\n{base_indent})"

    def to_yaml(self) -> str:
        """Return config as a YAML-formatted string."""
        return yaml.dump(
            self._to_yaml_dict(asdict(self)), default_flow_style=False, sort_keys=False
        )

    def _to_yaml_dict(self, d: dict[str, Any]) -> dict[str, Any]:
        """Convert dict for YAML serialization, handling callables nicely."""
        result = {}
        for key, value in d.items():
            if isinstance(value, dict):
                # Check if it's a dict of callables (like eval_metrics)
                if value and all(callable(v) for v in value.values()):
                    # Show just the keys as a list
                    result[key] = list(value.keys())
                else:
                    result[key] = self._to_yaml_dict(value)
            elif callable(value):
                # Single callable - show its name
                result[key] = f"{value.__name__}()"
            else:
                result[key] = value
        return result


@dataclass(repr=False)
class ModelConfig(PrettyReprMixin):
    """Model architecture configuration.

    Attributes:
        d_model: Model embedding dimension.
        n_heads: Number of attention heads.
        n_layers: Number of transformer layers.
        d_ff: Feed-forward hidden dimension.
        dropout: Dropout probability.
        max_depth: Maximum nesting depth for KVPE position encoding.
        max_array_position: Maximum array index for position embeddings.
        kvpe_pooling: Pooling strategy for KVPE.
        kvpe_pooling_kwargs: Additional kwargs for pooling strategy.
        backbone: Type of sequence modeling backbone.
        lstm_num_layers: Number of LSTM layers.
        num_mixture_components: Number of mixture components for continuous head.
    """

    # Architecture
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 512
    dropout: float = 0.0

    # Position encoding
    max_depth: int = 32
    max_array_position: int = 256
    kvpe_pooling: Literal["sum", "weighted", "rotary", "gru", "transformer"] = "sum"
    kvpe_pooling_kwargs: dict[str, Any] = field(default_factory=dict)

    # Backbone
    backbone: Literal["transformer", "lstm", "mamba"] = "transformer"
    lstm_num_layers: int = 2

    # Continuous head
    use_continuous_head: bool = False
    num_mixture_components: int = 5
    continuous_loss_weight: float = -1.0  # -1 = auto-calculate from NUM token proportion

    # Sequence limits
    max_seq_length: int = 2048

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )
        if self.n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {self.n_layers}")
        if self.dropout < 0 or self.dropout > 1:
            raise ValueError(f"dropout must be in [0, 1], got {self.dropout}")
        valid_poolings = {"sum", "weighted", "rotary", "gru", "transformer"}
        if self.kvpe_pooling not in valid_poolings:
            raise ValueError(
                f"kvpe_pooling must be one of {valid_poolings}, got '{self.kvpe_pooling}'"
            )
        valid_backbones = {"transformer", "lstm", "mamba", "cached_transformer"}
        if self.backbone not in valid_backbones:
            raise ValueError(f"backbone must be one of {valid_backbones}, got '{self.backbone}'")


@dataclass(repr=False)
class TrainingConfig(PrettyReprMixin):
    """Training hyperparameters configuration.

    Attributes:
        batch_size: Training batch size.
        learning_rate: Learning rate for optimizer.
        num_epochs: Number of training epochs.
        warmup_steps: Number of warmup steps for LR scheduler.
        weight_decay: Weight decay for optimizer.
        use_accelerate: Whether to use Hugging Face Accelerate for multi-GPU
            training when installed. Set to False to disable even if available.
        dataloader_num_workers: Number of worker processes for DataLoader.
            Set > 0 to enable parallel data loading and grammar mask computation.
            With grammar constraints enabled, this is critical for performance as
            grammar masks are computed in parallel by workers while GPU trains.
        shuffle_keys: Whether to shuffle key order during tokenization.
        eval_strategy: When to evaluate - "no", "steps", or "epoch".
        eval_steps: Evaluate every N steps (when eval_strategy="steps").
        eval_epochs: Evaluate every N epochs (when eval_strategy="epoch").
        eval_metrics: Dict mapping prefixes to metric names or functions.
            Example: {"acc": "accuracy", "f1": "array_f1"}
        eval_sample_size: If set, sample this many examples for evaluation.
        eval_on_train: Whether to also evaluate on training data.
        target_key: Key to predict for prediction-based metrics.
        target_loss_weight: Relative weight for target value token loss vs other tokens.
            Default 1.0 (uniform weighting). Higher values (e.g., 10.0) make the model
            focus more on predicting the target correctly. Weights are normalized so
            total loss magnitude (and effective learning rate) stays the same.
        allow_complex_values: Whether to allow complex values (objects/arrays) during
            evaluation predictions. If None (default), auto-detected based on metrics.
            Set True for array_f1, array_jaccard, object_key_accuracy metrics.
        constrain_grammar: If True (default), apply grammar constraints during training
            to ensure valid JSON syntax. The grammar PDA is created and attached to
            the model, and is also used during inference.
        constrain_schema: If True, apply schema constraints during training
            (intersected with grammar mask). If False (default), schema constraints
            are only applied during inference. Training with schema masks can
            reduce effective vocabulary per position, making loss artificially
            low without improving generation quality.
        best_metric: Metric key for triggering on_best callback. Must be a key from
            eval_metrics (e.g., "acc" if eval_metrics={"acc": "accuracy"}) or "loss"
            which is always computed. Default "loss" preserves backwards compatibility.
        best_metric_direction: Optimization direction for best_metric. "maximize" means
            higher is better (e.g., accuracy), "minimize" means lower is better (e.g., loss).
            If None (default), auto-detected from METRIC_DIRECTION registry. Required for
            custom metrics not in the registry.
    """

    # Optimization
    batch_size: int = 32
    learning_rate: float = 1e-3
    num_epochs: int = 10
    warmup_steps: int = 1000
    weight_decay: float = 0.01

    # Multi-GPU training
    use_accelerate: bool = True  # Use accelerate for distributed training if installed
    mixed_precision: Literal["no", "fp16", "bf16"] = "no"  # Mixed precision mode for accelerate

    # Data loading
    dataloader_num_workers: int = 0  # Number of DataLoader workers (0 = main process only)

    # Data augmentation
    shuffle_keys: bool = True

    # Evaluation scheduling
    eval_strategy: Literal["no", "steps", "epoch"] = "epoch"
    eval_steps: int = 100
    eval_epochs: int = 1

    # Evaluation options
    eval_metrics: dict[str, str | Callable[[list[Any], list[Any]], float]] | None = None
    eval_sample_size: int | None = None
    eval_on_train: bool = False
    target_key: str | None = None
    target_loss_weight: float = 1.0
    allow_complex_values: bool | None = None

    # Constraints
    constrain_grammar: bool = True
    constrain_schema: bool = False

    # Best model selection
    best_metric: str = "loss"  # Metric key for on_best callback (e.g., "loss", "acc")
    best_metric_direction: Literal["maximize", "minimize"] | None = None  # None = auto-detect

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate}")
        if self.num_epochs < 1:
            raise ValueError(f"num_epochs must be >= 1, got {self.num_epochs}")

        valid_eval_strategies = {"no", "steps", "epoch"}
        if self.eval_strategy not in valid_eval_strategies:
            raise ValueError(
                f"eval_strategy must be one of {valid_eval_strategies}, got '{self.eval_strategy}'"
            )

        # Validate best_metric_direction
        valid_directions = {None, "maximize", "minimize"}
        if self.best_metric_direction not in valid_directions:
            raise ValueError(
                f"best_metric_direction must be None, 'maximize', or 'minimize', "
                f"got '{self.best_metric_direction}'"
            )


@dataclass(repr=False)
class DataConfig(PrettyReprMixin):
    """Data preprocessing configuration.

    Attributes:
        numeric_mode: How to handle high-cardinality numeric fields.
            - "none": Pass through as categorical tokens
            - "discretize": Bin into discrete categories
            - "scale": Use continuous head with Mixture of Gaussians
        cat_threshold: Fields with more unique values than this are processed.
        n_bins: Number of bins for discretization (numeric_mode="discretize").
        bin_strategy: Binning strategy for discretization.
        max_vocab_size: Maximum vocabulary size. 0 = unlimited.
            If > 0, rare tokens are replaced with UNK after tokenizer.fit().
        schema: Optional JSON Schema dict to constrain model outputs.
            Applied as semantic constraints on top of the syntactic grammar PDA.
        infer_schema: If True, auto-infer a JSON Schema from training data.
            When both schema and infer_schema are set, the user-provided
            schema takes precedence.
    """

    numeric_mode: Literal["disabled", "discretize", "scale"] = "disabled"
    cat_threshold: int = 100
    n_bins: int = 20
    bin_strategy: Literal["quantile", "uniform", "kmeans"] = "quantile"
    max_vocab_size: int = 0  # 0 = unlimited
    schema: dict | None = None
    infer_schema: bool = False

    def __post_init__(self) -> None:
        """Validate configuration."""
        valid_numeric_modes = {"disabled", "discretize", "scale"}
        if self.numeric_mode not in valid_numeric_modes:
            raise ValueError(
                f"numeric_mode must be one of {valid_numeric_modes}, got '{self.numeric_mode}'"
            )
        valid_bin_strategies = {"quantile", "uniform", "kmeans"}
        if self.bin_strategy not in valid_bin_strategies:
            raise ValueError(
                f"bin_strategy must be one of {valid_bin_strategies}, got '{self.bin_strategy}'"
            )
        if self.cat_threshold < 1:
            raise ValueError(f"cat_threshold must be >= 1, got {self.cat_threshold}")
        if self.numeric_mode == "discretize" and self.n_bins < 2:
            raise ValueError(f"n_bins must be >= 2, got {self.n_bins}")
        if self.max_vocab_size < 0:
            raise ValueError(f"max_vocab_size must be >= 0, got {self.max_vocab_size}")


@dataclass(repr=False)
class OrigamiConfig(PrettyReprMixin):
    """Root configuration composing all sub-configs.

    This is the main configuration class that composes model, training, and data
    configurations into a single unified config. It provides a clean interface
    for configuring all aspects of the Origami pipeline.

    Attributes:
        model: Model architecture configuration.
        training: Training hyperparameters.
        data: Data preprocessing configuration.
        device: Device for training/inference ("auto", "cpu", "cuda", "mps").

    Example:
        ```python
        # All defaults
        config = OrigamiConfig()

        # Customize model architecture
        config = OrigamiConfig(model=ModelConfig(d_model=256, n_layers=8))

        # Full customization
        config = OrigamiConfig(
            model=ModelConfig(d_model=256),
            training=TrainingConfig(batch_size=64),
            data=DataConfig(numeric_mode="scale"),
            device="cuda",
        )
        ```
    """

    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    device: str = "auto"

    def __post_init__(self) -> None:
        """Validate configuration."""
        valid_devices = {"auto", "cpu", "cuda", "mps"}
        # Also allow specific CUDA devices like "cuda:0"
        if self.device not in valid_devices and not self.device.startswith("cuda:"):
            raise ValueError(
                f"device must be one of {valid_devices} or 'cuda:N', got {self.device}"
            )
