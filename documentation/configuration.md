# Configuration Reference

Origami uses a nested configuration structure. The root `OrigamiConfig` composes four sub-configs:

```python
from origami import OrigamiConfig, ModelConfig, TrainingConfig, DataConfig, InferenceConfig

config = OrigamiConfig(
    model=ModelConfig(...),
    training=TrainingConfig(...),
    data=DataConfig(...),
    inference=InferenceConfig(...),
    device="auto",
)
```

All parameters have sensible defaults. You only need to specify what you want to change:

```python
# All defaults
config = OrigamiConfig()

# Only change model size
config = OrigamiConfig(model=ModelConfig(d_model=256, n_layers=6))
```

## OrigamiConfig

Top-level configuration.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `ModelConfig` | `ModelConfig()` | Model architecture settings |
| `training` | `TrainingConfig` | `TrainingConfig()` | Training hyperparameters |
| `data` | `DataConfig` | `DataConfig()` | Data preprocessing settings |
| `inference` | `InferenceConfig` | `InferenceConfig()` | Inference constraint settings |
| `device` | `str` | `"auto"` | Training device: `"auto"`, `"cpu"`, `"cuda"`, `"cuda:N"`, or `"mps"` |

`"auto"` selects CUDA if available, then Apple Silicon MPS, then CPU.

## ModelConfig

Model architecture parameters.

### Core Architecture

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `d_model` | `int` | `128` | Hidden dimension size. Larger values increase model capacity but require more memory. Must be divisible by `n_heads`. |
| `n_heads` | `int` | `4` | Number of attention heads. Must divide `d_model` evenly. |
| `n_layers` | `int` | `4` | Number of transformer layers. More layers enable learning deeper patterns. |
| `d_ff` | `int` | `512` | Feed-forward intermediate dimension. Typically 4x `d_model`. |
| `dropout` | `float` | `0.0` | Dropout probability (0.0 to 1.0). Helps prevent overfitting on small datasets. |
| `max_seq_length` | `int` | `2048` | Maximum token sequence length. |

### Position Encoding

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `position_encoding` | `str` | `"kvpe"` | `"kvpe"` (JSON-structure-aware) or `"sequential"` (standard positional encoding). |
| `max_depth` | `int` | `32` | Maximum JSON nesting depth. |
| `max_array_position` | `int` | `256` | Maximum array index for position embeddings. |
| `kvpe_pooling` | `str` | `"sum"` | How path element embeddings are combined. See below. |
| `kvpe_pooling_kwargs` | `dict` | `{}` | Additional keyword arguments for the pooling strategy. |

**KVPE pooling strategies:**

| Strategy | Description |
|----------|-------------|
| `"sum"` | Add all path embeddings together. Simple, effective default. Order-independent. |
| `"weighted"` | Learned weights per depth level. Useful when depth hierarchy matters. |
| `"rotary"` | Rotary position encoding applied per depth. Position-sensitive at each level. |
| `"gru"` | Process path elements sequentially with a GRU. Fully order-dependent — `a.b` differs from `b.a`. |
| `"transformer"` | Self-attention over path elements. Maximum expressiveness, higher cost. |

### Backbone

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `backbone` | `str` | `"transformer"` | Sequence model backbone: `"transformer"`, `"lstm"`, or `"mamba"`. |
| `lstm_num_layers` | `int` | `2` | Number of LSTM layers (when `backbone="lstm"`). |

### Continuous Output Head

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_continuous_head` | `bool` | `False` | Enable mixture-of-Gaussians head for continuous numerics. **Set automatically** when `DataConfig.numeric_mode="scale"` — you rarely need to set this manually. |
| `num_mixture_components` | `int` | `5` | Number of Gaussian components in the mixture. |
| `continuous_loss_weight` | `float` | `-1.0` | Loss weight for continuous head. `-1` = auto-calculate from data. |

## TrainingConfig

Training hyperparameters.

### Optimization

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `batch_size` | `int` | `32` | Number of objects per training batch. |
| `learning_rate` | `float` | `1e-3` | Initial learning rate (Adam optimizer). |
| `num_epochs` | `int` | `10` | Number of training passes over the data. |
| `warmup_steps` | `int` | `1000` | Linear warmup steps for learning rate schedule. |
| `weight_decay` | `float` | `0.01` | L2 regularization weight. |
| `lr_scheduler` | `str` | `"linear"` | Decay schedule after warmup: `"linear"` or `"cosine"`. |
| `lr_cosine_exponent` | `float` | `1.0` | Exponent for cosine decay. Values >1 spend more time at lower learning rates. Only applies when `lr_scheduler="cosine"`. |
| `lr_min` | `float` | `0.0` | Minimum learning rate floor. Prevents the schedule from decaying all the way to zero (e.g., set to `1e-6`). Applies to both linear and cosine schedules. |

### Data Augmentation

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `shuffle_keys` | `bool` | `True` | Randomly shuffle JSON key order each time an object is seen. Since JSON key order is meaningless, this forces the model to learn from content rather than position. See [Concepts: Key-Order Shuffling](concepts.md#key-order-shuffling). |

### Evaluation

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `eval_strategy` | `str` | `"epoch"` | When to evaluate: `"no"`, `"steps"`, or `"epoch"`. |
| `eval_steps` | `int` | `100` | Evaluate every N steps (when `eval_strategy="steps"`). |
| `eval_epochs` | `int` | `1` | Evaluate every N epochs (when `eval_strategy="epoch"`). |
| `eval_metrics` | `dict \| None` | `None` | Metrics to compute during evaluation. Dict mapping result key to metric name. |
| `eval_sample_size` | `int \| None` | `None` | Subsample for faster evaluation. `None` = full dataset. |
| `eval_on_train` | `bool` | `False` | Also evaluate on training data (useful for detecting overfitting). |
| `target_key` | `str \| None` | `None` | Field to predict for prediction-based metrics. Required if prediction metrics are configured; not required for `"loss"` alone. |
| `target_loss_weight` | `float` | `1.0` | Relative weight for target field loss vs. other tokens. Higher values (e.g., 10.0) make the model focus more on predicting the target correctly. |
| `allow_complex_values` | `bool \| None` | `None` | Allow arrays/objects in eval predictions. `None` = auto-detect from metrics. |

**Example: Evaluation during training**

```python
TrainingConfig(
    num_epochs=50,
    target_key="label",
    eval_metrics={"acc": "accuracy"},
    eval_strategy="epoch",
    best_metric="acc",
)
```

This evaluates accuracy at the end of each epoch, and the `on_best` callback fires when accuracy improves.

### Best Model Selection

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `best_metric` | `str` | `"loss"` | Metric to track for triggering `on_best` callback. Must be a key from `eval_metrics` or `"loss"`. |
| `best_metric_direction` | `str \| None` | `None` | `"maximize"` or `"minimize"`. Auto-detected for built-in metrics. |

### Constraints

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `constrain_grammar` | `bool` | `True` | Apply grammar constraints during training (ensures model learns valid JSON patterns). |
| `constrain_schema` | `bool` | `False` | Apply schema constraints during training. Requires a schema (via `DataConfig.schema` or `DataConfig.infer_schema`). |

### Advanced

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_accelerate` | `bool` | `True` | Use HuggingFace Accelerate for multi-GPU training when installed. |
| `mixed_precision` | `str` | `"no"` | Mixed precision: `"no"`, `"fp16"`, `"bf16"`. |
| `dataloader_num_workers` | `int` | `0` | Parallel data loading workers. Set > 0 for better throughput, especially with grammar constraints. |
| `group_by_length` | `bool` | `False` | Batch together similar-length sequences to reduce padding waste and attention compute. |
| `length_group_pool_mult` | `int` | `50` | Mega-batch size, as a multiple of `batch_size`, used when `group_by_length=True`. Higher values give tighter length grouping at the cost of less random batch ordering. |

## DataConfig

Data preprocessing settings.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `numeric_mode` | `str` | `"disabled"` | How to handle high-cardinality numeric fields. See below. |
| `cat_threshold` | `int` | `100` | Fields with more unique values than this are considered high-cardinality. |
| `n_bins` | `int` | `20` | Number of bins for discretization (when `numeric_mode="discretize"`). |
| `bin_strategy` | `str` | `"quantile"` | Binning strategy: `"quantile"`, `"uniform"`, or `"kmeans"`. |
| `max_vocab_size` | `int` | `0` | Maximum vocabulary size. `0` = unlimited. When exceeded, rare value tokens are pruned. |
| `schema` | `dict \| None` | `None` | JSON Schema dict for output constraints. |
| `infer_schema` | `bool` | `False` | Auto-infer JSON Schema from training data. |

### Numeric Modes

The most important data configuration decision is how to handle numeric fields. See [Concepts: Handling Numeric Fields](concepts.md#handling-numeric-fields) for detailed guidance.

**`"disabled"`** — All values are discrete tokens. Best for data with few unique numeric values.

```python
DataConfig(numeric_mode="disabled")  # Default
```

**`"discretize"`** — High-cardinality numerics are binned into categories. Good middle ground.

```python
DataConfig(numeric_mode="discretize", n_bins=20, bin_strategy="quantile")
```

**`"scale"`** — High-cardinality numerics are normalized and predicted with a continuous output head. Best for truly continuous values.

```python
DataConfig(numeric_mode="scale", cat_threshold=50)
```

Only fields with more than `cat_threshold` unique values are affected. Fields below the threshold are always treated as discrete tokens.

### Vocabulary Pruning

For large datasets with many rare values, `max_vocab_size` limits the vocabulary by removing the least frequent value tokens:

```python
DataConfig(max_vocab_size=10000)
```

Structural tokens (grammar, keys) are never pruned — only value tokens are affected. Pruned values are mapped to an unknown token during tokenization.

## InferenceConfig

Inference-time constraint settings.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `constrain_grammar` | `bool` | `True` | Ensure all generated output is valid JSON. |
| `constrain_schema` | `bool` | `False` | Enforce schema constraints during inference. Requires a schema. |

### Independent from Training

Inference and training constraints are configured independently. A common pattern is to train without schema constraints but apply them at inference time:

```python
config = OrigamiConfig(
    training=TrainingConfig(constrain_schema=False),
    inference=InferenceConfig(constrain_schema=True),
    data=DataConfig(infer_schema=True),
)
```

Grammar constraints are enabled by default for both training and inference. Disabling them is rarely needed.
