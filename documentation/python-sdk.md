# Python SDK

The `OrigamiPipeline` class is the recommended API for training and inference. It handles preprocessing, tokenization, model creation, training, and all inference modes in a single unified interface.

All imports come from the top-level `origami` package:

```python
from origami import OrigamiPipeline, OrigamiConfig, ModelConfig, TrainingConfig, DataConfig
```

## Training

### `fit(data, eval_data=None, epochs=None, verbose=False, callbacks=None)`

Train a model end-to-end. This is the simplest way to get started.

```python
from origami import OrigamiPipeline

pipeline = OrigamiPipeline()
pipeline.fit(data, epochs=20)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | `list[dict]` | required | Training data — a list of JSON objects (Python dicts) |
| `eval_data` | `list[dict] \| None` | `None` | Validation data for evaluation during training |
| `epochs` | `int \| None` | `None` | Number of epochs (overrides `TrainingConfig.num_epochs`) |
| `verbose` | `bool` | `False` | Print model info (vocab size, parameter count, device) |
| `callbacks` | `list \| None` | `None` | Training callbacks. Defaults to `[ProgressCallback()]` |

**Data format:** Each dict is a JSON object. All objects should share the same keys. Values can be strings, numbers, booleans, `None`, lists, or nested dicts.

**Variants:**

```python
# With validation data
pipeline.fit(train_data, eval_data=val_data, epochs=50)

# With custom callbacks
from origami.training import TableLogCallback
pipeline.fit(data, callbacks=[TableLogCallback(print_every=10)])

# Silent training (no progress output)
pipeline.fit(data, callbacks=[])

# Verbose mode (prints model architecture summary)
pipeline.fit(data, verbose=True)
```

### Two-Step API

For advanced use cases (e.g., inspecting the model before training, or resuming training from a checkpoint), you can separate preprocessing and training:

```python
pipeline = OrigamiPipeline(config)
pipeline.preprocess(data, eval_data=val_data, verbose=True)

# Inspect model
print(f"Vocabulary size: {len(pipeline.tokenizer.vocab)}")
print(f"Parameters: {pipeline.model.get_num_parameters():,}")

# Train
pipeline.train(epochs=50)
```

### Callbacks

Callbacks control all training output. Three options are available:

| Callback | Description |
|----------|-------------|
| `ProgressCallback()` | tqdm progress bars with loss and learning rate (default) |
| `TableLogCallback(print_every=10)` | Single-line table format logging |
| `TrainerCallback` | Base class — subclass to create custom callbacks |

```python
from origami.training import ProgressCallback, TableLogCallback, TrainerCallback

# Custom callback
class SaveBestCallback(TrainerCallback):
    def on_best(self, trainer, state, payload):
        trainer.pipeline.save("best_model.pt")

pipeline.fit(data, callbacks=[ProgressCallback(), SaveBestCallback()])
```

Available callback events: `on_train_begin`, `on_train_end`, `on_epoch_begin`, `on_epoch_end`, `on_batch_begin`, `on_batch_end`, `on_evaluate`, `on_best`, `on_interrupt`.

### Experiment Tracking

Custom callbacks make it easy to integrate with experiment tracking systems like [Weights & Biases](https://wandb.ai), TensorBoard, or MLflow. Implement the callback events you need and call the tracking API:

```python
import wandb
from origami.training import TrainerCallback

class WandBCallback(TrainerCallback):
    def on_train_begin(self, trainer, state, payload):
        wandb.init(project="origami", config=trainer.config.to_yaml())

    def on_evaluate(self, trainer, state, payload):
        # payload is a dict like {"val_loss": 0.5, "val_accuracy": 0.85}
        wandb.log(payload, step=state.global_step)

    def on_epoch_end(self, trainer, state, payload):
        # payload is an EpochStats with training throughput info
        wandb.log({
            "train_loss": payload.loss,
            "tokens_per_second": payload.tokens_per_second,
            "lr": state.current_lr,
        }, step=state.global_step)

    def on_train_end(self, trainer, state, payload):
        wandb.finish()

pipeline.fit(data, eval_data=val_data, callbacks=[WandBCallback()])
```

## Prediction

### `predict(obj, target_key) -> Any`

Predict the value of a single field.

```python
value = pipeline.predict(
    {"city": "Tokyo", "country": None},
    target_key="country",
)
# "Japan"
```

The `target_key`'s current value in the object is ignored — the model always predicts from the other fields.

### `predict_batch(objects, target_key, batch_size=32) -> list[Any]`

Predict values for multiple objects.

```python
objects = [
    {"city": "Tokyo", "country": None},
    {"city": "Sydney", "country": None},
]
values = pipeline.predict_batch(objects, target_key="country")
# ["Japan", "Australia"]
```

### `predict_proba(obj, target_key, top_k=None) -> dict | list[tuple]`

Get the probability distribution over possible values.

```python
# Full distribution (dict)
probs = pipeline.predict_proba(obj, target_key="country")
# {"Japan": 0.85, "China": 0.04, "South Korea": 0.03, ...}

# Top-k (sorted list of tuples)
top5 = pipeline.predict_proba(obj, target_key="country", top_k=5)
# [("Japan", 0.85), ("China", 0.04), ("South Korea", 0.03), ...]
```

**Additional parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `values` | `list \| None` | `None` | Restrict to specific values |
| `top_k` | `int \| None` | `None` | Return only top-k values (returns list of tuples instead of dict) |

### Complex Values

By default, predictions are restricted to primitive values (strings, numbers, booleans, null). To allow arrays and nested objects as predicted values, pass `allow_complex_values=True`:

```python
# Predict an array-valued field
tags = pipeline.predict(
    {"name": "Alice", "tags": None},
    target_key="tags",
    allow_complex_values=True,
)
# ["admin", "active"]
```

## Generation

### `generate(num_samples=1, temperature=1.0, ...) -> list[dict]`

Generate complete JSON objects from scratch.

```python
samples = pipeline.generate(num_samples=10)
```

**Sampling parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_samples` | `int` | `1` | Number of objects to generate |
| `batch_size` | `int` | `32` | Objects generated in parallel |
| `max_length` | `int` | `512` | Maximum token sequence length |
| `temperature` | `float` | `1.0` | Randomness. Lower = more predictable, higher = more varied |
| `top_k` | `int \| None` | `None` | Only consider the k most likely tokens at each step |
| `top_p` | `float \| None` | `None` | Only consider tokens with cumulative probability >= p |
| `seed` | `int \| None` | `None` | Random seed for reproducible generation |

**Temperature** controls how random the output is:
- `temperature=0.0` — always pick the most likely token (deterministic)
- `temperature=1.0` — sample according to the model's probabilities (default)
- `temperature>1.0` — increase randomness (more creative, less accurate)

```python
# Deterministic generation
samples = pipeline.generate(num_samples=5, temperature=0.0)

# More varied output
samples = pipeline.generate(num_samples=5, temperature=1.2, top_k=50)

# Reproducible
samples = pipeline.generate(num_samples=5, seed=42)
```

## Embedding

### `embed(obj, pooling="mean", normalize=True) -> np.ndarray`

Get a dense vector embedding for a JSON object.

```python
vec = pipeline.embed({"city": "London", "country": "UK"})
# numpy array of shape (d_model,), e.g., (128,)
```

### `embed_batch(objects, pooling="mean", normalize=True) -> np.ndarray`

Get embeddings for multiple objects.

```python
vecs = pipeline.embed_batch(objects)
# numpy array of shape (n, d_model)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pooling` | `str` | `"mean"` | How to aggregate token-level embeddings (see below) |
| `target_key` | `str \| None` | `None` | Required when `pooling="target"` |
| `normalize` | `bool` | `True` | L2-normalize embeddings (unit length) |
| `enable_grad` | `bool` | `False` | Return `torch.Tensor` with gradients instead of numpy array |

**Pooling strategies:**

| Strategy | Description |
|----------|-------------|
| `"mean"` | Average all token embeddings (default, good general-purpose) |
| `"max"` | Max-pool over token embeddings |
| `"last"` | Last token's embedding |
| `"target"` | Embedding at the position of a specific key (requires `target_key`) |

```python
# Use target-specific embeddings for classification
vec = pipeline.embed(obj, pooling="target", target_key="label")
```

## Evaluation

### `evaluate(data, target_key=None, metrics=None) -> dict[str, float]`

Evaluate model performance. Loss is opt-in: request it with the reserved metric
spec `"loss"`, just like any other metric. When no `metrics` are passed, loss is
computed by default.

```python
# Loss only (default when no metrics are passed)
results = pipeline.evaluate(test_data)
print(f"Loss: {results['loss']:.4f}")

# Prediction metrics only — loss is NOT computed (faster)
results = pipeline.evaluate(
    test_data,
    target_key="label",
    metrics={"acc": "accuracy"},
)
print(f"Accuracy: {results['acc']:.2%}")

# Opt into both loss and prediction metrics
results = pipeline.evaluate(
    test_data,
    target_key="label",
    metrics={"loss": "loss", "acc": "accuracy", "f1": "array_f1"},
)
print(f"Loss: {results['loss']:.4f}, Accuracy: {results['acc']:.2%}")
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | `list[dict]` | required | Test data |
| `target_key` | `str \| None` | `None` | Key to predict (required if prediction metrics are provided; not needed for `"loss"` alone) |
| `metrics` | `dict \| None` | `None` | Dict mapping names to metric strings. E.g., `{"acc": "accuracy"}`. Use `"loss"` to compute model loss. `None` defaults to loss only. |
| `sample_size` | `int \| None` | `None` | Evaluate on a random subset (faster) |
| `batch_size` | `int` | `32` | Batch size for evaluation |
| `verbose` | `bool` | `False` | Show progress bars |

### Available Metrics

| Name | Type | Description |
|------|------|-------------|
| `loss` | Model | Model loss (reserved spec; computed from model outputs, not predictions) |
| `accuracy` | Classification | Exact match fraction |
| `array_f1` | Classification | Set-based F1 score for array values |
| `array_precision` | Classification | Set-based precision for array values |
| `array_recall` | Classification | Set-based recall for array values |
| `array_jaccard` | Classification | Jaccard similarity for array values |
| `object_key_accuracy` | Classification | Per-key accuracy for object values |
| `mse` | Regression | Mean squared error |
| `mae` | Regression | Mean absolute error |
| `rmse` | Regression | Root mean squared error |

Metrics are specified as strings in the `metrics` dict. The dict keys become the result keys:

```python
results = pipeline.evaluate(data, target_key="price", metrics={"err": "rmse"})
print(results["err"])  # RMSE value
```

## Save and Load

### `save(path)`

Save the complete pipeline to a single file.

```python
pipeline.save("model.pt")
```

Everything needed to use the model is saved: model weights, tokenizer vocabulary, preprocessor state, configuration, and schemas. Training state (optimizer, scheduler, current epoch) is included by default, enabling checkpoint resumption.

### `load(path) -> OrigamiPipeline`

Load a saved pipeline.

```python
pipeline = OrigamiPipeline.load("model.pt")
```

### Resume Training

Since training state is preserved in checkpoints, you can resume training from where you left off:

```python
pipeline = OrigamiPipeline.load("checkpoint.pt")
pipeline.fit(data, eval_data=val_data, epochs=100)  # Continues from saved epoch
```

### Inference-Only Checkpoints

To save a smaller file without training state:

```python
pipeline.save("model_inference.pt", include_training_state=False)
```

## Device Management

By default, Origami auto-detects the best available device:
- **Training** uses GPU (CUDA) or Apple Silicon (MPS) when available, falling back to CPU
- **Inference** defaults to CPU, which is typically faster for autoregressive generation

You can override the inference device on any method:

```python
value = pipeline.predict(obj, target_key="b", device="mps")
vecs = pipeline.embed_batch(objects, device="cuda")
results = pipeline.evaluate(data, device="cuda")
```

To set the training device, use `OrigamiConfig`:

```python
config = OrigamiConfig(device="cuda:1")  # Specific GPU
```

## Public Exports

Everything available from `from origami import ...`:

```python
from origami import (
    # Pipeline (recommended API)
    OrigamiPipeline,

    # Configuration
    OrigamiConfig,
    ModelConfig,
    TrainingConfig,
    DataConfig,
    InferenceConfig,

    # Low-level components (advanced usage)
    OrigamiModel,       # Core transformer model
    OrigamiOutput,      # Model forward pass output
    JSONTokenizer,      # JSON tokenizer
    EncodedBatch,       # Batch of tokenized sequences
    Vocabulary,         # Token vocabulary
)
```
