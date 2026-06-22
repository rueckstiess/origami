# Origami

**A machine learning model for JSON data.**

[![PyPI](https://img.shields.io/pypi/v/origami-ml)](https://pypi.org/project/origami-ml/)
[![Python](https://img.shields.io/pypi/pyversions/origami-ml)](https://pypi.org/project/origami-ml/)
[![License](https://img.shields.io/pypi/l/origami-ml)](https://github.com/rueckstiess/origami/blob/main/LICENSE)

Origami trains models that learn the relationships between fields in JSON objects. Given a dataset of JSON records, an Origami model can:

- **Predict** missing field values based on the other fields
- **Generate** new synthetic JSON objects that follow the patterns in your data
- **Embed** JSON objects as dense vectors for similarity search or downstream tasks

Unlike tabular ML models that require flat feature vectors, Origami works directly with JSON structure — including nested objects and arrays.

## Installation

```bash
pip install origami-ml
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv add origami-ml
```

Origami v2 is a breaking rewrite of the original `origami-ml` package. See
[Migrating from v1 to v2](documentation/migration-v2.md) if you are upgrading from v1.

Requires Python 3.11+. PyTorch is installed automatically. GPU acceleration (CUDA, Apple Silicon MPS) is auto-detected — no configuration needed.

## Quick Start

```python
from origami import OrigamiPipeline

# Your data: a list of JSON objects (Python dicts)
data = [
    {"product": "Wireless Headphones", "categories": ["audio", "wireless"], "price": 79.99,  "rating": 4.2},
    {"product": "USB-C Hub",           "categories": ["accessories"],       "price": 34.99,  "rating": 4.5},
    {"product": "Mechanical Keyboard", "categories": ["peripherals"],       "price": 129.99, "rating": 4.7},
    # ... more records
]

# Train with default settings
pipeline = OrigamiPipeline()
pipeline.fit(data, epochs=20)

# Predict a missing value (including arrays)
prediction = pipeline.predict(
    {"product": "Bluetooth Speaker", "categories": None},
    target_key="categories",
    allow_complex_values=True,
)
print(prediction)  # ["audio", "wireless"]

# Generate new synthetic records
samples = pipeline.generate(num_samples=5, temperature=0.8)

# Get a vector embedding
embedding = pipeline.embed({"product": "Wireless Headphones", "categories": ["audio", "wireless"]})
# numpy array of shape (128,)

# Save and load
pipeline.save("model.pt")
loaded = OrigamiPipeline.load("model.pt")
```

## Configuration

For more control, pass an `OrigamiConfig` with nested configuration objects:

```python
from origami import OrigamiPipeline, OrigamiConfig, ModelConfig, TrainingConfig, DataConfig

config = OrigamiConfig(
    model=ModelConfig(
        d_model=256,       # Larger hidden dimension (default: 128)
        n_layers=6,        # More transformer layers (default: 4)
    ),
    training=TrainingConfig(
        batch_size=64,
        num_epochs=50,
        target_key="price",                  # Track prediction metrics during training
        eval_metrics={"acc": "accuracy"},     # Compute accuracy each epoch
    ),
    data=DataConfig(
        numeric_mode="scale",  # Handle numeric fields as continuous values
    ),
)

pipeline = OrigamiPipeline(config)
pipeline.fit(train_data, eval_data=val_data)
```

- **ModelConfig** controls the model architecture (size, depth, position encoding strategy)
- **TrainingConfig** controls training hyperparameters (learning rate, batch size, evaluation)
- **DataConfig** controls data preprocessing (how numeric fields are handled, vocabulary size)
- **InferenceConfig** controls inference-time constraints (grammar and schema enforcement)

See [Configuration Reference](documentation/configuration.md) for all options.

## Command-Line Interface

Origami includes a CLI for training, prediction, generation, evaluation, and embedding:

```bash
# Train a model
origami train -d data.jsonl -t label -e 20 -o model.pt

# Predict missing values
origami predict -m model.pt -d test.jsonl -t label

# Generate synthetic data
origami generate -m model.pt -n 100 --temp 0.8

# Evaluate model performance (loss + accuracy by default)
origami evaluate -m model.pt -d test.jsonl -t label
```

See [CLI Reference](documentation/cli.md) for all commands and options.

## Documentation

- **[Concepts](documentation/concepts.md)** — How Origami works: tokenization, position encoding, grammar constraints
- **[Python SDK](documentation/python-sdk.md)** — Complete API reference for `OrigamiPipeline`
- **[CLI Reference](documentation/cli.md)** — All commands, options, and supported data formats
- **[Configuration](documentation/configuration.md)** — Every configuration parameter explained
- **[Migrating from v1 to v2](documentation/migration-v2.md)** — Breaking changes and migration guidance

## How It Works

Origami converts each JSON object into a sequence of tokens that preserve the hierarchical structure — keys, values, nesting, and arrays are all explicitly represented. Instead of encoding token position as a simple index (1st, 2nd, 3rd...), Origami uses **Key-Value Position Encoding (KVPE)**, which encodes the _path_ through the JSON tree. This lets the model understand which key each value belongs to, regardless of key order.

A **grammar constraint** system (a pushdown automaton) ensures that every model output is valid JSON — no syntax errors, ever. This is applied automatically with no configuration needed.

For numeric fields with many distinct values (like prices or measurements), Origami can model them as continuous distributions rather than discrete tokens, using a **mixture density output head**.

For a deeper explanation of these concepts, see the [Concepts](documentation/concepts.md) page.

## License

Apache-2.0
