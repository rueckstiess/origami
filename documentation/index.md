# Origami Documentation

Origami is a machine learning model for JSON data. It learns the relationships between fields in a dataset of JSON objects, then uses that knowledge to predict missing values, generate new synthetic objects, and produce dense vector embeddings.

This documentation covers the Python API, command-line interface, and all configuration options.

## Pages

| Page | Description |
|------|-------------|
| [Concepts](concepts.md) | How Origami works — tokenization, position encoding, grammar constraints, numeric handling |
| [Python SDK](python-sdk.md) | Complete API reference for `OrigamiPipeline` — training, prediction, generation, embedding, evaluation |
| [CLI Reference](cli.md) | Command-line tools for training, prediction, generation, embedding, and evaluation |
| [Configuration](configuration.md) | All configuration options — model architecture, training, data preprocessing, inference |

## Quick Links

| I want to... | Python SDK | CLI |
|--------------|------------|-----|
| Train a model | [Training](python-sdk.md#training) | [`origami train`](cli.md#origami-train) |
| Predict missing values | [Prediction](python-sdk.md#prediction) | [`origami predict`](cli.md#origami-predict) |
| Generate synthetic data | [Generation](python-sdk.md#generation) | [`origami generate`](cli.md#origami-generate) |
| Create embeddings | [Embedding](python-sdk.md#embedding) | [`origami embed`](cli.md#origami-embed) |
| Evaluate a model | [Evaluation](python-sdk.md#evaluation) | [`origami evaluate`](cli.md#origami-evaluate) |
| Handle numeric fields | [Concepts: Numeric Fields](concepts.md#handling-numeric-fields) | [Configuration: DataConfig](configuration.md#dataconfig) |
| Tune model size | [Configuration: ModelConfig](configuration.md#modelconfig) | [`origami train -D ... -L ...`](cli.md#origami-train) |
| Understand the architecture | [Concepts](concepts.md) | — |
