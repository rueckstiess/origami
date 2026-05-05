# CLI Reference

The Origami command-line interface is installed automatically with the package. The entry point is `origami` with five subcommands.

```
origami <command> [options]

Commands:
  train       Train a model
  predict     Predict missing values
  generate    Generate synthetic data
  evaluate    Evaluate model performance
  embed       Create vector embeddings
```

All commands support `-v` / `--verbose` to display model configuration.

## Data Formats

All commands that accept data (`-d` / `--data`) auto-detect the format:

| Format | Extension | Description |
|--------|-----------|-------------|
| JSONL | `.jsonl` | One JSON object per line (recommended) |
| JSON | `.json` | Array of JSON objects |
| CSV | `.csv` | Auto-converts numeric strings, booleans, and nulls |
| MongoDB | `mongodb://` | Direct connection via URI (requires `--db` and `-c`) |

All data commands also support:
- `--skip N` — skip the first N records
- `--limit N` — use at most N records (0 = unlimited)
- `--project '{"field": 1}'` — MongoDB-style field projection (include or exclude fields)

## `origami train`

Train an Origami model on data.

### Basic Options

| Flag | Default | Description |
|------|---------|-------------|
| `-d, --data` | required | Training data path or MongoDB URI |
| `-t, --target-key` | — | Target field for prediction metrics during training |
| `-e, --epochs` | `10` | Number of training epochs |
| `-o, --output` | required | Output path for trained model (e.g., `model.pt`) |
| `--seed` | `42` | Random seed |

### Model Architecture

| Flag | Default | Description |
|------|---------|-------------|
| `-D, --d-model` | `128` | Hidden dimension |
| `-H, --n-heads` | `4` | Number of attention heads |
| `-L, --n-layers` | `4` | Number of transformer layers |

### Training

| Flag | Default | Description |
|------|---------|-------------|
| `-b, --batch-size` | `32` | Batch size |
| `-l, --lr` | `1e-3` | Learning rate |

### Data Preprocessing

| Flag | Default | Description |
|------|---------|-------------|
| `-n, --numeric-mode` | `disabled` | `disabled`, `discretize`, or `scale` |

### Validation

| Flag | Default | Description |
|------|---------|-------------|
| `--val` | — | Separate validation data file |
| `--val-collection` | — | MongoDB validation collection (uses same `--db`) |
| `--train-ratio` | — | Split training data (e.g., `0.8` for 80/20 split) |
| `--eval-sample-size` | — | Subsample N examples for faster evaluation |

### Advanced: `--set`

Set any configuration parameter directly. Useful for parameters not exposed as dedicated flags.

```bash
origami train -d data.jsonl -t label -o model.pt \
  --set d_ff=1024 \
  --set dropout=0.1 \
  --set warmup_steps=500 \
  --set kvpe_pooling=weighted
```

Parameters are auto-mapped to the correct config section. You can also use dotted paths:

```bash
--set model.n_heads=8
--set training.weight_decay=0.01
--set data.max_vocab_size=5000
```

See [Configuration Reference](configuration.md) for all available parameters.

### Examples

```bash
# Basic training
origami train -d data.jsonl -t label -e 20 -o model.pt

# With 80/20 validation split
origami train -d data.jsonl -t label --train-ratio 0.8 -o model.pt

# Separate validation file
origami train -d train.jsonl --val val.jsonl -t label -o model.pt

# Larger model with continuous numerics
origami train -d data.jsonl -t price -D 256 -L 6 -n scale -o model.pt

# From MongoDB
origami train -d mongodb://localhost:27017 --db mydb -c train -t label -o model.pt
```

## `origami predict`

Predict target values for input data.

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `-m, --model` | required | Path to trained model (`.pt` file) |
| `-d, --data` | required | Input data |
| `-t, --target-key` | required | Field to predict |
| `-o, --output` | stdout | Output file |
| `-f, --format` | `values` | Output format (see below) |
| `-b, --batch-size` | `32` | Batch size |

### Output Formats

| Format | Description |
|--------|-------------|
| `values` | One predicted value per line (default, good for piping) |
| `json` | JSON array of all predictions |
| `jsonl` | Original objects with predicted values filled in |

### Examples

```bash
# Predictions to stdout
origami predict -m model.pt -d test.jsonl -t label

# Save as JSONL with original data
origami predict -m model.pt -d test.jsonl -t label -f jsonl -o predictions.jsonl

# Save as JSON array
origami predict -m model.pt -d test.jsonl -t label -f json -o predictions.json
```

## `origami generate`

Generate synthetic JSON objects from a trained model.

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `-m, --model` | required | Path to trained model (`.pt` file) |
| `-n, --count` | `10` | Number of samples to generate |
| `--temp` | `1.0` | Sampling temperature (higher = more random) |
| `--top-k` | — | Top-k sampling (keep top k tokens) |
| `--top-p` | — | Nucleus sampling (cumulative probability threshold) |
| `--seed` | — | Random seed for reproducibility |
| `-o, --output` | stdout | Output file (JSONL format) |
| `-b, --batch-size` | `32` | Batch size for generation |
| `--max-length` | `512` | Maximum sequence length |

### Examples

```bash
# Generate 10 samples to stdout
origami generate -m model.pt

# Generate with lower temperature (more predictable)
origami generate -m model.pt -n 100 --temp 0.7

# Generate with top-k sampling
origami generate -m model.pt -n 50 --top-k 50

# Reproducible generation to file
origami generate -m model.pt -n 1000 --seed 42 -o samples.jsonl
```

## `origami evaluate`

Evaluate a model on test data with loss and prediction metrics.

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `-m, --model` | required | Path to trained model (`.pt` file) |
| `-d, --data` | required | Test data |
| `-t, --target-key` | required | Field to evaluate predictions on |
| `--metrics` | `accuracy` | Metrics to compute (can be repeated) |
| `--sample-size` | — | Evaluate on a random subset of N examples |

### Available Metrics

| Name | Type | Description |
|------|------|-------------|
| `accuracy` | Classification | Exact match fraction |
| `array_f1` | Classification | Set-based F1 for array values |
| `array_precision` | Classification | Set-based precision for array values |
| `array_recall` | Classification | Set-based recall for array values |
| `array_jaccard` | Classification | Jaccard similarity for array values |
| `object_key_accuracy` | Classification | Per-key accuracy for object values |
| `mse` | Regression | Mean squared error |
| `mae` | Regression | Mean absolute error |
| `rmse` | Regression | Root mean squared error |

### Examples

```bash
# Evaluate accuracy (default)
origami evaluate -m model.pt -d test.jsonl -t label

# Multiple metrics
origami evaluate -m model.pt -d test.jsonl -t label \
  --metrics accuracy --metrics array_f1

# Regression metric on subset
origami evaluate -m model.pt -d test.jsonl -t price \
  --metrics rmse --sample-size 500
```

## `origami embed`

Create dense vector embeddings from JSON documents.

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `-m, --model` | required | Path to trained model (`.pt` file) |
| `-d, --data` | required | Input data |
| `-o, --output` | required | Output file (format from extension) |
| `-p, --pooling` | `mean` | Pooling strategy: `mean`, `max`, `last`, `target` |
| `-t, --target-key` | — | Target key (required for `pooling=target`) |
| `--no-normalize` | — | Disable L2 normalization |

### Output Formats

The output format is determined by the file extension:

| Extension | Format |
|-----------|--------|
| `.npy` | NumPy array |
| `.csv` | CSV file |
| `.pt` | PyTorch tensor |

### Examples

```bash
# Embeddings as NumPy array
origami embed -m model.pt -d data.jsonl -o embeddings.npy

# Embeddings as CSV
origami embed -m model.pt -d data.jsonl -o embeddings.csv

# Target-specific embeddings
origami embed -m model.pt -d data.jsonl -o emb.npy -p target -t label

# Without normalization
origami embed -m model.pt -d data.jsonl -o emb.npy --no-normalize
```
