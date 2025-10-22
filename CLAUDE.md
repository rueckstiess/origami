# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ORiGAMi is a transformer-based machine learning model for supervised classification from semi-structured data (MongoDB documents, JSON files). It directly operates on JSON data without requiring manual feature extraction or flattening to tabular format.

## Development Commands

### Installation and Setup
```bash
# Install in development mode
pip install -e .

# Install with dev dependencies
pip install -e ".[dev]"

# Install from requirements file
pip install -r requirements.txt
```

### Code Quality
```bash
# Run linting with ruff
ruff check .

# Format code with ruff
ruff format .

# Run tests
pytest
```

### CLI Usage
```bash
# Basic CLI help
origami --help

# Train a model
origami train <source> [options]

# Make predictions
origami predict <source> --target-field <field> [options]
```

## Architecture Overview

### Core Components

- **`origami/cli/`**: Command-line interface with `train` and `predict` commands
- **`origami/model/`**: Core transformer model implementation
  - `origami.py`: Main ORiGAMi transformer model
  - `vpda.py`: Variable Position Discriminant Analysis (VPDA) for guardrails
  - `positions.py`: Position encoding implementations
- **`origami/preprocessing/`**: Data preprocessing pipeline
  - `encoder.py`: Tokenization and encoding of JSON data
  - `pipelines.py`: Data processing pipelines
  - `df_dataset.py`: Dataset handling for pandas DataFrames
- **`origami/inference/`**: Model inference and prediction
  - `predictor.py`: Main prediction interface
  - `embedder.py`: Embedding generation
  - `autocomplete.py`: Autocompletion functionality
  - `mc_estimator.py`: Monte Carlo cardinality estimator for query selectivity
- **`origami/utils/`**: Utilities and configuration
  - `config.py`: Configuration classes using OmegaConf
  - `query_utils.py`: Query evaluation and selectivity calculation utilities
  - `common.py`: Common utilities, symbols, operators, and helper functions

### Key Architecture Features

1. **Transformer-based**: Uses transformer architecture for processing sequential JSON tokens
2. **Guardrails System**: Three modes (NONE, STRUCTURE_ONLY, STRUCTURE_AND_VALUES) to enforce valid JSON generation
3. **Position Encoding**: Multiple methods (INTEGER, SINE_COSINE, KEY_VALUE) for sequence positioning
4. **Shuffled Training**: Can train with shuffled key/value pairs for better generalization
5. **Schema-aware**: Tracks field paths and value vocabularies for validation
6. **Cardinality Estimation**: Monte Carlo estimator for query selectivity prediction

### Configuration System

The project uses OmegaConf for configuration management with dataclasses:
- `ModelConfig`: Model architecture parameters
- `TrainConfig`: Training hyperparameters
- `PipelineConfig`: Data processing options
- `DataConfig`: Data source configuration

### Data Sources Supported

- MongoDB collections (with +srv URI support)
- JSON files (.json, .jsonl)
- CSV files
- Directories containing supported file types

### Model Presets

Available model sizes: xs, small (default), medium, large, xl
- Default: 4 layers, 4 attention heads, 128 hidden dimensions

## Testing

Tests are organized by component:
- `tests/cli/`: CLI functionality tests
- `tests/model/`: Model component tests
- `tests/preprocessing/`: Data preprocessing tests
- `tests/inference/`: Inference component tests (including MC estimator)
- `tests/utils/`: Utility function tests

Run tests with: `pytest`

## Key Files

- `setup.py`: Package configuration with dependencies
- `pyproject.toml`: Build system and ruff configuration
- `CLI.md`: Detailed CLI documentation
- `requirements.txt`: Python dependencies
- `experiments/`: Experiment scripts for paper reproduction
- `notebooks/`: Example Jupyter notebooks
  - `example_origami_mc_ce.ipynb`: Monte Carlo cardinality estimation demo

## Monte Carlo Cardinality Estimation

The `MCEstimator` class provides query selectivity estimation using Monte Carlo sampling:

### Usage Example
```python
from origami.inference import MCEstimator
from mdbrtools.query import Query, Predicate

# Initialize estimator with trained model and pipeline
estimator = MCEstimator(model, pipeline, batch_size=1000)

# Create a query
query = Query()
query.add_predicate(Predicate('field_name', 'gte', (min_value,)))
query.add_predicate(Predicate('field_name', 'lte', (max_value,)))

# Estimate selectivity
probability, samples = estimator.estimate(query, n=1000)
cardinality = probability * collection_size
```

### How It Works
1. Calculates query region size |E| (number of discrete states matching query)
2. Generates n uniform samples within the query region
3. Computes model probability f(x) for each sample
4. Returns Monte Carlo estimate: P(query) = |E| * mean(f(x))

### Query Utilities
- `evaluate_ground_truth(query, docs)`: Count documents matching query predicates
- `calculate_selectivity(query, docs)`: Calculate fraction of documents matching query
- `compare_estimate_to_ground_truth(query, docs, estimated_prob)`: Compare estimates with actual counts and compute error metrics (q-error, relative error, etc.)