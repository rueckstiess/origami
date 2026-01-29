# CLAUDE.md - Claude Code Context for Origami v2

## Project Overview

Origami is a transformer-based architecture for JSON object classification and generation. Unlike standard language models that process flat sequences, Origami understands JSON structure through Key-Value Position Encoding (KVPE) and enforces valid JSON syntax via grammar constraints.

**Core use case:** Given a JSON object with a missing field, predict the value of that field based on the other fields in the object.

## Installation

This is Origami v2, a complete rewrite published on PyPI as `origami-ml`. Currently in alpha:

```bash
# Install v2 alpha (opt-in)
pip install origami-ml==2.0.0a1

# Or install any pre-release
pip install origami-ml --pre

# Default install still gets v1 (0.3.0)
pip install origami-ml
```

**GitHub:** https://github.com/rueckstiess/origami (branch: `v2`)

## Architecture Overview

```
Input JSON → Tokenizer → Model → Prediction/Generation
                ↓
         [Tokens + Paths]
                ↓
    ┌───────────────────────┐
    │   Token Embeddings    │
    │         +             │
    │   KVPE (path info)    │
    └───────────────────────┘
                ↓
    ┌───────────────────────┐
    │  Transformer Backbone │
    │   (causal attention)  │
    └───────────────────────┘
                ↓
    ┌───────────────────────┐
    │     Output Heads      │
    │  Discrete / Continuous│
    └───────────────────────┘
```

## Key Architecture Differences from Standard Transformers

### 1. Key-Value Position Encoding (KVPE)
Instead of standard positional encoding (1, 2, 3...), KVPE encodes the **path through the JSON structure**:

```python
# Standard transformer: position = token index
# Origami: position = path through JSON tree

{"user": {"name": "Alice"}}
# Path to "Alice" = [KeyElement("user"), KeyElement("name")]
```

Five pooling strategies combine path element embeddings:
- `sum`: Simple sum (commutative - order doesn't matter)
- `weighted`: Learnable depth weights
- `rotary`: Rotary position encoding per depth
- `gru`: GRU processes path sequentially (order matters)
- `transformer`: Self-attention over path elements

### 2. Grammar Constraints via PDA
A Pushdown Automaton (PDA) enforces valid JSON syntax:
- **Training**: Full grammar mask computed for all positions
- **Inference**: Incremental grammar state updated per step (O(1) per step)

```python
# During training (labels provided):
output = model(input_ids, ..., labels=labels)  # Grammar mask applied

# During inference (no labels):
output = model(input_ids, ...)  # No grammar mask - Generator handles it
```

### 3. Left-Padding for Batched Prediction
Sequences are **left-padded** so all end at the same position:

```
# Right-padding (standard) - sequences end at different positions:
[START, a, 1, END, PAD, PAD]  <- ends at pos 3
[START, a, 1, b, 2, END]      <- ends at pos 5

# Left-padding (Origami) - all sequences end at same position:
[PAD, PAD, START, a, 1, END]  <- ends at pos 5
[START, a, 1, b, 2, END]      <- ends at pos 5
```

This enables `logits[:, -1, :]` to get next-token predictions for all sequences simultaneously.

### 4. Key-Order Shuffling (Data Augmentation)
JSON object keys have no inherent order, so we randomly shuffle key order during training:

```python
# Original: {"name": "Alice", "age": 30}
# Shuffled: {"age": 30, "name": "Alice"}
```

Shuffling is controlled by `TrainingConfig.shuffle_keys` (default True). Each time a sample is accessed during training, a fresh random permutation is generated. This forces the model to learn from key semantics rather than position.

### 5. Continuous Numeric Handling (MoG Head)
High-cardinality numeric fields can be handled in two ways:
- **Discretize**: Bin into categories (e.g., 20 bins) - treated as categorical
- **Scale**: Normalize with StandardScaler, use Mixture of Gaussians (MoG) output head

With `numeric_mode="scale"`:
1. `NumericScaler` transforms high-cardinality numerics to mean=0, std=1
2. Values are wrapped in `ScaledNumeric(value)` objects
3. Tokenizer emits `NUM` token (ID 9) with the scaled value in `numeric_values` tensor
4. Model's continuous head outputs MoG parameters (weights, means, log_vars)
5. At inference, sample from MoG when `NUM` token is generated
6. Pipeline inverse-transforms predictions back to original scale

```python
# Flow through the system:
{"price": 50000}  # Original
{"price": ScaledNumeric(1.23)}  # After NumericScaler.transform()
[..., KEY("price"), NUM, ...]  # Tokenized (NUM token)
numeric_values = [..., 0, 1.23, ...]  # Scaled value stored separately
# Model forward pass uses numeric_values for conditioning
# MoG head predicts distribution for next NUM value
```

### 6. Vocabulary Size Limiting (max_vocab_size)
Large datasets can create vocabularies with many rare values. The `max_vocab_size` config limits vocabulary by pruning least-frequent ValueTokens:

```python
DataConfig(max_vocab_size=10000)  # Limit to 10k tokens
```

**Pruning rules:**
- Grammar tokens (IDs 0-9) are ALWAYS preserved - required for JSON syntax
- KeyTokens are ALWAYS preserved - needed for KVPE and NumericScaler inverse_transform
- ValueTokens are pruned by frequency (least frequent first)
- Pruned values encode to `UNK_VALUE` (ID 8) during tokenization

```python
# After fitting with max_vocab_size, access pruning statistics:
tokenizer.fit(data, max_vocab_size=5000)
stats = tokenizer.pruning_stats
print(f"Pruned {stats.num_values_pruned} values")
print(f"Frequency threshold: {stats.value_frequency_threshold}")

# Vocabulary also provides frequency inspection methods:
vocab.get_value_frequencies()  # Returns dict[Any, int]
vocab.get_most_common_values(10)  # Returns list[(value, count)]
```

If `max_vocab_size` is smaller than `num_grammar_tokens + num_keys`, a `ValueError` is raised.

## Inference Architecture

### Generator Contains ALL Generation Logic
The `OrigamiGenerator` is the single source of truth for generation:
- Applies grammar constraints incrementally via PDA
- Handles token sampling (temperature, top-k, top-p)
- Samples from MoG head when NUM token is generated
- Manages path state tracking for KVPE

```
Generator
├── generate()              # Generate full objects from scratch
├── generate_from_tensors() # Core loop - ALL generation goes through here
├── get_value_distribution() # Get grammar-constrained probabilities (no sampling)
└── _sample()               # Token sampling with temperature/top-k/top-p
```

### Predictor Delegates to Generator
The `OrigamiPredictor` does NOT contain generation logic. It:
1. Prepares input tensors (reorder keys, tokenize, truncate at target)
2. Calls Generator's methods
3. Formats results

```python
# Predictor.predict_batch() internally does:
truncated = self._truncate_at_target_key(batch, target_key)
values = self._generator.generate_from_tensors(
    ...,
    numeric_values=truncated.numeric_values,  # MUST pass for conditioning!
    stop_after_value=True,
    temperature=0.0,  # Greedy decoding
)
```

**Critical**: `numeric_values` must be passed to Generator for conditioning on numeric context. Without it, the model ignores numeric field values when predicting.

### Prediction API
```python
# Simple prediction - returns the value directly
value = predictor.predict(obj, target_key)  # Returns: Any

# Batch prediction - returns list of values
values = predictor.predict_batch(objects, target_key)  # Returns: list[Any]

# Probability distribution - returns grammar-constrained probabilities
probs = predictor.predict_proba(obj, target_key)  # Returns: dict[Any, float]
top_k = predictor.predict_proba(obj, target_key, top_k=5)  # Returns: list[(Any, float)]
```

## Pipeline API (OrigamiPipeline)

The `OrigamiPipeline` provides an end-to-end API for training and inference:

```python
from origami import OrigamiPipeline, OrigamiConfig, ModelConfig, TrainingConfig, DataConfig

# Configure with nested structure
config = OrigamiConfig(
    model=ModelConfig(d_model=64, n_layers=4),
    training=TrainingConfig(num_epochs=20),
    data=DataConfig(numeric_mode="scale", cat_threshold=100),
)

# Train
pipeline = OrigamiPipeline(config)
pipeline.fit(train_data, eval_data=eval_data)

# Predict
value = pipeline.predict({"a": 1, "b": None}, target_key="b")
values = pipeline.predict_batch(objects, target_key="b")
probs = pipeline.predict_proba(obj, target_key="b", top_k=5)

# Generate
samples = pipeline.generate(num_samples=10, temperature=0.8)

# Evaluate (computes loss and optional prediction metrics)
from origami.training import accuracy
results = pipeline.evaluate(test_data)  # Just loss
results = pipeline.evaluate(test_data, target_key="b", metrics={"acc": accuracy})

# Save/Load (preserves all state: model, tokenizer, preprocessor)
pipeline.save("model.pt")
loaded = OrigamiPipeline.load("model.pt")
```

The pipeline handles:
- Automatic preprocessing (NumericScaler or NumericDiscretizer)
- Tokenizer fitting
- Model creation with appropriate config
- Training loop
- Inverse transformation of predictions back to original scale

## Project Structure

```
origami/
├── tokenizer/           # JSON tokenization
│   ├── json_tokenizer.py   # Main tokenizer
│   ├── vocabulary.py       # Token types and vocab
│   └── path.py            # Path representation
├── position_encoding/   # KVPE implementation
│   ├── kvpe.py            # Main KVPE module
│   └── pooling.py         # Pooling strategies
├── model/               # Model components
│   ├── origami_model.py   # Main model
│   ├── embeddings.py      # Embedding layer
│   ├── backbone.py        # Transformer backbone
│   └── heads.py           # Output heads (discrete + continuous MoG)
├── constraints/         # Grammar constraints
│   └── json_grammar.py    # PDA implementation
├── inference/           # Inference components
│   ├── generator.py       # JSON generation (ALL generation logic here)
│   ├── predictor.py       # Field prediction (delegates to Generator)
│   ├── embedder.py        # Embedding extraction
│   └── evaluator.py       # Unified loss and metrics evaluation
├── pipeline/            # High-level API
│   └── pipeline.py        # OrigamiPipeline end-to-end API
├── config.py            # All config classes (ModelConfig, TrainingConfig, DataConfig, OrigamiConfig)
├── training/            # Training components
│   ├── trainer.py         # Training loop
│   ├── dataset.py         # Dataset classes
│   ├── collator.py        # Batch collation
│   ├── callbacks.py       # Training callbacks (ProgressCallback, MetricsCallback)
│   └── metrics.py         # Evaluation metrics (accuracy, rmse, etc.)
├── preprocessing/       # Data preprocessing
│   ├── numeric_scaler.py      # StandardScaler for continuous numerics
│   ├── numeric_discretizer.py # Binning high-cardinality numerics
│   └── target_field.py        # Target field utilities
├── cli/                 # Command-line interface
│   ├── main.py             # Entry point and train command
│   ├── predict.py          # Predict subcommand
│   ├── generate.py         # Generate subcommand
│   ├── evaluate.py         # Evaluate subcommand
│   ├── embed.py            # Embed subcommand
│   └── data_loaders.py     # Shared data loading utilities
└── utils/               # Utilities
    └── device.py          # Device management
```

## Development Commands

### Package Management (uv)
```bash
# Install dependencies
uv sync

# Run any command with uv
uv run <command>
```

### Linting and Formatting (ruff)
```bash
# Check for issues
uv run ruff check .

# Auto-fix issues
uv run ruff check --fix .

# Format code
uv run ruff format .
```

### Testing (pytest)
```bash
# Run all tests
uv run pytest tests/

# Run with verbose output
uv run pytest tests/ -v

# Run specific test file
uv run pytest tests/test_model.py

# Run specific test class
uv run pytest tests/test_model.py::TestOrigamiModel

# Run with coverage
uv run pytest tests/ --cov=origami
```

### Training Example
```bash
uv run python examples/train_jsonl.py --data datasets/car.jsonl --target-key target
```

### CLI Commands (origami)
```bash
# Train a model
origami train -d data.jsonl -t label -o model.pt

# Predict values
origami predict -m model.pt -d input.jsonl -t label

# Generate synthetic data
origami generate -m model.pt -n 100 -o samples.jsonl

# Evaluate model performance
origami evaluate -m model.pt -d test.jsonl -t label --metrics accuracy

# Create embeddings
origami embed -m model.pt -d data.jsonl -o embeddings.npy

# All commands support -v/--verbose to display model configuration
origami predict -m model.pt -d input.jsonl -t label -v
```

## Common Pitfalls and Mistakes

### 1. Grammar Constraints: Training vs Inference
Grammar masking is ONLY applied during training (when `labels` is provided):

```python
# WRONG: Expecting grammar mask during inference
output = model(input_ids, ...)  # No grammar mask!

# RIGHT: Use Generator for inference with grammar
generator = OrigamiGenerator(model, tokenizer)
results = generator.generate(...)  # Handles grammar incrementally
```

### 2. Left-Padding Alignment
When working with left-padded batches, path encoding must align with tokens:

```python
# PAD positions should have zero path_lengths
# Real tokens start at position (seq_len - original_len)
```

### 3. Device Management
Inference components (Generator, Predictor) use the model's current device dynamically:

```python
# Predictor/Generator adapt to whatever device the model is on
model.to("cpu")  # Move to CPU for faster inference
predictor = OrigamiPredictor(model, tokenizer)
assert predictor.device == torch.device("cpu")

# During training evaluation, device management is automatic:
# The OrigamiEvaluator handles model mode (eval/train) transitions
```

### 4. Ruff Linting Rules
Common issues to avoid:
- **B905**: Always use `strict=True` with `zip()` for equal-length iterables
- **UP007**: Use `X | Y` instead of `Union[X, Y]`
- **F841**: Remove unused variables
- **B007**: Prefix unused loop variables with `_`

### 5. Test Fixtures vs Direct Instantiation
Use fixtures for shared setup, but create fresh instances when testing state:

```python
# Use fixture for read-only access
def test_something(self, tokenizer):
    ...

# Create fresh instance when modifying state
def test_modification(self):
    tokenizer = JSONTokenizer()
    tokenizer.fit(data)
    ...
```

### 6. Grammar State Initialization
When generating from a prefix, grammar state must be initialized from the prefix tokens:

```python
# Initialize grammar state from existing tokens (not from scratch)
state = pda.init_state_from_tokens(prefix_tokens, batch_size, device)
```

### 7. Path Types in KVPE
Path types are encoded as integers:
- `0`: Padding/empty
- `1`: Key element
- `2`: Index element

### 8. Attention Mask Convention
Attention mask uses `True` for real tokens, `False` for padding:

```python
attention_mask = torch.tensor([
    [False, False, True, True, True],  # 2 PAD + 3 real tokens
])
```

### 9. Numeric Values Must Be Passed for Conditioning
When predicting with continuous mode, `numeric_values` MUST be passed to Generator:

```python
# WRONG: Model ignores numeric context, predicts unconditional mean
values = generator.generate_from_tensors(input_ids, ..., numeric_values=None)

# RIGHT: Model conditions on numeric values in context
values = generator.generate_from_tensors(input_ids, ..., numeric_values=batch.numeric_values)
```

Without `numeric_values`, the model predicts the marginal distribution (mean) instead of the conditional distribution given the context.

### 10. Predictor API Changed
The prediction API was simplified:

```python
# OLD (removed):
predictions = predictor.predict_batch(objects, target_key, top_k=1)  # Returned list[list[tuple]]
prediction = predictor.predict(obj, target_key, top_k=1, return_probs=True)

# NEW:
predictions = predictor.predict_batch(objects, target_key)  # Returns list[Any]
prediction = predictor.predict(obj, target_key)  # Returns Any
probs = predictor.predict_proba(obj, target_key, top_k=5)  # Returns list[(Any, float)]
```

## Key Classes and Their Responsibilities

| Class | Responsibility |
|-------|---------------|
| `OrigamiPipeline` | End-to-end API: preprocessing, training, inference, save/load |
| `OrigamiConfig` | Root configuration composing model, training, and data configs |
| `ModelConfig` | Model architecture parameters (d_model, n_layers, etc.) |
| `TrainingConfig` | Training hyperparameters (batch_size, learning_rate, etc.) |
| `DataConfig` | Data preprocessing parameters (numeric_mode, cat_threshold, etc.) |
| `JSONTokenizer` | Tokenize JSON objects, manage vocabulary |
| `OrigamiModel` | Main model with embeddings, backbone, heads |
| `KeyValuePositionEncoding` | Encode paths through JSON structure |
| `JSONGrammarPDA` | Enforce valid JSON syntax |
| `OrigamiGenerator` | ALL generation logic: grammar, sampling, MoG, path tracking |
| `OrigamiPredictor` | Prepare tensors, delegate to Generator, format results |
| `OrigamiEmbedder` | Extract embeddings for downstream tasks |
| `OrigamiEvaluator` | Unified loss and prediction metrics evaluation |
| `OrigamiTrainer` | Training loop with LR warmup, checkpointing |
| `ProgressCallback` | Training progress bars and logging |
| `MetricsCallback` | Compute and log evaluation metrics during training |
| `OrigamiDataCollator` | Batch sequences with left-padding |
| `OrigamiDataset` | Dataset wrapper with optional key-order shuffling |
| `NumericScaler` | StandardScaler for continuous numeric fields |
| `NumericDiscretizer` | Bin numeric fields into categories |
| `Vocabulary` | Token storage with frequency tracking and pruning support |
| `PruningStats` | Statistics from vocabulary pruning (original/pruned size, threshold) |

## Configuration

Configuration uses nested dataclasses. The root `OrigamiConfig` composes `ModelConfig`, `TrainingConfig`, and `DataConfig`:

### ModelConfig (Model Architecture)
```python
ModelConfig(
    d_model=128,          # Hidden dimension
    n_heads=4,            # Attention heads
    n_layers=4,           # Transformer layers
    d_ff=512,             # Feedforward dimension
    dropout=0.1,
    max_depth=32,         # Maximum JSON nesting depth
    kvpe_pooling="sum",   # KVPE pooling: "sum", "weighted", "rotary", "gru", "transformer"
    backbone="transformer",  # "transformer", "lstm", "mamba"
    num_mixture_components=5,  # MoG components for continuous output
)
```

### TrainingConfig (Training Hyperparameters)
```python
TrainingConfig(
    batch_size=32,
    learning_rate=1e-3,
    num_epochs=10,
    warmup_steps=1000,
    weight_decay=0.01,
    shuffle_keys=True,     # Key-order shuffling augmentation
    target_key=None,       # Target field for prediction tasks
    constrain_grammar=True,  # Apply grammar constraints during training
    constrain_schema=False,  # Apply schema constraints during training
)
```

### DataConfig (Data Preprocessing)
```python
DataConfig(
    numeric_mode="none",   # "none", "discretize", or "scale"
    cat_threshold=100,     # Fields with >N unique values get preprocessed
    n_bins=20,             # Number of bins for discretization
    max_vocab_size=0,      # Max vocabulary size (0 = unlimited). Prunes rare ValueTokens.
    infer_schema=False,    # Auto-infer JSON Schema from training data
)
```

### OrigamiConfig (Root - Nested Composition)
```python
OrigamiConfig(
    model=ModelConfig(d_model=256, n_layers=6),
    training=TrainingConfig(batch_size=64, num_epochs=20),
    data=DataConfig(numeric_mode="scale"),
    device="auto",  # "auto", "cpu", "cuda", "mps"
)
```

Note: `vocab_size` is not in config - the model derives it from `len(vocab)` when constructed.

## Implementation Status

### Complete (Phases 1-6)
- Tokenization with path tracking
- KVPE with 5 pooling strategies
- Transformer backbone with causal attention
- Grammar constraints (PDA)
- Training loop with key-order shuffling
- Training callbacks (ProgressCallback, MetricsCallback)
- Inference: Generator, Predictor, Embedder, Evaluator
- NumericDiscretizer for high-cardinality fields
- NumericScaler + ContinuousHead (MoG) for continuous numerics
- OrigamiPipeline end-to-end API
- Unified prediction API with predict/predict_batch/predict_proba
- Pipeline.evaluate() for loss and metrics computation
- Vocabulary pruning with max_vocab_size and frequency tracking
- CLI with train, predict, generate, evaluate, embed subcommands

### Partial
- LSTM/Mamba backbones (stubs only)
- TabularTokenizer (not implemented)
- HuggingFace integration (not implemented)

### Not Started (Phase 7)
- Validation experiments
- Benchmarking

## Testing Patterns

### Parametrized Device Tests
```python
@pytest.mark.parametrize("device", AVAILABLE_DEVICES)
def test_on_device(self, device):
    model = model.to(device)
    ...
```

### Fixtures for Common Setup
```python
@pytest.fixture
def tokenizer(self):
    tokenizer = JSONTokenizer()
    tokenizer.fit([{"a": 1}, {"b": 2}])
    return tokenizer
```

### Testing Grammar Constraints
```python
def test_grammar(self):
    config = ModelConfig()
    model = OrigamiModel(config, vocab=tokenizer.vocab)
    # Attach grammar PDA (normally done by trainer when constrain_grammar=True)
    from origami.constraints.json_grammar import JSONGrammarPDA
    model._grammar_pda = JSONGrammarPDA(tokenizer.vocab, max_depth=config.max_depth)
    # Grammar applied via compute_grammar_mask() when labels provided
    output = model(..., labels=labels)
```

### End-to-End Validation Dataset
The **car** dataset (MongoDB: `json2vec.car`) is a good smoke test for verifying the model learns correctly. It should reach close to 100% accuracy after ~100 epochs. Use this to catch regressions in the training pipeline.

## References

- **Original ORIGAMI paper**: [arXiv:2412.17348](https://arxiv.org/abs/2412.17348)
- **Original implementation**: https://github.com/rueckstiess/origami
