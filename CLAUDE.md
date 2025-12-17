# CLAUDE.md - Claude Code Context for Origami v2

## Project Overview

Origami is a transformer-based architecture for JSON object classification and generation. Unlike standard language models that process flat sequences, Origami understands JSON structure through Key-Value Position Encoding (KVPE) and enforces valid JSON syntax via grammar constraints.

**Core use case:** Given a JSON object with a missing field, predict the value of that field based on the other fields in the object.

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

This is controlled by `UpscaledDataset` with `upscale_factor` parameter.

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
│   ├── config.py          # Configuration
│   ├── embeddings.py      # Embedding layer
│   ├── backbone.py        # Transformer backbone
│   └── heads.py           # Output heads
├── constraints/         # Grammar constraints
│   └── json_grammar.py    # PDA implementation
├── inference/           # Inference components
│   ├── generator.py       # JSON generation
│   ├── predictor.py       # Field prediction
│   └── embedder.py        # Embedding extraction
├── training/            # Training components
│   ├── trainer.py         # Training loop
│   ├── dataset.py         # Dataset classes
│   └── collator.py        # Batch collation
├── preprocessing/       # Data preprocessing
│   ├── numeric_discretizer.py  # Binning high-cardinality numerics
│   └── target_field.py    # Target field utilities
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
Inference components (Generator, Predictor, Embedder) always use CPU:

```python
# Model may be on GPU, but inference happens on CPU
predictor = OrigamiPredictor(model, tokenizer)
assert predictor.device == torch.device("cpu")
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

## Key Classes and Their Responsibilities

| Class | Responsibility |
|-------|---------------|
| `JSONTokenizer` | Tokenize JSON objects, manage vocabulary |
| `OrigamiModel` | Main model with embeddings, backbone, heads |
| `KeyValuePositionEncoding` | Encode paths through JSON structure |
| `JSONGrammarPDA` | Enforce valid JSON syntax |
| `OrigamiGenerator` | Generate complete JSON objects |
| `OrigamiPredictor` | Predict field values |
| `OrigamiEmbedder` | Extract embeddings for downstream tasks |
| `OrigamiTrainer` | Training loop with LR warmup, checkpointing |
| `OrigamiDataCollator` | Batch sequences with left-padding |
| `UpscaledDataset` | Data augmentation via key-order shuffling |

## Configuration

### OrigamiConfig (Model)
```python
OrigamiConfig(
    vocab_size=1000,
    d_model=256,
    n_heads=8,
    n_layers=6,
    d_ff=1024,
    max_depth=8,
    dropout=0.1,
    pooling_type="sum",  # or "weighted", "rotary", "gru", "transformer"
    use_grammar_constraints=True,
)
```

### TrainingConfig
```python
TrainingConfig(
    batch_size=32,
    learning_rate=1e-4,
    num_epochs=10,
    warmup_steps=1000,
    gradient_clip=1.0,
)
```

## Implementation Status

### Complete (Phases 1-5)
- Tokenization with path tracking
- KVPE with 5 pooling strategies
- Transformer backbone with causal attention
- Grammar constraints (PDA)
- Training loop with key-order shuffling
- Inference: Generator, Predictor, Embedder
- NumericDiscretizer for high-cardinality fields

### Partial (Phase 6)
- ContinuousHead for numeric prediction (done)
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
    config = OrigamiConfig(..., use_grammar_constraints=True)
    model = OrigamiModel(config, vocab=tokenizer.vocab)
    # Grammar only applied when labels provided
    output = model(..., labels=labels)
```

## References

- **Original ORIGAMI paper**: [arXiv:2412.17348](https://arxiv.org/abs/2412.17348)
- **Original implementation**: https://github.com/rueckstiess/origami
