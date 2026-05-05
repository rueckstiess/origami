# Concepts

This page explains the key ideas behind Origami for readers who want to understand how it works. No prior knowledge of machine learning is assumed — we explain concepts in plain terms.

## JSON Tokenization

Before Origami can learn from JSON data, it needs to convert each JSON object into a sequence of **tokens** — small, discrete units that the model processes one at a time. This is similar to how a language model breaks text into words or subwords.

Origami's tokenizer preserves the full hierarchical structure of JSON. Each structural element gets its own token:

```
Input:  {"name": "Alice", "age": 30}

Tokens: START  OBJ_START  KEY("name")  "Alice"  KEY("age")  30  OBJ_END  END
```

- `START` and `END` mark the boundaries of the sequence
- `OBJ_START` and `OBJ_END` mark object boundaries
- `KEY("name")` indicates that the next value belongs to the "name" field
- `"Alice"` and `30` are value tokens

Nested objects and arrays are tokenized the same way, preserving the full tree structure:

```
Input:  {"user": {"name": "Alice"}, "tags": ["admin", "active"]}

Tokens: START  OBJ_START  KEY("user")  OBJ_START  KEY("name")  "Alice"  OBJ_END
                   KEY("tags")  ARR_START  "admin"  "active"  ARR_END  OBJ_END  END
```

This is different from tabular approaches that flatten data into fixed columns. Origami can handle nested objects, arrays, and mixed types natively.

## Key-Value Position Encoding (KVPE)

In most sequence models, each token is assigned a position number: the 1st token, the 2nd token, and so on. This works well for text, where word order matters. But in JSON, the order of keys in an object is meaningless — `{"a": 1, "b": 2}` and `{"b": 2, "a": 1}` are the same object.

Origami replaces sequential position numbers with **path-based positions**. Instead of saying "this is the 5th token", KVPE says "this token is the value of the 'name' key inside the 'user' object." The position of each token is defined by its path through the JSON tree:

```
{"user": {"name": "Alice", "age": 30}}

Token       Path
─────       ────
OBJ_START   (root)
KEY(user)   (root)
OBJ_START   user
KEY(name)   user
"Alice"     user → name
KEY(age)    user
30          user → age
OBJ_END     user
OBJ_END     (root)
```

The value `"Alice"` has path `[user, name]`, while `30` has path `[user, age]`. These paths are converted to vector embeddings and added to the token embeddings, giving the model structural awareness.

### Pooling Strategies

Path elements are individually embedded, then combined into a single vector using a **pooling strategy**. Five strategies are available:

| Strategy | Description | When to use |
|----------|-------------|-------------|
| `sum` | Add all path element embeddings together | Default. Good general-purpose choice. |
| `weighted` | Learned weights per depth level | When depth hierarchy matters (deeper elements should be weighted differently). |
| `rotary` | Rotary position encoding applied per depth | When you want position-sensitive encoding at each depth level. |
| `gru` | Process path elements sequentially with a GRU | When the order of nesting matters (e.g., `a.b` should differ from `b.a`). |
| `transformer` | Self-attention over path elements | Maximum expressiveness. Higher computational cost. |

The default `sum` pooling works well for most use cases. Try alternatives if your data has deep nesting where the hierarchy structure carries important information.

## Grammar Constraints

When a model generates tokens one at a time, there's no guarantee that the output will be syntactically valid. It might produce an `ARR_END` token before it has produced an `ARR_START` token, or emit a value token where a key is expected.

Origami solves this with a **grammar constraint system**. At each generation step, a pushdown automaton (a type of parser) computes which tokens are valid given the current output. Invalid tokens are masked out — the model simply cannot choose them.

For example, after generating `OBJ_START KEY("name")`, the grammar only allows value tokens (`"Alice"`, `OBJ_START`, `ARR_START`, etc.) — not `OBJ_END` or another key. After generating `OBJ_START KEY("name") "Alice"`, it allows either another `KEY(...)` (for another field) or `OBJ_END` (to close the object).

The result: every model output is **guaranteed** to be syntactically valid JSON. This is enabled by default and requires no configuration.

### Schema Constraints

On top of grammar constraints, Origami optionally supports **schema constraints**. If you provide a JSON Schema (or let Origami infer one from your training data), the model is further restricted to only produce values that conform to the schema — correct data types, valid enum values, and proper object structure.

```python
config = OrigamiConfig(
    data=DataConfig(infer_schema=True),             # Learn schema from training data
    inference=InferenceConfig(constrain_schema=True), # Apply during inference
)
```

This is particularly useful for generation tasks where you want outputs to match the structure of your training data exactly.

## Key-Order Shuffling

Since JSON key order is meaningless, Origami randomly shuffles the order of keys each time an object is seen during training. The same object might be presented as:

```
Epoch 1: {"name": "Alice", "age": 30, "city": "London"}
Epoch 2: {"city": "London", "name": "Alice", "age": 30}
Epoch 3: {"age": 30, "city": "London", "name": "Alice"}
```

This forces the model to learn from field _content_ and _relationships_, not from the accident of which key appears first. It acts as free data augmentation — each shuffle is like a different training example.

Key-order shuffling is enabled by default (`TrainingConfig.shuffle_keys=True`).

## Handling Numeric Fields

By default, every distinct value in your data becomes its own token. This works well for categorical fields (country names, labels, boolean flags) but is a poor fit for continuous numeric fields like prices or temperatures, where there may be thousands of unique values.

Origami offers three modes for handling numeric fields, configured via `DataConfig.numeric_mode`:

### `"disabled"` (default)

Every value is treated as a discrete token. Best for data where numeric fields have few unique values (e.g., ratings 1-5, counts 0-10).

### `"discretize"`

High-cardinality numeric fields are automatically binned into a fixed number of categories. For example, a "price" field with values from 0 to 100,000 might be split into 20 bins. The model predicts which bin, and the bin center is returned as the prediction.

Good when you need approximate numeric predictions and want to keep things simple.

### `"scale"`

High-cardinality numeric fields are normalized (centered to mean 0, scaled to standard deviation 1), and the model uses a dedicated continuous output head to predict a probability distribution over possible values. At inference, values are sampled from this distribution and converted back to the original scale.

Best for truly continuous values (prices, measurements, coordinates) where you need precise numeric predictions.

### Which mode should I use?

- **Most numeric fields have <100 unique values** (ratings, counts, categories with numeric labels) → `"disabled"`
- **Numeric fields have many values, approximate predictions are fine** → `"discretize"`
- **Numeric fields are truly continuous, precision matters** → `"scale"`

The `cat_threshold` parameter (default: 100) controls which fields are considered "high-cardinality." Only fields with more than `cat_threshold` unique values are affected by `"discretize"` or `"scale"` — fields below the threshold are always treated as discrete tokens regardless of the mode.

```python
# Continuous numeric handling for fields with 50+ unique values
DataConfig(numeric_mode="scale", cat_threshold=50)
```

See [Configuration: DataConfig](configuration.md#dataconfig) for all options.
