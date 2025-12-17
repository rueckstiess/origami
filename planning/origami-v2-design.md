# ORIGAMI Reimplementation: Design Document

## Overview

ORIGAMI (Object RepresentatIon via Generative Autoregressive ModellIng) is a transformer-based architecture for supervised learning directly on semi-structured JSON data. This document specifies a clean reimplementation with:

- Modular, extensible architecture
- HuggingFace ecosystem compatibility
- Support for continuous value modeling
- Pluggable backbone architectures (Transformer, LSTM, SSM)
- Simplified tabular data mode

**Target**: Python 3.11+, PyTorch 2.0+

---

## Key Design Decisions (v2 vs Original Paper)

| Aspect | Original Paper | v2 Design | Rationale |
|--------|---------------|-----------|-----------|
| **Array tokens** | `Array(n)` with length | `ARRAY_START`/`ARRAY_END` | Generalizes to unseen lengths, no vocab bloat |
| **Root structure** | Implicit `START...END` | Explicit `START OBJ_START...OBJ_END END` | Uniform structure, prepares for root arrays |
| **Unknown tokens** | Single `UNK` | `UNK_KEY` + `UNK_VALUE` | Grammar can distinguish key vs value context |
| **KVPE pooling** | Sum (commutative) | Pluggable (sum, rotary, GRU, etc.) | Address `a[1][2]` vs `a[2][1]` ambiguity |
| **Continuous values** | All discrete | Optional MoG head for high-cardinality | Better regression support |
| **Backbones** | Transformer only | Pluggable (Transformer, LSTM, Mamba) | Ablation studies |

**MVP Scope**: Transformer backbone, sum pooling KVPE, discrete head only. Extensions designed but deferred.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                           ORIGAMI Model                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐    ┌──────────────┐    ┌───────────────────────┐  │
│  │  Tokenizer   │───▶│   Encoder    │───▶│    Backbone           │  │
│  │              │    │  (Embed+PE)  │    │ (Transformer/LSTM/SSM)│  │
│  └──────────────┘    └──────────────┘    └───────────┬───────────┘  │
│         │                                            │              │
│         │                                            ▼              │
│         │                               ┌───────────────────────┐   │
│         │                               │      Output Heads     │   │
│         │                               │  ┌─────────────────┐  │   │
│         │                               │  │  Discrete Head  │  │   │
│         │                               │  └─────────────────┘  │   │
│         │                               │  ┌─────────────────┐  │   │
│         │                               │  │ Continuous Head │  │   │
│         │                               │  │ (MoG, optional) │  │   │
│         │                               │  └─────────────────┘  │   │
│         │                               └───────────────────────┘   │
│         │                                            │              │
│         ▼                                            ▼              │
│  ┌──────────────┐                        ┌──────────────────────┐   │
│  │  Vocabulary  │                        │ Constrained Decoder  │   │
│  └──────────────┘                        │   (Grammar Mask)     │   │
│                                          └──────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Module Specifications

### 1. Tokenizer Module (`origami/tokenizer/`)

#### 1.1 Vocabulary (`vocabulary.py`)

**Purpose**: Manage bidirectional mapping between tokens and integer IDs.

**Token Types**:
```python
class TokenType(Enum):
    GRAMMAR = auto()  # Structural tokens
    KEY = auto()      # JSON keys
    VALUE = auto()    # JSON values

@dataclass(frozen=True)
class Token:
    """Base class - tokens are immutable and hashable."""
    token_type: TokenType

@dataclass(frozen=True)
class GrammarToken(Token):
    """Structural tokens: START, END, OBJ_START, OBJ_END, ARRAY_START, ARRAY_END, PAD, UNK_KEY, UNK_VALUE, NUM"""
    value: str  # "START", "END", etc.
    token_type: TokenType = field(default=TokenType.GRAMMAR, init=False)

@dataclass(frozen=True)
class KeyToken(Token):
    """JSON keys: Key("name"), Key("age")"""
    key: str
    token_type: TokenType = field(default=TokenType.KEY, init=False)

@dataclass(frozen=True)
class ValueToken(Token):
    """JSON values: "hello", 42, True, None - extensible to BSON types"""
    value: Any  # str, int, float, bool, None - type preserved via Python type
    token_type: TokenType = field(default=TokenType.VALUE, init=False)
```

**Design Notes**:
- `ValueToken.value: Any` allows extension to arbitrary types (e.g., BSON types for MongoDB)
- Frozen dataclasses provide `__hash__` and `__eq__` for free
- `ValueToken(42)` (int) and `ValueToken("42")` (str) hash differently, preserving type information

**Special Tokens**:
| Token | ID | Purpose |
|-------|-----|---------|
| `START` | 0 | Document start |
| `END` | 1 | Document end |
| `OBJ_START` | 2 | Object start `{` |
| `OBJ_END` | 3 | Object end `}` |
| `ARRAY_START` | 4 | Array start `[` |
| `ARRAY_END` | 5 | Array end `]` |
| `PAD` | 6 | Padding |
| `UNK_KEY` | 7 | Unknown key (for grammar constraints on unseen keys) |
| `UNK_VALUE` | 8 | Unknown value (for grammar constraints on unseen values) |
| `NUM` | 9 | Numeric placeholder (for continuous head) |

Dynamic tokens (keys and values) start at ID 10, interleaved as encountered during `fit()`.

**Design Note**: Unlike the original ORIGAMI paper which used `Array(n)` tokens encoding array length, we use `ARRAY_START`/`ARRAY_END` delimiters. This:
- Eliminates vocabulary bloat (no need for Array(1), Array(2), ..., Array(1000))
- Enables generalization to array lengths not seen during training
- Simplifies the grammar (no countdown tracking needed)
- Matches actual JSON syntax more closely

Array positions are tracked separately during tokenization and encoded via position embeddings.

**Vocabulary Class Interface**:
```python
class Vocabulary:
    def __init__(self): ...

    # Building vocabulary (during fit)
    def add_key(self, key: str) -> int: ...    # Idempotent, returns existing ID if already added
    def add_value(self, value: Any) -> int: ... # Idempotent, returns existing ID if already added
    def freeze(self) -> None: ...               # After freeze, add_key/add_value raise errors

    # Encoding/Decoding
    def encode(self, token: Token) -> int: ...  # Returns UNK_KEY/UNK_VALUE for unknown tokens
    def decode(self, token_id: int) -> Token: ...

    # Type queries (for grammar constraints)
    def is_key_token(self, token_id: int) -> bool: ...
    def is_value_token(self, token_id: int) -> bool: ...
    def is_grammar_token(self, token_id: int) -> bool: ...

    # Token set queries (for grammar mask generation)
    def get_all_key_ids(self) -> set[int]: ...            # All keys including UNK_KEY
    def get_all_primitive_value_ids(self) -> set[int]: ... # All values including UNK_VALUE, NUM

    # Persistence (using pickle for BSON extensibility)
    def save(self, path: str) -> None: ...
    @classmethod
    def load(cls, path: str) -> Vocabulary: ...

    # Properties
    @property
    def size(self) -> int: ...
    @property
    def pad_token_id(self) -> int: ...         # 6
    @property
    def unk_key_id(self) -> int: ...           # 7
    @property
    def unk_value_id(self) -> int: ...         # 8
    @property
    def num_token_id(self) -> int: ...         # 9
    @property
    def start_id(self) -> int: ...             # 0
    @property
    def end_id(self) -> int: ...               # 1
    @property
    def obj_start_id(self) -> int: ...         # 2
    @property
    def obj_end_id(self) -> int: ...           # 3
    @property
    def array_start_id(self) -> int: ...       # 4
    @property
    def array_end_id(self) -> int: ...         # 5
```

**Grammar constraint composition** (in `JSONGrammarConstraint`, not Vocabulary):
```python
def get_valid_value_tokens(self) -> set[int]:
    """Tokens valid when expecting a value (primitives + complex starters)."""
    return (
        self.vocab.get_all_primitive_value_ids()
        | {self.vocab.array_start_id, self.vocab.obj_start_id}
    )
```

#### 1.2 JSON Tokenizer (`json_tokenizer.py`)

**Purpose**: Convert JSON objects to/from token sequences with path tracking.

**Tokenization Rules**:
1. Document starts with `START`, ends with `END`
2. Root object wrapped in `OBJ_START` ... `OBJ_END` (enables future root-level arrays)
3. Each key-value pair: `Key(k)` followed by value token(s)
4. Nested objects: `OBJ_START` ... `OBJ_END`
5. Arrays: `ARRAY_START` value* `ARRAY_END`
6. Key-value pairs at same level can be shuffled (order-invariant)

**Example**:
```json
{
  "name": "Alice",
  "scores": [90, 85],
  "meta": {"active": true}
}
```
Tokenizes to:
```
START OBJ_START Key("name") "Alice" Key("scores") ARRAY_START 90 85 ARRAY_END Key("meta") OBJ_START Key("active") True OBJ_END OBJ_END END
```

**Future root-level array** (not MVP, but structure supports it):
```json
["item1", "item2"]
```
Would tokenize to:
```
START ARRAY_START "item1" "item2" ARRAY_END END
```

**Path Representation** (unified with KVPE module):
```python
@dataclass(frozen=True)
class KeyElement:
    key: str

@dataclass(frozen=True)
class IndexElement:
    index: int

Path = tuple[KeyElement | IndexElement, ...]
```

**Path Tracking**:
Each token is associated with its full path (keys and array indices).
Structural tokens (`OBJ_START`, `ARRAY_END`, etc.) receive the path of their container:

| Token | Path |
|-------|------|
| `START` | `()` |
| `OBJ_START` | `()` |
| `Key("name")` | `()` |
| `"Alice"` | `(KeyElement("name"),)` |
| `Key("scores")` | `()` |
| `ARRAY_START` | `(KeyElement("scores"),)` |
| `90` | `(KeyElement("scores"), IndexElement(0))` |
| `85` | `(KeyElement("scores"), IndexElement(1))` |
| `ARRAY_END` | `(KeyElement("scores"),)` |
| `Key("meta")` | `()` |
| `OBJ_START` | `(KeyElement("meta"),)` |
| `Key("active")` | `(KeyElement("meta"),)` |
| `True` | `(KeyElement("meta"), KeyElement("active"))` |
| `OBJ_END` | `(KeyElement("meta"),)` |
| `OBJ_END` | `()` |
| `END` | `()` |

**Nested array example**: `{"matrix": [[1, 2], [3, 4]]}`

| Token | Path |
|-------|------|
| `ARRAY_START` (outer) | `(KeyElement("matrix"),)` |
| `ARRAY_START` (row 0) | `(KeyElement("matrix"), IndexElement(0))` |
| `1` | `(KeyElement("matrix"), IndexElement(0), IndexElement(0))` |
| `2` | `(KeyElement("matrix"), IndexElement(0), IndexElement(1))` |
| `ARRAY_END` | `(KeyElement("matrix"), IndexElement(0))` |
| `ARRAY_START` (row 1) | `(KeyElement("matrix"), IndexElement(1))` |
| `3` | `(KeyElement("matrix"), IndexElement(1), IndexElement(0))` |
| `4` | `(KeyElement("matrix"), IndexElement(1), IndexElement(1))` |
| `ARRAY_END` | `(KeyElement("matrix"), IndexElement(1))` |
| `ARRAY_END` (outer) | `(KeyElement("matrix"),)` |

This unified representation means `matrix[0][1]` has path `(KeyElement("matrix"), IndexElement(0), IndexElement(1))` which is distinct from `matrix[1][0]` with path `(KeyElement("matrix"), IndexElement(1), IndexElement(0))` — non-commutative pooling strategies preserve this distinction.

**Tokenizer Interface**:
```python
@dataclass
class TokenizedInstance:
    tokens: list[Token]
    paths: list[Path]                    # Typed path tuples (KeyElement | IndexElement)
    numeric_values: list[float | None]   # Original values for NUM tokens (future continuous head)

@dataclass
class EncodedBatch:
    input_ids: torch.LongTensor         # (batch, seq_len)
    path_types: torch.LongTensor        # (batch, seq_len, max_depth) - 0=pad, 1=key, 2=index
    path_ids: torch.LongTensor          # (batch, seq_len, max_depth) - key vocab ID or array index
    path_lengths: torch.LongTensor      # (batch, seq_len) - path depth at each position
    attention_mask: torch.BoolTensor    # (batch, seq_len)
    numeric_values: torch.FloatTensor   # (batch, seq_len) - scaled values for NUM tokens
    numeric_mask: torch.BoolTensor      # (batch, seq_len) - True where NUM token
    lengths: torch.LongTensor           # (batch,) - sequence lengths before padding

class JSONTokenizer:
    def __init__(
        self,
        vocab: Vocabulary | None = None,
        max_depth: int = 32,
        max_array_position: int = 256,
    ): ...

    # Vocabulary building (separate from preprocessing pipeline)
    def fit(self, objects: Iterable[dict]) -> JSONTokenizer:
        """Build vocabulary from all keys and values. Does NOT handle numeric binning."""
        ...

    # Tokenization
    def tokenize(self, obj: dict, shuffle: bool = False) -> TokenizedInstance:
        """
        Convert JSON object to token sequence with path tracking.
        shuffle=True permutes key order at each level (for training).
        shuffle=False uses deterministic key order (for inference/eval).
        """
        ...

    def encode(self, obj: dict, shuffle: bool = False) -> EncodedInstance: ...
    def encode_batch(self, objects: list[dict], shuffle: bool = False) -> EncodedBatch: ...

    # Decoding
    def decode(self, token_ids: list[int]) -> dict:
        """
        Reconstruct JSON object from token sequence.
        Raises DecodeError with token sequence for invalid/malformed input.
        """
        ...

    def save(self, path: str) -> None: ...
    @classmethod
    def load(cls, path: str) -> JSONTokenizer: ...

class DecodeError(Exception):
    """Raised when decode() encounters invalid token sequence."""
    def __init__(self, message: str, tokens: list[int], position: int):
        self.tokens = tokens
        self.position = position
        super().__init__(f"{message} at position {position}\nTokens: {tokens}")
```

**Shuffling**: The `shuffle` parameter controls key-order permutation:
- `tokenize(..., shuffle=True)`: Random permutation (training)
- `tokenize(..., shuffle=False)`: Deterministic order (inference, evaluation)
- Shuffling happens at tokenization time, keeping tokenizer stateless

#### 1.3 Numeric Discretizer (`numeric_discretizer.py`)

**Purpose**: Handle high-cardinality numeric values in discrete mode (MVP).

For discrete-only training, we bin high-cardinality numeric fields using sklearn's `KBinsDiscretizer`. Low-cardinality fields pass through unchanged.

```python
class NumericDiscretizer:
    """
    Preprocessing step: bins high-cardinality numerics, preserves low-cardinality.
    Part of sklearn-style preprocessing pipeline.
    """

    def __init__(
        self,
        cardinality_threshold: int = 100,  # Fields with > this many unique values get binned
        n_bins: int = 20,
        strategy: Literal["uniform", "quantile", "kmeans"] = "quantile",
    ): ...

    def fit(self, objects: Iterable[dict]) -> NumericDiscretizer:
        """
        Analyze all numeric fields by path. For each:
        - Count unique values
        - If > threshold: fit KBinsDiscretizer (per-field, to handle different ranges)
        - If <= threshold: mark as pass-through
        """
        ...

    def transform(self, obj: dict) -> dict:
        """
        Replace high-cardinality values with bin centers (floats).
        Low-cardinality values pass through with original type (int stays int).

        Example:
            {"age": 37, "rating": 4}  # age high-cardinality, rating low
            → {"age": 35.0, "rating": 4}  # age binned to center, rating unchanged
        """
        ...

    def fit_transform(self, objects: Iterable[dict]) -> list[dict]: ...

    def save(self, path: str) -> None: ...
    @classmethod
    def load(cls, path: str) -> NumericDiscretizer: ...
```

**Preprocessing Pipeline** (sklearn-style):
```python
# Discrete mode (MVP)
pipeline = [
    NumericDiscretizer(cardinality_threshold=100, n_bins=20, strategy="quantile"),
    # JSONTokenizer.fit() called after discretization
]

# Fit pipeline on training data
discretizer = NumericDiscretizer(...)
train_data = discretizer.fit_transform(raw_train_data)

tokenizer = JSONTokenizer()
tokenizer.fit(train_data)

# Transform test data (no fit, just transform)
test_data = discretizer.transform(raw_test_data)
```

#### 1.4 Tabular Tokenizer (`tabular_tokenizer.py`)

**Purpose**: Simplified tokenizer for tabular data (fixed schema).

For tabular data, we don't need the full JSON structure. Each row becomes:
```
START value_1 value_2 ... value_n END
```

Position encoding uses column names directly (no nesting).

```python
class TabularTokenizer:
    def __init__(
        self,
        columns: list[str],
        vocab: Vocabulary | None = None,
        continuous_columns: list[str] | None = None,
    ): ...

    def fit(self, rows: Iterable[dict]) -> TabularTokenizer: ...
    def encode(self, row: dict) -> EncodedInstance: ...
    def encode_batch(self, rows: list[dict]) -> EncodedBatch: ...
```

---

### 2. Position Encoding Module (`origami/position_encoding/`)

#### 2.1 Key-Value Position Encoding (`kvpe.py`)

**Purpose**: Encode hierarchical paths (keys and array indices) as position embeddings.

**Design Principle**: Position encoding is **pluggable** — multiple strategies can be swapped for experimentation and ablation studies.

**Unified Path Representation**:
Both keys and array indices are path elements:
```python
# a.b[2] → 
path = (KeyElement("a"), KeyElement("b"), IndexElement(2))

# a[1][2] → nested arrays
path = (KeyElement("a"), IndexElement(1), IndexElement(2))

# a[0].b → object inside array
path = (KeyElement("a"), IndexElement(0), KeyElement("b"))

class PathElement(ABC):
    """Base class for path elements."""
    pass

@dataclass(frozen=True)
class KeyElement(PathElement):
    key: str

@dataclass(frozen=True)  
class IndexElement(PathElement):
    index: int
```

**Pooling Strategies** (all implement `PathPooling` interface):

| Strategy | Order-Aware | Parallelizable | Notes |
|----------|-------------|----------------|-------|
| `sum` | ❌ | ✅ | Baseline (original paper), commutative |
| `weighted` | Partial | ✅ | Learned depth weights |
| `rotary` | ✅ | ✅ | Rotation by depth, approximate |
| `gru` | ✅ | ❌ | Sequential, fully non-commutative |
| `transformer` | ✅ | ✅ | Self-attention over path |

```python
class SumPooling(PathPooling):
    """Simple sum - commutative (baseline for comparison)."""
    def forward(self, path_embeds, path_lengths):
        mask = make_depth_mask(path_lengths)
        return (path_embeds * mask.unsqueeze(-1)).sum(dim=2)

class WeightedSumPooling(PathPooling):
    """Weighted sum with learned or fixed depth weights."""
    def __init__(self, max_depth, learnable=True):
        self.weights = nn.Parameter(torch.ones(max_depth)) if learnable else \
                       torch.tensor([0.9 ** d for d in range(max_depth)])

class RotaryPooling(PathPooling):
    """Apply rotation based on depth before summing."""
    def __init__(self, d_model, max_depth, theta_base=10.0):
        self.theta = theta_base ** (-torch.arange(0, d_model, 2) / d_model)

class GRUPooling(PathPooling):
    """Process path elements sequentially with GRU."""
    def __init__(self, d_model, num_layers=1):
        self.gru = nn.GRU(d_model, d_model, num_layers, batch_first=True)

class TransformerPooling(PathPooling):
    """Self-attention over path elements with depth positional encoding."""
    def __init__(self, d_model, num_layers=1, num_heads=4, max_depth=32):
        self.depth_embedding = nn.Embedding(max_depth, d_model)
        self.layers = nn.TransformerEncoder(...)
```

**KVPE Module Interface**:
```python
class KeyValuePositionEncoding(nn.Module):
    """
    Pluggable position encoding for JSON paths.
    """

    POOLING_CLASSES = {
        "sum": SumPooling,
        "weighted": WeightedSumPooling,
        "rotary": RotaryPooling,
        "gru": GRUPooling,
        "transformer": TransformerPooling,
    }

    def __init__(
        self,
        d_model: int,
        max_depth: int = 32,
        max_array_position: int = 256,
        pooling: str = "sum",  # MVP: sum (matches original paper)
        share_key_embeddings: bool = True,  # Share with token embeddings
        **pooling_kwargs,
    ):
        self.index_embeddings = nn.Embedding(max_array_position, d_model)
        self.pooling = self.POOLING_CLASSES[pooling](d_model, max_depth, **pooling_kwargs)

        # Key embeddings: shared or separate
        self.share_key_embeddings = share_key_embeddings
        if share_key_embeddings:
            self.key_embeddings = None  # Set externally via set_key_embeddings()
        else:
            self.key_embeddings = None  # Created during model init with correct vocab size

    def set_key_embeddings(self, key_embeddings: nn.Embedding):
        """Share key embeddings with token embedding layer (for transfer learning)."""
        assert self.share_key_embeddings, "Cannot set shared embeddings when share_key_embeddings=False"
        self.key_embeddings = key_embeddings

    def forward(
        self,
        path_types: torch.LongTensor,   # (batch, seq_len, max_depth) - 0=pad, 1=key, 2=index
        path_ids: torch.LongTensor,     # (batch, seq_len, max_depth) - key vocab ID or array index
        path_lengths: torch.LongTensor, # (batch, seq_len)
    ) -> torch.FloatTensor:             # (batch, seq_len, d_model)
        # Embed path elements by type (keys use shared or separate embeddings)
        embeds = self.embed_path_elements(path_types, path_ids)
        return self.pooling(embeds, path_lengths)
```

**Key Embeddings Strategy**:
- **Shared (default)**: Path keys use same embeddings as token keys. Enables transfer learning — model recognizes "name" in position encoding as same concept as "name" token.
- **Separate (configurable)**: Independent embeddings for position vs token. May be better for complex pooling strategies.
- MVP uses shared embeddings (simpler, matches original paper's sum approach).

**Path Tracking During Tokenization**:
```python
def tokenize_value(value, path: tuple[PathElement, ...]):
    """Recursively tokenize with path tracking."""
    if isinstance(value, dict):
        yield OBJ_START, path
        for key, v in value.items():
            yield KeyToken(key), path
            new_path = path + (KeyElement(key),)
            yield from tokenize_value(v, new_path)
        yield OBJ_END, path
        
    elif isinstance(value, list):
        yield ARRAY_START, path
        for i, item in enumerate(value):
            new_path = path + (IndexElement(i),)
            yield from tokenize_value(item, new_path)
        yield ARRAY_END, path
        
    else:
        yield ValueToken(value), path
```

---

### 3. Constraints Module (`origami/constraints/`)

#### 3.1 JSON Grammar (`json_grammar.py`)

**Purpose**: Define context-free grammar for valid JSON token sequences.

**Grammar Rules** (EBNF-style):
```
document   → START root END
root       → object | array              # MVP: object only, array support prepared
object     → OBJ_START pair* OBJ_END
pair       → KEY value
value      → PRIMITIVE | array | object
array      → ARRAY_START value* ARRAY_END
```

Note: Unlike the original paper which had implicit root object, we explicitly wrap with `OBJ_START`/`OBJ_END` for uniformity and future array support.

**State Machine for Constraint Tracking**:

The grammar is context-free, requiring a stack to track nesting. Valid next tokens depend on:
1. Current context (in object vs in array vs at root)
2. Whether we just saw a KEY (expecting value)

```
Stack symbols: ROOT, OBJ, ARRAY, AWAIT_VALUE

Transitions:
  START        → push ROOT, expect OBJ_START (or ARRAY_START for future)
  OBJ_START    → push OBJ, expect key-or-obj-end
  KEY(k)       → push AWAIT_VALUE, expect value
  PRIMITIVE    → if top is AWAIT_VALUE: pop AWAIT_VALUE, expect key-or-obj-end
                 if top is ARRAY: expect value-or-array-end
  OBJ_END      → pop OBJ, return to parent context
  ARRAY_START  → if top is AWAIT_VALUE: pop AWAIT_VALUE
                 push ARRAY, expect value-or-array-end
  ARRAY_END    → pop ARRAY, return to parent context
  END          → pop ROOT, accept (stack must be empty)

Valid next tokens by context:
  After START:       OBJ_START (MVP), ARRAY_START (future)
  After OBJ_START:   KEY, OBJ_END (empty object)
  After KEY:         PRIMITIVE, OBJ_START, ARRAY_START
  After PRIMITIVE:
    - in object:     KEY, OBJ_END
    - in array:      PRIMITIVE, OBJ_START, ARRAY_START, ARRAY_END
    - at root:       END
  After OBJ_END:
    - in object:     KEY, OBJ_END
    - in array:      PRIMITIVE, OBJ_START, ARRAY_START, ARRAY_END
    - at root:       END
  After ARRAY_START: PRIMITIVE, OBJ_START, ARRAY_START, ARRAY_END (empty array)
  After ARRAY_END:
    - in object:     KEY, OBJ_END
    - in array:      PRIMITIVE, OBJ_START, ARRAY_START, ARRAY_END
    - at root:       END
```

**Implementation Options**:

1. **Custom PDA** (recommended for training): 
   - Full control over mask generation
   - Can be vectorized for batch processing
   - Simpler than original Array(n) countdown logic

2. **Outlines/lm-format-enforcer** (for inference):
   - Battle-tested constrained decoding
   - May be overkill if we only need training masks

```python
class JSONGrammarConstraint:
    """Stack-based grammar constraint tracker."""
    
    def __init__(self, vocab: Vocabulary): ...
    
    def get_valid_tokens(self, token_ids: list[int]) -> set[int]:
        """Given token sequence so far, return valid next token IDs."""
        # Replay sequence through PDA, return valid transitions from final state
        ...
    
    def get_mask_batch(
        self, 
        token_ids: torch.LongTensor,  # (batch, seq_len)
        lengths: torch.LongTensor,     # (batch,)
    ) -> torch.BoolTensor:             # (batch, seq_len, vocab_size)
        """
        Compute validity mask for each position in batch.
        
        For training: mask[b, t, v] = True if token v is valid at position t+1
        given tokens[b, :t+1].
        """
        ...
```

#### 3.2 Constrained Loss (`constrained_loss.py`)

**Purpose**: Apply grammar constraints during training by masking invalid logits.

```python
class ConstrainedCrossEntropyLoss(nn.Module):
    def __init__(self, grammar: JSONGrammarConstraint, vocab: Vocabulary): ...
    
    def forward(
        self,
        logits: torch.FloatTensor,    # (batch, seq_len, vocab_size)
        targets: torch.LongTensor,    # (batch, seq_len)
        input_ids: torch.LongTensor,  # (batch, seq_len) for grammar state
    ) -> torch.FloatTensor:
        # 1. Get grammar mask for each position
        # 2. Set invalid logits to -inf
        # 3. Compute cross-entropy loss
        ...
```

---

### 4. Model Module (`origami/model/`)

#### 4.1 Configuration (`config.py`)

```python
@dataclass
class OrigamiConfig:
    # Vocabulary
    vocab_size: int
    max_depth: int = 32           # Maximum nesting depth for KVPE
    max_array_position: int = 256 # Maximum array index for position embeddings
    
    # Architecture
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 1024
    dropout: float = 0.1
    
    # Backbone (pluggable)
    backbone: Literal["transformer", "lstm", "mamba"] = "transformer"
    # LSTM-specific
    lstm_bidirectional: bool = False
    lstm_num_layers: int = 2
    
    # Position encoding (pluggable)
    kvpe_pooling: Literal["sum", "weighted", "rotary", "gru", "transformer"] = "sum"  # MVP: sum (original paper)
    kvpe_pooling_kwargs: dict = field(default_factory=dict)  # e.g., {"num_layers": 2} for GRU
    
    # Continuous head (optional)
    use_continuous_head: bool = False
    num_mixture_components: int = 5
    
    # Sequence limits
    max_seq_length: int = 512
    
    # Grammar constraints
    use_grammar_constraints: bool = True


@dataclass  
class TrainingConfig:
    # Optimization
    learning_rate: float = 1e-3
    batch_size: int = 32
    num_epochs: int = 100
    warmup_steps: int = 1000
    weight_decay: float = 0.01
    
    # Shuffling and upscaling (key features!)
    shuffle_keys: bool = True
    upscale_factor: int = 1  # 1 = no upscaling, >1 = create multiple permutations
    
    # Continuous loss weight (if using continuous head)
    continuous_loss_weight: float = 1.0
    
    # Checkpointing
    save_every_n_epochs: int = 10
    eval_every_n_steps: int = 500
```

#### 4.2 Embedding Layer (`embeddings.py`)

```python
class OrigamiEmbeddings(nn.Module):
    """Token embeddings + KVPE position encoding."""

    def __init__(self, config: OrigamiConfig, vocab: Vocabulary):
        # Token embeddings (shared with KVPE for key position encoding)
        self.token_embedding = nn.Embedding(vocab.size, config.d_model)

        # KVPE with shared key embeddings
        self.kvpe = KeyValuePositionEncoding(
            d_model=config.d_model,
            vocab_size=vocab.size,
            max_depth=config.max_depth,
            max_array_index=config.max_array_position,
            pooling=config.kvpe_pooling,
            share_key_embeddings=True,
            **config.kvpe_pooling_kwargs,
        )
        self.kvpe.set_key_embeddings(self.token_embedding)  # Share embeddings

        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        input_ids: torch.LongTensor,        # (batch, seq_len)
        path_types: torch.LongTensor,       # (batch, seq_len, max_depth)
        path_ids: torch.LongTensor,         # (batch, seq_len, max_depth)
        path_lengths: torch.LongTensor,     # (batch, seq_len)
    ) -> torch.FloatTensor:
        # 1. Token embeddings
        embeds = self.token_embedding(input_ids)  # (batch, seq_len, d_model)

        # 2. Add position encoding (KVPE)
        pos_embeds = self.kvpe(path_types, path_ids, path_lengths)

        return self.dropout(embeds + pos_embeds)

        # Note: NUM token scaling for continuous head added in Phase 6
```

**Numeric Value Preprocessing** (in tokenizer/data pipeline):
```python
class NumericScaler:
    """Scales high-cardinality numeric fields for continuous head."""
    
    def __init__(
        self,
        scaler_type: Literal["standard", "minmax", "robust"] = "standard",
        cardinality_threshold: int = 100,  # Only scale if > this many unique values
    ): ...
    
    def fit(self, objects: Iterable[dict]) -> NumericScaler:
        """Fit scalers on training data, one per numeric field path."""
        ...
    
    def transform(self, obj: dict) -> tuple[dict, dict[str, float]]:
        """
        Transform object, replacing high-cardinality numerics with NUM token.
        
        Returns:
            transformed_obj: Object with NUM tokens replacing scaled numerics
            numeric_values: Dict mapping field paths to scaled values
        """
        ...
```

The scaler is fit once on training data and applied consistently during tokenization.
Low-cardinality numeric fields (e.g., age buckets, ratings 1-5) remain as discrete tokens.

#### 4.3 Backbone Module (`backbone.py`)

**Design Principle**: Backbones are **pluggable** — different sequence models can be swapped for experimentation.

```python
class BackboneBase(nn.Module, ABC):
    """Abstract base for sequence modeling backbones."""
    
    @abstractmethod
    def forward(
        self,
        hidden_states: torch.FloatTensor,      # (batch, seq_len, d_model)
        attention_mask: torch.BoolTensor | None = None,  # (batch, seq_len)
    ) -> torch.FloatTensor:                    # (batch, seq_len, d_model)
        """Process sequence, return hidden states."""
        ...


class TransformerBackbone(BackboneBase):
    """
    Decoder-only transformer with causal attention.
    Standard choice for autoregressive modeling.
    """
    def __init__(self, config: OrigamiConfig):
        self.layers = nn.ModuleList([
            TransformerBlock(config.d_model, config.n_heads, config.d_ff, config.dropout)
            for _ in range(config.n_layers)
        ])
    
    def forward(self, hidden_states, attention_mask=None):
        causal_mask = make_causal_mask(hidden_states.size(1))
        for layer in self.layers:
            hidden_states = layer(hidden_states, causal_mask, attention_mask)
        return hidden_states


class LSTMBackbone(BackboneBase):
    """
    LSTM backbone for comparison with RNN-based approaches.
    Useful for ablation: does attention matter for JSON?
    """
    def __init__(self, config: OrigamiConfig):
        self.lstm = nn.LSTM(
            config.d_model,
            config.d_model,
            num_layers=config.lstm_num_layers,
            batch_first=True,
            bidirectional=config.lstm_bidirectional,
            dropout=config.dropout if config.lstm_num_layers > 1 else 0,
        )
        if config.lstm_bidirectional:
            self.proj = nn.Linear(config.d_model * 2, config.d_model)
    
    def forward(self, hidden_states, attention_mask=None):
        # Pack if variable length
        if attention_mask is not None:
            lengths = attention_mask.sum(dim=1).cpu()
            packed = pack_padded_sequence(hidden_states, lengths, 
                                          batch_first=True, enforce_sorted=False)
            output, _ = self.lstm(packed)
            hidden_states, _ = pad_packed_sequence(output, batch_first=True)
        else:
            hidden_states, _ = self.lstm(hidden_states)
        
        if hasattr(self, 'proj'):
            hidden_states = self.proj(hidden_states)
        return hidden_states


class MambaBackbone(BackboneBase):
    """
    Mamba (S4/SSM) backbone for efficient long sequences.
    Requires mamba-ssm package.
    """
    def __init__(self, config: OrigamiConfig):
        from mamba_ssm import Mamba
        self.layers = nn.ModuleList([
            Mamba(d_model=config.d_model, d_state=16, d_conv=4, expand=2)
            for _ in range(config.n_layers)
        ])
    
    def forward(self, hidden_states, attention_mask=None):
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


# Factory function
def create_backbone(config: OrigamiConfig) -> BackboneBase:
    BACKBONE_CLASSES = {
        "transformer": TransformerBackbone,
        "lstm": LSTMBackbone,
        "mamba": MambaBackbone,
    }
    return BACKBONE_CLASSES[config.backbone](config)
```

#### 4.4 Output Heads (`heads.py`)

```python
class DiscreteHead(nn.Module):
    """Standard next-token prediction head."""
    
    def forward(self, hidden: torch.FloatTensor) -> torch.FloatTensor:
        # Returns logits: (batch, seq_len, vocab_size)
        ...

class ContinuousHead(nn.Module):
    """
    Mixture of Gaussians head for continuous values.
    
    Outputs mixture parameters:
    - weights: (batch, seq_len, n_components) - softmax normalized
    - means: (batch, seq_len, n_components)
    - log_vars: (batch, seq_len, n_components)
    """
    
    def __init__(self, d_model: int, n_components: int = 5): ...
    
    def forward(self, hidden: torch.FloatTensor) -> tuple[Tensor, Tensor, Tensor]:
        ...
    
    def nll_loss(
        self,
        weights: torch.FloatTensor,
        means: torch.FloatTensor,
        log_vars: torch.FloatTensor,
        targets: torch.FloatTensor,
        mask: torch.BoolTensor,
    ) -> torch.FloatTensor:
        """Negative log-likelihood under mixture of Gaussians."""
        ...
    
    def sample(
        self,
        weights: torch.FloatTensor,
        means: torch.FloatTensor,
        log_vars: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """Sample from the mixture distribution."""
        ...
```

#### 4.5 Main Model (`origami_model.py`)

```python
class OrigamiModel(nn.Module):
    """
    Complete ORIGAMI model for JSON classification/generation.
    """
    
    def __init__(self, config: OrigamiConfig, vocab: Vocabulary): ...
    
    def forward(
        self,
        input_ids: torch.LongTensor,           # (batch, seq_len)
        path_types: torch.LongTensor,          # (batch, seq_len, max_depth) - 0=pad, 1=key, 2=index
        path_ids: torch.LongTensor,            # (batch, seq_len, max_depth) - key vocab ID or array index
        path_lengths: torch.LongTensor,        # (batch, seq_len)
        attention_mask: torch.BoolTensor | None = None,
        labels: torch.LongTensor | None = None,
        numeric_values: torch.FloatTensor | None = None,  # Phase 6: for continuous head
        numeric_labels: torch.FloatTensor | None = None,  # Phase 6: for continuous head
    ) -> OrigamiOutput:
        # 1. Embeddings
        hidden = self.embeddings(input_ids, path_types, path_ids, path_lengths)
        
        # 2. Backbone
        hidden = self.backbone(hidden, attention_mask)
        
        # 3. Discrete head
        logits = self.discrete_head(hidden)
        
        # 4. Apply grammar constraints (mask invalid tokens)
        if self.config.use_grammar_constraints:
            logits = self.apply_grammar_mask(logits, input_ids)
        
        # 5. Continuous head (if enabled)
        continuous_params = None
        if self.config.use_continuous_head:
            continuous_params = self.continuous_head(hidden)
        
        # 6. Compute loss if labels provided
        loss = None
        if labels is not None:
            loss = self.compute_loss(logits, labels, continuous_params, numeric_labels)
        
        return OrigamiOutput(
            loss=loss,
            logits=logits,
            continuous_params=continuous_params,
            hidden_states=hidden,
        )
    
    def predict(self, obj: dict, target_key: str) -> Any:
        """Predict value for target_key given other fields."""
        ...
    
    def generate(self, partial_obj: dict, max_tokens: int = 100) -> dict:
        """Auto-complete a partial JSON object."""
        ...

@dataclass
class OrigamiOutput:
    loss: torch.FloatTensor | None
    logits: torch.FloatTensor
    continuous_params: tuple[Tensor, Tensor, Tensor] | None
    hidden_states: torch.FloatTensor
```

#### 4.6 HuggingFace Integration (`hf_model.py`)

```python
class OrigamiPreTrainedModel(PreTrainedModel):
    """HuggingFace-compatible wrapper."""
    
    config_class = OrigamiConfig
    base_model_prefix = "origami"
    
    # Implement required methods for save/load compatibility
    ...

class OrigamiForClassification(OrigamiPreTrainedModel):
    """ORIGAMI configured for classification tasks."""
    ...
```

---

### 5. Training Module (`origami/training/`)

#### 5.1 Data Collator (`collator.py`)

```python
class OrigamiDataCollator:
    """Collate tokenized instances into batches."""
    
    def __init__(
        self,
        tokenizer: JSONTokenizer,
        max_length: int | None = None,
        shuffle_keys: bool = True,
        upscale_factor: int = 1,
    ): ...
    
    def __call__(self, instances: list[dict]) -> EncodedBatch:
        # 1. Tokenize each instance (with optional key shuffling)
        # 2. Pad to max length in batch
        # 3. Return EncodedBatch
        ...
```

#### 5.2 Trainer (`trainer.py`)

```python
class OrigamiTrainer:
    """
    Training loop with support for:
    - Grammar-constrained loss
    - Key-order shuffling / upscaling
    - Mixed discrete + continuous loss
    """
    
    def __init__(
        self,
        model: OrigamiModel,
        tokenizer: JSONTokenizer,
        train_data: Dataset,
        eval_data: Dataset | None = None,
        config: TrainingConfig = None,
    ): ...
    
    def train(self) -> None: ...
    def evaluate(self) -> dict[str, float]: ...
    def save(self, path: str) -> None: ...
```

---

## Data Flow

### Training Flow

```
1. Raw JSON objects
       │
       ▼
2. NumericScaler.fit() → fit sklearn scalers on numeric fields
       │
       ▼
3. JSONTokenizer.fit() → builds Vocabulary from all keys/values
       │
       ▼
4. Training Loop:
       │
       ├──▶ Sample batch of objects
       │         │
       │         ▼
       │    Upscaling: create N copies of each object
       │         │
       │         ▼
       │    Shuffling: permute key order in each copy
       │         │
       │         ▼
       │    JSONTokenizer.encode_batch() 
       │         │
       │         ▼
       │    EncodedBatch {input_ids, path_types, path_ids, ...}
       │         │
       │         ▼
       │    OrigamiModel.forward()
       │         │
       │         ├─▶ OrigamiEmbeddings (token + KVPE)
       │         │
       │         ├─▶ Backbone (Transformer/LSTM/Mamba)
       │         │
       │         ├─▶ DiscreteHead → logits
       │         │
       │         ├─▶ Grammar mask (set invalid logits to -inf)
       │         │
       │         ├─▶ ContinuousHead → (weights, means, vars) [optional]
       │         │
       │         ▼
       │    Loss = CE_loss + λ * NLL_continuous_loss
       │         │
       │         ▼
       │    Backprop, optimizer step
       │
       └──◀ Repeat for all batches
```

### Inference Flow

```
1. Partial JSON object + target key
       │
       ▼
2. Tokenize with target key value masked
       │
       ▼
3. Forward pass to get logits at target position
       │
       ▼
4. Apply grammar constraints
       │
       ▼
5. If discrete: argmax/sample from logits
   If continuous (NUM predicted): sample from MoG
       │
       ▼
6. Decode token(s) back to value
```

---

## Shuffling and Upscaling

**Core Insight**: JSON objects have unordered keys (`{a:1, b:2}` ≡ `{b:2, a:1}`), but sequence models see them in a fixed order. This creates a risk of learning spurious correlations based on key ordering rather than semantic relationships.

### Key Shuffling

During tokenization, key-value pairs at each level are randomly permuted:

```python
def tokenize_object(obj: dict, path: tuple, shuffle: bool = True):
    items = list(obj.items())
    if shuffle:
        random.shuffle(items)
    
    for key, value in items:
        yield KeyToken(key), path
        yield from tokenize_value(value, path + (KeyElement(key),))
```

**Benefits**:
- Forces model to learn from key semantics, not position in sequence
- Acts as data augmentation
- Aligns with JSON specification (keys are unordered)

**When to shuffle**:
- Training: Always (unless studying order effects)
- Validation/Test: Use a fixed seed for reproducibility
- Inference: Not applicable (single pass)

### Upscaling

Create multiple copies of each document with different shuffle permutations. Implemented at **Dataset level** for clean DataLoader integration:

```python
class UpscaledDataset(Dataset):
    """
    Dataset wrapper that presents upscaled view of base data.
    Each base item appears `upscale_factor` times with different shuffle permutations.
    """

    def __init__(
        self,
        base_data: list[dict],
        tokenizer: JSONTokenizer,
        upscale_factor: int = 10,
    ):
        self.base_data = base_data
        self.tokenizer = tokenizer
        self.upscale_factor = upscale_factor

    def __len__(self) -> int:
        return len(self.base_data) * self.upscale_factor

    def __getitem__(self, idx: int) -> TokenizedInstance:
        base_idx = idx // self.upscale_factor
        obj = self.base_data[base_idx]
        # Each access gets a fresh shuffle (different permutation each time)
        return self.tokenizer.tokenize(obj, shuffle=True)
```

**Upscaling factors by dataset size** (empirical guidance from paper):

| Dataset Size | Recommended Upscale Factor |
|--------------|---------------------------|
| < 500 | 100-1000 |
| 500-5000 | 10-100 |
| 5000-50000 | 1-10 |
| > 50000 | 1 (shuffling alone suffices) |

**Usage with DataLoader**:
```python
# Create upscaled dataset
train_dataset = UpscaledDataset(train_data, tokenizer, upscale_factor=100)

# Standard DataLoader (its shuffle will interleave different permutations)
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,  # DataLoader shuffle, not key shuffle
    collate_fn=collator,
)
```

**Epoch handling with upscaling**:
- Logical epoch = one pass through original data
- Physical epoch = `upscale_factor` passes with different permutations
- Learning rate scheduling should use logical epochs
- Dataset's `__len__` returns `base_size * upscale_factor`

### Ablation: Why This Works

From Section 4.4.1 of the paper:
- Without shuffling: Model overfits to key order, poor generalization
- With shuffling, low upscale (1-5x): Still overfits, test loss increases after initial drop
- With shuffling, high upscale (100x+): Train/test loss converge, best generalization

The mechanism is similar to dropout — the model can't rely on any fixed subset of context.

---

## File Structure

```
origami/
├── __init__.py
├── tokenizer/
│   ├── __init__.py
│   ├── vocabulary.py         # Token types + Vocabulary class
│   ├── path.py               # KeyElement, IndexElement, Path type
│   ├── json_tokenizer.py     # JSON ↔ token conversion
│   ├── tabular_tokenizer.py  # Simplified fixed-schema mode
│   └── errors.py             # DecodeError
├── preprocessing/
│   ├── __init__.py
│   ├── numeric_discretizer.py  # KBinsDiscretizer for discrete mode (MVP)
│   └── numeric_scaler.py       # StandardScaler for continuous mode (future)
├── position_encoding/
│   ├── __init__.py
│   ├── kvpe.py               # Key-Value Position Encoding main module
│   └── pooling.py            # SumPooling, RotaryPooling, GRUPooling, etc.
├── constraints/
│   ├── __init__.py
│   ├── json_grammar.py       # Stack-based PDA grammar constraint
│   └── constrained_loss.py   # Masked cross-entropy loss
├── model/
│   ├── __init__.py
│   ├── config.py             # OrigamiConfig, TrainingConfig
│   ├── embeddings.py         # Token + KVPE embeddings
│   ├── backbone.py           # Transformer, LSTM, Mamba backends
│   ├── heads.py              # DiscreteHead, ContinuousHead (MoG)
│   ├── origami_model.py      # Main model assembly
│   └── hf_model.py           # Optional HuggingFace integration
├── training/
│   ├── __init__.py
│   ├── dataset.py            # UpscaledDataset wrapper
│   ├── collator.py           # Batch collation
│   └── trainer.py            # Training loop
└── utils/
    ├── __init__.py
    └── visualization.py      # Debugging tokenization, attention, etc.

tests/
├── test_vocabulary.py
├── test_tokenizer.py
├── test_numeric_discretizer.py
├── test_kvpe.py
├── test_grammar.py
├── test_model.py
└── fixtures/
    └── sample_data.json

examples/
├── train_jsonified.py        # UCI datasets (Section 4.1)
├── train_ddxplus.py          # Medical diagnosis (Section 4.2)
├── train_codenet.py          # Code classification (Section 4.3)
└── train_tabular.py          # Simple tabular mode
```

---

## Dependencies

```toml
[project]
name = "origami"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "torch>=2.0",
    "numpy>=1.24",
    "scikit-learn>=1.3",  # For NumericScaler
    "tqdm",
]

[project.optional-dependencies]
hf = [
    "transformers>=4.30",  # For PreTrainedModel integration
]
dev = [
    "pytest>=7.0",
    "pytest-cov",
    "ruff",
    "mypy",
]
mamba = [
    "mamba-ssm",  # For Mamba backbone
]
```

**Note**: Grammar constraints are implemented via custom PDA (no external dependency).
HuggingFace integration is optional.

---

## Implementation Status

### Phase 1: Core Tokenization ✅ COMPLETE
1. ✅ `vocabulary.py` - Token types (GrammarToken, KeyToken, ValueToken) and vocabulary management
2. ✅ `json_tokenizer.py` - JSON to token conversion with unified path tracking (keys + indices)
3. ✅ Tests: tokenization round-trip, path correctness for nested structures

### Phase 2: Position Encoding ✅ COMPLETE
4. ✅ `kvpe.py` - Key-Value Position Encoding with **all 5 pooling strategies**:
   - `sum` - Commutative baseline (original paper)
   - `weighted` - Learned depth weights
   - `rotary` - Rotation by depth, parallelizable
   - `gru` - Sequential processing, fully non-commutative
   - `transformer` - Self-attention over path elements
5. ✅ Tests: verify non-commutativity (a.b ≠ b.a), array index ordering

### Phase 3: Model Core ✅ COMPLETE
6. ✅ `config.py` - Configuration dataclass with all options
7. ✅ `embeddings.py` - Token embeddings + KVPE (shared key embeddings)
8. ✅ `backbone.py` - TransformerBackbone using PyTorch built-in layers
9. ✅ `heads.py` - DiscreteHead and ContinuousHead (MoG)
10. ✅ `origami_model.py` - Assemble components, forward pass with grammar constraints
11. ✅ `encode_batch()` in JSONTokenizer - converts TokenizedInstance to tensors

### Phase 4: Grammar Constraints ✅ COMPLETE
12. ✅ `json_grammar.py` - Stack-based PDA with batch-parallel state updates
13. ✅ Grammar masking integrated into model (no separate `constrained_loss.py` needed)
14. ✅ Integration tests: verify only valid sequences can be generated

**Implementation notes:**
- Grammar constraints applied during training only (when `labels` provided)
- Incremental O(1) per-step grammar for inference via `get_next_token_mask()`
- `init_state_from_tokens()` for initializing state from prefix (handles left-padding)

### Phase 5: Training & Inference ✅ COMPLETE
15. ✅ `collator.py` - **Left-padding** for batched prediction (all sequences end at same position)
16. ✅ `trainer.py` - Training loop with LR warmup, checkpointing, upscaling support
17. ✅ `dataset.py` - UpscaledDataset for key-order permutation augmentation
18. ✅ End-to-end training script (`examples/train_jsonl.py`)

**Inference modules:**
19. ✅ `generator.py` - Autoregressive JSON generation with incremental grammar (O(n) total)
20. ✅ `predictor.py` - Field value prediction using Generator
21. ✅ `embedder.py` - Extract hidden state representations

**Architecture decisions:**
- Left-padding enables `logits[:, -1, :]` for batched next-token prediction
- Generator owns all inference grammar logic (model only masks during training)
- Generator API: `generate()` and `generate_from_tensors(stop_after_value=True/False)`
- Predictor uses Generator internally for complex value completion

### Phase 6: Extensions ⏳ PARTIAL
22. ✅ `numeric_discretizer.py` - Bins high-cardinality numerics to categorical tokens
23. ✅ `ContinuousHead` in `heads.py` - Mixture of Gaussians output (implemented)
24. ❌ `numeric_scaler.py` - sklearn-based scaling for continuous head input
25. ❌ NUM token scaling in embeddings layer
26. ❌ `tabular_tokenizer.py` - Simplified fixed-schema mode
27. ❌ `LSTMBackbone` in `backbone.py` (stub exists, raises NotImplementedError)
28. ❌ `MambaBackbone` in `backbone.py` (stub exists, raises NotImplementedError)
29. ❌ HuggingFace integration (`hf_model.py` with PreTrainedModel)

### Phase 7: Validation ❌ NOT STARTED
30. ❌ Reproduce JSONified dataset results (Section 4.1)
31. ❌ Reproduce DDXPlus results (Section 4.2)
32. ❌ Reproduce CodeNet results (Section 4.3)

---

## Design Questions (Resolved)

1. **KVPE pooling default**: ✅ RESOLVED
   - **Decision**: Default to `sum` (matches original paper)
   - All 5 strategies implemented and switchable via config

2. **Continuous head scaler integration**: ⏳ DEFERRED
   - NumericDiscretizer implemented for discrete mode
   - NumericScaler for continuous head still needed

3. **Backbone abstraction**: ✅ RESOLVED
   - **Decision**: Minimal interface `forward(hidden, mask) -> hidden`
   - TransformerBackbone implemented, LSTM/Mamba stubs ready

4. **HuggingFace integration**: ⏳ DEFERRED
   - Custom save/load implemented via `OrigamiModel.save()` / `OrigamiModel.load()`
   - Saves model config, weights, and tokenizer state together

5. **Test coverage**: ✅ RESOLVED
   - 364 tests covering all core functionality
   - Tokenization, KVPE, grammar, training, inference all tested
   - Left-padding tests added for batched prediction

6. **Vocabulary edge cases**: ✅ RESOLVED (for MVP)
   - Keep simple: no truncation/hashing, store values as-is
   - Unicode handled naturally by Python strings

7. **Padding direction**: ✅ RESOLVED (added during implementation)
   - **Decision**: Left-padding for all batches
   - Enables `logits[:, -1, :]` for batched next-token prediction
   - Critical for efficient Predictor implementation

8. **Grammar during inference**: ✅ RESOLVED (added during implementation)
   - **Decision**: Model applies grammar only during training (when labels provided)
   - Generator maintains incremental grammar state for O(n) total inference
   - Avoids O(n²) from replaying all tokens each step

---

## References

- Original ORIGAMI paper: arXiv:2412.17348
- Outlines library: https://github.com/outlines-dev/outlines
- HuggingFace PreTrainedModel: https://huggingface.co/docs/transformers/main_classes/model
