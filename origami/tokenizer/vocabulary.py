"""Token types and vocabulary management for ORIGAMI.

This module defines the token classes (GrammarToken, KeyToken, ValueToken)
and the Vocabulary class that manages bidirectional token-to-ID mapping.
"""

import pickle
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any

from .errors import VocabularyFrozenError


class TokenType(Enum):
    """Types of tokens in the ORIGAMI vocabulary."""

    GRAMMAR = auto()  # Structural tokens (START, END, OBJ_START, etc.)
    KEY = auto()  # JSON object keys
    VALUE = auto()  # JSON primitive values


@dataclass(frozen=True)
class Token:
    """Base class for all tokens. Tokens are immutable and hashable."""

    token_type: TokenType


@dataclass(frozen=True)
class GrammarToken(Token):
    """Structural tokens for JSON grammar.

    These include START, END, OBJ_START, OBJ_END, ARRAY_START, ARRAY_END,
    PAD, UNK_KEY, UNK_VALUE, and NUM.
    """

    value: str
    token_type: TokenType = field(default=TokenType.GRAMMAR, init=False)

    def __repr__(self) -> str:
        return f"GrammarToken({self.value!r})"


@dataclass(frozen=True)
class KeyToken(Token):
    """A JSON object key token.

    Example: KeyToken("name") represents the key "name" in {"name": "Alice"}.
    """

    key: str
    token_type: TokenType = field(default=TokenType.KEY, init=False)

    def __repr__(self) -> str:
        return f"KeyToken({self.key!r})"


@dataclass(frozen=True)
class ValueToken(Token):
    """A JSON primitive value token.

    The value's Python type is preserved (int, float, str, bool, None),
    so ValueToken(42) != ValueToken("42").
    """

    value: Any
    token_type: TokenType = field(default=TokenType.VALUE, init=False)

    def __repr__(self) -> str:
        return f"ValueToken({self.value!r})"


# Grammar token constants with fixed IDs
START = GrammarToken("START")  # ID: 0
END = GrammarToken("END")  # ID: 1
OBJ_START = GrammarToken("OBJ_START")  # ID: 2
OBJ_END = GrammarToken("OBJ_END")  # ID: 3
ARRAY_START = GrammarToken("ARRAY_START")  # ID: 4
ARRAY_END = GrammarToken("ARRAY_END")  # ID: 5
PAD = GrammarToken("PAD")  # ID: 6
UNK_KEY = GrammarToken("UNK_KEY")  # ID: 7
UNK_VALUE = GrammarToken("UNK_VALUE")  # ID: 8
NUM = GrammarToken("NUM")  # ID: 9

# Ordered list of grammar tokens (their index is their ID)
GRAMMAR_TOKENS = [
    START,
    END,
    OBJ_START,
    OBJ_END,
    ARRAY_START,
    ARRAY_END,
    PAD,
    UNK_KEY,
    UNK_VALUE,
    NUM,
]

# First ID for dynamic tokens (keys and values)
DYNAMIC_TOKEN_START_ID = len(GRAMMAR_TOKENS)  # 10


@dataclass(frozen=True)
class PruningStats:
    """Statistics from vocabulary pruning.

    Attributes:
        original_vocab_size: Vocabulary size before pruning.
        pruned_vocab_size: Vocabulary size after pruning.
        num_values_pruned: Number of unique values removed.
        pruned_values: List of (value, count) tuples for pruned values,
            sorted by count descending.
        value_frequency_threshold: Minimum frequency to be kept.
            Values with count < threshold were pruned.
    """

    original_vocab_size: int
    pruned_vocab_size: int
    num_values_pruned: int
    pruned_values: list[tuple[Any, int]]
    value_frequency_threshold: int

    def __repr__(self) -> str:
        return (
            f"PruningStats(pruned={self.num_values_pruned} values, "
            f"vocab: {self.original_vocab_size} -> {self.pruned_vocab_size}, "
            f"threshold={self.value_frequency_threshold})"
        )


class Vocabulary:
    """Manages bidirectional mapping between tokens and integer IDs.

    Grammar tokens have fixed IDs (0-9). Dynamic tokens (keys and values)
    are assigned IDs starting from 10, interleaved as they are added.

    The vocabulary can be frozen after building to prevent accidental
    modifications. After freezing, encode() returns UNK_KEY/UNK_VALUE
    for unknown tokens instead of raising an error.
    """

    def __init__(self):
        # Grammar tokens have fixed IDs
        self._token_to_id: dict[Token, int] = {token: i for i, token in enumerate(GRAMMAR_TOKENS)}
        self._id_to_token: dict[int, Token] = {i: token for i, token in enumerate(GRAMMAR_TOKENS)}

        # Track key and value IDs separately for type queries
        self._key_ids: set[int] = set()
        self._value_ids: set[int] = set()

        # Track value frequencies for pruning
        self._value_counts: dict[ValueToken, int] = {}

        # Next ID to assign
        self._next_id = DYNAMIC_TOKEN_START_ID

        # Freeze state
        self._frozen = False

    @property
    def frozen(self) -> bool:
        """Whether the vocabulary is frozen (no new tokens can be added)."""
        return self._frozen

    @property
    def size(self) -> int:
        """Total number of tokens in the vocabulary."""
        return len(self._token_to_id)

    def __len__(self) -> int:
        """Total number of tokens in the vocabulary."""
        return len(self._token_to_id)

    # Fixed grammar token IDs
    @property
    def start_id(self) -> int:
        return 0

    @property
    def end_id(self) -> int:
        return 1

    @property
    def obj_start_id(self) -> int:
        return 2

    @property
    def obj_end_id(self) -> int:
        return 3

    @property
    def array_start_id(self) -> int:
        return 4

    @property
    def array_end_id(self) -> int:
        return 5

    @property
    def pad_token_id(self) -> int:
        return 6

    @property
    def unk_key_id(self) -> int:
        return 7

    @property
    def unk_value_id(self) -> int:
        return 8

    @property
    def num_token_id(self) -> int:
        return 9

    def add_key(self, key: str) -> int:
        """Add a key to the vocabulary and return its ID.

        If the key already exists, returns the existing ID (idempotent).
        Raises VocabularyFrozenError if the vocabulary is frozen.
        """
        if self._frozen:
            raise VocabularyFrozenError("add_key")

        token = KeyToken(key)
        if token in self._token_to_id:
            return self._token_to_id[token]

        token_id = self._next_id
        self._next_id += 1
        self._token_to_id[token] = token_id
        self._id_to_token[token_id] = token
        self._key_ids.add(token_id)
        return token_id

    def add_value(self, value: Any) -> int:
        """Add a value to the vocabulary and return its ID.

        If the value already exists, returns the existing ID (idempotent).
        Raises VocabularyFrozenError if the vocabulary is frozen.

        Note: Frequency is tracked for every call, even for existing values.
        This is used for vocabulary pruning with max_vocab_size.
        """
        if self._frozen:
            raise VocabularyFrozenError("add_value")

        token = ValueToken(value)

        # Track frequency for every occurrence (used for pruning)
        self._value_counts[token] = self._value_counts.get(token, 0) + 1

        if token in self._token_to_id:
            return self._token_to_id[token]

        token_id = self._next_id
        self._next_id += 1
        self._token_to_id[token] = token_id
        self._id_to_token[token_id] = token
        self._value_ids.add(token_id)
        return token_id

    def freeze(self) -> None:
        """Freeze the vocabulary, preventing further additions."""
        self._frozen = True

    def encode(self, token: Token) -> int:
        """Encode a token to its integer ID.

        If the vocabulary is frozen and the token is unknown:
        - Returns unk_key_id for KeyToken
        - Returns unk_value_id for ValueToken

        Raises KeyError for unknown tokens if not frozen.
        """
        if token in self._token_to_id:
            return self._token_to_id[token]

        if self._frozen:
            if isinstance(token, KeyToken):
                return self.unk_key_id
            elif isinstance(token, ValueToken):
                return self.unk_value_id

        raise KeyError(f"Unknown token: {token}")

    def decode(self, token_id: int) -> Token:
        """Decode an integer ID to its token.

        Raises KeyError if the ID is not in the vocabulary.
        """
        if token_id not in self._id_to_token:
            raise KeyError(f"Unknown token ID: {token_id}")
        return self._id_to_token[token_id]

    def is_grammar_token(self, token_id: int) -> bool:
        """Check if a token ID corresponds to a grammar token."""
        return 0 <= token_id < DYNAMIC_TOKEN_START_ID

    def is_key_token(self, token_id: int) -> bool:
        """Check if a token ID corresponds to a key token (including UNK_KEY)."""
        return token_id in self._key_ids or token_id == self.unk_key_id

    def is_value_token(self, token_id: int) -> bool:
        """Check if a token ID corresponds to a value token (including UNK_VALUE, NUM)."""
        return (
            token_id in self._value_ids
            or token_id == self.unk_value_id
            or token_id == self.num_token_id
        )

    def get_all_key_ids(self) -> set[int]:
        """Get all key token IDs (including UNK_KEY).

        Useful for grammar constraint masks.
        """
        return self._key_ids | {self.unk_key_id}

    def get_all_primitive_value_ids(self) -> set[int]:
        """Get all primitive value token IDs (including UNK_VALUE, NUM).

        Useful for grammar constraint masks. Does not include OBJ_START
        or ARRAY_START (complex value starters).
        """
        return self._value_ids | {self.unk_value_id, self.num_token_id}

    def get_value_frequencies(self) -> dict[Any, int]:
        """Get frequency counts for all values.

        Returns:
            Dict mapping value -> count.
        """
        return {token.value: count for token, count in self._value_counts.items()}

    def get_most_common_values(self, n: int = 10) -> list[tuple[Any, int]]:
        """Get the n most common values.

        Returns:
            List of (value, count) tuples sorted by count descending.
        """
        sorted_items = sorted(
            self._value_counts.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        return [(token.value, count) for token, count in sorted_items[:n]]

    def prune_to_size(self, max_vocab_size: int) -> PruningStats:
        """Prune rare ValueTokens to fit within max_vocab_size.

        Keeps all grammar tokens (IDs 0-9) and all KeyTokens.
        Prunes least-frequent ValueTokens, which will then encode to UNK_VALUE.

        Args:
            max_vocab_size: Target maximum vocabulary size. Must be large enough
                to fit all grammar tokens + all keys.

        Returns:
            PruningStats with details of what was pruned.

        Raises:
            VocabularyFrozenError: If the vocabulary is already frozen.
            ValueError: If max_vocab_size is too small for grammar + keys.
        """
        if self._frozen:
            raise VocabularyFrozenError("prune_to_size")

        # Calculate minimum required size
        num_grammar = DYNAMIC_TOKEN_START_ID  # 10
        num_keys = len(self._key_ids)
        min_required = num_grammar + num_keys

        if max_vocab_size < min_required:
            raise ValueError(
                f"max_vocab_size ({max_vocab_size}) is too small. "
                f"Need at least {min_required} for grammar ({num_grammar}) + keys ({num_keys})."
            )

        # How many value slots do we have?
        max_values = max_vocab_size - min_required
        current_values = len(self._value_ids)
        original_size = self.size

        if current_values <= max_values:
            # No pruning needed
            return PruningStats(
                original_vocab_size=original_size,
                pruned_vocab_size=original_size,
                num_values_pruned=0,
                pruned_values=[],
                value_frequency_threshold=0,
            )

        # Need to prune: sort values by frequency, keep top max_values
        sorted_values = sorted(
            self._value_counts.items(),
            key=lambda x: x[1],
            reverse=True,  # Most frequent first
        )

        kept_values = sorted_values[:max_values]
        pruned_values = sorted_values[max_values:]

        # Frequency threshold is the count of the lowest kept value
        threshold = kept_values[-1][1] if kept_values else 0

        # Rebuild vocabulary with contiguous IDs
        self._rebuild_with_kept_values(kept_values)

        return PruningStats(
            original_vocab_size=original_size,
            pruned_vocab_size=self.size,
            num_values_pruned=len(pruned_values),
            pruned_values=[(token.value, count) for token, count in pruned_values],
            value_frequency_threshold=threshold,
        )

    def _rebuild_with_kept_values(self, kept_values: list[tuple[ValueToken, int]]) -> None:
        """Rebuild vocabulary keeping only specified values.

        Reassigns contiguous IDs: grammar tokens (0-9) -> keys -> kept values.
        """
        # Collect all KeyTokens (preserve original order by ID)
        keys_by_id = sorted(
            [(token_id, self._id_to_token[token_id]) for token_id in self._key_ids],
            key=lambda x: x[0],
        )

        # Reset to fresh state (grammar tokens only)
        self._token_to_id = {token: i for i, token in enumerate(GRAMMAR_TOKENS)}
        self._id_to_token = {i: token for i, token in enumerate(GRAMMAR_TOKENS)}
        self._key_ids = set()
        self._value_ids = set()
        self._next_id = DYNAMIC_TOKEN_START_ID

        # Re-add all keys (maintains original relative order)
        for _, key_token in keys_by_id:
            token_id = self._next_id
            self._next_id += 1
            self._token_to_id[key_token] = token_id
            self._id_to_token[token_id] = key_token
            self._key_ids.add(token_id)

        # Re-add kept values (most frequent first)
        new_value_counts: dict[ValueToken, int] = {}
        for value_token, count in kept_values:
            token_id = self._next_id
            self._next_id += 1
            self._token_to_id[value_token] = token_id
            self._id_to_token[token_id] = value_token
            self._value_ids.add(token_id)
            new_value_counts[value_token] = count

        self._value_counts = new_value_counts

    def to_dict(self) -> dict:
        """Serialize vocabulary to a dictionary.

        Returns:
            Dictionary containing all vocabulary state.
        """
        return {
            "token_to_id": self._token_to_id,
            "id_to_token": self._id_to_token,
            "key_ids": self._key_ids,
            "value_ids": self._value_ids,
            "value_counts": self._value_counts,
            "next_id": self._next_id,
            "frozen": self._frozen,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Vocabulary":
        """Reconstruct vocabulary from a dictionary.

        Args:
            data: Dictionary from to_dict()

        Returns:
            Reconstructed Vocabulary instance.
        """
        vocab = cls()
        vocab._token_to_id = data["token_to_id"]
        vocab._id_to_token = data["id_to_token"]
        vocab._key_ids = data["key_ids"]
        vocab._value_ids = data["value_ids"]
        vocab._value_counts = data.get("value_counts", {})  # Backwards compat
        vocab._next_id = data["next_id"]
        vocab._frozen = data["frozen"]
        return vocab

    def save(self, path: str | Path) -> None:
        """Save the vocabulary to a file using pickle."""
        path = Path(path)
        with path.open("wb") as f:
            pickle.dump(self.to_dict(), f)

    @classmethod
    def load(cls, path: str | Path) -> "Vocabulary":
        """Load a vocabulary from a pickle file."""
        path = Path(path)
        with path.open("rb") as f:
            data = pickle.load(f)
        return cls.from_dict(data)

    def __repr__(self) -> str:
        return (
            f"Vocabulary(size={self.size}, "
            f"keys={len(self._key_ids)}, "
            f"values={len(self._value_ids)}, "
            f"frozen={self._frozen})"
        )
