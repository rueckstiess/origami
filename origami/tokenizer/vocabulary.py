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
        """
        if self._frozen:
            raise VocabularyFrozenError("add_value")

        token = ValueToken(value)
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
