"""JSON tokenizer for ORIGAMI.

Converts JSON objects to/from token sequences with path tracking.
Supports key-order shuffling for data augmentation during training.
"""

import pickle
import random
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path as FilePath
from typing import Any

import torch
from torch import Tensor

from origami.preprocessing.numeric_scaler import ScaledNumeric

from .errors import DecodeError
from .path import IndexElement, KeyElement, Path
from .vocabulary import (
    ARRAY_END,
    ARRAY_START,
    END,
    NUM,
    OBJ_END,
    OBJ_START,
    START,
    KeyToken,
    PruningStats,
    Token,
    ValueToken,
    Vocabulary,
)


@dataclass
class TokenizedInstance:
    """A tokenized JSON object with path information.

    Attributes:
        tokens: List of tokens representing the JSON structure.
        paths: Path for each token (where it is in the JSON hierarchy).
        numeric_values: Original numeric values for NUM tokens (future use).
    """

    tokens: list[Token]
    paths: list[Path]
    numeric_values: list[float | None]

    def __len__(self) -> int:
        return len(self.tokens)


@dataclass
class EncodedBatch:
    """A batch of encoded JSON objects ready for model input.

    All tensors are padded to the maximum sequence length in the batch.

    Attributes:
        input_ids: Token IDs of shape (batch, seq_len)
        path_types: Path element types (0=pad, 1=key, 2=index)
            of shape (batch, seq_len, max_depth)
        path_ids: Path element IDs (vocab ID for keys, index for arrays)
            of shape (batch, seq_len, max_depth)
        path_lengths: Path depths of shape (batch, seq_len)
        attention_mask: Boolean mask where True = valid position,
            shape (batch, seq_len)
        numeric_values: Scaled numeric values for NUM tokens,
            shape (batch, seq_len)
        numeric_mask: Boolean mask for NUM token positions,
            shape (batch, seq_len)
        lengths: Sequence lengths before padding, shape (batch,)
        labels: Target token IDs for training, shape (batch, seq_len).
            Only present when include_labels=True during collation.
        grammar_mask: Pre-computed grammar validity mask, shape (batch, seq_len, vocab_size).
            Only present when grammar is computed during collation for parallel processing.
    """

    input_ids: Tensor
    path_types: Tensor
    path_ids: Tensor
    path_lengths: Tensor
    attention_mask: Tensor
    numeric_values: Tensor
    numeric_mask: Tensor
    lengths: Tensor
    labels: Tensor | None = None
    grammar_mask: Tensor | None = None

    def to(self, device: torch.device) -> "EncodedBatch":
        """Move all tensors to the specified device."""
        return EncodedBatch(
            input_ids=self.input_ids.to(device),
            path_types=self.path_types.to(device),
            path_ids=self.path_ids.to(device),
            path_lengths=self.path_lengths.to(device),
            attention_mask=self.attention_mask.to(device),
            numeric_values=self.numeric_values.to(device),
            numeric_mask=self.numeric_mask.to(device),
            lengths=self.lengths.to(device),
            labels=self.labels.to(device) if self.labels is not None else None,
            grammar_mask=self.grammar_mask.to(device) if self.grammar_mask is not None else None,
        )


class JSONTokenizer:
    """Tokenizes JSON objects to/from token sequences.

    The tokenizer maintains a vocabulary of keys and values seen during fit().
    Tokenization produces a sequence like:
        START OBJ_START Key("name") "Alice" Key("age") 30 OBJ_END END

    Each token is associated with a path indicating its position in the
    JSON hierarchy, used for position encoding (KVPE).
    """

    def __init__(
        self,
        vocab: Vocabulary | None = None,
        max_depth: int = 32,
        max_array_index: int = 256,
    ):
        """Initialize the tokenizer.

        Args:
            vocab: Existing vocabulary to use. If None, creates a new one.
            max_depth: Maximum nesting depth (for validation).
            max_array_index: Maximum array index (for validation).
        """
        self.vocab = vocab if vocab is not None else Vocabulary()
        self.max_depth = max_depth
        self.max_array_index = max_array_index
        self._pruning_stats: PruningStats | None = None

    def fit(self, objects: Iterable[dict], max_vocab_size: int = 0) -> "JSONTokenizer":
        """Build vocabulary from a collection of JSON objects.

        Iterates through all objects, extracting keys and values to build
        the vocabulary. After fitting, the vocabulary is frozen so that
        unknown tokens at inference time are mapped to UNK_KEY/UNK_VALUE.

        If called on an already-fitted tokenizer, creates a fresh vocabulary.

        Does NOT handle numeric binning - that should be done in a
        preprocessing step.

        Args:
            objects: Iterable of JSON-like dictionaries.
            max_vocab_size: Maximum vocabulary size. 0 = unlimited.
                If set, prunes rare values after building vocabulary.

        Returns:
            self, for method chaining.

        Raises:
            ValueError: If max_vocab_size is too small for all grammar + keys.
        """
        # Create fresh vocabulary if re-fitting
        if self.vocab.frozen:
            self.vocab = Vocabulary()
            self._pruning_stats = None

        for obj in objects:
            self._fit_value(obj)

        # Prune vocabulary if max_vocab_size is set
        if max_vocab_size > 0:
            self._pruning_stats = self.vocab.prune_to_size(max_vocab_size)
        else:
            self._pruning_stats = None

        # Freeze vocabulary so unknown tokens map to UNK at inference time
        self.vocab.freeze()
        return self

    @property
    def pruning_stats(self) -> PruningStats | None:
        """Statistics from vocabulary pruning, if max_vocab_size was used."""
        return self._pruning_stats

    def _fit_value(self, value: Any) -> None:
        """Recursively add all keys and values from a JSON value."""
        if isinstance(value, dict):
            for k, v in value.items():
                self.vocab.add_key(k)
                self._fit_value(v)
        elif isinstance(value, list):
            for item in value:
                self._fit_value(item)
        elif isinstance(value, ScaledNumeric):
            # ScaledNumeric uses the built-in NUM token, no need to add to vocab
            pass
        else:
            # Add primitive values (str, int, float, bool, None)
            # None maps to JSON null
            self.vocab.add_value(value)

    def tokenize(self, obj: dict, shuffle: bool = False) -> TokenizedInstance:
        """Convert a JSON object to a token sequence with paths.

        Args:
            obj: JSON-like dictionary to tokenize.
            shuffle: If True, randomly permute key order at each level.
                     Use True for training, False for inference/evaluation.

        Returns:
            TokenizedInstance with tokens, paths, and numeric_values.
        """
        tokens: list[Token] = []
        paths: list[Path] = []
        numeric_values: list[float | None] = []

        # START token
        tokens.append(START)
        paths.append(())
        numeric_values.append(None)

        # Root object
        self._tokenize_value(obj, (), tokens, paths, numeric_values, shuffle)

        # END token
        tokens.append(END)
        paths.append(())
        numeric_values.append(None)

        return TokenizedInstance(tokens=tokens, paths=paths, numeric_values=numeric_values)

    def _tokenize_value(
        self,
        value: Any,
        path: Path,
        tokens: list[Token],
        paths: list[Path],
        numeric_values: list[float | None],
        shuffle: bool,
    ) -> None:
        """Recursively tokenize a JSON value."""
        if isinstance(value, dict):
            # OBJ_START
            tokens.append(OBJ_START)
            paths.append(path)
            numeric_values.append(None)

            # Key-value pairs (optionally shuffled)
            items = list(value.items())
            if shuffle:
                random.shuffle(items)

            for key, val in items:
                # Key token (path is the containing object's path)
                tokens.append(KeyToken(key))
                paths.append(path)
                numeric_values.append(None)

                # Value (path includes this key)
                new_path = path + (KeyElement(key),)
                self._tokenize_value(val, new_path, tokens, paths, numeric_values, shuffle)

            # OBJ_END
            tokens.append(OBJ_END)
            paths.append(path)
            numeric_values.append(None)

        elif isinstance(value, list):
            # ARRAY_START
            tokens.append(ARRAY_START)
            paths.append(path)
            numeric_values.append(None)

            # Array elements
            for i, item in enumerate(value):
                new_path = path + (IndexElement(i),)
                self._tokenize_value(item, new_path, tokens, paths, numeric_values, shuffle)

            # ARRAY_END
            tokens.append(ARRAY_END)
            paths.append(path)
            numeric_values.append(None)

        elif isinstance(value, ScaledNumeric):
            # Scaled numeric value - emit NUM token with the scaled value
            tokens.append(NUM)
            paths.append(path)
            numeric_values.append(value.value)

        else:
            # Primitive value (str, int, float, bool, None)
            tokens.append(ValueToken(value))
            paths.append(path)
            numeric_values.append(None)

    def decode(self, token_ids: list[int]) -> dict:
        """Reconstruct a JSON object from a token sequence.

        Args:
            token_ids: List of token IDs to decode.

        Returns:
            The reconstructed JSON object.

        Raises:
            DecodeError: If the token sequence is invalid or malformed.
        """
        if not token_ids:
            raise DecodeError("Empty token sequence", token_ids, 0)

        pos = 0

        # Expect START
        if token_ids[pos] != self.vocab.start_id:
            raise DecodeError(f"Expected START token, got {token_ids[pos]}", token_ids, pos)
        pos += 1

        # Expect OBJ_START for root object
        if pos >= len(token_ids) or token_ids[pos] != self.vocab.obj_start_id:
            raise DecodeError("Expected OBJ_START token after START", token_ids, pos)

        # Decode root value
        result, pos = self._decode_value(token_ids, pos)

        # Expect END
        if pos >= len(token_ids) or token_ids[pos] != self.vocab.end_id:
            raise DecodeError(
                f"Expected END token, got {token_ids[pos] if pos < len(token_ids) else 'EOF'}",
                token_ids,
                pos,
            )

        return result

    def _decode_value(self, token_ids: list[int], pos: int) -> tuple[Any, int]:
        """Decode a value starting at position pos.

        Returns:
            Tuple of (decoded_value, new_position).
        """
        if pos >= len(token_ids):
            raise DecodeError("Unexpected end of sequence", token_ids, pos)

        token_id = token_ids[pos]

        if token_id == self.vocab.obj_start_id:
            return self._decode_object(token_ids, pos)
        elif token_id == self.vocab.array_start_id:
            return self._decode_array(token_ids, pos)
        else:
            # Primitive value
            token = self.vocab.decode(token_id)
            if isinstance(token, ValueToken):
                return token.value, pos + 1
            else:
                raise DecodeError(f"Expected value token, got {token}", token_ids, pos)

    def _decode_object(self, token_ids: list[int], pos: int) -> tuple[dict, int]:
        """Decode an object starting at OBJ_START."""
        if token_ids[pos] != self.vocab.obj_start_id:
            raise DecodeError("Expected OBJ_START", token_ids, pos)
        pos += 1

        result: dict[str, Any] = {}

        while pos < len(token_ids):
            token_id = token_ids[pos]

            if token_id == self.vocab.obj_end_id:
                return result, pos + 1

            # Expect a key
            token = self.vocab.decode(token_id)
            if not isinstance(token, KeyToken):
                raise DecodeError(f"Expected key token, got {token}", token_ids, pos)
            key = token.key
            pos += 1

            # Decode the value
            value, pos = self._decode_value(token_ids, pos)
            result[key] = value

        raise DecodeError("Unterminated object (missing OBJ_END)", token_ids, pos)

    def _decode_array(self, token_ids: list[int], pos: int) -> tuple[list, int]:
        """Decode an array starting at ARRAY_START."""
        if token_ids[pos] != self.vocab.array_start_id:
            raise DecodeError("Expected ARRAY_START", token_ids, pos)
        pos += 1

        result: list[Any] = []

        while pos < len(token_ids):
            token_id = token_ids[pos]

            if token_id == self.vocab.array_end_id:
                return result, pos + 1

            # Decode the next value
            value, pos = self._decode_value(token_ids, pos)
            result.append(value)

        raise DecodeError("Unterminated array (missing ARRAY_END)", token_ids, pos)

    def encode_tokens(self, instance: TokenizedInstance) -> list[int]:
        """Convert tokens to integer IDs.

        Args:
            instance: TokenizedInstance to encode.

        Returns:
            List of token IDs.
        """
        return [self.vocab.encode(token) for token in instance.tokens]

    def save(self, path: str | FilePath) -> None:
        """Save the tokenizer (including vocabulary) to a file."""
        path = FilePath(path)
        with path.open("wb") as f:
            pickle.dump(
                {
                    "vocab": self.vocab,
                    "max_depth": self.max_depth,
                    "max_array_index": self.max_array_index,
                    "pruning_stats": self._pruning_stats,
                },
                f,
            )

    @classmethod
    def load(cls, path: str | FilePath) -> "JSONTokenizer":
        """Load a tokenizer from a file."""
        path = FilePath(path)
        with path.open("rb") as f:
            data = pickle.load(f)

        tokenizer = cls(
            vocab=data["vocab"],
            max_depth=data["max_depth"],
            max_array_index=data["max_array_index"],
        )
        tokenizer._pruning_stats = data.get("pruning_stats")  # Backwards compat
        return tokenizer

    def __repr__(self) -> str:
        return f"JSONTokenizer(vocab={self.vocab})"
