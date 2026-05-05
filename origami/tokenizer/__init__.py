"""ORIGAMI tokenization module.

This module provides tokenization for JSON objects, converting them to
sequences of tokens with path information for position encoding.
"""

from .errors import DecodeError, VocabularyFrozenError
from .json_tokenizer import EncodedBatch, JSONTokenizer, TokenizedInstance
from .path import IndexElement, KeyElement, Path, PathElement, path_to_string
from .vocabulary import (
    ARRAY_END,
    ARRAY_START,
    END,
    GRAMMAR_TOKENS,
    NUM,
    OBJ_END,
    OBJ_START,
    PAD,
    START,
    UNK_KEY,
    UNK_VALUE,
    GrammarToken,
    KeyToken,
    PruningStats,
    Token,
    TokenType,
    ValueToken,
    Vocabulary,
)

__all__ = [
    # Errors
    "DecodeError",
    "VocabularyFrozenError",
    # Tokenizer
    "JSONTokenizer",
    "TokenizedInstance",
    "EncodedBatch",
    # Path
    "KeyElement",
    "IndexElement",
    "Path",
    "PathElement",
    "path_to_string",
    # Vocabulary and tokens
    "Vocabulary",
    "PruningStats",
    "Token",
    "TokenType",
    "GrammarToken",
    "KeyToken",
    "ValueToken",
    # Grammar token constants
    "START",
    "END",
    "OBJ_START",
    "OBJ_END",
    "ARRAY_START",
    "ARRAY_END",
    "PAD",
    "UNK_KEY",
    "UNK_VALUE",
    "NUM",
    "GRAMMAR_TOKENS",
]
