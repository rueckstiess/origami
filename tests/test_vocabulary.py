"""Tests for the vocabulary module."""

import tempfile
from pathlib import Path

import pytest

from origami.tokenizer import (
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
    TokenType,
    ValueToken,
    Vocabulary,
    VocabularyFrozenError,
)


class TestTokenTypes:
    """Tests for token dataclasses."""

    def test_grammar_token_creation(self):
        """GrammarToken should have GRAMMAR type."""
        token = GrammarToken("TEST")
        assert token.token_type == TokenType.GRAMMAR
        assert token.value == "TEST"

    def test_key_token_creation(self):
        """KeyToken should have KEY type."""
        token = KeyToken("name")
        assert token.token_type == TokenType.KEY
        assert token.key == "name"

    def test_value_token_creation(self):
        """ValueToken should have VALUE type and preserve value."""
        token = ValueToken(42)
        assert token.token_type == TokenType.VALUE
        assert token.value == 42

    def test_tokens_are_frozen(self):
        """Tokens should be immutable (frozen dataclasses)."""
        token = KeyToken("test")
        with pytest.raises(AttributeError):
            token.key = "other"

    def test_token_hashing(self):
        """Tokens should be hashable for use in dicts/sets."""
        token1 = KeyToken("name")
        token2 = KeyToken("name")
        token3 = KeyToken("other")

        assert hash(token1) == hash(token2)
        assert token1 == token2
        assert token1 != token3

        # Can be used in sets
        token_set = {token1, token2, token3}
        assert len(token_set) == 2

    def test_value_token_type_preservation(self):
        """ValueToken(42) should not equal ValueToken("42")."""
        int_token = ValueToken(42)
        str_token = ValueToken("42")
        float_token = ValueToken(42.0)

        # String vs numeric types are different
        assert int_token != str_token
        assert str_token != float_token
        # Note: int_token == float_token due to Python's 42 == 42.0

    def test_grammar_token_constants(self):
        """Grammar token constants should have correct values."""
        assert START.value == "START"
        assert END.value == "END"
        assert OBJ_START.value == "OBJ_START"
        assert OBJ_END.value == "OBJ_END"
        assert ARRAY_START.value == "ARRAY_START"
        assert ARRAY_END.value == "ARRAY_END"
        assert PAD.value == "PAD"
        assert UNK_KEY.value == "UNK_KEY"
        assert UNK_VALUE.value == "UNK_VALUE"
        assert NUM.value == "NUM"

    def test_grammar_tokens_list_order(self):
        """GRAMMAR_TOKENS should have tokens in ID order."""
        assert len(GRAMMAR_TOKENS) == 10
        assert GRAMMAR_TOKENS[0] == START
        assert GRAMMAR_TOKENS[1] == END
        assert GRAMMAR_TOKENS[2] == OBJ_START
        assert GRAMMAR_TOKENS[3] == OBJ_END
        assert GRAMMAR_TOKENS[4] == ARRAY_START
        assert GRAMMAR_TOKENS[5] == ARRAY_END
        assert GRAMMAR_TOKENS[6] == PAD
        assert GRAMMAR_TOKENS[7] == UNK_KEY
        assert GRAMMAR_TOKENS[8] == UNK_VALUE
        assert GRAMMAR_TOKENS[9] == NUM


class TestVocabulary:
    """Tests for the Vocabulary class."""

    def test_empty_vocabulary(self):
        """New vocabulary should have only grammar tokens."""
        vocab = Vocabulary()
        assert vocab.size == 10  # Grammar tokens only
        assert not vocab.frozen

    def test_grammar_token_ids(self):
        """Grammar tokens should have fixed IDs 0-9."""
        vocab = Vocabulary()
        assert vocab.start_id == 0
        assert vocab.end_id == 1
        assert vocab.obj_start_id == 2
        assert vocab.obj_end_id == 3
        assert vocab.array_start_id == 4
        assert vocab.array_end_id == 5
        assert vocab.pad_token_id == 6
        assert vocab.unk_key_id == 7
        assert vocab.unk_value_id == 8
        assert vocab.num_token_id == 9

    def test_add_key(self):
        """Adding keys should assign IDs starting from 10."""
        vocab = Vocabulary()
        id1 = vocab.add_key("name")
        id2 = vocab.add_key("age")
        id3 = vocab.add_key("name")  # Duplicate

        assert id1 == 10
        assert id2 == 11
        assert id3 == 10  # Same ID for duplicate
        assert vocab.size == 12

    def test_add_value(self):
        """Adding values should assign IDs sequentially."""
        vocab = Vocabulary()
        id1 = vocab.add_value("Alice")
        id2 = vocab.add_value(30)
        id3 = vocab.add_value("Alice")  # Duplicate

        assert id1 == 10
        assert id2 == 11
        assert id3 == 10  # Same ID for duplicate
        assert vocab.size == 12

    def test_interleaved_keys_and_values(self):
        """Keys and values should be interleaved in ID assignment."""
        vocab = Vocabulary()
        key_id = vocab.add_key("name")
        val_id = vocab.add_value("Alice")
        key2_id = vocab.add_key("age")
        val2_id = vocab.add_value(30)

        assert key_id == 10
        assert val_id == 11
        assert key2_id == 12
        assert val2_id == 13

    def test_encode_grammar_tokens(self):
        """Encoding grammar tokens should return fixed IDs."""
        vocab = Vocabulary()
        assert vocab.encode(START) == 0
        assert vocab.encode(END) == 1
        assert vocab.encode(OBJ_START) == 2
        assert vocab.encode(OBJ_END) == 3

    def test_encode_key_token(self):
        """Encoding key tokens should return their assigned IDs."""
        vocab = Vocabulary()
        vocab.add_key("name")
        assert vocab.encode(KeyToken("name")) == 10

    def test_encode_value_token(self):
        """Encoding value tokens should return their assigned IDs."""
        vocab = Vocabulary()
        vocab.add_value("Alice")
        assert vocab.encode(ValueToken("Alice")) == 10

    def test_encode_unknown_token_raises(self):
        """Encoding unknown token in unfrozen vocab should raise KeyError."""
        vocab = Vocabulary()
        with pytest.raises(KeyError):
            vocab.encode(KeyToken("unknown"))

    def test_decode_grammar_tokens(self):
        """Decoding grammar token IDs should return the tokens."""
        vocab = Vocabulary()
        assert vocab.decode(0) == START
        assert vocab.decode(1) == END
        assert vocab.decode(6) == PAD

    def test_decode_key_value_tokens(self):
        """Decoding key/value IDs should return the original tokens."""
        vocab = Vocabulary()
        vocab.add_key("name")
        vocab.add_value("Alice")

        assert vocab.decode(10) == KeyToken("name")
        assert vocab.decode(11) == ValueToken("Alice")

    def test_decode_unknown_id_raises(self):
        """Decoding unknown ID should raise KeyError."""
        vocab = Vocabulary()
        with pytest.raises(KeyError):
            vocab.decode(999)


class TestVocabularyFreezing:
    """Tests for vocabulary freeze behavior."""

    def test_freeze_prevents_add_key(self):
        """Frozen vocabulary should reject add_key."""
        vocab = Vocabulary()
        vocab.freeze()
        with pytest.raises(VocabularyFrozenError):
            vocab.add_key("new_key")

    def test_freeze_prevents_add_value(self):
        """Frozen vocabulary should reject add_value."""
        vocab = Vocabulary()
        vocab.freeze()
        with pytest.raises(VocabularyFrozenError):
            vocab.add_value("new_value")

    def test_frozen_encode_unknown_key_returns_unk(self):
        """Frozen vocab should return UNK_KEY for unknown keys."""
        vocab = Vocabulary()
        vocab.add_key("known")
        vocab.freeze()

        assert vocab.encode(KeyToken("unknown")) == vocab.unk_key_id

    def test_frozen_encode_unknown_value_returns_unk(self):
        """Frozen vocab should return UNK_VALUE for unknown values."""
        vocab = Vocabulary()
        vocab.add_value("known")
        vocab.freeze()

        assert vocab.encode(ValueToken("unknown")) == vocab.unk_value_id

    def test_frozen_encode_known_tokens_still_works(self):
        """Frozen vocab should still encode known tokens."""
        vocab = Vocabulary()
        vocab.add_key("name")
        vocab.add_value("Alice")
        vocab.freeze()

        assert vocab.encode(KeyToken("name")) == 10
        assert vocab.encode(ValueToken("Alice")) == 11


class TestVocabularyTypeQueries:
    """Tests for token type query methods."""

    def test_is_grammar_token(self):
        """is_grammar_token should identify grammar token IDs."""
        vocab = Vocabulary()
        vocab.add_key("name")
        vocab.add_value("Alice")

        # Grammar tokens (0-9)
        for i in range(10):
            assert vocab.is_grammar_token(i)

        # Dynamic tokens (10+)
        assert not vocab.is_grammar_token(10)
        assert not vocab.is_grammar_token(11)

    def test_is_key_token(self):
        """is_key_token should identify key token IDs."""
        vocab = Vocabulary()
        vocab.add_key("name")
        vocab.add_value("Alice")

        assert vocab.is_key_token(10)  # "name"
        assert not vocab.is_key_token(11)  # "Alice" is a value
        assert vocab.is_key_token(vocab.unk_key_id)  # UNK_KEY counts

    def test_is_value_token(self):
        """is_value_token should identify value token IDs."""
        vocab = Vocabulary()
        vocab.add_key("name")
        vocab.add_value("Alice")

        assert not vocab.is_value_token(10)  # "name" is a key
        assert vocab.is_value_token(11)  # "Alice"
        assert vocab.is_value_token(vocab.unk_value_id)  # UNK_VALUE counts
        assert vocab.is_value_token(vocab.num_token_id)  # NUM counts

    def test_get_all_key_ids(self):
        """get_all_key_ids should return all key IDs including UNK_KEY."""
        vocab = Vocabulary()
        vocab.add_key("name")
        vocab.add_key("age")
        vocab.add_value("Alice")

        key_ids = vocab.get_all_key_ids()
        assert 10 in key_ids  # "name"
        assert 11 in key_ids  # "age"
        assert 12 not in key_ids  # "Alice" is a value
        assert vocab.unk_key_id in key_ids

    def test_get_all_primitive_value_ids(self):
        """get_all_primitive_value_ids should return value IDs including UNK_VALUE and NUM."""
        vocab = Vocabulary()
        vocab.add_key("name")
        vocab.add_value("Alice")
        vocab.add_value(30)

        value_ids = vocab.get_all_primitive_value_ids()
        assert 10 not in value_ids  # "name" is a key
        assert 11 in value_ids  # "Alice"
        assert 12 in value_ids  # 30
        assert vocab.unk_value_id in value_ids
        assert vocab.num_token_id in value_ids


class TestVocabularySerialization:
    """Tests for vocabulary save/load."""

    def test_save_load_roundtrip(self):
        """Saved vocabulary should be identical when loaded."""
        vocab = Vocabulary()
        vocab.add_key("name")
        vocab.add_key("age")
        vocab.add_value("Alice")
        vocab.add_value(30)
        vocab.freeze()

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = Path(f.name)

        try:
            vocab.save(path)
            loaded = Vocabulary.load(path)

            # Check state is preserved
            assert loaded.size == vocab.size
            assert loaded.frozen == vocab.frozen

            # Check encoding is preserved
            assert loaded.encode(KeyToken("name")) == 10
            assert loaded.encode(KeyToken("age")) == 11
            assert loaded.encode(ValueToken("Alice")) == 12
            assert loaded.encode(ValueToken(30)) == 13

            # Check decoding is preserved
            assert loaded.decode(10) == KeyToken("name")
            assert loaded.decode(13) == ValueToken(30)

            # Check type queries are preserved
            assert loaded.is_key_token(10)
            assert loaded.is_value_token(12)
        finally:
            path.unlink()

    def test_load_preserves_frozen_state(self):
        """Loaded vocabulary should preserve frozen state."""
        vocab = Vocabulary()
        vocab.add_key("test")
        # Not frozen

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = Path(f.name)

        try:
            vocab.save(path)
            loaded = Vocabulary.load(path)

            assert not loaded.frozen
            # Should be able to add more
            loaded.add_key("new_key")
        finally:
            path.unlink()
