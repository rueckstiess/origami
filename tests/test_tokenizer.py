"""Tests for the JSON tokenizer module."""

import tempfile
from pathlib import Path

import pytest

from origami.tokenizer import (
    ARRAY_END,
    ARRAY_START,
    END,
    OBJ_END,
    OBJ_START,
    START,
    DecodeError,
    IndexElement,
    JSONTokenizer,
    KeyElement,
    KeyToken,
    ValueToken,
    path_to_string,
)


class TestPathFunctions:
    """Tests for path representation and utilities."""

    def test_path_to_string_empty(self):
        """Empty path should display as <root>."""
        assert path_to_string(()) == "<root>"

    def test_path_to_string_single_key(self):
        """Single key path should display as key name."""
        path = (KeyElement("name"),)
        assert path_to_string(path) == "name"

    def test_path_to_string_nested_keys(self):
        """Nested keys should be dot-separated."""
        path = (KeyElement("user"), KeyElement("name"))
        assert path_to_string(path) == "user.name"

    def test_path_to_string_array_index(self):
        """Array indices should use bracket notation."""
        path = (KeyElement("items"), IndexElement(1))
        assert path_to_string(path) == "items[1]"

    def test_path_to_string_complex(self):
        """Complex paths should combine dot and bracket notation."""
        path = (
            KeyElement("users"),
            IndexElement(0),
            KeyElement("address"),
            KeyElement("city"),
        )
        assert path_to_string(path) == "users[0].address.city"


class TestTokenizerFit:
    """Tests for tokenizer vocabulary building."""

    def test_fit_simple_object(self):
        """Fit should add keys and values from simple objects."""
        tokenizer = JSONTokenizer()
        tokenizer.fit([{"name": "Alice", "age": 30}])

        vocab = tokenizer.vocab
        assert vocab.encode(KeyToken("name")) >= 10
        assert vocab.encode(KeyToken("age")) >= 10
        assert vocab.encode(ValueToken("Alice")) >= 10
        assert vocab.encode(ValueToken(30)) >= 10

    def test_fit_nested_objects(self):
        """Fit should traverse nested objects."""
        tokenizer = JSONTokenizer()
        tokenizer.fit([{"user": {"name": "Alice"}}])

        vocab = tokenizer.vocab
        assert vocab.encode(KeyToken("user")) >= 10
        assert vocab.encode(KeyToken("name")) >= 10
        assert vocab.encode(ValueToken("Alice")) >= 10

    def test_fit_arrays(self):
        """Fit should traverse arrays."""
        tokenizer = JSONTokenizer()
        tokenizer.fit([{"items": ["a", "b", "c"]}])

        vocab = tokenizer.vocab
        assert vocab.encode(KeyToken("items")) >= 10
        assert vocab.encode(ValueToken("a")) >= 10
        assert vocab.encode(ValueToken("b")) >= 10
        assert vocab.encode(ValueToken("c")) >= 10

    def test_fit_multiple_objects(self):
        """Fit should accumulate vocabulary from multiple objects."""
        tokenizer = JSONTokenizer()
        tokenizer.fit([{"name": "Alice"}, {"name": "Bob", "city": "NYC"}])

        vocab = tokenizer.vocab
        assert vocab.encode(KeyToken("name")) >= 10
        assert vocab.encode(KeyToken("city")) >= 10
        assert vocab.encode(ValueToken("Alice")) >= 10
        assert vocab.encode(ValueToken("Bob")) >= 10

    def test_fit_returns_self(self):
        """Fit should return self for method chaining."""
        tokenizer = JSONTokenizer()
        result = tokenizer.fit([{"name": "Alice"}])
        assert result is tokenizer


class TestTokenizerTokenize:
    """Tests for tokenization."""

    def test_tokenize_simple_object(self):
        """Tokenize simple object with START/END framing."""
        tokenizer = JSONTokenizer()
        tokenizer.fit([{"name": "Alice"}])

        instance = tokenizer.tokenize({"name": "Alice"})

        assert instance.tokens[0] == START
        assert instance.tokens[1] == OBJ_START
        assert instance.tokens[2] == KeyToken("name")
        assert instance.tokens[3] == ValueToken("Alice")
        assert instance.tokens[4] == OBJ_END
        assert instance.tokens[5] == END
        assert len(instance) == 6

    def test_tokenize_paths_simple(self):
        """Paths should track token positions in JSON hierarchy."""
        tokenizer = JSONTokenizer()
        tokenizer.fit([{"name": "Alice"}])

        instance = tokenizer.tokenize({"name": "Alice"})

        # START, END have empty path
        assert instance.paths[0] == ()  # START
        assert instance.paths[5] == ()  # END

        # OBJ_START, Key, OBJ_END have container path
        assert instance.paths[1] == ()  # OBJ_START (root object)
        assert instance.paths[2] == ()  # Key("name")
        assert instance.paths[4] == ()  # OBJ_END

        # Value has path including key
        assert instance.paths[3] == (KeyElement("name"),)

    def test_tokenize_nested_object(self):
        """Tokenize nested objects with correct structure."""
        tokenizer = JSONTokenizer()
        obj = {"user": {"name": "Alice"}}
        tokenizer.fit([obj])

        instance = tokenizer.tokenize(obj)

        expected_tokens = [
            START,
            OBJ_START,
            KeyToken("user"),
            OBJ_START,
            KeyToken("name"),
            ValueToken("Alice"),
            OBJ_END,
            OBJ_END,
            END,
        ]
        assert instance.tokens == expected_tokens

    def test_tokenize_nested_paths(self):
        """Paths should correctly track nested positions."""
        tokenizer = JSONTokenizer()
        obj = {"user": {"name": "Alice"}}
        tokenizer.fit([obj])

        instance = tokenizer.tokenize(obj)

        # Inner OBJ_START is at path ("user",)
        assert instance.paths[3] == (KeyElement("user"),)

        # Inner key "name" is at path ("user",)
        assert instance.paths[4] == (KeyElement("user"),)

        # Value "Alice" is at path ("user", "name")
        assert instance.paths[5] == (KeyElement("user"), KeyElement("name"))

    def test_tokenize_array(self):
        """Tokenize arrays with ARRAY_START/END."""
        tokenizer = JSONTokenizer()
        obj = {"items": ["a", "b"]}
        tokenizer.fit([obj])

        instance = tokenizer.tokenize(obj)

        expected_tokens = [
            START,
            OBJ_START,
            KeyToken("items"),
            ARRAY_START,
            ValueToken("a"),
            ValueToken("b"),
            ARRAY_END,
            OBJ_END,
            END,
        ]
        assert instance.tokens == expected_tokens

    def test_tokenize_array_paths(self):
        """Array element paths should include IndexElement."""
        tokenizer = JSONTokenizer()
        obj = {"items": ["a", "b"]}
        tokenizer.fit([obj])

        instance = tokenizer.tokenize(obj)

        # ARRAY_START at path ("items",)
        assert instance.paths[3] == (KeyElement("items"),)

        # Elements at indexed paths
        assert instance.paths[4] == (KeyElement("items"), IndexElement(0))
        assert instance.paths[5] == (KeyElement("items"), IndexElement(1))

    def test_tokenize_empty_object(self):
        """Empty object should have OBJ_START, OBJ_END only."""
        tokenizer = JSONTokenizer()

        instance = tokenizer.tokenize({})

        assert instance.tokens == [START, OBJ_START, OBJ_END, END]

    def test_tokenize_empty_array(self):
        """Empty array in object."""
        tokenizer = JSONTokenizer()
        obj = {"items": []}
        tokenizer.fit([obj])

        instance = tokenizer.tokenize(obj)

        assert KeyToken("items") in instance.tokens
        assert ARRAY_START in instance.tokens
        assert ARRAY_END in instance.tokens

    def test_tokenize_nested_arrays(self):
        """Nested arrays should tokenize correctly."""
        tokenizer = JSONTokenizer()
        obj = {"matrix": [[1, 2], [3, 4]]}
        tokenizer.fit([obj])

        instance = tokenizer.tokenize(obj)

        # Should have outer ARRAY_START/END and inner ones
        array_starts = instance.tokens.count(ARRAY_START)
        array_ends = instance.tokens.count(ARRAY_END)
        assert array_starts == 3  # One outer, two inner
        assert array_ends == 3

    def test_tokenize_mixed_types(self):
        """Tokenize various JSON value types."""
        tokenizer = JSONTokenizer()
        obj = {
            "string": "hello",
            "int": 42,
            "float": 3.14,
            "bool_true": True,
            "bool_false": False,
            "null": None,
        }
        tokenizer.fit([obj])

        instance = tokenizer.tokenize(obj)

        assert ValueToken("hello") in instance.tokens
        assert ValueToken(42) in instance.tokens
        assert ValueToken(3.14) in instance.tokens
        assert ValueToken(True) in instance.tokens
        assert ValueToken(False) in instance.tokens
        assert ValueToken(None) in instance.tokens


class TestTokenizerShuffle:
    """Tests for key shuffling during tokenization."""

    def test_shuffle_produces_different_orders(self):
        """Shuffle should produce different key orderings."""
        tokenizer = JSONTokenizer()
        obj = {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}
        tokenizer.fit([obj])

        # Collect many shuffled tokenizations
        orders = []
        for _ in range(50):
            instance = tokenizer.tokenize(obj, shuffle=True)
            # Extract key order from tokens
            keys = [t.key for t in instance.tokens if isinstance(t, KeyToken)]
            orders.append(tuple(keys))

        # Should see multiple different orderings
        unique_orders = set(orders)
        assert len(unique_orders) > 1

    def test_shuffle_false_preserves_order(self):
        """Without shuffle, key order should be deterministic."""
        tokenizer = JSONTokenizer()
        obj = {"name": "Alice", "age": 30}
        tokenizer.fit([obj])

        instances = [tokenizer.tokenize(obj, shuffle=False) for _ in range(10)]
        token_lists = [tuple(inst.tokens) for inst in instances]

        # All should be identical
        assert len(set(token_lists)) == 1

    def test_shuffle_preserves_key_value_pairs(self):
        """Shuffle should keep key-value pairs together."""
        tokenizer = JSONTokenizer()
        obj = {"name": "Alice", "age": 30, "city": "NYC"}
        tokenizer.fit([obj])

        for _ in range(20):
            instance = tokenizer.tokenize(obj, shuffle=True)

            # Find each key and verify its value follows
            for i, token in enumerate(instance.tokens):
                if isinstance(token, KeyToken):
                    if token.key == "name":
                        assert instance.tokens[i + 1] == ValueToken("Alice")
                    elif token.key == "age":
                        assert instance.tokens[i + 1] == ValueToken(30)
                    elif token.key == "city":
                        assert instance.tokens[i + 1] == ValueToken("NYC")

    def test_shuffle_at_all_nesting_levels(self):
        """Shuffle should apply at each nesting level independently."""
        tokenizer = JSONTokenizer()
        obj = {"outer1": {"inner1": 1, "inner2": 2}, "outer2": {"inner3": 3, "inner4": 4}}
        tokenizer.fit([obj])

        outer_orders = []

        for _ in range(50):
            instance = tokenizer.tokenize(obj, shuffle=True)

            # Extract outer key order
            outer_keys = []
            depth = 0
            for token in instance.tokens:
                if token == OBJ_START:
                    depth += 1
                elif token == OBJ_END:
                    depth -= 1
                elif isinstance(token, KeyToken) and depth == 1:
                    outer_keys.append(token.key)
            outer_orders.append(tuple(outer_keys))

        # Should see variation in outer keys
        assert len(set(outer_orders)) > 1


class TestTokenizerDecode:
    """Tests for decoding token sequences back to JSON."""

    def test_decode_simple_object(self):
        """Decode should reconstruct simple objects."""
        tokenizer = JSONTokenizer()
        obj = {"name": "Alice", "age": 30}
        tokenizer.fit([obj])
        tokenizer.vocab.freeze()

        instance = tokenizer.tokenize(obj)
        token_ids = tokenizer.encode_tokens(instance)
        decoded = tokenizer.decode(token_ids)

        assert decoded == obj

    def test_decode_nested_object(self):
        """Decode should reconstruct nested objects."""
        tokenizer = JSONTokenizer()
        obj = {"user": {"name": "Alice", "info": {"age": 30}}}
        tokenizer.fit([obj])
        tokenizer.vocab.freeze()

        instance = tokenizer.tokenize(obj)
        token_ids = tokenizer.encode_tokens(instance)
        decoded = tokenizer.decode(token_ids)

        assert decoded == obj

    def test_decode_with_arrays(self):
        """Decode should reconstruct arrays."""
        tokenizer = JSONTokenizer()
        obj = {"items": ["a", "b", "c"], "nested": [[1, 2], [3, 4]]}
        tokenizer.fit([obj])
        tokenizer.vocab.freeze()

        instance = tokenizer.tokenize(obj)
        token_ids = tokenizer.encode_tokens(instance)
        decoded = tokenizer.decode(token_ids)

        assert decoded == obj

    def test_decode_empty_structures(self):
        """Decode should handle empty objects and arrays."""
        tokenizer = JSONTokenizer()
        obj = {"empty_obj": {}, "empty_arr": []}
        tokenizer.fit([obj])
        tokenizer.vocab.freeze()

        instance = tokenizer.tokenize(obj)
        token_ids = tokenizer.encode_tokens(instance)
        decoded = tokenizer.decode(token_ids)

        assert decoded == obj

    def test_decode_roundtrip_with_shuffle(self):
        """Shuffled tokenization should decode to same object."""
        tokenizer = JSONTokenizer()
        obj = {"a": 1, "b": 2, "c": {"d": 3, "e": 4}}
        tokenizer.fit([obj])
        tokenizer.vocab.freeze()

        for _ in range(10):
            instance = tokenizer.tokenize(obj, shuffle=True)
            token_ids = tokenizer.encode_tokens(instance)
            decoded = tokenizer.decode(token_ids)
            assert decoded == obj


class TestDecodeErrors:
    """Tests for decode error handling."""

    def test_decode_empty_sequence(self):
        """Empty sequence should raise DecodeError."""
        tokenizer = JSONTokenizer()
        with pytest.raises(DecodeError) as exc:
            tokenizer.decode([])
        assert "Empty" in str(exc.value)

    def test_decode_missing_start(self):
        """Missing START token should raise DecodeError."""
        tokenizer = JSONTokenizer()
        tokenizer.vocab.freeze()

        # Sequence without START
        token_ids = [
            tokenizer.vocab.obj_start_id,
            tokenizer.vocab.obj_end_id,
            tokenizer.vocab.end_id,
        ]
        with pytest.raises(DecodeError) as exc:
            tokenizer.decode(token_ids)
        assert "START" in str(exc.value)

    def test_decode_missing_end(self):
        """Missing END token should raise DecodeError."""
        tokenizer = JSONTokenizer()
        tokenizer.vocab.freeze()

        # Sequence without END
        token_ids = [
            tokenizer.vocab.start_id,
            tokenizer.vocab.obj_start_id,
            tokenizer.vocab.obj_end_id,
        ]
        with pytest.raises(DecodeError) as exc:
            tokenizer.decode(token_ids)
        assert "END" in str(exc.value)

    def test_decode_unterminated_object(self):
        """Object without OBJ_END should raise DecodeError."""
        tokenizer = JSONTokenizer()
        tokenizer.vocab.freeze()

        # OBJ_START without OBJ_END - decoder expects key but gets END
        token_ids = [
            tokenizer.vocab.start_id,
            tokenizer.vocab.obj_start_id,
            tokenizer.vocab.end_id,
        ]
        with pytest.raises(DecodeError) as exc:
            tokenizer.decode(token_ids)
        # Error indicates structural problem (expected key, got END)
        assert exc.value.position >= 0

    def test_decode_error_shows_context(self):
        """DecodeError should include helpful context."""
        tokenizer = JSONTokenizer()
        with pytest.raises(DecodeError) as exc:
            tokenizer.decode([99, 98, 97])

        error = exc.value
        assert error.position == 0
        assert error.tokens == [99, 98, 97]
        assert "position" in str(error).lower()


class TestTokenizerSerialization:
    """Tests for tokenizer save/load."""

    def test_save_load_roundtrip(self):
        """Saved tokenizer should be functionally identical when loaded."""
        tokenizer = JSONTokenizer()
        obj = {"name": "Alice", "items": [1, 2, 3]}
        tokenizer.fit([obj])
        tokenizer.vocab.freeze()

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = Path(f.name)

        try:
            tokenizer.save(path)
            loaded = JSONTokenizer.load(path)

            # Should produce identical tokenization
            instance1 = tokenizer.tokenize(obj)
            instance2 = loaded.tokenize(obj)

            ids1 = tokenizer.encode_tokens(instance1)
            ids2 = loaded.encode_tokens(instance2)

            assert ids1 == ids2

            # Should decode identically
            decoded = loaded.decode(ids2)
            assert decoded == obj
        finally:
            path.unlink()

    def test_load_preserves_vocab_state(self):
        """Loaded tokenizer should preserve vocabulary freeze state."""
        tokenizer = JSONTokenizer()
        tokenizer.fit([{"x": 1}])
        tokenizer.vocab.freeze()

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = Path(f.name)

        try:
            tokenizer.save(path)
            loaded = JSONTokenizer.load(path)

            assert loaded.vocab.frozen
        finally:
            path.unlink()


class TestTokenizerEncodeTokens:
    """Tests for token to ID encoding."""

    def test_encode_tokens_basic(self):
        """encode_tokens should convert tokens to IDs."""
        tokenizer = JSONTokenizer()
        obj = {"name": "Alice"}
        tokenizer.fit([obj])

        instance = tokenizer.tokenize(obj)
        ids = tokenizer.encode_tokens(instance)

        assert ids[0] == tokenizer.vocab.start_id
        assert ids[1] == tokenizer.vocab.obj_start_id
        assert ids[-2] == tokenizer.vocab.obj_end_id
        assert ids[-1] == tokenizer.vocab.end_id

    def test_encode_tokens_length_matches(self):
        """Encoded IDs should have same length as tokens."""
        tokenizer = JSONTokenizer()
        obj = {"a": 1, "b": [2, 3], "c": {"d": 4}}
        tokenizer.fit([obj])

        instance = tokenizer.tokenize(obj)
        ids = tokenizer.encode_tokens(instance)

        assert len(ids) == len(instance.tokens)
        assert len(ids) == len(instance.paths)


class TestUnknownTokenHandling:
    """Tests for handling unknown keys and values after fitting."""

    def test_unknown_key_maps_to_unk_key(self):
        """Unknown keys should map to UNK_KEY token after fit."""
        tokenizer = JSONTokenizer()
        tokenizer.fit([{"known_key": "value"}])

        # Tokenize object with unknown key
        obj = {"unknown_key": "value"}
        instance = tokenizer.tokenize(obj)
        ids = tokenizer.encode_tokens(instance)

        # Should contain UNK_KEY instead of raising error
        assert tokenizer.vocab.unk_key_id in ids

    def test_unknown_value_maps_to_unk_value(self):
        """Unknown values should map to UNK_VALUE token after fit."""
        tokenizer = JSONTokenizer()
        tokenizer.fit([{"key": "known_value"}])

        # Tokenize object with unknown value
        obj = {"key": "unknown_value"}
        instance = tokenizer.tokenize(obj)
        ids = tokenizer.encode_tokens(instance)

        # Should contain UNK_VALUE instead of raising error
        assert tokenizer.vocab.unk_value_id in ids

    def test_mixed_known_unknown(self):
        """Objects with mix of known and unknown tokens should work."""
        tokenizer = JSONTokenizer()
        tokenizer.fit([{"name": "Alice", "age": 30}])

        # Object with known key, unknown value + unknown key, known value
        obj = {"name": "Bob", "city": 30}  # "Bob" unknown, "city" unknown
        instance = tokenizer.tokenize(obj)
        ids = tokenizer.encode_tokens(instance)

        # Should have both UNK types
        assert tokenizer.vocab.unk_key_id in ids  # "city"
        assert tokenizer.vocab.unk_value_id in ids  # "Bob"

    def test_collate_objects_with_unknown_tokens(self):
        """collate_objects should handle unknown tokens gracefully."""
        from origami.training import OrigamiDataCollator

        tokenizer = JSONTokenizer()
        tokenizer.fit([{"name": "Alice"}])

        # Batch with known and unknown tokens
        objects = [
            {"name": "Alice"},  # All known
            {"name": "Bob", "extra": "data"},  # Unknown value + unknown key
        ]
        collator = OrigamiDataCollator(tokenizer, include_labels=False)
        batch = collator.collate_objects(objects)

        # Should not raise, should produce valid tensors
        assert batch.input_ids.shape[0] == 2
        # UNK tokens should be present in second sequence
        assert (batch.input_ids[1] == tokenizer.vocab.unk_key_id).any()
        assert (batch.input_ids[1] == tokenizer.vocab.unk_value_id).any()

    def test_vocab_frozen_after_fit(self):
        """Vocabulary should be frozen after fit."""
        tokenizer = JSONTokenizer()
        tokenizer.fit([{"key": "value"}])

        assert tokenizer.vocab.frozen


class TestTokenizerMaxVocabSize:
    """Tests for max_vocab_size vocabulary pruning."""

    def test_fit_with_max_vocab_size(self):
        """fit() with max_vocab_size should prune vocabulary."""
        # Create data with many unique values
        data = [{"key": f"value_{i}"} for i in range(20)]

        tokenizer = JSONTokenizer()
        # Grammar(10) + key(1) + values(5) = 16
        tokenizer.fit(data, max_vocab_size=16)

        assert tokenizer.vocab.size == 16
        assert tokenizer.pruning_stats is not None
        assert tokenizer.pruning_stats.num_values_pruned == 15  # 20 - 5 = 15

    def test_fit_without_max_vocab_size(self):
        """fit() without max_vocab_size should keep all tokens."""
        data = [{"key": v} for v in ["a", "b", "c"]]

        tokenizer = JSONTokenizer()
        tokenizer.fit(data)

        assert tokenizer.pruning_stats is None
        assert tokenizer.vocab.size == 14  # Grammar(10) + key(1) + values(3)

    def test_pruned_values_map_to_unk(self):
        """Pruned values should encode to UNK_VALUE."""
        # Create data with frequency distribution
        data = [{"k": "common"}] * 10 + [{"k": "rare"}]

        tokenizer = JSONTokenizer()
        # Grammar(10) + key(1) + value(1) = 12, so "rare" gets pruned
        tokenizer.fit(data, max_vocab_size=12)

        # Tokenize with the rare value
        instance = tokenizer.tokenize({"k": "rare"})
        ids = tokenizer.encode_tokens(instance)

        # "rare" should map to UNK_VALUE
        assert tokenizer.vocab.unk_value_id in ids

    def test_common_values_preserved(self):
        """Most frequent values should be preserved."""
        # Create data with clear frequency hierarchy
        data = []
        for _ in range(100):
            data.append({"k": "very_common"})
        for _ in range(10):
            data.append({"k": "common"})
        for _ in range(1):
            data.append({"k": "rare"})

        tokenizer = JSONTokenizer()
        # Grammar(10) + key(1) + values(2) = 13
        tokenizer.fit(data, max_vocab_size=13)

        # very_common and common should be preserved
        instance = tokenizer.tokenize({"k": "very_common"})
        ids = tokenizer.encode_tokens(instance)
        assert tokenizer.vocab.unk_value_id not in ids

        instance = tokenizer.tokenize({"k": "common"})
        ids = tokenizer.encode_tokens(instance)
        assert tokenizer.vocab.unk_value_id not in ids

        # rare should map to UNK
        instance = tokenizer.tokenize({"k": "rare"})
        ids = tokenizer.encode_tokens(instance)
        assert tokenizer.vocab.unk_value_id in ids

    def test_all_keys_preserved(self):
        """All keys should be preserved regardless of max_vocab_size."""
        data = [{"key1": "v1"}, {"key2": "v2"}, {"key3": "v3"}]

        tokenizer = JSONTokenizer()
        # Grammar(10) + keys(3) + values(0) = 13
        tokenizer.fit(data, max_vocab_size=13)

        # All keys should work
        for key in ["key1", "key2", "key3"]:
            instance = tokenizer.tokenize({key: "new_val"})
            ids = tokenizer.encode_tokens(instance)
            # Key should NOT be UNK
            assert tokenizer.vocab.unk_key_id not in ids

    def test_max_vocab_size_too_small_raises(self):
        """max_vocab_size too small for grammar+keys should raise."""
        data = [{"k1": "v"}, {"k2": "v"}, {"k3": "v"}]

        tokenizer = JSONTokenizer()
        # Need grammar(10) + keys(3) = 13 minimum
        with pytest.raises(ValueError, match="too small"):
            tokenizer.fit(data, max_vocab_size=12)

    def test_pruning_stats_serialization(self):
        """Saved tokenizer should preserve pruning stats."""
        data = [{"k": "common"}] * 10 + [{"k": "rare"}]

        tokenizer = JSONTokenizer()
        tokenizer.fit(data, max_vocab_size=12)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = Path(f.name)

        try:
            tokenizer.save(path)
            loaded = JSONTokenizer.load(path)

            assert loaded.pruning_stats is not None
            assert loaded.pruning_stats.num_values_pruned == 1
        finally:
            path.unlink()

    def test_refit_clears_pruning_stats(self):
        """Re-fitting tokenizer should clear previous pruning stats."""
        tokenizer = JSONTokenizer()

        # First fit with pruning
        data1 = [{"k": f"v{i}"} for i in range(10)]
        tokenizer.fit(data1, max_vocab_size=12)
        assert tokenizer.pruning_stats is not None

        # Second fit without pruning
        data2 = [{"k": "v"}]
        tokenizer.fit(data2)
        assert tokenizer.pruning_stats is None
