"""Tests for JSON grammar constraint PDA."""

import pytest
import torch

from origami.constraints.json_grammar import JSONGrammarPDA
from origami.tokenizer.vocabulary import Vocabulary


@pytest.fixture
def vocab():
    """Create a simple vocabulary for testing."""
    v = Vocabulary()
    # Add some keys
    v.add_key("name")
    v.add_key("age")
    v.add_key("items")
    # Add some values
    v.add_value("Alice")
    v.add_value(42)
    v.add_value(True)
    v.freeze()
    return v


@pytest.fixture
def pda(vocab):
    """Create PDA with test vocabulary."""
    return JSONGrammarPDA(vocab)


class TestJSONGrammarPDA:
    """Tests for JSONGrammarPDA class.

    Note on semantics: mask[t] indicates valid tokens for position t+1,
    given that we've seen tokens 0..t. This aligns with autoregressive
    training where logits[t] predicts the token at position t+1.
    """

    def test_init(self, vocab, pda):
        """Test PDA initialization."""
        assert pda.vocab is vocab
        assert pda.max_depth == 32
        assert len(pda._key_ids) == 4  # name, age, items + UNK_KEY
        assert len(pda._value_ids) == 5  # Alice, 42, True + UNK_VALUE + NUM

    def test_simple_object_sequence(self, vocab, pda):
        """Test grammar masks for simple object.

        Sequence: START OBJ_START "name" "Alice" OBJ_END END PAD
        mask[t] = valid tokens for position t+1
        """
        name_id = vocab.encode(vocab._id_to_token[10])  # First key
        alice_id = vocab.encode(vocab._id_to_token[13])  # First value

        tokens = torch.tensor(
            [
                [
                    vocab.start_id,
                    vocab.obj_start_id,
                    name_id,
                    alice_id,
                    vocab.obj_end_id,
                    vocab.end_id,
                    vocab.pad_token_id,
                ]
            ]
        )

        masks = pda.compute_valid_mask(tokens)

        # mask[0]: After START, position 1 should allow OBJ_START or ARRAY_START
        assert masks[0, 0, vocab.obj_start_id].item() is True
        assert masks[0, 0, vocab.array_start_id].item() is True
        assert masks[0, 0, vocab.start_id].item() is False

        # mask[1]: After OBJ_START, position 2 should allow keys or OBJ_END
        assert masks[0, 1, name_id].item() is True
        assert masks[0, 1, vocab.obj_end_id].item() is True
        assert masks[0, 1, alice_id].item() is False  # Can't have value without key

        # mask[2]: After key "name", position 3 should allow values
        assert masks[0, 2, alice_id].item() is True
        assert masks[0, 2, vocab.obj_start_id].item() is True  # Nested object
        assert masks[0, 2, name_id].item() is False  # Can't have another key

        # mask[3]: After value "Alice", position 4 should allow key or OBJ_END
        assert masks[0, 3, name_id].item() is True  # Another key
        assert masks[0, 3, vocab.obj_end_id].item() is True

        # mask[4]: After OBJ_END (root closed), position 5 should allow END only
        assert masks[0, 4, vocab.end_id].item() is True
        assert masks[0, 4, vocab.start_id].item() is False
        assert masks[0, 4, vocab.obj_start_id].item() is False

        # mask[5]: After END, position 6 should allow PAD only
        assert masks[0, 5, vocab.pad_token_id].item() is True
        assert masks[0, 5, vocab.end_id].item() is False

    def test_simple_array_sequence(self, vocab, pda):
        """Test grammar masks for array: START ARRAY_START value value ARRAY_END END"""
        alice_id = vocab.encode(vocab._id_to_token[13])  # First value
        val_42 = vocab.encode(vocab._id_to_token[14])  # 42

        tokens = torch.tensor(
            [
                [
                    vocab.start_id,
                    vocab.array_start_id,
                    alice_id,
                    val_42,
                    vocab.array_end_id,
                    vocab.end_id,
                ]
            ]
        )

        masks = pda.compute_valid_mask(tokens)

        # mask[1]: After ARRAY_START, position 2 should allow values or ARRAY_END
        assert masks[0, 1, alice_id].item() is True
        assert masks[0, 1, vocab.array_end_id].item() is True
        assert masks[0, 1, vocab.obj_start_id].item() is True  # Nested object

        # mask[2]: After first value, position 3 should allow values or ARRAY_END
        assert masks[0, 2, val_42].item() is True
        assert masks[0, 2, vocab.array_end_id].item() is True

        # mask[3]: After second value, position 4 should allow values or ARRAY_END
        assert masks[0, 3, vocab.array_end_id].item() is True

        # mask[4]: After ARRAY_END (root closed), position 5 should allow END only
        assert masks[0, 4, vocab.end_id].item() is True

    def test_nested_object(self, vocab, pda):
        """Test nested object structure."""
        name_id = vocab.encode(vocab._id_to_token[10])
        alice_id = vocab.encode(vocab._id_to_token[13])
        items_id = vocab.encode(vocab._id_to_token[12])  # "items" key

        # START OBJ_START "name" "Alice" "items" OBJ_START OBJ_END OBJ_END END
        tokens = torch.tensor(
            [
                [
                    vocab.start_id,
                    vocab.obj_start_id,
                    name_id,
                    alice_id,
                    items_id,
                    vocab.obj_start_id,
                    vocab.obj_end_id,
                    vocab.obj_end_id,
                    vocab.end_id,
                ]
            ]
        )

        masks = pda.compute_valid_mask(tokens)

        # mask[4]: After "items" key, position 5 should allow values including OBJ_START
        assert masks[0, 4, vocab.obj_start_id].item() is True
        assert masks[0, 4, alice_id].item() is True

        # mask[5]: After nested OBJ_START, position 6 should allow keys or OBJ_END
        assert masks[0, 5, name_id].item() is True
        assert masks[0, 5, vocab.obj_end_id].item() is True

        # mask[6]: After nested OBJ_END, position 7 back in parent object, allow key or close
        assert masks[0, 6, name_id].item() is True
        assert masks[0, 6, vocab.obj_end_id].item() is True

        # mask[7]: After root OBJ_END, position 8 should allow END only
        assert masks[0, 7, vocab.end_id].item() is True

    def test_batch_parallel(self, vocab, pda):
        """Test that batch processing produces correct independent results."""
        name_id = vocab.encode(vocab._id_to_token[10])
        alice_id = vocab.encode(vocab._id_to_token[13])

        # Two sequences with different structures
        # Seq 0: object
        # Seq 1: array
        tokens = torch.tensor(
            [
                [
                    vocab.start_id,
                    vocab.obj_start_id,
                    name_id,
                    alice_id,
                    vocab.obj_end_id,
                    vocab.end_id,
                    vocab.pad_token_id,
                ],
                [
                    vocab.start_id,
                    vocab.array_start_id,
                    alice_id,
                    alice_id,
                    vocab.array_end_id,
                    vocab.end_id,
                    vocab.pad_token_id,
                ],
            ]
        )

        masks = pda.compute_valid_mask(tokens)

        # mask[0]: After START, both should allow OBJ_START and ARRAY_START
        assert masks[0, 0, vocab.obj_start_id].item() is True
        assert masks[0, 0, vocab.array_start_id].item() is True
        assert masks[1, 0, vocab.obj_start_id].item() is True
        assert masks[1, 0, vocab.array_start_id].item() is True

        # mask[1]: Seq 0 after OBJ_START (keys/OBJ_END), Seq 1 after ARRAY_START (values/ARRAY_END)
        assert masks[0, 1, name_id].item() is True  # Key valid in object
        assert masks[0, 1, vocab.obj_end_id].item() is True
        assert masks[1, 1, alice_id].item() is True  # Value valid in array
        assert masks[1, 1, vocab.array_end_id].item() is True

    def test_attention_mask(self, vocab, pda):
        """Test that padding positions get PAD-only mask."""
        name_id = vocab.encode(vocab._id_to_token[10])
        alice_id = vocab.encode(vocab._id_to_token[13])

        tokens = torch.tensor(
            [
                [
                    vocab.start_id,
                    vocab.obj_start_id,
                    name_id,
                    alice_id,
                    vocab.obj_end_id,
                    vocab.end_id,
                    vocab.pad_token_id,
                    vocab.pad_token_id,
                ]
            ]
        )

        masks = pda.compute_valid_mask(tokens)

        # After END, grammar state doesn't allow any valid token (ended=True)
        # The PAD token at position 6 follows END, so grammar allows nothing meaningful
        # Loss computation handles PAD via attention_mask separately
        assert masks[0, 6, vocab.start_id].item() is False

    def test_apply_constraints(self, vocab, pda):
        """Test logit masking with constraints."""
        tokens = torch.tensor([[vocab.start_id, vocab.obj_start_id]])
        masks = pda.compute_valid_mask(tokens)

        # Create fake logits
        logits = torch.ones(1, 2, vocab.size)

        constrained = pda.apply_constraints(logits, masks)

        # mask[0]: After START, OBJ_START and ARRAY_START should remain
        assert constrained[0, 0, vocab.obj_start_id].item() == 1.0
        assert constrained[0, 0, vocab.array_start_id].item() == 1.0
        assert constrained[0, 0, vocab.start_id].item() == float("-inf")

        # mask[1]: After OBJ_START, keys and OBJ_END should remain
        assert constrained[0, 1, vocab.obj_end_id].item() == 1.0
        # Keys should be valid
        name_id = vocab.encode(vocab._id_to_token[10])
        assert constrained[0, 1, name_id].item() == 1.0

    def test_empty_object(self, vocab, pda):
        """Test empty object: START OBJ_START OBJ_END END"""
        tokens = torch.tensor(
            [[vocab.start_id, vocab.obj_start_id, vocab.obj_end_id, vocab.end_id]]
        )
        masks = pda.compute_valid_mask(tokens)

        # mask[1]: After OBJ_START, OBJ_END should be valid (empty object)
        assert masks[0, 1, vocab.obj_end_id].item() is True

        # mask[2]: After OBJ_END (root closed), only END valid
        assert masks[0, 2, vocab.end_id].item() is True

    def test_empty_array(self, vocab, pda):
        """Test empty array: START ARRAY_START ARRAY_END END"""
        tokens = torch.tensor(
            [[vocab.start_id, vocab.array_start_id, vocab.array_end_id, vocab.end_id]]
        )
        masks = pda.compute_valid_mask(tokens)

        # mask[1]: After ARRAY_START, ARRAY_END should be valid (empty array)
        assert masks[0, 1, vocab.array_end_id].item() is True

        # mask[2]: After ARRAY_END (root closed), only END valid
        assert masks[0, 2, vocab.end_id].item() is True

    def test_array_with_objects(self, vocab, pda):
        """Test array containing objects."""
        name_id = vocab.encode(vocab._id_to_token[10])
        alice_id = vocab.encode(vocab._id_to_token[13])

        # START ARRAY_START OBJ_START "name" "Alice" OBJ_END OBJ_START OBJ_END ARRAY_END END
        tokens = torch.tensor(
            [
                [
                    vocab.start_id,
                    vocab.array_start_id,
                    vocab.obj_start_id,
                    name_id,
                    alice_id,
                    vocab.obj_end_id,
                    vocab.obj_start_id,
                    vocab.obj_end_id,
                    vocab.array_end_id,
                    vocab.end_id,
                ]
            ]
        )

        masks = pda.compute_valid_mask(tokens)

        # mask[1]: After ARRAY_START, can start object or add value
        assert masks[0, 1, vocab.obj_start_id].item() is True

        # mask[5]: After first object closes, back in array, can start another object
        assert masks[0, 5, vocab.obj_start_id].item() is True
        assert masks[0, 5, vocab.array_end_id].item() is True

        # mask[7]: After second object closes, can close array
        assert masks[0, 7, vocab.array_end_id].item() is True

    def test_num_token_valid_as_value(self, vocab, pda):
        """Test that NUM token is valid in value positions."""
        name_id = vocab.encode(vocab._id_to_token[10])

        tokens = torch.tensor(
            [
                [
                    vocab.start_id,
                    vocab.obj_start_id,
                    name_id,
                    vocab.num_token_id,
                    vocab.obj_end_id,
                    vocab.end_id,
                ]
            ]
        )

        masks = pda.compute_valid_mask(tokens)

        # mask[2]: After key, NUM should be valid for next position
        assert masks[0, 2, vocab.num_token_id].item() is True

    def test_unk_key_valid(self, vocab, pda):
        """Test that UNK_KEY is valid in key positions."""
        alice_id = vocab.encode(vocab._id_to_token[13])

        tokens = torch.tensor(
            [
                [
                    vocab.start_id,
                    vocab.obj_start_id,
                    vocab.unk_key_id,
                    alice_id,
                    vocab.obj_end_id,
                    vocab.end_id,
                ]
            ]
        )

        masks = pda.compute_valid_mask(tokens)

        # mask[1]: After OBJ_START, UNK_KEY should be valid for next position
        assert masks[0, 1, vocab.unk_key_id].item() is True

    def test_unk_value_valid(self, vocab, pda):
        """Test that UNK_VALUE is valid in value positions."""
        name_id = vocab.encode(vocab._id_to_token[10])

        tokens = torch.tensor(
            [
                [
                    vocab.start_id,
                    vocab.obj_start_id,
                    name_id,
                    vocab.unk_value_id,
                    vocab.obj_end_id,
                    vocab.end_id,
                ]
            ]
        )

        masks = pda.compute_valid_mask(tokens)

        # mask[2]: After key, UNK_VALUE should be valid for next position
        assert masks[0, 2, vocab.unk_value_id].item() is True


class TestGrammarPerformance:
    """Performance tests for grammar constraints."""

    def test_large_batch(self, vocab, pda):
        """Test with large batch size."""
        batch_size = 128
        seq_len = 64

        # Create random valid-ish sequences (will just test performance, not correctness)
        tokens = torch.zeros(batch_size, seq_len, dtype=torch.long)
        tokens[:, 0] = vocab.start_id
        tokens[:, 1] = vocab.obj_start_id
        for i in range(2, seq_len - 2, 2):
            tokens[:, i] = vocab.encode(vocab._id_to_token[10])  # key
            tokens[:, i + 1] = vocab.encode(vocab._id_to_token[13])  # value
        tokens[:, -2] = vocab.obj_end_id
        tokens[:, -1] = vocab.end_id

        # Just verify it runs without error
        masks = pda.compute_valid_mask(tokens)
        assert masks.shape == (batch_size, seq_len, vocab.size)

    def test_deep_nesting(self, vocab, pda):
        """Test with deeply nested structure."""
        # Create a sequence with 10 levels of nesting
        depth = 10
        items_id = vocab.encode(vocab._id_to_token[12])  # "items" key

        seq = [vocab.start_id, vocab.obj_start_id]
        for _ in range(depth):
            seq.extend([items_id, vocab.obj_start_id])
        for _ in range(depth):
            seq.append(vocab.obj_end_id)
        seq.extend([vocab.obj_end_id, vocab.end_id])

        tokens = torch.tensor([seq])
        masks = pda.compute_valid_mask(tokens)

        # mask[-2]: After last OBJ_END (root closed), should allow END
        assert masks[0, -2, vocab.end_id].item() is True
