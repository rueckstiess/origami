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


class TestPDAStateTransitions:
    """Tests for internal PDA state transitions.

    These tests verify the correctness of _update_state by checking
    the state variables (stack, depth, awaiting_value, etc.) after
    processing each token.
    """

    def test_init_state(self, pda):
        """Test initial state is correct."""
        state = pda.init_state(batch_size=2, device=torch.device("cpu"))
        stack, depth, awaiting_value, seen_start, root_closed, ended = state

        assert stack.shape == (2, 32)  # batch_size x max_depth
        assert depth.shape == (2,)
        assert (stack == 0).all()  # STACK_EMPTY
        assert (depth == 0).all()
        assert not awaiting_value.any()
        assert not seen_start.any()
        assert not root_closed.any()
        assert not ended.any()

    def test_start_token_transition(self, vocab, pda):
        """Test state after START token."""
        state = pda.init_state(batch_size=1, device=torch.device("cpu"))
        token = torch.tensor([vocab.start_id])

        mask, new_state = pda.get_next_token_mask(token, state)
        _, depth, _, seen_start, root_closed, ended = new_state

        # After START: seen_start=True, everything else unchanged
        assert seen_start[0].item() is True
        assert depth[0].item() == 0
        assert root_closed[0].item() is False
        assert ended[0].item() is False

        # Next valid tokens: OBJ_START or ARRAY_START only
        assert mask[0, vocab.obj_start_id].item() is True
        assert mask[0, vocab.array_start_id].item() is True
        assert mask[0, vocab.start_id].item() is False
        assert mask[0, vocab.end_id].item() is False

    def test_obj_start_transition(self, vocab, pda):
        """Test state after OBJ_START token."""
        state = pda.init_state(batch_size=1, device=torch.device("cpu"))

        # Process START
        token = torch.tensor([vocab.start_id])
        _, state = pda.get_next_token_mask(token, state)

        # Process OBJ_START
        token = torch.tensor([vocab.obj_start_id])
        mask, state = pda.get_next_token_mask(token, state)
        stack, depth, awaiting_value, _, root_closed, _ = state

        # After OBJ_START: depth=1, stack[0]=STACK_OBJECT, awaiting_value=False
        assert depth[0].item() == 1
        assert stack[0, 0].item() == 1  # STACK_OBJECT
        assert awaiting_value[0].item() is False
        assert root_closed[0].item() is False

        # Next valid: keys or OBJ_END
        name_id = vocab.encode(vocab._id_to_token[10])
        assert mask[0, name_id].item() is True
        assert mask[0, vocab.obj_end_id].item() is True
        assert mask[0, vocab.array_end_id].item() is False

    def test_key_transition(self, vocab, pda):
        """Test state after key token."""
        state = pda.init_state(batch_size=1, device=torch.device("cpu"))
        name_id = vocab.encode(vocab._id_to_token[10])

        # Process START -> OBJ_START -> key
        for token_id in [vocab.start_id, vocab.obj_start_id, name_id]:
            token = torch.tensor([token_id])
            mask, state = pda.get_next_token_mask(token, state)

        _, depth, awaiting_value, _, _, _ = state

        # After key: awaiting_value=True
        assert awaiting_value[0].item() is True
        assert depth[0].item() == 1

        # Next valid: values (primitives, OBJ_START, ARRAY_START, NUM)
        alice_id = vocab.encode(vocab._id_to_token[13])
        assert mask[0, alice_id].item() is True
        assert mask[0, vocab.obj_start_id].item() is True
        assert mask[0, vocab.array_start_id].item() is True
        assert mask[0, vocab.num_token_id].item() is True
        # Not valid: keys, OBJ_END
        assert mask[0, name_id].item() is False
        assert mask[0, vocab.obj_end_id].item() is False

    def test_value_transition(self, vocab, pda):
        """Test state after value token."""
        state = pda.init_state(batch_size=1, device=torch.device("cpu"))
        name_id = vocab.encode(vocab._id_to_token[10])
        alice_id = vocab.encode(vocab._id_to_token[13])

        # Process START -> OBJ_START -> key -> value
        for token_id in [vocab.start_id, vocab.obj_start_id, name_id, alice_id]:
            token = torch.tensor([token_id])
            mask, state = pda.get_next_token_mask(token, state)

        _, depth, awaiting_value, _, _, _ = state

        # After value: awaiting_value=False
        assert awaiting_value[0].item() is False
        assert depth[0].item() == 1

        # Next valid: keys or OBJ_END
        assert mask[0, name_id].item() is True
        assert mask[0, vocab.obj_end_id].item() is True
        assert mask[0, alice_id].item() is False

    def test_obj_end_closes_root(self, vocab, pda):
        """Test that OBJ_END at depth=1 sets root_closed."""
        state = pda.init_state(batch_size=1, device=torch.device("cpu"))
        name_id = vocab.encode(vocab._id_to_token[10])
        alice_id = vocab.encode(vocab._id_to_token[13])

        # Process START -> OBJ_START -> key -> value -> OBJ_END
        for token_id in [vocab.start_id, vocab.obj_start_id, name_id, alice_id, vocab.obj_end_id]:
            token = torch.tensor([token_id])
            mask, state = pda.get_next_token_mask(token, state)

        _, depth, _, _, root_closed, ended = state

        # After OBJ_END at root: depth=0, root_closed=True
        assert depth[0].item() == 0
        assert root_closed[0].item() is True
        assert ended[0].item() is False

        # Next valid: END only
        assert mask[0, vocab.end_id].item() is True
        assert mask[0, vocab.obj_start_id].item() is False
        assert mask[0, name_id].item() is False

    def test_end_token_transition(self, vocab, pda):
        """Test state after END token."""
        state = pda.init_state(batch_size=1, device=torch.device("cpu"))
        name_id = vocab.encode(vocab._id_to_token[10])
        alice_id = vocab.encode(vocab._id_to_token[13])

        # Process full sequence: START -> OBJ_START -> key -> value -> OBJ_END -> END
        for token_id in [
            vocab.start_id,
            vocab.obj_start_id,
            name_id,
            alice_id,
            vocab.obj_end_id,
            vocab.end_id,
        ]:
            token = torch.tensor([token_id])
            mask, state = pda.get_next_token_mask(token, state)

        _, _, _, _, _, ended = state

        # After END: ended=True
        assert ended[0].item() is True

        # Next valid: PAD only
        assert mask[0, vocab.pad_token_id].item() is True
        assert mask[0, vocab.end_id].item() is False
        assert mask[0, vocab.start_id].item() is False

    def test_nested_object_depth_tracking(self, vocab, pda):
        """Test depth increases/decreases correctly with nested objects."""
        state = pda.init_state(batch_size=1, device=torch.device("cpu"))
        items_id = vocab.encode(vocab._id_to_token[12])

        # START -> OBJ_START -> key -> OBJ_START -> key -> OBJ_START -> OBJ_END -> OBJ_END -> OBJ_END -> END
        tokens = [
            vocab.start_id,  # depth: 0
            vocab.obj_start_id,  # depth: 1
            items_id,  # depth: 1
            vocab.obj_start_id,  # depth: 2
            items_id,  # depth: 2
            vocab.obj_start_id,  # depth: 3
            vocab.obj_end_id,  # depth: 2
            vocab.obj_end_id,  # depth: 1
            vocab.obj_end_id,  # depth: 0
            vocab.end_id,  # depth: 0
        ]
        expected_depths_after = [0, 1, 1, 2, 2, 3, 2, 1, 0, 0]

        for token_id, expected_depth in zip(tokens, expected_depths_after, strict=True):
            token = torch.tensor([token_id])
            _, state = pda.get_next_token_mask(token, state)
            _, depth, _, _, _, _ = state
            assert depth[0].item() == expected_depth, (
                f"After token {token_id}, expected depth {expected_depth}, got {depth[0].item()}"
            )

    def test_array_start_transition(self, vocab, pda):
        """Test state after ARRAY_START token."""
        state = pda.init_state(batch_size=1, device=torch.device("cpu"))

        # Process START -> ARRAY_START
        for token_id in [vocab.start_id, vocab.array_start_id]:
            token = torch.tensor([token_id])
            mask, state = pda.get_next_token_mask(token, state)

        stack, depth, _, _, _, _ = state

        # After ARRAY_START: depth=1, stack[0]=STACK_ARRAY
        assert depth[0].item() == 1
        assert stack[0, 0].item() == 2  # STACK_ARRAY

        # Next valid: values or ARRAY_END
        alice_id = vocab.encode(vocab._id_to_token[13])
        assert mask[0, alice_id].item() is True
        assert mask[0, vocab.obj_start_id].item() is True
        assert mask[0, vocab.array_start_id].item() is True
        assert mask[0, vocab.array_end_id].item() is True
        # Not valid: keys
        name_id = vocab.encode(vocab._id_to_token[10])
        assert mask[0, name_id].item() is False

    def test_array_value_does_not_set_awaiting_value(self, vocab, pda):
        """Test that values in array don't affect awaiting_value."""
        state = pda.init_state(batch_size=1, device=torch.device("cpu"))
        alice_id = vocab.encode(vocab._id_to_token[13])

        # Process START -> ARRAY_START -> value -> value
        for token_id in [vocab.start_id, vocab.array_start_id, alice_id, alice_id]:
            token = torch.tensor([token_id])
            mask, state = pda.get_next_token_mask(token, state)

        _, _, awaiting_value, _, _, _ = state

        # In array context, awaiting_value should be False
        assert awaiting_value[0].item() is False

        # Can still add more values or close
        assert mask[0, alice_id].item() is True
        assert mask[0, vocab.array_end_id].item() is True

    def test_nested_container_as_value_clears_awaiting(self, vocab, pda):
        """Test that OBJ_END/ARRAY_END after nested container clears awaiting_value."""
        state = pda.init_state(batch_size=1, device=torch.device("cpu"))
        name_id = vocab.encode(vocab._id_to_token[10])

        # START -> OBJ_START -> key -> OBJ_START -> OBJ_END
        # After the nested OBJ_END, we're back in parent object with value completed
        for token_id in [
            vocab.start_id,
            vocab.obj_start_id,
            name_id,
            vocab.obj_start_id,
            vocab.obj_end_id,
        ]:
            token = torch.tensor([token_id])
            mask, state = pda.get_next_token_mask(token, state)

        _, depth, awaiting_value, _, _, _ = state

        # After nested object closes, awaiting_value should be False
        assert awaiting_value[0].item() is False
        assert depth[0].item() == 1

        # Next valid: another key or OBJ_END
        assert mask[0, name_id].item() is True
        assert mask[0, vocab.obj_end_id].item() is True

    def test_init_state_from_tokens(self, vocab, pda):
        """Test init_state_from_tokens recreates correct state."""
        name_id = vocab.encode(vocab._id_to_token[10])
        alice_id = vocab.encode(vocab._id_to_token[13])

        # Create a prefix sequence
        prefix = torch.tensor([vocab.start_id, vocab.obj_start_id, name_id, alice_id])

        # Initialize state from tokens
        state = pda.init_state_from_tokens(prefix, batch_size=1, device=torch.device("cpu"))
        stack, depth, awaiting_value, seen_start, root_closed, ended = state

        # Should match state after processing these tokens incrementally
        assert seen_start[0].item() is True
        assert depth[0].item() == 1
        assert stack[0, 0].item() == 1  # STACK_OBJECT
        assert awaiting_value[0].item() is False  # Just saw a value
        assert root_closed[0].item() is False
        assert ended[0].item() is False

    def test_init_state_from_tokens_with_padding(self, vocab, pda):
        """Test init_state_from_tokens skips PAD tokens."""
        name_id = vocab.encode(vocab._id_to_token[10])
        alice_id = vocab.encode(vocab._id_to_token[13])

        # Left-padded sequence
        prefix = torch.tensor(
            [
                vocab.pad_token_id,
                vocab.pad_token_id,
                vocab.start_id,
                vocab.obj_start_id,
                name_id,
                alice_id,
            ]
        )

        state = pda.init_state_from_tokens(prefix, batch_size=1, device=torch.device("cpu"))
        _, depth, awaiting_value, _, _, _ = state

        # Should be same as without padding
        assert depth[0].item() == 1
        assert awaiting_value[0].item() is False

    def test_init_state_from_tokens_replicates_for_batch(self, vocab, pda):
        """Test init_state_from_tokens creates correct batch_size copies."""
        prefix = torch.tensor([vocab.start_id, vocab.obj_start_id])

        state = pda.init_state_from_tokens(prefix, batch_size=4, device=torch.device("cpu"))
        stack, depth, _, seen_start, _, _ = state

        # All batch items should have same state
        assert stack.shape[0] == 4
        assert (depth == 1).all()
        assert seen_start.all()

    def test_incremental_matches_full_computation(self, vocab, pda):
        """Test that incremental get_next_token_mask matches compute_valid_mask."""
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
                ]
            ]
        )

        # Get masks from full computation
        full_masks = pda.compute_valid_mask(tokens)

        # Get masks incrementally
        state = pda.init_state(batch_size=1, device=torch.device("cpu"))
        incremental_masks = []
        for t in range(tokens.shape[1]):
            token = tokens[:, t]
            mask, state = pda.get_next_token_mask(token, state)
            incremental_masks.append(mask)

        incremental_masks = torch.stack(incremental_masks, dim=1)

        # Should match exactly
        assert torch.equal(full_masks, incremental_masks)

    def test_batch_independence(self, vocab, pda):
        """Test that batch items are processed independently."""
        name_id = vocab.encode(vocab._id_to_token[10])
        alice_id = vocab.encode(vocab._id_to_token[13])

        state = pda.init_state(batch_size=2, device=torch.device("cpu"))

        # Process different tokens for each batch item
        # Batch 0: START -> OBJ_START -> key
        # Batch 1: START -> ARRAY_START -> value
        sequences = [
            ([vocab.start_id, vocab.start_id], [0, 0]),  # Both START
            ([vocab.obj_start_id, vocab.array_start_id], [1, 1]),  # OBJ vs ARRAY
            ([name_id, alice_id], [1, 1]),  # key vs value
        ]

        for tokens, expected_depths in sequences:
            token = torch.tensor(tokens)
            _, state = pda.get_next_token_mask(token, state)
            _, depth, _, _, _, _ = state

            assert depth[0].item() == expected_depths[0]
            assert depth[1].item() == expected_depths[1]

        # Final state checks
        stack, _, awaiting_value, _, _, _ = state

        # Batch 0: in object, awaiting_value=True (just saw key)
        assert stack[0, 0].item() == 1  # STACK_OBJECT
        assert awaiting_value[0].item() is True

        # Batch 1: in array, awaiting_value=False
        assert stack[1, 0].item() == 2  # STACK_ARRAY
        assert awaiting_value[1].item() is False


class TestGrammarStructures:
    """Comprehensive tests for different JSON structures.

    Tests flat objects, nested objects, arrays, nested arrays,
    and mixed object/array nesting to ensure PDA handles all
    valid JSON structures correctly.
    """

    def test_flat_object_single_field(self, vocab, pda):
        """Test flat object with single key-value pair: {"name": "Alice"}"""
        state = pda.init_state(batch_size=1, device=torch.device("cpu"))
        name_id = vocab.encode(vocab._id_to_token[10])
        alice_id = vocab.encode(vocab._id_to_token[13])

        tokens = [
            vocab.start_id,
            vocab.obj_start_id,
            name_id,
            alice_id,
            vocab.obj_end_id,
            vocab.end_id,
        ]
        expected_depths = [0, 1, 1, 1, 0, 0]

        for token_id, expected_depth in zip(tokens, expected_depths, strict=True):
            token = torch.tensor([token_id])
            _, state = pda.get_next_token_mask(token, state)
            _, depth, _, _, _, _ = state
            assert depth[0].item() == expected_depth

        _, _, _, _, _, ended = state
        assert ended[0].item() is True

    def test_flat_object_multiple_fields(self, vocab, pda):
        """Test flat object with multiple fields: {"name": "Alice", "age": 42}"""
        state = pda.init_state(batch_size=1, device=torch.device("cpu"))
        name_id = vocab.encode(vocab._id_to_token[10])
        age_id = vocab.encode(vocab._id_to_token[11])
        alice_id = vocab.encode(vocab._id_to_token[13])
        val_42 = vocab.encode(vocab._id_to_token[14])

        tokens = [
            vocab.start_id,
            vocab.obj_start_id,
            name_id,
            alice_id,  # "name": "Alice"
            age_id,
            val_42,  # "age": 42
            vocab.obj_end_id,
            vocab.end_id,
        ]

        awaiting_values = []
        for token_id in tokens:
            token = torch.tensor([token_id])
            _, state = pda.get_next_token_mask(token, state)
            _, _, awaiting_value, _, _, _ = state
            awaiting_values.append(awaiting_value[0].item())

        # awaiting_value should toggle: F, F, T, F, T, F, F, F
        # After key -> True, after value -> False
        assert awaiting_values[2] is True  # After first key
        assert awaiting_values[3] is False  # After first value
        assert awaiting_values[4] is True  # After second key
        assert awaiting_values[5] is False  # After second value

    def test_empty_object(self, vocab, pda):
        """Test empty object: {}"""
        state = pda.init_state(batch_size=1, device=torch.device("cpu"))

        tokens = [vocab.start_id, vocab.obj_start_id, vocab.obj_end_id, vocab.end_id]

        for token_id in tokens:
            token = torch.tensor([token_id])
            mask, state = pda.get_next_token_mask(token, state)

        _, depth, _, _, root_closed, ended = state
        assert depth[0].item() == 0
        assert root_closed[0].item() is True
        assert ended[0].item() is True

    def test_empty_array(self, vocab, pda):
        """Test empty array: []"""
        state = pda.init_state(batch_size=1, device=torch.device("cpu"))

        tokens = [vocab.start_id, vocab.array_start_id, vocab.array_end_id, vocab.end_id]

        for token_id in tokens:
            token = torch.tensor([token_id])
            mask, state = pda.get_next_token_mask(token, state)

        _, depth, _, _, root_closed, ended = state
        assert depth[0].item() == 0
        assert root_closed[0].item() is True
        assert ended[0].item() is True

    def test_flat_array_single_element(self, vocab, pda):
        """Test flat array with single element: ["Alice"]"""
        state = pda.init_state(batch_size=1, device=torch.device("cpu"))
        alice_id = vocab.encode(vocab._id_to_token[13])

        tokens = [vocab.start_id, vocab.array_start_id, alice_id, vocab.array_end_id, vocab.end_id]

        for token_id in tokens:
            token = torch.tensor([token_id])
            _, state = pda.get_next_token_mask(token, state)

        _, _, _, _, _, ended = state
        assert ended[0].item() is True

    def test_flat_array_multiple_elements(self, vocab, pda):
        """Test flat array with multiple elements: ["Alice", 42, true]"""
        state = pda.init_state(batch_size=1, device=torch.device("cpu"))
        alice_id = vocab.encode(vocab._id_to_token[13])
        val_42 = vocab.encode(vocab._id_to_token[14])
        val_true = vocab.encode(vocab._id_to_token[15])

        tokens = [
            vocab.start_id,
            vocab.array_start_id,
            alice_id,
            val_42,
            val_true,
            vocab.array_end_id,
            vocab.end_id,
        ]

        for token_id in tokens:
            token = torch.tensor([token_id])
            mask, state = pda.get_next_token_mask(token, state)

        _, _, _, _, _, ended = state
        assert ended[0].item() is True

    def test_nested_objects_two_levels(self, vocab, pda):
        """Test nested object: {"items": {"name": "Alice"}}"""
        state = pda.init_state(batch_size=1, device=torch.device("cpu"))
        items_id = vocab.encode(vocab._id_to_token[12])
        name_id = vocab.encode(vocab._id_to_token[10])
        alice_id = vocab.encode(vocab._id_to_token[13])

        tokens = [
            vocab.start_id,
            vocab.obj_start_id,  # depth: 0, 1
            items_id,
            vocab.obj_start_id,  # depth: 1, 2
            name_id,
            alice_id,  # depth: 2, 2
            vocab.obj_end_id,
            vocab.obj_end_id,  # depth: 1, 0
            vocab.end_id,  # depth: 0
        ]
        expected_depths = [0, 1, 1, 2, 2, 2, 1, 0, 0]

        for token_id, expected_depth in zip(tokens, expected_depths, strict=True):
            token = torch.tensor([token_id])
            _, state = pda.get_next_token_mask(token, state)
            _, depth, _, _, _, _ = state
            assert depth[0].item() == expected_depth

    def test_nested_objects_three_levels(self, vocab, pda):
        """Test three levels of nesting: {"a": {"b": {"c": "val"}}}"""
        state = pda.init_state(batch_size=1, device=torch.device("cpu"))
        items_id = vocab.encode(vocab._id_to_token[12])
        alice_id = vocab.encode(vocab._id_to_token[13])

        tokens = [
            vocab.start_id,
            vocab.obj_start_id,
            items_id,
            vocab.obj_start_id,
            items_id,
            vocab.obj_start_id,
            items_id,
            alice_id,
            vocab.obj_end_id,
            vocab.obj_end_id,
            vocab.obj_end_id,
            vocab.end_id,
        ]
        expected_depths = [0, 1, 1, 2, 2, 3, 3, 3, 2, 1, 0, 0]

        for token_id, expected_depth in zip(tokens, expected_depths, strict=True):
            token = torch.tensor([token_id])
            _, state = pda.get_next_token_mask(token, state)
            _, depth, _, _, _, _ = state
            assert depth[0].item() == expected_depth

    def test_nested_arrays_two_levels(self, vocab, pda):
        """Test nested arrays: [["Alice", 42]]"""
        state = pda.init_state(batch_size=1, device=torch.device("cpu"))
        alice_id = vocab.encode(vocab._id_to_token[13])
        val_42 = vocab.encode(vocab._id_to_token[14])

        tokens = [
            vocab.start_id,
            vocab.array_start_id,
            vocab.array_start_id,
            alice_id,
            val_42,
            vocab.array_end_id,
            vocab.array_end_id,
            vocab.end_id,
        ]
        expected_depths = [0, 1, 2, 2, 2, 1, 0, 0]

        for token_id, expected_depth in zip(tokens, expected_depths, strict=True):
            token = torch.tensor([token_id])
            _, state = pda.get_next_token_mask(token, state)
            _, depth, _, _, _, _ = state
            assert depth[0].item() == expected_depth

    def test_nested_arrays_three_levels(self, vocab, pda):
        """Test three levels of array nesting: [[["Alice"]]]"""
        state = pda.init_state(batch_size=1, device=torch.device("cpu"))
        alice_id = vocab.encode(vocab._id_to_token[13])

        tokens = [
            vocab.start_id,
            vocab.array_start_id,
            vocab.array_start_id,
            vocab.array_start_id,
            alice_id,
            vocab.array_end_id,
            vocab.array_end_id,
            vocab.array_end_id,
            vocab.end_id,
        ]
        expected_depths = [0, 1, 2, 3, 3, 2, 1, 0, 0]

        for token_id, expected_depth in zip(tokens, expected_depths, strict=True):
            token = torch.tensor([token_id])
            _, state = pda.get_next_token_mask(token, state)
            _, depth, _, _, _, _ = state
            assert depth[0].item() == expected_depth

    def test_array_of_objects(self, vocab, pda):
        """Test array containing objects: [{"name": "Alice"}, {"name": "Bob"}]"""
        state = pda.init_state(batch_size=1, device=torch.device("cpu"))
        name_id = vocab.encode(vocab._id_to_token[10])
        alice_id = vocab.encode(vocab._id_to_token[13])

        tokens = [
            vocab.start_id,
            vocab.array_start_id,
            vocab.obj_start_id,
            name_id,
            alice_id,
            vocab.obj_end_id,
            vocab.obj_start_id,
            name_id,
            alice_id,
            vocab.obj_end_id,
            vocab.array_end_id,
            vocab.end_id,
        ]
        expected_depths = [0, 1, 2, 2, 2, 1, 2, 2, 2, 1, 0, 0]

        for token_id, expected_depth in zip(tokens, expected_depths, strict=True):
            token = torch.tensor([token_id])
            _, state = pda.get_next_token_mask(token, state)
            stack, depth, _, _, _, _ = state
            assert depth[0].item() == expected_depth

            # Verify stack types at depth transitions
            if expected_depth == 1:
                assert stack[0, 0].item() == 2  # STACK_ARRAY
            if expected_depth == 2:
                assert stack[0, 1].item() == 1  # STACK_OBJECT

    def test_object_with_array_value(self, vocab, pda):
        """Test object with array value: {"items": ["Alice", 42]}"""
        state = pda.init_state(batch_size=1, device=torch.device("cpu"))
        items_id = vocab.encode(vocab._id_to_token[12])
        alice_id = vocab.encode(vocab._id_to_token[13])
        val_42 = vocab.encode(vocab._id_to_token[14])

        tokens = [
            vocab.start_id,
            vocab.obj_start_id,
            items_id,
            vocab.array_start_id,
            alice_id,
            val_42,
            vocab.array_end_id,
            vocab.obj_end_id,
            vocab.end_id,
        ]
        expected_depths = [0, 1, 1, 2, 2, 2, 1, 0, 0]

        awaiting_values = []
        for token_id, expected_depth in zip(tokens, expected_depths, strict=True):
            token = torch.tensor([token_id])
            _, state = pda.get_next_token_mask(token, state)
            _, depth, awaiting_value, _, _, _ = state
            awaiting_values.append(awaiting_value[0].item())
            assert depth[0].item() == expected_depth

        # After "items" key, awaiting_value=True
        assert awaiting_values[2] is True
        # After ARRAY_END (which completes the value), awaiting_value=False
        assert awaiting_values[6] is False

    def test_nested_empty_structures(self, vocab, pda):
        """Test nested empty structures: {"items": {}}"""
        state = pda.init_state(batch_size=1, device=torch.device("cpu"))
        items_id = vocab.encode(vocab._id_to_token[12])

        tokens = [
            vocab.start_id,
            vocab.obj_start_id,
            items_id,
            vocab.obj_start_id,
            vocab.obj_end_id,
            vocab.obj_end_id,
            vocab.end_id,
        ]

        for token_id in tokens:
            token = torch.tensor([token_id])
            _, state = pda.get_next_token_mask(token, state)

        _, _, _, _, _, ended = state
        assert ended[0].item() is True

    def test_array_with_empty_objects(self, vocab, pda):
        """Test array containing empty objects: [{}, {}]"""
        state = pda.init_state(batch_size=1, device=torch.device("cpu"))

        tokens = [
            vocab.start_id,
            vocab.array_start_id,
            vocab.obj_start_id,
            vocab.obj_end_id,
            vocab.obj_start_id,
            vocab.obj_end_id,
            vocab.array_end_id,
            vocab.end_id,
        ]

        for token_id in tokens:
            token = torch.tensor([token_id])
            _, state = pda.get_next_token_mask(token, state)

        _, _, _, _, _, ended = state
        assert ended[0].item() is True

    def test_object_with_empty_array(self, vocab, pda):
        """Test object with empty array: {"items": []}"""
        state = pda.init_state(batch_size=1, device=torch.device("cpu"))
        items_id = vocab.encode(vocab._id_to_token[12])

        tokens = [
            vocab.start_id,
            vocab.obj_start_id,
            items_id,
            vocab.array_start_id,
            vocab.array_end_id,
            vocab.obj_end_id,
            vocab.end_id,
        ]

        for token_id in tokens:
            token = torch.tensor([token_id])
            _, state = pda.get_next_token_mask(token, state)

        _, _, _, _, _, ended = state
        assert ended[0].item() is True

    def test_deeply_nested_mixed(self, vocab, pda):
        """Test deeply nested mixed structure: {"a": [{"b": [1]}]}"""
        state = pda.init_state(batch_size=1, device=torch.device("cpu"))
        items_id = vocab.encode(vocab._id_to_token[12])
        val_42 = vocab.encode(vocab._id_to_token[14])

        tokens = [
            vocab.start_id,
            vocab.obj_start_id,  # depth: 0, 1
            items_id,
            vocab.array_start_id,  # depth: 1, 2
            vocab.obj_start_id,  # depth: 3
            items_id,
            vocab.array_start_id,  # depth: 3, 4
            val_42,  # depth: 4
            vocab.array_end_id,  # depth: 3
            vocab.obj_end_id,  # depth: 2
            vocab.array_end_id,  # depth: 1
            vocab.obj_end_id,
            vocab.end_id,  # depth: 0, 0
        ]
        expected_depths = [0, 1, 1, 2, 3, 3, 4, 4, 3, 2, 1, 0, 0]

        for token_id, expected_depth in zip(tokens, expected_depths, strict=True):
            token = torch.tensor([token_id])
            _, state = pda.get_next_token_mask(token, state)
            _, depth, _, _, _, _ = state
            assert depth[0].item() == expected_depth


class TestGrammarEdgeCases:
    """Edge case tests for grammar constraints."""

    def test_max_depth_limit(self, vocab, pda):
        """Test that depth doesn't exceed max_depth."""
        state = pda.init_state(batch_size=1, device=torch.device("cpu"))
        items_id = vocab.encode(vocab._id_to_token[12])

        # Try to nest deeper than max_depth
        token = torch.tensor([vocab.start_id])
        _, state = pda.get_next_token_mask(token, state)

        for i in range(pda.max_depth + 5):
            token = torch.tensor([vocab.obj_start_id])
            _, state = pda.get_next_token_mask(token, state)
            if i < pda.max_depth - 1:
                token = torch.tensor([items_id])
                _, state = pda.get_next_token_mask(token, state)

        _, depth, *_ = state
        # Depth should be clamped at max_depth
        assert depth[0].item() <= pda.max_depth


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
