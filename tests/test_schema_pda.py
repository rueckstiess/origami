"""Tests for schema-aware constraint PDA."""

import pytest
import torch

from origami.constraints.schema_pda import (
    SchemaPDA,
    SchemaState,
)
from origami.tokenizer.path import IndexElement, KeyElement
from origami.tokenizer.vocabulary import KeyToken, ValueToken, Vocabulary


@pytest.fixture
def vocab():
    """Create a vocabulary with various token types for testing."""
    v = Vocabulary()
    # Keys
    v.add_key("name")  # ID 10
    v.add_key("age")  # ID 11
    v.add_key("color")  # ID 12
    v.add_key("items")  # ID 13
    v.add_key("price")  # ID 14
    v.add_key("extra")  # ID 15
    # Values - strings
    v.add_value("Alice")  # ID 16
    v.add_value("Bob")  # ID 17
    v.add_value("red")  # ID 18
    v.add_value("blue")  # ID 19
    # Values - integers
    v.add_value(25)  # ID 20
    v.add_value(30)  # ID 21
    # Values - float
    v.add_value(1.5)  # ID 22
    # Values - boolean
    v.add_value(True)  # ID 23
    v.add_value(False)  # ID 24
    # Values - null
    v.add_value(None)  # ID 25
    v.freeze()
    return v


@pytest.fixture
def simple_schema():
    """Schema with string and integer fields."""
    return {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "enum": ["Alice", "Bob"],
            },
            "age": {
                "type": "integer",
                "minimum": 0,
                "maximum": 150,
            },
        },
        "required": ["name", "age"],
    }


@pytest.fixture
def schema_with_array():
    """Schema with an array field."""
    return {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "price": {"type": "number"},
                    },
                },
                "minItems": 1,
                "maxItems": 5,
            },
        },
    }


@pytest.fixture
def schema_no_additional(vocab):
    """Schema with additionalProperties: false."""
    return {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
        },
        "additionalProperties": False,
    }


class TestSchemaPDAInit:
    def test_basic_init(self, vocab, simple_schema):
        pda = SchemaPDA(simple_schema, vocab)
        assert pda.num_paths > 0
        assert pda._vocab_size == vocab.size

    def test_mask_table_shape(self, vocab, simple_schema):
        pda = SchemaPDA(simple_schema, vocab)
        # Row 0 = all-True, plus one row per compiled path
        assert pda.mask_table.shape == (pda.num_paths + 1, vocab.size)

    def test_row_zero_all_true(self, vocab, simple_schema):
        pda = SchemaPDA(simple_schema, vocab)
        assert pda.mask_table[0].all()

    def test_schema_preserved(self, vocab, simple_schema):
        pda = SchemaPDA(simple_schema, vocab)
        assert pda.schema is simple_schema


class TestTypeToTokenMapping:
    def test_string_tokens(self, vocab, simple_schema):
        pda = SchemaPDA(simple_schema, vocab)
        string_ids = pda._type_to_token_ids["string"]
        # Should include Alice, Bob, red, blue
        alice_id = vocab.encode(ValueToken("Alice"))
        bob_id = vocab.encode(ValueToken("Bob"))
        assert alice_id in string_ids
        assert bob_id in string_ids

    def test_integer_tokens(self, vocab, simple_schema):
        pda = SchemaPDA(simple_schema, vocab)
        int_ids = pda._type_to_token_ids["integer"]
        assert vocab.encode(ValueToken(25)) in int_ids
        assert vocab.encode(ValueToken(30)) in int_ids
        # NUM token also valid for integer
        assert vocab.num_token_id in int_ids

    def test_number_includes_int_and_float(self, vocab, simple_schema):
        pda = SchemaPDA(simple_schema, vocab)
        num_ids = pda._type_to_token_ids["number"]
        assert vocab.encode(ValueToken(25)) in num_ids
        assert vocab.encode(ValueToken(1.5)) in num_ids
        assert vocab.num_token_id in num_ids

    def test_boolean_tokens(self, vocab, simple_schema):
        pda = SchemaPDA(simple_schema, vocab)
        bool_ids = pda._type_to_token_ids["boolean"]
        assert vocab.encode(ValueToken(True)) in bool_ids
        assert vocab.encode(ValueToken(False)) in bool_ids
        # int should NOT be in boolean
        assert vocab.encode(ValueToken(25)) not in bool_ids

    def test_null_tokens(self, vocab, simple_schema):
        pda = SchemaPDA(simple_schema, vocab)
        null_ids = pda._type_to_token_ids["null"]
        assert vocab.encode(ValueToken(None)) in null_ids

    def test_object_type(self, vocab, simple_schema):
        pda = SchemaPDA(simple_schema, vocab)
        assert pda._type_to_token_ids["object"] == {vocab.obj_start_id}

    def test_array_type(self, vocab, simple_schema):
        pda = SchemaPDA(simple_schema, vocab)
        assert pda._type_to_token_ids["array"] == {vocab.array_start_id}


class TestPathNormalization:
    def test_empty_path(self, vocab, simple_schema):
        pda = SchemaPDA(simple_schema, vocab)
        assert pda.normalize_path(()) == ""

    def test_single_key(self, vocab, simple_schema):
        pda = SchemaPDA(simple_schema, vocab)
        path = (KeyElement("name"),)
        assert pda.normalize_path(path) == "name"

    def test_nested_keys(self, vocab, simple_schema):
        pda = SchemaPDA(simple_schema, vocab)
        path = (KeyElement("user"), KeyElement("name"))
        assert pda.normalize_path(path) == "user.name"

    def test_array_index_wildcard(self, vocab, simple_schema):
        pda = SchemaPDA(simple_schema, vocab)
        path = (KeyElement("items"), IndexElement(0))
        assert pda.normalize_path(path) == "items.*"

    def test_array_index_wildcard_any_index(self, vocab, simple_schema):
        pda = SchemaPDA(simple_schema, vocab)
        path0 = (KeyElement("items"), IndexElement(0))
        path5 = (KeyElement("items"), IndexElement(5))
        assert pda.normalize_path(path0) == pda.normalize_path(path5)

    def test_nested_in_array(self, vocab, simple_schema):
        pda = SchemaPDA(simple_schema, vocab)
        path = (KeyElement("items"), IndexElement(0), KeyElement("price"))
        assert pda.normalize_path(path) == "items.*.price"


class TestSchemaCompilation:
    def test_compiled_paths(self, vocab, simple_schema):
        pda = SchemaPDA(simple_schema, vocab)
        paths = pda.get_compiled_paths()
        # Root + name + age
        assert "" in paths
        assert "name" in paths
        assert "age" in paths

    def test_compiled_paths_with_array(self, vocab, schema_with_array):
        pda = SchemaPDA(schema_with_array, vocab)
        paths = pda.get_compiled_paths()
        assert "" in paths
        assert "items" in paths
        assert "items.*" in paths
        assert "items.*.name" in paths
        assert "items.*.price" in paths

    def test_constraints_stored(self, vocab, simple_schema):
        pda = SchemaPDA(simple_schema, vocab)
        age_constraints = pda.get_constraints("age")
        assert age_constraints is not None
        assert age_constraints.minimum == 0
        assert age_constraints.maximum == 150

    def test_required_key_ids(self, vocab, simple_schema):
        pda = SchemaPDA(simple_schema, vocab)
        root_constraints = pda.get_constraints("")
        assert root_constraints is not None
        assert root_constraints.required_key_ids is not None
        name_id = vocab.encode(KeyToken("name"))
        age_id = vocab.encode(KeyToken("age"))
        assert name_id in root_constraints.required_key_ids
        assert age_id in root_constraints.required_key_ids

    def test_array_bounds(self, vocab, schema_with_array):
        pda = SchemaPDA(schema_with_array, vocab)
        items_constraints = pda.get_constraints("items")
        assert items_constraints is not None
        assert items_constraints.min_items == 1
        assert items_constraints.max_items == 5


class TestValueMasks:
    def test_string_enum_mask(self, vocab, simple_schema):
        """Name field with enum ["Alice", "Bob"] should only allow those values."""
        pda = SchemaPDA(simple_schema, vocab)
        mask = pda.get_mask_for_schema_path("name")

        alice_id = vocab.encode(ValueToken("Alice"))
        bob_id = vocab.encode(ValueToken("Bob"))
        red_id = vocab.encode(ValueToken("red"))
        int_id = vocab.encode(ValueToken(25))

        # Enum values allowed
        assert mask[alice_id].item()
        assert mask[bob_id].item()
        # Non-enum values blocked
        assert not mask[red_id].item()
        assert not mask[int_id].item()

        # UNK_VALUE always allowed
        assert mask[vocab.unk_value_id].item()

        # Grammar/key tokens pass through (grammar handles them)
        assert mask[vocab.start_id].item()
        assert mask[vocab.end_id].item()
        assert mask[vocab.obj_end_id].item()

    def test_integer_type_mask(self, vocab, simple_schema):
        """Age field with type integer should allow integer values + NUM."""
        pda = SchemaPDA(simple_schema, vocab)
        mask = pda.get_mask_for_schema_path("age")

        # Integer values allowed
        assert mask[vocab.encode(ValueToken(25))].item()
        assert mask[vocab.encode(ValueToken(30))].item()
        # NUM allowed for integer
        assert mask[vocab.num_token_id].item()

        # String values blocked
        assert not mask[vocab.encode(ValueToken("Alice"))].item()
        # OBJ_START blocked (not integer)
        assert not mask[vocab.obj_start_id].item()

    def test_number_type_mask(self, vocab, schema_with_array):
        """Price field with type number should allow int, float, and NUM."""
        pda = SchemaPDA(schema_with_array, vocab)
        mask = pda.get_mask_for_schema_path("items.*.price")

        assert mask[vocab.encode(ValueToken(25))].item()
        assert mask[vocab.encode(ValueToken(1.5))].item()
        assert mask[vocab.num_token_id].item()

        # String blocked
        assert not mask[vocab.encode(ValueToken("Alice"))].item()

    def test_object_type_mask(self, vocab):
        """Object type should allow only OBJ_START among values."""
        schema = {
            "type": "object",
            "properties": {
                "nested": {"type": "object"},
            },
        }
        pda = SchemaPDA(schema, vocab)
        mask = pda.get_mask_for_schema_path("nested")

        assert mask[vocab.obj_start_id].item()
        assert not mask[vocab.array_start_id].item()
        assert not mask[vocab.encode(ValueToken("Alice"))].item()

    def test_array_type_mask(self, vocab):
        """Array type should allow only ARRAY_START among values."""
        schema = {
            "type": "object",
            "properties": {
                "items": {"type": "array"},
            },
        }
        pda = SchemaPDA(schema, vocab)
        mask = pda.get_mask_for_schema_path("items")

        assert mask[vocab.array_start_id].item()
        assert not mask[vocab.obj_start_id].item()
        assert not mask[vocab.encode(ValueToken("Alice"))].item()

    def test_multi_type_mask(self, vocab):
        """Multiple types should allow tokens from all types."""
        schema = {
            "type": "object",
            "properties": {
                "val": {"type": ["string", "null"]},
            },
        }
        pda = SchemaPDA(schema, vocab)
        mask = pda.get_mask_for_schema_path("val")

        assert mask[vocab.encode(ValueToken("Alice"))].item()
        assert mask[vocab.encode(ValueToken(None))].item()
        # Integer blocked
        assert not mask[vocab.encode(ValueToken(25))].item()

    def test_no_constraint_path(self, vocab, simple_schema):
        """Unknown paths should return all-True mask."""
        pda = SchemaPDA(simple_schema, vocab)
        mask = pda.get_mask_for_schema_path("unknown_field")
        assert mask.all()


class TestKeyMasks:
    def test_additional_properties_false(self, vocab, schema_no_additional):
        """With additionalProperties: false, only defined keys should be allowed."""
        pda = SchemaPDA(schema_no_additional, vocab)
        mask = pda.get_mask_for_schema_path("")  # root object

        name_id = vocab.encode(KeyToken("name"))
        age_id = vocab.encode(KeyToken("age"))
        color_id = vocab.encode(KeyToken("color"))
        extra_id = vocab.encode(KeyToken("extra"))

        # Defined keys allowed
        assert mask[name_id].item()
        assert mask[age_id].item()
        # Undefined keys blocked (including UNK_KEY)
        assert not mask[color_id].item()
        assert not mask[extra_id].item()
        assert not mask[vocab.unk_key_id].item()

        # Non-key tokens pass through
        assert mask[vocab.obj_end_id].item()
        assert mask[vocab.start_id].item()

    def test_additional_properties_default_true(self, vocab, simple_schema):
        """Without additionalProperties: false, all keys should be allowed."""
        pda = SchemaPDA(simple_schema, vocab)
        mask = pda.get_mask_for_schema_path("")  # root object

        # All keys should be allowed
        for kid in vocab.get_all_key_ids():
            assert mask[kid].item()


class TestCombinedMasks:
    def test_value_and_key_combined(self, vocab, schema_no_additional):
        """Root mask should restrict both keys and values."""
        pda = SchemaPDA(schema_no_additional, vocab)
        mask = pda.get_mask_for_schema_path("")  # root object

        # Value constraint: type "object" → OBJ_START allowed
        assert mask[vocab.obj_start_id].item()
        # String values blocked (root type is "object", not "string")
        assert not mask[vocab.encode(ValueToken("Alice"))].item()

        # Key constraint: only "name" and "age" allowed
        name_id = vocab.encode(KeyToken("name"))
        color_id = vocab.encode(KeyToken("color"))
        assert mask[name_id].item()
        assert not mask[color_id].item()


class TestMaskTableGather:
    def test_gather_basic(self, vocab, simple_schema):
        pda = SchemaPDA(simple_schema, vocab)

        # Create indices: first position → "name", second → "age"
        name_idx = pda._path_to_index["name"]
        age_idx = pda._path_to_index["age"]
        indices = torch.tensor([[name_idx, age_idx]])  # (1, 2)

        result = pda.gather_masks(indices)
        assert result.shape == (1, 2, vocab.size)

        # Verify row content matches
        assert torch.equal(result[0, 0], pda.mask_table[name_idx])
        assert torch.equal(result[0, 1], pda.mask_table[age_idx])

    def test_gather_default_index(self, vocab, simple_schema):
        pda = SchemaPDA(simple_schema, vocab)

        # Index 0 should give all-True
        indices = torch.tensor([[0]])
        result = pda.gather_masks(indices)
        assert result[0, 0].all()

    def test_gather_batch(self, vocab, simple_schema):
        pda = SchemaPDA(simple_schema, vocab)

        name_idx = pda._path_to_index["name"]
        age_idx = pda._path_to_index["age"]
        indices = torch.tensor(
            [
                [name_idx, 0],
                [0, age_idx],
            ]
        )  # (2, 2)

        result = pda.gather_masks(indices)
        assert result.shape == (2, 2, vocab.size)


class TestResolveMaskIndices:
    def test_simple_sequence(self, vocab, simple_schema):
        """Test mask index resolution for a simple object sequence."""
        pda = SchemaPDA(simple_schema, vocab)

        # Sequence: START OBJ_START KEY(name) VALUE(Alice) OBJ_END END
        # Paths:    ()    ()         ()        (name,)      ()      ()
        paths = [
            (),
            (),
            (),
            (KeyElement("name"),),
            (),
            (),
        ]

        indices = pda.resolve_mask_indices(
            batch_paths=[paths],
            batch_size=1,
            seq_len=6,
            start_positions=[0],
        )

        assert indices.shape == (1, 6)

        # mask[0] constrains token at pos 1 (OBJ_START). Next path is ().
        # Path "" is the root object path.
        assert indices[0, 0].item() == pda._path_to_index.get("", 0)

        # mask[2] constrains token at pos 3 (VALUE). Next path is (name,).
        # Should map to "name" schema path.
        assert indices[0, 2].item() == pda._path_to_index["name"]

        # mask[5] is the last position — no next token, should be 0
        assert indices[0, 5].item() == 0

    def test_left_padded(self, vocab, simple_schema):
        """Test that left-padding offset is handled correctly."""
        pda = SchemaPDA(simple_schema, vocab)

        paths = [(), (), (KeyElement("name"),)]
        indices = pda.resolve_mask_indices(
            batch_paths=[paths],
            batch_size=1,
            seq_len=5,
            start_positions=[2],  # 2 positions of padding
        )

        # Positions 0, 1 should be 0 (padding)
        assert indices[0, 0].item() == 0
        assert indices[0, 1].item() == 0

        # Position 2 (first real token): constrains next token
        # Next path is () → root
        assert indices[0, 2].item() == pda._path_to_index.get("", 0)


class TestMaskTableDeviceCache:
    def test_device_cache(self, vocab, simple_schema):
        pda = SchemaPDA(simple_schema, vocab)
        device = torch.device("cpu")

        table1 = pda.get_mask_table_on_device(device)
        table2 = pda.get_mask_table_on_device(device)
        assert table1 is table2  # Same cached object


class TestSchemaState:
    def test_push_pop_object(self):
        state = SchemaState()
        state.push_object()
        assert len(state.container_stack) == 1
        assert state.container_stack[-1] == "object"
        assert len(state.seen_keys) == 1

        state.pop_context()
        assert len(state.container_stack) == 0
        assert len(state.seen_keys) == 0

    def test_push_pop_array(self):
        state = SchemaState()
        state.push_array()
        assert len(state.container_stack) == 1
        assert state.container_stack[-1] == "array"
        assert len(state.array_counts) == 1

        state.pop_context()
        assert len(state.container_stack) == 0
        assert len(state.array_counts) == 0

    def test_record_key(self):
        state = SchemaState()
        state.push_object()
        state.record_key(10)
        state.record_key(11)
        assert state.seen_keys[-1] == {10, 11}

    def test_increment_array_count(self):
        state = SchemaState()
        state.push_array()
        assert state.array_counts[-1] == 0
        state.increment_array_count()
        assert state.array_counts[-1] == 1
        state.increment_array_count()
        assert state.array_counts[-1] == 2

    def test_nested_contexts(self):
        state = SchemaState()
        state.push_object()  # root object
        state.push_array()  # array inside object
        state.push_object()  # object inside array

        assert len(state.container_stack) == 3
        assert state.container_stack == ["object", "array", "object"]

        state.pop_context()  # pop inner object
        assert len(state.container_stack) == 2
        state.pop_context()  # pop array
        assert len(state.container_stack) == 1
        state.pop_context()  # pop root object
        assert len(state.container_stack) == 0

    def test_clone(self):
        state = SchemaState()
        state.push_object()
        state.record_key(10)
        state.push_array()
        state.increment_array_count()

        clone = state.clone()
        assert clone.container_stack == state.container_stack
        assert clone.seen_keys == state.seen_keys
        assert clone.array_counts == state.array_counts

        # Modifications to clone don't affect original
        clone.increment_array_count()
        assert clone.array_counts[-1] != state.array_counts[-1]


class TestInitStateFromTokens:
    def test_replay_simple(self, vocab, simple_schema):
        pda = SchemaPDA(simple_schema, vocab)
        name_id = vocab.encode(KeyToken("name"))
        alice_id = vocab.encode(ValueToken("Alice"))

        tokens = [
            vocab.start_id,
            vocab.obj_start_id,
            name_id,
            alice_id,
            vocab.obj_end_id,
            vocab.end_id,
        ]
        state = pda.init_state_from_tokens(tokens, vocab)

        # After full sequence, should be empty
        assert len(state.container_stack) == 0

    def test_replay_partial(self, vocab, simple_schema):
        pda = SchemaPDA(simple_schema, vocab)
        name_id = vocab.encode(KeyToken("name"))

        # Partial: START OBJ_START KEY(name)
        tokens = [vocab.start_id, vocab.obj_start_id, name_id]
        state = pda.init_state_from_tokens(tokens, vocab)

        # Should be inside object with "name" key seen
        assert len(state.container_stack) == 1
        assert state.container_stack[-1] == "object"
        assert name_id in state.seen_keys[-1]

    def test_replay_skips_pad(self, vocab, simple_schema):
        pda = SchemaPDA(simple_schema, vocab)

        tokens = [vocab.pad_token_id, vocab.pad_token_id, vocab.start_id, vocab.obj_start_id]
        state = pda.init_state_from_tokens(tokens, vocab)

        assert len(state.container_stack) == 1


class TestGrammarSchemaIntersection:
    """Test that schema mask correctly intersects with grammar mask."""

    def test_value_position_intersection(self, vocab, simple_schema):
        """At a value position, grammar allows values+containers.
        Schema for string field should restrict to string values only.
        Intersection: only string values."""
        from origami.constraints.json_grammar import JSONGrammarPDA

        grammar_pda = JSONGrammarPDA(vocab)
        schema_pda = SchemaPDA(simple_schema, vocab)

        # Grammar mask for value position (after a key in an object)
        # Set up grammar state: inside object, awaiting value
        state = grammar_pda.init_state(1, torch.device("cpu"))
        # Feed: START, OBJ_START, KEY(name)
        name_id = vocab.encode(KeyToken("name"))
        for token in [vocab.start_id, vocab.obj_start_id, name_id]:
            grammar_mask, state = grammar_pda.get_next_token_mask(torch.tensor([token]), state)

        # grammar_mask should allow all values + containers
        assert grammar_mask[0, vocab.encode(ValueToken("Alice"))].item()
        assert grammar_mask[0, vocab.encode(ValueToken(25))].item()
        assert grammar_mask[0, vocab.obj_start_id].item()

        # Schema mask for "name" (type: string, enum: [Alice, Bob])
        schema_mask = schema_pda.get_mask_for_schema_path("name")

        # Intersection
        combined = grammar_mask[0] & schema_mask
        assert combined[vocab.encode(ValueToken("Alice"))].item()
        assert combined[vocab.encode(ValueToken("Bob"))].item()
        assert not combined[vocab.encode(ValueToken(25))].item()
        assert not combined[vocab.obj_start_id].item()

    def test_key_position_intersection(self, vocab, schema_no_additional):
        """At a key position, grammar allows keys+OBJ_END.
        Schema with additionalProperties: false should restrict keys.
        Intersection: only allowed keys + OBJ_END."""
        from origami.constraints.json_grammar import JSONGrammarPDA

        grammar_pda = JSONGrammarPDA(vocab)
        schema_pda = SchemaPDA(schema_no_additional, vocab)

        # Grammar state: inside object, not awaiting value (key position)
        state = grammar_pda.init_state(1, torch.device("cpu"))
        for token in [vocab.start_id, vocab.obj_start_id]:
            grammar_mask, state = grammar_pda.get_next_token_mask(torch.tensor([token]), state)

        # grammar_mask should allow all keys + OBJ_END
        assert grammar_mask[0, vocab.encode(KeyToken("name"))].item()
        assert grammar_mask[0, vocab.encode(KeyToken("color"))].item()
        assert grammar_mask[0, vocab.obj_end_id].item()

        # Schema mask for root (additionalProperties: false)
        schema_mask = schema_pda.get_mask_for_schema_path("")

        # Intersection
        combined = grammar_mask[0] & schema_mask
        assert combined[vocab.encode(KeyToken("name"))].item()
        assert combined[vocab.encode(KeyToken("age"))].item()
        assert not combined[vocab.encode(KeyToken("color"))].item()
        assert not combined[vocab.encode(KeyToken("extra"))].item()
        assert combined[vocab.obj_end_id].item()


class TestUniqueItems:
    def test_unique_items_constraint_stored(self, vocab):
        """uniqueItems: true should be stored in compiled constraints."""
        schema = {
            "type": "object",
            "properties": {
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "uniqueItems": True,
                },
            },
        }
        pda = SchemaPDA(schema, vocab)
        constraints = pda.get_constraints("tags")
        assert constraints is not None
        assert constraints.unique_items is True

    def test_unique_items_false_by_default(self, vocab, simple_schema):
        """unique_items should be False when not specified."""
        pda = SchemaPDA(simple_schema, vocab)
        constraints = pda.get_constraints("name")
        assert constraints is not None
        assert constraints.unique_items is False

    def test_seen_array_values_tracked(self, vocab):
        """SchemaState should track seen values in arrays."""
        schema = {
            "type": "object",
            "properties": {
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "uniqueItems": True,
                },
            },
        }
        pda = SchemaPDA(schema, vocab)

        # Build a token sequence: START OBJ_START KEY(items) ARRAY_START VALUE(Alice)
        # Using the "items" key since that's what's in vocab — but let's use
        # actual array start/value tokens
        alice_id = vocab.encode(ValueToken("Alice"))
        bob_id = vocab.encode(ValueToken("Bob"))

        state = SchemaState()
        pda.update_state(vocab.start_id, state)
        pda.update_state(vocab.obj_start_id, state)
        pda.update_state(vocab.array_start_id, state)
        pda.update_state(alice_id, state)

        # Alice should be in seen_array_values
        assert alice_id in state.seen_array_values[-1]
        assert bob_id not in state.seen_array_values[-1]
        assert state.array_counts[-1] == 1

        pda.update_state(bob_id, state)
        assert bob_id in state.seen_array_values[-1]
        assert state.array_counts[-1] == 2

    def test_seen_array_values_cleared_on_pop(self, vocab):
        """seen_array_values should be cleared when array context pops."""
        schema = {"type": "object", "properties": {}}
        pda = SchemaPDA(schema, vocab)

        state = SchemaState()
        pda.update_state(vocab.array_start_id, state)

        alice_id = vocab.encode(ValueToken("Alice"))
        pda.update_state(alice_id, state)
        assert len(state.seen_array_values) == 1
        assert alice_id in state.seen_array_values[-1]

        pda.update_state(vocab.array_end_id, state)
        assert len(state.seen_array_values) == 0

    def test_init_state_from_tokens_tracks_seen_values(self, vocab):
        """init_state_from_tokens should populate seen_array_values."""
        schema = {"type": "object", "properties": {}}
        pda = SchemaPDA(schema, vocab)

        alice_id = vocab.encode(ValueToken("Alice"))
        bob_id = vocab.encode(ValueToken("Bob"))

        tokens = [
            vocab.start_id,
            vocab.obj_start_id,
            vocab.array_start_id,
            alice_id,
            bob_id,
        ]
        state = pda.init_state_from_tokens(tokens, vocab)

        assert len(state.seen_array_values) == 1
        assert alice_id in state.seen_array_values[-1]
        assert bob_id in state.seen_array_values[-1]

    def test_clone_preserves_seen_values(self):
        """Clone should deep-copy seen_array_values."""
        state = SchemaState()
        state.push_array()
        state.record_array_value(42)

        clone = state.clone()
        assert clone.seen_array_values[-1] == {42}

        clone.record_array_value(99)
        assert 99 in clone.seen_array_values[-1]
        assert 99 not in state.seen_array_values[-1]

    def test_unique_items_in_summary(self, vocab):
        """Summary should display uniqueItems constraint."""
        schema = {
            "type": "object",
            "properties": {
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "uniqueItems": True,
                },
            },
        }
        pda = SchemaPDA(schema, vocab)
        summary = pda.summary()
        assert "uniqueItems" in summary


class TestArrayOfObjectsCounting:
    """Test that array element counting is correct for arrays of objects.

    Bug fix: Previously, every value token incremented the parent array's count,
    even values inside nested objects. Now only top-level elements (OBJ_START,
    ARRAY_START, primitive values directly in array) increment the count.
    """

    def test_object_in_array_counts_as_one(self, vocab):
        """An object in an array should count as one element, regardless of how
        many fields it contains."""
        schema = {"type": "object", "properties": {}}
        pda = SchemaPDA(schema, vocab)

        state = SchemaState()
        # Enter root object and then an array
        pda.update_state(vocab.start_id, state)
        pda.update_state(vocab.obj_start_id, state)
        pda.update_state(vocab.array_start_id, state)
        assert state.array_counts[-1] == 0

        # Start first object in array -> counts as 1 element
        # array_counts has one entry (for the one array), objects don't add entries
        pda.update_state(vocab.obj_start_id, state)
        assert state.array_counts[-1] == 1

        # Add fields inside the object — these should NOT increment array count
        name_id = vocab.encode(KeyToken("name"))
        alice_id = vocab.encode(ValueToken("Alice"))
        age_id = vocab.encode(KeyToken("age"))
        val_25 = vocab.encode(ValueToken(25))

        pda.update_state(name_id, state)
        pda.update_state(alice_id, state)
        pda.update_state(age_id, state)
        pda.update_state(val_25, state)
        # Array count should still be 1 (values are inside nested object)
        assert state.array_counts[-1] == 1

        # Close the object
        pda.update_state(vocab.obj_end_id, state)
        assert state.array_counts[-1] == 1  # back to array context

        # Start second object -> count becomes 2
        pda.update_state(vocab.obj_start_id, state)
        assert state.array_counts[-1] == 2

        pda.update_state(name_id, state)
        pda.update_state(alice_id, state)
        pda.update_state(vocab.obj_end_id, state)

        # Still 2
        assert state.array_counts[-1] == 2

    def test_primitive_array_counts_correctly(self, vocab):
        """Primitive values directly in an array should each count as one element."""
        schema = {"type": "object", "properties": {}}
        pda = SchemaPDA(schema, vocab)

        state = SchemaState()
        pda.update_state(vocab.array_start_id, state)
        assert state.array_counts[-1] == 0

        alice_id = vocab.encode(ValueToken("Alice"))
        bob_id = vocab.encode(ValueToken("Bob"))

        pda.update_state(alice_id, state)
        assert state.array_counts[-1] == 1

        pda.update_state(bob_id, state)
        assert state.array_counts[-1] == 2

    def test_nested_array_in_array_counts_as_one(self, vocab):
        """A sub-array inside an array should count as one element."""
        schema = {"type": "object", "properties": {}}
        pda = SchemaPDA(schema, vocab)

        state = SchemaState()
        pda.update_state(vocab.array_start_id, state)  # outer array
        assert state.array_counts[-1] == 0

        # Start inner array -> counts as 1 in outer
        pda.update_state(vocab.array_start_id, state)
        assert state.array_counts[-2] == 1  # outer array count

        # Values in inner array don't affect outer count
        alice_id = vocab.encode(ValueToken("Alice"))
        pda.update_state(alice_id, state)
        assert state.array_counts[-2] == 1  # outer unchanged
        assert state.array_counts[-1] == 1  # inner has 1

        pda.update_state(vocab.array_end_id, state)  # close inner
        assert state.array_counts[-1] == 1  # outer still 1

    def test_init_state_array_of_objects_count(self, vocab):
        """init_state_from_tokens should correctly count objects in arrays."""
        schema = {"type": "object", "properties": {}}
        pda = SchemaPDA(schema, vocab)

        name_id = vocab.encode(KeyToken("name"))
        alice_id = vocab.encode(ValueToken("Alice"))

        tokens = [
            vocab.start_id,
            vocab.obj_start_id,  # root object
            vocab.array_start_id,  # array
            vocab.obj_start_id,  # first object in array
            name_id,
            alice_id,
            vocab.obj_end_id,  # close first object
            vocab.obj_start_id,  # second object in array
            name_id,
            alice_id,
            vocab.obj_end_id,  # close second object
        ]
        state = pda.init_state_from_tokens(tokens, vocab)

        # Should be inside the array with 2 objects counted
        assert state.container_stack == ["object", "array"]
        assert state.array_counts[-1] == 2


class TestUNKTokenHandling:
    """Test that UNK_VALUE and UNK_KEY tokens are handled correctly in state tracking."""

    def test_unk_value_counted_in_array(self, vocab):
        """UNK_VALUE should count as an array element when directly in array."""
        schema = {"type": "object", "properties": {}}
        pda = SchemaPDA(schema, vocab)

        state = SchemaState()
        pda.update_state(vocab.array_start_id, state)
        pda.update_state(vocab.unk_value_id, state)
        assert state.array_counts[-1] == 1

    def test_unk_value_not_counted_in_object(self, vocab):
        """UNK_VALUE inside an object should not increment the parent array count."""
        schema = {"type": "object", "properties": {}}
        pda = SchemaPDA(schema, vocab)

        state = SchemaState()
        pda.update_state(vocab.array_start_id, state)
        pda.update_state(vocab.obj_start_id, state)  # object in array (count=1)
        # Record a key then UNK_VALUE inside the object
        name_id = vocab.encode(KeyToken("name"))
        pda.update_state(name_id, state)
        pda.update_state(vocab.unk_value_id, state)

        # Array count should be 1 (the object), not 2
        # array_counts only has entries for arrays, not objects
        assert state.array_counts[-1] == 1

    def test_unk_key_recorded_in_seen_keys(self, vocab):
        """UNK_KEY should be recorded in seen_keys like a normal key."""
        schema = {"type": "object", "properties": {}}
        pda = SchemaPDA(schema, vocab)

        state = SchemaState()
        pda.update_state(vocab.obj_start_id, state)
        pda.update_state(vocab.unk_key_id, state)
        assert vocab.unk_key_id in state.seen_keys[-1]

    def test_unk_value_tracked_for_unique_items(self, vocab):
        """UNK_VALUE in array should be tracked in seen_array_values."""
        schema = {"type": "object", "properties": {}}
        pda = SchemaPDA(schema, vocab)

        state = SchemaState()
        pda.update_state(vocab.array_start_id, state)
        pda.update_state(vocab.unk_value_id, state)
        assert vocab.unk_value_id in state.seen_array_values[-1]

    def test_init_state_with_unk_tokens(self, vocab):
        """init_state_from_tokens should handle UNK_VALUE and UNK_KEY correctly."""
        schema = {"type": "object", "properties": {}}
        pda = SchemaPDA(schema, vocab)

        tokens = [
            vocab.start_id,
            vocab.obj_start_id,
            vocab.unk_key_id,  # acts like a key
            vocab.unk_value_id,  # acts like a value
        ]
        state = pda.init_state_from_tokens(tokens, vocab)

        # UNK_KEY should be in seen_keys
        assert vocab.unk_key_id in state.seen_keys[-1]
        # Should be in object context
        assert state.container_stack == ["object"]


class TestUNKFlags:
    """Test configurable allow_unk_key and allow_unk_value flags."""

    def test_default_allow_unk_key_false(self, vocab, simple_schema):
        """By default, allow_unk_key is False."""
        pda = SchemaPDA(simple_schema, vocab)
        assert pda._allow_unk_key is False

    def test_default_allow_unk_value_true(self, vocab, simple_schema):
        """By default, allow_unk_value is True."""
        pda = SchemaPDA(simple_schema, vocab)
        assert pda._allow_unk_value is True

    def test_unk_value_blocked_with_enum(self, vocab, simple_schema):
        """With allow_unk_value=False, UNK_VALUE is blocked at enum positions."""
        pda = SchemaPDA(simple_schema, vocab, allow_unk_value=False)
        mask = pda.get_mask_for_schema_path("name")

        # Enum values still allowed
        assert mask[vocab.encode(ValueToken("Alice"))].item()
        assert mask[vocab.encode(ValueToken("Bob"))].item()
        # UNK_VALUE blocked
        assert not mask[vocab.unk_value_id].item()

    def test_unk_value_blocked_with_type(self, vocab, simple_schema):
        """With allow_unk_value=False, UNK_VALUE is blocked at type-constrained positions."""
        pda = SchemaPDA(simple_schema, vocab, allow_unk_value=False)
        mask = pda.get_mask_for_schema_path("age")

        # Integer values and NUM still allowed
        assert mask[vocab.encode(ValueToken(25))].item()
        assert mask[vocab.num_token_id].item()
        # UNK_VALUE blocked
        assert not mask[vocab.unk_value_id].item()

    def test_unk_value_allowed_with_enum(self, vocab, simple_schema):
        """With allow_unk_value=True (default), UNK_VALUE is allowed at enum positions."""
        pda = SchemaPDA(simple_schema, vocab, allow_unk_value=True)
        mask = pda.get_mask_for_schema_path("name")
        assert mask[vocab.unk_value_id].item()

    def test_unk_key_allowed_with_additional_properties_false(self, vocab, schema_no_additional):
        """With allow_unk_key=True, UNK_KEY is allowed even with additionalProperties: false."""
        pda = SchemaPDA(schema_no_additional, vocab, allow_unk_key=True)
        mask = pda.get_mask_for_schema_path("")  # root object

        # UNK_KEY allowed
        assert mask[vocab.unk_key_id].item()
        # Defined keys still allowed
        assert mask[vocab.encode(KeyToken("name"))].item()
        assert mask[vocab.encode(KeyToken("age"))].item()
        # Other specific keys still blocked
        assert not mask[vocab.encode(KeyToken("color"))].item()
        assert not mask[vocab.encode(KeyToken("extra"))].item()

    def test_unk_key_blocked_by_default_with_additional_properties_false(
        self, vocab, schema_no_additional
    ):
        """With default allow_unk_key=False, UNK_KEY is blocked when additionalProperties: false."""
        pda = SchemaPDA(schema_no_additional, vocab)
        mask = pda.get_mask_for_schema_path("")
        assert not mask[vocab.unk_key_id].item()

    def test_strict_mode_both_blocked(self, vocab, simple_schema):
        """With both flags False (strict/generation mode), neither UNK is allowed."""
        # Need a schema with additionalProperties: false for key constraint
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string", "enum": ["Alice", "Bob"]},
            },
            "additionalProperties": False,
        }
        pda = SchemaPDA(schema, vocab, allow_unk_key=False, allow_unk_value=False)

        # Value position: UNK_VALUE blocked
        name_mask = pda.get_mask_for_schema_path("name")
        assert not name_mask[vocab.unk_value_id].item()

        # Key position: UNK_KEY blocked
        root_mask = pda.get_mask_for_schema_path("")
        assert not root_mask[vocab.unk_key_id].item()

    def test_lenient_mode_both_allowed(self, vocab):
        """With both flags True (evaluation mode), both UNKs are allowed."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string", "enum": ["Alice", "Bob"]},
            },
            "additionalProperties": False,
        }
        pda = SchemaPDA(schema, vocab, allow_unk_key=True, allow_unk_value=True)

        # Value position: UNK_VALUE allowed
        name_mask = pda.get_mask_for_schema_path("name")
        assert name_mask[vocab.unk_value_id].item()

        # Key position: UNK_KEY allowed
        root_mask = pda.get_mask_for_schema_path("")
        assert root_mask[vocab.unk_key_id].item()


class TestSummary:
    def test_summary_output(self, vocab, simple_schema):
        pda = SchemaPDA(simple_schema, vocab)
        summary = pda.summary()
        assert "SchemaPDA" in summary
        assert "name" in summary
        assert "age" in summary
