"""Tests for SchemaPostProcessor."""

import pytest

from origami.preprocessing.postprocessor import SchemaPostProcessor, _is_integer_type


class TestIsIntegerType:
    """Tests for the _is_integer_type helper."""

    def test_integer_string(self):
        assert _is_integer_type("integer") is True

    def test_number_string(self):
        assert _is_integer_type("number") is False

    def test_string_type(self):
        assert _is_integer_type("string") is False

    def test_none(self):
        assert _is_integer_type(None) is False

    def test_list_integer_only(self):
        assert _is_integer_type(["integer", "null"]) is True

    def test_list_integer_and_number(self):
        # number subsumes integer, so not treated as strictly integer
        assert _is_integer_type(["integer", "number"]) is False

    def test_list_no_integer(self):
        assert _is_integer_type(["string", "null"]) is False


class TestProcessValue:
    """Tests for process_value on individual values."""

    @pytest.fixture
    def integer_schema(self):
        return SchemaPostProcessor(
            {
                "type": "object",
                "properties": {
                    "age": {"type": "integer", "minimum": 0, "maximum": 120},
                },
            }
        )

    @pytest.fixture
    def enum_schema(self):
        return SchemaPostProcessor(
            {
                "type": "object",
                "properties": {
                    "rating": {
                        "type": "integer",
                        "enum": [1, 2, 3, 4, 5],
                    },
                    "score": {
                        "type": "number",
                        "enum": [0.5, 1.0, 1.5, 2.0],
                    },
                },
            }
        )

    def test_integer_rounding_down(self, integer_schema):
        assert integer_schema.process_value(25.03, "age") == 25
        assert isinstance(integer_schema.process_value(25.03, "age"), int)

    def test_integer_rounding_up(self, integer_schema):
        assert integer_schema.process_value(25.7, "age") == 26
        assert isinstance(integer_schema.process_value(25.7, "age"), int)

    def test_integer_rounding_midpoint(self, integer_schema):
        # Python rounds 0.5 to even (banker's rounding)
        assert integer_schema.process_value(25.5, "age") == 26

    def test_integer_already_int(self, integer_schema):
        result = integer_schema.process_value(25, "age")
        assert result == 25
        assert isinstance(result, int)

    def test_bounds_clipping_above(self, integer_schema):
        assert integer_schema.process_value(150.0, "age") == 120

    def test_bounds_clipping_below(self, integer_schema):
        assert integer_schema.process_value(-5.0, "age") == 0

    def test_enum_snap_exact(self, enum_schema):
        assert enum_schema.process_value(3.0, "rating") == 3

    def test_enum_snap_nearest(self, enum_schema):
        assert enum_schema.process_value(2.7, "rating") == 3

    def test_enum_snap_preserves_type(self, enum_schema):
        # Enum values are ints, so result should be int
        result = enum_schema.process_value(2.7, "rating")
        assert isinstance(result, int)

    def test_enum_snap_float_values(self, enum_schema):
        assert enum_schema.process_value(1.23, "score") == 1.0
        assert enum_schema.process_value(1.74, "score") == 1.5

    def test_noop_for_string(self, integer_schema):
        assert integer_schema.process_value("hello", "age") == "hello"

    def test_noop_for_boolean(self, integer_schema):
        assert integer_schema.process_value(True, "age") is True

    def test_noop_for_none(self, integer_schema):
        assert integer_schema.process_value(None, "age") is None

    def test_unknown_field_passthrough(self, integer_schema):
        # Field not in schema → pass through unchanged
        assert integer_schema.process_value(25.7, "unknown") == 25.7

    def test_no_constraint_passthrough(self):
        # Field exists in schema but has no type or enum
        pp = SchemaPostProcessor(
            {
                "type": "object",
                "properties": {"x": {}},
            }
        )
        assert pp.process_value(25.7, "x") == 25.7


class TestProcessObject:
    """Tests for recursive object post-processing."""

    def test_flat_object(self):
        pp = SchemaPostProcessor(
            {
                "type": "object",
                "properties": {
                    "age": {"type": "integer"},
                    "name": {"type": "string"},
                    "score": {"type": "number"},
                },
            }
        )
        result = pp.process_object({"age": 25.3, "name": "Alice", "score": 3.14})
        assert result == {"age": 25, "name": "Alice", "score": 3.14}
        assert isinstance(result["age"], int)

    def test_nested_object(self):
        pp = SchemaPostProcessor(
            {
                "type": "object",
                "properties": {
                    "user": {
                        "type": "object",
                        "properties": {
                            "age": {"type": "integer"},
                        },
                    },
                },
            }
        )
        result = pp.process_object({"user": {"age": 25.3}})
        assert result == {"user": {"age": 25}}
        assert isinstance(result["user"]["age"], int)

    def test_array_items(self):
        pp = SchemaPostProcessor(
            {
                "type": "object",
                "properties": {
                    "scores": {
                        "type": "array",
                        "items": {"type": "integer"},
                    },
                },
            }
        )
        result = pp.process_object({"scores": [1.1, 2.7, 3.0]})
        assert result == {"scores": [1, 3, 3]}
        assert all(isinstance(v, int) for v in result["scores"])

    def test_unknown_field_in_object_passthrough(self):
        pp = SchemaPostProcessor(
            {
                "type": "object",
                "properties": {
                    "age": {"type": "integer"},
                },
            }
        )
        result = pp.process_object({"age": 25.3, "extra": 3.14})
        assert result["age"] == 25
        assert result["extra"] == 3.14  # Not in schema, passed through

    def test_empty_object(self):
        pp = SchemaPostProcessor(
            {
                "type": "object",
                "properties": {"x": {"type": "integer"}},
            }
        )
        assert pp.process_object({}) == {}

    def test_enum_snap_in_object(self):
        pp = SchemaPostProcessor(
            {
                "type": "object",
                "properties": {
                    "grade": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
                },
            }
        )
        result = pp.process_object({"grade": 2.7})
        assert result == {"grade": 3}
        assert isinstance(result["grade"], int)


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_enum_with_none_values(self):
        """Enum containing None should not affect numeric snapping."""
        pp = SchemaPostProcessor(
            {
                "type": "object",
                "properties": {
                    "x": {"type": ["integer", "null"], "enum": [None, 1, 2, 3]},
                },
            }
        )
        assert pp.process_value(1.7, "x") == 2
        assert pp.process_value(None, "x") is None

    def test_enum_with_booleans(self):
        """Boolean enum values should not be treated as numeric."""
        pp = SchemaPostProcessor(
            {
                "type": "object",
                "properties": {
                    "x": {"enum": [True, False, 0, 1]},
                },
            }
        )
        # Booleans are filtered out of numeric_enums, only 0 and 1 remain
        assert pp.process_value(0.3, "x") == 0

    def test_mixed_int_float_type_not_integer(self):
        """type: ["integer", "number"] should NOT round (number subsumes integer)."""
        pp = SchemaPostProcessor(
            {
                "type": "object",
                "properties": {
                    "x": {"type": ["integer", "number"]},
                },
            }
        )
        assert pp.process_value(25.3, "x") == 25.3

    def test_bounds_applied_before_enum_snap(self):
        """Bounds clipping happens before enum snapping."""
        pp = SchemaPostProcessor(
            {
                "type": "object",
                "properties": {
                    "x": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 5,
                        "enum": [1, 2, 3, 4, 5],
                    },
                },
            }
        )
        # Value 10 → clipped to 5 → snapped to 5
        assert pp.process_value(10, "x") == 5

    def test_deeply_nested(self):
        pp = SchemaPostProcessor(
            {
                "type": "object",
                "properties": {
                    "a": {
                        "type": "object",
                        "properties": {
                            "b": {
                                "type": "object",
                                "properties": {
                                    "c": {"type": "integer"},
                                },
                            },
                        },
                    },
                },
            }
        )
        result = pp.process_object({"a": {"b": {"c": 3.7}}})
        assert result == {"a": {"b": {"c": 4}}}
