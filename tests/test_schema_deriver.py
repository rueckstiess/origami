"""Tests for JSON Schema derivation from training data."""

import pytest

from origami.constraints.schema_deriver import SchemaDeriver
from origami.preprocessing.numeric_scaler import ScaledNumeric


@pytest.fixture
def deriver():
    """Default deriver with no enum threshold (always includes enum)."""
    return SchemaDeriver()


class TestSchemaDeriverInit:
    def test_default_threshold(self):
        d = SchemaDeriver()
        assert d.enum_threshold is None

    def test_custom_threshold(self):
        d = SchemaDeriver(enum_threshold=50)
        assert d.enum_threshold == 50

    def test_none_threshold(self):
        d = SchemaDeriver(enum_threshold=None)
        assert d.enum_threshold is None

    def test_invalid_threshold(self):
        with pytest.raises(ValueError, match="enum_threshold must be >= 1 or None"):
            SchemaDeriver(enum_threshold=0)


class TestSchemaDeriverValidation:
    def test_empty_data(self, deriver):
        with pytest.raises(ValueError, match="Cannot derive schema from empty data"):
            deriver.derive([])

    def test_non_dict_data(self, deriver):
        with pytest.raises(ValueError, match="All items in data must be dicts"):
            deriver.derive([{"a": 1}, "not a dict"])


class TestFlatObjectSchema:
    def test_single_string_field(self, deriver):
        data = [{"name": "Alice"}, {"name": "Bob"}]
        schema = deriver.derive(data)

        assert schema["type"] == "object"
        assert "name" in schema["properties"]
        assert schema["properties"]["name"]["type"] == "string"
        assert schema["properties"]["name"]["enum"] == ["Alice", "Bob"]
        assert schema["required"] == ["name"]

    def test_single_integer_field(self, deriver):
        data = [{"age": 25}, {"age": 30}]
        schema = deriver.derive(data)

        assert schema["properties"]["age"]["type"] == "integer"
        assert schema["properties"]["age"]["minimum"] == 25
        assert schema["properties"]["age"]["maximum"] == 30
        assert schema["properties"]["age"]["enum"] == [25, 30]

    def test_single_float_field(self, deriver):
        data = [{"score": 1.5}, {"score": 2.5}]
        schema = deriver.derive(data)

        assert schema["properties"]["score"]["type"] == "number"
        assert schema["properties"]["score"]["minimum"] == 1.5
        assert schema["properties"]["score"]["maximum"] == 2.5

    def test_boolean_field(self, deriver):
        data = [{"active": True}, {"active": False}]
        schema = deriver.derive(data)

        assert schema["properties"]["active"]["type"] == "boolean"
        assert schema["properties"]["active"]["enum"] == [False, True]

    def test_null_field(self, deriver):
        data = [{"value": None}, {"value": None}]
        schema = deriver.derive(data)

        assert schema["properties"]["value"]["type"] == "null"
        assert schema["properties"]["value"]["enum"] == [None]

    def test_multiple_fields(self, deriver):
        data = [
            {"name": "Alice", "age": 25, "active": True},
            {"name": "Bob", "age": 30, "active": False},
        ]
        schema = deriver.derive(data)

        assert set(schema["properties"].keys()) == {"name", "age", "active"}
        assert schema["required"] == ["active", "age", "name"]

    def test_mixed_int_float(self, deriver):
        """When a field has both int and float values, consolidate to 'number'."""
        data = [{"val": 1}, {"val": 2.5}]
        schema = deriver.derive(data)

        assert schema["properties"]["val"]["type"] == "number"
        assert schema["properties"]["val"]["minimum"] == 1
        assert schema["properties"]["val"]["maximum"] == 2.5


class TestRequiredFields:
    def test_all_present(self, deriver):
        data = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
        schema = deriver.derive(data)
        assert schema["required"] == ["a", "b"]

    def test_optional_field(self, deriver):
        data = [{"a": 1, "b": 2}, {"a": 3}]
        schema = deriver.derive(data)
        assert schema["required"] == ["a"]

    def test_no_required(self, deriver):
        data = [{"a": 1}, {"b": 2}]
        schema = deriver.derive(data)
        assert "required" not in schema


class TestEnumThreshold:
    def test_default_always_includes_enum(self):
        """Default (None) threshold always includes enum regardless of cardinality."""
        deriver = SchemaDeriver()
        data = [{"id": f"val_{i}"} for i in range(500)]
        schema = deriver.derive(data)
        assert "enum" in schema["properties"]["id"]
        assert len(schema["properties"]["id"]["enum"]) == 500

    def test_below_threshold(self):
        deriver = SchemaDeriver(enum_threshold=5)
        data = [{"color": c} for c in ["red", "green", "blue"]]
        schema = deriver.derive(data)
        assert "enum" in schema["properties"]["color"]
        assert sorted(schema["properties"]["color"]["enum"]) == ["blue", "green", "red"]

    def test_above_threshold(self):
        deriver = SchemaDeriver(enum_threshold=5)
        data = [{"id": i} for i in range(10)]
        schema = deriver.derive(data)
        assert "enum" not in schema["properties"]["id"]

    def test_at_threshold(self):
        deriver = SchemaDeriver(enum_threshold=5)
        data = [{"x": i} for i in range(5)]
        schema = deriver.derive(data)
        assert "enum" in schema["properties"]["x"]

    def test_one_above_threshold(self):
        deriver = SchemaDeriver(enum_threshold=5)
        data = [{"x": i} for i in range(6)]
        schema = deriver.derive(data)
        assert "enum" not in schema["properties"]["x"]


class TestNestedObjectSchema:
    def test_nested_object(self, deriver):
        data = [
            {"user": {"name": "Alice", "age": 25}},
            {"user": {"name": "Bob", "age": 30}},
        ]
        schema = deriver.derive(data)

        user_schema = schema["properties"]["user"]
        assert user_schema["type"] == "object"
        assert "name" in user_schema["properties"]
        assert "age" in user_schema["properties"]
        assert user_schema["required"] == ["age", "name"]

    def test_deeply_nested(self, deriver):
        data = [
            {"a": {"b": {"c": "deep"}}},
            {"a": {"b": {"c": "value"}}},
        ]
        schema = deriver.derive(data)

        c_schema = schema["properties"]["a"]["properties"]["b"]["properties"]["c"]
        assert c_schema["type"] == "string"
        assert c_schema["enum"] == ["deep", "value"]


class TestArraySchema:
    def test_string_array(self, deriver):
        data = [
            {"tags": ["a", "b"]},
            {"tags": ["c"]},
        ]
        schema = deriver.derive(data)

        tags_schema = schema["properties"]["tags"]
        assert tags_schema["type"] == "array"
        assert tags_schema["items"]["type"] == "string"
        assert tags_schema["minItems"] == 1
        assert tags_schema["maxItems"] == 2

    def test_integer_array(self, deriver):
        data = [
            {"nums": [1, 2, 3]},
            {"nums": [4, 5]},
        ]
        schema = deriver.derive(data)

        nums_schema = schema["properties"]["nums"]
        assert nums_schema["type"] == "array"
        assert nums_schema["items"]["type"] == "integer"
        assert nums_schema["items"]["minimum"] == 1
        assert nums_schema["items"]["maximum"] == 5
        assert nums_schema["minItems"] == 2
        assert nums_schema["maxItems"] == 3

    def test_empty_array(self, deriver):
        data = [{"items": []}, {"items": [1]}]
        schema = deriver.derive(data)

        items_schema = schema["properties"]["items"]
        assert items_schema["type"] == "array"
        assert items_schema["minItems"] == 0
        assert items_schema["maxItems"] == 1

    def test_array_of_objects(self, deriver):
        data = [
            {"items": [{"name": "a", "price": 10}, {"name": "b", "price": 20}]},
            {"items": [{"name": "c", "price": 30}]},
        ]
        schema = deriver.derive(data)

        items_schema = schema["properties"]["items"]["items"]
        assert items_schema["type"] == "object"
        assert "name" in items_schema["properties"]
        assert "price" in items_schema["properties"]

    def test_nested_arrays(self, deriver):
        data = [
            {"matrix": [[1, 2], [3, 4]]},
            {"matrix": [[5, 6]]},
        ]
        schema = deriver.derive(data)

        matrix_schema = schema["properties"]["matrix"]
        assert matrix_schema["type"] == "array"
        assert matrix_schema["items"]["type"] == "array"
        assert matrix_schema["items"]["items"]["type"] == "integer"


class TestMixedTypes:
    def test_string_or_null(self, deriver):
        data = [{"val": "hello"}, {"val": None}]
        schema = deriver.derive(data)

        val_schema = schema["properties"]["val"]
        assert sorted(val_schema["type"]) == ["null", "string"]

    def test_string_or_integer(self, deriver):
        data = [{"val": "hello"}, {"val": 42}]
        schema = deriver.derive(data)

        val_schema = schema["properties"]["val"]
        assert sorted(val_schema["type"]) == ["integer", "string"]

    def test_object_or_null(self, deriver):
        data = [{"val": {"x": 1}}, {"val": None}]
        schema = deriver.derive(data)

        val_schema = schema["properties"]["val"]
        assert sorted(val_schema["type"]) == ["null", "object"]
        assert "properties" in val_schema

    def test_array_or_null(self, deriver):
        data = [{"val": [1, 2]}, {"val": None}]
        schema = deriver.derive(data)

        val_schema = schema["properties"]["val"]
        assert sorted(val_schema["type"]) == ["array", "null"]


class TestNumericBounds:
    def test_integer_bounds(self, deriver):
        data = [{"x": 10}, {"x": 50}, {"x": 100}]
        schema = deriver.derive(data)

        assert schema["properties"]["x"]["minimum"] == 10
        assert schema["properties"]["x"]["maximum"] == 100

    def test_float_bounds(self, deriver):
        data = [{"x": 0.1}, {"x": 0.9}]
        schema = deriver.derive(data)

        assert schema["properties"]["x"]["minimum"] == 0.1
        assert schema["properties"]["x"]["maximum"] == 0.9

    def test_negative_numbers(self, deriver):
        data = [{"x": -10}, {"x": 5}]
        schema = deriver.derive(data)

        assert schema["properties"]["x"]["minimum"] == -10
        assert schema["properties"]["x"]["maximum"] == 5

    def test_no_bounds_for_strings(self, deriver):
        data = [{"x": "a"}, {"x": "b"}]
        schema = deriver.derive(data)

        assert "minimum" not in schema["properties"]["x"]
        assert "maximum" not in schema["properties"]["x"]


class TestEdgeCases:
    def test_single_object(self, deriver):
        schema = deriver.derive([{"a": 1}])
        assert schema["properties"]["a"]["type"] == "integer"
        assert schema["required"] == ["a"]

    def test_empty_objects(self, deriver):
        schema = deriver.derive([{}, {}])
        assert schema["type"] == "object"
        assert schema["properties"] == {}
        assert "required" not in schema

    def test_boolean_not_counted_as_int(self, deriver):
        """Booleans should be typed as 'boolean', not 'integer'."""
        data = [{"x": True}, {"x": False}]
        schema = deriver.derive(data)
        assert schema["properties"]["x"]["type"] == "boolean"

    def test_boolean_and_int_mixed(self, deriver):
        """When True/False and integers are mixed, both types appear."""
        data = [{"x": True}, {"x": 42}]
        schema = deriver.derive(data)
        val_schema = schema["properties"]["x"]
        types = val_schema["type"] if isinstance(val_schema["type"], list) else [val_schema["type"]]
        assert "boolean" in types
        assert "integer" in types

    def test_deterministic_output(self, deriver):
        """Schema output should be deterministic (sorted keys, sorted enums)."""
        data = [
            {"z": 1, "a": "x", "m": True},
            {"z": 2, "a": "y", "m": False},
        ]
        schema1 = deriver.derive(data)
        schema2 = deriver.derive(data)
        assert schema1 == schema2

    def test_all_none_field(self, deriver):
        data = [{"x": None}, {"x": None}]
        schema = deriver.derive(data)
        assert schema["properties"]["x"]["type"] == "null"


class TestUniqueItems:
    def test_unique_arrays_detected(self, deriver):
        """Arrays with all unique elements should get uniqueItems: true."""
        data = [
            {"tags": ["a", "b", "c"]},
            {"tags": ["d", "e"]},
        ]
        schema = deriver.derive(data)
        assert schema["properties"]["tags"]["uniqueItems"] is True

    def test_duplicate_arrays_not_unique(self, deriver):
        """Arrays with duplicates should NOT get uniqueItems."""
        data = [
            {"tags": ["a", "b", "a"]},
            {"tags": ["d", "e"]},
        ]
        schema = deriver.derive(data)
        assert "uniqueItems" not in schema["properties"]["tags"]

    def test_single_element_arrays_unique(self, deriver):
        """Single-element arrays are trivially unique."""
        data = [
            {"tags": ["a"]},
            {"tags": ["b"]},
        ]
        schema = deriver.derive(data)
        assert schema["properties"]["tags"]["uniqueItems"] is True

    def test_empty_arrays_unique(self, deriver):
        """Empty arrays are trivially unique, but no uniqueItems without elements."""
        data = [{"tags": []}, {"tags": []}]
        schema = deriver.derive(data)
        # No elements means no uniqueItems (nothing to constrain)
        assert "uniqueItems" not in schema["properties"]["tags"]

    def test_mixed_unique_and_duplicate(self, deriver):
        """If ANY array has duplicates, uniqueItems should not be set."""
        data = [
            {"tags": ["a", "b"]},
            {"tags": ["c", "c"]},  # duplicate
        ]
        schema = deriver.derive(data)
        assert "uniqueItems" not in schema["properties"]["tags"]

    def test_integer_arrays_unique(self, deriver):
        """Integer arrays with unique elements."""
        data = [
            {"ids": [1, 2, 3]},
            {"ids": [4, 5]},
        ]
        schema = deriver.derive(data)
        assert schema["properties"]["ids"]["uniqueItems"] is True

    def test_integer_arrays_with_duplicates(self, deriver):
        data = [
            {"ids": [1, 2, 2]},
        ]
        schema = deriver.derive(data)
        assert "uniqueItems" not in schema["properties"]["ids"]

    def test_object_arrays_skip_uniqueness(self, deriver):
        """Arrays of objects should skip uniqueness check (unhashable)."""
        data = [
            {"items": [{"a": 1}, {"b": 2}]},
        ]
        schema = deriver.derive(data)
        # Objects are unhashable, so uniqueItems should not be set
        assert "uniqueItems" not in schema["properties"]["items"]

    def test_unique_items_with_merge(self, deriver):
        """When merging schemas, uniqueItems uses intersection (AND)."""
        # Both array and non-array values for same field
        data = [
            {"val": ["a", "b"]},
            {"val": ["c", "d"]},
        ]
        schema = deriver.derive(data)
        assert schema["properties"]["val"]["uniqueItems"] is True


class TestAdditionalProperties:
    def test_root_object(self, deriver):
        """Root object should get additionalProperties: false."""
        data = [{"name": "Alice"}, {"name": "Bob"}]
        schema = deriver.derive(data)
        assert schema["additionalProperties"] is False

    def test_nested_object(self, deriver):
        """Nested objects should also get additionalProperties: false."""
        data = [
            {"user": {"name": "Alice", "age": 25}},
            {"user": {"name": "Bob", "age": 30}},
        ]
        schema = deriver.derive(data)
        assert schema["additionalProperties"] is False
        assert schema["properties"]["user"]["additionalProperties"] is False

    def test_array_of_objects(self, deriver):
        """Object items in arrays should get additionalProperties: false."""
        data = [
            {"items": [{"name": "a", "price": 10}]},
            {"items": [{"name": "b", "price": 20}]},
        ]
        schema = deriver.derive(data)
        items_schema = schema["properties"]["items"]["items"]
        assert items_schema["additionalProperties"] is False

    def test_empty_object(self, deriver):
        """Empty objects should also get additionalProperties: false."""
        schema = deriver.derive([{}, {}])
        assert schema["additionalProperties"] is False

    def test_object_or_null_preserves_additional_properties(self, deriver):
        """When object is merged with null, additionalProperties should survive."""
        data = [{"val": {"x": 1}}, {"val": None}]
        schema = deriver.derive(data)
        val_schema = schema["properties"]["val"]
        assert val_schema["additionalProperties"] is False

    def test_deeply_nested(self, deriver):
        """All levels of nesting should get additionalProperties: false."""
        data = [
            {"a": {"b": {"c": "deep"}}},
            {"a": {"b": {"c": "value"}}},
        ]
        schema = deriver.derive(data)
        assert schema["additionalProperties"] is False
        assert schema["properties"]["a"]["additionalProperties"] is False
        assert schema["properties"]["a"]["properties"]["b"]["additionalProperties"] is False


class TestScaledNumericSchema:
    """Tests for ScaledNumeric value handling in schema derivation."""

    def test_scaled_numeric_produces_number_type(self, deriver):
        """ScaledNumeric values should produce type: 'number' with min/max."""
        data = [
            {"price": ScaledNumeric(value=-1.5)},
            {"price": ScaledNumeric(value=0.0)},
            {"price": ScaledNumeric(value=2.3)},
        ]
        schema = deriver.derive(data)

        price_schema = schema["properties"]["price"]
        assert price_schema["type"] == "number"
        assert price_schema["minimum"] == -1.5
        assert price_schema["maximum"] == 2.3

    def test_scaled_numeric_no_enum(self, deriver):
        """ScaledNumeric fields should NOT have enum (continuous values)."""
        data = [{"x": ScaledNumeric(value=float(i) / 10)} for i in range(100)]
        schema = deriver.derive(data)

        x_schema = schema["properties"]["x"]
        assert x_schema["type"] == "number"
        assert "enum" not in x_schema

    def test_scaled_numeric_required(self, deriver):
        """ScaledNumeric fields present in all objects should be required."""
        data = [
            {"val": ScaledNumeric(value=1.0)},
            {"val": ScaledNumeric(value=2.0)},
        ]
        schema = deriver.derive(data)
        assert "val" in schema["required"]

    def test_mixed_scaled_and_primitive(self, deriver):
        """Mixed ScaledNumeric and primitive values should merge correctly."""
        data = [
            {"x": ScaledNumeric(value=1.0)},
            {"x": "unknown"},
        ]
        schema = deriver.derive(data)

        x_schema = schema["properties"]["x"]
        # Should have both number (from ScaledNumeric) and string types
        assert isinstance(x_schema["type"], list)
        assert "number" in x_schema["type"]
        assert "string" in x_schema["type"]

    def test_scaled_numeric_bounds_are_from_scaled_values(self, deriver):
        """Bounds should reflect the scaled (z-score) values, not originals."""
        data = [
            {"temp": ScaledNumeric(value=-2.0)},
            {"temp": ScaledNumeric(value=3.0)},
        ]
        schema = deriver.derive(data)

        temp_schema = schema["properties"]["temp"]
        assert temp_schema["minimum"] == -2.0
        assert temp_schema["maximum"] == 3.0

    def test_single_scaled_value(self, deriver):
        """Single ScaledNumeric value: min == max."""
        data = [{"x": ScaledNumeric(value=0.5)}]
        schema = deriver.derive(data)

        x_schema = schema["properties"]["x"]
        assert x_schema["type"] == "number"
        assert x_schema["minimum"] == 0.5
        assert x_schema["maximum"] == 0.5
