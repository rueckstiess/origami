"""Tests for target field preprocessing utilities."""

import pytest

from origami.preprocessing import move_target_last


class TestMoveTargetLast:
    """Tests for move_target_last function."""

    def test_simple_key_move(self):
        """Test moving a simple key to last position."""
        obj = {"a": 1, "b": 2, "c": 3}
        result = move_target_last(obj, "a")

        # a should be last
        assert list(result.keys()) == ["b", "c", "a"]
        assert result == {"b": 2, "c": 3, "a": 1}

    def test_key_already_last(self):
        """Test that key already last stays in place."""
        obj = {"a": 1, "b": 2, "c": 3}
        result = move_target_last(obj, "c")

        assert list(result.keys()) == ["a", "b", "c"]
        assert result == {"a": 1, "b": 2, "c": 3}

    def test_middle_key(self):
        """Test moving middle key to last."""
        obj = {"x": 10, "y": 20, "z": 30}
        result = move_target_last(obj, "y")

        assert list(result.keys()) == ["x", "z", "y"]
        assert result["y"] == 20

    def test_nested_key_single_level(self):
        """Test moving nested key (one level deep)."""
        obj = {"outer": {"a": 1, "b": 2, "c": 3}, "other": 99}
        result = move_target_last(obj, "outer.a")

        # outer should be last at root
        assert list(result.keys()) == ["other", "outer"]
        # a should be last within outer
        assert list(result["outer"].keys()) == ["b", "c", "a"]

    def test_nested_key_multi_level(self):
        """Test moving deeply nested key."""
        obj = {
            "level1": {
                "level2": {
                    "target": "value",
                    "other": "data",
                },
                "sibling": "here",
            },
            "root_sibling": "there",
        }
        result = move_target_last(obj, "level1.level2.target")

        # level1 should be last at root
        assert list(result.keys()) == ["root_sibling", "level1"]
        # level2 should be last in level1
        assert list(result["level1"].keys()) == ["sibling", "level2"]
        # target should be last in level2
        assert list(result["level1"]["level2"].keys()) == ["other", "target"]

    def test_does_not_mutate_input(self):
        """Test that original object is not modified."""
        obj = {"a": 1, "b": 2, "c": 3}
        original_keys = list(obj.keys())

        result = move_target_last(obj, "a")

        # Original should be unchanged
        assert list(obj.keys()) == original_keys
        # Result should be different object
        assert result is not obj

    def test_nested_does_not_mutate_input(self):
        """Test that nested objects are also not mutated."""
        obj = {"outer": {"a": 1, "b": 2}, "other": 3}
        original_inner_keys = list(obj["outer"].keys())

        result = move_target_last(obj, "outer.a")

        # Original nested dict should be unchanged
        assert list(obj["outer"].keys()) == original_inner_keys
        # Result's nested dict should be different object
        assert result["outer"] is not obj["outer"]

    def test_preserves_values(self):
        """Test that all values are preserved correctly."""
        obj = {
            "string": "hello",
            "number": 42,
            "float": 3.14,
            "bool": True,
            "none": None,
            "list": [1, 2, 3],
            "nested": {"x": 1},
        }
        result = move_target_last(obj, "string")

        # All values should be preserved
        assert result["string"] == "hello"
        assert result["number"] == 42
        assert result["float"] == 3.14
        assert result["bool"] is True
        assert result["none"] is None
        assert result["list"] == [1, 2, 3]
        assert result["nested"] == {"x": 1}

    def test_missing_key_inserted_with_none(self):
        """Test that missing key is inserted with None value."""
        obj = {"a": 1, "b": 2}
        result = move_target_last(obj, "missing")

        # missing should be last with None value
        assert list(result.keys()) == ["a", "b", "missing"]
        assert result["missing"] is None

    def test_missing_nested_key_inserted_with_none(self):
        """Test that missing nested key is inserted with None value."""
        obj = {"outer": {"a": 1}}
        result = move_target_last(obj, "outer.missing")

        # outer should be last, missing should be last within outer
        assert list(result.keys()) == ["outer"]
        assert list(result["outer"].keys()) == ["a", "missing"]
        assert result["outer"]["missing"] is None

    def test_missing_intermediate_key_creates_nested_structure(self):
        """Test that missing intermediate keys create nested dicts."""
        obj = {"a": 1}
        result = move_target_last(obj, "x.y.z")

        # x should be last at root
        assert list(result.keys()) == ["a", "x"]
        # nested structure should be created
        assert result["x"] == {"y": {"z": None}}

    def test_missing_root_key_with_nested_path(self):
        """Test missing root key when target is nested."""
        obj = {"existing": "value"}
        result = move_target_last(obj, "new.nested")

        assert list(result.keys()) == ["existing", "new"]
        assert result["new"] == {"nested": None}

    def test_nested_path_through_non_dict_raises(self):
        """Test that path through non-dict value raises KeyError."""
        obj = {"outer": "not a dict"}

        with pytest.raises(KeyError, match="Cannot access"):
            move_target_last(obj, "outer.nested")

    def test_empty_key_raises(self):
        """Test that empty target_key raises ValueError."""
        obj = {"a": 1}

        with pytest.raises(ValueError, match="cannot be empty"):
            move_target_last(obj, "")

    def test_single_key_object(self):
        """Test object with only one key."""
        obj = {"only": "value"}
        result = move_target_last(obj, "only")

        assert result == {"only": "value"}
        assert list(result.keys()) == ["only"]

    def test_complex_nested_structure(self):
        """Test with complex nested structure including arrays."""
        obj = {
            "user": {
                "name": "Alice",
                "scores": [90, 85, 95],
                "metadata": {"created": "2024-01-01"},
            },
            "status": "active",
            "target": "predict_me",
        }
        result = move_target_last(obj, "target")

        assert list(result.keys()) == ["user", "status", "target"]
        # Nested structures should be preserved
        assert result["user"]["scores"] == [90, 85, 95]
        assert result["user"]["metadata"]["created"] == "2024-01-01"

    def test_move_nested_preserves_siblings(self):
        """Test that sibling keys at each level are preserved."""
        obj = {
            "a": {"x": 1, "y": 2, "z": 3},
            "b": {"p": 10, "q": 20},
            "c": 100,
        }
        result = move_target_last(obj, "a.y")

        # At root: a should be last, b and c preserved
        assert "b" in result
        assert "c" in result
        assert result["b"] == {"p": 10, "q": 20}
        assert result["c"] == 100

        # In a: y should be last, x and z preserved
        assert result["a"]["x"] == 1
        assert result["a"]["z"] == 3
