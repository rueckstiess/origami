"""Tests for the CLI data loaders module."""

from pathlib import Path

import click
import pytest

from origami.cli.data_loaders import (
    _delete_nested_key,
    apply_projection,
    get_nested_value,
    load_data,
    parse_projection,
    set_nested_value,
)


class TestGetNestedValue:
    """Tests for get_nested_value helper."""

    def test_simple_key(self):
        """Simple top-level key access."""
        obj = {"a": 1, "b": 2}
        assert get_nested_value(obj, "a") == 1

    def test_nested_key(self):
        """Dot notation for nested access."""
        obj = {"foo": {"bar": {"baz": 42}}}
        assert get_nested_value(obj, "foo.bar.baz") == 42

    def test_nested_returns_dict(self):
        """Accessing intermediate path returns nested dict."""
        obj = {"foo": {"bar": 1, "baz": 2}}
        assert get_nested_value(obj, "foo") == {"bar": 1, "baz": 2}

    def test_missing_key_raises(self):
        """Missing key raises KeyError."""
        obj = {"a": 1}
        with pytest.raises(KeyError):
            get_nested_value(obj, "b")

    def test_missing_nested_key_raises(self):
        """Missing nested key raises KeyError."""
        obj = {"foo": {"bar": 1}}
        with pytest.raises(KeyError):
            get_nested_value(obj, "foo.baz")

    def test_path_through_non_dict_raises(self):
        """Path through non-dict value raises KeyError."""
        obj = {"foo": 42}
        with pytest.raises(KeyError):
            get_nested_value(obj, "foo.bar")


class TestSetNestedValue:
    """Tests for set_nested_value helper."""

    def test_simple_key(self):
        """Set top-level key."""
        obj: dict = {}
        set_nested_value(obj, "a", 1)
        assert obj == {"a": 1}

    def test_nested_key_creates_intermediate(self):
        """Nested path creates intermediate dicts."""
        obj: dict = {}
        set_nested_value(obj, "foo.bar.baz", 42)
        assert obj == {"foo": {"bar": {"baz": 42}}}

    def test_overwrites_existing(self):
        """Existing values are overwritten."""
        obj = {"foo": {"bar": 1}}
        set_nested_value(obj, "foo.bar", 2)
        assert obj == {"foo": {"bar": 2}}

    def test_adds_to_existing_nested(self):
        """Adds new key to existing nested dict."""
        obj = {"foo": {"bar": 1}}
        set_nested_value(obj, "foo.baz", 2)
        assert obj == {"foo": {"bar": 1, "baz": 2}}


class TestDeleteNestedKey:
    """Tests for _delete_nested_key helper."""

    def test_delete_simple_key(self):
        """Delete top-level key."""
        obj = {"a": 1, "b": 2}
        _delete_nested_key(obj, "a")
        assert obj == {"b": 2}

    def test_delete_nested_key(self):
        """Delete nested key."""
        obj = {"foo": {"bar": 1, "baz": 2}}
        _delete_nested_key(obj, "foo.bar")
        assert obj == {"foo": {"baz": 2}}

    def test_delete_missing_key_silent(self):
        """Deleting missing key does nothing."""
        obj = {"a": 1}
        _delete_nested_key(obj, "b")
        assert obj == {"a": 1}

    def test_delete_missing_nested_key_silent(self):
        """Deleting missing nested key does nothing."""
        obj = {"foo": {"bar": 1}}
        _delete_nested_key(obj, "foo.baz")
        assert obj == {"foo": {"bar": 1}}


class TestParseProjection:
    """Tests for parse_projection helper."""

    def test_inclusion_projection(self):
        """Parse inclusion projection."""
        result = parse_projection('{"a": 1, "b": 1}')
        assert result == {"a": 1, "b": 1}

    def test_exclusion_projection(self):
        """Parse exclusion projection."""
        result = parse_projection('{"x": 0}')
        assert result == {"x": 0}

    def test_nested_keys(self):
        """Parse projection with dot notation keys."""
        result = parse_projection('{"foo.bar": 1}')
        assert result == {"foo.bar": 1}

    def test_invalid_json_raises(self):
        """Invalid JSON raises BadParameter."""
        with pytest.raises(click.BadParameter):
            parse_projection("not json")

    def test_non_object_raises(self):
        """Non-object JSON raises BadParameter."""
        with pytest.raises(click.BadParameter):
            parse_projection("[1, 2, 3]")

    def test_invalid_value_raises(self):
        """Values other than 0 or 1 raise BadParameter."""
        with pytest.raises(click.BadParameter):
            parse_projection('{"a": 2}')

    def test_mixed_mode_raises(self):
        """Mixed inclusion/exclusion raises BadParameter."""
        with pytest.raises(click.BadParameter):
            parse_projection('{"a": 1, "b": 0}')

    def test_empty_projection(self):
        """Empty projection is valid."""
        result = parse_projection("{}")
        assert result == {}


class TestApplyProjection:
    """Tests for apply_projection helper."""

    def test_inclusion_simple(self):
        """Inclusion mode keeps only specified fields."""
        obj = {"a": 1, "b": 2, "c": 3}
        result = apply_projection(obj, {"a": 1, "b": 1})
        assert result == {"a": 1, "b": 2}

    def test_exclusion_simple(self):
        """Exclusion mode removes specified fields."""
        obj = {"a": 1, "b": 2, "c": 3}
        result = apply_projection(obj, {"b": 0})
        assert result == {"a": 1, "c": 3}

    def test_inclusion_nested(self):
        """Inclusion with nested path."""
        obj = {"x": {"y": 1, "z": 2}, "a": 3}
        result = apply_projection(obj, {"x.y": 1})
        assert result == {"x": {"y": 1}}

    def test_exclusion_nested(self):
        """Exclusion with nested path."""
        obj = {"x": {"y": 1, "z": 2}, "a": 3}
        result = apply_projection(obj, {"x.z": 0})
        assert result == {"x": {"y": 1}, "a": 3}

    def test_missing_field_in_inclusion(self):
        """Missing fields silently skipped in inclusion mode."""
        obj = {"a": 1}
        result = apply_projection(obj, {"a": 1, "b": 1})
        assert result == {"a": 1}

    def test_empty_projection_returns_original(self):
        """Empty projection returns original object."""
        obj = {"a": 1, "b": 2}
        result = apply_projection(obj, {})
        assert result == {"a": 1, "b": 2}

    def test_inclusion_does_not_mutate_original(self):
        """Inclusion mode doesn't mutate original."""
        obj = {"a": 1, "b": 2}
        result = apply_projection(obj, {"a": 1})
        assert result == {"a": 1}
        assert obj == {"a": 1, "b": 2}

    def test_exclusion_does_not_mutate_original(self):
        """Exclusion mode doesn't mutate original."""
        obj = {"a": 1, "b": 2}
        result = apply_projection(obj, {"b": 0})
        assert result == {"a": 1}
        assert obj == {"a": 1, "b": 2}


class TestLoadDataSkipLimit:
    """Tests for skip/limit functionality in load_data."""

    def test_skip_jsonl(self, tmp_path: Path):
        """Skip skips samples at beginning."""
        data_file = tmp_path / "data.jsonl"
        data_file.write_text('{"a": 1}\n{"a": 2}\n{"a": 3}\n{"a": 4}\n')

        result = load_data(str(data_file), skip=2)
        assert len(result) == 2
        assert result[0] == {"a": 3}
        assert result[1] == {"a": 4}

    def test_limit_jsonl(self, tmp_path: Path):
        """Limit limits number of samples."""
        data_file = tmp_path / "data.jsonl"
        data_file.write_text('{"a": 1}\n{"a": 2}\n{"a": 3}\n{"a": 4}\n')

        result = load_data(str(data_file), limit=2)
        assert len(result) == 2
        assert result[0] == {"a": 1}
        assert result[1] == {"a": 2}

    def test_skip_and_limit_jsonl(self, tmp_path: Path):
        """Skip and limit combine correctly."""
        data_file = tmp_path / "data.jsonl"
        data_file.write_text('{"a": 1}\n{"a": 2}\n{"a": 3}\n{"a": 4}\n{"a": 5}\n')

        result = load_data(str(data_file), skip=1, limit=2)
        assert len(result) == 2
        assert result[0] == {"a": 2}
        assert result[1] == {"a": 3}

    def test_skip_beyond_data(self, tmp_path: Path):
        """Skip beyond data length returns empty list."""
        data_file = tmp_path / "data.jsonl"
        data_file.write_text('{"a": 1}\n{"a": 2}\n')

        result = load_data(str(data_file), skip=10)
        assert result == []

    def test_limit_zero_means_unlimited(self, tmp_path: Path):
        """Limit of 0 means unlimited."""
        data_file = tmp_path / "data.jsonl"
        data_file.write_text('{"a": 1}\n{"a": 2}\n{"a": 3}\n')

        result = load_data(str(data_file), limit=0)
        assert len(result) == 3

    def test_skip_with_csv(self, tmp_path: Path):
        """Skip works with CSV files."""
        data_file = tmp_path / "data.csv"
        data_file.write_text("a,b\n1,2\n3,4\n5,6\n")

        result = load_data(str(data_file), skip=1)
        assert len(result) == 2
        assert result[0] == {"a": 3, "b": 4}

    def test_limit_with_json(self, tmp_path: Path):
        """Limit works with JSON files."""
        data_file = tmp_path / "data.json"
        data_file.write_text('[{"a": 1}, {"a": 2}, {"a": 3}]')

        result = load_data(str(data_file), limit=2)
        assert len(result) == 2


class TestLoadDataProject:
    """Tests for projection functionality in load_data."""

    def test_project_inclusion(self, tmp_path: Path):
        """Inclusion projection filters fields."""
        data_file = tmp_path / "data.jsonl"
        data_file.write_text('{"a": 1, "b": 2, "c": 3}\n')

        result = load_data(str(data_file), project='{"a": 1, "b": 1}')
        assert len(result) == 1
        assert result[0] == {"a": 1, "b": 2}

    def test_project_exclusion(self, tmp_path: Path):
        """Exclusion projection removes fields."""
        data_file = tmp_path / "data.jsonl"
        data_file.write_text('{"a": 1, "b": 2, "c": 3}\n')

        result = load_data(str(data_file), project='{"b": 0}')
        assert len(result) == 1
        assert result[0] == {"a": 1, "c": 3}

    def test_project_nested_inclusion(self, tmp_path: Path):
        """Nested path inclusion."""
        data_file = tmp_path / "data.jsonl"
        data_file.write_text('{"user": {"name": "Alice", "age": 30}, "id": 1}\n')

        result = load_data(str(data_file), project='{"user.name": 1}')
        assert len(result) == 1
        assert result[0] == {"user": {"name": "Alice"}}

    def test_skip_limit_project_combined(self, tmp_path: Path):
        """All three options work together."""
        data_file = tmp_path / "data.jsonl"
        lines = [
            '{"a": 1, "b": 10}',
            '{"a": 2, "b": 20}',
            '{"a": 3, "b": 30}',
            '{"a": 4, "b": 40}',
        ]
        data_file.write_text("\n".join(lines))

        result = load_data(str(data_file), skip=1, limit=2, project='{"a": 1}')
        assert len(result) == 2
        assert result[0] == {"a": 2}
        assert result[1] == {"a": 3}
