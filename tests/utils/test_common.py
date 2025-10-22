import unittest
from collections import OrderedDict

from origami.utils.common import (
    Symbol,
    flatten_docs,
    get_value_at_path,
    parse_path,
    reorder_with_target_last,
    sort_dict_fields,
    walk_all_leaf_kvs,
)


class TestWalkAllLeafKVs(unittest.TestCase):
    def test_walk_all_leaf_kvs_nested(self):
        d = {"foo": 1, "bar": {"baz": 2, "buz": 3}}

        results = list(
            map(
                lambda x: {k: v for k, v in x.items() if k != "parent"},
                walk_all_leaf_kvs(d),
            )
        )
        self.assertEqual(
            results,
            [
                {"idx": "foo", "key": "foo", "pos": None, "path": "foo", "value": 1},
                {"idx": "baz", "key": "baz", "pos": None, "path": "bar.baz", "value": 2},
                {"idx": "buz", "key": "buz", "pos": None, "path": "bar.buz", "value": 3},
            ],
        )

    def test_walk_all_leaf_kvs_list(self):
        d = {"foo": 1, "bar": [2, 3, 4]}

        results = list(
            map(
                lambda x: {k: v for k, v in x.items() if k != "parent"},
                walk_all_leaf_kvs(d),
            )
        )
        self.assertEqual(
            results,
            [
                {"idx": "foo", "key": "foo", "pos": None, "path": "foo", "value": 1},
                {"idx": 0, "key": "bar", "pos": 0, "path": "bar.[]", "value": 2},
                {"idx": 1, "key": "bar", "pos": 1, "path": "bar.[]", "value": 3},
                {"idx": 2, "key": "bar", "pos": 2, "path": "bar.[]", "value": 4},
            ],
        )

    def test_walk_all_leaf_kvs_list_with_pos(self):
        d = {"foo": 1, "bar": [2, 3, 4]}

        results = list(
            map(
                lambda x: {k: v for k, v in x.items() if k != "parent"},
                walk_all_leaf_kvs(d, include_pos_in_path=True),
            )
        )
        self.assertEqual(
            results,
            [
                {"idx": "foo", "key": "foo", "pos": None, "path": "foo", "value": 1},
                {"idx": 0, "key": "bar", "pos": 0, "path": "bar.[0]", "value": 2},
                {"idx": 1, "key": "bar", "pos": 1, "path": "bar.[1]", "value": 3},
                {"idx": 2, "key": "bar", "pos": 2, "path": "bar.[2]", "value": 4},
            ],
        )

    def test_walk_all_leaf_kvs_list_of_dicts(self):
        d = {"foo": [{"bar": 1}, {"bar": 2}]}

        results = list(
            map(
                lambda x: {k: v for k, v in x.items() if k != "parent"},
                walk_all_leaf_kvs(d),
            )
        )
        self.assertEqual(
            results,
            [
                {"idx": "bar", "key": "bar", "pos": 0, "path": "foo.[].bar", "value": 1},
                {"idx": "bar", "key": "bar", "pos": 1, "path": "foo.[].bar", "value": 2},
            ],
        )

    def test_walk_all_leaf_kvs_list_of_dicts_with_pos(self):
        d = {"foo": [{"bar": 1}, {"bar": 2}]}

        results = list(
            map(
                lambda x: {k: v for k, v in x.items() if k != "parent"},
                walk_all_leaf_kvs(d, include_pos_in_path=True),
            )
        )
        self.assertEqual(
            results,
            [
                {"idx": "bar", "key": "bar", "pos": 0, "path": "foo.[0].bar", "value": 1},
                {"idx": "bar", "key": "bar", "pos": 1, "path": "foo.[1].bar", "value": 2},
            ],
        )


class TestFlattenDocs(unittest.TestCase):
    def test_flatten_docs(self):
        docs = [
            {"foo": 1, "bar": {"baz": 2, "buz": 3}},
            {"foo": 4, "bar": ["test", {"a": "b"}]},
        ]

        flat = flatten_docs(docs)

        self.assertEqual(
            flat, [{"foo": 1, "bar.baz": 2, "bar.buz": 3}, {"foo": 4, "bar.[0]": "test", "bar.[1].a": "b"}]
        )


class TestDictionaryUtils(unittest.TestCase):
    def test_simple_path(self):
        self.assertEqual(parse_path("a"), ["a"])

    def test_nested_path(self):
        self.assertEqual(parse_path("a.b.c"), ["a", "b", "c"])

    def test_simple_retrieval(self):
        d = {"a": 1}
        value, found = get_value_at_path(d, ["a"])
        self.assertEqual(value, 1)
        self.assertTrue(found)

    def test_nested_retrieval(self):
        d = {"a": {"b": {"c": 42}}}
        value, found = get_value_at_path(d, ["a", "b", "c"])
        self.assertEqual(value, 42)
        self.assertTrue(found)

    def test_missing_key(self):
        d = {"a": 1}
        value, found = get_value_at_path(d, ["b"])
        self.assertFalse(found)
        self.assertIsNone(value)

    def test_nested_missing_key(self):
        d = {"a": {"b": 1}}
        value, found = get_value_at_path(d, ["a", "b", "c"])
        self.assertFalse(found)
        self.assertIsNone(value)

    def test_simple_reorder(self):
        input_dict = {"a": 1, "b": 2, "c": 3}
        expected = OrderedDict([("a", 1), ("c", 3), ("b", 2)])
        result, value = reorder_with_target_last(input_dict, "b")
        self.assertEqual(dict(result), dict(expected))
        self.assertEqual(value, 2)
        self.assertEqual(list(result.keys())[-1], "b")

    def test_nested_reorder(self):
        input_dict = {"a": 1, "b": {"b1": True, "b2": False}, "c": "test"}
        result, value = reorder_with_target_last(input_dict, "b.b1")
        self.assertEqual(list(result.keys())[-1], "b")
        self.assertEqual(list(result["b"].keys())[-1], "b1")
        self.assertTrue(value)

    def test_deep_nesting(self):
        input_dict = {"l1": {"l2": {"l3": {"target": "value", "other": "other_value"}}}}
        result, value = reorder_with_target_last(input_dict, "l1.l2.l3.target")
        self.assertEqual(list(result["l1"]["l2"]["l3"].keys())[-1], "target")
        self.assertEqual(value, "value")

    def test_empty_dict(self):
        result, value = reorder_with_target_last({}, "any_key")
        self.assertEqual(value, Symbol.UNKNOWN)
        self.assertEqual(dict(result), {"any_key": Symbol.UNKNOWN})

    def test_various_value_types(self):
        input_dict = {
            "str": "string",
            "int": 42,
            "bool": True,
            "list": [1, 2, 3],
            "dict": {"nested": "value"},
            "none": None,
        }
        for key in input_dict:
            result, value = reorder_with_target_last(input_dict, key)
            self.assertEqual(list(result.keys())[-1], key)
            self.assertEqual(value, input_dict[key])

    def test_preserve_nested_structure(self):
        input_dict = {"a": {"b": {"c": 1, "d": 2}, "e": 3}, "f": 4}
        result, _ = reorder_with_target_last(input_dict, "a.b.c")
        self.assertIsInstance(result["a"], OrderedDict)
        self.assertIsInstance(result["a"]["b"], OrderedDict)

    def test_target_already_last(self):
        input_dict = OrderedDict([("a", 1), ("b", 2), ("target", 3)])
        result, value = reorder_with_target_last(input_dict, "target")
        self.assertEqual(list(result.keys()), ["a", "b", "target"])
        self.assertEqual(value, 3)

    def test_multiple_nested_fields(self):
        input_dict = {"a": {"x": 1, "y": 2, "z": {"target": "value", "other1": "val1", "other2": "val2"}}, "b": "test"}
        result, value = reorder_with_target_last(input_dict, "a.z.target")
        self.assertEqual(list(result["a"]["z"].keys())[-1], "target")
        self.assertEqual(value, "value")
        self.assertIsInstance(result["a"], OrderedDict)
        self.assertIsInstance(result["a"]["z"], OrderedDict)

    def test_target_is_list(self):
        input_dict = {"a": 1, "foo": [1, 2, 3], "b": 2}
        result, value = reorder_with_target_last(input_dict, "foo")
        self.assertEqual(list(result.keys())[-1], "foo")
        self.assertEqual(value, [1, 2, 3])

    def test_target_is_dict(self):
        input_dict = {"a": 1, "foo": {"nested": "value", "other": 42}, "b": 2}
        result, value = reorder_with_target_last(input_dict, "foo")
        self.assertEqual(list(result.keys())[-1], "foo")
        self.assertEqual(value, {"nested": "value", "other": 42})

    def test_missing_field(self):
        input_dict = {"a": 1, "b": {"b1": True, "b2": False}, "c": "test"}

        # Test missing top-level field
        result, value = reorder_with_target_last(input_dict, "nonexistent")
        expected = {"a": 1, "b": {"b1": True, "b2": False}, "c": "test", "nonexistent": Symbol.UNKNOWN}
        self.assertEqual(dict(result), expected)
        self.assertEqual(value, Symbol.UNKNOWN)

        # Test missing nested field
        result, value = reorder_with_target_last(input_dict, "b.nonexistent")
        expected = {"a": 1, "b": {"b1": True, "b2": False, "nonexistent": Symbol.UNKNOWN}, "c": "test"}
        self.assertEqual(dict(result), expected)
        self.assertEqual(value, Symbol.UNKNOWN)

        # Test path through non-dict value
        result, value = reorder_with_target_last(input_dict, "a.something")
        # Should still treat 'a' as a terminal value since it's not a dict
        expected = {"a": 1, "b": {"b1": True, "b2": False}, "c": "test", "a.something": Symbol.UNKNOWN}
        self.assertEqual(dict(result), expected)
        self.assertEqual(value, Symbol.UNKNOWN)

        # Additional test for multiple levels of missing nested fields
        result, value = reorder_with_target_last(input_dict, "b.new.deeper")
        expected = {"a": 1, "b": {"b1": True, "b2": False, "new": {"deeper": Symbol.UNKNOWN}}, "c": "test"}
        self.assertEqual(dict(result), expected)
        self.assertEqual(value, Symbol.UNKNOWN)


class TestSortDictFields(unittest.TestCase):
    def test_sort_simple_dict(self):
        """Test sorting a simple dictionary."""
        doc = {"z": 1, "a": 2, "m": 3}
        result = sort_dict_fields(doc)

        self.assertIsInstance(result, OrderedDict)
        self.assertEqual(list(result.keys()), ["a", "m", "z"])
        self.assertEqual(result["a"], 2)
        self.assertEqual(result["m"], 3)
        self.assertEqual(result["z"], 1)

    def test_sort_already_sorted(self):
        """Test sorting an already sorted dictionary."""
        doc = {"a": 1, "b": 2, "c": 3}
        result = sort_dict_fields(doc)

        self.assertEqual(list(result.keys()), ["a", "b", "c"])
        self.assertEqual(dict(result), doc)

    def test_sort_empty_dict(self):
        """Test sorting an empty dictionary."""
        doc = {}
        result = sort_dict_fields(doc)

        self.assertIsInstance(result, OrderedDict)
        self.assertEqual(len(result), 0)

    def test_sort_preserves_values(self):
        """Test that sorting preserves all value types."""
        doc = {
            "z_string": "test",
            "a_int": 42,
            "m_float": 3.14,
            "b_bool": True,
            "n_none": None,
            "d_list": [1, 2, 3],
            "c_dict": {"nested": "value"}
        }
        result = sort_dict_fields(doc)

        self.assertEqual(list(result.keys()), ["a_int", "b_bool", "c_dict", "d_list", "m_float", "n_none", "z_string"])
        self.assertEqual(result["a_int"], 42)
        self.assertEqual(result["b_bool"], True)
        self.assertEqual(result["c_dict"], {"nested": "value"})
        self.assertEqual(result["d_list"], [1, 2, 3])
        self.assertEqual(result["m_float"], 3.14)
        self.assertIsNone(result["n_none"])
        self.assertEqual(result["z_string"], "test")

    def test_sort_numeric_string_keys(self):
        """Test sorting with numeric string keys."""
        doc = {"10": "ten", "2": "two", "1": "one"}
        result = sort_dict_fields(doc)

        # String sorting: "1" < "10" < "2"
        self.assertEqual(list(result.keys()), ["1", "10", "2"])

    def test_sort_case_sensitive(self):
        """Test that sorting is case-sensitive."""
        doc = {"Z": 1, "a": 2, "A": 3, "z": 4}
        result = sort_dict_fields(doc)

        # Uppercase comes before lowercase in ASCII
        self.assertEqual(list(result.keys()), ["A", "Z", "a", "z"])


if __name__ == "__main__":
    unittest.main()
