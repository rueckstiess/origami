import unittest

import torch
from pandas import DataFrame

from storm_ml.data.utils import detokenize, docs_to_df, target_collate_fn, tokenize
from storm_ml.utils import ArrayStart, FieldToken, Symbol


class TestDataUtils(unittest.TestCase):
    def test_docs_to_df(self):
        docs = [{"a": 1}, {"a": 2, "b": 3}]
        df = docs_to_df(docs)

        self.assertTrue(isinstance(df, DataFrame))
        self.assertEqual(df.shape, (2, 2))
        self.assertTrue("id" in df.columns)
        self.assertTrue("docs" in df.columns)
        self.assertEqual(df.loc[0, "docs"], {"a": 1})
        self.assertEqual(df.loc[1, "docs"], {"a": 2, "b": 3})

    def test_tokenize(self):
        t = tokenize({"foo": 1, "bar": [1, 2, 3], "baz": {"baz_inner": True}})

        self.assertEqual(
            t,
            [
                Symbol.START,
                FieldToken("foo"),
                1,
                FieldToken("bar"),
                ArrayStart(3),
                1,
                2,
                3,
                FieldToken("baz"),
                Symbol.SUBDOC_START,
                FieldToken("baz.baz_inner"),
                True,
                Symbol.SUBDOC_END,
                Symbol.END,
            ],
        )

    def test_tokenize_without_path(self):
        t = tokenize({"foo": 1, "bar": [1, 2, 3], "baz": {"baz_inner": True}}, path_in_field_tokens=False)

        self.assertEqual(
            t,
            [
                Symbol.START,
                FieldToken("foo"),
                1,
                FieldToken("bar"),
                ArrayStart(3),
                1,
                2,
                3,
                FieldToken("baz"),
                Symbol.SUBDOC_START,
                FieldToken("baz_inner"),
                True,
                Symbol.SUBDOC_END,
                Symbol.END,
            ],
        )

    def test_detokenize(self):
        t = detokenize(
            [
                Symbol.START,
                FieldToken("foo"),
                1,
                FieldToken("bar"),
                ArrayStart(3),
                1,
                2,
                3,
                FieldToken("baz"),
                Symbol.SUBDOC_START,
                FieldToken("baz.baz_inner"),
                True,
                Symbol.SUBDOC_END,
                Symbol.END,
            ],
        )
        self.assertEqual(t, {"foo": 1, "bar": [1, 2, 3], "baz": {"baz_inner": True}})

    def test_detokenize_without_path(self):
        t = detokenize(
            [
                Symbol.START,
                FieldToken("foo"),
                1,
                FieldToken("bar"),
                ArrayStart(3),
                1,
                2,
                3,
                FieldToken("baz"),
                Symbol.SUBDOC_START,
                FieldToken("baz_inner"),
                True,
                Symbol.SUBDOC_END,
                Symbol.END,
            ],
        )
        self.assertEqual(t, {"foo": 1, "bar": [1, 2, 3], "baz": {"baz_inner": True}})

    def test_target_collate_fn(self):
        # test that the collate_fn returned by target_collate_fn() works correctly

        t = [torch.tensor([1, 2, 3, 4, 5, 7]), torch.tensor([2, 1, 0, 4, 6, 8])]
        collate_fn = target_collate_fn(4)

        result = collate_fn(t)
        self.assertTrue(torch.equal(result, torch.tensor([[1, 2, 3, 4], [2, 1, 0, 4]])))
