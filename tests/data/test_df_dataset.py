import tempfile
import unittest

import pandas as pd
import torch

from storm_ml.data import DFDataset


class TestDFDataset(unittest.TestCase):
    def test_df_dataset_init(self):
        df = pd.DataFrame({"tokens": [[1, 2, 3, 4], [5, 4, 3, 2], [1, 3, 2, 5]]})
        ds = DFDataset(df)

        self.assertEqual(len(df), 3)
        self.assertIsInstance(ds.tokens, torch.Tensor)
        self.assertEqual(ds.tokens.shape, (3, 4))

    def test_df_dataset_getitem(self):
        df = pd.DataFrame({"tokens": [[1, 2, 3, 4], [5, 4, 3, 2], [1, 3, 2, 5]]})
        ds = DFDataset(df)

        # single index
        self.assertEqual(ds[0].tolist(), [1, 2, 3, 4])
        self.assertEqual(ds[1].tolist(), [5, 4, 3, 2])
        self.assertEqual(ds[2].tolist(), [1, 3, 2, 5])

        # index with slice
        self.assertEqual(ds[0:2].tolist(), [[1, 2, 3, 4], [5, 4, 3, 2]])
        self.assertEqual(ds[1:3].tolist(), [[5, 4, 3, 2], [1, 3, 2, 5]])

        # index with list
        self.assertEqual(ds[[0, 2]].tolist(), [[1, 2, 3, 4], [1, 3, 2, 5]])
        self.assertEqual(ds[[1, 0]].tolist(), [[5, 4, 3, 2], [1, 2, 3, 4]])

    def test_df_dataset_sample(self):
        tokens = torch.rand((100, 3))
        df = pd.DataFrame({"tokens": tokens.tolist()})
        original = DFDataset(df)

        # sample by number
        sampled = original.sample(n=5)
        self.assertEqual(sampled.tokens.shape, (5, 3))
        self.assertEqual(len(original), 100)
        self.assertEqual(len(sampled), 5)

        # sample by fraction
        sampled = original.sample(frac=0.5)
        self.assertEqual(sampled.tokens.shape, (50, 3))
        self.assertEqual(len(original), 100)
        self.assertEqual(len(sampled), 50)

    def test_df_dataset_save_load(self):
        # with temporary file
        df = pd.DataFrame({"tokens": [[1, 2, 3, 4], [5, 4, 3, 2], [1, 3, 2, 5]]})
        ds = DFDataset(df)

        with tempfile.NamedTemporaryFile() as f:
            ds.save(f.name, compression=None)
            ds = DFDataset.load(f.name, compression=None)

        self.assertEqual(len(df), 3)
        self.assertIsInstance(ds.tokens, torch.Tensor)
        self.assertEqual(ds.tokens.shape, (3, 4))
