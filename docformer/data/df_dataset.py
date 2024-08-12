import gzip
import lzma
import pickle

import pandas as pd
import torch
from torch.utils.data import Dataset


class DFDataset(Dataset):
    """PyTorch Dataset class for a DataFrame containing a column of tokens."""

    def __init__(self, df: pd.DataFrame, token_column: str = "tokens"):
        self.df = df
        # for efficiency we preprocess the token column into a tensor
        self.tokens = torch.tensor([toks for toks in df[token_column]])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        return self.tokens[idx]

    def sample(self, *args, **kwargs) -> "DFDataset":
        return DFDataset(self.df.sample(*args, **kwargs))

    def save(self, path: str, compression: None | str = "gzip") -> None:
        """saves the dataset to disk. compression options are:
        - None : uncompressed (fastest, but no compression)
        - gzip : use gzip compression (in the middle)
        - lzma : use lzma compression (slowest, but smallest file size)
        """

        if compression == "gzip":
            with gzip.open(path, "wb") as f:
                pickle.dump(self, f)

        elif compression == "lzma":
            with lzma.open(path, "wb") as f:
                pickle.dump(self, f)
        else:
            with open(path, "wb") as f:
                pickle.dump(self, f)

    @staticmethod
    def load(path: str, compression: None | str = "gzip") -> "DFDataset":
        """loads the dataset from file and creates a new DFDataset instance."""

        if compression == "gzip":
            with gzip.open(path, "rb") as f:
                dataset = pickle.load(f)

        elif compression == "lzma":
            with lzma.open(path, "rb") as f:
                dataset = pickle.load(f)

        else:
            with open(path, "rb") as f:
                dataset = pickle.load(f)

        return dataset
