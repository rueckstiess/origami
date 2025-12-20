"""ORIGAMI dataset utilities.

Provides a dataset wrapper for training and evaluation with optional key-order shuffling.
"""

from typing import TYPE_CHECKING

from torch.utils.data import Dataset

if TYPE_CHECKING:
    from origami.tokenizer.json_tokenizer import JSONTokenizer, TokenizedInstance


class OrigamiDataset(Dataset):
    """Dataset for ORIGAMI training and evaluation.

    When shuffle=True, each access returns a fresh key-order permutation
    for data augmentation. This forces the model to learn from key semantics
    rather than position.

    When shuffle=False, tokenization is deterministic for reproducible evaluation.

    Attributes:
        data: List of JSON objects
        tokenizer: JSONTokenizer for tokenization
        shuffle: Whether to shuffle key order during tokenization
    """

    def __init__(
        self,
        data: list[dict],
        tokenizer: "JSONTokenizer",
        shuffle: bool = True,
    ):
        """Initialize dataset.

        Args:
            data: List of JSON objects
            tokenizer: Tokenizer for converting objects to tokens
            shuffle: Whether to shuffle key order during tokenization (default True).
                     Use True for training, False for evaluation.
        """
        self.data = data
        self.tokenizer = tokenizer
        self.shuffle = shuffle

    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.data)

    def __getitem__(self, idx: int) -> "TokenizedInstance":
        """Get item with optionally shuffled key order.

        When shuffle=True, each access returns a fresh shuffle permutation.
        When shuffle=False, tokenization is deterministic.

        Args:
            idx: Index in range [0, len(data))

        Returns:
            TokenizedInstance with shuffled or deterministic key order
        """
        obj = self.data[idx]
        return self.tokenizer.tokenize(obj, shuffle=self.shuffle)
