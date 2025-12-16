"""ORIGAMI dataset utilities.

Provides dataset wrappers for upscaling via key-order shuffling.
"""

from typing import TYPE_CHECKING

from torch.utils.data import Dataset

if TYPE_CHECKING:
    from origami.tokenizer.json_tokenizer import JSONTokenizer, TokenizedInstance


class UpscaledDataset(Dataset):
    """Dataset wrapper that presents upscaled view via shuffle permutations.

    Each base item appears `upscale_factor` times with different key-order
    shuffles. This is a core data augmentation technique for ORIGAMI that
    forces the model to learn from key semantics rather than position.

    From the original paper (Section 4.4.1):
    - Without shuffling: Model overfits to key order, poor generalization
    - With shuffling + high upscale (100x+): Best generalization

    Recommended upscale factors by dataset size:
    - < 500 samples: 100-1000
    - 500-5000 samples: 10-100
    - 5000-50000 samples: 1-10
    - > 50000 samples: 1 (shuffling alone suffices)

    Attributes:
        base_data: List of JSON objects
        tokenizer: JSONTokenizer for tokenization
        upscale_factor: Number of times each item appears with different shuffles
    """

    def __init__(
        self,
        base_data: list[dict],
        tokenizer: "JSONTokenizer",
        upscale_factor: int = 1,
        shuffle: bool = True,
    ):
        """Initialize upscaled dataset.

        Args:
            base_data: List of JSON objects to present
            tokenizer: Tokenizer for converting objects to tokens
            upscale_factor: Multiplier for dataset size (default 1 = no upscaling)
            shuffle: Whether to shuffle key order during tokenization (default True).
                     If False, upscale_factor is forced to 1 since upscaling without
                     shuffling would just duplicate identical samples.
        """
        if upscale_factor < 1:
            raise ValueError(f"upscale_factor must be >= 1, got {upscale_factor}")

        self.base_data = base_data
        self.tokenizer = tokenizer
        self.shuffle = shuffle
        # Upscaling without shuffling doesn't make sense - every copy would be identical
        self.upscale_factor = upscale_factor if shuffle else 1

    def __len__(self) -> int:
        """Return upscaled length."""
        return len(self.base_data) * self.upscale_factor

    def __getitem__(self, idx: int) -> "TokenizedInstance":
        """Get item with optionally shuffled key order.

        When shuffle=True, each access returns a fresh shuffle permutation.
        When upscale_factor > 1, the same base object can be accessed at
        multiple indices, each returning a different shuffle.

        Args:
            idx: Index in range [0, len(base_data) * upscale_factor)

        Returns:
            TokenizedInstance with shuffled or deterministic key order
        """
        base_idx = idx // self.upscale_factor
        obj = self.base_data[base_idx]
        return self.tokenizer.tokenize(obj, shuffle=self.shuffle)

    def get_base_item(self, idx: int) -> dict:
        """Get the original JSON object at base index.

        Args:
            idx: Index in range [0, len(base_data))

        Returns:
            Original JSON object
        """
        return self.base_data[idx]

    @property
    def base_size(self) -> int:
        """Return size of underlying dataset without upscaling."""
        return len(self.base_data)


class EvalDataset(Dataset):
    """Dataset for evaluation with deterministic tokenization.

    Unlike UpscaledDataset, this does not shuffle keys, providing
    deterministic tokenization for reproducible evaluation.

    Attributes:
        data: List of JSON objects
        tokenizer: JSONTokenizer for tokenization
    """

    def __init__(
        self,
        data: list[dict],
        tokenizer: "JSONTokenizer",
    ):
        """Initialize evaluation dataset.

        Args:
            data: List of JSON objects
            tokenizer: Tokenizer for converting objects to tokens
        """
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.data)

    def __getitem__(self, idx: int) -> "TokenizedInstance":
        """Get item with deterministic key order.

        Args:
            idx: Index in range [0, len(data))

        Returns:
            TokenizedInstance with deterministic key order
        """
        obj = self.data[idx]
        # No shuffle for evaluation - deterministic order
        return self.tokenizer.tokenize(obj, shuffle=False)
