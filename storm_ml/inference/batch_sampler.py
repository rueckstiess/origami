import math
from collections import Counter
from typing import Iterator

import torch
from torch.utils.data import Sampler

from storm_ml.preprocessing import DFDataset


class TargetTokenBatchSampler(Sampler):
    """A sampler that batches documents with the same target token position together.

    Use with PyTorch DataLoader and set batch_sampler=TargetTokenBatchSampler(...)
    """

    def __init__(self, data: DFDataset | torch.Tensor, target_token_id: int, max_batch_size: int) -> None:
        if isinstance(data, DFDataset):
            data = data.tokens

        self.max_batch_size = max_batch_size

        # Create a boolean mask where the token ID is equal to `target_token_id`
        tok_mask = data == target_token_id

        # Find position of `target_token_id` in each row
        tok_idxs = torch.argmax(tok_mask.float(), dim=1)

        # sort and count groups
        self.sorted_ids, self.sorted_idxs = torch.sort(tok_idxs, descending=True, stable=True)
        self.batch_counter = Counter(self.sorted_ids.tolist())

        # calculate number of batches
        self.total_batches = sum(math.ceil(count / max_batch_size) for count in self.batch_counter.values())

    def __len__(self) -> int:
        return self.total_batches

    def __iter__(self) -> Iterator[list[int]]:
        unique_lengths = list(self.batch_counter.keys())

        for length in unique_lengths:
            # Get indices of sequences with the current length
            length_indices = self.sorted_idxs[self.sorted_ids == length]

            # Chunk the indices into batches
            chunks = torch.chunk(length_indices, math.ceil(len(length_indices) / self.max_batch_size))

            for chunk in chunks:
                yield chunk.tolist()
