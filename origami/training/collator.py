"""ORIGAMI data collation.

Provides collation utilities for batching tokenized instances.
"""

from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from origami.tokenizer.json_tokenizer import JSONTokenizer, TokenizedInstance


class OrigamiDataCollator:
    """Collator for batching tokenized JSON instances.

    Takes a list of TokenizedInstance objects from the dataset and
    creates batched tensors ready for model input. Uses LEFT-PADDING
    so all sequences end at the same position, enabling easy batched
    prediction where `logits[:, -1, :]` gives the next token for all.

    This separates the tokenization (with shuffling) from batching,
    allowing the dataset to control key-order permutations while
    the collator handles padding and tensor creation.

    Attributes:
        tokenizer: JSONTokenizer for vocabulary and path encoding
        max_length: Maximum sequence length (truncate if exceeded)
        device: Device for output tensors
    """

    def __init__(
        self,
        tokenizer: "JSONTokenizer",
        max_length: int | None = None,
        device: torch.device | None = None,
    ):
        """Initialize collator.

        Args:
            tokenizer: Tokenizer with vocabulary for encoding
            max_length: Optional max sequence length for truncation
            device: Device for output tensors (default: CPU)
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device

    def __call__(
        self,
        instances: list["TokenizedInstance"],
    ) -> dict[str, Tensor]:
        """Collate tokenized instances into a batch.

        Args:
            instances: List of TokenizedInstance from dataset

        Returns:
            Dictionary with batched tensors ready for model.forward():
                - input_ids: (batch, seq_len)
                - path_types: (batch, seq_len, max_depth)
                - path_ids: (batch, seq_len, max_depth)
                - path_lengths: (batch, seq_len)
                - attention_mask: (batch, seq_len)
                - labels: (batch, seq_len) - same as input_ids for autoregressive
        """
        from origami.position_encoding import PATH_TYPE_INDEX, PATH_TYPE_KEY
        from origami.tokenizer.path import IndexElement, KeyElement
        from origami.tokenizer.vocabulary import KeyToken

        if not instances:
            raise ValueError("Cannot collate empty batch")

        # Convert tokens to IDs
        batch_token_ids = [self.tokenizer.encode_tokens(inst) for inst in instances]
        batch_paths = [inst.paths for inst in instances]

        # Determine dimensions
        batch_size = len(instances)
        max_seq_len = max(len(ids) for ids in batch_token_ids)

        # Apply max_length truncation if specified
        if self.max_length is not None:
            max_seq_len = min(max_seq_len, self.max_length)
            batch_token_ids = [ids[:max_seq_len] for ids in batch_token_ids]
            batch_paths = [paths[:max_seq_len] for paths in batch_paths]

        lengths = torch.tensor([len(ids) for ids in batch_token_ids], dtype=torch.long)

        # Initialize tensors
        vocab = self.tokenizer.vocab
        max_depth = self.tokenizer.max_depth

        input_ids = torch.full((batch_size, max_seq_len), vocab.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros(batch_size, max_seq_len, dtype=torch.bool)
        path_types = torch.zeros(batch_size, max_seq_len, max_depth, dtype=torch.long)
        path_ids = torch.zeros(batch_size, max_seq_len, max_depth, dtype=torch.long)
        path_lengths = torch.zeros(batch_size, max_seq_len, dtype=torch.long)

        # Fill tensors with LEFT-PADDING
        # Content is placed at the END of the sequence, PADs at the START
        for b, (token_ids, paths) in enumerate(zip(batch_token_ids, batch_paths, strict=True)):
            seq_len = len(token_ids)
            # Left-pad: content goes at positions [max_seq_len - seq_len : max_seq_len]
            start_pos = max_seq_len - seq_len
            input_ids[b, start_pos:] = torch.tensor(token_ids, dtype=torch.long)
            attention_mask[b, start_pos:] = True

            # Encode paths at the correct (left-padded) positions
            for t, path in enumerate(paths):
                pos = start_pos + t  # Actual position in padded sequence
                depth = min(len(path), max_depth)
                path_lengths[b, pos] = depth

                for d, element in enumerate(path[:depth]):
                    if isinstance(element, KeyElement):
                        path_types[b, pos, d] = PATH_TYPE_KEY
                        key_token = KeyToken(element.key)
                        path_ids[b, pos, d] = vocab.encode(key_token)
                    elif isinstance(element, IndexElement):
                        path_types[b, pos, d] = PATH_TYPE_INDEX
                        path_ids[b, pos, d] = min(element.index, self.tokenizer.max_array_index - 1)

        # Move to device if specified
        if self.device is not None:
            input_ids = input_ids.to(self.device)
            path_types = path_types.to(self.device)
            path_ids = path_ids.to(self.device)
            path_lengths = path_lengths.to(self.device)
            attention_mask = attention_mask.to(self.device)
            lengths = lengths.to(self.device)

        return {
            "input_ids": input_ids,
            "path_types": path_types,
            "path_ids": path_ids,
            "path_lengths": path_lengths,
            "attention_mask": attention_mask,
            "labels": input_ids.clone(),  # For autoregressive training
            "lengths": lengths,
        }
