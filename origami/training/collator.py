"""ORIGAMI data collation.

Provides collation utilities for batching tokenized instances.
"""

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from origami.constraints.json_grammar import JSONGrammarPDA
    from origami.constraints.schema_pda import SchemaPDA
    from origami.tokenizer.json_tokenizer import EncodedBatch, JSONTokenizer, TokenizedInstance


class OrigamiDataCollator:
    """Single source of EncodedBatch creation for training and inference.

    Takes a list of TokenizedInstance objects from the dataset and
    creates batched tensors ready for model input. Uses LEFT-PADDING
    so all sequences end at the same position, enabling easy batched
    prediction where `logits[:, -1, :]` gives the next token for all.

    This is the SINGLE code path for creating batched tensors from
    tokenized instances, used by both training (DataLoader) and
    inference (Generator, Predictor, Embedder).

    Attributes:
        tokenizer: JSONTokenizer for vocabulary and path encoding
        max_length: Maximum sequence length (truncate if exceeded)
        include_labels: Whether to include labels tensor (for training)
        device: Device for output tensors
        grammar_pda: Optional PDA for computing grammar masks during collation.
            When provided, grammar masks are pre-computed in DataLoader workers
            for parallel processing.
        schema_pda: Optional SchemaPDA for computing schema masks during collation.
            When provided, schema masks are computed via mask table gather and
            intersected with grammar masks.
    """

    def __init__(
        self,
        tokenizer: "JSONTokenizer",
        max_length: int | None = None,
        include_labels: bool = True,
        device: torch.device | None = None,
        grammar_pda: "JSONGrammarPDA | None" = None,
        schema_pda: "SchemaPDA | None" = None,
    ):
        """Initialize collator.

        Args:
            tokenizer: Tokenizer with vocabulary for encoding
            max_length: Optional max sequence length for truncation
            include_labels: If True, include labels tensor (for training).
                           If False, labels will be None (for inference).
            device: Device for output tensors (default: CPU)
            grammar_pda: Optional PDA for pre-computing grammar masks.
                When provided and using DataLoader with num_workers > 0,
                grammar masks are computed in parallel by worker processes.
            schema_pda: Optional SchemaPDA for pre-computing schema masks.
                When provided, schema masks are computed via mask table gather
                and intersected with grammar masks.
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.include_labels = include_labels
        self.device = device
        self.grammar_pda = grammar_pda
        self.schema_pda = schema_pda

    def __call__(
        self,
        instances: list["TokenizedInstance"],
    ) -> "EncodedBatch":
        """Collate tokenized instances into a batch.

        Args:
            instances: List of TokenizedInstance from dataset

        Returns:
            EncodedBatch with all tensors ready for model.forward().
            If include_labels=True, labels tensor is set to input_ids.clone().
            If include_labels=False, labels tensor is None.

            Note: The model handles shift internally for both discrete labels
            and numeric values during loss computation.
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

        # Numeric values for continuous head
        numeric_values = torch.zeros(batch_size, max_seq_len, dtype=torch.float)
        numeric_mask = torch.zeros(batch_size, max_seq_len, dtype=torch.bool)

        # Fill tensors with LEFT-PADDING
        # Content is placed at the END of the sequence, PADs at the START
        for b, (token_ids, paths, inst) in enumerate(
            zip(batch_token_ids, batch_paths, instances, strict=True)
        ):
            seq_len = len(token_ids)
            # Left-pad: content goes at positions [max_seq_len - seq_len : max_seq_len]
            start_pos = max_seq_len - seq_len
            input_ids[b, start_pos:] = torch.tensor(token_ids, dtype=torch.long)
            attention_mask[b, start_pos:] = True

            # Fill numeric values at the correct left-padded positions
            for t, num_val in enumerate(inst.numeric_values[:seq_len]):
                if num_val is not None:
                    pos = start_pos + t
                    numeric_values[b, pos] = num_val
                    numeric_mask[b, pos] = True

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

        # Compute grammar mask if PDA is provided (for parallel processing in DataLoader workers)
        grammar_mask = None
        if self.grammar_pda is not None:
            # Grammar computation runs on CPU - this is the expensive operation
            # that we're parallelizing across DataLoader workers.
            grammar_mask = self.grammar_pda.compute_valid_mask(input_ids)

        # Compute schema mask if SchemaPDA is provided
        schema_mask = None
        if self.schema_pda is not None:
            start_positions = [max_seq_len - len(ids) for ids in batch_token_ids]
            schema_indices = self.schema_pda.resolve_mask_indices(
                batch_paths, batch_size, max_seq_len, start_positions
            )
            schema_mask = self.schema_pda.gather_masks(schema_indices)

        # Intersect grammar and schema masks
        if grammar_mask is not None and schema_mask is not None:
            grammar_mask = grammar_mask & schema_mask
        elif schema_mask is not None:
            grammar_mask = schema_mask

        # Move to device if specified
        if self.device is not None:
            input_ids = input_ids.to(self.device)
            path_types = path_types.to(self.device)
            path_ids = path_ids.to(self.device)
            path_lengths = path_lengths.to(self.device)
            attention_mask = attention_mask.to(self.device)
            numeric_values = numeric_values.to(self.device)
            numeric_mask = numeric_mask.to(self.device)
            lengths = lengths.to(self.device)
            if grammar_mask is not None:
                grammar_mask = grammar_mask.to(self.device)

        from origami.tokenizer.json_tokenizer import EncodedBatch

        return EncodedBatch(
            input_ids=input_ids,
            path_types=path_types,
            path_ids=path_ids,
            path_lengths=path_lengths,
            attention_mask=attention_mask,
            numeric_values=numeric_values,
            numeric_mask=numeric_mask,
            lengths=lengths,
            labels=input_ids.clone() if self.include_labels else None,
            grammar_mask=grammar_mask,
        )

    def collate_objects(
        self,
        objects: list[dict],
        shuffle: bool = False,
    ) -> "EncodedBatch":
        """Convenience method: tokenize objects then collate.

        Used by inference components (Generator, Predictor, Embedder).

        Args:
            objects: List of JSON-like dictionaries to encode.
            shuffle: If True, randomly permute key order at each level.

        Returns:
            EncodedBatch ready for model input.
        """
        instances = [self.tokenizer.tokenize(obj, shuffle=shuffle) for obj in objects]
        return self(instances)
