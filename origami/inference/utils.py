"""Shared utilities for inference components.

Provides common functionality used across Generator, Predictor, and Embedder.
"""

from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from origami.tokenizer.vocabulary import Vocabulary


class GenerationError(Exception):
    """Raised when generation or decoding fails.

    This replaces silent None returns with explicit errors,
    making debugging easier and preventing silent failures.

    Attributes:
        message: Human-readable error description
        token_ids: Full token sequence being decoded (if available)
        position: Position in token sequence where error occurred
        context_window: Number of tokens to show around error position
    """

    def __init__(
        self,
        message: str,
        token_ids: list[int] | None = None,
        position: int | None = None,
        vocab: "Vocabulary | None" = None,
    ):
        self.message = message
        self.token_ids = token_ids
        self.position = position
        self.vocab = vocab

        # Build detailed error message
        full_message = message
        if token_ids is not None:
            full_message += f"\n  Token sequence length: {len(token_ids)}"
            full_message += f"\n  Token IDs: {token_ids}"
            if vocab is not None:
                # Decode tokens to readable format
                decoded = []
                for i, tid in enumerate(token_ids):
                    try:
                        token = vocab.decode(tid)
                        marker = " <-- ERROR" if i == position else ""
                        decoded.append(f"    [{i}] {tid}: {token!r}{marker}")
                    except KeyError:
                        marker = " <-- ERROR" if i == position else ""
                        decoded.append(f"    [{i}] {tid}: <unknown>{marker}")
                full_message += "\n  Decoded tokens:\n" + "\n".join(decoded)
            if position is not None:
                full_message += f"\n  Error at position: {position}"

        super().__init__(full_message)


def find_target_positions(
    input_ids: Tensor,
    target_key: str,
    vocab: "Vocabulary",
) -> Tensor:
    """Find position of target key token in each sequence.

    Used by Predictor and Embedder to locate where the target key
    appears in tokenized sequences.

    Args:
        input_ids: (batch, seq_len) token IDs
        target_key: Key to find (dot notation → uses leaf key only)
        vocab: Vocabulary for encoding the key token

    Returns:
        Tensor of shape (batch,) with position indices

    Raises:
        ValueError: If target key not found in any sequence
    """
    from origami.tokenizer.vocabulary import KeyToken

    # Get the leaf key (last part of dot-separated path)
    leaf_key = target_key.split(".")[-1]
    key_token = KeyToken(leaf_key)
    key_id = vocab.encode(key_token)

    batch_size = input_ids.size(0)
    positions = torch.zeros(batch_size, dtype=torch.long, device=input_ids.device)

    for i in range(batch_size):
        matches = (input_ids[i] == key_id).nonzero(as_tuple=True)[0]
        if len(matches) == 0:
            raise ValueError(f"Target key '{leaf_key}' not found in sequence {i}")
        # Use the last occurrence (after move_target_last, target is last)
        positions[i] = matches[-1]

    return positions
