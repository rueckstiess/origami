"""Key-Value Position Encoding (KVPE) for JSON paths.

This module encodes hierarchical paths (keys and array indices) as position
embeddings. The path at each token position is encoded into a single vector
that can be added to token embeddings.

Path elements are embedded individually, then pooled using a configurable
strategy (sum, rotary, gru, etc.) to produce the final position embedding.
"""

from typing import Literal

import torch
import torch.nn as nn
from torch import Tensor

from .pooling import PathPooling, create_pooling

# Path type constants
PATH_TYPE_PAD = 0
PATH_TYPE_KEY = 1
PATH_TYPE_INDEX = 2


class KeyValuePositionEncoding(nn.Module):
    """Pluggable position encoding for JSON paths.

    Encodes paths like `user.address.city` or `items[0].name` into position
    embeddings. Keys use vocabulary embeddings (shared or separate), array
    indices use learned embeddings.

    Attributes:
        d_model: Embedding dimension
        max_depth: Maximum path depth
        max_array_index: Maximum array index for position embeddings
        pooling: Pooling strategy for aggregating path elements
        share_key_embeddings: Whether to share embeddings with token layer
    """

    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        max_depth: int = 32,
        max_array_index: int = 256,
        pooling: Literal["sum", "weighted", "rotary", "gru", "transformer"] = "sum",
        share_key_embeddings: bool = True,
        **pooling_kwargs,
    ):
        """Initialize KVPE module.

        Args:
            d_model: Model dimension
            vocab_size: Size of vocabulary (for separate key embeddings)
            max_depth: Maximum nesting depth
            max_array_index: Maximum array index for embeddings
            pooling: Pooling strategy name
            share_key_embeddings: If True, key embeddings are set externally.
                                  If False, create separate embeddings.
            **pooling_kwargs: Additional arguments for pooling strategy
        """
        super().__init__()

        self.d_model = d_model
        self.max_depth = max_depth
        self.max_array_index = max_array_index
        self.share_key_embeddings = share_key_embeddings

        # Index embeddings for array positions
        self.index_embeddings = nn.Embedding(max_array_index, d_model)

        # Key embeddings (shared or separate)
        if share_key_embeddings:
            # Will be set via set_key_embeddings()
            self._shared_key_embeddings: nn.Embedding | None = None
        else:
            # Create separate embeddings for position encoding
            self.key_embeddings = nn.Embedding(vocab_size, d_model)

        # Pooling strategy
        self.pooling: PathPooling = create_pooling(
            pooling,
            d_model=d_model,
            max_depth=max_depth,
            **pooling_kwargs,
        )

        # Zero embedding for padding
        self.register_buffer("zero_embed", torch.zeros(d_model))

    def set_key_embeddings(self, key_embeddings: nn.Embedding) -> None:
        """Share key embeddings with token embedding layer.

        This enables transfer learning where the model recognizes "name" in
        position encoding as the same concept as the "name" token.

        Args:
            key_embeddings: Embedding layer to share (usually token embeddings)

        Raises:
            AssertionError: If share_key_embeddings=False was set in __init__
        """
        if not self.share_key_embeddings:
            raise ValueError("Cannot set shared embeddings when share_key_embeddings=False")
        self._shared_key_embeddings = key_embeddings

    def _get_key_embeddings(self) -> nn.Embedding:
        """Get the key embedding layer (shared or separate)."""
        if self.share_key_embeddings:
            if self._shared_key_embeddings is None:
                raise RuntimeError("Key embeddings not set. Call set_key_embeddings() first.")
            return self._shared_key_embeddings
        else:
            return self.key_embeddings

    def forward(
        self,
        path_types: Tensor,  # (batch, seq_len, max_depth) - 0=pad, 1=key, 2=index
        path_ids: Tensor,  # (batch, seq_len, max_depth) - key vocab ID or array index
        path_lengths: Tensor,  # (batch, seq_len) - path depth at each position
    ) -> Tensor:  # (batch, seq_len, d_model)
        """Compute position embeddings from path information.

        Args:
            path_types: Type of each path element (0=pad, 1=key, 2=index)
            path_ids: ID of each element (vocab ID for keys, index for arrays)
            path_lengths: Number of valid elements in each path

        Returns:
            Position embeddings of shape (batch, seq_len, d_model)
        """
        batch_size, seq_len, max_depth = path_types.shape

        # Embed all path elements
        path_embeds = self._embed_path_elements(path_types, path_ids)

        # Pool into position embeddings
        return self.pooling(path_embeds, path_lengths)

    def _embed_path_elements(
        self,
        path_types: Tensor,  # (batch, seq_len, max_depth)
        path_ids: Tensor,  # (batch, seq_len, max_depth)
    ) -> Tensor:  # (batch, seq_len, max_depth, d_model)
        """Embed each path element based on its type.

        Keys use key embeddings, indices use index embeddings,
        padding positions get zero embeddings.

        Note: Uses element-wise multiplication instead of masked_scatter
        for MPS compatibility. masked_scatter with boolean indexing can
        cause intermittent gradient tracking issues on MPS during backward.
        """
        key_emb = self._get_key_embeddings()

        # Embed all positions as if they were keys (clamped to valid range)
        key_ids_clamped = path_ids.clamp(0, key_emb.num_embeddings - 1)
        key_embeds = key_emb(key_ids_clamped)  # (batch, seq_len, max_depth, d_model)

        # Embed all positions as if they were indices (clamped to valid range)
        index_ids_clamped = path_ids.clamp(0, self.max_array_index - 1)
        index_embeds = self.index_embeddings(index_ids_clamped)

        # Create masks for each type, expanded to embedding dimension
        # Using float conversion for gradient-safe multiplication
        key_mask = (path_types == PATH_TYPE_KEY).unsqueeze(-1).float()
        index_mask = (path_types == PATH_TYPE_INDEX).unsqueeze(-1).float()

        # Select appropriate embeddings using element-wise multiplication
        # Padding positions (type=0) get zeros since neither mask is True
        embeds = key_embeds * key_mask + index_embeds * index_mask

        return embeds


class PathEncoder:
    """Utility class to encode Path tuples into tensor format.

    Converts the Path type from the tokenizer (tuple of KeyElement/IndexElement)
    into the tensor format expected by KVPE:
    - path_types: 0=pad, 1=key, 2=index
    - path_ids: vocab ID for keys, position for indices
    - path_lengths: depth of each path
    """

    def __init__(
        self,
        vocab: "Vocabulary",  # noqa: F821 - forward reference
        max_depth: int = 32,
        max_array_index: int = 256,
    ):
        """Initialize path encoder.

        Args:
            vocab: Vocabulary for encoding keys to IDs
            max_depth: Maximum path depth (truncates longer paths)
            max_array_index: Maximum array index (clamps larger indices)
        """
        self.vocab = vocab
        self.max_depth = max_depth
        self.max_array_index = max_array_index

    def encode_paths(
        self,
        paths: list["Path"],  # noqa: F821 - forward reference
        device: torch.device | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Encode a list of paths into tensors.

        Args:
            paths: List of Path tuples (from tokenizer)
            device: Target device for tensors

        Returns:
            Tuple of (path_types, path_ids, path_lengths) tensors
            All have shape (len(paths), max_depth) except path_lengths: (len(paths),)
        """
        from origami.tokenizer import IndexElement, KeyElement, KeyToken

        seq_len = len(paths)
        path_types = torch.zeros(seq_len, self.max_depth, dtype=torch.long)
        path_ids = torch.zeros(seq_len, self.max_depth, dtype=torch.long)
        path_lengths = torch.zeros(seq_len, dtype=torch.long)

        for i, path in enumerate(paths):
            depth = min(len(path), self.max_depth)
            path_lengths[i] = depth

            for j, element in enumerate(path[:depth]):
                if isinstance(element, KeyElement):
                    path_types[i, j] = PATH_TYPE_KEY
                    # Encode key to vocab ID
                    key_token = KeyToken(element.key)
                    path_ids[i, j] = self.vocab.encode(key_token)
                elif isinstance(element, IndexElement):
                    path_types[i, j] = PATH_TYPE_INDEX
                    # Clamp index to valid range
                    path_ids[i, j] = min(element.index, self.max_array_index - 1)

        if device is not None:
            path_types = path_types.to(device)
            path_ids = path_ids.to(device)
            path_lengths = path_lengths.to(device)

        return path_types, path_ids, path_lengths

    def encode_batch(
        self,
        batch_paths: list[list["Path"]],  # noqa: F821
        device: torch.device | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Encode a batch of path sequences.

        Args:
            batch_paths: List of path sequences (one per instance in batch)
            device: Target device for tensors

        Returns:
            Tuple of tensors with batch dimension:
            - path_types: (batch, max_seq_len, max_depth)
            - path_ids: (batch, max_seq_len, max_depth)
            - path_lengths: (batch, max_seq_len)
        """
        batch_size = len(batch_paths)
        max_seq_len = max(len(paths) for paths in batch_paths)

        path_types = torch.zeros(batch_size, max_seq_len, self.max_depth, dtype=torch.long)
        path_ids = torch.zeros(batch_size, max_seq_len, self.max_depth, dtype=torch.long)
        path_lengths = torch.zeros(batch_size, max_seq_len, dtype=torch.long)

        for b, paths in enumerate(batch_paths):
            pt, pi, pl = self.encode_paths(paths)
            seq_len = len(paths)
            path_types[b, :seq_len] = pt
            path_ids[b, :seq_len] = pi
            path_lengths[b, :seq_len] = pl

        if device is not None:
            path_types = path_types.to(device)
            path_ids = path_ids.to(device)
            path_lengths = path_lengths.to(device)

        return path_types, path_ids, path_lengths
