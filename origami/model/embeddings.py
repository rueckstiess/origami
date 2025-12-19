"""ORIGAMI embedding layer.

Combines token embeddings with Key-Value Position Encoding (KVPE).
"""

import torch
import torch.nn as nn
from torch import Tensor

from origami.config import ModelConfig
from origami.position_encoding import KeyValuePositionEncoding


class OrigamiEmbeddings(nn.Module):
    """Token embeddings + KVPE position encoding.

    This module:
    1. Embeds input tokens using learned embeddings
    2. Computes position embeddings from JSON paths via KVPE
    3. Adds token and position embeddings together
    4. For NUM tokens (when continuous head enabled), uses multiplicative
       embedding: scaled_value × learnable_vector

    The token embedding layer is shared with KVPE for key position encoding,
    enabling transfer learning where the model recognizes "name" in position
    encoding as the same concept as the "name" token.

    Attributes:
        token_embedding: Learned token embedding layer
        kvpe: Key-Value Position Encoding module
        dropout: Dropout layer
        num_embedding: Learnable direction vector for scaled numeric values
            (only present when use_continuous_head=True)
    """

    # NUM token ID is fixed at 9 in the vocabulary
    NUM_TOKEN_ID = 9

    def __init__(self, config: ModelConfig, vocab_size: int):
        """Initialize embedding layer.

        Args:
            config: Model configuration
            vocab_size: Size of the vocabulary
        """
        super().__init__()

        self.config = config

        # Token embeddings (shared with KVPE for key position encoding)
        self.token_embedding = nn.Embedding(vocab_size, config.d_model)

        # KVPE with shared key embeddings
        self.kvpe = KeyValuePositionEncoding(
            d_model=config.d_model,
            vocab_size=vocab_size,
            max_depth=config.max_depth,
            max_array_index=config.max_array_position,
            pooling=config.kvpe_pooling,
            share_key_embeddings=True,
            **config.kvpe_pooling_kwargs,
        )
        # Share token embeddings with KVPE for key position encoding
        self.kvpe.set_key_embeddings(self.token_embedding)

        self.dropout = nn.Dropout(config.dropout)

        # Learnable direction vector for scaled numeric values
        # Embedding = scaled_value × num_embedding
        if config.use_continuous_head:
            self.num_embedding = nn.Parameter(torch.randn(config.d_model) * 0.02)

    def forward(
        self,
        input_ids: Tensor,  # (batch, seq_len)
        path_types: Tensor,  # (batch, seq_len, max_depth)
        path_ids: Tensor,  # (batch, seq_len, max_depth)
        path_lengths: Tensor,  # (batch, seq_len)
        numeric_values: Tensor | None = None,  # (batch, seq_len) - scaled values for NUM tokens
    ) -> Tensor:
        """Compute embeddings from tokens and paths.

        Args:
            input_ids: Token IDs of shape (batch, seq_len)
            path_types: Path element types (0=pad, 1=key, 2=index)
                of shape (batch, seq_len, max_depth)
            path_ids: Path element IDs (vocab ID for keys, index for arrays)
                of shape (batch, seq_len, max_depth)
            path_lengths: Number of valid elements in each path
                of shape (batch, seq_len)
            numeric_values: Scaled numeric values for NUM token positions.
                Only used when use_continuous_head=True. Shape (batch, seq_len).

        Returns:
            Combined embeddings of shape (batch, seq_len, d_model)
        """
        # 1. Token embeddings
        embeds = self.token_embedding(input_ids)  # (batch, seq_len, d_model)

        # 2. Replace NUM token embeddings with multiplicative numeric embeddings
        # Embedding = scaled_value × learnable_direction_vector
        if hasattr(self, "num_embedding") and numeric_values is not None:
            is_num = input_ids == self.NUM_TOKEN_ID  # (batch, seq_len)
            if is_num.any():
                # Compute numeric embedding: scaled_value × num_embedding
                # numeric_values: (batch, seq_len) → (batch, seq_len, 1)
                # num_embedding: (d_model,) → broadcasts
                num_embeds = numeric_values.unsqueeze(-1) * self.num_embedding
                # Replace embeddings at NUM positions
                embeds = torch.where(is_num.unsqueeze(-1), num_embeds, embeds)

        # 3. Add position encoding (KVPE)
        pos_embeds = self.kvpe(path_types, path_ids, path_lengths)

        return self.dropout(embeds + pos_embeds)
