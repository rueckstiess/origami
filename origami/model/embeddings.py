"""ORIGAMI embedding layer.

Combines token embeddings with position encoding (KVPE or sequential).
"""

import torch
import torch.nn as nn
from torch import Tensor

from origami.config import ModelConfig
from origami.position_encoding import KeyValuePositionEncoding


class OrigamiEmbeddings(nn.Module):
    """Token embeddings + position encoding.

    This module:
    1. Embeds input tokens using learned embeddings
    2. Computes position embeddings via KVPE (JSON paths) or sequential positions
    3. Adds token and position embeddings together
    4. For NUM tokens (when continuous head enabled), uses multiplicative
       embedding: scaled_value × learnable_vector

    When using KVPE, the token embedding layer is shared with KVPE for key
    position encoding, enabling transfer learning where the model recognizes
    "name" in position encoding as the same concept as the "name" token.

    Attributes:
        token_embedding: Learned token embedding layer
        kvpe: Key-Value Position Encoding module (only when position_encoding="kvpe")
        position_embedding: Learned positional embedding (only when position_encoding="sequential")
        dropout: Dropout layer
        num_embedding: Learnable direction vector for scaled numeric values
            (only present when use_continuous_head=True)
    """

    # Fixed grammar token IDs from the vocabulary
    NUM_TOKEN_ID = 9

    def __init__(self, config: ModelConfig, vocab_size: int):
        """Initialize embedding layer.

        Args:
            config: Model configuration
            vocab_size: Size of the vocabulary
        """
        super().__init__()

        self.config = config

        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, config.d_model)

        # Position encoding
        if config.position_encoding == "kvpe":
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
        else:  # "sequential"
            self.position_embedding = nn.Embedding(config.max_seq_length, config.d_model)
            # self.position_embedding.weight.requires_grad = False

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
        position_ids: Tensor | None = None,  # (batch, seq_len) - for sequential PE
    ) -> Tensor:
        """Compute embeddings from tokens and positions.

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
            position_ids: Sequential position IDs of shape (batch, seq_len).
                Only used when position_encoding="sequential". Computed by the
                model from attention_mask if not provided.

        Returns:
            Combined embeddings of shape (batch, seq_len, d_model)
        """
        # 1. Token embeddings
        embeds = self.token_embedding(input_ids)  # (batch, seq_len, d_model)

        # 2. Replace NUM token embeddings with multiplicative numeric embeddings
        # Embedding = scaled_value × learnable_direction_vector
        # ARRAY_START tokens keep their learned token embedding — the continuous
        # head predicts array length from context but the embedding doesn't need
        # to encode it (length is enforced via mask overrides during generation).
        if hasattr(self, "num_embedding") and numeric_values is not None:
            is_num = input_ids == self.NUM_TOKEN_ID
            if is_num.any():
                num_embeds = numeric_values.unsqueeze(-1) * self.num_embedding
                embeds = torch.where(is_num.unsqueeze(-1), num_embeds, embeds)

        # 3. Add position encoding
        if self.config.position_encoding == "kvpe":
            pos_embeds = self.kvpe(path_types, path_ids, path_lengths)
        else:  # "sequential"
            pos_embeds = self.position_embedding(position_ids)

        return self.dropout(embeds + pos_embeds)
