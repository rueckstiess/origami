"""Pooling strategies for KVPE path embeddings.

These strategies aggregate path element embeddings into a single position
embedding vector. Different strategies have different properties:

- sum: Commutative baseline (original paper)
- weighted: Learned depth weights, partial order-awareness
- rotary: Rotation by depth, parallelizable
- gru: Sequential processing, fully non-commutative
- transformer: Self-attention over path elements
"""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch import Tensor


class PathPooling(nn.Module, ABC):
    """Base class for path pooling strategies.

    All pooling strategies take path element embeddings and path lengths,
    and produce a single embedding vector per position.
    """

    @abstractmethod
    def forward(
        self,
        path_embeds: Tensor,  # (batch, seq_len, max_depth, d_model)
        path_lengths: Tensor,  # (batch, seq_len)
    ) -> Tensor:  # (batch, seq_len, d_model)
        """Pool path element embeddings into position embeddings."""
        ...


def make_depth_mask(path_lengths: Tensor, max_depth: int) -> Tensor:
    """Create a mask for valid path elements.

    Args:
        path_lengths: (batch, seq_len) - depth at each position
        max_depth: Maximum depth dimension

    Returns:
        mask: (batch, seq_len, max_depth) - True where path element is valid
    """
    batch_size, seq_len = path_lengths.shape
    depth_indices = torch.arange(max_depth, device=path_lengths.device)
    # (batch, seq_len, max_depth)
    mask = depth_indices.unsqueeze(0).unsqueeze(0) < path_lengths.unsqueeze(-1)
    return mask


class SumPooling(PathPooling):
    """Simple sum pooling - commutative baseline.

    This matches the original ORIGAMI paper's approach.
    Fast and parallelizable, but loses path order information.
    """

    def forward(
        self,
        path_embeds: Tensor,  # (batch, seq_len, max_depth, d_model)
        path_lengths: Tensor,  # (batch, seq_len)
    ) -> Tensor:
        max_depth = path_embeds.size(2)
        mask = make_depth_mask(path_lengths, max_depth)
        # Expand mask for broadcasting: (batch, seq_len, max_depth, 1)
        mask = mask.unsqueeze(-1)
        # Sum valid embeddings
        return (path_embeds * mask).sum(dim=2)


class WeightedSumPooling(PathPooling):
    """Weighted sum with learned or fixed depth weights.

    Provides partial order-awareness by weighting deeper elements differently.
    Still parallelizable like sum pooling.
    """

    def __init__(self, max_depth: int, learnable: bool = True):
        """Initialize weighted sum pooling.

        Args:
            max_depth: Maximum path depth
            learnable: If True, learn weights. If False, use exponential decay.
        """
        super().__init__()
        self.max_depth = max_depth

        if learnable:
            self.weights = nn.Parameter(torch.ones(max_depth))
        else:
            # Exponential decay: deeper elements have less weight
            decay = 0.9
            weights = torch.tensor([decay**d for d in range(max_depth)])
            self.register_buffer("weights", weights)

    def forward(
        self,
        path_embeds: Tensor,  # (batch, seq_len, max_depth, d_model)
        path_lengths: Tensor,  # (batch, seq_len)
    ) -> Tensor:
        max_depth = path_embeds.size(2)
        mask = make_depth_mask(path_lengths, max_depth)

        # Apply depth weights: (max_depth,) -> (1, 1, max_depth, 1)
        weights = self.weights[:max_depth].view(1, 1, -1, 1)
        weighted_embeds = path_embeds * weights

        # Mask and sum
        mask = mask.unsqueeze(-1)
        return (weighted_embeds * mask).sum(dim=2)


class RotaryPooling(PathPooling):
    """Apply rotation based on depth before summing.

    Applies rotary position embeddings based on depth, then sums.
    This provides order-awareness while remaining parallelizable.
    Similar to RoPE in transformers, but applied to path depth.
    """

    def __init__(self, d_model: int, max_depth: int, theta_base: float = 10000.0):
        """Initialize rotary pooling.

        Args:
            d_model: Model dimension (must be even)
            max_depth: Maximum path depth
            theta_base: Base for frequency computation
        """
        super().__init__()
        assert d_model % 2 == 0, "d_model must be even for rotary embeddings"

        self.d_model = d_model
        self.max_depth = max_depth

        # Compute rotation frequencies
        inv_freq = 1.0 / (theta_base ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer("inv_freq", inv_freq)

        # Precompute sin/cos for each depth
        positions = torch.arange(max_depth).float()
        # (max_depth, d_model/2)
        freqs = torch.outer(positions, inv_freq)
        # (max_depth, d_model/2)
        self.register_buffer("cos_cached", freqs.cos())
        self.register_buffer("sin_cached", freqs.sin())

    def _rotate(self, x: Tensor, depth_indices: Tensor) -> Tensor:
        """Apply rotary embedding based on depth.

        Args:
            x: (batch, seq_len, max_depth, d_model)
            depth_indices: (max_depth,) - just 0, 1, 2, ...

        Returns:
            Rotated embeddings with same shape
        """
        # Split into even and odd dimensions
        x1 = x[..., 0::2]  # (batch, seq_len, max_depth, d_model/2)
        x2 = x[..., 1::2]

        # Get cos/sin for each depth: (max_depth, d_model/2)
        cos = self.cos_cached[: x.size(2)]
        sin = self.sin_cached[: x.size(2)]

        # Expand for broadcasting: (1, 1, max_depth, d_model/2)
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

        # Apply rotation
        x1_rot = x1 * cos - x2 * sin
        x2_rot = x1 * sin + x2 * cos

        # Interleave back
        result = torch.empty_like(x)
        result[..., 0::2] = x1_rot
        result[..., 1::2] = x2_rot
        return result

    def forward(
        self,
        path_embeds: Tensor,  # (batch, seq_len, max_depth, d_model)
        path_lengths: Tensor,  # (batch, seq_len)
    ) -> Tensor:
        max_depth = path_embeds.size(2)
        mask = make_depth_mask(path_lengths, max_depth)

        # Apply depth-based rotation
        depth_indices = torch.arange(max_depth, device=path_embeds.device)
        rotated = self._rotate(path_embeds, depth_indices)

        # Mask and sum
        mask = mask.unsqueeze(-1)
        return (rotated * mask).sum(dim=2)


class GRUPooling(PathPooling):
    """Process path elements sequentially with GRU.

    Fully non-commutative: path order matters completely.
    Cannot be parallelized across depth dimension.
    Most expressive but slowest.
    """

    def __init__(self, d_model: int, num_layers: int = 1):
        """Initialize GRU pooling.

        Args:
            d_model: Model dimension
            num_layers: Number of GRU layers
        """
        super().__init__()
        self.gru = nn.GRU(
            d_model,
            d_model,
            num_layers=num_layers,
            batch_first=True,
        )

    def forward(
        self,
        path_embeds: Tensor,  # (batch, seq_len, max_depth, d_model)
        path_lengths: Tensor,  # (batch, seq_len)
    ) -> Tensor:
        batch_size, seq_len, max_depth, d_model = path_embeds.shape

        # Reshape to process all positions together
        # (batch * seq_len, max_depth, d_model)
        flat_embeds = path_embeds.view(batch_size * seq_len, max_depth, d_model)
        flat_lengths = path_lengths.view(batch_size * seq_len)

        # Clamp lengths to at least 1 for pack_padded_sequence
        flat_lengths = flat_lengths.clamp(min=1)

        # Pack for efficient GRU processing
        packed = nn.utils.rnn.pack_padded_sequence(
            flat_embeds,
            flat_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        # Run GRU
        _, hidden = self.gru(packed)
        # hidden: (num_layers, batch * seq_len, d_model)

        # Take final layer's hidden state
        output = hidden[-1]  # (batch * seq_len, d_model)

        # Reshape back
        return output.view(batch_size, seq_len, d_model)


class TransformerPooling(PathPooling):
    """Self-attention over path elements with depth positional encoding.

    Applies self-attention to path elements, allowing complex interactions.
    Parallelizable and order-aware through depth positional encoding.
    """

    def __init__(
        self,
        d_model: int,
        max_depth: int,
        num_layers: int = 1,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        """Initialize transformer pooling.

        Args:
            d_model: Model dimension
            max_depth: Maximum path depth
            num_layers: Number of transformer encoder layers
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()

        self.depth_embedding = nn.Embedding(max_depth, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Learnable query for aggregation (like CLS token)
        self.aggregate_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

    def forward(
        self,
        path_embeds: Tensor,  # (batch, seq_len, max_depth, d_model)
        path_lengths: Tensor,  # (batch, seq_len)
    ) -> Tensor:
        batch_size, seq_len, max_depth, d_model = path_embeds.shape

        # Add depth positional encoding
        depth_indices = torch.arange(max_depth, device=path_embeds.device)
        depth_pos = self.depth_embedding(depth_indices)  # (max_depth, d_model)
        path_embeds = path_embeds + depth_pos.unsqueeze(0).unsqueeze(0)

        # Create attention mask (True = masked out)
        mask = ~make_depth_mask(path_lengths, max_depth)  # (batch, seq_len, max_depth)

        # Reshape for batch processing
        # (batch * seq_len, max_depth, d_model)
        flat_embeds = path_embeds.view(batch_size * seq_len, max_depth, d_model)
        # (batch * seq_len, max_depth)
        flat_mask = mask.view(batch_size * seq_len, max_depth)

        # Apply transformer encoder
        encoded = self.encoder(flat_embeds, src_key_padding_mask=flat_mask)

        # Aggregate using mean of valid positions
        valid_mask = ~flat_mask  # (batch * seq_len, max_depth)
        valid_mask_expanded = valid_mask.unsqueeze(-1)  # (batch * seq_len, max_depth, 1)

        # Avoid division by zero for empty paths
        valid_counts = valid_mask.sum(dim=1, keepdim=True).clamp(min=1)  # (batch * seq_len, 1)
        output = (encoded * valid_mask_expanded).sum(dim=1) / valid_counts

        # Reshape back
        return output.view(batch_size, seq_len, d_model)


# Registry for easy instantiation
POOLING_CLASSES: dict[str, type[PathPooling]] = {
    "sum": SumPooling,
    "weighted": WeightedSumPooling,
    "rotary": RotaryPooling,
    "gru": GRUPooling,
    "transformer": TransformerPooling,
}


def create_pooling(
    pooling_type: str,
    d_model: int,
    max_depth: int,
    **kwargs,
) -> PathPooling:
    """Factory function to create a pooling strategy.

    Args:
        pooling_type: One of "sum", "weighted", "rotary", "gru", "transformer"
        d_model: Model dimension
        max_depth: Maximum path depth
        **kwargs: Additional arguments for specific pooling types

    Returns:
        PathPooling instance
    """
    if pooling_type not in POOLING_CLASSES:
        raise ValueError(
            f"Unknown pooling type: {pooling_type}. Available: {list(POOLING_CLASSES.keys())}"
        )

    cls = POOLING_CLASSES[pooling_type]

    # Route arguments based on class requirements
    if pooling_type == "sum":
        return cls()
    elif pooling_type == "weighted":
        return cls(max_depth=max_depth, **kwargs)
    elif pooling_type == "rotary":
        return cls(d_model=d_model, max_depth=max_depth, **kwargs)
    elif pooling_type == "gru":
        return cls(d_model=d_model, **kwargs)
    elif pooling_type == "transformer":
        return cls(d_model=d_model, max_depth=max_depth, **kwargs)
    else:
        raise ValueError(f"Unhandled pooling type: {pooling_type}")
