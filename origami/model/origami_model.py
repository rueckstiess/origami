"""ORIGAMI main model.

Complete ORIGAMI model for JSON classification/generation.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .backbone import create_backbone
from .config import OrigamiConfig
from .embeddings import OrigamiEmbeddings
from .heads import ContinuousHead, DiscreteHead

if TYPE_CHECKING:
    from origami.tokenizer.vocabulary import Vocabulary


@dataclass
class OrigamiOutput:
    """Output from ORIGAMI model forward pass.

    Attributes:
        loss: Combined loss (discrete + continuous if enabled). None if no labels.
        logits: Discrete token logits of shape (batch, seq_len, vocab_size)
        continuous_params: Tuple of (weights, means, log_vars) for MoG head,
            or None if continuous head disabled.
        hidden_states: Final hidden states of shape (batch, seq_len, d_model)
    """

    loss: Tensor | None
    logits: Tensor
    continuous_params: tuple[Tensor, Tensor, Tensor] | None
    hidden_states: Tensor


class OrigamiModel(nn.Module):
    """Complete ORIGAMI model for JSON classification/generation.

    Combines:
    - Token embeddings + KVPE position encoding
    - Transformer (or other) backbone
    - Discrete next-token prediction head
    - Optional continuous (MoG) head for numeric values

    The model processes tokenized JSON sequences with path information
    and produces next-token predictions.

    Attributes:
        config: Model configuration
        embeddings: Token + position embedding layer
        backbone: Sequence modeling backbone (Transformer/LSTM/Mamba)
        discrete_head: Next-token prediction head
        continuous_head: Optional MoG head for numeric values
    """

    def __init__(self, config: OrigamiConfig, vocab: "Vocabulary | None" = None):
        """Initialize ORIGAMI model.

        Args:
            config: Model configuration
            vocab: Vocabulary instance (required if use_grammar_constraints=True)
        """
        super().__init__()

        self.config = config

        # Embeddings (token + KVPE)
        self.embeddings = OrigamiEmbeddings(config)

        # Backbone
        self.backbone = create_backbone(config)

        # Discrete head (always present)
        self.discrete_head = DiscreteHead(config)

        # Continuous head (optional)
        self.continuous_head: ContinuousHead | None = None
        if config.use_continuous_head:
            self.continuous_head = ContinuousHead(config)

        # Grammar constraints PDA (optional)
        self._grammar_pda = None
        if config.use_grammar_constraints:
            if vocab is None:
                raise ValueError(
                    "vocab is required when use_grammar_constraints=True"
                )
            from origami.constraints.json_grammar import JSONGrammarPDA
            self._grammar_pda = JSONGrammarPDA(vocab, max_depth=config.max_depth)

    def forward(
        self,
        input_ids: Tensor,  # (batch, seq_len)
        path_types: Tensor,  # (batch, seq_len, max_depth)
        path_ids: Tensor,  # (batch, seq_len, max_depth)
        path_lengths: Tensor,  # (batch, seq_len)
        attention_mask: Tensor | None = None,  # (batch, seq_len)
        labels: Tensor | None = None,  # (batch, seq_len)
        numeric_values: Tensor | None = None,  # (batch, seq_len) - Phase 6
        numeric_mask: Tensor | None = None,  # (batch, seq_len) - Phase 6
    ) -> OrigamiOutput:
        """Forward pass through the model.

        Args:
            input_ids: Token IDs of shape (batch, seq_len)
            path_types: Path element types (0=pad, 1=key, 2=index)
                of shape (batch, seq_len, max_depth)
            path_ids: Path element IDs of shape (batch, seq_len, max_depth)
            path_lengths: Path depths of shape (batch, seq_len)
            attention_mask: Boolean mask where True = valid position.
                Shape (batch, seq_len). If None, all positions are valid.
            labels: Target token IDs for loss computation.
                Shape (batch, seq_len). If None, no loss computed.
            numeric_values: Scaled numeric values for continuous head (Phase 6)
            numeric_mask: Boolean mask for NUM token positions (Phase 6)

        Returns:
            OrigamiOutput with logits, optional loss, and hidden states
        """
        # 1. Embeddings
        hidden = self.embeddings(input_ids, path_types, path_ids, path_lengths)

        # 2. Backbone
        hidden = self.backbone(hidden, attention_mask)

        # 3. Discrete head
        logits = self.discrete_head(hidden)

        # 4. Apply grammar constraints (both training and inference)
        # By masking invalid tokens, the model focuses probability mass on valid tokens only.
        if self.config.use_grammar_constraints:
            logits = self._apply_grammar_mask(logits, input_ids, attention_mask)

        # 5. Continuous head (if enabled)
        continuous_params = None
        if self.continuous_head is not None:
            continuous_params = self.continuous_head(hidden)

        # 6. Compute loss if labels provided
        loss = None
        if labels is not None:
            loss = self._compute_loss(
                logits=logits,
                labels=labels,
                attention_mask=attention_mask,
                continuous_params=continuous_params,
                numeric_values=numeric_values,
                numeric_mask=numeric_mask,
            )

        return OrigamiOutput(
            loss=loss,
            logits=logits,
            continuous_params=continuous_params,
            hidden_states=hidden,
        )

    def _apply_grammar_mask(
        self,
        logits: Tensor,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        """Apply grammar constraints to logits.

        Uses the PDA to compute which tokens are grammatically valid at each
        position, then masks invalid tokens with -inf.

        Args:
            logits: Raw logits of shape (batch, seq_len, vocab_size)
            input_ids: Input token IDs for grammar state
            attention_mask: Optional attention mask

        Returns:
            Logits with invalid tokens masked to -inf
        """
        if self._grammar_pda is None:
            return logits

        # Compute valid token mask for each position
        valid_mask = self._grammar_pda.compute_valid_mask(input_ids, attention_mask)

        # Apply mask: set invalid tokens to -inf
        return self._grammar_pda.apply_constraints(logits, valid_mask)

    def _compute_loss(
        self,
        logits: Tensor,  # (batch, seq_len, vocab_size)
        labels: Tensor,  # (batch, seq_len)
        attention_mask: Tensor | None,
        continuous_params: tuple[Tensor, Tensor, Tensor] | None,
        numeric_values: Tensor | None,
        numeric_mask: Tensor | None,
    ) -> Tensor:
        """Compute combined discrete and continuous loss.

        For autoregressive training, we shift labels so position i predicts
        token at position i+1.

        Args:
            logits: Predicted logits
            labels: Target token IDs
            attention_mask: Mask for valid positions (True = valid, False = padding)
            continuous_params: MoG parameters if continuous head enabled
            numeric_values: Target numeric values for continuous loss
            numeric_mask: Mask for NUM token positions

        Returns:
            Scalar loss value
        """
        # Shift for autoregressive: predict next token
        # logits[:, :-1] predicts labels[:, 1:]
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        # Mask out padding positions by setting labels to -100
        if attention_mask is not None:
            # Shift attention_mask to match shifted labels
            shift_mask = attention_mask[:, 1:].contiguous()
            # Set padding positions to ignore_index
            shift_labels = shift_labels.masked_fill(~shift_mask, -100)

        # Flatten for cross-entropy
        vocab_size = shift_logits.size(-1)
        loss = F.cross_entropy(
            shift_logits.view(-1, vocab_size),
            shift_labels.view(-1),
            ignore_index=-100,  # Ignore padding
            reduction="mean",
        )

        # Add continuous loss if enabled (Phase 6)
        if continuous_params is not None and numeric_values is not None:
            weights, means, log_vars = continuous_params
            # Shift continuous params and values too
            shift_weights = weights[:, :-1]
            shift_means = means[:, :-1]
            shift_log_vars = log_vars[:, :-1]
            shift_numeric_values = numeric_values[:, 1:]
            shift_numeric_mask = numeric_mask[:, 1:] if numeric_mask is not None else None

            if shift_numeric_mask is not None and shift_numeric_mask.any():
                continuous_loss = self.continuous_head.nll_loss(
                    shift_weights,
                    shift_means,
                    shift_log_vars,
                    shift_numeric_values,
                    shift_numeric_mask,
                )
                loss = loss + continuous_loss

        return loss

    def get_num_parameters(self, trainable_only: bool = True) -> int:
        """Count model parameters.

        Args:
            trainable_only: If True, only count trainable parameters

        Returns:
            Number of parameters
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
