"""ORIGAMI main model.

Complete ORIGAMI model for JSON classification/generation.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from origami.config import ModelConfig
from origami.tokenizer.vocabulary import Vocabulary

from .backbones import create_backbone
from .embeddings import OrigamiEmbeddings
from .heads import ContinuousHead, DiscreteHead

if TYPE_CHECKING:
    from origami.model.backbones import KVCache
    from origami.tokenizer.json_tokenizer import JSONTokenizer


@dataclass
class OrigamiOutput:
    """Output from ORIGAMI model forward pass.

    Attributes:
        loss: Combined loss (discrete + continuous if enabled). None if no labels.
        logits: Discrete token logits of shape (batch, seq_len, vocab_size)
        continuous_params: Tuple of (weights, means, log_vars) for MoG head,
            or None if continuous head disabled.
        hidden_states: Final hidden states of shape (batch, seq_len, d_model)
        past_key_values: KV cache from CachedTransformerBackbone, or None.
    """

    loss: Tensor | None
    logits: Tensor
    continuous_params: tuple[Tensor, Tensor, Tensor] | None
    hidden_states: Tensor
    past_key_values: "KVCache | None" = None


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

    def __init__(self, config: ModelConfig, vocab: Vocabulary):
        """Initialize ORIGAMI model.

        Args:
            config: Model architecture configuration
            vocab: Vocabulary instance (required for embeddings and grammar)
        """
        super().__init__()

        self.config = config
        self.vocab = vocab
        vocab_size = len(vocab)

        # Embeddings (token + KVPE)
        self.embeddings = OrigamiEmbeddings(config, vocab_size)

        # Backbone
        self.backbone = create_backbone(config)

        # Discrete head (always present)
        self.discrete_head = DiscreteHead(config, vocab_size)

        # Continuous head (optional)
        self.continuous_head: ContinuousHead | None = None
        if config.use_continuous_head:
            self.continuous_head = ContinuousHead(config)

        # Grammar constraints PDA (set by trainer if constrain_grammar=True)
        self._grammar_pda = None

    def forward(
        self,
        input_ids: Tensor,  # (batch, seq_len)
        path_types: Tensor,  # (batch, seq_len, max_depth)
        path_ids: Tensor,  # (batch, seq_len, max_depth)
        path_lengths: Tensor,  # (batch, seq_len)
        attention_mask: Tensor | None = None,  # (batch, seq_len)
        labels: Tensor | None = None,  # (batch, seq_len)
        numeric_values: Tensor | None = None,  # (batch, seq_len)
        numeric_mask: Tensor | None = None,  # (batch, seq_len)
        grammar_mask: Tensor | None = None,  # (batch, seq_len, vocab_size) - explicit!
        loss_weights: Tensor | None = None,  # (batch, seq_len) - per-token loss weights
        past_key_values: "KVCache | None" = None,
        use_cache: bool = False,
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
            numeric_values: Scaled numeric values for continuous head
            numeric_mask: Boolean mask for NUM token positions
            grammar_mask: Boolean mask for valid tokens at each position.
                Shape (batch, seq_len, vocab_size). If provided, invalid tokens
                are masked to -inf. Computed by caller (Trainer or Generator).
            loss_weights: Per-token loss weights of shape (batch, seq_len).
                If provided, applies weighted cross-entropy loss. Should be
                pre-normalized so mean weight = 1.0 to maintain stable gradients.
            past_key_values: KV cache from previous forward pass (for generation).
            use_cache: Whether to compute and return KV cache (for generation).

        Returns:
            OrigamiOutput with logits, optional loss, hidden states, and optionally cache
        """
        # 1. Embeddings (with numeric values for continuous head)
        hidden = self.embeddings(input_ids, path_types, path_ids, path_lengths, numeric_values)

        # 2. Backbone (with optional KV caching)
        new_cache = None
        if (
            use_cache
            and hasattr(self.backbone, "forward")
            and "past_key_values" in self.backbone.forward.__code__.co_varnames
        ):
            # CachedTransformerBackbone supports caching
            result = self.backbone(
                hidden, attention_mask, past_key_values=past_key_values, use_cache=True
            )
            hidden, new_cache = result
        else:
            # Standard backbone without caching
            hidden = self.backbone(hidden, attention_mask)

        # 3. Discrete head
        logits = self.discrete_head(hidden)

        # 4. Apply grammar mask if provided (explicit interface)
        # Caller is responsible for computing the mask (Trainer or Generator)
        if grammar_mask is not None:
            logits = logits.masked_fill(~grammar_mask, float("-inf"))

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
                loss_weights=loss_weights,
                continuous_params=continuous_params,
                numeric_values=numeric_values,
                numeric_mask=numeric_mask,
            )

        return OrigamiOutput(
            loss=loss,
            logits=logits,
            continuous_params=continuous_params,
            hidden_states=hidden,
            past_key_values=new_cache,
        )

    def compute_grammar_mask(
        self,
        input_ids: Tensor,
    ) -> Tensor | None:
        """Compute grammar mask for training.

        Computes which tokens are grammatically valid at each position.
        Called by Trainer to compute the mask before forward pass.

        Note: Grammar computation runs on CPU for performance, as the
        PDA state updates involve many small tensor operations that cause
        excessive synchronization overhead on MPS/CUDA.

        Args:
            input_ids: Input token IDs of shape (batch, seq_len)

        Returns:
            Boolean mask of shape (batch, seq_len, vocab_size) where True
            indicates valid tokens. Returns None if grammar constraints
            are not enabled.
        """
        if self._grammar_pda is None:
            return None

        original_device = input_ids.device

        # Run grammar computation on CPU for performance (avoids MPS/CUDA sync overhead)
        input_ids_cpu = input_ids.cpu()
        valid_mask_cpu = self._grammar_pda.compute_valid_mask(input_ids_cpu)

        # Clone and make contiguous before device transfer to avoid shared storage issues
        return valid_mask_cpu.clone().contiguous().to(original_device)

    def _compute_loss(
        self,
        logits: Tensor,  # (batch, seq_len, vocab_size)
        labels: Tensor,  # (batch, seq_len)
        attention_mask: Tensor | None,
        loss_weights: Tensor | None,  # (batch, seq_len)
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
            loss_weights: Per-token loss weights (already normalized)
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

        if loss_weights is not None:
            # Weighted cross-entropy: use reduction="none" and apply weights
            per_token_loss = F.cross_entropy(
                shift_logits.view(-1, vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
                reduction="none",
            )

            # Shift loss_weights to match shifted labels
            shift_loss_weights = loss_weights[:, 1:].contiguous()

            # Zero out weights for padding positions (where labels are -100)
            valid_mask = (shift_labels != -100).float()
            shift_loss_weights = shift_loss_weights * valid_mask

            # Flatten weights to match per_token_loss
            flat_weights = shift_loss_weights.view(-1)

            # Compute weighted mean (weights should be pre-normalized)
            weight_sum = flat_weights.sum().clamp(min=1e-8)
            loss = (per_token_loss * flat_weights).sum() / weight_sum
        else:
            loss = F.cross_entropy(
                shift_logits.view(-1, vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,  # Ignore padding
                reduction="mean",
            )

        # Add continuous loss if enabled
        if continuous_params is not None and numeric_values is not None:
            weights, means, log_vars = continuous_params
            # Shift continuous params and values too
            shift_weights = weights[:, :-1]
            shift_means = means[:, :-1]
            shift_log_vars = log_vars[:, :-1]
            shift_numeric_values = numeric_values[:, 1:]
            shift_numeric_mask = numeric_mask[:, 1:] if numeric_mask is not None else None

            if shift_numeric_mask is not None and shift_numeric_mask.any():
                # Pass loss_weights to continuous head for consistent weighting
                shift_loss_weights = (
                    loss_weights[:, 1:].contiguous() if loss_weights is not None else None
                )
                continuous_loss = self.continuous_head.nll_loss(
                    shift_weights,
                    shift_means,
                    shift_log_vars,
                    shift_numeric_values,
                    shift_numeric_mask,
                    loss_weights=shift_loss_weights,
                )

                # Calculate loss weight
                if self.config.continuous_loss_weight < 0:
                    # Auto-calculate: proportion of NUM tokens in the batch
                    num_tokens = shift_numeric_mask.sum().float()
                    total_tokens = shift_numeric_mask.numel()
                    weight = max(num_tokens / total_tokens, 0.001)
                else:
                    weight = self.config.continuous_loss_weight

                loss = loss + weight * continuous_loss

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

    def save(
        self,
        path: str | Path,
        tokenizer: "JSONTokenizer | None" = None,
    ) -> None:
        """Save model to a checkpoint file.

        Saves model config, state dict, and optionally the tokenizer.
        The saved checkpoint can be loaded with `OrigamiModel.load()`.

        Args:
            path: Path to save checkpoint to (typically .pt file)
            tokenizer: Optional tokenizer to save with the model
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model_config": asdict(self.config),
            "model_state_dict": self.state_dict(),
        }

        if tokenizer is not None:
            # Serialize tokenizer state
            checkpoint["tokenizer_state"] = {
                "vocab": tokenizer.vocab.to_dict(),
                "max_depth": tokenizer.max_depth,
                "max_array_index": tokenizer.max_array_index,
            }

        torch.save(checkpoint, path)

    @classmethod
    def load(
        cls,
        path: str | Path,
        device: torch.device | str | None = None,
    ) -> tuple["OrigamiModel", "JSONTokenizer | None"]:
        """Load model from a checkpoint file.

        Creates a new model instance with the saved configuration and
        loads the weights. Also loads the tokenizer if it was saved.

        Args:
            path: Path to checkpoint file
            device: Device to load model to. If None, uses CPU.

        Returns:
            Tuple of (model, tokenizer). Tokenizer is None if not saved.

        Example:
            ```python
            model, tokenizer = OrigamiModel.load("checkpoint.pt")
            generator = OrigamiGenerator(model, tokenizer)
            ```
        """
        from origami.tokenizer import JSONTokenizer

        path = Path(path)
        device = device or "cpu"

        checkpoint = torch.load(path, map_location=device, weights_only=False)

        # Reconstruct config
        config = ModelConfig(**checkpoint["model_config"])

        # Reconstruct tokenizer (required - vocab needed for model)
        if "tokenizer_state" not in checkpoint:
            raise ValueError(
                "Checkpoint does not contain tokenizer state. "
                "Model cannot be loaded without vocabulary."
            )

        tokenizer_state = checkpoint["tokenizer_state"]
        vocab = Vocabulary.from_dict(tokenizer_state["vocab"])
        tokenizer = JSONTokenizer()
        tokenizer.vocab = vocab
        tokenizer.max_depth = tokenizer_state["max_depth"]
        tokenizer.max_array_index = tokenizer_state["max_array_index"]
        tokenizer._fitted = True

        # Create model with vocab
        model = cls(config, vocab)

        # Load weights
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)

        return model, tokenizer
