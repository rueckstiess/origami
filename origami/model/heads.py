"""ORIGAMI output heads.

Provides output heads for discrete (next-token) and continuous (MoG) prediction.
MVP implements DiscreteHead only; ContinuousHead is Phase 6.
"""

import torch
import torch.nn as nn
from torch import Tensor

from origami.config import ModelConfig


class DiscreteHead(nn.Module):
    """Standard next-token prediction head.

    Projects hidden states to vocabulary logits for discrete token prediction.

    Attributes:
        proj: Linear projection to vocabulary size
    """

    def __init__(self, config: ModelConfig, vocab_size: int):
        """Initialize discrete head.

        Args:
            config: Model configuration
            vocab_size: Size of the vocabulary
        """
        super().__init__()

        self.proj = nn.Linear(config.d_model, vocab_size)

    def forward(self, hidden: Tensor) -> Tensor:
        """Compute vocabulary logits.

        Args:
            hidden: Hidden states of shape (batch, seq_len, d_model)

        Returns:
            Logits of shape (batch, seq_len, vocab_size)
        """
        return self.proj(hidden)


class ContinuousHead(nn.Module):
    """Mixture of Gaussians head for continuous values.

    Outputs mixture parameters for modeling continuous numeric values:
    - weights: Mixture component weights (softmax normalized)
    - means: Component means
    - log_vars: Component log-variances

    Implemented in Phase 6.
    """

    def __init__(self, config: ModelConfig):
        """Initialize continuous head.

        Args:
            config: Model configuration with num_mixture_components
        """
        super().__init__()

        self.n_components = config.num_mixture_components
        self.d_model = config.d_model

        # Project to mixture parameters: weights, means, log_vars
        # Output size: 3 * n_components (weights, means, log_vars)
        self.proj = nn.Linear(config.d_model, 3 * self.n_components)

        # Pre-compute log(2*pi) as a buffer for efficiency
        self.register_buffer("log_2pi", torch.log(torch.tensor(2 * torch.pi)))

    def forward(self, hidden: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Compute mixture of Gaussians parameters.

        Args:
            hidden: Hidden states of shape (batch, seq_len, d_model)

        Returns:
            Tuple of (weights, means, log_vars), each of shape
            (batch, seq_len, n_components)
        """
        # Project to parameters
        params = self.proj(hidden)  # (batch, seq_len, 3 * n_components)

        # Split into weights, means, log_vars
        weights, means, log_vars = torch.chunk(params, 3, dim=-1)

        # Normalize weights via softmax
        weights = torch.softmax(weights, dim=-1)

        return weights, means, log_vars

    def nll_loss(
        self,
        weights: Tensor,  # (batch, seq_len, n_components)
        means: Tensor,  # (batch, seq_len, n_components)
        log_vars: Tensor,  # (batch, seq_len, n_components)
        targets: Tensor,  # (batch, seq_len)
        mask: Tensor,  # (batch, seq_len)
        loss_weights: Tensor | None = None,  # (batch, seq_len) - per-token weights
    ) -> Tensor:
        """Compute negative log-likelihood under mixture of Gaussians.

        Args:
            weights: Mixture weights (softmax normalized)
            means: Component means
            log_vars: Component log-variances
            targets: Target continuous values
            mask: Boolean mask where True indicates NUM token positions
            loss_weights: Optional per-token loss weights for weighted averaging.
                If provided, applies weighted mean instead of simple mean.

        Returns:
            Scalar NLL loss averaged over valid positions
        """
        # Expand targets for broadcasting: (batch, seq_len, 1)
        targets = targets.unsqueeze(-1)

        # Compute log probability for each component
        # log N(x; mu, sigma^2) = -0.5 * (log(2*pi) + log_var + (x - mu)^2 / exp(log_var))
        var = torch.exp(log_vars)
        log_probs = -0.5 * (self.log_2pi + log_vars + (targets - means) ** 2 / var)

        # Mixture log probability: log sum_k w_k * N(x; mu_k, sigma_k^2)
        # = log sum_k exp(log w_k + log N(x; mu_k, sigma_k^2))
        log_weights = torch.log(weights + 1e-10)
        log_mixture = torch.logsumexp(log_weights + log_probs, dim=-1)

        # Mask and average (with optional per-token weighting)
        if mask.any():
            masked_nll = -log_mixture[mask]
            if loss_weights is not None:
                masked_loss_weights = loss_weights[mask]
                nll = (masked_nll * masked_loss_weights).sum() / masked_loss_weights.sum().clamp(
                    min=1e-8
                )
            else:
                nll = masked_nll.mean()
        else:
            # Return zero loss on same device as input
            nll = weights.new_zeros(())

        return nll

    def sample(
        self,
        weights: Tensor,  # (batch, seq_len, n_components)
        means: Tensor,  # (batch, seq_len, n_components)
        log_vars: Tensor,  # (batch, seq_len, n_components)
    ) -> Tensor:
        """Sample from the mixture distribution.

        Args:
            weights: Mixture weights
            means: Component means
            log_vars: Component log-variances

        Returns:
            Samples of shape (batch, seq_len)
        """
        batch_size, seq_len, n_components = weights.shape

        # Sample component indices
        indices = torch.multinomial(weights.view(-1, n_components), num_samples=1).view(
            batch_size, seq_len
        )

        # Gather means and stds for selected components
        indices_expanded = indices.unsqueeze(-1)
        selected_means = torch.gather(means, dim=-1, index=indices_expanded).squeeze(-1)
        selected_log_vars = torch.gather(log_vars, dim=-1, index=indices_expanded).squeeze(-1)
        selected_stds = torch.exp(0.5 * selected_log_vars)

        # Sample from Gaussian
        samples = selected_means + selected_stds * torch.randn_like(selected_means)

        return samples
