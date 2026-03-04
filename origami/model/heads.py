"""ORIGAMI output heads.

Provides output heads for discrete (next-token) and continuous (MoG) prediction.
Supports truncated MoG with schema-derived bounds and discretized NLL for integers.
"""

import math

import torch
import torch.nn as nn
from torch import Tensor

from origami.config import ModelConfig

# Precomputed constant
_LOG_2PI = math.log(2 * math.pi)


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

    @staticmethod
    def _standard_normal_cdf(x: Tensor) -> Tensor:
        """Standard normal CDF using the error function."""
        return 0.5 * (1 + torch.erf(x / math.sqrt(2)))

    def nll_loss(
        self,
        weights: Tensor,  # (batch, seq_len, n_components)
        means: Tensor,  # (batch, seq_len, n_components)
        log_vars: Tensor,  # (batch, seq_len, n_components)
        targets: Tensor,  # (batch, seq_len)
        mask: Tensor,  # (batch, seq_len)
        loss_weights: Tensor | None = None,  # (batch, seq_len) - per-token weights
        lower: Tensor | None = None,  # (batch, seq_len) - truncation lower bounds
        upper: Tensor | None = None,  # (batch, seq_len) - truncation upper bounds
        is_integer: Tensor | None = None,  # (batch, seq_len) - bool mask for discretized NLL
    ) -> Tensor:
        """Compute negative log-likelihood under (optionally truncated) mixture of Gaussians.

        Supports three modes depending on the arguments:
        1. Standard MoG NLL (no bounds, no is_integer) — identical to the original.
        2. Truncated MoG NLL (bounds provided) — normalizes each component by
           Z_k = Phi((upper - mu_k) / sigma_k) - Phi((lower - mu_k) / sigma_k).
        3. Discretized NLL (is_integer provided) — for integer-valued targets,
           computes P(k) = CDF_trunc(k+0.5) - CDF_trunc(k-0.5) instead of density.

        Args:
            weights: Mixture weights (softmax normalized)
            means: Component means
            log_vars: Component log-variances
            targets: Target continuous values
            mask: Boolean mask where True indicates continuous token positions
            loss_weights: Optional per-token loss weights for weighted averaging
            lower: Per-position lower bounds for truncation (default: -inf)
            upper: Per-position upper bounds for truncation (default: +inf)
            is_integer: Boolean mask where True uses discretized NLL

        Returns:
            Scalar NLL loss averaged over valid positions
        """
        if not mask.any():
            return weights.new_zeros(())

        has_bounds = lower is not None or upper is not None
        has_integers = is_integer is not None and is_integer.any()

        # Fast path: no truncation, no integers — original logic
        if not has_bounds and not has_integers:
            return self._nll_loss_standard(weights, means, log_vars, targets, mask, loss_weights)

        # Compute log-mixture probability for each position
        log_mixture = self._log_mixture_prob(
            weights, means, log_vars, targets, lower, upper, is_integer
        )

        # Mask and average
        masked_nll = -log_mixture[mask]
        if loss_weights is not None:
            masked_loss_weights = loss_weights[mask]
            return (masked_nll * masked_loss_weights).sum() / masked_loss_weights.sum().clamp(
                min=1e-8
            )
        return masked_nll.mean()

    def _nll_loss_standard(
        self,
        weights: Tensor,
        means: Tensor,
        log_vars: Tensor,
        targets: Tensor,
        mask: Tensor,
        loss_weights: Tensor | None,
    ) -> Tensor:
        """Original unbounded continuous MoG NLL (fast path)."""
        targets_expanded = targets.unsqueeze(-1)
        var = torch.exp(log_vars)
        log_probs = -0.5 * (self.log_2pi + log_vars + (targets_expanded - means) ** 2 / var)
        log_weights = torch.log(weights + 1e-10)
        log_mixture = torch.logsumexp(log_weights + log_probs, dim=-1)

        masked_nll = -log_mixture[mask]
        if loss_weights is not None:
            masked_loss_weights = loss_weights[mask]
            return (masked_nll * masked_loss_weights).sum() / masked_loss_weights.sum().clamp(
                min=1e-8
            )
        return masked_nll.mean()

    def _log_mixture_prob(
        self,
        weights: Tensor,  # (batch, seq_len, K)
        means: Tensor,  # (batch, seq_len, K)
        log_vars: Tensor,  # (batch, seq_len, K)
        targets: Tensor,  # (batch, seq_len)
        lower: Tensor | None,  # (batch, seq_len) or None
        upper: Tensor | None,  # (batch, seq_len) or None
        is_integer: Tensor | None,  # (batch, seq_len) or None
    ) -> Tensor:
        """Compute log mixture probability with truncation and optional discretization.

        Returns:
            (batch, seq_len) log-probabilities
        """
        stds = torch.exp(0.5 * log_vars)  # (batch, seq_len, K)
        log_weights = torch.log(weights + 1e-10)

        # Prepare bounds — expand to (batch, seq_len, 1) for component broadcasting.
        # Clamp ±inf to finite values to avoid NaN gradients: the CDF computation
        # Phi((bound - mean) / std) gives correct forward values at ±inf, but the
        # gradient involves PDF(inf) * (-inf / std²) = 0 * inf = NaN.
        # ±100 is effectively unbounded (normal PDF is exactly 0 for |x| > ~27).
        _BOUND_CLAMP = 100.0
        lo = lower.clamp(min=-_BOUND_CLAMP).unsqueeze(-1) if lower is not None else means.new_full((1,), -_BOUND_CLAMP)
        hi = upper.clamp(max=_BOUND_CLAMP).unsqueeze(-1) if upper is not None else means.new_full((1,), _BOUND_CLAMP)

        # Normalization constant: Z_k = Phi((hi - mu_k) / sigma_k) - Phi((lo - mu_k) / sigma_k)
        z_hi = self._standard_normal_cdf((hi - means) / stds)
        z_lo = self._standard_normal_cdf((lo - means) / stds)
        log_Z = torch.log((z_hi - z_lo).clamp(min=1e-12))

        # Start with continuous truncated NLL for all positions:
        # log f_k(x) = -0.5 * ((x-mu)/sigma)^2 - log(sigma) - 0.5*log(2pi) - log(Z_k)
        targets_expanded = targets.unsqueeze(-1)  # (batch, seq_len, 1)
        z_scores = (targets_expanded - means) / stds
        log_component = -0.5 * z_scores**2 - torch.log(stds) - 0.5 * _LOG_2PI - log_Z
        log_mixture = torch.logsumexp(log_weights + log_component, dim=-1)

        # For integer positions, replace with discretized probability:
        # P_k(x) = (CDF_trunc(x + 0.5) - CDF_trunc(x - 0.5))
        # where CDF_trunc(t) = (Phi((t-mu)/sigma) - Phi((lo-mu)/sigma)) / Z_k
        if is_integer is not None and is_integer.any():
            x_hi = (targets_expanded + 0.5).clamp(max=hi)
            x_lo = (targets_expanded - 0.5).clamp(min=lo)
            cdf_hi = self._standard_normal_cdf((x_hi - means) / stds)
            cdf_lo = self._standard_normal_cdf((x_lo - means) / stds)
            # P_k(x) = (cdf_hi - cdf_lo) / Z_k  (already unnormalized CDF differences)
            Z = (z_hi - z_lo).clamp(min=1e-12)
            p_component = ((cdf_hi - cdf_lo) / Z).clamp(min=1e-20)
            # Mixture: P(x) = sum_k w_k * P_k(x)
            p_discrete = (weights * p_component).sum(dim=-1)
            log_mixture_discrete = torch.log(p_discrete.clamp(min=1e-20))
            # Replace only at integer positions
            log_mixture = torch.where(is_integer, log_mixture_discrete, log_mixture)

        return log_mixture

    def sample(
        self,
        weights: Tensor,  # (batch, seq_len, n_components)
        means: Tensor,  # (batch, seq_len, n_components)
        log_vars: Tensor,  # (batch, seq_len, n_components)
        lower: Tensor | None = None,  # (batch, seq_len) or None
        upper: Tensor | None = None,  # (batch, seq_len) or None
    ) -> Tensor:
        """Sample from the mixture distribution, optionally truncated to [lower, upper].

        When bounds are provided, uses inverse CDF sampling from the truncated
        distribution: reweight components by their mass within [lower, upper],
        then sample via u ~ Uniform(CDF(lower), CDF(upper)), x = ICDF(u).

        Args:
            weights: Mixture weights
            means: Component means
            log_vars: Component log-variances
            lower: Per-position lower bounds, or None for unbounded
            upper: Per-position upper bounds, or None for unbounded

        Returns:
            Samples of shape (batch, seq_len)
        """
        batch_size, seq_len, n_components = weights.shape

        if lower is None and upper is None:
            # Unconstrained path — original sampling logic
            indices = torch.multinomial(weights.view(-1, n_components), num_samples=1).view(
                batch_size, seq_len
            )
            indices_expanded = indices.unsqueeze(-1)
            selected_means = torch.gather(means, dim=-1, index=indices_expanded).squeeze(-1)
            selected_log_vars = torch.gather(log_vars, dim=-1, index=indices_expanded).squeeze(-1)
            selected_stds = torch.exp(0.5 * selected_log_vars)
            return selected_means + selected_stds * torch.randn_like(selected_means)

        # Truncated sampling via inverse CDF
        stds = torch.exp(0.5 * log_vars)  # (batch, seq, n_components)
        dist = torch.distributions.Normal(means, stds)

        # Expand bounds to (batch, seq, 1) for broadcasting with components
        lo = (
            lower.unsqueeze(-1) if lower is not None else means.new_full(means.shape, float("-inf"))
        )
        hi = upper.unsqueeze(-1) if upper is not None else means.new_full(means.shape, float("inf"))

        # CDF at bounds per component
        cdf_lo = dist.cdf(lo)  # (batch, seq, n_components)
        cdf_hi = dist.cdf(hi)

        # Reweight components by mass within [lower, upper]
        mass = (cdf_hi - cdf_lo).clamp(min=1e-12)
        reweighted = weights * mass
        reweighted = reweighted / reweighted.sum(dim=-1, keepdim=True).clamp(min=1e-12)

        # Sample component index from reweighted distribution
        indices = torch.multinomial(reweighted.view(-1, n_components), num_samples=1).view(
            batch_size, seq_len
        )
        idx = indices.unsqueeze(-1)

        # Gather selected component parameters and CDF bounds
        sel_cdf_lo = torch.gather(cdf_lo, -1, idx).squeeze(-1)
        sel_cdf_hi = torch.gather(cdf_hi, -1, idx).squeeze(-1)
        sel_means = torch.gather(means, -1, idx).squeeze(-1)
        sel_stds = torch.gather(stds, -1, idx).squeeze(-1)

        # Inverse CDF: u ~ Uniform(cdf_lo, cdf_hi), x = icdf(u)
        u = sel_cdf_lo + (sel_cdf_hi - sel_cdf_lo) * torch.rand_like(sel_means)
        u = u.clamp(1e-6, 1 - 1e-6)  # Avoid numerical issues at tails

        # x = mean + std * Phi_inv(u) where Phi_inv is standard normal ICDF
        standard_normal = torch.distributions.Normal(
            torch.zeros_like(sel_means), torch.ones_like(sel_stds)
        )
        samples = sel_means + sel_stds * standard_normal.icdf(u)

        return samples
