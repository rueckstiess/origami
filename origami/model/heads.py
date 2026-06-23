"""ORIGAMI output heads.

Provides output heads for discrete (next-token) and continuous (MoG) prediction.

Float positions use standard Gaussian MoG NLL. Integer positions (array lengths)
use discretized logistic mixture NLL (PixelCNN++ style) with boundary absorption.
Truncation bounds are applied during sampling via ``sample(lower, upper)``.
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

        # Optionally spread the mixture components for the normalized [0, 1]
        # (array-length) domain. The default init (means ~0, scale ~1) suits
        # standardized N(0,1) numerics but collapses on multi-modal [0, 1]
        # targets, so "unit" spreads and localizes the components there.
        if config.continuous_init == "unit":
            self._init_unit_mixture_biases()

        # Pre-compute log(2*pi) as a buffer for efficiency
        self.register_buffer("log_2pi", torch.log(torch.tensor(2 * torch.pi)))

    def _init_unit_mixture_biases(self) -> None:
        """Spread/localize mixture components across the normalized [0, 1] range.

        Means are spread across [0, 1] and the (logistic) scale is set to about
        the component spacing so each component starts separated and localized.
        With the default init (means ~0, scale ~1.0) components overlap across the
        whole range and collapse onto a single mode, leaking probability into the
        gaps of multi-modal targets (e.g. bimodal array lengths).
        """
        k = self.n_components
        if k <= 1:
            return
        with torch.no_grad():
            spacing = 1.0 / (k - 1)
            # proj output layout: [weights | means | log_vars], each of size k.
            self.proj.bias[k : 2 * k] = torch.linspace(0.0, 1.0, k)
            # log_var is read as a (logistic/Gaussian) log-scale; set scale ~ spacing.
            self.proj.bias[2 * k : 3 * k] = 2.0 * torch.log(torch.tensor(spacing))

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
        is_integer: Tensor | None = None,  # (batch, seq_len) - bool mask for integer positions
        discretization_step: Tensor | None = None,  # (batch, seq_len) - bin width
    ) -> Tensor:
        """Compute negative log-likelihood for continuous and integer positions.

        Float positions use standard Gaussian MoG NLL. Integer positions (where
        ``is_integer=True``) use discretized logistic mixture NLL, which computes
        proper bin probabilities via sigmoid CDF differences with boundary absorption.

        Args:
            weights: Mixture weights (softmax normalized)
            means: Component means
            log_vars: Component log-variances (reinterpreted as log-scales for logistic)
            targets: Target continuous values
            mask: Boolean mask where True indicates continuous token positions
            loss_weights: Optional per-token loss weights for weighted averaging
            is_integer: Boolean mask for integer-valued positions (e.g., array lengths)
            discretization_step: Bin width per position (e.g., 1/max_items)

        Returns:
            Scalar NLL loss averaged over valid positions
        """
        if not mask.any():
            return weights.new_zeros(())

        # Fast path: no integer positions → all Gaussian
        if is_integer is None or not is_integer.any():
            return self._nll_loss_standard(weights, means, log_vars, targets, mask, loss_weights)

        integer_mask = mask & is_integer
        float_mask = mask & ~is_integer

        # All integer → all discretized logistic
        if not float_mask.any():
            return self._nll_loss_discretized_logistic(
                weights,
                means,
                log_vars,
                targets,
                integer_mask,
                discretization_step,
                loss_weights,
            )

        # Mixed: compute both, combine as weighted average by position count
        gauss_loss = self._nll_loss_standard(
            weights,
            means,
            log_vars,
            targets,
            float_mask,
            loss_weights,
        )
        logistic_loss = self._nll_loss_discretized_logistic(
            weights,
            means,
            log_vars,
            targets,
            integer_mask,
            discretization_step,
            loss_weights,
        )

        if loss_weights is not None:
            n_float = loss_weights[float_mask].sum()
            n_int = loss_weights[integer_mask].sum()
        else:
            n_float = float_mask.sum().float()
            n_int = integer_mask.sum().float()
        total = (n_float + n_int).clamp(min=1e-8)
        return (n_float * gauss_loss + n_int * logistic_loss) / total

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

    def _nll_loss_discretized_logistic(
        self,
        weights: Tensor,
        means: Tensor,
        log_vars: Tensor,
        targets: Tensor,
        mask: Tensor,
        discretization_step: Tensor,
        loss_weights: Tensor | None,
    ) -> Tensor:
        """Discretized logistic mixture NLL for integer-valued positions.

        Computes bin probabilities from logistic CDF differences (PixelCNN++ style).
        Boundary bins at 0 and 1 absorb all out-of-range mass, so no Z normalization
        is needed — unlike truncated Gaussian NLL which was numerically unstable.

        Targets are normalized to [0, 1] (e.g., array_length / max_items).
        discretization_step = 1 / max_items gives the bin width.
        """
        targets_expanded = targets.unsqueeze(-1)  # (batch, seq, 1)
        half_step = discretization_step.unsqueeze(-1) / 2  # (batch, seq, 1)

        # Reinterpret log_vars as logistic scale: s = exp(0.5 * log_vars)
        scales = torch.exp(0.5 * log_vars)  # (batch, seq, K)

        # CDF arguments for bin edges
        plus = (targets_expanded + half_step - means) / scales
        minus = (targets_expanded - half_step - means) / scales

        # Boundary handling: absorb all out-of-range mass
        # Lower boundary (target near 0): sigmoid(minus) → 0
        at_lower = targets_expanded < half_step
        minus = minus.masked_fill(at_lower, -20.0)
        # Upper boundary (target near 1): sigmoid(plus) → 1
        at_upper = targets_expanded > 1.0 - half_step
        plus = plus.masked_fill(at_upper, 20.0)

        # Bin probability per component: CDF(upper_edge) - CDF(lower_edge)
        bin_prob = (torch.sigmoid(plus) - torch.sigmoid(minus)).clamp(min=1e-12)
        log_bin_prob = torch.log(bin_prob)  # (batch, seq, K)

        # Mixture log-probability
        log_weights = torch.log(weights + 1e-10)
        log_mixture = torch.logsumexp(log_weights + log_bin_prob, dim=-1)  # (batch, seq)

        masked_nll = -log_mixture[mask]
        if loss_weights is not None:
            masked_loss_weights = loss_weights[mask]
            return (masked_nll * masked_loss_weights).sum() / masked_loss_weights.sum().clamp(
                min=1e-8
            )
        return masked_nll.mean()

    def sample_integer(
        self,
        weights: Tensor,  # (batch, seq_len, n_components)
        means: Tensor,  # (batch, seq_len, n_components) - normalized [0, 1]
        log_vars: Tensor,  # (batch, seq_len, n_components) - logistic log-scale
        norm: Tensor,  # (batch, seq_len) - normalization divisor (grid scale)
        min_values: Tensor | None = None,  # (batch, seq_len) - min integer, default 0
        max_values: Tensor | None = None,  # (batch, seq_len) - max integer cap (optional)
    ) -> Tensor:
        """Sample integers from discretized logistic mixture.

        Computes bin probabilities for each integer k in [0, norm] using the
        logistic CDF with boundary absorption (same math as training loss), then
        samples from the categorical distribution.

        ``norm`` is the grid scale and MUST equal the normalization divisor used
        during training (``length / norm``); otherwise samples are mis-scaled.
        ``min_values``/``max_values`` are *separate* constraint caps (e.g. schema
        minItems/maxItems) that mask which integers are valid — they do not change
        the grid.

        Args:
            weights: Mixture weights (softmax normalized)
            means: Component means in normalized [0, 1] space
            log_vars: Component log-scales (reinterpreted as logistic scale)
            norm: Normalization divisor per position (training grid scale)
            min_values: Optional minimum integer per position.
            max_values: Optional maximum integer cap per position.

        Returns:
            Sampled integers of shape (batch, seq_len)
        """
        batch_size, seq_len, n_components = weights.shape
        device = weights.device

        max_n = int(norm.max().long().item())
        n_bins = max_n + 1

        # Compute normalized targets for each bin: k / norm
        # k_values: (n_bins,), norm_expanded: (batch, seq, 1)
        k_values = torch.arange(n_bins, device=device, dtype=weights.dtype)
        norm_expanded = norm.unsqueeze(-1)  # (batch, seq, 1)
        targets = k_values.view(1, 1, -1) / norm_expanded  # (batch, seq, n_bins)
        step = 1.0 / norm_expanded  # (batch, seq, 1)
        half_step = step / 2  # (batch, seq, 1)

        # Broadcast for (batch, seq, n_bins, n_components)
        t = targets.unsqueeze(-1)  # (batch, seq, n_bins, 1)
        mu = means.unsqueeze(2)  # (batch, seq, 1, K)
        scales = torch.exp(0.5 * log_vars).unsqueeze(2)  # (batch, seq, 1, K)
        hs = half_step.unsqueeze(-1)  # (batch, seq, 1, 1)

        # Logistic CDF bin edges
        plus = (t + hs - mu) / scales
        minus = (t - hs - mu) / scales

        # Boundary absorption (same as training loss)
        at_lower = t < hs
        minus = minus.masked_fill(at_lower, -20.0)
        at_upper = t > 1.0 - hs
        plus = plus.masked_fill(at_upper, 20.0)

        # Bin probability per component
        bin_prob = (torch.sigmoid(plus) - torch.sigmoid(minus)).clamp(min=1e-12)

        # Mixture probability: weighted sum over components
        w = weights.unsqueeze(2)  # (batch, seq, 1, K)
        mixture_prob = (w * bin_prob).sum(dim=-1)  # (batch, seq, n_bins)

        # Mask bins outside the grid (k > norm) and outside optional constraint caps
        k_expanded = k_values.long().view(1, 1, -1).expand(batch_size, seq_len, -1)
        valid_bins = k_expanded <= norm.unsqueeze(-1).long()
        if max_values is not None:
            valid_bins = valid_bins & (k_expanded <= max_values.unsqueeze(-1).long())
        if min_values is not None:
            valid_bins = valid_bins & (k_expanded >= min_values.unsqueeze(-1).long())
        mixture_prob = mixture_prob.masked_fill(~valid_bins, 0.0)

        # Normalize and sample from categorical
        mixture_prob = mixture_prob / mixture_prob.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        sampled = torch.multinomial(mixture_prob.view(-1, n_bins), num_samples=1).view(
            batch_size, seq_len
        )

        return sampled

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
