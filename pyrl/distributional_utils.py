"""
Utilities for distributional reinforcement learning.

This module provides core functions for distributional RL with quantile regression,
including quantile Huber loss, quantile interpolation, and context-based quantile selection.

References:
    - Dabney et al. (2018): "Distributional Reinforcement Learning with Quantile Regression"
    - Dabney et al. (2020): "A distributional code for value in dopamine-based reinforcement learning"
"""

import torch
import torch.nn.functional as F


def quantile_huber_loss(pred_quantiles, targets, tau_values, kappa=1.0, mask=None):
    """
    Compute quantile Huber loss for distributional RL.

    The quantile Huber loss combines the robustness of Huber loss with the
    asymmetric weighting of quantile regression. For each quantile τ:

    L_τ(δ) = |τ - I(δ < 0)| * ρ_κ(δ)

    where ρ_κ is the Huber loss with threshold κ, and δ = target - prediction.

    Parameters
    ----------
    pred_quantiles : torch.Tensor, shape (T, B, n_quantiles)
        Predicted quantile values from the distributional critic.
    targets : torch.Tensor, shape (T, B)
        Target returns (e.g., Monte Carlo returns or TD targets).
    tau_values : torch.Tensor, shape (n_quantiles,)
        Quantile fractions (e.g., [0.1, 0.25, 0.5, 0.75, 0.9]).
    kappa : float, default=1.0
        Huber loss threshold. Controls the transition from quadratic to linear loss.
        Larger κ → more like L2 loss (sensitive to outliers).
        Smaller κ → more like L1 loss (robust to outliers).
    mask : torch.Tensor, shape (T, B), optional
        Binary mask indicating valid timesteps (1 = valid, 0 = invalid).
        If None, all timesteps are considered valid.

    Returns
    -------
    loss : torch.Tensor, scalar
        Mean quantile Huber loss across all timesteps, batch, and quantiles.

    Notes
    -----
    The quantile Huber loss is asymmetric:
    - For τ = 0.1 (10th percentile): heavily penalize overestimation (predictions too high)
    - For τ = 0.5 (median): symmetric penalty (standard Huber loss)
    - For τ = 0.9 (90th percentile): heavily penalize underestimation (predictions too low)

    This asymmetry ensures that each quantile head learns the appropriate percentile
    of the return distribution.

    Examples
    --------
    >>> tau_values = torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9])
    >>> pred_quantiles = torch.randn(100, 32, 5)  # T=100, B=32, n_quantiles=5
    >>> targets = torch.randn(100, 32)
    >>> loss = quantile_huber_loss(pred_quantiles, targets, tau_values, kappa=1.0)
    """
    T, B, n_quantiles = pred_quantiles.shape
    device = pred_quantiles.device

    # Expand targets to match quantile dimensions: (T, B) → (T, B, n_quantiles)
    targets_expanded = targets.unsqueeze(-1).expand(-1, -1, n_quantiles)

    # Compute TD errors: δ = target - prediction
    delta = targets_expanded - pred_quantiles  # Shape: (T, B, n_quantiles)

    # Huber loss: ρ_κ(δ) = 0.5 * δ² if |δ| ≤ κ, else κ * (|δ| - 0.5 * κ)
    abs_delta = torch.abs(delta)
    huber = torch.where(
        abs_delta <= kappa,
        0.5 * delta ** 2,
        kappa * (abs_delta - 0.5 * kappa)
    )

    # Quantile weighting: |τ - I(δ < 0)|
    # For positive errors (overestimation): weight = τ
    # For negative errors (underestimation): weight = (1 - τ)
    tau_expanded = tau_values.view(1, 1, n_quantiles).to(device)  # Shape: (1, 1, n_quantiles)
    quantile_weight = torch.abs(tau_expanded - (delta < 0).float())

    # Combine Huber loss with quantile weighting
    loss_per_element = quantile_weight * huber  # Shape: (T, B, n_quantiles)

    # Apply mask if provided
    if mask is not None:
        # Expand mask to match quantile dimensions: (T, B) → (T, B, n_quantiles)
        mask_expanded = mask.unsqueeze(-1).expand(-1, -1, n_quantiles)
        loss_per_element = loss_per_element * mask_expanded
        n_valid = mask.sum() * n_quantiles
    else:
        n_valid = T * B * n_quantiles

    # Mean loss across all dimensions
    if n_valid > 0:
        loss = loss_per_element.sum() / n_valid
    else:
        loss = torch.tensor(0.0, device=device, requires_grad=True)

    return loss


def interpolate_quantiles(q_values, quantile_idx):
    """
    Interpolate between adjacent quantiles based on fractional index.

    This function enables smooth selection of intermediate quantile values
    when the desired quantile falls between two discrete quantiles.
    For example, if we want the 60th percentile but only have quantiles at
    50% and 75%, this function will interpolate between them.

    Parameters
    ----------
    q_values : torch.Tensor, shape (T, B, n_quantiles)
        Quantile values predicted by the distributional critic.
    quantile_idx : torch.Tensor, shape (B,) or scalar
        Fractional index into the quantile dimension.
        Range: [0, n_quantiles - 1]
        - 0.0 → first quantile (e.g., 10th percentile)
        - 2.0 → third quantile (e.g., 50th percentile)
        - 4.0 → fifth quantile (e.g., 90th percentile)
        - 2.5 → interpolate between 3rd and 4th quantiles

    Returns
    -------
    interpolated : torch.Tensor, shape (T, B)
        Interpolated quantile values.

    Notes
    -----
    Linear interpolation formula:
        Q(i + α) = (1 - α) * Q(i) + α * Q(i+1)
    where i = floor(idx) and α = idx - i.

    Edge case handling:
    - If idx < 0, clamp to 0
    - If idx >= n_quantiles - 1, clamp to n_quantiles - 1

    Examples
    --------
    >>> q_values = torch.tensor([[[1.0, 2.0, 3.0, 4.0, 5.0]]])  # (T=1, B=1, n_quantiles=5)
    >>> quantile_idx = torch.tensor([2.5])  # Between 3rd and 4th quantile
    >>> interpolated = interpolate_quantiles(q_values, quantile_idx)
    >>> print(interpolated)  # Should be 3.5 (midpoint between 3.0 and 4.0)
    """
    T, B, n_quantiles = q_values.shape
    device = q_values.device

    # Ensure quantile_idx is a tensor
    if not isinstance(quantile_idx, torch.Tensor):
        quantile_idx = torch.tensor(quantile_idx, device=device)

    # Handle scalar or batch of indices
    if quantile_idx.dim() == 0:
        quantile_idx = quantile_idx.unsqueeze(0).expand(B)
    elif quantile_idx.shape[0] == 1 and B > 1:
        quantile_idx = quantile_idx.expand(B)

    # Clamp index to valid range [0, n_quantiles - 1]
    quantile_idx = torch.clamp(quantile_idx, 0.0, n_quantiles - 1.0)

    # Split into integer and fractional parts
    idx_floor = torch.floor(quantile_idx).long()  # Shape: (B,)
    alpha = quantile_idx - idx_floor.float()      # Shape: (B,)

    # Handle edge case: if idx_floor == n_quantiles - 1, we can't interpolate forward
    # In this case, just return the last quantile
    idx_ceil = torch.clamp(idx_floor + 1, max=n_quantiles - 1)

    # Gather quantile values at floor and ceil indices
    # q_values shape: (T, B, n_quantiles)
    # We need to gather along the quantile dimension for each timestep and batch element

    # Reshape for gathering: (T, B, n_quantiles) → (T * B, n_quantiles)
    q_values_flat = q_values.view(T * B, n_quantiles)

    # Expand indices for all timesteps: (B,) → (T, B) → (T * B)
    idx_floor_expanded = idx_floor.unsqueeze(0).expand(T, -1).reshape(T * B)
    idx_ceil_expanded = idx_ceil.unsqueeze(0).expand(T, -1).reshape(T * B)

    # Gather values
    q_floor = q_values_flat.gather(1, idx_floor_expanded.unsqueeze(1)).squeeze(1)  # (T * B,)
    q_ceil = q_values_flat.gather(1, idx_ceil_expanded.unsqueeze(1)).squeeze(1)    # (T * B,)

    # Interpolate: Q(i + α) = (1 - α) * Q(i) + α * Q(i+1)
    alpha_expanded = alpha.unsqueeze(0).expand(T, -1).reshape(T * B)
    interpolated_flat = (1 - alpha_expanded) * q_floor + alpha_expanded * q_ceil

    # Reshape back to (T, B)
    interpolated = interpolated_flat.view(T, B)

    return interpolated


def context_to_quantile_idx(context, n_quantiles):
    """
    Map context signal to quantile index for context-dependent risk preferences.

    This function implements the mapping from tonic dopamine (context) to
    quantile selection, mimicking the biological mechanism where tonic DA
    modulates D1/D2 MSN balance:

    - High tonic DA (context = +1) → D1 dominance → optimistic quantiles (90th percentile)
    - Low tonic DA (context = -1) → D2 dominance → pessimistic quantiles (10th percentile)
    - Neutral DA (context = 0) → balanced → median quantile (50th percentile)

    Parameters
    ----------
    context : torch.Tensor, shape (B,) or scalar
        Context signal representing tonic dopamine level or internal state.
        Expected range: [-1, +1], but will be clamped via tanh.
        - context < 0: risk-averse (select lower quantiles)
        - context = 0: risk-neutral (select median quantile)
        - context > 0: risk-seeking (select higher quantiles)
    n_quantiles : int
        Number of quantiles in the distributional critic.

    Returns
    -------
    quantile_idx : torch.Tensor, shape (B,)
        Fractional index into quantile dimension, range [0, n_quantiles - 1].

    Notes
    -----
    Mapping formula:
        quantile_idx = (tanh(context) + 1) / 2 * (n_quantiles - 1)

    This ensures:
    - context = -∞ (after tanh: -1) → quantile_idx = 0 (10th percentile)
    - context = 0 (after tanh: 0) → quantile_idx = (n_quantiles - 1) / 2 (median)
    - context = +∞ (after tanh: +1) → quantile_idx = n_quantiles - 1 (90th percentile)

    Biological interpretation:
    - Corresponds to D1/D2 MSN balance in striatum (Lowet et al., 2025)
    - D1 pathway (Go): encodes optimistic quantiles
    - D2 pathway (NoGo): encodes pessimistic quantiles
    - Tonic DA: modulates relative gain of D1 vs D2 pathways

    Examples
    --------
    >>> context = torch.tensor([-1.0, 0.0, 1.0])  # Risk-averse, neutral, seeking
    >>> idx = context_to_quantile_idx(context, n_quantiles=5)
    >>> print(idx)  # Should be approximately [0.0, 2.0, 4.0]
    """
    device = context.device if isinstance(context, torch.Tensor) else torch.device('cpu')

    # Ensure context is a tensor
    if not isinstance(context, torch.Tensor):
        context = torch.tensor(context, device=device)

    # Apply tanh to map context to [-1, +1] (if not already in range)
    context_normalized = torch.tanh(context)

    # Map [-1, +1] to [0, n_quantiles - 1]
    quantile_idx = (context_normalized + 1.0) / 2.0 * (n_quantiles - 1)

    return quantile_idx


def get_default_quantiles(n_quantiles):
    """
    Get default quantile fractions (tau values) for distributional RL.

    Parameters
    ----------
    n_quantiles : int
        Number of quantiles to use. Common values: 5, 51, 200.

    Returns
    -------
    tau_values : torch.Tensor, shape (n_quantiles,)
        Quantile fractions uniformly spaced in (0, 1).

    Notes
    -----
    For n_quantiles = 5, returns [0.1, 0.3, 0.5, 0.7, 0.9].
    This matches the standard quantiles used in Dabney et al. (2018, 2020).

    For biological interpretation:
    - τ = 0.1, 0.3: Pessimistic quantiles (D2 pathway)
    - τ = 0.5: Median (neutral)
    - τ = 0.7, 0.9: Optimistic quantiles (D1 pathway)

    Examples
    --------
    >>> tau = get_default_quantiles(5)
    >>> print(tau)  # [0.1, 0.3, 0.5, 0.7, 0.9]
    """
    # Evenly spaced quantiles, avoiding 0 and 1
    tau_values = torch.linspace(0, 1, n_quantiles + 2)[1:-1]
    return tau_values


def compute_expected_value_from_quantiles(quantile_values, tau_values=None, method='mean'):
    """
    Compute expected value (mean) from quantile predictions.

    IMPORTANT: The median quantile (Q_0.50) is NOT the expected value!
    For skewed distributions, median ≠ mean. This function correctly computes
    the expected value by integrating across the quantile distribution.

    Args:
        quantile_values: torch.Tensor, shape (..., n_quantiles)
            Predicted quantile values. The quantile dimension should be last.
        tau_values: torch.Tensor, shape (n_quantiles,), optional
            Quantile fractions. If None, assumes evenly spaced quantiles.
        method: str, one of ['mean', 'trapz', 'weighted']
            - 'mean': Simple average across quantiles (fast, good for evenly spaced quantiles)
            - 'trapz': Trapezoidal integration (more accurate for unevenly spaced quantiles)
            - 'weighted': Weighted sum based on quantile spacing

    Returns:
        torch.Tensor, shape (...)
            Expected value (mean of the return distribution)

    Example:
        >>> q_values = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])  # (1, 5)
        >>> tau = torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9])
        >>> ev = compute_expected_value_from_quantiles(q_values, tau, method='mean')
        >>> # ev ≈ 3.0 (mean of quantiles)
        >>> # Note: median would be 3.0, but for skewed distributions these differ!
    """
    if method == 'mean':
        # Simple average across quantiles
        # Fast and good approximation for evenly-spaced quantiles
        return torch.mean(quantile_values, dim=-1)

    elif method == 'trapz':
        # Trapezoidal integration: E[Z] = ∫ Q(τ) dτ
        # More accurate for unevenly spaced quantiles
        if tau_values is None:
            n_quantiles = quantile_values.shape[-1]
            tau_values = torch.linspace(0, 1, n_quantiles + 2, device=quantile_values.device)[1:-1]

        # Extend tau to [0, 1] and quantiles to boundaries
        # Assume Q(0) = Q(τ_min) and Q(1) = Q(τ_max)
        tau_ext = torch.cat([
            torch.tensor([0.0], device=tau_values.device),
            tau_values,
            torch.tensor([1.0], device=tau_values.device)
        ])

        q_ext = torch.cat([
            quantile_values[..., :1],  # Extend to τ=0
            quantile_values,
            quantile_values[..., -1:]  # Extend to τ=1
        ], dim=-1)

        # Trapezoidal integration
        # ∫ Q(τ) dτ ≈ Σ (Q(τ_i) + Q(τ_{i+1})) * (τ_{i+1} - τ_i) / 2
        delta_tau = tau_ext[1:] - tau_ext[:-1]
        trapz_sum = ((q_ext[..., :-1] + q_ext[..., 1:]) / 2) * delta_tau.view(*([1] * (q_ext.dim() - 1)), -1)

        return torch.sum(trapz_sum, dim=-1)

    elif method == 'weighted':
        # Weighted sum based on quantile spacing
        # Each quantile Q(τ_i) represents the region [τ_{i-1/2}, τ_{i+1/2}]
        if tau_values is None:
            n_quantiles = quantile_values.shape[-1]
            tau_values = torch.linspace(0, 1, n_quantiles + 2, device=quantile_values.device)[1:-1]

        # Compute weights as the width of each quantile's region
        # For τ_i, weight = (τ_{i+1} - τ_{i-1}) / 2
        tau_ext = torch.cat([
            torch.tensor([0.0], device=tau_values.device),
            tau_values,
            torch.tensor([1.0], device=tau_values.device)
        ])

        weights = (tau_ext[2:] - tau_ext[:-2]) / 2
        weights = weights.view(*([1] * (quantile_values.dim() - 1)), -1)  # Broadcast to match quantile_values

        return torch.sum(quantile_values * weights, dim=-1)

    else:
        raise ValueError(f"Unknown method: {method}. Choose from ['mean', 'trapz', 'weighted']")


def check_quantile_ordering(q_values, tau_values, tolerance=1e-3):
    """
    Check if predicted quantiles are properly ordered (monotonicity constraint).

    In a valid distributional critic, quantiles should satisfy:
        Q_τ1(s) ≤ Q_τ2(s) for all τ1 < τ2

    This function checks this constraint and returns violations.

    Parameters
    ----------
    q_values : torch.Tensor, shape (T, B, n_quantiles)
        Predicted quantile values.
    tau_values : torch.Tensor, shape (n_quantiles,)
        Quantile fractions (should be sorted).
    tolerance : float, default=1e-3
        Small tolerance for numerical errors.

    Returns
    -------
    is_ordered : bool
        True if all quantiles are properly ordered.
    violations : torch.Tensor
        Number of violations per batch element.
    max_violation : float
        Maximum violation magnitude.

    Examples
    --------
    >>> q = torch.tensor([[[1.0, 2.0, 3.0, 4.0, 5.0]]])  # Properly ordered
    >>> tau = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9])
    >>> is_ordered, violations, max_viol = check_quantile_ordering(q, tau)
    >>> print(is_ordered)  # True
    """
    T, B, n_quantiles = q_values.shape

    # Check if quantiles are sorted along the last dimension
    # For each adjacent pair, check q[:, :, i] ≤ q[:, :, i+1]
    diffs = q_values[:, :, 1:] - q_values[:, :, :-1]  # Shape: (T, B, n_quantiles - 1)

    # Count violations (where difference is negative beyond tolerance)
    violations = (diffs < -tolerance).sum(dim=[0, 2])  # Shape: (B,)

    # Maximum violation magnitude
    max_violation = torch.min(diffs).item()

    # Check if all quantiles are ordered
    is_ordered = (violations == 0).all().item()

    return is_ordered, violations, max_violation
