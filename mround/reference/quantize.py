# Copyright 2026 The MRound Authors
# SPDX-License-Identifier: Apache-2.0
"""Differentiable fake quantization, in NumPy.

Implements DOCUMENTATION.md section 5.2 exactly, for both symmetry modes, with
analytic gradients.

The single most important detail in this file is the sign of the symmetric
scale, which is inverted relative to the group's dominant extreme. That is not a
quirk to be normalized away; it is the mechanism that makes the asymmetric
two's-complement code range work, and removing it costs 50 percent of every
group's largest weight at 2 bits. See :func:`symmetric_scale` and
DOCUMENTATION.md section 5.2.1.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from mround.exceptions import SchemeError
from mround.schemes import DEFAULT_EPS_FP32, QuantScheme, ScaleInit, Symmetry

__all__ = [
    "V_BOUND",
    "QuantParams",
    "QuantResult",
    "asymmetric_scale",
    "dequantize",
    "fake_quantize",
    "init_params",
    "project_params",
    "quantize_grad",
    "regroup",
    "round_half_to_even",
    "symmetric_scale",
    "ungroup",
]

Array = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]

# The rounding perturbation is bounded to half a code in each direction. Within
# it a weight reaches either adjacent representable value and no further, which
# is what makes the method learned *rounding*. See MEMORY.md D-008 and D-010.
V_BOUND: float = 0.5

# A weight matrix is two-dimensional; named so the check reads as intent.
MATRIX_NDIM: int = 2


class QuantParams:
    """The learnable quantities for one weight matrix.

    Attributes:
        v: Per-weight rounding perturbation, shaped like the grouped weight.
            Initialized to zero, projected to ``[-V_BOUND, V_BOUND]``.
        alpha: Per-group clipping coefficient on the upper extreme.
        beta: Per-group clipping coefficient on the lower extreme.
    """

    __slots__ = ("alpha", "beta", "v")

    def __init__(self, v: Array, alpha: Array, beta: Array) -> None:
        """Store the three learnable arrays without copying."""
        self.v = v
        self.alpha = alpha
        self.beta = beta

    def copy(self) -> QuantParams:
        """Return an independent copy, for best-state tracking."""
        return QuantParams(self.v.copy(), self.alpha.copy(), self.beta.copy())


class QuantResult:
    """The outcome of quantizing one weight matrix.

    Attributes:
        qdq: The dequantized approximation, in the original ungrouped shape.
        codes: Integer codes in grouped shape. Signed for symmetric schemes,
            unsigned for asymmetric ones.
        scale: Per-group scale, shaped ``(rows, n_groups, 1)``. May be negative
            under symmetric schemes; that is deliberate.
        zero_point: Per-group zero point, or ``None`` for symmetric schemes.
    """

    __slots__ = ("codes", "qdq", "scale", "zero_point")

    def __init__(
        self,
        qdq: Array,
        codes: Array,
        scale: Array,
        zero_point: Array | None,
    ) -> None:
        """Store the result arrays without copying."""
        self.qdq = qdq
        self.codes = codes
        self.scale = scale
        self.zero_point = zero_point


def round_half_to_even(x: Array) -> Array:
    """Round to nearest, ties to even.

    NumPy's ``rint`` already does this, but the function exists so the tie rule
    is named rather than assumed. It is load-bearing: the analysis of the
    reference implementation found that its own self-tests fail under
    round-half-away-from-zero, which makes this the most likely source of a
    silent divergence when porting.
    """
    return np.rint(x)


def regroup(weight: Array, group_size: int) -> tuple[Array, int]:
    """Reshape a weight matrix into per-group rows, padding if needed.

    Args:
        weight: Shaped ``(out_features, in_features)``.
        group_size: Weights sharing one scale. ``-1`` means one group per output
            channel.

    Returns:
        ``(grouped, pad)`` where ``grouped`` is ``(out_features, n_groups,
        group_size)`` and ``pad`` is the number of padding elements added to the
        final group.

    Raises:
        SchemeError: If ``weight`` is not two-dimensional.
    """
    if weight.ndim != MATRIX_NDIM:
        msg = f"expected a 2-D weight matrix, got shape {weight.shape}"
        raise SchemeError(msg)

    rows, cols = weight.shape
    if group_size == -1:
        return weight.reshape(rows, 1, cols), 0

    pad = (-cols) % group_size
    if pad:
        # Pad with zeros. A padded element never influences the scale, because
        # the group extremes are clamped to include zero, and its reconstructed
        # value is discarded by ungroup.
        weight = np.concatenate([weight, np.zeros((rows, pad), dtype=weight.dtype)], axis=1)
    return weight.reshape(rows, -1, group_size), pad


def ungroup(grouped: Array, shape: tuple[int, int], pad: int) -> Array:
    """Invert :func:`regroup`, discarding padding."""
    flat = grouped.reshape(shape[0], -1)
    if pad:
        flat = flat[:, : shape[1]]
    return flat


def _clipped_extremes(grouped: Array, alpha: Array, beta: Array) -> tuple[Array, Array]:
    """Return the per-group extremes after clipping, with zero forced into range.

    Forcing zero into the range matters for groups whose weights are all one
    sign. Without it the asymmetric zero point falls outside the code range and
    no format storing an unsigned integer zero point can represent the group.
    """
    w_max = np.clip(grouped.max(axis=-1, keepdims=True), 0.0, None) * alpha
    w_min = np.clip(grouped.min(axis=-1, keepdims=True), None, 0.0) * beta
    return w_max, w_min


def _clamp_magnitude(scale: Array, eps: float) -> Array:
    """Move ``scale`` away from zero to at least ``eps``, preserving its sign.

    Zero has no sign, so the degenerate case is specified rather than left to
    the implementation: an all-zero group yields a positive epsilon. Without
    that branch, an all-zero group reintroduces exactly the division by zero
    that epsilon exists to prevent.
    """
    sign = np.sign(scale)
    sign = np.where(sign == 0.0, 1.0, sign)
    return sign * np.maximum(np.abs(scale), eps)


def symmetric_scale(
    grouped: Array,
    alpha: Array,
    beta: Array,
    bits: int,
    eps: float,
) -> tuple[Array, BoolArray]:
    """Compute the symmetric per-group scale, and which side dominates.

    The scale carries the sign OPPOSITE to the dominant extreme. When the
    positive side dominates the scale is negative, so the largest positive
    weight maps onto the code ``-2**(bits-1)``, which is the one extra code the
    two's-complement range provides. It therefore reconstructs exactly rather
    than clipping. The mirror case holds when the negative side dominates.

    See DOCUMENTATION.md section 5.2.1 for why this is strictly better than
    either naive alternative.

    Args:
        grouped: Weights in grouped shape.
        alpha: Per-group upper clipping coefficient.
        beta: Per-group lower clipping coefficient.
        bits: Bit width.
        eps: Scale epsilon.

    Returns:
        ``(scale, positive_dominates)``. The boolean array records which branch
        was taken, which the gradient needs because only one of the two
        coefficients receives gradient in each group.
    """
    maxq = float(2 ** (bits - 1))
    w_max, w_min = _clipped_extremes(grouped, alpha, beta)

    pos_mag = w_max  # already non-negative
    neg_mag = -w_min  # w_min is non-positive, so this is non-negative

    positive_dominates = pos_mag >= neg_mag
    dominant = np.where(positive_dominates, pos_mag, neg_mag)
    signed = np.where(positive_dominates, -dominant, dominant)

    scale = _clamp_magnitude(signed / maxq, eps)
    return scale, positive_dominates


def asymmetric_scale(
    grouped: Array,
    alpha: Array,
    beta: Array,
    bits: int,
    eps: float,
) -> tuple[Array, Array]:
    """Compute the asymmetric per-group scale and zero point.

    Because zero was forced into the range by :func:`_clipped_extremes`, the
    zero point lies in ``[0, 2**bits - 1]`` and every consuming format can
    represent it.

    Returns:
        ``(scale, zero_point)``. The scale is strictly positive here, so a plain
        minimum clamp is correct.
    """
    maxq = float(2**bits - 1)
    w_max, w_min = _clipped_extremes(grouped, alpha, beta)
    scale = np.maximum((w_max - w_min) / maxq, eps)
    zero_point = round_half_to_even(-w_min / scale)
    return scale, zero_point


def fake_quantize(
    weight: Array,
    params: QuantParams,
    scheme: QuantScheme,
    *,
    eps: float = DEFAULT_EPS_FP32,
    init_scale: Array | None = None,
) -> QuantResult:
    """Quantize and immediately dequantize.

    Args:
        weight: Shaped ``(out_features, in_features)``.
        params: The learnable quantities.
        scheme: Target representation.
        eps: Scale epsilon.
        init_scale: The searched per-group scale, required by and only by a
            ``SEARCHED`` scheme. See :mod:`mround.reference.scale_search`.

    Returns:
        The reconstruction together with the codes and scale parameters.
    """
    grouped, pad = regroup(weight, scheme.group_size)
    q_min, q_max = scheme.code_range

    # The searched parameterization (DOCUMENTATION.md 5.3): the scale is the
    # searched value times the learned coefficient, and the observed extremes
    # play no part. beta is the multiplier and alpha is inert, mirroring the
    # reference, whose searched branch reads max_scale and ignores min_scale.
    if scheme.scale_init is ScaleInit.SEARCHED:
        if init_scale is None:
            msg = (
                "a SEARCHED scheme needs the searched scale: pass init_scale "
                "from scale_search.search_scales"
            )
            raise ValueError(msg)
        # The search reports one scale per group in row-major order; the
        # grouped layout here is (rows, groups_per_row, group_size), and the
        # two flatten identically, so a reshape is exact rather than a guess.
        base = np.asarray(init_scale, dtype=grouped.dtype).reshape(
            grouped.shape[0], grouped.shape[1], 1
        )
        scale = _clamp_magnitude(base * params.beta, eps)
        codes = np.clip(round_half_to_even(grouped / scale + params.v), q_min, q_max)
        return QuantResult(
            qdq=ungroup(scale * codes, weight.shape, pad),
            codes=codes,
            scale=scale,
            zero_point=None,
        )
    if init_scale is not None:
        msg = "init_scale was given but the scheme's scale_init is not SEARCHED"
        raise ValueError(msg)

    if scheme.symmetry is Symmetry.SYMMETRIC:
        scale, _ = symmetric_scale(grouped, params.alpha, params.beta, scheme.bits, eps)
        raw = round_half_to_even(grouped / scale + params.v)
        codes = np.clip(raw, q_min, q_max)
        qdq_grouped = scale * codes
        zero_point = None
    else:
        scale, zero_point = asymmetric_scale(grouped, params.alpha, params.beta, scheme.bits, eps)
        raw = round_half_to_even(grouped / scale + params.v) + zero_point
        codes = np.clip(raw, q_min, q_max)
        qdq_grouped = scale * (codes - zero_point)

    return QuantResult(
        qdq=ungroup(qdq_grouped, weight.shape, pad),
        codes=codes,
        scale=scale,
        zero_point=zero_point,
    )


def dequantize(
    codes: Array,
    scale: Array,
    zero_point: Array | None,
    shape: tuple[int, int],
    scheme: QuantScheme,
) -> Array:
    """Reconstruct weights from codes and scale parameters."""
    grouped = scale * codes if zero_point is None else scale * (codes - zero_point)
    _, pad = regroup(np.zeros(shape), scheme.group_size)
    return ungroup(grouped, shape, pad)


def init_params(weight: Array, scheme: QuantScheme) -> QuantParams:
    """Create learnable parameters at their initial values.

    The rounding perturbation starts at zero (round to nearest) and both
    clipping coefficients start at one (no clipping), so the initial
    reconstruction is exactly round-to-nearest. That property is what makes the
    tuning improvement measurable: the loss at step zero is the RTN baseline.
    """
    grouped, _ = regroup(weight, scheme.group_size)
    group_shape = (grouped.shape[0], grouped.shape[1], 1)
    return QuantParams(
        v=np.zeros_like(grouped),
        alpha=np.ones(group_shape, dtype=grouped.dtype),
        beta=np.ones(group_shape, dtype=grouped.dtype),
    )


def project_params(params: QuantParams, scheme: QuantScheme) -> None:
    """Clip the learnable quantities back into their permitted ranges, in place.

    Called after every optimizer step, never inside the loss. Signed gradient
    descent respects no constraints of its own, so this is the only thing
    keeping the rounding perturbation within half a code.

    Omitting this does not fail loudly. It produces a plausible implementation
    of unconstrained weight learning that will pass most tests. See MEMORY.md
    D-008.
    """
    lo, hi = scheme.coefficient_bounds
    np.clip(params.v, -V_BOUND, V_BOUND, out=params.v)
    np.clip(params.alpha, lo, hi, out=params.alpha)
    np.clip(params.beta, lo, hi, out=params.beta)


def _searched_grad(
    grouped: Array,
    params: QuantParams,
    scheme: QuantScheme,
    g: Array,
    init_scale: Array,
    eps: float,
) -> QuantParams:
    """The searched branch of :func:`quantize_grad`, on the grouped arrays."""
    q_min, q_max = scheme.code_range
    base = np.asarray(init_scale, dtype=grouped.dtype).reshape(
        grouped.shape[0], grouped.shape[1], 1
    )
    scale = _clamp_magnitude(base * params.beta, eps)
    raw = round_half_to_even(grouped / scale + params.v)
    inside = ((raw >= q_min) & (raw <= q_max)).astype(grouped.dtype)
    codes = np.clip(raw, q_min, q_max)

    grad_v = g * scale * inside
    # d(qdq)/ds = codes - inside * weight / scale, and ds/dbeta is the searched
    # value itself. The epsilon clamp on the magnitude binds only where the
    # product has collapsed to nothing, and there the scale is not a function
    # of beta at all, so its derivative is dropped.
    d_qdq_d_scale = codes - inside * grouped / scale
    grad_scale = (g * d_qdq_d_scale).sum(axis=-1, keepdims=True)
    grad_beta = grad_scale * base
    return QuantParams(v=grad_v, alpha=np.zeros_like(grad_beta), beta=grad_beta)


def quantize_grad(
    weight: Array,
    params: QuantParams,
    scheme: QuantScheme,
    grad_qdq: Array,
    *,
    eps: float = DEFAULT_EPS_FP32,
    init_scale: Array | None = None,
) -> QuantParams:
    """Backpropagate a gradient on the reconstruction to the learnable parameters.

    The straight-through estimator makes ``round`` behave as the identity in the
    backward pass; the clamp contributes a zero-one mask. Everything else is
    ordinary calculus on the scale.

    One consequence is worth stating because it surprises people and it explains
    an observation from the reference implementation: in symmetric mode only ONE
    of ``alpha`` and ``beta`` receives gradient in each group, namely the one on
    the dominant side. The other has no influence on the scale at all, so its
    gradient is exactly zero and the optimizer moves it on the sign of nothing.

    Under the searched parameterization the scale is the searched value times
    ``beta`` alone, so ``beta`` takes the whole scale gradient through the
    searched value and ``alpha`` receives exactly zero (DOCUMENTATION.md 5.3,
    MEMORY.md D-030). Until this branch existed a searched scheme was
    silently differentiated as an observed-range one here, which is the
    opposite assignment of the two coefficients.

    Args:
        weight: The matrix being quantized.
        params: The current parameter values.
        scheme: Target representation.
        grad_qdq: Gradient of the loss with respect to the reconstruction,
            in the same ungrouped shape as ``weight``.
        eps: Scale epsilon.
        init_scale: The searched per-group scale, required by and only by a
            ``SEARCHED`` scheme, exactly as :func:`fake_quantize` takes it.

    Returns:
        Gradients with respect to ``v``, ``alpha``, and ``beta``.
    """
    grouped, pad = regroup(weight, scheme.group_size)
    if pad:
        grad_flat = np.concatenate(
            [grad_qdq, np.zeros((weight.shape[0], pad), dtype=grad_qdq.dtype)], axis=1
        )
    else:
        grad_flat = grad_qdq
    g = grad_flat.reshape(grouped.shape[0], -1, grouped.shape[2])

    q_min, q_max = scheme.code_range

    if scheme.scale_init is ScaleInit.SEARCHED:
        if init_scale is None:
            msg = (
                "a SEARCHED scheme needs the searched scale: pass init_scale "
                "from scale_search.search_scales"
            )
            raise ValueError(msg)
        return _searched_grad(grouped, params, scheme, g, init_scale, eps)
    if init_scale is not None:
        msg = "init_scale was given but the scheme's scale_init is not SEARCHED"
        raise ValueError(msg)

    if scheme.symmetry is Symmetry.SYMMETRIC:
        scale, positive_dominates = symmetric_scale(
            grouped, params.alpha, params.beta, scheme.bits, eps
        )
        raw = round_half_to_even(grouped / scale + params.v)
        inside = ((raw >= q_min) & (raw <= q_max)).astype(grouped.dtype)
        codes = np.clip(raw, q_min, q_max)

        # d(qdq)/dv = scale, gated by the clamp.
        grad_v = g * scale * inside

        # d(qdq)/ds = codes - inside * weight / scale
        d_qdq_d_scale = codes - inside * grouped / scale
        grad_scale = (g * d_qdq_d_scale).sum(axis=-1, keepdims=True)

        # ds/dalpha and ds/dbeta. Only the dominant side contributes.
        maxq = float(2 ** (scheme.bits - 1))
        w_max_raw = np.clip(grouped.max(axis=-1, keepdims=True), 0.0, None)
        w_min_raw = np.clip(grouped.min(axis=-1, keepdims=True), None, 0.0)
        d_scale_d_alpha = np.where(positive_dominates, -w_max_raw / maxq, 0.0)
        d_scale_d_beta = np.where(positive_dominates, 0.0, -w_min_raw / maxq)

        grad_alpha = grad_scale * d_scale_d_alpha
        grad_beta = grad_scale * d_scale_d_beta
    else:
        scale, zero_point = asymmetric_scale(grouped, params.alpha, params.beta, scheme.bits, eps)
        raw = round_half_to_even(grouped / scale + params.v) + zero_point
        inside = ((raw >= q_min) & (raw <= q_max)).astype(grouped.dtype)
        codes = np.clip(raw, q_min, q_max)

        grad_v = g * scale * inside

        # qdq = scale * (codes - zp). Substituting the straight-through forms
        # makes this collapse pleasantly: inside the clamp the zero point
        # cancels entirely and qdq = weight + scale * (v + rounding residual),
        # while outside it qdq = scale * bound + w_min - scale * zp residual.
        _, w_min = _clipped_extremes(grouped, params.alpha, params.beta)
        d_zp_d_scale = w_min / (scale * scale)
        d_codes_d_scale = inside * (-grouped / (scale * scale) + d_zp_d_scale)
        d_qdq_d_scale = (codes - zero_point) + scale * (d_codes_d_scale - d_zp_d_scale)
        grad_scale = (g * d_qdq_d_scale).sum(axis=-1, keepdims=True)

        maxq = float(2**scheme.bits - 1)
        w_max_raw = np.clip(grouped.max(axis=-1, keepdims=True), 0.0, None)
        w_min_raw = np.clip(grouped.min(axis=-1, keepdims=True), None, 0.0)

        # The lower coefficient reaches the reconstruction by two routes, not
        # one: through the scale, and directly through the zero point, since
        # w_min = w_min_raw * beta appears in zp as well. The direct route
        # survives only where the clamp is active, because inside the clamp the
        # zero point cancels out of the reconstruction entirely. Omitting this
        # term is the kind of error that leaves the optimization still
        # descending, just along a slightly wrong direction.
        direct_beta = (g * (1.0 - inside) * w_min_raw).sum(axis=-1, keepdims=True)

        grad_alpha = grad_scale * (w_max_raw / maxq)
        grad_beta = grad_scale * (-w_min_raw / maxq) + direct_beta

    return QuantParams(v=grad_v, alpha=grad_alpha, beta=grad_beta)
