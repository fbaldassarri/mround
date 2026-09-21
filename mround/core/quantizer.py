# Copyright 2026 The MRound Authors
# SPDX-License-Identifier: Apache-2.0
"""Differentiable fake quantization, in MLX.

The Metal implementation of DOCUMENTATION.md section 5.2. It mirrors
:mod:`mround.reference.quantize` function for function and argument for
argument, deliberately, so that the two can be compared directly and any
disagreement localizes to a single call rather than to "somewhere in the port".

Three MLX-specific decisions are made here and each is load-bearing.

**Rounding does not use ``mx.round``.** The tie rule matters (DOCUMENTATION.md
section 5.1) and depending on another framework's default is exactly the kind of
assumption that produces a silent divergence. :func:`round_half_to_even` builds
the rule from floor and a parity test instead, so it is correct regardless of
what ``mx.round`` happens to do.

**Gradients come from ``mx.value_and_grad``, not from hand derivation.** The
reference implementation derives them by hand because NumPy has no autodiff, and
doing that caught a real error there (a missing term where the lower clipping
coefficient reaches the reconstruction through the zero point as well as through
the scale). MLX can differentiate the forward pass directly, so it should. The
two routes agreeing is then a genuine cross-check rather than a tautology.

**The clamp counts its own boundary as inside.** ``mx.clip`` does not, and the
difference is not cosmetic: the value reaching the clamp here has already been
rounded, so it is an integer, and landing exactly on a boundary is the common
case rather than a measure-zero one. At 2 bits it happens to 43 percent of
weights. :func:`clip_ste` therefore supplies the boundary convention explicitly
instead of inheriting one. See MEMORY.md D-013.

**Nothing is mutated in place.** MLX arrays are immutable, and the projection
that bounds the learnable quantities happens after the optimizer step rather
than inside the traced function. That is a real behavioral difference from
implementations that clamp mid-forward, and it is recorded in MEMORY.md D-010.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import mlx.core as mx

from mround.exceptions import SchemeError
from mround.schemes import DEFAULT_EPS_FP32, QuantScheme, ScaleInit, Symmetry

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = [
    "MATRIX_NDIM",
    "V_BOUND",
    "asymmetric_scale",
    "clip_ste",
    "dequantize",
    "fake_quantize",
    "init_params",
    "project_params",
    "regroup",
    "round_half_to_even",
    "round_ste",
    "symmetric_scale",
    "ungroup",
]

# The rounding perturbation is bounded to half a code in each direction. See
# MEMORY.md D-008 and D-010.
V_BOUND: float = 0.5

# A weight matrix is two-dimensional; named so the check reads as intent.
MATRIX_NDIM: int = 2

Params = dict[str, mx.array]


def round_half_to_even(x: mx.array) -> mx.array:
    """Round to nearest, ties to even, without relying on the framework default.

    Constructed from floor rather than from ``mx.round`` on purpose. The tie
    rule is load-bearing (the reference implementation's own self-tests fail
    under round-half-away-from-zero), and inheriting whichever rule a framework
    happens to implement is how a port acquires a divergence that no test
    written against the port itself will ever catch.

    The construction: split the input into its floor and its fraction, round
    up where the fraction exceeds a half, and on an exact half round up only
    where the floor is odd. An earlier form took ``floor(x + 0.5)`` and stepped
    back on ties, and the addition made it wrong for exactly one float32 value
    in every unit interval, ``0.5 + 1 ulp``, where ``x + 0.5`` rounds to the
    integer and is mistaken for a tie: the value rounded to 0 instead of 1.
    Brute force over every float32 in ``[0.25, 16]`` agrees with ``np.rint``
    with this form and disagrees once with the old one.
    """
    half = 0.5
    base = mx.floor(x)
    fraction = x - base
    is_odd = mx.abs(mx.remainder(base, 2.0)) == 1.0
    up = (fraction > half) | ((fraction == half) & is_odd)
    return mx.where(up, base + 1.0, base)


def round_ste(x: mx.array) -> mx.array:
    """Round with a straight-through gradient.

    Forward rounds; backward behaves as though this were the identity. Without
    it the gradient is zero almost everywhere and nothing can be learned.
    """
    return x + mx.stop_gradient(round_half_to_even(x) - x)


def clip_ste(x: mx.array, lo: float, hi: float) -> mx.array:
    """Clip, counting a value exactly on the boundary as inside the clamp.

    Forward is ``mx.clip``. Backward passes the gradient through wherever
    ``lo <= x <= hi``, inclusive at both ends, and blocks it outside.

    ``mx.clip`` is exclusive at the boundary: differentiate through it and a
    value sitting exactly on ``lo`` or ``hi`` receives nothing. That would be an
    irrelevant detail if the boundary were reached only by coincidence, but here
    the input has already been rounded, so it is an integer and hitting the
    boundary exactly is the ordinary case. At 2 bits it is 43 percent of weights,
    at 4 bits 7 percent, at 8 bits none, which is exactly the bit-width pattern
    the parity suite reported before this function existed.

    Mathematically either convention is a valid subgradient: the straight-through
    surrogate is ``clip(w/s + v + c, lo, hi)`` with the rounding residual ``c``
    frozen, its clamp argument sits precisely at the kink, and the subderivative
    of ``clip`` there is the whole interval ``[0, 1]``. So this is a choice, and
    it is made on behavior rather than on calculus. A weight whose code sits on
    ``hi`` is not saturated in both directions; it can still move down. Passing
    the gradient keeps that descent direction available, and a step that would
    push it past ``hi`` is simply absorbed by the forward clip, costing one
    wasted step. Blocking the gradient freezes the weight at its
    round-to-nearest value for the entire optimization, which at 2 bits discards
    nearly half the learnable degrees of freedom. A wasted step is recoverable;
    a frozen parameter is not. See MEMORY.md D-013.
    """
    inside = mx.stop_gradient(((x >= lo) & (x <= hi)).astype(x.dtype))
    passthrough = x * inside
    return passthrough + mx.stop_gradient(mx.clip(x, lo, hi) - passthrough)


def regroup(weight: mx.array, group_size: int) -> tuple[mx.array, int]:
    """Reshape a weight matrix into per-group rows, padding if needed.

    Returns ``(grouped, pad)`` with ``grouped`` shaped
    ``(out_features, n_groups, group_size)``.

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
        # Zeros never influence the scale, because the group extremes are
        # clamped to include zero, and their reconstruction is discarded.
        weight = mx.concatenate([weight, mx.zeros((rows, pad), dtype=weight.dtype)], axis=1)
    return weight.reshape(rows, -1, group_size), pad


def ungroup(grouped: mx.array, shape: tuple[int, int], pad: int) -> mx.array:
    """Invert :func:`regroup`, discarding padding."""
    flat = grouped.reshape(shape[0], -1)
    if pad:
        flat = flat[:, : shape[1]]
    return flat


def _clipped_extremes(
    grouped: mx.array, alpha: mx.array, beta: mx.array
) -> tuple[mx.array, mx.array]:
    """Per-group extremes after clipping, with zero forced into the range.

    Forcing zero in is what guarantees the asymmetric zero point stays
    representable for groups whose weights are all one sign.
    """
    w_max = mx.clip(grouped.max(axis=-1, keepdims=True), 0.0, None) * alpha
    w_min = mx.clip(grouped.min(axis=-1, keepdims=True), None, 0.0) * beta
    return w_max, w_min


def _check_init_scale(scheme: QuantScheme, init_scale: mx.array | None) -> None:
    """Refuse mismatches between the scheme and the searched scale, both ways.

    A SEARCHED scheme without its scale would silently have to fall back to the
    observed range, which is a different algorithm; an init_scale under an
    OBSERVED_RANGE scheme means the caller thinks the search is on when it is
    not. Both are bugs upstream, and both are cheaper here than in a result.
    """
    if scheme.scale_init is ScaleInit.SEARCHED and init_scale is None:
        msg = (
            "a SEARCHED scheme needs the searched scale: pass init_scale from "
            "scale_search.search_scales"
        )
        raise ValueError(msg)
    if scheme.scale_init is not ScaleInit.SEARCHED and init_scale is not None:
        msg = "init_scale was given but the scheme's scale_init is not SEARCHED"
        raise ValueError(msg)


def _clamp_magnitude(scale: mx.array, eps: float) -> mx.array:
    """Move ``scale`` away from zero to at least ``eps``, preserving its sign.

    A plain minimum clamp would be wrong here, because the symmetric scale is
    genuinely signed. Zero has no sign, so that case is specified rather than
    left to the implementation.
    """
    sign = mx.sign(scale)
    sign = mx.where(sign == 0.0, 1.0, sign)
    return sign * mx.maximum(mx.abs(scale), eps)


def symmetric_scale(
    grouped: mx.array,
    alpha: mx.array,
    beta: mx.array,
    bits: int,
    eps: float,
) -> mx.array:
    """The symmetric per-group scale, signed opposite to the dominant extreme.

    That inversion maps the dominant extreme onto ``-2**(bits-1)``, the extra
    code the two's-complement range provides, so it reconstructs exactly rather
    than clipping. See DOCUMENTATION.md section 5.2.1.
    """
    maxq = float(2 ** (bits - 1))
    w_max, w_min = _clipped_extremes(grouped, alpha, beta)

    pos_mag = w_max
    neg_mag = -w_min

    positive_dominates = pos_mag >= neg_mag
    dominant = mx.where(positive_dominates, pos_mag, neg_mag)
    signed = mx.where(positive_dominates, -dominant, dominant)
    return _clamp_magnitude(signed / maxq, eps)


def asymmetric_scale(
    grouped: mx.array,
    alpha: mx.array,
    beta: mx.array,
    bits: int,
    eps: float,
) -> tuple[mx.array, mx.array]:
    """The asymmetric per-group scale and zero point."""
    maxq = float(2**bits - 1)
    w_max, w_min = _clipped_extremes(grouped, alpha, beta)
    scale = mx.maximum((w_max - w_min) / maxq, eps)
    zero_point = round_ste(-w_min / scale)
    return scale, zero_point


def _unpack_params(
    params: Mapping[str, mx.array] | None, dtype: mx.Dtype
) -> tuple[mx.array | None, mx.array, mx.array]:
    """The three learnable quantities, or their round-to-nearest values.

    ``None`` means round-to-nearest, and it is not the same as passing
    :func:`init_params`. Those are numerically identical but allocate a
    perturbation the size of the weight matrix and two per-group coefficient
    arrays, all of them constant. Across a whole model that is a large amount of
    memory traffic to add zero to a number, so the round-to-nearest path skips
    the addition entirely and broadcasts scalar coefficients instead.
    """
    if params is None:
        one = mx.array(1.0, dtype=dtype)
        return None, one, one
    return params["v"], params["alpha"], params["beta"]


def fake_quantize(
    weight: mx.array,
    params: Mapping[str, mx.array] | None,
    scheme: QuantScheme,
    *,
    eps: float = DEFAULT_EPS_FP32,
    init_scale: mx.array | None = None,
) -> mx.array:
    """Quantize and immediately dequantize, differentiably.

    Returns only the reconstruction, because that is all the loss needs and
    returning it alone keeps the function trivially differentiable by
    ``mx.value_and_grad``. Use :func:`quantize_codes` when the integer codes and
    scale parameters are wanted, for example at export time.

    Args:
        weight: Shaped ``(out_features, in_features)``.
        params: Mapping with keys ``v``, ``alpha``, ``beta``, or ``None`` for
            round-to-nearest.
        scheme: Target representation.
        eps: Scale epsilon.
        init_scale: The searched per-group scale, required by and only by a
            ``SEARCHED`` scheme. See :mod:`mround.core.scale_search`.

    Returns:
        The reconstruction, in the original ungrouped shape.
    """
    grouped, pad = regroup(weight, scheme.group_size)
    q_min, q_max = scheme.code_range
    v, alpha, beta = _unpack_params(params, grouped.dtype)

    # The searched parameterization: scale is the searched value times the
    # learned beta, alpha is inert, and the observed extremes play no part.
    # Mirrors the reference's searched branch. See DOCUMENTATION.md 5.3.
    _check_init_scale(scheme, init_scale)
    if scheme.scale_init is ScaleInit.SEARCHED:
        assert init_scale is not None
        # One scale per group, row-major from the search, reshaped to the
        # (rows, groups_per_row, 1) layout the grouped weight uses. Explicit
        # rather than matching beta's shape, because beta is a scalar on the
        # round-to-nearest path.
        base = init_scale.astype(grouped.dtype).reshape(grouped.shape[0], grouped.shape[1], 1)
        scale = _clamp_magnitude(base * beta, eps)
        raw = grouped / scale if v is None else grouped / scale + v
        codes = clip_ste(round_ste(raw), q_min, q_max)
        return ungroup(scale * codes, (weight.shape[0], weight.shape[1]), pad)

    # clip_ste rather than mx.clip: the boundary counts as inside for the
    # gradient, which is the common case here and not an edge case. See D-013.
    if scheme.symmetry is Symmetry.SYMMETRIC:
        scale = symmetric_scale(grouped, alpha, beta, scheme.bits, eps)
        raw = grouped / scale if v is None else grouped / scale + v
        codes = clip_ste(round_ste(raw), q_min, q_max)
        qdq_grouped = scale * codes
    else:
        scale, zero_point = asymmetric_scale(grouped, alpha, beta, scheme.bits, eps)
        raw = grouped / scale if v is None else grouped / scale + v
        codes = clip_ste(round_ste(raw) + zero_point, q_min, q_max)
        qdq_grouped = scale * (codes - zero_point)

    return ungroup(qdq_grouped, (weight.shape[0], weight.shape[1]), pad)


def quantize_codes(
    weight: mx.array,
    params: Mapping[str, mx.array] | None,
    scheme: QuantScheme,
    *,
    eps: float = DEFAULT_EPS_FP32,
    init_scale: mx.array | None = None,
) -> tuple[mx.array, mx.array, mx.array | None]:
    """Return ``(codes, scale, zero_point)`` without the reconstruction.

    Separate from :func:`fake_quantize` because the tuning loop only ever wants
    the reconstruction, and keeping the differentiated function small avoids
    dragging unused outputs through the graph.

    Args:
        weight: Shaped ``(out_features, in_features)``.
        params: Learned quantities, or ``None`` for round-to-nearest.
        scheme: Target representation.
        eps: Scale epsilon.
        init_scale: The searched per-group scale, required by and only by a
            ``SEARCHED`` scheme. See :mod:`mround.core.scale_search`.

    Returns:
        ``(codes, scale, zero_point)``, all grouped. ``zero_point`` is ``None``
        under symmetric schemes, where the fixed ``2 ** (bits - 1)`` applies.
    """
    grouped, _ = regroup(weight, scheme.group_size)
    q_min, q_max = scheme.code_range
    v, alpha, beta = _unpack_params(params, grouped.dtype)

    _check_init_scale(scheme, init_scale)
    if scheme.scale_init is ScaleInit.SEARCHED:
        assert init_scale is not None
        base = init_scale.astype(grouped.dtype).reshape(grouped.shape[0], grouped.shape[1], 1)
        scale = _clamp_magnitude(base * beta, eps)
        raw = grouped / scale if v is None else grouped / scale + v
        codes = mx.clip(round_half_to_even(raw), q_min, q_max)
        return codes, scale, None

    if scheme.symmetry is Symmetry.SYMMETRIC:
        scale = symmetric_scale(grouped, alpha, beta, scheme.bits, eps)
        raw = grouped / scale if v is None else grouped / scale + v
        codes = mx.clip(round_half_to_even(raw), q_min, q_max)
        return codes, scale, None

    scale, zero_point = asymmetric_scale(grouped, alpha, beta, scheme.bits, eps)
    raw = grouped / scale if v is None else grouped / scale + v
    codes = mx.clip(round_half_to_even(raw) + zero_point, q_min, q_max)
    return codes, scale, zero_point


def dequantize(
    codes: mx.array,
    scale: mx.array,
    zero_point: mx.array | None,
    shape: tuple[int, int],
    scheme: QuantScheme,
) -> mx.array:
    """Reconstruct weights from codes and scale parameters."""
    grouped = scale * codes if zero_point is None else scale * (codes - zero_point)
    _, pad = regroup(mx.zeros(shape), scheme.group_size)
    return ungroup(grouped, shape, pad)


def init_params(weight: mx.array, scheme: QuantScheme) -> Params:
    """Learnable parameters at their initial values.

    Zero perturbation and unit coefficients, so the initial reconstruction is
    exactly round-to-nearest. That is what makes the tuning improvement
    measurable against a known baseline rather than an arbitrary start.
    """
    grouped, _ = regroup(weight, scheme.group_size)
    group_shape = (grouped.shape[0], grouped.shape[1], 1)
    return {
        "v": mx.zeros(grouped.shape, dtype=grouped.dtype),
        "alpha": mx.ones(group_shape, dtype=grouped.dtype),
        "beta": mx.ones(group_shape, dtype=grouped.dtype),
    }


def project_params(params: Mapping[str, mx.array], scheme: QuantScheme) -> Params:
    """Clip the learnable quantities back into their permitted ranges.

    Returns a new mapping; nothing is mutated. Called after every optimizer
    step, never inside the differentiated function.

    This is not hygiene. Signed gradient descent respects no constraints of its
    own, so this is the only thing keeping the rounding perturbation within half
    a code. Omitting it produces a plausible implementation of unconstrained
    weight learning that passes most tests. See MEMORY.md D-008.
    """
    lo, hi = scheme.coefficient_bounds
    return {
        "v": mx.clip(params["v"], -V_BOUND, V_BOUND),
        "alpha": mx.clip(params["alpha"], lo, hi),
        "beta": mx.clip(params["beta"], lo, hi),
    }
