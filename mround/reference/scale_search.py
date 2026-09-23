# Copyright 2026 The MRound Authors
# SPDX-License-Identifier: Apache-2.0
"""The v2 candidate scale grid search, in NumPy.

This is the feature that separates SignRound v2 from v1, and the evidence for
why it exists is MEMORY.md D-028: learned rounding optimizes placement within a
fixed grid, and at 2 bits the grid itself is wrong. The search fixes the grid
before the rounding is learned, by trying a family of clipping thresholds per
group and keeping the one with the lowest weighted reconstruction error.

The candidate family is a one-dimensional line search over the effective code
range. The anchor maps the group's signed largest-magnitude element onto
``-nmax`` exactly, which is D-009's signed symmetric scale; each candidate
replaces ``nmax`` by ``nmax - step * i``. A negative ``i`` widens the effective
range, so the mapping slope shrinks and the largest elements land beyond the
clamp: it clips the tail and raises resolution for the bulk. A positive ``i``
narrows it, so the largest element lands inside the code range and the outer
codes go unused: coarser everywhere, never clipped. The clamp bounds never
move; only the mapping slope does.

Fidelity notes, each verified against the reference implementation's source
(analysis 04, sections 5.2 and 4.3) and each a place where a plausible
alternative would silently change the result:

- **The candidate count is fixed** at 200 plus the anchor for every width
  except 2 bits, where a hard-coded finer window applies: 180 candidates at
  step 0.01, spanning effective ranges 1.10 to 2.90. The ``ratio`` parameter
  moves the span, never the count.
- **Ties break toward the earliest candidate**, and the iteration runs from the
  widest effective range to the narrowest, with the anchor winning all ties.
  Strict improvement is required to replace it.
- **An all-zero group returns scale 0**, which no candidate can beat; the
  caller clamps it away from zero, exactly as the quantizer's epsilon does.
- **The rounding rule is the project's round-half-to-even**, which is also what
  the reference's ``torch.round`` implements.

One deliberate divergence: losses accumulate in float64 here rather than the
reference's float32, because this module is the oracle and the MLX port is the
one that pays platform precision. Near-exact ties can therefore resolve
differently between the two; the parity suite compares the quality of the
chosen scales rather than demanding identical choices.

Specification: DOCUMENTATION.md sections 1.5 and 5.3.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import numpy.typing as npt

from mround.reference.quantize import round_half_to_even
from mround.schemes import QuantScheme, Symmetry

__all__ = ["SearchGrid", "search_scales"]

Array = npt.NDArray[np.float64]

# The 2-bit special case, hard-coded in the reference: a finer step over a
# narrower window, bypassing the ratio entirely. Ninety steps each way at 0.01
# scans effective ranges 1.10 to 2.90 around nmax = 2.
_TWO_BIT_HALF_COUNT = 90
_TWO_BIT_STEP = 0.01

# Elsewhere the half-count is always grid/2 = 100 regardless of width or ratio:
# step = (nmax * ratio) / grid * 2, then half_count = int(nmax * ratio / step).
_GRID = 200


@dataclasses.dataclass(frozen=True, slots=True)
class SearchGrid:
    """The candidate family, as perturbations of the effective code range.

    Attributes:
        ratio: How far the perturbation extends, as a fraction of ``nmax``.
            Controls the span only; the candidate count is fixed by the grid.
            The reference's default, and the span measurements behind it, are
            in analysis 04 section 5.2.
    """

    ratio: float = 0.75

    def steps(self, bits: int) -> tuple[float, int]:
        """The step size and half-count for ``bits``.

        Returns:
            ``(step, half_count)``: candidates are ``i`` in
            ``[-half_count, half_count]`` excluding zero, each with effective
            range ``nmax - step * i``.
        """
        if bits == 2:  # noqa: PLR2004
            return _TWO_BIT_STEP, _TWO_BIT_HALF_COUNT
        nmax = float(2 ** (bits - 1))
        span = nmax * self.ratio
        step = span / _GRID * 2
        return step, int(span / step)


def _reciprocal(values: Array) -> Array:
    """Elementwise reciprocal with zero mapped to zero, as the reference does.

    The guard is what makes an all-zero group return scale 0 rather than
    dividing by zero, and it also covers the candidate whose effective range
    crosses exactly zero at ``ratio >= 1``.
    """
    out = np.zeros_like(values)
    np.divide(1.0, values, out=out, where=values != 0)
    return out


def _grouped(weight: Array, group_size: int) -> Array:
    """Reshape to ``(n_groups, group_size)``, refusing ragged input."""
    if weight.ndim != 2 or weight.shape[-1] % group_size:  # noqa: PLR2004
        msg = (
            f"search expects a 2-D weight whose last dimension is a multiple "
            f"of group_size={group_size}, got shape {weight.shape}"
        )
        raise ValueError(msg)
    return weight.reshape(-1, group_size)


def search_scales(
    weight: npt.NDArray[np.floating],
    scheme: QuantScheme,
    *,
    importance: npt.NDArray[np.floating] | None = None,
    grid: SearchGrid | None = None,
) -> Array:
    """Choose an initial scale per group by minimizing reconstruction error.

    Args:
        weight: Weight matrix, shaped ``(out_features, in_features)``.
        scheme: Target representation. Symmetric only; the searched
            parameterization is defined for it alone.
        importance: Per-input-channel weights, typically the summed squared
            activations each channel sees. ``None`` weights every element
            equally, making this a plain least-squares search. Only the
            relative weighting matters, so any global normalization of the
            importance is irrelevant to the result.
        grid: Candidate family. ``None`` uses the default.

    Returns:
        One signed scale per group, shaped ``(n_groups, 1)`` where groups run
        row-major over the grouped weight. Zero for an all-zero group, which
        the caller is expected to clamp, exactly as it clamps the observed
        scale by epsilon.

    Raises:
        ValueError: If the scheme is not symmetric, or the weight does not
            group evenly, or the importance does not match the input width.
    """
    if scheme.symmetry is not Symmetry.SYMMETRIC:
        msg = "the searched initialization is defined for symmetric schemes only"
        raise ValueError(msg)

    grouped = _grouped(np.asarray(weight, dtype=np.float64), scheme.group_size)
    nmax = float(2 ** (scheme.bits - 1))
    lo, hi = -nmax, nmax - 1.0

    if importance is None:
        qw = None
    else:
        flat = np.asarray(importance, dtype=np.float64).reshape(-1)
        if flat.size != weight.shape[-1]:
            msg = (
                f"importance has {flat.size} channels but the weight has "
                f"{weight.shape[-1]} input features"
            )
            raise ValueError(msg)
        # Every row of the weight sees the same per-input-channel importance,
        # so the grouped layout tiles it across rows.
        qw = np.tile(flat, weight.shape[0]).reshape(-1, scheme.group_size)

    # The anchor: the signed largest-magnitude element mapped onto -nmax
    # exactly, which is the same convention the quantizer's symmetric scale
    # uses (D-009). Sign travels with the scale.
    anchor_index = np.argmax(np.abs(grouped), axis=-1, keepdims=True)
    group_max = np.take_along_axis(grouped, anchor_index, axis=-1)

    def evaluate(effective_range: float) -> tuple[Array, Array]:
        """The scale and weighted loss for one candidate, per group."""
        inverse = -effective_range * _reciprocal(group_max)
        codes = np.clip(round_half_to_even(inverse * grouped), lo, hi)
        scale = _reciprocal(inverse)
        residual = (scale * codes - grouped) ** 2
        if qw is not None:
            residual = residual * qw
        # keepdims so the loss broadcasts against the (n_groups, 1) scale in
        # the selection below rather than fanning out to (n_groups, n_groups).
        return scale, np.sum(residual, axis=-1, keepdims=True)

    best_scale, best_loss = evaluate(nmax)

    step, half_count = (grid or SearchGrid()).steps(scheme.bits)
    for index in range(-half_count, half_count + 1):
        if index == 0:
            continue
        scale, loss = evaluate(nmax - step * index)
        improved = loss < best_loss
        best_scale = np.where(improved, scale, best_scale)
        best_loss = np.where(improved, loss, best_loss)

    return best_scale
