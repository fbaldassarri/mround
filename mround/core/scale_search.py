# Copyright 2026 The MRound Authors
# SPDX-License-Identifier: Apache-2.0
"""The v2 candidate scale grid search, in MLX.

Mirrors :mod:`mround.reference.scale_search` candidate for candidate; the
algorithm, the grid, the anchor convention, and every fidelity note live there
and in DOCUMENTATION.md sections 1.5, 5.3 and 5.10. When the two disagree, the
reference is presumed right.

Two things here are MLX design decisions rather than translations.

**Candidates are evaluated in chunks, not one by one and not all at once.** One
at a time is two hundred graph evaluations per layer; all at once materializes a
``candidates x weights`` tensor that reaches tens of gigabytes on an embedding
matrix. A chunk of candidates is stacked on a leading axis, reduced to per-group
losses, and folded into the running best, which bounds memory at ``chunk x
weight`` while keeping the evaluation count small.

**Losses accumulate in float32.** The reference oracle uses float64, which the
GPU does not have. The parity suite therefore compares the quality of the chosen
scales rather than demanding identical choices, since a near-exact tie can
resolve differently across that precision gap; the property that matters, that
neither implementation's choice reconstructs measurably worse, survives it.

The search is forward-only. Nothing here is differentiated, so plain ``mx.clip``
is correct and ``clip_ste`` deliberately does not appear.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import mlx.core as mx

from mround.core.quantizer import round_half_to_even
from mround.reference.scale_search import SearchGrid
from mround.schemes import QuantScheme, Symmetry

if TYPE_CHECKING:
    import numpy.typing as npt

__all__ = ["SearchGrid", "search_scales"]

# Candidates evaluated per graph launch. Memory per chunk is roughly
# chunk x weight in float32; sixteen keeps a 50k x 576 embedding under one
# gigabyte per step while cutting launches to about thirteen per layer.
_CHUNK = 16


def _reciprocal(values: mx.array) -> mx.array:
    """Elementwise reciprocal with zero mapped to zero, as the reference does."""
    return mx.where(values == 0, mx.zeros_like(values), 1.0 / values)


def search_scales(
    weight: mx.array,
    scheme: QuantScheme,
    *,
    importance: mx.array | npt.NDArray | None = None,  # type: ignore[type-arg]
    grid: SearchGrid | None = None,
) -> mx.array:
    """Choose an initial scale per group by minimizing reconstruction error.

    Args:
        weight: Weight matrix, shaped ``(out_features, in_features)``.
        scheme: Target representation. Symmetric only.
        importance: Per-input-channel weights. ``None`` weights every element
            equally. Only the relative weighting matters.
        grid: Candidate family. ``None`` uses the default.

    Returns:
        One signed scale per group, shaped ``(n_groups, 1)``, float32, already
        evaluated. Zero for an all-zero group, for the caller to clamp.

    Raises:
        ValueError: If the scheme is not symmetric, the weight does not group
            evenly, or the importance does not match the input width.
    """
    if scheme.symmetry is not Symmetry.SYMMETRIC:
        msg = "the searched initialization is defined for symmetric schemes only"
        raise ValueError(msg)
    if weight.ndim != 2 or weight.shape[-1] % scheme.group_size:  # noqa: PLR2004
        msg = (
            f"search expects a 2-D weight whose last dimension is a multiple "
            f"of group_size={scheme.group_size}, got shape {weight.shape}"
        )
        raise ValueError(msg)

    grouped = weight.astype(mx.float32).reshape(-1, scheme.group_size)
    nmax = float(2 ** (scheme.bits - 1))
    lo, hi = -nmax, nmax - 1.0

    if importance is None:
        qw = None
    else:
        flat = mx.array(importance).astype(mx.float32).reshape(-1)
        if flat.size != weight.shape[-1]:
            msg = (
                f"importance has {flat.size} channels but the weight has "
                f"{weight.shape[-1]} input features"
            )
            raise ValueError(msg)
        qw = mx.broadcast_to(flat[None, :], weight.shape).reshape(-1, scheme.group_size)

    anchor_index = mx.argmax(mx.abs(grouped), axis=-1, keepdims=True)
    group_max = mx.take_along_axis(grouped, anchor_index, axis=-1)

    def evaluate(effective: mx.array) -> tuple[mx.array, mx.array]:
        """Scales and weighted losses for a ``(k, 1, 1)`` stack of candidates.

        Returns ``(k, n_groups, 1)`` scales and losses, reduced but lazy.
        """
        inverse = -effective * _reciprocal(group_max)[None, :, :]
        codes = mx.clip(round_half_to_even(inverse * grouped[None, :, :]), lo, hi)
        scale = _reciprocal(inverse)
        residual = mx.square(scale * codes - grouped[None, :, :])
        if qw is not None:
            residual = residual * qw[None, :, :]
        return scale, mx.sum(residual, axis=-1, keepdims=True)

    best_scale, best_loss = evaluate(mx.array([[[nmax]]], dtype=mx.float32))
    best_scale, best_loss = best_scale[0], best_loss[0]

    step, half_count = (grid or SearchGrid()).steps(scheme.bits)
    offsets = [index for index in range(-half_count, half_count + 1) if index != 0]

    for start in range(0, len(offsets), _CHUNK):
        chunk = offsets[start : start + _CHUNK]
        effective = mx.array([nmax - step * index for index in chunk], dtype=mx.float32).reshape(
            -1, 1, 1
        )
        scales, losses = evaluate(effective)

        # Fold the chunk into the running best in candidate order, so ties
        # break toward the earliest candidate exactly as the reference's
        # sequential loop does. A chunk-wide argmin would break them toward
        # the lowest index within the chunk under float equality, which is the
        # same rule, but interleaving with the running best would not be.
        for position in range(len(chunk)):
            improved = losses[position] < best_loss
            best_scale = mx.where(improved, scales[position], best_scale)
            best_loss = mx.where(improved, losses[position], best_loss)

        # One evaluation per chunk: the fold above is cheap elementwise work,
        # and evaluating here keeps the graph from growing across two hundred
        # candidates while paying thirteen synchronizations per layer, not two
        # hundred.
        mx.eval(best_scale, best_loss)

    return best_scale
