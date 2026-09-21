# Copyright 2026 The MRound Authors
# SPDX-License-Identifier: Apache-2.0
"""Block reconstruction losses, in NumPy.

Implements DOCUMENTATION.md sections 1.3 and 5.5, with gradients.

The outlier-suppressed loss has three properties that are counterintuitive and
that change the gradient. They are implemented deliberately here rather than
being allowed to emerge, and each is asserted in the test suite:

- Selection is over the flattened batch, not per row.
- Excluded elements are zeroed in the numerator, but the denominator stays the
  full element count, so they dilute the mean rather than leaving it.
- The reduction is always the mean, including under gradient accumulation.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

__all__ = [
    "OUTLIER_FRACTION",
    "outlier_suppressed_loss",
    "reconstruction_loss",
]

Array = npt.NDArray[np.float64]

# Fraction of elements excluded from the objective under outlier suppression.
OUTLIER_FRACTION: float = 0.001


def _masked(shape: tuple[int, ...], attention_mask: Array | None) -> Array:
    if attention_mask is None:
        return np.ones(shape)
    return np.broadcast_to(attention_mask, shape).astype(np.float64)


def reconstruction_loss(
    predicted: Array,
    reference: Array,
    *,
    attention_mask: Array | None = None,
) -> tuple[float, Array]:
    """Mean squared error between a block's quantized and original outputs.

    Computed in float64 here (float32 in the MLX implementation). The
    reconstruction errors are small and reduced precision loses them, so wide
    accumulation is not negotiable.

    Args:
        predicted: Output of the block with fake-quantized weights.
        reference: Cached output of the same block at full precision.
        attention_mask: Broadcast over the feature dimension to exclude padding.
            ``None`` weights every position equally.

    Returns:
        ``(loss, grad)`` where ``grad`` is the derivative with respect to
        ``predicted``.
    """
    weights = _masked(predicted.shape, attention_mask)
    residual = (predicted - reference) * weights
    n = predicted.size
    loss = float(np.sum(residual**2) / n)
    grad = 2.0 * residual * weights / n
    return loss, grad


def outlier_suppressed_loss(
    predicted: Array,
    reference: Array,
    *,
    attention_mask: Array | None = None,
    fraction: float = OUTLIER_FRACTION,
) -> tuple[float, Array]:
    """Reconstruction loss with the largest errors excluded.

    At very low bit widths a handful of enormous errors dominate the mean and
    destabilize the optimization; excluding them stabilizes tuning.

    Args:
        predicted: Output of the block with fake-quantized weights.
        reference: Cached output of the same block at full precision.
        attention_mask: Broadcast over the feature dimension to exclude padding.
        fraction: Share of elements to exclude, as a fraction of the total.

    Returns:
        ``(loss, grad)`` where ``grad`` is the derivative with respect to
        ``predicted``.
    """
    weights = _masked(predicted.shape, attention_mask)
    residual = (predicted - reference) * weights

    n = predicted.size
    k = max(1, int(n * fraction))

    # Selection is global over the flattened batch, not per row. Taking the
    # top k per row would exclude k times as many elements and would exclude
    # them from rows that have no outliers at all.
    flat = np.abs(residual).reshape(-1)
    if k >= n:
        keep = np.zeros(n, dtype=bool)
    else:
        cutoff_index = np.argpartition(flat, n - k)[n - k :]
        keep = np.ones(n, dtype=bool)
        keep[cutoff_index] = False
    keep_mask = keep.reshape(residual.shape).astype(np.float64)

    kept = residual * keep_mask
    # The denominator stays the full element count. Excluded elements therefore
    # dilute the mean rather than leaving it, which is a deliberate choice and
    # not an oversight: it keeps the loss magnitude comparable across steps as
    # the set of outliers changes.
    loss = float(np.sum(kept**2) / n)
    grad = 2.0 * kept * keep_mask * weights / n
    return loss, grad
