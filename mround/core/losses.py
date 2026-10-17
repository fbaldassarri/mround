# Copyright 2026 The MRound Authors
# SPDX-License-Identifier: Apache-2.0
"""Block reconstruction losses, in MLX.

Mirrors :mod:`mround.reference.losses`. These return the scalar loss only, and
let ``mx.value_and_grad`` supply the derivative, which is why they carry no
gradient code of their own.

The outlier-suppressed loss has three properties that are counterintuitive and
that change the gradient. They are implemented deliberately rather than allowed
to emerge, and the first two are checked in tests/unit/test_mlx_parity.py by
agreement with the reference's value:

- Selection is over the flattened batch, not per row.
- Excluded elements are zeroed in the numerator, but the denominator stays the
  full element count, so they dilute the mean rather than leaving it.
- The reduction is always the mean over the whole batch. A batch run in micro
  batches (``TuningConfig.micro_batches``) keeps it: each piece's sum is
  divided by the whole batch's element count (:func:`piece_loss`), and the
  threshold is taken over the whole batch before any piece is differentiated
  (:func:`outlier_threshold`), so the pieces add up to the batch's own loss.

DOCUMENTATION.md sections 1.3 and 5.5.
"""

from __future__ import annotations

import mlx.core as mx

__all__ = [
    "OUTLIER_FRACTION",
    "outlier_suppressed_loss",
    "outlier_threshold",
    "piece_loss",
    "reconstruction_loss",
]

# Fraction of elements excluded from the objective under outlier suppression.
OUTLIER_FRACTION: float = 0.001


def reconstruction_loss(
    predicted: mx.array,
    reference: mx.array,
    *,
    attention_mask: mx.array | None = None,
) -> mx.array:
    """Mean squared error between a block's quantized and original outputs.

    Accumulated in float32. The reconstruction errors are small and reduced
    precision loses them, so this is not negotiable (DOCUMENTATION.md 5.1).
    """
    residual = predicted.astype(mx.float32) - reference.astype(mx.float32)
    if attention_mask is not None:
        residual = residual * attention_mask.astype(mx.float32)
    return mx.mean(residual * residual)


def outlier_suppressed_loss(
    predicted: mx.array,
    reference: mx.array,
    *,
    attention_mask: mx.array | None = None,
    fraction: float = OUTLIER_FRACTION,
) -> mx.array:
    """Reconstruction loss with the largest errors excluded.

    At very low bit widths a handful of enormous errors dominate the mean and
    destabilize the optimization; excluding them stabilizes tuning.

    MLX offers no boolean mask assignment, so the mask is built by a threshold
    comparison against the k-th largest absolute error rather than by scattering
    into a boolean array. That is the MLX-idiomatic form of the same operation,
    and it has one edge case worth naming: when several errors tie at exactly the
    threshold, a threshold comparison excludes all of them rather than exactly
    k. With floating-point errors on real data that is vanishingly unlikely, and
    excluding a few extra outliers is the benign direction to err.
    """
    residual = predicted.astype(mx.float32) - reference.astype(mx.float32)
    if attention_mask is not None:
        residual = residual * attention_mask.astype(mx.float32)

    n = residual.size
    # Selection is global over the flattened batch. Per-row selection would
    # excise k elements from every row, including rows with no outliers at all.
    threshold = outlier_threshold(residual, fraction=fraction)
    keep = (mx.abs(residual) < threshold).astype(mx.float32)

    kept = residual * keep
    # The denominator stays the full element count, so excluded elements dilute
    # the mean rather than leaving it. That keeps the loss magnitude comparable
    # across steps as the set of outliers changes.
    return mx.sum(kept * kept) / n


def outlier_threshold(residual: mx.array, *, fraction: float = OUTLIER_FRACTION) -> mx.array:
    """The magnitude from which the outlier rule excludes an error.

    The ``k``-th largest absolute value of the flattened ``residual``, with
    ``k = max(1, int(n * fraction))``; an error is kept when it is strictly
    smaller. :func:`outlier_suppressed_loss` takes it from the batch it is
    given; a batch run in micro batches takes it once over the whole batch,
    before any piece is differentiated, and hands it to :func:`piece_loss`.
    """
    n = residual.size
    k = max(1, int(n * fraction))
    return mx.sort(mx.abs(residual).reshape(-1))[n - k]


def piece_loss(
    predicted: mx.array,
    reference: mx.array,
    *,
    elements: int,
    threshold: mx.array | None = None,
) -> mx.array:
    """One piece's share of a whole batch's loss.

    The sum of the piece's squared errors over ``elements``, the whole batch's
    element count, so the shares of a batch's pieces add up to the batch's
    mean and their gradients to its gradient, up to the order of the
    additions. With ``threshold`` the errors at or above it are zeroed first,
    which is the outlier rule with the threshold the whole batch set
    (:func:`outlier_threshold`); without it this is the plain reconstruction
    loss. Accumulated in float32, as both losses are.
    """
    residual = predicted.astype(mx.float32) - reference.astype(mx.float32)
    if threshold is not None:
        residual = residual * (mx.abs(residual) < threshold).astype(mx.float32)
    return mx.sum(residual * residual) / elements
