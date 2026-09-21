# Copyright 2026 The MRound Authors
# SPDX-License-Identifier: Apache-2.0
"""Block reconstruction losses, in MLX.

Mirrors :mod:`mround.reference.losses`. These return the scalar loss only, and
let ``mx.value_and_grad`` supply the derivative, which is why they carry no
gradient code of their own.

The outlier-suppressed loss has three properties that are counterintuitive and
that change the gradient. They are implemented deliberately rather than allowed
to emerge, and each is asserted in the parity tests:

- Selection is over the flattened batch, not per row.
- Excluded elements are zeroed in the numerator, but the denominator stays the
  full element count, so they dilute the mean rather than leaving it.
- The reduction is always the mean, including under gradient accumulation.

DOCUMENTATION.md sections 1.3 and 5.5.
"""

from __future__ import annotations

import mlx.core as mx

__all__ = ["OUTLIER_FRACTION", "outlier_suppressed_loss", "reconstruction_loss"]

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
    k = max(1, int(n * fraction))

    magnitude = mx.abs(residual).reshape(-1)
    # Selection is global over the flattened batch. Per-row selection would
    # excise k elements from every row, including rows with no outliers at all.
    threshold = mx.sort(magnitude)[n - k]
    keep = (mx.abs(residual) < threshold).astype(mx.float32)

    kept = residual * keep
    # The denominator stays the full element count, so excluded elements dilute
    # the mean rather than leaving it. That keeps the loss magnitude comparable
    # across steps as the set of outliers changes.
    return mx.sum(kept * kept) / n
