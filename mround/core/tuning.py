# Copyright 2026 The MRound Authors
# SPDX-License-Identifier: Apache-2.0
"""The learned-rounding tuning loop, in MLX.

Mirrors :mod:`mround.reference.tuning`, and operates on a single linear layer
for the same reason: it is the smallest unit containing the whole method, it
needs no model, and it is the granularity at which comparison against the
reference is meaningful. The block loop adds orchestration around this; it does
not add mathematics.

Two things here are MLX design decisions rather than translations.

**Where ``mx.eval`` is called.** MLX evaluates lazily, and both failure modes are
reachable from a loop like this one: reading a scalar every step forces a
synchronization that can dominate runtime, while never evaluating lets the graph
grow until memory is exhausted. This loop evaluates the parameters and the loss
once per step, together, in one call. That bounds the graph to a single step's
work while paying exactly one synchronization, which is the cheapest correct
choice. The losses are kept as a Python list because they are already
materialized by that point.

**Gradients come from ``mx.value_and_grad``.** The reference derives them by
hand out of necessity; here there is no reason to, and letting MLX do it removes
a whole class of error. That the two agree is then a real cross-check.

**The layer error is one matrix product, not the difference of two.** Writing it
the obvious way, as ``x @ Wq.T`` minus a cached ``x @ W.T``, subtracts two nearly
equal quantities, and MLX's matrix product does not run at full float32: measured
relative error is 8.5e-4, roughly four orders of magnitude worse than float32.
That error is proportional to the large product, not to the small difference, so
the subtraction hands the whole of it to the residual. At 8 bits it amounts to 20
percent of the signal being optimized. Multiplying the weight difference instead
keeps the error proportional to the residual itself, and costs one product rather
than two. See MEMORY.md D-014.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import mlx.core as mx

from mround.core.losses import outlier_suppressed_loss, reconstruction_loss
from mround.core.quantizer import fake_quantize, init_params, project_params
from mround.core.signsgd import LinearDecay, SignSGD
from mround.schemes import DEFAULT_EPS_FP32, QuantScheme, TuningConfig

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = ["TuningResult", "round_to_nearest", "tune_layer"]

Params = dict[str, mx.array]


@dataclasses.dataclass(frozen=True, slots=True)
class TuningResult:
    """What tuning a single layer achieved.

    Attributes:
        params: The learned quantities at the best step seen.
        qdq: The reconstruction those parameters produce.
        initial_loss: Output error before tuning, which is exactly what
            round-to-nearest achieves.
        final_loss: Output error at the best step.
        losses: The loss at every step, for inspecting the trajectory.
        best_step: Which step produced ``final_loss``.
    """

    params: Params
    qdq: mx.array
    initial_loss: float
    final_loss: float
    losses: list[float]
    best_step: int

    @property
    def improvement(self) -> float:
        """Fractional loss reduction against round-to-nearest."""
        if self.initial_loss <= 0.0:
            return 0.0
        return 1.0 - (self.final_loss / self.initial_loss)


def round_to_nearest(
    weight: mx.array,
    scheme: QuantScheme,
    *,
    eps: float = DEFAULT_EPS_FP32,
    init_scale: mx.array | None = None,
) -> mx.array:
    """Quantize with no learning, the baseline every result is measured against."""
    return fake_quantize(
        weight, init_params(weight, scheme), scheme, eps=eps, init_scale=init_scale
    )


def tune_layer(
    weight: mx.array,
    activations: mx.array,
    scheme: QuantScheme,
    config: TuningConfig | None = None,
    *,
    eps: float = DEFAULT_EPS_FP32,
    init_scale: mx.array | None = None,
) -> TuningResult:
    """Learn the rounding for one linear layer.

    Args:
        weight: Shaped ``(out_features, in_features)``.
        activations: Calibration inputs, shaped ``(n_samples, in_features)``.
        scheme: Target representation.
        config: Tuning hyperparameters. ``None`` uses the standard recipe.
        eps: Scale epsilon.
        init_scale: The searched per-group scale, required by and only by a
            ``SEARCHED`` scheme. Tuning then starts from the search result,
            with beta as a multiplier on it. See MEMORY.md D-030.

    Returns:
        The best parameters found and what they achieved.

    Raises:
        ValueError: If the activation width does not match the weight.
    """
    if activations.shape[1] != weight.shape[1]:
        msg = (
            f"activations have {activations.shape[1]} features but the weight "
            f"expects {weight.shape[1]}"
        )
        raise ValueError(msg)

    config = config or TuningConfig()
    suppress = config.resolved_suppress_outliers(scheme.bits)

    # The losses take (predicted, reference) because the block-level objective in
    # Phase 2 genuinely has two output tensors to compare. Here the residual is
    # available directly and more accurately, so it is passed as the prediction
    # against a zero reference rather than reconstructed from two large products
    # that would then be subtracted. See MEMORY.md D-014.
    zero = mx.zeros((), dtype=weight.dtype)

    def objective(params: Mapping[str, mx.array]) -> mx.array:
        delta = fake_quantize(weight, params, scheme, eps=eps, init_scale=init_scale) - weight
        residual = activations @ delta.T
        if suppress:
            return outlier_suppressed_loss(residual, zero)
        return reconstruction_loss(residual, zero)

    loss_and_grad = mx.value_and_grad(objective)

    params = init_params(weight, scheme)
    schedule = LinearDecay(config.resolved_lr(scheme.bits), config.iters)
    optimizer = SignSGD(schedule)

    initial = objective(params)
    mx.eval(initial)
    initial_loss = float(initial)

    best_loss = initial_loss
    best_params = dict(params)
    best_step = 0
    losses: list[float] = [initial_loss]

    for step in range(config.iters):
        loss, grads = loss_and_grad(params)
        updated = optimizer.apply(params, grads)

        # The projection is load-bearing, not hygiene. See MEMORY.md D-008.
        updated = project_params(updated, scheme)

        # One synchronization per step, covering the loss and the new
        # parameters together. Evaluating less often lets the graph grow across
        # steps; evaluating the loss separately would pay two round trips.
        mx.eval(loss, *updated.values())

        value = float(loss)
        losses.append(value)
        if value < best_loss:
            best_loss = value
            best_params = dict(params)
            best_step = step

        params = updated

    qdq = fake_quantize(weight, best_params, scheme, eps=eps, init_scale=init_scale)
    mx.eval(qdq)
    return TuningResult(
        params=best_params,
        qdq=qdq,
        initial_loss=initial_loss,
        final_loss=best_loss,
        losses=losses,
        best_step=best_step,
    )
