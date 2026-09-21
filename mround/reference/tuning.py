# Copyright 2026 The MRound Authors
# SPDX-License-Identifier: Apache-2.0
"""The learned-rounding tuning loop, in NumPy.

This is the method assembled: quantize a weight matrix with learnable rounding,
measure how far the layer's output drifts from the original, and walk the
learnable quantities downhill with signed gradient descent.

It operates on a single linear layer rather than a transformer block, which is
deliberate. A layer is the smallest unit that contains the whole method, it
needs no model to exercise, and it is exactly the granularity at which parity
against a reference is meaningful. The block loop in the MLX implementation adds
orchestration around this; it does not add mathematics.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import numpy.typing as npt

from mround.reference.losses import outlier_suppressed_loss, reconstruction_loss
from mround.reference.optimizer import LinearDecay, SignSGD
from mround.reference.quantize import (
    QuantParams,
    fake_quantize,
    init_params,
    project_params,
    quantize_grad,
)
from mround.schemes import DEFAULT_EPS_FP32, QuantScheme, TuningConfig

__all__ = ["TuningResult", "round_to_nearest", "tune_layer"]

Array = npt.NDArray[np.float64]


@dataclasses.dataclass(frozen=True, slots=True)
class TuningResult:
    """What tuning a single layer achieved.

    Attributes:
        params: The learned quantities at the best step seen.
        qdq: The reconstruction those parameters produce.
        initial_loss: Output error before tuning, which is exactly what
            round-to-nearest achieves, since the initial parameters reproduce it.
        final_loss: Output error at the best step.
        losses: The loss at every step, for inspecting the trajectory.
        best_step: Which step produced ``final_loss``.
    """

    params: QuantParams
    qdq: Array
    initial_loss: float
    final_loss: float
    losses: list[float]
    best_step: int

    @property
    def improvement(self) -> float:
        """Fractional loss reduction against round-to-nearest.

        The honest measure of whether learned rounding earned its cost. A value
        at or below zero means tuning did not help.
        """
        if self.initial_loss <= 0.0:
            return 0.0
        return 1.0 - (self.final_loss / self.initial_loss)


def round_to_nearest(weight: Array, scheme: QuantScheme) -> Array:
    """Quantize with no learning, the baseline every result is measured against."""
    return fake_quantize(weight, init_params(weight, scheme), scheme).qdq


def _layer_loss(
    activations: Array,
    weight_q: Array,
    weight: Array,
    *,
    suppress_outliers: bool,
) -> tuple[float, Array]:
    """Loss and its gradient with respect to the quantized weight.

    The layer computes ``out = x @ W.T``, so the chain rule from the output
    gradient back to the weight is ``dL/dW = (dL/dout).T @ x``.

    The residual is formed by multiplying the weight *difference*, rather than by
    subtracting two large products. In float64 that is a distinction without a
    difference; it is written this way so the reference and the MLX
    implementation compute the same expression, and there the distinction is
    severe. See MEMORY.md D-014.
    """
    residual = activations @ (weight_q - weight).T
    zero = np.zeros(())
    if suppress_outliers:
        loss, grad_out = outlier_suppressed_loss(residual, zero)
    else:
        loss, grad_out = reconstruction_loss(residual, zero)
    return loss, grad_out.T @ activations


def tune_layer(
    weight: Array,
    activations: Array,
    scheme: QuantScheme,
    config: TuningConfig | None = None,
    *,
    eps: float = DEFAULT_EPS_FP32,
    init_scale: Array | None = None,
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

    params = init_params(weight, scheme)

    schedule = LinearDecay(config.resolved_lr(scheme.bits), config.iters)
    optimizer = SignSGD(schedule)

    initial_loss, _ = _layer_loss(
        activations,
        fake_quantize(weight, params, scheme, eps=eps, init_scale=init_scale).qdq,
        weight,
        suppress_outliers=suppress,
    )

    best_loss = initial_loss
    best_params = params.copy()
    best_step = 0
    losses: list[float] = [initial_loss]

    for step in range(config.iters):
        result = fake_quantize(weight, params, scheme, eps=eps, init_scale=init_scale)
        loss, grad_weight = _layer_loss(activations, result.qdq, weight, suppress_outliers=suppress)
        losses.append(loss)

        if loss < best_loss:
            best_loss = loss
            best_params = params.copy()
            best_step = step

        grads = quantize_grad(weight, params, scheme, grad_weight, eps=eps, init_scale=init_scale)
        updated = optimizer.apply(
            {"v": params.v, "alpha": params.alpha, "beta": params.beta},
            {"v": grads.v, "alpha": grads.alpha, "beta": grads.beta},
        )
        params = QuantParams(updated["v"], updated["alpha"], updated["beta"])

        # The projection is not hygiene. Signed gradient descent respects no
        # constraints of its own, so this is the only thing keeping the
        # rounding perturbation within half a code. See MEMORY.md D-008.
        project_params(params, scheme)

    return TuningResult(
        params=best_params,
        qdq=fake_quantize(weight, best_params, scheme, eps=eps, init_scale=init_scale).qdq,
        initial_loss=initial_loss,
        final_loss=best_loss,
        losses=losses,
        best_step=best_step,
    )
