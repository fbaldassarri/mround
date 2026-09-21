# Copyright 2026 The MRound Authors
# SPDX-License-Identifier: Apache-2.0
"""Signed gradient descent, in MLX.

Mirrors :mod:`mround.reference.optimizer`. The update rule is one line; the rest
is the schedule, and the schedule is the part that actually constrains the
method.

Because every step moves a parameter by exactly the current rate and the rate
decays linearly to zero, total travel is finite and known: about
``lr * iters / 2``. With the default ``lr = c / iters`` that is ``c / 2``,
independent of the step count, which is why the learning rate and the iteration
count are coupled and cannot be tuned separately. DOCUMENTATION.md section 5.4
and MEMORY.md D-010.

Parameters are transformed functionally rather than mutated, following the MLX
convention. Writing the optimizer natively is simpler than adapting one built
around mutable tensors and parameter identity, which is the shape the reference
implementation's version has.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import mlx.core as mx

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = ["LinearDecay", "SignSGD", "sign_update", "total_excursion"]

Params = dict[str, mx.array]


class LinearDecay:
    """A learning rate falling linearly from an initial value to exactly zero.

    Reaching exactly zero is the point rather than an artifact: it makes total
    travel finite and turns the trajectory into a coarse-to-fine search that
    terminates rather than converges.
    """

    __slots__ = ("initial_lr", "total_steps")

    def __init__(self, initial_lr: float, total_steps: int) -> None:
        """Create a schedule.

        Raises:
            ValueError: If ``total_steps`` is not positive.
        """
        if total_steps <= 0:
            msg = f"total_steps must be positive, got {total_steps}"
            raise ValueError(msg)
        self.initial_lr = initial_lr
        self.total_steps = total_steps

    def __call__(self, step: int) -> float:
        """Return the rate at ``step``, clamped to zero past the budget."""
        if step >= self.total_steps:
            return 0.0
        return self.initial_lr * (1.0 - step / self.total_steps)


def total_excursion(initial_lr: float, total_steps: int) -> float:
    """Total distance a parameter can travel under this schedule.

    Exposed because it is the quantity that constrains the method, so a change
    to the schedule can be checked against its effect on the budget rather than
    discovered later.
    """
    schedule = LinearDecay(initial_lr, total_steps)
    return sum(schedule(step) for step in range(total_steps))


def sign_update(param: mx.array, grad: mx.array, lr: float) -> mx.array:
    """Apply one signed gradient step: ``param - lr * sign(grad)``.

    Every parameter moves by exactly ``lr`` regardless of the gradient's
    magnitude. Two consequences matter elsewhere: the update is invariant to any
    positive rescaling of the loss, so loss scaling is a no-op and MRound omits
    it; and a parameter with exactly zero gradient does not move, since
    ``sign(0) == 0``.
    """
    return param - lr * mx.sign(grad)


class SignSGD:
    """Signed gradient descent over a flat mapping of named arrays."""

    __slots__ = ("_buffers", "_step", "learning_rate", "momentum")

    def __init__(
        self,
        learning_rate: float | LinearDecay,
        *,
        momentum: float = 0.0,
    ) -> None:
        """Create an optimizer.

        Args:
            learning_rate: A constant rate, or a schedule called with the step.
            momentum: Exponential moving average factor on the gradient. Zero
                gives pure signed gradient descent, which is the default and
                what the published method uses.
        """
        self.learning_rate = learning_rate
        self.momentum = momentum
        self._step = 0
        self._buffers: Params = {}

    @property
    def step_count(self) -> int:
        """Steps applied so far, which drives the schedule."""
        return self._step

    def current_lr(self) -> float:
        """The rate the next :meth:`apply` will use."""
        if isinstance(self.learning_rate, LinearDecay):
            return self.learning_rate(self._step)
        return self.learning_rate

    def apply(self, params: Mapping[str, mx.array], grads: Mapping[str, mx.array]) -> Params:
        """Return updated parameters. The inputs are not modified.

        Raises:
            KeyError: If a parameter has no matching gradient.
        """
        lr = self.current_lr()
        updated: Params = {}
        for name, value in params.items():
            if name not in grads:
                msg = f"no gradient supplied for parameter {name!r}"
                raise KeyError(msg)
            grad = grads[name]
            if self.momentum:
                buffer = self._buffers.get(name)
                buffer = grad if buffer is None else self.momentum * buffer + grad
                self._buffers[name] = buffer
                grad = buffer
            updated[name] = sign_update(value, grad, lr)
        self._step += 1
        return updated
