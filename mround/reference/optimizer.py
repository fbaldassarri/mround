# Copyright 2026 The MRound Authors
# SPDX-License-Identifier: Apache-2.0
"""Signed gradient descent, in NumPy.

Implements DOCUMENTATION.md sections 1.4 and 5.4. The whole method is one line
of arithmetic; everything else in this module is the schedule and the bookkeeping
that makes the schedule reproducible.

The schedule matters more than it looks. Because every step moves a parameter by
exactly the current rate and the rate decays linearly to zero, the total distance
a parameter can travel is finite and known: about ``lr * iters / 2``. With the
default ``lr = c / iters`` that is ``c / 2``, independent of the step count. The
learning rate and the iteration count are therefore coupled, and tuning one
without the other changes how far rounding decisions can move.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

__all__ = ["LinearDecay", "SignSGD", "sign_update", "total_excursion"]

Array = npt.NDArray[np.float64]


class LinearDecay:
    """A learning rate falling linearly from an initial value to exactly zero.

    Reaching exactly zero is the point, not an artifact: it is what makes the
    total travel finite and turns the trajectory into a coarse-to-fine search
    that terminates rather than converges.
    """

    __slots__ = ("initial_lr", "total_steps")

    def __init__(self, initial_lr: float, total_steps: int) -> None:
        """Create a schedule.

        Args:
            initial_lr: Rate at step zero.
            total_steps: Steps over which the rate reaches zero.

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

    Exposed because it is the quantity that actually constrains the method, and
    because a reimplementation that changes the schedule should be able to check
    what it did to the budget rather than discovering it later.
    """
    schedule = LinearDecay(initial_lr, total_steps)
    return sum(schedule(step) for step in range(total_steps))


def sign_update(param: Array, grad: Array, lr: float) -> Array:
    """Apply one signed gradient step.

    The entire method: ``param - lr * sign(grad)``. Every parameter moves by
    exactly ``lr``, in the direction the gradient indicates, regardless of the
    gradient's magnitude.

    Two consequences follow and both are load-bearing elsewhere. The update is
    invariant to any positive rescaling of the loss, so loss scaling is a no-op
    here (it exists only to prevent gradient underflow in reduced precision).
    And a parameter with exactly zero gradient does not move at all, since
    ``sign(0) == 0``.
    """
    return param - lr * np.sign(grad)


class SignSGD:
    """Signed gradient descent over a flat collection of named arrays.

    Parameters are transformed functionally rather than mutated by the optimizer
    itself, which mirrors the MLX convention and keeps the port mechanical.
    Momentum is supported but defaults to zero, which makes the update pure
    ``sign(grad)``; with momentum the sign is taken of the accumulator instead.
    """

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
        self._buffers: dict[str, Array] = {}

    @property
    def step_count(self) -> int:
        """Steps applied so far, which drives the schedule."""
        return self._step

    def current_lr(self) -> float:
        """The rate that the next call to :meth:`apply` will use."""
        if isinstance(self.learning_rate, LinearDecay):
            return self.learning_rate(self._step)
        return self.learning_rate

    def apply(self, params: dict[str, Array], grads: dict[str, Array]) -> dict[str, Array]:
        """Return updated parameters. The inputs are not modified.

        Args:
            params: Named parameter arrays.
            grads: Named gradient arrays, matching ``params`` in keys and shapes.

        Returns:
            A new dictionary of updated arrays.

        Raises:
            KeyError: If a parameter has no matching gradient.
        """
        lr = self.current_lr()
        updated: dict[str, Array] = {}
        for name, value in params.items():
            if name not in grads:
                msg = f"no gradient supplied for parameter {name!r}"
                raise KeyError(msg)
            grad = grads[name]
            if self.momentum:
                buffer = self._buffers.get(name)
                buffer = grad.copy() if buffer is None else self.momentum * buffer + grad
                self._buffers[name] = buffer
                grad = buffer
            updated[name] = sign_update(value, grad, lr)
        self._step += 1
        return updated
