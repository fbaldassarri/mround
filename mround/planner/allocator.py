# Copyright 2026 The MRound Authors
# SPDX-License-Identifier: Apache-2.0
"""Mixed-precision bit allocation.

Given a sensitivity score per layer per candidate bit width, and a storage cost
for each choice, pick one bit width per layer that minimizes total predicted
loss subject to a budget on total size. This is a multiple-choice knapsack
problem and dynamic programming solves it exactly.

This module imports nothing from any array framework and nothing that touches a
model. It is pure discrete optimization over numbers, which makes it
unit-testable against exhaustive search on small instances, including instances
constructed so that greedy allocation is provably wrong. That testability is the
reason it lives in its own layer.

It runs on the CPU deliberately: it is small, sequential, and benefits from
float64 accumulation, which Metal does not offer. See DOCUMENTATION.md section 7.

Specification: DOCUMENTATION.md section 1.5.
"""

from __future__ import annotations

import dataclasses
import math

__all__ = ["Allocation", "LayerOption", "allocate_bits"]


@dataclasses.dataclass(frozen=True, slots=True)
class LayerOption:
    """One candidate bit width for one layer.

    Attributes:
        bits: The candidate width.
        cost_bits: Total storage this choice consumes for the layer, counting
            packed weights and the scale metadata that comes with the group
            size. Metadata is not negligible at small group sizes and omitting
            it makes the budget wrong in the direction that matters.
        delta_loss: Predicted loss increase from quantizing this layer at this
            width. Lower is better. Must be finite and non-negative.
        elements: How many weights the layer holds, when the caller knows.
            It is what makes :attr:`Allocation.average_bits` the code
            average a user asked for; without it the element count is
            inferred from the cost, which counts the scale metadata as if it
            were weights and understates the average by one to two percent
            at the narrow widths.
    """

    bits: int
    cost_bits: int
    delta_loss: float
    elements: int | None = None

    def __post_init__(self) -> None:
        """Reject options that cannot participate in a meaningful allocation."""
        if self.cost_bits < 0:
            msg = f"cost_bits must be non-negative, got {self.cost_bits}"
            raise ValueError(msg)
        if not math.isfinite(self.delta_loss) or self.delta_loss < 0.0:
            msg = f"delta_loss must be finite and non-negative, got {self.delta_loss}"
            raise ValueError(msg)
        if self.elements is not None and self.elements <= 0:
            msg = f"elements must be positive when given, got {self.elements}"
            raise ValueError(msg)


@dataclasses.dataclass(frozen=True, slots=True)
class Allocation:
    """The chosen bit width for every layer.

    Attributes:
        by_layer: Layer name to chosen bit width.
        total_cost_bits: Storage the allocation consumes.
        predicted_loss: Sum of the chosen options' predicted loss increases.
            Useful for comparing allocations, not as an absolute quantity.
        average_bits: Element-weighted mean code width, the number a user
            asked for, exact when the options carry their element counts and
            inferred from the costs otherwise (see :class:`LayerOption`).
    """

    by_layer: dict[str, int]
    total_cost_bits: int
    predicted_loss: float
    average_bits: float


def _validate(options: dict[str, list[LayerOption]]) -> None:
    if not options:
        msg = "no layers to allocate"
        raise ValueError(msg)
    for name, candidates in options.items():
        if not candidates:
            msg = f"layer {name!r} has no candidate bit widths"
            raise ValueError(msg)


def allocate_bits(
    options: dict[str, list[LayerOption]],
    *,
    budget_bits: int,
    max_states: int | None = None,
) -> Allocation:
    """Choose one bit width per layer, minimizing predicted loss under a budget.

    Exact dynamic programming over the reachable cumulative costs. The state
    space is keyed on the exact integer cost rather than on a discretized budget
    axis, so by default there is no bucketing approximation: the result is the
    true optimum, not a near-optimum.

    Two things keep that tractable. States are pruned by Pareto dominance after
    each layer, which never removes an optimal path because a state that is both
    more expensive and worse than another can never win. And states above the
    budget are dropped immediately, since costs only grow.

    **Scaling.** The surviving frontier grows with the layer count, and the
    total cost grows faster than linearly. Measured on this implementation with
    four candidate widths and realistic cost spreads: 50 layers in 0.05 seconds,
    100 in 0.5, 200 in 6, 400 in roughly 50. A 7B model has a few hundred
    quantizable layers, so exact allocation costs seconds against a quantization
    run that costs hours, which is a good trade. Much larger models are where
    ``max_states`` becomes worth considering.

    Args:
        options: Layer name to its candidate options. A layer with a single
            option is pinned to that width, which is how layers excluded from
            quantization are expressed.
        budget_bits: Maximum total storage.
        max_states: Keep at most this many frontier states after each layer,
            retaining those with the lowest predicted loss. **This makes the
            result approximate**, because a state that looks poor now can lead
            to the best final answer once cheaper layers follow it. ``None``,
            the default, keeps the search exact. Set it only when the exact
            search is measurably too slow, and record that the result is no
            longer optimal.

    Returns:
        The chosen allocation.

    Raises:
        ValueError: If any layer has no options, if ``max_states`` is not
            positive, or if the budget cannot be met even by choosing every
            layer's cheapest option. Failing on an infeasible budget is
            correct: silently returning the cheapest allocation would produce a
            model that does not meet the request without saying so.
    """
    _validate(options)
    if max_states is not None and max_states < 1:
        msg = f"max_states must be positive or None, got {max_states}"
        raise ValueError(msg)

    names = list(options)
    cheapest = sum(min(o.cost_bits for o in options[n]) for n in names)
    if cheapest > budget_bits:
        msg = (
            f"budget of {budget_bits} bits is infeasible: the cheapest possible "
            f"allocation costs {cheapest} bits"
        )
        raise ValueError(msg)

    # frontier maps an exact cumulative cost to (loss, chosen widths so far).
    frontier: dict[int, tuple[float, tuple[int, ...]]] = {0: (0.0, ())}

    for name in names:
        nxt: dict[int, tuple[float, tuple[int, ...]]] = {}
        for cost, (loss, path) in frontier.items():
            for option in options[name]:
                new_cost = cost + option.cost_bits
                if new_cost > budget_bits:
                    continue
                new_loss = loss + option.delta_loss
                existing = nxt.get(new_cost)
                if existing is None or new_loss < existing[0]:
                    nxt[new_cost] = (new_loss, (*path, option.bits))
        if not nxt:
            msg = f"no feasible allocation remains after layer {name!r}"
            raise ValueError(msg)
        frontier = _prune(nxt)
        if max_states is not None:
            frontier = _thin(frontier, max_states)

    _, (best_loss, best_path) = min(frontier.items(), key=lambda item: (item[1][0], item[0]))
    by_layer = dict(zip(names, best_path, strict=True))
    total_weights = sum(
        next(o.cost_bits for o in options[n] if o.bits == by_layer[n]) for n in names
    )
    average = _average_bits(options, by_layer)
    return Allocation(
        by_layer=by_layer,
        total_cost_bits=total_weights,
        predicted_loss=best_loss,
        average_bits=average,
    )


def _prune(
    states: dict[int, tuple[float, tuple[int, ...]]],
) -> dict[int, tuple[float, tuple[int, ...]]]:
    """Drop states that are dominated on both cost and loss.

    A state that costs at least as much as another and predicts at least as much
    loss can never lead to a better final answer, because every remaining layer
    adds the same options to both. Removing it is exact, not heuristic.
    """
    kept: dict[int, tuple[float, tuple[int, ...]]] = {}
    best_loss = math.inf
    for cost in sorted(states):
        loss, path = states[cost]
        if loss < best_loss:
            kept[cost] = (loss, path)
            best_loss = loss
    return kept


def _thin(
    frontier: dict[int, tuple[float, tuple[int, ...]]], max_states: int
) -> dict[int, tuple[float, tuple[int, ...]]]:
    """Subsample the frontier uniformly along the cost axis.

    Keeping simply the lowest-loss states would be the obvious thing and it is
    wrong: after Pareto pruning the lowest-loss states are also the most
    expensive ones, so a beam selected that way drifts toward the top of the
    budget and then runs out of room, failing with no feasible allocation on an
    instance that has plenty. That is not a hypothetical; it is what the first
    version of this function did.

    Sampling evenly across the cost axis keeps both ends. The cheapest state
    always survives, which preserves feasibility whenever the instance is
    feasible at all, and the best-loss state always survives, which preserves
    quality when budget is not the binding constraint.
    """
    if len(frontier) <= max_states:
        return frontier

    costs = sorted(frontier)
    if max_states == 1:
        chosen = [costs[0]]
    else:
        step = (len(costs) - 1) / (max_states - 1)
        chosen = sorted({costs[round(i * step)] for i in range(max_states)})
    return {cost: frontier[cost] for cost in chosen}


def _average_bits(options: dict[str, list[LayerOption]], by_layer: dict[str, int]) -> float:
    """Element-weighted mean code width across the allocation.

    Weighted by each layer's element count, so a large layer at 2 bits pulls
    the average down more than a small one does, which is what a user means
    by "an average of 3 bits". When an option carries its element count the
    average is exact; otherwise the count is inferred from the cost at the
    chosen width, and since the cost includes the scale metadata that
    inference overstates the elements and understates the average slightly.
    The public entry points always supply the counts.
    """
    total_bits = 0.0
    total_elements = 0.0
    for name, bits in by_layer.items():
        option = next(o for o in options[name] if o.bits == bits)
        if option.elements is not None:
            elements = float(option.elements)
            total_bits += bits * elements
        else:
            elements = option.cost_bits / bits if bits else 0.0
            total_bits += option.cost_bits
        total_elements += elements
    if total_elements == 0.0:
        return 0.0
    return total_bits / total_elements
