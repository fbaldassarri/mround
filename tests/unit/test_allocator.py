# Copyright 2026 The MRound Authors
# SPDX-License-Identifier: Apache-2.0
"""Tests for mixed-precision bit allocation.

The allocator claims to find the exact optimum, not a good approximation, so the
central test is exhaustive search: enumerate every possible assignment on small
instances and check the allocator finds the best one. That is a complete proof
on those instances rather than a spot check.

Instances where greedy allocation is provably wrong get their own tests, because
"greedy happens to agree" is the failure mode that would let a broken allocator
pass a naive suite.
"""

from __future__ import annotations

import dataclasses
import itertools
import math

import numpy as np
import pytest

from mround.planner.allocator import Allocation, LayerOption, allocate_bits


def brute_force(
    options: dict[str, list[LayerOption]], budget_bits: int
) -> tuple[float, dict[str, int]] | None:
    """Exhaustively find the true optimum, for comparison."""
    names = list(options)
    best: tuple[float, dict[str, int]] | None = None
    for combo in itertools.product(*(options[n] for n in names)):
        cost = sum(o.cost_bits for o in combo)
        if cost > budget_bits:
            continue
        loss = sum(o.delta_loss for o in combo)
        if best is None or loss < best[0]:
            best = (loss, {n: o.bits for n, o in zip(names, combo, strict=True)})
    return best


def greedy(options: dict[str, list[LayerOption]], budget_bits: int) -> dict[str, int] | None:
    """Start at the cheapest option everywhere, then buy the best upgrades.

    A reasonable-looking heuristic, included so tests can assert the allocator
    beats it on instances designed to defeat it.
    """
    chosen = {n: min(options[n], key=lambda o: o.cost_bits) for n in options}
    spent = sum(o.cost_bits for o in chosen.values())
    if spent > budget_bits:
        return None
    while True:
        best_upgrade = None
        for name, current in chosen.items():
            for option in options[name]:
                extra = option.cost_bits - current.cost_bits
                gain = current.delta_loss - option.delta_loss
                if extra <= 0 or gain <= 0 or spent + extra > budget_bits:
                    continue
                ratio = gain / extra
                if best_upgrade is None or ratio > best_upgrade[0]:
                    best_upgrade = (ratio, name, option, extra)
        if best_upgrade is None:
            break
        _, name, option, extra = best_upgrade
        chosen[name] = option
        spent += extra
    return {n: o.bits for n, o in chosen.items()}


def random_instance(
    seed: int, n_layers: int, widths: tuple[int, ...] = (2, 4, 8)
) -> dict[str, list[LayerOption]]:
    gen = np.random.default_rng(seed)
    options: dict[str, list[LayerOption]] = {}
    for i in range(n_layers):
        elements = int(gen.integers(64, 512))
        options[f"layer.{i}"] = [
            LayerOption(
                bits=b,
                cost_bits=elements * b,
                delta_loss=float(gen.uniform(0.1, 10.0)) / b,
            )
            for b in widths
        ]
    return options


class TestExactness:
    @pytest.mark.parametrize("seed", range(12))
    def test_matches_exhaustive_search(self, seed: int) -> None:
        options = random_instance(seed, n_layers=5)
        cheapest = sum(min(o.cost_bits for o in v) for v in options.values())
        priciest = sum(max(o.cost_bits for o in v) for v in options.values())
        budget = (cheapest + priciest) // 2

        expected = brute_force(options, budget)
        assert expected is not None
        result = allocate_bits(options, budget_bits=budget)
        assert result.predicted_loss == pytest.approx(expected[0], rel=1e-12)

    @pytest.mark.parametrize("n_layers", [1, 2, 3, 6])
    def test_matches_exhaustive_search_across_sizes(self, n_layers: int) -> None:
        options = random_instance(seed=99, n_layers=n_layers)
        cheapest = sum(min(o.cost_bits for o in v) for v in options.values())
        budget = int(cheapest * 1.6)
        expected = brute_force(options, budget)
        assert expected is not None
        result = allocate_bits(options, budget_bits=budget)
        assert result.predicted_loss == pytest.approx(expected[0], rel=1e-12)

    @pytest.mark.parametrize("seed", range(6))
    def test_respects_the_budget(self, seed: int) -> None:
        options = random_instance(seed, n_layers=6)
        cheapest = sum(min(o.cost_bits for o in v) for v in options.values())
        budget = int(cheapest * 1.4)
        result = allocate_bits(options, budget_bits=budget)
        assert result.total_cost_bits <= budget


class TestGreedyIsNotEnough:
    def test_beats_greedy_on_a_constructed_instance(self) -> None:
        # Two layers, budget 15. Greedy buys the best gain-per-bit upgrade first
        # (layer b, 1.5 loss per bit) which costs 1 bit and then leaves too
        # little for layer a's much larger win. Taking a's upgrade instead and
        # leaving b alone costs exactly the budget and is strictly better.
        #
        #   greedy: a=2 (10.0) + b=8 (8.5) = 18.5 at cost 11
        #   exact:  a=4 ( 3.0) + b=2 (10.0) = 13.0 at cost 15
        options = {
            "a": [
                LayerOption(bits=2, cost_bits=5, delta_loss=10.0),
                LayerOption(bits=4, cost_bits=10, delta_loss=3.0),
            ],
            "b": [
                LayerOption(bits=2, cost_bits=5, delta_loss=10.0),
                LayerOption(bits=8, cost_bits=6, delta_loss=8.5),
            ],
        }
        budget = 15

        exact = brute_force(options, budget)
        assert exact is not None
        result = allocate_bits(options, budget_bits=budget)
        assert result.predicted_loss == pytest.approx(exact[0])

        greedy_choice = greedy(options, budget)
        assert greedy_choice is not None
        greedy_loss = sum(
            next(o.delta_loss for o in options[n] if o.bits == b) for n, b in greedy_choice.items()
        )
        assert result.predicted_loss < greedy_loss, (
            "instance failed to defeat greedy, so it proves nothing"
        )

    @pytest.mark.parametrize("seed", range(20))
    def test_never_worse_than_greedy(self, seed: int) -> None:
        options = random_instance(seed, n_layers=5)
        cheapest = sum(min(o.cost_bits for o in v) for v in options.values())
        budget = int(cheapest * 1.5)

        result = allocate_bits(options, budget_bits=budget)
        greedy_choice = greedy(options, budget)
        assert greedy_choice is not None
        greedy_loss = sum(
            next(o.delta_loss for o in options[n] if o.bits == b) for n, b in greedy_choice.items()
        )
        assert result.predicted_loss <= greedy_loss + 1e-12


class TestBudgetBehaviour:
    def test_a_generous_budget_buys_the_best_option_everywhere(self) -> None:
        options = random_instance(seed=5, n_layers=4)
        priciest = sum(max(o.cost_bits for o in v) for v in options.values())
        result = allocate_bits(options, budget_bits=priciest)
        for name, bits in result.by_layer.items():
            best = min(options[name], key=lambda o: o.delta_loss)
            assert bits == best.bits

    def test_a_tight_budget_forces_the_cheapest_everywhere(self) -> None:
        options = random_instance(seed=6, n_layers=4)
        cheapest = sum(min(o.cost_bits for o in v) for v in options.values())
        result = allocate_bits(options, budget_bits=cheapest)
        assert result.total_cost_bits == cheapest

    def test_an_infeasible_budget_raises_rather_than_approximating(self) -> None:
        # Silently returning the cheapest allocation would produce a model that
        # does not meet the request without saying so.
        options = random_instance(seed=7, n_layers=3)
        cheapest = sum(min(o.cost_bits for o in v) for v in options.values())
        with pytest.raises(ValueError, match="infeasible"):
            allocate_bits(options, budget_bits=cheapest - 1)

    def test_more_budget_never_produces_a_worse_allocation(self) -> None:
        options = random_instance(seed=8, n_layers=5)
        cheapest = sum(min(o.cost_bits for o in v) for v in options.values())
        losses = [
            allocate_bits(options, budget_bits=int(cheapest * f)).predicted_loss
            for f in (1.0, 1.2, 1.5, 2.0)
        ]
        assert all(a >= b - 1e-12 for a, b in itertools.pairwise(losses))


class TestScaleInvariance:
    """Multiplying every score by one positive constant changes nothing.

    This is not an idle algebraic property. Scores accumulate per batch, so
    splitting a scoring draw into more, smaller batches, which is what a longer
    sequence forces on a fixed memory budget
    (:func:`mround.planner.sensitivity.scoring_batch_size`), raises every score
    by roughly the same factor. The claim that the allocation survives that is
    load bearing, so it is pinned here rather than argued in a docstring.
    """

    @pytest.mark.parametrize("seed", range(6))
    @pytest.mark.parametrize("factor", [0.25, 1.7, 4.0, 32.0, 1.0 / 3.0, 7.77e5])
    def test_a_positive_rescaling_leaves_the_choice_alone(self, seed: int, factor: float) -> None:
        options = random_instance(seed, 14)
        budget = int(0.6 * sum(max(o.cost_bits for o in per) for per in options.values()))
        scaled = {
            name: [dataclasses.replace(o, delta_loss=o.delta_loss * factor) for o in per]
            for name, per in options.items()
        }
        plain = allocate_bits(options, budget_bits=budget)
        rescaled = allocate_bits(scaled, budget_bits=budget)
        assert rescaled.by_layer == plain.by_layer
        # The reported loss does move, by exactly the factor, which is why a
        # predicted_loss is comparable only against runs batched the same way.
        assert rescaled.predicted_loss == pytest.approx(plain.predicted_loss * factor)


class TestPinnedLayers:
    def test_a_single_option_pins_the_layer(self) -> None:
        # How a layer excluded from quantization is expressed.
        options = {
            "keep_fp16": [LayerOption(bits=16, cost_bits=1600, delta_loss=0.0)],
            "free": [
                LayerOption(bits=2, cost_bits=200, delta_loss=5.0),
                LayerOption(bits=4, cost_bits=400, delta_loss=1.0),
            ],
        }
        result = allocate_bits(options, budget_bits=2000)
        assert result.by_layer["keep_fp16"] == 16
        assert result.by_layer["free"] == 4


class TestReporting:
    def test_average_bits_is_element_weighted(self) -> None:
        # A large layer at 2 bits must pull the average down further than a
        # small one does, which is what a user means by "an average of N bits".
        options = {
            "big": [LayerOption(bits=2, cost_bits=2000, delta_loss=1.0)],
            "small": [LayerOption(bits=8, cost_bits=80, delta_loss=0.0)],
        }
        result = allocate_bits(options, budget_bits=10_000)
        # 1000 elements at 2 bits, 10 elements at 8 bits.
        expected = (2000 + 80) / (1000 + 10)
        assert result.average_bits == pytest.approx(expected)
        assert 2.0 < result.average_bits < 2.1

    def test_average_bits_is_exact_when_the_options_carry_their_elements(self) -> None:
        # With the scale metadata in the cost, inferring the element count from
        # cost / bits overstates it and understates the average: two equal
        # 256 x 256 layers at 2 and 4 bits, group 64, 16 bit scales and
        # biases, came out at 2.947 instead of 3.000. The element count makes
        # it the code average the user asked for, and api.py's
        # achieved_code_bits agrees with it.
        elements = 256 * 256
        metadata = 2 * 256 * (256 // 64) * 16
        options = {
            "a": [
                LayerOption(
                    bits=2, cost_bits=elements * 2 + metadata, delta_loss=1.0, elements=elements
                )
            ],
            "b": [
                LayerOption(
                    bits=4, cost_bits=elements * 4 + metadata, delta_loss=0.0, elements=elements
                )
            ],
        }
        result = allocate_bits(options, budget_bits=10 * elements)
        assert result.average_bits == pytest.approx(3.0)
        without = {
            name: [dataclasses.replace(option, elements=None) for option in choices]
            for name, choices in options.items()
        }
        assert allocate_bits(without, budget_bits=10 * elements).average_bits < 3.0

    def test_returns_an_allocation_for_every_layer(self) -> None:
        options = random_instance(seed=9, n_layers=7)
        cheapest = sum(min(o.cost_bits for o in v) for v in options.values())
        result = allocate_bits(options, budget_bits=int(cheapest * 1.3))
        assert isinstance(result, Allocation)
        assert set(result.by_layer) == set(options)


class TestValidation:
    def test_empty_input_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="no layers"):
            allocate_bits({}, budget_bits=100)

    def test_a_layer_with_no_options_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="no candidate bit widths"):
            allocate_bits({"a": []}, budget_bits=100)

    def test_negative_cost_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="cost_bits must be non-negative"):
            LayerOption(bits=4, cost_bits=-1, delta_loss=1.0)

    @pytest.mark.parametrize("bad", [-1.0, math.nan, math.inf])
    def test_invalid_loss_is_rejected(self, bad: float) -> None:
        with pytest.raises(ValueError, match="delta_loss must be finite"):
            LayerOption(bits=4, cost_bits=100, delta_loss=bad)


class TestBeamSearch:
    """The optional approximate mode, for layer counts where exact is too slow."""

    def test_beam_matches_exact_on_small_instances(self) -> None:
        # A generous beam should reproduce the exact answer, which is the check
        # that the beam machinery itself is not broken.
        options = random_instance(seed=20, n_layers=5)
        cheapest = sum(min(o.cost_bits for o in v) for v in options.values())
        budget = int(cheapest * 1.5)
        exact = allocate_bits(options, budget_bits=budget)
        beamed = allocate_bits(options, budget_bits=budget, max_states=10_000)
        assert beamed.predicted_loss == pytest.approx(exact.predicted_loss)

    def test_beam_is_never_better_than_exact(self) -> None:
        # It is an approximation, so it can only tie or lose. If it ever won,
        # the exact search would be wrong.
        for seed in range(8):
            options = random_instance(seed=seed, n_layers=6)
            cheapest = sum(min(o.cost_bits for o in v) for v in options.values())
            budget = int(cheapest * 1.4)
            exact = allocate_bits(options, budget_bits=budget)
            beamed = allocate_bits(options, budget_bits=budget, max_states=3)
            assert beamed.predicted_loss >= exact.predicted_loss - 1e-12

    def test_beam_still_respects_the_budget(self) -> None:
        options = random_instance(seed=21, n_layers=20)
        cheapest = sum(min(o.cost_bits for o in v) for v in options.values())
        budget = int(cheapest * 1.4)
        result = allocate_bits(options, budget_bits=budget, max_states=5)
        assert result.total_cost_bits <= budget

    def test_non_positive_beam_is_rejected(self) -> None:
        options = random_instance(seed=22, n_layers=2)
        with pytest.raises(ValueError, match="max_states must be positive"):
            allocate_bits(options, budget_bits=10**9, max_states=0)


class TestScale:
    def test_stays_fast_at_a_small_model_scale(self) -> None:
        # Fast enough to keep in the default suite.
        options = random_instance(seed=10, n_layers=60, widths=(2, 3, 4, 8))
        cheapest = sum(min(o.cost_bits for o in v) for v in options.values())
        result = allocate_bits(options, budget_bits=int(cheapest * 1.5))
        assert len(result.by_layer) == 60

    def test_beam_handles_a_large_model_scale_quickly(self) -> None:
        # Exact allocation at 400 layers takes roughly a minute on this
        # implementation, which is fine once per quantization run but far too
        # slow for a test suite. The beam is what makes that size cheap.
        options = random_instance(seed=11, n_layers=400, widths=(2, 3, 4, 8))
        cheapest = sum(min(o.cost_bits for o in v) for v in options.values())
        budget = int(cheapest * 1.5)
        result = allocate_bits(options, budget_bits=budget, max_states=200)
        assert len(result.by_layer) == 400
        assert result.total_cost_bits <= budget

    @pytest.mark.slow
    def test_exact_handles_a_realistic_layer_count(self) -> None:
        # A 7B model has a few hundred quantizable linear layers. Marked slow
        # because it takes seconds, but kept because the scaling claim in the
        # allocator's docstring should be backed by something that runs.
        options = random_instance(seed=12, n_layers=250, widths=(2, 3, 4, 8))
        cheapest = sum(min(o.cost_bits for o in v) for v in options.values())
        result = allocate_bits(options, budget_bits=int(cheapest * 1.5))
        assert len(result.by_layer) == 250
