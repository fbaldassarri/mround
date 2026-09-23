# Copyright 2026 The MRound Authors
# SPDX-License-Identifier: Apache-2.0
"""Tests for signed gradient descent, its schedule, and the losses."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from mround.reference.losses import outlier_suppressed_loss, reconstruction_loss
from mround.reference.optimizer import (
    LinearDecay,
    SignSGD,
    sign_update,
    total_excursion,
)
from mround.reference.quantize import V_BOUND
from mround.schemes import TuningConfig


class TestSignUpdate:
    def test_every_parameter_moves_by_exactly_the_rate(self) -> None:
        param = np.zeros(5)
        grad = np.array([0.001, -1e9, 3.0, -0.5, 7.0])
        out = sign_update(param, grad, 0.01)
        # Magnitudes differ by twelve orders of magnitude; the step does not.
        assert np.allclose(np.abs(out), 0.01)

    def test_direction_follows_the_sign(self) -> None:
        out = sign_update(np.zeros(2), np.array([5.0, -5.0]), 0.1)
        assert out[0] < 0
        assert out[1] > 0

    def test_zero_gradient_does_not_move(self) -> None:
        out = sign_update(np.array([3.0]), np.array([0.0]), 0.1)
        assert out[0] == 3.0

    @pytest.mark.parametrize("factor", [1e-6, 0.5, 1.0, 1000.0, 1e9])
    def test_invariant_to_positive_loss_rescaling(self, factor: float) -> None:
        # The property that makes loss scaling a no-op. It is why the reference
        # implementation's factor of 1000 changes nothing, and why MRound omits
        # it entirely while accumulating in wide precision.
        param = np.array([1.0, -2.0, 0.5])
        grad = np.array([0.3, -0.7, 0.0])
        base = sign_update(param, grad, 0.1)
        scaled = sign_update(param, grad * factor, 0.1)
        assert np.array_equal(base, scaled)


class TestLinearDecay:
    def test_starts_at_the_initial_rate(self) -> None:
        assert LinearDecay(0.005, 200)(0) == pytest.approx(0.005)

    def test_reaches_zero_at_the_budget(self) -> None:
        schedule = LinearDecay(0.005, 200)
        assert schedule(200) == 0.0
        assert schedule(1000) == 0.0

    def test_is_monotonically_decreasing(self) -> None:
        schedule = LinearDecay(0.01, 50)
        values = [schedule(i) for i in range(50)]
        assert all(a > b for a, b in itertools.pairwise(values))

    def test_non_positive_budget_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="total_steps must be positive"):
            LinearDecay(0.01, 0)


class TestExcursionBudget:
    """The property that makes the schedule the real constraint on rounding.

    DOCUMENTATION.md section 5.4 and MEMORY.md D-010.
    """

    @pytest.mark.parametrize("iters", [50, 200, 500, 1000, 4000])
    def test_budget_is_half_the_constant_regardless_of_step_count(self, iters: int) -> None:
        # The exact sum of a linear decay from c/T to zero over T steps is
        # c/2 * (1 + 1/T): half the constant plus one half step, which is why
        # the budget is "about" half regardless of the step count. Pinned to
        # the closed form rather than to 0.5 within 0.01, because at T = 50 the
        # exact value is 0.51 and a tolerance of 0.01 passed only through the
        # rounding of a naive float sum.
        config = TuningConfig(iters=iters)
        expected = 0.5 * (1 + 1 / iters)
        assert total_excursion(config.resolved_lr(4), iters) == pytest.approx(expected, rel=1e-9)

    @pytest.mark.parametrize("iters", [200, 1000])
    def test_low_bit_widths_get_twice_the_budget(self, iters: int) -> None:
        config = TuningConfig(iters=iters)
        low = total_excursion(config.resolved_lr(2), iters)
        high = total_excursion(config.resolved_lr(4), iters)
        assert low == pytest.approx(2 * high, rel=1e-9)

    def test_the_four_bit_budget_equals_the_half_code_bound(self) -> None:
        # The reason the coupling exists: at 4 bits and above, total travel is
        # exactly the width of the rounding perturbation's permitted range in
        # one direction. The schedule and the bound are the same statement.
        config = TuningConfig(iters=200)
        excursion = total_excursion(config.resolved_lr(4), 200)
        assert excursion == pytest.approx(V_BOUND * (1 + 1 / 200), rel=1e-9)
        assert abs(excursion - V_BOUND) < 0.01

    def test_changing_iterations_without_the_rate_changes_the_budget(self) -> None:
        # The failure mode the coupling guards against: an implementation that
        # exposes iters and lr as independent knobs silently changes the method
        # when either is tuned alone.
        fixed_lr = 0.005
        assert total_excursion(fixed_lr, 100) < total_excursion(fixed_lr, 400)


class TestSignSGD:
    def test_applies_the_schedule(self) -> None:
        opt = SignSGD(LinearDecay(0.1, 10))
        params = {"v": np.zeros(3)}
        grads = {"v": np.ones(3)}
        first = opt.apply(params, grads)
        second = opt.apply(first, grads)
        # Each step is smaller than the last.
        assert abs(first["v"][0]) > abs(second["v"][0] - first["v"][0])

    def test_does_not_mutate_its_inputs(self) -> None:
        opt = SignSGD(0.1)
        params = {"v": np.zeros(3)}
        opt.apply(params, {"v": np.ones(3)})
        assert np.array_equal(params["v"], np.zeros(3))

    def test_step_count_advances(self) -> None:
        opt = SignSGD(0.1)
        assert opt.step_count == 0
        opt.apply({"v": np.zeros(1)}, {"v": np.ones(1)})
        assert opt.step_count == 1

    def test_missing_gradient_is_an_error(self) -> None:
        opt = SignSGD(0.1)
        with pytest.raises(KeyError, match="no gradient supplied"):
            opt.apply({"v": np.zeros(1), "alpha": np.ones(1)}, {"v": np.ones(1)})

    def test_momentum_takes_the_sign_of_the_accumulator(self) -> None:
        # With momentum the recent history can outvote the current gradient,
        # which pure signed descent cannot do.
        opt = SignSGD(0.1, momentum=0.9)
        params = {"v": np.zeros(1)}
        for _ in range(5):
            params = opt.apply(params, {"v": np.array([1.0])})
        # One small opposing gradient should not flip the direction.
        moved = opt.apply(params, {"v": np.array([-0.01])})
        assert moved["v"][0] < params["v"][0]

    def test_default_has_no_momentum(self) -> None:
        assert SignSGD(0.1).momentum == 0.0


class TestReconstructionLoss:
    def test_zero_when_identical(self) -> None:
        x = np.random.default_rng(0).normal(size=(4, 8))
        loss, grad = reconstruction_loss(x, x)
        assert loss == 0.0
        assert np.allclose(grad, 0.0)

    def test_gradient_matches_finite_differences(self) -> None:
        gen = np.random.default_rng(1)
        pred = gen.normal(size=(3, 5))
        ref = gen.normal(size=(3, 5))
        _, grad = reconstruction_loss(pred, ref)

        step = 1e-7
        numeric = np.zeros_like(pred)
        for idx in np.ndindex(pred.shape):
            pred[idx] += step
            plus, _ = reconstruction_loss(pred, ref)
            pred[idx] -= 2 * step
            minus, _ = reconstruction_loss(pred, ref)
            pred[idx] += step
            numeric[idx] = (plus - minus) / (2 * step)
        assert np.allclose(grad, numeric, atol=1e-7)

    def test_attention_mask_excludes_positions(self) -> None:
        pred = np.ones((2, 4))
        ref = np.zeros((2, 4))
        mask = np.array([[1.0], [0.0]])
        masked, _ = reconstruction_loss(pred, ref, attention_mask=mask)
        unmasked, _ = reconstruction_loss(pred, ref)
        assert masked < unmasked


class TestOutlierSuppressedLoss:
    """The three subtle properties from DOCUMENTATION.md section 5.5."""

    def test_selection_is_global_not_per_row(self) -> None:
        # One row holds every outlier. Global selection excludes them all;
        # per-row selection would excise elements from the other row too. That
        # other row carries small nonzero residuals on purpose: with an all
        # zero row a per-row implementation excluding five zeros would have
        # passed this test unnoticed.
        pred = np.full((2, 1000), 0.01)
        ref = np.zeros((2, 1000))
        pred[0, :5] = 100.0

        loss, grad = outlier_suppressed_loss(pred, ref, fraction=0.005)
        # All ten excluded elements came from row 0 (five outliers plus five
        # of its 0.01 residuals, since the fraction is 0.5 percent of 2000),
        # so row 1 contributes every one of its elements to loss and gradient.
        assert np.allclose(grad[0, :5], 0.0)
        assert np.all(grad[1] != 0.0)
        kept_in_row_0 = 1000 - 10
        expected_loss = (kept_in_row_0 + 1000) * 0.01**2 / pred.size
        assert loss == pytest.approx(expected_loss)

    def test_denominator_is_the_full_element_count(self) -> None:
        # Excluded elements dilute the mean rather than leaving it. If the
        # denominator shrank to n - k, removing an outlier would raise the
        # reported loss on the remainder, which it must not.
        gen = np.random.default_rng(2)
        pred = gen.normal(size=(10, 100))
        ref = np.zeros_like(pred)
        n = pred.size
        k = max(1, int(n * 0.001))

        loss, _ = outlier_suppressed_loss(pred, ref, fraction=0.001)
        residual = np.abs(pred).reshape(-1)
        kept = np.sort(residual)[: n - k]
        assert loss == pytest.approx(float(np.sum(kept**2) / n))
        assert loss != pytest.approx(float(np.sum(kept**2) / (n - k)))

    def test_excluded_elements_receive_no_gradient(self) -> None:
        pred = np.zeros((1, 2000))
        ref = np.zeros((1, 2000))
        pred[0, 0] = 1e6
        _, grad = outlier_suppressed_loss(pred, ref)
        assert grad[0, 0] == 0.0

    def test_at_least_one_element_is_always_excluded(self) -> None:
        # k = max(1, ...) means even a tiny tensor drops its worst element.
        pred = np.array([[5.0, 0.0]])
        ref = np.zeros((1, 2))
        loss, _ = outlier_suppressed_loss(pred, ref, fraction=1e-9)
        assert loss == 0.0

    def test_reduces_the_influence_of_a_single_huge_error(self) -> None:
        gen = np.random.default_rng(3)
        pred = gen.normal(scale=0.01, size=(10, 200))
        ref = np.zeros_like(pred)
        pred[0, 0] = 500.0
        plain, _ = reconstruction_loss(pred, ref)
        suppressed, _ = outlier_suppressed_loss(pred, ref)
        assert suppressed < plain / 1000

    def test_gradient_matches_finite_differences_away_from_the_boundary(self) -> None:
        gen = np.random.default_rng(4)
        pred = gen.normal(size=(4, 500))
        ref = np.zeros_like(pred)
        _, grad = outlier_suppressed_loss(pred, ref)

        step = 1e-7
        # Check a sample of kept elements; perturbing an element near the
        # selection cutoff can change which elements are excluded, which makes
        # the finite difference meaningless there.
        keep = np.argsort(np.abs(pred).reshape(-1))[:100]
        for flat_index in keep[::10]:
            idx = np.unravel_index(flat_index, pred.shape)
            pred[idx] += step
            plus, _ = outlier_suppressed_loss(pred, ref)
            pred[idx] -= 2 * step
            minus, _ = outlier_suppressed_loss(pred, ref)
            pred[idx] += step
            assert grad[idx] == pytest.approx((plus - minus) / (2 * step), abs=1e-7)
