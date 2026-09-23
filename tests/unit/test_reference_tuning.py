# Copyright 2026 The MRound Authors
# SPDX-License-Identifier: Apache-2.0
"""Tests that learned rounding actually works.

Everything else in the reference test suite checks that a piece of arithmetic is
what the specification says. These tests check the thing that matters: that
optimizing the rounding produces a measurably better layer than rounding to
nearest, and that the advantage grows as bits get scarcer.

They run in under a second, need no model, no MLX, and no Apple hardware. That
is the whole argument for building the framework-free core first.
"""

from __future__ import annotations

import numpy as np
import pytest

from mround.reference.quantize import V_BOUND, QuantParams, fake_quantize, project_params
from mround.reference.tuning import round_to_nearest, tune_layer
from mround.schemes import QuantScheme, Symmetry, TuningConfig


def synthetic_layer(
    seed: int, out_features: int = 16, in_features: int = 64, n_samples: int = 128
) -> tuple[np.ndarray, np.ndarray]:
    """A weight matrix and correlated activations.

    Activations are deliberately correlated across features rather than white
    noise. Learned rounding exploits the fact that errors in different weights
    interact through the activation covariance; with perfectly white inputs
    there is much less for it to find, and the test would understate the method.
    """
    gen = np.random.default_rng(seed)
    weight = gen.normal(scale=0.05, size=(out_features, in_features))
    basis = gen.normal(size=(in_features, in_features))
    covariance = basis @ basis.T / in_features
    activations = gen.multivariate_normal(np.zeros(in_features), covariance, size=n_samples)
    return weight, activations


class TestLearnedRoundingBeatsRoundToNearest:
    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_tuning_reduces_output_error(self, bits: int) -> None:
        weight, activations = synthetic_layer(seed=bits)
        scheme = QuantScheme(bits=bits, group_size=32)
        result = tune_layer(weight, activations, scheme, TuningConfig(iters=200))

        assert result.final_loss < result.initial_loss
        assert result.improvement > 0.0

    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_improvement_is_material_not_marginal(self, bits: int) -> None:
        # A method that buys one percent would not be worth the engineering.
        weight, activations = synthetic_layer(seed=100 + bits)
        scheme = QuantScheme(bits=bits, group_size=32)
        result = tune_layer(weight, activations, scheme, TuningConfig(iters=200))
        assert result.improvement > 0.05, (
            f"only {result.improvement:.1%} improvement at {bits} bits"
        )

    def test_the_advantage_grows_as_bits_get_scarcer(self) -> None:
        # The qualitative claim behind the whole project: learned rounding
        # matters most exactly where round-to-nearest struggles.
        improvements = {}
        for bits in (2, 4, 8):
            weight, activations = synthetic_layer(seed=7)
            scheme = QuantScheme(bits=bits, group_size=32)
            result = tune_layer(weight, activations, scheme, TuningConfig(iters=200))
            improvements[bits] = result.improvement
        assert improvements[2] > improvements[8], improvements

    @pytest.mark.parametrize("symmetry", list(Symmetry))
    def test_both_symmetry_modes_improve(self, symmetry: Symmetry) -> None:
        weight, activations = synthetic_layer(seed=42)
        scheme = QuantScheme(bits=3, group_size=32, symmetry=symmetry)
        result = tune_layer(weight, activations, scheme, TuningConfig(iters=150))
        assert result.improvement > 0.0

    @pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
    def test_improvement_is_not_a_lucky_seed(self, seed: int) -> None:
        weight, activations = synthetic_layer(seed=seed)
        scheme = QuantScheme(bits=2, group_size=32)
        result = tune_layer(weight, activations, scheme, TuningConfig(iters=150))
        assert result.improvement > 0.0


class TestBaseline:
    def test_initial_loss_equals_round_to_nearest(self) -> None:
        # The property that makes improvement measurable: at step zero the
        # parameters reproduce RTN exactly, so initial_loss is the RTN baseline
        # rather than an arbitrary starting point.
        weight, activations = synthetic_layer(seed=11)
        scheme = QuantScheme(bits=4, group_size=32)
        result = tune_layer(weight, activations, scheme, TuningConfig(iters=5))

        rtn = round_to_nearest(weight, scheme)
        reference = activations @ weight.T
        rtn_loss = float(np.mean((activations @ rtn.T - reference) ** 2))
        assert result.initial_loss == pytest.approx(rtn_loss, rel=1e-12)

    def test_tuned_weights_differ_from_round_to_nearest(self) -> None:
        weight, activations = synthetic_layer(seed=12)
        scheme = QuantScheme(bits=3, group_size=32)
        result = tune_layer(weight, activations, scheme, TuningConfig(iters=100))
        assert not np.array_equal(result.qdq, round_to_nearest(weight, scheme))


class TestTrajectory:
    def test_best_state_is_tracked_not_just_the_last_step(self) -> None:
        # Signed gradient descent does not converge, it terminates. The final
        # step is not necessarily the best one, so the loop must remember.
        weight, activations = synthetic_layer(seed=13)
        scheme = QuantScheme(bits=2, group_size=32)
        result = tune_layer(weight, activations, scheme, TuningConfig(iters=200))
        assert result.final_loss <= min(result.losses)
        assert result.final_loss <= result.losses[-1]

    def test_more_iterations_do_not_make_it_worse(self) -> None:
        weight, activations = synthetic_layer(seed=14)
        scheme = QuantScheme(bits=3, group_size=32)
        short = tune_layer(weight, activations, scheme, TuningConfig(iters=50))
        long = tune_layer(weight, activations, scheme, TuningConfig(iters=400))
        assert long.final_loss <= short.final_loss * 1.05


class TestConstraints:
    def test_the_loop_projects_after_every_update(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The bound test below reads the best step's parameters, and the best
        # step is usually early enough that a dropped projection would still
        # pass it (verified by neutralizing project_params: best_step 17, max
        # |v| 0.085). So pin the mechanism itself: the loop must project once
        # after every one of its updates, and every projected parameter set
        # must lie inside the ranges the projection exists to enforce.
        seen: list[float] = []

        def spy(params: QuantParams, scheme: QuantScheme) -> None:
            project_params(params, scheme)
            lo, hi = scheme.coefficient_bounds
            assert np.abs(params.v).max() <= V_BOUND + 1e-12
            assert params.alpha.min() >= lo - 1e-12
            assert params.alpha.max() <= hi + 1e-12
            assert params.beta.min() >= lo - 1e-12
            assert params.beta.max() <= hi + 1e-12
            seen.append(float(np.abs(params.v).max()))

        monkeypatch.setattr("mround.reference.tuning.project_params", spy)
        weight, activations = synthetic_layer(seed=15)
        # Four steps at 2 bits: the first rate is 2 / 4 = 0.5, so a single
        # update reaches the bound and every later one would cross it.
        tune_layer(weight, activations, QuantScheme(bits=2, group_size=32), TuningConfig(iters=4))
        assert len(seen) == 4
        assert max(seen) == pytest.approx(V_BOUND)

    def test_the_rounding_perturbation_stays_bounded(self) -> None:
        # If the projection were dropped, signed descent would walk v past a
        # full code and the method would quietly become unconstrained weight
        # learning. See MEMORY.md D-008.
        weight, activations = synthetic_layer(seed=15)
        scheme = QuantScheme(bits=2, group_size=32)
        result = tune_layer(weight, activations, scheme, TuningConfig(iters=400))
        assert np.abs(result.params.v).max() <= V_BOUND + 1e-12

    def test_clipping_coefficients_stay_in_range(self) -> None:
        weight, activations = synthetic_layer(seed=16)
        scheme = QuantScheme(bits=4, group_size=32)
        result = tune_layer(weight, activations, scheme, TuningConfig(iters=200))
        lo, hi = scheme.coefficient_bounds
        assert result.params.alpha.min() >= lo - 1e-12
        assert result.params.alpha.max() <= hi + 1e-12
        assert result.params.beta.min() >= lo - 1e-12
        assert result.params.beta.max() <= hi + 1e-12

    def test_codes_remain_representable_after_tuning(self) -> None:
        weight, activations = synthetic_layer(seed=17)
        scheme = QuantScheme(bits=3, group_size=32)
        result = tune_layer(weight, activations, scheme, TuningConfig(iters=200))
        codes = fake_quantize(weight, result.params, scheme).codes
        lo, hi = scheme.code_range
        assert codes.min() >= lo
        assert codes.max() <= hi


class TestDeterminism:
    def test_identical_inputs_give_identical_results(self) -> None:
        # There is no randomness in the method itself, so a run is reproducible
        # from its inputs alone. That is what makes parity testing possible.
        weight, activations = synthetic_layer(seed=18)
        scheme = QuantScheme(bits=4, group_size=32)
        config = TuningConfig(iters=100)
        first = tune_layer(weight, activations, scheme, config)
        second = tune_layer(weight, activations, scheme, config)
        assert np.array_equal(first.qdq, second.qdq)
        assert first.losses == second.losses


class TestValidation:
    def test_mismatched_activation_width_is_rejected(self) -> None:
        weight = np.zeros((4, 32))
        activations = np.zeros((8, 16))
        with pytest.raises(ValueError, match="activations have 16 features"):
            tune_layer(weight, activations, QuantScheme(group_size=16))

    def test_the_inputs_are_untouched_by_tuning(self) -> None:
        # The loop works on copies: the weight and the activations it was
        # handed are the caller's and come back byte for byte as they went in.
        # (An earlier form of this test checked a parameter set the loop was
        # never given, which nothing could have failed.)
        weight, activations = synthetic_layer(seed=19)
        scheme = QuantScheme(bits=4, group_size=32)
        weight_before, activations_before = weight.copy(), activations.copy()
        result = tune_layer(weight, activations, scheme, TuningConfig(iters=20))
        assert np.array_equal(weight, weight_before)
        assert np.array_equal(activations, activations_before)
        assert not np.array_equal(result.params.v, np.zeros_like(result.params.v))
