# Copyright 2026 The MRound Authors
# SPDX-License-Identifier: Apache-2.0
"""The MLX implementation against the NumPy reference.

This is the test that makes the port trustworthy. Both implementations are ours,
they share no code, and one of them has exhaustive tests and
finite-difference-verified gradients. So a disagreement here is a porting defect
in the MLX side, not an open question about which is right. See MEMORY.md D-011.

Every test skips when MLX is unavailable, so a Linux machine still runs a green
suite. The whole file is the reason MRound built the reference first.

**Tolerances.** MLX computes in float32 by default and the reference in float64,
so exact agreement is not expected on anything involving accumulation. Integer
codes are a different matter: those must match exactly except where a value sits
close enough to a rounding boundary that float32 and float64 land on opposite
sides, which the tests measure as a fraction rather than forbidding outright.

Two of these tolerances are not float32 tolerances and are labeled where they
appear. ``INITIAL_LOSS_TOL`` holds a real, measured, unexplained divergence
(MEMORY.md Q-007); calling it precision would have been the comfortable answer
and it is false. The gradient tests, by contrast, are tight, because the clamp
boundary convention that used to make them fail is now specified on both sides
rather than inherited from whichever framework got there first (D-013).
"""

from __future__ import annotations

import numpy as np
import pytest

from mround.reference import quantize as ref_quant
from mround.reference import tuning as ref_tuning
from mround.schemes import QuantScheme, ScaleInit, Symmetry, TuningConfig

pytestmark = pytest.mark.needs_mlx

# importorskip triggers the skip on machines without MLX; the plain import
# that follows is what gives the type checker a name to resolve.
pytest.importorskip("mlx.core", reason="MLX is not installed")

import mlx.core as mx  # noqa: E402

from mround.core import losses as mlx_losses  # noqa: E402
from mround.core import quantizer as mlx_quant  # noqa: E402
from mround.core import signsgd as mlx_sgd  # noqa: E402
from mround.core import tuning as mlx_tuning  # noqa: E402
from mround.reference import losses as ref_losses  # noqa: E402

# float32 has about 7 decimal digits, and these quantities pass through a
# division by a scale that can be small, so this is deliberately loose. Codes
# are checked separately and much more tightly.
FLOAT32_TOL = 2e-5

# The round-to-nearest layer loss. This is NOT a float32 tolerance, and calling
# it one would be wrong. It is inherited from a measured property of the hardware:
# mx.matmul returns 8.5e-4 relative error on float32 operands, and the loss
# squares the residual, so it carries about twice that. See MEMORY.md D-014.
#
# Measured on Apple Silicon after D-014: 1.42e-3, 1.43e-3, 1.41e-3, 1.43e-3 at 2,
# 3, 4, and 8 bits. Twice the matmul error, as predicted, and flat across bit
# widths, which is the property D-014 was after: the error now scales with the
# residual rather than with the whole layer output. 3e-3 is that number with room
# for a different machine or MLX build, and it still fails on any real regression,
# since a single differing code out of a thousand moves this loss by one to four
# percent.
#
# Tighten it only against a measurement. check_mlx.sh prints the achieved figure
# every run for exactly that purpose.
INITIAL_LOSS_TOL = 3e-3


def synthetic(
    seed: int, out_features: int = 16, in_features: int = 64, n_samples: int = 128
) -> tuple[np.ndarray, np.ndarray]:
    """Weights and correlated activations, identical to the reference tests."""
    gen = np.random.default_rng(seed)
    weight = gen.normal(scale=0.05, size=(out_features, in_features))
    basis = gen.normal(size=(in_features, in_features))
    covariance = basis @ basis.T / in_features
    activations = gen.multivariate_normal(np.zeros(in_features), covariance, size=n_samples)
    return weight, activations


def to_mx(array: np.ndarray) -> mx.array:
    """Move a NumPy array into MLX at float32."""
    return mx.array(array.astype(np.float32))


def relative_error(got: np.ndarray, expected: np.ndarray) -> float:
    """Max absolute difference, normalized by the expected magnitude."""
    denom = max(1e-12, float(np.abs(expected).max(initial=0.0)))
    return float(np.abs(got - expected).max(initial=0.0) / denom)


class TestRounding:
    """The tie rule, checked first because everything downstream depends on it."""

    @pytest.mark.parametrize(
        ("value", "expected"),
        [(0.5, 0.0), (1.5, 2.0), (2.5, 2.0), (3.5, 4.0), (-0.5, 0.0), (-1.5, -2.0), (-2.5, -2.0)],
    )
    def test_ties_go_to_even(self, value: float, expected: float) -> None:
        # Built from floor rather than mx.round precisely so this does not
        # depend on the framework default. If this fails, the construction in
        # mround.core.quantizer.round_half_to_even is wrong, and every number
        # produced downstream is suspect.
        got = float(mlx_quant.round_half_to_even(mx.array([value]))[0])
        assert got == expected, f"round({value}) gave {got}, expected {expected}"

    def test_the_value_just_above_a_half_rounds_up(self) -> None:
        # The one float32 value the floor(x + 0.5) construction got wrong:
        # 0.5 + 1 ulp added to 0.5 rounds to exactly 1.0, looked like a tie,
        # and 1 is odd, so it came out 0. np.rint says 1, and so must this.
        value = np.nextafter(np.float32(0.5), np.float32(1.0))
        got = float(mlx_quant.round_half_to_even(mx.array(np.array([value])))[0])
        assert got == 1.0
        assert float(np.rint(value)) == 1.0

    def test_matches_the_reference_on_a_grid(self) -> None:
        # Includes exact halves, which is where the two could disagree.
        grid = np.arange(-8.0, 8.0, 0.25)
        expected = ref_quant.round_half_to_even(grid)
        got = np.array(mlx_quant.round_half_to_even(mx.array(grid.astype(np.float32))))
        mismatched = np.flatnonzero(got != expected)
        assert mismatched.size == 0, (
            f"disagree at {grid[mismatched].tolist()}: "
            f"MLX {got[mismatched].tolist()} vs reference {expected[mismatched].tolist()}"
        )


class TestScales:
    @pytest.mark.parametrize("bits", [2, 3, 4, 8])
    @pytest.mark.parametrize("group_size", [32, -1])
    def test_symmetric_scale_matches(self, bits: int, group_size: int) -> None:
        weight, _ = synthetic(seed=bits)
        scheme = QuantScheme(bits=bits, group_size=group_size)

        grouped, _ = ref_quant.regroup(weight, group_size)
        params = ref_quant.init_params(weight, scheme)
        expected, _ = ref_quant.symmetric_scale(grouped, params.alpha, params.beta, bits, 1e-8)

        m_grouped, _ = mlx_quant.regroup(to_mx(weight), group_size)
        m_params = mlx_quant.init_params(to_mx(weight), scheme)
        got = np.array(
            mlx_quant.symmetric_scale(m_grouped, m_params["alpha"], m_params["beta"], bits, 1e-8)
        )

        assert relative_error(got, expected) < FLOAT32_TOL

    @pytest.mark.parametrize("bits", [2, 4, 8])
    def test_symmetric_scale_signs_match_exactly(self, bits: int) -> None:
        # The sign is the whole mechanism (DOCUMENTATION.md 5.2.1), and unlike
        # the magnitude it is not subject to float32 error, so it must match
        # exactly rather than approximately.
        weight, _ = synthetic(seed=100 + bits)
        scheme = QuantScheme(bits=bits, group_size=32)

        grouped, _ = ref_quant.regroup(weight, 32)
        params = ref_quant.init_params(weight, scheme)
        expected, _ = ref_quant.symmetric_scale(grouped, params.alpha, params.beta, bits, 1e-8)

        m_grouped, _ = mlx_quant.regroup(to_mx(weight), 32)
        m_params = mlx_quant.init_params(to_mx(weight), scheme)
        got = np.array(
            mlx_quant.symmetric_scale(m_grouped, m_params["alpha"], m_params["beta"], bits, 1e-8)
        )
        assert np.array_equal(np.sign(got), np.sign(expected))

    @pytest.mark.parametrize("bits", [2, 4, 8])
    def test_asymmetric_scale_and_zero_point_match(self, bits: int) -> None:
        weight, _ = synthetic(seed=200 + bits)
        scheme = QuantScheme(bits=bits, group_size=32, symmetry=Symmetry.ASYMMETRIC)

        grouped, _ = ref_quant.regroup(weight, 32)
        params = ref_quant.init_params(weight, scheme)
        exp_scale, exp_zp = ref_quant.asymmetric_scale(
            grouped, params.alpha, params.beta, bits, 1e-8
        )

        m_grouped, _ = mlx_quant.regroup(to_mx(weight), 32)
        m_params = mlx_quant.init_params(to_mx(weight), scheme)
        got_scale, got_zp = mlx_quant.asymmetric_scale(
            m_grouped, m_params["alpha"], m_params["beta"], bits, 1e-8
        )

        assert relative_error(np.array(got_scale), exp_scale) < FLOAT32_TOL
        assert np.array_equal(np.array(got_zp), exp_zp), "zero points must match exactly"


class TestReconstruction:
    @pytest.mark.parametrize("bits", [2, 3, 4, 8])
    @pytest.mark.parametrize("symmetry", list(Symmetry))
    def test_reconstruction_matches(self, bits: int, symmetry: Symmetry) -> None:
        weight, _ = synthetic(seed=bits * 7)
        scheme = QuantScheme(bits=bits, group_size=32, symmetry=symmetry)

        expected = ref_quant.fake_quantize(
            weight, ref_quant.init_params(weight, scheme), scheme
        ).qdq
        got = np.array(
            mlx_quant.fake_quantize(
                to_mx(weight), mlx_quant.init_params(to_mx(weight), scheme), scheme
            )
        )
        assert relative_error(got, expected) < FLOAT32_TOL

    @pytest.mark.parametrize("bits", [2, 3, 4, 8])
    def test_codes_match_almost_exactly(self, bits: int) -> None:
        # Codes are integers, so they should agree exactly. The exception is a
        # weight sitting close enough to a rounding boundary that float32 and
        # float64 land on opposite sides. That is measured rather than tolerated
        # silently: a large fraction means a real bug, not precision.
        weight, _ = synthetic(seed=300 + bits, out_features=32, in_features=256)
        scheme = QuantScheme(bits=bits, group_size=32)

        expected = ref_quant.fake_quantize(
            weight, ref_quant.init_params(weight, scheme), scheme
        ).codes
        codes, _, _ = mlx_quant.quantize_codes(
            to_mx(weight), mlx_quant.init_params(to_mx(weight), scheme), scheme
        )
        got = np.array(codes)

        differing = float(np.mean(got != expected))
        assert differing < 0.001, (
            f"{differing:.3%} of codes differ at {bits} bits, which is too many "
            "to explain as float32 boundary effects"
        )

    @pytest.mark.parametrize("bits", [2, 4])
    def test_dominant_extreme_still_reconstructs_exactly(self, bits: int) -> None:
        # The property the signed scale exists to provide, checked on the MLX
        # side independently rather than inferred from the reconstruction match.
        scheme = QuantScheme(bits=bits, group_size=-1)
        weight = np.array([[1.0, -0.37, 0.1]])
        got = np.array(
            mlx_quant.fake_quantize(
                to_mx(weight), mlx_quant.init_params(to_mx(weight), scheme), scheme
            )
        )
        assert got[0, 0] == pytest.approx(1.0, abs=1e-6)


class TestGradients:
    """MLX autodiff against the reference's hand-derived gradients.

    This is the most valuable comparison in the file. The reference gradients
    were derived by hand and verified against finite differences; MLX's come
    from automatic differentiation. Two independent routes agreeing is real
    evidence, and a disagreement points at the straight-through estimator or the
    clamp, which are the two places a quantization gradient goes wrong.
    """

    @pytest.mark.parametrize("bits", [2, 4, 8])
    @pytest.mark.parametrize("symmetry", list(Symmetry))
    def test_autodiff_matches_hand_derived(self, bits: int, symmetry: Symmetry) -> None:
        weight, _ = synthetic(seed=400 + bits, out_features=4, in_features=32)
        target = np.random.default_rng(bits).normal(size=weight.shape) * 0.01
        scheme = QuantScheme(bits=bits, group_size=8, symmetry=symmetry)

        params = ref_quant.init_params(weight, scheme)
        gen = np.random.default_rng(bits + 1)
        params.v = params.v + gen.uniform(-0.3, 0.3, size=params.v.shape)
        params.alpha = params.alpha * (1.0 + gen.uniform(-0.15, 0.15, size=params.alpha.shape))
        params.beta = params.beta * (1.0 + gen.uniform(-0.15, 0.15, size=params.beta.shape))

        qdq = ref_quant.fake_quantize(weight, params, scheme).qdq
        expected = ref_quant.quantize_grad(
            weight, params, scheme, 2.0 * (qdq - target) / weight.size
        )

        m_weight, m_target = to_mx(weight), to_mx(target)
        m_params = {
            "v": to_mx(params.v),
            "alpha": to_mx(params.alpha),
            "beta": to_mx(params.beta),
        }

        def objective(p: dict[str, mx.array]) -> mx.array:
            residual = mlx_quant.fake_quantize(m_weight, p, scheme) - m_target
            return mx.mean(residual * residual)

        _, grads = mx.value_and_grad(objective)(m_params)

        for name, want in (("v", expected.v), ("alpha", expected.alpha), ("beta", expected.beta)):
            got = np.array(grads[name])
            assert relative_error(got, want) < 1e-3, (
                f"{name} gradient disagrees: MLX autodiff versus the reference's "
                f"hand-derived and finite-difference-verified value "
                f"(relative error {relative_error(got, want):.2e})"
            )


class TestRoundToNearestShortcut:
    """Passing ``None`` for the parameters must be exactly round-to-nearest.

    The shortcut exists to avoid allocating a zero perturbation the size of every
    weight matrix in a model, plus two constant coefficient arrays per group,
    only to add zero and multiply by one. Across a model that is a great deal of
    memory traffic for no arithmetic.

    Exactly is the operative word. This is a shortcut through code that has been
    verified against the reference, so anything less than bit-identical would
    mean the round-to-nearest baseline and the tuned result no longer start from
    the same place, and every improvement figure would be measured against the
    wrong thing.
    """

    @pytest.mark.parametrize("bits", [2, 3, 4, 8])
    @pytest.mark.parametrize("symmetry", list(Symmetry))
    @pytest.mark.parametrize("group_size", [32, -1])
    def test_none_is_bit_identical_to_initial_parameters(
        self, bits: int, symmetry: Symmetry, group_size: int
    ) -> None:
        weight, _ = synthetic(seed=900 + bits)
        scheme = QuantScheme(bits=bits, group_size=group_size, symmetry=symmetry)
        m_weight = to_mx(weight)

        explicit = mlx_quant.quantize_codes(
            m_weight, mlx_quant.init_params(m_weight, scheme), scheme
        )
        shortcut = mlx_quant.quantize_codes(m_weight, None, scheme)

        for name, want, got in zip(
            ("codes", "scale", "zero_point"), explicit, shortcut, strict=True
        ):
            if want is None:
                assert got is None
                continue
            assert np.array_equal(np.array(got), np.array(want)), f"{name} differs"

    @pytest.mark.parametrize("bits", [2, 4])
    @pytest.mark.parametrize("symmetry", list(Symmetry))
    def test_the_reconstruction_is_bit_identical_too(self, bits: int, symmetry: Symmetry) -> None:
        weight, _ = synthetic(seed=950 + bits)
        scheme = QuantScheme(bits=bits, group_size=32, symmetry=symmetry)
        m_weight = to_mx(weight)

        explicit = mlx_quant.fake_quantize(
            m_weight, mlx_quant.init_params(m_weight, scheme), scheme
        )
        shortcut = mlx_quant.fake_quantize(m_weight, None, scheme)
        assert np.array_equal(np.array(shortcut), np.array(explicit))


class TestClampBoundary:
    """The clamp boundary convention, pinned by a case built to sit on it.

    This exists because the convention was not pinned once, and the parity suite
    found it: MLX autodiff through ``mx.clip`` gives no gradient to a value
    exactly on ``q_min`` or ``q_max``, while the reference and PyTorch both do.
    The value reaching the clamp has already been rounded, so it is an integer
    and the boundary is the ordinary case rather than a coincidence, which is why
    the failure appeared at 2 and 4 bits and not at 8. See MEMORY.md D-013.
    """

    # Contrived so that, at 2 bits with one group, the codes come out
    # [-2, -2, -1, 1]: two on q_min, one interior, one on q_max.
    ON_BOUNDARY_WEIGHT = np.array([[1.0, 0.9, 0.5, -0.3]])

    def test_the_fixture_really_does_sit_on_the_boundary(self) -> None:
        # A test whose premise has quietly stopped holding is worse than no
        # test, so the premise is checked rather than assumed.
        scheme = QuantScheme(bits=2, group_size=-1)
        weight = self.ON_BOUNDARY_WEIGHT
        codes = ref_quant.fake_quantize(weight, ref_quant.init_params(weight, scheme), scheme).codes
        q_min, q_max = scheme.code_range
        on_boundary = (codes == q_min) | (codes == q_max)
        assert codes.ravel().tolist() == [-2.0, -2.0, -1.0, 1.0]
        assert int(on_boundary.sum()) == 3, "fixture no longer exercises the boundary"

    def test_boundary_codes_still_receive_gradient(self) -> None:
        scheme = QuantScheme(bits=2, group_size=-1)
        weight = self.ON_BOUNDARY_WEIGHT

        m_weight = to_mx(weight)
        m_params = mlx_quant.init_params(m_weight, scheme)

        def objective(p: dict[str, mx.array]) -> mx.array:
            qdq = mlx_quant.fake_quantize(m_weight, p, scheme)
            return mx.mean(qdq * qdq)

        _, grads = mx.value_and_grad(objective)(m_params)
        got = np.array(grads["v"]).ravel()

        # Under mx.clip's own convention three of these four are exactly zero,
        # which freezes those weights at round-to-nearest for the whole run.
        assert np.all(got != 0.0), (
            f"v gradient is zero where a code sits on the clamp boundary: {got.tolist()}"
        )

    @pytest.mark.parametrize("bits", [2, 4])
    @pytest.mark.parametrize("symmetry", list(Symmetry))
    def test_boundary_gradient_matches_the_reference(self, bits: int, symmetry: Symmetry) -> None:
        # The general version of the above: a wide layer at a low bit width puts
        # plenty of codes on the boundary without contriving anything.
        weight, _ = synthetic(seed=800 + bits, out_features=8, in_features=64)
        scheme = QuantScheme(bits=bits, group_size=16, symmetry=symmetry)

        params = ref_quant.init_params(weight, scheme)
        qdq = ref_quant.fake_quantize(weight, params, scheme).qdq
        expected = ref_quant.quantize_grad(weight, params, scheme, 2.0 * qdq / weight.size)

        m_weight = to_mx(weight)
        m_params = mlx_quant.init_params(m_weight, scheme)

        def objective(p: dict[str, mx.array]) -> mx.array:
            reconstruction = mlx_quant.fake_quantize(m_weight, p, scheme)
            return mx.mean(reconstruction * reconstruction)

        _, grads = mx.value_and_grad(objective)(m_params)
        assert relative_error(np.array(grads["v"]), expected.v) < 1e-3


class TestOptimizer:
    def test_sign_update_matches(self) -> None:
        param = np.array([1.0, -2.0, 0.5, 0.0])
        grad = np.array([0.3, -0.7, 0.0, 1e-9])
        expected = np.array(param) - 0.1 * np.sign(grad)
        got = np.array(mlx_sgd.sign_update(to_mx(param), to_mx(grad), 0.1))
        assert relative_error(got, expected) < FLOAT32_TOL

    @pytest.mark.parametrize("iters", [50, 200, 1000])
    def test_excursion_budget_matches(self, iters: int) -> None:
        config = TuningConfig(iters=iters)
        lr = config.resolved_lr(4)
        assert mlx_sgd.total_excursion(lr, iters) == pytest.approx(ref_quant.V_BOUND, abs=0.01)

    def test_schedule_reaches_zero(self) -> None:
        schedule = mlx_sgd.LinearDecay(0.005, 200)
        assert schedule(200) == 0.0


class TestLosses:
    def test_reconstruction_loss_matches(self) -> None:
        gen = np.random.default_rng(11)
        pred, ref = gen.normal(size=(8, 64)), gen.normal(size=(8, 64))
        expected, _ = ref_losses.reconstruction_loss(pred, ref)
        got = float(mlx_losses.reconstruction_loss(to_mx(pred), to_mx(ref)))
        assert got == pytest.approx(expected, rel=1e-4)

    def test_outlier_suppressed_loss_matches(self) -> None:
        gen = np.random.default_rng(12)
        pred = gen.normal(size=(10, 200))
        ref = np.zeros_like(pred)
        expected, _ = ref_losses.outlier_suppressed_loss(pred, ref)
        got = float(mlx_losses.outlier_suppressed_loss(to_mx(pred), to_mx(ref)))
        assert got == pytest.approx(expected, rel=1e-3)

    def test_outlier_suppression_excludes_the_worst(self) -> None:
        pred = np.zeros((1, 2000))
        pred[0, 0] = 1e6
        got = float(mlx_losses.outlier_suppressed_loss(to_mx(pred), mx.zeros((1, 2000))))
        assert got == pytest.approx(0.0, abs=1e-6)


class TestTuning:
    """End to end: does the MLX loop learn, and does it learn the same thing?"""

    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_tuning_improves_on_round_to_nearest(self, bits: int) -> None:
        weight, activations = synthetic(seed=bits)
        scheme = QuantScheme(bits=bits, group_size=32)
        result = mlx_tuning.tune_layer(
            to_mx(weight), to_mx(activations), scheme, TuningConfig(iters=200)
        )
        assert result.improvement > 0.05, (
            f"only {result.improvement:.1%} improvement at {bits} bits"
        )

    @pytest.mark.parametrize("bits", [2, 4])
    def test_initial_loss_matches_the_reference(self, bits: int) -> None:
        # Step zero is round-to-nearest in both, so this isolates the forward
        # pass from the optimization trajectory.
        weight, activations = synthetic(seed=500 + bits)
        scheme = QuantScheme(bits=bits, group_size=32)
        config = TuningConfig(iters=5)

        expected = ref_tuning.tune_layer(weight, activations, scheme, config)
        got = mlx_tuning.tune_layer(to_mx(weight), to_mx(activations), scheme, config)
        assert got.initial_loss == pytest.approx(expected.initial_loss, rel=INITIAL_LOSS_TOL)

    @pytest.mark.parametrize("bits", [2, 4])
    def test_the_reconstruction_behind_that_loss_is_identical(self, bits: int) -> None:
        # This guards the diagnosis of Q-007, not the loss. The loose tolerance
        # above is defensible only while the reconstruction feeding it is exact;
        # if the gap ever turns out to be a differing code, this test fails first
        # and names the real cause, because one code step is four orders of
        # magnitude larger than FLOAT32_TOL. Same seeds as the test above,
        # deliberately.
        weight, _ = synthetic(seed=500 + bits)
        scheme = QuantScheme(bits=bits, group_size=32)

        expected = ref_quant.fake_quantize(weight, ref_quant.init_params(weight, scheme), scheme)
        codes, _, _ = mlx_quant.quantize_codes(
            to_mx(weight), mlx_quant.init_params(to_mx(weight), scheme), scheme
        )
        assert np.array_equal(np.array(codes), expected.codes), (
            "codes differ, so the initial-loss gap is a quantizer defect rather "
            "than the open question Q-007 records"
        )

    @pytest.mark.parametrize("bits", [2, 4])
    def test_final_quality_is_comparable(self, bits: int) -> None:
        # Trajectories will diverge, because float32 versus float64 changes a
        # sign somewhere eventually and signed descent then takes a different
        # path. What must hold is that both arrive somewhere similarly good.
        weight, activations = synthetic(seed=600 + bits)
        scheme = QuantScheme(bits=bits, group_size=32)
        config = TuningConfig(iters=200)

        expected = ref_tuning.tune_layer(weight, activations, scheme, config)
        got = mlx_tuning.tune_layer(to_mx(weight), to_mx(activations), scheme, config)

        assert got.final_loss < expected.initial_loss, "MLX tuning did not beat RTN"
        assert got.final_loss == pytest.approx(expected.final_loss, rel=0.25), (
            f"MLX reached {got.final_loss:.4e}, reference reached "
            f"{expected.final_loss:.4e}; more than 25 percent apart suggests a "
            "defect rather than float32 drift"
        )

    def test_parameters_stay_bounded(self) -> None:
        weight, activations = synthetic(seed=700)
        scheme = QuantScheme(bits=2, group_size=32)
        result = mlx_tuning.tune_layer(
            to_mx(weight), to_mx(activations), scheme, TuningConfig(iters=300)
        )
        lo, hi = scheme.coefficient_bounds
        assert float(mx.abs(result.params["v"]).max()) <= mlx_quant.V_BOUND + 1e-6
        assert float(result.params["alpha"].min()) >= lo - 1e-6
        assert float(result.params["alpha"].max()) <= hi + 1e-6


class TestScaleSearchParity:
    """The MLX scale search against the NumPy one.

    Choices can legitimately differ where two candidates tie within the
    float32/float64 gap, so identical selection is asserted as a high fraction
    rather than an absolute, and where the choices differ the reconstruction
    quality must not: that is the property the search exists for.
    """

    @pytest.mark.parametrize("bits", [2, 3, 4, 8])
    def test_choices_match_and_quality_never_degrades(self, bits: int) -> None:
        from mround.core import scale_search as mlx_search  # noqa: PLC0415
        from mround.reference import scale_search as ref_search  # noqa: PLC0415

        rng = np.random.default_rng(bits)
        weight = rng.normal(size=(16, 64)).astype(np.float32)
        scheme = QuantScheme(bits=bits, group_size=32)

        ours = np.array(mlx_search.search_scales(mx.array(weight), scheme))
        theirs = ref_search.search_scales(weight.astype(np.float64), scheme)

        same = np.isclose(ours, theirs, rtol=1e-5, atol=1e-8)
        assert same.mean() >= 0.98, f"only {same.mean():.1%} of groups chose the same scale"

        # Where they differ, both scales must reconstruct equally well: a tie
        # resolved differently, not a worse answer.
        nmax = float(2 ** (bits - 1))
        grouped = weight.astype(np.float64).reshape(-1, 32)
        for row in np.flatnonzero(~same.ravel()):
            losses = []
            for scale in (float(ours[row, 0]), float(theirs[row, 0])):
                inverse = 0.0 if scale == 0 else 1.0 / scale
                codes = np.clip(
                    ref_quant.round_half_to_even(inverse * grouped[row]), -nmax, nmax - 1
                )
                losses.append(float(np.sum((scale * codes - grouped[row]) ** 2)))
            assert losses[0] <= losses[1] * (1 + 1e-4), (row, losses)

    def test_importance_weighting_ports(self) -> None:
        from mround.core import scale_search as mlx_search  # noqa: PLC0415
        from mround.reference import scale_search as ref_search  # noqa: PLC0415

        rng = np.random.default_rng(42)
        weight = rng.normal(size=(8, 32)).astype(np.float32)
        importance = rng.uniform(0.1, 10.0, size=32)
        scheme = QuantScheme(bits=3, group_size=16)

        ours = np.array(
            mlx_search.search_scales(
                mx.array(weight), scheme, importance=importance.astype(np.float32)
            )
        )
        theirs = ref_search.search_scales(weight.astype(np.float64), scheme, importance=importance)
        same = np.isclose(ours, theirs, rtol=1e-5, atol=1e-8)
        assert same.mean() >= 0.95


class TestSearchedBranchParity:
    """The searched branch: forward parity, and gradients against hand math.

    The gradient check is the one that earns its keep. Autodiff through the
    searched branch flows through the STE, the clamp mask, and the scale
    product; the hand derivation is dW'/dv = scale inside the clamp and
    dW'/dbeta = s0 * q - inside * w / beta, summed per group. Disagreement
    means the branch is differentiating something other than the specification.
    """

    def _searched_case(self) -> tuple[np.ndarray, np.ndarray, QuantScheme]:
        # Constructed from codes and fractions rather than sampled, so no
        # element sits near a rounding boundary and finite precision cannot
        # flip a code between the two implementations. Two elements per group
        # land outside the clamp on purpose, to exercise the mask.
        rng = np.random.default_rng(11)
        scheme = QuantScheme(bits=3, group_size=16, scale_init=ScaleInit.SEARCHED)
        rows, gpr = 4, 2
        s0 = rng.uniform(0.1, 0.3, size=(rows * gpr, 1)) * rng.choice([-1.0, 1.0], (rows * gpr, 1))
        codes = rng.integers(-3, 3, size=(rows * gpr, 16)).astype(np.float64)
        codes[:, 0] = 6.0  # clamps to 3 after rounding
        codes[:, 1] = -7.0  # clamps to -4
        frac = rng.uniform(-0.35, 0.35, size=codes.shape)
        weight = (s0 * (codes + frac)).reshape(rows, gpr * 16)
        return weight.astype(np.float64), s0, scheme

    def test_forward_matches_the_reference(self) -> None:
        from mround.core import quantizer as mlx_quant  # noqa: PLC0415
        from mround.reference import quantize as ref  # noqa: PLC0415

        weight, s0, scheme = self._searched_case()
        params = ref.init_params(weight, scheme)
        theirs = ref.fake_quantize(weight, params, scheme, init_scale=s0)
        ours = np.array(
            mlx_quant.fake_quantize(
                mx.array(weight.astype(np.float32)),
                None,
                scheme,
                init_scale=mx.array(s0.astype(np.float32)),
            )
        )
        np.testing.assert_allclose(ours, theirs.qdq, rtol=1e-5, atol=1e-6)

    def test_gradients_match_the_hand_derivation(self) -> None:
        from mround.core import quantizer as mlx_quant  # noqa: PLC0415

        weight, s0, scheme = self._searched_case()
        rng = np.random.default_rng(12)
        rows, gpr, gs = 4, 2, 16
        v0 = rng.uniform(-0.3, 0.3, size=(rows, gpr, gs)).astype(np.float32)
        beta0 = rng.uniform(0.7, 1.3, size=(rows, gpr, 1)).astype(np.float32)

        w32 = mx.array(weight.astype(np.float32))
        init = mx.array(s0.astype(np.float32))
        params = {
            "v": mx.array(v0),
            "alpha": mx.ones((rows, gpr, 1)),
            "beta": mx.array(beta0),
        }

        def total(p: dict[str, mx.array]) -> mx.array:
            return mx.sum(mlx_quant.fake_quantize(w32, p, scheme, init_scale=init))

        grads = mx.grad(total)(params)
        mx.eval(grads)

        # Hand derivation, float64.
        s0g = s0.reshape(rows, gpr, 1)
        scale = s0g * beta0.astype(np.float64)
        grouped = weight.reshape(rows, gpr, gs)
        u = grouped / scale + v0
        q = np.round(u)  # constructed away from .5 boundaries
        inside = (q >= -4) & (q <= 3)
        qc = np.clip(q, -4, 3)
        want_v = scale * inside
        want_beta = np.sum(s0g * qc - inside * grouped / beta0, axis=-1, keepdims=True)

        np.testing.assert_allclose(np.array(grads["v"]), want_v, rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(np.array(grads["beta"]), want_beta, rtol=1e-4, atol=1e-3)
        np.testing.assert_allclose(np.array(grads["alpha"]), 0.0, atol=1e-7)

        # And against the oracle's own analytic gradient, now that the
        # reference has a searched branch too: same point, unit upstream
        # gradient, three arrays.
        ref_params = ref_quant.QuantParams(
            v=v0.astype(np.float64), alpha=np.ones((rows, gpr, 1)), beta=beta0.astype(np.float64)
        )
        theirs = ref_quant.quantize_grad(
            weight, ref_params, scheme, np.ones_like(weight), init_scale=s0
        )
        np.testing.assert_allclose(np.array(grads["v"]), theirs.v, rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(np.array(grads["beta"]), theirs.beta, rtol=1e-4, atol=1e-3)
        np.testing.assert_allclose(theirs.alpha, 0.0, atol=0.0)

    def test_tuning_on_the_searched_grid_matches_the_reference_loop(self) -> None:
        # The recipe every 2 bit and mixed ledger row was produced under, run
        # through both loops from the same searched scale. Until the reference
        # loop accepted init_scale this comparison did not exist for the
        # branch that produced the project's headline numbers.
        from mround.reference import scale_search as ref_search  # noqa: PLC0415

        rng = np.random.default_rng(21)
        scheme = QuantScheme(bits=3, group_size=16, scale_init=ScaleInit.SEARCHED)
        weight = rng.normal(scale=0.05, size=(8, 64))
        activations = rng.normal(size=(32, 64))
        config = TuningConfig(iters=30, batch_size=1, n_samples=1)
        s0 = ref_search.search_scales(weight, scheme)

        theirs = ref_tuning.tune_layer(weight, activations, scheme, config, init_scale=s0)
        ours = mlx_tuning.tune_layer(
            mx.array(weight.astype(np.float32)),
            mx.array(activations.astype(np.float32)),
            scheme,
            config,
            init_scale=mx.array(np.asarray(s0).astype(np.float32)),
        )
        # INITIAL_LOSS_TOL, not a float32 tolerance. This assertion was written
        # with rel=1e-4 and passed on MLX's CPU backend, where it agrees with
        # the float64 reference to 3.8e-7, then failed on Metal at 1.32e-3.
        # That is not a divergence in the searched branch: identical code and
        # identical dtype on CPU land on the reference, so what the Mac adds is
        # the matmul kernel D-014 measured at 8.5e-4, and this loss squares the
        # residual and carries about twice it. 1.32e-3 sits in the band this
        # constant was measured over (1.41e-3 to 1.43e-3 at 2, 3, 4 and 8 bits).
        # The lesson is the one CLAUDE.md states: a suite green on CPU and red
        # on Metal is a finding, and here the finding was in the test.
        assert ours.initial_loss == pytest.approx(theirs.initial_loss, rel=INITIAL_LOSS_TOL)
        # The two loops take the same signed steps from the same point, so
        # their trajectories agree closely for the first steps and stay in
        # family to the end; whether this toy layer improves in 30 steps is
        # not the question, tracking is. Each step's loss is measured the same
        # way as the initial one, so it carries the same floor and cannot be
        # asserted below it.
        for step in range(1, 6):
            assert ours.losses[step] == pytest.approx(theirs.losses[step], rel=INITIAL_LOSS_TOL)
        assert ours.final_loss == pytest.approx(theirs.final_loss, rel=0.05)
