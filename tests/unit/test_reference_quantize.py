# Copyright 2026 The MRound Authors
# SPDX-License-Identifier: Apache-2.0
"""Tests for the NumPy reference quantizers.

Two kinds of test live here. The first kind checks the formulas against
hand-computed values and against the structural properties DOCUMENTATION.md
section 5.2 claims for them. The second kind checks the analytic gradients
against finite differences.

The second kind is not decoration. Gradients hand-derived through a
straight-through estimator and a clamp are exactly the sort of thing that is
wrong in a way no other test notices: the optimization still runs, still
descends, and still produces a plausible model that is quietly worse.
"""

from __future__ import annotations

import numpy as np
import pytest

from mround.exceptions import SchemeError
from mround.reference.quantize import (
    V_BOUND,
    QuantParams,
    asymmetric_scale,
    dequantize,
    fake_quantize,
    init_params,
    project_params,
    quantize_grad,
    regroup,
    round_half_to_even,
    symmetric_scale,
    ungroup,
)
from mround.reference.scale_search import search_scales
from mround.schemes import QuantScheme, ScaleInit, Symmetry


def rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


class TestRounding:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [(0.5, 0.0), (1.5, 2.0), (2.5, 2.0), (3.5, 4.0), (-0.5, -0.0), (-1.5, -2.0)],
    )
    def test_ties_go_to_even(self, value: float, expected: float) -> None:
        # Load-bearing. The reference implementation's own self-tests fail under
        # round-half-away-from-zero, which makes this the most likely silent
        # divergence when porting to another framework.
        assert round_half_to_even(np.array([value]))[0] == expected


class TestGrouping:
    def test_even_division_round_trips(self) -> None:
        w = rng().normal(size=(4, 128))
        grouped, pad = regroup(w, 32)
        assert grouped.shape == (4, 4, 32)
        assert pad == 0
        assert np.array_equal(ungroup(grouped, w.shape, pad), w)

    def test_uneven_division_pads_and_round_trips(self) -> None:
        w = rng().normal(size=(4, 100))
        grouped, pad = regroup(w, 32)
        assert grouped.shape == (4, 4, 32)
        assert pad == 28
        assert np.array_equal(ungroup(grouped, w.shape, pad), w)

    def test_per_channel_is_one_group(self) -> None:
        w = rng().normal(size=(4, 100))
        grouped, pad = regroup(w, -1)
        assert grouped.shape == (4, 1, 100)
        assert pad == 0

    def test_non_matrix_is_rejected(self) -> None:
        with pytest.raises(SchemeError, match="2-D weight matrix"):
            regroup(np.zeros(10), 32)


class TestSymmetricScaleSign:
    """DOCUMENTATION.md section 5.2.1.

    The scale carries the sign OPPOSITE to the dominant extreme. These tests
    pin that down, because an implementation that normalizes the sign away
    still runs and still produces a model, just a measurably worse one.
    """

    @staticmethod
    def _group(values: list[float]) -> np.ndarray:
        return np.array([[values]], dtype=np.float64)

    def _ones(self) -> np.ndarray:
        return np.ones((1, 1, 1))

    def test_scale_is_negative_when_positive_side_dominates(self) -> None:
        g = self._group([1.0, -0.6, 0.2])
        scale, positive_dominates = symmetric_scale(g, self._ones(), self._ones(), 4, 1e-8)
        assert bool(positive_dominates[0, 0, 0])
        assert scale[0, 0, 0] < 0.0

    def test_scale_is_positive_when_negative_side_dominates(self) -> None:
        g = self._group([0.6, -1.0, 0.2])
        scale, positive_dominates = symmetric_scale(g, self._ones(), self._ones(), 4, 1e-8)
        assert not bool(positive_dominates[0, 0, 0])
        assert scale[0, 0, 0] > 0.0

    @pytest.mark.parametrize("bits", [2, 3, 4, 8])
    @pytest.mark.parametrize("dominant_sign", [1.0, -1.0])
    def test_dominant_extreme_reconstructs_exactly(self, bits: int, dominant_sign: float) -> None:
        # The property the sign inversion exists to provide. The dominant
        # extreme maps onto -2**(bits-1), which is representable, so it
        # reconstructs with zero error rather than clipping.
        scheme = QuantScheme(bits=bits, group_size=-1, symmetry=Symmetry.SYMMETRIC)
        dominant = dominant_sign * 1.0
        w = np.array([[dominant, -0.37 * dominant_sign, 0.1 * dominant_sign]])
        result = fake_quantize(w, init_params(w, scheme), scheme)
        assert result.qdq[0, 0] == pytest.approx(dominant, abs=1e-12)

    @pytest.mark.parametrize("bits", [2, 3, 4, 8])
    def test_dominant_extreme_maps_to_the_extra_code(self, bits: int) -> None:
        scheme = QuantScheme(bits=bits, group_size=-1, symmetry=Symmetry.SYMMETRIC)
        w = np.array([[1.0, -0.4, 0.25]])
        result = fake_quantize(w, init_params(w, scheme), scheme)
        assert result.codes[0, 0, 0] == -(2 ** (bits - 1))

    @pytest.mark.parametrize("bits", [2, 4, 8])
    def test_naive_unsigned_scale_would_clip(self, bits: int) -> None:
        # Demonstrates what the sign inversion buys, by computing the
        # alternative that an earlier draft of the specification described.
        # At 2 bits the naive form loses half the extreme's magnitude.
        w_extreme = 1.0
        maxq = 2 ** (bits - 1)
        naive_scale = w_extreme / maxq
        naive_code = min(round(w_extreme / naive_scale), maxq - 1)
        naive_recon = naive_scale * naive_code
        assert naive_recon < w_extreme
        assert 1.0 - naive_recon / w_extreme == pytest.approx(1.0 / maxq)

    def test_all_zero_group_does_not_divide_by_zero(self) -> None:
        scheme = QuantScheme(bits=4, group_size=-1, symmetry=Symmetry.SYMMETRIC)
        w = np.zeros((2, 8))
        result = fake_quantize(w, init_params(w, scheme), scheme)
        assert np.all(np.isfinite(result.qdq))
        assert np.all(result.qdq == 0.0)


class TestCodeRanges:
    @pytest.mark.parametrize("bits", [2, 3, 4, 5, 6, 8])
    @pytest.mark.parametrize("symmetry", list(Symmetry))
    def test_codes_stay_in_range(self, bits: int, symmetry: Symmetry) -> None:
        scheme = QuantScheme(bits=bits, group_size=16, symmetry=symmetry)
        w = rng(bits).normal(scale=3.0, size=(6, 64))
        params = init_params(w, scheme)
        params.v = rng(1).uniform(-V_BOUND, V_BOUND, size=params.v.shape)
        result = fake_quantize(w, params, scheme)
        lo, hi = scheme.code_range
        assert result.codes.min() >= lo
        assert result.codes.max() <= hi

    @pytest.mark.parametrize("bits", [2, 4, 8])
    def test_asymmetric_zero_point_is_representable(self, bits: int) -> None:
        # Forcing zero into the extremes is what guarantees this. Without it a
        # single-sign group produces a zero point outside the code range, which
        # no format storing an unsigned integer zero point can represent.
        for w in (
            np.array([[-2.0, -1.5, -1.0]]),  # all negative
            np.array([[1.0, 1.5, 2.0]]),  # all positive
            np.array([[-1.0, 0.0, 1.0]]),  # straddling
        ):
            ones = np.ones((1, 1, 1))
            _, zp = asymmetric_scale(regroup(w, -1)[0], ones, ones, bits, 1e-8)
            assert 0 <= zp[0, 0, 0] <= 2**bits - 1


class TestReconstruction:
    @pytest.mark.parametrize("bits", [2, 4, 8])
    @pytest.mark.parametrize("symmetry", list(Symmetry))
    def test_dequantize_inverts_the_quantization(self, bits: int, symmetry: Symmetry) -> None:
        scheme = QuantScheme(bits=bits, group_size=32, symmetry=symmetry)
        w = rng(bits).normal(size=(4, 128))
        result = fake_quantize(w, init_params(w, scheme), scheme)
        again = dequantize(result.codes, result.scale, result.zero_point, w.shape, scheme)
        assert np.allclose(again, result.qdq, atol=1e-12)

    @pytest.mark.parametrize("bits", [3, 4, 8])
    def test_more_bits_reconstruct_better(self, bits: int) -> None:
        w = rng(7).normal(size=(8, 128))
        errors = []
        for b in (2, bits):
            scheme = QuantScheme(bits=b, group_size=32)
            result = fake_quantize(w, init_params(w, scheme), scheme)
            errors.append(float(np.mean((result.qdq - w) ** 2)))
        assert errors[1] < errors[0]

    def test_smaller_groups_reconstruct_better(self) -> None:
        w = rng(8).normal(size=(8, 256))
        errors = []
        for g in (256, 32):
            scheme = QuantScheme(bits=4, group_size=g)
            result = fake_quantize(w, init_params(w, scheme), scheme)
            errors.append(float(np.mean((result.qdq - w) ** 2)))
        assert errors[1] < errors[0]

    def test_initial_parameters_reproduce_round_to_nearest(self) -> None:
        # v = 0 and alpha = beta = 1 must give exactly RTN, because that is what
        # makes the tuning improvement measurable against a known baseline.
        scheme = QuantScheme(bits=4, group_size=32)
        w = rng(9).normal(size=(4, 64))
        params = init_params(w, scheme)
        result = fake_quantize(w, params, scheme)

        grouped, pad = regroup(w, 32)
        scale, _ = symmetric_scale(grouped, params.alpha, params.beta, 4, 1e-8)
        lo, hi = scheme.code_range
        rtn = scale * np.clip(round_half_to_even(grouped / scale), lo, hi)
        assert np.allclose(result.qdq, ungroup(rtn, w.shape, pad), atol=1e-12)


class TestProjection:
    def test_projection_bounds_every_parameter(self) -> None:
        scheme = QuantScheme(bits=4, group_size=16)
        w = rng(3).normal(size=(2, 32))
        params = init_params(w, scheme)
        params.v += 5.0
        params.alpha += 5.0
        params.beta -= 5.0
        project_params(params, scheme)

        lo, hi = scheme.coefficient_bounds
        assert params.v.max() <= V_BOUND
        assert params.v.min() >= -V_BOUND
        assert params.alpha.max() <= hi
        assert params.beta.min() >= lo

    def test_searched_initialization_permits_growth_above_one(self) -> None:
        # Under the searched parameterization the coefficient multiplies the
        # search result, so forbidding values above 1.0 would forbid the
        # optimizer from ever increasing the scale.
        observed = QuantScheme(scale_init=ScaleInit.OBSERVED_RANGE)
        searched = QuantScheme(scale_init=ScaleInit.SEARCHED)
        assert observed.coefficient_bounds[1] == 1.0
        assert searched.coefficient_bounds[1] > 1.0

    def test_lower_bound_is_strictly_positive(self) -> None:
        # At exactly zero the group range collapses and every weight saturates
        # to a single code, so the floor makes that state unreachable.
        for scheme in (QuantScheme(), QuantScheme(scale_init=ScaleInit.SEARCHED)):
            assert scheme.coefficient_bounds[0] > 0.0


def _ste_surrogate(
    w: np.ndarray,
    params: QuantParams,
    scheme: QuantScheme,
    frozen: dict[str, np.ndarray],
    eps: float = 1e-8,
) -> np.ndarray:
    """The smooth function whose gradient the straight-through estimator computes.

    Finite-differencing the real forward pass cannot validate a straight-through
    gradient, and the first version of this test made exactly that mistake. The
    real forward is piecewise constant in the rounding perturbation, so its true
    derivative is zero almost everywhere; that is precisely the problem the
    estimator exists to solve, and reporting zero is the finite difference being
    right rather than the gradient being wrong.

    What the estimator actually computes is the exact gradient of this function:
    the same arithmetic with each rounding operation replaced by adding back the
    residual it produced at the base point, held constant, and with the clamp
    replaced by the mask it produced at the base point, also held constant.
    Differencing this validates the analytic gradient properly.
    """
    grouped, pad = regroup(w, scheme.group_size)
    lo, hi = scheme.code_range

    if scheme.scale_init is ScaleInit.SEARCHED:
        # The searched parameterization: the scale is the searched value
        # times beta, and alpha plays no part (DOCUMENTATION.md 5.3, D-030).
        base = frozen["base"].reshape(grouped.shape[0], grouped.shape[1], 1)
        scale = base * params.beta
        raw = grouped / scale + params.v + frozen["residual"]
        codes = np.where(frozen["mask"], raw, frozen["bound"])
        qdq_grouped = scale * codes
    elif scheme.symmetry is Symmetry.SYMMETRIC:
        scale, _ = symmetric_scale(grouped, params.alpha, params.beta, scheme.bits, eps)
        raw = grouped / scale + params.v + frozen["residual"]
        codes = np.where(frozen["mask"], raw, frozen["bound"])
        qdq_grouped = scale * codes
    else:
        w_max, w_min = _clipped_extremes_for_test(grouped, params.alpha, params.beta)
        maxq = float(2**scheme.bits - 1)
        scale = np.maximum((w_max - w_min) / maxq, eps)
        zero_point = -w_min / scale + frozen["zp_residual"]
        raw = grouped / scale + params.v + frozen["residual"] + zero_point
        codes = np.where(frozen["mask"], raw, frozen["bound"])
        qdq_grouped = scale * (codes - zero_point)

    _ = (lo, hi)
    return ungroup(qdq_grouped, w.shape, pad)


def _clipped_extremes_for_test(
    grouped: np.ndarray, alpha: np.ndarray, beta: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Mirror of the private helper, so the surrogate stays self-contained."""
    w_max = np.clip(grouped.max(axis=-1, keepdims=True), 0.0, None) * alpha
    w_min = np.clip(grouped.min(axis=-1, keepdims=True), None, 0.0) * beta
    return w_max, w_min


def _freeze(
    w: np.ndarray,
    params: QuantParams,
    scheme: QuantScheme,
    eps: float = 1e-8,
    init_scale: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Capture the rounding residuals and clamp mask at the current point."""
    grouped, _ = regroup(w, scheme.group_size)
    lo, hi = scheme.code_range
    base = np.zeros((grouped.shape[0], grouped.shape[1]))

    if scheme.scale_init is ScaleInit.SEARCHED:
        assert init_scale is not None
        base = np.asarray(init_scale, dtype=float).reshape(grouped.shape[0], grouped.shape[1])
        scale = base[..., None] * params.beta
        pre = grouped / scale + params.v
        rounded = round_half_to_even(pre)
        zp_residual = np.zeros_like(scale)
    elif scheme.symmetry is Symmetry.SYMMETRIC:
        scale, _ = symmetric_scale(grouped, params.alpha, params.beta, scheme.bits, eps)
        pre = grouped / scale + params.v
        rounded = round_half_to_even(pre)
        zp_residual = np.zeros_like(scale)
    else:
        scale, zero_point = asymmetric_scale(grouped, params.alpha, params.beta, scheme.bits, eps)
        _, w_min = _clipped_extremes_for_test(grouped, params.alpha, params.beta)
        zp_residual = zero_point - (-w_min / scale)
        pre = grouped / scale + params.v
        rounded = round_half_to_even(pre) + zero_point

    mask = (rounded >= lo) & (rounded <= hi)
    return {
        "residual": round_half_to_even(grouped / scale + params.v) - (grouped / scale + params.v),
        "mask": mask,
        "bound": np.clip(rounded, lo, hi),
        "zp_residual": zp_residual,
        "base": base,
    }


class TestGradients:
    """Analytic gradients against central finite differences of the surrogate.

    The loss is a plain sum of squares against a fixed target, which makes the
    gradient with respect to the reconstruction trivially known and isolates
    what is under test: the path from the reconstruction back to the three
    learnable arrays.
    """

    def _check(
        self,
        scheme: QuantScheme,
        *,
        seed: int,
        step: float = 1e-6,
        tol: float = 1e-6,
    ) -> None:
        gen = rng(seed)
        w = gen.normal(size=(3, 24))
        target = gen.normal(size=(3, 24)) * 0.1
        params = init_params(w, scheme)
        # Move off the initial point so the test exercises a general position
        # rather than the special case at v = 0 and alpha = beta = 1.
        params.v += gen.uniform(-0.3, 0.3, size=params.v.shape)
        params.alpha *= 1.0 + gen.uniform(-0.15, 0.15, size=params.alpha.shape)
        params.beta *= 1.0 + gen.uniform(-0.15, 0.15, size=params.beta.shape)

        init_scale = None
        if scheme.scale_init is ScaleInit.SEARCHED:
            init_scale = search_scales(w, scheme)
        frozen = _freeze(w, params, scheme, init_scale=init_scale)

        def loss(p: QuantParams) -> float:
            return float(np.sum((_ste_surrogate(w, p, scheme, frozen) - target) ** 2))

        qdq = fake_quantize(w, params, scheme, init_scale=init_scale).qdq
        analytic = quantize_grad(w, params, scheme, 2.0 * (qdq - target), init_scale=init_scale)

        for name in ("v", "alpha", "beta"):
            arr = getattr(params, name)
            got = getattr(analytic, name)
            numeric = np.zeros_like(arr)
            for idx in np.ndindex(arr.shape):
                original = arr[idx]
                arr[idx] = original + step
                plus = loss(params)
                arr[idx] = original - step
                minus = loss(params)
                arr[idx] = original
                numeric[idx] = (plus - minus) / (2.0 * step)

            denom = max(1.0, float(np.abs(numeric).max(initial=1.0)))
            assert np.allclose(got / denom, numeric / denom, atol=tol), (
                f"{name} gradient disagrees with finite differences under {scheme}: "
                f"max abs error {np.abs(got - numeric).max():.3e}"
            )

    @pytest.mark.parametrize("bits", [2, 4, 8])
    def test_symmetric_gradients(self, bits: int) -> None:
        self._check(QuantScheme(bits=bits, group_size=8), seed=100 + bits)

    @pytest.mark.parametrize("bits", [2, 4, 8])
    def test_asymmetric_gradients(self, bits: int) -> None:
        self._check(
            QuantScheme(bits=bits, group_size=8, symmetry=Symmetry.ASYMMETRIC),
            seed=200 + bits,
        )

    def test_per_channel_gradients(self) -> None:
        self._check(QuantScheme(bits=4, group_size=-1), seed=300)

    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_searched_gradients(self, bits: int) -> None:
        # The branch every 2 bit and mixed ledger row was produced under. Until
        # it existed the oracle differentiated a searched scheme as an observed
        # range one, handing alpha the gradient and beta exactly zero, the
        # opposite of what the MLX core does and of what D-030 records.
        self._check(
            QuantScheme(bits=bits, group_size=8, scale_init=ScaleInit.SEARCHED),
            seed=400 + bits,
        )

    def test_searched_gradient_leaves_alpha_inert_and_needs_the_scale(self) -> None:
        scheme = QuantScheme(bits=4, group_size=8, scale_init=ScaleInit.SEARCHED)
        w = rng(7).normal(size=(2, 16))
        params = init_params(w, scheme)
        grad = quantize_grad(
            w, params, scheme, np.ones_like(w), init_scale=search_scales(w, scheme)
        )
        assert np.all(grad.alpha == 0.0)
        assert np.any(grad.beta != 0.0)
        with pytest.raises(ValueError, match="needs the searched scale"):
            quantize_grad(w, params, scheme, np.ones_like(w))
        with pytest.raises(ValueError, match="not SEARCHED"):
            quantize_grad(
                w, params, QuantScheme(bits=4, group_size=8), np.ones_like(w), init_scale=w[:, :2]
            )

    def test_only_the_dominant_side_receives_coefficient_gradient(self) -> None:
        # A property worth pinning: in symmetric mode the non-dominant
        # coefficient has no influence on the scale, so its gradient is exactly
        # zero and the optimizer moves it on the sign of nothing. This explains
        # why one of the two coefficients appears inert.
        scheme = QuantScheme(bits=4, group_size=-1)
        w = np.array([[1.0, -0.4, 0.2, 0.05]])
        params = init_params(w, scheme)
        grad = quantize_grad(w, params, scheme, np.ones_like(w))
        assert grad.beta[0, 0, 0] == 0.0
        assert grad.alpha[0, 0, 0] != 0.0


class TestClampBoundaryConvention:
    """A code exactly on the boundary still receives gradient.

    DOCUMENTATION.md section 5.7 and MEMORY.md D-013. This lives here, not only
    in the MLX parity suite, because it is a specification item rather than a
    porting detail, and because the parity suite does not run without MLX. It was
    also not a hypothetical: the MLX port inherited the opposite convention from
    ``mx.clip`` and the first run on Apple hardware failed on it.
    """

    # At 2 bits with one group the codes come out [-2, -2, -1, 1]: two on q_min,
    # one interior, one on q_max.
    WEIGHT = np.array([[1.0, 0.9, 0.5, -0.3]])
    SCHEME = QuantScheme(bits=2, group_size=-1)

    def test_the_fixture_sits_on_both_boundaries(self) -> None:
        # Checked rather than assumed, because a fixture whose premise has
        # quietly stopped holding is worse than no test at all.
        codes = fake_quantize(self.WEIGHT, init_params(self.WEIGHT, self.SCHEME), self.SCHEME).codes
        assert codes.ravel().tolist() == [-2.0, -2.0, -1.0, 1.0]

    def test_boundary_codes_receive_gradient(self) -> None:
        params = init_params(self.WEIGHT, self.SCHEME)
        grad = quantize_grad(self.WEIGHT, params, self.SCHEME, np.ones_like(self.WEIGHT))
        assert np.all(grad.v != 0.0), (
            f"a code on the clamp boundary received no gradient: {grad.v.ravel().tolist()}"
        )

    def test_a_clamped_code_receives_none(self) -> None:
        # The other half of the convention, and the reason it is not simply
        # "always pass the gradient": a code the clamp actually moved is
        # saturated, and there the zero is correct.
        params = init_params(self.WEIGHT, self.SCHEME)
        # Push v past half a code so the largest weight rounds beyond q_min.
        params.v = params.v - 1.0
        grad = quantize_grad(self.WEIGHT, params, self.SCHEME, np.ones_like(self.WEIGHT))
        result = fake_quantize(self.WEIGHT, params, self.SCHEME)
        q_min, _ = self.SCHEME.code_range
        raw = round_half_to_even(self.WEIGHT / result.scale.reshape(1, 1) + params.v.reshape(1, -1))
        assert np.any(raw < q_min), "fixture no longer drives anything outside the range"
        assert np.all(grad.v.ravel()[(raw < q_min).ravel()] == 0.0)


class TestSearchedParameterization:
    """The v2 searched branch: scale = init_scale * beta, alpha inert.

    Mirrors the reference implementation's searched symmetric path, which reads
    only the max-side coefficient and ignores the observed extremes entirely.
    DOCUMENTATION.md section 5.3 is the coupling this exists to pin: the
    coefficient means something different here, and so does its range.
    """

    def scheme(self, bits: int = 3, group_size: int = 4) -> QuantScheme:
        return QuantScheme(bits=bits, group_size=group_size, scale_init=ScaleInit.SEARCHED)

    def test_hand_computed_forward(self) -> None:
        # scale -0.25: 0.9/-0.25 = -3.6 -> -4 (in range for 3 bits), and the
        # reconstruction is scale * codes exactly.
        weight = np.array([[0.9, -0.3, 0.2, 0.1]])
        params = init_params(weight, self.scheme())
        init_scale = np.array([[-0.25]])
        result = fake_quantize(weight, params, self.scheme(), init_scale=init_scale)
        # Codes come back in the grouped (rows, groups, group_size) layout,
        # exactly as the observed-range paths return them.
        np.testing.assert_array_equal(result.codes, [[[-4.0, 1.0, -1.0, 0.0]]])
        np.testing.assert_allclose(result.qdq, [[1.0, -0.25, 0.25, 0.0]])

    def test_beta_multiplies_the_searched_scale(self) -> None:
        weight = np.array([[0.9, -0.3, 0.2, 0.1]])
        params = init_params(weight, self.scheme())
        halved = QuantParams(v=params.v, alpha=params.alpha, beta=params.beta * 0.5)
        init_scale = np.array([[-0.25]])
        wide = fake_quantize(weight, halved, self.scheme(), init_scale=init_scale)
        narrow = fake_quantize(weight, params, self.scheme(), init_scale=init_scale * 0.5)
        # beta on the scale and the same factor folded into init_scale are the
        # same grid, so the codes must agree exactly.
        np.testing.assert_array_equal(wide.codes, narrow.codes)
        np.testing.assert_allclose(wide.scale, narrow.scale)

    def test_alpha_is_inert(self) -> None:
        # The reference tunes min_scale but its searched symmetric branch never
        # reads it. Same here, and pinned so a refactor cannot quietly make
        # alpha meaningful in one implementation only.
        weight = np.array([[0.9, -0.3, 0.2, 0.1]])
        params = init_params(weight, self.scheme())
        shifted = QuantParams(v=params.v, alpha=params.alpha * 0.7, beta=params.beta)
        init_scale = np.array([[-0.25]])
        a = fake_quantize(weight, params, self.scheme(), init_scale=init_scale)
        b = fake_quantize(weight, shifted, self.scheme(), init_scale=init_scale)
        np.testing.assert_array_equal(a.qdq, b.qdq)

    def test_searched_coefficient_bounds_apply(self) -> None:
        # (0.5, 1.5) under SEARCHED against (0.1, 1.0) observed: the coupling
        # DOCUMENTATION.md 5.3 calls the most confusing part of the spec.
        assert self.scheme().coefficient_bounds == (0.5, 1.5)
        assert QuantScheme(bits=3, group_size=4).coefficient_bounds == (0.1, 1.0)

    def test_missing_init_scale_is_refused(self) -> None:
        weight = np.ones((1, 4))
        with pytest.raises(ValueError, match="init_scale"):
            fake_quantize(weight, init_params(weight, self.scheme()), self.scheme())

    def test_unwanted_init_scale_is_refused(self) -> None:
        weight = np.ones((1, 4))
        observed = QuantScheme(bits=3, group_size=4)
        with pytest.raises(ValueError, match="not SEARCHED"):
            fake_quantize(
                weight,
                init_params(weight, observed),
                observed,
                init_scale=np.array([[0.1]]),
            )

    def test_searched_asymmetric_is_refused_at_construction(self) -> None:
        with pytest.raises(ValueError, match="symmetric"):
            QuantScheme(
                bits=3,
                group_size=4,
                symmetry=Symmetry.ASYMMETRIC,
                scale_init=ScaleInit.SEARCHED,
            )
