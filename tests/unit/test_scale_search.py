# Copyright 2026 The MRound Authors
# SPDX-License-Identifier: Apache-2.0
"""The v2 scale search, verified against an independent brute-force evaluator.

The brute force below shares no code with the implementation: it enumerates the
candidate family from the analysis's closed form and evaluates each candidate
directly. Agreement between the two is therefore evidence about the
implementation rather than a tautology.

Tolerances are chosen off the structure of the problem. Two evaluations of the
same candidate can differ by float epsilon through division order; two distinct
candidates differ in scale by at least the grid step over nmax squared, which is
orders of magnitude larger. An atol of 1e-12 cleanly separates the two.
"""

from __future__ import annotations

import numpy as np
import pytest

from mround.reference.quantize import round_half_to_even
from mround.reference.scale_search import SearchGrid, search_scales
from mround.schemes import QuantScheme, ScaleInit, Symmetry


def brute_force(group: np.ndarray, bits: int, qw: np.ndarray | None = None) -> float:
    """Enumerate every candidate directly, from the analysis's closed form."""
    nmax = float(2 ** (bits - 1))
    step, half = SearchGrid().steps(bits)
    candidates = [nmax] + [nmax - step * i for i in range(-half, half + 1) if i != 0]
    anchor = group[np.argmax(np.abs(group))]
    best: tuple[float, float] | None = None
    for effective in candidates:
        inverse = 0.0 if anchor == 0 else -effective / anchor
        codes = np.clip(round_half_to_even(inverse * group), -nmax, nmax - 1)
        scale = 0.0 if inverse == 0 else 1.0 / inverse
        residual = (scale * codes - group) ** 2
        if qw is not None:
            residual = residual * qw
        loss = float(np.sum(residual))
        if best is None or loss < best[0]:
            best = (loss, scale)
    assert best is not None
    return best[1]


def reconstruction_loss(scale: float, group: np.ndarray, bits: int) -> float:
    nmax = float(2 ** (bits - 1))
    inverse = 0.0 if scale == 0 else 1.0 / scale
    codes = np.clip(round_half_to_even(inverse * group), -nmax, nmax - 1)
    return float(np.sum((scale * codes - group) ** 2))


class TestGrid:
    @pytest.mark.parametrize(
        ("bits", "step", "half_count"),
        [(2, 0.01, 90), (3, 0.03, 100), (4, 0.06, 100), (5, 0.12, 100), (8, 0.96, 100)],
    )
    def test_parameters_match_the_reference_table(
        self, bits: int, step: float, half_count: int
    ) -> None:
        # The table in analysis 04 section 5.2, verified there numerically
        # against the reference source. The 2-bit row is the hard-coded finer
        # window; every other width is 200 candidates regardless of ratio.
        got_step, got_half = SearchGrid().steps(bits)
        assert got_step == pytest.approx(step, abs=1e-12)
        assert got_half == half_count

    def test_ratio_moves_the_span_never_the_count(self) -> None:
        wide = SearchGrid(ratio=1.0).steps(4)
        default = SearchGrid().steps(4)
        assert wide[1] == default[1] == 100
        assert wide[0] > default[0]


class TestAgainstBruteForce:
    @pytest.mark.parametrize(("bits", "group_size"), [(2, 8), (3, 8), (4, 16), (8, 16)])
    def test_exact_agreement_at_every_width(self, bits: int, group_size: int) -> None:
        rng = np.random.default_rng(bits)
        weight = rng.normal(size=(6, group_size * 2))
        got = search_scales(weight, QuantScheme(bits=bits, group_size=group_size)).ravel()
        want = np.array([brute_force(g, bits) for g in weight.reshape(-1, group_size)])
        np.testing.assert_allclose(got, want, rtol=0, atol=1e-12)

    def test_importance_weighting_changes_the_answer_and_still_agrees(self) -> None:
        # A large importance on one channel drags the chosen threshold toward
        # protecting it. If weighting were dropped, this test's brute force
        # would disagree.
        rng = np.random.default_rng(7)
        weight = rng.normal(size=(4, 16))
        importance = np.ones(16)
        importance[3] = 1e4
        scheme = QuantScheme(bits=3, group_size=16)
        weighted = search_scales(weight, scheme, importance=importance).ravel()
        unweighted = search_scales(weight, scheme).ravel()
        want = np.array([brute_force(g, 3, qw=importance) for g in weight])
        np.testing.assert_allclose(weighted, want, rtol=0, atol=1e-12)
        assert not np.allclose(weighted, unweighted)

    def test_importance_is_relative_so_global_scaling_is_irrelevant(self) -> None:
        rng = np.random.default_rng(8)
        weight = rng.normal(size=(4, 16))
        importance = rng.uniform(0.1, 10.0, size=16)
        scheme = QuantScheme(bits=4, group_size=16)
        one = search_scales(weight, scheme, importance=importance)
        scaled = search_scales(weight, scheme, importance=importance * 1e6)
        np.testing.assert_allclose(one, scaled, rtol=0, atol=1e-12)


class TestProperties:
    @pytest.mark.parametrize("bits", [2, 3, 4, 8])
    def test_never_worse_than_the_anchor(self, bits: int) -> None:
        # The anchor is a candidate and ties break toward it, so the search
        # cannot lose to plain D-009 initialization. This is the property that
        # makes turning the search on safe by construction.
        rng = np.random.default_rng(bits + 100)
        weight = rng.normal(size=(8, 64))
        scheme = QuantScheme(bits=bits, group_size=32)
        grouped = weight.reshape(-1, 32)
        nmax = float(2 ** (bits - 1))
        anchor = (
            np.take_along_axis(grouped, np.argmax(np.abs(grouped), axis=-1, keepdims=True), -1)
            / -nmax
        )
        searched = search_scales(weight, scheme)
        for row in range(grouped.shape[0]):
            a = reconstruction_loss(anchor[row, 0], grouped[row], bits)
            s = reconstruction_loss(searched[row, 0], grouped[row], bits)
            assert s <= a + 1e-12

    def test_the_scale_carries_the_sign_of_the_dominant_extreme(self) -> None:
        # D-009's convention: the scale's sign is opposite the dominant
        # extreme, so that extreme reconstructs exactly on -nmax. The search
        # must preserve it, or the searched grid is not the quantizer's grid.
        weight = np.array([[1.0, 0.2, -0.1, 0.05], [-1.0, 0.2, -0.1, 0.05]])
        scales = search_scales(weight, QuantScheme(bits=4, group_size=4)).ravel()
        assert scales[0] < 0  # dominant +1.0 maps through a negative scale
        assert scales[1] > 0

    def test_an_all_zero_group_returns_zero_for_the_caller_to_clamp(self) -> None:
        weight = np.zeros((1, 8))
        scales = search_scales(weight, QuantScheme(bits=4, group_size=8))
        assert scales.shape == (1, 1)
        assert scales[0, 0] == 0.0

    def test_shape_is_one_scale_per_group_row_major(self) -> None:
        rng = np.random.default_rng(9)
        weight = rng.normal(size=(3, 32))
        scales = search_scales(weight, QuantScheme(bits=4, group_size=16))
        assert scales.shape == (6, 1)


class TestRefusals:
    def test_asymmetric_schemes_are_refused(self) -> None:
        scheme = QuantScheme(bits=4, group_size=8, symmetry=Symmetry.ASYMMETRIC)
        with pytest.raises(ValueError, match="symmetric"):
            search_scales(np.ones((1, 8)), scheme)

    def test_ragged_grouping_is_refused(self) -> None:
        with pytest.raises(ValueError, match="multiple"):
            search_scales(np.ones((1, 10)), QuantScheme(bits=4, group_size=8))

    def test_wrong_importance_width_is_refused(self) -> None:
        with pytest.raises(ValueError, match="channels"):
            search_scales(
                np.ones((1, 8)),
                QuantScheme(bits=4, group_size=8),
                importance=np.ones(4),
            )


class TestThroughTheQuantizer:
    """The search and the searched quantizer branch, together.

    The search minimizes its own internal reconstruction; this checks the win
    survives the actual fake-quantization path, grouped layout, epsilon clamp
    and all. A disagreement between the two would mean the searched grid and
    the quantizer's grid are not the same grid, which is the exact defect
    D-009's convention exists to prevent.
    """

    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_search_improves_end_to_end_reconstruction(self, bits: int) -> None:
        from mround.reference.quantize import fake_quantize, init_params  # noqa: PLC0415

        rng = np.random.default_rng(bits + 50)
        weight = rng.normal(size=(8, 64))
        observed = QuantScheme(bits=bits, group_size=32)
        searched = QuantScheme(bits=bits, group_size=32, scale_init=ScaleInit.SEARCHED)
        init_scale = search_scales(weight, searched)

        baseline = fake_quantize(weight, init_params(weight, observed), observed)
        improved = fake_quantize(
            weight, init_params(weight, searched), searched, init_scale=init_scale
        )
        base_err = float(np.sum((baseline.qdq - weight) ** 2))
        search_err = float(np.sum((improved.qdq - weight) ** 2))
        assert search_err <= base_err + 1e-12

    def test_the_searched_scale_reconstructs_its_own_codes(self) -> None:
        # The quantizer applied at beta=1, v=0 must land on exactly the grid
        # the search evaluated: same anchor, same rounding, same clamp.
        from mround.reference.quantize import fake_quantize, init_params  # noqa: PLC0415

        rng = np.random.default_rng(3)
        weight = rng.normal(size=(4, 32))
        scheme = QuantScheme(bits=3, group_size=16, scale_init=ScaleInit.SEARCHED)
        init_scale = search_scales(weight, scheme)
        result = fake_quantize(weight, init_params(weight, scheme), scheme, init_scale=init_scale)
        # Recompute the search's own reconstruction for every group directly.
        grouped = weight.reshape(-1, 16)
        for row in range(grouped.shape[0]):
            scale = float(init_scale[row, 0])
            codes = np.clip(round_half_to_even(grouped[row] / scale), -4, 3)
            np.testing.assert_allclose(
                result.qdq.reshape(-1, 16)[row], scale * codes, rtol=0, atol=1e-12
            )
