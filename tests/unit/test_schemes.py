# Copyright 2026 The MRound Authors
# SPDX-License-Identifier: Apache-2.0
"""Tests for quantization schemes and tuning configuration.

Every expectation here is hand-computed from DOCUMENTATION.md section 5. Where a
test and the document disagree, one of them is wrong and the disagreement gets
resolved rather than the test adjusted until it passes.
"""

from __future__ import annotations

import pytest

from mround.schemes import (
    SUPPORTED_BITS,
    QuantScheme,
    ScaleInit,
    Symmetry,
    TuningConfig,
)


class TestQuantSchemeValidation:
    @pytest.mark.parametrize("bits", sorted(SUPPORTED_BITS))
    def test_supported_bits_are_accepted(self, bits: int) -> None:
        assert QuantScheme(bits=bits).bits == bits

    @pytest.mark.parametrize("bits", [0, 1, 9, 16, -4])
    def test_unsupported_bits_are_rejected(self, bits: int) -> None:
        with pytest.raises(ValueError, match="bits must be one of"):
            QuantScheme(bits=bits)

    def test_per_channel_group_size_is_accepted(self) -> None:
        assert QuantScheme(group_size=-1).is_per_channel

    @pytest.mark.parametrize("group_size", [0, -2, -128])
    def test_invalid_group_sizes_are_rejected(self, group_size: int) -> None:
        with pytest.raises(ValueError, match="group_size must be"):
            QuantScheme(group_size=group_size)

    def test_schemes_are_immutable(self) -> None:
        scheme = QuantScheme()
        with pytest.raises(AttributeError):
            scheme.bits = 8  # type: ignore[misc]


class TestCodeRange:
    """The symmetric range is asymmetric by one code. DOCUMENTATION.md 5.2."""

    @pytest.mark.parametrize(
        ("bits", "expected"),
        [(2, (-2, 1)), (4, (-8, 7)), (8, (-128, 127))],
    )
    def test_symmetric_range(self, bits: int, expected: tuple[int, int]) -> None:
        scheme = QuantScheme(bits=bits, symmetry=Symmetry.SYMMETRIC)
        assert scheme.code_range == expected

    @pytest.mark.parametrize(
        ("bits", "expected"),
        [(2, (0, 3)), (4, (0, 15)), (8, (0, 255))],
    )
    def test_asymmetric_range(self, bits: int, expected: tuple[int, int]) -> None:
        scheme = QuantScheme(bits=bits, symmetry=Symmetry.ASYMMETRIC)
        assert scheme.code_range == expected

    @pytest.mark.parametrize("bits", sorted(SUPPORTED_BITS))
    def test_symmetric_range_offsets_onto_unsigned_storage_exactly(self, bits: int) -> None:
        # The property the MLX export relies on: adding the fixed zero point
        # 2**(bits-1) maps the signed range onto [0, 2**bits - 1] with no
        # clipping and no wasted codes. DOCUMENTATION.md section 6.2.
        low, high = QuantScheme(bits=bits, symmetry=Symmetry.SYMMETRIC).code_range
        offset = 2 ** (bits - 1)
        assert (low + offset, high + offset) == (0, 2**bits - 1)

    @pytest.mark.parametrize("bits", sorted(SUPPORTED_BITS))
    def test_range_holds_exactly_two_to_the_bits_codes(self, bits: int) -> None:
        for symmetry in Symmetry:
            low, high = QuantScheme(bits=bits, symmetry=symmetry).code_range
            assert high - low + 1 == 2**bits

    @pytest.mark.parametrize(
        ("bits", "shortfall"),
        [(8, 1 / 128), (4, 1 / 8), (2, 1 / 2)],
    )
    def test_the_unsigned_scale_would_clip_the_dominant_extreme(
        self, bits: int, shortfall: float
    ) -> None:
        # The naive symmetric scale, s = max_v / 2**(b-1) with a positive
        # sign, maps the group's largest-magnitude weight one code above the
        # range, where it clips to (2**(b-1) - 1) / 2**(b-1) of its magnitude:
        # half at 2 bits. MRound does not use that scale. The signed scale of
        # DOCUMENTATION.md section 5.2.1 (D-009, which closed Q-006) puts the
        # dominant extreme on the code with the larger magnitude instead, and
        # test_reference_quantize.py pins that it reconstructs exactly. This
        # test keeps the size of the shortfall that choice avoids on record.
        _, high = QuantScheme(bits=bits, symmetry=Symmetry.SYMMETRIC).code_range
        reconstruction = high / 2 ** (bits - 1)
        assert 1.0 - reconstruction == pytest.approx(shortfall)


class TestPacking:
    @pytest.mark.parametrize("bits", [2, 4, 8])
    def test_widths_dividing_32_pack_evenly(self, bits: int) -> None:
        assert QuantScheme(bits=bits).packs_evenly

    @pytest.mark.parametrize("bits", [3, 5, 6, 7])
    def test_other_widths_need_the_bitstream_layout(self, bits: int) -> None:
        # 6 does not divide 32, which is the case worth naming explicitly since
        # it is the one people assume goes the other way.
        assert not QuantScheme(bits=bits).packs_evenly


class TestGroupsPerRow:
    def test_even_division(self) -> None:
        assert QuantScheme(group_size=128).groups_per_row(4096) == 32

    def test_per_channel_is_one_group(self) -> None:
        assert QuantScheme(group_size=-1).groups_per_row(4096) == 1

    def test_uneven_division_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="not divisible"):
            QuantScheme(group_size=128).groups_per_row(4000)


class TestWithBits:
    def test_only_the_bit_width_changes(self) -> None:
        # Mixed-precision search varies the width and nothing else. The
        # non-default fields here must form a VALID scheme: searched demands
        # symmetric, so asymmetric exercises the symmetry field on its own
        # elsewhere and this test uses the searched-symmetric pair.
        base = QuantScheme(
            bits=4,
            group_size=64,
            scale_init=ScaleInit.SEARCHED,
        )
        derived = base.with_bits(2)
        assert derived.bits == 2
        assert derived.group_size == base.group_size
        assert derived.symmetry == base.symmetry
        assert derived.scale_init == base.scale_init


class TestLearningRate:
    """DOCUMENTATION.md section 5.4."""

    @pytest.mark.parametrize(("bits", "constant"), [(8, 1.0), (4, 1.0), (3, 2.0), (2, 2.0)])
    def test_derived_rate_scales_inversely_with_iterations(
        self, bits: int, constant: float
    ) -> None:
        config = TuningConfig(iters=200)
        assert config.resolved_lr(bits) == pytest.approx(constant / 200)

    def test_explicit_rate_overrides_the_derivation(self) -> None:
        config = TuningConfig(iters=200, lr=0.01)
        assert config.resolved_lr(2) == 0.01
        assert config.resolved_lr(8) == 0.01

    @pytest.mark.parametrize("iters", [50, 200, 500, 1000])
    def test_total_excursion_is_independent_of_the_step_count(self, iters: int) -> None:
        # The property the derivation exists to give: with a rate decaying
        # linearly to zero, the sum of step sizes is about c/2 regardless of
        # how many steps are taken. For 4 bits and above that is 0.5, exactly
        # the half-code bound on the rounding perturbation.
        config = TuningConfig(iters=iters)
        lr = config.resolved_lr(4)
        excursion = sum(lr * (1 - step / iters) for step in range(iters))
        assert excursion == pytest.approx(0.5, abs=0.01)

    def test_low_bit_widths_get_twice_the_excursion(self) -> None:
        config = TuningConfig(iters=200)
        assert config.resolved_lr(2) == pytest.approx(2 * config.resolved_lr(4))

    def test_minmax_rate_defaults_to_the_main_rate(self) -> None:
        config = TuningConfig(iters=200)
        assert config.resolved_minmax_lr(4) == config.resolved_lr(4)

    def test_a_separate_minmax_rate_is_refused_rather_than_ignored(self) -> None:
        # No tuning loop reads minmax_lr. Accepting it and then applying lr to
        # the coefficients anyway would be a silent substitution, so the
        # configuration refuses at construction, where the caller can see it.
        with pytest.raises(ValueError, match="minmax_lr is not implemented"):
            TuningConfig(iters=200, minmax_lr=0.02)

    def test_gradient_accumulation_is_refused_rather_than_ignored(self) -> None:
        with pytest.raises(ValueError, match="gradient_accumulate_steps"):
            TuningConfig(iters=200, gradient_accumulate_steps=4)
        assert TuningConfig(iters=200, gradient_accumulate_steps=1).gradient_accumulate_steps == 1

    def test_string_enum_values_are_coerced_at_construction(self) -> None:
        # Every consumer tests the enums by identity; a plain string compares
        # equal to the member without being it and would have selected the
        # unsigned grid while reporting itself as symmetric.
        scheme = QuantScheme(bits=4, group_size=64, symmetry="sym", scale_init="searched")  # type: ignore[arg-type]
        assert scheme.symmetry is Symmetry.SYMMETRIC
        assert scheme.scale_init is ScaleInit.SEARCHED
        assert scheme.code_range == (-8, 7)
        with pytest.raises(ValueError, match="signed"):
            QuantScheme(bits=4, symmetry="signed")  # type: ignore[arg-type]


class TestOutlierSuppression:
    """DOCUMENTATION.md section 5.5."""

    @pytest.mark.parametrize("bits", [2, 3])
    def test_enabled_below_four_bits_by_default(self, bits: int) -> None:
        assert TuningConfig().resolved_suppress_outliers(bits)

    @pytest.mark.parametrize("bits", [4, 5, 6, 8])
    def test_disabled_at_four_bits_and_above_by_default(self, bits: int) -> None:
        # This is why a 4-bit first milestone does not need the outlier loss,
        # which makes Phase 1 smaller than the paper implies.
        assert not TuningConfig().resolved_suppress_outliers(bits)

    @pytest.mark.parametrize("explicit", [True, False])
    def test_explicit_setting_overrides_the_default(self, explicit: bool) -> None:
        config = TuningConfig(suppress_outliers=explicit)
        assert config.resolved_suppress_outliers(2) is explicit
        assert config.resolved_suppress_outliers(8) is explicit


class TestTuningConfigValidation:
    @pytest.mark.parametrize("iters", [0, -1])
    def test_non_positive_iterations_are_rejected(self, iters: int) -> None:
        with pytest.raises(ValueError, match="iters must be positive"):
            TuningConfig(iters=iters)

    def test_non_positive_batch_size_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="batch_size must be positive"):
            TuningConfig(batch_size=0)

    def test_sample_count_below_batch_size_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="smaller than"):
            TuningConfig(n_samples=4, batch_size=8)
