# Copyright 2026 The MRound Authors
# SPDX-License-Identifier: Apache-2.0
"""Tests for reference bit packing.

Packing is one of the few places where an exhaustive test is both cheap and
conclusive, so these tests lean on that rather than on inspection. A packing bug
produces a checkpoint that loads without complaint and generates nonsense, which
is the worst failure mode available.
"""

from __future__ import annotations

import numpy as np
import pytest

from mround.reference.packing import pack_codes, packed_width, unpack_codes
from mround.schemes import SUPPORTED_BITS

EVEN_BITS = [b for b in sorted(SUPPORTED_BITS) if 32 % b == 0]
STREAM_BITS = [b for b in sorted(SUPPORTED_BITS) if 32 % b != 0]


class TestPackedWidth:
    @pytest.mark.parametrize(("bits", "cols", "expected"), [(2, 64, 4), (4, 64, 8), (8, 64, 16)])
    def test_even_layout(self, bits: int, cols: int, expected: int) -> None:
        assert packed_width(cols, bits) == expected

    @pytest.mark.parametrize("bits", STREAM_BITS)
    def test_stream_layout_is_exactly_bits_words_per_32_codes(self, bits: int) -> None:
        # The defining property of the cross-word layout: 32 codes x b bits
        # equals 32b bits equals exactly b words of 32 bits, with nothing wasted.
        assert packed_width(32, bits) == bits
        assert packed_width(320, bits) == bits * 10

    @pytest.mark.parametrize("bits", sorted(SUPPORTED_BITS))
    def test_no_space_is_wasted(self, bits: int) -> None:
        cols = 32 * 4
        assert packed_width(cols, bits) * 32 == cols * bits

    def test_untileable_row_is_rejected(self) -> None:
        # 8 bits packs 4 codes per word, so 62 leaves a partial word.
        with pytest.raises(ValueError, match="not a multiple of 4"):
            packed_width(62, 8)
        # The bitstream layout needs whole groups of 32 codes, and 64 is a
        # multiple of 32, so use something that is not.
        with pytest.raises(ValueError, match="not a multiple of 32"):
            packed_width(48, 3)

    def test_unsupported_bits_rejected(self) -> None:
        with pytest.raises(ValueError, match="bits must be one of"):
            packed_width(64, 16)


class TestRoundTrip:
    @pytest.mark.parametrize("bits", sorted(SUPPORTED_BITS))
    def test_random_codes_round_trip_exactly(self, bits: int) -> None:
        gen = np.random.default_rng(bits)
        codes = gen.integers(0, 2**bits, size=(7, 32 * 5), dtype=np.int64)
        packed = pack_codes(codes, bits)
        assert packed.dtype == np.uint32
        assert np.array_equal(unpack_codes(packed, codes.shape[1], bits), codes)

    @pytest.mark.parametrize("bits", sorted(SUPPORTED_BITS))
    def test_extremes_round_trip(self, bits: int) -> None:
        # Saturated codes exercise the high bit of every field, which is where
        # a sign-extension or mask error would show up.
        hi = 2**bits - 1
        codes = np.array([[0, hi] * 16, [hi, 0] * 16], dtype=np.int64)
        packed = pack_codes(codes, bits)
        assert np.array_equal(unpack_codes(packed, codes.shape[1], bits), codes)

    @pytest.mark.parametrize("bits", sorted(SUPPORTED_BITS))
    def test_all_saturated_round_trips(self, bits: int) -> None:
        hi = 2**bits - 1
        codes = np.full((3, 64), hi, dtype=np.int64)
        packed = pack_codes(codes, bits)
        assert np.array_equal(unpack_codes(packed, 64, bits), codes)


class TestLayout:
    @pytest.mark.parametrize("bits", EVEN_BITS)
    def test_element_zero_is_in_the_least_significant_field(self, bits: int) -> None:
        per_word = 32 // bits
        codes = np.zeros((1, per_word), dtype=np.int64)
        codes[0, 0] = 1
        packed = pack_codes(codes, bits)
        assert packed[0, 0] == 1

    @pytest.mark.parametrize("bits", EVEN_BITS)
    def test_element_one_sits_at_the_next_field(self, bits: int) -> None:
        per_word = 32 // bits
        codes = np.zeros((1, per_word), dtype=np.int64)
        codes[0, 1] = 1
        packed = pack_codes(codes, bits)
        assert packed[0, 0] == 1 << bits

    def test_four_bit_layout_is_little_endian_nibbles(self) -> None:
        # The canonical spot check: 0..7 at 4 bits packs to 0x76543210.
        codes = np.arange(8, dtype=np.int64).reshape(1, 8)
        assert pack_codes(codes, 4)[0, 0] == 0x76543210

    @pytest.mark.parametrize("bits", STREAM_BITS)
    def test_stream_element_positions(self, bits: int) -> None:
        # Element i must occupy absolute bits [i*bits, (i+1)*bits) of a
        # little-endian word stream. Setting one element to all ones and
        # reading the stream back as a bit array proves the position.
        for element in (0, 1, 5, 31):
            codes = np.zeros((1, 32), dtype=np.int64)
            codes[0, element] = 2**bits - 1
            packed = pack_codes(codes, bits)
            stream = 0
            for word_index, word in enumerate(packed[0]):
                stream |= int(word) << (32 * word_index)
            expected = (2**bits - 1) << (element * bits)
            assert stream == expected, f"element {element} misplaced at {bits} bits"

    @pytest.mark.parametrize("bits", STREAM_BITS)
    def test_stream_elements_straddle_word_boundaries(self, bits: int) -> None:
        # The property that distinguishes this layout: at least one element in
        # every 32 must cross a word boundary, otherwise the layout would be the
        # simple one.
        straddlers = [i for i in range(32) if (i * bits) // 32 != ((i + 1) * bits - 1) // 32]
        assert straddlers, f"{bits} bits should produce straddling elements"


class TestValidation:
    def test_out_of_range_codes_are_rejected(self) -> None:
        # Masking instead of refusing would corrupt weights silently, so the
        # contract is to fail loudly.
        codes = np.array([[0, 16, 3, 4, 5, 6, 7, 8]], dtype=np.int64)
        with pytest.raises(ValueError, match=r"codes must lie in \[0, 15\]"):
            pack_codes(codes, 4)

    def test_negative_codes_are_rejected(self) -> None:
        codes = np.array([[-1, 2, 3, 4, 5, 6, 7, 8]], dtype=np.int64)
        with pytest.raises(ValueError, match="codes must lie in"):
            pack_codes(codes, 4)

    def test_non_matrix_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="2-D code matrix"):
            pack_codes(np.zeros(8, dtype=np.int64), 4)

    def test_wrong_word_count_is_rejected_on_unpack(self) -> None:
        packed = np.zeros((2, 3), dtype=np.uint32)
        with pytest.raises(ValueError, match="expected 8 words per row"):
            unpack_codes(packed, 64, 4)


class TestSymmetricStorageOffset:
    @pytest.mark.parametrize("bits", sorted(SUPPORTED_BITS))
    def test_signed_range_offsets_into_unsigned_storage_exactly(self, bits: int) -> None:
        # Symmetric schemes produce signed codes and storage needs unsigned
        # ones, so a fixed offset of 2**(bits-1) is applied at pack time. This
        # checks the offset covers the full range with nothing wasted and
        # nothing clipped, then round-trips it.
        half = 2 ** (bits - 1)
        signed = np.arange(-half, half, dtype=np.int64)
        unsigned = signed + half
        assert unsigned.min() == 0
        assert unsigned.max() == 2**bits - 1

        cols = len(unsigned)
        pad = (-cols) % 32
        if pad:
            unsigned = np.concatenate([unsigned, np.zeros(pad, dtype=np.int64)])
        row = unsigned.reshape(1, -1)
        packed = pack_codes(row, bits)
        recovered = unpack_codes(packed, row.shape[1], bits) - half
        assert np.array_equal(recovered[0, :cols], signed)


class TestMLXLayoutCompatibility:
    """The packed layout must match MLX's, byte for byte.

    This is the load-bearing interoperability claim in the whole formats layer.
    A checkpoint packed to a layout MLX does not share loads without complaint,
    passes every shape validation, and generates nonsense, which is the worst
    failure mode available to a quantizer.

    So the claim is checked rather than asserted, by implementing MLX's own
    documented algorithm here from its description and comparing. Two
    independent constructions agreeing is evidence; one construction and a
    comment saying it matches is not. Deliberately in the reference suite rather
    than the MLX parity suite, so it runs everywhere and cannot be skipped on a
    machine without Metal. See MEMORY.md D-017.
    """

    # mx.quantize rejects 7-bit outright, and accepts nothing outside this set.
    MLX_BITS = (2, 3, 4, 5, 6, 8)

    @staticmethod
    def mlx_style_pack(codes: np.ndarray, bits: int) -> np.ndarray:
        """MLX's algorithm, written from its description rather than ported.

        Explode each code into individual bits, least significant first,
        reshape the resulting stream to width 32, and reassemble.
        """
        rows = codes.shape[0]
        exploded = (
            (codes[:, :, None].astype(np.uint32) >> np.arange(bits, dtype=np.uint32)) & 1
        ).reshape(rows, -1)
        words = exploded.reshape(rows, -1, 32).astype(np.uint64)
        place = np.uint64(1) << np.arange(32, dtype=np.uint64)
        packed: np.ndarray = (words * place).sum(axis=-1).astype(np.uint32)
        return packed

    @pytest.mark.parametrize("bits", MLX_BITS)
    def test_packing_is_byte_identical_to_mlx(self, bits: int) -> None:
        codes = np.random.default_rng(bits).integers(0, 2**bits, size=(8, 128), dtype=np.uint32)
        ours = pack_codes(codes, bits)
        theirs = self.mlx_style_pack(codes, bits)
        assert ours.shape == theirs.shape
        mismatched = int(np.count_nonzero(ours != theirs))
        assert mismatched == 0, (
            f"{mismatched} of {ours.size} words differ from MLX's layout at {bits} bits; "
            "a checkpoint packed this way would load and produce nonsense"
        )

    @pytest.mark.parametrize("bits", MLX_BITS)
    @pytest.mark.parametrize("in_features", [32, 64, 128, 4096])
    def test_packed_width_matches_mlxs_formula(self, in_features: int, bits: int) -> None:
        # MLX sizes the packed axis as in_features * bits / 32 with no padding
        # case, since group_size times bits is always a multiple of 32 for the
        # combinations it accepts.
        assert packed_width(in_features, bits) == in_features * bits // 32

    def test_seven_bits_packs_here_but_has_no_mlx_representation(self) -> None:
        # 7 is supported by this layout and rejected by mx.quantize. The
        # asymmetry is deliberate: packing serves GGUF as well, and the export
        # layer is where the narrower constraint belongs. Pinned so that a
        # future widening of one is a decision rather than a surprise.
        assert 7 in SUPPORTED_BITS
        assert 7 not in self.MLX_BITS
        codes = np.random.default_rng(7).integers(0, 128, size=(4, 64), dtype=np.uint32)
        assert np.array_equal(unpack_codes(pack_codes(codes, 7), 64, 7), codes)
