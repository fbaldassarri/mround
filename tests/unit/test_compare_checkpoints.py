# Copyright 2026 The MRound Authors
# SPDX-License-Identifier: Apache-2.0
"""The code by code comparison of two fake quantized checkpoints.

Built on synthetic groups whose scales and codes are known, so every count the
script prints is checked against the arithmetic that produced the values, and
on a safetensors file written by hand, so the reader is checked against the
format rather than against a library that would hide a misread.
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from compare_checkpoints import (
    coarse_sides,
    compare,
    compare_layer,
    find_checkpoint,
    implied_scales,
    linear_weights,
    main,
    read_header,
    read_tensor,
    same_architecture,
)


def _groups(codes: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Dequantized values the way a fake export holds them, in float32."""
    return (codes.astype(np.float32) * scales.astype(np.float32)[:, None]).astype(np.float64)


class TestImpliedScales:
    def test_recovers_the_scale_from_adjacent_codes(self) -> None:
        codes = np.array([[-2, -1, 0, 1, 1, 0, -2, -1]])
        scales = np.array([0.0125])
        assert implied_scales(_groups(codes, scales)) == pytest.approx([0.0125], rel=1e-6)

    def test_a_group_of_zeros_has_no_scale(self) -> None:
        assert np.isinf(implied_scales(np.zeros((1, 8))))

    def test_the_gap_serves_when_no_code_is_one(self) -> None:
        # Codes 2 and 3 only: the smallest magnitude is two scales, the gap is one.
        codes = np.array([[2, 3, 2, 3, 3, 2, 2, 3]])
        assert implied_scales(_groups(codes, np.array([0.5]))) == pytest.approx([0.5])


class TestCompareLayer:
    def test_counts_flips_on_one_lattice(self) -> None:
        rng = np.random.default_rng(0)
        codes_a = rng.integers(-2, 2, size=(4, 64))
        codes_b = codes_a.copy()
        codes_b[0, :5] += 1  # five codes one step up
        codes_b[1, 0] = -2  # two codes that move further
        codes_b[1, 1] = 1
        codes_a[1, 0], codes_a[1, 1] = 1, -2  # both three steps apart
        scales = np.array([0.01, 0.02, 0.03, 0.04])
        left, right = _groups(codes_a, scales), _groups(codes_b, scales)
        # Every group needs a code of magnitude one for the scale to be exact.
        assert all((np.abs(codes_a[i]) == 1).any() for i in range(4))

        result = compare_layer("layer", left, right, 64)
        summary = result.summary
        assert summary.groups == 4
        assert summary.same_lattice == 4
        assert summary.identical_scale == 4
        assert summary.off_lattice == 0
        assert summary.codes == 256
        assert summary.flipped == 7
        assert summary.one_apart == 5
        assert result.width == 2
        assert result.differing == 7
        assert result.elements == 256
        assert result.relative_disagreement > 0

    def test_a_drifted_scale_is_a_different_lattice(self) -> None:
        codes = np.array([[-2, -1, 0, 1] * 16, [1, 0, -1, 1] * 16])
        left = _groups(codes, np.array([0.01, 0.01]))
        right = _groups(codes, np.array([0.01, 0.011]))  # second group's clip drifted
        summary = compare_layer("layer", left, right, 64).summary
        assert summary.same_lattice == 1
        assert summary.identical_scale == 1
        assert summary.codes == 64
        assert summary.flipped == 0
        assert summary.scale_drift == pytest.approx([0.1], rel=1e-3)

    def test_a_layer_with_no_shared_lattice_still_has_its_width(self) -> None:
        # Two tunings of one layer whose every scale moved: no group shares a
        # lattice, so no code is compared, and the layer is still the 3 bit
        # layer its left side says it is. Read from the shared groups alone
        # the width came out as 0, which the report then printed as a width.
        codes = np.array([[-4, -1, 3, 1] * 16, [2, -3, 0, 1] * 16])
        left = _groups(codes, np.array([0.01, 0.02]))
        right = _groups(codes, np.array([0.012, 0.023]))
        result = compare_layer("layer", left, right, 64)
        assert result.summary.same_lattice == 0
        assert result.summary.codes == 0
        assert result.width == 3

    def test_the_width_comes_from_every_group_of_the_left_side(self) -> None:
        # The one group that shares a lattice holds small codes only; the
        # layer's width comes from every group of the left side, not from it.
        codes = np.array([[-1, 0, 1, 0] * 16, [-8, 7, 1, -1] * 16])
        left = _groups(codes, np.array([0.01, 0.02]))
        right = _groups(codes, np.array([0.01, 0.025]))
        result = compare_layer("layer", left, right, 64)
        assert result.summary.same_lattice == 1
        assert result.width == 4

    def test_the_width_is_the_left_side_whatever_the_right_holds(self) -> None:
        # The two sides at different widths and on different lattices: the
        # width reported is the left's, read on the left's own scales, so
        # swapping the sides swaps the answer.
        narrow = np.array([[-2, -1, 0, 1] * 16, [1, 0, -1, -2] * 16])
        wide = np.array([[-8, 7, 1, -1] * 16, [1, 0, -1, 1] * 16])
        two_bit = _groups(narrow, np.array([0.01, 0.02]))
        four_bit = _groups(wide, np.array([0.004, 0.009]))
        assert compare_layer("layer", two_bit, four_bit, 64).width == 2
        assert compare_layer("layer", four_bit, two_bit, 64).width == 4

    def test_the_width_does_not_wait_for_the_right_side_to_have_a_lattice(self) -> None:
        # The left side's widest group against a group of zeros on the right,
        # which has no lattice at all: the width is still read from the left.
        codes = np.array([[-8, 7, 1, -1] * 16, [1, 0, -1, 1] * 16])
        left = _groups(codes, np.array([0.01, 0.02]))
        right = left.copy()
        right[0] = 0.0
        assert compare_layer("layer", left, right, 64).width == 4

    def test_one_wide_code_among_thousands_sets_the_width(self) -> None:
        # The width is the widest code's and not a typical one's: a single -8
        # among 4,096 codes of magnitude one or less makes a 4 bit layer.
        codes = np.tile(np.array([1, 0, -1, 1] * 16), (64, 1))
        codes[37, 11] = -8
        left = _groups(codes, np.full(64, 0.01))
        assert compare_layer("layer", left, left.copy(), 64).width == 4

    @pytest.mark.parametrize(
        ("row", "width"),
        [
            ([-1, 0, 1, 0], 2),
            ([0, 0, 0, 0], 2),
            ([-4, -1, 1, 0], 3),
            ([4, 1, -1, 0], 3),
            ([-3, 3, 1, 0], 3),
            ([-5, 1, 0, -1], 4),
            ([5, 1, 0, -1], 4),
            ([-128, 127, 1, -1], 8),
        ],
    )
    def test_the_width_is_the_narrowest_range_that_holds_the_widest_code(
        self, row: list[int], width: int
    ) -> None:
        # A width of w bits holds codes of magnitude up to 2 ** (w - 1) on
        # either side, because the reference signs its scale and the scale
        # recovered here is a magnitude. So 4 and -4 are both 3 bit codes, 5
        # and -5 both need 4 bits, and 2 bits is the floor, for a group of
        # ones as for a group of zeros.
        codes = np.array([row * 16, [1, 0, -1, 1] * 16])
        left = _groups(codes, np.array([0.01, 0.02]))
        assert compare_layer("layer", left, left.copy(), 64).width == width

    def test_a_code_read_off_a_gap_is_rounded_before_it_is_ranged(self) -> None:
        # No code of magnitude one here, so the scale is recovered from the
        # gap between 2 and 3, which in float32 is not exactly the scale, and
        # -4 comes out a little past it. Unrounded, that asks for a fourth bit.
        codes = np.array([[-4, 3, 2, 2] * 16, [1, 0, -1, 1] * 16])
        left = _groups(codes, np.array([0.019, 0.02]))
        assert abs(left[0, 0] / implied_scales(left)[0]) > 2 ** (3 - 1)
        assert compare_layer("layer", left, left.copy(), 64).width == 3

    @pytest.mark.parametrize("stray", [float("inf"), float("-inf")])
    def test_an_infinite_weight_does_not_decide_the_width(self, stray: float) -> None:
        # Its group keeps its scale, so the weight reads as a code larger than
        # any width: it would run the search to its bound, and before the
        # search had a bound it never let it end.
        codes = np.array([[-8, 7, 1, -1] * 16, [1, 0, -1, 1] * 16])
        left = _groups(codes, np.array([0.01, 0.02]))
        left[1, 5] = stray
        with np.errstate(invalid="ignore"):
            assert compare_layer("layer", left, left.copy(), 64).width == 4

    def test_a_nan_takes_its_group_out_of_the_width(self) -> None:
        # A group holding a NaN has no scale of its own, so the width is read
        # from the other groups: the layer's own width when the NaN sits in a
        # narrow group, the remaining groups' when it sits in the widest.
        codes = np.array([[-8, 7, 1, -1] * 16, [1, 0, -1, 1] * 16])
        clean = _groups(codes, np.array([0.01, 0.02]))
        for group, width in ((1, 4), (0, 2)):
            left = clean.copy()
            left[group, 5] = np.nan
            with np.errstate(invalid="ignore"):
                assert compare_layer("layer", left, left.copy(), 64).width == width

    def test_a_group_with_a_nan_does_not_lend_its_values_as_codes(self) -> None:
        # Read at the fallback scale of one, the other values of such a group
        # would pass for codes: here 17, which would make a 2 bit layer a 6
        # bit one.
        codes = np.array([[-2, -1, 0, 1] * 16, [1, 0, -1, 1] * 16])
        left = _groups(codes, np.array([0.01, 17.0]))
        left[1, 5] = np.nan
        with np.errstate(invalid="ignore"):
            assert compare_layer("layer", left, left.copy(), 64).width == 2

    def test_a_group_with_a_nan_is_on_no_lattice_even_against_itself(self) -> None:
        # Two groups of zeros count as one lattice, holding code zero on any.
        # A group with a NaN on both sides has no scale on either and counts
        # as none, so its values are not compared as codes, whatever they are.
        codes = np.array([[-2, -1, 0, 1] * 16, [1, 0, -1, 1] * 16, [0, 0, 0, 0] * 16])
        left = _groups(codes, np.array([0.01, 17.0, 1.0]))
        left[1, 5] = np.nan
        right = left.copy()
        right[1, :5] = -left[1, :5]
        with np.errstate(invalid="ignore"):
            summary = compare_layer("layer", left, right, 64).summary
        assert summary.groups == 3
        assert summary.same_lattice == 2
        assert summary.identical_scale == 2
        assert summary.codes == 128
        assert summary.flipped == 0
        assert summary.scale_drift == []

    def test_zeros_on_one_side_alone_are_not_one_lattice(self) -> None:
        # A group of zeros counts as one lattice only with another group of
        # zeros. Against a group that has a scale it shares none, whichever
        # side it is on, and having no scale it adds no scale distance.
        codes = np.array([[-2, -1, 0, 1] * 16, [1, 0, -1, 1] * 16])
        full = _groups(codes, np.array([0.01, 0.02]))
        emptied = full.copy()
        emptied[0] = 0.0
        for left, right in ((full, emptied), (emptied, full)):
            summary = compare_layer("layer", left, right, 64).summary
            assert summary.same_lattice == 1
            assert summary.identical_scale == 1
            assert summary.codes == 64
            assert summary.flipped == 0
            assert summary.scale_drift == []

    def test_zeros_against_a_group_with_a_nan_are_not_one_lattice(self) -> None:
        # Neither side has a scale, and for different reasons: the zeros are
        # code zero on any lattice, the group with a NaN is on none.
        codes = np.array([[-2, -1, 0, 1] * 16, [1, 0, -1, 1] * 16])
        full = _groups(codes, np.array([0.01, 0.02]))
        emptied, holed = full.copy(), full.copy()
        emptied[0] = 0.0
        holed[0, 5] = np.nan
        for left, right in ((emptied, holed), (holed, emptied)):
            with np.errstate(invalid="ignore"):
                summary = compare_layer("layer", left, right, 64).summary
            assert summary.same_lattice == 1
            assert summary.identical_scale == 1
            assert summary.codes == 64

    def test_identical_sides_flip_nothing(self) -> None:
        codes = np.array([[-4, -1, 3, 1] * 16])
        left = _groups(codes, np.array([0.5]))
        result = compare_layer("layer", left, left.copy(), 64)
        assert result.summary.flipped == 0
        assert result.differing == 0
        assert result.width == 3
        assert result.relative_disagreement == 0.0

    def test_the_disagreement_is_relative_to_the_left_side(self) -> None:
        # The root mean square difference over the left side's own root mean
        # square, so the figure depends on the order of the two sides: a
        # checkpoint against its double is 1.0 one way and 0.5 the other.
        codes = np.array([[-2, -1, 0, 1] * 16, [1, 0, -1, 1] * 16])
        small = _groups(codes, np.array([0.01, 0.02]))
        double = 2.0 * small
        there = compare_layer("layer", small, double, 64).relative_disagreement
        back = compare_layer("layer", double, small, 64).relative_disagreement
        assert there == pytest.approx(1.0)
        assert back == pytest.approx(0.5)

    def test_one_unit_in_the_last_place_is_a_weight_that_differs(self) -> None:
        # The count of weights that differ has no tolerance. "0 differ" is how
        # a checkpoint produced twice is told from one that is merely near.
        codes = np.array([[-2, -1, 0, 1] * 16, [1, 0, -1, 1] * 16])
        left = _groups(codes, np.array([0.01, 0.02]))
        right = left.copy()
        right[1, 3] = np.nextafter(np.float32(right[1, 3]), np.float32(1.0))
        assert right[1, 3] != left[1, 3]
        result = compare_layer("layer", left, right, 64)
        assert result.differing == 1
        assert result.elements == 128

    def test_refuses_shapes_that_differ(self) -> None:
        with pytest.raises(ValueError, match="shapes differ"):
            compare_layer("layer", np.zeros((2, 64)), np.zeros((3, 64)), 64)

    def test_refuses_a_group_size_that_does_not_divide(self) -> None:
        with pytest.raises(ValueError, match="do not divide"):
            compare_layer("layer", np.zeros((2, 60)), np.zeros((2, 60)), 64)


def _bfloat16_bits(array: np.ndarray) -> bytes:
    """The top halves of the float32 words, rounded to nearest even as torch does."""
    words = array.astype(np.float32).view(np.uint32).astype(np.uint64)
    rounding = ((words >> 16) & 1) + 0x7FFF
    return ((words + rounding) >> 16).astype(np.uint16).tobytes()


def _write_safetensors(path: Path, tensors: dict[str, np.ndarray], dtype: str = "F32") -> None:
    header: dict[str, object] = {}
    payload = b""
    for name, array in tensors.items():
        raw = _bfloat16_bits(array) if dtype == "BF16" else array.astype(np.float32).tobytes()
        header[name] = {
            "dtype": dtype,
            "shape": list(array.shape),
            "data_offsets": [len(payload), len(payload) + len(raw)],
        }
        payload += raw
    encoded = json.dumps(header).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


def _checkpoint(
    directory: Path, tensors: dict[str, np.ndarray], dtype: str = "F32", **config: object
) -> Path:
    directory.mkdir(parents=True)
    settings = {"model_type": "qwen2", "hidden_size": 64, "num_hidden_layers": 1, "vocab_size": 10}
    settings.update(config)
    (directory / "config.json").write_text(json.dumps(settings), encoding="utf-8")
    _write_safetensors(directory / "model.safetensors", tensors, dtype)
    return directory


class TestReader:
    def test_reads_back_what_was_written(self, tmp_path: Path) -> None:
        tensors = {
            "model.layers.0.mlp.up_proj.weight": np.arange(128, dtype=np.float32).reshape(2, 64),
            "model.layers.0.input_layernorm.weight": np.ones(64, dtype=np.float32),
            "model.embed_tokens.weight": np.zeros((10, 64), dtype=np.float32),
        }
        path = tmp_path / "model.safetensors"
        _write_safetensors(path, tensors)
        entries = read_header(path)
        assert [e.name for e in entries] == sorted(tensors)
        for entry in entries:
            np.testing.assert_array_equal(read_tensor(path, entry), tensors[entry.name])
        # Only the block's two dimensional weights count as linear layers.
        assert [e.name for e in linear_weights(entries)] == ["model.layers.0.mlp.up_proj.weight"]

    def test_decodes_bfloat16_as_torch_writes_it(self, tmp_path: Path) -> None:
        # 1.0 and 0.5 are exact in bfloat16; 1/3 is not and rounds to nearest even.
        values = np.array([[1.0, 0.5, 1.0 / 3.0, -2.5]], dtype=np.float32)
        path = tmp_path / "model.safetensors"
        _write_safetensors(path, {"model.layers.0.mlp.up_proj.weight": values}, "BF16")
        (entry,) = read_header(path)
        assert entry.dtype == "BF16"
        decoded = read_tensor(path, entry)
        assert decoded[0, 0] == 1.0
        assert decoded[0, 1] == 0.5
        assert decoded[0, 3] == -2.5
        assert decoded[0, 2] == pytest.approx(1.0 / 3.0, rel=2**-8)
        assert decoded[0, 2] != 1.0 / 3.0

    def test_finds_the_checkpoint_under_a_parent(self, tmp_path: Path) -> None:
        inner = _checkpoint(tmp_path / "reference" / "model-w2g64", {})
        assert find_checkpoint(tmp_path / "reference") == inner
        assert find_checkpoint(inner) == inner
        with pytest.raises(FileNotFoundError):
            find_checkpoint(tmp_path)


class TestMain:
    def test_end_to_end_on_two_small_checkpoints(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        codes = np.array([[-2, -1, 0, 1] * 16, [1, -1, 0, 0] * 16])
        moved = codes.copy()
        moved[0, 0] = -1
        name = "model.layers.0.self_attn.q_proj.weight"
        left = _checkpoint(tmp_path / "left", {name: _groups(codes, np.array([0.1, 0.2]))})
        right = _checkpoint(tmp_path / "right", {name: _groups(moved, np.array([0.1, 0.2]))})
        assert same_architecture(left, right) is None
        reports = compare(left, right, 64)
        assert len(reports) == 1
        assert reports[0].summary.flipped == 1
        assert main([str(left), str(right), "--group-size", "64"]) == 0
        out = capsys.readouterr().out
        assert "flipped 1 (0.78%)" in out
        assert "2 bit, 1 layers" in out

    def test_a_bfloat16_side_gets_the_float_figures_only(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        codes = np.array([[-2, -1, 0, 1] * 16, [1, -1, 0, 0] * 16])
        name = "model.layers.0.self_attn.q_proj.weight"
        values = _groups(codes, np.array([0.1, 0.2]))
        held = _checkpoint(tmp_path / "held", {name: values})
        default = _checkpoint(tmp_path / "default", {name: values}, "BF16")
        assert coarse_sides(held, held) == []
        assert coarse_sides(held, default) == ["right (BF16)"]
        reports = compare(held, default, 64)
        # No lattice was read, so no code was counted, but the float figures
        # stand: the bfloat16 rounding of 0.1 and 0.2 multiples is small.
        assert reports[0].width == 0
        assert reports[0].summary.codes == 0
        assert reports[0].summary.flipped == 0
        assert 0 < reports[0].relative_disagreement < 2**-8
        assert main([str(held), str(default)]) == 0
        out = capsys.readouterr().out
        assert "not resolvable: right (BF16)" in out
        assert "flipped" not in out
        assert "mean relative disagreement" in out

    def test_refuses_two_different_models(self, tmp_path: Path) -> None:
        name = "model.layers.0.self_attn.q_proj.weight"
        left = _checkpoint(tmp_path / "left", {name: np.zeros((1, 64))})
        right = _checkpoint(tmp_path / "right", {name: np.zeros((1, 64))}, hidden_size=128)
        assert "hidden_size" in (same_architecture(left, right) or "")
        assert main([str(left), str(right)]) == 1

    def test_refuses_checkpoints_with_different_layers(self, tmp_path: Path) -> None:
        left = _checkpoint(
            tmp_path / "left", {"model.layers.0.mlp.up_proj.weight": np.zeros((1, 64))}
        )
        right = _checkpoint(
            tmp_path / "right", {"model.layers.0.mlp.down_proj.weight": np.zeros((1, 64))}
        )
        assert main([str(left), str(right)]) == 1
