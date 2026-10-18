# Copyright 2026 The MRound Authors
# SPDX-License-Identifier: Apache-2.0
"""The code by code comparison of two fake quantized checkpoints.

Built on synthetic groups whose scales and codes are known, so every count the
script prints is checked against the arithmetic that produced the values, and
on a safetensors file written by hand, so the reader is checked against the
format rather than against a library that would hide a misread. MRound's packed
checkpoints are written by hand too, their words packed here bit by bit, and
once by MRound's own exporter where MLX is installed. A sharded checkpoint is
the one file's bytes split across shards with an index, and is read as one.
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
    PackedLayer,
    coarse_sides,
    compare,
    compare_layer,
    compare_packed,
    find_checkpoint,
    implied_scales,
    linear_weights,
    main,
    packed_layers,
    read_checkpoint,
    read_header,
    read_packed,
    read_tensor,
    same_architecture,
    unpack_words,
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


def _reshard(directory: Path, parts: list[list[str]]) -> list[str]:
    """Split a checkpoint's one weight file into shards holding ``parts``, with an index.

    The bytes of every tensor are copied as they are, so a shard holds what the
    one file held, dtype for dtype; the names are the ones a sharded checkpoint
    carries, ``model-00001-of-00002.safetensors`` and on.
    """
    source = directory / "model.safetensors"
    entries = {entry.name: entry for entry in read_header(source)}
    raw = source.read_bytes()
    assert sorted(name for part in parts for name in part) == sorted(entries)
    files = [f"model-{i:05d}-of-{len(parts):05d}.safetensors" for i in range(1, len(parts) + 1)]
    weight_map = {}
    for file, part in zip(files, parts, strict=True):
        header: dict[str, object] = {}
        payload = b""
        for name in part:
            entry = entries[name]
            chunk = raw[entry.start : entry.end]
            header[name] = {
                "dtype": entry.dtype,
                "shape": list(entry.shape),
                "data_offsets": [len(payload), len(payload) + len(chunk)],
            }
            payload += chunk
            weight_map[name] = file
        encoded = json.dumps(header).encode("utf-8")
        (directory / file).write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)
    source.unlink()
    index = {"metadata": {"total_size": len(raw)}, "weight_map": weight_map}
    (directory / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")
    return files


_SHARDED = {
    "model.embed_tokens.weight": np.zeros((10, 64), dtype=np.float32),
    "model.layers.0.mlp.up_proj.weight": np.arange(128, dtype=np.float32).reshape(2, 64),
    "model.layers.0.input_layernorm.weight": np.ones(64, dtype=np.float32),
}
_PARTS = [
    ["model.embed_tokens.weight"],
    ["model.layers.0.input_layernorm.weight", "model.layers.0.mlp.up_proj.weight"],
]


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
            np.testing.assert_array_equal(read_tensor(entry), tensors[entry.name])
        # Only the block's two dimensional weights count as linear layers.
        assert [e.name for e in linear_weights(entries)] == ["model.layers.0.mlp.up_proj.weight"]

    def test_decodes_bfloat16_as_torch_writes_it(self, tmp_path: Path) -> None:
        # 1.0 and 0.5 are exact in bfloat16; 1/3 is not and rounds to nearest even.
        values = np.array([[1.0, 0.5, 1.0 / 3.0, -2.5]], dtype=np.float32)
        path = tmp_path / "model.safetensors"
        _write_safetensors(path, {"model.layers.0.mlp.up_proj.weight": values}, "BF16")
        (entry,) = read_header(path)
        assert entry.dtype == "BF16"
        decoded = read_tensor(entry)
        assert decoded[0, 0] == 1.0
        assert decoded[0, 1] == 0.5
        assert decoded[0, 3] == -2.5
        assert decoded[0, 2] == pytest.approx(1.0 / 3.0, rel=2**-8)
        assert decoded[0, 2] != 1.0 / 3.0

    def test_reads_a_sharded_checkpoint_as_one(self, tmp_path: Path) -> None:
        root = _checkpoint(tmp_path / "sharded", _SHARDED)
        files = _reshard(root, _PARTS)
        entries = read_checkpoint(root)
        assert sorted(entries) == sorted(_SHARDED)
        for file, part in zip(files, _PARTS, strict=True):
            for name in part:
                assert entries[name].file == root / file
                np.testing.assert_array_equal(read_tensor(entries[name]), _SHARDED[name])

    def test_one_file_needs_no_index(self, tmp_path: Path) -> None:
        root = _checkpoint(tmp_path / "one", _SHARDED)
        entries = read_checkpoint(root)
        assert {entry.file for entry in entries.values()} == {root / "model.safetensors"}

    def test_a_tensor_in_two_files_is_refused(self, tmp_path: Path) -> None:
        # A shard of a larger checkpoint left beside a smaller one written
        # without removing it: a loader would keep whichever copy it read last.
        root = _checkpoint(tmp_path / "stale", _SHARDED)
        _reshard(root, _PARTS)
        stale = {"model.embed_tokens.weight": np.ones((10, 64), dtype=np.float32)}
        _write_safetensors(root / "model-00003-of-00003.safetensors", stale)
        with pytest.raises(ValueError, match=r"model\.embed_tokens\.weight in both"):
            read_checkpoint(root)

    def test_a_missing_shard_is_refused_by_the_index(self, tmp_path: Path) -> None:
        root = _checkpoint(tmp_path / "partial", _SHARDED)
        files = _reshard(root, _PARTS)
        (root / files[1]).unlink()
        with pytest.raises(ValueError, match="in no weight file"):
            read_checkpoint(root)

    def test_an_index_that_places_a_tensor_elsewhere_is_refused(self, tmp_path: Path) -> None:
        root = _checkpoint(tmp_path / "moved", _SHARDED)
        files = _reshard(root, _PARTS)
        index_path = root / "model.safetensors.index.json"
        index = json.loads(index_path.read_text(encoding="utf-8"))
        index["weight_map"]["model.embed_tokens.weight"] = files[1]
        index_path.write_text(json.dumps(index), encoding="utf-8")
        with pytest.raises(ValueError, match="in another file than the index names"):
            read_checkpoint(root)

    def test_a_directory_without_weights_is_refused(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        root = _checkpoint(tmp_path / "empty", {})
        (root / "model.safetensors").unlink()
        with pytest.raises(FileNotFoundError, match="no weight file"):
            read_checkpoint(root)
        assert main([str(root), str(root)]) == 1
        assert "refusing" in capsys.readouterr().err

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


def _pack(codes: np.ndarray, bits: int) -> np.ndarray:
    """MLX's layout, one bit at a time: code i at bits [i * bits, (i + 1) * bits)."""
    rows, columns = codes.shape
    words = np.zeros((rows, columns * bits // 32), dtype=np.uint64)
    for row in range(rows):
        for column in range(columns):
            for place in range(bits):
                if (int(codes[row, column]) >> place) & 1:
                    position = column * bits + place
                    words[row, position // 32] |= np.uint64(1 << (position % 32))
    return words.astype("<u4")


def _write_mixed(path: Path, tensors: dict[str, tuple[str, np.ndarray]]) -> None:
    """A safetensors file whose tensors each keep their own dtype."""
    header: dict[str, object] = {}
    payload = b""
    for name, (dtype, array) in tensors.items():
        if dtype == "BF16":
            raw = _bfloat16_bits(array)
        elif dtype == "U32":
            raw = array.astype("<u4").tobytes()
        else:
            raw = array.astype(np.float32).tobytes()
        header[name] = {
            "dtype": dtype,
            "shape": list(array.shape),
            "data_offsets": [len(payload), len(payload) + len(raw)],
        }
        payload += raw
    encoded = json.dumps(header).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


def _packed_checkpoint(
    directory: Path,
    layers: dict[str, tuple[np.ndarray, np.ndarray, int, int]],
    dense: dict[str, np.ndarray] | None = None,
) -> Path:
    """An MLX checkpoint: each layer's stored codes, signed scales, width and group size.

    Written as MRound writes a symmetric layer: the bias is minus the stored
    scale times ``2 ** (bits - 1)``, and the first layer's width and group
    size are the defaults of the quantization block, the others overrides.
    """
    directory.mkdir(parents=True)
    tensors: dict[str, tuple[str, np.ndarray]] = {}
    first = next(iter(layers.values()))
    block: dict[str, object] = {"group_size": first[3], "bits": first[2], "mode": "affine"}
    for path, (codes, scales, bits, group_size) in layers.items():
        stored = np.frombuffer(_bfloat16_bits(scales), dtype=np.uint16).astype(np.uint32) << 16
        rounded = stored.view(np.float32).reshape(scales.shape)
        tensors[f"{path}.weight"] = ("U32", _pack(codes, bits))
        tensors[f"{path}.scales"] = ("BF16", rounded)
        tensors[f"{path}.biases"] = ("BF16", -rounded * 2.0 ** (bits - 1))
        if (bits, group_size) != (first[2], first[3]):
            block[path] = {"group_size": group_size, "bits": bits, "mode": "affine"}
    for name, values in (dense or {}).items():
        tensors[name] = ("BF16", values)
    settings = {"model_type": "qwen2", "hidden_size": 64, "num_hidden_layers": 1, "vocab_size": 10}
    settings["quantization"] = block
    (directory / "config.json").write_text(json.dumps(settings), encoding="utf-8")
    _write_mixed(directory / "model.safetensors", tensors)
    return directory


class TestUnpackWords:
    @pytest.mark.parametrize("bits", [2, 3, 4, 5, 6, 8])
    def test_reads_back_every_width_mlx_packs(self, bits: int) -> None:
        rng = np.random.default_rng(bits)
        codes = rng.integers(0, 2**bits, size=(3, 64))
        raw = _pack(codes, bits).tobytes()
        np.testing.assert_array_equal(unpack_words(raw, 3, 64, bits), codes)

    def test_a_code_that_straddles_two_words_is_read_whole(self) -> None:
        # At 3 bits code 10 occupies bits 30, 31 and 32: two in the first word
        # and one in the second.
        codes = np.zeros((1, 32), dtype=np.int64)
        codes[0, 10] = 0b101
        words = _pack(codes, 3)
        assert int(words[0, 0]) == 1 << 30
        assert int(words[0, 1]) == 1
        assert unpack_words(words.tobytes(), 1, 32, 3)[0, 10] == 0b101

    def test_refuses_words_that_do_not_hold_the_codes(self) -> None:
        raw = _pack(np.zeros((2, 64), dtype=np.int64), 2).tobytes()
        with pytest.raises(ValueError, match="do not hold"):
            unpack_words(raw, 2, 64, 4)


def _layer(codes: np.ndarray, scales: list[float], bits: int = 2) -> PackedLayer:
    stored = np.array(scales, dtype=np.float64).reshape(codes.shape[0], -1)
    group_size = codes.shape[1] // stored.shape[1]
    return PackedLayer(codes, stored, -stored * 2.0 ** (bits - 1), bits, group_size)


class TestComparePacked:
    def test_counts_the_stored_codes_that_moved(self) -> None:
        rng = np.random.default_rng(0)
        codes = rng.integers(0, 4, size=(2, 64))
        moved = codes.copy()
        moved[0, :3] = (moved[0, :3] + 1) % 4  # one step, or three where it wrapped
        moved[1, 0] = (moved[1, 0] + 2) % 4
        report = compare_packed("layer", _layer(codes, [0.5, -0.25]), _layer(moved, [0.5, -0.25]))
        summary = report.summary
        assert summary.groups == 2
        assert summary.identical_scale == 2
        assert summary.same_lattice == 2
        assert summary.off_lattice == 0
        assert summary.codes == 128
        assert summary.flipped == 4
        steps = np.abs(codes - moved)
        assert summary.one_apart == int((steps == 1).sum())
        assert report.width == 2
        assert report.differing == 4
        assert report.relative_disagreement > 0

    def test_identical_layers_agree_to_the_bit(self) -> None:
        codes = np.tile(np.arange(4), (2, 16))
        report = compare_packed("layer", _layer(codes, [0.5, 0.25]), _layer(codes, [0.5, 0.25]))
        assert report.summary.flipped == 0
        assert report.differing == 0
        assert report.relative_disagreement == 0.0

    def test_a_scale_that_changed_sign_is_another_lattice(self) -> None:
        codes = np.tile(np.arange(4), (1, 16))
        report = compare_packed("layer", _layer(codes, [0.5]), _layer(codes, [-0.5]))
        assert report.summary.same_lattice == 0
        assert report.summary.codes == 0
        assert report.summary.scale_drift == [2.0]

    def test_a_moved_bias_is_another_lattice(self) -> None:
        codes = np.tile(np.arange(4), (1, 16))
        left = _layer(codes, [0.5])
        right = PackedLayer(codes, left.scales, left.biases + 0.5, 2, 64)
        report = compare_packed("layer", left, right)
        assert report.summary.identical_scale == 0
        assert report.summary.same_lattice == 0

    def test_the_values_are_the_scale_times_the_signed_code(self) -> None:
        codes = np.array([[0, 1, 2, 3] * 16])
        values = _layer(codes, [0.5]).values()
        np.testing.assert_array_equal(values[0, :4], [-1.0, -0.5, 0.0, 0.5])

    def test_refuses_two_group_sizes(self) -> None:
        codes = np.zeros((1, 64), dtype=np.int64)
        with pytest.raises(ValueError, match="groups of 64 on the left and 32"):
            compare_packed("layer", _layer(codes, [1.0]), _layer(codes, [1.0, 1.0]))


class TestPackedCheckpoints:
    def test_two_runs_of_one_model_compared_end_to_end(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        rng = np.random.default_rng(1)
        q = "model.layers.0.self_attn.q_proj"
        up = "model.layers.0.mlp.up_proj"
        dense = {"model.layers.0.mlp.down_proj.weight": rng.normal(size=(64, 48))}
        q_codes = rng.integers(0, 4, size=(4, 128))
        up_codes = rng.integers(0, 8, size=(2, 64))
        q_scales = np.array([[0.5, -0.25], [0.125, 0.75], [-0.5, 0.25], [1.0, -1.0]])
        up_scales = np.array([[0.25, 0.5], [-0.125, 1.5]])
        left = _packed_checkpoint(
            tmp_path / "left",
            {q: (q_codes, q_scales, 2, 64), up: (up_codes, up_scales, 3, 32)},
            dense,
        )
        moved = q_codes.copy()
        moved[0, 0] = (moved[0, 0] + 1) % 4
        right = _packed_checkpoint(
            tmp_path / "right",
            {q: (moved, q_scales, 2, 64), up: (up_codes, up_scales, 3, 32)},
            dense,
        )
        entries = read_header(left / "model.safetensors")
        assert packed_layers(entries) == {f"{q}.weight", f"{up}.weight"}
        assert coarse_sides(left, right) == []

        reports = {r.name: r for r in compare(left, right)}
        assert reports[f"{q}.weight"].summary.flipped == 1
        assert reports[f"{q}.weight"].width == 2
        assert reports[f"{up}.weight"].summary.flipped == 0
        assert reports[f"{up}.weight"].width == 3
        assert reports[f"{up}.weight"].summary.groups == 4
        assert reports["model.layers.0.mlp.down_proj.weight"].dense

        assert main([str(left), str(right)]) == 0
        out = capsys.readouterr().out
        assert "1 layers left dense by a packed side: 0 of 3,072 weights differ" in out
        assert "flipped 1 (0.16%)" in out
        assert "2 bit, 1 layers" in out
        assert "3 bit, 1 layers" in out

    def test_a_sharded_checkpoint_compares_as_the_same_one_in_one_file(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # The shard ends between a layer's words and its scales, which is
        # where a split of consecutive tensors can fall.
        rng = np.random.default_rng(2)
        q = "model.layers.0.self_attn.q_proj"
        layers = {q: (rng.integers(0, 4, size=(2, 64)), np.array([[0.5], [-0.25]]), 2, 64)}
        dense = {"model.layers.0.mlp.down_proj.weight": rng.normal(size=(64, 48))}
        one = _packed_checkpoint(tmp_path / "one", layers, dense)
        sharded = _packed_checkpoint(tmp_path / "sharded", layers, dense)
        _reshard(
            sharded,
            [
                [f"{q}.weight", "model.layers.0.mlp.down_proj.weight"],
                [f"{q}.scales", f"{q}.biases"],
            ],
        )
        entries = read_checkpoint(sharded)
        assert entries[f"{q}.weight"].file != entries[f"{q}.scales"].file
        reports = {r.name: r for r in compare(one, sharded)}
        assert reports[f"{q}.weight"].summary.identical_scale == 2
        assert reports[f"{q}.weight"].differing == 0
        assert reports[f"{q}.weight"].relative_disagreement == 0.0
        assert reports["model.layers.0.mlp.down_proj.weight"].dense
        assert reports["model.layers.0.mlp.down_proj.weight"].differing == 0
        assert main([str(sharded), str(one)]) == 0
        assert "flipped 0 (0.00%)" in capsys.readouterr().out

    def test_a_group_size_given_against_the_checkpoint_is_refused(self, tmp_path: Path) -> None:
        codes = np.zeros((1, 64), dtype=np.int64)
        layer = {"model.layers.0.self_attn.q_proj": (codes, np.array([[1.0]]), 2, 64)}
        left = _packed_checkpoint(tmp_path / "left", layer)
        right = _packed_checkpoint(tmp_path / "right", layer)
        with pytest.raises(ValueError, match="groups of 64, not the 32 given"):
            compare(left, right, 32)
        assert main([str(left), str(right), "--group-size", "64"]) == 0

    def test_a_packed_side_against_a_fake_export_uses_its_stored_scales(
        self, tmp_path: Path
    ) -> None:
        # The packed group holds signed codes -2 and 0 only, so its values
        # imply twice its scale; the fake side's one -1 makes its own values
        # imply the true one. On the stored scale the two share a lattice and
        # one code moved; on the implied ones they would share none.
        name = "model.layers.0.self_attn.q_proj"
        codes = np.array([[0, 2] * 32])
        signed = codes - 2
        signed[0, 0] = -1
        packed = _packed_checkpoint(tmp_path / "packed", {name: (codes, np.array([[0.5]]), 2, 64)})
        fake = _checkpoint(tmp_path / "fake", {f"{name}.weight": signed * 0.5})
        (report,) = compare(packed, fake)
        assert report.summary.same_lattice == 1
        assert report.summary.flipped == 1
        assert report.summary.one_apart == 1
        assert report.differing == 1
        assert report.width == 2
        implied = compare_layer("layer", (codes - 2) * 0.5, signed * 0.5, 64)
        assert implied.summary.same_lattice == 0


class TestMRoundExport:
    """The reader against the exporter it has to agree with, where MLX is installed."""

    @pytest.mark.parametrize("bits", [2, 3, 4, 8])
    def test_reads_the_codes_mround_wrote(self, tmp_path: Path, bits: int) -> None:
        mx = pytest.importorskip("mlx.core", reason="MRound's exporter is MLX's")
        quantizer = pytest.importorskip("mround.core.quantizer")
        mlx_export = pytest.importorskip("mround.formats.mlx_export")
        schemes = pytest.importorskip("mround.schemes")

        scheme = schemes.QuantScheme(bits=bits, group_size=32, symmetry=schemes.Symmetry.SYMMETRIC)
        mx.random.seed(bits)
        weight = mx.random.normal((8, 64))
        codes, scale, zero_point = quantizer.quantize_codes(weight, None, scheme)
        path = "model.layers.0.mlp.up_proj"
        config = {"model_type": "qwen2", "hidden_size": 64, "num_hidden_layers": 1}
        mlx_export.export_mlx(
            tmp_path / "out",
            {"model.layers.0.input_layernorm.weight": mx.ones((64,))},
            {path: mlx_export.QuantizedLayer(codes, scale, zero_point, scheme)},
            config,
        )
        root = tmp_path / "out"
        entries = {e.name: e for e in read_header(root / "model.safetensors")}
        layer = read_packed(root, entries, f"{path}.weight")
        expected = np.array(codes.astype(mx.int32)).reshape(8, 64) + 2 ** (bits - 1)
        np.testing.assert_array_equal(layer.codes, expected)
        stored = np.array(scale.astype(mx.bfloat16).astype(mx.float32)).reshape(8, 2)
        np.testing.assert_array_equal(layer.scales, stored)
        np.testing.assert_array_equal(
            layer.values(), stored.repeat(32, axis=1) * np.array(codes).reshape(8, 64)
        )
        (report,) = compare(root, root)
        assert report.width == bits
        assert report.summary.identical_scale == report.summary.groups == 16
        assert report.differing == 0

    @pytest.mark.parametrize("bits", [2, 3, 4, 8])
    def test_reads_the_shards_mround_wrote(self, tmp_path: Path, bits: int) -> None:
        # 256 bytes of float32 norm, then the layer's 64 * bits bytes of words
        # fill the first shard, and its scales and biases start the second.
        mx = pytest.importorskip("mlx.core", reason="MRound's exporter is MLX's")
        quantizer = pytest.importorskip("mround.core.quantizer")
        mlx_export = pytest.importorskip("mround.formats.mlx_export")
        schemes = pytest.importorskip("mround.schemes")

        scheme = schemes.QuantScheme(bits=bits, group_size=32, symmetry=schemes.Symmetry.SYMMETRIC)
        mx.random.seed(bits)
        weight = mx.random.normal((8, 64))
        codes, scale, zero_point = quantizer.quantize_codes(weight, None, scheme)
        path = "model.layers.0.mlp.up_proj"
        config = {"model_type": "qwen2", "hidden_size": 64, "num_hidden_layers": 1}
        layer = {path: mlx_export.QuantizedLayer(codes, scale, zero_point, scheme)}
        norm = {"model.layers.0.input_layernorm.weight": mx.ones((64,))}
        for label, limit in (("one", mlx_export.SHARD_BYTES), ("shards", 256 + 64 * bits)):
            mlx_export.export_mlx(tmp_path / label, norm, layer, config, shard_bytes=limit)
        entries = read_checkpoint(tmp_path / "shards")
        assert entries[f"{path}.weight"].file.name == "model-00001-of-00002.safetensors"
        assert entries[f"{path}.scales"].file.name == "model-00002-of-00002.safetensors"
        (report,) = compare(tmp_path / "one", tmp_path / "shards")
        assert report.width == bits
        assert report.summary.identical_scale == report.summary.groups == 16
        assert report.differing == 0
