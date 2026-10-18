# Copyright 2026 The MRound Authors
# SPDX-License-Identifier: Apache-2.0
"""The formats layer, against the NumPy reference and against MLX's conventions.

Two different kinds of check live here and they are worth telling apart.

The packing tests are parity tests: MRound has two implementations of the same
layout and they must agree, which is the same argument as
`test_mlx_parity.py` makes for the quantizer.

The export tests are conformance tests, and they are weaker by nature. They
check that the checkpoint has the structure MLX's source says it needs, and one
of them, since the checkpoint can be split (ROAD-TO-1.0.md X2), has mlx-lm's
own ``load_model``, which reads the weights and needs no tokenizer, load a tiny
model from one file and from shards. That a real model loaded this way
generates is an integration test's to show; these establish that the shapes,
names, dtypes, files and configuration keys are right, and no more. See
MEMORY.md D-017 for what was verified against MLX's source and what was not.
"""

from __future__ import annotations

import itertools
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from mround.exceptions import ExportError, UnsupportedSchemeError
from mround.reference import packing as ref_packing
from mround.schemes import SUPPORTED_BITS, QuantScheme, Symmetry

pytestmark = pytest.mark.needs_mlx

pytest.importorskip("mlx.core", reason="MLX is not installed")

import mlx.core as mx  # noqa: E402

from mround.core import quantizer as mlx_quant  # noqa: E402
from mround.formats import mlx_export  # noqa: E402
from mround.formats import packing as mlx_packing  # noqa: E402


def codes_for(bits: int, rows: int = 8, cols: int = 128) -> np.ndarray:
    """Unsigned codes covering the full range at this width."""
    return np.random.default_rng(bits).integers(0, 2**bits, size=(rows, cols), dtype=np.int64)


class TestPacking:
    @pytest.mark.parametrize("bits", sorted(SUPPORTED_BITS))
    def test_round_trips(self, bits: int) -> None:
        codes = codes_for(bits)
        packed = mlx_packing.pack_codes(mx.array(codes), bits)
        recovered = mlx_packing.unpack_codes(packed, codes.shape[1], bits)
        assert np.array_equal(np.array(recovered), codes)

    @pytest.mark.parametrize("fields", [1, 2, 3, 4, 5, 6, 7, 8, 16, 32])
    def test_the_fold_keeps_every_field(self, fields: int) -> None:
        # MLX has no bitwise-or reduction, so the fold is hand-written as a
        # halving loop, and a halving loop needs a power-of-two axis. It did not
        # have one: the bitstream unpack folds `bits` fields, so five and seven
        # raised and three and six survived only because or is idempotent and
        # the duplicated field did no harm. Distinct single bits per field make
        # a dropped or duplicated one visible either way.
        placed = mx.array([[1 << i for i in range(fields)]], dtype=mx.uint32)
        assert int(mlx_packing._fold_or(placed)[0]) == (1 << fields) - 1

    @pytest.mark.parametrize("bits", sorted(SUPPORTED_BITS))
    def test_matches_the_reference_byte_for_byte(self, bits: int) -> None:
        # Packing is integer work, so unlike the quantizer there is no float32
        # allowance here. Anything other than exact agreement is a defect.
        codes = codes_for(bits)
        ours = np.array(mlx_packing.pack_codes(mx.array(codes), bits))
        expected = ref_packing.pack_codes(codes, bits)
        assert np.array_equal(ours, expected.astype(ours.dtype))

    @pytest.mark.parametrize("bits", sorted(SUPPORTED_BITS))
    def test_packed_width_matches_the_reference(self, bits: int) -> None:
        for cols in (32, 64, 128, 4096):
            assert mlx_packing.packed_width(cols, bits) == ref_packing.packed_width(cols, bits)

    def test_out_of_range_codes_are_rejected_rather_than_masked(self) -> None:
        # Masking would corrupt weights silently, which is the failure mode this
        # whole layer is written to avoid.
        with pytest.raises(ValueError, match=r"must lie in \[0, 15\]"):
            mlx_packing.pack_codes(mx.array([[16, 0, 0, 0, 0, 0, 0, 0]]), 4)


class TestToAffine:
    """The conversion to ``W = scales * q_u + biases``."""

    @pytest.mark.parametrize("bits", [2, 3, 4, 8])
    @pytest.mark.parametrize("symmetry", list(Symmetry))
    def test_affine_form_reproduces_the_reconstruction(self, bits: int, symmetry: Symmetry) -> None:
        weight = mx.array(
            np.random.default_rng(bits).normal(scale=0.05, size=(8, 64)).astype(np.float32)
        )
        scheme = QuantScheme(bits=bits, group_size=32, symmetry=symmetry)
        params = mlx_quant.init_params(weight, scheme)

        codes, scale, zero_point = mlx_quant.quantize_codes(weight, params, scheme)
        expected = mlx_quant.fake_quantize(weight, params, scheme)

        scales, biases = mlx_export.to_affine(scale, zero_point, scheme, dtype=mx.float32)
        unsigned = codes + (2 ** (bits - 1) if symmetry is Symmetry.SYMMETRIC else 0)
        rebuilt = (scales * unsigned + biases).reshape(weight.shape)

        assert np.allclose(np.array(rebuilt), np.array(expected), atol=1e-6)

    @pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16, mx.float32])
    @pytest.mark.parametrize("bits", [2, 4, 8])
    def test_zero_stays_exactly_representable_at_storage_dtype(
        self, dtype: mx.Dtype, bits: int
    ) -> None:
        # The reason to_affine derives the bias from the already-rounded scale.
        # bfloat16 has eight mantissa bits, so a bias computed from the exact
        # scale and then rounded would leave a residue on every group in the
        # model, biasing every weight that quantized to zero.
        scheme = QuantScheme(bits=bits, group_size=32)
        scale = mx.array(np.random.default_rng(bits).normal(size=(8, 2, 1)).astype(np.float32))
        scales, biases = mlx_export.to_affine(scale, None, scheme, dtype=dtype)
        at_zero = scales * (2 ** (bits - 1)) + biases
        assert np.array_equal(np.array(at_zero.astype(mx.float32)), np.zeros((8, 2, 1)))

    def test_scales_may_be_negative_and_that_is_the_normal_case(self) -> None:
        # MLX's own quantizer negates the scale for roughly half of all groups,
        # so this is the format's ordinary path rather than something it
        # tolerates. Pinned because it looks like a bug to anyone reading the
        # published docstring, which still shows the unsigned formula.
        weight = mx.array(
            np.random.default_rng(0).normal(scale=0.05, size=(64, 128)).astype(np.float32)
        )
        scheme = QuantScheme(bits=4, group_size=32)
        _, scale, _ = mlx_quant.quantize_codes(
            weight, mlx_quant.init_params(weight, scheme), scheme
        )
        scales, _ = mlx_export.to_affine(scale, None, scheme, dtype=mx.float32)
        negative = float(np.mean(np.array(scales) < 0))
        assert 0.2 < negative < 0.8, (
            f"{negative:.0%} of scales are negative; a symmetric weight "
            "distribution should give roughly half"
        )


def a_layer(
    bits: int = 4, group_size: int = 32, out_features: int = 8, in_features: int = 64
) -> mlx_export.QuantizedLayer:
    """One quantized layer, produced the way the pipeline will produce it."""
    weight = mx.array(
        np.random.default_rng(bits)
        .normal(scale=0.05, size=(out_features, in_features))
        .astype(np.float32)
    )
    scheme = QuantScheme(bits=bits, group_size=group_size)
    codes, scale, zero_point = mlx_quant.quantize_codes(
        weight, mlx_quant.init_params(weight, scheme), scheme
    )
    return mlx_export.QuantizedLayer(codes=codes, scale=scale, zero_point=zero_point, scheme=scheme)


class TestExport:
    def test_writes_the_names_and_shapes_mlx_expects(self, tmp_path: Path) -> None:
        layer = a_layer()
        mlx_export.export_mlx(
            tmp_path,
            dense_weights={"model.embed_tokens.weight": mx.zeros((10, 64), dtype=mx.bfloat16)},
            quantized={"model.layers.0.self_attn.q_proj": layer},
            config={"model_type": "llama"},
        )

        tensors = mx.load(str(tmp_path / "model.safetensors"))
        path = "model.layers.0.self_attn.q_proj"
        # The packed tensor keeps the .weight name; there is no qweight here.
        assert tensors[f"{path}.weight"].dtype == mx.uint32
        assert tensors[f"{path}.weight"].shape == (8, 64 * 4 // 32)
        assert tensors[f"{path}.scales"].shape == (8, 2)
        assert tensors[f"{path}.biases"].shape == (8, 2)
        assert tensors[f"{path}.scales"].dtype == mx.bfloat16
        assert tensors[f"{path}.biases"].dtype == mx.bfloat16
        assert "model.embed_tokens.weight" in tensors

    def test_writes_both_configuration_keys(self, tmp_path: Path) -> None:
        # mlx-lm keys loading off `quantization` and duplicates the block to
        # `quantization_config` for tooling that reads the Hugging Face
        # convention. Writing only one of them half-works, which is worse than
        # failing.
        mlx_export.export_mlx(
            tmp_path,
            dense_weights={},
            quantized={"layers.0.mlp": a_layer(bits=4)},
            config={"model_type": "llama"},
        )
        config = json.loads((tmp_path / "config.json").read_text())
        assert config["quantization"] == {"group_size": 32, "bits": 4, "mode": "affine"}
        assert config["quantization_config"] == config["quantization"]
        assert config["model_type"] == "llama", "the source configuration must survive"

    def test_mixed_precision_lands_as_flat_per_layer_overrides(self, tmp_path: Path) -> None:
        # The override schema is flat: module paths sit in the same dictionary
        # as the defaults. Nested would be the natural guess and would load as
        # a uniform model, quietly.
        mlx_export.export_mlx(
            tmp_path,
            dense_weights={},
            quantized={
                "layers.0.mlp": a_layer(bits=4),
                "layers.1.mlp": a_layer(bits=4),
                "layers.2.mlp": a_layer(bits=2),
            },
            config={},
        )
        block = json.loads((tmp_path / "config.json").read_text())["quantization"]
        assert block["bits"] == 4, "the majority scheme becomes the default"
        assert block["layers.2.mlp"] == {"group_size": 32, "bits": 2, "mode": "affine"}
        assert "layers.0.mlp" not in block, "layers matching the default need no override"

    def test_writes_a_weight_index(self, tmp_path: Path) -> None:
        mlx_export.export_mlx(
            tmp_path, dense_weights={}, quantized={"layers.0.mlp": a_layer()}, config={}
        )
        index = json.loads((tmp_path / "model.safetensors.index.json").read_text())
        assert set(index["weight_map"]) == {
            "layers.0.mlp.weight",
            "layers.0.mlp.scales",
            "layers.0.mlp.biases",
        }
        assert set(index["weight_map"].values()) == {"model.safetensors"}

    def test_the_index_counts_parameters_the_way_mlx_lm_does(self, tmp_path: Path) -> None:
        # mlx-lm writes total_size as the stored bytes and total_parameters as
        # the weights represented, counting a packed uint32 word as 32 / bits
        # parameters. Counting stored elements reported an eighth of the truth
        # at 4 bits. One 4 bit 8 x 64 layer plus a 10 x 64 dense embedding:
        # 512 + 640 parameters, in 8 x 8 words + 8 x 2 scales + 8 x 2 biases
        # + 640 dense elements of storage.
        embedding = mx.zeros((10, 64), dtype=mx.float16)
        mlx_export.export_mlx(
            tmp_path,
            dense_weights={"embed.weight": embedding},
            quantized={"layers.0.mlp": a_layer(bits=4, group_size=32)},
            config={},
            dtype="float16",
        )
        index = json.loads((tmp_path / "model.safetensors.index.json").read_text())
        assert index["metadata"]["total_parameters"] == 8 * 64 + 10 * 64
        words = 8 * (64 * 4 // 32)
        assert index["metadata"]["total_size"] == words * 4 + (8 * 2 + 8 * 2) * 2 + 640 * 2

    def test_seven_bits_is_refused_with_a_reason(self, tmp_path: Path) -> None:
        # MRound packs 7-bit and mx.quantize rejects it. The export layer is
        # where that asymmetry has to surface, and it should say what to do.
        with pytest.raises(UnsupportedSchemeError, match="no other export path"):
            mlx_export.export_mlx(
                tmp_path,
                dense_weights={},
                quantized={"layers.0.mlp": a_layer(bits=7, group_size=32)},
                config={},
            )

    def test_mlx_itself_still_rejects_seven_bits(self) -> None:
        # The MLX half of the pin in test_reference_packing.py: a release that
        # accepts 7 bits makes MLX_SUPPORTED_BITS and the refusal above stale,
        # and this is where that surfaces.
        assert 7 not in mlx_export.MLX_SUPPORTED_BITS
        with pytest.raises(ValueError, match="bits"):
            mx.quantize(mx.zeros((8, 64)), group_size=32, bits=7)

    def test_per_channel_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(UnsupportedSchemeError, match="per-channel"):
            mlx_export.export_mlx(
                tmp_path,
                dense_weights={},
                quantized={"layers.0.mlp": a_layer(group_size=-1, in_features=64)},
                config={},
            )

    def test_a_group_size_mlx_does_not_accept_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(UnsupportedSchemeError, match="group sizes"):
            mlx_export.export_mlx(
                tmp_path,
                dense_weights={},
                quantized={"layers.0.mlp": a_layer(group_size=16)},
                config={},
            )

    def test_a_dense_and_quantized_collision_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ExportError, match="both a dense and a quantized"):
            mlx_export.export_mlx(
                tmp_path,
                dense_weights={"layers.0.mlp.weight": mx.zeros((4, 4))},
                quantized={"layers.0.mlp": a_layer()},
                config={},
            )

    def test_exporting_nothing_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ExportError, match="no quantized layers"):
            mlx_export.export_mlx(tmp_path, dense_weights={}, quantized={}, config={})

    def test_round_trips_through_mlx_dequantize(self, tmp_path: Path) -> None:
        # The closest thing to an end-to-end check available without a model:
        # write the checkpoint, read it back, and have MLX itself reconstruct
        # the weights from it. If the packing layout or the affine conversion
        # were wrong, this is where it shows.
        bits, group_size = 4, 32
        weight = mx.array(
            np.random.default_rng(1).normal(scale=0.05, size=(8, 64)).astype(np.float32)
        )
        scheme = QuantScheme(bits=bits, group_size=group_size)
        params = mlx_quant.init_params(weight, scheme)
        codes, scale, zero_point = mlx_quant.quantize_codes(weight, params, scheme)
        expected = np.array(mlx_quant.fake_quantize(weight, params, scheme))

        mlx_export.export_mlx(
            tmp_path,
            dense_weights={},
            quantized={
                "l": mlx_export.QuantizedLayer(codes, scale, zero_point, scheme),
            },
            config={},
            dtype="float32",
        )
        tensors = mx.load(str(tmp_path / "model.safetensors"))
        rebuilt = mx.dequantize(
            tensors["l.weight"],
            scales=tensors["l.scales"],
            biases=tensors["l.biases"],
            group_size=group_size,
            bits=bits,
        )
        assert np.allclose(np.array(rebuilt), expected, atol=1e-5), (
            "MLX's own dequantize does not reproduce MRound's reconstruction, so "
            "either the packing layout or the affine conversion is wrong"
        )

    @pytest.mark.parametrize("group_size", sorted(mlx_export.MLX_GROUP_SIZES))
    @pytest.mark.parametrize("bits", sorted(mlx_export.MLX_SUPPORTED_BITS))
    def test_packing_is_word_identical_to_mlx_at_every_width(
        self, bits: int, group_size: int
    ) -> None:
        # The layout claim is "byte identical with mx.quantize at every width
        # MLX accepts", and until this test the suite checked it against a
        # re-derivation of MLX's algorithm written in test_reference_packing.py,
        # with real MLX consulted at 4 bits and group 32 only. This asks MLX
        # itself, on every width and group size it accepts: recover the codes
        # MLX chose from its own scales and biases, pack them with MRound's
        # packer, and compare the uint32 words; then unpack MLX's words and
        # compare the codes.
        weight = mx.array(
            np.random.default_rng(bits * 100 + group_size)
            .normal(scale=0.05, size=(8, 256))
            .astype(np.float32)
        )
        words, scales, biases = mx.quantize(weight, group_size=group_size, bits=bits)
        grouped = weight.reshape(8, 256 // group_size, group_size)
        codes = mx.clip(
            mx.round((grouped - biases[..., None]) / scales[..., None]), 0, 2**bits - 1
        ).reshape(8, 256)
        packed = mlx_packing.pack_codes(codes.astype(mx.uint32), bits)
        assert np.array_equal(np.array(packed), np.array(words)), (
            f"MRound's packing of MLX's own codes differs from mx.quantize's words "
            f"at {bits} bits, group size {group_size}"
        )
        unpacked = mlx_packing.unpack_codes(words, 256, bits)
        assert np.array_equal(np.array(unpacked), np.array(codes).astype(np.uint32))


def _weight_files(directory: Path) -> list[str]:
    """The weight files mlx-lm would load from ``directory``, by its own glob."""
    return sorted(path.name for path in directory.glob("model*.safetensors"))


def _shard_contents(directory: Path) -> dict[str, dict[str, mx.array]]:
    return {name: mx.load(str(directory / name)) for name in _weight_files(directory)}


class TestShards:
    """Past :data:`mlx_export.SHARD_BYTES` the checkpoint is split as mlx-lm splits one.

    The limit is the argument's, so these run on bytes rather than gigabytes:
    one 4 bit layer of 8 by 64 at group size 32 stores 256 bytes of words and
    32 each of scales and biases.
    """

    def test_past_the_limit_it_is_split_into_numbered_shards(self, tmp_path: Path) -> None:
        layers = {f"model.layers.{i}.mlp.up_proj": a_layer(bits=4) for i in range(5)}
        dense = {"model.embed_tokens.weight": mx.zeros((10, 64), dtype=mx.bfloat16)}
        limit = 1000
        mlx_export.export_mlx(tmp_path, dense, layers, config={}, shard_bytes=limit)

        files = _weight_files(tmp_path)
        assert len(files) > 1
        assert files == mlx_export.shard_names(len(files))
        shards = _shard_contents(tmp_path)
        held: dict[str, str] = {}
        sizes = []
        for file in files:
            shard = shards[file]
            assert shard, f"{file} holds nothing"
            sizes.append(sum(t.nbytes for t in shard.values()))
            assert sizes[-1] <= limit or len(shard) == 1, f"{file} is past the limit"
            for name in shard:
                assert name not in held, f"{name} is in {held.get(name)} and in {file}"
                held[name] = file
        # Greedy, as mlx-lm's own split is: no two neighbours would fit in one.
        assert all(a + b > limit for a, b in itertools.pairwise(sizes))
        index = json.loads((tmp_path / "model.safetensors.index.json").read_text())
        assert index["weight_map"] == held
        assert set(held) == {"model.embed_tokens.weight"} | {
            f"{path}.{kind}" for path in layers for kind in ("weight", "scales", "biases")
        }
        assert index["metadata"]["total_size"] == sum(sizes)

    def test_a_checkpoint_of_exactly_the_limit_stays_one_file(self, tmp_path: Path) -> None:
        layers = {"layers.0.mlp": a_layer(bits=4), "layers.1.mlp": a_layer(bits=4)}
        size = 2 * (256 + 32 + 32)
        mlx_export.export_mlx(tmp_path / "at", {}, layers, config={}, shard_bytes=size)
        mlx_export.export_mlx(tmp_path / "under", {}, layers, config={}, shard_bytes=size - 1)
        assert _weight_files(tmp_path / "at") == ["model.safetensors"]
        index = json.loads((tmp_path / "at" / "model.safetensors.index.json").read_text())
        assert set(index["weight_map"].values()) == {"model.safetensors"}
        assert _weight_files(tmp_path / "under") == mlx_export.shard_names(2)

    def test_a_tensor_past_the_limit_fills_a_shard_of_its_own(self, tmp_path: Path) -> None:
        # Each layer's 256 bytes of words are past a limit of 100, and its
        # scales and biases fit together under it. mlx-lm's own split closes
        # the empty shard it starts with when the first tensor is too large,
        # and writes it; this writes none.
        layers = {"layers.0.mlp": a_layer(bits=4), "layers.1.mlp": a_layer(bits=4)}
        mlx_export.export_mlx(tmp_path, {}, layers, config={}, shard_bytes=100)
        shards = _shard_contents(tmp_path)
        assert [sorted(shard) for shard in shards.values()] == [
            ["layers.0.mlp.weight"],
            ["layers.0.mlp.biases", "layers.0.mlp.scales"],
            ["layers.1.mlp.weight"],
            ["layers.1.mlp.biases", "layers.1.mlp.scales"],
        ]

    def test_the_names_are_mlx_lms(self) -> None:
        assert mlx_export.shard_names(1) == ["model.safetensors"]
        assert mlx_export.shard_names(3) == [
            "model-00001-of-00003.safetensors",
            "model-00002-of-00003.safetensors",
            "model-00003-of-00003.safetensors",
        ]
        with pytest.raises(ExportError, match="at least one"):
            mlx_export.shard_names(0)

    def test_the_limit_is_five_binary_gigabytes(self) -> None:
        assert mlx_export.SHARD_BYTES == 5 * 1024**3

    def test_an_export_over_an_earlier_one_leaves_none_of_its_weight_files(
        self, tmp_path: Path
    ) -> None:
        # mlx-lm loads every model*.safetensors in the directory, so a shard
        # left from a larger checkpoint would be loaded with the new one.
        layers = {f"layers.{i}.mlp": a_layer(bits=4) for i in range(3)}
        (tmp_path / "other.safetensors").write_bytes(b"not a weight file of this export")
        mlx_export.export_mlx(tmp_path, {}, layers, config={}, shard_bytes=300)
        assert _weight_files(tmp_path) == mlx_export.shard_names(5)
        mlx_export.export_mlx(tmp_path, {}, layers, config={}, shard_bytes=700)
        assert _weight_files(tmp_path) == mlx_export.shard_names(2)
        mlx_export.export_mlx(tmp_path, {}, layers, config={})
        assert _weight_files(tmp_path) == ["model.safetensors"]
        mlx_export.export_mlx(tmp_path, {}, layers, config={}, shard_bytes=700)
        assert _weight_files(tmp_path) == mlx_export.shard_names(2)
        assert (tmp_path / "other.safetensors").read_bytes() == b"not a weight file of this export"

    def test_mlx_lm_loads_the_shards_as_it_loads_one_file(self, tmp_path: Path) -> None:
        # mlx-lm's own loader, which reads the weights and needs no tokenizer,
        # on a tiny Llama quantized whole at 4 bits: loaded from shards it is
        # the model loaded from one file, parameter for parameter, and strict
        # loading leaves no tensor missing or extra.
        mlx_lm_utils = pytest.importorskip("mlx_lm.utils", reason="the loader is mlx-lm's")
        from mlx.utils import tree_flatten  # noqa: PLC0415
        from mlx_lm.models import llama  # noqa: PLC0415

        config = {
            "model_type": "llama",
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "intermediate_size": 128,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "rms_norm_eps": 1e-5,
            "vocab_size": 64,
            "tie_word_embeddings": True,
        }
        mx.random.seed(0)
        weights = dict(tree_flatten(llama.Model(llama.ModelArgs(**config)).parameters()))
        scheme = QuantScheme(bits=4, group_size=32)
        quantized = {}
        for name, weight in weights.items():
            if ".layers." in name and weight.ndim == 2:
                codes, scale, zero_point = mlx_quant.quantize_codes(
                    weight, mlx_quant.init_params(weight, scheme), scheme
                )
                quantized[name.removesuffix(".weight")] = mlx_export.QuantizedLayer(
                    codes=codes, scale=scale, zero_point=zero_point, scheme=scheme
                )
        dense = {n: w for n, w in weights.items() if n.removesuffix(".weight") not in quantized}
        for label, limit in (("one", mlx_export.SHARD_BYTES), ("shards", 20_000)):
            mlx_export.export_mlx(
                tmp_path / label, dense, quantized, config, dtype="float32", shard_bytes=limit
            )
        assert len(_weight_files(tmp_path / "shards")) > 2

        loaded = {
            label: dict(tree_flatten(mlx_lm_utils.load_model(tmp_path / label)[0].parameters()))
            for label in ("one", "shards")
        }
        assert loaded["one"].keys() == loaded["shards"].keys()
        assert "model.layers.1.mlp.down_proj.scales" in loaded["shards"]
        for name, value in loaded["one"].items():
            assert mx.array_equal(value, loaded["shards"][name]).item(), name


# Two layers at 2 bits and two at 4: a tie on the count that picks the default.
_TIED_BLOCK = """
import json
import sys

sys.path.insert(0, sys.argv[1])

import mlx.core as mx

from mround.formats.mlx_export import QuantizedLayer, _quantization_block
from mround.schemes import QuantScheme


def layer(bits):
    scheme = QuantScheme(bits=bits, group_size=64)
    return QuantizedLayer(mx.zeros((1, 1, 64)), mx.ones((1, 1, 1)), None, scheme)


quantized = {"m.q": layer(2), "m.k": layer(2), "m.v": layer(4), "m.o": layer(4)}
print(json.dumps(_quantization_block(quantized), sort_keys=True))
"""


class TestTiedDefault:
    def test_a_tie_writes_the_same_configuration_in_every_process(self) -> None:
        # The default scheme was the first maximum of a set of schemes, and a
        # set iterates in the order of string hashes, which Python randomizes
        # per process: the same layers wrote 2 bits as the default under some
        # seeds and 4 bits under others. The loaded model was the same, the
        # configuration file was not. One process cannot show it, since its
        # seed is fixed at startup, so this runs fresh ones.
        root = str(Path(__file__).resolve().parents[2])
        outputs = set()
        for seed in range(10):
            result = subprocess.run(
                [sys.executable, "-c", _TIED_BLOCK, root],
                capture_output=True,
                text=True,
                check=True,
                env=dict(os.environ, PYTHONHASHSEED=str(seed)),
            )
            outputs.add(result.stdout)
        assert len(outputs) == 1, outputs
        block = json.loads(outputs.pop())
        assert block["bits"] == 2, "a tie goes to the narrower width"
        assert block["m.v"] == {"group_size": 64, "bits": 4, "mode": "affine"}
        assert "m.q" not in block
