# Copyright 2026 The MRound Authors
# SPDX-License-Identifier: Apache-2.0
"""The block loop's memory levers, and that none of them moves a number.

Each lever here (MEMORY.md, 2026-10-10) was chosen because it lowers the peak
without changing what the loop computes: the targets overwrite the reference
inputs they come from, the quantized stream advances in place, finished
blocks give their weights back, and the codes wait for the export one byte
each. These tests hold each lever to that promise where a unit can, comparing
to the bit rather than to a tolerance. Whether a whole run on Metal writes the
checkpoint it wrote before is a measurement on the Mac, recorded in MEMORY.md.
The scale search's slices (M1d, 2026-10-11) are held to the same promise.
"""

from __future__ import annotations

import inspect
from typing import Any

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core", reason="the block loop is the MLX layer")

from mlx import nn  # noqa: E402

from mround import api  # noqa: E402
from mround.core import scale_search  # noqa: E402
from mround.core.quantizer import quantize_codes  # noqa: E402
from mround.formats import mlx_export  # noqa: E402
from mround.pipeline import blocks, loader, runner  # noqa: E402
from mround.pipeline.device import StageMeter  # noqa: E402
from mround.schemes import QuantScheme, ScaleInit, Symmetry, TuningConfig  # noqa: E402

STAGES = {"targets", "search", "score", "tune", "commit", "advance"}


def _same_bits(a: Any, b: Any) -> bool:
    """Equal to the bit, dtype included, which a tolerance cannot say.

    Compared as raw words, because NumPy has no bfloat16 to hold them in.
    """
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    if a.dtype == mx.bfloat16:
        a, b = a.view(mx.uint16), b.view(mx.uint16)
    return bool(np.array(a).tobytes() == np.array(b).tobytes())


class TestNarrowCodes:
    """The codes wait for the export one byte each, and export the same."""

    @pytest.mark.parametrize(
        ("bits", "symmetry", "dtype"),
        [
            (2, Symmetry.SYMMETRIC, "int8"),
            (8, Symmetry.SYMMETRIC, "int8"),
            (4, Symmetry.ASYMMETRIC, "uint8"),
            (8, Symmetry.ASYMMETRIC, "uint8"),
        ],
    )
    def test_the_narrow_type_holds_every_code(
        self, bits: int, symmetry: Symmetry, dtype: str
    ) -> None:
        scheme = QuantScheme(bits=bits, group_size=32, symmetry=symmetry)
        lo, hi = scheme.code_range
        codes = mx.array([[[float(lo), float(hi), 0.0 if lo < 0 else float(hi // 2)]]])
        narrowed = mlx_export.narrow_codes(codes, scheme)
        assert str(narrowed.dtype).endswith(dtype)
        assert np.array_equal(np.array(narrowed).astype(np.float64), np.array(codes))

    @pytest.mark.parametrize("bits", [2, 3, 4, 5, 6, 8])
    @pytest.mark.parametrize("symmetry", [Symmetry.SYMMETRIC, Symmetry.ASYMMETRIC])
    def test_the_exported_tensors_do_not_depend_on_how_the_codes_were_held(
        self, bits: int, symmetry: Symmetry
    ) -> None:
        scheme = QuantScheme(bits=bits, group_size=32, symmetry=symmetry)
        mx.random.seed(bits)
        weight = mx.random.normal((8, 64))
        codes, scale, zero_point = quantize_codes(weight, None, scheme)
        wide = mlx_export.QuantizedLayer(codes, scale, zero_point, scheme)
        narrow = mlx_export.QuantizedLayer(
            mlx_export.narrow_codes(codes, scheme), scale, zero_point, scheme
        )
        expected = mlx_export._tensors_for_layer("l", wide, mx.bfloat16)
        got = mlx_export._tensors_for_layer("l", narrow, mx.bfloat16)
        assert expected.keys() == got.keys()
        for key in expected:
            assert _same_bits(got[key], expected[key]), key


class _Doubler(nn.Module):  # type: ignore[misc]
    """A block that is easy to check: it doubles what it is given."""

    def __call__(self, x: Any) -> Any:
        return x * 2


class TestApplyBlockInPlace:
    """The streams are advanced over themselves."""

    def test_in_place_writes_the_same_outputs_into_the_inputs_list(self) -> None:
        inputs = [mx.array([1.0, 2.0]), mx.array([3.0])]
        shared = list(inputs)
        cache = blocks.ActivationCache(inputs=inputs, outputs=[], args=[(), ()], kwargs=[{}, {}])
        fresh = blocks.apply_block(_Doubler(), cache)
        written = blocks.apply_block(_Doubler(), cache, in_place=True)
        assert written is inputs
        for got, expected in zip(written, fresh, strict=True):
            assert _same_bits(got, expected)
        # A list that shared the arrays without being the cache's own keeps
        # the inputs, which is what the first block's two streams rely on.
        assert [float(a[0]) for a in shared] == [1.0, 3.0]

    def test_a_cache_whose_lists_disagree_is_still_refused(self) -> None:
        cache = blocks.ActivationCache(
            inputs=[mx.array([1.0])], outputs=[], args=[(), ()], kwargs=[{}, {}]
        )
        with pytest.raises(ValueError, match="zip"):
            blocks.apply_block(_Doubler(), cache, in_place=True)


class TestStageMeter:
    """Every stage reports at least what it held."""

    def test_a_stage_that_allocates_nothing_still_reports_what_was_live(self) -> None:
        held = mx.zeros((1 << 16,))
        mx.eval(held)
        meter = StageMeter()
        with meter.stage("idle"):
            pass
        idle = meter.stages["idle"]
        assert idle >= held.nbytes
        assert meter.peak >= idle
        assert meter.take() == {"idle": idle}
        assert meter.take() == {}


def _tiny_model(seed: int = 1) -> Any:
    """Two Qwen2 blocks whose down projection cannot be grouped by 32."""
    qwen2 = pytest.importorskip("mlx_lm.models.qwen2", reason="a block stack from mlx-lm")
    args = qwen2.ModelArgs(
        model_type="qwen2",
        hidden_size=64,
        num_hidden_layers=2,
        intermediate_size=48,
        num_attention_heads=2,
        rms_norm_eps=1e-6,
        vocab_size=64,
        num_key_value_heads=1,
    )
    mx.random.seed(seed)
    model = qwen2.Model(args)
    model.set_dtype(mx.bfloat16)
    mx.eval(model.parameters())
    return model


def _run(model: Any, *, quantized_inputs: bool = True) -> tuple[Any, dict[str, Any]]:
    scheme = QuantScheme(bits=2, group_size=32, symmetry=Symmetry.SYMMETRIC)
    tuning = TuningConfig(iters=3, n_samples=4, seq_len=16, batch_size=2)
    mx.random.seed(2)
    batches = [mx.random.randint(0, 64, (2, 16)) for _ in range(2)]
    return runner.QuantizationRunner(scheme, tuning, quantized_inputs=quantized_inputs).run(
        model, batches
    )


class TestRunMemory:
    """What a run leaves behind, and what it says about its memory."""

    @pytest.mark.parametrize("quantized_inputs", [True, False])
    def test_finished_blocks_give_back_the_weights_of_their_quantized_layers(
        self, quantized_inputs: bool
    ) -> None:
        model = _tiny_model()
        _, quantized = _run(model, quantized_inputs=quantized_inputs)
        assert quantized
        for block in blocks.discover_blocks(model):
            for name, linear in blocks.iter_quantizable_linears(block.module):
                path = f"{block.name}.{name}"
                if path in quantized:
                    assert linear.weight.size == 0, path
                else:
                    # Left dense because 48 is not a multiple of 32: kept,
                    # since the export writes it as it is.
                    assert name.endswith("down_proj")
                    assert linear.weight.shape == (64, 48)

    def test_the_codes_are_one_byte_each(self) -> None:
        _, quantized = _run(_tiny_model())
        assert {str(layer.codes.dtype) for layer in quantized.values()} == {"mlx.core.int8"}

    def test_every_block_reports_every_stage_and_the_run_the_largest(self) -> None:
        summary, _ = _run(_tiny_model())
        assert set(summary.before_blocks) == {"entrance"}
        for block in summary.blocks:
            assert set(block.stage_peaks) == STAGES
            assert block.peak_memory_bytes >= max(block.stage_peaks.values())
            assert f"peak {max(block.stage_peaks.values()) / 1e9:.2f} GB in" in block.describe()
        by_stage = summary.stage_peaks()
        assert summary.peak_memory_bytes >= max(by_stage.values())
        record = summary.memory_record()
        assert record["peak_bytes"] == summary.peak_memory_bytes
        assert record["by_stage_bytes"] == by_stage
        assert [entry["block"] for entry in record["by_block_bytes"]] == [0, 1]

    def test_the_end_measurements_hold_one_batch_at_a_time(self) -> None:
        # The losses before and after a block's tuning were one lazy sum over
        # every batch, which kept many batches' forward passes alive at once:
        # the largest stage of a 3B run (25.42 GB against the steps' 20.07,
        # log0149), and 14.86 MB here against the targets' 6.35, where each
        # batch is evaluated before the next. Now the measurement holds what
        # computing the targets holds.
        #
        # On the CPU stream, for the reason the scoring pass's depth test
        # gives: at these sizes the Mac's GPU holds small kernels' buffers
        # until their command buffer completes, and measured 9.76 MB against
        # the targets' 5.84 with the measurement already taken a batch at a
        # time (2026-10-12). The schedule is held here; whether a 3B run on
        # Metal peaks at its steps is log0156's measurement.
        qwen2 = pytest.importorskip("mlx_lm.models.qwen2")
        args = qwen2.ModelArgs(
            model_type="qwen2",
            hidden_size=64,
            num_hidden_layers=2,
            intermediate_size=256,
            num_attention_heads=4,
            rms_norm_eps=1e-6,
            vocab_size=64,
            num_key_value_heads=2,
        )
        previous = mx.default_device()
        mx.set_default_device(mx.cpu)
        try:
            mx.random.seed(1)
            model = qwen2.Model(args)
            model.set_dtype(mx.bfloat16)
            mx.eval(model.parameters())
            scheme = QuantScheme(bits=2, group_size=32, symmetry=Symmetry.SYMMETRIC)
            tuning = TuningConfig(iters=2, n_samples=32, seq_len=128, batch_size=4)
            mx.random.seed(2)
            batches = [mx.random.randint(0, 64, (4, 128)) for _ in range(8)]
            summary, _ = runner.QuantizationRunner(scheme, tuning).run(model, batches)
        finally:
            mx.set_default_device(previous)
        for block in summary.blocks:
            assert block.stage_peaks["score"] < 1.1 * block.stage_peaks["targets"]

    def test_the_same_inputs_give_the_same_codes(self) -> None:
        _, first = _run(_tiny_model())
        _, second = _run(_tiny_model())
        assert first.keys() == second.keys()
        for path in first:
            assert _same_bits(first[path].codes, second[path].codes), path
            assert _same_bits(first[path].scale, second[path].scale), path


class TestLazyLoading:
    """A quantization run asks mlx-lm to leave the weights on disk."""

    @pytest.mark.parametrize("lazy", [True, False])
    def test_the_flag_reaches_mlx_lm(self, monkeypatch: pytest.MonkeyPatch, lazy: bool) -> None:
        mlx_lm = pytest.importorskip("mlx_lm")
        seen: dict[str, Any] = {}

        def fake_load(_source: str, **kwargs: Any) -> tuple[Any, Any, dict[str, Any]]:
            seen.update(kwargs)
            return nn.Linear(4, 4), object(), {"model_type": "fake"}

        monkeypatch.setattr(loader, "require_apple_silicon", lambda: None)
        monkeypatch.setattr(mlx_lm, "load", fake_load)
        loader.load_model("some/model", lazy=lazy)
        assert seen == {"return_config": True, "lazy": lazy}

    def test_the_quantize_entry_point_loads_lazily(self) -> None:
        assert "load_model(model, lazy=True)" in inspect.getsource(api.quantize)


class TestSearchInSlices:
    """The scale search takes a layer a slice of rows at a time (M1d)."""

    @pytest.mark.parametrize("bits", [2, 3, 4])
    @pytest.mark.parametrize("weighted", [False, True])
    def test_every_group_gets_the_scale_the_whole_matrix_gives_it(
        self, monkeypatch: pytest.MonkeyPatch, bits: int, weighted: bool
    ) -> None:
        scheme = QuantScheme(
            bits=bits, group_size=32, symmetry=Symmetry.SYMMETRIC, scale_init=ScaleInit.SEARCHED
        )
        mx.random.seed(bits)
        weight = mx.random.normal((37, 128)).astype(mx.bfloat16)
        importance = mx.random.uniform(shape=(128,)) if weighted else None
        whole = scale_search.search_scales(weight, scheme, importance=importance)
        # Three rows a slice, and one: thirteen slices, the last of one row,
        # and thirty seven.
        for rows in (3, 1):
            monkeypatch.setattr(scale_search, "_SLICE_BYTES", rows * scale_search._CHUNK * 128 * 4)
            sliced = scale_search.search_scales(weight, scheme, importance=importance)
            assert _same_bits(sliced, whole), rows

    def test_the_search_holds_a_slice_of_its_intermediates(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        scheme = QuantScheme(
            bits=2, group_size=64, symmetry=Symmetry.SYMMETRIC, scale_init=ScaleInit.SEARCHED
        )
        mx.random.seed(0)
        weight = mx.random.normal((1024, 512))
        mx.eval(weight)

        def peak() -> int:
            base = mx.get_active_memory()
            mx.reset_peak_memory()
            mx.eval(scale_search.search_scales(weight, scheme))
            return int(mx.get_peak_memory() - base)

        whole = peak()
        monkeypatch.setattr(scale_search, "_SLICE_BYTES", 128 * scale_search._CHUNK * 512 * 4)
        assert peak() < 0.3 * whole
