# Copyright 2026 The MRound Authors
# SPDX-License-Identifier: Apache-2.0
"""The mixed precision scoring pass in less memory, and the same scores.

M1c (MEMORY.md, 2026-10-11) took three levers to the scoring pass, each the
same arithmetic in less memory: every block runs under gradient checkpointing,
so the backward holds one block's activations; each width keeps its searched
grid rather than its perturbation, which is rebuilt from the original weight
when the layer's gradient arrives; and the gradients are reduced and dropped
a block at a time, last block first. ``test_scoring.py`` holds the scores to
finite differences and the NumPy definition; these tests hold the levers to
the scores they had, to the bit, and to the memory they were taken for.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core", reason="the scoring pass is the MLX layer")

from mlx.utils import tree_flatten  # noqa: E402

from mround.core.quantizer import fake_quantize  # noqa: E402
from mround.core.scale_search import search_scales  # noqa: E402
from mround.pipeline import blocks, scoring  # noqa: E402
from mround.schemes import QuantScheme, ScaleInit, Symmetry  # noqa: E402

SEARCHED_2 = QuantScheme(
    bits=2, group_size=64, symmetry=Symmetry.SYMMETRIC, scale_init=ScaleInit.SEARCHED
)


def _model(hidden: int = 128, layers: int = 3, intermediate: int = 256) -> Any:
    qwen2 = pytest.importorskip("mlx_lm.models.qwen2", reason="a block stack from mlx-lm")
    args = qwen2.ModelArgs(
        model_type="qwen2",
        hidden_size=hidden,
        num_hidden_layers=layers,
        intermediate_size=intermediate,
        num_attention_heads=4,
        num_key_value_heads=2,
        rms_norm_eps=1e-6,
        vocab_size=256,
        tie_word_embeddings=True,
    )
    mx.random.seed(1)
    model = qwen2.Model(args)
    model.set_dtype(mx.bfloat16)
    mx.eval(model.parameters())
    return model


def _batches(rows: int = 2, tokens: int = 33, count: int = 2) -> list[Any]:
    mx.random.seed(2)
    return [mx.random.randint(0, 256, (rows, tokens)) for _ in range(count)]


@contextlib.contextmanager
def _not_checkpointed(_model: Any, _blocks: Any) -> Iterator[None]:
    yield


class TestCheckpointedBlocks:
    def test_the_forward_and_the_gradients_are_unchanged(self) -> None:
        model = _model()
        batch = _batches(count=1)[0]
        layer = model.model.layers[1].mlp.up_proj

        def loss(weight: Any) -> Any:
            layer.weight = weight
            return model(batch).astype(mx.float32).sum()

        weight = layer.weight
        plain_value, plain = mx.value_and_grad(loss)(weight)
        with blocks.checkpointed_blocks(model, blocks.discover_blocks(model)):
            value, grad = mx.value_and_grad(loss)(weight)
        assert float(value) == float(plain_value)
        np.testing.assert_array_equal(
            np.array(grad.astype(mx.float32)), np.array(plain.astype(mx.float32))
        )

    def test_the_blocks_are_restored_on_every_path(self) -> None:
        model = _model()
        found = blocks.discover_blocks(model)
        before = [block.module for block in found]
        seen: list[bool] = []

        def fail_inside() -> None:
            with blocks.checkpointed_blocks(model, found):
                seen.append(
                    all(m is not b for m, b in zip(model.model.layers, before, strict=True))
                )
                msg = "raised inside"
                raise RuntimeError(msg)

        with pytest.raises(RuntimeError, match="inside"):
            fail_inside()
        assert seen == [True]
        assert all(m is b for m, b in zip(model.model.layers, before, strict=True))

    def test_a_stand_in_answers_for_its_block(self) -> None:
        model = _model()
        with blocks.checkpointed_blocks(model, blocks.discover_blocks(model)):
            stand_in = model.model.layers[0]
            assert stand_in.mlp is stand_in._wrapped.mlp
            with pytest.raises(AttributeError, match="no_such"):
                _ = stand_in.no_such


class TestScoresUnchanged:
    """The levers change where memory goes and nothing the scores are made of."""

    @pytest.mark.parametrize("source", ["widest", "own_width"])
    def test_checkpointing_moves_no_score(
        self, monkeypatch: pytest.MonkeyPatch, source: str
    ) -> None:
        model, batches = _model(hidden=64, layers=2, intermediate=128), _batches()
        with_checkpoints = scoring.score_widths(
            model, batches, [2, 3, 4], SEARCHED_2, gradient_source=source
        ).scores
        monkeypatch.setattr(scoring, "checkpointed_blocks", _not_checkpointed)
        without = scoring.score_widths(
            model, batches, [2, 3, 4], SEARCHED_2, gradient_source=source
        ).scores
        assert with_checkpoints == without

    def test_a_rebuilt_perturbation_is_the_stored_one(self) -> None:
        # What every width's perturbation was before M1c: the quantize-
        # dequantize of the float32 original, searched first, less the
        # original, all held at once. Rebuilt per layer it is the same array.
        model = _model()
        layer = model.model.layers[0].self_attn.q_proj
        weight32 = layer.weight.astype(mx.float32)
        for bits in (2, 3, 4):
            scheme = QuantScheme(
                bits=bits, group_size=64, symmetry=Symmetry.SYMMETRIC, scale_init=ScaleInit.SEARCHED
            )
            grid = scoring._grids({"q": layer.weight}, scheme)["q"]
            stored = fake_quantize(
                weight32,
                None,
                scheme,
                eps=1e-5,
                init_scale=search_scales(weight32, scheme),
            )
            rebuilt = scoring._quantized(weight32, scheme, 1e-5, grid)
            np.testing.assert_array_equal(np.array(rebuilt - weight32), np.array(stored - weight32))

    def test_the_model_comes_back_as_it_went_in(self) -> None:
        model = _model(hidden=64, layers=2, intermediate=128)
        before = {
            name: np.array(value.astype(mx.float32))
            for name, value in tree_flatten(model.parameters())
        }
        layers_before = list(model.model.layers)
        scoring.score_widths(model, _batches(), [2, 4], SEARCHED_2, gradient_source="widest")
        after = dict(tree_flatten(model.parameters()))
        assert before.keys() == after.keys()
        for name, value in before.items():
            np.testing.assert_array_equal(
                np.array(after[name].astype(mx.float32)), value, err_msg=name
            )
        assert all(m is b for m, b in zip(model.model.layers, layers_before, strict=True))


class TestScoringMemory:
    @pytest.mark.parametrize("source", ["widest", "own_width"])
    def test_the_pass_holds_about_one_quantized_copy(self, source: str) -> None:
        # Where the weights dominate (many parameters, few tokens), the pass
        # held the quantized copy, a perturbation per candidate and every
        # gradient at once: 5.48 float32 copies of the block weights under
        # ``widest`` and 5.01 under ``own_width`` at three widths on this
        # model, measured on the CPU backend before M1c. It now holds the copy
        # the backward runs through and a layer's worth of everything else,
        # 1.37 and 1.25. Observed range scales, so that the grid search, whose
        # temporaries M1c does not touch, is not what is measured.
        model = _model(hidden=256, layers=8, intermediate=1024)
        copy = 4 * sum(
            value.size
            for name, value in tree_flatten(model.parameters())
            if ".layers." in name and name.endswith("proj.weight")
        )
        scheme = QuantScheme(bits=2, group_size=64, symmetry=Symmetry.SYMMETRIC)
        base = mx.get_active_memory()
        mx.reset_peak_memory()
        scoring.score_widths(
            model, _batches(rows=1, tokens=17), [2, 3, 4], scheme, gradient_source=source
        )
        assert mx.get_peak_memory() - base < 2.0 * copy
        # And nothing is left behind.
        assert mx.get_active_memory() - base < 0.1 * copy

    def test_the_backward_holds_one_block_s_activations(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Where the activations dominate (few parameters, many tokens), the
        # pass is measured at two depths with checkpointing and without, and
        # what is compared is how much it grows with the depth: under
        # checkpointing a block's activations are recomputed in the backward
        # and released, so the twelve extra blocks add their quantized copy
        # and little else. From four blocks to sixteen it grew by 4.7 MB with
        # checkpointing and 72.0 without.
        #
        # On the CPU stream, deliberately. On the Mac's GPU the same pass grew
        # by 43.6 MB with checkpointing and 72.0 without (2026-10-11), at op
        # sizes a kernel launch costs more than the arithmetic, and the figure
        # moved between runs. At op sizes near a 3B scoring pass's the GPU
        # removed 95 percent of the growth (scripts/probe_checkpointing.py,
        # log0148, MEMORY.md "M1b, M1c and M1d"), so what the toy measured on
        # the GPU was its small kernels; this test holds the schedule itself,
        # on the backend that frees an array when its last reader is done.
        scheme = QuantScheme(bits=2, group_size=64, symmetry=Symmetry.SYMMETRIC)

        def peak(layers: int) -> int:
            model = _model(hidden=64, layers=layers, intermediate=256)
            batches = _batches(rows=2, tokens=257, count=1)
            mx.eval(batches)
            base = mx.get_active_memory()
            mx.reset_peak_memory()
            scoring.score_widths(model, batches, [2, 3, 4], scheme, gradient_source="widest")
            return int(mx.get_peak_memory() - base)

        previous = mx.default_device()
        mx.set_default_device(mx.cpu)
        try:
            # Once untimed, so that nothing a first call allocates is counted.
            peak(4)
            checkpointed = peak(16) - peak(4)
            monkeypatch.setattr(scoring, "checkpointed_blocks", _not_checkpointed)
            plain = peak(16) - peak(4)
        finally:
            mx.set_default_device(previous)
        assert checkpointed < 0.25 * plain
