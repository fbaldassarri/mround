# Copyright 2026 The MRound Authors
# SPDX-License-Identifier: Apache-2.0
"""A step whose batch runs in pieces is still that batch's step.

``TuningConfig.micro_batches`` (MEMORY.md, M1b) splits each tuning step's
batch into pieces so the step holds one piece's activations. What it must not
change is the step: the loss is the whole batch's mean, and the outlier rule
excludes the whole batch's largest errors, which is why its threshold is taken
over the whole batch before any piece is differentiated. These tests hold the
pieces to the whole batch's loss and gradient, to a tolerance where the order
of the additions differs and exactly where it does not, and show that a
threshold taken piece by piece would have been a different rule. Whether the
change of order moves a whole run's checkpoint on Metal is an A/B on the Mac.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core", reason="the tuning step is the MLX layer")

from mlx.utils import tree_flatten, tree_unflatten  # noqa: E402

from mround.core.losses import (  # noqa: E402
    outlier_suppressed_loss,
    outlier_threshold,
    piece_loss,
    reconstruction_loss,
)
from mround.exceptions import ArchitectureError, SchemeError  # noqa: E402
from mround.pipeline import blocks, runner  # noqa: E402
from mround.schemes import QuantScheme, Symmetry, TuningConfig  # noqa: E402


class TestPieceBounds:
    @pytest.mark.parametrize(
        ("size", "pieces", "expected"),
        [
            (8, 1, [(0, 8)]),
            (8, 4, [(0, 2), (2, 4), (4, 6), (6, 8)]),
            (8, 3, [(0, 3), (3, 6), (6, 8)]),
            (2, 8, [(0, 1), (1, 2)]),
            (1, 4, [(0, 1)]),
        ],
    )
    def test_contiguous_even_and_never_empty(
        self, size: int, pieces: int, expected: list[tuple[int, int]]
    ) -> None:
        assert blocks.piece_bounds(size, pieces) == expected


class TestPieceLoss:
    """The pieces' losses add up to the batch's, and so do their gradients."""

    def _residual(self) -> Any:
        mx.random.seed(0)
        return mx.random.normal((4, 32, 16))

    def test_one_piece_with_the_batch_threshold_is_the_outlier_loss_to_the_bit(self) -> None:
        residual = self._residual()
        zero = mx.zeros_like(residual)
        whole = outlier_suppressed_loss(residual, zero)
        piece = piece_loss(
            residual, zero, elements=residual.size, threshold=outlier_threshold(residual)
        )
        assert float(whole) == float(piece)

    def test_pieces_add_up_to_the_outlier_loss_and_its_gradient(self) -> None:
        residual = self._residual()
        zero = mx.zeros_like(residual)
        threshold = outlier_threshold(residual)

        def whole(predicted: Any) -> Any:
            return outlier_suppressed_loss(predicted, zero)

        def in_pieces(predicted: Any) -> Any:
            shares = [
                piece_loss(
                    predicted[low:high],
                    zero[low:high],
                    elements=residual.size,
                    threshold=threshold,
                )
                for low, high in blocks.piece_bounds(4, 3)
            ]
            return shares[0] + shares[1] + shares[2]

        loss, grad = mx.value_and_grad(whole)(residual)
        pieces_loss, pieces_grad = mx.value_and_grad(in_pieces)(residual)
        assert float(pieces_loss) == pytest.approx(float(loss), rel=1e-6)
        # Each element's gradient is its own: the same arithmetic, so exact,
        # and zero at exactly the same excluded elements.
        np.testing.assert_array_equal(np.array(pieces_grad), np.array(grad))
        excluded = int((np.array(grad) == 0).sum())
        assert excluded == max(1, int(residual.size * 0.001))

    def test_without_a_threshold_the_pieces_add_up_to_the_mean(self) -> None:
        residual = self._residual()
        zero = mx.zeros_like(residual)
        total = sum(
            float(piece_loss(residual[low:high], zero[low:high], elements=residual.size))
            for low, high in blocks.piece_bounds(4, 4)
        )
        assert total == pytest.approx(float(reconstruction_loss(residual, zero)), rel=1e-6)

    def test_a_threshold_taken_piece_by_piece_would_be_another_rule(self) -> None:
        # The batch's largest errors all in its first sequence. The batch's
        # rule excludes its two largest errors, both there; a rule taken piece
        # by piece would exclude one there and the largest small error of each
        # other piece, which is a different objective. This is why the
        # threshold is taken over the whole batch first.
        rng = np.random.default_rng(0)
        residual = (0.01 * rng.uniform(0.5, 1.5, size=(4, 32, 16))).astype(np.float32)
        residual[0, 0, :8] = 100.0 + np.arange(8, dtype=np.float32)
        whole = mx.array(residual)
        assert max(1, int(whole.size * 0.001)) == 2
        batch = np.abs(residual) >= float(outlier_threshold(whole))
        by_piece = np.concatenate(
            [
                np.abs(residual[low:high]) >= float(outlier_threshold(whole[low:high]))
                for low, high in blocks.piece_bounds(4, 4)
            ]
        )
        assert batch.sum() == 2
        assert batch[0, 0, 6:8].all()
        assert by_piece.sum() == 4
        assert by_piece[0, 0, 7]
        assert not by_piece[0, 0, 6]
        assert by_piece[1:].sum() == 3


class TestTuningConfig:
    def test_one_piece_by_default(self) -> None:
        assert TuningConfig().micro_batches == 1

    @pytest.mark.parametrize("pieces", [0, 9])
    def test_refuses_pieces_the_batch_cannot_have(self, pieces: int) -> None:
        with pytest.raises(SchemeError, match="micro_batches"):
            TuningConfig(batch_size=8, micro_batches=pieces)

    def test_gradient_accumulation_is_still_refused_and_points_here(self) -> None:
        with pytest.raises(SchemeError, match="micro_batches splits"):
            TuningConfig(gradient_accumulate_steps=2)


class TestRequireBatchFree:
    def test_the_masks_mlx_lm_passes_are_accepted(self) -> None:
        hidden = mx.zeros((2, 16, 8))
        blocks.require_batch_free(hidden, ("causal", None), {}, block="b")
        blocks.require_batch_free(hidden, (mx.zeros((16, 16)), None), {}, block="b")

    def test_an_array_that_may_hold_the_batch_is_refused(self) -> None:
        hidden = mx.zeros((2, 16, 8))
        with pytest.raises(ArchitectureError, match="micro_batches=1"):
            blocks.require_batch_free(hidden, (), {"mask": mx.zeros((2, 1, 16, 16))}, block="b")


def _tiny_model() -> Any:
    qwen2 = pytest.importorskip("mlx_lm.models.qwen2", reason="a block stack from mlx-lm")
    args = qwen2.ModelArgs(
        model_type="qwen2",
        hidden_size=64,
        num_hidden_layers=2,
        intermediate_size=128,
        num_attention_heads=2,
        rms_norm_eps=1e-6,
        vocab_size=64,
        num_key_value_heads=1,
    )
    mx.random.seed(1)
    model = qwen2.Model(args)
    model.set_dtype(mx.bfloat16)
    mx.eval(model.parameters())
    return model


def _block_under_tuning(model: Any, batch: Any) -> tuple[Any, Any, dict[str, Any]]:
    """Block zero with its inputs, targets and tunable stand-ins installed."""
    block = blocks.discover_blocks(model)[0]
    cache = blocks.capture_block_inputs(model, block, [batch])
    cache.inputs = [hidden.astype(mx.float32) for hidden in cache.inputs]
    cache.outputs = [target.astype(mx.float32) for target in cache.outputs]
    scheme = QuantScheme(bits=2, group_size=32, symmetry=Symmetry.SYMMETRIC)
    tunable = runner._tunable_type()
    tunables = {
        name: tunable(linear, scheme, 1e-5)
        for name, linear in blocks.iter_quantizable_linears(block.module)
    }
    block.module.update_modules(tree_unflatten(list(tunables.items())))
    return block, cache, tunables


class TestStepInPieces:
    @pytest.mark.parametrize("suppress", [True, False])
    @pytest.mark.parametrize("pieces", [2, 4])
    def test_the_step_is_the_whole_batch_s(self, suppress: bool, pieces: int) -> None:
        mx.random.seed(3)
        block, cache, tunables = _block_under_tuning(
            _tiny_model(), mx.random.randint(0, 64, (4, 16))
        )
        params = {name: dict(layer.params) for name, layer in tunables.items()}
        loss_fn = outlier_suppressed_loss if suppress else reconstruction_loss

        def whole(values: Any) -> Any:
            for name, layer in tunables.items():
                layer.params = values[name]
            predicted = blocks.block_output(
                block.module, cache.inputs[0], cache.args[0], cache.kwargs[0]
            )
            return loss_fn(predicted, cache.outputs[0])

        loss, grads = mx.value_and_grad(whole)(params)
        tuning = TuningConfig(iters=1, n_samples=4, seq_len=16, batch_size=4, micro_batches=pieces)
        quantizer = runner.QuantizationRunner(
            QuantScheme(bits=2, group_size=32, symmetry=Symmetry.SYMMETRIC), tuning
        )
        piece_loss_value, piece_grads = quantizer._step_in_pieces(
            block, cache, tunables, params, 0, pieces, suppress=suppress
        )
        assert float(piece_loss_value) == pytest.approx(float(loss), rel=1e-5)
        expected = dict(tree_flatten(grads))
        got = dict(tree_flatten(piece_grads))
        assert expected.keys() == got.keys()
        for key, value in expected.items():
            a, b = np.array(value), np.array(got[key])
            scale = float(np.abs(a).max()) or 1.0
            np.testing.assert_allclose(b, a, rtol=1e-4, atol=1e-5 * scale, err_msg=key)
            # What a sign step reads: the signs agree wherever the gradient is
            # not lost in the order of the additions.
            clear = np.abs(a) > 1e-4 * scale
            assert (np.sign(a[clear]) == np.sign(b[clear])).all(), key


def _run(pieces: int, *, batch: int = 4, seq: int = 16) -> tuple[Any, dict[str, Any]]:
    scheme = QuantScheme(bits=2, group_size=32, symmetry=Symmetry.SYMMETRIC)
    tuning = TuningConfig(
        iters=3, n_samples=2 * batch, seq_len=seq, batch_size=batch, micro_batches=pieces
    )
    mx.random.seed(2)
    batches = [mx.random.randint(0, 64, (batch, seq)) for _ in range(2)]
    return runner.QuantizationRunner(scheme, tuning).run(_tiny_model(), batches)


class TestRunInPieces:
    def test_both_ends_are_measured_as_the_whole_run_measures_them(self) -> None:
        whole, _ = _run(1)
        pieces, _ = _run(4)
        # Block zero's starting loss is taken before any step, from the same
        # inputs, so only the pieces' forward passes stand between the two.
        assert pieces.blocks[0].initial_loss == pytest.approx(
            whole.blocks[0].initial_loss, rel=1e-5
        )

    def test_every_block_reports_the_score_and_tune_stages(self) -> None:
        summary, _ = _run(2)
        for block in summary.blocks:
            assert {"score", "tune"} <= set(block.stage_peaks)

    def test_the_step_holds_less_the_more_pieces_it_runs_in(self) -> None:
        # Measured on the CPU backend: the tune stage held 15.9 MB whole and
        # 9.0, 5.8 and 4.2 in two, four and eight pieces. Without evaluating
        # each piece before the next, the pieces' graphs pile up and eight
        # pieces held 11.2, more than two did.
        peaks = {pieces: _run(pieces, batch=8, seq=128)[0].stage_peaks() for pieces in (1, 2, 8)}
        assert peaks[2]["tune"] < peaks[1]["tune"]
        assert peaks[8]["tune"] < peaks[2]["tune"]
        assert peaks[8]["tune"] < 0.5 * peaks[1]["tune"]
