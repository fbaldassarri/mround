# Copyright 2026 The MRound Authors
# SPDX-License-Identifier: Apache-2.0
"""The zero-shot gate's verdict, on result sets built by hand.

``scripts/check_zeroshot.py`` decides whether MRound's evaluator may record a
number (MEMORY.md D-049), so what it calls agreement is tested here without
either evaluator: float32 noise passes, a request that moved fails, a decision
that changed fails unless its choices were a near tie, and a per document
quantity such as truthfulqa_mc2's is compared by value rather than as a
decision, which is the mistake its first draft made. Its defaults are tested
too, because they are the gate: MLX on the CPU stream, and ``lm_eval``'s side
in float64, weights and log softmax alike, since in float32 torch's own log
softmax lifts long answers, on the Mac by more than the tolerance (MEMORY.md,
2026-09-29).
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from check_zeroshot import (
    CONTINUOUS,
    TOLERANCE,
    compare,
    lm_eval_batch_size,
    lm_eval_model_args,
    parse_arguments,
)


def sample(doc: int, scores: list[float], **metrics: Any) -> dict[str, Any]:
    """One document as ``lm_eval`` logs it: a response and arguments per choice."""
    return {
        "doc_id": doc,
        "resps": [[(score, False)] for score in scores],
        "arguments": [("context", f" choice {i}") for i in range(len(scores))],
        **metrics,
    }


def results() -> dict[str, Any]:
    return {
        "samples": {
            "arc_easy": [
                sample(0, [-10.0, -12.0, -15.0], acc=1.0, acc_norm=1.0),
                sample(1, [-20.0, -20.0005], acc=1.0),
            ],
            "truthfulqa_mc2": [sample(0, [-3.0, -4.0], acc=0.7310585786)],
        }
    }


def test_identical_results_pass() -> None:
    report = compare(results(), results())
    assert report["passed"]
    assert report["max_abs"] == 0.0


def test_float32_noise_passes() -> None:
    theirs = results()
    theirs["samples"]["arc_easy"][0]["resps"][0] = [(-10.0 + TOLERANCE / 2, False)]
    theirs["samples"]["truthfulqa_mc2"][0]["acc"] += CONTINUOUS / 10
    assert compare(results(), theirs)["passed"]


def test_a_long_answer_is_judged_relative_to_itself() -> None:
    # One and a half millinats is outside the absolute tolerance. On a 170 nat
    # answer it is nine parts in a million, inside the relative one; on a 12
    # nat answer it is not, and nor are two and a half millinats on 170 nats,
    # fifteen parts in a million.
    ours, theirs = results(), results()
    ours["samples"]["arc_easy"][0]["resps"][2] = [(-170.0, False)]
    theirs["samples"]["arc_easy"][0]["resps"][2] = [(-170.0015, False)]
    assert compare(ours, theirs)["passed"]
    short = copy.deepcopy(theirs)
    short["samples"]["arc_easy"][0]["resps"][1] = [(-12.0015, False)]
    assert compare(ours, short)["outside_tolerance"] == [["arc_easy", 0, 1]]
    wider = copy.deepcopy(theirs)
    wider["samples"]["arc_easy"][0]["resps"][2] = [(-170.0025, False)]
    assert compare(ours, wider)["outside_tolerance"] == [["arc_easy", 0, 2]]


def test_a_request_that_moved_fails() -> None:
    theirs = results()
    theirs["samples"]["arc_easy"][0]["resps"][1] = [(-12.01, False)]
    report = compare(results(), theirs)
    assert not report["passed"]
    assert report["outside_tolerance"] == [["arc_easy", 0, 1]]


def test_a_greedy_flag_that_differs_fails() -> None:
    theirs = results()
    theirs["samples"]["arc_easy"][0]["resps"][0] = [(-10.0, True)]
    assert not compare(results(), theirs)["passed"]


def test_a_changed_decision_fails_unless_it_was_a_near_tie() -> None:
    theirs = results()
    theirs["samples"]["arc_easy"][0]["acc"] = 0.0
    report = compare(results(), theirs)
    assert not report["passed"]
    assert report["decision_flips"] == [
        {"task": "arc_easy", "doc": 0, "metric": "acc", "near_tie": False}
    ]
    # Document 1's two choices are half a millinat apart: a flip there is
    # rounding deciding a coin toss, reported and not failed.
    theirs = results()
    theirs["samples"]["arc_easy"][1]["acc"] = 0.0
    report = compare(results(), theirs)
    assert report["passed"]
    assert report["decision_flips"][0]["near_tie"]


def test_a_quantity_is_compared_by_value_not_as_a_decision() -> None:
    # Two parts in ten thousand, a number rather than a multiple of CONTINUOUS,
    # so that loosening the gate's own constant fails here.
    theirs = results()
    theirs["samples"]["truthfulqa_mc2"][0]["acc"] += 2e-4
    report = compare(results(), theirs)
    assert not report["passed"]
    assert report["continuous_drifts"] == [{"task": "truthfulqa_mc2", "doc": 0, "metric": "acc"}]
    assert report["decision_flips"] == []


def test_different_requests_are_not_compared_at_all() -> None:
    theirs = copy.deepcopy(results())
    theirs["samples"]["arc_easy"].pop()
    report = compare(results(), theirs)
    assert not report["passed"]
    assert "different requests" in report["reason"]


def test_the_defaults_are_the_gate() -> None:
    args = parse_arguments([])
    assert args.mlx_device == "cpu"
    assert args.lm_eval_dtype == "float64"
    assert lm_eval_model_args("org/model", args.lm_eval_dtype) == {
        "pretrained": "org/model",
        "dtype": "float64",
        "add_bos_token": False,
        "softmax_dtype": "float64",
    }
    # Half the float32 batch, so a batch's float64 logits fit beside the model.
    assert lm_eval_batch_size("float64") == lm_eval_batch_size("float32") // 2


def test_the_earlier_comparison_can_be_repeated() -> None:
    args = parse_arguments(["--lm-eval-dtype", "float32", "--mlx-device", "gpu"])
    assert (args.lm_eval_dtype, args.mlx_device) == ("float32", "gpu")
    model_args = lm_eval_model_args("org/model", args.lm_eval_dtype)
    assert model_args["dtype"] == model_args["softmax_dtype"] == "float32"
