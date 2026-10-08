# Copyright 2026 The MRound Authors
# SPDX-License-Identifier: Apache-2.0
"""The gate of MEMORY.md D-049: MRound's zero-shot evaluator against lm_eval's own.

No zero-shot number enters the ledger until MRound's evaluator agrees with the
standard one. Both score the same documents of the same tasks with the same
weights: MRound's through MLX in float32 (:mod:`mround.eval.zeroshot`),
``lm_eval``'s through its Hugging Face adapter on torch in float64, with
``add_bos_token`` false on both, the reference's evaluation setting. The
script compares every request's log likelihood and greedy flag and every
document's decision, and exits nonzero if they disagree by more than float32
arithmetic explains: a request more than :data:`TOLERANCE` nats and more than
:data:`RELATIVE` of itself apart, a greedy flag that differs, a decision that
differs where the choices were not within twice the tolerance of each other on
either side, or a per document quantity (truthfulqa_mc2's) that moves by more
than :data:`CONTINUOUS`.

What it found the first time it ran (MEMORY.md, 2026-09-28), with torch in
float32: MLX's float32 log sum of exponentials over a whole vocabulary row, on
its CPU backend, comes out about 1.3e-4 low, which moved every multi token
answer by up to 1e-3 nats against torch; the evaluator now reduces in blocks,
and on SmolLM2-135M-Instruct, twenty documents of each task, 5,325 requests,
the two agreed to 4.1e-4 nats at worst and 2.3e-5 on average, with no greedy
flag and no decision different; most of what remains is torch's own float32
rounding.

**The gate runs MLX on the CPU stream.** On the Apple GPU, MLX's float32
matrix product takes its inputs at float16 precision (MEMORY.md D-015, 8.5e-4
relative error), which no float32 tolerance survives: run there on 2026-09-28,
the two evaluators differed by 1.2e-2 nats per request on average, while the
decisions changed only where two choices were within 2e-3 nats of each other.
That is the platform's arithmetic, the same under every perplexity this project
has recorded, and not the evaluator's logic, which is what this script checks;
so it sets MLX's default device to the CPU unless told ``--mlx-device gpu``,
which measures what the GPU's precision does to the same sample instead. The
maintainer decided the same day that zero-shot rows are scored on the GPU, as
perplexity is, once this gate has passed on the CPU stream.

**lm_eval's side runs in float64.** On the Mac's CPU stream the gate failed
with torch in float32 (2026-09-28; MEMORY.md, 2026-09-29): 8 of
SmolLM2-135M-Instruct's 5,325 requests and 23 of Qwen2.5-0.5B-Instruct's
1,322 fell outside the tolerance, every one a long answer and every one with
torch above MRound. Scored in float64, MRound was within 1.2e-4 nats of the
exact value on each of the twenty worst, and torch's float32 between 9.7e-4
and 3.9e-3 above it on every one, 1e-5 to 4e-5 of the answer. The container
shows why: split into its forward and its log softmax, torch's float32 error
on nine of those requests is all in the log softmax, whose normalizer comes
out low by 8e-6 nats per token over SmolLM2's vocabulary and 2.8e-5 over
Qwen2.5's, which is three times the size, while the forward agrees with
float64 to 5.1e-5 a request; the Mac's torch errs the same way, four times as
far. A float32 reference thus brings a systematic error of its own, larger
than MRound's; a float64 one brings none worth counting, and the tolerance
bounds MRound's float32 arithmetic alone. ``--lm-eval-dtype float32`` repeats
the earlier comparison.

Usage, from the repository root, in an environment with MLX, ``lm_eval``,
torch, transformers and accelerate (``mround-dev`` has all five)::

    python scripts/check_zeroshot.py --model HuggingFaceTB/SmolLM2-135M-Instruct --limit 20

The model must be a Hugging Face repository both sides can load. ``--limit``
is documents per task; MMLU is 57 tasks, so it dominates the cost. In float64
the torch side is slower and takes four requests at a time rather than eight
(:func:`lm_eval_batch_size`).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

# Per request, in nats, or relative to the request's own log likelihood,
# whichever is looser: a long continuation sums more float32 roundings, and a
# 170 nat answer 4e-4 apart agrees to 2e-6. Measured agreement is well inside
# both; see the module docstring.
TOLERANCE = 1e-3
RELATIVE = 1e-5
# A per document metric that is not a decision (truthfulqa_mc2 scores the
# probability mass on the true answers) is compared by value, to this.
CONTINUOUS = 1e-4


def _flatten(results: dict[str, Any]) -> dict[tuple[str, int, int], tuple[float, bool, Any]]:
    """Every request's (log likelihood, greedy, arguments), keyed by task, document and choice."""
    table = {}
    for task, samples in results["samples"].items():
        for sample in samples:
            for choice, response in enumerate(sample["resps"]):
                loglikelihood, greedy = response[0]
                table[(task, int(sample["doc_id"]), choice)] = (
                    float(loglikelihood),
                    bool(greedy),
                    tuple(sample["arguments"][choice]),
                )
    return table


def _near_tie(
    table: dict[tuple[str, int, int], tuple[float, bool, Any]], task: str, doc: int
) -> bool:
    """Whether a document's two best choices are within twice the tolerance."""
    scores = sorted(
        (value[0] for key, value in table.items() if key[0] == task and key[1] == doc),
        reverse=True,
    )
    return len(scores) > 1 and scores[0] - scores[1] < 2 * TOLERANCE


def _is_decision(value: object) -> bool:
    """Whether a per document metric is a right or wrong call rather than a quantity."""
    return isinstance(value, bool | int | float) and float(value) in (0.0, 1.0)


def compare(mlx: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    """Request by request and decision by decision agreement between two result sets.

    Args:
        mlx: ``lm_eval`` results from MRound's evaluator, with samples logged.
        reference: The same from ``lm_eval``'s Hugging Face adapter.

    Returns:
        The report, whose ``passed`` says whether the gate holds.
    """
    ours, theirs = _flatten(mlx), _flatten(reference)
    if ours.keys() != theirs.keys():
        return {"passed": False, "reason": "the two evaluators scored different requests"}
    differences = []
    outside = []
    greedy = []
    for key in sorted(ours):
        (a, a_greedy, a_args), (b, b_greedy, b_args) = ours[key], theirs[key]
        if a_args != b_args:
            return {"passed": False, "reason": f"request {key} has different arguments"}
        gap = abs(a - b)
        differences.append((gap, key, a, b))
        if gap > TOLERANCE and gap > RELATIVE * abs(b):
            outside.append(key)
        if a_greedy != b_greedy:
            greedy.append(key)
    differences.sort(reverse=True)

    flips = []
    drifts = []
    for task, samples in mlx["samples"].items():
        by_doc = {int(s["doc_id"]): s for s in reference["samples"][task]}
        for sample in samples:
            doc = int(sample["doc_id"])
            for metric in ("acc", "acc_norm"):
                if metric not in sample:
                    continue
                mine, standard = sample[metric], by_doc[doc][metric]
                if _is_decision(mine) and _is_decision(standard):
                    if float(mine) != float(standard):
                        tie = _near_tie(ours, task, doc) or _near_tie(theirs, task, doc)
                        flips.append({"task": task, "doc": doc, "metric": metric, "near_tie": tie})
                elif abs(float(mine) - float(standard)) > CONTINUOUS:
                    drifts.append({"task": task, "doc": doc, "metric": metric})

    unexplained = [flip for flip in flips if not flip["near_tie"]]
    return {
        "passed": not outside and not greedy and not unexplained and not drifts,
        "requests": len(differences),
        "max_abs": differences[0][0] if differences else 0.0,
        "max_relative": max((d / max(abs(b), 1e-12) for d, _, _, b in differences), default=0.0),
        "mean_abs": sum(d[0] for d in differences) / max(1, len(differences)),
        "worst": [
            {"task": k[0], "doc": k[1], "choice": k[2], "mround": a, "lm_eval": b, "abs": d}
            for d, k, a, b in differences[:10]
        ],
        "outside_tolerance": [list(key) for key in outside],
        "greedy_mismatches": [list(key) for key in greedy],
        "decision_flips": flips,
        "continuous_drifts": drifts,
    }


def lm_eval_model_args(model: str, dtype: str) -> dict[str, Any]:
    """The Hugging Face adapter's arguments: weights and log softmax in one precision."""
    return {"pretrained": model, "dtype": dtype, "add_bos_token": False, "softmax_dtype": dtype}


def lm_eval_batch_size(dtype: str) -> int:
    """Requests per torch forward: four in float64, whose logits are twice the size.

    The adapter holds a batch's logits and their log softmax at once. Eight of
    Qwen2.5-0.5B's longest inputs, 1,271 tokens over 151,936 entries, come to 25
    GB of the two in float64, which a 32 GB Mac cannot spare; four come to 12.
    The batch changes nothing but the order of float64 roundings.
    """
    return 4 if dtype == "float64" else 8


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """The command line, whose defaults are the gate: MLX on the CPU, lm_eval in float64."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    parser.add_argument("--limit", type=int, default=20, help="Documents per task.")
    parser.add_argument("--tasks", default=None, help="Comma separated; default the suite.")
    parser.add_argument("--json", type=Path, default=None, help="Write the report here too.")
    parser.add_argument(
        "--mlx-device",
        choices=["cpu", "gpu"],
        default="cpu",
        help=(
            "where MLX runs: cpu, float32 arithmetic, is the gate; gpu measures what the "
            "GPU's float16 precision matrix product (D-015) does to the same sample"
        ),
    )
    parser.add_argument(
        "--lm-eval-dtype",
        choices=["float64", "float32"],
        default="float64",
        help=(
            "precision of lm_eval's side: float64, the exact answer, is the gate; float32 "
            "repeats the comparison of 2026-09-28"
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run both evaluators on one sample and report whether they agree."""
    args = parse_arguments(argv)

    import mlx.core as mx  # noqa: PLC0415
    from lm_eval.evaluator import simple_evaluate  # noqa: PLC0415
    from mlx_lm import load  # noqa: PLC0415

    mx.set_default_device(mx.cpu if args.mlx_device == "cpu" else mx.gpu)

    from mround.eval.zeroshot import (  # noqa: PLC0415
        DEFAULT_TASKS,
        context_window,
        evaluate_zeroshot,
        summarize,
    )

    tasks = args.tasks.split(",") if args.tasks else list(DEFAULT_TASKS)

    # The weights as the harness scores them: loaded by mlx_lm, set to float32.
    model, tokenizer, config = load(args.model, return_config=True)
    model.set_dtype(mx.float32)
    started = time.monotonic()
    summary, ours = evaluate_zeroshot(
        model,
        tokenizer,
        tasks=tasks,
        limit=args.limit,
        max_length=context_window(config),
        bootstrap_iters=1000,
        log_samples=True,
    )
    mlx_seconds = time.monotonic() - started
    # MLX keeps what it freed in its cache, gigabytes after a Qwen2.5 sized
    # vocabulary, and torch needs that memory next.
    del model
    mx.clear_cache()

    started = time.monotonic()
    theirs = simple_evaluate(
        model="hf",
        model_args=lm_eval_model_args(args.model, args.lm_eval_dtype),
        tasks=tasks,
        num_fewshot=None,
        limit=args.limit,
        batch_size=lm_eval_batch_size(args.lm_eval_dtype),
        device="cpu",
        bootstrap_iters=1000,
        log_samples=True,
        verbosity="WARNING",
    )
    hf_seconds = time.monotonic() - started

    report = compare(ours, theirs)
    report.update(
        model=args.model,
        mlx_device=args.mlx_device,
        lm_eval_dtype=args.lm_eval_dtype,
        limit=args.limit,
        tasks=tasks,
        mround_seconds=round(mlx_seconds, 1),
        lm_eval_seconds=round(hf_seconds, 1),
        n_scored_tokens=summary.n_scored_tokens,
        accuracy={
            ours_task.task: [ours_task.accuracy, theirs_task.accuracy]
            for ours_task, theirs_task in zip(summary.tasks, summarize(theirs, tasks), strict=True)
        },
    )
    print(f"\n{summary.describe()}\n")
    print(f"requests compared   {report.get('requests')}")
    largest = report.get("max_abs", float("nan"))
    relative = report.get("max_relative", float("nan"))
    print(f"largest difference  {largest:.2e} nats, {relative:.1e} relative")
    print(f"tolerance           {TOLERANCE:.0e} nats or {RELATIVE:.0e} relative")
    print(f"outside it          {len(report.get('outside_tolerance', []))}")
    print(f"mean difference     {report.get('mean_abs', float('nan')):.2e} nats")
    print(f"greedy mismatches   {len(report.get('greedy_mismatches', []))}")
    flips = report.get("decision_flips", [])
    ties = sum(1 for flip in flips if flip["near_tie"])
    print(f"decision flips      {len(flips)} ({ties} at near ties)")
    print(f"continuous drifts   {len(report.get('continuous_drifts', []))}")
    for task, (mine, standard) in report["accuracy"].items():
        print(f"  {task:<16} acc {mine:.4f} against {standard:.4f}")
    print(f"mlx device          {args.mlx_device}")
    print(f"lm_eval dtype       {args.lm_eval_dtype}")
    if report["passed"]:
        print("\nPASS: the two evaluators agree to float32 arithmetic")
    else:
        print(f"\nFAIL: {report.get('reason', 'see the counts above')}")
        if args.mlx_device == "gpu":
            print(
                "  On the GPU stream a failure at float32 tolerance is what D-015 predicts;"
                " the gate is --mlx-device cpu."
            )
        if args.lm_eval_dtype == "float32":
            print(
                "  In float32 lm_eval's own side scores long answers high (MEMORY.md,"
                " 2026-09-29); the gate is --lm-eval-dtype float64."
            )
    if args.json is not None:
        args.json.write_text(json.dumps(report, indent=1) + "\n")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
