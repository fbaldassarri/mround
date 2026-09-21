# Copyright 2026 The MRound Authors
# SPDX-License-Identifier: Apache-2.0
"""Learned rounding against round-to-nearest, on the same model in one run.

Run on an Apple Silicon Mac with the model stack installed::

    pip install -e '.[models,eval]'
    python examples/tune_model.py --eval wikitext2

This is the measurement the whole project exists to make. It quantizes one model
twice, once with nothing learned and once with the block loop, scores both
against the original, and prints the two penalties side by side. Learned rounding
has to close some of the gap that round-to-nearest opens, or it has not earned
its runtime.

Both checkpoints come out of the same loader, the same quantizer, and the same
export path, so the only difference between them is the learning. That is the
entire point of running them together rather than comparing against a number from
a previous session.

The defaults are smaller than the standard recipe so that a first run finishes in
minutes rather than an afternoon. They are not the settings a published number
should use; ``--iters 200 --samples 128 --seq-len 2048`` is.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mround.exceptions import MRoundError
from mround.schemes import ScaleInit, Symmetry

if TYPE_CHECKING:
    from collections.abc import Callable

MODEL = "mlx-community/SmolLM2-135M-Instruct"

# Reduced from the standard recipe. Enough calibration to tune against and few
# enough tokens to finish quickly on a small model.
ITERS = 200
SAMPLES = 64
SEQ_LEN = 512
BATCH_SIZE = 4

# Two accountings of the same checkpoint agree to the last bit or they disagree.
# This tolerance exists only to absorb float formatting, not real drift.
SIZE_TOLERANCE = 1e-6


def _score(source: str, dataset: str, max_tokens: int) -> tuple[float, int, dict[str, Any]]:
    """Perplexity of one checkpoint, the vocabulary size, and what was scored.

    The third value is the ledger's ``evaluation`` mapping, taken from the
    measurement rather than from the request: 65536 requested tokens score
    65504 under non overlapping 2048 token windows, and this harness scores
    at the checkpoint's own dtype where the parity harness scores at float32,
    which is why the dtype is named.
    """
    from mround.eval.perplexity import evaluate_perplexity  # noqa: PLC0415
    from mround.pipeline.loader import load_model  # noqa: PLC0415

    # allow_quantized because two of the three checkpoints scored here are
    # exactly that, and the loader refuses them by default so nothing quantizes
    # one twice by accident.
    bundle = load_model(source, allow_quantized=True)
    measured = evaluate_perplexity(
        bundle.model, bundle.tokenizer, dataset=dataset, seq_len=2048, max_tokens=max_tokens
    )
    print(f"  {measured.describe()}")
    evaluation = {
        "dataset": measured.dataset,
        "tokens": measured.n_tokens,
        "requested_tokens": max_tokens,
        "seq_len": measured.seq_len,
        "stride": measured.stride,
        "dtype": str(bundle.dtype).removeprefix("mlx.core."),
    }
    return measured.perplexity, int(bundle.config.get("vocab_size") or 0), evaluation


def _timed(label: str, work: Callable[[], Any]) -> tuple[Any, float]:
    """Run something noisy and report how long it took."""
    print(f"\n{label}")
    started = time.perf_counter()
    result = work()
    elapsed = time.perf_counter() - started
    print(f"  done in {elapsed:.1f}s")
    return result, elapsed


def _same_size(baseline: Any, tuned: Any) -> bool:
    """Whether the two checkpoints cost the same, which the comparison assumes.

    Quality at equal size is the claim. Two checkpoints of different sizes can be
    ranked on perplexity all day and the ranking means nothing, so this is
    checked rather than assumed: if the block loop ever left a tensor dense that
    the baseline quantized, the tuned model would look better for a reason that
    has nothing to do with learning.

    The byte comparison carries a small tolerance, learned from the first mixed
    run: a mixed checkpoint whose payload matched the uniform baseline to the
    predicted byte still differed by a few bytes of safetensors header, because
    per-layer packed shapes print as different digit counts, and an exact
    equality turned that into a warning about payload it was not. A genuinely
    dense-left tensor costs hundreds of kilobytes; one part in a thousand
    separates the two cleanly, and the delta is printed either way so nothing
    hides in the tolerance.
    """
    left, right = baseline.bits_per_weight, tuned.storage.bits_per_weight
    on_disk = [
        sum(f.stat().st_size for f in Path(d).glob("*.safetensors"))
        for d in (baseline.output_dir, tuned.output_dir)
    ]
    print(f"\n  round-to-nearest {left:.3f} bits per weight, {on_disk[0] / 1e6:.1f} MB on disk")
    print(f"  mround           {right:.3f} bits per weight, {on_disk[1] / 1e6:.1f} MB on disk")

    delta = on_disk[1] - on_disk[0]
    if abs(left - right) < SIZE_TOLERANCE and abs(delta) <= max(1, on_disk[0] // 1000):
        if delta:
            print(
                f"  sizes differ by {delta:+,} bytes ({delta / on_disk[0]:+.5%}): "
                f"container metadata, not payload"
            )
        return True
    print(
        "\n  WARNING: the two checkpoints are not the same size, so comparing "
        "their quality\n  ranks two different things. Whatever differs between "
        "them is doing part of the\n  work the learning is about to be credited "
        f"with. The gap is {delta:+,} bytes\n  ({delta / on_disk[0]:+.3%}) and "
        f"{right - left:+.6f} bits per weight."
    )
    for path, reason in tuned.skipped:
        print(f"    mround left dense: {path}: {reason}")
    return False


def _parse() -> argparse.Namespace:
    """Command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--iters", type=int, default=ITERS, help="tuning steps per block")
    parser.add_argument("--samples", type=int, default=SAMPLES, help="calibration sequences")
    parser.add_argument("--seq-len", type=int, default=SEQ_LEN, help="calibration tokens each")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument(
        "--eval",
        default=None,
        help="dataset or text file to score. Omitted skips scoring, which measures nothing",
    )
    parser.add_argument("--eval-tokens", type=int, default=65536)
    parser.add_argument(
        "--holdout",
        type=int,
        default=0,
        help=(
            "calibration batches kept out of tuning and used only to judge "
            "whether tuning helped. Zero, the reference's behaviour, makes that "
            "judgement on the data being fit and cannot see overfitting"
        ),
    )
    parser.add_argument(
        "--scale-init",
        choices=["observed_range", "searched"],
        default=None,
        help=(
            "how each group's scale is initialized. searched runs the v2 grid "
            "search per group, importance-weighted, which is the feature "
            "MEMORY.md D-028 says very low widths need. Left unset it resolves "
            "from the narrowest width this run can assign (D-046), which is "
            "searched below 4 bits and observed range at 4 and above; this "
            "example used to default to observed_range and would hand anyone "
            "running it at 2 bits the configuration D-030 measured at 77,365 "
            "perplexity. The RTN baseline stays observed either way, so the "
            "comparison isolates the whole v2 delta"
        ),
    )
    parser.add_argument(
        "--avg-bits",
        type=float,
        default=None,
        help=(
            "target average code bits per weight over the block layers, which "
            "engages mixed precision: a sensitivity scoring pass per candidate "
            "width, then exact allocation under the size budget the average "
            "implies. An integer average competes under exactly the uniform "
            "width's ceiling, so --avg-bits 3 against a --bits 3 run is the "
            "controlled comparison. --bits still sets the RTN baseline and the "
            "tensors outside the block loop take the supported width nearest "
            "the average"
        ),
    )
    parser.add_argument(
        "--options",
        default="2,3,4",
        help=(
            "candidate widths the planner may assign per layer, comma "
            "separated. Read only with --avg-bits. The graded default is a "
            "measured decision (MEMORY.md D-035): a menu without the "
            "intermediate width forces extreme demotions and lost to uniform "
            "at the standard recipe. Each width costs one scoring pass"
        ),
    )
    parser.add_argument(
        "--scoring-gradients",
        choices=["own_width", "widest"],
        default="widest",
        help=(
            "where the sensitivity gradients come from. widest, the default, "
            "won the A/B on both averages tested (MEMORY.md D-034); own_width "
            "is the reference's procedure, kept as the reference-faithful "
            "control. Read only with --avg-bits"
        ),
    )
    parser.add_argument(
        "--remainder-bits",
        type=int,
        default=None,
        help=(
            "width for the tensors outside the block loop in a mixed run, "
            "overriding the nearest-to-average default. At the exact midpoint "
            "the default put a tied output head at the lower width, a "
            "measured confound. Read only with --avg-bits"
        ),
    )
    parser.add_argument(
        "--quantized-inputs",
        choices=["on", "off"],
        default="on",
        help=(
            "feed each block the activations the already quantized blocks "
            "produced, rather than the original model's. On is the reference's "
            "choice and MEMORY.md Q-001's answer; off halves activation memory"
        ),
    )
    return parser.parse_args()


def _record(
    args: Any,
    tuning: Any,
    tuned: Any,
    seconds: float,
    scores: dict[str, float],
    evaluation: dict[str, Any],
) -> None:
    """Append this run to the project ledger, best-effort.

    Every measurement is paper data (MEMORY.md D-029), so scored runs record
    themselves rather than relying on someone transcribing a terminal. Best
    effort because a ledger problem must not discard twenty minutes of tuning:
    the run's own output still exists either way.
    """
    try:
        from mround.eval.results import (  # noqa: PLC0415
            RunRecord,
            append_record,
            capture_environment,
        )

        run_config = json.loads(tuned.config_path.read_text(encoding="utf-8"))
        # Read off the checkpoint rather than off the flag. Since D-046 the flag
        # can be unset while the run is still searched, and a ledger row has to
        # say what the checkpoint was, not what was asked for.
        produced = next(iter(tuned.scheme_by_layer.values()), None)
        scheme: dict[str, Any] = {
            "bits": None if tuned.mixed is not None else args.bits,
            "group_size": args.group_size,
            "symmetry": "sym",
            "scale_init": None if produced is None else produced.scale_init.value,
        }
        notes = tuned.summary.describe()
        if tuned.mixed is not None:
            scheme["avg_bits"] = tuned.mixed["average_bits"]
            scheme["candidate_bits"] = tuned.mixed["candidate_bits"]
            scheme["scoring_gradients"] = tuned.mixed["gradient_source"]
            notes += (
                f"; mixed {tuned.mixed['layers_by_width']} at "
                f"{tuned.mixed['achieved_code_bits']:.3f} achieved code bits, "
                f"remainder {tuned.mixed['remainder_bits']} bits, "
                f"scored in {tuned.mixed['scoring_seconds']}s"
            )
        record = RunRecord(
            kind="quantize",
            model=args.model,
            scheme=scheme,
            tuning={
                "iters": tuning.iters,
                "n_samples": tuning.n_samples,
                "seq_len": tuning.seq_len,
                "batch_size": tuning.batch_size,
                "quantized_inputs": args.quantized_inputs == "on",
                "holdout_batches": args.holdout,
            },
            calibration=run_config.get("calibration"),
            evaluation=evaluation,
            perplexity={
                "original": scores["original"],
                "rtn": scores["round-to-nearest"],
                "quantized": scores["mround"],
            },
            storage={
                "bits_per_weight": tuned.storage.bits_per_weight,
                "disk_bytes": tuned.storage.stored_bytes,
            },
            cost={
                "quantize_seconds": round(seconds, 1),
                "peak_memory_bytes": tuned.summary.peak_memory_bytes,
                "device": "gpu",
            },
            environment=capture_environment(),
            notes=notes,
        )
        print(f"\n  recorded in {append_record(record)}")
    except Exception as exc:
        print(f"\n  WARNING: this run was not recorded in the ledger: {exc}")
        print("  The numbers above are otherwise unaffected. Record it by hand: D-029.")


def _run_names(args: Any, widths: list[int]) -> tuple[str, str]:
    """Output directory stem and the banner line for this configuration.

    The stem carries the model, learned the hard way: the first run on a
    second model silently overwrote the first model's checkpoints, because
    the stem carried only the scheme and every model shared it.
    """
    slug = str(args.model).rstrip("/").rsplit("/", 1)[-1].lower()
    if args.avg_bits is not None:
        joined = "-".join(str(width) for width in widths)
        stem = f"{slug}-avg{args.avg_bits:g}bit-o{joined}-g{args.group_size}"
        if args.scoring_gradients == "widest":
            stem += "-widest"
        if args.remainder_bits is not None:
            stem += f"-r{args.remainder_bits}"
        banner = (
            f"{args.model} at an average of {args.avg_bits:g} code bits "
            f"over {widths}, group size {args.group_size}"
        )
    else:
        stem = f"{slug}-{args.bits}bit-g{args.group_size}"
        banner = f"{args.model} at {args.bits} bits, group size {args.group_size}"
    # The resolved value, not the flag. Two runs that differ only in how the
    # scale started are two experiments, and after D-046 the flag can be unset
    # while the run is still searched.
    if _resolved_scale_init(args, widths) is ScaleInit.SEARCHED:
        stem += "-searched"
    return stem, banner


def _resolved_scale_init(args: Any, widths: list[int]) -> ScaleInit:
    """What this run's scale initialization will actually be.

    Asked three times, for the directory name, for the run and for the ledger
    row, so it is resolved once here. The rule itself lives in the API rather
    than being restated: the narrowest width a run can assign decides, and a
    mixed run over {2, 3, 4} is searched even though its base width is 4.
    """
    from mround.api import default_scale_init  # noqa: PLC0415

    if args.scale_init is not None:
        return ScaleInit(args.scale_init)
    narrowest = min(widths) if args.avg_bits is not None else int(args.bits)
    # This example has no asymmetric option, so the symmetry it resolves under
    # is never in question.
    return default_scale_init(narrowest, Symmetry.SYMMETRIC)


def main() -> int:
    """Quantize twice, score three times, and report the two penalties."""
    args = _parse()

    from mround import api  # noqa: PLC0415
    from mround.schemes import TuningConfig  # noqa: PLC0415

    widths = sorted({int(part) for part in args.options.split(",") if part.strip()})
    stem, banner = _run_names(args, widths)
    baseline_dir = f"./rtn-{stem}"
    tuned_dir = f"./mround-{stem}"

    tuning = TuningConfig(
        iters=args.iters,
        n_samples=args.samples,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
    )

    print(banner)
    print(f"calibration {args.samples} x {args.seq_len} tokens, {args.iters} steps per block")

    try:
        baseline, _ = _timed(
            "round-to-nearest",
            lambda: api.quantize_round_to_nearest(
                args.model,
                output_dir=baseline_dir,
                bits=args.bits,
                group_size=args.group_size,
            ),
        )
        print(f"  {baseline.describe()}")

        tuned, tuned_seconds = _timed(
            "learned rounding",
            lambda: api.quantize(
                args.model,
                output_dir=tuned_dir,
                bits=args.bits,
                group_size=args.group_size,
                average_bits=args.avg_bits,
                candidate_bits=widths,
                sensitivity_gradients=args.scoring_gradients,
                remainder_bits=args.remainder_bits,
                tuning=tuning,
                quantized_inputs=args.quantized_inputs == "on",
                holdout_batches=args.holdout,
                scale_init=_resolved_scale_init(args, widths),
            ),
        )
        print(f"  {tuned.summary.describe()}")
        print(f"  {tuned.describe()}")
    except MRoundError as exc:
        print(f"\nfailed: {exc}")
        return 1

    comparable = _same_size(baseline, tuned)

    if args.eval is None:
        print("\nno evaluation set given, so quality is unmeasured. Pass --eval to score it.")
        return 0

    print(f"\nscoring on {args.eval}")
    scores = {}
    vocab = 0
    evaluation: dict[str, Any] = {}
    for label, source in (
        ("original", args.model),
        ("round-to-nearest", baseline_dir),
        ("mround", tuned_dir),
    ):
        print(f"  {label}")
        scores[label], vocab, evaluation = _score(source, args.eval, args.eval_tokens)

    _record(args, tuning, tuned, tuned_seconds, scores, evaluation)

    # Penalties, never the raw perplexities. Perplexity moves with how much text
    # was scored, so two runs with different --eval-tokens are not comparable in
    # absolute terms. The ratio survives that.
    rtn_penalty = scores["round-to-nearest"] / scores["original"] - 1
    tuned_penalty = scores["mround"] / scores["original"] - 1
    print(f"\n  round-to-nearest penalty {rtn_penalty:+.1%}")
    print(f"  mround penalty           {tuned_penalty:+.1%}")

    if rtn_penalty > 0:
        recovered = 1 - tuned_penalty / rtn_penalty
        print(f"  gap closed               {recovered:+.1%}")
        print(f"  cost                     {tuned_seconds:.0f}s of tuning")
        if not comparable:
            print("  ...at unequal size, so the figure above is not a like-for-like result.")

    if vocab and scores["mround"] > vocab:
        print(
            f"\n  WARNING: {scores['mround']:,.0f} is worse than guessing uniformly over\n"
            f"  {vocab:,} tokens. This model is destroyed rather than degraded, and the\n"
            f"  figure above no longer measures quality."
        )

    if tuned_penalty >= rtn_penalty:
        print(
            "\n  Learned rounding did not beat round-to-nearest here. Before "
            "concluding\n  anything about the method, check how many blocks "
            "reverted: a run where\n  most of them did has a recipe problem "
            "rather than a marginal result."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
