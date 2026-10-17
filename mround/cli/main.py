# Copyright 2026 The MRound Authors
# SPDX-License-Identifier: Apache-2.0
"""Command-line entry point.

Four subcommands, all implemented. ``version`` and ``doctor`` answer the
question a console script has to answer before anything else, whether this
machine can run the thing at all. ``quantize`` and ``eval`` are thin over
:mod:`mround.api`, :mod:`mround.eval.perplexity` and
:mod:`mround.eval.zeroshot`: argument translation,
progress that is already printed by the layer underneath, and errors turned
into a message and an exit status rather than a traceback.

**Thin is the design, not a shortcut.** Everything this file could decide for
itself is decided in the API instead, because the API is what the measurements
run through and a command line that resolved its own defaults would be a second
set of defaults to keep honest. The one thing decided here is the exit status,
which is the only thing a shell can read.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import TYPE_CHECKING, Any

from mround import __version__
from mround.exceptions import PlatformError

if TYPE_CHECKING:
    from mround.schemes import Symmetry

__all__ = ["build_parser", "main"]

# Column width for the human readable report. Wide enough for the longest label
# below, which is "reference layer" at 15 characters, with three to spare.
_LABEL = 18


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser.

    Separate from :func:`main` so that tests can exercise argument handling
    without running anything.

    Returns:
        The parser, with every subcommand registered.
    """
    parser = argparse.ArgumentParser(
        prog="mround",
        description="Weight-only quantization for LLMs on Apple Silicon.",
    )
    parser.add_argument("--version", action="version", version=f"mround {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    sub.add_parser("version", help="Print the version and exit.")

    doctor = sub.add_parser(
        "doctor",
        help="Report whether this machine can run MRound, and what it found.",
    )
    doctor.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable output.",
    )

    quantize = sub.add_parser("quantize", help="Quantize a model.")
    quantize.add_argument("model", help="Hugging Face repository id or local path.")
    quantize.add_argument(
        "-o",
        "--output-dir",
        default=None,
        help="Where to write. Required unless --dry-run.",
    )
    quantize.add_argument(
        "-b",
        "--bits",
        type=int,
        default=4,
        choices=[2, 3, 4, 5, 6, 7, 8],
        help="Uniform bit width (default: 4).",
    )
    quantize.add_argument(
        "-g",
        "--group-size",
        type=int,
        default=64,
        help=(
            "Weights per scale; -1 for per-channel (default: 64). A layer can "
            "only be grouped when its input dimension is a multiple of this, "
            "and 64 divides strictly more models than 128 does."
        ),
    )
    quantize.add_argument(
        "--asym",
        action="store_true",
        help="Use an asymmetric code range with a learned zero point.",
    )
    quantize.add_argument(
        "--average-bits",
        type=float,
        default=None,
        help="Target average width. Engages mixed-precision allocation.",
    )
    quantize.add_argument(
        "--format",
        default="mlx",
        choices=["mlx", "gguf"],
        help="Export format (default: mlx).",
    )
    quantize.add_argument(
        "--options",
        default="2,3,4",
        help="Candidate widths for --average-bits, comma separated (default: 2,3,4).",
    )
    quantize.add_argument(
        "--remainder-bits",
        type=int,
        default=None,
        help="Width for the tensors outside the block stack (default: nearest the average).",
    )
    quantize.add_argument(
        "--iters",
        type=int,
        default=None,
        help="Optimization steps per block (default: the standard recipe's 200).",
    )
    quantize.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Calibration sequences (default: the standard recipe's 128).",
    )
    quantize.add_argument(
        "--seq-len",
        type=int,
        default=None,
        help="Tokens per calibration sequence (default: 2048).",
    )
    quantize.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Calibration sequences per step (default: 8).",
    )
    quantize.add_argument(
        "--micro-batches",
        type=int,
        default=None,
        help=(
            "Run each step's batch in this many pieces, summing their gradients: "
            "the step's activation memory falls by about this factor and the "
            "step is still the whole batch's, but the run is slower and its "
            "checkpoint is another draw, so use it only where a step does not "
            "fit (default: 1)."
        ),
    )
    quantize.add_argument(
        "--calibration",
        default="auto",
        help="Calibration corpus identifier (default: auto, which is pile-10k).",
    )
    quantize.add_argument("--seed", type=int, default=None, help="Calibration seed.")
    quantize.add_argument(
        "--allow-dense",
        action="store_true",
        help=(
            "Proceed even when the group size can only reach a small part of the "
            "model. Without this, a run that would leave most layers at full "
            "precision is refused before it starts rather than after an hour."
        ),
    )
    quantize.add_argument(
        "--memory-budget",
        type=float,
        default=None,
        metavar="GB",
        help=(
            "Memory the run may be predicted to reach, in GB of 10^9 bytes, the unit "
            "run records print (default: 85 percent of this Mac's memory). The peak "
            "is predicted from the model's configuration before anything is loaded, "
            "and a run predicted past the budget is refused with the --micro-batches "
            "that would fit."
        ),
    )
    quantize.add_argument(
        "--skip-memory-check",
        action="store_true",
        help="Predict and record the peak, but run even when it is past the budget.",
    )
    quantize.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Predict the run's peak memory against the budget and stop, reading the "
            "model's configuration and nothing else. Exits 0 when it fits, 1 when "
            "the run would be refused."
        ),
    )

    evaluate = sub.add_parser("eval", help="Evaluate a checkpoint's quality.")
    evaluate.add_argument("model", help="Path to a checkpoint.")
    evaluate.add_argument(
        "--perplexity",
        action="store_true",
        help="Measure perplexity. The default when nothing else is asked for.",
    )
    evaluate.add_argument(
        "--zeroshot",
        action="store_true",
        help=(
            "Run the zero-shot task suite: the reference implementation's own default "
            "tasks less wikitext, through lm_eval's task definitions. Needs the eval "
            "extra (pip install -e '.[eval]')."
        ),
    )
    evaluate.add_argument(
        "--tasks",
        default=None,
        help="Comma separated lm_eval task names for --zeroshot (default: the suite).",
    )
    evaluate.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Score only the first N documents of each task, for a quick check. The "
            "default scores the full sets, which a published number requires."
        ),
    )
    evaluate.add_argument(
        "--dataset",
        default="wikitext2",
        help="Evaluation set, or a path to a text file (default: wikitext2).",
    )
    evaluate.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Stop after this many tokens. The default scores the whole set.",
    )

    return parser


def _version_of(distribution: str) -> str | None:
    """Installed version of a distribution, or ``None`` when it is absent."""
    from importlib import metadata  # noqa: PLC0415

    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def _doctor_report() -> dict[str, Any]:
    """What this machine can and cannot run, as plain data.

    The environment facts come from :func:`mround.eval.results.capture_environment`,
    the same function every ledger row records itself with, so a doctor report
    and a measurement taken minutes later cannot disagree about what was
    installed. ``packages=False`` keeps it from writing a manifest: doctor
    reports on an environment, it does not record one.

    Every check is attempted and reported rather than short-circuited at the
    first failure, because the useful output on a machine that cannot run
    MRound is the whole list, not the first line of it.

    Returns:
        A mapping with stable keys, which is what ``--json`` prints.
    """
    from mround.eval.results import capture_environment  # noqa: PLC0415

    environment = capture_environment(packages=False)
    packages = {
        "mlx": environment.get("mlx"),
        "mlx-lm": environment.get("mlx_lm"),
        "numpy": _version_of("numpy"),
        "transformers": _version_of("transformers"),
        "safetensors": _version_of("safetensors"),
        "datasets": _version_of("datasets"),
    }

    checks: list[dict[str, Any]] = []

    def record(name: str, ok: bool, detail: str) -> bool:
        checks.append({"name": name, "ok": ok, "detail": detail})
        return ok

    # The verdict comes from the function the loader itself calls, not from a
    # copy of its rule here, so the two cannot drift apart. Its message already
    # explains the Rosetta case, which is the one that bites.
    from mround.pipeline.loader import require_apple_silicon  # noqa: PLC0415

    try:
        require_apple_silicon()
    except PlatformError as exc:
        platform_ok = record("apple silicon", False, str(exc))
    else:
        platform_ok = record(
            "apple silicon", True, f"{environment.get('system')} on {environment.get('machine')}"
        )

    device: dict[str, Any] = {
        "default": None,
        "peak_memory_reported": False,
        "memory_bytes": None,
        "memory_budget_bytes": None,
    }
    try:
        import mlx.core as mx  # noqa: PLC0415

        from mround.pipeline.device import (  # noqa: PLC0415
            device_memory,
            peak_memory,
            reset_peak_memory,
        )
    except ImportError as exc:
        mlx_ok = record("mlx", False, f"not importable: {exc}")
    else:
        from mround.planner.memory import default_budget  # noqa: PLC0415

        device["default"] = str(mx.default_device())
        # A megabyte, allocated and evaluated, so that the peak is something
        # this build has actually been asked to report rather than a zero that
        # might mean either "nothing allocated yet" or "no accessor".
        reset_peak_memory()
        mx.eval(mx.zeros((512, 512), dtype=mx.float32))
        device["peak_memory_reported"] = peak_memory() > 0
        # What the check before a run holds a prediction to, so that the
        # budget a refusal names can be read here before any run is tried.
        memory = device_memory()
        device["memory_bytes"] = memory or None
        device["memory_budget_bytes"] = default_budget(memory)
        mlx_ok = record("mlx", True, f"{packages['mlx']}, default device {device['default']}")

    # Checked by metadata rather than by importing it: mlx-lm pulls in
    # transformers, which costs seconds, and the loader imports it at call time
    # anyway. Its presence is what doctor can honestly report.
    loader_ok = record(
        "mlx-lm",
        packages["mlx-lm"] is not None,
        f"{packages['mlx-lm']}"
        if packages["mlx-lm"]
        else "missing: model loading needs it, install with pip install -e '.[models]'",
    )
    numpy_ok = record(
        "numpy",
        packages["numpy"] is not None,
        f"{packages['numpy']}" if packages["numpy"] else "missing, and everything needs it",
    )
    corpus_ok = record(
        "datasets",
        packages["datasets"] is not None,
        f"{packages['datasets']}"
        if packages["datasets"]
        else (
            "missing: the named calibration and evaluation corpora need it, "
            "though a local text file does not"
        ),
    )

    return {
        "mround": environment.get("mround") or __version__,
        "python": environment.get("python"),
        "system": environment.get("system"),
        "machine": environment.get("machine"),
        "processor": environment.get("processor"),
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "packages": packages,
        "device": device,
        "checks": checks,
        "ready": {
            "quantize": platform_ok and mlx_ok and loader_ok and numpy_ok and corpus_ok,
            # The NumPy reference layer is the whole quantization mathematics
            # and runs anywhere, which is worth saying on a machine that has
            # just been told it cannot run the rest.
            "reference": numpy_ok,
        },
    }


def _render_doctor(report: dict[str, Any]) -> str:
    """The human readable form of :func:`_doctor_report`."""
    lines = ["", "mround doctor", ""]

    def row(label: str, value: Any) -> None:
        lines.append(f"  {label:<{_LABEL}}{'not installed' if value is None else value}")

    row("mround", report["mround"])
    row("python", report["python"])
    row("system", report["system"])
    row("machine", f"{report['machine']} (processor {report['processor']})")
    row("conda env", report["conda_env"] or "none")
    lines.append("")
    for name, version in report["packages"].items():
        row(name, version)
    lines.append("")
    row("default device", report["device"]["default"])
    row("peak memory", "reported" if report["device"]["peak_memory_reported"] else "not reported")
    memory = report["device"].get("memory_bytes")
    budget = report["device"].get("memory_budget_bytes")
    row("device memory", f"{memory / 1e9:.2f} GB" if memory else "not reported")
    row(
        "memory budget",
        f"{budget / 1e9:.2f} GB, what a run may be predicted to reach" if budget else "none",
    )

    lines.extend(["", "  checks"])
    for check in report["checks"]:
        mark = "ok  " if check["ok"] else "FAIL"
        lines.append(f"    {mark}  {check['name']:<{_LABEL - 2}}{check['detail']}")

    ready = report["ready"]
    lines.append("")
    row("quantization", "ready" if ready["quantize"] else "not ready")
    row("reference layer", "ready" if ready["reference"] else "not ready")
    if not ready["quantize"]:
        lines.append("")
        lines.append("  Every failing check above says what is missing and how to get it.")
    lines.append("")
    return "\n".join(lines)


def _doctor(as_json: bool) -> int:
    """Run the doctor command.

    Returns:
        Zero when this machine can quantize, one when it cannot, so that a
        script can gate on it without parsing anything.
    """
    report = _doctor_report()
    print(json.dumps(report, indent=2, sort_keys=True) if as_json else _render_doctor(report))
    return 0 if report["ready"]["quantize"] else 1


def _widths(options: str) -> tuple[int, ...]:
    """Parse ``--options`` into candidate widths.

    Raises:
        ValueError: On anything that is not a comma separated list of integers,
            so the failure arrives as a message rather than as a traceback from
            somewhere inside the planner an hour later.
    """
    try:
        return tuple(int(piece) for piece in options.split(",") if piece.strip())
    except ValueError as exc:
        msg = f"--options must be comma separated integers, not {options!r}"
        raise ValueError(msg) from exc


def _tuning(args: argparse.Namespace) -> Any:
    """Build the tuning recipe, leaving anything unset at the library default.

    Unset rather than restated: the defaults live in ``TuningConfig`` and are
    the standard recipe of every head to head row but row 48, so a flag nobody
    passed must not quietly become a second opinion about what the recipe is.
    """
    from mround.schemes import TuningConfig  # noqa: PLC0415

    standard = TuningConfig()
    return TuningConfig(
        iters=standard.iters if args.iters is None else args.iters,
        n_samples=standard.n_samples if args.samples is None else args.samples,
        seq_len=standard.seq_len if args.seq_len is None else args.seq_len,
        batch_size=standard.batch_size if args.batch_size is None else args.batch_size,
        micro_batches=standard.micro_batches if args.micro_batches is None else args.micro_batches,
    )


def _describe_request(args: argparse.Namespace, candidates: tuple[int, ...]) -> str:
    """What this command is about to do, printed before it does it.

    A quantization run is half an hour and its first output arrives a minute
    in, after the first block. Without this the only confirmation that the
    right thing was typed is the shape of the numbers that start appearing
    later, which is too late to retype anything.

    The scale initialization is included because it is the one setting nobody
    passes and it decides whether a low-width run works at all (D-030, D-046):
    somebody quantizing at 2 bits should be able to see that the search is on.
    """
    from mround.api import default_scale_init  # noqa: PLC0415
    from mround.schemes import ScaleInit, Symmetry, TuningConfig  # noqa: PLC0415

    mixed = args.average_bits is not None
    # The same widths ``api.quantize`` resolves against: the base width, every
    # candidate, and the remainder width when one was asked for, since the
    # remainder tensors take that width and one initialization covers them all.
    widths = {args.bits}
    if mixed:
        widths.update(candidates)
        if args.remainder_bits is not None:
            widths.add(int(args.remainder_bits))
    narrowest = min(widths)
    width = (
        f"an average of {args.average_bits:g} bits over {list(candidates)}"
        if mixed
        else f"{args.bits} bits"
    )
    symmetry = Symmetry.ASYMMETRIC if args.asym else Symmetry.SYMMETRIC
    start = default_scale_init(narrowest, symmetry)
    scale = "searched per group" if start is ScaleInit.SEARCHED else "from the observed range"
    standard = TuningConfig()
    steps = standard.iters if args.iters is None else args.iters
    samples = standard.n_samples if args.samples is None else args.samples
    sequence = standard.seq_len if args.seq_len is None else args.seq_len
    batch = standard.batch_size if args.batch_size is None else args.batch_size
    pieces = standard.micro_batches if args.micro_batches is None else args.micro_batches
    split = f", each run in {pieces} pieces" if pieces > 1 else ""
    group = "per channel" if args.group_size == -1 else f"group size {args.group_size}"
    lines = [
        "",
        f"  {'model':<{_LABEL}}{args.model}",
        f"  {'scheme':<{_LABEL}}{width}, {group}, {symmetry.value}, scale {scale}",
        f"  {'calibration':<{_LABEL}}{args.calibration}, {samples} sequences of "
        f"{sequence} tokens, batches of {batch}{split}",
        f"  {'tuning':<{_LABEL}}{steps} steps per block",
        f"  {'writing to':<{_LABEL}}{args.output_dir}"
        if args.output_dir is not None
        else f"  {'writing to':<{_LABEL}}nothing, a dry run",
        "",
    ]
    return "\n".join(lines)


def _quantize(args: argparse.Namespace) -> int:
    """Run a quantization and report where it went.

    Returns:
        Zero on success, one on any error MRound raises deliberately. A
        traceback would be the wrong output here: every one of these errors is
        something the person can act on, and they are all worded to say what.
    """
    from mround import api  # noqa: PLC0415
    from mround.exceptions import MRoundError  # noqa: PLC0415
    from mround.schemes import Symmetry  # noqa: PLC0415

    if args.memory_budget is not None and not args.memory_budget > 0:
        print(
            f"mround: --memory-budget is a positive number of GB, not {args.memory_budget}",
            file=sys.stderr,
        )
        return 1
    budget = None if args.memory_budget is None else round(args.memory_budget * 1e9)
    symmetry = Symmetry.ASYMMETRIC if args.asym else Symmetry.SYMMETRIC
    try:
        candidates = _widths(args.options)
        print(_describe_request(args, candidates), flush=True)
        if args.dry_run:
            return _dry_run(args, candidates, symmetry, budget)
        result = api.quantize(
            args.model,
            output_dir=args.output_dir,
            bits=args.bits,
            group_size=args.group_size,
            symmetry=symmetry,
            average_bits=args.average_bits,
            candidate_bits=candidates,
            remainder_bits=args.remainder_bits,
            tuning=_tuning(args),
            calibration=args.calibration,
            export_format=args.format,
            min_block_coverage=0.0 if args.allow_dense else api.MIN_BLOCK_COVERAGE,
            seed=args.seed,
            memory_budget=budget,
            memory_check=not args.skip_memory_check,
        )
    except (MRoundError, ValueError) as exc:
        print(f"mround: {exc}", file=sys.stderr)
        return 1

    print("")
    print(f"  {result.summary.describe()}")
    print(f"  {result.describe()}")
    for path, reason in result.skipped:
        print(f"  left dense  {path}: {reason}")
    print(f"  wrote {result.output_dir}")
    print(f"  settings in {result.config_path.name}, which reproduces this run")
    return 0


def _dry_run(
    args: argparse.Namespace,
    candidates: tuple[int, ...],
    symmetry: Symmetry,
    budget: int | None,
) -> int:
    """Predict the run's peak memory against the budget, and stop.

    Returns:
        Zero when the run fits the budget, one when it would be refused, so
        that a script can ask before it commits a machine for hours.
    """
    from mround import api  # noqa: PLC0415

    check = api.predict_memory(
        args.model,
        bits=args.bits,
        group_size=args.group_size,
        symmetry=symmetry,
        average_bits=args.average_bits,
        candidate_bits=candidates,
        remainder_bits=args.remainder_bits,
        tuning=_tuning(args),
        memory_budget=budget,
    )
    print(f"  {check.describe()}")
    stages = check.prediction.stages
    for stage, reached in sorted(stages.items(), key=lambda item: -item[1]):
        print(f"    {stage:<{_LABEL - 2}}{reached / 1e9:.2f} GB")
    if check.fits:
        return 0
    print(f"mround: {check.explain()}", file=sys.stderr)
    return 1


def _evaluate(args: argparse.Namespace) -> int:
    """Score a checkpoint and print the measurement.

    Returns:
        Zero when a number was produced, one otherwise.
    """
    from mround.exceptions import MRoundError  # noqa: PLC0415

    if args.limit is not None and args.limit < 1:
        print(f"mround: --limit must be at least 1; got {args.limit}", file=sys.stderr)
        return 1
    tasks = None if args.tasks is None else [t.strip() for t in args.tasks.split(",") if t.strip()]
    if tasks is not None and not tasks:
        print("mround: --tasks names no task", file=sys.stderr)
        return 1
    # Perplexity is the default when nothing else is asked for, and runs
    # beside the task suite when both are.
    perplexity = args.perplexity or not args.zeroshot

    from mround.eval.perplexity import evaluate_perplexity  # noqa: PLC0415
    from mround.eval.zeroshot import (  # noqa: PLC0415
        DEFAULT_TASKS,
        context_window,
        evaluate_zeroshot,
    )
    from mround.pipeline.loader import load_model  # noqa: PLC0415

    try:
        # allow_quantized because scoring a quantized checkpoint is the entire
        # point here, while the loader refuses one by default so that nothing
        # quantizes a quantized model by accident.
        bundle = load_model(args.model, allow_quantized=True)
        if perplexity:
            measured = evaluate_perplexity(
                bundle.model,
                bundle.tokenizer,
                dataset=args.dataset,
                max_tokens=args.max_tokens,
            )
            print(f"  {measured.describe()}", flush=True)
        if args.zeroshot:
            summary, _ = evaluate_zeroshot(
                bundle.model,
                bundle.tokenizer,
                tasks=DEFAULT_TASKS if tasks is None else tasks,
                limit=args.limit,
                max_length=context_window(bundle.config),
            )
            for line in summary.describe().splitlines():
                print(f"  {line}")
    except MRoundError as exc:
        print(f"mround: {exc}", file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the command line.

    Args:
        argv: Arguments, defaulting to :data:`sys.argv`.

    Returns:
        A process exit status.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 1

    if args.command == "version":
        print(f"mround {__version__}")
        return 0

    if args.command == "doctor":
        return _doctor(args.json)

    if args.command == "quantize":
        if args.output_dir is None and not args.dry_run:
            parser.error("quantize needs -o/--output-dir, unless it is a --dry-run")
        return _quantize(args)

    return _evaluate(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
