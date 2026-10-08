# Copyright 2026 The MRound Authors
# SPDX-License-Identifier: Apache-2.0
"""Go or no go: can the reference implementation tune on this accelerator?

Until 2026-09-24 every reference arm in the ledger had run on a CPU
(provenance rows 55 to 79); the accelerator arms since are rows 80 (retracted),
82 to 87 and 90 to 94.
The measurement roadmap's first item puts the reference on the accelerator of
the machine it runs on, the Intel Arc graphics of the Ubuntu box through
``xpu`` and, since the reference's own device manager offers it, the Apple GPU
through ``mps`` on the Mac. Before a week goes into environments and a five
hour run, this answers the hour long question: does the reference's tuning loop
execute on the requested device at all, without falling back to the CPU?

Three steps, each printed as it happens so a failure names its stage:

1. The device, as torch sees it: availability, name, memory, driver.
2. Arithmetic on it, against the CPU, so a device that exists but computes
   nothing sensible is caught before the model loads.
3. One block of one small model through the reference at a handful of steps,
   with the device read back from the reference after the run, the peak device
   allocation where the backend reports one, and every warning the reference
   logged, because the reference falls back to the CPU by *warning* rather than
   by error. The same block runs on the CPU afterwards, so the log carries a
   first speed ratio; it is a probe figure, not a measurement, and it enters no
   ledger.

The verdict is GO when the device is available, the reference reports having
tuned on it after the run, and no warning mentions a fallback. Anything else is
NO GO with the stage that failed. Nothing here touches the ledger; the log is
the artifact.

Float32 with autocast off is the default, because that is what every recorded
CPU arm ran (MEMORY.md, 2026-09-07): on an accelerator the reference would
otherwise cast the model to bfloat16 and tune under autocast, which is a second
difference from the CPU arms on top of the device. ``--amp`` runs that path too
so both are on record before the calibration run is designed.

Usage::

    python scripts/probe_reference_device.py --device xpu
    python scripts/probe_reference_device.py --device mps --no-compare-cpu

Runs in any environment that has torch, transformers and auto-round; it
imports nothing from this package so a throwaway environment can run it.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

DEFAULT_MODEL = "mlx-community/SmolLM2-135M-Instruct"
DEFAULT_BLOCK = r"^model\.layers\.0$"

# Warnings that mean the reference did not do what it was asked. The device
# manager's default on Apple Silicon and the OOM fallback both announce
# themselves this way rather than by failing.
FALLBACK_PATTERNS = (
    re.compile(r"using cpu", re.IGNORECASE),
    re.compile(r"MPS detected"),
    re.compile(r"fall(ing|s)? ?back", re.IGNORECASE),
    re.compile(r"out of memory", re.IGNORECASE),
)

# Float32 accumulation order differs across devices; this is far above any of
# that and far below a wrong answer.
ARITHMETIC_TOLERANCE = 1e-4

RULE = "=" * 78


class Verdict:
    """What the probe found, accumulated as it goes."""

    def __init__(self) -> None:
        self.problems: list[str] = []
        self.notes: list[str] = []

    def fail(self, problem: str) -> None:
        """Record a reason for NO GO."""
        self.problems.append(problem)
        print(f"  PROBLEM: {problem}")

    def note(self, text: str) -> None:
        """Record a fact worth having in the log."""
        self.notes.append(text)
        print(f"  {text}")


class CapturedLog(logging.Handler):
    """What the reference logs, kept for the verdict and for its own timer.

    The reference prints its block loop time as ``quantization tuning time``
    at INFO level, and rows 9 and 44 took their reference costs from it (row
    44's note); the comparison harness records its own wall clock around
    tuning and export instead (row 83: 5379.7 s against this timer's 5371.2 s).
    The probe's ratio uses the timer rather than a wall clock that includes
    loading a model and preprocessing a dataset.
    """

    TIMER = re.compile(r"quantization tuning time ([0-9.]+)")
    COMPILE = re.compile(r"`torch.compile` is (enabled|disabled.*)")

    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.warnings: list[str] = []
        self.tuning_seconds: float | None = None
        self.compile_state: str | None = None

    def emit(self, record: logging.LogRecord) -> None:
        """Keep warnings, and the timer when it goes by."""
        message = record.getMessage()
        if record.levelno >= logging.WARNING:
            self.warnings.append(message)
        matched = self.TIMER.search(message)
        if matched:
            self.tuning_seconds = float(matched.group(1))
        compiled = self.COMPILE.search(message)
        if compiled:
            self.compile_state = compiled.group(1)


def _describe_xpu(verdict: Verdict) -> bool:
    import torch  # noqa: PLC0415

    xpu = getattr(torch, "xpu", None)
    available = bool(xpu is not None and xpu.is_available())
    verdict.note(f"torch.xpu.is_available() = {available}")
    if xpu is None or not available:
        verdict.fail(
            "torch cannot see an XPU. Either this torch is not the +xpu build, or the "
            "Level Zero runtime for the graphics is not installed, or this user is not "
            "in the render group. The shell script that called this prints the checks."
        )
        return False
    count = xpu.device_count()
    verdict.note(f"{count} XPU device(s)")
    for index in range(count):
        properties = xpu.get_device_properties(index)
        verdict.note(f"xpu:{index} {xpu.get_device_name(index)}")
        for field in ("platform_name", "driver_version", "total_memory", "max_compute_units"):
            value = getattr(properties, field, None)
            if value is not None:
                verdict.note(f"    {field}: {value}")
    return True


def _describe_mps(verdict: Verdict) -> bool:
    import torch  # noqa: PLC0415

    built = torch.backends.mps.is_built()
    available = torch.backends.mps.is_available()
    verdict.note(f"torch.backends.mps.is_built() = {built}, is_available() = {available}")
    if not available:
        verdict.fail("torch cannot see the Apple GPU through MPS")
        return False
    recommended = getattr(torch.mps, "recommended_max_memory", None)
    if recommended is not None:
        verdict.note(f"recommended_max_memory: {recommended():,} bytes")
    return True


def _describe_cuda(verdict: Verdict) -> bool:
    import torch  # noqa: PLC0415

    available = bool(torch.cuda.is_available())
    verdict.note(f"torch.cuda.is_available() = {available}")
    if available:
        verdict.note(str(torch.cuda.get_device_name(0)))
    return available


def _describe_cpu(verdict: Verdict) -> bool:
    # Exercises the probe itself on a machine with no accelerator; the verdict
    # then says nothing about any device.
    verdict.note("cpu requested: this run checks the probe, not an accelerator")
    return True


DESCRIBE = {
    "xpu": _describe_xpu,
    "mps": _describe_mps,
    "cuda": _describe_cuda,
    "cpu": _describe_cpu,
}


def step_device(device: str, verdict: Verdict) -> bool:
    """Step 1: is the requested device there, and what is it."""
    import torch  # noqa: PLC0415

    print(RULE)
    print(f" step 1: {device} as torch {torch.__version__} sees it")
    print(RULE)
    kind = device.split(":", maxsplit=1)[0]
    accelerator = getattr(torch, "accelerator", None)
    if accelerator is not None:
        try:
            current = accelerator.current_accelerator() if accelerator.is_available() else None
        except (RuntimeError, AssertionError):
            current = None
        verdict.note(f"torch.accelerator reports {current if current is not None else 'none'}")
    describe = DESCRIBE.get(kind)
    if describe is None:
        verdict.fail(f"this probe knows {', '.join(DESCRIBE)}, not {kind}")
        return False
    return describe(verdict)


def step_arithmetic(device: str, verdict: Verdict) -> bool:
    """Step 2: a matrix product on the device agrees with an exact reference.

    Both the device and the CPU are measured against a float64 product, so the
    two float32 errors are on record separately. The first MPS run compared the
    device against the CPU alone and reported an error of exactly zero, which
    two different backends rarely produce and which said nothing about whether
    either was right.
    """
    import torch  # noqa: PLC0415

    print(RULE)
    print(f" step 2: float32 arithmetic on {device} and on the CPU against float64")
    print(RULE)
    generator = torch.Generator().manual_seed(0)
    left = torch.randn(1024, 1024, generator=generator)
    right = torch.randn(1024, 1024, generator=generator)
    exact = left.double() @ right.double()
    scale = float(exact.abs().max())
    cpu_product = left @ right
    cpu_error = float((cpu_product.double() - exact).abs().max()) / scale
    try:
        started = time.perf_counter()
        product = (left.to(device) @ right.to(device)).to("cpu")
        elapsed = time.perf_counter() - started
    except (RuntimeError, AssertionError) as exc:
        verdict.fail(f"a matrix product on {device} failed: {exc}")
        return False
    error = float((product.double() - exact).abs().max()) / scale
    verdict.note(
        f"1024x1024 float32 product against float64: {device} {error:.2e}, cpu {cpu_error:.2e}, "
        f"{elapsed * 1e3:.1f} ms on {device}"
    )
    if device != "cpu" and torch.equal(product, cpu_product):
        verdict.note(
            "the device's result is bit identical to the CPU's, which two backends rarely "
            "are; the reference run below is the test that counts"
        )
    if error > ARITHMETIC_TOLERANCE:
        verdict.fail(f"the device's arithmetic is off by {error:.2e} against float64")
        return False
    return True


def _peak_device_bytes(device: str) -> int | None:
    """The backend's own high water mark, where it has one."""
    import torch  # noqa: PLC0415

    kind = device.split(":", maxsplit=1)[0]
    if kind == "xpu":
        return int(torch.xpu.max_memory_allocated())
    if kind == "cuda":
        return int(torch.cuda.max_memory_allocated())
    if kind == "mps":
        # MPS exposes the current driver allocation, not a peak.
        return int(torch.mps.driver_allocated_memory())
    return None


def run_reference_block(
    device: str,
    *,
    model_id: str,
    block: str,
    iters: int,
    samples: int,
    seq_len: int,
    batch_size: int,
    amp: bool,
    compile_: bool | None,
    verdict: Verdict,
) -> tuple[float | None, str | None]:
    """One block through the reference; returns (block loop seconds, device it reports)."""
    import torch  # noqa: PLC0415
    from auto_round import AutoRound  # noqa: PLC0415
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    captured = CapturedLog()
    # The reference's logger is named "autoround" and does not propagate.
    for name in ("", "auto_round", "autoround"):
        logging.getLogger(name).addHandler(captured)

    print(f"\n  loading {model_id} in float32")
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # torch.compile is left at the reference's own default unless asked,
    # because the compiled path is part of what a calibration run would use
    # and a backend whose compiler fails is a finding, not noise.
    settings: dict[str, Any] = {}
    if compile_ is not None:
        settings["enable_torch_compile"] = compile_
    rounder = AutoRound(
        model,
        tokenizer,
        bits=4,
        group_size=64,
        sym=True,
        iters=iters,
        nsamples=samples,
        seqlen=seq_len,
        batch_size=min(samples, batch_size),
        seed=42,
        device_map=device,
        amp=amp,
        to_quant_block_names=block,
        **settings,
    )
    if hasattr(rounder, "post_init"):
        rounder.post_init()
    before = str(getattr(rounder, "device", None))
    context = getattr(rounder, "model_context", None)
    verdict.note(f"reference resolved device_map={device!r} to {before}")
    if context is not None:
        verdict.note(
            f"reference amp={getattr(context, 'amp', 'unrecorded')}, "
            f"amp_dtype={getattr(context, 'amp_dtype', 'unrecorded')}"
        )

    with tempfile.TemporaryDirectory(prefix="mround-probe-") as scratch:
        started = time.perf_counter()
        try:
            rounder.quantize_and_save(output_dir=scratch, format="fake")
        except (RuntimeError, AssertionError, NotImplementedError) as exc:
            verdict.fail(f"the reference failed on {device}: {type(exc).__name__}: {exc}")
            return None, before
        elapsed = time.perf_counter() - started
        written = sorted(p.name for p in Path(scratch).rglob("*.safetensors"))
    after = str(getattr(rounder, "device", None))
    verdict.note(
        f"one block, {iters} steps over {samples}x{seq_len} tokens: {elapsed:.1f} s wall clock "
        "including input caching and export"
    )
    if captured.tuning_seconds is not None:
        verdict.note(
            f"the reference's own block loop timer: {captured.tuning_seconds:.2f} s, "
            f"{captured.tuning_seconds / iters:.3f} s per step"
        )
        elapsed = captured.tuning_seconds
    verdict.note(f"torch.compile {captured.compile_state or 'state not logged'}")
    verdict.note(f"reference reports its device as {after} after the run")
    verdict.note(f"wrote {len(written)} safetensors file(s) in fake format")
    peak = _peak_device_bytes(device) if device != "cpu" else None
    if peak is not None:
        verdict.note(f"backend memory figure for {device}: {peak:,} bytes")

    verdict.note(f"{len(captured.warnings)} warning(s) logged by the reference")
    for text in captured.warnings:
        print(f"    warning: {text}")
    for text in captured.warnings:
        if any(pattern.search(text) for pattern in FALLBACK_PATTERNS) and device != "cpu":
            verdict.fail(f"a warning reads like a fallback: {text}")
    for name in ("", "auto_round", "autoround"):
        logging.getLogger(name).removeHandler(captured)
    return elapsed, after


def main(argv: list[str] | None = None) -> int:
    """Run the three steps and print GO or NO GO."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--device", required=True, help="xpu, xpu:0, mps or cuda; cpu exercises the probe itself"
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--block", default=DEFAULT_BLOCK, help="regex for the block(s) to tune")
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="the recorded head to head rows used 8; capped at --samples",
    )
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--amp", action="store_true", help="also run the bfloat16 autocast path")
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="pass enable_torch_compile=False instead of the reference's default",
    )
    parser.add_argument(
        "--no-compare-cpu",
        action="store_true",
        help="skip the same block on the CPU (it is what gives the log a speed ratio)",
    )
    args = parser.parse_args(argv)

    verdict = Verdict()
    print(f"reference device probe: {args.device}, {args.model}, block {args.block!r}")
    try:
        import auto_round  # noqa: PLC0415

        print(f"auto-round {auto_round.__version__}")
    except ImportError:
        verdict.fail("auto-round is not installed in this environment")
        print("\nNO GO")
        return 1

    if not step_device(args.device, verdict) or not step_arithmetic(args.device, verdict):
        print("\nNO GO: " + "; ".join(verdict.problems))
        return 1

    print(RULE)
    print(f" step 3: one block of the reference on {args.device}, float32, autocast off")
    print(RULE)
    common: dict[str, Any] = {
        "model_id": args.model,
        "block": args.block,
        "iters": args.iters,
        "samples": args.samples,
        "seq_len": args.seq_len,
        "batch_size": args.batch_size,
        "compile_": False if args.no_compile else None,
        "verdict": verdict,
    }
    seconds, reported = run_reference_block(args.device, amp=False, **common)
    requested_kind = args.device.split(":", maxsplit=1)[0]
    if reported is not None and not reported.startswith(requested_kind):
        verdict.fail(f"asked for {args.device}, the reference reports {reported}")

    if args.amp and seconds is not None:
        print(RULE)
        print(f" step 3b: the same block on {args.device} with autocast on")
        print(RULE)
        run_reference_block(args.device, amp=True, **common)

    if not args.no_compare_cpu and seconds is not None and requested_kind != "cpu":
        print(RULE)
        print(" step 4: the same block on the CPU, for a first ratio (a probe figure, not data)")
        print(RULE)
        cpu_seconds, _ = run_reference_block("cpu", amp=False, **common)
        if cpu_seconds:
            verdict.note(
                f"cpu / {args.device} block loop time on this block: {cpu_seconds / seconds:.2f} "
                f"(one block at {args.iters} steps over {args.samples}x{args.seq_len} tokens, "
                "a probe figure and not a ledger measurement)"
            )

    print(RULE)
    if verdict.problems:
        print("NO GO: " + "; ".join(verdict.problems))
        return 1
    print(f"GO: the reference tuned one block on {args.device} and still reports {reported}")
    print("  This is a probe. No number above enters the ledger.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
