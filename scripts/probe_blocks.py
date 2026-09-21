# Copyright 2026 The MRound Authors
# SPDX-License-Identifier: Apache-2.0
"""Verify block discovery and activation capture on a real model.

Run on an Apple Silicon Mac with the model stack installed::

    python scripts/probe_blocks.py

Why this exists. The block loop rests on three claims that cannot be checked
away from Apple hardware, and every one of them would fail quietly rather than
loudly if it were wrong:

1. **A block can be swapped for a stand-in.** ``update_modules`` is documented to
   replace child modules, but whether a model built by ``mlx-lm`` actually calls
   the replacement, and whether the original comes back afterwards, is a fact
   about a running model rather than about the API.
2. **What a block receives is knowable.** MRound assumes the hidden state is the
   first positional argument and that everything else is passed through
   untouched. If some architecture hands its blocks a keyword-only hidden state,
   or something that is not an array, the capture would record the wrong thing.
3. **Propagation is faithful.** The block loop runs one block at a time and
   feeds each block's output to the next, reusing the mask and position
   information captured once at the entrance. That is only sound if block N+1
   really does receive block N's output and the same auxiliary arguments. A
   model that varies them per layer would produce activations that look
   plausible and are wrong.

The third is the one worth the wall clock. It is checked by capturing block one
independently, through a real forward pass, and comparing what it was handed
against what block zero produced. Those two should agree exactly. Anything else
means propagation is an approximation, and a block loop built on it would be
optimizing against activations no real forward pass ever produces.

Nothing here is a benchmark. It is a set of yes-or-no questions asked on the one
kind of machine that can answer them, and the exit status is the answer.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mround.exceptions import MRoundError  # noqa: E402

if TYPE_CHECKING:
    from types import ModuleType

RULE = "=" * 78
MODEL = "mlx-community/SmolLM2-135M-Instruct"

# Deliberately tiny. The questions here are structural, and a longer sequence
# answers them no better while making the run slower to iterate on.
SAMPLES = 4
SEQ_LEN = 256
BATCH_SIZE = 2

# Two forwards of the same unquantized model on the same tokens should agree to
# the last bit. Anything above zero here is nondeterminism in the kernels, which
# would matter to every measurement this project makes.
EXACT = 0.0


def describe(mx: ModuleType, value: Any, depth: int = 0) -> str:
    """A short, comparable description of one captured argument."""
    if isinstance(value, mx.array):
        return f"array {value.dtype} {tuple(value.shape)}"
    if value is None:
        return "None"
    if isinstance(value, list | tuple) and depth < 2:  # noqa: PLR2004
        inner = ", ".join(describe(mx, item, depth + 1) for item in value[:4])
        more = ", ..." if len(value) > 4 else ""  # noqa: PLR2004
        return f"{type(value).__name__}[{len(value)}]({inner}{more})"
    if isinstance(value, dict) and depth < 2:  # noqa: PLR2004
        inner = ", ".join(f"{k}={describe(mx, v, depth + 1)}" for k, v in list(value.items())[:4])
        return f"dict[{len(value)}]({inner})"
    if isinstance(value, bool | int | float | str):
        return repr(value)
    return type(value).__name__


def max_difference(mx: ModuleType, left: Any, right: Any) -> float:
    """Largest absolute elementwise difference, in float32."""
    delta = left.astype(mx.float32) - right.astype(mx.float32)
    return float(mx.max(mx.abs(delta)))


def _build_batches(mx: ModuleType, bundle: Any, corpus: str) -> tuple[list[Any], str]:
    """Calibration batches, falling back to random ids if the corpus is unreachable.

    The corpus builder is worth exercising in the same run, since each run costs
    a round trip to a machine that is not this one. It is not worth blocking the
    capture checks on, since those are about the model rather than the data.
    """
    from mround.exceptions import CalibrationError  # noqa: PLC0415
    from mround.pipeline.calibration import build_calibration_set  # noqa: PLC0415

    try:
        calibration = build_calibration_set(
            bundle.tokenizer,
            source=corpus,
            n_samples=SAMPLES,
            seq_len=SEQ_LEN,
            batch_size=BATCH_SIZE,
        )
    except CalibrationError as exc:
        print(f"  corpus unavailable, using random ids instead:\n    {exc}")
        vocab = int(bundle.config.get("vocab_size") or 32000)
        ids = mx.random.randint(0, vocab, (SAMPLES, SEQ_LEN))
        batches = [ids[start : start + BATCH_SIZE] for start in range(0, SAMPLES, BATCH_SIZE)]
        return batches, "random ids"

    print(f"  {calibration.describe()}")
    return calibration.batches, calibration.content_hash


def check_discovery(bundle: Any) -> list[Any]:
    """Locate the block stack and check it against the configuration."""
    from mround.pipeline.blocks import discover_blocks  # noqa: PLC0415

    expected = bundle.config.get("num_hidden_layers")
    blocks = discover_blocks(bundle.model, expected=expected)
    print(f"  {len(blocks)} blocks, configuration says {expected}")
    print(f"  first {blocks[0].name!r}, last {blocks[-1].name!r}")
    print(f"  block type {type(blocks[0].module).__name__}")
    return blocks


def check_capture(mx: ModuleType, bundle: Any, blocks: list[Any], batches: list[Any]) -> Any:
    """Capture block zero and report exactly what it was handed."""
    from mround.pipeline.blocks import capture_block_inputs  # noqa: PLC0415

    cache = capture_block_inputs(bundle.model, blocks[0], batches)
    print(f"  captured {len(cache)} batches, {cache.nbytes() / 1e6:.1f} MB of activations")
    print(f"  hidden state    {describe(mx, cache.inputs[0])}")
    print(f"  block output    {describe(mx, cache.outputs[0])}")
    for position, value in enumerate(cache.args[0], start=1):
        print(f"  positional {position}    {describe(mx, value)}")
    for name, value in cache.kwargs[0].items():
        print(f"  keyword {name:<8}{describe(mx, value)}")
    if not cache.args[0] and not cache.kwargs[0]:
        print("  nothing else: the block takes the hidden state and no other argument")
    return cache


def check_restored(mx: ModuleType, bundle: Any, blocks: list[Any], before: Any) -> bool:
    """The model must be exactly what it was before the stand-in was installed.

    The module tree is walked directly rather than through ``discover_blocks``.
    A leaked stand-in would make discovery itself fail, and a failure there would
    be reported as an architecture problem rather than as what it is.
    """
    present = dict(bundle.model.named_modules())
    leaked = [block.name for block in blocks if present.get(block.name) is not block.module]
    after = bundle.model(before[0])
    mx.eval(after)
    drift = max_difference(mx, before[1], after)

    print(f"  every block object restored   {not leaked}")
    print(f"  logits after capture drift by {drift:.3e}")
    if leaked:
        print(f"  WARNING: a stand-in is still installed at {leaked[:3]}")
    if drift > EXACT:
        print("  WARNING: a forward pass after capture no longer matches one before it.")
    return not leaked and drift <= EXACT


def check_flags(blocks: list[Any]) -> bool:
    """Report per-block settings that a model's own loop may branch on.

    ``mlx-lm``'s Llama loop picks between a sliding-window mask and a full one by
    reading ``layer.use_sliding`` before calling the layer. A flag that varies
    down the stack means the mask varies with it, and capturing the mask once at
    the entrance would hand most blocks the wrong one.
    """
    values: dict[str, list[Any]] = {}
    for block in blocks:
        for key, value in vars(block.module).items():
            if not key.startswith("_") and isinstance(value, bool | str):
                values.setdefault(key, []).append(value)

    present = {key: seen for key, seen in values.items() if len(seen) == len(blocks)}
    varying = {key: sorted(set(seen)) for key, seen in present.items() if len(set(seen)) > 1}
    uniform = {key: seen[0] for key, seen in present.items() if len(set(seen)) == 1}

    print(f"  same on every block   {uniform or 'nothing'}")
    print(f"  varies down the stack {varying or 'nothing'}")
    if varying:
        print(
            "\n  WARNING: the blocks are not interchangeable. If the model's own loop\n"
            "  branches on one of these, each block is handed different auxiliary\n"
            "  arguments and they cannot be captured once."
        )
    return not varying


def _auxiliary_matches(mx: ModuleType, left: Any, right: Any, label: str) -> bool:
    """Whether two captures were handed the same non-hidden-state arguments."""
    ok = True
    drift = 0.0
    for one, other in zip(left.args, right.args, strict=True):
        if len(one) != len(other) or [describe(mx, v) for v in one] != [
            describe(mx, v) for v in other
        ]:
            ok = False
            continue
        for a, b in zip(one, other, strict=True):
            if isinstance(a, mx.array) and isinstance(b, mx.array):
                drift = max(drift, max_difference(mx, a, b))
    for one_kw, other_kw in zip(left.kwargs, right.kwargs, strict=True):
        if set(one_kw) != set(other_kw):
            ok = False

    ok = ok and drift <= EXACT
    print(f"  auxiliary arguments, {label:<16} {'match' if ok else 'DIFFER'} ({drift:.3e})")
    return ok


def check_propagation(mx: ModuleType, bundle: Any, blocks: list[Any], batches: list[Any]) -> bool:
    """Block one's real input, against what block zero produced.

    The claim the whole block loop rests on. If it holds, one block-forward per
    block is enough and the mask can be captured once. If it does not, every
    block needs its own prefix pass and the loop is quadratic in depth.

    The auxiliary arguments are compared at both ends of the stack rather than
    only at block one. An architecture that alternates between two attention
    patterns would agree between blocks zero and one about as often as it
    disagreed, and checking one adjacent pair would decide it by coin flip.
    """
    from mround.pipeline.blocks import capture_block_inputs  # noqa: PLC0415

    first = capture_block_inputs(bundle.model, blocks[0], batches)
    second = capture_block_inputs(bundle.model, blocks[1], batches, with_outputs=False)
    final = capture_block_inputs(bundle.model, blocks[-1], batches, with_outputs=False)

    worst = max(
        max_difference(mx, produced, received)
        for produced, received in zip(first.outputs, second.inputs, strict=True)
    )
    print(f"  block 1 input against block 0 output   {worst:.3e}")

    adjacent = _auxiliary_matches(mx, first, second, "block 0 and 1")
    across = _auxiliary_matches(mx, first, final, f"block 0 and {len(blocks) - 1}")

    ok = worst <= EXACT and adjacent and across
    if not ok:
        print(
            "\n  WARNING: propagation is not exact. The block loop assumes block N+1\n"
            "  receives block N's output and the same auxiliary arguments. Tuning\n"
            "  against propagated activations would be optimizing for inputs no real\n"
            "  forward pass produces."
        )
    return ok


def check_determinism(mx: ModuleType, bundle: Any, blocks: list[Any], batches: list[Any]) -> bool:
    """Two captures of the same block on the same tokens must be identical."""
    from mround.pipeline.blocks import capture_block_inputs  # noqa: PLC0415

    once = capture_block_inputs(bundle.model, blocks[0], batches, with_outputs=False)
    twice = capture_block_inputs(bundle.model, blocks[0], batches, with_outputs=False)
    worst = max(max_difference(mx, a, b) for a, b in zip(once.inputs, twice.inputs, strict=True))
    print(f"  two captures of block 0 differ by      {worst:.3e}")
    return worst <= EXACT


def _print_guide() -> None:
    """What each outcome means for the block loop."""
    print()
    print(RULE)
    print(" How to read this")
    print(RULE)
    print(" everything passes                 the block loop can be built as designed:")
    print("                                   capture once at the entrance, propagate,")
    print("                                   one block-forward per block.")
    print(" settings vary down the stack      the blocks are not interchangeable. A")
    print("                                   model whose own loop branches on one of")
    print("                                   those flags hands each block a different")
    print("                                   mask, and one capture cannot serve them all.")
    print(" propagation not exact             the mask or position information varies")
    print("                                   per layer on this architecture. Each block")
    print("                                   needs its own auxiliary arguments, captured")
    print("                                   in one pass over the stack rather than")
    print("                                   inherited from block zero.")
    print(" capture reports something odd     the hidden state is not the first")
    print("                                   positional argument here, and the capture")
    print("                                   contract needs widening before it is used.")
    print(" model not restored                the stand-in leaked. Every measurement")
    print("                                   taken after a capture is suspect until")
    print("                                   that is fixed.")


def main() -> int:
    """Run every check and return whether all of them passed."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument(
        "--corpus",
        default="pile-10k",
        help="calibration corpus, or a path to a local text file",
    )
    args = parser.parse_args()

    try:
        import mlx.core as mx  # noqa: PLC0415

        from mround.pipeline.loader import load_model  # noqa: PLC0415
    except ImportError as exc:
        print(f"needs MLX and the model stack: {exc}")
        return 1

    print(RULE)
    print(f" {args.model}")
    print(RULE)

    try:
        bundle = load_model(args.model)
        print(f"\ncalibration ({SAMPLES} x {SEQ_LEN} tokens)")
        batches, provenance = _build_batches(mx, bundle, args.corpus)

        # Taken before anything is swapped, so the restore check has something
        # from an untouched model to compare against.
        reference = bundle.model(batches[0])
        mx.eval(reference)

        print("\ndiscovery")
        blocks = check_discovery(bundle)
        if len(blocks) < 2:  # noqa: PLR2004
            print("  only one block: the propagation check needs two")
            return 1

        print("\ncapture")
        cache = check_capture(mx, bundle, blocks, batches)

        print("\nrestoration")
        restored = check_restored(mx, bundle, blocks, (batches[0], reference))

        print("\nblock settings")
        interchangeable = check_flags(blocks)

        print("\ndeterminism")
        deterministic = check_determinism(mx, bundle, blocks, batches)

        print("\npropagation")
        faithful = check_propagation(mx, bundle, blocks, batches)
    except MRoundError as exc:
        print(f"\nfailed: {exc}")
        return 1

    print()
    print(RULE)
    print(f" corpus {provenance}, {len(cache)} batches captured")
    print(
        f" discovery yes  restoration {'yes' if restored else 'NO'}  "
        f"settings {'uniform' if interchangeable else 'VARY'}  "
        f"determinism {'yes' if deterministic else 'NO'}  "
        f"propagation {'yes' if faithful else 'NO'}"
    )
    print(RULE)
    _print_guide()
    return 0 if (restored and deterministic and faithful) else 1


if __name__ == "__main__":
    raise SystemExit(main())
