# Copyright 2026 The MRound Authors
# SPDX-License-Identifier: Apache-2.0
"""Quantize a real model to 4-bit round-to-nearest, then measure what it cost.

Run on an Apple Silicon Mac with the model stack installed::

    pip install -e '.[models]'
    python examples/round_to_nearest_model.py

This is the baseline, not the method. It applies MRound's scale computation and
rounding rule with nothing learned, which is exactly round-to-nearest, and it is
the number learned rounding will have to beat. Measuring it now, through the
same loader and the same export path the tuned version will use, is what makes a
later improvement attributable to the learning rather than to anything else that
changed along the way.

It downloads a model on first run. The default is small on purpose: the point is
to exercise the whole chain quickly, not to produce a publishable figure.
"""

from __future__ import annotations

import argparse
import time

from mround.exceptions import MRoundError

# Verified to exist, to be unquantized bfloat16, and to have every layer
# dimension divisible by 64. Its word embeddings are tied, so quantizing the
# embedding also quantizes the output head, which is the case D-019 is about.
MODEL = "mlx-community/SmolLM2-135M-Instruct"

# How far the predicted checkpoint size may sit from the written one before
# the two are treated as disagreeing rather than rounding.
SIZE_DRIFT_TOLERANCE = 0.02


def _report_storage(result: object, bits: int, group_size: int) -> None:
    """Check the size accounting against the file, and the width against the ask.

    Two independent ways of being wrong about what was produced. The first
    compares arithmetic against a file written by different code. The second
    catches tensors that declined to be grouped and stayed dense, which makes
    "this model is 4-bit" false in a way nobody can see.
    """
    written = sum(f.stat().st_size for f in result.output_dir.glob("*.safetensors"))  # type: ignore[attr-defined]
    print(
        f"  weights on disk {written / 1e6:.1f} MB against "
        f"{result.stored_bytes / 1e6:.1f} MB "  # type: ignore[attr-defined]
        f"predicted, from {result.original_bytes / 1e6:.1f} MB"  # type: ignore[attr-defined]
    )
    drift = abs(written - result.stored_bytes) / max(1, result.stored_bytes)  # type: ignore[attr-defined]
    if drift > SIZE_DRIFT_TOLERANCE:
        print(
            f"  WARNING: those differ by {drift:.1%}. The size accounting and the "
            f"exporter disagree\n  about what is being stored, and the reported "
            f"bits per weight is the suspect one."
        )
    for path, reason in result.skipped:  # type: ignore[attr-defined]
        print(f"    left dense: {path}: {reason}")

    # A fully quantized model costs the requested width plus one pair of 16-bit
    # scale parameters per group. Much above that means tensors stayed dense.
    expected = bits + 2 * 16 / group_size
    if result.bits_per_weight > expected * 1.1:  # type: ignore[attr-defined]
        print(
            f"\n  WARNING: {result.bits_per_weight:.2f} bits per weight against "  # type: ignore[attr-defined]
            f"{expected:.2f} expected\n  for {bits} bits at group size "
            f"{group_size}. Tensors stayed dense. A smaller\n  --group-size "
            f"usually fixes it: every dimension has to be a multiple of it,\n"
            f"  and 64 divides more shapes than 128 does."
        )


def main() -> int:
    """Quantize, then score both the original and the result."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL, help=f"model to quantize (default {MODEL})")
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "where to write the checkpoint. Defaults to a name derived from the "
            "settings, so successive runs do not overwrite each other or end up "
            "labelled with a width they do not have."
        ),
    )
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument(
        "--symmetry",
        choices=["sym", "asym"],
        default="sym",
        help=(
            "sym constrains the bias to -scale * 2**(bits-1); asym learns it per "
            "group. asym is strictly more expressive at the same width and costs "
            "the same to store, so the comparison is worth running (MEMORY.md Q-002)"
        ),
    )
    parser.add_argument(
        "--eval",
        default=None,
        help=(
            "text file or dataset name to score. Omitted skips evaluation, which "
            "is the fast path when you only want to see the checkpoint written."
        ),
    )
    parser.add_argument(
        "--eval-tokens",
        type=int,
        default=8192,
        help="stop scoring after this many tokens; a real number uses the whole set",
    )
    args = parser.parse_args()
    if args.output is None:
        args.output = f"./rtn-{args.bits}bit-g{args.group_size}-{args.symmetry}"

    # Imported here rather than at module scope so that --help works on a
    # machine without MLX. Everything below this line needs Apple Silicon;
    # reading the options does not.
    from mround import api  # noqa: PLC0415
    from mround.eval.perplexity import evaluate_perplexity  # noqa: PLC0415
    from mround.pipeline.loader import load_model  # noqa: PLC0415
    from mround.schemes import Symmetry  # noqa: PLC0415

    print(
        f"quantizing {args.model} to {args.bits} bits, "
        f"group size {args.group_size}, {args.symmetry}"
    )
    started = time.perf_counter()
    try:
        result = api.quantize_round_to_nearest(
            args.model,
            output_dir=args.output,
            bits=args.bits,
            group_size=args.group_size,
            symmetry=Symmetry(args.symmetry),
        )
    except MRoundError as exc:
        # One clear line. A wall of chained Hub tracebacks buries the sentence
        # that says what to do, and every one of these failures is actionable.
        print(f"\nfailed: {exc}")
        return 1
    elapsed = time.perf_counter() - started

    print(f"  {result.describe()}")
    print(f"  written to {result.output_dir} in {elapsed:.1f}s")
    _report_storage(result, args.bits, args.group_size)

    if args.eval is None:
        print("\nno evaluation set given, so quality is unmeasured. Pass --eval to score it.")
        return 0

    print(f"\nscoring both models on {args.eval}")
    scores = {}
    vocab = 0
    for label, source in (("original", args.model), ("quantized", args.output)):
        # allow_quantized on the second pass: the loader refuses these by
        # default so that nothing quantizes a checkpoint twice, and scoring one
        # is the exception that refusal exists to permit.
        bundle = load_model(source, allow_quantized=True)
        vocab = int(bundle.config.get("vocab_size") or 0)
        measured = evaluate_perplexity(
            bundle.model,
            bundle.tokenizer,
            dataset=args.eval,
            seq_len=2048,
            max_tokens=args.eval_tokens,
        )
        scores[label] = measured.perplexity
        print(f"  {label:>9}  {measured.describe()}")

    # The penalty, not the two raw numbers. Perplexity depends on how much text
    # was scored, so runs with different --eval-tokens are not comparable in
    # absolute terms and quoting them side by side credits or blames a change
    # for a difference in the evaluation set. The ratio survives that.
    penalty = scores["quantized"] / scores["original"] - 1
    print(f"\n  quantization penalty {penalty:+.1%}")

    # Perplexity above the vocabulary size is worse than guessing uniformly,
    # which means the model is confidently wrong rather than merely uninformed.
    # Past that point the number has stopped measuring quality, and comparing
    # two of them ranks two kinds of broken.
    if vocab and scores["quantized"] > vocab:
        print(
            f"\n  WARNING: {scores['quantized']:,.0f} is {scores['quantized'] / vocab:.0f} "
            f"times worse than guessing\n  uniformly over {vocab:,} tokens. This model is "
            f"destroyed, not degraded, and the\n  figure above no longer measures quality. "
            f"Comparing it with another run\n  in this state ranks two kinds of broken."
        )
    print(
        "  Compare that figure across runs, never the raw perplexities: they "
        "move with\n  --eval-tokens. Learned rounding has to close some of this "
        "to earn its runtime."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
