# Copyright 2026 The MRound Authors
# SPDX-License-Identifier: Apache-2.0
"""Compare MRound's round-to-nearest against MLX's own, on real weights.

Run on an Apple Silicon Mac with the model stack installed::

    python scripts/probe_quantizer_quality.py

Why this exists. The first real quantization put SmolLM2-135M's perplexity from
18.86 to 29.59 at 4 bits. That is a large gap and it has at least three possible
causes, which a perplexity number cannot tell apart:

1. Round-to-nearest is simply lossy at 4 bits on a 135M model, which is expected
   and is the headroom learned rounding exists to recover.
2. MRound's symmetric scheme constrains the bias to ``-scale * 2**(b-1)``, while
   MLX's affine quantizer leaves it free. The second is strictly more expressive
   at the same width, so some of the gap may be the scheme rather than the
   rounding.
3. Something in MRound's quantizer is wrong.

This separates them by measuring the reconstruction error of the weights
directly, layer by layer, three ways: MRound symmetric, MRound asymmetric, and
`mx.quantize` followed by `mx.dequantize`. MLX's quantizer is an independent
implementation of the same idea by people who are not us, which makes it the
strongest oracle available here without a second machine.

Reading it: if MRound symmetric sits far above MLX affine, cause 2 or 3. If
MRound asymmetric matches MLX affine closely, the quantizer is sound and the
difference is the scheme, which is Q-002. If both MRound columns sit far above
MLX, cause 3, and the perplexity gap is a symptom rather than the finding.

This measures weight reconstruction, not model quality. The two are related and
not the same: a large error in a layer whose outputs barely matter costs less
than a small one in a layer that feeds everything. Perplexity remains the
arbiter; this says where to look.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mround.schemes import QuantScheme, Symmetry  # noqa: E402

if TYPE_CHECKING:
    from collections.abc import Sequence
    from types import ModuleType

RULE = "=" * 78
MODEL = "mlx-community/SmolLM2-135M-Instruct"


def relative_error(mx: ModuleType, original: Any, rebuilt: Any) -> float:
    """Root mean square reconstruction error, normalized by the weight's own scale.

    Normalized so that layers of different magnitudes can be compared and
    averaged. An unnormalized error would be dominated by whichever layer
    happens to hold the largest numbers.
    """
    error = (original - rebuilt).astype(mx.float32)
    reference = original.astype(mx.float32)
    return float(mx.sqrt(mx.mean(error * error) / mx.mean(reference * reference)))


def mround_error(mx: ModuleType, quantizer: ModuleType, weight: Any, scheme: QuantScheme) -> float:
    """What MRound's round-to-nearest costs this weight matrix."""
    rebuilt = quantizer.fake_quantize(weight.astype(mx.float32), None, scheme)
    return relative_error(mx, weight, rebuilt)


def mlx_error(mx: ModuleType, weight: Any, bits: int, group_size: int) -> float:
    """What MLX's own affine quantizer costs the same matrix."""
    codes, scales, biases = mx.quantize(weight.astype(mx.float32), group_size, bits)
    rebuilt = mx.dequantize(codes, scales=scales, biases=biases, group_size=group_size, bits=bits)
    return relative_error(mx, weight, rebuilt)


class LayerError(NamedTuple):
    """One layer measured three ways, sized so the averages can be weighted."""

    size: int
    symmetric: float
    asymmetric: float
    mlx: float
    kind: str


def _weighted(rows: Sequence[LayerError]) -> tuple[int, float, float, float]:
    """Size-weighted means, so a 28M-parameter embedding outweighs a 3M projection."""
    size = sum(r.size for r in rows)
    return (
        size,
        sum(r.size * r.symmetric for r in rows) / size,
        sum(r.size * r.asymmetric for r in rows) / size,
        sum(r.size * r.mlx for r in rows) / size,
    )


def _print_by_layer(rows: Sequence[LayerError]) -> None:
    """Per-layer-kind breakdown, shown only when a single width was asked for.

    Worth seeing at least once: the cost of the symmetric constraint is not
    spread evenly, and on a model with tied word embeddings the tensor that pays
    most is also the output head.
    """
    kinds: dict[str, list[LayerError]] = {}
    for row in rows:
        kinds.setdefault(row.kind, []).append(row)

    print()
    header = f"{'kind':>18} {'weights':>12} {'sym':>9} {'asym':>9} {'mlx':>9} {'sym costs':>11}"
    print(header)
    print("-" * len(header))
    for kind, group in sorted(kinds.items()):
        size, sym, asym, mlx_ = _weighted(group)
        print(
            f"{kind:>18} {size:>12,} {sym:>9.4f} {asym:>9.4f} {mlx_:>9.4f} {sym / asym - 1:>+10.1%}"
        )
    print()


def _print_guide() -> None:
    """What each outcome means, so the table is read rather than skimmed."""
    print()
    print(RULE)
    print(" How to read this")
    print(RULE)
    print(" mround asym close to mlx affine   the quantizer is sound, and the")
    print("                                   symmetric column is the scheme's")
    print("                                   cost, which is Q-002's question.")
    print(" both mround columns far above     a defect in the quantizer, and the")
    print("                                   perplexity gap is a symptom.")
    print(" mround sym close to mlx affine    the schemes cost the same at that")
    print("                                   width, and what remains is round-to-")
    print("                                   nearest being lossy, which is what")
    print("                                   learned rounding exists to recover.")


def main() -> int:
    """Sweep the bit widths, measuring all three quantizers on every layer."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument(
        "--bits",
        default="2,3,4,8",
        help=(
            "comma-separated widths to sweep. A single width also prints the "
            "per-layer breakdown, which a sweep suppresses to stay readable."
        ),
    )
    parser.add_argument("--group-size", type=int, default=64)
    args = parser.parse_args()
    widths = [int(b) for b in args.bits.split(",")]

    try:
        import mlx.core as mx  # noqa: PLC0415

        from mround.core import quantizer  # noqa: PLC0415
        from mround.pipeline.blocks import iter_quantizable_modules  # noqa: PLC0415
        from mround.pipeline.loader import load_model  # noqa: PLC0415
    except ImportError as exc:
        print(f"needs MLX and the model stack: {exc}")
        return 1

    bundle = load_model(args.model)
    layers = [
        (path, module.weight)
        for path, module, eligible in iter_quantizable_modules(
            bundle.model, group_size=args.group_size
        )
        if eligible
    ]

    print(RULE)
    print(f" {args.model}, group size {args.group_size}, {len(layers)} layers")
    print(RULE)
    print("Relative reconstruction error of the weights, lower is better.")
    print()
    sweep = (
        f"{'bits':>6} {'mround sym':>12} {'mround asym':>13} {'mlx affine':>12} {'sym costs':>11}"
    )
    print(sweep)
    print("-" * len(sweep))

    for bits in widths:
        symmetric = QuantScheme(bits=bits, group_size=args.group_size)
        asymmetric = QuantScheme(
            bits=bits, group_size=args.group_size, symmetry=Symmetry.ASYMMETRIC
        )
        rows = [
            LayerError(
                size=int(weight.size),
                symmetric=mround_error(mx, quantizer, weight, symmetric),
                asymmetric=mround_error(mx, quantizer, weight, asymmetric),
                mlx=mlx_error(mx, weight, bits, args.group_size),
                kind=path.rsplit(".", 1)[-1],
            )
            for path, weight in layers
        ]
        _, sym, asym, mlx_ = _weighted(rows)
        print(f"{bits:>6} {sym:>12.4f} {asym:>13.4f} {mlx_:>12.4f} {sym / asym - 1:>+10.1%}")
        if len(widths) == 1:
            _print_by_layer(rows)

    print()
    print("The last column is what constraining the bias to -scale * 2**(bits-1)")
    print("costs. It is free to remove: both schemes export identical affine")
    print("tensors, so a checkpoint cannot tell which produced it.")
    print()
    print("Squaring the ratio of the first two columns predicts the ratio of the")
    print("perplexity penalties, and did so to within half a percent at 4 bits.")
    print("Whether that holds as the width drops is the thing to watch: it would")
    print("make this seconds-long probe a stand-in for two full model passes.")
    _print_guide()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
