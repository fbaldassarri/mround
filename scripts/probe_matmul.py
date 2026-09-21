# Copyright 2026 The MRound Authors
# SPDX-License-Identifier: Apache-2.0
"""Characterize MLX's float32 matrix product, and find out what recovers it.

Run on an Apple Silicon Mac with the project environment active::

    python scripts/probe_matmul.py

Why this exists. `scripts/probe_forward.py` closed MEMORY.md Q-007 by measuring
`mx.matmul` at 8.5e-4 relative error on float32 inputs, roughly four orders of
magnitude worse than float32 warrants and squarely in reduced-precision
territory. D-014 removed the consequence from the single-layer loss by
multiplying the weight difference instead of subtracting two large products, but
that trick is specific to one linear layer. The Phase 2 block objective compares
two activation tensors with nonlinearities between them and cannot be rewritten
that way, so the underlying behavior has to be understood rather than routed
around a second time.

Five questions. The first two are answered, and their answers are recorded in
MEMORY.md D-015; they are kept because both need re-measuring on other machines
and after MLX upgrades.

1. How does the error scale with the reduction length? *Answered: it does not.*
   The relative error is flat from K=16 to K=4096, which rules out accumulation
   and points at the inputs being truncated before the product.
2. Is it the GPU matrix-product kernel specifically? *Answered: yes.* The same
   product on the CPU stream is accurate to 2.3e-7, and an explicit
   multiply-then-reduce on the GPU is accurate to 8.2e-8. Only the GPU kernel
   loses precision.
3. What is the effective format? Compare against float16 and bfloat16 inputs.
4. What recovers accuracy, and at what cost? Split-precision passes, which are
   the standard remedy when a kernel truncates its inputs.
5. What did all that mean for the layer residual, both ways round?

Nothing here is a benchmark. It measures accuracy only, and it prints numbers
rather than verdicts. Paste the whole output when reporting.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if TYPE_CHECKING:
    from types import ModuleType

Array = npt.NDArray[np.float64]

RULE = "=" * 74
ROWS = 128
COLS = 16

# Split-precision pass counts at which each cross term switches on.
CROSS_TERM_RIGHT = 2
CROSS_TERM_LEFT = 3


def operands(k: int, seed: int = 0) -> tuple[Array, Array]:
    """A well-conditioned pair with a reduction length of ``k``."""
    gen = np.random.default_rng(seed)
    return gen.normal(size=(ROWS, k)), gen.normal(size=(k, COLS))


def rel(got: Array, expected: Array) -> float:
    """Max absolute difference, normalized by the expected magnitude."""
    denom = max(1e-300, float(np.abs(expected).max(initial=0.0)))
    return float(np.abs(got - expected).max(initial=0.0) / denom)


def as_numpy(mx: ModuleType, value: object) -> Array:
    """Bring an MLX array back to float64 on the host.

    The widening to float32 happens inside MLX, before crossing the boundary.
    NumPy has no bfloat16, so handing it one raises a buffer-protocol error about
    item sizes that says nothing about the actual problem. This function takes
    ``mx`` for that single reason.
    """
    return np.array(value.astype(mx.float32)).astype(np.float64)  # type: ignore[attr-defined]


def split_matmul(mx: ModuleType, left: object, right: object, passes: int) -> Any:
    """Matrix product recovered by splitting each operand into two parts.

    If the kernel truncates its inputs to a narrow format, the high parts survive
    that truncation exactly and the cross terms carry back the bits that were
    dropped. Two passes recover roughly half the lost precision, three most of
    it, at two and three times the arithmetic.
    """
    left_hi = left.astype(mx.bfloat16).astype(mx.float32)  # type: ignore[attr-defined]
    right_hi = right.astype(mx.bfloat16).astype(mx.float32)  # type: ignore[attr-defined]
    out = mx.matmul(left_hi, right_hi)
    if passes >= CROSS_TERM_RIGHT:
        out = out + mx.matmul(left_hi, right - right_hi)
    if passes >= CROSS_TERM_LEFT:
        out = out + mx.matmul(left - left_hi, right_hi)
    return out


def report_scaling(mx: ModuleType) -> None:
    """Relative error against the reduction length."""
    print(RULE)
    print(" 1  error against reduction length K")
    print(RULE)
    print(f"   {'K':>6}  {'rel err':>10}  {'err/sqrt(K)':>12}   float32 would be ~1e-7")
    for k in (16, 64, 256, 1024, 4096):
        left, right = operands(k)
        got = as_numpy(
            mx, mx.matmul(mx.array(left.astype(np.float32)), mx.array(right.astype(np.float32)))
        )
        error = rel(got, left @ right)
        print(f"   {k:>6}  {error:>10.3e}  {error / np.sqrt(k):>12.3e}")
    print()
    print("   A flat first column is the informative outcome, and it is what this")
    print("   machine gave. Error growing with K would mean a narrow accumulator.")
    print("   Error holding still while the reduction gets 256 times longer means")
    print("   the accumulator is fine and the inputs are rounded before they are")
    print("   multiplied: each product loses a fixed relative amount, and so does")
    print("   their sum.")
    print()


def report_devices(mx: ModuleType) -> None:
    """The same product on each device, and without the matrix-product kernel."""
    print(RULE)
    print(" 2  is it the GPU matrix-product kernel?")
    print(RULE)
    left, right = operands(64)
    exact = left @ right

    for name in ("gpu", "cpu"):
        device = getattr(mx, name, None)
        if device is None:
            print(f"   {name:>26}   not available in this MLX build")
            continue
        with mx.stream(device):
            got = as_numpy(
                mx, mx.matmul(mx.array(left.astype(np.float32)), mx.array(right.astype(np.float32)))
            )
        print(f"   mx.matmul on {name:>13}   {rel(got, exact):.3e}")

    # Elementwise multiply then reduce. Same mathematics, different code path,
    # and it never reaches the matrix-product kernel. Memory-hungry, so this is a
    # diagnostic rather than a proposal.
    m_left = mx.array(left.astype(np.float32))
    m_right = mx.array(right.astype(np.float32))
    manual = as_numpy(mx, mx.sum(m_left[:, :, None] * m_right[None, :, :], axis=1))
    print(f"   {'multiply then reduce':>26}   {rel(manual, exact):.3e}")
    print()


def report_formats(mx: ModuleType) -> None:
    """Where the float32 path sits relative to the narrow formats."""
    print(RULE)
    print(" 3  what precision is it actually delivering?")
    print(RULE)
    left, right = operands(64)
    exact = left @ right
    for label, dtype in (
        ("float32", mx.float32),
        ("bfloat16", mx.bfloat16),
        ("float16", mx.float16),
    ):
        got = as_numpy(
            mx,
            mx.matmul(
                mx.array(left.astype(np.float32)).astype(dtype),
                mx.array(right.astype(np.float32)).astype(dtype),
            ),
        )
        print(f"   inputs as {label:>9}   {rel(got, exact):.3e}")
    print()
    print("   Section 1 established that the inputs are being rounded rather than")
    print("   the accumulator being narrow. This says how far. Whichever narrow row")
    print("   the float32 row sits closest to is the effective mantissa width the")
    print("   kernel is working at, and it is what a split has to work around.")
    print()


def report_recovery(mx: ModuleType) -> None:
    """What split-precision buys, and at what multiple of the arithmetic."""
    print(RULE)
    print(" 4  what recovers accuracy, and at what cost?")
    print(RULE)
    left, right = operands(64)
    exact = left @ right
    m_left = mx.array(left.astype(np.float32))
    m_right = mx.array(right.astype(np.float32))
    plain = rel(as_numpy(mx, mx.matmul(m_left, m_right)), exact)
    print(f"   {'1 product (as written today)':>32}   {plain:.3e}")
    for passes in (2, 3):
        got = as_numpy(mx, split_matmul(mx, m_left, m_right, passes))
        gain = plain / max(1e-300, rel(got, exact))
        print(
            f"   {f'{passes} products (split precision)':>32}   {rel(got, exact):.3e}   "
            f"{gain:.0f}x better"
        )
    print()


def report_layer_residual(mx: ModuleType) -> None:
    """The thing that actually matters: the residual, both ways round."""
    from mround.core import quantizer as mlx_quant  # noqa: PLC0415
    from mround.reference import quantize as ref_quant  # noqa: PLC0415
    from mround.schemes import QuantScheme  # noqa: PLC0415

    print(RULE)
    print(" 5  the layer residual: three ways of getting it")
    print(RULE)
    print("   Errors are relative to the residual's own magnitude, which is the")
    print("   quantity being minimized, not to the layer output.")
    print()
    print("   subtracted    x@Wq.T minus a cached x@W.T. What the code did before")
    print("                 D-014, and the only shape available to a block loss.")
    print("   split         the same subtraction with split-precision products.")
    print("                 The Q-008 candidate: 3 times the arithmetic.")
    print("   differenced   x@(Wq-W).T. What the code does now. Not available to a")
    print("                 block loss, which is why the middle column matters.")
    print()
    print(
        f"   {'bits':>5}  {'subtracted':>12}  {'split':>12}  {'differenced':>12}"
        f"  {'split gains':>12}"
    )

    for bits in (2, 3, 4, 8):
        gen = np.random.default_rng(bits)
        weight: Array = gen.normal(scale=0.05, size=(16, 64))
        basis = gen.normal(size=(64, 64))
        activations: Array = gen.multivariate_normal(np.zeros(64), basis @ basis.T / 64, size=128)
        scheme = QuantScheme(bits=bits, group_size=32)

        m_weight = mx.array(weight.astype(np.float32))
        m_act = mx.array(activations.astype(np.float32))
        qdq = mlx_quant.fake_quantize(m_weight, mlx_quant.init_params(m_weight, scheme), scheme)

        exact_qdq = ref_quant.fake_quantize(
            weight, ref_quant.init_params(weight, scheme), scheme
        ).qdq
        exact = activations @ (exact_qdq - weight).T

        subtracted = as_numpy(mx, m_act @ qdq.T - m_act @ m_weight.T)
        split = as_numpy(
            mx,
            split_matmul(mx, m_act, qdq.T, CROSS_TERM_LEFT)
            - split_matmul(mx, m_act, m_weight.T, CROSS_TERM_LEFT),
        )
        differenced = as_numpy(mx, m_act @ (qdq - m_weight).T)
        old, mid, new = rel(subtracted, exact), rel(split, exact), rel(differenced, exact)
        print(
            f"   {bits:>5}  {old:>12.3e}  {mid:>12.3e}  {new:>12.3e}"
            f"  {old / max(1e-300, mid):>11.0f}x"
        )
    print()
    print("   Read the middle column against the 8-bit row of the first. That is")
    print("   the Phase 2 trade: three products against however much of a 20")
    print("   percent noise floor it removes.")
    print()


def main() -> int:
    """Answer the four questions in order, then show what it meant for the layer."""
    try:
        import mlx.core as mx  # noqa: PLC0415
    except ImportError as exc:
        print(f"MLX is unavailable, so there is nothing to probe: {exc}")
        return 1

    print(RULE)
    print(" Device")
    print(RULE)
    print(f"   mlx            {getattr(mx, '__version__', 'unknown')}")
    print(f"   default device {mx.default_device()}")
    try:
        for key, value in mx.metal.device_info().items():
            print(f"   {key:<14} {value}")
    except (AttributeError, RuntimeError) as exc:
        print(f"   device info unavailable: {exc}")
    print()

    report_scaling(mx)
    report_devices(mx)
    report_formats(mx)
    report_recovery(mx)
    report_layer_residual(mx)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
