# Copyright 2026 The MRound Authors
# SPDX-License-Identifier: Apache-2.0
"""Localize a forward-pass disagreement between MLX and the NumPy reference.

Run on an Apple Silicon Mac with the project environment active::

    python scripts/probe_forward.py

Why this exists, and what it found. It was written to localize a 0.13 percent gap
between MLX's round-to-nearest layer loss and the reference's, after a faithful
float32 emulation ruled out the obvious explanation. It answered the question at
stage 0 on its first run: `mx.matmul` returns 8.5e-4 relative error on float32
operands, four orders of magnitude worse than the dtype implies, and the loss was
being formed by subtracting two large products, which handed that whole error to
the small residual. See MEMORY.md D-014, which closed Q-007.

It is kept because that finding is a property of the hardware and the framework,
not of a bug that was fixed, so it will need re-measuring on other machines and
after MLX upgrades. Stage 0 is now the headline: if `mx.matmul` ever reports near
1e-7, several decisions recorded in MEMORY.md are worth revisiting.

This script walks those four operations in order and prints what each contributes.
Read it top to bottom and stop at the first stage that disagrees; everything below
that point is downstream of the cause. The stages are the quantizer, the two
matrix products, the residual, and the reduction, and each is compared against an
exact float64 computation of the same quantity.

The point is to answer one question per stage rather than to produce a verdict.
Paste the whole output when reporting.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

# Run from a plain checkout without an editable install. Every other entry point
# gets the package from `pip install -e .`; this one is typed by hand at a moment
# when something is already confusing, so it should not also fail on setup.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mround.reference import quantize as ref_quant  # noqa: E402
from mround.reference.losses import outlier_suppressed_loss, reconstruction_loss  # noqa: E402
from mround.schemes import QuantScheme, TuningConfig  # noqa: E402

if TYPE_CHECKING:
    from types import ModuleType

Array = npt.NDArray[np.float64]

BIT_WIDTHS = (2, 3, 4, 8)
RULE = "=" * 74


def synthetic(seed: int) -> tuple[Array, Array]:
    """Weights and correlated activations, identical to the parity suite's."""
    gen = np.random.default_rng(seed)
    weight: Array = gen.normal(scale=0.05, size=(16, 64))
    basis = gen.normal(size=(64, 64))
    covariance = basis @ basis.T / 64
    activations: Array = gen.multivariate_normal(np.zeros(64), covariance, size=128)
    return weight, activations


def rel(got: Array, expected: Array) -> float:
    """Max absolute difference, normalized by the expected magnitude."""
    denom = max(1e-300, float(np.abs(expected).max(initial=0.0)))
    return float(np.abs(got - expected).max(initial=0.0) / denom)


def probe_primitives(mx: ModuleType) -> None:
    """Check that matmul and the reductions are accurate before blaming anything else.

    A relative error near 1e-7 is float32 working as specified. Near 1e-3 would
    mean reduced precision somewhere in the runtime, and would explain the whole
    discrepancy on its own.
    """
    print(RULE)
    print(" Stage 0  are the primitives themselves accurate?")
    print(RULE)

    gen = np.random.default_rng(0)
    left: Array = gen.normal(size=(128, 64))
    right: Array = gen.normal(size=(64, 16))
    got = np.array(
        mx.matmul(mx.array(left.astype(np.float32)), mx.array(right.astype(np.float32)))
    ).astype(np.float64)
    print(f"   mx.matmul  (128,64)@(64,16)   relative error {rel(got, left @ right):.3e}")

    flat: Array = gen.normal(size=2048) ** 2
    device = mx.array(flat.astype(np.float32))
    exact_mean = float(flat.mean())
    exact_sum = float(flat.sum())
    got_mean = float(mx.mean(device))
    got_sum = float(mx.sum(device))
    print(
        f"   mx.mean    2048 elements      relative error "
        f"{abs(got_mean - exact_mean) / exact_mean:.3e}"
    )
    print(
        f"   mx.sum     2048 elements      relative error "
        f"{abs(got_sum - exact_sum) / exact_sum:.3e}"
    )
    print()


def probe_one_width(mx: ModuleType, quantizer: ModuleType, losses: ModuleType, bits: int) -> None:
    """Walk the four forward stages at one bit width."""
    weight, activations = synthetic(seed=bits)
    scheme = QuantScheme(bits=bits, group_size=32)
    suppress = TuningConfig().resolved_suppress_outliers(bits)

    print(RULE)
    print(f" {bits} bits, outlier suppression {'on' if suppress else 'off'}")
    print(RULE)

    # ---- Stage 1: the quantizer -----------------------------------------
    expected = ref_quant.fake_quantize(weight, ref_quant.init_params(weight, scheme), scheme)
    m_weight = mx.array(weight.astype(np.float32))
    m_params = quantizer.init_params(m_weight, scheme)
    m_qdq = np.array(quantizer.fake_quantize(m_weight, m_params, scheme)).astype(np.float64)
    m_codes, m_scale, _ = quantizer.quantize_codes(m_weight, m_params, scheme)

    differing = int(np.sum(np.array(m_codes).astype(np.float64) != expected.codes))
    step = float(np.abs(expected.scale).max())
    print(" Stage 1  quantizer")
    print(f"   codes differing from the reference   {differing} of {expected.codes.size}")
    print(
        f"   max |qdq_mlx - qdq_ref|              {np.abs(m_qdq - expected.qdq).max():.3e}"
        f"   (one code step is {step:.3e})"
    )
    print(
        f"   max relative scale difference        "
        f"{rel(np.array(m_scale).astype(np.float64), expected.scale):.3e}"
    )

    # ---- Stage 2: the two matrix products -------------------------------
    # Both exact products use MLX's own qdq, so only the matmul varies here.
    m_act = mx.array(activations.astype(np.float32))
    m_ref_out = m_act @ m_weight.T
    m_pred = m_act @ mx.array(m_qdq.astype(np.float32)).T
    exact_ref_out = activations @ weight.T
    exact_pred = activations @ m_qdq.T
    print(" Stage 2  matmuls")
    print(
        f"   reference_output vs exact            "
        f"{rel(np.array(m_ref_out).astype(np.float64), exact_ref_out):.3e}"
    )
    print(
        f"   predicted vs exact                   "
        f"{rel(np.array(m_pred).astype(np.float64), exact_pred):.3e}"
    )

    # ---- Stage 3: the residual ------------------------------------------
    m_residual = np.array(m_pred - m_ref_out).astype(np.float64)
    exact_residual = exact_pred - exact_ref_out
    ratio = float(np.sqrt((m_residual**2).mean()) / np.sqrt((exact_residual**2).mean()))
    print(" Stage 3  residual")
    print(f"   rms(mlx) / rms(exact)                {ratio:.9f}")
    print(
        f"   max |residual_mlx - residual_exact|  {np.abs(m_residual - exact_residual).max():.3e}"
    )

    # ---- Stage 4: the reduction -----------------------------------------
    # The decisive comparison. If the loss MLX reports equals the float64
    # reduction of MLX's OWN residual, the reduction is innocent and the cause is
    # upstream. If it instead equals the reference figure, the reduction is where
    # the difference enters.
    reported = float(
        (losses.outlier_suppressed_loss if suppress else losses.reconstruction_loss)(
            m_pred, m_ref_out
        )
    )
    ref_loss = outlier_suppressed_loss if suppress else reconstruction_loss
    on_mlx, _ = ref_loss(m_residual, np.zeros_like(m_residual))
    on_exact, _ = ref_loss(exact_residual, np.zeros_like(exact_residual))
    print(" Stage 4  reduction")
    print(f"   loss reported by MLX                 {reported:.17g}")
    print(
        f"   float64 reduction of MLX's residual  {on_mlx:.17g}"
        f"   ({(reported - on_mlx) / on_mlx:+.4%} vs MLX)"
    )
    print(
        f"   float64 reduction of exact residual  {on_exact:.17g}"
        f"   ({(reported - on_exact) / on_exact:+.4%} vs MLX)"
    )
    print()


def main() -> int:
    """Walk the forward pass stage by stage and report where MLX diverges."""
    # Imported here rather than at module scope on purpose: this script is run by
    # hand on a machine that may not have MLX, and a one-line explanation beats a
    # traceback out of an import statement.
    try:
        import mlx.core as mx  # noqa: PLC0415

        from mround.core import losses as mlx_losses  # noqa: PLC0415
        from mround.core import quantizer as mlx_quant  # noqa: PLC0415
    except ImportError as exc:
        print(f"MLX is unavailable, so there is nothing to probe: {exc}")
        return 1

    probe_primitives(mx)
    for bits in BIT_WIDTHS:
        probe_one_width(mx, mlx_quant, mlx_losses, bits)

    print(RULE)
    print(" How to read this")
    print(RULE)
    print(" Stage 1 nonzero      the quantizer diverges; the loss gap is a symptom.")
    print(" Stage 2 above 1e-6   a matmul is not running at full float32.")
    print(" Stage 3 ratio off 1  the residual is systematically scaled, which is the")
    print("                      only mechanism that can LOWER the loss.")
    print(" Stage 4 rows 1 and 2 apart               the reduction introduces it.")
    print(" Stage 4 rows 1 and 2 equal, row 3 apart  everything upstream does.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
