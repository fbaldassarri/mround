#!/usr/bin/env bash
# Copyright 2026 The MRound Authors
# SPDX-License-Identifier: Apache-2.0
#
# Verify the MLX implementation against the NumPy reference.
#
#   conda activate <your mround environment>
#   bash scripts/check_mlx.sh
#
# Run this on an Apple Silicon Mac. It reports the environment, checks that the
# prerequisites are present, installs MLX if absent, runs the parity suite, and
# finishes with a side-by-side of the two implementations on the same layer.
# Paste the whole output when reporting.
#
# The environment name does not matter; the script uses whichever Python is
# active. What matters is that the project is installed into it.

set -uo pipefail

BOLD=$(tput bold 2>/dev/null || echo "")
PLAIN=$(tput sgr0 2>/dev/null || echo "")

rule() { printf '==================================================================\n'; }
section() { echo; rule; echo " $1"; rule; }

section "Environment"
python - <<'PYTHON'
import platform
import sys

print(f"python      {sys.version.split()[0]}  ({sys.executable})")
print(f"processor   {platform.processor()}   <- must be arm, not i386")
print(f"machine     {platform.machine()}")
print(f"macOS       {platform.mac_ver()[0] or 'n/a'}")

for module, label in (("numpy", "numpy"), ("pytest", "pytest"), ("mround", "mround")):
    try:
        mod = __import__(module)
    except ImportError:
        print(f"{label:<11} NOT INSTALLED")
    else:
        version = getattr(mod, "__version__", "present")
        print(f"{label:<11} {version}")

try:
    import mlx.core as mx
except ImportError:
    print("mlx         not installed")
else:
    print(f"mlx         {getattr(mx, '__version__', 'present')}")
PYTHON

# ---------------------------------------------------------------------------
# Prerequisites, checked before anything else runs.
#
# The point of doing this first is that a missing package should produce one
# clear message rather than a cascade of unrelated failures further down. An
# earlier version of this script skipped straight to pytest and reported "No
# module named pytest" followed by "No module named numpy", which is two
# symptoms of one cause and says nothing about the fix.
# ---------------------------------------------------------------------------
MISSING=""
for module in numpy pytest mround; do
  python -c "import $module" >/dev/null 2>&1 || MISSING="$MISSING $module"
done

if [ -n "$MISSING" ]; then
  section "Prerequisites missing"
  echo "Not importable in this environment:$MISSING"
  echo
  echo "The usual cause is an environment created bare, for example with"
  echo "  conda create -n mround-dev python=3.11"
  echo "which never runs the pip section of environment.yml."
  echo
  echo "${BOLD}Fix, from the repository root with the environment active:${PLAIN}"
  echo
  echo "    pip install -e '.[dev]'"
  echo
  echo "Then run this script again. Nothing below would have worked, so it is"
  echo "skipped rather than run to produce more failures."
  rule
  exit 1
fi

# ---------------------------------------------------------------------------
# MLX is deliberately not a default in environment.yml, so that the file also
# works on a machine used only for the framework-free reference layer. Install
# it here when it is absent.
# ---------------------------------------------------------------------------
if ! python -c "import mlx.core" >/dev/null 2>&1; then
  section "Installing MLX"
  pip install "mlx>=0.32" || {
    echo
    echo "MLX failed to install. The usual cause is a non-native Python; check"
    echo "that 'processor' above says arm rather than i386. See environment.yml."
    exit 1
  }
fi

section "Parity suite: MLX against the NumPy reference"
python -m pytest tests/unit/test_mlx_parity.py -v --tb=short
PARITY=$?

section "Side by side, several seeds per bit width"
python - <<'PYTHON'
import statistics

import numpy as np

from mround.reference import losses as ref_losses
from mround.reference import tuning as ref
from mround.schemes import QuantScheme, TuningConfig

try:
    import mlx.core as mx

    from mround.core import tuning as mlx_tuning
except ImportError as exc:  # pragma: no cover
    raise SystemExit(f"MLX unavailable, nothing to compare: {exc}")

SEEDS = (0, 100, 200, 300, 400)
CONFIG = TuningConfig(iters=200)


def synthetic(seed, out_features=16, in_features=64, n_samples=128):
    gen = np.random.default_rng(seed)
    weight = gen.normal(scale=0.05, size=(out_features, in_features))
    basis = gen.normal(size=(in_features, in_features))
    covariance = basis @ basis.T / in_features
    activations = gen.multivariate_normal(np.zeros(in_features), covariance, size=n_samples)
    return weight, activations


def score(weight, activations, qdq, bits):
    """The float64 reference loss of a reconstruction, whoever produced it.

    Both sides get judged by the same yardstick. Comparing each implementation's
    self-reported loss would compare two measurements as well as two solutions,
    and MLX's measurement carries a known bias (MEMORY.md D-014). Taking the
    minimum over 200 steps of a noisy estimate is also biased low, so scoring the
    learned reconstruction afterwards removes both effects at once.
    """
    residual = activations @ (np.asarray(qdq, dtype=np.float64) - weight).T
    reduce = (
        ref_losses.outlier_suppressed_loss
        if CONFIG.resolved_suppress_outliers(bits)
        else ref_losses.reconstruction_loss
    )
    value, _ = reduce(residual, np.zeros(()))
    return value


print("Round-to-nearest is the same computation on both sides, so its disagreement")
print("is a precision measurement and it is stable. The tuned figures are not:")
print("signed gradient descent is chaotic, one flipped sign sends the two down")
print("different paths, and the reference alone varies by up to 17 points across")
print("seeds at 2 bits. So several seeds are run and the spread is printed. Judge")
print("by the medians and by that spread, never by a single row.")
print()
header = (
    f"{'bits':>5} {'seed':>5} {'RTN ref':>12} {'RTN mlx':>12} {'RTN gap':>10}"
    f" {'ref best':>12} {'mlx best':>12} {'gap':>8}"
)
print(header)
print("-" * len(header))

worst_rtn = 0.0
worst_tuned = 0.0
for bits in (2, 3, 4, 8):
    scheme = QuantScheme(bits=bits, group_size=32)
    ref_scores, mlx_scores, gaps = [], [], []

    for seed in SEEDS:
        weight, activations = synthetic(seed=seed + bits)
        r = ref.tune_layer(weight, activations, scheme, CONFIG)
        m = mlx_tuning.tune_layer(
            mx.array(weight.astype(np.float32)),
            mx.array(activations.astype(np.float32)),
            scheme,
            CONFIG,
        )

        rtn_gap = abs(m.initial_loss - r.initial_loss) / max(1e-30, r.initial_loss)
        worst_rtn = max(worst_rtn, rtn_gap)

        ref_best = score(weight, activations, r.qdq, bits)
        mlx_best = score(weight, activations, np.array(m.qdq), bits)
        gap = abs(mlx_best - ref_best) / max(1e-30, ref_best)
        worst_tuned = max(worst_tuned, gap)

        ref_scores.append(ref_best)
        mlx_scores.append(mlx_best)
        gaps.append(gap)
        print(
            f"{bits:>5} {seed + bits:>5} {r.initial_loss:>12.4e} {m.initial_loss:>12.4e}"
            f" {rtn_gap:>10.2e} {ref_best:>12.4e} {mlx_best:>12.4e} {gap:>7.1%}"
        )

    print(
        f"{bits:>5} {'med':>5} {'':>12} {'':>12} {'':>10}"
        f" {statistics.median(ref_scores):>12.4e} {statistics.median(mlx_scores):>12.4e}"
        f" {statistics.median(gaps):>7.1%}"
    )
    print()

print(f"worst RTN disagreement    {worst_rtn:.2e}   <- INITIAL_LOSS_TOL goes above this")
print(f"worst tuned disagreement  {worst_tuned:.1%}   (both scored by the reference loss)")
print()
print("A flat RTN gap across bit widths is the signature D-014 predicts: the error")
print("is proportional to the residual now, not to the full layer output, so it no")
print("longer grows as the residual shrinks. A gap that climbs with bit width would")
print("mean the subtraction has crept back in somewhere.")
PYTHON

section "Result"
if [ $PARITY -eq 0 ]; then
  echo " Parity suite PASSED. Phase 1b is verified on this machine."
else
  echo " Parity suite FAILED (exit $PARITY). The output above names what."
fi
rule
exit $PARITY
