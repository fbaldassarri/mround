# Copyright 2026 The MRound Authors
# SPDX-License-Identifier: Apache-2.0
"""Watch learned rounding beat round-to-nearest, in about a second.

Run it:

    conda activate mround-dev
    python examples/first_quantization.py

This uses ``mround.reference``, the NumPy implementation of the quantization
method. It needs no model, no MLX, and no Apple hardware, which is why it works
as a first thing to run on any machine.
"""

from __future__ import annotations

import numpy as np

from mround.reference.tuning import round_to_nearest, tune_layer
from mround.schemes import QuantScheme, TuningConfig


def synthetic_layer(
    seed: int = 0,
    out_features: int = 32,
    in_features: int = 128,
    n_samples: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """A weight matrix and correlated activations to feed it.

    The activations are correlated across features rather than white noise,
    because that is what real ones are. Learned rounding works by exploiting the
    fact that errors in different weights interact through the activation
    covariance, so white inputs would understate what the method does.
    """
    gen = np.random.default_rng(seed)
    weight = gen.normal(scale=0.05, size=(out_features, in_features))
    basis = gen.normal(size=(in_features, in_features))
    covariance = basis @ basis.T / in_features
    activations = gen.multivariate_normal(np.zeros(in_features), covariance, size=n_samples)
    return weight, activations


def main() -> None:
    """Quantize one layer at several bit widths and report the improvement."""
    weight, activations = synthetic_layer()
    reference_output = activations @ weight.T

    print(f"layer: {weight.shape[0]} x {weight.shape[1]}, {activations.shape[0]} samples\n")
    print(f"{'bits':>5} {'group':>6} {'round-to-nearest':>18} {'learned':>14} {'better by':>11}")
    print("-" * 60)

    for bits in (2, 3, 4, 8):
        scheme = QuantScheme(bits=bits, group_size=128)

        # The baseline: quantize with no learning at all.
        rtn = round_to_nearest(weight, scheme)
        rtn_error = float(np.mean((activations @ rtn.T - reference_output) ** 2))

        # The method: learn the rounding against this layer's own output.
        result = tune_layer(weight, activations, scheme, TuningConfig(iters=200))

        print(
            f"{bits:>5} {scheme.group_size:>6} {rtn_error:>18.4e} "
            f"{result.final_loss:>14.4e} {result.improvement:>10.1%}"
        )

    print(
        "\nEvery row is the same weights and the same bit budget. The only "
        "difference\nis whether the rounding decisions were chosen or learned."
    )


if __name__ == "__main__":
    main()
