# Copyright 2026 The MRound Authors
# SPDX-License-Identifier: Apache-2.0
"""Importance matrix accumulation.

Accumulates the sum of squared activations per input channel over the
calibration set, so that reconstruction error can be weighted by how much each
input actually matters. A weight that is always multiplied by near-zero
activations does not deserve precision.

MLX has no forward-hook mechanism, so this cannot intercept a running model. The
accumulator is driven explicitly by the block runner instead, which is more code
and also more legible. See DOCUMENTATION.md section 7.

Specification: DOCUMENTATION.md section 1.5.

Status: Not implemented, and Phase 3 no longer needs it. The block loop's live
importance collection landed inside the runner's tunable wrapper
(``mround/pipeline/runner.py``), where the activation stream actually flows and
where accumulating stays lazy; see MEMORY.md D-030. What remains for this
module is the standalone framework-free accumulator for the GGUF imatrix
export path, which is later-phase work.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import mlx.core as mx

__all__ = ["ImportanceAccumulator"]


class ImportanceAccumulator:
    """Collects per-input-channel activation energy across calibration batches.

    Accumulation is in float32. The quantity summed grows with the calibration
    set and reduced precision loses the tail of the distribution, which is
    precisely the part that distinguishes channels worth protecting.
    """

    def __init__(self) -> None:
        """Create an empty accumulator."""
        raise NotImplementedError

    def update(self, name: str, activations: mx.array) -> None:
        """Fold one batch of a layer's inputs into the running sum.

        Args:
            name: Layer identifier, matching what the runner uses elsewhere.
            activations: Layer inputs, shaped ``(..., in_features)``. Leading
                dimensions are flattened before reduction.
        """
        raise NotImplementedError

    def get(self, name: str) -> mx.array | None:
        """Return a layer's accumulated importance, or ``None`` if unseen."""
        raise NotImplementedError

    def normalized(self, name: str) -> mx.array | None:
        """Return a layer's importance scaled to unit mean.

        Normalizing keeps the search objective's magnitude independent of
        calibration set size, which matters because the scale search compares
        weighted errors against each other rather than against a threshold.
        """
        raise NotImplementedError

    def clear(self) -> None:
        """Drop all accumulated state.

        Called between blocks. Importance is per layer and holding it for the
        whole model wastes memory that the block loop needs.
        """
        raise NotImplementedError

    def __contains__(self, name: str) -> bool:
        """Whether any activations have been accumulated for ``name``."""
        raise NotImplementedError
