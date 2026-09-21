# Copyright 2026 The MRound Authors
# SPDX-License-Identifier: Apache-2.0
"""What the array framework will tell us about the device it is running on.

One function, in its own module, because two pipeline stages need it and
neither belongs upstream of the other: the block runner reports its peak into
the run record, and the sensitivity scorer reports its peak into the progress
lines that keep a long scoring pass legible.

The peak matters more here than it would elsewhere. Apple Silicon's memory is
unified, so an MLX allocation that exceeds physical RAM does not fail, it
swaps, and a swapping run looks exactly like a hung one from the outside. A
number printed while the run is still going is what tells those two apart.
"""

from __future__ import annotations

import mlx.core as mx

__all__ = ["peak_memory", "reset_peak_memory"]


def peak_memory() -> int:
    """Peak MLX allocation so far, in bytes, or zero if this build will not say.

    The accessor moved from ``mx.metal`` to the top level between MLX releases.
    Both are tried and neither is required, because returning a zero from a
    reporting field is better than ending a twenty-minute run over it.
    """
    for owner in (mx, getattr(mx, "metal", None)):
        getter = getattr(owner, "get_peak_memory", None)
        if callable(getter):
            return int(getter())
    return 0


def reset_peak_memory() -> None:
    """Start the peak over, where this build allows it.

    The counter is process wide and MLX never resets it on its own, so a
    second quantization in the same process would otherwise report the first
    one's peak: ``examples/tune_model.py`` runs the round to nearest arm and
    then the tuned arm and records the tuned arm's figure. Each public entry
    point calls this first, so a run's peak is its own and, for a mixed run,
    includes its scoring pass by design (MEMORY.md D-036).
    """
    for owner in (mx, getattr(mx, "metal", None)):
        reset = getattr(owner, "reset_peak_memory", None)
        if callable(reset):
            reset()
            return
