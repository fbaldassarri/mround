# Copyright 2026 The MRound Authors
# SPDX-License-Identifier: Apache-2.0
"""What the array framework will tell us about the device it is running on.

A few functions and one small class, in their own module. Two pipeline stages
need the peak and neither belongs upstream of the other: the block runner
reports it into the run record, stage by stage since 2026-10-10, and the
sensitivity scorer reports it into the progress lines that keep a long scoring
pass legible. Resetting starts the count over, and each public entry point in
``mround.api`` does so before loading a model. The check before a run
(``mround/planner/memory.py``) needs the device's size, which only MLX can
say, and reads it here.

The peak matters more here than it would elsewhere. Apple Silicon's memory is
unified, so an MLX allocation that exceeds physical RAM does not fail, it
swaps, and a swapping run looks exactly like a hung one from the outside. A
number printed while the run is still going is what tells those two apart.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

import mlx.core as mx

if TYPE_CHECKING:
    from collections.abc import Iterator

__all__ = ["StageMeter", "active_memory", "device_memory", "peak_memory", "reset_peak_memory"]


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


def active_memory() -> int:
    """MLX allocation in use now, in bytes, or zero if this build will not say."""
    for owner in (mx, getattr(mx, "metal", None)):
        getter = getattr(owner, "get_active_memory", None)
        if callable(getter):
            return int(getter())
    return 0


def device_memory() -> int:
    """The memory of the device MLX runs on, in bytes, or zero if it will not say.

    On a Mac this is the whole unified memory, which the GPU and everything
    else share: a 32 GB Mac has 34.36 GB of 10^9 bytes. The CPU build reports
    no size at all, which is the zero a caller has to treat as unknown rather
    than as none. The accessor moved to the top level from ``mx.metal``
    between MLX releases, and the old one is asked only where the new one is
    missing, since it warns that it is going.
    """
    owner = mx if callable(getattr(mx, "device_info", None)) else getattr(mx, "metal", None)
    info = getattr(owner, "device_info", None)
    if not callable(info):
        return 0
    try:
        size = info().get("memory_size", 0)
    except (RuntimeError, AttributeError, TypeError):
        return 0
    return int(size) if isinstance(size, int) and size > 0 else 0


class StageMeter:
    """The peak of each stage of a run, and of the run as a whole.

    One process wide counter serves both, so each stage starts it over and the
    run's peak is kept here as the largest stage seen. What a run reached before
    its first stage (loading, a mixed run's scoring pass, the calibration set)
    is read once, when the meter is made, which is what the entry points' reset
    makes the run's own figure.

    A stage reads at least the memory that was live when it began. MLX raises
    its peak only when it allocates, so a stage that allocated nothing would
    otherwise report less than it held throughout.

    Attributes:
        peak: The largest figure observed so far, across every stage.
        stages: The peak of each stage since the last :meth:`take`.
    """

    def __init__(self) -> None:
        """Start a meter, taking in whatever the run reached before it."""
        self.peak = max(peak_memory(), active_memory())
        self.stages: dict[str, int] = {}

    @contextlib.contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """Measure the peak of the code run inside, under ``name``.

        A name seen twice since the last :meth:`take` keeps the larger figure.
        """
        floor = active_memory()
        reset_peak_memory()
        try:
            yield
        finally:
            reached = max(peak_memory(), floor)
            self.stages[name] = max(self.stages.get(name, 0), reached)
            self.peak = max(self.peak, reached)

    def take(self) -> dict[str, int]:
        """The stages measured since the last call, and start a new set."""
        taken, self.stages = self.stages, {}
        return taken
