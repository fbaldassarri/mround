# Copyright 2026 The MRound Authors
# SPDX-License-Identifier: Apache-2.0
"""Typed errors.

MRound refuses rather than approximates. A quantizer that quietly produces
something other than what was requested is worse than one that fails, so the
error types here exist to make refusal specific enough to act on.
"""

from __future__ import annotations

__all__ = [
    "ArchitectureError",
    "CalibrationError",
    "ExportError",
    "MRoundError",
    "MemoryBudgetError",
    "PlatformError",
    "SchemeError",
    "UnsupportedSchemeError",
]


class MRoundError(Exception):
    """Base class for every error MRound raises deliberately."""


class PlatformError(MRoundError):
    """The host cannot run this operation.

    Raised when the host is not macOS on Apple Silicon or the interpreter is
    not a native arm64 build (``pipeline/loader.py``). MRound targets Apple
    Silicon only and does not degrade to a portable fallback.
    """


class SchemeError(MRoundError, ValueError):
    """A quantization scheme is malformed or inapplicable to this weight.

    Also a :class:`ValueError`, because that is what a malformed dataclass
    argument is and what a caller checking the value of one field expects;
    the hierarchy adds the ability to catch every deliberate MRound refusal
    under :class:`MRoundError` without losing that.
    """


class UnsupportedSchemeError(SchemeError):
    """A target format cannot represent the requested scheme faithfully.

    Raised by exporters rather than approximating. The message must name both
    the scheme and the format so the caller can choose another.
    """


class CalibrationError(MRoundError):
    """Calibration data is missing, malformed, or inconsistent with the model."""


class ArchitectureError(MRoundError):
    """The model's structure could not be interpreted.

    Raised when block discovery finds no transformer blocks, or finds a layout
    the pipeline does not know how to walk.
    """


class ExportError(MRoundError):
    """A checkpoint could not be written."""


class MemoryBudgetError(MRoundError):
    """A run is predicted to need more memory than it may use.

    Raised before the model is loaded, by the check in ``mround/api.py`` that
    applies the rule of ``mround/planner/memory.py``. On Apple Silicon a run
    past the machine's memory does not fail, it swaps, hours in, so the
    message says what the run would reach, against what, and what would fit.
    """
