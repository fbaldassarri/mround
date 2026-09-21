# Copyright 2026 The MRound Authors
# SPDX-License-Identifier: Apache-2.0
"""Core ML export, for Neural Engine execution.

**This is a research track, not a supported path.** It is Phase 7, it is
timeboxed, and it is permitted to conclude that a Core ML export is not worth
maintaining. A negative result published clearly is a successful outcome. See
MEMORY.md D-004.

What is settled: the ANE cannot run the tuning loop. It has no automatic
differentiation, no custom kernels, and no mechanism for arbitrary programs. It
is reachable only by compiling a model through Core ML, and it is inference
only. No amount of engineering changes that.

What is open, and what this module exists to answer: whether an MRound-quantized
model exported to Core ML runs on the ANE at a latency or energy advantage over
the same model on the GPU.

What is expected to constrain any attempt, from Apple's documentation rather
than from measurement: that Core ML's low-bit weight support is principally
compression with decompression before the arithmetic, so the benefit is
footprint and bandwidth rather than throughput; that per-channel scales are
preferred over the per-block granularity MRound's quality depends on; and that
integer 8-bit is the best-supported path, leaving 2-bit least likely to map.
Verifying those is the work.

Status: Phase 7, research. Not implemented.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from mround.schemes import QuantScheme

__all__ = ["ANECompatibility", "check_ane_compatibility", "export_coreml"]


@dataclasses.dataclass(frozen=True, slots=True)
class ANECompatibility:
    """Whether a scheme can plausibly execute on the Neural Engine.

    Attributes:
        compatible: Whether an export is worth attempting at all.
        reasons: Why not, when it is not. Empty when compatible.
        warnings: Concerns that do not block an export but will affect the
            result, such as a granularity that costs quality.
        requires_scale_conversion: Whether per-group scales must be coarsened to
            per-channel, which loses quality and must be reported to the user
            rather than done quietly.
    """

    compatible: bool
    reasons: list[str]
    warnings: list[str]
    requires_scale_conversion: bool


def check_ane_compatibility(scheme: QuantScheme) -> ANECompatibility:
    """Report whether ``scheme`` has a plausible Neural Engine path.

    Answers from documented constraints, not from measurement. Until Phase 7
    produces benchmarks, a ``compatible`` result means "worth trying", never
    "will be faster".

    Args:
        scheme: The representation to check.

    Returns:
        The assessment, with reasons attached either way.
    """
    raise NotImplementedError


def export_coreml(
    output_path: Path,
    model_dir: Path,
    scheme_by_layer: dict[str, QuantScheme],
) -> None:
    """Compile a quantized checkpoint to a Core ML package.

    Args:
        output_path: Destination ``.mlpackage``.
        model_dir: An MRound-quantized checkpoint.
        scheme_by_layer: The scheme each layer used.

    Raises:
        UnsupportedSchemeError: If any layer's scheme has no Core ML
            representation.
        ExportError: If compilation fails.
    """
    raise NotImplementedError
