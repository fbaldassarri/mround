# Copyright 2026 The MRound Authors
# SPDX-License-Identifier: Apache-2.0
"""Quantization schemes and tuning configuration.

Every architectural layer needs to agree on what a quantization scheme is, so
the definition lives here rather than in any one of them. This module imports
nothing from MLX and nothing from the rest of the package beyond its error
types, which is what keeps it usable from the planner (which knows no MLX) and
the formats layer (which knows no models).

Defaults follow DOCUMENTATION.md section 5. Where this module and that document
disagree, one of them is wrong and the disagreement must be resolved rather than
tolerated.
"""

from __future__ import annotations

import dataclasses
from enum import StrEnum
from typing import Self

from mround.exceptions import SchemeError

__all__ = [
    "DEFAULT_EPS_FP32",
    "DEFAULT_EPS_LOW_PRECISION",
    "LOW_BIT_THRESHOLD",
    "SUPPORTED_BITS",
    "QuantScheme",
    "ScaleInit",
    "Symmetry",
    "TuningConfig",
]

# Bit widths the packing layout and export paths support. 8, 4, and 2 are the
# focus (MEMORY.md D-007); the rest come along with the cross-word packing
# layout for widths that do not divide 32 evenly.
SUPPORTED_BITS: frozenset[int] = frozenset({2, 3, 4, 5, 6, 7, 8})

# Below this width, quantization behaves differently enough to change defaults:
# the learning rate constant doubles and outlier suppression switches on. Named
# because it appears in more than one place and the two uses must agree.
LOW_BIT_THRESHOLD: int = 4

# Scale epsilon, guarding against all-zero groups. DOCUMENTATION.md section 5.2.
DEFAULT_EPS_FP32: float = 1e-8
DEFAULT_EPS_LOW_PRECISION: float = 1e-5


class Symmetry(StrEnum):
    """Whether the quantization grid is centered on zero.

    SYMMETRIC uses a signed code range and no learned zero point; the scale
    carries the sign opposite to the group's larger-magnitude extreme, so that
    the extreme lands on the code with the larger magnitude and reconstructs
    exactly (DOCUMENTATION.md section 5.2.1, MEMORY.md D-009). ASYMMETRIC uses
    an unsigned code range and a learned zero point that participates in
    gradients.
    """

    SYMMETRIC = "sym"
    ASYMMETRIC = "asym"


class ScaleInit(StrEnum):
    """How the group scale is initialized before tuning.

    OBSERVED_RANGE derives the scale from the clipped group extremes, which is
    the v1 parameterization. SEARCHED runs the v2 grid search over candidate
    scales and makes the clipping coefficient a multiplier on the winner, which
    changes both its meaning and its permitted range.

    The two are not interchangeable at the parameter level. See
    DOCUMENTATION.md sections 5.2 and 5.3.
    """

    OBSERVED_RANGE = "observed_range"
    SEARCHED = "searched"


@dataclasses.dataclass(frozen=True, slots=True)
class QuantScheme:
    """How a single layer's weights are quantized.

    This describes the target representation, not how it is reached. The tuning
    procedure is described by :class:`TuningConfig`.

    Attributes:
        bits: Weight bit width. Must be in :data:`SUPPORTED_BITS`.
        group_size: Weights sharing one scale, along the input dimension.
            ``-1`` means one group per output channel (per-channel scales).
            The default is 64 rather than the 128 the reference uses, because
            64 divides strictly more models: a layer is only groupable when its
            input dimension is a multiple of the group size, every dimension
            divisible by 128 is divisible by 64, and the converse fails often.
            SmolLM2-135M is the measured case, hidden size 576, where 128
            leaves six of every seven layers dense. MEMORY.md, 2026-09-20.
        symmetry: Signed or unsigned code range.
        scale_init: Which scale parameterization applies.
    """

    bits: int = 4
    group_size: int = 64
    symmetry: Symmetry = Symmetry.SYMMETRIC
    scale_init: ScaleInit = ScaleInit.OBSERVED_RANGE

    def __post_init__(self) -> None:
        """Reject schemes that cannot be represented, at construction time.

        The two enums are also coerced from their string values here. Every
        consumer tests them by identity (``scheme.symmetry is
        Symmetry.SYMMETRIC``), and a plain ``"sym"`` compares equal to the enum
        member without being it, so a scheme built from a string would have
        quantized on the unsigned grid while reporting itself as symmetric.
        Coercing once at construction makes every construction site safe;
        a value that is neither an enum member nor one of its strings is
        refused by the enum itself.
        """
        object.__setattr__(self, "symmetry", Symmetry(self.symmetry))
        object.__setattr__(self, "scale_init", ScaleInit(self.scale_init))
        if self.bits not in SUPPORTED_BITS:
            supported = ", ".join(str(b) for b in sorted(SUPPORTED_BITS))
            msg = f"bits must be one of {{{supported}}}, got {self.bits}"
            raise SchemeError(msg)
        if self.group_size != -1 and self.group_size <= 0:
            msg = f"group_size must be positive or -1 (per-channel), got {self.group_size}"
            raise SchemeError(msg)
        if self.scale_init is ScaleInit.SEARCHED and self.symmetry is not Symmetry.SYMMETRIC:
            msg = (
                "the searched scale initialization is defined for symmetric "
                "schemes only; the reference implementation refuses the "
                "asymmetric combination as well"
            )
            raise SchemeError(msg)

    @property
    def is_per_channel(self) -> bool:
        """Whether one scale covers an entire output channel."""
        return self.group_size == -1

    @property
    def packs_evenly(self) -> bool:
        """Whether codes tile a 32-bit word without straddling boundaries.

        False selects the cross-word bitstream packing layout described in
        DOCUMENTATION.md section 6.1.
        """
        return 32 % self.bits == 0

    @property
    def code_range(self) -> tuple[int, int]:
        """Inclusive ``(min, max)`` of the integer code, before storage offset.

        Symmetric schemes use the two's-complement range, which is asymmetric by
        one code. That asymmetry costs nothing at the group's dominant extreme:
        the signed scale of DOCUMENTATION.md section 5.2.1 (MEMORY.md D-009,
        which closed Q-006) puts that extreme on the code with the larger
        magnitude, where it reconstructs exactly, and only the opposite
        extreme of a tied group can clip.
        """
        if self.symmetry is Symmetry.SYMMETRIC:
            half = 2 ** (self.bits - 1)
            return (-half, half - 1)
        return (0, 2**self.bits - 1)

    @property
    def coefficient_bounds(self) -> tuple[float, float]:
        """Permitted range for the clipping coefficients under this scheme.

        The range depends on the scale parameterization because the
        coefficients mean different things in each. Under
        :attr:`ScaleInit.OBSERVED_RANGE` they multiply the observed group
        extremes, so 1.0 means "no clipping" and values above it are
        meaningless. Under :attr:`ScaleInit.SEARCHED` they multiply the searched
        scale, so 1.0 means "accept the search result" and useful adjustments
        run in both directions. DOCUMENTATION.md section 5.3.

        The lower bound is a guard rather than a tuning choice: at exactly zero
        the group range collapses, the scale falls to epsilon, and every weight
        in the group saturates to a single code.
        """
        if self.scale_init is ScaleInit.SEARCHED:
            return (0.5, 1.5)
        return (0.1, 1.0)

    def groups_per_row(self, in_features: int) -> int:
        """How many scales one output channel needs for ``in_features`` inputs.

        Raises:
            ValueError: If ``in_features`` is not a whole number of groups.
        """
        if self.is_per_channel:
            return 1
        if in_features % self.group_size != 0:
            msg = f"in_features={in_features} is not divisible by group_size={self.group_size}"
            raise SchemeError(msg)
        return in_features // self.group_size

    def with_bits(self, bits: int) -> Self:
        """Return a copy at a different bit width, for mixed-precision search."""
        return dataclasses.replace(self, bits=bits)


@dataclasses.dataclass(frozen=True, slots=True)
class TuningConfig:
    """How the learned rounding is optimized.

    Defaults are the standard recipe from DOCUMENTATION.md section 5.4. The
    reference's own higher-quality recipe raises ``iters`` to 1000 and
    ``n_samples`` to 512; it is not offered as a preset here because MEMORY.md
    D-028 measured five times the steps on a fixed corpus as harmful, and no
    ledger row has yet measured that recipe on this implementation.

    Attributes:
        iters: Optimization steps per transformer block.
        batch_size: Calibration sequences per step.
        n_samples: Calibration sequences in total.
        seq_len: Tokens per calibration sequence.
        lr: Initial learning rate. ``None`` derives it from ``iters`` so that a
            parameter's total possible excursion is independent of the step
            count. See :meth:`resolved_lr`.
        minmax_lr: Separate learning rate for the clipping coefficients. The
            reference accepts one; no tuning loop here consumes it yet, so
            any value other than ``None`` is refused rather than silently
            ignored. The coefficients share ``lr``.
        gradient_accumulate_steps: Micro-batches accumulated before a step.
            Only ``1`` is implemented; anything else is refused.
        suppress_outliers: Exclude the largest errors from the loss. ``None``
            enables it only below 4 bits, which is the condition under which it
            was introduced.
    """

    iters: int = 200
    batch_size: int = 8
    n_samples: int = 128
    seq_len: int = 2048
    lr: float | None = None
    minmax_lr: float | None = None
    gradient_accumulate_steps: int = 1
    suppress_outliers: bool | None = None

    def __post_init__(self) -> None:
        """Reject configurations that cannot produce a valid run."""
        if self.iters <= 0:
            msg = f"iters must be positive, got {self.iters}"
            raise SchemeError(msg)
        if self.batch_size <= 0:
            msg = f"batch_size must be positive, got {self.batch_size}"
            raise SchemeError(msg)
        if self.n_samples < self.batch_size:
            msg = f"n_samples={self.n_samples} is smaller than batch_size={self.batch_size}"
            raise SchemeError(msg)
        # Refused rather than accepted and ignored. Both knobs were documented
        # and validated for months while no tuning loop read them, which is
        # the silent substitution the rest of this package refuses.
        if self.minmax_lr is not None:
            msg = (
                "minmax_lr is not implemented: the clipping coefficients share lr in "
                "every tuning loop. Pass None, or set lr for all parameters at once"
            )
            raise SchemeError(msg)
        if self.gradient_accumulate_steps != 1:
            msg = (
                "gradient_accumulate_steps other than 1 is not implemented, got "
                f"{self.gradient_accumulate_steps}"
            )
            raise SchemeError(msg)

    def resolved_lr(self, bits: int) -> float:
        """Learning rate for ``bits``, honoring an explicit override.

        The derived rate is ``c / iters`` with ``c = 2.0`` below 4 bits and
        ``1.0`` at or above. Since every step moves a parameter by exactly the
        current rate and the rate decays linearly to zero, total excursion is
        about ``c / 2``, which for 4 bits and above is exactly the half-code
        bound on the rounding perturbation. DOCUMENTATION.md section 5.4.
        """
        if self.lr is not None:
            return self.lr
        return (2.0 if bits < LOW_BIT_THRESHOLD else 1.0) / self.iters

    def resolved_minmax_lr(self, bits: int) -> float:
        """Clipping-coefficient learning rate, defaulting to the main rate."""
        if self.minmax_lr is not None:
            return self.minmax_lr
        return self.resolved_lr(bits)

    def resolved_suppress_outliers(self, bits: int) -> bool:
        """Whether outlier suppression applies at ``bits``.

        Defaults to enabled below 4 bits only. DOCUMENTATION.md section 5.5.
        """
        if self.suppress_outliers is not None:
            return self.suppress_outliers
        return bits < LOW_BIT_THRESHOLD
