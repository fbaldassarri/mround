# Copyright 2026 The MRound Authors
# SPDX-License-Identifier: Apache-2.0
"""Zero-shot task evaluation.

Perplexity is sensitive and cheap but it is not what anyone uses a model for.
Task accuracy is the check that a low-bit model is still useful rather than
merely still fluent, and the two can disagree: a quantization that barely moves
perplexity can still lose several points of task accuracy.

Status: Phase 0. Not implemented.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mlx import nn

__all__ = ["DEFAULT_TASKS", "TaskResult", "ZeroShotResult", "evaluate_zeroshot"]

# A small suite chosen to run in reasonable time while covering distinct
# capabilities: commonsense, science, and reading comprehension.
DEFAULT_TASKS: tuple[str, ...] = (
    "arc_easy",
    "arc_challenge",
    "hellaswag",
    "piqa",
    "winogrande",
)


@dataclasses.dataclass(frozen=True, slots=True)
class TaskResult:
    """One task's outcome.

    Attributes:
        task: Task identifier.
        accuracy: Fraction correct.
        stderr: Standard error of the accuracy estimate. Reported because
            quantization differences are often within it, and comparing two
            numbers whose error bars overlap is not evidence of anything.
        n_examples: Examples scored.
    """

    task: str
    accuracy: float
    stderr: float
    n_examples: int


@dataclasses.dataclass(frozen=True, slots=True)
class ZeroShotResult:
    """Results across the suite.

    Attributes:
        tasks: Per-task outcomes.
        mean_accuracy: Unweighted mean across tasks. A summary for tracking
            trends, not a substitute for the per-task numbers.
    """

    tasks: list[TaskResult]
    mean_accuracy: float


def evaluate_zeroshot(
    model: nn.Module,
    tokenizer: object,
    *,
    tasks: tuple[str, ...] = DEFAULT_TASKS,
    limit: int | None = None,
) -> ZeroShotResult:
    """Evaluate a model on zero-shot tasks.

    Args:
        model: A loaded MLX model, quantized or not.
        tokenizer: Its tokenizer.
        tasks: Task identifiers.
        limit: Examples per task, for quick checks. ``None`` uses the full set.

    Returns:
        Per-task accuracies and their mean.
    """
    raise NotImplementedError
