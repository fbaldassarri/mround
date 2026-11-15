# Copyright 2026 The MRound Authors
# SPDX-License-Identifier: Apache-2.0
"""How long a quantization run will take, worked out before it starts.

The memory rule says whether a run fits (:mod:`mround.planner.memory`); this
one says how long it will hold the Mac, which for a 7B is most of a day
(ROAD-TO-1.0.md, M2). It predicts the seconds a run records from the model's
configuration and the run's settings alone, and it is a fit to the runs of one
machine, the M5 MacBook Pro with 32 GB on which every MRound run in the ledger
was made. On any other Mac it says what that machine would take, which is a
reference point and not a prediction for the Mac it runs on.

**What a run records, and so what the rule predicts.** A run times its block
loop, from the capture of the first block's inputs to the end of the last
block, and a mixed run its scoring pass before the loop
(``result.total_seconds`` and ``mixed.scoring_seconds`` in
``mround_run.json``). The download, the load and the export after the loop
were not timed, and are not predicted; since 2026-11-13 the export and the
remainder are timed, so a later fit can add them. Inside the loop each block's
progress line gives the seconds of its own tuning: the scale search, the
losses at both ends, the steps and the commit. Outside that timer are the
entrance and, for every block, the passes that make its targets and advance
the stream.

**The law.** At the standard recipe (200 steps of 8 sequences of 2,048
tokens, 128 calibration sequences) a block's tuning without its search takes
26.9 s times W to the power 0.627 on that Mac, W being the block's weights in
millions: 59.4 s at SmolLM2-135M, 146.4 at Qwen2.5-0.5B, 299.9 at 1.5B and
410.0 at 3B. Time grows more slowly than the work, as a small block keeps the
GPU less busy; the exponent was fitted, not derived. On top of it:

- the outlier rule (on below 4 bits unless a run turns it off, DOCUMENTATION.md
  5.5) costs 5.8 percent;
- pieces (``TuningConfig.micro_batches``) multiply it by 1.893, and by 2.513
  with the outlier rule, both measured on Qwen2.5-7B in four pieces, the one
  model that needs them on this Mac;
- the scale search takes 0.307 s per million block weights;
- around the blocks, the entrance and the targets and advance passes take 166
  ns per block weight, for every block;
- a mixed run's scoring pass takes 989 s per billion weights of the model at
  the default scoring budget of 128 sequences of 1,024 tokens, and 1.42 times
  that with a backward at each width (``own_width``).

The losses at both ends of a block's tuning and its commit cost about ten
steps, so a recipe of other length scales the tuning by its steps plus ten
over 210. Another batch scales the steps by its tokens, and another
calibration set the passes by its tokens. No run measured either: the scaling
is an assumption the next such run checks.

**What it rests on, and how far to trust it.** Fifty five runs at the
standard recipe on the trees since 2026-10-12: SmolLM2-135M at mixed 2.5,
Qwen2.5-0.5B, 1.5B and 3B at 2 and 4 bits and at mixed 2.5, and Qwen2.5-7B at
2 and 4 bits in four pieces. Their recorded seconds, the scoring pass
included, took 0.914 to 1.346 times the rule, 1.018 on the median, the tenth
and ninetieth percentiles 0.956 and 1.141, and 49 of the 55 within 15 percent.
The six above it are slow stretches of the machine, not of the work:
Qwen2.5-0.5B's 4 bit run of 2026-11-12 wrote its September checkpoint to the
byte, block for block the same work, and took 4,465 s where September's took
3,606. So the rule is the pace of the machine undisturbed, and a run can take
a third longer when the Mac slows for hours.
The constants are fitted on the logs' block lines (the 25th percentile of
each run's blocks, to read the undisturbed machine) and on the run records,
and ``tests/unit/test_time_rule.py`` holds the rule to every run.

**Outside the fit, labelled as such.** A whole 7B step was never measured,
since it does not fit the Mac, so the pieces' cost is one factor fitted
against the law's extrapolation to 7B, and a smaller model in pieces is
predicted high: Qwen2.5-1.5B in eight pieces, on the tree of 2026-10-11, took
0.74 times the rule. Other architectures, other machines and other recipes
are the law extended, and every run records the prediction beside what it
measured, so each one checks it. This module imports nothing from MLX.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

from mround.planner.sensitivity import (
    SENSITIVITY_SAMPLES,
    SENSITIVITY_SEQ_LEN,
    gradient_source_for,
)

if TYPE_CHECKING:
    from collections.abc import Collection

    from mround.planner.memory import ModelShape, RunShape
    from mround.schemes import TuningConfig

__all__ = [
    "MACHINE",
    "RULE_VERSION",
    "SPREAD",
    "TimePrediction",
    "predict_run_time",
    "predict_time",
]

# Which fit this is, recorded with every prediction, as the memory rule's is.
RULE_VERSION = "2026-11-13"

# The one machine every run behind the fit was made on.
MACHINE = "M5 MacBook Pro with 32 GB"

# How the fifty five runs' recorded seconds fell against the rule, lowest and
# highest: the band a reader should put around a prediction on that machine.
SPREAD = (0.914, 1.346)

# The fitted constants, in seconds. The configurations behind them are named in
# the module docstring and in tests/unit/test_time_rule.py.
_BLOCK = 26.9  # a block's tuning at the standard recipe, per million weights to _EXPONENT
_EXPONENT = 0.627
_PASSES = 10.0  # the losses at both ends and the commit, in steps
_OUTLIER_RULE = 1.058  # the outlier suppressed loss, steps whole
_PIECES = 1.893  # a step in pieces, against the whole step the law extrapolates
_PIECES_OUTLIER_RULE = 2.513  # and with the outlier rule
_SEARCH = 0.307e-6  # the scale search, per block weight
_BETWEEN = 166e-9  # the entrance, targets and advance, per block weight and block
_SCORING = 989e-9  # the mixed scoring pass, per model weight, widest gradients
_OWN_WIDTH = 1.42  # and with a backward at each width

# The standard recipe the constants were measured at.
_STEPS = 200
_STEP_TOKENS = 8 * 2048
_CALIBRATION_TOKENS = 128 * 2048
_SCORING_TOKENS = SENSITIVITY_SAMPLES * SENSITIVITY_SEQ_LEN

_MINUTE = 60
_HOUR = 60 * _MINUTE


def _duration(seconds: float) -> str:
    """A span as a person reads it: 4h28m, 7m14s, 45s."""
    # Anything that rounds to an hour or more is said in hours and minutes.
    if seconds >= _HOUR - _MINUTE / 2:
        minutes = round(seconds / _MINUTE)
        return f"{minutes // _MINUTE}h{minutes % _MINUTE:02d}m"
    whole = round(seconds)
    if whole >= _MINUTE:
        return f"{whole // _MINUTE}m{whole % _MINUTE:02d}s"
    return f"{whole}s"


@dataclasses.dataclass(frozen=True, slots=True)
class TimePrediction:
    """The seconds a run is predicted to record, on the machine the rule was fitted on.

    Attributes:
        stages: Seconds by stage: ``scoring``, a mixed run's scoring pass;
            ``blocks``, every block's own tuning as its progress line reports
            it, the search included; and ``between_blocks``, the entrance and
            the passes that make each block's targets and advance the stream.
        block: Seconds one block's line is predicted to report.
        blocks: How many blocks the loop runs.
        iters: Steps a block takes.
        outlier_rule: Whether the steps run the outlier suppressed loss.
        pieces: Pieces each step runs in.
        searched: Whether each block searches its scales.
        scoring_source: ``"widest"`` or ``"own_width"`` for a mixed run's
            scoring pass, ``None`` for a run without one.
        scoring_tokens: Tokens the scoring pass scores, ``None`` without one.
    """

    stages: dict[str, float]
    block: float
    blocks: int
    iters: int
    outlier_rule: bool
    pieces: int
    searched: bool
    scoring_source: str | None = None
    scoring_tokens: int | None = None

    @property
    def total(self) -> float:
        """Every stage, which is what the run records as its seconds."""
        return sum(self.stages.values())

    def describe(self) -> str:
        """One line: the prediction, the machine it is for, and its band there."""
        scoring = self.stages.get("scoring", 0.0)
        parts = f"the scoring pass {_duration(scoring)} and " if scoring else ""
        return (
            f"time: predicted {_duration(self.total)} on the {MACHINE} the rule was "
            f"fitted on ({parts}{self.blocks} blocks of {_duration(self.block)}), where "
            f"runs took {SPREAD[0]:.2f} to {SPREAD[1]:.2f} times the rule"
        )

    def record(self) -> dict[str, Any]:
        """Plain data for the run record, beside what the run measures."""
        return {
            "rule": RULE_VERSION,
            "machine": MACHINE,
            "seconds": round(self.total, 1),
            "by_stage_seconds": {name: round(value, 1) for name, value in self.stages.items()},
            "block_seconds": round(self.block, 1),
            "blocks": self.blocks,
            "iters": self.iters,
            "outlier_rule": self.outlier_rule,
            "pieces": self.pieces,
            "searched": self.searched,
            "scoring_source": self.scoring_source,
            "scoring_tokens": self.scoring_tokens,
            "spread": list(SPREAD),
        }


def predict_time(
    model: ModelShape,
    run: RunShape,
    *,
    iters: int,
    outlier_rule: bool,
    scoring_tokens: int | None = None,
    scoring_source: str | None = None,
) -> TimePrediction:
    """Predict the seconds a run records, on the machine the rule was fitted on.

    Args:
        model: The model's shape, as the memory rule reads it.
        run: The run's settings, as the memory rule reads them; a mixed run
            (``run.mixed``) is predicted with its scoring pass.
        iters: Steps each block takes.
        outlier_rule: Whether the steps run the outlier suppressed loss.
        scoring_tokens: Tokens the scoring pass scores, ``None`` for the
            default budget.
        scoring_source: ``"own_width"`` for a backward at each width, anything
            else for the widest width's alone.

    Returns:
        The prediction, stage by stage.

    Raises:
        ValueError: If ``iters`` is not a positive number of steps.
    """
    if iters <= 0:
        msg = f"a run takes a positive number of steps, got {iters}"
        raise ValueError(msg)
    weights = model.block_weights
    calibration = run.samples * run.seq_len / _CALIBRATION_TOKENS
    passes = _PASSES * calibration
    pieces = run.micro_batches > 1
    if pieces:
        factor = _PIECES_OUTLIER_RULE if outlier_rule else _PIECES
    else:
        factor = _OUTLIER_RULE if outlier_rule else 1.0
    tuning = (
        _BLOCK
        * (weights / 1e6) ** _EXPONENT
        * (iters + passes)
        / (_STEPS + _PASSES)
        * (run.batch_size * run.seq_len / _STEP_TOKENS)
        * factor
    )
    block = tuning + (_SEARCH * weights if run.searched else 0.0)
    stages: dict[str, float] = {}
    tokens = None
    source = None
    if run.mixed:
        tokens = _SCORING_TOKENS if scoring_tokens is None else int(scoring_tokens)
        source = "own_width" if scoring_source == "own_width" else "widest"
        stages["scoring"] = (
            _SCORING
            * model.model_weights
            * tokens
            / _SCORING_TOKENS
            * (_OWN_WIDTH if source == "own_width" else 1.0)
        )
    stages["blocks"] = block * model.layers
    stages["between_blocks"] = _BETWEEN * weights * model.layers * calibration
    return TimePrediction(
        stages=stages,
        block=block,
        blocks=model.layers,
        iters=iters,
        outlier_rule=outlier_rule,
        pieces=run.micro_batches,
        searched=run.searched,
        scoring_source=source,
        scoring_tokens=tokens,
    )


def predict_run_time(
    model: ModelShape,
    run: RunShape,
    tuning: TuningConfig,
    bits: int,
    *,
    candidate_bits: Collection[int] = (),
    per_layer_recipe: bool = False,
    scoring_samples: int | None = None,
    scoring_seq_len: int | None = None,
    gradient_source: str = "auto",
) -> TimePrediction:
    """:func:`predict_time` with the run's settings resolved as the run resolves them.

    The outlier rule from the base width, as the block loop resolves it, or
    from the narrowest candidate width when every layer takes its own recipe,
    which assumes the rule in every block and errs long; the scoring budget's
    defaults; and the scoring gradients by size, as :func:`mround.api.quantize`
    takes them (D-053).

    Args:
        model: The model's shape.
        run: The run's settings.
        tuning: The tuning recipe.
        bits: The base width, the scheme's own.
        candidate_bits: A mixed run's candidate widths.
        per_layer_recipe: As :func:`mround.api.quantize` takes it.
        scoring_samples: As :func:`mround.api.quantize` takes it.
        scoring_seq_len: As :func:`mround.api.quantize` takes it.
        gradient_source: As :func:`mround.api.quantize` takes it.

    Returns:
        The prediction.

    Raises:
        SchemeError: If ``gradient_source`` names no source a mixed run takes.
    """
    widths = sorted(int(width) for width in candidate_bits)
    base = widths[0] if per_layer_recipe and widths else bits
    tokens = None
    source = None
    if run.mixed:
        tokens = (SENSITIVITY_SAMPLES if scoring_samples is None else int(scoring_samples)) * (
            SENSITIVITY_SEQ_LEN if scoring_seq_len is None else int(scoring_seq_len)
        )
        source = gradient_source_for(gradient_source, model.layers * model.block_weights)
    return predict_time(
        model,
        run,
        iters=tuning.iters,
        outlier_rule=tuning.resolved_suppress_outliers(base),
        scoring_tokens=tokens,
        scoring_source=source,
    )
