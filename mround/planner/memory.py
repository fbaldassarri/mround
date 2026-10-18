# Copyright 2026 The MRound Authors
# SPDX-License-Identifier: Apache-2.0
"""How much memory a quantization run will reach, worked out before it starts.

A run used to learn its peak by reaching it. On Apple Silicon an allocation
beyond the machine does not fail cleanly, it swaps, and a run that will not fit
finds out at the stage that does not fit, after its download and part of its
hours. This module predicts the peak from the model's configuration and the
run's settings alone, so that :func:`mround.api.quantize` can refuse such a run
before it loads anything and say what would fit (ROAD-TO-1.0.md, M5).

**What the run holds is two kinds of memory, and only one is fitted.** Between
its stages a block of the loop holds what accounting gives exactly: two float32
activation streams over the whole calibration set, the embedding at the
model's width, and the codes of every finished block at one byte a weight with
a float32 scale a group (and a zero point a group when asymmetric), beside the
calibration tokens themselves. On every run recorded since M1 those terms
reproduce the entrance to a few bytes and the growth from one block to the
next to the megabyte, in 1.5B, 3B and 7B runs alike. On top of that each stage
holds a working set of its own, and that part is fitted, in bytes per weight of
the block (W) and per token of a batch or of a piece of it, times the MLP's
width (i) or the hidden width (h):

- ``targets``, ``advance``: the block's own weights at the model's width and
  one batch's forward pass without a gradient, 8.5 bytes per token and unit of
  i and 38 per token and unit of h. ``advance`` also holds the block's codes.
  Counting the MLP's width alone, as the first fit did, was right on Qwen2.5,
  whose MLP is 5.3 to 5.8 times its hidden width, and up to 24 percent short on
  the Llama layout, at 2.7 to 3.5.
- ``search``: the tunables (the block's float32 weight and rounding
  perturbation beside its own weights), the importance pass and the scale
  search's slices, at most 0.5 GB above the pass.
- ``score``: the losses at both ends of a block's tuning, one batch at a time:
  the tunables, 12.9 bytes a weight of fake quantized layers, a batch's forward
  (or a piece's) at 7.7 bytes per token and unit of i, its loss over the batch
  at 30.4 per token and unit of h, and 0.47 GB whatever the shape.
- ``tune``: the steps. About 30 bytes a weight, the tunables included; 1.39
  float32 tensors of the attention scores (heads by sequence by sequence, for
  every sequence in a piece) held throughout; and at the step's peak the larger
  of the MLP's tensors, 38.4 bytes per token and unit of i, and 2.8 more score
  sized tensors. Pieces (``TuningConfig.micro_batches``) add about nine copies
  of one batch's hidden state, which the outlier threshold's pass over the
  whole batch holds.
- ``commit``: the block's codes being made, about 25 bytes a weight and 0.1 GB.

**The step's peak is the larger of two terms, not their sum.** A step's
activations are part MLP, growing with the tokens of a piece, and part
attention, whose backward holds score sized tensors growing with the square of
the sequence length; at 2,048 tokens on Qwen2.5 the two grow together and no
run can tell them apart. ``scripts/probe_memory.py`` held a step's tokens at
16,384 and moved the sequence length from 512 to 4,096 (log0169): the step
grew by 0.54, 1.72 and 6.46 GB at each doubling, faster than any sum of the
two allows and as the larger of them does. The first fit summed them, with
the MLP's share assumed, and was 22 percent short at 512 tokens.

**What the fit rests on, and which way it errs.** Sixteen configurations: the
five runs on the current tree (1.5B and 7B in eight pieces, 3B whole twice and
at mixed 2.5) and the eleven that ``scripts/probe_memory.py`` measured on
Metal, a 7B whole and in two, four and eight pieces, SmolLM2-135M, Qwen2.5-0.5B
and Llama 3 8B, and the four sequence lengths. Where the probe and a run meet,
its two blocks hold what the run's last block held above the accounting to the
megabyte, in eleven of the twelve stages the two share, which is what lets it
stand in for one. The step was fitted with an under prediction weighted three
times an over prediction, so the rule errs toward refusing: every peak is
predicted between 1.0 percent short and 8.2 percent over.

**The mixed precision scoring pass** reads every weight of the model and holds
one float32 quantized copy of the block weights, the searched grids of every
candidate width, a block's gradients, the logits of one scoring batch and one
block's activations under checkpointing (MEMORY.md D-036, M1c). Its one
measurement on the current tree is a 3B at the matched budget, 23.95 GB
(log0160), on which the logits' share is fitted.

**After the loop, measured once.** The tensors outside the blocks are rounded
one at a time in float32, unless the run leaves them dense, and the checkpoint
is packed a layer at a time and written with every block's codes still held.
The first run to measure either stage, Qwen2.5-7B-Instruct at 2 bits in four
pieces (log0170), found both doing more than that: its export packed every
layer at once and reached 35.95 GB, past the Mac it ran on, and its remainder
rounded the embedding and the head only to set them aside. Both are fixed
(``mround/formats/mlx_export.py``, ``mround/api.py``). That discarded rounding
is the one measurement of what rounding holds: 15.9 GB above the codes and the
tensors, for a tensor of 545 million weights at 2 bits with its scale searched.
On MLX's CPU backend the same rounding holds 7.4 float32 copies of the tensor
without the search and 1.46 GB more for the search's slices, which the rule
takes, so that it errs high by a tenth on that run; a dense remainder holds
nothing. The export holds the packed checkpoint beside the layer being packed,
at eight bytes a weight, which a test measures on the CPU backend. The next run
measures both stages as they are now.

The rule is a fit to the runs it names and nothing more. Every run records its
prediction beside its measured stages, so the next measurement checks it, and
``scripts/probe_memory.py`` measures the stages of synthetic blocks of any
shape on the Mac without downloading a model. This module imports nothing from
MLX: it reads shapes and settings, which is what lets it run before anything is
loaded, and on a machine that cannot run MLX at all.
"""

from __future__ import annotations

import dataclasses
import math
from typing import TYPE_CHECKING, Any

from mround.planner.sensitivity import SENSITIVITY_SAMPLES, SENSITIVITY_SEQ_LEN
from mround.planner.sensitivity import scoring_batch_size as derived_scoring_batch
from mround.schemes import ScaleInit, Symmetry

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping, Sequence

    from mround.schemes import QuantScheme, TuningConfig

__all__ = [
    "BUDGET_SHARE",
    "RULE_VERSION",
    "MemoryCheck",
    "MemoryPrediction",
    "ModelShape",
    "RunShape",
    "check_memory",
    "default_budget",
    "fitting_micro_batches",
    "predict_memory",
]

# Which fit this is. Recorded with every prediction, so that a run record says
# which rule its prediction came from when the constants below are refitted.
# "2026-10-17" was fitted on the run records alone, "2026-10-17b" on them and
# the probe of log0169, and "2026-10-18" holds the stages after the loop to the
# first run that measured them, log0170.
RULE_VERSION = "2026-10-18"

# The share of the Mac's memory a run may be predicted to reach. The largest
# peak a run on the 32 GB Mac reached and finished is 30.11 GB, 88 percent of
# its 34.36 GB (log0151, a two step probe of a 3B mixed run); the run seen to
# stall there asked for 52 GB (log0059). The process's own resident set (1.4 to
# 6.5 GB in the runs recorded there) and the system share the same memory, and
# the rule is a fit, so the default stops short of the largest that finished:
# 29.2 GB on that Mac. A caller who knows better passes a budget of their own.
BUDGET_SHARE = 0.85

# The fitted constants. Each is in bytes; "per token and unit of width"
# multiplies the tokens a pass holds by the MLP's intermediate size, or by the
# hidden size where it says so. The configurations behind them are named in the
# module docstring and in tests/unit/test_memory_rule.py, which holds the rule
# to every one.
_FORWARD_MLP = 8.5  # a forward pass without gradient, per token and unit of MLP width
_FORWARD_HIDDEN = 38.0  # and per token and unit of hidden width
_SEARCH = 0.5e9  # the scale search and the importance above the pass, at most
_SCORE_WEIGHT = 12.9  # the fake quantized layers of a measurement, beyond the tunables
_SCORE_MLP = 7.7  # a measurement's forward, per token and unit of MLP width
_SCORE_HIDDEN = 30.4  # its loss over the batch, per token and unit of hidden width
_SCORE_FIXED = 0.47e9  # what a measurement holds whatever the shape
_STEP_WEIGHT = 20.3  # a step's weight sized tensors beyond the tunables, per weight
_STEP_SCORES = 1.39  # score sized float32 tensors held through a step, per sequence
_STEP_MLP = 38.4  # the MLP's tensors at a step's peak, per token and unit of MLP width
_STEP_ATTENTION = 2.8  # or attention's, score sized float32 tensors per sequence
_PIECE_COPIES = 8.8  # copies of one batch's hidden state when a step runs in pieces
_COMMIT_WEIGHT = 15.0  # the codes being made, per weight, beyond the tunables
_COMMIT_FIXED = 0.1e9  # and what a commit holds whatever the shape
_TUNABLE_WEIGHT = 8.0  # the float32 weight and the rounding perturbation, per weight
_SCORING_LOGITS = 2.40  # logits sized float32 tensors in the scoring backward
_SCORING_GRADIENT = 16.0  # one block's gradients and rebuilt perturbations, per weight
_SCORING_FIXED = 0.89e9  # what the GPU held in flight whatever the depth (log0148)
_REMAINDER_COPIES = 7.4  # float32 copies of the tensor outside the blocks being rounded
_REMAINDER_SEARCH = 1.46e9  # and the scale search's slices over it, when it is searched
_EXPORT_LAYER = 8.0  # bytes a weight of the layer the export is packing

# Architectures whose MLP is two matrices rather than three. Everything else
# is read as the gated layout of Llama, Qwen2, Mistral and Gemma, which counts
# more weights when it is wrong, never fewer.
_UNGATED = frozenset({"phi", "gpt_neox", "starcoder2", "gpt2", "gpt_bigcode", "falcon", "opt"})

_BYTES_BY_DTYPE = {"bfloat16": 2, "float16": 2, "float32": 4}

_GB = 1e9


def _gb(value: float) -> str:
    return f"{value / _GB:.1f} GB"


@dataclasses.dataclass(frozen=True, slots=True)
class ModelShape:
    """What the rule needs to know about a model, from its configuration.

    Attributes:
        hidden: The residual stream's width.
        intermediate: The MLP's inner width.
        layers: Transformer blocks.
        heads: Attention heads.
        kv_heads: Key and value heads.
        head_dim: Width of one head.
        vocab: Vocabulary size.
        tied: Whether the output head is the embedding.
        dtype_bytes: Bytes a weight takes at the model's own width.
        gated: Whether the MLP has a gate projection beside its up projection.
    """

    hidden: int
    intermediate: int
    layers: int
    heads: int
    kv_heads: int
    head_dim: int
    vocab: int
    tied: bool = True
    dtype_bytes: int = 2
    gated: bool = True

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> ModelShape | None:
        """Read the shape of a dense decoder from its Hugging Face configuration.

        A multimodal configuration keeps its language model under
        ``text_config``, which is read when the top level does not say. A
        mixture of experts is not a dense decoder, and MRound refuses one for
        now (Q-010), so it gets no shape.

        Returns:
            The shape, or ``None`` when the configuration does not describe a
            dense decoder this rule can read, in which case no prediction is
            made rather than a wrong one.
        """
        settings: Mapping[str, Any] = config
        if "hidden_size" not in settings and isinstance(config.get("text_config"), dict):
            settings = config["text_config"]
        if any(
            settings.get(key) for key in ("num_local_experts", "num_experts", "n_routed_experts")
        ):
            return None
        try:
            hidden = int(settings["hidden_size"])
            intermediate = int(settings["intermediate_size"])
            layers = int(settings["num_hidden_layers"])
            heads = int(settings["num_attention_heads"])
            vocab = int(settings["vocab_size"])
            kv_heads = int(settings.get("num_key_value_heads") or heads)
            head_dim = int(settings.get("head_dim") or hidden // heads)
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            return None
        if min(hidden, intermediate, layers, heads, vocab, kv_heads, head_dim) <= 0:
            return None
        dtype = str(settings.get("torch_dtype") or settings.get("dtype") or "bfloat16")
        model_type = str(settings.get("model_type") or config.get("model_type") or "")
        return cls(
            hidden=hidden,
            intermediate=intermediate,
            layers=layers,
            heads=heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
            vocab=vocab,
            # Untied unless the configuration says otherwise: an untied head
            # is one more matrix to hold, so the guess errs high.
            tied=bool(
                config.get("tie_word_embeddings", settings.get("tie_word_embeddings", False))
            ),
            dtype_bytes=_BYTES_BY_DTYPE.get(dtype.removeprefix("torch."), 2),
            gated=model_type not in _UNGATED,
        )

    @property
    def block_weights(self) -> int:
        """Weights in the linear layers of one block, which is what is tuned."""
        attention = 2 * self.hidden * self.heads * self.head_dim
        keys_values = 2 * self.hidden * self.kv_heads * self.head_dim
        mlp = (3 if self.gated else 2) * self.hidden * self.intermediate
        return attention + keys_values + mlp

    @property
    def embedding_weights(self) -> int:
        """Weights in the embedding, which the block loop holds throughout."""
        return self.vocab * self.hidden

    @property
    def outside_weights(self) -> int:
        """Weights outside the blocks: the embedding, and an untied head."""
        return self.embedding_weights * (1 if self.tied else 2)

    @property
    def largest_layer_weights(self) -> int:
        """Weights of a block's largest linear layer, an MLP projection as a rule."""
        return max(self.hidden * self.intermediate, self.hidden * self.heads * self.head_dim)

    @property
    def model_weights(self) -> int:
        """Every weight a forward pass reads: blocks, embedding and head."""
        return self.layers * self.block_weights + self.outside_weights


@dataclasses.dataclass(frozen=True, slots=True)
class RunShape:
    """What a run's settings decide about its memory.

    Attributes:
        samples: Calibration sequences, every one of which both streams hold.
        seq_len: Tokens per sequence.
        batch_size: Sequences per step.
        micro_batches: Pieces each step's batch runs in.
        group_size: Weights per scale, ``-1`` per channel.
        bits: The widest width the run assigns, which sizes the packed codes.
        asymmetric: Whether each group stores a zero point too.
        searched: Whether the scale search runs.
        mixed: Whether a mixed precision scoring pass runs first.
        candidate_widths: How many widths the scoring pass scores.
        scoring_sequences: Sequences per scoring batch.
        scoring_seq_len: Tokens per scoring sequence.
        dense_remainder: Whether the tensors outside the blocks stay at the
            model's width, so that nothing is rounded after the loop.
    """

    samples: int = 128
    seq_len: int = 2048
    batch_size: int = 8
    micro_batches: int = 1
    group_size: int = 64
    bits: int = 4
    asymmetric: bool = False
    searched: bool = True
    mixed: bool = False
    candidate_widths: int = 3
    scoring_sequences: int = 2
    scoring_seq_len: int = 1024
    dense_remainder: bool = False

    @classmethod
    def of(
        cls,
        scheme: QuantScheme,
        tuning: TuningConfig,
        *,
        candidate_bits: Sequence[int] | None = None,
        scoring_samples: int | None = None,
        scoring_seq_len: int | None = None,
        scoring_batch_size: int | None = None,
        dense_remainder: bool = False,
    ) -> RunShape:
        """The shape of a run with these settings, resolved as the run resolves them.

        Args:
            scheme: The run's resolved scheme.
            tuning: Its tuning recipe.
            candidate_bits: The candidate widths of a mixed run, ``None`` for
                a uniform one.
            scoring_samples: As :func:`mround.api.quantize` takes it.
            scoring_seq_len: As :func:`mround.api.quantize` takes it.
            scoring_batch_size: As :func:`mround.api.quantize` takes it, or
                ``None`` for the batch the scoring budget derives.
            dense_remainder: As :func:`mround.api.quantize` takes it.
        """
        widths = sorted({int(width) for width in candidate_bits or ()})
        samples = SENSITIVITY_SAMPLES if scoring_samples is None else int(scoring_samples)
        sequence = SENSITIVITY_SEQ_LEN if scoring_seq_len is None else int(scoring_seq_len)
        batch = (
            derived_scoring_batch(samples, sequence)
            if scoring_batch_size is None
            else max(1, min(samples, int(scoring_batch_size)))
        )
        return cls(
            samples=tuning.n_samples,
            seq_len=tuning.seq_len,
            batch_size=tuning.batch_size,
            micro_batches=tuning.micro_batches,
            group_size=scheme.group_size,
            bits=max([scheme.bits, *widths]),
            asymmetric=scheme.symmetry is Symmetry.ASYMMETRIC,
            searched=scheme.scale_init is ScaleInit.SEARCHED,
            mixed=bool(widths),
            candidate_widths=max(1, len(widths)),
            scoring_sequences=batch,
            scoring_seq_len=sequence,
            dense_remainder=dense_remainder,
        )

    def with_micro_batches(self, pieces: int) -> RunShape:
        """The same run with its steps in ``pieces`` pieces."""
        return dataclasses.replace(self, micro_batches=pieces)


@dataclasses.dataclass(frozen=True, slots=True)
class MemoryPrediction:
    """The peak a run is predicted to reach, stage by stage.

    Attributes:
        stages: Bytes each stage is predicted to reach, the whole process's
            MLX allocation as a run record counts it: ``entrance``,
            ``scoring`` on a mixed run, the block loop's stages at its last
            block, ``remainder`` (the tensors outside the blocks) and
            ``export``.
        parts: What the block loop holds between its stages at its last
            block, by accounting: ``streams``, ``embedding``, ``codes`` and
            ``tokens``, which add up to what every stage there holds beneath
            its own working set.
        model: The shape the prediction was made for.
        run: The settings it was made for.
    """

    stages: dict[str, int]
    parts: dict[str, int]
    model: ModelShape
    run: RunShape

    @property
    def peak(self) -> int:
        """The largest stage, which is the run's predicted peak."""
        return max(self.stages.values())

    @property
    def stage(self) -> str:
        """The stage that sets the peak."""
        return max(self.stages, key=self.stages.__getitem__)

    def restricted(self, names: Collection[str]) -> MemoryPrediction:
        """The same prediction over only the stages named, for a partial run."""
        return dataclasses.replace(
            self, stages={name: value for name, value in self.stages.items() if name in names}
        )

    def record(self) -> dict[str, Any]:
        """Plain data for the run record, beside what the run measures."""
        return {
            "rule": RULE_VERSION,
            "peak_bytes": self.peak,
            "stage": self.stage,
            "by_stage_bytes": dict(self.stages),
            "parts_bytes": dict(self.parts),
            "model": dataclasses.asdict(self.model),
            "run": dataclasses.asdict(self.run),
        }


def _codes_per_weight(run: RunShape, model: ModelShape) -> float:
    """Bytes a finished block keeps per weight until the export."""
    # Per channel, one scale a row: bounded here by the narrowest input a
    # block's layers have, which is the hidden width.
    group = run.group_size if run.group_size > 0 else model.hidden
    return 1.0 + (8.0 if run.asymmetric else 4.0) / group


def _forward(model: ModelShape, tokens: int, *, mlp: float, hidden: float) -> float:
    """A pass over ``tokens`` tokens, at so many bytes per unit of either width."""
    return tokens * (mlp * model.intermediate + hidden * model.hidden)


def _step_activations(model: ModelShape, sequences: int, seq_len: int) -> float:
    """What a step's backward over ``sequences`` sequences holds at its peak.

    The attention scores it keeps throughout, and the larger of the MLP's
    tensors and attention's own, since the two are not live at once.
    """
    scores = sequences * model.heads * seq_len * seq_len * 4
    mlp = _STEP_MLP * sequences * seq_len * model.intermediate
    return _STEP_SCORES * scores + max(mlp, _STEP_ATTENTION * scores)


def predict_memory(model: ModelShape, run: RunShape) -> MemoryPrediction:
    """Predict the peak of every stage of a quantization run.

    The block loop's stages are taken at its last block, where the codes of
    every block before it are held and every stage is at its largest.

    Args:
        model: The model's shape.
        run: The run's settings.

    Returns:
        The prediction, stage by stage.
    """
    weights = model.block_weights
    width = model.dtype_bytes
    tokens = run.samples * run.seq_len
    batch_tokens = run.batch_size * run.seq_len
    piece_sequences = math.ceil(run.batch_size / run.micro_batches)
    pieces = run.micro_batches > 1
    codes = _codes_per_weight(run, model) * weights
    embedding = model.embedding_weights * width

    # What the loop holds between stages at its last block, by accounting:
    # the two streams, the embedding, the finished blocks' codes, and the
    # calibration set's own tokens, four bytes each.
    streams = 2 * tokens * model.hidden * 4
    finished = (model.layers - 1) * codes
    resident = streams + embedding + finished + 4 * tokens

    forward = _forward(model, batch_tokens, mlp=_FORWARD_MLP, hidden=_FORWARD_HIDDEN)
    tunables = (width + _TUNABLE_WEIGHT) * weights
    batch_copies = _PIECE_COPIES * batch_tokens * model.hidden * 4 if pieces else 0.0
    # A measurement runs a batch's forward a piece at a time when the step does,
    # and takes the loss over the whole batch either way.
    forward_tokens = piece_sequences * run.seq_len if pieces else batch_tokens
    measured = (
        _SCORE_MLP * forward_tokens * model.intermediate
        + _SCORE_HIDDEN * batch_tokens * model.hidden
        + _SCORE_FIXED
    )

    # After the loop: every code and the tensors outside the blocks at the
    # model's width, and then what the stage itself holds. The remainder
    # rounds the tensors outside the blocks one at a time, holding the
    # rounding of the largest beside the codes of any rounded before it, and
    # nothing when the run leaves them dense. The export holds the packed
    # checkpoint (the packed codes, and a scale and a bias a group at the
    # model's width) beside the layer it is packing.
    all_codes = model.layers * codes
    outside = model.outside_weights * width
    group = run.group_size if run.group_size > 0 else model.hidden
    per_weight = _codes_per_weight(run, model)
    packed_per_weight = run.bits / 8 + 2 * width / group
    packed = model.layers * weights * packed_per_weight
    rounded = 0 if run.dense_remainder else model.outside_weights
    rounding = 0.0
    if rounded:
        rounding = (
            _REMAINDER_COPIES * 4 * model.embedding_weights
            + (_REMAINDER_SEARCH if run.searched else 0.0)
            + (rounded - model.embedding_weights) * per_weight
        )
    packing = model.embedding_weights if rounded else model.largest_layer_weights

    stages: dict[str, float] = {
        # The reference stream in float32, the captured inputs at the model's
        # width, the embedding and the tokens: exact on every recorded run.
        "entrance": tokens * model.hidden * (4 + width) + embedding + 4 * tokens,
        "targets": resident + width * weights + forward,
        "search": resident + tunables + (forward + _SEARCH if run.searched else 0.0),
        "score": resident + tunables + _SCORE_WEIGHT * weights + measured,
        "tune": resident
        + tunables
        + _STEP_WEIGHT * weights
        + _step_activations(model, piece_sequences, run.seq_len)
        + batch_copies,
        "commit": resident + tunables + _COMMIT_WEIGHT * weights + _COMMIT_FIXED,
        "advance": resident + width * weights + codes + forward,
        "remainder": all_codes + outside + rounding,
        "export": all_codes
        + outside
        + rounded * (per_weight + packed_per_weight)
        + packed
        + _EXPORT_LAYER * max(packing, model.largest_layer_weights),
    }
    if run.mixed:
        scoring_tokens = run.scoring_sequences * (run.scoring_seq_len - 1)
        grids = (
            4 * run.candidate_widths * model.layers * weights / run.group_size
            if run.searched and run.group_size > 0
            else 0.0
        )
        stages["scoring"] = (
            width * model.model_weights
            + 4 * model.layers * weights
            + grids
            + _SCORING_GRADIENT * weights
            + _SCORING_LOGITS * scoring_tokens * model.vocab * 4
            + _step_activations(model, run.scoring_sequences, run.scoring_seq_len - 1)
            + _SCORING_FIXED
        )
    return MemoryPrediction(
        stages={stage: round(value) for stage, value in stages.items()},
        parts={
            "streams": streams,
            "embedding": embedding,
            "codes": round(finished),
            "tokens": 4 * tokens,
        },
        model=model,
        run=run,
    )


def fitting_micro_batches(model: ModelShape, run: RunShape, budget: int) -> int | None:
    """The fewest pieces a step can run in for the run to fit ``budget``.

    Fewest, because each piece costs time and a run in pieces is another draw
    of the checkpoint (ledger row 122). Only the steps and the measurements at
    both ends of a block's tuning shrink with the pieces, so a run whose other
    stages do not fit has no answer here.

    Returns:
        The number of pieces, or ``None`` when no number between one and the
        batch size brings the predicted peak under the budget.
    """
    for pieces in range(1, run.batch_size + 1):
        if predict_memory(model, run.with_micro_batches(pieces)).peak <= budget:
            return pieces
    return None


def default_budget(device: int) -> int | None:
    """:data:`BUDGET_SHARE` of the device's memory, or ``None`` when it is unknown."""
    return int(device * BUDGET_SHARE) if device > 0 else None


@dataclasses.dataclass(frozen=True, slots=True)
class MemoryCheck:
    """A run's predicted peak against what it may reach.

    Attributes:
        prediction: The prediction, stage by stage.
        budget: Bytes the run may reach, or ``None`` when nothing says: the
            caller passed no budget and this machine's memory is unknown.
        device: This machine's memory in bytes, or zero when unknown.
        caller_budget: Whether the budget is the caller's rather than
            :data:`BUDGET_SHARE` of the device.
        micro_batches: The fewest pieces that fit the budget, when the run
            as asked does not fit and some number of pieces does.
        only: The stages checked, when the entry point runs only part of a
            run; ``None`` for all of them.
    """

    prediction: MemoryPrediction
    budget: int | None
    device: int
    caller_budget: bool = False
    micro_batches: int | None = None
    only: frozenset[str] | None = None

    @property
    def fits(self) -> bool:
        """Whether the predicted peak is within the budget, if there is one."""
        return self.budget is None or self.prediction.peak <= self.budget

    def _budget(self) -> str:
        """The budget and where it came from, for a sentence."""
        if self.caller_budget:
            device = f", on a Mac of {_gb(self.device)}" if self.device > 0 else ""
            return f"the budget of {_gb(self.budget or 0)} it was given{device}"
        return (
            f"the budget of {_gb(self.budget or 0)}, "
            f"{BUDGET_SHARE:.0%} of this Mac's {_gb(self.device)}"
        )

    def describe(self) -> str:
        """One line: the predicted peak, where it falls, and against what."""
        prediction = self.prediction
        line = f"memory: predicted peak {_gb(prediction.peak)} in {prediction.stage}"
        if self.budget is None:
            return f"{line}; this machine's memory is unknown, so nothing was checked against it"
        return f"{line}, {'within' if self.fits else 'over'} {self._budget()}"

    def _uniform(self, model: ModelShape, run: RunShape) -> str:
        """What the same run would need without its scoring pass."""
        uniform = dataclasses.replace(run, mixed=False)
        alone = predict_memory(model, uniform).peak
        pieces = fitting_micro_batches(model, uniform, self.budget or 0)
        line = f"A uniform run does not score: at these settings it is predicted at {_gb(alone)}"
        if pieces is not None and pieces != run.micro_batches:
            fitted = predict_memory(model, uniform.with_micro_batches(pieces)).peak
            line += f", and at {_gb(fitted)} with --micro-batches {pieces}"
        return line + "."

    def explain(self) -> str:
        """Why the run does not fit, and what would. Empty when it fits."""
        if self.fits:
            return ""
        prediction = self.prediction
        model, run = prediction.model, prediction.run
        lines = [
            f"this run is predicted to reach {_gb(prediction.peak)} in {prediction.stage}, "
            f"over {self._budget()}."
        ]
        if self.micro_batches is not None:
            fitted = predict_memory(model, run.with_micro_batches(self.micro_batches))
            lines.append(
                f"With each step's batch in {self.micro_batches} pieces it is predicted at "
                f"{_gb(fitted.peak)}: rerun with --micro-batches {self.micro_batches} "
                f"(TuningConfig(micro_batches={self.micro_batches}) in Python). Pieces cost "
                f"time, and a run in pieces is another draw of the checkpoint."
            )
        else:
            smallest = predict_memory(model, run.with_micro_batches(run.batch_size))
            if self.only is not None:
                smallest = smallest.restricted(self.only)
            if smallest.stage == "scoring":
                lines.append(
                    "Pieces do not change the mixed precision scoring pass, which holds the "
                    "whole model at its own width and a float32 copy of every block weight."
                )
                if self.only is None:
                    lines.append(self._uniform(model, run))
            else:
                parts = smallest.parts
                lines.append(
                    f"No number of pieces fits: in {run.batch_size} it is still "
                    f"{_gb(smallest.peak)} in {smallest.stage}. Beside a stage's own working "
                    f"set the loop holds two float32 activation streams over the calibration "
                    f"set, {_gb(parts['streams'])}, which fewer --samples or a shorter "
                    f"--seq-len shrink, and the codes of the finished blocks, "
                    f"{_gb(parts['codes'])}."
                )
        lines.append(
            "To run as asked anyway, raise the budget with --memory-budget or turn the "
            "check off with --skip-memory-check (memory_budget and memory_check in Python)."
        )
        return " ".join(lines)

    def record(self, *, enforced: bool) -> dict[str, Any]:
        """Plain data for the run record.

        Args:
            enforced: Whether a run over the budget would have been refused.
        """
        source = None
        if self.budget is not None:
            source = "caller" if self.caller_budget else f"{BUDGET_SHARE:g} of the device"
        return {
            **self.prediction.record(),
            "budget_bytes": self.budget,
            "budget_source": source,
            "device_bytes": self.device,
            "fits": self.fits,
            "enforced": enforced,
            "fitting_micro_batches": self.micro_batches,
        }


def check_memory(
    model: ModelShape,
    run: RunShape,
    *,
    budget: int | None = None,
    device: int = 0,
    stages: Collection[str] | None = None,
) -> MemoryCheck:
    """Predict a run and hold the prediction to a budget.

    Args:
        model: The model's shape.
        run: The run's settings.
        budget: Bytes the run may reach, or ``None`` for
            :data:`BUDGET_SHARE` of ``device``.
        device: This machine's memory in bytes, zero when unknown.
        stages: Only these stages, for an entry point that runs part of a
            run (the mixed precision planning pass is ``{"scoring"}``).

    Returns:
        The check, with the fewest pieces that fit when the run does not.

    Raises:
        ValueError: If ``budget`` is not a positive number of bytes.
    """
    if budget is not None and budget <= 0:
        msg = f"a memory budget is a positive number of bytes, got {budget}"
        raise ValueError(msg)
    prediction = predict_memory(model, run)
    if stages is not None:
        prediction = prediction.restricted(stages)
    limit = budget if budget is not None else default_budget(device)
    pieces = None
    if limit is not None and prediction.peak > limit and stages is None:
        pieces = fitting_micro_batches(model, run, limit)
    return MemoryCheck(
        prediction=prediction,
        budget=limit,
        device=device,
        caller_budget=budget is not None,
        micro_batches=pieces,
        only=None if stages is None else frozenset(stages),
    )
