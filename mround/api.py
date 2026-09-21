# Copyright 2026 The MRound Authors
# SPDX-License-Identifier: Apache-2.0
"""The public Python API.

This module is the only stable surface. Everything under :mod:`mround` other
than this and :mod:`mround.schemes` is internal and may change without notice.

Typical use::

    from mround import api

    result = api.quantize(
        "mlx-community/Qwen2.5-0.5B",
        bits=4,
        output_dir="./qwen-4bit",
    )
    print(result.summary.total_seconds)

Both entry points are implemented. `quantize_round_to_nearest` applies the same
scale computation and rounding rule with nothing learned, and exists to be the
number `quantize` has to beat, measured through the same loader and the same
export path so that a difference between them is attributable to the learning
rather than to anything else in the chain.

Mixed precision is `average_bits`: a sensitivity scoring pass per candidate
width (DOCUMENTATION.md 1.5), an exact allocation under the size budget the
average implies, and the block loop honoring the per-layer result. The budget
is defined so that an integer average competes under exactly the ceiling the
uniform width at that number costs, which is what makes "mixed at 3" against
"uniform 3" a fair fight. MEMORY.md D-031 and D-033 record the decisions.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mround.schemes import LOW_BIT_THRESHOLD, QuantScheme, ScaleInit, Symmetry, TuningConfig

if TYPE_CHECKING:
    from collections.abc import Collection, Iterator, Mapping, Sequence

    from mround.formats.mlx_export import QuantizedLayer
    from mround.pipeline.loader import ModelBundle
    from mround.pipeline.runner import BlockResult, RunSummary
    from mround.pipeline.scoring import ScoringProgress

__all__ = [
    "Coverage",
    "QuantScheme",
    "QuantizeResult",
    "RoundToNearestResult",
    "ScaleInit",
    "Storage",
    "Symmetry",
    "TuningConfig",
    "block_coverage",
    "default_scale_init",
    "plan_mixed_precision",
    "quantize",
    "quantize_round_to_nearest",
]


@dataclasses.dataclass(frozen=True, slots=True)
class QuantizeResult:
    """What a quantization run produced.

    Attributes:
        output_dir: Where the checkpoint was written.
        summary: Per-block results and run-level measurements.
        scheme_by_layer: The scheme each layer ended up with, which is not
            uniform when mixed precision is in play.
        config_path: A configuration file capturing every input to this run.
            Re-running from it reproduces the result, which is what makes a
            published number checkable.
        storage: What the checkpoint costs. Reported by the same accounting the
            baseline uses, because a quality comparison between two checkpoints
            means nothing until they are known to be the same size.
        skipped: Module paths left dense, with the reason.
        mixed: What the planner decided, when ``average_bits`` engaged it:
            target and achieved averages, layer counts per width, the budget,
            and the scoring cost. ``None`` on uniform runs.
    """

    output_dir: Path
    summary: RunSummary
    scheme_by_layer: dict[str, QuantScheme]
    config_path: Path
    storage: Storage
    skipped: tuple[tuple[str, str], ...] = ()
    mixed: dict[str, Any] | None = None

    def describe(self) -> str:
        """One line summarizing what was produced."""
        return (
            f"{len(self.scheme_by_layer)} modules tuned, "
            f"{len(self.skipped)} left dense, "
            f"{self.storage.bits_per_weight:.2f} bits per weight overall "
            f"({self.storage.compression:.2f}x smaller)"
        )


def default_scale_init(narrowest_width: int, symmetry: Symmetry) -> ScaleInit:
    """Which scale initialization a run gets when the caller does not say.

    Searched below four bits on symmetric schemes, observed range otherwise.
    That is the position D-030 measured and CLAUDE.md states, applied here
    rather than left to whoever calls, because the consequence of getting it
    wrong is not a few percent. At 2 bits the observed grid never got below 77,365 perplexity and
    the searched grid took the identical recipe to 1,361: the difference between
    a broken model and a working one. At 4 bits the search is quality neutral,
    so the cheaper initialization stays the default there, by measurement.

    The rule keys on the narrowest width the run can assign rather than on
    ``bits``, because one scale initialization covers every layer in a
    checkpoint. A mixed run over {2, 3, 4} has a base width of 4 and must still
    be searched, or its 2-bit layers get the initialization that was measured at
    77,365.

    **Asymmetric schemes always get the observed range**, at every width, and
    that is not a judgement about quality. The search has no asymmetric branch:
    ``QuantScheme`` refuses the combination, as the reference implementation
    does. Resolving to a value the scheme cannot hold would turn a default
    nobody chose into an error about a setting nobody asked for, which is what
    ``--asym --bits 2`` did before this argument existed. What it does mean is
    that D-030's result, measured on symmetric schemes, says nothing about how
    an asymmetric 2-bit run behaves, and nothing here has measured one.

    Args:
        narrowest_width: Fewest code bits any layer in this run can receive.
        symmetry: The code range this run will use. Required rather than
            defaulted, because a silent default is what this whole family of
            functions exists to stop.

    Returns:
        The initialization to use when ``scale_init`` was not given.
    """
    if symmetry is not Symmetry.SYMMETRIC:
        return ScaleInit.OBSERVED_RANGE
    return ScaleInit.SEARCHED if narrowest_width < LOW_BIT_THRESHOLD else ScaleInit.OBSERVED_RANGE


# Distinct offending dimensions named in a coverage message before it gives up
# and says "and so on". Three is enough to show whether one dimension is the
# problem or several, which is the thing a reader needs.
_DIMENSIONS_SHOWN = 3

# Below this share of the block stack, a run is refused rather than producing a
# checkpoint mostly at full precision. Half is deliberately generous: one odd
# projection in an architecture leaves coverage in the high nineties, while the
# case this exists for was 14 percent.
MIN_BLOCK_COVERAGE = 0.5


@dataclasses.dataclass(frozen=True, slots=True)
class Coverage:
    """How much of a block stack a group size can actually reach.

    A layer is groupable only when its input dimension is a multiple of the
    group size, and a layer that is not groupable is left at full precision.
    One or two of those is ordinary. Most of them is a different model from the
    one that was asked for, and it is not obvious from the output: the run that
    prompted this reported "30 modules at 4 bits, 181 left dense" in one line
    and then wrote a 230 MB checkpoint for somebody who asked for 4 bits.

    Attributes:
        total: Linear layers inside the block stack.
        blocked: Each layer that cannot be grouped, with the input dimension
            that stopped it, so a message can name the number rather than
            saying the group size does not fit.
    """

    total: int
    blocked: tuple[tuple[str, int], ...] = ()

    @property
    def representable(self) -> int:
        """Layers this group size can quantize."""
        return self.total - len(self.blocked)

    @property
    def fraction(self) -> float:
        """Share of the block stack that will be quantized, 0 to 1.

        A model with no block linears at all counts as fully covered rather
        than as zero: there is nothing there to fail to quantize, and a
        division by zero here would refuse a run for the wrong reason.
        """
        return 1.0 if self.total == 0 else self.representable / self.total

    def alternatives(self, supported: Collection[int]) -> tuple[int, ...]:
        """Which of ``supported`` would divide every dimension that blocked.

        Takes the candidates rather than importing them, because the set that
        matters belongs to the export format and this layer does not depend on
        any export format.
        """
        dimensions = {dimension for _, dimension in self.blocked}
        return tuple(sorted(size for size in supported if all(d % size == 0 for d in dimensions)))

    def explain(self, group_size: int, supported: Collection[int]) -> str:
        """Why this group size does not fit, and what would."""
        dimensions = sorted({dimension for _, dimension in self.blocked})
        shown = ", ".join(str(d) for d in dimensions[:_DIMENSIONS_SHOWN])
        if len(dimensions) > _DIMENSIONS_SHOWN:
            shown += ", ..."
        works = self.alternatives(supported)
        remedy = (
            f"Group sizes that divide every one of them: {', '.join(str(s) for s in works)}."
            if works
            else "No supported group size divides all of them."
        )
        return (
            f"group size {group_size} can only quantize {self.representable} of "
            f"{self.total} layers in the block stack ({self.fraction:.0%}); the rest "
            f"would be left at full precision and the checkpoint would be far larger "
            f"than the width asked for. Input dimensions that do not divide: {shown}. "
            f"{remedy}"
        )


def _require_coverage(bundle: Any, scheme: QuantScheme, minimum: float) -> None:
    """Refuse a group size that would leave most of the block stack dense.

    Called after the model is loaded, before the calibration corpus is built,
    and long before anything is tuned. The runner already refuses a block with
    nothing quantizable in it, but that guard is per block and all or nothing,
    so it never fires when every block keeps one layer and loses the other six,
    which is exactly what a group size of 128 does to a model with a 576-wide
    hidden state.
    """
    from mround.exceptions import UnsupportedSchemeError  # noqa: PLC0415
    from mround.formats.mlx_export import MLX_GROUP_SIZES  # noqa: PLC0415

    coverage = block_coverage(
        bundle.model,
        scheme.group_size,
        expected_blocks=bundle.config.get("num_hidden_layers"),
    )
    if coverage.fraction < minimum:
        raise UnsupportedSchemeError(coverage.explain(scheme.group_size, MLX_GROUP_SIZES))


def block_coverage(model: Any, group_size: int, *, expected_blocks: int | None = None) -> Coverage:
    """Measure what a group size can reach, before anything is tuned.

    Walks the same blocks and the same layers the runner will, so the answer is
    the runner's answer rather than an estimate of it. Cheap: it reads shapes
    and touches no weights.

    Args:
        model: A loaded MLX model.
        group_size: The group size to test. Per-channel (``-1``) reaches
            everything by construction.
        expected_blocks: Passed to block discovery, which uses it to check the
            stack it found is the stack the config describes.

    Returns:
        The coverage this group size would achieve.
    """
    from mround.pipeline.blocks import (  # noqa: PLC0415
        discover_blocks,
        iter_quantizable_linears,
    )

    total = 0
    blocked: list[tuple[str, int]] = []
    for block in discover_blocks(model, expected=expected_blocks):
        for name, linear in iter_quantizable_linears(block.module):
            total += 1
            dimension = int(linear.weight.shape[-1])
            if group_size > 0 and dimension % group_size:
                blocked.append((f"{block.name}.{name}", dimension))
    return Coverage(total=total, blocked=tuple(blocked))


def _require_supported_host() -> None:
    """Refuse a host that cannot run MRound, before anything imports MLX.

    Every public entry point below calls this first, and *first* is the whole
    point. The host this check is written for, a Mac whose Python is running
    under Rosetta, cannot install MLX at all, so an import placed above the
    check turns a sentence the person can act on into a ModuleNotFoundError
    traceback from somewhere in the pipeline. `mround doctor` reported that
    machine correctly while `mround quantize` did not, which is how it was
    found. `pipeline/loader.py` is MLX-free at import time so that this
    ordering is possible.
    """
    from mround.pipeline.loader import require_apple_silicon  # noqa: PLC0415

    require_apple_silicon()


def quantize(
    model: str | Path,
    *,
    output_dir: str | Path,
    bits: int = 4,
    group_size: int = 64,
    symmetry: Symmetry = Symmetry.SYMMETRIC,
    average_bits: float | None = None,
    candidate_bits: Sequence[int] = (2, 3, 4),
    sensitivity_gradients: str = "widest",
    remainder_bits: int | None = None,
    dense_remainder: bool = False,
    scoring_samples: int | None = None,
    scoring_seq_len: int | None = None,
    scoring_batch_size: int | None = None,
    scheme: QuantScheme | None = None,
    tuning: TuningConfig | None = None,
    calibration: str = "auto",
    export_format: str = "mlx",
    quantized_inputs: bool = True,
    holdout_batches: int = 0,
    per_layer_recipe: bool = False,
    scale_init: ScaleInit | str | None = None,
    min_block_coverage: float = MIN_BLOCK_COVERAGE,
    seed: int | None = None,
) -> QuantizeResult:
    """Quantize a model and write a checkpoint.

    Args:
        model: Hugging Face repository id or local path.
        output_dir: Where to write the result.
        bits: Uniform bit width. Ignored when ``scheme`` is given.
        group_size: Weights sharing one scale. ``-1`` for per-channel. 64
            rather than the reference's 128, because a layer is groupable only
            when its input dimension is a multiple of the group size and 64
            divides strictly more models. Measured on SmolLM2-135M, whose
            hidden size is 576: at 128 only the one layer per block with a
            1536-wide input can be grouped, 181 of 211 modules stay dense, and
            the checkpoint comes out at 13.68 bits per weight instead of 4.50.
        symmetry: Signed or unsigned code range.
        average_bits: Target average code bits per weight over the block
            layers, for mixed precision. Setting this engages the planner: a
            sensitivity scoring pass per candidate width, then an exact
            allocation under the size budget the average implies, defined so
            that an integer average competes under exactly the uniform width's
            ceiling. Must lie within the candidate range. The block layers
            then take the widths the planner assigns rather than ``bits``,
            but ``bits`` is not ignored: the block loop resolves its learning
            rate constant and its outlier suppression from it, once per
            block, for every layer in the block, unless
            ``per_layer_recipe`` is set (MEMORY.md D-043). The
            tensors outside the block loop take the supported width nearest
            the average, ties toward more bits (D-034).
        candidate_bits: Widths the planner may assign per layer. Ignored when
            ``average_bits`` is ``None``. The default covers the 2-to-4
            average band the method serves, and its granularity is a measured
            decision (MEMORY.md D-035): a menu missing the intermediate width
            forces every demotion to be extreme, and at the standard
            calibration recipe that lost to uniform where the graded menu
            wins. Scoring cost and scoring memory grow linearly with the
            count; averages outside the band need their own menu.
        sensitivity_gradients: Where the scoring gradients come from.
            ``"widest"``, the default, takes one backward at the widest
            candidate and scores every width's perturbation against it; it
            won the A/B on both averages tested and is MRound's measured
            divergence from the reference (MEMORY.md D-034). ``"own_width"``
            is the reference's procedure, each width scored on the model
            quantized at that width, retained as the reference-faithful
            control. Ignored without ``average_bits``.
        remainder_bits: Width for the tensors outside the block loop in a
            mixed run, overriding the default of the supported width nearest
            the average, ties toward more bits. The tie direction is a
            measurement: a tied output head at the lower width cost five
            times the perplexity (D-034). Ignored without ``average_bits``.
        dense_remainder: Leave the tensors outside the block loop at the
            model's own precision instead of quantizing them. Off by default
            and deliberately so: D-019 exists because a tied-embedding model
            that keeps its largest tensor at full precision is barely
            compressed at all, and on Qwen2.5-0.5B that tensor is 27.6 percent
            of the weights. This exists as a comparison control. Intel
            AutoRound leaves the output head dense on a tied model and refuses
            to do otherwise (``set_layer_config`` resets ``quant_lm_head`` to
            false when ``tie_word_embeddings`` is set, in both 0.14.2 and
            0.15.0), and it never quantizes embeddings outside its GGUF path,
            so matching its coverage can only be done from this side. A run
            with this on measures the block algorithm; a run without it
            measures the product.
        scheme: A fully specified scheme, overriding ``bits``, ``group_size``,
            and ``symmetry``.
        tuning: How the learned rounding is optimized. ``None`` uses the
            standard recipe.
        calibration: Corpus identifier, or ``"auto"`` to select a default
            appropriate to the model.
        export_format: ``"mlx"``. GGUF is planned (ROADMAP.md Phase 4) and
            refused until then.
        quantized_inputs: Tune each block against the activations the already
            quantized blocks produced, rather than the original model's. On by
            default, which is MEMORY.md Q-001's answer. Exposed because turning
            it off halves activation memory, and whether that trade is worth
            making on a large model is a measurement this parameter exists to
            allow rather than a preference.
        holdout_batches: Calibration batches kept out of tuning and used only to
            judge whether tuning helped. Zero measures that judgement on the data
            being fit, which cannot see overfitting, and overfitting is the
            binding problem at low widths. See MEMORY.md D-028.
        per_layer_recipe: Under a mixed plan, give every layer the learning
            rate its own assigned width implies, and resolve the outlier rule
            from the narrowest width in each block, rather than resolving both
            once from ``bits``. Off by default, which is what every published
            mixed measurement was produced with; this exists so the two can be
            compared on one model before either becomes the rule. Ignored
            without ``average_bits``, since a uniform run has one width.
            MEMORY.md D-043.
        scale_init: How each group's scale is initialized. ``OBSERVED_RANGE``
            derives it from the group extremes; ``SEARCHED`` runs the v2 grid
            search per group, importance-weighted by the activations each
            channel actually sees, and tuning starts from the winner with the
            clipping coefficient reinterpreted as a multiplier on it. The
            search is what separates v2 from v1, and MEMORY.md D-028 is the
            evidence for why very low widths need it. Symmetric schemes only.
            ``None``, the default, resolves it from the narrowest width this run
            can assign: see :func:`default_scale_init`, and do not override it
            below 4 bits without reading D-030 first.
        min_block_coverage: Refuse before tuning when the group size can reach
            less than this share of the block stack. Zero permits any coverage,
            for the caller who really does want a mostly dense checkpoint.
        scoring_samples: Sequences drawn for the mixed precision scoring pass,
            or ``None`` for the reference's 16. Ignored without
            ``average_bits``.
        scoring_seq_len: Tokens per scoring sequence, or ``None`` for the
            reference's 256. Ignored without ``average_bits``. Both defaults
            came from the reference (D-031), and the reference now warns that
            they are too small below 3 bits, so they are exposed to be
            measured rather than inherited.
        scoring_batch_size: Sequences per scoring batch, or ``None`` to derive
            it from the budget at a fixed token count per batch. Overriding it
            is for reproducing a measurement taken before that derivation
            existed; raising it raises the scoring peak in proportion, and the
            run that prompted the derivation asked a 32 GB machine for 52 GB.
        seed: Calibration sampling seed, or ``None`` for the corpus default,
            which is the reference's 42. The seed is part of the corpus content
            hash, so a different one shows up in the run configuration rather
            than silently. Exposed for the repeat the results ask for: learned
            rounding's spread across seeds runs to tens of percent at low
            widths, so a single-seed gap is a reading rather than a result.

    Returns:
        Where the checkpoint went and what the run measured.

    Raises:
        PlatformError: If the host is not a supported Apple Silicon Mac.
        SchemeError: If the requested scheme is malformed.
        ArchitectureError: If the model cannot be loaded or walked.
        CalibrationError: If the corpus cannot be built.
        UnsupportedSchemeError: If ``export_format`` cannot represent the scheme
            faithfully, or if a requested feature is not implemented yet.
    """
    _require_supported_host()

    import json  # noqa: PLC0415

    from mround.exceptions import ArchitectureError, UnsupportedSchemeError  # noqa: PLC0415
    from mround.formats.mlx_export import require_mlx_representable  # noqa: PLC0415
    from mround.pipeline.calibration import (  # noqa: PLC0415
        DEFAULT_CORPUS,
        DEFAULT_SEED,
        build_calibration_set,
    )
    from mround.pipeline.device import reset_peak_memory  # noqa: PLC0415
    from mround.pipeline.loader import load_model  # noqa: PLC0415
    from mround.pipeline.runner import QuantizationRunner  # noqa: PLC0415

    if export_format != "mlx":
        msg = (
            f"export_format={export_format!r} is not implemented; only 'mlx' is. "
            f"See ROADMAP.md Phase 4."
        )
        raise UnsupportedSchemeError(msg)

    # Every width this run can assign, not just ``bits``: a mixed run reaches
    # widths ``bits`` alone does not name, the candidates and the remainder
    # included, and two things below have to see all of them, the
    # representability check and the scale initialization.
    widths = {scheme.bits if scheme is not None else bits}
    if average_bits is not None:
        widths.update(int(width) for width in candidate_bits)
        if remainder_bits is not None:
            widths.add(int(remainder_bits))

    resolved = scheme or QuantScheme(
        bits=bits,
        group_size=group_size,
        symmetry=symmetry,
        scale_init=(
            default_scale_init(min(widths), symmetry)
            if scale_init is None
            else ScaleInit(scale_init)
        ),
    )
    tuning = tuning or TuningConfig()
    corpus = DEFAULT_CORPUS if calibration == "auto" else calibration
    destination = Path(output_dir)

    # Refused before the model is loaded, not after it is tuned. The exporter
    # applies the same check per layer at the end of the run, which is an hour
    # too late for a width or a group size MLX cannot represent.
    for width in sorted(widths):
        require_mlx_representable(dataclasses.replace(resolved, bits=width))

    reset_peak_memory()
    bundle = load_model(model)
    _refuse_remote_code(bundle.config)

    _require_coverage(bundle, resolved, min_block_coverage)

    calibration_set = build_calibration_set(
        bundle.tokenizer,
        source=corpus,
        n_samples=tuning.n_samples,
        seq_len=tuning.seq_len,
        batch_size=tuning.batch_size,
        seed=DEFAULT_SEED if seed is None else seed,
    )

    # Mixed precision plans before any weight is touched: the scoring backward
    # needs the full-precision model, and the runner consumes the result as
    # per-layer overrides it already knew how to honor.
    scheme_overrides: dict[str, QuantScheme] | None = None
    plan_info: dict[str, Any] | None = None
    remainder_scheme = resolved
    if average_bits is not None:
        scheme_overrides, plan_info = _plan_mixed(
            bundle,
            resolved,
            average_bits,
            candidate_bits,
            corpus,
            expected_blocks=bundle.config.get("num_hidden_layers"),
            gradient_source=sensitivity_gradients,
            remainder_bits=remainder_bits,
            scoring_samples=scoring_samples,
            scoring_seq_len=scoring_seq_len,
            scoring_batch_size=scoring_batch_size,
            seed=seed,
        )
        remainder_scheme = dataclasses.replace(resolved, bits=int(plan_info["remainder_bits"]))
        print(f"  {plan_info['describe']}")

    runner = QuantizationRunner(
        resolved,
        tuning,
        progress=_log_block,
        quantized_inputs=quantized_inputs,
        holdout_batches=holdout_batches,
        per_layer_recipe=per_layer_recipe,
    )
    summary, quantized = runner.run(
        bundle.model,
        calibration_set.batches,
        scheme_by_layer=scheme_overrides,
        expected_blocks=bundle.config.get("num_hidden_layers"),
    )

    # Everything outside the block stack: embeddings, the output head, and the
    # final normalization. Learned rounding has nothing to optimize for a lookup
    # table, so these take round-to-nearest at the model level, which is what
    # D-019 is about and what keeps a tied-embedding model from storing its
    # largest tensor at full precision.
    skipped: list[tuple[str, str]] = []
    for path, layer, reason in _round_to_nearest_layers(
        bundle, remainder_scheme, skip=set(quantized)
    ):
        if layer is None:
            skipped.append((path, reason))
        elif dense_remainder:
            # Left at the model's own precision on purpose, and recorded as a
            # skip so the checkpoint's own configuration says what was done to
            # it rather than leaving a reader to infer it from the file size.
            skipped.append((path, "dense by request, to match a reference's coverage"))
        else:
            quantized[path] = layer
    if not quantized:
        msg = f"nothing in {model!r} could be quantized at group_size={resolved.group_size}."
        raise ArchitectureError(msg)

    dense = _write_checkpoint(bundle, destination, quantized)
    storage = _storage(bundle, quantized, dense)
    config_path = destination / "mround_run.json"
    config_path.write_text(
        json.dumps(
            {
                "model": str(model),
                "scheme": dataclasses.asdict(resolved),
                "dense_remainder": dense_remainder,
                # Recorded because it changes how every block was tuned and
                # cannot be recovered from the weights (D-043).
                "per_layer_recipe": per_layer_recipe,
                "tuning": dataclasses.asdict(tuning),
                "calibration": {
                    "source": calibration_set.source,
                    "packing": str(calibration_set.packing),
                    "n_samples": calibration_set.n_samples,
                    "seq_len": calibration_set.seq_len,
                    "seed": calibration_set.seed,
                    "holdout_batches": holdout_batches,
                    "content_hash": calibration_set.content_hash,
                },
                "mixed": None
                if plan_info is None
                else {key: value for key, value in plan_info.items() if key != "describe"},
                "result": {
                    "blocks": len(summary.blocks),
                    "reverted": summary.reverted,
                    "mean_improvement": summary.mean_improvement,
                    "total_seconds": summary.total_seconds,
                    "peak_memory_bytes": summary.peak_memory_bytes,
                    "bits_per_weight": storage.bits_per_weight,
                    "stored_bytes": storage.stored_bytes,
                    "skipped": [list(entry) for entry in skipped],
                },
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    return QuantizeResult(
        output_dir=destination,
        summary=summary,
        scheme_by_layer=summary.scheme_by_layer,
        config_path=config_path,
        storage=storage,
        skipped=tuple(sorted(skipped)),
        mixed=plan_info,
    )


@dataclasses.dataclass(frozen=True, slots=True)
class RoundToNearestResult:
    """What a round-to-nearest run produced.

    Attributes:
        output_dir: Where the checkpoint was written.
        scheme: The scheme applied uniformly.
        quantized: Module paths that were quantized.
        skipped: Module paths left dense, with the reason. Reported rather than
            counted, because "this model is 4-bit" is false in a way nobody can
            see if a large layer quietly opted out.
        bits_per_weight: Stored bits divided by original weights, counting
            everything the checkpoint has to hold: the packed codes, the scales
            and biases, and any tensor left dense. Always above the requested
            width, and the number that predicts the file size.
        stored_bytes: What the weights occupy on disk, as computed. Comparing it
            against the file actually written is a cheap check that the
            accounting and the writing agree.
        original_bytes: What they occupied before, for the compression ratio.
    """

    output_dir: Path
    scheme: QuantScheme
    quantized: tuple[str, ...]
    skipped: tuple[tuple[str, str], ...]
    bits_per_weight: float
    stored_bytes: int
    original_bytes: int

    @property
    def compression(self) -> float:
        """How much smaller the weights got."""
        return self.original_bytes / max(1, self.stored_bytes)

    def describe(self) -> str:
        """One line summarizing what actually happened."""
        return (
            f"{len(self.quantized)} modules at {self.scheme.bits} bits, "
            f"{len(self.skipped)} left dense, "
            f"{self.bits_per_weight:.2f} bits per weight overall "
            f"({self.compression:.2f}x smaller)"
        )


def quantize_round_to_nearest(
    model: str | Path,
    *,
    output_dir: str | Path,
    bits: int = 4,
    group_size: int = 64,
    symmetry: Symmetry = Symmetry.SYMMETRIC,
) -> RoundToNearestResult:
    """Quantize a model without learning anything, and write a checkpoint.

    This is the baseline, not the method. It applies the same scale computation
    and the same rounding rule as the full pipeline with the learned quantities
    held at their initial values, which is exactly round-to-nearest. Its purpose
    is to be the number learned rounding has to beat, measured on the same model
    with the same export path, so that any later improvement is attributable to
    the learning rather than to anything else in the chain.

    It needs no calibration data and no block loop, which is what makes it
    available now.

    Args:
        model: Hugging Face repository id or local path.
        output_dir: Where to write the result.
        bits: Uniform bit width.
        group_size: Weights sharing one scale, along the input dimension.
        symmetry: Signed or unsigned code range.

    Returns:
        Where the checkpoint went, and what was and was not quantized.

    Raises:
        PlatformError: If the host is not a supported Apple Silicon Mac.
        ArchitectureError: If the model cannot be loaded, is already quantized,
            or needs handling MRound does not yet have.
        UnsupportedSchemeError: If the scheme has no MLX representation.
    """
    _require_supported_host()

    from mround.exceptions import ArchitectureError  # noqa: PLC0415
    from mround.formats.mlx_export import require_mlx_representable  # noqa: PLC0415
    from mround.pipeline.device import reset_peak_memory  # noqa: PLC0415
    from mround.pipeline.loader import load_model  # noqa: PLC0415

    scheme = QuantScheme(bits=bits, group_size=group_size, symmetry=symmetry)
    require_mlx_representable(scheme)
    destination = Path(output_dir)
    reset_peak_memory()
    bundle = load_model(model)
    _refuse_remote_code(bundle.config)

    quantized: dict[str, QuantizedLayer] = {}
    skipped: list[tuple[str, str]] = []
    for path, layer, reason in _round_to_nearest_layers(bundle, scheme):
        if layer is None:
            skipped.append((path, reason))
        else:
            quantized[path] = layer

    if not quantized:
        msg = (
            f"nothing in {model!r} could be quantized at group_size={group_size}. "
            f"Every candidate module failed the divisibility check."
        )
        raise ArchitectureError(msg)

    dense = _write_checkpoint(bundle, destination, quantized)
    storage = _storage(bundle, quantized, dense)
    return RoundToNearestResult(
        output_dir=destination,
        scheme=scheme,
        quantized=tuple(sorted(quantized)),
        skipped=tuple(sorted(skipped)),
        bits_per_weight=storage.bits_per_weight,
        stored_bytes=storage.stored_bytes,
        original_bytes=storage.original_bytes,
    )


def plan_mixed_precision(
    model: str | Path,
    average_bits: float,
    candidate_bits: Sequence[int] = (2, 3, 4),
    *,
    bits: int = 4,
    group_size: int = 64,
    symmetry: Symmetry | str = Symmetry.SYMMETRIC,
    scale_init: ScaleInit | str = ScaleInit.SEARCHED,
    scheme: QuantScheme | None = None,
    calibration: str = "auto",
    sensitivity_gradients: str = "widest",
    remainder_bits: int | None = None,
    scoring_samples: int | None = None,
    scoring_seq_len: int | None = None,
    scoring_batch_size: int | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Score and allocate, and stop there. Nothing is tuned and nothing written.

    The allocation is the whole of what a scoring budget can change. Tuning
    that follows it is deterministic given the plan, the calibration set, and
    the seed, so two runs whose allocations agree produce the same checkpoint
    and cannot differ in perplexity. That makes this the cheap half of any
    experiment about scoring: plan at the new budget, compare the assignments
    against a run already on disk, and only pay for the block loop if they
    moved. On a 0.5B model that is minutes to an hour and a half instead of two
    and a half hours.

    Args:
        model: Hugging Face repository id or a local checkpoint directory.
        average_bits: Target average of code bits per weight.
        candidate_bits: Widths the allocator may choose between.
        bits: Width of the base scheme, which only supplies the properties the
            candidates share; the allocator overrides the width itself.
        group_size: Weights sharing one scale.
        symmetry: Signed or unsigned code range.
        scale_init: How each group's scale starts.
        scheme: A complete scheme, overriding the four fields above.
        calibration: Corpus identifier, or ``"auto"`` for the project default.
        sensitivity_gradients: ``"widest"`` or ``"own_width"``; see
            :func:`mround.pipeline.scoring.score_widths`.
        remainder_bits: Width for the tensors outside the block stack, or
            ``None`` for the supported width nearest the average.
        scoring_samples: Sequences drawn for scoring, or ``None`` for
            ``SENSITIVITY_SAMPLES``.
        scoring_seq_len: Tokens per scoring sequence, or ``None`` for
            ``SENSITIVITY_SEQ_LEN``. Both defaults are the budget every
            published measurement used, not the reference's own, which they
            were until D-031 measured the difference at 4.70 percent.
        scoring_batch_size: Sequences per scoring batch, or ``None`` to derive
            it from the budget at a fixed token count per batch.
        seed: Calibration seed.

    Returns:
        The same plain-data summary a mixed :func:`quantize` records under
        ``mixed``, including the per layer ``assignments``.

    Raises:
        PlatformError: If the host is not a supported Apple Silicon Mac.
        ArchitectureError: If the model cannot be loaded or planned.
        SchemeError: If the average lies outside the candidate range.
    """
    _require_supported_host()

    from mround.pipeline.calibration import DEFAULT_CORPUS  # noqa: PLC0415
    from mround.pipeline.device import reset_peak_memory  # noqa: PLC0415
    from mround.pipeline.loader import load_model  # noqa: PLC0415

    resolved = scheme or QuantScheme(
        bits=bits,
        group_size=group_size,
        symmetry=Symmetry(symmetry),
        scale_init=ScaleInit(scale_init),
    )
    reset_peak_memory()
    bundle = load_model(model)
    _refuse_remote_code(bundle.config)
    _, info = _plan_mixed(
        bundle,
        resolved,
        average_bits,
        candidate_bits,
        DEFAULT_CORPUS if calibration == "auto" else calibration,
        expected_blocks=bundle.config.get("num_hidden_layers"),
        gradient_source=sensitivity_gradients,
        remainder_bits=remainder_bits,
        scoring_samples=scoring_samples,
        scoring_seq_len=scoring_seq_len,
        scoring_batch_size=scoring_batch_size,
        seed=seed,
    )
    return info


# ---------------------------------------------------------------------------
# Shared between the two entry points. Kept here rather than in the pipeline
# layer because they are about assembling a deliverable, not about quantizing.
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class Storage:
    """What a checkpoint costs, counting everything it has to hold.

    Attributes:
        bits_per_weight: Stored bits divided by original weights. Always above
            the requested width, and the number that predicts the file size.
        stored_bytes: What the weights occupy, as computed. Comparing it against
            the file actually written is a cheap check that the accounting and
            the writing agree.
        original_bytes: What they occupied before.
    """

    bits_per_weight: float
    stored_bytes: int
    original_bytes: int

    @property
    def compression(self) -> float:
        """How much smaller the weights got."""
        return self.original_bytes / max(1, self.stored_bytes)


def _storage(
    bundle: ModelBundle,
    quantized: Mapping[str, QuantizedLayer],
    dense: Mapping[str, Any],
) -> Storage:
    """Account for every bit the checkpoint stores.

    The scales and biases are part of the checkpoint, not overhead to be netted
    out. One pair of them per group, at the model's dtype, and without them the
    codes are meaningless. Counting only the codes reports 4.00 for a 4-bit model
    that is really 4.50, which is the kind of number that ends up in a comparison
    table.

    Width is taken per layer rather than from one scheme, so this stays correct
    when mixed precision makes them differ.
    """
    dtype_bits = 16 if str(bundle.dtype).endswith(("float16", "bfloat16")) else 32
    code_bits = sum(int(layer.codes.size) * layer.scheme.bits for layer in quantized.values())
    groups = sum(
        int(layer.codes.shape[0]) * int(layer.codes.shape[1]) for layer in quantized.values()
    )
    quantized_weights = sum(int(layer.codes.size) for layer in quantized.values())
    dense_weights = sum(int(array.size) for array in dense.values())

    stored_bits = (
        code_bits  # packed codes
        + 2 * groups * dtype_bits  # scales and biases
        + dense_weights * dtype_bits  # whatever stayed dense
    )
    total_weights = quantized_weights + dense_weights
    return Storage(
        bits_per_weight=stored_bits / total_weights,
        stored_bytes=stored_bits // 8,
        original_bytes=total_weights * dtype_bits // 8,
    )


def _refuse_remote_code(config: Mapping[str, Any]) -> None:
    """Refuse a model whose Python has to travel with its weights.

    Raises:
        ArchitectureError: If the configuration names custom modeling code.
    """
    from mround.exceptions import ArchitectureError  # noqa: PLC0415

    if config.get("auto_map") is None:
        return
    msg = (
        "this model carries custom modeling code (auto_map in its config), "
        "whose Python files have to travel with the checkpoint. MRound does "
        "not copy them yet, so the result would load only where the original "
        "already is. Point it at a local snapshot directory and copy the .py "
        "files across by hand, or quantize a model without remote code."
    )
    raise ArchitectureError(msg)


def _log_block(result: BlockResult) -> None:
    """Print one line per block, because silence during a long run reads as a hang."""
    print(result.describe())


def _log_scoring(step: ScoringProgress) -> None:
    """Print one line per scoring pass, for the same reason, flushed.

    Flushed because this one is often watched through a redirect: log0059 is a
    scoring pass that was reported as stuck while it was running, and a line
    sitting in a block buffer would have looked exactly the same.
    """
    print(step.describe(), flush=True)


def _plan_mixed(
    bundle: ModelBundle,
    base: QuantScheme,
    average_bits: float,
    candidate_bits: Sequence[int],
    corpus: str,
    *,
    expected_blocks: int | None,
    gradient_source: str = "widest",
    remainder_bits: int | None = None,
    scoring_samples: int | None = None,
    scoring_seq_len: int | None = None,
    scoring_batch_size: int | None = None,
    seed: int | None = None,
) -> tuple[dict[str, QuantScheme], dict[str, Any]]:
    """Score, allocate, and return per-layer schemes plus what was decided.

    The procedure is DOCUMENTATION.md 1.5's: one causal-LM backward per
    candidate width on its own small calibration draw, the cancellation-free
    DeltaLoss per layer, and the exact allocator under the budget the average
    implies. The scoring draw comes from the same corpus and seed as the main
    calibration, at the reference's scoring size, so the plan's identity is
    fixed by the run configuration alone.

    **The scoring size is now an argument rather than a constant**, because the
    reference has started warning that its own default is too small: "2-bit
    scheme(s) detected. For better results, consider nsamples>=128 and
    seqlen>=1024 (current: nsamples=16, seqlen=256)". Those current values are
    where D-031 took MRound's from, so every mixed measurement in this project,
    D-034 and D-035 included, was made on a budget the reference now calls
    inadequate at the widths that matter most. Scoring cost is linear in the
    product, so the recommended budget is thirty two times the work of the
    default: on a 0.5B model, an hour and a half rather than three minutes.

    **The batch size follows from the budget rather than standing still.** It
    was a hard coded 8, which at 256 tokens is 2048 tokens per batch and at
    1024 tokens is four times the largest activation in a causal LM backward,
    the logits, on top of the resident model copies D-036 already accounts for.
    A 0.5B mixed run peaks at 17.14 GB at the default budget; the same run at 8
    by 1024 stopped making visible progress on a 32 GB machine (log0059).
    :func:`~mround.planner.sensitivity.scoring_batch_size` holds the tokens per
    batch fixed instead, which returns exactly 8 at the default and reproduces
    every earlier measurement, and which leaves the allocation at any other
    budget unchanged because the allocator's argmin does not move under a
    positive scale. Measured afterwards, which settled it: the run that
    prompted this peaked at 52.08 GB on a 32 GB machine.

    ``scoring_batch_size`` overrides that derivation. It exists because a
    measurement was taken at the old fixed 8 before the derivation landed, and
    a recorded number that the current code cannot reproduce is worse than an
    extra argument.

    Returns:
        Per-layer scheme overrides keyed by the runner's dotted paths, and a
        plain-data summary for the run record. The summary's
        ``remainder_bits`` is the width the tensors outside the block loop
        take: the supported width nearest the average, ties toward more bits
        (MEMORY.md D-034, which reversed D-033's tie rule on measurement),
        so an integer average treats them exactly as the uniform run at that
        width does.

    Raises:
        SchemeError: If there are no candidates, a candidate is unsupported,
            or the average lies outside the candidate range, which no
            allocation can reach.
    """
    import math  # noqa: PLC0415

    from mround.exceptions import SchemeError  # noqa: PLC0415
    from mround.pipeline.calibration import (  # noqa: PLC0415
        DEFAULT_SEED,
        build_calibration_set,
    )
    from mround.pipeline.scoring import score_widths  # noqa: PLC0415
    from mround.planner.allocator import allocate_bits  # noqa: PLC0415
    from mround.planner.sensitivity import (  # noqa: PLC0415
        SENSITIVITY_SAMPLES,
        SENSITIVITY_SEQ_LEN,
        mixed_budget_bits,
        score_layer_options,
    )
    from mround.planner.sensitivity import (  # noqa: PLC0415
        scoring_batch_size as derived_batch,
    )

    widths = sorted({int(width) for width in candidate_bits})
    if not widths:
        msg = "mixed precision needs at least one candidate width"
        raise SchemeError(msg)
    for width in widths:
        # Constructing the scheme is the validation: unsupported widths and
        # illegal combinations are refused with the scheme's own message.
        dataclasses.replace(base, bits=width)
    if not widths[0] <= average_bits <= widths[-1]:
        msg = (
            f"average_bits={average_bits} is outside the achievable range "
            f"[{widths[0]}, {widths[-1]}] for candidate_bits={widths}"
        )
        raise SchemeError(msg)

    samples = SENSITIVITY_SAMPLES if scoring_samples is None else int(scoring_samples)
    sequence = SENSITIVITY_SEQ_LEN if scoring_seq_len is None else int(scoring_seq_len)
    batch = (
        derived_batch(samples, sequence)
        if scoring_batch_size is None
        else max(1, min(samples, int(scoring_batch_size)))
    )
    scoring_set = build_calibration_set(
        bundle.tokenizer,
        source=corpus,
        n_samples=samples,
        seq_len=sequence,
        batch_size=batch,
        seed=DEFAULT_SEED if seed is None else seed,
    )
    passes = len(scoring_set.batches) * (1 if gradient_source == "widest" else len(widths))
    print(
        f"  scoring {samples} sequences of {sequence} tokens in "
        f"{len(scoring_set.batches)} batches of {batch}, {passes} backward "
        f"{'pass' if passes == 1 else 'passes'} at {sorted(widths)} bits",
        flush=True,
    )
    scored = score_widths(
        bundle.model,
        scoring_set.batches,
        widths,
        base,
        expected_blocks=expected_blocks,
        gradient_source=gradient_source,
        progress=_log_scoring,
    )

    dtype_bits = 16 if str(bundle.dtype).endswith(("float16", "bfloat16")) else 32
    options = score_layer_options(scored.scores, scored.shapes, base, dtype_bits=dtype_bits)
    budget = mixed_budget_bits(scored.shapes, average_bits, base.group_size, dtype_bits=dtype_bits)
    allocation = allocate_bits(options, budget_bits=budget)

    overrides = {
        path: dataclasses.replace(base, bits=width) for path, width in allocation.by_layer.items()
    }
    elements = sum(out * inner for out, inner in scored.shapes.values())
    metadata = sum(
        2 * out * base.groups_per_row(inner) * dtype_bits for out, inner in scored.shapes.values()
    )
    achieved = (allocation.total_cost_bits - metadata) / elements
    counts = {
        width: sum(1 for chosen in allocation.by_layer.values() if chosen == width)
        for width in widths
    }
    # Nearest supported width, ties toward MORE bits: floor(average + 0.5),
    # clamped into the widths the packing supports. The tie direction is
    # D-034's measurement: ties-down put a tied output head at 2 bits at the
    # exact midpoint and cost five times the perplexity. Overridable either
    # way.
    if remainder_bits is None:
        remainder = min(8, max(2, math.floor(average_bits + 0.5)))
    else:
        remainder = int(remainder_bits)
        dataclasses.replace(base, bits=remainder)
    summary = ", ".join(f"{count} layers at {width} bits" for width, count in counts.items())
    info: dict[str, Any] = {
        "average_bits": average_bits,
        "candidate_bits": widths,
        "achieved_code_bits": achieved,
        "layers_by_width": {str(width): count for width, count in counts.items()},
        "remainder_bits": remainder,
        "budget_bits": budget,
        "allocated_cost_bits": allocation.total_cost_bits,
        "predicted_loss": allocation.predicted_loss,
        "scoring_seconds": round(scored.seconds, 1),
        "scoring_samples": samples,
        "scoring_seq_len": sequence,
        # Derived from the two above unless the caller overrode it, and
        # recorded because it is the divisor that makes predicted_loss
        # comparable: scores sum over batches, so a run at a different
        # batching is on a different scale even though its allocation is
        # unaffected.
        "scoring_batch_size": batch,
        "gradient_source": gradient_source,
        # The full per-layer map, because the first measured allocation was
        # only diagnosable by digging it back out of the checkpoint config.
        "assignments": {path: int(width) for path, width in sorted(allocation.by_layer.items())},
        "describe": (
            f"planned {summary}; {achieved:.3f} code bits per weight against a "
            f"target of {average_bits:g}, remainder at {remainder} bits, "
            f"scored {samples}x{sequence} in {scored.seconds:.1f}s with "
            f"gradients at "
            f"{'the widest width' if gradient_source == 'widest' else 'each own width'}"
        ),
    }
    return overrides, info


def _round_to_nearest_layers(
    bundle: ModelBundle, scheme: QuantScheme, *, skip: Collection[str] = ()
) -> Iterator[tuple[str, QuantizedLayer | None, str]]:
    """Quantize every eligible module with nothing learned.

    Yields ``(path, layer, reason)`` where exactly one of ``layer`` and
    ``reason`` is meaningful. Reporting the skips rather than counting them is
    the point: "this model is 4-bit" is false in a way nobody can see if a large
    tensor quietly stayed dense.
    """
    import mlx.core as mx  # noqa: PLC0415

    from mround.core.quantizer import quantize_codes  # noqa: PLC0415
    from mround.formats.mlx_export import QuantizedLayer  # noqa: PLC0415
    from mround.pipeline.blocks import iter_quantizable_modules  # noqa: PLC0415

    for path, module, eligible in iter_quantizable_modules(
        bundle.model, group_size=scheme.group_size
    ):
        if path in skip:
            continue
        if not eligible:
            reason = (
                f"last dimension {module.weight.shape[-1]} is not a multiple of {scheme.group_size}"
            )
            yield path, None, reason
            continue
        # float32 for the forward, whatever the model stores. DOCUMENTATION 5.1.
        widened = module.weight.astype(mx.float32)
        init_scale = None
        if scheme.scale_init is ScaleInit.SEARCHED:
            # No importance here: these tensors sit outside the block loop, so
            # no activation stream is cached for them, and an unweighted search
            # is honest where an invented weighting would not be. The search
            # itself still applies, so a searched model is searched everywhere.
            from mround.core.scale_search import search_scales  # noqa: PLC0415

            init_scale = search_scales(widened, scheme)
        codes, scale, zero_point = quantize_codes(widened, None, scheme, init_scale=init_scale)
        yield path, QuantizedLayer(codes, scale, zero_point, scheme), ""


def _write_checkpoint(
    bundle: ModelBundle, destination: Path, quantized: Mapping[str, QuantizedLayer]
) -> dict[str, Any]:
    """Write the checkpoint and its tokenizer, and return what stayed dense."""
    from mlx.utils import tree_flatten  # noqa: PLC0415

    from mround.formats.mlx_export import export_mlx  # noqa: PLC0415

    replaced = {f"{path}.weight" for path in quantized}
    dense = {
        name: array
        for name, array in tree_flatten(bundle.model.parameters())
        if name not in replaced
    }
    export_mlx(
        destination,
        dense_weights=dense,
        quantized=quantized,
        config=bundle.config,
        dtype=str(bundle.dtype).removeprefix("mlx.core."),
    )
    # The tokenizer wrapper forwards this to the Hugging Face tokenizer beneath
    # it, which is what makes the checkpoint self-contained.
    bundle.tokenizer.save_pretrained(destination)  # type: ignore[attr-defined]
    return dense
