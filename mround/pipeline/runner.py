# Copyright 2026 The MRound Authors
# SPDX-License-Identifier: Apache-2.0
"""Block-by-block quantization orchestration.

Holds the loop that walks a model one transformer block at a time: capture the
block's inputs and reference outputs, wrap its linear layers, optimize the
learned rounding, commit the quantized weights, release everything, move on.

**Two activation streams run at once**, which is Q-001's answer and not an
accident of implementation. The reconstruction *target* for block N is what the
original block produces from the original model's activations, so the target
chain never touches a quantized weight. The *input* fed to block N during tuning
comes from the already quantized blocks before it, so each block is tuned against
the errors it will actually receive rather than against a clean signal it will
never see. Two streams therefore propagate side by side, which is roughly twice
the activation memory of one, and that cost is the reason this module reports
peak memory per block rather than leaving it to be discovered.

**The arithmetic runs in float32 and the streams are stored in it too.** The loss
here is a difference of two block outputs with nonlinearities between them, so
unlike the single-layer loss it cannot be rewritten to avoid the subtraction (see
MEMORY.md D-014). Storing either side at the model's own width would put a
rounding error of a few parts in a thousand on one side of a difference whose
whole magnitude is a few parts in a hundred, which is a noise floor larger than
the signal being optimized. That choice costs memory and is measured rather than
assumed.

**Memory discipline is this module's responsibility.** Peak usage is governed by
the largest single block plus its cached activations rather than by the whole
model, and that property is why a Mac can quantize models a discrete GPU cannot
hold. Anything added here that keeps a per-block quantity alive across blocks
breaks it silently.

So is evaluation placement. MLX evaluates lazily, and both failure modes are
reachable from a loop like this one: reading a scalar loss every step forces a
synchronization that can dominate runtime, while never evaluating lets the graph
grow until memory is exhausted. This loop evaluates the loss and every learnable
parameter together, once per step. See DOCUMENTATION.md section 7.
"""

from __future__ import annotations

import dataclasses
import functools
import time
from typing import TYPE_CHECKING, Any

import mlx.core as mx

from mround.core.losses import outlier_suppressed_loss, reconstruction_loss
from mround.core.quantizer import fake_quantize, init_params, project_params, quantize_codes
from mround.core.scale_search import search_scales
from mround.core.signsgd import LinearDecay, SignSGD
from mround.exceptions import ArchitectureError
from mround.pipeline.blocks import (
    ActivationCache,
    apply_block,
    block_output,
    capture_block_inputs,
    discover_blocks,
    iter_quantizable_linears,
)
from mround.pipeline.device import peak_memory
from mround.schemes import DEFAULT_EPS_FP32, QuantScheme, ScaleInit, TuningConfig

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from mlx import nn

    from mround.formats.mlx_export import QuantizedLayer
    from mround.pipeline.blocks import BlockRef

__all__ = ["BlockResult", "QuantizationRunner", "RunSummary", "block_recipe"]

Params = dict[str, mx.array]


@dataclasses.dataclass(frozen=True, slots=True)
class _Tuned:
    """What one block's optimization produced, before it is packaged."""

    initial_loss: float
    final_loss: float
    generalization_gap: float | None
    params: dict[str, Params]
    reverted: bool


@dataclasses.dataclass(frozen=True, slots=True)
class BlockResult:
    """What one block's tuning produced.

    Attributes:
        index: Block position.
        name: Dotted path from the model root.
        initial_loss: Reconstruction loss before tuning, which is what
            round-to-nearest achieves.
        final_loss: Reconstruction loss after tuning.
        iterations: Optimization steps actually run.
        seconds: Wall-clock time.
        peak_memory_bytes: Peak MLX allocation observed up to the end of this
            block.
        tuned: Layer names within the block that were tuned.
        skipped: Layer names left dense, with the reason.
        reverted: Whether tuning made this block worse and round-to-nearest was
            kept instead. Counting these is how a recipe that is quietly not
            working announces itself: a run where most blocks revert has a
            learning rate or an objective problem, not a marginal result.
        generalization_gap: How much better tuning did on its own batches than
            on held-out ones, as a difference of fractional improvements.
            ``None`` when nothing was held out, in which case overfitting is
            undetectable rather than absent. Positive means the block learned
            something that does not carry.
        search_seconds: Wall clock spent collecting importance and running the
            v2 scale search, zero when the scheme did not ask for it. Recorded
            because Q-004's kernel question needs this number.
    """

    index: int
    name: str
    initial_loss: float
    final_loss: float
    iterations: int
    seconds: float
    peak_memory_bytes: int
    tuned: tuple[str, ...] = ()
    skipped: tuple[tuple[str, str], ...] = ()
    reverted: bool = False
    generalization_gap: float | None = None
    search_seconds: float = 0.0

    @property
    def improvement(self) -> float:
        """Fractional loss reduction against round-to-nearest.

        The honest per-block measure of whether learned rounding earned its
        cost. Exactly zero means the block reverted, which is the floor: tuning
        cannot make a block worse than the baseline, it can only fail to help.
        """
        if self.initial_loss <= 0.0:
            return 0.0
        return 1.0 - (self.final_loss / self.initial_loss)

    def describe(self) -> str:
        """One line for a progress log."""
        note = "  reverted to round-to-nearest" if self.reverted else ""
        if self.generalization_gap is not None:
            note = f"  gap {self.generalization_gap:+.1%}{note}"
        if self.search_seconds:
            note = f"  search {self.search_seconds:.1f}s{note}"
        return (
            f"block {self.index:>3} {self.name:<24} "
            f"loss {self.initial_loss:.4e} to {self.final_loss:.4e} "
            f"({self.improvement:+.1%}) in {self.seconds:.1f}s{note}"
        )


@dataclasses.dataclass(frozen=True, slots=True)
class RunSummary:
    """The outcome of quantizing an entire model.

    Attributes:
        blocks: Per-block results, in execution order.
        total_seconds: Wall-clock time for the whole run.
        peak_memory_bytes: Peak across the run, which is the figure that
            determines the minimum machine for a given model.
        scheme_by_layer: The scheme each layer was quantized under, which is not
            uniform when mixed precision is in play.
    """

    blocks: list[BlockResult]
    total_seconds: float
    peak_memory_bytes: int
    scheme_by_layer: dict[str, QuantScheme]

    @property
    def mean_improvement(self) -> float:
        """Average fractional loss reduction across blocks.

        Weighted equally per block rather than by loss magnitude. A block whose
        loss is large contributes more to the model's error, but it is also the
        one whose improvement is easiest to achieve, and averaging by magnitude
        would let one bad block speak for the run.
        """
        if not self.blocks:
            return 0.0
        return sum(block.improvement for block in self.blocks) / len(self.blocks)

    @property
    def reverted(self) -> int:
        """Blocks where tuning did not beat round-to-nearest and was discarded."""
        return sum(1 for block in self.blocks if block.reverted)

    @property
    def mean_generalization_gap(self) -> float | None:
        """How much of what tuning achieved did not carry to unseen data.

        ``None`` when nothing was held out, which means undetectable rather than
        absent. A large positive value is the clearest signal available that the
        recipe is fitting the calibration set: it has been measured at 31.6
        percent against a held-out improvement of 23.9 percent, meaning most of
        what the optimizer achieved was specific to the batches it saw.
        """
        gaps = [b.generalization_gap for b in self.blocks if b.generalization_gap is not None]
        return sum(gaps) / len(gaps) if gaps else None

    def describe(self) -> str:
        """One line summarizing what the run achieved."""
        note = f", {self.reverted} reverted" if self.reverted else ""
        gap = self.mean_generalization_gap
        if gap is not None:
            note = f"{note}, mean gap {gap:+.1%}"
        return (
            f"{len(self.blocks)} blocks tuned in {self.total_seconds:.1f}s, "
            f"mean loss reduction {self.mean_improvement:+.1%}{note}, "
            f"peak memory {self.peak_memory_bytes / 1e9:.2f} GB"
        )


@functools.cache
def _tunable_type() -> type:
    """Build the fake-quantized linear class on first use.

    Deferred for the same reason the recording stand-in in ``blocks.py`` is: it
    subclasses an MLX type, and building it at import time would make this module
    unimportable without MLX. Cached because a model has as many blocks as it has
    layers and the class is identical for all of them.
    """
    from mlx import nn as nn_  # noqa: PLC0415

    class _TunableLinear(nn_.Module):  # type: ignore[misc]
        """A linear layer whose weight is fake-quantized by learnable parameters.

        Stands in for the real layer during tuning so that the block can be run
        normally and differentiated end to end. The weight it holds is the
        original one, widened to float32 and never updated: what is learned is
        the rounding, not the weight.

        The learnable quantities are injected before each forward rather than
        held as module parameters. That is what MLX's own ``nn.value_and_grad``
        does, and it keeps the differentiated tree exactly the three tensors per
        layer that are being optimized rather than everything the block happens
        to contain.
        """

        def __init__(self, linear: Any, scheme: QuantScheme, eps: float) -> None:
            super().__init__()
            self.weight = linear.weight.astype(mx.float32)
            self.bias = getattr(linear, "bias", None)
            self.scheme = scheme
            self.eps = eps
            self.params = init_params(self.weight, scheme)
            self.init_scale: mx.array | None = None
            self.collecting = False
            self.importance: mx.array | None = None

        def __call__(self, x: mx.array) -> mx.array:
            """Apply the layer, fake-quantized, or collect statistics through it.

            Collection mode is the hook-free equivalent of the reference's
            imatrix forward hooks: the layer accumulates the summed squared
            activation per input channel while applying its original,
            unquantized weight, which is the reference's stats forward. The
            inputs it sees are whatever stream the runner feeds, so with
            quantized inputs enabled the importance comes from the quantized
            stream from block 1 onward and the original stream at block 0,
            which is the reference's documented collection order.
            """
            if self.collecting:
                flat = x.reshape(-1, x.shape[-1]).astype(mx.float32)
                seen = mx.sum(mx.square(flat), axis=0)
                self.importance = seen if self.importance is None else self.importance + seen
                y = x @ self.weight.T.astype(x.dtype)
                return y if self.bias is None else y + self.bias
            weight = fake_quantize(
                self.weight, self.params, self.scheme, eps=self.eps, init_scale=self.init_scale
            )
            y = x @ weight.T
            return y if self.bias is None else y + self.bias

    return _TunableLinear


def block_recipe(
    tuning: TuningConfig,
    base_bits: int,
    widths: Mapping[str, int],
    *,
    per_layer: bool,
) -> tuple[bool, dict[str, float]]:
    """The outlier rule and the per layer learning rates for one block.

    Two readings of DOCUMENTATION.md section 5.4 under a mixed plan, and the
    difference between them is MEMORY.md D-043. ``per_layer`` off is what every
    published mixed run was produced with: the block loop resolves both the
    learning rate constant and the outlier rule once from the run's base
    ``bits``, so a 2 bit layer inside a 4 bit base run is tuned at ``1.0/iters``
    with no outlier mask, where the uniform 2 bit arm of the same comparison
    uses ``2.0/iters`` with the mask on. ``per_layer`` on reads the section the
    way it is written, giving every layer the rate its own assigned width
    implies.

    The outlier rule cannot be per layer in the same way, because the objective
    is one loss over the whole block rather than one loss per layer. It is
    resolved from the narrowest width present, since suppression exists to stop
    a handful of large errors dominating a sign gradient and the narrowest layer
    is what produces them.

    Returns:
        Whether to suppress outliers, and the initial learning rate per layer.
    """
    if not per_layer:
        rate = tuning.resolved_lr(base_bits)
        return tuning.resolved_suppress_outliers(base_bits), dict.fromkeys(widths, rate)
    narrowest = min(widths.values(), default=base_bits)
    return tuning.resolved_suppress_outliers(narrowest), {
        name: tuning.resolved_lr(bits) for name, bits in widths.items()
    }


class QuantizationRunner:
    """Walks a model block by block, quantizing each in turn."""

    def __init__(
        self,
        scheme: QuantScheme,
        tuning: TuningConfig,
        *,
        progress: Callable[[BlockResult], None] | None = None,
        eps: float = DEFAULT_EPS_FP32,
        quantized_inputs: bool = True,
        holdout_batches: int = 0,
        per_layer_recipe: bool = False,
    ) -> None:
        """Create a runner.

        Args:
            scheme: Default representation. Mixed precision overrides it per
                layer.
            tuning: How the learned rounding is optimized.
            progress: Called after each block completes. Quantization takes long
                enough that silence is a usability problem.
            eps: Scale epsilon.
            quantized_inputs: Feed each block the activations produced by the
                already quantized blocks before it, rather than the original
                model's. On by default, which is the reference's choice for
                learned rounding and MEMORY.md Q-001's answer. Turning it off
                halves activation memory and tunes every block against a signal
                the deployed model never produces.
            holdout_batches: Trailing calibration batches excluded from tuning
                and used only to decide whether tuning helped. Zero, the
                reference's behaviour, measures that decision on the data being
                fit, which cannot detect overfitting. It is off by default
                because turning it on both changes the method and spends
                calibration data, and whether that trade pays has not been
                measured.
            per_layer_recipe: Under a mixed plan, resolve the learning rate and
                the outlier rule from each layer's assigned width rather than
                once from ``scheme.bits``. Off by default, which is what every
                published mixed measurement was produced with. See
                :func:`block_recipe` and MEMORY.md D-043.
        """
        self.scheme = scheme
        self.tuning = tuning
        self.progress = progress
        self.eps = eps
        self.quantized_inputs = quantized_inputs
        self.holdout_batches = holdout_batches
        self.per_layer_recipe = per_layer_recipe

    def _scheme_for(self, path: str, overrides: Mapping[str, QuantScheme] | None) -> QuantScheme:
        """The scheme for one layer, honoring a planner override."""
        if overrides is None:
            return self.scheme
        return overrides.get(path, self.scheme)

    def run(
        self,
        model: nn.Module,
        calibration_batches: Sequence[mx.array],
        *,
        scheme_by_layer: Mapping[str, QuantScheme] | None = None,
        expected_blocks: int | None = None,
    ) -> tuple[RunSummary, dict[str, QuantizedLayer]]:
        """Quantize every block of ``model`` in place.

        Args:
            model: A loaded MLX model. Modified in place; the caller keeps no
                usable full-precision copy afterwards.
            calibration_batches: Tokenized calibration data.
            scheme_by_layer: Per-layer overrides from the planner, keyed by full
                dotted path. ``None`` applies the default scheme uniformly.
            expected_blocks: Block count from the model configuration, checked
                against what discovery finds.

        Returns:
            The run summary, and the quantized layers ready for export, keyed by
            full dotted path.

        Raises:
            ArchitectureError: If the block stack cannot be located, or a block
                behaves in a way the capture contract does not cover.
        """
        started = time.perf_counter()
        blocks = discover_blocks(model, expected=expected_blocks)

        # Captured once at the entrance. Propagation down the stack was verified
        # exact on hardware, so this is one prefix pass rather than one per
        # block. See MEMORY.md D-023.
        entrance = capture_block_inputs(model, blocks[0], calibration_batches, with_outputs=False)
        reference_stream = [hidden.astype(mx.float32) for hidden in entrance.inputs]
        mx.eval(reference_stream)
        entrance.inputs = []

        quantized_stream = list(reference_stream) if self.quantized_inputs else None

        results: list[BlockResult] = []
        exported: dict[str, QuantizedLayer] = {}
        schemes: dict[str, QuantScheme] = {}

        for block in blocks:
            # Computed before the block is touched, so the target chain is the
            # original model's activations all the way down. Q-001.
            targets = apply_block(
                block.module,
                ActivationCache(
                    inputs=reference_stream,
                    outputs=[],
                    args=entrance.args,
                    kwargs=entrance.kwargs,
                ),
            )

            cache = ActivationCache(
                inputs=quantized_stream if quantized_stream is not None else reference_stream,
                outputs=targets,
                args=entrance.args,
                kwargs=entrance.kwargs,
            )
            result, layers = self.quantize_block(
                block, cache, scheme_by_layer=scheme_by_layer, path_prefix=block.name
            )
            results.append(result)
            exported.update(layers)
            for path in layers:
                schemes[path] = self._scheme_for(path, scheme_by_layer)
            if self.progress is not None:
                self.progress(result)

            # Advance both streams, then drop everything this block held. The
            # target the loop just used is by construction the next reference
            # input, so it is moved rather than recomputed.
            if quantized_stream is not None:
                quantized_stream = apply_block(block.module, cache)
            reference_stream = targets
            cache.clear()

        total = time.perf_counter() - started
        return (
            RunSummary(
                blocks=results,
                total_seconds=total,
                peak_memory_bytes=peak_memory(),
                scheme_by_layer=schemes,
            ),
            exported,
        )

    def quantize_block(
        self,
        block: BlockRef,
        cache: ActivationCache,
        *,
        scheme_by_layer: Mapping[str, QuantScheme] | None = None,
        path_prefix: str = "",
    ) -> tuple[BlockResult, dict[str, QuantizedLayer]]:
        """Tune and quantize a single block, in place.

        Exposed separately because it is the unit of work a parity test wants:
        one block, fixed inputs, comparable outputs.

        Args:
            block: The block to quantize.
            cache: Inputs the block should be tuned against, the full-precision
                targets to reconstruct, and the auxiliary arguments each call
                needs.
            scheme_by_layer: Per-layer overrides, keyed by full dotted path.
            path_prefix: Prepended to layer names to form those paths.

        Returns:
            What tuning achieved, and the quantized layers keyed by full path.

        Raises:
            ArchitectureError: If nothing in the block can be quantized at this
                group size, which would otherwise leave a block silently dense.
        """
        from mlx.utils import tree_unflatten  # noqa: PLC0415

        started = time.perf_counter()
        tunable_type = _tunable_type()

        originals: dict[str, Any] = {}
        tunables: dict[str, Any] = {}
        schemes: dict[str, QuantScheme] = {}
        skipped: list[tuple[str, str]] = []

        for name, linear in iter_quantizable_linears(block.module):
            path = f"{path_prefix}.{name}" if path_prefix else name
            scheme = self._scheme_for(path, scheme_by_layer)
            if linear.weight.shape[-1] % scheme.group_size:
                skipped.append(
                    (
                        name,
                        f"last dimension {linear.weight.shape[-1]} is not a "
                        f"multiple of {scheme.group_size}",
                    )
                )
                continue
            originals[name] = linear
            tunables[name] = tunable_type(linear, scheme, self.eps)
            schemes[name] = scheme

        if not tunables:
            reasons = "; ".join(f"{name}: {why}" for name, why in skipped)
            msg = (
                f"nothing in {block.name!r} can be quantized at this group size, "
                f"so the block would be left at full precision without saying so. "
                f"{reasons}"
            )
            raise ArchitectureError(msg)

        try:
            block.module.update_modules(tree_unflatten(list(tunables.items())))
            search_seconds = self._search_scales(block, cache, tunables)
            outcome = self._tune(block, cache, tunables)
        finally:
            # Restored before anything else touches the block, including on the
            # way out of a failure. A stand-in left installed would be found by
            # the next block's propagation and quietly used as a real layer.
            block.module.update_modules(tree_unflatten(list(originals.items())))

        exported: dict[str, QuantizedLayer] = {}
        for name, linear in originals.items():
            path = f"{path_prefix}.{name}" if path_prefix else name
            exported[path] = self._commit(
                linear, tunables[name], outcome.params[name], schemes[name]
            )

        return (
            BlockResult(
                index=block.index,
                name=block.name,
                initial_loss=outcome.initial_loss,
                final_loss=outcome.final_loss,
                iterations=self.tuning.iters,
                seconds=time.perf_counter() - started,
                peak_memory_bytes=peak_memory(),
                tuned=tuple(sorted(tunables)),
                skipped=tuple(skipped),
                reverted=outcome.reverted,
                generalization_gap=outcome.generalization_gap,
                search_seconds=search_seconds,
            ),
            exported,
        )

    def _search_scales(
        self, block: BlockRef, cache: ActivationCache, tunables: Mapping[str, Any]
    ) -> float:
        """Run the v2 scale search for every layer whose scheme asks for it.

        One extra forward per calibration batch collects the importance, which
        is the summed squared activation each input channel actually sees, then
        each layer's grid search runs once on the GPU. The cost is recorded
        because Q-004 asks whether this stage ever earns a custom kernel, and
        that question needs numbers, not impressions.

        Returns:
            Wall-clock seconds spent, zero when nothing searched.
        """
        searching = {
            name: layer
            for name, layer in tunables.items()
            if layer.scheme.scale_init is ScaleInit.SEARCHED
        }
        if not searching:
            return 0.0

        started = time.perf_counter()
        for layer in tunables.values():
            layer.collecting = True
        try:
            for hidden, rest, keywords in zip(cache.inputs, cache.args, cache.kwargs, strict=True):
                produced = block_output(block.module, hidden.astype(mx.float32), rest, keywords)
                # Evaluated per batch for the same reason capture does it: the
                # graph reaching back through the whole block must not
                # accumulate across the corpus.
                mx.eval(
                    produced,
                    *[
                        layer.importance
                        for layer in tunables.values()
                        if layer.importance is not None
                    ],
                )
        finally:
            for layer in tunables.values():
                layer.collecting = False

        for layer in searching.values():
            layer.init_scale = search_scales(
                layer.weight, layer.scheme, importance=layer.importance
            )
        return time.perf_counter() - started

    def _commit(
        self, linear: Any, tunable: Any, params: Params, scheme: QuantScheme
    ) -> QuantizedLayer:
        """Write the learned quantization into the real layer and package it.

        The layer's weight is replaced by its own reconstruction so that the
        propagated activation stream is what the quantized model produces, and
        the codes are packaged separately for export. The two agree by
        construction: they are the same rounding, expressed once as a float and
        once as an integer with its scale.
        """
        from mround.formats.mlx_export import QuantizedLayer as Layer  # noqa: PLC0415

        codes, scale, zero_point = quantize_codes(
            tunable.weight, params, scheme, eps=self.eps, init_scale=tunable.init_scale
        )
        reconstructed = fake_quantize(
            tunable.weight, params, scheme, eps=self.eps, init_scale=tunable.init_scale
        )
        linear.weight = reconstructed.astype(linear.weight.dtype)
        mx.eval(linear.weight, codes, scale, zero_point)
        return Layer(codes, scale, zero_point, scheme)

    def _tune(self, block: BlockRef, cache: ActivationCache, tunables: Mapping[str, Any]) -> _Tuned:
        """Run signed gradient descent on the block's rounding parameters.

        **The parameters kept are the last ones, not the best-scoring ones.**
        The reference keeps whichever step scored lowest, and that comparison is
        between losses measured on different calibration batches, which are not
        comparable: a step can win by drawing an easy batch. Signed gradient
        descent with a rate decaying linearly to zero is built to land somewhere
        rather than to wander, so the last step is the intended endpoint and
        needs no selection.

        What replaces the safety net is coarser and sound. The loss is measured
        at both ends and a block that got worse is discarded in favour of
        round-to-nearest. Where those two measurements are taken is the thing
        that matters, and it is what ``holdout_batches`` controls.

        **Measuring on the batches being tuned cannot detect overfitting**, and
        overfitting is not hypothetical here: five times the steps on a fixed
        corpus lowered the calibration loss of all thirty blocks and made the
        model four times worse on held-out text. With a holdout configured, the
        last batches are excluded from tuning entirely and both endpoints are
        measured on them, so the comparison is against data the rounding has
        never seen and the revert rule can see a block that has memorized rather
        than learned.

        Returns the loss before tuning, the loss after, the parameters to
        commit, and whether the fallback fired.
        """
        suppress, rates = block_recipe(
            self.tuning,
            self.scheme.bits,
            {name: layer.scheme.bits for name, layer in tunables.items()},
            per_layer=self.per_layer_recipe,
        )
        loss_fn = outlier_suppressed_loss if suppress else reconstruction_loss
        n_batches = len(cache.inputs)

        # At least one batch always stays available to tune on, so a holdout
        # larger than the corpus degrades rather than divides by zero.
        held = min(self.holdout_batches, n_batches - 1)
        tuning_batches = list(range(n_batches - held))
        scoring_batches = list(range(n_batches - held, n_batches)) if held else tuning_batches

        def objective_for(index: int) -> Callable[[Mapping[str, Params]], mx.array]:
            """Build the loss for one calibration batch.

            The batch index is closed over rather than passed as an argument.
            ``mx.value_and_grad`` differentiates a function of arrays and trees
            of arrays, and a Python integer among its arguments is neither.
            """

            def objective(params: Mapping[str, Params]) -> mx.array:
                # Injecting the parameters into the modules is how the block can
                # be run normally and still be differentiated: the block calls
                # its layers, and its layers read what was just put there. MLX's
                # own nn.value_and_grad works the same way.
                for name, layer in tunables.items():
                    layer.params = params[name]
                predicted = block_output(
                    block.module,
                    cache.inputs[index].astype(mx.float32),
                    cache.args[index],
                    cache.kwargs[index],
                )
                return loss_fn(predicted, cache.outputs[index])

            return objective

        def mean_loss(params: Mapping[str, Params], over: list[int]) -> float:
            """The loss averaged over a set of batches."""
            mean = sum(objective_for(i)(params) for i in over) / len(over)
            mx.eval(mean)
            return float(mean)

        start = {name: dict(layer.params) for name, layer in tunables.items()}
        optimizers = {
            name: SignSGD(LinearDecay(rates[name], self.tuning.iters)) for name in tunables
        }

        initial_loss = mean_loss(start, scoring_batches)
        initial_tuning = mean_loss(start, tuning_batches) if held else initial_loss
        params = start

        for step in range(self.tuning.iters):
            # Cycled rather than sampled. Every batch is then used the same
            # number of times, which a random draw only approaches, and the run
            # is reproducible without carrying a second seed.
            batch = tuning_batches[step % len(tuning_batches)]
            loss, grads = mx.value_and_grad(objective_for(batch))(params)

            params = {
                name: project_params(
                    optimizers[name].apply(params[name], grads[name]), layer.scheme
                )
                for name, layer in tunables.items()
            }

            # One synchronization per step, covering the loss and every new
            # parameter together. Evaluating less often lets the graph grow
            # across steps; evaluating the loss separately pays two round trips.
            mx.eval(loss, *[value for group in params.values() for value in group.values()])

        final_loss = mean_loss(params, scoring_batches)
        final_tuning = mean_loss(params, tuning_batches) if held else final_loss

        # The gap between what tuning achieved on its own data and what it
        # achieved on data it never saw. Positive means the block learned
        # something that does not carry, which no amount of further optimization
        # will fix and which the calibration loss alone cannot show.
        gap = None
        if held:
            gap = (1.0 - final_tuning / initial_tuning) - (1.0 - final_loss / initial_loss)

        if final_loss >= initial_loss:
            return _Tuned(initial_loss, initial_loss, gap, start, reverted=True)
        return _Tuned(initial_loss, final_loss, gap, params, reverted=False)
