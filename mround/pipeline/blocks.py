# Copyright 2026 The MRound Authors
# SPDX-License-Identifier: Apache-2.0
"""Transformer block discovery and activation capture.

MLX has no forward-hook mechanism, so nothing here intercepts a running model.
Blocks are located structurally and then run explicitly to the boundary whose
activations are needed. That is more code than hooks would be, and it is also
easier to follow when something goes wrong.

**Discovery is structural, not a table of architecture names.** A transformer
block stack is the one thing every decoder has in common: a list of modules of
one type, indexed from zero with no gaps, each containing linear layers. Looking
for that shape covers architectures nobody has written a rule for, and a table
of names covers exactly the models someone remembered to add. Where two
candidate stacks are equally plausible this refuses rather than picking, because
quantizing the wrong stack produces a checkpoint that loads and generates
nonsense, which is the most expensive failure mode available.

**Capture works by standing in for a block.** A recording stand-in is swapped in
where the block sits, the model is run normally, and the stand-in records exactly
the arguments the block was handed before unwinding the forward. This is
deliberately incurious about what those arguments mean: the attention mask, the
rotary position information, and the cache are whatever the architecture decided
they are, and passing them back unchanged is what lets one code path serve every
model instead of one per family. The stand-in also answers attribute lookups on
behalf of the block it displaced, because a model's own loop is entitled to read
state off a block and not merely to call it. It is removed in a ``finally``, so a
failure mid-capture leaves the model as it was found.

**One block's activations are resident at a time.** The alternative, recording
every block in a single pass, is one forward instead of many and costs the depth
of the model in memory. On anything past a few billion parameters that is the
difference between running and not, so blocks are captured once at the entrance
and then propagated: the output of block N becomes the input of block N+1.

Reusing the auxiliary arguments down the stack is the part of that which is an
assumption rather than a guarantee. Sliding-window architectures build two masks
and choose between them per layer, so a model can hand different blocks different
masks. `scripts/probe_blocks.py` checks it at both ends of the stack rather than
between one adjacent pair, since an alternating pattern would agree between
blocks zero and one about half the time by luck.

Specification: DOCUMENTATION.md section 3. The tuning loop that consumes all of
this lives in ``runner.py``.
"""

from __future__ import annotations

import dataclasses
import functools
from typing import TYPE_CHECKING, Any

from mround.exceptions import ArchitectureError

if TYPE_CHECKING:
    from collections.abc import Collection, Iterable, Iterator, Mapping

    import mlx.core as mx
    from mlx import nn

__all__ = [
    "ActivationCache",
    "BlockRef",
    "apply_block",
    "block_output",
    "capture_block_inputs",
    "discover_blocks",
    "iter_quantizable_linears",
    "iter_quantizable_modules",
]

# A stack of one is a module that happens to live in a list, not a block stack.
# Every real decoder has at least two.
_MIN_STACK = 2


@dataclasses.dataclass(frozen=True, slots=True)
class BlockRef:
    """A located transformer block.

    Attributes:
        index: Position in the stack, zero-based.
        name: Dotted path from the model root, for logging and checkpoints.
        module: The block itself.
    """

    index: int
    name: str
    module: nn.Module


@dataclasses.dataclass(slots=True)
class ActivationCache:
    """Cached inputs and reference outputs for one block.

    Holding these is the dominant memory cost of a quantization run after the
    weights themselves, which is why the runner clears the cache between blocks
    rather than keeping the model's worth.

    The hidden state is separated from everything else the block was called
    with. That split is the whole point: the hidden state is what changes from
    block to block and what the loss is computed against, while the mask, the
    position information, and the cache are shared down the stack and are passed
    back exactly as they arrived. Nothing here interprets them.

    Attributes:
        inputs: Hidden states entering the block, one per calibration batch.
        outputs: Corresponding full-precision outputs, the reconstruction
            target.
        args: Remaining positional arguments per batch, in order.
        kwargs: Keyword arguments per batch. Passed through unchanged.
    """

    inputs: list[mx.array]
    outputs: list[mx.array]
    args: list[tuple[Any, ...]]
    kwargs: list[dict[str, Any]]

    def __len__(self) -> int:
        """Number of cached batches."""
        return len(self.inputs)

    def clear(self) -> None:
        """Release every cached array.

        Dropping the references is all this can do. MLX keeps freed buffers in
        its own pool for reuse, so the process footprint does not fall
        immediately and that is not a leak.
        """
        self.inputs = []
        self.outputs = []
        self.args = []
        self.kwargs = []

    def nbytes(self) -> int:
        """Bytes held by the cached hidden states and targets.

        Only the hidden states are counted. The masks and position information
        are shared across the stack and are negligible beside a batch of
        activations.
        """
        return sum(int(a.nbytes) for a in (*self.inputs, *self.outputs))


class _CapturedCall(Exception):  # noqa: N818
    """Control flow, not a failure: a block's arguments leaving a forward pass.

    MLX has no forward hooks, so the only way to see what a block receives is to
    stand in for it. Once the arguments are recorded there is nothing sensible
    for the stand-in to return, since the shape of a block's output is exactly
    what it does not know, so it unwinds the forward rather than inventing a
    hidden state. Raised and caught within this module and never seen elsewhere.

    The recorded call is held under names that do not collide with
    ``BaseException.args``, which is a real attribute and would otherwise be
    quietly overwritten.
    """

    def __init__(self, call_args: tuple[Any, ...], call_kwargs: dict[str, Any]) -> None:
        super().__init__("calibration capture: control flow, not a failure")
        self.call_args = call_args
        self.call_kwargs = call_kwargs


@functools.cache
def _recorder_type() -> type:
    """Build the recording stand-in class on first use.

    Deferred so that this module imports on a machine without MLX, which is what
    keeps the discovery rules below testable in continuous integration. Cached
    because the class only needs building once.
    """
    from mlx import nn as nn_  # noqa: PLC0415

    class _Recorder(nn_.Module):  # type: ignore[misc]
        """Stands where a block sits, records the call, and unwinds the forward.

        It keeps the block it displaced and forwards attribute lookups to it.
        That is not tidiness. A model's own block loop reads attributes off a
        block *before* calling it, and ``mlx-lm``'s Llama loop is one of them::

            mask = swa_mask if layer.use_sliding else fa_mask
            h = layer(h, mask, cache[i])

        An empty stand-in fails on the first line, before it has recorded
        anything. Delegating is also the only version of this that stays generic:
        which attributes a loop reads is a per-architecture fact, and enumerating
        them would put this module back in the business of knowing about model
        families.
        """

        def __init__(self, wrapped: Any) -> None:
            super().__init__()
            self._wrapped = wrapped

        def __getattr__(self, key: str) -> Any:
            """Answer as the displaced block would.

            Reached only when normal lookup fails. Both lookups go through the
            dictionary protocol rather than attribute access, so neither can
            recurse back through here, and the second guards the window during
            ``__init__`` before there is a block to delegate to.
            """
            if key in self:
                return self[key]
            if "_wrapped" not in self:
                msg = (
                    f"{type(self).__name__} has no attribute {key!r} and no block to answer for it"
                )
                raise AttributeError(msg)
            return getattr(self["_wrapped"], key)

        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            """Record and abort. Never returns."""
            raise _CapturedCall(args, kwargs)

    return _Recorder


def _group_by_index(paths: Iterable[str]) -> dict[str, dict[int, str]]:
    """Group dotted module paths by their parent, keyed on a trailing integer.

    ``model.layers.0`` and ``model.layers.1`` group under ``model.layers``.
    Anything whose final segment is not an integer is not a list element and is
    skipped.
    """
    groups: dict[str, dict[int, str]] = {}
    for path in paths:
        parent, _, last = path.rpartition(".")
        if not last.isdigit():
            continue
        groups.setdefault(parent, {})[int(last)] = path
    return groups


def _is_stack(indices: Collection[int]) -> bool:
    """Whether these indices number a stack: at least two, from zero, no gaps."""
    return len(indices) >= _MIN_STACK and sorted(indices) == list(range(len(indices)))


def _select_stack(candidates: Mapping[str, int]) -> str:
    """Choose the outermost of several structurally valid block stacks.

    Depth breaks the tie because a mixture-of-experts model has a list of
    experts inside every block, and those lists are stacks by every structural
    test that matters. The outer one is the transformer.

    Args:
        candidates: Parent path to number of members.

    Returns:
        The winning parent path.

    Raises:
        ArchitectureError: If there are no candidates, or if two sit at the same
            depth. Two equally plausible stacks is not a case to guess at:
            quantizing the wrong one yields a checkpoint that loads cleanly and
            generates nonsense.
    """
    if not candidates:
        msg = (
            "no transformer block stack found. MRound looks for a list of "
            "modules of one type, indexed from zero with no gaps, each holding "
            "linear layers. A model that stores its blocks some other way needs "
            "explicit support rather than a looser rule, because a looser rule "
            "would match something else on a different model."
        )
        raise ArchitectureError(msg)

    depth = min(path.count(".") for path in candidates)
    outermost = sorted(path for path in candidates if path.count(".") == depth)
    if len(outermost) > 1:
        found = ", ".join(f"{path} ({candidates[path]} members)" for path in outermost)
        msg = (
            f"found {len(outermost)} equally plausible block stacks: {found}. "
            f"MRound will not guess between them, because quantizing the wrong "
            f"one produces a checkpoint that loads and generates nonsense. This "
            f"architecture needs explicit support."
        )
        raise ArchitectureError(msg)
    return outermost[0]


def discover_blocks(model: nn.Module, *, expected: int | None = None) -> list[BlockRef]:
    """Locate the transformer blocks in ``model``.

    Finds the block stack structurally, by looking for the repeated homogeneous
    sequence that characterizes a transformer, rather than by matching known
    architecture names. Architectures that need special handling are the
    exception and are registered explicitly.

    Args:
        model: A loaded MLX model.
        expected: Block count from the model configuration, usually
            ``num_hidden_layers``. Checked against what was found. This is the
            cheapest guard there is against having picked the wrong stack, and
            it costs one dictionary lookup at the call site.

    Returns:
        Blocks in execution order.

    Raises:
        ArchitectureError: If no block stack is found, if the structure is
            ambiguous, or if the count contradicts ``expected``. Guessing here
            would produce a model that quantizes without error and generates
            nonsense.
    """
    from mlx import nn as nn_  # noqa: PLC0415

    modules = dict(model.named_modules())
    viable: dict[str, dict[int, str]] = {}
    for parent, members in _group_by_index(modules).items():
        if not _is_stack(members.keys()):
            continue
        kinds = {type(modules[path]) for path in members.values()}
        if len(kinds) != 1:
            continue
        first = modules[members[0]]
        if not any(isinstance(m, nn_.Linear) for _, m in first.named_modules()):
            continue
        viable[parent] = members

    chosen = _select_stack({parent: len(members) for parent, members in viable.items()})
    members = viable[chosen]

    if expected is not None and len(members) != expected:
        msg = (
            f"found {len(members)} blocks at {chosen!r} but the configuration "
            f"says {expected}. Either the wrong stack was selected or the "
            f"configuration does not describe this checkpoint, and both are "
            f"reasons to stop rather than quantize part of a model."
        )
        raise ArchitectureError(msg)

    return [
        BlockRef(index=index, name=members[index], module=modules[members[index]])
        for index in range(len(members))
    ]


def _arrays_in(mx: Any, value: Any) -> Iterator[Any]:
    """Walk a captured argument, yielding the MLX arrays inside it.

    Block arguments are arrays, ``None``, or shallow containers of those: a
    cache is a list, and some architectures pass a small dictionary. Anything
    deeper than that is not something this code should be evaluating anyway.
    """
    if isinstance(value, mx.array):
        yield value
    elif isinstance(value, list | tuple):
        for item in value:
            yield from _arrays_in(mx, item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _arrays_in(mx, item)


def _hidden_state(mx: Any, value: Any, *, what: str) -> Any:
    """Extract the single array a block passes along, or refuse.

    Every ``mlx-lm`` decoder block takes one hidden state and returns one. A
    block that returns several is carrying state forward that this code would be
    dropping, and dropping it silently would corrupt the calibration in a way
    that only shows up as a slightly worse model.
    """
    if isinstance(value, mx.array):
        return value
    if isinstance(value, list | tuple) and len(value) == 1 and isinstance(value[0], mx.array):
        return value[0]
    kind = type(value).__name__
    msg = (
        f"{what} is a {kind}, not a single array. MRound assumes a block takes "
        f"one hidden state and returns one, which is true of every mlx-lm "
        f"decoder block. This architecture carries something else along, and "
        f"quantizing it would drop that silently."
    )
    raise ArchitectureError(msg)


def capture_block_inputs(
    model: nn.Module,
    block: BlockRef,
    batches: Iterable[mx.array],
    *,
    with_outputs: bool = True,
) -> ActivationCache:
    """Run the model to ``block`` and cache what it sees and what it produces.

    Works by replacing the block with a recording stand-in, running the model
    forward once per batch, and unwinding as soon as the stand-in is reached.
    Everything after the block is therefore never computed. The original block
    is restored before returning, including when a batch fails.

    Capturing block zero and then propagating with :func:`apply_block` costs one
    block-forward per block. Calling this for every block instead costs a
    prefix-forward per block, which is quadratic in depth. Use it for block
    zero, and for checking that the propagation is faithful.

    Args:
        model: A loaded MLX model.
        block: The block to capture around.
        batches: Tokenized calibration batches.
        with_outputs: Also run the block on what was captured, filling the
            reconstruction targets. Off when only the inputs are wanted, such as
            when the targets will come from a different copy of the model.

    Returns:
        Cached inputs, reference outputs, and the rest of each call.

    Raises:
        ArchitectureError: If the model never reaches the block, or hands it
            something other than a hidden state first.
    """
    import mlx.core as mx  # noqa: PLC0415
    from mlx.utils import tree_unflatten  # noqa: PLC0415

    # Built before the swap, so the restore below cannot depend on anything the
    # swap did.
    original = tree_unflatten([(block.name, block.module)])
    stand_in = tree_unflatten([(block.name, _recorder_type()(block.module))])

    cache = ActivationCache(inputs=[], outputs=[], args=[], kwargs=[])
    try:
        model.update_modules(stand_in)
        for batch in batches:
            try:
                model(batch)
            except _CapturedCall as captured:
                call_args, call_kwargs = captured.call_args, captured.call_kwargs
            else:
                msg = (
                    f"the forward pass completed without reaching {block.name!r}. "
                    f"The block was found by walking the module tree but is not "
                    f"on the path the model actually executes, so nothing here "
                    f"describes what it receives."
                )
                raise ArchitectureError(msg)

            if not call_args:
                msg = (
                    f"{block.name!r} was called with keyword arguments only. "
                    f"MRound identifies the hidden state as the first positional "
                    f"argument, which every mlx-lm decoder block uses."
                )
                raise ArchitectureError(msg)

            hidden = _hidden_state(mx, call_args[0], what=f"the first argument to {block.name!r}")
            rest = tuple(call_args[1:])
            keywords = dict(call_kwargs)

            # Materialize now. These are unevaluated graph nodes reaching back
            # through the whole prefix of the model, and leaving them lazy would
            # keep every intermediate of every batch alive at once.
            mx.eval(
                [
                    *_arrays_in(mx, hidden),
                    *_arrays_in(mx, rest),
                    *_arrays_in(mx, keywords),
                ]
            )
            cache.inputs.append(hidden)
            cache.args.append(rest)
            cache.kwargs.append(keywords)
    finally:
        model.update_modules(original)

    if with_outputs:
        cache.outputs = apply_block(block.module, cache)
    return cache


def apply_block(module: nn.Module, cache: ActivationCache) -> list[mx.array]:
    """Run ``module`` over every cached input, returning the hidden states.

    This is what propagation is made of. Running the full-precision block gives
    the reconstruction targets; running the quantized block over the same inputs
    gives what the next block will actually see.

    Args:
        module: A block, quantized or not.
        cache: Inputs and the rest of each call, from
            :func:`capture_block_inputs`.

    Returns:
        One output per cached batch, already evaluated.

    Raises:
        ArchitectureError: If the block returns something other than a single
            hidden state.
    """
    import mlx.core as mx  # noqa: PLC0415

    outputs: list[Any] = []
    for hidden, rest, keywords in zip(cache.inputs, cache.args, cache.kwargs, strict=True):
        produced = block_output(module, hidden, rest, keywords)
        # Per batch rather than at the end: a whole corpus of unevaluated block
        # outputs is the same memory problem the capture above avoids.
        mx.eval(produced)
        outputs.append(produced)
    return outputs


def block_output(
    module: nn.Module,
    hidden: mx.array,
    args: tuple[Any, ...] = (),
    kwargs: Mapping[str, Any] | None = None,
) -> mx.array:
    """Run one block on one hidden state and return the hidden state it produces.

    Separate from :func:`apply_block` because the tuning loop calls a block
    inside a differentiated function, one batch at a time, and must not evaluate
    anything while doing so.

    Args:
        module: A block, quantized or not.
        hidden: The hidden state entering it.
        args: The remaining positional arguments the block was captured with.
        kwargs: The keyword arguments it was captured with.

    Returns:
        The hidden state the block produces, unevaluated.

    Raises:
        ArchitectureError: If the block returns something other than a single
            hidden state.
    """
    import mlx.core as mx  # noqa: PLC0415

    produced = module(hidden, *args, **(kwargs or {}))
    return _hidden_state(mx, produced, what="what the block returned")


def iter_quantizable_linears(block: nn.Module) -> Iterator[tuple[str, nn.Module]]:
    """Yield the linear layers within a block that are candidates for tuning.

    Every ``nn.Linear`` under the block, by type; normalization layers and
    anything else are excluded by the same type test. There is no sensitivity
    rule here: a layer the scheme cannot group is reported by the runner when
    it skips it, not by this walk.

    This is the block loop's view, and it is deliberately narrower than
    :func:`iter_quantizable_modules`. Learned rounding needs activations flowing
    through a matrix product to have anything to optimize against; an embedding
    is a lookup and has none, so it is quantized by round-to-nearest at the
    model level and never appears here.

    Args:
        block: A transformer block.

    Yields:
        ``(name, module)`` pairs, with names relative to the block.
    """
    import mlx.nn as nn_  # noqa: PLC0415

    for name, module in sorted(block.named_modules()):
        if name and isinstance(module, nn_.Linear):
            yield name, module


def iter_quantizable_modules(
    model: nn.Module, *, group_size: int
) -> Iterator[tuple[str, nn.Module, bool]]:
    """Yield every module in ``model`` that can be quantized, and say which cannot.

    Covers linear layers and embeddings alike, which is what ``mlx-lm`` does and
    therefore what makes a size comparison against it meaningful. Tied word
    embeddings make that more than a convention: where they are tied there is no
    separate output head at all, so declining to quantize the embedding would
    leave the largest tensor in the model at full precision.

    The third element of each tuple is whether the module is *eligible*. A
    module whose last dimension is not a multiple of ``group_size`` cannot be
    grouped and stays dense. ``mlx-lm`` applies the same rule and applies it
    silently; here it is yielded so the caller can report it, because "this
    model is 4-bit" is false in a way nobody can see if a large layer quietly
    opted out.

    Args:
        model: A loaded MLX model.
        group_size: Weights sharing one scale, along the last dimension.

    Yields:
        ``(dotted_path, module, eligible)``, sorted by path.

    Raises:
        ArchitectureError: If the model exposes its own quantization predicate,
            which several architectures do to protect layers that must not be
            quantized. Honoring it needs a deliberate decision per architecture
            rather than a silent override.
    """
    import mlx.nn as nn_  # noqa: PLC0415

    if getattr(model, "quant_predicate", None) is not None:
        msg = (
            "this architecture ships its own quantization predicate, which "
            "exists to protect layers that must not be quantized or must be "
            "quantized differently. MRound does not yet honor it, and ignoring "
            "it would produce a checkpoint that loads and generates nonsense. "
            "See MEMORY.md Q-010."
        )
        raise ArchitectureError(msg)

    for path, module in sorted(model.named_modules()):
        if not path or not isinstance(module, nn_.Linear | nn_.Embedding):
            continue
        eligible = module.weight.shape[-1] % group_size == 0
        yield path, module, eligible
