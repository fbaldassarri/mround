# Copyright 2026 The MRound Authors
# SPDX-License-Identifier: Apache-2.0
"""Zero-shot task evaluation.

Perplexity is sensitive and cheap but it is not what anyone uses a model for.
Task accuracy is the check that a low-bit model is still useful rather than
merely still fluent, and the two can disagree: a quantization that barely moves
perplexity can still lose several points of task accuracy.

Three things about this evaluator are decided here rather than inherited from
whichever library happens to be called (MEMORY.md D-049).

**The suite is the opponent's.** :data:`DEFAULT_TASKS` is the default task list
of the reference implementation's own evaluation command (auto-round 0.15.0,
``auto_round/eval/eval_cli.py``) less ``wikitext``, which the perplexity
evaluator covers. A suite of our own choosing would be one more thing a reader
has to trust; the opponent's removes the choice from the result. Every request
those twelve tasks make is a log likelihood, zero shot, over the full sets.

**Only the log likelihood is ours.** ``lm_eval`` supplies the tasks, the
prompts, the metrics and, through its ``TemplateLM`` base class, the pairing of
context and continuation its own Hugging Face adapter uses: a context's
trailing whitespace moves into the continuation, the continuation's tokens are
the whole string's tokens past the context's, and an empty context is
conditioned on the prefix token. Tokenization follows the reference's
evaluation command, which passes ``add_bos_token=False``. MRound supplies the
scoring of token sequences over an MLX model, one code path for the original
and for both quantized arms, with the log softmax in float32 as
``perplexity.py`` decides and its normalizer reduced in blocks, which
:func:`mlx_forward` explains. Not ``mlx_lm``'s own ``mlxlm`` adapter: in 0.31.3 it
slices the whole string at the context's token count without moving the
whitespace, which scores a different continuation from the one the literature
scores.

**Each distinct model input runs once.** A request's input is its context and
continuation less the last token, so the four answers of a question whose
continuations are one token long share one input, and so do requests whose
split points differ over the same tokens. Grouping by input rather than by
context is exact, because a causal model's prediction at a position depends
only on the tokens before it, and it is why the default suite runs 88,988
inputs and 7.4 million tokens per checkpoint on Qwen2.5 rather than 140,977
requests and 13.7 million (MMLU alone goes from 6.8 million to 1.7). Rows are
right padded, which a causal model cannot see, and batched by a token budget,
longest first.

The planning and the assembly of results are plain Python, testable with no
model and no MLX; MLX and ``lm_eval`` are imported inside the functions that
need them, so this module imports anywhere.

Status: rolling log likelihood and generation are not implemented, and a task
that asks for either is refused with that message; log likelihood requests,
which are every request the default suite makes, are.
"""

from __future__ import annotations

import dataclasses
import math
import numbers
from typing import TYPE_CHECKING, Any, TypeAlias

from mround.exceptions import MRoundError

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from mlx import nn

__all__ = [
    "DEFAULT_TASKS",
    "DEFAULT_TOKEN_BUDGET",
    "Forward",
    "InputScores",
    "RequestPlan",
    "TaskResult",
    "ZeroShotError",
    "ZeroShotResult",
    "assemble",
    "batches",
    "context_window",
    "evaluate_zeroshot",
    "mlx_forward",
    "plan_requests",
    "score_pairs",
    "score_plan",
    "summarize",
    "template_lm",
]

# The reference's own default suite, in its own order, less wikitext
# (auto_round/eval/eval_cli.py in 0.15.0). See the module docstring.
DEFAULT_TASKS: tuple[str, ...] = (
    "lambada_openai",
    "hellaswag",
    "winogrande",
    "piqa",
    "mmlu",
    "truthfulqa_mc1",
    "truthfulqa_mc2",
    "openbookqa",
    "boolq",
    "rte",
    "arc_easy",
    "arc_challenge",
)

# Tokens per forward pass, padding included. The logits of one pass are
# budget times vocabulary floats: about 2.5 GB at float32 for Qwen2.5's 151,936
# entries, which a unified memory Mac holds beside the model with room to
# spare, and enough rows to keep the GPU busy on the short requests.
DEFAULT_TOKEN_BUDGET = 4096

# Vocabulary entries per first level block of the log softmax normalizer; see
# mlx_forward for why the reduction is taken in two levels.
NORMALIZER_BLOCK = 512

# The Hugging Face adapter's last fallback when a configuration names no
# window. Before it the adapter tries the tokenizer's model_max_length, which
# a configuration mapping does not carry, so the two can differ there.
_DEFAULT_WINDOW = 2048
_WINDOW_KEYS = ("n_positions", "max_position_embeddings", "n_ctx")


class ZeroShotError(MRoundError):
    """A zero-shot evaluation could not be run as asked."""


@dataclasses.dataclass(frozen=True, slots=True)
class TaskResult:
    """One task's outcome, as ``lm_eval`` aggregated it.

    Attributes:
        task: Task or group identifier, as requested.
        accuracy: The task's ``acc``. For ``truthfulqa_mc2`` that is the
            probability mass on the true answers, which is what the task
            defines under that name.
        stderr: Standard error of ``accuracy``, or ``None`` where ``lm_eval``
            reported none. Reported because quantization differences are often
            within it, and comparing two numbers whose error bars overlap is
            not evidence of anything.
        n_examples: Documents scored, summed over a group's members.
        accuracy_norm: ``acc_norm`` where the task defines it, else ``None``.
            Kept beside ``acc`` rather than instead of it, so the headline
            metric means the same thing on every task.
        stderr_norm: Standard error of ``accuracy_norm``.
        metrics: Every metric ``lm_eval`` reported for the task, by its own
            key (``acc,none``, ``perplexity,none`` and so on), so nothing it
            computed is lost to the fields above.
    """

    task: str
    accuracy: float
    stderr: float | None
    n_examples: int
    accuracy_norm: float | None = None
    stderr_norm: float | None = None
    metrics: dict[str, float] = dataclasses.field(default_factory=dict)

    def describe(self) -> str:
        """One line per task, with its error bar and its size."""
        line = f"{self.task:<16} acc {self.accuracy:.4f}"
        if self.stderr is not None:
            line += f" ± {self.stderr:.4f}"
        if self.accuracy_norm is not None:
            line += f"   acc_norm {self.accuracy_norm:.4f}"
            if self.stderr_norm is not None:
                line += f" ± {self.stderr_norm:.4f}"
        return f"{line}   ({self.n_examples} examples)"


@dataclasses.dataclass(frozen=True, slots=True)
class ZeroShotResult:
    """Results across the suite, with what it took to produce them.

    Attributes:
        tasks: Per task outcomes, in the order requested.
        mean_accuracy: Unweighted mean of ``accuracy`` across tasks. A summary
            for tracking trends, not a substitute for the per task numbers.
        n_requests: Log likelihood requests scored.
        n_scored_tokens: Tokens run through the model after inputs shared by
            several requests were merged, padding excluded.
        limit: Documents per task, or ``None`` for the full sets. A number
            computed on a limit is a check, never a result.
        lm_eval_version: The task library's version, because a task's prompt
            and its documents are that library's and can change with it.
    """

    tasks: list[TaskResult]
    mean_accuracy: float
    n_requests: int
    n_scored_tokens: int
    limit: int | None
    lm_eval_version: str | None

    def describe(self) -> str:
        """The per task lines and the mean, with the settings that bound them."""
        scope = "full sets" if self.limit is None else f"first {self.limit} documents per task"
        lines = [task.describe() for task in self.tasks]
        lines.append(
            f"{'mean':<16} acc {self.mean_accuracy:.4f}   over {len(self.tasks)} tasks, "
            f"{scope}, {self.n_requests} requests, {self.n_scored_tokens} tokens scored"
        )
        return "\n".join(lines)


@dataclasses.dataclass(frozen=True, slots=True)
class RequestPlan:
    """Requests mapped onto the distinct model inputs that answer them.

    Attributes:
        inputs: Each distinct model input, longest first, which is the order
            the batches run in.
        needed: Per input, how many trailing positions any request reads.
        finals: Per input, the distinct targets requested at its last
            position, which is the only position where requests sharing an
            input can want different tokens.
        requests: Per request, in the caller's order: the index of its input,
            its continuation length and its final continuation token.
    """

    inputs: list[tuple[int, ...]]
    needed: list[int]
    finals: list[tuple[int, ...]]
    requests: list[tuple[int, int, int]]

    @property
    def n_scored_tokens(self) -> int:
        """Tokens the plan runs through the model, padding excluded."""
        return sum(len(tokens) for tokens in self.inputs)


# Per input: log probabilities of the input's own next tokens over the needed
# window less its last position, the model's argmax at every needed position,
# and the log probability of each requested final target, in ``finals`` order.
InputScores: TypeAlias = tuple[list[float], list[int], list[float]]

# Scores one batch: its inputs, the trailing positions each needs, and the
# final targets each has, to one InputScores per input.
Forward: TypeAlias = (
    "Callable[[Sequence[tuple[int, ...]], Sequence[int], Sequence[tuple[int, ...]]], "
    "list[InputScores]]"
)


def context_window(config: Mapping[str, Any]) -> int:
    """The context window a configuration declares, as the Hugging Face adapter reads it.

    Args:
        config: A model configuration mapping.

    Returns:
        The first of ``n_positions``, ``max_position_embeddings`` and
        ``n_ctx`` present, else 2048, the adapter's last fallback; the
        adapter first tries the tokenizer's ``model_max_length``, which a
        configuration does not carry.
    """
    for key in _WINDOW_KEYS:
        value = config.get(key)
        if isinstance(value, int) and value > 0:
            return value
    return _DEFAULT_WINDOW


def plan_requests(
    pairs: Sequence[tuple[Sequence[int], Sequence[int]]], *, max_length: int
) -> RequestPlan:
    """Map (context, continuation) token pairs onto distinct model inputs.

    Truncation is the Hugging Face adapter's: the last ``max_length + 1``
    tokens of context and continuation together, less the final one, so a
    request too long for the window loses context from the left and never
    continuation.

    Args:
        pairs: Context and continuation token ids per request. Both must be
            non empty, which ``TemplateLM`` guarantees by conditioning an empty
            context on the prefix token.
        max_length: The model's context window.

    Returns:
        The plan, with requests in the caller's order.

    Raises:
        ZeroShotError: If a request has an empty side, or a continuation
            longer than the window, which no truncation can score.
    """
    index: dict[tuple[int, ...], int] = {}
    unique: list[tuple[int, ...]] = []
    needed: list[int] = []
    finals: list[dict[int, None]] = []
    raw: list[tuple[int, int, int]] = []

    for number, (context, continuation) in enumerate(pairs):
        if not context or not continuation:
            msg = f"request {number} has an empty context or continuation, which scores nothing"
            raise ZeroShotError(msg)
        if len(continuation) > max_length:
            msg = (
                f"request {number} has a continuation of {len(continuation)} tokens, "
                f"longer than the model's window of {max_length}"
            )
            raise ZeroShotError(msg)
        whole = [*context, *continuation][-(max_length + 1) :]
        tokens = tuple(whole[:-1])
        slot = index.get(tokens)
        if slot is None:
            slot = len(unique)
            index[tokens] = slot
            unique.append(tokens)
            needed.append(0)
            finals.append({})
        needed[slot] = max(needed[slot], len(continuation))
        finals[slot][continuation[-1]] = None
        raw.append((slot, len(continuation), continuation[-1]))

    # Longest first, as the Hugging Face adapter orders its batches: the first
    # batch is the largest, so a budget that is too big fails at once rather
    # than an hour in.
    order = sorted(range(len(unique)), key=lambda slot: (-len(unique[slot]), unique[slot]))
    position = {slot: rank for rank, slot in enumerate(order)}
    return RequestPlan(
        inputs=[unique[slot] for slot in order],
        needed=[needed[slot] for slot in order],
        finals=[tuple(finals[slot]) for slot in order],
        requests=[(position[slot], length, final) for slot, length, final in raw],
    )


def batches(lengths: Sequence[int], token_budget: int) -> list[range]:
    """Consecutive runs of inputs, longest first, within a token budget.

    Args:
        lengths: Input lengths, in non increasing order.
        token_budget: Rows times padded width allowed per forward pass. A
            single input longer than the budget still runs, alone.

    Returns:
        Index ranges into ``lengths``.
    """
    runs: list[range] = []
    start = 0
    while start < len(lengths):
        width = lengths[start]
        rows = max(1, token_budget // max(1, width))
        stop = min(len(lengths), start + rows)
        runs.append(range(start, stop))
        start = stop
    return runs


def assemble(plan: RequestPlan, scores: Sequence[InputScores]) -> list[tuple[float, bool]]:
    """Each request's log likelihood and whether greedy decoding would produce it.

    Args:
        plan: The plan the scores answer.
        scores: Per input, in the plan's order: see :data:`InputScores`.

    Returns:
        ``(log likelihood, is greedy)`` per request, in the caller's order,
        which is what ``lm_eval`` expects back.
    """
    results: list[tuple[float, bool]] = []
    for slot, length, final in plan.requests:
        inner, greedy, final_scores = scores[slot]
        tokens = plan.inputs[slot]
        window = plan.needed[slot]
        # The continuation's tokens before its last are the input's own last
        # length - 1 tokens; its last is the request's own final target.
        loglikelihood = math.fsum(inner[len(inner) - (length - 1) :]) if length > 1 else 0.0
        loglikelihood += final_scores[plan.finals[slot].index(final)]
        targets = [*tokens[len(tokens) - (length - 1) :], final] if length > 1 else [final]
        is_greedy = greedy[window - length :] == targets
        results.append((loglikelihood, is_greedy))
    return results


def score_plan(
    plan: RequestPlan,
    forward: Forward,
    *,
    token_budget: int = DEFAULT_TOKEN_BUDGET,
    progress: Callable[[int], None] | None = None,
) -> list[tuple[float, bool]]:
    """Run a plan's inputs through ``forward`` in batches and assemble the results.

    Args:
        plan: From :func:`plan_requests`.
        forward: Scores one batch: given its inputs, the positions each needs
            and the final targets each has, returns :data:`InputScores` per
            input. :func:`mlx_forward` is the one this module uses; tests pass
            their own.
        token_budget: See :func:`batches`.
        progress: Called with the number of inputs finished after each batch.

    Returns:
        Per request results in the caller's order.
    """
    scores: list[InputScores] = []
    for run in batches([len(tokens) for tokens in plan.inputs], token_budget):
        rows = [plan.inputs[i] for i in run]
        scores.extend(forward(rows, [plan.needed[i] for i in run], [plan.finals[i] for i in run]))
        if progress is not None:
            progress(len(run))
    return assemble(plan, scores)


def score_pairs(
    pairs: Sequence[tuple[Sequence[int], Sequence[int]]],
    forward: Forward,
    *,
    max_length: int,
    token_budget: int = DEFAULT_TOKEN_BUDGET,
) -> list[tuple[float, bool]]:
    """Plan, batch, score and assemble, for callers that hold token pairs.

    Args:
        pairs: Context and continuation token ids per request.
        forward: See :func:`score_plan`.
        max_length: The model's context window.
        token_budget: See :func:`batches`.

    Returns:
        Per request results in the caller's order.
    """
    plan = plan_requests(pairs, max_length=max_length)
    return score_plan(plan, forward, token_budget=token_budget)


def mlx_forward(model: nn.Module, *, pad_id: int, block: int = NORMALIZER_BLOCK) -> Forward:
    """A batch scorer over an MLX causal language model.

    The logits are cast to float32 before the log softmax, as the perplexity
    evaluator casts them, and the softmax is taken only at the positions some
    request reads, so the float32 copy is a few rows rather than the batch.

    The normalizer is a log sum of exponentials over the vocabulary taken in
    two levels, within blocks and then across them, rather than by one call
    over the whole row. Measured 2026-09-28 on MLX 0.32.0's CPU backend, one
    call over SmolLM2's 49,152 entries comes out low by 1.3e-4 on average
    (float64 as the truth; torch's float32 ``logsumexp`` is within 1e-7, its
    float32 ``log_softmax`` is not, MEMORY.md 2026-09-29), which lifts every
    token's log probability by that much and a twelve token answer by 1.6e-3;
    in blocks of 512 the error is 6e-7 on average, at most 2.5e-6. On Metal
    one call is 2.2e-7 to 2.4e-7 low on average, at most 1.7e-6 (Q-014), so
    the blocks matter on the CPU stream, where the gate runs; they are taken
    on both, since the order of a reduction is the backend's choice and this
    one is written down.

    Args:
        model: A loaded MLX model, quantized or not, called as ``model(ids)``
            for logits.
        pad_id: Any valid token id; right padding is never read.
        block: Vocabulary entries per first level block.

    Returns:
        A ``forward`` for :func:`score_plan`.
    """
    import mlx.core as mx  # noqa: PLC0415

    def normalizer(scores: mx.array) -> mx.array:
        rows, vocabulary = scores.shape
        pad = (-vocabulary) % block
        if pad:
            filler = mx.full((rows, pad), -math.inf, dtype=scores.dtype)
            scores = mx.concatenate([scores, filler], axis=-1)
        partial = mx.logsumexp(scores.reshape(rows, -1, block), axis=-1)
        return mx.logsumexp(partial, axis=-1)

    def forward(
        rows: Sequence[tuple[int, ...]],
        needed: Sequence[int],
        finals: Sequence[tuple[int, ...]],
    ) -> list[InputScores]:
        width = max(len(row) for row in rows)
        ids = mx.array([[*row, *([pad_id] * (width - len(row)))] for row in rows], dtype=mx.int32)
        logits = model(ids)
        pending = []
        for i, row in enumerate(rows):
            length, window = len(row), needed[i]
            scores = logits[i, length - window : length, :].astype(mx.float32)
            total = normalizer(scores)
            greedy = mx.argmax(scores, axis=-1)
            if window > 1:
                targets = mx.array(row[length - window + 1 :], dtype=mx.int32)
                picked = mx.take_along_axis(scores[:-1], targets[:, None], axis=-1)[:, 0]
                inner = picked - total[:-1]
            else:
                inner = mx.zeros((0,), dtype=mx.float32)
            final = scores[-1][mx.array(finals[i], dtype=mx.int32)] - total[-1]
            pending.append((inner, greedy, final))
        # One evaluation per batch: MLX is lazy, and without it the graph of
        # every batch would accumulate until memory ran out.
        mx.eval([array for triple in pending for array in triple])
        return [
            (inner.tolist(), greedy.tolist(), final.tolist()) for inner, greedy, final in pending
        ]

    return forward


def _unwrap(tokenizer: object) -> Any:
    """The Hugging Face tokenizer inside ``mlx_lm``'s wrapper, or the object itself."""
    return getattr(tokenizer, "_tokenizer", tokenizer)


class _Progress:
    """A one line progress report on standard error, at most once a second.

    Written here rather than taken from ``tqdm``, which ships no type
    information, so that the strict type check does not depend on which stub
    packages a machine happens to have.
    """

    def __init__(self, total: int, tokens: int) -> None:
        import time  # noqa: PLC0415

        self._clock = time.monotonic
        self._total = total
        self._tokens = tokens
        self._done = 0
        self._drawn = -1
        self._started = self._clock()
        self._shown = 0.0

    def advance(self, finished: int) -> None:
        """Count ``finished`` more inputs and redraw if a second has passed."""
        self._done += finished
        now = self._clock()
        if now - self._shown >= 1.0:
            self._shown = now
            self._draw(now)

    def finish(self) -> None:
        """Draw the final state, unless it is already on screen, and end the line."""
        import sys  # noqa: PLC0415

        if self._drawn != self._done:
            self._draw(self._clock())
        print(file=sys.stderr, flush=True)

    def _draw(self, now: float) -> None:
        import sys  # noqa: PLC0415

        elapsed = now - self._started
        self._drawn = self._done
        print(
            f"\rscoring {self._done}/{self._total} inputs "
            f"({self._tokens} tokens in all), {elapsed:.0f} s",
            end="",
            file=sys.stderr,
            flush=True,
        )


def template_lm(
    forward: Forward,
    tokenizer: object,
    *,
    max_length: int,
    token_budget: int = DEFAULT_TOKEN_BUDGET,
    add_bos_token: bool | None = False,
    counters: dict[str, int] | None = None,
) -> Any:
    """An ``lm_eval`` language model that scores through :func:`score_plan`.

    Built inside a function because its base class is ``lm_eval``'s, and this
    module has to import where ``lm_eval`` is not installed. Takes a
    ``forward`` rather than a model, so the tokenization and the request
    handling can be tested without one.

    Args:
        forward: Scores one batch; :func:`mlx_forward` in real use.
        tokenizer: ``mlx_lm``'s wrapper or a Hugging Face tokenizer.
        max_length: The model's context window.
        token_budget: See :func:`batches`.
        add_bos_token: The Hugging Face adapter's flag of the same name.
        counters: If given, ``requests`` and ``tokens`` are added to it.

    Returns:
        An instance of a ``TemplateLM`` subclass.

    Raises:
        ZeroShotError: If ``lm_eval`` is not installed.
    """
    try:
        from lm_eval.api.model import TemplateLM  # noqa: PLC0415
    except ImportError as exc:
        msg = (
            "the zero-shot suite needs the evaluation stack. Install it with:\n"
            "    pip install -e '.[eval]'"
        )
        raise ZeroShotError(msg) from exc

    hf = _unwrap(tokenizer)
    tally = counters if counters is not None else {"requests": 0, "tokens": 0}

    class MLXTemplateLM(TemplateLM):  # type: ignore[misc]
        """``TemplateLM`` over an MLX model; see the module docstring."""

        def __init__(self) -> None:
            super().__init__()
            self.tokenizer = hf

        @property
        def eot_token_id(self) -> int:
            return int(hf.eos_token_id)

        @property
        def prefix_token_id(self) -> int:
            # The Hugging Face adapter's choice: the BOS token where the
            # tokenizer has one, else the end of text token.
            bos = hf.bos_token_id
            return int(bos) if bos is not None else self.eot_token_id

        @property
        def max_length(self) -> int:
            return max_length

        def tok_encode(
            self,
            string: str,
            add_special_tokens: bool | None = None,
            left_truncate_len: int | None = None,
            **kwargs: Any,  # noqa: ARG002
        ) -> list[int]:
            # The Hugging Face adapter's rule, with the reference's setting:
            # an explicit request wins, then add_bos_token, then the
            # tokenizer's own default, and a string that already starts with
            # the prefix token never gets a second one.
            options: dict[str, bool] = {}
            if add_special_tokens is not None:
                options["add_special_tokens"] = add_special_tokens
            elif add_bos_token is not None:
                options["add_special_tokens"] = add_bos_token
            if add_special_tokens is None and string.startswith(hf.decode(self.prefix_token_id)):
                options["add_special_tokens"] = False
            encoding = list(hf.encode(string, **options))
            if left_truncate_len:
                encoding = encoding[-left_truncate_len:]
            return encoding

        def tok_decode(self, tokens: Sequence[int], skip_special_tokens: bool = True) -> str:
            return str(hf.decode(tokens, skip_special_tokens=skip_special_tokens))

        def _loglikelihood_tokens(
            self,
            requests: list[tuple[tuple[str, str], list[int], list[int]]],
            disable_tqdm: bool = False,
            **kwargs: Any,  # noqa: ARG002
        ) -> list[tuple[float, bool]]:
            pairs = [(context, continuation) for _, context, continuation in requests]
            plan = plan_requests(pairs, max_length=max_length)
            report = None if disable_tqdm else _Progress(len(plan.inputs), plan.n_scored_tokens)
            results = score_plan(
                plan,
                forward,
                token_budget=token_budget,
                progress=None if report is None else report.advance,
            )
            if report is not None:
                report.finish()
            tally["requests"] += len(requests)
            tally["tokens"] += plan.n_scored_tokens
            for (request, _, _), answer in zip(requests, results, strict=True):
                self.cache_hook.add_partial("loglikelihood", request, answer)
            return results

        def loglikelihood_rolling(self, requests: Any, disable_tqdm: bool = False) -> list[float]:
            msg = (
                "rolling log likelihood is not implemented: wikitext perplexity is "
                "mround.eval.perplexity's, and no task in the default suite asks for it"
            )
            raise NotImplementedError(msg)

        def generate_until(self, requests: Any, disable_tqdm: bool = False) -> list[str]:
            msg = "generation is not implemented, and no task in the default suite asks for it"
            raise NotImplementedError(msg)

    return MLXTemplateLM()


def _pad_id(hf: Any) -> int:
    """A token id to pad with. Never read, since padding is on the right."""
    for candidate in (hf.pad_token_id, hf.eos_token_id):
        if candidate is not None:
            return int(candidate)
    return 0


def _as_float(value: object) -> float | None:
    """A metric value, or ``None`` for the strings ``lm_eval`` writes when it has none.

    ``numbers.Real`` rather than ``float``, because numpy's scalars other than
    float64 are not float subclasses and an aggregation can return one.
    """
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return None
    return float(value)


def _n_examples(name: str, results: Mapping[str, Any]) -> int:
    """Documents scored for a task, summing a group's members."""
    samples = results.get("n-samples", {})
    if name in samples:
        return int(samples[name].get("effective", samples[name].get("original", 0)))
    members = results.get("group_subtasks", {}).get(name, [])
    return sum(_n_examples(member, results) for member in members)


def summarize(results: Mapping[str, Any], tasks: Sequence[str]) -> list[TaskResult]:
    """Pull each requested task's metrics out of ``lm_eval``'s results.

    Args:
        results: What ``lm_eval.simple_evaluate`` returned.
        tasks: The task or group names that were requested.

    Returns:
        One result per requested name, in the order requested.

    Raises:
        ZeroShotError: If a requested task has no ``acc`` in the results.
    """
    by_task = results.get("results", {})
    summary: list[TaskResult] = []
    for name in tasks:
        reported = by_task.get(name)
        if reported is None:
            msg = f"lm_eval returned no results for {name!r}"
            raise ZeroShotError(msg)
        # Every metric lm_eval reports is keyed "name,filter"; what else sits
        # beside them (the alias, and a sample_len since 0.4.12) is not one.
        metrics = {
            key: number
            for key, value in reported.items()
            if "," in key and (number := _as_float(value)) is not None
        }
        accuracy = metrics.get("acc,none")
        if accuracy is None:
            msg = f"{name!r} reports no acc; it reported {sorted(metrics)}"
            raise ZeroShotError(msg)
        summary.append(
            TaskResult(
                task=name,
                accuracy=accuracy,
                stderr=metrics.get("acc_stderr,none"),
                n_examples=_n_examples(name, results),
                accuracy_norm=metrics.get("acc_norm,none"),
                stderr_norm=metrics.get("acc_norm_stderr,none"),
                metrics=metrics,
            )
        )
    return summary


def evaluate_zeroshot(
    model: nn.Module,
    tokenizer: object,
    *,
    tasks: Sequence[str] = DEFAULT_TASKS,
    limit: int | None = None,
    max_length: int = _DEFAULT_WINDOW,
    token_budget: int = DEFAULT_TOKEN_BUDGET,
    add_bos_token: bool | None = False,
    bootstrap_iters: int = 100_000,
    log_samples: bool = False,
    verbose: bool = False,
) -> tuple[ZeroShotResult, dict[str, Any]]:
    """Evaluate a model on zero-shot tasks.

    Args:
        model: A loaded MLX model, quantized or not.
        tokenizer: Its tokenizer, ``mlx_lm``'s wrapper or a Hugging Face one.
        tasks: Task or group identifiers, in ``lm_eval``'s names.
        limit: Documents per task, for quick checks. ``None`` scores the full
            sets, which is what a published number requires.
        max_length: The model's context window; :func:`context_window` reads
            it from a configuration. Only a request longer than the window
            depends on it, and none in the default suite is.
        token_budget: Tokens per forward pass; see :func:`batches`.
        add_bos_token: Passed to tokenization as the Hugging Face adapter's
            flag of the same name. ``False`` is what the reference's evaluation
            command passes.
        bootstrap_iters: Resamples for the metrics whose standard error
            ``lm_eval`` bootstraps (LAMBADA's perplexity); ``acc`` has a
            closed form. ``lm_eval``'s default.
        log_samples: Keep per document responses in the returned results, for
            comparing two evaluators request by request.
        verbose: Set ``lm_eval``'s log level to INFO rather than WARNING,
            through its deprecated ``verbosity`` argument, which 0.4.12
            applies only once its evaluation has run. Progress bars are drawn
            either way: ``lm_eval``'s own while it builds and tokenizes the
            requests, and this module's line while it scores.

    Returns:
        The summary, and ``lm_eval``'s full results for anything the summary
        does not carry.

    Raises:
        ZeroShotError: If the evaluation stack is missing, ``lm_eval`` does
            not know a task, a task asks for generation or rolling log
            likelihood, a request cannot be scored, or a task reports no
            accuracy.
    """
    from importlib import metadata  # noqa: PLC0415

    counters = {"requests": 0, "tokens": 0}
    language_model = template_lm(
        mlx_forward(model, pad_id=_pad_id(_unwrap(tokenizer))),
        tokenizer,
        max_length=max_length,
        token_budget=token_budget,
        add_bos_token=add_bos_token,
        counters=counters,
    )
    from lm_eval.evaluator import simple_evaluate  # noqa: PLC0415

    # num_fewshot None leaves every task at its own default, which for these
    # twelve is zero, and is what the reference's evaluation command passes.
    try:
        results = simple_evaluate(
            model=language_model,
            tasks=list(tasks),
            num_fewshot=None,
            limit=limit,
            bootstrap_iters=bootstrap_iters,
            log_samples=log_samples,
            verbosity="INFO" if verbose else "WARNING",
        )
    except NotImplementedError as exc:
        # A task that asks for generation or rolling log likelihood, which
        # this evaluator declares unfinished (see the Status line).
        raise ZeroShotError(str(exc)) from exc
    except KeyError as exc:
        # lm_eval 0.4.12 reports a task name it does not know as a KeyError.
        msg = f"lm_eval could not load the tasks asked for: {exc.args[0] if exc.args else exc}"
        raise ZeroShotError(msg) from exc
    if results is None:
        msg = "lm_eval returned no results"
        raise ZeroShotError(msg)
    summary = summarize(results, tasks)
    try:
        version: str | None = metadata.version("lm_eval")
    except metadata.PackageNotFoundError:
        version = None
    return (
        ZeroShotResult(
            tasks=summary,
            mean_accuracy=sum(task.accuracy for task in summary) / len(summary),
            n_requests=counters["requests"],
            n_scored_tokens=counters["tokens"],
            limit=limit,
            lm_eval_version=version,
        ),
        dict(results),
    )
