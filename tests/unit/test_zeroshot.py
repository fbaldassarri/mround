# Copyright 2026 The MRound Authors
# SPDX-License-Identifier: Apache-2.0
"""The zero-shot evaluator's request handling, against an oracle that scores one request at a time.

The part of a task evaluator that is quietly wrong is the bookkeeping, not the
forward pass: which positions a continuation is read from, which token each
position is scored against, what a shared input may and may not be shared for,
and what truncation keeps. Every one of those errors produces plausible
numbers. So the planner and the assembly are checked against the most literal
implementation possible, the Hugging Face adapter's arithmetic done request by
request over a toy causal model, in float64, where any disagreement is a bug
and not rounding.

Three layers, by what they need: the plan and the assembly need nothing; the
``lm_eval`` adapter needs ``lm_eval`` and is skipped without it; the MLX batch
scorer needs MLX and carries the ``needs_mlx`` marker.
"""

from __future__ import annotations

import math
import random
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from mround.eval.zeroshot import (
    DEFAULT_TASKS,
    RequestPlan,
    TaskResult,
    ZeroShotError,
    ZeroShotResult,
    batches,
    context_window,
    plan_requests,
    score_pairs,
    score_plan,
    summarize,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mround.eval.zeroshot import InputScores

VOCAB = 13
REPO_ROOT = Path(__file__).resolve().parents[2]


def toy_logits(prefix: Sequence[int]) -> np.ndarray:
    """Logits after ``prefix``: a function of the prefix alone, so causal by construction."""
    seed = hash(tuple(prefix)) & 0xFFFFFFFF
    return np.random.default_rng(seed).normal(scale=3.0, size=VOCAB)


def log_softmax(logits: np.ndarray) -> np.ndarray:
    top = float(logits.max())
    return np.asarray(logits - top - math.log(float(np.exp(logits - top).sum())))


def oracle(
    context: Sequence[int], continuation: Sequence[int], max_length: int
) -> tuple[float, bool]:
    """One request, the Hugging Face adapter's way, with nothing shared."""
    whole = [*context, *continuation][-(max_length + 1) :]
    inputs = whole[:-1]
    total, greedy = 0.0, True
    for offset in range(len(continuation)):
        position = len(inputs) - len(continuation) + offset
        scores = log_softmax(toy_logits(inputs[: position + 1]))
        target = whole[position + 1]
        total += float(scores[target])
        greedy &= int(np.argmax(scores)) == target
    return total, greedy


def toy_forward(
    rows: Sequence[tuple[int, ...]],
    needed: Sequence[int],
    finals: Sequence[tuple[int, ...]],
) -> list[InputScores]:
    """What ``mlx_forward`` returns, computed from the toy model in float64."""
    out: list[InputScores] = []
    for row, window, final_targets in zip(rows, needed, finals, strict=True):
        length = len(row)
        inner, greedy = [], []
        for position in range(length - window, length):
            scores = log_softmax(toy_logits(row[: position + 1]))
            greedy.append(int(np.argmax(scores)))
            if position < length - 1:
                inner.append(float(scores[row[position + 1]]))
        last = log_softmax(toy_logits(row))
        out.append((inner, greedy, [float(last[token]) for token in final_targets]))
    return out


def random_requests(rng: random.Random, count: int) -> list[tuple[list[int], list[int]]]:
    """Requests with the sharing the real suite has: common contexts, one token answers."""
    contexts = [[rng.randrange(VOCAB) for _ in range(rng.randint(1, 12))] for _ in range(6)]
    requests = []
    for _ in range(count):
        context = rng.choice(contexts)
        length = rng.choice([1, 1, 1, 2, 3, 5])
        requests.append((context, [rng.randrange(VOCAB) for _ in range(length)]))
    # A pair whose split points differ over the same tokens: one input, two reads.
    requests.append(([1, 2, 3], [4, 5]))
    requests.append(([1, 2, 3, 4], [5]))
    return requests


class TestPlan:
    def test_answers_sharing_a_context_share_one_input(self) -> None:
        # The MMLU case: four one token answers after one question.
        plan = plan_requests([([7, 8, 9], [answer]) for answer in (1, 2, 3, 4)], max_length=64)
        assert plan.inputs == [(7, 8, 9)]
        assert plan.needed == [1]
        assert plan.finals == [(1, 2, 3, 4)]
        assert [slot for slot, _, _ in plan.requests] == [0, 0, 0, 0]

    def test_different_split_points_over_the_same_tokens_share_too(self) -> None:
        plan = plan_requests([([1, 2], [3, 4]), ([1, 2, 3], [4])], max_length=64)
        assert plan.inputs == [(1, 2, 3)]
        assert plan.needed == [2]

    def test_truncation_keeps_the_continuation_and_drops_context_from_the_left(self) -> None:
        # The Hugging Face adapter's slice: the last max_length + 1 tokens of
        # the whole request, less the final one.
        plan = plan_requests([([1, 2, 3, 4, 5, 6], [7, 8])], max_length=4)
        assert plan.inputs == [(4, 5, 6, 7)]

    def test_inputs_run_longest_first_and_requests_keep_their_order(self) -> None:
        pairs = [([1], [2]), ([1, 2, 3, 4], [5]), ([9, 9], [1])]
        plan = plan_requests(pairs, max_length=64)
        assert [len(tokens) for tokens in plan.inputs] == [4, 2, 1]
        assert [plan.inputs[slot] for slot, _, _ in plan.requests] == [(1,), (1, 2, 3, 4), (9, 9)]

    @pytest.mark.parametrize("pair", [([], [1]), ([1], [])])
    def test_an_empty_side_is_refused(self, pair: tuple[list[int], list[int]]) -> None:
        with pytest.raises(ZeroShotError, match="empty"):
            plan_requests([pair], max_length=8)

    def test_a_continuation_longer_than_the_window_is_refused(self) -> None:
        # Truncating it would score a different continuation; the Hugging Face
        # adapter asserts the same.
        with pytest.raises(ZeroShotError, match="longer than"):
            plan_requests([([1], [2, 3, 4])], max_length=2)

    def test_scored_tokens_count_each_shared_input_once(self) -> None:
        plan = plan_requests([([5, 6, 7], [answer]) for answer in range(4)], max_length=64)
        assert plan.n_scored_tokens == 3


class TestBatches:
    def test_every_input_runs_once_in_order_within_the_budget(self) -> None:
        lengths = [9, 9, 7, 5, 5, 5, 2, 1, 1]
        runs = batches(lengths, token_budget=20)
        assert [i for run in runs for i in run] == list(range(len(lengths)))
        for run in runs:
            assert len(run) * lengths[run.start] <= 20 or len(run) == 1

    def test_an_input_longer_than_the_budget_still_runs_alone(self) -> None:
        assert batches([50, 3], token_budget=10) == [range(0, 1), range(1, 2)]


class TestAgainstTheOracle:
    @pytest.mark.parametrize("seed", range(5))
    @pytest.mark.parametrize(("max_length", "budget"), [(64, 7), (64, 1000), (6, 12)])
    def test_every_request_matches_one_at_a_time_scoring(
        self, seed: int, max_length: int, budget: int
    ) -> None:
        # Sharing, batching and truncation must change nothing: each request's
        # log likelihood and greedy flag are what scoring it alone gives. Six
        # tokens of window forces truncation on most requests; a budget of
        # seven forces most batches down to one or two rows.
        rng = random.Random(seed)
        pairs = random_requests(rng, 40)
        got = score_pairs(pairs, toy_forward, max_length=max_length, token_budget=budget)
        for (context, continuation), (loglikelihood, greedy) in zip(pairs, got, strict=True):
            expected, expected_greedy = oracle(context, continuation, max_length)
            assert math.isclose(loglikelihood, expected, rel_tol=1e-12, abs_tol=1e-12)
            assert greedy == expected_greedy

    def test_a_greedy_continuation_is_recognised(self) -> None:
        # Build a continuation from the toy model's own argmax, so the flag
        # has to come out true somewhere.
        context = [3, 1, 4]
        continuation: list[int] = []
        for _ in range(3):
            continuation.append(int(np.argmax(toy_logits([*context, *continuation]))))
        ((_, greedy),) = score_pairs([(context, continuation)], toy_forward, max_length=64)
        assert greedy

    def test_progress_counts_every_input(self) -> None:
        plan = plan_requests(random_requests(random.Random(7), 30), max_length=64)
        seen: list[int] = []
        score_plan(plan, toy_forward, token_budget=9, progress=seen.append)
        assert sum(seen) == len(plan.inputs)


def lm_eval_results() -> dict[str, Any]:
    """The shape ``lm_eval.simple_evaluate`` returns, reduced to what summarize reads."""
    return {
        "results": {
            "arc_easy": {
                "alias": "arc_easy",
                "acc,none": 0.5,
                "acc_stderr,none": 0.1,
                "acc_norm,none": 0.25,
                "acc_norm_stderr,none": 0.05,
            },
            "lambada_openai": {
                "alias": "lambada_openai",
                "sample_len": 5153,
                "perplexity,none": 12.5,
                "perplexity_stderr,none": "N/A",
                "acc,none": 0.4,
                "acc_stderr,none": "N/A",
            },
            "mmlu": {"alias": "mmlu", "acc,none": 0.3, "acc_stderr,none": 0.01},
            "mmlu_stem": {"alias": " - stem", "acc,none": 0.2},
        },
        "group_subtasks": {"mmlu": ["mmlu_stem", "mmlu_other"], "mmlu_stem": ["mmlu_physics"]},
        "n-samples": {
            "arc_easy": {"original": 2376, "effective": 20},
            "lambada_openai": {"original": 5153, "effective": 5153},
            "mmlu_physics": {"original": 102, "effective": 102},
            "mmlu_other": {"original": 300, "effective": 300},
        },
    }


class TestSummary:
    def test_accuracy_its_error_and_the_normalized_one(self) -> None:
        (arc,) = summarize(lm_eval_results(), ["arc_easy"])
        assert (arc.accuracy, arc.stderr, arc.accuracy_norm, arc.stderr_norm) == (
            0.5,
            0.1,
            0.25,
            0.05,
        )
        assert arc.n_examples == 20  # effective, which a limit reduces

    def test_a_missing_error_bar_is_none_and_other_metrics_are_kept(self) -> None:
        (lambada,) = summarize(lm_eval_results(), ["lambada_openai"])
        assert lambada.stderr is None
        assert lambada.accuracy_norm is None
        assert lambada.metrics["perplexity,none"] == 12.5
        assert "perplexity_stderr,none" not in lambada.metrics
        assert "sample_len" not in lambada.metrics, "a count beside the metrics is not one"

    def test_numpy_scalars_are_numbers(self) -> None:
        results = lm_eval_results()
        results["results"]["arc_easy"]["acc,none"] = np.float32(0.5)
        (arc,) = summarize(results, ["arc_easy"])
        assert arc.accuracy == 0.5

    def test_a_group_counts_its_members_documents(self) -> None:
        (mmlu,) = summarize(lm_eval_results(), ["mmlu"])
        assert mmlu.n_examples == 402

    def test_a_task_without_accuracy_is_refused(self) -> None:
        results = lm_eval_results()
        results["results"]["arc_easy"].pop("acc,none")
        with pytest.raises(ZeroShotError, match="no acc"):
            summarize(results, ["arc_easy"])

    def test_a_task_absent_from_the_results_is_refused(self) -> None:
        with pytest.raises(ZeroShotError, match="no results"):
            summarize(lm_eval_results(), ["piqa"])

    def test_the_description_carries_scope_and_error_bars(self) -> None:
        tasks = summarize(lm_eval_results(), ["arc_easy", "lambada_openai"])
        result = ZeroShotResult(
            tasks=tasks,
            mean_accuracy=0.45,
            n_requests=100,
            n_scored_tokens=5000,
            limit=20,
            lm_eval_version="0.4.12",
        )
        text = result.describe()
        assert "± 0.1000" in text
        assert "first 20 documents per task" in text
        assert isinstance(tasks[0], TaskResult)


class TestSuite:
    def test_the_window_is_read_as_the_hugging_face_adapter_reads_it(self) -> None:
        assert context_window({"max_position_embeddings": 32768}) == 32768
        assert context_window({"n_positions": 1024, "max_position_embeddings": 4096}) == 1024
        assert context_window({}) == 2048

    def test_the_default_suite_is_the_references_own_less_wikitext(self) -> None:
        # Read from the vendored reference where this checkout has it; the
        # published tree does not carry the reference, so there it skips.
        source = REPO_ROOT / "auto-round-0.15.0" / "auto_round" / "eval" / "eval_cli.py"
        if not source.is_file():
            pytest.skip("the vendored reference is not in this tree")
        text = source.read_text()
        start = text.index('default="lambada_openai')
        # The default is two adjacent string literals, which Python joins.
        literals = re.findall(r'"([^"]*)"', text[start : text.index("help=", start)])
        reference = [task for task in "".join(literals).split(",") if task]
        assert "wikitext" in reference
        assert [task for task in reference if task != "wikitext"] == list(DEFAULT_TASKS)


class FakeTokenizer:
    """Characters as tokens, with the attributes the adapter reads."""

    bos_token_id = None
    eos_token_id = 0
    pad_token_id = None

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def encode(self, text: str, **options: Any) -> list[int]:
        self.calls.append(options)
        return [ord(character) % VOCAB for character in text]

    def decode(self, tokens: Any, **options: Any) -> str:  # noqa: ARG002
        return "<eos>"


class TestTemplateLM:
    def _instances(self, pairs: Sequence[tuple[str, str]]) -> list[Any]:
        instance = pytest.importorskip("lm_eval.api.instance")
        return [
            instance.Instance(request_type="loglikelihood", doc={}, arguments=pair, idx=i)
            for i, pair in enumerate(pairs)
        ]

    def test_requests_are_scored_as_the_oracle_scores_them(self) -> None:
        pytest.importorskip("lm_eval")
        from mround.eval.zeroshot import template_lm  # noqa: PLC0415

        tokenizer = FakeTokenizer()
        counters = {"requests": 0, "tokens": 0}
        model = template_lm(toy_forward, tokenizer, max_length=64, counters=counters)
        pairs = [("Question: why?\nAnswer:", " because"), ("Answer: ", "yes"), ("", "alone")]
        got = model.loglikelihood(self._instances(pairs), disable_tqdm=True)
        for (context, continuation), result in zip(pairs, got, strict=True):
            if context:
                # The trailing space moves into the continuation before
                # tokenizing, as TemplateLM does for every adapter.
                kept = context.rstrip()
                moved = context[len(kept) :] + continuation
                context_ids = tokenizer.encode(kept)
                continuation_ids = tokenizer.encode(kept + moved)[len(context_ids) :]
            else:
                # An empty context is conditioned on the prefix token, the end
                # of text token for a tokenizer with no BOS.
                context_ids, continuation_ids = [0], tokenizer.encode(continuation)
            expected = oracle(context_ids, continuation_ids, 64)
            assert math.isclose(result[0], expected[0], rel_tol=1e-12)
            assert result[1] == expected[1]
        assert counters["requests"] == 3

    def test_special_tokens_follow_the_references_setting(self) -> None:
        pytest.importorskip("lm_eval")
        from mround.eval.zeroshot import template_lm  # noqa: PLC0415

        tokenizer = FakeTokenizer()
        template_lm(toy_forward, tokenizer, max_length=64).tok_encode("text")
        assert tokenizer.calls[-1] == {"add_special_tokens": False}
        template_lm(toy_forward, tokenizer, max_length=64, add_bos_token=None).tok_encode("text")
        assert tokenizer.calls[-1] == {}

    def test_what_is_not_implemented_says_so(self) -> None:
        pytest.importorskip("lm_eval")
        from mround.eval.zeroshot import template_lm  # noqa: PLC0415

        model = template_lm(toy_forward, FakeTokenizer(), max_length=64)
        with pytest.raises(NotImplementedError, match="rolling"):
            model.loglikelihood_rolling([])
        with pytest.raises(NotImplementedError, match="generation"):
            model.generate_until([])


@pytest.mark.needs_mlx
class TestMLXForward:
    def test_scores_match_float64_and_padding_changes_nothing(self) -> None:
        mx = pytest.importorskip("mlx.core")
        from mround.eval.zeroshot import mlx_forward  # noqa: PLC0415

        # A position wise model: the logits at a position depend on that
        # position's token alone, which is causal, so right padding must not
        # move a single result.
        table = np.random.default_rng(3).normal(scale=4.0, size=(VOCAB, 1500)).astype(np.float32)
        weights = mx.array(table)

        def model(ids: Any) -> Any:
            return weights[ids]

        forward = mlx_forward(model, pad_id=0, block=64)
        rows = [(1, 5, 7, 2, 9), (4, 4)]
        got = forward(rows, [3, 2], [(3, 11), (6,)])
        alone = forward([rows[1]], [2], [(6,)])
        for row, window, finals, (inner, greedy, final) in zip(
            rows, [3, 2], [(3, 11), (6,)], got, strict=True
        ):
            positions = range(len(row) - window, len(row))
            reference = [log_softmax(table[row[p]].astype(np.float64)) for p in positions]
            assert greedy == [int(np.argmax(scores)) for scores in reference]
            expected_inner = [reference[k][row[p + 1]] for k, p in enumerate(positions[:-1])]
            np.testing.assert_allclose(inner, expected_inner, rtol=0, atol=2e-5)
            np.testing.assert_allclose(final, [reference[-1][t] for t in finals], atol=2e-5)
        np.testing.assert_allclose(alone[0][0], got[1][0], atol=1e-6)
        np.testing.assert_allclose(alone[0][2], got[1][2], atol=1e-6)

    def test_the_normalizer_is_accurate_over_a_real_vocabulary(self) -> None:
        # The reason it is taken in blocks: over 150,000 entries a single
        # float32 reduction on the CPU backend drifts by 1e-4 (mlx_forward's
        # docstring has the measurement, 1.3e-4 over SmolLM2's 49,152).
        mx = pytest.importorskip("mlx.core")
        from mround.eval.zeroshot import mlx_forward  # noqa: PLC0415

        vocabulary = 151_936
        row = np.random.default_rng(11).normal(scale=5.0, size=vocabulary).astype(np.float32)
        weights = mx.array(row[None, :])

        def model(ids: Any) -> Any:
            return weights[mx.zeros_like(ids)]

        ((inner, _, final),) = mlx_forward(model, pad_id=0)([(0, 0)], [2], [(123,)])
        reference = log_softmax(row.astype(np.float64))
        assert abs(inner[0] - reference[0]) < 2e-5
        assert abs(final[0] - reference[123]) < 2e-5


def test_a_plan_is_plain_data() -> None:
    # Nothing in the plan refers to a framework, so it can be logged, compared
    # and reasoned about anywhere.
    plan = plan_requests([([1, 2], [3])], max_length=8)
    assert isinstance(plan, RequestPlan)
    assert plan.inputs == [(1, 2)]
