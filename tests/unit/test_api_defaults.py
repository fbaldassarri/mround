# Copyright 2026 The MRound Authors
# SPDX-License-Identifier: Apache-2.0
"""What the public entry point does when the caller says nothing.

Every number in `benchmarks/ledger.jsonl` was produced by a harness that passes
`scale_init="searched"` explicitly. Nothing published was ever produced with the
value the entry point shipped, and at 2 bits that value was measured at 77,365
perplexity against the searched grid's 1,361 (MEMORY.md D-030). A default
nobody who measures anything actually uses is not a default, it is a trap for
the first person to call the function the documented way.

These tests are the rule from D-030 pinned where it is applied, and they run
without MLX because `mround.api` imports it lazily, which is the point: a
default this consequential should be checkable without a GPU.
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

import pytest

from mround import api
from mround.api import Coverage, default_scale_init
from mround.cli.main import build_parser
from mround.schemes import LOW_BIT_THRESHOLD, QuantScheme, ScaleInit, Symmetry

if TYPE_CHECKING:
    from collections.abc import Callable


@pytest.mark.parametrize("width", [2, 3])
def test_low_widths_search(width: int) -> None:
    """Below four bits the observed grid is the broken one."""
    assert default_scale_init(width, Symmetry.SYMMETRIC) is ScaleInit.SEARCHED


@pytest.mark.parametrize("width", [4, 5, 6, 8])
def test_four_bits_and_wider_keep_the_cheaper_start(width: int) -> None:
    """Quality neutral at 4 bits, so the search is not made mandatory there."""
    assert default_scale_init(width, Symmetry.SYMMETRIC) is ScaleInit.OBSERVED_RANGE


@pytest.mark.parametrize("width", [2, 3, 4, 8])
def test_an_asymmetric_scheme_never_resolves_to_the_search(width: int) -> None:
    """Because the scheme cannot hold it, not because it would score worse.

    ``QuantScheme`` refuses searched plus asymmetric, as the reference does.
    Before the symmetry argument existed, ``--asym --bits 2`` resolved to
    searched and then died on an error about a setting the person never chose.
    A resolved default has to be a value the scheme can actually take.
    """
    assert default_scale_init(width, Symmetry.ASYMMETRIC) is ScaleInit.OBSERVED_RANGE
    QuantScheme(
        bits=width,
        symmetry=Symmetry.ASYMMETRIC,
        scale_init=default_scale_init(width, Symmetry.ASYMMETRIC),
    )


def test_the_rule_is_the_documented_threshold() -> None:
    """Pinned against the constant rather than against a literal 4.

    The learning rate rule and the outlier suppression rule key on the same
    threshold. If one of them ever moves, this test should move with it rather
    than quietly disagreeing.
    """
    assert default_scale_init(LOW_BIT_THRESHOLD, Symmetry.SYMMETRIC) is ScaleInit.OBSERVED_RANGE
    assert default_scale_init(LOW_BIT_THRESHOLD - 1, Symmetry.SYMMETRIC) is ScaleInit.SEARCHED


def test_a_mixed_run_is_decided_by_its_narrowest_width() -> None:
    """The case the rule exists for, and the one keying on ``bits`` gets wrong.

    A mixed run over {2, 3, 4} carries a base width of 4, and one scale
    initialization covers every layer in the checkpoint. Resolving on the base
    width would hand the 2-bit layers the initialization D-030 measured at
    77,365, which is exactly the headline configuration of this project.
    """
    candidates = (2, 3, 4)
    assert default_scale_init(min(candidates), Symmetry.SYMMETRIC) is ScaleInit.SEARCHED


def _keyword_default(function: Callable[..., object], name: str) -> object:
    """The declared default of a keyword-only parameter."""
    return inspect.signature(function).parameters[name].default


def test_every_entry_point_agrees_on_the_group_size() -> None:
    """The defect this test exists for was a three way disagreement.

    ``api.quantize`` defaulted to 128 while ``quantize_round_to_nearest`` and
    ``plan_mixed_precision`` defaulted to 64, and the command line agreed with
    the odd one out. Nobody noticed because every measurement passes the group
    size explicitly, which is the same reason the other two default defects
    survived: the harness never exercises a default.
    """
    defaults = {
        "quantize": _keyword_default(api.quantize, "group_size"),
        "quantize_round_to_nearest": _keyword_default(api.quantize_round_to_nearest, "group_size"),
        "plan_mixed_precision": _keyword_default(api.plan_mixed_precision, "group_size"),
        "QuantScheme": QuantScheme().group_size,
        "cli": _cli_group_size_default(),
    }
    assert len(set(defaults.values())) == 1, defaults


def _cli_group_size_default() -> object:
    """What ``mround quantize`` uses when ``--group-size`` is not given.

    Parsed rather than introspected. Reaching into argparse's internals would
    test the declaration; parsing a command line tests what a user gets, which
    is the thing that was wrong.
    """
    return build_parser().parse_args(["quantize", "a-model", "-o", "out"]).group_size


def test_the_group_size_default_divides_more_models_than_128() -> None:
    """Why 64 won, stated as the property that decided it.

    A layer is groupable only when its input dimension is a multiple of the
    group size. Every dimension divisible by 128 is divisible by 64 and the
    converse fails often, SmolLM2-135M's hidden size of 576 being the case that
    was measured: at 128 it left 181 of 211 modules dense and produced 13.68
    bits per weight instead of 4.50.
    """
    smollm2_hidden, smollm2_intermediate = 576, 1536
    assert smollm2_hidden % 64 == 0
    assert smollm2_hidden % 128 != 0
    assert smollm2_intermediate % 128 == 0


class TestCoverage:
    """What a group size can actually reach, which one run found out the hard way.

    SmolLM2-135M at group size 128: 30 of 211 modules quantized, 181 left at
    full precision, 13.68 bits per weight where 4 was asked for, and a
    checkpoint three times larger than the 64 arm. Nothing refused it. These
    tests pin the arithmetic and the message, not the threshold, which is a
    policy and lives in one named constant.
    """

    def test_a_clean_model_is_fully_covered(self) -> None:
        assert Coverage(total=210).fraction == 1.0
        assert Coverage(total=210).representable == 210

    def test_the_measured_case(self) -> None:
        blocked = tuple((f"model.layers.{i}.self_attn.q_proj", 576) for i in range(180))
        coverage = Coverage(total=210, blocked=blocked)
        assert coverage.representable == 30
        assert coverage.fraction == pytest.approx(30 / 210)
        assert coverage.fraction < api.MIN_BLOCK_COVERAGE

    def test_a_model_with_no_block_linears_is_not_a_division_by_zero(self) -> None:
        """Nothing there to fail to quantize, so nothing to refuse."""
        assert Coverage(total=0).fraction == 1.0

    def test_the_alternatives_are_the_sizes_that_divide_every_blocker(self) -> None:
        coverage = Coverage(total=4, blocked=(("a", 576), ("b", 1536)))
        assert coverage.alternatives({32, 64, 128}) == (32, 64)

    def test_no_alternative_is_reported_rather_than_invented(self) -> None:
        coverage = Coverage(total=2, blocked=(("a", 100),))
        assert coverage.alternatives({32, 64, 128}) == ()
        assert "No supported group size" in coverage.explain(128, {32, 64, 128})

    def test_the_message_names_the_dimension_and_the_way_out(self) -> None:
        coverage = Coverage(total=210, blocked=(("a", 576),) * 180)
        message = coverage.explain(128, {32, 64, 128})
        assert "30 of 210" in message
        assert "576" in message
        assert "32, 64" in message
