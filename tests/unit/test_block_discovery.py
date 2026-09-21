# Copyright 2026 The MRound Authors
# SPDX-License-Identifier: Apache-2.0
"""The rules that decide which list of modules is the transformer.

Discovery has to run on a loaded model, but the decision it makes is about
dotted path strings and nothing else, so the decision is testable here without
MLX and without a download. That split is deliberate: the part most likely to
pick the wrong thing on an architecture nobody has tried is the part that costs
nothing to test exhaustively.

The failure this guards against does not raise. A quantizer pointed at the wrong
stack writes a checkpoint that loads cleanly and generates nonsense, and the
first symptom is a perplexity number somebody spends a day trying to explain.
"""

from __future__ import annotations

import pytest

from mround.exceptions import ArchitectureError
from mround.pipeline.blocks import _group_by_index, _is_stack, _select_stack

# A Llama-shaped model as `named_modules` reports it, trimmed to two blocks.
LLAMA_PATHS = [
    "",
    "model",
    "model.embed_tokens",
    "model.layers",
    "model.layers.0",
    "model.layers.0.input_layernorm",
    "model.layers.0.mlp",
    "model.layers.0.mlp.down_proj",
    "model.layers.0.mlp.gate_proj",
    "model.layers.0.mlp.up_proj",
    "model.layers.0.self_attn",
    "model.layers.0.self_attn.k_proj",
    "model.layers.0.self_attn.o_proj",
    "model.layers.0.self_attn.q_proj",
    "model.layers.0.self_attn.v_proj",
    "model.layers.1",
    "model.layers.1.input_layernorm",
    "model.layers.1.mlp",
    "model.layers.1.mlp.down_proj",
    "model.layers.1.self_attn",
    "model.layers.1.self_attn.q_proj",
    "model.norm",
    "lm_head",
]


class TestGrouping:
    def test_a_block_stack_groups_under_its_parent(self) -> None:
        groups = _group_by_index(LLAMA_PATHS)
        assert groups["model.layers"] == {0: "model.layers.0", 1: "model.layers.1"}

    def test_paths_without_a_trailing_integer_are_not_list_elements(self) -> None:
        groups = _group_by_index(LLAMA_PATHS)
        assert "model" not in groups
        assert all(not path.endswith(("norm", "proj", "head")) for path in groups)

    def test_the_root_is_ignored(self) -> None:
        # `named_modules` reports the model itself under the empty path, and
        # rpartition on it yields an empty final segment rather than an index.
        assert _group_by_index([""]) == {}

    def test_indices_are_integers_not_strings(self) -> None:
        # The bug this catches: ordering blocks lexicographically puts layer 10
        # between 1 and 2, which produces a model quantized in the wrong order
        # against activations from the wrong depth.
        paths = [f"model.layers.{i}" for i in range(12)]
        keys = _group_by_index(paths)["model.layers"].keys()
        assert sorted(keys) == list(range(12))


class TestStackShape:
    def test_a_contiguous_run_from_zero_is_a_stack(self) -> None:
        assert _is_stack({0, 1, 2, 3})

    def test_one_element_is_not_a_stack(self) -> None:
        # A single module that happens to live in a list. Every real decoder has
        # at least two blocks, and treating a list of one as a stack would match
        # things like a wrapped output head.
        assert not _is_stack({0})

    def test_a_gap_is_not_a_stack(self) -> None:
        assert not _is_stack({0, 1, 3})

    def test_not_starting_at_zero_is_not_a_stack(self) -> None:
        assert not _is_stack({1, 2, 3})

    def test_empty_is_not_a_stack(self) -> None:
        assert not _is_stack(set())


class TestSelection:
    def test_the_only_candidate_wins(self) -> None:
        assert _select_stack({"model.layers": 32}) == "model.layers"

    def test_the_outermost_stack_wins(self) -> None:
        # A mixture-of-experts model has a list of experts inside every block,
        # and those lists pass every structural test a block stack passes. The
        # transformer is the outer one.
        candidates = {
            "model.layers": 32,
            "model.layers.0.mlp.experts": 8,
            "model.layers.1.mlp.experts": 8,
        }
        assert _select_stack(candidates) == "model.layers"

    def test_a_larger_but_deeper_stack_does_not_win(self) -> None:
        # Depth decides, not size. A model with more experts than layers is
        # ordinary and must not flip the answer.
        assert _select_stack({"model.layers": 4, "model.layers.0.mlp.experts": 128}) == (
            "model.layers"
        )

    def test_two_stacks_at_the_same_depth_refuse_to_be_guessed_between(self) -> None:
        with pytest.raises(ArchitectureError, match="equally plausible"):
            _select_stack({"model.layers": 12, "model.cross_layers": 12})

    def test_the_refusal_names_both_candidates(self) -> None:
        # The message is the entire value of refusing. Without the names the
        # reader cannot tell which architecture confused it.
        with pytest.raises(ArchitectureError) as caught:
            _select_stack({"model.layers": 12, "model.cross_layers": 6})
        assert "model.layers" in str(caught.value)
        assert "model.cross_layers" in str(caught.value)

    def test_no_candidates_says_what_was_looked_for(self) -> None:
        with pytest.raises(ArchitectureError, match="no transformer block stack"):
            _select_stack({})

    def test_a_stack_at_the_root_is_reachable(self) -> None:
        # Nothing requires a model to nest its blocks under a submodule, and a
        # depth rule that assumed one would exclude the shallowest case there is.
        assert _select_stack({"layers": 24, "model.layers.0.mlp.experts": 8}) == "layers"
