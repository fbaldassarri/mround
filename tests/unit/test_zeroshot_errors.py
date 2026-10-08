# Copyright 2026 The MRound Authors
# SPDX-License-Identifier: Apache-2.0
"""Failures inside ``lm_eval`` reach the caller as ``ZeroShotError``.

``mround eval`` turns an ``MRoundError`` into a message and an exit status and
lets anything else through as a traceback, so a task name ``lm_eval`` does not
know, or a task that asks for generation, has to arrive as this module's own
error. Both cases are built offline: an unknown name fails at task lookup, and
the generation task reads a JSON file written beside the test, with the
dataset cache kept in the test's own directory.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

pytest.importorskip("mlx.core", reason="the evaluator builds an MLX forward")
pytest.importorskip("lm_eval", reason="the task machinery is lm_eval's")

from mround.eval.zeroshot import ZeroShotError, evaluate_zeroshot

if TYPE_CHECKING:
    from pathlib import Path


class _Tokenizer:
    bos_token_id = None
    eos_token_id = 1
    pad_token_id = 0

    def encode(self, text: str, **_: Any) -> list[int]:
        return [2 + (ord(c) % 60) for c in text]

    def decode(self, ids: Any, **_: Any) -> str:
        if isinstance(ids, int):
            ids = [ids]
        return "".join(chr(97 + (i % 26)) for i in ids)


def test_an_unknown_task_name_is_a_zero_shot_error() -> None:
    with pytest.raises(ZeroShotError, match="no_such_task"):
        evaluate_zeroshot(None, _Tokenizer(), tasks=["no_such_task"], limit=1)


def test_a_generation_task_is_refused_as_a_zero_shot_error(tmp_path: Path) -> None:
    data = tmp_path / "data.jsonl"
    data.write_text('{"text": "hello world"}\n', encoding="utf-8")
    task = tmp_path / "generation.yaml"
    task.write_text(
        "task: mround_generation_probe\n"
        "dataset_path: json\n"
        "dataset_kwargs:\n"
        "  data_files:\n"
        f"    test: {data}\n"
        f"  cache_dir: {tmp_path / 'datasets'}\n"
        "test_split: test\n"
        "output_type: generate_until\n"
        'doc_to_text: "{{text}}"\n'
        'doc_to_target: "{{text}}"\n'
        "generation_kwargs:\n"
        '  until: ["\\n"]\n'
        "metric_list:\n"
        "  - metric: exact_match\n",
        encoding="utf-8",
    )
    with pytest.raises(ZeroShotError, match="generation is not implemented"):
        evaluate_zeroshot(None, _Tokenizer(), tasks=[str(task)], limit=1)
