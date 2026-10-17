# Copyright 2026 The MRound Authors
# SPDX-License-Identifier: Apache-2.0
"""Measure the block loop's memory on Metal, stage by stage, against the rule.

Run on an Apple Silicon Mac with the project environment active::

    python scripts/probe_memory.py                  # every suite, about half an hour
    python scripts/probe_memory.py --suite pieces   # one of them
    python scripts/probe_memory.py --predict        # the predictions alone, anywhere

Why this exists. ``mround/planner/memory.py`` predicts a run's peak before the
run starts and refuses one predicted past the Mac's budget. Its first fit
rested on three Qwen2.5 sizes at the standard recipe's 2,048 tokens, which
could not tell it two things; this probe measured both on Metal (log0169), and
the rule was refitted on what it found.

1. How a step's activations divide between the MLP, which grows with the
   tokens a step holds, and attention, whose backward grows with the square of
   the sequence length. The ``split`` suite holds the tokens a step sees fixed
   and moves the sequence length from 512 to 4,096, which moves attention's
   share alone: the step grew faster than a sum of the two allows, as the
   larger of them does, and the probe prints that growth beside the rule's.
2. What a 7B step holds whole and in two and four pieces, which decides how
   many pieces a 7B run needs on a 32 GB Mac: the ``pieces`` suite.

The ``shapes`` suite takes three shapes far from the first fit: SmolLM2-135M,
whose attention is wide against its MLP, Qwen2.5-0.5B, and Llama 3 8B, whose
MLP is narrow against its hidden width. Run it again after a change to the
block loop, on another Mac, or for a shape added to :data:`SHAPES`.

Each configuration builds a model of two blocks at the real shape with random
weights, so nothing is downloaded, and runs the real block loop on random
tokens, two steps a block, in a process of its own so that one configuration's
allocations cannot reach the next one's peak. The second block is the one
compared: it holds the first block's codes, as the last block of a real run
holds every finished block's. The rule is asked about the same two block
model, so the comparison is stage for stage, and a configuration whose
predicted peak is past this Mac's budget is skipped, by the rule itself.

Nothing here is a benchmark. It prints numbers, the predicted beside the
measured; paste the whole output when reporting.
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mround.planner.memory import (  # noqa: E402
    BUDGET_SHARE,
    MemoryPrediction,
    ModelShape,
    RunShape,
    predict_memory,
)

RULE = "=" * 78

# The shapes as the models' configurations give them. Two blocks of each are
# built; the vocabulary is small, since the embedding is accounting and a real
# one would add the same bytes to every stage.
SHAPES: dict[str, dict[str, Any]] = {
    "smollm2-135m": {"family": "llama", "hidden": 576, "mlp": 1536, "heads": 9, "kv": 3},
    "qwen2.5-0.5b": {"family": "qwen2", "hidden": 896, "mlp": 4864, "heads": 14, "kv": 2},
    "qwen2.5-3b": {"family": "qwen2", "hidden": 2048, "mlp": 11008, "heads": 16, "kv": 2},
    "qwen2.5-7b": {"family": "qwen2", "hidden": 3584, "mlp": 18944, "heads": 28, "kv": 4},
    "llama-3-8b": {"family": "llama", "hidden": 4096, "mlp": 14336, "heads": 32, "kv": 8},
}
BLOCKS = 2
VOCAB = 1024
STAGES = ("targets", "search", "score", "tune", "commit", "advance")


@dataclasses.dataclass(frozen=True)
class Probe:
    """One configuration: a shape, and the settings its two blocks are tuned at.

    Attributes:
        shape: A key of :data:`SHAPES`.
        batch_size: Sequences per step.
        seq_len: Tokens per sequence.
        micro_batches: Pieces each step's batch runs in.
        batches: Calibration batches, which is what the two streams hold.
    """

    shape: str
    batch_size: int
    seq_len: int
    micro_batches: int = 1
    batches: int = 2

    @property
    def samples(self) -> int:
        """Calibration sequences in all."""
        return self.batch_size * self.batches

    def label(self) -> str:
        """The configuration in a few words."""
        pieces = f", {self.micro_batches} pieces" if self.micro_batches > 1 else ""
        return f"{self.shape}, {self.batch_size} x {self.seq_len}{pieces}"

    def model_shape(self) -> ModelShape:
        """The two block model as the rule sees it."""
        spec = SHAPES[self.shape]
        return ModelShape(
            hidden=spec["hidden"],
            intermediate=spec["mlp"],
            layers=BLOCKS,
            heads=spec["heads"],
            kv_heads=spec["kv"],
            head_dim=spec["hidden"] // spec["heads"],
            vocab=VOCAB,
            tied=True,
            dtype_bytes=2,
        )

    def run_shape(self) -> RunShape:
        """The settings as the rule sees them: 2 bits, searched, group size 64."""
        return RunShape(
            samples=self.samples,
            seq_len=self.seq_len,
            batch_size=self.batch_size,
            micro_batches=self.micro_batches,
            bits=2,
            searched=True,
        )

    def predict(self) -> MemoryPrediction:
        """What the rule says the second block's stages reach."""
        return predict_memory(self.model_shape(), self.run_shape())


SUITES: dict[str, list[Probe]] = {
    # The same 16,384 tokens a step, from 32 sequences of 512 to 4 of 4,096.
    "split": [
        Probe("qwen2.5-3b", 32, 512),
        Probe("qwen2.5-3b", 16, 1024),
        Probe("qwen2.5-3b", 8, 2048),
        Probe("qwen2.5-3b", 4, 4096),
    ],
    "pieces": [Probe("qwen2.5-7b", 8, 2048, pieces) for pieces in (1, 2, 4, 8)],
    "shapes": [
        Probe("smollm2-135m", 8, 2048),
        Probe("qwen2.5-0.5b", 8, 2048),
        Probe("llama-3-8b", 8, 2048),
    ],
}


def measure(probe: Probe) -> dict[str, int]:
    """Run the two blocks and return the second block's stage peaks, in bytes."""
    import mlx.core as mx  # noqa: PLC0415
    from mlx_lm.models import llama, qwen2  # noqa: PLC0415

    from mround.pipeline.device import reset_peak_memory  # noqa: PLC0415
    from mround.pipeline.runner import QuantizationRunner  # noqa: PLC0415
    from mround.schemes import QuantScheme, ScaleInit, TuningConfig  # noqa: PLC0415

    spec = SHAPES[probe.shape]
    family: Any = qwen2 if spec["family"] == "qwen2" else llama
    arguments = family.ModelArgs(
        model_type=spec["family"],
        hidden_size=spec["hidden"],
        num_hidden_layers=BLOCKS,
        intermediate_size=spec["mlp"],
        num_attention_heads=spec["heads"],
        rms_norm_eps=1e-6,
        vocab_size=VOCAB,
        num_key_value_heads=spec["kv"],
        tie_word_embeddings=True,
    )
    mx.random.seed(0)
    model = family.Model(arguments)
    model.set_dtype(mx.bfloat16)
    mx.eval(model.parameters())
    batches = [
        mx.random.randint(0, VOCAB, (probe.batch_size, probe.seq_len)) for _ in range(probe.batches)
    ]
    mx.eval(batches)
    mx.clear_cache()
    reset_peak_memory()
    scheme = QuantScheme(bits=2, group_size=64, scale_init=ScaleInit.SEARCHED)
    tuning = TuningConfig(
        iters=2,
        n_samples=probe.samples,
        seq_len=probe.seq_len,
        batch_size=probe.batch_size,
        micro_batches=probe.micro_batches,
    )
    summary, _ = QuantizationRunner(scheme, tuning).run(model, batches)
    return {stage: int(peak) for stage, peak in summary.blocks[-1].stage_peaks.items()}


def measure_apart(probe: Probe) -> dict[str, Any]:
    """Measure one configuration in a process of its own.

    Returns:
        ``{"stages": ...}`` with the second block's peaks, or ``{"error": ...}``
        with the last line the child printed to its error stream.
    """
    command = [sys.executable, str(Path(__file__).resolve()), "--one"]
    completed = subprocess.run(
        [*command, json.dumps(dataclasses.asdict(probe))],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        lines = completed.stderr.strip().splitlines() or [f"exit {completed.returncode}"]
        return {"error": lines[-1]}
    return {"stages": json.loads(completed.stdout.strip().splitlines()[-1])}


def budget_bytes(gigabytes: float | None) -> int | None:
    """The budget configurations are held to, or ``None`` where nothing says."""
    if gigabytes is not None:
        return round(gigabytes * 1e9)
    try:
        from mround.pipeline.device import device_memory  # noqa: PLC0415
    except ImportError:
        return None
    device = device_memory()
    return round(device * BUDGET_SHARE) if device else None


def report(probe: Probe, prediction: MemoryPrediction, measured: dict[str, int] | None) -> None:
    """Print one configuration's stages, predicted beside measured."""
    print(f"\n{probe.label()}")
    for stage in STAGES:
        predicted = prediction.stages[stage]
        line = f"  {stage:<9}predicted {predicted / 1e9:7.3f} GB"
        if measured is not None and stage in measured:
            error = (predicted - measured[stage]) / measured[stage]
            line += f"   measured {measured[stage] / 1e9:7.3f} GB   {error:+7.1%}"
        print(line)


def report_split(rows: list[tuple[Probe, MemoryPrediction, dict[str, int]]]) -> None:
    """How the step grows with the sequence length at a fixed number of tokens.

    The suite holds a step's tokens fixed, so what grows from one configuration
    to the next is attention's share alone. A sum of an MLP term and an
    attention term would grow in proportion to the sequence length; the larger
    of the two, which is the rule's form since log0169, grows faster once
    attention's overtakes. Printed beside what the rule predicts, the
    increments say which the measurement follows.
    """
    if len(rows) < len(SUITES["split"]):
        print("\nthe split needs every configuration of its suite; one did not run")
        return
    print(f"\n{RULE}\nThe step against the sequence length, at a fixed number of tokens\n{RULE}")
    for (before, was, then), (after, now, measured) in itertools.pairwise(rows):
        grew = (measured["tune"] - then["tune"]) / 1e9
        expected = (now.stages["tune"] - was.stages["tune"]) / 1e9
        print(
            f"  {before.seq_len:>5} to {after.seq_len:<5} tokens a sequence: the step grew "
            f"{grew:6.3f} GB, predicted {expected:6.3f}"
        )


def main() -> int:
    """Run the suites asked for and print what they found."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--suite", choices=sorted(SUITES), action="append", default=None)
    parser.add_argument("--predict", action="store_true", help="Print the predictions alone.")
    parser.add_argument("--memory-budget", type=float, default=None, metavar="GB")
    parser.add_argument("--output", type=Path, default=None, help="Write every result as JSON.")
    parser.add_argument("--one", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.one is not None:
        print(json.dumps(measure(Probe(**json.loads(args.one)))), flush=True)
        return 0

    budget = budget_bytes(args.memory_budget)
    results: list[dict[str, Any]] = []
    split: list[tuple[Probe, MemoryPrediction, dict[str, int]]] = []
    for suite in args.suite or sorted(SUITES):
        print(f"\n{RULE}\nSuite: {suite}\n{RULE}", flush=True)
        for probe in SUITES[suite]:
            prediction = probe.predict()
            entry: dict[str, Any] = {
                "suite": suite,
                "probe": dataclasses.asdict(probe),
                "predicted": prediction.stages,
            }
            measured: dict[str, int] | None = None
            if args.predict:
                pass
            elif budget is not None and prediction.peak > budget:
                entry["skipped"] = f"predicted {prediction.peak / 1e9:.2f} GB, past the budget"
            else:
                outcome = measure_apart(probe)
                entry.update(outcome)
                measured = outcome.get("stages")
            report(probe, prediction, measured)
            for note in ("skipped", "error"):
                if note in entry:
                    print(f"  {note}: {entry[note]}")
            if suite == "split" and measured is not None:
                split.append((probe, prediction, measured))
            results.append(entry)
            sys.stdout.flush()
    if split:
        report_split(split)
    if args.output is not None:
        args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
