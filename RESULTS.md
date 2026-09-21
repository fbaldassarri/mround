# Results

Every number MRound publishes, what produced it, and what it does not show.

Each row here corresponds to a line in [`benchmarks/ledger.jsonl`](benchmarks/ledger.jsonl),
cited by its row number so that any figure can be traced to the run that made
it. The ledger is append only: corrections are new rows, never edits, so it
contains superseded numbers as well as current ones and row numbers are stable.
Every scored run also records the full package inventory of the machine it ran
on as a hashed manifest under `benchmarks/environments/`, cited from the row.

## The bar

MRound's acceptance criterion is not a threshold someone chose. It is Intel
AutoRound's own result at the same model, the same scheme, the same calibration
data and the same evaluator, with both sides' absolute numbers reported beside
the comparison so that a weak opponent cannot flatter a broken model.

AutoRound is the reference implementation of the method MRound implements. It
is a serious piece of work by the people who published the method, and beating
it is the only comparison that means anything.

## What is measured, and how

**Quality** is perplexity on wikitext2, 65,504 tokens in non overlapping windows
of 2,048, computed in float32 for every model in a comparison including the
original. One evaluator scores all of them in one process. A perplexity is only
comparable to another perplexity measured the same way, which is why the
original is reported next to every result rather than assumed.

**Both sides quantize the same tensors.** The reference leaves the language
model head at full precision. On a model that ties its embedding to its head,
matching that is the only way to compare like with like, so MRound is told to
leave the embedding, the head and the final normalization dense too. This
matters more than it sounds: on Qwen2.5-0.5B at 4 bits, the same MRound
checkpoint is 4.90 percent worse than the reference when it also quantizes the
embedding the reference keeps (row 29) and 0.02 percent worse when coverage is
matched (row 30). A comparison that does not say which it did is not a
comparison.

**Size is reported with quality, always.** A quantizer can buy perplexity with
bytes. MRound's checkpoints run 2.4 to 4.9 percent larger than the reference's
at the same nominal width, because it stores its scales differently, and that
difference is stated in every table below rather than left out.

**Cost is the weakest number here and is labelled as such.** MRound runs on the
Apple GPU through MLX; the reference has no Metal path, so it runs on a Linux
CPU. Every wall clock ratio below therefore compares two machines as much as two
implementations. It is reported because it is what a user experiences, not
because it isolates anything.

## Uniform 4 bits

| model | original | reference | MRound | MRound against it | size | ledger |
|---|---|---|---|---|---|---|
| Qwen2.5-0.5B-Instruct | 14.3248 | 15.0777 | 15.0807 | +0.02 percent | +2.43 percent | row 30 |
| SmolLM2-135M-Instruct | 17.5077 | 22.1758 | 18.9400 | **-14.59 percent** | +3.00 percent | row 54 |

At 4 bits on the larger model the two implementations are indistinguishable:
three parts in ten thousand, which is smaller than the difference between two
seeds of either. On the smaller model, where 4 bits is a harder ask relative to
the model's capacity, MRound is ahead by 14.59 percent.

## Uniform 2 bits

Three seeds on Qwen2.5-0.5B-Instruct, changing only the calibration draw:

| seed | reference | MRound | MRound against it | ledger |
|---|---|---|---|---|
| 42 | 58.3402 | 41.2332 | -29.32 percent | row 31 |
| 43 | 49.4642 | 36.9670 | -25.27 percent | row 34 |
| 44 | 50.3438 | 39.3524 | -21.83 percent | row 37 |
| **mean** | **52.7161** | **39.1842** | **-25.67 percent** | |

The margin survives the spread: MRound's worst draw beats the reference's best
by 16.64 percent. Size is +3.02 percent throughout. The mean quoted is the ratio
of the means; averaging the three per seed ratios instead gives 25.47 percent.

Two more models at the same scheme, one seed each:

| model | original | reference | MRound | MRound against it | size | ledger |
|---|---|---|---|---|---|---|
| SmolLM2-135M-Instruct | 17.5077 | 86.9820 | 58.4582 | -32.79 percent | +3.92 percent | row 35 |
| Qwen2.5-1.5B-Instruct | 9.6651 | 23.9091 | 21.4767 | -10.17 percent | +4.91 percent | row 36 |

Note the absolute numbers. At 2 bits both implementations produce models far
worse than the original, and the interesting claim is the gap between them, not
that either result is good. The margin also narrows as the model grows, which is
the honest caveat on every number in this document: they come from models under
two billion parameters.

## Mixed precision at an average of 2.5 bits

Per layer widths chosen from {2, 3, 4} under an exact size budget, group size 64,
with the tensors outside the block stack protected at 3 bits. Both sides receive
the same sensitivity scoring budget. Three seeds each, on two architectures:

**Qwen2.5-0.5B-Instruct**, original 14.3248:

| seed | reference | MRound | MRound against it | ledger |
|---|---|---|---|---|
| 42 | 28.4685 | 22.9964 | -19.22 percent | row 43 |
| 43 | 28.3806 | 21.6389 | -23.75 percent | row 45 |
| 44 | 27.7606 | 22.4770 | -19.03 percent | row 46 |
| **mean** | **28.2033** | **22.3707** | **-20.68 percent** | |

**SmolLM2-135M-Instruct**, original 17.5077:

| seed | reference | MRound | MRound against it | ledger |
|---|---|---|---|---|
| 42 | 51.3966 | 31.2683 | -39.16 percent | row 49 |
| 43 | 52.3740 | 29.8179 | -43.07 percent | row 50 |
| 44 | 52.3194 | 30.0735 | -42.52 percent | row 51 |
| **mean** | **52.0300** | **30.3866** | **-41.60 percent** | |

On both models the worst MRound draw beats the best reference draw: by 17.16
percent on Qwen and 39.16 on SmolLM2. Size is +2.85 and +3.64 percent.

One structural difference is worth reporting because it is stable across both
architectures. Given the same three candidate widths and the same budget, the
reference never allocated a single layer at the widest option, spending the
budget as a near binary choice between 2 and 3 bits, while MRound used all
three. That is a property of the two allocators rather than of one model.

## The reference at its own strongest recipe

Every comparison above runs both sides at the reference's default recipe, 200
optimization steps over 128 calibration samples. The reference also ships a
stronger one, 1,000 steps over 512 samples. Both sides at both recipes, seed 42,
mixed 2.5, Qwen2.5-0.5B-Instruct:

| recipe | reference | MRound | MRound against it | ledger |
|---|---|---|---|---|
| 200 steps, 128 samples | 28.4685 | 22.9964 | -19.22 percent | row 43 |
| 1,000 steps, 512 samples | 25.8199 | 22.3231 | -13.54 percent | row 48 |

The longer schedule helps the reference three times as much as it helps MRound,
so the margin narrows from 19.22 to 13.54 percent. It does not close. MRound at
its default still beats the reference at its best, by 10.94 percent.

## Which release of the reference

The reference's own version matters at low widths and not at 4 bits, which is
worth knowing before reading any of the above.

Every mixed 2.5 row here comes from auto-round 0.15.0, one release throughout.
The 4 bit SmolLM2 comparison was rerun on 0.15.0 with the packed export, and the
reference moved 0.12 percent from its August result under 0.14.2 with a
dequantized export. **The three seed 2 bit table is the exception**: it comes
from 0.14.2, and only seed 43 has been reproduced under 0.15.0, where the
reference moved from 49.4642 to 49.3534, or 0.22 percent (row 33). At mixed 2.5
the release has been measured as worth about three percent in either direction,
so a 2 bit table fully reproduced on 0.15.0 is work still to do.

## What these numbers do not show

They are three models, all under two billion parameters, on one evaluation
corpus, scored by perplexity alone. No zero shot task suite has been run. The
largest model measured is Qwen2.5-1.5B and it shows the smallest margin, so
nothing here should be read as a claim about what happens at seven billion.

Cost ratios compare an Apple GPU against a Linux CPU, as above.

Perplexity is a proxy. It is the standard one in this literature and it is what
makes these results comparable to published work, but a model that scores well
on it can still be worse in ways a user notices.

## Reproducing any of this

Every row names its model, scheme, calibration seed and recipe, and every run
writes a configuration file beside its checkpoint that reproduces it. The
reference arms were produced with Intel AutoRound at the version the row's notes
name, on a Linux machine whose package inventory is recorded in
`benchmarks/environments/` under the hash the row cites. The scoring harness
that produced the comparisons is not part of this repository, because it drives
the reference implementation; what is here is enough to reproduce MRound's side
and to score any two checkpoints against each other.

```bash
mround quantize mlx-community/SmolLM2-135M-Instruct -o ./smollm2-4bit --bits 4
mround eval ./smollm2-4bit
```
