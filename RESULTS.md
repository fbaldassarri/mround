# Results

Every number MRound publishes, what produced it, and what it does not show.

Each row here corresponds to a line in [`benchmarks/ledger.jsonl`](benchmarks/ledger.jsonl),
cited by its row number so that any figure can be traced to the run that made
it. The ledger is append only: corrections are new rows, never edits, so it
contains superseded numbers as well as current ones and row numbers are stable.
Every scored run from row 29 on also records the full package inventory of the
machine it ran on as a hashed manifest under `benchmarks/environments/`, cited
from the row.

## The bar

MRound's acceptance criterion is not a threshold someone chose. It is Intel
AutoRound's own result at the same model, the same scheme, the same calibration
data and the same evaluator, with both sides' absolute numbers reported beside
the comparison so that a weak opponent cannot flatter a broken model.

AutoRound is the reference implementation of the method MRound implements. It
is a serious piece of work by the people who published the method, and beating
it is the only comparison that means anything.

One setting is matched in six tables only, and it is the reference's own
algorithm. AutoRound keeps its implementation of SignRound v2's tuning, a
searched initial scale and, below 4 bits, the outlier suppressed loss, behind an
option, `enable_alg_ext`, that is off by default, and its documentation says to
turn it on to reproduce the SignRound v2 results. Six tables have been run both
ways. At 2 bits on Qwen2.5 at 0.5, 1.5, 3 and 7 billion parameters, with MRound
at its current learning rate (rows 151 to 174): MRound is 9.10, 6.38, 3.43 and
2.05 percent ahead with the option on and 30.52, 17.21, 15.42 and 1.96 with it
off, ahead on every seed of all eight (Uniform 2 bits, below, with what the
pairs leave open). At mixed 2.5 on Qwen2.5-0.5B-Instruct, rows 108 to 110 beside
rows 43, 45 and 46: 12.88 percent ahead with it on and 20.68 with it off (Mixed
precision, below). At mixed 2.5 on Qwen2.5-3B-Instruct, MRound as it ships, its
own allocation scored with a gradient at each width's own quantized point as the
reference scores (rows 201 to 206): 2.02 percent ahead with it on and 6.71 with
it off, on every seed. Before its default took those gradients at that size,
MRound took them at the widest width and was 3.46 percent behind with the option
on and 1.50 ahead with it off (rows 181 to 186), the one table where it missed
the bar; tuned at the reference's own allocation it was 0.75 and 5.51 percent
ahead (rows 195 to 200), so the gap was the gradient its allocator scored with.
Every other reference arm in this document ran with it off. The mixed precision
arms use the reference's own allocator for that method, AutoScheme, with the
option and without it. So outside those six tables every comparison below sets
MRound against the reference's default tuning, how far the option would move the
reference there has not been measured yet, and no table, those six included,
sets MRound against the configuration the reference's documentation recommends
at 2 bits (What these numbers do not show).

## What is measured, and how

**Quality** is perplexity on wikitext2, 65,504 tokens in non overlapping windows
of 2,048, computed in float32 for every model in a comparison including the
original. One evaluator scores all of them in one process. A perplexity is only
comparable to another perplexity measured the same way, which is why the
original is reported next to every result rather than assumed.

One wrinkle in the ledger is worth naming, since anyone checking these numbers
against it will find it. Rows 10 to 44 record that window as 65,536, which was
the count requested rather than the count scored; from row 45 the rows carry
both, `requested_tokens` and `tokens`. The evaluation itself never changed, and
the rows on either side of that boundary prove it: they report the same model's
original perplexity as the same seventeen significant figures, which two
different token counts could not produce.

**Zero shot task accuracy** has been measured on seven tables so far, all on
Qwen2.5-0.5B-Instruct and all below: the 4 bit one with its group size pair, the
mixed 2.5 ones against each of the reference's two tunings, and the 2 bit ones
against each, on MRound's current checkpoints and on those of its earlier rate.
Against its default tuning the suite shows the perplexity lead clearly at 2
bits, equivocally at mixed 2.5, and not at 4 bits, where the perplexity itself
calls a tie; against its own SignRound v2 tuning it puts MRound 0.13 points
ahead at 2 bits on the current checkpoints, behind on two seeds of three, where
those of the earlier rate were 0.49 ahead on every seed, and 0.10 ahead at mixed
2.5. The suite is the reference implementation's own default less wikitext:
twelve tasks through `lm_eval` 0.4.12's task definitions, full sets, zero shot,
with `acc` the headline on every task. One evaluator, MRound's, scores the
original and both checkpoints in one process, each after reproducing its row's
perplexity to a millionth. It was checked against `lm_eval`'s own evaluator on
the same documents before any number was recorded (`scripts/check_zeroshot.py`,
whose output is not a ledger row): run on the Mac's CPU against `lm_eval` in
float64, every request agreed to 2.2e-4 nats and every right or wrong call was
identical (twenty documents a task on SmolLM2-135M-Instruct, five on
Qwen2.5-0.5B-Instruct). The recorded scores ran on the Apple GPU, as every
perplexity here does, where MLX's float32 matrix product takes its inputs at
about float16's precision. The same check run there moved each request by about
a hundredth of a nat and changed two of its 1,800 calls, both between answers
within two thousandths of a nat of each other; at that rate the noise moves a
task's accuracy by much less than the task's standard error, and it is stated
here rather than removed. It is not a draw between runs: the original model,
scored again beside each of the twenty comparisons, gave the same figures to the
last digit every time.

**Both sides quantize the same tensors.** The reference leaves the language
model head at full precision. On a model that ties its embedding to its head,
matching that is the only way to compare like with like, so MRound is told to
leave the embedding, the head and the final normalization dense too. This
matters more than it sounds: on Qwen2.5-0.5B at 4 bits, the same MRound
configuration is 4.90 percent worse than the reference when it also quantizes
the embedding the reference keeps (row 29) and 0.02 percent worse when coverage
is matched (row 30, an older MRound tree against the reference's arm on the
Mac's CPU, the 4 bit comparison this document quoted until 2026-09-28). A
comparison that does not say which it did is not a comparison.

**Size is reported with quality, always.** A quantizer can buy perplexity with
bytes. MRound's checkpoints run 2.4 to 6.2 percent larger than the reference's
at the same nominal width and group size 64, and 1.2 percent larger at group
size 128, because the format it writes stores a bias beside every group's
scale, and that difference is stated in every table
below rather than left out. The bias costs a fixed share of the weights a
side quantizes and nothing of the tensors both keep dense, so the difference
grows with the share of the model its block stack holds: at 2 bits it is 3.02
percent on Qwen2.5-0.5B-Instruct and 6.19 on Qwen2.5-3B-Instruct, and in both
the tensors MRound stores exceed the reference's packed bytes by exactly the
biases. Where the two are compared at the same bytes
instead, it says so.

**Cost is the weakest number here and is not reported as a ratio.** MRound
runs on the Apple GPU through MLX; the reference defaults to the CPU on Apple
Silicon (its device manager redirects there and offers `--device_map=mps` as an
override), so every reference arm in the tables below ran on a CPU, on the
Linux machine for every row but row 35, which ran on the Mac's own CPU,
except the Qwen2.5-1.5B-Instruct and Qwen2.5-3B-Instruct tables (rows 92 to
94 and 124 to 129), whose reference arms ran on the Linux machine's integrated
GPU, as described below. Every row
records which under `cost.reference_host`, and rows 55 to 79 restate each
reference arm's machine, release, torch build and device in a
`reference_environment` block, one provenance row per head to head row. A
wall clock ratio between a GPU and a CPU compares two devices as much as two
implementations, and one between two machines compares those as well, so no
table below carries one. Since 2026-09-24 the reference has also been run on
the Mac's own GPU, through torch's MPS backend under `--device_map=mps`, held
to float32 with autocast off as its CPU arms were (rows 82 and 83, one seed
each). That is the one pair where a wall clock ratio compares two
implementations on one device: at 4 bits on Qwen2.5-0.5B-Instruct the
reference's block loop took 5,379.7 s on the Apple GPU against MRound's
3,495.7 s on the same GPU, and its perplexity there was 15.2537 against
15.0777 for its own CPU arm and 15.0807 for MRound; on SmolLM2-135M-Instruct
3,090.0 s against 1,885.5 s, at 22.1518 against MRound's 18.9400. The
reference ran uncompiled, which is auto-round 0.14.2's default, and its own
default precision on an accelerator, bfloat16 under autocast, was not used;
both are stated because either could move its time. Run at that default
precision on the same GPU, still uncompiled (row 85, one seed), the
reference's block loop took 3,730.1 s at a perplexity of 15.1561, so with
each implementation at its own default precision MRound's block loop is
faster by 6.7 percent and its perplexity better by half a percent, not the
factor of 1.54 the float32 pair shows; that pair is the matched precision
comparison and this one is the reference at its own default precision on that
GPU under 0.14.2. Both are stated. Rerun on the tree of 2026-09-25 (row 86,
the same
seed and recipe), MRound's block loop took 3,464.6 s at 15.0260 against that
arm's 3,730.1 s at 15.1561, 7.7 percent faster and 0.86 percent better, and
that row is the first with memory on both sides: MRound's MLX allocator peaked
at 13.58 GB and the reference's own monitor reported 12.66 GB reserved on the
device in the run's log (it prints 11.79, its gigabytes being binary; the row
leaves that field empty, and this document said 11.79 beside 13.58 until
2026-09-27), each framework's own count over the same
unified memory; the processes' maximum resident sets were 1.91 GB and 3.49
GB, which on a Mac exclude Metal's allocations and are stated as what they
are. Neither run approached the machine's memory at this model size. Run
again three days later on the next labelled tree with the same configuration
(row 95), MRound wrote the
identical checkpoint in 3,606.0 s at an allocator peak of 13.92 GB, 4.1 and
2.5 percent above row 86's, which is how far its own time and memory move
between two identical runs. The Linux machine's own
integrated GPU, Intel Arc graphics, has run the reference too, under
auto-round 0.15.0 with float32 held and `torch.compile` on by that release's
default (row 84, mixed 2.5 on Qwen2.5-0.5B-Instruct, one seed): 6,041.8 s of
block loop against 17,445.3 s for the same production on that machine's CPU
(row 45), at 29.2404 against the CPU arm's 28.3806 and MRound's 21.6389. At
its own default precision there, bfloat16 under autocast, the reference could
not be compiled: the graphics compiler shipped for that device refuses the
bfloat16 SPIR-V extension the kernels use, a fact about the platform that is
recorded rather than worked around. Uncompiled at that precision (row 87) it
took 4,947.2 s at 27.3604, its fastest and best arm at this seed, against
MRound's 3,831.7 s at 21.6389 on the Apple GPU. The same arm at 2 bits on
Qwen2.5-0.5B-Instruct, seeds 43 and 44 (rows 90 and 91), took 4,952.9 and
4,958.6 s against 17,750.0 and 17,333.9 s for the CPU arms of the same seeds
and release (rows 33 and 89), at 45.1310 and 55.4526 against the CPU arms'
49.3534 and 63.1077 and MRound's 36.9670 and 39.3524 in 3,836.4 and 3,908.3
s on the Apple GPU; the Arc arm is 3.5 times faster than the CPU arm at both
operating points measured, and MRound's block loop is 1.3 times faster than
the Arc arm at both. At Qwen2.5-1.5B-Instruct, 2 bits, three seeds (rows 92
to 94), the Arc arm took 10,983.5 to 11,003.8 s against MRound's 8,658.5 to
9,419.8 s; on the seed where the Mac was doing nothing else (seed 44; a
block's work does not depend on the seed, and the other two runs' logs show
blocks up to 42 percent slower) MRound's block loop was 1.270 times faster, as
it was at 0.5B on the same seed (1.268). Both implementations took 2.2 times
as long for three times the parameters, where the reference on the Linux CPU
took 4.1 times as long (rows 31 and 36, under 0.14.2), so the lead that
widens with the model is the Arc graphics' over its own CPU, not MRound's
over the reference. At that size MRound's allocator peaked at 27.27 GB, four
fifths of the Mac's 32 GiB, and the reference's pool on the Arc graphics at
12.05 GB with 10.87 GB resident (MRound's current tree writes the same
checkpoint to the byte at a peak of 19.34 GB, row 121, having stopped holding
the whole model, a third activation stream and its codes in float32, and at
8.26 GB with every tuning step in eight pieces, row 122, which is another
draw rather than the same checkpoint: 21.1863 against 20.8947, the two
checkpoints 0.33 apart, as two seeds of the reference are); the
reference moves one block at a time to
its device and keeps the rest on the host, so those two are not one quantity
either. At Qwen2.5-3B-Instruct, 2 bits, three seeds (rows 124 to 126), the Arc
arm took 21,758.6 to 21,770.9 s against MRound's 16,497.2 to 16,561.5 s, and
MRound's block loop was 1.31 to 1.32 times faster on every seed; on seed 44
the Arc arm took 1.98 times as long as at 1.5 billion and MRound, on a later
tree, 1.91 times, for twice the parameters. There MRound's allocator peaked at
20.07 GB on every seed, three fifths of the Mac's 32 GiB, and the reference's
pool on the Arc graphics at 15.90 GB with 20.18 GB resident. Those pairs put
each implementation on its own machine's integrated GPU, which removes the CPU
against GPU category difference and not the device difference, so they are
reported with both machines named and not as a ratio.
Before these rows the only same machine pair was 4 bit Qwen2.5-0.5B-Instruct
at 3,495.7 s for MRound on Metal against 6,718.3 s for the reference on the
Mac's CPU (row 44), which compares a GPU with a CPU.

**A reference arm is one draw, and the ledger now says by how much.** Every
accelerator arm above moved the reference's perplexity against its CPU arm at
the same seed: by 1.17 and 0.52 percent at 4 bits on Qwen2.5-0.5B-Instruct
(float32 held, and the reference's own bfloat16 default, against the Mac's
CPU arm of row 30; the Linux CPU under 0.15.0, row 95, sits 0.94 percent from
that arm), 3.03 percent at mixed 2.5 and 3.59 percent the other way at the
reference's own precision there (rows 84 and 87 against row 45), 0.22 and 0.11
percent the other way on SmolLM2-135M-Instruct (row 82 against the CPU arms of
rows 9 and 54), and 8.56 and 12.13 percent the other way at 2 bits (rows 90 and
91 against 33 and 89). At Qwen2.5-1.5B-Instruct the change of release, device
and precision together moved seed 42 by 1.10 percent (row 36 against row 92).
Comparing the checkpoints themselves code by code
(`scripts/compare_checkpoints.py`) shows why: two reference arms that differ
only in the device they tuned on differ in 64 percent of their weights at
mixed 2.5 and 85 percent at 4 bits, with the learned clipping ending a median
of 2 percent away in 94 percent of groups, and that is exactly what a change
of release (66 percent, at the same seed) or of calibration seed (65 percent)
also does, and what a change of tuning precision on one device does (0.095 at
4 bits), while the same seed produced twice on one device and release
reproduces all 357,826,560 weights, under either of the reference's tunings,
and still does after the Linux machine's operating system was upgraded to a new
release and then reinstalled, its software environment rebuilt (rows 117 and
118 against rows 88 and 105, below). The reference on the integrated GPU
repeats itself too: seed 44's 2 bit arm, produced again there after the
reinstall, on another release of the GPU's compute runtime and in its
environment rebuilt, is row 91's checkpoint to the bit, with row 91's
perplexity and the same device memory to the byte (row 123). MRound does the
same: one configuration
produced twice on the Apple GPU, three days apart and on two labelled trees
(rows 86 and 95), wrote the same checkpoint file to the byte. The block losses
the reference prints do not see the device change: the accelerator arms' losses
agree with the CPU arms' to a fifth of a percent on average, to six decimals
where the release prints six, with the same best step on nearly every block.
Two consequences for reading this document. The weight level disagreement
between MRound and the reference (the mean relative root mean square
difference over layers, relative to MRound's own weights: 0.46 to 0.48 at 2
bits against the reference's default checkpoints, which at 0.5 billion is 0.45
relative to theirs, and 0.37 against its SignRound v2 checkpoints on either
denominator at 0.5 billion and 0.35 at 3 billion; 0.33 at mixed 2.5, and 0.27
to 0.28 there against its SignRound v2 checkpoints on either denominator; 0.10
to 0.12 at 4 bits) has a floor: the
reference disagrees with itself by 0.30 to 0.34 at 2
bits (across a change of release or of seed), 0.22 to 0.24 at mixed 2.5
(device, release or seed) and 0.095 at 4 bits (device or precision),
so at 4 bits the two implementations are about as alike as two productions of
the reference are. At 2 bits the floor was measured on the three release
pairs of the first 2 bit table below (rows 31, 34 and 37 against 88, 33 and
89): the three checkpoints moved by the same 0.31 to 0.32, 54.5 to 54.6 percent
of their weights and one shared lattice code in eighteen, while their
perplexities moved by 2.30,
0.22 and 25.35 percent, so the distance between two checkpoints says nothing
about the distance between their perplexities, any more than the block losses
do; two calibration seeds under one release sit 0.34 apart there (the
reference checkpoints of rows 88, 33 and 89, measured on 2026-10-02, before
which this document gave the 2 bit floor as 0.30 to 0.32). At
Qwen2.5-1.5B-Instruct the reference's two seed 42 arms, rows 36 and 92,
which differ in release, device and precision at once, sit 0.30 apart and score
1.10 percent apart, while MRound's checkpoints sit 0.47 from the reference's.
At 0.5 billion MRound is 1.32 to 1.45 times as far from the reference's default
arms as two of those arms are from each other, both taken relative to a
default arm (0.45 against 0.31 to 0.34). At 1.5 billion the figures as
printed, MRound's relative to its own weights, give about 1.6 times (0.47
against 0.30), and the two have not been put on one denominator there. At 3
billion MRound's checkpoints sit 0.46 to 0.47 from the reference's default arms
and 0.35 from its arms with the option on, on MRound's denominator, and no two
of the reference's arms there have been compared. The
reference's own
SignRound v2 tuning is a change of another kind (the reference checkpoints
behind rows 105 to 107 beside those behind rows 88, 33 and 89, compared with
`scripts/compare_checkpoints.py`): at one seed its two checkpoints sit 0.40 to
0.41 apart relative to the default checkpoint (0.43 relative to the other, the
denominator used below), share a lattice in three groups in a thousand where
two draws
share 1.6 to 5 percent (4 to 5 between two default arms, 1.6 to 1.7 between two
arms with the option on), and differ in their scales by a median of 14 to 15
percent of the default checkpoint's scale (17 of the other's) where two draws
differ by 3 to 4. Those two checkpoints were also produced under different
operating system releases of one machine, and that changed nothing: produced
again on one system, each is the same checkpoint to the bit (rows 117 and 118),
so the two differ in the option alone.
Measured from the checkpoint with the option on and relative to it, MRound's
checkpoint sits 0.37 away and the reference's own default checkpoint 0.43,
MRound the nearer in 164 or 165 of the 168 layers, with scales a median of 5.4
to 5.5 percent from that arm's over all groups, where the default arm's are
16.5 to 16.8; MRound's
figures there are not the script's, which reads float exports only, but the
scoring harness's comparison and MRound's stored scales taken on that
denominator. Two arms with the option on, at different seeds, sit 0.30 to 0.31
apart, so MRound is 1.2 times as far from such an arm as two of them are from
each other, and the reference's default arm 1.4 times. At mixed 2.5 the same
comparison (the reference checkpoints behind rows 108 to 110 beside those
behind rows 43, 45 and 46, all six of which give every layer the same width)
reads the same way and smaller: the two checkpoints of a seed sit 0.284 to
0.285 apart relative to the default checkpoint (0.296 to 0.297 relative to the
other), share a lattice in seven groups in a thousand where two draws share 2
to 5 percent, and differ in their scales by a median of 9 to 10 percent where
two draws differ by 3, two seeds sitting 0.22 to 0.24 apart under either
tuning. Layer by layer the figure belongs to the width and not to the scheme:
0.40 between the two tunings at the 2 bit layers and 0.22 at the 3 bit ones,
where two seeds sit 0.31 to 0.35 and 0.18 apart. Measured from the checkpoint
with the option on and relative to it, MRound's checkpoint sits 0.27 to 0.28
away and the reference's own default checkpoint 0.30, 1.2 against 1.3 times
the distance between two arms with the option on. That average mixes layers
the two allocators gave different widths; on the 91 to 94 of the 168 layers
they gave one width MRound is the nearer in 88 to 90, at 0.26 against 0.33,
with scales a median of 3.7 to 3.8 percent from that arm's where the default
arm's are 10.2 to 10.5. And a single reference
number
carries a
spread whatever produced it: 2.5 percent between seeds on one setup at mixed
2.5, and seven percent across the reference's own devices, precisions and
releases at one seed (27.36 to 29.24 over rows 41, 45, 84 and 87); at 4 bits on
Qwen2.5-0.5B-Instruct, 1.17 percent across four such arms at one seed
(15.0777 to 15.2537 over rows 30, 85, 95 and 83); at 2 bits, 24 percent
between seeds on one setup (rows 88, 33 and 89) and 25 percent for one change
of release at one seed (rows 37 and 89) on Qwen2.5-0.5B-Instruct, and 10
percent between seeds on Qwen2.5-1.5B-Instruct (rows 92 to 94) and 2 percent
on Qwen2.5-3B-Instruct under either tuning (rows 124 to 129), which is why the
tables below give
three seeds on one named setup where the claim depends on it.

## Uniform 4 bits

| model | original | reference | MRound | MRound against it | size | ledger |
|---|---|---|---|---|---|---|
| Qwen2.5-0.5B-Instruct | 14.3248 | 15.2197 | 15.0260 | -1.27 percent | +2.43 percent | row 95 |
| SmolLM2-135M-Instruct | 17.5077 | 22.1758 | 18.9400 | **-14.59 percent** | +3.00 percent | row 54 |

At 4 bits on the larger model the margin is the size of the reference's own
spread, and its sign depends on what is held equal. MRound is 1.27 percent ahead
at the same group size and 0.50 percent behind at the same bytes (the group size
table below), while the reference's own four productions of this configuration
at this seed, on two CPUs and one GPU under two releases, span 1.17 percent
(15.0777 to 15.2537, rows 30, 85, 95 and 83). At the same group size MRound's
checkpoint is ahead of all four of them, by 0.34 to 1.49 percent; at the same
bytes all four are ahead of MRound's, by 0.28 to 1.45 percent. This row is a
tie, not a win. It faces auto-round 0.15.0 on the Linux machine's CPU and
MRound's current tree, as the 2 bit tables do; until 2026-09-28 it was row 30,
15.0777 against 15.0807, the reference under 0.14.2 on the Mac's CPU against an
older MRound tree, which stays in the ledger as the comparison. On 2026-11-12
MRound's current tree wrote row 95's checkpoint again to the byte, and the same
comparison gave the same three figures (row 233), so the row stands for the tree
as it is. On the smaller model, where 4 bits is a harder ask relative to the
model's capacity, MRound is ahead by 14.59 percent.

**Above half a billion parameters, against round-to-nearest only.** No reference
arm exists at 4 bits above half a billion parameters, so Qwen2.5-1.5B-Instruct,
Qwen2.5-3B-Instruct and Qwen2.5-7B-Instruct were set against round-to-nearest at
the same width, group size and coverage, which rounds the same tensors with
nothing learned, all three arms scored by the evaluator of the tables here (seed
42, the tensors outside the block stack dense, the 7B with every step in four
pieces). Row 95's checkpoint went through the same comparison:

| model | original | round-to-nearest | MRound | MRound against it | its loss taken back | ledger |
|---|---|---|---|---|---|---|
| Qwen2.5-0.5B-Instruct | 14.3248 | 17.0562 | 15.0260 | -11.90 percent | 74.3 percent | rows 230 and 234 |
| Qwen2.5-1.5B-Instruct | 9.6651 | 10.7108 | 10.1055 | -5.65 percent | 57.9 percent | row 235 |
| Qwen2.5-3B-Instruct | 8.5719 | 9.5814 | 8.8948 | -7.17 percent | 68.0 percent | row 236 |
| Qwen2.5-7B-Instruct | 7.3308 | 7.6402 | 7.5829 | -0.75 percent | 18.5 percent | row 232 |

Each pair of checkpoints holds the same bytes. The last column is the share of
round-to-nearest's loss over the original that MRound takes back. MRound's own
loss over the original falls steadily with size, 4.89, 4.56, 3.77 and 3.44
percent; round-to-nearest's does not, 19.07, 10.82, 11.78 and 4.22, and the
share follows it rather than the size, so the small share at 7 billion reads as
a model that leaves 4 bits less to take back. Every MRound checkpoint started
from searched scales, as every MRound arm here does, so each gap holds the scale
search as well as the tuning. This states no bar, which is the reference's own
result, and no arm of the reference exists at these sizes and this width:
whether it takes back more is open. The runs took 4,465, 10,410, 21,643 and
57,008 s on a 32 GB Mac, the 0.5B run 24 percent longer than the same checkpoint
took in September, which says how much the machine itself varies, and peaked at
11.49, 14.25, 20.07 and 28.84 GB (rows 233 to 236 and 232).

**Group size.** Every other row in this document is at group size 64. The
reference's own default is 128, so both sides were run at both on
Qwen2.5-0.5B-Instruct at 4 bits, seed 42, one release and one tree:

| group size | reference | MRound | MRound against it | size | ledger |
|---|---|---|---|---|---|
| 64 | 15.2197 | 15.0260 | -1.27 percent | +2.43 percent | row 95 |
| 128 | 15.5079 | 15.2961 | -1.37 percent | +1.24 percent | row 96 |

Both lose about the same to the coarser grouping, the reference 1.89 percent
and MRound 1.80, so the margin does not depend on the finer group. The size
column is the other half of the reading. MRound's checkpoint at group size 128
holds exactly as many tensor bytes as the reference's at 64, 462,507,776,
because the format MRound writes, MLX's, stores a bias beside each scale where
the reference stores the scale alone; the file on disk is 0.015 percent larger
for its header. At that one size the reference is ahead, 15.2197 against
15.2961, by 0.50 percent. Under a symmetric scheme that bias is always minus
eight times the scale at 4 bits and carries no information, so row 95's
checkpoint with its biases dropped would hold the reference's byte count
exactly and be 1.27 percent ahead of it; no runtime MRound writes for loads a
checkpoint that way, so the 0.50 percent is what that format costs today.

**Zero shot tasks.** The same three checkpoints as row 95, through the twelve
task suite (row 97, scored on the Apple GPU as described above), accuracy in
percent, the last column taken before rounding:

| task | original | reference | MRound | MRound against it |
|---|---|---|---|---|
| mean of the twelve | 48.42 | 47.58 | 47.42 | -0.16 points |
| MMLU | 45.84 | 43.74 | 45.04 | +1.30 |
| ARC Easy | 65.57 | 64.90 | 66.08 | +1.18 |
| ARC Challenge | 30.80 | 29.52 | 30.46 | +0.94 |
| TruthfulQA MC1 | 26.81 | 26.56 | 27.29 | +0.73 |
| HellaSwag | 40.59 | 39.18 | 39.75 | +0.58 |
| Winogrande | 55.80 | 56.20 | 56.43 | +0.24 |
| OpenBookQA | 24.00 | 22.40 | 22.40 | 0.00 |
| LAMBADA | 50.09 | 47.78 | 47.58 | -0.19 |
| PIQA | 70.40 | 70.29 | 69.75 | -0.54 |
| TruthfulQA MC2 | 41.82 | 42.23 | 41.64 | -0.59 |
| BoolQ | 67.89 | 67.92 | 65.96 | -1.96 |
| RTE | 61.37 | 60.29 | 56.68 | -3.61 |

The suite agrees with the perplexity: a tie. MRound is ahead on six tasks and
behind on five, with one even, and the two means are within a sixth of a
point; both checkpoints sit about a point below the original, 0.83 for the
reference and 0.99 for MRound. Measured against MRound's own standard error,
two tasks move by more than two of them, MMLU (+1.30 against 0.41) toward
MRound and BoolQ (-1.96 against 0.83) toward the reference; across twelve tasks
that is not remarkable, and the ledger keeps accuracies rather than per
document outcomes, so no paired test can be taken from it. On `acc_norm`,
over the five tasks that define it, the reference leads by 0.43 points (50.13
against 49.71, the original 50.12). LAMBADA's own perplexity, which that task
reports beside its accuracy, orders the three as wikitext does: 11.73 for the
original, 13.77 for the reference and 13.58 for MRound. Every standard error,
the `acc_norm` figures and the document counts are in the row; TruthfulQA
MC2's accuracy is the probability mass on the true answers, by that task's own
definition.

The group size pair went through the suite as well (row 104, the three
checkpoints of row 96). At group size 128 the reference's mean is 47.30 and
MRound's 46.98, 0.32 points apart, MRound ahead on five tasks and behind on
seven, and `acc_norm` agrees (49.65 for the reference against 48.82). Both lose
to the coarser grouping, the reference 0.28 points and MRound 0.44. At the same
bytes, MRound's checkpoint at 128 against the reference's at 64, the reference
is 0.60 points ahead, 47.58 against 46.98, where the perplexity put it 0.50
percent ahead. So at 4 bits the suite leans the reference's way by a third of a
point or less at either group size, while the perplexity leans MRound's way by
1.27 and 1.37 percent: the same tie, read from the other side. LAMBADA's own
perplexity still favors MRound at 128, 13.42 against 15.07. One task moved a
long way between the two group sizes: MRound's difference on ARC Easy went
from +1.18 points at 64 to -2.99 at 128, so a swing of several points on one
task between two checkpoints is not confined to low widths.

## Uniform 2 bits

**At the reference's learning rate, on one tree.** Below 4 bits MRound follows
the reference's learning rate rule since 0.1.0a6 ([CHANGELOG.md](CHANGELOG.md)):
at the standard 200 steps a 2 bit run starts at `1.0 / iters`, where MRound
started at `2.0 / iters`. Every 2 bit table was measured again at it, four sizes
of Qwen2.5 at three seeds each, each MRound checkpoint scored against the
reference's arms of its own seed under both of the reference's tunings
(auto-round 0.15.0; at 0.5 billion parameters on the Linux machine's CPU in
float32, above it on its integrated GPU under bfloat16 autocast, uncompiled).
MRound leaves the tensors outside the block stack dense, and runs each step of
the 7 billion parameter model in four pieces, which draws another checkpoint, as
a change of seed does:

| model | seed | MRound | the reference's SignRound v2 | MRound against it | the reference's default | MRound against it | ledger |
|---|---|---|---|---|---|---|---|
| Qwen2.5-0.5B-Instruct | 42 | 41.1546 | 47.7076 | -13.74 percent | 57.0002 | -27.80 percent | rows 159, 160 |
| | 43 | 37.3463 | 40.2103 | -7.12 percent | 49.3534 | -24.33 percent | rows 161, 162 |
| | 44 | 39.2488 | 41.6129 | -5.68 percent | 63.1077 | -37.81 percent | rows 163, 164 |
| | **mean** | **39.2499** | 43.1769 | **-9.10 percent** | 56.4871 | **-30.52 percent** | |
| Qwen2.5-1.5B-Instruct | 42 | 20.8777 | 21.9922 | -5.07 percent | 24.1720 | -13.63 percent | rows 165, 166 |
| | 43 | 20.3012 | 21.6616 | -6.28 percent | 23.6660 | -14.22 percent | rows 167, 168 |
| | 44 | 20.0450 | 21.7452 | -7.82 percent | 26.1110 | -23.23 percent | rows 169, 170 |
| | **mean** | **20.4080** | 21.7997 | **-6.38 percent** | 24.6497 | **-17.21 percent** | |
| Qwen2.5-3B-Instruct | 42 | 17.2495 | 17.4454 | -1.12 percent | 20.1332 | -14.32 percent | rows 151, 152 |
| | 43 | 16.2552 | 17.6121 | -7.70 percent | 19.8525 | -18.12 percent | rows 155, 156 |
| | 44 | 16.9670 | 17.2059 | -1.39 percent | 19.6877 | -13.82 percent | rows 157, 158 |
| | **mean** | **16.8239** | 17.4211 | **-3.43 percent** | 19.8912 | **-15.42 percent** | |
| Qwen2.5-7B-Instruct | 42 | 13.5454 | 13.9075 | -2.60 percent | 13.7508 | -1.49 percent | rows 173, 174 |
| | 43 | 12.7227 | 13.0874 | -2.79 percent | 12.9913 | -2.07 percent | rows 153, 154 |
| | 44 | 12.9915 | 13.0846 | -0.71 percent | 13.3012 | -2.33 percent | rows 171, 172 |
| | **mean** | **13.0865** | 13.3598 | **-2.05 percent** | 13.3478 | **-1.96 percent** | |

The originals score 14.3248, 9.6651, 8.5719 and 7.3308, and MRound's checkpoints
are 3.02, 4.91, 6.19 and 5.08 percent larger than the reference's packed at the
same width. The 3 billion parameter arms and the 7 billion one at seed 43 ran
before the change, with the rate given by hand; the tree that made the change
writes at its default the same checkpoint it writes with that rate given by
hand, to the byte (checked at 0.5 billion parameters, seed 42), so they are its
own. Round-to-nearest at 2 bits, the same tensors rounded with nothing learned,
scores 3,306.8675 at 7 billion parameters (row 231, set there against the seed
42 checkpoint of MRound's earlier rate, 14.8169), so at this width the scale
search and the tuning together take back nearly all that rounding loses, as at
the smaller sizes.

MRound is ahead on every seed of every size against both tunings, 24
comparisons. Against the reference's own SignRound v2 tuning the lead shrinks as
the model grows, 9.10, 6.38, 3.43 and 2.05 percent, and only at 1.5 billion are
the ranges apart, MRound's worst draw 3.62 percent ahead of the reference's
best; at 0.5, 3 and 7 billion MRound's worst draw is 2.35, 0.25 and 3.52 percent
behind the reference's best, so those three support the paired statement,
MRound ahead on each of three calibration draws against the reference running
the same method, and not a separation. Against the default tuning the ranges
are apart from 0.5 to 3 billion and overlap at 7 billion, where the reference's
two tunings are level, 13.3598 and 13.3478, and MRound is about 2 percent ahead
of both. The means are ratios of the means; averaging the per seed ratios moves
none by more than 0.54 points. In the checkpoint distance the scoring runs print
(their own output, not a ledger figure), MRound's weights sit 0.297 to 0.312
from the reference's SignRound v2 weights of the same seed at every size, as far
as the reference's own checkpoints sit from each other when only the seed or the
release changes, 0.30 to 0.34.

**Zero shot tasks on these checkpoints**, the three seeds of the 0.5 billion
parameter table, each MRound checkpoint against both of the reference's arms of
its seed (rows 175 to 180, scored on the Apple GPU as described above), accuracy
in percent, the reference and MRound as the mean of the three seeds and each per
seed column the range of the difference across them:

| task | original | MRound | the reference's default | MRound against it | per seed | the reference's SignRound v2 | MRound against it | per seed |
|---|---|---|---|---|---|---|---|---|
| mean of the twelve | 48.42 | 38.67 | 36.97 | +1.70 points | +1.37 to +1.87 | 38.53 | +0.13 points | -0.72 to +1.22 |
| LAMBADA | 50.09 | 23.18 | 15.25 | +7.94 | +7.20 to +9.08 | 23.00 | +0.19 | -1.05 to +0.95 |
| BoolQ | 67.89 | 61.60 | 57.35 | +4.25 | +4.10 to +4.40 | 60.47 | +1.13 | -0.98 to +4.62 |
| ARC Easy | 65.57 | 50.06 | 45.99 | +4.07 | +3.41 to +4.88 | 49.69 | +0.36 | -0.84 to +2.31 |
| OpenBookQA | 24.00 | 17.67 | 15.07 | +2.60 | +1.80 to +3.40 | 17.87 | -0.20 | -1.00 to +0.20 |
| PIQA | 70.40 | 61.68 | 60.08 | +1.60 | +1.36 to +1.80 | 60.74 | +0.94 | +0.44 to +1.31 |
| HellaSwag | 40.59 | 31.35 | 30.27 | +1.07 | +0.72 to +1.45 | 31.09 | +0.26 | +0.21 to +0.29 |
| ARC Challenge | 30.80 | 21.73 | 20.73 | +1.00 | -0.09 to +2.39 | 20.62 | +1.11 | +0.09 to +2.82 |
| MMLU | 45.84 | 23.37 | 22.97 | +0.40 | +0.10 to +0.98 | 23.41 | -0.05 | -1.03 to +0.95 |
| TruthfulQA MC1 | 26.81 | 26.89 | 26.89 | 0.00 | -0.37 to +0.49 | 25.74 | +1.14 | +0.61 to +1.47 |
| TruthfulQA MC2 | 41.82 | 44.03 | 44.34 | -0.31 | -1.76 to +1.73 | 42.97 | +1.06 | -0.01 to +1.82 |
| Winogrande | 55.80 | 50.99 | 52.04 | -1.05 | -1.74 to -0.55 | 52.07 | -1.08 | -2.76 to +0.71 |
| RTE | 61.37 | 51.50 | 52.71 | -1.20 | -3.25 to 0.00 | 54.75 | -3.25 | -7.94 to +0.72 |

Against the default tuning the perplexity margin shows on the tasks: MRound's
mean accuracy is ahead on every seed, by 1.37 to 1.87 points (38.18, 38.96 and
38.87 against 36.81, 37.11 and 37.00), and it leads on eight, eight and nine of
the twelve tasks. Against the reference's own SignRound v2 tuning the suite
calls a tie: 0.13 points on the mean, MRound behind on seeds 42 and 44, by 0.72
and 0.09 points, and ahead on seed 43 by 1.22 (against 38.90, 37.74 and 38.96),
leading on five, eleven and seven tasks; RTE, 277 documents with a standard
error of three points, makes 0.66 of seed 42's 0.72. At every seed MRound leads
both tunings on PIQA, on HellaSwag and on `acc_norm` over its five tasks (39.32
to 39.52 against 37.08 to 37.95 and 38.52 to 39.15), and the SignRound v2 tuning
on ARC Challenge and TruthfulQA MC1 as well; Winogrande favors the reference.
LAMBADA's own perplexity, 113 to 124 for MRound, is better than the default
tuning's at every seed (275 to 352) and than the SignRound v2 tuning's on two
seeds of three (99 to 146; 11.73 for the original). The reference's arms are
those the earlier rate's tables below were scored against, and their figures
came out the same to the last digit. MRound's checkpoints of the earlier rate
scored 39.02 against them, 0.36 points more than these, about half of it on RTE,
where on wikitext2 the two sets are level (0.17 percent apart): at the earlier
rate the suite put MRound 2.05 and 0.49 points ahead of the two tunings, both on
every seed.

At its earlier rate MRound was 9.25, 0.78 and 1.25 percent ahead of the
SignRound v2 tuning on the means and 5.87 behind (rows 105 to 107, 131 to 133,
127 to 129, and 134, 137 and 139), and 30.63, 12.25 and 13.51 ahead of the
default tuning and 5.96 behind (rows 88, 33 and 89, 92 to 94, 124 to 126, and
135, 138 and 140). The change moved the 1.5 billion table by 5.65 percent, the 3
billion by 2.20 and the 7 billion by 7.47, and left the 0.5 billion level, 0.17
percent, against checkpoints from earlier trees as well as an earlier rate. The
tables that follow were tuned at that rate and stay as its record: each stands
as measured, and none is the current default's.

**At MRound's earlier rate.** Three seeds on Qwen2.5-0.5B-Instruct (original
14.3248), changing only the calibration draw, the reference at auto-round 0.15.0
and at its default tuning on every row:

| seed | reference | MRound | MRound against it | ledger |
|---|---|---|---|---|
| 42 | 57.0002 | 41.2332 | -27.66 percent | row 88 |
| 43 | 49.3534 | 36.9670 | -25.10 percent | row 33 |
| 44 | 63.1077 | 39.3524 | -37.64 percent | row 89 |
| **mean** | **56.4871** | **39.1842** | **-30.63 percent** | |

The margin survives the spread: MRound's worst draw beats the reference's best
by 16.45 percent. Size is +3.02 percent throughout. The mean quoted is the ratio
of the means; averaging the three per seed ratios instead gives 30.13 percent,
and the per seed ratios themselves run from 25.10 to 37.64, so the mean is not
quoted without them. The reference's own three draws span 24.35 percent of
their mean here, MRound's 10.89.

This table replaced an earlier one on 2026-09-26. The same three seeds against
auto-round 0.14.2 (rows 31, 34 and 37) gave 58.3402, 49.4642 and 50.3438, a
mean of 52.7161 against the same 39.1842 and a margin of 25.67 percent. Those
rows stay in the ledger as valid measurements against a named release; they are
the release comparison below, not three more samples of this table, and the two
must not be pooled.

**Against the reference's own SignRound v2 tuning.** The same three seeds
again with the reference's `enable_alg_ext` turned on and no other setting
changed (auto-round 0.15.0, the same recipe, the same machine's CPU in float32;
that machine's operating system release did change in between, which moved
neither tuning's checkpoint, as said below), each arm scored against the MRound
checkpoint the table above scored its default arm against, so MRound's column
is the same three checkpoints:

| seed | reference | MRound | MRound against it | the reference's default arm | ledger |
|---|---|---|---|---|---|
| 42 | 47.7076 | 41.2332 | -13.57 percent | 57.0002 | row 105 |
| 43 | 40.2103 | 36.9670 | -8.07 percent | 49.3534 | row 106 |
| 44 | 41.6129 | 39.3524 | -5.43 percent | 63.1077 | row 107 |
| **mean** | **43.1769** | **39.1842** | **-9.25 percent** | 56.4871 | |

MRound is ahead on every seed, and by much less. With the option on the
reference scores 16.30, 18.53 and 34.06 percent below its default arm of the
same seed, 23.56 percent on the mean, which closes 77 percent of the gap in mean
perplexity. Each of those is one draw against another (the change of system
between them moved neither arm at seed 42, below), and at this width a default
arm has moved by 25 percent under a change of release alone (rows 37 and 89), so
each depends on the default draw it is taken against: against the default arms
of the same seeds under 0.14.2 (rows 31, 34 and 37) the same three arms are
18.23, 18.71 and 17.34 percent lower, and the gap closed is 70 percent. Read
together they say that most of the margin in the earlier rate's first table sat
between the reference's two tunings and not between the two implementations. The
two ranges overlap here, as they did in no other three seed perplexity table in
this document when it was measured: MRound's worst draw is 2.54 percent behind
the reference's best. The mean of the per seed ratios is 9.02 percent and they
run from 5.43 to 13.57; a 95 percent Student interval on those three points,
quoted to show how little three points constrain, runs from 19.3 percent ahead
to 1.3 behind. So this table supports a paired statement, that MRound's
checkpoint scored better on each of three calibration draws against the
reference running the same method, and not the stronger one that table supports.
Size is the same +3.02 percent. With the option on the reference's three draws
span 17.36 percent of their mean, against 24.35 for the default arms under this
release and 16.84 for the default arms of the same seeds under 0.14.2, so three
draws do not show the option steadying it.

One thing this table does not settle. The reference's documentation recommends
the option together with its stronger recipe at 2 bits, and no arm on either
side has run that recipe at this width. A second, listed here until 2026-10-09,
is settled. The Linux machine's operating system changed release between the
two sets of reference arms (rows 105 to 107 record Ubuntu 26.04.1, rows 88, 33
and 89 Ubuntu 24.04, the package inventory being the same one), and the machine
was then reinstalled and its environment rebuilt. On the reinstalled system
both of seed 42's arms were produced again (rows 117 and 118), and each is the
checkpoint of row 88 or of row 105 to the bit, with the same perplexity: the
change of system moved neither tuning, so the two reference columns differ in
the option alone, measured at seed 42 and taken at seeds 43 and 44 on the
strength of it. The head to head in each row never depended on that. The same
two arms give the option's price at one seed on one system: the block loop took
20,012 s with the option on against 17,678 s without, 13.2 percent longer, with
9.3 percent more resident memory, and the same pair produced again under
auto-round 0.16.0, which wrote the same two checkpoints (rows 119 and 120),
took 19,840 s against 17,359 s, 14.3 percent longer, with 9.6 percent more.
Read across the change of system the three
seeds had shown 15 to 17 percent, 16.9 at seed 42 (totals of 20,102 to 20,440 s
with the option on against 17,334 to 17,750 s without), and the reinstalled
system timed the default arm 1.8 percent slower than row 88's and the arm with
the option 1.4 percent faster than row 105's, so a time read across two systems
carries a couple of percent either way. Part of that price is the release's and
not the method's: under auto-round 0.15.0 and 0.16.0 alike the option's tuning
runs its block forward uncompiled where the default tuning compiles it.

**Zero shot tasks** on the three pairs of checkpoints of the earlier rate's
first table, the reference at its default tuning (rows 98 to 100, scored on the
Apple GPU as described above), accuracy in percent, the reference and MRound as
the mean of the three seeds and the last column the range of the difference
across them:

| task | original | reference | MRound | MRound against it | per seed |
|---|---|---|---|---|---|
| mean of the twelve | 48.42 | 36.97 | 39.02 | +2.05 points | +1.94 to +2.22 |
| LAMBADA | 50.09 | 15.25 | 24.12 | +8.87 | +7.84 to +9.74 |
| BoolQ | 67.89 | 57.35 | 61.95 | +4.60 | +3.88 to +5.54 |
| ARC Easy | 65.57 | 45.99 | 50.31 | +4.32 | +3.03 to +5.39 |
| ARC Challenge | 30.80 | 20.73 | 22.75 | +2.02 | +1.02 to +2.65 |
| OpenBookQA | 24.00 | 15.07 | 17.00 | +1.93 | +1.00 to +3.00 |
| HellaSwag | 40.59 | 30.27 | 31.44 | +1.17 | +0.67 to +1.66 |
| RTE | 61.37 | 52.71 | 53.55 | +0.84 | 0.00 to +2.17 |
| PIQA | 70.40 | 60.08 | 60.90 | +0.82 | -0.16 to +2.29 |
| TruthfulQA MC2 | 41.82 | 44.34 | 45.14 | +0.80 | -1.50 to +3.45 |
| MMLU | 45.84 | 22.97 | 23.43 | +0.46 | -0.01 to +1.31 |
| Winogrande | 55.80 | 52.04 | 52.09 | +0.05 | -0.79 to +1.42 |
| TruthfulQA MC1 | 26.81 | 26.89 | 25.62 | -1.26 | -2.94 to +0.86 |

That table's perplexity margin shows on the tasks. MRound's mean accuracy is
ahead on every seed, by 1.94 to 2.22 points (39.03, 39.05 and 38.99 against
36.81, 37.11 and 37.00), the two ranges separated by 1.88 points, and it leads
on nine or ten of the twelve tasks at each seed. Much of the suite is at its
floor at this width, though, and the reading has to say which tasks still
discriminate. MMLU sits at 23 percent for both against a quarter for a guess,
ARC Challenge and OpenBookQA are below a quarter, and Winogrande is about two
points above a half. The reference scores exactly RTE's majority class share,
52.71 percent, on all three seeds, which is what answering one way throughout
gives. BoolQ's 62.17 percent is its share of yes answers, which MRound's seeds
sit at (one of them a single document above it) and the reference's fall below,
so neither does better than answering yes by more than one document in 3,270.
TruthfulQA runs against quality altogether here, the reference's MC2 above the
original's. Where both stay clear of their floor, on LAMBADA, ARC Easy,
HellaSwag and PIQA, MRound leads on all four on average and on the first three
at every seed, LAMBADA by 7.8 to 9.7 points. The spread that dominates that
perplexity table is not in the suite: the reference's three perplexities span 24
percent of their mean and its three task means 0.30 points, and its worst seed
by perplexity is in the middle on the tasks. `acc_norm` over its five tasks
agrees (39.20 to 39.46 for MRound against 37.08 to 37.95), and so does LAMBADA's
own perplexity, 78 to 116 for MRound against 275 to 352 (11.73 for the
original).

**Zero shot tasks against the reference's own SignRound v2 tuning**, on the
three pairs of the table with the option on (rows 111 to 113, scored as above),
in the same form:

| task | original | reference | MRound | MRound against it | per seed |
|---|---|---|---|---|---|
| mean of the twelve | 48.42 | 38.53 | 39.02 | +0.49 points | +0.03 to +1.31 |
| TruthfulQA MC2 | 41.82 | 42.97 | 45.14 | +2.17 | +0.25 to +3.53 |
| ARC Challenge | 30.80 | 20.62 | 22.75 | +2.13 | +1.19 to +2.82 |
| BoolQ | 67.89 | 60.47 | 61.95 | +1.48 | -0.12 to +4.10 |
| LAMBADA | 50.09 | 23.00 | 24.12 | +1.12 | +0.89 to +1.49 |
| ARC Easy | 65.57 | 49.69 | 50.31 | +0.62 | -0.76 to +2.82 |
| HellaSwag | 40.59 | 31.09 | 31.44 | +0.36 | +0.24 to +0.48 |
| PIQA | 70.40 | 60.74 | 60.90 | +0.16 | -0.71 to +1.36 |
| Winogrande | 55.80 | 52.07 | 52.09 | +0.03 | -2.37 to +1.66 |
| MMLU | 45.84 | 23.41 | 23.43 | +0.01 | -1.15 to +1.15 |
| TruthfulQA MC1 | 26.81 | 25.74 | 25.62 | -0.12 | -1.47 to +1.84 |
| OpenBookQA | 24.00 | 17.87 | 17.00 | -0.87 | -1.80 to -0.20 |
| RTE | 61.37 | 54.75 | 53.55 | -1.20 | -2.53 to +1.44 |

MRound's column is the same three checkpoints as in the table above, and its
figures came out identical to the last digit, as the original's did. Against
the reference's own tuning the suite's lead shrinks the way the gap in
perplexity did: 0.49 points of mean accuracy where the default arms gave 2.05,
so the option closed 76 percent of it, against 77 percent of the gap in mean
perplexity. MRound is still ahead on every seed, but by 0.14, 1.31 and 0.03
points (39.03, 39.05 and 38.99 against 38.90, 37.74 and 38.96), so on two seeds
of three by less than a sixth of a point, and it leads on eight, nine and five
of the twelve tasks. What still favors MRound at every seed is the part of the
suite nearest to perplexity: LAMBADA, by 0.9 to 1.5 points, and its own
perplexity, 78 to 116 for MRound against 99 to 146; HellaSwag, by 0.2 to 0.5;
ARC Challenge, by 1.2 to 2.8; and `acc_norm` over its five tasks (39.20 to
39.46 against 38.52 to 39.15). OpenBookQA favors the reference at every seed.
The perplexity spread is not on the tasks here either: the reference's best
seed by perplexity, 43, is its worst on the suite.

**Qwen2.5-1.5B-Instruct**, original 9.6651, three seeds, the reference at
auto-round 0.15.0 on the Linux machine's integrated GPU at its default
precision there (bfloat16 under autocast, uncompiled because that device's
compiler refuses the bfloat16 kernels; on that machine's CPU the table would
have taken three days), MRound on the current tree:

| seed | reference | MRound | MRound against it | ledger |
|---|---|---|---|---|
| 42 | 24.1720 | 20.8947 | -13.56 percent | row 92 |
| 43 | 23.6660 | 21.4645 | -9.30 percent | row 93 |
| 44 | 26.1110 | 22.5277 | -13.72 percent | row 94 |
| **mean** | **24.6497** | **21.6290** | **-12.25 percent** | |

MRound's worst draw beats the reference's best by 4.81 percent. Size is +4.91
percent. The mean of the per seed ratios is 12.19 percent. This table replaced
a single seed on 2026-09-27, row 36: 23.9091 against 21.4767, a margin of
10.17 percent, with the reference on the Linux CPU under 0.14.2 and MRound on
an older tree. The reference moved 1.10 percent between the two seed 42 arms
and MRound 2.71 percent between its two trees; row 36 stays in the ledger as
the comparison and is not pooled with this table.

**Qwen2.5-3B-Instruct**, original 8.5719, three seeds, the reference as in the
table above (auto-round 0.15.0 on the integrated GPU, bfloat16 under
autocast, uncompiled), MRound on one tree:

| seed | reference | MRound | MRound against it | ledger |
|---|---|---|---|---|
| 42 | 20.1332 | 17.9583 | -10.80 percent | row 124 |
| 43 | 19.8525 | 16.8479 | -15.13 percent | row 125 |
| 44 | 19.6877 | 16.8030 | -14.65 percent | row 126 |
| **mean** | **19.8912** | **17.2031** | **-13.51 percent** | |

MRound's worst draw beats the reference's best by 8.78 percent. Size is +6.19
percent. The mean of the per seed ratios is 13.53 percent; a 95 percent
Student interval on the three runs from 7.6 to 19.4 percent ahead, where the
same interval on the 1.5 billion table runs from 6.0 to 18.4. The reference's
three draws span 2.24 percent of their mean, against 9.92 at 1.5 billion and
24.35 at 0.5 billion, and MRound's 6.72, so this is the first table here in
which the reference's draws sit closer together than MRound's. The runs' own
logs, which are not ledger rows, show both sides finding the same thing at the
model's second block: rounded to the nearest code it starts three to four
orders of magnitude above the blocks beside it, on each side's own loss, and
tuning brings it down to their level, so 2 bit round-to-nearest fails there
for the model and not for one implementation.

**Against the reference's own SignRound v2 tuning, at 3 billion.** The same
three seeds with the reference's `enable_alg_ext` turned on and no other
setting changed, each pair of arms produced on one system in one environment,
and each arm scored against the MRound checkpoint its default twin was scored
against, so MRound's column is the same three checkpoints:

| seed | reference | MRound | MRound against it | the reference's default arm | ledger |
|---|---|---|---|---|---|
| 42 | 17.4454 | 17.9583 | +2.94 percent | 20.1332 | row 127 |
| 43 | 17.6121 | 16.8479 | -4.34 percent | 19.8525 | row 128 |
| 44 | 17.2059 | 16.8030 | -2.34 percent | 19.6877 | row 129 |
| **mean** | **17.4211** | **17.2031** | **-1.25 percent** | 19.8912 | |

Here the two implementations are level. MRound is ahead on two seeds and behind
on the third, 1.25 percent ahead on the mean (the mean of the per seed ratios
is the same), and the two ranges overlap almost entirely: MRound's worst draw
is 4.37 percent behind the reference's best and its best 4.59 percent ahead of
the reference's worst, and a 95 percent Student interval on the three runs from
10.6 percent ahead to 8.1 behind. This row is a tie, read as the 4 bit
Qwen2.5-0.5B-Instruct row is. With the option on the reference scores 13.35,
11.29 and 12.61 percent below its default arm of the same seed, 12.42 on the
mean, which closes 92 percent of the gap in mean perplexity, where on
Qwen2.5-0.5B-Instruct at 2 bits it closed 77: at this size nearly all of the
default table's margin sat between the reference's two tunings. The seed on
which MRound is behind is its own worst draw, 6.7 percent above its other two,
which agree to 0.27 percent, while the reference's three arms with the option
on span 2.33 percent of their mean. Size is +6.19 percent. Both tunings ran
uncompiled on that device, so the pair gives the option's cost without a
compiler's share in it: the block loop took 22,936 to 22,945 s with the option
on against 21,754 to 21,767 s without, 5.4 to 5.5 percent longer, and the
device pool peaked at 14.08 GB against 15.90.

One more model at the same scheme, one seed:

| model | original | reference | MRound | MRound against it | size | ledger |
|---|---|---|---|---|---|---|
| SmolLM2-135M-Instruct | 17.5077 | 86.9820 | 58.4582 | -32.79 percent | +3.92 percent | row 35 |

Note the absolute numbers. At 2 bits both implementations produce models far
worse than the original, and the interesting claim is the gap between them, not
that either result is good. At MRound's earlier rate, against the reference's
default tuning the margin narrowed as the model grew to 1.5 billion parameters,
32.79 percent at 135 million, 30.63 at 0.5 billion and 12.25 at 1.5 billion, and
then stopped narrowing: it is 13.51 percent at 3 billion, inside the range of
the 1.5 billion table's per seed margins, the last three sizes on three seeds
each. MRound's penalty over the original is 51 to 65 percent of the reference's
at 0.5 billion, 77 to 84 percent at 1.5 billion and 73 to 81 percent at 3
billion. Against the reference's own SignRound v2 tuning, the method MRound
implements, the margin was 9.25 percent at 0.5 billion and 1.25 at 3 billion,
where the two were level, and MRound's penalty is 81 to 92 percent of the
reference's at the smaller size and 92 to 106 at the larger: against that tuning
the lead shrank as the model grew, to nothing at 3 billion, on the two sizes
these tables measure. Most of each margin against the default tuning is the
difference between the reference's two tunings, 77 percent of the gap in mean
perplexity at 0.5 billion and 92 at 3 billion, and the curve against the default
says little about the two implementations. At the current rate (the table at the
head of this section) the margins against the default tuning are 30.52, 17.21,
15.42 and 1.96 percent from 0.5 to 7 billion, and against the SignRound v2
tuning 9.10, 6.38, 3.43 and 2.05: the lead still shrinks with size against the
method MRound implements, and holds on every seed to 7 billion. The caveat on
every number in this document stands either way: they come from models of seven
billion parameters or fewer, the largest at 2 bits only.

## Uniform 8 and 3 bits, against round-to-nearest only

No reference arm exists at either width, so Qwen2.5-0.5B-Instruct at 8 and 3
bits was set against round-to-nearest at the same width, group size 64 and
coverage, as the larger models at 4 bits are above, every arm scored by the
evaluator of the tables here (seed 42, the tensors outside the block stack
dense). Its 4 bit checkpoint went through the same comparison:

| width | original | round-to-nearest | MRound | MRound against it | its loss taken back | ledger |
|---|---|---|---|---|---|---|
| 8 bits | 14.3248 | 14.3224 | 14.3246 | +0.02 percent | none to take back | row 237 |
| 4 bits | 14.3248 | 17.0562 | 15.0260 | -11.90 percent | 74.3 percent | row 234 |
| 3 bits | 14.3248 | 41.5136 | 18.1285 | -56.33 percent | 86.0 percent | row 238 |

Each pair of checkpoints holds the same bytes. At 8 bits there is nothing to
take back: round-to-nearest lands 0.02 percent below the original and MRound
0.001 percent below it, and the 0.0022 between them says nothing about either;
learned rounding moved the weights little and changed nothing measurable, after
an hour's work (4,554 s) where round-to-nearest took 8 s. At 3 bits
round-to-nearest loses 190 percent of the original's perplexity and MRound 26.6,
taking back 86.0 percent of the loss, more than at 4 bits: the narrower the
width, the more there is to take back. Both MRound checkpoints started from
searched scales, so each gap holds the scale search as well as the tuning. This
states no bar, which is the reference's own result, and no arm of the reference
exists at these widths. The zero shot task suite on both rows is next.

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

**Against the reference's own SignRound v2 tuning.** The three
Qwen2.5-0.5B-Instruct seeds again with the reference's `enable_alg_ext` turned
on and no other setting changed (auto-round 0.15.0, the same recipe and
scoring budget, the same machine's CPU in float32; that machine's operating
system release did change in between, as at 2 bits, where that moved neither
tuning's checkpoint), each arm scored against
the MRound checkpoint the table above scored its default arm against, so
MRound's column is the same three checkpoints:

| seed | reference | MRound | MRound against it | the reference's default arm | ledger |
|---|---|---|---|---|---|
| 42 | 26.3575 | 22.9964 | -12.75 percent | 28.4685 | row 108 |
| 43 | 25.1924 | 21.6389 | -14.11 percent | 28.3806 | row 109 |
| 44 | 25.4831 | 22.4770 | -11.80 percent | 27.7606 | row 110 |
| **mean** | **25.6777** | **22.3707** | **-12.88 percent** | 28.2033 | |

MRound is ahead on every seed, and here the two ranges stay apart: its worst
draw beats the reference's best by 8.72 percent, 2.20 points of perplexity
where the default arms left 4.76. The mean of the per seed ratios is the same
12.88 percent and they run from 11.80 to 14.11; a 95 percent Student interval
on those three points, quoted as at 2 bits to show how little three points
constrain, runs from 15.8 percent ahead to 10.0 ahead. So this table supports
the statement the first one supports, on a smaller margin. With the option on
the reference scores 7.42, 11.23 and 8.20 percent below its default arm of the
same seed, 8.95 percent on the mean, which closes 43 percent of the gap in
mean perplexity (38.6 to 47.3 percent seed by seed) where at 2 bits it closed
77: at this scheme less than half of the margin sat between the reference's
two tunings. Each of those is one draw against another, and at this scheme a
default arm
has moved by 2.8 to 3.6 percent under a change of device or of release alone
(rows 84 and 87 against 45, rows 45 and 46 against 41 and 42); the three arms
with the option on, 25.19 to 26.36, sit a point below every default arm the
ledger holds at this recipe, 27.36 to 29.24. The reference's allocation is no
part of the difference: all six of its checkpoints, with the option and
without, give every layer the same width, 58 at 2 bits and 110 at 3. Size is
the same +2.85 percent. With the option on the reference's three draws span
4.54 percent of their mean against 2.51 for the default arms, so here too
three draws do not show the option steadying it.

What this table does not settle is what the 2 bit one does not. The option
together with the reference's stronger recipe has not been run on either side:
at seed 42 the recipe alone took the reference to 25.8199 (The reference at its
stronger recipe, below) and the option alone to 26.3575. These arms too were
produced after the Linux machine's operating system changed release (rows 108
to 110 record Ubuntu 26.04.1, rows 43, 45 and 46 Ubuntu 24.04, the package
inventory being the same one). The 2 bit control, both tunings of one seed
produced again on one system and each the same checkpoint to the bit, is taken
to hold here as well, the block loop being the same code and the allocation the
same in all six exports, so the two reference columns are read as differing in
the option alone, which no mixed arm produced twice has shown; the head to head
in each row does not depend on it. The reference's times are stated for the
record: 20,123 to 20,125 s with the option on against 17,417 to 22,004 s
without, the longest of those holding a 70 minute scheme search that the others
read from cache, and block loops 13.0 to 15.6 percent longer with the option
on, read across the change of system, where at 2 bits that reading gave 15 to
17 percent and one system 13.2. SmolLM2-135M-Instruct's table has no such arms.

**Qwen2.5-3B-Instruct, against both of the reference's tunings**, original
8.5719. The reference's six arms ran on the Linux machine's integrated GPU in
bfloat16 autocast under auto-round 0.15.0, three with `enable_alg_ext` and three
without, each with the reference's own allocator, AutoScheme, which gave all six
one allocation, layer for layer: 135 layers at 2 bits, 31 at 3 and 86 at 4.
MRound with its own allocation as it planned it until the change below, with the
widest width's gradients (rows 181 to 186):

| seed | reference, default | reference, SignRound v2 | MRound | against the default | against SignRound v2 | ledger |
|---|---|---|---|---|---|---|
| 42 | 12.1190 | 11.4254 | 12.1590 | +0.33 percent | +6.42 percent | rows 182, 181 |
| 43 | 11.8836 | 11.3671 | 11.2585 | -5.26 percent | -0.96 percent | rows 184, 183 |
| 44 | 11.9185 | 11.4064 | 11.9642 | +0.38 percent | +4.89 percent | rows 186, 185 |
| **mean** | **11.9737** | **11.3996** | **11.7939** | **-1.50 percent** | **+3.46 percent** | |

This was the one table where MRound missed the bar: 3.46 percent behind the
reference's own SignRound v2 tuning on the mean and on two seeds of three, and
level with its default, ahead on the mean and behind on two seeds of three,
where at uniform 2 bits on the same model and seeds it is 3.43 percent ahead of
that tuning on every seed. The gap was MRound's allocation, not its tuning.
Tuned at the reference's own allocation, read back out of each arm's export,
which MRound's checkpoints then hold in all 252 layers, and with everything else
as above (rows 195 to 200):

| seed | reference, default | reference, SignRound v2 | MRound at the reference's allocation | against the default | against SignRound v2 | ledger |
|---|---|---|---|---|---|---|
| 42 | 12.1190 | 11.4254 | 11.4829 | -5.25 percent | +0.50 percent | rows 196, 195 |
| 43 | 11.8836 | 11.3671 | 11.2437 | -5.39 percent | -1.09 percent | rows 198, 197 |
| 44 | 11.9185 | 11.4064 | 11.2144 | -5.91 percent | -1.68 percent | rows 200, 199 |
| **mean** | **11.9737** | **11.3996** | **11.3137** | **-5.51 percent** | **-0.75 percent** | |

MRound is then ahead of the SignRound v2 tuning on the mean and on two seeds of
three, the ranges overlapping, and ahead of the default on every seed with the
ranges apart, its worst draw 3.37 percent below the default's best. The
reference's allocation is worth 4.07 percent to MRound on the mean, and it
steadies it: the three draws span 2.37 percent of their mean, where those of
MRound's own allocation span 7.64. The two allocations spend one budget at
opposite ends of the stack. The second block holds 3.41 bits a weight in the
reference's and 2.14 to 2.19 in MRound's, blocks 8 to 22 hold 2.12 against 2.67
to 2.79, and blocks 23 to 35 hold 2.94 against 2.31 to 2.43. At the second
block, where the reference keeps two of the three MLP projections at 4 bits and
MRound gives them 2, MRound's error on the original model's input is six times
the reference's at its own allocation and level with it at the reference's (a
block probe on four windows at seed 42, which is not a ledger row). One other
departure of MRound's was tested and ruled out: tuned with the outlier
suppressed loss in every block that holds a layer below 4 bits, as the
reference's SignRound v2 is, MRound scored 0.43 to 2.23 percent worse on every
seed (rows 187 to 192). Size is +5.51 percent against the reference packed at
its widths, at either allocation. What MRound's allocator got wrong at this size
was the gradient its scores were taken from.

**MRound's own allocation, scored as the reference scores: MRound as it ships.**
Until this table MRound's allocator scored each layer with a gradient taken once
per batch at the widest width and reused for the narrower ones, a divergence
from the reference chosen on one small model at a quicker recipe
(DOCUMENTATION.md, 1.5 What v2 adds); the reference takes the gradient at each
width's own quantized point, `quantize`'s `sensitivity_gradients="own_width"`,
which MRound's default now takes at this size (below). With that one change from
the first table (rows 201 to 206):

| seed | reference, default | reference, SignRound v2 | MRound, `own_width` | against the default | against SignRound v2 | ledger |
|---|---|---|---|---|---|---|
| 42 | 12.1190 | 11.4254 | 11.4003 | -5.93 percent | -0.22 percent | rows 202, 201 |
| 43 | 11.8836 | 11.3671 | 11.0107 | -7.35 percent | -3.14 percent | rows 204, 203 |
| 44 | 11.9185 | 11.4064 | 11.0985 | -6.88 percent | -2.70 percent | rows 206, 205 |
| **mean** | **11.9737** | **11.3996** | **11.1699** | **-6.71 percent** | **-2.02 percent** | |

MRound is then ahead of both tunings on every seed, its worst draw 0.29 percent
above the SignRound v2 tuning's best and 4.07 percent below the default's best,
and it is 1.27 percent better on the mean than tuned at the reference's own
allocation, better on every seed. Its plan is one plan at all three seeds, 128
layers at 2 bits, 52 at 3 and 72 at 4, agreeing with AutoScheme's on 226 of 252
layers, where the default gradients gave each seed a plan of its own, agreeing
on 135 to 146. It spends the budget where AutoScheme does, 3.01 bits a weight at
the second block, 2.18 in blocks 8 to 22 and 2.89 in blocks 23 to 35, and at the
second block its error on the original model's input is level with the
reference's (the block probe at seed 42). The three draws span 3.49 percent of
their mean. Size is +5.51 percent, as at either other allocation. This table is
what MRound produces at 3B as it ships, and the first is what it produced before
the change, which the next paragraph says is made by size.

**Which gradients MRound scores with, by size.** The widest width's gradients
were chosen on a 135 million parameter model, at a quicker recipe and one seed,
so before the default changed both sources were run on the two smaller models
above, three seeds each on one tree, every checkpoint scored against the
reference arms of those tables (rows 207 to 224):

| model | `own_width` | widest | `own_width` against widest | per seed | ledger |
|---|---|---|---|---|---|
| SmolLM2-135M-Instruct | 35.8926 | 30.3866 | +18.12 percent | +16.50 to +20.91 | rows 207 to 212 |
| Qwen2.5-0.5B-Instruct | 25.0236 | 22.5618 | +10.91 percent | +7.71 to +12.71 | rows 213 to 224 |

There the widest gradients are better on every seed, where on
Qwen2.5-3B-Instruct `own_width` is 5.29 percent better on the mean and on every
seed (11.1699 against 11.7939). So MRound's default takes them by size: a
gradient at each width's own quantized point from 2.5 billion weights in the
scored layers up, which Qwen2.5-3B-Instruct's 2.77 billion are, and at the
widest width below, which keeps both tables above as MRound produces them
(DOCUMENTATION.md, 1.5). The rule is measured at its two ends and not between
them: no model between the 0.36 billion scored weights of Qwen2.5-0.5B-Instruct
and those 2.77 billion has been run both ways, and Qwen2.5-7B-Instruct, which
the rule gives its own width's gradients, has no mixed table. With either source
MRound leads the reference's default arms on every seed of both models; on
Qwen2.5-0.5B-Instruct `own_width` leads the arms with the option on by 2.55
percent on the mean, behind on seed 44 by 0.08, where the widest gradients lead
them by 12.13 on every seed.

The widest runs repeat the published tables' configuration on the current tree.
On SmolLM2-135M-Instruct they reproduce rows 49 to 51 exactly. On
Qwen2.5-0.5B-Instruct they score 23.4063, 21.6517 and 22.6273, which is 1.78,
0.06 and 0.67 percent above rows 43, 45 and 46 and 0.85 on the mean, with
allocations of the same counts at every seed; the cause has not been isolated.
Against those runs MRound leads the reference's default arms by 20.00 percent on
the mean where the table above reads 20.68, and the arms with the option on by
12.13 where it reads 12.88, its worst draw still ahead of the reference's best
in both.

**Zero shot tasks** on the first Qwen2.5-0.5B-Instruct table's three pairs of
checkpoints, the reference at its default tuning (rows 101 to 103, scored on
the Apple GPU as described above), in the form of the 2 bit one:

| task | original | reference | MRound | MRound against it | per seed |
|---|---|---|---|---|---|
| mean of the twelve | 48.42 | 41.83 | 42.52 | +0.69 points | -0.46 to +1.82 |
| LAMBADA | 50.09 | 33.12 | 38.40 | +5.28 | +2.64 to +7.72 |
| BoolQ | 67.89 | 59.22 | 62.61 | +3.39 | -1.53 to +10.95 |
| ARC Easy | 65.57 | 55.68 | 57.93 | +2.24 | +0.04 to +3.91 |
| Winogrande | 55.80 | 53.75 | 55.22 | +1.47 | +1.10 to +2.05 |
| MMLU | 45.84 | 28.48 | 29.72 | +1.24 | -8.17 to +7.96 |
| PIQA | 70.40 | 64.51 | 65.54 | +1.03 | +0.76 to +1.31 |
| HellaSwag | 40.59 | 34.14 | 34.55 | +0.42 | +0.26 to +0.52 |
| TruthfulQA MC2 | 41.82 | 42.98 | 42.53 | -0.45 | -0.94 to 0.00 |
| TruthfulQA MC1 | 26.81 | 27.09 | 26.44 | -0.65 | -1.22 to +0.24 |
| ARC Challenge | 30.80 | 25.51 | 24.77 | -0.74 | -0.85 to -0.60 |
| OpenBookQA | 24.00 | 20.87 | 19.53 | -1.33 | -2.40 to +0.60 |
| RTE | 61.37 | 56.56 | 52.95 | -3.61 | -9.39 to +0.36 |

At mixed 2.5 the suite is equivocal. The perplexity margin of 20.68 percent,
with the two sides' ranges 4.76 points of perplexity apart, becomes 0.69
points of mean accuracy, and not on every seed: the means are 41.46, 41.20 and
42.81 for the reference against 42.18, 43.02 and 42.35 for MRound, so the
ranges overlap and MRound is behind on seed 44; it leads on six, eleven and six
of the twelve tasks. What decides the sign is three tasks whose spread
across seeds within one implementation is larger than any difference between
the two: the reference's MMLU runs from 24.15 to 34.17 percent, its BoolQ from
51.68 to 63.15 and its RTE from 52.71 to 62.45, and MRound's MMLU from 26.00
to 32.11, where every other task spreads by 3.2 points or less on either side.
On those three a score says more about which checkpoint a seed happened to
produce than about which implementation produced it, and they turn the 2 bit
table's pattern around: there the reference's perplexities spanned 24 percent
of their mean and its task means 0.30 points, here 2.5 percent and 1.61
points, 1.09 of them from those three tasks. The tasks steady enough to read
favor MRound at every seed on LAMBADA (by 2.6 to 7.7 points), Winogrande,
PIQA, HellaSwag and ARC Easy (the last by 0.04 on one seed), and the reference
at every seed on ARC Challenge (by 0.6 to 0.9). The mean of the nine without
the three swinging tasks puts MRound ahead at every seed, by 0.85, 1.42 and
0.16 points, but that rule was chosen after seeing the scores, so the twelve
task mean stays the headline. `acc_norm` over its five tasks favors MRound at
every seed (44.44, 44.59 and 44.52 against 42.81, 43.00 and 43.74; 50.12 for
the original), and so does LAMBADA's own perplexity, 25 to 28 for MRound
against 37 to 46 (11.73 for the original). SmolLM2-135M-Instruct's table has
not been through the suite.

**Zero shot tasks against the reference's own SignRound v2 tuning**, on the
three pairs of the table with the option on (rows 114 to 116), in the same
form:

| task | original | reference | MRound | MRound against it | per seed |
|---|---|---|---|---|---|
| mean of the twelve | 48.42 | 42.42 | 42.52 | +0.10 points | -0.34 to +0.37 |
| LAMBADA | 50.09 | 35.99 | 38.40 | +2.41 | +1.79 to +2.76 |
| ARC Easy | 65.57 | 56.59 | 57.93 | +1.33 | -0.42 to +2.65 |
| Winogrande | 55.80 | 54.72 | 55.22 | +0.50 | -1.03 to +1.42 |
| RTE | 61.37 | 52.47 | 52.95 | +0.48 | +0.36 to +0.72 |
| BoolQ | 67.89 | 62.32 | 62.61 | +0.29 | -0.52 to +1.10 |
| HellaSwag | 40.59 | 34.30 | 34.55 | +0.25 | -0.01 to +0.74 |
| PIQA | 70.40 | 65.40 | 65.54 | +0.15 | -0.11 to +0.54 |
| TruthfulQA MC2 | 41.82 | 42.44 | 42.53 | +0.09 | -0.99 to +1.12 |
| ARC Challenge | 30.80 | 24.86 | 24.77 | -0.09 | -0.09 to -0.09 |
| TruthfulQA MC1 | 26.81 | 27.25 | 26.44 | -0.82 | -1.47 to +0.12 |
| OpenBookQA | 24.00 | 20.53 | 19.53 | -1.00 | -1.60 to -0.20 |
| MMLU | 45.84 | 32.10 | 29.72 | -2.38 | -5.82 to -0.43 |

Here the suite's lead is 0.10 points where the default arms gave 0.69, ahead on
two seeds and behind on seed 44 by 0.34 (42.18, 43.02 and 42.35 against 41.90,
42.65 and 42.69), the ranges overlapping; MRound leads on nine, seven and two
of the twelve tasks. The option closed 85 percent of the suite's lead at this
scheme, where it closed 43 percent of the gap in mean perplexity. It also
steadies the reference on the three tasks that decided the sign against the
default arms: with it on, the reference's MMLU runs from 31.82 to 32.55
percent, its BoolQ from 62.14 to 62.48, about its share of yes answers, and its
RTE from 51.99 to 52.71, at or two documents below its majority class share.
The swing left is MRound's own MMLU, 26.00 to 32.11, which on seed 42 alone
moves the difference between the two means by 0.49 points against it. LAMBADA
favors MRound at every seed, by 1.8 to 2.8 points, and so do its perplexity,
24.9 to 27.8 against 30.7 to 36.4, and `acc_norm` (44.44 to 44.59 against 43.49
to 43.91); OpenBookQA and MMLU favor the reference at every seed, and so does
ARC Challenge, by one question in 1,172 on each. So against its own tuning at
mixed 2.5 the reference matches MRound on the twelve task mean, while MRound
keeps the measures nearest to perplexity.

One structural difference was reported here as one between the two allocators,
and it is one between their gradients. Given the same three candidate widths and
the same budget, the reference never allocated a single layer at the widest
option on either smaller model, spending the budget as a near binary choice
between 2 and 3 bits, while MRound with the widest gradients used all three.
Scored with the reference's gradients, MRound's allocator leaves the widest
option almost unused too, one layer at 4 bits at most, and agrees with the
reference's allocation on 183 to 187 of SmolLM2-135M-Instruct's 210 layers and
153 to 155 of Qwen2.5-0.5B-Instruct's 168, where with the widest gradients it
agrees on 160 to 164 and 91 to 94 (rows 207 to 224). At Qwen2.5-3B-Instruct
AutoScheme used all three widths, 86 layers at 4 bits, and MRound's allocator
given the reference's gradients agrees with it on 226 of 252 layers. With the
same gradients the two allocators nearly agree at every size measured; the plan
nearest the reference's is the better one at three billion parameters and the
worse one below.

## The reference at its stronger recipe

Every comparison above runs both sides at the reference's default recipe, 200
optimization steps over 128 calibration samples. The reference also ships a
stronger one, 1,000 steps over 512 samples. Both sides at both recipes, seed 42,
mixed 2.5, Qwen2.5-0.5B-Instruct:

| recipe | reference | MRound | MRound against it | ledger |
|---|---|---|---|---|
| 200 steps, 128 samples | 28.4685 | 22.9964 | -19.22 percent | row 43 |
| 1,000 steps, 512 samples | 25.8199 | 22.3231 | -13.54 percent | row 48 |

The longer schedule helps the reference about three times as much as it helps
MRound in relative terms, 9.30 percent against 2.93, and about four times as
much in absolute perplexity, 2.65 points against 0.67, so the margin narrows
from 19.22 to 13.54 percent. It does not close. MRound at its default still
beats the reference at that recipe, by 10.94 percent. Neither run turned on the
reference's `enable_alg_ext` (see The bar), which its documentation pairs with
this recipe for 2 bit weights, so this is the reference's stronger recipe and
not yet its best configuration. Each of the two has now been run alone at this
seed: the recipe took the reference from 28.4685 to 25.8199, 9.30 percent, and
the option at the default recipe to 26.3575, 7.42 percent (row 108, under Mixed
precision). The two together have not been run.

## Which release of the reference

The reference's own version matters most at low widths, and at 4 bits it moves
the result by about as much as the 4 bit margin itself, which is worth knowing
before reading any of the above.

Every three seed table here faces auto-round 0.15.0, one release throughout:
the two mixed 2.5 tables since they were produced, the 2 bit table since
2026-09-26, when its seed 42 and seed 44 arms were reproduced under 0.15.0
(rows 88 and 89; seed 43 had been, row 33), and the two Qwen2.5-0.5B-Instruct
tables against the reference's own SignRound v2 tuning since they were produced
(rows 105 to 107 and 108 to 110).
The 4 bit SmolLM2 comparison was
rerun on 0.15.0 on the Linux machine's CPU, and the reference moved 0.12
percent from its August result under 0.14.2 on the Mac's CPU (rows 9 and 54,
both arms scored from the reference's dequantized export). The
Qwen2.5-1.5B-Instruct table faces 0.15.0 since 2026-09-27 (rows 92 to 94,
replacing row 36), the two Qwen2.5-3B-Instruct tables, one under each of the
reference's tunings, since they were produced (rows 124 to 129), and the 4 bit
Qwen2.5-0.5B row since 2026-09-28 (row 95,
replacing row 30), where the change of release and machine together moved the
reference by 0.94 percent. The single seed rows that still face 0.14.2 are the
2 bit SmolLM2 row (35) and the Apple GPU arms (82, 83, 85 and 86), 0.14.2
being the release installed on the Mac.

**A later release, auto-round 0.16.0, writes the same checkpoints.** It was
published while the tables above were being completed. Seed 42's two 2 bit
reference arms, one under each of the reference's tunings, were produced again
under 0.16.0 on the machine and system of the controls of rows 117 and 118, in
an environment that differs from theirs in the release and in the version of
setuptools, and both came out the 0.15.0 checkpoints to the bit, all
357,826,560 weights, with the perplexities of rows 88 and 105 (rows 119 and
120, 2026-10-09). Read side by side, the two releases differ on the path these
comparisons take in nothing that computes a number at these settings: the scale
and rounding code, the optimizer and the calibration path are the same files.
So the tables here are tables of 0.16.0 as well at the point measured, and are
expected to be at the others, where no 0.16.0 arm has been run.

**Its patch release, auto-round 0.16.1, writes the same checkpoints as well.**
It was published on 2026-10-15 and read file by file against 0.16.0. On the path
these comparisons take, its one new option, a grid search for round-to-nearest,
is off by default and gated wherever it reaches; its other fixes concern the
GGUF export and mixture of experts models; and the rest is a pass of a linter
whose rewrites compute what the code computed before. Run side by side with
0.16.0 on SmolLM2-135M-Instruct at toy budgets, under both of the reference's
tunings, in bfloat16 and in float32, and at mixed 2.5, the two releases wrote
the same checkpoints, byte for byte, and the same scores in the reference's
scheme search. At the published recipe, row 88's and row 105's reference arms
produced again under 0.16.1 on the same machine came out the 0.16.0 checkpoints
to the bit, all 357,826,560 weights; scored against the MRound checkpoint of
rows 159 and 160, they repeat those rows to the last digit (rows 193 and 194,
2026-11-05), so the tables here are tables of 0.16.1 too at the point measured.

**How much the release is worth depends on the width, and at 2 bits it is not
small.** At mixed 2.5 the release has been measured as worth about three
percent in either direction, on two seeds. At 2 bits the three seeds now
measured under both releases moved by 2.30 percent one way (58.3402 to
57.0002), 0.22 percent (49.4642 to 49.3534) and 25.35 percent the other way
(50.3438 to 63.1077, seed 44, rows 37 and 89). Until 2026-09-26 this document
quoted the 0.22 as what the release is worth at 2 bits; it was one draw. The
reading that fits all of it is the one the checkpoint comparisons give above:
a change of release, like a change of seed or of device, produces a different
reference checkpoint, and at 2 bits the perplexity of a different checkpoint
can sit a quarter away. At its default tuning the reference's own three seeds
under one release span 24.35 percent of their mean at 2 bits on
Qwen2.5-0.5B-Instruct (9.92 at 1.5 billion and 2.24 at 3 billion), against 2.5
percent at mixed 2.5, which is why this document quotes the per seed ratios and
their range beside the ratio of the means at 2 bits.

## What these numbers do not show

They are five models, of seven billion parameters at most, on one evaluation
corpus, scored by perplexity, and seven tables of them by a zero shot task suite
as well (the 4 bit one, the mixed 2.5 ones against each of the reference's two
tunings, and the 2 bit ones against each, on MRound's current checkpoints and on
those of its earlier rate, all on Qwen2.5-0.5B-Instruct); the other tables have
not been through the suite yet, and on the seven that have, the suite confirms
the perplexity lead clearly only at 2 bits against the reference's default
tuning: against its own SignRound v2 tuning MRound is 0.13 points ahead there on
its current checkpoints, behind on two seeds of three (0.49 on those of its
earlier rate, on every seed), and 0.10 at mixed 2.5. Outside six tables every
reference arm ran AutoRound's default tuning, without the option that turns on
its own SignRound v2 tuning (The bar). On the six tables measured both ways the
margin with that option on is, at 2 bits and MRound's current rate, 9.10, 6.38,
3.43 and 2.05 percent from 0.5 to 7 billion parameters, MRound ahead on every
seed and the ranges apart only at 1.5 billion, where it is 30.52, 17.21, 15.42
and 1.96 with the option off; 12.88 percent at mixed 2.5 on
Qwen2.5-0.5B-Instruct, the ranges still apart, where it is 20.68; and at mixed
2.5 on Qwen2.5-3B-Instruct, as MRound ships, 2.02 percent on every seed with the
ranges overlapping, where it is 6.71 with the option off; before MRound's
default took the reference's gradients at that size it was 3.46 percent behind
there, on two seeds of three. In the Qwen2.5-0.5B-Instruct and
Qwen2.5-1.5B-Instruct tables the reference arms with the option on were produced
under a different operating system release from their default twins, and arms
produced again on the new system came out the same checkpoints to the bit (both
tunings of one seed on the CPU, rows 117 and 118, and a default arm on the
integrated GPU, row 123); the 3 and 7 billion pairs were produced on one system.
The other margins have not been measured against it. No arm has paired the
option with the reference's stronger recipe, which its documentation recommends
at 2 bits, so no margin here says how MRound compares with the reference's best
configuration at low widths. The largest model measured is Qwen2.5-7B, against
the reference at 2 bits only: MRound is 1.96 percent ahead of the reference's
default tuning there and 2.05 ahead of its own SignRound v2 tuning, on every
seed with the ranges overlapping; at 4 bits it has been set against
round-to-nearest alone above half a billion parameters, 5.65, 7.17 and 0.75
percent ahead at 1.5, 3 and 7 billion (rows 235, 236 and 232), which states no
bar, and against the method MRound implements the lead shrinks as the model
grows, so nothing here should be read as a claim about larger models, or about 4
bits above half a billion parameters or mixed precision above three. MRound's
default chooses its mixed precision scoring gradients by size, and the
threshold, 2.5 billion weights in the scored layers, lies between the two models
measured on either side of it, 0.36 and 2.77 billion, with nothing measured
between them. At 1.5 billion MRound's block loop once used four fifths of the
Mac it ran on at the standard recipe; the current tree's 2 bit runs peak at
10.93, 14.14 and 20.07 GB from 0.5 to 3 billion parameters (rows 159, 165 and
151) and at 28.84 GB at 7 billion with every step in four pieces (rows 171 and
173), as its 4 bit run did, which took 57,008 s (row 232), on a 32 GB Mac, and a
3B mixed 2.5 run peaked at 23.95 GB in its scoring pass in a two step memory
probe, which is not a ledger row.

Cost is not reported as a ratio in any table, as above: every comparison in
the tables crosses a device or a machine, and the pairs that share one GPU are
single seeds, stated there with what qualifies them.

Perplexity is a proxy. It is the standard one in this literature and it is what
makes these results comparable to published work, but a model that scores well
on it can still be worse in ways a user notices.

## Reproducing any of this

Every row names its model, scheme, calibration seed and recipe, and every run
writes a configuration file beside its checkpoint that reproduces it. The
reference arms were produced with Intel AutoRound, on the Linux machine for most
rows, on the Mac's own CPU for rows 9, 29, 30 and 35 (and row 44, which repeats
row 30 with its cost corrected) and on the Mac's GPU for rows 82, 83, 85 and
86; each row's `cost.reference_host` says which, and for row 9 its provenance
row, 55, does. For the rows written before 2026-09-23 the
provenance row that cites it (rows 55 to 79, `source: "row N"`) restates the
producing machine's release, torch build, interpreter, system and device in
one block, and every row since carries that block itself
(`reference_environment`); that machine's package inventory is recorded in
`benchmarks/environments/` under the hash the rows cite. A row's own
`environment` block is the machine that scored it, which for every head to
head row is the Mac. MRound's side reproduces exactly: the same configuration
produced twice on the Mac (rows 86 and 95) gave the same checkpoint file, byte
for byte, and the same perplexity to every digit. The scoring harness
that produced the comparisons is not part of this repository, because it drives
the reference implementation; what is here is enough to reproduce MRound's side
and to score any two checkpoints against each other. The head to head arms
call `mround.api.quantize` with `dense_remainder=True`, the matched coverage
described above, and `scale_init="searched"` at every width, including 4 bits,
where the default is the observed range; the command line exposes neither, so
the commands below show the workflow rather than reproduce a row. The zero
shot suite is `mround eval --zeroshot` (with the `eval` extra), and the check
of MRound's evaluator against `lm_eval`'s own is `scripts/check_zeroshot.py`.

```bash
mround quantize mlx-community/SmolLM2-135M-Instruct -o ./smollm2-4bit --bits 4
mround eval ./smollm2-4bit
mround eval ./smollm2-4bit --zeroshot
```
