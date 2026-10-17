# Changelog

Notable changes to MRound, newest first. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the versions follow
[semantic versioning](https://semver.org/spec/v2.0.0.html), with the alpha
caveat that public interfaces may change between pre-release versions without a
major bump. Each version is dated by its release tag.

## 0.1.0a4

The release that makes memory something a run predicts rather than discovers. A
uniform run no longer holds the whole model, a tuning step can run its batch in
pieces, and every run predicts its peak memory from the model's configuration
before it loads anything, refusing a run that would not fit and naming the
number of pieces that would: four for Qwen2.5-7B-Instruct on a 32 GB Mac, by
that prediction, though no 7 billion parameter model has been quantized end to
end yet. The ledger grows from 116 rows to 129, appended and never edited: the
2 bit table on Qwen2.5-3B-Instruct under each of the reference's two tunings,
the fourth model and the first above two billion parameters; controls showing
that the reference repeats its checkpoints to the bit across a reinstalled
system, another release of a GPU's compute runtime and a later release of its
own, auto-round 0.16.0; and two rows showing that the memory work moved no
checkpoint, and what running a step in pieces does move.

### Changed

`mround_run.json` measures the two stages after the block loop, `remainder`,
the tensors outside the blocks rounded to nearest, and `export`, and the run's
peak is the whole run's, the export included, where it was the block loop's.

The block loop holds less memory and computes the same numbers. A quantization
run loads the model lazily and reads each block's weights when it reaches the
block, releasing the block's quantized layers once the quantized stream has
passed through them, so a uniform run no longer holds the whole model. The
reconstruction targets are written over the reference inputs they are computed
from and the quantized stream advances in place, so two activation streams are
resident while a block is tuned rather than three. The codes wait for the
export at one byte each rather than in float32, in the block loop and on the
round-to-nearest path alike, and the export widens them before adding the
storage offset. On a three block model on MLX's CPU backend every exported
tensor is the same to the bit as before, in four runs: 2 bit searched with and
without quantized inputs, 4 bit with a holdout, and 8 bit asymmetric. On Metal,
row 92's configuration, Qwen2.5-1.5B-Instruct at 2 bits, wrote the same
`model.safetensors` to the byte and peaked at 19.34 GB where it had peaked at
27.27 (row 121). Each
block's progress line now gives its peak and the stage that reached it, and
`mround_run.json` records the peak of every stage of every block under
`result.memory`; the losses measured at both ends of a block's tuning are now
a stage of their own, `score`, apart from the steps, `tune`.

The mixed precision scoring pass holds less memory and computes the same
scores. It held the quantized copy of the block weights the backward runs
through, one float32 perturbation per candidate width and every gradient at
once, and every block's activations through the backward. It now keeps each
width's searched grid rather than its perturbation, rebuilds a layer's
perturbations when the layer's gradient arrives, reduces and drops the
gradients a block at a time from the last block, and runs every block under
gradient checkpointing, so the backward holds one block's activations. On
MLX's CPU backend the scores are the same to the bit, under both gradient
sources. The cost is one more pass through the quantizer per narrower width
per layer per batch, and one more forward through each block in the backward.

The losses measured before and after a block's tuning are taken one batch at a
time rather than as one lazy sum over the batches, which held many batches'
forward passes at once and was the largest stage of a 3B run, 25.42 GB against
the steps' 20.07. The same 3B run now peaks at 20.07 GB, the measurement at
12.43, and writes the same checkpoint to the byte. A mixed precision run loads
the model lazily again once it has scored it, since the scoring pass reads
every weight and the scored model kept them resident through the block loop,
5.4 GB more at a 3B model's first block. With both changes a 3B mixed 2.5 run
peaks at its scoring pass's 23.95 GB, where its block loop had reached 30.11
GB; its block loop holds what a uniform run's does, and it writes the same
checkpoint to the byte.

The scale search takes a layer a slice of rows at a time, each slice's
intermediates near 256 MB, where it held about 270 bytes per weight of the
layer at once: 6 GB for a 3B model's widest layer and 18 GB for a 7B's. Every
group's search is its own, so every scale is the same to the bit.

### Added

A memory check before every run. `api.quantize` and `api.plan_mixed_precision`
predict a run's peak memory from the model's configuration before anything is
loaded, and for a Hub model before its weights are downloaded, and refuse a run
predicted past `memory_budget`, 85 percent of the Mac's memory unless given,
with the stage that sets the peak and the fewest `TuningConfig.micro_batches`
that would fit; `memory_check=False` predicts, records and runs anyway.
`api.predict_memory` makes the prediction alone, on any machine. `mround
quantize` takes `--memory-budget` in GB, `--skip-memory-check` and `--dry-run`,
which prints the prediction stage by stage without loading the model and exits
1 when the run would be refused, so `-o` is required only without it; `mround
doctor` shows the device's memory and the budget. The prediction
(`mround/planner/memory.py`) is a fit to the stage peaks of the five runs on
the current tree, Qwen2.5-1.5B-Instruct and Qwen2.5-7B-Instruct at 2 bits in
eight pieces and Qwen2.5-3B-Instruct at 2 bits and at mixed 2.5, and of eleven
configurations that `scripts/probe_memory.py` measured on Metal, two synthetic
blocks at a model's shape with nothing downloaded: a 7B whole and in two, four
and eight pieces, SmolLM2-135M, Qwen2.5-0.5B and Llama 3 8B, and four sequence
lengths at a fixed number of tokens. Its entrance is accounting, exact to eight
bytes on every run recorded; a step's peak is the larger of its MLP's tensors
and its attention's, not their sum, which the sequence lengths showed; and the
step was fitted to err toward refusing, so every peak is predicted between 1.0
percent short and 8.2 percent over, and the five runs' within 1.1 percent. On a
32 GB Mac it refuses Qwen2.5-7B-Instruct at the standard recipe whole,
predicted at 39.5 GB, and names four pieces, predicted at 28.7. Each run
records its prediction in `mround_run.json` under `memory_check`, beside what
it measured.

`TuningConfig.micro_batches`, and `--micro-batches` on `mround quantize`, run
each tuning step's batch in that many pieces, forward and backward, with the
pieces' gradients summed, so a step holds one piece's activations: for a model
whose step does not fit whole. The step is still the batch's. Each piece's loss
is divided by the whole batch's element count, and the outlier rule's threshold
is taken over the whole batch from a forward pass without gradient before the
pieces run, which costs that pass. The default, 1, runs every batch whole and
writes the same checkpoint as before. A larger value changes only the order of
the step's additions, and on Metal at 2 bits that moved Qwen2.5-1.5B-Instruct's
checkpoint as far as a change of seed and took 1.9 times as long a block in
eight pieces (row 122), so it is for a model whose step does not fit whole. It
is not the reference's `gradient_accumulate_steps`, which makes the batch
larger and which MRound still refuses. The comparison harness takes
`--micro-batches` for MRound's side, puts it in the run's directory name and
records it in the ledger row's `tuning`.

The modules below the API export what these are built from.
`mround.pipeline.loader.read_config` reads a model's configuration without its
weights, and `load_model` takes `lazy`; `mround.pipeline.device` exports
`device_memory`, `active_memory` and `StageMeter`, which keeps the peak of each
stage of a run; `mround.pipeline.blocks` exports `checkpointed_blocks`,
`piece_bounds` and `require_batch_free`, and `apply_block` takes `in_place`;
`mround.core.losses` exports `outlier_threshold` and `piece_loss`;
`mround.formats.mlx_export` exports `narrow_codes`; and `mround.exceptions`
exports `MemoryBudgetError`, which the memory check raises. `BlockResult`
carries each stage's peak, and `RunSummary` the stages before and after the
block loop. Every new argument defaults to what the code did before.

Row 122 is row 121's configuration with every tuning step in eight pieces,
scored against the same reference arm. MRound's allocator peaked at 8.26 GB
against 19.34, its block loop took 16,159 s against 11,004.8 (about 570 s a
block against the 301 of the blocks of row 121 that ran with nothing else on
the Mac), and its perplexity is 21.1863 against 20.8947, 12.35 percent below
the reference's 24.1720 where row 121 is 13.56. The two checkpoints differ in
62.6 percent of their weights and sit 0.33 apart, which is the reference's own
distance between two seeds at 2 bits; MRound's three seeds at that size scored
20.89 to 22.53.

`scripts/compare_checkpoints.py` reads MRound's own checkpoints: MLX's packed
codes, with the scales and biases beside them and each layer's width and group
size from `config.json`. Two packed checkpoints are compared on what they
store, codes and scales, with no lattice inferred; a packed checkpoint against
a fake export is compared on the packed side's stored scales.

Row 121 is row 92's MRound arm produced again after that change and scored
against row 92's reference arm. Both perplexities repeat row 92's to the last
digit, and MRound's allocator peaked at 19,338,070,546 bytes against
27,265,701,016, its resident set at 1.44 GB against 4.03.

Rows 117 and 118 are seed 42's two reference arms of the 2 bit
Qwen2.5-0.5B-Instruct table produced again, the default tuning's (row 88) and
the reference's own SignRound v2 tuning's (row 105), on the Linux machine after
its operating system had been reinstalled and its environment rebuilt. Each
checkpoint is the one its row scored, all 357,826,560 weights, and each row
repeats its original's perplexities to the last digit. The change of operating
system release that RESULTS.md listed among what the two SignRound v2 tables
leave open therefore moved neither tuning at that seed, and the two reference
columns are read as differing in the option alone. On one system the option
made the reference's block loop 13.2 percent longer at that seed, where the
reading across the two systems had given 16.9. RESULTS.md and README.md now say
so where they raised the question, and `scripts/compare_checkpoints.py` prints
the reproduction across a change of operating system beside the reference's
other floors.

Rows 119 and 120 are the same two arms produced under auto-round 0.16.0, the
reference's current release when they were produced, on the same machine and
system. Both checkpoints are the 0.15.0 ones to the bit, and both rows repeat
the perplexities of rows 88 and 105 to the last digit, so the tables, all of
them produced under 0.15.0, hold for 0.16.0 at that point; RESULTS.md says so
under Which release of the reference. Under 0.16.0 the option made the block
loop 14.3 percent longer, where the 0.15.0 pair had given 13.2.

Row 123 is row 91's reference arm, seed 44 of the 2 bit Qwen2.5-0.5B-Instruct
table on the Linux machine's integrated GPU, produced again there after the
reinstall, on another release of the GPU's compute runtime and in its
environment rebuilt. The checkpoint is row 91's, all 357,826,560 weights, the
row repeats row 91's perplexities to the last digit and its device memory to
the byte, and the arm took 4,995.8 s against 4,958.6. The reference's arms on
that GPU repeat themselves across a change of system and runtime as its CPU
arms do; RESULTS.md says so beside the reference's other floors, and
`scripts/compare_checkpoints.py` prints the runtime among the changes that
move nothing.

Rows 124 to 126 are the 2 bit table on Qwen2.5-3B-Instruct, the fourth model
and the first above two billion parameters: seeds 42, 43 and 44, the reference
at auto-round 0.15.0 under its default tuning on the Linux machine's
integrated GPU, in the environment row 123 cites, and MRound on one tree.
MRound is ahead on every seed, 17.2031 against 19.8912 on the mean, 13.51
percent, its worst seed 8.78 percent better than the reference's best, at 6.19
percent more bytes. Against that tuning the 2 bit margin had narrowed from
30.63 percent on Qwen2.5-0.5B-Instruct to 12.25 on Qwen2.5-1.5B-Instruct; at
3 billion it did not narrow further. MRound's allocator peaked at 20.07 GB on
every seed and its block loop took 16,497 to 16,562 s on the Apple GPU,
against the reference's 21,759 to 21,771 s on the integrated GPU. RESULTS.md
has the table and README.md its row, beside a row for the 1.5 billion table
it had left out.

Rows 127 to 129 are the same three Qwen2.5-3B-Instruct arms with the
reference's own SignRound v2 tuning on, each produced on the integrated GPU on
one system and in one environment beside its default twin, and scored against
the same MRound checkpoint. The two implementations are level: MRound is ahead
on seeds 43 and 44, by 4.34 and 2.34 percent, and 2.94 percent behind on seed
42, 1.25 percent ahead on the mean (17.2031 against 17.4211). The option takes
the reference 12.42 percent below its default arms on the mean and closes 92
percent of the gap in mean perplexity, where on Qwen2.5-0.5B-Instruct at 2 bits
it closed 77; it made the block loop 5.4 percent longer, both arms uncompiled.
RESULTS.md, README.md and the paper carry the table, and the paper's Figures
3 and 4 gain the 3B table of rows 124 to 126.

### Tests

`tests/unit/test_results_ledger.py` holds rows 117 and 118 to rows 88 and 105
in every field but those of the system the reference arm was produced on, and
to each other in every field of that system but the tuning, and rows 119 and
120 to rows 117 and 118 in every field but the release and the environment that
carries it, row 121 to row 92 in every field but what producing MRound's arm
cost, row 122 to row 121 in every field but MRound's perplexity, its cost and
the field that names the pieces, row 123 to row 91 in every field but those of
the system its reference arm was produced on, its device memory to the byte,
rows 124 to 126 to one configuration on three seeds, differing in what the seed
decides alone, and rows 127 to 129 likewise and to their default twins in every
field but the reference arm's, whose blocks differ in the tuning alone.
`tests/unit/test_runner_memory.py` holds the memory levers to the bit: the
export from one byte codes against the export from float32 codes at every width
MLX stores, the in place stream against a fresh one, the weights a finished
block gives back and the ones it keeps, the stage peaks, the lazy load reaching
`mlx-lm`, every scale the sliced search gives against the whole matrix's, and
the losses at both ends of a block's tuning to what computing the targets
holds. `tests/unit/test_api_run_record.py` holds a mixed run to tuning a second
lazy load rather than the scored model. `tests/unit/test_micro_batches.py`
holds a step in pieces to the whole batch's loss and gradient, shows that a
threshold taken piece by piece would be another rule, and holds the step's
memory to falling with the pieces. `tests/unit/test_scoring_memory.py` holds
the scoring pass's scores to the bit with checkpointing on and off, its rebuilt
perturbations to the stored ones, and its memory to about one quantized copy of
the block weights and one block's activations.
`tests/unit/test_compare_checkpoints.py` reads packed words packed bit by bit
at every width MLX stores, and a checkpoint MRound's own exporter wrote. Each
lever was shown to fail its test when removed. `tests/unit/test_memory_rule.py`
holds the memory rule to every run and every probe configuration it was fitted
on, stage by stage, and `tests/unit/test_memory_check.py` and
`tests/unit/test_api_run_record.py` hold the check to refusing before the model
loads, to the configuration it reads and to the width the weights loaded at,
and the command line to its budget, its switch and its dry run; 56 of 57
mutations of the rule and 12 of 12 of its wiring fail them. The published suite
collects 1,329 tests where MLX is installed, against 1,027 at 0.1.0a3.

### Infrastructure

`benchmarks/environments/` carries `91d06b756b312979`, the reference's Linux
CPU environment as rebuilt on 2026-10-08 and inventoried by the process that
ran the reference, which rows 117 and 118 cite, and the environment of
auto-round 0.16.0 beside it, `a76996b3cffc0659` as its check inventories it and
`ac1862f7cb75f30a` as the reference's process does, which rows 119 and 120
cite, and `24635a7aabaa22ff`, the integrated GPU's environment as rebuilt on
2026-10-14, which row 123 cites.

## 0.1.0a3

The release in which the measurements outgrew the first two. The ledger grows
from 54 rows to 116, appended and never edited: every three seed table now
faces one release of the reference, auto-round 0.15.0; two tables also face
the reference's own SignRound v2 tuning, which no arm had run before; the
reference has run on two GPUs as well as on CPUs; and five tables have been
through the zero shot task suite this release adds. It also fixes code that
did something other than its documentation said, and corrects the published
documents where a second audit of the repository found them wrong.

### Changed

The run ledger is at schema 2. A row's `environment` block is now stated to be
the machine that wrote the row, which for a head to head row is the machine
that scored it and not the one that produced the reference arm, and two blocks
are added: `reference_environment`, the reference arm's manifest hash, release,
torch build, interpreter, system, host and the device the reference reported
tuning on, read from the sidecar the producing machine writes; and `memory`,
peak figures where a harness observed them, resident set and allocator peak
kept apart. Rows 55 to 79 restate the reference environment of every head to
head row written under schema 1, one `provenance` row each, because the ledger
corrects by appending and never by editing; readers averaging perplexities skip
rows of that kind, whose measurement fields are all null. Every reference arm
recorded under schema 1 ran on a CPU, and each provenance row says how that is
known.
`mround.eval.results.peak_resident_bytes` reports a process's maximum resident
set in bytes on Linux and macOS alike.

The 4 bit Qwen2.5-0.5B-Instruct comparison in RESULTS.md, README.md and the
paper is row 95, the reference under auto-round 0.15.0 on the Linux CPU
against MRound's current tree, in place of row 30, the reference under 0.14.2
on the Mac's CPU against an older tree: 15.2197 against 15.0260, MRound 1.27
percent ahead, where row 30 read 15.0777 against 15.0807, 0.02 percent
behind. The new margin is the size of the 1.17 percent the reference's own
four productions of that configuration span, and at the same bytes the
reference is 0.50 percent ahead (row 96), so the documents call it a tie where
they had said "matches". Every table there now faces auto-round 0.15.0 except the
single seed 2 bit SmolLM2-135M-Instruct row.

RESULTS.md, the README and the paper now say which of the reference's tunings
each comparison faces. AutoRound keeps its own SignRound v2 tuning, a searched
initial scale and, below 4 bits, the outlier suppressed loss, behind its
`enable_alg_ext` option, off by default. Every reference arm up to row 104 ran
with it off, so those margins are against the reference as it ships; rows 105
to 110, under Added, are the arms with it on, and the documents set the 2 bit
and the mixed 2.5 bit Qwen2.5-0.5B-Instruct margins against both tunings. The
section on the reference's longer recipe no longer calls that recipe its best.

### Added

The reference implementation has been run on the Apple GPU, through torch's
MPS backend, held to float32 as its CPU arms were: ledger rows 82 and 83, the
first head to head rows whose reference arm and MRound arm ran on the same
device. RESULTS.md carries the pair with its qualifications. Row 80 is void
and retracted by row 81, the ledger's first `retraction` row; the scoring
harness now refuses two checkpoints that are not the same model before it
scores anything. Row 84 is the reference on the Linux machine's own
integrated GPU, Intel Arc graphics, at mixed 2.5 under auto-round 0.15.0:
2.89 times faster than its CPU arm and 3.03 percent worse. Row 85 is the
reference on the Apple GPU at its own default precision there, bfloat16
under autocast: at 4 bits on Qwen2.5-0.5B-Instruct its block loop is within
seven percent of MRound's on the same GPU, and RESULTS.md now states that pair
beside the float32 one. Row 86 reruns MRound's side of that pair on the
current tree and is the first row with memory on both sides: MRound's MLX
peak and resident set, the reference's resident set, and, in the sidecar
from now on, the reference's device peak with the name of the counter it
came from. Row 87 is the reference at its default precision on the Arc
graphics with the compiler off, since the platform's compiler refuses the
bfloat16 kernels: its fastest and best arm at that seed, and still 20.91
percent behind MRound. Rows 88 and 89 are the 2 bit Qwen2.5-0.5B-Instruct
reference at seeds 42 and 44 under auto-round 0.15.0, produced on the Linux
CPU; with row 33 they make the three seed 2 bit table face one release, the
same one as the mixed 2.5 tables. RESULTS.md, README.md and the paper now
carry that table: a mean margin of 30.63 percent, against 25.67 for the
0.14.2 table it replaces, whose rows 31, 34 and 37 stay in the ledger as the
release comparison. The release moved the reference by 2.30, 0.22 and 25.35
percent on the three seeds, so the statement that it is "worth about three
percent" is now labelled as the mixed 2.5 figure it always was. Rows 90 and
91 are the same two seeds with the reference on the Arc graphics at its
default precision there, uncompiled: 3.5 times faster than the CPU arm at
2 bits as at mixed 2.5, 8.56 and 12.13 percent better than the CPU arms of
the same release, 18.09 and 29.03 percent behind MRound, and the first rows
whose reference device peak comes from a device counter. Comparing the three
release pairs of the 2 bit table code by code found the reference 0.31 to
0.32 from itself at every seed, the seed whose perplexity moved a quarter
included; RESULTS.md reads MRound's distance from the reference's default
checkpoints at 2 bits against that floor: 0.47 to 0.48 relative to MRound's
own weights, 0.45 at 0.5 billion relative to theirs.
Rows 92 to 94 are Qwen2.5-1.5B-Instruct at 2 bits on three seeds, the reference
on the Arc graphics under 0.15.0 and MRound on the current tree: a margin of
12.25 percent, replacing the single seed of row 36 (10.17 percent) in
RESULTS.md and the paper, whose Figure 3 gains a fourth panel. On a GPU the
two implementations take the same 2.2 times as long for three times the
parameters, so the lead that grows with model size against a CPU arm is the
CPU's, and the paper's cost paragraph says so. The same rows carry memory on
both sides: MRound's allocator peaked at 27.27 GB at 1.5B, four fifths of the
Mac it ran on. Rows 95 and 96 are Qwen2.5-0.5B-Instruct at 4 bits at group
sizes 64 and 128, the reference's default being 128, with the reference under
0.15.0 on the Linux CPU and MRound on the current tree: MRound 1.27 and 1.37
percent ahead at the same group size, the reference losing 1.89 percent and
MRound 1.80 to the coarser group. MRound's checkpoint at 128 holds exactly the
tensor bytes of the reference's at 64, because MLX stores a bias beside every
scale, and at that one size the reference is 0.50 percent ahead; RESULTS.md and
the paper carry the pair as a table. MRound's side of row 95 repeats row 86's
configuration and wrote the same checkpoint to the byte, with a wall clock 4.1
percent and an allocator peak 2.5 percent above row 86's.

Rows 105 to 107 are the 2 bit Qwen2.5-0.5B-Instruct table again with the
reference's own SignRound v2 tuning turned on (`enable_alg_ext`), every other
setting matched and MRound's three checkpoints unchanged: 47.7076, 40.2103 and
41.6129 for the reference against 41.2332, 36.9670 and 39.3524, a margin of
9.25 percent where the default arms give 30.63, so most of the 2 bit
Qwen2.5-0.5B-Instruct margin sat between the reference's two tunings and not
between the two implementations. MRound is ahead on every seed, by 5.43 to
13.57 percent, but the two ranges overlap. RESULTS.md carries the table beside
the first one with what it leaves open: the option together with the
reference's stronger recipe, and a change of operating system release on the
machine that produced the two sets of reference arms. A row whose reference
arm was produced from 2026-10-01 on says
in its `reference_environment` which tuning the reference ran (`algorithm`)
and whether its block forward was compiled (`block_forward_compile`).

Rows 108 to 110 are the mixed 2.5 Qwen2.5-0.5B-Instruct table with that tuning
turned on in the same way, MRound's three checkpoints again unchanged: 26.3575,
25.1924 and 25.4831 for the reference against 22.9964, 21.6389 and 22.4770, a
margin of 12.88 percent where the default arms give 20.68. MRound is ahead on
every seed, by 11.80 to 14.11 percent, and here the two ranges stay 2.20
points apart. The option closed 43 percent of the gap in mean perplexity at
this scheme, where it closed 77 at 2 bits, and the reference's allocation was
no part of it: all six of its checkpoints give every layer the same width.
RESULTS.md carries the table under Mixed precision with what it leaves open,
which is what the 2 bit one leaves open, and sets the option beside the
stronger recipe at the one seed that has both.

Rows 111 to 116 score the six pairs of those two tables on the zero shot task
suite. MRound's checkpoints are the ones rows 98 to 103 scored, and their
figures came out identical to the last digit. Against the reference's own
SignRound v2 tuning MRound's mean accuracy over the twelve tasks is 0.49
points ahead at 2 bits, on every seed but by 0.14 and 0.03 on two of them,
where the default arms gave 2.05, and 0.10 points ahead at mixed 2.5, behind
on one seed of three, where they gave 0.69. LAMBADA, its perplexity and
`acc_norm` still favor MRound at every seed of both. RESULTS.md carries both
tables beside the default arms' ones.

`mround eval --zeroshot` runs a zero-shot task suite: the reference
implementation's own default tasks less wikitext (twelve, all log
likelihood), through `lm_eval`'s task definitions and MRound's MLX scoring,
with `--tasks` and `--limit` for narrower runs. `mround.eval.zeroshot`
merges requests that share a model input, so the four answers of an MMLU
question cost one forward, and takes the log softmax normalizer in blocks,
because a single float32 reduction over a vocabulary row on MLX's CPU backend
comes out 1.3e-4 low. `scripts/check_zeroshot.py` runs it beside `lm_eval`'s
own Hugging Face adapter on the same documents, MLX on the CPU and `lm_eval`
in float64, and fails unless every request agrees to float32 arithmetic and
every decision is the same; on the Mac it agreed to 1.1e-4 nats at worst on
SmolLM2-135M-Instruct, twenty documents of each task, and 2.2e-4 on
Qwen2.5-0.5B-Instruct at five. In float32 `lm_eval`'s side is no reference at
that precision: torch's float32 log softmax lifts every token's log
probability, by 3.3e-5 to 1.06e-4 nats on the Mac depending on the
vocabulary, which moved long answers by up to 3.9e-3 nats, and
`--lm-eval-dtype float32` repeats that comparison. The first zero-shot result
is ledger row 97, which RESULTS.md and the paper now carry: the three
checkpoints of the 4 bit Qwen2.5-0.5B-Instruct comparison, scored on the
Apple GPU, with mean accuracies over the twelve tasks of 48.42 percent for the
original, 47.58 for the reference and 47.42 for MRound, a tie as the
perplexity is. Rows 98 to 100 score the three seeds of the 2 bit
Qwen2.5-0.5B-Instruct table against the reference's default tuning (rows 88,
33 and 89) the same way: MRound's mean accuracy is 39.02
percent against the reference's 36.97, ahead on every seed by 1.94 to 2.22
points, with much of the suite at its floor at that width, which RESULTS.md
states beside the table. Rows 101 to 103 score the mixed 2.5 bit
Qwen2.5-0.5B-Instruct table's three seeds, where the suite is equivocal:
42.52 percent for MRound against 41.83, ahead on two seeds and behind on the
third, the sign decided by MMLU, BoolQ and RTE, which swing by up to 11.5
points between seeds of one implementation. Row 104 scores the 4 bit group
size 128 pair: 46.98 for MRound against 47.30, and at the same bytes the
reference is 0.60 points ahead, as it is 0.50 percent ahead in perplexity. On
the GPU the evaluator's log likelihoods sit about
a hundredth of a nat from the CPU's, and the documents state that beside the
tables. The `eval` extra now asks for `lm-eval>=0.4.12`, the release the
evaluator was checked against. The run ledger gains a `zeroshot` kind and a
`tasks` field for those results: one row per head to head row, the original,
the reference and MRound side by side with each task's accuracy, standard
error and document count, naming the row whose checkpoints it scored in
`evaluation.head_to_head_row` and, from row 98, the MLX device that scored
them in `evaluation.mlx_device`.

`scripts/compare_checkpoints.py` compares two fake quantized checkpoints of
one model code by code: it recovers each group's scale from the exported
values, says whether the two sides share a lattice, counts the codes that
moved where they do, and prints the mean relative disagreement the head to
head harness prints. Run on the ledger's own checkpoints it found that two
reference arms differing only in the device they tuned on differ in 64
percent of their weights at mixed 2.5 and 85 percent at 4 bits, as much as a
change of release or of seed, while one seed produced twice on one device
reproduces every weight; RESULTS.md says what that does to the reading of
every weight comparison and every single reference number. The script prints
that floor beside every result, width by width and with the changes each
width's was measured across; the 2 bit figure, 0.30 to
0.34, comes from the three release pairs of the 2 bit table (rows 31, 34 and
37 against 88, 33 and 89), from row 36 against row 92, and from the seed pairs
of rows 88, 33 and 89 and of rows 105 to 107. Between the reference's default
tuning and its own SignRound v2 tuning at one seed it found a difference of
another kind, 0.40 to 0.41 relative to the default checkpoint, with three
groups in a thousand on a shared lattice. Taken relative to the SignRound v2
checkpoint, MRound's checkpoints sit 0.37 from it (the scoring harness's
comparison taken on that denominator, which this script does not compute for a
packed MRound checkpoint) and the reference's default checkpoint 0.43. At
mixed 2.5 (rows 43, 45 and 46 beside rows 108 to 110) the same change of
tuning is 0.284 to 0.285, with seven groups in a thousand on a shared lattice,
where two seeds sit 0.22 to 0.24 apart under either tuning, and layer by layer
the figure follows the width, 0.40 at the 2 bit layers and 0.22 at the 3 bit
ones; the script prints the mixed figure beside the 2 bit one. A
layer's width is read from its left side, so a layer whose two sides share no
lattice is not printed as 0 bit.

Three more scripts serve the ledger. `scripts/retract_row.py` voids a row by
appending a `retraction` row that names it, which is how row 81 retracts row
80; `scripts/backfill_reference_provenance.py` wrote rows 55 to 79 from the
sidecars the producing machines left beside their checkpoints, and stops on any
disagreement with the row it restates; `scripts/probe_reference_device.py`
answers, before a five hour run, whether the reference tunes on a given
accelerator at all or falls back to the CPU. `mround.core.quantizer` exports
`quantize_codes`, and `mround.reference.tuning.round_to_nearest` takes the
`eps` and `init_scale` its MLX counterpart takes.

### Corrected

RESULTS.md said the reference implementation has no Metal path. Its device
manager defaults to the CPU on Apple Silicon and offers `--device_map=mps` as
an override, which rows 82 and 83 exercised; the document now says what it
does and what it found.

RESULTS.md and the paper set the reference's device memory on the Apple GPU,
11.79 GB, beside MRound's 13.58 GB as if the two were in one unit. The
reference's monitor prints binary gigabytes; the figure is 12.66 GB, and the
two are 7.3 percent apart rather than 15.

`api.quantize` resolved the wrong scale initialization when `symmetry` was
given as a string: `symmetry="sym"` at 2 bits fell to the observed range, the
configuration that left SmolLM2-135M-Instruct at a perplexity of 77,365 at 2
bits in August, where the enum resolved the searched grid. `default_scale_init`
now treats the string as the member. A negative `holdout_batches` divided by
zero inside the first block; `quantize` now refuses it before anything loads,
and so does the block runner, as `plan_mixed_precision` refuses a
`scoring_batch_size` below 1 instead of quietly running at 1. `mround_run.json`
now records `quantized_inputs`, which changes how every block is tuned and
cannot be recovered from the weights. `evaluate_perplexity` refuses a token
budget below one, which sliced the text from its end or blamed the data set,
and a one token window, which divided by zero. `save_calibration_set`
substituted its suffix rather than adding it, so `pile.seq2048` was written as
`pile.safetensors` and two names differing after a dot wrote one file. And when
two widths covered the same number of layers, a checkpoint's `config.json`
named as its default whichever came first in a set whose order Python
randomizes per process, so one run could write 2 bits and its repeat 4; it now
names the narrower one, and the loaded model was the same either way. None of
these can move a published number.

The README's Python example passed `output=`, which `api.quantize` does not
take; it is `output_dir=`. The development setups disagreed with each other and
with what a first run needs: environment.yml and CONTRIBUTING.md installed
`.[dev]`, which lacks the model stack, and GETTING-STARTED.md's manual setup
`.[models]`, which lacks the test tools; all three install
`.[dev,models,eval]`. The README said the largest block governs a run's peak
memory, when as built the whole model is loaded; it now says the whole model
has to fit, and gives the two peaks measured. DOCUMENTATION.md now says that
the block loop `quantize` runs keeps the parameters of its last step and
reverts a block whose loss did not fall, where the layer level `tune_layer`
keeps the best measured ones (5.4); that the opposite extreme of a symmetric
group clips a code whenever it lies within half a code of the dominant one,
about 70 percent of Gaussian groups of 64 at 2 bits with unit coefficients, and
not only on an exact tie (5.2.1); that negating a group's codes is not an exact
way to make its scale positive (6.2 and 6.3); that the 43 percent of codes on a
boundary at 2 bits comes from one parity test and is not a property of model
weights (1.2 and 5.7); and that ledger rows 10 to 44 hold the requested 65,536
tokens in `evaluation.tokens` where 65,504 were scored (8.1).

### Tests

The published suite collects 1,027 tests where MLX is installed, against 880 at
0.1.0a2. Besides the tests of everything above, a second audit tightened tests
that could not fail, each shown by a mutation of the code under test that left
it green: a clause of the zero shot check, six ledger tests that skipped
silently when run from outside the repository root, the scale search quality
loop, which never ran at 32 groups, a momentum test that a moving average
passed, and a dequantize sign error the parity suite missed. Ledger tests hold
every row from 55 on to its kind, every zero shot row to the head to head row
it cites, and the rows with the reference's own SignRound v2 tuning to their
default twins.

### Infrastructure

The pre-commit type check installs numpy 2.4.6, as continuous integration
does; without it strict mode failed every numerical file. environment.yml
installs the `models` and `eval` extras with the development tools. NOTICE
lists `datasets` and `lm-eval`. `.gitignore` covers the output folders the
documents tell a reader to type. `benchmarks/environments/` carries two more
package manifests: the reference's environment on the Arc graphics
(`5cf84f77cc0b531e`) and its Linux CPU environment as rebuilt on 2026-10-08
(`5d38c3bb6cad589b`), which no row cites yet.

## 0.1.0a2

A corrections release, one day after the first. A full audit of the repository
found statements in the published documents that the ledger contradicted and
one default the project had already decided against. No measurement changed;
every figure in RESULTS.md is the same ledger row it was.

### Corrected

RESULTS.md said the reference implementation's arms were produced on a Linux
machine. For three of the rows it cites, rows 29, 30 and 35, they were produced
on the Mac's own CPU, as the ledger's `cost.reference_host` recorded. The
document now says which rows ran where. It also described wall clock ratios it
never printed; it now states that no cost ratio is reported, and why.

The mixed precision example in README.md and GETTING-STARTED.md named
`mlx-community/Qwen2.5-0.5B-Instruct`, a repository no measurement used. It
now names `Qwen/Qwen2.5-0.5B-Instruct`, the model every result was produced
with.

`mround.api.plan_mixed_precision` hard coded the searched scale initialization
where `quantize` resolves it from the narrowest width a run can assign, so an
asymmetric plan raised an error about a search nobody asked for. Both entry
points now default to `None` and resolve the same way. The command line's
preview of that resolution includes the remainder width, and `--iters` defers
to the standard recipe as the other recipe flags always did.

### Changed

`TuningResult.losses` from the layer level `tune_layer`, in both the NumPy
reference and the MLX implementation, has one entry per step: the initial loss
was measured once before the loop and again at step zero, and the duplicate is
gone. `initial_loss` is `losses[0]`. Parameters, best step and every loss value
are bit identical to 0.1.0a1, verified on 48 layers across four widths and two
step budgets.

DOCUMENTATION.md fixes three conventions the code already followed and the
specification did not state: which parameters a layer keeps and why the final
update is never scored (5.4), the floor of at least one excluded element in the
outlier suppressed loss (5.5), and the candidate grid, anchor and tie rules of
the searched scale initialization (5.10). Each follows the reference
implementation where the published method is silent and is named as such.

### Tests

Eight tests added and several tightened. The projection bound on the rounding
perturbation is checked at loop level in both implementations and fails when
the projection is removed. The rule that no residual is formed by subtracting
two matrix products is enforced structurally on the numerical layers. The
license header, status marker and no torch checks parse rather than grep, and
the header check covers every authored directory. A Mac without MLX fails a
test instead of skipping the MLX suite. The learning rate excursion tests
assert the exact closed form.

### Infrastructure

Continuous integration and pre-commit pin ruff 0.16.2 and mypy 2.3.0, the
versions the maintainer's gate runs; pre-commit's mypy hook now covers
`examples/` as the gate always did.

## 0.1.0a1

The first public release. Everything below is new, so this entry describes the
state of the project rather than a set of changes against a predecessor.

### The method

Learned rounding by signed gradient descent, implemented twice and checked
against itself. `mround.reference` is a complete NumPy implementation of
SignRound v2's mathematics that runs on any machine and needs no model:
quantizers at every supported width, the learned rounding perturbation, both
loss functions and bit packing, with finite difference verified gradients, and
the mixed-precision allocator beside it in `mround.planner`. `mround.core` is
the MLX port, mirroring it function for function, with a parity suite that
agrees at a median of 0.0 percent on learned rounding.

Uniform quantization at 8, 4 and 2 bits, symmetric and asymmetric, group-wise
and per-channel. The searched scale initialization is the default below 4 bits
for symmetric schemes, which is what makes 2 bits produce a model rather than
noise; at 4 bits it is quality neutral by measurement, so the observed range
stays the default there.

Mixed precision over a menu of widths under an exact size budget, with
gradient weighted sensitivity scoring deciding which layers get which width.
Three levers are set by measurement rather than by taste: gradients are taken at
the widest candidate width, tensors outside the block stack are protected at a
higher width, and the candidate menu is graded rather than binary.

### Interfaces

`mround.api.quantize` and `mround.api.quantize_round_to_nearest`, both writing
MLX checkpoints that `mlx-lm` loads, each with a configuration file beside the
checkpoint that reproduces the run.

A command line with four commands: `quantize`, `eval`, `doctor` and `version`.
`quantize` refuses rather than proceeding when its group size would leave most
of the block stack unquantized, because a run that silently quantizes a third
of a model and reports success is worse than one that stops.

### Measurement

`benchmarks/ledger.jsonl`, append only, one line per quotable number, each
scored row from row 29 on citing a hashed manifest of the machine's full
package inventory under `benchmarks/environments/`. [RESULTS.md](RESULTS.md)
reads from it and cites it by row.

`scripts/measure_overlap.py`, which compares every function in MRound against
every function in an installed reference implementation, as identifier blind
token streams and as syntax tree shapes, so that the claim of independent
implementation can be checked rather than asserted.

### Not implemented

GGUF export, Core ML export and zero-shot task evaluation all raise
`NotImplementedError` with a pointer, and say so in their own docstrings. A test
checks that those markers and the code agree, so neither can drift.

### Known limitations

Measured on three models, all under two billion parameters, on one evaluation
corpus, by perplexity alone. Memory requirements are not yet characterized.
Wall clock comparisons against the reference implementation compare an Apple GPU
against a Linux CPU, because the reference has no Metal path, and are reported
for what a user experiences rather than as a like for like measurement.
