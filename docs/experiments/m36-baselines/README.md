<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# M3.6 baselines and the 1% protocol

The fixed protocol for M3.6's migration gate
([engine-components.md](../../engine-components.md#migration-and-its-gate)),
the frozen binary every migrated family is compared with, and that
binary's speed, quality controls, peak memory and swap rows.

## The frozen baseline

`main` at `1facc957` ("Project rename to llmp"), tree `a2cd8f7b`, built
with the `spark-native` preset `--locked` (core profile, no optional
module) and SDK `aarch64-2c4eb5b2f68f3fa5` on `spark`. Installed at
`~/.local/share/llmp/m36-baseline/1facc957/` on both `spark` and
`spark-b`, read-only, with the build-tree layout so the binaries' run
paths resolve:

| File | Identity |
| --- | --- |
| `src/runtime/llmp-runtime` | SHA-256 `28890a54aec22c4d892ece6e7e3d115e1e03047638bc3566b55a5803ab8a07e1` |
| `IDENTITY.sha256` (every file of the install: the runtime, `src/cli/llmp`, `lib/llmp`'s cuBLAS 13 pair, all 87 benchmark executables, the build receipt, `BASELINE.json`, the source inventory) | SHA-256 `82627c1f666e8f802518b51fe7cc3871d8aa7439fe144521fbedc88a662648aa` |
| `source-inventory.sha256` (`sha256sum` of every tracked file at `1facc957`; the Spark copy built from was checked against it) | SHA-256 `3b6d0b212c411504235afe4b41ab8998da792bb6aca062141cbe3c0d31e8741e` |

Check a copy with `cd ~/.local/share/llmp/m36-baseline/1facc957 &&
sha256sum -c IDENTITY.sha256`. It stays until M3.6 exits.

**The fixed baseline, `1facc957-fixed`,** is `1facc957` with exactly the
two fixes of its [defects](#defects-in-the-baseline) applied
(`m36-fixes.patch`, in the install; its `src/` hunks are the fix as it
lands, its `tests/` hunks an earlier revision of the tests, which no
installed binary contains), built the same way on `spark` from a clean
export of `1facc957` and installed read-only beside the first, with the
same layout plus the patch, on `spark` and `spark-b`. It is the baseline
of the rows the defects kept from the first:
DeepSeek's DSpark cohort cells and the swap table's Qwen3.8-as-A rows
(below). Every other row stays the first baseline's. The two builds are not
bit-identical even without the patch (an unpatched rebuild in the other
directory gave runtime `c696749c…`: the build paths differ), so the fixed
build's own rows are measured on it.

| File | Identity |
| --- | --- |
| `src/runtime/llmp-runtime` | SHA-256 `f5443363bc5a586bfeea450e5c51b9eefe720d392a312ed9732c665d06abe930` |
| `IDENTITY.sha256` (every file of the install, as above, and `m36-fixes.patch`) | SHA-256 `519bb045b05b9d26c93a61295c66fbed744e768ccfac505fe56b3157370a7693` |
| `m36-fixes.patch` | SHA-256 `954bed152b3b2a3dd6ea1342e66c43a561d421e06bdb3d8501940ebc2f55d0ff` |
| `source-inventory.sha256` (the first's tracked paths, hashed in the patched tree) | SHA-256 `a4bd2f8ac198a1487210cf936ba4a51f89c5361c41cea4d3c9e15f96107467b3` |

## The protocol

Fixed on 2026-10-08, before any migration run; the constants live in
[`tools/speed_gate.py`](../../../tools/speed_gate.py), the protocol's only
implementation, and change only with this report. It was written before
the same-binary runs below, which then set its block counts; two
choices changed while they ran, both from that evidence: Gemma2 and Gemma3
went from two measured passes a run to six (their first block's runs were
discarded and repeated), and the cohort decode rate became the per-request
chunk-to-chunk form below (recomputed from the recorded chunk times; no run
repeated). The review that followed, still before any migration run,
changed the verdict and the runs' hygiene, not what a run measures: each
look's bound became one-sided 97.5% (the stopping rule below), runs that
failed or recorded a problem make a comparison invalid instead of being
skipped, blocks past the maximum do not count, `verdict` reports through
its exit status, each arm may be given its own configuration, and a run
cannot leave its runtime behind or talk to another's.

**Host and conditions.** `spark` (GB10, driver 580.178.04), each run
under the installed GPU supervisor (`spark-job start --gpu`), no other GPU
job and no other heavy job on the node (the memory metric is whole-node).
A run records the GPU compute processes present before it starts; any
makes it a problem. Old and new binaries read the same installed artifacts
and inputs unless an arm is given its own configuration (below).

**A run** is one fresh `llmp-runtime` process with a fresh data
directory (no calibration record, no kept conversation, no spill: the
state, spill and checkpoint roles all resolve inside it), bound to
loopback. The runtime runs in the harness's process group, so a
supervisor's timeout or kill stops it too, and a run refuses to start
while anything listens on its port. For an LLM family the client drives
the chat route: one warm-up pass of every cell, then the family's
measured passes. A cell releases
its requests together at a barrier; each is a distinct prompt (a leading
request number, a fixed instruction and a passage of the pinned corpus
starting at a line), greedy (`temperature` 0), streamed, `max_tokens`
128, and must complete 128 tokens with zero cached prompt tokens. Every
run pays the same prompts in the same order. From the client's monotonic
clock, per cell:

| Metric | Definition |
| --- | --- |
| `prefill_s` | release to the last request's first streamed token |
| `decode_tok_s` | the sum of the requests' rates in the window where every request decodes (after the last first token, before the first last token), each measured between its own streamed chunks inside the window (a chunk is a step, worth its request's mean tokens a chunk), so a chunk landing either side of the window's edge does not move it. With one request: (tokens − 1) over its first-to-last chunk span |
| `wall_tok_s` | all completed tokens over release to the last request's end |
| `peak_gib` (a run) | the largest drop of whole-node `MemAvailable`, sampled every 20 ms from before the process starts until it has exited |

A run's value is the mean of its measured passes (each pass pays
different prompts, the same in every run). Qwen-Image's run is
one `chat --image-prompt` process (one generation a process is what this
build supports): `generation_s` is the turn's generation time from its
model being ready to its pixels (the report's `generation_seconds` minus
the activation's `ready`). The frozen old arm retains pixels `3b7770ca…`.
Prospective candidates may declare their own output digest with
`plan --new-image-pixels-sha256 SHA256` after reference quality qualification
and before timing. Each arm must reproduce its own pin throughout the
comparison; changing the pin within an arm invalidates the comparison.
Direct `run` uses `--image-pixels-sha256`. Without these options both arms
retain the historical pin, and old records retain their original meaning.
An old arm that is not the frozen baseline (since the v0 stores were retired,
a later commit built as the old arm) declares its own pin the same way,
`plan --old-image-pixels-sha256 SHA256`, which its priming runs carry too.
This checks repeatability within an implementation, not image quality or
bit equality with another implementation. Workload, timing and stopping rules
are unchanged.

**Families** (`speed_gate.py families`):

| Family | Configuration | Cells | Measured passes a run | Blocks, min–max |
| --- | --- | --- | --- | --- |
| `gemma2` | Gemma 2 2B Q8_0 `eb18d30d…`, context 8,192, 128-row chunks, 2 slots | C1, C2 | 6 | 5–10 |
| `gemma3` | Gemma 3 4B QAT `8c710341…`, context 4,096, 128-row chunks, 2 slots | C1, C2 | 6 | 7–12 |
| `gemma26` | Gemma 4 26B-A4B `4ddb360c…`, context 8,192, 1,024-row chunks, 4 slots | C1, C4 | 2 | 9–12 |
| `gemma31` | Gemma 4 31B `32c92e07…`, context 8,192, 256-row chunks, 4 slots | C1, C4 | 1 | 7–12 |
| `qwen38` | Qwen3.8 NVFP4/MXFP8 `c4fb47a9…` + MTP `8600a998…` (selected head), context 33,792, 4,096-row chunks, 4 slots, `depth_cost_ratio` 1.16 pinned | C1, C2, C4 | 2 | 6–12 |
| `qwen38-gguf` | Qwen3.8 UD-IQ3_XXS `5356b5b0…`, plain, context 33,792, 4,096-row chunks, 4 slots | C1, C2, C4 | 1 | 3–6 |
| `deepseek` | DeepSeek V4 Flash 0731 `8a355bfb…` + DSpark `dd2d3f9c…`, context 262,144, 4,096-row chunks, 4 slots, `wave_form = "plain"` | C1 (DSpark), C2, C4 (plain waves) | 1 | 3–6 |
| `image` | Qwen-Image-2.1 `eca21baa…`, 1,024², 40 steps, the pinned prompt and latents | generation | 1 a process | 6–12 |

Prompts are 26,000 corpus characters (Gemma3: 11,500; Qwen and DeepSeek:
30,000): 6,000–8,500 prompt tokens (Gemma3 2,400–3,600). Adaptive
policies whose in-run calibration could change behaviour between runs
are pinned at their fallbacks (Qwen's depth cost ratio, DeepSeek's wave
form); every other setting is the configuration above or its default.

Judgement calls in this table, not measured necessities: Gemma4's two
profiles are judged at C1 and C4, the cells their M3 qualification
measured, not C2; `gemma31`, `qwen38-gguf` and `deepseek` run one
measured pass a run (2.5–3 minutes a run), so each run's C1 pays a single
prompt; the `deepseek` family's cohorts are plain waves (defect 1 below),
and its DSpark cohort waves are measured with the same family and a
configuration template that sets `wave_form = "speculative"`, against
the fixed baseline ([DSpark cohort waves](#dspark-cohort-waves-on-the-fixed-baseline)).

**Each arm's configuration.** By default both arms get the family's
configuration above. A migrated runtime that needs other keys (the
per-family diagnostics may go, D-107's importer re-imports artifacts) gets
its own with `--config-template FILE` (a whole node configuration with
`{share}`, `{data}` and `{port}`; `plan --old-config/--new-config` passes
each arm its file). It must pay the same work: the same model name,
weights, context, chunk rows, slots and pinned policies. Every record
keeps its template's text and digest, and a comparison whose old or new
runs used more than one is invalid. The image family's command line
(`chat --image-prompt … --report`) and report fields are the frozen
build's; a runtime that changes them needs a harness change, recorded
here.

**A block** is four runs, old/new/new/old (ABBA), so linear drift
cancels. Its slowdown for a metric is ln(mean(new)/mean(old)) for a time
and the inverse for a rate, positive when new is slower; for `peak_gib`
it is the absolute difference. ABBA cancels only linear drift: a curved
drift or a position effect (the first run of a supervised job after an
idle GPU, say) falls on the old positions and biases the estimate
against the new binary. Split the steps into jobs at block boundaries.
*Added 2026-10-08, after the same-binary runs below:* every job (a
session: one `plan` output, or one of the step files `plan --job-runs K
--steps PREFIX` writes) starts with a priming run of the old arm that no
statistic reads but that must succeed before the session's runs, or the
family is INVALID; the old arm, so that whatever one arm's run leaves
cold for the other falls on the new arm's first run, never in its favour.
A session is one job: split a schedule with `--job-runs`, never by hand
(a hand-split file's later jobs would run unprimed, which the records
cannot show). `spread` reads runs with or without a session, the
same-binary records below among them, and counts the unprimed.

**The statistic and stopping rule.** Non-inferiority at a 1% margin: the
unit is the block, so over n blocks, with mean d and standard deviation
s of the block slowdowns, the bound is d ± t(0.975, n−1)·s/√n, a
one-sided 97.5% bound at each look.

- PASS: the upper bound is at most ln(1.01);
- FAIL: the lower bound exceeds ln(1.01);
- otherwise run another block, up to the family's maximum, after which
  the metric has NOT PASSED (the owner decides). Only the first
  maximum blocks count.

The rule looks after every block from the minimum on, so each look's
bound is stricter than the 95% the gate aims for: at 95% a look, a true
1% slowdown would pass about 8–11% of comparisons. `speed_gate.py operating`
simulates the rule for each family's block range (4,000 sequences, seed
1): a true 1% slowdown passes 4.0–6.0% of them at 97.5% a look, against
7.8–10.9% at 95%, and a true zero with the block sd each minimum was
chosen for passes 98.0–100% in 3.2–9.2 blocks on average.

No verdict is given before the family's minimum blocks, chosen below from
the measured same-binary spread so that a one-sided 95% bound's
half-width is expected to be at most 0.5%, half the gap judged (at the
97.5% look bound that half-width is 1.24–1.47 times larger). Peak memory
is reported as an informational paired absolute difference. The owner
(2026-10-13) superseded the former 0.5GiB rejection: more memory is allowed
for worthwhile speed or quality when the full workload safely fits the target.
`speed_gate.py verdict` retains peak estimates/bounds as `INFORMATIONAL`;
only speed metrics determine its statistical verdict. Admission, physical
headroom, completion and run validity remain required. Historical recorded
gates keep their original provenance. A family
passes when every speed metric of every cell passes and its runs are valid:
every block has its four runs, none failed or recorded a problem (a reply
short of 128 tokens, cached prompt tokens, a cell without a common decode
window, the image's pixels, another GPU process), and the old runs used
one runtime and one configuration, the new runs likewise, all with the
protocol's current per-run constants. Otherwise the family is INVALID
until the cause is fixed and the affected block rerun (a rerun replaces
its records).

**Commands.**

```sh
T=tools/speed_gate.py; OLD=~/.local/share/llmp/m36-baseline/1facc957/src/runtime/llmp-runtime
python3 -B $T plan --family F --old $OLD --new NEW --out DIR > F.steps   # min_blocks of ABBA runs
#   [--new-config NEW.toml]: the new arm's own configuration template
#   [--new-env LLMP_DIAGNOSTIC_X=1]: a diagnostic switch for the new arm's runtime (defect 3)
#   [--job-runs K --steps F]: F-01.steps, F-02.steps ..., K runs a job, each primed
spark-job start --gpu --name F-ab1 --timeout 600 --steps F.steps       # one job a steps file (<= 600 s)
python3 -B $T verdict DIR --family F                                   # PASS / FAIL / CONTINUE ...
python3 -B $T plan ... --blocks 1 --first N                            # one more block while CONTINUE
python3 -B $T quality --family F --runtime NEW --label new --out DIR   # and --runtime $OLD --label old
```

DeepSeek's DSpark cohort cells: `--family deepseek` with
`OLD=~/.local/share/llmp/m36-baseline/1facc957-fixed/src/runtime/llmp-runtime`
and `--old-config` the family's configuration with
`wave_form = "speculative"` (the new arm's template likewise, or its own).

`verdict` exits 0 only when every family it judges passes, 2 while one
needs more blocks (CONTINUE or INCOMPLETE) and 1 otherwise (FAIL, NOT
PASSED, INVALID, or no runs); a script must treat anything but 0 as not
passed.

`quality` serves the family once: two fixed greedy C1 replies (digests
only) and the mean negative log-likelihood of a fixed 4,000-character
corpus passage through the literal route's prompt scores. It is a quick
runtime-level control beside each family's own oracle controls, not a
replacement for them.

## Same-binary spread

The block counts rest on same-binary (A-vs-A) runs: the frozen runtime as
both "old" and "new", in the protocol's own schedule on `spark`,
2026-10-08. Three blocks (twelve runs) a family; two blocks (eight runs)
for `gemma31`, `qwen38-gguf` and `deepseek`, whose runs take 2.5–3
minutes. All 84 counted runs completed, every request with 128 tokens and
zero cached prompt tokens. The run sd is a run's log value about its
block's mean, pooled; for ABBA blocks of independent runs it is also a
block slowdown's expected sd, and min_blocks is the fewest blocks (at
least three) whose one-sided t bound at the larger of that and the
observed block sd is within 0.5% (`speed_gate.py spread`). Every block
mean is within 0.4% of zero, as it should be for one binary.

| Family | Cell | Prefill s (run sd) | Decode tok/s (run sd) | Wall tok/s (run sd) | Prompt tokens a request |
| --- | --- | ---: | ---: | ---: | --- |
| `gemma2` | C1 | 1.297 (0.51%) | 62.55 (0.18%) | 38.48 (0.24%) | 6,170–7,384 |
| `gemma2` | C2 | 1.944 (0.49%) | 104.7 (0.22%) | 58.72 (0.30%) | 6,018–7,730 |
| `gemma3` | C1 | 0.6163 (0.68%) | 74.93 (0.16%) | 55.36 (0.28%) | 2,830–3,374 |
| `gemma3` | C2 | 1.101 (0.37%) | 133.0 (0.18%) | 85.08 (0.14%) | 2,446–3,545 |
| `gemma26` | C1 | 2.113 (0.55%) | 48.23 (0.35%) | 26.97 (0.33%) | 6,635, 7,266 |
| `gemma26` | C4 | 8.921 (0.63%) | 103.2 (0.56%) | 37.80 (0.59%) | 6,036–7,433 |
| `gemma31` | C1 | 9.039 (0.63%) | 9.903 (0.06%) | 5.854 (0.28%) | 6,635 |
| `gemma31` | C4 | 44.85 (0.14%) | 30.88 (0.11%) | 8.535 (0.13%) | 6,036–6,830 |
| `qwen38` | C1 | 3.223 (0.15%) | 47.80 (0.45%) | 21.73 (0.23%) | 6,857, 7,417 |
| `qwen38` | C2 | 7.153 (0.22%) | 63.69 (0.31%) | 22.77 (0.23%) | 7,464–8,041 |
| `qwen38` | C4 | 15.81 (0.17%) | 76.93 (0.96%) | 23.33 (0.23%) | 6,910–8,478 |
| `qwen38-gguf` | C1 | 6.572 (0.18%) | 30.39 (0.15%) | 11.90 (0.12%) | 7,417 |
| `qwen38-gguf` | C2 | 13.96 (0.16%) | 41.78 (0.09%) | 12.79 (0.13%) | 7,467, 8,041 |
| `qwen38-gguf` | C4 | 28.68 (0.15%) | 63.10 (0.22%) | 13.87 (0.13%) | 6,910–8,478 |
| `deepseek` | C1 | 8.184 (0.09%) | 32.63 (0.22%) | 10.60 (0.06%) | 6,964 |
| `deepseek` | C2 | 17.77 (0.08%) | 37.53 (0.19%) | 10.35 (0.07%) | 7,112, 7,732 |
| `deepseek` | C4 | 35.33 (0.10%) | 49.62 (0.09%) | 11.27 (0.08%) | 6,614–8,026 |

Values are means over the runs; they are the baseline speed rows.
Prefill seconds pay the cell's whole prompt set (C4: four prompts
released together). `image` generation: 33.43 s (33.28–34.05 s, run sd
0.57%), pixels `3b7770ca…` in all twelve runs.

| Family | Blocks needed (the metric that sets it) | Peak whole-node `MemAvailable` drop, GiB (run sd; range) | Run time | One comparison at min_blocks |
| --- | --- | --- | ---: | ---: |
| `gemma2` | 5 (C1 and C2 prefill) | 5.913 (0.031; 0.093) | 57 s | 19 min |
| `gemma3` | 7 (C1 prefill, 0.62 s) | 4.637 (0.057; 0.222) | 40 s | 19 min |
| `gemma26` | 9 (C4 prefill) | 21.035 (0.041; 0.139) | 61 s | 37 min |
| `gemma31` | 7 (C1 prefill) | 27.972 (0.029; 0.078) | 178 s | 83 min |
| `qwen38` | 6 chosen; C4 decode needs 12 (below) | 82.601 (0.046; 0.219) | 131 s | 52 min |
| `qwen38-gguf` | 3 | 58.833 (0.075; 0.236) | 146 s | 29 min |
| `deepseek` | 3 | 109.944 (0.025; 0.075) | 181 s | 36 min |
| `image` | 6 (generation) | 33.483 (0.031; 0.118) | 38 s | 15 min |

Every metric resolves 0.5% within nine blocks except **Qwen3.8's C4
decode rate** (run sd 0.96%), which needs twelve, the cap; at six blocks
its 95% half-width is about 0.79% (the 97.5% look bound's, about 1.0%).
Its replies repeat exactly across runs;
what varies is the step rate of the four-slot speculative waves while all
four are active (pass means 72.3–83.1 tok/s for the same prompts), which
depends on how the waves meet. A judgement call, made by the main agent
under the owner's delegated threshold authority (2026-10-08): judge it with the rest of the family from six blocks and continue
to twelve while it is unresolved, reading it beside the C4 prefill and
wall rates (run sd 0.17% and 0.23%), rather than raising the family's
minimum to twelve. Simulated at its sd, a true zero then passes this
metric in 93% of comparisons, in 7.5 blocks on average, and is NOT PASSED
in the rest. The 1% speed gate itself is unchanged. The historical peak-memory margin
of 0.5GiB was at least six times every family's run sd; it is informational
under the owner's 2026-10-13 memory clarification above.

Same-binary determinism: every C1 reply repeats byte for byte across all
runs of a family, and so does every cohort reply of DeepSeek, both
Qwen3.8 artifacts and both Gemma4 profiles. Gemma2's and Gemma3's C2
replies have two variants in 13 and 4 of their 21 request positions:
arrival-dependent joined prefill and decode, as their bounded routes
record. A migrated family's quality is judged by its controls below, not
by these digests.

### DSpark cohort waves on the fixed baseline

DeepSeek's C2 and C4 with DSpark waves: the `deepseek` family (its
prompts, passes and constants) with `--old-config`/`--new-config` set to
the family's configuration plus `wave_form = "speculative"` (template
SHA-256 `d4ecc40b…` as the records keep it), the fixed runtime as both
arms, three ABBA blocks (twelve runs) on `spark`, 2026-10-08, in jobs of
two runs. All twelve runs completed with no problem recorded (every request
128 tokens, no cached prompt tokens).

| Cell | Prefill s (run sd) | Decode tok/s (run sd) | Wall tok/s (run sd) |
| --- | ---: | ---: | ---: |
| C1 (DSpark, as the family's) | 8.194 (0.14%) | 32.54 (0.57%) | 10.58 (0.14%) |
| C2 (DSpark waves) | 17.78 (0.14%) | 45.02 (0.22%) | 11.04 (0.14%) |
| C4 (DSpark waves) | 36.40 (0.18%) | 53.01 (1.47%) | 11.50 (0.62%) |

Peak whole-node `MemAvailable` drop 110.005 GiB (run sd 0.121; range
0.436). Blocks needed (`spread`): three for every prefill, C2 and wall
metric except C4 wall (8) and C1 decode (7); **C4 decode** does not
resolve 0.5% within twelve blocks at its measured sd. Its spread is one
run: the first run of the first job (block 1's old, the fixed build's
first run, ten minutes after its build and install) decoded 50.83 tok/s,
the other eleven 52.92–53.81 tok/s (pooled run sd 0.45% without it, which
`blocks_needed` puts at five blocks; computed from the same records); the
same run has the lowest C1 decode (31.96). Its runtime log shows it slower
before the measured pass too: page-in 9.04 s against 8.09–8.12 s in the
other eleven, graph captures 2.15 s against 1.15–1.27 s; and it stayed
slower through its own warm-up pass (about two minutes of GPU work), so a
cold GPU alone does not explain it. One run cannot tell the idle GPU,
the fresh build and install, or something else apart. With the
cause not established the harness gets no extra warm-up step or discard
rule (each run already has its warm-up pass). A comparison of these cells should
split its jobs at block boundaries as the protocol says and read C4
decode beside C4 prefill and wall, as Qwen3.8's C4 decode is read; the
block counts are the main agent's call.

Replies: every DSpark reply repeats byte for byte across the twelve runs,
and C1's equal the plain-wave family's. The four-slot DSpark replies equal
two-slot DSpark replies: one run of the same template at `max_slots = 2`
on the *first* baseline (where two-slot DSpark waves worked) and one on the
fixed baseline gave identical digests for all 14 prompts, and the same
digests as the four-slot runs. DSpark's fast verify is not the plain wave's
arithmetic: its C2 and C4 replies differ from the plain-wave family's in
4 of 4 and 7 of 8 prompts (DSpark's bit-identity to plain is exact mode's,
[dspark](../dspark/README.md)), the same in both builds.

## The Gemma4 31B assistant

Assistants are harness-only in this build (the runtime refuses Gemma4
speculation). Their speed baseline is the existing
[repeated greedy screen](../gemma-assistant-throughput/README.md) rerun on
the frozen binary: `llmp_gemma_greedy_reference` from the frozen install,
the 31B target `32c92e07…`, its assistant `447a5c20…`, the canonical P=64
query inputs (`inputs.i32`, SHA-256 `b2d7aaf6…`) and mode `unit32`: 32
committed greedy tokens a timed round, two timed rounds a process, three
ABBA blocks of processes on `spark`. Every round verifies 46 target rows,
drafts 34 and commits 32, as in the committed screen.

Mean 1.3880 s for 32 committed tokens (23.05 tok/s), 1.3720–1.3971 s a
process; run sd 0.74%, so eight blocks reach a 0.5% half-width. The
three blocks' "new" positions measured +0.44%, +0.65% and +0.73% over
their "old" bookends with one binary, the only same-binary set with a
consistent sign; a comparison on this harness should run its eight
blocks and treat an estimate under about 0.7% as within its noise. The
committed screen measured 1.315 s on 2026-10-06 with an earlier build;
this run does not attribute the difference. The harness is
model-specific and goes with Gemma4's runner; its migrated counterpart
must pay the same work (target, assistant, inputs, 32 committed tokens at
P=64) to be compared with this number.

## Quality controls

**Prospective addendum, owner 2026-10-12, clarified 2026-10-13.** A reference
implementation of the same quantized checkpoint is the primary comparison.
Original unquantized evidence is useful when available and required for a
closer-to-original claim, not for same-quantized-model qualification.
The tables and historical verdicts
below are frozen observations, including their exact claims. Changed bits
alone do not fail a prospective candidate or require unfusing, extra rounding
or owner serialization. Keep the same teacher IDs/history and full-vocabulary
NLL/perplexity, fix reference provenance, coverage and acceptance bounds before
qualification, and retain existing behavioral controls. Old tolerance budgets
do not automatically apply to a new oracle. Record missing reference
evidence honestly; no new quality pass is asserted here. Exact
same-implementation continuation/restore, source integrity, selected-expert
execution and the matched 1% timing protocol remain unchanged. See the
[audit and recovery protocol](../m36-numerical-audit/README.md).

**Runtime-level control** (`speed_gate.py quality`): each LLM family
served twice by the frozen runtime. Both processes gave identical scores
and identical greedy replies, so a migrated family is compared with these
values: where its op order is unchanged the same scores and replies are
expected; any other difference is investigated, and the family's oracle
controls below decide its quality.

| Family | Scored tokens | Mean NLL (nats) | Scores SHA-256 |
| --- | ---: | ---: | --- |
| `gemma2` | 1,070 | 3.5635004927 | `cf15e993…` |
| `gemma3` | 1,070 | 3.6671293402 | `ab21ad8d…` |
| `gemma26` | 1,070 | 9.7916674254 | `3d69b78c…` |
| `gemma31` | 1,070 | 9.5749963928 | `12c22f37…` |
| `qwen38` | 996 | 2.7655865453 | `b7ed7b14…` |
| `qwen38-gguf` | 996 | 2.8538864116 | `f5ce8833…` |
| `deepseek` | 966 | 2.7939955482 | `6adb9558…` |

The passage is raw corpus markdown scored without a chat template; the
Gemma4 instruct models' high values are in line with their own raw-corpus
controls (10.60 and 11.19 nats below).

**Each family's own controls against its same-format oracle** are the
gate's quality evidence; their latest committed values are the baseline.
They run in per-family harnesses (the frozen install keeps every one),
which go with the per-family code, so each migration ports its family's
controls to the shared engine and must meet these values.

| Family | Oracle | Controls and latest committed values |
| --- | --- | --- |
| Gemma2 2B | llama.cpp v0.6.0 (`d812350`) | C1 teacher heads 33/33 byte-exact, mean NLL 3.72364433815 in both ([gemma2-execution](../gemma2-execution/README.md)); 256/768-prefix serving screen 76/76 exact heads ([gemma2-serving](../gemma2-serving/README.md)); prefill without cache copies: exact choices and final heads ([gemma-prefill-copies](../gemma-prefill-copies/README.md)) |
| Gemma3 4B QAT | llama.cpp v0.6.0 | checked norm chains 33/33 bounded heads exact; joint prefill 216/216 strict transitions; C2 decode zero strict differences, 64/66 exact heads; trained-maximum screen 65/65 heads ([gemma3-execution](../gemma3-execution/README.md)) |
| Gemma4 26B-A4B | llama.cpp v0.6.0 | 129/129 tokens at C1 and C4; 1,024-row corpus heads byte-identical, mean NLL 10.60303447 ([gemma26-production](../gemma26-production/README.md)) |
| Gemma4 31B | llama.cpp v0.6.0 | C1/C4 8K zero predicted-ID differences, 128/129 and 512/516 exact heads; 1,024-position corpus heads all exact, NLL 11.1856652775 ([gemma31-serving-bridge](../gemma31-serving-bridge/README.md)); assistant transaction heads and features byte-exact ([gemma-assistant-greedy-reference](../gemma-assistant-greedy-reference/README.md)) |
| Qwen3.8 NVFP4/MXFP8 | Mia's vLLM, same checkpoint | short prompts 180/192 equal, 11 near-ties, one owner-accepted step outside; 32K 477/35/0 and 128K 486/26/0 at the 1.765 bound, PPL −1.2% / −1.4% ([long-context](../long-context/README.md)); four-slot oracle, 8 cells of 477/35/0 ([qwen38-concurrent-oracle](../qwen38-concurrent-oracle/README.md)); MTP forced rejections 0/86 states differ, sampled TV within 0.1 ([qwen38-mtp](../qwen38-mtp/README.md)) |
| Qwen3.8 GGUF | llama.cpp b11254, same GGUF | teacher-forced 188/192 equal, 4 near-ties; PPL −0.23% against the unfused arm; expert layout bit for bit ([qwen38-gguf](../qwen38-gguf/README.md)) |
| DeepSeek V4 Flash | llama.cpp, same GGUF | exact mode 256/256 bit-identical; default-on output-A/HCA: 32K 491/19/2 under the tie-aware rule, 128K 499/13/0, 128K PPL 1.928842, the 127K answer task passes ([ds4-output-prefix](../ds4-output-prefix/README.md)); DSpark exact-mode speculation bit-identical to plain, sampled TV within its bound ([dspark](../dspark/README.md)) |
| Qwen-Image-2.1 | diffusers BF16 | 39/39 tokens; 40-step DiT worst 0.00937 (bound 0.046); image PSNR 41.770 dB, SSIM 0.99604; pixels `3b7770ca…` ([qwen-image-native](../qwen-image-native/README.md)), reproduced by every image run above |
| Qwen2.5-0.5B FP16 / EXL3 fixtures | the FP16 bridge / ExLlamaV3 `6b84a21b` | FP16: all four arms bit-identical to the bridge ([P2](../backend-proof-p2/README.md)); EXL3: 1,570/1,570 linears exact, Tier C's 750 statistics pass ([P3](../backend-proof-p3/README.md)) |

The fixtures have no speed baseline: their harnesses run recorded
trajectories without a timer, they are not served, and resident fixture
speed is M5's (BP-F3). Their M3.6 gate is the correctness controls above
(a judgement call: the gate's speed clause is applied to served families
only).

## Swap rows

The swap code changed after the last complete 32-row table (2026-10-07 at
`9bc2270b`, in [vmm-batching](../vmm-batching/README.md): worst
LLM-to-LLM 8.532 s): `0de25d2e`, `46cfd5c7`, `16bb9a7d`, `05974c8a`,
`7520f126`, `cc70d69f` and `eede847d` touch the pager, state and swap
paths. So the table was rerun on the frozen binary, with the
[final M3 table](../m3-final-swap/README.md)'s configuration and inputs
(context text `6b159ff2…`, latents `eb733389…`, expected pixels
`3b7770ca…`), on `spark`.

**It does not complete.** 21 of 32 rows ran, every one exact (state,
continuation, graphs and image controls). The next, Qwen3.8's restored 8K
continuation (Qwen3.8 → DeepSeek → Qwen3.8, first use), stops the command
with "the swap continuation needs complete finite target-vocabulary rows"
([defects](#defects-in-the-baseline)); the other rows with Qwen3.8 as A
did not run.

| A / B | Row | Endpoint s | Evict s | Restore s | Page-in s | First output s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| deepseek / image | A→B, first use (8K) | 3.874 | 0.123 | 0.000 | 2.298 | 1.448 |
| deepseek / image | B→A, first use (8K) | 8.249 | 0.022 | 0.091 | 8.013 | 0.111 |
| deepseek / image | A→B, prepared (8K) | 3.745 | 0.094 | 0.000 | 2.297 | 1.349 |
| deepseek / image | B→A, prepared (8K) | 8.242 | 0.017 | 0.089 | 8.023 | 0.100 |
| deepseek / image | A→B, prepared (0) | 3.720 | 0.052 | 0.000 | 2.310 | 1.352 |
| deepseek / image | B→A, prepared (0) | 8.351 | 0.017 | 0.000 | 8.100 | 0.223 |
| deepseek / qwen3.8 | A→B, first use (8K) | 6.221 | 0.119 | 0.000 | 5.764 | 0.326 |
| deepseek / qwen3.8 | B→A, first use (8K) | 8.340 | 0.078 | 0.094 | 8.045 | 0.111 |
| deepseek / qwen3.8 | A→B, prepared (8K) | 6.138 | 0.125 | 0.000 | 5.760 | 0.240 |
| deepseek / qwen3.8 | B→A, prepared (8K) | 8.313 | 0.064 | 0.100 | 8.035 | 0.101 |
| deepseek / qwen3.8 | A→B, prepared (0) | 6.102 | 0.081 | 0.000 | 5.769 | 0.240 |
| deepseek / qwen3.8 | B→A, prepared (0) | 8.365 | 0.051 | 0.000 | 8.069 | 0.233 |
| image / deepseek | A→B, first use | 8.364 | 0.042 | 0.000 | 8.084 | 0.227 |
| image / deepseek | B→A, first use | 3.754 | 0.080 | 0.000 | 2.307 | 1.362 |
| image / deepseek | A→B, prepared | 8.327 | 0.037 | 0.000 | 8.069 | 0.208 |
| image / deepseek | B→A, prepared | 3.704 | 0.050 | 0.000 | 2.304 | 1.345 |
| image / qwen3.8 | A→B, first use | 6.015 | 0.018 | 0.000 | 5.789 | 0.198 |
| image / qwen3.8 | B→A, first use | 3.704 | 0.070 | 0.000 | 2.306 | 1.325 |
| image / qwen3.8 | A→B, prepared | 5.988 | 0.032 | 0.000 | 5.753 | 0.193 |
| image / qwen3.8 | B→A, prepared | 3.711 | 0.066 | 0.000 | 2.304 | 1.337 |
| qwen3.8 / deepseek | A→B, first use (8K) | 8.428 | 0.128 | 0.000 | 8.065 | 0.224 |

Worst completed LLM-to-LLM row 8.428 s, worst prepared 8.365 s, all within
the 20 s / 40 s bounds; the largest per-row tracked peak is 111.8 GiB.

**The complete table on the fixed baseline** (2026-10-08, `spark`): the
same configuration, inputs and command with `--pairs` splitting it into
two runs of at most ten minutes (the Qwen3.8-as-A pairs, then the other
four). All 32 rows are exact (state, continuation, graphs, image
controls). The 21 rows the first baseline completed are within 0.066 s of
its endpoints; the 11 it could not run, and its last row again:

| A / B | Row | Endpoint s | Evict s | Restore s | Page-in s | First output s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| qwen3.8 / deepseek | A→B, first use (8K) | 8.494 | 0.095 | 0.000 | 8.077 | 0.311 |
| qwen3.8 / deepseek | B→A, first use (8K) | 5.972 | 0.048 | 0.080 | 5.734 | 0.097 |
| qwen3.8 / deepseek | A→B, prepared (8K) | 8.390 | 0.099 | 0.000 | 8.072 | 0.207 |
| qwen3.8 / deepseek | B→A, prepared (8K) | 5.968 | 0.050 | 0.078 | 5.734 | 0.092 |
| qwen3.8 / deepseek | A→B, prepared (0) | 8.366 | 0.066 | 0.000 | 8.080 | 0.208 |
| qwen3.8 / deepseek | B→A, prepared (0) | 6.041 | 0.072 | 0.000 | 5.770 | 0.186 |
| qwen3.8 / image | A→B, first use (8K) | 3.753 | 0.096 | 0.000 | 2.301 | 1.353 |
| qwen3.8 / image | B→A, first use (8K) | 5.933 | 0.018 | 0.073 | 5.735 | 0.095 |
| qwen3.8 / image | A→B, prepared (8K) | 3.727 | 0.116 | 0.000 | 2.296 | 1.312 |
| qwen3.8 / image | B→A, prepared (8K) | 5.934 | 0.023 | 0.073 | 5.739 | 0.088 |
| qwen3.8 / image | A→B, prepared (0) | 3.709 | 0.057 | 0.000 | 2.307 | 1.341 |
| qwen3.8 / image | B→A, prepared (0) | 5.929 | 0.025 | 0.000 | 5.753 | 0.141 |

Worst LLM-to-LLM row 8.494 s (Qwen3.8 → DeepSeek, first use), worst
prepared 8.390 s, within the 20 s / 40 s bounds; the largest per-row
tracked peak 111.2 GiB. These 12 rows are the fixed baseline's; the 20
other rows stay the first baseline's above.

**Gemma near-8K switch controls**, rerun on the frozen binary on `spark`
with the configurations, input (`4e41c6aa…`) and command of the
[near-8K report](../gemma-near8k-swaps/README.md) and the
[Gemma3 public 8K pair](../gemma3-execution/README.md#public-scalar-8k-and-one-model-switch-pair-2026-10-07):
all 24 rows pass, every saved-state return with its snapshot hash and all
64 continuation tokens and heads exact, prepared returns keeping and
replaying graphs.

| Pair with Qwen3.8 | Saved tokens | Worst first use s | Worst prepared s | Prepared graphs kept / replays | Largest row `peak_bytes` |
| --- | ---: | ---: | ---: | --- | ---: |
| Gemma2 2B | 8,063 | 5.989764 | 5.890972 | 33 / 63 | 82,078,470,144 |
| Gemma3 4B QAT (context 8,448) | 8,192 | 5.973466 | 5.851851 | 34 / 63 | 81,909,743,616 |
| Gemma4 26B-A4B | 8,063 | 5.996112 | 5.898778 | 1 / 63 | 81,837,359,104 |
| Gemma4 31B | 8,063 | 6.087986 | 5.998251 | 1 / 63 | 82,927,976,448 |

## Rows on the fixed baseline

The rows the defects below kept from the first baseline are now the
fixed baseline's (`1facc957-fixed`):

- DeepSeek with DSpark cohort waves, C2 and C4 (two- and four-request):
  [above](#dspark-cohort-waves-on-the-fixed-baseline). The `deepseek`
  family's plain-wave cohorts stay the first baseline's.
- The swap table's 11 rows with Qwen3.8 as A, and its first row again:
  [above](#swap-rows).

The fixes change no plan or check that passed before: the graph-reader
allowance only admits more, and the swap-table check now compares rows
with the model's logit width. The start's memory guard counts the
allowance, so the host memory it sets apart beside the plans grows by a
quarter of the measured index (6.5 MiB at DeepSeek's four-slot 27.2 MB). Every other row stays the first baseline's;
the fixed build's C1 DeepSeek replies, its two-slot DSpark replies and its
20 other swap rows (exact, endpoints within 0.066 s) agree with the first
build's. No row is pending.

## Defects in the baseline

Two failures of the frozen build (`main` at `1facc957`), found here and
fixed in their own change, whose diff is the fixed baseline's patch
(causes and fixes after the list):

1. **DeepSeek DSpark waves of four 8K requests stop the runtime.** With
   `wave_form = "speculative"`, and with the default `"auto"` (whose
   exploration runs speculative waves), the first four-request wave of
   ~7K-token prompts fails with "graph reader index exceeds the startup
   scratch allowance" (`src/engine/planned.cc:126`) and the service stops
   after the failure; at context 65,536 too. Two-request speculative
   waves and four-request plain waves work. The `deepseek` family
   therefore baselines plain cohort waves, and DSpark cohort waves have no
   baseline. The allowance is the startup measurement's largest
   graph-reader index; `16bb9a7d` (startup placement skipped) and
   `03676661` (the index) are the commits that touch it.
2. **`swap-table` cannot continue Qwen3.8 after a saved-state swap.** The
   continuation check `last.size() != l.tokenizer().size()`
   (`src/runtime/serve_commands.cc:795`, from `46cfd5c7`) compares the
   target's logit row, 248,320 wide for Qwen3.8, with its tokenizer's
   248,077 entries, so every Qwen3.8-as-A pair fails, with MTP and plain
   (reproduced alone with `--pairs qwen3.8:image`). DeepSeek, Gemma and
   the image pass. The serving route does not use this check.

- **1, the cause:** the start measures DeepSeek's widest waves, every
  slot one row and every slot `share` rows (4 a slot at four slots), and
  admitted no margin over the largest graph-reader index they need. A
  wave's graph nodes do not grow with its rows: the fused query head
  (`Dsv4QHead`, one node a layer) needs 16 rows, while past 8 rows the
  float products split into slot groups. A four-slot verify wave of 9 to
  15 rows (slots verify different row counts) has both, 43 nodes (one a
  layer) more than the 16-row wave: 27,324,416 index bytes against the
  27,236,352 measured. Two-slot waves stay within 8 rows at the default
  three drafts. **The fix**
  (`engine/planned.h` `GraphIndexAllowance`): runtime plans get the
  largest measured index and a quarter more, as the activations and pool
  do; a four-slot DSpark test at context 65,536 (`FourSlotDsparkGpu`)
  failed before it and passes. The review counted the index of every
  reachable DSpark wave (joined slots of 2 to the verify's rows, every
  slot count to eight, three position patterns, verifies of 4 and 8 rows) and of prefill, injected
  and verify chunks from 1 to 4,096 rows at 13 starts, built on the CPU
  with a binding typed as the UD-Q2_K_XL GGUF, against the start's probes
  (its four-slot measurement, 27,228,160 bytes, within 0.03% of the
  runtime's): the worst wave is 0.32% over (four slots of 3, 2, 2, 2
  rows: the case above; at two slots, a `draft_rows` of 7, past the
  drafter's block of five that the settings allow, would reach 0.51%); the
  worst chunk 3.51% over the chunk probes, a 9-row injected chunk, which
  matters to a one-slot DeepSeek with DSpark (no waves measured), where
  the frozen build would refuse it (counted, not run). The quarter covers
  them about seven times over; the probes do not cover each row band, so
  the margin, not the probes, is the bound.
- **2, the fix:** `Llm::logit_width()` (each family's output vocabulary)
  replaces the tokenizer's size in the continuation check
  (`CompleteContinuationRows`). Nothing masks logit rows past the
  tokenizer: greedy and sampling choose over the whole row, as before
  ([RE-050](../../rough-edges.md#re-050-qwen38s-output-head-is-wider-than-its-tokenizerjson)).

A third defect, found by the Gemma 2 and Gemma 3 migration's gate
(2026-10-09), is not fixed in the baseline (its classes are deleted after
that gate):

3. **The M3 Gemma 2 and Gemma 3 classes never save a turn checkpoint past
   about 1K tokens.** `Gemma2Runner::CheckpointRanges`
   (`src/engine/gemma2_runner.cc:322`) and `Gemma3Runner::CheckpointRanges`
   (`src/engine/gemma3_runner.cc:326`) return one range a K/V tensor over
   all its used cells (8 MiB at a 4K window), and `CheckpointFile::Capture`
   refuses any range over one 2 MiB page (`src/engine/checkpoint_file.cc:86`,
   "invalid checkpoint page range"), so the save is skipped with a log
   line and the turn goes on. The baseline's `gemma2` and `gemma3` prefill
   therefore does less work than a runtime that saves the checkpoint (the
   component engine copies 177–274 MiB to the host and writes it before the
   prompt's last chunk, 28–51 ms). The gate compares equal work: those
   families' new arm runs with the gate-only diagnostic
   `LLMP_DIAGNOSTIC_SKIP_TURN_CHECKPOINTS=1` (`plan --new-env`), which
   skips the save at the point where the baseline's failed (after the
   branch settled and its ranges were taken). Moving the checkpoint's copy
   off the first token's path, for every family, is separate work.

## Gemma 2 and Gemma 3 on the shared engine (slice G1c)

The first family through the gate: the new arm is the component engine
(Gemma 2 and Gemma 3 served by `EngineLlm`), the old arm the frozen
binary's M3 classes, both with the checkpoint-equal-work switch (defect 3:
the new arm ran with `LLMP_DIAGNOSTIC_SKIP_TURN_CHECKPOINTS=1`).

**First candidate (`g1c-4bafb4f2`): Gemma 2 FAIL, Gemma 3 PASS only at 12
blocks.** Gemma 2 failed on C2 prefill, +1.45% (lower bound +1.24%), with
C2 wall not passed at +0.92%; Gemma 3 passed only at 12 blocks, its C2
prefill +0.69%. The cause was host cost, not GPU time (the joined graphs
took 32.59 against 32.72 ms each): the component layouts packed the state
tensors at 256 B instead of one 2 MiB extent each, so a prompt's state
crossed an extent boundary every one or two chunks and `LiveState` grew
(about 1 to 2.5 ms each time), and the unequal last chunks of a joined
prompt were not built ahead. The fix: each state tensor starts on its own
paged extent (`components::StageLayout` with `kExtent`, as the M3 layouts
did), and the lookahead builds the unequal last chunks
([EngineLlmTest.UnequalLastChunksAreBuiltAhead](../../../tests/unit/engine_llm_test.cc)).

**Passing candidate (`g1c-cd4726d1`).** Slowdown of the new arm against the
old, with the upper bound of its interval (the limit is ln 1.01, +0.995%); a
negative number is faster.

Gemma 2, PASS at 5 blocks (peak memory +0.008 GiB):

| Metric | Slowdown | Upper bound |
|---|---|---|
| C1 decode | -0.14% | -0.03% |
| C1 prefill | +0.05% | +0.90% |
| C1 wall | -0.05% | +0.26% |
| C2 decode | +0.13% | +0.18% |
| C2 prefill | -0.07% | +0.23% |
| C2 wall | +0.12% | +0.25% |

Gemma 3, PASS at 7 blocks (peak memory +0.019 GiB):

| Metric | Slowdown | Upper bound |
|---|---|---|
| C1 decode | -0.22% | -0.06% |
| C1 prefill | -0.44% | +0.07% |
| C1 wall | -0.29% | -0.07% |
| C2 decode | +0.41% | +0.73% |
| C2 prefill | -0.73% | -0.09% |
| C2 wall | +0.14% | +0.54% |

The quality controls are identical to the baseline's: Gemma 2's NLL is
3.5635004927 and Gemma 3's 3.6671293402, with the same scores and reply
digests. Raw records stay outside Git on `spark`:
`~/scratch/m36-gate/g1c-4bafb4f2` and `~/scratch/m36-gate/g1c-cd4726d1`
(and `-quality`). The classes the baseline compares with were deleted with
this gate's slice (G1d); the frozen binary keeps them.

## Gemma 4 26B-A4B and 31B on the shared engine (slice G2c)

The second family through the gate: the new arm is the component engine
(Gemma 4 served by `EngineLlm`, its steps the `gemma4` mapping's recipe for
the serving envelope), the old arm the fixed baseline's M3 class
(`1facc957-fixed`). The M3 Gemma 4 class has defect 3 too (its turn
checkpoints past about 1K tokens fail with "invalid checkpoint page range",
seen in an old-arm run's log), so the new arm ran with
`LLMP_DIAGNOSTIC_SKIP_TURN_CHECKPOINTS=1`, as Gemma 2's and Gemma 3's did.
The baseline build refuses Gemma speculation, so the families judge plain
serving at the cohorts the baseline recorded (C1 and C4, context 8,192,
four slots); the assistant is judged on the [31B screen](#the-gemma4-31b-assistant)
the baseline recorded, and the 26B-A4B's assistant, which has no baseline
(that screen runs the 31B alone), is served and controlled but not gated.

**Candidate `g2c-c9f7f2db`** (worktree on `main` at `4a6710ce`, tree
`c9f7f2db…`, built with `spark-native --locked` on `spark` and installed
read-only as `g1c-cd4726d1` was: the runtime SHA-256 `198cbae4…`, the
engine screen `llmp_gemma_engine_greedy` `1a1c59d2…`). Jobs of one block
for `gemma26` (a priming run and four), of half a block for `gemma31` (a
priming run and two: its runs take three minutes), each under ten minutes.

Slowdown of the new arm against the old, with the upper bound of its
interval (the limit is ln 1.01, +0.995%); a negative number is faster.

Gemma 4 26B-A4B, PASS at 9 blocks (peak memory −0.204 GiB):

| Metric | Slowdown | Upper bound |
|---|---|---|
| C1 decode | +0.22% | +0.48% |
| C1 prefill | +0.02% | +0.21% |
| C1 wall | +0.13% | +0.28% |
| C4 decode | +0.41% | +0.70% |
| C4 prefill | +0.11% | +0.42% |
| C4 wall | +0.28% | +0.58% |

Gemma 4 31B, PASS at 7 blocks (peak memory −0.083 GiB):

| Metric | Slowdown | Upper bound |
|---|---|---|
| C1 decode | +0.05% | +0.12% |
| C1 prefill | −0.18% | +0.14% |
| C1 wall | −0.05% | +0.11% |
| C4 decode | +0.26% | +0.38% |
| C4 prefill | +0.07% | +0.22% |
| C4 wall | +0.10% | +0.18% |

The decode rates' small cost (C4's +0.26–0.41%) was not measured further;
its likely sources are host time between steps: the engine plans a new
shape (a decode crossing a 256-cell read boundary, a prompt's last chunk)
in about 11 ms against the M3 runner's 8.8 (Gemma 26B-A4B's 86 plans in
0.94 s against 0.76 s in one run), and admits a joined step's owners twice
(alone, then for the join). The spark-job queue admitted the jobs out of
order, so some blocks' halves (`gemma31`) ran apart in time; every block
has its four runs and none recorded a problem.

**The assistant screen.** `benchmarks/gemma_engine_greedy.cc`
(`llmp_gemma_engine_greedy`) pays the M3 screen's work on the shared
engine through serving's own build (`runtime/component_build.h`): the 31B
target and its assistant, C1 at context 4,096 and 128-row chunks, the
canonical P=64 inputs, the screen's arithmetic (its explicit norm-rope and
norm-add chains: a policy given in place of the mapping's), and 32
committed greedy tokens a round (units of min(3, remaining − 1) drafts
and a verify, a plain decode for the last token), its output lines the M3
screen's. Its trajectory is the screen's: 46 rows verified and 34 drafted
in 12 units, the same keeps; its tokens, every verify's heads, each unit's
last draft head and the final head are byte-identical to the frozen
screen's. Eight ABBA blocks of processes against the frozen
`llmp_gemma_greedy_reference`, each process's value the mean of its two
timed rounds, the block slowdown ln(new/old) under the protocol's one-sided
97.5% bound (a script outside the repository, as the screen is outside
`speed_gate.py`):

| Block | Old s | New s | Slowdown |
|---|---:|---:|---:|
| 1 | 1.37148 | 1.32143 | −3.72% |
| 2 | 1.39158 | 1.32427 | −4.96% |
| 3 | 1.39207 | 1.33598 | −4.11% |
| 4 | 1.39543 | 1.33739 | −4.25% |
| 5 | 1.39436 | 1.33242 | −4.54% |
| 6 | 1.39498 | 1.33137 | −4.67% |
| 7 | 1.39664 | 1.33581 | −4.45% |
| 8 | 1.39877 | 1.33624 | −4.57% |

PASS at 8 blocks: −4.41%, interval [−4.73%, −4.09%]: 32 committed tokens
in 1.331 s against 1.391 s (24.0 against 23.0 tok/s).

The engine's verify saves nothing: Gemma 4's caches truncate a rejected
row (a ring holds a chunk's rows beside its window, so no later query reads
it), where the M3 runner snapshot and restored every row it wrote.

**Controls.** The quality control's scores and greedy replies are the
baseline's for both profiles (26B-A4B NLL 9.7916674254, scores
`3d69b78c…`; 31B NLL 9.5749963928, scores `12c22f37…`; the same reply
digests). With the assistant served (four slots, context 8,192), the
scores are the same, the 31B's greedy replies the same, and the 26B-A4B's
differ (a verify's four-row arithmetic is not a one-row decode's; both are
greedy agreement with the target's choice); no verify was discarded (96
and 106 units). A check of that difference (four literal
prompts, 128 greedy tokens each, every generated token scored again by the
plain server's prompt log-probabilities): the served verify rows deviate
from the scorer's no more than plain one-row decodes do (26B-A4B mean
absolute log-probability difference 0.03–0.21 against 0.01–0.12, largest
5.0 against 7.0; 31B 0.0005–0.16 against 0.0004–0.16; the 31B's
literal replies differ too, on one prompt of the four), the first verify
after a prompt already differs (no earlier verify's rows exist then), and
at each first differing token the scorer ranks the two candidates in
either order: arithmetic, not a stale or wrong context. The near-8K
switch pairs with Qwen3.8 (the baseline's
command and input) pass: every row exact (state, continuation, heads),
prepared returns keeping one graph and replaying it 63 times; worst first
use 6.928 s (26B-A4B; that row's Qwen3.8 page-in ran at 11.6 GB/s) and
6.112 s (31B), worst prepared 6.002 s and 5.972 s, largest row
`peak_bytes` 81,816,698,880 and 82,429,718,528. A conversation the frozen
build kept (record version 1) is refused at the candidate's start ("it is
of format version 1; this model keeps version 2"), its next turn served
fresh. The served controls of `gemma4_serving_gpu_test` (spill, restore,
kept restart, cross-variant refusal, capacity refusal, sampling, scoring,
switches) pass on the engine, and `component_runner_gemma4_gpu_test` holds
both profiles' and both assistants' steps bit for bit the M3 runner's
(drafts at a past of 256 among them).

**A verify's four rows.** `component_runner_gemma4_gpu_test` also holds the
engine's speculative verify against the M3 runner's `Wave` with every row's
outputs (cases `gemma26_verify` and `gemma31_verify`, context 4,096, two
owners): a verify step of four rows with four outputs (phase `kVerify`,
committed by `Accept`) alone three times (the shape's second and third
runs), joined over both owners (2 x 4 rows, eight outputs, twice), ragged
(2 and 4 rows), then a joined one-row decode over the verified state,
after a joined 37-row prompt. Every output of both profiles is bit for bit
the M3 runner's (13 owner outputs compared a case, none differing in a
bit; spark, 2026-10-10).

Raw records stay outside the repository on `spark`:
`~/scratch/m36-gate/g2c-c9f7f2db` (with `assistant31/`) and
`~/scratch/m36-gate/g2c-c9f7f2db-{quality,swaps,kept}`.

## The gate's tools

The final verification pass (M3.6's exit gate) runs on four committed tools, all
under [`tools/`](../../../tools/) with unit tests in `tools/tests/` (the workstation
runs those; the tools themselves run on a Spark, in jobs of at most 600 s). Each
header holds its full contract.

**Same work in both arms (`speed_gate.py verdict`).** Besides the validity rules
above, `verdict` compares every run's recorded workload with the first old run's:
per pass and cell, the prompt indices, the width and each request's
`prompt_tokens`. The arms read the same prompts, so a different token count is a
tokenizer difference (a re-imported v1 artifact against the v0 one, say) and the
arms did different work: the family is INVALID, with each mismatching cell
under `invalid` (and `verdict`'s exit status 1), never a silent pass. Records
without recorded passes (images, failed runs) are not compared; their own rules
apply.

**The 31B assistant screen (`assistant_gate.py`).**

    assistant_gate.py steps --out DIR --old BIN --new BIN \
        --old-artifacts STORE --new-artifacts STORE --target ID --assistant ID \
        --inputs inputs.i32 [--new-target ID --new-assistant ID] [--first B] [--blocks N] > FILE
    assistant_gate.py verdict DIR [--min 8] [--max 11] [--json]

`BIN` is `llmp_gemma_greedy_reference` (old, the frozen install) and
`llmp_gemma_engine_greedy` (new); each arm reads the target and assistant from
its own store (`STORE/ID`; `--new-target/--new-assistant` when a re-imported store
names them differently), so v0 and v1 artifacts are compared in one set of blocks.
`steps` prints a supervised job's steps (a priming old run, then ABBA blocks; two
blocks a job, `--first 1`, `--first 3`, ...; the baseline's eight-block
minimum is the default `--min`), and records `DIR/plan.json` (binaries'
and inputs' SHA-256, artifact paths). `verdict` is `speed_gate.py`'s rule on the
screen's metric (the mean of the two timed rounds' seconds for 32 committed
tokens): 1% margin, one-sided 97.5% bound at each look, only the first `--max`
blocks count. It is INVALID when a log is incomplete or missing, a priming log is
incomplete, or the retired line's workload identity (profile, mode, C, P, prefill
query, emitted, endpoint, warm/quality/timed rounds) differs between any two runs.
Exit 0 PASS, 2 CONTINUE, 1 otherwise.

**Quality pairing (`quality_pair.py`).**

    quality_pair.py pair --candidate FILE --reference FILE --prereg PREREG.json \
        [--report OUT.json] [--json]
    quality_pair.py extract --candidate FILE --out LOSSES.jsonl

Pairs the native literal route's per-target losses with retained reference losses,
keyed by (document, position) and the target token ID, against bounds fixed in a
pre-registration before the run. The candidate and reference are each a
`losses.jsonl` (`{"document","position","target","nll"}` a line, the form
[`collect-qwen-reference.py`](../m36-numerical-audit/collect-qwen-reference.py) writes
and `extract` writes from a native record, to retain one), a `speed_gate.py quality`
record (its sibling score-response file, hash-verified, holds every target ID and
loss), or a K1 arm record (`FAMILY-k1-ENCODING.json`: per-token scores and
`ids_sha256`; it keeps no IDs, so it pairs with a reference that has them, whose IDs
over those positions must hash to the recorded digest). The pairing refuses (exit 2)
different coverage, a target ID that differs at any position (a tokenizer or
history difference), non-finite or positive log-probabilities, duplicates, an
echoed token that disagrees with its scored target, a changed score response, a
failed record and a reference whose SHA-256 is not the pre-registered one.
`speed_gate.py quality` records of the frozen baseline predate per-target
retention (they keep the aggregate NLL and `scores_sha256` only), so a reference
for them is a rerun of the current `quality` command, or the Mia losses in the
[numerical audit](../m36-numerical-audit/qwen-quantized-reference.md).

The report gives each side's mean NLL and perplexity, the NLL delta (with its
paired standard error) and PPL delta in percent, per-document PPL deltas, the mean
and maximum absolute per-target difference and where the largest is, how many
targets are lower, higher and identical, the agreement (the fraction within a
tolerance), and each bound's outcome. Exit 0 every bound holds, 1 one fails.

    {"format": "llmp-quality-prereg/1", "name": "qwen3.8 v1 against Mia, 4 docs",
     "reference_sha256": "...",
     "bounds": {"expected_targets": 1016, "max_abs_ppl_delta_pct": 1.0,
                "max_abs_doc_ppl_delta_pct": 4.0, "max_mean_abs_diff": 0.05,
                "max_abs_diff": 2.0, "min_agreement": 0.9, "agreement_tolerance": 0.1}}

The bounds are `expected_targets`, `min_targets`, `max_ppl_increase_pct` (candidate
worse only), `max_abs_ppl_delta_pct`, `max_abs_nll_delta`,
`max_abs_doc_ppl_delta_pct`, `max_mean_abs_diff`, `max_abs_diff`, `min_agreement`
with its `agreement_tolerance`; at least one, an unknown key refused. The values
are the judgment of whoever registers them from the measured baselines, committed
before the run they govern (docs/workflow.md); the tool supplies none.

**The swap table (`swap_table.py`).**

    swap_table.py plan --case m3|gemma2|gemma3|gemma26|gemma31 --runtime R --store DIR \
        --out DIR [--ids FILE] [--image-pin SHA256] [--pairs-per-job 2] > STEPS
    swap_table.py run  --case CASE ... --name NAME [--pairs A:B,...]
    swap_table.py check DIR... --case CASE [--partial] [--json]

Replaces the baseline's `swaps.sh` and `swaps-fixed.sh`. `plan` prints the steps
of a spark-job (three runs of two ordered pairs for M3, each well inside 600 s;
one run for a Gemma case), `run` renders the node configuration (the M3 artifact
IDs by default; `--ids` overrides IDs, tokenizer files and extra model settings
for re-imported artifacts, `--config-template` replaces the whole file), verifies
the pinned inputs (context text `6b159ff2…` / Gemma `4e41c6aa…`, latents
`eb733389…`), runs `llmp-runtime swap-table` with the baseline's arguments and
keeps `table.out`, `table.log`, `report.json` and the runtime's SHA-256. `check`
reads the runs of one case together: all 32 rows (six rows for each LLM A, four
for the image A; six for a Gemma case) once each, every row exact and
state-exact, the report's `problems` empty, every restored 8K state hash equal to
the one A left with, the unswapped and continued logit hashes equal over the
report's 16 (M3) or 64 (Gemma) continuation tokens and the full target
vocabulary, every prepared LLM return to A having kept graphs (and with context
replayed them, capturing and running eagerly none), every image control equal to
the pin and present for an image A, prepared rows at most 20 s and first-use rows
at most 40 s, and one runtime, context text and length across the runs. It prints
each row's seconds and `peak_bytes`, the worst LLM-to-LLM row and the worst
prepared one (the table's headline), the largest peak and the lowest
`MemAvailable`. Exit 0 PASS, 1 FAIL, 2 INCOMPLETE (`--partial` with only rows
missing). Checked against the fixed baseline's `swaps-fixed/{qwenA,rest}` (32
rows, worst LLM-to-LLM 8.494 s, largest peak 111.19 GiB) and the four Gemma
reports under `swaps/`: all pass.

## Qwen-Image-2.1 on the shared engine

**`image-check02` is invalid.** Its records (`spark-b:~/scratch/m36-gate/image-check02`)
show the new arm run with `LLMP_DIAGNOSTIC_COMPONENT_SERVING=qwen_image_2_1`,
a switch no source reads, and both arms producing the legacy pixels
`3b7770ca…` in the same time (32.96 s): both ran the legacy pipeline, so
its PASS judged nothing, and the legacy image runner's deletion
(`22741330`) rested on no measurement of the shared path. The harness now
refuses a `--runtime-env` switch the runtime binary does not name (`plan`
when the binary exists, and every `run`).

**Default configuration.** After `22741330` an image composition served
only with the internal `LLMP_DIAGNOSTIC_*` startup tuple; from an ordinary
configuration it was refused ("image serving requires paid startup host
memory"). Without a diagnostic ceiling the image's parser loan and
construction ceiling now come from the node's measured memory
(`OrdinaryStartupBudget`, `runtime/memory_guard.h`). Every run below uses
the family's ordinary configuration and no diagnostic switch on either
arm. The new arm runs the shared engine: its pixels are `f4006496…`, not
the legacy `3b7770ca…`, and its log shows the composition's three shared
plans.

**Screens (2026-10-15, `spark`, two ABBA blocks each, primed).** Old arm the
frozen `1facc957` runtime (`28890a54…`), pixels `3b7770ca…` in every run;
new arm pinned to its own `f4006496…`, reproduced in every run.

| New runtime | Change | Generation, old / new (s) | Slowdown (upper bound) |
| --- | --- | --- | ---: |
| `e3e43dde…` | main at `7792be05` plus the default-configuration fix | 33.441 / 34.806 | +4.08% (+6.66%) |
| `2a6dac2d…` | a direct step with no wall history naps after its first ms | 33.375 / 34.523 | +3.44% (+4.21%) |
| `4c0ad76f…` | the image keeps its request across stages | 33.520 / 34.378 | +2.56% (+8.10%) |
| `86d0cc9c…` | each stage's graph digest computed once | 33.355 / 34.186 | +2.49% (+2.59%) |
| `ced1764c…` | the tiled F32 convolution rounds its prefetched patch after the products | 33.697 / 34.271 | +1.70% (+1.73%) |
| `cdb599f4…` | rebased on `d36922cb`; per-stream step history; refusals settle the lease | 33.539 / 34.348 | +2.41% (+3.35%) |
| `6ca53167…` | the same changes rebased on `093efe3c` | 33.384 / 34.100 | +2.14% (+2.57%) |

The later rows include main's changes since `7792be05`. A matched A/B on
the same session separates them: `093efe3c` itself (`f9829228…`, ordinary
configuration, the same pixels `f4006496…`) generates in 34.961 s, these
changes in 34.179 s, 2.26% faster (two ABBA blocks, −2.34% and −2.18%,
primed).

Attribution (nsys, both arms, run in both orders to cancel the GB10's
in-run warming; ordinary counters first found the gap):

- *Driver spin.* Each image stage is a request's only step, so the
  direct-step driver had no wall history and queried its fence back to back
  for the whole 0.8 s step: 34.2 million `cuEventQuery` calls against the
  legacy's 0.66 million, and every flow kernel 4–6% slower than the
  legacy's in that trace. The screened fix napped after a step's first
  millisecond (0.45 million queries); the landed form keeps each stream's
  step history across its requests, as the device lanes do, so only a
  stream's first step lacks one, and that spins 20 ms before backing off
  from 50 µs, doubling to 1 ms.
- *Host work between stages*, 12–15 ms each against the legacy's 0.7 ms:
  the lease cycled down to the common fence and back for every stage, and
  the stage's IR graph was hashed for its plan key every step (4.4 ms).
- *The VAE decoder*, about +0.34 s: the shared F32-activation 3×3
  convolution (`Conv3x3ImplicitF32Kernel`, 490–553 ms for 40 calls)
  against the legacy's BF16 one (226 ms), and the F32 channel norm. The
  convolution rounded each prefetched F32 patch to BF16 as it loaded it,
  stalling on the loads before the chunk's products; it now holds the F32
  pairs and rounds them when it stores the next stage (same bits, pixels
  unchanged).
- *The first step*, about +0.11 s (the text encoder's GGML path and the
  first flow step's plan build).
- *Flow-step compute* is at parity with the legacy's (same cuBLAS kernels
  and grids; layout transposes about +0.06 s over 40 steps).

Almost all of the shared path's speed comes from `BoundGraph::Run`'s
run-time fusion (`kernels/ggml/executor.cc`, `b37f763f`): with it disabled
the same generation takes 95.70 s (2.75× the shared path, 2.86× legacy). Its matchers, their dependencies and their reference tests are recorded in
[m36-runtime-fusion](../m36-runtime-fusion/README.md).

Every change above keeps the new arm's pixels `f4006496…`. Quality, on
the `86d0cc9c…` build (`ImageGenerationGpu.OrdinaryServerConsumesInstalledSourceAndRequests`
with `LLMP_IMAGE_SERVICE_UPSTREAM_QUALITY`, the same 1,024², 40-step
request; output `f4006496…`): PSNR 36.456 dB, luma SSIM 0.98944 against the
retained diffusers reference `9de442f1…` ([compare.py](../qwen-image-native/compare.py),
floors 32 dB / 0.98), passing.

The first planner patterns that replace run-time matchers (group 2a,
[m36-runtime-fusion](../m36-runtime-fusion/README.md)) change the pixels to
`2ee8f010…`: the text encoder's QK norms and rotations now take the fused
`llmp.image.rms_norm_mul_rope_sections` (the flow's and the decoder's
fusions are the same kernels, now selected by the plan). Quality on that
state (Spark A, 2026-10-15, the same test and request): PSNR 40.687 dB,
luma SSIM 0.99418 against `9de442f1…`, passing. Speed: a two-block screen
against the frozen legacy arm reads +2.03% (upper bound +2.30%; the
pre-2a screen07 read +2.14%), and an ABBA A/B against `b29380a7` (its
pixels `f4006496…`) reads −0.01% in both blocks (2a 36.292 s, `b29380a7`
36.296 s): no measurable change. Peak memory is unchanged (+1.59 GiB over
legacy, +1.63 GiB before).

Every run-time matcher is now a planner pattern and `BoundGraph::Run`'s
run-time fusion is deleted (sub-task 3,
[m36-runtime-fusion](../m36-runtime-fusion/README.md#the-run-time-path-deleted-sub-task-3)).
The image's pixels become `9d0b448f…` (the BF16 branch and residual rows
the shipping path already used, now planned): PSNR 41.26 dB, luma SSIM
0.99419 against `9de442f1…`, passing. A two-block screen against the
frozen legacy arm reads +1.42% (33.674 s against 33.203 s; +1.50% and
+1.34% by block), peak memory +0.81 GiB.

Raw records stay outside Git on
`spark`: `~/scratch/m36-image-screen0{1,2,3,4,5,6,7,8,9}`, `~/scratch/m36-image-imgab02`, `~/scratch/m36-image-2aab01`,
`~/scratch/m36-image-quality01`, `~/scratch/m36-image-plan/q2a`, `~/scratch/m36-image-plan/q4`,
`~/scratch/m36-image-nsys01` and `~/scratch/m36-image-nofuse01`.

Sub-task 5 closed the rest of the gap
([m36-runtime-fusion](../m36-runtime-fusion/README.md#the-speed-gap-sub-task-5)):
inputs unchanged within a generation uploaded and their prologue run
once, stage plans and recordings made at load, repeated and unread nodes
removed by the planner, the output norm joined with the last residual
and a one-read VAE channel norm. The pixels become `93356574…` (PSNR
41.26 dB, luma SSIM 0.99419). **Status: PASS** (Spark A, 2026-10-16, 6
primed ABBA blocks against `1facc957`): `generation_s` +0.22%, upper bound
+0.37% (legacy 33.792 s, new 33.867 s); peak memory +0.83 GiB. The model
is ready 0.24 s later after a first load (its recordings), so ready plus
generation reads +0.88%. Records: `~/scratch/m36-image-s5gate`. After
the review's fixes, on main `f05323d1`: PASS again, −0.36% (upper bound
−0.08%), peak +0.81 GiB, ready 2.758 s against 2.310 s
(`~/scratch/m36-image-r2gate`).

## Provenance

`spark` (`spark-c4e2`), NVIDIA GB10, driver 580.178.04 (CUDA 13.0 driver
API), the frozen binary above, SDK `aarch64-2c4eb5b2f68f3fa5` (CUDA 13.4,
its cuBLAS 13 in the install). Artifacts from the installed store
`~/.local/share/llmp/m3-artifacts` (IDs above); the Qwen3.8 GGUF artifact
`5356b5b0…` was copied from `spark-b`'s store by rsync (whole-file
checksums) to `~/.local/share/llmp/qgguf-artifacts` on `spark`. The
prompt corpus is `docs/decisions.md`, `docs/architecture.md` and
`docs/plan.md` at `1facc957`, concatenated (SHA-256 `55381941…`, pinned in
the harness), at `~/.local/share/llmp/m36-baselines/inputs/`. Raw run
records (timings, token counts and reply digests; no prompt or reply
text) stay outside Git under `~/.local/share/llmp/m36-baselines/` on
`spark` (`aa/`, `quality-aa/`, `assistant31/`, `swaps/`; for the fixed
baseline `aa-fixed-dspark/`, `dspark-2slot-frozen/`, `dspark-2slot-fixed/`
and `swaps-fixed/`) until M3.6 exits. The fixed baseline's source tree,
patch, DSpark template and swap script are under
`~/scratch/m36-fixed-baseline/` on `spark`.
Spark time for this record, build and pilots included: about three and a
half hours, in supervised jobs of at most ten minutes.
