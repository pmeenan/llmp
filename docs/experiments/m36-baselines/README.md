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
the activation's `ready`), and the pixels must equal the pinned
`3b7770ca…`.

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
prompt; DeepSeek's cohorts are plain waves (defect 1 below).

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
is judged the same way against an absolute margin of 0.5 GiB. A family
passes when every metric of every cell passes and its runs are valid:
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
spark-job start --gpu --name F-ab1 --timeout 600 --steps F.steps       # jobs of <= 600 s, whole blocks
python3 -B $T verdict DIR --family F                                   # PASS / FAIL / CONTINUE ...
python3 -B $T plan ... --blocks 1 --first N                            # one more block while CONTINUE
python3 -B $T quality --family F --runtime NEW --label new --out DIR   # and --runtime $OLD --label old
```

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
in the rest. The 1% gate itself is unchanged. The peak memory margin, 0.5 GiB, is at least six times every
family's run sd.

Same-binary determinism: every C1 reply repeats byte for byte across all
runs of a family, and so does every cohort reply of DeepSeek, both
Qwen3.8 artifacts and both Gemma4 profiles. Gemma2's and Gemma3's C2
replies have two variants in 13 and 4 of their 21 request positions:
arrival-dependent joined prefill and decode, as their bounded routes
record. A migrated family's quality is judged by its controls below, not
by these digests.

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
Until the defect is fixed, these 21 rows are the table's baseline, and the
complete table at `9bc2270b` (8.532 s) is the last full evidence.

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

## Pending rows

Until the defects below are fixed and the fixed build is frozen again as
the baseline, these rows have no baseline:

- DeepSeek with DSpark cohort waves: the `deepseek` family's C2 and C4
  run plain waves (`wave_form = "plain"` applies to every width, so the
  two-request DSpark waves that work are not baselined either); its C1
  speculates with DSpark.
- The swap table's 11 rows with Qwen3.8 as A: `qwen3.8 / deepseek` after
  its first row (5) and `qwen3.8 / image` (6). The 21 completed rows above
  stand.

A fix that touches a baselined path also reruns that family's rows here
on the rebuilt baseline.

## Defects in the baseline

Two failures of the frozen build (`main` at `1facc957`), found here and
not fixed here (a fix is its own change, and the rows it touches then
need a rebuilt baseline):

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
`spark` (`aa/`, `quality-aa/`, `assistant31/`, `swaps/`) until M3.6 exits.
Spark time for this record, build and pilots included: about three and a
half hours, in supervised jobs of at most ten minutes.
