<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# M3.6 final verification: the LLM families

The matched 1% speed gate, quality controls and peak memory of every LLM
family on the final M3.6 tree (`main` at `a32ea923`), against the fixed
baseline. The image family, the swap table and the Gemma near-8K cases are
not here (they wait for the last image slice). Protocol, constants and the
frozen baseline's numbers:
[m36-baselines](../m36-baselines/README.md); this report does not change them.

Status: **pre-registration written; no timed run has started.** Everything
under "Pre-registration" was fixed before any timed run and is not adjusted
after a result. Results are appended below it.

## Pre-registration

### Candidate

`main` at `a32ea923ca50473f4751a8f4e12d2804d01950e8` ("Reserve late state
regions on the backing lane"), tree `0e69f54e761abb22e3b273fb20cc9122acf9948f`,
synced to `spark-b:~/src/m36-final` (`rsync -rlpc`, no `.git`, no `build`),
built with `spark-native --locked` (SDK `aarch64-2c4eb5b2f68f3fa5`; 992 build
steps, 4 min), installed read-only as
`spark-b:~/.local/share/llmp/m36-candidates/final-0e69f54e` with the
baseline's layout:

| File | SHA-256 |
| --- | --- |
| `src/runtime/llmp-runtime` | `9f37fabbe09d8c1e4ef7816b4c529b8ee497b581009805bde182200d8574b665` |
| `benchmarks/llmp_gemma_engine_greedy` | `782a6344ae051bffae0b8f00b51d253f1eec0e65dda28f8686b9f61e2db29cc1` |
| `IDENTITY.sha256` (every installed file) | `17e5dffc54b3dcd2385c24acf31e8f905cb3025df1f58e442eb6d526a1ce69ba` |

The install's `gate/` holds `speed_gate.py` and `quality_pair.py` as at
`a32ea923`, `assistant_gate.py` with one fix (its retired-line pattern read
only the reference harness's `GREEDY_NATIVE_RETIRED`, while the engine
harness, the new arm, prints `GREEDY_ENGINE_RETIRED`, so every new-arm log
would have been INVALID; the pattern now takes both, with a test) and
[`score_docs.py`](score_docs.py) (below). Those two files are uncommitted
changes in the worktree that wrote this report.

### Arms

- **Old:** `~/.local/share/llmp/m36-baseline/1facc957-fixed` on `spark-b`
  (runtime SHA-256 `f5443363…`, `IDENTITY.sha256` `519bb045…`, checked with
  `sha256sum -c`), on the v0 artifacts of the family's template (`m3-artifacts`,
  `qgguf-artifacts`), no diagnostic switch. The fixed build is the baseline of
  every family here (the fixes touch only DSpark cohort graphs and `swap-table`),
  so the plain families' first-baseline rows carry over.
- **New:** the candidate runtime on the v1 imports in `m36-v1-artifacts`
  (the in-tree pinned map `tests/unit/data/v1_frozen/v1-ids.json`; the IDs in
  `speed_gate.py`'s `V1_MODELS` equal it for all nine), served with no
  diagnostic environment **except `LLMP_DIAGNOSTIC_SKIP_TURN_CHECKPOINTS=1` for
  `gemma2`, `gemma3`, `gemma26` and `gemma31`** (`plan --new-env`; baseline
  defect 3: the M3 Gemma classes never save a turn checkpoint past about 1K
  tokens; no other family has it). The candidate binary names that switch
  (`src/runtime/serve_api.cc`), which `plan` checks. The quality controls run
  the new arm without it.
- **Templates** (`speed_gate.py`'s defaults per arm, rendered under
  `spark-b:~/scratch/m36-gate/final/templates`; SHA-256 prefixes). Each new
  template validates under `llmp-runtime settings`; the new arm's settings
  equal the old arm's apart from the new `kv_cache_encoding "f16"` line (every
  family) and the absence of DeepSeek's `prefill_outa_hca`/`prefill_outa_hca_partial`
  lines (the candidate removed those settings):

| Family | Old | New |
| --- | --- | --- |
| gemma2 | `a87f8fba` | `8c0067e3` |
| gemma3 | `60906614` | `d031addd` |
| gemma26 | `b2fc0c44` | `927c9f59` |
| gemma31 | `d92d80bc` | `36bcca7d` |
| qwen38 | `d5d88c17` | `3fec10f5` |
| qwen38-gguf | `1bce0339` | `8159e866` |
| deepseek (plain waves) | `5c915e31` | `99b37230` |
| deepseek DSpark waves (`wave_form = "speculative"`) | `d4ecc40b` | `5cb621dd` |
| gemma26 / gemma31 with the assistant (new arm only, served controls) | | `43366c68` / `fe8a4cef` |

### Host, conditions, departures

- **Host `spark-b`** for everything, `--gpu` jobs of at most 600 s through the
  installed `spark-job`, nothing else running (`spark-job busy` free before every
  session). The protocol names `spark`; `spark` belongs to the image work for
  the duration, so this is a **recorded departure**. The old and new arm run on
  the same host, so the comparison stays matched; the absolute means can differ
  from the baseline's `spark` numbers by the two hosts' difference, which is
  why the screens (below) compare with the baseline's means only as a
  tripwire and the verdicts use each block's own old runs.
- Page cache: the build tree's and the SDK cache's clean pages are dropped
  (`POSIX_FADV_DONTNEED`, RE-057) before timing starts (buff/cache 44.8 to 25.9 GB,
  swap 337 MB used before and after).
- `plan --job-runs K`: every job is one session, primed by an old-arm run, as the
  protocol requires; no hand-split steps.

### Gates: blocks and jobs

`speed_gate.py plan --family F --old OLD --new NEW --old-config/--new-config`
only where the table names a template, `--steps`; `speed_gate.py verdict` after
every block; the family minimum and maximum are the protocol's (`families`).
While the verdict is CONTINUE one block is added (`plan --blocks 1 --first N+1`),
up to the maximum; blocks past it do not count. DeepSeek's two gates schedule
all six blocks upfront, since they pass only at about six. Jobs: K runs plus the
priming run, each under 600 s at the family's run time.

| Family | Blocks (min-max) | K runs a job (jobs a block) | Run s | GPU min at the scheduled blocks |
| --- | --- | --- | ---: | ---: |
| gemma2 | 5-10 | 8 (0.5) | 57 | 22 |
| gemma3 | 7-12 | 12 (0.33) | 40 | 21 |
| gemma26 | 9-12 | 4 (1) | 61 | 46 |
| gemma31 | 7-12 | 2 (2) | 178 | 125 |
| qwen38 (+ MTP) | 6-12, first look at 6 | 3 (1.33) | 131 | 70 |
| qwen38-gguf | 3-6 | 2 (2) | 146 | 44 |
| deepseek plain | 3-6, **all 6 upfront** | 2 (2) | 181 | 109 |
| deepseek DSpark waves | 3-6, **all 6 upfront**, `--old-config` d4ecc40b, `--new-config` 5cb621dd | 2 (2) | 181 | 109 |
| gemma31 assistant (`assistant_gate.py`) | 8-11 | 2 blocks a job (priming run first) | about 1.4 s a screen | about 40 |

The gemma31 jobs hold half a block (178 s a run, 600 s a job), as G2c's did; the
protocol's "split at block boundaries" cannot hold there. `qwen38` jobs are
three runs, so its blocks straddle jobs as well. Known risks, stated before the
runs, not reasons to adjust anything: DSpark's C4 decode had run sd 1.47%
(one slow first run; 0.45% without it) and cannot resolve 0.5% at its maximum of
six blocks if that sd recurs, so it may end NOT PASSED; Qwen's C4 decode (sd about
1%) may need the full twelve; DeepSeek's C4 prefill (baseline +0.6% against the
old arm on a v0 candidate) passes at six blocks only with a block sd near 0.36%.

**The 26B-A4B assistant has no speed gate.** Both screen harnesses refuse it
(`llmp_gemma_greedy_reference` repeated modes return 2 for profile 26;
`llmp_gemma_engine_greedy` takes only profile `31`), and the baseline records
no speed for it. It is served and controlled (quality below) and its peak memory
is recorded, as G2c did, but not gated. The 31B screen: old
`1facc957-fixed/benchmarks/llmp_gemma_greedy_reference` on `m3-artifacts`
(target `32c92e07…`, assistant `447a5c20…`), new
`$CAND/benchmarks/llmp_gemma_engine_greedy` on `m36-v1-artifacts` (target
`4d9a255a…`, assistant `07067ae0…`), the pinned `inputs.i32` (SHA-256
`b2d7aaf6…`, copied from the baseline's own run directory on `spark`), mode
`unit32`, 8 blocks and up to 11, the baseline's "estimate under about 0.7% is
within the harness's noise" read beside the verdict.

### Screens (before any gate)

One single-arm new run per family (`speed_gate.py run --label new --block 1
--position 1`, the new arm's template and environment, in its own job, no
priming), `spark-b`. Compared with the baseline same-binary means
([m36-baselines](../m36-baselines/README.md#same-binary-spread)): prefill
seconds, decode and wall tok/s per cell, and each cell's per-request
`prompt_tokens` with the baseline's recorded runs (`spark:~/.local/share/llmp/m36-baselines/aa/runs`).
**Any metric more than 1% slower stops the run and is reported to the
coordinator before anything else starts.** A screen is a first run after
idle, which the protocol notes can be slow by itself; if one metric trips,
the same single run is repeated once in a fresh job before it is reported, and
both are reported. Screen records are not gate data.

| Family | C1 prefill s / decode / wall | C2 (gemma26, gemma31: C4) prefill s / decode / wall | C4 prefill s / decode / wall |
| --- | --- | --- | --- |
| gemma2 | 1.297 / 62.55 / 38.48 | 1.944 / 104.7 / 58.72 | |
| gemma3 | 0.6163 / 74.93 / 55.36 | 1.101 / 133.0 / 85.08 | |
| gemma26 | 2.113 / 48.23 / 26.97 | | 8.921 / 103.2 / 37.80 |
| gemma31 | 9.039 / 9.903 / 5.854 | | 44.85 / 30.88 / 8.535 |
| qwen38 | 3.223 / 47.80 / 21.73 | 7.153 / 63.69 / 22.77 | 15.81 / 76.93 / 23.33 |
| qwen38-gguf | 6.572 / 30.39 / 11.90 | 13.96 / 41.78 / 12.79 | 28.68 / 63.10 / 13.87 |
| deepseek plain | 8.184 / 32.63 / 10.60 | 17.77 / 37.53 / 10.35 | 35.33 / 49.62 / 11.27 |
| deepseek DSpark waves (fixed baseline) | 8.194 / 32.54 / 10.58 | 17.78 / 45.02 / 11.04 | 36.40 / 53.01 / 11.50 |

### Quality

Every LLM family: `speed_gate.py quality` on the new arm (the new template, no
diagnostic switch) **and on the old arm** (the fixed baseline, v0, the current
tool, which retains per-target losses; the frozen records kept only the
aggregate), same passage (`passage(corpus, 100003, 4000)`), same corpus. Both
sets of records are kept. The decision rule per family, fixed now:

1. **Gemma 2, Gemma 3, Gemma4 26B-A4B and 31B.** The new arm's NLL, `scores_sha256` and
   two reply digests equal the baseline table's
   (`cf15e993…`, `ab21ad8d…`, `3d69b78c…`, `12c22f37…`; NLL 3.5635004927,
   3.6671293402, 9.7916674254, 9.5749963928): **PASS, and the llama.cpp
   oracle evidence of the baseline carries over unchanged** (copy-only v1 GGUF
   publication, G1c/G2c's same rule). If anything differs, the pairing
   (`quality_pair.py pair`, new against the old rerun) decides under
   `prereg/pair-gemma*.json`, and a failure there goes to the coordinator
   (the K1 llama-perplexity pairing is not rerun unless asked).
2. **Qwen3.8 native NVFP4/MXFP8 (+ MTP).** The numerics changed (expert relayout,
   planner), so equality is not expected. Decision: the four-document
   1,016-target corpus of the Mia pairing
   (`spark-b:~/scratch/m36-qwen-quant-reference03`; Mia's vLLM, the same
   checkpoint, `losses.jsonl` SHA-256 `42da92d6…`, mean NLL 3.559477762612,
   PPL 35.144838403, kept; the reference side is not rerun) scored through the
   literal route by the v1 target (`score_docs.py`, the gate's template), paired
   by (document, position, target ID) under
   [`prereg/qwen38-mia.json`](prereg/qwen38-mia.json): the aggregate PPL within
   1.0% of Mia's and every document's within 4.0% (observed on this corpus
   before: native A +0.39% with documents -1.8% to +3.5%, C -0.25%), mean
   absolute per-target difference at most 0.25 and maximum at most 5.0 nats
   (tripwires for a broken path, not tuned to a measurement: none was retained
   for this corpus). The pairing's other statistics are reported. The paired
   v1-against-v0 comparison on the gate passage (`prereg/pair-qwen38.json`) is
   informational for this family.
3. **Qwen3.8 GGUF and DeepSeek V4 Flash (+ DSpark).** No retained
   same-checkpoint reference losses for the gate passage exist on this host (llama.cpp
   b11254 and the DeepSeek `collect_reference` losses were not retained). The decision
   is the pairing of the new arm against the old arm's per-target losses under
   `prereg/pair-qwen38-gguf.json` and `prereg/pair-deepseek.json` (aggregate PPL
   within 0.5%, mean absolute difference at most 0.05, maximum at most 1.0, 95% of
   targets within 0.1 nats), on the same checkpoint, whose own oracle evidence
   is the baseline's table above. Reruns of `Dsv4CorpusScoreManual.Natural257`/`Natural2561`
   against the retained llama.cpp losses and of the MTP controls (`Qwen38JoinedMtpGpu`)
   need inputs and a reference this report did not find; they are listed as owed unless
   the coordinator names them.
4. **Assistants.** The two assistant-served configurations
   (`gemma26`/`gemma31` with the drafter, new arm only): the scores equal the plain new
   arm's (NLL and `scores_sha256`; G2c's observation), the 31B's two reply digests equal
   the plain arm's too; the 26B-A4B's replies may differ (a verify's four-row
   arithmetic is not a one-row decode's, as recorded in G2c) and are recorded.
   Peak memory of both is read from a `speed_gate.py run --prime` record.

Bounds are in [`prereg/`](prereg), SHA-256 prefixes `77f65729` (gemma2),
`4f9e2e74` (gemma3), `1c5afeaa` (gemma26), `d915aa85` (gemma31), `75214254`
(qwen38 pair), `452d5d25` (qwen38-gguf), `95072f47` (deepseek), `b37b1c7d`
(qwen38-mia).

### Peak memory

Informational, from each verdict's `peak_gib` (paired absolute difference of the
whole-node `MemAvailable` drop; the baseline's values are 5.913, 4.637, 21.035,
27.972, 82.601, 58.833 and 109.944 GiB, DSpark 110.005); admission, headroom and
run validity still bind (the owner, 2026-10-13, allows more memory than the
reference when it fits).

### Order

Screens, then quality, then the gates in this order: gemma2, gemma3, gemma26,
gemma31, the 31B assistant, qwen38, qwen38-gguf, deepseek plain, deepseek DSpark.
Total GPU time about 9.5 h at the scheduled blocks (minimums; DeepSeek's six), more
for each extension, serially on this one host.

## Amendment — tiered gates on both Sparks (owner, 2026-10-16)

Registered before any gate run, after the screens above. The owner chose to cut
the final pass from about 9.5 GPU-hours on one host to a tiered pass on both:

- **Hosts.** Families are split across `spark` and `spark-b`; each family runs
  wholly on one host, against that host's frozen `1facc957-fixed` install (the
  old arm matched its baseline means on spark-b within 0.7–2.5%).
- **Tier 1, confirmation.** Every family first runs a 2-block ABBA set
  (`speed_gate.py plan --blocks 2`). A family whose every gated metric has a
  2-block mean slowdown of at most +0.5% passes on that confirmation.
- **Tier 2, full gate.** A family with any metric above +0.5% in tier 1, and
  every family not validly gated on the migrated engine (DeepSeek DSpark
  speculative waves; Gemma 4 26B after its regression fix), runs the registered
  full gate (minimum blocks above, extended to the registered maximum while
  the verdict says CONTINUE).
- Quality, peak-memory reporting and the bounds above are unchanged. The
  image family keeps its full gate on `c8f0f80b` (PASS, −0.36%); after the
  image constants slice it is re-confirmed by tier 1.

## State layout (2026-10-16, investigation of the tier-1 regressions)

Screens on `spark`, 2 blocks each against `1facc957-fixed`, raw records in
`spark:~/scratch/m36-imgreg/` (deleted at the milestone's close).

- **Gemma 2 prefill.** `ef215344` packed every state tensor at 256 bytes.
  Gemma 2's 8.25 MiB local rings then stagger every later tensor across
  extents, so a prompt grows its state every chunk or two: the runtime's
  lookahead `prepared` count rose from 13 to 42 a run, and C1/C2 prefill
  were +1.8%/+2.1% (`c8f0f80b`), +0.83%/+1.30% (`343d375e`), while the G1c
  candidate measured +0.00%/+0.30% in the same window. `model::ExtentLayout`
  (each growing tensor of 2 MiB or more on its own extents, the rest packed
  at 256 bytes, a compressed cache contiguous with its raw window) gives
  `prepared` 13 and C1/C2 prefill −0.63%/−0.20% (decode −0.52%/−0.37%).
- **Slot state, in 2 MiB extents** (packed → rule → every tensor on its own
  extents; the region, then what 256 positions back; `state_layouts.h`
  measures it in `spec_models_test` and `components_*_models_test`):

  | Model (context/chunk) | Region | 256 positions | Growths over 8K |
  | --- | --- | --- | --- |
  | Gemma 2 (8192/128) | 319 → 338 → 338 | 52 → 52 → 52 | 23 → 8 → 8 |
  | Gemma 3 (4096/128) | 113 → 156 → 156 | 68 → 68 → 68 | 9 → 4 → 4 |
  | Gemma 4 26B (8192/1024) | 280 → 280 → 280 | 60 → 60 → 60 | 8 → 8 → 8 |
  | Gemma 4 31B (8192/256) | 820 → 820 → 820 | 120 → 120 → 120 | 18 → 18 → 18 |
  | Qwen3.8 target (33792/4096) | 564 → 589 → 649 | 106 → 108 → 157 | 2 → 2 → 2 |
  | Qwen3.8 MTP (33792/4096) | 123 → 126 → 127 | 84 → 85 → 86 | 2 → 2 → 2 |
  | DeepSeek V4 (262144/4096) | 958 → 968 → 1113 | 164 → 191 → 357 | 2 → 2 → 1 |
  | DeepSeek V4 (8448/256, C2's) | 49 → 49 → 250 | 49 → 49 → 250 | 1 → 1 → 1 |
  | DSpark (either) | 1 → 1 → 3 | 1 → 1 → 3 | 1 → 1 → 1 |

- **Image.** Not a code change: `c8f0f80b`'s binary, −0.4 to −1% in the
  morning, measured +3.28% in the afternoon beside `103686f5` (+3.39%) and
  `343d375e` (+3.94%), with `1facc957` against `1facc957-fixed` at +0.05%.
  After 17 idle minutes (GPU 41 °C, from 74 °C idle under the queue)
  `343d375e` measured −0.1% in one block: the migrated image path slows more
  than the M3 one when the host is heat-soaked, at the same SM clocks.

## Results

### Screens (2026-10-16, `spark-b`): stopped, four families over the 1% tripwire

Method as registered: one new-arm run per family in its own job, `speed_gate.py run`,
compared with the baseline same-binary means. "Slower %" is positive when the new arm is
slower (prefill seconds higher, rates lower). Records:
`spark-b:~/scratch/m36-gate/final/{screen,screen-repeat,screen-repeat3,screen-old}`; none is
gate data. Every run completed, no problems recorded, 128 tokens a request. To tell host from
candidate, the old arm (fixed baseline) was also run once on spark-b for gemma26, qwen38 and
DSpark: it reproduces the baseline's means within 0.1-2.5% (all inside run-to-run noise;
its worst cell is DSpark C4 decode 2.5% *faster*), so spark-b and `spark` agree.

| Family | Cell | First screen: prefill / decode / wall | Repeat: prefill / decode / wall |
| --- | --- | --- | --- |
| gemma2 | C1 | +0.32 / -0.16 / +0.04 | |
| gemma2 | C2 | +0.71 / -0.12 / +0.42 | |
| gemma3 | C1 | +0.18 / -0.66 / -0.43 | |
| gemma3 | C2 | -0.01 / -1.76 / -0.71 | |
| gemma26 | C1 | -0.79 / **+0.86** / +0.12 | +3.57 / +1.71 / +2.58; third run -1.02 / **+1.36** / +0.30 |
| gemma26 | C4 | -0.13 / **+1.59** / +0.87 | +3.50 / +3.22 / +3.98; third run +0.39 / **+1.84** / +1.22 |
| gemma31 | C1 | -0.58 / +0.47 / +0.03 | -0.36 / +0.44 / +0.11 |
| gemma31 | C4 | -0.34 / **+1.21** / +0.12 | -0.24 / +0.95 / +0.14 |
| qwen38 | C1 | -0.57 / -1.13 / -0.66 | -3.21 / -1.63 / -2.51 |
| qwen38 | C2 | -4.51 / -5.92 / -4.40 | -3.87 / -5.54 / -3.90 |
| qwen38 | C4 | **+9.93** / -1.75 / **+5.77** | +0.16 / -5.90 / -1.92 |
| qwen38-gguf | C1/C2/C4 | all faster (-1.9 to -10.8) | |
| deepseek plain | C1/C2/C4 | all within -1.6 to +0.2 | |
| deepseek DSpark waves | C1 | -1.47 / **+2.01** / -0.34 | -1.09 / **+2.21** / -0.01 |
| deepseek DSpark waves | C2 | -0.82 / **+6.52** / **+0.95** | -0.97 / **+7.64** / **+1.49** |
| deepseek DSpark waves | C4 | -0.19 / **+1.62** / **+1.41** | -0.86 / **+1.80** / **+1.21** |

Reading (single runs; the protocol's run sd is 0.1-1.5% by cell, so only repeats count):

- **qwen38**: the first screen's C4 prefill +9.9% did not repeat (+0.16%); no metric over 1% in the
  repeat. Peak 92.4 and 91.5 GiB against the baseline's 82.6 (+9 GiB, informational).
- **gemma31**: C4 decode +1.21% then +0.95%: at the line, not over it on the repeat.
- **gemma26**: decode is slower in all three new-arm runs (C1 +0.86, +1.71, +1.36; C4 +1.59, +3.22,
  +1.84%) while the old arm on this host is +0.55/+0.39% (a noise-sized difference); the prefill
  and wall excursions of the second run (+3.5%) did not recur. A decode cost of about 1.5-2%.
- **DSpark waves**: C2 decode +6.5% and +7.6%, C1 decode +2.0/+2.2%, C4 decode +1.6/+1.8% and C2/C4
  wall +1.0 to +1.5% in both new-arm runs, while the old arm on this host is faster than the
  baseline by 0.7-2.5%. This is a real slowdown of the DSpark speculative waves (plain waves
  are fine: DeepSeek plain is at or below the baseline in every cell).

Per the registered rule the gates did not start; reported to the coordinator.

### Superseded candidate 343d375e (tree 43c55509): `spark` results

Candidate `final-43c55509` (main `343d375e`, tree `43c55509…`), built with `spark-native --locked`
in a fresh tree on each host (not bit-identical across hosts) and installed read-only at
`~/.local/share/llmp/m36-candidates/final-43c55509`:

| Host | `llmp-runtime` SHA-256 | `IDENTITY.sha256` SHA-256 |
| --- | --- | --- |
| `spark` | `01afd4aced9492d09581ccfbc18d345af966b9e1bc6381e6ab6cbfbbf2d2b42b` | `c0144cf8602d2969ce1b926bccd80d85b3cba9a08694958e39c2e39960c3f0ce` |
| `spark-b` | `29be55ac65707f0577221491b610115379b150a3e3df61bf954a9bad1f19bdf1` | `9d1b48360e68b50566337a94b567f703b7e25a674ad7cad960090fd890a0b018` |

**Superseded by the coordinator on 2026-10-17:** the image and gemma2 regressions are being fixed
and those families re-gated on a new candidate; the queued `spark` jobs (gemma2 blocks 6-10, the rest
of gemma31, the 31B assistant blocks, the 31B assistant's peak run) were killed. Everything below is
this candidate's, kept as the record of what was measured. Tier 1 reads each gated metric's 2-block
mean slowdown against +0.5%; tier 2 is the registered gate, continuing the same output directory
(the tier-1 blocks are blocks 1-2). Records: `spark:~/scratch/m36-gate/final2/`.

| Family | Tier | Result (new arm against the fixed baseline, same host) |
| --- | --- | --- |
| image, pin `93356574…` | 1 then 2 | **FAIL at 6 blocks**: `generation_s` +3.19%, upper +3.82%, sd 0.57% (old 33.940 s, new 35.025 s); pixels equal the pin in every new run; peak +0.821 GiB |
| gemma2 | 1 then 2 | tier 1 not passed (C1/C2 prefill +1.03/+1.06%); **CONTINUE at 5 of 10 blocks**: C1 prefill +0.98% (upper +1.19%), C2 prefill +1.27% (upper +1.61%), C2 wall +0.43% (upper +0.72%); decode -0.39%/-0.41% (PASS), C1 wall +0.15% (PASS); peak -0.026 GiB |
| gemma3 | 1 | PASS: C1 decode -1.25%, C1 prefill +0.29%, C1 wall -0.82%, C2 decode -2.41%, C2 prefill -0.98%, C2 wall -1.69%; peak -0.108 GiB |
| gemma26 (tier 2 upfront, 9 blocks) | 2 | **PASS at 9 blocks**: C1 decode +0.46% (upper +0.70%), C1 prefill -0.91%, C1 wall -0.16%, C4 decode +0.53% (upper +0.86%), C4 prefill -0.81%, C4 wall -0.20%; peak -0.227 GiB |
| gemma31 | 1 | partial (one block of two): all six metrics within +-0.5% (C1 decode +0.28%, C4 decode +0.31%); INVALID as a record (block 2 incomplete, jobs killed) |
| gemma31 assistant (`assistant_gate.py`) | 1 | two blocks: -2.84%, -4.22% (mean -3.53%, bound [-12.3%, +5.2%]); INCOMPLETE by the screen's 8-block minimum |

Quality on `spark` (`speed_gate.py quality`, the new arm without any diagnostic, and the old arm
rerun with the same tool; same 1,070-token passage):

| Family | New arm NLL / scores | Old arm | Against the baseline table |
| --- | --- | --- | --- |
| gemma2 | 3.5635004927, `cf15e993…` | identical scores and both replies | equal: PASS by the rule (oracle evidence carries over) |
| gemma26 | 9.7916674254, `3d69b78c…` | identical | equal: PASS |
| gemma31 | 9.5749963928, `12c22f37…` | identical | equal: PASS |
| gemma3 | 3.6681064003, `142d30d2…` (baseline 3.6671293402, `ab21ad8d…`); both replies differ | 3.6671293402, `ab21ad8d…` | **not equal; pairing FAILS its pre-registration** |

Gemma3's pairing (`quality_pair.py`, `prereg/pair-gemma3.json`, new against old, 1,070 paired
targets, no target-ID difference): mean NLL +0.000977, PPL +0.0978% (bound 0.5%: ok), 519 targets
lower and 551 higher, mean absolute per-target difference 0.0869 (bound 0.05: **FAIL**), maximum
0.837 at position 89 (bound 1.0: ok), 70.6% within 0.1 nats (bound 95%: **FAIL**). G1c measured this
family's NLL equal to the baseline's; the difference is new since then (an arithmetic change in the
Gemma 3 path, not a tokenizer difference). The aggregate is within 0.1% but the registered per-target
bounds fail, so it is reported as FAIL, not adjusted.

The assistant-served 26B-A4B and 31B (new arm, drafter): scores equal the plain new arm's
(NLL 9.7916674254 `3d69b78c…`; 9.5749963928 `12c22f37…`), the 31B's two reply digests equal the
plain arm's (`5318071d`, `fd56b9a1`), the 26B-A4B's differ (`90ce24e7`, `27d60ddd` against `3a602ba8`,
`394783db`; as G2c recorded). The 26B-A4B assistant configuration's peak: 21.642 GiB (one
`run --prime`, C1 decode 96.2 tok/s, C4 178.4 tok/s); the 31B's peak run was killed.

### Candidate 343d375e (tree 43c55509): `spark-b` results

Same candidate (spark-b runtime `29be55ac…`), against the fixed baseline on spark-b. Records:
`spark-b:~/scratch/m36-gate/final2/`. All runs completed, 128 tokens a request, no cached prompt
tokens, no problems, and the verdicts' same-work check (per-request `prompt_tokens` equal between the
arms, v0 against v1 artifacts) holds (no INVALID).

| Family | Tier | Blocks | Result (slowdown of the new arm; negative is faster) | Peak (new minus old) |
| --- | --- | --- | --- | --- |
| deepseek plain | 1: PASS | 2 | C1 prefill -0.74%, decode -1.48%, wall -0.98%; C2 prefill -1.17%, decode -1.12%, wall -1.28%; C4 prefill -0.53%, decode -2.15%, wall -0.97% (no metric above +0.5%) | +0.212 GiB |
| deepseek DSpark waves | 2: **PASS** (6 upfront) | 6 | C1 decode -0.29% (upper +0.36%), prefill -1.38%, wall -1.03%; C2 decode -1.50%, prefill -1.21%, wall -1.24%; C4 decode -1.09% (upper -0.47%), prefill -0.64%, wall +0.14% (upper +0.53%) | -0.093 GiB |
| qwen38 (+MTP) | 1: PASS | 2 | C1 prefill -3.14%, decode -1.33%, wall -2.31%; C2 prefill -3.52%, decode -6.01%, wall -3.96%; C4 prefill -2.64%, decode -4.56%, wall -4.58% | +8.142 GiB (82.650 to 90.793) |
| qwen38-gguf | 1: PASS | 2 | C1 prefill -2.09%, decode -1.42%, wall -1.83%; C2 prefill -3.41%, decode -12.59%, wall -5.88%; C4 prefill -2.15%, decode -2.69%, wall -2.66% | +7.999 GiB (58.760 to 66.759) |

Tier 1 passes by the amendment's rule (every gated metric's 2-block mean at most +0.5%). The DSpark
gate is the registered one (all six blocks, upper bounds at most +0.995% on every metric: PASS).

**Quality (`spark-b`, `speed_gate.py quality` on the new arm and, with the same tool, the old arm;
`quality_pair.py` under the fixed pre-registrations).**

| Family | New arm | Old arm / reference | Pairing |
| --- | --- | --- | --- |
| deepseek | NLL 2.7939955482, `6adb9558…` (equals the baseline table) | identical scores, all 966 losses identical | **PASS**: PPL delta 0.0000%, mean absolute difference 0, agreement 100%; the two greedy replies differ from the old arm's (`9f9e77cd`/`026da45b` against `d421be83`/`52669598`) |
| qwen38-gguf | NLL 2.8596571307, `4096043a…` (baseline 2.8538864116, `f5ce8833…`) | old arm reproduces the baseline | **FAIL**: PPL +0.5787% (bound 0.5%), mean absolute difference 0.2307 (0.05), maximum 3.108 at position 512 (1.0), agreement 53.6% (95%); 446 targets lower, 486 higher, 64 identical |
| qwen38, against the old arm | NLL 2.7564454945, `9266b500…` (baseline 2.7655865453, `b7ed7b14…`) | old arm reproduces the baseline | informational per the registration: PPL -0.9099%, mean absolute difference 0.1650, maximum 2.465 at position 563, agreement 62.2% |
| qwen38, against Mia's vLLM (four documents, `prereg/qwen38-mia.json`) | 1,016 targets, NLL 3.568635890, PPL 35.468178 | Mia NLL 3.559477763, PPL 35.144838 | **FAIL**: aggregate PPL +0.9200% (bound 1.0%: ok) but documents +1.22%, **+6.91%**, -2.32%, -1.87% (bound 4.0%: FAIL, document 1); mean absolute per-target difference 0.524 (tripwire 0.25: FAIL), maximum 6.352 at document 1 position 147 (tripwire 5.0: FAIL) |

For the record, earlier native observations on this same corpus (before the later changes) were
aggregate +0.39% and document 1 at +3.5%; the document-1 excursion has grown. The MTP controls run in the
final models set, which has not run.

## Final candidate 53c2610b (tree d373a1bf): the verification

Main `53c2610ba4e8e737c7abd2524eb26dcecd191db7`, tree `d373a1bf4f6bb047ac77e1cb2d1f258afca68d9a`
(over 343d375e: the per-tensor state layout `0c6bcc9a`, the resident shared workspace `be5c4db6`, the
gate tools' v1 image default `53c2610b`, and a test-only change). Built with `spark-native --locked`
in a fresh tree on each host (4 min each) and installed read-only as
`~/.local/share/llmp/m36-candidates/final-d373a1bf`; the two hosts' binaries differ:

| Host | `llmp-runtime` SHA-256 | `llmp_gemma_engine_greedy` | `IDENTITY.sha256` SHA-256 |
| --- | --- | --- | --- |
| `spark` | `55fd8e9de3379b0683b0c0b8a35ead8cd65a13827aaf2a80fab014f149a16150` | `d7d9e5f0…` | `ac8df42794ce13841acba16c4377d8c25afdce192b0f3e01eec3ff1c6c8f0676` |
| `spark-b` | `66e343ee47007a9888dfcbf622ddf9fac28bc450f27d3ddf1c6df8197b540537` | `d03d42c3…` | `f92dda74f591268c43ce64c1313d6dfe90029fc7d17913f3373a9be6cc5e91c9` |

Method as pre-registered and amended: each family wholly on one host against that host's
`1facc957-fixed` install (runtime `f5443363…`); tier 1 = a 2-block ABBA confirmation passing when
every gated metric's 2-block mean is at most +0.5%; tier 2 = the registered gate (the tier-1 blocks
count as blocks 1-2, extended while CONTINUE). New arm on the v1 store, the gate templates of the
pre-registration (the image's new arm serves the v1 composition `2170db12…` by default, pin
`93356574…`), `LLMP_DIAGNOSTIC_SKIP_TURN_CHECKPOINTS=1` for the four Gemma families only. Before each
session the page cache of the artifact stores was dropped (RE-057) and the GPU temperature logged
(`temps.log`: before each session's priming run and after every run). Records:
`spark:~/scratch/m36-gate/final3/` and `spark-b:~/scratch/m36-gate/final3/`.

### Speed

| Family (host) | Tier | Blocks | Verdict | Worst metric (mean, upper) | Peak memory, new minus old |
| --- | --- | ---: | --- | --- | --- |
| image, pin `93356574…` (spark) | 2 (6 then 8) | 8 | **PASS** | `generation_s` +0.13%, upper +0.73% (old 34.373 s, new 34.418 s; at 6 blocks +0.25%, upper +1.08%: CONTINUE) | +0.816 GiB |
| gemma2 (spark) | 2 (5 then 7) | 7 | **PASS** | C1 prefill +0.09%, upper +0.73%; C2 prefill +0.10% (+0.57%); decode -0.55%/-0.66%; wall -0.30%/-0.20% | +0.034 GiB |
| gemma26 (spark) | 2 | 9 | **PASS** | C4 decode +0.50%, upper +0.70%; C4 wall +0.44% (+0.61%); C1 decode +0.20% (+0.42%); prefill +0.08%/+0.20% | -0.227 GiB |
| gemma31 (spark) | 1 | 2 | PASS | C1 prefill +0.19%, wall +0.07%, decode -0.01%; C4 decode +0.10%, prefill +0.02%, wall +0.07% | -0.008 GiB |
| gemma3 (spark) | 1 | 2 | PASS | C1 prefill -0.21%, decode -1.05%; C2 prefill -0.47%, decode -2.72% (all at most +0.5%) | +0.054 GiB |
| gemma31 assistant, `assistant_gate.py` (spark) | 8 | 8 | **PASS** | -3.85%, bound [-4.34%, -3.37%] (1.3106-1.3300 s against 1.3610-1.3872 s for 32 committed tokens; every block -3.0% to -4.9%) | n/a |
| deepseek plain (spark-b) | 1 | 2 | PASS | C4 prefill +0.29%, C2 prefill +0.31% (the +0.5% line not crossed); decodes -1.06% to -2.69% | +0.282 GiB |
| qwen38 + MTP (spark-b) | 1 | 2 | PASS | every metric faster: C1 prefill -2.11%, C2 decode -5.74%, C4 prefill -1.74%, C4 wall -3.76% | +7.932 GiB (82.634 to 90.566) |
| qwen38-gguf (spark-b) | 1 | 2 | PASS | every metric faster: C1 prefill -0.78%, C2 decode -13.03%, C4 prefill -0.88% | +7.684 GiB (58.851 to 66.535) |
| deepseek DSpark waves (spark-b) | 1 then 2 | 6 | **NOT PASSED** | **C4 wall +0.94%, upper +1.20%** (limit +0.995%; lower bound +0.68%); C4 prefill +0.55% (upper +0.71%, PASS); the other metrics PASS (C1 decode -0.31%, C2 decode -1.82%, C4 decode -1.45%, prefill and wall at most +0.12%) | +0.106 GiB |

Notes:

- **DSpark C4 wall** is the one gate not passed on this candidate. Per block the C4 wall slowdowns
  are +0.98, +0.78, +1.11, +0.75, +1.33, +0.68% (sd 0.25%), C4 prefill +0.42, +0.63, +0.60, +0.44,
  +0.79, +0.38%: a steady small cost, not noise. The same cells measured +0.14% wall and -0.64% prefill
  on 343d375e; the change since is the layout rule that moves DeepSeek's 262,144-context slot state
  (region 958 to 1113 MiB, 256 positions 164 to 357 MiB, see "State layout" above). The plain-wave
  gate passed tier 1 on the same layout (C4 prefill +0.29%); only the speculative waves' C4 cell reads
  over the line. The registered maximum is 6 blocks, so it stays NOT PASSED.
- **Image** at 6 blocks (+0.25%, upper +1.08%) was CONTINUE, not a pass; it passed at 8. GPU 63-72 C
  across the runs (heat-soaked after a 45 C start): the old arm drifted from 33.9 to 34.5 s while the
  new arm held 34.2-34.5 s, the heat effect the investigation described.
- Gemma2's C1 prefill was CONTINUE at 5 blocks (+0.10%, upper +1.15%, sd 0.85%) and passed at 7.
- Temperatures (GPU C after each run, session starts in brackets): gemma2 66-68 (59-68), gemma26 64-70
  (60-69), gemma31 66-67 (60-67), gemma3 68-69 (67), image 63-72 (45, 69, 70, 63); spark-b: deepseek
  plain and DSpark 64-70 (45 to 70), qwen38 63-65 (63-67), qwen38-gguf 62-68 (61-65). SM clocks
  2405-2411 MHz throughout (no throttling). Both hosts were 45-46 C when the first session started.
- Peak memory is informational. The assistant-served 26B-A4B configuration peaks at 21.522 GiB and the
  31B at 28.468 GiB (`run --prime`, one run each).

### Quality

Re-run on the final candidate (new arm without any diagnostic; old arm with the same tool on the
fixed baseline): **every score digest, NLL and reply digest equals 343d375e's**, so the layout and
workspace changes did not change numerics.

| Family | New arm NLL, scores | Equal to the baseline table? | Pairing |
| --- | --- | --- | --- |
| gemma2 | 3.5635004927, `cf15e993` | yes (and the old arm) | PASS by the equality rule |
| gemma26 | 9.7916674254, `3d69b78c` | yes | PASS |
| gemma31 | 9.5749963928, `12c22f37` | yes | PASS |
| deepseek | 2.7939955482, `6adb9558` | yes (all 966 losses identical to the old arm) | PASS (`prereg/pair-deepseek.json`) |
| gemma3 | 3.6681064003, `142d30d2` (baseline 3.6671293402, `ab21ad8d`) | no | exceeds its registered tripwires (below) |
| qwen38-gguf | 2.8596571307, `4096043a` (baseline 2.8538864116) | no | exceeds (below) |
| qwen38 | 2.7564454945, `9266b500` (baseline 2.7655865453) | no | against Mia exceeds (below) |

The assistant-served configurations: scores equal the plain arm's (26B-A4B `3d69b78c`, 31B `12c22f37`);
the 31B's reply digests equal the plain arm's, the 26B-A4B's differ (as G2c recorded).

Registered pairings that exceed their bounds, with the coordinator's classification (2026-10-17,
evidence `spark-b:~/scratch/m36-quality-reg/`, accepted on the reference comparison):

- **gemma3** (new against old, `prereg/pair-gemma3.json`): PPL +0.098% (ok), mean absolute per-target
  difference 0.0869 (bound 0.05), 70.6% of targets within 0.1 (bound 95%). Against llama.cpp v0.6.0 on
  the same GGUF, the **old** arm differs by PPL -0.52%, mean absolute difference 0.0765 (max 0.99), the
  **new** arm by PPL -0.42%, 0.0825 (max 1.24): both arms sit at the same inter-implementation
  distance, so the 0.05 tripwire is below the noise floor between any two implementations of this
  checkpoint (pre-existing), and the arithmetic change is within it.
- **qwen38-gguf** (new against old): PPL +0.58% (bound 0.5%), mean absolute difference 0.231. Against
  llama.cpp b11254: new PPL +0.02% (mean absolute 0.311), old PPL -0.55% (0.324); llama.cpp's own fused
  against unfused build differ by PPL -1.75%, mean absolute 0.296, max 5.5. The new arm is closer to
  the reference in aggregate than the old, and the old/new difference is the size of llama.cpp's own
  fusion variation: legitimate numerics.
- **qwen38 against Mia** (four documents, `prereg/qwen38-mia.json`): aggregate PPL +0.92% (bound 1.0%:
  ok); document PPLs +1.22%, **+6.91%**, -2.32%, -1.87% (bound 4.0%); mean absolute 0.524 (tripwire
  0.25), max 6.35 (tripwire 5.0). The **old** arm against the same reference: PPL +0.73%, mean
  absolute 0.521, max 6.32 (on the gate passage against Mia's other reference: old -2.85%/0.518, new
  -3.74%/0.525): the per-target spread is Mia's (FP8 KV, BF16 SSM) against ours, present in the old
  arm. The absolute-difference bounds are tripwires, not measured limits.
- Not re-run here (owed by the registration): llama.cpp b11254 and the DeepSeek `collect_reference`
  reruns. The Qwen MTP controls are in the check set below.

### Swap tables (spark, `swap_table.py`, v1 IDs, image pin `93356574…`)

M3 table, 32 rows in three jobs of two ordered pairs (context text `6b159ff2…`, 8,192 tokens, 16
continuation tokens, 2 cycles). **PASS**: every row exact and state-exact, every restored 8K hash
equal, every continuation token and logit row equal, prepared returns kept and replayed their graphs,
every image regeneration equal to the pin, prepared rows at most 20 s and first-use rows at most 40 s.
Worst LLM<->LLM row **8.665 s** (qwen3.8 to deepseek, first use, 8K) against 8.494 s (fixed baseline)
and 9.853 s (M3); worst prepared LLM<->LLM row 8.505 s; largest tracked peak 111.73 GiB; lowest
MemAvailable 4.69 GiB. The image-A rows: image to deepseek first use 27.354 s (the workspace pages for
the first load beside DeepSeek, as expected), its prepared return 8.804 s.

```
swap table m3: 32 of 32 rows
  qwen3.8 <-> deepseek   A->B, first use (8192 context tokens)            8.665 s  peak  111.37 GiB (119582449664 B)
  qwen3.8 <-> deepseek   B->A, first use (8192 context tokens)            6.228 s  peak  111.42 GiB (119636574208 B)
  qwen3.8 <-> deepseek   A->B, prepared (8192 context tokens)             8.427 s  peak  111.44 GiB (119657615360 B)
  qwen3.8 <-> deepseek   B->A, prepared (8192 context tokens)             6.222 s  peak  111.44 GiB (119660109824 B)
  qwen3.8 <-> deepseek   A->B, prepared (0 context)                       8.346 s  peak  111.45 GiB (119664852992 B)
  qwen3.8 <-> deepseek   B->A, prepared (0 context)                       6.357 s  peak  111.45 GiB (119664852992 B)
  qwen3.8 <-> image      A->B, first use (8192 context tokens)            3.896 s  peak   88.89 GiB (95447707648 B)
  qwen3.8 <-> image      B->A, first use (8192 context tokens)            6.291 s  peak   88.99 GiB (95553331200 B)
  qwen3.8 <-> image      A->B, prepared (8192 context tokens)             3.801 s  peak   88.94 GiB (95501488128 B)
  qwen3.8 <-> image      B->A, prepared (8192 context tokens)             6.237 s  peak   89.01 GiB (95570669568 B)
  qwen3.8 <-> image      A->B, prepared (0 context)                       3.736 s  peak   89.01 GiB (95570669568 B)
  qwen3.8 <-> image      B->A, prepared (0 context)                       6.360 s  peak   88.75 GiB (95294029824 B)
  deepseek <-> image     A->B, first use (8192 context tokens)            4.124 s  peak  110.94 GiB (119126188032 B)
  deepseek <-> image     B->A, first use (8192 context tokens)            8.283 s  peak  111.26 GiB (119459717120 B)
  deepseek <-> image     A->B, prepared (8192 context tokens)             3.885 s  peak  111.29 GiB (119498944512 B)
  deepseek <-> image     B->A, prepared (8192 context tokens)             8.229 s  peak  111.27 GiB (119474786304 B)
  deepseek <-> image     A->B, prepared (0 context)                       3.915 s  peak  111.28 GiB (119487590400 B)
  deepseek <-> image     B->A, prepared (0 context)                       8.452 s  peak  111.25 GiB (119449931776 B)
  deepseek <-> qwen3.8   A->B, first use (8192 context tokens)            6.479 s  peak  111.30 GiB (119505690624 B)
  deepseek <-> qwen3.8   B->A, first use (8192 context tokens)            8.340 s  peak  111.73 GiB (119964717056 B)
  deepseek <-> qwen3.8   A->B, prepared (8192 context tokens)             6.374 s  peak  111.70 GiB (119939899392 B)
  deepseek <-> qwen3.8   B->A, prepared (8192 context tokens)             8.293 s  peak  111.70 GiB (119938928640 B)
  deepseek <-> qwen3.8   A->B, prepared (0 context)                       6.356 s  peak  111.70 GiB (119938813952 B)
  deepseek <-> qwen3.8   B->A, prepared (0 context)                       8.505 s  peak  111.67 GiB (119903993856 B)
  image <-> deepseek     A->B, first use (image)                         27.354 s  peak  108.73 GiB (116752592896 B)
  image <-> deepseek     B->A, first use (image)                         16.274 s  peak  108.94 GiB (116974841856 B)
  image <-> deepseek     A->B, prepared (image)                           8.804 s  peak  109.35 GiB (117412573184 B)
  image <-> deepseek     B->A, prepared (image)                           3.872 s  peak  109.38 GiB (117442572288 B)
  image <-> qwen3.8      A->B, first use (image)                          7.202 s  peak   86.68 GiB (93074264064 B)
  image <-> qwen3.8      B->A, first use (image)                          3.752 s  peak   86.68 GiB (93074558976 B)
  image <-> qwen3.8      A->B, prepared (image)                           6.306 s  peak   86.71 GiB (93105139712 B)
  image <-> qwen3.8      B->A, prepared (image)                           3.768 s  peak   86.71 GiB (93105139712 B)
worst LLM<->LLM row: 8.665 s (qwen3.8 <-> deepseek, A->B, first use (8192 context tokens), peak 111.37 GiB)
worst prepared LLM<->LLM row: 8.505 s (deepseek <-> qwen3.8, B->A, prepared (0 context), peak 111.67 GiB)
largest tracked peak: 111.73 GiB
lowest MemAvailable: 4.69 GiB
PASS
```

Gemma near-8K cases (one Gemma model against Qwen3.8 plain, 8,063 tokens at context 8,192; gemma3 8,192
at 8,448; 64 continuation tokens; 6 rows each, 24 rows): **all PASS**, every row exact, prepared
returns keeping and replaying graphs.

| Pair | Worst first use | Worst prepared | Largest peak |
| --- | ---: | ---: | ---: |
| gemma2 / qwen3.8 | 6.027 s | 5.869 s | 76.49 GiB |
| gemma3 / qwen3.8 | 6.016 s | 5.843 s | 76.33 GiB |
| gemma26 / qwen3.8 | 6.421 s | 5.876 s | 76.25 GiB |
| gemma31 / qwen3.8 | 6.132 s | 5.961 s | 77.31 GiB |

(The baseline's 5.99-6.09 s; the Gemma models' B-to-A returns take 0.29-1.69 s.)

### Check set (final tree `53c2610b`)

All green; no test failed.

| Step | Host | Result |
| --- | --- | --- |
| `mise run build -- spark-native --locked` | spark-b | ok (the tree installed from) |
| shared, `ctest -LE 'models\|timing'` | spark-b | 2839 of 2839 passed |
| CPU-only guardrail: the `spark-native` preset with `LLMP_CUDA=OFF` in `build/spark-cpu` (the `cpu` preset is x86-64 only), `ctest -LE 'models\|timing'` | spark-b | 2273 of 2273 passed |
| exclusive `-L 'models\|timing'`, 214 tests in 8 disjoint area groups (`ctest -N` total 214 = 77 + 27 + 17 + 17 + 22 + 10 + 29 + 15) | spark (the models jobs ran on spark in parallel with spark-b's shared jobs, not on spark-b) | all passed: tokenizer/chat/import/spec (77), Gemma 2/3 engine and serving (27), Gemma 4 and assistants (17), DeepSeek and DSpark (17), Qwen3.8 incl. MTP (22), image (10), Qwen2/EXL3/FP16 fixtures (29), catch-all (15) |

The image generation diagnostics ran with `LLMP_IMAGE_GENERATION_STORE` and
`LLMP_IMAGE_GROWTH_GEMMA_STORE` at the v1 store, `LLMP_IMAGE_GENERATION_NOISE` at `latents_init.bf16`,
and `LLMP_RUN_SHARED_IMAGE_{GENERATION,REQUEST,REPLACEMENT,GROWTH,SERVICE}=1`: 9 of the 10 image tests
ran and passed. The Qwen2/EXL3/FP16 group ran with the recorded fixture inputs (`g5-final-oracle-inputs`,
`g5-final-exl3-inputs`, `LLMP_TEST_QWEN2_ARM=v1`, `LLMP_TEST_V1_FROZEN_ONLY=1`): 10 of 29 ran and passed.

**Skipped, each read with `-V`** (explicit env-gated diagnostics whose inputs this pass did not have):

- 19 Qwen2/EXL3/FP16 fixture tests: the sealed corpus, trajectory and held-out oracles
  (`LLMP_QWEN2_CORPUS_IDS`, `LLMP_EXL3_CORPUS256_IDS`, `LLMP_EXL3_TRAJECTORY_ORACLE*`,
  `LLMP_QWEN2_FP16_HELDOUT_*`) are not wired to this run; the fixtures are not served.
- Five importer tests on real checkpoints: ModelOpt x2 (`LLMP_TEST_MODELOPT_SOURCE`,
  `LLMP_TEST_MODELOPT_PUBLICATION_STORE`), MLX, DSpark and the Gemma 4 assistants' source checkpoints.
- Two Qwen3.8 quantized-head tests (they need the authenticated selected-head fixture).
- `ImageGenerationGpu.SameSourceMigrationScreen` (needs the explicit same-source migration screen).

**Tiers not run (named):** `check`, `check:full` (sanitizers, container, offline copyleft-disabled)
and `check:spark` (cross/ASan/TSan); the package fixture; the workstation `native --gpu` sm_86 proof
(owed; it needs the workstation under `hostlock`); the K1 llama-perplexity pairing for Gemma. All are
owed before a package ships (D-085).

### Summary

PASS: image, gemma2, gemma26, gemma31, gemma3, the 31B assistant screen, deepseek plain, qwen38 (+MTP),
qwen38-gguf, the M3 swap table (32 rows), the four Gemma near-8K cases (24 rows) and the check set.
**NOT PASSED: deepseek DSpark waves, C4 wall +0.94% (upper +1.20%) at the registered 6 blocks.**
Quality: equal to the baseline for gemma2, gemma26, gemma31 and deepseek; the gemma3, qwen38-gguf and
qwen38 pairings exceed their registered tripwires and are classified above as legitimate numerics or
pre-existing differences.
