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
