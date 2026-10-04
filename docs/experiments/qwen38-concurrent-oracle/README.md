<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Current Qwen four-slot conditional oracle qualification

On 2026-10-04, all eight complete 512-row cells pass the unchanged Qwen
32K oracle bound, with exact complete logits and initialized target/MTP
state across two repeats. This qualifies one current four-slot target-row
geometry on a frozen oracle history. It is a numerical diagnostic, with
no natural acceptance, sampled-distribution or performance claim.

## Result

| Check | Actual result |
| --- | --- |
| Each of four slots, twice | 477 greedy agreements, 35 near ties, zero outside the 1.765 bound; maximum oracle margin among near ties 1.125. All eight strict passes. |
| Oracle-continuation ratio | 1.017589472; retained reference 1.020176456, allowed 1.025176456 under the unchanged 0.005 tolerance. No tolerance or reference change. |
| Full rows and device verdicts | Each cell saves all 512 × 248,320 finite F32 logits; all 511 post-prefill device IDs equal actual CPU argmaxes of those complete rows. |
| Joined execution per repeat | 170 twelve-row waves plus one four-row tail; 171 real depth-two draft calls per slot. Verify capture/replay counts are 1/168 and 1/170. |
| Complete own repeats | All four 512-row comparisons are byte-identical; maximum absolute difference zero. All eight row hashes also agree across slots. |
| Final initialized state | Each slot retains 1,438,564,352 initialized target/MTP bytes; hash identical across slots and both repeats. |
| Retirement | Native child and analysis launcher/container exit zero; children reaped, container removed, no forced native kill or cleanup error. All four model/analysis admission and terminal 105 GiB probes pass. |

The joined waves retain the literal history at every step. The controller
independently checks both 171-entry conditioning transcripts: positions,
rows, outputs-before and SHA-256 of the actual prompt-plus-oracle inputs.
The corresponding transcripts match exactly. Rows, initialized state,
counts and capture/replay are native witnesses, checked again by the
controller; quality and repeat comparisons use the unchanged registered
long-context judge.

## Frozen profile and method

Four independent slots clone the Mia oracle's 31,743-token prompt. Each
prefills through the runtime's `RunPrefillChunks`, split at stable boundary
31,738, with maximum chunk 4,096 and MTP injection. The last prefill row
scores output zero. Context is 33,792. Production nominal draft capacity
stays three; the actual shared draft depth is two. Graphs and wave lanes
are on, with the existing attention read alignment. Adaptive depth,
confidence window, profiler and optional fixture are off.

Before every verify the driver actually calls scalar `Draft(2)` for all
four slots, preserving the current C4 catch-up path. It discards those
proposals, supplies the oracle anchor and following oracle tokens to
`VerifyWave`, then calls `Accept(rows)` on the supplied rows. The final
tail has one row per slot. This teacher-forced path exercises current
joined target products, attention, state and real MTP catch-up without
following naturally accepted drafts. It does not qualify a natural reply
trajectory, stochastic verdicts, arbitrary arrivals or ragged cohorts.

The external driver replaces only `Harness::Wave` and admits this frozen
literal-ID wave profile in its private parser. Production source/defaults
stay unchanged. One external translation unit links the authenticated
`cpuclos1` engine archives and support object; the relocated executable's
cuBLAS RUNPATH is adjusted. Compiler flags, static runtimes, all 66 build
inputs, linked libraries and original archives remain authenticated before
and after. The 491-file C++ source/header inventory includes benchmarks;
it is a tree inventory, not a claim every file was compiled into the driver.

The measured engine is the retained native Spark build of `03dcec0` plus
the reviewed CPU-source-receipt repair, qualified by the 1,562-test suite.
Its model/kernel bytes match the subsequent source/package/SDK repairs;
this run is not relabeled as a clean execution of those later commits.
It uses the original `aarch64-e0a0c85c42806fb1` SDK identity. Its native
SDK bytes are verified identical to the repaired SDK's native tree.

## Oracle and retained reference

The oracle is the fixed deterministic Mia `qw-mia-det/qwen3.8-32k.json`.
The target is `c4fb47a911207c11f935f932d05196dc1701aa0d886eac1b5e91934e554b5a93`;
the drafter is `8600a99819ce583a719ebfb457de8cac40b4d0bd1ebe557ceb13dff5961aee40`.
Requested draft vocabulary 65,536 executes the actual selected 47,172-row
head. Tokenizer and template are pinned from
`Mia-AiLab/Qwen3.8-Flash-Next-NVFP4@925d7be6`.

The unchanged judge uses the owner's retained `hq-32k-fast/q32k.logits.f32`
reference ([original qualification](../long-context/README.md)). No
contemporaneous reference checksum was recorded. Its complete original
capture SHA is recorded retrospectively here, without regeneration or
rebasing. Authentication joins all 512 argmaxes to the untouched retained
summary, its geometry/path/timestamps and the original report; the fixed
judge reproduces 477 agreements, 35 near ties, maximum oracle margin 1.125
and zero outside. The full SHA is then mandatory for this frozen run.

The analysis uses the pinned PyTorch image below in a read-only, networkless
CPU `runc` container, with no GPU device requests or model-module imports.
It applies the unchanged judge to all eight complete captures and four
complete repeats. Unique ownership labels reconcile container creation
before cleanup even if creation is interrupted.

## Sampling evidence applicability

Serving `Llm::Choose`/`Keep` and both model judges call the same
`execution::Sample`/`VerifyDraft`, using stream zero and absolute predicted
token position. Each branch owns its seed, parameters and scratch. Resume
preserves the reported anchor and absolute cursor; per-slot `Accept` and
failed-slot `DiscardVerify` keep commits isolated. Qwen sampling borrows
complete target rows with fixed depth-two offers; sampled DeepSeek bypasses
adaptive form exploration with its fixed draft maximum. Source mapping
found no unsupported key, row ownership or commit/resume dispatch path.

Meaningful `llm_scores_test.cc` controls cover absolute-position resumption,
four interleaved branches, completed peers on sampling error, failed-slot
prefix/cursor isolation, preemption rebuild and spilled-state resumption.
Shared sampler tests cover rejection correction, exact distributions,
Philox and keyed reproducibility. These are source/unit mapping evidence.
Native literal controls additionally compare C1 seeded sampling against
the old serial path; their mixed-cohort/model-turn controls are greedy.

The registered per-model histograms remain evidence for their measured
profiles: four short prompts, 256 seeds, eight outputs, temperature one,
top-16-plus-other TV bound 0.1. Qwen's historical MTP values are
0.0317/0.0396/0.0093/0.0332 ([report](../qwen38-mtp/README.md)); refreshed
DeepSeek frontier/HCA values are 0.0034/0.0098/0.0186/0.0112
([report](../dsv4-frontier-head/README.md)). Qwen's historical original
prefix head differs from the selected 8600 head here. This conditional
C4 check does not execute stochastic verdicts or measure a current HTTP/C4
histogram. A fresh cohort histogram, or complete row/key/verdict/commit
equivalence to the registered profile, would be required to claim that
particular measured TV result. No new distribution pass is inferred.

## Provenance and replay

| Frozen object | SHA-256 |
| --- | --- |
| Oracle JSON | `b111e1395e2436f4c79cbbc5f27bcbe16470ee9c96b116c54338e5815891a0b8` |
| Prompt LE I32 IDs | `306136a7dc2a5f66c94cbf4ac874e9011c0af79b79072731407a144bfba1fbd5` |
| Retained reference complete rows | `18a76456b1277a1c148c7bda3003472b773fe9a4ed39e4875cf5bd5cb80df8b2` |
| Retained reference original summary | `a10c9616a88ff8ed1d1236a7f70e915937b3f67d3ce749a4f84832a7a9e464e0` |
| Unchanged judge | `115ea29757c8c682b30d6fd418afe71ba081bc528c2b9f77b90562e978b7e854` |
| External driver source | `c4741dc87735197d6bb8f7e3bc85667e2456c5bb1aca7fc66d096d31465640e2` |
| External executable | `a8375282f8d327d9679e0dd7a74f6d69495a7432e842964ae05c621686f2863d` |
| 491-file source inventory | `54e6cd8f0106d072c90ef98a546afcabb950fdf2c1d363e599049cced97c01d2` |
| Complete rows, every cell | `87c518afa8346b84251d6effba183b738e83a8f210e1c1f703b3bdcef07bffb0` |
| Initialized target/MTP state, every cell | `ddab2adbf2273ff5a0da0380fc932f8541c4fc1871de4738d448d62309e25c6b` |
| Native receipt | `ccb1d37bc82398ec2e5249266f193a468d8c20d8590cf60e122621ccdcefd061` |
| Analysis result | `6aa06cf21cf1d95286631dfa4ea77f872d28d692f6a8876fc29d65d8dbd04deb` |
| Analysis receipt | `1818280fa61392e2270f4f328c488cb681053f713dbede1415b5f84416f59a9d` |

The analysis image is
`nvcr.io/nvidia/pytorch@sha256:2140e699b3beaf7f96a0081fd9c9406bc3832b435cdb60dfa2d261f7d2f34a1c`.
Raw rows, initialized-state hashes/byte counts, traces, helpers, source/archive/library pins
and receipts remain outside Git on Spark B under
`~/scratch/qwen-concurrent-oracle/`; small local evidence is under
`~/scratch/jitllm-m3-qwen-concurrent-oracle-2026-10-04/`.
The fixed oracle/reference stay in their retained `m3lc`/`m3lc2` locations.

Installed GPU-supervised jobs, each timeout 600 and completed/waited zero:
`qwen-concurrent-oracle-build`, `qwen-concurrent-reference-proof`,
`qwen-concurrent-oracle-model2` and `qwen-concurrent-oracle-analysis`.
Replay uses the retained `build_driver.py`, `reference_proof.py`, `run.py`
and `analyze.py` through installed
`~/.local/bin/spark-job start --gpu --name N --timeout 600 -- python3 -B SCRIPT`,
then `~/.local/bin/spark-job wait N`, with fresh owned output paths and
refrozen helper pins. Model and analysis qualification wall times were
235.50 s and 19.35 s; they are not speed measurements.

The first model attempt stopped because the controller applied a deleted
library check to every mapped path, including nonlibrary mappings. Its
failed receipt, controller and manifest are retained, child reaped with
no forced kill and terminal gate passed; no qualification is claimed for
it. The corrected watcher refuses deleted shared libraries and records
nonlibrary deleted paths. Driver, model inputs and engine archives stayed
unchanged; the owned `native2` retry supplies the successful evidence.
