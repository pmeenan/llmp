<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Current Gemma4 serving reference

This screen compares the ordinary corrected Gemma26 and dense31
8K/C1/C4 serving recipes against original llama.cpp v0.6.0/d812 public backend
greedy sampling. Historical CPU-head stock results retain their pins and
publication/reset policies. No production arithmetic or recipe changes here.

## Actual quality and short paid cycles

Spark B (GB10), NVIDIA driver 580.178.04, 2026-10-07,
base `fed648c`; the final report preserves current Gemma3 serving additions from `e4b7392`. Each quality job completed before that
workload's timing. All 1,290 complete 262144-element heads below are byte-identical
to stock, with zero positive-margin/tie/argmax differences and zero conditional
loss delta. Every stock quality history is also its natural greedy history.
Native first/repeat complete heads, initialized state and history are exact;
stock's independent first/repeat complete heads and histories are exact.

| Profile / owners | Exact full heads | Scored targets | Mean target NLL, both engines |
| --- | ---: | ---: | ---: |
| gemma26 / 1 | 129 | 128 | 0.1171729763 |
| gemma26 / 4 | 516 | 512 | 0.1165692847 |
| gemma31 / 1 | 129 | 128 | 0.1270158000 |
| gemma31 / 4 | 516 | 512 | 0.0230142821 |

The C4 final decode plans record rows 4/segments 4 and 30/60 owner-attention
selections for Gemma26/dense31. Native norm/RoPE and norm/ADD selections are
60/90 and 120/120 respectively. These are actual last-built plans; replay
counts are not inferred from them.

The table reports n=2 fresh processes per engine, separately for the second
and third paid phases; values separated by `/` retain that order. All warm,
second and third cycles preserve each owner's 129 emitted choices and 8191
history IDs against the frozen quality result, including both stock bookends.
Full-head/state export belongs to the separate quality run.

| Profile / owners | Native cycle seconds, second / third | Stock cycle seconds, second / third | Native latency delta, second / third |
| --- | ---: | ---: | ---: |
| gemma26 / 1 | 4.970812 / 4.910356 | 4.959429 / 4.971499 | +0.2295% / -1.2299% |
| gemma26 / 4 | 14.249751 / 14.112735 | 14.141084 / 14.151138 | +0.7684% / -0.2714% |
| gemma31 / 1 | 23.762629 / 23.638147 | 23.663719 / 23.668452 | +0.4180% / -0.1280% |
| gemma31 / 4 | 60.158033 / 59.594202 | 59.612856 / 59.661907 | +0.9145% / -0.1135% |

These are short resident serving-driver comparisons with the stated ordinary
stock cache and sampler policy. They do not establish sustained performance,
peak-memory qualification or actual HTTP per-unit overhead. No production
arithmetic or option changed for this comparison.

## Reproducible caller and provenance

The native benchmark target is `llmp_gemma26_frontier_control`, with
`LLMP_BENCH_FULL_FINAL_FFN` and `LLMP_BENCH_PHASES` unset. Invoke its
`proof quality OWNERS INPUT OUT --config CONFIG --anchor NEW_ANCHOR` mode
before `proof cycles` with the same arguments. Both use ordinary resolved
settings and private fresh configurations. The constructor stamp must report
`full_final_ffn=1`, slots 1 or 4 and matching head capacity.

Build [llama_cycle.cc](llama_cycle.cc) as a C++23 ARMv8-A caller against the
frozen release headers and retained original-image `libllama/libggml/libggml-base`
closure, then run it inside that unchanged image. Its arguments are
`MODEL INPUT NEW_OUT gemma26|gemma31 1|4 quality|cycles [NATIVE_FIRST]`;
quality requires the already frozen native first-cycle output directory.
Use the installed supervised GPU job wrapper with owned container retirement.
Never append timing to a failed quality gate. The reusable [analyzer](analyze.py)
actions are `native ROOT OWNERS`, `quality ROOT OWNERS`,
`arm ROOT OWNERS R1|N1|N2|R2` and `timing ROOT OWNERS`; each arm is checked
immediately before the next process starts.

The native build is `spark-native`/`RelWithDebInfo`, SDK
`aarch64-c09daba6ac31edee`, with the pinned GGML/CUDA source lock.
A full 2081-path checksum source inventory and empty itemized sync dry-run
bind the warm build. Actual native proof SHA-256 is
`cbafc9deb7b100cf5fbab2050568bb1a8fc142731a01a3d54e68fbeaa7a958f3`;
stock caller is
`c07d4d72e452c550a6829e33cf1ecb44d99727eddd477f05790bb8ce24cd629f`;
build receipt is
`874aaf7a5967cfbe91054e0d8fc1a0630f952e0e8eb54b831d09719f1e08ce89`.
The caller source is
`d25c4929015805b4b856484404d8a33ede01e458ac24d134d2366434aa7738e6`.
Prepared manifests bind the previously authenticated raw model SHA/size;
current file inode/mtime/size are rechecked before every process. This unit
reuses approved payload identity, rather than claiming a new full-payload audit.
All installed jobs completed successfully; each owned container retired before
its analysis.
Raw logs, full heads/state, method/source inventories and aggregate receipts
remain external under `m35-gemma4-backend-greedy/run1` on Spark B and
`/tmp/llmp-m35-coordination/gemma4-backend-greedy-raw` on the workstation.

The native caller is `benchmarks/gemma31_production.cc`, through its existing
constructor-check target with the control environment unset. It must witness
full final FFN, ordinary exact options and head capacity equal to configured
slots. Quality mode has independent first/repeat cycles and publishes 129
complete heads per owner, initialized state and completed history. Native own
byte equality and finite publication precede stock comparison. C4 must record
four actual rows/segments and selected joined owner attention in its final
decode plan; these are plan selections, not per-replay kernel counts.

The tiny original-public-API `llama_cycle.cc` loads the approved raw GGUFs
already authenticated for the prepared artifacts. It uses context8192 per
sequence, C1/C4, independent sequence caches, F16 K/V, `swa_full=false` and
`kv_unified=false` (common CLI/server cache policy), 1024/256 prefill chunks
for Gemma26/dense31, stock fusions/graphs and a separate ordinary greedy chain
for every owner. Each owner prefills its own 8063 tokens; C4 decode submits
four scalar rows together. No extra synchronization follows state-only chunks.
Every phase uses logical memory clear with `data=false`, matching ordinary
llama-bench's reset rather than paying a full KV memset.

The ordinary d812 greedy chain computes GPU argmax but retains full
`data.logits`; `llama-graph.cpp` exports `t_sampled_logits` and the context
copies them independently of the separate `needs_raw_logits` guard. The
caller records `129 * owners * 262144 * sizeof(float)` sampled-logit bytes per
cycle. This is an unmodified stock backend-greedy reference with full sampled
logit transfer, not a token-only stock traffic claim. No custom output filter,
sampler shim, kernel patch or controller observer is used.

Stock quality repeats the frozen native generated history as supplied input,
recording its own argmax at every head. All 129 full heads remain comparable
even if stock disagrees; only exact choice agreement establishes natural stock
lineage. The existing zero-positive-margin and 3% conditional-loss rules apply,
with all choice IDs additionally required to agree before timing. The first
128 heads score native generated targets0..127; the last pending-token head
is unscored. Failure remains a failure and stops that workload's timing.

Timing uses R/N/N/R, two fresh processes per engine. Each process runs the
original warm/second/third 8063-prompt/128-wave cycles. Every owner emits129
choices, consumes128 and retains8191 history IDs. Native `cycles` has no
per-decode logits callback and returns before final-head/state export; stock
uses its ordinary sampled-token API, with its actual internal sampled-logit
copies explicit. Neither caller adds a paid final verification-head request.
Native initial prompt frontier publication remains part of the actual serving
path. Both paid phases must preserve the frozen quality histories and all
same-policy repeats. Timings include logical reset/prefill/decode and ordinary
per-cycle completion; startup, warm phase and output-file writes are excluded.

The dense31 C4 four-arm timing takes about12 minutes total and is needed now
to settle current-reference qualification of the affected four-slot recipe.
Each arm runs as a separate installed supervised job of at most600 seconds,
with official completion/retirement before the next. The other workloads stage
Gemma26 C1, Gemma26 C4 and dense31 C1 first. This n=2 screen makes no sustained
throughput claim. Full suites and historical trace/archive audits are not part
of this focused comparison.

Supply the external replay input from
`~/.local/share/llmp/gemma31-serving-bridge/inputs.i32` on the owned Spark
or an independently supplied file with the same identity. It contains four
little-endian 8192-ID histories (131072 bytes), SHA-256
`90bf82fd26b7040c513e7370eca2cd283ac8fd15ce89a2e8d05d6f7e670b5ba2`.
Approved artifacts remain Gemma26
`4ddb360c9ce08f1e984ab304b6af918be44246d52346734066b06443f7c249d3`
and dense31
`32c92e077a6816b54aa988e2dee61a3639c958fd510ea99e25f3621f10b2aa08`.
The original image is
`ghcr.io/ggml-org/llama.cpp@sha256:c604ea4f1c2e8d5c8b27d89fef727384d59e23c5b07e369cde5393820e0607db`.
Runtime/source inventories, actual build hashes, model identity/unchanged-file
bindings and owned container retirement are retained externally.

Per-task TensorFold inspection at this screen's start on 2026-10-07 resolved
[`ed78d6fc204d89d90b045bf033d6551e7714f3a1`](https://github.com/ashhart/TensorFold/tree/ed78d6fc204d89d90b045bf033d6551e7714f3a1).
Package version is 0.6.6. Its documented Gemma26 path is MLX; there is no matching Gemma26/dense31
CUDA GB10/GGUF recipe. This is an applicability check, not a TensorFold timing
claim, and does not alter historical pins.

Actual HTTP driver overhead remains a separate gate: this production proof
selects its cohort once per cycle, whereas HTTP selects it each completed
unit. Reference parity here cannot establish that per-unit overhead is closed.
