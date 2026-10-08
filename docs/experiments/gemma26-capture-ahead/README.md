<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Gemma26 plain-prefill capture ahead (2026-10-08)

Capture ahead works on the Gemma26 diagnostic recipe but does not improve this
8192-row prefill screen meaningfully. The runner context is 16384, followed
by three anchors and 32 forced units; this is not public 8K admission or parity. Ordinary Gemma26 and Gemma31 retain capture off and
one future plan. The two-slot shared hook remains available internally for
further compatible qualification; this result does not establish neutrality
at every context, row size, cohort or feature/assistant contract.

| Seconds | Off/1 O1 | On/2 A1 | On/2 A2 | Off/1 O2 |
| --- | ---: | ---: | ---: | ---: |
| Paid prefill | 2.50768 | 2.50015 | 2.49719 | 2.49427 |
| Paid 32 forced units | 0.663709 | 0.663785 | 0.663140 | 0.665748 |
| Sum of paid endpoints | 3.171389 | 3.163935 | 3.160330 | 3.160018 |

Mean prefill changes from 2.500975 to 2.498670 s (−0.0922%, 2.305 ms),
inside the off-arm bookend movement (−0.5348%, 13.410 ms). Off range is
2.49427–2.50768 s; on range is 2.49719–2.50015 s. Decode changes −0.1905%
and the paid sum −0.1128%, also within this short n=2 screen's variation.
No speedup, tail repair, sustained result or reference parity follows.

This is an actual capture comparison: each on arm builds and installs one
pair of future plans, captures six future graphs and replays six of eight
paid prefill chunks. Each off arm has zero pairs, ahead captures or prefill
replays. All arms build/cache seven future plans, with zero refusal and zero
dropped graphs. Full prefill and final F32 heads are finite and byte exact
across all four arms; all 32 incoming argmax choices and supplied forced IDs
agree. All 592,445,440 initialized state bytes (60 ranges, position 8227)
match SHA-256 `83a2bb16922d212723ba9354d4707e4f39fd823282e74d2203f023e14c6ed787`.
The complete layout is
`gemma26-f16-kv-scalar-device-v1:16384:1024:16384:2048`.

## What changed

Gemma4 now composes the existing independent funded lookahead slots through
`PrefillLookaheadGroup<Gemma4Planned, 2>`. The scalar serving adapter forwards
its existing next/after hint. Complete current/next/after keys are deduplicated
and borrowed before any optional graph or plan charge; the whole next cohort
is authenticated before deriving after positions. Both CPU planning choices
are resolved on the driver, then plans are built sequentially during current
device work. Binding, coverage and near-to-far insertion require successful
current completion. Each slot retains independent refusal/cleanup.

Only plain non-greedy, non-verify, frontier/state-only execution without retained
features may capture ahead. Layout mismatch drops an ahead graph; unknown
capture completion quarantines current state through the existing job path.
Feature/assistant and verification retain their execution path. Off/1 predicts
only the next chunk, preserving Gemma31's existing policy. No future state,
public setting, context limit, batching admission or production default changes.
Bounded descriptor vectors use the existing startup-funded 1 MiB host slack;
each future plan takes its optional allowance before plan allocation.

The benchmark adds an internal trailing capture switch and binds actual
prefill pair/capture/replay counters. Both arms fund two temporary maximum
plans and the same retained-plan/graph allowance. Hints name only later actual
calls, so the actual-call bound also bounds retained keys. All four complete
budget receipts agree. This funding makes no performance arm privileged.

## Method and checks

Spark A ran the pinned `aarch64-c09daba6ac31edee` SDK. Source base is `81d1a78`,
with the five production/test/benchmark files recorded in [results](results.json).
Qualified source manifest SHA is
`314703d125138c2595f49c8c18799503b1422fce75828933aa02f8046aa390fd`;
benchmark ELF SHA is
`6971d7e88d677561aa8baae32bb46e40441fb9e9dd3d218d5dca672fb32a20a5`.
The official receipt and actual first-resolved pinned cuBLAS/Lt identities are
recorded in results, rather than inferred from a library directory.

Twelve unique focused controls pass: nine shared optional slot/group lifetime,
funding/build/install/refusal tests; one new option-capacity/default refusal
control; the existing Gemma4 defaults control; and the real Gemma31 state-only
head/state continuation/replay control. All five affected targets compile,
including runtime serving. The first supervised job failed only in final
binding because the controller named nonexistent `bin/llmp-runtime`.
All build/test children had already exited 0. A separately supervised bind-only
correction authenticated their original records/XML and the correct
`src/runtime/llmp-runtime`; it retired successfully without repeating tests
or compilation. The failure is retained separately, not relabeled as a pass.
Source/method review and independent adversarial review preceded the jobs.

The four-arm installed GPU-exclusive screen retired successfully, with
observed native exit 0, post-teardown markers and empty GPU after every arm.
No new kernel warnings/errors occurred. Before/after GPU samples reached
2411 MHz after O1; O1 began idle at 208 MHz. These are endpoint samples, not
continuous clocks or clock locks; six discarded warm rows precede paid work.
Exact telemetry, all endpoints/ranges, budgets and payload identities are in
results. The full regression suite remains deferred. This negative screen
adds no default policy, so an adoption qualification ladder was not run.

Task-entry TensorFold was native
[`f8fe17d2`](https://github.com/ashhart/TensorFold/blob/f8fe17d24629aedabf90bbf78279dd776e6d62e7/README.md)
(v1.0, Gemma4 under qualification) and retained Python
[`ed78d6fc`](https://github.com/ashhart/TensorFold/blob/ed78d6fc204d89d90b045bf033d6551e7714f3a1/README.md)
(v0.6.6, Gemma26 via MLX only). Neither supplies this approved CUDA/GGUF
recipe. Existing llama.cpp remains the applicable competitive reference;
this same-native optimization preserves its own baseline exactly and makes
no fresh cross-engine quality/performance claim.

## Reproduction without disposable controllers

Use approved prepared artifact
`4ddb360c9ce08f1e984ab304b6af918be44246d52346734066b06443f7c249d3` and its
index SHA `e748198836025cc2d1bc8b0b61a1d39dc1eefdc2d34feb158a425f6f913f1171`.
Supply the retained canonical 8227-token fixture, 32908 LE I32 bytes, SHA
`6b6567ca51a3fbe5000521cb71fcf168ef485623bdbfea2abab30d57f414d96b`.
It can also be regenerated from the full trimmed War and Peace corpus SHA
`c7156148ecaa12b6416cf816540d8dede2014982554a835e61076f0dd8bf0c2d`
using the [corpus map/build helper](../long-context/corpus.json) and
[existing pinned tokenizer preparation](../gemma-prefill/llama_prefill.cc):
`llama_prefill RAW_GGUF CORPUS NEW_INPUT prepare 1024` emits its first 8227 IDs
with explicit BOS and parse_special=false. Authenticate the ID SHA before use.
Raw measurement cleanup must preserve this standing input or its reproducible
corpus/tokenizer sources.

Build `llmp_gemma_prefill` under the pinned Spark toolchain. In one GPU-exclusive
installed supervised job, run four fresh processes in order off/on/on/off,
with identical ELF/receipt/actual first-resolved cuBLAS and fresh output names:

```text
llmp_gemma_prefill ARTIFACT IDS NEW_OUT 26 all 1024 normmul-off state-only lookahead-on phases-off state-chunked capture-ahead-off
llmp_gemma_prefill ARTIFACT IDS NEW_OUT 26 all 1024 normmul-off state-only lookahead-on phases-off state-chunked capture-ahead-on
llmp_gemma_prefill ARTIFACT IDS NEW_OUT 26 all 1024 normmul-off state-only lookahead-on phases-off state-chunked capture-ahead-on
llmp_gemma_prefill ARTIFACT IDS NEW_OUT 26 all 1024 normmul-off state-only lookahead-on phases-off state-chunked capture-ahead-off
```

Each process discards six weight-warm rows, clears, and pays all planning,
capture and chunked state growth for 8192 rows in eight 1024-row chunks.
Only the final prefill head is requested. Three untimed anchors
`[236761,108,236913]` precede 32 timed forced-prefix one-row units; incoming
argmax scans are paid and do not choose subsequent inputs. Prefill and decode
seconds are separate; their sum excludes anchors and snapshots. After both
timers, authenticate two complete 262144-vocabulary finite F32 heads, every
choice/forced pair and the complete initialized state/layout. Compare all four
byte identities before interpreting speed. Require finite positive endpoints,
matched budget receipts, candidate positive pairs/ahead/replays, baseline zero,
no refused/dropped future, positive teardown marker and observed exit 0.

Require empty GPU and at least 16 GiB MemAvailable before/after each arm,
same boot, new per-arm kernel cursor and no NVRM/Xid/UVM/OOM evidence. Bound
each native process at 145 s and the installed batch at 600 s, stop on failure,
raise supervisor signals through owned child-group kill/wait cleanup, and
positively wait for supervisor retirement. Report both individual ranges,
means `(O1+O2)/2` and `(A1+A2)/2`, delta `100*(on_mean/off_mean-1)` and off
bookend movement `100*(O2/O1-1)`. Do not pool different methods or add reruns.
Aggregate results/provenance remain in Git; raw logs, heads and 592 MB states
remain external under `scratch/m35-gemma26-capture-ahead` until M3.5 closes.

## Compatible feature extension and remaining gates

The retained-feature/assistant consumer is not covered by this negative
plain result. The subsequent [retained-frontier control](../gemma26-feature-capture/README.md)
qualifies an explicit one-feature-per-owner option with modest target gain and
exact assistant consumption; default adoption and broader contracts keep
G26's composite T23 cell open. Source inspection shows
a concrete compatible route: the plan already writes normalized features,
and ordinary second-run graph replay leaves the caller's feature D2D copy
after `Queue`. Capturing a future graph does not execute it or its feature
writer, so the actual future unit can keep that same ordered D2D copy and its
existing completion/quarantine checks. Its complete key already records
`feature_outputs`, and `Layout` includes the fresh feature-ID input.

The bounded extension uses one frontier feature row per owner,
with the current head/feature publication envelopes, independently funded
plans/graphs and current/near/far key protection. It continues excluding
all-output/all-feature, verification and greedy execution. Full feature bytes,
heads, initialized KV, borrowed-peer identity and feature-copy failure ownership
are covered by the subsequent focused controls. Their evidence is separate from
this plain negative screen; T23 remains open for policy adoption and broader
compatible envelopes.
