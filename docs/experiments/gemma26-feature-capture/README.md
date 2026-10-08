<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Gemma26 retained-frontier capture ahead (2026-10-08)

Explicit retained-frontier capture reduces prefill by 0.484% in this short
same-native screen, with exact heads, features, choices and initialized state.
Focused controls also prove exact three-step assistant proposals after captured
target prefill. Ordinary capture remains off/one future, including successful
`SetupAssistant`: this qualifies an internal experimental target policy, not
assistant serving, default adoption or competitive reference parity.
The preceding [plain-prefill negative screen](../gemma26-capture-ahead/README.md)
is a separate result and remains unadopted.

| Seconds | Off/1 O1 | On/2 A1 | On/2 A2 | Off/1 O2 |
| --- | ---: | ---: | ---: | ---: |
| Paid prefill | 2.636070 | 2.630280 | 2.618610 | 2.638340 |
| Paid 32 forced units | 0.663829 | 0.665670 | 0.661614 | 0.662936 |
| Sum of paid endpoints | 3.299899 | 3.295950 | 3.280224 | 3.301276 |

Prefill means are 2.637205 → 2.624445 s (−0.48385%, 12.760 ms).
Off range 2.636070–2.638340 s and on range 2.618610–2.630280 s do not overlap;
off bookend movement is +0.08611%. The paid sum changes −0.37874%.
Decode changes +0.03912%, within variation; there is no decode gain claim.
This is n=2 per arm, not sustained or all-context evidence. O1 began idle at
208 MHz and ended at 2411 MHz; subsequent endpoint samples were 2411 MHz,
with temperatures 38–47°C. Six discarded warm rows precede paid work.
Endpoint samples do not prove continuous clocks or clock locks.

## Contract and measured work

The runner permits future capture for non-verify, non-greedy frontier execution
with exactly one retained feature per owner. All-output/all-feature paths stay
excluded. Capturing the future graph executes neither its feature writer nor
KV stores. Actual `Queue` or replay keeps the existing fresh D2D feature copy
outside the captured plan, with unchanged completion/quarantine handling.
Full feature IDs, complete shape identity and layout checks remain in the
existing current/next/after protection and independently funded group slots.
A read-only borrowed epoch accessor supports authenticated diagnostics.

Both measured arms retain features. Diagnostic context is 16384, prefill 8192
in eight 1024-row chunks, then three untimed anchors and 32 paid forced units.
This does not qualify public 8K admission. Retained features require a frontier
head on every chunk: seven intermediate heads are paid, even though the prompt
requests state-only. Both arms use the same 26,125,944,576-byte budget, including
5,007,104 bytes for two temporary plan grants and an independently funded
11,264-byte feature snapshot. No arm moves planning, capture, state growth or
feature D2D publication out of its paid endpoint.

Each on arm builds/caches one future pair and captures/replays six future
chunks; off arms select zero pairs, ahead captures and prefill replays.
Every built future installs, with no refusal or dropped graph. Complete
262144-vocabulary prefill/final F32 heads are finite and byte exact across
all four arms. The complete 2816-value feature rows at positions 8192 and 8227
also match exactly, with authenticated borrowed epochs 11 → 46 (35 actual
anchor/forced units). Feature snapshots and generation checks run outside
paid endpoints through catalog-funded pinned storage. All 32 incoming argmax
choices, forced IDs and anchors agree. All 592,445,440 initialized state bytes
in 60 ranges at position 8227 match SHA-256
`83a2bb16922d212723ba9354d4707e4f39fd823282e74d2203f023e14c6ed787`,
layout `gemma26-f16-kv-scalar-device-v1:16384:1024:16384:2048`.
[Aggregate results](results.json) retain all endpoint, payload and budget identities.

## Focused qualification

Four prerequisites and five later focused executions pass (eight distinct
cases; the defaults case runs in both sets). All affected runtime/benchmark
targets compile. The later build changes only the two test targets; measured
production source and benchmark ELF remain unchanged.

The new real target test clears state/plans and compares an unhinted ordinary
execution baseline with a hinted round on the same explicit opt-in runner.
Three 256-row chunks change KV buckets, proving actual pair construction,
ahead capture and replay only in the hinted round. Full heads and features
after every chunk and initialized state at 768 match. A captured-feature borrow
refuses same-slot mutation while a peer progresses, preserving full feature/KV
bytes. Spill/restore preserves KV but invalidates features; fresh progress to
769 restores borrowability and matches the ordinary continuation head, feature
and state. This no-hint execution baseline is distinct from the immutable
option-off/on timing factor above.

The new actual assistant test uses vocabulary-authenticated target/component
artifacts. After equivalent unhinted/hinted target rounds, three endogenous
joined C2 proposals have byte-exact full heads/features. One owner has the
captured 768-row prefix and the peer an ordinary independent five-row prefix.
Borrow checks, cursors and complete target KV/feature witnesses remain exact.
Clearing both plan caches and resetting the first recurrent input prevents
cross-round reuse from masquerading as equivalence. This proves consumption of
captured target features; it does not measure assistant speed or two-owner
joined target prefill. Existing frozen-borrow/peer/spill and failed-copy pinned
ownership tests supply unchanged failure controls; defaults remain off/one.

Both installed supervised jobs and the four-process timing job retired with
exit 0, positive post-teardown markers and empty GPU. All nine test executions
are positive, without failures, errors, skips or disabled cases. No new kernel
warning/error occurred during timing. Whole regression and shipment gates
remain deferred under the owner override. Final local REUSE and embedded-header
checks cover 1918 files with zero problems; the portability boundary check covers
421 files with zero problems. All three new report files are included, alongside
pinned formatting, Markdown transfer/plan consistency, link and JSON checks.

## Provenance and replay

Base is `5de4e24`. Measured source manifest is
`be328fc83fe0b91071d1bd202cdb204d24ae0898f5b74f98a3435eea98ef0f12`,
benchmark ELF `97bfac7f069513c4a1d5b044f2289c191796137bffdda3d5a9c358f11c0624ea`.
The qualification manifest additionally binds new tests and assistant sources;
[results](results.json) retain exact source, receipt, ELF, actual first-resolved
cuBLAS/Lt, method and positive evidence identities.

Use the approved artifact
`4ddb360c9ce08f1e984ab304b6af918be44246d52346734066b06443f7c249d3`,
index SHA `e748198836025cc2d1bc8b0b61a1d39dc1eefdc2d34feb158a425f6f913f1171`,
and canonical 8227-token/32908-byte LE I32 fixture SHA
`6b6567ca51a3fbe5000521cb71fcf168ef485623bdbfea2abab30d57f414d96b`.
The preceding report gives the reproducible corpus/tokenizer inputs:
`llama_prefill RAW_GGUF CORPUS NEW_INPUT prepare 1024` must reproduce that SHA.
Keep this standing fixture or its preparation sources when raw outputs are deleted.

Build the existing `jitllm_gemma_prefill` under the pinned Spark SDK. Run four
fresh processes off/on/on/off with identical executable, receipt, actual
first-resolved cuBLAS and budgets, using a different NEW_OUT for each arm:

```text
jitllm_gemma_prefill ARTIFACT IDS NEW_OUT 26 all 1024 normmul-off state-only lookahead-on phases-off state-chunked capture-ahead-off features-frontier
jitllm_gemma_prefill ARTIFACT IDS NEW_OUT 26 all 1024 normmul-off state-only lookahead-on phases-off state-chunked capture-ahead-on features-frontier
jitllm_gemma_prefill ARTIFACT IDS NEW_OUT 26 all 1024 normmul-off state-only lookahead-on phases-off state-chunked capture-ahead-on features-frontier
jitllm_gemma_prefill ARTIFACT IDS NEW_OUT 26 all 1024 normmul-off state-only lookahead-on phases-off state-chunked capture-ahead-off features-frontier
```

Require actual seven intermediate heads, retained-feature selection, positive
candidate pair/ahead/replay and zero baseline counters, no refused/dropped
future, finite positive endpoints, both full heads/features and all choices,
complete initialized state/layout, feature cursor/epoch/generation checks and
matched full budget receipts. Compare every cross-arm payload before speed.
Report individual ranges, means `(O1+O2)/2`, `(A1+A2)/2`, delta
`100*(on_mean/off_mean-1)` and bookend `100*(O2/O1-1)`; do not pool methods.
Prefill/decode endpoints exclude anchors and diagnostic snapshots.

Use installed Spark GPU-exclusive supervision, native deadline 145 s,
batch 600 s/grace 30 s/stop on first failure; propagate signals through
owned child-group kill/wait. Require empty GPU and at least 16 GiB MemAvailable
before/after, same boot, new kernel cursor per arm and no NVRM/Xid/UVM/OOM.
Authenticate source/ELF/receipt/IDs/artifact metadata/libraries before/after and
prove native exit 0, teardown and supervisor retirement. Raw logs, heads,
features and states stay external under `scratch/m35-gemma26-feature-capture`
until M3.5 closes; normalized results and replay inputs remain reproducible.

## Comparator and remaining transfers

Task-entry TensorFold native
[`f8fe17d2`](https://github.com/ashhart/TensorFold/blob/f8fe17d24629aedabf90bbf78279dd776e6d62e7/README.md)
(v1.0) marks Gemma4 under qualification; retained Python
[`ed78d6fc`](https://github.com/ashhart/TensorFold/blob/ed78d6fc204d89d90b045bf033d6551e7714f3a1/README.md)
(v0.6.6) has an MLX-only Gemma26 recipe. Neither supplies this approved CUDA/GGUF
comparison. Existing llama.cpp remains applicable; no new competitive parity
claim follows from this same-native optimization.

G26's composite T23 remains open for default policy adoption, joined target
prefill and broader contexts/output contracts. Its plain screen remains neutral;
its retained-frontier opt-in now has bounded performance and actual assistant
consumption evidence. G31's same compatible feature contract is unmeasured and
reopens composite T23, preserving the prior plain-neutral rejection. Gemma2/Gemma3
retain their adopted plain policies, with no new feature claim. DeepSeek and both
Qwen formats remain open lookahead/group/capture recipients under their own
contracts; Qwen-Image has no token-chunk prefill. No future state is initialized.
