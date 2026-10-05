<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Gemma26 8K ring-cache reference screen

The subsequent [matched prefill profile](../gemma26-prefill-profile/README.md)
preserves these head/choice/native-state identities. Its diagnostic spans locate
most of the difference outside recorded GPU activity; CPU and wait attribution
remains open. The untraced competitive timings below remain the reference result.

With the pinned original reference explicitly using `swa_full=false`, the
unchanged native all1024 policy has zero strict differences across 32
incoming-head greedy choices. Both complete retained heads still differ.
Native paid prefill is 57.6433% slower and decode 0.4880% slower than the
ring-reference bookend mean. No production policy, full-quality, state-restore,
memory or performance gate is selected.

| Complete retained head | Byte-exact | Maximum raw delta | Full-distribution TV | Absolute reference-argmax NLL delta |
| --- | --- | ---: | ---: | ---: |
| Prefill, after 8,192 rows | No | 0.857209206 | 0.034499968 | 0.042114447 |
| Final, after 8,227 rows | No | 0.224399567 | 0.013208429 | 0.014269904 |

The two ring reference runs have identical prefill/final heads and all 32
choices. Fresh native first/repeat and its bookend have identical two heads,
choices and complete initialized state. Those native outputs additionally
match the authenticated historical all1024 result. Only two full vectors are
retained: zero choice differences does not establish all 32-vector agreement,
calibrated margin noise, corpus PPL or complete cross-engine KV-state identity.
The likelihood column uses the reference head's argmax.

| Paid arm, in order | Prefill seconds | 32 decode units, seconds |
| --- | ---: | ---: |
| Ring reference | 2.41568 | 0.684102 |
| Native all1024 | 3.80551 | 0.688280 |
| Ring reference | 2.41232 | 0.685773 |

Native publishes and pays for eight prefill heads; reference requests one
final prefill head. Both publish each completed decode head and pay for the
incoming head's lowest-index argmax before the forced input. Setup, six
discarded warm rows, reset, three common anchors, file writes and the native
state snapshot are outside the paid blocks. The original fusion/graph policy
and existing native timers remain intact.

## Actual cache recipe and policy boundary

Pinned llama.cpp `b29c606e` has public C-API defaults `swa_full=true`,
`kv_unified=false`; CLI/server common defaults instead set `swa_full=false`.
The dedicated [copied client](llama_prefill_ring.cc) explicitly sets both flags
false and closes approved 26/C1/ubatch1024. Actual context-creation logs establish
2,048 local and 16,384 global cells in both new reference runs. Original math
libraries, F16 KV, context16,384, batch8,192 and one sequence remain unchanged.

Native has the same physical capacities, a 1,024-token local window,
position-modulo-local-capacity writes and padded bounded reads. Its unchanged
[benchmark](../../../benchmarks/gemma_prefill.cc) requests the existing compound
policy: norm/RoPE 60, norm/residual 90, routing 30 and scaled reduction 30 selected
steps for prefill and decode. Ordinary products, device masks and zero
shared-vector, row-invariant, generic norm fusion and cache-store choices remain.
These are selected-plan counts, not observed CUDA launches. Uniform native
routing/reduction does not establish original ubatch1024 eligibility parity.
The negative26 packed-C4 comparison does not qualify or disqualify this different
shape by analogy; this screen measures the all1024 configuration directly.

This is the backward transfer of the [dense31 ring-recipe finding](../gemma-swa-ring-h1/README.md).
Its byte-exact final head did not transfer to26. The
[historical full-cache screen](../gemma-prefill-large/README.md) remains a record
of its actual recipe and five strict all1024 choice differences. The new
reference configuration comparison keeps native outputs unchanged but does not
attribute every remaining arithmetic difference. Historical speed/memory
observations do not establish representative ring-cache ratios or parity.
No physical peak measurement was taken. Native charged capacity is
25,868,105,408 bytes, separately recorded in [results](results.json).

Qwen/DeepSeek transfer depends on their own cache/window and attention contracts.
No production default changes, additional policy arms or context ladders follow
from this short screen. Remaining 26 prefill speed and full-head gaps stay open.

## Frozen provenance and reproduction

Source base is `437d59f`. No independently retained historical c3d089 executable
was identified, so the unchanged native benchmark was compiled on the integrated
source and acquired fresh first/repeat before any ring oracle. Its source stays
`8fc6e4b3…`; the resulting native executable `858ce306…` and SDK receipt
`b2174122…` match the preceding H1 build. Source freeze `25e5e05a…` encloses eight
source paths, both binaries, actual build receipt, eight pinned headers, all
four original libraries, canonical IDs, manifest/index and source ancestry.
The receipt has no Git version origin; checksum-authenticated source ancestry,
not that empty version field, identifies the compiled base.

Native own freeze `a1f7cea3…` precedes ring acquisition and records finite full
heads, 32 choices and exact 592,445,440-byte initialized state, layout
`gemma26-f16-kv-scalar-device-v1:16384:1024:16384:2048`.
No older noise allowance was inherited. Production/source-lock/toolchain
content bundle `afc9dc19…` covers 457 files including 421 under `src/`; its
manifest-file SHA is `0751117b…`. The comparator normalizes tuple/list token
identity from the start, tests equal and changed greedy/input/incomplete
streams, and authenticates its own supplied SHA. No analysis-source repair
or measured-source change was needed. [Provenance](provenance.json) gives exact
hashes and all six successful official retirement records.

Measurements ran on Spark-b (`spark-56f5`), NVIDIA GB10, driver 580.178.04 queried
before the bookend. Native SDK `aarch64-c09daba6ac31edee` pins CUDA 13.4.92.
The same immutable original image's historical authenticated environment is
CUDA 13.3.0/CUDART13.3.29-1, not freshly queried here. Task-entry primary
[TensorFold](https://github.com/ashhart/TensorFold) refresh at
2026-10-05T14:25:19Z resolved 609ca419/version 0.6.5; its fresh Gemma recipe remains
26B-A4B MLX-only and supplies no same-format CUDA comparator.

Follow [PROTOCOL](PROTOCOL.md) with approved artifact4ddb360c/indexe7481988 and
the matched Q4_K_M GGUF. Supply canonical 8,227 LE I32 IDs SHA6b6567ca… and
the exact external pinned headers. Checksum-sync sources using `rsync -rlpc`
without timestamps and build only native `jitllm_gemma_prefill` through the
locked SDK and `reference.sh build`. Record the actual source ancestry:
base/head, empty tracked production diff, dirty source-path allowlist and all
eight `SOURCES` byte/SHA records. Supply the authenticated complete production
manifest. No new historical model payload scan is required.

Run `own_freeze.py source ROOT SCRATCH BUILD_JOB` after successful installed
build retirement. Acquire fresh `native-first` and `native-repeat` using
`26 all 1024`, then `own_freeze.py own ROOT SCRATCH NATIVE_JOB SOURCE_SHA`.
After authenticating that exclusive own freeze, acquire exactly
`reference.sh replay ring-first`, native `native-bookend`, and
`reference.sh replay ring-repeat`. Every job uses installed GPU supervision,
timeout 600 and successful official retirement. Never overwrite a freeze.

`compare.py SCRATCH ROOT OWN_SHA BOOKEND_JOB ANALYZER_SHA` checks source/environment,
native head/state repeats, input/choice alignment, actual reference capacities
and repeats, and full finite head metrics. Its `--self-test` checks normalized
equality and rejects changed greedy/input or incomplete streams. Raw heads,
states, IDs, per-step choices and logs remain external. Existing funded pinned
snapshots and whole-owner quarantine are unchanged. Narrow builds and the
source/own/bookend/analysis controls passed; diagnostic workflow scope omits a
routine unit suite.
