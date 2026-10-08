<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# DeepSeek target GPU masks — 2026-10-07

The target's raw causal/ring F16 masks now use the existing GPU producer.
On one repeated two-owner ring screen, prefill falls from 11.47395 to
11.39495 seconds (−0.689%) and the paid cycle from 12.41565 to 12.33465
seconds (−0.652%). These are two samples per policy, with nonoverlapping
prefill/paid ranges; decode overlaps and earns no speed claim. Every complete
head, greedy choice and initialized-state hash agrees exactly. The first
screen had substantial host-baseline drift and does not support its much
larger apparent gain. The two screens are reported separately below.

This ports one existing optimization to an existing M3 model. It establishes
no new competitive parity, maximum-context, corpus or quant qualification.
The accepted M3 speed gaps and separate reference gates remain as recorded.
DSpark's draft-block masks and the optional CSA/HCA/LID visible-count masks
remain separate open transfers.

## Contract and shared mechanism

`CausalRingMask` extends the existing `Gemma4Mask` constructor with an
explicit authenticated `Exact` versus `Pad32` row layout. The legacy producer
tag and CUDA arithmetic are unchanged; Gemma2/3/4 keep their padded
contract. DeepSeek takes exact contiguous query rows because `Dsv4SparseMask`
reads the raw mask's `ne[1]` as its real query count. Padding it to 32 would
change sparse consumer shapes.

`engine/graph_mask_inputs.h` shares source authentication, funding policy
and host-reference staging. It checks all producer parameters, actual parent
identity, unique node membership and absence from host inputs, with no extra
source aliases. The DeepSeek adapter additionally checks real positions,
state indices, context, raw width and complete scalar/joined position extents
before allocating embedding or joined payloads. A stale enlarged scalar
positions descriptor is refused before staging can read beyond its host
vector. Host reference source order remains unchanged.

Ordinary `Dsv4Options` select target device masks. Low-level `Dsv4Model`
plans retain their explicit host-reference default; the factor harness sets
its policy explicitly in either arm. Raw host construction is independently
skippable from compressed-mask construction. Source and activation grants
precede payload allocation; the GPU mask remains a funded activation, so
this result makes no overall memory-reduction claim. Captured graphs consume
fresh staged absolute positions and each joined segment's actual offset.

DSpark does not share the target formula: its entire current draft block is
visible, including later positions in that block. Its mask remains on the
host, and target-only device options are refused by draft constructors.
Compressed masks have visible-count/top-k semantics and remain unchanged.
No target result closes these drafter/compressed audit items.

## Matched factor screen

Spark A (`spark-c4e2`, GB10), SDK `aarch64-c09daba6ac31edee`, pinned CUDA
13.4/cuBLAS 13.8. The same native binary runs host/device/device/host in
fresh processes. The prepared M3 UD-Q2_K_XL artifact is
`8a355bfb27c90e1150fbd7fa62ea6e63f6bf34fcca33934e52d22773f1508234`
(`unsloth/DeepSeek-V4-Flash-0731-GGUF@fbbb5b93`); there is no drafter in
this factor. Serving's output-A/HCA policy and frontier head are enabled,
context is 8,704, and chunks use the production 4,096-row setting rather
than the harness's 512-row default. Raw ring capacity is 4,352, window 128.

Two actual native-tokenized texts supply 4,352 and 4,608 IDs. Preparation
checks vocabulary 129,280, BOS 0 and the checkpoint's disabled automatic
BOS/EOS flags, then explicitly prepends BOS once as production serving does.
Complete ID SHA-256s are
`d74255dd4e8ae5d806824e2378007d1599f8115e8ac80d886b60a865350ecadd`
and `b59367a57eeba410e0ce051780164d1ef8b414a64f2d195e1b855759f04e145d`.
The first owner reaches the ring end at its prefill frontier and wraps on
its first decode row; the second already wraps during prefill. Independent
prefills are followed by 16 actual joined C2 greedy decode steps.

Paid prefill includes Clear, state growth, planning, input construction,
staging, GPU completion and frontier-head publication. Paid decode includes
frontier retention, greedy scans, all joined work and full-head retention.
Setup/load, initialized-state copying/hashing and output serialization are
outside those intervals. There is no excluded model-execution warm-up.
Diagnostic output storage and pinned state-copy storage are charged.

| Screen / arm | Prefill s | Decode s | Paid s |
| --- | ---: | ---: | ---: |
| Initial H1 | 13.3003 | 0.969791 | 14.2701 |
| Initial D1 | 11.3774 | 0.938805 | 12.3162 |
| Initial D2 | 11.4281 | 0.940835 | 12.3689 |
| Initial H2 | 11.4488 | 0.939856 | 12.3887 |
| Repeat H1 | 11.4891 | 0.940831 | 12.4299 |
| Repeat D1 | 11.3591 | 0.937811 | 12.2969 |
| Repeat D2 | 11.4308 | 0.941623 | 12.3724 |
| Repeat H2 | 11.4588 | 0.942513 | 12.4014 |

The initial host bookends move 13.92%; their apparent mean prefill reduction
of 7.85% is drift-dominated. The identical repeat reuses the exact binary and
ID payloads, with fresh outputs and no rebuild. Its host bookends move
0.264%, and its prefill ranges are 11.4588–11.4891 host versus
11.3591–11.4308 device. The 28 ms separation is a bounded screen, not a
confidence interval. No first-screen cause is established and the samples
are not pooled. Existing telemetry finds the repeat's sole compute process
owned by its installed supervisor; observed SM clocks are 2,398–2,457 MHz.
Those snapshots are not per-kernel timing attribution.

All eight arms have 34 complete finite F32 vocabulary rows, 32 identical
natural choices, identical initialized range identities and state hashes.
The full head payload hashes to
`ce4ed9944b578d6016c054ef38ea9d83e2f067210388ce89b211b559ab894977`.
Device arms select six mask-producing plan steps; host arms select zero.
Each arm actually captures one joined decode graph and replays it 14 times.
The counter records bound plan selections, including the kernels retained
inside captures, rather than counting graph-replayed kernel invocations.
All arms retire successfully with no memory coverage violations.

## Checks and provenance

The narrow Spark build and 55 focused checks pass: 42 affected Gemma plan
controls, seven DeepSeek input/default controls, two DeepSeek graph/ring
controls and four complete-buffer GPU mask controls. They cover exact rows,
nonzero owner offsets, every relevant ring byte, captured fresh positions,
canaries, stale/extra parents and invalid row-layout refusals. A separate
adversarial source challenge is clean. Compilation and synthetic-fixture
corrections were resolved before the factor; a mistaken preparation BOS
assumption was corrected to the actual production behavior before inference.
Their failed installed-job records remain available. No full regression
suite runs during this transfer.

The production default then passes the existing four-row DeepSeek/DSpark ↔
Qwen3.8/MTP pair: 8,192 input tokens, 16-token continuations, two cycles and
zero-context rows disabled. Both actual DeepSeek initialized-state restores
and every continuation token/full logit row equal their unswapped controls;
all four output checks pass. The diagnostic records ten actual target
mask-producing plan steps, 28 completed DSpark draft graph steps, 786,432
bytes of drafter state and 22 graph replays on DeepSeek returns. The
prepared return retains four graphs and replays 14. This checks the target
mask default during actual draft/verify, rollback and state restoration;
it does not qualify a changed DSpark mask or establish a swap speed ratio.
The default-focused seven-test input/plan suite also passes again. Installed
jobs `m35-ds4-mask-production-build1` and `m35-ds4-mask-production-run1`
complete zero and are waited on; the latter takes 102 seconds. Source
manifest `03402329cd6411d897b76afbee6a034f60349efab9877ffeeb704126d3aecc9f`
and method `8ed7e199d8002df0ce560d716fd37c4bed2952378c6589e5ff2620b68723c1d2`
bind that qualification. Raw records are at Spark A
`~/.local/share/jitllm/ds4-mask-production-check1/`, with workstation copies
in `/tmp/jitllm-m35-coordination/ds4-mask-production-evidence1/`.

The final candidate is integrated on main `763bcbd`, preserving the completed
Gemma2/Gemma3 prefill changes. A narrow combined Spark build and check passes:
all seven DeepSeek input/default cases, plus the actual Gemma2 serving
adapter's four off/on/on/off arms with device masks and joined lookahead.
Those arms preserve complete heads and state; each selects 32 GPU-mask plan
steps, while ahead arms build/cache six plans and capture five ahead. This
adds one distinct serving test to the 55 focused controls (56 unique tests),
not another claimed Gemma timing result. Installed job
`m35-ds4-mask-integration1` completes zero in 35 seconds and is waited on.
The integration code manifest is
`8ab8c757fbb876ba382b19353ecbeb9a6bfc52f40d3cf6f8869a26530dc12ada`;
method manifest is
`35457899052f5c48ed42d6cd985732ce3d2357619d6fd1a65c754a761ef29f7c`.
The original DS production qualification is retained rather than rerun:
its model/runner logic is unchanged by that committed Gemma integration.

Task-entry TensorFold native HEAD is
`f8fe17d24629aedabf90bbf78279dd776e6d62e7`; the retained `python-0.6` HEAD is
`ed78d6fc204d89d90b045bf033d6551e7714f3a1`. Both pinned READMEs were inspected:
native DeepSeek is under qualification, while retained Python documents
DeepSeek for MLX on a 256 GB Mac, with a different quantization recipe.
Neither supplies this CUDA comparison. This existing-model factor makes no
new reference speed claim.

Raw outputs, manifests, exact payloads, logs and scripts remain outside Git:
Spark A `~/.local/share/jitllm/ds4-mask{1,2,3,4}` and
`ds4-mask-method{1..7}`; workstation
`/tmp/jitllm-m35-coordination/ds4-mask-*`. Installed supervised screen jobs
`m35-ds4-mask-screen1` and `m35-ds4-mask-screen2` complete zero and are waited
on. Methods bind all source/input/binary/receipt identities and the
first-resolved cuBLAS libraries. Reproduction does not depend on keeping
those temporary scripts or captured input files: the two prompt texts are
`git show 2fe9c65:docs/architecture.md` (SHA-256
`be75fd969b69ca5702451584c026844bf92d0e4b6506f60111c465ad5f51d001`) and
`git show 0ea0a18:docs/engine.md` (SHA-256
`032947b69000b281fa7d89ca59b88d3c59511b6e1ff5b6de94ba3335165fd55a`). Build the benchmark from this slice,
retrieve those texts and prepare actual IDs with the retained artifact's first
GGUF metadata. Under an installed GPU-exclusive supervisor, run the four
fresh-process arms in this order, with a fresh output parent directory:

```sh
git show 2fe9c65:docs/architecture.md > text0.txt
git show 0ea0a18:docs/engine.md > text1.txt
probe=build/spark-native/benchmarks/jitllm_dsv4_mask_probe
artifact=/home/pmeenan/.local/share/jitllm/m3-artifacts/8a355bfb27c90e1150fbd7fa62ea6e63f6bf34fcca33934e52d22773f1508234
metadata="$artifact/meta/DeepSeek-V4-Flash-0731-UD-Q2_K_XL-00001-of-00003.kv.gguf"
"$probe" prepare "$metadata" text0.txt 4352 ids0.i32
"$probe" prepare "$metadata" text1.txt 4608 ids1.i32
"$probe" "$artifact" ids0.i32 ids1.i32 H1 host
"$probe" "$artifact" ids0.i32 ids1.i32 D1 device
"$probe" "$artifact" ids0.i32 ids1.i32 D2 device
"$probe" "$artifact" ids0.i32 ids1.i32 H2 host
```

Use the pinned SDK/cuBLAS environment and check prepared ID hashes against
the identities above. Compare all complete `heads.f32`, `choices.i32` and
`state-ranges.txt` payloads plus both state hashes; require actual mask
selections, captured/replayed joined graphs and successful retirement in each
arm. These committed text sources plus native preparation recreate the
required replay inputs after milestone cleanup. The factor binary SHA-256 is
`49656597a893882d35a56887fb2da9f01b3c28d7ded18195b66b34c941a0a7df`.
The successful screen source manifest is
`8064ea1d1aaa2cc959633bfa42fa55e4f8f1795f1f5cde48d3d3dda7dbe5dfc0`,
and method manifests are
`da3073e703f609483ce65165beebb9b64b9392299de26ee7bb5ef3eb341aa3ec`
(initial) and
`ede9ca2ee11a20da4e172311a6883c7f7303c3448f8f1efaec05f38d6c1f56ed`
(repeat). Raw records are removed from every host when M3.5 closes.
