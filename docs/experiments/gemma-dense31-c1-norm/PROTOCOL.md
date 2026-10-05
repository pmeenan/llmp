<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Dense31 C1 ordinary-product checked-norm first screen

Task-entry 2026-10-05T13:11:15Z; fresh primary TensorFold HEAD
609ca419abecebdc5a059498a613680bd3aa847f/version0.6.5. README, package/version
and Gemma recipe freshly fetched at that pin13:11:38UTC. Recipe remains
Gemma26 MLX-only, no dense31 CUDA comparator. Source base0a1a0a6.

## Closed candidate and control

Dedicated copied helper accepts ARTIFACT NEW_OUTPUT ordinary|norm IDS_I32.
Only dense31/scalar/C1/supplied IDs are possible. Historical gemma_joined and
C4 clients remain unchanged. Candidate selects existing checked normRoPE and
normADD, ordinary product dispatch, row_invariant=false; all other optional
flags remain false. Ordinary control sets both norm flags false. No production,
engine, graph, kernel, registry, runtime or source-lock edits. No comparison
between C1 and C4 is required or inferred.

Approved prepared artifact32c92e077a6816b54aa988e2dee61a3639c958fd510ea99e25f3621f10b2aa08,
index9ae365a17ed73528262d7befdb505bf07e6695e03f0fa36f69e69099dbd5f34c;
matched same-format source c1ac76e99d5513b141e8adde7288b85c3f9c32ec,
18,822,970,304 bytes/SHA9e92cb6236044c6a9870af406029c74a76e0571c157a6f95df724dcc8c7a1575.
Canonical1,024 little-endian IDs SHA
b2d7aaf6aa2ef06d82591a3794f36640e192f429539ec934fd74bf4d81df1610,
BOS2 once, from complete War and Peace corpus3,274,124 bytes/SHA
c7156148ecaa12b6416cf816540d8dede2014982554a835e61076f0dd8bf0c2d.
Source payload prior approval remains provenance; no gratuitous historical
full-model rescan. Manifest/index, headers, libraries, source and IDs are
reauthenticated before acquisition and against the source freeze afterward.

## Actual shape, paid units and lifetimes

Native context256/max_rows128/slots1. Prefill first64 supplied IDs as one
64-row/one-segment native chunk; eight discarded scalar warm units; clear and
repeat that prefill; three untimed anchors. Paid step s=0..31 processes ID
67+s at absolute position67+s; its labelled target likelihood is ID68+s.
First-built prefill(rows64/segments1) and decode(rows1/segments1) counters must
show candidate normRoPE120/normADD120, ordinary0/0; row products, generic norm
fusion, RoPE/store, sharedQ8 and MoE flags0 in both. These are selected-plan
counts, not a per-invocation kernel observer. Record capture/replay counters.

Timer includes32 complete262144-value head copies, lowest-index argmax scans
and32 paid scalar units/groups. Native complete finite scan is outside the
paid interval, matching the original analysis checks; disk writes also excluded.
Retained/working heads precharged35,651,584 bytes, separately from catalog
output, state, plan, scratch and pinned capacity. Stable heap owner retains
node/runner/buffers on unproven retirement. Existing held request/completion
and catalogue lifetime remain unchanged. Capacity is not measured physical peak.

## Blind own freeze and primary reference

Build native helper and original thin client first. Freeze exact source, native
and original binaries, actual locked build receipt, immutable source ancestry,
headers, libraries, manifest/index and input identities into an exclusive
source-identities.json before any model acquisition. Then acquire ordinary-control
once, candidate-first and candidate-repeat into independent fresh directories.
Validate all three complete32MiB finite files and candidate whole-byte repeat
identity, shapes, policies and actual completed counts; source/environment must
still equal the source freeze. Freeze own evidence exclusively after official
successful acquisition retirement, before any stock load/comparison.

Only the candidate has a measured own-repeat bound. The ordinary control has
no independent repeat: do not classify its differences against zero candidate
noise or claim an ordinary positive-margin-outside-noise metric. Candidate-to-
ordinary movement is a policy difference. Do not inherit128-row/C4 noise or
scores, or retrofit a bound after the stock result.

After own freeze authentication, fresh originalC1/u128 -> candidateC1 ->
originalC1/u128 short bookend. Unchanged committed original C-API client and
pinned b29c606e28a01b1bc8c1351026a0fa6e616bf6c4 digest837fc732... image/math;
context256 per sequence, batch128/ubatch128/one sequence, F16KV,
kv_unified=false default, fusion and graphs enabled. Its64-ID prefill fits
one physical chunk, matching this recipe without claiming identical operations.
No eval callback, kept intermediate or traced timing changes reference policy.

Authenticate fresh stock whole-head repeats and candidate bookend identity.
Compare all32 full heads for bytes, raw delta and lowest-index greedy choices.
Strict differences, positive reference winner-over-candidate margins outside
frozen candidate zero own movement, and exact reference ties remain separate.
Selected rows s=0/7/15/30 carry full-distribution TV and target NLL at ID68+s;
no corpus PPL claim. Raw heads/IDs/per-step telemetry stay external, aggregate
metrics/source/retirement identities only in Git. No ordinary calibrated-margin
claim without its own independent repeat.

Stop after a negative first screen; no C12, context or depth ladder. No policy
adoption, fused-new-kernel, target state or whole-model qualification claim.
Even bounded positive C1 evidence still needs subsequent state/continuation
and both-approved-Gemma/backward-transfer gates; Qwen/DeepSeek applicability
requires actual format/activation/shape checks.
