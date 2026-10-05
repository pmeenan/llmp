<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Gemma26 8K ring-cache reference recipe probe

Source base437d59f3530ec8d7b560a4127745ef5fd23b1938, which includes the
dense31 H1 recipe comparison. No production,
historical helper/result or kernel policy changes. Task-entry primary TensorFold
refresh 2026-10-05T14:25:19.159540Z resolved
609ca419abecebdc5a059498a613680bd3aa847f, version0.6.5. Four primary files are
retained with hashes; the approved26 recipe remains MLX-only, no same-format
CUDA comparator. This is the required backward transfer of the cache-recipe
learning, not a transfer of dense31 output agreement.

## Closed source change

Copy the original pinned prefill reference client into a dedicated26 experiment.
Close it to approved Gemma26, one sequence, screen mode, context16384,
batch8192/physical ubatch1024, F16 K/V, explicit swa_full=false and
kv_unified=false. Keep original image837fc732…/llama b29c606… math, fusion,
graphs, token recipe and timers unchanged. No observer/kept graph nodes or
target/assistant policy changes. Record all four original libraries and exact
headers, executable and effective flags; require actual cache-creation log
evidence of local2048/global16384.

Pinned common CLI/server defaults are false/false whereas C-API defaults are
true/false. Pinned iswa construction pads min(context,window+ubatch) to256
when swa_full=false; native pads min(context,window+max_rows) likewise. Window1024
and ubatch1024 give local2048. This agrees on physical capacity only: initialized
mapping, masks, occupied read widths and arithmetic are measured questions.
Historical full-cache timings and quality retain their actual recipe; no native
defect or retroactive waiver follows the configuration difference.

Native benchmark benchmarks/gemma_prefill.cc remains byte-identical
8fc6e4b377f5fa120d1dcc82f44531c2b201eb0462905fbe9ff1c68e45b772a2.
Invoke only `ARTIFACT IDS NEW_DIRECTORY 26 all 1024`. This preserves the current
diagnostic compound axis: selected norm/RoPE60, norm/residual90, routing30 and
scaled reduction30 for prefill and decode; row-invariant products, shared-vector,
generic norm fusion and cache-store fusion0. These are selected-plan counters,
not observed kernel launches. Uniform native routing/reduction does not claim
original fusion eligibility parity, C4 evidence transfer or accepted quality.

## Authentication and own boundary before any ring original load

Approved26 target artifact4ddb360c9ce08f1e984ab304b6af918be44246d52346734066b06443f7c249d3
and matched raw GGUF f2c28b3dc4776931ac6f879e11f203dec637ea0f14267a86ec8f6165f63f293f,
16947541728B, revisionc099eb48e663fd284577b04978a94ffccb261841. Authenticate
small manifest/index and existing verified transport/import receipts, not new
historical payload scans. Common supplied IDs are8227 LE I32,32908B,
SHA6b6567ca51a3fbe5000521cb71fcf168ef485623bdbfea2abab30d57f414d96b,
BOS2 once, from the authenticated War-and-Peace corpus/tokenization recipe.

The historical c3d089a1f6db28fc8db75d113764a0f0c716e1291196d3aef3219aca02fb90f8
native binary is reusable only if an actually retained executable and its
af910cb38aeeddf819af640e8ea6dcc1b97f594ec0100bd98a073c291ad135ea receipt
authenticate. No independent retained executable has been identified; the new measurement
therefore uses fresh own controls on the unchanged benchmark source.
Otherwise compile unchanged native source on437d59f and take new all1024
first/repeat two fullheads,32 incoming choices and initialized state. Create an
exclusive own freeze BEFORE the new ring original sees weights/executes. No
inherited dense31/C4/128-row allowance and no post-oracle bound changes.

Pre-freeze encloses actual binary/build receipt, full synchronized production
manifest (separately label content-bundle and JSON-file hash), base/dirty-path
allowlist, client/analyzers, source-lock/SDK, original image/header/library hashes,
IDs and target metadata. Official installed build/acquisition/freeze successful
retirement records are part of the own boundary. Authenticate these identities
again before/after the paid bookend. Use v2 tuple/list-normalized comparison from
the start, with exact-equality and real-mismatch/incomplete-stream controls.

Historical all1024 own receipt c5999e5e4119fbf87929d1d139d68dd4bdd3bef08c3d38c216e8749b17ae7383
and source log72f9733810cfce5384c24ae4c4984ab02b2b5df3b6076d2e141fc8204bb65537
remain independent history. A new binary may compare to its old two heads/state
for an unchanged-math witness, but that does not replace new own controls.

## One paid bookend and bounded output claims

Owned Spark-b m3gm31, installed --gpu supervision/timeout600; coordinate quiet
measurement window. Native first/repeat expected under1min plus state hashing;
then one ring-reference / native all1024 / ring-reference bookend, roughly1min.
No second policy, profile, context ladder or memory observer in this first screen.

Six discarded warm rows, clear, paid8192 prefill, three common untimed anchors,
then32 one-row forced units endpoint8195→8227. Both clients compute the
lower-index argmax of the INCOMING head before each forced input. Native pays
eight full prefill-head publications (seven intermediates); original requests one
final prefill head. Both pay32 full decode-head publications. No extra retained32
vectors or finite scan inside a timer. Setup, output writes and snapshots remain
outside timers. Preserve all existing paid plan/capture/staging/state-growth work.

Native initialized state592445440B/60ranges, layout
`gemma26-f16-kv-scalar-device-v1:16384:1024:16384:2048`.
New own first/repeat/bookend state, both finite full heads and32 incoming choices
must match the authenticated new own freeze. Original bookends independently
establish repeat identity. Compare32 strict greedy choices and exact input/position
alignment; compare complete prefill/final vectors (bytes, maxraw, TV and explicitly
absolute reference-argmax NLL difference). Only two vectors are retained, so no
all32-vector raw-noise or margin calibration claim. No PPL, state-restore, context
qualification, target correctness gate or automatic adoption.

Existing closure/residency, completion-proven pinned snapshots and whole-owner
quarantine remain unchanged. Historical all1024 accounted capacity25868105408B
is provenance, not a new physical peak measurement; report actual fresh funding
and host/GPU/driver/toolkit identities. Reference cache capacity does not establish
a whole-process memory ratio. Raw heads/state/tokens/logs stay external; Git stores
only source, aggregate metrics and enough provenance to reproduce the recipe.

Stop after this representative recipe comparison, positive or negative. Any new
policy or narrowed-final-block attribution is separate work. Qwen/DeepSeek cache
topology applicability needs their own source/state contracts; no default transfer.
