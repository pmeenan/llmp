<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# One C1/P64 Gemma assistant greedy transaction

This manual diagnostic starts with the approved 31B target/Q8 assistant pair.
The C++ operands also support the approved 26B pair, whose transfer requires a
separate source-bound original raw-model admission. No runtime, serving,
settings, kernel or arithmetic default changes. TensorFold is re-pinned at
task entry; its MLX-only recipe supplies no comparable CUDA assistant result.

Use the canonical 1,024-ID input with SHA-256
`b2d7aaf6aa2ef06d82591a3794f36640e192f429539ec934fd74bf4d81df1610`.
Both engines prefill IDs 0..63 in **one actual 64-query chunk**, with configured
max rows/ubatch 128, C1/context 4096 and ordinary F16 split caches. Native uses
head cap 4, verify cap 4 and explicit retained features. Other native numerical
options remain their defaults. Public uses the pinned original image
`837fc732...`, graphs/fusions enabled, `swa_full=false`, `kv_unified=false` and
local/global capacities 1280/4096. Construct target and assistant contexts
before prefill. The initial complete head predicts the pending anchor at 64;
the assistant receives the actual POST-finalnorm feature at 63 and frozen
target cache. Global V is the original raw K-as-V projection, never a rewritten
or fabricated cache.

Draft three tokens at constant query position 64, using the first target feature
and then actual assistant recurrence. Record every complete finite draft head
and postprojection feature. Each draft leaves native requested initialized
target KV and feature unchanged; public checks whole opaque target state,
physical borrowed caches and cell metadata. Release/synchronize the borrow
before target writes. Verify the authoritative anchor plus three drafts in one
four-query target operation. Draft j is judged against target head row j.
With m matches, keep=m+1 commits only the anchor and matched drafts. Row keep−1
supplies the selected complete target head/feature and a separate, **uncommitted**
pending next anchor. No generation past that anchor is performed.

Native records an independent ordinary normal Wave4 over the manually observed proposal,
including complete target heads/features and accepted semantic-prefix KV hash.
Clear plus another query64 prefill must restore the complete initial head,
feature and requested initialized KV exactly before the real `GreedyUnit`.
That unit must match the normal four-query outputs, selected carry, semantic
accepted prefix and exact rejected-row restoration. This extra oracle/reset is
untimed diagnostic work. It is not a scalar-width comparison or latency claim.
The unit does not expose its full internal proposal. The manual proposal is
witnessed by those complete target rows/accepted prefix and by byte equality of
the actual last draft head against the manual last recurrence; no diagnostic API
is added to claim direct observation of its internal token array.
Public verifies its proposal in one decode, then removes rejected logical cells
with `llama_memory_seq_rm`. Check split-cache endpoint/membership and unchanged
accepted semantic rows. Public rejected physical bytes need not become zero;
there is no cross-engine KV-byte claim.

The pinned original target graph publishes `h_nextn` after final output norm and
before output-ID gather. With `embeddings_nextn(true,false)`, decode copies all
four token rows and `llama_get_embeddings_nextn_ith` indexes them densely by raw
token row. Copy all four immediately after synchronized verify; select keep−1
before any later decode. The source evidence is retained externally:
`src/models/gemma4.cpp` SHA-256
`765ee856e30b3ddc126fa75c8252e5fbca6bae0e11abd9b71f4c21d911810ecb`
and `src/llama-context.cpp`
`6429ebec7c926945987e6fe037317af0f99265490bc14b0606d9487a62a76453`
from original `b29c606e28a01b1bc8c1351026a0fa6e616bf6c4`.

Repeat the complete reset/prefill/draft/verify/accept chronology twice per engine.
Each engine independently freezes all finite full heads/features, token arrays,
state witnesses and completion metadata after official successful retirement,
before cross-engine output reads. Output directories/files and own-proof files
must be new. Authenticate current source/frame, binaries/SDK, prepared artifacts,
retained raw checkpoint sizes/identities, headers/libraries and canonical IDs
before dispatch and after retirement. Public container ownership/retirement uses
the existing `1fd85804...` helper and checked absence of its unique CID.

The initial anchor and all three proposal tokens must agree before full-transaction
comparison. Report the first differing proposal index and preserve that failed
gate. A separately released **teacher** control may consume only predeclared
canonical IDs 64..67 in one four-query target operation, without assistant dispatch.
It never rescues the endogenous proposal gate and cannot borrow a new allowance.

For identical transaction operands, PASS requires zero strict positive-reference-
margin target choice differences, matching accepted count/pending next anchor and
relative conditional loss increase at most 3%. Exact tie differences are reported
separately. Record complete row/feature byte equality and maximum raw differences;
no calibration or numerical tolerance is introduced. The four supplied likelihood
targets are initial head→ID64 and verify heads 0..2→IDs65..67. These are explicitly
conditional on the generated proposal, **not teacher-force corpus PPL**. Final
verify row 3 predicts position 68 under the complete proposed block and is unscored.
The actual pending next anchor predicts position 64+keep from selected row keep−1.
Original/native own-repeat
success and official rc0 are completion evidence, not these comparison gates.

Caller vector growth is funded first: native has 512 MiB bounded authenticated
vocabulary admission (destroyed before release/startup), then a retained 64 MiB
output/range grant with actual vector-capacity check. State/feature diagnostics
use separately cataloged pinned buffers; uncertain copies stay owned. Component
workspace and verify snapshots are included by `SetupAssistant` before startup.
Public opaque sequence state is bounded before allocation at 64 MiB for 31B
(32 MiB for 26B); the known simultaneous vector envelope is 384 MiB/256 MiB.
The latter includes protected, verified and accepted borrowed-cache witnesses,
two opaque states and full row records. It is not an unchanged-peak claim.
Successful cleanup synchronizes and destroys assistant before target; unknown
effects retain owners/fail-stop, never masquerade as a completed transaction.

The first source/build packet authorizes no inference by itself. Acquisition,
independent own freezes, comparison and any conditional teacher control each need
their actual source/official identities bound before supervisor release. No
32-unit timing, serving, batching, sampled acceptance or broader quality claim
follows from this one-unit screen. Raw outputs stay external; Git retains only
reusable source, identities and aggregate outcomes.
