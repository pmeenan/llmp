<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# First bounded all-cost assistant screen

This is a source-only extension of the matched Gemma31 C1/P64 transaction.
Only profile31 `plain32` and `unit32` are admitted. The existing `unit`/`teacher`
chronologies remain available. No core/API, serving, default or26 policy changes.
Raw traces stay external. Build success alone releases no model acquisition.

All four arms request exactly32 endogenous greedy tokens committed after one
query64 prefix: native target-only, native actual `GreedyUnit`, pinned original
target-only, and pinned original target-plus-assistant. Every arm intentionally
ignores EOG for this bounded literal diagnostic and records authenticated EOS106
occurrences. It is not a completed answer or chat-stop control. C1/context4096,
max rows/ubatch128 and ordinary F16 local/global1280/4096 caches are fixed.
Native31 uses its existing norm/RoPE and norm/add chains in both modes. Plain
needs one head and no retained features or assistant admission. Assistant mode
needs four heads/verify rows and actual postnorm features. Residency/workspace
occupancy is separate; no equal-peak or offload claim follows.

Each actual native unit uses depth=min(3,remaining−1), so accepted rows never
exceed the remaining output request. The final one-token tail uses ordinary
Wave1. The anchor comes from the completed head; every draft in a unit uses
constant queryP and starts from actual target featureP−1. The existing unit
releases its borrow before Verify, retires Accept and publishes only the accepted
prefix with selected keep−1 head/feature and a separate uncommitted next anchor.
The original helper uses its public assistant and target decode/`seq_rm` seam,
copying every verify feature immediately and selecting keep−1 before any next
decode. Original logical rejection does not promise physically zero rejected rows.

One untimed fixed32 warm trajectory precedes two untimed quality repeats and two
minimal paid repeats per mode/process. Reset and query64 prefix occur outside
all paid intervals. Timer starts immediately before the first endogenous step
and stops after all32 commits and normal retirement/head/feature publication.
Actual target/draft work, snapshots/Verify, rollback/Accept, inherent full-head
copies/finite judges and result carry remain paid. Timed loops retain only fixed
emitted IDs/unit counts and normal workspace/final carry. Full diagnostic archival
copies happen only in quality repeats. End-state checks/hashes, final finite scan
and file writes are post-timer. No idle/phase subtraction or causal attribution.
Quality traces retain32 predictive heads, all target verify heads/features and
available last draft heads (the real unit does not expose earlier internal draft
heads/features). Pending position96 is not counted among the32 committed tokens.

Native vectors are funded before reserve with64MiB plain/256MiB assistant,
actual total capacities limited to grant−1MiB for bounded metadata. Retained
features, state/verify snapshots and plans are measured/funded by Setup before
startup; Q8 metadata admission remains temporary512MiB. A catalog-funded pinned
buffer supplies endpoint state hashing after timing. A separately catalog-funded
feature buffer for a final scalar tail is allocated before the paid loop, copied
and published inside it, and released only on successful retirement; refusal
retains the owner. No wholeKV host vector is created. Original known vectors
are bounded at192MiB plain/384MiB assistant, including a128MiB maximum opaque
sequence snapshot and1MiB metadata/descriptor allowance. Library/context/model
occupancy is separate. The prior P64 opaque snapshot was57,674,696 bytes;
P96 needs its own larger bound (actual size remains unmeasured). The legacy
P64/26 bounds are unchanged. No unchanged peak or new injected-failure claim.

Each producer must retire successfully, then independently freeze all finite
quality payloads, exact own repeats, inputs/teardown and timed counters/final
head/feature/state. Timed outputs must match its own quality chronology exactly.
Native own proofs precede first original-output exposure. Cross comparisons
reauthenticate every frozen identity. Require speculative emitted32 IDs equal
its engine's plain continuation, native plain equal original plain, and native
speculative equal original speculative, with pending-anchor agreement. On first
common-prefix divergence record reference margin/tie and stop interpreting later
heads from different histories; preserve FAIL. No scalar/query4 full-head byte
oracle, noise bound, tolerance, new PPL gate or teacher-force rescue is introduced.
This exact continuation gate qualifies timing interpretation only for this short
literal screen, not broad model quality.

Only if those gates pass report both actual elapsed repeats/spreads for identical32
committed work, accepted distribution, unit/depth/draft/verify/tail counts and
normal full-head publication work. No assistant gain follows from a draft-only
microbenchmark. First producer bound: at most32 units,128 target query rows and90
draft rows per trajectory; five trajectories plus five query64 prefills fit a
600-second supervisor job. Forecast50–90seconds/model pair plus startup from the
prior short controls is an estimate; actual official elapsed remains authoritative.
No full suite, depth/context ladder,26 transfer or sustained/serving/default claim.

Use the existing `llmp_gemma_greedy_reference` target and the separate
`performance_reference.sh` wrapper under the unchanged837fc732/b29c606e original
pin. Source/build, native acquisition/own and original acquisition/own/comparison
pipeline templates are reviewed once; later successful producer/own identities
are literal root-authenticated substitutions. Latest TensorFold at entry remains
cb2ebf0540f42604e2759b2ddef497861e928248/version0.6.6, with no comparable CUDA
Q8 assistant measurement. The owner's later llama.cpp update does not alter this
in-flight reference.
