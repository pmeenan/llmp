<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Gemma26 packed C4 + checked MoE: closed Stage B draft

Benchmark-only implementation based on Stage A commit c309575. Root approved
this closed protocol before source changes. Completed packed26 and
Stage A sources, binaries, captures, own calibrations and receipts stay unchanged.

Fresh TensorFold entry at 2026-10-05 14:48:25 UTC resolves default HEAD
609ca419abecebdc5a059498a613680bd3aa847f, version 0.6.5. Exact README, version
and Gemma recipe snapshots are in gemma26-compound-stageb-planning/ with their
SHA receipt. Gemma26 remains MLX-only, so the pinned same-format original CUDA
image remains this screen's reference. Recheck before a later new batch if the
implementation is substantially delayed.

Question: does adding the two existing checked MoE policies reduce the remaining
full-head mismatch and paid gap of the actual packed26 C4 path? CONTROL is
packed local/global attention + norm/RoPE + norm/residual, with routing/reduction
off. COMPOUND changes only fuse_gemma_route and fuse_gemma_reduce to true. Both
use ordinary products; broad fusion, fuse_norms, shared Q8, row-invariant,
RoPE-store and feature retention stay off. No kernel, engine, runtime, graph
builder, matcher, operand format, weights or production-default changes.

Use an additive manual benchmark main and dedicated scripts/report; link the
existing reviewed gemma26_attention_packed_wrap.cc unchanged. Set
LLMP_GEMMA26_PACKED_C4=1 in BOTH process arms. The new main accepts only the
two closed policies, fixed before Setup and for the entire process; no toggles
behind PlanCache. Both policies apply throughout prefill and decode. C1 prefill
attention remains the real untransformed builder in both arms, but COMPOUND's
routing/reduction also changes its prefill arithmetic. That accumulated KV
history is part of the factor and must be disclosed.

Closed fixture: approved 26B-A4B Q4_K_M artifact, width 2816, all 30 layers,
16 Q heads, local D256/KV8 and global D512/KV2, context 256, max_rows128,
C4 one row per distinct slot, read256, four full canonical heads, no features.
Preserve Q4_K gate/up, Q5_1 routed down layers0–28, Q8_0 routed down layer29,
all scales/slab pitches and existing product dispatch. Expected final C4 plans:
60 norm/RoPE, 90 norm/residual, 30 packed attention nodes; CONTROL route/reduce0,
COMPOUND route30/reduce30. Authenticate actual selected counts and assert all
other optional counts zero; refuse an unexpected plan instead of silently
claiming the policy. Record actual prefill counts separately. All ten/seventeen
matched descriptors and full-sort roots remain funded; tail-reader fallback,
checked operand spans and activation lifetime remain existing contracts.

Stage A observed stock decode4 selecting route30/reduce30. Each physical
prefill64/65/66/67 selects route only in layers0–27 and reduce in all30:
layers28/29 pass structural/shape gates and fail the physical-memory gate.
Thus uniform COMPOUND is NOT an identical stock recipe. Do not emulate overlap,
whitelist either layer, call this a bug, predict zero delta, or infer a single
operator cause from the outcome. This first screen can decide whether a later
same-input late-block probe is needed; that probe is not part of this task.

Reuse authenticated 1024 LE-I32 IDs (4096 bytes, SHA
b2d7aaf6aa2ef06d82591a3794f36640e192f429539ec934fd74bf4d81df1610).
Owner i prefills [0,64+i), then the unchanged eight-warm/reset/three-anchor
recipe; 32 paid C4 waves at positions67+i+s publish128 complete vocabulary
rows. Completed endpoints are99/100/101/102. Exact input IDs and alignment must
be authenticated before metrics.

Before inference freeze complete compiled-source checkout, prepared source/
toolchain build receipt, new executable, unchanged pack wrapper, original
client/header/math libraries, artifact manifest/index and actual IDs. Source
ADV precedes model loading. Run a narrow locked manual-target build plus the
existing24 descriptor controls, with only additional policy-count admission
controls required by this new main. No routine unit suite or scope expansion.
Then acquire CONTROL first/repeat and COMPOUND first/repeat (expected20–40s).
Each arm must have finite128 full heads and exact own-repeat heads plus all four
initialized-state witnesses (57671680 bytes per owner, 60 initialized ranges).
Keep exclusive own receipts BEFORE any fresh original acquisition or comparison.
Do not require CONTROL=COMPOUND heads/state or overwrite historical failures.

Fund128 retained heads and the node-owned pinned witness before admission.
CopyState retires before host reuse; full owner bundles survive unknown
completion. Packing, scratch, descriptors, weights and state stay under the
existing catalog/closure/budget. No unmanaged allocation or VA-pitch assumption.

After root authenticates the own freeze, run one fresh original/COMPOUND/original
bookend on the same Spark A (expected15–30s). Original b29 pin, digest837fc732…,
unchanged client01889d8c… and original libggml-cuda5a13585e…; physicalC4/u128,
context256/F16KV, normal fusion/graphs enabled, identical history/forced IDs.
Reauthenticate exact binary/headers/image library and approved raw/artifact
metadata. Record effective original cache defaults/capacities; positions<=102
remain before the1024 window wraps. No original selection/allocator/logger change.

Timer pays staging, packing, model work, completion, full-head publication and
CPU argmax in both engines. Complete finite scans, state copies, hashes and
writes remain outside. Paid COMPOUND full heads/states must match its own freeze.
Report native actual capture/replay separately from original GGML reuse and
never infer model latency from supervisor walltime.

Compare all128 complete rows: true byte equality, raw delta, strict winner
mismatch and reference-winner-over-native-choice margin against unchanged own
noise. Keep first4, first12 and later116 summaries. TV/NLL only on the same
predeclared16 rows (waves0/1/2/31); report signed means separately from maximum
absolute deltas. No corpus PPL, new tolerance, representative quality/support
or optimized-serving claim. Preserve a negative result without adding another
arm or ladder.

Use installed A m3fixb spark-job GPU supervision,600s/stop-on-fail per bounded
batch, after checksum sync and busy coordination. Expected individual jobs are
well below ten minutes. Final aggregate/provenance only in Git, raw heads/state/
logs outside; GENERAL + distinct root ADV/light before root commits. No C12,
context/ubatch ladder, production adoption or pointer-root implementation.
