<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Gemma C4 attention over variable-width independent roots

Removing balanced K/V packing at 8K lowers paid native C4 decode latency
**53.95% for 31B and 43.71% for 26B**, with all 128 complete heads, four initialized
states and 128 choices byte-exact across packed/owner repeats. Fresh stock
bookends leave native 31B **2.31% slower** and 26B **5.59% faster**. Strict quality
still fails: **4/128 and 25/128 positive-reference-margin choice differences**.
The checked native owner option defaults off; these timings do not qualify
optimized serving, corpus quality or wider cohorts.

| Profile | Packed intervals (s) | Owner intervals (s) | Copy-removal change |
| --- | --- | --- | --- |
| 31B | 8.92466 / 9.05816 | 4.13991 / 4.14030 | −53.95% |
| 26B | 2.09161 / 2.10545 | 1.18650 / 1.17605 | −43.71% |

The same binary changes only K/V packing. Q/mask packing, ordinary products,
plain norm fusion and the existing normRoPE/normADD policy remain identical.
31B selects 120/120 norm chains and 121 plain norms; 26B uses 60/90, 121 plain
norms and 30 route/reduction fusions. Counts describe selected implementations
in the final built plan, not CUDA launches or every intermediate prefill plan.
Original compiled grid/reduction geometry is retained independently of owner
kernel occupancy. Actual local D256 grids use 48 blocks; actual global D512
8192/8448-cell plans use 96 blocks for both profiles. The 26B startup 256-cell
D512 plan uses 64 blocks, so short-context geometry is not extrapolated.

Each process warms, clears and prefills four prefixes of 8188–8191 tokens,
then takes three forced anchors and 32 paid one-query C4 waves. Native uses
context 16384 per owner, max rows 128, local ring capacity 1280 and intermediate
state-only prefill. The canonical 8227 IDs are 32,908 bytes with SHA `6b6567ca…`.
The first paid global widths are [8192, 8448, 8448, 8448], causing whole-graph
independent fallback; later waves use uniform 8448-cell owner attention.
All paired native arms capture one paid graph and replay 31. Publication of
128 full vocabulary heads and argmax are paid; complete finite scans and funded
state copies follow the timer. Each 31B state witness contains 120 ranges /
1,740,636,160 bytes; each 26B witness has 60 ranges / 435,159,040 bytes. Completed
positions are 8223–8226. Every native repeat, packed/owner control and subsequent
paid reference-bookend native arm matches the admitted full byte identities.

| Fresh bookend | Stock intervals (s) | Native intervals (s) | Native versus stock | Strict choices |
| --- | --- | --- | --- | --- |
| 31B | 4.04441 / 4.04922 | 4.13533 / 4.14498 | +2.31% | 4/128 fail |
| 26B | 1.24793 / 1.25027 | 1.17114 / 1.18745 | −5.59% | 25/128 fail |

Stock repeats match all complete heads exactly. No cross-engine row is
byte-exact. Stock uses physical C4, separate F16 caches, explicit ring SWA,
unified KV off, and fast **untimed prefill rows 256 for 31B / 1024 for 26B**,
versus native 128. Stock26's ring capacity is **2048 versus native1280**;
stock31 and native31 both use 1280. These are explicit history differences,
not an allowance or a demonstrated explanation. No cause is attributed to the
remaining misses. All four 31B misses occur after the first three waves; 26B
has three in the first three waves and 22 later. `input_position` is the
zero-based consumed query; `completed_positions` is that position plus one.

Sixteen selected forced-token rows give maximum TV 0.213623 / 0.483989 and
maximum absolute NLL delta 1.897662 / 2.379197 for 31B / 26B. Their signed mean
NLL deltas are −0.355818 / −0.385088. These selected rows are diagnostics,
not corpus PPL, a quality gate or permission to widen tolerances.
[Aggregate results](results.json) retain the indexed strict misses and margins.

The native option routes only checked C4/query1 owner graphs to ten actual
sources: Q, mask and eight writer-derived K/V views. Each source retains its
exact SET_ROWS dependency and physical root. Read cells are multiples of 256
through 16384 when the real parent spans fit the 64 MiB bound; no unbacked cell,
contiguous eight-root pitch or imaginary read span is supplied. Unsupported
counts, multirow/features, unequal widths and oversized parents return the
original whole graph before mutation. Descriptors, activation/scratch placement,
coverage and execution closures are funded. CPU descriptor helpers retain no
CUDA linker dependency. Shader tile/fixup arithmetic and license terms are
unchanged; source-lock wording records the new default-off native consumer.

Measured source v1 is `4e461ba0…`, helper `3cb15371…`; v2 is `3545d847…`, helper
`57b4ad44…`. V2 changes only the long26 manual host envelope after the initial
fourth-owner prefill exceeded its capacity. The fixed recipe has 137 unique
keys (132 scalar plus five C4); its checked 40,597,151,984-byte execution budget
is a capacity ceiling, not allocated occupancy. All 31B runs retain private v1;
26B runs use v2. Final code `9f791840…` reconciles the complete Task69 head-cap
logic on `9761bb0`; measured manual output capacity remains 128. Task71's actual
cap4/owner-on integration control remains separate.

Spark A is `spark` / `spark-c4e2`, GB10; driver 580.178.04 is inherited from the
checked environment. The locked SDK receipt is `fbf84c74…` (NVCC 13.4.92,
CUDA toolkit 13.4.2, Clang 22.1.8). The new tracked public-backend 8K client is
`4ee48672…`, built against the unchanged pinned image `837fc732…` and its
inherited CUDA 13.3 toolchain. Artifact manifests are `32c92e07…` / `4ddb360c…`.
TensorFold at entry is `609ca419…` / 0.6.5, with the primary MLX26 recipe and no
matching CUDA recipe. Explicit source and SDK identities qualify checksum-synced
builds whose receipt has no Git origin.

The initial26 budget refusal, reference include-path build failure, wrapper
execute-permission failure and analysis parser attempts remain external. The
parser correction removes only the known two interleaved destructor diagnostic
lines; unknown grammar still refuses. Model outputs were not rerun for parser
fixes. Final focused checks preserve the missing-mise attempt and the subsequent
stale diagnostic dependency-record refusal. Full suites remain owner-deferred;
raw heads, state, logs and telemetry are outside Git. Original processes retire
through public teardown and checked owned-container absence.

Final B checks pass 36 metadata cases, 17 graph controls, 12 plan controls and
four default/head-cap controls. The CPU object symbol check and four source
contract checks pass. The stale dependency-record repair rebuilds only three
existing manual targets; measured kernels and outputs are unchanged.

Reproduce with `jitllm_gemma_owner_c4 ARTIFACT NEW_DIR 31|26 4 joined
norm|compound IDS_I32 8k`, fixing `JITLLM_GEMMA_OWNER_C4=packed|owners` and
`JITLLM_GEMMA_C4_NORMMUL=1` before construction. Use installed Spark supervision,
private outputs and the [tracked reference wrapper](../gemma-owner-root-c4/reference_8k.sh)
with the fixed client SHA. The [shared protocol](../gemma-owner-root-c4/PROTOCOL.md)
and earlier [closed consumer](../gemma-owner-root-c4/README.md) describe the lifetime
contract. Wider cohorts, serving recipes and genuine quality qualification
remain open.
