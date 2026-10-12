<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Qwen2 fixture migration: shared specs and FP16 control

The approved Qwen2.5-0.5B FP16 and EXL3 4.0/4.5 bpw fixtures now synthesize
specs over the shared embedding, attention, GLU feed-forward, norm and head
components. The FP16 fixture also executes through `ComponentStepSource`,
`ModelRunner` and the shared GGML lowering. These are fixture controls;
there is no serving or speed claim. G5b's complete EXL3 lowering, G5c's remaining
correctness/lifecycle controls and G5d's deletion remain open. The first
[shared EXL3 reconstruction screen](../m36-exl3-selected/README.md) now passes.

The shared component additions preserve their existing defaults: optional
Q/K/V bias, explicit scores with score scaling, and the EXL3 F16 rounding
boundaries. EXL3 norm weights stay BF16 in resident storage and widen only
as transient arithmetic requires. Its four weight parts retain explicit
roles. The 21 approved reconstruction algorithm records publish 18 complete,
output-type-specific tuning facts at rows 145, 1023 and 1024 on sm_121;
unpinned types, partial matrices and architectures are refused. These facts
alone are not an EXL3 shared-engine execution proof.

## Evidence

On 2026-10-11, native Spark builds and focused controls passed. The final
checkpoint's 32 host cases include actual-size FP16 planning/lowering,
component arithmetic and negative fusion-policy checks. Earlier installed
metadata controls passed for all three exact artifact IDs. All runs checked
source and recipe manifests before and after execution; zero failures and
zero skips were required. [results.json](results.json) records the aggregates
and source identities; raw logs remain in the external coordination scratch.

The final carry onto native v1 writer main passed 2,539 native and 1,994 CPU
non-model cases, followed by 130 model/timing cases with zero failures or
skips. Three files required formatting-only corrections; the affected 96
native and 85 CPU cases and the format, header, portability and license checks
then passed. The model run includes all four FP16 controls below.

| Shared FP16 control | Token rows evaluated | Result |
| --- | ---: | --- |
| Unfused screen | 32-row prefill + first decode row | bit-identical |
| Unfused full | 32-row prefill + 44 decode rows | bit-identical |
| Fused screen | 32-row prefill + first decode row | bit-identical |
| Fused full | 32-row prefill + 44 decode rows | bit-identical |

The 76 token IDs and legacy logits were regenerated and required to match
the frozen hashes before candidate comparison. The token-file SHA-256 is
`37e46a2304b5ab5648d0f2b7fdb94224596ed690caaddc1726df27110b585e21`;
unfused logits are `3560d3370b167ce87d4134b0be70c0a59aa230db469f1bbbc5fb4a6b5829d6b1`,
and fused logits `bb8ae5e7e3ac6da734173edb1111160a0c80a55c4279b94e67a8f90b142e7571`.
The comparison checks every F32 bit, across all 151,936 vocabulary entries
of every evaluated row.

The initial fused candidate passed prefill but failed the first decode
row. Shared selection lacked the existing float MMVF product-add and
product-SwiGLU fusions. Two opt-in patterns now select those kernels for
eligible one-column F16 products, with exact IR coverage and the existing
native fusion/device checks. Default policies are unchanged. A missing
optional device predicate or a row-invariant policy is refused, not invoked
or silently substituted. Both full controls then passed.

The harness pays a 4 GiB startup/compiler/result allowance before file
reads or compiler allocation. Its checked named source inventory is
3,746,801,328 bytes, including the 46,188,544-byte oracle, metadata/parser
owners and retained plans. The complete execution domain is capped at
8 GiB, with the existing 6 GiB driver margin checked independently. Bounded
reads precede allocations; only two authenticated shape/parameter keys may
enter the plan cache. One heap owner retains the node and every borrowed
resource if teardown cannot prove retirement. This is a conservative
source bound, not a measured memory peak.

## FP16 checkpoint and spill continuation

On 2026-10-12, the existing four controls passed on Spark B with zero failures
or skips. Each full trajectory saves a checkpoint after the 32-row prefill,
executes eight decode writes and requires the saved pages to have changed.
It restores all 6,291,456 checkpoint bytes, counters and the complete used
footprint, then replays all eight rows against the same frozen logits. At
48 positions it spills and restores all 6,291,456 used state bytes, verifies
bytes, counters and footprint, and continues the remaining control rows.
Both fused and unfused full cases compare 84 rows: the original 76 plus the
eight replayed rows. The two 33-row screen cases are unchanged.

Completion-aware state copies use two bounded 16 MiB pinned buffers, retained
by the node until retirement. The additional observation inventory is
33,701,888 bytes, within the existing 4 GiB host allowance and 8 GiB execution
domain. The actual four-arm run was confined to 8 GiB with zero swap and
observed a 706,785,280-byte cgroup memory peak with no memory-limit or OOM
events. These are
state-copy and continuation equivalence controls for this one approved
fixture, not independent historical state-oracle or restart-adoption proofs.
The test source SHA-256 is
`53537626bff77947b14c5532f5c1ac14745906ac37dfa004da4f4e72d1976dec`.

## Remaining gate

The four cases above are screen/full evaluations of the same control
trajectory. They are **not** the four historical [P2 arms](../backend-proof-p2/README.md),
which are fused/unfused × control/heldout. The heldout trajectories, repeated
state and kept-state/restart controls remain open beyond the checkpoint and
in-process spill controls above. EXL3 must
pass its exact 1,570-linear and Tier C 750-case controls through the shared
runner. Legacy execution remains until those controls pass.
