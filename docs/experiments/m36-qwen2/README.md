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

The original control harness pays a 4 GiB startup/compiler/result allowance before file
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

## Historical FP16 heldout trajectories

On 2026-10-12, all six controls passed on Spark A with zero failures or skips.
The two added cases use the original 577-token heldout trajectory: 16 and 17
prefill rows, 16 decode rows, a 512-row prefill and 16 suffix rows, with context
1,024. Every F32 logit matches the historical oracle across all 151,936
vocabulary entries. The complete 1,040-ID input file has SHA-256
`6dd8da897821f5615e7796f4d882795faf306a941f050e2eb68b619096e3fb0c`.
Unfused logits match
`69ff08210fb2450388e32401091eed2bb02df7325ca966268a9076009a407ae0`;
fused logits match
`bfb36f192317c4a32723e65753dbe6a75f4c5e45d4d885736d2b26e92431e9d5`.
Each oracle is 350,668,288 bytes; the existing native reference regenerated
both hashes and repeated each evaluation exactly before shared comparison.

Each heldout case restores 12,582,912 checkpoint bytes saved at 33 positions
after eight real decode writes, replays those eight rows, and restores all
12,582,912 used bytes after spill at 49 positions. Counters and complete used
footprints match, followed by exact remaining logits. Each case compares 585
rows including replay. Original cases retain their input pins and funding.

Five authenticated plan keys and the larger outputs have a checked host
inventory of 5,530,412,208 bytes, prepaid within an 8 GiB host allowance and
16 GiB execution domain plus the separate 6 GiB driver margin. Descriptor,
program, pinned-copy and retirement bounds remain enforced. The six-arm run
used a 16 GiB zero-swap cgroup cap, observed cgroup memory peak 1,638,457,344
bytes and no memory-limit or OOM events. This counter does not measure total
catalog/device occupancy. Test source SHA-256:
`f38183d3d41138431f9a25043890c60bc33b1e536fc1c95ba0c01d2f60467ef1`.

## Repeated FP16 execution and full weight eviction

On 2026-10-12, all ten cases passed on Spark A with zero failures or skips.
Four new cases repeat both complete trajectories in both fusion modes. Each
performs four evaluations on the same shared runner, clearing state before
evaluations two through four and repeating the checkpoint and spill controls.
Every compared F32 logit remains exact against the same frozen oracles:
336 full-vocabulary rows per repeated control case and 2,340 per repeated
heldout case, including checkpoint replay.

Evaluations three and four end the completed chunk's request after 32 control
or 33 heldout positions, evict all 490 weight extents and require every
extent to be nonresident and unheld. The physical backing count drops by
exactly 490. The next request reloads every extent with an advanced backing
generation and unchanged content generation; before another write, complete
checkpoint bytes, positions and used footprint still match. Each case
records two evictions and 980 restored weight extents. State remains resident
at these weight-only eviction points.

The added generation inventory and operation scratch have a 128 KiB source
bound. Total checked host inventory is 3,746,932,400 bytes for control and
5,530,543,280 for heldout, within the existing 4/8 GiB host allowances and
8/16 GiB execution domains plus the separate 6 GiB driver margin. Evaluations
reuse the same two/five plan keys, oracle and pinned state-copy buffers.
The ten-case run used a 16 GiB zero-swap cgroup cap, observed memory peak
1,665,028,096 bytes and no memory-limit or OOM events. This
counter does not measure total catalog/device occupancy. Test source SHA-256:
`8054a898c7b3340e01b7711ce444d3685b45d9d02f258a69587808f9e50097a9`.

## Remaining gate

The shared controls now include the original control and heldout trajectories,
repeated execution and full weight eviction/reload in both fusion modes.
Historical [P2](../backend-proof-p2/README.md) also relocates restored weights;
address relocation and kept-state/restart adoption remain open. EXL3 must pass
its exact 1,570-linear and Tier C 750-case controls through the shared runner.
Legacy execution remains until those controls pass.
