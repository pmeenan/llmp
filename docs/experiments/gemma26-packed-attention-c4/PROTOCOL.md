<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Closed Gemma26 packed-attention C4 screen

Benchmark-only base 802c33b. TensorFold task-entry observation 2026-10-05T12:59:21.998666+00:00 at 609ca419abecebdc5a059498a613680bd3aa847f, version0.6.5; primary Gemma26 recipe is MLX-only.

## Question and closed policy

Does the measured dense31 joint local-attention dispatch and local/global stream-geometry change transfer to the actual Gemma26 Q4_K_M artifact? Compare an unchanged native control with a benchmark-only derivative of the reviewed CONCAT wrapper. Both use ordinary products plus norm/RoPE and norm/residual (BOTH); row-invariant, shared Q8 preparation, routed topk/reduction, RoPE-store and broad fusion remain off. C1 prefill is unchanged. Do not inherit dense31 exactness, quality bounds or a production choice.

Candidate eligibility is exactly the approved Gemma26 profile: width 2,816, 30 full layers, 16 Q heads, local D256/KV8 and global D512/KV2, context 256, max rows 128, four distinct owners with one row each, local/global read width 256, four head outputs and no features. Other profiles/shapes/options return the real builder unchanged and report fallback. Process-fixed mode prevents cached plans from silently changing policy.

The actual fixture has Q4_K fused routed gate/up, Q5_1 routed down in layers 0–28 and Q8_0 down in layer 29; shared FFN and token/head weights use their existing Q8_0 bindings. There is no Q6_K tensor in this approved Gemma26 fixture. Preserve all expert descriptors, quant padding/slab pitches, routing arithmetic, scales and ordered sums unchanged. The prior stock layer-28 topk physical-alias eligibility mismatch remains a known independent limitation: do not imitate allocation overlap, whitelist a layer or promise full-model byte identity.

## Checked graph construction and funding

Authenticate all four existing FLASH descriptors per layer and eight exact SET_ROWS writers. Each raw K/V view must refer to that owner's actual cache leaf and use the corresponding SET_ROWS result as its source; authenticate src[2]/view_src ownership and all eight write-before-attention dependencies. Balanced CONCAT raw K/V [D,KVheads,256,1] along dimension 3, then permute 0,2,1,3. Preserve the reviewed Q construction and concatenate separately padded [256,32] masks along dimension 3. Use the existing checked MMA operation; ne3=4 naturally excludes the local vector selector. The packed output [D,Qheads,1,4] supplies exact owner views to unchanged downstream nodes.

Rewire direct sources, view sources and named consumers; remove the old FLASH roots before bounded GraphOrder reconstruction. Assert 30 candidate attention nodes versus 120 control nodes, all cache writers preceding each packed node, full mask padding, output owner order and existing readable keeps. Assert the actual final C4 norm selections (expected 60 norm/RoPE and 90 norm/residual), with all other optional policy counts zero.

Retain the reviewed bound of 64 extra descriptors per layer (1,920 for 26) through TensorArena/GraphTensors and existing Setup/PlaceAndPlan funding. Actual activation/scratch placement and execution closure include every packing root, cache extent and workspace; no unmanaged allocation, VA-pitch assumption, VMM change or production selector. Run the bounded 24 descriptor cases over 26/31, owners 1/2/4, rows 1/2 and both process modes, updating only the exact approved transformation scope.

## Immutable input and native own freeze

Use the existing 1,024-ID little-endian fixture, 4,096 bytes, SHA256 b2d7aaf6aa2ef06d82591a3794f36640e192f429539ec934fd74bf4d81df1610; authenticate actual loaded IDs and target alignment before any metric. Owner i prefills IDs [0,64+i), uses the same eight warm waves/reset and three untimed anchors, then 32 paid C4 waves (128 complete vocabulary rows), at positions 67+i+s. Record exact counts and endpoints 99/100/101/102.

Freeze complete checkout/source, prepared tree/toolchain build receipt, native executable, original unchanged client/image libraries, artifact manifest/index and input identities before inference. Acquire control first/repeat and candidate first/repeat. Require finite complete heads, same-policy whole-byte head repeats and all four initialized-state repeats before an exclusive own receipt is frozen. Do not require control and candidate states to match: both write real new rows. Preserve failed receipts and historical quality/calibration files.

Fund the retained 128 full heads and node-owned pinned state witness before admission. CopyState must retire before pinned-buffer reuse; retain the complete owner bundle on unproven teardown. The analytical context-256 state size is 57,671,680 bytes per owner across 60 K/V ranges; authenticate actual ranges/bytes from the runner rather than treat this estimate as measured occupancy or peak. Raw heads/states/logs remain owner-only outside Git.

## Fresh original comparison and paid bookend

Only after native own freeze, use the unchanged original b29c606e28a01b1bc8c1351026a0fa6e616bf6c4 image pinned at sha256:837fc732fea84b0d795097a3c8c5706bb16774f1722dab0f70bf6093c60aecc7. Fresh Gemma26 physical C4/ubatch128, context256/F16KV, fusion and graphs enabled; identical forced prefixes, publication and timers. Reauthenticate original binary/header/library and approved raw-GGUF metadata identities before a short reference/candidate/reference bookend. No new original math or eligibility override.

Timer pays all input staging, CONCAT packing, model work, completion, full-vocabulary publication and CPU argmax in both engines. Complete finite scans, initialized-state copies, file writes and hashes are outside the interval. Report native actual capture/replay separately from original GGML graph reuse; never equate supervisor wall time with model latency or GGML reuse with CUDA replay.

Compare all 128 full rows for actual byte identity, raw deltas, strict winner differences and reference-winner-over-native-choice margins; separate the first wave, first three waves and later units. TV/NLL only on explicitly designated rows (for example waves 0/1/2/31), with signed mean versus maximum absolute delta stated; no sampled corpus PPL or quality gate. Paid candidate heads/states must match its own frozen same-policy outputs. Preserve the fastest production reference settings and any exactness failure without adjusting bounds.

## Execution and boundary

After source review and locked build/descriptor controls, use the installed Spark A m3fixb GPU supervisor, timeout 600 and stop-on-fail. Expect roughly 20–40 seconds for four native arms and 15–30 seconds for the fresh three-arm bookend; announce exact jobs before launch. No C12, context/ubatch ladder or production selection. Freeze aggregate results/provenance and reproducible scripts, obtain independent GENERAL plus root ADV/light, and leave committing to root. Full-model qualification, optimized serving and assistant transfer remain separate tasks regardless of this result.
