<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# First real global D51231 owner-root proof

Source base `ab88b9d`. This adds separate manual capture/replay derivatives;
committed Task20 and Task55 helpers and the owner-root kernel remain untouched.
No production selection, dynamic readable width or full-model owner-root factor
is included. Source and this recipe require review before any build or GPU job.

Capture closed dense31 C4: context 256 / max rows 128, independent prefixes 64–67, three
forced anchors and one final wave at positions 67–70. Ordinary products, normRoPE
and normADD on; normmul explicitly off, row products/MoE/rope-store off. The
explicit normmul value preserves the earlier carrier policy despite the newer
production default. Default capture policy/cached graphs remain unchanged.

The generic MMA wrapper skips multi-query scalar prefill calls without consuming
indices, then accepts only the first four one-query D512 calls each wave. Q's
view ancestry must contain the named `blk.5.q_rope`, confirming the first global
layer. It validates actual types/ne/nb and bounded real root offsets before
queuing pinned D2H snapshots on the authenticated held target stream. No keep,
allocation hint, fusion choice or attention launch argument is changed. Pinned
destinations survive every recorded graph and replay; save only after the model
job has proven completion. The stable heap bundle is retained on uncertain
teardown, including its host metadata charge; release that charge only after
successful teardown. Warm/reset/replay chronology uses the existing manual capture recipe.

After reviewed narrow two-target build and the existing 27 no-device owner
metadata controls, root releases four native capture/control processes. Capture
first/repeat and control first/repeat must retain the same finite four complete
heads and the same 4096-byte canonical IDs (`b2d7aaf6…`) in all four arms.
The adapter authenticates those IDs before reading any output payload. The capture adapter requires the layer 5 metadata and
all sixteen input payloads to repeat exactly, including cold unread KV padding
and padded mask columns. No original-model operands or outputs are introduced.

Per owner: F32 Q[512,1,32,1], F16 K/V[512,256,4,1], F16 mask[256,32,1,1].
Construct four-stream cell-major roots by concatenating the complete owner bytes;
no conversion/arithmetic. The packed carrier is Q 262144 B, K/V 4194304 B each,
mask 65536 B and parameters 64 B. Replay validates closed lengths, scale 1 / F32
parameters and finite Q/K/V; masks contain zero or negative infinity.

Root separately releases actual D512 owner first/repeat, packed first/repeat on
these immutable inputs. Preserve the original compiled packed grid/reduction
partitions; the owner specialization's occupancy may refuse but cannot alter
the grid. Eager/capture/fresh/restored full 65,536-value output checks, complete
packed and eight independent uneven-pitch KV byte witnesses, held closure,
funded host/pinned/mapped storage and uncertain-completion retention reuse the
Task55 helper. Every output is 262144 B. Operator publication/finite/argmax work is
paid; upload/packing/witnesses remain outside the timer, so operator timing
cannot establish removal of model packing costs.

No fresh original-image run, full model owner-root consumer, wider context or
whole suite is part of this first proof. D256 remains the earlier measured
operator proof; D512 arithmetic is unproved until this acquisition completes.
Raw inputs/outputs and official records remain external. TensorFold task entry
at 2026-10-06T00:08:29Z resolves primary `609ca419` / 0.6.5, actual Gemma recipe MLX26
only; no matching Gemma31 CUDA recipe.
