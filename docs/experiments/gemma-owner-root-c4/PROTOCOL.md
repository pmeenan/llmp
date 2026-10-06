<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Closed Gemma31 C4 owner-root consumer

Initial source entry is `5cf8f5b`; additive Task60 adoption moves this tree to
`fbac6dc` before source freeze. Actual TensorFold HEAD at entry is `609ca419` /
0.6.5, observed 2026-10-06T00:34:25Z. The primary Gemma recipe remains MLX26 only,
without a matching Gemma31 CUDA recipe. This is source-only until root review;
no build, model acquisition or production selection is authorized by this file.

The explicit custom operation takes ten real source descriptors: packed Q,
packed mask, K0–K3 and V0–V3. Each cache view sources its exact current SET_ROWS
writer and has the same owned leaf as its physical view root. Generic placement,
dependency ordering, coverage and lane access walk all ten sources. The distinct
registry entry validates complete typed spans, current addresses and aliases
through the existing owner-root checker. It does not reinterpret a legacy FLASH
descriptor or invent a contiguous KV allocation. Fixed scale 1 / F32 query
conversion and reduction arithmetic belong to the existing launcher; CUSTOM tag
parameters remain intact. Both scratch-planning modes use its checked plan,
whose grid comes from the original compiled packed kernel rather than the new
kernel's occupancy. No default graph emits this operation.

Separate manual targets derive the previously measured packed builder/helper;
historical capture, replay and packed files remain unchanged. The process fixes
`JITLLM_GEMMA_OWNER_C4=packed|owners` before any plan construction. Both arms
pack Q and padded masks identically. Only the owner arm omits balanced K/V
CONCAT trees and reads eight actual independent writer-backed views. Supported
graphs are dense31, context 256, max rows 128, four independent one-query rows,
256 local/global readable cells, four heads requested and no features. Other
shapes return the original builder unchanged. C1 prefill stays unchanged.

All extra descriptors use GraphTensors/Reserve/TensorArena funding. Replace the
four original attention results by stable owner views, rewire source/view/name
edges, remove original attention nodes from executable roots, then rebuild the
bounded arena-funded order. Check all eight writes precede each new attention,
60 attention operations remain, and the owner slots retain their order. Runtime
plans must keep 120 normRoPE / 120 normADD and all other optional policies zero.
Ordinary products and normBOTH match the prior C4 recipe. The immutable optional
process flag `JITLLM_GEMMA_C4_NORMMUL=0|1` defaults to zero and selects only the
existing plain RMSnorm/MUL fusion. The first copy-removal factor uses zero;
its exact recipe remains unchanged. A separately reviewed transfer screen may
use one, requiring 121 selected plain-norm fusions instead of zero while
preserving 120 normRoPE / 120 normADD and all other optional counts. Log this
mode explicitly; these are selected plan implementations, not CUDA launch counts.

No-device metadata controls cover both profiles, owner counts 1/2/4 and query
rows 1/2, including eligible transformation and exact unsupported fallback.
Additional custom-node controls reject wrong tags, a missing tenth source,
unsupported parameters, wrong output geometry and output/cross-owner aliases
for D256 and D512. No shader or original kernel arithmetic changes.

After reviewed source and narrow build, acquire only the separately released
four native arms: packed first/repeat and owners first/repeat. Canonical 4096-byte
IDs have SHA `b2d7aaf6…`; the helper verifies their complete hash before opening
the model. Prefixes 64–67, three forced anchors, 32 paid decode waves and context
256 keep every cache read at 256 cells. Full publication and argmax are paid;
complete finite scans and funded four-owner initialized-state witnesses are
outside the timer in both arms. Expected completed endpoints are 99–102.
Each policy's complete 128 heads and initialized states must repeat exactly;
cross-policy comparisons are recorded without silently widening tolerance.
Own-source/input/output completion is fixed before any fresh original C4/u128
counterpart bookend. Installed Spark supervision and proven node teardown govern
all output reads; unknown completion retains the complete ownership bundle.

The operator proofs at D256 and D512 justify this consumer experiment, not its
speed or model qualification. A fresh physical-C4 original/candidate/original
bookend requires separate root release. It pays all remaining packing, full
head publication and argmax in the same format. No full suite, context ladder,
production default or assistant inference claim. Actual Gemma26 backward
transfer follows a useful Gemma31 result and requires its own controls.

The subsequent [phase diagnostic](../gemma-owner-c4-phases/README.md) adds
`JITLLM_GEMMA_C4_PHASES=0|1`, default off, to the same helper. It resets counters
before paid waves and reads them after the timer; it changes no graph policy.
