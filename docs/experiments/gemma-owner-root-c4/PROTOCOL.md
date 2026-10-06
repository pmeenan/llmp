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

## Closed Gemma26 backward transfer (Task68)

This derivative starts at `017c25e`, preserving the completed 31 sources and
receipts in history. Fresh primary TensorFold task entry resolves `609ca419` /
0.6.5: the actual recipe remains MLX26 only, with no matching CUDA recipe.
The transfer changes only these manual benchmark sources and their linker
observer. No core, CUDA launcher, operation API, default production selector,
cache ownership, read-width or context change is proposed.

Add the exact approved26 profile (width 2816, 30 layers, 16 query heads; local
D256 / KV8 and global D512 / KV2) to the existing wrapper. It still requires
four independent one-query owners, context 256 / max rows 128, readable KV256,
four canonical head rows and no features. Unsupported shapes return the exact
original builder. Thirty attention operations replace 120 owner attention
nodes, with all eight SET_ROWS dependencies checked per layer. Metadata
controls now admit both head counts 16 and 32 at D256/D512, retain all ten
source/tag/parameter/alias refusals, and exercise both eligible profiles and
owner/query-row fallbacks. The fixed arrays remain bounded by 60 layers and
extra descriptors remain arena-funded.

The new CLI is `ARTIFACT NEW_DIR 26 4 joined compound IDS_I32`. Both packed and
owner arms enable the existing routing and reduction fusions throughout scalar
prefill 64–67 and C4 decode, with actual first-built counters required at every
shape: 60 normRoPE, 90 normADD, 30 routing, 30 reduction and 121 plain norms.
All other optional policies stay zero. Plain norm defaults on for 26; explicit
`JITLLM_GEMMA_C4_NORMMUL=0|1` remains available for diagnosis. The 31 CLI and its
plain-norm-off default remain unchanged. Phases remain default off.

A benchmark-only link wrapper forwards the exact existing
`PlanFlashAttnOwners(const LaunchContext&, const FlashAttnOwners&)` call and
prints only the first successful head 16 D256/D512 plan. It records original
blocks/occupancy/columns/group/scratch and owner resource fit, changing none.
Planning derives geometry from the actual compiled original packed kernel and
real KV-head count; do not inherit 31 grids or label analytic estimates as
measurements. These bounded logs occur while constructing the first owner
plans, before the paid captured replay interval. No-launch metadata controls
cannot establish runtime geometry.

After source review and separately released narrow build, the **first native
arm is packed/plain-on**. Its full 128 finite heads must match the historical
compound hash `983388cd…`, and each of four initialized state witnesses must
match the historical 57,671,680-byte / 60-range identity. The inputs remain the
complete canonical 4096-byte `b2d7aaf6…` sequence; endpoints are 99–102. If enabling
plain norms changes this identity, stop the owner factor and report a separate
norm transfer rather than silently selecting a different recipe.

Only after that identity gate may the separately released paired
packed / owners / owners / packed same-binary runs measure K/V-copy removal.
Full 128 heads, all four states and 128 choices must repeat and agree across the
factor. Publication/argmax and all remaining packing stay paid; finite scans
and state/file witnesses stay outside the clock. The installed supervisor and
existing completion-aware ownership bundle govern every read. A fresh
physical C4/u128 original/current/current/original bookend requires its own
release after native evidence. Report all 128 strict choices, first byte
variation and only the existing selected 16 likelihood/TV screen; no PPL or
whole quality claim. The previous compound screen's two positive-margin misses
remain known evidence, not an expected correctness pass or an assumed cause.
Mixed original prefill routing eligibility at layers 28/29 does not authorize
keeping layers, imitating physical aliases or changing a production rule.
No wider-context, assistant, production batching or full-suite claim.
