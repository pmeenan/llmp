<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Gemma26 late-prefill common-input MoE controls

Native routing and scaled reduction match the original library byte-for-byte
under each tested policy on actual captured inputs from layers 28 and 29.
The stock recipe uses primitive routing and fused reduction. Switching those
recipes changes rounding but leaves all selected expert IDs identical. This
proves the ports on these operands; it does not explain the two remaining
choices in the [compound C4 screen](../gemma26-compound-packed-c4/README.md)
or select a production policy.

| Complete comparison, both layers | Result |
| --- | --- |
| Native primitive versus original primitive, routing and reduction | Byte-exact |
| Native fused versus original fused, routing and reduction | Byte-exact |
| Original primitive routing versus captured stock routing | Byte-exact IDs and weights |
| Original fused reduction versus captured stock reduction | Byte-exact full sums |

| Original fused versus primitive | Layer 28 maximum absolute delta | Layer 29 maximum absolute delta |
| --- | ---: | ---: |
| Routing: all 512 weights per layer | 1.1920928955078125e-7 | 5.960464477539063e-8 |
| Reduction: all 180,224 sums per layer | 2.9802322387695312e-8 | 5.960464477539063e-8 |

All 512 selected IDs per layer agree across routing policies. Each native
policy repeats its full outputs exactly across eager execution, three captured
replays, fresh layer-29 inputs and an independent process. Original controls
repeat exactly too. Complete input-byte witnesses stay unchanged. Actual plans
retain all 10 routing or 17 reduction descriptors; primitive routing executes
six steps/captured nodes, primitive reduction nine, and each fused path one.
The original primitive controls call six or nine actual single-node exports.

The capture is owner 0's second, post-reset 64-row prefill in the unchanged
physical C4/context-256/F16-KV recipe. All 128 complete heads remain byte-exact
with both untouched stock files (SHA 5808306e…), canonical inputs remain b2d7…,
and every evaluated phase's routing/reduction/product decisions match the
[dispatch observation](../gemma26-dispatch-observation/README.md). It preserves
all original fusion predicates, graph policy and allocator hints.

All 14 finite/canonical snapshots total 13,058,048 bytes. The selected call
actually uses `use_graph=0`, `update=0`; all D2H copies have stream capture status
NONE. Stable pinned copies retire after explicit public/backend teardown and
final device completion, and each owned Docker container is checked absent.
The active-capture D2H branch was not exercised. No timing or memory claim
follows from this instrumented run.

Routing uses packed F32 logits `[128,64]`, a funded full-128 ARGSORT root with
512-byte row pitch, its first-eight ID view, and clamp 2^-14 normalization.
Reduction uses coherent captured F32 down/scales/original weights at
`[2816,8,64]`, preserving ascending slots and every scale. The probe starts after
the actual quantized products and measures no gate/up/down product arithmetic.
Native whole-model frontier prefill executes the final-layer FFN at one row;
the captured original layer-29 FFN has 64. This operator replay does not establish
native whole-model layer-29 geometry or a unique history cause. Layer 28 may
feed layer-29 cached K/V; layer 29 does not feed a later prefill layer's cache.

Measured base is 23de091bf5c73450a8df8255eefe7b895dff1627, source frame 7b08b869…,
native executable 7dd94203…, and compiled PRE 0fc02cef…. The original controller
reverses exactly to source 523470d6…; floating operators resolve in immutable
image library 5a13585e…. Rebuilt code comprises the controller including its
unchanged integer pointer-preparation kernel, thin clients and host hash/check
helpers. Native SDK is CUDA 13.4.92/Clang 22.1.8; original CUDA 13.3.0/CUDART
13.3.29-1 facts are inherited image metadata, not an original floating-TU build
receipt. Actual native defining-TU compile entries and all eight copied public
headers are authenticated. Final docs and additive CMake preserve the measured
source frame externally; no acquisition code or binary changed after exposure.
[Provenance](provenance.json) records the exact identities and blind boundaries;
[results](results.json) contain aggregate comparisons only.

Follow the [closed protocol](PROTOCOL.md) with the installed Spark GPU supervisor,
600-second timeout and stop-on-fail. Supply the already verified raw/artifact and
b2d7 input file from the previous C4 recipe; raw captured operands/outputs remain
under the owner-only `~/.local/share/llmp/gemma26-late-moe` directory on Spark A.
Keep Python bytecode out of the source directory:

```sh
python3 -B docs/experiments/gemma26-late-moe/source_snapshot.py CHECKOUT NEW_EXTERNAL_MANIFEST
python3 -B docs/experiments/gemma26-late-moe/test_controls.py
bash docs/experiments/gemma26-late-moe/reference.sh build
build/spark-native/benchmarks/llmp_gemma26_late_moe --metadata
python3 -B docs/experiments/gemma26-late-moe/prepare.py SOURCE_MANIFEST NEW_PRE_CAPTURE_RECEIPT
```

Capture admission requires complete stock-head/phase fidelity. Release only
logits first, then freeze native routing before exposing original IDs/weights.
Reduction input release requires that admitted routing freeze; native reduction
must freeze before original sums. Every gate validates externally fixed prior
receipt SHAs before parsing or reading outputs, and comparison requires all four
PRE/native-own/capture/original-admission identities. Never derive a prior gate's
expected SHA from the current candidate file. Actual official timestamps prove
both native own-freezes preceded the corresponding original acquisitions.

The locked narrow build, four no-launch metadata plans, 14 parser/admission
controls, capture/fidelity gate and all primitive/fused own and counterpart
acquisitions/comparisons passed with supervised retirement. Raw tensors, logs
and addresses stay external. There is no model quality/PPL, performance,
production fallback change, layer whitelist or support claim.
