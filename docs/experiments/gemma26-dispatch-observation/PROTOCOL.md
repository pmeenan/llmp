<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Gemma26 C4 stock dispatch observation

Observe stock routing, scaled ordered reduction and actual product choices at
the exact completed packed26 recipe before choosing another arithmetic factor.
This is a host-controller diagnosis, with no native policy or kernel changes.
Base 2eb0574. Fresh TensorFold primary observation at 2026-10-05 13:28:16 UTC:
609ca419abecebdc5a059498a613680bd3aa847f, version 0.6.5, Gemma26 MLX-only.

The logger accepts only original controller SHA
523470d6604755b82d0208414ce40f1378941b10bc1349763bbdf02edaab9634.
Removing every marked insertion reconstructs those exact bytes. No predicates,
allocation hints, graph readers, graph order, math launcher arguments or fusion
settings change. Extra host predicate evaluations are read-only; complete head
fidelity is required before attribution. Rebuilding the controller translation
unit also compiles its unchanged integer pointer-preparation kernel. Floating
model kernels, compiled with the original CUDA 13.3 toolchain, remain in the
pinned image library; the external controller uses the existing pinned CUDA
13.4 SDK and original image g++.

Use image sha256:837fc732fea84b0d795097a3c8c5706bb16774f1722dab0f70bf6093c60aecc7
and b29c606e28a01b1bc8c1351026a0fa6e616bf6c4 sources. The unchanged original
client 01889d8c4a281110f611cedcab43235a9ac80bcc21abf094d982f14dd2112f61
loads the approved Gemma26 raw GGUF, context 256 per owner, physical C4,
ubatch 128, F16 KV, fusion enabled and CUDA graphs allowed. Actual 1,024-ID
input SHA b2d7aaf6aa2ef06d82591a3794f36640e192f429539ec934fd74bf4d81df1610.
Owners prefill 64/65/66/67 rows; eight warm waves, clear/reset, three untimed
anchors and 32 forced C4 waves follow the committed recipe unchanged.

Log original route recognition, structural/shape/physical-memory gates and
selected outcomes. Log scaled-reduction match, node count, scale presence,
physical-memory gate and selected outcome. Record the branch actually chosen
for ordinary/routed products, including fused sites, with bounded tensor
type, dimensions, strides and physical address/span metadata. Keep addresses
and individual graph-call records outside Git. Aggregate by phase and layer.
The four prefill row counts exceed the topk alias allowance of eight; decode
has four rows. Prefill decisions may affect the KV state consumed by decode.
Do not carry the historical 128-row layer-28 refusal into this different shape,
reproduce its incidental allocation, or infer that mixed eligibility is a bug.

The logger emits at most 200,000 records and aborts on overflow. Graph call
headers distinguish evaluation/capture and unchanged graph launches. Predicate
and product counts describe host evaluation/build observations; they are not
counts of GPU kernels or CUDA graph replay. Traced timing is not competitive
performance evidence.

Before any model load, freeze source/generated-controller, original client,
headers, linked image libraries, SDK and approved input/artifact metadata
identities, after a supervised narrow build and source/parser controls. Preserve
the original client's full publication and explicit batch/context/model/public
backend retirement. Successful process and supervisor retirement are required.
Require all 128 full heads (134,217,728 bytes) to match BOTH untouched prior
stock bookends, SHA 5808306e0f261896500bdf9f3b44d4b1d0f18b34ba99d976da15a916aa8ff164,
and authenticate each actual input file. A mismatch is an intrusive diagnostic,
not evidence about the remaining arithmetic gap. Raw outputs/logs remain
external and owner-only; retain aggregate selection/fidelity results.

Use the installed Spark A supervisor, timeout 600, stop on failure, after
checksum source sync. Expected narrow compile and acquisition each take
roughly 10–20 seconds. Review source before inference. This task stops at the
validated decision table; compound-policy or block-operand probes require a
separate reviewed scope. No ladder, default change, alias whitelist or support
claim follows from these observations.
