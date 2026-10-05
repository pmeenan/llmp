<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# H1 dense31 8K sliding-cache reference recipe probe

Task-entry primary TensorFold refresh 2026-10-05T13:45:21.930892Z resolved
609ca419abecebdc5a059498a613680bd3aa847f, version0.6.5. Four primary files
freshly fetched at that pin; Gemma26 recipe remains MLX-only, no dense31 CUDA
comparator. Source base ccb2b1c; benchmark-only diagnostic. No production settings, kernel
policy or historical result is changed.

## Source premise and closed change

At original b29c606e28a01b1bc8c1351026a0fa6e616bf6c4:
- `src/llama-context.cpp:3652–3653`: public C-API defaults swa_full=true,
  kv_unified=false; context setup passes swa_full to cache construction at390.
- `common/common.h:572–573`: CLI/server common defaults swa_full=false,
  kv_unified=false. `common/common.cpp:1748–1749` forwards those fields.
- `src/llama-kv-cache-iswa.cpp:70–80`: ring capacity
  PAD(min(base_context, window*(unified?sequence_max:1)+ubatch),256), overridden
  to base_context only when swa_full=true.
- `src/llama-kv-cache.cpp:1250–1263`: read width comes from occupied maximum
  cell index, padded at least256 and capped by physical cache capacity.
- Native `src/model/gemma4.cc:489–490,573–596`: local capacity
  PAD(min(context,window+max_rows),256), writes position modulo local capacity;
  padded reads capped by that capacity, masks based on initialized positions.

For dense31/context16384/ubatch256/one sequence, both local capacities are
1280 with swa_full=false. That is a configuration agreement; occupied mapping,
read order, selected arithmetic and exact outputs remain measurement questions.
The full-cache references remain valid records of their actual recipe. No
native arithmetic defect or retroactive quality waiver follows this source fact.

Copy the existing original `llama_prefill.cc` into a dedicated experiment.
Close invocation to dense31/screen/ubatch256, set cp.swa_full=false explicitly,
cp.kv_unified=false explicitly, and report those effective flags. Preserve
all original math libraries, graph/fusion policy, paidwork and input positions.
Historical client, results, native benchmark and production files stay unchanged.
No operation observers, kept intermediates or kernel policy changes.

## Exact native provenance gate before ring stock

Prefer the retained original native executable c3d089a1f6db28fc8db75d113764a0f0c716e1291196d3aef3219aca02fb90f8
and measured benchmark source8fc6e4b377f5fa120d1dcc82f44531c2b201eb0462905fbe9ff1c68e45b772a2,
basec2d2147, locked receipt af910cb38aeeddf819af640e8ea6dcc1b97f594ec0100bd98a073c291ad135ea.
Reuse only after authenticating retained binary/receipt and the historical own
freeze bf7f95063dfad774871404bd97d6f3fce65c9ea868a077c8fe9817f006a5cf7e,
its source log and completed outputs. No model payload/historical rescans.
If exact native binary is unavailable, build the unchanged native source on the
new integrated base and acquire fresh BOTH256 first/repeat heads+initialized
state, then create exclusive own freeze BEFORE any ring stock load. Do not
apply old calibration to a rebuilt binary or ratchet a bound after results.
Either path freezes new client source/binary, all four original library/header
identities, IDs, artifact manifest/index and actual build/source ancestry.

## First screen and paid work

Owned B m3gm31, installed GPU supervision, timeout600. Fresh ring reference /
unchanged native BOTH256 / fresh ring reference, distinct new output directories.
One bounded bookend only; no depth/context ladder, memory sweep or new kernel.
All use 8227 LE I32 IDs SHA6b6567ca51a3fbe5000521cb71fcf168ef485623bdbfea2abab30d57f414d96b,
BOS2 once, approved dense31 artifact32c92e and matched GGUF9e92cb source.
Context16384, F16KV, one sequence, native max_rows256, original batch8192/ubatch256.
Native existing normBOTH choices120ROPE/120ADD, other optional choices0.
Original fusion/graphs allowed; native device masks and captured plans unchanged.

Six discarded warm rows then clear. Paid prefill8192 includes actual plan/capture,
state growth, staging and full-head publication: native32 heads, reference one
final requested head. Three common untimed anchors to endpoint8195. Paid32
one-row units through endpoint8227 each argmax the INCOMING head before appending
the supplied input, as both existing prefill clients do. This differs from the
C1 helper's post-wave argmax and must not be mixed. Do not add32 retained full
heads or change timer costs. Setup/model loading, disk writes and state snapshots
remain outside timers. Original public context creation logs must establish the
actual local1280/global16384 capacities; recipe flags alone are not occupancy.

Compare32 strict greedy choices with exact forced-input and position alignment;
repeat equality per engine. Compare complete finite prefill/final heads (TV/raw,
bytes, selected reference-argmax NLL as labelled) only, not an all32 vector claim.
Native full initialized snapshot1740636160B/layout
`gemma31-f16-kv-scalar-device-v1:16384:256:16384:1280`, two fullheads and choices
must match its authenticated own freeze/log. Reference own bookends establish
new-recipe repeat identity separately. Strict differences are not automatically
outside a calibrated margin bound; only two retained heads supply raw movement.

Native snapshot storage remains completion-proven/catalog-pinned and existing
whole-owner quarantine retained; original library teardown must retire successfully.
No new funding/lifetime/timer implementation. Funded capacity is not measured
physical peak. No claim of corpus PPL, full-support/state restoration or production
selection. Any agreement is limited to this new recipe and representative short8K
screen; a failure stops this axis before a ladder. Historical full-cache speed
and memory observations remain attached to that topology.

## Proposed parent documentation after reviewed implementation

Add to `docs/reference-comparisons.md`: baseline recipe records explicit cache
topology, swa_full/kv_unified, chunk/window/read policy and actual capacities.
Competitive timing should use matched production CLI/server settings and disclose
any difference. Historical nonrepresentative full-cache timings remain measurements
of their recipe and do not establish representative memory ratio/speed parity.
Link the new evidence without rewriting historical clients or reported numbers.
Gemma26 transfer is a separate representative profile/ubatch/cache control, not
an inference from this dense31 result. Qwen/DeepSeek applicability depends on their
actual state and attention contracts; no default policy transfer by analogy.
