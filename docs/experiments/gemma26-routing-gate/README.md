<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Gemma26 current routing refusal

The original controller refuses layer 28 routing fusion because its weights output
would overwrite computed router logits that the fused operation still reads.
All 30 actual 1024-row candidates passed structural and shape eligibility;
only layer 28 failed the memory gate, leaving 29 selected routes. This is an
actual placement-dependent refusal, not a model-layer whitelist or shape ban.

| Layer28 descriptor | Offset from logits, bytes | Public span, bytes |
| --- | ---: | ---: |
| Computed router logits | 0 | 524288 |
| Normalized routing weights output | 4096 | 32768 |
| Selected expert IDs view output | 524288 | 523808 |

The entire 32768-byte weights span overlaps the logits span. The IDs span begins
at the logits endpoint and does not overlap it under the original half-open
range test. Public allocated span equals `ggml_nbytes` for these descriptors;
this refusal does not require extra backend padding. The logits are computed
outside the ten-node routing chain (`MUL_MAT`), so they are not an elided internal
source that the memory gate may ignore. All 29 other candidates passed memory;
only layer 28 had weights/logits overlap.

The pinned original gate permits logits/output alias only when rows<=8:
one block then reads all logits before writing. This 1024-row arm exceeds that
exception, and the actual overlap makes its original memory predicate false.
The general rule is about dependency lifetime and allocated spans at the actual
placement. It does not justify hardcoding layer 28 or assuming its refusal at
other row widths, contexts, graph readers or allocation layouts.

The reused rich host observer preserved every original controller byte after
removal of marked logging insertions, including original allocator dependency
hints. The unchanged scorer produced all 1024 complete heads, 1073741824 bytes,
with SHA `ce5fd2ecd80d64f2e53252a8fe2be39f814881ea6e4e2d20a1f39aeed5b7be9f`,
exactly the frozen [Task57 reference](../gemma-current-quality/README.md).
Actual logs confirmed score-ring 1024/context 4096/all outputs, F16 local 2048/global 4096,
SWA-full=false and unified=false. FlashAttention, original fusion and graph support
were retained. Checked scorer exit followed public backend teardown, the owned
Docker container was absent, and both supervised jobs retired DONE0.

[Task63](../gemma-keep28-routing/README.md) already showed that preserving the
layer 28 primitive route resolves the nine corpus disagreements byte-for-byte.
This observation explains the reference's refusal reason for that factor.
Native placement and its checked fusion eligibility differ, so neither this
specific external-reader diagnostic nor the stock layer list is a general native
production policy. This observation does not qualify a native policy or other shapes; no numerical
allowance, default, performance or corpus score was changed here.

## Reproduction and provenance

Reuse the compiled Task39 controller 306e33c7 at
`~/.local/share/llmp/gemma26-dispatch-observation/controller.so`, verifying its
full identity in [results.json](results.json). Actual generated source is
`controller-v2.cu` 4c042fcd, which reconstructs original 523470d6 exactly;
historical raw `controller.cu` is the earlier version and is not this binary's
source. The [existing generator](../gemma26-dispatch-observation/observe_controller.py)
records the pinned original. No compiler ran for this observation.

Use the unchanged Task57 scorer 225c40af/IDs b2d7aaf6 and original b29 image 837fc732.
Adapt [the owned reference wrapper](../gemma-current-dispatch/reference.sh) only
for a new output/CID name and private 306e observer; remove the unused policy/trace
environment variables (this rich observer retains unconditional original selectors).
The exact reviewed wrapper 8de6dce5 and source/recipe authentication identities are
recorded in results. Run once under installed `spark-job start --gpu --timeout 600
--stop-on-fail`, then authenticate full head SHA, actual recipe/cache logs and owned
container absence BEFORE interpreting gate records. Source JSON 53712b0a and original
native zero-repeat freeze f2885242 remained unchanged. No PPL scan, recalibration,
new native arm, timing comparison or regression suite was run.

Fresh TensorFold primary 609ca419/version 0.6.5 was checked at task entry; its Gemma
recipe has no applicable CUDA comparator. The observer was historically built
with NVCC 13.4.92; floating operator libraries remain the pinned original CUDA 13.3
image. Raw heads, logs and process addresses remain external. Git retains only
relative spans, counts and provenance; Spark B was free after retirement.
