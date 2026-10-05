<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Bounded Gemma 31B native runner

The dense 31B approved profile now executes through the existing `Gemma4Runner`,
with the same state, funding, catalog, graph and completion machinery as 26B.
Ordinary solo/wave replay and checkpoint/spill continuations pass. The first
same-format representative screen **fails** against the full-fusion reference:
PPL is **14.610785% higher**, with 330 strict argmax differences outside the new
zero native noise allowance. Gemma 31B remains unsupported; there is no 31B HTTP
route, assistant, selected optimization or optimized-batching qualification.

The [protocol](PROTOCOL.md) pins the complete source/prepared artifact,
TensorFold refresh, same-format llama image, corpus, tokenizer, noise freeze,
paid work and limits. The [transfer audit](transfers.md) records eligible and
ineligible Qwen/DeepSeek mechanisms and actual ordinary choices. Defaults remain
unchanged: device masks and requested frontier heads, ordinary products, all
optional norm/Q8/row/store/lane policies off.

## Numerical control

Fresh 31B tokenization supplies 1,024 common IDs including BOS 2, in eight
128-row teacher-forced calls with every full-vocabulary head returned.
Exactly 1,023 target transitions are scored with FP64 sums; the final retained
head and BOS are not scored. Changed argmax choices never change later prefixes.
Context is 4,096, F16 KV, one sequence, FlashAttention, no template/drafts/EOG.

Native ordinary, norm-fused diagnostic and independent ordinary repeat each
produce the same complete 1 GiB output, SHA-256
`87d2274ad1420412885cd118a33dc79e614cc743989f96eb0c953e5f8ffc3720`.
The new 31B calibration is frozen **before reference model scoring**, SHA-256
`6ab1fd0bc4a39c303c112ff476b96faa0527a9fa8242cc67b3abd5873350f038`,
p99 top-two margin movement 0, all 1,024 rows byte exact, no own argmax mismatch.
This is separate from 26B calibration and remains unchanged after the oracle.

| Arm | Mean target NLL | PPL | Difference from full reference |
| --- | ---: | ---: | ---: |
| Native ordinary 31B | 3.713697329892001 | 41.00513611237161 | +14.610784960847623% |
| Pinned full-fusion reference 31B | 3.5773256064214842 | 35.77772905610885 | — |
| Pinned fusion-OFF diagnostic 31B | 3.713697329892001 | 41.00513611237161 | +14.610784960847623% |

Full-fusion reference repeats all 1,024 complete heads byte for byte. Native
matches none of those rows byte for byte, with 330 strict argmax mismatches,
all outside the unchanged zero allowance. Maximum raw logit delta is
27.125676155090332, maximum full-softmax TV 0.9999985664465206, maximum chosen
NLL delta 28.484634767495475 and maximum scored target NLL delta 22.680170599418748.
Aggregate prose PPL does not replace these distribution differences.

The native/OFF diagnostic and its repeat match all 1,024 full-vocabulary heads
byte for byte, including signed-zero representation. Independent direct
whole-file hashing authenticates all three retained 1 GiB files at the same
`87d227...` SHA. This narrows the difference to the compared fusion-enabled
execution policies; it does not identify a defective reference kernel, prove
one normalization family sufficient, generalize the 26B six-row routed-input
attribution to dense 31B, or qualify the full reference/model. 31B has no MoE.
The later [dense norm control](../gemma31-reference-fusions/README.md) reproduces
all measured stock heads with those two norm fusion families; this is reference
attribution, with native implementation and qualification still owed.

## State and funding

The focused official control passes all 8 tests: six CPU/profile/refusal/ledger
controls, the existing 26B checkpoint/spill/peer regression and a new dense 31B
real-model test. The 31B control covers ordinary 1/2/4-request waves, exact same
shape repeated full heads and initialized KV bytes, capture/replay, checkpoint,
clear, spill/restore and exact continuing heads. It verifies zero expert slab/
pitch padding and no optional selected policy. These checks establish replay
and state ownership; no scalar-versus-joined equivalence or optimized batching
claim follows.

Source metadata accompanies checkpoint bytes. The existing 26B layout string
is unchanged; 31B's distinct string prevents cross-variant restore/adoption.
Nonzero valid footprints with wrong opposite-variant, empty and malformed tags
are rejected before mutation, in both directions, preserving cursors, state
extents, catalog occupancy and peer usability. Original position, footprint,
completed-copy and retirement guards remain. Runtime 26B still validates kept
record identity before the family adapter invokes the engine. Diagnostic raw
prefix checkpoints persist a bounded `.layout` sidecar and refuse untagged
historical saves; old receipts retain their original measured source.

Caller publication capacity is charged before allocation. Pinned checkpoint
buffers are catalog-funded and retained automatically if a copy's retirement
is unproven. The 31B harnesses use the same device VMM, growing state, registered
host/staging buffers, plan/graph accounting and clean refusal paths as 26B.
No whole-node physical peak-memory, maximum-context or multi-model budget
qualification follows from the planned envelopes or the short lifecycle test.

## Paid 8K baseline

The independently screened reference ubatches 128/256/512/1024/2048/4096/8192
complete paid 8K prefill in 13.0872/12.4148/12.6260/12.8311/13.0179/14.1329/16.6058
seconds. Select **256 as the fastest screened arm**, without constraining the
reference to native's 128-row envelope. This bounded screen does not prove a
global optimum over every possible reference setting.

The official A/B/A/B/A bookend completes every arm on common forced IDs:

| Arm | Paid 8K prefill, seconds | Subsequent 32 completed calls, seconds |
| --- | ---: | ---: |
| Reference 256 A1 | 12.3839 | 4.04028 |
| Native ordinary B1 | 13.7986 | 3.31346 |
| Reference 256 A2 | 12.3682 | 4.04535 |
| Native ordinary B2 | 13.8346 | 3.32141 |
| Reference 256 A3 | 12.4470 | 4.05598 |

Both arms warm weights on six discarded rows, reset, pay fresh 8,192-row
prefill, then append the same three untimed IDs and 32 paid IDs. Publication
of full-vocabulary F32 logits and completion waits remain inside both timers.
Native pays 64 prefill chunks and 63 extra intermediate heads; reference pays
only the requested frontier head. The distinct requested work is explicit,
with no subtraction from native latency. The source of the prefill gap is not
isolated. Faster native fixed-prefix calls do not waive failed quality or imply
generated-history throughput/optimized-batching qualification.

All five arms validate exactly 8,227 completed positions and 32 paid calls;
all forced IDs match the independently prepared common file. Both native
repeats and all three reference bookends reproduce their 32 recorded argmax
IDs and two retained complete heads exactly. Native differs from reference
at one of the 32 recorded argmax choices. These choices do not alter later
teacher-forced input IDs. Only prefill and final heads are retained: max raw
delta/TV/chosen-NLL delta are 0.9030848741531372 / 0.06173169807899873 /
0.13269519353393022 for prefill and 6.342041075229645 /
0.00010629039997664943 / 0.00010623151470312564 for the final head. No new
8K numerical allowance, full-row quality or memory gate is introduced.

The planned native budget is **31,460,337,344 bytes**: fixed mapped capacity
538,969,088; weights 18,895,339,520; twice the 2,390,753,280-byte state capacity;
1,048,576 caller publication bytes; and 7,243,473,600 bytes of derived retained
plan/graph capacity. The latter bounds 100 possible calls at a 2,194,992-byte
plan floor and graph ratio 32, rather than a fitted arbitrary reserve. It is
funding, not measured peak consumption. Actual graph counters are 33 captures
and 33 replays. Physical whole-node peak remains unqualified.

## Source, reproduction and checks

Measurements use an isolated tree based on `c8edc06` on physical Spark-b
`spark-56f5`. Its eight modified source/test files were frozen by the external
`source-manifest.json`, bundle `01f2d058b4e177b2de73c3b9e91a0add432b178f357e54cd4e16e55a5fc8aab7`
(sorted path, TAB, file SHA, LF encoding). The final handoff is based on
`24e9cbe` for its unchanged-source parent links and independent 26B diagnosis;
that documentation update does not relabel the measured native ancestry.

Successful locked build4 uses SDK `aarch64-c09daba6ac31edee`; 441 compile
commands reference the new `m3gm31` tree and the changed runner/runtime/tests
visibly rebuild. Quality harness binary SHA is
`7bd4f1b8f9f1aa8708be0c272ff10edf10e1e5a7488562ae4d32bd07f2a73250`,
paid-prefill binary SHA
`567e0480495bed0b71ea6b97594687c1f122e6c7ce07fdbd3facb8fc26d204d8`.
Pinned-image quality client SHA is
`e291d8864695c1e4c80a1324fb0758ce127700584384aa3069239ee28cf593f0`,
prefill client SHA
`bd2dd3aca6e1eeac227f18a3af6099c7a3a8631be7aae27206eb0bac50b28e56`.
The build receipt reports `0.1.0-dev+unknown`, origin `none`, because remote
Git metadata is excluded from source synchronization. The measured source
ancestry is authenticated by the isolated local Git tree, checksum sync and
source manifest, not a remote Git assertion. Both native measured binaries
are preserved separately under external `gemma31-quality/measured-bin`.
The exact b29 `llama.h` SHA is
`fedb52ea9291c9900e637ed6ffec339919dd27c3dadc2c8cf216c552e0488dda`.
Floating-point reference math remains in the unchanged digest-pinned image.

Reuse the committed `gemma_quality` and `gemma_prefill` native harnesses with
an explicit trailing `31` argument; omission preserves 26B. Build the existing
`gemma-quality/llama_quality.cc` and `gemma-prefill/llama_prefill.cc` against the
image's `/app` libraries using the pinned headers. Freshly prepare 31B IDs, run
native ordinary/norm/repeat and the analyzer's `calibrate` action in a new
external directory, then acquire reference `score`/`score-unfused` repeats and
run `oracle`/`oracle-unfused`. The existing analyzer authenticates the now
independently re-observed shared token-file hashes; its freeze is new 31B data.

The [reference wrapper](reference.sh) supplies the build and explicit-ID calls.
Populate each external scratch directory with `include/llama.h` and
`ggml/include/{ggml,ggml-alloc,ggml-backend,ggml-cpu,ggml-cuda,ggml-opt,gguf}.h`
from `git show b29c606e28a01b1bc8c1351026a0fa6e616bf6c4:<path>`;
keep the header basenames and authenticate the pinned `llama.h` hash above.
The combined wrapper reproduces the measured calls; the original split quality
and prefill wrappers remain with the external logs.

Raw data remain external under Spark-b `~/.local/share/jitllm/gemma31-quality`,
`gemma31-prefill` and installed job directories; local logs/receipts are under
`/tmp/jitllm-m35-coordination/gemma31-runner-raw`. Official successful jobs are
`m35-gemma31-build4`, `controls1`, `prepare1`, `native1`, `reference1` and
`analysis1`, `screen2`, `screen3`, `bookend1` and `compare1` with the same prefix.
All use installed GPU admission, 600-second
limits, stop on failure and official waits. No raw vectors are committed.

Historical failed build1 rejected copied absolute CMake cache paths before
compiling; build2 lacked the login PATH before compiling; build3 exposed a
missed benchmark API call, fixed before build4. The warm copy's generated
text prefixes were relocated only in the owned tree, then locked CMake
regenerated metadata; original source hashes, binaries and dependency data
were not rewritten. Paid-screen1 refused a malformed wrapper command before
inference; the corrected screen is recorded separately.

Verification: integrated Spark-b suite 1,615/1,615; local format, REUSE/header
and boundary checks pass.
