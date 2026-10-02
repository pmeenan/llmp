<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# DeepSeek inverse rotation and output-A prefix

The existing ds4 output-A core is a useful private performance factor:
it combines inverse RoPE, F16 input staging, Q8 weight dequantization and
the grouped product, emitting the low-rank result in canonical order.
The whole 8K native pipeline gains 4.08% throughput even when every
invocation pays weight-plane packing and an extra copy back into the
original graph's output. Its arithmetic differs from native MMQ.
The original-checkpoint 32K forced trajectory passes the registered greedy
bound and repeats all full logits exactly; its native baseline fails one
step. These are focused attribution/quality results; no production dispatch
changes here and broader acceptance is separate.

## Captured first-layer screen

The prior ordered profile charged 392.734 ms to inverse RoPE, 804.997 ms
to output-A and 97.567 ms to the layout copy across the 8K pass: 1.295299 s
of a 13.308827-s whole wall, a 9.73% ideal ceiling rather than a predicted
gain. A copy-only change could remove at most 0.74%, so it was ruled out
without another build/model run.

The ordinary graph supplies one 4,096-row window layer's unrotated F32
attention heads, I32 positions and raw Q8_0 output-A weights. A clone
retains the native inverse-RoPE → grouped MMQ → layout-copy descriptors,
selected implementation names, dimensions, strides and operation parameters.
The candidate uses the existing `RunDs4OutA` core with no D4 emission.
Every timed call includes raw 34-byte Q8 block packing into separate
scale/code planes and the kernel's rotation-table zero/preparation.
All physical planes are checked against the original weight bytes.

| Seven-repeat median | Native before | Candidate | Native after |
| --- | ---: | ---: | ---: |
| Paid wall, ms | 14.774397 | 8.115490 | 15.079596 |
| GPU interval, ms | 14.763840 | 8.109536 | 15.032960 |

The candidate reduces prefix latency by 45.6321% (+83.9322% throughput);
the native bookends move 2.0657%. Both complete native outputs repeat
byte for byte. Across all 33,554,432 candidate low values, 33,554,345
change: maximum absolute difference 0.031732, RMS 0.0043611, NMSE
0.00010647. These are descriptive arithmetic differences, not a quality
gate. The unchanged original graph resumes after the screen and produces
both known native full frontier heads exactly.

Common clone allocation/input setup is separately recorded at 55.370 ms
and 1,377,845,248 bytes. Candidate extra allocation plus first packing/run
takes 20.528 ms and adds 170,917,888 bytes. Native scratch is 151,013,376
bytes within the existing 190,840,832-byte pool. The allocation figures
are outside the repeated operator intervals; no cold-speed claim follows.

## Whole 8K factor

Three fresh Spark A processes use the same community IQ2_XXS artifact,
literal 8,192-token input, 8K state ceiling, two 4,096-row chunks and
complete 129,280-value F32 frontier head per chunk. All retain the actual
selected Q-head fusion, native sparse attention/cache stages, compact
experts, ordered reduction, head and original output-B. HCA and Q2 D2R
remain off. The two ordinary plans are prebound outside the paid clock;
input assembly/copies, every layer, dispatch, fences and head copies are paid.

| Original before, s | Candidate, s | Original after, s | Throughput gain | Bookend movement |
| ---: | ---: | ---: | ---: | ---: |
| 12.376010162 | 11.915388466 | 12.428143403 | 4.08454% | 0.42124% |

The original placement can reuse the attention input once native RoPE
finishes. The diagnostic therefore uses one disjoint candidate low buffer
and a paid same-stream copy into the original CONT destination. It keeps
the original 3,527-step plans and activation placement in both arms.
One 170,917,888-byte scale/code/table/low allocation is initialized before
the paid clock in both arms (6.09–6.56 ms). Every one of the candidate's
86 layer/chunk prefixes pays packing, table zero/preparation, the core
and its 134,217,728-byte copy back. There are no persistent 43-layer
weight replicas, prepacked-weight credit or allocations in the hot loop.
All arms retain 1,530,920,960 activation bytes, 190,840,832 scratch bytes,
264,126,464 state bytes and 207,618,048 host staging bytes.

All four ordinary full heads match the previously qualified native hashes.
Both candidate vectors are finite and contain nonzero values; all 129,280
logit values change in each chunk. Maximum absolute differences
are 0.985439 / 0.587285, RMS 0.146911 / 0.116283 and NMSE
0.000945511 / 0.000408083. The two argmax IDs remain 1393 / 554.
These two frontier predictions do not establish task or context quality.
One bookended pass establishes a representative speed signal, not a
confidence interval, context matrix or same-quality acceptance.

## Accuracy diagnostic and transfer

On 72 fixed coordinates from the captured first-layer operands, a common
FP64 inverse rotation of the original F32 heads followed by exact original
Q8 scale/code products and `math.fsum` gives native MMQ RMS error
0.00365744 versus candidate 0.000135466, about 27× lower for the candidate.
Maximum errors are 0.0109526 / 0.000313467. This small numerical control
supports activation quantization as a cause of the native/candidate
difference on these points. It does not replace model quality tests.

The reusable pieces are inverse rotation during product input staging,
register dequantization of Q8 weights, and direct canonical output.
This measured core requires G8, K4096, rank1024, head512 and normal64-tail
rotation. Qwen3.8 and EXL3 use different projection/quantization contracts;
they cannot select this kernel by name. Compare eligible preparation and
layout pieces independently, and preserve their own precision, strides,
workspace and completion contracts when transferring the technique.

## Focused 32K quality method

The fixed-history comparison uses the original
`unsloth/DeepSeek-V4-Flash-0731-GGUF@fbbb5b93` UD-Q2_K_XL artifact
`8a355bfb…`, whose output-A projections have the same guarded Q8_0
geometry. The saved oracle is llama.cpp b11254 (`8019dc563`), flash
attention, all layers on GPU, one sequence, context 262,144, no prompt
reuse, fit off and five log-probabilities. jitLLM holds capacity 32,768,
chunk size 4,096, compact scheduling, frontier heads and HCA off in both
arms. Seven full chunks replace 301 output-A prefixes in the candidate;
the existing 3,032+1 remainder and 511 one-row decode steps stay native.
Both fresh processes consume the identical 31,705 prompt IDs and 512
forced oracle IDs and retain 512 complete 129,280-value F32 logit rows.

The registered oracle near-tie bound remains 0.947 nats. A differing
argmax in the oracle's top five has an exact oracle margin. An argmax
absent from those five has only a lower bound from the fifth log-probability;
that can establish a violation when above 0.947, and otherwise remains
unresolved. Baseline and candidate verdicts are separate. Direct
native/candidate vector drift and native margins are descriptive controls.
This screen does not measure perplexity, real-answer quality or a new
matched performance comparison.

The first fresh ordinary/candidate runs give:

| Against the same-checkpoint oracle, 512 rows | Equal | Within-bound near ties | Outside bound | Unresolved |
| --- | ---: | ---: | ---: | ---: |
| Current native, HCA off | 492 | 19 | 1 | 0 |
| Output-A candidate only, HCA off | 490 | 22 | 0 | 0 |

The native exception is step 249, absolute position 31,954: native chooses
10386 while the oracle chooses 82437 with an exact 2.616249211-nat
margin, exceeding the bound by 1.669249211 nats. The candidate chooses
82437 at that step. Its largest oracle near-tie margin is 0.725294083
nats, 0.221705917 below the unchanged bound. All candidate differences
have exact oracle margins; no top-five lower bounds are needed in this run.
The native baseline at this chunk size therefore fails the fixed oracle
bound, while this candidate passes it. The raw aggregate `quality_pass`
field requires both arms to pass and is false; individual verdicts are
reported separately.

Direct native/candidate argmaxes agree at 489 of 512 rows. Their 23
disagreements have native margins 0.00295639–0.81016922 nats. Across all
66,191,360 values per arm, 66,191,323 numeric values change; RMS is
0.298776545, NMSE 0.002602943 and maximum absolute difference 8.41839504.
Both full vectors are finite and their actual 7/2/511 chunk counts and
301 candidate prefixes match the protocol. Prefill/decode clocks are
48.3284/24.3297 s for native and 45.7612/24.4302 s for the candidate;
these single quality-run clocks are descriptive, without a new bookend
or matched throughput claim.

One fresh no-build candidate repeat produces all 66,191,360 logit values
byte for byte and the same 512 argmaxes, actual execution counts and
budgets. Its complete-logit SHA256 is
`dcd660fb70f8bdc58b2df616d8454a0b5f478f7900d720db1c29b1a8c1c6e9a8`.
The two candidate processes therefore pass this fixed-history
greedy/repeat control. The repeat's prefill/decode clocks are
45.6086/24.2922 s, again descriptive. No broader answer, PPL or
context-quality conclusion follows from the isolated FP64 sample or this
one fixed trajectory.

## Provenance

Measured on `spark-c4e2`, 2026-10-02, SDK
`aarch64-e0a0c85c42806fb1` (Clang22.1.8/NVCC13.4.92), SASS `sm_121a`,
the actual native math flags and pinned cuBLAS payloads. Original source,
static/compiler/library identities, fixed inputs, complete outputs and
successful supervised retirement are retained. The final whole-factor
gate finds 117.129 GiB available; fixed-quality and fresh-repeat terminal
gates find 117.324 / 117.290 GiB. The completed whole/quality/repeat
comparisons finish rc0,
with no native/GPU/container model processes and `spark-job busy` free.

- Artifact manifest ID: `cd39d504dc2dbfe911a4a521fa8efc8053dc3e80e99738a9b25fa6b70c97a1ac`.
- Literal input SHA256: `0cbafc4bafd7a2c1e1c83f4a346815cdd500d2fb0bcb45afcfd826d591ae8e9a`.
- Captured heads SHA256: `6e1823b781b60f3999daff6341f0421d8d3ab3804e51993b95e1dec33452c0c4`.
- Captured raw Q8 SHA256: `aab88a1dd63fe965d6148f1cdefee9bbe8d01316eeeab63169f73acb91f50c51`.
- Original quality artifact ID: `8a355bfb27c90e1150fbd7fa62ea6e63f6bf34fcca33934e52d22773f1508234`.
- Quality prompt/force SHA256: `0961e248af647f3e9c838d43ba2cbad821fc48a7e9f066679ae7b2e8f118e6cb` /
  `e70be6122f5df8d83357b8223aab63d0f826cdd1dac85760e759cd574471480a`.
- Saved same-checkpoint oracle SHA256: `49de2d51973b4bef32cf174c861d44d74bf63b6ec8ee576641dcba8bcc042aa6`.
- Raw local evidence: `/home/pmeenan/scratch/m3-ds4-qhead-short-records/outa-operator-r3/`
  and `outa-whole-r1/`, `outa-quality-r1/`, `outa-quality-repeat-r1/`;
  sources/results on Spark are under
  `/home/pmeenan/scratch/m3-ds4-qhead-short-r1/`.

The first attempt stopped before model loading on routine host warnings.
The second captured window-layer operands but refused before timing because
the diagnostic wrapper requires positive beta metadata even with YaRN off.
An explicitly recorded 32/1 sentinel substitution applies only when
extension and both original betas are zero; the original table kernel
does not read those betas in that arm. Native descriptors remain unchanged.
Both failed supervisors retired; their records remain alongside the screen.
No new unit suite was run for these private diagnostic screens. A focused
32K fixed-history comparison uses the original UD-Q2_K_XL checkpoint and
its existing b11254 oracle; the community speed result above remains a
separate checkpoint/shape measurement.
