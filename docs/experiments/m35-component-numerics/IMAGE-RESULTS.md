<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# U4 image transfer results

Both registered complete Qwen-Image cases pass source quality and the 10%
speed bound on Spark B. Native A and B produce identical pixels in both cases.
These results qualify the image transfer only; the later
[LLM results](LLM-RESULTS.md) and [U4 status](README.md) record the remaining
transfer controls and final integrated checks. The prospective protocol fixes the inputs,
comparison methods and bounds before these runs.

## Complete quality

| Case | A and B PSNR, RGB8 | A and B SSIM, luma | Native A/B pixels |
| --- | ---: | ---: | --- |
| 512 × 512, 2 steps, seed 43 | 43.3565548361 dB | 0.9979492076 | identical |
| 1024 × 1024, 40 steps, seed 42 | 41.2603289700 dB | 0.9941931912 | identical |

The fixed bounds are 32 dB and 0.98. Native token IDs match the reference.
The 512 case uses the vase prompt and the 1024 case the teapot prompt specified
in the protocol. Reference pixels are from the pinned official pipeline;
native pixels differ from that reference. The native RGBA8 SHA-256 values are
`58cc4f47b79a86c00be63e7c5b070177b24c000a89af6a6bafe34f7f209fa405`
and `93356574afbfb533bf234b2335f5b095319cf5163a588814360706e62c479478`,
respectively.

The 512 fixture-input receipt is
`da41873073df85454cafbab6ea683cc0f18321734354334e72e4fcfbbe018d78`;
its separately generated reference-output receipt is
`f50052d2a0e5c41341cc84264decdcc0ec2619480d0e898ec91671ecdb2a9af5`.
The 1024 case uses historical receipt
`fdf34f49865f4eadc86baf0f9f94d4a498377ac7ae1d8f8eb0c5843c11564edd`.
Inputs and output receipts have different purposes and are not interchangeable.

Final B quality jobs `m35-u4-b512-quality-frozen-r1` and
`m35-u4-b1024-quality-frozen-r1` recorded the frozen executable hash before and
after each run and retired successfully. They reproduce the earlier B pixels
and comparisons. Their quality-export timings are not timing samples.

## Matched speed and memory

Each process completes two generations: one warm-up and one measured generation.
The three process pairs use AB, BA, AB order, with exports and debug taps off.
Every pair has identical fixture identity, geometry, steps, token counts,
selected weights, admission footprint and completed work across its arms.

| Case | Pair 1 B/A | Pair 2 B/A | Pair 3 B/A | Median B/A |
| --- | ---: | ---: | ---: | ---: |
| 512 | 1.000262606 | 0.996829836 | 0.999825815 | 0.999825815 |
| 1024 | 1.005801558 | 1.004142739 | 0.998603181 | 1.004142739 |

Both medians pass the predeclared 1.10 ceiling. Per-arm medians below are
descriptive; the verdict uses the median of paired ratios above.

| Case / arm | Setup / ready, s | Denoising complete, s | Generation, s | Text / transformer / VAE, s |
| --- | ---: | ---: | ---: | --- |
| 512 A | 0.723609 | 0.442792 | 5.921717 | 0.077887 / 0.363253 / 5.478205 |
| 512 B | 0.694685 | 0.442435 | 5.911067 | 0.077955 / 0.362949 / 5.468012 |
| 1024 A | 0.902085 | 32.559220 | 33.074961 | 0.078538 / 32.478610 / 0.515233 |
| 1024 B | 0.867312 | 32.641214 | 33.156018 | 0.078790 / 32.559323 / 0.513435 |

Ready is the fixture's setup endpoint immediately before request execution,
not a separately measured HTTP readiness interval. Stage columns describe the
second generation and include the fixture's repeated text stage. Denoising
complete is the fixture's `FirstOutput` boundary, before VAE decoding; generation
time includes the completed pixels.

| Case | Paid footprint, both arms, bytes | A sampled unified peak, GiB | B sampled unified peak, GiB |
| --- | ---: | --- | --- |
| 512 | 35,973,665,840 | 31.0905–31.2994 | 31.2121–31.3755 |
| 1024 | 39,013,097,904 | 33.8077–34.0630 | 33.9262–34.1222 |

The unified peak is the system `MemAvailable` decrease sampled every 20 ms,
covering the shared CPU/GPU physical budget. It is an estimate, not exact device
occupancy. Child peak RSS ranges are 1,020,174,336–1,020,461,056 bytes (512 A),
1,020,096,512–1,020,289,024 (512 B), 1,127,051,264–1,127,407,616 (1024 A),
and 1,126,551,552–1,126,862,848 (1024 B). The paid footprint is an admission
bound rather than a measured peak. Neither quantity is a separate Spark memory
tier.

## Component and request controls

Component controls cover rows 17, 77 and 264, widths 256 and 4096, D128
attention, products, GLU and modulation, BF16 carry at width 8192, and refusal
above 8192. The registered oracle limits and stricter per-operation assertions
pass, with candidate selection checked. A singleton-layout defect found during
these controls was repaired in the shared layout/planner path before the final
candidate was frozen.

Job `m35-u4-image-request-ownership` passes two separate cases on frozen B:
the 2048 × 512 one-step request completes two generations and six stages;
the 1024-square replacement case completes three requests and ten stages,
releasing two request owners while reusing retained captures. These are
independent cases, not a combined rectangular replacement test. Their finite
pixels establish request/stage ownership controls, not independent source
quality.

## Reproduction identity

Runs use Spark B (`spark-56f5`), NVIDIA GB10, driver 580.178.04, and the
repository's pinned SDK. Baseline source is
`5ec811a4413982630c05e7b6c2448f51f1191883` plus the identical numerical fixture.
Candidate source adds patch SHA-256
`08c82f5832358a8c68489bfbb2b1a48757e71ec9ca5331293b459e8da3332a30`.
The fixture source SHA-256 is
`bf460f8ca5687ca63d7714d3dbbc0996b1343485b30c568307fab5f657b9c460`.
The preregistration commit is
`3965d69d439d1545b13bb77022a81ac45f8255df`.

The frozen image executables, retained in the original build directory so their
library paths remain valid, are:

| Arm | Filename under `build/spark-native/tests/unit/` | SHA-256 |
| --- | --- | --- |
| A | `m35-baseline-image-generation` | `efd4793a9595a8f8109717105c6509e83d4ae20679a786c1f2db33dbdea1245e` |
| B | `m35-u4-final-image-generation` | `237c57d0944c1213e96a3d1120fe30de45e732b155b33968ae551eb939c908d2` |

Raw receipts, samples, XML and pixels remain outside Git under
`~/.local/share/llmp/m35-u4/` on Spark B. Timing records are
`timing-{512,1024}-pair{1,2,3}/record.json`; final B comparison records are
`b{512,1024}-comparison-frozen-r1.json`. They are retained until milestone
closeout. This aggregate supplies the result without requiring those raw files.
Correctness eligibility over these domains does not claim every fusion is
fastest for every shape; shape-class selection measurements remain U6 work.
