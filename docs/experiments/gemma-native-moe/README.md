<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Checked native Gemma MoE dispatch

Explicit default-off routing and scaled-reduction policies now bind the
original pinned GGML launchers through the native registry. A graph-order
change makes the complete ordered routed sum contiguous without changing its
arithmetic. The ordinary 26B output remains byte exact to the historical
1,024-head file. The compound norm/MoE policy still fails the representative
quality control: nine greedy differences exceed the unchanged allowance.
A short native-only speed gain does not select a serving default.

## Contracts and state

`DeviceChoices::fuse_gemma_route` and `fuse_gemma_reduce` are independent false
choices, mirrored by engine diagnostic options. Broad fusion remains false.
Routing requires the exact ten-node 128-expert/top-eight softmax/normalized
weight chain and `2^-14` denominator clamp. It initializes both selected IDs
and weights. The full 128-entry ARGSORT root is still funded, but only its
first eight IDs are written. Full-sort, probability or normalization readers
and keeps require primitives. The reduction requires seventeen descriptors:
`(expert * selected_scale) * routing_weight`, eight selected-slot views and
seven ascending additions at width 2,816. Keeping a product, contribution or
partial sum requires its ordinary producer chain.

Both registered operations retain every descriptor for placement and recheck
structure, bounds, aliases and current views at bind/run. Bounded graph/keep
preflight precedes storage scans. The launchers use zero scratch, the original
fast-math source, and jitLLM's stream/completion ownership. Logical operand
spans describe required backing; callers still fund residency and protect it
through completion and replay. This is not activation-storage elision. The
routing tail is protected against readers and aliases. Dense31 selects neither
MoE operation.

The raw routed sum is expanded before its post-normalization, moving the seven
additions ahead of independent shared FFN work. Post-norm/residual eligibility
is retained. Actual 26B graphs have 30 routing and 30 reduction matches for
1/2/4 independent segments. CPU controls cover default/keep fallback, malformed
metadata and post-placement `SamePlan` with complete sort backing. Registered
GPU controls cover rows 1/2/4/8/128, deterministic ties, both outputs, untouched
tails, fresh captured operands and clean stale-binding refusal. Complete
local/global layers and real 26B joined own-replay, initialized KV, checkpoint,
spill/restore and peer preservation pass. Same-policy joined repeat is not
scalar-versus-joined reference qualification.

## Representative numerical control

The approved 26B artifact, same-format pinned llama.cpp image and exact
War and Peace token fixture are unchanged from the
[representative protocol](../gemma-quality/PROTOCOL.md). Context is 4,096,
KV is F16, and eight 128-row teacher-forced chunks publish all 1,024 complete
262,144-vocabulary heads. Row `j` scores input ID `j+1`; BOS is unscored and the
last head has no target. Input SHA-256 is
`b2d7aaf6aa2ef06d82591a3794f36640e192f429539ec934fd74bf4d81df1610`.
Native shared-Q8, row-invariant and cache-store policies remain off.

The graph-order-only ordinary file remains
`1c66029f1f088e04a7a4001fbdd0e24afbb97ad3b9e85f20fb73c58292e4b340`.
The compound policy selects 60 norm/RoPE, 90 norm/residual, 30 routing and
30 reduction steps per chunk. Its first/repeat complete files both have SHA
`0c8a920e565e67fb026337d62c3041f147accd1ee767a4a8a79f440812650065`.
Candidate-only finite/head/hash calibration was recorded and frozen before
stock comparison, SHA
`8fff558a279efe8c3597a4bdcb9247cb9a8f5ee9722a446613bffef0c031d510`.
Repeat variation is zero. The earlier ordinary calibration
`2c97b2a4d65277f00b242180d7f54128a395284490d84db0db24e634adcfc901`
and failed baseline are retained; ordinary-to-candidate change is not noise.

| Full-row result against production reference | Compound candidate |
| --- | ---: |
| Byte-exact heads | 8 / 1,024 |
| Strict argmax differences / outside original zero allowance | 9 / 9 |
| Maximum raw-logit delta | 1.80902195 |
| Maximum full-softmax TV | 0.09075770 |
| Mean target NLL, candidate / reference | 6.9808697463 / 6.9795193682 |
| PPL, candidate / reference | 1,075.85368 / 1,074.40185 |
| Relative PPL change | +0.135129% |

These are full-vocabulary calculations over all 1,023 targets, with actual byte
comparisons including signed zero. The original representative greedy/PPL
failure remains historical evidence; this narrower gap is also unqualified.
The fixed analyzer includes finite, signed-zero, tie, actual freeze/compare
and immutable-overwrite controls.

## Original reference routing eligibility

A controller-only read-only diagnostic prints the original structural, shape
and physical memory predicates without retaining extra tensors or changing
stock math or allocator hints. Its complete output is byte exact to stock SHA
`c07c711bfc732ce498182917c0cc3cfd74f21374975e9fa69c90ec88461cc256`.
Every routing chain passes structure and shape. Layer 28 alone fails physical
memory eligibility on all eight calls: its 4,096-byte normalized-weight output
allocation overlaps the 65,536-byte router-logit input. At 128 rows the original
one-block alias exception is unavailable. This explains the stock 29-versus-
native-30 routing count, without attributing all remaining head differences.
Native disjoint placement is retained; no layer whitelist or forced alias
reproduces this incidental reference allocation.

`gate_trace.py` requires the exact committed stock controller; the unchanged
image's math library SHA is
`5a13585ed1dc0263639e47f5154b0ea51c542df555e391a7b753daa73a56934e`.
The diagnostic controller source/library hashes are
`554c04f45ad1ebab48ccbeb48c9f47ae04dcd663ab492cda9769c0eb442f1921` /
`150fcb0eb9918a48ce6245714c1ce9c7d7c213985c1a632f28cf9f30a4a11158`.
Only the host controller is rebuilt; original CUDA math remains in the image.

## Paid short native bookends

All three arms use literal IDs `[2,818,5279,529,7001,563]`, eight initial warm
units, clear/reset, then three untimed units for build/capture/replay. The timed
prefix is `[2,818,5279,529,7001,563,45518,107,101]` at past 9. Each timer pays
32 completed scalar chunks, full-vocabulary staging/publication and CPU
argmax. Disk log writing occurs afterward. Prefixes and all 32 timed IDs match.

| Native policy | Seconds | Completed units/s |
| --- | ---: | ---: |
| Ordinary first | 0.632246 | 50.6132 |
| Compound norm/MoE | 0.612738 | 52.2246 |
| Ordinary second | 0.631533 | 50.6704 |

Compound time is 3.0308% below the mean ordinary bookend; ordinary movement is
0.1128%. This is one short native-only screen, without a fresh matched
competitive reference, comparable physical peak, 8K/depth ladder or optimized
batching gate. The quality failure keeps all new policies off by default.

## Provenance and reproduction

Measured native base is `622dfe5` plus this dispatch slice on `spark-c4e2`.
SDK is `aarch64-c09daba6ac31edee`; prepared GGML is `026f1ac94af98011`,
upstream `b29c606e28a01b1bc8c1351026a0fa6e616bf6c4`, compiler/fast-math pins
unchanged. The original-library primitive fidelity remains the
[standalone evidence](../gemma-moe-primitives/README.md).

| Measured source or executable | SHA-256 |
| --- | --- |
| Quality harness source | `a3a8603bd26dbbc9ced6169b7a7493e175bb1e5584017fe819e73fb9eb3efd0e` |
| Quality executable | `f41c9c94d99a61de467132783fc5f028aa59eb7874f01b3c9cb4c69d96cb1922` |
| Paid harness source | `9349d111438724cac1ee0e1a7040899df2c3c979f47b4140637ffef0311432d8` |
| Paid executable | `046eb812cd5a3c64922e1a656c4c5687e30dfc8f7d43bff2d323227fc4da1902` |
| Measured analyzer | `8b8e36fab244d187e3db9dc2eef44ddd45059ec2efe65944e56ecdfc4501cadd` |

The final analyzer adds bounded entry-point selftests without changing the
measured 1,024-row formulas. The final model fixture adds explicit funding for
retained diagnostic copies. The final fixture and both benchmark helpers
retain their complete node/runner/execution-owner bundle if retirement cannot
be proved. These lifetime-only changes preserve production arithmetic and the
measured timer scope. A final successful quality run compares all 1,024 heads
byte-for-byte with the measured candidate; the successful paid helper path
also passes. The historical measured identities above remain distinct from
the final checked sources.

Task-entry refresh on 2026-10-05 at 04:53:58 UTC observed latest TensorFold
`609ca419abecebdc5a059498a613680bd3aa847f`, version 0.6.5. Its pinned
[README](https://github.com/ashhart/TensorFold/blob/609ca419abecebdc5a059498a613680bd3aa847f/README.md)
and Gemma recipe still expose Gemma26 through MLX, without a CUDA comparator.
The same-format pinned llama.cpp remains the numerical oracle.

For fresh external output directories: run `jitllm_gemma_quality ARTIFACT
IDS_I32 NEW_OUTPUT_DIR 128 all 26` twice, authenticate source/build identities,
then `analyze.py freeze ROOT IDS_I32 SOURCE_IDENTITIES` before
`analyze.py compare ROOT RETAINED_QUALITY FREEZE_SHA`. Root contains
`all-first/logits.f32` and `all-repeat/logits.f32`. `reference_gates.sh` requires
the retained pinned stock-controller headers and must run under installed
GPU supervision. Paid commands are `jitllm_gemma_runner ARTIFACT NEW_OUTPUT_DIR
2,818,5279,529,7001,563 32 1 ordinary|all warm device`; summarize the three
ordered arms with `compare_paid.py OFFICIAL_LOG`.

Raw vectors, source/build identities and official logs remain outside Git at
`spark:~/.local/share/jitllm/gemma-moe-dispatch`, with receipts under
`~/.local/share/jitllm/jobs/m35-gemma-moe-dispatch-{order1,check5,native1,compare1,gates1,paid1,final1}`.
Locked Spark-native build and all 23 final focused controls passed without
skips (12 CPU, 11 GPU, including six model controls); the separate CPU-only
set passed 28/28.

The final integrated Spark-b suite passed 1,659/1,659 targets without skips
(434.42 s).
