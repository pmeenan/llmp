<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Four Qwen requests: two versus four active slots

Rejected on spark-b, 2026-10-02. Both profiles use selected HC/ragged
sharing, unchanged adaptive depth and the 47,172-token draft vocabulary.
Context is 33,792, prefill chunks 4,096. Four concurrent canonical chats
each render 8,266 tokens, with zero cached tokens and 256 length-capped
completion tokens. The unrelated one-token weight prime is excluded.

| Arm | C4 wall, s | Aggregate tokens/s | Completion latencies u0/u1/u2/u3, s |
| --- | ---: | ---: | --- |
| Two slots before | 36.586368078 | 27.988566611 | 22.048931 / 36.585647 / 36.347421 / 18.609434 |
| Four slots | 42.475213933 | 24.108177574 | 42.475050 / 42.306112 / 41.609741 / 42.143860 |
| Two slots after | 37.188018477 | 27.535750544 | 37.187725 / 20.664015 / 36.610180 / 18.853296 |

Mean two-slot duration divided by four-slot duration gives **−13.155956%**
throughput. Bookend movement is **+1.644466%**. Four active slots delay
the first completion and lower aggregate rate in this condition. Keep
two funded slots; no additional capacity matrix is needed for this decision.
Pairwise product sharing remains pairwise with four slots.

| Requested budget | Two slots, bytes | Four slots, bytes |
| --- | ---: | ---: |
| Fixed catalog | 4,176,973,612 | 7,683,411,756 |
| Activation | 3,049,259,008 | 6,096,420,864 |
| Scratch | 442,499,072 | 884,998,144 |
| Host input | 18,874,368 | 35,651,584 |

Registered state virtual extent remains 5,762,973,696 bytes, largest
weight extent 76,900,466,688 and margin 6,442,450,944. These are guarded
budgets, not measured peaks. All 12 measured replies preserve capped
text, reasoning, usage and finish fields exactly; all 15 raw/nested/parsed
responses match and have unique matching HTTP 200 terminals. This does
not establish completed-answer quality or native token-ID equality.

The private build reuses the qualified 37-input closure, replacing only
wave/runner in a copied engine archive; four slots alone replaces one
serving object. The checked warm tree stays untouched. All 17 HTTP
commands and all three services finish rc0 and are reaped with no cleanup
errors; final memory/process probes are clear at 116.385 GiB.

Provenance: build `a36cf41d`, HTTP `25856109`, supervisor `3717319`.
Raw records: `/home/pmeenan/scratch/m3-qwen-combined-capacity-records/{build-r1,http-r2}`.
The first controller filename shadowed Python's HTTP package and refused
before any model load. That attempt is retained; the unchanged controller
was restaged under an unambiguous filename.
