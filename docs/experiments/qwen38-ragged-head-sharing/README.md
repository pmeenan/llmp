<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Ragged Qwen full-head sharing: short adaptive C2 screen

Allowing compatible three-plus-four full target heads to share the existing
ordinary seven-column MMF gains 1.50456% against the OFF bookend mean.
Bookend movement is 0.27109%. Retain the small positive for focused combined
HC/head output and state controls; this screen makes no default or quality claim.

| Arm | Pair wall, s | Aggregate output, tok/s | First completion, s |
| --- | ---: | ---: | ---: |
| OFF before | 19.403240229 | 26.387345307 | 19.223431276 |
| ON | 19.089723144 | 26.820713749 | 18.910128654 |
| OFF after | 19.350640320 | 26.459072751 | 19.000209367 |

The source delta removes only the equal-row condition in PrepareHeads.
Each original remains a guarded three/four-row full head; shared immutable
weight identity, type/layout/parameters, original and replacement ordinary MMF
selectors, paid concatenation, independent split views and funded placement
remain checked. HC sharing is off in both arms. Adaptive depth, existing
MXFP8/Moe pairing and every other selected operation are unchanged.

Each fresh service completes the same excluded unrelated one-token weight
prime, then the canonical two requests: 8,266 rendered prompt tokens,
zero cache, 256 generated tokens and length finish. Context is 262,144,
prefill chunk 4,096, selected drafter 47,172. Pair clocks include prefill,
queueing, generation and HTTP. All arms preserve exact public text, reasoning,
usage and finish; answer text is empty because the capped output is reasoning.
No native pair counts, generated IDs or complete-vector/state claim is implied.

All nine saved raw bodies parse exactly to their saved bodies and equal the
nested receipt rows. Their nine unique IDs each have one terminal-200 service
log. All seventeen command children exit zero, are reaped and have null errors
in PGID 3642903; all three services exit zero and are reaped.
Supervisor 3642902 completes zero; terminal free memory is 116.632 GiB
with native/model, GPU and container probes clear. B is idle.

Build 4a8a6500 binds checked OFF runtime 7e477cac and ON runtime 080c6dbf.
One wave-plan host object and one member of a disjoint copied engine archive
change, with normal 37-input link/SDK/libraries and nineteen source bindings.
Warm checked source/build/binary stay untouched. Eleven build children exit
zero and are reaped; wave compilation takes 2.49 s. No suites or ladders.

Measured on Spark B (spark-56f5), 2026-10-02, SDK e0a0c85c42806fb1,
target c4fb47a9 and drafter 8600a998. Outer receipt
e6f501a2f697a62cbd13801c50d5621bb8114a4c4f02846a033ad1c8cef9c393.
Local records: /home/pmeenan/scratch/m3-qwen-ragged-head-records/{build-r1,http-r1}.
B source/records: /home/pmeenan/scratch/m3-qwen-ragged-head-r1/.
Git source blobs: wave44f1791e, builder2428a861, HTTPb2112be5,
protocol0e3fc370, descriptive client1e3909da.

