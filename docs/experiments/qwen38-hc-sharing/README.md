<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Qwen HC BF16 sharing: short adaptive C2 screen

The private HC factor gains **1.60%** against the OFF bookend mean, with
**0.28%** bookend movement and identical capped public responses. Retain this
small positive pending focused numerical/state adoption controls; it does not
change production or the default.

| Arm | Pair wall, s | Aggregate output, tok/s | First completion, s |
| --- | ---: | ---: | ---: |
| HC OFF before | 19.409141 | 26.379322 | 19.060654 |
| HC ON | 19.076304 | 26.839581 | 18.893508 |
| HC OFF after | 19.353940 | 26.454562 | 19.176623 |

Both profiles retain adaptive depth two/three, equal-three/four head sharing,
original MXFP8/MoE pairing, independent target state and the funded two-request
workspace. Only target-verification `GemvBf16` products with 2–4 rows each,
combined at most eight, share a paid BF16 concat and output views. Immutable
weights, complete parameters, original F32/BF16 output types and ordinary
cuBLAS precision are preserved. One-row draft and wide or unsupported paths
stay original. A separate HC sequence/order preflight preserves old pairing
if HC pairing is refused. Existing final placement/registry checks enforce
the unchanged funded bound.

Each fresh service completes the same excluded unrelated cap-one weight prime,
then serves the canonical `spec-c4-u2/u3` pair: 8,266 rendered prompt tokens,
zero cache, 256 generated tokens with length finish, context 262,144,
prefill chunk 4,096 and a 47,172-row drafter. Pair clocks include fresh prefill,
queueing, generation and HTTP; they do not measure pure decode. All three arms
preserve text, reasoning, usage and finish exactly. Public answer content is
empty because these capped replies are still reasoning. Native vectors, state
and HC pair counts are not qualified by this HTTP screen.

All nine raw bodies reconcile with parsed/nested replies and one successful
terminal service record per own response ID. Three services and seventeen
command children exit zero and are reaped without errors. Supervisor 3638954
ends successfully; the terminal admission probe reports 117.130 GiB free with
model/GPU/container probes clear.

Measured on Spark B (`spark-56f5`), 2026-10-02, SDK `e0a0c85c42806fb1`,
target `c4fb47a9` and drafter `8600a998`. Build receipt `55955d86` binds the
exact checked OFF runtime `7e477cac` and private ON runtime `a3ba3f8d`.
Only `qwen38_wave_plan.cc` and its member in a copied engine archive change;
the checked source/build/runtime stay untouched. The ordinary 37-input link,
current SDK/libraries and nineteen source identities are retained.
Compilation takes 2.33 s; eleven build children exit zero and are reaped.
No unit/style suite, context ladder or workstation execution was run.

HTTP receipt:
`9227e85455df3f50430e3d7105d32d29093438c94cb2493bce3b1939886e97ed`.
Raw records/source remain outside Git at
`/home/pmeenan/scratch/m3-qwen-hc-sharing-records/{build-r1,http-r1}` and
`/home/pmeenan/scratch/m3-qwen-hc-sharing-r1/` on Spark B.
Source Git blobs: wave `efebeece`, builder `684f84d6`, HTTP `24456349`,
protocol `9f77444a`, unchanged descriptive client `1e3909da`.
