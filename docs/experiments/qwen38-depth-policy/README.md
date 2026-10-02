<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Qwen fixed-three depth: short concurrent-request screen

Fixed greedy depth three improves throughput by **2.35%** against the mean
adaptive bookends in one fresh C2 screen. The adaptive bookends move **1.18%**,
and fixed three changes both reasoning trajectories. Keep the adaptive default;
this modest result does not select a broader ladder or production change.

| Policy | Pair wall, s | Aggregate output, tok/s | First completion, s |
| --- | ---: | ---: | ---: |
| Adaptive before | 19.568098 | 26.165037 | 19.353608 |
| Fixed three | 19.007182 | 26.937187 | 18.896399 |
| Adaptive after | 19.337902 | 26.476502 | 19.157214 |

Each fresh service receives the same two `spec-c4-u2/u3` prompts: 8,266
rendered tokens, zero cached tokens, 256 generated tokens and length finish
per request. An identical unrelated 57-token, cap-one weight prime is excluded.
The pair clock includes prefill, queueing, generation and HTTP; it is not a
decode-only measurement. Context is 262,144, prefill chunks are 4,096, and the
selected drafter has 47,172 vocabulary rows. These conditions differ from the
older Mia/TensorFold screens, so this table establishes no cross-engine ratio.

Only the greedy `DraftDepthFor` return changes from adaptive two/three to
three. Sampling and drafters supporting fewer than three rows retain their
depth-two fallback. Both arms use the current shared execution, independent
request state, paired products and equal-three/four full-head sharing. A copied
serving object and archive produce the candidate through the ordinary runtime
link; the checked source, build and binary remain untouched.

Adaptive after reproduces adaptive before text, reasoning, usage and finish
exactly. Fixed three first differs in reasoning at character 687 for u2 and
249 for u3. All public answer content is empty because these capped replies
are still reasoning: this is not a completed-answer quality comparison.
Changed histories can also change later acceptance and work, limiting causal
performance attribution. No new native token-ID, acceptance or depth counters
were added, and no quality bound or default changed.

Measured on Spark B (`spark-56f5`), 2026-10-02, with AArch64 SDK
`e0a0c85c42806fb1`, target artifact `c4fb47a9` and drafter `8600a998`.
Production qualification `8b029fdb` binds adaptive runtime `7e477cac`;
private build receipt `87fd7e86` binds fixed-three runtime `c5c98643`.
The ordinary 37-input link and current SDK/library bindings are retained.
Both profiles retain the funded two-request workspace and memory guard;
maximum populated-context memory and a comparator memory ratio were not
measured. Incremental build took 16.66 s, including 6.47 s compilation.

HTTP receipt `01017b6a565c31d1f7b4a43d518fb42e995e5c28db16b617fa9da0aec54a5cac`
retains all nine replies, raw bodies, terminal service logs and timings.
All three services and seventeen supervised children exit zero and are reaped;
the final admission probe reports 117.105 GiB free with model/GPU/container
probes clear. A first wrapper attempt failed before compilation because an
exclusive receipt writer was called twice; its record is retained separately.
No unit/style suite or workstation execution was needed for this diagnostic.

Raw records and source remain outside Git at
`/home/pmeenan/scratch/m3-qwen-depth-policy-records/{build-r1,build-r2,http-r1}`
and `m3-qwen-depth-policy-{build.py,http.py,http-client.py,protocol.md}`.
Source Git blobs: builder `6965b2e3`, HTTP `cee125e6`, client `1e3909da`,
private serving source `714bde90`. Spark B copies are under
`/home/pmeenan/scratch/m3-qwen-depth-policy-r1/`.
