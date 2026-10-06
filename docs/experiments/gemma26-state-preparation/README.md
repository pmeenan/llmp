<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Gemma26 paid grouped state preparation

Preparing the same initialized KV footprint once inside paid prefill reduced
mean elapsed time by **20.375 ms (0.822%)** in this fixed Gemma26 ABBA.
All four arms retained byte-exact heads, initialized state and 32 greedy choices.
This is a diagnostic transfer of [the Gemma31 preparation screen](../gemma-state-preparation/README.md),
with no production default change or fresh reference comparison.

| Arm | Preparation | Prefill, s | State phase, ms | Execution, s |
| --- | --- | ---: | ---: | ---: |
| A1 | Chunked | 2.48265 | 62.7400 | 2.38874 |
| B1 | Upfront | 2.45789 | 43.2540 | 2.38349 |
| B2 | Upfront | 2.45717 | 42.2985 | 2.38362 |
| A2 | Chunked | 2.47316 | 59.5016 | 2.38248 |

Chunked/upfront means were 2.477905/2.457530 s. Their respective two-arm
spreads were 9.49/0.72 ms. The mean state interval decreased by 18.34455 ms;
mean execution decreased by 2.055 ms. This short factor supports a small
paid state-preparation saving, without establishing stable competitive parity.
The earlier matched reference gap in [Task59](../gemma-current-reference/README.md)
is historical context, not a reference bookend for these four arms.

The unchanged helper used C1, context 16384, F16 local/global capacities
2048/16384, 8192 supplied rows in eight 1024-row chunks, six discarded warm rows,
Clear, three anchors and 32 incoming-head greedy/forced steps. It selected
`all`, `normmul-on`, `state-only`, `lookahead-on` and `phases-on`.
Counts remained 121 plain norm, 60 norm/RoPE, 90 norm/ADD and 30 routing/reduction;
other optional product/store flags stayed off. Each arm recorded eight
required-plan calls, seven hits and one miss. Task63's routing keep factor was
not applied and no equality with stock's 8K routing policy is inferred.

Upfront `ReserveStateThrough(0,8192)` runs after Clear and counter reset,
inside the existing prefill timer, under the same budget and held request.
The normal materialization, initialization, closure/lease replacement,
chunk loop, heads and teardown remain unchanged. All four budget fields matched.
Bind/coverage/cache are nested subintervals, not additive phase costs;
CUDA stream elapsed includes submission gaps and is not active-kernel time.

Both complete retained heads were finite and equal to the saved baseline,
as were each 592445440-byte initialized state, layout and 32 choices/forced pairs.
These controls cover two full vectors per arm, not all 32 decode vectors or
corpus quality. The four model arms and fixed checker officially retired DONE0
on `spark-56f5`; Spark B was free afterward. No new build, trace, stock run or
full regression suite was run. The owner defers full suites during optimization.

For reproduction, reuse the retained Task58 helper SHA68368ed6 and SDK receipt
b2174122 listed fully in [results.json](results.json), together with the fixed
artifact/index and 8227 IDs. Invoke its existing prefill CLI with `26 all 1024
normmul-on state-only lookahead-on phases-on`, alternating trailing
`state-chunked`, `state-upfront`, `state-upfront`, `state-chunked` into new directories.
Run under installed `spark-job start --gpu --timeout 600 --stop-on-fail`; verify the
saved head/state/input identities outside paid intervals. The helper comes from
845617f plus Task58's benchmark change, not latest main. Native NVCC is 13.4.92
(toolkit 13.4.2). Fresh TensorFold primary609ca419/version0.6.5 has no CUDA Gemma
comparator, so no TensorFold run was made. Raw records stay external.
