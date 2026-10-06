<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Prepare the known prompt state inside paid prefill

One bulk call to the existing `ReserveStateThrough(0, 8192)` reduces the measured
Gemma31 state-preparation interval by 61.1245 ms / 33.45%, with all allocation
and initialization still inside the paid prefill timer. The four-arm comparison
does not establish a wall-time improvement: upfront mean is 12.85 ms / 0.115%
slower, with a 337.2 ms spread mostly in execution. This remains a benchmark
diagnostic; production continues preparing state per completed chunk.

| Milliseconds | Chunked first | Upfront first | Upfront repeat | Chunked repeat |
| --- | ---: | ---: | ---: | ---: |
| Prefill wall | 11218.6 | 11026.1 | 11363.3 | 11145.1 |
| State preparation | 190.527 | 120.195 | 123.012 | 174.929 |
| Completed execution | 10921.7 | 10798.8 | 11133.0 | 10863.1 |
| Publication and cleanup | 66.726 | 68.655 | 69.126 | 68.070 |
| Nested binding | 45.238 | 46.651 | 47.059 | 45.587 |

Mean state time falls from 182.728 to 121.6035 ms. Mean execution rises from
10892.4 to 10965.9 ms, offsetting that saving in this screen. Chunked wall
spread is 73.5 ms; upfront spread is 337.2 ms. The experiment combines grouped
closure/acquire work, VMM materialization and zero initialization. It does not
isolate a particular subcallee. Per-chunk `ReserveStateThrough` calls still
occur, but find the needed cells initialized and take resident fast paths.
No current reference comparison or sustained speed claim follows.

The helper adds only an optional trailing `state-chunked|state-upfront` argument,
defaulting to the old chunked behavior. Both modes discard the same warm rows,
clear state, reset counters and start the existing timer. Upfront then calls
the existing state API once before the unchanged chunk loop. Completed token
positions do not advance during preparation. A partial preparation failure
returns through normal teardown without a fallback over partially changed
metadata. The existing budget funds the same complete state; all four budgets
are identical. No allocation is moved outside the paid interval, and no
production allocation, provider batching or concurrency API changes.

All eight complete retained heads are finite and native-byte-exact to the prior
baseline, as are all 1,740,636,160 initialized state bytes and 32 continuation
choices per arm. Each arm retains 32 model jobs, 32 required-plan calls with
31 hits / one miss, and 31 built/cached forecasts with zero refusal. The existing
state phase includes the initial upfront call and the later resident checks;
model job counters exclude backing-service acquisition work. Nested binding
remains part of planning/publication, and CUDA event spans include CPU submission
gaps. Full phase data and exact-output identities are in [results](results.json).

Reproduce using `jitllm_gemma_prefill ARTIFACT IDS NEW_OUT 31 both 256
normmul-on state-only lookahead-on phases-on STATE_MODE`, in order chunked,
upfront, upfront, chunked. The [lookahead recipe](../gemma-prefill-lookahead/README.md)
uses six discarded warm rows, Clear, 8,192 paid rows, three untimed anchors and
32 forced units; context 16,384, F16 KV and local/global cells 1,280/16,384.
Snapshots, finite scans and hashes remain outside timers. The recipes explicitly
retain previously checked research norm chains; ordinary serving policies are
not changed by this diagnostic.

Source base is `845617f`; only `benchmarks/gemma_prefill.cc` changes. The helper
is built on Spark B (`spark-56f5`) and privately copied to Spark A (`spark-c4e2`).
Helper SHA starts `68368ed6`, SDK receipt `b2174122`; full pins and budget are in
results. Native NVCC is 13.4.92, toolkit 13.4.2, driver 580.178.04. Task-entry
TensorFold primary HEAD check at 2026-10-05 23:56:29 UTC remains `609ca419` /
0.6.5: its [pinned model table](https://github.com/ashhart/TensorFold/blob/609ca419abecebdc5a059498a613680bd3aa847f/README.md#models)
provides Gemma26 on MLX, with no matching CUDA Gemma26/31 target. Installed
supervisor build, private stage, four-arm screen and exact checker retire DONE0.
Raw logs and vectors remain outside Git under `gemma-state-preparation-raw`.

The result warrants retaining the grouped-state preparation lead, without
production adoption. The subsequent [26B transfer](../gemma26-state-preparation/README.md)
reduces paid prefill by 20.375 ms / 0.822% with exact heads and state.
Early full-prompt materialization would
need pressure, partial refusal and continuation-lifetime qualification before
selection. A fresh current/reference comparison should establish the remaining
solo gap before another small optimization. Corpus quality, C4 batching and
whole accumulated regression gates remain separate; no full suite ran here.
