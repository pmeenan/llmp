<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Gemma4 held-cohort selection

The ordinary HTTP driver selects a cohort at every completed unit. Gemma4
formerly refreshed its execution closure through the scheduler even when the
mask was unchanged inside the same held request. The existing Qwen/DeepSeek
fast path now applies: after ordinary admission, duplicate/range checks and
frozen borrowed/verify-peer exclusions, unchanged held selection returns
without refreshing. Changed selections, growth, Clear and restore retain
refreshes. Submitted GPU work still checks its actual closure is held.
No arithmetic, graph policy or public surface changes.

The focused controls and cached HTTP bookend passed on Spark B on 2026-10-07.
Both runtime arms use
base `3ce04ac`; their production sources differ only by the three-line guard.
A narrow baseline runtime build records its actual binary, receipt and complete
source inventory before the candidate source is synchronized. Relocated
binaries use the same pinned cuBLAS closure through an explicit library path;
actual resolved paths and library hashes are checked.

Two-profile GPU controls exercise repeated held selection, malformed masks,
changed-owner admission, Clear refresh, exact peer heads/continuations and
outside-request selection. Existing borrow/verify controls now positively
reselect the same cohort while retaining peer-removal refusal. Restore/spill
controls and both profiles' ordinary four-case HTTP lifecycle checks passed:
seven GPU controls and eight HTTP cases in total. Same-geometry literal
token/logprob repeats and stock top-two IDs, cached chat checkpoint replay,
SSE/stops/disconnect recovery and client-observed peer completion all passed.

[The HTTP bookend](http_cycle.py) measures old/new/new/old, two fresh runtime
processes per policy, with client concurrency one and four. Each owner requests
32 ordinary greedy chat tokens. A descriptive cold primer precedes two exact
cached warm repeats and the paid cached request; cached choices and usage must
also agree across all four arms. No cold-versus-cached equality assumption is
made. The endpoint uses context 8192, max_slots 4 and ordinary profile defaults,
without diagnostic forcing. Completion and clean runtime exit are required.
Four clients do not establish a precise backend batch shape; the separate
[current reference](../gemma-current-backend-greedy/README.md) has the actual
joined-C4 model-plan witness. This n=2 endpoint screen makes no stock-parity or
sustained-throughput claim.

| Profile | Clients | Old mean (ms) | New mean (ms) | Change |
| --- | ---: | ---: | ---: | ---: |
| Gemma26 | 1 | 661.102 | 623.009 | −5.762% |
| Gemma26 | 4 | 1308.796 | 1258.152 | −3.870% |
| Gemma31 | 1 | 3111.339 | 3068.163 | −1.388% |
| Gemma31 | 4 | 4649.640 | 4592.885 | −1.221% |

Each mean has two paid samples from fresh runtime processes. All cached warm
and paid responses and usage agree within and across policy arms; every owner
reports 32 completion tokens and 27 cached prompt tokens. All eight bookend
runtimes and both lifecycle runtimes exited cleanly. These short endpoint
measurements establish the cost of this guard on the tested requests, rather
than a sustained rate or a new model-reference comparison.

The baseline runtime SHA-256 is
`9e4d43a0d485d836c74043890bb60b6b45b32f279a5cb3f43eaae41449caa724`;
the candidate is
`4423b036932b0908d1b91e72d643e009915be03d8760151d17f39e4301281626`.
Both retain receipt
`874aaf7a5967cfbe91054e0d8fc1a0630f952e0e8eb54b831d09719f1e08ce89`.
Complete baseline/candidate source inventories and empty checksum-sync dry
runs bind the separate builds; the candidate inventory contains 2,091 files.
The successful build's binding is
`ba1b8605f3db731f7df38ef9548f240d9517c1c3eb8b60ef0d63f0024851f4a2`.
All six installed supervised jobs completed with rc 0. Raw responses, logs,
inventories and receipts stay outside Git under
`/tmp/llmp-m35-coordination/gemma4-cohort-guard-raw` and Spark B's
`~/scratch/m35-gemma4-cohort-guard/run1`.

Per-task TensorFold HEAD on 2026-10-07 is
[`041d14a94e951834470fd514ed33e65b8be1059a`](https://github.com/ashhart/TensorFold/tree/041d14a94e951834470fd514ed33e65b8be1059a).
Its README declares native Zig TensorFold 1.0.0 and lists Gemma4 as still under
qualification; the qualified CUDA GB10 model is Nemotron 3.5. Python 0.6.6 is
retained on the `python-0.6` line. No matching qualified Gemma4 CUDA/GGUF
comparator is documented, and no TensorFold timing is claimed. The earlier
reference screen retains its own ed78/v0.6.6 observation.
