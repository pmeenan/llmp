<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Exact source-use indexing

Fusion gates now reuse the call-local graph index for exact source-pointer edge
counts. Each duplicate edge, repeated graph node and producer-self edge counts
separately. Views remain distinct pointers even when they share logical storage;
keeps and output flags do not affect this count. Local subgraph/suffix checks,
null queries and standalone or mismatched-graph callers retain the original scan.
Both planning passes construct fresh indices and retain SamePlan and all existing
eligibility, alias, view, output and keep checks. No selected policy or kernel
arithmetic changes. One count field per indexed descriptor fits the existing
conservative startup scratch allowance; nothing persists between passes.

Each untraced screen uses one physical host for original, unchanged native,
candidate first/repeat and original. Kernel math and selected policies remain
unchanged, including plain norm fusion off.

| Profile / host / policy | Before prefill (s) | Candidate first/repeat (s) | Reference bookends (s) | Less prefill vs before | Remaining prefill gap |
| --- | ---: | --- | --- | ---: | ---: |
| 26 / Spark-b / all 1024 | 2.79087 | 2.71533 / 2.71006 | 2.41486 / 2.41293 | 2.8011% | 12.3783% |
| 31 / Spark / both 256 | 12.4061 | 12.1004 / 12.1393 | 10.8444 / 10.9154 | 2.3073% | 11.3967% |

Decode is 0.3994% / 0.4123% slower than the fresh unchanged 26/31 controls;
remaining reference gaps are 0.5530% / 2.3237%. One fresh before arm and two
candidate arms do not establish a decode gain, stable small effects or parity.

Both candidate repeats and the unchanged controls preserve both complete retained
heads, all initialized native state (592,445,440 / 1,740,636,160 B) and all 32
choices exactly against their prior native baselines. Reference bookends repeat
their heads and choices; all 32 cross-engine choices agree. Both native 26 heads
still differ from reference. At 31 the final head is byte-exact and prefill retains
the known maximum raw difference 0.41361475. This is a two-head control, not an
all-32-vector or complete quality gate.

The unchanged [26](../gemma26-swa-ring-transfer/README.md) and
[31](../gemma-swa-ring-h1/README.md) ring recipes use context 16,384, F16 KV,
local 2,048/1,280 cells, global 16,384, row cap/ubatch 1,024/256, explicit
`swa_full=false` and `kv_unified=false`. Both engines discard six warm rows,
clear, pay 8,192 prefill rows, append three untimed anchors and complete 32 forced
rows. Native publishes 7/31 additional prefill heads. Timers, publication, CPU
argmax, graph replay, cache/mask and state policies remain unchanged. Selected
norm/RoPE, norm/residual, routing and reduction counts are 60/90/30/30 at 26 and
120/120/0/0 at 31; other optional math flags stay off. Counts describe plan
implementations, not CUDA launches.

The shared planner also passes 86 focused exact-count, fusion, norm, graph,
placed-plan, plan-cache, DeepSeek and Qwen controls. Existing primitive plans
without reader queries allocate no index. Raw logs, vectors, states and official
job records remain external; [results](results.json) retains aggregates and pins.
Full model qualification and optimized batching remain open.

The final Spark-b suite passed 1,700/1,700 tests without skips (599.17 s),
including 311 GPU and 57 model tests. Local REUSE/header checks cover 1,589 files
and boundary checks cover 394; changed formatting, diff and JSON checks pass.
