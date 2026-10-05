<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Current Gemma31 prefill phases

With state-only prefill, lookahead and checked plain norms enabled, synchronous
planning costs about 12 ms: 31 of 32 required plans are cache hits. Deferred
binding remains a concrete optimization lead. Across the 32 cache insertions,
`BindPlanned` costs about 175 ms, coverage checks 17 ms and cache accounting /
insertion 5 ms. These elapsed counters do not isolate an individual binding
callee or establish CPU busy time.

| Final first / repeat, ms | First | Repeat |
| --- | ---: | ---: |
| Enclosing prefill wall | 11257.0 | 11627.2 |
| Checks | 26.669 | 28.966 |
| Inputs, excluding state growth | 2.274 | 2.269 |
| State growth | 161.848 | 185.188 |
| Required planning | 12.516 | 12.286 |
| Staging | 2.066 | 2.150 |
| Completed execution | 10858.3 | 11203.1 |
| Publication and cleanup | 193.317 | 193.283 |
| Nested binding | 174.949 | 174.757 |
| Nested coverage | 16.995 | 17.111 |
| Nested cache accounting / insertion | 4.925 | 5.023 |

The seven outer runner phases are disjoint. Binding, coverage and insertion
are nested within the first required plan and the 31 deferred plans; do not
add them to required planning or publication. Coverage measures `CheckCoverage`
only; descriptor-vector construction and policy scans remain outside it.
Insertion includes the actual host-charge transfer and `plans.Add`, while
preceding byte/node estimates remain outside it. Publication also includes
output assignment and local cleanup.

The final pair spends 218.027 / 223.331 ms constructing future CPU plans while
current execution runs. That overlapping work is not additive. Job stream
spans are 10841.7 / 11186.1 ms and include CPU submission gaps; they are not
active-kernel time. Job wall, submission and completion waits overlap the
outer execution phase. The 370 ms enclosing-wall spread mostly appears in
execution, so this diagnostic supplies no stable latency or competitive claim.

The preceding unsplit pair remains recorded: 11323.4 / 11737.0 ms wall,
10894.5 / 11323.9 ms execution, 193.905 / 176.054 ms state and
192.342 / 192.820 ms publication. Only the later pair adds the three nested
timers. All four arms retain exact complete prefill/final heads, all
1,740,636,160 initialized state bytes and 32 choices against the prior baseline.
All eight retained heads are finite. This checks two vectors per arm, not
every decode vector or corpus quality.

The next bounded production candidate should target binding without reusing
address-bearing plans across shapes. `BindPlanned` includes scratch planning
and registry binding. Attention scratch planning queries CUDA kernel occupancy;
caching immutable, device-specific occupancy results is a source-supported
lead, not yet the measured cause of all 175 ms. Registry binding also rebuilds
an execution-plan identity from the ordered implementation names and identities
using portable SHA-256; its contribution is unmeasured. Preserve fresh descriptor,
workspace and catalog validation, then judge a change with exact outputs and
an untraced before/after. State growth remains another separate cost.

Reproduce with `jitllm_gemma_prefill ARTIFACT IDS NEW_OUT 31 both 256
normmul-on state-only lookahead-on phases-on`. Accounting is off by default;
the benchmark resets runner and node counters after the discarded warm rows,
immediately before the existing prefill timer, and reads them after that timer
ends, before anchors. Arithmetic, memory policy, heads and work are unchanged.
The inherited [lookahead recipe](../gemma-prefill-lookahead/README.md) uses six
discarded warm rows, Clear, 8,192 paid rows, three untimed anchors and 32 forced
units; context 16,384 with F16 local/global KV cells 1,280/16,384. Only one
prefill head is published. Finite scans, snapshots and hashes are outside timers.

Source base is `756ab6e`; explicit normmul-on matches the later adopted default.
Builds run on Spark B and the privately copied helpers run on Spark A. Native
NVCC is 13.4.92, toolkit 13.4.2. The task-entry primary TensorFold check at
2026-10-05 23:11:15 UTC remains `609ca419` / 0.6.5, with Gemma26 on MLX and
no matching Gemma31 CUDA recipe. [Results](results.json) retain complete source,
binary, input and state identities and both measured pairs. Both narrow builds,
copies, pairs and exact checkers retire successfully. No trace, fresh reference
or full suite ran; the owner defers full regression while performance work
continues. Raw logs, vectors and snapshots remain external.
