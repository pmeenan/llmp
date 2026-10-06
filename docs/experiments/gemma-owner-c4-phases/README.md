<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Current Gemma31 C4 phase accounting

Two owner-root/plain-norm-on runs spend a mean **78.73 ms outside runner
execution** within the 3.321385 s paid interval. All 128 complete heads, four
initialized states and 128 choices are byte-exact to the
[prior C4 factor](../gemma-owner-root-c4/README.md). This diagnostic adds only
optional accounting to that manual helper; it changes no operation or policy.

| Paid component | First (ms) | Repeat (ms) |
| --- | ---: | ---: |
| Complete paid interval | 3315.960 | 3326.810 |
| Runner checks | 24.745 | 24.737 |
| Inputs | 2.415 | 2.388 |
| State preparation | 0.343 | 0.336 |
| Required planning | 0.028 | 0.029 |
| Staging | 0.124 | 0.152 |
| Execution | 3238.250 | 3247.070 |
| Publication / cleanup | 6.902 | 8.114 |
| Paid interval minus all runner phases | 43.153 | 43.985 |
| Device event stream span, overlapping execution | 3236.560 | 3244.650 |

Both runs make 32 planning calls, all hits, with no binding, coverage or cache
insertion time. Their mean outside-execution interval is 2.37% of paid time;
checks account for 24.74 ms and caller/unclassified time outside all runner
phases for 43.57 ms. Paid time minus the stream span averages 80.78 ms. These
quantities are of similar scale to the earlier 68.43 ms native/original gap,
but **there is no fresh original arm or causal attribution here**. The stream
span includes possible host submission gaps and is not active GPU time.
The seven runner phases are disjoint; nested binding/coverage/cache counters
and Node step timings must not be added to them. The residual includes caller
publication and argmax plus unclassified overhead.

The unchanged closed recipe uses Gemma31, four unequal prefixes of 64–67 tokens,
three anchors, context 256 / max rows 128 and 32 one-query C4 waves. Independent
K/V roots retain Q/mask packing, ordinary products, 120 normRoPE / 120 normADD
and 121 plain norm fusions; all other optional policy counts are zero. Each
paid interval replays 32 graphs and captures none. Complete head publication
and argmax are paid; full finite admission and state/output file writes occur
after the clock. Each state witness has 120 initialized ranges and 230,686,720
bytes, ending at positions 99–102. All heads are finite, with the existing
`9de4f80b…` full-array hash. Both own repeats and Task62 heads/states/choices
are exact. No original/native state serialization comparison is made.

Accounting and Node step counters reset immediately before the paid 32 waves;
snapshots follow the paid clock. Reproduce with the inherited
`jitllm_gemma_owner_c4 ARTIFACT NEW_DIR 31 4 joined norm IDS_I32`, setting
`JITLLM_GEMMA_OWNER_C4=owners`, `JITLLM_GEMMA_C4_NORMMUL=1` and
`JITLLM_GEMMA_C4_PHASES=1` before construction. Phases default off. Canonical
IDs are 4096 bytes with SHA `b2d7aaf6…`. Use installed Spark supervision,
private output paths and the matching SDK cuBLAS library path.

Source is based on `3b47791`; only `benchmarks/gemma_owner_c4.cc` changes
(SHA `0e9bba8b…`). Private helper `58433a0d…` uses locked SDK receipt
`fbf84c74…`, NVCC 13.4.92 / CUDA toolkit 13.4.2 / Clang 22.1.8. Spark A is
`spark-c4e2` / GB10; driver 580.178.04 is inherited from the previously checked
environment, not newly probed. TensorFold was refreshed at entry to
`609ca419…` / 0.6.5, with MLX26 only and no matching Gemma31 CUDA recipe.
Target-only build (six steps), two native arms and the exact checker all
retired officially with rc 0; A then had no jobs, waiters or compute processes.
[Results and identities](results.json) retain the concise measurements;
raw logs and output arrays remain external. No new stock run, trace or full
suite was required under the owner’s diagnostic override. Production selection,
Gemma26 transfer and wider-context qualification remain open.
