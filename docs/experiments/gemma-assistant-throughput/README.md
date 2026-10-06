<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Gemma31 repeated greedy assistant screen

The first bounded C1/P64 screen commits the same 32 greedy tokens in native
and pinned original target-only and target-plus-assistant paths. Each also carries
the same uncommitted next anchor at position 96. Native actual `GreedyUnit`
execution takes 1.315482 s versus 2.939108 s for native target-only: 55.24% less
elapsed time, or 2.2342× the committed-token rate. It is 1.25% slower than the
original assistant path. This is a short literal diagnostic with EOG ignored
consistently, without serving, chat-stop, 26B, sampling or sustained qualification.

| Arm | Two paid repetitions (s) | Mean (s) | Committed tokens/s | Spread (ms) |
| --- | --- | --- | --- | --- |
| Native target-only | 2.940045 / 2.938170 | 2.939108 | 10.8877 | 1.875 |
| Native target + assistant | 1.315688 / 1.315276 | 1.315482 | 24.3257 | 0.412 |
| Original target-only | 2.900468 / 2.900182 | 2.900325 | 11.0332 | 0.286 |
| Original target + assistant | 1.300446 / 1.298140 | 1.299293 | 24.6288 | 2.306 |

Both assistant trajectories complete 12 units, verify/publish 46 target rows
and draft 34 assistant rows. Accepted-prefix counts are four units keeping one,
two keeping two, and six keeping four: 32 committed tokens, including 20 accepted
drafts. The final unit verifies two rows and keeps both; no scalar tail occurs
in this particular trajectory. The implementation caps depth at
`min(3, remaining−1)` and uses ordinary Wave1 when one token remains. Target-only
uses 32 completed scalar waves. Extra speculative work stays inside the clock.

The actual engine unit releases every frozen cache borrow before target Verify,
settles rejection/acceptance before publication, and carries head/feature at
keep−1. There is no second native transaction implementation in the paid loop.
All arms use the same canonical query64 prefix, C1/context4096, max rows/ubatch128,
ordinary F16 local1280/global4096 caches and explicit31 target norm/RoPE plus
norm/add chains. Target-only does not admit the assistant or retain unused full
features. Assistant depth is at most three, head/verify cap four. EOG is ignored
for exactly 32 requested committed tokens; EOS 106 happens zero times here.

Each producer executes one untimed warm trajectory, two untimed quality repeats
and two minimal timed repeats. Setup, prefix, reset, allocation/reserve,
diagnostic archival trace copies, endpoint state hashing, full archive scans
and file writes are outside elapsed time. Paid work includes actual target,
drafts, snapshot/verify/rejection/acceptance, job retirement, normal workspace
full-head copies, finite judges and result carry. Native full scans/copies inside
`GreedyUnit` remain real paid work. This measures a resident warm suffix, with
no startup or prefill speed claim and no diagnostic-copy subtraction.

Independent own-freezes prove all retained trace values finite, full quality
repeats byte-exact, and timed token/head/feature/state witnesses equal to their
own quality trajectory. Native freezes precede all original acquisition.
Predeclared continuation gates pass between scalar and assistant within each
engine and between engines: all32 committed IDs and pending anchors agree.
Cross-engine pending heads are byte-exact for each matching mode. Scalar versus
assistant pending heads differ by up to 3.2960844 while choosing the same next
anchor; no scalar/query-four byte equality is assumed. This comparison does not
claim complete cross-engine head/feature equality for every intermediate row.
Only the available last assistant draft head is archived per native unit; its
hidden earlier recurrence heads are not claimed as a complete trace. There is
no corpus/conditional-loss result or new numerical allowance in this screen.
The [earlier plain-norm transaction failure](../gemma-assistant-greedy-reference/README.md)
remains a failure; this screen uses the separately qualified31 norm chains.

Caller vector envelopes are charged before allocation: native 64 MiB plain and
256 MiB assistant, with 1 MiB metadata headroom and checked actual capacities
34,603,008 / 211,548,160 bytes. Temporary vocabulary admission is separately
512 MiB. Native post-clock full semantic state witnesses use catalog-funded
pinned buffers. The original known-vector bounds are 192 / 384 MiB, including
an explicit 128 MiB endpoint opaque snapshot cap plus 1 MiB metadata. Its actual
endpoint96 snapshot is 86,511,304 bytes; legacy P64 caps remain unchanged.
Original allocated vector capacities are 34,603,008 / 210,499,584 bytes.
Original logical-cell witnesses and native semantic initialized-state hashes
are checked independently; no cross-engine KV-byte or unchanged peak-memory
claim follows. Owners survive until retirement; this screen injects no new
completion failures.

The narrow build passes three legacy and three new analyzer controls plus
pre-context26 refusal. All ten official jobs complete successfully, including
four producers, four own-freezes and the comparison; the build is the tenth.
The original build and two producer containers retire with authenticated absence.
Raw IDs, full row payloads, caches and logs remain external. No full suite,
additional arms, new calibration or new original revision was acquired.
[Aggregate results](results.json) contain source, binary, SDK, own, completion
and container identities. TensorFold was re-pinned at entry to cb2ebf05/version
0.6.6; its retained MLX recipe supplies no comparable CUDA Q8 assistant result.

Reproduce using manual `jitllm_gemma_greedy_reference` with mode `plain32` or
`unit32`, or the pinned [original wrapper](../gemma-assistant-greedy-reference/performance_reference.sh)
with the same mode. The [frozen method](../gemma-assistant-greedy-reference/PERFORMANCE.md)
sets paid boundaries and funding. Use
[`performance_analyze.py`](../gemma-assistant-greedy-reference/performance_analyze.py)
`own` after each successful official producer retirement, then `compare` with
all four fixed proof hashes; authenticate inputs and retired markers again.
Preserve original exposure after native own-freeze and exclusive output paths.
Current engine/serving defaults and model support remain unchanged. Other
prefixes, terminal behavior, context transitions and approved26 transfer need
separate evidence before assistant serving or broad performance adoption.
