<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Gemma assistant: bounded engine greedy transaction

An explicit C1 `Gemma4Assistant::GreedyUnit` now drafts, verifies and accepts a
bounded target-authoritative greedy prefix. Both approved target/Q8 assistant
pairs pass eleven focused controls: five CPU judge cases and six real GPU cases.
This is engine transaction evidence. Serving, scalar-width quality, sampled
acceptance, batching and performance remain unqualified; defaults are unchanged.

The caller holds the request and supplies a completed target frontier at prefix
P. Its greedy token is the authoritative anchor. One to three assistant steps
borrow target cache/POST-finalnorm feature P−1, keep every query at P and use
separate recurrence. Every round starts from that target feature; all frozen
borrows end before target verification writes. Verify consumes the anchor plus
drafts in at most four rows. Draft j is judged against target head row j. With m
matches, keep=m+1 commits the anchor and matched drafts, while head row keep−1
predicts a separate, **uncommitted** next anchor. Acceptance restores rejected KV
and commits the selected feature before publishing cursor, tokens or full outputs.

The existing immutable `max_verify_rows` defaults to zero. Enabling the unit
requires explicit retained features and head capacity covering verification;
no serving head cap expands. Caller-funded/reserved workspace and result vectors
are checked before borrow/dispatch. At depth 3 the workspace/result payload bound is 6V+5W F32
values: 6,398,976 bytes for 31B or 6,347,776 for 26B, plus actual capacity and metadata.
The supplied completed frontier is separately caller-owned/funded.
History/proposals use fixed four-token arrays. The fixture charges the established
512 MiB bounded vocabulary-admission envelope before parsing authenticated metadata,
then 64 MiB for caller outputs/ranges. State/feature copies use separately cataloged
pinned allocations. Unknown completion ownership remains inherited from Step and
the target verifier; no new post-write fault was injected. Uncertain owners stay
retained until a proven fence. Completed cancellation uses Verify/Discard phases.

The GPU controls use real target+assistant artifacts for both profiles, context 1536,
max rows 16, explicit head cap/verify cap 4 and six-row prefixes. A second independent
slot supplies a same-options **four-query normal target Wave** oracle; assistant
calls remain C1. All complete four-row heads/features compare exactly. Semantic
accepted KV ranges, rejected-tail bytes, actual selected features and projected
peer continuations match; a following real unit resets recurrence from the new
target feature. Whole-discard restores requested reserved initialized state and
old features. Capacity/cursor/depth/nonfinite refusal leaves result and peer state
unchanged and dispatches no assistant step. CPU controls cover keep1..4, lowest-ID
ties, canonical/shape refusals and nonfinite unused/final rows. Actual proposals
need not exercise every acceptance length; the independent
[target verifier controls](../gemma-target-verify/README.md) remain separate.
No scalar-versus-four-query byte oracle or numerical allowance is used.

The first official attempt compiled and passed CPU5/GPU4, but two cases expected
target replay after only two chronology rounds. A shape runs eagerly first and
captures second. The sole fixture correction adds the third unchanged round,
retaining every byte assertion and actual replay proof. The failed attempt is
preserved. The final nine-step job retired DONE0 in 38.184 s with 11 PASS/0 skip and
successful source guards. Only the two new targets ran; no full suite, benchmark,
public-engine acquisition or numerical calibration. Private executables and SDK
receipt were frozen after success. Raw logs and official records remain external; tests hash cataloged state copies
without storing KV in Git.

Source8 is based on 3140a9a; final source 2 changes only the fixture loop. Parent
docs were seeded from abe31db. The SDK receipt reports `0.1.0-dev+unknown`;
source/frame/checkout identities authenticate the compiled unit independently. Reproduce by building `gemma4_greedy_test` and
`gemma4_greedy_gpu_test` with the pinned SDK, then run both under the installed
Spark GPU supervisor with explicit qualified cuBLAS library path. The former
must report 5 PASS, the latter 6 PASS and neither may skip. Use the four approved
prepared target/assistant artifact IDs in [results.json](results.json). TensorFold
was rechecked at cb2ebf0540f42604e2759b2ddef497861e928248 /0.6.6; its MLX-only
Gemma26 recipe supplies no comparable CUDA Q8 assistant/31 result here.
