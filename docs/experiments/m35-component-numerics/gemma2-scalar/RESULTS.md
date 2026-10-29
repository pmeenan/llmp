<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Gemma2 scalar Q8 transfer: FAIL

The registered first screen **fails**: 1,006 of 1,070 teacher targets are within
0.1 nat, or 94.0187%, below the unchanged 95% bound. Both model collections are
valid. No inference was repeated, no bound changed, and no frozen expectation
was updated. This does not qualify the wider Gemma families, assistants, speed,
or U3 closeout.

Registration `78747648169b2643cdc031680b304ce5a43ab09a` fixes the
[protocol](PROTOCOL.md), arm/source/dependency pins and existing
Gemma2 quality bounds. [results.json](results.json) records the exact metrics,
launches, completed-plan counts and raw evidence hashes. The bound file's reused
historical v0/v1 display name does not describe this experiment: these are U4
A/D runtimes using the same canonical Gemma2 Q8_0 artifact.

| Metric | Observed | Registered bound | Result |
| --- | ---: | ---: | --- |
| Teacher targets | 1,070 | 1,070 | PASS |
| Absolute PPL difference | 0.206875916% | ≤0.5% | PASS |
| Mean absolute target NLL difference | 0.030601437 nats | ≤0.05 | PASS |
| Maximum absolute target NLL difference | 0.239483572 nats | ≤1.0 | PASS |
| Fraction within 0.1 nat | 94.018691589% | ≥95% | **FAIL** |

A mean NLL/PPL is 3.5635004927423646 / 35.28650134833719; D is
3.5655671149652917 / 35.35950062117154. The maximum difference occurs at
position 884, teacher ID 6659. Exactly 64 targets exceed the 0.1-nat tolerance.
The small aggregate PPL difference does not override the failed agreement rule.

Both raw responses contain the same complete 1,071-token input: one leading
BOS 2, a leading null score and 1,070 finite, nonpositive teacher scores with
exact echoed-ID/teacher-key alignment. The full ID hash, using UTF-8
`json.dumps(list)` with its default separators, is
`ee85f4ffc9be5825936ff379d873322a78c048018375805ffbb4a5cb5b1946c4`.
Both use the registered 4,000-byte passage, embedded tokenizer metadata,
F16 KV, context 8,192, prefill chunk 1 and two slots, with sequential C1 requests.

Completed score plans in both arms are stage 0 / phase 4, one row and one
segment. Five plans have 256, 256, 256, 256 and 47 successful uses, totaling
1,071. Per scalar plan, A selects 105 prepared MMVQ, 26 GGML MMVQ and 26 fused
GEGLU operations; D selects 130 VecQ dense, 26 VecQ GLU and one prepared MMVQ.
The 156 changed product selections all use Q8_0 weights. The intended transfer
is present in completed execution, rather than inferred from configuration.

Both greedy controls finish by length at 128 tokens. Their different texts are
coherent, relevant openings, but both hash-table explanations misleadingly call
the index unique, and A implies similar keys cause collisions. The truncated
explanations and stories do not establish completed task fulfillment or a
behavior-pack pass. Existing client
counters give decode rates A 76.07 / 77.21 versus D 77.05 / 78.21 tokens/s.
End-to-end times are A 2.401 / 2.156 versus D 2.462 / 2.178 seconds. These are
single fixed-prompt samples with debug-plan logging, **not a speed gate**.
In registered prompt order, A/D decode-rate ratios are 0.987243 / 0.987137;
A/D end-to-end-time ratios are 0.975224 / 0.990198. These observational ratios
use the exact retained client clocks, not the rounded values above.

## Collection and retained harness failure

Spark B ran A then D in separate installed exclusive jobs, each with a
600-second timeout and 30-second grace period. Both collectors returned zero
and both runtime processes retired through TERM with return code zero.

Supervisor `m35-u3-gemma2-scalar-a-r1` reports **FAIL**, retained unchanged:
the outer wrapper compared raw `ldd` text containing different ASLR load
addresses. Independent root inspection and the offline audit found that this
was the sole differing pre/post field. Address-normalized transcripts are equal;
resolved dependency paths and all ten library hashes, registered files,
runtime, artifact and corpus identities match exactly. Raw score, input,
completed-plan, response and retirement checks pass. A was recovered from its
existing output without repeating inference.

D's wrapper changes only the mechanical address comparison and records its
own hash. It preserves raw transcripts and compares their normalized form plus
all other identity fields. Supervisor `m35-u3-gemma2-scalar-d-r1` reports PASS.
The post-pair receipt records no active jobs or GPU processes on B.

Full responses, SSE, teacher scores, plans, runtime logs, launch/identity receipts,
supervisors and the A failure diagnosis remain outside Git at
`/tmp/llmp-u3-gemma2-quality-78747648/`. The one fixed comparison's `pair.json`
SHA-256 is `1b0f3c4958df0352a1fdde29ce343d734722819af02fb0bda06337cdfd566c82`.
The aggregate JSON binds those raw files; it preserves the numerical FAIL
separately from recovery of the outer harness failure.
