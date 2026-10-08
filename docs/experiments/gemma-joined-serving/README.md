<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Bounded joined Gemma serving controls

The shared runtime adapter can explicitly join independent Gemma26/31 decode
owners through the existing native runner. Both internal options remain false
by default. Native same-policy solo/joined full heads, histories and initialized
state agree in the bounded controls; matched stock quality fails on the natural
prefix screen. This implements a diagnostic path without selecting optimized
batching or qualifying either model.

## Completed-unit ownership

Each owner prepares its own one-token state unit. The hook validates every
tuple before dispatch, then runs ordered groups of at most eight owners; C12
is eight plus four. This is the existing row-invariant product bound, distinct
from the 128 routed-pair bound. A successful group commits its independent
native cursors and complete heads before the outer driver samples and publishes
each owner's result. Clean group refusal marks only that group; completed and
usable peers retain their results. A whole-cohort fault returns before callbacks.
Independent preparation refusal, departure, callback stop and cancellation
retain the existing session machinery.

The internal `gemma_joined` and `gemma_row_invariant` options add no CLI or
configuration schema. The ordinary scalar route and uncalibrated production
caps of 128 prefill rows/twelve owners remain unchanged. Norm chains, shared
Q8, MoE routing/reduction and RoPE/store selection remain off. Native device
masks remain on. [Transfer audit](transfer-audit.md) records the existing
Qwen/DeepSeek mechanisms reused and the unavailable or unselected contracts.

Serving continues to fund two full-vocabulary F32 rows and fixed two-vocabulary
sampling candidates per owner, with separately funded native 128-row pinned
output and selected-profile state/plan/workspace/staging. Model fixture copies
have explicit host grants; snapshots use catalog pinned memory. The paid helper
charges `(32+2)*C*262144*4` bytes for retained complete heads and working rows:
142,606,336 bytes at C4 and 427,819,008 at C12. These are capacity charges,
not measured physical peaks. No memory qualification follows from them.

## Fixed-input paid screens

Every arm discards eight warm units, resets and prefills its identical private
prefix, then runs three untimed units before 32 paid waves. All `32*C` complete
heads and greedy decisions are copied/published inside the timer; disk output
is outside. Native C12 pays 64 GPU groups; stock C12 pays 32 genuine
multi-sequence batches. Both have effective context 256 per owner. The reference
screens physical ubatches C/32/128 using the original pinned math library,
F16 KV, stock fusion and graphs. These fixed-input rates are not generated-token
throughput. [Protocol](PROTOCOL.md) defines IDs, positions and selected targets.

The first synthetic C4 screen uses a six-token prompt plus owner padding.
Same-policy scalar bookends and joined repeats are byte exact across all 128
heads, frozen before oracle access. Ordinary arithmetic differs and is retained
as a separate control. No allowance was widened to accommodate that movement.
The row policy applies to every chunk with at most eight rows, so synthetic
six/seven/eight-token prefills also select it (the nine-token owner does not).
This ordinary-versus-rows comparison does not isolate decode arithmetic.
Natural 64+owner prefills disable the row policy in every arm.

| Profile | Scalar row-policy bookends (s) | Joined first/repeat (s) | Reference screened first/repeat range (s) | Strict reference differences /128 |
| --- | --- | --- | --- | --- |
| 26B-A4B | 2.536 / 2.530 | 0.924 / 0.924 | 0.855–1.067 | 1–2 |
| 31B | 12.490 / 12.498 | 3.417 / 3.429 | 3.341–3.349 | 1 |

The broader reference timing movement is retained; there is no synthetic
winner or quality gate. [Synthetic aggregate](results-synthetic.json) contains
the actual timings, counts, complete-file identities and selected-row summaries.

The second axis uses the independently authenticated War and Peace token
control: owner `i` has its own 64+i token prefix, then successive supplied
anchors. All ordinary, row-policy scalar bookends and joined repeats are
byte exact per native axis. Each own freeze completes before its matched oracle
job. The stock comparator uses distinct sequence IDs and batch-index head
mapping, including all twelve owners in one real batch.

| Profile/cohort | Scalar row-policy bookends (s) | Joined first/repeat (s) | Reference screened range (s) | Strict reference differences | Maximum selected-row TV |
| --- | --- | --- | --- | --- | --- |
| 26B C4 | 2.534 / 2.531 | 1.025 / 1.024 | 1.025–1.047 | 66–77 /128 | 0.998909 |
| 26B C12 | 7.710 / 7.652 | 2.615 / 2.607 | 1.554–1.562 | 199–205 /384 | 0.998841 |
| 31B C4 | 12.453 / 12.463 | 3.414 / 3.416 | 3.351–3.367 | 32–37 /128 | 0.903782 |
| 31B C12 | 37.701 / 37.496 | 8.014 / 8.019 | 4.438–4.461 | 96–99 /384 | 0.981119 |

This is a material reference quality failure. Joining preserves the measured
native math, but that math is insufficient for selection. Every reference arm
repeats exactly; all listed strict differences also have positive reference
winner margin outside the frozen zero own-repeat movement. Exact reference
ties are counted separately. Selected likelihood/TV summaries cover eight
explicit rows per comparison, not corpus PPL. Reference physical ubatches also
change prefill math, so these results do not isolate a norm or compiler cause.
The C12 8+4 performance gap remains open. [Natural aggregate](results-natural.json)
retains each axis and ubatch rather than selecting a favorable comparator.

## State, publication and actual HTTP controls

The model fixture covers both profiles at C1/2/4/8/12 with sparse slots,
unequal pasts, complete own heads, greedy histories, initialized state and fresh
joined repeats. Captured replay and stable addresses are exercised. A C12
callback stop and independent cancellation leave other owners progressing,
with sampling storage released. Exact target scoring uses the preceding
owned frontier, verifies partial callback completion and unchanged peer state.
Actual independent state-growth refusal preserves completed prefixes and peer
progress. Separate CPU controls directly exercise the production 8+4 helper's
late-group refusal, early-group refusal, whole-cohort fault and invalid envelope.

Two-owner branch snapshots restore complete heads and initialized state.
After explicit spill and server restart, each saved history is restored with
the existing resume contract: the previously published pending anchor is
processed once, and callbacks publish only the three new tokens. The resulting
joined continuation matches each owner's same-state scalar continuation.
Stable lifetime bundles and sticky retirement failure retain all borrowed
configuration, node, catalog and state owners if completion is unproven.

The dedicated diagnostic HTTP binary forwards both default-false switches into
the ordinary service implementation. Actual startup status confirms
`joined-diagnostic`, row-invariant and optional optimization status; final
counters confirm 26 completed shared groups/52 owned units per profile.
Eight actual HTTP cases passed with runtime exit zero: offered concurrent
C1/2/4/8/12 versus each profile's same-policy solo, chat/SSE/stops, tool refusals,
disconnect/follow-up, 127 aligned likelihood rows with first-null convention,
and pending 26→31/31→26 switches. Each direction records its own positive
source pause delta of one under `model_turn_seconds=1`, with exact owned SSE
output. The offered HTTP C12 case does not prove a fixed twelve-owner native
cohort; that is established by the dedicated model driver control. Thinking,
generated tools and assistants remain unavailable. [HTTP aggregate](results-http.json)
records status, completion, case counts and per-direction pauses without output
token vectors.

## Provenance and limits

Measurement ancestry is c2d2147. The final tree additively incorporates 9d14b40;
its larger-prefill benchmark is not consumed by this measurement. The two
measured source manifests, native/reference executable identities and task-entry
upstream pin are retained in [provenance](provenance.json). Synthetic and natural
archives remain immutable externally; subsequent changes add HTTP diagnostic
propagation/counters and correct the restart fixture, without changing paid
helper arithmetic. Approved prepared artifacts are 26B `4ddb360c…` and dense31
`32c92e07…`; raw source revisions/formats and full hashes are in the protocol.

On Spark-b, the official first control passed two CTest targets (both profiles
per target); two further controls passed in a three-target run whose restart
fixture failed. The corrected restart-only target passed both profiles in
31.37 s after the final locked build. All four CPU group controls and the actual
HTTP run passed. Historical screen1/native3 build/startup failures and controls5–8
fixture/compile failures are preserved externally: default-branch snapshot
assumption, no-frontier resume misuse, full-prefix recomputation and a wrong
accessor. They are not reported as passing runs. All paid, freeze, reference
and comparison jobs completed through the installed supervisor with exit zero.

Native/reference quality, selected norm/product policies at joined shapes,
representative PPL, physical peak, long context, full swap and optimized-batching
qualification remain open. No default or supported status is selected. Raw
heads, IDs, token vectors, per-step likelihoods and runtime logs remain external.

The final integrated Spark-b locked build passed; all 1,668 tests passed in
592.01 s with no skipped tests. Final light checks passed REUSE/headers on 1,383 files,
portability on 384 sources and changed-source format/diff; author script syntax
and signed-zero/tie/non-finite controls passed. Raw logs remain external.
