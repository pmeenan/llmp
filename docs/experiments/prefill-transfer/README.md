<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Shared funded prefill lookahead

## Gemma3 and Gemma4 lifecycle extraction (2026-10-07)

`engine/prefill_lookahead.h` centralizes the optional host allowance and
CPU-built future plan previously managed separately by Gemma3 and Gemma4.
Their prediction, output and capture policies remain in the adapters. This
slice makes no performance claim and changes no production defaults.

The helper takes an optional allowance before future plan allocation.
It builds only CPU descriptors during the current device job; implementation
binding, coverage checks and cache insertion remain after the driver's existing
successful-completion checks. A binding or coverage refusal destroys the future
plan while its temporary allowance remains charged. Successful insertion gives
back that allowance immediately before the cache's normal required charge.
Abandoning the prediction destroys its plan before returning the allowance.
Existing `PlanStep` scopes protect the current plan and capture throughout.
No future state backing or cursor is prepared by the helper.

Five host-only tests exercise optional refusal, failed and oversized builders,
abandoned predictions and failed current units, installation failure, and
single transfer to the normal cache while the current plan/capture remain
protected. Model controls compare exact state and heads for unhinted, hinted,
wrongly hinted and warm Gemma3 runs; they also check Gemma31 pressure refusal
and abandoned predictions, and Gemma26/Gemma31 state-only continuation/replay.

On Spark B, installed GPU-exclusive job `prefill-shared-check1` incrementally
built the three affected test targets against source base `0ea0a18` plus this
slice, using the pinned `aarch64-c09daba6ac31edee` SDK and CUDA 13.4 cuBLAS
130800. All **17 focused tests pass**, with positive XML counts and no skips:
13 host-only plan/cache tests (including the five new lifecycle tests), one
Gemma3 hinted-prefill test, and three Gemma26/Gemma31 controls listed above.
The job's final state is `done`, exit 0, after 95.04 seconds including the build.

The Gemma3 artifact was copied to B in supervised job `prefill-shared-fixture1`.
Its manifest hashes to approved identity `8c7103418a6608022e5eda50a0dcc4b7688a0d59ef239813c9de0984161397fb`;
every declared payload's length and SHA-256 were independently checked after
the copy. Gemma26/Gemma31 used their existing approved fixture artifacts.
A separate source review and adversarial lifecycle challenge found no defects.
No new timing comparison was needed for this exact policy-neutral extraction;
exact state, heads, capture counters and memory accounting are the controls.
The full regression suite remains deferred under the owner's optimization
check override. Further transfers, including Gemma2 lookahead/capture and joined
prefill hints, remain separate work; this extraction does not close those ports.


## Gemma2 capture and joined Gemma hints (2026-10-07)

Gemma2 now uses the shared lifecycle with two-ahead shape prediction and graph
capture selected in ordinary serving. Internal false overrides retain matched
controls. Gemma2 and Gemma3 production adapters forward joined prompt hints
from `PreparedPrefill`, calculated after current history/state funding by the
same `PromptSession::NextPrefillHint` used by scalar advancement. Scoring and
checkpoint boundaries retain their original suppression rules; scoring stays
on its scalar path. Each future
stage filters ended owners independently and requires compatible head intent;
a mixed next-head stage can still predict a valid after stage. Row descriptors
remain available for its absolute position. No speculative future state is
funded, allocated or published.

Gemma4 retains its current next-only capture policy and one-owner production
prefill capacity. Forwarding hints through generic scalar fallback does not
admit joined Gemma26/Gemma31 prompts: per-owner chunk sizing, a funded total-row
envelope and capture/state/failure controls remain open. DeepSeek and both Qwen
formats remain open lookahead consumers. These are explicit inventory/plan
items, separate from the four completed Gemma2/Gemma3 transfer cells.

### Native first traversal and warm control

Spark B used the same binary, approved Gemma2 artifact
`eb18d30d0a7de3a95c7b6994b65a12a057ffbf42866add6f128873de8b7aa870`,
actual tokenizer inputs from the earlier mask control, per-owner 128 chunks,
total-wave 256, bounded roots and GPU masks. Prefixes are 4352/4864. Input SHA-256:
`102c7b555b1caed5aed3d9880a173aae153f8f8dc1534e6a4c58685c243b6c0c` and
`a930726bd964ae88ef0448f50a51d2e376ce2487313f26063ab84c1b0b41d77f`.
Both fresh-process comparisons use off/on/on/off order, two samples per arm.

| Control | Off seconds, in order | On seconds, in order | Mean change |
| --- | --- | --- | --- |
| Weight-warm, plan-cold first prefill | 1.486540 / 1.485690 | 1.407230 / 1.408910 | −5.252% |
| Corrected warm prefill | 1.304370 / 1.304560 | 1.308830 / 1.315860 | +0.604% |

For first traversal, three supplied rows per owner load weights off clock;
both completed waves precede logical Clear and DropPlans, also off clock.
The paid prefill includes complete prefixes, all remaining state growth,
required planning, optional planning, capture and execution. Three supplied
rows per owner after prefill remain off clock. Decode pays 32 natural joined
steps, including the final full-head copy. All 64 generated choices and both
complete final heads match bytes across every arm. Enabled paid prefill builds
and caches 18 future plans, captures four current shapes on their first run and
14 ahead graphs, and replays 31 chunks; disabled paid prefill has zero replays.
Corrected warm prefill builds no future plans in either arm. The first-traversal
gain supports adoption; the short warm result provides no warm speed benefit.
Neither comparison establishes sustained performance or new stock parity.

The boolean prediction contract describes full heads or state-only work.
This benchmark's cycle final frontier instead requests a GPU token, so future
final-head prediction stages are suppressed with nullopt while their row
positions remain available. Ordinary serving publishes a full-logit prompt
frontier and uses the boolean contract directly. An earlier warm screen
(+0.783%) included legal but unused speculative full-head plans; the corrected
warm control above supersedes that policy evidence. Its separate one-pair
full-head/state diagnostic was exact, including checkpoint, spill/restore and
continuation files; its single timing sample is not an adoption speed claim.

The corrected cold and warm screens used source base `2fe9c65` plus the
candidate, source-manifest SHA-256
`021c6d45d09d4eb10755a6dfe72cadae80e6b9b64cef5307788f864bd130d390`
and probe binary SHA-256
`c6fb88254b93bd670dc06f064a67214f4e92984e4c365933a23423703f6ca111`.
Installed GPU-exclusive jobs `gemma2-lookahead-cold-screen1` and
`gemma2-lookahead-warm-control2` complete with exit 0 in 16.65 and 19.06 seconds,
respectively, including the cold screen's incremental probe build.

### Production adapters and focused correctness

A new GPU fixture constructs the real Server and production Llm adapters, then
admits actual PromptSessions and dispatches RunPromptWave. Each family uses
fresh-server off/on/on/off controls, valid fixture-token prefixes 1280/1536,
per-owner 128 chunks and total-wave 256. A three-row warmup, Clear, DropPlans and
scalar reuse precede the paid prompt; remaining state growth, planning,
capture, actual adapter work and final full-head publication stay paid.
Reversed incoming ownership, mixed future head modes and departing owners
exercise joined and scalar fallback paths. Every arm has nine actual joined
prefill groups. Enabled arms build/cache six future plans, capture one first
shape and five ahead graphs, with no dropped/refused prediction. Disabled
arms build/capture no future plans. Complete finite heads of 256000/262208
values and initialized state hashes match exactly across all four arms.

| Actual adapter control | Off seconds, in order | On seconds, in order | Mean change |
| --- | --- | --- | --- |
| Gemma2 full-head prompt | 0.433557 / 0.426443 | 0.413130 / 0.412373 | −4.011% |
| Gemma3 full-head prompt | 0.535721 / 0.531607 | 0.504107 / 0.513575 | −4.651% |

These are separate short representative production-adapter effects, not HTTP
latency, sustained qualification or reference comparisons. They must not be
substituted for the longer Gemma2 native first-traversal result above.

Focused qualification has **41 unique positive tests**, zero skips: 27 host
prompt/prefill/hint tests; six Gemma2 checkpoint/lifetime/kept-restart and new
mixed-stage/refusal controls; six Gemma3 serving/state/capture controls; and
two parameterized real-adapter tests. A suppressed-next/valid-after test crosses
padded read width 256→512, requires the correctly positioned after plan to be
built, captured and replayed, and compares exact final heads/state. Optional
pressure refusal and abandoned predictions preserve the completed prefix,
never overcharge host memory, and release the temporary allowance.

The first qualification job stopped on the new pressure test: its charge
filled FreeBytes but left unused startup host-floor allowance, so optional
planning correctly still fit. The corrected fixture uses one shared HostFloor
calculation, fills free budget plus unused host-floor allowance, and proves
FreeBytes is zero before prediction. This was a test setup correction; no
production source changed. Its 27 host and five successful Gemma2 checks remain
evidence. The follow-up reruns only the failed pressure test and previously
unrun six Gemma3/two adapter checks; all nine pass. Source/default review and
adversarial challenge are clean.

Both qualification jobs are installed GPU-exclusive jobs with 600-second
limits; the successful follow-up `prefill-transfer-qualification2` completes
in 46.94 seconds. The final source-manifest SHA-256 is
`7ed2f989cddb9c70d8a9ddd9322f768ba903e36669c6a9dcea036ca435b0e994`;
the linked runtime SHA-256 is
`bb530ad86919598507b6eed32e85ce7c7f2e8b604b5c61800edf21a2457d46c6`.
SDK/CUDA/cuBLAS pins remain those of the lifecycle extraction above. Raw logs,
XML, binary/source bindings and comparison records stay outside Git under
`~/.local/share/jitllm/` on B and the session coordination scratchpad. The full
regression suite and workstation tiers remain deferred under the owner's
override. The previous matched warm llama.cpp result remains a 6.721% prefill
gap; this first-traversal transfer does not close it. A current matched native/
reference prefill timeline is the next attribution step.

At task entry, TensorFold native main was
[`f8fe17d24629aedabf90bbf78279dd776e6d62e7`](https://github.com/ashhart/TensorFold/blob/f8fe17d24629aedabf90bbf78279dd776e6d62e7/README.md)
(1.0.0), and python-0.6 was
[`ed78d6fc204d89d90b045bf033d6551e7714f3a1`](https://github.com/ashhart/TensorFold/blob/ed78d6fc204d89d90b045bf033d6551e7714f3a1/README.md)
(0.6.6). Neither lists a Gemma2/Gemma3 CUDA recipe for these approved GGUF
artifacts; the Python Gemma4 recipe is 26B/MLX. llama.cpp remains the applicable
CUDA reference. The previous fresh Gemma2 reference pin is
`d81235049384534c167caea52b85a694f6103d14`; no new reference run is claimed here.
