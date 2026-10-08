<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
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
`~/.local/share/llmp/` on B and the session coordination scratchpad. The full
regression suite and workstation tiers remain deferred under the owner's
override. This first-traversal transfer left the earlier 6.721% warm prefill
gap unresolved. The subsequent [matched timeline and actual-root transfer](../gemma-prefill-copies/README.md)
attributed it to joined K/V packing and removed those copies. Fresh final
Gemma2 bookends are level within the short n=2 screen, with exact choices and
final heads; broader shape/context and sustained qualification remain open.

At task entry, TensorFold native main was
[`f8fe17d24629aedabf90bbf78279dd776e6d62e7`](https://github.com/ashhart/TensorFold/blob/f8fe17d24629aedabf90bbf78279dd776e6d62e7/README.md)
(1.0.0), and python-0.6 was
[`ed78d6fc204d89d90b045bf033d6551e7714f3a1`](https://github.com/ashhart/TensorFold/blob/ed78d6fc204d89d90b045bf033d6551e7714f3a1/README.md)
(0.6.6). Neither lists a Gemma2/Gemma3 CUDA recipe for these approved GGUF
artifacts; the Python Gemma4 recipe is 26B/MLX. llama.cpp remains the applicable
CUDA reference. The previous fresh Gemma2 reference pin is
`d81235049384534c167caea52b85a694f6103d14`; no new reference run is claimed here.

## Two distinct future shapes (2026-10-08)

Gemma2 and Gemma3 now select capacity two, composing the existing optional
lifecycle in a fixed `PrefillLookaheadGroup<Planned, 2>`. Their ordinary
policy builds both eligible next and after-next missing shapes while the current device unit runs. This
lets a cold 256-row traversal find the next graph ready to capture, instead
of finding only its CPU plan. Gemma4 retains its existing single-future
policy; no new Gemma26/31 performance result is claimed.

Before any optional graph funding, each runner predicts complete shape keys,
excludes the current shape and duplicate futures, and protects every cached
future through the current `PlanStep`. Each missing prediction gets its own
existing host allowance before allocation. CPU builds run sequentially in one
meanwhile callback. Ready plans install near-to-far only after successful
current completion; refusal of the second grant, build or insertion leaves
an independently successful first plan installable. The first cache insertion
stays protected during the second charge. Current failure installs neither
plan and destroys both before their grants are released. Predictions never
initialize future state or advance a cursor. No public context, batch or
chunk admission limit changes.

The representative factors use one process per arm in the fixed order
capacity 1 / 2 / 2 / 1, with the same ELF, pinned SDK and actual first-resolved
cuBLAS libraries. The `first-cycle` harness loads weights using three rows
per owner, then completes Clear and drops all plans/graphs before paid
prefill. Remaining state growth, CPU planning, binding, graph capture, device
work and head publication remain paid. Decode is separate. Complete final
initialized-state hashes, all 64 natural greedy choices and both finite full
vocabulary heads match byte for byte across every arm of each factor.
Snapshots use a common 8 MiB funded pinned buffer and run outside paid
endpoints. Each process and installed supervisor positively retires; no new
NVIDIA/kernel error appears.

| Cold configured-256 C2 workload | Capacity 1 prefill seconds, bookends | Capacity 2 prefill seconds | Mean change | Capacity 1 bookend drift |
| --- | --- | --- | --- | --- |
| Gemma2, prefixes 4352/4864, context 8192 | 1.177670 / 1.174090 | 1.169700 / 1.169250 | −0.545% (6.405 ms) | −0.304% |
| Gemma3, prefixes 1280/1536, context 4096 | 0.473840 / 0.468759 | 0.466613 / 0.469646 | −0.673% (3.170 ms) | −1.072% |
| Gemma3, prefixes 3072/3584, same context 4096 | 1.060920 / 1.059260 | 1.058260 / 1.056470 | −0.257% (2.725 ms) | −0.156% |

Gemma3's short comparison overlaps and is neutral at this sample size. The
longer compatible context is measured directly; it does not require a wider
public envelope. Gemma2 and longer Gemma3 have disjoint short ranges, but
these remain modest n=2 effects, not sustained throughput or reference-parity
claims. Prefill-plus-decode mean changes are −0.431%, −0.154% and −0.156%;
separate decode changes of −0.202%, +0.354% and +0.060% are within noise.
The factors are separate and are not pooled.

Actual selection explains the narrow gain. Gemma2 builds/caches 16 future
plans in either arm; capacity 2 installs one pair and captures/replays 15
prefill graphs ahead, against one at capacity 1. Short Gemma3 builds/caches
three futures and captures/replays two against zero. Longer Gemma3
builds/caches eleven and captures/replays ten against one. Every capacity-2
arm executes exactly one paired build/cache; capacity-1 arms execute none.
No refused or dropped future is reported. The configured-256 screens use
state max_rows 512 and the already checked stock-matched local capacities
4608/1536, total wave rows 512; those are internal runner controls.

The final focused qualification has **20 unique positive checks**, no errors,
skips or disabled tests: nine shared lifecycle/group controls; two Gemma2
refusal/departure controls; two Gemma3 wrong-hint/warm/departure controls;
four real production-adapter and configured larger-root cases; and three
unchanged Gemma26/31 policy/state controls. The larger-root cases execute
three 256-row C2 waves, require a paired construction/cache and ahead
capture/replay, compare complete packed/root heads and initialized states,
then protect the peer through checkpoint restore and kept restart at 768
positions with exact continuation at 769. Every intermediate cursor reflects
only current completed work. Existing malformed/duplicate adoption refusals
remain checked. No full regression suite runs for this slice.

The candidate was merged onto `d468286ef2bfa0b9d3a04c8b293b64004d0e37d3`
without changing the measured group/runner implementation. Current shared
F16/F32 mask authentication retains the same Gemma F16 arithmetic and
descriptors; the merged serving fixture preserves earlier direct-256 and
optional-node state-read controls. The only production policy change after
cold measurements is the default capacity of two; every factor explicitly
sets capacity one or two. The final focused source manifest is
`72ef05e0411468317948999f7e6983956fd1524ffdd36f0c3a34e602e368df0e`,
checked aggregate
`05e436b5fc28c443f673f6690759f0002234fcc86d8116238e8805fdcf6ac159`.
The merged G2/G3 benchmark ELF identities are
`c107fd86c51835046f7860821ce102eeac1f94495cc123e70e778a863fcc3f00`
and `f71500abb583afc2f921b34d13000de387c4ebb7ad049fbdbf7537048d92b5e1`.

The ordinary 128 retained-plan warm control uses the same approved short
prefixes, context 8192/4096, state max_rows 128 and total-wave 256, with local
capacities 4352/1280 and no stock-ring override. `cycle` executes a complete
warm traversal and eight joined decode steps, then Clear retains nonempty
backing/plans/graphs before the paid traversal. Capacity 1/2/2/1 uses one
newly built ELF per family. No paid future build/cache/pair, optional
first/ahead capture or eager path occurs; ordinary second-visit captures stay
paid. All 64 choices, complete finite heads and initialized states remain exact.

| Ordinary 128 retained-plan control | Capacity 1 prefill bookends | Capacity 2 prefill | Mean change | Old drift |
| --- | --- | --- | --- | --- |
| Gemma2 | 1.206860 / 1.209570 | 1.206250 / 1.203400 | −0.281% | +0.225% |
| Gemma3 | 0.462252 / 0.461005 | 0.455457 / 0.460459 | −0.795% | −0.270% |

Gemma2's paid prefill-plus-decode changes −0.177%; decode +0.039% is noise.
Every arm performs three ordinary paid captures and 36 replays covering all 39
prefill units, with 66 total warm+paid joined groups and 16896 joined rows.
Gemma3's paid prefill-plus-decode changes −0.293%; its sub-millisecond mean
decode difference is +0.198%, with no decode gain claim. Every arm performs
three ordinary paid captures and ten replays covering all 13 prefill units,
with 18 total warm+paid joined groups and 4608 joined rows.
These short controls establish no meaningful warm regression, not a
sustained warm speedup. The initial G2 attempt stopped after one successful,
retired native arm because the controller incorrectly required zero ordinary
paid captures. `PlanRuns::CaptureDue` deliberately captures on a retained
plan's second eager visit; that normal behavior remains paid and recorded in
the corrected control. The first attempt is preserved separately and is not
pooled; there is no added warmup or production source change.

Warm checked aggregates are
`ba51484bdb28c9e4056da273b051e6211e8d49ce8f93efcd46a4012238939b70`
and `2fc95dac3fe80fdab27ee0d79f3142e35f4a7163c132768ec7f3f77f66094d87`.
To reproduce the warm control, use the same fixed
capacity 1/2/2/1 order and approved short IDs, change mode to `cycle` and chunk
to `chunk=128`, and omit `stock-ring`. Require actual retained state/plans/graphs,
88 GPU-token publications, positive replay, complete logical-work accounting,
exact outputs/state and positive retirement. Ordinary second-visit captures
must be reported rather than falsely excluded from the paid interval.


Separate decode samples remain part of the paid sum:

| Workload | Capacity 1 decode seconds, bookends | Capacity 2 decode seconds | Decode mean change | Paid prefill + decode mean change |
| --- | --- | --- | --- | --- |
| G2 cold 256-row | 0.587177 / 0.584801 | 0.584217 / 0.585389 | -0.202% | -0.431% |
| G3 short cold 256-row | 0.480195 / 0.480950 | 0.482409 / 0.482139 | +0.354% | -0.154% |
| G3 long cold 256-row | 0.494101 / 0.492866 | 0.492826 / 0.494735 | +0.060% | -0.156% |
| G2 warm 128-row | 0.580737 / 0.580422 | 0.580592 / 0.581019 | +0.039% | -0.177% |
| G3 warm 128-row | 0.471463 / 0.471974 | 0.472784 / 0.472519 | +0.198% | -0.293% |

Adding each decode sample to the corresponding prefill sample above
reconstructs all four paid sums; no intermediate publication or snapshot
moves across the stated endpoints.

Reproduction uses the retained native benchmark rather than disposable
controllers. Approved short IDs and their source-text/native-tokenizer
preparation are described in [the configured-prefill report](../gemma-prefill-copies/README.md).
For each family, use fresh output directories and invoke
`llmp_gemma{2,3}_joint_prefill_probe ARTIFACT IDS0 IDS1 OUT first-cycle
bounded-roots device-masks prefill-ahead owner-prefill flexible-owner-prefill
chunk=256 stock-ring lookahead-capacity=N`, in the fixed N=1/2/2/1 order.
Check the actual summary, full output bytes, initialized-state hashes and
retirement marker, not just the requested capacity. Final positions are
4387/4899 for Gemma2 and 1315/1571 for short Gemma3, including three supplied
untimed seed rows and 32 measured continuations per owner.

For the longer Gemma3 input, retrieve `docs/engine.md` and
`docs/async-model.md` from commit `46cfd5c7b7e9e8731a169001c9e112c3d261f767`.
Take the first 20000 bytes of engine.md, removing only an incomplete trailing
UTF-8 character; keep all async-model.md bytes. Their SHA-256 values are
`5f420b26bc12eb5f77bb1ff599b2351935b6ef7f73374908f82aa19ba5e09c5a`
and `cd7b0613fd963dd35d1f81a4abdfd867484d9747b8ae01c572507ab33bf6094b`.
Use the existing Gemma3 benchmark `prepare METADATA TEXT 3111 INPUT0_DIR`
and `prepare METADATA TEXT 3623 INPUT1_DIR` commands with the approved artifact's
`meta/gemma-3-4b-it-qat-Q4_0.kv.gguf` (6514895 bytes, SHA-256
`1b703d735f6f344c5910aa45f49b132db45b332bfbe675dcc0ef29d8db8f1266`).
Each new input directory contains `ids.i32`. Actual ID hashes must be
`e550a181088bc3d4924de3d45091067a38cd75c7c9ee676947e8880334a83c7c`
and `c8e853512376e0148c7c5798974bd91e42849c918f8f23017fb3e22eef23cfee`.
The same first-cycle invocation then pays prefixes 3072/3584 and finishes at
3107/3619. Required replay inputs survive raw-sample cleanup or can be
recreated from these frozen Git texts and the native tokenizer.

The G2 factor source manifest is
`379c04c4f1f7376a9cef66a4a6cec10cea10583f6d2523112cfbd084a1473d08`,
ELF `029f63b30b285a91c86ab56ace713f830a141541d1664d060d4f3447a38d75ae`;
its checked aggregate is
`6f060982fc76c4c35fcdcd97fd0249071b7940ce831ce3a094c96923077d5d83`.
The G3 factors' source manifest is
`235bfbc5f2161af8a713476c8692ddae7d6bd02a0535d3a6db21c4bb363cc156`,
ELF `4b78985c6a5e073d1e510c00fccd61022fadb85263005023c96639d85c53cb9a`;
short/long checked aggregates are
`c45852f24c3a3d8898615955d44dbb1b83ea9335733e688b43d0f0231f1a50a7`
and `9ad65c116602c708938a0d3bbf3857d8ed959372ad1fd9da7b0f3474ece7881a`.
G3's parser/cold-mode omissions were caught before inference, corrected and
rebuilt: `first-cycle` explicitly seeds three rows, completes Clear/DropPlans
and requires zero retained plans/graphs/cursors. No warmed run is mislabeled
as cold. All factors use official `aarch64-c09daba6ac31edee`, first-resolved
cuBLAS SHA `ee7c1657a03695c0de790aa79e34cef9c9649756b1846b11dd44caca20ba656b`
and cuBLASLt SHA
`ba3b942f4ea43433b65e8c492a7b73de887534dc20146506ddaa4a78c79c5d30`.

The fresh task-entry TensorFold native/Python pins and applicability remain
those recorded above: neither provides an applicable approved G2/G3 CUDA/GGUF
recipe. The latest applicable recorded llama.cpp comparisons remain separate;
this transfer makes no new stock-parity claim. Raw process logs, XML, kernel
observations and four-arm payloads stay outside Git until milestone cleanup.

Final docs were fast-forwarded to `61003ccf15a5089081551fca6be93d50bea2c57c`
with all 32 qualified source-file bytes unchanged. G2/G3 select capacity two;
G26 keeps one future until its open capture-ahead transfer is qualified, and
G31 keeps its previously neutral capture-ahead disposition with no new claim.
DeepSeek and native/GGUF Qwen remain open consumers of the compatible funded
future-group protocol; image phases have no token-chunk prefill loop.
Configured rows above 256, mixed-width roots, other cohorts and public
larger-chunk admission remain T93, explicitly open. No full regression or
workstation shipment tier was run under the owner's current deferral.
