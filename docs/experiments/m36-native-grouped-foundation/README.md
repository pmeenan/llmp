<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Native grouped-operation foundation

N4b6a provides checked CPU worklist inversion and multipart descriptor geometry.
The isolated helper build and all eight focused host controls pass. Main also
passes the full native build and the same eight controls, with zero failures or
skips. Independent whole-source challenge is clean. That helper slice ran no GPU or model.

## Contracts

`GroupedRouteBound` computes exact input, output and scratch storage before the
caller allocates it. `ValidateGroupedRouteInto` requires all five complete route
arrays, zero-based monotonic offsets covering every pair, a complete permutation,
and matching token/expert entries. It preserves repeated expert contributions and
returns original routed slots for output scatter. Token inputs and per-route
inputs use distinct row bases. Experts up to 4,096 are representable, subject to
the checked route-domain limits; a 512-expert control prevents a fixture-sized
cap from excluding that geometry. Refusal may change scratch, never retained
outputs. Inputs must remain immutable during the synchronous call.

`GroupedBindingBound` sizes one descriptor per expert. `BindGroupedInto` uses the
format's complete part roles, checked extents and actual per-part matrix pitches.
It preserves the original encoding parameters and stacked scale layout. Consumed
intervals may share unused slab padding but cannot overlap readable bytes, even
across experts. Every descriptor is initialized, including empty expert intervals;
validation finishes before descriptor writes. No payload is read or decoded.
Successful helper paths allocate nothing; callers supply storage and retain source
metadata. These geometry checks confer neither source identity nor residency.

## Focused evidence — 2026-10-12

One installed shared Spark-A job built the two helper targets and completed four
`GroupedWorklistTest` and four `GroupedBindingTest` cases. The controls cover
repeated contributions, both input bases, malformed routes, exact bounds and
aliases, 512 experts, padded multipart pitches, stacked2 scales, interleaved slab
views, cross-expert overlap, missing/duplicate/misaligned parts and atomic refusal.
Source and recipe guards passed before and after the job. No broader suite ran.

- Result receipt SHA-256: `02896dd7bb0ff931e067445b770f862fea1d593c16a294d35da684958eda4840`.
- Worklist binary SHA-256: `6d11d5a8c1fb5858ba72c723100d0ee9fa47a5dc319462b62ea3585186ee3189`.
- Binding binary SHA-256: `95adffb81d65f0ea6fc12dbb3918b16d81c0aad9f172859051e938760019230f`.
- Whole source01+02 review SHA-256: `8411d6d70d35e0feb00f968b3798f1505bf7922ca0b3aa022c1f99e831ab45c2`.

Raw results remain outside Git in
`native-grouped-foundation-focused01/results` under the M3.6 coordination scratch
root. The implementation has no kernel arithmetic, catalog mutation, device
submission or model admission change.

## Main integration — 2026-10-12

Root carried the twelve reviewed paths onto `922d4f63`. One installed shared
Spark-A job completed the full native build and all eight affected host cases
in 45 seconds. Source and recipe guards passed before and after execution.
No broader host suite or GPU check ran: these allocation-free helpers submit
no device work and are not yet called by the production engine.

- Result receipt SHA-256: `d7f96bf9e668186e5d65f65ad61dd45601b7d76f7bd03b9a18f34579bec76b52`.
- Worklist binary SHA-256: `74dfec1e0a660f3303501dc12f919ab69a4470a25e28ec6320c398067e6dadf0`.
- Binding binary SHA-256: `238d12c41dd830562ea0c2c5452c8ce38d2e6d99ee67bda75d3c9728eea7a46d`.
- Main carry review SHA-256: `eb1cb7150bb0e28d83b01284757a1be3c5cbd8ed2963fd415fcfcda98934bc3f`.

Raw results remain outside Git in `native-grouped-foundation-main01/results`
under the M3.6 coordination scratch root.

## Owned grouped execution — 2026-10-12

The test-only `PreparedGrouped` consumes an ordinary tiny v1 artifact with four
NVFP4-split experts, padded part pitches and stacked2 tensor scales. It runs the
existing `MoeRoute`, waits for complete route data, validates and inverts it, then
materializes only the selected expert closure. Each product job holds the real
scheduler lease and rechecks current content/backing generations, the prepared
epoch, workspace and launch identity. No production kernel, selection policy,
ModelRunner interface or family admission changes in this slice.

Two actual Spark-A GPU cases cover token-row and original-routed-slot inputs.
Each case checks all 288 output values against an independent representable
oracle across three routing/activation revisions, including repeated experts.
Each records three completed route jobs, three product jobs, one capture and two
launches of that retained capture. The first closure excludes an unused expert;
invalid IDs, stale tokens and a stale local backing receipt refuse before new
product effects. Both cases passed with zero failures or skips; all source,
recipe and binary guards passed, no GPU process remained and all memory-event
counters were zero. This is operation correctness, not a model quality or speed
measurement and not a cross-library bit-equality gate.

The fixed test budget is a 256 MiB managed node plus a prepaid 16 MiB host metadata
allowance (8 MiB charged for this inventory), 4 KiB catalog-accounted pinned
staging and bounded source/file-cache storage. A 10 GiB process/cache cgroup cap
and 6 GiB available physical margin surround the case; the cgroup is not a total
CUDA occupancy limit. Complete source, provider, graph, staging, workspace and
weights ownership is retained on unknown retirement. No owner overlaps another
case unless the first completed teardown successfully.

- Isolated compile/list-only receipt SHA-256: `9fa0a1cb4bf4bb47613c3ebf0baf446d7c6953f1d1da077b53e78544caf75cc8`.
- Actual two-case receipt SHA-256: `2bd9349894a15e621577a9310b28491b85bb0e2e07ffad8c7cf7c7801fce0d59`.
- Executed binary SHA-256: `724b50da6af6995b6ced557fe70f41b81c5540d02a1fec6c2eb858c1ab3b60f1`.
- Whole source/funding review SHA-256: `c699d98518581dbb14bbf4ac75ae8256eaef645868be195bdfbe721412e68524`.
- GPU recipe review SHA-256: `abc5bc7ed9c273da47d13cb4bd9791e179b3c73a18d95b0eb781b37a16aac903`.

The actual observed process/cache peak was 211,628,032 bytes. Raw XML, log and
receipt remain outside Git in `native-grouped-prepared-gpu01/results` under the
M3.6 coordination scratch root. No broad suite or model inference ran.

## Owned operation main integration — 2026-10-12

Root carried the exact two reviewed source paths onto `d8aa6388`. Spark-A
passes the full native build and all eight helper host controls in ten seconds.
The separate installed exclusive job passes both actual GPU cases in one second,
with all 576 output values checked and the same complete route, selected-only
residency and changed-routing capture witnesses. Source, recipe and binary
guards pass, no GPU process remains and all memory-event counters are zero.
The observed process/cache peak is 212,557,824 bytes.

- Full native build and eight host controls receipt SHA-256: `85bf2992650f676784e77fa15ced383d2ac71b80754482330c738f967095eca4`.
- Actual two-case main receipt SHA-256: `40057a00ffff10f2f013431693caede5187fd76cfc0d73be6019e4ea5c373b19`.
- Main binary SHA-256: `7f6d8d60c681f3bce64f4e665f60e8d715e485eed4ccab34da844d5257d1204e`.
- Main GPU recipe review SHA-256: `046ea632b7bfe6f119e174b0d6d4a36cf23714b49d87613cbbbfe61a99d9a1f0`.

Raw evidence remains outside Git in `native-grouped-prepared-main01/results`
and `native-grouped-prepared-main-gpu01/results` under the coordination root.
No additional source payloads or full-model checks ran.

## Supplemental selected closure — 2026-10-12

N4b6c adds `PagedNode::SupplementalJob` over the ordinary scheduler. The
original request lease stays held while selected extents materialize and receive
their own lease. Whole-closure validation runs before file opens or page-in;
ordinary `Job` and `Step` retain their subset-only contract. This prerequisite
uses current scheduler capacity and adds no victim selection or reclaim policy.

The admitted owner retains immutable closure/ID copies, the complete caller
payload, metadata charge and request external counter. A `gone` observation alone
is insufficient: offered work must positively retire its own task and every
selected page-in before releasing ownership. Failed but retired work cancels the
request. Unknown completion retains the complete owner and both lease domains,
preventing continuation and teardown. Payload retention conveys lifetime, never
source or residency authority. At most one supplemental operation is pending per
node. Retained control funding is 4096 + 256 bytes per selected extent, with a
separate 4096-byte wrapper allowance; caller payload/job funding remains separate.

Six fake-provider cases exercise the actual scheduler path: selected-only
materialization while the base lease stays held; whole malformed-closure refusal;
insufficient funding and counter overflow; cancellation with a queued fence;
`gone` preceding withdrawn page-in retirement; and unknown completion retaining
both leases and the complete payload. No arithmetic or numerical policy changes.

Main's full native build passes on Spark A. All 145 affected host controls pass:
HeldLease 21, Scheduler 29, LiveStateFake 36, ModelRunner 54 and WeightFiles 5.
The first build exposed an incomplete-type destructor instantiation from the
inline `PagedNode` constructor; moving that same constructor out of line fixed
it and received a narrow independent review. The subsequent build and first
140 controls passed; WeightFiles initially refused setup because the direct
launcher omitted its existing artifact-corpus environment. A separately reviewed
completion receipt authenticates those unchanged results and runs only the
missing five with the correct environment. No GPU or model inference ran.

- Original build/140-control receipt SHA-256: `4c30b4d3856ffec41f15d2d16e7e7faed3e2445b6b47ea325f689107f3d154a3` (literal status FAIL from missing fixture environment).
- Five-control completion receipt SHA-256: `b9da62ef5af933410a940ed619dc5cbd63bae86ed2bc2cc290e25e64d76fd2d9` (PASS; unchanged source and binary guards).
- Whole-source independent challenge SHA-256: `40e2ca7a7b7ea4fd40a2702017f1446b91e969edb94fe995f6c511f51f26c303`.
- Constructor correction review SHA-256: `c3c8d9df2d1584679b1bcb0f58515298af073a7792aa97ba4d436bb046b3e756`.
- Narrow completion recipe review SHA-256: `b5f94b144bb8416207b14b0e5bdc72407f080535897661a57a98e30d539c222a`.

Raw failed and successful evidence stays outside Git in the coordination root's
`native-supplemental-closure-main01`, `main02` and `main03` packages.

## Remaining scope

N4b6b's bounded owned operation controls, N4b6c's supplemental lease
prerequisite, N4b6d's one-group shared routed step and N4b6e's ordered-group
extension are complete. Native ModelOpt target admission
and full-model qualification remain open. Existing operation evidence makes no
speed or model-quality claim.

## Shared routed step — 2026-10-12

N4b6d connects one actual `ComponentStepSource` step to a completed routing
prefix and selected-expert continuation. The bounded fixture includes embedding,
shared dense attention with F16 KV state, MoE and a head. Direct NVFP4-split
`MatMulId` operands use the existing grouped decoder; stacked gate/up scales,
per-part padded pitches, repeated contributions and original routed slots remain
explicit. No kernel arithmetic or family numerical policy changes.

Lowering keeps the original whole-plan placement and checks every reader of the
routed weights. The runner binds complete read declarations to the effective
backend identity, source cache keys and actual opened artifact bindings before
excluding pages. Ordinary readers and shared pages stay in the base closure.
After prefix completion, all five route arrays are validated; a scheduler-minted
selected closure protects descriptor upload and continuation under the original
request lease. Every capture replay repeats that validation/admission with a
fresh epoch. Ordinary whole-step Run, Record and Queue cannot bypass the segment
contract. Unknown completion retains the entire flight and blocks continuation
and plan eviction; successful explicit teardown is required to release it.

The fixed fixture admits 384 MiB host metadata before construction, including a
256 MiB compiler inventory (derived upper bound 206,595,584 bytes), 64 MiB owner
allowance and up to 32 MiB overlapping plans/graphs. The managed node is 512 MiB;
program activations and pool each cap at 32 MiB, route staging at 16 KiB and state
layout at 128 KiB (two 32 KiB F16 banks with D64 attention). Five weight pages plus a slab read slot use at most 12 MiB +
8 KiB within the existing 16 MiB weight/read allowance. Actual confinement uses a
10 GiB process/cache cap, zero swap and 6 GiB additional physical margin, not a
total CUDA-device allocation guarantee. The first teardown failure retains its
owner and prevents another fixture domain; repeated teardown is not proof.

Full native main build and 135 host controls pass: 101 routed/lowering/owner
controls plus 34 fusion/validation controls. All three GPU cases pass without
failures or skips. The real step executes eager/captured/replayed revisions with
repeated and changed expert IDs, checking 288 complete output values against
independent scalar arithmetic, selected-page residency, one retained program,
three completed steps and state frontier 9. The other cases check pre-admission
missing-owner/misbound-source refusals and immutable selected-source declarations.
Six exact dynamic part descriptors defer residency; ordinary weights retain
their eager coverage checks. F16 attention-state writes execute, but the zero
attention-output projection means there is no independent full attention-state
byte oracle. The existing absolute 2e-5 plus relative 2e-3 tolerance permits the
decoded BF16 MMA arithmetic; it is neither a cross-library bit gate nor a corpus
quality threshold.

The final corrections preserve RoPE/cache-store fusion. Elided RoPE metadata
borrows its actual read operand, while an externally retained rotation remains
materialized. The matcher accepts only the verified current destination-cache
view edge, retaining all real operand-disjointness checks. Lowering authenticates
the complete selected-only reader inventory before omitting exact multipart
leaves from eager coverage. Catalog ownership is checked for each actual expert
part slice; padding and unselected experts do not become resident requirements.
Selected pages still require fresh scheduler admission before continuation.

Main evidence (2026-10-12), with raw files retained outside Git:

- `native-grouped-runner-main12/result.json`: full native build and all 135 host
  cases PASS; SHA-256 `b8167587a4bc55f2ecba908e59fde804b8f247d5938f92c477d10e778875256a`.
- `native-grouped-runner-gpu05/result.json`: all three cases PASS, unchanged
  pre/post source and binary guards, zero remaining compute processes and zero
  memory-event counters; SHA-256
  `9e1651e53fc480b2df49a004537ec1077adbcce41484803c3e8c5f897b7ae947`.
- Shared source-map SHA-256:
  `d6103609fc5af377f96acf6b24f6a89befe33abd2dc8145dd2ddbdd52caee060`.
  Executed binary: 204,105,576 bytes, SHA-256
  `a8503ac467819a4fe9d7d3eed21d791c2112dfa89ef67189c6459392e44d7aa0`.
  Observed process/cache accounting peak: 288,223,232 bytes; this is not a
  total opaque CUDA-device allocation measurement.

Final `native-grouped-runner-quality03` completes with exit 0: formatting for
25 C++ paths, 20 boundary checks, 2,213 headers and REUSE 6.2.0 pass. Its literal
supervisor status and log are retained alongside the guarded source export.

This initial one-group proof is extended by N4b6e below; routed lookahead
remains refused. Actual model admission, original-model quality and speed remain
open. Page lookup currently scans all pages per expert/part; the small
fixture does not qualify 512-expert, 48-layer scaling. Existing supplemental fake
controls cover cancellation and unknown retirement; whole-runner failure injection
is not claimed. No new import or real-model payload read is needed for this proof.

## Ordered routed groups — 2026-10-12

N4b6e extends the same shared step to G ordered routing groups and G+1 segments,
with group/product counts derived from the actual selected plan. All segments
retain the original whole-plan placement and source/execution/cache identity.
Each group has a stable paid pinned slice. Its completed router output is fully
validated before acquiring its current selected closure under the still-held
base request. Capture construction does not count as completion; ordinal,
epoch and Bind-generation checks govern replay. Intermediate segments copy only
the next route; only the last publishes final outputs and carries. Unknown
completion retains the complete flight and quarantines state. Ordinary whole-step
entry points still refuse routed programs.

The new real ComponentStepSource fixture has two nonuniform width-32 MoEs after
D64 attention. The second router consumes the first group's actual output;
revisions change both selected sets while preserving repeated contributions.
All 288 full outputs pass the same independent scalar tolerance as N4b6d. A
private test-only friend inspects 36 actual IDs from completed retained pinned
slices, with current epochs; it grants no launch authority. Nine segments across
three eager/captured/replayed steps complete, one program is retained, 12 exact
part descriptors defer residency, and the state frontier reaches 9. Each group
requires fresh selected-page admission; ordinary/shared-page coverage remains
held. This is F16 state execution/frontier evidence, not an independent full
attention-state byte oracle: the attention output projection remains zero.

Three additional host cases check ordered completion/preparation, unknown
second-segment sequencing and real supplemental-scheduler behavior after a
successful first group. Underpayment refuses before the later launch; unknown
completion retains the base and selected leases, external counter and complete
payload. The injected failures use fake providers; full-runner GPU fault
injection is not claimed. Existing one-group and malformed-source cases pass
unchanged. No kernel arithmetic or numerical policy changed.

The fixed two-group fixture prepays 576 MiB catalog metadata, including a
384 MiB compiler allowance (derived bound 340,812,288 bytes), a 96 MiB complete
owner allowance and 64 MiB overlapping program/graph accounts. Its managed node
is 1 GiB; activation and pool each cap at 64 MiB. Nine weight pages plus the
slab read slot use at most 20 MiB + 8 KiB within a 32 MiB allowance. F16 cache
payload is 65,536 bytes, state layout stays within 128 KiB, and four times actual
state capacity is included before start. Two stable route slices total at most
32 KiB. Generic group metadata is charged before allocation using actual
selected-product counts; no model-specific count cap is introduced. Existing
serial complete-owner retirement remains mandatory. The actual job uses a
10 GiB process/cache cap, zero swap and a 6 GiB physical-margin check, not a
total opaque CUDA-device allocation cap.

Main evidence (raw retained outside Git):

- `native-grouped-ordered-main01/result.json`: full native build, 104 routed
  host controls plus 34 fusion/validation controls, no failures or skips;
  SHA-256 `a34963ec04a9137e0eea6c14985170051af9b58f218baa07bb1f65f5cf4430e7`.
- `native-grouped-ordered-gpu01/result.json`: all four cases PASS, unchanged
  source/binary guards, zero remaining compute processes and zero memory-event
  counters; SHA-256
  `9550430b58a7a4fb43f1cbdb19fd6dca430496d67081764e28409def07fb5e2f`.
- Matching source-map SHA-256:
  `27a5ff34c9bf5ece498298391d8d2e170fb5fae556c560ef39de16736f665347`.
  Executed binary: 204,695,792 bytes, SHA-256
  `b66ab7139c42d8ddd2ee32c90dd5cc1a4da48c52cb33c850011ae70a92169b12`.
  Observed process/cache accounting peak is 313,475,072 bytes, not total opaque
  device occupancy.
- `native-grouped-ordered-quality01` literal supervisor status/log: exit 0,
  format 12 paths, nine boundary checks, 2,213 headers and REUSE 6.2.0 PASS.

Native model admission, original-model quality, serving and performance gates
remain open. The bounded two-group proof does not qualify full 48-layer or
512-expert execution. Page inventory still uses full source scans; no page index,
fully-resident bypass, new import or real-model payload traversal is part of this
slice. CPU cost, memory traffic and reuse improvements may be useful independently
of model latency, but none is measured or claimed by these correctness controls.

## Ordinal page lookup CPU screen — 2026-10-12

Adopt the allocation-free ordinal lookup for complete current device-only
reservations. Its only production callers are the two `ExtentsTouching` calls
in `model_runner_routed.cc` that derive selected multipart weight closures.
Ordinary model execution does not call this helper. The change reuses existing
reservation/source arrays, adds no persistent memory or index, and preserves
page order, source ownership and scheduler residency admission. Host, partial,
released/retried and otherwise ineligible inventories retain the original scan.
Every touched entry is validated before any destination write.

The measured benefit is CPU lookup cost for large inventories and small selected
spans. Adoption accepts reproducible regressions elsewhere: at 64 pages with all
64 touched, pooled median wall cost rises from 103.1 to 319.9 ns/lookup; host
fallback rises from 55.8 to 129.4 ns at 64 pages and from 925.4 to 2,898.1 ns at
1,024 pages. At 16,384 pages, host fallback is also slower. These are measured
tradeoffs, not an unqualified speedup. No cause is assigned to compiler code
layout, frequency or caches: those were not measured. Short process duration and
timer precision limit generalization. The large selected-device CPU saving and
zero added persistent memory justify this scoped change; no model-latency or
migration speed-gate claim follows.

A/B/B/A ran on Spark A, with A the actual prior helper and B the candidate.
Both compiled source inventories contain 2,482 files and differ only in
`src/engine/paged_weights.cc`; the test, inputs and output work are identical.
Each fresh process runs twelve cells, one warmup and five recorded batches of
256 successful lookups per cell. All four complete output checksums agree;
every case has exactly 182 validated properties. Independent scan comparisons
remain outside timing. Larger inventories are paid test metadata over fake
virtual reservations: the 32 GiB address range has no payload backing or large
artifact. Small real Open/Reserve controls separately cover lifecycle geometry.

The following values are median [minimum–maximum] **nanoseconds per lookup**,
computed from ten recorded batch values per arm (five in each of two processes).
CPU and wall columns are separate. CPU output has a 1 microsecond reporting
quantum per 256-call batch (about 3.9 ns/lookup), not that accuracy per call;
steady-clock nanosecond units do not establish effective resolution. Zero values
would remain visible and would not be used as ratio denominators.

| Pages | Touched | Path | A wall | B wall | A CPU | B CPU |
| ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 64 | 1 | device | 70.3 [69.6–71.2] | 19.8 [19.6–20.2] | 70.3 [70.3–74.2] | 19.5 [19.5–23.4] |
| 64 | 8 | device | 74.1 [73.8–87.1] | 52.6 [52.5–95.4] | 74.2 [74.2–89.8] | 54.7 [50.8–97.7] |
| 64 | 64 | device | 103.1 [100.5–105.4] | 319.9 [319.2–320.2] | 103.5 [101.6–109.4] | 320.3 [320.3–324.2] |
| 64 | 1 | host fallback | 55.8 [55.5–55.9] | 129.4 [128.5–130.9] | 58.6 [54.7–58.6] | 130.9 [128.9–132.8] |
| 1,024 | 1 | device | 1288.4 [1280.1–1296.7] | 20.0 [19.7–20.5] | 1289.1 [1281.2–1296.9] | 23.4 [19.5–23.4] |
| 1,024 | 8 | device | 1280.6 [1273.6–1288.1] | 55.5 [55.1–56.1] | 1281.2 [1273.4–1289.1] | 58.6 [54.7–58.6] |
| 1,024 | 64 | device | 1307.3 [1300.3–1317.2] | 322.1 [321.5–348.5] | 1308.6 [1300.8–1320.3] | 324.2 [324.2–351.6] |
| 1,024 | 1 | host fallback | 925.4 [923.6–940.1] | 2898.1 [2885.5–2911.6] | 927.7 [921.9–941.4] | 2900.4 [2886.7–2914.1] |
| 16,384 | 1 | device | 28109.4 [27407.8–28841.6] | 21.0 [20.0–26.9] | 28111.3 [27410.2–28843.8] | 23.4 [19.5–27.3] |
| 16,384 | 8 | device | 28099.6 [27392.8–28862.1] | 27.9 [26.4–30.9] | 28101.6 [27394.5–28820.3] | 29.3 [27.3–31.2] |
| 16,384 | 64 | device | 28140.4 [27424.6–28856.2] | 156.3 [150.3–163.1] | 28142.6 [27425.8–28859.4] | 158.2 [152.3–164.1] |
| 16,384 | 1 | host fallback | 22224.7 [21859.0–22586.6] | 24562.4 [24359.9–24748.4] | 22226.6 [21863.3–22585.9] | 24562.5 [24359.4–24750.0] |

[The compact aggregate](ordinal-cpu.json) retains per-process cell ranges and
outer process observations. A1/A2 outer wall times are 0.2660/0.2662 seconds;
B1/B2 are 0.1156/0.1156 seconds. Their process CPU times are
0.2392/0.2277 and 0.0938/0.0932 seconds respectively. These include startup,
validation and reporting and are not isolated helper timings. Both candidate
processes reproduce the slower small dense and host-fallback cases. The combined
screen took 0.7652 seconds under its fixed 60-second ceiling.

Algorithmically the prior successful scan visits all P source records twice;
the eligible ordinal path checks two endpoints and T touched records, then
copies T IDs. These are source-derived loop counts, not measured bytes, cache
misses or memory bandwidth. A fully resident execution bypass and broader
inventory optimizations remain separate; no new numerical policy is installed.

The native fixed metadata bound remains at most 12,845,056 bytes within its
256 MiB envelope. The whole CPU job enforced a 1 GiB process/cache limit and
zero swap; observed peak was 20,336,640 bytes, with no memory-limit/OOM events.
Pre/post source, binary and recipe guards passed, all four child process groups
retired, and no case failed or skipped. No GPU work occurs in this timing job;
exclusive supervisor admission merely isolates the host.

Main evidence, with raw logs/XML outside Git:

- `paged-weight-ordinal-main02/result.json`: full native build and 157 affected
  host cases PASS, including 19 release/mapping cases (five new controls), no
  failures/skips. Receipt SHA-256
  `bad9a9e6bbc0962cbd996c883cbbe9a4427d72aba7398cc50a4e8a8116132e80`.
  The baseline `parent02` build also passes all 19 release/mapping controls.
- `paged-weight-ordinal-gpu01/result.json`: all four routed cases PASS, source
  guards unchanged, zero remaining compute processes and memory-event counters;
  SHA-256 `fc034044b1ae615f128a1d52e610f80f2ce6c1fda5a07b05cd9a4b9aa70a80d2`.
- `paged-weight-ordinal-cpu01/results/result.json`: four screened processes PASS;
  SHA-256 `d307a2543713dee16c9c8177eb87b5f35543a1ef6e1f7d0d6b5fb85b5e8f0f6b`.
  Candidate binary: 38,363,136 bytes,
  `55c95415ff1d655b96618a20ce2fdd1147e828885aadf10e7e1de4380628aa16`;
  baseline: 38,360,384 bytes,
  `604886d86a2f9f8ce714567c214c2784650c55c20ca80f1d99d02b2874573327`.
- Initial `main01`/`parent01` builds stopped at the test fixture's omitted
  `PageSource.read` initializer. Those failed receipts remain preserved; no
  timing or tests ran in them. Explicit empty initialization corrected the
  warning without changing production behavior.

Full-model quality, serving and performance qualification remain open. This
screen neither executes model payloads nor changes accepted arithmetic.
