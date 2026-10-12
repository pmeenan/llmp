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

N4b6b's bounded owned operation controls are complete. The shared
ModelRunner mid-step routing/admission continuation remains open: production
still needs explicit completed routing segments and source-selected grouped
binding over the now-checked supplemental admission prerequisite. These tests do not provide
that interface. Native ModelOpt target admission and full-model qualification
remain open. This evidence makes no speed or model-quality claim.
