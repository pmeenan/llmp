<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Native grouped-operation foundation

N4b6a provides checked CPU worklist inversion and multipart descriptor geometry.
The isolated helper build and all eight focused host controls pass. Main also
passes the full native build and the same eight controls, with zero failures or
skips. Independent whole-source challenge is clean. No GPU or model was executed.

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

## Remaining scope

N4b6b is open: a bounded owned operation must derive descriptors from actual
source bindings, acquire only selected expert closures, preserve current leases
and preparation epochs, and prove eager/captured execution with changed routing.
The shared ModelRunner mid-step routing/admission continuation is also open;
these helpers do not add it. Native ModelOpt target admission and full-model
qualification remain open. This evidence makes no speed or model-quality claim.
