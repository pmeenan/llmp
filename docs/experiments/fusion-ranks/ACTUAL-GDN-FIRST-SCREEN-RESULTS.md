<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Actual full GDN first screen

The [registered full-program screen](ACTUAL-GDN-FIRST-SCREEN-REGISTRATION.md),
commit `1ff8d420127b06000123dbba730ee739303f4273`, ran once on `spark-b`.
The actual default program median is **84.864 us** versus **85.312 us** for
its own NormGate decomposition, with GdnStep retained in both. The registered
10% classifier returns **neutral**. Alternative bookends are 84.352 and
84.544 us, a **0.228%** difference, within the prospective bound. This closes
the first correctness/performance screen for these actual programs; it does not
establish a transferable Ready key or qualify a model optimization.

All eight logits and both complete updated state arrays are checked against
the original-IR CPU reference after every arm. Both ordinary and captured
programs first validate two changed token/state frames. Every replay starts
with independently NaN-poisoned activation storage, identical restored state
and its own staged input slots, with completion before timing begins. The
captures contain 21/23 actual graph nodes, including two timing events and
exactly one complete program. Required activation storage is 205,824 bytes
for each plan; both use a fixed 32 MiB arena and scratch pool, and full state
is 3,268,608 bytes.

All 36 ordered observations, nine samples per arm, frozen source, binary,
backend and protocol identities match registration. Normal completion follows
recording/program/launch/backend/stream teardown; supervisor exit is zero and
a separate post-run check reports the host free of GPU compute processes.
Independent numerical, event-order, protocol, identity and retirement review
is clean. No rank row is produced or installed;
the diagnostic result explicitly sets `adoptable=false` and denies Ready-key
transfer. The existing default stays in place. Broader fusion shape/format
measurement coverage and U6 remain open.

Raw medians are `84863.99799585342` and `85312.00140714645` ns. Half-up rounding
gives 84,864 and 85,312 ns for the registered integer classifier. Raw alternative
bookends are `84352.00154781342` and `84544.00300979614` ns. Protocol SHA-256 is
`1fe490da95280eff980acda68a94aeab3d5c32ccec5aa6758c5457283dd694bd`.
Raw evidence stays outside Git at
`/tmp/llmp-m35-orchestration/u6-resume/captured-events-r47/gdn-screen.log`
(SHA-256 `ddcfa81ebc9080333d8b3001fc865b07a041b773e697bd4d71255bb48d4a74b3`).
