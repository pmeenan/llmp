<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Ready NormGate median-bookend screen

The [registered revised screen](READY-NORMGATE-MEDIAN-BOOKENDS-REGISTRATION.md),
commit `f53312bb9c2a6e976dd9c5390e0ca3b0a4e4544f`, ran once on `spark-b`.
The F32 NormGate 128-dimension, 48-head, one-row fused body median is
**3,712 ns**, versus **5,664 ns** for its own normally planned decomposition,
classified as an incumbent win. Median-of-nine alternative bookends are
**5,600 and 5,824 ns**, a **4%** difference within the unchanged 10% bound.
The earlier [20% single-bookend refusal](READY-NORMGATE-CAPTURED-EVENTS-RESULTS.md)
remains refused; no observation from that run is used here.

All 52 observations remain in their registered order, including every one of
the eighteen individual bookend readings. Those readings range from 4,736 to
5,920 ns. Median drift measures typical latency movement; it does not claim
individual-sample stability or explain the earlier refusal. No reading is
filtered, and no automatic retry ran.

Both changed original-IR frames pass before comparison. Every arm independently
restages both original F32 inputs, finishes its pure prefix, replays one complete
body, waits for completion, then checks all CPU outputs, immutable gamma and
actual 3/5-node capture inventory. The two extra graph nodes are timing events.
Normal completion follows teardown, supervisor exit is zero, source/binary
identities match before and after, and a separate post-run check reports no
remaining GPU compute process.

Independent result review is clean. The raw candidate retains `UNREVIEWED`
provenance; a separate reviewed row is installed. Native qualification on
`spark` passes 85 targeted checks without skips. At implementation identity
`5bfcfa9ed5b5d01720b9aedf656d9051b93e066930d1ca694827094ff0645ed5`,
the one-row descriptor matches its incumbent preference, a two-row control
remains unmatched, and the NEOX descriptor still matches. The generated table
contains fourteen recipe rows and twelve unchanged legacy rows.

Its isolated physical key differs from the actual full GDN programs, whose
first screen was neutral; this result transfers no Ready key and qualifies no
model optimization. Other shape/format coverage and U6 remain open. Future
implementation changes require fresh exact keys; this row is never re-pinned.

Recipe-key digest is
`e55a5bb1499432c3147f39f18730844ac8638a137603bdd91ff3fc98406f5452`;
protocol SHA-256 is
`b06453b54106e89271ee79e8c2b2670cbc31f9de25c12780d43971a7620ca395`.
The compact ordered-event receipt is
`12f69e73b93d0491f3765266483ff54e758009c7ab0a76e6df9b9e444d6d459f`.
Raw evidence remains outside Git at
`/tmp/llmp-m35-orchestration/u6-resume/normgate-median-r50/screen.log`
(SHA-256 `434a535897a83b9ac6eca501ec8bb17f9b9e698cd2722a6f9e6ce697908b8b6e`).

The subsequent F16 comparison-proof extension changes backend identity to
`6dd80faece88fa94656671d353adf4a1b1401d1470e23bf0339dfa2377db300c`.
This qualification is historical: validation-only checks confirm the unchanged
row no longer matches. It is not re-pinned, and no replacement timing ran.
