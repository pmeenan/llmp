<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# DeepSeek routed expert-list preparation screen

2026-10-04, Spark B: preparing a routed product's expert-token lists once
instead of rediscovering them in every output-row block is inconclusive
on the representative DSpark four-request wave. Candidate median latency
falls 0.544%, while the control bookends move down 0.984%. Keep the current
inline ballot scan; no production implementation changes.

## Isolated factor and paid screen

The private candidate adds a warp per selected token/expert pair to prepare
its first-owner count and ordered list. The vector products copy that list
to shared memory before their original arithmetic. Routing, pair order,
quantization and per-output sums stay fixed. This differs from the rejected
[multi-token weight-decoding passes](../deepseek-expert-passes/README.md).

Selection is bounded to six selected experts, 256 available experts and
more than four product rows. Other shapes keep the original inline scan.
Each list has 129 I32 entries (count plus at most 128 encoded pairs);
workspace is declared, charged and drawn from the existing stream pool,
at most 66,048 bytes before 256-byte rounding. Graph capture records the
preparation kernel and the consumer over the same stable scratch address.
No host route discovery or persistent weight replica is introduced.

| Arm, fresh process | Four-request draft/verify median ms | Timed joined waves s |
| --- | ---: | ---: |
| Production before | 232.720 | 8.6456 |
| Prepared lists | 230.315 | 7.6001 |
| Production after | 230.430 | 7.5755 |

Candidate latency is 0.544% below the mean control median. The earlier
control is slower across several wave widths, so the joined-wave total difference
is not a defensible candidate gain. One triplet does not establish a stable
benefit. No HTTP, context, swap or expanded quality ladder follows.

Every arm has 32 waves, including twenty at width four, seven at width
three and five at width two. Each completes 380 solo and 336 wave tokens,
with the cancelled member's truncated contribution. Graph counters are
19 eager, eight captured and 154 replays, with no refusals. The existing
benchmark checks DSpark drafts and complete verify outputs against their
solo steps, discard/rerun, initialized state outside permitted draft-ring
writes, and the departed member's unchanged state. All report zero exact
mismatches/stale bytes, exact discard/rerun and unchanged departed state.
Its `rows_compared`/`rows_identical` fields count plain decode rows, so both
are zero in this verify-only screen; they do not count the DSpark checks.
These checks are against each arm's solo path, not a saved cross-binary
full-state digest collection.

All three children exit zero and are reaped. Six strong admission/retirement
gates require at least 105 GiB available and clear GPU, container and native
model-process probes. Timing includes joined draft plus verify, not verify
alone; scalar fallback steps are excluded from the joined-wave total.
No bandwidth, registers, hardware occupancy or general calibration is measured.

## Reproduce and evidence

Compile the current production control `86b378f` and the private four-file
patch in separate source snapshots. Build only `llmp_spec_runner`; source
sync uses `rsync -rlpc --exclude=.git --exclude=/build`. Every build and
model run uses installed `~/.local/bin/spark-job start --gpu --name NAME
--timeout 600`, then `wait NAME`.

Use the first four `fast-swap/prompts.json` entries with:

```text
--check wave --tokens 96 --context 16384 --max-rows 4096 --draft 3
--slots 4 --wave-mode verify --wave-lanes on --joined-drafts on
--graphs on --prefill-outa-hca on
```

The community target artifact is
`cd39d504dc2dbfe911a4a521fa8efc8053dc3e80e99738a9b25fa6b70c97a1ac`,
DSpark `dd2d3f9c66f070fb231d27d5a11f38ff22c78dc8f089cecedbb67721e9b4bec5`.
Host `spark-56f5`, GB10, driver 580.178.04, SDK CUDA 13.4.92; the same
SDK cuBLAS path is supplied to every arm. The controller checks frozen
binary hashes, each arm's separately retained compiled-source inventory,
and the live candidate sources before and after every measured arm. The
live tree is not the control's compiled source. Each inventory covers 487
implementation/build files.

Raw controls, source patch/inventories, binaries and logs remain in
`spark-b:~/scratch/dsv4-expert-worklists/`, final run `screen2/`; the warm
private tree is `~/src/llmp-wt/dswl0001/`. Local evidence is retained in
`~/scratch/llmp-m3-dsv4-expert-worklists-2026-10-04/`. The baseline build,
corrected candidate build and `dsv4-worklists-screen2` finish zero and are
waited on. An initial candidate build lacked an explicit unsigned grid
cast; an initial screen copied binaries without execute permission and
launched no model. Both failures are corrected and excluded. Rejected
private code stays out of production; no repeated full suite or deferred
workstation/package checks are claimed.

| Evidence | SHA-256 |
| --- | --- |
| Production benchmark | `72124875fe8d7fc4cf781074b72827f8e352477db78d8bb2eb9a8a7c37e1a596` |
| Candidate benchmark | `fffb47d63b70e0799ea4f29b9a3d9fe35bba06bc7448827e687acc8eed66d2ae` |
| Production source inventory | `4a05996429a4d9f1988dda047782ab61897c50fe83fff61096d5266149b77235` |
| Candidate source inventory | `2a02418ff5f46dd0960e141eea54d6b8a5efa5ae5e394a7b8b95e62b0b7f7188` |
| Private source patch | `b5ca35e3439fe637b39e1acc59777efc523f07f4153dd595bb8dddc084f1ca93` |
| Controller | `00d2b2379c00e5aaf101a5cbcdb03c31828481d59d42baffb92c7b5db14bf545` |
| Final receipt | `3ed2ce9ca78d93f57ea13e0841a15642f12047023518b4ff8ca522e364962a05` |
