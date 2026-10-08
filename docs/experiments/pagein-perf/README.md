<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Page-in through the landing zone at disk speed — 2026-09-27

Backend proof P2 measured the runtime's page-in through the landing zone
(D-081) at 11.4 GB/s, and 11.9 with backing mapped once, against 13.8 for
direct reads in place into host VMM
([P2](../backend-proof-p2/README.md#rungs-4-and-5-paged-into-device-vmm-through-the-landing-zone)).
D-081's standalone measurement had found no cost to the copy at four reads
in flight ([dmabuf-direct](../dmabuf-direct/README.md)). That is D-081's
reopen condition. This experiment profiles the runtime's pipeline end to
end, fixes what limited it, and measures the zone against in-place reads
and the standalone probe on the same file and host.

**Answer: the gap was in the runtime's lanes, and it is closed at two and
four in flight.** There, loads through the zone now match in-place reads
and the probe: on an idle `spark` at four in flight, 14.93 GB/s with
backing mapped once and 14.69 with backing made per load, against 14.70 in
place and 14.90 for the probe's zone (medians, 8 GiB of a freshly written
file). At two in flight all are at 13.0–13.3. Before, the same harness
measured 12.35 and 12.04 through the zone against 14.54 in place. `spark-b`
agrees within 0.15 GB/s at four. At eight, backing mapped once matches too;
backing made per load varied widely on `spark` (median 13.91, below). Three
causes, each measured on its own:

1. **Reads reached the SSD out of order (RE-026).** The direct reader
   started queued reads in the order of their keys, which are mailbox
   indices the board reuses. Through the zone, about 60% of consecutive
   submissions were locally swapped; in place, the keys were fresh and
   in order. At 2 MiB-multiple offsets the SSD does not care (14.3–14.9
   GB/s either way), but at the artifact's 4 KiB-aligned offsets the
   shuffled reads ran at 11.7–12.4 GB/s, with the copy or without it, and
   at 8, 16 or 32 slots and depth 4 or 8 alike. The reader now starts reads
   in arrival order, continuations and retries first
   (`providers/direct_reader.h`).
2. **Lanes slept between a load's reads and copies (RE-017).** The device
   submission lane slept on its queue after each copy and took 94–199 µs
   at the median to wake for the next (p90 ~380 µs); the storage lane
   slept in `io_uring_enter` and took up to ~200 µs (p90) to pick up a new
   read. Each landing slot turns over only after its read, the copy and
   those hand-offs, so with 2 × depth slots the storage queue ran below
   depth. Both lanes now poll for 200 µs after their last progress, as the
   scheduler already did (`DeviceSettings::poll_window`, `StorageService`'s
   `poll_window`), before they sleep.
3. **VMM work held up the copies.** With backing made on each load (D-033),
   the submission lane also ran `cuMemCreate` (~70–76 µs per 2 MiB extent),
   `cuMemMap` (~1 µs) and `cuMemSetAccess` (~40–43 µs): 78% of its time at
   disk speed, and copies waited behind it (133 µs at the median, 353 at
   p90). A VMM lane of its own (`BackingService`) now takes it; nothing in
   the protocol relied on the queue ordering them (the scheduler publishes
   a copy only after its mapping completed, and an unmap only after every
   lease is released).

Not a cause: the zone's size, the copy engine, and reading per chunk. With
the fixes, 2 × depth slots suffice, and the storage-queue study found
equal bandwidth for 2, 4 and 8 MiB requests at equal bytes in flight
([storage-queue](../storage-queue/README.md)), so coalescing reads (BP-P1)
would reduce operations, not raise bandwidth. (Built later, it did not,
and is off by default:
[BP-P1](../backend-proof/README.md#bp-p1-coalesced-reads).)

**What disk speed is (RE-027).** The ~14.9 GB/s above, like every earlier
measurement of this SSD (io-path, storage-queue, dmabuf-direct), reads a
file written shortly before. Files at rest read at ~13.3 GB/s by any path:
a six-day-old GGUF shard read at 13.4 through the zone with backing mapped
once, 13.2 with backing made per load, 13.2 in place and 13.3 for the
probe (`spark-b`), and the FP16 artifact's shard, installed four days
before, is why P2's harness never showed more. Loads through the zone
match in-place reads in both regimes; the owner's ~15 GB/s is this
drive's bandwidth only for recently written data.

## Changes

- `DirectReader` starts reads in arrival order: a read's requests go to
  the provider before a later read's; a short transfer's remainder and a
  retry go ahead of reads not yet started. It also no longer walks every
  queued read twice on each poll, only the reads a completion or a
  withdrawal touched: with ~2,000 reads queued in place, a completion had
  taken 48 µs at the median (p99 183) to be published.
- `StorageService` and `DeviceService`'s submission lane poll (with
  `yield`) for a bounded window after their last progress before they
  sleep: 200 µs by default, the scheduler's window, not a tuned value;
  zero never polls; bounded at an hour, checked when built.
- `BackingService`, the VMM lane: managed backing's create, map and set
  access, and unmap and release, carried out as the device lane did, on a
  thread of its own. `Lanes::backing` routes VMM work to it; without one
  it goes to the device lane, as before. Programs that page through the
  zone (`llmp_fp16_paged`, the new harness) wire it.
- `PageInObserver` (`SchedulerSettings::observer`): the scheduler tells an
  optional observer when a page-in's backing is mapped, its read is
  published, and it is published resident or fails. The harness times
  per-extent latency with it.
- `llmp_pagein_bench`: the scheduler and every lane on its own thread,
  loading a file of 2 MiB extents through the zone or in place, with
  backing mapped once or made per load, at a chosen depth, zone size and
  file offset, verified against the file on request; or a decode-like
  loop with sparse page-ins, reporting each lane's CPU (below).
- One caller per device-memory provider (`device_memory.h`): with a VMM
  lane, the device lane is given no provider, and a change to a provider
  while another is under way is fatal
  (`DeviceMemoryCallerDeathTest`).

Tests: `unit.VmmWork/PageInTest.*` now runs every case twice, with VMM
work on the device lane and on a VMM lane (`OnTheDeviceLane`,
`OnAVmmLane`); `VmmLaneTest` checks that a copy is carried out while VMM work waits and
that the VMM lane refuses other work; `StorageLanePollTest` and
`DeviceLanePollTest` check that a polling lane takes every command and
stops on close however long its window, and that unbounded windows are
refused; `ReaderOrderTest` checks arrival order against key order, with a
continuation first; `unit.VmmWork/CudaPageIn.*` (`gpu`) runs both ways
on the Spark. Rungs 4 and 5 re-ran exact with the new lanes (P2's
report).

## Results

Session `s1` on an idle `spark` (conditions below): 8 GiB of a freshly
written pattern file, 2 MiB extents, reads starting 4 KiB into the file, a
zone of 2 × depth slots. GB/s, median (range); n is loads for the harness
(four per process after its first, two processes) and passes for the
probe (three per process, two processes).

| Depth | Probe, zone + copy | Probe, in place | Zone, backing once | Zone, backing per load | In place, backing once | Before: zone, once | Before: zone, per load | Before: in place |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 13.15 (13.10–13.24) | 13.28 (13.16–13.36) | 13.20 (12.81–13.36) | 13.04 (12.75–13.18) | 13.16 (13.09–13.20) | 9.93 (8.94–10.99) | 10.11 (10.03–10.29) | 12.12 (11.97–12.22) |
| 4 | 14.90 (14.78–14.93) | 14.93 (14.82–14.95) | 14.93 (14.91–14.94) | 14.69 (14.67–14.92) | 14.70 (14.70–14.71) | 12.35 (12.31–12.47) | 12.04 (11.78–12.31) | 14.54 (14.45–14.61) |
| 8 | 14.94 (14.93–14.96) | 14.94 (14.93–14.96) | 14.93 (14.93–14.95) | 13.91 (12.98–14.70) | 14.71 (14.70–14.72) | 12.18 (12.11–12.55) | 12.28 (11.60–12.60) | 14.71 (14.69–14.95) |
| n | 6 | 6 | 8 | 8 | 8 | 8 | 8 | 8 |

Per-extent latency, µs: the medians over loads (or passes) of each one's
p50 and p99.

| Depth | Probe, zone: submission → usable | Zone, backing once: read published → resident | Zone, backing per load | Before: zone, once |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 350 / 544 | 622 / 942 | 618 / 940 | 784 / 1,635 |
| 4 | 597 / 726 | 1,118 / 1,260 | 1,117 / 1,256 | 1,322 / 2,491 |
| 8 | 1,155 / 1,362 | 2,235 / 2,408 | 2,234 / 2,494 | 2,207 / 7,635 |

Session `b2` on `spark-b`, the same design with another agent on the host
(conditions below), agrees at four in flight: zone 14.87 (14.84–14.89)
with backing once and 14.63 (14.53–14.86) per load, in place 14.64
(14.61–14.81), probe zone 14.76 (14.64–14.83); before, 12.16, 12.03 and
14.39. At eight: 14.89, 14.62 and 14.63 against the probe's 14.89. At two:
12.92, 13.01 and 13.05 against the probe's 12.97.

The runtime's figure includes each read's wait for room behind the reads
ahead of it (a zone's worth, 2 × depth × ~140 µs); the probe submits only
when there is room. Measured from its submission to the kernel, as the
probe measures, an extent took 612 µs at the median (p99 829) at depth 4
(the trace below). In place, every read is published at once, so the same
figure is the load's own length (~300 ms) and is not shown.

- **Against D-081's condition:** at four and eight in flight, loads through
  the zone with backing mapped once equal the probe's (14.93 against 14.90
  and 14.94 on `spark`) and exceed the runtime's in-place reads (14.70,
  14.71). With backing made per load they are 1.6% below backing mapped
  once at four in flight and equal to in place; at eight their spread was
  wide on `spark` (12.98–14.70, median 13.91; `spark-b` gave 14.60–14.75).
  The VMM lane runs at ~78% of a thread at disk speed (below), the one
  stage with little room, and part of the remaining gap is there: in a
  later look on `spark-b` (the pattern file then at rest, depth 8, three
  processes of five loads each way), backing made per load ran at
  12.47–13.07 GB/s against 13.20–13.43 mapped once. The slowest process
  (12.47–12.59) had at least a tenth of its extents' reads waiting on
  their mapping in every load (map lead p10 of 0), but loads whose
  mappings stayed 0.3–2 ms ahead still ran ~3% below backing mapped
  once, so the VMM lane's queue is not all of it (not investigated). At two
  in flight, zone, in place and the probe are within 1–2% of each other,
  ~11% below four, as the storage-queue study found (13.3 GB/s).
- **In place** (backing mapped once) is 1–2% below the probe's in place at
  four and eight in flight: the whole load's reads are published at once
  and pass through the storage lane's queue while it also harvests. Not
  pursued: it is D-034's path, which D-081 replaced for execution.

The FP16 artifact through `llmp_fp16_paged` (`run_paged.sh loads`'s
variants: its 490 device weight chunks, 988 MB, depth 4, 8 slots; five
loads per process after its first, two processes), GB/s, on `spark`
(`spark-b` in brackets):

| | Zone, backing per load | Zone, backing once | In place, backing once |
| --- | ---: | ---: | ---: |
| After | 13.14 (12.90–13.24) [13.10] | 13.35 (13.25–13.40) [13.25] | 13.28 (13.07–13.41) [13.22] |
| Before | 10.94 (10.30–11.66) [10.89] | 11.50 (11.31–11.75) [11.21] | 12.71 (11.87–13.24) [12.54] |

These stop at ~13.3 GB/s because of the file (RE-027): the harness read a
fresh `dd` copy of the same shard at 14.3–14.9 GB/s by either path, the
installed one at 12.2–13.3.

## Where the time went

Stage timings from a temporary trace (per-operation timestamps at the
scheduler's publications and observations, the storage lane's command,
submission, reap and completion, the device lane's pop and issue, and the
fence seen), on `spark-b`, reading 2 GiB of the pattern file at a 4 KiB
offset, depth 4, 8 slots, the third load of a process. The trace is not
committed. µs, median (p90).

| Stage | Before | Reads in order | All three fixes |
| --- | ---: | ---: | ---: |
| Consecutive submissions out of file order | 627 of 1,024 | 0 | 0 |
| Kernel read, submission to reap | 618 (774) | 527 (651) | 527 (545) |
| Copy published → device lane takes it | 94 (115) | 199 (380) | 0.7 (1.5) |
| Copy issued → fence seen | 88 (359) | 99 (112) | 44 (46) |
| Copy published → observed complete | 204 (486) | 302 (499) | 53 (62) |
| Time with 4 reads in the kernel | 61% | 66% | 77% |
| GB/s (this load) | 11.92 | 13.68 | 14.61 |

The last column is from an 8 GiB load with backing made per load, before
the storage lane polled. Another process's kernels ran on the GPU during
the first; it was idle for the other two. The copies' duration varied
between processes (36–44 µs in some, 88–100 in others), not investigated.
With the fixes, 23% of the time the kernel held three reads rather than
four, as in place: that is the storage lane's own turnaround between a
completion and the next submission, which both paths share. At depth 4 an extent's
latency from its read's submission to the kernel until it was published
resident was 612 µs at the median (p99 829), against the probe's 596–628
from submission until usable, on the same host.

With backing made per load and the VMM work on the submission lane, that
lane was busy 78% of the load: maps took 110 µs (p90 121) each and copies
waited 133 µs (p90 353) behind them. `cuMemCreate` and `cuMemSetAccess`
cost the same per extent at 1,024 and 4,096 live allocations
(69–74 and 39–41 µs), and batching `cuMemSetAccess` saved little: 43 µs
per extent in batches of 8, 36 in batches of 64. Two or four threads
making backing concurrently took longer in total than one (114–116 and
127–144 µs per extent against 101–104): the driver serializes them. (A
standalone driver-API loop on `spark-b`, with other processes' CPU load
at 10–17.) Fresh device backing therefore costs ~100–110 µs of one
thread per 2 MiB extent, a ceiling of ~19–20 GB/s for loads that must
create it: above disk speed with ~35% to spare. D-033's hand-off of a
victim's backing to a load (not built) would skip the create.

## Polling's cost during decode

`llmp_pagein_bench --decode` runs a decode-like loop: one task's steps,
each a device job that spins 100 µs on the host and queues a memset of
16, 256 or 1,024 MiB (steps of ~0.17, ~1.6 and ~5.6 ms), the next once its
fence completes, while a load of one more 2 MiB extent through the zone,
with backing made per load, is posted every 5 ms. On `spark-b`
(2026-09-27, three interleaved rounds, each process started once no other
compute process had held the GPU for 10 s), with the six threads confined
to three cores (`taskset -c 0-2`), the storage and submission lanes' 200 µs
windows against none:

| Step | Step p50 / p99, µs (200 µs / none) | Submission lane, cores | Storage lane, cores | Process, cores |
| ---: | ---: | ---: | ---: | ---: |
| ~0.17 ms | 173 / 341 against 182 / 397 | 0.99 against 0.56 | 0.037 against 0.021 | 2.45 against 1.97 |
| ~1.6 ms | 1,552 / 1,720 against 1,554 / 1,760 | 0.22 against 0.08 | 0.050 against 0.021 | 1.55 against 1.39 |
| ~5.6 ms | 5,603 / 5,986 against 5,601 / 6,006 | 0.079 against 0.025 | 0.049 against 0.019 | 1.37 against 1.29 |

(medians of three runs.) The windows cost a lane at most its window after
each step or page-in, and bought a faster loop when steps were shorter
than the window; they did not slow the steps even with more threads than
cores. Unconfined, the runs varied more (another agent used the host
between them) and showed the same CPU. Most of the loop's CPU was
already the device completion lane, which yields while a fence is
pending (0.87–0.97 of a core here; `DeviceSettings::poll_sleep`, since
replaced by the runtime wake, D-094), and the
scheduler (0.19–0.33, or all of one core with the shortest steps).

## Conditions and provenance

- **Hosts.** `spark` (`spark-c4e2`) and `spark-b` (`spark-56f5`): GB10,
  kernel 7.0.0-1019-nvidia, driver 580.178.04, persistence mode on; each
  SSD a Samsung `MZALC4T0HBL1-00B07` holding the ext4 root. Each host's
  16 GiB pattern file (`dmabuf_probe create`) was written for this
  experiment, on `spark` two minutes before its session. The FP16 artifact
  is `b93cdc32…` as installed on 2026-09-23.
- **Session `s1`** on `spark`, 2026-09-27 22:45–23:00 UTC, 60 processes,
  two of the three planned rounds (stopped when the host was needed for
  another measurement; no process of it overlapped one). Every start found
  no other compute process, a 1-minute load of 0.31–0.88 and the GPU in
  P8. Binaries: harness `2e6b5956…` (after) and `d0e6bff5…` (before),
  `llmp_fp16_paged` `2fe3f465…` and `b00e5609…`, probe `3faaa9ed…`.
- **Session `b2`** on `spark-b`, 2026-09-27 21:12–21:27 UTC, 60 processes, two rounds.
  Another agent was using the host: each process started once no compute
  process had held the GPU for 10 s and the 1-minute load average stayed
  under 2.0. At the starts the load average was 0.51–1.54, one start found
  another compute process on the GPU, and the GPU was in P8 (P0 once).
  Binaries: harness `a15647a3…` (after, built before two last edits: a
  comment, and where VMM work goes when there is no VMM lane) and `d0e6bff5…` (before), `llmp_fp16_paged`
  `7b01c4c4…` and `771627df…`, probe `d231f159…`.
- **Builds.** `spark-native`, SDK `aarch64-e0a0c85c42806fb1`; the probe
  with CUDA 13.0 `nvcc` and `probe_offset.patch`.
- **Traces and single measurements** in the other sections, and the
  at-rest file, were taken on `spark-b` the same day during development,
  with the conditions stated where they matter. Raw results stay on each
  host under `~/.local/share/llmp/pagein-perf-20260927/`.
- **Not measured on an idle `spark`:** files at rest (RE-027).

## Method

- **Harness.** [`benchmarks/pagein_bench.cc`](../../../benchmarks/pagein_bench.cc)
  registers one extent per 2 MiB of the file and loads them all with one
  task that materializes them together, then evicts them (untimed). A load
  is timed from posting its task to the task's finish on the scheduler
  thread. Per-extent latency is from the read's publication to the storage
  lane until published resident (the observer), so it includes the wait
  for room in the kernel's queue behind the reads ahead of it: about
  depth × 2 × 140 µs at disk speed.
- **Before.** Commit `c3e441a` (this change's parent) with only the
  harness and the scheduler's observer hook added, built the same way.
- **Probe.** `dmabuf_probe restore` from
  [dmabuf-direct](../dmabuf-direct/README.md), with
  [`probe_offset.patch`](probe_offset.patch) so its reads start 4 KiB into
  the file as the harness's do (its own verification then reports every
  word bad; the reads and copies are unchanged). Its landing zone is
  2 × depth slots, as the runtime's. Latency is from submission to the
  kernel until usable.
- **Session.** [`session.sh`](session.sh) interleaves every variant in each
  round, each process starting only once no compute process has held the
  GPU for 10 s and the 1-minute load average is low; the conditions at the
  start of each process are recorded with its results. A harness process
  loads 8 GiB five times and the first load is dropped as a warm-up; a
  probe process restores 8 GiB four times and pass 0 is dropped. The FP16
  harness (`run_paged.sh loads`'s variants) loads its 988 MB six times,
  first dropped.

## Limitations

- One SSD model, sequential loads of one file, nothing else reading or
  writing, and no GPU work during the load. The copy's effect on
  concurrent kernels (D-081) is still unmeasured, and so is paging during
  decode, where the lanes' polling windows start and stop more often.
- Polling costs a core per polling lane while a load runs (the device
  completion lane already yields while fences are pending); between
  sparse page-ins in a decode-like loop it cost little (below). The
  windows are not tuned; M3 onward sets them against per-token latency.
- Fresh device backing runs at ~78% of one thread at disk speed (above);
  a slower driver or a larger device would make the VMM lane the limit.
- RE-027's cause is not isolated: fragmentation is ruled out (`filefrag`),
  and the drive's write cache is inferred from the timing, not proven.

## Reproduce

On a Spark with the pinned SDK:

```sh
mise run build -- spark-native --locked
# the probe with the offset patch, and a 16 GiB pattern file
cd docs/experiments/dmabuf-direct && git apply ../pagein-perf/probe_offset.patch
./build.sh /path/probe_off && /path/probe_off create /path/pattern-16g.bin 16
git checkout dmabuf_probe.cu
build/spark-native/benchmarks/llmp_pagein_bench --file /path/pattern-16g.bin \
  --gib 8 --offset 4096 --mode zone --backing managed --depth 4 --loads 5 --verify
sh docs/experiments/pagein-perf/session.sh AFTER BEFORE PROBE FILE ARTIFACT TOKENS OUT 3
python3 -B docs/experiments/pagein-perf/summarize.py OUT/results.txt
```
