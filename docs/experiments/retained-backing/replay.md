<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Retained-backing deterministic replay — 2026-09-27

Part (a)'s deterministic replay of the
[retained-backing comparison](../../backend-proof.md#retained-backing-comparison)
(D-035, D-079): the 13 designs the frozen criteria name, replayed on the
fake backend over the [swap trace](README.md)'s primary seed at its three
budgets, with the deterministic criteria applied mechanically. The
confirmation seed was not generated, read or replayed, and no timed session
has run.

**Result.** No slab or hybrid design meets every deterministic criterion at
every budget, so none can replace D-033 on this seed (the amending rule
needs every budget and both seeds). Nine designs meet them at 64 GiB only:
contiguous-run eviction and compaction at each slab size, size classes at
32 MiB and the hybrid at 32 and 256 MiB. At 53 and 40 GiB every design's
peak waste is above D-033's (6.3–19.4 GiB against 5.16 at 53 GiB,
5.9–18.9 against 3.98 at 40 GiB).

## Harness

- [`benchmarks/retained_backing/`](../../../benchmarks/retained_backing/):
  `llmp_rb_replay` reads the trace after checking the manifest's and the
  file's SHA-256 against the [identity](README.md#identity), and refuses the
  confirmation seed unless it is named. It replays each design twice, each
  time on a fresh fake provider, and a disagreement voids the run.
- The designs drive the fake device-memory provider in its address-only
  mode: the same rules and capacity as the ordinary fake, no bytes behind
  the backing. Provider calls are counted at the provider; with one kind of
  backing each is one driver call in the CUDA provider. io_uring buffer
  registrations are counted as the design would make them (below).
- The replay recomputes the reference (`swap_trace.py`'s LRU) and requires
  the trace's `evict` and `restore` records to be exactly its choices, so a
  design's extra victims follow the reference's recency order. After every
  tick it checks that held backing is within the budget less any
  outstanding shrink, equals what the fake has created and covers the
  resident groups; every 1,000 ticks and at every shrink probe it also
  checks each design's structures (placements disjoint, 4 KiB aligned and
  on held backing; counts) and that its resident set is within the
  reference's.
- [`replay_report.py`](replay_report.py) checks that the run is whole and
  agreed, then applies the criteria. Unit tests:
  `unit.Rb*` (`tests/unit/rb_replay_test.cc`) and
  `tools/tests/test_rb_replay_report.py`, both in `mise run check`.

## Designs as replayed

These are candidates for a measurement, not the memory manager. The timed
sessions must replay the same decisions: their call counts must equal the
[exact counts](#exact-counts).

Common to all:
- Every design evicts the reference's victims first. It evicts more, or
  relocates, only when it cannot otherwise place a restore or meet a
  shrink. Unless its hole policy chooses, an extra victim is the least
  recently used evictable group (resident, unleased, not wanted by the
  access) in the reference's recency order. An access's missing groups
  are placed in the trace's order. Refusal: the design has evicted every
  evictable group (and relocated, for compaction) and still cannot place.
- Backing is GPU-accessible host memory (D-034), in 2 MiB granules.

**D-033** (`d033`). Each model gets a virtual range on its first restore;
each group a 2 MiB-aligned region in it. A restore maps one 2 MiB handle
per chunk (`ceil(stored / 2 MiB)`), then sets access once over the group.
An eviction unmaps the group once; its handles go to the current access's
restores, and whatever that access does not take is released when it
ends. A new handle is created only when none is handed over and the budget
allows. No free pool; no io_uring registration (the runtime's provider
registers none, and unregistered buffers reach the device's bandwidth at
depth two or more, [storage-queue](../storage-queue/README.md)).

**Slabs** (`slab{32m,256m,1g}-{policy}`). One virtual arena of four times
the budget, cut into slab slots. A slab is created, mapped, given access and
registered with io_uring once, and stays so until it is released
(unregistered, unmapped, released). Groups are suballocated at 4 KiB
offsets (resources keep their 256-byte alignment inside); a group may span
adjacent held slabs. Empty slabs are kept until the budget needs their room
(for a slab elsewhere, or at a shrink probe), then released highest first.
General placement:
1. best fit in held space: the smallest free run that fits, lowest address
   first;
2. else growth: the placement needing the fewest new slabs, then the lowest
   address, releasing empty slabs outside it if the budget needs their room;
3. else the hole policy's step;
4. else one extra LRU victim, and try again.

A shrink probe releases empty slabs, then (compaction only) empties a slab
by relocation, then evicts the slab whose newest group is oldest (then
fewest bytes, lowest index), whose groups are all evictable; failing that,
one LRU victim at a time.

The hole policies:
- **`run`**, contiguous-run eviction: evict the window of the group's
  length, over held general space, whose newest group is oldest (then
  fewest bytes, lowest address) among windows whose groups are all
  evictable.
- **`class`**, size classes: every group of at most one slab is placed in a
  slab of its exact stored size's class (slots at multiples of that size);
  larger groups use general space and `run`. A class takes a free slot,
  else an empty slab, else a new slab, else evicts its own class's LRU
  evictable group, unless some slab (of any class, or general space)
  holds only older evictable groups: then it empties the oldest such
  slab, chosen as at a shrink, and takes it. When general space must
  grow and every other step fails, a design with class slabs empties the
  oldest slab rather than evicting LRU victims one at a time.
- **`hybrid`**: expert closures by size class as in `class`, dense groups in
  general space by `run`.
- **`compact`**, activity-sorted compaction at M2's scope: before `run`'s
  eviction, the window of free space and relocatable groups (dense,
  unleased, not wanted) with the fewest bytes to move (then lowest address)
  whose groups fit, best fit and largest first, in holes outside it; they
  move, and the restore that waited is a delayed admission. Experts never
  move (M7 completes that case), so this is the owner's proposal restricted
  to dense groups. The window is chosen by bytes to move, not by activity:
  only unleased dense groups may move, and they are kept, not evicted. An
  activity-guided choice (moving the recently used dense groups out of the
  coldest window and evicting the rest) was replayed only as a sensitivity
  check ([Limitations](#limitations)).

## Conditions

- Host: the x86-64 workstation (Intel Core i9-11900KF, kernel
  7.0.0-31-generic), `native` preset (RelWithDebInfo), SDK
  `x86_64-e0a0c85c42806fb1`, on base commit `14aaf08` with this change.
  The deterministic metrics do not depend on the host: the same sources
  built with the `spark-native` preset on `spark-b` (arm64) gave records
  identical to these in every field, digests included.
- Trace: primary seed 20260926, manifest `44f9f2b4…`, files as in the
  [identity](README.md#identity); budgets 64, 53 and 40 GiB.
- Sources as run (SHA-256): `designs.cc` `b173ba34…`, `replay.cc`
  `02d8bb63…`, `trace.cc` `b7610097…`, `main.cc` `e09ea292…`,
  `fake_device_memory.cc` `047b5968…`.
- `llmp_rb_replay --threads 12 --check-every 1000 TRACE_DIR`; every one
  of the 78 replays agreed with its twin. The raw output
  (`results.jsonl`, SHA-256 `9da54daa…`) is outside Git in
  `~/.local/share/llmp/retained-backing-20260926/replay-primary-2/`.
  An earlier run (`replay-primary-1/`, `designs.cc` `afb54a2b…`) had
  size classes evict their own class's newer groups before reclaiming an
  older slab of another class; its D-033, `run` and `compact` records are
  identical to these, digests included.

## Results

Waste is held backing minus the resident groups' stored bytes after each
tick. Content lost is extra restored bytes (beyond the reference's) plus
bytes evicted at shrink probes beyond its victims; a group evicted at a
probe and later restored counts in both, as the criteria read. "Eligible"
compares with D-033 at the same budget, `<=` with ties passing.

### 64 GiB (`trace-r5-4.jsonl`)

| Design | Waste mean GiB | Waste peak GiB | Content lost GiB | Extra restored GiB | Shrink extra GiB | Refusals | Eligible |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `d033` | 5.461 | 6.022 | 38.641 | 30.306 | 8.335 | 0 | baseline |
| `slab32m-run` | 2.434 | 4.469 | 23.048 | 15.966 | 7.082 | 0 | yes |
| `slab32m-class` | 3.062 | 5.032 | 16.039 | 11.288 | 4.750 | 0 | yes |
| `slab32m-hybrid` | 2.630 | 4.753 | 15.911 | 10.460 | 5.451 | 0 | yes |
| `slab32m-compact` | 2.411 | 4.642 | 22.762 | 15.710 | 7.052 | 0 | yes |
| `slab256m-run` | 2.423 | 4.705 | 24.834 | 16.287 | 8.547 | 0 | yes |
| `slab256m-class` | 4.834 | 7.455 | 39.991 | 31.822 | 8.169 | 0 | no: waste_peak, content_lost |
| `slab256m-hybrid` | 2.339 | 4.478 | 17.308 | 11.843 | 5.464 | 0 | yes |
| `slab256m-compact` | 2.386 | 4.836 | 24.697 | 15.958 | 8.739 | 0 | yes |
| `slab1g-run` | 2.639 | 4.681 | 32.967 | 20.238 | 12.729 | 0 | yes |
| `slab1g-class` | 12.900 | 20.905 | 87.170 | 76.824 | 10.346 | 0 | no: waste_mean, waste_peak, content_lost |
| `slab1g-hybrid` | 3.922 | 6.147 | 30.977 | 24.824 | 6.154 | 0 | no: waste_peak |
| `slab1g-compact` | 2.625 | 4.681 | 34.312 | 21.583 | 12.729 | 0 | yes |

Reported, not gated (calls per restored GiB; io_uring registrations and unregistrations; relocation):

| Design | Restored GiB | Create | Release | Map | Set access | Unmap | Reserve | Reg / unreg | Delayed admissions | Relocated GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `d033` | 156.99 | 283.01 | 77.55 | 566.16 | 375.68 | 222.74 | 8 | 0 / 0 | 0 | 0.000 |
| `slab32m-run` | 142.65 | 15.58 | 1.44 | 15.58 | 15.58 | 1.44 | 1 | 2222 / 206 | 0 | 0.000 |
| `slab32m-class` | 137.97 | 16.03 | 1.41 | 16.03 | 16.03 | 1.41 | 1 | 2211 / 195 | 0 | 0.000 |
| `slab32m-hybrid` | 137.14 | 16.11 | 1.41 | 16.11 | 16.11 | 1.41 | 1 | 2209 / 193 | 0 | 0.000 |
| `slab32m-compact` | 142.39 | 15.60 | 1.45 | 15.60 | 15.60 | 1.45 | 1 | 2222 / 206 | 3 | 0.234 |
| `slab256m-run` | 142.97 | 1.98 | 0.22 | 1.98 | 1.98 | 0.22 | 1 | 283 / 31 | 0 | 0.000 |
| `slab256m-class` | 158.50 | 1.82 | 0.23 | 1.82 | 1.82 | 0.23 | 1 | 289 / 37 | 0 | 0.000 |
| `slab256m-hybrid` | 138.52 | 2.01 | 0.19 | 2.01 | 2.01 | 0.19 | 1 | 279 / 27 | 0 | 0.000 |
| `slab256m-compact` | 142.64 | 1.98 | 0.21 | 1.98 | 1.98 | 0.21 | 1 | 282 / 30 | 4 | 0.466 |
| `slab1g-run` | 146.92 | 0.51 | 0.08 | 0.51 | 0.51 | 0.08 | 1 | 75 / 12 | 0 | 0.000 |
| `slab1g-class` | 203.51 | 0.37 | 0.06 | 0.37 | 0.37 | 0.06 | 1 | 75 / 12 | 0 | 0.000 |
| `slab1g-hybrid` | 151.51 | 0.47 | 0.05 | 0.47 | 0.47 | 0.05 | 1 | 71 / 8 | 0 | 0.000 |
| `slab1g-compact` | 148.26 | 0.51 | 0.08 | 0.51 | 0.51 | 0.08 | 1 | 75 / 12 | 5 | 0.243 |

### 53 GiB (`trace-r3-2.jsonl`)

| Design | Waste mean GiB | Waste peak GiB | Content lost GiB | Extra restored GiB | Shrink extra GiB | Refusals | Eligible |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `d033` | 4.490 | 5.157 | 34.691 | 25.777 | 8.914 | 0 | baseline |
| `slab32m-run` | 2.762 | 6.599 | 33.242 | 22.753 | 10.489 | 0 | no: waste_peak |
| `slab32m-class` | 3.804 | 12.561 | 32.739 | 25.731 | 7.008 | 0 | no: waste_peak |
| `slab32m-hybrid` | 3.483 | 12.433 | 31.031 | 24.599 | 6.432 | 0 | no: waste_peak |
| `slab32m-compact` | 2.762 | 6.657 | 32.592 | 22.329 | 10.264 | 0 | no: waste_peak |
| `slab256m-run` | 2.789 | 6.668 | 35.294 | 23.256 | 12.038 | 0 | no: waste_peak, content_lost |
| `slab256m-class` | 4.500 | 14.344 | 33.644 | 24.085 | 9.559 | 0 | no: waste_mean, waste_peak |
| `slab256m-hybrid` | 2.910 | 13.985 | 25.027 | 20.044 | 4.982 | 0 | no: waste_peak |
| `slab256m-compact` | 2.807 | 6.630 | 34.741 | 22.705 | 12.036 | 0 | no: waste_peak, content_lost |
| `slab1g-run` | 2.894 | 6.316 | 37.242 | 23.059 | 14.183 | 0 | no: waste_peak, content_lost |
| `slab1g-class` | 12.446 | 19.438 | 43.455 | 35.679 | 7.776 | 0 | no: waste_mean, waste_peak, content_lost |
| `slab1g-hybrid` | 3.883 | 13.291 | 28.250 | 22.814 | 5.436 | 0 | no: waste_peak |
| `slab1g-compact` | 2.898 | 6.385 | 33.326 | 19.836 | 13.490 | 0 | no: waste_peak |

Reported, not gated (calls per restored GiB; io_uring registrations and unregistrations; relocation):

| Design | Restored GiB | Create | Release | Map | Set access | Unmap | Reserve | Reg / unreg | Delayed admissions | Relocated GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `d033` | 223.65 | 189.29 | 70.25 | 565.67 | 385.02 | 287.41 | 8 | 0 / 0 | 0 | 0.000 |
| `slab32m-run` | 220.63 | 8.96 | 1.42 | 8.96 | 8.96 | 1.42 | 1 | 1977 / 313 | 0 | 0.000 |
| `slab32m-class` | 223.61 | 8.97 | 1.53 | 8.97 | 8.97 | 1.53 | 1 | 2006 / 342 | 0 | 0.000 |
| `slab32m-hybrid` | 222.47 | 8.87 | 1.39 | 8.87 | 8.87 | 1.39 | 1 | 1973 / 309 | 0 | 0.000 |
| `slab32m-compact` | 220.20 | 8.99 | 1.44 | 8.99 | 8.99 | 1.44 | 1 | 1980 / 316 | 12 | 0.725 |
| `slab256m-run` | 221.13 | 1.13 | 0.19 | 1.13 | 1.13 | 0.19 | 1 | 249 / 41 | 0 | 0.000 |
| `slab256m-class` | 221.96 | 1.15 | 0.21 | 1.15 | 1.15 | 0.21 | 1 | 255 / 47 | 0 | 0.000 |
| `slab256m-hybrid` | 217.92 | 1.11 | 0.16 | 1.11 | 1.11 | 0.16 | 1 | 242 / 34 | 0 | 0.000 |
| `slab256m-compact` | 220.58 | 1.13 | 0.19 | 1.13 | 1.13 | 0.19 | 1 | 249 / 41 | 16 | 1.909 |
| `slab1g-run` | 220.93 | 0.30 | 0.06 | 0.30 | 0.30 | 0.06 | 1 | 66 / 14 | 0 | 0.000 |
| `slab1g-class` | 233.55 | 0.28 | 0.06 | 0.28 | 0.28 | 0.06 | 1 | 65 / 13 | 0 | 0.000 |
| `slab1g-hybrid` | 220.69 | 0.27 | 0.04 | 0.27 | 0.27 | 0.04 | 1 | 60 / 8 | 0 | 0.000 |
| `slab1g-compact` | 217.71 | 0.30 | 0.06 | 0.30 | 0.30 | 0.06 | 1 | 66 / 14 | 9 | 0.708 |

### 40 GiB (`trace-r2-1.jsonl`)

| Design | Waste mean GiB | Waste peak GiB | Content lost GiB | Extra restored GiB | Shrink extra GiB | Refusals | Eligible |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `d033` | 3.457 | 3.981 | 13.968 | 2.538 | 11.430 | 0 | baseline |
| `slab32m-run` | 2.627 | 6.235 | 18.198 | 4.070 | 14.128 | 0 | no: waste_peak, content_lost |
| `slab32m-class` | 3.097 | 6.328 | 10.404 | 4.148 | 6.255 | 0 | no: waste_peak |
| `slab32m-hybrid` | 2.751 | 6.151 | 10.388 | 4.041 | 6.347 | 0 | no: waste_peak |
| `slab32m-compact` | 2.611 | 5.888 | 16.447 | 3.504 | 12.942 | 0 | no: waste_peak, content_lost |
| `slab256m-run` | 2.638 | 6.451 | 20.773 | 5.791 | 14.982 | 0 | no: waste_peak, content_lost |
| `slab256m-class` | 3.656 | 8.196 | 12.768 | 3.629 | 9.139 | 0 | no: waste_mean, waste_peak |
| `slab256m-hybrid` | 2.748 | 6.501 | 9.317 | 3.765 | 5.551 | 0 | no: waste_peak |
| `slab256m-compact` | 2.666 | 6.230 | 18.763 | 4.546 | 14.217 | 0 | no: waste_peak, content_lost |
| `slab1g-run` | 2.842 | 7.092 | 21.105 | 5.280 | 15.824 | 0 | no: waste_peak, content_lost |
| `slab1g-class` | 9.498 | 18.874 | 42.770 | 26.967 | 15.803 | 0 | no: waste_mean, waste_peak, content_lost |
| `slab1g-hybrid` | 4.006 | 8.600 | 19.867 | 10.975 | 8.892 | 0 | no: waste_mean, waste_peak, content_lost |
| `slab1g-compact` | 2.846 | 7.096 | 20.453 | 4.639 | 15.814 | 0 | no: waste_peak, content_lost |

Reported, not gated (calls per restored GiB; io_uring registrations and unregistrations; relocation):

| Design | Restored GiB | Create | Release | Map | Set access | Unmap | Reserve | Reg / unreg | Delayed admissions | Relocated GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `d033` | 242.50 | 172.10 | 89.76 | 564.30 | 384.09 | 308.15 | 8 | 0 / 0 | 0 | 0.000 |
| `slab32m-run` | 244.03 | 6.61 | 1.50 | 6.61 | 6.61 | 1.50 | 1 | 1614 / 366 | 0 | 0.000 |
| `slab32m-class` | 244.11 | 6.76 | 1.64 | 6.76 | 6.76 | 1.64 | 1 | 1649 / 401 | 0 | 0.000 |
| `slab32m-hybrid` | 244.00 | 6.69 | 1.58 | 6.69 | 6.69 | 1.58 | 1 | 1633 / 385 | 0 | 0.000 |
| `slab32m-compact` | 243.47 | 6.63 | 1.50 | 6.63 | 6.63 | 1.50 | 1 | 1614 / 366 | 15 | 1.005 |
| `slab256m-run` | 245.75 | 0.84 | 0.20 | 0.84 | 0.84 | 0.20 | 1 | 206 / 50 | 0 | 0.000 |
| `slab256m-class` | 243.59 | 0.87 | 0.23 | 0.87 | 0.87 | 0.23 | 1 | 211 / 55 | 0 | 0.000 |
| `slab256m-hybrid` | 243.73 | 0.82 | 0.18 | 0.82 | 0.82 | 0.18 | 1 | 200 / 44 | 0 | 0.000 |
| `slab256m-compact` | 244.51 | 0.84 | 0.20 | 0.84 | 0.84 | 0.20 | 1 | 206 / 50 | 16 | 1.478 |
| `slab1g-run` | 245.24 | 0.23 | 0.07 | 0.23 | 0.23 | 0.07 | 1 | 57 / 18 | 0 | 0.000 |
| `slab1g-class` | 266.93 | 0.22 | 0.07 | 0.22 | 0.22 | 0.07 | 1 | 58 / 19 | 0 | 0.000 |
| `slab1g-hybrid` | 250.94 | 0.20 | 0.05 | 0.20 | 0.20 | 0.05 | 1 | 51 / 12 | 0 | 0.000 |
| `slab1g-compact` | 244.60 | 0.23 | 0.07 | 0.23 | 0.23 | 0.07 | 1 | 56 / 17 | 10 | 0.319 |

### Exact counts

What a timed run's own calls must equal (backend-proof.md), per budget and design:

| Budget | Design | Create | Release | Map | Set access | Unmap | Reserve | Register | Unregister | Restored bytes | Relocated bytes |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | `d033` | 44429 | 12174 | 88880 | 58977 | 34967 | 8 | 0 | 0 | 168563814400 | 0 |
| 64 | `slab32m-run` | 2222 | 206 | 2222 | 2222 | 206 | 1 | 2222 | 206 | 153166319616 | 0 |
| 64 | `slab32m-class` | 2211 | 195 | 2211 | 2211 | 195 | 1 | 2211 | 195 | 148143886336 | 0 |
| 64 | `slab32m-hybrid` | 2209 | 193 | 2209 | 2209 | 193 | 1 | 2209 | 193 | 147254140928 | 0 |
| 64 | `slab32m-compact` | 2222 | 206 | 2222 | 2222 | 206 | 1 | 2222 | 206 | 152891891712 | 250826752 |
| 64 | `slab256m-run` | 283 | 31 | 283 | 283 | 31 | 1 | 283 | 31 | 153511170048 | 0 |
| 64 | `slab256m-class` | 289 | 37 | 289 | 289 | 37 | 1 | 289 | 37 | 170191908864 | 0 |
| 64 | `slab256m-hybrid` | 279 | 27 | 279 | 279 | 27 | 1 | 279 | 27 | 148739932160 | 0 |
| 64 | `slab256m-compact` | 282 | 30 | 282 | 282 | 30 | 1 | 282 | 30 | 153157574656 | 500641792 |
| 64 | `slab1g-run` | 75 | 12 | 75 | 75 | 12 | 1 | 75 | 12 | 157754101760 | 0 |
| 64 | `slab1g-class` | 75 | 12 | 75 | 75 | 12 | 1 | 75 | 12 | 218512482304 | 0 |
| 64 | `slab1g-hybrid` | 71 | 8 | 71 | 71 | 8 | 1 | 71 | 8 | 162677596160 | 0 |
| 64 | `slab1g-compact` | 75 | 12 | 75 | 75 | 12 | 1 | 75 | 12 | 159197745152 | 260722688 |
| 53 | `d033` | 42335 | 15711 | 126513 | 86110 | 64280 | 8 | 0 | 0 | 240143601664 | 0 |
| 53 | `slab32m-run` | 1977 | 313 | 1977 | 1977 | 313 | 1 | 1977 | 313 | 236896518144 | 0 |
| 53 | `slab32m-class` | 2006 | 342 | 2006 | 2006 | 342 | 1 | 2006 | 342 | 240094265344 | 0 |
| 53 | `slab32m-hybrid` | 1973 | 309 | 1973 | 1973 | 309 | 1 | 1973 | 309 | 238879195136 | 0 |
| 53 | `slab32m-compact` | 1980 | 316 | 1980 | 1980 | 316 | 1 | 1980 | 316 | 236441223168 | 778715136 |
| 53 | `slab256m-run` | 249 | 41 | 249 | 249 | 41 | 1 | 249 | 41 | 237436989440 | 0 |
| 53 | `slab256m-class` | 255 | 47 | 255 | 255 | 47 | 1 | 255 | 47 | 238326804480 | 0 |
| 53 | `slab256m-hybrid` | 242 | 34 | 242 | 242 | 34 | 1 | 242 | 34 | 233988567040 | 0 |
| 53 | `slab256m-compact` | 249 | 41 | 249 | 249 | 41 | 1 | 249 | 41 | 236845637632 | 2049945600 |
| 53 | `slab1g-run` | 66 | 14 | 66 | 66 | 14 | 1 | 66 | 14 | 237225283584 | 0 |
| 53 | `slab1g-class` | 65 | 13 | 65 | 65 | 13 | 1 | 65 | 13 | 250776231936 | 0 |
| 53 | `slab1g-hybrid` | 60 | 8 | 60 | 60 | 8 | 1 | 60 | 8 | 236961894400 | 0 |
| 53 | `slab1g-compact` | 66 | 14 | 66 | 66 | 14 | 1 | 66 | 14 | 233764847616 | 759971840 |
| 40 | `d033` | 41735 | 21767 | 136843 | 93141 | 74727 | 8 | 0 | 0 | 260383039488 | 0 |
| 40 | `slab32m-run` | 1614 | 366 | 1614 | 1614 | 366 | 1 | 1614 | 366 | 262028029952 | 0 |
| 40 | `slab32m-class` | 1649 | 401 | 1649 | 1649 | 401 | 1 | 1649 | 401 | 262112235520 | 0 |
| 40 | `slab32m-hybrid` | 1633 | 385 | 1633 | 1633 | 385 | 1 | 1633 | 385 | 261996716032 | 0 |
| 40 | `slab32m-compact` | 1614 | 366 | 1614 | 1614 | 366 | 1 | 1614 | 366 | 261420630016 | 1079136256 |
| 40 | `slab256m-run` | 206 | 50 | 206 | 206 | 50 | 1 | 206 | 50 | 263875973120 | 0 |
| 40 | `slab256m-class` | 211 | 55 | 211 | 211 | 55 | 1 | 211 | 55 | 261554692096 | 0 |
| 40 | `slab256m-hybrid` | 200 | 44 | 200 | 200 | 44 | 1 | 200 | 44 | 261701033984 | 0 |
| 40 | `slab256m-compact` | 206 | 50 | 206 | 206 | 50 | 1 | 206 | 50 | 262539313152 | 1587118080 |
| 40 | `slab1g-run` | 57 | 18 | 57 | 57 | 18 | 1 | 57 | 18 | 263327821824 | 0 |
| 40 | `slab1g-class` | 58 | 19 | 58 | 58 | 19 | 1 | 58 | 19 | 286613504000 | 0 |
| 40 | `slab1g-hybrid` | 51 | 12 | 51 | 51 | 12 | 1 | 51 | 12 | 269442043904 | 0 |
| 40 | `slab1g-compact` | 56 | 17 | 56 | 56 | 17 | 1 | 56 | 17 | 262639042560 | 342188032 |

### Eligibility for timed sessions

| Design | 64 GiB | 53 GiB | 40 GiB | Every budget |
| --- | --- | --- | --- | --- |
| `slab32m-run` | yes | **no** | **no** | **no** |
| `slab32m-class` | yes | **no** | **no** | **no** |
| `slab32m-hybrid` | yes | **no** | **no** | **no** |
| `slab32m-compact` | yes | **no** | **no** | **no** |
| `slab256m-run` | yes | **no** | **no** | **no** |
| `slab256m-class` | **no** | **no** | **no** | **no** |
| `slab256m-hybrid` | yes | **no** | **no** | **no** |
| `slab256m-compact` | yes | **no** | **no** | **no** |
| `slab1g-run` | yes | **no** | **no** | **no** |
| `slab1g-class` | **no** | **no** | **no** | **no** |
| `slab1g-hybrid` | **no** | **no** | **no** | **no** |
| `slab1g-compact` | yes | **no** | **no** | **no** |

## Readings

- **Where slabs win and lose.** At 64 GiB D-033's handle padding costs it:
  its mean waste (5.46 GiB) and the content it loses to padding beyond the
  reference (38.6 GiB) are above the nine eligible designs'. At 53 and
  40 GiB the reference holds less, D-033's padding shrinks with it, and
  peak waste decides: the `run` and `compact` designs' peaks are
  5.9–7.1 GiB, against D-033's 5.16 and 3.98, and the class-based
  designs' 6.2–19.4 GiB. At 40 GiB the `run` and `compact` designs also
  lose more content than D-033 (16.4–21.1 GiB against 14.0), most of it at
  shrink probes, which must return whole slabs; size classes and the
  hybrid at 32 and 256 MiB lose less (9.3–12.8 GiB).
- **Mean waste stays lower for `run` and `compact`** at every budget; the
  gate is on the peak as well. The peak is holes and slab tails, not
  retained empty slabs: a diagnostic build that also recorded the empty
  slabs at the peak tick (`slab32m-run`, `slab32m-compact` and
  `slab1g-run` at 53 and 40 GiB; same peaks as above) found 0–0.66 GiB of
  empty slabs in peaks of 5.9–7.1 GiB, five of the six while a shrink
  probe was outstanding.
- **Size classes keep partly filled slabs per class.** Classes are exact
  stored sizes, dense groups included, so each size holds its own slabs;
  their free slots serve no other size. At 53 GiB their peaks
  (12.4–19.4 GiB) are two to three times `run`'s. At 1 GiB each class's
  slab is a large share of the pool: `slab1g-class` restores 1.6 times
  the reference's bytes at 64 GiB.
- **Compaction moves little** (0.2–1.9 GiB, 3–16 delayed admissions per
  budget) because only dense groups may move at M2, and holes big enough
  for a dense group are rare; its results stay close to `run`'s.
- **Driver calls.** D-033 makes 170–280 creates and about 565 maps per
  restored GiB; slabs make at most 16.1 creates per restored GiB (32 MiB),
  and at most 0.5 at 1 GiB. These are reported, not gated; their cost is
  in the timed metrics.

## What this decides

- Timed sessions on `spark` are owed only where a design is eligible: the
  nine designs above at 64 GiB. Even if they pass there, no design can
  replace D-033 on this seed, since the amending rule needs every budget;
  such a design is reported for the owner. The confirmation seed stays
  unread: it is reserved for a winner, and there can be none.
- Any design that moves addresses would still owe BP-P5, BP-L2, BP-L6 and
  the alias and captured-pointer checks. Every slab design moves a group's
  address when it is restored into another slot, and `compact` also moves
  resident groups.

## Readings of the criteria

Where the criteria left a choice, the stricter reading was taken:
- A replay that stops at a refusal covers only a prefix of the trace, so
  its other metrics are not compared and it is ineligible at that budget.
  A shrink probe a design cannot meet counts as a refusal. (No replay
  refused.)
- Content lost counts a group evicted at a probe and later restored twice,
  as the two clauses read.
- Waste is sampled after each tick, when an access's handoff has ended;
  held backing, handoff included, is checked against the budget less the
  outstanding shrink before every creation and after every tick.
- Registrations are a model: a slab registers once for its lifetime, and
  D-033 registers nothing, as the runtime's provider does today. The
  timed sessions must make the same (un)registrations.

## Limitations

- These are the designs as specified above. Other parameters are other
  designs: the criteria name 12 alternatives, and the replay gave each
  one a fixed, deterministic rule.
- Two variants were replayed at 53 GiB only, as sensitivity checks, for
  `slab32m-run` and `slab1g-run`: releasing empty slabs at once (peaks
  6.667 and 6.316 GiB) and, at a shrink probe, emptying the slab with the
  fewest resident bytes first (5.968 and 6.343 GiB). Neither brings the
  peak under D-033's 5.157 GiB.
- Further sensitivity checks, from diagnostic builds of this harness,
  each replayed once on the primary seed (their sources and results are
  outside Git, in `sensitivity-1/` beside the replay output). None
  changes the verdict: every variant still fails at 40 GiB, on peak
  waste.
  - *The peak as one tick.* The criteria's peak is the largest value after
    any tick. At 53 GiB, `run` and `compact` at 32 and 256 MiB exceed
    D-033's peak on 1,275–3,041 of the 783,410 ticks, and their 99th
    percentiles (4.84–5.14 GiB) are below it. At 40 GiB every design
    exceeds D-033's peak on at least 109,839 ticks (14%), and every 99th
    percentile (5.18 GiB and up) is above it.
  - *Shrink victims.* If a design keeps the reference's victims at a
    shrink probe, and evicts them only when it needs their room (its
    resident set may then exceed the reference's), the peaks fall by at
    most 2.6 GiB, counting those groups as resident: 5.93–19.44 GiB at
    53 GiB and 5.75–18.84 at 40 (D-033's are unchanged).
  - *Rounded size classes.* Classes at powers of two, or at four steps
    per doubling, instead of exact stored sizes: mean and peak waste
    above D-033's at every budget (peaks 8.26–18.29 GiB).
  - *Activity-guided compaction.* In the window, dense groups more
    recently used than its newest expert move to holes outside it
    (hottest first, best fit) and the rest are evicted; the window whose
    newest victim is oldest wins, and a shrink probe does the same over a
    slab. It passes at 64 GiB at each slab size, as `compact` does, and
    fails peak waste at 53 GiB (6.57–7.38 GiB) and 40 GiB (5.78–6.84),
    and content lost at 40 GiB (17.7–21.2 GiB against 14.0).
  - *Model-affine placement.* A group goes first to a hole on its own
    model's slabs, then to new slabs, then anywhere by best fit, for
    `run` and `hybrid`: peaks 6.51–13.73 GiB at 53 GiB and 6.15–8.12 at
    40.
- The fake backend proves the decisions and counts, not their cost; timing
  is part (a)'s timed sessions and part (b).
- One seed. The trace's substitutions (captured routes, synthetic library
  models, whole-group tables) are listed in the [README](README.md#what-substitutes-for-what).
