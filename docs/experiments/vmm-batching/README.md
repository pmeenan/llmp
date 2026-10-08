<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# VMM call costs, batching, the lazy handoff and the handle reserve — 2026-10-07

Why do swaps and cold state growth pay for CUDA VMM calls, and which of
them can be removed? [main.cc](main.cc) is a standalone probe of D-033's 2 MiB
extents on a GB10; the swap measurements use the production runtime's
`swap-table` through `benchmarks/swap_pager.cc`, which only chooses the
pager's serving options (`ServingOptions::lazy_handoff`, and since the
reserve slice `zero_state` and `handle_reserve`).

## Probe

`spark-b`, NVIDIA GB10, driver API 13000 (driver 580.178.04), pinned SDK
`aarch64-c09daba6ac31edee` (clang++ and CUDA 13.4 headers, linked against the
driver stub), binary `6b980b41…`, source `1ee321a6…`; seven rounds a case,
medians, two runs (the second's figures in parentheses where they differ by
more than a few percent). Per 2 MiB extent, µs:

| Adjacent extents | create | map | release | access, one call each | access, one call | unmap, one call each | unmap, one call |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 53.8 | 0.5 | 28.4 | 37.6 | 35.7 | 63.4 | 37.7 |
| 16 | 71.9 | 0.9 | 36.4 | 38.0 | 31.5 | 37.5 | 35.3 |
| 256 | 78.3 | 0.6 | 35.4 | 34.5 | 32.2 | 33.8 | 32.8 |
| 1,024 | 78.5 | 0.5 | 34.9 | 34.0 | 33.1 | 34.0 | 34.0 |

- One cuMemSetAccess or cuMemUnmap over a range of separately created
  mappings is accepted, but costs the same per extent: **batching buys
  nothing**.
- One handle of N extents' size: create 54.7 / 60.0 / 70.4 / 70.6 µs a
  2 MiB at 1 / 8 / 64 / 256 extents (per byte); access 40.1 / 20.4 / 13.2 /
  12.5 µs; unmap 62.9 / 25.7 / 15.6 / 14.7 µs. Access and unmap carry about
  20 µs per handle; create does not shrink.
- cuMemCreate from 1 / 2 / 4 / 8 threads: 79.8 / 76.8 / 86.7 / 97.0 µs of
  wall a 2 MiB (99.7 / 88.3 / 89.4 / 101.7): the driver serializes it.
- Zero fill of a mapped 2 MiB: cuMemsetD8Async 7.7 µs against a pinned
  host-to-device copy 41 µs.
- Host-to-device copy bandwidth (2 MiB, synchronized each) is 50.1 GB/s
  alone and 49.6 / 43.7 / 43.4 GB/s beside a thread looping unmap+map /
  unmap+map+access / create+map+access+unmap+release: VMM calls do not
  starve the zone's copies.

## Swaps

Production runtime, `spark-b`, the [final M3 table](../m3-final-swap/README.md)'s
configuration (DeepSeek 0731 + DSpark, Qwen3.8 NVFP4 + MTP), pair
`deepseek:qwen3.8` only, 8,192 saved context tokens, 16 continued, two
cycles plus a zero-context pair, handoff on; one binary, fresh data
directories, lazy off / on / on / off. Every row's state and continuation
are exact and every output digest matches across arms.

| Swap | Off: evict | Off: total | On: evict | On: total |
| --- | ---: | ---: | ---: | ---: |
| DeepSeek → Qwen, first use (8K) | 1.841 / 1.848 | 8.010 / 8.025 | 0.089 / 0.097 | 6.258 / 6.268 |
| Qwen → DeepSeek, first use (8K) | 1.314 / 1.321 | 9.534 / 9.560 | 0.039 / 0.038 | 8.257 / 8.258 |
| DeepSeek → Qwen, prepared (8K) | 1.855 / 1.922 | 7.892 / 7.967 | 0.090 / 0.145 | 6.102 / 6.166 |
| Qwen → DeepSeek, prepared (8K) | 1.364 / 1.369 | 9.575 / 9.592 | 0.037 / 0.068 | 8.244 / 8.282 |
| DeepSeek → Qwen, prepared (0) | 1.927 / 1.938 | 7.969 / 7.993 | 0.102 / 0.105 | 6.113 / 6.136 |
| Qwen → DeepSeek, prepared (0) | 1.358 / 1.376 | 9.663 / 9.711 | 0.066 / 0.062 | 8.376 / 8.396 |

Seconds. Page-in stays 5.77–5.80 s (76.8 GB) and 8.00–8.10 s (97 GB), the
SSD's rate, in both arms: each load's unmap of its donor's old place (about
34 µs) runs before its read, beside the reads of the loads ahead of it
(about 150 µs a 2 MiB). 36,669 extents were handed
off in every swap.

The full image-inclusive table (all six ordered pairs, 32 swaps; the final M3
table's configuration and inputs, lazy handoff by default) passes with no
problems: every LLM state and continuation exact, the image's pixels equal
to the expected digest. The worst LLM-to-LLM swap is 8.532 s (M3 final:
9.853 s); every swap into the image takes 3.70–3.89 s (M3: 5.03–5.78 s) and
every return from it 5.95–8.46 s (M3: 6.48–8.93 s). Some DeepSeek output
digests differ from the M3 table's, identically with lazy handoff off in the
same build: changes since 2026-10-04, not this one.

An earlier attempt overlapped evictions with page-ins at the swap program's
level instead (lockstep rounds of loads and evictions, the unmaps still at
eviction). It measured no gain (7.92–7.94 / 9.61–9.68 s): each round queued
its unmaps on the VMM lane ahead of the maps its loads requested as they
entered the landing window, so reads stalled behind them. It was not kept.

## Handle reserve and device zero-fill

Cold state growth paid a cuMemCreate (55–80 µs) and a read of the sparse
spill file's hole through the landing zone plus its copy for every fresh
2 MiB of conversation state. Two changes take both off the growth path
(D-033 amended again):

- **Handle reserve.** The VMM lane keeps 32 device handles (64 MiB) created
  while it has no command; a map of that class and size takes one before
  creating. A plain unmap's backing, or kept backing released, refills a
  short reserve instead of being released. The node charges all 32 as one
  pinned runtime extent before anything pages, so the reserve never holds
  backing the catalog does not count. The runtime keeps it; harnesses keep
  none unless they ask (`NodeSettings::handle_reserve`).
- **Device zero-fill.** A state's page source is marked `zero`: until
  something is written back, its contents are zeros, so a load maps its
  backing and zeroes it with one `cuMemsetD8Async` on the zone's copy stream
  (`DeviceExecution::Zero`), with no read and no landing slot. A spill
  file kept across a restart (D-105) is read as before: its adopted
  extents hold what the process before wrote, though nothing is written
  back yet.

Gemma3 4B QAT trained maximum (131,072 positions, 128-row chunks, one slot),
`spark`, one binary, fresh data directories; growth timed around every
`ReserveStateThrough` call (1,088 a traversal) by a temporary timer that is
not kept. Every arm's heads, choices and final head match the reference
digests.

| Arm | State growth, s | Prefill, s |
| --- | ---: | ---: |
| Neither (control) | 0.476 / 0.459 | 31.640 / 31.422 / 31.370 / 31.590 |
| Zero-fill only | 0.411 | 31.530 / 31.427 |
| Reserve only | 0.433 | 31.874 / 31.789 |
| Both (default) | 0.286 / 0.294 | 31.350 / 31.480 / 31.387 / 31.348 |
| Stock llama.cpp (bookends) | | 31.690 / 31.535 |

Arms ran in the order default, control, zero-fill, reserve, control,
default, twice (stock only around the first). Growth falls by 0.18 s (38%)
with both, more than either alone; prefill means 31.39 s (default) against
31.51 s (control) and 31.61 s (stock), the earlier +1.31% to stock having
been measured while the host copied checkpoints to the NAS. The reserve-only
arm's prefill was the slowest in both runs, in the same fourth place each
time; it is not a shipped configuration and was not investigated further. The
remaining 0.27 ms a growth call is mostly cuMemSetAccess (about 33 µs an
extent) and the scheduler's round trips; it is serial with the device
because growth runs before each chunk is dispatched.

Swaps: the same DeepSeek ↔ Qwen3.8 configuration as above, `spark-b`, arms
on / plain / plain / on twice (`plain` keeps lazy handoff but turns both
changes off). Every row of every arm is exact with the same output digests.
Swap times match within noise (DeepSeek → Qwen 6.07–6.29 s, Qwen → DeepSeek
8.25–8.38 s in both arms): page-in stays disk-bound, and a swap drains the
reserve in its first milliseconds. The first arm of the first run (an
`on` arm) was slow throughout (page-in 8.6 s, evictions 8.9 s), with about
790 driver out-of-memory messages in the kernel log during that run only and
host planning three times slower; the repeat, with the log checked between
arms, had none in any arm, and a later `plain` arm showed the same kind of
page-in blip. It is recorded as host memory pressure outside the runtime.

## What follows

The 2 MiB extent stays (D-033, amended): larger handles would cut only the
per-handle part of access and unmap, which the lazy handoff already moves
off the critical path, at the cost of coarser reclamation. Cold state growth
still runs before each chunk; growing the next chunk's state beside the
current chunk's device run would hide the rest. A swap still evicts the
outgoing model whole; evicting only what the incoming model needs would let
a return skip re-reading what stayed.

Raw outputs stay outside Git under `spark-b` `~/.local/share/jitllm/vmm-batching1`,
`~/scratch/pager-overlap1` and `~/scratch/pager-reserve1`, and `spark`
`~/.local/share/jitllm/g3gap-pager1`.
