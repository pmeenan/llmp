<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# VMM call costs, batching and the lazy handoff — 2026-10-07

Why do swaps and cold state growth pay for CUDA VMM calls, and which of
them can be removed? [main.cc](main.cc) is a standalone probe of D-033's 2 MiB
extents on a GB10; the swap measurements use the production runtime's
`swap-table` through `benchmarks/swap_lazy.cc`, which only chooses
`ServingOptions::lazy_handoff`.

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

## What follows

The 2 MiB extent stays (D-033, amended): larger handles would cut only the
per-handle part of access and unmap, which the lazy handoff already moves
off the critical path, at the cost of coarser reclamation. Cold state growth
still pays cuMemCreate on the critical path; a reserve of handles created off
it, and filling new state on the device instead of reading a sparse file's
hole, are the next pager items. A swap still evicts the outgoing model whole;
evicting only what the incoming model needs would let a return skip
re-reading what stayed.

Raw outputs stay outside Git under `spark-b` `~/.local/share/jitllm/vmm-batching1`
and `~/scratch/pager-overlap1`.
