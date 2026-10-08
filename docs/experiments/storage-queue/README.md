<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Storage queue depth and request size through llmpalooza's providers — 2026-09-25

M2's providers item asks for storage queue depths and run sizes to be
measured; M3 and M6 tune them against spill write-back. This experiment measures
the runtime's own paging path on `spark`:

- `UringStorage`, the raw io_uring ring with one submission per request;
- reading from an `O_DIRECT` regular file;
- straight into GPU-accessible host backing from the CUDA provider
  (`VmmProvider`, host-NUMA VMM mapped at its 2 MiB granularity, D-034).

M0 measured the same path with its own probe, which used registered buffers
([io-path](../io-path/README.md)). The runtime's provider does not register
buffers.

The harness is
[`benchmarks/storage_bench.cc`](../../../benchmarks/storage_bench.cc).

## Method

The harness first writes an unnamed 8 GiB file (`O_TMPFILE`, gone when the
process exits) with direct I/O on the Spark's NVMe root filesystem. It then
reads the file sequentially once per setting:

- queue depths of 1, 2, 4 and 8 requests in flight;
- requests of 2, 4 and 8 MiB, a run of one, two or four 2 MiB chunks;
- each request into its own slot of one 64 MiB host-VMM buffer.

The recorded latency runs from the return of `Submit` to harvest: it
excludes submission overhead and any work completed inside that call.
The harvesting thread waits in `io_uring_enter`, so it sleeps between
completions and does not poll. The current harness starts the timer before
`Submit`; the historical latency values below have not been remeasured.

- **Host:** `spark-c4e2`, GB10, DGX OS 7.6.0, kernel 7.0.0-1019-nvidia,
  driver 580.178.04.
- **SSD:** a Samsung `MZALC4T0HBL1-00B07` (firmware `NXHB202Q`) holding the
  ext4 root. The block layer splits requests at 128 KiB, and the scheduler
  is `none`.
- **Build:** the `cross` preset with the pinned SDK
  `x86_64-f469d317c88c3044`.
- **Conditions:** load average 0.01–0.87 around the runs (the lanes
  benchmark ran just before). No settings were
  changed, and clocks were not locked. Three consecutive processes ran;
  the tables give the range across them.

Source identities (SHA-256):

| File | SHA-256 |
| --- | --- |
| `benchmarks/storage_bench.cc` | `2abeecbc02609e1ce743e5213ed22410b11ad023125e7645bf641b59b29c9e96` |
| `src/providers/uring_storage.cc` | `bc6457f33d7887fa3eddf558e0f8d1c9b13635ce24bce32511bc83151839b192` |
| `src/platform/io_uring.cc` | `b73246c81d8c0c19c04e612085f085ad8ab4174d14b93a4130583848da57f02d` |
| `src/providers/cuda/cuda_device_memory.cc` | `8c09b9526c2f9c7e3411b47b64afc8ee5794fd19eacfb5542e02cbbe802196c9` |
| `src/providers/device_memory.cc` | `78c2a441694a570706190aafc39abd6f1b64f7d126c5d8d7b35e5ac36a121fdd` |

These are the sources measured. Later review fixes cover cancellation
accounting and uncertain memory-provider outcomes, which these successful
runs did not exercise. The current harness also includes submission in its
latency interval and rejects failed reads instead of reporting their
requested bytes as throughput. The tables preserve the original successful
runs and their post-submit latency interval.

## Results

| Depth | Request | In flight | GB/s | p50 µs | p99 µs |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2 MiB | 2 MiB | 10.120–10.519 | 164–172 | 193–383 |
| 2 | 2 MiB | 4 MiB | 13.278–13.415 | 283–285 | 389–416 |
| 4 | 2 MiB | 8 MiB | 14.911–14.933 | 520–533 | 633–676 |
| 8 | 2 MiB | 16 MiB | 14.948–14.964 | 1071–1090 | 1198–1326 |
| 1 | 4 MiB | 4 MiB | 11.795–11.879 | 301 | 336–348 |
| 2 | 4 MiB | 8 MiB | 14.841–14.919 | 468–508 | 618–693 |
| 4 | 4 MiB | 16 MiB | 14.951–14.968 | 1058–1063 | 1213–1282 |
| 8 | 4 MiB | 32 MiB | 14.962–14.965 | 2174–2176 | 2416–2471 |
| 1 | 8 MiB | 8 MiB | 11.873–12.612 | 527–558 | 597–650 |
| 2 | 8 MiB | 16 MiB | 14.922–14.956 | 916–1006 | 1106–1181 |
| 4 | 8 MiB | 32 MiB | 14.962–14.971 | 1993–2110 | 2201–2350 |
| 8 | 8 MiB | 64 MiB | 14.946–14.961 | 4234–4350 | 4563–4675 |

## Conclusions

1. **The providers reach the device's bandwidth without registered
   buffers.** About 8 MiB in flight gives 14.84–14.93 GB/s, as either four
   2 MiB requests or two 4 MiB. M0's registered-buffer probe measured
   14.903–14.968 GB/s at four 2 MiB requests. Unregistered buffers cost
   bandwidth only at depth one (10.12–10.52 GB/s here, against M0's
   11.58–11.61).
2. **At least two concurrent requests and about 8 MiB in flight saturate
   this device.** A single 8 MiB request reaches only 11.87–12.61 GB/s;
   bytes in flight alone do not explain the result. With concurrent
   requests, beyond about 8 MiB in flight, bandwidth stays flat while
   latency grows in proportion.
   At the same 8 MiB in flight, a request completes in about the same time
   whatever its size (468–558 µs p50). Larger requests therefore buy
   nothing, and 2 MiB requests let each chunk be used as soon as it lands.
3. **Starting settings for the storage lane:**
   - bulk page-in: 2 MiB requests with 8 MiB in flight;
   - latency-first: a single 2 MiB dependency at depth one or two
     (164–172 µs, or 283–285 µs p50).

   These are starting points, not tuned values. M3 and M6 set them against spill
   write-back and concurrent GPU work, and nothing here changes D-033's
   2 MiB extents.

## Limitations

- **Workload:** sequential reads of one file, one ring, no competing I/O,
  no GPU work, and no writes.
- **Reader and completions:** the whole-read logic (`DirectReader`) and the
  completion board are not in this path. The board is measured in
  [task-lanes](../task-lanes/README.md); `DirectReader`'s cost is unmeasured.
- **Scope:** three runs on one SSD, which the M0 runs also used.

## Reproduce

Build and deploy the cross build:

```bash
tools/build deploy --host spark cross
```

Then run it on the Spark, with a directory on the filesystem to measure (the
file it writes is unnamed and disappears on exit):

```bash
ssh spark 'mkdir -p ~/.cache/llmp/bench && ~/.cache/llmp/deploy/cross-<id>/benchmarks/llmp_storage_bench ~/.cache/llmp/bench 8'
```
