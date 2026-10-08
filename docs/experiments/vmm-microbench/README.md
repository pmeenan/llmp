<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# M0 VMM microbench — 2026-09-21

This experiment isolates memory allocation, mapping/access, unmapping, and
physical-handle release on `spark` (GB10, driver 580.178.04). It does **not**
transfer from SSD or measure page-in latency. CUDA VMM is the driver mechanism
llmpalooza uses to control physical backing behind GPU virtual addresses; our
runtime still owns eviction, loading, and completion policy. It is not an
automatic fault-driven pager. The next I/O spike measures the transfer path.

## Method and reproducibility

Cross-built on the x86-64 workstation with the D-032 LLVM/LLD 22.1.8 and
CUDA Toolkit 13.4.2 (NVCC 13.4.92) extracted SDK and Spark sysroot from the
[toolchain smoke](../toolchain-smoke/README.md). Host code is C++23 with
`-O2 -Wall -Wextra -Werror`, `aarch64-linux-gnu -march=armv8-a`;
the CUDA cubin uses C++23, `-O3 -arch=sm_121`. Execution uses the driver API,
not libcudart, and disables PTX JIT. `cuDriverGetVersion` reports 13000;
this differs from the 13.4 headers (`CUDA_VERSION=13040`). No driver,
clock, power, system compiler, or security settings were changed.

The target is `spark-c4e2`, Ubuntu 24.04.5 / DGX OS 7.6.0 OTA,
kernel `7.0.0-1019-nvidia`, 48 SMs. No other compute application was reported
before the experiment. Clocks were not locked, CPUs were not pinned, and
system background activity was not disabled. A mid-run observation showed
40 C and 2535 MHz SM clock; this is a snapshot, not a thermal trace.
Three consecutive processes ran on 2026-09-21; the aggregate results are
recorded below, with source/build identities in `artifacts.json`. Raw outputs
stay in external scratch and are not versioned. SDK provenance remains in
the smoke's manifest. Binaries and downloaded SDKs stay outside the repository.

Reproduce from the repository root with the existing pinned scratch SDK:

```bash
export LD_LIBRARY_PATH=/tmp/llmp-clang22/sdk-amd64/usr/lib/x86_64-linux-gnu
export CXX=/tmp/llmp-clang22/sdk-amd64/usr/bin/clang++-22
export LLD=/tmp/llmp-clang22/sdk-amd64/usr/bin/ld.lld-22
export SYSROOT=/tmp/llmp-toolchain-smoke/sysroot
export CUDA_ROOT=/tmp/llmp-toolchain-smoke/sdk-134/usr/local/cuda-13.4
bash docs/experiments/vmm-microbench/build.sh /tmp/llmp-vmm/build
ssh spark 'mkdir -p /tmp/llmp-vmm/build'
rsync -a /tmp/llmp-vmm/build/ spark:/tmp/llmp-vmm/build/
ssh spark 'for run in 1 2 3; do CUDA_DISABLE_PTX_JIT=1 /tmp/llmp-vmm/build/vmm-microbench /tmp/llmp-vmm/build/kernel.cubin > /tmp/llmp-vmm/final-run-$run.csv || exit; done'
rsync -a 'spark:/tmp/llmp-vmm/final-run-*.csv' /tmp/llmp-vmm/
python3 docs/experiments/vmm-microbench/summarize.py /tmp/llmp-vmm/final-run-*.csv
```

Each process queries minimum and recommended granularity for a device-local
`CU_MEM_ALLOCATION_TYPE_PINNED` allocation with no export handle. It sweeps
2, 8, 32, and 128 MiB (rounded to the queried minimum, deduplicated). These
are independent physical handles, each mapped in full; this does not prove
arbitrary subrange unmapping of one large allocation.

For each size/mode, discard five warm-up iterations and retain every subsequent
sample: 100 idle iterations, 30 with one background block, and 30 with 48
background blocks. Each iteration separately times `cuMemCreate`, `cuMemMap`,
`cuMemSetAccess`, `cuMemUnmap`, and `cuMemRelease` with host steady-clock
wall time. Address reservation is outside the timed loop. No timer-overhead
subtraction or outlier trimming is applied. Every allocation is filled before
unmap, and that use completes before backing changes. Create/release samples
use fresh API handles, but the warmed driver/OS may internally recycle pages;
these are not cold-boot allocation costs. Printed output is outside timers.

Before **each** concurrent operation, launch a fresh independent kernel on a
nonblocking stream. It performs register arithmetic for approximately 10 ms
by GPU global timer and writes a small output buffer. A system-scope atomic
start handshake from every block must arrive, and the completion event must
still be pending, before timing the host call. Record whether the event is
pending immediately after the call; then wait for it before advancing.
48 blocks equals the reported SM count, but does **not** establish one block
per SM, full occupancy, or a saturated GPU. Event durations are recorded;
this duration-controlled kernel cannot measure compute-throughput loss.

The pool experiment reserves 1 GiB of virtual addresses and creates sixteen
64 MiB physical handles. It fills and verifies every 32-bit word with a
handle-specific pattern, unmaps all handles while retaining them, then remaps
and verifies every word again. A deliberate one-word corruption must produce
exactly one mismatch, followed by zero after restoration. The error counter
reset and verification kernels are ordered in the same stream. At each phase,
a context synchronization and 250 ms settling interval precede snapshots of
`cuMemGetInfo` and `/proc/meminfo` MemAvailable. These are system-level
observations with background noise, not exact per-allocation accounting.
No memory-pressure/OOM experiment is performed; peak probe-owned physical
memory is approximately 1 GiB plus context/module/test buffers.

## Results

Both minimum and recommended granularity were **2,097,152 bytes (2 MiB)**.
All three processes passed full-content verification before and after remap,
the verifier negative control, and complete sample validation.

Host-call latency below is the **range of per-run medians**, in microseconds
(three runs; not a confidence interval). The analysis includes all samples,
including tails. `summarize.py` reports each run's median, nearest-rank p95,
maximum, event duration, and pending count.

| Background load | Extent MiB | Create µs | Map µs | Set access µs | Unmap µs | Release µs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Idle | 2 | 48.72–53.41 | 0.50–0.54 | 35.94–37.70 | 46.66–63.21 | 26.61–27.04 |
| Idle | 8 | 178.57–196.75 | 0.72–1.26 | 72.10–81.10 | 116.77–156.06 | 43.62–46.66 |
| Idle | 32 | 797.66–925.62 | 4.40–5.06 | 251.44–261.12 | 321.18–327.59 | 134.56–142.52 |
| Idle | 128 | 3384.05–3811.23 | 5.34–5.63 | 804.44–835.35 | 985.80–1010.94 | 449.96–472.11 |
| 1 block | 2 | 49.43–53.90 | 0.61–0.64 | 36.34–38.62 | 45.74–63.41 | 30.67–31.17 |
| 1 block | 8 | 178.94–198.19 | 0.74–1.29 | 72.42–82.10 | 118.36–155.80 | 55.67–58.33 |
| 1 block | 32 | 803.75–928.13 | 3.90–4.74 | 248.53–258.09 | 331.97–335.93 | 178.74–185.98 |
| 1 block | 128 | 3385.79–3808.15 | 4.75–4.94 | 799.34–818.63 | 1030.30–1042.61 | 617.56–642.56 |
| 48 blocks | 2 | 49.40–53.70 | 0.59–0.64 | 36.24–38.44 | 45.93–63.24 | 30.77–30.86 |
| 48 blocks | 8 | 179.17–198.60 | 0.75–1.34 | 72.41–82.66 | 118.76–155.54 | 55.15–59.09 |
| 48 blocks | 32 | 798.55–927.59 | 4.06–4.82 | 249.85–258.13 | 332.61–336.41 | 179.13–185.57 |
| 48 blocks | 128 | 3383.46–3814.23 | 4.82–5.02 | 799.91–819.28 | 1031.78–1041.77 | 621.45–650.52 |

`cuMemMap` alone understates the work required to make a mapping usable:
`cuMemSetAccess` is much more expensive here. At 2 MiB, idle p95 ranges were
0.56–0.62 µs for map, 37.34–44.35 µs for access, and 56.45–87.94 µs for
unmap. Worst observed 2 MiB unmap was 253.57 µs (one-block mode, run 1).
These small samples do not establish production tail bounds.

All **3,600 measured concurrent calls** returned while the background
completion event was still pending. Their per-group median kernel event
durations were 10.004–10.011 ms. This did not reveal a wait for whole-kernel
completion in this test, but does not rule out shorter GPU stalls. Large
allocation release was slower with background work: at 128 MiB its median
was 450–472 µs idle versus 618–651 µs with background kernels.

Pool observations, relative to the pre-reservation baseline, from
`cuMemGetInfo` (range across runs):

| Phase | Change in free memory, MiB |
| --- | ---: |
| Reserve 1 GiB virtual address range | 0.00 |
| Create sixteen 64 MiB handles | −1034.00 to −1033.92 |
| Map and fill every word | −1034.74 to −1034.00 |
| Unmap, keep physical handles | −1034.74 to −1034.00 |
| Remap and verify every word | −1034.74 to −1034.00 |
| Unmap and release all handles | +3.99 to +5.47 |

MemAvailable changed by essentially the same amounts (within 0.04 MiB of
the CUDA deltas). The approximately 10 MiB excess over the requested 1 GiB
and the few-MiB baseline drift are not isolated allocator overhead estimates.
The strong signal is that unmapping returned none of that approximately
1 GiB, whereas releasing the handles did. The virtual address reservation
itself did not consume a comparable physical allocation.

## Interpretation and limits

Use 2 MiB as the initial independently reclaimable extent on this measured
provider (D-033), and query granularity at runtime. Bigger I/O reads and
scheduling batches can span several extents. This is a starting policy,
**not an SSD or end-to-end optimum**. Actual expert sizes, packing waste,
I/O amplification, driver-call batching, and model traces remain unmeasured.

Unmapping separates an address from physical backing; it does not return the
backing while a physical handle remains held. A retained handle can be reused,
with bytes intact, after remapping and restoring access. Destroying the last
handle after unmapping returned approximately the allocated capacity here.
A reusable free pool therefore consumes the same physical budget as useful
resident contents. Start without a standing cache of unused handles, but
allow completion-safe handoff of compatible backing to a waiting admitted
load. Do not evict useful contents to prefill a pool. D-033 specifies this
policy and its reopening triggers.

The concurrency result is scoped to this driver, context, and independent
register-heavy kernel. It does not guarantee nonblocking behavior, bounded
latency, or no GPU stalls for all workloads. No memory-bandwidth load,
allocator contention, multiple host threads/contexts, memory pressure,
CUDA graph replay, external registrations, peer mappings, host-accessible
VMM allocations, SSD I/O, or model kernels were tested. `spark-b` and the
workstation GPU were not tested. No unsupported-address access is attempted;
accessing absent backing remains a runtime bug.

Sources for API semantics:
[NVIDIA VMM guide](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/virtual-memory-management.html),
[driver API reference](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__VA.html).
Spark's separate storage constraint is documented in the
[GDS release notes](https://docs.nvidia.com/gpudirect-storage/release-notes/index.html):
compatibility mode only. The following I/O experiment will also test GPU
in-place access to system-allocated memory populated from NVMe.

## Handoff

Builder: cross-build passed on the workstation; three complete executions
passed on `spark` with PTX JIT disabled. The retained analyzer validated all
9,600 timing samples and pool/negative-control completion markers. Shell
syntax and whitespace checks passed. An independent source review found an
unordered verifier-counter reset; it was fixed to use the verifier stream,
and a negative control was added **before all three retained runs**. No
application code, SDK installation, or driver changes were made. Remaining
scope limitations are listed above; the working tree is uncommitted.

Final independent review and adversarial challenge: clean after the verifier
ordering fix. The reviewer independently cross-built the final sources;
both rebuilt binary hashes matched the manifest. A separate full execution
on `spark`, with PTX JIT disabled, passed all 3,200 timing samples, full-content
remap verification, and the deliberate-corruption negative control. All
1,200 concurrent calls in that run also returned with the event pending.
Reviewer outputs remain separate in `/tmp/llmp-vmm-review` on the workstation
and Spark; they are not included in the three-run reported measurements.

The reviewer checked every manifest file hash and all 60 reported median
ranges against raw data, plus the p95/max, concurrent counts, and memory
deltas. The analyzer rejected nine malformed/incomplete fixtures (missing
completion, sample, negative-control marker, pool marker, or memory phase;
duplicate sample; nonfinite or negative timing; invalid pending flag).
The executable rejected a missing argument and missing cubin with nonzero
status on Spark. Shell syntax and whitespace checks passed. Review covered
buffer bounds, background-kernel handshake, stream ordering, and backing
lifetimes; no remaining actionable findings. The untested workload and
platform limits above still apply.
