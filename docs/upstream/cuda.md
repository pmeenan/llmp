<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# NVIDIA: CUDA, cuBLAS, cuFile, the GB10 and DGX OS

- **jitLLM's pins:** CUDA toolkit 13.4.2 (NVCC 13.4.92) and cuBLAS 13.8.0.4
  in the SDK (D-076, [toolchains/](../../toolchains/README.md)). On
  2026-09-29 NVIDIA's apt repository tops out at those versions, so there is
  nothing newer to move to.
- **The Sparks:** DGX Spark (GB10, `sm_121`), DGX OS 7.6.0, kernel
  7.0.0-1019-nvidia, driver 580.178.04 (inventory in
  [environment.md](../environment.md)). Whether a newer driver exists was
  not checked.
- **Where to report:** the NVIDIA Developer Forums (the DGX Spark / GB10
  category) for platform behaviour and questions; NVIDIA's developer bug
  portal for driver and library bugs (it needs an NVIDIA developer
  account); [NVIDIA/cccl](https://github.com/NVIDIA/cccl) issues for CUB and
  Thrust. None of these reports has been filed yet.

## Bounded Nsight capture completed before its launched model (RE-039)

- **Status:** worked around; no isolated upstream defect claimed.
- **Found:** 2026-09-30, `spark`, Nsight Systems 2025.3.2, driver
  580.178.04 and CUDA 13.4.92.
- **Problem:** `nsys profile --trace=cuda --sample=none --cpuctxsw=none
  --delay=2700 --duration=60 --kill=none --wait=all --export=sqlite`
  finished its trace/export with exit zero while its launched native
  model remained on the GPU, reparented to PID 1 with a separate process
  group. No complete native JSON or application exit status was retained
  after the profiler exited. The surrounding `spark-job`
  supervised the profiler, not that surviving process.
- **jitLLM's workaround:** separate the long validation command from
  bounded profiling, and keep the application's lifetime/output explicitly
  supervised. `spark-job busy` plus all-query memory/process probes caught
  the surviving model before a second model could load. The unfinished
  maximum-state result is excluded; only the completed bounded trace is
  retained as a diagnostic.
- **Proposed action:** before reporting an upstream bug, isolate this
  option combination with a cheap target and confirm documented wait and
  output behavior. No reproduction run or report has been made.
- **Links:** [RE-039](../rough-edges.md#re-039-a-bounded-nsight-cli-capture-ended-before-its-profiled-model-leaving-the-target-outside-the-jobs-supervision--2026-09-30-status-worked-around);
  `spark:~/scratch/m3-extrapolation-ds/final-max-swap/`.

## A stream holds about 1,020 pending operations, then a launch blocks its thread (RE-029)

- **Status:** open.
- **Found:** 2026-09-27/28, `spark` and `spark-b`, driver 580.178.04, driver
  API 13000, CUDA 13.4.92 headers.
- **Problem:** behind a `cuStreamWaitValue32` gate on a mapped host flag,
  1,020 empty-kernel launches into one stream returned at once, and the
  1,021st blocked the host thread until the gate opened. Behind a 3 s
  spinning kernel instead, the 1,022nd blocked. A CUDA graph replay counts
  as one entry, and other streams are unaffected. The depth is not
  documented, and there is no query for it and no non-blocking launch.
  Repro: gate a stream on a host flag, launch 1,100 empty kernels, and time
  each launch.
  Side note: `CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS_V1` (92) reads 0 on
  the GB10 while the current `CAN_USE_64_BIT_STREAM_MEM_OPS` (122) and
  `CAN_USE_STREAM_WAIT_VALUE_NOR` (123) read 1. Code that reads the
  deprecated attribute wrongly concludes the GB10 lacks stream memory
  operations ([vllm.md](vllm.md) has one such case).
- **jitLLM's workaround:** page-in copies run on a copy lane of their own,
  a separate device service with its own threads and stream (`Lanes::copy`,
  `src/scheduler/scheduler.h`). DeepSeek's decode steps replay as graphs, one
  entry each (D-090). Its prefill chunks, about 4,972 launches, can still
  fill their stream. Cost: one more lane and stream.
- **Upstream refs:** none found.
- **Proposed action:** a forum post or bug asking NVIDIA to document the
  depth, add a query for it, or offer a launch that fails instead of
  blocking. Low effort: the facts above and a 50-line probe.
- **Links:** RE-029 in [rough-edges.md](../rough-edges.md);
  `unit.CudaPagedNodeTest.ACopyLaneLandsPageInsWhileAJobFillsItsStream`,
  `unit.CudaGraphTest.AGraphOfManyKernelsIsOneOperationInItsStream`;
  [swap](../experiments/fast-swap/swap.md).

## A thread blocked on a full stream blocks `cuEventCreate` and `cuStreamCreate` in other threads (RE-029)

- **Status:** open.
- **Found:** 2026-09-28, `spark-b`, driver 580.178.04.
- **Problem:** while one thread is blocked launching into a full stream
  (entry above), `cuEventCreate` and `cuStreamCreate` on any other thread
  block too, for as long as it is: 30 s in the probe, until the gate opened.
  `cuEventRecord`, `cuEventQuery`, `cuMemcpyAsync` and a synchronize on
  another, idle stream returned at once. It looks like a context-wide lock
  held across the blocked launch. Repro: thread A fills a gated stream with
  1,100 launches; thread B then calls `cuEventCreate` and times it.
- **jitLLM's workaround:** the CUDA device-execution provider takes fences'
  events from a pool made when it opens (`src/providers/cuda/cuda_device_execution.h`);
  the paged node makes 1,042 up front. Cost: the pool's size is a bound on
  fences in flight.
- **Upstream refs:** none found.
- **Proposed action:** a bug report to NVIDIA with the probe. It is the
  more surprising half of RE-029: a separate thread and stream are not
  enough to isolate one full stream.
- **Links:** RE-029 in [rough-edges.md](../rough-edges.md).

## cuBLAS keeps a 64 MiB default workspace pool that `cublasSetWorkspace` does not free (RE-028)

- **Status:** open.
- **Found:** 2026-09-27, `spark` and `spark-b`, cuBLAS 13.8.0.4, Nsight
  Systems 2025.3.2.
- **Problem:** `cublasCreate` makes three `cudaMalloc`s: 1,024 bytes,
  128 KiB and 64 MiB. They stay until `cublasDestroy`, even after
  `cublasSetWorkspace` supplies the workspace later calls use.
  `CUBLAS_WORKSPACE_CONFIG` resizes the pool, but it also affects
  numerics, which jitLLM refuses. Also, nsys's memory trace gives each
  allocation's size but not its caller, and its timestamps count
  `CLOCK_MONOTONIC_RAW` without saying so.
- **jitLLM's workaround:** none. Each handle costs 64.1 MiB beyond the
  workspace jitLLM gives it; the owner accepted that (backend-proof.md,
  "Memory and workspace").
- **Upstream refs:** the cuBLAS documentation (section 2.4.8) says the pool
  is allocated at context creation, not that `cublasSetWorkspace` leaves it.
- **Proposed action:** a feature request: free the default pool, or skip
  making it, when the caller sets its own workspace. Or at least document
  it. Low priority.
- **Links:** RE-028 in [rough-edges.md](../rough-edges.md).

## The GB10's L2 does not cache host-located memory (RE-022)

- **Status:** open (a limitation to track).
- **Found:** 2026-09-27, `spark`, driver 580.178.04.
- **Problem:** GPU re-reads of memory CUDA allocates at a host location
  (`cuMemCreate` at `HOST_NUMA` or `HOST`, `cudaMallocHost`) miss L2 every
  time: 0 of 8,388,608 sector hits and 243 GB/s, against 98.4% and
  1,952 GB/s from device memory. Streaming reads run at ~240 GB/s from both,
  so a bandwidth scan hides it. The CUDA 13.4 headers offer no flag to
  change it, and a persisting access-policy window does not.
- **jitLLM's workaround:** weights and state live in device VMM, and host VMM
  is only a landing zone that the GPU copies from (D-081). Cost: one GPU copy
  per page-in extent.
- **Upstream refs:** none found; NVIDIA's documentation does not mention it.
- **Proposed action:** ask on the forums whether this is intended and
  whether an allocation attribute is planned. It would make direct reads
  into GPU-usable memory possible. Low effort.
- **Links:** RE-022 in [rough-edges.md](../rough-edges.md);
  [host-vmm-diagnosis](../experiments/host-vmm-diagnosis/README.md).

## GB10 device memory cannot be exported as a dma-buf, and dma-buf mappings refuse direct I/O (RE-025)

- **Status:** open (a limitation to track).
- **Found:** 2026-09-27, `spark`, driver 580.178.04, CUDA 13.0.
- **Problem:** `CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED` is 0, and
  `cuMemGetHandleForAddressRange(..., DMA_BUF_FD, ...)` returns
  `CUDA_ERROR_INVALID_VALUE` (not `NOT_SUPPORTED`) for device VMM, host VMM
  and `cuMemAlloc`. Only `cuMemAllocHost` exports, and its `mmap` is a PFN
  map, so `O_DIRECT`, io_uring reads and registered buffers into it fail
  with `EFAULT`. Importing a `udmabuf` of a shmem memfd works, although the
  documentation lists it for Jetson Thor only, and the GPU treats it as
  host memory (no L2 reuse).
- **jitLLM's workaround:** the landing zone and a GPU copy, as above (D-081).
- **Upstream refs:** none found. The open kernel module's
  `nv_dma_buf_mmap` still maps by PFN in 610.57.04's source (read, not
  tested).
- **Proposed action:** a feature request for dma-buf export of device
  memory on the GB10, which would let the SSD read straight into device
  memory. Track with the next driver.
- **Links:** RE-025 in [rough-edges.md](../rough-edges.md);
  [dmabuf-direct](../experiments/dmabuf-direct/README.md).

## VMM backing is not charged to the process's cgroup (RE-019)

- **Status:** open.
- **Found:** 2026-09-25, a Spark, driver 580.178.04, cgroup v2.
- **Problem:** 8 GiB of `cuMemCreate` backing, device or host-NUMA, moved
  the cgroup's `memory.current` by at most 40 MiB while `MemAvailable` fell
  by 8 GiB. The driver's ~34 KiB of slab per 2 MiB extent is not charged
  either.
- **jitLLM's workaround:** the runtime's own budget is the bound, and the
  memory breakdown reconciles against `MemAvailable`. `MemoryMax=` on the
  service does not bound backing.
- **Upstream refs:** none found.
- **Proposed action:** a bug or feature request: charge pinned backing to
  the allocating cgroup, at least on unified-memory systems where it is
  host memory. Low priority.
- **Links:** RE-019 in [rough-edges.md](../rough-edges.md);
  [vmm-counters](../experiments/vmm-counters/README.md).

## Waking a sleeping thread takes hundreds of microseconds on the Spark (RE-017)

- **Status:** open (a limitation to track).
- **Found:** 2026-09-24 to 28, `spark-c4e2` and `spark-b`, DGX OS 7.6.0,
  cpuidle `acpi_idle` with the `menu` governor.
- **Problem:** a thread sleeping on a condition variable took 207–283 µs at
  p50 and ~450–485 µs at p99 to run after a 100–400 µs idle gap
  (2.7–72 µs on an x86 workstation). A blocking-sync event's wait returned
  1.0–1.4 ms after the GPU finished, and a host function 1.4–1.7 ms; a host
  function also holds its stream until the driver's callback thread runs it.
  A PM QoS request of 0 µs (`/dev/cpu_dma_latency`) cut each hop to ~5 µs,
  which confirms the cores' deep idle states (LPI-2 and LPI-3, exit
  latencies 231 and 433 µs) as the cause.
- **jitLLM's workaround:** the runtime wake (D-094): the completion lane
  sleeps through most of a fence's expected length and spins only around
  its likely end. Cost: 0.11–0.12 of a core while stepping. The latency
  hold is not adopted (D-095).
- **Upstream refs:** none found.
- **Proposed action:** optional: ask on the forums whether DGX OS's idle
  settings are tuned for GPU completion latency, and why blocking-sync
  waits take ~1 ms. Nothing to fix in jitLLM.
- **Links:** RE-017 in [rough-edges.md](../rough-edges.md);
  [runtime-wake](../experiments/runtime-wake/README.md),
  [task-lanes](../experiments/task-lanes/README.md).

## cuFile compatibility mode rejects a descriptor opened with `O_NOFOLLOW` (RE-002)

- **Status:** open.
- **Found:** 2026-09-21, `spark`, libcufile 1.15.1.6-1 (API 2.12), ext4.
- **Problem:** `cuFileHandleRegister` on a descriptor opened
  `O_RDONLY | O_DIRECT | O_CLOEXEC | O_NOFOLLOW` fails with 5019
  (`CU_FILE_INVALID_FILE_OPEN_FLAG`); the log reports unsupported flags
  `229376`. `O_NOFOLLOW` and `O_CLOEXEC` say nothing about how the file is
  read, so rejecting them is overly strict.
- **jitLLM's workaround:** only in the I/O comparison harness: reopen
  `/proc/self/fd/<fd>` without `O_NOFOLLOW`, check device and inode, then
  register. jitLLM's runtime does not use cuFile (D-034).
- **Upstream refs:** none found.
- **Proposed action:** check a current libcufile; if unchanged, a short bug
  report. Low priority.
- **Links:** RE-002 in [rough-edges.md](../rough-edges.md);
  [io-path](../experiments/io-path/README.md).

## CUDA 13.0 NVCC rejects `-std=c++23` (RE-001)

- **Status:** fixed upstream at CUDA 13.4 (NVCC 13.4.92 accepts it; jitLLM's
  pin).
- **Found:** 2026-09-21, NVCC 13.0.88.
- **Proposed action:** none. Kept so nobody falls back to an installed 13.0
  NVCC and expects C++23.
- **Links:** RE-001 in [rough-edges.md](../rough-edges.md);
  [toolchain-smoke](../experiments/toolchain-smoke/README.md).

## Not tracked here

- RE-035, NVCC contracting `a*b + c*d` and `a*b - c*d` into different FMAs:
  allowed by `-fmad=true`; write explicit intrinsics where bits matter.
- RE-026 and RE-027, the Spark SSD's read speed by order and data age:
  drive firmware behaviour with no clear reporting route.
