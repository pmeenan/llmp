<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Host-VMM diagnosis: why BP-F1's products slowed, and what each option costs — 2026-09-27

BP-F1 ([backend-proof.md](../../backend-proof.md), report under
[backend-proof-p1](../backend-proof-p1/README.md)) failed. With every buffer in host VMM,
GGML's matrix products ran 1.10–4.9× slower than with `cudaMalloc`. That
contradicts the evidence D-034 rests on: GPU scans of host VMM matched device
VMM at 242 GB/s ([io-path](../io-path/README.md)). This experiment asks
why, and what the alternatives cost. It is a diagnosis and reports; it
gates nothing. It does not change BP-F1's rule, harness or verdict, and it
does not choose for D-034.

**Results in brief.**

- **Mechanism: the GB10's GPU L2 does not keep lines from host-located CUDA
  memory.** Nsight Compute shows it directly. A kernel re-reads a 4 MiB
  buffer 64 times. With `cudaMalloc` or device VMM backing, 98.4% of its
  8,388,608 L2 read sectors hit, and it runs at 1,952 GB/s. With host VMM,
  every sector misses (0 hits), and it runs at 243 GB/s, the DRAM rate. The
  pattern is the same for every host-located CUDA allocation:
  `cuMemCreate` at `HOST_NUMA` (llmpalooza's provider) or `HOST`, mapped for
  the GPU alone or for the CPU too, in one handle or in 2 MiB handles, and
  `cudaMallocHost`. The GB10 is integrated (`CU_DEVICE_ATTRIBUTE_INTEGRATED`
  = 1), so to L2 every allocation, `cudaMalloc` included, is in the
  "sysmem" aperture. What differs is whether L2 keeps the lines.
- **Why D-034's scan missed it.** The I/O experiment's scan read each byte
  exactly once (256 MiB, grid-stride), so no line was ever re-read, and a
  streaming read runs at DRAM rate either way. This experiment reproduces
  it: 228–230 GB/s for all three kinds; 240–247 GB/s with 16-byte loads.
  Matrix products re-read weight tiles, activations and scratch through L2.
  They lose exactly what the scan could not show.
- **A second, smaller effect: writes to host-located memory slow a kernel's
  blocks.** Consider a grid of 151,936 blocks in which one thread per block
  writes a 32-bit word (the matrix-vector product's output pattern, without
  its reads). It takes 2.5× as long when the output is in host VMM. A GGML
  matrix-vector product at one row, with only its output in host VMM, runs
  2.13× slower (output head). With all its buffers in host VMM, its L2
  read sectors are unchanged (51.1 M either way); write sectors were not
  counted. The throughput of dense or scattered writes is not lower
  (215 vs 195 GB/s dense). The cause is not determined; see Limitations.
- **Which buffers matter.** Weights in host VMM with everything else in
  device memory costs nothing at 1 and 16 rows: 0.94–1.01× of
  `cudaMalloc` for every projection. At 512 rows it costs 1.38–1.96×,
  because the GEMM re-reads weight tiles. Its L2 hit rate falls from 86.2%
  to 48.7%. Most of BP-F1's slowdown at 1 and 16 rows comes from the
  activations, inputs and outputs (up to 4.42× with them alone in host
  VMM). At 512 rows GGML's scratch and the cuBLAS workspace alone cost the
  projections 2.07–3.01×. A KV cache in host VMM costs attention 1.16–1.54×
  at 1 and 16 rows. Device VMM matches `cudaMalloc`
  in every case (0.98–1.02×), so VMM mapping itself costs nothing. The cost
  comes from host backing.
- **Options, measured (details below).** (A) Weights in host VMM, everything
  else in device VMM: the sum of the covered kernels per token is 1.00× at
  1 and 16 rows and 1.49× at 512 rows. (B) A host-VMM landing zone copied
  into device VMM: the kernels run at `cudaMalloc` speed. A restore through
  llmpalooza's io_uring provider runs at 14.92–14.95 GB/s, against 14.96 GB/s
  in place, and each 2 MiB extent is usable 29–39 µs later at the median.
  (C) Ordinary memory read through ATS (huge-page-backed, registered): L2
  does keep its lines, but its streaming read tops out at 165 GB/s.
  Covered kernels are 1.49× at 1 row and 1.02× at 512 rows. Not available:
  device VMM cannot be mapped for the CPU (`cuMemSetAccess`:
  `CUDA_ERROR_NOT_SUPPORTED`), so a direct read cannot land in it. No
  allocation property in the CUDA 13 headers controls caching. A persisting
  L2 access-policy window over host VMM changes nothing (242 GB/s).

## Conditions and provenance

- **Host.** `spark` (`spark-c4e2`): GB10, 48 SMs, 24 MiB L2, kernel
  7.0.0-1019-nvidia, driver 580.178.04 (driver API 13000), persistence mode
  on, application clock 2,418 MHz, CPU governor `performance`, 4 KiB base
  pages, transparent huge pages `madvise`. One NUMA node (0), which is also
  the device's host NUMA id. Minimum and recommended granularity are 2 MiB
  for device, host-NUMA and host locations.
- **Session `s2`** (the one reported), 2026-09-27 07:29–07:39 UTC: 79
  processes run by [`diag_session.py`](diag_session.py). Before each
  process the driver waited until there was no compute process on the GPU
  and the load average was below 0.5; loads before processes were
  0.14–0.50. At process ends the SM clock was 2,411–2,515 MHz and the GPU
  41–53 °C, with no active throttle reason. Before 41 of the processes the
  GPU had dropped to P8 (208 MHz); before the other 38 it was still in P0
  (2,411–2,515 MHz) from the process before. No compute process was on the
  GPU at any process's start or end. An earlier
  session, `s1` (07:13–07:21), used a binary without the write tests and
  the split-output placements. It gave the same results to within its
  ranges (for example, output head at one row, all host VMM: 2.14× then,
  2.12× now). It is kept with the raw data but not reported.
- **Counters.** [`ncu_counters.sh`](ncu_counters.sh), 07:39–07:40 UTC, Nsight
  Compute 2025.3.1 (the Spark's `/usr/local/cuda/bin/ncu`), run as root
  because the driver has `RmProfilingAdminOnly=1`. These are counts; ncu
  serializes kernels, so its durations are not the timings reported.
- **Binary.** `llmp_vmm_diag_bench` from the `cross` preset, SDK
  `x86_64-e0a0c85c42806fb1`, SHA-256 `9e26330a…`, beside the pinned cuBLAS
  13.8.0.4 (`libcublas.so.13` `ee7c1657…`, `libcublasLt.so.13` `ba3b942f…`).
  Sources: commit `d3b4f2ab` plus this directory's uncommitted work;
  [`../../../benchmarks/vmm_diag_bench.cc`](../../../benchmarks/vmm_diag_bench.cc)
  `4eab72c9…`, `vmm_diag_kernels.cu` `10778874…`, `vmm_diag_kernels.h`
  `a5457b6a…`. The session's manifest records all of these, and
  [`diag-results.json`](diag-results.json) holds every aggregate below.
  Raw CSVs, logs and ncu output stay outside Git, on `spark` in
  `~/.local/share/llmp/hostvmm-diag-20260927/raw/` (`s1`, `s2`, `ncu2`).

## Method

The benchmark allocates each buffer as one of these **memory arms**:

| Arm | Allocation |
| --- | --- |
| `malloc` | `cudaMalloc` |
| `dvmm` | `cuMemCreate` at the device, mapped for the device |
| `hvmm` | `cuMemCreate` at `HOST_NUMA` 0, one handle, mapped read-write for the device and the CPU: what llmpalooza's CUDA provider does |
| `hvmm-jit` | the same through llmpalooza's `VmmProvider` itself |
| `hvmm-gpu` | host-NUMA backing mapped for the device only |
| `hvmm-2m` | host-NUMA backing in 2 MiB handles, the pager's extent size |
| `hvmm-host` | `CU_MEM_LOCATION_TYPE_HOST` backing |
| `pinned` | `cudaMallocHost` |
| `registered` | `aligned_alloc` (2 MiB aligned), touched, then `cudaHostRegister` |
| `pageable` | `aligned_alloc`, touched; the GPU reads it through the host page tables (ATS) |
| `pageable-thp`, `registered-thp` | the same with `madvise(MADV_HUGEPAGE)` before the first touch (smaps confirms huge pages) |
| `managed`, `managed-prefetch` | `cudaMallocManaged`; the second touched by the CPU and prefetched to the device |
| `dvmm-cpu` | device backing with CPU access requested: refused, `CUDA_ERROR_NOT_SUPPORTED` |

Every CPU-mapped arm took a 64 MiB `O_DIRECT` read, and the GPU read back
the same bytes as a buffered read (`info`).

- **Microkernels** (`micro`): each arm has a 1 GiB buffer. Each process
  runs three rounds. In each round every arm runs every test: two warm-up
  launches, then ten timed launches, each bracketed by CUDA events. Three
  processes shift the arm order by one arm each, giving 90 samples per
  cell. L2 is not flushed between launches, and the random tests read the
  same lines in every launch. The tests:
  - `scan4`: the I/O experiment's scan kernel as it was launched (256
    blocks of 256 threads, 4-byte loads, grid-stride), over 256 MiB.
  - `scan16`: the same with 16-byte loads and 768 blocks.
  - `write16`: a dense write of 256 MiB.
  - `write-sparse`: 2^20 4-byte writes, one per 128-byte line.
  - `write-per-block`: 151,936 blocks, each with one 4-byte write.
  - `reread-N`: an N-MiB buffer read repeatedly with L2-cached (`ld.cg`)
    16-byte loads, 256 MiB in total. Each pass starts every block at a new
    offset. `-persist` adds a persisting access-policy window over the
    buffer.
  - `rand-*`: each warp reads 64 random 128-byte lines. `all` draws them
    from the whole 1 GiB. `2m` and `64k` confine each warp to one random
    2 MiB or 64 KiB region, which tests translation locality with the same
    DRAM pattern. `8m-span` draws them from an 8 MiB span that fits in L2.
  - `copy16-self`: a 256 MiB copy within the arm.
- **Copies** (`copy`): from one arm into another, by `cudaMemcpyAsync` (the
  copy engine) or by a 16-byte SM copy kernel. Sizes are 2 MiB (successive
  extents through the buffer), 64 MiB and 1 GiB, with ten samples after two
  warm-ups, in each of three processes.
- **Restores** (`restore`): llmpalooza's `UringStorage` reads an unnamed 8 GiB
  `O_DIRECT` file on the root NVMe in 2 MiB extents, with two or four
  reads in flight. The reads go either in place into `hvmm` (D-034's path)
  or into a landing zone of 2 × depth 2 MiB `hvmm` slots. When a read
  completes, a copy (copy engine or SM kernel, one stream) moves the extent
  into its place in an 8 GiB `dvmm` destination, and the slot is reused
  once the copy's event completes. Latency is measured from a read's
  submission to its extent being usable (for the landing zone, after the
  copy). There are three passes per process and three processes.
- **GGML products** (`ggml`): BP-F1's matrix products, using its shapes and
  the implementation its recorded plan chose. They are the Qwen2.5-0.5B
  projections at 1, 16 and 512 rows (MMVF, MMF, cuBLAS) and attention's KQ
  and KQV. For attention, the "weights" operand is the F16 KV cache. The
  products run through llmpalooza's launch context and cuBLAS handle on
  llmpalooza's provider stream. Each buffer group is placed separately:
  weights or KV cache (W), inputs and outputs (A, or inputs A and outputs
  O when split), GGML's scratch (S) and the 32 MiB cuBLAS workspace (K).
  Rings rotate as BP-F1's do: more sets than fit four times in L2, per
  group, so no invocation reads what the one before it used. Each
  sample is one CUDA-graph replay of ten invocations divided by ten, timed
  by events recorded on the stream around the replay (BP-F1 captures its
  events inside the graph). There
  are five warm replays and 31 samples per case per process, and three
  processes per placement, in a rotated order. Every case produced the same
  output hash under every placement. This benchmark does not verify the
  captured launches against BP-F1's recorded plan, as BP-F1's harness does.
  For two of the 21 cases, ncu shows the recorded plan's kernels, grids
  and blocks under each of three placements: `mul_mat_vec_f<half, half, 1,
  224>` for the output head at one row, and the F32-to-F16 conversion,
  `cutlass_80_tensorop_h16816gemm_128x128_64x3_tn_align8` and the F16-to-F32
  conversion for gate/up at 512 rows. The other 19 rest on their times:
  every case's all-`malloc` median is within −4.4% to +5.6% of BP-F1's,
  and its all-`hvmm` ratio within −9.6% to +9.0% of BP-F1's (output head
  at one row: 1,048 and 2,222 µs here, 1,019 and 2,184 µs in BP-F1).

Every figure below is the median of all samples; the range in parentheses
is that of the per-process medians.

## Results

### Microkernels, GB/s (useful bytes per second)

| Arm | `scan4` | `scan16` | `write16` | `write-sparse` | `write-per-block` | `reread-4m` | `reread-4m-persist` | `reread-16m` | `reread-64m` | `rand-all` | `rand-2m` | `rand-64k` | `rand-8m-span` | `copy16-self` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `malloc` | 228 (227-236) | 247 (237-257) | 195 (194-195) | 12 (11-12) | 6 (6-6) | 1952 (1949-1953) | 1911 (1910-1936) | 1953 (1952-1967) | 378 (377-378) | 276 (274-279) | 206 (204-207) | 216 (215-217) | 377 (377-377) | 216 (216-219) |
| `dvmm` | 230 (230-241) | 240 (238-256) | 195 (194-195) | 12 (11-12) | 6 (6-6) | 1953 (1952-1979) | 1912 (1910-1938) | 1953 (1952-1954) | 391 (373-405) | 282 (279-282) | 206 (206-207) | 217 (216-217) | 377 (377-377) | 214 (211-221) |
| `hvmm` | 229 (229-230) | 241 (239-244) | 215 (214-215) | 14 (14-15) | 2 (2-3) | 243 (240-261) | 242 (240-259) | 242 (240-244) | 243 (241-245) | 214 (212-216) | 210 (210-211) | 213 (212-213) | 221 (219-229) | 217 (216-220) |
| `hvmm-jit` | 230 (228-234) | 245 (238-246) | 214 (214-215) | 15 (15-15) | 3 (2-3) | 248 (243-262) | 248 (242-261) | 242 (241-260) | 244 (241-260) | 212 (211-213) | 210 (208-210) | 213 (213-214) | 223 (222-236) | 218 (217-220) |
| `hvmm-gpu` | 229 (225-235) | 243 (242-257) | 215 (214-215) | 15 (15-15) | 2 (2-3) | 249 (245-263) | 249 (245-262) | 241 (241-261) | 242 (241-260) | 213 (213-214) | 209 (208-210) | 213 (212-214) | 224 (223-235) | 220 (218-221) |
| `hvmm-2m` | 229 (225-234) | 246 (241-248) | 214 (212-215) | 14 (14-15) | 2 (2-3) | 261 (260-262) | 261 (260-262) | 258 (242-259) | 247 (242-259) | 212 (212-212) | 209 (209-210) | 213 (212-215) | 232 (231-234) | 221 (220-222) |
| `hvmm-host` | 231 (225-234) | 244 (239-248) | 214 (213-215) | 15 (15-15) | 2 (2-3) | 261 (261-261) | 261 (261-262) | 244 (243-259) | 244 (241-259) | 213 (213-214) | 208 (208-209) | 213 (213-214) | 235 (235-236) | 219 (215-224) |
| `pinned` | 221 (220-229) | 234 (232-240) | 212 (211-212) | 14 (14-15) | 2 (2-2) | 229 (228-231) | 229 (227-230) | 234 (224-235) | 233 (231-234) | 3 (2-3) | 3 (2-3) | 208 (114-215) | 212 (207-213) | 218 (213-219) |
| `registered` | 163 (157-165) | 164 (164-165) | 114 (114-114) | 6 (6-6) | 6 (6-6) | 1924 (1817-1952) | 1885 (1780-1910) | 335 (324-361) | 183 (183-184) | 4 (3-4) | 5 (4-5) | 185 (165-186) | 377 (377-377) | 162 (161-163) |
| `pageable` | 163 (162-165) | 165 (164-165) | 114 (114-114) | 6 (6-6) | 6 (6-6) | 1952 (1951-1952) | 1910 (1910-1911) | 320 (317-325) | 182 (182-183) | 4 (4-4) | 5 (4-5) | 187 (175-187) | 377 (377-377) | 161 (158-163) |
| `pageable-thp` | 165 (165-165) | 165 (165-165) | 114 (114-114) | 6 (6-6) | 6 (6-6) | 1953 (1952-1956) | 1912 (1910-1938) | 315 (306-323) | 181 (178-182) | 183 (179-185) | 178 (178-179) | 187 (186-188) | 377 (377-377) | 157 (155-157) |
| `registered-thp` | 165 (164-165) | 164 (164-165) | 114 (114-114) | 6 (6-6) | 6 (6-6) | 1953 (1952-1953) | 1914 (1911-1937) | 318 (314-330) | 181 (179-184) | 181 (181-183) | 178 (177-179) | 187 (186-187) | 377 (377-377) | 158 (154-160) |
| `managed` | 165 (162-165) | 164 (164-165) | 114 (114-114) | 6 (6-6) | 6 (6-6) | 1953 (1952-1953) | 1911 (1910-1937) | 322 (317-326) | 184 (183-184) | 185 (183-185) | 179 (178-180) | 187 (187-188) | 377 (377-377) | 157 (155-159) |
| `managed-prefetch` | 165 (162-165) | 165 (164-165) | 114 (114-114) | 6 (6-6) | 6 (6-6) | 1952 (1952-1953) | 1910 (1910-1911) | 327 (322-334) | 181 (180-186) | 183 (183-185) | 178 (178-179) | 187 (186-189) | 377 (377-378) | 156 (155-157) |

90 samples per cell from 3 processes. For `write-per-block` the medians
are 96 µs (`malloc`, `dvmm`, the ATS arms), 241–247 µs (the `hvmm` arms)
and 277 µs (`pinned`).

The arms fall into three families:

- **Device-located** (`malloc`, `dvmm`): streaming ~240 GB/s, L2 re-reads
  ~1,950 GB/s.
- **Host-located CUDA allocations** (every `hvmm` variant, `pinned`):
  streaming at the same ~240 GB/s, but every re-read costs a DRAM access
  (229–261 GB/s whatever the working set). Random access within an 8 MiB
  span gets 212–235 GB/s instead of 377. `cudaMallocHost` is also catastrophic
  for random access across 1 GiB (3 GB/s), which host VMM is not
  (213 GB/s).
- **Ordinary memory through ATS** (`registered`, `pageable`, `managed`, with
  or without huge pages): L2 keeps its lines (1,952 GB/s at 4 MiB), but
  streaming stops at ~165 GB/s, dense writes at 114 GB/s, and a working set
  of 16 MiB re-reads at only ~320 GB/s. With 4 KiB pages random access
  across 1 GiB collapses to 4 GB/s; huge pages restore it to ~181 GB/s.

### L2 counters

| Run | Kernel | Read sectors | L2 hits | Hit rate |
| --- | --- | ---: | ---: | ---: |
| `reread-4m`, `malloc`, `dvmm`, `managed`, `registered-thp` | reread | 8,388,608 | 8,257,536 | 98.4% |
| `reread-4m`, `pageable` | reread | 8,388,608 | 8,225,572 | 98.1% |
| `reread-4m`, `hvmm`, `hvmm-gpu`, `hvmm-2m`, `pinned` | reread | 8,388,608 | 0 | 0.0% |
| `scan16`, every arm | scan | 8,388,608 | 0 | 0.0% |
| gate/up at 512 rows, all `malloc` | cuBLAS GEMM (2 launches) | 4,358,144 | 3,756,032 | 86.2% |
| gate/up at 512 rows, W `hvmm`, rest `dvmm` | cuBLAS GEMM (2 launches) | 4,358,144 | 2,121,728 | 48.7% |
| gate/up at 512 rows, all `hvmm` | cuBLAS GEMM (2 launches) | 4,358,144 | 0 | 0.0% |
| output head at one row, all `malloc` | MMVF (6 launches) | 51,109,052 | 55,768 | 0.1% |
| output head at one row, all `hvmm` | MMVF (6 launches) | 51,109,608 | 0 | 0.0% |

Every read in every run was in the sysmem aperture. The device aperture
counted zero sectors, even for `cudaMalloc`. The 131,072 misses of the
4 MiB reread are its first pass. In the output head at one row, L1 serves
the input vector: its 17.0 M L1 global-load sector hits were identical
with inputs in `malloc` and in `hvmm` (a separate ncu check on the
development binary, with the same GGML kernel; its output was not
kept). Its L2
reads are the weights, read once, which miss in every arm. That product's
slowdown with everything in host VMM therefore does not come from reads.
It comes from its output (below).

### GGML products by placement

All-`cudaMalloc` median µs, and each placement's median as a ratio to it.
W is the weights (or KV cache), A the activations (inputs and outputs), O
the outputs when split from A, S GGML's scratch, and K the cuBLAS
workspace.

| Case | all `malloc`, µs | all `dvmm` | all `hvmm` (BP-F1) | all `hvmm-jit` | W `hvmm`, rest `malloc` | W `hvmm`, rest `dvmm` | W `hvmm-2m`, rest `dvmm` | A `hvmm` (in + out) | inputs `hvmm` | outputs `hvmm` | S + K `hvmm` | W `registered-thp`, rest `dvmm` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `linear.q_o@1` | 8.2 | 1.01 | 2.68 | 2.67 | 1.00 | 1.01 | 1.00 | 2.66 | 1.37 | 2.64 | 1.00 | 1.39 |
| `linear.k_v@1` | 2.7 | 1.01 | 2.21 | 2.20 | 0.94 | 0.96 | 1.00 | 2.26 | 1.84 | 2.37 | 0.99 | 1.15 |
| `linear.gate_up@1` | 34.6 | 1.01 | 2.30 | 2.29 | 1.00 | 0.99 | 1.00 | 2.32 | 1.07 | 2.33 | 1.00 | 1.56 |
| `linear.down@1` | 36.3 | 1.01 | 1.30 | 1.30 | 1.01 | 1.00 | 1.00 | 1.30 | 1.17 | 1.17 | 0.99 | 1.51 |
| `linear.lm_head@1` | 1048.2 | 1.00 | 2.12 | 2.12 | 1.00 | 1.00 | 1.00 | 2.12 | 1.00 | 2.13 | 1.00 | 1.53 |
| `attn.kq.kv768@1` | 3.6 | 0.98 | 1.46 | 1.46 | 1.17 | 1.20 | 1.16 | 1.28 | 1.33 | 1.00 | 0.99 | 0.97 |
| `attn.kqv.kv768@1` | 4.7 | 1.01 | 5.10 | 5.10 | 1.54 | 1.54 | 1.54 | 4.42 | 1.99 | 3.61 | 1.01 | 1.01 |
| `linear.q_o@16` | 10.0 | 1.01 | 1.73 | 1.73 | 0.99 | 0.99 | 1.01 | 1.75 | 1.93 | 0.99 | 1.01 | 1.12 |
| `linear.k_v@16` | 5.4 | 1.00 | 1.04 | 1.04 | 0.96 | 0.96 | 0.96 | 1.04 | 1.05 | 1.00 | 1.00 | 1.00 |
| `linear.gate_up@16` | 41.6 | 1.00 | 1.85 | 1.87 | 1.00 | 1.00 | 1.00 | 1.86 | 1.91 | 0.99 | 1.01 | 1.21 |
| `linear.down@16` | 43.3 | 1.00 | 2.12 | 2.11 | 0.99 | 1.00 | 1.00 | 2.12 | 2.15 | 0.99 | 1.00 | 1.18 |
| `linear.lm_head@16` | 1174.0 | 1.00 | 1.28 | 1.28 | 1.00 | 1.00 | 1.00 | 1.27 | 1.29 | 1.00 | 1.00 | 1.24 |
| `attn.kq.kv256@16` | 3.8 | 1.00 | 1.78 | 1.75 | 1.29 | 1.29 | 1.29 | 1.44 | 1.48 | 1.07 | 1.01 | 1.04 |
| `attn.kqv.kv256@16` | 4.9 | 1.00 | 1.38 | 1.37 | 1.25 | 1.25 | 1.25 | 1.13 | 1.16 | 1.00 | 0.98 | 1.00 |
| `linear.q_o@512` | 35.0 | 1.00 | 3.15 | 3.08 | 1.60 | 1.61 | 1.60 | 0.91 | 0.98 | 0.91 | 2.48 | 0.98 |
| `linear.k_v@512` | 17.0 | 1.02 | 2.48 | 2.52 | 1.38 | 1.38 | 1.40 | 0.96 | 1.01 | 0.97 | 2.07 | 1.00 |
| `linear.gate_up@512` | 137.5 | 1.00 | 2.92 | 2.90 | 1.80 | 1.79 | 1.80 | 0.89 | 1.00 | 0.87 | 2.17 | 1.06 |
| `linear.down@512` | 122.2 | 1.01 | 3.95 | 3.97 | 1.96 | 1.94 | 1.97 | 0.90 | 0.97 | 0.94 | 3.01 | 1.01 |
| `linear.lm_head@512` | 4109.6 | 1.00 | 2.84 | 2.87 | 1.52 | 1.53 | 1.52 | 0.99 | 1.05 | 1.04 | 2.27 | 1.02 |
| `attn.kq.kv768@512` | 126.6 | 1.01 | 1.67 | 1.66 | 1.00 | 1.00 | 1.01 | 1.28 | 1.34 | 1.01 | 1.36 | 1.02 |
| `attn.kqv.kv768@512` | 184.5 | 1.00 | 1.75 | 1.75 | 1.39 | 1.38 | 1.38 | 0.96 | 0.99 | 0.98 | 1.30 | 1.01 |

93 samples per cell from 3 processes; the largest range of per-process
medians is 9.0% of its median. In the "inputs `hvmm`" column the outputs
are in `malloc`; in the "outputs `hvmm`" column the inputs are.

What each buffer group costs in host VMM:

- **Weights**: nothing at 1 and 16 rows (the matrix-vector and small-batch
  kernels read each weight once). 1.38–1.96× at 512 rows (the GEMM re-reads
  weight tiles from L2).
- **KV cache** (the "weights" of attention): 1.16–1.54× at 1 and 16 rows
  (query heads grouped over two KV heads re-read it). 1.00–1.39× at 512.
- **Inputs**: 1.05–2.15× at 1 and 16 rows, except the output head at one
  row (1.00×, where L1 holds its input vector). The projections at 512
  rows are 0.97–1.05×, because GGML first converts the inputs to F16 in
  scratch; KQ at 512 rows is 1.34×.
- **Outputs**: 1.17–3.61× for the matrix-vector (MMVF) kernels at one row,
  the per-block write effect. 0.99–1.07× at 16 rows. At 512 rows
  0.87–1.04×, the faster end matching the faster dense writes.
- **Scratch and cuBLAS workspace**: 1.30–3.01× at 512 rows, where the GEMM
  reads its converted operand from scratch.
- **2 MiB handles and the provider** change nothing (`hvmm-2m`, `hvmm-jit`
  equal `hvmm`).

### Per token: sums of the covered kernels

These sums are **derived** from the medians above, not a measured token.
Per layer they add two `q_o`, two `k_v`, two `gate_up`, one `down`, one KQ
and one KQV, over Qwen2.5-0.5B's 24 layers, plus one output head. The
attention products are the measured ones: 768 cache cells at 1 and 512
rows, 256 at 16. They exclude every operation llmpalooza does not have yet
(RoPE, softmax, norms, SiLU, copies). `summarize.py` prints this table.

| Placement | 1 row, µs | 16 rows | 512 rows |
| --- | ---: | ---: | ---: |
| all `malloc` | 4,298 (1.00×) | 5,155 (1.00×) | 23,606 (1.00×) |
| all `dvmm` | 1.005× | 1.003× | 1.002× |
| all `hvmm` (what BP-F1 measured) | 2.14× | 1.71× | 2.66× |
| weights and KV cache `hvmm`, the rest `dvmm` | 1.016× | 1.009× | 1.56× |
| weights `hvmm`, KV cache and the rest `dvmm` | 0.998× | 0.998× | 1.49× |
| weights `registered-thp`, the rest `dvmm` | 1.49× | 1.19× | 1.02× |

### Copies between kinds

| From | To | Method | 2 MiB, µs | 2 MiB, GB/s | 64 MiB, GB/s | 1 GiB, GB/s |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `hvmm` | `dvmm` | copy engine | 35.9 (35.8-35.9) | 58 | 60 | 60 |
| `hvmm` | `dvmm` | SM kernel | 20.6 (20.5-21.2) | 102 | 110 | 107 |
| `hvmm` | `malloc` | copy engine | 35.9 (35.8-36.0) | 58 | 60 | 60 |
| `hvmm` | `malloc` | SM kernel | 20.4 (20.0-20.5) | 103 | 110 | 107 |
| `pinned` | `dvmm` | copy engine | 36.2 (36.1-36.4) | 58 | 59 | 59 |
| `registered-thp` | `dvmm` | copy engine | 35.8 (35.8-35.8) | 59 | 60 | 60 |
| `dvmm` | `dvmm` | copy engine | 22.5 (21.8-22.7) | 93 | 116 | 115 |
| `dvmm` | `dvmm` | SM kernel | 21.2 (21.1-21.2) | 99 | 110 | 107 |
| `malloc` | `malloc` | copy engine | 22.6 (21.7-23.1) | 93 | 115 | 115 |
| `hvmm` | `hvmm` | copy engine | 92.2 (91.4-94.1) | 23 | 24 | 25 |
| `hvmm` | `hvmm` | SM kernel | 20.7 (20.6-20.7) | 101 | 112 | 108 |

30 samples per cell from 3 processes; [`diag-results.json`](diag-results.json)
has every range. GB/s counts bytes copied; memory traffic is twice that.
The copy engine reads host-located memory at 60 GB/s, half its
device-to-device rate. An SM copy kernel runs at ~107 GB/s from any kind,
but it uses the SMs.

### Restores

8 GiB in 2 MiB direct reads through llmpalooza's io_uring provider:

| Path | In flight | GB/s | p50 µs | p99 µs |
| --- | ---: | ---: | ---: | ---: |
| in place into `hvmm` (D-034) | 2 | 13.443 (13.361-13.451) | 308 (308-308) | 420 (379-449) |
| `hvmm` landing, copy engine to `dvmm` | 2 | 13.144 (13.128-13.236) | 347 (347-347) | 618 (542-685) |
| `hvmm` landing, SM copy kernel to `dvmm` | 2 | 13.162 (12.819-13.350) | 339 (337-365) | 525 (461-736) |
| in place into `hvmm` (D-034) | 4 | 14.957 (14.916-14.967) | 558 (558-558) | 637 (624-705) |
| `hvmm` landing, copy engine to `dvmm` | 4 | 14.923 (14.919-14.936) | 597 (597-597) | 726 (694-756) |
| `hvmm` landing, SM copy kernel to `dvmm` | 4 | 14.949 (14.925-14.960) | 587 (587-587) | 678 (668-729) |

Nine passes per row from three processes. The landing zone was four (at
two in flight) or eight (at four) 2 MiB slots. The in-place rows reproduce
D-034's 14.903–14.968 GB/s and 558 µs. The M0 I/O experiment's staged
io_uring path, a registered staging buffer copied into device VMM,
measured 14.947–14.955 GB/s. Its staging memory was of a different kind,
but it agrees.

## Why D-034's evidence did not show this

The M0 scan ([`kernel.cu`](../io-path/kernel.cu), `scan`) reads each 32-bit
word of 256 MiB exactly once. Every byte comes from DRAM whatever the
backing, so L2 caching cannot matter, and a streaming read on the GB10 runs
at the same rate from every CUDA-allocated kind. The scan measured the
bandwidth of a single pass, which is real. Nothing in it re-read data,
which is what the products do: GEMM tiles, attention's grouped KV reads,
matrix-vector inputs, and scratch. The I/O report noted this limit ("It is
still a scan, not GGML or quantized GEMM").

## Hypotheses tested and excluded

- **VMM mapping in general**: device VMM equals `cudaMalloc` in every
  microkernel and every GGML case (0.98–1.02×).
- **GPU page size or translation**: from host VMM, random 128-byte lines
  run at the same rate across the whole 1 GiB as confined to 2 MiB or
  64 KiB windows (208–214 GB/s for every `hvmm` arm), and the windowed
  rates equal `cudaMalloc`'s (206–217 GB/s). Granularity is 2 MiB for
  every location; 2 MiB handles change nothing. (Across the whole 1 GiB,
  device memory is faster, 276–282 GB/s. The same lines are read in every
  launch without an L2 flush, so this may be lines kept across launches;
  it was not separated.) (Translation does matter for the
  ATS arms with 4 KiB pages: 4 GB/s random.)
- **The CPU access grant**: mapping host VMM for the device alone
  (`hvmm-gpu`) behaves identically. No L2 hits, and the same write effect.
- **NUMA or location type**: `HOST` and `HOST_NUMA` backing behave
  identically; the Spark has one NUMA node.
- **An L2 access-policy window** marking the host range persisting: no
  change (242 vs 243 GB/s).

## Options for D-034, with their measured costs

None is recommended here; the call is the owner's.

1. **Weights in host VMM, consumed in place; everything else (activations,
   outputs, scratch, cuBLAS workspace, KV cache) in device VMM.**
   - Decode and small batches cost nothing: 0.94–1.01× for every projection
     at 1 and 16 rows. The covered kernels per token come to 0.998×.
   - Prefill-size GEMMs cost 1.38–1.96× per projection, 1.49× for the
     covered 512-row kernels.
   - D-034's direct read in place is kept for weights. KV and conversation
     state restore (D-019) could no longer land in place: device VMM cannot
     be mapped for the CPU. It needs option 2's copy, or it costs attention
     1.16–1.54× if left in host VMM.
   - The allocation classes (D-006/D-033) must place kinds by role.
2. **A host-VMM landing zone, then a copy into device VMM (weights and
   state).**
   - Kernels run at `cudaMalloc` speed (device VMM 0.98–1.02×).
   - A restore keeps disk speed: 14.92–14.95 GB/s against 14.96 in place
     at four in flight; 13.14–13.16 against 13.44 at two.
   - Each extent becomes usable 29–39 µs later at the median, and 40–200 µs
     later at p99.
   - Costs: a bounded landing pool (2 × depth × 2 MiB, 8–16 MiB here) and
     2× the restored bytes in extra DRAM traffic (about 30 GB/s during a
     14.9 GB/s restore, against ~240 GB/s streaming).
   - Either the copy engine (60 GB/s from host-located memory, no SM use)
     or an SM kernel (~107 GB/s, which competes with compute).
   - This is the staged path D-034 replaced, but with host VMM as the
     staging kind. It needs no CPU payload copy.
   - Not measured here: the copy's effect on concurrently running kernels.
3. **Both, by phase or by use.** Weights that are only ever consumed at
   decode-size batches stay in place (option 1's zero cost). Anything
   re-read heavily (KV, prefill-heavy dense weights) is copied (option 2).
   The costs are the two above, plus policy complexity.
4. **Ordinary huge-page memory read through ATS** (`registered-thp`, or
   `managed`).
   - L2 keeps its lines: 512-row products 0.98–1.06×, the covered kernels
     1.02×.
   - Streaming stops at ~165 GB/s: 1-row projections 1.15–1.56×, the covered
     kernels 1.49× at 1 row and 1.19× at 16.
   - It accepts direct reads, but it is not CUDA VMM. Backing, mapping and
     reclaim are the OS's (D-006), and huge pages are not guaranteed.
     Without them, random access falls to 4 GB/s.
5. **Keep everything in host VMM (D-034 as written).** BP-F1 measured the
   per-product cost (1.10–4.9×). This experiment's all-`hvmm` placement
   reproduces it; its derived per-token sum is 2.14× the covered kernels
   at one row, 1.71× at 16, 2.66× at 512.

Not available on this platform, as measured:

- **CPU-mapped device VMM**, which would let a direct read land in
  L2-cacheable memory: `cuMemSetAccess` returns
  `CUDA_ERROR_NOT_SUPPORTED`. Not tried: `mmap` of a dma-buf exported
  from device memory (`cuMemGetHandleForAddressRange`, which the CUDA 13.4
  header says maps as cached memory on coherent ARM platforms when
  `CU_DEVICE_ATTRIBUTE_DMA_BUF_MMAP_SUPPORTED` is set), and whether a direct
  read into such a mapping works.
- **An allocation or access flag that makes host VMM cacheable.** None
  exists to try: in the CUDA 13.4 headers, `CUmemAllocationProp` offers
  compression, RDMA and usage hints only (none was set here), and
  `cuMemSetAccess` offers protection only. Tried: host and host-NUMA
  locations, GPU-only and GPU-plus-CPU access, 2 MiB handles,
  `cudaMallocHost`, and a persisting access-policy window. Load cache
  hints other than `ld.cg` were not tried.

## Limitations and what was not determined

- **The write effect's cause is not determined.** Blocks that write to
  host-located memory take longer to complete. Hypothesis: stores to
  memory L2 does not keep complete only at DRAM, and a block cannot retire
  until its stores are acknowledged. That would explain 246 vs 96 µs for
  151,936 one-write blocks, while throughput-bound write loops are not
  slower. It is untested. It matters only for options that put outputs in
  host VMM.
- **Why the L2 does not keep host-located lines** (a page-table attribute
  the driver sets for coherence with the CPU, or a hardware rule for this
  aperture) cannot be seen from user space, and this work did not search
  NVIDIA's documentation for it. Whether a driver setting changes it is
  unknown.
- **Kernels only.** The per-token figures are sums of the kernels llmpalooza
  has, at one model's shapes. Real steps add operations and overlap. No
  quantized kernel (EXL3, GGML quants) was measured; their tile reuse will
  differ.
- **One host, one driver, one day.** GB10 on driver 580.178.04; a
  discrete-GPU or Grace-Hopper system, where device memory is separate, was
  not measured.
- **Restores ran alone.** Copy traffic's effect on concurrent kernels, and
  kernels' effect on the restore, were not measured. (The I/O experiment
  measured a background scan losing about 10% of its rate during in-place
  io_uring reads and about 20% during staged reads by CPU workers.)
- **No launch verification.** Unlike BP-F1's harness, this benchmark does
  not compare captured launches with the recorded plan. ncu confirms the
  kernels for two cases; for the other 19, reproducing BP-F1's times
  within about 10% is the only evidence that it runs the same kernels.

## Reproduction

On the workstation, build the `cross` preset and copy the benchmark beside
the SDK's cuBLAS, as the BP-F1 harness is:

```bash
mise run build -- cross
ssh spark 'mkdir -p ~/diag/{benchmarks,cublas}'
rsync -aL build/cross/cublas/ spark:diag/cublas/
rsync -a build/cross/benchmarks/llmp_vmm_diag_bench spark:diag/benchmarks/
scp docs/experiments/host-vmm-diagnosis/{diag_session.py,ncu_counters.sh} spark:diag/
```

On `spark`, with a 64 MiB file for the direct-read check and a directory on
the SSD for the restore's unnamed 8 GiB file:

```bash
cd ~/diag && mkdir -p scratch && head -c 67108864 /dev/urandom > scratch/dio-64m.bin
python3 -B diag_session.py raw/s2 --bench benchmarks/llmp_vmm_diag_bench --repeats 3 \
  --file scratch/dio-64m.bin --dir scratch --note '…'
bash ncu_counters.sh benchmarks/llmp_vmm_diag_bench raw/ncu2
```

Then, on the workstation, with the raw directories copied back:

```bash
python3 -B docs/experiments/host-vmm-diagnosis/summarize.py raw/s2 --ncu raw/ncu2 \
  --json docs/experiments/host-vmm-diagnosis/diag-results.json
```
