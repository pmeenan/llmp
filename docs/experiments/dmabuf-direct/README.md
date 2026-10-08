<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Direct landing in GPU device memory through a dma-buf — 2026-09-27

The [host-VMM diagnosis](../host-vmm-diagnosis/README.md) found that the
GB10's L2 never keeps lines from host-located CUDA memory (RE-022). Device
VMM is cached, but `cuMemSetAccess` refuses to map it for the CPU, so a direct
read cannot land in it. The measured fallback lands reads in a small host-VMM
zone and copies each extent into device VMM. This experiment tests the one
path the diagnosis left untried. It exports device memory as a dma-buf,
`mmap`s that for the CPU, and has `O_DIRECT` land in it with no copy. The CUDA
13.4 header says such mappings are "regular cached memory" on coherent ARM
platforms when `CU_DEVICE_ATTRIBUTE_DMA_BUF_MMAP_SUPPORTED` is set. It also
tries the reverse direction, ordinary memory imported into CUDA as a dma-buf.
It gates nothing and informs D-034.

**Answer: it does not work on this platform. The landing-and-copy path
stands.**

- **Device memory cannot be exported as a dma-buf on driver 580.178.04.**
  The driver reports `CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED` = 0, and the
  CUDA 13.0 and 13.4 headers make that attribute the condition for exporting
  `cuMemAlloc` or `cuMemAddressReserve`/`cuMemMap` ranges.
  `cuMemGetHandleForAddressRange(..., CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, ...)`
  returns `CUDA_ERROR_INVALID_VALUE` for every device allocation tried:
  device VMM in one handle, in 2 MiB handles, and with
  `requestedHandleTypes` set to a POSIX fd, each with and without
  `CU_MEM_RANGE_FLAG_DMA_BUF_MAPPING_TYPE_PCIE`, and over 128, 129 and
  4,096 handles. It does the same for `cuMemAlloc` memory. (The wide
  exports and `cuMemAlloc` were tried without the flag.) Requesting
  `gpuDirectRDMACapable` makes `cuMemCreate` fail with `INVALID_DEVICE`; the
  driver reports `GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED` = 0.
  This driver does not know attribute 152 (`DMA_BUF_MMAP_SUPPORTED`, from the
  13.4 header): querying it returns `INVALID_VALUE`. Host VMM is refused as
  well. Only `cuMemAllocHost` memory exports and maps, and that memory is
  host-located, so L2 does not keep its lines (RE-022). The VMM's own POSIX fd
  (`cuMemExportToShareableHandle`) is a handle on `/dev/nvidiactl`, and
  `mmap` of it fails with `EINVAL`.
- **Even where a driver does map a dma-buf, direct I/O cannot target the
  mapping.** NVIDIA's dma-buf `mmap` (`nv_dma_buf_mmap` in the open kernel
  module) builds the mapping with `remap_pfn_range`, so the VMA is
  `VM_PFNMAP | VM_IO` (smaps: `pf io`). `pin_user_pages` refuses such VMAs.
  On the `cuMemAllocHost` export, which does map, every direct path fails
  with `EFAULT`: `O_DIRECT` `pread`, `O_DIRECT` `pwrite` from the mapping,
  io_uring `READ` on an `O_DIRECT` fd, and `IORING_REGISTER_BUFFERS`. A
  buffered `pread` works, because the kernel copies from the page cache with
  the CPU. The mapping is CPU-cached: 28 GB/s single-thread reads, the same
  as `malloc`. The driver source says so directly ("We can't support use case
  to call pin_user_pages() on dma-buf's CPU VA", in `nv_dma_buf_mmap`,
  `nvidia/nv-dmabuf.c`, beside its `nv_remap_page_range` call, which wraps
  `remap_pfn_range`). This is true in the installed 580.178.04 source
  (`/usr/src/nvidia-580.178.04`) and in the newest packaged driver's source,
  610.57.04 (Ubuntu `nvidia-kernel-source-610-open`
  610.57.04-0ubuntu0.24.04.3, downloaded, not installed), which also adds
  `BUG_ON(!(vma->vm_flags & VM_PFNMAP))` after the call. A newer driver that
  allowed device-memory export would therefore give a mapping for CPU copies,
  not a target for direct reads. That was read from source, not tested; no
  newer driver was installed.
- **No dma-buf-aware file-read path in kernel 7.0.** In its uapi header
  (7.0.0-1019-nvidia), io_uring's only dma-buf registration is its network
  zero-copy receive (`IORING_ZCRX_AREA_DMABUF`).
  No block or file read can target a dma-buf.
- **The reverse direction works but does not help.** A `udmabuf` export of
  an ordinary shmem `memfd` (4 KiB pages) imports into CUDA through
  `cuImportExternalMemory(CU_EXTERNAL_MEMORY_HANDLE_TYPE_DMABUF_FD)`. The
  header documents this for Jetson Thor only, yet it works on GB10.
  Direct reads land in it: 8 GiB restores reach 14.70 GB/s at four and eight
  in flight, and the GPU reads every byte exactly. CUDA's pointer attributes
  call it device memory (`MEMORY_TYPE` 2), but the GPU treats it as
  host-located memory:
  - L2 does not keep its lines (4 MiB re-read: 226 GB/s, against 1,979 for
    device VMM).
  - With 4 KiB pages, random 128-byte reads across 1 GiB fall to 2.6 GB/s.
  - Blocks that write to it finish slowly, like host VMM.
  - A hugetlb-backed `udmabuf` is refused (`CUDA_ERROR_NOT_SUPPORTED`).
  - It lives outside CUDA VMM: the driver picks the address, and
    `cuMemImportFromShareableHandle` on the dma-buf fd returns
    `CUDA_ERROR_UNKNOWN`.

  It is strictly worse than host VMM.
- **Landing and copying, reproduced in the same session and now checked
  word for word.** Over an 8 GiB file, in 2 MiB `O_DIRECT` io_uring reads:
  - At four in flight: in place into host VMM 14.941 GB/s; landing zone plus
    copy engine 14.918; landing zone plus SM copy kernel 14.953.
  - At eight: 14.952, 14.955 and 14.956.
  - Each extent is usable 29 µs (SM kernel) or 39 µs (copy engine) later at
    the median, at every depth.
  - Every pass matched the file's pattern with 0 bad words, and the negative
    controls caught the one corrupted word.

  This matches the diagnosis (14.92–14.95 against 14.96; +29–39 µs).

**Recommendation for D-034.** Adopt the landing-and-copy path (the
diagnosis's option 2): weights and state live in device VMM. Direct reads
land in a bounded host-VMM zone (2 × depth × 2 MiB), and each extent is copied
into device VMM. On this Spark, no direct-landing route leads to L2-cacheable
memory. Device VMM refuses CPU mapping and dma-buf export. NVIDIA's dma-buf
mappings are PFN maps that direct I/O cannot pin. Imported ordinary memory is
not cached by L2. At four or more reads in flight, the copy costs no
measurable restore bandwidth. Keep the copy engine as the default: it uses no
SMs and adds ~39 µs per extent. The SM kernel is ~10 µs faster but competes
with compute. Neither copy's effect on concurrently running kernels was
measured. Reopen this only if a driver both exports device memory and maps it
with struct pages (`vm_insert_page`/`VM_MIXEDMAP`) rather than PFNs, or if
the kernel gains a dma-buf read path for files.

## Conditions and provenance

- **Host.** `spark` (`spark-c4e2`): GB10 (48 SMs, 24 MiB L2, integrated),
  kernel 7.0.0-1019-nvidia, NVIDIA open kernel module and driver 580.178.04
  (driver API 13000), persistence mode on, CPU governor `performance`, THP
  `madvise`, shmem THP `never`. The NVMe is the root Samsung PCIe 5.0 ×4
  (ext4), the same device as the [I/O experiment](../io-path/README.md). The
  test file is a 16 GiB file of a 32-bit hash pattern
  (`pattern(word index)`), written with `O_DIRECT`. Restores read its first
  8 GiB.
- **Session `s1`**, 2026-09-27 14:16–14:21 UTC, 43 processes run by
  [`session.sh`](session.sh) as root (`/dev/udmabuf` is `root:kvm`). Before
  each process the driver waited until no compute process was on the GPU and
  the 1-minute load was below 0.5. Loads were 0.33–0.47, the GPU was at
  39–45 °C with no throttle reason, and it was in P0 before 34 processes
  (2,411–2,535 MHz) and in P8 before 9. The session raised the `udmabuf`
  `size_limit_mb` from its default 64 to 1,024. The limit was restored to 64
  afterwards (by hand: the value was already 1,024 from a trial when the
  session began, so its restore step kept 1,024). A trial set
  `vm.nr_hugepages` to 600 for the hugetlb import test, and it was restored
  to 0.
- **Binary.** [`dmabuf_probe.cu`](dmabuf_probe.cu), built on the Spark by
  [`build.sh`](build.sh): CUDA 13.0 `nvcc` V13.0.88, host GCC 13.3.0,
  `-std=c++20 -O3 -arch=sm_121 -Xcompiler -Wall,-Wextra`, linked with
  `-lcuda` and the static cudart. It is a standalone probe, not the SDK
  build. Session `s1` used the binary with SHA-256 `66fd2941…` (source
  `a714a48a…`). The committed source (`3c58eaa5…`) adds two things after the
  session: a CPU-bandwidth measurement of the `cuMemAllocHost` dma-buf
  mapping, and a fix to the restore's negative control (below). Those were
  run as `s1b` (binaries `4a6be4fb…` and `4c2f191c…`), by hand rather than
  through `session.sh`, so without its idle wait or conditions record.
  `s1b`'s manifest records only `4c2f191c…`, the committed source's build;
  its `feasibility` output predates that build and came from `4a6be4fb…`,
  whose source differs only in the negative control.
- **Raw output** stays outside Git, on `spark` in
  `~/.local/share/llmp/dmabuf-20260927/raw/` (`s1`, `s1b`, and the trial
  runs). [`results.json`](results.json) holds the session's aggregates.

## Method

The probe's subcommands (see its usage text):

- `info`: device attributes and VMM granularity (2 MiB for device and host
  NUMA).
- `feasibility`: allocate 64 MiB of each kind, try to export it (VMM kinds
  with and without the PCIe flag), and try to `mmap` the dma-buf read-write
  and read-only. Where a mapping exists, it runs `O_DIRECT` `pread`, `O_DIRECT`
  `pwrite` from the mapping, an io_uring `READ`, io_uring registered buffers
  with `READ_FIXED`, and a buffered `pread`. The GPU checks each landed
  extent against the pattern. CPU bandwidth is 64 MiB read (4 × 64-bit
  accumulators), `memset` and `memcpy` in, on 1 and 8 unpinned threads, with
  10 repetitions after 2 warm-ups. (The branches for
  a mapped device export never ran: no export succeeded.)
- `udmabuf`: a `memfd` (shmem, or hugetlb with `4k`/`2m`), sealed against
  shrinking and populated. `UDMABUF_CREATE` makes the dma-buf, and
  `cuImportExternalMemory` plus `cuExternalMemoryGetMappedBuffer` give the GPU
  pointer. It then records pointer attributes, the VMM-handle import, and the
  direct paths into the CPU mapping.
- `micro`: the diagnosis's microkernels, unchanged, over 1 GiB of each kind
  in one process: device VMM in 2 MiB handles (`dvmm`), `cuMemAlloc`
  (`malloc`), host VMM mapped for the CPU (`hvmm`), and the `udmabuf` import
  (`udmabuf-4k`). Each cell is 2 warm-up and 10 timed launches, bracketed by
  events. Three processes rotate the arm order. The kernels:
  - `scan16`: a 256 MiB streaming read with 16-byte loads.
  - `write16`: a dense 256 MiB write.
  - `write-per-block`: 151,936 blocks with one 4-byte write each.
  - `reread-N`: an N MiB buffer re-read with `ld.cg`, 256 MiB in total.
  - `rand-*`: 64 random 128-byte lines per warp, drawn from 1 GiB, from
    2 MiB windows, or from an 8 MiB span.

  Then come the coherence checks. The first word pattern is read directly
  into 4 MiB, and the GPU verifies it and primes L2 with 8 re-read passes.
  A second direct read (DMA) replaces it, and the GPU verifies. The GPU
  primes L2 again, CPU stores replace the data, and the GPU verifies. GPU
  stores then replace it, and the CPU verifies. There are 20 rounds per
  process.
- `restore`: `O_DIRECT` io_uring `READ`s of 2 MiB (non-fixed buffers, as
  llmpalooza's `UringStorage` issues), with 2, 4 or 8 in flight, over 8 GiB. The
  modes:
  - `hvmm-inplace`: reads land in place in host VMM (D-034 as written).
  - `land-ce` / `land-sm`: reads land in 2 × depth host-VMM slots. On each
    completion a copy-engine `cudaMemcpyAsync` or an SM copy kernel (192
    blocks) on one stream moves the extent into 2 MiB-handle device VMM. A
    slot is reused once its copy's event completes.
  - `udmabuf-4k`: reads land in the CPU mapping of an imported `udmabuf`
    (1 GiB imports).

  Latency runs from a read's submission until its extent is usable (for the
  landing modes, after the copy). The loop is the diagnosis's. Each process
  runs 4 passes, and the report drops pass 0 as a warm-up. In 33 of 36
  processes, pass 0's slowest read took 10–11 ms, and the median pass 0 ran
  0.21 GB/s below the later passes. That leaves 9 passes
  per row from 3 processes. Before each pass the destination is zeroed.
  After it, the GPU checks all 8 GiB against the pattern.
- `cost`: over 1,000 iterations, times making one 2 MiB extent of an
  existing `memfd` GPU-visible (create, import, map) and tearing it down
  (`cuMemFree`, `cuDestroyExternalMemory`, `close`). It then runs 200
  allocate/free cycles of 256 MiB, a teardown with the CPU side closed first,
  and a forked child (no CUDA) writing through its own mapping of the `memfd`.

## Results

### Export and mapping (feasibility, `s1`)

| Allocation | dma-buf export | `mmap` | Direct I/O into the mapping |
| --- | --- | --- | --- |
| device VMM, 1 handle / 2 MiB handles | `INVALID_VALUE` (both flags) | — | — |
| device VMM, 128, 129, 4,096 × 2 MiB handles | `INVALID_VALUE` (no flag) | — | — |
| device VMM with POSIX-fd handle type | `INVALID_VALUE` | — | — |
| device VMM, `gpuDirectRDMACapable` | `cuMemCreate`: `INVALID_DEVICE` | — | — |
| `cuMemAlloc` | `INVALID_VALUE` | — | — |
| host VMM (`HOST_NUMA`), 1 handle / 2 MiB handles | `INVALID_VALUE` | — | — |
| `cuMemAllocHost` | success | success, `VmFlags: … pf io …` | `EFAULT` for `O_DIRECT` `pread`/`pwrite`, io_uring `READ`, registered buffers; buffered `pread` works (0 bad words) |
| device VMM's POSIX shareable fd | (`/dev/nvidiactl`) | `EINVAL` | — |

CPU bandwidth (`s1b`, 64 MiB, median of 10, GB/s): the `cuMemAllocHost`
dma-buf mapping read 28.3, wrote 45.4 and took `memcpy` in at 22.1 on one
thread (116.1, 72.7 and 64.1 on eight). `malloc` measured 31.3, 38.1 and 22.6
(118.7, 74.9 and 61.3), and host VMM 28.2, 44.7 and 24.3 (117.4, 72.5 and
61.6). The dma-buf mapping is cached, as the header says. It is host memory,
though, and direct I/O cannot pin it.

### Microkernels, GB/s

Median of the three per-process medians (range of per-process medians); 10
timed launches per process.

| Arm | `scan16` | `write16` | `write-per-block` | `reread-4m` | `reread-16m` | `reread-64m` | `rand-all` | `rand-2m` | `rand-8m-span` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `dvmm` | 238 (238-239) | 197 (197-197) | 6.3 (6.3-6.3) | 1979 (1926-1980) | 1977 (1944-1980) | 382 (371-386) | 282 (279-282) | 208 (206-209) | 389 (389-389) |
| `malloc` | 243 (234-245) | 197 (197-197) | 6.3 (6.3-6.3) | 1954 (1935-1978) | 1924 (1923-1980) | 379 (375-379) | 282 (280-282) | 207 (205-208) | 389 (388-389) |
| `hvmm` | 243 (240-244) | 216 (215-216) | 2.6 (2.4-2.6) | 248 (248-256) | 240 (239-242) | 241 (240-248) | 214 (208-215) | 210 (208-211) | 225 (219-227) |
| `udmabuf-4k` | 236 (234-245) | 214 (213-215) | 2.4 (2.1-2.5) | 226 (224-233) | 232 (229-234) | 234 (232-240) | 2.6 (2.6-2.7) | 3.1 (3.1-3.2) | 212 (209-215) |

`write-per-block` medians: 97 µs (`dvmm`), 96 (`malloc`), 234 (`hvmm`), 256
(`udmabuf-4k`). The first three rows reproduce the diagnosis. The imported
`udmabuf` falls in the diagnosis's "host-located" family, re-reads at DRAM
rate. With 4 KiB pages it also collapses on random access, as
`cudaMallocHost` did there (3 GB/s). All four kinds read back unchanged after
the tests (0 bad words).

**Coherence.** For `hvmm` and `udmabuf-4k`, 60 rounds of 4 MiB each (3
processes × 20) saw 0 stale or wrong words in every direction:
- DMA after an L2 prime.
- CPU stores after an L2 prime.
- GPU stores read by the CPU.

L2 does not keep these kinds' lines, so the L2 part of the check is weak
here. It would matter for a cached kind, and none could be mapped.

### Restores

8 GiB in 2 MiB direct reads; median of 9 passes (range of the three
per-process medians); every pass 0 bad words.

| Path | In flight | GB/s | p50 µs | p99 µs |
| --- | ---: | ---: | ---: | ---: |
| in place into `hvmm` (D-034) | 2 | 13.291 (13.279-13.342) | 310 (310-311) | 434 (402-440) |
| `hvmm` landing, copy engine to `dvmm` | 2 | 13.184 (13.154-13.198) | 348 (347-349) | 546 (543-547) |
| `hvmm` landing, SM copy kernel to `dvmm` | 2 | 13.192 (13.181-13.302) | 339 (339-339) | 497 (489-561) |
| in place into `udmabuf-4k` | 2 | 13.146 (13.022-13.178) | 308 (308-309) | 454 (437-566) |
| in place into `hvmm` (D-034) | 4 | 14.941 (14.936-14.957) | 558 (558-558) | 638 (617-640) |
| `hvmm` landing, copy engine to `dvmm` | 4 | 14.918 (14.817-14.959) | 597 (597-597) | 805 (684-897) |
| `hvmm` landing, SM copy kernel to `dvmm` | 4 | 14.953 (14.940-14.959) | 587 (587-587) | 666 (665-666) |
| in place into `udmabuf-4k` | 4 | 14.697 (14.658-14.704) | 558 (558-558) | 656 (640-706) |
| in place into `hvmm` (D-034) | 8 | 14.952 (14.933-14.953) | 1116 (1116-1116) | 1217 (1193-1241) |
| `hvmm` landing, copy engine to `dvmm` | 8 | 14.955 (14.942-14.967) | 1155 (1155-1155) | 1505 (1503-1564) |
| `hvmm` landing, SM copy kernel to `dvmm` | 8 | 14.956 (14.947-14.956) | 1145 (1145-1146) | 1289 (1258-1372) |
| in place into `udmabuf-4k` | 8 | 14.702 (14.681-14.725) | 1117 (1116-1117) | 1203 (1203-1224) |

The landing path's added median latency is 38–39 µs (copy engine) and
29 µs (SM kernel) at every depth. Its bandwidth equals in-place at four and
eight in flight, and is 0.8% lower at two. Eight in flight adds latency and
no bandwidth, as D-034 found. Reads into the 4 KiB-page `udmabuf` run
1.6–1.7% slower at four and eight. The cause was not examined; one plausible
explanation is pinning 512 pages per read against host VMM's larger pages.
The `udmabuf` path also kept pass 0's 10–11 ms slowest read in 24 of its 27
counted passes; no other path had one after pass 0. That was not examined
either.

In `s1` one of 36 negative controls reported 0 instead of 1 (`land-ce`, two
in flight). The control wrote its corrupt word with `cuMemsetD32` on the
legacy stream, which for device memory may return before the write lands,
and the check ran on a non-blocking stream that does not order with it.
Only the control raced. Each pass's own check clears and reads on the one
stream after the copies have synchronized.
The fixed control (`cuMemsetD32Async` on the checking stream) caught the
word in all 6 reruns (`s1b`: `land-ce` and `hvmm-inplace`, 3 each).

### `udmabuf` lifecycle and cost

- Per 2 MiB extent (1,000 iterations, p50/p99 µs):
  - To make it visible: create 23.1/26.5, import 23.8/31.2, map 41.0/46.0,
    about 88 µs in all.
  - To tear it down: `cuMemFree` 21.1/23.2, destroy 10.7/12.6, `close`
    10.2/10.7.

  For comparison, device-VMM create, map and access totals about 90 µs
  (D-033).
- The dma-buf fd stays open after import; the caller owns it.
- The import keeps the pages alive. With the dma-buf fd, the CPU mapping and
  the `memfd` all closed, the GPU still read 256 MiB exactly (0 bad words).
- A forked process without CUDA mapped the `memfd`, verified it and wrote
  a word, and the GPU saw that write.
- The 200 allocate/free cycles of 256 MiB left the fd count and `Shmem`
  unchanged. `MemAvailable` fell by 363 MiB over the loop (and
  `cuMemGetInfo`'s free by 362). Whether this is a leak, the driver's
  one-time growth, or freed 4 KiB pages parked on the per-CPU page lists,
  which `MemAvailable` does not count (RE-024), was not determined.

## Limitations and what was not determined

- **One driver.** Device-memory export was tested only on 580.178.04. For
  the packaged 610.57.04 (and 595/590), only the open kernel module's source
  was read. It still maps dma-bufs by PFN, so direct I/O still cannot pin
  them. Whether its user-space driver exports GB10 device memory at all, or
  reports attribute 152, was not tested, because installing a driver changes
  the platform.
- **The PFN-map conclusion for device memory is inferred.** The `EFAULT`s
  were measured on the one NVIDIA dma-buf that maps (`cuMemAllocHost`). The
  device path would go through the same `nv_dma_buf_mmap` code.
- **Export variants not tried:** the wide exports, `cuMemAlloc` or
  `cuMemAllocHost` with the PCIe flag; `CU_MEM_HANDLE_TYPE_FABRIC` or `CU_MEM_LOCATION_TYPE_HOST`
  allocations; stream-ordered pool (`cuMemAllocAsync`), managed,
  `cuMemHostAlloc` and `cuMemHostRegister` memory. The headers do not list
  pool or managed memory as exportable, they gate every device-memory export
  on the attribute that reads 0, and the host kinds are host-located.
- **Not tried:** a kernel module or P2PDMA route that hands the NVMe
  controller the device pages' physical addresses; SPDK/raw NVMe (the SSD
  holds the mounted root); shmem THP- or hugetlb-backed `udmabuf` beyond the
  one refused hugetlb import. The larger pages might fix the imported
  memory's random access. By the diagnosis's host-VMM results (2 MiB
  handles, still uncached), they would probably not make L2 keep its lines;
  that is untested.
- **Copy interference.** The copy's effect on kernels running at the same
  time, and theirs on the restore, were not measured.
- **Kernels only.** No model or GGML product was run on the `udmabuf`
  import; its microkernel family already rules it out.

## Reproduction

On `spark`:

```bash
d=~/.local/share/llmp/dmabuf-20260927 && mkdir -p $d/{bin,raw,scratch}
# copy dmabuf_probe.cu, build.sh and session.sh into $d/bin, then:
cd $d && bin/build.sh bin/dmabuf_probe
bin/dmabuf_probe create scratch/pattern-16g.bin 16
bin/session.sh bin/dmabuf_probe scratch/pattern-16g.bin raw/s1   # needs sudo
```

Then, with `raw/s1` copied to the workstation:

```bash
python3 -B docs/experiments/dmabuf-direct/summarize.py raw/s1 \
  --json docs/experiments/dmabuf-direct/results.json
```
