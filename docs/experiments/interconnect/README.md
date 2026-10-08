<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Spark direct-link baseline — 2026-09-21

M0 experiment on the owner's two-node `sparky` cluster, after NVIDIA Sync
configured the direct DAC connection. The owner's approximately **185 Gb/s**
Sync result is a comparison supplied by the owner; its raw output, workload,
and settings were not available to this experiment.

**Completed:** 78 host-buffer test pairs and 27 two-GPU NCCL runs. Combined
host writes reach **184.76 Gb/s**; large out-of-place NCCL SendRecv reaches
**22.35 GB/s per rank** and AllReduce **22.20 GB/s**. All supported NCCL
result checks passed. This completes the M0 interconnect baseline; M4 still
owns sharded-model testing, and M8 pressure, cancellation and failure testing.

## Topology and conditions

Both active interfaces share the **right-hand physical QSFP port**; they are
separate PCIe paths to that port, not two independent 200 Gb/s cables. This
matches [NVIDIA's Spark port mapping](https://docs.nvidia.com/dgx/dgx-spark/spark-clustering.html).
Both nodes expose PCIe Gen5 **32 GT/s ×4** for each path, measured from sysfs.
The unused `f0` interfaces remained down. These names and addresses are the
measured deployment, not application topology embedded in llmpalooza.

| Label | RDMA device | Network interface | `spark` | `spark-b` |
| --- | --- | --- | --- | --- |
| H0 / domain 0 | `rocep1s0f1` | `enp1s0f1np1` | `10.100.208.2` | `10.100.208.1` |
| H1 / domain 2 | `roceP2p1s0f1` | `enP2p1s0f1np1` | `10.100.209.2` | `10.100.209.1` |

All addresses are `/24`. Ethernet MTU is 1500; the measured RC QP MTU is
**1024 bytes**, with RoCE v2 IPv4 GID index 3. Link rate reports 200 Gb/s.
Neither Ethernet MTU nor drivers, firmware, system packages, security settings,
CPU affinity, or power policy were changed. No reboot was needed: active RDMA
ports, successful transfers and two-rank NCCL initialization establish that
this configuration was recognized.

Both nodes: Ubuntu 24.04.5, kernel `7.0.0-1019-nvidia`, GB10,
driver `580.178.04`, ConnectX firmware `28.45.4028`. No inference workloads
were running. This is a synthetic idle-system baseline, not contention,
thermal endurance, model sharding or fault-recovery validation.

## Host-buffer results

Three-run medians, 8 MiB transfers. H0/H1 columns run each interface alone;
the combined column sums two concurrent streams within each repetition and
then takes the median. Decimal Gb/s; bidirectional rows sum both directions.

| Payload operation/direction | H0 alone | H1 alone | Both paths together |
| --- | ---: | ---: | ---: |
| Write, `spark-b` → `spark` | 108.99 | 109.01 | **184.76** |
| Write, `spark` → `spark-b` | 109.00 | 109.02 | **184.76** |
| Read, `spark` → `spark-b` | 80.84 | 80.84 | **150.10** |
| Read, `spark-b` → `spark` | 80.84 | 80.84 | **150.10** |
| Writes, both directions summed | 212.96 | 213.16 | **369.28** |

Combined forward-write repetitions span 184.76–184.77 Gb/s; reverse writes
all round to 184.76. Read totals span 150.10–150.11 in either direction.
Bidirectional totals span 369.28–369.30. The bidirectional total is about
184.64 Gb/s each way, not 369 Gb/s one-way capacity. Client launches were
within 0.376 ms and process completions within 4.971 ms for the paired tests;
the independent six-second measurement windows are substantially overlapping.

Small-message latency, medians of three perftest median RTT/2 values:

| Payload | H0, µs | H1, µs |
| --- | ---: | ---: |
| 8 B | 1.39 | 1.40 |
| 64 B | 1.50 | 1.50 |
| 1 KiB | 2.34 | 2.34 |
| 64 KiB | 8.42 | 8.41 |
| 2 MiB | 157.61 | 157.60 |

The measured combined write result agrees closely with the owner's Sync
reading. It establishes a useful host-buffer baseline for explicit object
transfer; it does not price model load/restore or prove a sharding strategy.

## GPU communication results

Three-run medians at 512 MiB. These are NCCL **bus bandwidth**, decimal GB/s,
not a sum across ranks. Out-of-place variants:

| Operation | H0 alone | H1 alone | Default, both HCAs | Default range |
| --- | ---: | ---: | ---: | ---: |
| SendRecv | 12.77 | 12.69 | **22.35** | 22.28–22.36 |
| AllReduce | 12.25 | 12.16 | **22.20** | 22.19–22.21 |
| AllGather | 11.73 | 11.86 | **20.40** | 20.33–20.47 |

Default in-place medians are 22.21 GB/s for AllReduce and 21.14 GB/s for
AllGather. AllGather's out-of-place algorithm bandwidth is 40.81 GB/s,
because its reported buffer size includes both ranks' contributions;
that is not 40.81 GB/s of one-way network payload.

Default out-of-place operation latency in microseconds, median [min–max]
of three runs:

| Operation | 32 B | 2 MiB |
| --- | ---: | ---: |
| SendRecv | 41.08 [24.99–41.98] | 142.61 [141.52–418.57] |
| AllReduce | 30.62 [15.96–35.88] | 578.21 [414.46–606.71] |
| AllGather | 28.04 [21.19–36.34] | 358.27 [350.44–388.25] |

Large-transfer throughput is consistent; small and medium operation times
vary substantially in this short run. Do not extrapolate the 512 MiB rate
to expert fetches, per-token collectives or a generation-stall guarantee.

All 27 jobs exited successfully with two ranks on the two expected GB10s and
zero reported out-of-bounds values. There are **369 nonzero payload cases**
across all runs and **612 supported variant checks** (SendRecv out-of-place;
AllReduce/AllGather both variants). Nine additional zero-payload AllGather
rows were validated but excluded from payload aggregates. No swap-in,
swap-out or OOM-kill counter increases occurred on either node across the
measurement session. Final process audits found no benchmark or MPI workers.

## Method and interpretation

Host-buffer tests use the installed `perftest 24.01.0+0.38-1build2` (its binary
reports version 6.20), `rdma-core`/`libibverbs1 50.0-2ubuntu0.2`, ordinary
host allocations and RDMA CM (`-R`). No CUDA buffer option is used.

- Three sweeps per HCA of write bandwidth, read bandwidth and send latency:
  23 powers of two from 2 bytes through 8 MiB, 1,000 iterations per size.
- Three 8-second duration runs per HCA of 8 MiB writes and reads in both
  client/server orientations, plus simultaneous bidirectional writes. One
  second at each end is excluded (`-f 1`), leaving a six-second timed window.
- Repeat those duration arms with both HCAs active together. Each process
  has its own timer; combined values sum the paired, substantially overlapping
  streams from the same repetition, not separate best runs. They are not a
  timestamp-synchronized common-window counter measurement.
- RC, one QP, default depth/CQ settings retained: bandwidth output records TX
  depth 128; duration write CQ moderation is 1, sweep write moderation 100;
  read output records 16 outstanding reads. No queue tuning.
- Perftest values are decimal Gb/s. Read requests travel client→server while
  read payload travels server→client. Bidirectional totals sum both directions.
  Send latency is the tool's half-round-trip measurement, not a separately
  timestamped one-way transit latency. Host perftest is not a payload checksum
  test; NCCL below supplies end-to-end GPU result validation.

NCCL uses one process and one GB10 per host. The unfiltered default is compared
with exact HCA selectors `=rocep1s0f1:1` and `=roceP2p1s0f1:1`. Each arm runs
SendRecv, AllReduce and AllGather three times: 14 requested message sizes from
8 bytes to 512 MiB, factor four, 5 warmup iterations, 20 timed iterations, float data,
validation enabled (`-c 1`), no CUDA graphs. PTX JIT is disabled.
AllGather aligns each rank's contribution to 16 bytes; the requested 8-byte
case therefore becomes a zero-payload operation. Its 13 nonzero measured
sizes start at 32 bytes. The zero row is excluded from latency and bandwidth
claims. This rounding is explicit in the pinned `all_gather.cu`.

NCCL bootstrap and MPI control traffic use management Ethernet `enP7s7`.
MPI uses TCP (`pml ob1`, `btl self,tcp`); NCCL chooses its own data transport.
Timing excludes connection/build/startup costs. Reported NCCL GB/s are decimal.
For two ranks, AllReduce `busbw = algbw`, AllGather `busbw = algbw / 2`, and
SendRecv `busbw = algbw`. SendRecv exchanges messages in both directions;
its printed bandwidth is per rank, not the sum of both directions.
[NCCL test bandwidth definitions](https://github.com/NVIDIA/nccl-tests/blob/b4d5beebca8a76cf01335f724d154b9b9d394d96/doc/PERFORMANCE.md).

**SendRecv's in-place correctness check is unsupported by this pinned test.**
Its source disables that check, so only out-of-place SendRecv is reported.
Both variants of AllReduce/AllGather are checked. Small-message numbers are
operation latency averages, not p99 samples; three-run ranges expose run-to-run
variation but do not establish a long-tail service guarantee.

## Transport and staging evidence

The read-only CUDA probe returns `GPU_DIRECT_RDMA_SUPPORTED=0` and
`DMA_BUF_SUPPORTED=0` on both nodes. Pageable memory access, native host
atomics and mapped host memory support return 1.

NCCL's actual channel logs identify `NET/IB`; default selection includes both
HCAs. Each selected HCA logs GPU Direct RDMA disabled, and channel send/receive
lines have no `/GDRDMA` suffix. Mapped communication allocations are recorded
by `Cuda Host Alloc` at `transport/net.cc:957` and `:1135` (send/receive proxy
connection), and `:679` (shared point-to-point buffers), rather than inferred from the available-device list.

All **54 rank logs** and their before/after snapshots were checked: default
jobs have data channels and increasing transmit/receive/write counters on
both HCAs; filtered jobs have those increases on exactly the chosen HCA and
zero changes on the other. Audited RDMA error/drop/retry counters did not
increase. Management TCP bootstrap is distinct from the RDMA payload path.

Source audit of the pinned NCCL explains the selected path:
`net.cc` allocates mapped host communication FIFOs through
`ncclCudaHostCalloc`, whose `alloc.h` implementation uses
`cudaHostAllocMapped`. It registers those buffers for the network and gives
the GPU their mapped addresses. GPU primitives load/store or reduce/copy
between user buffers and these communication FIFOs. The CPU proxy polls
completion state and posts network requests; it does not copy the payload.
See the pinned [network transport](https://github.com/NVIDIA/nccl/blob/73cf112295c33aee2b895f329f592f2a9b4b0f97/src/transport/net.cc),
[allocator](https://github.com/NVIDIA/nccl/blob/73cf112295c33aee2b895f329f592f2a9b4b0f97/src/include/alloc.h), and
[GPU primitives](https://github.com/NVIDIA/nccl/blob/73cf112295c33aee2b895f329f592f2a9b4b0f97/src/device/prims_simple.h).

This establishes **RDMA on mapped host communication buffers**, with GPU
copy/reduction work. It does not establish direct registration of the user's
CUDA allocation, zero staging, measured copy-byte totals or GPUDirect RDMA.
No CUDA timeline tracing or direct-device-memory transfer experiment ran.
This agrees with [NVIDIA's Spark memory guidance](https://docs.nvidia.com/dgx/dgx-spark-porting-guide/porting/cuda.html).

## Build, provenance and dependency boundary

`pins.json` records exact source revisions, executable/library hashes and all
22 downloaded Debian package identities/hashes. Packages were extracted to
the experiment directory, not installed. Main-run binaries are identical on
both nodes. The build recipe retains the tested commands and explicit targets.
Binary hashes identify this measured build, not a bit-for-bit reproducibility
promise for a future rebuild. A rebuilt artifact with a different hash must
have its source/toolchain and resulting identity reviewed and recorded before
updating the preflight pins; the harness will not silently accept a substitute.

| Component | Pin / role | License and category |
| --- | --- | --- |
| NCCL | `v2.30.7-1`, `73cf112295c33aee2b895f329f592f2a9b4b0f97` | External measurement library; upstream Apache-2.0 with BSD/file-specific notices, preserved in external source checkout |
| nccl-tests | `2.20.0`, `b4d5beebca8a76cf01335f724d154b9b9d394d96` | External measurement tool; upstream BSD-3-Clause |
| Open MPI | Ubuntu `4.1.6-7ubuntu2` | Extracted measurement launcher/library; upstream BSD-style plus component notices in package copyright |
| perftest | Ubuntu `24.01.0+0.38-1build2`, reported 6.20 | Existing measurement tool; package copyright says BSD-MIT or GPL-2 for upstream, GPL-2 packaging |
| RDMA, PMIx, libevent and other extracted/system support | Exact package pins in manifest; installed RDMA packages above | Tools/platform dependencies under D-017; retain each package's copyright notices |
| NVCC / CUDA runtime | Installed 13.0.88 compiler / CUDA 13.0 | NVIDIA toolkit/platform terms, existing target installation |
| GCC / GNU libraries | Installed g++ 13.3.0, glibc 2.39 | Compiler/system dependencies under D-017; not incorporated project implementation |

CPU target is `armv8.2-a`, GPU target is native `sm_121` only. External NCCL
and its tests use their C++17 build dialect; the llmpalooza-authored capability
probe uses C++23. The NVIDIA Spark [playbook](https://github.com/NVIDIA/dgx-spark-playbooks/tree/main/nvidia/nccl)
provided the NCCL version/build reference; no downloaded setup script was
executed. Open MPI's C++ compatibility library is explicitly linked for these
Ubuntu headers. The relocated environment reported its optional PMIx
compression plugin unavailable; all main jobs still initialized and passed
their supported result checks. Initial compiler attempts and the two-rank
pilot were excluded. The two small native nccl-tests support objects were
rebuilt with the explicit CPU target before any main NCCL run.

These are experiment tools, not a choice to incorporate NCCL/MPI/perftest into
llmpalooza's core or a new runtime dependency policy. This work changes no
load-bearing constraint and requires no new architectural decision.

## Reproduction and evidence

Run the native recipe on `spark`, with this directory's files copied outside
the repository to `/home/pmeenan/.local/share/llmp/interconnect/recipe/`.
`env.sh` documents the measured absolute experiment root. Transfer `mpi-root`,
`nccl-runtime`, `nccl-tests-runtime`, the probe, `env.sh`, `snapshot.py`, and
`orted-wrapper.sh`, `audit.py` and `pins.json` to the same root on `spark-b`; place the support
scripts at the root on both hosts. SSH host verification stays enabled.
Source checkouts, packages and build objects can remain on the build node.

```bash
bash /home/pmeenan/.local/share/llmp/interconnect/recipe/build.sh
```

From the workstation, in the repository, choose a fresh external output root;
each phase refuses to reuse an existing output directory. The harness accepts
host, IP and HCA overrides for another configured deployment.

```bash
python3 docs/experiments/interconnect/run-host.py --phase sweep --out /tmp/llmp-interconnect/host-sweep-main
python3 docs/experiments/interconnect/run-host.py --phase large --out /tmp/llmp-interconnect/host-large
python3 docs/experiments/interconnect/run-host.py --phase combined --out /tmp/llmp-interconnect/host-combined
python3 docs/experiments/interconnect/run-nccl.py --mode main --out /tmp/llmp-interconnect/nccl-main
python3 docs/experiments/interconnect/analyze.py /tmp/llmp-interconnect --out /tmp/llmp-interconnect/aggregate.json
```

Raw receipts, samples, diagnostic attempts, node snapshots and copied NCCL
logs are frozen on the workstation at
`/home/pmeenan/.local/share/llmp/interconnect-results/2026-09-21/`.
Its `raw-manifest.json` indexes 499 files (28,464,513 bytes), SHA-256
`61bec87ab002402bbd9d948176225d3829d2b9bda607771f482df3f6ae8f233b`.
The working copy remains at `/tmp/llmp-interconnect/`. Native builds and
original NCCL logs remain at `/home/pmeenan/.local/share/llmp/interconnect/`
on each Spark.
No models, prompts, keys or conversation data are part of this experiment.
Only reusable recipes, provenance and aggregate results belong in Git.

[aggregate.csv](aggregate.csv) retains the full nonzero message-size summary
(median/min/max and three-run count), including supported in-place variants.
Regenerate it with the analyzer's `--csv-out` option. Copy the per-node NCCL
log directories into the raw root's `nccl-logs/spark` and `nccl-logs/spark-b`
before running `verify-transport.py RAW --out /tmp/transport-validation.json`.
The analyzer rejects missing/duplicate repetitions, wrong message grids,
nonfinite values, mismatches and inconsistent receipts; it does not silently
turn partial runs into results.

One original host-duration case, `ib_write_bw-reverse-h1-r0`, overlapped the
short native support-object rebuild. A clean isolated repeat with identical
arguments replaced it after the NCCL suite, again measuring 109.02 Gb/s.
The original logs/receipt remain under `excluded-build-overlap` and the
replacement history in `build-overlap-exclusion.json` in external raw storage.
The reported 78 host pairs exclude that original measurement.

Handoff: benchmarks and capability probes ran on both Sparks; aggregation,
transport verification and 15 regression checks ran on the workstation.
Pinned source and runtime identities were audited before NCCL and after the
suite. All supported correctness checks and the 54-rank transport audit
passed; both nodes ended with no benchmark/MPI workers. No sharded model,
failure injection, CUDA timeline profiling or direct GPU-memory RDMA test
was run. The working tree remains uncommitted for human review.

One excluded pilot using `ib_write_bw -R -a -n 1000 --perform_warm_up` timed
out at 90 seconds on both hosts. Removing the optional warmup flag completed
the sweep; the duration baseline instead discards its first/last second.
The timeout is recorded as an option-combination observation, not a diagnosed
network failure. Pilot data and initial build attempts are excluded from the
reported baseline.

Independent review (2026-09-21): a separate agent reviewed the interconnect
recipes, provenance, source-supported staging explanation, aggregate report
and affected AGENTS/architecture/plan/RE-005 changes against the previously
reviewed tree. Duplicate-repetition acceptance, incomplete failure receipts
and runtime-library identity checks were corrected. Fifteen workstation
regression tests pass, including malformed result sets, unsupported SendRecv
checks, AllGather's zero-count case and mocked timeout/cleanup failures.
All 78 accepted host pairs and 27 NCCL jobs reconcile with the final aggregate,
all 877 CSV summaries and every printed result-table value. The clean host
replacement and retained exclusion record were checked. All 54 rank logs,
per-HCA counter sets, source/runtime identities and refreshed final process
and VM audits agree with the report. The frozen manifest and hashes of all
499 archived evidence files were verified. No additional transfer benchmarks were
run by the reviewer. `git diff --check` passed; no outstanding correctness
finding remains within this baseline's stated scope. Combined host rates
remain sums of separately timed overlapping streams; exact copy volumes,
latency tails and M4/M8 execution/failure behavior remain unproven. Changes
remain uncommitted for the human commit gate.
