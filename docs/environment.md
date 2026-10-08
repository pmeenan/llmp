<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Development and target environments

The owner's workstation and DGX Spark inventories, and the platform follow-ups
measured on them during M0. These are environment facts, not application
configuration: host names, addresses and paths here are never baked into
jitLLM (D-023). [architecture.md](architecture.md) relies on the measured
results through the decisions they informed (D-032–D-034, D-054, D-065);
the linked experiment reports hold methods, pins and limitations.

## Development host baseline

Captured 2026-09-20 on the reference workstation, read-only. The brief asked
for this before choosing pins; it is a baseline, not a pin.

| Item | Observed |
| --- | --- |
| OS | Ubuntu 24.04.5 LTS (noble), x86-64 |
| Kernel | 7.0.0-31-generic (Ubuntu-packaged) |
| CPU / RAM | 16 logical CPUs, 62 GiB |
| NVIDIA driver | 595.91.07, open kernel module |
| GPU | NVIDIA GeForce RTX 3080 Ti, 12 GiB, compute capability 8.6 (not a GB10 and not a Spark proxy; a secondary discrete target since D-082, whose `gpu-discrete` tests run on it when asked). On 2026-09-27 (driver 595.91.07): not integrated, VMM, 2 MiB device-local and host NUMA node 0 granularity; its PCI NUMA node reads -1 |
| CUDA toolkit | not installed (no `/usr/local/cuda*`, no `nvcc`) |
| Clang / LLD / clang-tidy | not installed |
| GCC | 13.3.0 (system) |
| CMake | not installed |
| Ninja, clang-format | present only via `~/depot_tools` (Chromium tooling; not a project pin) |
| mise | not installed |
| Docker | 29.8.1 |
| Python | 3.12.3 |

Implication for M1: the workstation needs LLVM, CMake, the CUDA toolkit, and
mise provisioned by the project's own setup path. A local NVIDIA GPU means
some CUDA smoke tests can run on the workstation before a Spark, but Spark
capabilities (VMM behaviour on GB10, GDS mode, unified-memory accounting) are
only measurable on a Spark. The Spark-side inventory follows.

### Toolchain smoke follow-up (2026-09-21)

The [M0 smoke](experiments/toolchain-smoke/README.md) passed native C++23,
AArch64 cross CPU/CUDA execution on `spark`, and a native Spark fallback.
D-032 pins LLVM/LLD 22.1.8, the measured GCC/libstdc++/glibc components
(the C++ runtime is now D-060's statically linked GCC 16.2),
NVCC/cudart 13.4.92 (Toolkit 13.4.2), and the hashed target snapshot.
Host and CUDA translation units both use C++23, including an `if consteval`
host/device probe; the installed 13.0 comparison has a dialect limit (RE-001).
GB10 was validated with `sm_121` and PTX JIT disabled. The workstation
compiler SDK was extracted to scratch; the baseline above is historical,
and system compiler defaults and drivers were not changed. Full M1
provisioning is still pending.

On 2026-09-22 the SDK added the same-version `libclang-rt-22-dev` packages
for x86-64 and ARM64. Clang ASan/UBSan passed the
[CPU-only task/completion experiment](experiments/async-model/README.md)
natively on the workstation and cross-built on `spark`; pins and reproduction
are in the smoke manifest and experiment report.

### M1 SDK provisioning (2026-09-23)

With the owner's approval, mise 2026.9.12 was installed as a user-level
binary at `~/.local/bin/mise` on the workstation and on `spark`, from the
GitHub release archives. Each archive matched the SHA-256 in the release's
`SHASUMS256.asc`, whose signature verified against the mise release key
(`24853EC9F655CE80B48E6C3A8B81C9D17413A06D`). No system package or shell
file changed. D-070's SDK lives under each host's
`~/.local/share/jitllm/sdk/`, with its cache in `~/.cache/jitllm/`. The
native Spark fallback runs from a copy of the working tree at
`spark:~/src/jitLLM`. The workstation already had
`binutils-aarch64-linux-gnu` 2.42-4ubuntu2.10, which the cross-built GCC
runtime needs. The baseline table above is the historical 2026-09-20
snapshot.

## Target nodes (DGX Sparks)

Captured 2026-09-20 over SSH, read-only, no sudo. Both nodes are identical in
software. These names are the owner's environment, not application
configuration (D-023). `spark` (master, also `spark-a`) and `spark-b` resolve from the
workstation and from each other, and SSH is configured in both directions
between the nodes (verified 2026-09-20 with a hop from each to the other).
That initial traffic used management Ethernet. The owner configured the
direct DAC cluster on 2026-09-21; its current link inventory follows the
historical platform snapshot below.

| Item | `spark` (hostname `spark-c4e2`) and `spark-b` (hostname `spark-56f5`) |
| --- | --- |
| Platform | NVIDIA DGX Spark, DGX OS 7.5.0 base with OTA 7.6.0 applied 2026-09-20 |
| OS / kernel | Ubuntu 24.04.5 LTS, kernel 7.0.0-1019-nvidia, aarch64 |
| CPU / RAM | 20 logical CPUs; 121 GiB visible of 128 GB unified memory, about 118 GiB free at idle |
| GPU / driver | NVIDIA GB10, compute capability 12.1; driver 580.178.04, open kernel module. `nvidia-smi` reports no discrete memory total (unified memory, see D-004) |
| CUDA | Toolkit 13.0 (`nvcc` V13.0.88, package cuda-toolkit-13-0 13.0.3-1) at `/usr/local/cuda-13.0` |
| Storage | One Samsung NVMe (MZALC4T0HBL1), 3.7 TB, root filesystem, about 3.5 TB free. No separate data volume |
| GDS / cuFile | GDS 1.15.1.6, libcufile 2.12, gds-tools installed. `use_compat_mode: true`, `allow_compat_mode: true`, `nvidia_fs` not loaded, cuFile RDMA library not loaded. Matches the compatibility-mode-only constraint in D-004 |
| RDMA / interconnect | Initial pre-DAC snapshot: `mlx5_core`, `mlx5_ib`, `ib_core`, `ib_uverbs`, `rdma_cm` loaded; rdma-core 50.0; no devices under `/sys/class/infiniband` or ConnectX netdevs listed. Superseded by the link inventory below |
| NCCL | No `libnccl2` package installed |
| glibc | 2.39 |
| Distro toolchain | clang 18.1.3, gcc 13.3.0, cmake 3.28.3, python 3.12.3, git 2.43, docker 29.6.2, nvidia-container-toolkit 1.20.1. No ninja, no mise. Distro defaults, not project pins |
| Privileges | Passwordless sudo is configured for the SSH user (owner-stated); nothing in this inventory used it |

Implications for the M0 spikes: the target driver is 580.178.04, so the cross
toolchain's CUDA toolkit pin has to stay within that driver's compatibility
range. The installed toolkit remains 13.0; the later D-032 smoke validated
extracted 13.4.2 components using native GB10 code on the existing R580
driver through CUDA minor-version compatibility. New driver-dependent
features and PTX/JIT paths still need separate validation. The workstation
driver (595.91.07) is newer than the targets', so a kernel that runs locally is not proof it runs on Spark. The I/O spike has one
NVMe and one filesystem to work with, shared with the OS. Direct-link
performance was subsequently validated in the M0 baseline below.

### Direct DAC cluster follow-up (2026-09-21)

The owner reports cluster **`sparky`**, two directly connected devices.
Read-only SSH checks on both nodes confirmed the supplied addresses, link
state, local routes and RDMA-device mappings. These are environment inventory,
not hardcoded application topology or a settled jitLLM configuration format.

| SSH alias | Network interface | IPv4 address | RDMA device / port |
| --- | --- | --- | --- |
| `spark-b` | `enp1s0f1np1` | `10.100.208.1/24` | `rocep1s0f1/1` |
| `spark-b` | `enP2p1s0f1np1` | `10.100.209.1/24` | `roceP2p1s0f1/1` |
| `spark` | `enp1s0f1np1` | `10.100.208.2/24` | `rocep1s0f1/1` |
| `spark` | `enP2p1s0f1np1` | `10.100.209.2/24` | `roceP2p1s0f1/1` |

All four interfaces report `UP`, **200,000 Mb/s** link rate and **MTU 1500**
(9000 since 2026-09-28: [jumbo frames](#jumbo-frames-on-the-direct-link-2026-09-28));
their RDMA ports report `ACTIVE / LINK_UP`. Each node's route to its peer's
address selects the corresponding interface and local source address.
The other two ConnectX netdevs (`enp1s0f0np0`, `enP2p1s0f0np0`) are down.
Both active interfaces map to the same right-hand physical QSFP port through
separate PCIe Gen5 ×4 paths (measured 32 GT/s ×4 on each node). The
[NVIDIA port map](https://docs.nvidia.com/dgx/dgx-spark/spark-clustering.html)
explains these two functions; they are not separate 200 Gb/s cables.

Both hosts have `rdma-core 50.0-2ubuntu0.2` and
`perftest 24.01.0+0.38-1build2`; `ib_write_bw --version` reports 6.20.
`ib_write_bw`, `ib_read_bw`, `ib_send_lat` and `ibv_devinfo` are available.
No host `libnccl2` package was reported by `dpkg-query` in the initial link
inventory; the later baseline used a pinned native build in external scratch.
These inventory checks ran without sudo
and changed no network or driver settings. No transfer benchmark, NCCL test,
end-to-end data validation or GPUDirect RDMA validation was performed in
this inventory update.

The subsequent [M0 baseline](experiments/interconnect/README.md) completed
78 host-buffer test pairs and 27 two-GPU NCCL runs on 2026-09-21, without a
reboot or network/driver changes. Three-run medians: each HCA alone reaches
about 109 Gb/s for 8 MiB host writes; together they reach **184.76 Gb/s**
in either direction (consistent with the owner's approximately 185 Gb/s
Sync result). Combined reads reach **150.10 Gb/s** with default queues.
Bidirectional writes total 369.28 Gb/s, about 184.64 Gb/s each way.
Small 8-byte send latency is 1.39–1.40 µs median RTT/2.

Native `sm_121` NCCL 2.30.7 and pinned nccl-tests 2.20.0 reached
**22.35 GB/s SendRecv**, **22.20 GB/s AllReduce** and **20.40 GB/s AllGather
bus bandwidth** at 512 MiB, default HCA selection, out-of-place medians.
All supported result checks passed; SendRecv's in-place check is unsupported
and excluded. A rounded-to-zero AllGather case is excluded from payload
metrics. Small/medium operation latency varied materially across repeats;
the report records size sweeps and ranges rather than extrapolating peak
bandwidth to generation latency.

All 54 rank logs and per-HCA counter snapshots confirm the selected RDMA
paths. CUDA reports GPUDirect RDMA and DMA-BUF support as zero. NCCL channel
and allocation logs, checked against its pinned source, establish **mapped
host communication buffers**: GPU kernels copy/reduce between user buffers
and those buffers, and the NIC performs RDMA on them. This is not direct
registration of user CUDA allocations or zero staging. Exact copy-byte
counts and CUDA timeline tracing were not measured. Sharded execution is M4
work; asymmetric memory pressure, cancellation and failure tests remain M8 work.

Independent inventory review (2026-09-21): a separate agent repeated the
read-only address, link, route, RDMA mapping and installed-tool checks on
both nodes and found the inventory consistent. That pre-benchmark review left
measured throughput, aggregate link capacity and GPUDirect support unproven;
M0 baseline testing and M4/M8 execution/failure testing remain distinct.
No settings changed or transfer benchmarks ran during this review.

### VMM microbench follow-up (2026-09-21)

Three runs on `spark` / GB10, driver 580.178.04, using the D-032 cross SDK,
measured device-local pinned VMM allocations with no export handles.
Minimum and recommended granularity were both **2 MiB**. Host-call latency
ranges below are per-run medians, in microseconds; they exclude SSD I/O.

| Extent | Create | Map | Set access | Unmap | Release |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2 MiB | 48.72–53.41 | 0.50–0.54 | 35.94–37.70 | 46.66–63.21 | 26.61–27.04 |
| 8 MiB | 178.57–196.75 | 0.72–1.26 | 72.10–81.10 | 116.77–156.06 | 43.62–46.66 |
| 32 MiB | 797.66–925.62 | 4.40–5.06 | 251.44–261.12 | 321.18–327.59 | 134.56–142.52 |
| 128 MiB | 3384.05–3811.23 | 5.34–5.63 | 804.44–835.35 | 985.80–1010.94 | 449.96–472.11 |

These are idle values; the [report](experiments/vmm-microbench/README.md)
summarizes p95/max and concurrency measurements, with source/build provenance,
commands, and limitations; raw output stays outside Git. All 3,600 timed calls
with independent background kernels returned while their completion events
remained pending.
This does not prove no GPU stalls or model-throughput impact. At 128 MiB,
release medians rose to 618–651 µs with background work.

Reserving 1 GiB of virtual addresses did not change observed free memory.
Creating sixteen 64 MiB handles reduced free memory by about 1034 MiB;
unmapping them while retaining the handles left that footprint intact.
Every word survived remapping and verification. Releasing the handles after
unmapping recovered the allocation, with 4–5 MiB baseline drift in the
system-level snapshots. A deliberate corruption verified the check itself.

D-033 starts with 2 MiB independent physical extents, compatible backing
handoff to waiting admitted loads, and no standing unused-handle cache.
Live useful contents remain resident until policy reclaims them. Granularity
is queried, not baked into core identities or on-disk formats; read batches
can span extents. SSD and model measurements may revise this initial policy.

### I/O path follow-up (2026-09-21)

The [M0 comparison](experiments/io-path/README.md) selects **regular files,
direct I/O, and GPU-accessible host-backed VMM** (D-034), amending D-004's
mandatory staging copy. On the Spark's Samsung PCIe 5.0 ×4 SSD, the native
io_uring path measured 14.903–14.968 GB/s with four 2 MiB reads in flight;
the 180-second run sustained 14.962 GB/s without a sustained thermal decline.
GPU scanning of host VMM matched device VMM at about 242 GB/s. CPU submission
and completion work remains; CPU payload copies and a separate staging copy
are absent from the selected path. Capability checks and DMA evidence are
retained in the report, not inferred from unified memory alone. The scan
re-read nothing; D-081 (2026-09-27) moved weights and state to device VMM
behind a host-VMM landing zone, because the GB10's L2 does not cache
host-located memory (RE-022).

Start with two 2 MiB requests for latency-sensitive loads and up to four for
bulk reads: 4–8 MiB of catalog-charged destination backing, with no extra
staging allocation on this path. A device-VMM fallback needs its own bounded,
charged DMA staging buffers; under D-081 that is the selected path, and its
landing zone is 2 × depth 2 MiB extents (8–16 MiB). Cold/warm OS-cache measurements are separated;
buffered full-file reads under 100 GiB of held memory forced file-cache
reclamation, whereas direct reads kept file cache empty. Concurrent memory
scans lost about 10% throughput with the in-place path; one physical budget
also means shared bandwidth. The scan is not a GGML/model performance proof.

The SSD's observed interrupt-coalescing feature is a separate latency tuning
point, tested with the original value restored afterwards. Raw block,
NVMe passthrough, and SPDK were not timed because the only drive holds mounted
root; no raw performance advantage is claimed. M2 still validates actual
GGML pointers/kernels and cancellation/registration/reclaim lifetimes; M6
settles mixed read/write scheduling and spill retention/write-rate limits.

### Reference-engine follow-up (2026-09-21)

The [pinned llama.cpp container](experiments/reference-setup/README.md) now
runs the Gemma 4 26B A4B UD-Q4_K_M text GGUF on `spark`, with all layers
offloaded and PTX JIT disabled. The 16.95 GB artifact contains 128 experts
per layer, top-8 plus a shared FFN across 30 layers; the report gives exact
expert closures, scales, non-expert bytes, and reference allocations.
Two short synthetic save/restart/restore tests reused all 627 saved tokens
and matched 32 continuation token IDs. This required `--swa-full`: default
windowed retention restored the API counts but re-prefilled the prompt
(RE-004). At context 8192, f16 KV rises from 460 to 1760 MiB with that
workaround. Docker's cgroup statistics/limit do not establish the node's
CUDA occupancy or a validated physical-memory pressure mechanism. No host
baseline settings changed by setup.

The subsequent [A→B→A reference experiment](experiments/reference-aba/README.md)
passed 27 cycles using Gemma as A and MIT Ornith 1.5 Q4_K_M as B. A's
18,339-token continuation reused 18,297 tokens and processed 42 after restore;
all 118 output IDs matched the resident reference. A separate early/late
notebook recall check also matched and returned the correct facts. Ornith's
94-token recurrent/KV state survived unload/reload, reused its prefix and
matched a seven-token continuation. The report includes exact expert closures,
primary/MTP storage accounting, native LRU behavior and durable-save steps.

An 80 GiB verified locked allocation leaves 41.688 GiB of physical capacity;
the matched reference's combined CUDA model/state/compute buffers need
42.603 GiB before host overhead. With cold incoming file caches, median
first-token waits were 21.232 s A→B and 18.304 s B→A with state restored;
full re-prefill returned to A in 25.236 s. Warm-cache restore returned in
4.062 s and live residency in 0.089 s. Each number has three repeats and
an observed range in the report, together with actual block I/O, 622.424 MiB
logical spill, sampled memory and bounded whole-node swap activity.

Default-SWA restore still re-prefilled all 18,339 tokens (RE-004). Those
normal-optimization probes enforce the same one-model policy; simultaneous
normal placement under pressure was not validated. A decode speed also
differs across live full-SWA, restored and default-SWA paths, so the slower
path cannot alone define the generation comparison. The later
[bounded full paging-feasibility study](experiments/paging-feasibility/full-study.md)
is complete: it covers the exact reference trace, longer alternating
Gemma/Ornith requests, four-sequence decode, and a DeepSeek/Qwen library whose
combined storage exceeds one node's memory. Offline replay compares partial
extent retention, eager active-model loading, and whole-model replacement at
matched budgets, with resident state, bounded spill, and recomputation. It
includes actual allocation accounting and explicit storage/overlap scenarios;
these are not measured jitLLM paging or switching speedups.

Qwen's captured-route estimates are conditional: exact prediction equivalence
failed, including between untraced controls. Byte-identical sequence snapshots
also failed to establish continuation correctness. Gemma snapshots omit SWA
history needed after prefix rollback (RE-007); safe coverage checks select
recomputation, and unvalidated large-model spill reuse is modeled conservatively
with recomputation. The earlier matching Gemma continuation above remains a
narrow historical observation; the validated recompute arm supplies the usable
correctness floor. Restore metadata must describe valid context coverage.

The study supports keeping M6 partial retention ahead of M7 expert paging,
without promising one-layer prefetch can hide misses. Actual pager execution,
physical admission safety, and end-to-end latency validation remain runtime
work. The owner accepted switching-benefit and generation-stall targets in
D-036 after this study; those targets are not measured achievements.

### Long-term model store (2026-09-22)

The owner's Synology NAS (`192.168.0.3`, share `llm`; owner-reported: more
than 20 TB free, four bonded 1 Gb/s links, magnetic disks) is the long-term
store for D-054. It is mounted at `/mnt/llm` on the workstation, `spark` and
`spark-b`. This is owner environment, not application configuration. Each
host's `/etc/fstab` gained the entry below; the prior file is kept as
`/etc/fstab.bak-2026-09-22`:

```text
//192.168.0.3/llm  /mnt/llm  cifs  guest,vers=3,uid=1000,gid=1000,iocharset=utf8,file_mode=0777,dir_mode=0777,nofail,x-systemd.automount,x-systemd.mount-timeout=30  0  0
```

The mounts negotiated SMB 3.1.1 with `sec=none`, `soft` and 4 MiB read/write
sizes; automount with `nofail` keeps an absent NAS from blocking boot. The
workstation needed `cifs-utils`; the Sparks already had it. The NAS
advertised NFS v2–v4 but exported nothing; the mounts follow the owner's
existing guest SMB mount of another share on this NAS, which needs no UID
mapping. The permissive modes protect nothing
extra, since any LAN host can read, write or delete as guest; integrity
comes from import verification (D-009, D-054), but nothing on the share is
confidential or safe from deletion. The Sparks' LAN ports (`enP7s7`) negotiate
2.5 Gb/s, yet each client's reads below match one 1 Gb/s link, consistent
with the NAS's bond carrying a client on one member link.

One `dd` sample per case, 4 MiB blocks, on 2026-09-22. Reads used a 4 GiB
random file, cold on each reader (first read on that host, or page cache
dropped):

| Case | Result |
| --- | --- |
| Sequential read, one client (`spark-b`, workstation, `spark`) | 118 MB/s each |
| Both Sparks reading concurrently | 117 MB/s each |
| Workstation and both Sparks concurrently | 82.7 / 71.4 / 70.5 MB/s, about 225 MB/s total |
| Write with `fsync` from `spark`, random data | 77.9 MB/s streaming 4 GiB from `/dev/urandom`; 79.7 MB/s for 2 GiB from tmpfs |
| Write with `fsync` from `spark`, 2 GiB of zeros | 92.9 MB/s |

The owner expects the NAS's magnetic disks to limit aggregate throughput,
but the NAS itself was not instrumented and its cache state was not
controlled, so these reads may not reflect its disks, and the cause of the
three-client limit is not established. Renaming over an existing file works, creating a
symlink fails with `EOPNOTSUPP`, and a file written on `spark` hashed
identically on `spark-b`. Test files were removed.

For D-054's peer replication, an 8 GiB random file was copied from `spark`'s
SSD to `spark-b`'s over the DAC (`10.100.208.x`), with the source cache
dropped and the receiver finishing with `fsync`. One sample each:
single-stream unencrypted TCP (`nc`) took 8.19 s, **1.05 GB/s**; `ssh`
with AES-128-GCM took 18.93 s, **0.45 GB/s**. `spark-b`'s local 8 GiB
`fsync`'d write of zeros ran at 4.0 GB/s. These rates are far below both
SSDs' local rates and the link's 184.76 Gb/s RDMA baseline above, so the
copy method limited them; the bottleneck within it was not profiled. No
network, driver or SSH settings changed, and the test files were removed.

### M3 model store (2026-09-28)

M3's checkpoints ([pins](experiments/fast-swap/pins.json)) live on each
Spark's NVMe under `~/.local/share/jitllm/models/<org>/<repo>@<revision
prefix>/`, with the repository's own file layout. Owner environment, not
application configuration.

| Checkpoint | `spark` | `spark-b` |
| --- | --- | --- |
| `unsloth/DeepSeek-V4-Flash-0731-GGUF@fbbb5b93` (UD-Q2_K_XL, DSpark Q8_0, READMEs) | yes | yes |
| `Mia-AiLab/Qwen3.8-Flash-Next-NVFP4@925d7be6` | yes | yes |
| `Qwen/Qwen-Image-2.1@790c9263` (BF16; without `assets/qr.png`) | yes | yes |
| `Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP@dadefa80` | yes | yes (downloaded on `spark-b` from Hugging Face, 2026-09-28) |

Every file was checked on each host against the pinned SHA-256 (or Git
blob, for small files) and size. Binary reference outputs kept out of Git
(the diffusers reference image, [baselines](experiments/fast-swap/baselines.md))
live on both Sparks under `~/.local/share/jitllm/references/`. The older store,
`~/.local/share/jitllm/reference-models/`, keeps M0's GGUFs, including
DeepSeek V4 Flash at `e3aa0d6a` (`D/`) and Qwen3.8's UD-IQ3_XXS (`Q/`).

Each checkpoint was downloaded once, on `spark`, then copied to `spark-b`
over the direct link; TensorFold's MLX checkpoint `spark-b` later
downloaded itself, verified the same way. One sample each, on 2026-09-28:
- **Internet** (`curl` from `huggingface.co`, resumable): about 55 MB/s
  for one stream in a 1 GiB range probe, and about 80 MB/s with three
  files in parallel (213.7 GB in about 44 minutes).
- **Direct link** (`rsync -a --inplace` over SSH, OpenSSH 9.6p1 default
  cipher, new files at the destination): DeepSeek (107.7 GB) to
  `10.100.208.1` and the NVFP4 checkpoint (105.9 GB) to `10.100.209.1`
  in parallel ran at 0.350 and 0.338 GB/s, 0.69 GB/s together; Qwen-Image
  (33.1 GB) alone to `10.100.208.1` ran at 0.39 GB/s. `spark` was also
  downloading at the time, and part of each source was in its page cache.
  That is far below the link; SSH's default cipher, chacha20-poly1305,
  caps one stream at about 0.60 GB/s
  ([jumbo frames](#jumbo-frames-on-the-direct-link-2026-09-28), which also
  gives the faster recipe).

### Jumbo frames on the direct link (2026-09-28)

Approved by the owner and applied on 2026-09-28. Owner environment, not
application configuration. Both Sparks' QSFP interfaces (`enp1s0f1np1`,
`enP2p1s0f1np1`) have **MTU 9000**, set persistently in
`/etc/netplan/99-nvidia-sync-cluster.yaml` (the prior file kept as
`.bak-20260928` beside it), rendered by NetworkManager, and applied live
with `ip link`. The management interface `enP7s7` is unchanged. RoCE
`active_mtu` is 4096.

One sample each, `spark` → `spark-b`, a single stream of zeros to
`/dev/null` unless noted, on 2026-09-28:

| Transfer | MTU 1500 | MTU 9000 |
| --- | --- | --- |
| SSH, default cipher (chacha20-poly1305) | 0.60 GB/s | — |
| SSH, `aes128-gcm@openssh.com` | 1.12 GB/s | 1.01 GB/s |
| Raw TCP (`nc`) | 1.75 GB/s | 2.45 GB/s |
| Four parallel `nc` streams, two per address | — | 8.07 GB/s |

So the default cipher's single-stream ceiling is consistent with the
0.35 GB/s per stream of the `rsync` copy above, which also read from disk
and was not profiled. **To copy between the Sparks,** use parallel `nc`
streams, or SSH with AES-128-GCM (`scp -c aes128-gcm@openssh.com`;
`rsync -e 'ssh -c aes128-gcm@openssh.com'`, since rsync's own `-c` means
`--checksum`), split across both addresses (`10.100.208.x` and
`10.100.209.x`).

### Front-door TLS certificates (2026-09-23)

Owner environment for D-065, not application configuration. At the owner's
request, certbot 5.8.0 (snap, classic confinement) and its
`certbot-dns-cloudflare` 5.8.0 plugin are installed on `spark` and
`spark-b`, with `trust-plugin-with-root=ok` and a `/usr/bin/certbot` symlink
(certbot's snap instructions). Each host holds the owner's Cloudflare token
(`Zone:DNS:Edit`, scoped to `meenan.us`) in
`/root/.secrets/certbot/cloudflare.ini`, root 0600 in a 0700 directory. The
owner wrote those files; they were checked only for mode and format. The
ACME account was registered with the owner's agreement to the Let's Encrypt
Subscriber Agreement and no email.

| Node | Name | LAN address | Certificate |
| --- | --- | --- | --- |
| `spark` | `spark.meenan.us` | 192.168.0.100 | Let's Encrypt YE2, 2026-09-23 to 2026-12-22 |
| `spark-b` | `spark-b.meenan.us` | 192.168.0.101 | Let's Encrypt YE2, 2026-09-23 to 2026-12-22 |

- **Checks run:** a staging `--dry-run`, then production issuance with DNS-01
  and a 30 s propagation wait, then `certbot renew --dry-run`, all passed on
  both hosts. Afterwards Cloudflare's authoritative servers returned no
  leftover `_acme-challenge` TXT records.
- **Renewal:** runs from `snap.certbot.renew.timer` with the settings saved
  in `/etc/letsencrypt/renewal/<name>.conf`. No deploy hook is set until
  M5's jitLLM hook, which is added with
  `certbot reconfigure --cert-name <name> --deploy-hook …`.
- **Local DNS:** the names resolve only on the LAN, through the router's
  local DNS at `192.168.0.1`; `meenan.us` has no public records for them.
  The router drops public answers that contain private addresses, so public
  A records would not help LAN clients.
- **Zone:** `meenan.us` has no CAA records and no DNSSEC.
- **Public exposure:** both names are now in public Certificate Transparency
  logs.

**Tailscale.** The owner installed Tailscale 1.102.4 on both Sparks,
enabled `tailscaled`, and enabled HTTPS certificates for the tailnet
`coati-puffin.ts.net`. `/etc/default/tailscaled` has empty `FLAGS` and no
`TS_PERMIT_CERT_UID`, as D-065's root-run timer design needs.

| Node | Tailscale name | Tailscale IPv4 |
| --- | --- | --- |
| `spark` | `spark.coati-puffin.ts.net` | 100.96.29.72 |
| `spark-b` | `spark-b.coati-puffin.ts.net` | 100.114.118.63 |

- **Test fetch:** on 2026-09-23 one `sudo tailscale cert` per host, into a
  root-only temporary directory, returned a Let's Encrypt YE1 certificate
  valid 2026-09-23 to 2026-12-22 with the node's name as its only SAN. The
  key was mode 0600. The copies were deleted.
- **Renewal:** `tailscaled` keeps its own cache in
  `/var/lib/tailscale/certs` (ACME account key plus the node's certificate
  and key), and renews it when asked (D-065). Nothing re-fetches it until
  D-065's timer exists.
- **Workstation:** not on the tailnet (no `tailscale` installed), so it
  reaches the Sparks by LAN name.


## Port test hosts (2026-10-08)

Hosts the owner made available for the later ports (D-110). Nothing is
built or measured on them yet; each port records its inventory here when
it starts.

| Host | Hardware | Used for |
| --- | --- | --- |
| `mac` | Apple M5 Max, 128 GB unified memory | The Apple silicon port, the first after M3.6 and the families |
| `plex` | Intel Arc Pro B50 | The Intel port, after Apple |
| The Windows NVIDIA host (no SSH name yet) | NVIDIA GPU on Windows; the owner's desktop, not reachable over SSH today | The Windows operating-system port |

No AMD hardware is available yet.

## Crash-dump handling (2026-09-23)

Checked read-only on the workstation, `spark` and `spark-b` for the
architecture's rule that an abort must not leave request content in a core
image (D-014). All three hosts set `kernel.core_pattern` to the apport pipe
(`|/usr/share/apport/apport -p%p -s%s -c%c -d%d -P%P -u%u -g%g -F%F -- %E`)
and `fs.suid_dumpable` to 2. core(5) states that `RLIMIT_CORE` is ignored
when core dumps are piped to a program, so a unit's `LimitCORE=0` alone
does not stop one here. Measured on `spark` on 2026-09-24 (D-074): a
process that segfaults starts a dump (the wait status's core bit) and apport
keeps a report when its executable belongs to a package, while the same
process marked non-dumpable (`PR_SET_DUMPABLE` 0) starts no dump at all,
`fs.suid_dumpable=2` notwithstanding; the workstation behaved the same. The
installed runtime, killed with SIGABRT, left neither a core file nor an
apport report.
