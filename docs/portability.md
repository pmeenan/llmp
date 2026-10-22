<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Portability

How llmpalooza stays ready for other GPU platforms (Apple silicon, then Intel,
then AMD, D-110) and operating systems (macOS, Windows) without rewriting large parts of the
tree, while costing nothing on the GB10 (D-026, D-082). The owner's goal
(2026-09-29): every platform a compile target of the same tree, with shared
code maximized. Nothing here is built for another platform yet; this is
where the seams are, what checks them, and what a port would still have to
do.

The owner-root attention forms retain a GB10 qualification gate. Their
owner-cohort partition, mask scan and resource geometry have been checked
on `sm_121`; `sm_86` execution, occupancy and over-read controls are still
owed. A compiled cubin alone does not establish those properties. Ordinary
F16 attention has common-head vector/MMA forms at 64/80/96/112/128/192,
with vector instances only at multiples of 64 as upstream requires. The
locked Q/K192/V128 form is retained; equal-width 192 MMA adds a small local
configuration extension (RE-060 in [rough edges](rough-edges.md)). The
equal-width 192 vector retains its physical KV padding requirement, with
a mask excluding the logical tail.

Every CUDA GGML backend now owns the complete decode selected source (74
implementations: twelve format/layout decoders times six templates, plus two
mixed GLUs) and the compiled EXL3 source. Registration is independent of model
labels and policy text. Product adapters retain physical parts and per-matrix
rows, batch broadcasting and input pitches; resident table gathers support all
twelve decoders, while authenticated paged table gathers remain split NVFP4 and
affine. Row-interleaved NVFP4 retains optional and per-stack tensor scales;
the native gather requires one scale and an unbatched table. The actual stack
count also qualifies measured preferences. An unsupported physical shape must be
refused by the capability before selection, rather than admitted merely because
an encoding has a decoder.
Native MXFP8 vector products and short-block row gathers retain their 2D
operand domain; batched operands reach the general decoder. Native GGML NVFP4
expert products accept their one-part representation, while separate optional
tensor scales reach the grouped decoder under its own implementation identity.

The EXL3 factory queries current-device GEMV and packed occupancy facts on a
compiled architecture. It composes the four existing frozen G/O pin sets with
a separately named occupancy-derived packed GEMM/multi-GEMM source. The latter
has no borrowed reconstruction algorithms or measured pins. All paths still
require the four MCG physical parts; importing another codebook does not
establish execution support. Packed GEMM/GEMV pins require their exact rate;
reconstruction retains exact shape/output-typed cuBLASLt pins over reconstructed
F16 slices, independent of the original compiled trellis rate. Reconstruction is
not offered for decode-shaped products. Architectures
without a compiled EXL3 image publish no EXL3 candidates and require no EXL3
occupancy query. Each backend declares five disjoint lock workspaces
(4,203,008 bytes each, 21,015,040 bytes total), plus any explicitly supplied
source requirements. Its runner funds these separately in cataloged
`Runtime`/`Pinned` kernel memory: 22 MiB of physical backing per ordinary
backend at 2 MiB granularity, including backends whose graphs use only GGML.
They never borrow the shared discardable scratch pool. The runner retains
this memory through every model switch and releases it only after drain,
plan/recording destruction and launch retirement. CUDA lock contexts and
reconstruction handles are created only when an actual selected EXL3 call
binds, on the driver before submission or capture; unused sources retain
their paid backing but create no CUDA lock context.
Device names, architecture, SM counts and supported MMA
features come from the provider; profile labels do not establish these facts.

GGML quantized products decode on chip through MMVQ/MMQ. Float products that
actually route through cuBLAS with F32 accumulation declare the transient dense
conversion and cannot expand decode-shaped weights. A narrow vendor-free port
of the locked NVIDIA MMVF/MMF selectors proves the non-expanding paths; its
384-case physical-shape parity control compares against the compiled selector.
When the source pin or compiled images change, update that port and rerun the
parity control. Native block-scaled CUTLASS forms require the actual feature
bits and their compiled images. A device's intrinsic feature support alone
does not establish that an architecture-specific kernel was compiled.

Plan keys include the canonical implementation-set digest, immutable device and
tuning facts, checked-in measurement-table identity, exact explicitly supplied
source preferences and capture requirement,
while retaining the backend execution-domain seal. Explicit source preferences
rank after measured evidence and before the ordinary ladder; they preserve
existing exact pins without relabeling or combining different frozen profiles. Backend facts are hashed
once when a source is created, not on each routed step. Unmeasured device and
shape combinations retain the ordinary ladder. Recording additionally requires
the exact bound selected call's runtime opt-in, delegated through composites to
its owning child. Eager-only programs report their actual capture limitation;
ordinary programs on the same backend retain capture. Every matcher lookup,
including helper and batched queries, carries automatic-rewrite intent through
one borrowed selection view. Newly reachable decode GLUs and occupancy-derived
EXL3 paired products require measured preference before replacing primitive
products. Occupancy establishes executable capacity, not speed or quality.
Historical exact scalar paired pins retain their qualified domain, and direct
typed operations remain available. Final selection retains rewrite provenance
through pattern rounds, pruning and fallback decomposition; an eligible
incumbent never admits an unqualified competitor. Common-subexpression joining
keeps explicit and automatic operations separate when their selection contracts
differ. See the
[competition protocol](experiments/decode-ladder-r6.md).

## Where vendor and OS code may live

| Code | Only in | Everything else |
| --- | --- | --- |
| CUDA: headers, runtime and driver API names and types, kernels and launches, `.cu` files | `src/providers/cuda/` and `src/kernels/` | reaches the device through the provider interfaces, the device runtime and the kernels' own vendor-free headers |
| Kernel libraries: `ggml`, `exl3`, `cublas` and `cutlass` headers, namespaces, types and words | `src/kernels/` (for `engine/`, `runtime/`, `ir/` and `execution/`) | reaches the backend through `kernels/common/` (`DeviceBackend`, `StreamLibrary`, `SelectedSource`, `StepProgram`, `LowerOptions`) and `engine/runner_resources.h` |
| Linux-specific headers and calls: `<linux/...>`, epoll, eventfd, signalfd, io_uring, memfd, `O_DIRECT`, `O_TMPFILE`, `getrandom`, `accept4`, `SOCK_CLOEXEC`, `MSG_NOSIGNAL`, prctl, raw system calls, `/proc` and `/sys` paths | `src/platform/` and the Linux providers (`src/providers/uring_storage.*`) | calls the platform module |
| POSIX (open, poll, sockets, mmap, pthreads) | anywhere | macOS has it; a Windows port gives the platform module its own implementations (below) |

`tools/llmp_boundaries.py` enforces the first three rows over `src/`, in
the light check tier (`tools/check`'s `boundaries` step) and in the tools
tests (`tools/tests/test_boundaries.py` checks this tree). It reads code
with comments removed, matches names outside string literals and paths
inside them, and fails on an include of `providers/cuda/`,
`platform/io_uring.h` or `providers/uring_storage.h` elsewhere, or any
kernel-library header or identifier under `src/engine/`, `src/runtime/`,
`src/ir/` or `src/execution/`. The latter two also reject embedded vendor
identifiers, device names and diagnostic strings; provenance comments remain
allowed. Included `.inc` source tables are checked too. An
exception needs an entry in its `EXCEPTIONS` table with the reason, and a
stale exception is itself a failure. **There are no exceptions today.**
Tests and benchmarks drive the CUDA build directly and are not checked.

IR byte strides come from `format::AddressUnitOf`, which shares the size rule's
physical element/block geometry. NVFP4's packed block is 64 values in 36 bytes,
not its 16-value scale group. Multipart and tiled formats have no such unit;
shape validity, whole rows, overflow and readable padding remain the format
layer's `Validate`/`PartsOf` contract. Preparation requires an explicit target
layout chosen by its caller's implementation capability.

## The GPU seam

A backend is four things, all chosen at build time (D-028, D-053: no
runtime plugin ABI):

1. **The device-memory and device-execution providers**
   (`providers/device_memory.h`, `device_execution.h`): VMM reservations,
   backing and mapping; streams, fences and copies. Virtual interfaces,
   because the resource core also runs them over the fakes.
2. **The device runtime** (`providers/device_runtime.h`): opening the
   device, and what the engine's device jobs queue besides kernels: copies
   and fills on the job's stream, timing marks, recorded work (a captured
   and instantiated sequence replayed as one, CUDA graphs today), the
   thread's error state, pinned host memory, the device's architecture
   number and free memory. Plain functions the backend's provider module
   defines (`providers/cuda/cuda_device_runtime.cc` over the CUDA runtime
   API), one direct call around the backend's own: no virtual dispatch,
   no table lookup, no allocation on success. The fake device runtime
   (`providers/fake/fake_device_runtime.cc`) defines them over host memory
   and the fake providers, for the engine core's tests in every profile:
   queued work runs when a fence is queried, never because a call
   returned.
3. **Kernel modules** (`src/kernels/<source>/`), which unwrap the
   job's `NativeStream` handle themselves and register their
   implementations (D-053).
4. **The registry's plans**, which a backend can only run if every
   operation has an implementation it provides ([below](#the-registry-rule)).

The engine (`src/engine/`) is above all four and holds no vendor code: it
is built without CUDA's include path, so a CUDA header reaching it fails
the build, and the boundary check fails a CUDA name. Its core
(`llmp_engine_core`: the node, weights, live state, runner memory, plan
accounting and plan cache, prefill lookahead, the cohort, the shared slot
lifecycle, the one model runner over step programs with its step runs,
wave composer, prefill orchestrator and decoding modes, the component step
source and the spatial graph executor) names no kernel library either,
reaching the device through the device runtime and the paging kernels
(`kernels/paging/`, which have CUDA kernels and host loops for the fake
runtime), and builds in every profile; a program links one backend's
runtime and paging kernels. In CUDA builds, `llmp_engine` adds runner
resources (`runner_resources.h`) and the expert slab (`expert_slab.h`)
(D-096 as amended 2026-10-08).

### What moved out of the engine (2026-09-29)

The engine made 69 direct CUDA runtime calls. They now go through the
device runtime, one for one: 22 `cudaMemcpyAsync` → `CopyAsync`, 3
`cudaMemsetAsync` → `FillAsync`, `cudaMallocHost`/`cudaFreeHost` (9) →
`AllocatePinned`/`FreePinned` (and the node's `HavePinned` helper for a
runner's state copy), the node's per-stream timing events (create,
record, elapsed, destroy) → timing marks, the image runner's graph
capture, instantiate, launch and destroy → `BeginRecording`,
`EndRecording`, `Replay` and `RecordedWork`, `cudaMemGetInfo` →
`QueryDeviceMemory`, the compute-capability queries and
`cudaGetDeviceProperties` → `QueryDeviceFacts`, `cudaSetDevice` +
`cudaFree(nullptr)` and the CUDA providers' constructors → `OpenDevice`,
and `cudaGetLastError`/`cudaGetErrorString` → `TakeLastError` and
`DeviceStatus::text`. The two `.cu` kernels the engine held (the weights'
range fill and Qwen3.8's n-gram row gather) moved to `src/kernels/paging/`
with a vendor-free header. The engine also opens its storage rings through
`providers::OpenStorage` rather than naming the io_uring provider.

**Cost on the GB10:** none measurable. Plain decode through `llmp-runtime
chat --plain` (256 tokens, second turn of each model, two alternating
rounds; `spark`, 2026-09-29): DeepSeek V4 Flash 22.13 and 22.21 tok/s on
main, 22.10 and 22.16 on the slice; Qwen3.8 Flash Next 27.60 and 27.67 on
main, 27.64 and 27.72 on the slice.

**Graph capture and replay:** step programs record and replay through
`StepRuns` (`engine/step_runs.h`) and `StepLaunch::Record`
(`kernels/common/step_program.h`) over `RecordedWork` (implemented on CUDA
through the launch context in `kernels/ggml/launch.h` and
`providers/cuda/cuda_device_runtime.cc`, and on the fake runtime in
`providers/fake/fake_device_runtime.cc`); a backend without graphs (or a
step whose bound implementations opt out of capture) replays launch by
launch through `StepProgram::Launch`.

### What a new GPU backend supplies

`providers/<vendor>/` with the two providers and the device runtime (for
HIP, a near copy of the CUDA module over `hip*` calls; for Metal, MTLHeap
placement for backing, command queues for streams, MTLSharedEvent for
fences, and MTLIndirectCommandBuffer or plain re-encoding for recorded
work), kernel modules for its operations (GGML's HIP, SYCL and Metal
backends are kernel sources to adapt under llmpalooza's dispatch, D-053,
D-110), a block decoder per weight and state encoding (D-108), and a CMake
profile that builds them instead of CUDA's. A kernel source reaches the
engine as a `StepBackend` (`kernels/common/lower.h`): its registry with
each implementation's capability, its device's facts, the implementations
that must stay on the stream, what its implementations write beyond a
tensor's own bytes, and a lowering from a planned step to a
`StepProgram` with its `StepLaunch` (GGML's: `kernels/ggml/step_program.h`
over `lower.h`); the engine's component step source plans over it and
names no kernel library. CUDA's decoders and the
product templates over them (`src/kernels/decode`: vector, GLU, matrix,
grouped and rows; mma.sync from sm_80, and an F32 FMA matrix path that
needs no matrix unit, though no build has SASS below sm_80 and the
grouped template is mma.sync only) are the shape each backend's take: a
decoder is a struct that decodes a 16-value segment of a row into
registers, and the templates are shared, so a new encoding runs every
product shape from its decoder alone. On the discrete `sm_86` target
they already run every product shape of a Qwen3.8 layer, whose NVFP4 and
MXFP8 that GPU has no matrix units for (the format layer's first proof,
engine-components.md). The device-memory design
assumes explicit virtual reservation and mapping (D-006); where a platform
has no equivalent, the provider maps the rules D-033 needs onto what it has,
which is that port's decision to make.

## The OS seam

| Service | Interface | Linux | macOS | Windows |
| --- | --- | --- | --- | --- |
| Event loop for the chat route | `platform/event_loop.h`: `EventLoop` (level-triggered readiness: readable, writable, peer closed, hang-up, error, with a tag) and `Waker` | epoll, eventfd | kqueue (`EVFILT_READ`/`WRITE`, `EV_EOF`), `EVFILT_USER` | an I/O completion port with AFD poll requests for socket readiness (as wepoll and mio do), a posted completion as the waker |
| Signals as readiness | `SignalWatch` | signalfd over blocked signals | `EVFILT_SIGNAL` | a console control handler posting to a waker |
| Available memory | `platform::AvailableMemoryBytes` | `/proc/meminfo` MemAvailable | `host_statistics64` (free, inactive, purgeable) | `GlobalMemoryStatusEx` |
| Spill file | `OpenUnnamedDirectFile` | `O_TMPFILE` + `O_DIRECT` | `mkstemp`, unlinked at once, `F_NOCACHE` | `FILE_FLAG_DELETE_ON_CLOSE` + `FILE_FLAG_NO_BUFFERING` |
| Shard for direct reads | `OpenForDirectRead` (with an optional buffered fallback) | `O_DIRECT` | `F_NOCACHE` (a hint) | `FILE_FLAG_NO_BUFFERING` |
| File identity | `FileGeneration` | `FS_IOC_GETVERSION` | `st_gen` (root only) or none | the volume serial and file ID |
| Anonymous shared memory (the fake device memory) | `OpenAnonymousMemoryFile` | memfd | `shm_open` + `shm_unlink` | a pagefile-backed section |
| Sockets | `platform/sockets.h` | `SOCK_NONBLOCK`/`SOCK_CLOEXEC`, `accept4`, `MSG_NOSIGNAL` | `fcntl` after the call, `SO_NOSIGPIPE` | Winsock |
| Random bytes | `FillRandom` | `getrandom` | `getentropy`/`arc4random_buf` | `BCryptGenRandom` |

Behaviour and performance on Linux are unchanged: each is the call the
runtime made before, moved.

### Storage and direct I/O

The storage provider (`providers/storage.h`) is where a port's I/O goes:
`OpenStorage(depth)` is the system's implementation, io_uring's on Linux
(`providers/uring_storage.*`). Every request resolves as not started,
accepted or unknown, and only its completion retires its memory (D-048),
which any asynchronous mechanism can honour:

- **macOS:** a small thread pool issuing `pread`/`pwrite` on descriptors
  opened with `F_NOCACHE`, completions queued back to the lane, `Wake` by
  a waker; or `dispatch_io` with the same queue. There is no true direct
  I/O: `F_NOCACHE` asks the unified buffer cache not to keep the pages,
  which matters because the page cache comes out of the same unified
  memory budget, as on the GB10 (D-034).
- **Windows:** overlapped `ReadFile`/`WriteFile` on handles opened with
  `FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED` (sector-aligned, like
  `O_DIRECT`), completions from an I/O completion port, `Wake` by
  `PostQueuedCompletionStatus`.

Llmpalooza probes direct I/O per storage role at startup
(`platform::ProbeDirectIo`) and refuses a role without it, because on the
GB10 reads through the cache would spend the one memory budget.
`OpenForDirectRead`'s `buffered_fallback` exists for a platform where
direct I/O is only a preference (as TensorFold's reader falls back to
buffered reads when `O_DIRECT` is refused); llmpalooza never asks for it on
Linux, and a port that does decides what the role check accepts. Direct
reads pin the selected regular object before opening it for I/O; any
buffered retry reopens that same object, even if its pathname changed.
Final symlinks are refused, and no ownership or permission policy is added.

### What a new OS still needs

A Windows port gives the platform module a descriptor type of its own
(it is `int` throughout today), replaces the POSIX calls that the rest of
the tree makes directly (`open`/`openat` beneath a directory for the
artifact reader and the configuration's drop-ins, `poll` in the chat
route's driver, BSD sockets), and the process services (`platform/job.*`, `confine.*`,
`crash_policy.*`, `lock_file.*`, `sd_notify.*`), which are Linux's by
design. A macOS port needs only the platform module and the storage
provider: the rest is POSIX. Neither is started.

## The registry rule

(D-053's note of 2026-09-29; updated for M3.6/M3.5, D-107.) Every fused or
fast operation has a fallback composed of primitive operations in the
registry, so a new backend can run a model with primitives alone, and fused
kernels are optional speedups.

**How plans choose today.** Every model's step builds a primitive IR graph
from the component catalog (`components::BuildStep`, `src/components/step.h`).
`planner::Plan` (`src/planner/plan.h`) rewrites matched subgraphs using
`planner::StandardPatterns()` (`src/planner/fusions.h`) under `planner::Policy`
(`src/planner/policy.h`, unless `planner::PolicyMode::kPrimitive` or
`DisabledPatterns` disables them), then selects an implementation per node from
`execution::Registry` (`src/execution/registry.h`) using `CapabilityQuery`
(carrying `ir::OpKey` (`ir::Op` and `ir::FusedKind`, `src/ir/op.h`), operand
and output `TensorFormat`s, logical shapes, `StepShape`, `DeviceFacts`, node
parameters and `automatic_rewrite` intent) and `Registry::Candidates`
(`src/execution/registry.h`). Backends supply owned `execution::RankData`;
the GGML backend's checked-in measurements live in
`src/kernels/ggml/measured_ranks.h`. The registry digest covers that data,
so plan caches follow rank changes without importing backend tables. When
a fused or composite node finds no eligible implementation, the planner
decomposes it via `ir::Decompose` (`src/ir/decompose.h`) into primitive IR ops
before re-selecting.

**Audit across model families:**

| Model | Primitive path today | Where the rule still needs backend primitives |
| --- | --- | --- |
| Dense decoders (Llama, Mistral, Qwen2, Gemma 2/3) | Yes: `planner::Policy{.mode = planner::PolicyMode::kPrimitive}` (`kMatMul`, `kAttention`, `kRope`, `kRmsNorm`, `kUnary`, `kAdd`, `kMul`, `kGlu`, `kGetRows`, `kCacheStore`) | none |
| EXL3 fixtures | `planner::Policy{.mode = planner::PolicyMode::kPrimitive}` for norms, attention and GLU; EXL3 linear uses the selected EXL3 source (`kernels/exl3/selected_source.h`) | the EXL3 trellis dequant and GEMM (any backend needs its own trellis decoder) |
| DeepSeek V4 Flash | `planner::Policy{.mode = planner::PolicyMode::kPrimitive}` plus `ir::Decompose` for `Op::kHcMix` (`HcKind::kSinkhorn`) / `FusedKind::kSinkhornHcPre`, `FusedKind::kHcPre`, `FusedKind::kHcPost`, and `Op::kIndexScore` / `FusedKind::kRoundedIndexScore` | `kMatMul` with `Hint::kHadamard` (Hadamard rotation, or a dense `kMatMul` fallback), `kAttention` with sinks at D 512, `kVisibleTopK`, `kSelectedMask`, `kPrefixOverwrite` and `kArgmax` |
| Qwen3.8 Flash Next | `planner::Policy{.mode = planner::PolicyMode::kPrimitive}` plus `ir::Decompose` for `FusedKind::kHcCombine`, `FusedKind::kHcPre`, `FusedKind::kHcPost`, `Op::kIndexScore`, `Op::kGatedDeltaNet` (`FusedKind::kGdnStep`) and `Op::kSsmConv` (`FusedKind::kGdnConv`), and `kernels/decode`'s on-chip decode templates (or GGML quantized products on GGUF) | `kAttention` at D 256 and 512, IMROPE, `kArgmax`, and a block decoder for NVFP4/MXFP8 (or short-block row gathers on GGUF) |
| Qwen-Image-2.1 | `planner::Policy{.mode = planner::PolicyMode::kPrimitive}` plus `ir::Decompose` for the spatial fused kinds (`kSpatialChannelNorm`, `kRmsNormMulRopeSections`, `kLayerNormModulateSplit`, `kGatedResidualSplit`, `kGatedResidualModulateSplit`, `kDupUpAdd`) and primitive `Op::kGlu` and `Op::kConv2d` | BF16 GEMM, 3×3/1×1 `kConv2d`, non-causal and causal `kAttention` at D 128, 2D complex RoPE, `kChannelNorm`, `kUpsample` and `kConvert` |

**Minimum primitive set a new backend needs**, taking each model's most
unfused path that exists:

- **Qwen2 FP16:** F16 `mul_mat`, add, mul, `rms_norm`, NEOX RoPE, masked
  scaled `soft_max`, cont, SwiGLU, `get_rows` (F32), `set_rows` (F32 to
  F16).
- **DeepSeek V4 Flash (exact plan):** quantized `mul_mat` and
  `mul_mat_id` for Q8_0, Q4_K, Q5_K, Q6_K, IQ2_XS, IQ3_XXS and MXFP4;
  F32, F16 and BF16 products; the Hadamard rotation (or dense
  `mul_mat`); `rms_norm`; NEOX RoPE with offset, and its inverse;
  attention with sinks at D 512; add, mul, div, scale with bias, sigmoid,
  softplus, sqrt, clamp, fill, SwiGLU and clamped SwiGLU; cont, repeat,
  concat, `sum_rows`, `soft_max`; `get_rows`, `set_rows`; argsort top-k and
  `top_k`; and, with no fallback yet, `dsv4_hc_comb`, `dsv4_hc_pre`,
  `dsv4_hc_post` and `lightning_indexer`. DSpark adds argmax.
- **Qwen3.8 Flash Next (unfused, GGML expert layout):** NVFP4
  `mul_mat_id` and NVFP4-row `get_rows`; BF16 and F32 products; MXFP8
  GEMV or dequant to BF16 (or, for the NVFP4, MXFP8 and BF16 products
  and the NVFP4 rows, `kernels/decode`'s on-chip decode templates);
  `rms_norm`; multi-section RoPE (IMROPE);
  attention at D 256 and 512; add, mul, div, scale, sigmoid, SiLU,
  softplus, sqrt, ReLU, abs, sign, clamp, fill; repeat, concat, transpose,
  cont, `sum_rows`, `soft_max`; `get_rows`, `set_rows`; argsort top-k and
  `top_k`; `ssm_conv` and `gated_delta_net`. Verify and MTP add argmax.
  A GGUF checkpoint's form replaces NVFP4 and MXFP8 with `mul_mat` and
  `mul_mat_id` of its quantized types (MMVQ and MMQ) and F32 products,
  and the n-gram table's rows with `llmp.qrows.get_rows` (a 32-value
  block type's rows of 160 values, which GGML's `get_rows` reads only in
  256-value super-blocks; no fallback yet).
- **Qwen-Image-2.1:** BF16 GEMM; 3×3 and 1×1 convolution (im2col and
  GEMM); non-causal attention at D 128 and a short causal one; SiLU,
  GELU-tanh, SwiGLU, add, gated residual, scale-shift, all rounding to
  BF16 where diffusers does; RMS, zero-centred RMS, layer norm with
  modulation and channel norms; 2D complex and NEOX RoPE; row embedding,
  transpose, nearest 2× upsampling, the Euler step and F32-to-BF16
  conversion. All llmpalooza kernels today.
- **Gemma 4 graph:** quantized `mul_mat`/`mul_mat_id`, embedding `get_rows`,
  `rms_norm`, learned mul, GELU-tanh and split GeGLU, factor-aware NEOX RoPE,
  per-slot D256/GQA2 and D512/GQA8 masked attention, `set_rows`, routing top-k,
  softmax and ordered expert sum. Device-mask mode additionally needs a packed
  I32-position-to-F16 causal/ring `kFill` implementation (`llmp.gemma4.mask`);
  the funded host-reference-mask graph remains the primitive alternative for
  a backend without it. Descriptor planning/validation is CPU-buildable and
  the engine/provider interfaces contain no CUDA types.
- **Gemma Q8 assistant:** ordinary quantized `mul_mat` and target embedding
  `get_rows`, concat, RMS norm/learned mul, GeGLU-tanh, factor-aware NEOX RoPE,
  per-slot D256/D512 masked attention and full canonical head/post-projection.
  It has no `set_rows` cache update. Host frozen-prefix masks remain its funded
  primitive alternative; engine/provider interfaces carry no CUDA types.
- **EXL3:** the FP16 set without `soft_max`, plus F16 and F32
  conversion, attention at D 64, the EXL3 dequant and GEMM, and a half
  bias add. A port's trellis decoder is checked against the CPU
  reference's (`kernels/reference`, whose rotated weights match
  ExLlamaV3's `reconstruct_kernel` bit for bit), as CUDA's block decoders
  are.

The M3.6 G4b1 primitive lowering declares DeepSeek's existing top-k and
indexer operations, inverse/offset RoPE and homogeneous F16 mask adds.
The indexer also has a tested decomposition into products, ReLU,
head-weight multiplication, reduction and an F32 mask add; a half mask
widens exactly through the existing F16/F32 conversion kernel. Mixed
F32/F16 binary adds remain unsupported and are not advertised. Native
indexer selection is limited to D 128 and 32 or 64 heads; other valid
shapes take the decomposition. Conversion currently requires packed
F16/F32 operands and an I32-bounded element count. Device ahead masks
require the existing block kernel's full ring and positive window; other
valid ahead geometries can use the host mask.

G4b2a's four local fast forms are opt-in and keep typed primitive
definitions. Q-head takes F32 [512,64,T], 16–8192 rows and the 64-value
tail rotation; full-carry HC pre takes four streams of width 4096 or
8192 with F32/F16/BF16 mixing weights; HC post takes four packed F32
4096-value streams and returns F32 streams and norm. Other shapes keep
the primitive path. Compressor selection requires generated typed read
indices and the existing helper's ratio/grid bounds. Literal dummy fills
and a probability clamp preserve zero all-masked pools on GGML CUDA;
nonfinite-value probes cover the helper and primitive fallback. Default
fast-policy, whole-model parity and runtime migration remain G4b2b/G4b2c
and G4c work.

G4b2b1's opt-in state-bank attention has an ordinary concatenation and
Attention fallback. Only the native zero-copy recipe requires a layout
span fact; a directly authored fused node without that proof decomposes.
Its GGML recipe requires F32 output, packed F16 raw state and a compressed
prefix in the same declared region, D256/D512, more than four query heads,
a 256-cell multiple, no softcap or bias, and no sparse-any flag. Actual
bound strides, addresses and launch limits are checked again. Requested
F16/BF16 results use the fallback's explicit conversion. The source layout
and state-backing rules do not change, and the indexer's existing
arithmetic, TopK and masks remain in the graph.

The separate exact selection/mask primitives (G4b2b2a) have vendor-free
scalar implementations and native GGML recipes. `VisibleTopK` accepts
F32 scores/I32 counts, top 1–65536, C at most INT32_MAX/4 and an
I32-bounded output element count; top may exceed C. `SelectedMask` accepts
F16 raw masks and arbitrary I32 IDs, including duplicates, with total
width at most 65535×4096, an I32-bounded output element count and top at
most INT32_MAX/4. Both accept whole-element input strides, including
repeated views, and require packed outputs disjoint from all inputs.
Unsupported native shapes refuse selection; ordinary GGML TopK is not an
equivalent fallback for the exact operator. The existing score semantics,
MQA component version and default policy remain unchanged.

**What closing the gaps takes**, as the rule applies to new work from
now on: an unfused composition for the model-specific operations
(hyper-connection mixing and Sinkhorn, the lightning indexer, the gated
delta net and its convolution, argmax, NVFP4 rows) expressed in GGML
primitives, even slow, so a backend with GGML's primitives runs every
model; a dense fallback for the Hadamard product (the graph IR now carries
the hyper-connection, indexer, gated-delta-rule and convolution
compositions as primitive decompositions, `ir/decompose.h`, which the GGML
lowering can emit); a builder switch that
reaches the unfused Qwen3.8 graph from the served artifact (or an
artifact-independent expert layout); and for the image family a
primitive plan over GGML's operations or an equivalent set, with the
BF16 rounding points the reference needs. None of this changes the
GB10's plans, which keep their fastest implementations.

The rounded-query score composite has a real primitive fallback: funded
F32→F16→F32 conversion, ordinary product/ReLU/weighting/reduction, then a
literal prefix overwrite. Nonpacked F16 keys use funded `Cont`; the native
copy preserves half representations through its checked same-type copy paths.
`PrefixOverwrite` is an exact F32 bit-copy/fill
primitive with an independent CPU implementation and a basic strided-input
native launch. A backend must provide that operation to run this optional
composite; an additive mask is not an equivalent fallback for unseen NaN or
infinite scores. An optional native rounded score uses the existing Lid score
producer after a paid literal-tail initialization, with the original device
arithmetic and grouping. Its capability requires authenticated operand strides
and root alignment; unknown or unsupported metadata chooses this real fallback
before lowering. Provider reservation alignment is a separate optional physical
guarantee, forwarded through exact state bindings and authenticated graph passes.
New independent activation roots use the declared arena and placement alignment;
contained outputs, externally bound saved rows and hypothetical descriptors do
not inherit it. Final placement and binding recheck promises, addresses and overlap.
No component or mapping enables the native score yet, and the wider scalar remains an independent
mathematical reference rather than a claim of universal native bit identity.

## The runners' shared skeleton

The M3 runners grew one model at a time, and the audit of 2026-09-29
found about 1,000–1,300 of the two LLM runners' 3,750 lines
near-duplicates: weight paging (DeepSeek kept its own copy of
`PagedWeights`), setup, plan caches and `PlaceAndPlan`, graph capture and
replay, speculation's snapshot and rollback, state spill and restore,
teardown (copied six times with the node and two harness runners) and the
request's lease. M3.6 (D-107) replaced the per-family runners, plans and
serving classes with one skeleton and one runner in `src/engine/`
([engine.md](engine.md)):

| Concern | Now | Was |
| --- | --- | --- |
| Graph capture and replay (D-090) | `step_runs.h`: `StepRuns` over step programs (staging, input and output copies, capture, replay, the launch-by-launch fallback), `StepPlanRuns`; `run_stats.h`: `GraphStats`, `RunPath` | two `QueueRuns`, `Stage`, graph eviction inline, path counting copied three times; `Dsv4Path` and `Dsv4GraphStats` |
| State spill and restore | `live_state.h`: `LiveState` regions, spill file, clear, read, places, quarantine; `slot_lifecycle.h`: `SlotLifecycle` derived from each stage's `StateContract` | byte-identical `RegisterState`, `Clear`, `ReadState` in each |
| Plans and shape caches | `plan_cache.h`: `BasicPlanCache` (indexed by the key's hash); `model_runner.h`: one plan per step shape over `ComponentStepSource` (`component_source.h`); `coverage.h`: `CheckCoverage` | four identical `Planned` structs, `PlaceAndPlan` and `PlanPlaced`, four planning bodies, three coverage checks |
| Speculation mechanics | `slot_lifecycle.h` (`SlotLifecycle::Accept`, `DiscardVerify`, `Settle`) over `LiveState`'s verify snapshot (`live_state.h`: `Save`, `Accept`, `Rollback`, `Settle`), queued as the paging kernels' range copies on the job's stream; `decoding.h` (`DecodingOrchestrator`) and `execution/program.h` (`DecodingMode`) | the same semantics written twice |
| Setup and teardown | `runner_memory.h` (own memory, staging) and `runner_resources.h` (cuBLAS, the launch context and registry), their ordered release; `ReleaseMapped` for the node, `ModelRunner`, `StagedExecution` and the FP16 benchmark runner (`benchmarks/fp16_runner.h`) | copied per runner |
| Weight paging | `paged_weights.h` for every runner: host groups, expert slabs (`ExpertSlabsOf`, `expert_slab.h`), `PleTable` / `PleRead` / `PleRowPlan` / `PlanPleRows` (`ple_rows.h`), per-extent groups, place checks; `LayOutSlab` | DeepSeek's own `ReservePart`/`Register` |
| The request's lease (D-093) | `PagedNode::WithRequest` | three copies of begin-body-end |

Every LLM runs on `ModelRunner` (`engine/model_runner.h`) over
`ComponentStepSource` (`engine/component_source.h`) and `SlotLifecycle`
(`engine/slot_lifecycle.h`), served through `EngineLlm`
(`runtime/engine_llm.h`, assembled by `ComponentLlmBuild` in
`runtime/component_build.h`) over `Llm`'s conversation logic
(`runtime/llm.cc`), which builds in every profile. Image pipelines run on
`StagedExecution` (`engine/staged_execution.h`), served through
`EngineImage` (`runtime/engine_image.h`, assembled by
`ComponentImageGenerationBuild` and `ComponentImageServingBuild` in
`runtime/component_image_build.h`). The serving adapters name no kernel
library, but the two build units do: `component_build.cc` and
`component_image_build.cc` construct `kernels::ggml::GgmlBackend`
(`GgmlBackendOptions`) and `CublasHandle`, size the cuBLAS workspace from the
compute capability and open device 0, and `engine/runner_resources.h` and
`expert_slab.cc` include GGML headers. `llmp_boundaries.py` checks CUDA names,
not kernel-library names, so nothing enforces this edge; a port needs an
engine-side backend factory first. A port changes the skeleton once: a
backend without recorded work uses `StepRuns`' launch-by-launch fallback,
and a platform's spill file is `LiveState`'s. Both LLMs and image pipelines
verify their registered places after every swap.

## Distribution

D-098: the `.deb` is primary (systemd, cgroup-delegated jobs, io_uring and
`O_DIRECT`, unlimited memlock); an OCI image built from the same `.deb` is
secondary, run with a seccomp profile that allows `io_uring_setup`,
`io_uring_enter` and `io_uring_register` (Docker's default blocks them since
25.0), `--ulimit memlock=-1:-1`, `--gpus all` through NVIDIA's container
toolkit, and the storage roles bind-mounted from the host's filesystem.
Homebrew or `.pkg` and winget or MSI come with their ports.
