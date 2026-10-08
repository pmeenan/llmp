<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Early backend integration proof

D-028, D-051 and D-052 require the Qwen2.5-0.5B-Instruct FP16 control and
both small EXL3 fixtures to run natively on llmpalooza-owned memory before the
internal operation contract and executable layout are settled. Under D-053,
llmpalooza owns dispatch: GGML- and ExLlamaV3-derived kernels, and later other
sources or our own, are build-time implementations of operations, selected
per plan. This is the M0 scope for that proof, which M2 executes alongside the resource core. It
fixes stages, numerical oracles, cases and evidence, and records what the
pinned sources imply for them. It is not an implementation, a measurement or
a support claim. The artifact encoding is D-056's [v0 format](artifact-format.md);
sources are acquired under D-057's [source-dependency mechanism](source-dependencies.md),
and the [retained-backing comparison](#retained-backing-comparison) stays a
separate plan item; the proof
consumes or hosts them.

## What the proof settles

| Question | Settled by | Feeds |
| --- | --- | --- |
| How GGML-derived operations run under llmpalooza dispatch, per operation: GGML's launchers behind a llmpalooza-supplied context, or lifted kernels behind owned launchers | P1–P2, BP-A/BP-L cases | D-053 integration record and build-time patch set |
| Coexistence and swapping: several implementations of one operation, and several kernel sources, in one build and process, selected per plan | P1, P3, BP-S cases | Operation contract and implementation registry |
| Dispatch overhead against upstream's captured decode | BP-F4 | Whether M3 parity needs graph capture with a relocation proof |
| The operation contract: dependencies, workspace, streams/fences, captured pointers, backend allocations, errors (ideation §10) | All stages | Decision entry at M2 close, before M3 builds on it |
| Executable-layout constraints: alignment, padding, kernel-readable ranges, tile rules | P2–P4 against D-056's v0 encoding | Validates or amends the experimental artifact |
| Phase envelopes and fixed runtime overhead `F` for the declared profiles | P6 | D-050 admission numbers for M2/M3 |
| Whether actual kernels regress on llmpalooza's VMM (host VMM failed; device VMM, D-081, passed) | BP-F1 | D-081 reopen check |
| EXL3 per-kernel time and workspace parity with upstream | BP-F2 | D-052 M2 gate |

The proof is not complete with a loader, one matrix multiply, an external
reference process or a fake backend. It exercises the M2 catalog, lease and
D-048 completion code on real providers. It does not implement a pager only
for the proof.

## Entry conditions and ordering

- **M1 delivered:** SDK and toolchain (D-032/D-049), CMake presets, the
  CPU-only guardrail build, and the question-7 mechanism, proven on
  GoogleTest. P0 uses it to admit the pinned GGML subset and the selected
  ExLlamaV3 files, supplying any patch as a reviewed file with hashes and
  notices (plan.md).
- **P0/P1 need no artifacts.** They start once M1 builds. P2 onward runs from
  prepared artifacts in D-056's experimental v0 encoding. The M0 layout
  study built and verified all three fixtures in it. A proof-only file format does
  not satisfy "from prepared artifacts".
- **P4/P5 run on the M2 resource core:** catalog, leases, storage and device
  services. They validate the code M3 builds on.
- **Before evaluating a native result**, run the held-out trajectories on
  the references, record numerical profiles and cross-implementation bounds
  from reference controls ([first-slice.md](first-slice.md),
  [exl3-bringup.md](exl3-bringup.md)), and freeze the performance protocol.
  Each threshold is approved by the owner, or pre-registered under D-079,
  before any native output it governs is seen: this may come in parts
  ([P0 declarations](#p0-declarations)), but never after. A bound set or
  moved after its result is seen is not acceptance.
- **The GEMV provenance gate below closed on 2026-09-27** on the owner's
  judgment (D-080): the kernel is core-eligible.

## Pinned inputs

| Item | Identity |
| --- | --- |
| FP16 fixture | D-051 official GGUF; identities in [first-slice.md](first-slice.md) |
| EXL3 fixtures | D-052 4.0 bpw and mixed 4.5 bpw; identities in [exl3-bringup.md](exl3-bringup.md) |
| GGML source | `ggml/` subtree of llama.cpp [`b29c606e2`](https://github.com/ggml-org/llama.cpp/tree/b29c606e28a01b1bc8c1351026a0fa6e616bf6c4/ggml), root MIT; compiled into llmpalooza's build under owned dispatch (D-053) |
| EXL3 kernels | ExLlamaV3 [`6b84a21b`](https://github.com/turboderp-org/exllamav3/tree/6b84a21b6f1e5da3f291b9e1019061f0de788279), root MIT; the selected closure below |
| llama.cpp reference | Digest-pinned image: GNU 14.2.0, cudart 13.3.29, cuBLAS 13.5.1.27 ([pins](experiments/first-slice/pins.json)) |
| ExLlamaV3 reference | PyTorch 2.14.0+cu130, NVCC 13.0.88 `-O3 --use_fast_math`, ARM host-helper patch ([report](experiments/exl3-reference/README.md)) |
| Native toolchain | D-032: LLVM 22.1.8, NVCC 13.4.92, `sm_121`, C++23; cuBLAS 13.8.0.4, linked dynamically (D-076) |
| Hosts | Workstation: CPU-only, fake-backend and guardrail builds. `spark`: all CUDA, VMM, I/O and timing stages. `spark-b`: repeat evidence |

## Source findings at the pins

Read-only source inspection on 2026-09-22. These are facts about the code,
not measurements. The proof confirms each on Spark before relying on it.
The GGML backend-runtime findings are why D-053 moves dispatch into llmpalooza.
They also list what adapted launchers must not inherit.

### GGML (llama.cpp `b29c606e2`)

- **Tensors over llmpalooza memory.** Launchers dereference `tensor->data`, but
  some compute paths query `src->buffer` (usage and allocation size), so
  tensor descriptors still need a buffer object. The CUDA backend has no
  buffer-from-pointer entry point ([`buffer_from_host_ptr = NULL`](https://github.com/ggml-org/llama.cpp/blob/b29c606e28a01b1bc8c1351026a0fa6e616bf6c4/ggml/src/ggml-cuda/ggml-cuda.cu#L5590)).
  The workaround is a wrapper buffer: `ggml_backend_buffer_init` with the
  CUDA *buffer type*, a llmpalooza interface and a llmpalooza address range. That
  buffer passes the always-on asynchronous-transfer asserts, which compare
  buffer-type pointers ([example](https://github.com/ggml-org/llama.cpp/blob/b29c606e28a01b1bc8c1351026a0fa6e616bf6c4/ggml/src/ggml-cuda/ggml-cuda.cu#L2449)),
  and it passes `supports_buft`. GGML casts a buffer's context only inside
  CUDA's own buffer functions. Place tensors with `ggml_backend_tensor_alloc`.
  Nothing in `ggml-cuda` queries pointer attributes or needs `cudaMalloc`
  memory. `ggml_backend_buffer_init` and the event layout are declared only in
  the uninstalled `ggml-backend-impl.h`: a pinned internal interface that may
  change on upgrade. GGML's own CUDA buffer type allocates with
  [`cudaMalloc`](https://github.com/ggml-org/llama.cpp/blob/b29c606e28a01b1bc8c1351026a0fa6e616bf6c4/ggml/src/ggml-cuda/ggml-cuda.cu#L883-L900)
  and must never hold model data.
- **Activations.** If llmpalooza reuses GGML's graph allocator to plan activation
  offsets: `ggml_gallocr` obtains memory only through its buffer type's
  `alloc_buffer` and does not check the returned buffer's type. A llmpalooza
  buffer type can return wrapper buffers, putting the compute buffer in
  charged workspace. `ggml_gallocr_reserve_n_size` sizes it without
  allocating; `ggml_gallocr_reserve` allocates. Owned dispatch uses neither
  `ggml_backend_sched` nor GGML's CUDA graph-compute loop.
- **Streams.** GGML's backend creates and owns non-blocking streams and
  exposes no public way to supply or read one. Stock buffer set/get/clear
  functions use `cudaStreamPerThread` plus a synchronization and are
  unordered with compute. Owned dispatch does not use them; it supplies its
  stream through the launcher context below.
- **Operation launchers.** `ggml_cuda_op_*` functions take a
  `ggml_backend_cuda_context` and draw their stream, scratch and cuBLAS
  handle from it. Scratch goes through `ggml_cuda_pool`, an
  [abstract allocate/free interface](https://github.com/ggml-org/llama.cpp/blob/b29c606e28a01b1bc8c1351026a0fa6e616bf6c4/ggml/src/ggml-cuda/common.cuh#L1207-L1212).
  The stream, handle and pool members are public and are created only when
  absent, so llmpalooza can supply its own. Pre-setting the cuBLAS handle also
  skips GGML's workspace allocation and handle setup (stream binding,
  `CUBLAS_TF32_TENSOR_OP_MATH`, 32 MiB workspace); a supplied handle
  reproduces that setup or records its own in the numerical plan.
  Matrix-multiply routing among MMVF, MMF, MMVQ, MMQ and cuBLAS
  ([`ggml_cuda_mul_mat`](https://github.com/ggml-org/llama.cpp/blob/b29c606e28a01b1bc8c1351026a0fa6e616bf6c4/ggml/src/ggml-cuda/ggml-cuda.cu#L1823-L1877))
  and the cuBLAS path are `static` in `ggml-cuda.cu`; only per-kernel entry
  points are external. That file also defines symbols every launcher needs
  (`ggml_cuda_info`, `ggml_cuda_error`, the pool factory, the context
  destructor), and `ggml_cuda_info()` runs device initialization lazily on
  the first launcher call. Environment switches read at launch
  (`GGML_CUDA_CUBLAS_COMPUTE_TYPE`, `GGML_CUDA_PDL`) change numerics or launch
  attributes; the build fixes them.

  The context adapter needs build-time changes for:
  - the context [destructor](https://github.com/ggml-org/llama.cpp/blob/b29c606e28a01b1bc8c1351026a0fa6e616bf6c4/ggml/src/ggml-cuda/ggml-cuda.cu#L700-L721), which
    destroys whatever streams, handles, workspaces and pools it holds;
  - GGML's device initialization, which carries the GB10 device-flag side
    effect below;
  - recoverable failures: `CUDA_CHECK` and `CUBLAS_CHECK` still call
    [`ggml_cuda_error`, which aborts](https://github.com/ggml-org/llama.cpp/blob/b29c606e28a01b1bc8c1351026a0fa6e616bf6c4/ggml/src/ggml-cuda/ggml-cuda.cu#L100-L108).
    Pool consumers also use allocation results without checking for null.
    Supplying a bounded pool alone is insufficient: preflight its complete
    scratch requirement before launch and adapt error propagation through
    the selected launchers, or use an owned launcher. Failed submissions
    retain already-submitted work's resources until D-048 retirement; no
    exception may cross a C ABI.

  `ggml_cuda_pool_alloc` releases its pool allocation as the host launcher
  returns, while device work can still be in flight. The adapter may reuse
  those offsets in stream order, but must retain the workspace's backing
  and charge until all consumers retire. Pool `free` is not catalog release.

  Fusion decisions live in GGML's graph-compute loop (`ggml_cuda_can_fuse`),
  which owned dispatch replaces. Fused launchers such as
  `ggml_cuda_op_rms_norm_fused` are separate implementations that the plan
  may choose. Runtime-API launchers bind to the current runtime context; P1
  checks that against llmpalooza's context.
- **Hidden allocations in GGML's own backend.** None of these can be capped,
  pre-sized or queried through an API. Under owned dispatch the pool becomes
  a llmpalooza `ggml_cuda_pool` over declared workspace, and the cuBLAS handle and
  workspace are llmpalooza's. The usage analysis after this list sizes that
  workspace.
  - **Scratch pool.** One per device and stream. By default it reserves
    [32 GiB of virtual address space](https://github.com/ggml-org/llama.cpp/blob/b29c606e28a01b1bc8c1351026a0fa6e616bf6c4/ggml/src/ggml-cuda/ggml-cuda.cu#L536)
    and grows with device-location `cuMemCreate`. It never shrinks, and
    [growth failure aborts the process](https://github.com/ggml-org/llama.cpp/blob/b29c606e28a01b1bc8c1351026a0fa6e616bf6c4/ggml/src/ggml-cuda/ggml-cuda.cu#L580-L597).
    `GGML_CUDA_NO_VMM` selects a `cudaMalloc` pool instead.
  - **cuBLAS.** A lazy handle per stream plus a `cudaMalloc` workspace:
    [32 MiB at compute capability 9.0 and above](https://github.com/ggml-org/llama.cpp/blob/b29c606e28a01b1bc8c1351026a0fa6e616bf6c4/ggml/src/ggml-cuda/common.cuh#L1540-L1556),
    which includes GB10. cuBLAS's own internal memory comes on top.

  Among the dense Qwen2 operations, only matrix multiplications routed to
  cuBLAS use the pool. With F16 weights these are the ones with more than
  16 activation columns: the pool holds an F16 copy of the activations and
  an F16 output temporary. Smaller batches use MMVF/MMF kernels. With Flash
  Attention off, as in the reference, the KQ and KQV products follow the
  same routing. llama.cpp's graph marks KQ `GGML_PREC_F32`, so on the cuBLAS
  path the pool also holds an F32 copy of the K view, and KQV an F16 copy
  of the scores. Pool peaks therefore grow with context and prefill chunk.
- **Process side effect on GB10.** Device initialization on compute
  capability 12.1 calls
  [`cudaSetDeviceFlags(cudaDeviceScheduleSpin)`](https://github.com/ggml-org/llama.cpp/blob/b29c606e28a01b1bc8c1351026a0fa6e616bf6c4/ggml/src/ggml-cuda/ggml-cuda.cu#L366-L369),
  a process-wide synchronization policy that also affects llmpalooza's own waits.
  D-053 patches it out.
- **Duplication hazards.** Two settings convert whole F16 weights to F32 in
  the pool, which never shrinks: `GGML_PREC_F32` on a weight multiply, and
  `GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32`. CPU extra buffer types (repack, AMX)
  duplicate weights. KleidiAI makes transient per-call copies. Standalone
  defaults keep CUDA graphs and llamafile SGEMM off, whereas llama.cpp's own
  build enables both. A standalone build must still set the NCCL option off.
- **CUDA graphs**, if ever enabled, capture virtual addresses and recapture
  when node addresses or shapes change. Remapping physical backing behind
  the same address is invisible to GGML. Fusion is on by default and is part
  of the numerical plan (first-slice.md, RE-010). GGML's CUDA backend has
  **no custom operation**, so an EXL3 linear cannot be a node inside a GGML
  CUDA graph. Owned dispatch sidesteps this by sequencing both kinds of
  launch on one stream.
- **CPU backend.** `ggml_graph_plan`/`ggml_graph_compute` accept a
  caller-supplied work buffer and thread pool and compute on `tensor->data`
  only. Device VMM cannot be mapped for the CPU (D-081), so CPU diagnostics
  run on a copy of the backing. `ggml_init` accepts a caller-supplied metadata buffer.
  Spark's CPU heap draws on the same physical budget. This graph-compute
  path is a diagnostic, not a serving implementation under D-053.
- **Tied weights.** One tensor can feed both the embedding lookup and the
  output multiply. No placement restriction applies.

### ExLlamaV3 (`6b84a21b`)

- **Kernel closure for the two fixtures** (`mcg`, K=4/5/6/8, full-length
  side vectors):
  - `exl3_gemm_kernel` for each K, with FP16 and FP32 outputs and tile
    shapes 1–3. Shape 4 never fits these widths.
  - The K=4 GEMV kernel, the only rate these fixtures use in the GEMV range.
  - The fused gate/up `exl3_mgemm_kernel`, which upstream uses at up to
    32 rows.
  - `reconstruct` and `reconstruct_had` for each K, the 128-point Hadamard
    kernels, cuBLAS `GemmEx` with FP32 compute for reconstructed prefill,
    and an FP16 bias add.

  Upstream's Qwen2 writes [FP32 outputs for `o_proj`, gate, up and down](https://github.com/turboderp-org/exllamav3/blob/6b84a21b6f1e5da3f291b9e1019061f0de788279/exllamav3/architecture/llama.py#L93-L112).
  Output dtypes are part of the numerical plan. Every Qwen2.5-0.5B dimension
  is a multiple of 128, so no padding fallback applies. Other codebooks,
  fractional rates, MoE, tensor-parallel and packed-sign paths are outside
  this closure.
- **Separation.** Kernels are templates over raw pointers and integers, in
  headers with no ATen types. The host wrappers use ATen tensors, the current
  PyTorch stream and blas handle, `TORCH_CHECK` and `at::empty`, so they are
  rewritten, not ported. The reconstruct and Hadamard kernels share `.cu`
  files with their wrappers and must be split (done in P3 by a patch that
  removes the wrappers, below). Kernel headers rely on the
  including file for some macros. Upstream's
  [`cuda_check` calls `exit`](https://github.com/turboderp-org/exllamav3/blob/6b84a21b6f1e5da3f291b9e1019061f0de788279/exllamav3/exllamav3_ext/util.cuh#L92-L100);
  native wrappers return errors.
- **Extension state.** A per-device context holds 4,202,760 B of lock slots
  and a 16 MiB workspace that serves as the cuBLAS workspace. Kernels require
  zeroed lock slots and reset them on normal exit. One lock area is not safe
  for concurrent GEMMs on different streams: serialize, or give each admitted
  stream its own zeroed area.
- **Autotuning.** It times candidate tile shapes and grids with live buffers,
  allocates an L2-thrash buffer of up to twice the L2 size through the
  PyTorch allocator, synchronizes, and appends results to a
  cache file under `~/.cache/exllamav3`. The cache key omits the driver,
  toolkit and build identities. The chosen grid changes the split-K
  partition and thus FP16 rounding. The native plan therefore records a
  fixed shape and grid per case, obtained by an explicit tuning job outside
  the catalog lock. Forced shape/grid arguments give upstream the same plan
  for comparison. The reference's tuned choices were not recorded; P0
  records them.
- **Graphs.** Upstream decode captures attention and gated-MLP blocks.
  Capture forces a device-wide synchronization and fixes trellis, side
  vector, scratch, lock and pointer-table addresses at capture. Nothing
  invalidates a captured graph when weights move. The native path rejects
  capture until a relocation proof exists (exl3-bringup.md).
- **Alignment.** Source-derived minimums are 16 B for the trellis, the
  transformed input and FP32 outputs, and 8 B for side vectors, activations
  and FP16 outputs. Upstream packs small tensors at 256 B. Use at least
  256 B inside shared extents unless the proof shows cuBLAS and kernel
  selection are unchanged at smaller alignment. *Measured in P3:* every
  linear of both fixtures gives upstream's bits with every operand at
  exactly those minimums (biases at 2 B, the reconstruction GEMM's
  operands at the 16 B its pinned algorithms assume), and no kernel reads
  or writes outside its operands
  ([report](experiments/backend-proof-p3/README.md#placements-alignment-and-over-read)).
- **Numerics.** Accumulation is FP32 on GB10, and split-K partials are
  combined in a fixed lock-ordered sequence. The GEMV kernel accumulates in
  FP16 and folds into FP32. There are no floating-point atomics. For a fixed
  plan, grid and build, the dense path is repeatable. Across builds, results
  can differ with compile flags (`--use_fast_math`), tuning choices and cuBLAS
  heuristics.
- **GEMV provenance gate.** The K=4 GEMV kernel header describes a
  ["QTIP-style structure"](https://github.com/turboderp-org/exllamav3/blob/6b84a21b6f1e5da3f291b9e1019061f0de788279/exllamav3/exllamav3_ext/quant/exl3_gemv_kernel.cuh#L3-L4)
  and cites QTIP's `qtip-kernels/src/inference.cu`. QTIP's repository is
  GPL-3.0 ([licensing.md](licensing.md#early-exl3-companion-d-052)). On
  2026-09-27 the owner judged it not copyleft and closed the gate (D-080):
  GEMV is MIT and core-eligible. Before that, the native plan ran the EXL3
  GEMM kernel wherever upstream would select GEMV (m ≤ 8; upstream's
  `EXL3_GEMV=0`), and the owner accepted the gap for M2 (D-079). That
  acceptance expired with the gate: BP-F2 is gated against EXL3-O (its
  cases and reference arm are set at P3 entry), so the native plan needs
  the GEMV port. A native GEMV-on plan's GEMV linears are
  judged exactly against EXL3-O's at the same forced plan, and its full
  model against Tier C. The port landed in P3, exact against EXL3-O
  ([report](experiments/backend-proof-p3/README.md)).

## Dispatch and implementations (D-053)

Llmpalooza's dispatcher launches every operation on a llmpalooza stream, with
llmpalooza workspace and library handles. GGML's backend runtime does not execute
model work.

**GGML-derived operations.** For each operation, the proof chooses between
two approaches and records the choice:

- **K-C (context adapter).** Call GGML's CUDA operation launchers with a
  llmpalooza-populated context: llmpalooza's stream, its cuBLAS handle and workspace,
  and a `ggml_cuda_pool` over declared, charged workspace. Build-time patches
  cover context ownership on destruction, the GB10 device flag and error
  propagation through the selected launchers. Scratch sizing is checked
  before submission, and pool reuse preserves completion-owned lifetimes.
  Tensor descriptors use wrapper buffers over llmpalooza memory. Prefer this
  where it holds: it reuses upstream launch selection with the least new code.
  For matrix multiplication that selection is `static` in `ggml-cuda.cu`, so
  reuse needs a linkage patch there or a recorded llmpalooza copy.
- **K-L (lifted kernel).** Call the device kernel from a llmpalooza launcher.
  Use this when a launcher needs more than the adapter or small patches
  provide. The launcher keeps upstream's launch-parameter selection (D-013)
  or records the difference in implementation identity.

As built in P1 (D-077), K-C needs no destructor or device-flag patch.
Llmpalooza never compiles `ggml-cuda.cu`: it defines the five symbols the
launchers take from it, among them the device table, and the context
destructor, in `src/kernels/ggml/ggml_support.cu`. One patch drops
`ggml_cuda_error`'s `[[noreturn]]` so that errors propagate. Descriptors
carry no buffer, since the selected launchers never read one; a launcher
that does needs a wrapper buffer. Matrix multiplication does not reuse
GGML's `static` routing: each kernel family (MMVF, MMF, cuBLAS) is its own
implementation, and it accepts only operands that upstream's selection
would route to it. The cuBLAS path, `static` too, is a recorded llmpalooza
copy (`src/kernels/ggml/mul_mat_cublas.cu`). One host plan drives it,
fixing its conversions, entry point and scratch bound before launch. It
runs on a handle llmpalooza creates as upstream sets it up (TF32 math, the
context's stream, a declared workspace, whose size the FP16 gate fixes at
upstream's) and lends to the context. The handle refuses a cuBLAS other
than the pinned one and cuBLAS's own numerics switches in the environment.
A context without a handle refuses the path, so GGML never creates a
handle or workspace of its own.

Every GGML kernel in the FP16 bridge's recorded plan now has an
implementation (`src/kernels/ggml/ops.h`), fused launchers included:
RoPE with the K write, MMVF with a bias or residual add, and MMVF with
gate, up and SwiGLU. Each fused one is called with the node upstream
writes (the add, the GLU), so the launcher reads its precision from the
same parameters as upstream's: under the GLU, whose first parameter is its
GLU operation, F16 weights accumulate in F32, which is one of the precision
rules fusion changes. Where a launcher chooses its kernel from the operands
and addresses (get_rows' vector kernel by alignment, cont's copy or
kernel), the host check predicts the choice; soft_max's column template
follows from the column count alone.

Either way, operation scratch must fit declared workspace (BP-A1/A2).
Library handles and unavoidable driver/library allocations are separately
bounded and charged; supplying a cuBLAS workspace does not account for all
of cuBLAS's internal memory. Workspace exhaustion must return an error,
never abort (BP-V2). Fusion is an explicit plan choice among fused and
unfused implementations. GGML's graph-compute loop, and the fusion checks
in it, do not run. To fuse where FP16-F's bridge fused, a planner asks
llmpalooza's copies of upstream's gates (`src/kernels/ggml/fusion.h`), which
take the model graph's nodes in GGML's order and reproduce, per pattern,
upstream's conditions on operations, edges, uses and overlapping data
ranges.

**EXL3-derived operations.** Lifted kernels behind llmpalooza launchers
(exl3-bringup.md). The dispatcher sequences them and GGML-derived operations
on the same stream. No segment boundaries or cross-stream events are needed.

As built in P3, the kernels are the source lock's, built with the
reference's device flags: upstream's GEMM units and llmpalooza's instance unit
(`llmp/llmp_exl3_kernels.cu`, patch 0002) over the GEMV header and the
reconstruction, Hadamard and bias-add sources, which patch 0003 reduces to
their kernels. Their SASS equals the reference extension's. The launchers
(`src/kernels/exl3/`) replace upstream's ATen wrappers:
- a launch context on a provider stream holds a declared lock area,
  zeroed on that stream before its first launch and shared with no other
  live context, and the device's co-resident limit per cooperative
  kernel; a launch error faults it (`launch.h`);
- host checks in every profile refuse, before anything is queued, operands
  a kernel would read or write out of bounds, under-aligned or overlapping
  operands, and cooperative grids the device cannot hold at once
  (`validate.h`);
- the forced plan is data: tile shape and grid (and concurrency) as the
  decoded tuning record gives them, the GEMV's configuration and grid, and
  each reconstruction slice's pinned cuBLASLt algorithm with the GEMM it
  was pinned for, so a pin never runs another GEMM or an unpinned size
  (`recon_gemm.h`); the multi-GEMM's pointer tables come with the
  caller's record of what they hold, and a launch whose record is not its
  operands' weights is refused (BP-P5's stale-table case, before launch);
  a recorded copy of upstream's GEMV choice says where EXL3-O takes the
  GEMV (`upstream_gemv.h`);
- each linear path runs upstream's kernels in upstream's order, with the
  bias through `add_kernel_hhh` on every path (`linear.h`);
- the registry declares the GEMM, the GEMV (the pair of implementations of
  one operation at up to eight rows), both reconstruction paths and the
  fused gate/up multi-GEMM (`implementations.h`).

**Coexistence and swapping** are proof obligations, not later features.

- One build holds at least two implementations of one operation. The plan
  selects between them, and each passes its own reference comparison and
  envelope.
  - Natural first case: EXL3 GEMM versus GEMV at m ≤ 8 (core since
    D-080).
  - In the core build: GGML's fused versus unfused RMSNorm, inside the fused and
    unfused plans that have bridge fusion arms as oracles, or its MMF versus
    cuBLAS matrix-multiply paths at a shared shape.
- Implementation identity is part of plan identity. The implementation
  registry resolves every operation or rejects the plan; nothing is silently
  substituted.
- The FP16 and EXL3 models, using different kernel sources, are resident and
  run alternately in one process.

As built in P1, the core case is GGML's fused RMSNorm-mul
(`ggml.rms_norm_mul.fused`, FP16-F's launcher) against rms_norm then mul
(`ggml.rms_norm_mul.unfused`, FP16-U's). Both take the same pair of nodes.
The unfused one writes the norm into memory of its own, an intermediate the
plan provides, and refuses a norm over its input or the weight, so on nodes
both accept the two differ only in that intermediate. (Like GGML with
fusion off, the unfused one also accepts a norm broadcast as the mul's
second operand, which the fused one refuses.) The registry
(`src/execution/registry.h`) holds each compiled implementation's
identity: name, operation, source, the prepared source tree's digest, a
digest of every file of llmpalooza's own code in the module (written at build
time, so any edit there changes the identity), the SDK, target, device
architecture, build type, libstdc++ assertions (D-083) and sanitizers,
and a variant naming the launchers. A plan records each operation's
implementation and identity, and the plan's identity is a SHA-256 of them in
order. Resolution binds every operation or rejects the plan: a name the
build lacks is unsupported (BP-S4), a changed identity stale (BP-S2), and no
other implementation is ever bound in its place. The GGML module turns only
its own current declarations into kernels. Until P6 settles the operation
contract and the build-generated table, each program assembles the registry
from the modules it links.

**Dispatch overhead.** Upstream decodes these fixtures at about 3.4–3.7 ms per
token with captured blocks ([report](experiments/exl3-reference/README.md)),
and a decode token runs 169 linear layers plus norm, RoPE and attention
launches. Host cost per launch can
therefore threaten M3 parity even when M2 kernel parity passes. BP-F4
measures it early. Llmpalooza-owned graph capture remains possible only under
the captured-pointer relocation rules.

## Numerical oracles

Each rung isolates one source of difference. A difference is localized, not
absorbed into a tolerance.

1. **Reference.** The pinned external runs, as already recorded.
2. **Toolchain bridge.** The pinned llama.cpp source and the same reference
   harness, built with the llmpalooza SDK. This separates compiler, CUDA and
   cuBLAS effects from integration. For EXL3, the equivalent bridge compiles
   the upstream kernel sources with llmpalooza's NVCC and upstream's flags, and
   runs them on captured inputs with forced shape and grid.
3. **Native dispatch, conventional memory.** Llmpalooza's dispatcher on
   `cudaMalloc` memory. Expect bit-identical logits to the bridge when the
   plan reproduces the bridge's kernel selection, launch parameters, fusion,
   batch splits, cuBLAS paths and handle setup. Compare an unfused plan with
   a fusion-disabled bridge arm, the counterpart of the recorded reference
   arm (first-slice.md).
4. **Native, llmpalooza device VMM** loaded through the host-VMM landing zone
   (D-081). Expect results identical to rung 3.
5. **Native after eviction, restoration or relocation.** Must be identical to
   rung 4.

The cross-implementation bounds are declared before native evaluation:
native versus the image reference, and full-model EXL3 versus ExLlamaV3,
whose norm, RoPE and attention run in PyTorch or Triton. Localize EXL3
error with per-layer teacher-forced comparisons at the declared dtypes.
Individual packed linears must be byte-identical to upstream at the same
plan and inputs, or the bridge must explain the difference. Instrumented
observation points can change CUDA plans (RE-010). Every instrumented run
has an uninstrumented control.

**Inputs.**

- **FP16:** the existing 76-token trajectory, plus rows either side of the
  16/17-column MMF/cuBLAS boundary and a larger prefill chunk.
- **EXL3:** the existing prefixes and the declared held-out trajectory. Rows
  1, 8, 9, 16, 32, 33, 144, 145, 1,023 and 1,024 cover GEMV, the fused
  gate/up range, packed/reconstruct and fused-reconstruct boundaries.

P0 declares the finite context and prefill-chunk profile for each fixture.
First-slice context and envelope limits apply.

## P0 declarations

**Status.** Approval, or pre-registration under D-079, is by part, each
before the native output it governs.

- **Approved by the owner on 2026-09-26, in force:**
  - the numerical profiles;
  - the Tier E items marked *approved* below: the FP16 exactness gate with
    its recorded-plan match, rungs 4 and 5, and the EXL3 packed linears
    with their reconstructed weights.

  These are all that native FP16 work (P1, P2) needs.
- **Approved by the owner on 2026-09-26, in force (EXL3):**
  - Tier C: the rule, the bounds in `tierc.json` and their calibration;
  - the reconstruction-path Tier E item, scoped to the recorded chunk
    sizes;
  - the kernel-timing rule's statistic and decision procedure
    (`timing_protocol.py` at `c05fd2dd…`);
  - the native EXL3 operation plan and its record, `exl3-op-plan.json`;
  - the operation-level Tier E item;
  - the persistent-workspace limit.
- **The P3-entry items: pre-registered under D-079 (2026-09-27),** once
  the ExLlamaV3 port existed and before any native EXL3 timing or memory
  result was seen:
  - *BP-F2's reference arm,*
    [with the performance protocol](#performance-protocol-rule-approved-2026-09-26-bp-f2s-reference-pre-registered-at-p3-entry):
    gated against EXL3-O (GEMV on; D-079's GEMM-only acceptance expired
    with the gate, D-080), the fused gate/up kernel added to the cases,
    ExLlamaV3's bias add timed rather than PyTorch's, one frozen tuning
    cache governing the model plan and the timing cases, and the SASS
    match of the port (done in P3: every ExLlamaV3 function the port's
    binary holds has the SASS of its namesake in the NVCC 13.4.92
    reference, `aa8b9f16…`,
    [report](experiments/backend-proof-p3/README.md#sass)). The changed
    case set gets a new calibration and holdout under the approved rule,
    run with BP-F2 on `spark`; the NVCC 13.4.92 calibration and holdout
    below validated the rule's mechanics on the earlier cases.
  - *The EXL3 phase memory limits,* tightened against native's itemized
    buffer plan
    ([memory and workspace](#memory-and-workspace-the-m2-gate-in-exl3-bringupmd)).
    Since D-085 a native EXL3 run is judged by the coarse memory check
    against EXL3-O instead of a census.
  - *A recorded phase kind* the trajectories reach but P0's record lacked:
    the single-token step with K padded to 1,024 (the first step after
    the 1,023-row prefix), recorded from a reference-only run, with
    EXL3-O's record
    ([native EXL3 operation plan](#native-exl3-operation-plan-approved-2026-09-26)).
- **Delegated by the owner on 2026-09-26 (D-079):** the declared-departure
  contingency, the FP16 memory limits, BP-F1's calibration, the
  retained-backing criteria, the P3-entry items above and the M2 acceptance
  of EXL3-G's GEMM-only gap. Each is settled by the default D-079 records
  and written here, with its evidence, before any native result it judges
  is seen.
  The retained-backing criteria are pre-registered, with the trace's
  identity, [below](#retained-backing-comparison).
  - *Pre-registered on 2026-09-27 (D-079):* BP-F1's rule, frozen before any
    host-VMM timing ran: its 53 cases, operand placement, calibration and
    `z`, with a passing holdout
    ([BP-F1](#performance-protocol-rule-approved-2026-09-26-bp-f2s-reference-pre-registered-at-p3-entry),
    [report](experiments/backend-proof-p1/README.md)). Applied on
    2026-09-27, it fails: host VMM is slower. The owner answered with
    D-081 (device VMM). BP-F1 rule v2, for device VMM, was pre-registered
    the same day with its own harness, calibration and a passing holdout,
    before any device-VMM session ran on `spark`. Applied the same day, it
    passes: no case fails, and the aggregate passes in both sessions.
  - *Pre-registered on 2026-09-27 (D-079), before any native FP16 run:*
    the FP16 memory limits and M2's census rule, which D-085 replaced
    with a coarse peak check
    ([memory and workspace](#memory-and-workspace-the-m2-gate-in-exl3-bringupmd)).

Each part is approved, or pre-registered under D-079, before any native
result it would judge is seen.
Five rounds of review and challenge shaped the EXL3 parts. Their basis is
in the report: the
[third pass](experiments/backend-proof-p0/README.md#exl3-third-pass-calibrating-the-full-model-bound),
the [GGML operation study](experiments/backend-proof-p0/README.md#ggml-operations-for-native-exl3),
the [pinning probe](experiments/backend-proof-p0/README.md#reconstruction-gemm-pinning)
and the [timing controls](experiments/backend-proof-p0/README.md#timing-controls).

The measured basis is the [P0 report](experiments/backend-proof-p0/README.md),
all of it reference runs on `spark`:

- both toolchain bridges;
- the EXL3 reference arms (three passes), an FP64 oracle and the Tier C
  calibration;
- the FP16 bridge's recorded executed plan;
- the GGML operation study, the cuBLASLt pinning probe and the kernel
  launch record;
- 20 timing sessions, plus one stopped unevaluated.

### Numerical profiles (approved)

| Profile | Fixture | Settings | Trajectories |
| --- | --- | --- | --- |
| FP16-F | FP16 GGUF | first-slice.md's settings: F16 K/V, one sequence, no flash attention, no CUDA graphs, fusion on | `control`: 76 tokens, context 512, batch 64, chunks 32 then 44 × 1, restore after 32. `heldout`: 577 IDs, context 1,024, batch 512, chunks 16, 17, 16 × 1, 512, 16 × 1, restore after 33 |
| FP16-U | FP16 GGUF | FP16-F with fusion off | as FP16-F |
| EXL3-G | 4.0 and 4.5 bpw | Upstream's optimized profile with GEMV off (`EXL3_GEMV=0`); the reconstruction GEMM pinned to cuBLAS (`EXL3_HGEMM_F16ACC=0`, what upstream's timing probe chooses on GB10); F16 cache of 4,096 tokens; the frozen GEMM-only tuning caches `tune-40-gemvoff` and `tune-45-gemvoff` | Prefixes of 32, 144, 145, 1,023 and 1,024 held-out IDs, every prefill row's logits, then 16 single-token steps |
| EXL3-O (the GEMV-on reference, D-079; BP-F2's gated reference since D-080) | 4.0 and 4.5 bpw | EXL3-G with GEMV on, caches `tune-40` and `tune-45` | as EXL3-G |

The trajectories, the chunking and the harnesses are the P0 report's. The
held-out IDs have SHA-256 `6dd8da89…`. The caches' bytes and decoded choices
are in its `results.json`.

- **Context.** The proof covers FP16 trajectories of up to 577 tokens in a
  1,024-token context, and EXL3 up to 1,040 tokens in a 4,096-token cache.
  It covers nothing beyond, timing included.
- **GEMV.** The GEMV provenance gate closed on 2026-09-27 (D-080), so
  GEMV is core. EXL3-O is BP-F2's gated reference, with its cases set at P3
  entry, and the exact reference for a native GEMV-on plan's GEMV linears.
  The approved EXL3-G bounds and gates below are unchanged.
- **Per-linear sweep (BP-N5).** Rows 1, 8, 9, 16, 32, 33, 144, 145, 1,023
  and 1,024, on every real projection of both fixtures. It runs at a forced
  plan, the same tile shape, block, SMs and concurrency on both sides,
  recorded per case.

### Tier E: exact

A difference in any of these is a defect, to be localized and fixed. It is
never bounded. Logits and restored storage must be bit-identical.

- **FP16, native against the bridge (approved)**, on both FP16 profiles and
  both trajectories (rung 3), and therefore against the image reference as
  well. *Applied 2026-09-27 at rung 3 (`cudaMalloc`, `spark-b`):* on all
  four arms the native plan matches `fp16-plan.json` completely and then
  the logits are bit-identical, with BP-S1 exact on both implementations
  ([P2 report](experiments/backend-proof-p2/README.md)).
  - The native GGML kernels are built as the bridge's are, for `sm_121a`
    (GGML's CMake maps `121-real` to it), since the SASS hashes must match.
  - The native executed plan must first match the bridge's recorded plan
    (`fp16-plan.json`), per chunk shape. That means the ordered kernel
    sequence with each kernel's SASS hash, grid, block and shared memory;
    every cuBLAS call's parameters and resolved algorithm; and the handle's
    state: a 32 MiB workspace, TF32 math mode and the SM count.
  - The plan also has to reproduce the conditions the record lists:
    - the `n_kv` padding to 256 and the KV cell layout;
    - the MMVF/MMF and softmax variants chosen by `n_kv`;
    - the fusion gates that compare data ranges;
    - `get_rows`' alignment choice and 128-byte buffer alignment;
    - embedding lookup on the host;
    - the precision rules that fusion changes.
  - Only stream identity, addresses (subject to those alignment conditions)
    and the PDL launch attribute may differ.
  - Logits are compared only after the plans match. The
    [plan comparator](#p2-prerequisites) makes the comparison; only its
    complete match (exit 0) counts.
- **A declared departure (delegated, D-079).** A native plan may depart
  from the recorded plan only if the departure is written down before the
  run, with a bridge arm, built from a patch recorded in the report, that
  makes the same departure. Tier E then holds exactly against that arm.
  The challenge found exactness alone insufficient, so the departure also
  carries an accuracy bound for that arm against the recorded bridge,
  derived from reference measurements already recorded and written down
  with the departure, before the arm runs. A departure never answers a
  mismatch already seen: that is a defect, localized and fixed.
- **Rung 4 against rung 3, and rung 5 against rung 4 (approved)**, for every profile
  and fixture, including every eviction, restore and relocation arm and
  every repeat. *Applied 2026-09-27 for FP16 (`spark-b`):* on all four
  arms, paged into device VMM through the landing zone, the plan matches
  `fp16-plan.json` over four evaluations (the repeat and two restores, the
  second relocated), evaluation 1 has the bridge's logits, and the rest
  equal it bit for bit
  ([P2 report](experiments/backend-proof-p2/README.md#rungs-4-and-5-paged-into-device-vmm-through-the-landing-zone)).
  The cache stays resident at the restore point (no spill yet).
  *Applied 2026-09-27 for EXL3 (`spark-b`):* both fixtures in EXL3-G and
  EXL3-O, paged into device VMM through the landing zone: evaluation 1
  has rung 3's logits bit for bit, and the repeat and two restores after
  every prefill (the second relocated, the multi-GEMM tables rewritten)
  equal it; the executed plan of evaluation 1 matches the record
  ([P3 report](experiments/backend-proof-p3/README.md#rungs-4-and-5)).
- **EXL3 packed linears (up to 144 rows) (approved)** against upstream's kernel at the
  same forced plan and inputs, and reconstructed FP16 weights against
  upstream's reconstruction. *Applied 2026-09-27 (`spark-b`, BP-N5):* every
  packed case of both fixtures in EXL3-G and EXL3-O (GEMM, GEMV and the
  fused gate/up multi-GEMM at rows 1 to 144) and every linear's full
  reconstructed weights, rotated and fused, are bit-identical to upstream's,
  every intermediate buffer included, in `cudaMalloc` memory and device VMM
  ([P3 report](experiments/backend-proof-p3/README.md)).
- **EXL3 reconstruction-path linears (145 rows and more) (approved
  2026-09-26).** These are compared against upstream running cuBLAS
  13.8.0.4 (the report's substitution arm), at the same forced plan and
  inputs.
  - *Scope:* the recorded chunk sizes of 145, 1,023 and 1,024 rows. Before
    native runs the reconstruction path at any other size (BP-F3's
    512-row prefill, or BP-F2's synthetic shapes), that size's GEMMs are
    recorded and pinned in the same way, from a reference-only run.
  - Upstream's executed plan is recorded in
    [`exl3-recon-plan.json`](experiments/backend-proof-p0/exl3-recon-plan.json):
    21 distinct cuBLASLt GEMMs per fixture, with their layouts, compute
    descriptor (`COMPUTE_32F`, 48 SMs targeted), heuristic preferences
    (16 MiB workspace limit, 16-byte alignment, the logged `implMask`) and
    resolved algorithms.
  - The native plan pins each GEMM's complete algorithm configuration, all
    nine cuBLASLt attributes, recorded in
    [`exl3-recon-pin.json`](experiments/backend-proof-p0/exl3-recon-pin.json).
    None uses split-K or a workspace.
  - The report's pinning probe replayed all 21 GEMMs on the SDK's cuBLAS
    13.8.0.4 and showed:
    - the heuristic resolves the algorithms the arm logged;
    - an algorithm rebuilt from its configuration alone is bit-identical to
      ExLlamaV3's legacy `cublasGemmEx` call;
    - so is the heuristic's own choice.
  - Reconstructed FP16 weights must equal upstream's, and each GEMM's
    output must equal the arm's bit for bit. The per-linear harness maps
    only cuBLAS 13.8.0.4.
  - Native's cuBLAS kernel names and grids must equal the arm's. *Checked
    in P3* on an nsys trace of the per-linear sweep: every launch, cuBLAS's
    and ExLlamaV3's, is the arm's.
  - Under PyTorch's cuBLAS 13.1.1 the resolved algorithms differ for 19 of
    the 21 GEMMs, so the cuBLAS version is part of the plan.
  - No contingency is needed: pinning is shown to work. A future cuBLAS
    change requires a new record and approval.
  - *Applied 2026-09-27 (`spark-b`, BP-N5):* every reconstruction-path case
    of both fixtures (145, 1,023 and 1,024 rows, both arms) is
    bit-identical to upstream's at every step: the input transform, each
    slice's reconstructed weights, the GEMM, the output transform and the
    bias ([P3 report](experiments/backend-proof-p3/README.md)).
- **GGML-derived operations inside the EXL3 plan (approved
  2026-09-26).**
  - *Coverage.* Every native kernel in an EXL3 plan is either ExLlamaV3's
    (a linear or its bias add, gated above with the linear) or a
    GGML-derived operation gated here: norms (including the final norm),
    RoPE, attention (with its mask pre-pass and combine), SwiGLU gating,
    residual adds, embedding lookup and casts. The KV write is a byte copy,
    checked by the wiring rule below. The logits leave the plan as
    `lm_head`'s F16 output; widening them is exact and not a plan
    operation.
  - *Plan first.* Native's recorded executed plan must equal
    [`exl3-op-plan.json`](experiments/backend-proof-p0/exl3-op-plan.json)
    per phase kind before any operation or model comparison, as the FP16
    gate requires of `fp16-plan.json`. That means:
    - the same operations in the same order;
    - each launching the same kernels in the same order, with the same
      grid, block and shared memory. GGML kernels are identified by mangled
      name (NVCC's per-file `_INTERNAL_` hash normalized) and SASS hash from
      the NVCC 13.4.92 build; ExLlamaV3's and cuBLAS's by name;
    - the same linear paths and output dtypes.

    Only stream identity, addresses, the launch API and the PDL attribute
    may differ. A native that let GGML pick its MMA kernel for prefill fails
    here, before any accuracy check.
    - A phase kind is keyed by its row count, starting position and padded
      K length (n, P, Npad). Grids, the attention's parallel blocks, its
      mask pre-pass and the linear paths depend on all three.
    - An unrecorded kind needs a reference-only record, approved, before
      native runs it.
    - Besides the recorded kernels, native may only upload its host-built
      inputs (ids, positions, mask) and zero-fill K/V padding. It may launch
      no other kernel.
    - Kernel names are compared after normalizing NVCC's whole
      `_INTERNAL_…` token.
  - *Wiring.* Each operation's recorded input must equal, byte for byte,
    the recorded output of the operation that produced it; the record names
    the producer of every tensor. This catches a stale or mis-wired buffer
    that each operation's own check would pass. Attention's K/V input over
    cells `[0, Npad)` is wired too:
    - cells below `P + n` are the `kv_write` outputs of that layer, from
      every earlier phase of the trajectory and this one;
    - padded cells are the declared zeros.
  - *The check.* Native records each such operation's inputs and output,
    for every invocation in its own EXL3-G run. That recording must
    reproduce the uninstrumented output (RE-010). The bridge's GGML kernel
    (same source, flags and `sm_121a` build) recomputes the output from
    those inputs, and the two must be bit-identical.
  - *Where the bridge's inputs come from:*
    - its semantic parameters, from its own code, using the model's
      configuration and the trajectory: eps, RoPE base and positions,
      softmax scale `head_dim^-½`, `n_kv`, mask and KV slot. This harness
      is reference-only code in the P0 experiment directory. It shares no
      code with native's planner, and it is reviewed before any native
      output is seen;
    - its weights (norm scales, embedding, biases), from the artifact;
    - only implementation parameters (the kernel variant and launch
      configuration), from native's recorded plan, which the plan-first
      step has already matched to the record.
  - *Dtype chain.* Every tensor between operations must have the dtype and
    element format the record declares (its `tensors`). Casts are
    operations too, so a rounding to a narrower format and back fails the
    gate.
  - *Bias add.* The q/k/v bias add has one owner on every path:
    ExLlamaV3's `add_kernel_hhh`, in F16, gated with the linear. Its output
    must equal upstream's bit for bit, which on the reconstruction path is
    PyTorch's add (the native EXL3 operation plan below).
  - This gate is what catches the subtle faults the Tier C bound cannot
    see: a wrong scale, or a lower-precision intermediate in one layer.
  - *Applied 2026-09-27 (`spark-b`):* the plan first, on both fixtures in
    EXL3-G and EXL3-O: native's executed plan equals the record
    (`exl3-op-plan-g.json`, `-o.json`) in all 85 phases of 8 kinds
    (`op_plan_compare.py` exit 0). Then in EXL3-G, both fixtures: every
    GGML-derived operation of every phase and layer (30,855 per fixture),
    recomputed by the bridge's library from native's recorded inputs, is
    bit-identical; the wiring (80,166 inputs, K/V cells included) and the
    dtype chain (135,841 tensors) hold, and the recording reproduces the
    uninstrumented logits. The harness (`op_tier_e.py`) sits in P3's
    experiment directory. It was reviewed before any native numerical
    output was looked at (the native runs had completed; only their exit
    codes, the plan gate and native-to-native equality had been read), and
    changed after: it refused the first 4.5 bpw run (exit 2, before
    any verdict) by reading the fixture from `config.json`, and now reads
    it from the checkpoint's hash; and review and challenge added checks,
    never relaxed one: every weight an operation reads is recorded by the
    load-time hash of what lies at the address native bound for it and
    must be the artifact's tensor of that name, linear and layer (the
    embedding, norms and biases; every linear's trellis and side vectors;
    what each multi-GEMM table points at; none for an operation that reads
    no linear: 95,899 checks per fixture). Both
    fixtures were recorded and judged again after each change
    ([P3 report](experiments/backend-proof-p3/README.md#tier-e-operation-level-and-tier-c)).
- **BP-S1 (approved, part of the FP16 gate).** Each of the two
  implementations is exact against the bridge arm that uses it: fused
  RMSNorm against the fused arm, unfused against the unfused arm.

Basis:
- both bridges reproduce their references bit for bit, the EXL3 bridge now
  also over every 1,023- and 1,024-row prefill;
- every reference arm repeats and restores exactly;
- the executed-plan record reproduces its logits under profiling.

### Native EXL3 operation plan (approved 2026-09-26)

Native EXL3-G runs ExLlamaV3's kernels for the quantized linears (the
packed and reconstruction paths above) and their bias add. Its other
operations run GGML's CUDA kernels from the pinned `b29c606e2`, built as
the FP16 plan's are (NVCC 13.4.92, `-use_fast_math`, `sm_121a`).

**The plan is a record,**
[`exl3-op-plan.json`](experiments/backend-proof-p0/exl3-op-plan.json),
generated by [`exl3_op_plan.py`](experiments/backend-proof-p0/exl3_op_plan.py)
from real runs (the report's
[plan record](experiments/backend-proof-p0/README.md#the-operation-plan-record)).
It covers both fixtures and seven phase kinds: prefills of 32, 144, 145,
1,023 and 1,024 rows, and single-token steps with K padded to 256 and to
1,280 positions, the only two padded lengths the P0 trajectories reach.
Before native runs any other phase kind (another chunk size, another padded
length such as BP-F3's 768, or another parallel-block count), that kind is
recorded from a reference-only run in the same way, and approved. For
the embedding, one decoder layer (the same in all 24) and the output, it
lists in order:
- every operation, with its owner and GGML operation type;
- every kernel it launches, with grid, block and shared memory. GGML
  kernels also carry their mangled name and SASS hash;
- every tensor between operations, with its dtype and shape, and the
  operation that produces it;
- each linear's path and output dtype;
- the attention's parameters, the padding and mask rules, and the KV write.

The Tier E item above requires native's executed plan to equal the record
per phase kind before any comparison.

**P3's records** ([report](experiments/backend-proof-p3/README.md#part-2s-reference-side)),
made from reference-only runs in the same way before native ran them, under
D-079's delegation:
- [`exl3-op-plan-g.json`](experiments/backend-proof-p3/exl3-op-plan-g.json)
  adds the one phase kind the trajectories reach that P0's record lacks:
  the single-token step with K padded to 1,024, the first step after the
  1,023-row prefix. P0's seven kinds are unchanged in it but for step 32's
  `K_by_linear`, which no longer lists the next prefill's gate and up (P0's
  probe had appended them; no launch changed). The exhaustive bias-add
  check stays P0's; it was not rerun.
- [`exl3-op-plan-o.json`](experiments/backend-proof-p3/exl3-op-plan-o.json)
  is the same plan under EXL3-O: upstream's GEMV where it takes it in
  single-token steps, P0's EXL3-O tuning caches; everything else equals
  EXL3-G's.

Native's plan gate compares against these two.

In summary (the record has the kernels and launches):

| Operation | Plan | Against FP64 (the operation study) |
| --- | --- | --- |
| Embedding | GGML `get_rows` on the BF16 table, to F32 | bit-identical to upstream |
| RMSNorm, final norm | GGML `rms_norm` fused with `mul`, F32 on the F32 residual stream; cast to F16 for the linears | equal to upstream: bit-identical in 211 of 330 cases, one F16 rounding apart in the rest |
| q/k/v bias add | ExLlamaV3's `add_kernel_hhh`, F16, on every path | bit-identical to upstream (below) |
| RoPE (NEOX) | the F16 projections widened to F32, GGML `rope` in F32. Q stays F32 into attention; K is cast to F16 and copied into the cache with V | 0.003–0.16 times upstream's relative error |
| Attention | GGML's vector flash-attention kernel for every phase: F32 Q, F16 K/V and mask; output cast to F16 for `o_proj` | 2.3–4.1e-6 relative error against upstream's 2.2–3.6e-4. Its output rounded to F16: 0.82–0.99 times upstream's; with Q from F32 RoPE, as the plan chains them, 0.24–0.49 times |
| Residual adds | GGML `add` in F32 | bit-identical to upstream's |
| SwiGLU gating | GGML `swiglu` in F32 on F32 gate and up; cast to F16 | bit-identical to upstream |

The linears give F16 (q, k, v, `lm_head`) or F32 (`o_proj`, gate, up,
down) outputs, as upstream's do. Up to 144 rows they run packed; gate and
up run as one multi-linear up to 32 rows. From 145 rows they reconstruct,
and from 1,024 rows they use the fused reconstruction. The plan's output is
`lm_head`'s F16 logits.

**Bias add.** Upstream adds the bias inside its packed linear with
ExLlamaV3's `add_kernel_hhh`. On the reconstruction path it uses PyTorch's
elementwise add: F16 operands, F32 arithmetic, one rounding. The plan uses
ExLlamaV3's kernel on every path, so the bias add has one owner and one
precision and stays with its linear. The two adds are the same function:
- they agree on all 2^32 pairs of F16 inputs (every non-NaN result
  bit-identical, NaN in the same places);
- they agree on every reconstruction-path bias add of both probes.

A GGML F32 add would need F32 linear outputs, a dtype chain unlike
upstream's; it was not adopted.

**Attention** is declared explicitly:
- **Padding, in every phase.** Prefill and single-token steps alike attend
  K and V padded to a multiple of 256, as llama.cpp pads its cache. Padded
  cells must be finite: their scores are masked to −∞, so finite values
  get weight exactly 0, but Inf or NaN would poison the row. They are zero
  in the reference and declared zero.
- **Mask.** F16, one row per query: 0 up to the row's own position, −∞
  after it, including every padded column.
- **Decode.** With the padding, GGML would pick the vector kernel itself.
- **Prefill.** GGML would pick its MMA kernel. That kernel accumulates V·P
  in F16 and is up to 2.85 times *less* accurate than upstream at 1,023
  and 1,024 rows. Its precision flag is not read on CUDA. The plan
  therefore calls the vector kernel's launcher directly
  (`ggml_cuda_flash_attn_ext_vec_case<64, F16, F16>`) for every phase. From
  1,024 rows the launcher also runs a mask pre-pass that skips fully
  masked KV tiles.
- **Also rejected:** the tile kernel (13–24 times less accurate) and the
  non-flash path. On GB10 the non-flash path runs K·Q in TF32 through
  llama.cpp's handle, and it materializes the scores.

The vector kernel never materializes the scores. Its scratch is about
15 KB and 48 KB at decode (K padded to 256 and 1,280), and 0.36, 1.6 and
11.4 MB at 32, 144 and 1,024 prefill rows. It is counted in `E`, against
the memory limits below.

**Evidence.** The `ggml_ops` arm runs this plan inside upstream, with two
kinds of exception:
- operations left to upstream: the embedding and the MLP residual add
  (GGML's are bit-identical to them), and the reconstruction-path bias add
  (PyTorch's);
- its GGML library is built with GCC 13.3 as host compiler, as shared
  libraries, with `GGML_CUDA_GRAPHS=OFF`.

With and without cuBLAS 13.8.0.4 it passes every repeat, restore and
capture check. Its overall accuracy is within 1.5% of frozen EXL3-G's: the
full model is dominated by the F16 rounding at operation boundaries that
both share.

The record's probe runs the whole plan, those three operations included:
- its GGML library was built with NVCC 13.4.92;
- ExLlamaV3's extension was the container's NVCC 13.0.88 build
  (`7c9d383f…`), which is where the probe's ExLlamaV3 launches come from.

The probe's logits equal the arm's (cuBLAS 13.8.0.4) bit for bit in every
phase kind on both fixtures. Two more results:
- the four GGML kernels the FP16 bridge also launches have the same SASS
  in both builds, despite the different host compilers;
- the GGML library built with the container's NVCC 13.0.88 and with the
  SDK's 13.4.92 gave identical outputs for all 3,670 operation cases.

### Tier C: EXL3 full model against an FP64 oracle (approved 2026-09-26)

For BP-N6 the native plan cannot be exact. Upstream runs norm, RoPE,
attention, embedding and residuals in PyTorch or Triton, so the full model
has no common kernel to match. The bound is therefore accuracy against a
common oracle: a teacher-forced FP64 forward pass over the same decoded
EXL3 weights (`oracle.py`). Its calibration is in the report's
[third pass](experiments/backend-proof-p0/README.md#exl3-third-pass-calibrating-the-full-model-bound).

- **The rule.** For each fixture, the P0 report computes 750 statistics
  against the oracle, over every prefix of EXL3-G:
  - *averaged*: the logits' RMS error, prefill and single-token steps
    separately, and each block's row-averaged relative RMS error;
  - *extreme*: the worst row's logit RMS error, the largest absolute logit
    error, each block's worst-row relative error, and K's and V's worst
    position (RMS error relative to the oracle's magnitude at that
    position), over the prefix and over the positions the single-token
    steps wrote.

  Native EXL3-G fails if any statistic exceeds its bound: 2 (averaged) or
  8 (extreme) times the median of the legitimate arms' distinct values.
  There are 15 legitimate arms:
  - frozen EXL3-G and its four retunings;
  - GEMV on;
  - cuBLAS 13.8.0.4;
  - five variants using upstream's own alternative norm, RoPE and attention
    code, and all of them together with cuBLAS 13.8.0.4;
  - the native-like `ggml_ops` arm, with and without cuBLAS 13.8.0.4.

  The bounds are listed per statistic in
  [`tierc.json`](experiments/backend-proof-p0/tierc.json).
- **The run.** Native EXL3-G runs the trajectories with the report's
  captures. Its logits must be finite. A captured run must reproduce the
  uninstrumented logits exactly (RE-010). The first layer whose statistic
  exceeds its bound is the finding. Top-1 agreement with the oracle is
  reported, not gated.
- **Calibration** (from `tierc.json`):
  - Leave-one-out, the largest ratio any legitimate arm reaches is 1.25
    (averaged) and 2.14 (extreme) at 4.0 bpw, and 1.12 and 1.51 at
    4.5 bpw. The native-like arms reach 1.07–1.15 and 1.34–1.78.
  - Five gross faults fail, and a more accurate native plan passes easily:
    - a wrong norm epsilon;
    - Q alone shifted by one RoPE position;
    - Q and K shifted together, which RoPE's relative form nearly cancels,
      so only the absolute K statistic catches it;
    - the same shift in single-token steps only;
    - one corrupted key position.
- **What this bound cannot see.** Two subtle faults pass on both fixtures,
  as a legitimate implementation change would: a softmax scale 1% high in
  one layer, and one layer's MLP output rounded to BF16. The
  operation-level exactness gate and its dtype-chain check catch them
  (Tier E).
- *Applied 2026-09-27 (`spark-b`):* native passes on both fixtures in
  EXL3-G and in EXL3-O: none of the 750 statistics exceeds its bound
  (largest ratios to the legitimate median 1.145 averaged and 1.781
  extreme at 4.0 bpw, 1.099 and 1.351 at 4.5 bpw), every logit finite, the
  captured run's logits equal the uninstrumented run's; top-1 agreement
  with the oracle 2,423 (4.0 bpw) and 2,425 (4.5 bpw) of 2,448 rows
  ([P3 report](experiments/backend-proof-p3/README.md#tier-e-operation-level-and-tier-c)).

### Memory and workspace (the M2 gate in exl3-bringup.md)

**FP16: the limits are pre-registered under D-079 (2026-09-27)**, before
any native FP16 run. They follow the envelope rules below. Since D-085,
the memory check at the end of this section judges a run, and these
limits are the plan's declared budget. Evidence: the report's
[FP16 executed plan and workspace](experiments/backend-proof-p0/README.md#fp16-executed-plan-and-workspace)
and `fp16-plan.json`.

- **The bridge gives no per-phase activation peak.** Its compute buffer is
  one allocation, sized for the largest batch and held through every
  phase. The two buffers are 37.31 MiB (`control`, 64 rows) and 298.50 MiB
  (`heldout`, 512 rows).
  - Each equals the batch times 611,328 bytes, to the byte. Per row, that
    is the output head's F32 input (896 × 4) and its F32 logits
    (151,936 × 4), which are live together at the head. That is the plan's
    largest simultaneous need.
  - So each limit is the plan's own need at the phase's row count,
    itemized from the record. None exceeds the bridge's buffer.
- **The items of `E`,** in bytes, for a chunk of n rows:
  - *A, activations and intermediates:* n × 611,328.
    - One layer's tensors counted with no reuse, plus the chunk's device
      inputs, take at most n × 211,984 bytes (at n_kv 768), about 35% of
      A. That count includes attention's scores and softmax,
      2 × 56 × n_kv × n.
    - So A holds when the head's output reuses the layers' memory, as the
      bridge's allocator does. A is strict: native must reuse memory as
      that allocator does.
  - *S, the GGML pool scratch:* P0's recorded peak for the chunk. It is
    the output head's F16 copies on the cuBLAS path.
  - *I, the inputs built on the host:* the chunk's recorded input copies.
    They carry the embeddings looked up on the host, the positions, the K
    and V slots, the mask and the output ids.
  - *L, the logits delivered to the host:* n × 607,744, the chunk's
    recorded copy.

  | Phase kind (trajectory, n_kv) | A | S | I | L | Limit `E` (bytes) |
  | --- | ---: | ---: | ---: | ---: | ---: |
  | 32-row prefill (`control`, 256) | 19,562,496 | 9,781,248 | 180,736 | 19,447,808 | 48,972,288 |
  | single-token step (`control`, 256) | 611,328 | 0 | 5,648 | 607,744 | 1,224,720 |
  | 16-row prefill (`heldout`, 256) | 9,781,248 | 0 | 90,368 | 9,723,904 | 19,595,520 |
  | 17-row prefill (`heldout`, 256) | 10,392,576 | 5,196,288 | 96,016 | 10,331,648 | 26,016,528 |
  | single-token step (`heldout`, 256) | 611,328 | 0 | 5,648 | 607,744 | 1,224,720 |
  | 512-row prefill (`heldout`, 768) | 312,999,936 | 156,499,968 | 3,940,352 | 311,164,928 | 784,605,184 |
  | single-token step (`heldout`, 768) | 611,328 | 0 | 7,696 | 607,744 | 1,226,768 |

  - FP16-F and FP16-U have the same limits. Fusion changes neither the
    head step, nor the pool draws, nor the copies.
  - A phase kind not listed here is derived the same way and written here
    before native runs it.
  - `E` is judged on the bytes the phase's plan places. Rounding a region
    up to the 2 MiB granule is reported separately, as weight padding is.
- **Outside `E`:**
  - *KV* is the declared layout: 24 layers, each cell holding F16 K and V
    of 128 elements. That is 6,291,456 bytes for `control`'s 512 cells and
    12,582,912 for `heldout`'s 1,024.
  - *Weights* equal the artifact's bytes, with padding reported
    separately. For comparison, the bridge holds 942.43 MiB on the device,
    and on the host the 259.66 MiB token embedding it looks rows up in.
  - *Persistent library workspace* is at most 33,554,432 bytes: the cuBLAS
    workspace the record's handle sets. GGML's pool is phase scratch
    (`S`), not workspace.
  - *`F`* is reported, as for EXL3.

**EXL3: the persistent-workspace limit is approved (2026-09-26); the phase
limits are deferred to P3 entry.** The per-phase measurements are in the
report's
[per-phase memory](experiments/backend-proof-p0/README.md#per-phase-memory).
They are identical under cuBLAS 13.1.1 and 13.8.0.4.

- **Envelopes.**
  - Every native plan declares, per phase kind and profile, its envelope
    `E` before the run. `E` covers everything the phase allocates or holds
    as scratch: activations and intermediates, operation scratch (the
    attention pool), transient reconstruction and the logits output.
  - Its observed peak, from llmpalooza's catalog, must stay within `E`. Since
    D-085, the process as a whole is judged by the memory check below,
    not by a driver and library census.
  - Weights must equal the artifact's bytes. Padding is reported
    separately.
  - KV must equal the declared layout.
  - Persistent library workspaces are limited separately (below).
  - D-081's landing zone is a separately declared persistent pool, like
    the library workspaces; it is not in `E` or `F`.
  - `F` holds only handles and module state. It is reported, never used to
    hide a workspace.
- **EXL3 phase limits.** Native's `E` per phase may not exceed upstream's
  PyTorch peak allocation above the phase's start (FP16 logits,
  activations and reconstruction included) plus an allowance for the
  declared plan's own needs. The allowance has three parts, all taken from
  the plan and its measurements, not chosen:
  - *The vector attention kernel's pool scratch,* measured in the operation
    study. Upstream's Triton prefill needs none.
  - *The plan's F32 intermediates of one layer,* counted with no buffer
    reuse: `attn_norm.f32`, `rope_q.in`, `q_rope`, `rope_k.in`,
    `k_rope.f32`, `attn.f32`, `mlp_norm.f32` and `swiglu.f32` in
    `exl3-op-plan.json`. That is `n × 38,400` bytes. Upstream keeps these
    tensors in F16 or never materializes them.
  - *The attention mask,* F16 `[n, Npad]` in the record: `n × Npad × 2`
    bytes. Upstream's causal attention builds none.

  | Phase | Upstream peak | Attention scratch | F32 intermediates | Mask | Limit (bytes) |
  | --- | ---: | ---: | ---: | ---: | ---: |
  | prefill, 32 rows (Npad 256) | 9,953,792 | 354,816 | 1,228,800 | 16,384 | 11,553,792 |
  | prefill, 144 rows (Npad 256) | 44,790,272 | 1,596,672 | 5,529,600 | 73,728 | 51,990,272 |
  | prefill, 145 rows (Npad 256) | 103,822,336 | 1,607,760 | 5,568,000 | 74,240 | 111,072,336 |
  | prefill, 1,023 rows (Npad 1,024) | 376,915,456 | 11,343,024 | 39,283,200 | 2,095,104 | 429,636,784 |
  | prefill, 1,024 rows (Npad 1,024) | 375,390,720 | 11,356,160 | 39,321,600 | 2,097,152 | 428,165,632 |
  | single-token step (Npad up to 1,280) | 310,272 | 48,048 | 38,400 | 2,560 | 399,280 |

  - The allowance is 7–16% over upstream's peak in prefill and 29% in a
    single-token step. It is the approval this limit asks for: the
    declared plan's F32 chain and vector attention cost that much memory
    in exchange for their accuracy.
  - Native EXL3-G's logits are F16, as upstream's are.
  - Chunks between measured sizes use the next larger measured phase, up
    to 1,024 rows. Chunks above 1,024 rows are out of scope.
  - Upstream's peak is its caching allocator's allocated bytes. It does
    not include reserved-but-unused cache, which native has no counterpart
    for.
- **Persistent library workspaces.** Native's total may not exceed
  upstream's device context: ExLlamaV3's 16 MiB workspace and 4,202,760
  lock bytes, 20,979,976 bytes in all.
  - Native's pinned cuBLASLt GEMMs need no workspace.
  - PyTorch's 32 MiB cuBLAS workspace is not counted. ExLlamaV3 replaces
    it with its own 16 MiB before every call, so the recorded plan never
    uses it.
- **No permanent FP16 shadow** (BP-A3).

**The EXL3 phase limits, tightened against native's itemized buffer plan:
pre-registered under D-079 (2026-09-27), from the plan, before any native
EXL3 model run.** The approved limits above add one layer's F32
intermediates to a peak upstream reaches with its logits, when those
intermediates are already dead. Native's plan
(`model/qwen2_exl3.h` PlanPhase) places every tensor a phase computes, each
linear's scratch and the F16 logits in one region by lifetime (the same
slots in every layer), so its own need is known without running it:
- *Region:* the placement's extent, 256-byte aligned slots. The logits
  share bytes with layer tensors that are dead by the head; the residual
  stream, positions and mask live through every layer. Pinned by
  `unit.Qwen2Exl3Test.RegionsAreThePreRegisteredBufferPlan`; the same in
  EXL3-G and EXL3-O, since the GEMV takes the GEMM's transformed-input
  scratch.
- *Pool:* the GGML pool scratch of the forced vector attention, the only
  pool draw of the plan: from 1,024 rows the mask pre-pass's KV_max, then
  the partial results and their metadata, each block from a 256-byte
  boundary as llmpalooza's pool hands it out (`ops.h` PlanFlashAttnVec, whose
  parallel blocks equal the record's in six of the eight kinds in
  `unit.GgmlExl3OpsTest.VectorAttentionMatchesTheRecordAndAnFp64Reference`,
  and in all eight in the plan gate's attention grids).
- *Limit `E`* is region plus pool: the tightened limit of each phase's declared
  budget. Every one is below the approved limit.

  | Phase kind | Region | Pool | Limit `E` (bytes) | Approved limit |
  | --- | ---: | ---: | ---: | ---: |
  | prefill, 32 rows (Npad 256) | 9,838,592 | 354,816 | 10,193,408 | 11,553,792 |
  | prefill, 144 rows (Npad 256) | 44,273,664 | 1,596,672 | 45,870,336 | 51,990,272 |
  | prefill, 145 rows (Npad 256) | 103,301,376 | 1,607,936 | 104,909,312 | 111,072,336 |
  | prefill, 1,023 rows (Npad 1,024) | 373,247,744 | 11,343,104 | 384,590,848 | 429,636,784 |
  | prefill, 1,024 rows (Npad 1,024) | 371,720,192 | 11,356,160 | 383,076,352 | 428,165,632 |
  | single-token step, Npad 256 | 307,456 | 14,848 | 322,304 | 399,280 |
  | single-token step, Npad 1,024 (the P3 addendum's kind) | 307,456 | 48,128 | 355,584 | 399,280 |
  | single-token step, Npad 1,280 | 307,456 | 48,128 | 355,584 | 399,280 |

- *Outside `E`,* each exactly its bytes and reported:
  - the host inputs' pinned staging (ids, positions and the F16 mask, each
    from a 256-byte boundary) and the logits' host copy (rows × 151,936 ×
    2), which upstream's peak does not count either;
  - the persistent library workspace: ExLlamaV3's lock area, 4,202,760
    bytes, within the approved 20,979,976; the pinned cuBLASLt GEMMs use
    none;
  - KV in the declared layout: 24 layers × K and V × 4,096 cells × 256
    bytes, 50,331,648;
  - weights equal to the artifact's bytes, padding reported separately,
    and two derived items the plan declares: the norms widened exactly to
    F32 (the record's `attn_norm.w`, `mlp_norm.w`, `final_norm.w`: 49 ×
    3,584 = 175,616 bytes) and each layer's multi-GEMM tables (24 × 48 =
    1,152 bytes), rewritten when the weights move;
  - the GGML pool is allocated once at the run's largest phase: what a
    smaller phase does not draw is pool-held occupancy.

**The memory check (D-085, 2026-09-27).** Census rules v1–v3 and nsys
passes stopped under D-085. Memory is judged loosely: a native run's peak
may be at most about 10% above the reference's (≤ ~1.1×), by ordinary
counters. There is no calibration, holdout or pre-registration. This
check replaces the census for FP16 and for EXL3, and the itemized
limits above remain the plan's declared budget. History: version 1
(pre-registered, counters only) failed all four native FP16 arms on
sub-resolution and reversing charges. Neither later version was
registered (details are in Git history).

- **Method.** Each engine runs once per workload. Peak is the fall in
  `MemAvailable` (the GB10's free memory, which includes the driver's),
  sampled every 20 ms, below its median over the second before the start.
  The runner is [`peak_memory.sh`](experiments/backend-proof-p2/peak_memory.sh).
  Both FP16 engines run their census harnesses with a 50 ms settle:
  the bridge is `fp16_census` and native is `llmp_fp16_exec --census`
  (rung 3, `cudaMalloc`). They make the same 64 MiB controls, so both
  peaks include one pinned 64 MiB probe.
- **FP16, rung 3** (`spark`, idle, 2026-09-27; kernel 7.0.0-1019-nvidia,
  driver 580.178.04, cuBLAS 13.8.0.4 for both; native binary `35c99c03…`,
  bridge `33bbf331…`; every native run reproduced its arm's logits hash):

  | Arm | Native peak (MiB) | Bridge peak (MiB) | Ratio |
  | --- | ---: | ---: | ---: |
  | FP16-F `control` | 1,886 | 2,321 | 0.81 |
  | FP16-U `control` | 2,019 | 2,108 | 0.96 |
  | FP16-F `heldout` | 3,716 | 3,973 | 0.94 |
  | FP16-U `heldout` | 3,473 | 3,654 | 0.95 |

  All four pass. A first pass on `spark-b`, shared with other agents' GPU
  tests, is not used: its baseline fell by 4.5 GB during the batch. Its
  `control` ratios were 0.99 and 1.01, and FP16-F `heldout` read 1.66.
- **EXL3, rung 3, against EXL3-O** (`spark`, idle, 2026-09-27; the
  held-out trajectory, prefixes 32, 144, 145, 1,023 and 1,024 with 16
  single-token steps each). Native is `llmp_exl3_exec --arm O
  --evaluations 1` (binary `6a5574fb…`, P3's, cuBLAS 13.8.0.4, the
  `plan-NN-O` plans); its logits equal P3's rung 3 bit for bit. The
  reference is `exl3_heldout.py` in the reference container
  (`exl3_run.sh`, frozen caches `tune-40`/`tune-45`,
  `--hgemm-f16acc 0`, GEMV on, as P0's `o` arm), which repeats each
  prefix three times and restores it; it passed its own checks, and its
  caches were unchanged.

  | Fixture | Native peak (MiB) | EXL3-O peak (MiB) | Ratio |
  | --- | ---: | ---: | ---: |
  | 4.0 bpw | 4,813 | 5,856 | 0.82 |
  | 4.5 bpw | 4,813 | 5,557 | 0.87 |

  Both pass. The two sides are not like for like. The reference's peak is
  a Python process in a container, with PyTorch's CUDA context and its
  caching allocator's reserve. Its own allocation peak above each phase's
  start (`--memory`) was at most 376,915,456 bytes (the 1,023-row
  prefill), a different quantity from the process peak. Native's peak is
  the same on both fixtures: its buffer plan, not the weights, sets it.
- **What the nsys passes found** stays recorded: cuBLAS's handle creation
  keeps a 64.1 MiB default workspace pool that `cublasSetWorkspace` does
  not free (RE-028). By the owner's decision (2026-09-27), the pool does
  not count against the 32 MiB persistent-workspace figure, and native
  may hold what the bridge holds.

### Performance protocol (rule approved 2026-09-26; BP-F2's reference pre-registered at P3 entry)

The owner approved the kernel rule below (the statistic, thresholds,
aggregate test and confirmation procedure) for BP-F2, the EXL3 kernels.
BP-F2's reference arm, pre-registered at P3 entry (the BP-F2 item
below), is EXL3-O:
- upstream's extension built with the SDK's NVCC 13.4.92, as native's port
  is (`exllamav3_ext.so` `aa8b9f16…`);
- cuBLAS 13.8.0.4.

The NVCC 13.0.88 build shares identical SASS with the 13.4.92 build for
only 16 of its 1,506 kernels. So the rule was recalibrated and its holdout
rerun on the 13.4.92 build (below).
BP-F1 uses the same rule with its own calibration, pre-registered below
(D-079). The model cases (BP-F3) and BP-F4 are reported, not gated.

This protocol extends the frozen rule of the
[EXL3 reference](experiments/exl3-reference/README.md#memory-and-acceptance-limits).
Its departures from M0 come from the report's
[timing controls](experiments/backend-proof-p0/README.md#timing-controls):
the sessions that timed upstream against itself (A/A) and against its GEMV
configuration. [`timing_protocol.py`](experiments/backend-proof-p0/timing_protocol.py)
at `c05fd2dd…`, fixed before its calibration and holdout sessions ran,
is the measured implementation of the kernel rule. The current script also
rejects incomplete or nonfinite sessions and treats a constant positive
aggregate shift as a failure; these checks leave the recorded calibration
and holdout outcomes unchanged. Every BP-F2 session runs with the SDK's cuBLAS
13.8.0.4, which the native plan pins, bind-mounted over PyTorch's in the
reference container.

- **Session.** A session is one uninterrupted sequence on `spark`:
  - no other GPU workload runs, and the clock policy and idle states stay
    unchanged;
  - SM clock and temperature are recorded at every block boundary;
  - each arm tunes (or warms up) in a discarded process, and one more
    discarded process runs immediately before the first timed block;
  - eight timed blocks follow, each a fresh process that runs every case,
    in the order A1 B1 B2 A2 B3 A3 A4 B4 for a primary session and B1 A1
    A2 B2 A3 B3 B4 A4 for a confirmation. A is the reference, B the
    candidate.

  `spark-b` sessions are stability evidence only.
  - *Why not two blocks per arm, as M0:* separate processes of one plan
    differed by more than one pair of reference processes captured. M0's
    rule failed 26 of 176 A/A cases.
  - *The extra warm-up process* keeps the first timed block from starting
    on an idle GPU. It does not remove the first-block effect. With it, the
    first block still ran 0.1–0.45% fast in some sessions (`c1`, `h1`). The
    primary order therefore leans slightly against the candidate, and the
    mirrored confirmation leans slightly towards it. The calibrated `σ`
    includes this effect.
- **Kernel cases (BP-F1, BP-F2).**
  - Five warm calls, then 31 samples per block.
  - Each sample replays one CUDA graph of ten invocations, bracketed by
    events, and records the interval divided by ten.
  - The graph is captured only in the benchmark, identically on both sides.
  - Every launched kernel is verified per case before timing. The native
    case's ordered launches must equal the recorded ones: kernel name,
    grid, block, shared memory (static plus dynamic, as the profiler
    reports it) and registers per thread. For BP-F2 the record is
    [`exl3-launch.json`](experiments/backend-proof-p0/exl3-launch.json),
    re-recorded on the NVCC 13.4.92 reference with each ExLlamaV3 kernel's
    SASS hash, which native's must equal. The tuner's decoded choice is
    forced to match. The bias add follows the native operation plan's
    owner, not upstream's PyTorch kernel.
  - A stream-launched arm (no graph) is reported beside it.
- **Plans.**
  - Upstream runs on frozen tuning caches, recorded in the reference's
    session. The 96 synthetic shapes use those of the report's A/A
    sessions.
  - Native forces the same choices, and the caches are verified unchanged
    after every block.
  - Separately tuned upstream plans differed by up to 12% in speed, so the
    frozen caches fix the performance reference as well as the numerics.
- **Model cases (BP-F3).**
  - 15 warm-ups, then 31 trials per block, in the same eight-block order.
  - Prefill at 32, 144, 145, 512, 1,023 and 1,024 tokens, for every fixture.
  - Decode of 64 tokens after a 512-token prefix, 31 requests per block.
  - Reported, not gated, at M2.
- **Statistic.**
  - Each case's process-to-process noise `σ` comes from calibration A/A
    sessions, upstream against itself on one frozen plan. It is the
    relative standard deviation of block medians within a session, pooled
    over the sessions.
  - For BP-F2, four sessions on the reference arm set it (`c5`–`c8`, two in
    each order), recorded in
    [`timing-calibration.json`](experiments/backend-proof-p0/timing-calibration.json).
    - The median `σ` is 0.62%.
    - The per-case thresholds (`z · σ · √½`) are below 2% for 96 of the
      176 cases and below 5% for 146.
    - The noisiest, synthetic 14,336 × 4,096 K4 at 1 row, reaches 18.5%.
  - An earlier calibration on the NVCC 13.0.88 build (`c1`–`c4`, median
    `σ` 0.69%) is kept in `timing.json`.
  - Native's ExLlamaV3 port is compiled as the reference is: NVCC 13.4.92,
    `-O3 --use_fast_math`, SASS for `sm_121` (not `sm_121a`), with the same
    sources. That is what makes the launch record's SASS hashes a
    requirement native can meet. The locked build meets it for the 14 GEMM
    kernels the record names (the source lock's `exllamav3` component,
    [M2 record](m2-record.md)). It also reproduces the device-code flags that
    PyTorch's extension builder adds: the four `__CUDA_NO_HALF*` macros,
    `--expt-relaxed-constexpr` and C++20.
  - Per case and session:
    `d = (candidate median / reference median − 1) / (σ · √½)`. Here each
    arm's median is taken over all 124 of its samples, and √½ reflects the
    four processes behind each arm.
- **Pass.**
  - A case fails a session when `d` exceeds `z = 3.86`. That is the
    one-sided normal quantile for a family-wise false-failure rate of 1%
    over the 176 cases.
  - Aggregate: cases are not independent, because a whole process can run
    fast or slow. Each block's session-wide shift is the median, over the
    cases, of the block's median relative to that case's mean. A one-sided
    two-sample t-test compares the candidate's four shifts with the
    reference's (6 degrees of freedom). The aggregate fails above 3.143,
    the 1% quantile. This catches a systematic regression spread thinly
    over many cases.
  - If any case fails, or the aggregate fails, a confirmation session in
    the mirrored order repeats every case. A case fails the stage only
    when it fails both sessions, and the aggregate must also pass in the
    confirmation. The outcome is mechanical, and every session is
    reported.
  - There is no floor, no averaging away of a case, and no dropped sample.
    A stage failure stands until its cause is fixed.
- **Validation.** Earlier designs are recorded in the report:
  - M0's two blocks per arm;
  - in-session allowances;
  - spread checks that single outlier processes tripped;
  - a case-independent aggregate test that a common process shift broke.

  The rule was fixed at `c05fd2dd…` before any cuBLAS 13.8.0.4 session
  ran. It was declared in advance that a holdout pair failing the stage
  would reject the rule, and any redesign would need a fresh holdout. Each
  holdout was timed A/A and not used for calibration:
  - *NVCC 13.0.88 build (`h5`, primary; `h6`, mirrored):* the stage passes.
    - In `h5`, the 4.0 bpw `up_proj` at 145 rows (d = 4.23) and the
      4.5 bpw `down_proj` at 16 rows (d = 3.98) exceeded `z`, which
      triggered the confirmation.
    - In `h6`, neither failed.
    - The aggregate `t` was 0.89 and −0.04.
  - *The reference arm, NVCC 13.4.92 (`h7`, primary; `h8`, mirrored):* the
    stage passes in `h7` alone. No case exceeded `z`, the aggregate `t` was
    0.77, and no confirmation was needed. `h8` also passes on its own: no
    case over `z`, aggregate `t` −2.31.
  - Every in-sample pairing of either calibration passes, needing at most
    a confirmation.
- **Power** on the reference arm, from `timing.json`:
  - *One slowed case,* slowed in both sessions of a pair (the holdout pair
    and three calibration pairs): detected 48–54% of the time at 2%,
    63–68% at 3%, 79–80% at 5% and 87–90% at 10%. Most misses are the
    noisier cases.
  - *A slowed subset* (`timing_power.py`), on the holdout pair:
    - the stage fails from 0.5% when all cases or the 66 reconstruction
      cases are slowed;
    - it fails from 1% for the 22 fused-reconstruction cases;
    - it did not fail up to 3% when only the noisiest quarter was slowed.
- **Harness equivalence (before BP-F2 runs).** The native benchmark harness
  times upstream's own kernel as its candidate: the extension binary of the
  reference arm, identified by its SHA-256. `measure.py` on the same binary
  is the reference, under this rule. The session must pass before any
  native kernel is timed.
- **BP-F1: llmpalooza's VMM against `cudaMalloc`.** Host VMM failed rule v1
  below; under D-081 BP-F1 was rerun against device VMM under rule v2,
  pre-registered below, and passed.
  - Compares the same GGML kernels, at the held-out trajectory's chunk
    shapes (1, 16, 17 and 512 rows).
  - Each sample rotates through weight buffers whose total exceeds four
    times the queried L2 size, so the kernels read from memory, not L2.
  - Both memory kinds run in separate processes, as blocks, under the
    kernel rule above.
  - Before the first comparison, four A/A calibration sessions on
    `cudaMalloc` set BP-F1's `σ`, and two holdout sessions validate the
    rule for it. BP-F1's `σ` is pre-registered from them (D-079) before
    BP-F1 is gated.
  - A regression reopens the memory decision (D-034, now D-081) for the
    owner; it does not block other stages.
  - **The frozen rule (pre-registered 2026-09-27, D-079),** fixed before
    any host-VMM timing of these kernels ran
    ([report](experiments/backend-proof-p1/README.md)):
    - *Cases:* the 53 in
      [`bpf1-cases.txt`](experiments/backend-proof-p1/bpf1-cases.txt)
      (BP-F1 cases SHA-256: `fe78d03360b12824e1fad2f0d9e6f152b867d8c502aa6d451671684302e6a20c`),
      derived by `bpf1_cases.py` from
      `fp16-plan.json`: RMSNorm, fused RMSNorm-multiply, multiply, the
      896- and 128-wide bias adds and the residual add, the q/o, k/v,
      gate/up, down and output-head projections, and attention's KQ and
      KQV, at 1, 16, 17 and 512 rows with the implementation and launches
      the plan recorded (MMVF, MMF, GGML's cuBLAS path). Each block
      verifies every captured launch against them before timing. BP-F1
      judges these cases only: a kernel llmpalooza adds later (RoPE, softmax,
      the KV writes) needs its own calibration and holdout first.
    - *Placement:* arm A (the reference) holds every buffer the kernels are
      given in `cudaMalloc` memory; arm B (the candidate) in host-backed
      VMM from llmpalooza's provider. That is weights, activations, outputs,
      GGML's scratch and the cuBLAS workspace. Each case rotates through a ring of
      identical operand sets totalling more than four times the queried
      L2.
    - *Harness:* `llmp_ggml_vmm_bench` with cuBLAS 13.8.0.4, run by
      `bpf1_session.py`; the comparison uses this binary, kept with the
      raw sessions. A rebuild elsewhere differs: the binary embeds its
      source paths.
      BP-F1 harness SHA-256: `05348df868e83768a441302bc2831df8ebbe4fa609a13c5b5cd874d9048f3b92`
    - *Calibration:* `c1`–`c4` on `spark`, two in each order; median `σ`
      1.23% (0.10–2.75%). BP-F1 calibration SHA-256: `aa1581271357e8c1bfeed2b8da98ae98ee35031b4160df6500cb8b6e991b1369`
      ([`bpf1-calibration.json`](experiments/backend-proof-p1/bpf1-calibration.json)).
      The session driver refuses a host-VMM arm unless its calibration
      file, harness and case file have these hashes; every manifest
      records them, and `bpf1_stats.py` checks them again.
    - *Thresholds:* `z = 3.555`, the one-sided 1% family-wise quantile over
      53 cases; the aggregate limit stays 3.143. Per-case thresholds
      `z · σ · √½` are 0.25–6.9%, below 2% for 13 cases.
    - *Holdout,* declared in advance to reject the rule if either session,
      as the primary with the other as its confirmation, failed the stage:
      `h1` (primary) passes alone (aggregate `t` 1.87); `h2` (mirrored)
      needs `h1` as its confirmation for one case (fused RMSNorm-multiply
      at 16 rows, d = 3.73) and passes. The rule stands.
    - *Procedure:* a primary session in the order A1 B1 B2 A2 B3 A3 A4 B4;
      if a case or the aggregate fails, a mirrored confirmation. A case
      fails BP-F1 only when it fails both; the aggregate must pass in the
      confirmation too. The stream-launched arm is reported, not gated.
  - **Result (2026-09-27): BP-F1 fails; D-034 was reopened, and the owner
    moved weights and state to device VMM (D-081).**
    Run under the committed pre-registration (`d3b4f2a`) on `spark`: 41 of
    the 53 cases failed both the primary session `p1` and its mirrored
    confirmation `m1`, and the aggregate failed in both (`t` 54.16 and
    56.98). Every matrix product ran slower on host VMM, 1.10× to 4.9×
    (the output head at one row 2.14×); 12 elementwise and norm cases
    passed. Launches and outputs were identical in both memory kinds. This
    blocks no other stage
    ([comparison](experiments/backend-proof-p1/README.md#comparison-host-vmm-against-cudamalloc-bp-f1-gated)).
  - **BP-F1 rule v2 (device VMM, D-081),** pre-registered 2026-09-27
    (D-079) before any device-VMM session of these kernels ran on `spark`
    ([report](experiments/backend-proof-p1/README.md#rule-v2-device-vmm-d-081)).
    Rule v1 and its result stand as history. v2 changes the candidate's
    memory kind and the harness binary (now built natively), and so the
    calibration and holdout; the cases, statistic, `z` and procedure are
    v1's:
    - *Cases:* v1's 53, unchanged
      (BP-F1 v2 cases SHA-256: `fe78d03360b12824e1fad2f0d9e6f152b867d8c502aa6d451671684302e6a20c`).
    - *Placement:* arm A (the reference) holds every buffer the kernels
      are given in `cudaMalloc` memory; arm B (the candidate) in
      device-located VMM from llmpalooza's provider, where D-081 places
      weights and state: weights, activations, outputs, GGML's scratch and
      the cuBLAS workspace. The setup staging buffer (host VMM in both
      arms) is outside; no timed kernel touches it. Rings as in v1.
    - *Harness:* `llmp_ggml_vmm_bench` with a `device-vmm` memory kind,
      built by the `spark-native` preset on `spark-b` and copied, with
      cuBLAS 13.8.0.4, to `spark`; the comparison uses this binary, kept
      with the raw sessions.
      BP-F1 v2 harness SHA-256: `0c191da79557793ee779e2cac3de241072e83f052224d1f037c7adef596d5d8c`
    - *Calibration:* four A/A `cudaMalloc` sessions on `spark` with this
      binary, `c1`–`c4`, two in each order; median `σ` 1.17%
      (0.15–3.15%).
      BP-F1 v2 calibration SHA-256: `567cb8494dbb36022be6ba64fb185be272561bf7e7f89680c3413931f93bb3bc`
      ([`bpf1-v2-calibration.json`](experiments/backend-proof-p1/bpf1-v2-calibration.json)).
      The session driver refuses a device-VMM arm unless its calibration
      file, harness and case file have these v2 hashes (and a host-VMM
      arm unless they have v1's); every manifest records them, and
      `bpf1_stats.py` checks them again.
    - *Thresholds:* `z = 3.555` over the 53 cases; the aggregate limit
      stays 3.143. Per-case thresholds `z · σ · √½` are 0.37–7.9%,
      below 2% for 18 cases.
    - *Holdout,* declared in advance (in every v2 session's manifest) to
      reject the rule if either session, as the primary with the other as
      its confirmation, failed the stage: `h1` (primary) and `h2`
      (mirrored) each pass alone, no case over `z` (largest d 1.92 and
      0.94), aggregate `t` 0.03 and −0.59. The rule stands.
    - *Procedure:* v1's. A primary session A1 B1 B2 A2 B3 A3 A4 B4 with A
      `cudaMalloc` and B device VMM; if a case or the aggregate fails, a
      mirrored confirmation. A case fails BP-F1 only when it fails both;
      the aggregate must pass in the confirmation too. The stream-launched
      arm is reported, not gated. A failure reopens D-081.
  - **Result under rule v2 (2026-09-27): BP-F1 passes; D-081 stands.**
    Run under the committed pre-registration (`72c7c62`) on `spark`. The
    primary session `p1` had two cases over `z`: KQV at 17 rows (d =
    4.45) and the k/v projection at 17 rows (d = 3.65). Its aggregate
    passed (`t` 0.58). The mirrored confirmation `m1` had no case over
    `z` (those two at 2.2 and 2.0) and aggregate `t` −0.36, so no case
    fails the stage. Every case's ratio of arm medians was 0.957–1.037
    across both sessions, as in the six A/A sessions (0.957–1.039);
    the cases host VMM slowed most (the output head, single-row KQV)
    were at 0.999–1.000. Launches and outputs were identical in both memory kinds
    ([comparison](experiments/backend-proof-p1/README.md#comparison-device-vmm-against-cudamalloc-bp-f1-rule-v2-gated)).
- **BP-F2: EXL3 kernels.** *Not run: replaced on 2026-09-27 by D-085's
  end-to-end rule (each engine at least as fast as its reference once
  operational). The protocol below is kept as history.*
  - All 176 cases, against upstream EXL3-G with cuBLAS 13.8.0.4, the
    matched plan; EXL3-O since D-080, with the case set fixed at P3 entry
    (above).
  - Upstream's GEMV gap at 1 to 8 rows is measured in the report, as
    EXL3-O against EXL3-G. It applies only to the 4.0 bpw fixture: its
    GEMM-only kernels are 1.14–1.55× slower on q, k and down at 1 and 8
    rows, and 0.90–0.94× on the fused gate/up. The 4.5 bpw fixture
    launches the same kernels either way. The owner accepted the gap for
    M2 as an explicit tradeoff (D-079), to expire when the GEMV provenance
    gate closed. It closed on 2026-09-27 (D-080): BP-F2 is gated against
    EXL3-O, with the case set and reference arm fixed at P3 entry.
  - **BP-F2's reference arm and cases: pre-registered under D-079
    (2026-09-27), at P3 entry, before any native EXL3 kernel was timed.**
    No timing session has run under them; they run on an idle `spark`
    once this is committed. (The plan gate's nsys traces of native runs
    carry kernel timestamps; no tool or report reads their durations.)
    - *Reference arm:* EXL3-O: upstream's extension built with the SDK's
      NVCC 13.4.92 (`exllamav3_ext.so` `aa8b9f16…`), the SDK's cuBLAS
      13.8.0.4 bind-mounted as the only cuBLAS mapped,
      `EXL3_HGEMM_F16ACC=0`, and upstream's default GEMV (on). Each case
      is upstream's own call, so upstream chooses the GEMV, the GEMM or
      the reconstruction.
    - *Cases:* the 184 of
      [`bpf2-cases.txt`](experiments/backend-proof-p3/bpf2-cases.txt)
      (BP-F2 cases SHA-256:
      `8cb178044c6f55b6ad0bf65fd4ae148c13c1bd485d97ec754f9ced2086e2c777`),
      written by `bpf2_cases.py` from the M0 protocol: the 176 cases
      unchanged (`LinearEXL3.forward` under EXL3-O), and the fused gate/up
      multi-GEMM (`exl3_mgemm`, as the gated MLP calls it) of each
      fixture's layer 0 at 1, 8, 16 and 32 rows.
    - *Bias add:* in the q and k projection cases the timed work includes
      the bias through ExLlamaV3's `add_kernel_hhh` on every path, the
      native plan's owner: on the reconstruction path the reference
      harness calls upstream's `add` kernel where upstream's module calls
      PyTorch's, as the operation plan's probe did.
    - *One frozen tuning cache per case set,* governing the model plan
      and the timing cases alike. For each fixture it is P3 part 1's final
      EXL3-O cache (`results.json`, `4.0bpw EXL3-O` `69775060…` and
      `4.5bpw EXL3-O` `64cf8ecd…`): P0's frozen `tune-40` or `tune-45`,
      every record unchanged, with the 8-row bucket's records that
      upstream's tuner added in P3 part 1's discarded tuning pass. The
      8-row decision is those records, from reference data only; no
      trajectory of the model plan reaches 8 rows, and `model_plan.py`
      checks that every record the model plan uses is P0's. A key a BP-F2
      case needs and the cache lacks (the lone gate, up or down linear at
      a small bucket, the synthetic shapes' set) is tuned once by
      upstream's tuner in one discarded reference process on `spark` before
      the first calibration session and appended; no existing record may
      change (checked by byte comparison). From then on the cache is
      frozen: its SHA-256 goes into every session's manifest, and the
      sessions check it unchanged after every block.
    - *SASS:* the port's build reproduces the reference's SASS for every
      ExLlamaV3 function it holds (P3 part 1, above).
    - *Launches:* re-recorded for this case set on the reference arm
      (`exl3_launch_record.py`, EXL3-O settings, the frozen caches) before
      calibration; the native case's launches must equal them as the rule
      above requires.
    - *Calibration and holdout,* since the case set and arm changed: four
      A/A sessions of the reference arm on `spark`, two in each order,
      set every case's `σ` under the approved rule unchanged
      (`timing_protocol.py`; `z` for a 1% family-wise rate over the 184
      cases, 3.870; aggregate limit 3.143). Then a holdout pair, primary
      and mirrored, declared in advance to reject the rule if the pair
      fails the stage. Then the harness-equivalence session, which must
      pass before any native kernel is timed. Only then BP-F2's primary
      session, and its mirrored confirmation if a case or the aggregate
      fails.
- **BP-F4: host submission time.**
  - Reported per launch and per token, against both upstreams' decode:
    llama.cpp with CUDA graphs on and off, and ExLlamaV3's graph-captured
    decode.
  - Graph capture becomes an M3 prerequisite if native host time per token
    exceeds upstream's measured time per token minus native device time per
    token.

## Stages

| Stage | Needs | Exit evidence |
| --- | --- | --- |
| **P0** Bridges and controls | M1 build | Toolchain bridges run, FP16 with fusion on and off. Held-out trajectories run on both references. The reference EXL3 tuned shapes and grids are decoded. Numerical profiles, bounds and the performance protocol are frozen, owner-approved or pre-registered under D-079 |
| **P1** Substrate probes | M1; no artifacts | GGML launchers under a llmpalooza context (K-C): stream, handle and pool injection, runtime-context binding, patched destructor and device flag. Values and kernel times on llmpalooza's VMM versus `cudaMalloc` (device VMM since D-081). Allocation census. One native EXL3 linear byte-equal to upstream at a forced plan, including alignment probes. Two implementations of one operation selected by plan. Per-launch host cost |
| **P2** Resident FP16 | Question-5 encoding and importer; M2 catalog; the [P2 prerequisites](#p2-prerequisites) | Prepared-artifact execution on device VMM through the landing zone (D-081); oracle rungs 3–4; BP-A cases |
| **P3** Resident EXL3 | P2 infrastructure; the GEMV port for BP-F2 (gate closed, D-080) | Both fixtures; per-linear and full-model oracles; BP-F2 kernel parity |
| **P4** Paging | M2 leases and storage service | BP-P cases on both representations |
| **P5** Lifetime and failure | D-048 completion services | BP-L and BP-V cases on real providers |
| **P6** Envelopes and contract | P2–P5 | Complete accounting; phase envelopes and `F` per profile; contract draft; per-operation K-C/K-L choices, registry and D-052/D-053/D-081 status recorded |

The [retained-backing comparison](#retained-backing-comparison) runs on the
P4 harness, but its acceptance stays a separate plan item.

P4–P6's results, and where every case below stands, are in the
[aggregate report](experiments/backend-proof/README.md); P6's record is
D-086.

### P2 prerequisites

Three things are in place before any native FP16 run: the FP16 memory
limits, the memory check (D-085, which replaced the census rule; both are
under "Memory and workspace" above), and the plan comparator.

**The plan comparator** tells whether a native run executed the bridge's
recorded plan (the FP16 Tier E gate).

- **Recording.** A test or benchmark links `llmp_launch_recorder`
  (`tests/support/`). It is for tests and benchmarks only: the configure
  fails if a production binary links it.
  - At link time it wraps the CUDA runtime's launch, copy and memset
    entry points, the driver's copy and cuBLAS's GEMM entry points.
  - While a `Recording` is open, it notes each call the thread makes: each
    kernel's mangled name, grid, block and dynamic shared memory, with the
    runtime's registers and static and local memory; each copy's size;
    each cuBLAS call.
  - The harness marks the chunks and writes JSON lines (`plan_record.h`).
- **The run's other inputs,** captured as the bridge's were:
  - cuBLAS's and cuBLASLt's logs: parameters, math mode, heuristic
    preference, resolved algorithm and handle setup;
  - `cuobjdump -sass` of the binary, for the SASS hashes;
  - an nsys CUDA trace of the same run, for the kernels cuBLAS launches
    itself, which no in-process wrapper sees;
  - the hashes of the loaded cuBLAS libraries.
- **[`plan_compare.py`](experiments/backend-proof-p2/plan_compare.py)**
  converts the recording and compares it with the arm of `fp16-plan.json`,
  chunk by chunk and token by token:
  - chunk order, rows and positions;
  - each kernel's name (NVCC's per-file hashes masked), grid, block,
    shared memory, registers and SASS;
  - each copy's size, which is where a lookup moved off the host shows;
  - each cuBLAS call's parameters, math mode, cuBLASLt record and launched
    kernels;
  - the handle setup and the cuBLAS library hashes;
  - no kernel or cuBLAS call between chunks.
- **What it does not compare,** which the gate allows to differ: stream
  identity, addresses, the launch API (which carries the PDL attribute;
  the recorder names any other launch attribute, which is a difference)
  and the copies' memory kinds, which follow from the addresses. VMM
  backing of either location is device memory to CUDA (D-034, D-081).
  - Alignment is judged by its effects: `get_rows`' variant and cuBLASLt's
    alignment preferences.
  - The recording sees no addresses, so the harness must check that every
    tensor it binds is aligned to 128 bytes.
- **Its report.** It stops at the first difference and names the chunk,
  the token, and the token's place in the record's folded sequence.
  - Exit 0 means a complete match, the only result that lets the logits
    be compared.
  - Exit 2 means the recording matched but lacks one of the inputs above,
    or is a fragment of the run.

## Case matrix

Invariant numbers refer to architecture.md's pager invariants. Fake-backend
and CPU-only cases run on the workstation; everything else runs on `spark`.

**Backing and accounting**

- **BP-A1:** Every pointer a launch dereferences lies in a cataloged llmpalooza
  range or a cataloged backend allocation. Reconcile a complete run's CUDA
  allocation census with the catalog, distinguishing virtual reservations,
  physical backing and aliases. Use CUPTI callbacks or an equivalent whose
  coverage is verified with controls for direct driver allocations, runtime
  allocations and VMM backing. CUDA callbacks do not observe ordinary host
  `malloc`/`new`/`mmap`: instrument those separately, including allocator-held
  capacity, and validate that coverage with a host-allocation control. Charge
  opaque driver/library overhead conservatively and reconcile remaining
  physical-memory differences in BP-A5; a CUDA-only trace is not a complete
  Spark memory census (invariant 5, D-006/D-050). In M2, D-085 replaces
  this reconciliation with the memory check under "Memory and workspace":
  the catalog's in-process coverage check stands, and peak memory is
  compared with the reference's.
- **BP-A2:** Kernel scratch comes only from declared workspace. That
  covers GGML launchers' pool requests, cuBLAS workspace, EXL3 locks and
  workspace, and tuning allocations. Handles and unavoidable driver/library
  allocations are separately bounded under BP-A1. Each charge belongs in
  `F` or in the phase's peak working set `E`; no allocation escapes the
  envelope, including lazy library growth after warm-up.
- **BP-A3:** No hidden weight duplication: no F32 weight conversion, no CPU
  extra buffer types and no permanent FP16 shadow. Transient reconstruction
  is charged to the phase peak.
- **BP-A4:** After retirement and unmap, no backend object references
  llmpalooza backing. Submitting a stale graph or pointer table is rejected
  before launch. A negative control faults deterministically instead of
  reading stale memory (invariant 1).
- **BP-A5:** Node-level observations (`/proc/meminfo`, cgroup, CUDA memory
  queries) are reconciled with the catalog. Record what each one does and
  does not cover on unified memory.

**Numerics**

- **BP-N1:** The toolchain bridges versus the references; differences
  recorded.
- **BP-N2:** Native dispatch on conventional memory versus the bridge.
- **BP-N3:** Llmpalooza's VMM (device VMM, D-081) versus conventional memory.
- **BP-N4:** Native versus the reference within the declared bound, on
  held-out inputs.
- **BP-N5:** Per-linear EXL3 byte equality at a forced plan across the row
  sweep, together with a decode check of the reconstructed weights.
- **BP-N6:** Full-model EXL3 within the declared bound, with per-layer
  localization.
- **BP-N7:** CPU diagnostics on a copy of the device-VMM backing versus
  the reference's CPU path.

**Paging (weights and state)**

- **BP-P1:** Evict all weights and restore them with coalesced chunk-closure
  direct reads. Logits are bit-identical to resident. *Passes
  (2026-09-27): with coalescing on, all four FP16 arms and both EXL3
  fixtures restore every weight through coalesced, vectored reads
  bit-identically. A quick A/B measured coalescing slower than one read
  per chunk, so it is an option, off by default (D-085)
  ([aggregate report](experiments/backend-proof/README.md#bp-p1-coalesced-reads)).*
- **BP-P2:** Partial eviction of one layer, of the trellis only, of side
  vectors or biases only, of a shared small-tensor chunk, of a padded tail
  and of a tensor crossing a chunk boundary. An incomplete closure refuses
  launch (invariants 1–2).
- **BP-P3:** Views of the FP16 tied embedding and output compute
  identically whether storage is duplicated or shared. The EXL3 head is a
  different representation and is never deduplicated with the embedding.
- **BP-P4:** Evict and restore KV state after a prefill chunk and in
  mid-decode. The continuation is bit-identical. Test with poisoned live
  state (invariant 4).
- **BP-P5:** Relocate to different addresses or backing. Tensor descriptors,
  EXL3 pointer tables and any captured graphs are rebuilt or revalidated, and
  stale ones are rejected.
- **BP-P6:** Cold and warm restore times and bytes read are reported, not
  gated.

**Lifetime and cancellation**

- **BP-L1:** Cancel with GGML or EXL3 work in flight. Submission stops, and
  leases hold until the recorded completion (invariant 2).
- **BP-L2:** Cancel during a page-in, including the `io_uring` cancellation
  race. A late DMA cannot corrupt a reassigned extent (invariant 3).
- **BP-L3:** Cancel between reconstruction and the GEMM. The scratch
  allowance persists until retirement.
- **BP-L4:** D-048 event permutations with real providers: completion before
  acceptance, duplicate completion, and unknown outcome leading to
  quarantine (invariant 8).
- **BP-L5:** Shared EXL3 locks and workspace across plans or streams
  serialize or reject.
- **BP-L6:** Registered I/O buffers over host-VMM landing extents are unregistered
  before unmap or reassignment.

**Validation and failure**

- **BP-V1:** Importer negative cases, both FP16 and the EXL3 list in
  exl3-bringup.md: shapes, types, ranges, rate/codebook mismatches,
  truncation and hashes.
- **BP-V2:** Inject failures: VMM create or map failure, cuBLAS handle
  failure and workspace exhaustion. Each returns an error and gets a complete
  unwind with no leaked charge. No path aborts the process (invariant 7).
- **BP-V3:** A tight budget admits the largest reconstruction phase or
  rejects the plan as impossible.

**Performance** (under the frozen protocol, with the reference repeated
beside the candidate)

- **BP-F1:** GGML kernel times on llmpalooza's VMM versus `cudaMalloc` memory
  (the D-081 check; host VMM failed it, and device VMM passed it,
  2026-09-27).
- **BP-F2:** EXL3 kernel parity on the 176 declared cases (the D-052 M2
  gate).
- **BP-F3:** Resident full-model timings for all three fixtures, reported
  for M5's gates. *Moved to M5 by the owner (2026-09-27):* part of M5's
  end-to-end comparison of each engine with its reference (D-085).
- **BP-F4:** Dispatch overhead: host submission per launch and per token,
  against upstream's captured decode. *The per-token half moved to M3,
  measured on M3's models, by the owner (2026-09-27, D-087);* the
  per-launch half is reported.

**Coexistence and swapping**

- **BP-S1:** Two implementations of one operation in one build, selected by
  plan. Each passes its own oracle comparison and envelope.
- **BP-S2:** Changing an operation's implementation changes plan identity.
  A plan built for the old identity is rejected before launch. Retained state
  from the old plan is reused only with validated compatibility; otherwise it
  is recomputed.
- **BP-S3:** Models using different kernel sources are resident and run
  alternately in one process, with correct accounting of shared workspace
  (passed 2026-09-27: FP16 and EXL3 alternating on one node, each
  evicting the other's weights under one budget; the
  [aggregate report](experiments/backend-proof/README.md#bp-s3-fp16-and-exl3-in-one-process)).
- **BP-S4:** In the fake backend, a build profile lacking an eligible
  implementation reports the model as unsupported. Nothing is substituted.

D-050's M2 adversarial rows that need real allocation or retirement map to
BP-A1–A2, BP-P2–P3, BP-L1–L6, BP-V2 and BP-V3. The rest stay fake-backend
tests.

## Retained-backing comparison

This is the owner's D-035 follow-up. M0 deferred it to M2 because it needs
the M2 resource core and the P4 harness. It is not a BP case: the proof can
pass while it is still open. The internal contract is not settled, and M2
does not close, until D-033 is explicitly retained or amended. D-033 stays
the baseline unless the predeclared criteria below are met. The comparison
has two independent parts.

**(a) Physical backing.** Compare D-033's independent 2 MiB handles with
larger persistently mapped slabs, including 1 GiB, using software
suballocation and real executable tensor views. Both designs must keep
ordinary paging free of avoidable create/release cycles. Measure:

- warm reuse, plus remapping and registration costs where a design needs
  them;
- fragmentation;
- growing and shrinking the shared pool;
- concurrent compute;
- end-to-end restore latency.

Any design that changes addresses must pass BP-P5, BP-L2 and BP-L6, plus
checks for aliases and captured pointers. Both designs keep useful contents
within budget. Disk transfer size is independent of either design, and
D-056's layout serves both.

Evaluate these slab-hole policies:

- contiguous-run eviction;
- size classes;
- a hybrid;
- the owner's activity-sorted compaction. This is a completion-safe
  relocation that competes with decode for memory bandwidth
  ([artifact-format.md](artifact-format.md)).

**(b) Checkpoint-batch submission.** For batches that mix small and bulk
transfers, compare serial with bounded asynchronous submission. Measure the
effect of scheduling order and depth, the time to the last required
completion, and consumer stalls. The M0 I/O spike did not measure these
mixes. M6 extends this to simultaneous demand reads and state write-back,
with dependency safety and bounded queues.

**Inputs and open prerequisites.**

- The measured expert-closure sizes (about 1.8–10.9 MB) and the per-model
  memory padding of per-chunk handles (3.49–10.87%) come from D-056's
  [worked examples](artifact-format.md#worked-examples-measured-plans-of-real-files).
- **The cross-model swap trace** is
  [built](experiments/retained-backing/README.md). It has:
  - eight synthetic models with D-056's measured groups and closures;
  - 16 episodes shaped like the frozen A→B→A trace, routed by the
    paging-feasibility captures;
  - a 1 GiB shrink probe at every outward switch;
  - per budget, a fragmentation-free reference eviction/restore sequence.

  Like other replay inputs, it stays outside Git; its generator,
  parameters and identity are recorded.
- **The proof fixtures cannot exercise realistic fragmentation.** The FP16
  artifact's groups total 988,208,640 bytes, under one 1 GiB slab, the EXL3
  fixtures are smaller, and none has experts. Compare hole policies by
  replaying the trace on the fake backend. Measure costs on `spark` with the
  M2 memory manager and synthetic extents. Report end-to-end restore on the
  real fixtures. This uses synthetic extents only,
  not MoE execution, which runs resident in M3 and demand-paged in M7.
- **Expert compaction is partial at M2.** Relocation without VA remapping
  assumes pointer-table dispatch. For experts, M3 makes the initial
  dispatch choice per format, and the M7 GGML proof revisits it for
  compaction. M2 evaluates compaction for dense groups, and the expert
  case is completed in M7.

**Criteria** (pre-registered on 2026-09-26 under D-079, before any design
has run on the trace; frozen). D-033 stays unless a slab or hybrid design
meets all of them at every budget and on both seeds.

- **Trace** ([identity and retrieval](experiments/retained-backing/README.md#identity)):
  - generator `swap_trace.py` `e6ea1214…`, `params.json` `42c8a044…`,
    `library.json` `a64b453a…`;
  - primary seed 20260926 (manifest `44f9f2b4…`);
  - confirmation seed 926202601 (manifest `8aee64a5…`), never looked at
    while choosing.
- **Budgets.** Each budget is the largest multiple of 1 GiB not above the
  trace's unique stored bytes divided by 5/4, 3/2 and 2. For the primary
  that gives 64, 53 and 40 GiB. Held backing, handles in handoff
  included, never exceeds the budget less any outstanding shrink.
- **Designs.**
  - The baseline is D-033: one 2 MiB handle per chunk, no free pool, and
    handoff on reclaim.
  - The alternatives are persistently mapped slabs of 32 MiB, 256 MiB and
    1 GiB with software suballocation. Each slab size is tried with the
    four hole policies above, in that order, which makes 12 designs.
  - Suballocation places every group at a 4 KiB-aligned offset, because
    direct reads land at group starts (D-056). Resources keep their
    256-byte alignment inside the group, which covers the kernels' (at
    least 256 B, "Source findings"). Rung 5 covers relocation within a
    slab.
  - Every design evicts the reference's victims first. It evicts more, or
    relocates, only when it cannot otherwise place a restore or meet a
    shrink. Its resident set therefore stays within the reference's.
    Its extra victims are the least recently used unleased groups, in the
    reference's recency order, unless its hole policy chooses them.
    Relocation moves dense groups only at M2.
- **Replays.**
  - *Deterministic:* every design runs on the fake backend, primary seed
    first. Each replay runs twice, and the two must agree exactly, or the
    run is void.
  - *Timed:* sessions on `spark` with the M2 memory manager, for a design
    that meets every deterministic criterion at that budget.
    - Each block starts with an empty pool and replays one whole trace
      file back to back, without idle time; arrival seconds are
      informational.
    - Restores read a synthetic artifact file with O_DIRECT, coalesced
      as in D-056, into the design's backing. The file holds the trace's
      groups in id order at 4 KiB alignment, on the internal NVMe.
    - EXL3-G's decode runs alongside: the upstream reference arm, 4.0 bpw,
      BP-F3's 64-token decode.
- **Deterministic metrics** (exact). A tick is one lease, use or shrink.
  - *Waste:* held backing minus the stored bytes of resident groups after
    each tick, as a mean over ticks and a peak. For D-033 it is padding
    plus any handoff; for slabs it is holes, slab tails and padding.
  - *Useful content lost:* bytes restored beyond the reference's restores,
    and bytes evicted at shrink probes beyond the reference's victims.
  - *Refusals:* accesses the reference admits but the design cannot place,
    even after evicting every unleased group and relocating if its policy
    relocates. A replay stops at its first refusal.
  - Reported, not gated (their latency is in the timed metrics, through
    access wait):
    - delayed admissions, accesses whose restore waits for a relocation;
    - driver calls by kind per restored byte;
    - io_uring registrations and unregistrations;
    - bytes relocated.

    The timed runs' call counts must equal these.
- **Timed metrics,** per block:
  - *End-to-end restore latency:* for each access the reference restores
    for, from its issue until its groups are ready for their consumer.
    Every design is timed on the same accesses; its extra restores count
    as useful content lost. Reported at p50, p95 and p99, by nearest rank.
  - *Design time:* that latency minus the time during which at least one
    of the access's reads is in flight, from its submission to its
    completion. Backing work overlapped with a read in flight is not
    charged; work while none is in flight is, wherever it falls, and work
    that slows the transfer shows in end-to-end latency.
  - Each restore is classed by the trace, the same for every design, and
    reported separately:
    - *warm* if the reference evicts for it (the pool is full);
    - *cold* otherwise.
  - *Access wait:* the sum, over every access in the block (hits
    included), of the time from its issue until its groups are ready for
    their consumer. It charges waits for relocation that the restores'
    latency does not see.
  - *Shrink time:* from a shrink probe to the return of the last release
    that brings held backing within the reduced budget, at p95.
  - *Decode inflation:* the median time of decode tokens that overlap a
    restore or relocation, divided by the same block's quiet median before
    the replay.
  - Reported, not gated: driver time per restored byte.
- **The allowance** is run-to-run noise, under P0's session rules.
  - *Sessions:* one session per design, budget and seed.
    - The baseline is A and the candidate B.
    - A discarded baseline block runs first. Eight fresh-process blocks
      follow, in the primary order A1 B1 B2 A2 B3 A3 A4 B4, or mirrored
      for the confirmation seed.
    - Sessions run budgets in the order 5/4, 3/2, 2, and designs in the
      listed order. No comparison spans sessions.
  - *σ,* per timed metric, restore class, budget and seed, comes from four
    A/A calibration sessions: the baseline against itself, two in each
    order. It is the relative standard deviation of block values within a
    session, pooled over the sessions.
  - *Statistic:* `d = (candidate median / baseline median − 1) / (σ · √½)`,
    with each median over that arm's four blocks.
  - *Threshold:* `z = 3.30`, the one-sided normal quantile for a 1%
    family-wise false result over one design's 21 timed comparisons per
    seed (seven criteria at three budgets).
    - "Better beyond the allowance" means `d < −z`.
    - "No worse beyond the allowance" means `d ≤ z`.
  - *Holdout:* before any candidate is timed, two more A/A sessions per
    budget on the primary seed (one in each order, not used for
    calibration) must each pass their seven comparisons in both directions
    (`|d| ≤ z`).
    Otherwise the rule is rejected and redesigned, with a fresh holdout.
  - Deterministic metrics carry no allowance: "no worse" means `≤`, and a
    tie passes.
- **Amending D-033.** A design replaces the baseline only if, at every
  budget and on both seeds:
  - its warm design-time p95 is better beyond the allowance;
  - its cold design-time p95, and its warm and cold end-to-end p95, are
    no worse beyond the allowance;
  - its waste, mean and peak, is no more than the baseline's;
  - it loses no more useful content than the baseline;
  - it has no more refusals;
  - its access wait, shrink-time p95 and decode inflation are no worse
    beyond the allowance;
  - if it moves addresses, it passes BP-P5, BP-L2 and BP-L6 and the alias
    and captured-pointer checks.
- **Winner and confirmation.**
  - Among the designs that pass on the primary seed, the winner has the
    lowest geometric mean, over the budgets, of its warm design-time p95
    as a ratio to the baseline's. Ties go to the smaller slab, then to the
    listed policy order.
  - On the confirmation seed, the winner's deterministic replay runs, then
    its timed sessions, fresh, after that seed's own calibration (four A/A
    sessions per budget). If it fails anything there, D-033 stays. No other
    design is tried on that seed.
  - A design that passes at only some budgets is reported for the owner.
- **Part (b).**
  - *Mixes.* Each block is 256 batches, read with O_DIRECT from the
    synthetic artifact file into held, registered backing:
    - all small: 1,024 reads of 64 KiB per batch;
    - all bulk: 32 reads of 2 MiB;
    - small and bulk, one to one by count: 31 pairs of one 2 MiB and one
      64 KiB read;
    - D-056's closures: the primary trace's first 256 restores at the 3/2
      budget that serve a `use`, one batch each, one read per group.
  - *Depths.* Serial (depth 1) against depths 2, 4, 8, 16 and 32.
    Submission follows batch order; a small-first order is reported, not
    gated. The consumer takes reads in batch order.
  - *Metrics,* per block:
    - the median time to a batch's last completion;
    - the p95, over batches, of the consumer's total wait per batch (the
      stall).
  - *Allowance:* as in (a), with one session per mix and depth and serial
    submission as the baseline.
    - σ per mix and metric comes from four A/A sessions of serial
      submission.
    - `z = 3.48`, for 40 comparisons: two metrics, four mixes and five
      depths.
    - *Holdout:* before any depth is timed, two more serial A/A sessions
      per mix, one in each order, must each pass in both directions, as
      in (a).
  - Bounded asynchronous submission replaces serial if some depth, for
    every mix, has a time to last completion that is better beyond the
    allowance and a stall p95 that is no worse beyond it.
  - The chosen depth is the smallest passing depth whose time to last
    completion is within `z · σ · √½` of the best passing depth's for
    every mix (both as ratios to serial).

## Evidence and limits

Record aggregates in an `experiments/backend-proof/` report:

- identities and the per-operation implementation choices;
- the allocation census with each allocation's category;
- per-profile envelopes and `F`;
- oracle results per rung;
- timing statistics under the frozen rule.

Raw logits, traces and logs stay outside Git. The operation contract,
implementation registry and patch set become a decision entry when M2
closes.

Out of scope: MoE (M3) and routed closures (M7), sharding (M4), multimodal
components, tokenizer/sampling (M3), API (M3–M5), spill format and
retention (M6), other EXL3 variants, other GPUs, and CUDA graph capture beyond explicit
rejection or a completed relocation proof. Passing this proof establishes
the small dense fixtures only, not model support.
