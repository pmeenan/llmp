<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Backend proof P3: native EXL3 — 2026-09-27

P3 of the [backend proof](../../backend-proof.md#stages) in two parts:
[part 1](#part-1-the-linears), the native linears, and
[part 2](#part-2-the-native-model), both fixtures end to end under the
native operation plan.

## Part 1: the linears

This is the first part of P3:
llmpalooza's own launchers of ExLlamaV3's kernels, judged per linear by the
approved Tier E items for EXL3 packed linears (up to 144 rows) and
reconstruction-path linears (145 rows and more). It is BP-N5, the
per-linear sweep: every real projection of both EXL3 fixtures at rows 1,
8, 9, 16, 32, 33, 144, 145, 1,023 and 1,024, native against upstream at
the same forced plan and inputs, in both EXL3-G (GEMV off) and EXL3-O
(GEMV on, BP-F2's gated reference since D-080). The full-model run
(Tier C) and BP-F2's timing are part 2.

**Results in brief** (2026-09-27, `spark-b`; `cudaMalloc`, rung 3, and device VMM):

| Fixture | Arm | Cases exact | Stages exact | Weights and reconstructed weights | Placements |
| --- | --- | ---: | ---: | --- | --- |
| 4.0 bpw | EXL3-G | 1,570 of 1,570 | 5,398 | 169 of 169 linears | all four |
| 4.0 bpw | EXL3-O | 1,570 of 1,570 (240 through the GEMV) | 5,398 | 169 of 169 | all four |
| 4.5 bpw | EXL3-G | 1,570 of 1,570 | 5,398 | 169 of 169 | all four |
| 4.5 bpw | EXL3-O | 1,570 of 1,570 (28 through the GEMV) | 5,398 | 169 of 169 | all four |

Every buffer each step of each path writes (the transformed input, the
product, each reconstruction slice, the reconstruction GEMM's output, the
output transform, the bias) is bit-identical to upstream's, and so is the
final output: packed linears byte-equal to upstream's kernel output,
reconstructed F16 weights equal to upstream's reconstruction, and the
reconstruction path exact under the approved pinned cuBLASLt rule.
Native's composed implementations (the registry's) reproduce native's
steps in every case. The same holds with every operand at the smallest
alignment the host checks accept, and with every operand, weights
included, flush against an unmapped VMM granule on either side (the
over-read probe: **no ExLlamaV3 kernel the sweep launches reads or writes
outside its operands**, so v0's `readable_bytes` = `bytes` for these
fixtures' EXL3 resources stands).
Every kernel native launches, ExLlamaV3's and cuBLAS's, is the one upstream
launched, with the same grid and block ([below](#launches)), and the SASS
of every ExLlamaV3 kernel native launches equals the reference build's
([below](#sass)).

### What runs

**Kernels.** The source lock's `exllamav3` component now holds, besides
upstream's GEMM compilation units, the K = 4 GEMV kernel's header and the
three upstream sources that held the reconstruction, Hadamard and bias-add
kernels together with their ATen wrappers
([licensing.md](../../licensing.md#exllamav3-gemm-kernels-in-the-core-m2)):
- patch 0003 reduces `quant/reconstruct.cu`, `quant/hadamard.cu` and
  `add.cu` to their kernels, removing the ATen, c10 and PyTorch-API
  includes, the host wrappers and `reconstruct.cu`'s instance tables (and
  two no-op `register` specifiers). The kernels are unchanged;
- patch 0002 adds `llmp/llmp_exl3_kernels.cu`, llmpalooza's instance unit,
  built with upstream's flags beside the GEMM units: it includes those
  sources and the GEMV header and instantiates only what the linear
  launches (the eight K = 4 mcg GEMV instances of upstream's
  `exl3_gemv_select_kernel`, `reconstruct_kernel` and
  `reconstruct_had_kernel` at K = 4, 5, 6 and 8, three Hadamard variants,
  `add_kernel_hhh`, which `add.cu` defines with its other adds and two MoE
  bias adds, compiled but never launched), with host lookups of every
  kernel (`llmp_exl3_kernels.h`, no CUDA types).

**Launchers** (`src/kernels/exl3/`, K-L under D-053). They replace
upstream's host wrappers:
- `validate.h`, in every profile: each launch's operand and plan checks,
  run on the host before anything is queued: shapes, rates, alignment
  (the source-derived minimums), overlaps, the kernels' int bounds, and a
  cooperative grid no larger than the device holds at once;
- `upstream_gemv.h`: a recorded copy of upstream's GEMV choice
  (`exl3_gemv_cfg` and `exl3_gemv_try_launch`'s rules), MIT AND
  Apache-2.0;
- `launch.h`: the launch context. It launches on a provider stream
  (`DeviceExecution::Submission`), zeroes its lock area (a declared
  workspace of upstream's 4,202,760 bytes) on that stream before its first
  launch, refuses a lock area another live context holds, computes each
  cooperative kernel's co-resident limit from the occupancy calculator,
  launches cooperatively where `launch_contract.h` requires it, and returns
  a launch error as a fault (after which it refuses every launch). The
  multi-GEMM never passes an expert range, so its selection state is never
  touched;
- `recon_gemm.h`: the reconstruction GEMM on cuBLASLt with the algorithm
  rebuilt from its nine pinned attributes (`exl3-recon-pin.json`), checked
  by `cublasLtMatmulAlgoCheck` before any launch;
- `linear.h`: each path of the linear in upstream's order (packed through
  the GEMM or the GEMV, the fused gate/up multi-GEMM, the reconstruction
  path in slices of at most 32,768 columns, the fused reconstruction),
  with the bias on every path through `add_kernel_hhh` as the approved plan
  requires;
- `implementations.h`: five registry declarations (`exl3.linear.gemm`,
  `.gemv`, `.reconstruct`, `.reconstruct_fused`, `exl3.multi_linear.mgemm`)
  whose identities cover the prepared tree, a build-time digest of every
  file of the module, the SDK, target, architecture, build type and, for
  the reconstruction paths, cuBLASLt.

**The reference** ([`linear_reference.py`](linear_reference.py),
[`run_reference.sh`](run_reference.sh)) runs in P0's reference container
on upstream's extension built by the SDK's NVCC 13.4.92
(`exllamav3_ext.so` `aa8b9f16…`, BP-F2's reference arm) with the SDK's
cuBLAS 13.8.0.4 bind-mounted (the only cuBLASLt mapped, version 130800),
`EXL3_HGEMM_F16ACC=0`, and `EXL3_GEMV=0` for EXL3-G. It loads each
fixture through upstream's Model API and calls upstream's extension as
upstream's modules do, one step at a time (`exl3_gemm` as `BC_LinearEXL3`
calls it, `exl3_mgemm` as the gated MLP does, `reconstruct_hgemm`'s steps),
hashing every buffer a step writes. Each case also runs through upstream's
own module call, whose output the steps reproduce in every case, and runs
its steps a second time, which repeat bit for bit. A second process
(`repeat-*`) repeats every stage of every case.

**Inputs** ([`cases.py`](cases.py)): row-major F16 activations from a
counter-based generator both sides implement, each element (top 11 bits of
`splitmix64`) − 1,024 over 1,024, which F16 holds exactly; the seed is
SHA-256 of the fixture, the linear and the row count. Native's inputs hash
equal to the reference's in every case.

**Plans.** Upstream's tuner chooses each packed launch (tile shape, grid,
concurrency), keyed by the problem with the rows bucketed to the next power
of two up to 16. Each reference arm starts from P0's frozen model cache
(`tune-40-gemvoff`, `tune-45-gemvoff` for EXL3-G; `tune-40`, `tune-45` for
EXL3-O). They lack only the 8-row bucket, which no P0 trajectory reached:
a first, discarded pass let upstream's tuner add those records (6, 11, 2
and 10 of them; no frozen record changed), and the measured pass and its
repeat ran on the result unchanged (cache SHA-256 before and after equal).
[`native_plan.py`](native_plan.py) decodes each case's record from the
final cache and checks it against the launches the reference profiled
(kernel, grid, block): no case disagrees. Where EXL3-O launched the GEMV,
the plan is the configuration and grid it launched, and llmpalooza's copy of
upstream's choice (`upstream_gemv.h`, on the device's own occupancy) picks
exactly those, and the GEMM wherever upstream kept the GEMM. The
reconstruction paths take each slice's pinned algorithm; every
reconstruction GEMM of the sweep is one of the 21 recorded.

**Native** ([`benchmarks/exl3_linear_sweep.cc`](../../../benchmarks/exl3_linear_sweep.cc))
opens each v0 artifact with the native reader, reads each linear's
tensors, and runs every case through the launch context step by step,
hashing what the reference hashes, then again through the implementation
the registry binds for its path. [`compare.py`](compare.py) holds every
stage, every weight hash and the full reconstructed weights, rotated and
fused, to the reference's.

### Placements: alignment and over-read

The sweep ran four times per fixture and arm, with every operand placed:
- **malloc**: `cudaMalloc`, 256-byte aligned (rung 3);
- **minimal**: at exactly the smallest alignment the host checks accept,
  not more: the trellis, the transformed input and F32 outputs at 16
  bytes, side vectors, activations and F16 outputs at 8, biases at 2, the
  reconstruction GEMM's operands at the 16 its pinned algorithms assume;
- **flush-end** and **flush-start**: in device VMM (D-081), each operand
  (every weight tensor, input, scratch and output) ending exactly at the
  end of its mapping with an unmapped granule after it, or starting
  exactly at its start with one before it.

All four are exact in every case of every arm. An access past either end
of any operand would have faulted; none did. That settles the question
[artifact-format.md](../../artifact-format.md) left open: for the kernels
of these paths, at these shapes, rates and plans, EXL3 resources need no
over-read reservation (`readable_bytes` = `bytes`). The GEMV's prefetch
ring and every tile loop stop at their operands.

The sweep's plans launched the GEMV only in its narrow configuration at
one and eight rows, the GEMM at tile shapes 1 and 2 and the multi-GEMM at
2 and 3, each at its tuned grid. A GPU unit test
(`unit.Exl3LinearTest.NoPackedKernelReadsOutsideItsOperandsAtTheFixturesShapes`,
added in the challenge round) runs the rest on random weights at every
linear shape and rate of both fixtures (the eleven (k, n, K) of
`results.json`, F16 and F32 outputs), in `cudaMalloc` memory and flush
against an unmapped granule at either end: the GEMV in both
configurations at one to eight rows (140 launches in its row-guarded
mode at two to eight rows), the GEMM at every tile shape each shape takes
at 1, 3, 8 and 16 rows, the multi-GEMM at every tile shape of the
gate/up shape with concurrency 1 and 2, each at the co-resident grid and
at 7 blocks. No fixture shape takes tile shape 4 (its n must be a
multiple of 512, and none of 128, 896, 4,864 and 151,936 is), so it ran
on a synthetic 896 × 1,024. All 680 launches per placement ran without a
fault and gave the `cudaMalloc` run's bits (`spark-b`, 2026-09-27). The
verdict covers those kernels at these rates and shapes, not other
shapes, rates, codebooks or kernel variants.

### Launches

An nsys trace (`-t cuda`) of each sweep in `malloc` placement records
every kernel native launched. [`launches_compare.py`](launches_compare.py)
holds them, in order, to the launches the reference profiled for each
case (each case runs twice natively: its steps, then the registry's
implementation): the same kernel, grid and block, for all 6,994 launches
per arm (5,956 ExLlamaV3, 1,038 cuBLAS). The one declared difference is
the approved operation plan's: the reconstruction path's bias add is
ExLlamaV3's `add_kernel_hhh` (as `add_gr` launches it) where upstream uses
PyTorch's element-wise add. So native's cuBLAS kernel names and grids equal
the arm's, as the reconstruction-path Tier E item requires (13 distinct
cuBLAS kernels: nvjet and three CUTLASS kernels), and the GEMM, multi-GEMM
and GEMV grids are the decoded plans'.

### SASS

[`sass_compare.py`](sass_compare.py) hashes each function's SASS as
P0's `fp16_plan.py` does (instruction text with addresses stripped, and
the raw encodings), ending each function at the next or at the end of its
cubin, so no hash takes in another member's header (the defect of P0's
`sass_hashes.py`). With cuobjdump 13.0.85 on `spark-b`:
- all 27 ExLlamaV3 kernels the reference runs launched (the bias add; the
  GEMM at K = 4, 5, 6 and 8 in shapes 1 and 2 with both outputs as the
  plans use them; the multi-GEMM at K = 4 and 5; four GEMV instances; the
  three Hadamard variants; both reconstruction kernels at every rate) have
  text and encoding hashes identical to the reference extension's
  (`aa8b9f16…`) in native's sweep binary;
- so do all 93 ExLlamaV3 functions the binary holds (the 64 GEMM and
  multi-GEMM instances and the 29 of the instance unit), each against its
  namesake in the extension.

The port therefore runs the reference's own machine code, which is what
the bit-exact outputs above rely on, and what BP-F2's P3-entry item asks
("check that the port's build reproduces the reference's SASS").

### GPU unit tests

`unit.Exl3LinearTest.*` (label `gpu`, `spark-b`) runs every path on a
synthetic q_proj-shaped linear: bit-identical in `cudaMalloc` memory and
device VMM, with the GEMV within 5.7e-4 relative RMS of the GEMM and the
fused reconstruction within 9.9e-4 of the unfused one (reported, not
gated); the same over-read probe; a grid larger than the device holds
refused before launch; no two live contexts sharing lock slots; a fault
returned as a fault, after which the context refuses; and the registry
binding each implementation to its own calls only, a stale or foreign
declaration to none. Since the challenge round it also runs the over-read
probe at the fixtures' shapes ([above](#placements-alignment-and-over-read));
two contexts on two streams launching cooperative grids at the device's
co-resident limit back to back, 100 rounds of GEMM, multi-GEMM and GEMV
each, which both complete (5.0 ms on one stream, 8.3 ms on two: the grids
partly overlap and neither starves the other); a reconstruction GEMM
refusing a pin recorded for another GEMM (another row count, output or
row stride); and each identity recording libstdc++'s assertions (D-083).
`unit.Exl3ValidateTest.*` covers the host checks and the GEMV choice in
every profile, the multi-GEMM's refusal of tables written for tensors
that have since moved (BP-P5) among them.

### Not covered by part 1

- The over-read verdict holds for the kernels and shapes swept and
  probed ([above](#placements-alignment-and-over-read)); the artifact
  format keeps it as a measured fact of these fixtures, not a rule for
  other rates, codebooks or kernels.

## Part 2: the native model

Both fixtures end to end under the approved native operation plan, in
EXL3-G and EXL3-O, on the held-out trajectories: for each prefix of 32,
144, 145, 1,023 and 1,024 IDs, every prefill row's logits, then 16
single-token steps (85 phases per fixture and arm).

**Results in brief** (2026-09-27, `spark-b`; each judged only after the
plan gate's exit 0):

| Fixture | Arm | Plan gate | Rung 3 repeat | Tier E, operation level | Tier C (750 statistics) | Rungs 4 and 5 |
| --- | --- | --- | --- | --- | --- | --- |
| 4.0 bpw | EXL3-G | exit 0, 85 phases of 8 kinds | identical | exact: 30,855 GGML operations, 80,166 wirings, 135,841 dtypes, 95,899 weight checks | pass; worst ratio 1.145 averaged, 1.781 extreme; top-1 2,423 of 2,448 | identical to rung 3, every evaluation |
| 4.0 bpw | EXL3-O | exit 0 | identical | — (the gate is EXL3-G's) | pass; 1.145, 1.781; 2,423 of 2,448 | identical |
| 4.5 bpw | EXL3-G | exit 0 | identical | exact: 30,855, 80,166, 135,841, 95,899 | pass; 1.063, 1.332; 2,425 of 2,448 | identical |
| 4.5 bpw | EXL3-O | exit 0 | identical | — | pass; 1.099, 1.351; 2,425 of 2,448 | identical |

Every instrumented run (the launch recording, Tier C's captures, Tier
E's operation recording) gave the uninstrumented run's logits bit for bit
(RE-010).

### What runs

- **The adapter and plan** (`src/model/qwen2_exl3.h`, every profile):
  `BindQwen2Exl3` binds an EXL3 artifact's resources (mcg trellises at
  K = 4, 5, 6 or 8, F16 side vectors, the mcg flag, F16 q/k/v biases, BF16
  norms and embedding), refusing anything missing, mis-shaped or unread.
  `PlanPhase` gives a phase its operations in the record's order, each
  with its owner, registry implementation, tensors (the record's names and
  dtypes) and, for a linear, the forced launch plan of the fixture's
  table: tile shape and grid from the frozen tuning cache, the GEMV's
  configuration and grid where EXL3-O takes it, each reconstruction
  slice's pinned cuBLASLt algorithm with the GEMM it was pinned for. It
  refuses a phase kind the record lacks (`RecordedPhase`) and a case the
  table lacks or that is not upstream's path. It places every tensor, the
  linears' scratch and the logits in one region by lifetime; the regions
  are the pre-registered buffer plan
  ([backend-proof.md](../../backend-proof.md#memory-and-workspace-the-m2-gate-in-exl3-bringupmd)).
  The launch digest covers every operation, tensor and launch plan and the
  tuning data's identity.
- **The executor** (`src/kernels/exl3/qwen2.h`): binds a phase plan through
  the registry (GGML's norms, casts, RoPE, attention, adds, SwiGLU and
  embedding; ExLlamaV3's linears, multi-GEMM and the new `exl3.bias_add`),
  runs every host check before anything is queued, checks in EXL3-O that
  llmpalooza's copy of upstream's GEMV choice picks exactly the table's plan,
  and runs the phase on one stream. The plan identity is the registry
  plan's identity (every implementation's) with the launch digest.
- **New GGML implementations** (`src/kernels/ggml/`): `ggml.convert`
  (`cpy_scalar_contiguous`, F32↔F16), `ggml.get_rows` over a BF16 table,
  and `ggml.flash_attn_ext.vec`, the forced vector attention
  (`ggml_cuda_flash_attn_ext_vec_case<64, F16, F16>` instantiated in
  `fattn.cu` with GGML's flags; `PlanFlashAttnVec` copies
  `launch_fattn`'s arithmetic, so its parallel blocks and pool scratch are
  known before launch). The SASS of all eleven GGML kernels of the plan,
  the four flash-attention ones included, equals the record's.
- **The harnesses** (`benchmarks/`): `llmp_exl3_exec` runs the
  trajectories from the artifact on `cudaMalloc` memory (rung 3), with
  `--record`, `--capture` or `--record-ops`; `llmp_exl3_paged` runs them
  paged into device VMM through the landing zone by the scheduler and its
  lanes (rungs 4 and 5), as `llmp_fp16_paged` does for FP16. The launch
  recorder (`tests/support/`) now wraps `cudaLaunchKernel`,
  `cudaLaunchCooperativeKernel` and `cublasLtMatmul`, and the recording
  marks each operation (`OpLine`).
- **Launch plans:** the four model plans (`model_plan.py`, below) use only
  P0's frozen caches' records: `plan-40-G` `b34fe49d…`, `plan-40-O`
  `9665fd29…`, `plan-45-G` `153e8662…`, `plan-45-O` `45a1fa5b…`.

### The plan gate

[`op_plan_compare.py`](op_plan_compare.py) holds native's recording and an
nsys trace of the same run to the record (`exl3-op-plan-g.json` for
EXL3-G, `exl3-op-plan-o.json` for EXL3-O): every operation in order, every
launch's kernel (GGML's by normalized mangled name, SASS encoding hash,
registers and static shared memory; ExLlamaV3's and cuBLAS's by demangled
name), grid, block and shared memory, the copies' sizes, and nothing else
queued. All four arms: exit 0, 85 phases of 8 kinds (per fixture and arm,
53,646 recorded operations and 57,087 launches). The paged harness's
first evaluation also matches, for 4.0 bpw EXL3-G and 4.5 bpw EXL3-O
(rung 4's executed plan is rung 3's). Its tests
(`tools/tests/test_op_plan_compare.py`) synthesize runs from the records
and catch each forbidden departure.

### Tier E, operation level, and Tier C

Judged by the reference-side tools below, which were written and then
reviewed by a separate agent (who fixed four defects: the RE-010 control
made mandatory in both, the fixture tied to the bounds, the shim pinned)
before any native numerical output was looked at. One more defect showed
on the first native 4.5 bpw run and was fixed before its verdict: the
harness took the fixture from `config.json`'s quantization bits, which say
4 for the mixed 4.5 bpw checkpoint, and refused the run (exit 2); it now
takes it from the checkpoint's SHA-256.

- **Tier E:** in EXL3-G, every GGML operation of every phase, layer and
  prefix (the embedding, the three norms, every cast, both RoPEs, the
  attention, both residual adds, SwiGLU) recomputed by the bridge's GGML
  library from native's recorded inputs is bit-identical to native's
  output; every input equals its producer's output, attention's K and V
  over `[0, Npad)` equal the layer's KV writes and zeros; every tensor has
  the record's dtype and shape; the host-built inputs equal the harness's
  own. Both fixtures, 85 phases each. Every weight an operation reads is
  recorded by the load-time hash of what lies at the address the program
  bound for it (`Qwen2Program::BoundWeights`): the embedding, norms and
  biases, every linear's trellis and side vectors, and what the two
  addresses in each of the multi-GEMM's three device tables point at.
  Each must be the artifact's tensor of that name, linear and layer, so
  an operation bound to another layer's or tensor's weights, or a stale
  table, fails. The weights were added to the recording in review and
  challenge (after the first verdicts; checks added, none relaxed), and
  both fixtures were recorded and judged again with the final binaries
  (`spark-b`, 2026-09-27, under `p3b-challenge-20260927`).
- **Tier C:** every one of the 750 statistics within its approved bound,
  all four fixture and arm pairs (table above). EXL3-O is judged against
  the same bounds, whose legitimate arms include GEMV on (D-079).

### Rungs 4 and 5

`llmp_exl3_paged --restores 2 --relocate`: every weight chunk (292
extents at 4.0 bpw, 302 at 4.5 bpw) paged into device VMM through the
zone. Evaluation 1 has rung 3's logits bit for bit; the repeat, and two
evaluations that evict every weight after each prefill (releasing its
backing) and page it back before the steps, the second at another
reservation with the multi-GEMM tables rewritten and every phase bound
anew, equal it. Every address a bound program reads or writes lay in
cataloged, resident device memory of its class (723,904 ranges, no
violation). Each full restore took 53–121 ms (reported, not gated).

### Not covered here

- BP-F2's timing (pre-registered; below). The EXL3 census is replaced by
  D-085's coarse memory check, which passes: native's peak (EXL3-O, rung
  3, `spark`, 2026-09-27) is 0.82× upstream's at 4.0 bpw (4,813 against
  5,856 MiB) and 0.87× at 4.5 bpw (4,813 against 5,557 MiB). Method and
  caveats are in
  [backend-proof.md](../../backend-proof.md#memory-and-workspace-the-m2-gate-in-exl3-bringupmd).
- Tier E's operation-level recomputation runs on EXL3-G, as the approved
  item says; EXL3-O's GGML operations are the same launches (the plan
  gate).
- Allocations inside a phase, which quiescent readings cannot see.
  The plan-gate runs' nsys API traces (all four arms, one evaluation) and
  the paged harness's (two evaluations) show no allocation call after the
  first launch (no `cudaMalloc`, `cuMemAlloc`, `cuMemCreate`,
  `cudaMallocAsync` or pool call), and GGML's pool is bounded per
  operation (only the attention draws). They do show 30 lazy module loads
  (`cuLibraryLoadData`, cuBLASLt's kernels) in the first evaluation's
  reconstruction-path prefills, and none in the second: warm-up growth,
  which the coarse peak check includes.

### BP-F2, prepared

Pre-registered in [backend-proof.md](../../backend-proof.md#performance-protocol-rule-approved-2026-09-26-bp-f2s-reference-pre-registered-at-p3-entry)
before any native EXL3 kernel was timed. [`bpf2_cases.py`](bpf2_cases.py)
writes [`bpf2-cases.txt`](bpf2-cases.txt) (184 cases, `8cb17804…`).
[`bpf2_measure.py`](bpf2_measure.py) is the reference arm's harness:
M0's `measure.py` with its kernel modes over the new cases (upstream's
`LinearEXL3.forward` under EXL3-O, ExLlamaV3's add kernel for the
reconstruction path's bias, `exl3_mgemm` for the fused gate/up), run by P0's
`timing_session.sh` in `measure.py`'s place (`MEASURE`, with each
process's case set as `BPF2_SET`). Not yet written: the native
candidate's timing harness and the session driver that times it beside
the reference.


### Part 2's reference side

Written and checked on `spark-b` on 2026-09-27, before any native
full-model output was seen. None of it shares code with native.

**Operation plan records.** [`op_plan_record.py`](op_plan_record.py) runs
P0's probe and build unchanged, with one more phase kind. The trajectory
"prefix 1,023, then 16 steps" takes its first step at position 1,023, with
N = 1,024 and K padded to 1,024. P0's record lacks that kind. The probes ran
as P0's did: the reference container, P0's `cuda134` shim, the container's
extension build (`7c9d383f…`) and cuBLAS 13.8.0.4. Every tuning cache was
unchanged by its probe, and the SASS inputs equal P0's (`424c3d89…`,
`73aa268c…`).
- [`exl3-op-plan-g.json`](exl3-op-plan-g.json) is EXL3-G with eight phase
  kinds. Its logits equal the `ggml_ops` arm's in all eight, on both
  fixtures. P0's seven kinds are identical to P0's record in every entry
  (kernels compared by identity) but one. P0's probe appended the next
  kind's prefill calls to a single-token kind's call list, so P0's
  `K_by_linear` for step 32 also lists gate and up. This record keeps each
  phase's own calls; no launch was affected. The new kind launches what
  step 1,024 does. The probes ran without `--bias-exhaustive`, so the
  record's `bias_add.exhaustive` is null: the all-pairs result its text
  cites is P0's record's. Its `dtype_chain_checks` cover each phase's own
  calls: 8,560, against P0's 9,768 over P0's call lists.
- [`exl3-op-plan-o.json`](exl3-op-plan-o.json) is EXL3-O: GEMV on, caches
  `tune-40` and `tune-45`. Its prefill logits equal the EXL3-G arm's; its
  single-token steps differ, as expected. It differs from the G record in
  48 entries, all single-token linear launches. `exl3_gemv_kernel` replaces
  `exl3_gemm_kernel` for q, k, v, o and down at 4.0 bpw, and for the K = 4
  down projections at 4.5 bpw. The GEMV runs 28 blocks (4 for k and v) with
  2,048 bytes of shared memory. Everything else is identical: GGML launches,
  gate/up, `lm_head`, prefills, dtypes and transfers.

**Operation-level Tier E.** [`op_tier_e.py`](op_tier_e.py) reads native's
`--record-ops` recording (format in its help). It checks:
- the operation sequence and names against the G record;
- the dtype chain;
- wiring, including attention's K/V over `[0, Npad)`;
- the weights each operation read, by the load-time hash of what lay at
  the address native bound: the embedding, norms and biases, every
  linear's trellis and side vectors, and what the multi-GEMM's tables
  point at, each against the artifact's tensor of that name, linear and
  layer;
- the host inputs, against its own construction;
- each GGML operation, recomputed by P0's shim and required to be
  bit-identical;
- RE-010: every phase's logits equal those of an uninstrumented run,
  whose manifest must say so (no capture, no operation recording, the
  same fixture, arm and artifact). The control is required.

P0's shim gained a cast (`ggml_shim_cpy`, `GGMLOps.cast`). Rebuilt against
P0's `cuda134` build tree as `ggmlops/cuda134b` (`libggml_shim.so`
`70904fc3…`), it loads P0's libraries (`libggml-cuda` `86ea9b4d…`), and
`op_tier_e.py` checks them in its process map. Two clean rebuilds of
`libggml-cuda` gave other bytes (`2893f943…`, `c2609e11…`), so the rebuild
reuses P0's build tree. `op_tier_e.py` pins the shim itself to that
build too. [`test_op_tier_e.py`](test_op_tier_e.py) runs it on
a synthetic 32-row trajectory built with the same kernels. The unaltered
recording passes. Fifteen alterations are each caught at the right place: a
flipped output bit, corrupt bytes, a mis-wired input, a wrong dtype, a
missing layer, a stale K cell, a wrong mask, wrong KV cells, a norm scale
and a linear's trellis bound to another layer's, a multi-GEMM table
pointing at another layer's weights, a recording without the linears'
weights, a changed uninstrumented logit, an instrumented control and a
missing control.

**Tier C.** [`pack_run.py`](pack_run.py) packs native's `.npy` output into
the reference runs' layout. A reference run unpacked and packed again is
identical, array for array. It also writes `native.json`: the capture
run's fixture and arm, and whether its logits equal the uninstrumented
run's byte for byte (RE-010). The fixture is the artifact's
([`fixture_identity.py`](fixture_identity.py): the checkpoint it was
prepared from, as `op_tier_e.py` reads it), never the run's command-line
label, which must agree. `tierc_check.py` requires that file and the
artifact, checks that fixture and the oracle's (by its recorded SHA-256)
against the bounds it loads, and fails a run whose captured
logits differ (`--reference` judges P0's arms, which have none). [`tierc_check.py`](tierc_check.py) applies
`tierc.json`'s bounds with P0's `oracle_compare.score` and `tierc.flatten`:

| Arm (both fixtures) | Verdict | Largest ratio to the median, averaged / extreme |
| --- | --- | --- |
| `g`, `o`, `cublas138b`, `ggml_ops_cublas138b` | pass | ≤ 1.17 / ≤ 1.79 |
| `f_norm_eps` | fail, first layer 0 | 25.5 / 57.6 (4.0 bpw), 23.3 / 59.0 (4.5 bpw) |
| `f_q_rope_offset_l10`, `f_rope_offset_l10`, `f_one_key_l10` | fail, first layer 10 | as `tierc.json` |
| `f_decode_rope_l12` | fail, first layer 12 | as `tierc.json` |
| `f_softmax_scale_l8`, `f_bf16_mlp_l16` | pass (the subtle faults Tier E catches) | ≤ 1.41 / ≤ 2.51 |

A run with one capture removed is incomplete (exit 2). One with a NaN logit
fails.

**Model plans.** [`model_plan.py`](model_plan.py) restricts the sweep's
plans (`native_plan.py`) to the trajectories' row counts. It checks three
things:
- every tuning record used equals the arm's frozen P0 cache, so no 8-row
  record is used;
- every case is present, on upstream's path;
- every packed grid equals the operation plan record's.

The four plans (966 cases each) are on `spark-b` under
`p3b-20260927/plans/`: `plan-40-G` `b34fe49d…`, `plan-40-O` `9665fd29…`,
`plan-45-G` `153e8662…` and `plan-45-O` `45a1fa5b…`.

## Reproduction

On `spark-b`, with P0's reference assets (`exl3-reference-20260922`,
`p0-20260925/build-cache-nvcc134` and `cuda-bridge-torch`) and the
reference image copied from `spark`:
1. `native_plan.py caches DIR/frozen` writes P0's four frozen caches.
2. Per fixture and arm, copy the frozen cache and run `run_reference.sh`
   three times: a tuning pass (`--no-weights`, discarded), the measured
   pass (`--profile`) and its repeat.
3. `native_plan.py plan ref-F-A.json --out plan-F-A.txt`. Since the
   challenge round each reconstruction slice's plan names the GEMM its
   pin was recorded for (kind, m, k, n, ldc), which the native side
   requires; the recorded runs used the earlier format, and a re-run of
   all four arms with the new plans, in `malloc` and `flush-end`
   placement, was again exact in every case (`spark-b`, 2026-09-27).
4. `llmp_exl3_linear_sweep --artifact ART --fixture F --plan plan-F-A.txt
   --out native.jsonl --placement P` for each placement, then
   `compare.py ref-F-A.json native.jsonl --repeat repeat-F-A.json`.
5. `nsys profile -t cuda` of one sweep per arm, `nsys export --type
   sqlite`, then `launches_compare.py ref-F-A.json trace.sqlite`.
6. `sass_compare.py --reference exllamav3_ext.so --port
   llmp_exl3_linear_sweep --launches ref-*.json`, and `record.py DIR
   --out results.json`.

[`results.json`](results.json) holds per fixture every linear's weight and
reconstructed-weight SHA-256s; per arm its environment, libraries, final
tuning cache (bytes and SHA-256), every case's plan and final-output
SHA-256, the comparisons and the launch verdict; and the SASS comparison. Raw runs (every
stage's hash, the launch profiles) stay on `spark-b` under
`~/.local/share/llmp/p3a-20260927`.

Part 2's reference side, on `spark-b` (each script's help has the
details; `run_container.sh` runs a script in the reference container,
with `GPUS=""` for the CPU-only ones):
1. P0's `ggml_shim/build.sh cuda134` on a copy of P0's `cuda134` build tree,
   then install its `libggml_shim.so` as `p0-20260925/ggmlops/cuda134b`.
2. `op_plan_record.py probe --arm G|O` per fixture, with P0's probe
   options, the `cuda134` shim and a copy of the arm's frozen cache. Then
   `cuobjdump -sass` and `-res-usage` of `libggml-cuda.so.0.24.0`
   (`fp16_plan.py sass-hash`), and `op_plan_record.py build`.
3. `test_op_tier_e.py`, with the artifact store mounted at `/artifacts`.
4. `tierc_check.py --reference` over the reference arms. For
   `pack_run.py`, unpack a reference run into native's layout with
   capture and control manifests, pack it with `--uninstrumented` and
   `--artifact`, and
   compare; then `tierc_check.py` on the packed run, on one whose control
   has a changed logit (fails), against the other fixture's bounds, and
   with a run or oracle whose fixture is not its artifact's
   (malformed).
5. `model_plan.py --record exl3-op-plan-g.json|-o.json` per fixture and arm.

Raw outputs stay on `spark-b` under `~/.local/share/llmp/p3b-20260927`.

Part 2's native side, on `spark-b` with the `spark-native` build and the
model plans: [`run_model.sh`](run_model.sh) `40|45 G|O` runs, for one
fixture and arm, the recorded run under nsys and the plan gate, rung 3,
the captured run with `pack_run.py` and `tierc_check.py`, in EXL3-G the
operation recording with `op_tier_e.py`, and the paged run (rungs 4 and
5), printing each step's exit status. Read the plan gate's `gate 0` before
any other verdict.
