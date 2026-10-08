<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Backend proof P0: toolchain bridges and reference controls — 2026-09-25

This is P0 of the [backend proof](../../backend-proof.md#stages): run the
toolchain bridges, run the held-out trajectories on both references, and
decode the tuned EXL3 shapes and grids. After review and an adversarial
challenge, a second pass added:
- every prefill row;
- a cuBLAS substitution arm;
- an FP64 oracle with per-layer captures;
- the FP16 bridge's recorded executed plan and GGML's workspace peaks;
- timing controls.

This gives the numerical profiles, bounds and performance protocol a
measured basis. They are [declared](../../backend-proof.md#p0-declarations)
for the owner's approval, in parts, each before the native output it
governs. No llmpalooza inference code runs here. Every
harness is an external reference harness.

**Results in brief.**

- **The bridges reproduce the references exactly.**
  - The FP16 reference rebuilt from its pinned source with llmpalooza's SDK
    (Clang 22.1.8, NVCC 13.4.92, cuBLAS 13.8.0.4) produces the same logits
    as the digest-pinned image, bit for bit. That holds for both
    trajectories, with fusion on and with fusion off.
  - ExLlamaV3's extension rebuilt with NVCC 13.4.92 produces the same logits
    as the NVCC 13.0 reference, bit for bit, for both fixtures, when both
    use the same frozen tuning cache.
- **Every reference arm repeats and restores exactly.** Differences appear
  only between arms that change the numerical plan.
- **ExLlamaV3's autotuner is part of its numerical plan.**
  - A fresh tuning run can choose a different launch plan, and so different
    logits: at most 0.172 absolute, on a 32-row prefill.
  - A frozen tuning cache reproduces its logits exactly, across processes
    and across compilers.
  - Every recorded choice decodes to a known projection and row bucket.
- **Disabling GEMV changes only single-token steps**, by at most 0.047.
  Prefill is identical.
- **The cuBLAS version is part of EXL3's plan.** Upstream runs its
  reconstruction GEMM (145 rows and more) in PyTorch's cuBLAS 13.1.1.
  Swapping in the SDK's 13.8.0.4 changes those logits and nothing else.
  ExLlamaV3's own fp16-accumulating GEMM stays off: a timing probe rejects
  it on GB10.
- **Every upstream arm is about equally close to an FP64 oracle.** The
  largest per-layer relative RMS error, averaged over rows, is below 0.6%;
  at the worst row it reaches 4.7%. The least accurate
  path is upstream's fused reconstruction at 1,024 rows, with a largest
  logit error of 0.554 at 4.0 bpw.
- **The FP16 bridge's executed plan is recorded.**
  - It includes every kernel with its SASS hash and launch shape, and every
    cuBLAS call with its algorithm.
  - Several of its choices depend on `n_kv`, on data addresses and on the
    handle's TF32 math mode: KQ runs as TF32.
  - GGML's pool peak comes entirely from the output head: 156,499,968
    bytes at 512 rows.
- **The EXL3 full-model bound is calibrated** against the FP64 oracle.
  Every one of 15 legitimate arms passes leave-one-out, native-like ones
  included, and five gross injected faults fail. Two subtle faults pass: a
  1% softmax-scale error and a BF16 rounding in one layer. The
  operation-level exactness gate is what catches those.
- **Native EXL3's operations are chosen and recorded.** GGML's F32 vector
  attention and F32 RoPE are more accurate than upstream's. GGML's default
  MMA prefill kernel is less accurate, because it accumulates V·P in F16.
  [`exl3-op-plan.json`](exl3-op-plan.json) records the plan kernel by
  kernel.
- **The reconstruction GEMMs can be pinned.** All 21 reproduce ExLlamaV3's
  cuBLAS call bit for bit from their recorded cuBLASLt configurations.
- **Timing needs noise calibrated across processes.**
  - Separate processes of one plan differ more than any in-session estimate
    allows.
  - A rule calibrated on A/A sessions passed a holdout decided in advance.
  - The NVCC 13.0 and 13.4 builds of ExLlamaV3 share identical SASS for
    only 16 of 1,506 kernels, so the timing reference is rebuilt with 13.4.

## Conditions and provenance

- **Host.** `spark` (`spark-c4e2`): GB10, DGX OS 7.6.0, kernel
  7.0.0-1019-nvidia, driver 580.178.04. The clock policy and idle states
  were left unchanged. No other GPU workload ran.
- **Fixtures.**
  - The FP16 GGUF of [first-slice](../first-slice/README.md): SHA-256
    prefix `8e0ae26000627ed6`.
  - The 4.0 and 4.5 bpw EXL3 fixtures of the
    [EXL3 reference](../exl3-reference/README.md): all files hash-verified
    by the harness on every run against that report's `pins.json`.
- **References.**
  - llama.cpp: the digest-pinned image of [first-slice](../first-slice/pins.json).
  - ExLlamaV3: `llmp-exl3-reference:20260922` with the M0 source,
    patch and extension cache.
- **Held-out IDs.**
  - 1,040 little-endian int64 token IDs from numpy 2.2.6
    `default_rng(20260923).integers(0, 151643, size=1040, dtype=int64)`.
  - SHA-256 `6dd8da897821f5615e7796f4d882795faf306a941f050e2eb68b619096e3fb0c`,
    which matches the declaration in the
    [EXL3 reference](../exl3-reference/README.md).
  - The EXL3 harness checks this hash; the FP16 harness checks the file's
    size and the ID range.
  - The IDs lie in the regular vocabulary, below the first special token.
- **Aggregates.** [`results.json`](results.json) holds:
  - every run's identity, settings and logit hashes;
  - every comparison;
  - the four frozen tuning caches, byte for byte (base64) and decoded.

  Raw logits (about 350 MB per FP16 run) and logs stay on the Spark.

## FP16 control (llama.cpp `b29c606e2`)

[`fp16_reference.cc`](fp16_reference.cc) drives llama.cpp's public API. It
evaluates a trajectory three times: fresh, repeated in a new context, and
with the state saved, the context destroyed, and the state restored into a
new context partway through. The repeat and the restored continuation must be
bit-identical to the first evaluation.

The settings are the ones in [first-slice.md](../../first-slice.md#numerical-comparison-contract):

- F16 K/V, one sequence, flash attention off;
- `offload_kqv` and `op_offload` on for CUDA;
- 8 CPU threads;
- `CUDA_DISABLE_PTX_JIT=1` and `GGML_CUDA_DISABLE_GRAPHS=1`;
- every row's logits requested.

It writes the raw F32 logits, the token IDs and a summary.

| Trajectory | Tokens | Context / batch | Chunks | Restore after |
| --- | ---: | --- | --- | ---: |
| `control` | 76 | 512 / 64 | 32, then 44 single tokens | 32 |
| `heldout` | 577 | 1,024 / 512 | 16, 17, 16 × 1, 512, 16 × 1 | 33 |

`control` is M0's trajectory. The `heldout` chunks sit either side of GGML's
16/17-column MMF/cuBLAS boundary, and include single tokens and a
512-token prefill whose cuBLAS path the 17-row chunk also takes.

| Arm | Build | Logits SHA-256 (prefix) | Repeat / restore |
| --- | --- | --- | --- |
| control, CUDA, fused | image | `bb8ae5e7e3ac6da7` | exact / exact |
| control, CUDA, fused | bridge | `bb8ae5e7e3ac6da7` | exact / exact |
| control, CUDA, unfused | image | `3560d3370b167ce8` | exact / exact |
| control, CUDA, unfused | bridge | `3560d3370b167ce8` | exact / exact |
| heldout, CUDA, fused | image (two processes) | `bfb36f192317c4a3` | exact / exact |
| heldout, CUDA, fused | bridge | `bfb36f192317c4a3` | exact / exact |
| heldout, CUDA, unfused | image | `69ff08210fb24503` | exact / exact |
| heldout, CUDA, unfused | bridge | `69ff08210fb24503` | exact / exact |
| heldout, CPU | image | `6b78aaacac4d4420` | exact / exact |

The control arms reproduce M0's recorded hashes (fused `bb8ae5e7…`, unfused
`3560d337…`). "Unfused" is `GGML_CUDA_DISABLE_FUSION=1`.

Differences between reference arms:

| Comparison | Max abs | RMS | Row max p50 / p99 | Top-1 equal | Disagreeing rows' margins |
| --- | ---: | ---: | --- | ---: | --- |
| control, fused vs unfused | 0.0477 | 0.00330 | 0.0168 / 0.0381 | 76 / 76 | — |
| heldout, fused vs unfused | 0.1563 | 0.01318 | 0.0596 / 0.1037 | 568 / 577 | ≤ 0.035 |
| heldout, CUDA vs CPU | 0.1658 | 0.01309 | 0.0551 / 0.1135 | 564 / 577 | ≤ 0.033 |

Each disagreement is a near-tie in both runs. These are the plan-to-plan
and backend differences the proof must localize rather than absorb. They are
not tolerances.

### The bridge build

[`bridge/bridge.sh`](bridge/bridge.sh) configures the pinned llama.cpp source
with [`bridge/CMakeLists.txt`](bridge/CMakeLists.txt), which links the same
harness. The toolchain is the one the `spark-native` preset uses:

- the SDK's Clang with its static GCC 16.2 runtime and the host's GNU
  linker;
- NVCC 13.4.92 with that Clang as host compiler, SASS for `sm_121a` only
  (GGML's CMake maps `121-real` to `121a`);
- cudart 13.4.92, and cuBLAS 13.8.0.4 from NVIDIA's repository, which D-076
  now pins in the SDK.

The executable finds `libcublas.so.13` and `libcublasLt.so.13` through its
RUNPATH; `ldd` confirmed they resolve there. The image carries cudart
13.3.29 and cuBLAS 13.5.1.27, so the identical logits also show these
trajectories do not depend on those versions. It departs from upstream's
image build in three ways:

- static libraries instead of dynamically loaded backends, because the SDK's
  C++ runtime is static only;
- no OpenMP, because the SDK has no OpenMP runtime;
- `sm_121a` SASS only.

The first two touch only the CPU backend, which the CUDA arms do not
exercise. The bridge ran directly on the host rather than in the container.

## EXL3 held-out trajectory (ExLlamaV3 `6b84a21b`)

[`exl3_heldout.py`](exl3_heldout.py) follows the M0 reference's numerical
mode:

- Model API forward;
- `flash_attn` attention mode;
- an F16 cache of 4,096 tokens and one sequence;
- pinned staging for single-token steps;
- four CPU threads.

For each prefix of 32, 144, 145, 1,023 and 1,024 held-out IDs, it prefills
(recording every row's logits through 145 rows), then runs 16 teacher-forced
single-token steps. It repeats that three times. It then snapshots the cache,
poisons it with NaN, restores it and runs the suffix again. All repeats and
the restored suffix must be bit-identical, and the restored storage must hash
the same.

The arms run in the reference container
([`exl3_run.sh`](exl3_run.sh)). Each names its autotuning cache with
`EXLLAMAV3_TUNE_CACHE`:

- a missing cache is tuned during the run and written out;
- a copy of an existing cache is reused unchanged: the harness hashes the
  cache before and after.

The arms are:

- **optimized**, which tunes a fresh cache for each fixture: `tune-40`
  (`4760513062b614b2`) and `tune-45` (`d85a85ae1029cf1d`);
- **repeat**, the same profile with a copy of that cache;
- **attention_eager** (`EXL3_BC_ATTN=0`);
- **GEMV off** (`EXL3_GEMV=0`), which also tunes the shapes GEMV had served
  and so freezes the GEMM-only caches `tune-40-gemvoff`
  (`c6c01843042cde6e`) and `tune-45-gemvoff` (`e5de5226aa03bf12`);
- **GEMV off repeat**, with a copy of that cache;
- **bridge**, the optimized profile with the extension rebuilt by NVCC 13.4
  (below);
- **fresh tuning**, which retunes from an empty cache: once for the
  optimized profile, and four times for GEMM only.

All 22 runs passed every repeat and restore check. Every "exact" below is
also confirmed by the runs' raw-byte SHA-256s in `results.json`.

Differences from the optimized arm, or for GEMM-only arms from the frozen
GEMM-only arm. Prefill covers the 321 rows of the 32-, 144- and 145-row
prefills; suffix covers the 80 single-token steps.

| Fixture | Arm | Prefill max abs / RMS | Prefill top-1 | Suffix max abs / RMS | Suffix top-1 |
| --- | --- | --- | ---: | --- | ---: |
| 4.0 | repeat, eager, bridge | exact | 321 | exact | 80 |
| 4.0 | GEMV off | exact | 321 | 0.0469 / 0.0050 | 79 |
| 4.0 | fresh tuning | 0.1367 / 0.0061 | 320 | 0.0703 / 0.0077 | 80 |
| 4.0 | GEMM only: repeat | exact | 321 | exact | 80 |
| 4.0 | GEMM only: 4 fresh tunings | 0–0.1367 / ≤ 0.0100 | 317–321 | 0.0273–0.0605 / ≤ 0.0072 | 78–80 |
| 4.5 | repeat, eager, bridge | exact | 321 | exact | 80 |
| 4.5 | GEMV off | exact | 321 | 0.0254 / 0.0040 | 79 |
| 4.5 | fresh tuning | 0.0625 / 0.0065 | 320 | 0.0664 / 0.0075 | 79 |
| 4.5 | GEMM only: repeat | exact | 321 | exact | 80 |
| 4.5 | GEMM only: 4 fresh tunings | 0.0566–0.1719 / ≤ 0.0092 | 320–321 | 0.0498–0.0527 / ≤ 0.0075 | 79–80 |

RMS is the largest per-prefix value. The margin at every top-1 disagreement
is at most 0.0273, in both runs.

- **The 145-row prefill never changes.** Upstream reconstructs from 145
  rows, and that path does not consult the tuner.
- **Every retuning changes some of the other rows.** Each of the ten fresh
  tunings chose differently from its frozen cache, for two to eight of its
  9 to 24 keys (`results.json` lists each run's count).
- **The attention profile changes nothing.** Eager and optimized attention
  give identical logits at these prefixes.

The ExLlamaV3 reference is therefore reproducible only with its tuning cache
frozen. A native plan must force the recorded tile shape, block size, SM
count and concurrency to be comparable at all.

### Decoded tuning

[`decode_tuning.py`](decode_tuning.py) rebuilds the tuner's keys from the
pinned source (`exl3_gemm.cu`, `coop_autotune.cu`):

- The key is FNV-1a over the problem: half-K, the row bucket, k, n, K, FP32
  output, device, compute-capability class, SM count and codebook.
- The fused gate/up kernel also mixes its row block sizes and a
  sliced-launch flag.
- The key is then salted with the tuner's version.

The decoder enumerates Qwen2.5-0.5B's projections. Every record in every
cache matched a case; none was left unexplained.

The row bucket is the next power of two of the rows, capped at 16. So one
choice serves every packed GEMM of 16 to 144 rows. The frozen caches record
only the buckets the trajectory reached: 1 or 2 rows, and 16 or more.

Examples from `tune-40-gemvoff`, given as (tile shape, block, SMs,
concurrency):

| Projection (k × n) | Rows | Choice |
| --- | --- | --- |
| q/o 896 × 896, K 4 | ≥ 16 | 2, 512, 14, 1 |
| k/v 896 × 128, K 4 | ≥ 16 | 2, 512, 4, 1 |
| gate/up fused, K 4 | 1, and ≥ 16 | 3, 512, 24, 2 |
| down 4,864 × 896, K 4, FP32 out | ≥ 16 | 2, 512, 40, 1 |
| head 896 × 151,936, K 8 | ≥ 16 | 2, 512, 48, 1 |
| head 896 × 151,936, K 8 | ≤ 2 | 1, 256, 48, 1 |

`results.json` lists every record of the four frozen caches.

### The bridge build

The bridge compiles ExLlamaV3's extension with the SDK's NVCC 13.4.92. It
keeps upstream's flags (`-O3 --use_fast_math`, `sm_121`), the host compiler
wrapper, and PyTorch.

PyTorch's headers include CUDA libraries the SDK does not carry: cuSPARSE,
cuSOLVER, cuRAND and others. The build tree is a hard-linked copy of the
SDK's CUDA with cuBLAS 13.8.0.4 added, and with 96 missing headers copied
from the host's CUDA 13.0. None of those copies replaced an SDK file.

The build's dependency records show which copied headers the compiled code
reached:

- cuSPARSE and cuSOLVER, through PyTorch's API headers;
- cuRAND's device header, only in the four sampling translation units, which
  a teacher-forced run does not execute.

The extension imports no cuSPARSE, cuSOLVER or cuRAND symbol. Its cubins are
`sm_121`, and its version strings are NVCC 13.4.92.

At run time it uses PyTorch's own cudart and cuBLAS (`cu130`). This bridge
therefore separates the device compiler, not the runtime libraries. The
second pass measures the cuBLAS version separately (below).

## EXL3 second pass: every row, cuBLAS and an FP64 oracle

The harness's second version (`exl3_heldout.py`, `v2-*` runs):
- runs a full forward pass for every prefix, so the 1,023- and 1,024-row
  prefills record every logit as well;
- records the cuBLAS and cuBLASLt files mapped into the process, with
  cuBLASLt's version;
- records whether ExLlamaV3's fp16-accumulating reconstruction GEMM
  (`hgemm_recon`) is active. A timing probe decides that unless
  `EXL3_HGEMM_F16ACC` is set.

With `--capture`, a further instrumented pass per prefix records:
- the residual stream after every block;
- the final norm's output;
- the cache's K and V over the prefix.

That pass's logits must equal the uninstrumented prefill's.

The controls held:
- Its first run reproduced every hash that the first version's frozen
  GEMM-only arm recorded. The first version recorded no 1,023- or
  1,024-row prefill logits.
- Every capture reproduced its logits exactly.
- On GB10 the probe turns the fp16-accumulating GEMM off, so reconstruction
  calls `cublasGemmEx`: FP16 inputs, `CUBLAS_COMPUTE_32F`, a 16 MiB
  workspace, on PyTorch's handle.
- Every later arm pins the probe's answer (`--hgemm-f16acc 0`). The arm left
  to the probe is identical.

All 23 second-pass runs passed their repeat and restore checks. Against the
frozen GEMM-only arm (`v2-*-g-cap`):

| Arm | Differs from the frozen GEMM-only arm |
| --- | --- |
| repeat without capture; probe-decided GEMM; attention_eager; NVCC 13.4 bridge | nowhere: every prefill and suffix, 1,024 rows included |
| four retunings, each replayed from its recorded cache (`gemmfresh-1…4`) | at 144 rows or fewer, and in single-token steps; the 145-, 1,023- and 1,024-row prefills are identical |
| GEMV on | in single-token steps only |
| cuBLAS 13.8.0.4 | at the 145-, 1,023- and 1,024-row prefills and their suffixes only |

**cuBLAS substitution.** The first attempt preloaded the SDK's cuBLAS
13.8.0.4 (`v2-*-cublas138`). PyTorch's 13.1.1 stayed mapped as well, so
that arm is superseded. The clean arm (`v2-*-cublas138b`) bind-mounts the
two 13.8.0.4 libraries over PyTorch's files, and the process maps only
13.8.0.4 (cuBLASLt version 130800).

The version changes exactly the rows that reach cuBLAS. A native
reconstruction GEMM therefore has to be compared with this arm, not with
the 13.1.1 reference.

### FP64 oracle

- **Export.** [`exl3_export.py`](exl3_export.py) exports each fixture's 169
  EXL3 linears as upstream decodes them: the rotated-basis trellis
  reconstructed to FP16 by upstream's own kernel, plus the FP16 sign
  vectors. It adds the checkpoint's biases, norms and BF16 embedding.
- **Forward pass.** [`oracle.py`](oracle.py) rebuilds each weight in FP64:
  both 128-point Hadamard transforms and both sign vectors. It then runs
  Qwen2.5-0.5B's forward pass over all 1,040 held-out IDs in FP64 on the
  CPU. Causal attention makes row i the exact counterpart of every prefill
  or single-token row at position i.
- **Scoring.** [`oracle_compare.py`](oracle_compare.py) scores each run
  against the oracle:
  - per row: the largest absolute and the RMS logit error;
  - per section: their maxima, the overall RMS and top-1 agreement;
  - for captured runs, per layer: the residual stream's relative RMS error,
    and K and V's largest error over positions.

[`oracle-envelope.json`](oracle-envelope.json) holds each arm's scores and
the envelope: the worst value over each fixture's ten runs. Those are nine
distinct arms plus the uncaptured repeat, which is identical to the
captured arm. The superseded preload arm is left out.
`oracle_compare.py` produced it; the committed file drops the per-row and
per-position arrays. The table below rounds to the nearest; the JSON is
normative.

Envelope, per prefix:

| Fixture | Prefill max abs (32 / 144 / 145 / 1,023 / 1,024 rows) | Prefill worst-row RMS | Single-token max abs | Worst layer: relative RMS / K / V |
| --- | --- | --- | --- | --- |
| 4.0 bpw | 0.174 / 0.175 / 0.172 / 0.177 / 0.554 | 0.037 (1,024 rows: 0.102) | 0.041–0.072 | 0.0049 / 0.147 / 0.137 |
| 4.5 bpw | 0.207 / 0.199 / 0.219 / 0.215 / 0.391 | 0.043 (1,024 rows: 0.081) | 0.039–0.132 | 0.0056 / 0.123 / 0.121 |

The arms lie close together. Two observations:
- **The same computation, with the numerical plan varied upstream's way,
  keeps a characteristic error against exact arithmetic.** This supports
  the oracle as the bound's common reference.
- **Upstream's fused reconstruction at 1,024 rows** (Hadamard folded into
  FP16 reconstructed weights) is less accurate than the other prefills in
  every arm: about three times at 4.0 bpw, and 1.8–1.9 times at 4.5 bpw.

**The envelope does not work as a zero-slack bound.** A leave-one-out test
scored each arm against the envelope of the others. Nine of the twelve arms
that really use a different plan fail it:

| Fixture | Arm | Failing statistics |
| --- | --- | --- |
| 4.0 bpw | retuning 1 | 108 |
| 4.0 bpw | retuning 2 | 37 |
| 4.0 bpw | GEMV on | 2 |
| 4.0 bpw | cuBLAS 13.8 | 143 |
| 4.5 bpw | each retuning | 25–56 |
| 4.5 bpw | cuBLAS 13.8 | 142 |

The arms that pass are the ones with identical hashes, or plans identical
at prefill. At the reconstruction-path prefixes, only two arms really
differ, so the envelope is mostly extreme values. The Tier C bound therefore
stays proposed ([declarations](../../backend-proof.md#p0-declarations)).

## EXL3 third pass: calibrating the full-model bound

The second pass's envelope failed as a bound (above), because too few of
its arms really differ. The third pass (`v3-*`, all captured) runs
EXL3-G's frozen plan with 15 legitimate arms and 7 injected faults per
fixture. It also captures K and V after the 16 single-token steps.

**Legitimate arms:**
- frozen EXL3-G;
- its four retunings, replayed from their recorded caches;
- GEMV on;
- cuBLAS 13.8.0.4;
- five variants that swap in upstream's own alternative code for the
  non-linear operations (`exl3_heldout.py --variant`):
  - `rmsnorm_torch`: RMSNorm's PyTorch path (`forward_torch`), residual
    added first;
  - `rope_torch`: RoPE's PyTorch path (`apply_torch`);
  - `attn_general`: the general Triton paged attention, without the fast
    decode and prefill kernels;
  - `attn_sdpa`: PyTorch SDPA through upstream's paged fallback
    (`_torch_bighead_fallback`), with its head-size gate lifted;
  - `all_torch`: all three together, also run with cuBLAS 13.8.0.4;
- `ggml_ops`: the native-like arm, running the declared native operation
  plan's GGML kernels (below), with and without cuBLAS 13.8.0.4.

Variants and faults run with `EXL3_BC_ATTN=0`, which is bit-identical to the
optimized profile. The flag takes attention through the Python path the
patches reach. Each variant changes the logits at every prefix.

**Faults** (injected bugs):

| Fault | What it changes |
| --- | --- |
| `f_q_rope_offset_l10` | Q alone one RoPE position off in layer 10 |
| `f_rope_offset_l10` | Q and K one RoPE position off together in layer 10, which RoPE's relative form nearly cancels |
| `f_one_key_l10` | position 20's key overwritten by position 21's in layer 10 |
| `f_decode_rope_l12` | RoPE one position off in layer 12, single-token steps only |
| `f_norm_eps` | RMSNorm epsilon 1e-5 instead of 1e-6 |
| `f_softmax_scale_l8` | softmax scale 1% high in layer 8 |
| `f_bf16_mlp_l16` | layer 16's MLP output rounded to BF16 |

Each takes effect. The decode fault leaves every prefill untouched.

[`oracle_compare.py`](oracle_compare.py) now also scores K and V per
position relative to the oracle's magnitude there, over the prefix and over
the positions the single-token steps wrote.

[`tierc.py`](tierc.py) divides each of 750 statistics per fixture by the
median of the legitimate arms' distinct values. Arms bit-identical on a
statistic count once: for example, the retunings at the reconstruction-path
prefixes share the frozen arm's plan there. For a legitimate arm, the
median leaves out its own value. The statistics fall into two families:
- *averaged*: the logits' RMS error, and each block's row-averaged relative
  error;
- *extreme*: the worst row, the largest absolute error, and the worst K or
  V position.

Its result, in [`tierc.json`](tierc.json), as the largest ratio reached:

| | 4.0 bpw, averaged / extreme | 4.5 bpw, averaged / extreme |
| --- | --- | --- |
| worst legitimate arm (leave one out) | 1.25 / 2.14 | 1.12 / 1.51 |
| native-like arms (`ggml_ops`, with and without cuBLAS 13.8) | 1.09–1.15 / 1.78 | 1.07 / 1.34–1.46 |
| `f_q_rope_offset_l10` | 190 / 314 | 181 / 309 |
| `f_norm_eps` | 25.5 / 57.6 | 23.3 / 59.0 |
| `f_one_key_l10` | 7.9 / 188 | 7.5 / 156 |
| `f_rope_offset_l10` | 1.06 / 176 | 1.02 / 167 |
| `f_decode_rope_l12` | 3.5 / 147 | 4.4 / 145 |
| `f_softmax_scale_l8` | 1.41 / 2.27 | 1.32 / 2.51 |
| `f_bf16_mlp_l16` | 1.07 / 1.77 | 1.08 / 1.39 |

With thresholds of 2 (averaged) and 8 (extreme), every legitimate arm
passes on both fixtures. The five gross faults fail. Each exceeds a
threshold at least 1.75-fold (averaged) or 7-fold (extreme).

The two subtle faults pass. A 1% softmax-scale error or a BF16 rounding
confined to one layer moves the model's accuracy no more than a legitimate
change of implementation does. A full-model accuracy bound cannot see
them. The declarations therefore catch them elsewhere: with an
operation-level exactness gate on the GGML kernels, and a dtype-chain check.

## GGML operations for native EXL3

To choose the GGML paths native EXL3 uses for its non-linear operations,
this study
- read the pinned ggml-cuda source;
- built GGML's CUDA backend (unmodified `ggml/` at `b29c606e2`) as a shared
  library inside the reference container ([`ggml_shim/`](ggml_shim/));
- ran each candidate on operations captured from frozen EXL3-G
  ([`ggml_ops.py`](ggml_ops.py), [`ggml_ops_capture.py`](ggml_ops_capture.py),
  [`ggml_ops_score.py`](ggml_ops_score.py), [`ggml_ops_run.sh`](ggml_ops_run.sh)).

The captures cover layers 0, 6, 12, 18 and 23, each prefix's prefill and
single-token steps 0 and 15, on both fixtures. The capture reproduced frozen
EXL3-G's logits bit for bit. Every operation was recomputed in FP64 from its
captured inputs. The path study, with source citations, and every
measurement are in [`ggml-ops.json`](ggml-ops.json).

- **Build.** The library is built with GGML's own flags:
  `-use_fast_math`, `sm_121a`. Builds with the container's NVCC 13.0.88
  and the SDK's 13.4.92 gave identical outputs for all 3,670 operation
  cases.
- **Attention selection on GB10** (`fattn.cu`):
  - The vector kernel runs only for one query row with K's length a
    multiple of 256, which llama.cpp arranges by padding. Everything else
    runs the MMA kernel.
  - The vector kernel computes in F32.
  - The MMA kernel converts Q to F16 and accumulates V·P in F16.
  - The flash-attention precision flag is not read by any CUDA kernel.
- **Accuracy against FP64** (the worst relative RMS error in each phase;
  ranges over both fixtures and all phases, from `ggml-ops.json`'s
  `rel_rms_max`):

  | Operation | Upstream | GGML |
  | --- | --- | --- |
  | RMSNorm (F16 output) | 2.1–2.7e-4 | ratio 1.00; bit-identical in 211 of 330 cases |
  | residual add | — | bit-identical (150 of 150) |
  | RoPE, F32 output | 2.0–2.6e-4 | 6.2e-7–3.8e-5 |
  | attention, vector kernel, F32 output | 2.2–3.6e-4 | 2.3–4.1e-6 |
  | attention, MMA kernel (GGML's prefill default) | 2.2–3.6e-4 | up to 2.85 times upstream's at 1,023–1,024 rows |
  | attention, tile kernel | 2.2–3.6e-4 | 13–24 times upstream's |
  | SwiGLU | — | bit-identical (50 of 50) |
  | embedding lookup (BF16 table) | — | bit-identical (10 of 10) |

- **The recommended plan.** The declarations adopt it as the native EXL3
  operation plan:
  - F32 norm, add and RoPE;
  - the vector attention kernel for every phase, with F32 Q, forced for
    prefill;
  - F16 at the linears' inputs.

  Its attention scratch is 48 KB at decode, and 0.36, 1.6 and 11.4 MB at
  32, 144 and 1,024 prefill rows. It never materializes the scores.
- **The native-like arm.** `ggml_ops` runs that plan inside upstream,
  keeping ExLlamaV3's linears. It passes every repeat, restore and capture
  check. Its overall accuracy against the oracle is within 1.5% of frozen
  EXL3-G's (geometric mean ratio 0.99–1.01), since both round to F16 at
  the linears.

### The operation plan record

[`exl3-op-plan.json`](exl3-op-plan.json) is the native EXL3-G plan as a
machine-checkable record. [`exl3_op_plan.py`](exl3_op_plan.py) makes it in
three steps.

1. **Probe.** In the reference container, with the NVCC 13.4.92 build of
   the shim (`cuda134`) and cuBLAS 13.8.0.4, the probe starts from the
   `ggml_ops` variant. It moves the three operations that arm leaves to
   upstream onto their plan owners:
   - the embedding lookup, to GGML `get_rows`;
   - the MLP residual add, to GGML `add`;
   - the reconstruction-path bias add, to ExLlamaV3's `add_kernel_hhh`.

   A checked run of each phase kind compares each moved operation with
   upstream's on the same inputs:

   | Operation | Cases per fixture | Bit-identical |
   | --- | ---: | ---: |
   | embedding lookup | 9 | all |
   | MLP residual add | 216 | all |
   | bias add, reconstruction path | 288 | all |

   The run's logits equal the `ggml_ops` arm's with cuBLAS 13.8.0.4. Then
   the PyTorch profiler (CUPTI) records one run of each phase kind: every
   kernel, memcpy and memset, attributed to its operation through the
   runtime call that launched it. The profiled logits are equal too, in
   all 14 fixture and phase-kind pairs.
2. **SASS.** `cuobjdump` 13.0.85 hashes the same library's kernels as
   `fp16_plan.py` does, and gives their registers and static shared memory.
3. **Build.** It reduces both probes to one plan per phase kind:
   - the embedding, the output, and one decoder layer, after checking
     that the operation order and every GGML launch are the same in all
     24 layers and on both fixtures;
   - the linears' launches per layer, since the 4.5 bpw fixture mixes
     bit widths;
   - the dtype chain: the build checks every observed operation input and
     output (9,768 in all) against the declared tensors.

Two independent probe runs gave identical launches. The plan uses 11 GGML
kernels. Four of them are also in the FP16 bridge (`rms_norm_f32`,
`rope_neox`, the F32 add and the SwiGLU kernel), and their SASS is
identical in both builds despite the different host compilers. Mangled
names match only once NVCC's per-file `_INTERNAL_` hash is normalized, so
the record gives that form too.

The probe also ran ExLlamaV3's `add_kernel_hhh` against PyTorch's F16 add
on all 2^32 input pairs:
- every non-NaN result bit-identical (4,030,980,098 pairs);
- NaN in the same 263,987,198 pairs.

The raw probes are on `spark` under `p0-20260925/opplan/`. Reproduction is
in the script's help.

### Reconstruction GEMM pinning

[`cublaslt_pin_probe.cc`](cublaslt_pin_probe.cc) is built with the SDK
(Clang 22.1.8, cuBLAS 13.8.0.4, CUDA 13.4 runtime). It replays the 21
recorded reconstruction GEMMs on seeded random FP16 inputs, three ways:
- ExLlamaV3's legacy `cublasGemmEx` call;
- cuBLASLt with the heuristic's algorithm;
- cuBLASLt with an algorithm rebuilt by `cublasLtMatmulAlgoInit` from its
  complete configuration: all nine attributes.

All three outputs are bit-identical for all 21 GEMMs. The probe's legacy
calls resolve exactly the algorithms the reference arm logged. None uses
split-K or a workspace. The configurations are in
[`exl3-recon-pin.json`](exl3-recon-pin.json).

### Reconstruction GEMM plan

The `plan-*` runs repeat frozen EXL3-G with cuBLAS API logging and cuBLASLt
logging at level 5, under PyTorch's cuBLAS 13.1.1 and under 13.8.0.4.
[`cublas_plan.py`](cublas_plan.py) reduces the logs to
[`exl3-recon-plan.json`](exl3-recon-plan.json).

- Each fixture makes 21 distinct reconstruction GEMMs, 2,076 calls in all.
  They are `cublasLtHSHMatmul` (FP16 output) and `cublasLtHSSMatmul` (FP32
  output), with `COMPUTE_32F` and 48 SMs targeted.
- The heuristic is queried with a 16 MiB workspace limit and 16-byte
  alignment on every operand. The handle's math mode is
  `CUBLAS_DEFAULT_MATH`. PyTorch also sets a 32 MiB workspace on the same
  handle, but ExLlamaV3 resets it to 16 MiB before its calls.
- The two versions resolve different algorithms for 19 of the 21 GEMMs:
  different tiles, and split-K in some. That explains why the logits
  differ.
- The record lists each call's resolved algorithm: `algoId`, tile, stages,
  split-K and reduction scheme, custom option.

### Per-phase memory

The `mem-*` runs record PyTorch's peak allocation above each phase's start.
The results are the same for both fixtures:

| Phase | Peak above start (bytes) | FP16 logits part | The rest |
| --- | ---: | ---: | ---: |
| prefill, 32 rows | 9,953,792 | 9,723,904 | 229,888 |
| prefill, 144 rows | 44,790,272 | 43,757,568 | 1,032,704 |
| prefill, 145 rows | 103,822,336 | 44,061,440 | 59,760,896 |
| prefill, 1,023 rows | 376,915,456 | 310,861,056 | 66,054,400 |
| prefill, 1,024 rows | 375,390,720 | 311,164,928 | 64,225,792 |
| single-token step | 310,272 | 303,872 | 6,400 |

From 145 rows, the rest is mostly reconstruction. At 145 rows, the output
head's 896 × 32,768 FP16 slice is 58,720,256 bytes.

Outside PyTorch's per-phase peak there are two more buffers:
- ExLlamaV3's device context: a 16 MiB workspace and 4,202,760 lock bytes;
- PyTorch's cuBLAS workspace: 32 MiB, set on the handle as the cuBLAS log
  shows, and allocated before these phases.

The declarations limit each phase to this peak plus an allowance for the
native plan's own needs: the vector attention kernel's scratch, and the
plan's F32 intermediates. Persistent workspaces are limited separately, to
ExLlamaV3's device context. The `mem138-*` runs repeated these phases under
cuBLAS 13.8.0.4 with identical results. The "rest" column is only an
attribution: the logits
output is allocated at the end of the phase, so peak minus logits is not
the peak of everything else.

### Kernel launch record

[`exl3_launch_record.py`](exl3_launch_record.py) profiles one invocation of
each of the 176 BP-F2 cases with EXL3-G's settings, on the timing sessions'
frozen caches. It records every kernel launched, in order, with its grid,
block, shared memory (static plus dynamic, as the profiler reports it) and
registers.

It runs on the reference that native's timings are compared with:
- ExLlamaV3's extension built with the SDK's NVCC 13.4.92 (the EXL3 bridge
  build, SHA-256 `aa8b9f16…`);
- the SDK's cuBLAS 13.8.0.4, bind-mounted, the only cuBLAS mapped.

Along with the launches, [`exl3-launch.json`](exl3-launch.json) holds:
- the mapped libraries and their versions;
- the extension's hash;
- the caches' hashes before and after, which are unchanged;
- the matching rule;
- each ExLlamaV3 kernel's SASS hash from `cuobjdump`, addresses stripped:
  all 308 ExLlamaV3 launches matched one.

It supersedes two earlier records (the build comparison is
[`exl3-sass-compare.json`](exl3-sass-compare.json)):
- one made under PyTorch's cuBLAS 13.1.1, whose cuBLAS kernels native
  could not reproduce;
- one made on the NVCC 13.0.88 build. That build shares identical SASS
  with the 13.4.92 build for only 16 of its 1,506 kernels.

The launches are ExLlamaV3's kernels, cuBLAS's (`nvjet_*`, CUTLASS), and
PyTorch's `elementwise_kernel` for the bias add in 12 cases.

## FP16 executed plan and workspace

[`fp16_plan.py`](fp16_plan.py) builds the record in
[`fp16-plan.json`](fp16-plan.json) from three sources: an `nsys` CUDA trace
of each bridge arm, cuBLAS/cuBLASLt logging, and per-function SASS hashes
from `cuobjdump`. Every profiled run reproduced its logits hash, in all four
arms.

- **Build.**
  - All 143 ggml-cuda translation units share one flag set:
    `-O3 -use_fast_math`, SASS for `sm_121a`, `GGML_CUDA_USE_GRAPHS`.
  - Each of the 29 kernels the arms launch appears once in the executable.
    Its SASS matches its object's.
- **cuBLAS.**
  - One handle per context, set up at `common.cuh:1540-1553`:
    `CUBLAS_TF32_TENSOR_OP_MATH`, the context's stream, a 32 MiB workspace.
  - All GEMMs go through cuBLASLt with 48 SMs targeted: 21 distinct calls,
    217 per chunk above 16 rows. The record lists each call's parameters
    and resolved algorithm, including split-K.
  - KQ is `GemmBatchedEx` with `COMPUTE_32F`, which the math mode runs as
    TF32.
- **Conditions a native plan must reproduce.** The record lists each with
  its source line:
  - `n_kv`'s padding to 256;
  - MMVF versus MMF for KQ, which switches when `n_kv` reaches 768;
  - the softmax variant, which depends on `n_kv`;
  - the fusion gates that compare data ranges;
  - `get_rows`' vector variant, chosen by pointer alignment;
  - 128-byte buffer alignment;
  - the precision changes fusion brings: fused single-token gate/up
    accumulates in F32, unfused in F16;
  - embedding lookup on the host;
  - PDL launch attributes.
- **Workspace.**
  - An instrumented copy ([`fp16_pool_peak.sh`](fp16_pool_peak.sh),
    [`fp16-pool-peak.patch`](fp16-pool-peak.patch)) recompiles only
    `ggml-cuda.cu`. Its logits were bit-identical in all four arms.
  - GGML's pool peaks at 0, 0, 5,196,288, 9,781,248 and 156,499,968 bytes
    for 1-, 16-, 17-, 32- and 512-row chunks. Each peak is the output
    head's F16 output temporary plus the F16 copy of its input.
  - Compute buffers are 37.31 MiB (`control`) and 298.50 MiB (`heldout`).
  - KV is 6 and 12 MiB.

### CPU diagnostic

The bridge's CPU path is not bit-identical to the image's CPU reference:

| Run | CPU backend | Logits SHA-256 (prefix) |
| --- | --- | --- |
| bridge | built for `armv8-a` | `9f8f298e1ac25034` |
| image | the `armv8.6_2` variant, selected at run time | `6b78aaacac4d4420` |

The largest difference is 0.063 (RMS 0.0039), and top-1 agrees on 573 of
577 rows. The CPU path (BP-N7) is therefore a reported diagnostic, not a
gated rung.

## Timing controls

All sessions time the 176 EXL3 kernel cases of the
[M0 reference](../exl3-reference/README.md#packed-kernel-and-reconstruction-boundaries).
They use M0's unchanged `measure.py`, run by
[`timing_session.sh`](timing_session.sh) in the reference container on
frozen tuning caches, and summarized by [`timing_stats.py`](timing_stats.py).
Each sample is one graph replay of ten invocations. Each block is a separate
process of 31 samples per case. [`timing.json`](timing.json) holds every
session, condensed from `timing_stats.py`'s output. For each it records the
tuning caches (bytes and hashes), the arm settings and the script version:
`aa` and `gemv` ran before `ORDER` and the warm-up process existed, and
`aa4` and `aa4b` before the warm-up.

| Session | Arms | Blocks per arm | Passing, with the reference's allowance (pair / range) | Sign test (slower / faster, p) |
| --- | --- | --- | --- | --- |
| `aa` | EXL3-G against itself, one plan | 2 | 150 / 137 | 97 / 77, 0.15 |
| `aa4` | the same, cold GPU at block 1 | 4 | 156 / 166 | 106 / 68, 0.005 |
| `aa4b` | the same, cold GPU at block 1 | 4 | 144 / 154 | 96 / 79, 0.23 |
| `aa5` | the same, warm-up process first | 4 | 150 / 161 | 100 / 75, 0.07 |
| `aa5b` | the same, warm-up process first | 4 | 168 / 162 | 87 / 87, 1.0 |
| `gemv` | EXL3-O (reference) against EXL3-G | 2 | 128 / 121 | 109 / 67, 0.002 |

What the A/A sessions show:

- **Separate processes of one plan differ by more than one pair of
  reference processes suggests.** The first rule drafted (M0's rule on two
  blocks per arm) failed 26 of 176 A/A cases. The misses were 0.2–2.5%. In
  13 of them, the reference's own two blocks agreed within 0.3%.
- **The first timed block tends to run faster.** Its median against the
  arm's later blocks:

  | Session | Start | First block | Faster in |
  | --- | --- | --- | --- |
  | `aa4` | idle GPU (208 MHz at the boundary) | 0.5% faster | 78% of cases |
  | `aa4b` | idle GPU | 0.14% faster | 61% |
  | `aa5` | warm-up process first | 0.19% faster | 61% |
  | `aa5b` | warm-up process first | 0.07% slower | 45% |

  The sign test rejected `aa4` on a 0.07% median shift. A warm-up process
  does not clearly remove the effect.
- **Some synthetic shapes are unstable across processes.** k = 14,336 × n =
  4,096 and k = 4,096 × n = 14,336 at K = 4, 1 to 32 rows, spread their
  process medians by 10–15%. The 4.0 bpw `down_proj` at 145 rows is
  borderline.
- **Rules that estimate the noise inside one session fail A/A.**
  - The second draft's rule, with four blocks per arm, the range allowance
    and a confirmation session, still left about one false failure per
    pair of sessions.
  - Using the reference's allowance alone, with a two-of-three session
    rule, left one or two per trio: M0's rule and every in-session variant
    tried.
  - Four processes per arm per session are too few to estimate each case's
    process-to-process spread.

**The calibrated rule.** [`timing_protocol.py`](timing_protocol.py) takes
each case's noise σ from calibration A/A sessions instead. σ is the
relative standard deviation of block medians within a session, pooled over
the sessions: 32 process medians of one plan per case. Its development ran
on the cuBLAS 13.1.1 sessions:

- **First version.** Its σ came from `aa4`, `aa4b`, `aa5` and `aa5b` (median
  0.62%). It had three parts:
  - a per-case test, `d = (ratio − 1) / (σ · √½)` against 3.86 (a 1%
    family-wise false-failure rate over 176 cases);
  - an aggregate over cases treated as independent;
  - spread checks: a case failed when its candidate's block medians spread
    more than 3σ, and was inconclusive when its reference's did.

  It passed every in-sample pairing. Out of sample (σ from three sessions,
  the fourth tested), one or two cases failed a single session, and the
  confirmation cleared them.
- **First holdout pair (`h1`, `h2`).** The first version failed it:
  - The 4.0 bpw `up_proj` at 1 row tripped the spread check in both
    sessions: one of its eight processes ran 11.08 µs and another 11.45 µs,
    against about 11.26 µs.
  - `h1`'s aggregate reached 2.36, over its 2.33 limit. A whole process can
    run fast or slow, so the cases are not independent.
  - The spread checks were removed, and the rule was fixed at `c35d9acf…`.
- **The challenge after that.** It showed two further problems:
  - these sessions ran PyTorch's cuBLAS 13.1.1, while the native plan pins
    13.8.0.4;
  - the aggregate's null model was wrong.

  The second holdout pair (`h3`) was stopped unevaluated, and the rule was
  revised:
  - the aggregate became a block-level test: each block's session-wide shift
    is the median over cases of its block median relative to the case's
    mean, and the candidate's four shifts are compared with the reference's
    by a one-sided t-test, 6 degrees of freedom, limit 3.143;
  - the confirmation runs in the mirrored order, B1 A1 A2 B2 A3 B3 B4 A4.

**The approved rule** is `timing_protocol.py` at `c05fd2dd…`, fixed before
any of the following sessions ran.

**On the NVCC 13.0.88 build**, the container's, with the SDK's cuBLAS
13.8.0.4 bind-mounted:
- **Calibration:** `c1` to `c4`, two in each order. The median σ is 0.69%.
  Thresholds are below 2% for 92 cases and below 5% for 139; the largest is
  19%.
- **Holdout** (`h5` primary, `h6` mirrored), declared in advance to reject
  the rule if it failed: *the stage passes.*
  - In `h5`, the 4.0 bpw `up_proj` at 145 rows (d = 4.23) and the 4.5 bpw
    `down_proj` at 16 rows (d = 3.98) exceeded `z`. That triggered the
    confirmation.
  - In `h6`, neither failed.
  - The aggregate `t` was 0.89 and −0.04.

**On BP-F2's reference arm:** the challenge showed that the NVCC 13.0.88
build shares identical SASS with the 13.4.92 build for only 16 of its 1,506
kernels, so native's port must be timed against the 13.4.92 build. The
extension is `aa8b9f16…`, with `CUDA_HOME` set to the SDK's tree; the
sessions' manifests confirm it. The same approved rule, unchanged, gave:
- **Calibration** (`c5` to `c8`, two in each order), in
  [`timing-calibration.json`](timing-calibration.json). The earlier one is
  kept in `timing.json`.
  - The median σ is 0.62%.
  - Thresholds are below 2% for 96 of the 176 cases and below 5% for 146.
  - The noisiest, synthetic 14,336 × 4,096 K4 at 1 row, reaches 18.5%.
  - Every in-sample pairing passes. Those with `c7` as the primary session
    need a confirmation.
- **Holdout** (`h7` primary, `h8` mirrored): *the stage passes in `h7`
  alone.* No case exceeded `z`, and the aggregate `t` was 0.77.
- **Power,** estimated by slowing one case at a time in both sessions of a
  pair:

  | Slowdown | `h7` + `h8` | calibration pairs |
  | --- | --- | --- |
  | 2% | 48% | 51–54% |
  | 3% | 63% | 65–68% |
  | 5% | 80% | 79–80% |
  | 10% | 90% | 87–89% |

  Most misses are the noisier cases.
- **Subset power** ([`timing_power.py`](timing_power.py), which slows a
  whole subset in both sessions):
  - *Every case slowed:* the stage fails from 0.5% on every pair.
  - *The 66 cases of 145 rows or more:* it fails from 0.5% on the holdout
    pair, through one case failing both sessions (aggregate `t` 2.56 and
    0.42), and from 1% on the calibration pairs.
  - *The 22 fused-reconstruction cases:* it fails from 1%, through
    per-case failures.
  - *Only the noisiest quarter:* it does not fail up to 3%.

  The aggregate test catches broad shifts. Narrower ones are caught case
  by case.

  `timing.json` holds these results for the holdout and two calibration
  pairs.

**GEMV gap** (`gemv`: EXL3-G median over EXL3-O). Only the 4.0 bpw
fixture's real projections use GEMV at 1 and 8 rows. The 4.5 bpw arms
launched the same kernels either way.

| 4.0 bpw projection | 1 row | 8 rows |
| --- | --- | --- |
| q_proj | 1.36 | 1.14 |
| k_proj | 1.50 | 1.55 |
| up_proj (fused gate/up) | 0.90 | 0.94 |
| down_proj | 1.26 | 1.37 |
| lm_head | 1.02 | 1.01 |

The two arms were tuned separately: EXL3-O in this session, EXL3-G reusing
`aa`'s caches. Their kernels therefore also differ wherever the two tunings
chose differently: in 21 other cases, some of them above 8 rows. Those
differences are the tuner's, not GEMV's, and they are not small:

- the 4.0 bpw `k_proj` at 16, 32 and 144 rows is 1.03, 1.05 and 1.10 on
  `aa`'s tuning;
- synthetic 4,096 × 4,096 K8 at 16–144 rows is 1.09–1.12.

**The tuning run a reference freezes therefore sets its speed as well as
its logits.** The protocol times native against the same frozen choices,
so the comparison stays like for like. Upstream's own speed, however,
varies with the tuning run by this much.

## Reproduction

The records preserve the hashes of the scripts used for each measurement.
The current scoring scripts additionally reject incomplete and nonfinite
evidence: Tier C requires all 750 statistics; timing requires every
calibrated case, four blocks per arm and 31 finite, positive samples per
raw block; and a changed frozen tuning cache stops the session. Constant
blocks with a positive aggregate timing shift
fail instead of receiving a zero statistic. The recorded timing outcomes
are unchanged; CPU regressions in `tools/tests/test_backend_proof.py` and
`tools/tests/test_timing_inputs.py` cover these rejection paths and replay
the recorded holdouts.

On a Spark, use a working directory `P0` (outside the repository, and
exported as `P0` for `exl3_run.sh`) with this directory copied to
`P0/harness`. Write the held-out IDs to `P0/heldout-ids.i64le` with the
generator above (`.tofile()` of the int64 array on a little-endian host) and
check the hash. The FP16 references need the fixture and
the llama.cpp source at `b29c606e2`.

- **Image reference.**
  1. Compile the harness inside the pinned image:
     `g++ -std=c++23 -O2 -march=armv8-a -Wall -Wextra -Wpedantic -Werror`
     against `/app`'s `libllama`, `libggml` and `libggml-base`, with the
     source's `include` and `ggml/include`.
  2. Run it in the image with the GPU, `--network none --read-only`, and
     the environment above.
- **Bridge.**
  1. Unpack `libcublas-13-4` and `libcublas-dev-13-4` 13.8.0.4 into a copy
     of the SDK's CUDA tree.
  2. Run `bridge/bridge.sh SDK LLAMA_SOURCE CUDA BUILD` and build the
     `fp16_reference` target with the SDK's Ninja.

  The SDK now carries those libraries (D-076). A rebuild can instead point
  `CUDA` at the SDK and add cuBLAS's library directory to the RUNPATH.
- **Comparisons.** Run `compare.py LEFT RIGHT` (Python with numpy).
- **EXL3.**
  1. Set `EXL_RUN` to the M0 reference directory and put its `pins.json` in
     `P0/exl3/`.
  2. Run `exl3_run.sh NAME FIXTURE TUNE_CACHE [--profile …] [--gemv off]`.
     It needs `sudo -n docker`. A new cache name tunes; a copy of a frozen
     cache (base64 in `results.json`) reproduces. The GEMV-off arms started
     from a copy of `tune-40` or `tune-45` and added the shapes GEMV had
     served.
  3. For the bridge, set `CUDA` to the tree above with the headers overlaid,
     `CUDA_MOUNT=/cuda` and a fresh `EXT_CACHE`.
  4. Compare two arms with `exl3_compare.py LEFT RIGHT` in the same
     container.
- **Second pass.**
  - Every arm adds `--hgemm-f16acc 0`, and `--capture` where layers are
    scored.
  - The cuBLAS arm sets `DOCKER_EXTRA` to bind-mount the SDK's
    `libcublas.so.13` and `libcublasLt.so.13` (the resolved files) read-only
    over PyTorch's in `nvidia/cu13/lib`.
  - The retuning arms replay copies of the recorded `gemmfresh-*` caches
    (their hashes are in `results.json`).
- **Oracle.**
  1. Run `exl3_export.py --model --pins --output` in the container with the
     GPU.
  2. Run `oracle.py --weights --ids --output` in the container without it
     (about 10 s per fixture).
  3. Score the runs with `oracle_compare.py ORACLE RUN...`.
- **FP16 plan.**
  - Run each bridge arm under `nsys profile --trace=cuda`, with
    `CUBLAS_LOGINFO_DBG`/`CUBLAS_LOGDEST_DBG` and
    `CUBLASLT_LOG_LEVEL=5`/`CUBLASLT_LOG_FILE`, then `fp16_plan.py` (its
    help lists the inputs).
  - The pool peaks come from `fp16_pool_peak.sh SDK LLAMA_SOURCE
    BRIDGE_BUILD fp16-pool-peak.patch OUT`.
- **Third pass.**
  - *Arms and faults.* Run `exl3_run.sh v3-N-NAME FIXTURE CACHE --gemv off
    --hgemm-f16acc 0 --capture [--variant VARIANT]`. The cuBLAS 13.8.0.4
    arms add `DOCKER_EXTRA` as above.
  - *Scoring.* Run `oracle_compare.py ORACLE RUN...` over each fixture's
    `v3-*` runs, then `tierc.py SCORES --thresholds 2 8 --bounds BOUNDS`.
    `tierc.json` combines each fixture's report and bounds.
  - *Memory.* Run `exl3_run.sh mem-N FIXTURE CACHE --gemv off
    --hgemm-f16acc 0 --memory`.
  - *cuBLAS plan.* Run `exl3_run.sh plan-N-LIB …` with `DOCKER_EXTRA="-e
    CUBLASLT_LOG_LEVEL=5 -e CUBLASLT_LOG_FILE=/p0/… -e CUBLAS_LOGINFO_DBG=1
    -e CUBLAS_LOGDEST_DBG=/p0/…"`, plus the bind mounts for 13.8.0.4. Then
    run `cublas_plan.py --lt LT_LOG --api API_LOG`.
  - *Pinning probe.* Build `cublaslt_pin_probe.cc` on a Spark with the SDK's
    `clang++` (the `spark-native` flags: `--target=aarch64-linux-gnu
    -march=armv8-a`, the SDK's GCC install, `-fuse-ld=bfd
    -static-libstdc++ -static-libgcc`) against the SDK's CUDA and cuBLAS,
    with an RPATH to its `lib`. Feed it the 21 GEMMs as `KIND n k m lda ldb
    ldc` lines taken from `exl3-recon-plan.json`, with
    `CUBLASLT_LOG_LEVEL=5` set to compare the resolved algorithms.
- **GGML operations.** The steps are listed in `ggml-ops.json`'s
  `reproduction`:
  - `ggml_shim/build.sh` builds the library in the container;
  - `ggml_ops_run.sh` runs `ggml_ops_capture.py` and `ggml_ops_score.py`;
  - the native-like arm is `--variant ggml_ops`, with `GGML_OPS_LIB`
    naming the library.
- **Launch record.** Run `exl3_launch_record.py --model --protocol
  ../exl3-reference/protocol.json --mode kernels|synthetic --output` in the
  container. Use EXL3-G's settings (`EXL3_GEMV=0 EXL3_HGEMM_F16ACC=0`), a
  copy of a timing session's frozen cache as `EXLLAMAV3_TUNE_CACHE`, and
  the cuBLAS 13.8.0.4 bind mounts.
- **Timing.**
  1. Run `timing_session.sh SESSION A ENV_A B ENV_B` with `EXL_RUN` and `P0`
     set, and `ORDER` (the eight-block orders above). A session can reuse
     another's frozen caches with `FROZEN_FROM`. Since `c1`, every session
     sets `DOCKER_EXTRA` to the cuBLAS 13.8.0.4 bind mounts.
  2. BP-F2's reference sessions (`c5`–`h8`) also set `CUDA` to the SDK
     tree with the extension headers (`cuda-bridge-torch`),
     `CUDA_MOUNT=/cuda` and `EXT_CACHE=build-cache-nvcc134`.
  3. Summarize a session with `timing_stats.py SESSION_DIR A B` in the
     container. `timing.json` keeps the sessions as columns;
     `timing_from_json.py TIMING_JSON SESSION OUT` rebuilds a summary from
     it.
  4. Calibrate with `timing_protocol.py --calibrate STATS…`.
  5. Apply the rule with `timing_protocol.py CALIBRATION PRIMARY
     [CONFIRMATION]`. `--power` estimates single-case detection;
     `timing_power.py CALIBRATION PRIMARY CONFIRMATION` estimates subsets.
- **SASS.** `sass_hashes.py hash BINARY` gives per-kernel SASS hashes.
  - `compare` compared the two extension builds, giving
    [`exl3-sass-compare.json`](exl3-sass-compare.json).
  - `merge` added the 13.4.92 build's hashes to the launch recordings.
  - **Caveat (found 2026-09-27).** `hash` ends a function only at the
    next `Function :` line, so the last function of each cubin also
    hashes the text `cuobjdump` prints before the next one: the `Fatbin`
    header and, in an archive, the `member` line with its path. That
    hash depends on the container, not only the SASS. In llmpalooza's
    locked build it changes three kernels' hashes between an object, its
    archive and a linked binary. None of the 14 GEMM kernels the launch
    record names is affected: the locked build reproduces all 14. The 12
    other recorded ExLlamaV3 hashes (Hadamard, reconstruction, bias add)
    are unchecked. Before a native build is judged against them, re-hash
    the reference extension with a hasher that ends each function at a
    `Fatbin` line, as `fp16_plan.py sass-hash` does.
