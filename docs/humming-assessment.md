<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Humming assessment

On 2026-10-21 the owner asked whether vLLM's newly released
[Humming](https://github.com/vllm-project/humming) quantized GEMM library
offers anything llmpalooza is not already doing. This page is a source review
of commit
[`5b500b4`](https://github.com/vllm-project/humming/tree/5b500b407a716e9dfcede04a5eba97a6eece0508)
(2026-10-21; the repository has 324 commits, the first on 2026-02-12) and of
its [launch post](https://vllm.ai/blog/2026-10-21-humming). The checkout lived
in a session scratchpad and was treated as untrusted data: nothing from it was
built or run. Every Humming speed figure below is the vendor's, from the post,
on the hardware it names; none is an llmpalooza measurement, and the
repository's checked-in `benchmarks/results/` hold no `sm_86` or `sm_121`
numbers (they cover 4090, A800, H20 and H200). File references are relative to
that commit. The plan item this review produced is listed
[at the end](#what-the-plan-takes).

## What it is

- **Scope:** a JIT-compiled GEMM library for quantized inference: dense and
  MoE (`indexed`, `grouped_contiguous` and `grouped_masked` layouts,
  `humming/config/enum.py`), weights of any width up to 8 bits (INT1–8 with
  or without zero point, signed FP up to 8 bits including MXFP4, NVFP4, FP6
  and MXFP8) against FP16, BF16, FP8, FP4 or INT8/INT4 activations (README
  support matrix). Scale types are BF16, FP16, E4M3, E5M2 and UE8M0.
- **Hardware:** SM75 and later. It has separate heuristics per device class
  (`humming/tune/sm8x.py` including an `Sm86Heuristics`, `sm90.py`, `sm100.py`
  with a tcgen05 (UMMA) path, `sm120.py`, `sm121.py`). SM120 and SM121 run
  block-scaled `mma.sync` (the `MXMMA` type), not tcgen05.
- **Structure:** `humming/include/humming/**.cuh` holds the device code (about
  8–9k lines): one `humming<…>` kernel template
  (`kernel/humming.cuh`) over `LayerConfig`, `ComputeConfig` and
  `TuningConfig` types, plus separate kernels for weight packing and
  quantization, activation processing (`kernel/process_input*.cuh`) and MXFP4
  preparation. Python (`humming/`) renders a Jinja source file per
  configuration (`kernel/humming.py`, `CODE_TEMPLATE`), compiles it with
  NVRTC or NVCC (`jit/compiler.py`), post-processes the cubin, and launches
  through a C++ launcher (`csrc/launcher/`) behind PyTorch. A per-device
  Python heuristic picks the configuration for each M.
- **License:** Apache-2.0 (`LICENSE`, `pyproject.toml`).
- **Vendor claims** (post; theirs, on the named hardware; no GB10 or `sm_86`
  figure except the SM121 pair):

  | Claim | Hardware and baseline |
  | --- | --- |
  | W4A16 dense, small batch: roughly 3–4× | several devices, against cuBLAS (BF16) |
  | W4A16 dense, large batch: about 2× / above 3× | H200 / B300, against vLLM Marlin |
  | MoE W4A16 MXFP4: roughly 2× at medium to large batch | H20 and H200, against Marlin |
  | W8A8 blockwise FP8: about 2.6× / 2.5× | H20 / H200, against DeepGEMM Contiguous |
  | W4A8-MXFP4: about 4.2× / 4.5× | H20 / H200, against the DeepGEMM W8A8 baseline |
  | Blackwell MX W8A8 / W4A8 / W4A4: about 1.25× / 1.6× / 1.4× | B300 (SM103), against DeepGEMM Contiguous |
  | **SM121: W4A8 about 19% and W4A4 about 9% faster, small batch** | against FlashInfer's CUTLASS, not our 79d kernel |
  | DeepSeek-V4.1-Flash end to end: +14% to +43% output throughput, 25–32% lower mean TTFT | 8× H20, TP=8, Marlin W4A16 to Humming W4A8 with tokenwise FP8 activations |
  | One W4A16 kernel compiles in 1.82 s (NVCC) / 1.88 s (NVRTC) against 23.35 s for CUTLASS | H200, CUDA 13.4, caches off |

  None of these is a like-for-like comparison with llmpalooza's kernels, and
  the post's SM121 sentence gives no shapes.

## Fit as a build-time kernel source

Under D-053 (kernels are build-time implementations selected per operation,
from any compatible source), D-028 (no runtime plugin ABI) and D-010 (no
interpreter in the hot path) the useful reading is: take the device headers,
instantiate their templates with NVCC at build time, and leave the Python
JIT, NVRTC and the launcher behind.

**What makes it plausible:**
- **Self-contained headers.** Outside `humming/`, the device headers include
  only `cuda.h` (for `CUtensorMap`), `cuda_fp16/bf16/fp8.h`,
  `cooperative_groups.h`, `cuda_awbarrier_primitives.h` and `<cstdint>`. No
  PyTorch, no CUTLASS, no CCCL beyond the standard-type shims its NVRTC path
  injects (`NVRTCCompiler._STD_HEADER_SHIMS`: `std::is_same`,
  `conditional_t`, `enable_if`). `cuda.h` is the driver API we already use
  for VMM.
- **NVCC is a supported route.** `NVCCCompiler` in `jit/compiler.py` builds a
  cubin with `-std=c++17`, `--use_fast_math`, `--expt-relaxed-constexpr`,
  `--expt-extended-lambda` and `--ptxas-options=--register-usage-level=10`.
  Our CUDA translation units must be checked against those flags (language
  standard, fast math scope); that is part of the screen, not assumed.
- **Plain C++ kernel arguments.** The kernel takes tensor pointers or
  `CUtensorMap`s, routing arrays (`sorted_ids`, `expert_ids`,
  `num_tokens_padded`, `expert_layout`), a stream-K `locks` buffer and a
  `tensor_map_buffer` for device-built maps. All of it is workspace we would
  budget in the catalog (D-006, D-007) and launch on our own streams.

**What we would have to build** (this is the cost, and why an import of the
whole library is not proposed):
- **The configuration structs.** `LayerConfig`, `ComputeConfig`,
  `TuningConfig`, `MmaOpClass` and the shape types are rendered from Python
  dataclasses (`to_cpp_str`, `to_macro_cpp_str`, `CODE_TEMPLATE`). A build
  step in `tools/` (build-time Python is allowed, D-010) would emit one
  instantiation per screened winner; nothing is generated at run time.
- **Selection as data.** The per-M heuristics (`humming/tune/*.py`) are
  Python code. Our selection is checked-in rank data per device class
  (U6; owner, 2026-10-20), not autotuning, so only the candidate
  configurations the screen finds winning are instantiated and ranked. The
  vendor's warm-up of all supported M sizes for one layer (2–4 s with JIT)
  shows the instance count that would otherwise follow.
- **A launcher and the TMA maps.** Our own launch code (stream-K locks,
  tensor-map buffer, dynamic shared memory). Humming estimates shared memory
  on the host in Python (`utils/smem.py`, checked against the compiled
  kernel by `assert_smem_size_matches_estimate`); in C++ it would come from
  the `SharedStorage` type, to confirm.
- **The weight layout.** Humming's kernels read a kernel-defined layout
  produced by `transform_humming_tensors` (`transform.py`, with
  `kernel/pack_weight.cuh` and `quant_weight.cuh`). Under rule 11 that is an
  import-time relayout by tensor role and encoding, recorded as a layout in
  the artifact (D-108, D-113) and decodable by the CPU references. It
  competes with our NVFP4 scale-atom relayout rather than reading the split
  checkpoint tensors directly; whether the scale and tensor-scale tensors
  (`weight_scale_2` per expert) can stay in the checkpoint's form was not
  traced and is a screen question.
- **The cubin patch step**, if hidden formats are used
  ([below](#hidden-tensor-core-formats-a-research-lead)): a post-compile,
  pre-embed pass over the cubin. `csrc/patch_cubin.cpp` is a self-contained
  466-line C++ file with no PyTorch dependency, but our fatbin assembly
  would need the step inserted.

**Instance count and build time.** One NVCC compile of one W4A16 kernel is
1.82 s by the vendor's measure (H200); encodings × shape classes × M
buckets × `sm_121a` and `sm_86` multiplies it, and cubin size grows with it.
Only measured winners are instantiated.

**License tier.** Apache-2.0 is permissive, so it may enter the core without
a decision of its own (D-091), as incorporated code under D-017, with the
usual provenance record. Two caveats, neither a blocker:
- The headers carry no SPDX or copyright lines, and the README names DeepGEMM
  (MIT), Marlin (Apache-2.0), lmdeploy (Apache-2.0) and CUTLASS (BSD-3-Clause)
  as inspirations without saying which files adapt which. Before any file is
  copied into `src/`, check it for adapted third-party code and carry the
  upstream tag beside ours where it is a port (AGENTS.md rule 12). Writing our
  own kernels from the technique carries only Apache-2.0.
- The repository ships no `NOTICE`; our `NOTICE` entry follows the
  provenance check.

## Candidate consumers

Where Humming could fill a gap or beat what we ship, with the baseline it
meets. All of these are screen questions (D-085); none is a claim.

| Consumer | Humming's path | Our baseline today | What the screen compares |
| --- | --- | --- | --- |
| **NVFP4 W4A4 routed-expert GEMM** (Qwen3.8 prefill; `moe_cutlass.cu`) | `grouped_contiguous`, `a_dtype` FP4 with `dynamic_group_token` (group 16, E4M3 scales, a secondary token scale), `MXMMA` on `sm_121` (`config/config.py`, `mxmma_supported`) | CUTLASS 4.7.1 example-79d grouped GEMM, tile 128×128×256 ping-pong, tile chosen by a quick A/B | Same routed shapes and operands, BF16 output; the vendor's "+9% over FlashInfer CUTLASS" is a different baseline and says nothing about 79d |
| **MXFP4 and MXFP8 relaid expert products** (owed; [engine-components.md](engine-components.md): relaid MXFP4/MXFP8 tables have no kernel yet) | MXFP4 (E8M0 per 32) as W4A4, W4A8 or W4A16; MXFP8 weights; `kernel/process_mxfp4.cuh` and `schema/mxfp4.py`, `schema/modelopt.py` (NVFP4, MXFP8) | None for the relaid tables; MXFP4 tensors in GGUF (DSpark's experts) run through GGML's MXFP4 kernels | A product that reads the relaid atoms, at DSpark's decode and verify row counts, against the GGML path |
| **Split-scale NVFP4 items** (plan.md M3.5 carried list: a grouped vector product for decode rows and the block-scaled grouped GEMM over split scales) | Humming transforms scales into its own layout, so it answers the relaid-atom products, not "over split scales" directly; its W4A16 grouped path (FP16/BF16 activations) is the decode-row comparison | `llmp.decode.grouped.nvfp4.scale_atoms` (resident W4A16 grouped product) and GGML's NVFP4 vector kernel | One W4A16 grouped decode comparison (1 to 16 rows). Decode is bandwidth-bound on GB10, so parity is the likely result; the screen confirms it against the measured roofline |
| **`sm_86` encoding coverage** (U7d; rule 11) | W×A16 for INT1–8 (symmetric or zero point) and signed FP up to 8 bits on `Sm86Heuristics`; plus INT8-activation INT4 through patched `ldmatrix` (below) | A dequantize-to-F16 plus cuBLAS rung is U7d's plan for types without a tile kernel; IQ1_M and Q1_0 have no device path | Whether Humming instances compile under our toolchain for `sm_86` and run correct against the CPU references; speed only against the U7d rung. GB10 evidence does not carry to `sm_86` (owner, 2026-10-20), so the 3080 Ti is measured on its own |

**Reading the claims.** The 3–5× figures are against BF16 cuBLAS or Marlin on
datacenter parts; our decode products already read compressed weights, so
the gap to close is smaller. Operator wins have also repeatedly failed to
survive in the model here (T79's paired shared-expert GLU: 28.6% at the
operator, −0.09% in the model; Qwen NVFP4 rows8/warps4: +11.4% isolated,
neutral at the model, [optimization-inventory.md](optimization-inventory.md)),
so any operator win is followed by one model-level A/B before adoption.

**Quantization formats beyond ours.** GGML Q4_0/Q4_1 map onto INT4 with a
zero point and F16 scales per 32; the K-quants and I-quants (two-level scales,
codebooks) do not, and widening them to Humming's scale forms would break the
artifact's source encoding (D-108: bulk weights never widen, small per-channel
folds are fine). So Humming does not extend GGUF coverage; its reach is
ModelOpt-style, compressed-tensors, GPTQ/AWQ-style and MX/NVFP formats.

## Hidden tensor-core formats: a research lead

`csrc/patch_cubin.cpp` rewrites SASS encoding bits in a compiled cubin to
reach tensor-core formats PTX does not expose:

| Format | Meaning | Where |
| --- | --- | --- |
| **E0M3** | FP4 sign-magnitude integer {0, ±1…±7} | block-scaled `OMMA` (opcode `0x47f`, A format bit 78, B format bit 79), narrow-float convert `F2FP` (opcode `0x23e`) |
| **E3M4** | FP8, bias 3, 4 mantissa bits, ±30 | `QMMA` (`0x47a`/`0x27a`; A type bits {78,82,83}, B {79,84,85}), `F2FP` |

- **How it is applied.** The kernel is compiled with the nearest PTX type
  (`E2M1` for E0M3, `E5M2` for E3M4; `config/mma.py`, `UNDOCUMENTED_FP_DTYPES`),
  then the patcher flips the format bits of instructions that still carry the
  base format, and skips anything else. It refuses an E0M3 `OMMA` with
  `scale_vec::2X` because that encoding faults at launch. On `sm_10x` the
  same formats are selected through the tcgen05 instruction descriptor, with
  no patch (`utils/ptx/tcgen05.cuh`).
- **Evidence it gives for GB10.** The file's header says the constants were
  found by bit-flipping with on-GPU numeric regression on an RTX PRO 6000
  (`sm_120a`, CUDA 13.3) and "validated on GB10 (`sm_121a`, CUDA 13.3)".
  `UNDOCUMENTED_FP_SM_VERSIONS` lists 100, 103, 120 and 121; a comment notes
  `sm_107` raises an illegal instruction, and the post says Rubin lacks both
  formats. All of that is the vendor's account; we have run nothing.
- **What it buys.** A symmetric INT weight of up to 4 bits is exactly an
  E0M3 value, so the FP4 block-scaled tensor-core path can multiply INT4
  weights by FP4 activations without a software dequantization step (the
  "zero-exponent float" reinterpretation: `dequant_weight.cuh` and
  `quant_weight.cuh` describe `e0mX` as sign-magnitude fixed point; for
  INT8 activations `transform.py` converts such weights back to offset
  integers instead). E3M4's extra mantissa bit lets symmetric INT5–6 weights
  meet FP8 activations natively (`get_mma_b_dtype`, `config/mma.py`).

**Caveats, which make this a lead and not a plan:**
- **Undocumented hardware behaviour.** Nothing guarantees that a future
  driver, `ptxas` or silicon revision keeps these encodings. Any use needs
  a build-time self-test that executes the patched instruction against a CPU
  reference on the target device and refuses to select the instance when it
  disagrees, plus a fallback implementation in the registry (D-053).
- **Not portable across NVIDIA parts.** Rubin lacks it, and `sm_107`
  faults, so a plan must key the implementation by device capability.
- **Range.** Sign-magnitude has no −8. GGML's Q4_0 (values −8..7), and
  two's-complement symmetric INT4 from GPTQ-style tools, need the −8 code
  clamped to −7 or the weights re-quantized: a quality change.
- **Scales.** E0M3 requires a scale group of 16 with E8M0 or E4M3 scales and
  `scale_vec::4X` (`mxmma_supported` refuses `float4e0m3` with group 32;
  `tcgen05.cuh` asserts block 16). Q4_0's F16 scale per 32 would have to be
  converted: a lossy conversion where the E4M3 scale has three mantissa bits.
- **Activations.** The FP4 route quantizes activations to E2M1 (W4A4), which
  is a quality trade. Under AGENTS.md rule 11 it is a per-alias mode, off by
  default, unless a same-quality check passes; the vendor's post gives no
  quality figure for it.
- **`ldmatrix` patching is a second SASS edit.** For INT4 weights against
  INT8 activations (and INT2 against INT4) Humming patches `ldmatrix`
  (opcode `0x83b`) to widen S4→S8 or S2→S4 as the tile loads, on any `sm_75`
  and later when patching is enabled (`config/config.py`,
  `use_ldmatrix_dequant`); CUDA 13.4 on `sm_90`+ makes it native for INT4.
  This could matter for the 3080 Ti's INT8 tensor cores, but INT8
  activations are again a per-alias quality trade, and an `sm_86` instance
  would need its own self-test.

**Proposed handling.** First: the ordinary Humming paths (documented PTX
types only; `HUMMING_DISABLE_PATCH_CUBIN=1` turns patching off) in the
[screen](#what-the-plan-takes). Hidden formats only if that screen shows a
pattern worth chasing, then as a separate screen with the self-test and
fallback above; never as a default.

## M4 leads: FP8 activations for MoE and quantized all-to-all

- **FP8 activations for MoE experts.** The post's DeepSeek-V4.1-Flash run
  moves MXFP4-weight experts from Marlin W4A16 to Humming W4A8 with tokenwise
  FP8 activations and keeps the weights unchanged (+14% to +43% output
  throughput on 8× H20, vendor's figure). V4.1-Flash is M4's checkpoint, and
  an MXFP4-expert product with an FP8 activation operand is the same shape
  of product the owed MXFP4 item needs. Under rule 11 it is an opt-in
  per-alias activation mode (the activation precision changes numerics),
  qualified against F16 per model. Measure on GB10 only after the MXFP4
  screen; the vendor's gain comes from H20's compute shape and may not carry.
- **Quantized all-to-all.** The post lists quantized EP all-to-all as future
  work, with no implementation or numbers, so there is nothing to adopt.
  The question belongs to M4's EP-versus-TP decision
  ([Flare item 6](flare-assessment.md#6-two-node-transport-and-the-ep-vs-tp-decision-m4-entry)):
  across two Sparks an FP8 dispatch halves the dispatch bytes, but at
  decode-sized messages the collective is latency-bound (NCCL's 30.6 µs
  median all-reduce at 32 B against 2.34 µs RDMA write latency at 1 KiB,
  `experiments/interconnect/README.md`), so bytes may not matter there. Record it as a lead for
  that measurement.

## What we already do

So nobody repeats it:

| Humming does | llmpalooza already has |
| --- | --- |
| Separate `LayerConfig` / `ComputeConfig` / `TuningConfig` and per-M tuning | Per-implementation declarations and plan keys; U6's checked-in rank data per device class and shape class (owner, 2026-10-20) is the same split without run-time heuristics |
| A per-device heuristic that picks block, warp and stage shapes | A quick A/B at the model's shapes chosen the 79d tile (128×128×256 ping-pong); rows/warps schedules are screened per shape class (optimization inventory, T38, NVFP4 scheduling) |
| `indexed` and `grouped_contiguous` MoE layouts | Device-resident per-expert offsets written by a setup kernel before each grouped GEMM (`moe_cutlass.cu`), so nothing waits on the host |
| Separate activation processing kernel launched with PDL (`kernel/process_input.cuh`, `griddepcontrol`) | Fused and prepared input quantization (`ir::PreparedRows`, `mxfp8_inputs`, MXFP8 rows in CUTLASS's scale layout); PDL on the decode products |
| `triton.testing.do_bench` timing | Our comparisons rotate the actual routed working set, so L2 is cold (NVFP4 scheduling experiment); Triton's `do_bench` flushes L2 between repetitions by default (not read here) |
| FP4/FP8 block-scaled tensor-core products on `sm_120` family | CUTLASS 4.7.1 NVFP4 W4A4 grouped GEMM and MXFP8 products on `sm_121a`; W4A4 for MTP blocks measured 1.2–2.0% faster decode at equal acceptance ([engine-components.md](engine-components.md)) |
| Scale-layout relayout at weight preparation | Import-time relayouts by role and encoding recorded in the artifact (D-108, D-113): NVFP4 scale atoms done, MXFP4/MXFP8 owed |

## Not adopted

| Item | Why |
| --- | --- |
| The Python library, JIT, NVRTC and PyTorch launcher | No interpreter or runtime JIT in the serving path (D-010, D-028); run-time compilation also adds start-up work and a toolchain dependency on the target |
| `HummingLayer`, the schemas (`schema/*.py`) and `forward.py` | Framework glue; our importer and mapping tables already own checkpoint interpretation (D-109) |
| The tcgen05 (UMMA) and wgmma paths | `sm_100`/`sm_90` only; the GB10 has neither tcgen05 nor wgmma, and the 3080 Ti is `sm_86` |
| T-Head PPU and Windows support | Not our targets (D-082, D-110) |
| Humming's autotune (future work in the post) | Selection is checked-in rank data, not autotuning |

## What the plan takes

| Item | Shared home | Plan placement |
| --- | --- | --- |
| Humming as a build-time kernel source: coarse screen of NVFP4 W4A4 grouped GEMM against CUTLASS 79d, MXFP4 expert products for DSpark, one W4A16 grouped decode comparison | Kernel registry (D-053), importer relayouts | M3.5 (screen; the owed MXFP4/MXFP8 products and the split-scale items take the winner) |
| `sm_86` encoding coverage through Humming instances | Format-coverage rung | U7d, only after the screen shows Humming builds with our toolchain |
| Hidden E0M3/E3M4 formats | Kernel registry, per-alias modes | Research lead; separate screen only if the first screen finds a pattern worth chasing |
| FP8 MoE activations and quantized all-to-all | MoE component, collective provider | M4 leads (DeepSeek-V4.1-Flash), after the MXFP4 screen |

## Open questions

- Do `LayerConfig` structs and the kernel template compile under our NVCC
  (version, C++ standard, fast-math scope, `-fno-exceptions`) and fit our
  fatbin flow, and what is the instance count per winner?
- Does the Humming weight and scale layout accept the NVFP4 split tensors
  (`weight`, `weight_scale`, per-expert `weight_scale_2`) without the importer
  changing their values, or does it force a relayout the 79d kernel
  does not?
- Are any headers adapted from third-party code (file-level provenance)?
- Is the SM121 sentence in the post measured with `sm_121a` kernels on a
  GB10 or on SM120 hardware, and at which shapes? The repository's own
  benchmarks do not say.
