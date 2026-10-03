<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Plan

**This is a living document.** Milestones will be re-scoped, re-ordered, split,
or added as planning conversations and findings come in. That churn is
expected; what is *not* allowed is silent change. Update scope and progress
here as work lands. Only consequential choices that change a load-bearing
constraint or are expensive to reverse get a decision-log entry; routine
scope, ordering, and implementation changes do not (AGENTS.md rule 1).

Check a box only when the item is done and verified; partially done items stay
unchecked, optionally with a note. When a milestone exits, its section shrinks
to a short summary and its task-by-task history moves to a record beside this
file ([M0](m0-record.md), [M1](m1-record.md), [M2](m2-record.md)).

**Status legend:** `pending` · `in progress` · `done` · `parked`

## M0 — Plan the plan  `done`

Ran 2026-09-20 to 2026-09-23 and exited on the owner's approval of the plan.
M0 turned the [design brief](ideation.md) into the vision, the triaged feature
matrix, the approved architecture and decisions D-001–D-069, and gathered the
evidence they rest on: toolchain, VMM, I/O and interconnect spikes on the
Sparks; the llama.cpp, EXL3, image and MiMo reference runs; the measured A→B→A
reference cycle; the paging-feasibility study; and the v0 artifact layout
study. The [M0 record](m0-record.md) keeps each task's outcome, evidence links
and caveats. Everything M0 left open is owned by the milestone exits below.

## Milestone ladder

Rewritten from the provisional ladder at the end of M0 (2026-09-23; see the
[M0 record](m0-record.md)), then re-sequenced after M2 by the owner on
2026-09-27 (D-087): the fast full model swap, the owner's first real work,
comes first, on one Spark at M3 and on two at M4, and everything after moved
back two places (the old M3 is M5, M4 is M6, M4a is M6a, and so on to M8,
now M10). After the swap work the order is by risk: one resident model end
to end, with the importer and the full front door, at M5; the first useful
product at M6 (A→B→A with retention); configured placement at M6a;
demand-paged MoE with the first daily-driver models at M7; sharding under
pressure and failure at M8; performance and the remaining decoding modes at
M9; and the remaining product scope with the first tagged release at M10.
Each milestone leaves a usable, testable result. None has a promised date.

| Milestone | Result | Needs |
| --- | --- | --- |
| M1 Bootstrap | Pinned SDK, builds, local check gate, package skeleton, confined-job proof | M0 (done) |
| M2 Resource core | Catalog, admission and leases on a fake backend and a Spark; the backend proof settles the operation contract | M1 (done) |
| M3 Single-Spark fast swap | DeepSeek V4 Flash, Qwen3.8 Flash Next and Qwen-Image-2.1 swap A→B→A on one Spark, aiming at ~10 s to first token, as correct, fast and lean as their references | M2 |
| M3.5 Model families | The core engine runs the major open model families, MoE and dense (Gemma, Llama, MiMo and the other top-tier families), each correct, as fast as its reference and flat with context, before the system is built around it; image, video and audio file inputs on the models that take them, decision models over the Jev API, and image, video and speech generation routes (D-101) | M3 |
| M4 Two-Spark fast swap | GLM-5.3 Flash, then DeepSeek v4.1 Flash, sharded over both Sparks in the same cycle, with their image (and GLM's video) inputs | M3.5 |
| M5 One resident model | Importer, verifier, the three client protocols plus the Gemini API and code completion, TLS and management on the small fixtures | M4 |
| M6 First useful product | A→B→A with partial retention; switching policy chosen from measurement | M5 |
| M6a Configured placement | Conductor, enrolled nodes, whole-model placement and routing | M6 |
| M7 Demand-paged MoE | Exact expert paging; Gemma 4 and Ornith as daily drivers with reasoning and constrained output (their resident bring-up is M3.5's) | M6 |
| M8 Sharding under pressure | Sharded execution correct under asymmetric pressure, cancellation and failure, with coordinated admission | M6a, M7 |
| M9 Performance | D-036's benefit target on a library larger than memory; the remaining speculative and diffusion decoding | M7; M8 before exit |
| M10 Product and release | Dashboard, remaining API scope, signed apt repository, first tagged 0.x release | M9, for the release |

How to read the milestones below:

- **Exit criteria are the contract.** Scope lists say what a milestone
  builds; its entry work refines them into tasks. An exit criterion changes
  here, visibly and with the owner, never by drift.
- **Detail lives in the linked designs.** Where a gate cites a matrix or an
  acceptance section, every row that section assigns to the milestone is
  part of the gate.
- **Evidence rules apply to every gate.** Thresholds are declared before the
  measurement they judge, and inconclusive comparisons do not pass (D-036).
  Every result names the check tiers and hosts that produced it (D-061,
  [workflow.md](workflow.md)). No performance gate is met by giving up
  correctness.
- **Heavy path.** Work in a [blast-radius area](workflow.md#blast-radius-changes-get-the-heavy-path-by-default)
  passes its adversarial challenge before the milestone exits.
- **Support is earned per checkpoint.** From M3 the
  [support matrix](model-support.md) records
  what each exit validated; a milestone exit names its configurations.
- **Release.** The first tagged 0.x release (an owner-signed tag, D-062) is
  cut at M10's exit, after every milestone has exited (owner, 2026-09-23). The
  repository is already public (D-061); until then it carries only `-dev`
  builds.

## M1 — Bootstrap  `done`

Ran 2026-09-23 to 2026-09-24 and exited on the owner's word. M1 built the
pinned SDK and its reference container (D-070), the CMake presets and build
tasks, the source lock with GoogleTest and toml++ (D-057, D-073), the local
check gate (`check`, `check:full`, `check:spark`; D-061), license and
provenance records (D-071), versioning (D-062), `jitllm doctor` (D-072), the
node configuration and storage-role checks (D-073), and the arm64 package
with `jitllm-runtime`, `jitllm.service`, crash handling and confined jobs in
delegated cgroups (D-074), validated on `spark`. The
[M1 record](m1-record.md) keeps each item's outcome, verification and
hand-offs. What it handed on: the GPU, VMM, I/O and ARM stress suites in
`check:spark`, a system-call filter with io_uring, and a single reaper for jobs
started from other threads (M2; the last two moved to M5); front-door, TLS and switching-policy keys
(M5, M6); the importer on the confined-job mechanism and spill file names
(M5); cluster-member checks of credential files and enrollment records
(M6a); drain-before-restart upgrades and the apt repository (M10).
Milestone numbers here follow D-087's renumbering.

## M2 — Resource core and backend proof  `done`

Ran 2026-09-24 to 2026-09-27 and exited on the owner's word. M2 built the
resource core: the catalog, commitment ledger, LRU victim baseline and
materialization planning (D-006, D-007), admission with D-069's pauses
(D-050), D-048's task lanes and scheduler loop, and the providers with
their fakes, CUDA VMM and io_uring (D-026). Its backend proof (P0–P6) ran
GGML's and ExLlamaV3's kernels under jitLLM's dispatch (D-077, D-080),
selected by plan (D-053), with cuBLAS on a jitLLM handle (D-076): native
Qwen2.5-0.5B FP16 matches the bridge bit for bit and both EXL3 fixtures
match their approved record and bounds, paged at disk speed through a
host-VMM landing zone into device VMM (D-081), evicted, written back,
restored and relocated bit-identically, FP16 and EXL3 alternating on one
node. D-086 records the operation contract; D-033 is retained; D-068's
shapes are expressible; the `native` build also targets discrete `sm_86`
GPUs (D-082); and D-085 made performance and memory coarse end-to-end
checks, so BP-F2 did not run. The [M2 record](m2-record.md) keeps each
item's outcome, evidence and caveats, and the gate. What it handed on,
in D-087's numbering: BP-F4's per-token host cost, the D-033 handoff of an
evicted model's backing, and ReconGemm's descriptors prepared once per
bound GEMM (the swap work, M3 and M4); end-to-end parity of each engine
with its reference on the small fixtures (BP-F3), the importer,
suballocation with state blocks, D-050's moved rows (suballocation holes,
stalled-client termination, every queue full at once, capacity-loss
injection), and M1's io_uring system-call filter and single job reaper
(M5); victim selection on a miss, spill as retention with D-055's spill
format, the cohort pause with several paused peers, fork and
copy-on-write, cached-state promotion and fast-swap validation on the
discrete GPU (M6); the runtime closure-excess check (M7); repeated
speculation with prefetch (M9); and, with no milestone yet, `spark-native`
sanitizer presets and the EXL3 memory outside the catalog in the paged
harness.

## M3 — Single-Spark fast full swap  `in progress`

Goal: one user swaps among three large models on one Spark, A→B→A, and each
swap reaches its first token in about 10 s: as close to that as we can
get, and at most about 20 s at exit. A's conversation state comes back
without a re-prefill, and each model is as correct, as fast and as lean as
its reference. A full swap: the outgoing model leaves, and no expert is
demand-paged. The owner's first real work after M2 (D-087).

**Entry:** M2 exit. Before a model's first native evaluation, its
checkpoint, reference engines and their configurations are pinned and their
licenses recorded ([licensing.md](licensing.md); a model's weight license
is recorded for information and gates nothing, D-087), and its prompt set,
perplexity text, or image prompts and seeds, are fixed with the bounds
below. The provenance of llama.cpp's generated Unicode tables is cleared
under D-017 before the native tokenizer is adopted
([first-slice.md](first-slice.md)): traced to UCD 15.1.0 on 2026-09-28;
jitLLM generates its own tables, and the owner accepted D-088 on
2026-09-28.

**Models, in this order.** Each runs its reference's quantization.

1. **DeepSeek V4 Flash 0731**, GGUF UD-Q2_K_XL
   (`unsloth/DeepSeek-V4-Flash-0731-GGUF@fbbb5b93`, MIT; 96,832,508,352 B
   in 3 shards). It replaces the `e3aa0d6a` revision on the Sparks, which
   the M0 baselines and artifact-format.md's example used: same shard sizes,
   new hashes, a download of about 97 GB per node. 0731 has no usable MTP
   (llama.cpp PR #25784), so its drafter is DSpark.
2. **Qwen3.8 Flash Next** as Mia's single-Spark build: NVFP4 routed
   experts, MXFP8 attention and shared expert
   (`Mia-AiLab/Qwen3.8-Flash-Next-NVFP4@925d7be6`, a mirror of
   `local-inference-lab`'s; 105,935,742,983 B, 26.8 GiB of it the packed
   PLE table).
3. **Qwen-Image-2.1** in BF16, like diffusers: text encoder (Qwen3-VL-8B),
   single-stream DiT and VAE (Qwen Research License, non-commercial,
   recorded for information). GGUF quantizations may follow later if
   memory matters (owner, 2026-09-28).

**Oracles and comparators.** Each model's correctness is judged only
against its same-format oracle. A cross-quantization comparison reports
speed and memory only, never correctness, and is labeled as such wherever
it appears.

| Model | Correctness oracle (same format) | Performance comparators | Cross-quantization (speed and memory only) |
| --- | --- | --- | --- |
| DeepSeek V4 Flash 0731 | llama.cpp on the same GGUF | llama.cpp | vLLM or SGLang where they support it, unless on the same GGUF |
| Qwen3.8 Flash Next | Mia's vLLM on the same NVFP4 checkpoint, in the recipe's deterministic mode (its default launch is not repeatable) | Mia's vLLM; TensorFold 0.6.2 on the same NVFP4 checkpoint (different activation/draft arithmetic, quality unqualified) | Earlier TensorFold (MLX 4-bit); llama.cpp on a GGUF |
| Qwen-Image-2.1 | diffusers, BF16 | diffusers, BF16 | stable-diffusion.cpp's GGUFs, also for image quality |

**Scope:**

- [x] **Provenance and licenses** (D-017, D-080): pin and audit
      `MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark` and decide whether
      its AGPL scripts may be run as a baseline recipe; record each
      checkpoint's pins and license hashes, the 0731 GGUF's among them,
      and the Qwen NVFP4 card's Apache-2.0 beside the base model's Qwen
      Community License 1.0 (weight licenses are recorded for information
      and gate nothing, D-087); pin TensorFold and its checkpoint.
      *Done 2026-09-28* ([licensing.md](licensing.md#fast-swap-models-and-baselines-m3-m4),
      [pins](experiments/fast-swap/pins.json)): the recipe pinned at
      `b8439110` with all 70 tracked files classified, and run unmodified
      as a baseline (owner, 2026-09-28); TensorFold at `beddbb7b` and its
      checkpoint cleared (the pin moved to `71377a53`, 0.3.6.2, the same
      day, with the same MIT terms); the four checkpoints, the vLLM images
      and the llama.cpp pin recorded; the vLLM, FlashInfer and CUTLASS parts of
      Mia's NVFP4 and MXFP8 path identified with their licenses, a
      CuTe-DSL kernel's runtime among them under NVIDIA's proprietary
      terms. The checkpoints are on both Sparks' NVMe
      ([environment.md](environment.md#m3-model-store-2026-09-28)).
      The default vLLM image's commit (`8e685d198`) and the kernels its
      engine log selects were read in the baselines item.
- [x] **Baselines,** installed and run on the Sparks by us: MiaAI's
      configurations, TensorFold, llama.cpp for the GGUF, and vLLM or SGLang
      where they support these models; for the image, diffusers in BF16 as
      the speed and format reference (the fastest measured), and
      stable-diffusion.cpp's GGUFs as an additional quality and format
      comparison. Each reference's load or swap, prefill, decode
      and peak memory are measured on the same prompts as jitLLM's. Mia's
      Qwen3.8 cold start (creator-reported 10 min 51 s to `/health`) runs
      once, stated as needed under D-085, and is recorded as a measurement;
      its prefill and decode are then measured on the same warm server.
      *Measured 2026-09-28 on `spark`*
      ([baselines](experiments/fast-swap/baselines.md), on the fixed
      [prompt set](experiments/fast-swap/prompts.json)): llama.cpp on
      DeepSeek 0731 (load 92–104 s, prefill ~350 tok/s at 8K, decode 19.9
      tok/s, 30.8–31.9 with DSpark, peak 93–105 GiB) and one llama.cpp swap
      cycle with Qwen3.8's GGUF; Mia's vLLM (cold start 13 min 11 s, prefill
      2,066 tok/s at 8K, decode 37.9 tok/s with MTP 3 and 25.1–25.3 without,
      peak ~103 GiB); TensorFold and llama.cpp on Qwen3.8 as
      cross-quantization comparators; diffusers on Qwen-Image (1.26 s per
      step, 52.6 s per 1024², 40-step generation). Greedy and top-5 logprob
      references for both LLMs are saved beside the report, and the
      reference image on both Sparks by hash. Mia's default launch is not repeatable under greedy decoding,
      so Qwen3.8's oracle is the recipe's deterministic mode (MTP off,
      `VLLM_QSA_DET_TOPK=1`, `VLLM_MOE_DET_FINALIZE=1`), which took a second
      launch (10 min 52 s). vLLM or SGLang on DeepSeek 0731 is not
      available on one Spark (no GGUF path for this quantization; the
      native checkpoint does not fit). The additional
      [stable-diffusion.cpp GGUF comparison](experiments/image-gguf/README.md)
      is measured on `spark`: Q4_K_M, Q8_0 and BF16 controls, including
      1024²/40-step time, sampled memory and visual agreement. It is a
      format comparison, not native jitLLM GGUF image support.
- [x] **Import:** M0's Python prototype importer writes the D-056 artifacts
      for the three models, including NVFP4 and MXFP8 tensors and the image
      pipeline's BF16 components. The C++ importer and verifier stay in M5.
      *DeepSeek V4 Flash 0731:* `import_m3.py` (the pinned prototype, its
      sources checked against the M3 pins) wrote artifact `8a355bfb…`,
      96.84 GB, 11,053 groups, the plan of the D-056 worked example, in
      11 min 16 s on `spark-b`
      ([dsv4-native](experiments/dsv4-native/README.md)).
      *Qwen3.8 Flash Next (Mia's NVFP4):* `import_m3.py` with
      `modelopt_qwen38.py`, which repacks losslessly as it writes (experts
      into GGML NVFP4 blocks, the n-gram table into 90-byte rows, linear
      attention's value heads into tiled order; MXFP8 kept as is), wrote
      artifact `67617f87…`, 103.9 GB, 24,627 groups, in 9 min 16 s on
      `spark-b` ([qwen38-native](experiments/qwen38-native/README.md),
      [artifact-format.md](artifact-format.md#qwen38-flash-next-modelopt-nvfp4-and-mxfp8));
      since the prefill work the experts go into the CUTLASS layout the
      grouped GEMM reads, so nothing is rewritten at load or on a swap:
      artifact `c4fb47a9…`, 103.8 GB, in 12 min 49 s with its verification.
      *Qwen-Image-2.1:* one artifact per component and a composition naming
      them (D-089; [artifact-format.md](artifact-format.md#compositions)):
      `import_m3.py component` wrote the text encoder (17.53 GB), denoiser
      (14.23 GB) and VAE (1.35 GB, F32) and `compose` their composition
      `eca21baa…` on `spark` in under three minutes; `artifact/composition.h`
      reads it natively ([qwen-image-native](experiments/qwen-image-native/README.md)).
- [x] **Kernels and the source lock** (D-053, D-057, D-077): the pinned
      llama.cpp has much of what the models need (quantized matmul and
      `mul_mat_id`, MoE routing, the lightning indexer, `dsv4-hc`, gated
      delta net and `ssm-conv`, prefill flash attention at head dimensions
      256 and 512, and the image's 3D convolution and VAE operations), but
      the narrowed GGML build compiles none of it. Widening it is a
      source-lock change on the heavy path, with D-057's gates and the
      workstation tier D-084 requires. Qwen3.8's NVFP4 and MXFP8 matrix
      products take the fastest correct implementation from any source,
      chosen per operation by a quick A/B (D-085): GGML's NVFP4 MMQ (MXFP8
      in GGML is not verified), vLLM, FlashInfer or CUTLASS kernels, or our
      own, with licenses handled per D-080.
      *GGML widened for the two LLMs* (`kernels/ggml/ops_ext.h`): quantized
      MMVQ and MMQ products and `mul_mat_id` for DeepSeek's GGUF types (Q8_0,
      Q4_K, Q5_K, Q6_K, IQ2_XS, IQ3_XXS, MXFP4); tensor-core flash attention
      at head dimensions 256 and 512, 8 query heads per KV head, with sinks
      and DeepSeek's sparse gather; the lightning indexer, hyper-connections,
      gated delta net and `ssm_conv`; RoPE with offsets, YaRN and IMROPE,
      forward and back; argsort, top-k and the elementwise and row
      operations. Each is checked on the host in every profile and matches
      an FP64 reference on a GB10 within upstream's test-backend-ops bounds.
      *Qwen3.8's formats, by a quick A/B*
      ([qwen38-native](experiments/qwen38-native/README.md#kernel-ab-d-085)):
      NVFP4 experts on GGML's MMVQ and MMQ (the NVFP4 MMQ instance unit added
      to the lock); MXFP8 products on jitLLM's own vector product up to 8
      rows and otherwise dequantized to BF16 for cuBLAS (since the second
      prefill pass, the reference form's; the default takes CUTLASS's
      MXFP8 GEMM, 1.9–2.5× cuBLAS's at 4,096 rows); the n-gram table's
      NVFP4 rows on jitLLM's own lookup (`kernels/ggml/jitllm_ops.h`). Each
      matches an FP64 reference built from the format's dequantization on a
      GB10. For prefill, CUTLASS 4.7.1's NVFP4 grouped GEMM (BSD-3, a new
      lock component, headers only) now takes the routed experts over a
      CUTLASS layout the importer writes, with jitLLM's own vector products
      over it for decode; 1.16–2.84× GGML's MoE block at 512 to 8,192
      tokens
      ([qwen38-native](experiments/qwen38-native/README.md#prefill-d-085)).
      *The image pipeline's* (Qwen-Image-2.1, BF16, chosen per operation by
      speed): cuBLAS BF16 products, jitLLM's own FlashAttention-2 kernel
      (3.37 ms per denoiser block, as PyTorch's flash kernel; GGML's
      tensor-core kernel, built for D = 128 without head grouping, took
      19.9 ms) and jitLLM's fused BF16 kernels for the norms, modulation,
      rotary embeddings, residuals and the VAE, each rounding where
      diffusers rounds (`kernels/image`,
      [qwen-image-native](experiments/qwen-image-native/README.md)); the
      VAE's convolutions (its causal 3D convolutions are 2-D at one frame)
      were im2col and cuBLAS and are, since the speed slice, jitLLM's own
      implicit GEMM, so GGML's were not needed; the products are pinned
      cuBLASLt algorithms. The image's operations are declared in the
      registry and run through a bound plan. No source-lock change.
      Open: the vector attention at D = 256, which
      upstream picks for Qwen3.8's decode below 8,192 cells (the MMA kernel
      runs it meanwhile). CUB stays out although its licenses are
      cleared (D-091): upstream's CUB top-k is 2–3× faster for one row but
      under 1% of a decode step below 1M positions, and 3.6–71× slower for
      prefill's many rows (measured 2026-09-28,
      [licensing.md](licensing.md)).
- [x] **Model graphs and state** (pulled from M7 and M9): DeepSeek V4's
      compressed sparse attention with its indexer (CSA/HCA) and mHC;
      Qwen3.8's QSA, hyper-connections and Gated DeltaNet layers; the
      resident MoE execution both need (pulled from M7; no demand-paged
      experts); Qwen-Image's text encoder, DiT and VAE, releasing each
      component outside its phases. Each model's KV, indexer and recurrent
      state has a state adapter with spill and restore coverage (RE-004,
      RE-007).
      *DeepSeek V4 Flash, native and resident* (`model/dsv4.h`,
      `kernels/ggml/dsv4_graph.h`, `jitllm_dsv4_exec`): llama.cpp's
      `deepseek4.cpp` graph (CSA with the lightning indexer and top-k, HCA,
      the window, sinks, q/o LoRA and output groups, mHC with its Sinkhorn
      comb, 256 experts top-6 plus the shared one with sqrtsoftplus and
      noaux_tc routing, the hash-routed layers, and the head) planned
      unfused through the registry, from the artifact on `spark-b`. Against
      llama.cpp on the same 0731 GGUF with its fusion off, every logit of
      the 8 prompts' 256 greedy steps and every token's perplexity NLL is
      bit-identical, and the free-running continuations are identical
      ([dsv4-native](experiments/dsv4-native/README.md)). The KV, indexer
      and compressor state is explicit and bounded (`Dsv4StateLayout`, three
      D-068 representations); its spill and restore are the swap path's.
      *On the paged node* (`engine/dsv4_runner.h`): each chunk a device
      job over its leased closure (D-086), 46,232 extents paged through the
      landing zone, each layer's expert slab as 2 MiB pages whose contents
      land in pieces; all 8 prompts' 32 steps bit-identical to the resident
      harness's logits ([swap](experiments/fast-swap/swap.md)).
      Open: the executed-plan record against llama.cpp's, and the other
      models.
      *Qwen3.8 Flash Next, native and resident* (`model/qwen38.h`,
      `kernels/ggml/qwen38_graph.h`, `jitllm_qwen38_exec`): llama.cpp's
      `qwen4exp.cpp` operation plan (hyper-connections, the n-gram
      embedding layer, Gated DeltaNet, QSA with its indexer and budget, 512
      experts top-10 plus the gated shared one, the head) over the artifact
      on `spark-b`. Against Mia's vLLM (deterministic, MTP off) on the same
      checkpoint: 180 of 192 teacher-forced greedy steps agree, the other 12
      at oracle margins of at most 1.0 nats, within the 95th percentile of
      jitLLM's own kernel-to-kernel margin noise (a bound set after the
      first comparison, so not pre-registered; it fails at the 90th);
      perplexity 14.43 against 14.66 (−1.5%), top-1 accuracy equal. The
      KV, indexer, recurrent and convolution state is explicit and bounded
      (`Qwen38StateLayout`, three D-068 representations), and a spill and
      restore of it is bit-identical. Decode 0.99× the oracle's; prefill,
      after fused hyper-connection, MoE-output and Gated DeltaNet kernels
      and CUTLASS's grouped GEMM for the routed experts, 0.91× at 8,192
      tokens in 8,192-row chunks, within D-085's 10% gate (0.89× in
      4,096-row chunks misses it; 1.40× at 2,048, 1.10× at 512); peak
      memory 0.97× at a 4,096-token context against vLLM's 262,144, 1.07×
      while prefilling 8,192 tokens in one chunk
      ([qwen38-native](experiments/qwen38-native/README.md#prefill-d-085)).
      A second prefill pass under D-085's speed before bit exactness made
      the graph's default a fast form (the first pass's fusions stay as
      its reference form): the MXFP8 products on CUTLASS's MXFP8 GEMM over
      activations quantized to MXFP8, as the oracle runs them, and fused
      hyper-connection, routing, Gated DeltaNet and QSA kernels, QSA's
      selection making its mask on the device. Prefill 1.41× the oracle's
      at 8,192 tokens in 8,192-row chunks, 1.38× in 4,096-row chunks, 2.02×
      at 2,048, 1.56× at 512; decode unchanged; perplexity −0.8% to −1.2%; one of
      the 192 greedy steps now misses the near-tie bound (jitLLM's own
      margin there is 0.18 nats in the reference form; accepted by the owner,
      2026-09-28, as a known divergence)
      ([qwen38-native](experiments/qwen38-native/README.md#prefill-second-pass-speed-before-bit-exactness)).
      *On the paged node* (`engine/qwen38_runner.h`): chunks as device
      jobs over leased closures, the expert slabs as DeepSeek's pages, the
      28.8 GB n-gram table never resident but read by rows before each
      chunk (4 KiB-aligned direct reads, D-035's evidence recorded: 486×
      fewer bytes than whole chunks), the state spilled and restored
      through the swap path; the six prompts' 32 steps bit-identical to the
      resident harness's logits ([swap](experiments/fast-swap/swap.md)),
      again since the prefill work with the fused graph and CUTLASS's
      grouped GEMM over the CUTLASS-layout artifact, which it pages in as
      is (swaps into Qwen3.8 6.9–7.9 s, against 7.4–8.8 s before).
      Past 2,051 attended cells the QSA indexer's top-k is not repeatable
      (RE-031: GGML's radix select picks among ties by timing).
      Open: a deterministic top-k.
      *Qwen-Image-2.1, native* (`model/qwen_image.h`,
      `jitllm_qwen_image_exec`): the text encoder (Qwen3-VL's text path, the
      system turn dropped), the block-causal DiT with its text K/V prefix
      cache and the flow-matching Euler scheduler, and the VAE decoder, from
      the composition on `spark`, each phase bounded and, with
      `--phases released`, each component loaded for its phase and freed
      after it. Against diffusers BF16 at the fixed teapot prompt (1024², 40
      steps, seed 42, diffusers' initial latents), every pre-registered
      bound passes: the image at PSNR 41.8 dB and SSIM 0.996 (bounds 32 dB,
      0.98; an FP32 DiT gives 38.8 dB); full generation 36.9–37.0 s against
      diffusers' 52.6 s, 0.89 s per step against 1.26 s, peak memory 31.2
      GiB against 43.4 GiB resident and 15.9 GiB released
      ([qwen-image-native](experiments/qwen-image-native/README.md)).
      *On the paged node* (`engine/qwen_image_runner.h`): each component
      a set of extents, each phase a device job over its own component's
      closure, a generation one request leasing all three (D-093); the
      image pixel for pixel the resident
      harness's, generated before and after swaps
      ([swap](experiments/fast-swap/swap.md)).
      *Speed* (the same BF16 numerics): both harnesses run one pipeline
      dispatched through a plan bound against the registry (D-053,
      `kernels/image/pipeline.h`); pinned cuBLASLt products (bit for bit
      `cublasGemmEx`'s), the gated residual fused with the next norm, the
      prefix cache read in place and the query norm fused into the
      attention (all bit for bit the M3 slice's denoiser), implicit-GEMM
      VAE convolutions and a CUDA graph per step: full generation 33.4 s
      against 36.2 s before on the same host (0.64× diffusers' 52.6 s), a
      step 0.82 s, decode 0.36 s; image 41.77 dB, SSIM 0.996, repeatable
      ([qwen-image-native](experiments/qwen-image-native/README.md#speed)).
- [x] **Resident expert layout** (the initial choice pulled from M7):
      repacked expert groups get executable views that GGML's `mul_mat_id`
      and the NVFP4 path's grouped GEMM accept with every expert resident.
      Pointer-table or uniform-stride dispatch is chosen per format, by a
      quick A/B where both are viable (D-085), and proven by one MoE layer
      whose output is bit-identical to the reference layout's. Compaction
      and demand-paged dispatch stay in M7.
      *GGML, DeepSeek V4:* uniform stride over a per-layer slab with stock
      kernels ([artifact-format.md](artifact-format.md#executable-views));
      no A/B, since the pointer table needs a kernel patch and gives a
      resident model nothing. All 43 layers' routed products (387 cases,
      MMVQ and MMQ) equal the reference layout's bit for bit.
      *GGML NVFP4, Qwen3.8:* the same layout, S = 2,768,976 bytes; the down
      projection's short rows read their padding inside the slab. All 48
      layers' routed products (432 cases) equal the packed reference's bit
      for bit ([qwen38-native](experiments/qwen38-native/README.md)). A
      grouped-GEMM path brings its own view: the harness rewrites each slot
      in place into CUTLASS's layout (the same bytes, permuted; every layer
      converts back to the loaded bytes exactly), which the grouped GEMM
      and jitLLM's decode products read
      ([artifact-format.md](artifact-format.md#executable-views)).
- [x] **Tokenizer and chat templates** (pulled from M5; D-067): the native
      tokenizer, renderers for each model's pinned template (DeepSeek's
      upstream ships Python encoding scripts, not a template), stop rules,
      and greedy and seeded sampling. Landed in tests
      ([tokenizer.md](tokenizer.md)): token for token with llama.cpp and
      Hugging Face on a 92-item corpus for all three models, and both
      templates byte for byte; shipped binaries may link them (D-088,
      accepted 2026-09-28). The output
      parsers and the request mapping are the chat route's.
- [x] **Speculative decoding in the core** (pulled from M9; D-068): Qwen3.8's
      MTP layer and DeepSeek's DSpark drafter, with verify and rollback that
      keep only the accepted prefix, and forced draft rejection for the
      exit's speculation checks (moved from M9). Every published Mia and
      TensorFold decode number uses speculation.
      *DeepSeek's DSpark landed* ([dspark](experiments/dspark/README.md),
      D-092): the drafter its own artifact, binding the target's token
      table and head; its KV ring a D-068 representation; a verify of the
      anchor and 3 drafts on a row-invariant plan, so greedy speculation is
      bit-identical to greedy decoding; rollback by restoring the bytes a
      verify saved; draft and verify one job, both replayed as graphs.
      All four speculation checks pass on `spark-b`: greedy bit-identical
      (8 prompts), forced rejections equal to their control state by state
      (91 steps), rollback across a swap, and sampled speculation (total
      variation 0.0005–0.027, bound 0.1; plain sampling's run took 10.6
      min at a host load of 16–20, over D-085's 10). Decode 27.9 / 29.3
      tok/s on `prose` / `code` against llama.cpp's 30.8 / 31.9 with the
      same drafter (0.91× / 0.92× on the medians, a narrow pass: one
      `prose` repeat of three was 0.87×; acceptance 0.55 / 0.58), peak
      memory equal (104.7 GiB). Since then, with each generation a
      request (D-093) and the runtime wake (D-094): 29.7 / 30.8 tok/s,
      0.96× / 0.97× ([dspark](experiments/dspark/README.md#performance-and-memory)).
      Open: the verify's device time.
      *Qwen3.8's MTP layer landed* ([qwen38-mtp](experiments/qwen38-mtp/README.md);
      D-089's and D-092's notes). The MTP block is a drafter artifact of its
      own (1.6 GB, imported in 6.7 s; the target is unchanged) that binds
      the target's token table and head. Its caches and the target's
      streams are a D-068 state.
      The verify is batched (the owner's speed before bit exactness). It
      never writes the recurrent, convolution or n-gram state: a commit
      kernel replays the kept rows into them, and the rejected rows' KV
      and indexer cells are restored from a snapshot.
      On `spark`, at depth 2 with a 65,536-row draft head, it decodes at
      42.46 / 39.09 tok/s (`prose` / `code`, medians of three) against Mia's
      MTP 3 at 37.85 / 37.85: 1.12× / 1.03×. Acceptance is 0.631 / 0.539
      against Mia's 0.42, and the peak `MemAvailable` drop 75.8 GiB against
      103.4, not like for like: the 28.8 GB n-gram table stays on the SSD
      here (read by rows, D-035), and vLLM's 16.2 GiB KV pool is sized for
      its concurrency.
      The checks:
      - greedy on 8 prompts: every token is the plain argmax, or a near-tie
        within 1.0 logit (9 near-ties, 0 violations);
      - forced rejections at depths 2 and 3: every state equal to the
        control's;
      - sampled speculation: total variation 0.009–0.040 (bound 0.1);
      - rollback across a swap (Qwen3.8 out for the FP16 fixture and back,
        a commit owed across it): all 97 steps' states, 160 tokens and
        their logits equal the unswapped control's.
      The 1.0 bound is qwen38-native's, not dsv4-decode's later rule (the
      verify's own noise, p99 0.23–2.39 per prompt, measured afterwards);
      it is kept as the stricter test.
- [x] **The swap path:** evict the outgoing model and hand its backing to
      the incoming one (D-033's handoff, pulled from M6; D-081), with
      page-in through the landing zone overlapping the rest. Creating and
      mapping backing costs about 70–76 µs (`cuMemCreate`) and 40–43 µs
      (`cuMemSetAccess`) per 2 MiB extent (measured medians), about 5 s of
      serial work for DeepSeek's ~46.2k extents (computed), so reusing the
      evicted model's backing matters; unmapping the outgoing extents and
      one access call per contiguous range are not yet measured. Large
      sparse lookup tables (Qwen3.8's PLE) may stay on the SSD with rows
      paged on demand (D-035); everything else is resident before the first
      token. A's KV and recurrent state are spilled on swap-out through M2's
      write-back path and restored on return, with no re-prefill; that cost
      counts in the swap time. D-055's named spill format and the retention
      policy stay in M6.
      *Core landed, with DeepSeek* ([swap](experiments/fast-swap/swap.md)):
      the handoff (an eviction parks its unmapped backing, charged; a load
      of the same class and size takes it; the rest is released when the
      evicting task ends), state spill and restore through the zone, and
      the zone's copies on a copy lane of their own with fences from a
      pooled set of events (RE-029: `cuEventCreate` blocks too). DeepSeek
      (8,192 context tokens) ↔ the FP16 fixture on `spark-b`: B→A 7.36 s
      prepared, 7.44 s first use, A→B 1.79–1.86 s, DeepSeek evict-all
      reload 9.07 s, all from the request to the first token; page-in at
      13.4 GB/s; A resumed after B bit-identical to A never swapped. The
      handoff saves 1.3–1.4 s of a 46k-extent eviction.
      Graph restore: DeepSeek's decode graphs survive swaps (D-090).
      *M3's pairs* ([swap](experiments/fast-swap/swap.md#results-m3s-swap-pairs-spark-b-2026-09-28)):
      every ordered pair of DeepSeek, Qwen3.8 (its n-gram table paged by
      rows, D-035) and Qwen-Image (each phase over its own component),
      A→B→A with 8K and 0 context, first use and prepared, on `spark-b`:
      all 32 swaps under the ~10 s goal, the worst LLM↔LLM swap 9.38 s
      (8.76 s prepared and 9.04 s first use at 8K context; an earlier run
      of the same path reached 9.66 s; the margin is the SSD's at-rest
      rate for 75–97 GB), the image's first step 5.0–6.3 s after a swap
      from an LLM; an LLM A's restored state byte-identical
      and its continuation bit-identical to the same state's, the image
      regenerated pixel for pixel. Open: overlapping eviction with page-in,
      Qwen3.8's and the image's graphs, and a deterministic top-k for
      Qwen3.8 (RE-031).
      *Through `jitllm-runtime`* (D-096, [swap](experiments/fast-swap/swap.md#through-jitllm-runtime-d-096)):
      all three models registered in one process, every fast path on
      (speculation, so the drafters page in with their targets: DeepSeek
      108.4 GB, Qwen3.8 77.0), every ordered pair A→B→A, 8K and 0 context,
      first use and prepared on `spark-b`: all 32 swaps under the ~10 s goal
      and exact (A's state digest and its 16 continued tokens and logits,
      B's output, the image's pixels, graphs replayed after prepared
      returns); **the worst LLM↔LLM swap 9.72 s** (Qwen3.8 → DeepSeek, first
      use, 8K; 9.64 s prepared), 0.8 s of it paging DSpark's 10.9 GB; the
      image's first step 5.2–6.0 s after a swap; peak 108.7 GiB. With
      speculation off the LLM pairs take 7.5–8.9 s, within 0.6 s of the
      harness's. That is M3's swap table in a running process, against
      llama.cpp's 77 / 104 s, Mia's vLLM's 13 min 13 s, TensorFold's 141 s
      and diffusers' 212 s.
      *Final integrated rerun, 2026-09-30:* all 32 rows pass on the
      growing-state/turn-reuse/HCA/frontier default path, at 262K configured
      ceilings; worst LLM↔LLM/first use 9.749 s, prepared 9.701 s.
      Exact 8K continuations, retained graphs, fresh zero-context hashes
      and the qualified fast image RGBA all pass
      ([final table](experiments/fast-swap/swap.md#final-integrated-table-2026-09-30)).
- [x] **CUDA graphs for decode** (pulled from M9): captured per model and
      plan and replayed after swaps that restore every extent at the same
      virtual addresses, with setup and tuning state restored the same way.
      This reopens D-086 and needs the relocation proof first. BP-F4's
      per-token host cost is measured on these models (about 1.9 µs per
      launch measured on the fixtures; several milliseconds per MoE token
      is an estimate).
      *DeepSeek V4 Flash, on the paged node* (D-090,
      [graphs](experiments/fast-swap/graphs.md)): each one-row chunk shape
      is captured on its second step through the launch context (input
      copies, 4,972 steps, logits copy: one 5,920-node graph, about 25 ms)
      and replayed as one launch; the relocation proof pins the weights'
      and state's places in the scheduler, so a swap maps backing back at
      the addresses every graph names. Replayed steps are bit-identical to
      launch by launch and to the resident harness (so to llama.cpp with
      fusion off), and A resumed after B replays graphs captured before the
      swap without capturing again. Decode 19.05–19.53 tok/s against
      18.13–18.82 launch by launch; llama-bench's tg64 is 20.62 with fusion
      and graphs on, 20.04 with fusion off; the step's device time alone is
      48.3–48.6 ms, and the rest is the paged node's per-step round trip.
      The job's host time per token fell from ~41.5 ms (launches waiting on
      a full stream) to 0.13–0.16 ms. The owner (2026-09-28) questioned
      whether graphs are worth their complexity at 1.04–1.05×; re-measured
      once a request leases its closure once (D-093, below), they are still
      worth 1.046–1.055× and stay (D-090's note).
      *A lease per request* (D-093, [swap](experiments/fast-swap/swap.md#a-lease-per-request)):
      a request (a turn; for the image, one generation) leases its model's
      closure once and runs every chunk under it; with the lanes polling
      through a step (harness-polled) the per-step round trip fell from
      1.3–2.6 ms to 0.01 ms. DeepSeek decodes at 20.34–20.46 tok/s, 1.016–1.022×
      llama.cpp's fusion-off, graphs-on tg64 (20.02) and 0.990–0.996× its
      default (20.54), the same session (two runs); Qwen3.8 from the
      CUTLASS-layout artifact at 23.71–23.81 tok/s paged, 0.94× Mia's vLLM
      with speculation off: its job's device span alone is 1.03× vLLM's
      step, and ~1.3 ms a step is host work outside the job (the n-gram
      rows and inputs), not the lease. Everything stays bit for bit, and
      the DeepSeek ↔ Qwen3.8 swaps (first artifact) did not move. A holder
      of a request's lease is refused a wait for another's (no hold and
      wait).
      *Qwen3.8's decode graphs* ([qwen38-mtp](experiments/qwen38-mtp/README.md#performance-and-memory)):
      the n-gram row gather now reads its row count from pinned memory, so
      one graph serves every step. Verify and draft shapes are captured the
      same way. Plain decode on `spark-b` (a request lease, the runtime
      wake) runs at 25.65–25.88 tok/s against 23.84–24.05 launch by launch,
      and every step's logits are identical between the two. That is
      1.01–1.03× Mia's vLLM with speculation off (25.12 / 25.33). A step is
      37.4 ms on the device, and its host time is 0.07 ms. Across a swap
      (`jitllm_swap_pairs --a qwen38 --b image`), a prepared return
      replays the graph captured before it for every continued step,
      bit-identical to the unswapped continuation.
- [x] **RE-029's lead:** read `CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS`
      on the GB10, one `cuDeviceGetAttribute` call. Mia's
      `patch_ple_offload.py` reports it as 0, with `cuStreamWaitValue32`
      then blocking the host's next launch (creator-reported). The answer
      decides how page-in copies and phases share the submission lane
      during a swap. *Done 2026-09-28* ([RE-029](rough-edges.md#re-029-a-jobs-kernel-launches-can-block-its-lane-while-the-stream-is-busy--2026-09-27-status-worked-around)):
      the deprecated `_V1` attribute reads 0 and the current memory
      operations are supported; the wait does not block the next launch.
      A stream holds about 1,020 pending operations and a launch into a
      full one blocks the thread, so the swap path keeps the zone's copies
      off a thread that can launch into a full stream, or keeps each
      stream under the limit.
- [x] **The runtime wake** (D-094, [runtime-wake](experiments/runtime-wake/README.md)):
      benchmarks measure what the runtime does, not the harness's 100 ms
      of polling. On the GB10 a blocking-sync event or a host function
      woke 1.0–1.7 ms after a step's end; every sleeping thread on a
      step's path cost 0.1–0.6 ms (RE-017). The device completion lane
      now sleeps through most of a step, spins only around its likely
      ends and wakes the scheduler and the submission lane ahead of the
      completion; the scheduler polls after a step for about as long as
      its client takes to ask for the next. A synthetic step's round trip
      is 26–28 µs at 0.11–0.12 of a core while stepping and none idle
      (harness-polled: 9 µs, four cores; the old defaults: 0.56–0.67 ms,
      two). With it DeepSeek decodes at 20.42–20.50 tok/s (graphs),
      Qwen3.8 (after its fast prefill, `f9a4e0f`) at 23.85–24.09, and
      DSpark speculates at 29.67 / 30.85 tok/s (`prose` / `code`, 0.96× /
      0.97× llama.cpp's), each generation now a request; DeepSeek is
      1.020–1.024× llama.cpp's fusion-off tg64 and 0.999–1.003× its
      default in the same session, Qwen3.8 0.94–0.96× Mia's vLLM. Against
      the old window the wake costs 0.01–0.03 ms a step, inside the runs'
      spread. Prefill (8,192 tokens: DeepSeek 27.37–27.45 s, Qwen3.8
      6.84 s) and the DeepSeek ↔ Qwen3.8 swaps (7.1–8.8 s, page-in
      13.3–14.1 GB/s) did not move with the wake; every check stays exact.
- [x] **DeepSeek decode past llama.cpp** (the owner, 2026-09-28: speed
      before bit exactness, D-085's and D-092's notes;
      [dsv4-decode](experiments/dsv4-decode/README.md)): DeepSeek's fast
      plan is the default for decode, verify and draft chunks. One
      quantized vector kernel (`jitllm.vecq`) serves every product and
      reads each routed expert once for all the rows that select it (the
      batched verify), with fused routing, combine, hyper-connection
      pre-mix and compressor kernels, and PDL. A decode step drops from
      5,575 kernels to 2,558. With the runtime wake it decodes at
      21.90–22.21 tok/s against llama.cpp's 20.41 in the same session
      (1.07–1.09×; 1.10–1.11× its fusion-off arm), and DSpark at 31.6–31.7
      / 34.3–34.4 on `prose` / `code` (1.03× / 1.07–1.08× the recorded
      30.80 / 31.94). Correctness is coarse against llama.cpp: greedy
      244/256 and 250/256 equal, the rest near-ties (oracle margins under
      1.0); perplexity within 0.5%; speculation and forced rejections
      with no stale state byte; sampled speculation's total variation
      0.004–0.038 (bound 0.1). Swap, restore and repeat stay
      bit-identical. The exact plan and D-092's verify remain as
      `--exact on`. The recorded near-tie bound (6.11, twice the largest
      fast-against-reference move) is too loose to be a test; later
      slices take the 99th percentile of noise between two of jitLLM's
      own paths ([dsv4-decode](experiments/dsv4-decode/README.md#the-bound-going-forward)),
      under which one forced-run speculative token (a 3.62-nat
      disagreement) is flagged. It is diagnosed as kernel noise amplified
      by near-tied routing ([step 93](experiments/dsv4-decode/README.md#step-93-diagnosed)).
      Open: "decisive" (about 1.1×) is not reached on any target; the products run at
      about 200–210 GB/s in the model against 230–245 alone, unexplained.
- [x] **TensorFold's techniques, Qwen3.8's decode gap first** (the owner,
      2026-09-28; [tensorfold-techniques](experiments/tensorfold-techniques/README.md)):
      estimated logical selected-weight payload per plain-decode step was
      7.3–7.6 GB in Mia's NVFP4, MXFP8 and BF16 against 4.5 GB in MLX
      4-bit. This estimate does not isolate format versus engine costs or
      measure hardware traffic. Adopted within the format: the Gated DeltaNet state
      updated in place (`jitllm.gdn.step`; TensorFold double-buffers it),
      the hyper-connection prep across a cluster of blocks, a BF16 vector
      kernel for the mixes' one-row products, the one-row convolution
      fused, and PDL with L2 prefetch for the vector products. Plain decode
      +5.4–8.1% against main in the same session (25.83 → 27.36–27.39
      tok/s, 37.4 → 35.2 ms on the device); a speculative step 2.4%
      shorter, its rate moving with acceptance; the adaptive speculation
      window (the drafter's probability, TensorFold's 0.3 cut) is measured
      and left off; DeepSeek unchanged (its path already had these).
      Greedy agreement (bound recorded first), perplexity, forced
      rejections and the swap check pass; loads, evictions and decode
      migrated no pages unless another process pressed memory. Open: the
      causal attribution of the remaining gap, including MXFP8 products and
      the BF16 head; TensorFold's kernel-level profile (its container's CUPTI
      recorded nothing in this study).
- [x] **Swap runner:** a native CLI harness in `jitllm-runtime` that drives
      A→B→A in a running process (tokenize, prefill, decode, detokenize) and
      reports each part of the swap time.
      *First as harness binaries* (`jitllm_swap_runner`, `jitllm_swap_pairs`).
      *Done 2026-09-28 in the runtime* (D-096,
      [runtime-serving.md](runtime-serving.md)): the paged node and the three
      models' runners moved into an `engine` module and the task programs
      into the scheduler; the configuration names the models
      (`[models.<name>]`: artifact or composition, drafter, context,
      tokenizer and template); `jitllm-runtime chat --turn MODEL TEXT...`
      serves turns with the native tokenizer and renderers, speculative by
      default (DSpark, MTP), swapping as needed, and `jitllm-runtime
      swap-table` runs every ordered pair in one process. Through the
      runtime, on `spark-b`: greedy tokens equal the speculation
      harnesses' on the fixed prompts, speculative and plain, for both LLMs
      (6 chat prompts at 32 tokens and 2 decode prompts at 256; prompt IDs
      equal too); the image's pixels the reference's; decode at the
      harnesses' speeds (DeepSeek 30.6 / 34.6 tok/s speculative on `prose`
      / `code`, 21.9 / 22.2 plain; Qwen3.8 40.1 / 38.2 and 25.7 / 26.0); the
      harnesses' greedy, forced-rejection and swap speculation checks pass
      over the moved engine. The runtime's swap table is below (the swap
      path). The package now ships cuBLAS and GGML's and CUTLASS's notices.
      The package step of `check:full` (the arm64 package, its inventory and
      its install test), with REUSE, passed on the workstation on 2026-09-28;
      the header check ran on `spark-b`.
      Open: the image takes one prompt a process from a latents file (its
      runner fixes both at setup).
- [x] Start the model support matrix (moved from M5), recording template
      hashes. *Done 2026-09-28* ([model-support.md](model-support.md)):
      each model and drafter jitLLM runs, the M2 fixtures among them, with
      its checkpoint pin, artifact, template hash as `src/chat` keys it,
      tokenizer, decoding modes, context exercised, evidence, known
      divergences, status and level.
- [x] Last, once the swap floor is proven: a minimal OpenAI-compatible
      `/v1/chat/completions` on loopback (D-014), with numeric intake
      bounds fixed before it accepts input
      ([client-api-baseline.md](client-api-baseline.md#shared-correctness-and-limits)).
      The full front door stays in M5.
      *Done 2026-09-28* (D-097, [runtime-serving.md](runtime-serving.md#the-chat-route)):
      with models configured, the service listens on `[client] bind`
      (loopback only, default `127.0.0.1:8114`) for `POST
      /v1/chat/completions` (JSON and SSE) and `GET /v1/models`, over its
      own bounded HTTP/1.1 reader (no new dependency), one request at a
      time behind a queue of four; every bound (head, target, body, JSON,
      messages, message bytes, parts, `max_tokens` against the model's
      usable context, temperature, top_p, seed, stop strings, timeouts,
      queue) is tabulated there with its status; unknown fields are
      refused. Unit tests cover parsing, every bound at its edge and the
      server over a loopback socket (routes, browser guards, HTTP bounds,
      timeouts, the queue's 429, streaming, errors after the headers,
      disconnect and stop). On `spark-b` with DeepSeek and Qwen3.8
      configured, through curl: DeepSeek, a swap to Qwen3.8 streamed and
      a swap back (10.0, 9.1 and 11.0 s from request to response, the
      swaps 8.1, 7.8 and 9.5 s); each greedy reply equals `jitllm-runtime
      chat`'s on the same prompt (text, 52 and 37 tokens, prompt counts,
      finish), the stream ends in `[DONE]`, a seeded sampled request
      repeats exactly, a stop string ends the answer, and a request over
      each bound was refused (413, 400 with `context_length_exceeded`
      for the prompt and for `max_tokens`, 404, 403, 415, 405). The
      runtime now samples too (seeded, speculative sampling where there
      is a drafter); DeepSeek, like Qwen3.8, is refused at registration
      when its template has no renderer. Open: M5's front door
      (an optional API key, loopback-origin CORS, tools, reasoning
      controls, the other routes, client validation).
      *Amended by the owner 2026-09-28* (D-097 accepted as amended, and
      an owner note on D-014): it listens on loopback and the tailnet by
      default and on any configured address, authentication optional on
      each (unauthenticated listeners named at startup),
      ignores unknown fields by name with a table at
      `/jitllm/v1/ignored-fields`, honors `top_k` and `min_p`, and serves
      persistent connections from an epoll I/O thread (1,024 connections,
      64 queued, pipelining refused, slow clients isolated) with SSE
      keepalives through the queue, swaps and prefill. Unit tests cover
      the bind resolution over fake interface lists, the Host and Origin
      guard, unknown fields and their table, keep-alive, pipelining,
      idle connections and eviction, the queue, keepalive comments,
      early starts and slow clients. On `spark-b` (2026-09-28) with
      DeepSeek and Qwen3.8: it listened on 127.0.0.1, ::1 and both
      tailnet addresses, naming `spark-b.coati-puffin.ts.net`; two
      requests shared one connection; a request with unknown fields was a
      200 and both names appeared in the table (logged once, values
      never); through the tailnet address the MagicDNS name and short
      name passed the Host check while `evil.example` and a trailing-dot
      name got 403 and the table route 404; 8 concurrent streams all
      ended in `[DONE]` in 17.1 s (one swap); 6 streams alternating the
      two models (a swap each) started at 15.0 s in the queue, heard
      keepalive comments and all finished (the last in 52.8 s); greedy
      replies still equal `jitllm-runtime chat`'s (DeepSeek 52 tokens,
      Qwen3.8 37, streamed).
      *Fixed after an outside review, 2026-09-29* (D-096 and D-097
      amended, [runtime-serving.md](runtime-serving.md#prefill-chunks-and-cancellation)):
      a client leaving, the deadline or SIGTERM now stop a prefill
      between chunks (0.65–1.57 s on `spark`), the state keeping the
      chunks that ran, and the service serves on; each model's prefill
      chunk is its own (DeepSeek 2,048 rows, Qwen3.8 4,096, measured:
      1.48× and 1.78× the 512-row prefill at 8K tokens), configurable
      (`prefill_chunk`) and capped by the model at its context, so the
      minimum context, 512, starts for both.
- [x] **Engine cleanup before the gate** (the owner, 2026-09-29: leave M3
      with the code as clean as possible and set up for later work):
      - the per-model runners' shared mechanics become one engine
        skeleton: graph capture and replay, state spill and restore, plan
        caches, speculation mechanics, setup and teardown, weight paging.
        That's the shared helpers [portability.md](portability.md)
        recommends, so a new model family adds only its own plan and
        state layout;
      - behaviour is unchanged, bit for bit where the engine was: the
        same tokens, logits, pixels and swap results as before, and decode
        and prefill within noise;
      - main is clang-tidy clean, and the chat route's flaky half-close
        test is fixed.
      It lands before long context's optimization slices, so they change
      one skeleton rather than each runner.
      *Done 2026-09-29* ([engine.md](engine.md), [portability.md](portability.md#the-runners-shared-skeleton)):
      the skeleton is `paged_weights.h` (every runner's weights, DeepSeek's
      host table and slabs included), `live_state.h` (regions, spill,
      quarantine, a verify's snapshot, accept, rollback and a commit hook),
      `planned.h` (one `PlaceAndPlan`, plan caches, the graph cap, BP-A1's
      coverage), `graph_runs.h` (staging, capture, replay) and
      `runner_resources.h`, with `PagedNode::WithRequest` for a request's
      lease; engine.md says how a model family plugs in and where the
      long-context work goes. DeepSeek's runner went from 2,607 lines to
      1,610, Qwen3.8's from 2,136 to 1,558, the image's from 879 to 822,
      beside 1,407 of skeleton. Qwen3.8 now checks its pinned places after
      a swap too. On `spark-b` in one session, main (90660dd) against the
      slice: DeepSeek's and Qwen3.8's greedy tokens and logits, plain and
      speculative, on the eight fixed prompts (64 tokens) identical by
      SHA-256, DeepSeek's reference mode too (where speculative equals
      plain bit for bit); the forced-rejection checks 0 stale bytes
      (DeepSeek, 17.6 GB compared) and 0 state differences (Qwen3.8), the
      swap checks 0 states differ and 0 logits differ, graphs replayed
      after the swap; `swap-table --pairs deepseek:qwen3.8` exact in every
      row, totals 7.89–9.77 s against main's 7.91–9.67 s over two
      alternating passes; Qwen-Image's pixels `3b7770ca…` and its pair
      exact; through `jitllm-runtime chat` the same replies, prefill at 8K
      451–456 tok/s (DeepSeek) and 2,179–2,186 (Qwen3.8) on both, plain
      decode at 8K 19.06–19.17 and 25.54–25.58 tok/s on both. Main is
      clang-tidy clean over all of `src/`, `tests/` and `benchmarks/`
      (225 units, `spark-native`'s database). The half-close test's
      expectation was too strict: a client's shutdown can reach the I/O
      thread after the backend made the response, so it now expects close
      after an interim response and keep-alive without one, the server
      closing either way (a generation held until the shutdown is seen
      still must say close); 24 copies ×
      500 repeats passed 12,000 of 12,000 where main's test failed 136.
- [x] **ds4 study** (the owner, 2026-09-29), after long context's scaling
      slices. [ds4](https://github.com/Entrpi/ds4) is antirez's
      MIT-licensed C/CUDA engine for DeepSeek V4 Flash, in Entrpi's fork
      tuned for the Spark. It reports prefill of 960 tok/s at 2K and 933
      at 64K on one GB10 with a community IQ2_XXS GGUF, about 2× ours.
      Its single-stream decode (28 tok/s at 12K with DSpark) is level
      with ours, and its 59 tok/s is aggregate over 12 batched requests.
      All creator-reported
      ([forum](https://forums.developer.nvidia.com/t/1x-spark-deepseek-v4-flash-0731-1-000-tok-s-prefill-59-tok-s-multi-agent-serving/378855)).
      The study:
      - pin ds4 and its GGUF and measure it on a Spark;
      - import the same GGUF into jitLLM (adding IQ2_XXS products if
        needed) so the comparison is same-format;
      - profile a prefill chunk in both engines;
      - adopt and generalize the techniques that transfer, across
        models, not only DeepSeek.

      If it holds up, ds4 becomes a DeepSeek baseline beside llama.cpp,
      a same-format comparator on its GGUF.
      *Study completed 2026-09-29* ([report](experiments/ds4-study/README.md)):
      source/GGUF pinned, the same IQ2_XXS/Q2_K weights and F16 APE tables
      imported and run natively, fresh-process cold rungs to 128,817 IDs
      measured, and matched 4K profiles captured. Shared sparse query
      gathers and paired expert preparation retain F16 caches and their
      original weights; native 8K prefill changes from 500 to 625 tok/s.
      Default ds4 has FP8 KV/FP4 indexer caches; its F32-storage control
      retains those rounded values and leaves a large product gap too.
      Full same-GGUF reference quality qualification is pending. Wide
      sparse dispatch is opt-in for measured
      fast shapes because disjoint D512 lists regress. Cross-model and
      format controls are in the report. Compact expert-major scheduling
      is compiled and adopted for DeepSeek fast prefill from 2,048 rows:
      alternating 8K controls improve throughput 11.1% on the community
      weights and 9.8% on the original, with identical captured logits.
      Smaller-format regressions keep generic and exact defaults off;
      ds4 is a same-weight harness comparator, not a cache-equivalent
      quality or runtime/DSpark baseline yet.
      The owner's [cross-family inventory](optimization-inventory.md)
      covers existing implementations and reusable pieces of rejected
      kernels, so each new experiment checks prior consumers and outcomes.
      Its [current implementation snapshot](experiments/optimization-inventory/README.md)
      includes unadopted kernels' pieces and shared-request transfer bounds.
      A [direct MIT Q2_K product port](experiments/ds4-q2-d2r/README.md)
      uses the same captured native input and raw weight blocks without
      a permanent SoA copy: charged product latency falls about 26%,
      with NMSE 2.13e−7 versus compact MMQ. Fresh same-source community
      8K controls gain 3.68% mean prefill throughput and preserve all
      128 greedy IDs. It remains benchmark-only and off by default;
      long greedy, PPL and state/restore gates are owed before adoption.
      A separate [literal HCA core port](experiments/ds4-hca-tokentile/README.md)
      retains native F16 state and charges mirror/record preparation.
      Same-input attention latency falls from 89.03 to 7.06 ms at 4K
      rows and 46.56 to 3.56 ms at production-sized 2K rows. Separate
      fresh 8K ABBAs gain 17.88% / 15.76% prefill throughput, with identical
      128 IDs and exact own full-logit repeats. It remains benchmark-only
      and default off. Late 2K operands with 1,024 compressed cells also
      pass the operator controls, but two fresh original-GGUF 32K model
      repeats fail the fixed greedy bound at the same step 249:
      2.616249 versus 0.947 nats, 1.669249 nats over. Better isolated
      arithmetic does not close model quality; the default remains unchanged.
      The mixed eligible 128K selection passes fixed-window PPL at
      1.927031 versus 1.9298 (−0.1435%); the greedy failure still blocks adoption.
      A separate [frontier-head follow-up](experiments/dsv4-frontier-head/README.md)
      preserves all state/DSpark streams while selecting only the target
      head rows production consumes. Original Q4_K controls gain
      0.8–1.2% prefill throughput with the final HCA component; the planned
      1M workspace is unchanged.
      A faithful-floor 32K all-wide control reproduces an outside-bound
      oracle row, unchanged with compact scheduling off. Keeping CSA/window
      sharing while selecting ordinary MMA for count-based HCA passes fresh
      32K/128K all-head/frontier repeats at the unchanged 0.947 bound;
      matched 128K PPL is 1.926517 versus 1.9298. Production state,
      forced rejection and swap continuations remain exact. Fresh sampled
      plain/speculation TV is 0.0034 / 0.0098 / 0.0186 / 0.0112, below the
      unchanged 0.1 bound. Subsequent maximum-context timing, neutral
      retrieval and exact continuing-context swaps are recorded in the
      [final context checks](experiments/m3-final-context/README.md).
- [ ] **Complete ds4 performance reference and native restoration**
      (the owner, 2026-09-30). First reproduce the complete original
      pipeline inside jitLLM on the same community GGUF, precisions,
      context, chunks and output cadence; measure it independently of
      adoption's quality gates. Then restore native stages one at a time
      to bisect the speed difference, including charged preparation,
      weight/state layouts, launch boundaries and interacting stages.
      Adapt the mechanisms that explain the gains into our architecture
      and assess applicable consumers across models/kernels. The literal
      path is temporary benchmark scaffolding, not a permanent runtime;
      retire unused reference code from shipping builds after the study.
      [Complete reference and restoration evidence](experiments/ds4-complete-plan/README.md):
      the native 8K pipeline completes all 43 layers and matches all
      129,280 original logits byte for byte on the same community weights
      and IDs. Three fresh passes deliver 1,085 tok/s on Spark A versus
      original ds4's 1,112 tok/s on Spark B, including the final result copy,
      a 2.4% throughput gap. The first [native consumer restoration](experiments/ds4-native-outputb/README.md)
      replaces output-B with current GGML MMQ over identical original
      operands: rate ratio 0.9983, twelve complete heads and three complete
      operator outputs byte-identical. Its final Spark slice passes 1,199
      tests, including 256 GPU tests. The first [routed-FFN factor](experiments/ds4-routed-ffn-first-axis/README.md)
      compares paid producer/gather and fusion/storage: Materialized is
      2.35% slower than Direct, with all twelve final heads and complete
      captured down outputs byte-identical. Keep Direct. The [matched 32K
      pipeline](experiments/ds4-matched-32k/README.md) now matches every final
      logit byte on the same Spark and is 0.71% slower, including actual
      deep CUB selection. The [captured native IQ2 restoration](experiments/ds4-native-iq2/README.md)
      runs current compact gate/up plus the original suffix at 0.8659×
      Direct / 0.9201× Materialized operator speed, with small complete
      numerical differences and a paid duplicate map adapter. Keep Direct;
      the [preparation capture](experiments/ds4-native-iq2-prepared/README.md)
      now proves every native/original D4 payload byte equal. Borrowing
      either payload reproduces native gate/up exactly; producer removal
      changes resident pair time by about 4–6%. The [standalone same-D4
      consumer comparison](experiments/ds4-native-iq2-consumer/README.md)
      measures native at 0.880357× original resident rate. Its [J64 control](experiments/ds4-native-iq2-j64/README.md)
      preserves every output byte but is 4.92% slower than J128. Keep the
      current tile; isolate remaining loader/launch/arithmetic factors before
      restoring another piece. The [shared-worklist control](experiments/ds4-native-iq2-worklist/README.md)
      reduces two gate/up worklist builds to one with unchanged J128
      consumers and byte-identical complete outputs, but gains only 0.20%
      amid 0.15–0.34% bookend movement. Keep the separate path; no adoption
      follows. The later [occupancy-two screen](experiments/ds4-iq2-occ2/README.md)
      selects a distinct J64 compiler specialization: captured paired products
      gain 28.05%, and genuine native 8K prefill gains 4.15%, with all six
      complete heads byte-exact. Production dispatch retains generic
      configurations and is restricted to the measured GB10 paired shape;
      its coherent Spark build, unit/style/boundary checks and final compiled
      golden/state confirmation pass, as do REUSE/header checks.
      These comparisons cover one request.
      The [matched-input native
      stage profile](experiments/ds4-production-prefill-attribution/README.md)
      records 13.31 s at 8K with byte-equal complete heads and 0.49% ordinary
      bookend movement. The exact post-down weighting and six-slot sum occupy
      665 ms, about 5% of wall, selecting one focused reduction screen without
      claiming a whole-model gain or attributing the remaining literal gap.
      That captured-operand reduction now runs at 4.19× the ordinary rate,
      with every complete output byte unchanged. Its bookended complete 8K
      substitution gains 4.30% in tokens/s with all six full heads byte-equal.
      The selected native graph operation retains the original F32
      multiplication/addition order and exact/small-row/unsupported fallbacks;
      its final Spark check passes 1,245 tests, including 258 GPU tests.
      The [native Q-head fusion](experiments/dsv4-qhead/README.md) combines
      per-head RMSNorm and normal-tail RoPE: its corrected 8K screen gains
      3.17% with all six complete heads byte-identical. The guarded fast
      graph removes the norm intermediate; reference and unsupported
      shapes retain both primitives.
      The private [output-A prefix factor](experiments/ds4-output-prefix/README.md)
      gains 4.08% community-checkpoint 8K prefill throughput with packing,
      table preparation and copy-back paid. At 32K on the original checkpoint,
      its 512-row fixed-history comparison has zero outside-bound differences
      and all full logits repeat exactly. The same native baseline in that
      screen has one exception exceeding the fixed bound by 1.669249 nats.
      Adding the optional HCA attention to that output-A factor also has
      zero outside-bound differences, with all full logits exact on a fresh
      repeat. Likelihood on the same oracle-generated continuation is almost
      native's. The actual mixed 128K path passes registered held-out
      perplexity at 1.92884 versus 1.9298 (−0.05%) and preserves the frozen
      127K-token variable-binding answer exactly. The 128K forced-greedy
      control also passes: 500 exact choices, 12 within-bound differences,
      no outside/unresolved rows and exact agreement at step 315.
      The guarded native output-A graph operation then gains 4.97% in one
      paid 8K OFF/ON/OFF screen, matching the qualified private candidate's
      complete logits exactly and reducing required scratch. Its selected
      Spark slice passes 1,248 tests, including 259 GPU tests, plus format,
      tidy, boundaries, REUSE and headers. It remains default-off in the
      benchmark at 4096 rows. Genuine runner integration now preserves the
      qualified full heads, initialized state and ordinary one-row continuation;
      its allocation-only 128K control funds all 32 full-chunk plan shapes.
      It remains default-off, with position authentication before dispatch
      and passing Spark unit/style/boundary and final compiled head/state
      controls, plus REUSE/header checks. A paid native attention interaction at that
      geometry adds 21.19% throughput, reducing 8K prefill to 9.879 s.
      The private 2048-row extension gains 3.95% with HCA already on,
      with 0.18% bookend movement, but fails one original-checkpoint 32K
      fixed-history row: 2.616249 nats, 1.669249 above the unchanged bound.
      That extension is stopped; the passing 4096-row candidate stays separate.
      The [literal input-preparation inverse factor](experiments/ds4-attention-preparation/README.md)
      attributes 166.9 ms / 2.21% to fusion/reuse with exact complete heads;
      this is not a native integration gain. A separate [native flat-RMS
      launch-size screen](experiments/ds4-flat-rms/README.md) loses 1.30%
      and changes logits, so 1024 threads remain selected. The [HC projection's
      F32 accumulation](experiments/ds4-hc-projection/README.md) is neutral
      (+0.25%) and changes heads; 16F stays.
      The [ds4 stage mechanisms](experiments/ds4-prefill-stages/README.md)
      bring native 8K prefill to 7.64–7.74 s against literal ds4's ~7.57 s
      at the 4096-row community geometry: eight byte-exact changes plus the
      perplexity-qualified D2R down product. They are now the fast plan's
      defaults, each under its own shape guard. Runner heads, state and
      continuation are byte-exact at 2,048 rows on both checkpoints, and the
      original-checkpoint 32K fixed-history control is byte-identical. At
      production's 2,048-row chunks they gain 7.21% on the community
      checkpoint but 1.41% on the served original checkpoint, where then
      only F16 Q and dense Q8_0 pairs applied; runtime 8K prefill gained
      1.26%. Since 2026-10-03 they also take partial chunks and the
      original's F32 HC, IQ2_XS and K-quant shared-expert types (runtime 7K
      +3–4% on both,
      [partial chunks](experiments/ds4-prefill-stages/README.md#partial-chunks-and-other-quant-types)).
      Production DeepSeek chunks are now 4,096 rows, 12.1% faster at 8K
      than 2,048 ([DeepSeek concurrent](experiments/deepseek-concurrent/README.md)).
      These different-factor screens are not added together or recorded as
      a matched final ds4 speed ratio. The [closure status](m3-optimization-status.md)
      tracks completed and missing independent-review experiments.
      Broader quality remains open; output-A/HCA stays default-off.
      The [fixed
      long-context answer tests](experiments/ds4-long-context-tasks/README.md)
      complete four OFF/ON pairs without a clear candidate-specific answer
      regression, but both paths fail parts of the strict rubric. HCA stays
      off by default; these tests do not approve a quality exception.
- [ ] **Concurrent-request comparisons** (the owner, 2026-10-01).
      Engine comparisons cover one request and concurrent generation
      requests, beginning with 1, 2 and 4 on the same resident model.
      Compare actual aggregate throughput and each request's latency,
      completion, output and memory, with matched prompts, contexts,
      precision and speculation settings. TensorFold's explicit CUDA
      concurrency mode is included; cross-quantization results remain
      labelled. [Protocol](experiments/concurrent-requests/protocol.md).
      Qwen chat now shares two active requests; other families retain the queue.
      Solo performance does not establish concurrent parity. Measure the gap
      and use it to
      prioritize the continuous-batching work below, advancing the
      necessary implementation when the comparison requires it.
      The [first 8K screens](experiments/concurrent-requests/README.md)
      record TensorFold's 28 requests with exact solo controls: fresh-burst
      throughput scales 1.78× plain and 1.28× speculative at four requests,
      with serial prefills. Mia's speculative screen completes 14×256
      outputs and scales 23.44→40.26 tokens/s (1.72×), with all seven
      greedy repeats differing under DET=0; no quality pass follows.
      The native queue screen completes all 28 outputs with exact text,
      usage and finish against fourteen solo controls. Its aggregate rate
      stays near 20.2 tokens/s plain and 27.3–28.5 speculative at C1–C4,
      below both speculative references at C2/C4. Plain timing is retained
      descriptively after a post-shutdown memory-sampler failure; the
      separate speculative deployment completed cleanly. This measured
      gap advances native continuous batching. TensorFold's repeated
      strict-prefix warm screen completes all 63 requests and 21 exact fresh
      controls: median C1/C2/C4 rates 57.53/82.17/109.81 tokens/s. C4 varies
      14.01%; its post-shutdown sampler error leaves memory unqualified.
      Mia's warm screen also completes 63 requests: median C1/C2/C4 rates
      37.43/56.59/82.04 tokens/s, with 6,656 cached prompt tokens and a
      1,536-token tail versus TensorFold's one-token tail. All 21 greedy
      warm/fresh outputs differ under DET=0. Two outer bookkeeping errors
      leave the retained results descriptive, without memory or quality
      qualification. Native warm, Mia plain, stable decision-relevant concurrent
      controls and the longer ladder remain open; these screens do not
      establish concurrent parity.
      DeepSeek's short service screen returns all calibrated solo/C2 inputs
      and 64-token replies. Resident-weight C2 rates are 12.45 tokens/s
      in ds4 and 7.60 in the native queue, with different cache/chunk
      profiles; this selects batching work without qualifying a matched
      pipeline or quality comparison.
      The [private native C2/C4 mechanism controls](experiments/qwen38-request-batching/README.md)
      pair row-local products over independent request states: paid decode
      throughput improves 19.1%/19.6%, with exact IDs, acceptance and final
      state/cursors. These private results do not establish HTTP or
      cross-engine parity.
      The separate [shared BF16 target-head control](experiments/qwen38-target-head-sharing/README.md)
      adds 8.87% paid C4 decode throughput on that paired-product path, with
      byte-identical complete target vectors and exact IDs, acceptance and
      state. Their native Slot-wave/serving integration is described below.
      The first native Slot-wave C2 screen raises paid decode from
      41.13 to 45.21 tokens/s (+9.91%), with exact IDs, traces and range geometry.
      This single direction screen selects focused recovery controls and serving
      integration; it does not qualify state-page equality or HTTP parity.
      Focused rejected-request recovery and HTTP capacity/cancellation controls
      now pass. The first matched fresh cooperative HTTP C2 screen is neutral
      (+0.34% aggregate rate); fixed3 restores pairing but remains neutral
      (−0.28%). Skipping unchanged selection rebuilds gives +2.92%, and the
      same binaries with warm weights give +2.60%; first completion worsens.
      Separate cooperative head OFF/ON and reversed ON/OFF screens gain
      +5.40%/+4.99%, with exact full native vectors and HTTP responses.
      This selects compatible BF16 head sharing for production integration
      and checks; fixed3 observations do not qualify adaptive traffic or
      cross-engine parity. Retain the production adaptive policy.
      The [exact16 MXFP8 control](experiments/qwen38-mxfp8-sixteen/README.md)
      passes every output/state check but makes the paid four-request loop
      7.469 times slower. Keep paired-eight products; no wider MXFP8 kernel
      or selector change is adopted.
      The internal cooperative API driver and shared resumable generation
      session prepare that integration. Qwen now has four independent native
      request slots under one shared model owner; a real-model control passes
      64 output and 82 full-state comparisons, graph replay and all-slot
      swap/return, with 1236 Spark-native tests passing. Per-request admission,
      cancellation and proven retirement have fake-backend controls. The
      production node backend now runs two Qwen chat requests in shared
      target/draft waves, with four retained branches, separate adaptive
      policies and prefix reuse. Compatible three- or four-row BF16
      target heads share an ordinary six-, seven- or eight-column product;
      compatible multirow HC BF16 products share their original cuBLAS path.
      The [combined screen](experiments/qwen38-combined-sharing/README.md)
      gains 2.76% paid C2 HTTP throughput with exact native outputs, state
      and capped replies; all 1,253 locked Spark tests pass. Funding four
      active slots loses 13.16% in a short C4 screen, so two remain selected.
      A separate [conditional depth plus four-head screen](experiments/qwen38-conditional-heads/README.md)
      gains 7.05% against exact normal C4 serving, with 1.34% bookend rate
      movement. It delays first completion by 83.23%, raises median latency
      15.70%, funds 3.27 GiB more fixed memory and changes complete replies.
      Keep the current default while mechanism and depth-policy controls
      are checked separately. The separate four-head mechanism now passes
      13 complete heads and 20 initialized-state/cursor comparisons, including
      discard, retry and continuation; policy and sampling remain unqualified.
      A separate [four-request captured GDN factor](experiments/qwen38-gdn-cohort/README.md)
      lowers paid operator replay wall 26.51%, with exact F32 outputs and
      unchanged operands/state/guards. Graphs are disabled and both arms
      use the same private arithmetic-body refactor. Native wave integration,
      graph/recovery controls and actual serving impact are untested.
      All packing
      is paid. Overflow waits
      for a retired slot, while model changes and literal completions drain
      the group. Independent request failures retain completed peers; native
      fences prove retirement before releasing a frame. Matched comparison
      and final gate qualification remain open.
      Prompt preparation now uses a branch-owned resumable session: host-only
      admission, separate reuse/chunk/checkpoint units and cancellation that
      preserves the completed prefix. The scalar path drives the same core;
      Qwen serving interleaves these prompt units with ready peer decode.
      The [four-request waves](experiments/qwen38-four-request-waves/README.md)
      replace fixed pairs with groups of up to four requests (16-row joined
      products on bit-exact wide MXFP8 and expert-major routed kernels,
      2048-cell wave alignment for graph replay, depth 2 when shared):
      matched 8K HTTP C4 +20% and C2 +9.8%, C1 unchanged with an identical
      reply. Native leads current TensorFold NVFP4 at C1/C2, is level at C4 and trails
      legacy Mia 18% at C4; replies under concurrency vary with arrival timing.
- [ ] **Long context** (the owner, 2026-09-29: coding clients run at long
      context by default, so M3 measures and fully optimizes it, not only
      8K). Each LLM runs a context ladder of 8K, 32K, 64K and 128K, then
      its maximum on one Spark: Qwen3.8 to its configured 262,144, and
      DeepSeek V4 Flash toward its trained 1,048,576 (YaRN, 16× over
      65,536). The largest that fits beside its weights and drafter is
      measured and becomes the documented and configurable ceiling.
      [Final context checks](experiments/m3-final-context/README.md)
      record the completed compact-target quality and final runtime
      checks. The corrected HCA/frontier runtime's 32K–256K ladder completes
      all 512 outputs per mode: prefill 1.88–1.93× llama.cpp, plain decode
      1.15–1.33× and DSpark 1.12–1.40×, peak memory 1.025× / 1.015×.
      Final 1M timing also completes 512 outputs per mode: prefill
      2.18–2.27× llama.cpp, decode 2.01× plain / 1.88× DSpark, memory
      within 1.012×. Neutral completed-answer retrieval passes 8K–256K
      in both modes and at 1M capacity with DSpark. Qwen timings, neutral
      retrieval and both heads' 128K/maximum continuing swaps are complete.
      DeepSeek's direct maximum swap also passes: both 1,048,512-token
      prefills and 64 restored outputs/logits exact, longest chunk 14.72 s,
      spill 7.33 GB and restore 0.583 s. Retain the 100/5 tok/s watchdog
      floors; Qwen's MTP speed gap remains open.
      The [Qwen policy/phase controls](experiments/qwen38-policy-phase/README.md)
      keep that gate open: all four 32K head/depth combinations trail the
      faster fresh Mia control, while selected/adaptive is best at 64K.
      Verify accounts for 86–88% of decode time; these unchanged-harness
      diagnostics change no default or quality bound.
      - *Baseline first:* prefill throughput and decode speed at each
        depth, speculative and plain, through the runtime and against the
        same-format comparators at the same depths:
        - llama.cpp for DeepSeek, at a pin that includes upstream's sparse
          flash-attention prefill (#29298), so we are judged against
          upstream's best long-context path;
        - Mia's vLLM for Qwen3.8;
        - TensorFold beside them as cross-quantization information.

        The gaps found set the optimization work.
        *Baseline measured 2026-09-29*
        ([long-context](experiments/long-context/README.md); llama.cpp
        b11254 built by us, Mia's vLLM, jitLLM through the runtime; a
        per-kernel profile at 8K, 32K and 64K). The comparators stay
        nearly flat with depth; jitLLM's per-token cost grows with the
        whole context: DeepSeek prefill 333 / 230 tok/s at 32K / 64K
        (1.16× / 0.84× llama.cpp's), decode 14.7 / 10.8 (0.78× / 0.60×),
        with DSpark 31.3 / 21.4 (1.02× / 0.74×);
        Qwen3.8 prefill 1.19× Mia's at 32K, 0.70× at 64K, 0.27× at its
        262,144 maximum, decode 0.89× to 0.29×, and MTP refused past
        32,768. The cause is dense attention over every cached cell
        (DeepSeek's full-size window cache concatenated in every layer;
        Qwen3.8's masked attention and its GGML selection fallback past
        8,192 blocks), not the architectures' O(n) indexers (about 1% at
        64K). Maxima: Qwen3.8 262,144 (verified; MTP 32,768), DeepSeek
        262,144 plain (the configuration's bound, 1.3 GiB inside the
        guard) and 143,360 with DSpark. Correctness at 32K and 128K
        passes except one DeepSeek step at 32K (2.62 nats, subsequently
        diagnosed as near-tied routing/indexer noise; phase 2 agrees with
        the oracle at that step). Phase 1 DeepSeek does not repeat at
        32K (RE-031). Fixed to
        measure: RE-037, RE-038. The ranked gap list and fix plan are the
        report's; runs past 64K stopped there (the owner).
      - *Optimization, until every depth is at least the comparator's
        speed* (not only inside D-085's 10%). *DeepSeek done 2026-09-29*
        ([phase 2](experiments/long-context/README.md#phase-2-deepseek-flat-with-depth-2026-09-29)):
        its window cache a ring, attention gathering only the window and
        the selected or visible compressed rows, jitLLM's deterministic
        indexer (RE-031 closed for it) and the guard at 6 GiB with host
        inputs counted. Through the runtime, prefill 471 / 466 / 445 tok/s
        and plain decode 21.6 / 21.2 / 20.5 at 32K / 64K / 128K (1.65–1.72×
        and 1.15–1.22× llama.cpp b11254), DSpark 1.11–1.22×; the decode
        step's slope 0.026 ms per 1K tokens (was 0.83), the rest the
        indexer's O(n) scoring and selection; 262,144 fits with DSpark;
        swaps exact with a wrapped ring (the swap table with 32K saved,
        and rollback across a swap).
        Qwen3.8's items below are the next slice:
        - tiled, deterministic QSA selection past 8,192 blocks
          (TensorFold PR #93's technique), which also closes RE-031's
          long-context nondeterminism; *done for Qwen3.8 2026-09-29*
          with its block keys cached and attention over the kept cells
          alone: per-token cost flat to within the indexer (plain decode
          26.8 → 24.4 tok/s from 8K to 256K, prefill 2,314 → 2,178), at
          least Mia's speed plain at every depth, repeatable, MTP to
          262,144 ([long-context phase 2](experiments/long-context/README.md#phase-2-qwen38-flash-next-flat-with-depth));
          open: the final 512-output runtime ladder's default-prefix MTP
          reaches 39.12 / 37.55 / 45.03 / 35.63 tok/s at 32K / 64K /
          128K / 256K, versus Mia's MTP 3 at 37.25 / 40.45 / 48.65 /
          37.71. The selected head reaches 37.00 / 41.32 / 42.88 / 43.84;
          neither closes every depth's speed gate. Fixed depths 3–5,
          expert sharing, prepared Q8 and incremental confidence stopping
          were neutral. The bounded cuBLASLt library-product sweep was
          also neutral and left production unchanged. A bounded GB10
          MXFP8 schedule for 48/512/640-output products improves the
          fixed-depth-three 128K harness by 0.8% / 1.3% with prefix / curated
          heads, all 512 tokens identical to matched controls, but does
          not close the final adaptive runtime ladder's speed gate
          ([MXFP8 scheduling](experiments/qwen38-mxfp8-scheduling/README.md),
          [draft-head study](experiments/qwen38-draft-head/README.md)).
          The subsequent paired-prefix depth estimator regresses the
          default head, and the complete small-column tensor-core path
          is slower on every measured shape; both remain unadopted
          ([depth and crossover](experiments/qwen38-depth-crossover/README.md)).
          The bounded NVFP4 expert CTA schedule improves isolated down
          products but remains neutral with prefix / slightly slower with
          curated on the adaptive 128K trial; production stays unchanged
          ([NVFP4 scheduling](experiments/qwen38-nvfp4-scheduling/README.md)).
          Producer-fused down-Q8 export then passes exact intermediate,
          payload and graph controls, but gains only 2.4–2.9% on overlapping
          complete pairs and slightly regresses disjoint routes; no
          production fusion or model trial is justified by that result.
          Fresh same-ID 128K Mia controls vary 44.87/38.10 tok/s; the older
          ladder above remains historical and neither new run closes every
          depth's gate. Captured draft-head input RN is neutral and library
          heads with F32 or actual BF16 output are about 41% slower.
          Small-verify FP4/BF16 grouped experts lose 19%; the actual Mia
          cooperative tile/stage pair loses 33%, with native dynamic
          activation scaling retained. Both prototypes are removed
          ([draft input](experiments/qwen38-draft-input/README.md),
          [grouped factor](experiments/qwen38-grouped-verify/README.md),
          [fresh reference](experiments/qwen38-mtp-speed/README.md)).
          *Adaptive depth and selected-head
          import implemented 2026-09-29:* native depth 2–3 chosen by observed
          acceptance, deterministic across saved/restored runs. The 128K
          128-output harness trial reaches 44.3–45.0 tok/s with the prefix
          head; the externally supplied curated list reaches 41.4–41.9,
          so the prefix stays default. Curated 256K, rejection and swap
          controls are included in the study; the measured final runtime
          timing leaves the extrapolation speed gate open. The checked
          final runtime repeats complete all 512 outputs at every rung;
          prefix 39.38 / 37.62 / 45.33 / 36.73 tok/s, selected
          37.06 / 41.65 / 43.80 / 44.07. All fifteen neutral retrieval
          requests and both heads' exact 128K/maximum continuing swaps
          pass ([final context](experiments/m3-final-context/README.md)).
          The fresh same-ID two-pass Mia ladder completes all eight
          512-output requests: 42.23/38.62, 39.70/39.65, 39.45/41.35,
          42.89/38.77 tok/s. Prefix beats both at 128K but trails both at
          64K/256K; selected trails both at 32K and beats both elsewhere.
          Neither closes every rung. Generated reference histories vary;
          the fixed-depth-three FP8/BF16 cache settings remain distinct
          from native adaptive F16/F32. The actual installed FlashInfer
          complete routed consumer, including private static quantization,
          both products and finalization, is neutral/slower on twelve real
          T4 histories: traced fused +2.32% cold latency / −0.21% warm,
          ordered +1.75% / +1.73%. Fused finalization also varies on own
          repeats; no production port or full-model candidate follows
          ([complete consumer](experiments/qwen38-fi-down-stage/README.md#complete-consumer-follow-up)).
          The [literal MXFP8 projection](experiments/qwen38-mxfp8-consumer/README.md)
          also remains unadopted: on eight real T3/T4 inputs, Mia's traced
          tile/orientation costs 6.2–6.3% more than native with conversion,
          quantization and widening included. The fresh public API default
          selects a different orientation and is about 3× slower. These
          single-request operator results do not explain the whole decode gap.
          The remaining decode gap and reference-quality qualification
          remain open;
        - sparse flash-attention prefill for both models (llama.cpp
          #29298 and #28770);
        - DeepSeek's compressed attention and indexer at depth;
        - decode at depth (KV read bandwidth, graphs at long shapes);
        - the prefill chunk policy at depth.
      - *State that grows with the conversation*, not reserved at the
        ceiling, so a long ceiling costs memory only when used. Spill and
        restore move only the used state. *Implemented 2026-09-29*
        ([growing state](experiments/growing-state/README.md)): stable
        virtual addresses, initialized extents only, sparse spill and
        packed controls; both models registered at 262K, swapping exact
        8K state under 10 s. Subsequent maximum-context timing and
        continuing-context swaps are recorded in the final-context report.
      - *Turn-to-turn reuse at long context:* a coding agent resends the
        whole conversation each turn. The runtime reuses the longest common
        prefix of the previous turn's state, including when a client drops
        or rewrites earlier reasoning, rather than re-prefilling. The
        recurrent and indexer state keep checkpoints at turn boundaries
        where rollback to a prefix needs them. *Implemented 2026-09-29*
        ([turn reuse](experiments/turn-reuse/README.md)): two private disk
        checkpoints before the assistant opening, exact rollback and
        continuation across full swaps; at 64K, 0.51 s DeepSeek and
        0.42 s Qwen suffix preparation versus 141.6 / 29.1 s fresh.
      - *Defaults (implemented 2026-09-29):* context defaults to 262,144
        (was 8,704), with a generic cap of 1,048,576 and the checkpoint's
        trained ceiling checked before allocation: DeepSeek 1,048,576,
        Qwen3.8 262,144. Growing state keeps the physical guard; final
        DeepSeek timing and retrieval fit at 1M capacity with DSpark.
        HTTP bodies allow 16 MiB and message
        text 8 MiB, while aggregate intake bounds, chunk policy and schema
        version 2 stay unchanged (D-096, D-097).
      - *Request deadline (done 2026-09-29):* the route's fixed 600 s
        deadline is gone, so a long prefill no longer fails on the clock: a
        progress watchdog, scaled non-streaming deadlines and no deadline
        for streams (D-097's owner note,
        [runtime-serving.md](runtime-serving.md#progress-and-deadlines)):
        DeepSeek's 128K prompt, which the deadline stopped at 108,544
        tokens, now streams to its end (845.8 s on `spark`).

**Exit criteria:**

- **Swap time,** in a running process, with page-in, setup, graph and
  tuning restore and A's state spill and restore counted. Defaults are
  owner-adjustable (D-087):

  | Item | Rule |
  | --- | --- |
  | Pairs | Every ordered pair of the three models, each run as A→B→A: DeepSeek↔Qwen3.8 both ways, and each LLM↔Qwen-Image both ways. Each swap is reported separately; the headline is the worst LLM↔LLM swap |
  | Saved context | An LLM A holds 8K tokens of conversation; its KV, recurrent and indexer state are spilled on swap-out and restored on return. The bound applies here; a 0-context swap is reported too |
  | Graphs | Previously prepared: B ran earlier in this process, and its graphs and tuning are restored. First use: nothing is prepared for B in this process. The ~10 s goal and the ~20 s bound apply to previously prepared swaps; first use has its own target, no worse than about 2× the bound (~40 s) |
  | LLM endpoint | From the swap request to B's first generated token for a short prompt |
  | Image endpoint | From the swap request to the pipeline ready: the first denoising step's output produced. Full generation for a fixed prompt, size, step count and seed is timed separately under the Image criterion |
  | Also per swap | Bytes read, read throughput, peak memory, and each part of the swap time |

  The report gives each swap against the ~10 s goal and the ~20 s bound,
  and beside the baselines ([measured](experiments/fast-swap/baselines.md),
  cold page cache, one run each): Mia's vLLM Qwen3.8 at 13 min 13 s from
  start to first token (11–14 min creator-reported), TensorFold at 141 s
  at `beddbb7b` and 143 s at `71377a53` (about 90 s creator-reported),
  the pinned llama.cpp at 77 s from DeepSeek 0731 to Qwen3.8's GGUF and
  104 s back with A's 8K state restored (75–93 s per switch in M0's run
  of the older revision), and diffusers at 212 s from process start to
  Qwen-Image's first denoising step.
- **LLM correctness:** on a short prompt set, greedy tokens match the
  model's same-format oracle (table above), with small logit differences
  allowed, and perplexity on a fixed text is within a few percent of the
  oracle's. Greedy decoding with speculation gives the same tokens as
  without, except near-ties (the bound: the 99th percentile of the
  engine's own kernel noise between two of its paths, neither the
  oracle, recorded before the comparison).
  *Amended 2026-09-28 (the owner: speed before bit exactness, D-085's
  note): was "the same tokens as without"; the bit-for-bit form is the
  optional reference mode's check.*
- **Speculation correctness** (moved from M9, D-068), per drafter:
  - forced draft rejections at varied positions, all-reject and partial
    accept among them, leave no stale KV, recurrent (Gated DeltaNet or
    linear-attention), drafter or MTP, or indexer state: rollback stays
    exact in effect, checked against a control that drafted only the
    accepted tokens (the reference mode, bit for bit) or, on the default
    fast path, by every state byte outside the accepted rows' writes
    being as it was before the verify;
  - after each rejection, the output equals non-speculative greedy
    decoding's except near-ties (the same bound, from the verify's rows
    against one-row decoding); bit for bit, in the same engine with the same kernels, is an
    optional reference-mode check. *Amended 2026-09-28 (the owner, D-085's
    note): was "bit for bit" on the default path.*
  - rollback composes with swap: after rejected drafts, A is swapped out
    mid-conversation and restored, and continues exactly as the unswapped
    control;
  - where sampled speculation is enabled, it preserves the token
    distribution: on 4 fixed prompts, seeds 0–255 and the first 8
    generated tokens (8,192 sampled tokens per mode, sized to stay within
    D-085's 10 minutes), each prompt's pooled histogram over its 16 most
    frequent tokens plus an "other" bin is within a total-variation
    distance of 0.1 of plain sampling's (about twice the sampling noise
    expected at this size, estimated).
- **Swap correctness:** our own swap and restore cycles are bit-identical:
  A resumed after B gives the same logits and tokens as A never swapped out.
- **Image:** for fixed prompts and seeds, output is within a simple
  image-similarity bound of diffusers' BF16 pipeline's; the pipeline swaps in
  and out with the text models; and full generation for a fixed prompt,
  size, step count and seed is not more than 10% slower than diffusers
  (D-085).
- **Performance and memory** (D-085): prefill and decode are not more than
  about 10% slower than the same-format performance comparator, and peak
  memory is at most about 1.1× the comparator's. Cross-quantization
  comparators are reported beside them for speed and memory, not gated.
  Each comparison names the comparator, its format and whether both sides
  speculated.
  *Owner addition, 2026-10-01:* report both solo and concurrent generation
  requests at matched concurrency levels, initially 1, 2 and 4. Include
  aggregate completed-token throughput, each request's latency, errors,
  admission limits and memory. Establish parity in both scenarios before
  claiming a performance gate passes; existing single-request results
  qualify only that scenario. Do not count streamed chunks as tokens.
- **Long context** (the owner, 2026-09-29):
  - **Speed:** at 32K, 64K, 128K and each model's measured one-Spark
    maximum, prefill and decode (plain and speculative) are at least the
    same-format comparator's speed at the same depth, where the
    comparator can run that depth. Where it can't, the result is reported
    alone.
  - **Scaling** (the owner, 2026-09-29: "I'd love for our baseline result
    to be that it doesn't degrade at all as the context size
    increases"): per-token prefill and decode cost stays flat with depth.
    It is fixed at 64K first, then shown to extrapolate to 128K and the
    maximum. Any remaining slope is only what the architecture requires
    (e.g. the indexer's O(n) scoring), measured and named. The phase-1
    baseline (2026-09-29) found DeepSeek halving from 8K to 64K (prefill
    463 → ~230 tok/s, decode 22 → 10.8), while llama.cpp's sparse path
    stays nearly flat (prefill 286 → 229, decode 18.8 → 14.7 from 32K to
    256K).
  - **Correctness at depth:**
    - greedy tokens match the oracle except near-ties on long real
      prompts (code: a repository's files as context) at 32K and 128K;
    - perplexity on a long document is within a few percent of the
      oracle's;
    - a retrieval check (a fact placed at several depths, then asked for)
      passes at every rung up to the maximum;
    - the same long run repeats bit for bit in the same engine (RE-031
      closed).
  - **Swap with long saved context:** an LLM A holding 128K, and its
    maximum, is swapped out and back. Spill and restore time and bytes
    are reported, and the continuation is exact. The ~10 s swap goal stays
    defined at 8K; the long-context swap's target is restore at the SSD's
    measured rate.
  - **Turn reuse:** a multi-turn coding session at 64K+ re-prefills only
    each turn's new tokens, including with a reasoning model whose client
    drops earlier reasoning.
- A standard OpenAI-compatible client completes a chat with each LLM
  through the minimal route, swapping between them.
- The source-lock widening, the swap path and the chat route's request
  parser pass their adversarial challenge (heavy path).

**Open questions** (for the owner):

- The image pipeline, and a drafter that shares its target's tables, need
  manifest references to another artifact, with shared resources counted
  once ([artifact-format.md](artifact-format.md#deliberately-open)).
  *Settled by D-089 (2026-09-28, accepted under the owner's overnight
  delegation; the owner may amend):* one
  artifact per component and a composition naming them by ID; the
  pipeline is imported that way, and a drafter is to be a composition with
  its target. DSpark's binding is settled: its own artifact, binding the
  target artifact's token table and head at load (D-089's note).

## M3.5 — Model families  `pending`

Goal (the owner, 2026-09-29): build out the core engine across the major
open model families, MoE and dense, before the system is built around it
(M4 onward). Each family runs natively on one Spark from a prepared
artifact on M3's engine skeleton. Each is correct against its same-format
oracle, at least as fast as its same-format reference, and flat with
context wherever its architecture allows. A family that needs a new engine
mechanism exposes the gap now, while the engine is still cheap to change.

**Entry:** M3 exit, including its engine skeleton and its "adding a model
family" guide, and its long-context scaling work.

**Scope:**

- [x] **Family selection** (approved by the owner 2026-09-29: all 13
      checkpoints of [m35-families.md](m35-families.md), with the answers
      recorded there). The goal is
      capability coverage, not particular models (the owner, 2026-09-29).
      Survey each top-tier open family's current and older widely used
      generations, list every architectural feature they use, and mark
      which jitLLM already supports in a capability matrix. Then choose
      the smallest set of checkpoints that fit one Spark, each with a
      same-format reference engine, that covers every feature still in
      wide use. A feature only a too-large model uses is flagged. Record each pick's architecture
      class:
      - attention: dense, sliding window with dense global layers,
        compressed and sparse, or linear and recurrent;
      - MoE or dense;
      - positional scheme, normalisation and activations;
      - tokenizer and chat template;
      - MTP or companion drafters;
      - maximum context.

      Named by the owner: Gemma (Gemma 4 26B-A4B, M7's daily driver, and a
      dense Gemma 4), Llama, and MiMo. Already in the repo's plans:
      Ornith 1.5 35B-A3B (M7), Qwen3.8-27B (dense) and Nemotron. GLM-5.3
      Flash stays in M4 (two Sparks). Agents propose; the owner approves
      the list (AGENTS.md: agents never invent supported model
      combinations). Weight licenses are informational (D-087).
      Proposal, awaiting the owner: [m35-families.md](m35-families.md).
- [ ] **Per family**, on the engine skeleton, using the "adding a model
      family" guide, which M3.5 tests and corrects:
      - import to a v0 artifact;
      - its runner: plan, state layout and model-specific steps;
      - the native tokenizer and its chat template's rendering (native or
        interpreted, D-067), with the template hash
        recorded in the support matrix;
      - swaps in and out beside the M3 models;
      - speculation where the family ships MTP layers or drafters;
      - the D-053 rule: a primitive fallback for every fused operation.
- [ ] **Concurrent requests with continuous batching** (the owner,
      2026-09-29). The primary workload includes an agent plus
      subagents, which is several concurrent requests on the same
      resident model. The engine supports:
      - per-request state side by side;
      - decode steps that batch rows from different requests (plain and
        speculative);
      - chunked prefill interleaved with decode;
      - shared-prefix state with copy-on-fork for parallel agent
        branches;
      - admission within the memory budget.

      The chat route's one-request-at-a-time queue (D-097) becomes a
      batch scheduler. Batching applies to requests for the same model;
      different models still time-slice by swapping (D-019).
      M3's added concurrent-request comparisons now measure this gap;
      advance the necessary implementation into the optimization run
      where those comparisons require it (owner, 2026-10-01).
- [ ] **Skeleton gaps the M3 cleanup's review named**
      ([engine.md](engine.md)):
      - planning, capture and launch binding are GGML-only, so a family
        run on EXL3 (or another non-GGML backend) needs a step family of
        its own that plugs into the same skeleton;
      - runners assume one target and at most one drafter.
      M3's growing-state work has closed the setup allocation gap for
      DeepSeek and Qwen3.8: stable virtual regions, backing only when used.
- [ ] **Kernels:** operations new to a family come from GGML first, with
      our own kernels on measured need (D-053). Upstream findings go to
      docs/upstream/.
- [ ] **Quantization formats** (the owner, 2026-09-29), especially the
      variable-bit ones. A format coverage matrix sits beside the
      capability matrix. The covering set runs every format still in wide
      use:
      - GGUF K-quants, I-quants and dynamic per-tensor mixes;
      - MXFP4, NVFP4, MXFP8 and FP8;
      - AWQ and GPTQ;
      - MLX affine;
      - EXL3.

      Each is as fast as its same-format reference.
- [ ] **MLX affine import** (moved from M9 by the owner, 2026-09-29):
      import and run TensorFold's MLX 4-bit checkpoints (reconcile the
      group size, 32 or 64), starting with Qwen3.8 Flash Next's, so
      TensorFold is a same-format oracle and gated comparator (D-085)
      instead of cross-quantization information.
- [ ] **Legacy-tier features** (the owner, 2026-09-29): list the older
      generations' features that are not subsets of the covered ones
      (Gemma 2, Phi-3.5, Mistral 7B, Command R7B, Llama 3.2 and others),
      with which models and vendors used them and whether each was
      abandoned or just not updated. Implement those worth keeping.
      [Study](m35-families.md#legacy-tier-features). *Owner, 2026-09-29:*
      implement the seven small features:
      - classic SentencePiece;
      - linear RoPE scaling;
      - attention logit soft-capping;
      - LongRoPE;
      - LayerNorm with parallel attention and FFN blocks;
      - the older GGUF block types (Q4_0, Q4_1, Q5_0, Q5_1, IQ4_NL).

      Their four GGUF fixtures (Gemma 3 4B QAT Q4_0, Gemma 2 2B, Phi-3.5
      mini, Command R7B; about 17.9 GB, llama.cpp as reference) join
      M3.5's set under the same exit criteria. The deferred and dropped
      features stay as recorded. *Owner, 2026-09-29:* PrismML's Bonsai
      join too, both 1-bit and 2-bit, because they are hugely popular:
      - Bonsai-27B Q1_0 (with its Q4_1 drafter);
      - a 2-bit Ternary Bonsai.

      Which 2-bit build is settled at M3.5's start. Ternary-Bonsai-27B's
      Q2_0 on stock llama.cpp is unconfirmed, and Ternary-Bonsai-2-27B
      (on Qwen3.8-27B) needs PrismML's llama.cpp fork, so its reference
      is that fork, pinned and license-audited like any baseline. The
      stock-llama.cpp Q2_0 is preferred if it works; the Bonsai-2 card
      warns that rotated weights load silently and give garbage, so
      correctness is checked against its reference, not assumed.
- [ ] **EXL3 optimization** (the owner's particular interest): the
      trellis-encoded quants across their codebooks, bitrates and
      per-layer mixed widths, dense and MoE (grouped mixed-width routed
      experts). The target is faster than ExLlamaV3 on the GB10, with
      TensorFold's EXL3 path reported beside it, at decode and prefill.
      It builds on M2's native EXL3 linear (D-080).
      Study first: [TensorFold #42](https://github.com/ashhart/TensorFold/pull/42)
      (merged 2026-09-28, MIT). It reads EXL3 on CUDA for any codebook
      (3inst, mcg, mul1), 1–8 bits and mixed-K packs, with:
      - a row-invariant dense EXL3 linear (1–128 rows), bit for bit with
        ExLlamaV3's `reconstruct`;
      - coalesced low-bit word reads with prefetch;
      - one grouped kernel for mixed-width routed experts (no atomics or
        host syncs, graph-safe).

      It reports 1.5–1.6× ExLlamaV3 on Qwen3.8-27B and Flash Next, 1.5–3.6×
      on mixed-K expert layers, and drafted output equal to serial
      (creator-reported). Its prefill is untuned (decode kernels in 64-row
      chunks), and mcg and 3inst are verified on partial packs only.
      Measure it against ExLlamaV3 and us, then adopt what transfers
      (D-091).
- [ ] **Multimodal file inputs** (D-101, moved from M10 by the owner,
      2026-10-02): images (several per request), video files and audio
      files on every model here that takes them, brought up with its
      family. Live streams stay deferred (D-042). Encoders are components
      of the model's composition (D-089), paged and released like other
      extents ([facts and sources](m35-families.md#media-inputs-decision-models-and-generation-apis)).
      - **Qwen3-VL vision tower** (27 layers, patch 16, 2×2 merge,
        interleaved M-RoPE): Qwen3.8 Flash Next (from M3), Qwen3.8-27B,
        Ornith 1.5, Bonsai-27B, Clef and Clef-flash. Images and video
        (2 fps frame sampling, text timestamps). Qwen-Image-2.1's edit
        mode uses the same family of tower for its reference images.
      - **The other M3.5 encoders** as their pinned files carry them:
        - Gemma 4 26B and 31B, with images and video as frames; their
          image tokens are bidirectional on sliding layers only;
        - Gemma 3 4B (SigLIP);
        - Llama 4 Scout (tiles);
        - Mistral Small 4 (Pixtral);
        - Muse Glimmer (images; its video is unendorsed);
        - MiMo-V2.6 (images and video; the pinned EXL3 build dropped its
          audio).
      - **DeepSeek V4 Flash Vision-Exp** (images; kept by the owner,
        2026-10-02, with V4.1 Flash's vision following in M4): V4 Flash
        0731 is text-only, and the vision model is a separate,
        continued-trained checkpoint whose GGUF and encoder sit in antirez's pinned
        repository. Its row-pair "N-layout" fits CSA compression.
      - **Audio:** no pinned checkpoint keeps an audio encoder, so Gemma
        4 E4B-it joins as the audio carrier (owner, 2026-10-02): 16 kHz,
        128 mels, at most 30 s per clip, sharing Gemma 4's tokenizer,
        template and vision.
      - **Intake** on the chat route: Chat Completions `image_url`,
        `input_audio` and vLLM's `video_url`, inline data only, with
        remote URL fetching off by default. Byte, pixel, frame and
        duration bounds are fixed before decoding. The image, audio and
        video decoders are chosen under D-017 and D-080, and they and the
        intake take the heavy path.
- [ ] **Decision models over the Jev API** (D-101, the owner,
      2026-10-02): `POST /v1/systemone`, wire-compatible with TypeSafe's
      Jev/SystemOne OpenAPI spec, which the TypeSafe SDKs, gateways and
      Workers AI's Clef share.
      - **Test models:** `Cloudflare/clef` (Qwen3.8-27B backbone) and
        `Cloudflare/clef-flash` (Qwen3.5-9B backbone), each with a joint
        schema head of about 125M parameters.
      - **Decision program:** one prefill with no retained state. The head
        reads every position's final-norm hidden state and the output-head
        rows of each option's tokens. It answers all of a request's
        questions jointly, with images and videos as Clef accepts them.
      - **Wire behavior:** confidence follows TypeSafe's published
        formulas; Clef's reference reports the top probability instead,
        and that difference is recorded.
      - **Clients:** the TypeSafe Python and JavaScript SDKs, unmodified,
        pointed at jitLLM by base URL.
- [ ] **Media generation routes** (D-101, the owner, 2026-10-02):
      - `POST /v1/images/generations` and `/v1/images/edits` for
        Qwen-Image-2.1, in OpenAI's shape with vLLM-Omni's diffusion
        fields. Edits take reference images and a mask.
      - **Ming-Image-0.1-Design** (owner, 2026-10-03), a second
        text-to-image model to flesh out the image pipelines and API
        ([details](m35-families.md#generative-media-video-and-image)):
        a 6B DiT conditioned by a Bailing MoE multimodal encoder
        through a Qwen2 1.5B connector, and Qwen-Image's
        VAE with four channels, so it generates RGBA. Through
        `/v1/images/generations`, including OpenAI's
        `background: "transparent"`. Size, step count, guidance and seed
        are the caller's per request (OpenAI's `size`, vLLM-Omni's
        diffusion fields), as for Qwen-Image; the card's defaults (12
        steps, CFG 1.0, 2,048² or 1,024²) are the defaults and the tested
        points, not limits. Its components run as one composition (D-089), sharing
        the image phases, VAE and route code with Qwen-Image's rather than
        a second pipeline.
      - `/v1/videos` asynchronous jobs for MiniMax H3, on D-041's job
        machinery, with generated media kept only until fetched or
        expired.
      - `/v1/audio/speech` on two text-to-speech testbeds (owner,
        2026-10-02):
        - **Breeze-TTS-2:** 3.47B in BF16. A T5Gemma2 text encoder (26
          layers), a Qwen3 backbone (28 layers, 2048 wide), a depth
          decoder over 16 codebooks and a Mimi-style codec at 24 kHz.
          It does voice design from instructions, and voice cloning from
          a reference clip, which goes through the codec's encoder.
          Classifier-free guidance doubles its rows. Streaming output.
        - **Kokoro-82M:** StyleTTS 2 with an ISTFTNet vocoder and a
          PL-BERT encoder, over misaki phonemes, with 54 voice packs, at
          24 kHz. Its weights and voices are PyTorch pickles, so import
          reads them without executing them. misaki's grapheme-to-phoneme
          step runs natively, with no interpreter in serving (D-010); its
          espeak-ng fallback is GPL and belongs in the optional copyleft
          tier (D-080).
      - Stable Diffusion WebUI's `/sdapi/v1/txt2img`, `/img2img`,
        `/sd-models` and `/options` over the same pipeline, for Open
        WebUI, SillyTavern and LibreChat.
      - Images in chat responses (OpenRouter's `message.images` and
        `delta.images`, D-046), so a chat request to an image model
        returns its image.
      - `POST /v1/audio/transcriptions` on the audio carrier, in OpenAI's
        shape, for voice input in Open WebUI and LibreChat.
      - An optional **MCP media server**, a separate process calling these
        routes. It is how coding agents whose built-in image generation
        cannot point at a local server mix generated images into chat:
        Claude Code, OpenCode, Codex, Gemini CLI, and Antigravity where it
        accepts the server. Open WebUI and LibreChat call the routes
        directly, and Codex's own image tool is tried against them too.
- [ ] **Resident only.** Demand-paged experts stay M7's; M7 keeps its
      daily-driver usability work and builds on the families brought up
      here.

**Exit criteria**, per approved family and form (MoE, dense). D-101's
decision and speech models meet their own criteria below in place of
the correctness, speed and long-context ones:

- **Correctness:** greedy tokens match the same-format oracle except
  near-ties, under the recorded-first noise bound; perplexity within a few
  percent; speculation, where present, meets M3's speculation criteria.
- **Speed and memory** (D-085): prefill and decode at least as fast as the
  same-format reference, at 8K and at depth; peak memory at most about
  1.1× the reference's.
- **Long context:** M3's scaling criterion applies to the family's
  maximum context on one Spark, with any architectural floor measured and
  named. Dense global attention's per-token KV read is such a floor.
- **Swap:** each family swaps A→B→A with an M3 model within M3's swap
  goals, exact on return.
- **Concurrency:** with batching on, single-stream prefill and decode stay
  within noise of M3's. Aggregate decode throughput rises with the
  number of concurrent requests (reported at 1, 2, 4, 8 and 12, against
  ds4's batched serving and vLLM where they run the same model and
  format). Each request's greedy output equals its output when run
  alone, except near-ties. Forked branches share their prefix state
  without copying it until they diverge.
- **Formats:** every format in the approved covering set runs with the
  same correctness and speed criteria against its same-format reference.
  EXL3 decode and prefill are faster than ExLlamaV3's on the GB10, dense
  and MoE, at the covering set's bitrates, including mixed widths.
- **Media inputs:** for each model and modality, encoder outputs and the
  language model's teacher-forced logits on image, video and audio
  prompts match the same-format reference within bounds declared before
  evaluation, with preprocessing matched to the reference's pinned
  processor. Requests with several images, and video files, are among
  them. Generated text alone is not evidence. Inputs past their bounds
  are refused before any decode, and the intake passes its adversarial
  challenge.
- **Decision models:** Clef's and Clef-flash's per-option probabilities
  match their pinned reference within declared bounds, text-only and with
  images. The TypeSafe SDKs complete requests unmodified. Decision
  requests batch with each other, and no state outlives a request.
- **Speech:** with the same inputs, seed and sampling, Breeze's codebook
  logits match its reference's within declared bounds (teacher-forced),
  and its codec turns given codes into the reference's waveform within
  bounds. Kokoro's waveform for given phonemes and voice matches its
  reference within bounds. Time to first audio and the real-time factor
  are at least as good as each reference's on the GB10 (D-085). Open
  WebUI speaks through `/v1/audio/speech` unmodified.
- **Generation routes:** Qwen-Image's generated and edited images from
  the routes equal the native pipeline's for the same seed;
  Ming-Image-0.1-Design's opaque and transparent images match its
  reference pipeline's for the same seed within bounds, alpha included,
  at least as fast on the GB10 (D-085), and Open
  WebUI generates through them unmodified, and through the WebUI routes,
  SillyTavern does too. Open WebUI shows an image returned in a chat
  response. Transcriptions from the audio carrier match its reference
  output for the same clips. A MiniMax H3 job runs from
  create to download and cancels cleanly. Through the MCP media server,
  Claude Code and at least one other coding agent show a generated image
  in a chat turn.
- The support matrix lists every approved family with its evidence.

## M4 — Two-Spark fast full swap  `pending`

Goal: the same cycle for models too big for one Spark, sharded across both.
Each node holds its shard on disk, and the conductor loads both shards at
once.

**Entry:** M3.5 exit (M3 exit before 2026-09-29). By entry, model-parallel artifact partitioning is
decided ([artifact-format.md](artifact-format.md#deliberately-open); moved
from M8's entry), and each model's checkpoint, recipe and baselines are
pinned and audited as in M3.

**Models, in this order:**

*Note, 2026-09-29:* the recipe moves fast; re-pin it at M4 entry. Its
2026-09-28 changes (#281 FP8 on most of the path with KDA in BF16:
decode 3.9–5.2% and 16K/64K prefill TTFT 11.5–12.6% faster; #292
`GLM53_MODEL_PRESET=dense-h3`: dense EXL3 "H3" targets with matched
6-bpw DFlash2 drafts, up to 16% decode, estimated; all
creator-reported, @plotarmordev on X) are techniques to study: whether
the FP8 split is a default or a quality mode under D-085's note, and
quantized drafters matched to EXL3 targets (M3.5's EXL3 work).

1. **GLM-5.3 Flash** (`MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks@c1b7d4c`;
   EXL3 at about 4 bpw, TP2, about 80 GiB of weights per node,
   creator-reported): KDA linear attention, DSA sparse MLA with an indexer,
   mHC, and 288 routed experts.
2. **DeepSeek v4.1 Flash** (Mia's EXL3 at 2.9 bpw, TP2, 99.5 GiB of weights
   per node, creator-reported): a 552B encoder-decoder with CSA2 attention
   and a hierarchical indexer, which no GGML code covers. Its ~190 GiB of
   Engram lookup tables may stay row-paged from each node's SSD (D-035).

**Oracles and comparators,** under M3's rule (cross-quantization is speed
and memory only):

Also study and measure the owner's additional
[Kindling GLM reference](m4-references.md), pinned provisionally on
2026-09-29. Its NVFP4 TP2 recipe is a speed/memory comparator for the
EXL3 target; re-pin and audit it at entry, and distinguish its dense-layer
requantization from optimizations at unchanged quality.

| Model | Correctness oracle (same format) | Performance comparators | Cross-quantization (speed and memory only) |
| --- | --- | --- | --- |
| GLM-5.3 Flash | Mia's EXL3 configuration | Mia's configuration | TensorFold (MLX 4-bit) |
| DeepSeek v4.1 Flash | Mia's configuration | Mia's configuration | none |

**Scope:**

- [ ] **Provenance and licenses:** re-audit the GLM recipe if its baseline
      moves past the pin (HEAD is 65 commits ahead); record the GLM
      checkpoint mirror's license and the DFlash2 drafter's (CC BY-NC-ND
      4.0) for information (D-087); Mia's E3 and cooperative MoE kernels
      (AGPL or mixed, D-080); and whether Mia's ExLlamaV3 revisions'
      formats match our `6b84a21`.
- [ ] **Baselines:** Mia's two-Spark configurations and TensorFold, run by
      us, measured as in M3.
- [ ] **Conductor, minimal** (pulled from M6a): one configured two-node
      topology that loads both shards at once and runs each phase on both
      ranks. Discovery, enrollment, cluster trust and general placement
      stay in M6a. It is development-only (owner, 2026-09-28; D-087): it
      runs over the direct Spark-to-Spark link, trusted like loopback, is
      off by default, and is documented as development-only, not a
      supported deployment. Mutual TLS (D-038) arrives with M6a (D-014).
- [ ] **Sharded execution** (pulled from M8): TP2 over NCCL with
      conductor-issued distributed phase IDs, a separately budgeted
      communication-buffer pool that honors NCCL's registration and
      threading contracts, ordered collective submission and completion
      fences, and per-step collective latency measured before bandwidth.
- [ ] **Coordinated readiness, minimal** (the start of M8's coordinated
      admission): no collective or execution of B begins until both ranks
      report ready, their shard loaded and its memory obtained. A
      preparation failure on either rank, or no report within the stated
      timeout (default 30 s, owner-adjustable), aborts the swap: the other
      rank releases or rolls back what it prepared within that timeout, no
      rank enters a collective alone, and the swap reports failure cleanly.
      Memory a rank has not confirmed released stays counted as held; no
      timeout proves reclaim. General recovery, prepare/commit and the
      broader pressure and failure matrix stay in M8.
- [ ] **Kernels:** EXL3 MoE, the codebooks and rates these checkpoints use,
      sparse MLA decode and prefill on `sm_121`, KDA, the DSA indexer, CSA2
      and Engram row gathers, each the fastest correct implementation under
      D-053, chosen by a quick A/B.
- [ ] **Per model,** as in M3: graphs, state adapters, templates, the
      resident expert layout for EXL3 MoE, and speculation with M3's
      forced-rejection checks where the model supports it (for GLM, MTP or the
      DFlash2 drafter that Mia's GLM configuration uses, whichever is faster
      and correct; the
      nextn layers or DSpark head for v4.1). ReconGemm prepares its
      cuBLASLt descriptors once per bound GEMM (M2's hand-off).
- [ ] **Media inputs** (D-101), on M3.5's intake:
      - GLM-5.3 Flash's images and video: the GLM ViT (24 layers, patch
        14, 2D RoPE inside), video at 2 fps with per-frame timestamps;
      - DeepSeek V4.1 Flash's images: the DeepSeek-ViT with its 3×3
        aligner, at most 1,024 tokens an image in reading order.

      Both checkpoints keep their towers. Mia's V4.1 build says vision was
      not its validation target, and llama.cpp has no V4.1 architecture,
      so V4.1's oracle is its repository's own inference code.
- [ ] The minimal `/v1/chat/completions` serves the sharded models.

**Exit criteria:**

- **Swap time,** as in M3 across both nodes, with its owner-adjustable
  defaults:

  | Item | Rule |
  | --- | --- |
  | Pairs | GLM→v4.1→GLM and v4.1→GLM→v4.1, each swap reported separately; the headline is the worse |
  | Saved context | A holds 8K tokens; its KV, recurrent and indexer state on both ranks are spilled and restored, and count. The bound applies here; a 0-context swap is reported too |
  | Graphs | As in M3: the ~10 s goal and ~20 s bound apply to previously prepared swaps; first use no worse than about 2× the bound |
  | Endpoint | From the swap request to B's first generated token for a short prompt |
  | Also per swap | Per node: bytes read, read throughput and peak memory; the time between the two ranks' ready reports |

  The report sets each swap beside the baselines, Mia's GLM among them at
  65–70 s cold (creator-reported).
- M3's correctness, speculation, performance and memory criteria against
  each model's oracle and comparators above, with memory judged per node.
  Swap and restore are bit-identical on both ranks.
- **Readiness:** a load failure and a memory failure injected on one rank,
  each on either rank in turn, start no collective, the other rank rolls
  back within the timeout, the swap reports failure cleanly, neither
  catalog keeps a lease, grant or backing from B, and the next swap
  succeeds.
- **Media inputs:** M3.5's media-input criterion holds for both models'
  modalities.

## M5 — One resident model, end to end  `pending`

Goal: import the small fixtures, serve them resident through native GGML and
EXL3 execution, and complete chats from named clients over the three
baseline protocols, with bounded, explainable memory.

**Entry:** M4 exit. The native tokenizer, stop rules and sampling come from
M3.

**Scope:**

- [ ] **Importer and verifier** (D-009, D-056): the C++ importer and the
      standalone verifier, with M0's Python prototype as the exact
      accept/reject oracle, running on the workstation and on a Spark.
      Confined import jobs take sources only from the configured stores
      (the `checkpoints` role or the long-term store; D-054, D-063) or the
      Hugging Face Hub (resumable and verified; the token comes from a
      `.env` or config file and is never logged). Hard-linked and symlinked
      sources are rejected, and the user docs say to copy them in or
      download with `--local-dir` (D-063). Removal waits for
      quiescence. Jobs keep their records, locks and grants and report
      progress, status, cancellation and retry (D-041). An install that
      lacks space fails before transferring, reporting the space it needs
      and the removal candidates.
- [ ] **Registration and first use** ([model lifecycle](architecture.md#model-lifecycle)):
      the compact installed-model index cache, shallow checks on every
      startup, and detail and plans built on first use inside an `F`
      reservation. Startup rejects an installed store that fails the
      direct-I/O probe (D-054).
- [ ] **Resident execution:** the FP16 and both EXL3 fixtures under jitLLM
      dispatch, finite context and chunk profiles within the 8K context,
      and state block sizes and KV layouts for each adapter. Decode graphs
      and their relocation proof come from M3.
- [ ] **Tokenization and output:** D-067 renderers for both fixture
      template hashes on M3's native tokenizer, stop rules and sampling
      (moved to M3), and the tool-call parser; host versus device sampling
      is settled against D-052's decode gates.
- [ ] **Front door** ([M5 surface](client-api-baseline.md#m5-surface)),
      growing from M3's minimal Chat Completions:
      Chat Completions, stateless Responses, Messages with token counting,
      `/v1/models` in both shapes with D-046's metadata, and the discovery
      document (D-041); D-045's listener, authentication, CORS, `Host`,
      status and keepalive rules; D-047's non-streaming and storage rules;
      the Claude Code profile; and explicit rejection of `transforms` and
      `plugins`. Numeric intake bounds are fixed before any external input
      is accepted, with the total input limit reconciled with named-client
      tests ([cluster design](cluster-design.md)).
- [ ] **Gemini API and fill-in-the-middle** (D-101, the owner,
      2026-10-02):
      - Google's Gemini API (`v1beta/models`, `:generateContent`,
        `:streamGenerateContent?alt=sse`, `:countTokens`, image
        `inlineData` in and out) for Gemini CLI and the google-genai SDKs
        by base URL.
      - Code completion with a suffix: `suffix` on `/v1/completions` (and
        the Ollama profile's `/api/generate` in M10), Mistral's
        `/v1/fim/completions` and llama.cpp's `/infill`, on models whose
        tokenizers carry FIM tokens, confirmed at entry.
- [ ] **Local management API and CLI** (D-064): versioned routes for import,
      listing, representation inspection, removal, jobs and node health.
- [ ] **Service hardening** (D-074; M1's hand-offs, moved from M2 by the
      owner on 2026-09-27): a system-call filter for `jitllm.service`
      that admits io_uring, and a single reaper for jobs started from
      other threads.
- [ ] **TLS** (D-065): per-name certificate files selected by SNI, reload on
      change, key-match and expiry checks, the name-constrained local CA,
      the certbot deploy hook, and the Tailscale certificate timer. The
      root-run hook and timer touch only `/etc/jitllm/tls/`, their units are
      sandboxed to it, and they take the heavy path.
- [ ] **Surface definitions:** front-door, alias and TLS configuration keys;
      the individual `jitllm-` header and body-field names (D-062); the
      HTTP, TLS and JSON libraries, chosen under D-017, D-057 and D-066.
- [ ] Move the fixtures' rows in the model support matrix (started in M3,
      where they are listed as fixtures) to served, with their templates.

**Exit criteria:**

- Teacher-forced logits and declared intermediates for the three fixtures
  match their pinned references within bounds declared before evaluation
  ([first-slice.md](first-slice.md), [exl3-bringup.md](exl3-bringup.md)).
  Rendered bytes and token IDs match the golden fixtures.
- EXL3 resident prefill, time to first token and decode inter-token
  p50/p95/p99 meet the predeclared upstream parity bounds against upstream
  serving controls in matched and normal views, with memory inside the
  declared bounds; a regression needs a fix or an explicit owner-approved
  tradeoff (D-052).
- Each engine, served resident, is at least as fast as its reference end
  to end (GGML against llama.cpp, EXL3 against ExLlamaV3; D-085's coarse
  comparison), with BP-F3's resident timings reported in it (moved from M2
  by the owner, 2026-09-27; BP-F4's per-token host cost is measured in
  M3).
- The [M5 acceptance cases](client-api-baseline.md#acceptance-owed-in-m5)
  pass as scoped there. Gemini CLI completes a chat unmodified through
  the Gemini API, and Continue or llama.vscode completes code through
  fill-in-the-middle, with the inserted text equal to the model's greedy
  output for the same prompt. At least one named client completes a chat
  unmodified with each representation, and Cursor stays an explicit gap
  unless resolved. The context-compacted release moves to M6, which
  delivers D-041's close, and the Ollama-native checks move to M10 with the
  Ollama profile.
- Finite default context and output bounds bound every admitted request,
  and the M5 rows of D-050's matrix pass, with the parts moved to M5
  (suballocation holes, stalled-client termination, every queue full at
  once, capacity-loss injection). Memory use is bounded and
  explained by the memory breakdown.
- The importer, verifier and front-door parsers pass their adversarial
  challenge; an interrupted import never appears valid.
- Measured and recorded: startup time and metadata memory as the installed
  library grows, the cold-switch cost of on-demand detail, and both
  contexts' state bytes, from which D-055's
  [capacity values](retention-policy.md#bounds-and-defaults) are pinned.

## M6 — First useful product: A→B→A with partial retention  `pending`

Goal: two small model contexts share one local budget. Switching to B
displaces only what B needs, retained conversation state lets A resume
without a full re-prefill, and the switching policy is chosen from
measurement. Useful without MoE or sharding. M3 already restores one
model's state across a full swap; M6 adds partial retention and the
retention policy around it.

**Entry:** M5 exit with its capacity values pinned, plus everything
[M6 entry pins](retention-policy.md#what-m6-entry-pins): the frozen
transcript, budgets and reference paths among them, and the spill write
budget from the drive's rated endurance. EXL3 switching budgets come from
new matched controls ([exl3-bringup.md](exl3-bringup.md)).

**Scope:**

- [ ] **Retention** ([policy](retention-policy.md); D-024, D-031, D-055):
      prefix and continuation entries with their identity, restore
      boundaries, branches, sharing and refresh; capacity-driven expiry with
      24-hour idle caps; the victim-order baseline; spill with its protected
      directory, preallocation, direct I/O, digest check on restore and
      deletion at startup; miss reasons and fallback reporting.
- [ ] **Partial eviction** of a quiescent model, reloading only missing
      dependencies (D-008).
- [ ] **Swap pipelining and load order** (owner, 2026-10-02): explore
      overlapping page-in with execution, so each layer runs once its
      extents are resident instead of after the whole closure (graphs wait
      on per-layer load completion or split per layer), and order page-in
      by first use: dense and attention weights and restored state before
      later layers, a drafter (DeepSeek's DSpark, Qwen3.8's MTP) after the
      first token. Measure time to first token after a swap at several
      prompt lengths, cached continuation included, against M3's
      load-then-run full swap (DeepSeek's page-in is about 8 s of a
      9.7 s worst swap,
      [fast-swap](experiments/fast-swap/swap.md)); keep what pays.
- [ ] **Concurrency when it fits:** all-resident cohorts under
      full-envelope checks, reporting whether requests ran concurrently or
      time-sliced.
- [ ] **Switching policies** (D-069): the priority-aware default,
      run-to-completion and time-slicing with their guards, their
      configuration keys and per-alias overrides.
- [ ] **Request control** (D-042): interactive and background classes,
      maximum queue waits, cancellation and bounded progress events.
- [ ] **Release** (D-041, D-045): the final-turn flag, idempotent
      continuation close and the context-compacted release, with their wire
      names fixed here.
- [ ] **Warm jobs** (D-041): capacity-constrained, never an implicit
      download.
- [ ] **Diagnostics:** status, the admission what-if query, Perfetto trace
      export and eviction explanations; management controls for priorities,
      residency policies and trace capture; Prometheus `/metrics` with
      compatible health and load queries (D-044).
- [ ] **Storage scheduling:** demand reads mixed with spill write-back,
      inside the pinned write budget.
- [ ] Add the A→B→A workload and its regression thresholds to `check:spark`.
- [ ] **Discrete GPU, secondary** (D-082; after the two-Spark swap, D-087):
      the A→B→A fast swap, one model
      active and partial retention within device memory, on the
      workstation's discrete GPU (`mise run test -- native --gpu`), with
      its PCIe restore rate reported, within a configured device budget and
      with the landing zone reported apart from it. Whole-model swaps and
      paging within device memory only; no on-demand expert paging from the
      SSD there. Not an exit criterion.

**Exit criteria:**

- D-055's [timed workload](retention-policy.md#m6-acceptance-workload)
  passes its pass rule: Qwen2.5-0.5B FP16 and EXL3 4.0 bpw in both
  orientations, six jitLLM arms against fresh interleaved references, 72
  accepted repetitions per arm, orientation and cache condition. For each
  floor arm (J-partial, J-spill), orientation, direction and cache
  condition, at the median and at p95, jitLLM's one-sided 97.5% upper bound
  is at most the smallest one-sided 97.5% lower bound among the valid
  reference arms; a comparison with no valid reference arm does not pass
  (D-036). The report covers latency distributions, bytes read and written,
  peak memory and spill, prompt tokens reused versus recomputed, and deltas
  against jitLLM's whole-model control (M9's comparator).
- The [correctness gates](retention-policy.md#correctness-gates) pass: exact
  outputs and bit-identical teacher-forced logits against
  provenance-matched controls, and catalog state and events show that only
  selected extents were displaced.
- The [functional and adversarial cases](retention-policy.md#functional-and-adversarial-cases)
  and the M6 rows of D-050's matrix, with the parts moved to M6 (fork and
  copy-on-write, cached-state promotion), pass
  with an EXL3 context in the
  matrix; cache expiry never destroys admitted suspended work.
- Through at least one unmodified named client: a long conversation on A,
  B under pressure, then A resumed, over both resident reuse and forced
  spill/restore. The context-compacted release case deferred from M5
  passes.
- B arriving while A is still generating is measured under each D-069
  policy, with queue delay reported apart from paging and switch time and
  from first-token compute, together with pauses and bytes reloaded. The
  owner keeps or changes the default on these results.
- An all-resident control shows concurrent progress when both complete
  envelopes fit.

## M6a — Configured placement across nodes  `pending`

Goal: one configured conductor places whole models on enrolled nodes and
routes requests with retained-state affinity, so a subagent's model runs on
the other Spark while the main model stays resident. It follows M6,
independently of M7, and grows M4's minimal two-node conductor into the
cluster's placement layer; sharding under pressure and failure is M8.

**Entry:** M6 exit.

**Scope:**

- [ ] **Setup and discovery** (D-038, D-039, [cluster design](cluster-design.md)):
      inventory, trusted SSH, the mDNS window, layout classification,
      bounded dedicated-QSFP subnet scans, one-time enrollment and path
      re-detection for enrolled members.
- [ ] **Configuration and trust:** the shared and node-local v2 documents,
      the private CA with TLS 1.3 mutual authentication, epochs, session
      fencing and restart reconciliation, and protocol v1 with its bounded
      state, schema tests and exact per-message field catalog.
- [ ] **Conductor** (D-037): placement, affinity routing, single-attempt
      dispatch, credit-based streaming, health states and authoritative
      per-node admission; stale or aggregate reports never admit.
- [ ] **Availability** (D-041, D-046): the per-model `endpoints` shape.
- [ ] **One import per cluster** (D-054): peer replication of verified
      prepared artifacts over the cluster link, archive to and install from
      the long-term store, and the explicit archive-or-delete choice when a
      node lacks space.
- [ ] The package gains its rdma-core dependencies when jitLLM first links
      them (D-063).
- [ ] Decide whether a worker node serves its own loopback management
      listener.

**Exit criteria:**

- The [required validation](cluster-design.md#required-validation-and-handoff)
  challenge conditions and the conductor's fake-transport and Spark
  scenarios ([architecture](architecture.md#conductor-ownership-and-admission))
  pass.
- An unmodified standard client completes A→B→A through one endpoint, with B
  placed on the other node while A stays resident. Each node enforces its
  full local budget and compatible state reuse, and models run concurrently
  when placement permits.
- Stale capacity reports, node loss and cancellation end in bounded failure
  or unwind, never in unsafe admission or silent replay of a started
  stream.
- Remote management and cluster access require authentication and
  transport protection (D-014, D-065); inference authentication stays
  optional (D-014's owner note). A replicated or archived artifact is published only after
  verification against an identity held outside its source.

## M7 — Demand-paged MoE and the first daily drivers  `pending`

Goal: exact demand-paged routed-expert execution on the named Gemma 4
26B-A4B and Ornith 1.5 35B-A3B pair, and those two models usable day to day
through the named clients, reasoning and constrained output included (owner,
2026-09-23). Resident MoE execution arrives earlier, with M3's full swaps.

**Entry:** M6 exit; M6a is independent. The pair's checkpoints and
representations are pinned, and their GGML source closures, tokenizers and
chat templates are selected and audited (D-013, D-057, D-067). Approved
before measurement: the performance protocols; a resident-performance bound
against the pinned llama.cpp reference (prefill, time to first token and
decode inter-token p50/p95/p99, matched and normal views); the named
budgets, each with its loading policy, including at least one per model at
which its routed experts do not all fit beside its other extents, state and
headroom, so selected experts miss during both prefill and decode; and the
sustained-use schedule and duration.

**Scope:**

- [ ] **Routing boundary** (D-008, [architecture](architecture.md#routing-boundary-for-moe-7)):
      selected-expert leases, asynchronous misses and resumable tasks,
      brought up on a synthetic or tiny MoE before the named pair
      (features.md). Envelopes cover the worst-case union of every allowed
      closure and assume no fixed number of positions per phase (D-068).
- [ ] **Loading policies:** eager active-model loading that keeps inactive
      extents (the feasibility study's recommended first policy) and routed
      demand paging, both selectable, compared at the named budgets.
      With M6's swap pipelining: after a swap, load the routed experts
      each layer selects ahead of a background load of the rest, demand
      reads taking priority over background reads on the shared SSD
      bandwidth, and measure time to first token for a continuation
      against loading everything first (owner, 2026-10-02).
- [ ] **Expert layout:** expert compaction and demand-paged dispatch from
      the M7 GGML proof, building on M3's initial pointer-table or
      uniform-stride choice per format, and the MoE mapping in v0 artifacts.
- [ ] **Shapes that come with these models:** hybrid sliding-window and
      global attention (Gemma 4) and recurrent or linear-attention layers
      (Ornith), with their state adapters and explicit restore coverage
      (RE-004, RE-007), building on M3's Gated DeltaNet layers and state
      adapters. M6's retention matrix extends to them.
- [ ] **Traces:** native routing-trace capture and policy replay, checked
      against the M0 reference experiment; captured traces stay outside Git
      and replay by verified hash in the gate.
- [ ] **Reasoning** (D-043, D-046, D-047): protocol-specific reasoning,
      final and tool fields and streaming, model-supported thinking
      controls, signed blocks, `reasoning` and `reasoning_details`, and
      cached-token usage from real prefix reuse.
- [ ] **Constrained output** (D-043): `response_format` JSON object and
      schema, strict tool arguments and vLLM's `structured_outputs.json`,
      over a documented schema subset with explicit rejection of the rest.
- [ ] **Sustained use:** a bounded multi-hour agent/subagent session on the
      pair with repeated switches, branches, cancellations (during I/O and
      while paused included), expiry by capacity and by test-shortened idle
      caps, and spill and restore.
- [ ] Demand-paged EXL3 MoE only if claimed, with its own routed-expert
      closure, kernel, quality and performance baselines (D-052); resident
      EXL3 MoE arrives with M4's models.
- [ ] Assess artifact compatibility guarantees with M6's dense and M7's
      MoE evidence, in a separate decision (D-018).

**Exit criteria:**

- Acceptance exercises real demand misses. At each miss-forcing budget,
  catalog state and events record nonzero selected-expert misses in prefill
  and in decode, with their count, bytes and wait time and the phase and
  layer at which each suspended. There, a measured window without misses is
  inconclusive for the paging gates, and the next two criteria are judged
  on runs with misses.
- No unselected expert loads beyond declared metadata and read-ahead, and
  no expert is substituted or its contribution dropped, verified from
  catalog state and events.
- Numerics stay correct against the pinned references after eviction and
  restoration and across within-step misses, and the M7 rows of D-050's
  matrix, with the runtime closure-excess check moved to M7, pass with
  real routes.
- D-036's generation limits hold on the pair: at most 10% added generation
  time, continuation time to first token included, and at most 20 ms p95 /
  100 ms p99 added token gaps against a resident control with matched state
  provenance, at every named budget, the miss-forcing ones included. Pause
  gaps are reported separately (D-069). M6's switching floor still holds.
- Resident prefill, time to first token and decode inter-token p50/p95/p99
  on both models meet the approved bound against the pinned llama.cpp
  reference, or the owner approves an explicit tradeoff, as D-052 requires
  for EXL3. Paging limits compare jitLLM with itself, so they cannot stand
  in for this.
- The sustained-use run ends with every completed, cancelled and paused
  request retired and no lease, grant, task or I/O outstanding. Memory, the
  catalog and retained-entry metadata, task and job records and queue
  depths stay within their bounds and level off instead of growing with
  elapsed time; task and job records and queue depths return to their idle
  levels after each switch cycle. Spill stays within `S_spill` and the write
  budget, and the memory breakdown reconciles with OS counters at the end.
- Measured and reported: the routing boundary's resident-hit overhead, each
  routed phase kind's bound against its observed peak, and whether idle
  retained state starves weight residency.
- Named clients complete reasoning and tool round trips on both models,
  including acceptance case 7's reasoning checks, and constrained-output
  requests pass their pinned compatibility fixtures. Any OpenRouter-mode
  claim rests on D-046's pinned-client run. The support matrix records the
  client versions.

## M8 — Sharded execution under pressure and failure (two Sparks)  `pending`

Goal: a flagship model sharded across both Sparks, correct under asymmetric
pressure, cancellation and controlled failure. M4 already runs TP2 sharded
full swaps (its communication-buffer pool, collective ordering and
collective latency moved there, with a minimal ready-before-collective rule
and bounded abort), and placement-only use works at M6a.

**Entry:** M6a and M7 exits. By entry: the flagship checkpoint and its
validated parallelism recipe are named (M4's models, and the
MiMo-V2.6-Flash-RL TP=2/EP=2 reference, are the candidates); and the
two-node admission design chooses between prepare/commit and the deferred
mirrored-ledger shortcut (features.md). Model-parallel artifact
partitioning is decided at M4's entry.

**Scope:**

- [ ] Parallelism beyond M4's TP2, porting the recipe's first (TP, PP and
      EP are different plans), with conductor-issued distributed phase IDs.
- [ ] Coordinated admission, generalizing M4's minimal readiness rule:
      node-issued reservations and every rank ready before commit, with
      prepare failures unwound; an unknown completion never frees another
      rank's buffers.
- [ ] Sharded models under partial eviction and paging, not only full
      swaps.

**Exit criteria:**

- Both ranks stay correct against the pinned reference under asymmetric
  pressure, cancellation and controlled failure, including partial
  preparation, a lost commit, mismatched rank generations and node loss
  during a collective. No timeout is treated as proof of reclaimed memory.
- Sharded performance is reported against the recipe's reference deployment
  in both views.

## M9 — Performance and new decoding modes  `pending`

Goal: meet D-036's benefit target on a library larger than memory, and
execute the speculative and block-diffusion shapes designed since M0
(D-068) that M3 and M4 did not.

**Entry:** M7 exit. M9 may start before M8 exits, but MiMo's stored MTP
layers run only on M8's sharded execution, so that work waits for M8 and M9
exits after it. Before M9 planning, the bounded DiffusionGemma reference
study measures the per-step expert closures of wide phases, and untriggered
deferrals are reviewed ([features.md](features.md) and the table below).
Before the new modes execute, numerical and statistical bounds and trace
protocols for speculative sampling and diffusion decoding are declared;
manifest references to another artifact by ID are settled in M3. The named
configurations, including the over-memory library, and the
partial-retention benefit workload are pinned before acceptance runs
(D-036).

**Scope:**

- [ ] **Larger-than-memory library:** DeepSeek V4 Flash with Qwen3.8 Flash
      Next on one node is the canonical pair (D-036). M3 runs both, with
      their compressed attention and indexers, Qwen3.8's sparse n-gram rows
      and its linear-attention layers, as full swaps; M9 adds partial
      retention and paging on them. If the pair cannot be validated, name
      another whose prepared weights exceed physical memory rather than
      pass M9 on the small pair alone.
- [ ] **Speculative decoding** (D-068), beyond M3's and M4's: stored MTP
      layers for Ornith (and MiMo's once it runs sharded under M8) and
      Gemma 4 companion drafters, with speculative sampling at every
      supported setting, and draft-length and acceptance tuning.
- [ ] **Block diffusion** (D-068): DiffusionGemma-26B-A4B.
- [ ] **Execution speed:** CUDA graphs beyond M3's decode graphs, further
      kernels and plans (D-053), and target-assisted import tuning as an
      explicit, separately keyed mode.
- [ ] **TensorFold's format** (owner, 2026-09-28; D-087; *moved to M3.5 by the
      owner, 2026-09-29, as its "MLX affine import"*): import and run
      MLX-style affine 4-bit weights (TensorFold's checkpoints; reconcile
      the group size, 32 or 64) for Qwen3.8 Flash and GLM-5.3 Flash, and
      optimize them. TensorFold is then a same-format oracle and a gated
      comparator for those models (D-085's ~10% and ~1.1× bounds). Until
      then, in M3 and M4, it is a cross-quantization comparator: reported,
      not gated.
- [ ] **Victim policy:** compare global LRU, frequency/recency and the
      cost-aware heuristic on identical recorded traces, with the confirmed
      hysteresis, minimum-residency and reload-cost refinements
      ([baseline](architecture.md#victim-selection-initial-baseline)) and
      per-model statistics so no model monopolizes reclaimable bytes. A
      candidate replaces the baseline, and the cost-aware heuristic becomes
      the default, only if replay shows fewer miss bytes and reloads.
- [ ] Deferred optimizations whose triggers have fired (prefetch, residency
      warm-start and the others in features.md), each against its
      demand-only control.

**Exit criteria:**

- D-036's benefit target: at least 25% lower median return-switch latency
  than jitLLM's own whole-model control, with identical state handling at
  the same budget, on the agreed partial-retention workload, with at least
  one named library exceeding physical memory, while the switching floor
  and generation limits still hold.
- Measured and reported on the over-memory library: retention under
  physical pressure and whether idle retained state starves weight
  residency (D-055).
- The speculative verify path's teacher-forced logits match plain decoding
  within declared bounds, with top-1 agreement reported and free-running
  divergence reported at its first position (RE-008). These gates cover
  M3's and M4's drafters too. The targeted checks those drafters need
  (forced rejection, rollback across a swap and a coarse sampled
  distribution) moved to M3's exit and apply to M9's drafters as well.
- Speculative sampling preserves the target distribution. On recorded target
  and draft distributions, acceptance, rejection and residual resampling
  match a reference implementation of the rule exactly under fixed seeds;
  sampled outputs match plain sampling's distribution within declared
  statistical bounds, at the supported sampling settings.
- Rollback leaves exactly the accepted prefix. After rejected drafts,
  teacher-forced continuation matches a control that drafted only the
  accepted tokens, bit-identical where the plan is the same at both verify
  widths and otherwise within the declared bounds. No rejected position's
  KV, recurrent update or drafter state survives, none enters a retained
  entry (D-055, D-068), cancelling between draft and verify retires all
  draft work, and a pause there commits none of it before verify.
- Diffusion stays within declared bounds of its reference, compared step by
  step under a fixed seed. A cancelled canvas never commits, and a paused one
  commits nothing until it resumes and finishes.
- Matched and normal reference comparisons are reported, with no numerical
  or lifetime regression in any supported configuration.
- If prefetch is built, the part of D-050's
  [matrix](reservation-policy.md#worked-cases-and-implementation-gates)
  moved to M9 passes: repeated speculation keeps its full peak in `J` or
  the owning phase.

## M10 — Product and first release  `pending`

Goal: the remaining confirmed product scope, and a first tagged 0.x release
that a stranger can install from the project's package repository and run
(owner, 2026-09-23).

**Entry:** M9 exit. The owner may pull an item forward once its dependencies
exist (for example the Ollama profile or tokenization endpoints after M5, or
the dashboard after M6's management controls); the release itself waits for
every earlier exit. Each model added for embeddings or reranking is
named at entry with its pinned reference engine and numerical
bounds, declared before native evaluation.

**Scope:**

- [ ] **Dashboard** (D-064): a separate service that calls the management
      API from its own server side, including setting the Hugging Face
      token.
- [ ] **MCP management adapter** (D-042): a separate process exposing
      discovery, status and explicitly authorized actions.
- [ ] **Application permissions** (D-042): per-application credentials and
      scopes for inference, read-only status and model administration.
- [ ] **Session and hint extensions** (D-022, D-046): the optional session
      ID and release, client warm hints, and D-046's `session_id`, `user`
      and `metadata` hints. M5's front door already accepts them as
      advisory preferences, and M6 records `session_id` for release lookup.
- [ ] **Ollama subset** (D-041, D-045): listing, details, chat and
      generation with the `keep_alive` mapping, tested with a named
      Ollama-native client including its load-time bound.
- [ ] **Tokenization and diagnostics** (D-043, D-044): vLLM-compatible
      tokenize, detokenize, tokenizer information and prompt rendering; raw
      Completions with standard log-probabilities and bounded token
      diagnostics.
- [ ] **Pooled outputs** (D-042, D-044): embeddings and reranking on
      validated models named at entry, in the shapes their clients send
      (D-101): OpenAI's `/v1/embeddings`, Hugging Face TEI's `/embed` and
      `/rerank`, Cohere's `/v2/embed` and `/v2/rerank`, Jina's rerank,
      and Ollama's `/api/embed`.
- [ ] **File inputs** (D-042): text resources. Images, video and audio
      files moved to M3.5 and M4 with their families (D-101); here the
      Ollama profile carries `images` for the models validated there.
- [ ] **Packaging** (D-027): the signed arm64 apt repository and its signing
      keys, optional copyleft modules in the default install with a
      build-time opt-out (D-080), and drain-before-restart upgrades; and,
      secondary (D-098), an OCI image built from the release `.deb` with
      its published io_uring seccomp profile and documented run flags,
      checked by serving a request on a Spark.
- [ ] **Release readiness** (D-061, D-062): the release checklist, the
      published support matrix, notices and source obligations for every
      shipped profile, and user documentation.

**Exit criteria:**

- Each new route passes its pinned compatibility fixtures, including
  negative and interrupted-stream cases, with the unmodified clients its
  decision names; unsupported features fail explicitly.
- Every new model and output path meets numerical acceptance against its
  pinned reference within the declared bounds before any support claim:
  embedding vectors with the model's pooling and normalization; rerank
  scores and orderings (media inputs are judged in M3.5 and M4, D-101).
  Raw Completions log-probabilities agree with the validated
  teacher-forced logits, and tokenize and render output matches inference
  byte for byte and ID for ID.
  Generated text alone is not evidence.
- On a fresh Spark, installing from the apt repository and following only
  the checked-in docs reaches a running supported model (vision.md).
- The default and copyleft-disabled builds meet D-017 with a complete,
  audited closure; builds with optional modules enabled ship matching
  notices and source.
- The release commit passes `check`, `check:full` from a fresh clone and
  `check:spark` (D-061), and the owner tags the first 0.x release.

## Deferred delivery and proposals

Confirmed scope stays confirmed when its implementation is deferred. A
candidate stays unapproved until revisited; reaching a trigger is a reason
to evaluate it. Deferred optimization triggers (predictive prefetch,
dependency-group scoring, optimistic MoE) live in features.md.

| Item | Earliest work / revisit trigger | Scope |
| --- | --- | --- |
| Quality/performance modes: KV-cache and output-head compression (FP8 caches, TurboQuant and other algorithms; a quantized output head) and FP8 linears for Qwen-Image's DiT | After M3's speed work; earliest when long contexts or several resident conversations press on memory, or to shrink swap spill | [Per-model options with long-context quality checks](features.md#compute-backends-and-execution) (owner, 2026-09-28)  Exposed as model-mode aliases and the image endpoint's `quality` (owner, 2026-09-28) |
| Load/temperature-aware GPU operating policy | Earliest M9 evaluation, or earlier diagnosis if reproducible throttling or unexplained shutdowns occur | [Proposed telemetry and optional adaptive clock ceiling](features.md#load--and-temperature-aware-operating-policy-proposal); measure stock/fixed/adaptive policies first, no assumed fault or automatic host changes |
| Ollama registry and other management compatibility | After the basic subset and relevant native management operation, when a named client needs them | Deferred D-041 candidate; lifecycle mapping needs separate proof |
| Regex/grammar constrained output | After validated JSON/schema support, when a concrete client requires it | D-043 deferral; no automatic milestone delivery |
| LoRA adapters | Earliest M9 planning after validated base-model execution, when a concrete adapter workload needs them | D-044 deferral; no automatic delivery |
| Classification/reward/generic pooling APIs | Earliest M9 planning after validated base-model execution, when a concrete model/task workload needs them | D-044 deferral; not implied by embedding/reranking support. Decision models over the Jev API are the exception, in M3.5 (D-101) |
| Live audio/video input | Earliest M9 planning after initial file-input evidence, when a concrete workload establishes streaming/synchronization requirements | Deferred D-042 candidate; not an automatic M9 deliverable |
| Batch/background inference jobs | Earliest M9 planning after validated request scheduling, when a concrete workload justifies scheduling/storage needs | Deferred D-042 candidate; ordinary background request priority is already confirmed |
| Automatic membership changes | After M6a, when configured enrollment and explicit restart cannot reasonably serve membership churn | Candidate mechanism under D-023/D-038; bootstrap discovery and path refresh for enrolled nodes are already M6a scope |
| Conductor election | After M6a, when conductor failover becomes an explicit requirement; first define fencing and in-flight request handling | Candidate mechanism under D-023; one configured conductor initially |
| Automatic replica placement and balancing | After M6a, when measured overlapping demand on a small model causes waiting while another node has sufficient headroom | Confirmed D-023 scope with deferred delivery; preserve affinity and include duplicated weights/state in budgets |
| Discrete-GPU fast swap of large models | After the two-Spark swap (D-087); small models only fit a 12 GB card | M6 validates the A→B→A fast swap on the workstation's RTX 3080 Ti (D-082); a later slice or milestone takes the swap path further there |

Review untriggered items during M9 and M10 planning; they do not
automatically enter either milestone's scope or block earlier milestone exits.
