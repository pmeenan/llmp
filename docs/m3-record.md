<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# M3 record — Single-Spark fast full swap

**Frozen 2026-10-04: M3 is complete with the owner's speed exception.**
M3 began on M2's exit, 2026-09-27. This record retains its task history,
profiles, accepted exceptions and evidence. The owner authorized continued
work and commits through completion; on 2026-10-04 the owner accepted the
remaining Qwen/DeepSeek speed gaps and moved further tuning to M9's
full-engine optimization pass. This exception changes no correctness,
memory or swap bound and does not turn a recorded speed miss into parity.

The living [plan](plan.md) owns later work, [model support](model-support.md)
claims only its qualified profiles, and [decisions](decisions.md) owns the
contracts. Experiment reports retain their measured binaries, source pins,
profiles and limitations; a later documentation commit does not relabel
those runs as executions of the final Git revision.

## Final gate and evidence mapping

| Requirement | Evidence at exit |
| --- | --- |
| Prepared swaps and exact restore | The [final production table](experiments/m3-final-swap/README.md) covers 32 rows and all six ordered pairs. Worst prepared LLM swap is 9.853 s against the 20 s bound. All 8K states and 16-token continuations match; images match the fixed control and prepared LLM graphs replay. |
| Standard client | The unchanged OpenAI SDK 3.3.1 completes DeepSeek → Qwen → DeepSeek and streamed Qwen with natural stops, correct visible/reasoning channels and token counts ([report](experiments/m3-standard-client/README.md)). |
| Model numerical quality | Existing short oracle/held-out perplexity controls (including the owner-accepted Qwen French-prompt exception of 2026-09-28) and registered 32K/128K oracle histories qualify their named profiles. DeepSeek output-A/HCA defaults pass the fixed tie-aware rule ([acceptance](experiments/ds4-output-prefix/README.md#default-on-acceptance)). The [final Qwen concurrent control](experiments/qwen38-concurrent-oracle/README.md) qualifies four slots on the frozen 32K history: all eight 512-row cells give 477 agreements, 35 near-ties and zero outside steps under the unchanged 1.765 bound; complete rows and initialized target/MTP state match across slots and repeats. This is forced conditioning, not natural acceptance or unrestricted profile parity. |
| Image quality and execution | The BF16 pipeline passes its component and final-image bounds (PSNR 41.770 dB, SSIM 0.99604 against bounds 32 dB/0.98); resident generation averages 33.41 s versus diffusers 52.6 s, peak 31.5–31.8 versus 43.4 GiB ([native image](experiments/qwen-image-native/README.md)). The later final swap table preserves exact native image output. |
| Speculation and sampling | Forced rejection and swap controls retain exact state outside accepted writes. The registered per-drafter sampled histograms pass TV ≤ 0.1. Final source review maps serving to the same full-row Sample/VerifyDraft calls, parameters, seed/stream/absolute-position keys and per-branch scratch/Accept; resumed branch controls preserve those keys and anchors. This inherits the named harness distribution evidence rather than claiming a new C4 HTTP histogram measurement or relying on seeded repeatability alone. |
| Long contexts | Both LLMs complete their 262K configured range; DeepSeek also fits and completes the measured 1M profile. Retrieval and exact 128K/maximum-context continuations pass in the [final context record](experiments/m3-final-context/README.md). Earlier controls retain their original artifact, head and precision scopes. |
| Concurrent requests and model turns | Chat and literal completions join native cohorts; scored completions preserve likelihoods and Unicode offsets ([literal batching](experiments/literal-batching/README.md)). Pending model switches time-slice at completed units with exact resumed continuations ([model turns](experiments/model-turns/README.md)). Active responses keep making progress while new same-model requests wait for slots. |
| Memory | Growing state charges only initialized extents; shared plans/graphs stay inside the physical budget, idle conversations spill, and reclaim uses measured restore cost ([pressure controls](experiments/memory-pressure/README.md)). Matched Qwen C4 peak is about 81.28 GiB versus Mia's 101.42 GiB; DeepSeek's matched serving screen also uses less peak memory than ds4. |
| Speed exception | Latest matched Qwen C4 is about 15% below fast Mia; neither MTP head clears every strict long-context speed rung. DeepSeek's matched 7K one-output wall is 9.16% longer than ds4, while C2/C4 completed-token throughput leads 13.16%/18.36%. Remaining speed/scaling work is accepted for M3 and assigned to M9, not claimed as parity. |
| Build, provenance and package | CPU and native workstation tests, the final 1,567-test Spark suite, tooling and license/boundary checks pass in their recorded scopes. The final ARM cross/qemu run has 1,272 registered tests, with 1,243 executed passes and 29 environment/data skips. Its initial SDK discovery failure and later compound-comparison cancellation failure are repaired. The refreshed ARM package inventory and network-denied install/reinstall/remove/purge fixture passes on clean runtime commit `7ed5c1e` and the repaired SDK. The [final-checks report](experiments/m3-final-checks/README.md) records failures, repairs, skips and source identities. |

## Retained limitations

Qwen concurrent replies can vary with arrival-dependent wave composition;
HC and target-head products are not column-count invariant. DeepSeek's
automatic plain/speculative wave choice can also change greedy near-tie
trajectories; forcing the speculative form retains the recorded cohort
controls. Fixed seeded/forced profiles and their exact state restores are
separate evidence from cross-engine token equality.

Hang recovery eventually restarts a genuinely stuck GPU process; an
in-flight request at that last rung sees a closed connection. Kept
conversations are build-specific, and an active verify whose restore is
still owed while the batch holds its stream may not have a saved restart
record. The completed memory-pressure and switch controls do not claim
that every failure preserves all in-flight work.

## Work handed on

- **M9:** all remaining Qwen/DeepSeek kernel, producer/consumer, prefill,
  concurrent and long-context tuning; draft-acceptance diagnostics and
  broader D-103 calibration/schedule tables. No frozen prototype is selected
  without measured whole-engine benefit and its correctness controls.
- **M3.5:** the approved family and format track, including EXL3, alternative
  NVFP4 layouts/quants, additional drafters, Clef/Clef-flash decision models,
  media inputs and image generation. Compatible requests batch for decision
  and image models as well as LLMs; model-specific phases, latent/random state
  and resource limits remain explicit.
- **Shipment:** whole workstation check tiers, sanitizer/container/offline
  profiles and confined-job proofs were not all run. The focused checks below
  do not imply those tiers passed; owed shipment checks remain required before
  publishing a package.

The unchecked historical scope items below retain their original status.
Their remaining optimization portions move to M9 under the owner exception;
they are not silently converted into completed implementation claims.

## Retained task history

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
      A [same-history first-verify screen](experiments/qwen38-same-history/README.md)
      now compares four adjacent 32K anchors against the pinned fast Mia
      launcher: native accepts 9/12 actual drafts versus 11/12, with both
      nominal caps using the same selected 47,172-row head. The fourth anchor's deficit is a
      proposal difference where both targets agree on the shared input.
      [Cache-path and repeat controls](experiments/qwen38-same-history/README.md#cache-path-and-repeat-controls-2026-10-04)
      then find reference draft and target variation: the two-draft gap is
      not established as stable, and a repeated deterministic cache path
      matches native at 1/3. Smaller native prefill chunks retain its drafts.
      A [matched all-cold collection](experiments/qwen38-same-history/README.md#matched-all-cold-anchors-2026-10-04)
      now accepts 9/12 in both engines, with zero reference prompt reuse and
      a separate cold p3 repeat also matching native. All twenty conditioned
      target-row argmaxes agree; reference logits still vary on that repeat.
      Broader acceptance parity remains open; no bound or math default changes.
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
      The earlier [fixed long-context answer tests](experiments/ds4-long-context-tasks/README.md)
      complete four OFF/ON pairs without a clear candidate-specific answer
      regression, while both paths fail parts of that strict rubric. They
      supplied no quality exception. Subsequent [default-on acceptance](experiments/ds4-output-prefix/README.md#default-on-acceptance)
      qualifies output-A/HCA together under the documented tie-aware rule;
      both now serve eligible full and partial chunks. HCA alone remains
      default-off; broader final reference qualification remains open.
- [ ] **Concurrent-request comparisons** (the owner, 2026-10-01).
      The [current matched DeepSeek screen](experiments/deepseek-matched-serving/README.md)
      (2026-10-04) uses identical 7,043-token inputs with no prompt reuse,
      native buffered literal completions and ds4 buffered non-thinking chat.
      One-output native wall is 9.16% longer; at 256 outputs, C1 wall is
      1.71% longer and C2/C4 completed-token rates lead 13.16%/18.36%.
      All 21 requests complete, native memory drops are lower, and native
      bookend replies match. Different full reference replies, buffered phase
      clocks and the remaining protocols leave the broader gate open.
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
      use the same private arithmetic-body refactor. A subsequent
      [native wave integration screen](experiments/qwen38-gdn-cohort/README.md#native-wave-integration-screen--2026-10-04)
      matches every full target row and final state against the unmodified
      control with capture/replay, but has no measured gain: verify median
      107.399 ms versus disabled bookends 107.228 / 107.018 ms. It is not
      adopted; swap/recovery and actual serving impact remain untested.
      DeepSeek's [wide IQ2_XXS gate/up pass screens](experiments/deepseek-expert-passes/README.md)
      likewise preserve the existing scalar/wave and discard controls but
      give no measured C4 DSpark gain (four-token / two-token pass median
      latency +0.512% / +1.400% versus bookends); retain the current loop.
      All packing
      is paid. Overflow waits
      for a retired slot. Model changes now pause the group at completed
      units and resume its exact continuations after one substitute cohort
      ([model turns](experiments/model-turns/README.md)); literal completions
      now join those cohorts with their own scores and raw prompts
      ([controls](experiments/literal-batching/README.md)). Independent request failures
      retain completed peers; native fences prove retirement before releasing a frame. Matched comparison
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
      Qwen's [GGUF one-row waves](experiments/qwen38-gguf/README.md#one-row-gguf-waves-2026-10-03)
      also share dense/routed quantized products and the full head; eager and
      captured/replayed full heads and state are exact against unjoined waves.
      Matched plain HTTP gains 31.08% at C2 and 69.63% at C4 on 183-token
      prompts, 29.42% at C4 on 8,258-token prompts (256 outputs, UD-IQ3_XXS).
      This format-specific improvement does not close the NVFP4/Mia gap.
      Qwen's [wave lanes](experiments/qwen38-four-request-waves/README.md#wave-lanes)
      now run private attention and recurrence concurrently from four requests,
      with exact full logits and initialized state on NVFP4 and GGUF.
      Short C4 HTTP gains 3.76% with NVFP4/MTP and 7.65% with UD-IQ3_XXS;
      the [fresh 8K C4 Mia comparison](experiments/qwen38-four-request-waves/README.md#fast-start-mia-refresh--2026-10-04)
      measures 32.44 / 31.91 native bookends versus 37.86 completed tokens/s,
      a remaining 15.01% deficit (172-second fast-start Mia readiness).
      Native's peak MemAvailable drop is 0.801 times Mia's; this screen
      supplies no cross-engine quality verdict.
- [x] **Request slots sized by memory** (the owner, 2026-10-03). A
      model's number of concurrent request slots is no longer fixed at
      four: it follows the memory the node has free for slots (each
      slot's fixed buffers and state, under D-055's reclaim order and
      D-102), and grows or shrinks as that changes. Measure each model's
      throughput and per-request latency as slots rise, at short and long
      prompts; where returns diminish and individual requests are delayed
      significantly, that knee becomes the model's default cap,
      configurable. Wave kernels past their current 16-row joined
      products (four slots at depth four) need wider paths for this.
      Done (D-104, [request slots](experiments/request-slots/README.md)):
      `max_slots` per model (1 to 16) over a fallback knee of 4 for both
      LLMs (with plans warm, past four DeepSeek DSpark gains 9.3% then
      2.6% with short prompts and nothing with long while each request
      slows 24% then 12%, its wave-form chooser extended to widths 5–8,
      plain past five; Qwen3.8 gains nothing at any width; a burst's
      median request finishes later; a large cap also costs cold plan
      building, 68 s and 2.7 GiB of plans for Qwen3.8 at sixteen); a
      request joins only when its prompt's
      state fits beside its peers', else it waits in the queue; Qwen3.8's
      workspace is sized from its widest wave (7.45 → 2.17 GiB fixed at
      four slots), and a slot's fixed buffers cost 4–16 MiB. Wider joined
      products were not built: a wave's cost follows its rows (Qwen3.8 at
      depth 1, eight requests in one sixteen-row group, measured slower).
      The cap's calibration is D-103's.
      Owner decisions (2026-10-03): no preemption of long-running
      responses when slots fill (they keep making progress; a new request
      waits for a slot); **time-slice a pending model switch** instead of
      draining the batch (D-069's time-slicing, pulled from M6: the
      running batch's members are set aside with their state spilled and
      resumed exactly after the other model's turn); and **literal
      completions join the batch** as cohort members instead of waiting
      for it to drain. Model time-slicing is now implemented with a
      configurable 30-second resident-work turn, protected branch identity
      through spill/restore and original cached-token usage across partial
      prompt pauses ([controls](experiments/model-turns/README.md)). Literal
      completions now join the cohort, preserving likelihood rows and
      Unicode offsets across partial-prompt and generation switches
      ([controls](experiments/literal-batching/README.md)).
- [ ] **Model settings in three layers** (D-103, the owner, 2026-10-03):
      a table-driven `[models.NAME]` schema where every model setting can
      be overridden; one resolved settings record per model at
      registration, each value tagged derived / calibrated / override /
      fallback and logged and reportable; derived defaults from the
      artifact (including the checkpoint's generation-config sampling
      defaults and the template's reasoning markers); a calibration
      record under `roles.state` keyed by artifact, device and build, with
      the measured trades first (prefill rows, slot cap, wave costs per
      width up to the slot cap, draft depth and rows, floors, recompute
      cost), then the hand-tuned kernel schedule tables. Lands after the
      in-flight intake-limit, request-slot and DeepSeek wave changes,
      which all touch the same `[models]` keys.
      *Landed:* the table-driven schema with every setting a key, the
      resolved record with sources, derived defaults (context, sampling,
      draft rows, reasoning markers), the start's log and `jitllm-runtime
      settings` ([model settings](runtime-serving.md#model-settings)); the
      calibration record and passive first-use calibrations of the floors,
      recompute cost, DeepSeek's wave costs and Qwen3.8's depth cost
      ([calibration](runtime-serving.md#calibration)). Open: prefill chunk
      rows and the slot knee outside the controlled profiles below,
      then the kernel schedule tables.
      [Controlled Qwen chunk screens](experiments/qwen38-prefill-chunks/README.md)
      retain 4,096 rows: 8,192 gains 2.69% at fresh 8K C4 with 4.58% more
      memory, but another four-slot prompt family is slower. No calibration
      value is selected; DeepSeek chunks remain open. A
      [controlled Qwen slot-cap comparison](experiments/qwen38-slot-knee/README.md)
      retains four: six requests pay identical short inputs and 512 outputs
      at caps 4/6/4; throughput is −1.20% / +0.57% against bookends and
      median request latency rises about 48%. An owned settings-only record
      with the runtime-generated key accepts `max_slots: 4` as calibrated,
      while explicit slot overrides and a smaller head invalidate it.
      A [controlled DeepSeek comparison](experiments/deepseek-slot-knee/README.md)
      also retains four for the community+DSpark automatic profile: six
      gains 16.36% / 21.79% short throughput with 25.74% / 22.88% higher
      median latency; one long cell gains 6.70% with 9.30% / 10.26% higher
      mean/median latency. Six remains an explicit throughput/tail option.
      A separate current-build prime produces a real key for an owned
      slot-only record; four is calibrated, while explicit slot or wave-cost
      overrides and a changed context make it stale. Historical measurements
      remain on their original build/key. Production state and defaults
      are unchanged; broader profile calibration remains open.
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
- [x] **Memory policy** (the owner, 2026-10-02: use memory fully, scale
      down gracefully under pressure, never fixed cache counts; D-055 and
      D-090 as amended). *Done 2026-10-02*
      ([memory-pressure](experiments/memory-pressure/README.md)): one
      reclaim order, GreedyDual-Size over measured restore costs a byte
      (idle state spilled 0.17 s a GiB, graphs 0.2–0.4, plans 2–15;
      recomputing 64), all of what is asked or nothing. Plans and graphs
      are uncapped and charged inside the budget past one step's floor,
      and survive swaps. Plans shrink 3.2–4.1×. Idle conversations spill,
      writing only what changed, and resume exactly: 2.48 s with 20,012
      cached tokens against the base's 30.18 s re-prefill. Turn
      checkpoints' staging is set apart. The `[memory]` retention and
      spill budget are configurable. Pressure from outside is trimmed to
      a target headroom with hysteresis and back-off, never the running
      model's floor. The two-model DSpark configuration gains 0.61 GiB of
      state room, and its swap table now runs, exact. Open: plans to
      disk, per-slot plan sharing, the spill write budget (M6).

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
  **Final refresh passed 2026-10-04:** all 32 rows/six ordered pairs in one
  production process, worst LLM/prepared 9.853 s with corrected fresh-token
  endpoints, exact 8K states/16-token continuations and pinned regenerated
  images, with prepared 8K LLM graphs retained/replayed
  ([final table](experiments/m3-final-swap/README.md)).
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
  *Owner exception, 2026-10-04:* remaining Qwen/DeepSeek speed gaps,
  including solo/concurrent and plain/speculative prefill/decode, are
  accepted for M3 and deferred to M9's full-engine optimization pass.
  Recorded misses remain visible; this is not a measured parity pass.
  Memory and correctness bounds are unchanged.
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
    *Owner exception, 2026-10-04:* remaining Qwen/DeepSeek long-context
    speed and scaling gaps are accepted for M3 and deferred to M9. Depth
    correctness, exact saved-state continuation and swap bounds remain
    required.
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
  through the minimal route, swapping between them. **Passed 2026-10-04**:
  the unchanged OpenAI SDK 3.3.1 completes DeepSeek → Qwen → DeepSeek,
  then Qwen streaming; visible/reasoning channels and token counts equal
  native controls, with natural stops and clean retirement
  ([standard client](experiments/m3-standard-client/README.md)).
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

<!-- End of retained M3 task history. -->
