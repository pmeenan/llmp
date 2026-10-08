<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Qwen-Image-2.1 BF16, native (M3)

M3's third model slice ([plan](../../plan.md#m3--single-spark-fast-full-swap-in-progress)):
the Qwen-Image-2.1 pipeline (Qwen3-VL-8B text encoder, single-stream DiT,
VAE) run by llmpalooza natively on one Spark from its D-056 artifacts, one per
component, joined by a D-089 composition, against diffusers in BF16.

## Pre-registration (fixed before llmpalooza's first run on the model)

Inputs, fixed in the repository (D-087's entry rule):

- **Checkpoint:** `Qwen/Qwen-Image-2.1@790c9263`, pinned in
  [fast-swap/pins.json](../fast-swap/pins.json).
- **Oracle:** diffusers `8b3c707e` in the image-reference container
  ([image-reference](../image-reference/README.md), local image
  `sha256:700d6668…`, PyTorch 2.14.0+cu130), BF16 on the GPU, the fast-swap
  prompt set's `image` entry: the teapot prompt, 1024×1024, 40 steps, seed
  42, `true_cfg_scale` 1.0 (no guidance), prefix KV cache on.
- **Reference tensors:** [reference.py](reference.py) re-runs the pipeline's
  denoising loop by hand, in the same calls, order and dtypes as
  `QwenImage21Pipeline.__call__`, and keeps every tensor llmpalooza is compared
  with. Its image is pixel for pixel the baseline's (RGB SHA-256
  `7d00b052…`, [baselines.md](../fast-swap/baselines.md#qwen-image-21-diffusers-bf16)),
  so the kept tensors are the baseline's. They stay outside Git on `spark`
  under `~/.local/share/llmp/references/qwen-image-2.1/native/ref1/`
  (raw little-endian files; `reference.json` records each one's shape and
  SHA-256).
- **Seeded noise:** Llmpalooza does not reproduce PyTorch's CUDA generator. It
  starts from diffusers' initial latents for seed 42 (`latents_init`, the
  packed `randn_tensor` output), the simpler of the two routes the slice
  allowed.
- **Tokens:** Llmpalooza's native tokenizer and prompt renderer
  (`src/tokenizer`, `src/chat`) must give the
  reference's 39 token IDs exactly, and drop the 14 system-turn tokens.

**Calibration** (the scale of "small", from the same reference run on
`spark`, 2026-09-28): each component re-run in FP32 on the same inputs,
the pipeline with its DiT in FP32 over all 40 steps, the pipeline's own
cache-off sample and an unrelated sample (seed 43):

| Control | Result |
| --- | --- |
| Text encoder, BF16 vs FP32 (kept hidden states) | rel. RMS 0.101, cosine 0.99495 (a few massive-activation channels dominate) |
| DiT step 0, BF16 vs FP32 (same inputs) | rel. RMS 0.0063, cosine 0.999980 |
| DiT step 1, BF16 vs FP32 (same inputs) | rel. RMS 0.0229, cosine 0.999771 |
| VAE, BF16 vs FP32 (same latents), decoded tensor | rel. RMS 0.0027 |
| VAE, BF16 vs FP32, image | PSNR 53.5 dB, SSIM 0.9985 |
| Whole pipeline, DiT in FP32, image | PSNR 38.8 dB, SSIM 0.9922 |
| Prefix cache off, image | pixel-identical |
| Seed 43 instead of 42, image | PSNR 11.6 dB, SSIM 0.733 |

PSNR is over 8-bit RGB, SSIM the image-gguf study's 11×11 Gaussian
luminance SSIM ([compare.py](../image-gguf/compare.py)); relative RMS is
‖a − b‖ / ‖b‖ with the diffusers BF16 tensor as b.

**Bounds.** A correct BF16 re-implementation rounds differently from
diffusers, roughly as far as diffusers' BF16 is from FP32, so each bound
allows twice the measured BF16-versus-FP32 distance (twice the relative
RMS, twice the SSIM deficit, 6 dB below the PSNR), rounded outward:

1. **Tokens:** the reference's IDs exactly.
2. **Text encoder** (native tokens, the kept 25 rows × 4,096): relative
   RMS ≤ 0.20 and cosine ≥ 0.99 against diffusers' `prompt_embeds`.
3. **DiT, first step** (diffusers' embeddings and initial latents as
   inputs): relative RMS ≤ 0.0125 and cosine ≥ 0.99996 against diffusers'
   step-0 noise prediction.
4. **DiT, every step, teacher-forced** (each step fed diffusers' latents
   for that step): relative RMS ≤ 0.046 per step (twice step 1's
   calibration, the larger).
5. **VAE** (diffusers' final latents decoded): decoded tensor relative RMS
   ≤ 0.0055, image PSNR ≥ 47.5 dB against the reference image.
6. **Image, end to end** (native tokens and text encoder, the DiT's own
   trajectory from the seed's latents, the VAE): **PSNR ≥ 32.0 dB and SSIM
   ≥ 0.98** against the reference image. The unrelated-sample control
   (11.6 dB, 0.733) is far outside; the FP32-DiT control (38.8 dB, 0.9922)
   well inside.

Performance and memory are reported beside diffusers' (plan.md's image
criterion: full generation not more than 10% slower than diffusers'
52.6 s, D-085): per-step time, full generation (text encode, 40 steps, VAE
decode, with the weights resident as the baseline holds them) over plain
runs, and peak memory as the drop in `MemAvailable`.

## What runs

- **Artifacts** (D-089, [artifact-format.md](../../artifact-format.md#compositions)):
  [import_m3.py](../artifact-layout/import_m3.py) `component` imported the
  three components from the pinned checkpoint on `spark` (text encoder
  92 s, denoiser 75 s, VAE 6 s, full verify included) and `compose` wrote
  composition `eca21baad38229e471a44cb2479d392ffcf745fb812e8a41668f336139fa1acd`
  naming text encoder `ed89ed27…`, denoiser `d1184efd…` and VAE
  `44c1a20a…`, in `~/.local/share/llmp/m3-artifacts/`.
- **Load** (`llmp_qwen_image_exec`): the composition and each component
  opened as untrusted input (`artifact/composition.h`, `artifact.h`),
  bound to the compiled-in profile (`model/qwen_image.h`), and the groups
  a phase reads loaded with direct reads through pinned staging into
  `cudaMalloc` memory: the text encoder's table and 36 layers (15.14 GB,
  not its vision tower or `lm_head`), the denoiser (14.23 GB), the VAE's
  decoder (1.35 GB of F32, rounded to BF16 on the host as
  `from_pretrained(torch_dtype=bfloat16)` does, 0.51 GB on the device).
- **Phases:** encode (Qwen3-VL's text path, 39 tokens, the 14 of the system
  turn dropped), denoise (40 steps of the block-causal DiT: the first over
  the joint sequence of 25 text and 4,096 image rows, filling the text's
  K/V prefix cache, the rest over the image rows only), decode (the VAE
  decoder, one frame). `--phases released` loads each component at its
  phase's start and frees it at the end; `resident` holds all three.
- **Kernels** ([kernels/image](../../../src/kernels/image/ops.h)): BF16
  products on cuBLAS (`cublasGemmEx`, F32 accumulation, the call PyTorch
  makes for a BF16 `nn.Linear`), llmpalooza's FlashAttention-2 forward in BF16
  for the denoiser's attention, and llmpalooza's fused BF16 kernels for the
  norms, modulation, rotary embeddings, gated residuals, SwiGLU, the Euler
  step and the VAE's operations (im2col and cuBLAS for its convolutions).
  Each kernel rounds to BF16 where the pinned PyTorch code materializes a
  BF16 tensor, so fusing does not change the values: the scheduler's step,
  for one, rounds `dt` to BF16 before multiplying (0 of 10.2 million values
  differ from diffusers' steps that way, 287,316 with an F32 `dt`). Since
  the speed slice ([Speed](#speed)) both harnesses run the three phases
  through one pipeline ([pipeline.h](../../../src/kernels/image/pipeline.h))
  dispatched through a plan bound against the implementation registry
  (D-053): `--plan legacy` is the kernels above, `--plan fast` (the
  default) the speed slice's.

## Results (`spark`, 2026-09-28)

GB10, driver 580.178.04, the SDK's CUDA and cuBLAS (D-076). Raw outputs in
`~/.local/share/llmp/m3img-20260928/` on `spark` (the runs `full1`,
`first`, `forced`, `vae` and `released`, and compare.py's verdicts in
`verdicts.json`). A first attempt at the reference run exhausted `spark`'s
memory at 03:39 (its FP32 controls with the BF16 pipeline still resident);
the kernel's OOM killer ended it and several system services (tailscaled,
polkit, fwupd, the DGX telemetry, the user session), which systemd
restarted within two minutes. The reference was re-run with reference.py
as checked in and finished at 03:52; every llmpalooza run here started after
04:00 on the recovered host, and diffusers' full-generation and per-step
timings are the earlier baseline's, so no measurement overlaps the
incident.

**Correctness** ([compare.py](compare.py), bounds 1–6): every bound passes.

| Check | Result | Bound |
| --- | --- | --- |
| 1. Tokens | 39 of 39 equal, 14 dropped | exact |
| 2. Text encoder | rel. RMS 0.0437, cosine 0.99905 | ≤ 0.20, ≥ 0.99 |
| 3. DiT, first step | rel. RMS 0.00644, cosine 0.999980 | ≤ 0.0125, ≥ 0.99996 |
| 4. DiT, 40 teacher-forced steps | worst rel. RMS 0.0094 (step 39), median 0.0041 | ≤ 0.046 |
| 5. VAE on diffusers' latents | rel. RMS 0.0035; PSNR 55.9 dB, SSIM 0.9992 | ≤ 0.0055, ≥ 47.5 dB |
| 6. **Image, end to end** | **PSNR 41.8 dB, SSIM 0.99604** | ≥ 32.0 dB, ≥ 0.98 |

End to end, llmpalooza's final latents are 0.0174 from diffusers' in relative
RMS, closer than the FP32-DiT control's 0.0254; its image and the
reference, inspected side by side at 512², show the same red teapot, pose
and framing with no visible difference. With `--phases released` the pixels are identical to the
resident run's.

**Performance and memory** (reported; plan.md's image gate is ≤ 10% slower
than diffusers, D-085):

| | Llmpalooza | diffusers ([baselines](../fast-swap/baselines.md#qwen-image-21-diffusers-bf16)) | Ratio |
| --- | --- | --- | ---: |
| Full generation, weights resident (plain runs) | 36.94 / 36.99 s | 52.55 / 52.71 / 52.73 s | 0.70 |
| Per denoising step, median (synchronized) | 0.893 s [0.885–0.900] | 1.259 s [1.255–1.263] | 0.71 |
| First step (with the text's K/V prefill), from the loop's start | 0.889 s | 1.753 s | 0.51 |
| Text encode / VAE decode | 0.137 s / 1.08 s | 1.95 s / 1.46 s (this study's reference run) | |
| Peak memory, drop in `MemAvailable`, resident | 31.2 GiB | 43.4 GiB | 0.72 |
| Full generation, components released outside their phases | 39.89 / 40.15 s (loads 1.59 + 1.51 + 0.83 s, warm page cache) | | 0.76 |
| Peak memory, released | 15.9 GiB | | 0.37 |

Every run's image is the same pixels (RGBA SHA-256 `95fbcbc5…`), resident
or released, first or plain.

Where a step's time goes (nsys, steps 0–2, per image-row step): the 224
products 0.69 s (cuBLAS picks `nvjet` kernels for the MLP's and a CUTLASS
kernel at 1.76 ms for each 4,096³ one, where PyTorch's cuBLASLt path takes
`nvjet` at about 1.54 ms), attention 0.108 s (3.37 ms per block; PyTorch's
`flash_fwd_kernel` 3.38 ms), everything else about 0.10 s (diffusers'
unfused elementwise kernels: about 0.38 s per step, from its torch.profiler
table of steps 0 and 1).

**Attention, chosen by speed** (D-053): GGML's tensor-core kernel covers
D = 128 with one query head per KV head once its ncols2 = 1 instances are
built (`kernels/ggml/fattn_mma_d128.cu`, then `--attention ggml`, an arm
the speed slice retired from the harness; RE-030's sinks
issue does not arise without sinks) and matches FP64 within upstream's
bound, but takes 19.9 ms per block call here (0.64 s per step, 64 columns
per tile, stream-k), about 14 TFLOPS. Llmpalooza's own FlashAttention-2 kernel
(`kernels/image/flash_attention.cu`, BF16 `mma.sync`) takes 3.37 ms, as
PyTorch's does, so neither cuDNN (whose license D-017 would first have to
admit) nor a cuBLAS formulation was needed.

## Speed

The speed slice (2026-09-28, the owner: "see how much we can squeeze out of
it without sacrificing quality"): the same BF16 numerics, lower-precision
levers (FP8 linears, step caching) left to the quality/performance-modes
item ([plan](../../plan.md)). Each lever is an implementation a plan names
(D-053; [implementations.h](../../../src/kernels/image/implementations.h)),
so turning one back is one `--choose`:

- **Pinned cuBLASLt products** (`image.linear.cublaslt`,
  [gemm.h](../../../src/kernels/image/gemm.h)): for each product shape the
  pipeline makes at 1,024² with the teapot prompt, the fastest of cuBLASLt's
  heuristic candidates whose output equals `cublasGemmEx`'s bit for bit
  (five of the sixteen pins split K, reduced in a fixed order, as
  `cublasGemmEx`'s own choice evidently does there), timed sustained by
  `llmp_qwen_image_gemm_tune` (median of five batches of back-to-back
  launches after half a second of warm-up) and pinned by its nine
  algorithm attributes, as the EXL3 reconstruction's are; other shapes,
  and a pin a later cuBLASLt would refuse, take cuBLASLt's first heuristic
  choice. The 4,096³ products run `nvjet` 192×144 tiles at 1.48 ms against
  `cublasGemmEx`'s CUTLASS kernel at 1.64 ms, the MLP's up projections
  4.21 against 4.35 ms; its down projection has no faster candidate.
- **The gated residual fused with the next norm**
  (`image.gated_residual_norm`): the residual written back and the next
  block's (or `norm_out`'s) layer norm and modulation of the new rows in
  one pass, the norm's sums in `LayerNormModulate`'s order, so the same
  bits; the norm kernels keep their chunks in registers at a compile-time
  count, and SwiGLU takes eight elements per thread (both plans).
- **Attention reads the prefix cache in place**
  (`image.flash_attention.prefixed`): the text rows' K and V are read from
  the cache instead of being copied in front of the image rows each block
  (64 copies a step), the same keys in the same order.
- **The query norm and rotation in the attention's registers**
  (`image.flash_attention.norm_q`): each query row normed and rotated as
  `HeadNormRopeComplex` computes it, its sum of squares in that kernel's
  order and its contractions spelled out as that kernel compiles them
  (RE-035), so the same bits, and the rotated queries never written.
- **Implicit-GEMM convolutions** (`image.conv2d.implicit`,
  [conv.cu](../../../src/kernels/image/conv.cu)) for the VAE's 3×3
  convolutions: 128 pixels × 144 channels per block, the input patch with
  its halo and all nine taps' KRSC weights per 16 input channels in
  shared memory, `mma.sync` BF16 with F32 accumulation, the bias added
  before one rounding, as the im2col path's bias fill and `beta = 1` do.
  The sums run in another order than cuBLAS's over the im2col matrix, so
  this lever alone changes bits (the VAE's).
- **A CUDA graph per denoising step** (both harnesses): the third step
  captured once and replayed, its sinusoid and `dt` uploaded to fixed
  places first (the Euler step reads `dt` on the device); on the paged
  node the capture lives for the runner's life, every address it holds
  pinned (D-090).
- **Tried, not kept or not needed:** batching q, k and v (or gate and up)
  as one strided product gave nothing over the pinned single products
  (tuner: 4.40 ms for three 4,096³ against 3 × 1.47, and 8.52 ms for the
  batched pair against 2 × 4.12); a third key and value stage in the
  attention (96 KiB) ran it at 3.43 ms against 3.49 in the profile but
  32.56 s against 32.58 s end to end (four plain runs each, `spark-b`), so
  it stays at two; the prefix cache read in place, alone, did not move the
  full generation measurably (kept as the query norm's base, which does); the attention runs at about 79 TFLOPS against the
  products' 93–100, the rest of the gap needing another design than
  FlashAttention-2 on `mma.sync`. The text encoder takes 0.07 s and was
  not worked on beyond its products' pins.

**Results** (`spark`, 2026-09-28, the host otherwise idle for the batch;
GB10, driver 580.178.04, the SDK's CUDA 13.4 and cuBLAS 13.8.0.4; the M3
slice's build (`c9a17ac`) and this slice's, two plain runs per arm, the
fast plan run between every two other arms; raw outputs in
`~/.local/share/llmp/m3imgspd-20260928/` on `spark`, the `h-` runs):

| | M3 slice | Speed slice | diffusers |
| --- | ---: | ---: | ---: |
| Full generation, weights resident (plain runs) | 35.95–36.39 s (mean 36.19, 4 runs) | 32.97–34.17 s (mean 33.41, 14 runs) | 52.6 s |
| Against diffusers | 0.69 | **0.64** | 1 |
| Per denoising step, median | 0.870–0.874 s | 0.811–0.826 s | 1.259 s |
| Text encode / VAE decode (first run) | 0.137 / 1.09–1.10 s | 0.072 / 0.36–0.41 s | 1.95 / 1.46 s |
| Peak memory, resident (`MemAvailable` drop) | 30.5–32.3 GiB | 31.5–31.8 GiB | 43.4 GiB |
| Full generation, released, cold page cache | 40.36 s, peak 16.6 GiB | 37.70 s, peak 15.2 GiB | |
| Swap Qwen3.8 → image, to the first step's output (first use / prepared) | 5.28 / 5.21 s, of it encode and step 1.60 / 1.58 s | 5.11 / 5.09 s, 1.52 / 1.52 s | |

The drift over the hour (the fast plan's arms from 32.97 to 33.55 s) is
larger than a single lever's gain, so each lever is judged against the fast
arms either side of it:

| Lever turned back (`--choose`) | Full generation | Gain |
| --- | ---: | ---: |
| products on `cublasGemmEx` | 34.08 s | 0.91 s |
| VAE convolutions by im2col | 33.94 s | 0.53 s (decode 0.36 → 0.95 s) |
| attention without the query norm (prefix read in place kept) | 33.84 s | 0.39 s |
| attention as the M3 slice's (the prefix copied, the query normed apart) | 33.72 s | 0.31 s (the prefix alone: within noise) |
| no step graphs | 33.71 s | 0.29 s |
| residual and norm apart | 33.56 s | 0.11 s (0.5 s on `spark-b`) |
| the M3 slice's plan (`--plan legacy`) | 35.63 s | 2.2 s |

The legacy plan in this build is 0.56 s faster than the M3 slice's build:
both plans share the harness's changes (working memory laid out once
rather than allocated per phase, which took 0.065 s off the encode; the
norms' chunks in registers and the eight-wide SwiGLU).

Where a step goes (nsys, kernel time per step; before: all 40 steps of the
M3 slice's build; after: steps 0 and 1 of this slice's, launched one by
one, since nsys traced graphs as single launches): products 648 → 610 ms,
attention 112 → 111 ms (the query norm's 4.8 ms moved into it, its kernel
no slower), everything else 112 → 87 ms (the separate gated residual,
26 ms, gone into the norm; the rotation of q, 17.5 → 12.7 ms for k alone),
GPU idle between operations 2 ms a step before.

**Quality** ([compare.py](compare.py), bounds 1–6, run in the
`llmp-exl3-reference:20260922` container on `spark-b`, which has NumPy
and Pillow; the same numbers in `llmp-image-reference:20260922` on
`spark`): every bound passes, and every number but the VAE's and the
image's is the M3 slice's to the last digit, since the text encoder and the
denoiser write the same bits (the fast plan's final latents equal the
legacy plan's byte for byte):

| Check | Speed slice | M3 slice | Bound |
| --- | --- | --- | --- |
| 1. Tokens | 39 of 39 equal, 14 dropped | the same | exact |
| 2. Text encoder | rel. RMS 0.0437, cosine 0.99905 | the same | ≤ 0.20, ≥ 0.99 |
| 3. DiT, first step | rel. RMS 0.00644, cosine 0.999980 | the same | ≤ 0.0125, ≥ 0.99996 |
| 4. DiT, 40 teacher-forced steps | worst rel. RMS 0.00937 (step 39) | the same | ≤ 0.046 |
| 5. VAE on diffusers' latents | rel. RMS 0.00354; PSNR 55.93 dB, SSIM 0.9992 | 0.0035; 55.9 dB, 0.9992 | ≤ 0.0055, ≥ 47.5 dB |
| 6. **Image, end to end** | **PSNR 41.770 dB, SSIM 0.99604** | 41.768 dB, 0.99604 | ≥ 32.0 dB, ≥ 0.98 |

The image is repeatable: every run of the fast plan, first or plain,
resident, released or paged, gives the same pixels (RGBA SHA-256
`3b7770ca…`); the legacy plan still gives the M3 slice's (`95fbcbc5…`).
The two differ only through the VAE's convolutions' summation order
(decoded tensor 0.0180959 against 0.0180977 from diffusers' in relative
RMS).

## Judgement calls

- **Components as separate artifacts plus a composition** (D-089)
  rather than one artifact: no change to v0 artifacts or the pinned
  planner, and a shared text encoder is one artifact.
- **Importer policy:** `import_m3.py` runs the pinned `layout.py` with a
  per-component layer pattern (the language layers, the transformer
  blocks, the VAE decoder's up blocks) and the diffusers `_class_name` as
  architecture, both patched into its module instance; GGUF imports write
  exactly what `m3-1` wrote and keep its converter version.
- **Seeded noise from diffusers' saved latents** rather than a
  reimplementation of PyTorch's Philox generator.
- **Own kernels for the elementwise and norm operations** and for
  attention: GGML computes the former in F32 between operations where the
  reference rounds each to BF16, and its attention was 6× slower here.
- **Bounds from calibration:** twice the measured BF16-versus-FP32 distance
  per component, fixed before llmpalooza's first run.
- **Speed without new numbers** (the speed slice): every lever but the
  VAE's convolution keeps the M3 slice's bits. Pinned algorithms are chosen
  among the candidates that write `cublasGemmEx`'s bits, not the fastest
  of all (for three of the small text-encoder and `txt_in` shapes the
  fastest candidate writes other bits, about 2% faster on products of
  0.015–0.49 ms); fused kernels reproduce the unfused
  kernels' summation orders and contractions, checked byte for byte. The
  implicit convolution is the one lever that sums in another order; its
  image is 41.770 dB against the M3 slice's 41.768.
- **Pins recorded in the source** (`gemm.cc`, covered by the module's
  identity digest) for the GB10, the pinned cuBLASLt and the teapot's
  shapes, rather than tuned at start-up: deterministic and reviewable; any
  other shape or device takes cuBLASLt's first heuristic choice, also
  deterministic for a given cuBLASLt and device.
- **One pipeline for both harnesses** in `kernels/image` rather than two
  copies of the phases (the paged runner's was a copy): the registry-bound
  plan (D-053) that the runtime needs, and every lever lands once.
- **The GGML attention arm retired** from the harness: its A/B (19.9 ms
  against 3.37 ms per block) stands as measured, and keeping it would have
  kept a second attention path outside the bound plan.
- **A lever kept only if the full generation moves** (the owner's rule):
  the third attention stage and batched products were measured and
  dropped.

## Limits

- One prompt, size (1024²), step count and seed; text-to-image only (no
  condition images, no guidance, the prefix cache on). Other sizes are
  accepted (multiples of 32, 64–2,048) but not compared: one 512², 8-step
  run from seeded noise of llmpalooza's own (not diffusers') gave a coherent
  image, and the attention kernel's tests cover that size's sequence
  lengths.
- The phases here run on `cudaMalloc` memory. On the paged node each phase
  is a device job over its own component's closure, a generation is one
  request leasing all three components once (D-093), and the image is
  this harness's pixel for pixel (RGBA SHA-256 `95fbcbc5…` with the M3
  slice's kernels, `3b7770ca…` with the speed slice's,
  [swap](../fast-swap/swap.md#qwen-image-21-on-the-paged-node)). Both run
  the same pipeline through the same bound plan ([Speed](#speed)).
- The pinned products hold for the GB10 (48 SMs) and the pinned cuBLASLt
  at the teapot prompt's shapes (1,024², 39 tokens); elsewhere cuBLASLt's
  first heuristic choice runs, which need not write cublasGemmEx's bits.
  One other size was run, 992² (the teapot, 40 steps, the reference's
  initial latents cut to size; `spark`, 2026-09-28), whose VAE widths are
  not all multiples of 8: the fast plan's final latents equal the legacy
  plan's byte for byte, its image 62.4 dB from the legacy plan's (the
  convolutions' order); neither was compared with diffusers there.
- Load times are with a warm page cache.

## Reproduce

On `spark`, with the checkpoint in the M3 model store and a build of
`llmp_qwen_image_exec` (`benchmarks/`):

```sh
M=$HOME/.local/share/llmp/models/Qwen/Qwen-Image-2.1@790c9263
S=$HOME/.local/share/llmp/m3-artifacts
cd docs/experiments/artifact-layout
for role in text_encoder transformer vae; do
  python3 import_m3.py component $S ../fast-swap/pins.json qwen-image-2.1 $M $role \
    $([ $role = text_encoder ] && echo --meta text_encoder/generation_config.json)
done
python3 import_m3.py compose $S ../fast-swap/pins.json qwen-image-2.1 $M \
  text_encoder=<id> transformer=<id> vae=<id> --meta scheduler/scheduler_config.json \
  $(cd $M && for f in processor/*; do echo --meta $f; done)
# The reference (about 11 minutes, in the image-reference container, under
# torch.no_grad; 51 GB of CUDA allocations at its peak). Check MemAvailable
# first; --memory caps the container's host allocations:
sudo -n docker run --rm --memory 96g --device nvidia.com/gpu=all --network none -u $(id -u):$(id -g) \
  -e HOME=/tmp -v $M:/model:ro -v $PWD/..:/exp:ro -v $REF_PARENT:/out \
  llmp-image-reference:20260922 /exp/qwen-image-native/reference.py /model /out/ref1 \
  /exp/fast-swap/prompts.json
# Llmpalooza: end to end (with two plain timed runs), then each component alone.
# The fast plan and step graphs are the defaults; --plan legacy is the M3
# slice's kernels, --choose ROLE=IMPLEMENTATION turns one lever back, and
# --graphs off replays nothing (Speed).
B=build/spark-native/benchmarks/llmp_qwen_image_exec; C=eca21baa...; R=$REF_PARENT/ref1
$B --store $S --composition $C --out full --reference $R --runs 2
$B --store $S --composition $C --out first --reference $R --embeds reference --stop-after 1 --no-vae
$B --store $S --composition $C --out forced --reference $R --embeds reference --force-latents --no-vae
$B --store $S --composition $C --out vae --reference $R --embeds reference --stop-after 1 --decode-reference
$B --store $S --composition $C --out released --reference $R --phases released --runs 1
# The verdicts, in the image-reference container (or any with NumPy and
# Pillow; see Speed for the speed slice's):
python3 compare.py $R --tokens full --text full --image full --dit-first first \
  --dit-forced forced --vae vae
# The products' candidates at the pipeline's shapes (the pins' source):
build/spark-native/benchmarks/llmp_qwen_image_gemm_tune --candidates 16 --reps 10 \
  --shape 4096,4096,4096 --shape 4096,12288,4096 ...   # each m,n,k the pipeline makes
```
