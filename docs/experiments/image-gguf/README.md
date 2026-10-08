<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Qwen-Image-2.1 GGUF comparison on a GGML runner

External reference on `spark` (2026-09-22), extending the
[BF16 diffusers study](../image-reference/README.md) to the owner-supplied
[GGUF candidate](../model-candidates.md). It is not native llmpalooza image
support. Raw images, logs and per-case results stay outside Git;
[`aggregates.json`](aggregates.json) holds the reduced evidence.

**Result.** A pinned GGML runner (stable-diffusion.cpp) executes the complete
Qwen-Image-2.1 pipeline from the Q4_K_M and Q8_0 GGUFs and from a BF16
control on GB10, deterministically and with exact pixel agreement across
repeats, cold/warm loads, a second node, a rebuilt binary and phase-released
parameters. Quantization mainly shrinks the denoiser (4.29 vs 13.25 GiB of
parameters); it does not speed this runner up. The runner is 2–3× slower per
step than the diffusers BF16 baseline, and at 2048² its VAE decode reserves a
38.3 GiB compute buffer. Upstream GGML (the fork's merge base) matches the fork
exactly through 1024² but aborts in the 2048² VAE decode on a 32-bit stride
assert (RE-011). Disk-backed segmented denoising under a 3 GiB managed budget
was bit-identical to resident execution with the same VAE tiling and ran the
whole request in about 5 GiB of sampled host memory, at 2.6× the outward time.

## Results — 2026-09-22

Host `spark-c4e2` (GB10, driver 580.178.04), one observation per case,
sequential on an otherwise idle node; the cross-node controls ran on
`spark-56f5`. Seconds are sd.cpp's phase timers; *host GiB* is the sampled
`MemAvailable` drop. sd.cpp's "MB" figures are MiB and are given here in GiB;
read and file sizes are decimal GB. Page cache was warm unless noted.

| Case (resident) | Params GiB | Load | Sampling | Decode | Host GiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| Q4_K_M 512²/4 | 20.09 | 1.00 | 1.95 | 0.85 | 24.66 |
| Q4_K_M 1024²/4, cold (21.57 GB read) | 20.09 | 4.14 | 10.68 | 3.34 | 31.46 |
| Q4_K_M 1024²/4, warm ×2 | 20.09 | 1.00–1.01 | 10.55–10.62 | 3.25–3.37 | 32.61–33.30 |
| Q4_K_M 1024²/40 | 20.09 | 1.00 | 106.87 | 4.14 | 30.42 |
| Q4_K_M 2048²/40 | 20.09 | 1.00 | 762.02 | 14.86 | 62.81 |
| Q8_0 512²/4 | 22.87 | 1.21 | 1.77 | 0.84 | 28.43 |
| Q8_0 1024²/4 | 22.87 | 1.21 | 10.29 | 3.32 | 35.21 |
| Q8_0 1024²/40 | 22.87 | 1.21 | 103.05 | 3.30 | 33.86 |
| Q8_0 2048²/40 | 22.87 | 1.20 | 745.67 | 13.87 | 65.78 |
| BF16 512²/4 | 29.05 | 1.28 | 1.85 | 0.86 | 35.18 |
| BF16 1024²/4 ×2 | 29.05 | 1.41–1.50 | 10.06–10.11 | 3.29–3.35 | 39.01–42.79 |
| BF16 1024²/40 | 29.05 | 1.27 | 100.69 | 3.29 | 41.35 |
| BF16 2048²/40 | 29.05 | 1.28 | 669.64 | 15.47 | 71.88 |

Params are sd.cpp's reported totals: text encoder 15.17 GiB (749 of 750
tensors loaded), VAE 0.63 GiB, and the denoiser at 4.29/7.07/13.25 GiB.
Text conditioning takes 0.15–0.17 s. Denoiser compute buffers are
0.13–0.23 GiB at 512², 0.51–0.89 GiB at 1024² and 2.02–3.53 GiB at 2048²; the
VAE decode buffer is 2.39 GiB at 512², 9.57 GiB at 1024² and **38.26 GiB at
2048²**, the largest single allocation in the pipeline. The diffusers baseline generates 1024²/40 in 52.586 s and 2048²/40
in 252.796 s end to end; this runner's sampling alone is 1.9–2.0× and
2.6–3.0× that. At 2048² BF16 is the fastest representation here.

The container's cgroup `memory.peak` stayed at 2.3–2.9 GiB in warm resident
cases while host `MemAvailable` fell by 25–72 GiB: CUDA allocations on Spark
are not charged to the container's memory cgroup (only page cache read by the
container was, e.g. 21.8 GiB in the cold case). Container memory limits are
therefore not a Spark execution budget.

### Pixel agreement

Within the runner, against the sd.cpp BF16 control at the same size/steps:

| Denoiser | 512²/4 | 1024²/4 | 1024²/40 | 2048²/40 |
| --- | --- | --- | --- | --- |
| Q8_0, PSNR dB / SSIM | 44.61 / 0.9976 | 49.94 / 0.9987 | 18.31 / 0.8778 | 39.50 / 0.9945 |
| Q4_K_M, PSNR dB / SSIM | 35.14 / 0.9892 | 36.12 / 0.9939 | 19.33 / 0.8827 | 19.55 / 0.8839 |

At four steps both quantizations stay close to BF16, Q8_0 closer. Over forty
steps small perturbations can steer the trajectory to a different but valid
image; agreement is not monotonic in precision (Q8_0 diverges at 1024² but not
at 2048²). The three sd.cpp 1024²/40 images and the diffusers one were
visually inspected and all depict the requested red teapot on a wooden table;
the Q4_K_M and Q8_0 images keep the BF16 control's framing, with Q8_0's handle
and spout mirrored. These are agreement measures on
one prompt and seed, not a quality evaluation. sd.cpp BF16 versus the
diffusers baseline gives 23.16/18.78 dB (SSIM 0.941/0.961) at 512²/4 and
1024²/4 and 10.8–11.7 dB (SSIM ≈ 0.71) at 40 steps, where the diffusers image has
a different composition. Different noise and schedule implementations make
that expected, as stated above.

Exact-pixel controls (all identical): Q4_K_M cold vs warm vs repeat; BF16
repeat; phase-disk vs resident at 1024²/40; unconstrained all-disk vs
resident; the 3 GiB segmented case vs a resident run with the same VAE tiling;
a `spark-b` rebuild of the patched runner (whose `sd-cli` bytes differ from the
`spark` build) at 1024²/4; the upstream-GGML build at 512²/4, 1024²/4 and
1024²/40.

### Phase release and disk-backed paging (Q4_K_M)

| Case | Outward s | Sampling s | Decode s | Host GiB | Pixels vs resident |
| --- | ---: | ---: | ---: | ---: | --- |
| Resident 1024²/40 | 114.20 | 106.87 | 4.14 | 30.42 | — |
| Text encoder and VAE disk-backed, 1024²/40 | 116.19 | 109.22 | 3.54 | 17.18 | identical |
| Resident 1024²/4 | 17.34 | 10.55 | 3.37 | 33.30 | — |
| All disk-backed, no budget | 17.56 | 11.28 | 3.62 | 16.67 | identical |
| All disk-backed, 3 GiB budget | 45.62 | 28.85 | 7.06 | 5.23 | tiled VAE only |
| Same, page cache evicted first (20.25 GB read) | 46.53 | 28.57 | 7.04 | 4.66 | tiled VAE only |
| Resident with the same VAE tiling | 20.71 | 10.74 | 6.66 | 25.25 | tiled VAE only |
| All disk-backed, 2 GiB budget | failed | 32.76 | — | 4.32 | — |

Releasing text-encoder and VAE parameters outside their phases cut the sampled
host peak by 13.2 GiB for a 1.7% longer request, with identical pixels
(conditioning grows from 0.16 to 1.67 s because the encoder is loaded for its
phase). Under the 3 GiB budget sd.cpp split the denoiser into 34 segments and
the text encoder into 37, re-staging segments on every step: 126 load
events totaling 23.9 s, and sampling 2.7× slower than resident. Its VAE
decode (9.57 GiB buffer; 10.54 GiB requested with weights) could not fit, so sd.cpp retried with 32×32-latent
spatial tiling. That tiling alone explains the pixel change: a resident run
with the same tiling reproduces the segmented result byte for byte (57.27 dB,
max 6/255 against untiled). Segmented, disk-backed denoising therefore did not
change the computation. The cold run read the files once (20.25 GB); later
re-staging was served from page cache, which `MemAvailable` counts as
available, so these runs do not show behavior when the cache itself is under
pressure. With a 2 GiB budget even the tiled decode (3.36 GiB) did not fit and
the request failed explicitly after sampling; nothing was substituted.

For llmpalooza this is a working external instance of phase-bounded parameter
release and budgeted segment re-staging on GGML: the phase boundary is cheap
and exact, while paging the hot denoiser inside its step loop multiplies
step time and depends on page cache. Workspace, not weights, bounds image
decode at high resolution. None of this is llmpalooza's pager, admission or
completion tracking; the runner's budget excludes driver contexts and is not a
physical cap.

## Identity and reconciliation

[`pins.json`](pins.json) records every input's size and SHA-256:

- `abenzerps/Qwen-Image-2.1-GGUF` at `c4de66efa2183fb25ecbc185bac509ee37952b41`:
  Q8_0 (7,591,557,792 bytes) and Q4_K_M (4,604,557,984 bytes) denoisers, the
  BF16 Qwen3-VL text encoder and the BF16 VAE. The card states conversion
  with stable-diffusion.cpp `1330ceb` from Qwen revision `b3179ad`.
- `Comfy-Org/Qwen-Image-2.1` at `5dc5850eb514a3685f6a03a2641728a8f7549c69`:
  the BF16 denoiser, used as a same-runner control. The GGUF repository's
  text encoder and VAE are byte-identical to Comfy-Org's (same SHA-256).

Qwen revisions `b3179ad` and our baseline's `790c926` differ only in README,
`.gitattributes` and a QR image; weights and configs are identical.
[`identity.py`](identity.py) then matched the repackaged components against
the baseline's hash-verified diffusers files, tensor by tensor and ignoring
names:

| Component | Result |
| --- | --- |
| Text encoder (750 tensors, 17,534,247,392 bytes) | Every tensor byte-identical |
| BF16 denoiser (265 tensors, 14,230,249,472 bytes) | 233 identical; the other 32 are row-concatenations of two upstream projections each (fused); no upstream tensor unused |
| VAE (238 tensors, 675,480,808 bytes) | Every tensor equals the upstream FP32 weight rounded to BF16; 3D-conv shapes differ only by singleton depth |

So the sd.cpp BF16 control and the diffusers baseline execute the same weight
values, and within-runner differences between the BF16 control and a GGUF
isolate quantization. The GGUFs carry no `general.*` metadata; their lineage
beyond the card's statement rests on the hashes and on the output comparison
below. [`inspect_weights.py`](inspect_weights.py) inventories them: Q4_K_M
has 32 blocks of 139,985,408 bytes (Q4_K and Q6_K) plus 125,001,728 bytes
outside blocks; Q8_0 has 231,735,808-byte blocks plus 175,988,736 bytes.
Both keep 68 small BF16 tensors (67,657,728 bytes).

## Runner

[stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) (MIT)
at `c92d73c408515c94beef32161bb5960764fde7a0`, which added Qwen-Image-2.1 on
2026-09-20, with its default patched GGML fork `leejet/ggml@4bf5f600`. That
fork is upstream GGML at merge base `8846b79` (2026-08-12) plus 11 commits
(8,314 inserted lines): sage/sol attention, implicit-GEMM conv2d/conv3d,
FP8 and INT8-convrot support and a `ggml_permute` truncation fix.
[`Dockerfile`](Dockerfile) builds it on a digest-pinned
`nvidia/cuda:13.0.3-devel-ubuntu24.04` arm64 image (the host driver's CUDA
13.0) for `sm_121a` with an explicit `armv8.2-a` CPU target; the built image
is `sha256:a84cfa88…`. [`Dockerfile.upstream-ggml`](Dockerfile.upstream-ggml)
builds the same runner against upstream GGML at that merge base.

Every case uses the BF16 baseline's prompt, seed 42, CFG 1.0 (no
unconditional pass), Euler sampling with sd.cpp's automatic
resolution-dependent flow schedule, `--rng cuda` and flash attention in the
denoiser. The diffusers baseline additionally enabled its prefix cache.
Initial noise and schedule implementations differ between runners, so
**sd.cpp-versus-diffusers pixels are not expected to match**; that comparison
is reported for scale, not as a correctness test.

## Method

[`run.py`](run.py) runs one case in a fresh, network-less, read-only
container as the invoking user, with the three model files mounted
read-only: hash verification (once per denoiser per suite), optional
page-cache eviction (`posix_fadvise` `DONTNEED`), `mincore` residency before
start, 50 ms `MemAvailable` sampling, the container's cgroup `io.stat` block
reads and `memory.peak`, swap/OOM counters and sd.cpp's own phase timings
and buffer reports. [`suite.py`](suite.py) runs the cases sequentially on an
otherwise idle node.

Placements:

- **resident** — `--params-backend cuda0 --eager-load`: all parameters in
  GPU-visible memory, loaded before generation (auto-fit disabled).
- **phase-disk** — text encoder and VAE parameters are disk-backed (loaded
  for their phase, released at module completion); the denoiser stays
  resident.
- **all-disk** — every module disk-backed; with `--max-vram cuda0=N` the
  managed budget is below the denoiser's size, forcing sd.cpp's automatic
  segmented execution to re-stage denoiser segments during each step.

`max_host_used_delta` is the drop in host `MemAvailable` from before the
container started to its minimum; on Spark it covers parameters, compute
buffers and runtime state together. It is a sampled host observation, not
an allocator sum. `cgroup_read_bytes` counts block-device reads charged to
the container (page-cache hits excluded).

[`compare.py`](compare.py) reports exact pixel identity, PSNR and 11×11
Gaussian-window luminance SSIM against a named control. These describe
agreement, not perceptual quality, and no acceptance threshold is implied.
[`summarize.py`](summarize.py) produces the aggregate in the pinned
image-reference container.

## License and provenance boundary

The GGUFs, Comfy-Org files and upstream checkpoint are under the Qwen
Research License (non-commercial research/evaluation), exactly as in the BF16
study; conversion or repackaging does not remove that boundary. This work is
local evaluation; no weights are redistributed. The GGUF card also states the
repository has no content filter; the fixed benign prompt is the only input.
stable-diffusion.cpp and its GGML fork are MIT external reference tools under
D-017; this is not adoption into llmpalooza's core, and the fork's extra commits
are not evaluated for incorporation. CUDA images are platform dependencies
under their own terms.

## Reproduction

```sh
python3 docs/experiments/hf_fetch.py --pins docs/experiments/image-gguf/pins.json --section gguf \
  --output MODELS/gguf
python3 docs/experiments/hf_fetch.py --pins docs/experiments/image-gguf/pins.json --section bf16_control \
  --output MODELS/bf16-control
sudo -n docker build --platform linux/arm64 -t llmp-image-gguf:20260922 docs/experiments/image-gguf
python3 docs/experiments/image-gguf/suite.py --models MODELS --output NEW_PRIVATE_DIR
```

The resident VAE-tiling control is `suite.py --only q4-1024-4-vae-tiled`. The
cross-node arm builds both Dockerfiles on the second node (the upstream one
tagged `llmp-image-gguf-upstream:20260922`) and runs `run.py` with
`--image` for the Q4_K_M cases. `identity.py` and `inspect_weights.py` run in
the image-reference container against hash-verified inputs (the baseline via
`../image-reference/fetch.py`'s `verify_directory`). Summaries and
comparisons run there too, with the suite, the BF16 study's outputs and the
cross-node directory mounted read-only:
`summarize.py SUITE DIFFUSERS_SUITE CROSS_NODE_DIR > aggregates.json`.
The committed file is that output reduced by hand, values unchanged: host and
driver metadata and the three `identity.py` results added, per-case sd-cli
arguments trimmed to the placement tail (one full `example_sd_cli_args`),
segmented cases' load events summed, and comparison sizes dropped.
Harness tests: `python3 -m unittest discover -s docs/experiments -p 'test_hf_fetch.py'`
on the workstation; `test_compare.py` and `test_identity.py` inside that
container. Remove the private model and output directories when no longer
needed.
