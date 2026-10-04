<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Additional M4 references

The owner added [Kindling's GLM-5.3 Flash GB10 recipe](https://github.com/kindlingai/glm-5.3-flash-gx10)
on 2026-09-29. HEAD observed with `git ls-remote` is
`1a2d38af4561c52f274c15f39ddfd62d757a2cfa`; re-pin and audit at M4 entry,
alongside Mia's recipe. This is an additional baseline candidate and
technique source, not a measured jitLLM result or a change to M4's EXL3 target.

Its [README](https://github.com/kindlingai/glm-5.3-flash-gx10/blob/1a2d38af4561c52f274c15f39ddfd62d757a2cfa/README.md)
reports TP2 on NVIDIA's NVFP4 checkpoint: cold prefill 2,929 / 2,864 tok/s
at 32K / 128K, and single-stream decode 60.5 / 36.4 / 89.5 tok/s on
code / prose / structured prompts. Those are author-reported numbers,
with all optimization overlays, restored weight snapshots and warm kernels;
they do not establish a fastest-runtime ranking. Its decode uses RigMark,
temperature zero and low reasoning effort. TP2's stated maximum request
is 160K. This quantization is a speed/memory comparator for the EXL3
target, not its correctness oracle.

The [optimization notes](https://github.com/kindlingai/glm-5.3-flash-gx10/blob/1a2d38af4561c52f274c15f39ddfd62d757a2cfa/experimental/README.md)
are worth studying for adaptive DFlash2 verify lengths, replaying accepted
KDA tokens from one saved recurrent state, fused NVFP4 MoE prefill,
GB10 sparse MLA, sequence-parallel prefill and quantized gathers,
RDMA collectives using both ConnectX PCIe roots, and independent drafter
state pools. These are per-operation or scheduling ideas; reuse remains
subject to the actual files' licenses and D-053/D-080.

The headline configuration also requantizes checkpoint BF16 dense layers
to FP8/NVFP4; its README reports about 1% prose NLL cost. Under D-085,
measure that separately as a quality/performance mode, off by default
in jitLLM. Compare configurations with their dense-layer precision stated.
At M4 entry, record the image and checkpoint pins, overlay flags, topology,
memory, prompts, reasoning mode, warmup and restart conditions, and run
the same two-node workload ourselves before making performance claims.

## TensorFold on two Sparks

TensorFold is an M3.5 and M4 competitive target wherever it supports the
configuration. Use the latest revision **at each applicable task**, then
freeze it for matched runs, under [reference comparisons](reference-comparisons.md).
That document records the 2026-10-04 coverage observation, including
experimental CUDA EXL3/TR3 for GLM; confirm layout compatibility with M4's
target before selecting a same-format oracle. The pins below are historical
observations, not default versions for future work.

The previously inspected
[GLM recipe](https://github.com/ashhart/TensorFold/blob/9cd52ab4daba68ddd09be89be8f23ad43175e821/docs/recipes/glm-5.3-flash.md)
runs a CUDA engine on two Sparks over NCCL in NVIDIA's
`pytorch:26.07-py3` container. The checkpoint
`Vontra/GLM-5.3-Flash-MLX-4bit-MTP` uses MLX affine 4-bit weights,
groups of 64; “MLX” describes the weight format, not a Spark MLX backend.
Rank 1 starts first, then rank 0, both with `--tp 2` and the same master
address. MTP and optional DFlash2 drafting are documented.

The recipe also documents an experimental Mia EXL3 checkpoint path.
Measure the current version as a same-format comparator if its format and pins match
M4's final target, as well as the affine checkpoint's cross-quantization
speed/memory result. Re-pin at entry: M3's audited version is
`71377a5373ed7b394f1b480ba2a6a3986b03af1c`; the observed 2026-09-29
HEAD was `9cd52ab4daba68ddd09be89be8f23ad43175e821`.
