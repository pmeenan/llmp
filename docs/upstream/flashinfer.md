<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# FlashInfer

- **Repository:** [flashinfer-ai/flashinfer](https://github.com/flashinfer-ai/flashinfer),
  Apache-2.0.
- **Inspected pin:** 0.6.17, commit
  `a0a6b019b9b27d49d209f85d028a1ae5a9b347d7`, installed in Mia's image
  `sha256:fc120ece0a388cc0aa1caad4a9f1cd92113484ab7ec2fd0efadd62585be05bf8`.
  This is an external source experiment, not a jitLLM source-lock dependency.

## Literal NVFP4 downstream-stage transfer

- **Status:** won't fix; source behavior by design, no upstream defect claimed.
- **Found:** 2026-09-30, Spark GB10. The actual vLLM main-MoE wrapper uses
  `FLASHINFER_CUTLASS`, although its autotuning tags say
  `trtllm::fused_moe`. Tags do not identify a TRTLLM backend. The drafter
  uses Marlin.
- **Source boundary:** vLLM's NVFP4 prepare path first calls private
  `torch.ops._C.scaled_fp4_quant.out`. FlashInfer expands prequantized
  codes/scales there; its optional BF16-input expansion is a different
  branch. The installed Python source alone does not expose the private
  C quantizer implementation, so it cannot substantiate a literal first
  product transfer.
- **Visible downstream arithmetic:**
  `data/csrc/fused_moe/cutlass_backend/cutlass_fused_moe_kernels.cuh`,
  SHA256 `fd24f5f8234b0736f205dd2540f47dcaf90783a53c2fbbab66d0490c9494dbac`,
  carries NVIDIA2020–2025 Apache-2.0 notices. Its activation path computes
  F32 SwiGLU from BF16 products and rounds to BF16 before conversion.
  `data/csrc/nv_internal/tensorrt_llm/kernels/quantization_utils.cuh`,
  SHA256 `42dbbc12195c09279bf4f1bd3e170c8664138f2f7e7c59fd38b6618ae8253656`,
  carries NVIDIA2019–2023 Apache-2.0 notices. The default NVFP4 branch
  uses16-value scales, E4M3 RN, approximate-FTZ reciprocal and E2M1
  RN/saturate. Both holders apply to the external specialization.
- **Wrapper pins:** FlashInfer `fused_moe/core.py` SHA256
  `1fc7f2b942b253837e554b2a4c128e196a19a48679c4a11cdd7677e0cccfd5ca`;
  vLLM `model_executor/layers/fused_moe/experts/flashinfer_cutlass_moe.py`
  SHA256 `52887cec5fe628e0a4e7793deb546b40f6c29f7362b3b436c10ca36934ab9826`.
  Static activation scales are layer maxima over512 experts; reciprocal
  encoding and raw-weight-scale multiplication are F32 CUDA operations.
  A no-model pinned-image probe records their actual bits and environment.
- **jitLLM outcome:** external source specialization and captured-operand
  replay only. Complete G2 preparation/product/combine is20.95–27.64%
  slower cold, so no arithmetic integration or model quality claim follows.
  The original captured output and actual scalar bits pass before timing.
  This does not reject FlashInfer's fused whole implementation. Initial
  quantization, fused preparation and finalization need separate proof.
- **Upstream master:** not checked; this entry describes only the measured
  pinned consumer. No issue or PR is proposed.
- **Links:** [literal G2 study](../experiments/qwen38-fi-down-stage/README.md),
  [earlier grouped schedules](../experiments/qwen38-grouped-verify/README.md).

No FlashInfer implementation is vendored by this slice. A future retained
copy needs an archive identity, source lock, kept-file license audit,
original notices and changed-file attribution in the same unit.
