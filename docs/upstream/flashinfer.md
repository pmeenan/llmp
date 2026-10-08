<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# FlashInfer

- **Repository:** [flashinfer-ai/flashinfer](https://github.com/flashinfer-ai/flashinfer),
  Apache-2.0.
- **Inspected pin:** 0.6.17, commit
  `a0a6b019b9b27d49d209f85d028a1ae5a9b347d7`, installed in Mia's image
  `sha256:fc120ece0a388cc0aa1caad4a9f1cd92113484ab7ec2fd0efadd62585be05bf8`.
  This is an external source experiment, not a llmpalooza source-lock dependency.

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
- **Llmpalooza outcome:** external source specialization and captured-operand
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

## Pinned Python profile override is ignored (RE-040)

- **Status:** open; pinned-source finding, not reported upstream.
- **Found:** 2026-09-30, FlashInfer0.6.17/a0a6b019 in the exact image above.
- **Problem:** `fused_moe/core.py`, SHA256
  `1fc7f2b942b253837e554b2a4c128e196a19a48679c4a11cdd7677e0cccfd5ca`,
  documents `profile_ids` as a two-absolute-index override at1074–1076,
  but its SM120 wrapper at546–610 accepts the value and unconditionally
  invokes `AutoTuner.choose_one` for both products. Passing explicit IDs
  into this public Python path does not establish the requested tactics.
- **Native source:** `flashinfer_cutlass_fused_moe_binding.cu`, SHA256
  `af8afe9d012744f02aa50cda309beeaeda9fc60468e8607f0e3b68b6ad913480`,
  validates and uses two absolute IDs in `setRunnerProfiles` at839–873.
  `[-1,-1]` selects first native entries; it is not an API/autotuner cache
  selection. Finalize copies precede SwapAB copies in the pinned list.
- **Llmpalooza workaround:** external replay calls that actual installed
  native binding with source/trace-proved19/56 and ordered19/36, preserving
  all operand, weight-transform and graph proofs. The full traced fused
  consumer is2.32% slower cold and effectively neutral warm; ordered is
  1.75%/1.73% slower. No production implementation is imported.
- **Upstream master:** not checked. A narrow proposed action is to honor
  valid supplied IDs before autotuning, or clarify/remove the override.
  No issue or PR has been sent.
- **Links:** [RE-040](../rough-edges.md#re-040-flashinfer-accepts-a-python-profile-override-but-its-sm120-wrapper-still-autotunes-both-products--2026-09-30-status-worked-around),
  [complete-consumer report](../experiments/qwen38-fi-down-stage/README.md#complete-consumer-follow-up).
