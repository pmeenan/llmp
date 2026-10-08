<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Additional reference-model candidates

Owner-added and source-inspected 2026-09-22; the first bounded reference for
each ran the same day (reports linked below). These are external references,
not measured llmpalooza support. Existing reference results and the canonical
DeepSeek/Qwen switching workload remain unchanged.

| Candidate | Reference role | Status and next evidence |
| --- | --- | --- |
| Qwen-Image-2.1 GGUF | Quantized image-component footprint and repeated denoising, compared with the [BF16 study](image-reference/README.md) | [Bounded GGML-runner study done](image-gguf/README.md): pinned stable-diffusion.cpp, component identity reconciled, full-pipeline memory, same-runner quantization agreement, phase release and budgeted disk-backed denoising. Next, only if scheduled: native GGML execution in llmpalooza, a quality metric beyond pixel agreement, text↔image switching with this runner |
| MiMo-V2.6-Flash-RL | Large sparse MoE across two Spark memory domains | [Bounded text-only TP=2/EP=2 reference done](mimo-reference/README.md). Next: switching/state and multimodal cases separately, repeated boots and concurrency; native sharding remains M4 |
| Future smaller MiMo quantizations | Single-Spark resident or forced-paging case | Revisit when an artifact and compatible packed kernels exist; measure full execution budget and quality |

## Qwen GGUF

The [repackaged model card](https://huggingface.co/abenzerps/Qwen-Image-2.1-GGUF/blob/c4de66efa2183fb25ecbc185bac509ee37952b41/README.md)
provides denoiser GGUFs from Q4_0 (4,050,899,616 bytes) through Q8_0
(7,591,557,792 bytes); Q4_K_M is 4,604,557,984 bytes. These are file sizes,
not full-pipeline memory. Its companion choices include a 17.53 GB BF16 or
9.35 GB INT8 text encoder and roughly 676 MB BF16 VAE. The card points to
the leejet ComfyUI-GGUF fork; availability in GGUF does not establish native
GGML execution support. Record the chosen runner and companion-file hashes
before comparing identical prompts, seeds, sizes and step counts. Quantized
outputs need a quality comparison, not an assumption of BF16 pixel identity.
Audit repackaging provenance and applicable terms before execution; a format
conversion does not remove the original research-license boundary.
The quant card lists source revision
`b3179ad355be050328e483a9dfdd9e60cd62adfa`; our BF16 baseline uses the upstream
Qwen revision `790c92633540aa0cb11d9abf19eb46d861714758`. Reconcile component
identities before attributing output differences to quantization alone.
Reconciled 2026-09-22: the two revisions share all weights, and the
repackaged text encoder, VAE (after BF16 rounding) and BF16 denoiser match
the baseline tensor by tensor ([GGUF study](image-gguf/README.md#identity-and-reconciliation)).

## MiMo

The [official card](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-RL/blob/5711b268169967567844e1e560e8a3966da959b1/README.md)
describes 309B total / 15B active parameters and text, image, video and audio
input. The [pinned config](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-RL/blob/5711b268169967567844e1e560e8a3966da959b1/config.json)
has 48 layers, 256 routed experts with top-8 selection, no shared experts,
and a dense first layer. Attention mixes 128-token sliding windows and global
layers. Its configured 1,048,576-token context is not a validated Spark
capacity. `num_nextn_predict_layers` is 3, whereas the card describes five
MTP layers: resolve execution geometry from the selected artifact and runner
before measuring speculation or state. Do not execute checkpoint custom code
as part of native import.

[Mia's two-Spark recipe](https://github.com/MiaAI-Lab/MiMo-V2.6-Flash-2x-DGX-Sparks/tree/201be3e743d215291feee09fe470822ec453c63d)
reports SGLang TP=2 over RoCE, roughly 161 GiB of checkpoint weights and
82.7 GiB resident weights per rank. These are external observations, not our
measurements. Its MXFP4 experts must stay packed; it reports that older
loaders expand them to FP8 and exhaust memory. Pin and audit the full runner,
patches, dependencies and artifact terms before reproduction. Keep our
configured topology and local/authenticated access policy.
The recipe pins checkpoint revision `3b38d063180c3e4aed9691fdc735f3d10b266ee4`,
different from the card/config inventory above. Reconcile those artifacts
before using this inventory to interpret a reproduction. Reconciled 2026-09-22:
only the README, report PDF and a DFlash config JSON fix differ; weights,
configs and tokenizer are identical ([reference](mimo-reference/README.md#identity)). The recipe declares
AGPL-3.0; inspecting its results does not authorize incorporating its code
into the Apache-2.0 core.

Start with a bounded text workload and record per-node physical peaks,
weight/state/workspace bytes, loading, prefill/decode and transport behavior.
Account for optional draft/MTP weights and KV separately. Add modalities only
after that baseline passes. A future smaller quant is a revisit trigger,
not a promised arrival date or one-node fit: include KV, workspace, runtime
and headroom in the single physical budget. If it exceeds that budget, it
can still become a paging candidate once native architecture and kernels
are validated; reference success alone does not establish that support.
