<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# M3.5 model families: capability coverage and checkpoint proposal

**Status: proposal for the owner's approval (2026-09-29).** This answers
[plan.md](plan.md#m35--model-families-pending)'s family-selection item. It
is a desk study: nothing was downloaded or run. Revisions, parameter counts,
sizes and download counts were read from the Hugging Face API and model
files on 2026-09-29, and every fact links its source. Speeds from model
cards, forums and community repositories are creator-reported, not ours.
Weight licenses are recorded for information only (D-087).

The owner framed the goal as capability coverage: M3.5 should cover what
the major open families use, in their current and still widely used older
generations, with the fewest checkpoints. So the capability matrix and the
covering set come first, then formats (EXL3 in depth), then the
per-family picks.

## Summary

Ten checkpoints (1–10 below) cover the architectural features of the
current generations that fit one Spark, and the main formats: GGUF
K-quants and MXFP4 (more I-quants through the optional MiniMax M2.7),
NVFP4 in both layouts, FP8 blocks, AWQ, and EXL3 at several rates, mixed
ones included. GPTQ and per-tensor FP8 reuse the Marlin and FP8 paths;
MLX stays M9's. Three more (11–13) are named by the owner or the plan, or
needed by M7 or M4, and add little new surface. Each has a same-format
reference engine that runs on a GB10. Features that only older or
second-tier checkpoints use (Gemma 2's attention softcap, LongRoPE,
classic SentencePiece, Cohere's LayerNorm and parallel block, short
convolutions, grouped routing) have optional carriers
([below](#optional-checkpoints)); the older generations' features are
judged one by one in [Legacy-tier features](#legacy-tier-features).

| # | Checkpoint (family) | Form | Formats, reference | Max context | New for jitLLM |
| --- | --- | --- | --- | ---: | --- |
| 1 | Gemma 4 26B-A4B-it (Google) | MoE 25.2B/3.8B | GGUF UD-Q4_K_M, llama.cpp; EXL3 mcg 2.54/3.10/4.10, ExLlamaV3 | 262,144 | 5:1 sliding/global, per-type head dims, K=V globals, GeGLU, sandwich norms, final softcap, ▁-BPE 262K, dense MLP beside MoE, KV-sharing assistant drafter, EXL3 MoE |
| 2 | gpt-oss-120b (OpenAI) | MoE 117B/5.1B | GGUF MXFP4 + EAGLE3, llama.cpp | 131,072 | attention sinks with 1:1 window-128/full, head dim 64, biases, clamped (up+1) SwiGLU, softmax-after-top-k, o200k_harmony, harmony, EAGLE3 |
| 3 | MiMo-V2.6-Flash-RL (Xiaomi) | MoE 309B/15B | EXL3 2.27 bpw mixed + MTP + DFlash, ExLlamaV3 ≥ 1.5.2 | 1,048,576 (the quantizer reports 262,144 on one Spark) | EXL3 MoE at scale with mixed widths, window-128 sinks on local layers only, per-type KV heads, qk 192 ≠ v 128 without MLA, 3-layer MTP |
| 4 | Nemotron 3 Super 120B-A12B (NVIDIA) | MoE 120B/12B | NVFP4, vLLM | 262,144 | Mamba2, NoPE attention, squared ReLU, LatentMoE, 512 experts top-22, sigmoid bias-corrected router |
| 5 | Kimi-Linear-48B-A3B (Moonshot) | MoE 48B/3B | GGUF Q8_0, llama.cpp | 1,048,576 | KDA linear attention, MLA (NoPE), tiktoken 163K, Kimi template; previews M4's KDA |
| 6 | Mistral Small 4 119B-2603 (Mistral) | MoE 119B/6.5B | NVFP4 (llm-compressor, Mistral-native files) + EAGLE, vLLM | 262,144 (card) | MLA with q-LoRA, YaRN with Llama-4 position scaling, Tekken 131K, `[THINK]`/`[TOOL_CALLS]`, EAGLE on MLA, compressed-tensors NVFP4 |
| 7 | Llama 4 Scout 17B-16E (Meta) | MoE 109B/17B | GGUF UD-Q4_K_XL, llama.cpp | 10,485,760 (config) | chunked local attention with NoPE layers and their temperature, top-1 sigmoid routing on the expert input, 16 large experts, llama3 RoPE scaling |
| 8 | Llama 3.3 70B Instruct (Meta) | dense 70B | AWQ 4-bit (Marlin), vLLM | 131,072 | dense full-attention GQA at 70B (the KV-read floor), Llama 3 tokenizer, AWQ/Marlin |
| 9 | Muse Glimmer 30B (Meta) | dense 29.8B | GGUF Q4_K_M + DFlash drafter, llama.cpp | 131,072 | 3:1 sliding/NoPE-global, 16:1 GQA, gained QK-norm, output multiplier plus softcap, ATEM template, 202K vocab |
| 10 | Qwen3.8-27B (Qwen) | dense 27.8B | EXL3 mixed SC_3.00 + 4.00, ExLlamaV3; GGUF Q4_K_M + MTP + DFlash2, llama.cpp; FP8 block, vLLM | 262,144 | dense hybrid Gated DeltaNet with plain gated attention, DFlash2 drafter, EXL3 mul1 and mixed widths, FP8 128×128 blocks |
| 11 | Gemma 4 31B-it (Google, owner-named) | dense 31.3B | EXL3 mul1 2.50/3.00/4.00 + mcg 3.00, ExLlamaV3; GGUF UD-Q4_K_XL + assistant, llama.cpp | 262,144 | dense EXL3 at scale, a codebook A/B at one bitrate, the drafter that pays on dense (>2×, creator-reported) |
| 12 | Ornith 1.5 35B-A3B (Ornith, M7) | MoE 36B/3B | GGUF Q4_K_M, llama.cpp | 262,144 | Qwen3.5 MoE (256 experts top-8 + shared) without QSA, hyper-connections or PLE; M7's daily driver |
| 13 | GLM-4.7-Flash (Z.ai) | MoE 31B/3B | GGUF Q8_0, llama.cpp | 202,752 | MTP on MLA, the GLM tokenizer and template; M4's GLM-5.3 path |

Optional, by usage or a feature a smaller set already covers: MiniMax
M2.7 (IQ3_XXS), Laguna S 2.1, Step 3.7 Flash, Ling 3.0 flash, Nemotron 3.5
Lightning, Cohere North Mini Code, Mistral Medium 3.5, and a legacy tier
([below](#optional-checkpoints)). Generative media gets its own track:
MiniMax H3 for video and, owner-named, Ming-Image-0.1-Design as a second
image model ([below](#generative-media-video-and-image)).

Excluded: Phi and Granite (stale or weak, nothing unique in wide use),
Kimi K2/K3, MiniMax M3, GLM-5.x, Mistral Large 3, Llama 4 Maverick,
Nemotron Ultra, DeepSeek V4.1 (too big for one Spark; M4 holds the
two-Spark models). Details [at the end](#considered-and-excluded).

## Capability matrix

What jitLLM runs today comes from M3's models (DeepSeek V4 Flash,
`model/dsv4.h`; Qwen3.8 Flash Next, `model/qwen38.h`; Qwen-Image-2.1,
`model/qwen_image.h`) and the M2 fixtures (Qwen2.5-0.5B, `model/qwen2.h`,
FP16 GGUF and EXL3). "Have" means an M3 model exercises it; "partial"
means a neighbouring form exists. The covering-set column names the
checkpoints above by number.

### Attention

| Feature | Used by (generation) | jitLLM | Covered by | Only too-big checkpoints |
| --- | --- | --- | --- | --- |
| GQA with QK-norm, gated output | Qwen3/3.5/3.8, Laguna, Step | have (Qwen3.8 QSA layers) | 10, 12 | |
| MQA / one KV head | DeepSeek V4 | have | | |
| QKV / output bias | Qwen2.5, gpt-oss, GLM-4.5-Air, Seed-OSS, MiMo-7B | have (Qwen2 fixture) | 2 | |
| Sliding window with dense global layers, plain ring cache | local:global — Gemma 2 (1:1), Gemma 3 and 4 (5:1), Muse (3:1), Cohere 2 (3:1), gpt-oss (1:1), MiMo V2 (39:9), Laguna and Step (3:1) | partial (DeepSeek V4's window sits inside its compressed scheme) | 1, 2, 3, 9 | |
| Head dims or KV heads that differ by layer type | Gemma 4 (256/512), MiMo V2 (KV 8/4), Laguna, Step (query heads) | new | 1, 3 | |
| K = V global layers, V-norm | Gemma 4 | new | 1 | |
| Learned attention sinks | gpt-oss (all layers), MiMo V2 (local layers only), DeepSeek V4 | have (DeepSeek V4 `attn_sinks`) | 2, 3 | Hy4 |
| qk ≠ v head dims without MLA, value scale | MiMo V2 | new | 3 | |
| Attention logit softcap | Gemma 2 only | new | legacy tier | |
| NoPE layers | Llama 4 (every 4th), Muse and Cohere 2 (globals), Nemotron-H (all attention), Kimi Linear (MLA), EXAONE 4.5, Granite 4.0-H | new | 4, 5, 7, 9 | |
| Chunked local attention (8,192), NoPE temperature | Llama 4 | new | 7 | |
| MLA | DeepSeek V2/V3, Kimi K2, Mistral 4, GLM-4.7-Flash, Kimi Linear, Ling 3 | new (DeepSeek V4 has low-rank Q, not a latent KV) | 5, 6, 13 | Kimi K2's 384 experts, Mistral Large 3 |
| Compressed / sparse attention with an indexer | DeepSeek V3.2/V4, Qwen3.8 QSA, GLM-5.x DSA, MiniMax M3 MSA | have (CSA/HCA, QSA) | | GLM-5.x DSA (M4), MiniMax M3 MSA |
| Gated DeltaNet (3:1) | Qwen3-Next, 3.5, 3.6, 3.8, Ornith, OLMo-Hybrid | have (Qwen3.8) | 10, 12 | |
| Kimi Delta Attention (KDA) | Kimi Linear, Kimi K3, Ling 3, GLM-5.3 Flash | new | 5 | K3 |
| Mamba2 (SSD) | Nemotron-H, Nemotron 3, Granite 4.0-H, Falcon-H1 | new | 4 | |
| Lightning (linear) attention | MiniMax Text-01 / M1 | new | — | yes (456B) |
| Short gated convolutions | LFM2/2.5 | new | optional (LFM) | |
| Full attention in every layer at scale | Llama 3.x, Mistral Medium 3.5, MiniMax M2.x, Hy3, GLM-4-0414 | partial (fixture only) | 8 | |

### Positions, norms, activations

| Feature | Used by | jitLLM | Covered by | Only too big |
| --- | --- | --- | --- | --- |
| NeoX RoPE; consecutive-pair (GPT-J) RoPE; partial rotary | most; DeepSeek, GLM, Mistral 4, Cohere, ERNIE | have (Qwen2 NeoX; DeepSeek V4 pairs, `dsv4_graph.cc`) | | |
| YaRN | DeepSeek V3/V4, gpt-oss, Mistral 3/4, Qwen (opt-in), Laguna | have (DeepSeek V4) | 2, 6 | |
| llama3 RoPE scaling | Llama 3.x, Llama 4, Step, EXAONE | new | 7, 8 | |
| Llama-4 position temperature (`llama_4_scaling_beta`) | Llama 4, Mistral 3/4 | new | 6, 7 | |
| Theta and rotary fraction per layer type | Gemma 3/4 (p-RoPE on globals), MiMo V2, Laguna, Step | new | 1, 3 | |
| M-RoPE (text reduces to 1D) | Qwen3-VL, 3.5, 3.8 | have | | |
| LongRoPE | Phi-3.x, Phi-4-mini | new | legacy tier | |
| Linear RoPE scaling | Gemma 3 globals | new | legacy tier | |
| RMSNorm pre-norm | most | have | | |
| (1+w) zero-centred RMSNorm | Gemma 2/3, Qwen3-Next line, Step | not verified (Qwen3.8's handling not found in the code) | 10, 12 | |
| Sandwich norms | Gemma 2/3/4, Muse, GLM-4-0414, Trinity | new | 1, 9 | |
| Post-norm, reordered | OLMo 2/3, EXAONE 4 | new | optional (OLMo) | |
| LayerNorm in an LLM, parallel attention + FFN | Cohere Command R/A, Phi-3.5-MoE | partial (LayerNorm in the DiT) | legacy tier | Command A+ |
| Embedding √d, layer scalar, output multiplier, muP, logit scale | Gemma (√d, layer scalar), Muse (output multiplier), Granite (muP), Cohere (logit scale) | new | 1, 9 | |
| Final logit softcap | Gemma 2 and 4, Muse | new | 1, 9 | |
| SwiGLU; clamped SwiGLU | most; DeepSeek V4 | have | | |
| Clamped (up+1) SwiGLU with α | gpt-oss | new | 2 | |
| GeGLU (GELU tanh) | Gemma | new for LLMs (GELU exists in `kernels/image`) | 1 | |
| Squared ReLU | Nemotron, AFM | new | 4 | |

### MoE

| Feature | Used by | jitLLM | Covered by | Only too big |
| --- | --- | --- | --- | --- |
| Softmax top-k + shared expert with sigmoid gate | Qwen3-Next line, Qwen3.8 | have | 12 | |
| Sigmoid, bias-corrected (`noaux_tc`), dense first layer | DeepSeek V3, MiMo V2, GLM, Kimi, Nemotron, Laguna, Step | partial (DeepSeek V4 selects with a bias over √softplus scores, `dsv4_graph.cc`) | 3, 4, 5, 13 | |
| Hash-routed layers | DeepSeek V4 | have | | |
| Softmax after top-k, router bias, expert biases | gpt-oss | new | 2 | |
| Dense MLP added beside the MoE, per-expert scale | Gemma 4 | new | 1 | |
| Top-1 sigmoid scaling the expert input, 16 large experts | Llama 4 Scout | new | 7 | Maverick's interleaved dense/MoE layers |
| LatentMoE (experts in a 1,024-wide latent) | Nemotron 3 Super, Ultra | new | 4 | |
| Grouped routing (`n_group` > 1) | DeepSeek V3, Ling 3 | new | optional (Ling) | DeepSeek V3 |
| Sigmoid without renormalization | Cohere North | new (minor) | optional | |
| No shared expert, top-8 of 128–256 | Qwen3 MoE, MiMo V2, MiniMax M2 | shape variant | 3 | |

### Tokenizers, templates, drafters

| Feature | Used by | jitLLM | Covered by |
| --- | --- | --- | --- |
| Byte-level BPE, `qwen2`/`qwen35`/`deepseek-v3` pre-tokenizers | Qwen, DeepSeek, MiMo | have (tokenizer.md) | 3, 10, 12 |
| ▁-space BPE over raw UTF-8 (SentencePiece style), 262K | Gemma 3/4 | new | 1 |
| Classic SentencePiece, 32K | Llama 2, Mistral 7B, Phi-3.5, ERNIE | new | legacy tier |
| o200k_harmony (201,088) | gpt-oss | new | 2 |
| Llama 3 tiktoken-style BPE (128,256) | Llama 3.x, Nemotron-Super-49B, R1-Distill-Llama | new | 8 |
| Llama 4 and Muse BPE, 202,048 (whether the two are identical not verified) | Llama 4, Muse | new | 7, 9 |
| Tekken (131,072, tiktoken-based) | Mistral Nemo onward | new | 6 |
| tiktoken 163,840 | Kimi | new | 5 |
| Other byte-level BPE pre-tokenizers | GLM (151–155K), MiniMax (200K), Laguna (digit split), Granite (100K) | new per family | 13 |
| Templates: harmony channels, ATEM, Gemma channels, `[INST]`/`[THINK]`, Kimi sections, GLM `[gMASK]` | per family | Gemma 3 and 4 (and their Unsloth, E2B/E4B and 2026-04 variants) render natively and in linear time; the rest through the interpreter (D-067 as amended; gpt-oss, Kimi, GLM-4.7, Mistral Small 4, Nemotron 3, Llama, MiMo match transformers in the [template corpus](experiments/chat-template-corpus/README.md); ATEM not checked); stop rules (harmony's `<\|call\|>` and channel ends) and output parsing remain per family; a native renderer only where one is needed | 1, 2, 5, 6, 9, 13 |
| Stored MTP layer | Qwen3.5/3.8, Ornith, GLM, Nemotron | have (Qwen3.8, one layer) | 4, 10, 13 |
| Multi-layer MTP | MiMo V2 (3), Step (3) | new | 3 |
| KV-sharing assistant drafter (Q-only, centroid head) | Gemma 4 | new | 1, 11 |
| EAGLE / EAGLE3 heads | gpt-oss (NVIDIA), Mistral, Llama 3.3, Cohere North | new | 2, 6 |
| DFlash-family block drafters | DSpark (DeepSeek V4, Nemotron, Ling); DFlash2 (Qwen3.8-27B); Muse, Laguna, MiMo, Ornith | have (DSpark, llama.cpp arch `dflash`); DFlash2's non-causal blocks with multi-layer target taps are new | 9, 10 |

No feature in wide use is found only in a checkpoint too big for one Spark,
except lightning attention (MiniMax M1), Maverick's interleaved dense and
MoE layers, and the large models' MSA, DSA and hyper-connection variants,
of which M4 takes DSA and mHC on two Sparks.

## Format coverage

jitLLM already runs these (from the repository; see
[artifact-format.md](artifact-format.md) and [model-support.md](model-support.md)):

- GGUF through GGML: Q8_0, Q4_K, Q5_K, Q6_K, IQ2_XS, IQ3_XXS and MXFP4, in
  DeepSeek V4's UD-Q2_K_XL (IQ2_XS/IQ3_XXS experts on 41 layers, MXFP4 on
  2); FP16 in the fixture.
- ModelOpt NVFP4 experts on CUTLASS 4.7.1's grouped GEMM and MXFP8 linears,
  `sm_121a` only (Qwen3.8); GGML's NVFP4 vector kernel for decode.
- BF16 (Qwen-Image).
- EXL3: dense only, `mcg` codebook, K ∈ {4,5,6,8}, on ExLlamaV3 `6b84a21b`'s
  kernels behind `src/kernels/exl3/`; GEMV instances for K = 4 only;
  bit-identical to upstream ([P3](experiments/backend-proof-p3/README.md)).
  No speed comparison with ExLlamaV3 exists yet (BP-F2 did not run; D-085
  judges engines end to end).

| Format | Granularity, bpw | Dequantization | Reference kernels | GB10 (`sm_121`) | Publishers | jitLLM | M3.5 carrier |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GGUF legacy Q4_0, Q4_1, Q5_0, Q8_0 | blocks of 32; 4.5, 5.0, 5.5, 8.5 | `d·(q−8)`, `d·q+m`, `d·q` | llama.cpp MMVQ, MMQ, dequant + cuBLAS ([ggml-common.h](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/ggml/src/ggml-common.h)) | generic CUDA | ggml-org, bartowski, unsloth; Google QAT Q4_0 | Q8_0 | 5, 13 (Q8_0); Gemma QAT Q4_0 optional |
| K-quants Q2_K–Q6_K | 256-weight superblocks, 16/32 sub-blocks; 2.625–6.5625 | `d·sc·q − dmin·m` | MMVQ, MMQ | generic | same | Q4_K–Q6_K | 1, 7, 9, 10 (UD mixes) |
| I-quants IQ1–IQ4 | 256-weight superblocks (IQ4_NL 32); 1.5625–4.5 | lattice/grid codebook × signs × scale | MMVQ, MMQ | generic | unsloth, bartowski | IQ2_XS, IQ3_XXS | MiniMax M2.7 UD-IQ3_XXS (optional) |
| Unsloth "UD" dynamic | per-tensor type mix | per type | llama.cpp | generic | unsloth | UD-Q2_K_XL | 1, 7 |
| MXFP4 (GGUF, and gpt-oss safetensors) | 32 × E2M1 + E8M0; 4.25 | `2^(e−127)·fp4(q)` | llama.cpp native FP4 MMQ ([#17906](https://github.com/ggml-org/llama.cpp/pull/17906)); vLLM Marlin / FlashInfer | llama.cpp native; vLLM Marlin wrong first token on sm_121 ([#37030](https://github.com/vllm-project/vllm/issues/37030)) | ggml-org, openai, unsloth `MXFP4_MOE` | GGUF kernel yes | 2 (GGUF) |
| NVFP4, ModelOpt (W4A4) | 16 × E2M1 + E4M3 scale + FP32 global; ≈4.5 | `g·s·fp4(q)` | CUTLASS SM120 block-scaled, FlashInfer, llama.cpp native NVFP4 | native (jitLLM's CUTLASS on `sm_121a`) | nvidia, Mia | yes | 4 |
| NVFP4, compressed-tensors (llm-compressor; W4A4 or W4A16) | as above, different packing | as above | vLLM CUTLASS / FlashInfer; W4A16 via Marlin | as above | RedHatAI, mistralai, poolside | no importer | 6 |
| MXFP8 (ModelOpt) | 32 × E4M3 + E8M0; 8.25 | `2^(e−127)·fp8(q)` | CUTLASS SM120, FlashInfer | native; vLLM falls back to Marlin ([#43906](https://github.com/vllm-project/vllm/issues/43906)) | Mia, nvidia | yes | |
| FP8, 128×128 blocks | per block FP32 or UE8M0 scale, dynamic per-1×128 activations | `s_blk·fp8(q)` | CUTLASS SM120 blockwise, FlashInfer, DeepGEMM (sm90/100 only) | vLLM v0.30.0 adds "SM12x blockwise FP8 … for GB10" ([release](https://github.com/vllm-project/vllm/releases/tag/v0.30.0)) | Qwen, DeepSeek, MiniMax, RedHatAI | no | 10 (`Qwen/Qwen3.8-27B-FP8`) |
| FP8, per tensor / per channel | static or per channel, dynamic per token | `s·fp8(q)` | CUTLASS scaled_mm sm120, cuBLASLt FP8 | works after the sm_12.1 guard fixes ([spark-vllm-docker #143](https://github.com/eugr/spark-vllm-docker/issues/143)) | mistralai, RedHatAI | no | optional (a small step after block FP8) |
| AWQ | INT4, group 128, zero point | `s·(q−z)` | AWQ-Marlin | Marlin runs on sm_121; Machete is Hopper-only | casperhansen, cyankiwi | no | 8 |
| GPTQ (v1/v2, compressed-tensors w4a16) | INT4/INT8, group 128 or per channel, optional act-order | `s·(q−z)` with permutation | Marlin (repacks at load) | Marlin runs | RedHatAI | no | 8's sibling (same kernel); optional |
| MLX affine 4-bit | groups of 32/64, BF16 scale and bias | `s·q + b` | TensorFold's Triton kernels on CUDA | measured on our Sparks as a baseline | mlx-community, Vontra | no | M9 (unchanged; [tensorfold-assessment](tensorfold-assessment.md)) |
| EXL3 | K = 1–8 per tensor (fractional targets = per-tensor mixes; fractional trellis since v1.5.1) | trellis decode → codebook → Hadamard-128 with `suh`/`svh` | ExLlamaV3 | jitLLM runs its kernels on `sm_121` | turboderp, Mia, community | dense, mcg, K 4/5/6/8 | 1, 3, 10, 11 ([below](#exl3-in-depth)) |
| EXL2, bitsandbytes, HQQ | — | — | — | — | — | — | excluded ([below](#considered-and-excluded)) |

Format sources: the formats study's reads of
[llama.cpp](https://github.com/ggml-org/llama.cpp), vLLM's issues and
release notes above, [FlashInfer #3170](https://github.com/flashinfer-ai/flashinfer/issues/3170)
(b12x excluded from auto-selection on SM121), the checkpoint configs linked
per family, and [Red Hat on Machete](https://developers.redhat.com/articles/2024/10/14/introducing-machete-mixed-input-gemm-kernel).

### EXL3 in depth

**Encoding.** EXL3 follows [QTIP](https://arxiv.org/abs/2406.11235)'s
bitshift trellis: 16×16 weight tiles, each stored as `16·K` 16-bit words
(`[k/16, n/16, 16·K]`, [artifact-format.md](artifact-format.md)). A
codebook maps each trellis state to a weight
([codebook.cuh](https://raw.githubusercontent.com/turboderp-org/exllamav3/master/exllamav3/exllamav3_ext/quant/codebook.cuh)):

- **3INST** (0): `x·89226354 + 64248484`, a `lop3` mask and XOR into two
  FP16 halves, summed.
- **MCG** (1): the same with multiplier `0xCBAC1FED` and no addend.
- **MUL1** (2): multiply by `0x83DCD12D`, a byte sum by `dp4a`, then one
  `hfma`. ExLlamaV3 v1.0.0 made it the default
  ([release](https://api.github.com/repos/turboderp-org/exllamav3/releases/tags/v1.0.0));
  [convert.md](https://raw.githubusercontent.com/turboderp-org/exllamav3/master/doc/convert.md)
  says the int8 GEMV and CPU expert offload need it.

**Rates.** K is an integer 1–8 per tensor; the head (`-hb`, default 6), MTP
(`-mb`), vision and n-gram table have their own. A fractional target such
as 4.5 bpw is a per-tensor mix (jitLLM's 4.5 bpw fixture: 42 tensors at
K4, 118 at K5, 8 at K6). `sc_optimize.py` allocates K per tensor by
marginal KL divergence per stored bit and writes a recipe
([optimize.md](https://raw.githubusercontent.com/turboderp-org/exllamav3/master/doc/optimize.md));
turboderp's "SC_" branches are such self-calibrated mixes. v1.5.1 adds a
fractional-trellis mode ([releases](https://github.com/turboderp-org/exllamav3/releases)).

**Rotations and execution.** A 128-wide Hadamard transform on input and
output, with `suh`/`svh` FP16 vectors folding signs and scales. Up to 144
rows, GEMM or GEMV decodes the trellis in-kernel; above that, weights are
reconstructed to FP16 in slices and multiplied by cuBLAS
([exl3-bringup.md](exl3-bringup.md)).

**What jitLLM has** (M2, D-080): the dense EXL3 linear on ExLlamaV3's
locked kernels (`src/kernels/exl3/`), bit-identical to upstream on 1,570
cases per fixture and arm, peak memory 0.82–0.87× upstream's
([P3](experiments/backend-proof-p3/README.md)); mcg only, GEMV at K = 4
only; no EXL3 MoE (M4's kernels item); ExLlamaV3 pinned at `6b84a21b`
(v1.5.1+1; [upstream/exllamav3.md](upstream/exllamav3.md)).

**Current upstream.** ExLlamaV3 v1.5.3 (2026-09-27,
[release](https://api.github.com/repos/turboderp-org/exllamav3/releases/tags/v1.5.3))
lists Gemma 4, gpt-oss, Kimi Linear, GLM 4.7 Flash, MiMo-V2.6-Flash,
MiniMax-M2, Mistral 4, Nemotron-H, Qwen 3.5/3.8 and DeepSeek V4 among about
55 architectures ([README](https://raw.githubusercontent.com/turboderp-org/exllamav3/master/README.md)).
Its wheels have no aarch64 build; aarch64 guards landed in
[#393](https://github.com/turboderp-org/exllamav3/pull/393) (2026-09-24).
Its `mma.sync` kernels need none of the sm_100 features the GB10 lacks
([dgx-spark-playbooks #22](https://github.com/NVIDIA/dgx-spark-playbooks/issues/22)).

**TensorFold's EXL3 path** (0.3.6.x, creator-reported, from
[tensorfold-assessment.md](tensorfold-assessment.md)): grouped
mixed-width routed experts in one launch per projection with pointer
tables, 1.5–3.6× ExLlamaV3's `exl3_moe_mixedk` (179.6 against 49.9 GB/s at
one row); a row-invariant linear whose warps load one coalesced run of
words with the next k-step in flight, 186–193 GB/s at 2 bits, level with
ExLlamaV3's 176–233; Flash Next 1.51–1.55× ExLlamaV3 on a mixed-width pack
but 5–6% behind on the uniform 3.05 bpw pack.

**Levers on the GB10** (analysis, not measurement):

1. **Decode is bytes per token.** Active parameters × bpw / 8 against
   ≈273 GB/s peak (≈202–229 GB/s effective in jitLLM's measurements): a
   dense 27B at 3.0 bpw reads ≈10 GB a token, so ≈20 tok/s is the ceiling.
   Lower bpw raises the ceiling only while the trellis decode keeps up.
2. **Decode compute per byte rises as bpw falls.** Each weight costs a
   fixed few instructions (extract, multiply, `lop3`+`hadd` or
   `dp4a`+`hfma`), so instructions per byte roughly double from 4 to 2
   bpw; TensorFold's 76→186 GB/s at 2 bits came from the load pattern.
   mul1's `dp4a` path may be cheaper than mcg's.
3. **MoE grouped dispatch** with per-expert pointer tables and mixed widths
   (TensorFold's highest-value item); it also suits M7's demand-paged
   experts. Small-active MoEs (Gemma 4 26B-A4B reads ≈2 GB a token at 3
   bpw) are launch-bound, so graphs matter as much as bandwidth.
4. **Prefill** reconstructs to FP16 then runs cuBLAS; reconstructing to FP8
   or MXFP8 would double tensor-core throughput but changes numerics.
5. **Row invariance** for exact speculation is optional under D-085.

**EXL3 checkpoints for the candidates** (branch SHA prefixes from each
repository's `/refs`):

| Model | Repository | Branches | Codebook, ExLlamaV3 version |
| --- | --- | --- | --- |
| Gemma 4 26B-A4B | [turboderp/gemma-4-26B-A4B-it-exl3](https://huggingface.co/turboderp/gemma-4-26B-A4B-it-exl3) | 2.10 `d20a0b40`, 2.54 `5b468457`, 3.10 `fcd233b0`, 3.54 `2e8c3a89`, 4.10 `71026023`, 5.10 `241bce8f`, 6.10 `75510f9f` | mcg, v0.0.28; 3.10 = 13.60 GB, 4.10 = 16.92 GB |
| Gemma 4 31B | [turboderp/gemma-4-31b-it-exl3](https://huggingface.co/turboderp/gemma-4-31b-it-exl3) | mcg 2.00–6.00 (3.00 `d27c062b`); mul1 2.00 `37c1d76c`, 2.50 `d72250d4`, 3.00 `5b465217`, 3.50 `4b3e382c`, 4.00 `13e18523`, 6.00 `a54a4fbf` | mcg v0.0.28, mul1 v0.0.43; 4.00 mul1 = 19.69 GB |
| Qwen3.8-27B | [turboderp/Qwen3.8-27B-exl3](https://huggingface.co/turboderp/Qwen3.8-27B-exl3) | uniform 2.00–6.00 (4.00 `113cf7ab`); self-calibrated SC_1.40–SC_6.00 (SC_3.00bpw_H4 `86b95530`, SC_2.20bpw_H3 `d6e046da`) | mul1, v1.4.2, MTP kept; 4.00 = 16.86 GB, SC_3.00_H4 = 13.45 GB |
| Qwen3.8 Flash Next | [turboderp/Qwen3.8-Flash-Next-exl3](https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3) | 2.05 `65c89531`, 3.05 `69e33439`, 4.05 `55a732e0` | mul1, v1.4.4; 3.05 = 84.98 GB (32.64 GB n-gram table) |
| MiMo-V2.6-Flash-RL | [benthecarman/MiMo-V2.6-Flash-RL-exl3@5ae87b83](https://huggingface.co/benthecarman/MiMo-V2.6-Flash-RL-exl3) | one build: 2.27 bpw (experts 2.0–2.5, attention 4.0, head 6.0), 85.28 GiB, MTP and DFlash included | ExLlamaV3 ≥ v1.5.2; codebook not verified |
| Muse Glimmer 30B | [turboderp/Muse-Glimmer-30B-exl3](https://huggingface.co/turboderp/Muse-Glimmer-30B-exl3) | SC_1.75–SC_5.00, uniform 2.00–6.00 | mul1, v1.4.2 |
| Mistral Small 4 | [turboderp/Mistral-Small-4-119B-2603-exl3](https://huggingface.co/turboderp/Mistral-Small-4-119B-2603-exl3) | 2.03 `2a11afcd`, 3.03 `0c2fc726`, 4.03 `1aabb209` | mul1, v1.4.0 |
| gpt-oss-120b | [turboderp/gpt-oss-120b-exl3](https://huggingface.co/turboderp/gpt-oss-120b-exl3) | 2.00–4.00 | mul1, v0.0.43 |
| Nemotron 3 Super | amanwalksdownthestreet (search [here](https://huggingface.co/api/models?search=Nemotron-3-Super-120B-A12B-exl3)) | "-opt" mixed 2.27–4.13; uniform 2.0–5.0 | not checked |
| Others | GLM-4.7-Flash (dr-housemd 4bpw-H6), MiniMax-M2.7 (NeuroSenko 2.0–8.0), Ornith (community 2.75–5.0), DeepSeek V4 Flash 0731 (turboderp 2.04–3.04), Mistral Medium 3.5 (turboderp 2.00–4.00); none for Kimi Linear | | not checked |

**Proposed EXL3 checkpoints:**

- **MoE:** Gemma 4 26B-A4B at 2.54, 3.10 and 4.10 (mcg, the codebook jitLLM
  decodes, so MoE dispatch is separated from codebook work; mixes inside
  each build not verified); then MiMo-V2.6-Flash-RL at 2.27 bpw, mixed
  widths per expert at 309B, the case TensorFold's grouped kernel targets.
- **Dense:** Gemma 4 31B mul1 at 2.50, 3.00 and 4.00 plus the mcg 3.00
  twin (a codebook A/B at one bitrate); Qwen3.8-27B SC_3.00bpw_H4 (a
  per-tensor mix of K 2–4, from a truncated read) and uniform 4.00.
- **Stretch:** Qwen3.8 Flash Next 3.05, the same model as M3's NVFP4 build.
- **Kernel scope this implies:** mul1 at K 2–6, GEMV at every rate (only
  K = 4 mcg today), EXL3 MoE, and the head at K = 6.
- **Reference to beat:** ExLlamaV3 v1.5.3, which these builds need (a
  re-pin from `6b84a21b`, re-checking P3's bit-exactness), with `6b84a21b`
  kept as the parity anchor; TensorFold's EXL3 path for information.

## Per-family picks

Each entry: revision, shape, the architecture facts that matter, formats,
reference and GB10 evidence, license (informational), why top-tier, and
what is new. Architecture comes from each `config.json` and the linked
modeling code.

### Google Gemma (owner-named)

No Gemma newer than Gemma 4 exists on Google's
[HF org](https://huggingface.co/api/models?author=google&sort=createdAt&direction=-1&limit=60);
after March came the assistants (April), QAT builds (April–June),
`gemma-4-12B-it` and DiffusionGemma (June). The base repos changed only
their chat templates in July.

- **MoE: `google/gemma-4-26B-A4B-it@4d7ae4984b7db7de8f8457170b3f1a419ee76d52`**
  (M0's pin, still HEAD), 25.2B total / 3.8B active
  ([card](https://huggingface.co/google/gemma-4-26B-A4B-it),
  [API](https://huggingface.co/api/models/google/gemma-4-26B-A4B-it)).
  - Architecture ([config](https://huggingface.co/google/gemma-4-26B-A4B-it/raw/main/config.json),
    [llama.cpp gemma4.cpp](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/src/models/gemma4.cpp)):
    30 layers, 1,024-token window on 25, global on 5 (5, 11, 17, 23, 29);
    local 16 Q / 8 KV at head dim 256, global 2 KV at head dim 512 with
    `attention_k_eq_v`; Q, K and V RMS norms; attention scale 1.0; local
    RoPE θ 1e4, global "proportional" θ 1e6 on a quarter of the dims;
    sandwich norms, embeddings × √d, a per-layer scalar; GeGLU; 128
    experts top-8 (704 wide), softmax over a normed, scaled router input
    with per-expert scale, plus a dense 2,112-wide MLP added to the MoE
    output (the card's "1 shared"); final softcap 30; 262,144 vocab, tied.
  - Tokenizer: BPE over raw UTF-8 with ▁ spaces, llama.cpp pre-type
    `gemma4` ([PR 21343](https://github.com/ggml-org/llama.cpp/pull/21343)).
    Template: `<|turn>role … <turn|>`, thinking by `<|think|>`, reasoning
    in `<|channel>thought`, tool calls `<|tool_call>call:name{k:v}` with
    `<|"|>` quoting ([template](https://huggingface.co/google/gemma-4-31B-it/raw/main/chat_template.jinja)).
  - Drafter: `google/gemma-4-26B-A4B-it-assistant@6e5aaaf4c42b98394530b8fda2e95cadd65c151c`,
    ≈0.42B, 4 Q-only layers that read the target's KV cache and share its
    embeddings, centroid LM head
    ([config](https://huggingface.co/google/gemma-4-26B-A4B-it-assistant/raw/main/config.json),
    [vLLM #41745](https://github.com/vllm-project/vllm/pull/41745),
    [Google](https://ai.google.dev/gemma/docs/mtp/overview)). Google and
    llama.cpp's MTP PR both report little or no batch-1 gain on the MoE.
  - Formats: GGUF `unsloth/gemma-4-26B-A4B-it-GGUF@c099eb48e663fd284577b04978a94ffccb261841`
    UD-Q4_K_M 16,947,541,728 B plus `mtp-gemma-4-26B-A4B-it.gguf`
    461,766,816 B; also Q8_0 26.86 GB and MXFP4_MOE 16.55 GB
    ([tree](https://huggingface.co/api/models/unsloth/gemma-4-26B-A4B-it-GGUF/tree/main));
    Google QAT Q4_0 GGUF `@d1c082be9cf3c8a514acf63b8761f4b41935842e`
    14.44 GB (needs the QAT assistant); `nvidia/Gemma-4-26B-A4B-NVFP4@a19cfe00be84568a6867111c9a68c9c44fdcffe6`
    14.39 GB; EXL3 above. **Proposed:** UD-Q4_K_M with the MTP GGUF on
    llama.cpp (the M0 reference), and the EXL3 builds on ExLlamaV3.
  - Reference: llama.cpp `gemma4` and `gemma4-assistant`
    ([llama-arch.cpp](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/src/llama-arch.cpp)),
    MTP in [PR 23398](https://github.com/ggml-org/llama.cpp/pull/23398)
    (2026-06-07). GB10: PP512 2,888 t/s, TG128 69.9 t/s at Q4_K_M
    ([shamily](https://github.com/shamily/gemma4-llama-dgx-spark)).
  - License: Apache-2.0. Top-tier: 12.48M downloads, the most-downloaded
    Gemma; MMLU-Pro 82.6, GPQA-D 82.3 (card). M7's daily driver.
- **Dense: `google/gemma-4-31B-it@842da3794eaa0b77d5f08bae87a17459d91ff475`**,
  31.3B ([API](https://huggingface.co/api/models/google/gemma-4-31B-it)):
  the same block without experts, 60 layers (10 global), hidden 5,376,
  local 32 Q / 16 KV, global 4 KV at 512, GeGLU 21,504
  ([config](https://huggingface.co/google/gemma-4-31B-it/raw/main/config.json)).
  - Drafter `google/gemma-4-31B-it-assistant@627c5ec1458b9086b841a91e0512fd31fd2fbbf1`
    (≈0.47B). The MTP PR reports more than 2× on the dense model; vLLM's
    PR measured 3.19× on H100 (creator-reported).
  - Formats: GGUF `unsloth/gemma-4-31B-it-GGUF@c1ac76e99d5513b141e8adde7288b85c3f9c32ec`
    UD-Q4_K_XL 18,822,970,304 B plus `mtp-gemma-4-31B-it.gguf` 514,687,104 B;
    Google QAT Q4_0 `@59dde24573e7e61570dba08b18a2e1fe246955ed`;
    `nvidia/Gemma-4-31B-IT-NVFP4@4135a98a9b728a548947683219633b25682223ac`
    20.87 GB; EXL3 above. **Proposed:** EXL3 (mul1 2.50/3.00/4.00, mcg
    3.00) on ExLlamaV3, and UD-Q4_K_XL with the assistant on llama.cpp for
    speculation (ExLlamaV3's support for the assistant is not verified).
  - GB10: Q4_K_M PP512 685 t/s, TG128 11.0 t/s (shamily, above).
  - License Apache-2.0; 9.73M downloads; MMLU-Pro 85.2, GPQA-D 84.3.

### OpenAI gpt-oss

No OpenAI open-weight LLM after gpt-oss (August 2025) exists on the
[`openai` org](https://huggingface.co/api/models?author=openai&sort=createdAt&direction=-1&limit=40);
a June 2026 "gpt-oss 8B/20B/120B" story is contradicted by it.

- **MoE: `openai/gpt-oss-120b@b5c939de8f754692c1647ca79fbf85e8c1e70f8a`**,
  117B / 5.1B ([card](https://huggingface.co/openai/gpt-oss-120b)). No
  dense model; `gpt-oss-20b@6cee5e81ee83917806bbde320786a8fb61efebee`
  (21B / 3.6B, 12.1 GB GGUF) is the same architecture, a cheap fixture.
  - Architecture ([config](https://huggingface.co/openai/gpt-oss-120b/raw/main/config.json),
    [modeling](https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/gpt_oss/modeling_gpt_oss.py)):
    36 layers alternating window-128 and full attention (1:1); GQA 64/8,
    head dim 64, hidden 2,880; QKV and output biases; a learned sink per
    head, concatenated to the logits before the softmax and dropped after;
    RoPE θ 150,000 with YaRN ×32 over 4,096; pre-RMSNorm; 128 experts
    top-4, router with bias, softmax over the selected k; experts with
    biases and clamped SwiGLU (`gate ≤ 7`, `−7 ≤ up ≤ 7`,
    `(up+1)·gate·σ(1.702·gate)`); MXFP4 on the experts only.
  - Tokenizer o200k_harmony, 201,088 tokens; harmony template with
    `analysis`/`commentary`/`final` channels, TypeScript-style tool
    namespaces and `<|call|>` ([template](https://huggingface.co/openai/gpt-oss-120b/raw/main/chat_template.jinja),
    [openai/harmony](https://github.com/openai/harmony)).
  - Drafter: none from OpenAI; `nvidia/gpt-oss-120b-Eagle3-v3@975baaffe27cc031ba21ae370602735d8e3df052`
    (one EAGLE3 layer reading target layers 24, 30, 36; acceptance 2.95 at
    draft length 7, [card](https://huggingface.co/nvidia/gpt-oss-120b-Eagle3-v3));
    a Q8_0 GGUF of an EAGLE3 head ships in the ggml-org repo.
  - Formats: **proposed** `ggml-org/gpt-oss-120b-GGUF@238abdd290bb874b90a5da1b4549881b7d05c091`
    MXFP4, 63,387,346,208 B, plus its EAGLE3 Q8_0 849,103,296 B
    ([API](https://huggingface.co/api/models/ggml-org/gpt-oss-120b-GGUF?blobs=true)).
    Native MXFP4 safetensors (65.2 GB) and turboderp EXL3 2.0–4.0 also
    exist; no official NVFP4.
  - Reference: llama.cpp `gpt-oss`, native Blackwell MXFP4 MMQ
    ([#17906](https://github.com/ggml-org/llama.cpp/pull/17906); it
    quantizes activations to MXFP4, which matters for comparisons), EAGLE3
    merged ([#18039](https://github.com/ggml-org/llama.cpp/pull/18039)).
    GB10: pp2048 1,024 t/s, tg32 35 t/s
    ([discussion #16578](https://github.com/ggml-org/llama.cpp/discussions/16578)).
    vLLM on `sm_121` still has an open MXFP4 correctness issue
    ([#37030](https://github.com/vllm-project/vllm/issues/37030)).
  - License Apache-2.0. Top-tier: 20b and 120b are #12 and #20 among all
    text-generation downloads on HF (6.75M and 4.46M); AA index 12.

### Xiaomi MiMo (owner-named)

Every MiMo MoE is ≈309B or larger
([org](https://huggingface.co/api/models?author=XiaomiMiMo&sort=createdAt&direction=-1&limit=40)),
so one Spark means a ≈2-bit build.

- **MoE: `XiaomiMiMo/MiMo-V2.6-Flash-RL@5711b268169967567844e1e560e8a3966da959b1`**
  (the two-Spark reference's pin), 309B / 15B
  ([card](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-RL)).
  - Architecture ([config](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-RL/raw/main/config.json)):
    48 layers, 9 global and 39 window-128 (layer 0, then five local per
    global); 64 Q heads, KV 4 on global and 8 on local; qk head dim 192, v
    128, `attention_value_scale` 0.707; a learned sink bias on local
    layers only; partial RoPE 0.334, θ 1e7 global and 1e4 local; SwiGLU;
    256 experts top-8, sigmoid `noaux_tc`, no shared expert, layer 0 dense;
    Qwen2 byte-level BPE 152,576; ChatML with `<think>` and Qwen3-coder XML
    tool calls. MTP: 3 layers in the config, "5 SWA layers" on the card
    (unresolved).
  - Formats: **proposed** EXL3 `benthecarman/MiMo-V2.6-Flash-RL-exl3@5ae87b830eb950c705967aea6130601a8e1f74a5`,
    2.27 bpw, 85.28 GiB, with MTP and a DFlash drafter; the quantizer
    reports 31.5 tok/s plain, 40.7 with MTP, 49.5 with DFlash, and 262,144
    context with 15 GiB free on one Spark
    ([card](https://huggingface.co/benthecarman/MiMo-V2.6-Flash-RL-exl3)).
    GGUF alternatives: `AesSedai/MiMo-V2.6-Flash-RL-GGUF@05c13439c18ba7183cb6afe294924c75ba7aa7b4`
    BPW2.5 90.14 GB (PPL +11.5% against MXFP4); ggml-org's MTP and DFlash
    sidecars, but llama.cpp's sidecar MTP fails for `mimo2`
    ([#29345](https://github.com/ggml-org/llama.cpp/issues/29345)).
  - Reference: ExLlamaV3 `MiMoV2ForCausalLM` (README); needs ≥ v1.5.2, past
    jitLLM's pin. A newer drop-in, `MiMo-V2.6-Flash-MOPD@2479e2d0029eca9a34cc7e7f55a121925f81908e`
    (2026-09-27, fixes repeated tool calls), has an EXL3 build a day old.
  - License MIT. Top-tier: OpenRouter's #5 open model (7.86T tokens a
    week); MiMo-V2.6-Pro leads AA's open-weights index.
- **Dense: none proposed.** `MiMo-V2.6-Distill-Qwen-9B@2367e865d009c13ac81713a2878291d33ab28177`
  is a Qwen3.5-9B fine-tune (`qwen3_5`), and `MiMo-7B-RL@6299b5a2c45daf0c429285c92b8e61a5bd011c0d`
  (2025) is Qwen2-like with one MTP layer; neither adds surface beyond
  the Qwen picks.

### NVIDIA Nemotron (in the plan)

Current: Nemotron 3 Nano 4B and 30B-A3B, 3 Super 120B-A12B, 3 Ultra
550B-A55B, 3.5 Lightning 30B-A3B
([org](https://huggingface.co/api/models?author=nvidia&sort=createdAt&direction=-1&limit=60)).

- **MoE: `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4@ff433f5493e25d631c9f12b5d55c674229923d02`**,
  ≈80.3 GB ([API](https://huggingface.co/api/models/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4)).
  - Architecture ([config](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16/raw/main/config.json)):
    `nemotron_h`, 88 layers (48 Mamba2, 41 MoE, 9 attention); Mamba2 128
    heads × 64, state 128; attention GQA 32/2 at 128 with no RoPE
    (transformers' `NemotronHAttention` applies none,
    [modeling](https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/nemotron_h/modeling_nemotron_h.py));
    512 experts top-22 plus one shared, LatentMoE (latent 1,024); squared
    ReLU; 131,072 vocab; one MTP layer (attention + MoE); 262,144 context.
  - Reference: vLLM with Marlin and MTP, per NVIDIA's single-Spark guide
    ([guide](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/usage-cookbook/Nemotron-3-Super/SparkDeploymentGuide/README.md));
    llama.cpp `nemotron_h_moe`, latent MTP in
    [#29018](https://github.com/ggml-org/llama.cpp/pull/29018).
  - License: NVIDIA Nemotron Open Model License. Top-tier: 692,762
    downloads; AA medium index 13; OpenRouter's free tier.
- **Cheaper first step (optional): `NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4@bee7596271d1495f6992ae224aefde4410e816b8`**
  (2026-08, 21.6 GB, W4A16): the same family without LatentMoE (128
  experts top-6, 52 layers), with a DSpark drafter
  `@8a0177116d138011e63103110f136ec0ca09ebbf`; AA #1 for output speed;
  license OpenMDW-1.1.
- **Dense: none proposed.** The only native dense Nemotron 3 is Nano 4B
  (Mamba2 + squared-ReLU MLP), covered by Super's layers; the
  Llama-Nemotron line (DeciLM NAS) is a Llama 3 derivative.

### Moonshot Kimi (owner-added)

No Flash or small K2/K3 exists
([org](https://huggingface.co/api/models?author=moonshotai&sort=createdAt&direction=-1&limit=40)).
K3 is 2.8T / 104B, K2.6 1.03T (smallest listed GGUF 340 GB), K2.7-Code 1T:
too big even for two Sparks at a usable quant, so not M4 candidates.

- **MoE: `moonshotai/Kimi-Linear-48B-A3B-Instruct@e1df551a447157d4658b573f9a695d57658590e9`**,
  48B / 3B, MIT, 191,419 downloads
  ([card](https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct)).
  - Architecture ([config](https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct/raw/main/config.json)):
    27 layers, 20 KDA (32 heads × 128, short conv 4) and 7 MLA (kv_lora
    512, no q-LoRA, qk 128 nope + 64 rope, v 128, NoPE in use); 256
    experts top-8 plus one shared, sigmoid, layer 0 dense; tiktoken
    163,840; `<|im_user|>`/`<|im_middle|>` roles and tool-call sections
    ([template](https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct/raw/main/chat_template.jinja)).
    No MTP.
  - Formats: **proposed** `bartowski/moonshotai_Kimi-Linear-48B-A3B-Instruct-GGUF@228dbe476e5a02091624a19068f4c962caa8a1c5`
    Q8_0 52.25 GB. No EXL3 build found.
  - Reference: llama.cpp `kimi-linear`
    ([PR 18755](https://github.com/ggml-org/llama.cpp/pull/18755), KDA from
    generic GGML ops); vLLM `KimiLinearForCausalLM`. GB10 speed not
    verified.
  - Why: the only one-Spark Kimi, and the one-Spark route to KDA, which
    Kimi K3, Ling 3 and M4's GLM-5.3 Flash use.
- **Dense: none.** Kimi-Dev-72B is a Qwen2.5-72B fine-tune.

### Mistral AI (owner-added)

Current ([org](https://huggingface.co/api/models?author=mistralai&sort=createdAt&direction=-1&limit=40)):
Mistral Medium 3.5 128B dense (2026-03), Mistral Small 4 119B MoE
(2026-01), Leanstral 1.5 (Lean prover), Mistral Large 3 675B, Devstral 2,
Ministral 3.

- **MoE: `mistralai/Mistral-Small-4-119B-2603@a11f36bebf709121056b1dbcc943d1c6afbe494d`**,
  119B / 6.5B, Apache-2.0 ([card](https://huggingface.co/mistralai/Mistral-Small-4-119B-2603)).
  - Architecture ([config](https://huggingface.co/mistralai/Mistral-Small-4-119B-2603/raw/main/config.json)):
    `mistral4` text, 36 layers; MLA (q_lora 1,024, kv_lora 256, qk 64 nope
    + 64 rope, v 128, 32 heads; interleaved RoPE); YaRN ×128 over 8,192
    with `llama_4_scaling_beta` 0.1; 128 experts top-4 plus one shared,
    every layer MoE; SwiGLU; Tekken 131,072; `[SYSTEM_PROMPT]`, `[INST]`,
    `[THINK]`, `[MODEL_SETTINGS]`, `[TOOL_CALLS]name[ARGS]{json}`
    ([template](https://huggingface.co/mistralai/Mistral-Small-4-119B-2603/raw/main/chat_template.jinja)).
    Router function not stated in the config.
  - Drafter: `mistralai/Mistral-Small-4-119B-2603-eagle@9a8ea22dca0161ff7af7879f4a0000f65314797d`,
    two MLA layers, 0.39 GB.
  - Formats: **proposed** `mistralai/Mistral-Small-4-119B-2603-NVFP4@45331841b631f4e281df8e959ea3cc9beb84298a`,
    74.76 GB (llm-compressor; only Mistral-native `consolidated-*` files
    and `params.json`, so the importer reads Mistral's layout). Also
    `unsloth/Mistral-Small-4-119B-2603-GGUF@bd93c721735aa32c035c0f19e738cb3371fd56ff`
    UD-Q4_K_XL 74.16 GB and MXFP4_MOE 71.81 GB; EXL3 2.03–4.03.
  - Reference: vLLM `TRITON_MLA`; a GB10 report ran 33.2 → 17.7 tok/s
    decode from 2K to 60K after early builds rejected MLA head size 320 on
    `sm_121` ([forum](https://forums.developer.nvidia.com/t/running-mistral-small-4-119b-nvfp4-on-nvidia-dgx-spark-gb10/363863)).
    llama.cpp `mistral4`.
  - Top-tier: 51.6K downloads; AA #12 of 65 medium models.
- **Dense (optional): `mistralai/Mistral-Medium-3.5-128B@22b2b868a15677cfa6061277ed2f653d1349a9ab`**
  (Modified MIT): 88 layers, GQA 96/8, full attention, YaRN ×64, Tekken.
  `nvidia/Mistral-Medium-3.5-128B-NVFP4@b8c66d2098edd8c9c26bde2b2ff41b5967e111ae`
  is 83.84 GB plus a 3.07 GB EAGLE drafter: at the budget's edge, and
  6.9–9.1 tok/s with EAGLE on a Spark
  ([forum](https://forums.developer.nvidia.com/t/performance-report-mistral-medium-3-5-128b-nvfp4-eagle/368917)).
  Llama 3.3 70B covers dense full attention at scale more cheaply.

### Meta: Llama and Muse (owner named Llama)

`meta-llama` has published nothing since Llama 4 and Llama Guard 4 (April
2025, [org](https://huggingface.co/api/models?author=meta-llama&sort=createdAt&direction=-1&limit=40));
claims of a "Llama 4.5" or a 2026 405B Llama 4 are unverified blog spam.
Meta's newest open model is Muse Glimmer 30B, from Meta Superintelligence
Labs, in a new `meta-models` org
([post](https://research.meta.ai/blog/introducing-muse-glimmer-open-agentic-model)).

- **MoE: `meta-llama/Llama-4-Scout-17B-16E-Instruct@92f3b1597a195b523d8d9e5700e57e4fbb8f20d3`**,
  109B / 17B, gated, Llama 4 Community License
  ([blog](https://ai.meta.com/blog/llama-4-multimodal-intelligence/)).
  - Architecture (ungated [mirror config](https://huggingface.co/unsloth/Llama-4-Scout-17B-16E-Instruct/raw/main/config.json),
    [modeling](https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/llama4/modeling_llama4.py)):
    48 layers, GQA 40/8 at 128; every 4th layer NoPE, the others RoPE with
    chunked attention (8,192) and L2 QK-norm; NoPE layers scale queries by
    `log1p(floor((pos+1)/8192))·0.1 + 1`; RoPE θ 5e5, llama3 scaling ×16;
    16 experts top-1 plus a shared expert, the sigmoid of the top logit
    scaling the expert input; vocab 202,048. Template not read (gated).
  - Formats: **proposed** `unsloth/Llama-4-Scout-17B-16E-Instruct-GGUF@72a6853f56a66dc13a3a4b6bdc9cf7ee4c364b47`
    UD-Q4_K_XL 62.00 GB; `nvidia/Llama-4-Scout-17B-16E-Instruct-NVFP4@9417590c9bc99b359f3ac66d1346ecd54249888e`
    55.79 GB. Reference llama.cpp `llama4`; no GB10 report found.
  - Why: the only Llama MoE that fits, and the "few large experts, low
    top-k" shape M0 listed as a gap. It is 18 months old (182K downloads).
- **Dense (Llama): `meta-llama/Llama-3.3-70B-Instruct@6f6073b423013f6a7d4d9f39144961bfbfbc386b`**,
  license `llama3.3`, 810K downloads.
  - Architecture ([config](https://huggingface.co/unsloth/Llama-3.3-70B-Instruct/raw/main/config.json)):
    80 layers, GQA 64/8, RoPE θ 5e5 with llama3 scaling ×8, SwiGLU,
    RMSNorm, 128,256 vocab (the Llama 3 tokenizer, shared by the Llama 3.x
    line: 3.2-1B-Instruct alone has 7.48M downloads); full attention in
    every layer.
  - Formats: **proposed** `casperhansen/llama-3.3-70b-instruct-awq@64d255621f40b42adaf6d1f32a47e1d4534c0f14`
    (AWQ 4-bit, group 128, zero point, GEMM layout;
    [config](https://huggingface.co/casperhansen/llama-3.3-70b-instruct-awq/raw/main/config.json);
    607,024 downloads), on vLLM's AWQ-Marlin. Others: `nvidia/Llama-3.3-70B-Instruct-NVFP4`
    (204,609 downloads), `RedHatAI/...-FP8-dynamic`, GGUF Q4_K_M 42.52 GB;
    EAGLE3 heads from nvidia and RedHatAI
    ([search](https://huggingface.co/api/models?search=Llama-3.3-70B-Instruct&sort=downloads&direction=-1&limit=40)).
  - Why: the dense full-attention case whose per-token KV read sets the
    long-context floor, and the carrier for AWQ/Marlin.
- **Dense (Meta, current): `meta-models/Muse-Glimmer-30B@a4e59da52a7bc87ae7251dd5545c0dd437c44b68`**,
  29.8B, Apache-2.0, 275,549 downloads (1.2M for its GGUF).
  - Architecture ([config](https://huggingface.co/meta-models/Muse-Glimmer-30B/raw/main/config.json),
    [modeling](https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/muse_glimmer/modeling_muse_glimmer.py)):
    52 layers, window 2,048 on three of every four, global NoPE layers
    (`layer_rope_theta` 0); GQA 32/2 at 128; QK-norm with a 3.87 query
    gain; sandwich norms; SwiGLU; logits × 0.196 then softcap 20; vocab
    202,048; "ATEM" template (`<|start|>role<|message|>…<|eot|>`,
    `assistant to=`, reasoning strength, XML `atem:` tool calls,
    [template](https://huggingface.co/meta-models/Muse-Glimmer-30B/raw/main/chat_template.jinja)).
  - Drafter: `meta-models/Muse-Glimmer-30B-assistant@e8192f3a8f617f74be2ce220360c89ef4789f39f`,
    a DFlash block drafter (5 sliding layers, 16-token blocks).
  - Formats: **proposed** Meta's own `meta-models/Muse-Glimmer-30B-GGUF@70bf1b61ac09f91b24d39038091b41c582bc5d7a`
    Q4_K_M 16.76 GB plus the DFlash GGUF 1.63 GB; NVFP4 (nvidia, 19.81 GB),
    EXL3 (turboderp), BF16 59.6 GB.
  - Reference: llama.cpp `muse-glimmer`; vLLM ≥ 0.28 with a GB10 recipe
    ([recipe](https://recipes.vllm.ai/meta-models/Muse-Glimmer-30B)).
    GB10: 11.9 tok/s decode, 29.6 with DFlash, which its author flags as
    not lossless ([sxuff](https://github.com/sxuff/muse-glimmer-dgx-spark)).
    Whether upstream llama.cpp's DFlash covers it is not verified.
  - Top-tier: AA small-model index 17; SWE-Bench Verified 76.0 (card).

### Qwen, beyond M3 (Qwen3.8-27B named in the plan)

Qwen3.8 has three language checkpoints: 27B dense, Flash Next and
2.4T-A95B ([org](https://huggingface.co/api/models?author=Qwen&search=Qwen3.8&sort=createdAt&direction=-1&limit=40)).
The MoE slot is M3's Flash Next; a smaller Qwen MoE (Qwen3.6-35B-A3B) has
Ornith's architecture.

- **Dense: `Qwen/Qwen3.8-27B@1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`**,
  27.8B with vision, Apache-2.0, 7.02M downloads
  ([API](https://huggingface.co/api/models/Qwen/Qwen3.8-27B)).
  - Architecture ([config](https://huggingface.co/Qwen/Qwen3.8-27B/raw/main/config.json)):
    `qwen3_5`, 64 layers as 16 × (3 Gated DeltaNet + 1 gated full
    attention); DeltaNet 16 QK / 48 V heads at 128; attention GQA 24/4 at
    256, output gate; dense SwiGLU 17,408; partial RoPE 0.25, θ 1e7,
    M-RoPE; 248,320 vocab (the `qwen35` BPE jitLLM has); one MTP layer;
    ChatML with `<think>`, `reasoning_effort`, XML tool calls.
  - Drafter: DFlash2 `z-lab/Qwen3.8-27B-DFlash2@50307d4c4cde6860d4eee73e2547cd786fe8e8a4`
    (mirror of `incoai/…@015e795645c74b1a0eeef3b570031fb62e769bc5`, 1.92B,
    Apache-2.0): 5 non-causal window-2,048 layers reading target layers
    5/19/33/47/61, blocks of 8
    ([config](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2/raw/main/config.json)).
  - Formats: EXL3 (above); `ggml-org/Qwen3.8-27B-GGUF@71bc7b627595dc8a91039addd9c791ae548d6747`
    Q4_K_M 18.97 GB with `mtp-` and `dflash-` GGUFs;
    `Qwen/Qwen3.8-27B-FP8@017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`
    (FP8 E4M3, 128×128 blocks, dynamic activations, MTP kept,
    [config](https://huggingface.co/Qwen/Qwen3.8-27B-FP8/raw/main/config.json);
    5.17M downloads); `nvidia/Qwen3.8-27B-NVFP4@482ca0f3832238542f8f5295dde86b5f22711d80`
    (≈21.9 GB, MTP dropped). **Proposed:** EXL3 SC_3.00bpw_H4 and 4.00 on
    ExLlamaV3; the GGUF with MTP and DFlash2 on llama.cpp (MTP
    [#22673](https://github.com/ggml-org/llama.cpp/pull/22673), DFlash2
    [#27342](https://github.com/ggml-org/llama.cpp/pull/27342)); FP8 on
    vLLM. One architecture close to jitLLM's carries three formats, so
    format work is separated from architecture work.
  - GB10 (vLLM/SGLang): NVFP4 + MTP 18.5 tok/s, NVFP4 + DFlash2 47.9
    ([forum study](https://forums.developer.nvidia.com/t/comprehensive-qwen3-8-27b-study-on-dgx-sparks-quantization-speculative-decoding-and-tp-dp-scaling/381102)).
  - Top-tier: AA's #1 small open model (34); GPQA-D 89.2 (card).

### Ornith (M7's daily driver)

Ornith 1.0 and 1.5 are Qwen3.5-family fine-tunes (`qwen3_5`,
`qwen3_5_moe`); 1.5 came 2026-08-18, DFlash drafters 2026-09-20
([org](https://huggingface.co/api/models?author=ornith-ai&sort=createdAt&direction=-1&limit=30)).

- **MoE: `ornith-ai/Ornith-1.5-35B-A3B@10fbf86fed7ecee4a061f8b499a618f46001cac1`**
  (M0's base revision), MIT: 40 layers as 10 × (3 Gated DeltaNet + 1 gated
  attention), attention 16/2 at 256, 256 experts top-8 plus one shared
  (512 wide), one MTP layer, 262,144 context
  ([config](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B/raw/main/config.json)).
  **Proposed** `ornith-ai/Ornith-1.5-35B-A3B-GGUF@12393612fd4f730ff5aadc23e9b8f9648aa49ceb`
  Q4_K_M 21.71 GB (M0's reference file family), llama.cpp `qwen35moe`;
  NVFP4 `@94e431d9cc47fa1986a7a1a4e9a80f7f118b03aa` also exists. 3.68M
  GGUF downloads; SWE-bench Verified 79 (card).
- **Dense: none proposed.** `Ornith-1.5-9B` is a smaller Qwen3.8-27B.

### Z.ai GLM (M4 keeps GLM-5.3 Flash)

GLM-5.3-Flash (320B / 18B) fits one Spark only at 1-bit (UD-IQ1_S 93.1 GB,
[unsloth](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF)), so it stays
in M4.

- **MoE: `zai-org/GLM-4.7-Flash@7dd20894a642a0aa287e9827cb1a1f7f91386b67`**,
  31.2B / 3B, MIT, 1.85M downloads
  ([card](https://huggingface.co/zai-org/GLM-4.7-Flash)): 47 layers, MLA
  (20 heads, q_lora 768, kv_lora 512, qk 192 + 64, v 256); 64 experts
  top-4 plus one shared, `noaux_tc`, layer 0 dense; 154,880 vocab; one MTP
  layer; `[gMASK]<sop>`, `<|observation|>`, `<arg_key>`/`<arg_value>` tool
  calls ([config](https://huggingface.co/zai-org/GLM-4.7-Flash/raw/main/config.json)).
  **Proposed** `unsloth/GLM-4.7-Flash-GGUF@0d32489ecb9db6d2a4fc93bd27ef01519f95474d`
  Q8_0 31.84 GB on llama.cpp (arch `deepseek2`); GB10 52.5 tok/s decode
  ([forum](https://forums.developer.nvidia.com/t/llama-cpp-glm-4-7-flash-benchmark/358749)).
  llama.cpp MTP for it is not verified.
- **Dense: none proposed.** `GLM-4-32B-0414` (6K downloads) adds only
  sandwich norms and interleaved partial RoPE, which 1 and 9 cover.

### Optional checkpoints

Each fits one Spark with a same-format reference; each adds usage or a
feature the core set covers elsewhere.

| Checkpoint | Format, reference | Adds | Why optional |
| --- | --- | --- | --- |
| `MiniMaxAI/MiniMax-M2.7@d494266a4affc0d2995ba1fa35c8481cbd84294b` (229B MoE, full attention in all 62 layers, whole-projection QK-norm, 256 top-8 sigmoid, 200K BPE) | `unsloth/MiniMax-M2.7-GGUF@d2a05ccf69491b03db0cc40b335aec14bdaf7198` UD-IQ3_XXS 80.10 GB, llama.cpp `minimax-m2` | I-quants at scale, the MiniMax template, DFlash and EAGLE3 drafters | fits only at ≈3 bits; non-commercial license (informational); 1.19M downloads |
| `poolside/Laguna-S-2.1@0f573140834b11cfac0c2af97a101a7a69a13e22` (118B / 8B; 3 window-512 : 1 global, per-type head counts, softplus head gate, digit-split BPE) | `unsloth/Laguna-S-2.1-GGUF@750f92f90cf54159c4d7a610cb7b3e74498e75c6` UD-Q4_K_XL 73.4 GB, llama.cpp `laguna` | OpenRouter's #3 free model (1.19T tokens a week) | features mostly covered by 1 and 3; its DFlash needs Poolside's llama.cpp fork |
| `stepfun-ai/Step-3.7-Flash@5f6244077ac62e04eec3f320501ff8c2b293373a` (196B / 11B; 3 window-512 : 1 global, zero-centred norm, 3 MTP layers, 288 top-8 + shared) | unsloth UD-Q3_K_XL 89.4 GB + MTP 3.7 GB, llama.cpp `step35` | Q3_K, multi-layer MTP (also MiMo) | ≈93 GB total at 3 bits |
| `inclusionAI/Ling-3.0-flash@ef06d91fe382109ae82647da88ff99b0f11745b0` (124B / 5.1B; KDA + gated MLA 5:1, grouped sigmoid routing, 512 experts) | bartowski Q4_K_M 77.80 GB, llama.cpp `bailingmoe3` | grouped routing; AA #1 medium open model | KDA and MLA covered by 5 |
| Nemotron 3.5 Lightning 30B-A3B (above) | NVFP4 W4A16, vLLM | a fast Nemotron with DSpark | covered by 4 except its drafter |
| `CohereLabs/North-Mini-Code-1.0@d11e61a842617a22dc328552fa5bb86231ee4f37` (30B / 3B; 3:1 window-4,096 with NoPE globals, sigmoid without renormalization) | unsloth UD-Q4_K_XL 19.25 GB, llama.cpp `cohere2moe` | Cohere's template, an EAGLE head | low usage (9.8K; 302K for the GGUF); features covered |
| Mistral Medium 3.5 128B (above) | NVFP4 + EAGLE, vLLM | dense NVFP4 GEMM | 87 GB with its drafter, slow |
| Legacy tier: Gemma 2 9B, Phi-3.5-mini, Mistral-7B-Instruct-v0.3, Command R7B, Llama 3.2 1B | GGUF, llama.cpp | attention softcap and (1+w) norms; LongRoPE and classic SentencePiece; SentencePiece v3 control tokens; LayerNorm, parallel block and interleaved RoPE; tied embeddings | older generations still downloaded (Gemma 2 9B 1.1M, Phi-3.5-mini 355K, Mistral 7B v0.3 2.18M); superseded by [Legacy-tier features](#legacy-tier-features), which pins fixtures |

## Legacy-tier features

The owner's question 7: which features of older generations still in
use are not subsets of what the 13 checkpoints and M3's models already
cover, and is each worth implementing? A feature counts as covered only
if one of those models runs the same math. Downloads are the Hugging
Face API's 30-day counts on 2026-09-29 (CI test repos left out). Engine
status is llama.cpp and vLLM master on that date: llama.cpp's
[`src/models`](https://github.com/ggml-org/llama.cpp/tree/master/src/models),
and vLLM's
[registry](https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/model_executor/models/registry.py),
whose `_PREVIOUSLY_SUPPORTED_MODELS` list names the removals, plus its
[quantization methods](https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/layers/quantization).
Configs are linked from each vendor's repository and were read the same
day.

Cost is graded in three steps:

- **Small:** a variant of an existing op, or a GGML kernel that
  jitLLM's pin has but has not registered.
- **Medium:** a new kernel.
- **Large:** a new state class.

The vendor column says **abandoned** when the vendor's successors
dropped the feature, and **not updated** when it is still the vendor's
current design.

**Already covered, by composition.** None of these needs new math:

- **Mixtral 8x7B and 8x22B.** Mixtral takes a softmax over all experts,
  keeps the top 2 and renormalises
  ([modeling](https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/mixtral/modeling_mixtral.py)).
  That equals a softmax over the selected logits, which is gpt-oss's
  routing (2) without the bias. Its lack of a shared expert matches 3.
- **Mistral 7B.** v0.1's window of 4,096 on every layer is the ring
  cache (1, 2) with no global layers. v0.2 and v0.3 have no window
  ([config](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3/raw/main/config.json)).
- **Llama.** Llama 2's full multi-head attention is covered, as are
  Llama 3.x's llama3 scaling (7, 8) and 3.2's tied embeddings (1). Yi
  is a Llama.
- **Qwen dense.** Qwen 1.5, 2 and 2.5 are the Qwen2 fixture.
- **DeepSeek V2.** Its MLA, with or without q-LoRA, is covered by 5, 6
  and 13, and its YaRN `mscale` by DeepSeek V4.
- **GPT-J and NeoX partial rotary.** Qwen3.8 covers the NeoX layout at
  0.25, and DeepSeek V4 the pairs.
- **Gemma 1 and 2.** `query_pre_attn_scalar` is an attention scale.
  (1+w) norms are folded at conversion. The 1:1 window and the final
  softcap are covered by 1 and 9, and GeGLU by 1.
- **Cohere.** The logit scale is 9's multiplier; the GPT-J pairs,
  Command R+'s QK norms (built from item 5 below) and R7B's NoPE global
  layers (9) are covered too.
- **Non-gated MLPs** (Phi-2, Falcon, StarCoder2, Bloom, Nemotron-4,
  Jais 2) are Nemotron 3's squared-ReLU MLP (4), with GELU from 1 where
  used.
- **Import-time items.** DBRX's renormalised top-4 is 2's routing;
  Phi-3's fused QKV and gate-up need an importer split; BitNet's
  sub-norms are ordinary RMSNorms in new places.

**Not covered:**

| Feature (math) | Used by (vendor, first → last use) | Downloads a month | llama.cpp / vLLM | Vendor | Cost | Recommendation |
| --- | --- | --- | --- | --- | --- | --- |
| 1. Classic SentencePiece: score-ordered merges on ▁-normalised text, `<0xNN>` byte fallback, dummy prefix | Meta Llama 2 (2023-07); Mistral 7B and Mixtral (2023-09 → 2024-05); Google Gemma 1–3 (2024-02 → 2025-03); Microsoft Phi-3/3.5 (2024); Yi; InternLM2 | Gemma 3 1B 3.24M, 4B 1.45M; Mistral 7B v0.3 2.18M, v0.2 1.65M; Gemma 2 9B 1.13M; Llama 2 7B chat 484K | both | abandoned: Llama 3 moved to tiktoken-style BPE, Mistral to Tekken, Gemma 4 to ▁-BPE in `tokenizer.json`, Phi-4 to 100K/200K BPE | small–medium: a vocabulary mode beside 1's ▁-BPE | **Implement.** The largest legacy use by far; with it, Gemma 1 and Gemma 3 1B, Mistral 7B, Mixtral, Llama 2 and Yi have no other gap, and Gemma 2, Gemma 3 and the dense Phi-3.x each need only one more row |
| 2. Linear RoPE scaling (positions ÷ 8 on global layers) | Gemma 3 4B, 12B and 27B (2025-03; 1B has none) ([config](https://huggingface.co/unsloth/gemma-3-4b-it/raw/main/config.json)) | Gemma 3 4B 1.45M, 12B 530K, 27B 415K; derivatives MedGemma 4B 1.04M, 27B AWQ 1.15M, 27B GPTQ 710K | both | abandoned: Gemma 4's globals use proportional RoPE | small: GGML rope's `freq_scale` | **Implement:** with row 1 it completes Gemma 3 |
| 3. Attention logit softcap (`c·tanh(s/c)` on QKᵀ, c = 50) | Gemma 2 (2024-06) ([config](https://huggingface.co/unsloth/gemma-2-9b-it/raw/main/config.json)); xAI Grok-2 (too big) | Gemma 2 9B 1.13M, 2B 617K | llama.cpp FA `logit_softcap`; vLLM FA2, FlashInfer and Triton `logits_soft_cap` (FA2 on sm_12x) | abandoned: Gemma 3 dropped it for QK-norm, Gemma 4 keeps only the final softcap | small: GGML's FA has the variant; jitLLM's sparse-gather path is not taken with it (`fattn_mma.cu`) | **Implement** |
| 4. LongRoPE: per-dimension short and long divisor sets, a fixed attention factor, the long set once the context exceeds `original_max_position_embeddings` | Microsoft Phi-3-mini/medium-128k (2024-04), Phi-3.5-mini and MoE (2024-08), Phi-4-mini (2025-02, rotary 0.75) ([config](https://huggingface.co/microsoft/Phi-3.5-mini-instruct/raw/main/config.json)) | Phi-4-mini 378K, Phi-3.5-mini 355K, Phi-3-mini-128k 147K, Phi-3.5-MoE 138K | both. vLLM uses the long set for every position when `max_model_len` exceeds the original length ([code](https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/model_executor/layers/rotary_embedding/phi3_long_rope_scaled_rope.py)); llama.cpp does so when the per-sequence context does | not updated: Microsoft's newest Phi (Phi-4-reasoning-vision, 2026-01) has no RoPE scaling | small: per-dimension divisors take llama3 scaling's form (7, 8) and the factor is YaRN's; choosing the set is new | **Implement:** choose the set per context, as both references do |
| 5. LayerNorm in the LLM path (mean-subtracting; no bias at Cohere, bias elsewhere) | Cohere Command R (2024-03), R7B, A (2025-03), A+ (2026-05), North (2026-06); Inception Jais 2 (2025-12); Phi-2, Falcon, StableLM 2, StarCoder2, GPT-J/NeoX, Bloom (2021–2024) | Command R v01 183K, Command A+ 40K; Phi-2 576K; Pythia-160m 3.37M (research) | both | not updated at Cohere and Jais 2; abandoned elsewhere | small: `ggml_norm` (the DiT has a BF16 LayerNorm) | **Implement,** with row 6 |
| 6. Parallel attention and FFN from one norm (`x + attn + ffn`) | Cohere as in row 5 (`use_parallel_block` true through Command A+, [modeling](https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/cohere2/modeling_cohere2.py)); Falcon 1/2 (2023–2024), GPT-J/NeoX, Phi-2, StableLM 2 12B | as row 5 | both | not updated at Cohere; abandoned elsewhere (Falcon 3 is a Llama, Falcon-H1 a hybrid) | small: block topology | **Implement:** Cohere still ships it, and rows 5 and 6 together also cover Falcon, Phi-2, StableLM and GPT-J/NeoX |
| 7. Legacy GGUF blocks Q4_0, Q4_1, Q5_0, Q5_1, IQ4_NL | Google's QAT GGUFs (Q4_0: Gemma 3 2025-03 → Gemma 4 2026-06). Any K- or I-quant tensor whose row is not a multiple of 256 falls back: Q4_K→Q5_0, Q5_K→Q5_1, Q6_K→Q8_0, Q2_K/Q3_K→Q4_0, I-quants→IQ4_NL ([`llama-quant.cpp`](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/src/llama-quant.cpp)) | gemma-4-E2B QAT Q4_0 515K; fallback use not countable | llama.cpp MMVQ and MMQ; vLLM moved GGUF to a plugin ([#39612](https://github.com/vllm-project/vllm/pull/39612), 2026-06) | current | small: the kernels are in jitLLM's GGML pin | **Implement:** checkpoint 1's 704-wide down experts likely carry fallback types |
| 8. Softmax top-k without renormalisation, beside shared experts | Qwen1.5-MoE (2024-03), Qwen2-57B-A14B (2024-06), DeepSeek V2 and V2-Lite (2024-05) ([config](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite/raw/main/config.json)) | Coder-V2-Lite 906K, Qwen1.5-MoE-A2.7B 463K, V2-Lite 233K | both | abandoned: Qwen3+ and DeepSeek V3+ renormalise or use sigmoid | small: a router flag | **Defer** until one is wanted; Coder-V2-Lite (MLA covered) would carry it |
| 9. Sparsemixer routing (two masked softmaxes, weights not renormalised, [modeling](https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/phimoe/modeling_phimoe.py)) | Phi-3.5-MoE (2024-08), Phi-tiny/mini-MoE (2025-06) | 138K, 86K, 31K | both | not updated since 2025-06 | small | **Defer** |
| 10. Dynamic NTK RoPE | Qwen 1 (2023-08), InternLM2/2.5 (2024), InternLM3 (2025-01) | InternLM3 75K, InternLM2.5 39K | both (llama.cpp's dynamic handling not verified) | abandoned: InternLM now builds on Qwen and GLM | small | **Drop** |
| 11. logn attention (query × log base L0 of the position, past L0) | Qwen 1 (2023-08) | Qwen-7B-Chat 51K | llama.cpp `qwen`; vLLM removed it after 0.23.0 | abandoned | small: Llama 4's per-position query scale (7) | **Drop** |
| 12. Dual chunk attention | Qwen2.5-7B/14B-1M (2025-01) | 111K, 30K | vLLM only | superseded by Qwen's linear-attention hybrids | medium | **Defer** |
| 13. ALiBi | BigScience Bloom (2022), MosaicML MPT (2023), Falcon-RW (2023), Baichuan 2 13B (2023), Jais 1 (2023) | bloomz-560m 1.11M, bloom-560m 519K (research-sized); others under 10K; MPT withdrawn from HF | llama.cpp keeps it; vLLM removed MPT (0.28, [#53608](https://github.com/vllm-project/vllm/pull/53608)), Baichuan and Jais 1 | abandoned: Jais 2 moved to RoPE | small: GGML FA `max_bias` | **Drop** |
| 14. Learned absolute positions | GPT-2 (2019), OPT, StarCoder 1 | gpt2 15.6M (tests and tokenizer use) | both | abandoned | small | **Drop** |
| 15. Mamba1 selective scan (per-channel A) | state-spaces Mamba (2023-12), Falcon Mamba (2024-07), AI21 Jamba (2024-03) → Jamba2 (2026-01), Zamba 1, Hymba | mamba-130m 316K; Falcon Mamba 7B 33K; Jamba Reasoning 3B 11K, Jamba2-3B 7K | both | not updated at AI21; abandoned elsewhere | medium: a scan kernel on 4's Mamba2 state class | **Defer** |
| 16. RWKV-7 (diagonal-plus-low-rank delta rule, token shift) | BlinkDL RWKV-4 (2023) → RWKV-7 G1j (2026-08-31) | rwkv7-g1 23K; RWKV-7 GGUFs about 3K | llama.cpp `rwkv7`; vLLM none | active | large | **Defer** |
| 17. RG-LRU (RecurrentGemma) | Google (2024-04) | 15K | neither | abandoned | large | **Drop** |
| 18. Gemma 3n AltUp, LAuReL, activation sparsity | Google (2025-06) | E2B 174K, unsloth E4B 150K | both | abandoned: Gemma 4's E-models keep only per-layer embeddings and KV sharing ([config](https://huggingface.co/google/gemma-4-E2B-it/raw/main/config.json)) | medium | **Defer,** with the E-models (excluded above) |
| 19. BitNet b1.58 (ternary weights with one scale, int8 activations, ReLU², sub-norms) | Microsoft bitnet-b1.58-2B-4T (2025-04); TII Falcon-E (2025-04 → 2026-04) | 21K, 16K for its GGUF | llama.cpp `bitnet` (TQ types lack CUDA); vLLM none | Microsoft uses it in 2026 non-LLM releases | medium: a W1.58A8 kernel | **Defer** |
| 20. TQ1_0, TQ2_0 | llama.cpp's ternary types (2024) | — | no CUDA upstream ([#11183](https://github.com/ggml-org/llama.cpp/pull/11183) open) | superseded by Q2_0 | — | **Drop** |
| 21. GPTQ act-order (`desc_act`, runtime `g_idx` permutation) | AutoGPTQ-era checkpoints (TheBloke, 2023) | top TheBloke GPTQ 23K | vLLM removed it 2026-09-08 ([#54809](https://github.com/vllm-project/vllm/pull/54809)); llm-compressor's default `static` order needs no runtime permutation ([code](https://raw.githubusercontent.com/vllm-project/llm-compressor/main/src/llmcompressor/modifiers/gptq/base.py)) | abandoned | small | **Drop:** GPTQ with static order needs nothing new |
| 22. AQLM | ISTA-DASLab (2024 → 2025-03) | ≤ 209 | vLLM removed it 2025-08 ([#22943](https://github.com/vllm-project/vllm/pull/22943)) | abandoned | medium | **Drop** |
| 23. SqueezeLLM | Berkeley (2023) | ≤ 39 | vLLM removed it 2024-09 ([#8220](https://github.com/vllm-project/vllm/pull/8220)) | abandoned | medium | **Drop** |
| 24. QuIP# (E8 lattice) | Cornell RelaxML (2024) | ≤ 137 | neither | superseded by QTIP, which EXL3 implements | — | **Drop** |
| 25. Minor: clip QKV (DBRX, withdrawn from HF), NormHead (Baichuan 2), LayerNorm1p (Nemotron-4), fused QKV layouts (Phi-3, InternLM2) | 2023–2024 | low | mixed | abandoned | small | **Drop,** or importer work if their family is wanted |

Notes:

- **Why implement rows 1–7.** Each is small and serves hundreds of
  thousands of downloads a month, even where the vendor has moved on:
  the installed base of Gemma 2 and 3, Mistral 7B, Mixtral, Llama 2 and
  Phi-3.x outlives the vendor's switch. Rows 5 and 6 are also Cohere's
  current design. The deferred rows are medium or large, or small but
  with a single low-use carrier. The dropped rows are ones the
  references have removed, or ones whose use is research or CI.
- **Fallback types in approved files.** Gemma 4 26B's experts are 704
  wide, not a multiple of 256, so under llama.cpp's rule the
  UD-Q4_K_M file's `ffn_down_exps` would be Q5_0, Q5_1 or Q8_0. This
  is inferred from the rule; the file's tensor types were not read. The
  importer should list every GGUF's types before choosing kernels.
- **Found in passing (not legacy): 1- and 2-bit Bonsai.** PrismML's
  `Q1_0` (±1, one FP16 scale per 128) and `Q2_0` (ternary codes, one
  scale per 64) have been in upstream GGML since 2026-04 and 2026-07,
  CUDA included ([#21629](https://github.com/ggml-org/llama.cpp/pull/21629),
  [#25707](https://github.com/ggml-org/llama.cpp/pull/25707)); jitLLM's
  GGML pin has both in `mmvq.cu`. They carry Qwen3.6-27B (`qwen35`,
  checkpoint 10's architecture):
  - `prism-ml/Bonsai-27B-gguf`: 432K a month.
  - `Ternary-Bonsai-27B-gguf`: 635K.
  - `Ternary-Bonsai-2-27B-gguf` (2026-09-16, on Qwen3.8-27B): 3.58M.
    Its `PQ2_0` and `PTQ1_0` need PrismML's llama.cpp fork, which
    applies a Hadamard transform to activations; the upstream PR
    [#29077](https://github.com/ggml-org/llama.cpp/pull/29077) was
    closed unmerged
    ([formats](https://docs.prismml.com/download/formats)).

  **Proposed:** add Q1_0 and Q2_0 in M3.5 if the owner accepts a
  Bonsai fixture (small cost, heavy use, a covered architecture). Defer
  the fork-only formats until upstream takes them, or until the owner
  accepts the fork as a reference.

**Proposed fixtures.** Four checkpoints, about 17.9 GB together, cover
rows 1–6 and Q4_0. Each has a same-format reference in llama.cpp.
Revisions and sizes come from each repository's API on 2026-09-29.

| Fixture | Size | Reference | Covers |
| --- | ---: | --- | --- |
| `ggml-org/gemma-3-4b-it-qat-GGUF@bbcac0d065076c47042838c0675c602411b0dd4c`, `gemma-3-4b-it-qat-Q4_0.gguf` | 2,526,080,992 B | llama.cpp `gemma3` | SentencePiece 262K, linear RoPE ×8 on globals, Google's QAT Q4_0, the Gemma 3 template |
| `bartowski/gemma-2-2b-it-GGUF@855f67caed130e1befc571b52bd181be2e858883`, `gemma-2-2b-it-Q8_0.gguf` | 2,784,495,456 B | llama.cpp `gemma2` | attention softcap 50, SentencePiece 256K |
| `bartowski/Phi-3.5-mini-instruct-GGUF@6d70da17e749a471ccb62ade694486011a75cda3`, `Phi-3.5-mini-instruct-Q8_0.gguf` | 4,061,222,688 B | llama.cpp `phi3` | LongRoPE with both sets non-trivial (Phi-4-mini's short set is all ones); Llama 2's 32K SentencePiece, which Mistral 7B and Mixtral 8x7B also use |
| `bartowski/c4ai-command-r7b-12-2024-GGUF@bfc7a934c45cb839d84c8ca01d87f1cfa51aaa3f`, `c4ai-command-r7b-12-2024-Q8_0.gguf` | 8,541,100,160 B | llama.cpp `cohere2` | LayerNorm without bias, the parallel block, logit scale 0.25, Cohere's 256K BPE and template |
| Optional: `prism-ml/Bonsai-27B-gguf@f10afb355f104535e3e3e98cf7ab7795c72bd292`, `Bonsai-27B-Q1_0.gguf` and `Bonsai-27B-dspark-Q4_1.gguf` | ≈3.8 GB + ≈1.8 GB | llama.cpp upstream | Q1_0 and Q4_1 on checkpoint 10's architecture, with a DSpark drafter |

Q5_0, Q5_1 and IQ4_NL come from whichever approved file carries
fallback tensors. Q2_0 would need a Ternary Bonsai file, not pinned
here.

## Generative media: video and image

A new category beside M3's Qwen-Image, for the owner's MiniMax H3.

- **Video: `MiniMaxAI/MiniMax-H3@42ed227ee7df40d41602854ae760620d6eb651fe`**
  (2026-07-28; 3.63M downloads; video with synchronized stereo audio;
  [card](https://huggingface.co/MiniMaxAI/MiniMax-H3)).
  - DiT `MiniMaxH3Transformer3DModel`: 33B, 50 layers + 2 refiner, hidden
    5,376, 56 heads × 128, full 3D spatio-temporal attention (sparse
    attention was trained but not released), 3D RoPE over (t, h, w),
    QK-norm, one stream for video and audio tokens; ≈13B of AdaLN
    parameters that can be precomputed; CFG-distilled
    ([config](https://huggingface.co/MiniMaxAI/MiniMax-H3/raw/main/transformer/config.json),
    [GitHub](https://github.com/MiniMax-AI/MiniMax-H3)).
  - VAE `AutoencoderKLMiniMaxH3`: 16× spatial, 4× temporal, 24 channels; a
    convolutional encoder and a 36-layer transformer decoder over 17-frame
    clips ([config](https://huggingface.co/MiniMaxAI/MiniMax-H3/raw/main/vae/config.json));
    an audio VAE (32 kHz → 40 Hz latents).
  - Text encoder: Qwen3-VL-32B, hidden states from layer 50
    ([model_index](https://huggingface.co/MiniMaxAI/MiniMax-H3/raw/main/model_index.json)),
    a larger member of the Qwen3-VL family jitLLM already runs.
  - Output: 4–15 s at 24 fps, short side 768; flow matching, 50 steps
    suggested (8 with a turbo LoRA).
  - Fit: the full BF16 pipeline is ≈135 GB; vLLM's GB10 recipe makes FP8
    mandatory ([recipe](https://recipes.vllm.ai/MiniMaxAI/MiniMax-H3)).
    Comfy-Org's pruned BF16 DiT is 40.2 GB and its layer-50 encoder 51.5 GB
    ([repack](https://huggingface.co/api/models/Comfy-Org/MiniMax-H3/tree/main/diffusion_models)),
    so run as phases (encode, release, denoise, decode) each peaks near
    51 GB: jitLLM's composition and paging model (D-089). That "pruned"
    means precomputed AdaLN is inferred from sizes, not stated.
  - Reference: diffusers `MiniMaxH3ModularPipeline`, vLLM-Omni, SGLang,
    ComfyUI. GB10: 80–111 s a request at 768×448 with vLLM-Omni FP8
    ([joeynyc](https://github.com/joeynyc/MiniMax-H3-DGX-Spark)); one
    ComfyUI report of a host loss from unified-memory exhaustion
    ([#16587](https://github.com/Comfy-Org/ComfyUI/issues/16587)).
  - New for jitLLM: latent sequences of ≈31K tokens (5 s) to ≈92K (15 s)
    against Qwen-Image's ≈4K (arithmetic, assuming 1 + (F−1)/4 latent
    frames), so long-sequence attention dominates and FP8 attention is a
    likely lever; 3D RoPE; joint audio tokens; precomputed AdaLN; a
    transformer VAE decoder; an audio VAE.
  - License (informational): the MiniMax H3 Community License excludes the
    EU, UK, South Korea and the USA from its grant
    ([LICENSE](https://huggingface.co/MiniMaxAI/MiniMax-H3/raw/main/LICENSE)).
  - Top-tier: #1 open-weights text-to-video and image-to-video on
    Artificial Analysis ([board](https://artificialanalysis.ai/video/leaderboard/text-to-video/open-weights)).
- **Video fallback: `Lightricks/LTX-2.5@5e6e71018ee1756ed329b697a7b4aedc934dfce9`**
  (22B dual-stream audio-video DiT, Gemma 4 12B encoder, 8 distilled
  steps; DiT 42.0 GB + encoder 26.3 GB, fits at BF16; official NVFP4;
  LTX-2.x Community License). More new surface than H3.
- **Image:** Qwen-Image-2.1 already leads the open text-to-image board.
  The cheapest second family is Ideogram 4
  (`ideogram-ai/ideogram-4-fp8@ee79a7237b519f1402ceacf952f30c8a31ec5073`,
  9.3B single-stream DiT on a Qwen3-VL-8B encoder, FP8 and NF4 only,
  non-commercial); FLUX.2-klein-9B is the alternative.
- **Image, owner-named (2026-10-03):
  `inclusionAI/Ming-Image-0.1-Design@208087ada1486931692c1896f38d4cd16ff3df82`**
  (created 2026-09-17, updated 2026-09-23; MIT;
  [card](https://huggingface.co/inclusionAI/Ming-Image-0.1-Design)), a
  second text-to-image model for fleshing out the image pipelines and
  API. Text-rich design: UI screens, infographics, posters, with
  readable text, and RGBA output with transparent backgrounds. Read from
  its configs on 2026-10-03; nothing was downloaded or run.
  - DiT `DiffusionTransformer`: dim 3,840, 30 layers + 2 refiner layers,
    30 heads (no GQA), QK-norm, 3-axis RoPE (axes 32/48/48, θ 256),
    patch 2, 16 latent channels, caption features 2,560 wide; 12.31 GB in
    BF16, about 6.15B parameters
    ([config](https://huggingface.co/inclusionAI/Ming-Image-0.1-Design/raw/main/transformer/config.json)).
  - Conditioning: `mllm` is a `BailingMM2NativeForConditionalGeneration`
    (`bailingmm_moe_v2_lite`: 20 layers, hidden 2,048, 16 heads / 4 KV,
    256 experts a layer; 34.0 GB in BF16) whose config names a Qwen2.5-ViT
    tower (the weight index carries no vision tensors), then a `connector`,
    a Qwen2 1.5B causal LM (28 layers, hidden 1,536) stored in F32
    (6.17 GB), and an `mlp` projection
    ([mllm](https://huggingface.co/inclusionAI/Ming-Image-0.1-Design/raw/main/mllm/config.json),
    [connector](https://huggingface.co/inclusionAI/Ming-Image-0.1-Design/raw/main/connector/config.json)).
  - VAE `AutoencoderKLQwenImage` with `input_channels` 4 and z 16: Qwen-Image's
    VAE family taking RGBA
    ([config](https://huggingface.co/inclusionAI/Ming-Image-0.1-Design/raw/main/vae/config.json)).
  - Sampling: `FlowMatchEulerDiscreteScheduler`, shift 6.0, no dynamic
    shifting; the card recommends 12 steps, CFG 1.0 (no unconditional
    pass), 2,048² (or 1,024² for speed). These are defaults: size, steps,
    guidance and seed are per-request inputs, as in Qwen-Image's native
    pipeline. Sizes are bounded by the VAE's 8× and the patch of 2
    (multiples of 16) and by the RoPE table (512 positions an axis,
    about 8,192 pixels a side, arithmetic), not fixed.
  - Fit: about 53 GB of weights at their stored precisions, so it fits a
    Spark whole, and runs as phases under D-089 like Qwen-Image. The card
    validates only one 80 GiB GPU; GB10 behaviour is unknown.
  - Reference: [inclusionAI/Ming-Image](https://github.com/inclusionAI/Ming-Image)
    (`infer.py`, MIT), and vLLM-Omni per the card; pin both at entry.
  - New for jitLLM: an MoE multimodal encoder as text conditioner, a
    causal-LM connector, RGBA latents and alpha in the image API, a
    2,048² latent sequence (about 16K tokens at patch 2 over 8× VAE
    downsampling, arithmetic), and a second DiT layout beside
    Qwen-Image's (refiner layers, 3-axis RoPE). Shares the VAE family and the
    flow-matching scheduler with Qwen-Image.
  - Companion (not planned): `Ming-Image-0.1-Design-Layer` decomposes a
    finished design into editable transparent layers, an output shape no
    current route has.

## Media inputs, decision models and generation APIs

Added 2026-10-02 at the owner's request (D-101), as a desk study like the
rest of this file: nothing was downloaded or run. Facts come from the
Hugging Face API, configs, processor configs, cards and engine source on
that day. Encoder sizes marked "≈" are inferred from mmproj GGUF byte
sizes. GB10 claims are creator-reported.

### Which planned checkpoints take which media

| Model (as pinned) | Image | Video | Audio | Encoder | Notes |
| --- | --- | --- | --- | --- | --- |
| Qwen3.8 Flash Next (M3) | yes, several | yes | no | Qwen3-VL ViT: 27 layers, width 1152, patch 16, temporal 2, 2×2 merge, ≈0.45B | Interleaved M-RoPE in the LLM. Video at 2 fps, 4–768 frames, text timestamps. Mia's NVFP4 keeps the tower, quantized (MXFP8 plus NVFP4 fc2). The GGUF ships an mmproj ([config](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/raw/main/config.json)) |
| Qwen3.8-27B, Ornith 1.5, Bonsai-27B | yes | yes (Ornith's card is silent) | no | the same tower | EXL3 `SC_3.00` keeps it in BF16; the GGUFs ship an mmproj ([card](https://huggingface.co/Qwen/Qwen3.8-27B)) |
| DeepSeek V4 Flash 0731 (M3) | no | no | no | none | Text-only ([config](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/raw/main/config.json)) |
| DeepSeek V4 Flash Vision-Exp | yes, interleaved | no | no | DeepSeek-ViT: 32 layers, width 1024, patch 14, 2D RoPE, 3×3 aligner, ≈0.47B | A separate checkpoint continued-trained from 0731 ([card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp)). At most 384 tokens an image, in an "N-layout" padded to CSA's compression. Its GGUFs and encoder are in the pinned `antirez/deepseek-v4-gguf`. llama.cpp `deepseek4v`, DwarfStar on CUDA |
| DeepSeek V4.1 Flash (M4) | yes | no | no | the same ViT family | At most 1,024 tokens an image, in reading order ([vision.py](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/raw/main/inference/vision.py)). Mia's EXL3 keeps the tower but did not validate it. No llama.cpp architecture |
| GLM-5.3 Flash (M4) | yes | yes | no | GLM ViT: 24 layers, width 1024, patch 14, 2D RoPE, ≈0.58B | No M-RoPE in the LLM. Video at 2 fps with per-frame timestamps ([processor](https://huggingface.co/zai-org/GLM-5.3-Flash/raw/main/processor_config.json)). The large GLM-5.3 is text-only |
| Gemma 4 26B-A4B and 31B | yes | as frames (32 × 70 tokens) | no | ≈550M, patch 16, 3×3 pooling | Images at 70–1,120 tokens (280 by default). Image tokens attend bidirectionally on sliding layers only. Audio exists only on E2B, E4B and 12B ([card](https://huggingface.co/google/gemma-4-26B-A4B-it)) |
| Gemma 3 4B QAT (legacy) | yes | no | no | SigLIP, 896 px, 256 tokens | The pinned ggml-org GGUF includes the mmproj |
| Llama 4 Scout | yes, tested up to 5 | no | no | ViT, 336 px tiles (up to 16), ≈0.87B | The unsloth GGUF ships an mmproj |
| Mistral Small 4 | yes | no | no | Pixtral, patch 14, 2×2 merger | `[IMG_BREAK]`/`[IMG_END]` tokens |
| Muse Glimmer 30B | yes | processor and vLLM only; card says images | no | ViT-G/14, ≈1.8B, at most 4,096 tokens an image | Meta's GGUF ships a Q4_K_M mmproj ([card](https://huggingface.co/meta-models/Muse-Glimmer-30B)) |
| MiMo-V2.6-Flash-RL | yes | yes | original only | ViT 681M; audio tokenizer 308M + 127M, 24 kHz, 128 mels | The pinned EXL3 build drops the audio encoder ([card](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-RL)) |
| gpt-oss-120b, Nemotron 3 Super, Kimi-Linear, Llama 3.3 70B, GLM-4.7-Flash, Gemma 2 2B, Phi-3.5-mini, Command R7B | no | no | no | none | Text-only |

Reference engines for the media path:

- **llama.cpp mtmd:** images, audio and video, with video through ffmpeg
  at 4 fps. It has projectors for every vision family above except V4.1
  ([docs](https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md)).
- **vLLM:** registers every family above. Upstream MiMo is text-only.
- **SGLang:** covers MiMo's audio and Gemma 4's audio.
- **ExLlamaV3:** images on its EXL3 families. Video is unverified.

**Audio carrier.** No pinned checkpoint takes audio, so the owner chose
a carrier on 2026-10-02:
`google/gemma-4-E4B-it@ee0ef6023621cff504d758262d4e04895a5af4a2`
(Apache-2.0, 8.0B, 4.4M downloads in 30 days).
It takes 16 kHz audio as 128 mel bins, about 30 s a clip, through a
≈300M conformer, and shares Gemma 4's tokenizer, template and vision.
Longer audio is split into such windows, not refused (D-102).
llama.cpp, vLLM, SGLang and transformers run it. Alternatives:
- `nvidia/Nemotron-3-Nano-Omni-30B-A3B`: audio, image and video in one
  model, NVIDIA license;
- `Qwen/Qwen3-ASR-1.7B`;
- `openai/whisper-large-v3-turbo`, for transcription only.

### Decision models: Clef and the Jev API

- **The API.** TypeSafe's Jev / SystemOne: `POST /v1/systemone` and
  `GET /v1/models`, with bearer authentication
  ([API](https://docs.typesafe.ai/api.md),
  [OpenAPI 0.2.0](https://api.typesafe.ai/openapi.json)).
  - **Request:** `model`, `state` (string or JSON) and `questions` keyed
    by ID. Each question is `noul` (true/false), `choice` (up to 255
    named options) or `score` (2–10 ordered levels), with `instructions`
    and `criteria`.
  - **Response:** `answers`, `usage` and the resolved `model`.
    - noul: P(true);
    - choice: the argmax, `probabilities` and `confidence`;
    - score: Σ i·pᵢ, a `legend`, `probabilities` and `confidence`.

    Confidence is (p_max − 1/n)/(1 − 1/n) for choice, and a
    distance-normalized formula for score
    ([confidence](https://docs.typesafe.ai/confidence.md)).
  - **Not offered:** streaming or a batch route.
    M3.5 requires jitLLM to batch independent requests to each decision
    model internally; this does not require a batch wire route.
  - **Clients:** the TypeSafe SDKs (`typesafe-sdk` 0.7.2,
    `@typesafe-ai/sdk` 0.6.0, MIT; base URL from `TYPESAFE_BASE_URL`);
    Vercel AI Gateway and `@ai-sdk/typesafe-ai`; OpenRouter; Pydantic AI
    Gateway; TanStack AI; LiteLLM's Jev classifier.
- **Workers AI** serves Clef with the same body. Its limits are 1–64
  questions, at most 4 images (data URLs, PNG, JPEG or WebP) and 64K
  tokens ([model page](https://developers.cloudflare.com/workers-ai/models/clef/)).
  jitLLM does not adopt the counts as its limits (D-102, 2026-10-03): no
  question or image counts, TypeSafe's option (2–255) and score-level
  (2–10) ranges as wire ranges, image bytes and pixels from memory and
  the model's context.
- **The models.** Both are Apache-2.0
  ([clef](https://huggingface.co/Cloudflare/clef),
  [clef-flash](https://huggingface.co/Cloudflare/clef-flash)).

  | Model | Revision | Backbone | Backbone size | Head |
  | --- | --- | --- | --- | --- |
  | Clef | `2f3de3dd85f379784083b0814d997ab627200f0c` | Qwen3.8-27B (`qwen3_5`, 64 layers, 3 linear : 1 full attention) | 54.7 GB BF16 | 128M |
  | Clef-flash | `17f0b0ad64efb65d273590632833508766b2aae6` | Qwen3.5-9B (32 layers) | 18.8 GB BF16 | 122M |

  - Both keep the Qwen3-VL vision tower and drop MTP.
  - Against the base models, Clef's linear projections differ in layers
    40–63 and Clef-flash's in every layer; embeddings, output head,
    norms and vision are unchanged (merged adapters, inferred from the
    weights).
- **The forward pass** (`joint_schema_model.py`):
  1. One causal prefill without a cache over a fixed prompt: the state,
     any media placeholders, then each question's instruction span and
     one span per option.
  2. The head (2 evidence-routing layers and 4 decoder layers, width
     1024) reads every position's hidden state after the final norm.
     Its inputs are the mean of each question span, the mean of each
     option span, the output-head rows of each option's tokens, and the
     last token.
  3. Questions attend to each other, so answers are joint.
  4. Each question's option logits pass through a softmax.

  The reference is tested on transformers 5.10.2 with torch 2.11.
- **Where the reference differs from TypeSafe's documented behavior:**
  - its `confidence` is the top probability;
  - it echoes the requested `model`;
  - the question ID enters the prompt;
  - `instructions` is optional;
  - questions interact.
- **Other engines.**
  - vLLM would load only the backbone as a chat model; its `/v1/systemone`
    work is generic ([PR #59299](https://github.com/vllm-project/vllm/pull/59299)).
  - llama.cpp's text-only Clef support is open
    ([PR #29831](https://github.com/ggml-org/llama.cpp/pull/29831)), with
    GGUFs at `ggml-org/Clef-GGUF` and `ggml-org/Clef-Flash-GGUF`.

### Text-to-speech testbeds

Chosen by the owner on 2026-10-02.

- **Breeze-TTS-2**
  (`BreezeBlue/Breeze-TTS-2@3e28c5151381a722f1d8661b4118c298caa77aa4`,
  [card](https://huggingface.co/BreezeBlue/Breeze-TTS-2)).
  - **Ranking:** #1 open-weight model on Artificial Analysis's TTS board
    (creator-reported).
  - **Size:** 3.47B parameters, 6.97 GB in BF16, plus a 0.68 GB audio
    tokenizer.
  - **Components:** a T5Gemma2 text encoder (26 layers, 1152 wide, window
    512); a Qwen3 backbone (28 layers, 2048 wide, 16 query and 8 KV
    heads, llama3 RoPE scaling); a 12-layer depth decoder over 16
    codebooks of 2,051 entries; a Mimi-style codec at 12.5 Hz; and a
    Qwen3-TTS 12 Hz audio tokenizer. Audio is 24 kHz.
  - **Generation:** sampled at temperature 0.9, with classifier-free
    guidance (`cfg_scale` 4 for instructions). Voices come from a
    reference clip and its transcript (cloning), a description (design),
    or both (direction). Vocal events such as `(laugh)` are written
    inline. English and Chinese.
  - **Reference:** PyTorch inference code at
    [breezeblue-ai/breeze-tts](https://github.com/breezeblue-ai/breeze-tts)
    (Apache-2.0).
    - Its server answers a multipart `POST /v1/audio/speech` (`text`,
      `instruction`, `ref_audio`, `ref_text`, `cfg_scale`, `seed`) with
      streamed 24 kHz 16-bit PCM.
    - Stage-wise CUDA graphs give under 40 ms to first audio and a
      real-time factor of 0.32 on an H100 (creator-reported).
  - **License (informational, D-087):** the weights are under the
    BreezeBlue Research and Non-Commercial License.
- **Kokoro-82M**
  (`hexgrad/Kokoro-82M@f3ff3571791e39611d31c381e3a41a3af07b4987`,
  [card](https://huggingface.co/hexgrad/Kokoro-82M)).
  - **Popularity:** Apache-2.0, 11.4M downloads in 30 days.
  - **Architecture:** StyleTTS 2 with an ISTFTNet vocoder and a PL-BERT
    text encoder, over 178 phoneme tokens, at 24 kHz. Duration and
    prosody prediction are LSTM-based, and the decoder uses AdaIN
    convolutions.
  - **Weights:** `kokoro-v1_0.pth` (327 MB) and 54 voice packs (`.pt`),
    all PyTorch pickles. Import must read them without executing code
    (D-009). `onnx-community/Kokoro-82M-v1.0-ONNX` is an Apache-2.0
    conversion.
  - **Reference:** the `kokoro` package (≥ 0.9.2). Its phonemes come from
    `misaki`, which falls back to espeak-ng (GPL-3.0) for unknown words.

### Generation APIs and clients

- **OpenAI's image routes.**
  - `POST /v1/images/generations` returns `b64_json` and can stream
    partial images.
  - `POST /v1/images/edits` takes up to 16 multipart `image[]` and a mask.
    Codex sends a JSON `images[]` form instead.
  - `/v1/images/variations` is retired. Chat Completions cannot output
    images ([guide](https://developers.openai.com/api/docs/guides/image-generation)).
- **Local engines.** vLLM-Omni serves the image routes with
  `negative_prompt`, `num_inference_steps`, `guidance_scale` and `seed`, and
  `/v1/videos` as asynchronous jobs.
  - The job flow: create returns `queued`, `GET /{id}` reports progress,
    and `GET /{id}/content` downloads the result
    ([api_server.py](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/entrypoints/openai/api_server.py)).
  - OpenAI shut its own `/v1/videos` down on 2026-09-24, but SGLang and
    LiteLLM keep the shape.
- **Clients.**

  | Client | Image generation today | Can it use a local server? |
  | --- | --- | --- |
  | Open WebUI | `/images/generations` and `/images/edits`, A1111, ComfyUI or Gemini | Yes |
  | LibreChat | OpenAI image tools | Yes, through `IMAGE_GEN_OAI_BASEURL` |
  | Codex | Its built-in tool calls `{base_url}/images/generations` with `gpt-image-2` | Gated on OpenAI authentication; whether a custom provider passes is unverified |
  | Claude Code, OpenCode | None built in | Only through MCP |
  | Antigravity, Cursor | Hosted Nano Banana | No |
  | Gemini CLI | Its nanobanana extension, an MCP server | Its base URL is configurable |

  Claude Code, Codex, Gemini CLI, OpenCode, Zed and Cursor all pass an MCP
  tool's image content to the model; Continue fails on it. So an MCP
  server that calls the image routes is the one portable way to mix
  generated images into the coding agents' chats.
- **Demand.** On Hugging Face over 30 days, the generation tags rank
  text-to-image (SDXL, Qwen-Image-2.1) and image editing (Qwen-Image-Edit,
  FLUX.2) first. Image-to-video (Wan 2.2, LTX-2.5, MiniMax H3) outdraws
  text-to-video. Then come text-to-speech (Kokoro-82M at 11.4M, XTTS-v2,
  Qwen3-TTS) and speech recognition (whisper-large-v3-turbo). Music and 3D
  are niche.

## Order of bring-up

The owner-named families first, then by new surface; each step reuses the
one before.

1. **Gemma 4 26B-A4B, then 31B** (owner-named, M7): the sliding/global
   ring cache, per-type head dims, GeGLU, sandwich norms, softcap, the ▁
   tokenizer, the assistant drafter; then the EXL3 MoE (mcg) and EXL3
   dense (mul1, codebook A/B) on the same models.
2. **gpt-oss-120b:** sinks and the 1:1 window on the ring cache from step
   1; MXFP4 experts; o200k and harmony; EAGLE3.
3. **MiMo-V2.6-Flash-RL** (owner-named): EXL3 MoE at scale with mixed
   widths, the ExLlamaV3 re-pin, local-only sinks, multi-layer MTP.
4. **Meta:** Muse Glimmer (NoPE globals, DFlash), Llama 4 Scout (chunked
   attention, top-1 routing), Llama 3.3 70B (dense floor, Llama 3
   tokenizer, AWQ).
5. **Nemotron 3 Super** (Lightning first if wanted): Mamba2, NoPE, squared
   ReLU, LatentMoE.
6. **Kimi Linear:** KDA and MLA, ahead of M4's GLM-5.3 Flash.
7. **Mistral Small 4:** MLA with q-LoRA, Tekken, EAGLE on MLA,
   compressed-tensors NVFP4.
8. **Qwen3.8-27B:** EXL3 mixed widths, FP8 blocks, DFlash2; little new
   architecture.
9. **Ornith 1.5 and GLM-4.7-Flash:** M7's and M4's prerequisites, mostly
   covered by then.

An alternative is to start with Qwen3.8-27B, the smallest step from
Qwen3.8 Flash Next, as a shakedown of the "adding a model family" guide.

## Considered and excluded

| Family or checkpoint | Reason |
| --- | --- |
| Microsoft Phi | Newest text model is Phi-4-reasoning-plus (2025-04-30, 14B, 32K, 11.6K downloads, GPQA 68.9, [card](https://huggingface.co/microsoft/Phi-4-reasoning-plus)); no Phi-5 on HF ([org](https://huggingface.co/api/models?author=microsoft&sort=createdAt&direction=-1&limit=80)). Its unique features (LongRoPE, Phi-3.5-MoE's LayerNorm and sparsemixer) are in the legacy tier |
| IBM Granite | Granite 4.2 (2026-08) is plain dense (`granite-4.2-30b@9e668ce1c538387ef24d3644e9b0606647762636`, GPQA 66.4, 31K downloads); the Mamba2 hybrid 4.0-H-Small (2025-09) is weak (GPQA 40.6) and dropped by IBM ([card](https://huggingface.co/ibm-granite/granite-4.2-30b)). Mamba2 is covered by Nemotron |
| Cohere Command A+ (218B MoE) | Fits only below 3 bits (IQ2_M 76.6 GB); the official W4A4 targets a B200 |
| Kimi K2.x, K3; MiniMax M3 (427B), M1/Text-01 (456B); GLM-5.x and GLM-4.5–4.7 (355B); Mistral Large 3 (675B); Llama 4 Maverick (400B); Nemotron 3 Ultra; DeepSeek V4.1 Flash, V4 Pro, V3.x; Tencent Hy4, Hy3 (IQ1_M only); Inkling; Qwen3.8-2.4T; Ornith 397B | Too big for one Spark at a usable quant. MiniMax M3 (UD-Q2_K_XL 143 GB) and Hy3 are two-Spark candidates for after M4 |
| DiffusionGemma, `gemma-4-12B`, E2B/E4B | Block diffusion is M9's; the others are weaker or edge-sized (E-models' per-layer embeddings and KV sharing are the only unique parts) |
| gpt-oss-safeguard, `circuit-sparsity` | Same architecture; research model |
| Seed-OSS, EXAONE 4.5, OLMo 3.1, Apertus, Trinity, ERNIE 4.5, LFM2.5, Hunyuan 7B | Fit, but second-tier by usage; features they add (post-norm, xIELU, short convolutions, NTK-alpha, SentencePiece with interleaved RoPE) are niche. EXAONE is non-commercial (informational) |
| "Space Bunny Alpha" | OpenRouter's #1 free model is an anonymous stealth preview, not open weights ([page](https://openrouter.ai/stealth/space-bunny-alpha)) |
| EXL2 | Superseded by EXL3; turboderp's last EXL2 upload was 2025-05 |
| bitsandbytes NF4/INT8 | Fine-tuning and research oriented; no top publisher ships it for these families; no GB10 serving reference |
| HQQ | No top-publisher checkpoints; no GB10 reference |
| MLX affine 4-bit | Stays M9's import item (TensorFold is its reference); unchanged here |

## Open questions for the owner

*Answered by the owner, 2026-09-29:* (1) all 13 approved; (2) Muse Glimmer
alongside Llama 3.3 70B; (3) the community 2.27 bpw MiMo EXL3 build
accepted; (4) re-pin ExLlamaV3 in M3.5, keeping the old pin as the anchor;
(5) concentrate formats as proposed, which gives cleaner performance
comparisons; (6) no Mistral-native NVFP4 import: use its GGUF or EXL3
builds; (7) identify the legacy tier's features that are not subsets of
already-covered ones and judge whether each is worth implementing, noting
which models and vendors used it and whether it was abandoned or just not
updated (answered in [Legacy-tier features](#legacy-tier-features),
for the owner); (8) MiniMax H3 is in scope (generative media, last); (9) no
optional models for now. Also: bring in the MLX affine import (moved
from M9) so TensorFold becomes a same-format comparator.

1. **Set size.** Approve the ten-checkpoint core, the three additions
   (Gemma 4 31B, Ornith, GLM-4.7-Flash), or trim: Llama 4 Scout is stale
   and could drop, leaving chunked attention and top-1 routing uncovered.
2. **Meta.** Muse Glimmer is Meta's current open model but not a Llama;
   is it the Meta dense pick with Llama 3.3 70B, or instead of it?
3. **MiMo's format.** Only a 2.27 bpw EXL3 build (community-made, 85 GiB)
   fits with its drafters; its quality against the full model is not
   measured. Accept it, or wait for an official smaller MiMo?
4. **ExLlamaV3 re-pin.** The EXL3 picks need v1.5.2 or v1.5.3; jitLLM's
   pin is `6b84a21b`. Re-pin in M3.5 (and redo P3's parity), keeping the
   old pin as the anchor?
5. **Formats per checkpoint.** Qwen3.8-27B carries three formats (EXL3,
   GGUF, FP8) and Gemma two (GGUF, EXL3): is that the right way to cover
   formats, or should formats spread across families?
6. **Mistral's NVFP4 layout.** The official NVFP4 has only Mistral-native
   files; import that layout, or use the GGUF or EXL3 builds instead?
7. **Legacy tier.** Should older generations still in wide use (Gemma 2,
   Phi-3.5, Mistral 7B, Command R7B, Llama 3.2) get cheap fixtures for
   their unique features, or wait for demand?
8. **Media.** Is MiniMax H3 (video) in M3.5's scope, or a later milestone?
   Its license excludes the USA and EU (informational under D-087).
9. **Optional LLMs.** Any of MiniMax M2.7, Laguna S 2.1, Step 3.7 Flash,
   Ling 3.0 flash, Nemotron Lightning or Cohere North the owner wants for
   usage rather than coverage?

## Not verified

- MiMo's MTP layer count (config 3, card 5); the EXL3 build's codebook.
- The per-tensor K mixes inside turboderp's Gemma 4 builds; the SC_3.00
  mix beyond a truncated read.
- Mistral Small 4's router function; whether `use_parallel_block` acts in
  `cohere2_moe`; Llama 4 Scout's chat template (gated).
- llama.cpp's MTP for GLM-4.7-Flash, DFlash for Muse Glimmer, Laguna's and
  LFM's drafters upstream; ExLlamaV3's support for Gemma's assistant.
- GB10 speeds for Kimi Linear, Llama 4 Scout and MiniMax M2.7.
- Laguna's NVFP4 size (shard sum ≈99.7 GB, card ≈71 GB).
- OpenRouter's own ranking tables (read through its collection pages).
- Whether ExLlamaV3 v1.5.3 includes the aarch64 build guards.
- Legacy tier:
  - Which GGUF types the approved files actually hold: Gemma 4 26B's
    fallback types were inferred, not read.
  - Whether llama.cpp honours InternLM's dynamic NTK.
  - Whether Ternary-Bonsai-27B's `Q2_0` file is unrotated and runs on
    stock llama.cpp. Bonsai 2's card warns that its own rotated weights
    load there silently and give garbage.
  - The Bonsai file sizes, which were read as rounded GB.
  - Command R7B's official config: the repository is gated and the
    values come from mirrors.
