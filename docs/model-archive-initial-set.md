<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Initial model archive set

Download inventory assembled from the project documents on 2026-10-10.
This is the starting library, to expand after the planned models and formats
are covered. It records intended acquisitions, not completed downloads or
execution support. [The archive catalog](model-archive.md#the-catalog) records
what is actually present; [model-support.md](model-support.md) records what
runs and its qualification.

The sources are the [approved M3.5 covering set](m35-families.md), the
[current plan](plan.md) (including later media and decision-model additions),
the [M3 pins](experiments/fast-swap/pins.json), the M2 fixture pins, and
[M4 references](m4-references.md). Repository names, revisions and quantization
labels below originated in those sources. The acquisition queue now resolves
all selected revisions to full commits and validates file selections against
Hub metadata. The Gemma expansion and additional format carriers below
supplement the original planning tables; the frozen queue is the exact
transfer specification.

## Locations and first batch

| Purpose | Location in the owner's environment |
| --- | --- |
| NAS mount | `/mnt/llm`, SMB share `//192.168.0.3/llm` |
| Source checkpoint archive | `/mnt/llm/checkpoints/hf/<org>/<repo>@<full-commit>/` |
| Actual inventory | `/mnt/llm/checkpoints/catalog/catalog.md` and `catalog.json` |
| Older MLX offload to migrate | `/mnt/llm/huggingface-mlx/<org>--<repo>@<full-commit>/` |
| Staged checkpoints on each Spark | `~/.local/share/llmp/models/<org>/<repo>@<first-8-of-commit>/` |
| Older Spark reference checkpoints | `~/.local/share/llmp/reference-models/`; identify repo and commit before migration |

The archive was initialized on the mounted NAS on 2026-10-10. Acquisition
is running; consult the catalog and batch state for current completion.
Existing NAS and Spark copies were inventoried for reuse, with admission
conditional on the archive writer verifying their pinned hashes.

The owner authorized the complete inventory, including optional candidates,
on 2026-10-10. The queue starts with small fixtures and speech checkpoints,
then all Gemma 4 sizes, the approved covering set and comparisons, media,
M4 targets, optional families and additional format carriers. It contains
149 immutable revision selections, about 5.73 TB before source reuse.
Existing Spark checkpoint paths are preserved; verified older NAS MLX
files move into the archive.

The [downloader quickstart](model-archive.md#downloader-quickstart) is the
procedure. Check the catalog, then use `--dry-run` with the exact revision
and file selection. Check the mounted share before the first write with
`--init-root`. Every actual transfer runs under `hostlock shared`. Keep the
token in ignored `.env` as `HF_TOKEN`; mise exports it, never an argument.
Rerun the same pinned command to resume. A GGUF selection must include all
shards of the target quantization and explicitly include its needed
`mtp-*`, `dspark-*`, `dflash-*`, EAGLE and `mmproj*` files; those weight
files do not automatically accompany a target selected by `--include`.
Inspect actual names with the dry-run rather than guessing a glob.

For example, this selects the pinned DeepSeek target and its drafter:

```sh
mise run archive:catalog -- show deepseek
mise run archive -- --dry-run \
  unsloth/DeepSeek-V4-Flash-0731-GGUF@fbbb5b93fb787c21338159b0af3318bb3f4d9768 \
  --include 'UD-Q2_K_XL/*' --include 'dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf'
hostlock shared --label "llmp: archive DeepSeek V4 Flash 0731" -- \
  mise run archive -- \
  unsloth/DeepSeek-V4-Flash-0731-GGUF@fbbb5b93fb787c21338159b0af3318bb3f4d9768 \
  --include 'UD-Q2_K_XL/*' --include 'dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf'
```

For a new archive, add `--init-root` to that first write after checking
the mount, or to the first migration instead. Prefer the existing Spark
copy for this particular model. Stage only checkpoints needed for current
work: one NAS read on `spark`, then `--peer spark-b --peer-address
10.100.208.2` over the direct link. Prepared execution artifacts stay on
the nodes; [they are not archived](model-archive.md#prepared-artifacts-are-not-archived).

## M2 and M3 checkpoints to preserve

The [M3 pins](experiments/fast-swap/pins.json) include exact selected files,
sizes and hashes. The [support matrix](model-support.md) distinguishes
fixtures, served models and comparators.

| Model | Repository and revision | Container / quantization; companion |
| --- | --- | --- |
| Qwen2.5-0.5B-Instruct fixture | `Qwen/Qwen2.5-0.5B-Instruct-GGUF@9217f5db79a29953eb74d5343926648285ec7e67` | GGUF F16: `qwen2.5-0.5b-instruct-fp16.gguf`; [fixture pins](experiments/first-slice/pins.json) |
| Qwen2.5-0.5B-Instruct EXL3 fixtures | `blockblockblock/Qwen2.5-0.5B-Instruct-exl3-4.0bpw@7009334dc77ad71f6a98d942d6df3ec64017f565`; `blockblockblock/Qwen2.5-0.5B-Instruct-exl3-4.5bpw@030d3a4173c5a3e27e3c27dd7f6ab0fc00636d00` | EXL3 safetensors, MCG, 4.0 and mixed 4.5 bpw; [fixture pins](experiments/exl3-reference/pins.json) |
| DeepSeek V4 Flash 0731 | `unsloth/DeepSeek-V4-Flash-0731-GGUF@fbbb5b93fb787c21338159b0af3318bb3f4d9768` | GGUF UD-Q2_K_XL, three shards; DSpark Q8_0 in the same revision |
| DeepSeek V4 community / Vision-Exp | `antirez/deepseek-v4-gguf@f71f23d552d664e523b422157b2befbf74040380` | Existing text target: `DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf` (IQ2_XXS plus Q2_K and F16). Vision-Exp's separate continued-trained target and encoder also belong to the plan; select their exact files at this pin |
| Qwen3.8 Flash Next native | `Mia-AiLab/Qwen3.8-Flash-Next-NVFP4@925d7be6c14c6c9442ef83e8f05b5a3c39304f69` | ModelOpt safetensors: NVFP4 experts/PLE, MXFP8 linears, other stored dtypes; MTP in the same checkpoint |
| Qwen3.8 Flash Next GGUF | `unsloth/Qwen3.8-Flash-Next-GGUF@38bb39ee97821de2c9009abb7e93950eec396e66` | UD-IQ3_XXS, three shards; actual mix includes IQ2_S, IQ3_S, IQ4_NL, Q6_K, Q8_0, BF16 and F32 |
| Qwen3.8 Flash Next MLX | `Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP@dadefa8066e3be900a0d148d0f5a2f4eb1cf6534` | Safetensors, affine 4-bit, group 32, MTP; TensorFold comparator and M3.6's selected MLX admission checkpoint |
| Qwen-Image-2.1 | `Qwen/Qwen-Image-2.1@790c92633540aa0cb11d9abf19eb46d861714758` | Diffusers composition, BF16; retain text encoder, transformer, VAE, tokenizers, processor and scheduler configuration |

## Approved M3.5 covering set and legacy fixtures

These are the 13 [covering-set picks](m35-families.md#summary), plus the
approved legacy and Bonsai fixtures. Gemma's existing served quantizations
are included here once. EXL3 rates are expanded in the next table.

| Model | Repository and revision | Selected format / quantization; companions |
| --- | --- | --- |
| Gemma 4 26B-A4B-it | `unsloth/gemma-4-26B-A4B-it-GGUF@c099eb48e663fd284577b04978a94ffccb261841` | GGUF UD-Q4_K_M; `mtp-gemma-4-26B-A4B-it.gguf` is the Q8_0 assistant; vision mmproj for media work; EXL3 below |
| gpt-oss-120b | `ggml-org/gpt-oss-120b-GGUF@238abdd290bb874b90a5da1b4549881b7d05c091` | GGUF MXFP4 plus its Q8_0 EAGLE3 head |
| MiMo-V2.6-Flash-RL | `benthecarman/MiMo-V2.6-Flash-RL-exl3@5ae87b830eb950c705967aea6130601a8e1f74a5` | EXL3 mixed 2.27 bpw (experts 2.0–2.5, attention 4.0, head 6.0), MTP and DFlash. Vision retained, audio omitted by this pack |
| NVIDIA Nemotron 3 Super 120B-A12B | `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4@ff433f5493e25d631c9f12b5d55c674229923d02` | NVFP4 safetensors, including the stored MTP |
| Kimi Linear 48B-A3B Instruct | `bartowski/moonshotai_Kimi-Linear-48B-A3B-Instruct-GGUF@228dbe476e5a02091624a19068f4c962caa8a1c5` | GGUF Q8_0 |
| Mistral Small 4 119B-2603 | `unsloth/Mistral-Small-4-119B-2603-GGUF@bd93c721735aa32c035c0f19e738cb3371fd56ff`; EXL3 below | Approved choice is GGUF or EXL3; exact pack remains to select. GGUF candidates UD-Q4_K_XL and MXFP4_MOE. Preserve vision projector; EAGLE source `mistralai/Mistral-Small-4-119B-2603-eagle@9a8ea22dca0161ff7af7879f4a0000f65314797d` |
| Llama 4 Scout 17B-16E Instruct | `unsloth/Llama-4-Scout-17B-16E-Instruct-GGUF@72a6853f56a66dc13a3a4b6bdc9cf7ee4c364b47` | GGUF UD-Q4_K_XL; vision mmproj |
| Llama 3.3 70B Instruct | `casperhansen/llama-3.3-70b-instruct-awq@64d255621f40b42adaf6d1f32a47e1d4534c0f14` | Safetensors AWQ INT4, group 128, zero point, GEMM layout |
| Muse Glimmer 30B | `meta-models/Muse-Glimmer-30B-GGUF@70bf1b61ac09f91b24d39038091b41c582bc5d7a` | GGUF Q4_K_M, DFlash GGUF and Q4_K_M mmproj; original assistant `meta-models/Muse-Glimmer-30B-assistant@e8192f3a8f617f74be2ce220360c89ef4789f39f` |
| Qwen3.8-27B | `ggml-org/Qwen3.8-27B-GGUF@71bc7b627595dc8a91039addd9c791ae548d6747`; `Qwen/Qwen3.8-27B-FP8@017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`; EXL3 below | GGUF Q4_K_M, MTP, DFlash2 and mmproj; FP8 E4M3 block 128×128 safetensors. Original DFlash2 `z-lab/Qwen3.8-27B-DFlash2@50307d4c4cde6860d4eee73e2547cd786fe8e8a4` |
| Gemma 4 31B-it | `unsloth/gemma-4-31B-it-GGUF@c1ac76e99d5513b141e8adde7288b85c3f9c32ec` | GGUF UD-Q4_K_XL; `mtp-gemma-4-31B-it.gguf` is the Q8_0 assistant; vision mmproj; EXL3 below |
| Ornith 1.5 35B-A3B | `ornith-ai/Ornith-1.5-35B-A3B-GGUF@12393612fd4f730ff5aadc23e9b8f9648aa49ceb` | GGUF Q4_K_M; vision mmproj; MTP/DFlash companions need exact file/repo selection |
| GLM-4.7-Flash | `unsloth/GLM-4.7-Flash-GGUF@0d32489ecb9db6d2a4fc93bd27ef01519f95474d` | GGUF Q8_0; stored MTP/sidecar selection to inspect |
| Gemma 3 4B-it QAT | `ggml-org/gemma-3-4b-it-qat-GGUF@bbcac0d065076c47042838c0675c602411b0dd4c` | `gemma-3-4b-it-qat-Q4_0.gguf`; include its SigLIP mmproj |
| Gemma 2 2B-it | `bartowski/gemma-2-2b-it-GGUF@855f67caed130e1befc571b52bd181be2e858883` | `gemma-2-2b-it-Q8_0.gguf` |
| Phi-3.5-mini-instruct | `bartowski/Phi-3.5-mini-instruct-GGUF@6d70da17e749a471ccb62ade694486011a75cda3` | `Phi-3.5-mini-instruct-Q8_0.gguf` |
| Command R7B 12-2024 | `bartowski/c4ai-command-r7b-12-2024-GGUF@bfc7a934c45cb839d84c8ca01d87f1cfa51aaa3f` | `c4ai-command-r7b-12-2024-Q8_0.gguf` |
| Bonsai-27B, 1-bit | `prism-ml/Bonsai-27B-gguf@f10afb355f104535e3e3e98cf7ab7795c72bd292` | `Bonsai-27B-Q1_0.gguf` plus `Bonsai-27B-dspark-Q4_1.gguf`; vision projector if supplied |
| Ternary Bonsai, 2-bit | Exact build and commit pending | Prefer an actually correct upstream Q2_0 pack; alternative `prism-ml/Ternary-Bonsai-2-27B-gguf` uses PQ2_0/PTQ1_0 and the PrismML fork. [Selection rule](plan.md#m35--model-families--parked) |

The Mistral-native NVFP4 layout was rejected for this bring-up; it is not
an initial-set pick. The [format survey](m35-families.md#format-coverage)
also mentions Gemma QAT/NVFP4, Scout NVFP4, gpt-oss native MXFP4/EXL3,
Muse NVFP4/EXL3 and Qwen27 NVFP4 alternatives. They are now included in the acquisition queue as additional format carriers,
while remaining outside the approved execution covering set.

### EXL3 rates and codebooks

The [EXL3 study](m35-families.md#exl3-in-depth) supplies these selections.
Each branch resolves to a separate immutable revision. Prefixes here are
historical observations, not full archive pins; the queue records their resolved full commits. Retain the checkpoint's
tokenizer, conversion metadata, vision and stored drafting weights.

| Model / repository | Initial rates and codebooks | Documented revision prefixes |
| --- | --- | --- |
| `turboderp/gemma-4-26B-A4B-it-exl3` | MCG 2.54, 3.10, 4.10 bpw | `5b468457`, `fcd233b0`, `71026023`, respectively |
| `turboderp/gemma-4-31b-it-exl3` | MUL1 2.50, 3.00, 4.00; MCG 3.00 for the codebook comparison | `d72250d4`, `5b465217`, `13e18523`; MCG `d27c062b` |
| `turboderp/Qwen3.8-27B-exl3` | MUL1 mixed `SC_3.00bpw_H4` and uniform 4.00 | `86b95530`, `113cf7ab` |
| `benthecarman/MiMo-V2.6-Flash-RL-exl3` | Mixed 2.27 bpw, above; codebook unverified | Full commit in the covering set |
| `turboderp/Qwen3.8-Flash-Next-exl3` | Stretch comparison: MUL1 3.05 bpw | `69e33439` |
| `turboderp/Mistral-Small-4-119B-2603-exl3` | All three candidates: 2.03, 3.03, 4.03 | `2a11afcd`, `0c2fc726`, `1aabb209` |

### Existing NAS MLX copies

All four were found in `/mnt/llm/huggingface-mlx/` on 2026-10-10.
Their `config.json` declares affine 4-bit, group 64; the Qwen3.6 config
also declares 8-bit router/gate overrides. These repositories have moved
from `Vontra` to `TensorFold`; Hub metadata retains the same full commits.
The queue migrates verified files under the current repository IDs; their
presence does not add execution qualification. The separate Flash Next group-32
checkpoint is in the M3 table above.

| Model / repository | Existing full commit | Role |
| --- | --- | --- |
| `Vontra/GLM-5.3-Flash-MLX-4bit-MTP` | `76add2a341a1cd90ad0e86bb69839ea9c35827c6` | M4 TensorFold comparator, group 64, MTP |
| `Vontra/Qwen3.8-27B-MLX-4bit` | `70ae7fac63274ff2eac54152031433374cb80f2f` | Dense Qwen affine comparison, group 64 |
| `Vontra/Qwen3.6-35B-A3B-MLX-4bit-MTP` | `81169a9bc511a27c1b4eedb77a2cd98ced431847` | Existing library checkpoint, group 64 plus declared 8-bit overrides, MTP |
| `Vontra/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit` | `8bbcb5b6a46a4409d47832b6fdc2fcc9183c83e8` | Existing library checkpoint, group 64; optional family comparator |

## Owner-named media, decision and research additions

These belong to the initial planning inventory even where importer and
execution work is pending. Preserve complete compositions and their media
processors/codecs; a language-only repack may omit the required encoders.
Sources: [media research](m35-families.md#generative-media-video-and-image),
[speech research](m35-families.md#text-to-speech-testbeds) and
[the current plan](plan.md#m35--model-families--parked).

| Model | Repository / documented revision | Format / selection |
| --- | --- | --- |
| Gemma 4 E4B-it, audio carrier | `google/gemma-4-E4B-it@ee0ef6023621cff504d758262d4e04895a5af4a2` | Original BF16 safetensors retaining audio and vision; expanded size/quant selections below |
| Clef | `Cloudflare/clef@2f3de3dd85f379784083b0814d997ab627200f0c` | Original BF16 safetensors including schema head and vision; `ggml-org/Clef-GGUF@63840a1a68cb7084c88610cffc328509356b04cb`, Q4_K_M, Q8_0 and BF16 projector |
| Clef-flash | `Cloudflare/clef-flash@17f0b0ad64efb65d273590632833508766b2aae6` | Original BF16 safetensors including schema head and vision; `ggml-org/Clef-Flash-GGUF@4a192915ef971886004b5b13294f2b4c7a7fc39d`, Q4_K_M, Q8_0 and BF16 projector |
| Clef-Omni | `Cloudflare/clef-omni@0db1cd2607d76a7bdb2a382f659e7b313079f84b` | Qwen3-Omni-30B-A3B thinker, schema head, vision/audio encoders; source dtype/quant to inspect. Its talker/code2wav are unused by the decision program |
| Ming-Image-0.1-Design | `inclusionAI/Ming-Image-0.1-Design@208087ada1486931692c1896f38d4cd16ff3df82` | Original composition: BF16 DiT and multimodal MoE conditioner, F32 Qwen2 connector, projection and RGBA VAE |
| MiniMax H3 | `MiniMaxAI/MiniMax-H3@42ed227ee7df40d41602854ae760620d6eb651fe` | Original BF16 composition including text encoder, video/audio VAEs and DiT. `Comfy-Org/MiniMax-H3@e5eb578a89295337b8ff433a035929ce0279e0b6`, pruned BF16 and FP8-scaled FL2VA/Ref2VA plus BF16 encoder and VAEs |
| Breeze-TTS-2 | `BreezeBlue/Breeze-TTS-2@3e28c5151381a722f1d8661b4118c298caa77aa4` | BF16, plus audio tokenizer/codec (retain component weights) |
| Kokoro-82M | `hexgrad/Kokoro-82M@f3ff3571791e39611d31c381e3a41a3af07b4987` | `kokoro-v1_0.pth` plus `voices/*.pt` (54 packs in the study). Explicit `--include` required for these pickle files; import never executes them |
| Qwen3-TTS 0.6B Base | `Qwen/Qwen3-TTS-12Hz-0.6B-Base@5d83992436eae1d760afd27aff78a71d676296fc` | Original talker/code predictor safetensors; confirm stored dtypes, pair with codec below |
| Qwen3-TTS 1.7B Base | `Qwen/Qwen3-TTS-12Hz-1.7B-Base@fd4b254389122332181a7c3db7f27e918eec64e3` | Original weights, voice cloning; confirm dtypes |
| Qwen3-TTS 0.6B CustomVoice | `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice@85e237c12c027371202489a0ec509ded67b5e4b5` | Original weights and supplied voices; confirm dtypes |
| Qwen3-TTS 1.7B CustomVoice | `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice@0c0e3051f131929182e2c023b9537f8b1c68adfe` | Original weights and supplied voices; confirm dtypes |
| Qwen3-TTS 1.7B VoiceDesign | `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign@5ecdb67327fd37bb2e042aab12ff7391903235d3` | Original weights; confirm dtypes |
| Qwen3-TTS codec | `Qwen/Qwen3-TTS-Tokenizer-12Hz@7dd38ad4e9bad454aae9cd937d0cd577604fe229` | Shared 12 Hz audio tokenizer/decoder checkpoint; confirm dtypes |
| EmbeddingGemma 2 | `google/embeddinggemma-2@914f7f89142e33e77833254d9c9b90c3cef7303b`, original safetensors | Owner-requested research candidate; retain text/code, vision and audio encoders, pooling and normalization metadata |

## M4 two-Spark targets and comparators

M4 re-pins at entry. The recipe repositories below are GitHub code, not
Hugging Face checkpoint IDs; do not pass a recipe name to `archive-model`.

| Model | Source / status | Planned format and companions |
| --- | --- | --- |
| GLM-5.3 Flash target | `brandonmusic/GLM-5.3-Flash-tr3-4bpw@5ab363a8dcf6405955fd5f99671e01a1c9fb124b`, fallback explicitly named by the pinned Mia recipe | EXL3 about 4 bpw, TP2; preserve vision, MTP and selected DFlash2. The newer dense-H3/6-bpw-drafter preset is a comparison candidate, not a replacement pin |
| GLM-5.3 Flash affine comparator | `TensorFold/GLM-5.3-Flash-MLX-4bit-MTP`, existing full pin above | MLX affine 4-bit, group 64, MTP; DFlash2 selected below |
| GLM-5.3 Flash NVFP4 comparator | `nvidia/GLM-5.3-Flash-NVFP4@da920bb0b9f4a06727223a349e55468e38352348`, used by [Kindling's recipe](m4-references.md) | NVFP4 TP2; separately record any dense-layer FP8/NVFP4 requantization |
| DeepSeek V4.1 Flash target | `Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw@64ba41b6c916a587db06eae2e19b7845f7be6e6b` ([provenance](licensing.md)) | EXL3 2.9 bpw, TP2; preserve vision and Engram tables; all safetensors selected, including stored drafting weights |

## Formats and quants the set must exercise

Container, stored encoding and a publisher's variant label are different.
For example, UD-Q2_K_XL is a mixture of types, not proof that every tensor
is Q2_K; safetensors can hold BF16, FP8, affine or EXL3 weights. The archive
detects actual headers after acquisition. [Format coverage](m35-families.md#format-coverage)
and [the format-layer design](engine-components.md#weight-and-state-formats)
define the intended scope; this table does not claim all paths run today.

| Coverage | Initial carriers / remaining selection |
| --- | --- |
| F32, F16/FP16, BF16 | Mixed checkpoint tensors, Qwen2 F16 fixture, Qwen-Image BF16, Ming connector F32 and other original compositions |
| GGUF legacy Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, IQ4_NL | Gemma3 QAT; Bonsai drafter; Gemma2/Phi/Command/Kimi/GLM and assistants; mixed GGUF fallbacks (not standalone downloads of every fallback type) |
| GGUF K-quants Q2_K, Q3_K, Q4_K, Q5_K, Q6_K | DeepSeek and approved mixed GGUFs; optional Step's UD-Q3_K_XL. Inventory actual tensor types for remaining gaps |
| GGUF I-quants IQ1–IQ4 and Unsloth dynamic mixes | DeepSeek IQ2_XXS/IQ2_XS/IQ3_XXS and Qwen IQ2_S/IQ3_S/IQ4_NL; wider IQ1/IQ4 coverage needs additional variant selections |
| GGUF Q1_0, Q2_0, PQ2_0, PTQ1_0 | Bonsai, Ternary-Bonsai and Ternary-Bonsai-2 selected, with available DSpark and BF16 projectors |
| MXFP4 | gpt-oss GGUF and DeepSeek's mixed tensors; Mistral MXFP4_MOE candidate |
| ModelOpt NVFP4 and MXFP8 | Qwen Flash Next and Nemotron Super; GLM NVFP4 comparator selected |
| FP8 block 128×128 | Qwen3.8-27B-FP8 |
| FP8 per-tensor / per-channel | Llama 3.3 FP8-dynamic format comparator selected |
| AWQ INT4, group 128 | Llama 3.3 70B AWQ |
| GPTQ INT4/INT8, act-order and packing variants | Llama 3.3 GPTQ selected; INT8/packing alternatives remain later expansion |
| MLX affine 4-bit, group 32 and 64 | Flash Next (32), four older NAS copies (64; Qwen3.6 includes declared 8-bit overrides) |
| EXL3 MCG and MUL1, uniform and mixed bpw | M2 fixtures and selected Gemma/Qwen/MiMo rates; M4 GLM/DeepSeek; 3INST is described in research, no selected carrier |
| Diffusers / composed safetensors | Qwen-Image, Ming, MiniMax H3 and their complete component sets |
| PyTorch pickle weights | Kokoro `.pth` model and `.pt` voices, explicitly selected for storage; native safe import remains required |

Lossy KV/state encodings and locally prepared Q4_1 draft-head experiments
are execution/import choices, not Hugging Face source-download entries.
The initial set supplies representative quantizations; the product goal
remains every quant of supported architectures, with later library expansion.

## Optional candidates included in acquisition

The owner included these candidates in the first acquisition batch. Pins
and alternatives are in
[optional checkpoints](m35-families.md#optional-checkpoints) and
[media research](m35-families.md#generative-media-video-and-image).

| Candidate | Documented format / quant |
| --- | --- |
| MiniMax M2.7 | `unsloth/MiniMax-M2.7-GGUF@d2a05ccf69491b03db0cc40b335aec14bdaf7198`, UD-IQ3_XXS; DFlash/EAGLE3 |
| Laguna S 2.1 | `unsloth/Laguna-S-2.1-GGUF@750f92f90cf54159c4d7a610cb7b3e74498e75c6`, UD-Q4_K_XL; DFlash |
| Step 3.7 Flash | `unsloth/Step-3.7-Flash-GGUF@8dd62fe8a98f3c8994160d0c521542560b558f37`, UD-Q3_K_XL and BF16 projector; `Hikari07jp/Step-3.7-Flash-MTP-draft@6ecd45974127ea7fa6c7de3584c9412c96e13e32` |
| Ling 3.0 flash | `bartowski/Ling-3.0-flash-GGUF@7e1a1243566501d52af98e9aa7a7a61986279c6d`, all Q4_K_M shards |
| NVIDIA Nemotron 3.5 Lightning 30B-A3B | NVIDIA NVFP4 W4A16 (`bee7596271d1495f6992ae224aefde4410e816b8`), DSpark (`8a0177116d138011e63103110f136ec0ca09ebbf`); repository IDs recorded in the frozen queue; existing MLX copy above |
| Cohere North Mini Code 1.0 | `unsloth/North-Mini-Code-1.0-GGUF@e306bb4bf0df610f5471d97a01de2b6e0b24d356`, UD-Q4_K_XL; `CohereLabs/North-Mini-Code-1.0-eagle@8c7fcb575f107e9968b61cc93a756e6fc2c86713` |
| Mistral Medium 3.5 128B | `nvidia/Mistral-Medium-3.5-128B-NVFP4@b8c66d2098edd8c9c26bde2b2ff41b5967e111ae`, plus EAGLE |
| gpt-oss-20b | `ggml-org/gpt-oss-20b-GGUF@ef9b12f2ff56c69cf32153a02784e7a3c88bf524`, MXFP4 and EAGLE3 |
| LTX-2.5 video fallback | `Lightricks/LTX-2.5@5e6e71018ee1756ed329b697a7b4aedc934dfce9`, BF16 or official NVFP4; Gemma encoder |
| Ideogram 4 image alternative | `ideogram-ai/ideogram-4-fp8@ee79a7237b519f1402ceacf952f30c8a31ec5073`, FP8/NF4; not an added NF4 implementation commitment |
| FLUX.2-klein-9B image alternative | `black-forest-labs/FLUX.2-klein-9B@92196c8e11f7b6cf2b7493e037d8c5345c559216`, Diffusers components; skip duplicate root checkpoint |
| Older alternative carriers | Gemma2 9B, Mistral-7B-Instruct-v0.3, Llama3.2 1B GGUF; now included as Q8_0 alternatives to the four pinned legacy fixtures |

The research's excluded models and formats (including EXL2, bitsandbytes,
HQQ and dropped legacy formats) are not initial-set acquisitions. Models
merely cited for popularity or architectural comparison are not planned
downloads. Larger post-M4 candidates, such as MiniMax M3 and Tencent Hy3,
wait for a deliberate library expansion.

## Gemma 4 complete size coverage

All five published sizes are selected: E2B, E4B, 12B, 26B-A4B and 31B.
For each, archive Google's base and instruction-tuned BF16 safetensors,
the instruction-tuned BF16 GGUF, the original assistant and GGUF assistant,
and the BF16 projector. Preserve audio and vision assets wherever supplied.
Full precision provides the highest-fidelity source copy; smaller choices
are candidates for GPU use, with no measured quality ranking implied.

The sizes below are decimal GB of the selected GGUF target and projector
from pinned Hub file metadata. A 12/16 GB GPU comparison uses 12/16 GiB
of device capacity. KV cache, workspace, activations and runtime allocations
must also fit; these are size-based candidates, not tested fit guarantees.
Assistants are archived but are excluded from these totals and need their
own budget if loaded. Use the source checkpoint's supported runtime; archive
presence does not imply llmpalooza execution support.

| Size | Full-precision copy | 16 GiB candidate, target + BF16 projector | 12 GiB candidate, target + BF16 projector |
| --- | --- | --- | --- |
| E2B | BF16 safetensors and GGUF | BF16, about 10.30 GB (9.59 GiB) | BF16 at short contexts; Q8_0, about 6.04 GB (5.62 GiB), provides more headroom |
| E4B | BF16 safetensors and GGUF | Q8_0, about 9.19 GB (8.56 GiB) | Same Q8_0 candidate; Q4_K_M also archived |
| 12B | BF16 safetensors and GGUF | Q8_0, about 12.84 GB (11.96 GiB) | Q6_K, about 9.96 GB (9.28 GiB); Q4_K_M also archived |
| 26B-A4B | BF16 safetensors and sharded GGUF | UD-IQ4_XS, about 14.79 GB (13.78 GiB), target without assistant; UD-Q3_K_XL offers more headroom | UD-IQ2_M, about 10.02 GB (9.33 GiB), **text target only**; projector adds about 1.20 GB. EXL3 MCG 2.10 bpw has about 10.28 GB of weights including stored vision as another candidate |
| 31B | BF16 safetensors and sharded GGUF | Q3_K_S, about 14.41 GB (13.42 GiB), target without assistant | UD-IQ2_XXS, about 9.74 GB (9.07 GiB) |

The 26B-A4B UD-IQ2_M with projector is close to the 12 GiB budget, so prefer
text-only or a measured EXL3 configuration there. The archived NVFP4 26B/31B
files are about 18.8/32.6 GB and are not substitutes for the small-card picks.

Also acquire official QAT Q4_0 GGUF and its unquantized source for every size,
Unsloth's corresponding QAT repack with assistant, and official W4A16
compressed-tensors for E2B/E4B/12B/31B (no 26B-A4B W4A16 repository was
available). The existing 26B UD-Q4_K_M and 31B UD-Q4_K_XL remain included,
as do Q8_0 and the 26B MXFP4_MOE comparison. EXL3 adds 26B MCG 2.10 bpw
and 31B MUL1 2.00 bpw to all rates in the earlier EXL3 table. Muse and
gpt-oss EXL3 branches, and the format survey's NVFP4/native MXFP4 choices,
are in the exact queue too.

## Resolved sources and acquisition supervision

The complete immutable transfer specification, selected file audit, live
state and per-revision logs are under:

`/home/pmeenan/.local/share/llmp/archive-downloads/2026-10-10-library/`

`queue.json` records all 149 repository/commit pairs and include/exclude
patterns; `selection-audit.json` records every selected filename and per-revision byte total.
`state.json` records completed, running, interrupted, failed and access-waiting
tasks. These operational files stay outside Git. Completed downloads are
recorded separately by the archive's hashed manifests and catalog.

The persistent user service `llmp-archive-library.service` uses a frozen
copy of the existing archive tools. It resumes interrupted transfers,
uses two file workers per revision, and takes a shared hostlock for transfers
and hashing. Spark source copies hold the workstation shared hostlock and use the
installed `spark-job --shared` supervisor to write onto the same NAS share. A
failed reused copy falls back to the pinned Hub download where available;
original Spark files remain in place. Three actual failures end that task,
with errors retained in state; deliberate stops do not consume retries.
Access-denied tasks are deferred and probed every ten minutes while the
remaining library proceeds. The service remains waiting if access is still
missing after other tasks finish; a failed task is never reported complete.

The read-only progress view shows completed revisions and bytes, the current
checkpoint's received bytes, recorded failures/access waits and its latest
log message. Current-file bytes may still need verification. It refreshes
every ten seconds; Ctrl+C exits the view without stopping downloads:

```sh
llmp-archive-progress --watch
```

Omit `--watch` for one snapshot. After a forced stop, a partial download can
wait for its previous writer's fifteen-minute claim to expire before resuming.
The view shows the remaining claim time when that is the current wait.

Use these light commands to inspect or control the batch:

```sh
systemctl --user status llmp-archive-library --no-pager
systemctl --user stop llmp-archive-library
systemctl --user start llmp-archive-library
mise run archive:catalog -- show
```

Five selected repositories initially returned HTTP 403 for the configured
token. The owner accepted their terms; four passed a subsequent one-byte
weight-access check on 2026-10-10. MiniMax DFlash remains pending approval.
The affected Hub pages are:
[LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5),
[Ideogram 4](https://huggingface.co/ideogram-ai/ideogram-4-fp8),
[FLUX.2 Klein](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B),
[MiniMax DFlash](https://huggingface.co/z-lab/MiniMax-M2.7-DFlash), and
[Scout NVFP4](https://huggingface.co/nvidia/Llama-4-Scout-17B-16E-Instruct-NVFP4).

Two unavailable Hub sources need explicit provenance treatment:

- `Mia-AiLab/Qwen3.8-Flash-Next-NVFP4@925d7be6c14c6c9442ef83e8f05b5a3c39304f69`
  returned 404. Recover the 50 recorded files from the Spark checkpoint,
  using the exact sizes and SHA-256/Git blob IDs in the existing M3 pins.
  The revision manifest identifies historical project pins as the inventory
  source and retains their digest; it does not claim a current full Hub
  inventory. A similarly named newer mirror has different weight hashes
  and is not substituted.
- The M4 GLM Mia checkpoint is unavailable. The pinned
  [Mia recipe](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks/blob/c1b7d4c9f98c16af65640fef05fa493bbde52ccf/README.md)
  explicitly provides `brandonmusic/GLM-5.3-Flash-tr3-4bpw` as its fallback;
  acquire `5ab363a8dcf6405955fd5f99671e01a1c9fb124b`, with
  `incoai/GLM-5.3-Flash-DFlash2@bf582e4eacc1810f76656d1811693ff6c6737d2a`.
  The NVFP4 comparator is
  `nvidia/GLM-5.3-Flash-NVFP4@da920bb0b9f4a06727223a349e55468e38352348`.
