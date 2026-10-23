// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The approved checkpoints (runtime/approvals.h), a lock file in C++.
//
// Source receipts are the approved captured ones, as the importer verified
// and declared them at import (admission compares, never re-hashes them);
// inventories digest every tensor's complete native description
// (component_admission.cc Inventory), each re-pinned (D-118) from the
// published artifact that the earlier name/encoding/shape digests admitted
// (source pins: docs/experiments/m36-native-v1-admission/README.md); the
// spec digests are those of the v1 artifacts `llmp import` published from
// the staged, archive-verified sources and that executed bit for bit as
// their v0 artifacts did (docs/experiments/m36-numerical-audit/
// v1-equivalence.md and qwen-modelopt-v1.md; owner, 2026-10-15: serving is
// v1-only). The applied folds' and relayouts' records digest their tensors'
// sorted names (component_admission.cc Transforms). The envelopes,
// policies and pairings are the ones each family qualified in M3 and M3.6.
// Change an entry only with the evidence that approves it.

#include "runtime/approvals.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <span>
#include <string_view>

#include "base/sha256.h"
#include "spec/mappings/qwen38.h"

namespace llmp::runtime {
namespace {

constexpr std::array kGemma2Sources{
    SourceReceipt{"gemma-2-2b-it-Q8_0.gguf",
                  "2d448a9aab894b8e8e18168cf3f490cb9f65632222f29f93514ac9ecc754debe",
                  2784495456ULL},
};

constexpr std::array kGemma3Sources{
    SourceReceipt{"gemma-3-4b-it-qat-Q4_0.gguf",
                  "ee91c3e7a4ab95d8c95672f9fcb58bf236b257e9f217966bcf53a5a6df4ab49a",
                  2526080992ULL},
};

constexpr std::array kGemma26Sources{
    SourceReceipt{"gemma-4-26B-A4B-it-UD-Q4_K_M.gguf",
                  "f2c28b3dc4776931ac6f879e11f203dec637ea0f14267a86ec8f6165f63f293f",
                  16947541728ULL},
};

constexpr std::array kGemma31Sources{
    SourceReceipt{"gemma-4-31B-it-UD-Q4_K_XL.gguf",
                  "9e92cb6236044c6a9870af406029c74a76e0571c157a6f95df724dcc8c7a1575",
                  18822970304ULL},
};

constexpr std::array kGemma26AssistantSources{
    SourceReceipt{"mtp-gemma-4-26B-A4B-it.gguf",
                  "6326fb9f5e487aa8dcdd313a091e3c67724cb2a666ec3b7d2895b5b26d93ed1b", 461766816ULL},
};

constexpr std::array kGemma31AssistantSources{
    SourceReceipt{"mtp-gemma-4-31B-it.gguf",
                  "5ae8b0117bed601e8924c6305bd5b0585de361d51f0e77091bcb4252cf1f27de", 514687104ULL},
};

constexpr std::array kQwen2Fp16Sources{
    SourceReceipt{"qwen2.5-0.5b-instruct-fp16.gguf",
                  "8e0ae26000627ed62de0e78e41860af70094558b9d2913385c842a6aa06cf3fc",
                  1266425696ULL},
};

constexpr std::array kQwen2Trellis40Sources{
    SourceReceipt{"config.json", "966657b7e8c331ea16127a868eb6d6ff16ad0b16a5ee39d26279e12de0a3205c",
                  996ULL},
    SourceReceipt{"generation_config.json",
                  "e558847a8b4402616f1273797b015104dc266fe4b520056fca88823ba8f8ebe6", 242ULL},
    SourceReceipt{"model.safetensors",
                  "66b61bd1ca3ebb0fc1cea29148eda8bdaca45d232c6a7e319c6d2835b65c066b", 588951098ULL},
    SourceReceipt{"tokenizer.json",
                  "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539", 7031645ULL},
    SourceReceipt{"tokenizer_config.json",
                  "5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583", 7305ULL},
};

constexpr std::array kQwen2Trellis45Sources{
    SourceReceipt{"config.json", "dcd945f38ba6e7835c8c281a45c03d9a20759d03afe5761d09def8c62aa11f3c",
                  923ULL},
    SourceReceipt{"generation_config.json",
                  "e558847a8b4402616f1273797b015104dc266fe4b520056fca88823ba8f8ebe6", 242ULL},
    SourceReceipt{"model.safetensors",
                  "94c45f9838a56fbf23f43d0a04c47e187d840dd97f17b4a027a69a20339762bc", 611257914ULL},
    SourceReceipt{"tokenizer.json",
                  "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539", 7031645ULL},
    SourceReceipt{"tokenizer_config.json",
                  "5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583", 7305ULL},
};

constexpr std::array kLlama32Q8Sources{
    SourceReceipt{"Llama-3.2-1B-Instruct-Q8_0.gguf",
                  "432f310a77f4650a88d0fd59ecdd7cebed8d684bafea53cbff0473542964f0c3",
                  1321083008ULL},
};

constexpr std::array kLlama32Q4kmSources{
    SourceReceipt{"Llama-3.2-1B-Instruct-Q4_K_M.gguf",
                  "6f85a640a97cf2bf5b8e764087b1e83da0fdb51d7c9fab7d0fece9385611df83", 807694464ULL},
};

constexpr std::array kQwen38GgufSources{
    SourceReceipt{"Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf",
                  "268f81fdedf3149a538f252308927a4d5d1f6e062c178568a51e3b519744f8a8", 10946624ULL},
    SourceReceipt{"Qwen3.8-Flash-Next-UD-IQ3_XXS-00002-of-00003.gguf",
                  "cfe600b236b88c7fad1613a5ca5e83b9f2beb63cbd44c32b2be50a44747c695f",
                  49567921344ULL},
    SourceReceipt{"Qwen3.8-Flash-Next-UD-IQ3_XXS-00003-of-00003.gguf",
                  "f1912ba34c79427d2295a58dcb2b732b5931af5bef7a373c60557a57d9ee7250",
                  32382955968ULL},
};

constexpr std::array kQwen38ModelOptSources{
    SourceReceipt{"config.json", "2d655dfa625c3c018f9685031367bc83f8532d3faf3333d7c838b2bdc13b8a7e",
                  109897ULL},
    SourceReceipt{"hf_quant_config.json",
                  "d60c54caa65317fd8f1338bdeb81b9014512ac8f21ec18da22541cb88c6784b7", 99088ULL},
    SourceReceipt{"model-00001-of-00034.safetensors",
                  "3f870511a20ad170c997517765f03d475bcd364ae1a6011491085cab620d3c39",
                  2841505880ULL},
    SourceReceipt{"model-00002-of-00034.safetensors",
                  "fe8f4bdbce28f4911f10460b75581b6c88a8ade9b836d2f1ec22c6bee6f976a8",
                  1575009800ULL},
    SourceReceipt{"model-00003-of-00034.safetensors",
                  "2b7d1312a660cf17bb05c34e1ac79570820f86fb1e1115108ee40562e48acb5e",
                  1575009808ULL},
    SourceReceipt{"model-00004-of-00034.safetensors",
                  "25ba4508c209a5cc54c04edb1edb8d0a0b5efa80e5f99637e2263fa78d6a25e8",
                  1575009808ULL},
    SourceReceipt{"model-00005-of-00034.safetensors",
                  "a9848067bb49f44b8339bcdca9d60ca74d801a6df874c979c6f901c8da6ff545",
                  1575009808ULL},
    SourceReceipt{"model-00006-of-00034.safetensors",
                  "66fc496b58675a90b94d9f36765fe67692dd8a4c966ec0692e076d751f835c8d",
                  1575009808ULL},
    SourceReceipt{"model-00007-of-00034.safetensors",
                  "68f89e84be13069bd9f2928b44a9f9638d628fd62be5abb6a4fc956c1af25ea9",
                  1575009808ULL},
    SourceReceipt{"model-00008-of-00034.safetensors",
                  "b8bdaa0a7b027dc0fa38c2e43bb31e8ddf9ba9288d3d0f28c422e6947241d96e",
                  1575009808ULL},
    SourceReceipt{"model-00009-of-00034.safetensors",
                  "6c5a85990b16a93f0e7e28fb76f56713d5cbc05bef494bb15360188f575746e9",
                  1575009808ULL},
    SourceReceipt{"model-00010-of-00034.safetensors",
                  "8734a1fc3b5857391fbf2252d4305391a9f77894b70577949f472fc75386582f",
                  1575009808ULL},
    SourceReceipt{"model-00011-of-00034.safetensors",
                  "29792378d43c9f70d9f3fbdd07d0ee24b38f81fe27aff25a02ba0d3cde8df13c",
                  1575009808ULL},
    SourceReceipt{"model-00012-of-00034.safetensors",
                  "b23e615ebefff4753d982d9ec6a8be6ad57353a8900146a3ec4e217f81c10e61",
                  1575009808ULL},
    SourceReceipt{"model-00013-of-00034.safetensors",
                  "9da7b4570fc3f13cb781e5aca5e79d0edfc10be5555a18bc7bda8f47459d5acf",
                  1575009808ULL},
    SourceReceipt{"model-00014-of-00034.safetensors",
                  "53e770d0bdf6261e30abaae2477aa0b50f6686511e0240514ded0fd65edbfeba",
                  1575009808ULL},
    SourceReceipt{"model-00015-of-00034.safetensors",
                  "03440de56d127ff58e72963aafd0de0574b825667330c50b4441bb300dae9f06",
                  1575009816ULL},
    SourceReceipt{"model-00016-of-00034.safetensors",
                  "ce2e0474ccb34c56003a61608b46d3a8f668c218a9993c1f861239df83afe77b",
                  1575009824ULL},
    SourceReceipt{"model-00017-of-00034.safetensors",
                  "57f84845e0b8bcab738e70f0f880e941f1baa8da14663293d7d40481288c73a0",
                  1575009824ULL},
    SourceReceipt{"model-00018-of-00034.safetensors",
                  "3477df0be60910576fc5a1f8159d8dbbf7847a03872b9cb53664808e1d5dfac8",
                  1575009824ULL},
    SourceReceipt{"model-00019-of-00034.safetensors",
                  "43c22213b2fe0990c7fc2aa54c0ea65062eb93253b4ea419914d1a655e64b23a",
                  2438722732ULL},
    SourceReceipt{"model-00020-of-00034.safetensors",
                  "b7143694bf8698c5b704ae9b3d492a0e9b8ef57bdda8ae6f3f29f264c183ad17",
                  5133924592ULL},
    SourceReceipt{"model-00021-of-00034.safetensors",
                  "522bbbc894183865726e06ca44021d275d03cc0e8dcd60a2c7074d870e811d37",
                  5194891408ULL},
    SourceReceipt{"model-00022-of-00034.safetensors",
                  "5a4789f6da6b386b09cb9093b0315fe12a816142958e4d8a4ae83e4fbaa066e7",
                  5133933936ULL},
    SourceReceipt{"model-00023-of-00034.safetensors",
                  "cd0aaef1c8688cb4e35b488048ede2a6b653e1d558b20d4657567df24dc8be13",
                  5194908864ULL},
    SourceReceipt{"model-00024-of-00034.safetensors",
                  "d07198edd38ae8740d556487729ae87be1e21e31a88cf6624c1b3f9d0fc6de76",
                  5188755712ULL},
    SourceReceipt{"model-00025-of-00034.safetensors",
                  "c7d5e642968679ca14d07bdb1e8c2187c59898611d05a6909c209738642c9519",
                  5133940040ULL},
    SourceReceipt{"model-00026-of-00034.safetensors",
                  "47a7c7a70188cb3dc318071a5278260c4b05d7ce1b69c3b99e7bf1c0e42c353b",
                  5194907416ULL},
    SourceReceipt{"model-00027-of-00034.safetensors",
                  "a81fedf160434806e81f36731c853fe4e3f973dea2f0817043987f32b4915a54",
                  5133940216ULL},
    SourceReceipt{"model-00028-of-00034.safetensors",
                  "91c5d479fe4de1035c430ad7fd3de9d15214a46fd786ca7909d1f2fa40ef603a",
                  5194907672ULL},
    SourceReceipt{"model-00029-of-00034.safetensors",
                  "f31c6f61968f6f9e85d469f31c55c82ab6c5f2bc54c1d7a0cd6b23ac45f0ce15",
                  5133939992ULL},
    SourceReceipt{"model-00030-of-00034.safetensors",
                  "df8627f5b8b0711b2ebf415c1f2549ef0b50f964e0babb560469b191b33c6a14",
                  5194908960ULL},
    SourceReceipt{"model-00031-of-00034.safetensors",
                  "654c8f9ae1ab5de393cbcdd46a13bd0d3d1c1005e575879e0e660d40ad9ef425",
                  5188755536ULL},
    SourceReceipt{"model-00032-of-00034.safetensors",
                  "7de2e680802ad92a0e019f875ebc07f2d5d82c8bca1c64b0f817d547bc3c9de7",
                  5133940208ULL},
    SourceReceipt{"model-00033-of-00034.safetensors",
                  "0fbcdf16c13cd830447efdcd3be19b82ec0e76b7f57f0e96952c1f2b62002b11",
                  5021547836ULL},
    SourceReceipt{"model-00034-of-00034.safetensors",
                  "fe0a0a227e39c4d02dc66c726d3e1ad87ef082e90a6b023ac90b2bb301672203",
                  1597270800ULL},
    SourceReceipt{"model-inputscales.safetensors",
                  "49c7d04606ff7c8f103b4d1fc9742b44649a2937b25ecc896a5450ab1020f308", 9669936ULL},
    SourceReceipt{"model.safetensors.index.json",
                  "435eef76fc10fc6e932a208a1f85a86bc9c7ffc389b27b34ba83bb6a5d0371e9", 33074286ULL},
};

// The native importer's MTP drafter of the same checkpoint with Mia's
// 47,172-token draft head: its configuration documents, the shards holding
// mtp.* and lm_head.weight, and the head's list.
constexpr std::array kQwen38MtpSources{
    SourceReceipt{"config.json", "2d655dfa625c3c018f9685031367bc83f8532d3faf3333d7c838b2bdc13b8a7e",
                  109897ULL},
    SourceReceipt{"hf_quant_config.json",
                  "d60c54caa65317fd8f1338bdeb81b9014512ac8f21ec18da22541cb88c6784b7", 99088ULL},
    SourceReceipt{"mia-47172.txt",
                  "ee819d2560b52ba1351acdd1c4a0244b77bb60694a7660c6363097a482b0afb5", 274622ULL},
    SourceReceipt{"model-00033-of-00034.safetensors",
                  "0fbcdf16c13cd830447efdcd3be19b82ec0e76b7f57f0e96952c1f2b62002b11",
                  5021547836ULL},
    SourceReceipt{"model-00034-of-00034.safetensors",
                  "fe0a0a227e39c4d02dc66c726d3e1ad87ef082e90a6b023ac90b2bb301672203",
                  1597270800ULL},
    SourceReceipt{"model.safetensors.index.json",
                  "435eef76fc10fc6e932a208a1f85a86bc9c7ffc389b27b34ba83bb6a5d0371e9", 33074286ULL},
    SourceReceipt{"chat_template.jinja",
                  "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041", 8952ULL},
    SourceReceipt{"generation_config.json",
                  "735ec63e625c2086fbec18327233a03a7b060fddabfd7915af4e6fdcec3187a2", 2097ULL},
    SourceReceipt{"tokenizer.json",
                  "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3", 12809320ULL},
    SourceReceipt{"tokenizer_config.json",
                  "b11349aafa7cdc6a320767cf7ceb29ed82f7eda5d65e8e0819e76f0ce947bf27", 17928ULL},
};

constexpr std::array kQwen38MlxSources{
    SourceReceipt{"config.json", "4efd5f41c47e8de9f55c678df4a237cbbc78e4d517dad42a906889f7d94f79a2",
                  5361ULL},
    SourceReceipt{"model-00001-of-00022.safetensors",
                  "308f20b7e35a3e525f36eca0df286f6d1043cfeb621e7bbcf372a834c99cd66d",
                  5283195887ULL},
    SourceReceipt{"model-00002-of-00022.safetensors",
                  "8af1a616f2f39dbfe6ee92e959afb2414623fcb299aff1ced55984bfe810d608",
                  5250035137ULL},
    SourceReceipt{"model-00003-of-00022.safetensors",
                  "519a0472252ac9a51af020686f462aa6be82ce655f948fd5f054d70949de17a6",
                  5250035154ULL},
    SourceReceipt{"model-00004-of-00022.safetensors",
                  "b1ad7e951969eb8dc4e92762dc2e50bd47353c95990ae207db5da393d441400f",
                  5250035160ULL},
    SourceReceipt{"model-00005-of-00022.safetensors",
                  "9ca797304c563bb65f2356b67d444511b11744bc5b5c8ec8345ae7d97d3e251c",
                  5250035164ULL},
    SourceReceipt{"model-00006-of-00022.safetensors",
                  "8aab4c1dce7d8a0af645426b66d1caf30245d422d4c6b7029583959d675a1827",
                  5250035184ULL},
    SourceReceipt{"model-00007-of-00022.safetensors",
                  "7908da37b2e3fa7f0de8b9734dad0759be7dc445716f3acd59c20c4842638db9",
                  5067850140ULL},
    SourceReceipt{"model-00008-of-00022.safetensors",
                  "85beff20f250bfe63e95f12c0460512aab92bcde8f7408fe9c20e3de0321bc1d",
                  5337131618ULL},
    SourceReceipt{"model-00009-of-00022.safetensors",
                  "f7c03ae640e9bf0f7813ef2b2980afa83a883c02b3988e814e9921055fc08a28",
                  5337131558ULL},
    SourceReceipt{"model-00010-of-00022.safetensors",
                  "6897f52227533e720b243d6aef24c1293ad1eb02a1bb2ec6a6e4dc48d56d0c8f",
                  4970130103ULL},
    SourceReceipt{"model-00011-of-00022.safetensors",
                  "905d27f9db887284289a043f9638020258c5f90d52122cdce2e7aef941a431f2",
                  5341257482ULL},
    SourceReceipt{"model-00012-of-00022.safetensors",
                  "c4d2bb76fa4613f7386ddaa7045608c320f9736623211a814aa28fb370530dec",
                  4967968541ULL},
    SourceReceipt{"model-00013-of-00022.safetensors",
                  "1f20a5e1f2256b2fb6a78fef52f9e66daa970f4318ab1be3abb32a1effedbbf3",
                  5337131651ULL},
    SourceReceipt{"model-00014-of-00022.safetensors",
                  "860d12691792df5f08c9dfe245dc356773741f1edcc1fd430672ef06ac387635",
                  5337131799ULL},
    SourceReceipt{"model-00015-of-00022.safetensors",
                  "259fb7577b8248f6dcbe6a932fe5d2f1c8869eda761ef3394fe6ad06065f216c",
                  4970130195ULL},
    SourceReceipt{"model-00016-of-00022.safetensors",
                  "e931785f9ff4e0a567bf8c06df37639be46b4dae156e0f8b39b9d32a03d75792",
                  5341257440ULL},
    SourceReceipt{"model-00017-of-00022.safetensors",
                  "4f4610dc4a862bf9d93d70cfd0ec037e8e7943a734794d34eb5c0d748b74747a",
                  4967968481ULL},
    SourceReceipt{"model-00018-of-00022.safetensors",
                  "afdb4828920186600f9d61537d1465979fb36692f4eddb8c5a961a6b72964b77",
                  5337131695ULL},
    SourceReceipt{"model-00019-of-00022.safetensors",
                  "869d34c6e8f7db79bd4e54b9833d5b0db45eefac860c9bd64fbd312bf862a667",
                  5337131801ULL},
    SourceReceipt{"model-00020-of-00022.safetensors",
                  "5ca2f994fec1aa06fefef6646b9f072cf70f7f73aa8dab4c7c1a9a424afb4a52",
                  4970130135ULL},
    SourceReceipt{"model-00021-of-00022.safetensors",
                  "58def6762f29cc798437e5649a56ee75d053faca69138a2454dbdb34bb8a37c3",
                  5341257436ULL},
    SourceReceipt{"model-00022-of-00022.safetensors",
                  "30a3c82b573d813a1d2349f1c05c308c0d415f3ca1bb67caa0bfaad8a6afa7a3",
                  3715570974ULL},
    SourceReceipt{"model.safetensors.index.json",
                  "8848503860c7da8619fdc8d5e65201868f8f1d21d69d2e21166c95522379ff77", 420813ULL},
};

constexpr std::array kDsv4Sources{
    SourceReceipt{"DeepSeek-V4-Flash-0731-UD-Q2_K_XL-00001-of-00003.gguf",
                  "de65c5bb660817b95cc281bf54935f9d1bcc3b22535f2eae63b35a14c7c5724b", 5257664ULL},
    SourceReceipt{"DeepSeek-V4-Flash-0731-UD-Q2_K_XL-00002-of-00003.gguf",
                  "c8728a298862cd8736782e2e4056e5e3e03c415ba552a38173264494e41ef327",
                  49437013568ULL},
    SourceReceipt{"DeepSeek-V4-Flash-0731-UD-Q2_K_XL-00003-of-00003.gguf",
                  "a4bae0788c025c1b2388754592c98fefa1372981748ac9875039bf6a30e6eeb4",
                  47390237120ULL},
};

constexpr std::array kDsv4Iq2xxsSources{
    SourceReceipt{"DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf",
                  "ca22ae2f838e14077c22bc1c1417b71b45b5e5a3687bd96c2ac6e17fdb6261c0",
                  86720111488ULL},
};

constexpr std::array kDsparkSources{
    SourceReceipt{"dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf",
                  "2c7ac54b0b64a99df1f139a9f1371a00198265e1d6a614b77597d20a655a4249",
                  10896057440ULL},
};

// Qwen3.8's ModelOpt and MLX imports repack GDN's value heads tiled and
// apply the parameter folds (by role, D-108 as amended: the zero-centered
// norms' 1 + w and A's -exp(A_log) in F32). The ModelOpt import's
// relayouts from split parts besides: each layer's routed experts' gate-up
// and down tables (both parts) into scale atoms, the n-gram table's rows
// interleaved with its scale.
constexpr std::array<std::string_view, 3> kQwenNativeFolds{"gdn.value_heads_tiled", "norm.plus_one",
                                                           "ssm.a_neg_exp"};
constexpr std::array kModelOptFoldTensors{
    ApprovedTransform{"gdn.value_heads_tiled", 468,
                      "29a95338f00eb7e661ff6ce8f8439d57a9843607995f821477c628fe13855286"},
    ApprovedTransform{"norm.plus_one", 148,
                      "5da54d6f36f3f22b2270222f1c9a648e93d7a6ec1dd0e9dd25df4d7f97106627"},
    ApprovedTransform{"ssm.a_neg_exp", 36,
                      "81543b80122b4297547cbdbc21c1415d59b3a3dcd3bddebc9a4632ba77f02c79"},
};
// MLX's: the same parameter folds (its affine experts stay split).
constexpr std::array kMlxFoldTensors{
    ApprovedTransform{"gdn.value_heads_tiled", 648,
                      "4353c8ba6bb8035f12ad81de77796f49f96863e40be66829f8c24c0462e8669d"},
    ApprovedTransform{"norm.plus_one", 148,
                      "5da54d6f36f3f22b2270222f1c9a648e93d7a6ec1dd0e9dd25df4d7f97106627"},
    ApprovedTransform{"ssm.a_neg_exp", 36,
                      "81543b80122b4297547cbdbc21c1415d59b3a3dcd3bddebc9a4632ba77f02c79"},
};
constexpr std::array kModelOptRelayouts{
    ApprovedTransform{"split", 194,
                      "b32a3c70ebb8e0674db510fe8031faeefb16c1271e5170992221fe24edbce802"},
};
// Its routed experts' input scales: the W4A4 grouped GEMM's policy rule.
constexpr std::array<std::string_view, 1> kModelOptGates{"llmp.qwen4exp.expert_activation_scales"};
// The drafter's: its nine norms folded, its one MoE's two tables relaid.
constexpr std::array<std::string_view, 1> kMtpFolds{"norm.plus_one"};
constexpr std::array kMtpFoldTensors{
    ApprovedTransform{"norm.plus_one", 9,
                      "ee1c7aec4e0625327a9fee3dfd69eaf1d2c81ab2420af6aec10d902fb070e393"},
};
constexpr std::array kMtpRelayouts{
    ApprovedTransform{"split", 4,
                      "d92d1cade97145947781ef03383f2528fe7dc55ec91fb12e4bfae40a4fb468dd"},
};
// Gemma 4 26B-A4B's routed experts gate its policy rules.
constexpr std::array<std::string_view, 1> kGemma4Experts{"gemma4.expert_count"};
// Gemma 4's envelope is its mapping's rule domains: the joined recipes
// within a context of 8,192 and four owners, and beyond them the scalar
// owners-alone route at M3's bounded admission: twelve scalar owners
// qualified; context to the trained 262,144, long context not verified
// (docs/experiments/gemma31-serving; docs/model-support.md, "Envelope").
// Memory bounds it below. A default context stays within the joined
// domains (ApprovedDefaultContext).
constexpr std::uint32_t kGemma4Context = 262144;
constexpr std::uint32_t kGemma4Owners = 12;

constexpr std::uint32_t kAnyOwners = std::numeric_limits<std::uint32_t>::max();

// Same-source/spec historical arithmetic. These records preserve the
// accepted recipe inside its actual envelope; they do not qualify a new
// format, source, implementation reduction or unrelated smaller graph.
using NC = planner::NumericalChange;
constexpr std::array kProductAnchor{ir::Primitive(ir::Op::kMatMul)};
constexpr std::array kNormAnchor{ir::Primitive(ir::Op::kRmsNorm)};
constexpr std::array kReduceAnchor{ir::Primitive(ir::Op::kExpertReduce)};
constexpr std::array kVectorAnchors{
    ir::Primitive(ir::Op::kMatMul), ir::Primitive(ir::Op::kMatMulId), ir::Primitive(ir::Op::kGlu),
    ir::Fused(ir::FusedKind::kMatMulGlu), ir::Fused(ir::FusedKind::kMatMulIdGlu)};
constexpr std::array<std::uint64_t, 4> kQualifiedGgufVectorRows{1, 2, 3, 4};
constexpr std::array kNativeNumerics{
    NumericalEvidenceDomain{.change = NC::kBf16HyperConnectionIntermediates,
                            .evidence = kNativeNumericalEvidence,
                            .component_kind = "hc.mix",
                            .anchors = kNormAnchor,
                            .arch = 121,
                            .max_context = 262144,
                            .max_slots = 4,
                            .max_rows = 4096,
                            .stream_rows = true},
    NumericalEvidenceDomain{.change = NC::kBf16ProductInputAndResult,
                            .evidence = kNativeNumericalEvidence,
                            .component_kind = "attention.gdn",
                            .anchors = kProductAnchor,
                            .arch = 121,
                            .max_context = 262144,
                            .max_slots = 4,
                            .max_rows = 4096},
    NumericalEvidenceDomain{.change = NC::kMxfp8ProductInput,
                            .evidence = kNativeNumericalEvidence,
                            .anchors = kProductAnchor,
                            .arch = 121,
                            .max_context = 262144,
                            .max_slots = 4,
                            .max_rows = 4096},
    NumericalEvidenceDomain{.change = NC::kNvfp4ExpertIntermediates,
                            .evidence = kNativeNumericalEvidence,
                            .component_kind = "ffn.moe",
                            .anchors = kReduceAnchor,
                            .arch = 121,
                            .max_context = 262144,
                            .max_slots = 4,
                            .max_rows = 4096,
                            .row_parameter = "used"},
};
constexpr std::array kGgufVectorNumerics{
    NumericalEvidenceDomain{
        .change = NC::kQ8VectorReduction,
        .evidence = "8b9e1298f0e1759cfae0c3f1c792fe79052a1af02c9b9d4e93e9aca137512041",
        .anchors = kVectorAnchors,
        .arch = 121,
        .max_context = 262144,
        .max_slots = 4,
        .max_rows = 4,
        .rows = kQualifiedGgufVectorRows},
};
// Final accepted DeepSeek numerics used this selected VecQ schedule. The
// unchanged route/owner/native predicates remain required by every candidate.
constexpr std::array<std::uint64_t, 5> kQualifiedDeepSeekVectorRows{1, 2, 3, 4, 5};
constexpr std::array kDeepSeekNumerics{
    NumericalEvidenceDomain{.change = NC::kQ8VectorReduction,
                            .evidence = kFinalNumericalEvidence,
                            .anchors = kVectorAnchors,
                            .arch = 121,
                            .max_context = 1048576,
                            .max_slots = 4,
                            .max_rows = 5,
                            .rows = kQualifiedDeepSeekVectorRows,
                            .incumbent_recipe = true},
};

constexpr std::array kApprovals{
    Approval{.name = "gemma-2-2b-it-q8_0",
             .label = "Gemma2",
             .architecture = "gemma2",
             .container = "gguf",
             .sources = kGemma2Sources,
             .tensors = 288,
             .inventory = "caaa925eb5f24c566ffec8972f598d6af75ffd787b3b5629e09c783727c0fdff",
             .spec = "5d9a04370e581d5bddc1eeda9b1c4eb6222372ec5bd0d3db723bafacc999c28f",
             .tied_output = true,
             .max_context = 8192,
             .max_slots = 4},
    Approval{.name = "gemma-3-4b-it-qat-q4_0",
             .label = "Gemma3",
             .architecture = "gemma3",
             .container = "gguf",
             .sources = kGemma3Sources,
             .tensors = 444,
             .inventory = "9da3bfa834bc1c447c06b4588f7b6f12c16c06b3f71437b0893abc8fa176880e",
             .spec = "0f0346b9626697e65fb5ec0ba2506b34af86eb8e0e52000a53ff1b03722741fb",
             .tied_output = true,
             .max_context = 131072,
             .max_slots = 4,
             .wide_context = 4096},
    Approval{.name = "gemma-4-26b-a4b-it-ud-q4_k_m",
             .label = "Gemma26",
             .architecture = "gemma4",
             .container = "gguf",
             .sources = kGemma26Sources,
             .tensors = 658,
             .arrays = 60,
             .inventory = "cefd4cfe5a3b8e6d3c0facb34933a8124c49890761a56b905efd5208f627e5e6",
             .spec = "192b6aabb06a3aed27642931b9a34a446ec1ad131b4b5c4734584d36aadbcb05",
             .tied_output = true,
             .max_context = kGemma4Context,
             .max_slots = kGemma4Owners,
             .gates = kGemma4Experts},
    Approval{.name = "gemma-4-31b-it-ud-q4_k_xl",
             .label = "Gemma31",
             .architecture = "gemma4",
             .container = "gguf",
             .sources = kGemma31Sources,
             .tensors = 833,
             .inventory = "9109e8abbf01494c81640a14167774200debe1533c39fed86708708289bf2c3b",
             .spec = "c0f37f67eb41ae311aa5ccbc700c4bd2a33921a7725ba6737a0e0873afd4705d",
             .tied_output = true,
             .max_context = kGemma4Context,
             .max_slots = kGemma4Owners},
    Approval{.name = "mtp-gemma-4-26b-a4b-it",
             .label = "Gemma26 assistant",
             .architecture = "gemma4-assistant",
             .drafter = true,
             .container = "gguf",
             .sources = kGemma26AssistantSources,
             .tensors = 49,
             .inventory = "5a7a022877abf6adef7fa034e9aad391966b02449586764f0155c0674d5e433d",
             .spec = "68ff238bdd53a407398c6ce4a4a924545348afed0da25f835c9deee27c0ef045",
             .max_context = 262144,
             .max_slots = 4,
             .policy = ApprovedPolicy::kNone,
             .target = "gemma-4-26b-a4b-it-ud-q4_k_m"},
    Approval{.name = "mtp-gemma-4-31b-it",
             .label = "Gemma31 assistant",
             .architecture = "gemma4-assistant",
             .drafter = true,
             .container = "gguf",
             .sources = kGemma31AssistantSources,
             .tensors = 49,
             .inventory = "a87c724f9a0afb06f38abb486981111b7693d6c598ad876067fe9f12b3abef64",
             .spec = "95baca720b7d34edfab6fba8ee275aa2e64f94652748ed15819fdbee4bcdd591",
             .max_context = 262144,
             .max_slots = 4,
             .policy = ApprovedPolicy::kNone,
             .target = "gemma-4-31b-it-ud-q4_k_xl"},
    Approval{.name = "qwen2.5-0.5b-instruct-fp16",
             .label = "Qwen2.5 FP16",
             .architecture = "qwen2",
             .container = "gguf",
             .sources = kQwen2Fp16Sources,
             .tensors = 291,
             .inventory = "f80c4d5b2a974a164fe890b3695da3d94cf6f3526266df12ac67799bc0d90ac2",
             .spec = "d673370e16b0b37fe06a16305abcf032842a8d792afa6b9e30f77e10027f5403",
             .max_context = 8192,
             .max_slots = 1},
    Approval{.name = "qwen2.5-0.5b-instruct-exl3-4.0bpw",
             .label = "Qwen2.5 EXL3 4.0",
             .architecture = "qwen2",
             .container = "exl3",
             .sources = kQwen2Trellis40Sources,
             .tensors = 798,
             .inventory = "fac1aa40c2bd7e690872017f3cc02712f886fd411f726f601312d9201fffad3c",
             .spec = "7d52f0cb668ef4a6e29ddad3d11c8cb482ee98e8646e095d9906a3397a941477",
             .max_context = 8192,
             .max_slots = 1},
    Approval{.name = "qwen2.5-0.5b-instruct-exl3-4.5bpw",
             .label = "Qwen2.5 EXL3 4.5",
             .architecture = "qwen2",
             .container = "exl3",
             .sources = kQwen2Trellis45Sources,
             .tensors = 798,
             .inventory = "5d60dbe367c9e8dd7b5803bc69bf2d64cec762e4b65db795306e44cf0e46e426",
             .spec = "7d52f0cb668ef4a6e29ddad3d11c8cb482ee98e8646e095d9906a3397a941477",
             .max_context = 8192,
             .max_slots = 1},
    Approval{.name = "llama-3.2-1b-instruct-q8_0",
             .label = "Llama 3.2 1B Q8_0",
             .architecture = "llama",
             .container = "gguf",
             .sources = kLlama32Q8Sources,
             .tensors = 147,
             .inventory = "772f985544f2ab057c5fa4aa2dfa471469d9fc5cd3596ca20521843b34f99344",
             .spec = "cbf7950a9a03027877b4ff8fe5d60d0abb2ada79a3dd397b710a5b405ebc6357",
             .tied_output = true,
             .max_context = 131072,
             .max_slots = 4},
    Approval{.name = "llama-3.2-1b-instruct-q4_k_m",
             .label = "Llama 3.2 1B Q4_K_M",
             .architecture = "llama",
             .container = "gguf",
             .sources = kLlama32Q4kmSources,
             .tensors = 147,
             .inventory = "cc4c942c3533a7560bc05826a844376e434c72c8cfa355821cf2d376d746806b",
             .spec = "cbf7950a9a03027877b4ff8fe5d60d0abb2ada79a3dd397b710a5b405ebc6357",
             .tied_output = true,
             .max_context = 131072,
             .max_slots = 4},
    // Three shards, UD-IQ3_XXS. The writer verified their whole streams while
    // publishing; admission checks their receipts and the complete stored
    // inventory and spec without rehashing payloads.
    Approval{.name = "qwen3.8-flash-next-ud-iq3_xxs",
             .label = "Qwen3.8 GGUF",
             .architecture = "qwen4exp",
             .container = "gguf",
             .sources = kQwen38GgufSources,
             .tensors = 1224,
             .arrays = 144,
             .inventory = "a580f533d1b0ffe53e0650a17e5e479328cc8f3d1dd5aaeedb2987f7d0f1d01a",
             .spec = "74116647d445819311c58dc16336099c95d675d827a5cd4125cfd88fadbd47cb",
             .max_context = 262144,
             .max_slots = kAnyOwners,
             .numerical = kGgufVectorNumerics},
    // The native import (`llmp import`), its experts and n-gram table relaid
    // out and its norms and A folded; served beside its drafter below bit for
    // bit as the v0 pair (docs/experiments/m36-numerical-audit/
    // qwen-modelopt-v1.md). The W4A4 grouped GEMM where its experts' input
    // scales say the checkpoint quantized their activations.
    Approval{.name = "qwen3.8-flash-next-modelopt",
             .label = "Qwen3.8 ModelOpt",
             .architecture = "qwen4exp",
             .container = "safetensors",
             .folds = kQwenNativeFolds,
             .sources = kQwen38ModelOptSources,
             .fold_tensors = kModelOptFoldTensors,
             .relayouts = kModelOptRelayouts,
             .tensors = 1789,
             .arrays = 192,
             .inventory = "857e32b418f0f073e0315015fce98d69e2e0019ea93fa7cb89656f0bf8832a8c",
             .spec = "c2e2822e48502a42783ab9f1c4d8ecf3649f02e6331ea2c0ea133e2542983a0f",
             .max_context = 262144,
             .max_slots = kAnyOwners,
             .gates = kModelOptGates,
             .mapping = &spec::mappings::Qwen38NativeMapping,
             .numerical = kNativeNumerics},
    // Its MTP drafter (`llmp import --component mtp --draft-head`), with
    // Mia's selected head; composed with the approved native target.
    Approval{.name = "qwen3.8-flash-next-modelopt-mtp-mia47172",
             .label = "Qwen3.8 ModelOpt MTP",
             .architecture = "qwen4exp-mtp",
             .drafter = true,
             .container = "safetensors",
             .folds = kMtpFolds,
             .sources = kQwen38MtpSources,
             .fold_tensors = kMtpFoldTensors,
             .relayouts = kMtpRelayouts,
             .tensors = 38,
             .arrays = 4,
             .inventory = "41743b0b88b63e1c6b48fdee328f06cb20f8442d86840369526287f6a98f1bfc",
             .spec = "196d10be50adf4afbb8b0864672d09b76c63b66388255ddb4b5254525719159d",
             .max_context = 262144,
             .max_slots = kAnyOwners,
             .policy = ApprovedPolicy::kNone,
             .target = "qwen3.8-flash-next-modelopt",
             .mapping = &spec::mappings::Qwen38MtpNativeMapping,
             .numerical = kNativeNumerics},
    Approval{.name = "qwen3.8-flash-next-mlx",
             .label = "Qwen3.8 MLX",
             .architecture = "qwen4exp",
             .container = "mlx",
             .folds = kQwenNativeFolds,
             .sources = kQwen38MlxSources,
             .fold_tensors = kMlxFoldTensors,
             .tensors = 2954,
             .arrays = 432,
             .inventory = "8b049b07d06ba1da85d7ee7d3a7818a3d98c5ea546226e514355b75073d03974",
             .spec = "90e6fbbbbcad4297fc6d56e900674c2ec865aa29e96b9d3c5098a07c557eff0a",
             .max_context = 262144,
             .max_slots = kAnyOwners,
             .policy = ApprovedPolicy::kNone,
             .mapping = &spec::mappings::Qwen38NativeMapping},
    // Its mapping deliberately has no policy rules: the engine's default
    // outside the shared route's own recipe.
    Approval{.name = "deepseek-v4-flash-0731-ud-q2_k_xl",
             .label = "DeepSeek V4 Flash",
             .architecture = "deepseek4",
             .container = "gguf",
             .sources = kDsv4Sources,
             .tensors = 1328,
             .arrays = 129,
             .inventory = "e4bead1c37f8938de7d1fd186c3021140348533b9160c7726465d7b4344b71fd",
             .spec = "44c7eb37336be510acca6fe22ad676036bd8a40504593d1145e560879f7493de",
             .max_context = 1048576,
             .max_slots = kAnyOwners,
             .policy = ApprovedPolicy::kMappingOrNone,
             .numerical = kDeepSeekNumerics},
    // antirez/deepseek-v4-gguf@f71f23d5's community GGUF (IQ2_XXS gate/up
    // experts, Q2_K down, F16 compressor APE, mixing and router tables),
    // imported natively as v1 e7c0d0e4. It lists no shared-expert clamps, so
    // its shared expert clamps as its routed ones (10), as M3's profile
    // served it; its stored spec is byte-identical to the UD-Q2_K_XL
    // target's (the same digest). Its M3 evidence (v0 cd39d504, docs/
    // model-support.md#deepseek-v4-community-iq2_xxs, experiments/ds4-study
    // and deepseek-batching): plain greedy through 128,817 prompt tokens in
    // the native harness (one owner); through the runtime at context 16,384,
    // and at context 262,144 in four-slot plain waves of 7K prompts. The v1
    // artifact's initial evidence was its frozen-digest smoke test alone.
    // The scalar incumbent domain below adds its exact same-checkpoint
    // passage/reference evidence; it does not qualify a multirow recipe.
    // No drafter approval names this target.
    Approval{.name = "deepseek-v4-flash-iq2xxs-w2q2k-chat-v2-imatrix-0731",
             .label = "DeepSeek V4 Flash IQ2_XXS",
             .architecture = "deepseek4",
             .container = "gguf",
             .sources = kDsv4Iq2xxsSources,
             .tensors = 1328,
             .arrays = 129,
             .inventory = "b3dad08e426a83ab38991ba3b479b363b26ca8a6df581301ff8afd9675ce41ad",
             .spec = "ddaaa1d38bd6a501b9ef891bc0f639a70f41b19ff85a85674be5d77470e7a8e8",
             .max_context = 262144,
             .max_slots = 4,
             .policy = ApprovedPolicy::kMappingOrNone,
             .numerical = kDeepSeekNumerics},
    Approval{.name = "dspark-deepseek-v4-flash-0731-q8_0",
             .label = "DSpark",
             .architecture = "dflash",
             .drafter = true,
             .container = "gguf",
             .sources = kDsparkSources,
             .tensors = 81,
             .arrays = 9,
             .inventory = "a9905c3984a3d7598444825d5f9f281270c276d554c3e2f8c909adfb7e705b97",
             .spec = "ac3d1a4d67f8d77b288ba70c227c92a00b38c639dcaac3cad9e6a600b5857ccd",
             .max_context = 1048576,
             .max_slots = kAnyOwners,
             .policy = ApprovedPolicy::kNone,
             .target = "deepseek-v4-flash-0731-ud-q2_k_xl",
             .numerical = kDeepSeekNumerics},
};

// Qwen-Image-2.1 (Qwen/Qwen-Image-2.1@790c9263): the pipeline's kept files
// and its components' sources, as `llmp import` declared them from the
// staged, archive-verified pipeline; the family spec and component specs
// the importer synthesized (spec/mappings/qwen_image.cc), executed at the
// pixel pin (docs/model-support.md).
constexpr std::array kQwenImageSources{
    SourceReceipt{"model_index.json",
                  "cf1ecd104ea090855d60d8cec0895c1e9d6ee41b2f87e231b678beecd1cf7809", 447ULL},
    SourceReceipt{"processor/added_tokens.json",
                  "c0284b582e14987fbd3d5a2cb2bd139084371ed9acbae488829a1c900833c680", 707ULL},
    SourceReceipt{"processor/chat_template.jinja",
                  "3636d0f0bd6bef02654cdffdc447b79cb2cef8ab02cc75267345946291a489e4", 5292ULL},
    SourceReceipt{"processor/merges.txt",
                  "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5", 1671853ULL},
    SourceReceipt{"processor/preprocessor_config.json",
                  "93585062a80db5e8ca038efc7726a3e6411d9db948472d81d63c6303993be8c5", 782ULL},
    SourceReceipt{"processor/special_tokens_map.json",
                  "76862e765266b85aa9459767e33cbaf13970f327a0e88d1c65846c2ddd3a1ecd", 613ULL},
    SourceReceipt{"processor/tokenizer.json",
                  "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4", 11422654ULL},
    SourceReceipt{"processor/tokenizer_config.json",
                  "81ec7bb9530159b326c0bef1d0b6c33d392090524014ea3f0123a3c1eb9c2af5", 5445ULL},
    SourceReceipt{"processor/video_preprocessor_config.json",
                  "59c5c9eb52182eb14c06ffb10ca9effd29adce5f238a95de23ca14a38dbd2cb1", 817ULL},
    SourceReceipt{"processor/vocab.json",
                  "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910", 2776833ULL},
    SourceReceipt{"scheduler/scheduler_config.json",
                  "5895f3a167c14a967fe9ac70c64924ae5acc79799e0679fd12907e594a713cd1", 485ULL},
};

constexpr std::array kQwenImageTextSources{
    SourceReceipt{"config.json", "6001331949f7f86ab6d791a80b12246d4f8da5159fa08c3e62368a39f36d63a9",
                  1517ULL},
    SourceReceipt{"generation_config.json",
                  "4d9818c3d27895c0058828a5f68bc7a4de80c3ae1bcdac90936ce24178063f59", 213ULL},
    SourceReceipt{"model-00001-of-00004.safetensors",
                  "dde00291b5f7fb92013895310a3da0ddba78674df9f10d505d375243dc01fc6f",
                  4998056552ULL},
    SourceReceipt{"model-00002-of-00004.safetensors",
                  "9047faccc0a6d98496a52d55f27be1c94a9c259d1e283fbea0128d054a948d42",
                  4915962464ULL},
    SourceReceipt{"model-00003-of-00004.safetensors",
                  "8c54187654c0176b73ae73785bf791dc9a14c9df7fb4310083a09d42048cb57e",
                  4915962496ULL},
    SourceReceipt{"model-00004-of-00004.safetensors",
                  "5311532aaaeae3259eb6a7b2c600636be1159adf7ded35f53579f7d0e7d43cdd",
                  2704357976ULL},
    SourceReceipt{"model.safetensors.index.json",
                  "859146bff8db1d617fca8b457a5223edff963bb70076b3645f2189fe0540a147", 67795ULL},
};

constexpr std::array kQwenImageTransformerSources{
    SourceReceipt{"config.json", "56ae3281c4e6c2d1aa3658252d187488071815fd79bef15808bb0205fc1a2241",
                  370ULL},
    SourceReceipt{"diffusion_pytorch_model-00001-of-00002.safetensors",
                  "9e6bc2d641e67bf277895ea8777141044a38f3edb7101bc469b2961dd7c36b4b",
                  9968332504ULL},
    SourceReceipt{"diffusion_pytorch_model-00002-of-00002.safetensors",
                  "3aaf234dcbe128530479735854a346b5e3e66283b7c11db56f836bbd1c13ebaa",
                  4261951904ULL},
    SourceReceipt{"diffusion_pytorch_model.safetensors.index.json",
                  "17987f6623b1c814d0ef55a137d99142b7b3b040eb1bf241b5575dd35af803a2", 30283ULL},
};

constexpr std::array kQwenImageVaeSources{
    SourceReceipt{"config.json", "9785d527b278cb8b210e9f48a8028d92a4190968ce7e24c8fc85d9d1b82f6ba2",
                  2079ULL},
    SourceReceipt{"diffusion_pytorch_model.safetensors",
                  "a07a1b7c4ee2966a1b3bdc37de9b4f983d56937e46619f709a80b6e490675417",
                  1350989512ULL},
};

constexpr std::array kQwenImageComponents{
    Approval{.name = "qwen-image-2.1-text-encoder",
             .label = "QwenImage21",
             .architecture = "qwen3_vl",
             .container = "safetensors",
             .sources = kQwenImageTextSources,
             .tensors = 397,
             .inventory = "11829262ca3bf720fc5608a424668b1c3a2cb26f80971b1460ede5fca6317d5e",
             .spec = "b815577c4e8d31fd70a4b8be682c5d590f83ada9f4bb1c299b5a249ef6818f2d",
             .policy = ApprovedPolicy::kNone},
    Approval{.name = "qwen-image-2.1-transformer",
             .label = "QwenImage21",
             .architecture = "QwenImage21Transformer2DModel",
             .container = "safetensors",
             .sources = kQwenImageTransformerSources,
             .tensors = 297,
             .inventory = "984e1dd57dbe7ad5011c4bdbcee19bea18529ee1d4086e9428fd278f225849e9",
             .spec = "3adbff4e47d168ceafceb41e3da1961b1be3692be4a3af40e28329b356f07a43",
             .policy = ApprovedPolicy::kNone},
    Approval{.name = "qwen-image-2.1-vae",
             .label = "QwenImage21",
             .architecture = "AutoencoderKLQwenImage21",
             .container = "safetensors",
             .sources = kQwenImageVaeSources,
             .tensors = 128,
             .inventory = "dc21b79eddff66b8cc71a7297ab47cebb7875ee18509cde02312c30e934a485b",
             .spec = "acadec6d800a6971f6e8a28e6cc91090077c081b2631bddd7c232847f2d2de72",
             .policy = ApprovedPolicy::kNone},
};

constexpr std::array kImageGluAnchor{ir::Primitive(ir::Op::kGlu)};
constexpr std::array kImageAttentionAnchor{ir::Primitive(ir::Op::kAttention)};
constexpr std::array kImageCarryAnchor{ir::Fused(ir::FusedKind::kLayerNormModulateSplit)};
constexpr std::array kImageNumerics{
    NumericalEvidenceDomain{.change = NC::kBf16SwiGluIntermediates,
                            .evidence = kImageNumericalEvidence,
                            .anchors = kImageGluAnchor,
                            .arch = 121},
    NumericalEvidenceDomain{.change = NC::kBf16ProductInputAndResult,
                            .evidence = kImageNumericalEvidence,
                            .anchors = kProductAnchor,
                            .arch = 121},
    NumericalEvidenceDomain{.change = NC::kBf16AttentionIntermediates,
                            .evidence = kImageNumericalEvidence,
                            .anchors = kImageAttentionAnchor,
                            .arch = 121},
    NumericalEvidenceDomain{.change = NC::kBf16ResidualCarry,
                            .evidence = kImageNumericalEvidence,
                            .anchors = kImageCarryAnchor,
                            .arch = 121},
};

constexpr std::array kImageApprovals{
    ImageApproval{.name = "qwen-image-2.1",
                  .label = "QwenImage21",
                  .architecture = "QwenImage21Pipeline",
                  .sources = kQwenImageSources,
                  .spec = "ef266ac1b35803f731982a027d31e3656eb31ddd7fde94f4d549bc70ef70a88a",
                  .components = kQwenImageComponents,
                  .numerical = kImageNumerics,
                  .numerical_width = 1024,
                  .numerical_height = 1024,
                  .numerical_prompt_rows = 25},
};

}  // namespace

std::span<const Approval> Approvals() { return kApprovals; }
std::span<const ImageApproval> ImageApprovals() { return kImageApprovals; }

bool SourcesMatch(std::span<const SourceReceipt> receipts,
                  std::span<const artifact::Source> sources) {
  if (receipts.size() != sources.size()) return false;
  return std::ranges::all_of(receipts, [&](const SourceReceipt& r) {
    return std::ranges::count_if(sources, [&](const artifact::Source& s) {
             return s.name == r.name && s.bytes.value() == r.bytes &&
                    base::ToHex(s.sha256) == r.sha256;
           }) == 1;
  });
}

const Approval* FindApproval(std::string_view architecture,
                             std::span<const artifact::Source> sources) {
  for (const Approval& a : kApprovals)
    if (a.architecture == architecture && SourcesMatch(a.sources, sources)) return &a;
  return nullptr;
}

const Approval* FindApprovalNamed(std::string_view name) {
  const auto found = std::ranges::find(kApprovals, name, &Approval::name);
  return found == kApprovals.end() ? nullptr : &*found;
}

const ImageApproval* FindImageApproval(std::string_view architecture,
                                       std::span<const artifact::Source> sources) {
  for (const ImageApproval& a : kImageApprovals)
    if (a.architecture == architecture && SourcesMatch(a.sources, sources)) return &a;
  return nullptr;
}

}  // namespace llmp::runtime
