// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "runtime/component_admission.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <format>
#include <limits>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "base/sha256.h"
#include "components/step.h"
#include "model/dspark.h"
#include "model/dsv4.h"
#include "model/gemma2.h"
#include "model/gemma3.h"
#include "model/gemma4.h"
#include "model/gemma4_assistant.h"
#include "model/qwen2.h"
#include "model/qwen2_exl3.h"
#include "model/qwen38.h"
#include "runtime/gemma_profile.h"
#include "spec/mappings/dsv4.h"
#include "spec/mappings/gemma2.h"
#include "spec/mappings/gemma3.h"
#include "spec/mappings/gemma4.h"
#include "spec/mappings/qwen2.h"
#include "spec/mappings/qwen38.h"
#include "spec/stored.h"

namespace llmp::runtime {
namespace {

// The existing approved execution checkpoint, not the container's artifact
// ID: publication does not broaden the family/profile or format domain.
enum class ProfileId {
  kGemma2,
  kGemma3,
  kGemma26,
  kGemma31,
  kQwen2Gguf,
  kQwen2Exl3_4_0,
  kQwen2Exl3_4_5,
  kQwenGguf,
  kQwenModelOpt,
  kQwenMlx,
  kDsv4Gguf,
  kGemma4Assistant26,
  kGemma4Assistant31,
  kDsparkGguf
};
struct Profile {
  ProfileId id;
  std::string_view architecture, source_name, source_sha256;
  std::uint64_t source_bytes;
  std::size_t tensors, arrays;
  std::uint32_t max_context, max_slots;
  std::string_view inventory;
};
// Sorted name / native encoding / original dimensions from the approved
// captured tensor directories; algorithm and source pins are documented in
// docs/experiments/m36-native-v1-admission/README.md. This also fixes each
// layer's mixed quantization, rather than accepting a family-wide type set.
constexpr std::array kProfiles{
    Profile{ProfileId::kGemma2, "gemma2", "gemma-2-2b-it-Q8_0.gguf",
            "2d448a9aab894b8e8e18168cf3f490cb9f65632222f29f93514ac9ecc754debe", 2784495456ULL, 288,
            0, 8192, 2, "91678220743a65166375bad2f712380aed8e7cfec56203bfdc87321a5e5ed0f5"},
    Profile{ProfileId::kGemma3, "gemma3", "gemma-3-4b-it-qat-Q4_0.gguf",
            "ee91c3e7a4ab95d8c95672f9fcb58bf236b257e9f217966bcf53a5a6df4ab49a", 2526080992ULL, 444,
            0, 131072, 2, "8aedee1cffc43f331cb38dcd9dc60e98a2aa67cd98df16248f1c78e36bd4a8e7"},
    Profile{ProfileId::kGemma26, "gemma4", "gemma-4-26B-A4B-it-UD-Q4_K_M.gguf",
            "f2c28b3dc4776931ac6f879e11f203dec637ea0f14267a86ec8f6165f63f293f", 16947541728ULL, 658,
            60, 8192, 4, "97551b8fcfead217ce2feb8c0abe7228211b15bca6867158a955f3d5a7790e02"},
    Profile{ProfileId::kGemma31, "gemma4", "gemma-4-31B-it-UD-Q4_K_XL.gguf",
            "9e92cb6236044c6a9870af406029c74a76e0571c157a6f95df724dcc8c7a1575", 18822970304ULL, 833,
            0, 8192, 4, "a5ba88997fe0b72a54148526573c38e6a619f5a91980c59dcb5cbc5123760f52"},
    Profile{ProfileId::kQwen2Gguf, "qwen2", "qwen2.5-0.5b-instruct-fp16.gguf",
            "8e0ae26000627ed62de0e78e41860af70094558b9d2913385c842a6aa06cf3fc", 1266425696ULL, 291,
            0, 8192, 1, "0ac91b5fa50d1a17c55de9011431cb67b0224dff6d018e1e8c3e8d6112a5adbe"},
    Profile{ProfileId::kGemma4Assistant26, "gemma4-assistant", "mtp-gemma-4-26B-A4B-it.gguf",
            "6326fb9f5e487aa8dcdd313a091e3c67724cb2a666ec3b7d2895b5b26d93ed1b", 461766816ULL, 49, 0,
            262144, 4, "5762a622394aae1fa4092013fae5333471158416b98efc5af46f40e45bb0865c"},
    Profile{ProfileId::kGemma4Assistant31, "gemma4-assistant", "mtp-gemma-4-31B-it.gguf",
            "5ae8b0117bed601e8924c6305bd5b0585de361d51f0e77091bcb4252cf1f27de", 514687104ULL, 49, 0,
            262144, 4, "cf1cbbffb8974c67423242e2902ecbd7efbe94c945234be0277ae1313c9c6455"},
    Profile{ProfileId::kDsparkGguf, "dflash", "dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf",
            "2c7ac54b0b64a99df1f139a9f1371a00198265e1d6a614b77597d20a655a4249", 10896057440ULL, 81,
            9, model::kDsv4FlashContext, std::numeric_limits<std::uint32_t>::max(),
            "aa2938d65ee98872ab3c919b912f2e7f069f7e3c6ded06995450967249ea0a8d"}};

// Pinned three-shard UD-IQ3_XXS source. The ordinary writer verifies these
// whole streams while publishing; runtime admission checks their declared
// identity and the complete stored inventory/spec without rehashing payloads.
constexpr Profile kQwenGguf{ProfileId::kQwenGguf,
                            "qwen4exp",
                            {},
                            {},
                            0,
                            1224,
                            144,
                            model::kQwen38FlashContext,
                            std::numeric_limits<std::uint32_t>::max(),
                            "ebfde9be4041367912853e848599eb2bbfeb62f853ae373546ceb31416d482f6"};
constexpr Profile kQwenModelOpt{ProfileId::kQwenModelOpt,
                                "qwen4exp",
                                {},
                                {},
                                0,
                                1886,
                                432,
                                model::kQwen38FlashContext,
                                std::numeric_limits<std::uint32_t>::max(),
                                "8830408a944e1d2cd1d6b30c121ff1d12e73e34c941f59bbd4c052c0308a3264"};
constexpr Profile kQwenMlx{ProfileId::kQwenMlx,
                           "qwen4exp",
                           {},
                           {},
                           0,
                           2274,
                           432,
                           model::kQwen38FlashContext,
                           std::numeric_limits<std::uint32_t>::max(),
                           "8cabbbeebf6a6786dbc52ac0471b5dd34d360f4a9c000de7a806bb9234f57dd5"};
struct Origin {
  std::string_view name, sha256;
  std::uint64_t bytes;
};
constexpr std::array kQwenOrigins{
    Origin{"Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf",
           "268f81fdedf3149a538f252308927a4d5d1f6e062c178568a51e3b519744f8a8", 10946624ULL},
    Origin{"Qwen3.8-Flash-Next-UD-IQ3_XXS-00002-of-00003.gguf",
           "cfe600b236b88c7fad1613a5ca5e83b9f2beb63cbd44c32b2be50a44747c695f", 49567921344ULL},
    Origin{"Qwen3.8-Flash-Next-UD-IQ3_XXS-00003-of-00003.gguf",
           "f1912ba34c79427d2295a58dcb2b732b5931af5bef7a373c60557a57d9ee7250", 32382955968ULL}};
constexpr std::array kQwenModelOptOrigins{
    Origin{"config.json", "2d655dfa625c3c018f9685031367bc83f8532d3faf3333d7c838b2bdc13b8a7e",
           109897ULL},
    Origin{"hf_quant_config.json",
           "d60c54caa65317fd8f1338bdeb81b9014512ac8f21ec18da22541cb88c6784b7", 99088ULL},
    Origin{"model-00001-of-00034.safetensors",
           "3f870511a20ad170c997517765f03d475bcd364ae1a6011491085cab620d3c39", 2841505880ULL},
    Origin{"model-00002-of-00034.safetensors",
           "fe8f4bdbce28f4911f10460b75581b6c88a8ade9b836d2f1ec22c6bee6f976a8", 1575009800ULL},
    Origin{"model-00003-of-00034.safetensors",
           "2b7d1312a660cf17bb05c34e1ac79570820f86fb1e1115108ee40562e48acb5e", 1575009808ULL},
    Origin{"model-00004-of-00034.safetensors",
           "25ba4508c209a5cc54c04edb1edb8d0a0b5efa80e5f99637e2263fa78d6a25e8", 1575009808ULL},
    Origin{"model-00005-of-00034.safetensors",
           "a9848067bb49f44b8339bcdca9d60ca74d801a6df874c979c6f901c8da6ff545", 1575009808ULL},
    Origin{"model-00006-of-00034.safetensors",
           "66fc496b58675a90b94d9f36765fe67692dd8a4c966ec0692e076d751f835c8d", 1575009808ULL},
    Origin{"model-00007-of-00034.safetensors",
           "68f89e84be13069bd9f2928b44a9f9638d628fd62be5abb6a4fc956c1af25ea9", 1575009808ULL},
    Origin{"model-00008-of-00034.safetensors",
           "b8bdaa0a7b027dc0fa38c2e43bb31e8ddf9ba9288d3d0f28c422e6947241d96e", 1575009808ULL},
    Origin{"model-00009-of-00034.safetensors",
           "6c5a85990b16a93f0e7e28fb76f56713d5cbc05bef494bb15360188f575746e9", 1575009808ULL},
    Origin{"model-00010-of-00034.safetensors",
           "8734a1fc3b5857391fbf2252d4305391a9f77894b70577949f472fc75386582f", 1575009808ULL},
    Origin{"model-00011-of-00034.safetensors",
           "29792378d43c9f70d9f3fbdd07d0ee24b38f81fe27aff25a02ba0d3cde8df13c", 1575009808ULL},
    Origin{"model-00012-of-00034.safetensors",
           "b23e615ebefff4753d982d9ec6a8be6ad57353a8900146a3ec4e217f81c10e61", 1575009808ULL},
    Origin{"model-00013-of-00034.safetensors",
           "9da7b4570fc3f13cb781e5aca5e79d0edfc10be5555a18bc7bda8f47459d5acf", 1575009808ULL},
    Origin{"model-00014-of-00034.safetensors",
           "53e770d0bdf6261e30abaae2477aa0b50f6686511e0240514ded0fd65edbfeba", 1575009808ULL},
    Origin{"model-00015-of-00034.safetensors",
           "03440de56d127ff58e72963aafd0de0574b825667330c50b4441bb300dae9f06", 1575009816ULL},
    Origin{"model-00016-of-00034.safetensors",
           "ce2e0474ccb34c56003a61608b46d3a8f668c218a9993c1f861239df83afe77b", 1575009824ULL},
    Origin{"model-00017-of-00034.safetensors",
           "57f84845e0b8bcab738e70f0f880e941f1baa8da14663293d7d40481288c73a0", 1575009824ULL},
    Origin{"model-00018-of-00034.safetensors",
           "3477df0be60910576fc5a1f8159d8dbbf7847a03872b9cb53664808e1d5dfac8", 1575009824ULL},
    Origin{"model-00019-of-00034.safetensors",
           "43c22213b2fe0990c7fc2aa54c0ea65062eb93253b4ea419914d1a655e64b23a", 2438722732ULL},
    Origin{"model-00020-of-00034.safetensors",
           "b7143694bf8698c5b704ae9b3d492a0e9b8ef57bdda8ae6f3f29f264c183ad17", 5133924592ULL},
    Origin{"model-00021-of-00034.safetensors",
           "522bbbc894183865726e06ca44021d275d03cc0e8dcd60a2c7074d870e811d37", 5194891408ULL},
    Origin{"model-00022-of-00034.safetensors",
           "5a4789f6da6b386b09cb9093b0315fe12a816142958e4d8a4ae83e4fbaa066e7", 5133933936ULL},
    Origin{"model-00023-of-00034.safetensors",
           "cd0aaef1c8688cb4e35b488048ede2a6b653e1d558b20d4657567df24dc8be13", 5194908864ULL},
    Origin{"model-00024-of-00034.safetensors",
           "d07198edd38ae8740d556487729ae87be1e21e31a88cf6624c1b3f9d0fc6de76", 5188755712ULL},
    Origin{"model-00025-of-00034.safetensors",
           "c7d5e642968679ca14d07bdb1e8c2187c59898611d05a6909c209738642c9519", 5133940040ULL},
    Origin{"model-00026-of-00034.safetensors",
           "47a7c7a70188cb3dc318071a5278260c4b05d7ce1b69c3b99e7bf1c0e42c353b", 5194907416ULL},
    Origin{"model-00027-of-00034.safetensors",
           "a81fedf160434806e81f36731c853fe4e3f973dea2f0817043987f32b4915a54", 5133940216ULL},
    Origin{"model-00028-of-00034.safetensors",
           "91c5d479fe4de1035c430ad7fd3de9d15214a46fd786ca7909d1f2fa40ef603a", 5194907672ULL},
    Origin{"model-00029-of-00034.safetensors",
           "f31c6f61968f6f9e85d469f31c55c82ab6c5f2bc54c1d7a0cd6b23ac45f0ce15", 5133939992ULL},
    Origin{"model-00030-of-00034.safetensors",
           "df8627f5b8b0711b2ebf415c1f2549ef0b50f964e0babb560469b191b33c6a14", 5194908960ULL},
    Origin{"model-00031-of-00034.safetensors",
           "654c8f9ae1ab5de393cbcdd46a13bd0d3d1c1005e575879e0e660d40ad9ef425", 5188755536ULL},
    Origin{"model-00032-of-00034.safetensors",
           "7de2e680802ad92a0e019f875ebc07f2d5d82c8bca1c64b0f817d547bc3c9de7", 5133940208ULL},
    Origin{"model-00033-of-00034.safetensors",
           "0fbcdf16c13cd830447efdcd3be19b82ec0e76b7f57f0e96952c1f2b62002b11", 5021547836ULL},
    Origin{"model-00034-of-00034.safetensors",
           "fe0a0a227e39c4d02dc66c726d3e1ad87ef082e90a6b023ac90b2bb301672203", 1597270800ULL},
    Origin{"model-inputscales.safetensors",
           "49c7d04606ff7c8f103b4d1fc9742b44649a2937b25ecc896a5450ab1020f308", 9669936ULL},
    Origin{"model.safetensors.index.json",
           "435eef76fc10fc6e932a208a1f85a86bc9c7ffc389b27b34ba83bb6a5d0371e9", 33074286ULL}};
constexpr std::array kQwenMlxOrigins{
    Origin{"config.json", "4efd5f41c47e8de9f55c678df4a237cbbc78e4d517dad42a906889f7d94f79a2",
           5361ULL},
    Origin{"model-00001-of-00022.safetensors",
           "308f20b7e35a3e525f36eca0df286f6d1043cfeb621e7bbcf372a834c99cd66d", 5283195887ULL},
    Origin{"model-00002-of-00022.safetensors",
           "8af1a616f2f39dbfe6ee92e959afb2414623fcb299aff1ced55984bfe810d608", 5250035137ULL},
    Origin{"model-00003-of-00022.safetensors",
           "519a0472252ac9a51af020686f462aa6be82ce655f948fd5f054d70949de17a6", 5250035154ULL},
    Origin{"model-00004-of-00022.safetensors",
           "b1ad7e951969eb8dc4e92762dc2e50bd47353c95990ae207db5da393d441400f", 5250035160ULL},
    Origin{"model-00005-of-00022.safetensors",
           "9ca797304c563bb65f2356b67d444511b11744bc5b5c8ec8345ae7d97d3e251c", 5250035164ULL},
    Origin{"model-00006-of-00022.safetensors",
           "8aab4c1dce7d8a0af645426b66d1caf30245d422d4c6b7029583959d675a1827", 5250035184ULL},
    Origin{"model-00007-of-00022.safetensors",
           "7908da37b2e3fa7f0de8b9734dad0759be7dc445716f3acd59c20c4842638db9", 5067850140ULL},
    Origin{"model-00008-of-00022.safetensors",
           "85beff20f250bfe63e95f12c0460512aab92bcde8f7408fe9c20e3de0321bc1d", 5337131618ULL},
    Origin{"model-00009-of-00022.safetensors",
           "f7c03ae640e9bf0f7813ef2b2980afa83a883c02b3988e814e9921055fc08a28", 5337131558ULL},
    Origin{"model-00010-of-00022.safetensors",
           "6897f52227533e720b243d6aef24c1293ad1eb02a1bb2ec6a6e4dc48d56d0c8f", 4970130103ULL},
    Origin{"model-00011-of-00022.safetensors",
           "905d27f9db887284289a043f9638020258c5f90d52122cdce2e7aef941a431f2", 5341257482ULL},
    Origin{"model-00012-of-00022.safetensors",
           "c4d2bb76fa4613f7386ddaa7045608c320f9736623211a814aa28fb370530dec", 4967968541ULL},
    Origin{"model-00013-of-00022.safetensors",
           "1f20a5e1f2256b2fb6a78fef52f9e66daa970f4318ab1be3abb32a1effedbbf3", 5337131651ULL},
    Origin{"model-00014-of-00022.safetensors",
           "860d12691792df5f08c9dfe245dc356773741f1edcc1fd430672ef06ac387635", 5337131799ULL},
    Origin{"model-00015-of-00022.safetensors",
           "259fb7577b8248f6dcbe6a932fe5d2f1c8869eda761ef3394fe6ad06065f216c", 4970130195ULL},
    Origin{"model-00016-of-00022.safetensors",
           "e931785f9ff4e0a567bf8c06df37639be46b4dae156e0f8b39b9d32a03d75792", 5341257440ULL},
    Origin{"model-00017-of-00022.safetensors",
           "4f4610dc4a862bf9d93d70cfd0ec037e8e7943a734794d34eb5c0d748b74747a", 4967968481ULL},
    Origin{"model-00018-of-00022.safetensors",
           "afdb4828920186600f9d61537d1465979fb36692f4eddb8c5a961a6b72964b77", 5337131695ULL},
    Origin{"model-00019-of-00022.safetensors",
           "869d34c6e8f7db79bd4e54b9833d5b0db45eefac860c9bd64fbd312bf862a667", 5337131801ULL},
    Origin{"model-00020-of-00022.safetensors",
           "5ca2f994fec1aa06fefef6646b9f072cf70f7f73aa8dab4c7c1a9a424afb4a52", 4970130135ULL},
    Origin{"model-00021-of-00022.safetensors",
           "58def6762f29cc798437e5649a56ee75d053faca69138a2454dbdb34bb8a37c3", 5341257436ULL},
    Origin{"model-00022-of-00022.safetensors",
           "30a3c82b573d813a1d2349f1c05c308c0d415f3ca1bb67caa0bfaad8a6afa7a3", 3715570974ULL},
    Origin{"model.safetensors.index.json",
           "8848503860c7da8619fdc8d5e65201868f8f1d21d69d2e21166c95522379ff77", 420813ULL}};

constexpr Profile kQwen2Exl3_4_0{
    ProfileId::kQwen2Exl3_4_0,
    "qwen2",
    {},
    {},
    0,
    798,
    0,
    8192,
    1,
    "6168967c4decc23a1bc531830d6e10aa13382d7d545631c93caf59f2476c1c4a"};
constexpr std::array kQwen2Exl3_4_0Origins{
    Origin{"config.json", "966657b7e8c331ea16127a868eb6d6ff16ad0b16a5ee39d26279e12de0a3205c",
           996ULL},
    Origin{"model.safetensors", "66b61bd1ca3ebb0fc1cea29148eda8bdaca45d232c6a7e319c6d2835b65c066b",
           588951098ULL}};

constexpr Profile kQwen2Exl3_4_5{
    ProfileId::kQwen2Exl3_4_5,
    "qwen2",
    {},
    {},
    0,
    798,
    0,
    8192,
    1,
    "19bf365e055876d87db9c25a1109ac9a1be39b8a3be4e398dc8c6479e6c1f484"};
constexpr std::array kQwen2Exl3_4_5Origins{
    Origin{"config.json", "dcd945f38ba6e7835c8c281a45c03d9a20759d03afe5761d09def8c62aa11f3c",
           923ULL},
    Origin{"model.safetensors", "94c45f9838a56fbf23f43d0a04c47e187d840dd97f17b4a027a69a20339762bc",
           611257914ULL}};

constexpr Profile kDsv4Gguf{ProfileId::kDsv4Gguf,
                            "deepseek4",
                            {},
                            {},
                            0,
                            1328,
                            129,
                            model::kDsv4FlashContext,
                            std::numeric_limits<std::uint32_t>::max(),
                            "0c4a8d8bc83e96cac484de7c6272c75d282fe488d792ef60f686138f5a177c9c"};
constexpr std::array kDsv4Origins{
    Origin{"DeepSeek-V4-Flash-0731-UD-Q2_K_XL-00001-of-00003.gguf",
           "de65c5bb660817b95cc281bf54935f9d1bcc3b22535f2eae63b35a14c7c5724b", 5257664ULL},
    Origin{"DeepSeek-V4-Flash-0731-UD-Q2_K_XL-00002-of-00003.gguf",
           "c8728a298862cd8736782e2e4056e5e3e03c415ba552a38173264494e41ef327", 49437013568ULL},
    Origin{"DeepSeek-V4-Flash-0731-UD-Q2_K_XL-00003-of-00003.gguf",
           "a4bae0788c025c1b2388754592c98fefa1372981748ac9875039bf6a30e6eeb4", 47390237120ULL}};

const Profile* Approved(std::string_view architecture, std::span<const artifact::Source> origins) {
  for (const auto& [profile, expected_origins] :
       std::array{std::pair{&kQwen2Exl3_4_0, std::span<const Origin>(kQwen2Exl3_4_0Origins)},
                  std::pair{&kQwen2Exl3_4_5, std::span<const Origin>(kQwen2Exl3_4_5Origins)},
                  std::pair{&kQwenGguf, std::span<const Origin>(kQwenOrigins)},
                  std::pair{&kQwenModelOpt, std::span<const Origin>(kQwenModelOptOrigins)},
                  std::pair{&kQwenMlx, std::span<const Origin>(kQwenMlxOrigins)},
                  std::pair{&kDsv4Gguf, std::span<const Origin>(kDsv4Origins)}}) {
    if (architecture != profile->architecture || origins.size() != expected_origins.size())
      continue;
    bool matched = true;
    for (const auto& expected : expected_origins)
      if (std::ranges::count_if(origins, [&](const auto& origin) {
            return origin.name == expected.name && origin.bytes.value() == expected.bytes &&
                   base::ToHex(origin.sha256) == expected.sha256;
          }) != 1)
        matched = false;
    if (matched) return profile;
  }
  if (origins.size() != 1) return nullptr;
  for (const auto& profile : kProfiles)
    if (architecture == profile.architecture && origins[0].name == profile.source_name &&
        origins[0].bytes.value() == profile.source_bytes &&
        base::ToHex(origins[0].sha256) == profile.source_sha256)
      return &profile;
  return nullptr;
}

std::unexpected<std::string> Refused(std::string_view reason) {
  return std::unexpected("component artifact admission: " + std::string(reason));
}

const model::Gemma4Profile* GemmaProfile(const Profile& profile) {
  if (profile.id == ProfileId::kGemma26) return &model::Gemma4_26BA4B();
  if (profile.id == ProfileId::kGemma31) return &model::Gemma4_31B();
  return nullptr;
}
std::expected<std::vector<spec::SourceValue>, std::string> Values(const Profile& profile) {
  switch (profile.id) {
    case ProfileId::kQwen2Gguf:
      return spec::mappings::Qwen2V0Values(model::Qwen25Instruct05B());
    case ProfileId::kQwen2Exl3_4_0:
    case ProfileId::kQwen2Exl3_4_5:
      return spec::mappings::Qwen2V0Values(model::Qwen25Instruct05BExl3());
    case ProfileId::kDsv4Gguf:
      return spec::mappings::Dsv4V0Values(model::Dsv4Flash());
    case ProfileId::kQwenGguf: {
      // Independent source metadata captured with the approved tensor inventory.
      const model::Qwen38PleHash hash{
          .multipliers = {23703573157769, 20109073645365, 8052911324071},
          .offsets = {0, 20000003, 40000026, 60000059, 80000106, 100000165, 120000228, 140000297,
                      160000374, 180000455, 200000548, 220000655, 240000802, 260000955, 280001114,
                      300001275},
          .vocab = {20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077,
                    20000081, 20000093, 20000107, 20000147, 20000153, 20000159, 20000161,
                    20000171}};
      return spec::mappings::Qwen38GgufValues(model::Qwen38Flash(), hash);
    }
    case ProfileId::kQwenModelOpt: {
      const model::Qwen38PleHash hash{
          .multipliers = {23703573157769, 20109073645365, 8052911324071},
          .offsets = {0, 20000003, 40000026, 60000059, 80000106, 100000165, 120000228, 140000297,
                      160000374, 180000455, 200000548, 220000655, 240000802, 260000955, 280001114,
                      300001275},
          .vocab = {20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077,
                    20000081, 20000093, 20000107, 20000147, 20000153, 20000159, 20000161,
                    20000171}};
      auto values = spec::mappings::Qwen38GgufValues(model::Qwen38Flash(), hash);
      if (!values) return std::unexpected(values.error());
      values->push_back(
          spec::SourceValue{.key = "llmp.qwen4exp.indexer", .value = std::string("fused")});
      values->push_back(spec::SourceValue{.key = "llmp.qwen4exp.experts_stacked", .value = false});
      values->push_back(spec::SourceValue{.key = "llmp.qwen4exp.expert_scales", .value = false});
      return values;
    }
    case ProfileId::kQwenMlx: {
      const model::Qwen38PleHash hash{
          .multipliers = {23703573157769, 20109073645365, 8052911324071},
          .offsets = {0, 20000003, 40000026, 60000059, 80000106, 100000165, 120000228, 140000297,
                      160000374, 180000455, 200000548, 220000655, 240000802, 260000955, 280001114,
                      300001275},
          .vocab = {20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077,
                    20000081, 20000093, 20000107, 20000147, 20000153, 20000159, 20000161,
                    20000171}};
      model::Qwen38Profile p = model::Qwen38Flash();
      p.ple_row = 256;
      auto values = spec::mappings::Qwen38GgufValues(p, hash);
      if (!values) return std::unexpected(values.error());
      values->push_back(
          spec::SourceValue{.key = "llmp.qwen4exp.indexer", .value = std::string("fused")});
      values->push_back(spec::SourceValue{.key = "llmp.qwen4exp.experts_stacked", .value = false});
      values->push_back(spec::SourceValue{.key = "llmp.qwen4exp.expert_scales", .value = false});
      return values;
    }
    case ProfileId::kGemma2:
      return spec::mappings::Gemma2V0Values(model::Gemma2_2B());
    case ProfileId::kGemma3:
      return spec::mappings::Gemma3V0Values(model::Gemma3_4BQat());
    case ProfileId::kGemma26:
    case ProfileId::kGemma31:
      return spec::mappings::Gemma4V0Values(*GemmaProfile(profile));
    case ProfileId::kGemma4Assistant26:
      return spec::mappings::Gemma4AssistantV0Values(model::Gemma4Assistant26(),
                                                     model::Gemma4_26BA4B());
    case ProfileId::kGemma4Assistant31:
      return spec::mappings::Gemma4AssistantV0Values(model::Gemma4Assistant31(),
                                                     model::Gemma4_31B());
    case ProfileId::kDsparkGguf:
      return spec::mappings::DsparkV0Values(model::DsparkDeepSeekV4Flash());
  }
  return Refused("unknown profile");
}
std::expected<void, std::string> Inventory(const spec::Source& source, const Profile& profile) {
  const bool exl3 =
      profile.id == ProfileId::kQwen2Exl3_4_0 || profile.id == ProfileId::kQwen2Exl3_4_5;
  const bool modelopt = profile.id == ProfileId::kQwenModelOpt;
  const bool mlx = profile.id == ProfileId::kQwenMlx;
  std::vector<const spec::SourceTensor*> sorted;
  sorted.reserve(source.tensors.size());
  for (const auto& t : source.tensors) {
    const auto& f = t.format;
    if (!format::Validate(f) || f.stacked != 1)
      return Refused("native tensor layout, part, parameters or expert geometry differs");
    if (exl3) {
      const bool trellis = f.encoding == format::Encoding::kExl3Trellis;
      const auto bits = f.params.k_bits;
      if (t.experts ||
          (trellis
               ? (f.layout != format::Layout::kExl3Tiles || f.shape.rank != 2 ||
                  t.part == format::PartRole::kData ||
                  f.params.codebook != format::Exl3Codebook::kMcg || f.params.tensor_scale ||
                  (bits != 4 && bits != 5 && bits != 6 && bits != 8))
               : ((f.encoding != format::Encoding::kBF16 && f.encoding != format::Encoding::kF16) ||
                  f.layout != format::Layout::kDense || t.part != format::PartRole::kData ||
                  f.params != format::EncodingParams{} || f.shape.rank < 1 || f.shape.rank > 2)))
        return Refused("native tensor layout, part, parameters or expert geometry differs");
    } else if (modelopt) {
      const bool dense = f.layout == format::Layout::kDense;
      const bool split = f.layout == format::Layout::kSplit;
      if (dense) {
        if (f.encoding != format::Encoding::kBF16 || t.part != format::PartRole::kData ||
            f.params != format::EncodingParams{} || t.experts || f.shape.rank < 1 ||
            f.shape.rank > 2)
          return Refused("native tensor layout, part, parameters or expert geometry differs");
      } else if (split) {
        const bool nv = f.encoding == format::Encoding::kNvfp4;
        const bool mx = f.encoding == format::Encoding::kMxfp8;
        if ((!nv && !mx) ||
            (t.part != format::PartRole::kCodes && t.part != format::PartRole::kBlockScales &&
             t.part != format::PartRole::kTensorScale) ||
            f.params.tensor_scale != nv || f.params.k_bits != 0 ||
            f.params.codebook != format::Exl3Codebook::kNone ||
            (t.experts ? (f.shape.rank != 3 || f.shape.ne[2] != 512) : (f.shape.rank != 2)))
          return Refused("native tensor layout, part, parameters or expert geometry differs");
      } else {
        return Refused("native tensor layout, part, parameters or expert geometry differs");
      }
    } else if (mlx) {
      const bool dense = f.layout == format::Layout::kDense;
      const bool split = f.layout == format::Layout::kSplit;
      if (dense) {
        if (f.encoding != format::Encoding::kBF16 || t.part != format::PartRole::kData ||
            f.params != format::EncodingParams{} || t.experts || f.shape.rank < 1 ||
            f.shape.rank > 2)
          return Refused("native tensor layout, part, parameters or expert geometry differs");
      } else if (split) {
        if (f.encoding != format::Encoding::kAffine ||
            (t.part != format::PartRole::kCodes && t.part != format::PartRole::kBlockScales &&
             t.part != format::PartRole::kBiases) ||
            f.params.bits != 4 || f.params.group != 32 ||
            f.params.scale != format::AffineScale::kBF16 || f.params.zero_point ||
            f.params.tensor_scale || f.params.k_bits != 0 ||
            f.params.codebook != format::Exl3Codebook::kNone ||
            (t.experts ? (f.shape.rank != 3 || f.shape.ne[2] != 512) : (f.shape.rank != 2)))
          return Refused("native tensor layout, part, parameters or expert geometry differs");
      } else {
        return Refused("native tensor layout, part, parameters or expert geometry differs");
      }
    } else if (t.part != format::PartRole::kData || f.params != format::EncodingParams{} ||
               f.shape.rank < 1 || f.shape.rank > 3 || t.experts != (f.shape.rank == 3) ||
               f.layout != (format::Info(f.encoding).quantized ? format::Layout::kGgmlBlocks
                                                               : format::Layout::kDense)) {
      return Refused("native tensor layout, part, parameters or expert geometry differs");
    }
    sorted.push_back(&t);
  }
  std::ranges::sort(sorted, {}, [](const auto* tensor) { return tensor->name; });
  base::Sha256 hash;
  for (const auto* tensor : sorted) {
    hash.Update(tensor->name)
        .Update("\n")
        .Update(format::Info(tensor->format.encoding).name)
        .Update("\n");
    if (exl3)
      hash.Update(format::PartRoleName(tensor->part))
          .Update(":")
          .Update(std::to_string(tensor->format.params.k_bits))
          .Update("\n");
    for (std::uint8_t i = 0; i < tensor->format.shape.rank; ++i)
      hash.Update(std::to_string(tensor->format.shape.ne[i])).Update(",");
    hash.Update("\n");
  }
  const auto actual_hash = base::ToHex(hash.Finish());
  if (actual_hash != profile.inventory)
    return Refused(std::format(
        "native tensor inventory differs from the approved checkpoint (got {}, expected {})",
        actual_hash, profile.inventory));
  return {};
}

}  // namespace

// Size anchors for the closed serving mappings' transient settings synthesis.
// No graph, tensor payload or execution owner is constructed here.
static_assert(sizeof(spec::Param) <= 128 && sizeof(spec::WeightBinding) <= 128);
static_assert(sizeof(spec::ComponentRef) <= 128 && sizeof(spec::SourceTensor) <= 256);
static_assert(sizeof(spec::V0Tensor) <= 128 && sizeof(spec::Stage) <= 2048);

bool OnComponentEngine(std::string_view architecture) {
  return architecture == "gemma2" || architecture == "gemma3" || architecture == "gemma4" ||
         architecture == "gemma4-assistant";
}

std::expected<std::vector<spec::SourceValue>, std::string> ComponentV0Values(
    const artifact::Artifact& artifact) {
  const auto& architecture = artifact.model().architecture;
  if (architecture == "gemma2") return spec::mappings::Gemma2V0Values(model::Gemma2_2B());
  if (architecture == "gemma3") return spec::mappings::Gemma3V0Values(model::Gemma3_4BQat());
  if (architecture == "gemma4") {
    auto profile = ApprovedGemmaProfile(artifact);
    if (!profile) return std::unexpected(profile.error());
    return spec::mappings::Gemma4V0Values(**profile);
  }
  return std::unexpected(
      std::format("the component engine has no recipe for architecture {} yet", architecture));
}

std::expected<std::vector<spec::SourceValue>, std::string> ComponentDrafterV0Values(
    const artifact::Artifact& target, const artifact::Artifact& drafter) {
  if (drafter.model().architecture == "gemma4-assistant" &&
      target.model().architecture == "gemma4") {
    auto profile = ApprovedGemmaProfile(target);
    if (!profile) return std::unexpected(profile.error());
    const model::Gemma4AssistantProfile& assistant =
        *profile == &model::Gemma4_31B() ? model::Gemma4Assistant31() : model::Gemma4Assistant26();
    if (auto bound = model::BindGemma4Assistant(assistant, drafter); !bound) {
      return std::unexpected(bound.error());
    }
    return spec::mappings::Gemma4AssistantV0Values(assistant, **profile);
  }
  if (drafter.model().architecture == "dflash" && target.model().architecture == "deepseek4") {
    const auto& profile = model::Dsv4Flash();
    auto target_binding = model::BindDsv4(profile, target);
    if (!target_binding) return std::unexpected(target_binding.error());
    const auto& companion = model::DsparkDeepSeekV4Flash();
    if (auto bound = model::BindDspark(companion, drafter, profile, *target_binding); !bound) {
      return std::unexpected(bound.error());
    }
    return spec::mappings::DsparkV0Values(companion);
  }
  return std::unexpected(std::format("the component engine has no drafter {} for architecture {}",
                                     drafter.model().architecture, target.model().architecture));
}

std::expected<std::vector<spec::SourceValue>, std::string> ComponentDrafterValues(
    const artifact::Artifact& target, const artifact::Artifact& drafter) {
  if (target.format_version() == 0 && drafter.format_version() == 0) {
    return ComponentDrafterV0Values(target, drafter);
  }
  if (drafter.model().architecture == "gemma4-assistant" &&
      target.model().architecture == "gemma4") {
    const model::Gemma4Profile* target_profile = nullptr;
    if (target.format_version() == 1) {
      const auto* p = Approved(target.model().architecture, target.sources());
      if (p == nullptr)
        return Refused("target source identity has no approved v1 execution profile");
      target_profile = GemmaProfile(*p);
    } else {
      auto p = ApprovedGemmaProfile(target);
      if (!p) return std::unexpected(p.error());
      target_profile = *p;
    }
    if (target_profile == nullptr) return Refused("unknown gemma4 target profile");
    const model::Gemma4AssistantProfile& assistant = *target_profile == model::Gemma4_31B()
                                                         ? model::Gemma4Assistant31()
                                                         : model::Gemma4Assistant26();
    if (drafter.format_version() == 1) {
      const auto* dp = Approved(drafter.model().architecture, drafter.sources());
      if (dp == nullptr)
        return Refused("drafter source identity has no approved v1 execution profile");
      if (*target_profile == model::Gemma4_31B() && dp->id != ProfileId::kGemma4Assistant31) {
        return Refused("the approved assistant does not match the target model size");
      }
      if (*target_profile == model::Gemma4_26BA4B() && dp->id != ProfileId::kGemma4Assistant26) {
        return Refused("the approved assistant does not match the target model size");
      }
      return spec::mappings::Gemma4AssistantV0Values(assistant, *target_profile);
    }
    return ComponentDrafterV0Values(target, drafter);
  }
  if (drafter.model().architecture == "dflash" && target.model().architecture == "deepseek4") {
    if (drafter.format_version() == 1) {
      const auto* dp = Approved(drafter.model().architecture, drafter.sources());
      if (dp == nullptr)
        return Refused("drafter source identity has no approved v1 execution profile");
      if (dp->id != ProfileId::kDsparkGguf) {
        return Refused("unknown dflash profile");
      }
      return spec::mappings::DsparkV0Values(model::DsparkDeepSeekV4Flash());
    }
    return ComponentDrafterV0Values(target, drafter);
  }
  return std::unexpected(std::format("the component engine has no drafter {} for architecture {}",
                                     drafter.model().architecture, target.model().architecture));
}

std::expected<void, std::string> CheckComponentDrafterAdmission(
    const spec::Source& source, std::span<const artifact::Source> origins) {
  const auto* profile = Approved(source.architecture, origins);
  if (profile == nullptr) return Refused("source identity has no approved v1 execution profile");
  if (profile->id != ProfileId::kGemma4Assistant26 &&
      profile->id != ProfileId::kGemma4Assistant31 && profile->id != ProfileId::kDsparkGguf) {
    return Refused("not an approved drafter execution profile");
  }
  const std::string_view expected_container = "gguf";
  if (source.container != expected_container || !source.converter.empty() ||
      !source.folds.empty() || source.tensors.size() != profile->tensors)
    return Refused("source container, transforms or tensor inventory differs from its profile");
  if (auto inventory = Inventory(source, *profile); !inventory) return inventory;
  return {};
}

std::expected<void, std::string> CheckComponentAdmission(
    const spec::ModelSpec& model, const spec::Source& source,
    std::span<const artifact::Source> origins) {
  const auto* profile = Approved(source.architecture, origins);
  if (profile == nullptr) return Refused("source identity has no approved v1 execution profile");
  const bool exl3 =
      profile->id == ProfileId::kQwen2Exl3_4_0 || profile->id == ProfileId::kQwen2Exl3_4_5;
  const bool modelopt = profile->id == ProfileId::kQwenModelOpt;
  const bool mlx = profile->id == ProfileId::kQwenMlx;
  const std::string_view expected_container =
      exl3 ? "exl3" : (modelopt ? "safetensors" : (mlx ? "mlx" : "gguf"));
  const std::vector<std::string> expected_folds =
      (modelopt || mlx) ? std::vector<std::string>{"gdn.value_heads_tiled"}
                        : std::vector<std::string>{};
  if (source.container != expected_container || !source.converter.empty() ||
      source.folds != expected_folds || source.tensors.size() != profile->tensors)
    return Refused("source container, transforms or tensor inventory differs from its profile");
  if (model.pipeline != spec::PipelineKind::kLlm || model.stages.size() != 1 ||
      model.stages[0].role != spec::StageRole::kTarget || model.stages[0].artifact != 0)
    return Refused("the stored pipeline is not an approved standalone target");
  if (auto inventory = Inventory(source, *profile); !inventory) return inventory;
  const auto& stage = model.stages.front();
  if (profile->id != ProfileId::kQwen2Gguf && !exl3 && profile->id != ProfileId::kQwenGguf &&
      profile->id != ProfileId::kQwenModelOpt && profile->id != ProfileId::kQwenMlx &&
      profile->id != ProfileId::kDsv4Gguf &&
      (!stage.embedding || !stage.heads.output || stage.embedding->weights.size() != 1 ||
       stage.heads.output->weights.size() != 1 ||
       stage.embedding->weights.front().tensor != stage.heads.output->weights.front().tensor))
    return Refused("the approved target requires its output tied to its embedding");
  const auto values = Values(*profile);
  if (!values) return Refused(values.error());
  // Profile constants are an approval oracle only. Execute the authenticated
  // stored spec unchanged; never replace a disagreeing spec by this expected one.
  const auto mapping =
      (profile->id == ProfileId::kQwenModelOpt || profile->id == ProfileId::kQwenMlx)
          ? &spec::mappings::Qwen38NativeMapping()
          : spec::FindMapping(profile->architecture);
  const auto expected = spec::Synthesize(*mapping, source, *values);
  if (!expected) return Refused(expected.error());
  if (*expected != model) return Refused("stored spec differs from the approved component profile");
  const std::array<std::span<const spec::SourceTensor>, 1> tables{source.tensors};
  if (const auto valid = components::ValidateSpec(model, tables); !valid)
    return Refused(valid.error());
  return {};
}

std::expected<void, std::string> CheckComponentArtifactAdmission(
    const artifact::Artifact& artifact) {
  if (artifact.format_version() != 1) return Refused("requires a native v1 artifact");
  const auto* profile = Approved(artifact.model().architecture, artifact.sources());
  if (profile == nullptr) return Refused("source identity has no approved v1 execution profile");
  const bool modelopt = profile->id == ProfileId::kQwenModelOpt;
  const bool mlx = profile->id == ProfileId::kQwenMlx;
  const bool folds_ok = (modelopt || mlx)
                            ? (artifact.folds().size() == 1 &&
                               artifact.folds().front().name == "gdn.value_heads_tiled")
                            : artifact.folds().empty();
  if (!folds_ok || !artifact.relayouts().empty() ||
      artifact.resources().size() != profile->tensors - profile->arrays ||
      artifact.expert_arrays().size() != profile->arrays)
    return Refused("the approved target requires its copy-only native tensor inventory");
  return {};
}

std::expected<ComponentAdmission, std::string> ReadComponentAdmission(
    const artifact::Artifact& artifact) {
  if (auto checked = CheckComponentArtifactAdmission(artifact); !checked)
    return std::unexpected(checked.error());
  const auto* profile = Approved(artifact.model().architecture, artifact.sources());
  auto model = spec::StoredSpec(artifact);
  if (!model) return Refused(model.error());
  auto source = spec::SourceOf(artifact);
  if (!source) return Refused(source.error());
  if (auto checked = CheckComponentAdmission(*model, *source, artifact.sources()); !checked)
    return std::unexpected(checked.error());
  return ComponentAdmission{std::move(*model), std::move(*source), GemmaProfile(*profile)};
}

std::expected<spec::PolicyRule, std::string> ComponentAdmissionPolicy(
    const spec::ModelSpec& model, const spec::Source& source,
    std::span<const artifact::Source> origins, const spec::Envelope& envelope) {
  if (auto checked = CheckComponentAdmission(model, source, origins); !checked)
    return std::unexpected(checked.error());
  const auto* profile = Approved(source.architecture, origins);
  if (envelope.context == 0 || envelope.context > profile->max_context || envelope.slots == 0 ||
      envelope.slots > profile->max_slots)
    return Refused("execution envelope exceeds the approved component profile");
  if (profile->id == ProfileId::kGemma3 && envelope.context > 4096 && envelope.slots != 1)
    return Refused("Gemma3 wider context requires exactly one owner");
  if (profile->id == ProfileId::kQwenModelOpt || profile->id == ProfileId::kQwenMlx)
    return spec::PolicyRule{};
  // Reuse the approved mapping's policy. These constants have just been
  // proved equal to the complete stored spec, including whether every layer
  // has its routed block. No checkpoint key bag or v0 descriptor is invented.
  const auto values = Values(*profile);
  if (!values) return Refused(values.error());
  auto rule = spec::DefaultRule(*spec::FindMapping(profile->architecture), *values, envelope);
  if (!rule) return Refused(rule.error());
  // The existing DS mapping deliberately has no policy directives. Preserve
  // exactly that default instead of inventing a stored-only recipe.
  if (*rule == nullptr && profile->id == ProfileId::kDsv4Gguf) return spec::PolicyRule{};
  if (*rule == nullptr) return Refused("no approved policy for this execution envelope");
  return **rule;
}

}  // namespace llmp::runtime
