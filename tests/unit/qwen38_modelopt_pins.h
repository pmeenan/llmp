// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The pinned source files of the ModelOpt Qwen3.8 checkpoint
// (Mia-AiLab/Qwen3.8-Flash-Next-NVFP4@925d7be6): name, byte count and
// SHA-256, in the checkpoint's order. One table for the fixture
// (qwen38_fixture.h ModelOptOrigins) and the opt-in publication test
// (import_modelopt_models_test.cc), which adds only the tokenizer and
// generation files it also authenticates.

#ifndef LLMP_TESTS_UNIT_QWEN38_MODELOPT_PINS_H_
#define LLMP_TESTS_UNIT_QWEN38_MODELOPT_PINS_H_

#include <array>
#include <cstdint>
#include <string_view>
#include <tuple>

namespace llmp::test_support::qwen38 {

inline constexpr std::array<std::tuple<std::string_view, std::uint64_t, std::string_view>, 38>
    kModelOptPins{{
        {"config.json", 109897ULL,
         "2d655dfa625c3c018f9685031367bc83f8532d3faf3333d7c838b2bdc13b8a7e"},
        {"hf_quant_config.json", 99088ULL,
         "d60c54caa65317fd8f1338bdeb81b9014512ac8f21ec18da22541cb88c6784b7"},
        {"model-00001-of-00034.safetensors", 2841505880ULL,
         "3f870511a20ad170c997517765f03d475bcd364ae1a6011491085cab620d3c39"},
        {"model-00002-of-00034.safetensors", 1575009800ULL,
         "fe8f4bdbce28f4911f10460b75581b6c88a8ade9b836d2f1ec22c6bee6f976a8"},
        {"model-00003-of-00034.safetensors", 1575009808ULL,
         "2b7d1312a660cf17bb05c34e1ac79570820f86fb1e1115108ee40562e48acb5e"},
        {"model-00004-of-00034.safetensors", 1575009808ULL,
         "25ba4508c209a5cc54c04edb1edb8d0a0b5efa80e5f99637e2263fa78d6a25e8"},
        {"model-00005-of-00034.safetensors", 1575009808ULL,
         "a9848067bb49f44b8339bcdca9d60ca74d801a6df874c979c6f901c8da6ff545"},
        {"model-00006-of-00034.safetensors", 1575009808ULL,
         "66fc496b58675a90b94d9f36765fe67692dd8a4c966ec0692e076d751f835c8d"},
        {"model-00007-of-00034.safetensors", 1575009808ULL,
         "68f89e84be13069bd9f2928b44a9f9638d628fd62be5abb6a4fc956c1af25ea9"},
        {"model-00008-of-00034.safetensors", 1575009808ULL,
         "b8bdaa0a7b027dc0fa38c2e43bb31e8ddf9ba9288d3d0f28c422e6947241d96e"},
        {"model-00009-of-00034.safetensors", 1575009808ULL,
         "6c5a85990b16a93f0e7e28fb76f56713d5cbc05bef494bb15360188f575746e9"},
        {"model-00010-of-00034.safetensors", 1575009808ULL,
         "8734a1fc3b5857391fbf2252d4305391a9f77894b70577949f472fc75386582f"},
        {"model-00011-of-00034.safetensors", 1575009808ULL,
         "29792378d43c9f70d9f3fbdd07d0ee24b38f81fe27aff25a02ba0d3cde8df13c"},
        {"model-00012-of-00034.safetensors", 1575009808ULL,
         "b23e615ebefff4753d982d9ec6a8be6ad57353a8900146a3ec4e217f81c10e61"},
        {"model-00013-of-00034.safetensors", 1575009808ULL,
         "9da7b4570fc3f13cb781e5aca5e79d0edfc10be5555a18bc7bda8f47459d5acf"},
        {"model-00014-of-00034.safetensors", 1575009808ULL,
         "53e770d0bdf6261e30abaae2477aa0b50f6686511e0240514ded0fd65edbfeba"},
        {"model-00015-of-00034.safetensors", 1575009816ULL,
         "03440de56d127ff58e72963aafd0de0574b825667330c50b4441bb300dae9f06"},
        {"model-00016-of-00034.safetensors", 1575009824ULL,
         "ce2e0474ccb34c56003a61608b46d3a8f668c218a9993c1f861239df83afe77b"},
        {"model-00017-of-00034.safetensors", 1575009824ULL,
         "57f84845e0b8bcab738e70f0f880e941f1baa8da14663293d7d40481288c73a0"},
        {"model-00018-of-00034.safetensors", 1575009824ULL,
         "3477df0be60910576fc5a1f8159d8dbbf7847a03872b9cb53664808e1d5dfac8"},
        {"model-00019-of-00034.safetensors", 2438722732ULL,
         "43c22213b2fe0990c7fc2aa54c0ea65062eb93253b4ea419914d1a655e64b23a"},
        {"model-00020-of-00034.safetensors", 5133924592ULL,
         "b7143694bf8698c5b704ae9b3d492a0e9b8ef57bdda8ae6f3f29f264c183ad17"},
        {"model-00021-of-00034.safetensors", 5194891408ULL,
         "522bbbc894183865726e06ca44021d275d03cc0e8dcd60a2c7074d870e811d37"},
        {"model-00022-of-00034.safetensors", 5133933936ULL,
         "5a4789f6da6b386b09cb9093b0315fe12a816142958e4d8a4ae83e4fbaa066e7"},
        {"model-00023-of-00034.safetensors", 5194908864ULL,
         "cd0aaef1c8688cb4e35b488048ede2a6b653e1d558b20d4657567df24dc8be13"},
        {"model-00024-of-00034.safetensors", 5188755712ULL,
         "d07198edd38ae8740d556487729ae87be1e21e31a88cf6624c1b3f9d0fc6de76"},
        {"model-00025-of-00034.safetensors", 5133940040ULL,
         "c7d5e642968679ca14d07bdb1e8c2187c59898611d05a6909c209738642c9519"},
        {"model-00026-of-00034.safetensors", 5194907416ULL,
         "47a7c7a70188cb3dc318071a5278260c4b05d7ce1b69c3b99e7bf1c0e42c353b"},
        {"model-00027-of-00034.safetensors", 5133940216ULL,
         "a81fedf160434806e81f36731c853fe4e3f973dea2f0817043987f32b4915a54"},
        {"model-00028-of-00034.safetensors", 5194907672ULL,
         "91c5d479fe4de1035c430ad7fd3de9d15214a46fd786ca7909d1f2fa40ef603a"},
        {"model-00029-of-00034.safetensors", 5133939992ULL,
         "f31c6f61968f6f9e85d469f31c55c82ab6c5f2bc54c1d7a0cd6b23ac45f0ce15"},
        {"model-00030-of-00034.safetensors", 5194908960ULL,
         "df8627f5b8b0711b2ebf415c1f2549ef0b50f964e0babb560469b191b33c6a14"},
        {"model-00031-of-00034.safetensors", 5188755536ULL,
         "654c8f9ae1ab5de393cbcdd46a13bd0d3d1c1005e575879e0e660d40ad9ef425"},
        {"model-00032-of-00034.safetensors", 5133940208ULL,
         "7de2e680802ad92a0e019f875ebc07f2d5d82c8bca1c64b0f817d547bc3c9de7"},
        {"model-00033-of-00034.safetensors", 5021547836ULL,
         "0fbcdf16c13cd830447efdcd3be19b82ec0e76b7f57f0e96952c1f2b62002b11"},
        {"model-00034-of-00034.safetensors", 1597270800ULL,
         "fe0a0a227e39c4d02dc66c726d3e1ad87ef082e90a6b023ac90b2bb301672203"},
        {"model-inputscales.safetensors", 9669936ULL,
         "49c7d04606ff7c8f103b4d1fc9742b44649a2937b25ecc896a5450ab1020f308"},
        {"model.safetensors.index.json", 33074286ULL,
         "435eef76fc10fc6e932a208a1f85a86bc9c7ffc389b27b34ba83bb6a5d0371e9"},
    }};

}  // namespace llmp::test_support::qwen38

#endif  // LLMP_TESTS_UNIT_QWEN38_MODELOPT_PINS_H_
