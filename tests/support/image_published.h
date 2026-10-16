// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Qwen-Image-2.1's published v1 artifacts for tests: their IDs by the names
// the pinned map tests/unit/data/v1_frozen/v1-ids.json gives them
// ("qwen-image" the composition, "qwen-image-text-encoder",
// "qwen-image-transformer" and "qwen-image-vae" its components), and their
// directories in the published v1 store (v1_frozen.h V1Store).

#ifndef LLMP_TESTS_SUPPORT_IMAGE_PUBLISHED_H_
#define LLMP_TESTS_SUPPORT_IMAGE_PUBLISHED_H_

#include <filesystem>
#include <string>
#include <string_view>

#include "v1_frozen.h"

namespace llmp::test_support {

// The v1 ID the pinned map names `name`; empty when it names none.
inline std::string QwenImageId(std::string_view name) { return V1IdNamed(name).value_or(""); }
inline std::string QwenImageComposition() { return QwenImageId("qwen-image"); }
inline std::string QwenImageText() { return QwenImageId("qwen-image-text-encoder"); }
inline std::string QwenImageTransformer() { return QwenImageId("qwen-image-transformer"); }
inline std::string QwenImageVae() { return QwenImageId("qwen-image-vae"); }

// A published artifact's directory in the v1 store.
inline std::filesystem::path QwenImagePublished(std::string_view id) {
  return std::filesystem::path(V1Store()) / id;
}

}  // namespace llmp::test_support

#endif  // LLMP_TESTS_SUPPORT_IMAGE_PUBLISHED_H_
