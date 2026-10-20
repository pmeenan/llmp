// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Qwen-Image-2.1's family for tests (spec/image_family.h): its mapping, and
// the family spec its published v1 composition stores, frozen in-tree as
// tests/unit/data/v1_frozen/qwen_image.spec.json (the composition's
// spec.json byte for byte, whose digest the pipeline's approval pins), read
// back through the mapping as the runtime reads it. No test compiles in the
// family's numbers; a test that cannot read the file stops with the reason.
//
// Reading: LLMP_TEST_DATA/v1_frozen/qwen_image.spec.json, else ctest's
// working directory's data/.

#ifndef LLMP_TESTS_SUPPORT_IMAGE_FAMILY_H_
#define LLMP_TESTS_SUPPORT_IMAGE_FAMILY_H_

#include <cstdio>
#include <cstdlib>
#include <expected>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "model/image_pipeline.h"
#include "spec/image_family.h"
#include "spec/mappings/qwen_image.h"
#include "spec/spec.h"

namespace llmp::test_support {

struct QwenImageFamily {
  const spec::ImageMapping* mapping = nullptr;
  spec::ModelSpec spec{};
  model::ImageProfile profile{};
};

inline std::string QwenImageSpecPath() {
  const char* data = std::getenv("LLMP_TEST_DATA");  // NOLINT(concurrency-mt-unsafe)
  return std::string(data != nullptr ? data : "data") + "/v1_frozen/qwen_image.spec.json";
}

inline std::expected<QwenImageFamily, std::string> ReadQwenImageFamily() {
  const auto path = QwenImageSpecPath();
  std::ifstream in(path, std::ios::binary);
  if (!in) return std::unexpected("cannot read " + path);
  std::ostringstream text;
  text << in.rdbuf();
  auto spec = spec::FromJson(text.str());
  if (!spec) return std::unexpected(path + ": " + spec.error());
  if (spec::ToJson(*spec) != text.str())
    return std::unexpected(path + ": not the spec's own canonical JSON");
  const auto& mapping = spec::mappings::QwenImage21Mapping();
  auto profile = spec::ImageProfileOf(*spec, mapping);
  if (!profile) return std::unexpected(path + ": " + profile.error());
  return QwenImageFamily{
      .mapping = &mapping, .spec = std::move(*spec), .profile = std::move(*profile)};
}

// The family, read once.
inline const QwenImageFamily& QwenImage() {
  static const QwenImageFamily family = [] {
    auto read = ReadQwenImageFamily();
    if (!read) {
      std::fprintf(stderr, "Qwen-Image family: %s\n", read.error().c_str());
      std::abort();
    }
    return std::move(*read);
  }();
  return family;
}
inline const model::ImageProfile& QwenImageProfile() { return QwenImage().profile; }
inline const model::ImageNames& QwenImageNames() { return QwenImage().mapping->names; }
// The pipeline's planning policy (its mapping's).
inline std::string_view QwenImagePolicy() {
  return QwenImage().mapping->serving->serving_rule->text;
}

inline std::vector<model::ImageTensor> QwenImageTextTensors() {
  return model::TextEncoderTensors(QwenImageProfile().text, QwenImageNames().text);
}
inline std::vector<model::ImageTensor> QwenImageDenoiserTensors() {
  return model::DenoiserTensors(QwenImageProfile().denoiser, QwenImageNames().denoiser);
}
inline std::vector<model::ImageTensor> QwenImageVaeTensors() {
  return model::VaeDecoderTensors(QwenImageProfile().vae, QwenImageNames().vae);
}

}  // namespace llmp::test_support

#endif  // LLMP_TESTS_SUPPORT_IMAGE_FAMILY_H_
