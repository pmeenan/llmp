// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The M3 per-family binders (Qwen2 and its EXL3 checkpoints, DeepSeek V4,
// Qwen3.8, Qwen-Image-2.1, Gemma 2, 3 and 4 and the Gemma 4 assistant) read
// v0's representation descriptors, which a version 1 artifact leaves empty
// (D-113). Each refuses any other format version where it accepts an
// artifact, naming the version, rather than misreading it; a family reads
// v1 artifacts once it runs on the component engine (spec/stored.h).

#ifndef LLMP_MODEL_V0_ARTIFACT_H_
#define LLMP_MODEL_V0_ARTIFACT_H_

#include <expected>
#include <format>
#include <string>
#include <string_view>

#include "artifact/artifact.h"

namespace llmp::model {

inline std::expected<void, std::string> RequireV0Artifact(const artifact::Artifact& artifact,
                                                          std::string_view family) {
  if (artifact.format_version() == 0) {
    return {};
  }
  return std::unexpected(std::format(
      "{} reads version 0 prepared artifacts only; artifact {} is format version {} (D-113), "
      "which this family reads once it runs on the component engine",
      family, artifact.id(), artifact.format_version()));
}

}  // namespace llmp::model

#endif  // LLMP_MODEL_V0_ARTIFACT_H_
