// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#ifndef LLMP_IMPORT_QWEN_IMAGE_H_
#define LLMP_IMPORT_QWEN_IMAGE_H_

#include <array>
#include <cstdint>
#include <expected>

#include "import/checkpoint.h"
#include "import/normalize.h"
#include "import/writer.h"

namespace llmp::import {

struct QwenImagePipelinePlan {
  std::array<ComponentPlan, 3> components;
  std::uint64_t payload_bytes = 0;
};

std::expected<NormalizationCost, Error> QwenImagePipelineBound(const Checkpoint& source);

std::expected<QwenImagePipelinePlan, Error> PlanQwenImagePipeline(const Checkpoint& source);

}  // namespace llmp::import

#endif  // LLMP_IMPORT_QWEN_IMAGE_H_
