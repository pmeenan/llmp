// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// An image pipeline's import (D-089, D-109, D-113): a diffusers pipeline
// whose architecture has an image mapping (spec/image_family.h) becomes one
// artifact a component (the tensors its mapping names, at the family's
// shapes, and no others: a tensor the mapping neither reads nor names as
// left out is refused; with a spec of the component at a minimal
// geometry) and
// a composition that keeps the pipeline's own files and the family spec
// (SynthesizeImageFamily over the components' configs and the kept files
// the mapping reads).

#ifndef LLMP_IMPORT_IMAGE_PIPELINE_H_
#define LLMP_IMPORT_IMAGE_PIPELINE_H_

#include <cstdint>
#include <expected>
#include <span>
#include <string_view>
#include <vector>

#include "import/checkpoint.h"
#include "import/normalize.h"
#include "import/writer.h"
#include "model/image_pipeline.h"
#include "spec/spec.h"

namespace llmp::import {

struct ImagePipelinePlan {
  std::vector<ComponentPlan> components;  // in the mapping's (role) order
  spec::ModelSpec family{};               // the composition's spec
  std::uint64_t payload_bytes = 0;
};

// Whether a pipeline checkpoint's architecture has an image mapping.
bool IsImagePipeline(const Checkpoint& source);

std::expected<NormalizationCost, Error> ImagePipelineBound(const Checkpoint& source);

std::expected<ImagePipelinePlan, Error> PlanImagePipeline(const Checkpoint& source);

namespace image_pipeline_detail {
// Shared component metadata validation, also used by descriptor-only tests:
// every tensor of the component is one of `tensors` or one `ignored` names
// (spec::ImageComponentRule's ignored_tensors), or the plan is refused.
std::expected<ComponentPlan, Error> PlanComponent(const Checkpoint& component,
                                                  std::span<const model::ImageTensor> tensors,
                                                  std::span<const std::string_view> ignored,
                                                  const spec::ModelSpec& spec,
                                                  std::vector<ComponentGroup> groups);
}  // namespace image_pipeline_detail
}  // namespace llmp::import

#endif  // LLMP_IMPORT_IMAGE_PIPELINE_H_
