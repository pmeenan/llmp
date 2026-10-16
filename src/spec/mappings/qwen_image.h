// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Qwen-Image-2.1's pipeline (QwenImage21Pipeline) as an image mapping
// (spec/image_family.h): its components' roles and architectures, where
// each family parameter comes from in its configs (or, where the source
// library fixes it in code, the constant), what it requires besides, its
// checkpoints' tensor names, its planning policy and its serving recipe.
#ifndef LLMP_SPEC_MAPPINGS_QWEN_IMAGE_H_
#define LLMP_SPEC_MAPPINGS_QWEN_IMAGE_H_

#include "spec/image_family.h"

namespace llmp::spec::mappings {
const ImageMapping& QwenImage21Mapping();
}  // namespace llmp::spec::mappings
#endif  // LLMP_SPEC_MAPPINGS_QWEN_IMAGE_H_
