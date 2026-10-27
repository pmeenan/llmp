// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The Qwen-Image VAE decoder (diffusers' AutoencoderKLQwenImage and its
// Qwen-Image-2.1 variant) as one configured component mapping: one stage
// recipe whose values come from one environment (the pipeline family's
// image.vae_decoder parameters where a composition imports it, else its
// config.json's keys), in two source dialects the checkpoint itself
// selects. Kernel rank selects the convolutions: 2-D kernels, or the causal
// 3-D kernels a single frame reads only the last temporal tap of
// (conv.spatial's temporal_kernel). The config's is_residual selects the
// up path: a residual decoder's duplicating up blocks, or one whose
// upsamplers halve their channels. A composition's VAE (Qwen-Image-2.1's,
// spec/mappings/qwen_image.h) and a standalone VAE import (Ming-Image's)
// both import through it.

#ifndef LLMP_SPEC_MAPPINGS_QWEN_IMAGE_VAE_H_
#define LLMP_SPEC_MAPPINGS_QWEN_IMAGE_VAE_H_

#include "spec/mapping.h"

namespace llmp::spec::mappings {

const Mapping& QwenImageVaeMapping();

}  // namespace llmp::spec::mappings

#endif  // LLMP_SPEC_MAPPINGS_QWEN_IMAGE_VAE_H_
