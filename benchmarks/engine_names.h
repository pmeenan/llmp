// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The engine's model runners and planning (src/engine/), which the M3
// harnesses drive, under the names the harnesses have always used. CUDA
// builds only.

#ifndef LLMP_BENCHMARKS_ENGINE_NAMES_H_
#define LLMP_BENCHMARKS_ENGINE_NAMES_H_

#include "engine/dsv4_plan.h"
#include "engine/dsv4_runner.h"
#include "engine/qwen38_plan.h"
#include "engine/qwen38_runner.h"
#include "engine/qwen_image_runner.h"

namespace llmp::benchmarks {

using engine::Argmax;
using engine::Dsv4ChunkKind;
using Dsv4GraphStats = engine::GraphStats;  // the harnesses' old name
using engine::Dsv4Options;
using Dsv4Path = engine::RunPath;  // the harnesses' old name
using engine::Dsv4Runner;
using engine::kSlabSlotBytes;
using engine::PlanQwen38Chunk;
using engine::PleStats;
using engine::Qwen38HostInputs;
using engine::Qwen38Model;
using engine::Qwen38Options;
using engine::Qwen38Planned;
using engine::Qwen38Runner;
using engine::Qwen38Sources;
using engine::QwenImageOptions;
using engine::QwenImageRunner;

}  // namespace llmp::benchmarks

#endif  // LLMP_BENCHMARKS_ENGINE_NAMES_H_
