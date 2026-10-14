// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The engine's model runners and planning (src/engine/), which the M3
// harnesses drive, under the names the harnesses have always used. CUDA
// builds only.

#ifndef LLMP_BENCHMARKS_ENGINE_NAMES_H_
#define LLMP_BENCHMARKS_ENGINE_NAMES_H_

#include "engine/dsv4_plan.h"
#include "engine/dsv4_runner.h"

namespace llmp::benchmarks {

using engine::Argmax;
using engine::Dsv4ChunkKind;
using Dsv4GraphStats = engine::GraphStats;  // the harnesses' old name
using engine::Dsv4Options;
using Dsv4Path = engine::RunPath;  // the harnesses' old name
using engine::Dsv4Runner;
using engine::kSlabSlotBytes;

}  // namespace llmp::benchmarks

#endif  // LLMP_BENCHMARKS_ENGINE_NAMES_H_
