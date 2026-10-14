// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
#ifndef LLMP_IMPORT_MLX_QWEN38_H_
#define LLMP_IMPORT_MLX_QWEN38_H_

#include <expected>

#include "import/checkpoint.h"
#include "import/error.h"
#include "import/modelopt_qwen38.h"
#include "import/normalize.h"
#include "spec/spec.h"

namespace llmp::import {

// Metadata-only, allocation-free admission for the closed MLX Qwen3.8 target source domain.
// Caller separately owns/funds Checkpoint and its reader before calling this.
std::expected<NormalizationCost, Error> MlxTargetBound(const Checkpoint& source);

// Reads exactly 280 bytes after admission through authenticated source identities.
// No full payload hashing, publication, runtime admission or MTP/vision import.
std::expected<ModelOptTargetPlan, Error> NormalizeMlxTarget(Checkpoint&& source,
                                                            NormalizationLimits prepaid);

// Canonical native source for the approved MLX Qwen3.8 target profile (2,274 tensors).
spec::Source ApprovedMlxTargetSource();

}  // namespace llmp::import

#endif  // LLMP_IMPORT_MLX_QWEN38_H_
