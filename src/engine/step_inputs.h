// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#ifndef LLMP_ENGINE_STEP_INPUTS_H_
#define LLMP_ENGINE_STEP_INPUTS_H_

#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <string>

#include "kernels/common/step_program.h"
#include "planner/step_shape.h"

namespace llmp::engine::step_inputs {

// Canonical host staging shared by ordinary steps and internal commits.
std::expected<void, std::string> WriteRows(std::span<std::byte> out,
                                           std::span<const std::int32_t> values);
std::expected<void, std::string> WriteComputed(const kernels::InputSlot& input,
                                               const planner::StepShape& shape,
                                               std::span<std::byte> out);

}  // namespace llmp::engine::step_inputs

#endif  // LLMP_ENGINE_STEP_INPUTS_H_
