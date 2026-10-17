// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Internal inventory validation shared by admission and descriptor tests.
#ifndef LLMP_RUNTIME_COMPONENT_ADMISSION_INTERNAL_H_
#define LLMP_RUNTIME_COMPONENT_ADMISSION_INTERNAL_H_

#include <expected>
#include <string>

#include "runtime/approvals.h"
#include "spec/mapping.h"

namespace llmp::runtime::internal {

// Check native structure and the complete inventory digest against an
// explicit approval. This does not admit a model: production admission
// must resolve its source identity through the immutable approval table
// and check its stored spec, transforms and execution envelope separately.
std::expected<void, std::string> CheckInventory(const spec::Source& source,
                                                const Approval& approval);

}  // namespace llmp::runtime::internal
#endif  // LLMP_RUNTIME_COMPONENT_ADMISSION_INTERNAL_H_
