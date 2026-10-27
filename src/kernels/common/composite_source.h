// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// One SelectedSource over several (D-053): the build's kernel sources lent
// to a step backend as one. Each operation lowers in the first child that
// owns its implementation; each child's launch and persistent workspace are
// its own. Vendor-free: the composer names no kernel source.
#ifndef LLMP_KERNELS_COMMON_COMPOSITE_SOURCE_H_
#define LLMP_KERNELS_COMMON_COMPOSITE_SOURCE_H_

#include <memory>
#include <vector>

#include "kernels/common/selected_operation.h"

namespace llmp::kernels {

std::shared_ptr<const SelectedSource> CompositeSelectedSource(
    std::vector<std::shared_ptr<const SelectedSource>> sources);

}  // namespace llmp::kernels

#endif  // LLMP_KERNELS_COMMON_COMPOSITE_SOURCE_H_
