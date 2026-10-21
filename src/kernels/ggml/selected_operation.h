// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Build-time kernel-source adapters inside the shared ordered step program.
// This seam owns no model graph and never chooses a replacement operation:
// the IR planner has already selected the exact implementation. GGML's
// private descriptors retain the global placement and every part dependency.
#ifndef LLMP_KERNELS_GGML_SELECTED_OPERATION_H_
#define LLMP_KERNELS_GGML_SELECTED_OPERATION_H_

#include "ggml.h"
#include "kernels/common/selected_operation.h"
#include "kernels/ggml/tensors.h"

namespace llmp::kernels::ggml {

using kernels::BoundSelectedCall;
using kernels::KernelError;
using kernels::KernelFailure;
using kernels::SelectedCall;
using kernels::SelectedLaunch;
using kernels::SelectedNode;
using kernels::SelectedOperand;
using kernels::SelectedPart;
using kernels::SelectedRoute;
using kernels::SelectedRouteBuffers;
using kernels::SelectedSource;
using kernels::SelectedWorkspace;

}  // namespace llmp::kernels::ggml

#endif  // LLMP_KERNELS_GGML_SELECTED_OPERATION_H_
