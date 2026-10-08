// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The paged node (engine/paged_node.h) under the names the paged harnesses
// and their tests use.

#ifndef LLMP_TESTS_SUPPORT_PAGED_NODE_H_
#define LLMP_TESTS_SUPPORT_PAGED_NODE_H_

#include "engine/paged_node.h"
#include "paged_programs.h"

namespace llmp::test_support {

using engine::CountingStorage;
using engine::kPagedDepth;
using engine::kPagedExtent;
using engine::kPagedSlots;
using engine::kShared;
using engine::LoadStats;
using engine::Mapped;
using engine::NodeSettings;
using engine::PagedModel;
using engine::PagedNode;
using engine::ReleaseMapped;
using engine::Span;
using engine::Status;
using engine::StepTimes;

}  // namespace llmp::test_support

#endif  // LLMP_TESTS_SUPPORT_PAGED_NODE_H_
