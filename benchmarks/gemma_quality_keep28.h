// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#ifndef JITLLM_BENCHMARKS_GEMMA_QUALITY_KEEP28_H_
#define JITLLM_BENCHMARKS_GEMMA_QUALITY_KEEP28_H_

#include <optional>

namespace jitllm::engine::diagnostic {
// Read once before Setup: invalid or missing process environment refuses the
// closed diagnostic. A valid false value preserves the original caller keep.
std::optional<bool> Keep28Routing();
}  // namespace jitllm::engine::diagnostic
#endif  // JITLLM_BENCHMARKS_GEMMA_QUALITY_KEEP28_H_
