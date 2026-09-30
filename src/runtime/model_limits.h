// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// Registration checks the configured context against the supported
// checkpoint before opening the node or allocating a model's resources.

#ifndef JITLLM_RUNTIME_MODEL_LIMITS_H_
#define JITLLM_RUNTIME_MODEL_LIMITS_H_

#include <cstdint>
#include <expected>
#include <string>
#include <string_view>

namespace jitllm::runtime {

std::expected<void, std::string> CheckModelContext(std::string_view architecture,
                                                   std::uint32_t context);

}  // namespace jitllm::runtime

#endif  // JITLLM_RUNTIME_MODEL_LIMITS_H_
