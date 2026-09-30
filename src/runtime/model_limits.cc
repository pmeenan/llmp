// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "runtime/model_limits.h"

#include <format>

#include "config/node_config.h"
#include "model/dsv4.h"
#include "model/qwen38.h"

namespace jitllm::runtime {

std::expected<void, std::string> CheckModelContext(std::string_view architecture,
                                                   std::uint32_t context) {
  std::uint32_t ceiling = 0;
  if (architecture == "deepseek4") {
    ceiling = model::kDsv4FlashContext;
  } else if (architecture == "qwen4exp") {
    ceiling = model::kQwen38FlashContext;
  } else {
    return std::unexpected(std::format("no runner for architecture {}", architecture));
  }
  if (context < config::kMinContext || context > ceiling) {
    return std::unexpected(std::format("context {} is outside {}'s supported range {} to {}",
                                       context, architecture, config::kMinContext, ceiling));
  }
  return {};
}

}  // namespace jitllm::runtime
