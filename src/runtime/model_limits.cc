// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "runtime/model_limits.h"

#include <format>
#include <vector>

#include "config/node_config.h"
#include "model/dsv4.h"
#include "model/qwen38.h"

namespace jitllm::runtime {

bool Dsv4FrontierHeadForServing(const model::Dsv4Binding& binding) {
  return binding.output.type == "Q4_K" &&
         binding.output.ne == std::vector<std::uint64_t>{4096, 129280} &&
         binding.hc_head_fn.type == "F32" &&
         binding.hc_head_fn.ne == std::vector<std::uint64_t>{16384, 4};
}

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
