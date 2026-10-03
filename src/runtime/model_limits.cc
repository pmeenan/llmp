// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "runtime/model_limits.h"

#include <vector>

#include "model/dsv4.h"

namespace jitllm::runtime {

bool Dsv4FrontierHeadForServing(const model::Dsv4Binding& binding) {
  return binding.output.type == "Q4_K" &&
         binding.output.ne == std::vector<std::uint64_t>{4096, 129280} &&
         binding.hc_head_fn.type == "F32" &&
         binding.hc_head_fn.ne == std::vector<std::uint64_t>{16384, 4};
}

}  // namespace jitllm::runtime
