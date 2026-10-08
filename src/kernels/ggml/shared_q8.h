// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// Descriptor-only, per-graph preparation cache. The normal graph planner owns
// and funds each producer's activation; tensor identity is never inferred from
// names, shape, values or an address shared by two different descriptors.
#ifndef JITLLM_KERNELS_GGML_SHARED_Q8_H_
#define JITLLM_KERNELS_GGML_SHARED_Q8_H_

#include <cstdint>
#include <functional>
#include <unordered_map>

#include "kernels/ggml/jitllm_ops.h"

namespace jitllm::kernels::ggml {
class SharedQ8Inputs {
 public:
  explicit SharedQ8Inputs(ggml_context* context) : context_(context) {}
  ggml_tensor* Get(ggml_tensor* input) {
    if (const auto at = inputs_.find(input); at != inputs_.end()) return at->second;
    auto* prepared = QuantizeQ8(context_, input);
    inputs_.emplace(input, prepared);
    return prepared;
  }
  // The device selector is called only during CPU graph construction and is
  // never retained. Empty or MMQ selection leaves the ordinary graph intact,
  // before allocating any prepared node. Larger MMVQ batches remain bounded by
  // the original launcher and this qualified eight-column descriptor envelope.
  ggml_tensor* Product(ggml_tensor* weight, ggml_tensor* input, bool enabled,
                       const std::function<bool(ggml_type, std::int64_t)>& selected = {}) {
    const bool type = weight->type == GGML_TYPE_Q4_0 || weight->type == GGML_TYPE_Q4_K ||
                      weight->type == GGML_TYPE_Q5_K || weight->type == GGML_TYPE_Q6_K ||
                      weight->type == GGML_TYPE_Q8_0 || weight->type == GGML_TYPE_Q4_1 ||
                      weight->type == GGML_TYPE_Q5_0 || weight->type == GGML_TYPE_Q5_1 ||
                      weight->type == GGML_TYPE_IQ4_NL;
    if (!enabled || !type || input->type != GGML_TYPE_F32 || input->ne[0] != weight->ne[0] ||
        input->ne[0] <= 0 || input->ne[0] % 32 != 0 || input->ne[0] > INT32_MAX - 511 ||
        input->ne[1] < 1 || input->ne[1] > 8 || input->ne[2] != 1 || input->ne[3] != 1 ||
        !ggml_is_contiguous(input) || weight->op != GGML_OP_NONE || weight->view_src != nullptr ||
        weight->ne[2] != 1 || weight->ne[3] != 1 || !ggml_is_contiguous(weight) || !selected ||
        !selected(weight->type, input->ne[1]))
      return ggml_mul_mat(context_, weight, input);
    return MmvqPrepared(context_, weight, Get(input), input);
  }

 private:
  ggml_context* context_;
  std::unordered_map<ggml_tensor*, ggml_tensor*> inputs_;
};
}  // namespace jitllm::kernels::ggml
#endif  // JITLLM_KERNELS_GGML_SHARED_Q8_H_
