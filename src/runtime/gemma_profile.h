// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#ifndef JITLLM_RUNTIME_GEMMA_PROFILE_H_
#define JITLLM_RUNTIME_GEMMA_PROFILE_H_

#include <expected>
#include <span>
#include <string>
#include <string_view>

#include "model/gemma4.h"

namespace jitllm::runtime {
// Expert count only chooses a candidate; the complete approved binding must
// authenticate it before a serving runner is constructed. No filename/profile
// override is accepted. Returned profiles are the immutable model constants.
std::expected<const model::Gemma4Profile*, std::string> ApprovedGemmaProfile(
    const artifact::Artifact& artifact);
std::expected<const model::Gemma4Profile*, std::string> ApprovedGemmaProfile(
    std::string_view architecture, std::uint32_t experts,
    std::span<const model::Gemma4Resource> resources);
}  // namespace jitllm::runtime
#endif  // JITLLM_RUNTIME_GEMMA_PROFILE_H_
