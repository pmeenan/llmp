// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "runtime/gemma_profile.h"

#include "artifact/artifact.h"

namespace llmp::runtime {
namespace {
std::expected<const model::Gemma4Profile*, std::string> Candidate(std::string_view architecture,
                                                                  std::uint32_t experts) {
  if (architecture != "gemma4") return std::unexpected("not an approved Gemma text architecture");
  if (experts == model::Gemma4_26BA4B().experts) return &model::Gemma4_26BA4B();
  if (experts == model::Gemma4_31B().experts) return &model::Gemma4_31B();
  return std::unexpected("not an approved Gemma26/31 expert count");
}
}  // namespace
std::expected<const model::Gemma4Profile*, std::string> ApprovedGemmaProfile(
    const artifact::Artifact& artifact) {
  auto candidate = Candidate(artifact.model().architecture, artifact.model().expert_count);
  if (!candidate) return std::unexpected(candidate.error());
  if (auto binding = model::BindGemma4(**candidate, artifact); !binding)
    return std::unexpected(binding.error());
  return *candidate;
}
std::expected<const model::Gemma4Profile*, std::string> ApprovedGemmaProfile(
    std::string_view architecture, std::uint32_t experts,
    std::span<const model::Gemma4Resource> resources) {
  auto candidate = Candidate(architecture, experts);
  if (!candidate) return std::unexpected(candidate.error());
  if (auto binding = model::BindGemma4(**candidate, architecture, resources); !binding)
    return std::unexpected(binding.error());
  return *candidate;
}
}  // namespace llmp::runtime
