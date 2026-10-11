// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "runtime/component_admission.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>
#include <utility>

#include "base/sha256.h"
#include "components/step.h"
#include "model/gemma2.h"
#include "spec/mappings/gemma2.h"
#include "spec/stored.h"

namespace llmp::runtime {
namespace {

// The existing approved execution checkpoint, not the container's artifact
// ID: publication does not broaden the family/profile or format domain.
struct Profile {
  std::string_view architecture, source_name, source_sha256;
  std::uint64_t source_bytes;
  std::size_t tensors;
  std::uint32_t max_context, max_slots;
};
constexpr std::array kProfiles{
    Profile{"gemma2", "gemma-2-2b-it-Q8_0.gguf",
            "2d448a9aab894b8e8e18168cf3f490cb9f65632222f29f93514ac9ecc754debe", 2784495456ULL, 288,
            8192, 2}};

const Profile* Approved(std::string_view architecture, std::span<const artifact::Source> origins) {
  if (origins.size() != 1) return nullptr;
  for (const auto& profile : kProfiles)
    if (architecture == profile.architecture && origins[0].name == profile.source_name &&
        origins[0].bytes.value() == profile.source_bytes &&
        base::ToHex(origins[0].sha256) == profile.source_sha256)
      return &profile;
  return nullptr;
}

std::unexpected<std::string> Refused(std::string_view reason) {
  return std::unexpected("component artifact admission: " + std::string(reason));
}

}  // namespace

std::expected<void, std::string> CheckComponentAdmission(
    const spec::ModelSpec& model, const spec::Source& source,
    std::span<const artifact::Source> origins) {
  const auto* profile = Approved(source.architecture, origins);
  if (profile == nullptr) return Refused("source identity has no approved v1 execution profile");
  if (source.container != "gguf" || !source.converter.empty() || !source.folds.empty() ||
      source.tensors.size() != profile->tensors)
    return Refused("source container, transforms or tensor inventory differs from its profile");
  if (model.pipeline != spec::PipelineKind::kLlm || model.stages.size() != 1 ||
      model.stages[0].role != spec::StageRole::kTarget || model.stages[0].artifact != 0)
    return Refused("the stored pipeline is not an approved standalone target");
  // This declaration keeps the existing Q8_0 matrices and F32 vector norms.
  // Actual dimensions and weight roles are checked by the complete mapping
  // comparison and component validation below, not by tensor-name heuristics.
  for (const auto& tensor : source.tensors) {
    const auto& format = tensor.format;
    const bool matrix = format.shape.rank == 2 && format.encoding == format::Encoding::kQ8_0 &&
                        format.layout == format::Layout::kGgmlBlocks;
    const bool vector = format.shape.rank == 1 && format.encoding == format::Encoding::kF32 &&
                        format.layout == format::Layout::kDense;
    if ((!matrix && !vector) || tensor.experts || tensor.part != format::PartRole::kData ||
        format.stacked != 1 || format.params != format::EncodingParams{})
      return Refused("native tensor encoding, layout or part is outside the approved profile");
  }
  const auto& stage = model.stages.front();
  if (!stage.embedding || !stage.heads.output || stage.embedding->weights.size() != 1 ||
      stage.heads.output->weights.size() != 1 ||
      stage.embedding->weights.front().tensor != stage.heads.output->weights.front().tensor)
    return Refused("the approved target requires its output tied to its embedding");
  const auto values = spec::mappings::Gemma2V0Values(model::Gemma2_2B());
  if (!values) return Refused(values.error());
  // Profile constants are an approval oracle only. Execute the authenticated
  // stored spec unchanged; never replace a disagreeing spec by this expected one.
  const auto expected = spec::Synthesize(spec::mappings::Gemma2Mapping(), source, *values);
  if (!expected) return Refused(expected.error());
  if (*expected != model) return Refused("stored spec differs from the approved component profile");
  const std::array<std::span<const spec::SourceTensor>, 1> tables{source.tensors};
  if (const auto valid = components::ValidateSpec(model, tables); !valid)
    return Refused(valid.error());
  return {};
}

std::expected<ComponentAdmission, std::string> ReadComponentAdmission(
    const artifact::Artifact& artifact) {
  if (artifact.format_version() != 1) return Refused("requires a native v1 artifact");
  const auto* profile = Approved(artifact.model().architecture, artifact.sources());
  if (profile == nullptr) return Refused("source identity has no approved v1 execution profile");
  if (!artifact.folds().empty() || !artifact.relayouts().empty() ||
      artifact.resources().size() != profile->tensors || !artifact.expert_arrays().empty())
    return Refused("the approved target requires its copy-only native tensor inventory");
  auto model = spec::StoredSpec(artifact);
  if (!model) return Refused(model.error());
  auto source = spec::SourceOf(artifact);
  if (!source) return Refused(source.error());
  if (auto checked = CheckComponentAdmission(*model, *source, artifact.sources()); !checked)
    return std::unexpected(checked.error());
  return ComponentAdmission{std::move(*model), std::move(*source)};
}

std::expected<spec::PolicyRule, std::string> ComponentAdmissionPolicy(
    const spec::Mapping& mapping, const spec::Envelope& envelope) {
  const auto profile = std::ranges::find(kProfiles, mapping.architecture, &Profile::architecture);
  if (profile == kProfiles.end())
    return Refused("no stored-spec policy declaration for this architecture");
  if (envelope.context == 0 || envelope.context > profile->max_context || envelope.slots == 0 ||
      envelope.slots > profile->max_slots)
    return Refused("execution envelope exceeds the approved component profile");
  if (std::ranges::any_of(mapping.policies, [](const auto& rule) { return !rule.gate.empty(); }))
    return Refused("stored-spec policy cannot consult legacy source keys");
  auto rule = spec::DefaultRule(mapping, {}, envelope);
  if (!rule) return Refused(rule.error());
  if (*rule == nullptr) return Refused("no approved policy for this execution envelope");
  return **rule;
}

}  // namespace llmp::runtime
