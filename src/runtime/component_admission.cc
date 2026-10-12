// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "runtime/component_admission.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "base/sha256.h"
#include "components/step.h"
#include "model/gemma2.h"
#include "model/gemma3.h"
#include "model/gemma4.h"
#include "spec/mappings/gemma2.h"
#include "spec/mappings/gemma3.h"
#include "spec/mappings/gemma4.h"
#include "spec/stored.h"

namespace llmp::runtime {
namespace {

// The existing approved execution checkpoint, not the container's artifact
// ID: publication does not broaden the family/profile or format domain.
enum class ProfileId { kGemma2, kGemma3, kGemma26, kGemma31 };
struct Profile {
  ProfileId id;
  std::string_view architecture, source_name, source_sha256;
  std::uint64_t source_bytes;
  std::size_t tensors, arrays;
  std::uint32_t max_context, max_slots;
  std::string_view inventory;
};
// Sorted name / native encoding / original dimensions from the approved
// captured tensor directories; algorithm and source pins are documented in
// docs/experiments/m36-native-v1-admission/README.md. This also fixes each
// layer's mixed quantization, rather than accepting a family-wide type set.
constexpr std::array kProfiles{
    Profile{ProfileId::kGemma2, "gemma2", "gemma-2-2b-it-Q8_0.gguf",
            "2d448a9aab894b8e8e18168cf3f490cb9f65632222f29f93514ac9ecc754debe", 2784495456ULL, 288,
            0, 8192, 2, "91678220743a65166375bad2f712380aed8e7cfec56203bfdc87321a5e5ed0f5"},
    Profile{ProfileId::kGemma3, "gemma3", "gemma-3-4b-it-qat-Q4_0.gguf",
            "ee91c3e7a4ab95d8c95672f9fcb58bf236b257e9f217966bcf53a5a6df4ab49a", 2526080992ULL, 444,
            0, 131072, 2, "8aedee1cffc43f331cb38dcd9dc60e98a2aa67cd98df16248f1c78e36bd4a8e7"},
    Profile{ProfileId::kGemma26, "gemma4", "gemma-4-26B-A4B-it-UD-Q4_K_M.gguf",
            "f2c28b3dc4776931ac6f879e11f203dec637ea0f14267a86ec8f6165f63f293f", 16947541728ULL, 658,
            60, 8192, 4, "97551b8fcfead217ce2feb8c0abe7228211b15bca6867158a955f3d5a7790e02"},
    Profile{ProfileId::kGemma31, "gemma4", "gemma-4-31B-it-UD-Q4_K_XL.gguf",
            "9e92cb6236044c6a9870af406029c74a76e0571c157a6f95df724dcc8c7a1575", 18822970304ULL, 833,
            0, 8192, 4, "a5ba88997fe0b72a54148526573c38e6a619f5a91980c59dcb5cbc5123760f52"}};

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

const model::Gemma4Profile* GemmaProfile(const Profile& profile) {
  if (profile.id == ProfileId::kGemma26) return &model::Gemma4_26BA4B();
  if (profile.id == ProfileId::kGemma31) return &model::Gemma4_31B();
  return nullptr;
}
std::expected<std::vector<spec::SourceValue>, std::string> Values(const Profile& profile) {
  switch (profile.id) {
    case ProfileId::kGemma2:
      return spec::mappings::Gemma2V0Values(model::Gemma2_2B());
    case ProfileId::kGemma3:
      return spec::mappings::Gemma3V0Values(model::Gemma3_4BQat());
    case ProfileId::kGemma26:
    case ProfileId::kGemma31:
      return spec::mappings::Gemma4V0Values(*GemmaProfile(profile));
  }
  return Refused("unknown profile");
}
std::expected<void, std::string> Inventory(const spec::Source& source, const Profile& profile) {
  std::vector<const spec::SourceTensor*> sorted;
  sorted.reserve(source.tensors.size());
  for (const auto& t : source.tensors) {
    const auto& f = t.format;
    if (!format::Validate(f) || t.part != format::PartRole::kData || f.stacked != 1 ||
        f.params != format::EncodingParams{} || f.shape.rank < 1 || f.shape.rank > 3 ||
        t.experts != (f.shape.rank == 3) ||
        f.layout != (f.encoding == format::Encoding::kF32 ? format::Layout::kDense
                                                          : format::Layout::kGgmlBlocks))
      return Refused("native tensor layout, part, parameters or expert geometry differs");
    sorted.push_back(&t);
  }
  std::ranges::sort(sorted, {}, [](const auto* tensor) { return tensor->name; });
  base::Sha256 hash;
  for (const auto* tensor : sorted) {
    hash.Update(tensor->name)
        .Update("\n")
        .Update(format::Info(tensor->format.encoding).name)
        .Update("\n");
    for (std::uint8_t i = 0; i < tensor->format.shape.rank; ++i)
      hash.Update(std::to_string(tensor->format.shape.ne[i])).Update(",");
    hash.Update("\n");
  }
  if (base::ToHex(hash.Finish()) != profile.inventory)
    return Refused("native tensor inventory differs from the approved checkpoint");
  return {};
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
  if (auto inventory = Inventory(source, *profile); !inventory) return inventory;
  const auto& stage = model.stages.front();
  if (!stage.embedding || !stage.heads.output || stage.embedding->weights.size() != 1 ||
      stage.heads.output->weights.size() != 1 ||
      stage.embedding->weights.front().tensor != stage.heads.output->weights.front().tensor)
    return Refused("the approved target requires its output tied to its embedding");
  const auto values = Values(*profile);
  if (!values) return Refused(values.error());
  // Profile constants are an approval oracle only. Execute the authenticated
  // stored spec unchanged; never replace a disagreeing spec by this expected one.
  const auto expected =
      spec::Synthesize(*spec::FindMapping(profile->architecture), source, *values);
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
      artifact.resources().size() != profile->tensors - profile->arrays ||
      artifact.expert_arrays().size() != profile->arrays)
    return Refused("the approved target requires its copy-only native tensor inventory");
  auto model = spec::StoredSpec(artifact);
  if (!model) return Refused(model.error());
  auto source = spec::SourceOf(artifact);
  if (!source) return Refused(source.error());
  if (auto checked = CheckComponentAdmission(*model, *source, artifact.sources()); !checked)
    return std::unexpected(checked.error());
  return ComponentAdmission{std::move(*model), std::move(*source), GemmaProfile(*profile)};
}

std::expected<spec::PolicyRule, std::string> ComponentAdmissionPolicy(
    const spec::ModelSpec& model, const spec::Source& source,
    std::span<const artifact::Source> origins, const spec::Envelope& envelope) {
  if (auto checked = CheckComponentAdmission(model, source, origins); !checked)
    return std::unexpected(checked.error());
  const auto* profile = Approved(source.architecture, origins);
  if (envelope.context == 0 || envelope.context > profile->max_context || envelope.slots == 0 ||
      envelope.slots > profile->max_slots)
    return Refused("execution envelope exceeds the approved component profile");
  if (profile->id == ProfileId::kGemma3 && envelope.context > 4096 && envelope.slots != 1)
    return Refused("Gemma3 wider context requires exactly one owner");
  // Reuse the approved mapping's policy. These constants have just been
  // proved equal to the complete stored spec, including whether every layer
  // has its routed block. No checkpoint key bag or v0 descriptor is invented.
  const auto values = Values(*profile);
  if (!values) return Refused(values.error());
  auto rule = spec::DefaultRule(*spec::FindMapping(profile->architecture), *values, envelope);
  if (!rule) return Refused(rule.error());
  if (*rule == nullptr) return Refused("no approved policy for this execution envelope");
  return **rule;
}

}  // namespace llmp::runtime
