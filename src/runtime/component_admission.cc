// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "runtime/component_admission.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <format>
#include <limits>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "base/sha256.h"
#include "components/step.h"
#include "model/dsv4.h"
#include "model/gemma2.h"
#include "model/gemma3.h"
#include "model/gemma4.h"
#include "model/gemma4_assistant.h"
#include "model/qwen38.h"
#include "runtime/gemma_profile.h"
#include "spec/mappings/dsv4.h"
#include "spec/mappings/gemma2.h"
#include "spec/mappings/gemma3.h"
#include "spec/mappings/gemma4.h"
#include "spec/mappings/qwen38.h"
#include "spec/stored.h"

namespace llmp::runtime {
namespace {

// The existing approved execution checkpoint, not the container's artifact
// ID: publication does not broaden the family/profile or format domain.
enum class ProfileId { kGemma2, kGemma3, kGemma26, kGemma31, kQwenGguf, kDsv4Gguf };
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

// Pinned three-shard UD-IQ3_XXS source. The ordinary writer verifies these
// whole streams while publishing; runtime admission checks their declared
// identity and the complete stored inventory/spec without rehashing payloads.
constexpr Profile kQwenGguf{ProfileId::kQwenGguf,
                            "qwen4exp",
                            {},
                            {},
                            0,
                            1224,
                            144,
                            model::kQwen38FlashContext,
                            std::numeric_limits<std::uint32_t>::max(),
                            "ebfde9be4041367912853e848599eb2bbfeb62f853ae373546ceb31416d482f6"};
struct Origin {
  std::string_view name, sha256;
  std::uint64_t bytes;
};
constexpr std::array kQwenOrigins{
    Origin{"Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf",
           "268f81fdedf3149a538f252308927a4d5d1f6e062c178568a51e3b519744f8a8", 10946624ULL},
    Origin{"Qwen3.8-Flash-Next-UD-IQ3_XXS-00002-of-00003.gguf",
           "cfe600b236b88c7fad1613a5ca5e83b9f2beb63cbd44c32b2be50a44747c695f", 49567921344ULL},
    Origin{"Qwen3.8-Flash-Next-UD-IQ3_XXS-00003-of-00003.gguf",
           "f1912ba34c79427d2295a58dcb2b732b5931af5bef7a373c60557a57d9ee7250", 32382955968ULL}};

constexpr Profile kDsv4Gguf{ProfileId::kDsv4Gguf,
                            "deepseek4",
                            {},
                            {},
                            0,
                            1328,
                            129,
                            model::kDsv4FlashContext,
                            std::numeric_limits<std::uint32_t>::max(),
                            "0c4a8d8bc83e96cac484de7c6272c75d282fe488d792ef60f686138f5a177c9c"};
constexpr std::array kDsv4Origins{
    Origin{"DeepSeek-V4-Flash-0731-UD-Q2_K_XL-00001-of-00003.gguf",
           "de65c5bb660817b95cc281bf54935f9d1bcc3b22535f2eae63b35a14c7c5724b", 5257664ULL},
    Origin{"DeepSeek-V4-Flash-0731-UD-Q2_K_XL-00002-of-00003.gguf",
           "c8728a298862cd8736782e2e4056e5e3e03c415ba552a38173264494e41ef327", 49437013568ULL},
    Origin{"DeepSeek-V4-Flash-0731-UD-Q2_K_XL-00003-of-00003.gguf",
           "a4bae0788c025c1b2388754592c98fefa1372981748ac9875039bf6a30e6eeb4", 47390237120ULL}};

const Profile* Approved(std::string_view architecture, std::span<const artifact::Source> origins) {
  for (const auto& [profile, expected_origins] :
       std::array{std::pair{&kQwenGguf, std::span<const Origin>(kQwenOrigins)},
                  std::pair{&kDsv4Gguf, std::span<const Origin>(kDsv4Origins)}}) {
    if (architecture != profile->architecture || origins.size() != expected_origins.size())
      continue;
    for (const auto& expected : expected_origins)
      if (std::ranges::count_if(origins, [&](const auto& origin) {
            return origin.name == expected.name && origin.bytes.value() == expected.bytes &&
                   base::ToHex(origin.sha256) == expected.sha256;
          }) != 1)
        return nullptr;
    return profile;
  }
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
    case ProfileId::kDsv4Gguf:
      return spec::mappings::Dsv4V0Values(model::Dsv4Flash());
    case ProfileId::kQwenGguf: {
      // Independent source metadata captured with the approved tensor inventory.
      const model::Qwen38PleHash hash{
          .multipliers = {23703573157769, 20109073645365, 8052911324071},
          .offsets = {0, 20000003, 40000026, 60000059, 80000106, 100000165, 120000228, 140000297,
                      160000374, 180000455, 200000548, 220000655, 240000802, 260000955, 280001114,
                      300001275},
          .vocab = {20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077,
                    20000081, 20000093, 20000107, 20000147, 20000153, 20000159, 20000161,
                    20000171}};
      return spec::mappings::Qwen38GgufValues(model::Qwen38Flash(), hash);
    }
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
        f.layout != (format::Info(f.encoding).quantized ? format::Layout::kGgmlBlocks
                                                        : format::Layout::kDense))
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

// Size anchors for the closed serving mappings' transient settings synthesis.
// No graph, tensor payload or execution owner is constructed here.
static_assert(sizeof(spec::Param) <= 128 && sizeof(spec::WeightBinding) <= 128);
static_assert(sizeof(spec::ComponentRef) <= 128 && sizeof(spec::SourceTensor) <= 256);
static_assert(sizeof(spec::V0Tensor) <= 128 && sizeof(spec::Stage) <= 2048);

bool OnComponentEngine(std::string_view architecture) {
  return architecture == "gemma2" || architecture == "gemma3" || architecture == "gemma4";
}

std::expected<std::vector<spec::SourceValue>, std::string> ComponentV0Values(
    const artifact::Artifact& artifact) {
  const auto& architecture = artifact.model().architecture;
  if (architecture == "gemma2") return spec::mappings::Gemma2V0Values(model::Gemma2_2B());
  if (architecture == "gemma3") return spec::mappings::Gemma3V0Values(model::Gemma3_4BQat());
  if (architecture == "gemma4") {
    auto profile = ApprovedGemmaProfile(artifact);
    if (!profile) return std::unexpected(profile.error());
    return spec::mappings::Gemma4V0Values(**profile);
  }
  return std::unexpected(
      std::format("the component engine has no recipe for architecture {} yet", architecture));
}

std::expected<std::vector<spec::SourceValue>, std::string> ComponentDrafterV0Values(
    const artifact::Artifact& target, const artifact::Artifact& drafter) {
  if (drafter.model().architecture == "gemma4-assistant" &&
      target.model().architecture == "gemma4") {
    auto profile = ApprovedGemmaProfile(target);
    if (!profile) return std::unexpected(profile.error());
    const model::Gemma4AssistantProfile& assistant =
        *profile == &model::Gemma4_31B() ? model::Gemma4Assistant31() : model::Gemma4Assistant26();
    if (auto bound = model::BindGemma4Assistant(assistant, drafter); !bound) {
      return std::unexpected(bound.error());
    }
    return spec::mappings::Gemma4AssistantV0Values(assistant, **profile);
  }
  return std::unexpected(std::format("the component engine has no drafter {} for architecture {}",
                                     drafter.model().architecture, target.model().architecture));
}

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
  if (profile->id != ProfileId::kQwenGguf && profile->id != ProfileId::kDsv4Gguf &&
      (!stage.embedding || !stage.heads.output || stage.embedding->weights.size() != 1 ||
       stage.heads.output->weights.size() != 1 ||
       stage.embedding->weights.front().tensor != stage.heads.output->weights.front().tensor))
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

std::expected<void, std::string> CheckComponentArtifactAdmission(
    const artifact::Artifact& artifact) {
  if (artifact.format_version() != 1) return Refused("requires a native v1 artifact");
  const auto* profile = Approved(artifact.model().architecture, artifact.sources());
  if (profile == nullptr) return Refused("source identity has no approved v1 execution profile");
  if (!artifact.folds().empty() || !artifact.relayouts().empty() ||
      artifact.resources().size() != profile->tensors - profile->arrays ||
      artifact.expert_arrays().size() != profile->arrays)
    return Refused("the approved target requires its copy-only native tensor inventory");
  return {};
}

std::expected<ComponentAdmission, std::string> ReadComponentAdmission(
    const artifact::Artifact& artifact) {
  if (auto checked = CheckComponentArtifactAdmission(artifact); !checked)
    return std::unexpected(checked.error());
  const auto* profile = Approved(artifact.model().architecture, artifact.sources());
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
  // The existing DS mapping deliberately has no policy directives. Preserve
  // exactly that default instead of inventing a stored-only recipe.
  if (*rule == nullptr && profile->id == ProfileId::kDsv4Gguf) return spec::PolicyRule{};
  if (*rule == nullptr) return Refused("no approved policy for this execution envelope");
  return **rule;
}

}  // namespace llmp::runtime
