// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "runtime/component_admission.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <format>
#include <functional>
#include <limits>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include "artifact/gguf_metadata.h"
#include "base/json_stream.h"
#include "base/sha256.h"
#include "components/step.h"
#include "kernels/common/lower.h"
#include "kernels/common/step_program.h"
#include "planner/select.h"
#include "platform/direct_io.h"
#include "runtime/approvals.h"
#include "runtime/component_admission_internal.h"
#include "runtime/numerical_evidence.h"
#include "spec/image_family.h"
#include "spec/stored.h"

namespace llmp::runtime {
namespace {

std::unexpected<std::string> Refused(std::string_view reason) {
  return std::unexpected("component artifact admission: " + std::string(reason));
}

std::unexpected<std::string> Unexpected(std::string reason) {
  return std::unexpected(std::move(reason));
}

constexpr auto kRefusedInventory =
    "native tensor layout, part, parameters or expert geometry differs";

// Container origin is provenance, not executable-format permission. Actual
// operation support is proved by the execution registry while planning.
bool Admissible(const spec::SourceTensor& tensor) {
  const auto parts = format::PartsOf(tensor.format);
  return (!tensor.experts || tensor.format.shape.rank == 3) &&
         format::Validate(tensor.format).has_value() && parts && parts->Find(tensor.part);
}

}  // namespace

// The approved inventory: every tensor structurally valid, and the
// digest of every tensor's complete native description, sorted by name:
// its name, encoding, layout (by its enumerator), part role, stacked count,
// expert flag, every encoding parameter and its dimensions, one line each
// field.
std::expected<void, std::string> internal::CheckInventory(const spec::Source& source,
                                                          const Approval& approval) {
  std::vector<const spec::SourceTensor*> sorted;
  sorted.reserve(source.tensors.size());
  for (const auto& t : source.tensors) {
    if (!Admissible(t)) return Refused(kRefusedInventory);
    sorted.push_back(&t);
  }
  std::ranges::sort(sorted, {}, [](const auto* tensor) {
    return std::pair<std::string_view, format::PartRole>(tensor->name, tensor->part);
  });
  base::Sha256 hash;
  const auto field = [&](std::string_view text) { hash.Update(text).Update("\n"); };
  const auto number = [&](auto value) { field(std::to_string(value)); };
  for (const auto* tensor : sorted) {
    const auto& f = tensor->format;
    field(tensor->name);
    field(format::Info(f.encoding).name);
    number(std::to_underlying(f.layout));
    field(format::PartRoleName(tensor->part));
    number(f.stacked);
    number(tensor->experts ? 1 : 0);
    number(f.params.k_bits);
    number(std::to_underlying(f.params.codebook));
    number(f.params.tensor_scale ? 1 : 0);
    number(f.params.bits);
    number(f.params.group);
    number(std::to_underlying(f.params.scale));
    number(f.params.zero_point ? 1 : 0);
    for (std::uint8_t i = 0; i < f.shape.rank; ++i)
      hash.Update(std::to_string(f.shape.ne[i])).Update(",");
    hash.Update("\n");
  }
  const auto actual_hash = base::ToHex(hash.Finish());
  if (actual_hash != approval.inventory)
    return Refused(std::format(
        "native tensor inventory differs from the approved checkpoint (got {}, expected {})",
        actual_hash, approval.inventory));
  return {};
}

namespace {

// The source the approval names: its container, importer folds and tensor
// count.
bool SourceMatches(const spec::Source& source, const Approval& approval) {
  return source.container == approval.container &&
         std::ranges::equal(source.folds, approval.folds) &&
         source.tensors.size() == approval.tensors;
}

// An applied transform's record: its tensors' count and the SHA-256 of
// their sorted names, each followed by a line feed (approvals.h).
bool Recorded(const ApprovedTransform& want, std::string_view name,
              std::span<const std::string> tensors) {
  std::vector<std::string_view> sorted(tensors.begin(), tensors.end());
  std::ranges::sort(sorted);
  base::Sha256 hash;
  for (const std::string_view t : sorted) hash.Update(t).Update("\n");
  return want.name == name && want.tensors == tensors.size() &&
         base::ToHex(hash.Finish()) == want.digest;
}

// The artifact's applied folds and relayouts, exactly its approval's.
bool Transforms(const artifact::Artifact& artifact, const Approval& approval) {
  const auto folds = artifact.folds();
  if (!std::ranges::equal(folds, approval.folds, {},
                          [](const auto& fold) -> std::string_view { return fold.name; }))
    return false;
  if (!approval.fold_tensors.empty()) {
    if (approval.fold_tensors.size() != folds.size()) return false;
    for (std::size_t i = 0; i < folds.size(); ++i)
      if (!Recorded(approval.fold_tensors[i], folds[i].name, folds[i].tensors)) return false;
  }
  const auto relayouts = artifact.relayouts();
  if (relayouts.size() != approval.relayouts.size()) return false;
  for (std::size_t i = 0; i < relayouts.size(); ++i)
    if (!Recorded(approval.relayouts[i], format::Info(relayouts[i].from).name,
                  relayouts[i].tensors))
      return false;
  return true;
}

// A pinned spec digest against the actual spec's.
std::expected<void, std::string> Pinned(const Approval& approval, const spec::ModelSpec& model) {
  const auto actual = base::ToHex(model.Digest());
  if (actual != approval.spec)
    return Refused(std::format("the spec ({}) is not the one approved for {} ({})", actual,
                               approval.name, approval.spec));
  return {};
}

// GGUF's bool type.
constexpr std::uint32_t kGgufBool = 7;

// A stored artifact's source values for `mapping`: the keys it reads, from
// the GGUF metadata the artifact keeps, as the importer read them (reals
// widened from their stored precision, flags as flags, lists as lists).
std::expected<std::vector<spec::SourceValue>, std::string> KeptValues(
    const artifact::Artifact& artifact, const spec::Mapping& mapping) {
  const std::string kv = KeptGguf(artifact);
  if (kv.empty()) return Refused("the drafter keeps no GGUF metadata");
  auto bytes = artifact.ReadMetadata(kv);
  if (!bytes) return Refused(std::format("{}: {}", kv, bytes.error().ToString()));
  const auto owned_keys = spec::SourceKeys(mapping);
  const std::vector<std::string_view> keys(owned_keys.begin(), owned_keys.end());
  auto read = artifact::ReadGgufMetadata(std::as_bytes(std::span(*bytes)), keys);
  if (!read) return Refused(std::format("{}: {}", kv, read.error().ToString()));
  std::vector<spec::SourceValue> out;
  for (const auto& [key, value] : *read) {
    spec::ParamValue v;
    switch (value.kind) {
      case artifact::GgufValue::Kind::kInteger:
        if (value.type == kGgufBool)
          v = value.integer != 0;
        else
          v = value.integer;
        break;
      case artifact::GgufValue::Kind::kFloat:
        v = value.real;
        break;
      case artifact::GgufValue::Kind::kString:
        v = value.text;
        break;
      case artifact::GgufValue::Kind::kIntegers:
        if (value.integers.size() > spec::kMaxList)
          return Refused("retained mapping integer list exceeds the value bound");
        v = value.integers;
        break;
      case artifact::GgufValue::Kind::kFloats:
        if (value.reals.size() > spec::kMaxList)
          return Refused("retained mapping real list exceeds the value bound");
        v = value.reals;
        break;
      case artifact::GgufValue::Kind::kOther:
        return Refused("retained mapping value has an unsupported type");
    }
    out.push_back({.key = key, .value = std::move(v)});
  }
  return out;
}

}  // namespace

std::string KeptGguf(const artifact::Artifact& a) {
  std::string kv;
  for (const artifact::ListedFile& f : a.files()) {
    if (f.role == artifact::FileRole::kSourceMetadata && f.path.starts_with("meta/") &&
        f.path.ends_with(".kv.gguf") && (kv.empty() || f.path.contains("-00001-of-"))) {
      kv = f.path.substr(5);
    }
  }
  return kv;
}

std::expected<void, std::string> ReadPlaced(const artifact::Artifact& artifact,
                                            const artifact::Placement& place,
                                            const std::function<std::expected<void, std::string>(
                                                std::uint64_t, std::span<const std::byte>)>& each) {
  const auto& group = artifact.groups()[place.group];
  const auto& shard = artifact.shards()[group.shard];
  const auto begin = shard.data_offset.value() + group.offset.value() + place.offset.value();
  const auto end = begin + place.bytes.value();
  const auto shard_end = shard.data_offset.value() + shard.data_bytes.value();
  auto file = artifact.OpenShardForDirectRead(group.shard);
  if (!file) return std::unexpected(file.error().ToString());
  alignas(4096) std::array<std::byte, 4096> window{};
  for (auto at = begin / window.size() * window.size(); at < end; at += window.size()) {
    if (at > shard_end || window.size() > shard_end - at)
      return std::unexpected("a direct read passes its shard");
    if (auto read = platform::TransferDirectFile(file->get(), at, window, false); !read)
      return std::unexpected(std::format("a direct read failed: errno {}", read.error()));
    const auto first = std::max(at, begin), last = std::min(at + window.size(), end);
    if (auto taken = each(first - begin, std::span(window).subspan(first - at, last - first));
        !taken)
      return taken;
  }
  return {};
}

// Size anchors for the stored specs' transient settings reads. No graph,
// tensor payload or execution owner is constructed here.
static_assert(sizeof(spec::Param) <= 128 && sizeof(spec::WeightBinding) <= 128);
static_assert(sizeof(spec::ComponentRef) <= 128 && sizeof(spec::SourceTensor) <= 256);
static_assert(sizeof(spec::Stage) <= 2048);

template <class T>
std::expected<std::vector<T>, std::string> ReadRecipeIntegers(const artifact::Artifact& artifact,
                                                              std::uint32_t resource,
                                                              std::uint64_t count) {
  auto place = artifact.ResourcePlacement(resource);
  if (!place || place->bytes.value() != count * sizeof(T))
    return std::unexpected("integer recipe metadata has an inconsistent byte count");
  std::vector<T> out(count);
  auto read = ReadPlaced(
      artifact, *place,
      [&](std::uint64_t at, std::span<const std::byte> bytes) -> std::expected<void, std::string> {
        std::memcpy(reinterpret_cast<std::byte*>(out.data()) + at, bytes.data(), bytes.size());
        return {};
      });
  if (!read) return std::unexpected("reading integer recipe metadata: " + read.error());
  return out;
}

template std::expected<std::vector<std::int32_t>, std::string> ReadRecipeIntegers<std::int32_t>(
    const artifact::Artifact& artifact, std::uint32_t resource, std::uint64_t count);

bool OnComponentEngine(std::string_view architecture) {
  const spec::Mapping* mapping = spec::FindMapping(architecture);
  return mapping != nullptr &&
         (spec::ServingRecipeOf(*mapping) != nullptr || mapping->draft != nullptr);
}

const spec::Mapping* ComposedDrafterMapping(std::string_view architecture,
                                            std::string_view container) {
  return spec::FindMapping(architecture, container);
}

const spec::Mapping* ComposedDrafterMapping(const artifact::Artifact& drafter) {
  return ComposedDrafterMapping(drafter.model().architecture, drafter.model().container);
}

std::expected<void, std::string> RequireServedVersion(const artifact::Artifact& artifact) {
  if (artifact.format_version() == 1) return {};
  return std::unexpected(std::format(
      "artifact {} ({}) is prepared-artifact format version {}; serving reads version 1 only "
      "(D-118): re-import its source checkpoint with `llmp import SOURCE --store DIRECTORY` and "
      "configure the new artifact ID",
      artifact.id(), artifact.model().architecture, artifact.format_version()));
}

namespace {

bool OriginsWellFormed(std::span<const artifact::Source> origins) {
  if (origins.empty()) return false;
  for (std::size_t i = 0; i < origins.size(); ++i) {
    if (origins[i].name.empty() || origins[i].bytes.value() == 0) return false;
    for (std::size_t j = i + 1; j < origins.size(); ++j) {
      if (origins[i].name == origins[j].name) return false;
    }
  }
  return true;
}

bool FoldDeclared(const spec::Mapping& mapping, std::string_view container, std::string_view fold) {
  return std::ranges::any_of(mapping.folds, [&](const spec::FoldRule& rule) {
    return rule.fold == fold && rule.container == container;
  });
}

std::expected<void, std::string> CheckUnpinnedInventory(const spec::Source& source,
                                                        const spec::Mapping& mapping) {
  if (source.tensors.empty()) return Refused(kRefusedInventory);
  for (const std::string& fold : source.folds) {
    if (!FoldDeclared(mapping, source.container, fold)) {
      return Refused("source fold is not declared by the architecture's mapping");
    }
  }
  for (const auto& t : source.tensors) {
    if (!Admissible(t)) return Refused(kRefusedInventory);
  }
  return {};
}

}  // namespace

std::expected<std::vector<spec::SourceValue>, std::string> ComponentDrafterValues(
    const artifact::Artifact& target, const artifact::Artifact& drafter) {
  if (auto served = RequireServedVersion(target); !served) return Refused(served.error());
  if (auto served = RequireServedVersion(drafter); !served) return Refused(served.error());
  const spec::Mapping* mapping = ComposedDrafterMapping(drafter);
  if (mapping == nullptr || mapping->role != spec::StageRole::kDrafter ||
      mapping->draft == nullptr) {
    return Refused("the drafter's architecture has no drafter mapping");
  }
  const Approval* t = FindApproval(target.model().architecture, target.sources());
  const Approval* d = FindApproval(drafter.model().architecture, drafter.sources());
  if (t != nullptr && t->drafter)
    return Refused("target source identity is a drafter, not a target");
  if (d != nullptr && !d->drafter)
    return Refused("drafter source identity is a target, not a drafter");
  const spec::Mapping* target_mapping =
      spec::FindMapping(target.model().architecture, target.model().container);
  if (!target_mapping || target_mapping->role != spec::StageRole::kTarget)
    return Refused("target architecture/container has no target component mapping");
  // Recipes name canonical architectures; manifest names retain the source dialect.
  if (mapping->draft->target_architecture != target_mapping->architecture)
    return Refused(std::format("drafter {} does not target architecture {}",
                               drafter.model().architecture, target.model().architecture));
  // A drafter without kept metadata values: its recipe's, from its source.
  if (!mapping->draft->values.empty()) {
    auto source = spec::SourceOf(drafter);
    if (!source) return Refused(source.error());
    auto target_model = spec::StoredSpec(target);
    if (!target_model) return Refused(target_model.error());
    auto target_source = spec::SourceOf(target);
    if (!target_source) return Refused(target_source.error());
    auto values = spec::ExtractValues(*mapping, *target_model, *target_source, *source);
    if (!values) return Refused(values.error());
    return values;
  }
  return KeptValues(drafter, *mapping);
}

std::expected<void, std::string> ComponentDrafterEnvelope(const spec::Envelope& envelope) {
  // An approved drafter's envelope is its settings' default (model_settings
  // .cc), not a bound here: an override past it is served, with a warning.
  if (envelope.context == 0 || envelope.slots == 0)
    return Refused("drafter execution envelope must have non-zero context and slots");
  return {};
}

std::expected<std::optional<DraftHeadMap>, std::string> DraftHeadRows(
    const artifact::Artifact& drafter, std::string_view head_ids, std::uint64_t vocab) {
  if (head_ids.empty()) return std::nullopt;
  const auto index = drafter.FindResource(head_ids);
  if (!index) return std::nullopt;
  const auto& resource = drafter.resources()[*index];
  std::uint64_t rows = 0;
  if (resource.format && resource.format->encoding == format::Encoding::kI32 &&
      resource.format->layout == format::Layout::kDense && resource.format->shape.rank == 2 &&
      resource.format->shape.ne[0] == 1)
    rows = resource.format->shape.ne[1];
  if (rows == 0 || rows > vocab)
    return Refused("a selected draft head's token map is not I32 [1, rows] within the vocabulary");
  return DraftHeadMap{.resource = *index, .rows = rows};
}

std::expected<std::uint32_t, std::string> ConfigureDraftVocabulary(
    std::span<spec::SourceValue> values, const spec::DraftRecipe& recipe, std::uint32_t requested,
    std::uint32_t target_vocab, std::optional<std::uint32_t> selected_rows) {
  if (recipe.vocab_key.empty()) return 0;
  const std::uint32_t available = selected_rows.value_or(target_vocab);
  const std::uint32_t width = requested == 0 ? available : std::min(requested, available);
  if (width == 0 || available > target_vocab)
    return Refused("a draft head needs a positive row count within the target vocabulary");
  const auto found = std::ranges::find(values, recipe.vocab_key, &spec::SourceValue::key);
  if (found == values.end()) return Refused("a drafter has no draft vocabulary value");
  found->value = static_cast<std::int64_t>(width);
  return width;
}

std::expected<std::int64_t, std::string> KeptInteger(const artifact::Artifact& artifact,
                                                     std::string_view key) {
  const std::string kv = KeptGguf(artifact);
  if (kv.empty()) return Refused(std::format("no kept GGUF metadata holds {}", key));
  auto bytes = artifact.ReadMetadata(kv);
  if (!bytes) return Refused(std::format("{}: {}", kv, bytes.error().ToString()));
  const std::array<std::string_view, 1> keys{key};
  auto read = artifact::ReadGgufMetadata(std::as_bytes(std::span(*bytes)), keys);
  if (!read) return Refused(std::format("{}: {}", kv, read.error().ToString()));
  const auto found = read->find(key);
  if (found == read->end() || found->second.kind != artifact::GgufValue::Kind::kInteger)
    return Refused(std::format("the kept metadata has no integer {}", key));
  return found->second.integer;
}

std::expected<void, std::string> CheckComposedAdmission(const spec::ModelSpec& composed,
                                                        const artifact::Artifact& target,
                                                        const artifact::Artifact& drafter) {
  if (auto served = RequireServedVersion(drafter); !served) return Refused(served.error());
  if (composed.stages.size() < 2 || composed.stages.back().role != spec::StageRole::kDrafter)
    return Refused("composed spec has no drafter stage");
  auto target_source = spec::SourceOf(target);
  if (!target_source) return Refused(target_source.error());
  auto draft_source = spec::SourceOf(drafter);
  if (!draft_source) return Refused(draft_source.error());
  return CheckComposedAdmission(composed, *target_source, *draft_source);
}

std::expected<void, std::string> CheckComposedAdmission(const spec::ModelSpec& composed,
                                                        const spec::Source& target,
                                                        const spec::Source& drafter) {
  if (composed.stages.size() < 2 || composed.stages.back().role != spec::StageRole::kDrafter)
    return Refused("composed spec has no drafter stage");
  const std::array<std::span<const spec::SourceTensor>, 2> tables{target.tensors, drafter.tensors};
  if (const auto valid = components::ValidateSpec(composed, tables); !valid)
    return Refused(valid.error());
  return {};
}

std::expected<void, std::string> CheckComponentDrafterAdmission(
    const spec::Source& source, std::span<const artifact::Source> origins) {
  if (!OriginsWellFormed(origins)) return Refused("source origins are empty or duplicate");
  const spec::Mapping* mapping = ComposedDrafterMapping(source.architecture, source.container);
  if (mapping == nullptr || mapping->role != spec::StageRole::kDrafter || mapping->draft == nullptr)
    return Refused("architecture has no drafter component mapping");
  return CheckUnpinnedInventory(source, *mapping);
}

std::expected<void, std::string> CheckComponentAdmission(
    const spec::ModelSpec& model, const spec::Source& source,
    std::span<const artifact::Source> origins) {
  if (!OriginsWellFormed(origins)) return Refused("source origins are empty or duplicate");
  if (model.pipeline != spec::PipelineKind::kLlm || model.stages.size() != 1 ||
      model.stages[0].role != spec::StageRole::kTarget || model.stages[0].artifact != 0 ||
      model.stages[0].context == 0)
    return Refused("the stored pipeline is not a standalone target");
  const spec::Mapping* mapping = spec::FindMapping(source.architecture, source.container);
  if (mapping == nullptr || mapping->role != spec::StageRole::kTarget)
    return Refused("architecture has no target component mapping");
  if (auto inv = CheckUnpinnedInventory(source, *mapping); !inv) return inv;
  std::vector<std::string> expected_folds = source.folds;
  for (const spec::FoldRule& f : mapping->folds) {
    if (!f.imported && f.container == source.container &&
        !std::ranges::contains(expected_folds, f.fold)) {
      expected_folds.emplace_back(f.fold);
    }
  }
  std::ranges::sort(expected_folds);
  if (model.stages.front().folds != expected_folds)
    return Refused("stored stage folds disagree with source folds");
  const std::array<std::span<const spec::SourceTensor>, 1> tables{source.tensors};
  if (const auto valid = components::ValidateSpec(model, tables); !valid)
    return Refused(valid.error());
  return {};
}

std::expected<void, std::string> CheckArtifactPrerequisites(const artifact::Artifact& artifact) {
  if (auto served = RequireServedVersion(artifact); !served) return Refused(served.error());
  if (!OriginsWellFormed(artifact.sources()))
    return Refused("artifact source receipts are empty or duplicate");
  if (spec::FindMapping(artifact.model().architecture, artifact.model().container) == nullptr)
    return Refused("architecture has no component mapping");
  return CheckScaleAtomSlots(artifact);
}

std::expected<void, std::string> CheckComponentArtifactAdmission(
    const artifact::Artifact& artifact) {
  const spec::Mapping* mapping =
      spec::FindMapping(artifact.model().architecture, artifact.model().container);
  if (mapping != nullptr && mapping->role == spec::StageRole::kDrafter) {
    auto admitted = ReadComponentDrafterAdmission(artifact);
    if (!admitted) return std::unexpected(admitted.error());
    return {};
  }
  auto admitted = ReadComponentAdmission(artifact);
  if (!admitted) return std::unexpected(admitted.error());
  return {};
}

std::expected<ComponentDrafterAdmission, std::string> ReadComponentDrafterAdmission(
    const artifact::Artifact& drafter) {
  if (auto checked = CheckArtifactPrerequisites(drafter); !checked)
    return std::unexpected(checked.error());
  std::optional<spec::ModelSpec> model;
  if (drafter.spec_json().has_value()) {
    auto stored = spec::StoredSpec(drafter);
    if (!stored) return Refused(stored.error());
    model = std::move(*stored);
  }
  auto source = spec::SourceOf(drafter);
  if (!source) return Refused(source.error());
  if (auto checked = CheckComponentDrafterAdmission(*source, drafter.sources()); !checked)
    return std::unexpected(checked.error());
  return ComponentDrafterAdmission{
      .model = std::move(model),
      .source = std::move(*source),
      .approval = FindApproval(drafter.model().architecture, drafter.sources())};
}

std::expected<void, std::string> CheckScaleAtomSlots(const artifact::Artifact& artifact) {
  std::map<std::uint32_t, std::vector<std::uint32_t>> layers;
  const auto arrays = artifact.expert_arrays();
  for (std::uint32_t a = 0; a < arrays.size(); ++a) layers[arrays[a].layer].push_back(a);
  for (auto& [layer, list] : layers) {
    const auto atoms = [&](std::uint32_t a) {
      return arrays[a].format && arrays[a].format->layout == format::Layout::kScaleAtoms;
    };
    const bool stacked = std::ranges::any_of(
        list, [&](std::uint32_t a) { return atoms(a) && arrays[a].format->stacked > 1; });
    if (!stacked) continue;
    if (!std::ranges::all_of(list, atoms))
      return Refused("a layer mixes scale-atom expert tables with other layouts");
    // The stacked table first, each table's codes before its scales.
    std::ranges::stable_sort(list, [&](std::uint32_t a, std::uint32_t b) {
      return std::pair(-static_cast<int>(arrays[a].format->stacked), arrays[a].name) <
             std::pair(-static_cast<int>(arrays[b].format->stacked), arrays[b].name);
    });
    for (std::uint32_t e = 0; e < arrays[list.front()].count; ++e) {
      std::uint64_t at = 0;
      std::optional<std::uint32_t> group;
      for (const auto a : list) {
        const auto place = artifact.SlicePlacement(a, e);
        if (!place || (group && place->group != *group) || place->offset.value() != at)
          return Refused(std::format(
              "layer {}'s scale-atom expert tables are not back to back in expert {}'s slot", layer,
              e));
        group = place->group;
        at += place->bytes.value();
      }
    }
  }
  return {};
}

std::expected<void, std::string> CheckStoredHashTables(const artifact::Artifact& artifact,
                                                       const spec::ModelSpec& model) {
  for (const spec::Stage& stage : model.stages) {
    for (const spec::SideInput& side : stage.side) {
      if (side.source.kind != "side.hash") continue;
      const spec::Param* experts_param = side.source.Find("experts");
      const spec::Param* used_param = side.source.Find("used");
      const auto* experts =
          experts_param ? std::get_if<std::int64_t>(&experts_param->value) : nullptr;
      const auto* used = used_param ? std::get_if<std::int64_t>(&used_param->value) : nullptr;
      if (experts == nullptr || used == nullptr || *experts <= 0 || *used <= 0)
        return Unexpected("stored hash routing has no expert count");
      for (const spec::WeightBinding& binding : side.source.weights) {
        auto index = artifact.FindResource(binding.tensor);
        if (!index) return Unexpected("stored hash-routing table is missing");
        auto place = artifact.ResourcePlacement(*index);
        const auto& format = artifact.resources()[*index].format;
        // Its entries are read as I32: a table of any other encoding is not
        // reinterpreted.
        if (!format || format->encoding != format::Encoding::kI32 ||
            format->layout != format::Layout::kDense)
          return Unexpected("stored hash-routing table is not dense I32");
        if (format->shape.rank != 2 || format->shape.ne[0] != static_cast<std::uint64_t>(*used) ||
            format->shape.ne[1] == 0 || format->shape.ne[1] > (std::uint64_t{1} << 32U))
          return Unexpected("stored hash-routing table has another shape");
        const auto count = format->shape.ne[0] * format->shape.ne[1];
        if (!place || place->bytes.value() != count * sizeof(std::int32_t))
          return Unexpected("stored hash-routing table has an inconsistent byte count");
        const auto& group = artifact.groups()[place->group];
        const auto& shard = artifact.shards()[group.shard];
        if ((shard.data_offset.value() + group.offset.value() + place->offset.value()) %
            sizeof(std::int32_t))
          return Unexpected("stored hash-routing table is unaligned");
        // Aligned: each window holds whole entries (4,096 is a multiple of 4).
        bool invalid = false;
        auto read =
            ReadPlaced(artifact, *place,
                       [&](std::uint64_t,
                           std::span<const std::byte> bytes) -> std::expected<void, std::string> {
                         for (std::size_t at = 0; at + sizeof(std::int32_t) <= bytes.size();
                              at += sizeof(std::int32_t)) {
                           std::int32_t expert = 0;
                           std::memcpy(&expert, bytes.data() + at, sizeof(expert));
                           if (expert < 0 || std::cmp_greater_equal(expert, *experts)) {
                             invalid = true;
                             return std::unexpected("invalid");
                           }
                         }
                         return {};
                       });
        if (invalid) return Unexpected("stored hash-routing table names an invalid expert");
        if (!read) return Unexpected("reading stored hash-routing table: " + read.error());
      }
    }
  }
  return {};
}

std::expected<ComponentAdmission, std::string> ReadComponentAdmission(
    const artifact::Artifact& artifact) {
  if (auto checked = CheckArtifactPrerequisites(artifact); !checked)
    return std::unexpected(checked.error());
  auto model = spec::StoredSpec(artifact);
  if (!model) return Refused(model.error());
  auto source = spec::SourceOf(artifact);
  if (!source) return Refused(source.error());
  if (auto checked = CheckComponentAdmission(*model, *source, artifact.sources()); !checked)
    return std::unexpected(checked.error());
  return ComponentAdmission{std::move(*model), std::move(*source),
                            FindApproval(artifact.model().architecture, artifact.sources())};
}

std::uint32_t MappingDefaultContext(const spec::Mapping& mapping,
                                    std::span<const spec::SourceValue> gates) {
  std::uint32_t widest = 0;
  for (const spec::PolicyRule& rule : mapping.policies) {
    if (!rule.gate.empty()) {
      const auto at = std::ranges::find(gates, rule.gate, &spec::SourceValue::key);
      bool active = false;
      if (at != gates.end()) {
        if (const auto* flag = std::get_if<bool>(&at->value)) {
          active = *flag;
        } else if (const auto* n = std::get_if<std::int64_t>(&at->value)) {
          active = *n > 0;
        }
      }
      if (rule.gate_absent ? active : !active) continue;
    }
    if (rule.max_context != 0 && !rule.owners_alone) widest = std::max(widest, rule.max_context);
  }
  return widest;
}

std::uint32_t ApprovedDefaultContext(const Approval& approval) {
  const spec::Mapping* mapping = spec::FindMapping(approval.architecture, approval.container);
  std::uint32_t widest = 0;
  if (mapping != nullptr && approval.policy != ApprovedPolicy::kNone) {
    std::vector<spec::SourceValue> gates;
    gates.reserve(approval.gates.size());
    for (const std::string_view gate : approval.gates)
      gates.push_back({.key = std::string(gate), .value = true});
    widest = MappingDefaultContext(*mapping, gates);
  }
  return widest != 0 ? std::min(widest, approval.max_context) : approval.max_context;
}

std::expected<ImageAdmission, std::string> ReadImageAdmission(
    const artifact::Composition& composition,
    std::span<const artifact::Artifact* const> components) {
  if (composition.format_version() != 1 && composition.format_version() != 2)
    return Refused(std::format(
        "composition {} ({}) is format version {}; serving reads versions 1 and 2 (D-118): "
        "re-import its source pipeline with `llmp import SOURCE --store DIRECTORY` and configure "
        "the new composition ID",
        composition.id(), composition.architecture(), composition.format_version()));
  if (!OriginsWellFormed(composition.sources()))
    return Refused("the composition's source receipts are empty or duplicate");
  const spec::ImageMapping* mapping = spec::FindImageMapping(composition.architecture());
  if (mapping == nullptr) return Refused("the composition's architecture has no image mapping");
  const ImageApproval* approval =
      FindImageApproval(composition.architecture(), composition.sources());
  const auto& named = composition.components();
  if (named.size() != mapping->components.size() || components.size() != named.size() ||
      (approval != nullptr && approval->components.size() != named.size()))
    return Refused("the composition's components are not its image mapping's");
  for (std::size_t i = 0; i < named.size(); ++i) {
    const auto& rule = mapping->components[i];
    const artifact::Artifact* a = components[i];
    if (a == nullptr || named[i].role != rule.role || a->id() != named[i].artifact ||
        a->format_version() != 1 ||
        !std::ranges::contains(rule.architectures, a->model().architecture) ||
        !a->expert_arrays().empty() || !OriginsWellFormed(a->sources()))
      return Refused(
          std::format("image component {} is not a valid v1 component artifact", rule.role));
    auto source = spec::SourceOf(*a);
    if (!source) return Refused(source.error());
    auto stored = spec::StoredSpec(*a);
    if (!stored) return Refused(stored.error());
    if (approval != nullptr) {
      const Approval& want = approval->components[i];
      if (a->model().architecture != want.architecture)
        return Refused(std::format("image component {} is not its approved artifact", rule.role));
      if (!SourcesMatch(want.sources, a->sources()) || !Transforms(*a, want) ||
          a->resources().size() != want.tensors)
        return Refused(std::format("image component {}'s source identity or inventory is not {}'s",
                                   rule.role, want.name));
      if (!SourceMatches(*source, want))
        return Refused("source container, transforms or tensor inventory differs from its profile");
      if (auto inventory = internal::CheckInventory(*source, want); !inventory)
        return std::unexpected(inventory.error());
      if (auto pinned = Pinned(want, *stored); !pinned) return std::unexpected(pinned.error());
    } else {
      for (const auto& t : source->tensors) {
        if (!Admissible(t)) return Refused(kRefusedInventory);
      }
    }
    if (const auto valid = components::ValidateSpec(*stored, source->tensors); !valid)
      return Refused(valid.error());
  }
  auto family = spec::StoredSpec(composition, components);
  if (!family) return Refused(family.error());
  if (approval != nullptr) {
    if (const auto actual = base::ToHex(family->Digest()); actual != approval->spec)
      return Refused(std::format("the family spec ({}) is not the one approved for {} ({})", actual,
                                 approval->name, approval->spec));
  }
  auto profile = spec::ImageProfileOf(*family, *mapping);
  if (!profile) return Refused(profile.error());
  // Import preserves source encodings; the current image kernels have a
  // narrower execution contract. Report that boundary before building graphs.
  for (std::size_t i = 0; i < named.size(); ++i) {
    const auto kind = mapping->components[i].kinds.front();
    const auto expected = kind == spec::kImageTextKind
                              ? model::TextEncoderTensors(profile->text, mapping->names.text)
                          : kind == spec::kImageFlowKind
                              ? model::DenoiserTensors(profile->denoiser, mapping->names.denoiser)
                              : model::VaeDecoderTensors(profile->vae, mapping->names.vae);
    const auto source = spec::SourceOf(*components[i]);
    if (!source) return Refused(source.error());
    for (const auto& need : expected) {
      const auto tensor = std::ranges::find(source->tensors, need.name, &spec::SourceTensor::name);
      if (tensor == source->tensors.end())
        return Refused(
            std::format("image component {} tensor {}: current image runtime requires {}; source "
                        "encoding is preserved by import",
                        named[i].role, need.name, need.dtype));
      if (need.dtype == "F32") {
        if (tensor->format.encoding != format::Encoding::kF32)
          return Refused(
              std::format("image component {} tensor {}: current image runtime requires {}; source "
                          "encoding is preserved by import",
                          named[i].role, need.name, need.dtype));
      } else {
        if (tensor->format.encoding != format::Encoding::kBF16 || !Admissible(*tensor))
          return Refused(
              std::format("image component {} tensor {}: current image runtime requires {}; source "
                          "encoding is preserved by import",
                          named[i].role, need.name, need.dtype));
      }
    }
  }
  return ImageAdmission{.mapping = mapping,
                        .approval = approval,
                        .family = std::move(*family),
                        .profile = std::move(*profile)};
}

namespace {
std::expected<base::Sha256Digest, std::string> NumericalDigest(std::string_view hex) {
  base::Sha256Digest digest{};
  if (hex.size() != digest.size() * 2) return Refused("numerical evidence has no SHA-256 identity");
  for (std::size_t i = 0; i < digest.size(); ++i) {
    const auto digit = [](char c) -> int {
      return c >= '0' && c <= '9' ? c - '0' : c >= 'a' && c <= 'f' ? c - 'a' + 10 : -1;
    };
    const auto high = digit(hex[2 * i]), low = digit(hex[2 * i + 1]);
    if (high < 0 || low < 0) return Refused("numerical evidence identity is not canonical SHA-256");
    digest[i] = static_cast<std::uint8_t>(high * 16 + low);
  }
  return digest;
}
base::Sha256Digest ActualSourceIdentity(std::span<const artifact::Source> actual) {
  base::Sha256 hash;
  hash.Update("llmp.numerical.source-receipts.v1\n");
  const base::json::FragmentSink sink = [&](std::string_view text) { hash.Update(text); };
  const base::json::Stream out(sink);
  std::vector<const artifact::Source*> sorted;
  sorted.reserve(actual.size());
  for (const auto& s : actual) sorted.push_back(&s);
  std::ranges::sort(sorted, {}, &artifact::Source::name);
  for (const auto* source : sorted) {
    out.Quoted(source->name);
    out.Number(source->bytes.value());
    hash.Update(std::as_bytes(std::span(source->sha256)));
  }
  return hash.Finish();
}

struct NumericalPayment {
  std::shared_ptr<const kernels::ConstructionFunder> funder;
  std::uint64_t bytes = 0;
  ~NumericalPayment() {
    if (funder) funder->release(bytes);
  }
};
// Count/fill the identical deterministic projection before constructing any
// retained fact vector. Temporary admission/spec inventories use the existing
// compiler/parser allowance; this owns only the new immutable output.
template <class Emit>
std::expected<std::shared_ptr<const planner::NumericalQualifications>, std::string> ProjectNumerics(
    const base::Sha256Digest& source, const base::Sha256Digest& semantic,
    const std::shared_ptr<const kernels::ConstructionFunder>& funder, const Emit& emit) {
  std::uint64_t count = 0;
  bool overflow = false;
  auto counted = emit([&](const planner::NumericalQualification&) {
    if (count == UINT64_MAX)
      overflow = true;
    else
      ++count;
  });
  if (!counted) return std::unexpected(counted.error());
  if (count == 0) return std::shared_ptr<const planner::NumericalQualifications>{};
  if (overflow || count > std::numeric_limits<std::size_t>::max())
    return Refused("numerical qualification count overflows");
  auto storage = planner::NumericalQualificationBytes(static_cast<std::size_t>(count));
  if (!storage || *storage > UINT64_MAX - sizeof(NumericalPayment) - 64)
    return Refused("numerical qualification payment overflows");
  const auto bytes = *storage + sizeof(NumericalPayment) + 64;
  if (!funder || !funder->charge || !funder->release || !funder->charge(bytes))
    return Refused("numerical qualifications are not prepaid");
  auto payment = std::make_shared<NumericalPayment>();
  payment->funder = funder;
  payment->bytes = bytes;
  std::vector<planner::NumericalQualification> entries;
  entries.reserve(static_cast<std::size_t>(count));
  auto filled =
      emit([&](const planner::NumericalQualification& entry) { entries.push_back(entry); });
  if (!filled) return std::unexpected(filled.error());
  if (entries.size() != count) return Refused("numerical qualification projection changed");
  // Equivalent primitive sites may produce the same semantic grant. Keep
  // one lookup record; the paid vector capacity remains the exact pre-counted
  // construction inventory, including that deterministic temporary excess.
  std::ranges::sort(entries, {}, [](const auto& e) {
    return std::tuple(e.change, e.stage, e.component, e.anchor, e.min_rows, e.max_rows, e.max_k,
                      e.max_n, e.weight_role);
  });
  entries.erase(std::unique(entries.begin(), entries.end()), entries.end());
  return planner::NumericalQualifications::Bind(std::move(entries), source, semantic,
                                                std::move(payment));
}
}  // namespace

std::expected<std::shared_ptr<const planner::NumericalQualifications>, std::string>
AdmittedNumerics(const spec::ModelSpec& model, const spec::Source& source,
                 std::span<const artifact::Source> origins, const Approval*,
                 const base::Sha256Digest& admitted_semantic, const spec::Envelope& envelope,
                 std::uint16_t active_stage, const planner::DeviceFacts& device,
                 std::shared_ptr<const kernels::ConstructionFunder> funder) {
  // Origins bind a grant to the active source; historical approval membership
  // supplies neither permission nor a numerical default.
  if (!OriginsWellFormed(origins) || model.stages.empty() ||
      std::ranges::none_of(admitted_semantic, [](std::uint8_t byte) { return byte != 0; }))
    return Refused("numerical source facts differ from the active admitted source");
  if (model.stages.size() == 1 && model.Digest() != admitted_semantic)
    return Refused("numerical semantic identity differs from the admitted spec");
  for (const auto& tensor : source.tensors)
    if (!Admissible(tensor)) return Refused(kRefusedInventory);
  const std::size_t stage = model.stages.size() == 1 ? 0 : active_stage;
  if (stage >= model.stages.size()) return Refused("numerical stage is absent from the spec");
  auto instances = components::InstancesOf(model, stage);
  if (!instances) return std::unexpected(instances.error());
  const auto identity = ActualSourceIdentity(origins);
  const auto visit = [&](const auto& callback) {
    const auto optional = [&](const auto& instance) {
      if (instance) callback(*instance);
    };
    optional(instances->embedding);
    for (const auto& instance : instances->side) callback(instance);
    optional(instances->mixing);
    for (const auto& layer : instances->layers)
      for (const auto& instance : layer) callback(instance);
    optional(instances->final_norm);
    optional(instances->output);
  };
  return ProjectNumerics(
      identity, admitted_semantic, funder,
      [&](const auto& record) -> std::expected<void, std::string> {
        bool invalid = false;
        visit([&](const components::Instance& instance) {
          for (const auto& binding : instance.ref->weights) {
            // Foreign-stage roles need their own artifact's projection.
            if (!binding.stage.empty()) continue;
            const auto tensor =
                std::ranges::find(source.tensors, binding.tensor, &spec::SourceTensor::name);
            const auto ordinal = components::RoleIndex(*instance.def, binding.role);
            if (tensor == source.tensors.end() || !ordinal) {
              invalid = true;
              continue;
            }
            for (const auto& site : instance.def->numerical) {
              for (const auto anchor : site.anchors) {
                if (!components::NumericalSiteApplies(site, instance, anchor, binding.role,
                                                      tensor->format.encoding))
                  continue;
                for (const auto& evidence : ComponentNumericalEvidenceTable()) {
                  if (evidence.pipeline != NumericalPipeline::kLlm ||
                      evidence.change != site.change || evidence.semantic != site.semantic ||
                      !EvidenceDeviceMatches(evidence, device) ||
                      !std::ranges::contains(evidence.encodings, tensor->format.encoding))
                    continue;
                  if (evidence.rows.empty() &&
                      (envelope.slots == 0 || envelope.slots > evidence.owners))
                    continue;
                  auto digest = NumericalDigest(evidence.digest);
                  if (!digest) {
                    invalid = true;
                    continue;
                  }
                  std::uint64_t rows = evidence.rows_per_owner;
                  if (evidence.rows.empty() &&
                      (__builtin_mul_overflow(rows, std::uint64_t{envelope.slots}, &rows) ||
                       (evidence.stream_rows &&
                        __builtin_mul_overflow(rows, std::uint64_t{instance.stage.streams},
                                               &rows)) ||
                       (!evidence.row_parameter.empty() &&
                        __builtin_mul_overflow(
                            rows,
                            static_cast<std::uint64_t>(instance.params.Int(evidence.row_parameter)),
                            &rows)))) {
                    invalid = true;
                    continue;
                  }
                  const auto& shape = tensor->format.shape;
                  std::uint64_t k = 0, n = 0;
                  std::uint16_t role = *ordinal;
                  if (anchor == ir::Primitive(ir::Op::kRmsNorm)) {
                    if (shape.rank != 1 || instance.stage.streams == 0 ||
                        (evidence.stream_rows && shape.ne[0] % instance.stage.streams != 0)) {
                      invalid = true;
                      continue;
                    }
                    k = evidence.stream_rows ? shape.ne[0] / instance.stage.streams : shape.ne[0];
                    role = UINT16_MAX;
                  } else if (anchor == ir::Primitive(ir::Op::kExpertReduce)) {
                    if (shape.rank < 2) {
                      invalid = true;
                      continue;
                    }
                    k = shape.ne[1];
                  } else {
                    if (shape.rank < 2) {
                      invalid = true;
                      continue;
                    }
                    k = shape.ne[0];
                    n = shape.ne[1];
                  }
                  const auto emit = [&](std::uint64_t low, std::uint64_t high) {
                    record(planner::NumericalQualification{
                        .change = site.change,
                        .evidence = *digest,
                        .source = identity,
                        .semantic = admitted_semantic,
                        .stage = active_stage,
                        .component = instance.index,
                        .anchor = anchor,
                        .arch = evidence.arch,
                        .min_rows = low,
                        .max_rows = high,
                        .max_k = k,
                        .max_n = n,
                        .exact_dimensions = true,
                        .incumbent_recipe = evidence.incumbent_recipe,
                        .weight_role = role});
                  };
                  if (evidence.rows.empty())
                    emit(1, rows);
                  else
                    for (const auto row : evidence.rows) emit(row, row);
                }
              }
            }
          }
        });
        if (invalid) return Refused("numerical component binding or evidence domain is invalid");
        return {};
      });
}

std::expected<std::shared_ptr<const planner::NumericalQualifications>, std::string>
AdmittedNumerics(const spec::ModelSpec& model, const spec::Source& source,
                 std::span<const artifact::Source> origins, const Approval* approval,
                 const base::Sha256Digest& admitted_semantic, const spec::Envelope& envelope,
                 std::uint16_t active_stage, const kernels::StepBackend& backend) {
  const auto& effective = backend.owner ? backend.owner->backend() : backend;
  return AdmittedNumerics(model, source, origins, approval, admitted_semantic, envelope,
                          active_stage, effective.device, backend.ConstructionFunding());
}

std::expected<std::shared_ptr<const planner::NumericalQualifications>, std::string>
AdmittedImageNumerics(const spec::ModelSpec& family, const ImageApproval*,
                      const artifact::Composition& composition, const ir::Graph& graph,
                      const spec::ModelSpec& stage_spec, std::uint16_t stage,
                      const planner::DeviceFacts& device,
                      std::shared_ptr<const kernels::ConstructionFunder> funder) {
  if (!OriginsWellFormed(composition.sources()))
    return Refused("image numerical evidence has no valid active source identity");
  if (stage_spec.stages.size() != 1)
    return Refused("image numerical projection needs the generated stage spec");
  auto instances = components::InstancesOf(stage_spec, 0);
  if (!instances) return std::unexpected(instances.error());
  std::vector<const components::Instance*> by_component;
  const auto retain = [&](const auto& instance) {
    if (by_component.size() <= instance.index) by_component.resize(instance.index + 1);
    by_component[instance.index] = &instance;
  };
  if (instances->embedding) retain(*instances->embedding);
  for (const auto& instance : instances->side) retain(instance);
  if (instances->mixing) retain(*instances->mixing);
  for (const auto& layer : instances->layers)
    for (const auto& instance : layer) retain(instance);
  if (instances->final_norm) retain(*instances->final_norm);
  if (instances->output) retain(*instances->output);
  const auto identity = ActualSourceIdentity(composition.sources());
  const auto semantic = family.Digest();
  return ProjectNumerics(
      identity, semantic, funder, [&](const auto& record) -> std::expected<void, std::string> {
        for (const auto& node : graph.nodes()) {
          if (node.component >= by_component.size() || !by_component[node.component] ||
              node.nsrc == 0 || !node.src[0].valid())
            continue;
          const auto& instance = *by_component[node.component];
          for (const auto& site : instance.def->numerical) {
            if (!std::ranges::contains(site.origins.empty() ? site.anchors : site.origins,
                                       node.key))
              continue;
            const auto weight = planner::SelectionView::ResolveWeightOperand(graph, node, 0);
            std::string_view role;
            auto encoding = graph.tensor(node.src[0]).format.encoding;
            std::uint16_t ordinal = UINT16_MAX;
            if (!site.roles.empty()) {
              if (!weight || weight->reference.stage != stage ||
                  weight->reference.component != node.component ||
                  weight->reference.role >= instance.def->roles.size())
                continue;
              ordinal = weight->reference.role;
              role = instance.def->roles[ordinal].name;
              encoding = graph.tensor(weight->leaf).format.encoding;
            }
            for (const auto anchor : site.anchors) {
              if (!components::NumericalSiteApplies(site, instance, anchor, role, encoding))
                continue;
              for (const auto& evidence : ComponentNumericalEvidenceTable()) {
                if (evidence.pipeline != NumericalPipeline::kImage ||
                    evidence.change != site.change || evidence.semantic != site.semantic ||
                    !EvidenceDeviceMatches(evidence, device) ||
                    !std::ranges::contains(evidence.encodings, encoding))
                  continue;
                auto digest = NumericalDigest(evidence.digest);
                if (!digest) return std::unexpected(digest.error());
                std::array<const ir::TensorDesc*, ir::kMaxSources> sources{};
                std::array<const ir::TensorDesc*, ir::kMaxOutputs> outputs{};
                for (std::size_t i = 0; i < node.nsrc; ++i)
                  if (node.src[i].valid()) sources[i] = &graph.tensor(node.src[i]);
                for (std::size_t i = 0; i < node.nout; ++i) outputs[i] = &graph.tensor(node.out[i]);
                const auto shape =
                    planner::ShapeOfNode(node.key, std::span(sources).first(node.nsrc),
                                         std::span(outputs).first(node.nout));
                if (evidence.attention_head_dim != 0 && (shape.k != evidence.attention_head_dim ||
                                                         shape.n != evidence.attention_head_dim))
                  continue;
                record(
                    planner::NumericalQualification{.change = site.change,
                                                    .evidence = *digest,
                                                    .source = identity,
                                                    .semantic = semantic,
                                                    .stage = stage,
                                                    .component = node.component,
                                                    .anchor = anchor,
                                                    .arch = evidence.arch,
                                                    .min_rows = shape.rows,
                                                    .max_rows = shape.rows,
                                                    .max_k = shape.k,
                                                    .max_n = shape.n,
                                                    .exact_shape = true,
                                                    .incumbent_recipe = evidence.incumbent_recipe,
                                                    .weight_role = ordinal});
              }
            }
          }
        }
        return {};
      });
}

std::expected<const spec::PolicyRule*, std::string> ApprovedRule(const Approval& approval,
                                                                 const spec::Envelope& envelope) {
  if (approval.policy == ApprovedPolicy::kNone) return nullptr;
  const spec::Mapping* mapping = spec::FindMapping(approval.architecture, approval.container);
  if (mapping == nullptr) return Refused("the approval's architecture has no mapping");
  // The gates the checkpoint holds, as flags: the approval's, not a key bag
  // read again or a profile's.
  std::vector<spec::SourceValue> gates;
  for (const std::string_view gate : approval.gates)
    gates.push_back({.key = std::string(gate), .value = true});
  auto rule = spec::DefaultRule(*mapping, gates, envelope);
  if (!rule) return Refused(rule.error());
  return *rule;
}

bool WithinApprovedEnvelope(const Approval& approval, const spec::Envelope& envelope) {
  return envelope.context <= approval.max_context && envelope.slots <= approval.max_slots &&
         (approval.wide_context == 0 || envelope.context <= approval.wide_context ||
          envelope.slots == 1);
}

std::expected<spec::PolicyRule, std::string> ComponentAdmissionPolicy(
    const spec::ModelSpec& model, const spec::Source& source,
    std::span<const artifact::Source> origins, const spec::Envelope& envelope) {
  if (auto checked = CheckComponentAdmission(model, source, origins); !checked)
    return std::unexpected(checked.error());
  return AdmittedPolicy(model, source, FindApproval(source.architecture, origins), envelope);
}

std::expected<spec::PolicyRule, std::string> AdmittedPolicy(const spec::ModelSpec& model,
                                                            const spec::Source& source,
                                                            const Approval* approval,
                                                            const spec::Envelope& envelope) {
  // The stored stage context is a real bound (the spec's position tables).
  if (model.stages.empty() || envelope.context == 0 ||
      envelope.context > model.stages.front().context || envelope.slots == 0)
    return Refused("execution envelope exceeds the stored stage context or has zero slots");
  if (approval != nullptr) {
    // The approved envelope is a qualified default (D-119; the settings
    // resolve within it unless overridden, with a warning), not a refusal:
    // past it, no approved rule is promised and the engine's default plans.
    auto rule = ApprovedRule(*approval, envelope);
    if (!rule) return std::unexpected(rule.error());
    if (*rule != nullptr) return **rule;
    if (approval->policy == ApprovedPolicy::kMapping && WithinApprovedEnvelope(*approval, envelope))
      return Refused("no approved policy for this execution envelope");
    return spec::PolicyRule{};
  }
  const spec::Mapping* mapping = spec::FindMapping(source.architecture, source.container);
  if (mapping != nullptr) {
    const auto gates = spec::PolicyGates(*mapping, model, source.container);
    auto rule = spec::DefaultRule(*mapping, gates, envelope);
    if (!rule) return Refused(rule.error());
    if (*rule != nullptr) return **rule;
    if (mapping->serving != nullptr && mapping->serving->serving_rule != nullptr)
      return *mapping->serving->serving_rule;
  }
  return spec::PolicyRule{};
}

}  // namespace llmp::runtime
