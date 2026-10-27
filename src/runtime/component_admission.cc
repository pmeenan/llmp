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

#include "artifact/schema.h"
#include "base/json_stream.h"
#include "base/sha256.h"
#include "components/state_encoding.h"
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

}  // namespace

std::expected<void, std::string> ComponentDrafterEnvelope(const spec::Envelope& envelope) {
  // An approved drafter's envelope is its settings' default (model_settings
  // .cc), not a bound here: an override past it is served, with a warning.
  if (envelope.context == 0 || envelope.slots == 0)
    return Refused("drafter execution envelope must have non-zero context and slots");
  return {};
}

std::optional<std::uint64_t> SelectedHeadStorageBytes(const artifact::Resource& r) {
  if (!r.format || r.roles.empty() || r.part != format::PartRole::kData ||
      r.format->encoding != format::Encoding::kI32 || r.format->layout != format::Layout::kDense ||
      r.format->shape.rank != 2 || r.format->shape.ne[0] != 1 || r.format->shape.ne[1] == 0 ||
      r.format->shape.ne[1] > UINT32_MAX)
    return std::nullopt;
  std::uint64_t bytes = 0;
  if (__builtin_mul_overflow(r.format->shape.ne[1], std::uint64_t{sizeof(std::int32_t)}, &bytes) ||
      bytes != r.bytes.value())
    return std::nullopt;
  return bytes;
}

std::expected<std::optional<DraftHeadMap>, std::string> DraftHeadRows(
    const artifact::Artifact& drafter, std::string_view head_ids, std::uint64_t vocab) {
  if (head_ids.empty()) return std::nullopt;
  const auto index = drafter.FindResource(head_ids);
  if (!index) return std::nullopt;
  const auto& resource = drafter.resources()[*index];
  const auto bytes = SelectedHeadStorageBytes(resource);
  const auto rows = bytes ? *bytes / sizeof(std::int32_t) : 0;
  if (rows == 0 || rows > vocab)
    return Refused("a selected draft head's token map is not I32 [1, rows] within the vocabulary");
  return DraftHeadMap{.resource = *index, .rows = rows};
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
  // The final graph includes borrowed roles. Each receipt belongs to the
  // artifact holding those tensors, regardless of which stage uses them.
  if (auto folds = components::ValidateFoldReceipts(composed, 0, target); !folds)
    return Refused(folds.error());
  if (auto folds = components::ValidateFoldReceipts(composed, 1, drafter); !folds)
    return Refused(folds.error());
  const std::array<std::span<const spec::SourceTensor>, 2> tables{target.tensors, drafter.tensors};
  if (const auto valid = components::ValidateSpec(composed, tables); !valid)
    return Refused(valid.error());
  return {};
}

std::expected<void, std::string> CheckComponentDrafterAdmission(
    const spec::ModelSpec& model, const spec::Source& source,
    std::span<const artifact::Source> origins) {
  if (!OriginsWellFormed(origins)) return Refused("source origins are empty or duplicate");
  if (!artifact::schema::IsName(source.architecture) ||
      !artifact::schema::IsContainerName(source.container))
    return Refused("invalid source provenance names");
  if (model.pipeline != spec::PipelineKind::kLlm || model.stages.size() != 2 ||
      model.stages[0].role != spec::StageRole::kTarget || model.stages[0].artifact != 0 ||
      model.stages[1].role != spec::StageRole::kDrafter || model.stages[1].artifact != 1 ||
      !model.stages[1].drafting)
    return Refused("stored drafter needs its target interface and execution contract");
  if (auto valid = spec::Validate(model); !valid) return Refused(valid.error());
  if (source.tensors.empty() ||
      std::ranges::any_of(source.tensors, [](const auto& t) { return !Admissible(t); }))
    return Refused(kRefusedInventory);
  if (auto output = spec::DrafterOutput(model.stages[1]); !output) return Refused(output.error());
  if (auto folds = components::ValidateFoldReceipts(model, 1, source); !folds)
    return Refused(folds.error());
  return {};
}

std::expected<void, std::string> CheckComponentAdmission(
    const spec::ModelSpec& model, const spec::Source& source,
    std::span<const artifact::Source> origins) {
  if (!artifact::schema::IsName(source.architecture))
    return Refused("source architecture is not a valid provenance name");
  if (!artifact::schema::IsContainerName(source.container))
    return Refused("source container is not in the artifact schema");
  if (!OriginsWellFormed(origins)) return Refused("source origins are empty or duplicate");
  if (model.pipeline != spec::PipelineKind::kLlm || model.stages.size() != 1 ||
      model.stages[0].role != spec::StageRole::kTarget || model.stages[0].artifact != 0 ||
      model.stages[0].context == 0)
    return Refused("the stored pipeline is not a standalone target");
  if (source.tensors.empty() ||
      std::ranges::any_of(source.tensors, [](const auto& t) { return !Admissible(t); }))
    return Refused(kRefusedInventory);
  if (auto folds = components::ValidateFoldReceipts(model, 0, source); !folds)
    return Refused(folds.error());
  const std::array<std::span<const spec::SourceTensor>, 1> tables{source.tensors};
  if (const auto valid = components::ValidateSpec(model, tables); !valid)
    return Refused(valid.error());
  return {};
}

std::expected<void, std::string> CheckArtifactPrerequisites(const artifact::Artifact& artifact) {
  if (auto served = RequireServedVersion(artifact); !served) return Refused(served.error());
  if (!OriginsWellFormed(artifact.sources()))
    return Refused("artifact source receipts are empty or duplicate");
  return CheckScaleAtomSlots(artifact);
}

std::expected<void, std::string> CheckComponentArtifactAdmission(
    const artifact::Artifact& artifact) {
  if (auto checked = CheckArtifactPrerequisites(artifact); !checked) return checked;
  auto stored = spec::StoredSpec(artifact);
  if (!stored) return Refused(stored.error());
  auto source = spec::SourceOf(artifact);
  if (!source) return Refused(source.error());
  if (stored->stages.size() == 2 && stored->stages.back().role == spec::StageRole::kDrafter)
    return CheckComponentDrafterAdmission(*stored, *source, artifact.sources());
  return CheckComponentAdmission(*stored, *source, artifact.sources());
}

std::expected<ComponentDrafterAdmission, std::string> ReadComponentDrafterAdmission(
    const artifact::Artifact& drafter) {
  if (auto checked = CheckArtifactPrerequisites(drafter); !checked)
    return std::unexpected(checked.error());
  auto model = spec::StoredSpec(drafter);
  if (!model) return Refused(model.error());
  auto source = spec::SourceOf(drafter);
  if (!source) return Refused(source.error());
  if (auto checked = CheckComponentDrafterAdmission(*model, *source, drafter.sources()); !checked)
    return std::unexpected(checked.error());
  return ComponentDrafterAdmission{
      .model = std::move(*model),
      .source = std::move(*source),
      .approval = FindApproval(drafter.model().architecture, drafter.sources())};
}

std::expected<std::shared_ptr<const PreparedDrafterComposition>, std::string>
PreparedDrafterComposition::Make(const artifact::Artifact& target,
                                 const spec::ModelSpec& target_model,
                                 const spec::Source& target_source,
                                 const artifact::Artifact& drafter,
                                 const ComponentDrafterAdmission& admitted,
                                 std::uint32_t vocabulary, std::uint32_t requested_vocabulary) {
  if (!admitted.model || admitted.model->stages.size() != 2 ||
      !admitted.model->stages.back().drafting)
    return Refused("component composition requires stored drafter execution semantics");
  auto out = std::shared_ptr<PreparedDrafterComposition>(new PreparedDrafterComposition);
  out->target_ = target.id();
  out->drafter_ = drafter.id();
  out->target_semantic_ = target_model.Digest();
  out->target_vocabulary_ = vocabulary;
  auto canonical = spec::ComposeStoredDrafter(target_model, *admitted.model);
  if (!canonical) return Refused(canonical.error());
  if (auto checked = CheckComposedAdmission(*canonical, target_source, admitted.source); !checked)
    return Refused(checked.error());
  out->numerical_semantic_ = canonical->Digest();
  out->model_ = std::move(*canonical);
  const auto& drafting = *out->model_.stages.back().drafting;
  if (const auto* block = std::get_if<spec::BlockDraft>(&drafting);
      block && std::cmp_greater_equal(block->fill_token, vocabulary))
    return Refused("block fill token is outside the actual target vocabulary");
  out->narrowable_ = std::holds_alternative<spec::MtpDraft>(drafting);
  {
    auto output = spec::DrafterOutput(out->model_.stages.back());
    if (!output) return Refused(output.error());
    std::string_view map_name;
    for (const auto& w : (*output)->weights)
      if (w.role == "ids") map_name = w.tensor;
    auto map = DraftHeadRows(drafter, map_name, vocabulary);
    if (!map) return Refused(map.error());
    if (!map_name.empty() && !*map) return Refused("declared selected head map is absent");
    std::optional<std::uint32_t> weight_rows;
    for (const auto& binding : (*output)->weights) {
      if (binding.role != "weight") continue;
      const auto* holder = spec::HoldingStage(out->model_, out->model_.stages.back(), binding);
      if (!holder || holder->artifact > 1 || binding.experts)
        return Refused("draft output weight has no dense holding artifact");
      const auto& source = holder->artifact == 0 ? target_source : admitted.source;
      const auto tensor = std::ranges::find_if(source.tensors, [&](const auto& t) {
        return t.name == binding.tensor && t.part == binding.part && !t.experts;
      });
      if (tensor == source.tensors.end() || tensor->format.shape.rank != 2 ||
          tensor->format.shape.ne[1] == 0 || tensor->format.shape.ne[1] > vocabulary)
        return Refused("draft output weight rows exceed the actual target vocabulary");
      const auto rows = static_cast<std::uint32_t>(tensor->format.shape.ne[1]);
      if (weight_rows && *weight_rows != rows)
        return Refused("draft output weight parts disagree on rows");
      weight_rows = rows;
    }
    if (!weight_rows || (*map && (*map)->rows != *weight_rows))
      return Refused("draft output and selected map row counts disagree");
    std::uint32_t declared_rows = 0;
    if (const auto* param = (*output)->Find("rows")) {
      const auto* rows = std::get_if<std::int64_t>(&param->value);
      if (!rows || *rows < 0 || std::cmp_greater(*rows, *weight_rows))
        return Refused("stored draft output rows exceed its actual weight");
      declared_rows = static_cast<std::uint32_t>(*rows);
    }
    const auto available = out->narrowable_ || declared_rows == 0 ? *weight_rows : declared_rows;
    const auto width = !out->narrowable_ || requested_vocabulary == 0
                           ? available
                           : std::min(requested_vocabulary, available);
    if (width == 0 || available > vocabulary)
      return Refused("a draft head needs a positive row count within the target vocabulary");
    out->available_vocabulary_ = available;
    out->vocabulary_ = width;
    if (*map) {
      auto ids = ReadRecipeIntegers<std::int32_t>(drafter, (*map)->resource, (*map)->rows);
      if (!ids) return Refused(ids.error());
      for (std::size_t i = 0; i < ids->size(); ++i)
        if ((*ids)[i] < 0 || std::cmp_greater_equal((*ids)[i], vocabulary) ||
            (i != 0 && (*ids)[i] <= (*ids)[i - 1]))
          return Refused("a selected draft head's token IDs are not strictly ascending");
      ids->resize(width);
      out->head_ids_ = std::make_shared<const std::vector<std::int32_t>>(std::move(*ids));
    }
    // Canonical composition/admission above happens once. Only the head extent
    // changes for the configured vocabulary; taps and stage bindings survive.
    if (out->narrowable_) {
      if (auto configured = spec::ConfigureComposedDraftVocabulary(out->model_, width); !configured)
        return Refused(configured.error());
    }
    if (auto valid = CheckComposedAdmission(out->model_, target_source, admitted.source); !valid)
      return Refused(valid.error());
  }

  return std::shared_ptr<const PreparedDrafterComposition>(std::move(out));
}

std::expected<void, std::string> PreparedDrafterComposition::CheckBinding(
    const artifact::Artifact& target, const artifact::Artifact& drafter,
    const spec::ModelSpec& target_model, std::uint32_t vocabulary,
    std::uint32_t requested_vocabulary) const {
  if (target.id() != target_ || drafter.id() != drafter_ ||
      target_model.Digest() != target_semantic_ || vocabulary != target_vocabulary_)
    return Refused("prepared composition differs from its artifact or target structure binding");
  if (narrowable_ && vocabulary_ &&
      *vocabulary_ != (requested_vocabulary == 0
                           ? available_vocabulary_
                           : std::min(requested_vocabulary, available_vocabulary_)))
    return Refused("prepared composition differs from the effective draft vocabulary");
  return {};
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
    for (const auto a : list) {
      if (arrays[a].count != arrays[list.front()].count) {
        return Refused(
            std::format("layer {}'s scale-atom expert tables have different expert counts", layer));
      }
    }
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

std::expected<void, std::string> CheckStoredTables(const artifact::Artifact& artifact,
                                                   const spec::ModelSpec& model) {
  for (const spec::Stage& stage : model.stages) {
    std::expected<void, std::string> checked;
    spec::EachComponent(stage, [&](const spec::ComponentRef& ref, const std::string&) {
      if (!checked) return;
      const components::ComponentDef* def = components::Find(ref.kind, ref.version);
      if (def == nullptr || def->check_stored == nullptr) return;
      auto params = components::Params::Of(def->params, ref);
      if (!params) {
        checked = Unexpected(std::format("stored {}: {}", ref.kind, params.error()));
        return;
      }
      components::Instance instance;
      instance.def = def;
      instance.ref = &ref;
      instance.params = std::move(*params);
      for (const spec::WeightBinding& binding : ref.weights) {
        // Another stage's tensor is its own artifact's to check.
        if (!binding.stage.empty()) continue;
        const auto role = components::RoleIndex(*def, binding.role);
        const auto index = artifact.FindResource(binding.tensor);
        if (!role || !index) {
          checked = Unexpected(std::format("stored {} {} is missing", ref.kind, binding.role));
          return;
        }
        const auto place = artifact.ResourcePlacement(*index);
        const auto& format = artifact.resources()[*index].format;
        if (!place || !format) {
          checked =
              Unexpected(std::format("stored {} {} has no placement", ref.kind, binding.role));
          return;
        }
        const auto& group = artifact.groups()[place->group];
        const auto& shard = artifact.shards()[group.shard];
        const components::StoredWeight weight{
            .role = *role,
            .format = &*format,
            .bytes = place->bytes.value(),
            .file_offset = shard.data_offset.value() + group.offset.value() + place->offset.value(),
            .read = [&](const components::StoredBytes& each) {
              return ReadPlaced(artifact, *place, each);
            }};
        if (auto stored = def->check_stored(instance, weight); !stored) {
          checked = Unexpected(stored.error());
          return;
        }
      }
    });
    if (!checked) return checked;
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

std::expected<std::uint32_t, std::string> MappingDefaultContext(
    const spec::Mapping& mapping, std::span<const spec::SourceValue> gates,
    spec::PolicyContext context) {
  std::uint32_t widest = 0;
  auto policies = spec::PolicyRules(mapping, context);
  if (!policies) return std::unexpected(policies.error());
  for (const spec::PolicyRule& rule : *policies) {
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

std::expected<std::uint32_t, std::string> ApprovedDefaultContext(const Approval& approval) {
  const spec::Mapping* mapping = spec::FindMapping(approval.architecture, approval.container);
  std::uint32_t widest = 0;
  if (mapping != nullptr && approval.policy != ApprovedPolicy::kNone) {
    std::vector<spec::SourceValue> gates;
    gates.reserve(approval.gates.size());
    for (const std::string_view gate : approval.gates)
      gates.push_back({.key = std::string(gate), .value = true});
    auto bound = MappingDefaultContext(
        *mapping, gates, {.architecture = approval.architecture, .container = approval.container});
    if (!bound) return std::unexpected(bound.error());
    widest = *bound;
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
  const auto& named = composition.components();
  if (named.size() != mapping->components.size() || components.size() != named.size())
    return Refused("the composition's components are not its image mapping's");
  const ImageApproval* candidate =
      FindImageApproval(composition.architecture(), composition.sources(), components, named);
  bool reference_match = candidate != nullptr;
  for (std::size_t i = 0; i < named.size(); ++i) {
    const auto& rule = mapping->components[i];
    const artifact::Artifact* a = components[i];
    if (a == nullptr || named[i].role != rule.role || a->id() != named[i].artifact ||
        a->model().architecture != named[i].architecture || a->format_version() != 1 ||
        !std::ranges::contains(rule.architectures, a->model().architecture) ||
        !a->expert_arrays().empty() || !OriginsWellFormed(a->sources()))
      return Refused(
          std::format("image component {} is not a valid v1 component artifact", rule.role));
    auto source = spec::SourceOf(*a);
    if (!source) return Refused(source.error());
    if (!std::ranges::contains(rule.containers, source->container))
      return Refused(
          std::format("image component {} uses an unsupported source container", rule.role));
    auto stored = spec::StoredSpec(*a);
    if (!stored) return Refused(stored.error());
    if (stored->stages.size() != 1 || stored->stages.front().artifact != 0 ||
        stored->stages.front().role != rule.stage_role)
      return Refused(
          std::format("image component {} does not own its required local stage", rule.role));
    if (auto folds = components::ValidateFoldReceipts(*stored, 0, *source); !folds)
      return std::unexpected(folds.error());
    for (const auto& tensor : source->tensors)
      if (!Admissible(tensor)) return Refused(kRefusedInventory);
    if (candidate != nullptr) {
      const Approval& want = candidate->components[i];
      if (a->model().architecture != want.architecture ||
          !SourcesMatch(want.sources, a->sources()) || !Transforms(*a, want) ||
          a->resources().size() != want.tensors || !SourceMatches(*source, want) ||
          !internal::CheckInventory(*source, want) || !Pinned(want, *stored))
        reference_match = false;
    }
    if (const auto valid = components::ValidateSpec(*stored, source->tensors); !valid)
      return Refused(valid.error());
  }
  auto family = spec::StoredSpec(composition, components);
  if (!family) return Refused(family.error());
  if (candidate != nullptr && base::ToHex(family->Digest()) != candidate->spec)
    reference_match = false;
  auto profile = spec::ImageProfileOf(*family, *mapping);
  if (!profile) return Refused(profile.error());
  // Each component's stored spec and native source inventory are the
  // authoritative tensor contract. ValidateSpec above checks that every
  // declared role has a supported format, part and shape; execution planning
  // later qualifies the actual operation/backend combination.
  return ImageAdmission{.mapping = mapping,
                        .approval = reference_match ? candidate : nullptr,
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
base::Sha256Digest ImageCompositionIdentity(const artifact::Composition& composition) {
  base::Sha256 hash;
  hash.Update("llmp.numerical.image-composition.v1\n").Update(composition.id());
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
  // The order is total: grants for one site that differ only in their
  // evidence row (overlapping evidence, numerical_evidence.cc) keep a
  // defined order, an incumbent recipe's first, so the first match
  // (planner::SelectionView) attributes the same evidence and status in
  // every build.
  std::ranges::sort(entries, {}, [](const auto& e) {
    return std::tuple(e.change, e.stage, e.component, e.anchor, e.min_rows, e.max_rows, e.max_k,
                      e.max_n, e.weight_role, !e.incumbent_recipe, e.evidence, e.arch,
                      e.exact_shape, e.exact_dimensions, e.source, e.semantic);
  });
  entries.erase(std::unique(entries.begin(), entries.end()), entries.end());
  return planner::NumericalQualifications::Bind(std::move(entries), source, semantic,
                                                std::move(payment));
}
}  // namespace

std::expected<std::shared_ptr<const planner::NumericalQualifications>, std::string>
AdmittedNumerics(const spec::ModelSpec& model, const spec::Source& source,
                 std::span<const artifact::Source> origins,
                 const base::Sha256Digest& admitted_semantic, const spec::Envelope& envelope,
                 std::uint16_t active_stage, const planner::DeviceFacts& device,
                 std::shared_ptr<const kernels::ConstructionFunder> funder) {
  // Origins bind a grant to the active source; historical approval membership
  // supplies neither permission nor a numerical default.
  if (!OriginsWellFormed(origins) || model.stages.empty() ||
      std::ranges::none_of(admitted_semantic, [](std::uint8_t byte) { return byte != 0; }))
    return Refused("numerical source facts differ from the active admitted source");
  // A single stage is the admitted spec itself, or it under the alias-wide
  // Q8_0 KV encoding (the one per-alias transform, which no numerical site
  // reads); a composition's semantic is its canonical spec's.
  if (model.stages.size() == 1 && model.Digest() != admitted_semantic &&
      components::WithoutKvCacheEncoding(model).Digest() != admitted_semantic)
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
                  if (evidence.rows.empty()) {
                    const std::int64_t param_rows =
                        !evidence.row_parameter.empty()
                            ? instance.params.Int(evidence.row_parameter)
                            : 1;
                    if (param_rows <= 0 ||
                        __builtin_mul_overflow(rows, std::uint64_t{envelope.slots}, &rows) ||
                        (evidence.stream_rows &&
                         __builtin_mul_overflow(rows, std::uint64_t{instance.stage.streams},
                                                &rows)) ||
                        (!evidence.row_parameter.empty() &&
                         __builtin_mul_overflow(rows, static_cast<std::uint64_t>(param_rows),
                                                &rows))) {
                      invalid = true;
                      continue;
                    }
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
                 std::span<const artifact::Source> origins,
                 const base::Sha256Digest& admitted_semantic, const spec::Envelope& envelope,
                 std::uint16_t active_stage, const kernels::StepBackend& backend) {
  const auto& effective = backend.owner ? backend.owner->backend() : backend;
  return AdmittedNumerics(model, source, origins, admitted_semantic, envelope, active_stage,
                          effective.device, backend.ConstructionFunding());
}

std::expected<std::shared_ptr<const planner::NumericalQualifications>, std::string>
AdmittedImageNumerics(const spec::ModelSpec& family, const artifact::Composition& composition,
                      const ir::Graph& graph, const spec::ModelSpec& stage_spec,
                      std::uint16_t stage, const planner::DeviceFacts& device,
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
  const auto identity = ImageCompositionIdentity(composition);
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
  auto rule =
      spec::DefaultRule(*mapping, gates, envelope,
                        {.architecture = approval.architecture, .container = approval.container});
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
    const auto gates = spec::PolicyGates(*mapping, model, source.container, source.architecture);
    auto rule =
        spec::DefaultRule(*mapping, gates, envelope,
                          {.architecture = source.architecture, .container = source.container});
    if (!rule) return Refused(rule.error());
    if (*rule != nullptr) return **rule;
    if (mapping->serving != nullptr && mapping->serving->serving_rule != nullptr)
      return *mapping->serving->serving_rule;
  }
  return spec::PolicyRule{};
}

}  // namespace llmp::runtime
