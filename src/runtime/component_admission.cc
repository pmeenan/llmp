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
#include <utility>
#include <vector>

#include "artifact/gguf_metadata.h"
#include "base/sha256.h"
#include "components/step.h"
#include "platform/direct_io.h"
#include "runtime/approvals.h"
#include "runtime/component_admission_internal.h"
#include "spec/image_family.h"
#include "spec/mappings/qwen2.h"
#include "spec/mappings/qwen38.h"
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

// One tensor's native format within the approved inventory's form.
bool Admissible(const spec::SourceTensor& t, InventoryForm form) {
  const auto& f = t.format;
  const auto parts = format::PartsOf(f);
  if (!parts || !parts->Find(t.part)) return false;
  if (f.stacked != 1 && form != InventoryForm::kModelOpt) return false;
  switch (form) {
    case InventoryForm::kExl3: {
      const bool trellis = f.encoding == format::Encoding::kExl3Trellis;
      const auto bits = f.params.k_bits;
      if (t.experts) return false;
      if (trellis)
        return f.layout == format::Layout::kExl3Tiles && f.shape.rank == 2 &&
               t.part != format::PartRole::kData &&
               f.params.codebook == format::Exl3Codebook::kMcg && !f.params.tensor_scale &&
               (bits == 4 || bits == 5 || bits == 6 || bits == 8);
      return (f.encoding == format::Encoding::kBF16 || f.encoding == format::Encoding::kF16) &&
             f.layout == format::Layout::kDense && t.part == format::PartRole::kData &&
             f.params == format::EncodingParams{} && f.shape.rank >= 1 && f.shape.rank <= 2;
    }
    case InventoryForm::kModelOpt:
    case InventoryForm::kMlx: {
      const bool modelopt = form == InventoryForm::kModelOpt;
      // Dense: F16/BF16, or F32 (the importer's folded parameters; ModelOpt's
      // gathered tensor-scale vectors), and for ModelOpt I32 (a draft
      // head's map).
      if (f.layout == format::Layout::kDense)
        return (f.encoding == format::Encoding::kF16 || f.encoding == format::Encoding::kBF16 ||
                f.encoding == format::Encoding::kF32 ||
                (modelopt && f.encoding == format::Encoding::kI32)) &&
               t.part == format::PartRole::kData && f.params == format::EncodingParams{} &&
               f.stacked == 1 && !t.experts && f.shape.rank >= 1 && f.shape.rank <= 2;
      // ModelOpt's relayouts (import/expert_relayout.h): NVFP4 routed
      // experts in scale atoms (a MoE's gate and up stacked, their tensor
      // scales gathered apart), an NVFP4 row table row-interleaved.
      if (modelopt && f.layout == format::Layout::kScaleAtoms)
        return f.encoding == format::Encoding::kNvfp4 && t.experts && f.shape.rank == 3 &&
               (f.stacked == 1 || f.stacked == 2) &&
               (t.part == format::PartRole::kCodes || t.part == format::PartRole::kBlockScales) &&
               !f.params.tensor_scale && f.params.k_bits == 0;
      if (modelopt && f.layout == format::Layout::kRowInterleaved)
        return f.encoding == format::Encoding::kNvfp4 && !t.experts && f.shape.rank == 2 &&
               f.stacked == 1 &&
               (t.part == format::PartRole::kData || t.part == format::PartRole::kTensorScale) &&
               f.params.tensor_scale && f.params.k_bits == 0;
      if (f.layout != format::Layout::kSplit || f.stacked != 1) return false;
      const bool geometry =
          t.experts ? f.shape.rank == 3
                    : (form == InventoryForm::kModelOpt ? f.shape.rank == 2
                                                        : (f.shape.rank >= 1 && f.shape.rank <= 2));
      if (form == InventoryForm::kModelOpt) {
        const bool nv = f.encoding == format::Encoding::kNvfp4;
        const bool mx = f.encoding == format::Encoding::kMxfp8;
        return (nv || mx) &&
               (t.part == format::PartRole::kCodes || t.part == format::PartRole::kBlockScales ||
                t.part == format::PartRole::kTensorScale) &&
               f.params.tensor_scale == nv && f.params.k_bits == 0 &&
               f.params.codebook == format::Exl3Codebook::kNone && geometry;
      }
      // Affine widths, groups and scale types are the format layer's
      // structural rules, not the qualified checkpoint's constants (or a
      // decoder's capability list). The pinned inventory qualifies the
      // exact parameters; implementation selection checks capability.
      return f.encoding == format::Encoding::kAffine &&
             (t.part == format::PartRole::kCodes || t.part == format::PartRole::kBlockScales ||
              t.part == format::PartRole::kBiases) &&
             !f.params.zero_point && geometry;
    }
    case InventoryForm::kDense:
      return (f.encoding == format::Encoding::kBF16 || f.encoding == format::Encoding::kF32) &&
             f.layout == format::Layout::kDense && t.part == format::PartRole::kData &&
             f.params == format::EncodingParams{} && !t.experts && f.shape.rank >= 1 &&
             f.shape.rank <= 4;
    case InventoryForm::kGguf:
      break;
  }
  return t.part == format::PartRole::kData && f.params == format::EncodingParams{} &&
         f.shape.rank >= 1 && f.shape.rank <= 3 && t.experts == (f.shape.rank == 3) &&
         f.layout == (format::Info(f.encoding).quantized ? format::Layout::kGgmlBlocks
                                                         : format::Layout::kDense);
}

}  // namespace

// The approved inventory: every tensor admissible in its form, and the
// digest of every tensor's complete native description, sorted by name:
// its name, encoding, layout (by its enumerator), part role, stacked count,
// expert flag, every encoding parameter and its dimensions, one line each
// field.
std::expected<void, std::string> internal::CheckInventory(const spec::Source& source,
                                                          const Approval& approval) {
  std::vector<const spec::SourceTensor*> sorted;
  sorted.reserve(source.tensors.size());
  for (const auto& t : source.tensors) {
    if (!Admissible(t, approval.form)) return Refused(kRefusedInventory);
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
  const auto keys = spec::SourceKeys(mapping);
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
        if (value.integers.size() > spec::kMaxList) continue;
        v = value.integers;
        break;
      case artifact::GgufValue::Kind::kFloats:
        if (value.reals.size() > spec::kMaxList) continue;
        v = value.reals;
        break;
      case artifact::GgufValue::Kind::kOther:
        continue;
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
  return mapping != nullptr && (mapping->serving != nullptr || mapping->draft != nullptr);
}

namespace {

const spec::Mapping* TargetMappingFor(std::string_view architecture, std::string_view container) {
  if (architecture == "qwen4exp") {
    return container == "gguf" ? &spec::mappings::Qwen38GgufMapping()
                               : &spec::mappings::Qwen38NativeMapping();
  }
  if (architecture == "qwen2") {
    if (container == "exl3") return &spec::mappings::Qwen2Exl3Mapping();
    if (container == "safetensors") return &spec::mappings::Qwen2SafetensorsMapping();
    if (container == "gguf") return &spec::mappings::Qwen2GgufMapping();
  }
  return spec::FindMapping(architecture);
}

const spec::Mapping* DrafterMappingFor(std::string_view architecture, std::string_view container) {
  if (architecture == "qwen4exp-mtp" && !container.empty() && container != "gguf") {
    return &spec::mappings::Qwen38MtpNativeMapping();
  }
  return spec::FindMapping(architecture);
}

std::vector<spec::SourceValue> UnpinnedGates(const spec::ModelSpec& model,
                                             const spec::Source& source) {
  std::vector<spec::SourceValue> gates;
  if (model.stages.empty()) return gates;
  bool has_moe = false;
  bool moe_stacked = false;
  bool moe_scaled = false;
  for (const spec::Layer& layer : model.stages.front().layers) {
    for (const spec::ComponentRef& comp : layer.components) {
      if (comp.kind != "ffn.moe") continue;
      has_moe = true;
      if (const spec::Param* p = comp.Find("stacked")) {
        if (const auto* b = std::get_if<bool>(&p->value); b != nullptr && *b) moe_stacked = true;
      }
      if (const spec::Param* p = comp.Find("projection_scales")) {
        if (const auto* b = std::get_if<bool>(&p->value); b != nullptr && *b) moe_scaled = true;
      }
    }
  }
  if (has_moe) {
    gates.push_back({.key = std::format("{}.expert_count", source.architecture), .value = true});
  }
  if (source.architecture == "qwen4exp") {
    if (moe_stacked) {
      gates.push_back({.key = "llmp.qwen4exp.experts_stacked", .value = true});
      if (source.container == "safetensors") {
        gates.push_back({.key = "llmp.qwen4exp.expert_activation_scales", .value = true});
      }
    }
    if (moe_scaled) {
      gates.push_back({.key = "llmp.qwen4exp.expert_scales", .value = true});
    }
  }
  return gates;
}

}  // namespace

const spec::Mapping* ComposedDrafterMapping(std::string_view architecture) {
  const spec::Mapping* mapping = spec::FindMapping(architecture);
  if (mapping == nullptr || mapping->draft == nullptr || mapping->draft->composed == nullptr)
    return mapping;
  return &mapping->draft->composed();
}

const spec::Mapping* ComposedDrafterMapping(const artifact::Artifact& drafter) {
  const Approval* approval = FindApproval(drafter.model().architecture, drafter.sources());
  if (approval != nullptr && approval->mapping != nullptr) {
    const spec::Mapping* mapping = &approval->mapping();
    if (mapping->draft == nullptr || mapping->draft->composed == nullptr) return mapping;
    return &mapping->draft->composed();
  }
  if (auto source = spec::SourceOf(drafter)) {
    const spec::Mapping* mapping = DrafterMappingFor(source->architecture, source->container);
    if (mapping == nullptr || mapping->draft == nullptr || mapping->draft->composed == nullptr)
      return mapping;
    return &mapping->draft->composed();
  }
  return ComposedDrafterMapping(drafter.model().architecture);
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

std::optional<InventoryForm> ContainerForm(std::string_view container) {
  if (container == "gguf") return InventoryForm::kGguf;
  if (container == "exl3") return InventoryForm::kExl3;
  if (container == "mlx") return InventoryForm::kMlx;
  if (container == "safetensors") return InventoryForm::kModelOpt;
  return std::nullopt;
}

bool FoldDeclared(const spec::Mapping& mapping, std::string_view container, std::string_view fold) {
  const auto matches = [&](const spec::Mapping& m) {
    return std::ranges::any_of(m.folds, [&](const auto& f) {
      return f.fold == fold && (f.container.empty() || f.container == container);
    });
  };
  if (matches(mapping)) return true;
  if (const spec::Mapping* by_container = TargetMappingFor(mapping.architecture, container);
      by_container != nullptr && matches(*by_container)) {
    return true;
  }
  if ((mapping.architecture == "qwen4exp" || mapping.architecture == "qwen4exp-mtp") &&
      (container == "safetensors" || container == "mlx")) {
    return fold == "gdn.value_heads_tiled" || fold == "norm.plus_one" || fold == "ssm.a_neg_exp";
  }
  return false;
}

std::expected<void, std::string> CheckUnpinnedInventory(const spec::Source& source,
                                                        const spec::Mapping& mapping) {
  const auto form = ContainerForm(source.container);
  if (!form || source.tensors.empty()) return Refused(kRefusedInventory);
  for (const std::string& fold : source.folds) {
    if (!FoldDeclared(mapping, source.container, fold)) {
      return Refused("source fold is not declared by the architecture's mapping");
    }
  }
  for (const auto& t : source.tensors) {
    if (!Admissible(t, *form)) return Refused(kRefusedInventory);
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
  if (t != nullptr && d != nullptr) {
    if (t->drafter) return Refused("target source identity has no approved v1 execution profile");
    if (!d->drafter) return Refused("drafter source identity has no approved v1 execution profile");
    if (d->target != t->name)
      return Refused(
          std::format("the approved drafter {} is {}'s, not {}'s", d->name, d->target, t->name));
  } else {
    const spec::Mapping* target_mapping = spec::FindMapping(target.model().architecture);
    if (target_mapping == nullptr || target_mapping->role != spec::StageRole::kTarget)
      return Refused("target architecture has no target component mapping");
    const bool arch_ok =
        target.model().architecture == drafter.model().architecture ||
        std::ranges::any_of(mapping->draft->import_targets, [&](const auto& it) {
          return it.mapping != nullptr && it.mapping().architecture == target.model().architecture;
        });
    if (!arch_ok)
      return Refused(std::format("drafter {} does not target architecture {}",
                                 drafter.model().architecture, target.model().architecture));
  }
  // A drafter without kept metadata values: its recipe's, from its source.
  if (mapping->draft->values != nullptr) {
    auto source = spec::SourceOf(drafter);
    if (!source) return Refused(source.error());
    auto values = mapping->draft->values(*source);
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
  const Approval* d = FindApproval(drafter.model().architecture, drafter.sources());
  if (d == nullptr) return {};
  if (!d->drafter) return Refused("drafter source identity has no approved v1 execution profile");
  // A pinned drafter's own identity is its inventory (its source receipts,
  // container, folds and native tensors: CheckComponentDrafterAdmission).
  // Its pinned composition is its approved pairing's: beside its approved
  // target the composition is that one exactly; beside another target
  // (unpinned, of an architecture it drafts for: ComponentDrafterValues) it
  // is validated structurally, as an unpinned drafter's is.
  const Approval* t = FindApproval(target.model().architecture, target.sources());
  if (t == nullptr || t->name != d->target) return {};
  return Pinned(*d, composed);
}

std::expected<void, std::string> CheckComponentDrafterAdmission(
    const spec::Source& source, std::span<const artifact::Source> origins) {
  if (!OriginsWellFormed(origins)) return Refused("source origins are empty or duplicate");
  const Approval* approval = FindApproval(source.architecture, origins);
  if (approval != nullptr) {
    if (!approval->drafter) return Refused("not an approved drafter execution profile");
    if (!SourceMatches(source, *approval))
      return Refused("source container, transforms or tensor inventory differs from its profile");
    return internal::CheckInventory(source, *approval);
  }
  const spec::Mapping* mapping = ComposedDrafterMapping(source.architecture);
  if (mapping == nullptr || mapping->role != spec::StageRole::kDrafter || mapping->draft == nullptr)
    return Refused("architecture has no drafter component mapping");
  return CheckUnpinnedInventory(source, *mapping);
}

std::expected<void, std::string> CheckComponentAdmission(
    const spec::ModelSpec& model, const spec::Source& source,
    std::span<const artifact::Source> origins) {
  if (!OriginsWellFormed(origins)) return Refused("source origins are empty or duplicate");
  if (model.pipeline != spec::PipelineKind::kLlm || model.stages.size() != 1 ||
      model.stages[0].role != spec::StageRole::kTarget || model.stages[0].artifact != 0)
    return Refused("the stored pipeline is not an approved standalone target");
  const Approval* approval = FindApproval(source.architecture, origins);
  if (approval != nullptr) {
    if (approval->drafter) return Refused("source identity has no approved v1 execution profile");
    if (!SourceMatches(source, *approval))
      return Refused("source container, transforms or tensor inventory differs from its profile");
    if (auto inventory = internal::CheckInventory(source, *approval); !inventory) return inventory;
    const auto& stage = model.stages.front();
    if (approval->tied_output &&
        (!stage.embedding || !stage.heads.output || stage.embedding->weights.size() != 1 ||
         stage.heads.output->weights.size() != 1 ||
         stage.embedding->weights.front().tensor != stage.heads.output->weights.front().tensor))
      return Refused("the approved target requires its output tied to its embedding");
    // Execute the authenticated stored spec unchanged; never replace a
    // disagreeing spec by an expected one.
    if (auto pinned = Pinned(*approval, model); !pinned) return pinned;
  } else {
    if (std::ranges::any_of(Approvals(),
                            [&](const Approval& a) { return SourcesMatch(a.sources, origins); })) {
      return Refused("pinned source origins disagree with the declared architecture");
    }
    const spec::Mapping* mapping = TargetMappingFor(source.architecture, source.container);
    if (mapping == nullptr || mapping->role != spec::StageRole::kTarget)
      return Refused("architecture has no target component mapping");
    if (auto inv = CheckUnpinnedInventory(source, *mapping); !inv) return inv;
    for (const spec::TensorRule& tr : mapping->tensors) {
      if (tr.optional || tr.target || tr.unbound || !tr.layer_types.empty() ||
          !tr.fallback.empty()) {
        continue;
      }
      const auto slot_it = std::ranges::find(mapping->slots, tr.slot, &spec::SlotRule::slot);
      if (slot_it == mapping->slots.end() || !slot_it->layer_types.empty() ||
          !slot_it->gate.empty() || !slot_it->layers_key.empty() || !slot_it->count_key.empty()) {
        continue;
      }
      std::string name(tr.pattern);
      if (const auto pos = name.find(spec::kLayerField); pos != std::string::npos) {
        name.replace(pos, spec::kLayerField.size(), "0");
      }
      if (!std::ranges::any_of(source.tensors,
                               [&](const spec::SourceTensor& st) { return st.name == name; })) {
        return Refused(std::format("source is missing required mapping tensor {}", name));
      }
    }
    std::vector<std::string> expected_folds = source.folds;
    for (const spec::FoldRule& f : mapping->folds) {
      if (f.container == source.container && !std::ranges::contains(expected_folds, f.fold)) {
        expected_folds.emplace_back(f.fold);
      }
    }
    std::ranges::sort(expected_folds);
    if (model.stages.front().folds != expected_folds)
      return Refused("stored stage folds disagree with source folds");
  }
  const std::array<std::span<const spec::SourceTensor>, 1> tables{source.tensors};
  if (const auto valid = components::ValidateSpec(model, tables); !valid)
    return Refused(valid.error());
  return {};
}

std::expected<void, std::string> CheckArtifactPrerequisites(const artifact::Artifact& artifact) {
  if (auto served = RequireServedVersion(artifact); !served) return Refused(served.error());
  if (!OriginsWellFormed(artifact.sources()))
    return Refused("artifact source receipts are empty or duplicate");
  const Approval* approval = FindApproval(artifact.model().architecture, artifact.sources());
  if (approval != nullptr) {
    if (!Transforms(artifact, *approval) ||
        artifact.resources().size() != approval->tensors - approval->arrays ||
        artifact.expert_arrays().size() != approval->arrays)
      return Refused("the approved artifact requires its native tensor inventory and transforms");
  } else if (spec::FindMapping(artifact.model().architecture) == nullptr) {
    return Refused("architecture has no component mapping");
  }
  return CheckScaleAtomSlots(artifact);
}

std::expected<void, std::string> CheckComponentArtifactAdmission(
    const artifact::Artifact& artifact) {
  const spec::Mapping* mapping = spec::FindMapping(artifact.model().architecture);
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
  auto source = spec::SourceOf(drafter);
  if (!source) return Refused(source.error());
  if (auto checked = CheckComponentDrafterAdmission(*source, drafter.sources()); !checked)
    return std::unexpected(checked.error());
  return ComponentDrafterAdmission{std::move(*source),
                                   FindApproval(drafter.model().architecture, drafter.sources())};
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

std::uint32_t ApprovedDefaultContext(const Approval& approval) {
  const spec::Mapping* mapping =
      approval.mapping != nullptr ? &approval.mapping() : spec::FindMapping(approval.architecture);
  std::uint32_t widest = 0;
  if (mapping != nullptr && approval.policy != ApprovedPolicy::kNone) {
    for (const spec::PolicyRule& rule : mapping->policies) {
      const bool gated = std::ranges::contains(approval.gates, rule.gate);
      if (!rule.gate.empty() && gated == rule.gate_absent) continue;
      if (rule.max_context != 0 && !rule.owners_alone) widest = std::max(widest, rule.max_context);
    }
  }
  return widest != 0 ? std::min(widest, approval.max_context) : approval.max_context;
}

std::expected<ImageAdmission, std::string> ReadImageAdmission(
    const artifact::Composition& composition,
    std::span<const artifact::Artifact* const> components) {
  if (composition.format_version() != 1)
    return Refused(std::format(
        "composition {} ({}) is format version {}; serving reads version 1 only (D-118): "
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
        if (!Admissible(t, InventoryForm::kDense)) return Refused(kRefusedInventory);
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
  return ImageAdmission{.mapping = mapping,
                        .approval = approval,
                        .family = std::move(*family),
                        .profile = std::move(*profile)};
}

std::expected<const spec::PolicyRule*, std::string> ApprovedRule(const Approval& approval,
                                                                 const spec::Envelope& envelope) {
  if (approval.policy == ApprovedPolicy::kNone) return nullptr;
  const spec::Mapping* mapping =
      approval.mapping != nullptr ? &approval.mapping() : spec::FindMapping(approval.architecture);
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
  if (envelope.context == 0 || envelope.context > model.stages.front().context ||
      envelope.slots == 0)
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
  const spec::Mapping* mapping = TargetMappingFor(source.architecture, source.container);
  if (mapping != nullptr) {
    const auto gates = UnpinnedGates(model, source);
    auto rule = spec::DefaultRule(*mapping, gates, envelope);
    if (rule && *rule != nullptr) return **rule;
  }
  return spec::PolicyRule{};
}

}  // namespace llmp::runtime
