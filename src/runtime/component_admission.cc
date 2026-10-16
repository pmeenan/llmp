// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "runtime/component_admission.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <format>
#include <limits>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "artifact/gguf_metadata.h"
#include "base/sha256.h"
#include "components/step.h"
#include "model/dspark.h"
#include "model/dsv4.h"
#include "model/gemma2.h"
#include "model/gemma3.h"
#include "model/gemma4.h"
#include "model/gemma4_assistant.h"
#include "model/qwen38.h"
#include "platform/direct_io.h"
#include "runtime/approvals.h"
#include "runtime/gemma_profile.h"
#include "spec/mappings/dsv4.h"
#include "spec/mappings/gemma2.h"
#include "spec/mappings/gemma3.h"
#include "spec/mappings/gemma4.h"
#include "spec/mappings/qwen38.h"
#include "spec/stored.h"

namespace llmp::runtime {
namespace {

std::unexpected<std::string> Refused(std::string_view reason) {
  return std::unexpected("component artifact admission: " + std::string(reason));
}

constexpr auto kRefusedInventory =
    "native tensor layout, part, parameters or expert geometry differs";

// One tensor's native format within the approved inventory's form.
bool Admissible(const spec::SourceTensor& t, InventoryForm form) {
  const auto& f = t.format;
  if (!format::Validate(f)) return false;
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
      // Dense: BF16, or F32 (the importer's folded parameters; ModelOpt's
      // gathered tensor-scale vectors), and for ModelOpt I32 (a draft
      // head's map).
      if (f.layout == format::Layout::kDense)
        return (f.encoding == format::Encoding::kBF16 || f.encoding == format::Encoding::kF32 ||
                (modelopt && f.encoding == format::Encoding::kI32)) &&
               t.part == format::PartRole::kData && f.params == format::EncodingParams{} &&
               f.stacked == 1 && !t.experts && f.shape.rank >= 1 && f.shape.rank <= 2;
      // ModelOpt's relayouts (import/expert_relayout.h): NVFP4 routed
      // experts in scale atoms (a MoE's gate and up stacked, their tensor
      // scales gathered apart), an NVFP4 row table row-interleaved.
      if (modelopt && f.layout == format::Layout::kScaleAtoms)
        return f.encoding == format::Encoding::kNvfp4 && t.experts && f.shape.rank == 3 &&
               f.shape.ne[2] == 512 && (f.stacked == 1 || f.stacked == 2) &&
               (t.part == format::PartRole::kCodes || t.part == format::PartRole::kBlockScales) &&
               !f.params.tensor_scale && f.params.k_bits == 0;
      if (modelopt && f.layout == format::Layout::kRowInterleaved)
        return f.encoding == format::Encoding::kNvfp4 && !t.experts && f.shape.rank == 2 &&
               f.stacked == 1 &&
               (t.part == format::PartRole::kData || t.part == format::PartRole::kTensorScale) &&
               f.params.tensor_scale && f.params.k_bits == 0;
      if (f.layout != format::Layout::kSplit || f.stacked != 1) return false;
      const bool geometry =
          t.experts ? (f.shape.rank == 3 && f.shape.ne[2] == 512)
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
      return f.encoding == format::Encoding::kAffine &&
             (t.part == format::PartRole::kCodes || t.part == format::PartRole::kBlockScales ||
              t.part == format::PartRole::kBiases) &&
             f.params.bits == 4 && f.params.group == 32 &&
             f.params.scale == format::AffineScale::kBF16 && !f.params.zero_point &&
             !f.params.tensor_scale && f.params.k_bits == 0 &&
             f.params.codebook == format::Exl3Codebook::kNone && geometry;
    }
    case InventoryForm::kGguf:
      break;
  }
  return t.part == format::PartRole::kData && f.params == format::EncodingParams{} &&
         f.shape.rank >= 1 && f.shape.rank <= 3 && t.experts == (f.shape.rank == 3) &&
         f.layout == (format::Info(f.encoding).quantized ? format::Layout::kGgmlBlocks
                                                         : format::Layout::kDense);
}

// The approved inventory: every tensor admissible in its form, and the
// digest of every tensor's complete native description, sorted by name:
// its name, encoding, layout (by its enumerator), part role, stacked count,
// expert flag, every encoding parameter and its dimensions, one line each
// field.
std::expected<void, std::string> Inventory(const spec::Source& source, const Approval& approval) {
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

// The source the approval names: its container, importer folds and tensor
// count.
bool SourceMatches(const spec::Source& source, const Approval& approval) {
  return source.container == approval.container && source.converter.empty() &&
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

// The kept GGUF metadata (a single file's, or the first shard's); empty
// when none.
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

// Size anchors for the closed serving mappings' transient settings synthesis.
// No graph, tensor payload or execution owner is constructed here.
static_assert(sizeof(spec::Param) <= 128 && sizeof(spec::WeightBinding) <= 128);
static_assert(sizeof(spec::ComponentRef) <= 128 && sizeof(spec::SourceTensor) <= 256);
static_assert(sizeof(spec::V0Tensor) <= 128 && sizeof(spec::Stage) <= 2048);

template <class T>
std::expected<std::vector<T>, std::string> ReadRecipeIntegers(const artifact::Artifact& artifact,
                                                              std::uint32_t resource,
                                                              std::uint64_t count) {
  auto place = artifact.ResourcePlacement(resource);
  if (!place || place->bytes.value() != count * sizeof(T))
    return std::unexpected("integer recipe metadata has an inconsistent byte count");
  const auto& group = artifact.groups()[place->group];
  const auto& shard = artifact.shards()[group.shard];
  const auto begin = shard.data_offset.value() + group.offset.value() + place->offset.value();
  const auto end = begin + place->bytes.value();
  auto file = artifact.OpenShardForDirectRead(group.shard);
  if (!file) return std::unexpected(file.error().ToString());
  std::vector<T> out(count);
  alignas(4096) std::array<std::byte, 4096> window{};
  for (auto at = begin / window.size() * window.size(); at < end; at += window.size()) {
    if (at > shard.data_offset.value() + shard.data_bytes.value() ||
        window.size() > shard.data_offset.value() + shard.data_bytes.value() - at)
      return std::unexpected("integer recipe metadata's direct read passes its shard");
    if (auto read = platform::TransferDirectFile(file->get(), at, window, false); !read)
      return std::unexpected(
          std::format("reading integer recipe metadata: errno {}", read.error()));
    const auto first = std::max(at, begin), last = std::min(at + window.size(), end);
    std::memcpy(reinterpret_cast<std::byte*>(out.data()) + first - begin,
                window.data() + first - at, last - first);
  }
  return out;
}

template <class T>
std::expected<std::vector<T>, std::string> RecipeIntegers(const artifact::Artifact& artifact,
                                                          const model::Qwen38Tensor& tensor,
                                                          std::uint32_t maximum,
                                                          std::string_view type) {
  const bool vector = type == "I64" && tensor.plain && tensor.ne.size() == 1;
  const bool table = type == "I32" && !tensor.plain && tensor.ne.size() == 2 && tensor.ne[0] == 1;
  if (tensor.type != type || (!vector && !table))
    return std::unexpected("integer recipe metadata has an unsupported type or shape");
  const auto count = tensor.ne[table ? 1 : 0];
  if (count == 0 || count > maximum)
    return std::unexpected("integer recipe metadata exceeds its bound");
  return ReadRecipeIntegers<T>(artifact, tensor.index, count);
}

template std::expected<std::vector<std::int32_t>, std::string> ReadRecipeIntegers<std::int32_t>(
    const artifact::Artifact& artifact, std::uint32_t resource, std::uint64_t count);
template std::expected<std::vector<std::int64_t>, std::string> ReadRecipeIntegers<std::int64_t>(
    const artifact::Artifact& artifact, std::uint32_t resource, std::uint64_t count);
template std::expected<std::vector<std::int32_t>, std::string> RecipeIntegers<std::int32_t>(
    const artifact::Artifact& artifact, const model::Qwen38Tensor& tensor, std::uint32_t maximum,
    std::string_view type);
template std::expected<std::vector<std::int64_t>, std::string> RecipeIntegers<std::int64_t>(
    const artifact::Artifact& artifact, const model::Qwen38Tensor& tensor, std::uint32_t maximum,
    std::string_view type);

std::expected<model::Qwen38PleHash, std::string> RecipeHash(const artifact::Artifact& artifact,
                                                            const model::Qwen38Binding& binding) {
  const auto& p = model::Qwen38Flash();
  if (binding.gguf()) return model::ReadQwen38GgufHash(p, artifact, binding.ple_table.ne.at(1));
  const auto& layer = binding.layers.at(p.ple_layer);
  auto multipliers = RecipeIntegers<std::int64_t>(artifact, layer.ple_multipliers, p.ngram, "I64");
  auto offsets =
      RecipeIntegers<std::int64_t>(artifact, layer.ple_head_offsets, p.ple_heads(), "I64");
  auto vocab = RecipeIntegers<std::int64_t>(artifact, layer.ple_head_vocab, p.ple_heads(), "I64");
  if (!multipliers) return std::unexpected(multipliers.error());
  if (!offsets) return std::unexpected(offsets.error());
  if (!vocab) return std::unexpected(vocab.error());
  return model::CheckQwen38PleHash(p, *multipliers, *offsets, *vocab, binding.ple_table.ne.at(1));
}

bool OnComponentEngine(std::string_view architecture) {
  const spec::Mapping* mapping = spec::FindMapping(architecture);
  return mapping != nullptr && (mapping->serving != nullptr || mapping->draft != nullptr);
}

const spec::Mapping* ComposedDrafterMapping(std::string_view architecture) {
  const spec::Mapping* mapping = spec::FindMapping(architecture);
  if (mapping == nullptr || mapping->draft == nullptr || mapping->draft->composed == nullptr)
    return mapping;
  return &mapping->draft->composed();
}

const spec::Mapping* ComposedDrafterMapping(const artifact::Artifact& drafter) {
  const Approval* approval = drafter.format_version() == 1
                                 ? FindApproval(drafter.model().architecture, drafter.sources())
                                 : nullptr;
  if (approval == nullptr || approval->mapping == nullptr)
    return ComposedDrafterMapping(drafter.model().architecture);
  const spec::Mapping* mapping = &approval->mapping();
  if (mapping->draft == nullptr || mapping->draft->composed == nullptr) return mapping;
  return &mapping->draft->composed();
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
  if (architecture == "qwen4exp") {
    auto binding = model::BindQwen38(model::Qwen38Flash(), artifact);
    if (!binding) return std::unexpected(binding.error());
    auto hash = RecipeHash(artifact, *binding);
    if (!hash) return std::unexpected(hash.error());
    return spec::mappings::Qwen38V0Values(model::Qwen38Flash(), *binding, *hash);
  }
  if (architecture == "deepseek4") {
    auto binding = model::BindDsv4(model::Dsv4Flash(), artifact);
    if (!binding) return std::unexpected(binding.error());
    return spec::mappings::Dsv4V0Values(model::Dsv4Flash());
  }
  return std::unexpected(
      std::format("the component engine has no recipe for architecture {} yet", architecture));
}

std::expected<void, std::string> CheckComponentV0Admission(const artifact::Artifact& artifact) {
  const auto& architecture = artifact.model().architecture;
  if (architecture == "gemma2") {
    if (auto binding = model::BindApprovedGemma2(artifact); !binding)
      return std::unexpected(binding.error());
  } else if (architecture == "gemma3") {
    if (auto binding = model::BindApprovedGemma3(artifact); !binding)
      return std::unexpected(binding.error());
  } else if (architecture == "gemma4") {
    if (auto profile = ApprovedGemmaProfile(artifact); !profile)
      return std::unexpected(profile.error());
  }
  return {};
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
  if (drafter.model().architecture == "dflash" && target.model().architecture == "deepseek4") {
    const auto& profile = model::Dsv4Flash();
    auto target_binding = model::BindDsv4(profile, target);
    if (!target_binding) return std::unexpected(target_binding.error());
    const auto& companion = model::DsparkDeepSeekV4Flash();
    if (auto bound = model::BindDspark(companion, drafter, profile, *target_binding); !bound) {
      return std::unexpected(bound.error());
    }
    return spec::mappings::DsparkV0Values(companion);
  }
  if (drafter.model().architecture == "qwen4exp-mtp" && target.model().architecture == "qwen4exp") {
    auto binding = model::BindQwen38Mtp(model::Qwen38Flash(), drafter);
    if (!binding) return std::unexpected(binding.error());
    return spec::mappings::Qwen38MtpV0Values(model::Qwen38Flash(), *binding);
  }
  return std::unexpected(std::format("the component engine has no drafter {} for architecture {}",
                                     drafter.model().architecture, target.model().architecture));
}

std::expected<std::vector<spec::SourceValue>, std::string> ComponentDrafterValues(
    const artifact::Artifact& target, const artifact::Artifact& drafter) {
  if (target.format_version() == 0 && drafter.format_version() == 0)
    return ComponentDrafterV0Values(target, drafter);
  // Either is stored: the approved pairing, by both sources' identities.
  const Approval* t = FindApproval(target.model().architecture, target.sources());
  if (t == nullptr || t->drafter)
    return Refused("target source identity has no approved v1 execution profile");
  const Approval* d = FindApproval(drafter.model().architecture, drafter.sources());
  if (d == nullptr || !d->drafter)
    return Refused("drafter source identity has no approved v1 execution profile");
  if (d->target != t->name)
    return Refused(
        std::format("the approved drafter {} is {}'s, not {}'s", d->name, d->target, t->name));
  if (drafter.format_version() != 1) return ComponentDrafterV0Values(target, drafter);
  const spec::Mapping* mapping = ComposedDrafterMapping(drafter);
  if (mapping == nullptr || mapping->role != spec::StageRole::kDrafter)
    return Refused("the drafter's architecture has no drafter mapping");
  // A drafter without kept metadata values: its recipe's, from its source.
  if (mapping->draft != nullptr && mapping->draft->values != nullptr) {
    auto source = spec::SourceOf(drafter, mapping);
    if (!source) return Refused(source.error());
    auto values = mapping->draft->values(*source);
    if (!values) return Refused(values.error());
    return values;
  }
  return KeptValues(drafter, *mapping);
}

std::expected<void, std::string> ComponentDrafterEnvelope(const artifact::Artifact& drafter,
                                                          const spec::Envelope& envelope) {
  const Approval* d = FindApproval(drafter.model().architecture, drafter.sources());
  if (d == nullptr) {
    if (drafter.format_version() == 1)
      return Refused("drafter source identity has no approved v1 execution profile");
    return {};
  }
  if (envelope.context == 0 || envelope.context > d->max_context || envelope.slots == 0 ||
      envelope.slots > d->max_slots)
    return Refused(std::format("execution envelope exceeds the approved drafter {}", d->name));
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
  else if (!resource.format && resource.repr.type == "I32" && resource.repr.dims.size() == 2 &&
           resource.repr.dims[0] == 1)
    rows = resource.repr.dims[1];
  if (rows == 0 || rows > vocab)
    return Refused("a selected draft head's token map is not I32 [1, rows] within the vocabulary");
  return DraftHeadMap{.resource = *index, .rows = rows};
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
                                                        const artifact::Artifact& drafter) {
  if (drafter.format_version() != 1) return {};
  const Approval* d = FindApproval(drafter.model().architecture, drafter.sources());
  if (d == nullptr || !d->drafter)
    return Refused("drafter source identity has no approved v1 execution profile");
  return Pinned(*d, composed);
}

std::expected<void, std::string> CheckComponentDrafterAdmission(
    const spec::Source& source, std::span<const artifact::Source> origins) {
  const Approval* approval = FindApproval(source.architecture, origins);
  if (approval == nullptr) return Refused("source identity has no approved v1 execution profile");
  if (!approval->drafter) return Refused("not an approved drafter execution profile");
  if (!SourceMatches(source, *approval))
    return Refused("source container, transforms or tensor inventory differs from its profile");
  return Inventory(source, *approval);
}

std::expected<void, std::string> CheckComponentAdmission(
    const spec::ModelSpec& model, const spec::Source& source,
    std::span<const artifact::Source> origins) {
  const Approval* approval = FindApproval(source.architecture, origins);
  if (approval == nullptr || approval->drafter)
    return Refused("source identity has no approved v1 execution profile");
  if (!SourceMatches(source, *approval))
    return Refused("source container, transforms or tensor inventory differs from its profile");
  if (model.pipeline != spec::PipelineKind::kLlm || model.stages.size() != 1 ||
      model.stages[0].role != spec::StageRole::kTarget || model.stages[0].artifact != 0)
    return Refused("the stored pipeline is not an approved standalone target");
  if (auto inventory = Inventory(source, *approval); !inventory) return inventory;
  const auto& stage = model.stages.front();
  if (approval->tied_output &&
      (!stage.embedding || !stage.heads.output || stage.embedding->weights.size() != 1 ||
       stage.heads.output->weights.size() != 1 ||
       stage.embedding->weights.front().tensor != stage.heads.output->weights.front().tensor))
    return Refused("the approved target requires its output tied to its embedding");
  // Execute the authenticated stored spec unchanged; never replace a
  // disagreeing spec by an expected one.
  if (auto pinned = Pinned(*approval, model); !pinned) return pinned;
  const std::array<std::span<const spec::SourceTensor>, 1> tables{source.tensors};
  if (const auto valid = components::ValidateSpec(model, tables); !valid)
    return Refused(valid.error());
  return {};
}

std::expected<void, std::string> CheckComponentArtifactAdmission(
    const artifact::Artifact& artifact) {
  if (artifact.format_version() != 1) return Refused("requires a native v1 artifact");
  const Approval* approval = FindApproval(artifact.model().architecture, artifact.sources());
  if (approval == nullptr) return Refused("source identity has no approved v1 execution profile");
  if (!Transforms(artifact, *approval) ||
      artifact.resources().size() != approval->tensors - approval->arrays ||
      artifact.expert_arrays().size() != approval->arrays)
    return Refused("the approved artifact requires its native tensor inventory and transforms");
  return CheckScaleAtomSlots(artifact);
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

std::expected<ComponentAdmission, std::string> ReadComponentAdmission(
    const artifact::Artifact& artifact) {
  if (auto checked = CheckComponentArtifactAdmission(artifact); !checked)
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

std::expected<spec::PolicyRule, std::string> ComponentAdmissionPolicy(
    const spec::ModelSpec& model, const spec::Source& source,
    std::span<const artifact::Source> origins, const spec::Envelope& envelope) {
  if (auto checked = CheckComponentAdmission(model, source, origins); !checked)
    return std::unexpected(checked.error());
  const Approval* approval = FindApproval(source.architecture, origins);
  if (envelope.context == 0 || envelope.context > approval->max_context || envelope.slots == 0 ||
      envelope.slots > approval->max_slots)
    return Refused("execution envelope exceeds the approved component profile");
  if (approval->wide_context != 0 && envelope.context > approval->wide_context &&
      envelope.slots != 1)
    return Refused(std::format("{} wider context requires exactly one owner", approval->label));
  auto rule = ApprovedRule(*approval, envelope);
  if (!rule) return std::unexpected(rule.error());
  if (*rule != nullptr) return **rule;
  if (approval->policy == ApprovedPolicy::kMapping)
    return Refused("no approved policy for this execution envelope");
  return spec::PolicyRule{};
}

}  // namespace llmp::runtime
