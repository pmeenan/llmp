// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "runtime/memory_guard.h"

#include <algorithm>
#include <cstdint>
#include <expected>
#include <format>
#include <limits>
#include <string>
#include <utility>

#include "artifact/artifact.h"
#include "artifact/json.h"
#include "base/check.h"
#include "base/gguf.h"
#include "base/json.h"
#include "runtime/calibration.h"
#include "runtime/component_bindings.h"
#include "runtime/model_settings.h"
#include "tokenizer/gguf.h"
#include "tokenizer/tokenizer.h"

namespace llmp::runtime {

std::uint64_t ActiveFootprint(std::uint64_t weights, std::uint64_t activations,
                              std::uint64_t pool) {
  constexpr std::uint64_t extent = std::uint64_t{2} << 20U;
  constexpr std::uint64_t most = std::numeric_limits<std::uint64_t>::max();
  const auto region = [](std::uint64_t bytes) {
    return bytes > most - extent
               ? most / extent * extent
               : (std::max(bytes, std::uint64_t{1}) + extent - 1) / extent * extent;
  };
  std::uint64_t total = weights;
  for (const std::uint64_t part : {region(activations), region(pool)})
    if (__builtin_add_overflow(total, part, &total)) return most;
  return total;
}

std::uint64_t GuardReserve(const MemoryGuard& guard) {
  return guard.host_inputs + guard.plans + guard.requests + kUncountedMargin;
}

std::expected<void, std::string> CheckMemoryGuard(const MemoryGuard& guard) {
  const auto gib = [](std::uint64_t bytes) { return static_cast<double>(bytes) / (1ULL << 30U); };
  // Subtracted term by term, so no sum can wrap.
  const std::uint64_t a = guard.available;
  const bool fits =
      guard.largest <= a && guard.host_inputs <= a - guard.largest &&
      guard.plans <= a - guard.largest - guard.host_inputs &&
      kUncountedMargin <= a - guard.largest - guard.host_inputs - guard.plans &&
      guard.requests <= a - guard.largest - guard.host_inputs - guard.plans - kUncountedMargin;
  if (a != 0 && !fits) {
    return std::unexpected(std::format(
        "the largest model's weights and workspace ({:.1f} GiB), a model's largest host-built "
        "chunk inputs "
        "({:.1f} GiB), a model step's plans ({:.1f} GiB), the request memory ({:.1f} GiB, "
        "[client] request_memory_bytes) and a {:.0f} GiB margin do not fit the {:.1f} GiB "
        "available beside this node's fixed memory ({:.1f} GiB)",
        gib(guard.largest), gib(guard.host_inputs), gib(guard.plans), gib(guard.requests),
        gib(kUncountedMargin), gib(guard.available), gib(guard.fixed)));
  }
  return {};
}

std::expected<std::uint64_t, std::string> StartupBootstrapBytes(std::uint64_t aliases,
                                                                std::uint64_t configuration_bytes) {
  namespace a = artifact;
  // Canonical parser: old+new storage for nodes(24), kids(4), stack(4),
  // duplicate-key scratch(16). An extra node covers a rejected last value.
  constexpr std::uint64_t values = a::json::kMaxSeparators + a::json::kMaxContainers + 2;
  constexpr std::uint64_t canonical = 3 * values * (24 + 4 + 4 + 16);
  // Both manifest/index trees, plus transient spec parse. Text doubling,
  // index expansions/roles/maps, directory names and one rebuilt shard header.
  // The entry term includes the complete rejected-input growth inventory;
  // strings copied into it are separately bounded by the document bytes.
  constexpr std::uint64_t documents = a::kMaxManifestBytes + a::kMaxIndexBytes + a::kMaxSpecBytes;
  constexpr std::uint64_t artifact_open = 3 * canonical + 8 * documents + 4096 * a::kMaxEntries +
                                          3 * (a::kMaxShardHeader + a::kFileAlignment) +
                                          1536 * a::kMaxDirectoryEntries;
  constexpr base::json::Limits json;
  // General JSON node/kid/pending (20/value), duplicate-key string_view
  // scratch (16/value), copied text, input and bounded metadata source views.
  constexpr std::uint64_t metadata =
      36 * json.max_values + 4 * a::kMaxMetadataTotal + json.max_string_bytes;
  // One fact pass may have both opened artifacts. Retained artifact/spec/source
  // owners transfer to their own StartupArtifactPayment before the next pass.
  // Settings retained for all
  // aliases are paid independently of transient parser reuse (including names,
  // calibration fields and their canonical/digest scratch).
  constexpr std::uint64_t fixed = 2 * artifact_open + 2 * metadata;
  // Retained settings/calibration, their copies and formatting, and the pending
  // alias/shared-owner vector allocated before each artifact-specific payment.
  // Parser scratch is in metadata; configured strings are paid separately.
  constexpr std::uint64_t per_alias =
      16 * (kMaxCalibrationBytes + sizeof(ModelSettings) + sizeof(Calibration)) +
      4 * (sizeof(OpenedModelArtifacts) + sizeof(std::shared_ptr<OpenedModelArtifacts>) + 64);
  constexpr std::uint64_t alignment = a::kChunkBytes;
  auto chat = ChatReaderBytes();
  if (!chat) return std::unexpected(chat.error());
  std::uint64_t bytes = 0, configured = 0;
  if (aliases == 0 || __builtin_mul_overflow(configuration_bytes, std::uint64_t{16}, &configured) ||
      __builtin_mul_overflow(aliases, per_alias, &bytes) ||
      __builtin_add_overflow(bytes, configured, &bytes) ||
      __builtin_add_overflow(bytes, fixed, &bytes) ||
      __builtin_add_overflow(bytes, *chat, &bytes) ||
      __builtin_add_overflow(bytes, alignment - 1, &bytes))
    return std::unexpected("startup parser loan overflows");
  return bytes / alignment * alignment;
}

std::expected<std::uint64_t, std::string> ChatReaderBytes() {
  constexpr base::json::Limits json;
  constexpr auto input = artifact::kMaxMetadataBytes;  // named readers have the same 64 MiB cap
  static_assert(kMaxTokenizerBytes == input);
  // JSON parser node/kid/pending + duplicate-key scratch (36/value).
  // HF postprocessor pending stacks, ordered definition map nodes and compact
  // ids/slices coexist: 3x growth for each, covered by 256 bytes/value.
  // Input, parsed text and copied token/merge/regex/template strings coexist;
  // config can be parsed while the tokenizer source and spec remain live.
  constexpr std::uint64_t json_work = (36 + 256) * json.max_values;
  constexpr std::uint64_t text = 16 * input + 4 * json.max_string_bytes;
  // HF config may resize a sparse ID up to kMaxVocabulary before rejecting it.
  // SentencePiece vector growth and HF config's sorted text->ID index coexist
  // with strings, kinds, scores and five bit vectors: 3x old/new capacities.
  constexpr std::uint64_t vocabulary =
      3 * tokenizer::kMaxVocabulary *
      (sizeof(std::string) + sizeof(tokenizer::TokenKind) + sizeof(float) + 5 +
       sizeof(std::pair<std::string_view, tokenizer::TokenId>));
  // GGUF conversion can construct a spec before Create's vocabulary cap. The
  // combined string-array count is bounded by their 8-byte wire lengths;
  // converted views and token/merge owners coexist, including rejected inputs.
  constexpr std::uint64_t arrays =
      3 * std::min<std::uint64_t>(tokenizer::kMaxGgufArray, input / 8) *
      (sizeof(std::string_view) + 2 * sizeof(std::string) + sizeof(float) + 2);
  constexpr std::uint64_t keys = 3 * tokenizer::kMaxGgufKeys * (sizeof(base::gguf::KeyValue) + 128);
  std::uint64_t bytes = 4096;
  for (const auto term : {json_work, text, vocabulary, arrays, keys})
    if (__builtin_add_overflow(bytes, term, &bytes))
      return std::unexpected("chat reader inventory overflows");
  if (__builtin_add_overflow(bytes, artifact::kChunkBytes - 1, &bytes))
    return std::unexpected("chat reader inventory alignment overflows");
  return bytes / artifact::kChunkBytes * artifact::kChunkBytes;
}

bool StartupReaderLoan::Covers(const catalog::Catalog& catalog, catalog::DomainId domain,
                               std::uint64_t bytes) const {
  if (lifetime_.expired() || catalog_ != &catalog || domain_ != domain || bytes > bytes_)
    return false;
  const auto extent = catalog.Describe(extent_);
  return extent && extent->state == catalog::ExtentState::kResident &&
         extent->descriptor.domain == domain &&
         extent->descriptor.recovery == catalog::Recovery::kPinned &&
         extent->descriptor.memory_class == catalog::MemoryClass::kRuntime &&
         extent->descriptor.size.value() >= bytes_;
}

std::expected<std::uint64_t, std::string> StartupArtifactBytes(const artifact::Artifact& artifact) {
  // Drafters may contribute a source without a stored spec; target admission
  // still requires one. Pay the retained source and any spec before either
  // admission is read, independently of that semantic requirement.
  auto bound = ComponentAdmissionBytes(artifact);
  if (!bound) return std::unexpected(bound.error());
  std::uint64_t bytes = *bound;
  const auto add = [&](std::uint64_t count, std::uint64_t unit) {
    std::uint64_t term = 0;
    return !__builtin_mul_overflow(count, unit, &term) &&
           !__builtin_add_overflow(bytes, term, &bytes);
  };
  // Retained manifest strings and doubled vector/string capacities. Index and
  // spec document sizes bound names and roles; records, chunk digests, source
  // placements and shared alias/control owners have separate inventory terms.
  if (!add(artifact::kMaxManifestBytes, 8) || !add(1, 4096))
    return std::unexpected("startup artifact metadata bound overflows");
  for (const auto& file : artifact.files())
    if ((file.role == artifact::FileRole::kIndex || file.role == artifact::FileRole::kSpec) &&
        !add(file.bytes.value(), 8))
      return std::unexpected("startup artifact document inventory overflows");
  for (const auto& [count, unit] :
       {std::pair{artifact.resources().size(), 4096ULL},
        std::pair{artifact.expert_arrays().size(), 4096ULL},
        std::pair{artifact.chunk_sha256().size(), 1024ULL},
        std::pair{artifact.groups().size(), 4096ULL}, std::pair{artifact.shards().size(), 1024ULL}})
    if (!add(count, unit)) return std::unexpected("startup artifact descriptor bound overflows");
  return bytes;
}

std::expected<std::uint64_t, std::string> StartupStructureBytes(const artifact::Artifact& target,
                                                                const artifact::Artifact* drafter) {
  auto bytes = ComponentAdmissionBytes(target);
  if (!bytes) return std::unexpected(bytes.error());
  if (drafter) {
    auto draft_bytes = ComponentAdmissionBytes(*drafter);
    if (!draft_bytes) return std::unexpected(draft_bytes.error());
    if (__builtin_add_overflow(*bytes, *draft_bytes, &*bytes))
      return std::unexpected("startup composition inventory overflows");
    // Individual source/spec storage plus canonical and effective composition.
    // Each composition is bounded by the two contributing admission inventories;
    // their descriptor terms also cover legacy recipe-generated component records.
    if (__builtin_mul_overflow(*bytes, std::uint64_t{3}, &*bytes))
      return std::unexpected("startup composition copies overflow");
    const auto* mapping = ComposedDrafterMapping(*drafter);
    if (mapping && mapping->draft && !mapping->draft->head_ids.empty()) {
      if (auto resource = drafter->FindResource(mapping->draft->head_ids)) {
        std::uint64_t heads = 0;
        if (__builtin_mul_overflow(drafter->resources()[*resource].bytes.value(), std::uint64_t{3},
                                   &heads) ||
            __builtin_add_overflow(*bytes, heads, &*bytes))
          return std::unexpected("startup selected-head inventory overflows");
      }
    }
  }
  return *bytes;
}

std::expected<std::uint64_t, std::string> StartupModelBytes(const artifact::Artifact& target,
                                                            const artifact::Artifact* drafter) {
  auto bytes = StartupArtifactBytes(target);
  if (!bytes) return std::unexpected(bytes.error());
  if (drafter) {
    auto draft_bytes = StartupArtifactBytes(*drafter);
    auto structures = StartupStructureBytes(target, drafter);
    if (!draft_bytes) return std::unexpected(draft_bytes.error());
    if (!structures) return std::unexpected(structures.error());
    if (__builtin_add_overflow(*bytes, *draft_bytes, &*bytes) ||
        __builtin_add_overflow(*bytes, *structures, &*bytes))
      return std::unexpected("retained startup model inventory overflows");
  }
  return *bytes;
}

std::expected<std::unique_ptr<StartupArtifactPayment>, std::string> StartupArtifactPayment::Admit(
    catalog::Catalog& catalog, catalog::DomainId domain, std::uint64_t bytes, std::uint64_t cap,
    std::uint64_t available) {
  constexpr std::uint64_t extent = artifact::kChunkBytes;
  std::uint64_t total = 0;
  if (bytes == 0 || __builtin_add_overflow(bytes, std::uint64_t{4096}, &total) ||
      __builtin_add_overflow(total, extent - 1, &total))
    return std::unexpected("startup artifact payment overflows");
  total = total / extent * extent;
  const auto occupied = catalog.OccupancyOf(domain).Total().value();
  if (occupied > cap || total > cap - occupied || total > available ||
      kUncountedMargin > available - total)
    return std::unexpected("retained startup artifacts exceed the node execution budget");
  auto payment = catalog.AddExtent({.domain = domain,
                                    .memory_class = catalog::MemoryClass::kRuntime,
                                    .recovery = catalog::Recovery::kPinned,
                                    .size = base::Bytes(total),
                                    .content = {}},
                                   true);
  if (!payment) return std::unexpected("startup artifact payment could not be cataloged");
  return std::unique_ptr<StartupArtifactPayment>(
      new StartupArtifactPayment(catalog, domain, *payment, total));
}

std::expected<std::unique_ptr<StartupArtifactPayment>, std::string>
StartupArtifactPayment::AdmitReader(catalog::Catalog& catalog, catalog::DomainId domain,
                                    std::uint64_t bytes, std::uint64_t cap,
                                    std::uint64_t available) {
  auto owner = Admit(catalog, domain, bytes, cap, available);
  if (owner) (*owner)->reader_ = true;
  return owner;
}

StartupArtifactPayment::~StartupArtifactPayment() {
  lifetime_.reset();
  base::Check(
      catalog_->ReleasePinned(extent_).has_value() && catalog_->RemoveExtent(extent_).has_value(),
      "startup artifact payment lost its held catalog extent");
}

std::uint64_t MeasuredStartupCeiling(std::uint64_t available) {
  constexpr std::uint64_t extent = std::uint64_t{2} << 20U;
  return available > kUncountedMargin ? (available - kUncountedMargin) / extent * extent : 0;
}

std::expected<StartupBudget, std::string> OrdinaryStartupBudget(std::uint64_t aliases,
                                                                std::uint64_t available) {
  auto parser = StartupBootstrapBytes(aliases);
  if (!parser) return std::unexpected(parser.error());
  const std::uint64_t cap = MeasuredStartupCeiling(available);
  if (cap == 0)
    return std::unexpected("startup needs known physical memory past the uncounted margin");
  return StartupBudget{.parser_loan = *parser, .cap = cap};
}

std::expected<void, std::string> CheckStartupHostAllowance(std::uint64_t bytes, std::uint64_t cap,
                                                           std::uint64_t available) {
  constexpr std::uint64_t extent = std::uint64_t{2} << 20U;
  if (bytes == 0 || bytes % extent != 0 || cap == 0 || bytes > cap)
    return std::unexpected("invalid startup host allowance or execution cap");
  if (cap > available || kUncountedMargin > available - cap)
    return std::unexpected("startup host allowance requires known physical headroom");
  return {};
}

std::expected<void, std::string> CheckDiagnosticBudgetCap(const MemoryGuard& guard,
                                                          std::uint64_t dynamic_budget,
                                                          std::uint64_t cap,
                                                          std::uint64_t extra_pinned) {
  if (guard.available == 0)
    return std::unexpected("diagnostic budget cap requires known physical availability");
  if (auto checked = CheckMemoryGuard(guard); !checked) return checked;
  if (cap > dynamic_budget)
    return std::unexpected("diagnostic budget cap exceeds the ordinary dynamic budget");
  // Subtract the required terms rather than adding an overflowing minimum.
  if (guard.fixed > cap || guard.largest > cap - guard.fixed ||
      extra_pinned > cap - guard.fixed - guard.largest)
    return std::unexpected("diagnostic budget cap is below the required startup footprint");
  return {};
}

}  // namespace llmp::runtime
