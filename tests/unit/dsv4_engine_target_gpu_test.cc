// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Full prepared DeepSeek target through the serving factory and shared runner.
// Default-off diagnostic; ordinary complete weight closure, not selected-only
// expert paging. Exact comparisons are one implementation's snapshot replay.

#include <gtest/gtest.h>
#include <sys/stat.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <expected>
#include <filesystem>
#include <format>
#include <fstream>
#include <iomanip>
#include <memory>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "artifact/json.h"
#include "base/sha256.h"
#include "corpus_score.h"
#include "engine/component_source.h"
#include "engine/expert_slab.h"
#include "engine/paged_weights.h"
#include "engine/plan_cache.h"
#include "execution/registry.h"
#include "expected_error.h"
#include "kernels/common/step_program.h"
#include "kernels/ggml/graph_plan.h"
#include "model/dsv4.h"
#include "platform/host_probe.h"
#include "runtime/component_build.h"
#include "runtime/engine_llm.h"
#include "runtime/memory_guard.h"
#include "runtime/serving.h"
#include "runtime/snapshot_store.h"
#include "v1_frozen.h"

namespace {
namespace fs = std::filesystem;
namespace en = llmp::engine;
namespace rt = llmp::runtime;
namespace pl = llmp::planner;
namespace kn = llmp::kernels;
namespace ts = llmp::test_support;
using Status = std::expected<void, std::string>;
using llmp::test_support::Failed;
constexpr std::uint64_t kMiB = std::uint64_t{1} << 20U;
constexpr std::uint64_t kGiB = std::uint64_t{1} << 30U;
constexpr std::uint64_t kCap = 110 * kGiB;
constexpr std::uint64_t kConstruction = 4 * kGiB;
constexpr std::uint64_t kConstructionExtent = kConstruction + 2 * kMiB;
constexpr std::uint64_t kCaller = 32 * kMiB;
constexpr std::uint64_t kWeights = 100032053248ULL;
constexpr std::uint64_t kState = 73400320ULL;
constexpr std::uint64_t kDsparkWeights = 10898898944ULL;
constexpr std::uint64_t kDsparkState = 2097152ULL;
constexpr std::uint64_t kSnapshot = kState + 2 * kMiB;
constexpr std::uint32_t kContext = 3072, kChunk = 512, kPrefix = 2560, kTail = 8;
constexpr std::uint32_t kVocab = 129280;
constexpr std::string_view kArtifact =
    "8a355bfb27c90e1150fbd7fa62ea6e63f6bf34fcca33934e52d22773f1508234";

constexpr std::string_view kDsparkArtifact =
    "dd2d3f9c66f070fb231d27d5a11f38ff22c78dc8f089cecedbb67721e9b4bec5";

// Current compiler and fixed-source simultaneous ownership inventory. The
// external funding table enumerates phases and source anchors; this is a
// conservative bound, not an observed allocation or a generic model grant.
static_assert(llmp::ir::Graph::kMaxNodes == 65536 && llmp::ir::Graph::kMaxTensors == 131072);
static_assert(sizeof(llmp::ir::Node) <= 160 && sizeof(llmp::ir::TensorDesc) <= 160);
static_assert(sizeof(llmp::ir::Binding) <= 24 && sizeof(llmp::format::TensorFormat) <= 72);
static_assert(sizeof(pl::Selected) <= 128 && sizeof(pl::Applied) <= 64);
static_assert(sizeof(pl::Match) + sizeof(void*) <= 256);
static_assert(sizeof(pl::RootAlignmentFact) <= 16);
static_assert(sizeof(kn::InputSlot) <= 512 && sizeof(kn::TensorUse) <= 64);
static_assert(sizeof(kn::ggml::PlanStep) <= 64);
static_assert(sizeof(llmp::execution::OperandMemory) <= 48 &&
              sizeof(llmp::execution::OperandFormat) <= 112 &&
              sizeof(llmp::execution::CapabilityQuery) <= 256);
static_assert(llmp::execution::Registry::kMaxImplementations == 256);
static_assert(llmp::ir::kMaxSources == 10 && llmp::ir::kMaxOutputs == 4);
static_assert(sizeof(llmp::artifact::Resource) <= 512 &&
              sizeof(llmp::artifact::ExpertArray) <= 512 && sizeof(llmp::artifact::Group) <= 64);
static_assert(sizeof(en::SlabPage) <= 128 && sizeof(llmp::scheduler::PageSource) <= 288 &&
              sizeof(llmp::catalog::ExtentView) <= 192);
static_assert(sizeof(llmp::spec::SourceTensor) <= 128 && sizeof(llmp::model::Dsv4Tensor) <= 96);
static_assert(sizeof(std::string) <= 32 && sizeof(std::vector<void*>) <= 24);
static_assert(llmp::artifact::kMaxDirectoryEntries == 65536);
// The current common compiler's largest (including seven-graph MTP overlap,
// which this target never uses) plus the independently paid owners below.
// Additional current shared helper and required-owner metadata: eight full
// tensor-index records of <=80B plus an 8MiB registry/owner workspace.
constexpr std::uint64_t kCurrentCompilerMetadata =
    8 * llmp::ir::Graph::kMaxTensors * 80ULL + 8 * kMiB;
// Lowering::Run: T descriptors, two disjoint pattern rounds' <=8N
// replacement outputs, and <=15 per planned node (six-way ExpertReduce).
// The closed DS target has no GDN/PLE commit saves. This pays the full hard
// bound, without inheriting the historical fixture's 65,536-descriptor limit.
constexpr std::uint64_t kNativeDescriptors = 131072ULL + 16 + 23 * 65536ULL;
constexpr std::uint64_t kNativeArena = kNativeDescriptors * 512 + (1ULL << 22U) * 8 + 4096;
constexpr std::uint64_t kCurrentLower =
    1074071952ULL - 33554432ULL + kNativeArena + kNativeDescriptors * (12 * 8 + 128);
static_assert(kCurrentLower < 2684354560ULL);
constexpr std::uint64_t kCompiler = 2684354560ULL + kCurrentCompilerMetadata;
// Reassign the unused compiler envelope to the sequential reader index;
// the complete compiler and construction ceilings remain unchanged.
constexpr std::uint64_t kOtherLower = kCurrentLower - 32 * kMiB + kCurrentCompilerMetadata;
constexpr std::uint64_t kReaderScratch = kCompiler - kOtherLower;
static_assert(kOtherLower < kCompiler && kReaderScratch == 437958768ULL);
static_assert(kOtherLower + kReaderScratch == kCompiler);
// Post-Lower digest: two index arrays and the growing leaf-digest array.
static_assert(4 * (131072ULL + 65536ULL) + 3 * 65536ULL * 32 + 65536 < kReaderScratch);
constexpr std::uint64_t kArtifactParser =
    3 * 530000ULL * (24 + 4 + 4) + 16 * 530000ULL + 3 * 6 * kMiB;
constexpr std::uint64_t kArtifacts =
    4 * (3 * 1500ULL * 4096 + 3 * 12000ULL * 64 + 55000ULL * 32 + 3 * 4096ULL * 64);
constexpr std::uint64_t kTokenizer =
    3 * 4 * 130000ULL * 32 + 3 * 2 * 6 * kMiB + 2 * 130000ULL * (64 + 3 * 8) + 3 * 16 * kMiB +
    750000ULL * (12 + 4 + 4 + 16) + 3 * 2 * 130000ULL * 16 + 2 * kMiB;
constexpr std::uint64_t kCatalog = 55000ULL * (3 * 192 + 3 * 288 + 320 + 3 * 128 + 3 * 64 + 128);
constexpr std::uint64_t kBindings =
    6 * 1500ULL * 2048 + 3 * 4096ULL * 256 + 3 * 6 * kVocab * 4ULL + 4 * kMiB;
constexpr std::uint64_t kDirectory = 3 * 65536ULL * (32 + 2 * 256);
static_assert(kCompiler + kArtifactParser + kArtifacts + kTokenizer + kCatalog + kBindings +
                  kDirectory + kCaller <=
              kConstruction);
static_assert(2 * (kTail + 1) * kVocab * sizeof(float) + 4 * kMiB < kCaller);

constexpr std::uint32_t kScoreInputs = 257, kLongScoreInputs = 2561, kScoreChunk = 128;
constexpr std::uint64_t kScoreOutputs = 4ULL * kVocab * kScoreChunk;
constexpr std::uint64_t ScoreHost(std::uint32_t inputs) {
  return 16ULL * inputs + 4 * kScoreOutputs + 4ULL * kVocab + kMiB;
}
constexpr std::uint64_t ScoreFixed(std::uint32_t inputs) {
  return kCompiler + kArtifactParser + kArtifacts + kTokenizer + kCatalog + kBindings + kDirectory +
         kCaller + ScoreHost(inputs);
}
constexpr std::uint64_t ScoreCredit(std::uint32_t inputs) {
  return kConstruction - ScoreFixed(inputs);
}
static_assert(ScoreFixed(kLongScoreInputs) < kConstruction);
static_assert(ScoreCredit(kScoreInputs) == 567071792ULL);
// The factory retains this complete grant. Reuse its named residual; charge
// only the positive excess separately, never charge the held residual twice.
std::expected<std::uint64_t, std::string> HeldScoreCredit(
    const en::ExecutionMetadataAllowance& allowance, std::uint64_t plans,
    std::uint32_t inputs = kScoreInputs) {
  if ((inputs != kScoreInputs && inputs != kLongScoreInputs) ||
      allowance.capacity() < kConstruction || allowance.used() != kConstruction)
    return std::unexpected("scored plans lack the complete held factory construction charge");
  return std::min(plans, ScoreCredit(inputs));
}
// Complete dimensions are retained, while selected implementations are hashed
// in fixed storage. No graph-sized index strings or escaped copies accumulate.
constexpr std::uint64_t kClassReceipt = kMiB;
constexpr std::uint32_t kParameterWords = 512;
constexpr std::uint32_t kScoreClasses = 5 + (kLongScoreInputs + kScoreChunk - 1) / kScoreChunk;
static_assert(kScoreClasses * (kParameterWords * 21ULL + 1024) < kClassReceipt);
static_assert(4 * kClassReceipt + 4 * kMiB < kCaller);
static_assert(kScoreOutputs <= 64 * kMiB && ScoreHost(kScoreInputs) == 266335248ULL);

std::expected<std::vector<pl::StepShape>, std::string> ScoreSchedule(llmp::model::Batching caps,
                                                                     std::uint32_t context,
                                                                     std::uint32_t read_align,
                                                                     std::uint32_t inputs) {
  if (inputs < 2 || inputs > context || context > kContext)
    return std::unexpected("scored schedule exceeds the paid context");
  auto composer = en::WaveComposer::Make(caps, context, read_align);
  if (!composer) return std::unexpected(composer.error());
  std::vector<pl::StepShape> out;
  out.reserve((inputs + kScoreChunk - 1) / kScoreChunk);
  for (std::uint32_t past = 0; past < inputs; past += kScoreChunk) {
    const auto rows = std::min(kScoreChunk, inputs - past);
    const std::array pending{
        en::PendingWork{.phase = pl::Phase::kScore, .past = past, .rows = rows, .outputs = rows}};
    auto composed = composer->Compose(pending);
    if (!composed.refused.empty() || composed.steps.size() != 1)
      return std::unexpected("scored schedule did not compose to one complete owner");
    out.push_back(std::move(composed.steps.front().shape));
  }
  return out;
}

struct ScoreClass {
  pl::PlanKey key;
  bool execute = false, measured = false, lowered = false;
  std::uint64_t host_bytes = 0, launches = 0, selected_steps = 0, paid_bytes = 0;
  std::string selection;
  std::uint64_t indexers = 0, index_cells = 0, index_selected = 0;
};
static_assert(sizeof(ScoreClass) < 512);
static_assert(kScoreClasses * (2 * kParameterWords * sizeof(std::uint64_t) + 1024) < kMiB);

std::expected<std::uint64_t, std::string> ScorePlanPayment(std::span<const ScoreClass> classes) {
  std::uint64_t total = 0, largest = 0;
  for (std::size_t i = 0; i < classes.size(); ++i) {
    const auto& item = classes[i];
    if (!item.measured) return std::unexpected("incomplete score class measurement");
    if (!item.execute) continue;
    if (std::ranges::any_of(classes.first(i), [&](const auto& prior) {
          return prior.execute && prior.key == item.key;
        }))
      continue;
    std::uint64_t host = 0, launches = 0;
    for (const auto& peer : classes) {
      if (!peer.execute || peer.key != item.key) continue;
      if (peer.selection != item.selection || peer.selected_steps != item.selected_steps)
        return std::unexpected("cache-equivalent measured selections differ");
      host = std::max(host, peer.host_bytes);
      launches = std::max(launches, peer.launches);
    }
    if (host > kCap || launches > (kCap - host) / en::kGraphNodeHostBytes)
      return std::unexpected("cache-equivalent owner tuple exceeds node geometry");
    const auto bytes = host + launches * en::kGraphNodeHostBytes;
    if (bytes > kCap - total) return std::unexpected("retained scored plans exceed node geometry");
    total += bytes;
    largest = std::max(largest, bytes);
  }
  if (largest > kCap - total)
    return std::unexpected("scored replacement owner exceeds node geometry");
  return total + largest;
}

rt::ModelSettings Settings(bool companion = false) {
  rt::ModelSettings s;
  s.name = companion ? "deepseek-component-dspark" : "deepseek-component-target";
  s.architecture = "deepseek4";
  s.context.value = kContext;
  s.prefill_chunk.value = kChunk;
  s.max_slots.value = 1;
  s.wave_read_align.value = 256;
  s.speculation.value = companion;
  if (companion) s.draft_rows.value = 3;
  return s;
}
std::expected<std::shared_ptr<const en::NumericalDomain>, std::string> Domain(
    bool companion = false, bool scoring = false) {
  if (scoring) {
    if (companion) return std::unexpected("scored companion is not admitted");
    auto score_domain = en::NumericalDomain::Create(
        {{.stage = 0,
          .name = "target",
          .uses = {en::NumericalUse::kPrompt, en::NumericalUse::kDecode, en::NumericalUse::kScore},
          .phases = {{.phase = pl::Phase::kPrefill,
                      .max_owners = 1,
                      .max_owner_rows = kChunk,
                      .max_total_rows = kChunk},
                     {.phase = pl::Phase::kDecode,
                      .max_owners = 1,
                      .max_owner_rows = 1,
                      .max_total_rows = 1},
                     {.phase = pl::Phase::kScore,
                      .max_owners = 1,
                      .max_owner_rows = kScoreChunk,
                      .max_total_rows = kScoreChunk}}}});
    if (!score_domain) return std::unexpected(score_domain.error());
    return std::make_shared<const en::NumericalDomain>(std::move(*score_domain));
  }
  if (companion) {
    const std::vector uses{en::NumericalUse::kPrompt, en::NumericalUse::kDecode,
                           en::NumericalUse::kPromptCompanions, en::NumericalUse::kSpeculative};
    auto composed = en::NumericalDomain::Create({{.stage = 0,
                                                  .name = "target",
                                                  .uses = uses,
                                                  .phases = {{.phase = pl::Phase::kPrefill,
                                                              .max_owners = 1,
                                                              .max_owner_rows = kChunk,
                                                              .max_total_rows = kChunk},
                                                             {.phase = pl::Phase::kDecode,
                                                              .max_owners = 1,
                                                              .max_owner_rows = 1,
                                                              .max_total_rows = 1},
                                                             {.phase = pl::Phase::kVerify,
                                                              .max_owners = 1,
                                                              .max_owner_rows = 4,
                                                              .max_total_rows = 4}}},
                                                 {.stage = 1,
                                                  .name = "dspark",
                                                  .uses = uses,
                                                  .phases = {{.phase = pl::Phase::kDraft,
                                                              .max_owners = 1,
                                                              .max_owner_rows = 3,
                                                              .max_total_rows = 3}}}});
    if (!composed) return std::unexpected(composed.error());
    return std::make_shared<const en::NumericalDomain>(std::move(*composed));
  }
  auto d =
      en::NumericalDomain::Create({{.stage = 0,
                                    .name = "target",
                                    .uses = {en::NumericalUse::kPrompt, en::NumericalUse::kDecode},
                                    .phases = {{.phase = pl::Phase::kPrefill,
                                                .max_owners = 1,
                                                .max_owner_rows = kChunk,
                                                .max_total_rows = kChunk},
                                               {.phase = pl::Phase::kDecode,
                                                .max_owners = 1,
                                                .max_owner_rows = 1,
                                                .max_total_rows = 1}}}});
  if (!d) return std::unexpected(d.error());
  return std::make_shared<const en::NumericalDomain>(std::move(*d));
}

// Bounded streaming checks before JSON/tokenizer construction. Artifact::Open
// authenticates every prepared header again; no full weight hash pass is added.
Status File(const fs::path& path, std::uint64_t bytes, std::string_view digest) {
  std::FILE* f = std::fopen(path.c_str(), "rb");
  if (f == nullptr) return std::unexpected("missing fixed DeepSeek metadata: " + path.string());
  llmp::base::Sha256 hash;
  std::array<std::byte, 65536> buffer{};
  std::uint64_t read = 0;
  while (read < bytes) {
    const auto wanted =
        static_cast<std::size_t>(std::min<std::uint64_t>(bytes - read, buffer.size()));
    const auto got = std::fread(buffer.data(), 1, wanted, f);
    if (got != wanted) {
      std::fclose(f);
      return std::unexpected("short fixed DeepSeek metadata");
    }
    hash.Update(std::span(buffer).first(got));
    read += got;
  }
  const bool eof = std::fgetc(f) == EOF && !std::ferror(f);
  std::fclose(f);
  if (!eof || llmp::base::ToHex(hash.Finish()) != digest)
    return std::unexpected("fixed DeepSeek metadata identity differs");
  return {};
}
Status Metadata(const fs::path& artifact) {
  if (auto s = File(artifact / "manifest.json", 14492, kArtifact); !s) return s;
  if (auto s = File(artifact / "index.json", 4994976,
                    "68ddac15cff7cfcac5162beab7559884632244651797aed1b000cc7325ea24fc");
      !s)
    return s;
  return File(artifact / "meta/DeepSeek-V4-Flash-0731-UD-Q2_K_XL-00001-of-00003.kv.gguf", 5257664,
              "de65c5bb660817b95cc281bf54935f9d1bcc3b22535f2eae63b35a14c7c5724b");
}

struct Lifetime {
  explicit Lifetime(bool with_companion = false, bool with_score = false,
                    std::uint32_t inputs = kScoreInputs)
      : companion(with_companion), scoring(with_score), score_size(inputs) {}
  bool companion, scoring;
  std::uint32_t score_size;
  std::vector<ScoreClass> score_classes;
  pl::PlanKey score_base;
  bool score_keys_sealed = false;
  en::StepSource* source = nullptr;  // Owned by the actual EngineLlm runner.
  llmp::model::Batching score_caps{};
  std::array<std::int32_t, kLongScoreInputs> score_ids{};
  std::array<ts::CorpusTargetScore, kLongScoreInputs - 1> scores{};
  std::array<bool, kLongScoreInputs> seen{};
  std::span<const std::int32_t> ScoreIds() const { return std::span(score_ids).first(score_size); }
  std::uint32_t score_count = 0;
  rt::RequestMemory score_tokens{4 * kMiB};
  std::uint64_t StateBytes() const { return kState + (companion ? kDsparkState : 0); }
  std::uint64_t SnapshotBytes() const { return companion ? StateBytes() + 4 * kMiB : kSnapshot; }
  std::uint64_t WeightBytes() const { return kWeights + (companion ? kDsparkWeights : 0); }
  en::PagedNode node{{.slot_bytes = en::kSlabSlotBytes}};
  // Complete caller storage is prepaid with construction; local token memory
  // also enforces its own small limit before any history-vector growth.
  rt::RequestMemory tokens{kMiB};
  std::shared_ptr<en::ExecutionMetadataAllowance> allowance, score_plan_allowance;
  std::uint64_t score_plan_bytes = 0, score_extra_extent = 0;
  std::unique_ptr<rt::EngineLlm> model;
  std::optional<en::WaveComposer> score_composer;
  std::unique_ptr<ts::CorpusScore> scorer;
  std::unique_ptr<rt::SnapshotStore> prefix, end;
  std::vector<en::PagedModel*> entered;
  std::array<std::int32_t, kPrefix> ids{};
  std::vector<float> head;
  rt::Generation first, second;
  std::uint64_t programs = 0, full_programs = 0, footprint = 0, dynamic = 0;
  std::uint64_t draft_programs = 0, injected_programs = 0;
  std::uint64_t fixed_bytes = 0, activation_bytes = 0, pool_bytes = 0, input_bytes = 0,
                plan_bytes = 0;
  llmp::base::Sha256 recipes;
  std::string restored_state_digest;
  bool teardown_attempted = false;

  Status Validate(const kn::LowerInput& in, const kn::StepProgram& program) {
    if (!in.token) return std::unexpected("token source metadata is required");
    // This callback runs after Lower and before Bind, including measurement.
    // A real launch's BoundPlan does not exist at this point.
    if (in.token->shape.stage > (companion ? 1 : 0) || in.token->shape.segments.size() != 1 ||
        program.routed() || in.plan.steps.empty() || program.launches() == 0)
      return std::unexpected("the target did not lower to its ordinary owned full-step program");
    if (scoring) {
      if (source == nullptr) return std::unexpected("missing scored source");
      // ModelRunner prepares the actual backend identity before its first
      // measuring Build. Bind every key to that prepared identity, not the
      // factory's provisional KeyBase before PrepareExecution.
      if (!score_keys_sealed) {
        const auto identity = source->ExecutionIdentity();
        if (in.addresses.activations != 0 || !identity || !identity->has_value())
          return std::unexpected("scored key base was not prepared before measurement");
        score_base = source->KeyBase();
        for (auto& item : score_classes) {
          item.key.spec = score_base.spec;
          item.key.patterns = score_base.patterns;
          item.key.policy = score_base.policy;
          item.key.partition = score_base.partition;
        }
        score_keys_sealed = true;
      }
      auto key = source->KeyBase();
      key.shape = in.token->shape;
      key.parameters = source->ShapeParameters(in.token->shape);
      const auto record = std::ranges::find_if(score_classes, [&](const auto& item) {
        return item.key == key &&
               item.key.shape.segments.front().past == in.token->shape.segments.front().past;
      });
      if (record == score_classes.end())
        return std::unexpected("score/setup shape parameters were not prepaid and measured");
      // Unused maximum/setup programs are transient and cannot execute.
      const bool retained = record->execute;
      if (!retained && in.addresses.activations != 0)
        return std::unexpected("measurement-only setup class cannot execute in the scored run");
      const auto host = program.host_bytes(), launches = program.launches();
      if (host > kCompiler || (retained && launches > (kCap - host) / en::kGraphNodeHostBytes))
        return std::unexpected("measured program exceeds representable paid node geometry");
      const auto paid = host + (retained ? launches * en::kGraphNodeHostBytes : 0);
      if (in.addresses.activations != 0 &&
          (!record->measured || host > record->host_bytes || launches > record->launches ||
           paid > record->paid_bytes))
        return std::unexpected("bound scored program exceeds its measured admitted owner");
      if (in.addresses.activations == 0)
        record->measured = true;
      else {
        if (!record->measured) return std::unexpected("score shape ran before measurement");
        record->lowered = true;
      }
      if (in.token->shape.phase == pl::Phase::kScore) {
        if (auto head_ok = ts::CheckCorpusHead(in, program, kVocab); !head_ok) return head_ok;
      }
      if (!record->selection.empty() && record->selected_steps != in.plan.steps.size())
        return std::unexpected("measured and bound selection counts differ");
      record->host_bytes = std::max(record->host_bytes, host);
      record->launches = std::max(record->launches, launches);
      record->selected_steps = in.plan.steps.size();
      if (retained && record->launches > (kCap - record->host_bytes) / en::kGraphNodeHostBytes)
        return std::unexpected("combined measured tuple exceeds node geometry");
      record->paid_bytes =
          record->host_bytes + (retained ? record->launches * en::kGraphNodeHostBytes : 0);
      if (in.plan.steps.size() > llmp::ir::Graph::kMaxNodes)
        return std::unexpected("selection receipt exceeds prepaid source geometry");
      llmp::base::Sha256 selected;
      selected.Update("llmp.ds.selected-indices.v1");
      std::array<std::byte, 8> count{};
      for (std::size_t byte = 0; byte < count.size(); ++byte)
        count[byte] = static_cast<std::byte>(std::uint64_t{in.plan.steps.size()} >> (8 * byte));
      selected.Update(count);
      for (const auto& item : in.plan.steps) {
        if (item.implementation >= llmp::execution::Registry::kMaxImplementations)
          return std::unexpected("selection index exceeds registry geometry");
        const std::array index{static_cast<std::byte>(item.implementation)};
        selected.Update(index);
      }
      const auto digest = llmp::base::ToHex(selected.Finish());
      if (!record->selection.empty() && record->selection != digest)
        return std::unexpected("measured and bound selection differs for the same complete key");
      record->selection = digest;
      std::uint64_t indexers = 0, cells = 0, kept = 0;
      const auto& graph = in.token->step.graph;
      for (const auto& op : graph.nodes()) {
        if (op.key.op != llmp::ir::Op::kTopK) continue;
        const auto producer = graph.producer(op.src[0]);
        if (!producer.valid() || graph.node(producer).key.op != llmp::ir::Op::kIndexScore) continue;
        ++indexers;
        cells = std::max(cells, graph.tensor(op.src[0]).ne(0));
        kept = std::max(kept, graph.tensor(op.out[0]).ne(0));
      }
      if (record->indexers != 0 && (record->indexers != indexers || record->index_cells != cells ||
                                    record->index_selected != kept))
        return std::unexpected("measured and bound indexer geometry differs");
      record->indexers = indexers;
      record->index_cells = cells;
      record->index_selected = kept;
    }
    std::uint32_t products = 0, arrays = 0, hash_inputs = 0;
    for (const auto& weight : in.token->step.weights)
      if (weight.ref.expert_array) ++arrays;
    for (const auto& input : program.inputs())
      if (input.kind == kn::InputKind::kSideRows) ++hash_inputs;
    for (const auto& n : in.token->step.graph.nodes()) {
      if (n.key.op != llmp::ir::Op::kMatMulId) continue;
      ++products;
      const auto& a = in.token->step.graph.tensor(n.src[0]);
      const auto& selected_ids = in.token->step.graph.tensor(n.src[2]);
      if (a.shape().ne[2] != 256 || selected_ids.shape().ne[0] != 6)
        return std::unexpected(
            "an expert product omits the declared expert or selected-slot domain");
    }
    const bool drafting = in.token->shape.stage == 1;
    const bool headed = in.token->shape.segments.front().outputs != 0;
    // Injected target taps retain the final FFN even on no-head prompt chunks.
    const std::uint32_t expected = drafting ? 9U : (headed || companion ? 129U : 126U);
    if (arrays != expected || products != expected || hash_inputs != (drafting ? 0U : 3U))
      return std::unexpected(std::format(
          "live expert domain differs: stage={} arrays={} products={} hash_inputs={} expected={} "
          "rows={} outputs={}",
          in.token->shape.stage, arrays, products, hash_inputs, expected,
          in.token->shape.segments.front().rows, in.token->shape.segments.front().outputs));
    if (companion) {
      std::uint32_t injected = 0;
      for (const auto& state : in.token->step.state)
        if (state.ref.stage == 1 && state.written && state.tap) ++injected;
      if (!drafting && injected != 3)
        return std::unexpected("the real target graph lost a DSpark injection state write");
      if (drafting)
        ++draft_programs;
      else
        ++injected_programs;
    }
    ++programs;
    if (in.token->step.logits.valid() &&
        in.token->step.graph.tensor(in.token->step.logits).shape().ne[0] == kVocab)
      ++full_programs;
    // Record the actual lowered selection, then bind its registry indices to
    // the immutable execution identity after Setup. This is not a BoundPlan
    // identity or proof that a measuring program was executed.
    const auto graph = in.plan.graph.Digest();
    recipes.Update("llmp.ds.lowered-selection.v1");
    recipes.Update(std::as_bytes(std::span(graph)));
    const auto word = [this](std::uint64_t value) {
      recipes.Update(std::as_bytes(std::span(&value, 1)));
    };
    word(in.plan.steps.size());
    for (const auto& selected : in.plan.steps) {
      word(selected.implementation);
      word(selected.lane);
      word(selected.nodes.size());
      for (const auto node_id : selected.nodes) word(node_id.v);
    }
    return {};
  }

  Status Run(const fs::path& store, const fs::path& scratch) {
    if (auto s = Metadata(store / kArtifact); !s) return s;
    if (companion) {
      const auto path = store / kDsparkArtifact;
      if (auto s = File(path / "manifest.json", 1991, kDsparkArtifact); !s) return s;
      if (auto s = File(path / "index.json", 500693,
                        "437561ad6dd1e8d5fd3758be2a8eac52b8a1c81f6a1fb54b27e0b3badfeaa11c");
          !s)
        return s;
      if (auto s = File(path / "meta/dspark-DeepSeek-V4-Flash-0731-Q8_0.kv.gguf", 5248193,
                        "fcf6886653e8a64b70d51de8896febe2d69ae3dba8e300ed4684b518bbfcddff");
          !s)
        return s;
    }
    const auto before = llmp::platform::AvailableMemoryBytes().value_or(0);
    if (auto s = rt::CheckStartupHostAllowance(kConstructionExtent, kCap, before); !s)
      return std::unexpected(std::format("{} (available_bytes={}, cap_bytes={}, margin_bytes={})",
                                         s.error(), before, kCap, rt::kUncountedMargin));
    if (auto s = node.Open(); !s) return s;
    auto paid = en::ExecutionMetadataAllowance::Admit(
        node.catalog(), node.domain(), kConstructionExtent,
        llmp::platform::AvailableMemoryBytes().value_or(0), rt::kUncountedMargin);
    if (!paid) return std::unexpected(paid.error());
    allowance = std::move(*paid);
    auto domain = Domain(companion, scoring);
    if (!domain) return std::unexpected(domain.error());
    llmp::config::ModelEntry entry;
    entry.name = Settings(companion).name;
    entry.artifact = std::string(kArtifact);
    if (companion) entry.drafter = std::string(kDsparkArtifact);
    llmp::config::RuntimeRoles roles;
    roles.installed = store;
    roles.spill = scratch;
    auto build = rt::ComponentLlmBuild(
        node, store, entry, 0,
        {.target_policy = std::nullopt,
         .diagnostic_solo = true,
         .construction_metadata = allowance,
         .execution_domain = *domain,
         .validate_program = [this](const kn::LowerInput& in, const kn::StepProgram& program) {
           return Validate(in, program);
         }});
    if (scoring) {
      if (score_size != kScoreInputs && score_size != kLongScoreInputs)
        return std::unexpected("unknown captured scored corpus");
      build = [this, inner = std::move(build)](const rt::ModelSettings& settings) mutable
          -> std::expected<rt::EngineLlmParts, std::string> {
        auto parts = inner(settings);
        if (!parts) return std::unexpected(parts.error());
        source = parts->source.get();
        score_caps = parts->caps;
        const auto& slots = parts->config.slots;
        if (slots.stages.size() != 1 || slots.slots != 1)
          return std::unexpected("scored schedule requires its single target stage");
        const auto& stage = slots.stages.front();
        const auto setup_count = parts->config.setup_shapes.size();
        if (setup_count > kScoreClasses - (score_size + kScoreChunk - 1) / kScoreChunk)
          return std::unexpected("score setup class storage exceeds caller payment");
        auto actual =
            ScoreSchedule(score_caps, stage.layout.context,
                          stage.read_align ? stage.read_align : slots.read_align, score_size);
        if (!actual) return std::unexpected(actual.error());
        parts->config.setup_shapes.insert(parts->config.setup_shapes.end(), actual->begin(),
                                          actual->end());
        score_classes.reserve(parts->config.setup_shapes.size());
        for (std::size_t i = 0; i < parts->config.setup_shapes.size(); ++i) {
          auto key = score_base;
          key.shape = parts->config.setup_shapes[i];
          key.parameters = source->ShapeParameters(key.shape);
          if (key.parameters.size() > kParameterWords)
            return std::unexpected("score shape parameters exceed caller payment");
          score_classes.push_back(
              {.key = std::move(key), .execute = i >= setup_count, .selection = {}});
        }
        return parts;
      };
    }
    model = std::make_unique<rt::EngineLlm>(node, entry, Settings(companion), roles, 0,
                                            std::move(build));
    model->SetTokenMemory(tokens);
    model->set_turn_checkpoints(false);
    const auto setup = model->Setup();
    if (model->has_paged()) entered.push_back(&model->paged());
    if (!setup) return setup;
    if (scoring) {
      if (auto inputs = ScoreInputs(); !inputs) return inputs;
      auto measured = ScorePlanPayment(score_classes);
      if (!measured) return std::unexpected(measured.error());
      score_plan_bytes = *measured;
      const auto credit = HeldScoreCredit(*allowance, score_plan_bytes, score_size);
      if (!credit) return std::unexpected(credit.error());
      const auto excess = score_plan_bytes - *credit;
      if (excess != 0) {
        const auto overhead = en::ExecutionMetadataAllowance::OverheadBytes();
        if (excess > kCap - overhead - en::kPagedExtent)
          return std::unexpected("scored metadata extent cannot be represented");
        score_extra_extent =
            (excess + overhead + en::kPagedExtent - 1) / en::kPagedExtent * en::kPagedExtent;
      }
    }
    const auto execution = model->ExecutionIdentity();
    if (!execution || !execution->has_value())
      return std::unexpected("the actual target has no complete execution identity after Setup");
    recipes.Update("llmp.ds.execution.v1");
    recipes.Update(std::as_bytes(std::span(**execution)));
    if (model->logit_width() != kVocab || node.StateCapacity() != StateBytes())
      return std::unexpected(
          "the actual target head or independently derived state layout differs");
    const auto weight_bytes = model->weights().size() * en::kPagedExtent;
    if (weight_bytes != WeightBytes())
      return std::unexpected("the actual full dense-plus-padded-expert closure differs");
    std::uint64_t left = kCap;
    const auto take = [&left](std::uint64_t bytes) {
      if (bytes > left) return false;
      left -= bytes;
      return true;
    };
    const auto fixed = node.catalog().OccupancyOf(node.domain()).Total().value();
    const auto pool = std::max<std::uint64_t>(model->pool_needed(), en::kPagedExtent);
    fixed_bytes = fixed;
    activation_bytes = model->activations_needed();
    pool_bytes = pool;
    input_bytes = model->host_input_bytes();
    plan_bytes = model->plan_floor_bytes();
    // Before physical workspace or weight admission: all current measured
    // storage plus page rounding, the complete weight closure, conservative
    // live/copy/spill state overlap, caller host staging and minimum plans.
    if (!take(fixed) || !take(weight_bytes) || !take(4 * SnapshotBytes()) ||
        !take(model->activations_needed()) || !take(pool) || !take(2 * en::kPagedExtent) ||
        !take(model->host_input_bytes()) || !take(model->plan_floor_bytes()) ||
        (scoring &&
         (!take(ScoreHost(score_size)) || !take(score_plan_bytes) || !take(score_extra_extent))))
      return std::unexpected("actual Setup storage exceeds the unchanged 110 GiB domain");
    footprint = kCap - left;
    const auto available = llmp::platform::AvailableMemoryBytes().value_or(0);
    // Conservative complete footprint check even though existing fixed
    // allocations already reduce MemAvailable. Unknown is never authority.
    if (footprint > available || rt::kUncountedMargin > available - footprint)
      return std::unexpected("actual Setup footprint plus six GiB lacks physical headroom");
    if (score_extra_extent != 0) {
      auto extra = en::ExecutionMetadataAllowance::Admit(
          node.catalog(), node.domain(), score_extra_extent, available, rt::kUncountedMargin);
      if (!extra) return std::unexpected(extra.error());
      score_plan_allowance = std::move(*extra);
      if (!score_plan_allowance->Charge(score_plan_bytes - ScoreCredit(score_size)))
        return std::unexpected("scored metadata excess lacks its admitted payment");
    }
    if (auto s = node.MapWorkspace(model->activations_needed(), pool); !s) return s;
    const auto mapped = node.catalog().OccupancyOf(node.domain()).Total().value();
    const auto now = llmp::platform::AvailableMemoryBytes().value_or(0);
    rt::MemoryGuard guard{
        .largest = weight_bytes,
        .host_inputs = model->host_input_bytes(),
        .plans = model->plan_floor_bytes(),
        .available = now,
        .fixed = mapped,
        .requests = kCaller + (scoring ? ScoreHost(score_size) + score_plan_bytes : 0)};
    if (auto s = rt::CheckMemoryGuard(guard); !s) return s;
    const auto reserve = rt::GuardReserve(guard);
    if (now == 0 || now < reserve || mapped > kCap)
      return std::unexpected("unknown or insufficient ordinary dynamic budget");
    dynamic = mapped + (now - reserve) / en::kPagedExtent * en::kPagedExtent;
    if (auto s = rt::CheckDiagnosticBudgetCap(guard, dynamic, kCap, 4 * SnapshotBytes()); !s)
      return s;
    if (auto s = node.Start(llmp::base::Bytes(kCap)); !s) return s;
    if (auto s = model->Register(); !s) return s;
    if (auto s = model->Bind(); !s) return s;
    node.Run();
    // All runtime plans count against the domain; construction still covers
    // the concurrently executing compiler's transient working set.
    node.SetHostFloor(0);
    auto& branch = model->default_branch();
    const std::array active{&branch};
    if (auto s = model->SelectBranches(active); !s) return s;
    if (auto s = model->PrepareDefaultRequest(); !s) return s;
    if (scoring) return Score();
    for (std::uint32_t i = 0; i < ids.size(); ++i)
      ids[i] = static_cast<std::int32_t>(100 + (i * 7919U + 17U) % 30000U);
    prefix = std::make_unique<rt::SnapshotStore>(node, SnapshotBytes());
    end = std::make_unique<rt::SnapshotStore>(node, SnapshotBytes());
    return node.WithRequest(
        0, model->request_closure(), "DeepSeek shared target and snapshot", [&]() -> Status {
          if (auto s = branch.Prefill(std::span(ids).first(kPrefix), head); !s) return s;
          if (auto s = Head(); !s) return s;
          auto saved = branch.CaptureState(*prefix);
          if (!saved) return std::unexpected(saved.error());
          rt::GenerateOptions options;
          options.max_tokens = kTail + 1;
          options.stop = false;
          options.keep_logits = true;
          const auto generate = [&](rt::Generation& generated) -> Status {
            if (!companion) return branch.Generate(head, options, generated);
            auto observed = options;
            auto previous = model->counts();
            std::size_t emitted = 0;
            std::string failure;
            // Existing synchronous per-completed-unit notification. A failed
            // draft may fall back to plain verification in serving; this
            // diagnostic requires every requested block to have contributed.
            observed.on_tokens = [&](std::span<const std::int32_t> fresh) {
              const auto current = model->counts();
              const std::uint64_t expected =
                  emitted == 0 ? 0 : std::min<std::uint64_t>(3, kTail + 1 - emitted - 1);
              const std::uint64_t units = emitted == 0 ? 0 : 1;
              if (fresh.empty() || emitted + fresh.size() > kTail + 1 ||
                  current.drafted - previous.drafted != expected ||
                  current.speculative_units - previous.speculative_units != units ||
                  current.discarded != previous.discarded) {
                failure = "a completed generation unit omitted its requested DSpark proposals";
                return false;
              }
              emitted += fresh.size();
              previous = current;
              return true;
            };
            auto result = branch.Generate(head, observed, generated);
            if (!failure.empty()) return std::unexpected(failure);
            return result;
          };
          if (auto s = generate(first); !s) return s;
          if (auto s = Complete(first); !s) return s;
          auto completed = branch.CaptureState(*end);
          if (!completed) return std::unexpected(completed.error());
          auto digest = end->Digest(*completed);
          if (!digest) return std::unexpected(digest.error());
          if (auto s = branch.RestoreState(*prefix, *saved); !s) return s;
          if (model->runner().slots().positions(0) != kPrefix || branch.history().size() != kPrefix)
            return std::unexpected("snapshot restore lost its actual prefix coordinate");
          if (auto s = generate(second); !s) return s;
          if (auto s = Complete(second); !s) return s;
          if (first.tokens != second.tokens || first.steps != second.steps ||
              first.drafted != second.drafted || first.accepted != second.accepted)
            return std::unexpected("same-implementation restored generation differs");
          for (std::size_t i = 0; i < first.logits.size(); ++i)
            if (std::memcmp(first.logits[i].data(), second.logits[i].data(),
                            kVocab * sizeof(float)) != 0)
              return std::unexpected("same-implementation restored full head differs");
          auto repeated = branch.CaptureState(*end);
          if (!repeated) return std::unexpected(repeated.error());
          auto again = end->Digest(*repeated);
          if (!again) return std::unexpected(again.error());
          const auto history = std::span(branch.history());
          if (*again != *digest || model->runner().slots().positions(0) != kPrefix + kTail ||
              history.size() != kPrefix + kTail ||
              !std::ranges::equal(history.first(kPrefix), ids) ||
              !std::ranges::equal(history.subspan(kPrefix), std::span(first.tokens).first(kTail)))
            return std::unexpected("same-implementation state or history differs after replay");
          restored_state_digest = llmp::base::ToHex(*again);
          return {};
        });
  }
  Status ScoreInputs() {
    const char* text_path = std::getenv("LLMP_DSV4_CORPUS_TEXT");
    const char* ids_path = std::getenv("LLMP_DSV4_CORPUS_IDS");
    const char* ids_hash = std::getenv("LLMP_DSV4_CORPUS_SHA256");
    if (!text_path || !ids_path || !ids_hash || std::string_view(ids_hash).size() != 64)
      return std::unexpected("captured DeepSeek corpus identity is required");
    if (auto checked = File(text_path, 17210,
                            "772a8b2d503df4fefd1a0a974ecae3ad58f8a4d1f2940fbc6bd440b5c86fb431");
        !checked)
      return checked;
    if (auto checked = File(ids_path, ScoreIds().size_bytes(), ids_hash); !checked) return checked;
    std::string text(17210, '\0');
    std::ifstream text_in(text_path, std::ios::binary);
    if (!text_in.read(text.data(), static_cast<std::streamsize>(text.size())))
      return std::unexpected("captured corpus read failed");
    std::ifstream ids_in(ids_path, std::ios::binary);
    if (!ids_in.read(reinterpret_cast<char*>(score_ids.data()),
                     static_cast<std::streamsize>(ScoreIds().size_bytes())))
      return std::unexpected("captured IDs read failed");
    if (model->tokenizer().size() != kVocab || !model->tokenizer().bos() ||
        *model->tokenizer().bos() != score_ids.front())
      return std::unexpected("native corpus vocabulary/BOS differs");
    auto native = model->EncodeText(text, &score_tokens);
    if (!native) return std::unexpected(native.error());
    if (native->size() < score_size ||
        !std::ranges::equal(std::span(*native).first(score_size), ScoreIds()))
      return std::unexpected("native tokenizer differs from captured reference prefix");
    return {};
  }
  Status Score() {
    auto composer = en::WaveComposer::For(score_caps, model->runner().slots());
    if (!composer) return std::unexpected(composer.error());
    score_composer.emplace(std::move(*composer));
    const std::array documents{ts::CorpusDocument{.slot = 0, .tokens = ScoreIds()}};
    auto bound =
        ts::CorpusScore::Bound(documents, {.vocabulary = kVocab, .chunk_rows = kScoreChunk});
    if (!bound || *bound != ScoreHost(score_size))
      return std::unexpected("scorer host payment differs");
    auto made = ts::CorpusScore::Make(
        node, model->runner(), *score_composer, documents,
        {.vocabulary = kVocab, .chunk_rows = kScoreChunk},
        [this](const ts::CorpusTargetScore& row) -> Status {
          if (score_count >= score_size - 1 || row.document || row.slot ||
              row.position != score_count + 1 || row.target != score_ids[row.position])
            return std::unexpected("DeepSeek loss coordinates differ");
          scores[score_count++] = row;
          return {};
        },
        [this](std::uint32_t document, std::uint32_t slot, std::uint32_t position,
               std::span<const float> row) -> Status {
          if (document || slot || position >= score_size || seen[position] || row.size() != kVocab)
            return std::unexpected("DeepSeek full finite row coverage differs");
          seen[position] = true;
          return {};
        });
    if (!made) return std::unexpected(made.error());
    scorer = std::move(*made);
    auto completed = node.WithRequest(
        0, model->request_closure(), "DeepSeek fixed natural corpus", [&]() -> Status {
          for (std::uint32_t wave = 0; wave < (score_size + kScoreChunk - 1) / kScoreChunk;
               ++wave) {
            auto done = scorer->Step();
            if (!done) return std::unexpected(done.error());
            if (*done != ((wave + 1) * kScoreChunk >= score_size))
              return std::unexpected("DeepSeek scored cutoff differs");
          }
          return {};
        });
    if (!completed) return completed;
    if (score_size == kLongScoreInputs && !std::ranges::any_of(score_classes, [](const auto& item) {
          return item.execute && item.lowered && item.key.shape.segments.front().past >= 2048 &&
                 item.indexers != 0 && item.index_cells > item.index_selected;
        }))
      return std::unexpected("long corpus did not execute a selective post-2048 indexer");
    if (!scorer->complete() || score_count != score_size - 1 ||
        scorer->result().inputs != score_size || scorer->result().targets != score_size - 1 ||
        scorer->result().steps != (score_size + kScoreChunk - 1) / kScoreChunk ||
        model->runner().slots().positions(0) != score_size ||
        !std::ranges::all_of(std::span(seen).first(score_size), [](bool value) { return value; }))
      return std::unexpected("DeepSeek score completion differs");
    return {};
  }
  Status Complete(const rt::Generation& generated) const {
    if (generated.tokens.size() != kTail + 1 || generated.logits.size() != kTail + 1 ||
        (companion ? (generated.steps == 0 || generated.steps > kTail || generated.drafted == 0)
                   : generated.steps != kTail) ||
        generated.stopped || generated.cancelled ||
        !std::ranges::all_of(generated.tokens, [](std::int32_t id) {
          return id >= 0 && id < static_cast<std::int32_t>(kVocab);
        }))
      return std::unexpected(
          "the actual generation did not complete its bounded decode or draft/verify schedule");
    for (const auto& row : generated.logits)
      if (row.size() != kVocab ||
          !std::ranges::all_of(row, [](float x) { return std::isfinite(x); }))
        return std::unexpected("a complete generated head is absent or nonfinite");
    return {};
  }
  Status Head() const {
    if (head.size() != kVocab ||
        !std::ranges::all_of(head, [](float x) { return std::isfinite(x); }))
      return std::unexpected("the actual full vocabulary head is absent or nonfinite");
    return {};
  }
  Status Retire() {
    if (teardown_attempted) return std::unexpected("retirement was already attempted");
    teardown_attempted = true;
    // Successful copies may release before teardown; unproven stores retain
    // their pinned allocations in the complete node owner.
    prefix.reset();
    end.reset();
    return node.TearDown(entered);
  }
};

TEST(Dsv4EngineTargetInputs, ServingDomainUsesActualBlockStateAndOwnsItsPayment) {
  namespace md = llmp::model;
  using C = md::StateTensorContract;
  llmp::spec::ModelSpec spec;
  spec.pipeline = llmp::spec::PipelineKind::kLlm;
  spec.stages.resize(2);
  spec.stages[0].name = "target";
  spec.stages[0].role = llmp::spec::StageRole::kTarget;
  spec.stages[1].name = "block";
  spec.stages[1].role = llmp::spec::StageRole::kDrafter;
  const std::vector<std::pair<std::uint32_t, md::StateContract>> target = {
      {0, {.tensors = {C{.name = "kv", .cell_bytes = 256}}}}};
  const std::vector<std::pair<std::uint32_t, md::StateContract>> block = {
      {0,
       {.tensors = {C{.name = "ring",
                      .kind = md::StateKind::kDrafterRing,
                      .cell_bytes = 256,
                      .cells = md::CellRule::kPositionRing,
                      .window_or_ratio = 256,
                      .rollback = md::Rollback::kSnapshot,
                      .target_writes = true}}}}};
  auto a = md::Layout(target, 8192, 512);
  auto b = md::Layout(block, 8192, 512);
  ASSERT_TRUE(a) << Failed(a).value_or("");
  ASSERT_TRUE(b) << Failed(b).value_or("");
  const std::array layouts{*a, *b};
  en::DecodingMode mode{
      .kind = en::DecodingMode::Kind::kSpeculative,
      .drafters = {{.stage = 1, .shape = en::DrafterSpec::Shape::kBlock, .max_depth = 3}},
      .combine = en::DecodingMode::Drafters::kCascade,
      .row_invariant = false};
  llmp::catalog::Catalog catalog;
  auto allowance =
      en::ExecutionMetadataAllowance::Admit(catalog, catalog.AddDomain("serving domain"), 2 * kMiB,
                                            2 * kMiB + rt::kUncountedMargin, rt::kUncountedMargin);
  ASSERT_TRUE(allowance) << Failed(allowance).value_or("");
  for (const auto owners : {1U, 4U}) {
    auto domain = rt::ComponentServingDomain(spec, layouts, owners, mode, *allowance);
    ASSERT_TRUE(domain) << Failed(domain).value_or("");
    EXPECT_GT((*allowance)->used(), 0U);
    EXPECT_TRUE((*domain)->CheckMode(en::NumericalUse::kPrompt));
    EXPECT_TRUE((*domain)->CheckMode(en::NumericalUse::kSpeculative));
    EXPECT_TRUE((*domain)->CheckMode(en::NumericalUse::kPromptCompanions));
    EXPECT_FALSE((*domain)->CheckMode(en::NumericalUse::kJoinedRepeatPending));
    EXPECT_EQ((*domain)->stages()[0].phases[2].max_owners, owners);
    EXPECT_EQ((*domain)->stages()[0].phases[2].max_owner_rows, 4U);
  }
  EXPECT_EQ((*allowance)->used(), 0U);
  auto invalid = mode;
  invalid.drafters[0].stage = 0;
  EXPECT_FALSE(rt::ComponentServingDomain(spec, layouts, 4, invalid, *allowance));
  invalid = mode;
  invalid.drafters[0].max_depth = 512;
  EXPECT_FALSE(rt::ComponentServingDomain(spec, layouts, 4, invalid, *allowance));
  EXPECT_FALSE(rt::ComponentServingDomain(spec, layouts, 4, mode, nullptr));
  EXPECT_EQ((*allowance)->used(), 0U);
  const std::vector<std::pair<std::uint32_t, md::StateContract>> replay = {
      {0,
       {.tensors = {C{.name = "kv", .cell_bytes = 256, .replayable = true},
                    C{.name = "tap",
                      .kind = md::StateKind::kFeatures,
                      .encoding = md::Encoding::kF32,
                      .cell_bytes = 256,
                      .cells = md::CellRule::kPerRow,
                      .window_or_ratio = 1,
                      .rollback = md::Rollback::kSnapshot,
                      .target_writes = true},
                    md::DraftProgressControl("progress", md::DraftRelation::kTokenAfterCache)}}}};
  auto replay_layout = md::Layout(replay, 8192, 512);
  ASSERT_TRUE(replay_layout) << Failed(replay_layout).value_or("");
  const std::array replay_layouts{*a, *replay_layout};
  EXPECT_FALSE(rt::ComponentServingDomain(spec, replay_layouts, 4, mode, *allowance));
  auto block_shared = mode;
  block_shared.drafters[0].shared_wave_depth = 2;
  EXPECT_FALSE(rt::ComponentServingDomain(spec, layouts, 4, block_shared, *allowance));
  auto autoregressive = mode;
  autoregressive.drafters[0].shape = en::DrafterSpec::Shape::kAutoregressive;
  autoregressive.drafters[0].catch_up = true;
  autoregressive.drafters[0].features = true;
  autoregressive.drafters[0].shared_wave_depth = 2;
  {
    auto joined = rt::ComponentServingDomain(spec, replay_layouts, 4, autoregressive, *allowance);
    ASSERT_TRUE(joined) << Failed(joined).value_or("");
    EXPECT_TRUE((*joined)->CheckMode(en::NumericalUse::kJoinedRepeatPending));
  }
  autoregressive.drafters[0].shared_wave_depth = 4;
  EXPECT_FALSE(rt::ComponentServingDomain(spec, replay_layouts, 4, autoregressive, *allowance));
  EXPECT_EQ((*allowance)->used(), 0U);
  spec.stages.resize(1);
  auto plain = rt::ComponentServingDomain(spec, std::span(layouts).first(1), 4, {}, *allowance);
  ASSERT_TRUE(plain) << Failed(plain).value_or("");
  EXPECT_FALSE((*plain)->CheckMode(en::NumericalUse::kSpeculative));
}

TEST(Dsv4EngineTargetInputs, AutomaticServingRefusesUnpaidOrMixedAuthorityBeforeSourceRead) {
  en::PagedNode node{{.slot_bytes = en::kSlabSlotBytes}};
  llmp::config::ModelEntry entry;
  entry.name = Settings().name;
  entry.artifact = std::string(kArtifact);
  auto domain = Domain();
  ASSERT_TRUE(domain) << Failed(domain).value_or("");
  for (unsigned mutation = 0; mutation < 5; ++mutation) {
    auto settings = Settings();
    settings.context.value = 8192;
    settings.max_slots.value = 4;
    rt::ComponentBuildOptions options{.target_policy = std::nullopt,
                                      .diagnostic_solo = true,
                                      .automatic_startup_cap = kCap,
                                      .derive_execution_domain = true};
    if (mutation == 0) options.automatic_startup_cap = 0;
    if (mutation == 1) options.execution_domain = *domain;
    if (mutation == 2) options.diagnostic_solo = false;
    if (mutation == 3) options.target_policy = "";
    if (mutation == 4) settings.speculation.value = true;  // no drafter
    auto build = rt::ComponentLlmBuild(node, "/nonexistent-dsv4-serving-control", entry, 0,
                                       std::move(options));
    auto parts = build(settings);
    ASSERT_FALSE(parts);
    EXPECT_TRUE(parts.error().find("diagnostic envelope") != std::string::npos ||
                parts.error().find("internal request") != std::string::npos);
  }
}

TEST(Dsv4EngineTargetInputs, RejectsUnpaidOrDifferentEnvelopeBeforeArtifactRead) {
  // Served on the component engine since c96deafd (K1); a diagnostic build
  // still needs its paid single-owner envelope.
  EXPECT_TRUE(rt::OnComponentEngine("deepseek4"));
  en::PagedNode node{{.slot_bytes = en::kSlabSlotBytes}};
  auto domain = Domain();
  ASSERT_TRUE(domain) << Failed(domain).value_or("");
  llmp::config::ModelEntry entry;
  entry.name = Settings().name;
  entry.artifact = std::string(kArtifact);
  llmp::catalog::Catalog catalog;
  const auto owner = catalog.AddDomain("closed diagnostic test payment");
  auto admitted = en::ExecutionMetadataAllowance::Admit(catalog, owner, kConstructionExtent,
                                                        kConstructionExtent + rt::kUncountedMargin,
                                                        rt::kUncountedMargin);
  ASSERT_TRUE(admitted) << Failed(admitted).value_or("");
  for (unsigned mutation = 0; mutation < 7; ++mutation) {
    auto settings = Settings();
    if (mutation == 1) settings.context.value = kContext + 1;
    if (mutation == 2) settings.prefill_chunk.value = kChunk / 2;
    if (mutation == 3) settings.max_slots.value = 2;
    if (mutation == 4) settings.speculation.value = true;
    rt::ComponentBuildOptions options{.target_policy = std::nullopt,
                                      .diagnostic_solo = true,
                                      .construction_metadata = mutation == 0 ? nullptr : *admitted,
                                      .execution_domain = *domain};
    if (mutation == 5) options.target_policy = "";
    if (mutation == 6) options.execution_domain.reset();
    auto build = rt::ComponentLlmBuild(node, "/nonexistent-dsv4-target-control", entry, 0,
                                       std::move(options));
    auto parts = build(settings);
    ASSERT_FALSE(parts);
    EXPECT_NE(parts.error().find("paid single-owner diagnostic envelope"), std::string::npos);
    EXPECT_EQ((*admitted)->used(), 0U);
  }
  // Capacity refusal occurs before OpenInstalled, and a later ordinary file
  // refusal refunds the complete construction borrow exactly once.
  ASSERT_TRUE((*admitted)->Charge(4 * kMiB));
  auto build = rt::ComponentLlmBuild(node, "/nonexistent-dsv4-target-control", entry, 0,
                                     {.target_policy = std::nullopt,
                                      .diagnostic_solo = true,
                                      .construction_metadata = *admitted,
                                      .execution_domain = *domain});
  auto unpaid = build(Settings());
  ASSERT_FALSE(unpaid);
  EXPECT_NE(unpaid.error().find("admitted metadata allowance"), std::string::npos);
  EXPECT_EQ((*admitted)->used(), 4 * kMiB);
  (*admitted)->Release(4 * kMiB);
  auto absent = build(Settings());
  ASSERT_FALSE(absent);
  EXPECT_EQ((*admitted)->used(), 0U);
}

TEST(Dsv4EngineTargetInputs, LongCorpusUsesActualComposerAndTerminalCoordinate) {
  const llmp::model::Batching caps{.max_segment_rows = kScoreChunk};
  auto shapes = ScoreSchedule(caps, kContext, 256, kLongScoreInputs);
  ASSERT_TRUE(shapes) << Failed(shapes).value_or("");
  ASSERT_EQ(shapes->size(), 21U);
  for (std::uint32_t i = 0; i < shapes->size(); ++i) {
    const auto& shape = shapes->at(i);
    ASSERT_EQ(shape.segments.size(), 1U);
    const auto& segment = shape.segments.front();
    EXPECT_EQ(shape.phase, pl::Phase::kScore);
    EXPECT_EQ(segment.past, i * kScoreChunk);
    EXPECT_EQ(segment.rows, i == 20 ? 1U : kScoreChunk);
    EXPECT_EQ(segment.outputs, segment.rows);
    EXPECT_EQ(segment.read_cells, (segment.past + segment.rows + 255) / 256 * 256);
    EXPECT_TRUE(segment.want_head);
    EXPECT_FALSE(segment.greedy);
  }
  EXPECT_EQ(shapes->back().segments.front().read_cells, 2816U);
  EXPECT_FALSE(ScoreSchedule(caps, kContext, 256, kContext + 1));
  EXPECT_FALSE(ScoreSchedule({.max_segment_rows = 1}, kContext, 256, kLongScoreInputs));
  EXPECT_EQ(ScoreHost(kLongScoreInputs) - ScoreHost(kScoreInputs), 16U * 2304U);
}

TEST(Dsv4EngineTargetInputs, ScoresPayUniqueCompleteKeysAndOneReplacement) {
  auto shapes = ScoreSchedule({}, kContext, 256, kScoreInputs);
  ASSERT_TRUE(shapes) << Failed(shapes).value_or("");
  std::array<ScoreClass, 2> classes{};
  for (std::size_t i = 0; i < classes.size(); ++i) {
    classes[i].key.shape = shapes->at(i);
    classes[i].execute = classes[i].measured = true;
    classes[i].selection = "same-selection";
    classes[i].selected_steps = 3;
  }
  // The real key ignores runtime past. Independent tuple maxima still pay
  // a complete owner: a larger host observation and larger launch observation.
  classes[0].host_bytes = 100;
  classes[0].launches = 2;
  classes[1].host_bytes = 200;
  classes[1].launches = 1;
  ASSERT_EQ(classes[0].key, classes[1].key);
  auto paid = ScorePlanPayment(classes);
  ASSERT_TRUE(paid) << Failed(paid).value_or("");
  EXPECT_EQ(*paid, 2 * (200 + 2 * en::kGraphNodeHostBytes));
  classes[1].key.parameters = {1};
  paid = ScorePlanPayment(classes);
  ASSERT_TRUE(paid) << Failed(paid).value_or("");
  EXPECT_EQ(*paid, 2 * (100 + 2 * en::kGraphNodeHostBytes) + 200 + en::kGraphNodeHostBytes);
  classes[1].key.parameters.clear();
  classes[1].selection = "different-selection";
  EXPECT_FALSE(ScorePlanPayment(classes));
  classes[1].selection = classes[0].selection;
  classes[1].launches = kCap / en::kGraphNodeHostBytes + 1;
  EXPECT_FALSE(ScorePlanPayment(classes));
  classes[1].launches = 1;
  classes[1].measured = false;
  EXPECT_FALSE(ScorePlanPayment(classes));
}

TEST(Dsv4EngineTargetInputs, CreditsOnlyTheRetainedFactoryConstructionCharge) {
  llmp::catalog::Catalog catalog;
  const auto owner = catalog.AddDomain("held construction credit");
  auto paid = en::ExecutionMetadataAllowance::Admit(catalog, owner, kConstructionExtent,
                                                    kConstructionExtent + rt::kUncountedMargin,
                                                    rt::kUncountedMargin);
  ASSERT_TRUE(paid) << Failed(paid).value_or("");
  auto& allowance = **paid;
  EXPECT_FALSE(HeldScoreCredit(allowance, ScoreCredit(kScoreInputs)));
  // The real factory charges this whole grant before source construction and
  // retains it in ComponentHold through source/backend teardown.
  ASSERT_TRUE(allowance.Charge(kConstruction));
  EXPECT_EQ(allowance.capacity() - allowance.used(),
            2 * kMiB - en::ExecutionMetadataAllowance::OverheadBytes());
  EXPECT_FALSE(allowance.Charge(ScoreCredit(kScoreInputs)));  // Reproduces the old double charge.
  const auto used = allowance.used();
  for (const auto plans :
       {ScoreCredit(kScoreInputs) - 1, ScoreCredit(kScoreInputs), ScoreCredit(kScoreInputs) + 1}) {
    const auto credit = HeldScoreCredit(allowance, plans);
    ASSERT_TRUE(credit) << Failed(credit).value_or("");
    EXPECT_EQ(*credit, std::min(plans, ScoreCredit(kScoreInputs)));
    EXPECT_LE(ScoreFixed(kScoreInputs) + *credit, used);
    EXPECT_EQ(plans - *credit, plans > ScoreCredit(kScoreInputs) ? 1U : 0U);
    EXPECT_EQ(allowance.used(), used);
  }
  ASSERT_TRUE(allowance.Charge(1));
  EXPECT_FALSE(HeldScoreCredit(allowance, ScoreCredit(kScoreInputs)));
  allowance.Release(1);
  allowance.Release(kConstruction);
  EXPECT_FALSE(HeldScoreCredit(allowance, ScoreCredit(kScoreInputs)));
}

TEST(Dsv4EngineTargetInputs, ScoredTargetEnvelopeIsClosedBeforeArtifactRead) {
  en::PagedNode node{{.slot_bytes = en::kSlabSlotBytes}};
  llmp::catalog::Catalog catalog;
  const auto owner = catalog.AddDomain("closed scored target payment");
  auto paid = en::ExecutionMetadataAllowance::Admit(catalog, owner, kConstructionExtent,
                                                    kConstructionExtent + rt::kUncountedMargin,
                                                    rt::kUncountedMargin);
  ASSERT_TRUE(paid) << Failed(paid).value_or("");
  for (unsigned mutation = 0; mutation < 8; ++mutation) {
    auto scored = Domain(false, true);
    ASSERT_TRUE(scored) << Failed(scored).value_or("");
    std::vector<en::NumericalStage> stages((*scored)->stages().begin(), (*scored)->stages().end());
    if (mutation == 1) stages[0].phases.back().max_owner_rows = 129;
    if (mutation == 2) stages[0].phases.back().max_total_rows = 129;
    if (mutation == 3) stages[0].phases.back().max_owners = 2;
    if (mutation == 4) stages[0].phases.back().row_invariant = true;
    if (mutation == 5) stages[0].uses.pop_back();
    if (mutation == 6) stages[0].uses.push_back(en::NumericalUse::kPromptCompanions);
    auto domain = en::NumericalDomain::Create(std::move(stages));
    ASSERT_TRUE(domain) << Failed(domain).value_or("");
    llmp::config::ModelEntry entry;
    entry.name = Settings().name;
    entry.artifact = std::string(kArtifact);
    if (mutation == 7) entry.drafter = std::string(kDsparkArtifact);
    auto build = rt::ComponentLlmBuild(
        node, "/nonexistent-dsv4-score-control", entry, 0,
        {.target_policy = std::nullopt,
         .diagnostic_solo = true,
         .construction_metadata = *paid,
         .execution_domain = std::make_shared<const en::NumericalDomain>(std::move(*domain))});
    auto result = build(Settings());
    ASSERT_FALSE(result);
    const bool envelope =
        result.error().find("paid single-owner diagnostic envelope") != std::string::npos;
    EXPECT_EQ(envelope, mutation != 0) << result.error();
    EXPECT_EQ((*paid)->used(), 0U);
  }
}

TEST(Dsv4EngineTargetInputs, RejectsUnpaidOrDifferentCompanionBeforeArtifactRead) {
  en::PagedNode node{{.slot_bytes = en::kSlabSlotBytes}};
  auto domain = Domain(true);
  ASSERT_TRUE(domain) << Failed(domain).value_or("");
  llmp::catalog::Catalog catalog;
  const auto owner = catalog.AddDomain("closed DSpark diagnostic payment");
  auto admitted = en::ExecutionMetadataAllowance::Admit(catalog, owner, kConstructionExtent,
                                                        kConstructionExtent + rt::kUncountedMargin,
                                                        rt::kUncountedMargin);
  ASSERT_TRUE(admitted) << Failed(admitted).value_or("");
  for (unsigned mutation = 0; mutation < 7; ++mutation) {
    auto settings = Settings(true);
    llmp::config::ModelEntry entry;
    entry.name = settings.name;
    entry.artifact = std::string(kArtifact);
    entry.drafter = std::string(kDsparkArtifact);
    rt::ComponentBuildOptions options{.target_policy = std::nullopt,
                                      .diagnostic_solo = true,
                                      .construction_metadata = mutation == 0 ? nullptr : *admitted,
                                      .execution_domain = *domain};
    if (mutation == 1) settings.draft_rows.value = 2;
    if (mutation == 2) settings.speculation.value = false;
    if (mutation == 3) settings.max_slots.value = 2;
    if (mutation == 4) entry.drafter = std::string(kArtifact);
    if (mutation == 5) options.target_policy = "";
    if (mutation == 6) {
      auto target_only = Domain();
      ASSERT_TRUE(target_only) << Failed(target_only).value_or("");
      options.execution_domain = *target_only;
    }
    auto build =
        rt::ComponentLlmBuild(node, "/nonexistent-dspark-control", entry, 0, std::move(options));
    auto parts = build(settings);
    ASSERT_FALSE(parts);
    EXPECT_NE(parts.error().find("paid single-owner diagnostic envelope"), std::string::npos);
    EXPECT_EQ((*admitted)->used(), 0U);
  }
}

TEST(Dsv4EngineTargetGpu, FullTargetFiniteHeadsAndSnapshotContinuation) {
  if (std::getenv("LLMP_RUN_DSV4_TARGET_BRIDGE") == nullptr)
    GTEST_SKIP() << "explicit bounded target diagnostic required";
  const char* store = std::getenv("LLMP_DSV4_TARGET_STORE");
  const char* scratch = std::getenv("LLMP_TEST_SCRATCH");
  ASSERT_NE(store, nullptr);
  ASSERT_NE(scratch, nullptr);
  const fs::path out = fs::path(scratch) / "dsv4-engine-target";
  std::error_code ec;
  fs::create_directories(out, ec);
  ASSERT_FALSE(ec) << ec.message();
  auto life = std::make_unique<Lifetime>();
  const auto result = life->Run(store, out);
  const auto retired = life->Retire();
  if (!retired) {
    // Do not retry TearDown: its internal once flag is not a completion
    // certificate. Retain node, backend, borrowed callbacks and all state.
    std::ignore = life.release();
    FAIL() << Failed(retired).value_or("");
  }
  ASSERT_TRUE(result) << Failed(result).value_or("");
  EXPECT_GT(life->full_programs, 0U);
  RecordProperty("validated_full_target_programs", std::to_string(life->programs));
  RecordProperty("full_head_programs", std::to_string(life->full_programs));
  RecordProperty("source_artifact", std::string(kArtifact));
  RecordProperty("source_weight_extent_bytes", std::to_string(kWeights));
  RecordProperty("state_layout_bytes", std::to_string(kState));
  RecordProperty("setup_paid_footprint_bytes", std::to_string(life->footprint));
  RecordProperty("ordinary_dynamic_budget_bytes", std::to_string(life->dynamic));
  RecordProperty("restored_full_head_values", std::to_string(std::uint64_t{kTail + 1} * kVocab));
  RecordProperty("completed_frontier", std::to_string(kPrefix + kTail));
  RecordProperty("restored_initialized_state_sha256", life->restored_state_digest);
  RecordProperty("selected_plan_receipt", llmp::base::ToHex(life->recipes.Finish()));
}
TEST(Dsv4EngineTargetGpu, TargetAndDsparkFiniteHeadsAndSnapshotContinuation) {
  if (std::getenv("LLMP_RUN_DSPARK_SHARED_BRIDGE") == nullptr)
    GTEST_SKIP() << "explicit bounded target plus DSpark diagnostic required";
  const char* store = std::getenv("LLMP_DSV4_TARGET_STORE");
  const char* scratch = std::getenv("LLMP_TEST_SCRATCH");
  ASSERT_NE(store, nullptr);
  ASSERT_NE(scratch, nullptr);
  const fs::path out = fs::path(scratch) / "dsv4-engine-dspark";
  std::error_code ec;
  fs::create_directories(out, ec);
  ASSERT_FALSE(ec) << ec.message();
  auto life = std::make_unique<Lifetime>(true);
  const auto result = life->Run(store, out);
  const auto retired = life->Retire();
  if (!retired) {
    std::ignore = life.release();
    FAIL() << Failed(retired).value_or("");
  }
  ASSERT_TRUE(result) << Failed(result).value_or("");
  EXPECT_GT(life->draft_programs, 0U);
  EXPECT_GT(life->injected_programs, 0U);
  RecordProperty("target_artifact", std::string(kArtifact));
  RecordProperty("drafter_artifact", std::string(kDsparkArtifact));
  RecordProperty("setup_fixed_bytes", std::to_string(life->fixed_bytes));
  RecordProperty("setup_activation_bytes", std::to_string(life->activation_bytes));
  RecordProperty("setup_pool_bytes", std::to_string(life->pool_bytes));
  RecordProperty("setup_host_input_bytes", std::to_string(life->input_bytes));
  RecordProperty("setup_plan_floor_bytes", std::to_string(life->plan_bytes));
  RecordProperty("source_weight_extent_bytes", std::to_string(life->WeightBytes()));
  RecordProperty("state_layout_bytes", std::to_string(life->StateBytes()));
  RecordProperty("setup_paid_footprint_bytes", std::to_string(life->footprint));
  RecordProperty("ordinary_dynamic_budget_bytes", std::to_string(life->dynamic));
  RecordProperty("draft_programs", std::to_string(life->draft_programs));
  RecordProperty("injected_target_programs", std::to_string(life->injected_programs));
  RecordProperty("generation_steps", std::to_string(life->first.steps));
  RecordProperty("generation_proposed", std::to_string(life->first.drafted));
  RecordProperty("generation_accepted", std::to_string(life->first.accepted));
  RecordProperty("restored_full_head_values", std::to_string(std::uint64_t{kTail + 1} * kVocab));
  RecordProperty("completed_frontier", std::to_string(kPrefix + kTail));
  RecordProperty("restored_initialized_state_sha256", life->restored_state_digest);
  RecordProperty("selected_plan_receipt", llmp::base::ToHex(life->recipes.Finish()));
}
// Deliberately registered dynamically: this captured-input diagnostic is
// built in the existing owner fixture but has no automatic CTest entry.
class Dsv4NaturalCorpus : public testing::Test {
 public:
  explicit Dsv4NaturalCorpus(std::uint32_t inputs = kScoreInputs) : inputs_(inputs) {}
  void TestBody() override {
    const char* store = std::getenv("LLMP_DSV4_TARGET_STORE");
    const char* scratch = std::getenv("LLMP_TEST_SCRATCH");
    ASSERT_NE(store, nullptr);
    ASSERT_NE(scratch, nullptr);
    const fs::path out = fs::path(scratch) / "dsv4-corpus";
    std::error_code error;
    fs::create_directories(out, error);
    ASSERT_FALSE(error) << error.message();
    auto life = std::make_unique<Lifetime>(false, true, inputs_);
    const auto result = life->Run(store, out);
    const auto retired = life->Retire();
    if (!retired) {
      std::ignore = life.release();
      FAIL() << Failed(retired).value_or("");
    }
    ASSERT_TRUE(result) << Failed(result).value_or("");
    std::ostringstream losses, classes;
    losses << std::setprecision(17);
    for (const auto& row : std::span(life->scores).first(life->score_count))
      losses << row.document << ',' << row.position << ',' << row.target << ',' << row.nll << '\n';
    for (std::size_t i = 0; i < life->score_classes.size(); ++i) {
      const auto& item = life->score_classes[i];
      const auto& shape = item.key.shape;
      const auto& segment = shape.segments.front();
      classes << i << ',' << static_cast<unsigned>(shape.phase) << ',' << shape.stage << ','
              << shape.row_invariant << ',' << shape.read_align << ','
              << static_cast<unsigned>(segment.slot) << ',' << segment.rows << ',' << segment.past
              << ',' << segment.read_cells << ',' << segment.outputs << ','
              << static_cast<unsigned>(segment.group) << ',' << segment.want_head << ','
              << segment.greedy << ',' << item.measured << ',' << item.lowered << ','
              << item.host_bytes << ',' << item.launches << ',' << item.selected_steps << ','
              << item.paid_bytes << ',' << item.execute << "|parameters:";
      for (auto value : item.key.parameters) classes << value << ',';
      classes << "|selection_sha256:" << item.selection << "|indexer:" << item.indexers << ','
              << item.index_cells << ',' << item.index_selected << '\n';
    }
    ASSERT_LE(losses.tellp(), static_cast<std::streamoff>(kMiB));
    ASSERT_LE(classes.tellp(), static_cast<std::streamoff>(kClassReceipt));
    RecordProperty("losses", losses.str());
    RecordProperty("classes", classes.str());
    RecordProperty("plan_key_spec", llmp::base::ToHex(life->score_base.spec));
    RecordProperty("plan_key_patterns", llmp::base::ToHex(life->score_base.patterns));
    RecordProperty("plan_key_policy", llmp::base::ToHex(life->score_base.policy));
    RecordProperty("plan_key_partition", std::to_string(life->score_base.partition));
    RecordProperty("inputs", std::to_string(life->score_size));
    RecordProperty("targets", std::to_string(life->score_count));
    RecordProperty("finite_rows", std::to_string(std::ranges::count(
                                      std::span(life->seen).first(life->score_size), true)));
    RecordProperty("waves", std::to_string(life->scorer->result().steps));
    RecordProperty("scorer_host_bound", std::to_string(ScoreHost(life->score_size)));
    RecordProperty("retained_plan_bound_bytes", std::to_string(life->score_plan_bytes));
    RecordProperty("plan_metadata_extra_extent_bytes", std::to_string(life->score_extra_extent));
    RecordProperty("plan_metadata_overhead_bytes",
                   std::to_string(en::ExecutionMetadataAllowance::OverheadBytes()));
    RecordProperty("source_artifact", std::string(kArtifact));
    RecordProperty("source_weight_extent_bytes", std::to_string(kWeights));
    RecordProperty("state_layout_bytes", std::to_string(kState));
    RecordProperty("setup_paid_footprint_bytes", std::to_string(life->footprint));
    RecordProperty("setup_activation_bytes", std::to_string(life->activation_bytes));
    RecordProperty("setup_pool_bytes", std::to_string(life->pool_bytes));
    RecordProperty("setup_host_input_bytes", std::to_string(life->input_bytes));
    RecordProperty("setup_plan_floor_bytes", std::to_string(life->plan_bytes));
    RecordProperty("selected_plan_receipt", llmp::base::ToHex(life->recipes.Finish()));
  }

 private:
  std::uint32_t inputs_;
};
[[maybe_unused]] const auto* const kNaturalCorpus =
    testing::RegisterTest("Dsv4CorpusScoreManual", "Natural257", nullptr, nullptr, __FILE__,
                          __LINE__, []() -> Dsv4NaturalCorpus* { return new Dsv4NaturalCorpus; });
[[maybe_unused]] const auto* const kLongNaturalCorpus = testing::RegisterTest(
    "Dsv4CorpusScoreManual", "Natural2561", nullptr, nullptr, __FILE__, __LINE__,
    []() -> Dsv4NaturalCorpus* { return new Dsv4NaturalCorpus(kLongScoreInputs); });
namespace native_target {
// Reuse the complete initialized-state reader and stopped-owner payment
// handoff from the Qwen native publication proof, over this family's Server.
namespace cfg = llmp::config;
constexpr std::uint64_t kObservations = 128 * kMiB;
// Both arms: 36 full rows, growth capacity, six coordinate inventories,
// histories and the copy window. The catalog charge precedes every result.
static_assert(3 * 36ULL * kVocab * sizeof(float) + 6 * 4096 * 24 + 4 * kMiB < kObservations);
struct StateImage {
  std::vector<std::array<std::uint64_t, 3>> ranges;
  llmp::base::Sha256Digest digest{};
  std::uint32_t positions = 0;
  bool operator==(const StateImage&) const = default;
};
struct Result {
  std::uint64_t selected_budget = 0;
  std::array<StateImage, 3> states;
  std::array<std::vector<float>, 2> heads;
  std::array<rt::Generation, 2> generations;
  std::array<std::vector<std::int32_t>, 2> history;
};
struct Owner {
  cfg::NodeConfig config;
  cfg::RuntimeRoles roles;
  rt::ServingOptions options;
  std::unique_ptr<rt::Server> server;
  std::shared_ptr<en::ExecutionMetadataAllowance> observations;
  std::unique_ptr<rt::Llm::PromptSession> restoration;
  Result result;
  fs::path scratch;
  std::FILE* log = nullptr;
  ~Owner() {
    if (log) std::fclose(log);
  }
};
std::expected<StateImage, std::string> ReadState(Owner& life, rt::EngineLlm& model) {
  StateImage state;
  auto& slots = model.runner().slots();
  auto& node = life.server->node();
  state.positions = slots.positions(0);
  for (std::size_t stage = 0; stage < slots.stages(); ++stage)
    for (const auto& range : slots.live(0, stage).used_ranges()) {
      if (state.ranges.size() == 4096)
        return std::unexpected("state observation exceeds prepaid coordinates");
      state.ranges.push_back({stage, range.offset, range.bytes});
    }
  if (state.ranges.empty()) return std::unexpected("state observation is empty");
  constexpr std::uint64_t part_bytes = 1ULL << 20U;
  std::vector<llmp::catalog::ExtentId> extents;
  auto buffer = node.Pinned(part_bytes, 0, extents);
  if (!buffer) return std::unexpected(buffer.error());
  llmp::base::Sha256 hash;
  for (const auto& range : state.ranges)
    for (std::uint64_t at = 0; at < range[2]; at += part_bytes) {
      const en::LiveState::Range part{.region = range[0],
                                      .offset = range[1] + at,
                                      .bytes = std::min(part_bytes, range[2] - at)};
      auto retired = en::LiveState::CopyRetirement::kUnproven;
      auto copied = slots.CopyState(0, *buffer, std::span(&part, 1), true, &retired);
      if (retired == en::LiveState::CopyRetirement::kUnproven) {
        node.KeepPinned(*buffer);
        return std::unexpected("state copy retirement is unproven");
      }
      if (!copied) {
        (void)node.FreePinned(*buffer);
        return std::unexpected(copied.error());
      }
      hash.Update(std::span(static_cast<const std::byte*>(*buffer), part.bytes));
    }
  if (auto freed = node.FreePinned(*buffer); !freed) return std::unexpected(freed.error());
  state.digest = hash.Finish();
  return state;
}

std::expected<std::unique_ptr<Owner>, std::string> Run(const fs::path& store, std::string_view id,
                                                       Owner* previous = nullptr,
                                                       std::string_view drafter = {}) {
  const auto* scratch = std::getenv("LLMP_TEST_SCRATCH");
  if (!scratch) return std::unexpected("missing publication scratch");
  auto life = std::make_unique<Owner>();
  std::error_code ec;
  fs::create_directories(scratch, ec);
  if (ec) return std::unexpected(ec.message());
  auto path = (fs::path(scratch) / "ds-native-XXXXXX").string();
  if (!mkdtemp(path.data())) return std::unexpected("creating native target scratch");
  life->scratch = path;
  life->roles.installed = store;
  life->roles.spill = life->scratch / "spill";
  life->roles.state = life->scratch / "state";
  for (const auto& dir : {life->roles.spill, life->roles.state}) {
    fs::create_directory(dir, ec);
    if (ec || chmod(dir.c_str(), 0700)) return std::unexpected("protecting native target state");
  }
  cfg::ModelEntry entry;
  entry.name = "deepseek";
  entry.artifact = id;
  if (!drafter.empty()) entry.drafter = std::string(drafter);
  entry.overrides["context"] = std::int64_t{kContext};
  entry.overrides["prefill_chunk"] = std::int64_t{kChunk};
  entry.overrides["max_slots"] = std::int64_t{1};
  life->config.models.push_back(std::move(entry));
  life->options.diagnostic_auto_startup = true;
  // DSpark's weights and state need the node's full 112 GiB (the speed gate's cap).
  life->options.diagnostic_budget_cap_bytes = drafter.empty() ? kCap : 120259084288ULL;
  life->log = std::tmpfile();
  if (!life->log) return std::unexpected("opening native target log");
  life->server = std::make_unique<rt::Server>(life->config, life->roles, life->options, life->log);
  auto completed = [&]() -> Status {
    if (auto started = life->server->Start(true); !started) return started;
    auto& node = life->server->node();
    life->result.selected_budget = node.budget().value();
    if (auto paid = node.Call(
            [&]() -> en::Status {
              const auto occupied = node.catalog().OccupancyOf(node.domain()).Total().value();
              const auto cap = node.budget().value();
              const auto charge = kObservations + en::ExecutionMetadataAllowance::OverheadBytes();
              if (occupied > cap || charge > cap - occupied)
                return std::unexpected("native target observations exceed node capacity");
              auto admitted = en::ExecutionMetadataAllowance::Admit(
                  node.catalog(), node.domain(), charge,
                  llmp::platform::AvailableMemoryBytes().value_or(0), rt::kUncountedMargin);
              if (!admitted) return std::unexpected(admitted.error());
              life->observations = std::move(*admitted);
              if (!life->observations->Charge(kObservations))
                return std::unexpected("native target observations lack payment");
              return {};
            },
            "pay native DS observations");
        !paid)
      return paid;
    // Only the new paid owner may take over both arms' retained host results.
    if (previous) previous->server.reset();
    auto* model = dynamic_cast<rt::EngineLlm*>(life->server->Find("deepseek"));
    if (!model || model->logit_width() != kVocab || model->generation_wave_capacity() != 1)
      return std::unexpected("native target selected the wrong source or serving envelope");
    rt::SwapParts parts;
    if (auto activated = life->server->Activate(*model, parts); !activated) return activated;
    auto branch = model->branch(0);
    if (!branch) return std::unexpected(branch.error());
    std::array<rt::Llm::Branch*, 1> active{*branch};
    if (auto selected = life->server->SelectRequestBranches(*model, active); !selected)
      return selected;
    std::array<std::int32_t, kPrefix> prompt{};
    for (std::size_t i = 0; i < prompt.size(); ++i) prompt[i] = 100 + (i * 7919U + 17U) % 30000U;
    if (auto prefill = (*branch)->Prefill(prompt, life->result.heads[0]); !prefill) return prefill;
    for (unsigned turn = 0; turn != 2; ++turn) {
      rt::GenerateOptions options;
      options.max_tokens = 8;
      options.stop = false;
      options.keep_logits = true;
      auto& generation = life->result.generations[turn];
      if (auto generated = (*branch)->Generate(life->result.heads[turn], options, generation);
          !generated)
        return generated;
      if (generation.tokens.size() != 8 || generation.logits.size() != 8 ||
          (drafter.empty() ? generation.drafted != 0 : generation.drafted == 0) ||
          generation.stopped || generation.cancelled)
        return std::unexpected(drafter.empty()
                                   ? "native target returned incomplete plain generation"
                                   : "native target did not speculate with its drafter");
      life->result.history[turn] = (*branch)->history();
      auto state = ReadState(*life, *model);
      if (!state) return std::unexpected(state.error());
      life->result.states[turn == 0 ? 0 : 2] = std::move(*state);
      if (turn) continue;
      if (auto ended = life->server->EndRequestBranches(*model); !ended) return ended;
      if (auto spilled = model->SpillIdle(**branch); !spilled) return spilled;
      if (!(*branch)->spilled()) return std::unexpected("native target did not spill");
      if (auto selected = life->server->SelectRequestBranches(*model, active); !selected)
        return selected;
      const auto before = model->graphs();
      auto begun = (*branch)->BeginPrompt(life->result.history[0], 0, false, true);
      if (!begun) return std::unexpected(begun.error());
      life->restoration = std::move(*begun);
      while (!life->restoration->done())
        if (auto advanced = life->restoration->Advance(); !advanced) return advanced;
      if (life->restoration->reused() != life->result.history[0].size())
        return std::unexpected("native target restore recomputed its history");
      if (auto finished = life->restoration->Finish(); !finished) return finished;
      life->restoration.reset();
      const auto after = model->graphs();
      if (before.eager != after.eager || before.captured != after.captured ||
          before.replayed != after.replayed || (*branch)->spilled())
        return std::unexpected("native target restoration executed model work");
      auto restored = ReadState(*life, *model);
      if (!restored) return std::unexpected(restored.error());
      life->result.states[1] = std::move(*restored);
      if (life->result.states[0] != life->result.states[1])
        return std::unexpected("native target restored different state");
      auto next = (*branch)->history();
      next.push_back(generation.tokens.back());
      std::uint32_t reused = 0;
      if (auto continued = (*branch)->PreparePrompt(next, 0, life->result.heads[1], reused);
          !continued)
        return continued;
      if (reused + 1 != next.size())
        return std::unexpected("native target continuation lost prefix");
    }
    if (model->runner().coverage().violations)
      return std::unexpected("native target used uncovered memory");
    return life->server->EndRequestBranches(*model);
  }();
  if (!completed) {
    const auto error = completed.error();
    std::ignore = life.release();
    return std::unexpected(error + "; complete failed owner retained");
  }
  if (auto retired = life->server->TearDown(); !retired) {
    const auto error = retired.error();
    std::ignore = life.release();
    return std::unexpected(error + "; unproven owner retained");
  }
  return life;
}
// The v0 target (and DSpark) against the published v1 artifacts, both against
// the frozen digests. The v0 arm needs its artifacts; without them (or with
// LLMP_TEST_V1_FROZEN_ONLY) only the v1 arm runs.
class PublishedCase : public testing::Test {
 public:
  explicit PublishedCase(bool dspark) : dspark_(dspark) {}

 private:
  void TestBody() override {
    const char* old_env = std::getenv("LLMP_DSV4_TARGET_STORE");
    const char* native_store = std::getenv("LLMP_TEST_DSV4_V1_STORE");
    if (native_store == nullptr) native_store = std::getenv("LLMP_TEST_V1_STORE");
    ASSERT_NE(native_store, nullptr);
    const char* home = std::getenv("HOME");
    const std::string old_store = old_env != nullptr
                                      ? old_env
                                      : std::string(home != nullptr ? home : "") +
                                            "/.local/share/llmp/m3-artifacts";
    const char* given = std::getenv("LLMP_TEST_DSV4_V1_ID");
    std::string id = given != nullptr ? given : "";
    if (id.empty()) {
      const auto mapped = llmp::test_support::V1IdFor(kArtifact);
      if (mapped) id = *mapped;
    }
    ASSERT_EQ(id.size(), 64U);
    std::string drafter;
    if (dspark_) {
      const auto mapped = llmp::test_support::V1IdFor(kDsparkArtifact);
      ASSERT_TRUE(mapped) << "no v1 DSpark ID in the store's v1-ids.json";
      drafter = *mapped;
    }
    ASSERT_TRUE(std::ranges::all_of(id, [](char c) {
      return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
    }));
    std::error_code ec;
    const bool have_v0 = std::getenv("LLMP_TEST_V1_FROZEN_ONLY") == nullptr &&
                         fs::exists(fs::path(old_store) / kArtifact, ec) &&
                         (!dspark_ || fs::exists(fs::path(old_store) / kDsparkArtifact, ec));
    std::unique_ptr<Owner> old;
    if (have_v0) {
      auto checked = Metadata(fs::path(old_store) / kArtifact);
      ASSERT_TRUE(checked) << Failed(checked).value_or("");
      auto first = native_target::Run(old_store, kArtifact, nullptr,
                                      dspark_ ? kDsparkArtifact : std::string_view());
      ASSERT_TRUE(first) << Failed(first).value_or("");
      old = std::move(*first);
    }
    auto native = native_target::Run(native_store, id, old.get(), drafter);
    ASSERT_TRUE(native) << Failed(native).value_or("");
    llmp::test_support::FrozenDigests frozen(dspark_ ? "dsv4_dspark" : "dsv4_target");
    for (const auto* owner : {old.get(), native->get()}) {
      if (owner == nullptr) continue;
      SCOPED_TRACE(owner == old.get() ? "v0 against frozen" : "v1 against frozen");
      const auto& r = owner->result;
      using llmp::test_support::HexOfFloats;
      using llmp::test_support::HexOfInts;
      using llmp::test_support::HexOfRows;
      for (unsigned turn = 0; turn != 2; ++turn) {
        const auto t = std::to_string(turn);
        frozen.Expect("heads." + t, HexOfFloats(r.heads[turn]));
        frozen.Expect("tokens." + t, llmp::test_support::JoinTokens(r.generations[turn].tokens));
        frozen.Expect("logits." + t, HexOfRows(r.generations[turn].logits));
        frozen.Expect("history." + t, HexOfInts(r.history[turn]));
        if (dspark_) {
          frozen.Expect("drafted." + t, std::to_string(r.generations[turn].drafted));
          frozen.Expect("accepted." + t, std::to_string(r.generations[turn].accepted));
        }
      }
      // States: 0 after the first turn, 1 after spill and restore, 2 after the second.
      for (std::size_t i = 0; i < r.states.size(); ++i)
        frozen.Expect("state." + std::to_string(i),
                      llmp::base::ToHex(r.states[i].digest) + ':' +
                          std::to_string(r.states[i].positions) + ':' +
                          std::to_string(r.states[i].ranges.size()));
      if (frozen.recording()) break;  // one arm records
    }
    const auto& b = (*native)->result;
    const auto finite = [](std::span<const float> row) {
      EXPECT_EQ(row.size(), kVocab);
      EXPECT_TRUE(std::ranges::all_of(row, [](float f) { return std::isfinite(f); }));
    };
    for (unsigned turn = 0; turn != 2; ++turn) {
      finite(b.heads[turn]);
      for (const auto& row : b.generations[turn].logits) finite(row);
    }
    if (have_v0) {
      const auto& a = old->result;
      EXPECT_EQ(a.states, b.states);
      const auto compare = [&](std::span<const float> x, std::span<const float> y) {
        ASSERT_EQ(x.size(), kVocab);
        ASSERT_EQ(y.size(), kVocab);
        EXPECT_EQ(std::memcmp(x.data(), y.data(), x.size_bytes()), 0);
      };
      for (unsigned turn = 0; turn != 2; ++turn) {
        compare(a.heads[turn], b.heads[turn]);
        EXPECT_EQ(a.history[turn], b.history[turn]);
        EXPECT_EQ(a.generations[turn].tokens, b.generations[turn].tokens);
        EXPECT_EQ(a.generations[turn].drafted, b.generations[turn].drafted);
        EXPECT_EQ(a.generations[turn].accepted, b.generations[turn].accepted);
        ASSERT_EQ(a.generations[turn].logits.size(), 8U);
        ASSERT_EQ(b.generations[turn].logits.size(), 8U);
        for (unsigned row = 0; row != 8; ++row)
          compare(a.generations[turn].logits[row], b.generations[turn].logits[row]);
      }
      RecordProperty("v0_selected_node_budget", std::to_string(a.selected_budget));
    }
    RecordProperty("complete_head_rows", 36);
    RecordProperty("head_values_per_row", kVocab);
    RecordProperty("complete_state_images", 6);
    RecordProperty("restored_owners", 2);
    RecordProperty("prompt_tokens", kPrefix);
    RecordProperty("native_artifact", id);
    if (dspark_) RecordProperty("native_drafter", drafter);
    RecordProperty("v1_selected_node_budget", std::to_string(b.selected_budget));
    if (old) fs::remove_all(old->scratch);
    fs::remove_all((*native)->scratch);
    old.reset();
  }
  bool dspark_;
};
[[maybe_unused]] const auto* published = []() -> testing::TestInfo* {
  if (!std::getenv("LLMP_TEST_DSV4_V1_STORE") && !std::getenv("LLMP_TEST_V1_STORE")) return nullptr;
  testing::RegisterTest("Dsv4EngineTargetGpu", "PublishedDsparkMatchesV0HeadsStateAndSpill",
                        nullptr, nullptr, __FILE__, __LINE__,
                        []() -> testing::Test* { return new PublishedCase(true); });
  return testing::RegisterTest("Dsv4EngineTargetGpu", "PublishedGgufMatchesV0HeadsStateAndSpill",
                               nullptr, nullptr, __FILE__, __LINE__,
                               []() -> testing::Test* { return new PublishedCase(false); });
}();
}  // namespace native_target

}  // namespace
