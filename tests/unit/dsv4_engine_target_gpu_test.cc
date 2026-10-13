// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Full prepared DeepSeek target through the serving factory and shared runner.
// Default-off diagnostic; ordinary complete weight closure, not selected-only
// expert paging. Exact comparisons are one implementation's snapshot replay.

#include <gtest/gtest.h>

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
#include "runtime/snapshot_store.h"

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
constexpr std::uint64_t kState = 480247808ULL;
constexpr std::uint64_t kDsparkWeights = 10898898944ULL;
constexpr std::uint64_t kDsparkState = 6291456ULL;
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

constexpr std::uint32_t kScoreInputs = 257, kScoreChunk = 128;
constexpr std::uint64_t kScoreOutputs = 4ULL * kVocab * kScoreChunk;
constexpr std::uint64_t kScoreHost =
    16ULL * kScoreInputs + 4 * kScoreOutputs + 4ULL * kVocab + kMiB;
// Setup programs are transient. Their measured retained/capture costs are
// admitted separately before any runtime compilation or weight payload.
constexpr std::uint64_t kScoreFixed = kCompiler + kArtifactParser + kArtifacts + kTokenizer +
                                      kCatalog + kBindings + kDirectory + kCaller + kScoreHost;
static_assert(kScoreFixed < kConstruction);
constexpr std::uint64_t kScoreCredit = kConstruction - kScoreFixed;
static_assert(kScoreCredit == 567071792ULL);
// ComponentLlmBuild retains one complete kConstruction charge. These named
// subcategories reuse that held grant; charging its residual again would
// double-count it. This closed diagnostic has no second factory construction.
std::expected<std::uint64_t, std::string> HeldScoreCredit(
    const en::ExecutionMetadataAllowance& allowance, std::uint64_t plans) {
  if (allowance.capacity() < kConstruction || allowance.used() != kConstruction)
    return std::unexpected("scored plans lack the complete held factory construction charge");
  return std::min(plans, kScoreCredit);
}
// One selected index (0..255) plus separator per source-bounded graph node.
constexpr std::uint64_t kSelectionReceipt = 4ULL * llmp::ir::Graph::kMaxNodes;
constexpr std::uint64_t kClassReceipt = 3 * kMiB;
static_assert(8 * (kSelectionReceipt + 512 * 21 + 512) <= kClassReceipt);
// Retained selections, stream growth/string copies and GTest property copy;
// leave 4 MiB for the existing small corpus/observer working storage.
static_assert(8 * kSelectionReceipt + 4 * kSelectionReceipt + 4 * kClassReceipt + 4 * kMiB <=
              kCaller);
static_assert(kScoreOutputs <= 64 * kMiB && kScoreHost == 266335248ULL);

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
  explicit Lifetime(bool with_companion = false, bool with_score = false)
      : companion(with_companion), scoring(with_score) {}
  bool companion, scoring;
  struct ScoreClass {
    pl::StepShape shape;
    std::vector<std::uint64_t> parameters;
    bool measured = false, lowered = false;
    std::uint64_t host_bytes = 0, launches = 0, selected_steps = 0, paid_bytes = 0;
    std::string selection;
  };
  std::array<ScoreClass, 8> score_classes;
  en::StepSource* source = nullptr;  // Owned by the actual EngineLlm runner.
  llmp::model::Batching score_caps{};
  std::array<std::int32_t, kScoreInputs> score_ids{};
  std::array<ts::CorpusTargetScore, kScoreInputs - 1> scores{};
  std::array<bool, kScoreInputs> seen{};
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
      const auto parameters = source->ShapeParameters(in.token->shape);
      const auto record = std::ranges::find_if(score_classes, [&](const auto& item) {
        return item.shape == in.token->shape && item.parameters == parameters &&
               item.shape.segments.front().past == in.token->shape.segments.front().past;
      });
      if (record == score_classes.end())
        return std::unexpected("score/setup shape parameters were not prepaid and measured");
      // Only the final three classes belong to the fixed scored schedule.
      // Other setup programs are destroyed without entering the plan cache.
      const bool retained = record - score_classes.begin() >= 5;
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
      if (paid >= record->paid_bytes) {
        record->host_bytes = host;
        record->launches = launches;
        record->selected_steps = in.plan.steps.size();
        record->paid_bytes = paid;
      }
      // Registry indices are below the existing256 implementation limit:
      // at most three decimal digits plus a comma, checked before streaming.
      if (in.plan.steps.size() > llmp::ir::Graph::kMaxNodes)
        return std::unexpected("selection receipt exceeds prepaid text capacity");
      std::ostringstream selected;
      for (const auto& item : in.plan.steps) selected << item.implementation << ',';
      if (selected.tellp() > static_cast<std::streamoff>(kSelectionReceipt))
        return std::unexpected("selection receipt exceeds payment");
      const auto serialized = selected.str();
      if (!record->selection.empty() && record->selection != serialized)
        return std::unexpected("measured and bound selection differs for the same complete key");
      record->selection = serialized;
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
      build = [this, inner = std::move(build)](const rt::ModelSettings& settings) mutable
          -> std::expected<rt::EngineLlmParts, std::string> {
        auto parts = inner(settings);
        if (!parts) return std::unexpected(parts.error());
        if (parts->config.setup_shapes.size() != score_classes.size())
          return std::unexpected("score setup class inventory differs");
        source = parts->source.get();
        score_caps = parts->caps;
        for (std::size_t i = 0; i < score_classes.size(); ++i) {
          score_classes[i].shape = parts->config.setup_shapes[i];
          score_classes[i].parameters = source->ShapeParameters(score_classes[i].shape);
          if (score_classes[i].parameters.size() > 512)
            return std::unexpected("score shape parameters exceed fixed payment");
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
      if (!std::ranges::all_of(score_classes, [](const auto& item) { return item.measured; }))
        return std::unexpected("incomplete score class measurement");
      std::uint64_t largest = 0;
      for (std::size_t i = 5; i < score_classes.size(); ++i) {
        const auto bytes = score_classes[i].paid_bytes;
        if (bytes > kCap - score_plan_bytes)
          return std::unexpected("retained scored plans exceed node geometry");
        score_plan_bytes += bytes;
        largest = std::max(largest, bytes);
      }
      if (largest > kCap - score_plan_bytes)
        return std::unexpected("scored replacement owner exceeds node geometry");
      score_plan_bytes += largest;
      const auto credit = HeldScoreCredit(*allowance, score_plan_bytes);
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
        (scoring && (!take(kScoreHost) || !take(score_plan_bytes) || !take(score_extra_extent))))
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
      if (!score_plan_allowance->Charge(score_plan_bytes - kScoreCredit))
        return std::unexpected("scored metadata excess lacks its admitted payment");
    }
    if (auto s = node.MapWorkspace(model->activations_needed(), pool); !s) return s;
    const auto mapped = node.catalog().OccupancyOf(node.domain()).Total().value();
    const auto now = llmp::platform::AvailableMemoryBytes().value_or(0);
    rt::MemoryGuard guard{.largest = weight_bytes,
                          .host_inputs = model->host_input_bytes(),
                          .plans = model->plan_floor_bytes(),
                          .available = now,
                          .fixed = mapped,
                          .requests = kCaller + (scoring ? kScoreHost + score_plan_bytes : 0)};
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
    if (auto checked = File(ids_path, sizeof(score_ids), ids_hash); !checked) return checked;
    std::string text(17210, '\0');
    std::ifstream text_in(text_path, std::ios::binary);
    if (!text_in.read(text.data(), static_cast<std::streamsize>(text.size())))
      return std::unexpected("captured corpus read failed");
    std::ifstream ids_in(ids_path, std::ios::binary);
    if (!ids_in.read(reinterpret_cast<char*>(score_ids.data()), sizeof(score_ids)))
      return std::unexpected("captured IDs read failed");
    if (model->tokenizer().size() != kVocab || !model->tokenizer().bos() ||
        *model->tokenizer().bos() != score_ids.front())
      return std::unexpected("native corpus vocabulary/BOS differs");
    auto native = model->EncodeText(text, &score_tokens);
    if (!native) return std::unexpected(native.error());
    if (native->size() < score_ids.size() ||
        !std::ranges::equal(std::span(*native).first(score_ids.size()), score_ids))
      return std::unexpected("native tokenizer differs from captured reference prefix");
    return {};
  }
  Status Score() {
    auto composer = en::WaveComposer::For(score_caps, model->runner().slots());
    if (!composer) return std::unexpected(composer.error());
    score_composer.emplace(std::move(*composer));
    const std::array documents{ts::CorpusDocument{.slot = 0, .tokens = score_ids}};
    auto bound =
        ts::CorpusScore::Bound(documents, {.vocabulary = kVocab, .chunk_rows = kScoreChunk});
    if (!bound || *bound != kScoreHost) return std::unexpected("scorer host payment differs");
    auto made = ts::CorpusScore::Make(
        node, model->runner(), *score_composer, documents,
        {.vocabulary = kVocab, .chunk_rows = kScoreChunk},
        [this](const ts::CorpusTargetScore& row) -> Status {
          if (score_count >= scores.size() || row.document || row.slot ||
              row.position != score_count + 1 || row.target != score_ids[row.position])
            return std::unexpected("DeepSeek loss coordinates differ");
          scores[score_count++] = row;
          return {};
        },
        [this](std::uint32_t document, std::uint32_t slot, std::uint32_t position,
               std::span<const float> row) -> Status {
          if (document || slot || position >= seen.size() || seen[position] || row.size() != kVocab)
            return std::unexpected("DeepSeek full finite row coverage differs");
          seen[position] = true;
          return {};
        });
    if (!made) return std::unexpected(made.error());
    scorer = std::move(*made);
    auto completed = node.WithRequest(
        0, model->request_closure(), "DeepSeek fixed natural corpus", [&]() -> Status {
          for (std::uint32_t wave = 0; wave < 3; ++wave) {
            auto done = scorer->Step();
            if (!done) return std::unexpected(done.error());
            if (*done != (wave == 2)) return std::unexpected("DeepSeek scored cutoff differs");
          }
          return {};
        });
    if (!completed) return completed;
    if (!scorer->complete() || score_count != scores.size() ||
        scorer->result().inputs != kScoreInputs || scorer->result().targets != kScoreInputs - 1 ||
        scorer->result().steps != 3 || model->runner().slots().positions(0) != kScoreInputs ||
        !std::ranges::all_of(seen, [](bool value) { return value; }))
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

TEST(Dsv4EngineTargetInputs, RejectsUnpaidOrDifferentEnvelopeBeforeArtifactRead) {
  EXPECT_FALSE(rt::OnComponentEngine("deepseek4"));
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

TEST(Dsv4EngineTargetInputs, CreditsOnlyTheRetainedFactoryConstructionCharge) {
  llmp::catalog::Catalog catalog;
  const auto owner = catalog.AddDomain("held construction credit");
  auto paid = en::ExecutionMetadataAllowance::Admit(catalog, owner, kConstructionExtent,
                                                    kConstructionExtent + rt::kUncountedMargin,
                                                    rt::kUncountedMargin);
  ASSERT_TRUE(paid) << Failed(paid).value_or("");
  auto& allowance = **paid;
  EXPECT_FALSE(HeldScoreCredit(allowance, kScoreCredit));
  // The real factory charges this whole grant before source construction and
  // retains it in ComponentHold through source/backend teardown.
  ASSERT_TRUE(allowance.Charge(kConstruction));
  EXPECT_EQ(allowance.capacity() - allowance.used(),
            2 * kMiB - en::ExecutionMetadataAllowance::OverheadBytes());
  EXPECT_FALSE(allowance.Charge(kScoreCredit));  // Reproduces the old double charge.
  const auto used = allowance.used();
  for (const auto plans : {kScoreCredit - 1, kScoreCredit, kScoreCredit + 1}) {
    const auto credit = HeldScoreCredit(allowance, plans);
    ASSERT_TRUE(credit) << Failed(credit).value_or("");
    EXPECT_EQ(*credit, std::min(plans, kScoreCredit));
    EXPECT_LE(kScoreFixed + *credit, used);
    EXPECT_EQ(plans - *credit, plans > kScoreCredit ? 1U : 0U);
    EXPECT_EQ(allowance.used(), used);
  }
  ASSERT_TRUE(allowance.Charge(1));
  EXPECT_FALSE(HeldScoreCredit(allowance, kScoreCredit));
  allowance.Release(1);
  allowance.Release(kConstruction);
  EXPECT_FALSE(HeldScoreCredit(allowance, kScoreCredit));
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
  void TestBody() override {
    const char* store = std::getenv("LLMP_DSV4_TARGET_STORE");
    const char* scratch = std::getenv("LLMP_TEST_SCRATCH");
    ASSERT_NE(store, nullptr);
    ASSERT_NE(scratch, nullptr);
    const fs::path out = fs::path(scratch) / "dsv4-corpus";
    std::error_code error;
    fs::create_directories(out, error);
    ASSERT_FALSE(error) << error.message();
    auto life = std::make_unique<Lifetime>(false, true);
    const auto result = life->Run(store, out);
    const auto retired = life->Retire();
    if (!retired) {
      std::ignore = life.release();
      FAIL() << Failed(retired).value_or("");
    }
    ASSERT_TRUE(result) << Failed(result).value_or("");
    std::ostringstream losses, classes;
    losses << std::setprecision(17);
    for (const auto& row : life->scores)
      losses << row.document << ',' << row.position << ',' << row.target << ',' << row.nll << '\n';
    for (std::size_t i = 0; i < life->score_classes.size(); ++i) {
      const auto& item = life->score_classes[i];
      const auto& shape = item.shape;
      const auto& segment = shape.segments.front();
      classes << i << ',' << static_cast<unsigned>(shape.phase) << ',' << shape.stage << ','
              << shape.row_invariant << ',' << shape.read_align << ','
              << static_cast<unsigned>(segment.slot) << ',' << segment.rows << ',' << segment.past
              << ',' << segment.read_cells << ',' << segment.outputs << ','
              << static_cast<unsigned>(segment.group) << ',' << segment.want_head << ','
              << segment.greedy << ',' << item.measured << ',' << item.lowered << ','
              << item.host_bytes << ',' << item.launches << ',' << item.selected_steps << ','
              << item.paid_bytes << "|parameters:";
      for (auto value : item.parameters) classes << value << ',';
      classes << "|selection:" << item.selection << '\n';
    }
    ASSERT_LE(losses.tellp(), 64 * 1024);
    ASSERT_LE(classes.tellp(), kClassReceipt);
    RecordProperty("losses", losses.str());
    RecordProperty("classes", classes.str());
    RecordProperty("inputs", std::to_string(kScoreInputs));
    RecordProperty("targets", std::to_string(life->score_count));
    RecordProperty("finite_rows", std::to_string(std::ranges::count(life->seen, true)));
    RecordProperty("waves", std::to_string(life->scorer->result().steps));
    RecordProperty("scorer_host_bound", std::to_string(kScoreHost));
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
};
[[maybe_unused]] const auto* const kNaturalCorpus =
    testing::RegisterTest("Dsv4CorpusScoreManual", "Natural257", nullptr, nullptr, __FILE__,
                          __LINE__, []() -> Dsv4NaturalCorpus* { return new Dsv4NaturalCorpus; });
}  // namespace
