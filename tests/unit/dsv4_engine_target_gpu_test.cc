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
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "artifact/json.h"
#include "base/sha256.h"
#include "engine/component_source.h"
#include "engine/expert_slab.h"
#include "engine/paged_weights.h"
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
constexpr std::uint64_t kSnapshot = kState + 2 * kMiB;
constexpr std::uint32_t kContext = 3072, kChunk = 512, kPrefix = 2560, kTail = 8;
constexpr std::uint32_t kVocab = 129280;
constexpr std::string_view kArtifact =
    "8a355bfb27c90e1150fbd7fa62ea6e63f6bf34fcca33934e52d22773f1508234";

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
    4 * (3 * 1400ULL * 4096 + 3 * 12000ULL * 64 + 50000ULL * 32 + 3 * 4096ULL * 64);
constexpr std::uint64_t kTokenizer =
    3 * 4 * 130000ULL * 32 + 3 * 2 * 6 * kMiB + 2 * 130000ULL * (64 + 3 * 8) + 3 * 16 * kMiB +
    750000ULL * (12 + 4 + 4 + 16) + 3 * 2 * 130000ULL * 16 + 2 * kMiB;
constexpr std::uint64_t kCatalog = 55000ULL * (3 * 192 + 3 * 288 + 320 + 3 * 128 + 3 * 64 + 128);
constexpr std::uint64_t kBindings =
    6 * 1400ULL * 2048 + 3 * 4096ULL * 256 + 3 * 6 * kVocab * 4ULL + 4 * kMiB;
constexpr std::uint64_t kDirectory = 3 * 65536ULL * (32 + 2 * 256);
static_assert(kCompiler + kArtifactParser + kArtifacts + kTokenizer + kCatalog + kBindings +
                  kDirectory + kCaller <=
              kConstruction);
static_assert(2 * (kTail + 1) * kVocab * sizeof(float) + 4 * kMiB < kCaller);

rt::ModelSettings Settings() {
  rt::ModelSettings s;
  s.name = "deepseek-component-target";
  s.architecture = "deepseek4";
  s.context.value = kContext;
  s.prefill_chunk.value = kChunk;
  s.max_slots.value = 1;
  s.wave_read_align.value = 256;
  s.speculation.value = false;
  return s;
}
std::expected<std::shared_ptr<const en::NumericalDomain>, std::string> Domain() {
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
  en::PagedNode node{{.slot_bytes = en::kSlabSlotBytes}};
  // Complete caller storage is prepaid with construction; local token memory
  // also enforces its own small limit before any history-vector growth.
  rt::RequestMemory tokens{kMiB};
  std::shared_ptr<en::ExecutionMetadataAllowance> allowance;
  std::unique_ptr<rt::EngineLlm> model;
  std::unique_ptr<rt::SnapshotStore> prefix, end;
  std::vector<en::PagedModel*> entered;
  std::array<std::int32_t, kPrefix> ids{};
  std::vector<float> head;
  rt::Generation first, second;
  std::uint64_t programs = 0, full_programs = 0, footprint = 0, dynamic = 0;
  llmp::base::Sha256 recipes;
  std::string restored_state_digest;
  bool teardown_attempted = false;

  Status Validate(const kn::LowerInput& in, const kn::StepProgram& program) {
    if (!in.token) return std::unexpected("token source metadata is required");
    // This callback runs after Lower and before Bind, including measurement.
    // A real launch's BoundPlan does not exist at this point.
    if (in.token->shape.stage != 0 || in.token->shape.segments.size() != 1 || program.routed() ||
        in.plan.steps.empty() || program.launches() == 0)
      return std::unexpected("the target did not lower to its ordinary owned full-step program");
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
    // BuildStep's backwards liveness keeps the final attention's state writes
    // but omits its unread output and stateless FFN on a solo no-head chunk.
    // This fixture has no companion injection or feature tap retaining that FFN.
    const bool headed = in.token->shape.segments.front().outputs != 0;
    const std::uint32_t expected_products = 3U * (headed ? 43U : 42U);
    if (arrays != expected_products || products != expected_products || hash_inputs != 3)
      return std::unexpected(std::format(
          "target live domain differs: arrays={} products={} hash_inputs={} expected_products={} "
          "expected_hash_inputs=3 rows={} outputs={}",
          arrays, products, hash_inputs, expected_products, in.token->shape.segments.front().rows,
          in.token->shape.segments.front().outputs));
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
    auto domain = Domain();
    if (!domain) return std::unexpected(domain.error());
    llmp::config::ModelEntry entry;
    entry.name = Settings().name;
    entry.artifact = std::string(kArtifact);
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
    model = std::make_unique<rt::EngineLlm>(node, entry, Settings(), roles, 0, std::move(build));
    model->SetTokenMemory(tokens);
    model->set_turn_checkpoints(false);
    const auto setup = model->Setup();
    if (model->has_paged()) entered.push_back(&model->paged());
    if (!setup) return setup;
    const auto execution = model->ExecutionIdentity();
    if (!execution || !execution->has_value())
      return std::unexpected("the actual target has no complete execution identity after Setup");
    recipes.Update("llmp.ds.execution.v1");
    recipes.Update(std::as_bytes(std::span(**execution)));
    if (model->logit_width() != kVocab || node.StateCapacity() != kState)
      return std::unexpected(
          "the actual target head or independently derived state layout differs");
    const auto weight_bytes = model->weights().size() * en::kPagedExtent;
    if (weight_bytes != kWeights)
      return std::unexpected("the actual full dense-plus-padded-expert closure differs");
    std::uint64_t left = kCap;
    const auto take = [&left](std::uint64_t bytes) {
      if (bytes > left) return false;
      left -= bytes;
      return true;
    };
    const auto fixed = node.catalog().OccupancyOf(node.domain()).Total().value();
    const auto pool = std::max<std::uint64_t>(model->pool_needed(), en::kPagedExtent);
    // Before physical workspace or weight admission: all current measured
    // storage plus page rounding, the complete weight closure, conservative
    // live/copy/spill state overlap, caller host staging and minimum plans.
    if (!take(fixed) || !take(weight_bytes) || !take(4 * kSnapshot) ||
        !take(model->activations_needed()) || !take(pool) || !take(2 * en::kPagedExtent) ||
        !take(model->host_input_bytes()) || !take(model->plan_floor_bytes()))
      return std::unexpected("actual Setup storage exceeds the unchanged 110 GiB domain");
    footprint = kCap - left;
    const auto available = llmp::platform::AvailableMemoryBytes().value_or(0);
    // Conservative complete footprint check even though existing fixed
    // allocations already reduce MemAvailable. Unknown is never authority.
    if (footprint > available || rt::kUncountedMargin > available - footprint)
      return std::unexpected("actual Setup footprint plus six GiB lacks physical headroom");
    if (auto s = node.MapWorkspace(model->activations_needed(), pool); !s) return s;
    const auto mapped = node.catalog().OccupancyOf(node.domain()).Total().value();
    const auto now = llmp::platform::AvailableMemoryBytes().value_or(0);
    rt::MemoryGuard guard{.largest = weight_bytes,
                          .host_inputs = model->host_input_bytes(),
                          .plans = model->plan_floor_bytes(),
                          .available = now,
                          .fixed = mapped,
                          .requests = kCaller};
    if (auto s = rt::CheckMemoryGuard(guard); !s) return s;
    const auto reserve = rt::GuardReserve(guard);
    if (now == 0 || now < reserve || mapped > kCap)
      return std::unexpected("unknown or insufficient ordinary dynamic budget");
    dynamic = mapped + (now - reserve) / en::kPagedExtent * en::kPagedExtent;
    if (auto s = rt::CheckDiagnosticBudgetCap(guard, dynamic, kCap, 4 * kSnapshot); !s) return s;
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
    for (std::uint32_t i = 0; i < ids.size(); ++i)
      ids[i] = static_cast<std::int32_t>(100 + (i * 7919U + 17U) % 30000U);
    prefix = std::make_unique<rt::SnapshotStore>(node, kSnapshot);
    end = std::make_unique<rt::SnapshotStore>(node, kSnapshot);
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
          if (auto s = branch.Generate(head, options, first); !s) return s;
          if (auto s = Complete(first); !s) return s;
          auto completed = branch.CaptureState(*end);
          if (!completed) return std::unexpected(completed.error());
          auto digest = end->Digest(*completed);
          if (!digest) return std::unexpected(digest.error());
          if (auto s = branch.RestoreState(*prefix, *saved); !s) return s;
          if (model->runner().slots().positions(0) != kPrefix || branch.history().size() != kPrefix)
            return std::unexpected("snapshot restore lost its actual prefix coordinate");
          if (auto s = branch.Generate(head, options, second); !s) return s;
          if (auto s = Complete(second); !s) return s;
          if (first.tokens != second.tokens)
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
  Status Complete(const rt::Generation& generated) const {
    if (generated.tokens.size() != kTail + 1 || generated.logits.size() != kTail + 1 ||
        generated.steps != kTail || generated.stopped || generated.cancelled ||
        !std::ranges::all_of(generated.tokens, [](std::int32_t id) {
          return id >= 0 && id < static_cast<std::int32_t>(kVocab);
        }))
      return std::unexpected("the actual plain decode did not complete all eight steps");
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
}  // namespace
