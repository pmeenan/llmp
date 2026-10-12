// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// One owned target/MTP composition, two owners with different pending spans.
// This is same-implementation state/capture evidence, not a quality comparison.
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
#include <optional>
#include <span>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "artifact/artifact.h"
#include "base/sha256.h"
#include "corpus_score.h"
#include "engine/decoding.h"
#include "engine/draft_companions.h"
#include "engine/model_runner.h"
#include "engine/paged_weights.h"
#include "expected_error.h"
#include "kernels/ggml/graph_plan.h"
#include "model/qwen38.h"
#include "planner/plan.h"
#include "platform/host_probe.h"
#include "runtime/component_build.h"
#include "runtime/memory_guard.h"
#include "tokenizer_fixtures.h"

namespace {
namespace en = llmp::engine;
namespace kn = llmp::kernels;
namespace pl = llmp::planner;
namespace rt = llmp::runtime;
namespace ts = llmp::test_support;
namespace fs = std::filesystem;
using llmp::test_support::Failed;
constexpr std::uint64_t kMiB = 1ULL << 20, kGiB = 1ULL << 30;
constexpr std::uint64_t kDomain = 104 * kGiB, kHost = 12 * kGiB;
constexpr std::uint32_t kVocabulary = 248320, kDraftLimit = 65536;
// Factory: at most eleven classes for each of lone/joined owners. Actual
// script: two lone prompts + two companion tails + three joined shapes.
// Commit: three possible accepted widths for each of two real slots.
constexpr std::size_t kFactoryClasses = 2 * (1 + 2 + 4 + 4);
constexpr std::size_t kClasses = kFactoryClasses + 2 * 2 + 3 + 2 * 3;
constexpr std::uint64_t kProgram = 64 * kMiB;
constexpr std::uint64_t kNodes = llmp::ir::Graph::kMaxNodes;
constexpr std::uint64_t kTensors = llmp::ir::Graph::kMaxTensors;
// A verify can save four GDN intermediates and one PLE carry per
// layer/owner. Actual source has fewer; account for all 48 layers.
constexpr std::uint64_t kSaves = 5 * 48 * 2;
constexpr std::uint64_t kDescriptors = kTensors + 16 + (8 + 23) * kNodes + 4 * kSaves;
constexpr std::uint64_t kCompiler = 4 * 3 * (kNodes * 480 + kTensors * 192) +
                                    4096 * (kNodes + kTensors) + kDescriptors * (512 + 1024 + 32) +
                                    64 * kMiB;
constexpr std::uint64_t kRetained = (kClasses + 1) * (kProgram + 4096 * en::kGraphNodeHostBytes);
constexpr std::uint64_t kMetadata = 2 * kGiB, kObservations = 256 * kMiB;
static_assert(kCompiler + kRetained + kMetadata + kObservations <= kHost);
static_assert(kNodes == 65536 && kTensors == 131072);
static_assert(sizeof(llmp::ir::Node) <= 160 && sizeof(llmp::ir::TensorDesc) <= 160);
static_assert(sizeof(pl::Selected) <= 128 && sizeof(pl::Applied) <= 64);
static_assert(sizeof(llmp::kernels::ggml::PlanStep) <= 64);
static_assert(sizeof(std::string) <= 32 && sizeof(en::SlotLifecycle::Range) <= 24);
constexpr std::string_view kTarget =
    "c4fb47a911207c11f935f932d05196dc1701aa0d886eac1b5e91934e554b5a93";
constexpr std::string_view kDrafter =
    "8600a99819ce583a719ebfb457de8cac40b4d0bd1ebe557ceb13dff5961aee40";

std::string Env(const char* name) {
  const auto* value = std::getenv(name);  // NOLINT(concurrency-mt-unsafe)
  return value ? value : "";
}
llmp::base::Sha256Digest Hash(std::span<const std::byte> bytes) {
  return llmp::base::Sha256{}.Update(bytes).Finish();
}
// Fixed stack read before JSON/source/graph construction; source spellings
// cannot exceed the reviewed compiler's owned-string bound.
en::Status Metadata(const fs::path& file, std::uint64_t size, std::string_view digest,
                    bool spellings) {
  std::error_code ec;
  if (fs::file_size(file, ec) != size || ec) return std::unexpected("metadata extent changed");
  std::FILE* in = std::fopen(file.c_str(), "rb");
  if (!in) return std::unexpected("metadata open failed");
  std::array<std::byte, 65536> buffer{};
  llmp::base::Sha256 hash;
  bool quoted = false, valid = true;
  std::size_t length = 0;
  std::uint64_t total = 0;
  while (const auto got = std::fread(buffer.data(), 1, buffer.size(), in)) {
    hash.Update(std::span(buffer).first(got));
    total += got;
    if (spellings)
      for (const auto byte : std::span(buffer).first(got)) {
        const char c = static_cast<char>(byte);
        if (c == '\\') valid = false;
        if (c == '"') {
          quoted = !quoted;
          length = 0;
        } else if (quoted && ++length > 192)
          valid = false;
      }
  }
  const bool failed = std::ferror(in) != 0;
  const auto closed = std::fclose(in);
  if (failed || closed || total != size || !valid || (spellings && quoted) ||
      llmp::base::ToHex(hash.Finish()) != digest)
    return std::unexpected("metadata hash or source-name bound changed");
  return {};
}
struct Record {
  pl::StepShape shape;
  ts::CorpusPlanRecord selected;
  bool measured = false;
  std::vector<std::uint64_t> parameters;
  bool parameterized = false;
};
struct Saved {
  std::uint32_t positions = 0;
  llmp::model::DraftProgress progress;
  std::vector<en::SlotLifecycle::Range> pages, footprint;
  void* data = nullptr;
  std::uint64_t bytes = 0;
  std::vector<std::int32_t> history;
};
struct Lifetime {
  en::PagedNode node{{.slot_bytes = en::kSlabSlotBytes}};
  std::array<Record, kClasses> records{};
  std::size_t count = 0;
  std::shared_ptr<void> keep;
  std::unique_ptr<en::ModelRunner> runner;
  en::StepSource* source = nullptr;
  std::vector<en::PagedModel*> entered;
  std::optional<en::WaveComposer> composer;
  std::optional<en::DecodingOrchestrator> decoder;
  en::DecodingMode mode;
  std::array<std::vector<std::int32_t>, 2> history;
  std::array<Saved, 2> saved;
  std::array<llmp::base::Sha256Digest, 2> final_state{};
  std::array<llmp::base::Sha256Digest, 3 * 2 * 3> reference{};
  std::array<std::uint32_t, 2> draft_pass{};
  std::unique_ptr<float[]> floats;
  std::vector<llmp::catalog::ExtentId> state_extents;
  void* readback = nullptr;
  std::uint64_t state_capacity = 0, finite_rows = 0, exact_rows = 0, changed_rows = 0;
  std::uint32_t unit = 0, pass = 0, completed_units = 0, draft_width = 0;
  std::uint32_t restored_owners = 0, spilled_owners = 0;
  std::uint64_t changed_unequal_runs = 0, changed_graph_replays = 0;
  bool warmup = true, retirement_attempted = false, retired = false;
  fs::path scratch;
  bool Retire() {
    if (retirement_attempted) return retired;
    // PagedNode records teardown entry before fencing. A failed first fence
    // must never be retried through its already-torn-down success path.
    retirement_attempted = true;
    auto result = node.TearDown(entered);
    if (!result) {
      ADD_FAILURE() << "retaining complete unknown MTP owner: " << result.error();
      return false;
    }
    retired = true;
    return true;
  }
};
struct RetainUnknown {
  void operator()(Lifetime* p) const {
    if (p->Retire()) delete p;
  }
};
using Owner = std::unique_ptr<Lifetime, RetainUnknown>;

class RecordedSource final : public en::StepSource {
 public:
  RecordedSource(std::unique_ptr<en::StepSource> source, Lifetime& life)
      : inner_(std::move(source)), life_(life) {}
  pl::PlanKey KeyBase() const override { return inner_->KeyBase(); }
  en::Status PrepareExecution() override { return inner_->PrepareExecution(); }
  en::Status CheckExecutionUse(en::NumericalUse use) const override {
    return inner_->CheckExecutionUse(use);
  }
  en::Status CheckExecutionShape(const pl::StepShape& s) const override {
    return inner_->CheckExecutionShape(s);
  }
  auto ExecutionIdentity() const
      -> std::expected<std::optional<llmp::base::Sha256Digest>, std::string> override {
    return inner_->ExecutionIdentity();
  }
  std::vector<std::uint64_t> ShapeParameters(const pl::StepShape& s) const override {
    return inner_->ShapeParameters(s);
  }
  bool JoinedRepeatPending(std::uint16_t stage) const override {
    return inner_->JoinedRepeatPending(stage);
  }
  bool UsesSideRows(std::uint16_t stage) const override { return inner_->UsesSideRows(stage); }
  en::Status SideRows(std::uint16_t stage, std::uint16_t component, std::span<const std::int32_t> h,
                      std::uint32_t p, std::span<std::int32_t> out) const override {
    return inner_->SideRows(stage, component, h, p, out);
  }
  auto CommitBuffers(std::uint16_t stage, std::uint32_t rows) const
      -> std::expected<std::vector<en::CommitBuffer>, std::string> override {
    return inner_->CommitBuffers(stage, rows);
  }
  auto Build(const pl::StepShape& shape, const kn::Addresses& a)
      -> std::expected<std::unique_ptr<kn::StepProgram>, std::string> override {
    auto records = std::span(life_.records).first(life_.count);
    auto it = std::ranges::find(records, shape, &Record::shape);
    if (it == records.end())
      return std::unexpected(std::format("unpaid MTP shape stage{} phase{} rows{} owners{}",
                                         shape.stage, static_cast<unsigned>(shape.phase),
                                         shape.rows(), shape.segments.size()));
    auto params = inner_->ShapeParameters(shape);
    if (params.size() > 64 || (it->parameterized && params != it->parameters))
      return std::unexpected("MTP shape parameters changed within a paid class");
    if (!it->parameterized) {
      it->parameters = std::move(params);
      it->parameterized = true;
    }
    if (!a.measuring() && !it->measured)
      return std::unexpected("MTP shape was not measured before bound construction");
    auto p = inner_->Build(shape, a);
    if (!p) return std::unexpected(p.error());
    if ((*p)->host_bytes() > kProgram || (*p)->launches() > 4096)
      return std::unexpected("MTP program exceeds retained payment");
    if (a.measuring()) {
      it->measured = true;
      return std::move(*p);
    }
    return std::make_unique<ts::CorpusProgram>(std::move(*p), it->selected);
  }

 private:
  std::unique_ptr<en::StepSource> inner_;
  Lifetime& life_;
};

pl::StepShape Shape(std::uint16_t stage, pl::Phase phase,
                    std::initializer_list<std::pair<std::uint8_t, std::uint32_t>> owners,
                    std::uint32_t outputs, std::uint32_t cells = 256) {
  pl::StepShape out{.stage = stage, .phase = phase, .segments = {}, .read_align = 256};
  for (const auto [slot, rows] : owners)
    out.segments.push_back({.slot = slot,
                            .rows = rows,
                            .past = 0,
                            .read_cells = cells,
                            .outputs = outputs,
                            .group = 0,
                            .want_head = outputs != 0});
  return out;
}
en::Status Add(Lifetime& life, const pl::StepShape& shape) {
  if (std::ranges::find(std::span(life.records).first(life.count), shape, &Record::shape) !=
      std::span(life.records).first(life.count).end())
    return {};
  if (life.count == life.records.size())
    return std::unexpected("paid MTP class inventory exceeded");
  auto& r = life.records[life.count++];
  r.shape = shape;
  r.selected.rows = shape.rows();
  return {};
}
en::Status SetUp(Lifetime& life) {
  auto& node = life.node;
  if (llmp::model::Qwen38Flash().layers != 48 || llmp::model::Qwen38Flash().experts_used != 10 ||
      ggml_tensor_overhead() > 512)
    return std::unexpected("compiler source/ABI construction bound changed");
  node.SetHostFloor(0);
  if (auto x = node.Open(); !x) return x;
  const auto occupied = node.catalog().OccupancyOf(node.domain()).Total().value();
  if (occupied > kDomain - kHost) return std::unexpected("host prepayment exceeds domain");
  auto paid = node.catalog().AddExtent({.domain = node.domain(),
                                        .memory_class = llmp::catalog::MemoryClass::kRuntime,
                                        .recovery = llmp::catalog::Recovery::kPinned,
                                        .size = llmp::base::Bytes(kHost),
                                        .content = {}},
                                       true);
  if (!paid) return std::unexpected("host construction prepayment refused");
  life.floats = std::make_unique<float[]>(kVocabulary);
  const auto store = fs::path(ts::ModelsDir()) / "m3-artifacts";
  for (const auto& [id, bytes, hash] : std::array{
           std::tuple{kTarget, 8152364ULL,
                      std::string_view{
                          "5b65dcce8169374e638264d7ebdcb5cca517234b3dec26f1272a5f4f9c0e4c89"}},
           std::tuple{kDrafter, 157885ULL,
                      std::string_view{
                          "5cd45fc5354ab224d281c2416027f224c61e32e2ae0acf8e7580d063fc274d99"}}}) {
    if (auto x = Metadata(store / id / "index.json", bytes, hash, true); !x) return x;
    if (auto x = Metadata(store / id / "manifest.json", id == kTarget ? 23037 : 1861, id, true); !x)
      return x;
    if (auto x =
            Metadata(store / id / "meta/config.json", 109897,
                     "2d655dfa625c3c018f9685031367bc83f8532d3faf3333d7c838b2bdc13b8a7e", false);
        !x)
      return x;
  }
  std::vector<en::NumericalStage> stages;
  for (std::uint16_t stage = 0; stage < 2; ++stage) {
    en::NumericalStage s{
        .stage = stage,
        .name = stage ? "mtp" : "target",
        .uses = {en::NumericalUse::kPrompt, en::NumericalUse::kDecode,
                 en::NumericalUse::kPromptCompanions, en::NumericalUse::kSpeculative,
                 en::NumericalUse::kJoinedRepeatPending},
        .phases = {}};
    const auto add = [&](pl::Phase p, std::uint32_t rows) {
      s.phases.push_back(
          {.phase = p, .max_owners = 2, .max_owner_rows = rows, .max_total_rows = 2ULL * rows});
    };
    if (stage)
      add(pl::Phase::kDraft, 16);
    else {
      add(pl::Phase::kPrefill, 16);
      add(pl::Phase::kDecode, 1);
      add(pl::Phase::kVerify, 3);
      add(pl::Phase::kCommit, 3);
    }
    stages.push_back(std::move(s));
  }
  auto domain = en::NumericalDomain::Create(std::move(stages));
  if (!domain) return std::unexpected(domain.error());
  rt::ModelSettings settings;
  settings.name = "joined-mtp";
  settings.architecture = "qwen4exp";
  settings.max_slots.value = 2;
  settings.context.value = 512;
  settings.prefill_chunk.value = 16;
  settings.wave_read_align.value = 256;
  settings.speculation.value = true;
  settings.draft_rows.value = 2;
  settings.draft_vocab.value = kDraftLimit;
  llmp::config::ModelEntry entry;
  entry.name = "joined-mtp";
  entry.artifact = kTarget;
  entry.drafter = kDrafter;
  rt::ComponentBuildOptions options{
      .target_policy = {},
      .diagnostic_solo = true,
      .prefill_lookahead = false,
      .prepare_state = false,
      .state_only_prefill = false,
      .execution_domain = std::make_shared<const en::NumericalDomain>(std::move(*domain))};
  options.validate_program = [&life](const kn::LowerInput& in,
                                     const kn::StepProgram& program) -> en::Status {
    const auto& shape = in.shape;
    if (shape.stage > 1 || shape.segments.size() > 2 || in.context != 512 ||
        in.step.saves.size() > kSaves)
      return std::unexpected("actual lowered source escapes paid owner/stage domain");
    if (shape.phase == pl::Phase::kCommit) return {};
    std::array<bool, 2> seen{};
    for (const auto& output : program.outputs()) {
      if (output.features) continue;
      if (output.segment >= shape.segments.size() || seen[output.segment])
        return std::unexpected("duplicate or foreign head owner");
      const auto& own = shape.segments[output.segment];
      if (own.greedy) continue;  // factory separately measures ordinary device-token variants
      if (!own.want_head || !in.step.logits.valid() || output.tensor != in.step.logits)
        return std::unexpected("full head is not the source's actual logits");
      const auto width = shape.stage ? life.draft_width : kVocabulary;
      const auto& tensor = in.step.graph.tensor(in.step.logits);
      if (tensor.format.encoding != llmp::format::Encoding::kF32 || tensor.shape().ne[0] != width ||
          output.bytes != std::uint64_t{width} * own.outputs * sizeof(float))
        return std::unexpected("actual full-head geometry differs");
      seen[output.segment] = true;
    }
    for (std::size_t i = 0; i < shape.segments.size(); ++i)
      if (shape.segments[i].want_head && !shape.segments[i].greedy && !seen[i])
        return std::unexpected("actual owner head was omitted");
    return {};
  };
  auto build = rt::ComponentLlmBuild(node, store, entry, 0, std::move(options));
  auto parts = build(settings);
  if (!parts) return std::unexpected(parts.error());
  if (parts->logit_width != kVocabulary || parts->config.slots.stages.size() != 2 ||
      parts->config.setup_shapes.size() > kFactoryClasses || parts->caps.max_segments < 2 ||
      parts->mode.drafters.size() != 1)
    return std::unexpected("factory left the paid target/MTP domain");
  const auto& ids = parts->mode.drafters[0].head_ids;
  auto width = en::DrafterHeadRows(parts->mode.drafters[0], kVocabulary);
  if (!width || *width > kDraftLimit || !ids || ids->size() != *width)
    return std::unexpected("factory omitted the actual selected-head vocabulary");
  if (auto valid = llmp::model::CheckQwen38DraftIds(*ids, kVocabulary); !valid)
    return std::unexpected(valid.error());
  life.draft_width = *width;  // Set before any runner setup/lowering.
  // Consume the factory's already-read immutable table, not another source
  // read or a prefix assumption. The source itself still owns device mapping.
  testing::Test::RecordProperty("draft_head_rows", std::to_string(*width));
  testing::Test::RecordProperty("draft_head_id_rows", std::to_string(ids->size()));
  testing::Test::RecordProperty("draft_head_ids_sha256",
                                llmp::base::ToHex(Hash(std::as_bytes(std::span(*ids)))));
  parts->mode.drafters[0].device_tokens = false;
  life.keep = parts->keep;
  parts->config.slots.spill_directory = life.scratch / "spill";
  for (const auto& s : parts->config.setup_shapes)
    if (auto x = Add(life, s); !x) return x;
  const auto actual = std::array{Shape(0, pl::Phase::kPrefill, {{0, 7}}, 1),
                                 Shape(0, pl::Phase::kPrefill, {{1, 5}}, 1),
                                 Shape(1, pl::Phase::kDraft, {{0, 6}}, 0),
                                 Shape(1, pl::Phase::kDraft, {{1, 4}}, 0),
                                 Shape(1, pl::Phase::kDraft, {{0, 1}, {1, 1}}, 1),
                                 Shape(1, pl::Phase::kDraft, {{0, 1}, {1, 3}}, 1),
                                 Shape(0, pl::Phase::kVerify, {{0, 3}, {1, 3}}, 3)};
  auto composition = en::WaveComposer::Make(
      parts->caps, std::vector<en::StageReads>{{.context = 512, .read_align = 256},
                                               {.context = 512, .read_align = 256}});
  if (!composition) return std::unexpected(composition.error());
  for (const auto& s : actual) {
    std::vector<en::PendingWork> pending;
    for (const auto& segment : s.segments)
      pending.push_back({.slot = segment.slot,
                         .stage = s.stage,
                         .phase = s.phase,
                         .past = segment.past,
                         .rows = segment.rows,
                         .outputs = segment.outputs,
                         .want_head = segment.want_head});
    auto composed = composition->Compose(pending);
    if (!composed.refused.empty() || composed.steps.size() != 1 ||
        composed.steps.front().shape.segments.size() != s.segments.size())
      return std::unexpected("actual batching cannot compose the closed fixture shape");
    if (auto x = Add(life, composed.steps.front().shape); !x) return x;
  }
  for (std::uint8_t slot = 0; slot < 2; ++slot)
    for (std::uint32_t rows = 1; rows <= 3; ++rows)
      if (auto x = Add(life, Shape(0, pl::Phase::kCommit, {{slot, rows}}, 0, 512)); !x) return x;
  parts->config.setup_shapes.clear();
  for (const auto& r : std::span(life.records).first(life.count))
    if (r.shape.phase != pl::Phase::kCommit) parts->config.setup_shapes.push_back(r.shape);
  // Setup measures slot-zero commits itself. Measure the same real slot-one
  // binding classes too, without changing actual setup/workspace accounting.
  for (const auto& r : std::span(life.records).first(life.count))
    if (r.shape.phase == pl::Phase::kCommit && r.shape.segments[0].slot == 1)
      parts->config.setup_shapes.push_back(r.shape);
  auto source = std::make_unique<RecordedSource>(std::move(parts->source), life);
  life.source = source.get();
  auto runner = en::ModelRunner::Create(node, std::move(parts->config), std::move(source), 0, 0);
  if (!runner) return std::unexpected(runner.error());
  life.runner = std::move(*runner);
  life.entered.push_back(life.runner.get());
  if (auto x = life.runner->Setup(); !x) return x;
  if (!life.source->JoinedRepeatPending(0) || !life.source->JoinedRepeatPending(1))
    return std::unexpected("real composition did not seal joined permission");
  if (!std::ranges::all_of(std::span(life.records).first(life.count),
                           [](const auto& r) { return r.measured; }))
    return std::unexpected("an actual class was not measured");
  const auto before = node.catalog().OccupancyOf(node.domain()).Total().value();
  const auto a = life.runner->activations_needed(),
             pool = std::max<std::uint64_t>(en::kPagedExtent, life.runner->pool_needed());
  if (before > kDomain || a > kDomain - before || pool > kDomain - before - a ||
      2 * en::kPagedExtent > kDomain - before - a - pool)
    return std::unexpected("workspace exceeds paid domain");
  // Catalog reservations already name the complete target+MTP places and
  // every state region, but no model payload has been read. Conservatively
  // require that whole inventory, real measured workspace and all three
  // state-copy buffers fit together before registration/first request.
  for (std::size_t stage = 0; stage < 2; ++stage) {
    const auto bytes = life.runner->slots().layout(stage).bytes;
    if (!bytes || bytes > 512 * kMiB || bytes % en::kPagedExtent)
      return std::unexpected("state copy capacity exceeds the admitted physical domain");
    life.state_capacity += bytes;
  }
  std::uint64_t whole = 0;
  for (const auto& [id, extent] : node.catalog().ExtentsOf(node.domain())) {
    (void)id;
    if (extent.descriptor.size.value() > kDomain - whole)
      return std::unexpected("complete catalog inventory exceeds domain before weights");
    whole += extent.descriptor.size.value();
  }
  for (const auto bytes :
       std::array<std::uint64_t, 4>{a, pool, 3 * life.state_capacity, 2 * en::kPagedExtent}) {
    if (bytes > kDomain - whole)
      return std::unexpected(
          "whole weight/state/workspace/copy inventory does not fit before weights");
    whole += bytes;
  }
  testing::Test::RecordProperty("complete_admitted_physical_bytes", std::to_string(whole));
  if (auto x = node.MapWorkspace(a, pool); !x) return x;
  if (auto x = node.Start(llmp::base::Bytes(kDomain)); !x) return x;
  if (auto x = life.runner->Register(); !x) return x;
  if (auto x = life.runner->Bind(); !x) return x;
  node.Run();
  if (auto x = life.runner->slots().SelectSlots(std::array{0U, 1U}); !x) return x;
  auto composer = en::WaveComposer::For(parts->caps, life.runner->slots());
  if (!composer) return std::unexpected(composer.error());
  life.composer.emplace(std::move(*composer));
  auto identity = life.source->ExecutionIdentity();
  if (!identity || !*identity) return std::unexpected("missing real backend execution identity");
  testing::Test::RecordProperty("execution_identity", llmp::base::ToHex(**identity));
  testing::Test::RecordProperty("measured_classes", std::to_string(life.count));
  // Policy below consumes only completed copies and owns no backend authority.
  // Decoder consumes this factory's exact mode.
  life.decoder.reset();
  life.mode = std::move(parts->mode);
  return {};
}

std::expected<std::int32_t, std::string> Head(Lifetime& life, std::span<const std::byte> bytes,
                                              std::uint32_t width, std::uint32_t rows) {
  if (!rows || !width || width > kVocabulary ||
      bytes.size() < std::uint64_t{width} * rows * sizeof(float))
    return std::unexpected("completed head coverage is incomplete");
  std::int32_t best = 0;
  for (std::uint32_t row = 0; row < rows; ++row) {
    std::memcpy(life.floats.get(), bytes.data() + std::uint64_t{row} * width * sizeof(float),
                width * sizeof(float));
    auto values = std::span(life.floats.get(), width);
    if (!std::ranges::all_of(values, [](float x) { return std::isfinite(x); }))
      return std::unexpected("nonfinite complete head");
    best = static_cast<std::int32_t>(std::ranges::max_element(values) - values.begin());
    ++life.finite_rows;
  }
  return best;
}
en::Status Witness(Lifetime& life, std::uint32_t slot, std::uint32_t part,
                   std::span<const std::byte> head, std::uint32_t rows) {
  if (life.warmup) return {};
  if (life.unit >= 3 || slot >= 2 || part >= 3)
    return std::unexpected("head witness coordinate overflow");
  const auto index = life.unit * 6 + slot * 3 + part;
  auto digest = Hash(head);
  if (life.pass == 0)
    life.reference[index] = digest;
  else if (life.pass < 3) {
    if (digest != life.reference[index])
      return std::unexpected("same-selected restored head differs");
    life.exact_rows += rows;
  } else if (digest != life.reference[index])
    life.changed_rows += rows;
  return {};
}
en::Status Decoder(Lifetime& life) {
  en::DecodingPolicy policy;
  policy.propose =
      [&](std::uint32_t slot, std::size_t drafter, std::span<const std::byte> bytes,
          std::uint32_t want) -> std::expected<std::vector<std::int32_t>, std::string> {
    if (slot >= 2 || drafter != 0 || want != 1 || life.draft_pass[slot] >= 2)
      return std::unexpected("drafter contribution coordinate differs");
    auto token = Head(life, bytes, life.draft_width, 1);
    if (!token) return std::unexpected(token.error());
    auto checked = Witness(life, slot, life.draft_pass[slot]++,
                           bytes.first(life.draft_width * sizeof(float)), 1);
    if (!checked) return std::unexpected(checked.error());
    auto mapped = en::DrafterHeadToken(life.mode.drafters[drafter],
                                       static_cast<std::uint32_t>(*token), kVocabulary);
    if (!mapped) return std::unexpected(mapped.error());
    return std::vector<std::int32_t>{*mapped};
  };
  policy.judge = [&](std::uint32_t slot, std::span<const std::int32_t> verified,
                     std::span<const std::byte> bytes) -> std::expected<en::Verdict, std::string> {
    if (slot >= 2 || verified.size() != 3 || life.draft_pass[slot] != 2 ||
        bytes.size() != 3ULL * kVocabulary * sizeof(float))
      return std::unexpected("verify omits a selected draft contribution or full head");
    auto last = Head(life, bytes, kVocabulary, 3);
    if (!last) return std::unexpected(last.error());
    if (auto x = Witness(life, slot, 2, bytes, 3); !x) return std::unexpected(x.error());
    const std::uint32_t keep = slot == 0 ? 1 : 3;
    std::vector<std::int32_t> tokens(verified.begin() + 1, verified.begin() + keep);
    std::memcpy(life.floats.get(),
                bytes.data() + std::uint64_t{keep - 1} * kVocabulary * sizeof(float),
                kVocabulary * sizeof(float));
    const auto values = std::span(life.floats.get(), kVocabulary);
    tokens.push_back(static_cast<std::int32_t>(std::ranges::max_element(values) - values.begin()));
    return en::Verdict{.keep = keep, .tokens = std::move(tokens)};
  };
  auto decoder =
      en::DecodingOrchestrator::Make(*life.runner, *life.composer, life.mode, std::move(policy));
  if (!decoder) return std::unexpected(decoder.error());
  life.decoder.emplace(std::move(*decoder));
  return {};
}
en::Status Prompt(Lifetime& life) {
  auto& slots = life.runner->slots();
  for (std::uint32_t slot = 0; slot < 2; ++slot) {
    const std::uint32_t rows = slot == 0 ? 7 : 5;
    for (std::uint32_t i = 0; i < rows; ++i)
      life.history[slot].push_back(static_cast<std::int32_t>(17 + slot * 100 + i));
    en::PendingWork pending{.slot = slot, .phase = pl::Phase::kPrefill, .rows = rows, .outputs = 1};
    en::SegmentWork work{.tokens = life.history[slot]};
    const std::array<std::span<const std::int32_t>, 1> history{life.history[slot]};
    auto report = en::RunPromptWave(*life.runner, *life.composer, std::span(&pending, 1),
                                    std::span(&work, 1), history);
    if (report.outcomes.size() != 1) return std::unexpected("prompt result count differs");
    if (report.outcomes[0].refusal) return std::unexpected(report.outcomes[0].refusal->what);
    auto token = Head(life, report.outcomes[0].outputs, kVocabulary, 1);
    if (!token) return std::unexpected(token.error());
    life.history[slot].push_back(*token);
    if (slots.positions(slot) != rows) return std::unexpected("prompt frontier differs");
  }
  return {};
}
en::Status Unit(Lifetime& life) {
  auto& slots = life.runner->slots();
  std::array<en::DecodeWork, 2> work{};
  for (std::uint32_t slot = 0; slot < 2; ++slot) {
    const auto* progress = slots.progress(slot, 1);
    auto bounds = llmp::model::DraftBounds(slots.layout(1));
    if (!progress || !bounds || !*bounds) return std::unexpected("real pending progress missing");
    auto first = llmp::model::DraftFirstPass(progress->completed(), **bounds);
    if (!first || first->count != (life.warmup ? 1U : (slot == 0 ? 1U : 3U)))
      return std::unexpected("actual unequal pending schedule differs");
    if (life.history[slot].size() != slots.positions(slot) + 1)
      return std::unexpected("history/frontier mismatch");
    work[slot] = {.slot = slot, .history = life.history[slot]};
  }
  life.draft_pass = {};
  const auto report = life.decoder->Step(work);
  if (!report.speculated || report.outcomes.size() != 2)
    return std::unexpected("joined unit did not speculate");
  for (std::uint32_t slot = 0; slot < 2; ++slot) {
    const auto& out = report.outcomes[slot];
    if (out.refusal) return std::unexpected(out.refusal->what);
    if (out.drafts.size() != 1 || out.drafts[0].refusal || out.drafts[0].proposed != 2 ||
        out.drafts[0].steps != 2 || out.kept != (slot == 0 ? 1U : 3U) ||
        out.tokens.size() != out.kept || life.draft_pass[slot] != 2)
      return std::unexpected("joined unit omitted a full selected proposal/verdict");
    life.history[slot].insert(life.history[slot].end(), out.tokens.begin(), out.tokens.end());
    if (auto x = slots.Settle(slot); !x) return x;
  }
  ++life.completed_units;
  return {};
}
std::vector<en::SlotLifecycle::Range> Footprint(en::SlotLifecycle& slots, std::uint32_t slot) {
  std::vector<en::SlotLifecycle::Range> out;
  for (std::size_t stage = 0; stage < slots.stages(); ++stage)
    for (auto r : slots.live(slot, stage).used_ranges()) {
      r.region = stage;
      out.push_back(r);
    }
  return out;
}
en::Status Save(Lifetime& life) {
  auto& slots = life.runner->slots();
  if (!life.state_capacity || life.state_capacity > 1024 * kMiB)
    return std::unexpected("missing pre-admitted state buffer capacity");
  auto admitted = life.node.Call(
      [&]() -> en::Status {
        const auto used = life.node.catalog().OccupancyOf(life.node.domain()).Total().value();
        if (used > kDomain || life.state_capacity > (kDomain - used) / 3)
          return std::unexpected("three state buffers exceed paid domain");
        return {};
      },
      "joined MTP state copy payment");
  if (!admitted) return admitted;
  for (auto& saved : life.saved) {
    auto p = life.node.Pinned(life.state_capacity, 0, life.state_extents);
    if (!p) return std::unexpected(p.error());
    saved.data = *p;
  }
  auto p = life.node.Pinned(life.state_capacity, 0, life.state_extents);
  if (!p) return std::unexpected(p.error());
  life.readback = *p;
  for (std::uint32_t slot = 0; slot < 2; ++slot) {
    auto& saved = life.saved[slot];
    saved.positions = slots.positions(slot);
    saved.progress = slots.progress(slot, 1)->completed();
    saved.history = life.history[slot];
    saved.footprint = Footprint(slots, slot);
    auto ranges = slots.CheckpointRanges(slot, saved.positions);
    if (!ranges) return std::unexpected(ranges.error());
    saved.pages = std::move(*ranges);
    if (saved.pages.size() > 1024 || saved.footprint.size() > 1024)
      return std::unexpected("state range inventory exceeded");
    for (const auto& r : saved.pages) {
      if (r.region >= 2 || r.bytes > life.state_capacity - saved.bytes)
        return std::unexpected("checkpoint exceeds paid copy");
      saved.bytes += r.bytes;
    }
    if (auto x = slots.CopyState(slot, saved.data, saved.pages, true); !x) return x;
  }
  return {};
}
en::Status Restore(Lifetime& life, bool spill) {
  auto& slots = life.runner->slots();
  for (std::uint32_t slot = 0; slot < 2; ++slot) {
    const auto& saved = life.saved[slot];
    if (auto x =
            slots.PrepareRestore(slot, saved.positions, saved.footprint, slots.kept_layout_id());
        !x)
      return x;
    if (auto x = slots.CopyState(slot, saved.data, saved.pages, false); !x) return x;
    if (auto x = slots.CompleteRestore(slot, saved.positions); !x) return x;
    if (spill) {
      if (auto x = slots.Spill(slot); !x) return x;
      if (!slots.spilled(slot)) return std::unexpected("state did not spill");
      if (auto x = slots.Restore(slot); !x) return x;
      ++life.spilled_owners;
    }
    life.decoder->Reset(slot);
    life.history[slot] = saved.history;
    if (slots.positions(slot) != saved.positions ||
        slots.progress(slot, 1)->completed() != saved.progress)
      return std::unexpected("restored positions/pending feature coordinates differ");
    auto footprint = Footprint(slots, slot);
    if (footprint.size() != saved.footprint.size())
      return std::unexpected("restored used footprint differs");
    for (std::size_t i = 0; i < footprint.size(); ++i) {
      const auto a = footprint[i], b = saved.footprint[i];
      if (a.region != b.region || a.offset != b.offset || a.bytes != b.bytes)
        return std::unexpected("restored used ranges differ");
    }
    if (auto x = slots.CopyState(slot, life.readback, saved.pages, true); !x) return x;
    if (std::memcmp(saved.data, life.readback, saved.bytes))
      return std::unexpected("restored initialized bytes differ");
    ++life.restored_owners;
  }
  return {};
}
Record* Unequal(Lifetime& life) {
  auto records = std::span(life.records).first(life.count);
  auto at = std::ranges::find_if(records, [](const auto& r) {
    return r.shape.stage == 1 && r.shape.phase == pl::Phase::kDraft &&
           r.shape.segments.size() == 2 && r.shape.segments[0].rows == 1 &&
           r.shape.segments[1].rows == 3;
  });
  return at == records.end() ? nullptr : &*at;
}
en::Status FinalState(Lifetime& life) {
  auto& slots = life.runner->slots();
  for (std::uint32_t slot = 0; slot < 2; ++slot) {
    if (slots.positions(slot) !=
        life.saved[slot].positions + (life.pass == 3 ? 6U : 3U) * (slot == 0 ? 1U : 3U))
      return std::unexpected("final owner frontier differs");
    auto ranges = Footprint(slots, slot);
    if (ranges.size() > 1024) return std::unexpected("final range inventory exceeded");
    std::uint64_t bytes = 0;
    for (const auto& r : ranges) {
      if (r.bytes > life.state_capacity - bytes)
        return std::unexpected("final copy exceeds paid extent");
      bytes += r.bytes;
    }
    if (auto x = slots.CopyState(slot, life.readback, ranges, true); !x) return x;
    auto digest =
        Hash({static_cast<const std::byte*>(life.readback), static_cast<std::size_t>(bytes)});
    if (life.pass == 0)
      life.final_state[slot] = digest;
    else if (life.pass < 3 && life.final_state[slot] != digest)
      return std::unexpected("same-selected final initialized state differs");
  }
  return {};
}
TEST(Qwen38JoinedMtpGpu, OwnedUnequalPendingRestoresAndChangedInputCapture) {
  const auto available = llmp::platform::AvailableMemoryBytes();
  ASSERT_TRUE(available);
  ASSERT_GE(*available, kDomain + rt::kUncountedMargin);
  Owner life(new Lifetime);
  life->scratch = Env("LLMP_JOINED_MTP_SCRATCH");
  ASSERT_FALSE(life->scratch.empty());
  std::error_code ec;
  ASSERT_TRUE(fs::create_directory(life->scratch, ec));
  ASSERT_FALSE(ec);
  auto setup = ::SetUp(*life);
  ASSERT_TRUE(setup) << Failed(setup).value_or("");
  auto decoder = Decoder(*life);
  ASSERT_TRUE(decoder) << Failed(decoder).value_or("");
  auto result = life->node.WithRequest(
      0, life->runner->slots().execution(), "owned joined MTP trajectory", [&]() -> en::Status {
        if (auto x = Prompt(*life); !x) return x;
        if (auto x = Unit(*life); !x) return x;  // keep1/3 creates actual unequal pending spans
        life->warmup = false;
        if (auto x = Save(*life); !x) return x;
        for (life->pass = 0; life->pass < 4; ++life->pass) {
          if (life->pass == 1 || life->pass == 2) {
            if (auto x = Restore(*life, life->pass == 2); !x) return x;
          }
          // Continue from the completed replay without invalidating its plans.
          // Changed anchors must use the already-retained unequal capture.
          if (life->pass == 3) {
            auto* record = Unequal(*life);
            if (!record || !record->selected.runs)
              return std::unexpected("unequal selected program was not run");
            life->changed_unequal_runs = record->selected.runs;
            life->changed_graph_replays = life->runner->graph_stats().replayed;
            for (auto& history : life->history)
              history.back() = (history.back() + 1) % static_cast<std::int32_t>(kVocabulary);
          }
          for (life->unit = 0; life->unit < 3; ++life->unit)
            if (auto x = Unit(*life); !x) return x;
          if (auto x = FinalState(*life); !x) return x;
        }
        return {};
      });
  ASSERT_TRUE(result) << Failed(result).value_or("");
  EXPECT_EQ(life->completed_units, 13U);
  EXPECT_EQ(life->exact_rows, 60U);
  EXPECT_EQ(life->finite_rows, 132U);
  EXPECT_EQ(life->restored_owners, 4U);
  EXPECT_EQ(life->spilled_owners, 2U);
  EXPECT_GT(life->changed_rows, 0U);
  EXPECT_EQ(life->runner->coverage().violations, 0U);
  EXPECT_GT(life->runner->graph_stats().captured, 0U);
  EXPECT_GT(life->runner->graph_stats().replayed, 0U);
  const auto* record = Unequal(*life);
  ASSERT_NE(record, nullptr);
  EXPECT_GT(record->selected.runs, 0U);
  EXPECT_EQ(record->selected.runs, life->changed_unequal_runs)
      << "changed-anchor unequal drafts must replay the retained capture, not rerun eagerly";
  EXPECT_GT(life->runner->graph_stats().replayed, life->changed_graph_replays);
  EXPECT_FALSE(record->selected.implementations.empty());
  std::string plans;
  for (const auto& r : std::span(life->records).first(life->count)) {
    if (r.shape.segments.size() != 2 || r.selected.implementations.empty()) continue;
    const auto header = std::format(
        "stage={} phase={} rows={}/{} identity={} count={}\n", r.shape.stage,
        static_cast<unsigned>(r.shape.phase), r.shape.segments[0].rows, r.shape.segments[1].rows,
        llmp::base::ToHex(r.selected.identity), r.selected.implementations.size());
    if (header.size() > 2 * kMiB - plans.size()) FAIL() << "program report exceeds paid bound";
    plans += header;
    for (const auto& name : r.selected.implementations) {
      if (name.size() + 1 > 2 * kMiB - plans.size()) FAIL() << "program report exceeds paid bound";
      plans += name;
      plans += '\n';
    }
  }
  testing::Test::RecordProperty("actual_joined_programs", plans);
  testing::Test::RecordProperty("units", 13);
  testing::Test::RecordProperty("reference_head_rows", 30);
  testing::Test::RecordProperty("restored_head_rows", std::to_string(life->exact_rows));
  testing::Test::RecordProperty("changed_head_rows", std::to_string(life->changed_rows));
  testing::Test::RecordProperty("finite_head_rows", std::to_string(life->finite_rows));
  testing::Test::RecordProperty("restored_owners", std::to_string(life->restored_owners));
  testing::Test::RecordProperty("spilled_owners", std::to_string(life->spilled_owners));
  testing::Test::RecordProperty("host_payment_bytes", std::to_string(kHost));
  testing::Test::RecordProperty("compiler_construction_bytes", std::to_string(kCompiler));
  testing::Test::RecordProperty("state_copy_capacity", std::to_string(life->state_capacity));
  testing::Test::RecordProperty("unequal_bound_identity",
                                llmp::base::ToHex(record->selected.identity));
  testing::Test::RecordProperty("captured", std::to_string(life->runner->graph_stats().captured));
  testing::Test::RecordProperty("replayed", std::to_string(life->runner->graph_stats().replayed));
  EXPECT_TRUE(life->Retire());
}
}  // namespace
