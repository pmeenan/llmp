// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Bounded target-only shared scoring of the fixed authenticated Qwen52 corpus
// on the published v1 ModelOpt target (d145900e, from the v1 store).
// Source weights and histories match the independent quantized reference.
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <expected>
#include <filesystem>
#include <format>
#include <iomanip>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <tuple>
#include <utility>
#include <vector>

#include "base/sha256.h"
#include "compiler_inventory.h"
#include "corpus_score.h"
#include "engine/component_source.h"
#include "engine/decoding.h"
#include "engine/model_runner.h"
#include "engine/paged_weights.h"
#include "engine/prefill.h"
#include "kernels/ggml/graph_plan.h"
#include "planner/plan.h"
#include "platform/host_probe.h"
#include "providers/device_runtime.h"
#include "runtime/component_build.h"
#include "runtime/memory_guard.h"
#include "spec/mappings/qwen38.h"
#include "v1_frozen.h"

namespace {
namespace fs = std::filesystem;
namespace en = llmp::engine;
namespace kn = llmp::kernels;
namespace pl = llmp::planner;
namespace rt = llmp::runtime;
namespace ts = llmp::test_support;
namespace ir = llmp::ir;
constexpr std::uint32_t kVocabulary = 248320;
constexpr std::uint32_t kOwners = 4;
constexpr std::string_view kIdsHash =
    "3cd946c3876501eccbe079b9d2700e542e66242be241a8549546df8d9b23e468";
std::string Env(const char* name) {
  const char* value = std::getenv(name);  // NOLINT(concurrency-mt-unsafe)
  return value ? value : "";
}
std::string Hash(std::span<const std::byte> bytes) {
  llmp::base::Sha256 hash;
  hash.Update(bytes);
  return llmp::base::ToHex(hash.Finish());
}
std::string Hash(std::string_view text) { return Hash(std::as_bytes(std::span(text))); }
std::expected<std::string, std::string> ReadExact(const fs::path& path, std::uint64_t size,
                                                  std::string_view digest) {
  std::error_code error;
  if (fs::file_size(path, error) != size || error || size > 13 * 1024 * 1024U)
    return std::unexpected("input file extent differs from its fixed receipt");
  std::FILE* file = std::fopen(path.c_str(), "rb");
  if (!file) return std::unexpected("cannot open authenticated input");
  std::string data(static_cast<std::size_t>(size), '\0');
  const auto got = std::fread(data.data(), 1, data.size(), file);
  const auto extra = std::fgetc(file);
  const bool failed = std::ferror(file) != 0;
  const auto closed = std::fclose(file);
  if (got != data.size() || extra != EOF || failed || closed || Hash(data) != digest)
    return std::unexpected("input file bytes differ from its fixed receipt");
  return data;
}

constexpr std::uint64_t kMiB = std::uint64_t{1} << 20U;
constexpr std::uint64_t kDomainBytes = std::uint64_t{104} << 30U;
constexpr std::uint64_t kProgramBytes = 64 * kMiB;
constexpr std::size_t kFactoryShapes = 16, kShapes = kFactoryShapes + 4;
// Current-source construction bound, independent of completed program limits.
// The reviewed categories use Builder's enforced N/T limits,
// the two rewrite rounds, serial fallback passes and Lower's descriptor count.
// Metadata names are checked before source construction below. No per-class
// multiplier applies to transient graphs: Setup builds one class at a time.
constexpr std::uint64_t kNodes = ir::Graph::kMaxNodes;
constexpr std::uint64_t kTensors = ir::Graph::kMaxTensors;
constexpr std::uint64_t kGraphConstruction =
    llmp::test_support::compiler_inventory::GraphBytes(kNodes, kTensors);
constexpr std::uint64_t kIrWorksets = 4096 * (kNodes + kTensors);
// Two disjoint rewrite rounds, four outputs; top-ten ExpertReduce adds23.
constexpr std::uint64_t kNativeDescriptors = kTensors + 16 + 8 * kNodes + 23 * kNodes;
// Arena512, native planning/index/copy work1024, traversal<4 pointers each.
constexpr std::uint64_t kNativeConstruction =
    llmp::test_support::compiler_inventory::NativeBytes(kNativeDescriptors, kTensors);
constexpr std::uint64_t kCompilerEnvelope =
    kGraphConstruction + kIrWorksets + kNativeConstruction + 64 * kMiB;
constexpr std::uint64_t kRetainedPrograms =
    (kShapes + 1) *
    (kProgramBytes + 4096 * en::kGraphNodeHostBytes);  // Includes one conservative spare program.
constexpr std::uint64_t kMetadataEnvelope = 1024 * kMiB;
constexpr std::uint64_t kObservationEnvelope = 128 * kMiB;
// NumericalProof owns its observation descriptors. The canonical compiler
// inventory includes their actual sizeof and overlap; pay that sum rather
// than retaining the obsolete rounded 9GiB fixture ceiling.
constexpr std::uint64_t kHostRequired =
    kCompilerEnvelope + kRetainedPrograms + kMetadataEnvelope + kObservationEnvelope;
constexpr std::uint64_t kHostBytes =
    ((kHostRequired + 2 * kMiB - 1) / (2 * kMiB)) * (2 * kMiB);
static_assert(kCompilerEnvelope + kRetainedPrograms + kMetadataEnvelope + kObservationEnvelope <=
              kHostBytes);
// The old 8GiB factory payment is smaller than the updated type inventory.
// Round only this actual sum to the catalog's 2MiB extent granularity.
constexpr std::uint64_t kFactoryRequired =
    kCompilerEnvelope + (kFactoryShapes + 1) * (kProgramBytes + 4096 * en::kGraphNodeHostBytes) +
    kMetadataEnvelope + kObservationEnvelope;
constexpr std::uint64_t kFactoryHostBytes =
    ((kFactoryRequired + 2 * kMiB - 1) / (2 * kMiB)) * (2 * kMiB);
static_assert(kFactoryRequired <= kFactoryHostBytes && kFactoryHostBytes <= kHostBytes);
static_assert(ir::Graph::kMaxNodes == 65536 && ir::Graph::kMaxTensors == 131072);
static_assert(sizeof(ir::Node) <= 160 && sizeof(ir::TensorDesc) <= 160);
static_assert(sizeof(pl::Selected) <= 128);
static_assert(ir::kMaxSources == 10 && ir::kMaxOutputs == 4);
static_assert(sizeof(llmp::kernels::ggml::PlanStep) <= 64);
static_assert(sizeof(std::string) <= 32 && sizeof(en::SlotLifecycle::Range) <= 24);
static_assert(kShapes * (4096 * (32 + 129) + 64 * 1024) + 2 * 8 * kMiB +
                  (3 * 8 + 1) * kVocabulary * sizeof(float) + 4 * 3 * 512 * 24 + 4096 + 64 * 1024 +
                  2 * 512 * 1024 + 2 * 64 * 1024 <
              kObservationEnvelope);

constexpr std::string_view kArtifact =
    "d145900e058ff6470405c396affb9879b5c3c3770cc2b586d846e7e148c2058d";

bool AllowedShape(const pl::StepShape& shape) {
  if (shape.stage || shape.row_invariant || shape.read_align != 256 ||
      (shape.segments.size() != 1 && shape.segments.size() != 4))
    return false;
  const auto phase = shape.phase;
  if (phase != pl::Phase::kScore && phase != pl::Phase::kPrefill && phase != pl::Phase::kDecode)
    return false;
  for (std::size_t i = 0; i < shape.segments.size(); ++i) {
    const auto& segment = shape.segments[i];
    if (segment.slot >= kOwners || (shape.segments.size() == 4 && segment.slot != i) ||
        (phase != pl::Phase::kScore && shape.segments.size() == 1 && segment.slot) ||
        segment.group || !segment.want_head || segment.past > 512 ||
        segment.rows > 512 - segment.past)
      return false;
    if (phase == pl::Phase::kScore) {
      if ((segment.rows != 1 && segment.rows != 2) || segment.outputs != segment.rows ||
          segment.greedy || segment.read_cells != 256)
        return false;
    } else {
      if (segment.rows != (phase == pl::Phase::kPrefill ? 2U : 1U) || segment.outputs != 1 ||
          (segment.read_cells != 512 && (segment.read_cells != 256 || segment.greedy)) ||
          (phase == pl::Phase::kPrefill && segment.greedy))
        return false;
    }
    if (i && (segment.rows != shape.segments[0].rows || segment.greedy != shape.segments[0].greedy))
      return false;
  }
  return true;
}

enum class TeacherPolicy {
  kJoined,
  kTunedJoined,
  kDefaultSingleRow,
  kJoinedSingleRow,
  kDefaultOrdinary,
  kJoinedOrdinary
};
bool OrdinaryPolicy(TeacherPolicy policy) {
  return policy == TeacherPolicy::kDefaultOrdinary || policy == TeacherPolicy::kJoinedOrdinary;
}
bool DefaultPolicy(TeacherPolicy policy) {
  return policy == TeacherPolicy::kDefaultSingleRow || policy == TeacherPolicy::kDefaultOrdinary;
}
bool TunedPolicy(TeacherPolicy policy) { return policy == TeacherPolicy::kTunedJoined; }
std::uint32_t ScoreChunk(TeacherPolicy policy) {
  return policy == TeacherPolicy::kDefaultSingleRow || policy == TeacherPolicy::kJoinedSingleRow
             ? 1U
             : 2U;
}
std::string_view PolicyName(TeacherPolicy policy) {
  switch (policy) {
    case TeacherPolicy::kJoined:
      return "A-no-tunes";
    case TeacherPolicy::kTunedJoined:
      return "C-three-physical8-tunes";
    case TeacherPolicy::kDefaultSingleRow:
      return "mapping-default-single-row";
    case TeacherPolicy::kJoinedSingleRow:
      return "owner-policy-disabled-single-row";
    case TeacherPolicy::kDefaultOrdinary:
      return "mapping-default-ordinary";
    case TeacherPolicy::kJoinedOrdinary:
      return "owner-policy-disabled-ordinary";
  }
  return {};
}

struct ShapeRecord {
  pl::StepShape shape;
  std::vector<std::uint64_t> parameters;
  bool parameterized = false;
  bool measured = false, lowered = false;
  bool owner_policy = false, owner_invariant = false;
  ts::CorpusPlanRecord selected;
  // Planned surviving dense products: node, weight/input IDs, physical K/N/rows,
  // encodings, strides and binding. This is a descriptor receipt, distinct
  // from the selected bound implementation inventory recorded at Bind.
  std::string dense;
};

class RecordedSource final : public en::StepSource {
 public:
  RecordedSource(std::unique_ptr<en::StepSource> inner, std::span<ShapeRecord> records)
      : inner_(std::move(inner)), records_(records) {}
  pl::PlanKey KeyBase() const override { return inner_->KeyBase(); }
  en::Status PrepareExecution() override { return inner_->PrepareExecution(); }
  en::Status CheckExecutionUse(en::NumericalUse use) const override {
    return inner_->CheckExecutionUse(use);
  }
  en::Status CheckExecutionShape(const pl::StepShape& shape) const override {
    return inner_->CheckExecutionShape(shape);
  }
  std::expected<std::optional<llmp::base::Sha256Digest>, std::string> ExecutionIdentity()
      const override {
    return inner_->ExecutionIdentity();
  }
  void EmitShapeParameters(const pl::StepShape& shape,
                           llmp::model::ShapeParameterSink& sink) const override {
    inner_->EmitShapeParameters(shape, sink);
  }
  std::expected<en::SpecializationChange, en::SpecializationError> NextSpecializationChange(
      const en::SpecializationRequest& request) const override {
    return inner_->NextSpecializationChange(request);
  }

  bool JoinedRepeatPending(std::uint16_t stage) const override {
    return inner_->JoinedRepeatPending(stage);
  }
  bool UsesSideRows(std::uint16_t stage) const override { return inner_->UsesSideRows(stage); }
  en::Status SideRows(std::uint16_t stage, std::uint16_t component,
                      std::span<const std::int32_t> history, std::uint32_t position,
                      std::span<std::int32_t> out) const override {
    return inner_->SideRows(stage, component, history, position, out);
  }
  std::expected<std::vector<en::CommitBuffer>, std::string> CommitBuffers(
      std::uint16_t stage, std::uint32_t rows) const override {
    return inner_->CommitBuffers(stage, rows);
  }
  std::expected<std::unique_ptr<kn::StepProgram>, std::string> Build(
      const pl::StepShape& shape, const kn::Addresses& addresses) override {
    auto found = std::ranges::find_if(records_, [&](const auto& r) { return r.shape == shape; });
    if (found == records_.end()) {
      std::ostringstream description;
      description << "unpaid score/setup shape phase=" << static_cast<unsigned>(shape.phase)
                  << " stage=" << shape.stage << " align=" << shape.read_align
                  << " invariant=" << shape.row_invariant << " owners=" << shape.segments.size();
      for (const auto& segment :
           std::span(shape.segments).first(std::min<std::size_t>(4, shape.segments.size())))
        description << " owner=" << static_cast<unsigned>(segment.slot) << ':' << segment.rows
                    << ':' << segment.outputs << ':' << segment.read_cells << ':'
                    << static_cast<unsigned>(segment.group) << ':' << segment.want_head << ':'
                    << segment.greedy;
      return std::unexpected(description.str().substr(0, 1024));
    }
    auto parameters = inner_->ShapeParameters(shape);
    if (parameters.size() > 64 || (found->parameterized && found->parameters != parameters))
      return std::unexpected("unpaid position-dependent shape parameters");
    if (!found->parameterized) {
      found->parameters = std::move(parameters);
      found->parameterized = true;
    }
    auto program = inner_->Build(shape, addresses);
    if (!program) return std::unexpected(program.error());
    if ((*program)->host_bytes() > kProgramBytes || (*program)->launches() > 4096)
      return std::unexpected("program exceeds prepaid retained inventory");
    if (addresses.measuring()) return std::move(*program);
    return std::make_unique<ts::CorpusProgram>(std::move(*program), found->selected);
  }

 private:
  std::unique_ptr<en::StepSource> inner_;
  std::span<ShapeRecord> records_;
};

constexpr std::uint32_t kQualityRows = 255;
struct QualityWitness {
  std::array<std::int32_t, kOwners * kQualityRows> ids{};
  std::array<ts::CorpusTargetScore, kOwners*(kQualityRows - 1)> scores{};
  std::array<bool, kOwners * kQualityRows> seen{};
  std::uint32_t document_rows = 0, waves = 0;
  std::size_t score_count = 0;
};
static_assert(sizeof(QualityWitness) <= 64 * 1024);

constexpr std::uint32_t kPromptRows = 32, kDecodeWaves = 8, kCheckpointWave = 4;
constexpr std::uint64_t kStateLimit = 512 * kMiB;
struct SavedOwner {
  std::vector<en::SlotLifecycle::Range> pages, footprint;
  std::uint32_t position = 0;
  std::uint64_t bytes = 0;
};
struct OrdinaryWitness {
  std::array<SavedOwner, kOwners> saved{};
  std::array<void*, kOwners> buffers{};
  void* readback = nullptr;
  std::vector<llmp::catalog::ExtentId> extents;
  std::uint64_t capacity = 0;
  std::array<std::array<std::int32_t, kPromptRows + kDecodeWaves + 1>, kOwners> history{};
  decltype(history) checkpoint_history{};
  std::array<std::string, kOwners * kDecodeWaves> heads{};
  std::array<std::int32_t, kOwners * kDecodeWaves> tokens{};
  std::array<std::string, kOwners> final_state{};
  std::unique_ptr<float[]> row;
  std::uint32_t prompt_heads = 0, prompt_waves = 0, decode_waves = 0, reference_rows = 0,
                compared_rows = 0;
  std::uint32_t checkpoint_restores = 0, spill_restores = 0, final_state_checks = 0;
};
static_assert(sizeof(OrdinaryWitness) < 16 * 1024);

bool retirement_failed = false;

struct Lifetime {
  TeacherPolicy policy = TeacherPolicy::kJoined;
  en::PagedNode node{{.slot_bytes = en::kSlabSlotBytes}};
  std::array<ShapeRecord, kShapes> records{};
  std::size_t shape_count = kFactoryShapes;
  std::unique_ptr<OrdinaryWitness> ordinary;
  std::unique_ptr<QualityWitness> quality;  // Outlives scorer callbacks and runner.
  std::shared_ptr<void> keep;
  std::unique_ptr<en::ModelRunner> runner;
  en::StepSource* source = nullptr;
  std::vector<en::PagedModel*> entered;
  std::optional<en::WaveComposer> composer;
  std::unique_ptr<ts::CorpusScore> scorer;
  fs::path scratch;
  bool retirement_attempted = false, retired = false;
  bool Retire() {
    if (retirement_attempted) return retired;
    retirement_attempted = true;
    auto finished = node.TearDown(entered);
    if (!finished) {
      retirement_failed = true;
      ADD_FAILURE() << "unknown retirement; retaining complete joined scorer owner: "
                    << finished.error();
      return false;
    }
    retired = true;
    return true;
  }
};
struct RetainUnknown {
  void operator()(Lifetime* life) const {
    if (life->Retire()) delete life;
  }
};
using Owner = std::unique_ptr<Lifetime, RetainUnknown>;

en::Status CheckOrdinaryHead(const kn::LowerInput& input, const kn::StepProgram& program) {
  const auto& step = input.token->step;
  const auto owners = input.token->shape.segments.size();
  if (!step.logits.valid() || step.logits.v >= step.graph.tensor_count() || step.greedy.valid() ||
      step.features.valid() || !step.saves.empty() || program.outputs().size() != owners)
    return std::unexpected("ordinary operation has no complete original owner heads");
  const auto& logits = step.graph.tensor(step.logits);
  if (logits.format.encoding != llmp::format::Encoding::kF32 ||
      logits.shape() != ir::ShapeOf({kVocabulary, owners}) || !ir::IsContiguous(logits))
    return std::unexpected("ordinary original head is not packed F32 full vocabulary");
  std::array<const kn::OutputSlot*, kOwners> outputs{};
  for (const auto& out : program.outputs()) {
    if (out.segment >= owners || out.features || outputs[out.segment] || out.tensor != step.logits)
      return std::unexpected("ordinary outputs reorder or replace owner heads");
    outputs[out.segment] = &out;
  }
  auto address = outputs[0]->device_address;
  constexpr auto bytes = std::uint64_t{kVocabulary} * sizeof(float);
  for (const auto* out : std::span(outputs).first(owners)) {
    if (!out || out->bytes != bytes || out->device_address != address ||
        address > std::numeric_limits<std::uint64_t>::max() - bytes)
      return std::unexpected("ordinary packed owner head coverage differs");
    address += bytes;
  }
  return {};
}

en::Status ValidateProgram(Lifetime& life, const kn::LowerInput& input,
                           const kn::StepProgram& program) {
  if (!input.token) return std::unexpected("scorer requires actual token metadata");
  const bool policy_enabled = DefaultPolicy(life.policy);
  const bool invariant =
      policy_enabled && input.token->shape.segments.size() > 1 &&
      std::ranges::all_of(input.token->shape.segments, [](const auto& s) { return s.rows == 1; });
  if (!AllowedShape(input.token->shape) || input.token->context != 512 ||
      input.token->shape.stage != 0 || input.plan.owner_row_invariant != invariant ||
      input.plan.owner_rows_policy_enabled != policy_enabled || input.token->shape.row_invariant ||
      !input.plan.owner_products.empty() || input.token->step.weights.size() > 4096 ||
      input.token->step.state.size() > 4096 || input.token->step.inputs.size() > 256 ||
      !input.token->step.saves.empty())
    return std::unexpected("scorer source/state/policy exceeds closed paid domain");
  auto found = std::ranges::find_if(std::span(life.records).first(life.shape_count),
                                    [&](const auto& r) { return r.shape == input.token->shape; });
  if (found == std::span(life.records).first(life.shape_count).end())
    return std::unexpected("lowering escaped paid classes");
  found->owner_policy = policy_enabled;
  found->owner_invariant = invariant;
  if (input.token->shape.phase == pl::Phase::kScore) {
    if (auto valid = ts::CheckCorpusHead(input, program, kVocabulary); !valid) return valid;
  } else if (input.token->shape.segments.front().read_cells == 256) {
    if (!OrdinaryPolicy(life.policy)) return std::unexpected("unpaid ordinary short class");
    if (auto valid = CheckOrdinaryHead(input, program); !valid) return valid;
  }
  if (program.host_bytes() > kProgramBytes || program.launches() > 4096)
    return std::unexpected("lowering exceeds paid program inventory");
  if (input.addresses.measuring())
    found->measured = true;
  else
    found->lowered = true;
  // Dense metadata is collected only once per paid shape, including remainder4.
  // The real immutable owner and lowering remain the only dispatch authority.
  if (found->dense.empty()) {
    std::ostringstream out;
    std::size_t products = 0;
    for (const auto& selected : input.plan.steps) {
      for (auto id : selected.nodes) {
        const auto& node = input.plan.graph.node(id);
        if (node.key != ir::Primitive(ir::Op::kMatMul) || node.nsrc != 2 || node.nout != 1)
          continue;
        const auto& weight = input.plan.graph.tensor(node.src[0]);
        const auto& activation = input.plan.graph.tensor(node.src[1]);
        const auto& result = input.plan.graph.tensor(node.out[0]);
        if (weight.format.encoding != llmp::format::Encoding::kBF16 ||
            activation.format.encoding != llmp::format::Encoding::kF32)
          continue;
        if (++products > 256) return std::unexpected("dense product receipt count exceeds payment");
        out << id.v << ',' << node.src[0].v << ',' << node.src[1].v << ',' << node.out[0].v << ','
            << activation.ne(1) << ',' << weight.ne(0) << ',' << weight.ne(1) << ','
            << selected.implementation << ',' << static_cast<unsigned>(result.format.encoding);
        for (const auto* tensor : {&weight, &activation, &result})
          for (const auto stride : tensor->nb) out << ',' << stride;
        const auto& root = input.plan.graph.tensor(input.plan.graph.root(node.src[0]));
        if (const auto* ref = std::get_if<ir::WeightRef>(&root.binding))
          out << ",weight," << ref->stage << ',' << ref->component << ',' << ref->role << ','
              << ref->layer << ',' << ref->expert_array;
        else
          out << ",nonweight-root," << root.binding.index() << ','
              << static_cast<unsigned>(root.role);
        out << '\n';
      }
    }
    if (out.tellp() > 64 * 1024)
      return std::unexpected("dense receipt serialization exceeds payment");
    found->dense = out.str();
  }
  return {};
}

std::expected<rt::ComponentBuildOptions, std::string> Options(Lifetime& life) {
  auto policy = pl::ParsePolicy(llmp::spec::mappings::Qwen38Mapping().policies.front().text);
  if (!policy) return std::unexpected(policy.error());
  if (std::ranges::count(policy->enabled, "owner_row_invariant") != 1 ||
      std::ranges::find(policy->disabled, "owner_row_invariant") != policy->disabled.end())
    return std::unexpected("native mapping owner policy changed");
  if (!DefaultPolicy(life.policy)) {
    std::erase(policy->enabled, "owner_row_invariant");
    policy->disabled.emplace_back("owner_row_invariant");
  }
  pl::Canonicalize(*policy);
  auto domain = en::NumericalDomain::Create(
      {{.stage = 0,
        .name = "target",
        .uses = OrdinaryPolicy(life.policy)
                    ? std::vector{en::NumericalUse::kPrompt, en::NumericalUse::kDecode,
                                  en::NumericalUse::kScore}
                    : std::vector{en::NumericalUse::kDecode, en::NumericalUse::kScore},
        .phases = {{.phase = pl::Phase::kPrefill,
                    .max_owners = 4,
                    .max_owner_rows = 2,
                    .max_total_rows = 8},
                   {.phase = pl::Phase::kDecode,
                    .max_owners = 4,
                    .max_owner_rows = 1,
                    .max_total_rows = 4},
                   {.phase = pl::Phase::kScore,
                    .max_owners = 4,
                    .max_owner_rows = 2,
                    .max_total_rows = 8}}}});
  if (!domain) return std::unexpected(domain.error());
  rt::ComponentBuildOptions options{
      .target_policy = pl::ToText(*policy),
      .prefill_lookahead = false,
      .prepare_state = false,
      .state_only_prefill = false,
      .execution_domain = std::make_shared<const en::NumericalDomain>(std::move(*domain)),
      .tuning = {},
      .execution_metadata_cap_bytes = kMetadataEnvelope,
      .validate_program = [&life](const kn::LowerInput& input, const kn::StepProgram& program) {
        return ValidateProgram(life, input, program);
      }};
  if (TunedPolicy(life.policy)) {
    const auto device = llmp::providers::QueryDeviceFacts(0);
    if (!device) return std::unexpected(std::string(device.error().text()));
    // Same provider-to-planner conversion as GgmlBackend::Create. The factory
    // still derives its own actual immutable backend facts; this owns tunes.
    const auto arch = device->architecture / 10;
    pl::DeviceFacts tuning{.device_class = std::format("sm_{}", arch), .arch = arch};
    for (std::uint64_t n : {512U, 10240U, 2560U})
      tuning.tuned.push_back({.implementation = "llmp.mul_mat.mmvf_rows",
                              .rows = 8,
                              .k = 2560,
                              .n = n,
                              .output_encoding = llmp::format::Encoding::kF32});
    auto bound = pl::OwnedDeviceFacts::Bound(tuning);
    if (!bound) return std::unexpected("tuning metadata: " + bound.error());
    if (*bound > kMiB) return std::unexpected("tuning metadata exceeds prepaid allowance");
    auto owned = pl::OwnedDeviceFacts::Create(tuning, *bound);
    if (!owned) return std::unexpected(owned.error());
    options.tuning = std::make_shared<const pl::OwnedDeviceFacts>(std::move(*owned));
  }
  testing::Test::RecordProperty("effective_policy", *options.target_policy);
  testing::Test::RecordProperty("effective_policy_sha256", llmp::base::ToHex(pl::Digest(*policy)));
  // This diagnostic pays/validates its exact source inventory. These are
  // additional measurements, independent of the normal serving schedule.
  for (const auto rows : {1U, 2U}) {
    pl::StepShape scored{.stage = 0,
                         .phase = pl::Phase::kScore,
                         .segments = {},
                         .row_invariant = false,
                         .read_align = 256};
    for (std::uint32_t owner = 0; owner < kOwners; ++owner)
      scored.segments.push_back({.slot = static_cast<std::uint8_t>(owner),
                                 .rows = rows,
                                 .past = 256 - rows,
                                 .read_cells = 256,
                                 .outputs = rows,
                                 .group = 0,
                                 .want_head = true,
                                 .greedy = false});
    options.diagnostic_setup_shapes.push_back(scored);
    for (const auto& segment : scored.segments) {
      auto single = scored;
      single.segments = {segment};
      options.diagnostic_setup_shapes.push_back(std::move(single));
    }
  }
  return options;
}

// The v0 canonical parser forbids escapes. Bound every source spelling before
// parsing/synthesis can copy it into a graph; generated state/input labels in
// this pinned48-layer source add less than64 bytes. This scan allocates nothing.
bool BoundedSourceSpellings(std::string_view text) {
  bool quoted = false;
  std::size_t length = 0;
  for (const char c : text) {
    if (c == '\\') return false;
    if (c == '"') {
      quoted = !quoted;
      length = 0;
    } else if (quoted && ++length > 192) {
      return false;
    }
  }
  return !quoted;
}

en::Status SetUp(Lifetime& life) {
  auto& node = life.node;
  // All construction-domain prerequisites precede the first graph. The source
  // manifest/index/config hashes below bind the artifact.
  if (ggml_tensor_overhead() > 512)
    return std::unexpected("compiler construction domain or descriptor ABI changed");
  node.SetHostFloor(0);
  if (auto open = node.Open(); !open) return open;
  const auto host_bytes = OrdinaryPolicy(life.policy) ? kHostBytes : kFactoryHostBytes;
  const auto occupied = node.catalog().OccupancyOf(node.domain()).Total().value();
  if (occupied > kDomainBytes - host_bytes) return std::unexpected("startup payment does not fit");
  auto paid = node.catalog().AddExtent({.domain = node.domain(),
                                        .memory_class = llmp::catalog::MemoryClass::kRuntime,
                                        .recovery = llmp::catalog::Recovery::kPinned,
                                        .size = llmp::base::Bytes(host_bytes),
                                        .content = {}},
                                       true);
  if (!paid) return std::unexpected("startup catalog payment refused");
  const fs::path artifact = fs::path(ts::V1Store()) / kArtifact;
  for (const auto& file : std::array{
           std::tuple{"manifest.json", std::uint64_t{48495}, kArtifact},
           std::tuple{"index.json", std::uint64_t{8236769},
                      std::string_view{
                          "2b29dc0f905bd670522abda9d8ef5e17f0db328c1d4fcd9f8dae63e9079dc5dd"}},
           std::tuple{"meta/config.json", std::uint64_t{109897},
                      std::string_view{
                          "2d655dfa625c3c018f9685031367bc83f8532d3faf3333d7c838b2bdc13b8a7e"}}}) {
    auto verified = ReadExact(artifact / std::get<0>(file), std::get<1>(file), std::get<2>(file));
    if (!verified) return std::unexpected(verified.error());
    if (std::string_view(std::get<0>(file)) != "meta/config.json" &&
        !BoundedSourceSpellings(*verified))
      return std::unexpected("artifact source spellings exceed compiler payment");
  }
  testing::Test::RecordProperty("compiler_construction_bytes", std::to_string(kCompilerEnvelope));
  testing::Test::RecordProperty("compiler_descriptor_limit", std::to_string(kNativeDescriptors));
  auto options = Options(life);
  if (!options) return std::unexpected(options.error());
  rt::ModelSettings settings;
  settings.architecture = "qwen4exp";
  settings.container = "safetensors";
  settings.max_slots.value = 4;
  settings.context.value = 512;
  settings.prefill_chunk.value = 2;
  settings.wave_read_align.value = 256;
  settings.speculation.value = false;
  llmp::config::ModelEntry entry;
  entry.name = ScoreChunk(life.policy) == 1 ? "qwen-policy-single"
                                            : (TunedPolicy(life.policy) ? "qwen50-c" : "qwen50-a");
  entry.artifact = std::string(kArtifact);
  const fs::path store = fs::path(ts::V1Store());
  auto build = rt::ComponentLlmBuild(node, store, entry, 0, std::move(*options));
  auto parts = build(settings);
  if (!parts) return std::unexpected(parts.error());
  if (parts->logit_width != kVocabulary || parts->config.slots.stages.size() != 1 ||
      parts->config.setup_shapes.size() != kFactoryShapes || parts->caps.max_segments < kOwners ||
      !parts->config.slots.stages[0].layout.bytes)
    return std::unexpected("actual target parts exceed the declared scoring domain");
  if (OrdinaryPolicy(life.policy)) {
    // These are new measurements, not reuse of the larger read512 result.
    for (const auto phase : {pl::Phase::kPrefill, pl::Phase::kDecode}) {
      for (const std::size_t owners : {std::size_t{1}, std::size_t{kOwners}}) {
        const auto from = std::ranges::find_if(parts->config.setup_shapes, [&](const auto& shape) {
          return shape.phase == phase && shape.segments.size() == owners &&
                 !shape.segments.front().greedy && shape.segments.front().read_cells == 512;
        });
        if (from == parts->config.setup_shapes.end())
          return std::unexpected("factory omitted ordinary full-head shape template");
        auto shape = *from;
        for (auto& segment : shape.segments) {
          segment.read_cells = 256;
          segment.past = 256 - segment.rows;
        }
        if (!AllowedShape(shape) || std::ranges::find(parts->config.setup_shapes, shape) !=
                                        parts->config.setup_shapes.end())
          return std::unexpected("ordinary short class is not distinct and supported");
        parts->config.setup_shapes.push_back(std::move(shape));
      }
    }
    life.shape_count = kShapes;
  }
  life.keep = parts->keep;
  parts->config.slots.model = entry.name;
  parts->config.slots.slots = kOwners;
  parts->config.slots.spill_directory = life.scratch / "spill";
  for (std::size_t i = 0; i < life.shape_count; ++i) {
    const auto& shape = parts->config.setup_shapes[i];
    if (!AllowedShape(shape))
      return std::unexpected("factory setup shape escapes closed inventory");
    if (std::ranges::find_if(std::span(life.records).first(i), [&](const auto& r) {
          return r.shape == shape;
        }) != std::span(life.records).first(i).end())
      return std::unexpected("duplicate setup shape substituted for paid coverage");
    life.records[i].shape = shape;
    life.records[i].selected.rows = shape.rows();
  }
  auto source = std::make_unique<RecordedSource>(std::move(parts->source),
                                                 std::span(life.records).first(life.shape_count));
  life.source = source.get();
  auto made = en::ModelRunner::Create(node, std::move(parts->config), std::move(source), 0, 0);
  if (!made) return std::unexpected(made.error());
  life.runner = std::move(*made);
  life.entered.push_back(life.runner.get());
  if (auto setup = life.runner->Setup(); !setup) return setup;
  auto identity = life.source->ExecutionIdentity();
  if (!identity || !*identity) return std::unexpected("owned execution identity was not sealed");
  testing::Test::RecordProperty("execution_identity", llmp::base::ToHex(**identity));
  testing::Test::RecordProperty("kept_layout_id", life.runner->slots().kept_layout_id());
  if (!std::ranges::all_of(std::span(life.records).first(life.shape_count),
                           [](const auto& r) { return r.measured; }))
    return std::unexpected("not all actual setup classes were measured");
  const auto pool = std::max<std::uint64_t>(life.runner->pool_needed(), en::kPagedExtent);
  const auto before = node.catalog().OccupancyOf(node.domain()).Total().value();
  const auto activation = life.runner->activations_needed();
  if (before > kDomainBytes || activation > kDomainBytes - before ||
      pool > kDomainBytes - before - activation ||
      2 * en::kPagedExtent > kDomainBytes - before - activation - pool)
    return std::unexpected("workspace exceeds the prepaid physical domain");
  if (auto mapped = node.MapWorkspace(activation, pool); !mapped) return mapped;
  if (auto start = node.Start(llmp::base::Bytes(kDomainBytes)); !start) return start;
  if (auto registration = life.runner->Register(); !registration) return registration;
  if (auto bound = life.runner->Bind(); !bound) return bound;
  node.Run();
  if (auto selected = life.runner->slots().SelectSlots(std::array<std::uint32_t, 4>{0, 1, 2, 3});
      !selected)
    return selected;
  auto composer = en::WaveComposer::For(parts->caps, life.runner->slots());
  if (!composer) return std::unexpected(composer.error());
  life.composer.emplace(std::move(*composer));
  return {};
}

en::Status ScoreQuality(Lifetime& life) {
  auto& observed = *life.quality;
  const auto rows = observed.document_rows;
  if (rows != kQualityRows)
    return std::unexpected("quality schedule is not the frozen larger odd prefix");
  std::array<ts::CorpusDocument, kOwners> documents{};
  for (std::uint32_t owner = 0; owner < kOwners; ++owner)
    documents[owner] = {.slot = owner,
                        .tokens = std::span(observed.ids).subspan(owner * rows, rows)};
  auto score = ts::CorpusScore::Make(
      life.node, *life.runner, *life.composer, documents,
      {.vocabulary = kVocabulary, .chunk_rows = ScoreChunk(life.policy)},
      [&observed, rows](const ts::CorpusTargetScore& row) -> en::Status {
        if (observed.score_count >= kOwners * (rows - 1) || row.document != row.slot ||
            row.slot >= kOwners || !row.position || row.position >= rows ||
            row.target != observed.ids[row.slot * rows + row.position])
          return std::unexpected("larger NLL receipt coordinates changed");
        observed.scores[observed.score_count++] = row;
        return {};
      },
      [&observed, rows](std::uint32_t document, std::uint32_t slot, std::uint32_t position,
                        std::span<const float> head) -> en::Status {
        if (slot != document || slot >= kOwners || position >= rows || head.size() != kVocabulary ||
            observed.seen[slot * rows + position])
          return std::unexpected("larger finite full-head receipt coordinates changed");
        observed.seen[slot * rows + position] = true;
        return {};
      });
  if (!score) return std::unexpected(score.error());
  life.scorer = std::move(*score);
  observed.waves = (rows + ScoreChunk(life.policy) - 1) / ScoreChunk(life.policy);
  auto completed = life.node.WithRequest(
      0, life.runner->slots().execution(), "Qwen52 larger fixed natural teacher",
      [&]() -> en::Status {
        for (std::uint32_t step = 0; step < observed.waves; ++step) {
          auto done = life.scorer->Step();
          if (!done) return std::unexpected(done.error());
          if (*done != (step + 1 == observed.waves))
            return std::unexpected("larger teacher completed at another frozen cutoff");
        }
        return {};
      });
  if (!completed) return completed;
  if (!life.scorer->complete() || life.scorer->result().inputs != kOwners * rows ||
      life.scorer->result().targets != kOwners * (rows - 1) ||
      life.scorer->result().steps != observed.waves ||
      observed.score_count != kOwners * (rows - 1) ||
      !std::ranges::all_of(std::span(observed.seen).first(kOwners * rows),
                           [](bool x) { return x; }))
    return std::unexpected("larger teacher full input/target/head coverage differs");
  for (std::uint32_t owner = 0; owner < kOwners; ++owner)
    if (life.runner->slots().positions(owner) != rows)
      return std::unexpected("larger teacher final initialized frontier differs");
  return {};
}

en::Status OrdinaryBuffers(Lifetime& life) {
  auto& state = *life.ordinary;
  auto& slots = life.runner->slots();
  if (slots.stages() != 1 || !slots.layout(0).bytes || slots.layout(0).bytes > kStateLimit)
    return std::unexpected("ordinary state layout exceeds bounded capacity");
  state.capacity = slots.layout(0).bytes;
  auto admitted = life.node.Call(
      [&]() -> en::Status {
        const auto occupied = life.node.catalog().OccupancyOf(life.node.domain()).Total().value();
        if (occupied > kDomainBytes || state.capacity > (kDomainBytes - occupied) / 5)
          return std::unexpected("five ordinary state buffers exceed paid physical domain");
        return {};
      },
      "ordinary policy state-buffer payment");
  if (!admitted) return admitted;
  for (auto& buffer : state.buffers) {
    auto allocated = life.node.Pinned(state.capacity, 0, state.extents);
    if (!allocated) return std::unexpected(allocated.error());
    buffer = *allocated;
  }
  auto allocated = life.node.Pinned(state.capacity, 0, state.extents);
  if (!allocated) return std::unexpected(allocated.error());
  state.readback = *allocated;
  state.row = std::make_unique<float[]>(kVocabulary);  // Existing observation payment.
  return {};
}

en::Status SaveOrdinary(Lifetime& life) {
  auto& state = *life.ordinary;
  auto& slots = life.runner->slots();
  state.checkpoint_history = state.history;
  for (std::uint32_t owner = 0; owner < kOwners; ++owner) {
    auto& saved = state.saved[owner];
    saved.position = slots.positions(owner);
    saved.footprint = slots.live(owner, 0).used_ranges();
    auto ranges = slots.CheckpointRanges(owner, saved.position);
    if (!ranges) return std::unexpected(ranges.error());
    saved.pages = std::move(*ranges);
    if (saved.pages.empty() || saved.pages.size() > 512 || saved.footprint.size() > 512)
      return std::unexpected("ordinary state ranges exceed paid inventory");
    saved.bytes = 0;
    for (const auto& range : saved.pages) {
      if (range.region || !range.bytes || range.bytes > state.capacity - saved.bytes)
        return std::unexpected("ordinary checkpoint exceeds paid copy capacity");
      saved.bytes += range.bytes;
    }
    if (auto copy = slots.CopyState(owner, state.buffers[owner], saved.pages, true); !copy)
      return copy;
  }
  return {};
}

en::Status CheckOrdinaryState(Lifetime& life, bool final, bool compare) {
  auto& state = *life.ordinary;
  auto& slots = life.runner->slots();
  for (std::uint32_t owner = 0; owner < kOwners; ++owner) {
    const auto& saved = state.saved[owner];
    const auto footprint = slots.live(owner, 0).used_ranges();
    const auto expected = final ? kPromptRows + (owner == 0 ? 8U : 4U) : saved.position;
    if (slots.positions(owner) != expected || slots.spilled(owner) || slots.restoring(owner) ||
        footprint.empty() || footprint.size() > 512)
      return std::unexpected("ordinary state frontier/footprint differs");
    if (!final) {
      if (footprint.size() != saved.footprint.size())
        return std::unexpected("restored initialized footprint size differs");
      for (std::size_t i = 0; i < footprint.size(); ++i)
        if (footprint[i].region != saved.footprint[i].region ||
            footprint[i].offset != saved.footprint[i].offset ||
            footprint[i].bytes != saved.footprint[i].bytes)
          return std::unexpected("restored initialized footprint coordinates differ");
    }
    const auto& ranges = final ? footprint : saved.pages;
    std::uint64_t bytes = 0;
    for (const auto& range : ranges) {
      if (range.region || !range.bytes || range.bytes > state.capacity - bytes)
        return std::unexpected("ordinary state witness exceeds paid readback");
      bytes += range.bytes;
    }
    if (auto copy = slots.CopyState(owner, state.readback, ranges, true); !copy) return copy;
    const auto data =
        std::span(static_cast<const std::byte*>(state.readback), static_cast<std::size_t>(bytes));
    if (final) {
      llmp::base::Sha256 digest;
      for (const auto& range : ranges) {
        const std::array<std::uint64_t, 3> coordinates{range.region, range.offset, range.bytes};
        digest.Update(std::as_bytes(std::span(coordinates)));
      }
      digest.Update(data);
      const auto hash = llmp::base::ToHex(digest.Finish());
      if (compare && hash != state.final_state[owner])
        return std::unexpected("same-selected final initialized state differs");
      if (compare)
        ++state.final_state_checks;
      else
        state.final_state[owner] = hash;
    } else if (std::memcmp(state.buffers[owner], state.readback, static_cast<std::size_t>(bytes))) {
      return std::unexpected("restored initialized checkpoint bytes differ");
    }
  }
  return {};
}

en::Status OrdinarySchedule(Lifetime& life) {
  life.ordinary = std::make_unique<OrdinaryWitness>();
  if (auto paid = OrdinaryBuffers(life); !paid) return paid;
  auto& state = *life.ordinary;
  auto& slots = life.runner->slots();
  auto head = [&](std::span<const std::byte> bytes) -> std::expected<std::int32_t, std::string> {
    if (bytes.size() != kVocabulary * sizeof(float))
      return std::unexpected("ordinary head is not complete F32 vocabulary");
    std::memcpy(state.row.get(), bytes.data(), bytes.size());
    const std::span row(state.row.get(), kVocabulary);
    if (std::ranges::any_of(row, [](float x) { return !std::isfinite(x); }))
      return std::unexpected("ordinary head contains a nonfinite value");
    return static_cast<std::int32_t>(std::ranges::max_element(row) - row.begin());
  };
  return life.node.WithRequest(
      0, slots.execution(), "Qwen ordinary owner policy", [&]() -> en::Status {
        en::PrefillOrchestrator prompt(*life.runner, *life.composer,
                                       {.chunk_rows = 2,
                                        .lookahead = 0,
                                        .capture_ahead = false,
                                        .prepare_ahead = false,
                                        .state_only = false});
        for (const std::uint32_t owners : {1U, kOwners}) {
          for (std::uint32_t owner = 0; owner < owners; ++owner)
            std::copy_n(life.quality->ids.begin() + owner * kQualityRows, kPromptRows,
                        state.history[owner].begin());
          for (std::uint32_t at = 0; at < kPromptRows; at += 2) {
            std::array<en::PromptWork, kOwners> work{};
            for (std::uint32_t owner = 0; owner < owners; ++owner)
              work[owner] = {
                  .slot = owner,
                  .stage = 0,
                  .tokens = std::span(state.history[owner]).subspan(at, kPromptRows - at),
                  .past = at,
                  .want_head = true,
                  .all_outputs = false,
                  .boundary = 0,
                  .history = std::span(state.history[owner]).first(kPromptRows)};
            const auto report = prompt.Step(std::span(work).first(owners));
            if (report.steps != 1 || report.outcomes().size() != owners)
              return std::unexpected("ordinary prompt did not run one requested cohort");
            ++state.prompt_waves;
            for (std::uint32_t owner = 0; owner < owners; ++owner) {
              const auto& out = report.outcomes()[owner];
              if (out.refusal || out.rows != 2 || out.last != (at + 2 == kPromptRows) ||
                  !out.joined.empty() || slots.positions(owner) != at + 2)
                return std::unexpected("ordinary prompt owner completion differs");
              auto next = head(out.outputs);
              if (!next) return std::unexpected(next.error());
              ++state.prompt_heads;
              if (at + 2 == kPromptRows) state.history[owner][kPromptRows] = *next;
            }
          }
          if (owners == 1) {
            if (auto clear = slots.Clear(0); !clear) return clear;
          }
        }
        auto made = en::DecodingOrchestrator::Make(*life.runner, *life.composer,
                                                   {.kind = en::DecodingMode::Kind::kPlain}, {});
        if (!made) return std::unexpected(made.error());
        auto decoder = std::move(*made);
        auto wave = [&](std::uint32_t step, bool compare) -> en::Status {
          const auto owners = step % 2 == 0 ? 1U : kOwners;
          std::array<en::DecodeWork, kOwners> work{};
          for (std::uint32_t owner = 0; owner < owners; ++owner) {
            const auto past = kPromptRows + (owner == 0 ? step : step / 2);
            if (slots.positions(owner) != past)
              return std::unexpected("ordinary decode history frontier differs");
            work[owner] = {.slot = owner,
                           .history = std::span(state.history[owner]).first(past + 1),
                           .left = kDecodeWaves - step,
                           .greedy = false};
          }
          const auto report = decoder.Step(std::span(work).first(owners));
          if (report.steps != 1 || report.outcomes.size() != owners || report.speculated)
            return std::unexpected("ordinary decode did not run one plain requested cohort");
          ++state.decode_waves;
          for (std::uint32_t owner = 0; owner < owners; ++owner) {
            const auto& out = report.outcomes[owner];
            if (out.refusal || out.discarded || !out.joined.empty() || !out.tokens.empty() ||
                out.kept != 1)
              return std::unexpected("ordinary decode completion differs");
            auto next = head(out.outputs);
            if (!next) return std::unexpected(next.error());
            const auto index = step * kOwners + owner;
            const auto hash = Hash(std::as_bytes(std::span(state.row.get(), kVocabulary)));
            if (compare) {
              if (hash != state.heads[index] || *next != state.tokens[index])
                return std::unexpected("same-selected ordinary replay changed head/token");
              ++state.compared_rows;
            } else {
              state.heads[index] = hash;
              state.tokens[index] = *next;
              ++state.reference_rows;
            }
            const auto past = kPromptRows + (owner == 0 ? step : step / 2);
            if (slots.positions(owner) != past + 1)
              return std::unexpected("ordinary decode did not commit its row");
            state.history[owner][past + 1] = *next;
          }
          return {};
        };
        for (std::uint32_t step = 0; step < kDecodeWaves; ++step) {
          if (step == kCheckpointWave) {
            if (auto save = SaveOrdinary(life); !save) return save;
          }
          if (auto done = wave(step, false); !done) return done;
        }
        if (auto final = CheckOrdinaryState(life, true, false); !final) return final;
        for (const bool spill : {false, true}) {
          for (std::uint32_t owner = 0; owner < kOwners; ++owner) {
            const auto& saved = state.saved[owner];
            if (auto x = slots.PrepareRestore(owner, saved.position, saved.footprint,
                                              slots.kept_layout_id());
                !x)
              return x;
            if (auto x = slots.CopyState(owner, state.buffers[owner], saved.pages, false); !x)
              return x;
            if (auto x = slots.CompleteRestore(owner, saved.position); !x) return x;
            ++state.checkpoint_restores;
            decoder.Reset(owner);
          }
          state.history = state.checkpoint_history;
          if (auto check = CheckOrdinaryState(life, false, true); !check) return check;
          if (spill) {
            for (std::uint32_t owner = 0; owner < kOwners; ++owner) {
              if (auto x = slots.Spill(owner); !x) return x;
              if (!slots.spilled(owner)) return std::unexpected("ordinary state did not spill");
              if (auto x = slots.Restore(owner); !x) return x;
              ++state.spill_restores;
            }
            if (auto check = CheckOrdinaryState(life, false, true); !check) return check;
          }
          for (std::uint32_t step = kCheckpointWave; step < kDecodeWaves; ++step)
            if (auto done = wave(step, true); !done) return done;
          if (auto final = CheckOrdinaryState(life, true, true); !final) return final;
        }
        return {};
      });
}

void Screen(TeacherPolicy policy) {
  ASSERT_FALSE(retirement_failed) << "prior owner retirement remains unknown";
  if (ts::V1IdNamed("qwen38-modelopt") != kArtifact)
    GTEST_FAIL() << "no Qwen3.8 ModelOpt v1 target in " << ts::V1Store();
  const auto available = llmp::platform::AvailableMemoryBytes();
  ASSERT_TRUE(available);
  ASSERT_GE(*available, kDomainBytes + rt::kUncountedMargin);
  Owner life(new Lifetime);
  life->policy = policy;
  life->scratch = Env("LLMP_QWEN50_SCRATCH");
  ASSERT_FALSE(life->scratch.empty());
  std::error_code error;
  ASSERT_TRUE(fs::create_directory(life->scratch, error));
  ASSERT_FALSE(error);
  const auto id_sha = Env("LLMP_QWEN50_IDS_SHA256");
  ASSERT_EQ(id_sha, kIdsHash);
  constexpr auto document_rows = kQualityRows;
  auto ids =
      ReadExact(Env("LLMP_QWEN50_IDS"), kOwners * document_rows * sizeof(std::int32_t), id_sha);
  ASSERT_TRUE(ids) << ids.error();
  static_assert(std::endian::native == std::endian::little);
  for (std::size_t i = 0; i < kOwners * document_rows; ++i) {
    std::int32_t id = 0;
    std::memcpy(&id, ids->data() + i * sizeof(id), sizeof(id));
    ASSERT_GE(id, 0);
    ASSERT_LT(static_cast<std::uint32_t>(id), kVocabulary);
  }
  auto setup = SetUp(*life);
  ASSERT_TRUE(setup) << setup.error();
  // This witness is in the startup observation allowance admitted by SetUp.
  life->quality = std::make_unique<QualityWitness>();
  life->quality->document_rows = document_rows;
  std::memcpy(life->quality->ids.data(), ids->data(), ids->size());
  const auto start = std::chrono::steady_clock::now();
  auto completed = OrdinaryPolicy(policy) ? OrdinarySchedule(*life) : ScoreQuality(*life);
  const auto elapsed = std::chrono::steady_clock::now() - start;
  ASSERT_TRUE(completed) << completed.error();
  if (OrdinaryPolicy(policy)) {
    const auto& state = *life->ordinary;
    ASSERT_EQ(state.prompt_waves, 32U);
    ASSERT_EQ(state.prompt_heads, 80U);
    ASSERT_EQ(state.decode_waves, 16U);
    ASSERT_EQ(state.reference_rows, 20U);
    ASSERT_EQ(state.compared_rows, 20U);
    ASSERT_EQ(state.checkpoint_restores, 8U);
    ASSERT_EQ(state.spill_restores, 4U);
    ASSERT_EQ(state.final_state_checks, 8U);
    std::ostringstream states, heads;
    for (std::uint32_t owner = 0; owner < kOwners; ++owner)
      states << owner << ' ' << state.saved[owner].position << ' '
             << life->runner->slots().positions(owner) << ' ' << state.saved[owner].bytes << ' '
             << state.final_state[owner] << '\n';
    for (std::uint32_t step = 0; step < kDecodeWaves; ++step)
      for (std::uint32_t owner = 0; owner < (step % 2 == 0 ? 1U : kOwners); ++owner) {
        const auto index = step * kOwners + owner;
        heads << step << ' ' << owner << ' ' << state.tokens[index] << ' ' << state.heads[index]
              << '\n';
      }
    ASSERT_LE(states.tellp(), 4096);
    ASSERT_LE(heads.tellp(), 4096);
    testing::Test::RecordProperty("ordinary_state", states.str());
    testing::Test::RecordProperty("ordinary_heads", heads.str());
    testing::Test::RecordProperty("ordinary_counts", "32 80 16 20 20 8 4 8");
    testing::Test::RecordProperty("state_buffer_bytes", std::to_string(state.capacity));
    testing::Test::RecordProperty("state_layout_bytes",
                                  std::to_string(life->runner->slots().layout(0).bytes));
  }
  testing::Test::RecordProperty("measured_shape_classes", std::to_string(life->shape_count));
  testing::Test::RecordProperty(
      "retained_program_bound",
      std::to_string((life->shape_count + 1) * (kProgramBytes + 4096 * en::kGraphNodeHostBytes)));
  if (!OrdinaryPolicy(policy)) {
    std::ostringstream nll;
    nll << std::setprecision(17);
    const auto losses = std::span(life->quality->scores).first(life->quality->score_count);
    for (const auto& row : losses) {
      nll << row.document << ',' << row.slot << ',' << row.position << ',' << row.target << ','
          << row.nll << '\n';
    }
    ASSERT_LE(nll.tellp(), 512 * 1024);
    testing::Test::RecordProperty("corpus_nll_rows", nll.str());
    testing::Test::RecordProperty("document_rows", std::to_string(document_rows));
    testing::Test::RecordProperty(
        "teacher_elapsed_ns",
        std::to_string(std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count()));
    testing::Test::RecordProperty("corpus_ids_sha256", id_sha);
    testing::Test::RecordProperty("inputs", std::to_string(kOwners * document_rows));
    testing::Test::RecordProperty("targets", std::to_string(kOwners * (document_rows - 1)));
    testing::Test::RecordProperty("score_waves", std::to_string(life->quality->waves));
    testing::Test::RecordProperty("score_chunk_rows", std::to_string(ScoreChunk(policy)));
    testing::Test::RecordProperty("quality_scope",
                                  "same-checkpoint reference pairing; no original-model claim");
  }
  testing::Test::RecordProperty("candidate", std::string(PolicyName(policy)));
  testing::Test::RecordProperty("corpus_ids_sha256", id_sha);
  testing::Test::RecordProperty(
      "fixture_schedule_elapsed_ns",
      std::to_string(std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count()));
  std::ostringstream programs, policies;
  std::size_t executed = 0, vector8 = 0;
  for (const auto& record : std::span(life->records).first(life->shape_count)) {
    policies << "class " << static_cast<unsigned>(record.shape.phase) << ' ' << record.shape.stage
             << ' ' << record.shape.read_align << ' ' << record.shape.row_invariant << ' '
             << record.measured << ' ' << record.lowered << ' ' << record.owner_policy << ' '
             << record.owner_invariant << ' ' << record.selected.runs << '\n';
    for (const auto& segment : record.shape.segments)
      policies << "owner " << static_cast<unsigned>(segment.slot) << ' ' << segment.rows << ' '
               << segment.past << ' ' << segment.outputs << ' ' << segment.read_cells << ' '
               << static_cast<unsigned>(segment.group) << ' ' << segment.want_head << ' '
               << segment.greedy << '\n';
    if (!record.selected.runs) continue;
    ++executed;
    if (OrdinaryPolicy(policy)) {
      ASSERT_TRUE(record.shape.phase == pl::Phase::kPrefill ||
                  record.shape.phase == pl::Phase::kDecode);
      ASSERT_EQ(record.shape.segments.front().read_cells, 256U);
    } else {
      ASSERT_EQ(record.shape.phase, pl::Phase::kScore);
      ASSERT_EQ(record.shape.segments.size(), kOwners);
    }
    if (ScoreChunk(policy) == 1) {
      ASSERT_EQ(record.shape.rows(), kOwners);
      ASSERT_EQ(record.owner_policy, DefaultPolicy(policy));
      ASSERT_EQ(record.owner_invariant, DefaultPolicy(policy));
    }
    ASSERT_TRUE(record.lowered);
    ASSERT_FALSE(record.selected.implementations.empty());
    programs << "shape " << static_cast<unsigned>(record.shape.phase) << ' ' << record.shape.stage
             << ' ' << record.shape.read_align << ' ' << record.shape.row_invariant << '\n';
    for (const auto& segment : record.shape.segments)
      programs << "owner " << static_cast<unsigned>(segment.slot) << ' ' << segment.rows << ' '
               << segment.outputs << ' ' << segment.read_cells << ' '
               << static_cast<unsigned>(segment.group) << ' ' << segment.want_head << ' '
               << segment.greedy << '\n';
    programs << "parameters";
    for (const auto parameter : record.parameters) programs << ' ' << parameter;
    programs << "\nbound " << llmp::base::ToHex(record.selected.identity) << '\n';
    programs << "dense_descriptors\n" << record.dense;
    programs << "implementations\n";
    for (const auto& name : record.selected.implementations) {
      if (record.shape.rows() == 8 && name == "llmp.mul_mat.mmvf_rows") ++vector8;
      programs << name << '\n';
    }
  }
  ASSERT_EQ(executed, OrdinaryPolicy(policy) ? 4U : (ScoreChunk(policy) == 1 ? 1U : 2U));
  ASSERT_LE(policies.tellp(), 64 * 1024);
  testing::Test::RecordProperty("measured_policy_classes", policies.str());
  ASSERT_LE(programs.tellp(), 8 * kMiB);
  testing::Test::RecordProperty("selected_programs", programs.str());
  if (TunedPolicy(policy)) ASSERT_GT(vector8, 0U);
  if (!life->Retire()) {
    std::ignore = life.release();
    FAIL() << "complete scorer owner retained after unknown retirement";
  }
}

TEST(Qwen38JoinedScoreGpu, OrdinaryBroaderNaturalTeacher) { Screen(TeacherPolicy::kJoined); }
TEST(Qwen38JoinedScoreGpu, TunedBroaderNaturalTeacher) { Screen(TeacherPolicy::kTunedJoined); }
TEST(Qwen38JoinedScoreGpu, DefaultSingleRowNaturalTeacher) {
  Screen(TeacherPolicy::kDefaultSingleRow);
}
TEST(Qwen38JoinedScoreGpu, JoinedSingleRowNaturalTeacher) {
  Screen(TeacherPolicy::kJoinedSingleRow);
}

TEST(Qwen38JoinedScoreGpu, DefaultOrdinaryPhaseState) { Screen(TeacherPolicy::kDefaultOrdinary); }
TEST(Qwen38JoinedScoreGpu, JoinedOrdinaryPhaseState) { Screen(TeacherPolicy::kJoinedOrdinary); }

}  // namespace
