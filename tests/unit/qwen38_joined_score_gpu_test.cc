// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Bounded target-only shared scoring of the fixed authenticated Qwen52 corpus.
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
#include "corpus_score.h"
#include "engine/component_source.h"
#include "engine/model_runner.h"
#include "engine/paged_weights.h"
#include "kernels/ggml/graph_plan.h"
#include "model/qwen38.h"
#include "planner/plan.h"
#include "platform/host_probe.h"
#include "providers/device_runtime.h"
#include "runtime/component_build.h"
#include "runtime/memory_guard.h"
#include "spec/mappings/qwen38.h"
#include "tokenizer_fixtures.h"

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
constexpr std::uint64_t kHostBytes = std::uint64_t{8} << 30U;
constexpr std::uint64_t kProgramBytes = 64 * kMiB;
constexpr std::size_t kFactoryShapes = 16, kShapes = kFactoryShapes;
// Current-source construction bound, independent of completed program limits.
// funding.md derives these categories from Builder's enforced N/T limits,
// the two rewrite rounds, serial fallback passes and Lower's descriptor count.
// Metadata names are checked before source construction below. No per-class
// multiplier applies to transient graphs: Setup builds one class at a time.
constexpr std::uint64_t kNodes = ir::Graph::kMaxNodes;
constexpr std::uint64_t kTensors = ir::Graph::kMaxTensors;
constexpr std::uint64_t kGraphConstruction = 4 * 3 * (kNodes * 480 + kTensors * 192);
constexpr std::uint64_t kIrWorksets = 4096 * (kNodes + kTensors);
// Two disjoint rewrite rounds, four outputs; top-ten ExpertReduce adds23.
constexpr std::uint64_t kNativeDescriptors = kTensors + 16 + 8 * kNodes + 23 * kNodes;
// Arena512, native planning/index/copy work1024, traversal<4 pointers each.
constexpr std::uint64_t kNativeConstruction = kNativeDescriptors * (512 + 1024 + 32);
constexpr std::uint64_t kCompilerEnvelope =
    kGraphConstruction + kIrWorksets + kNativeConstruction + 64 * kMiB;
constexpr std::uint64_t kRetainedPrograms =
    (kShapes + 1) * (kProgramBytes + 4096 * en::kGraphNodeHostBytes);  // Retain paid17 ceiling.
constexpr std::uint64_t kMetadataEnvelope = 1024 * kMiB;
constexpr std::uint64_t kObservationEnvelope = 128 * kMiB;
static_assert(kCompilerEnvelope + kRetainedPrograms + kMetadataEnvelope + kObservationEnvelope <=
              kHostBytes);
static_assert(ir::Graph::kMaxNodes == 65536 && ir::Graph::kMaxTensors == 131072);
static_assert(sizeof(ir::Node) <= 160 && sizeof(ir::TensorDesc) <= 160);
static_assert(sizeof(pl::Selected) <= 128 && sizeof(pl::Applied) <= 64);
static_assert(ir::kMaxSources == 10 && ir::kMaxOutputs == 4);
static_assert(sizeof(llmp::kernels::ggml::PlanStep) <= 64);
static_assert(sizeof(std::string) <= 32 && sizeof(en::SlotLifecycle::Range) <= 24);
static_assert(kShapes * (4096 * (32 + 129) + 64 * 1024) + 2 * 8 * kMiB +
                  (3 * 8 + 1) * kVocabulary * sizeof(float) + 4 * 3 * 512 * 24 + 4096 + 64 * 1024 +
                  2 * 512 * 1024 <
              kObservationEnvelope);

constexpr std::string_view kArtifact =
    "c4fb47a911207c11f935f932d05196dc1701aa0d886eac1b5e91934e554b5a93";

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
          segment.read_cells != 512 || (phase == pl::Phase::kPrefill && segment.greedy))
        return false;
    }
    if (i && (segment.rows != shape.segments[0].rows || segment.greedy != shape.segments[0].greedy))
      return false;
  }
  return true;
}

struct ShapeRecord {
  pl::StepShape shape;
  std::vector<std::uint64_t> parameters;
  bool parameterized = false;
  bool measured = false, lowered = false;
  ts::CorpusPlanRecord selected;
  // Planned surviving dense products: node, weight/input IDs, physical K/N/rows,
  // encodings, strides and binding. This is a descriptor receipt, distinct
  // from the selected bound implementation inventory recorded at Bind.
  std::string dense;
};

class RecordedSource final : public en::StepSource {
 public:
  RecordedSource(std::unique_ptr<en::StepSource> inner, std::array<ShapeRecord, kShapes>& records)
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
  std::vector<std::uint64_t> ShapeParameters(const pl::StepShape& shape) const override {
    return inner_->ShapeParameters(shape);
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
  std::array<ShapeRecord, kShapes>& records_;
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

bool retirement_failed = false;

struct Lifetime {
  en::PagedNode node{{.slot_bytes = en::kSlabSlotBytes}};
  std::array<ShapeRecord, kShapes> records{};
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

en::Status ValidateProgram(Lifetime& life, const kn::LowerInput& input,
                           const kn::StepProgram& program) {
  if (!input.token) return std::unexpected("scorer requires actual token metadata");
  if (!AllowedShape(input.token->shape) || input.token->context != 512 ||
      input.token->shape.stage != 0 || input.plan.owner_row_invariant ||
      input.plan.owner_rows_policy_enabled || input.token->shape.row_invariant ||
      !input.plan.owner_products.empty() || input.token->step.weights.size() > 4096 ||
      input.token->step.state.size() > 4096 || input.token->step.inputs.size() > 256 ||
      !input.token->step.saves.empty())
    return std::unexpected("scorer source/state/policy exceeds closed paid domain");
  auto found = std::ranges::find_if(life.records,
                                    [&](const auto& r) { return r.shape == input.token->shape; });
  if (found == life.records.end()) return std::unexpected("lowering escaped sixteen paid classes");
  if (input.token->shape.phase == pl::Phase::kScore) {
    if (auto valid = ts::CheckCorpusHead(input, program, kVocabulary); !valid) return valid;
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

std::expected<rt::ComponentBuildOptions, std::string> Options(Lifetime& life, bool tuned) {
  auto policy = pl::ParsePolicy(llmp::spec::mappings::Qwen38Mapping().policies.front().text);
  if (!policy) return std::unexpected(policy.error());
  if (std::erase(policy->enabled, "owner_row_invariant") != 1)
    return std::unexpected("native mapping owner policy changed");
  policy->disabled.emplace_back("owner_row_invariant");
  pl::Canonicalize(*policy);
  auto domain =
      en::NumericalDomain::Create({{.stage = 0,
                                    .name = "target",
                                    .uses = {en::NumericalUse::kDecode, en::NumericalUse::kScore},
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
      .diagnostic_solo = true,
      .prefill_lookahead = false,
      .prepare_state = false,
      .state_only_prefill = false,
      .execution_domain = std::make_shared<const en::NumericalDomain>(std::move(*domain)),
      .tuning = {},
      .execution_metadata_bytes = 1 * kMiB,
      .validate_program = [&life](const kn::LowerInput& input, const kn::StepProgram& program) {
        return ValidateProgram(life, input, program);
      }};
  if (tuned) {
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

en::Status SetUp(Lifetime& life, bool tuned) {
  auto& node = life.node;
  // All construction-domain prerequisites precede the first graph. The source
  // manifest/index/config hashes below bind this compiled profile's artifact.
  const auto& profile = llmp::model::Qwen38Flash();
  if (profile.layers != 48 || profile.experts_used != 10 || profile.width != 2560 ||
      ggml_tensor_overhead() > 512)
    return std::unexpected("compiler construction domain or descriptor ABI changed");
  node.SetHostFloor(0);
  if (auto open = node.Open(); !open) return open;
  const auto occupied = node.catalog().OccupancyOf(node.domain()).Total().value();
  if (occupied > kDomainBytes - kHostBytes) return std::unexpected("startup payment does not fit");
  auto paid = node.catalog().AddExtent({.domain = node.domain(),
                                        .memory_class = llmp::catalog::MemoryClass::kRuntime,
                                        .recovery = llmp::catalog::Recovery::kPinned,
                                        .size = llmp::base::Bytes(kHostBytes),
                                        .content = {}},
                                       true);
  if (!paid) return std::unexpected("startup catalog payment refused");
  const fs::path artifact = fs::path(ts::ModelsDir()) / "m3-artifacts" / kArtifact;
  for (const auto& file : std::array{
           std::tuple{"manifest.json", std::uint64_t{23037}, kArtifact},
           std::tuple{"index.json", std::uint64_t{8152364},
                      std::string_view{
                          "5b65dcce8169374e638264d7ebdcb5cca517234b3dec26f1272a5f4f9c0e4c89"}},
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
  auto options = Options(life, tuned);
  if (!options) return std::unexpected(options.error());
  rt::ModelSettings settings;
  settings.architecture = "qwen4exp";
  settings.max_slots.value = 4;
  settings.context.value = 512;
  settings.prefill_chunk.value = 2;
  settings.wave_read_align.value = 256;
  settings.speculation.value = false;
  llmp::config::ModelEntry entry;
  entry.name = tuned ? "qwen50-c" : "qwen50-a";
  entry.artifact = std::string(kArtifact);
  const fs::path store = fs::path(ts::ModelsDir()) / "m3-artifacts";
  auto build = rt::ComponentLlmBuild(node, store, entry, 0, std::move(*options));
  auto parts = build(settings);
  if (!parts) return std::unexpected(parts.error());
  if (parts->logit_width != kVocabulary || parts->config.slots.stages.size() != 1 ||
      parts->config.setup_shapes.size() != kFactoryShapes || parts->caps.max_segments < kOwners ||
      !parts->config.slots.stages[0].layout.bytes)
    return std::unexpected("actual target parts exceed the declared scoring domain");
  life.keep = parts->keep;
  parts->config.slots.model = entry.name;
  parts->config.slots.slots = kOwners;
  parts->config.slots.spill_directory = life.scratch / "spill";
  for (std::size_t i = 0; i < kShapes; ++i) {
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
  auto source = std::make_unique<RecordedSource>(std::move(parts->source), life.records);
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
  if (!std::ranges::all_of(life.records, [](const auto& r) { return r.measured; }))
    return std::unexpected("not all sixteen actual setup classes were measured");
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
      {.vocabulary = kVocabulary, .chunk_rows = 2},
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
  observed.waves = (rows + 1) / 2;
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

void Screen(bool tuned) {
  ASSERT_FALSE(retirement_failed) << "prior owner retirement remains unknown";
  const auto available = llmp::platform::AvailableMemoryBytes();
  ASSERT_TRUE(available);
  ASSERT_GE(*available, kDomainBytes + rt::kUncountedMargin);
  Owner life(new Lifetime);
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
  auto setup = SetUp(*life, tuned);
  ASSERT_TRUE(setup) << setup.error();
  // This witness is in the startup observation allowance admitted by SetUp.
  life->quality = std::make_unique<QualityWitness>();
  life->quality->document_rows = document_rows;
  std::memcpy(life->quality->ids.data(), ids->data(), ids->size());
  const auto start = std::chrono::steady_clock::now();
  auto score = ScoreQuality(*life);
  const auto elapsed = std::chrono::steady_clock::now() - start;
  ASSERT_TRUE(score) << score.error();
  testing::Test::RecordProperty("measured_shape_classes", std::to_string(kShapes));
  testing::Test::RecordProperty("retained_program_bound", std::to_string(kRetainedPrograms));
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
  testing::Test::RecordProperty("score_waves", std::to_string((document_rows + 1) / 2));
  testing::Test::RecordProperty("candidate", tuned ? "C-three-physical8-tunes" : "A-no-tunes");
  testing::Test::RecordProperty("quality_scope",
                                "same-checkpoint reference pairing; no original-model claim");
  std::ostringstream programs;
  std::size_t executed = 0, vector8 = 0;
  for (const auto& record : life->records) {
    if (!record.selected.runs) continue;
    ++executed;
    ASSERT_EQ(record.shape.phase, pl::Phase::kScore);
    ASSERT_EQ(record.shape.segments.size(), kOwners);
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
  ASSERT_EQ(executed, 2U);  // Actual full8 and remainder4 score keys.
  ASSERT_LE(programs.tellp(), 8 * kMiB);
  testing::Test::RecordProperty("selected_programs", programs.str());
  if (tuned) ASSERT_GT(vector8, 0U);
  if (!life->Retire()) {
    std::ignore = life.release();
    FAIL() << "complete scorer owner retained after unknown retirement";
  }
}

TEST(Qwen38JoinedScoreGpu, OrdinaryBroaderNaturalTeacher) { Screen(false); }
TEST(Qwen38JoinedScoreGpu, TunedBroaderNaturalTeacher) { Screen(true); }

}  // namespace
