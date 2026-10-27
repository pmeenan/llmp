// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "fp16_runner.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <expected>
#include <format>
#include <fstream>
#include <limits>
#include <map>
#include <optional>
#include <span>
#include <utility>

#include "base/bytes.h"
#include "base/check.h"
#include "base/sha256.h"
#include "compiler_inventory.h"
#include "components/step.h"
#include "engine/component_source.h"
#include "engine/model_runner.h"
#include "engine/paged_weights.h"
#include "kernels/compose/kernel_set.h"
#include "kernels/ggml/cublas.h"
#include "kernels/ggml/step_program.h"
#include "planner/pattern.h"
#include "platform/host_probe.h"
#include "providers/device_probe.h"
#include "spec/mapping.h"
#include "spec/mappings/dense.h"
#include "spec/stored.h"

namespace llmp::benchmarks {
namespace {
namespace en = engine;
namespace kg = kernels::ggml;
namespace pl = planner;
using Status = test_support::Status;
constexpr std::uint64_t kMiB = 1ULL << 20U;
// Reuse the shared fixture's bounded artifact/spec owners and seven-graph
// compiler envelope. This adapter measures only five classes, without native
// oracle storage. Charges precede construction, and compiler storage is released
// only after synchronous setup/bind or a completed step.
constexpr std::uint64_t kMetadata = 1024 * kMiB;
// Same reviewed seven-graph inventory as the Qwen2 fixture, including current
// Applied headers, the independent two-round proof payload, and physical tables.
constexpr std::uint64_t kReviewedCompiler =
    2526477744ULL + test_support::compiler_inventory::kLegacyCompilerGrowth;
// Nominal native category already covered by the original compiler envelope.
// Its source is the shared fixture's checked 65,536-descriptor inventory, plus
// exact funder/payment controls. This is an accounting carve, NOT a native
// admission ceiling: actual demand uses the whole remaining paid host envelope.
constexpr std::uint64_t kNativeCarve =
    test_support::compiler_inventory::NativeBytes(65536, ir::Graph::kMaxTensors) +
    sizeof(kg::ConstructionPayment) + sizeof(kg::ConstructionFunder) + 4 * 64;
constexpr std::uint64_t kCompiler = std::max(3072 * kMiB, kReviewedCompiler + kNativeCarve);
constexpr std::uint64_t kIrCompiler = kCompiler - kNativeCarve;
static_assert(kIrCompiler >= kReviewedCompiler);
constexpr std::uint32_t kVocabulary = 151936;
constexpr std::uint64_t kControlEnvelope = 6144 * kMiB;
constexpr std::uint64_t kHeldoutEnvelope = 12288 * kMiB;
constexpr std::uint64_t kControlResults = 1024 * kMiB;
constexpr std::uint64_t kHeldoutResults = 6144 * kMiB;
constexpr std::uint64_t kProgram = 64 * kMiB;
constexpr std::uint64_t kLaunches = 4096;
constexpr std::uint64_t kRuntime = 5 * (kProgram + kLaunches * en::kGraphNodeHostBytes);
static_assert(kMetadata + kCompiler + kHeldoutResults + kRuntime +
                  3 * std::uint64_t{512} * kVocabulary * sizeof(float) <=
              kHeldoutEnvelope);

std::unexpected<std::string> Error(std::string text) { return std::unexpected(std::move(text)); }

// One real enclosing catalog payment, retained by backend construction and
// program payments. PagedNode outlives this adapter; the runner exposes no
// programs outside State. BuildAll is sequential in one host callback, and the
// adapter's synchronous Setup/Bind/Evaluate/Release do not overlap: no worker
// races these counters. Unknown retirement deliberately leaks the entire State.
struct HostAllowance {
  en::PagedNode& node;
  catalog::ExtentId extent;
  std::uint64_t limit = 0, used = 0;
  bool Charge(std::uint64_t bytes) {
    if (used > limit || bytes > limit - used) return false;
    used += bytes;
    return true;
  }
  void Release(std::uint64_t bytes) {
    base::Check(bytes <= used, "FP16 host payment refunds only held bytes");
    used -= bytes;
  }
};
// Shared owner/control and both retained callback capture allocations. The
// backend separately prepays its ConstructionFunder/control before moving them.
constexpr std::uint64_t kHostControl =
    sizeof(HostAllowance) + 2 * 64 + 2 * (sizeof(std::shared_ptr<HostAllowance>) + 64);
constexpr std::uint64_t kInitialHostUsed = kMetadata + kRuntime + kHostControl;
static_assert(kInitialHostUsed + kCompiler + kHeldoutResults +
                  3 * std::uint64_t{512} * kVocabulary * sizeof(float) <=
              kHeldoutEnvelope);

class TemporaryCharge {
 public:
  TemporaryCharge(std::shared_ptr<HostAllowance> owner, std::uint64_t bytes)
      : owner_(std::move(owner)), bytes_(bytes), held_(owner_ && owner_->Charge(bytes)) {}
  TemporaryCharge(const TemporaryCharge&) = delete;
  TemporaryCharge& operator=(const TemporaryCharge&) = delete;
  ~TemporaryCharge() {
    if (held_) owner_->Release(bytes_);
  }
  bool held() const { return held_; }

 private:
  std::shared_ptr<HostAllowance> owner_;
  std::uint64_t bytes_;
  bool held_ = false;
};

// The cells a step reads: its used cells padded to 256, at least 256 and
// at most the cache's (llama.cpp's n_kv for a single-sequence cache).
std::uint32_t PaddedKv(std::uint32_t used_cells, std::uint32_t cells) {
  constexpr std::uint32_t kPad = 256;
  const std::uint64_t padded = (std::uint64_t{used_cells} + kPad - 1) / kPad * kPad;
  return static_cast<std::uint32_t>(
      std::min<std::uint64_t>(cells, std::max<std::uint64_t>(kPad, padded)));
}

pl::StepShape Shape(std::uint32_t past, std::uint32_t rows, std::uint32_t cells) {
  return {.phase = rows == 1 ? pl::Phase::kDecode : pl::Phase::kPrefill,
          .segments = {{.slot = 0,
                        .rows = rows,
                        .past = past,
                        .read_cells = PaddedKv(past + rows, cells),
                        .outputs = rows}},
          .read_align = 256};
}
}  // namespace

struct Fp16Runner::State {
  State(test_support::PagedNode& n, const Fp16Options& o, int owner_id, std::uint32_t stream_id,
        bool record)
      : node(n), options(o), owner(owner_id), stream(stream_id), recording(record) {}
  en::PagedNode& node;
  const Fp16Options& options;
  int owner;
  std::uint32_t stream;
  bool recording;
  bool entered = false;
  bool bound = false;
  bool released = false;
  std::shared_ptr<HostAllowance> host;
  std::uint64_t host_limit = 0;
  std::uint64_t result_limit = 0;
  std::uint64_t result_charge = 0;
  std::vector<pl::StepShape> keys;
  Trajectory trajectory;
  std::vector<spec::SourceValue> values;
  std::optional<spec::Source> source;
  std::optional<spec::ModelSpec> specification;
  std::optional<model::StateLayoutDerived> layout;
  std::map<ir::WeightRef, std::uint32_t> resources;
  std::unique_ptr<kg::GgmlBackend> backend;
  std::unique_ptr<kg::CublasHandle> cublas;
  kg::SelectedWorkspace selected_state{};
  std::unique_ptr<en::ModelRunner> runner;
  catalog::Closure nothing;
  std::string artifact_id;
  std::chrono::steady_clock::time_point first_done;
};

Fp16Runner::Fp16Runner(test_support::PagedNode& node, const Fp16Options& options, int owner,
                       std::uint32_t stream, test_support::Recording* recording,
                       std::string& /*record*/)
    : state_(std::make_unique<State>(node, options, owner, stream, recording != nullptr)) {}

Fp16Runner::~Fp16Runner() {
  if (state_->entered && !state_->released) {
    // Active benchmark callers abort without stack teardown when node retirement
    // is uncertain. Preserve borrowers if another caller violates that protocol.
    (void)state_.release();
    return;
  }
  // State owns retained IR, descriptors, programs and callback borrowers even
  // after a failed Setup. Destroy all of them (and State itself) before the
  // final shared host owner refunds its catalog extent. External result buffers
  // already precede this adapter in the harness's destruction order.
  auto host = std::move(state_->host);
  state_.reset();
}

Status Fp16Runner::Setup() {
  auto& s = *state_;
  const auto& o = s.options;
  if (s.entered || s.released || s.host != nullptr)
    return Error("FP16 benchmark setup is not repeatable");
  if (s.recording || o.restores != 0 || o.relocate || o.load_only != 0 || o.weights_host ||
      o.premapped || o.partial || !o.spill.empty() || o.shared_embeddings)
    return Error("historical BP-P modes require their retired legacy protocol");
  // Harness nodes do not charge their counted host heap by default. Pay the
  // complete bounded adapter envelope explicitly without changing node policy;
  // active callers already include pinned occupancy in their Start budgets.
  const bool heldout = o.trajectory == "heldout";
  s.host_limit = heldout ? kHeldoutEnvelope : kControlEnvelope;
  s.result_limit = heldout ? kHeldoutResults : kControlResults;
  if (o.host_allowance < s.host_limit) return Error("FP16 host allowance is insufficient");
  const auto available = platform::AvailableMemoryBytes();
  const auto occupied = s.node.catalog().OccupancyOf(s.node.domain()).Total().value();
  constexpr std::uint64_t margin = 6144 * kMiB;
  if (!available || *available < s.host_limit + margin ||
      occupied > *available - s.host_limit - margin)
    return Error("FP16 setup lacks physical headroom for its paid envelope");
  // Reserve the control object and its shared allocator metadata before new.
  if (kInitialHostUsed > s.host_limit) return Error("FP16 host controls exceed admission");
  auto paid = s.node.catalog().AddExtent({.domain = s.node.domain(),
                                          .memory_class = catalog::MemoryClass::kRuntime,
                                          .recovery = catalog::Recovery::kPinned,
                                          .size = base::Bytes(s.host_limit),
                                          .content = {}},
                                         true);
  if (!paid) return Error("FP16 host envelope cannot be cataloged");
  s.host = std::shared_ptr<HostAllowance>(
      new HostAllowance{s.node, *paid, s.host_limit, kInitialHostUsed}, [](HostAllowance* owner) {
        auto* node = &owner->node;
        const auto extent = owner->extent;
        delete owner;
        (void)node->catalog().ReleasePinned(extent);
        (void)node->catalog().RemoveExtent(extent);
      });
  TemporaryCharge compiler(s.host, kIrCompiler);
  if (!compiler.held()) return Error("FP16 construction exceeds host admission");
  auto trajectory = LoadTrajectory(o.trajectory, o.tokens);
  if (!trajectory) return Error(trajectory.error());
  s.trajectory = std::move(*trajectory);
  if (s.trajectory.cells > 1024 || s.trajectory.tokens.size() > 577 ||
      s.trajectory.chunks.empty() || s.trajectory.chunks.size() > 64 ||
      std::ranges::any_of(s.trajectory.tokens,
                          [](auto id) { return id < 0 || id >= static_cast<int>(kVocabulary); }))
    return Error("FP16 trajectory exceeds the fixed fixture domain");
  auto weights = std::make_unique<en::PagedWeights>();
  if (auto opened = weights->Open(o.artifact); !opened) return opened;
  const auto& artifact = weights->artifact();
  // The v1 import's stored spec and tensors (D-118), planned under its
  // mapping's policy.
  const auto& mapping = spec::mappings::Qwen2Mapping();
  auto source = spec::SourceOf(artifact);
  if (!source) return Error(source.error());
  if (source->tensors.size() > 512 || std::ranges::any_of(source->tensors, [](const auto& tensor) {
        return tensor.name.size() > 128;
      }))
    return Error("FP16 prepared tensor metadata exceeds its fixed bound");
  s.source.emplace(std::move(*source));
  auto specification = spec::StoredSpec(artifact);
  if (!specification) return Error(specification.error());
  s.specification.emplace(std::move(*specification));
  if (s.specification->stages.size() != 1 || s.specification->stages.front().layers.size() != 24)
    return Error("FP16 benchmark requires the approved single-stage fixture");
  const auto max_rows = *std::ranges::max_element(s.trajectory.chunks);
  auto layout = components::StageLayout(*s.specification, 0, s.trajectory.cells, max_rows);
  if (!layout) return Error(layout.error());
  s.layout.emplace(std::move(*layout));
  std::vector<pl::StepShape> shapes;
  std::uint32_t past = 0;
  for (const auto rows : s.trajectory.chunks) {
    if (rows == 0 || rows > 512 || past > s.trajectory.tokens.size() ||
        rows > s.trajectory.tokens.size() - past)
      return Error("FP16 chunk does not fit its authenticated trajectory");
    shapes.push_back(Shape(past, rows, s.trajectory.cells));
    past += rows;
  }
  if (past != s.trajectory.tokens.size()) return Error("FP16 chunks omit trajectory rows");
  auto step = components::BuildStep(*s.specification, 0, shapes.front(),
                                    {.tensors = s.source->tensors, .state = &*s.layout});
  if (!step) return Error(step.error());
  if (step->weights.size() > 512 || step->graph.nodes().size() > 4096)
    return Error("FP16 primitive step exceeds its fixed inventory");
  for (const auto& weight : step->weights) {
    if (weight.parts.size() != 1) return Error("FP16 benchmark requires one part per weight");
    const auto found = artifact.FindResource(weight.parts.front().first);
    if (!found) return Error("FP16 step refers to an absent prepared resource");
    s.resources.emplace(weight.ref, *found);
  }
  static_cast<void>(llmp::kernels::RegisterBuildKernels());
  auto backend = kg::GgmlBackend::Create(0, s.node.execution(), s.node.stream(s.stream), nullptr);
  if (!backend) return Error(backend.error());
  s.backend = std::move(*backend);
  if (auto funded = s.backend->FundConstruction(
          [host = s.host](std::uint64_t bytes) { return host->Charge(bytes); },
          [host = s.host](std::uint64_t bytes) { host->Release(bytes); });
      !funded)
    return Error(funded.error());
  std::optional<pl::Policy> policy;
  if (!o.fusion) {
    policy.emplace();
    for (const auto& pattern : pl::Patterns()) policy->disabled.emplace_back(pattern.name);
    pl::Canonicalize(*policy);
  }
  auto checked_backend = s.backend->Backend();
  const auto lower = checked_backend.lower;
  checked_backend.lower = [&s, lower](const kernels::LowerInput& input)
      -> std::expected<std::unique_ptr<kernels::StepProgram>, std::string> {
    if (std::ranges::find(s.keys, input.token->shape) == s.keys.end())
      return Error("FP16 program is outside its measured plan classes");
    auto program = lower(input);
    if (!program) return program;
    if ((*program)->host_bytes() > kProgram || (*program)->launches() > kLaunches)
      return Error("FP16 program exceeds its paid retained-plan bound");
    return program;
  };
  auto component = en::ComponentStepSource::Create(
      {.spec = &*s.specification,
       .stage = 0,
       .bindings = {.tensors = s.source->tensors, .state = &*s.layout},
       .bound_digest = s.specification->Digest(),
       .mapping = &mapping,
       .values = s.values,
       .slots = 1,
       .policy = std::move(policy),
       .backend = std::move(checked_backend)});
  if (!component) return Error(component.error());
  // Measure one representative of each actual plan class. Runtime positions
  // remain data; refuse a position-dependent class instead of under-measuring.
  auto& keys = s.keys;
  std::vector<std::vector<std::uint64_t>> parameters;
  const auto paid_keys = s.trajectory.name == "heldout" ? 5U : 2U;
  for (const auto& shape : shapes) {
    const auto found = std::ranges::find(keys, shape);
    auto current = (*component)->ShapeParameters(shape);
    if (found == keys.end()) {
      if (keys.size() == paid_keys) return Error("FP16 shape exceeds its paid plan classes");
      keys.push_back(shape);
      parameters.push_back(std::move(current));
    } else if (parameters[static_cast<std::size_t>(found - keys.begin())] != current) {
      return Error("FP16 position changes its measured plan class");
    }
  }
  if (keys.size() != paid_keys) return Error("FP16 trajectory has an unexpected plan inventory");
  s.artifact_id = artifact.id();
  en::PagedWeights* paged = weights.get();
  en::RunnerConfig config;
  config.slots = {.model = std::format("benchmark-fp16-{}", s.owner),
                  .slots = 1,
                  .stages = {{.name = "target", .layout = *s.layout}},
                  .spec = s.specification->Digest(),
                  .read_align = 256,
                  .spill_place = {},
                  .spill_directory = o.out / "shared-state",
                  .invalidate = {},
                  .drop_plans = {}};
  std::vector<en::GroupPlace> places(artifact.groups().size(), en::GroupPlace::kDevice);
  config.weights.push_back(
      {.weights = std::move(weights), .places = std::move(places), .slabs = {}});
  config.setup_shapes = keys;
  const auto facts = providers::QueryDeviceFacts(0);
  if (!facts) return Error("FP16 backend device facts are unavailable");
  const auto cublas_bytes =
      kg::CublasHandle::UpstreamWorkspace(static_cast<int>(facts->architecture)).value();
  config.kernel_memory.push_back({.name = "FP16 benchmark cuBLAS", .bytes = cublas_bytes});
  const auto selected_bytes = s.backend->persistent_bytes();
  if (selected_bytes)
    config.kernel_memory.push_back({.name = "FP16 selected kernel state", .bytes = selected_bytes});
  config.prepare = [&s, cublas_bytes,
                    selected_bytes](std::span<const std::uint64_t> bases) -> Status {
    if (bases.size() != 1U + static_cast<std::size_t>(selected_bytes != 0))
      return Error("FP16 kernel workspaces are not mapped");
    s.selected_state = {selected_bytes ? bases[1] : 0, selected_bytes};
    auto handle =
        kg::CublasHandle::Create(0, s.node.execution(), s.node.stream(s.stream),
                                 {.base = bases.front(), .size = base::Bytes(cublas_bytes)});
    if (!handle) return Error(handle.error().detail);
    s.cublas = std::move(*handle);
    return s.backend->LendCublas(s.cublas.get());
  };
  config.retire = [&s] {
    s.backend->LaunchReleased();
    s.selected_state = {};
    s.cublas.reset();
  };
  config.launch =
      [&s](std::uint64_t) -> std::expected<std::unique_ptr<kernels::StepLaunch>, std::string> {
    return s.backend->Launch({.base = s.node.pool().base, .size = base::Bytes(s.node.pool().bytes)},
                             s.selected_state);
  };
  config.weight_address = [&s, paged](const ir::WeightRef& weight, format::PartRole) {
    const auto found = s.resources.find(weight);
    return found == s.resources.end() ? 0 : paged->resource_address(found->second);
  };
  auto runner =
      en::ModelRunner::Create(s.node, std::move(config), std::move(*component), s.owner, s.stream);
  if (!runner) return Error(runner.error());
  s.runner = std::move(*runner);
  s.entered = true;
  return s.runner->Setup();
}

Status Fp16Runner::Register() {
  if (!state_->runner || state_->released) return Error("FP16 benchmark is not set up");
  return state_->runner->Register();
}

Status Fp16Runner::Bind() {
  auto& s = *state_;
  if (!s.runner || s.bound || s.released) return Error("FP16 benchmark bind is not available");
  TemporaryCharge compiler(s.host, kIrCompiler);
  if (!compiler.held()) return Error("FP16 bind exceeds host admission");
  if (auto bound = s.runner->Bind(); !bound) return bound;
  s.bound = true;
  return {};
}

Status Fp16Runner::Evaluate(int evaluation, std::vector<float>& result) {
  auto& s = *state_;
  if (!s.bound || s.released || evaluation < 1) return Error("FP16 benchmark is not ready");
  const auto count = s.trajectory.tokens.size() * std::uint64_t{kVocabulary};
  const auto bytes = count * sizeof(float);
  if (result.capacity() > count || bytes > s.result_limit / 2 ||
      s.result_charge > s.result_limit - 2 * bytes || !s.host->Charge(2 * bytes))
    return Error("FP16 externally retained output exceeds its admitted domain");
  // Callers retain complete evaluations. Their bytes and allocation growth stay
  // paid until adapter destruction; failed attempts consume the same allowance.
  s.result_charge += 2 * bytes;
  const auto max_rows = *std::ranges::max_element(s.trajectory.chunks);
  const auto step_outputs = 3 * std::uint64_t{max_rows} * kVocabulary * sizeof(float);
  TemporaryCharge compiler(s.host, kIrCompiler + step_outputs);
  if (!compiler.held()) return Error("FP16 step exceeds host admission");
  result.resize(static_cast<std::size_t>(count));
  auto& slots = s.runner->slots();
  return s.node.WithRequest(s.stream, slots.execution(), "shared FP16 benchmark", [&]() -> Status {
    if (auto selected = slots.SelectSlots(std::vector<std::uint32_t>{0}); !selected)
      return selected;
    if (auto cleared = slots.Clear(0); !cleared) return cleared;
    std::uint32_t past = 0;
    for (const auto rows : s.trajectory.chunks) {
      const auto tokens = std::span(s.trajectory.tokens).subspan(past, rows);
      const std::array work{en::SegmentWork{.tokens = tokens}};
      auto completed = s.runner->RunStep(
          {.shape = Shape(past, rows, s.trajectory.cells), .work = work, .ahead = {}});
      if (!completed) return Error(completed.error().what);
      if (completed->outputs.size() != 1 ||
          completed->outputs.front().size() != std::uint64_t{rows} * kVocabulary * sizeof(float))
        return Error("FP16 benchmark returned incomplete full heads");
      std::memcpy(result.data() + std::uint64_t{past} * kVocabulary,
                  completed->outputs.front().data(), completed->outputs.front().size());
      if (past == 0) s.first_done = std::chrono::steady_clock::now();
      past += rows;
    }
    if (slots.positions(0) != s.trajectory.tokens.size() ||
        std::ranges::any_of(result, [](float value) { return !std::isfinite(value); }))
      return Error("FP16 benchmark did not complete its finite full trajectory");
    return {};
  });
}

Status Fp16Runner::RunAlone() {
  std::vector<test_support::LoadStats> log;
  if (auto loaded = state_->node.Load(weights(), "shared FP16 weights", log); !loaded)
    return loaded;
  std::vector<std::vector<float>> results(2);
  for (int i = 0; i < 2; ++i)
    if (auto evaluated = Evaluate(i + 1, results[static_cast<std::size_t>(i)]); !evaluated)
      return evaluated;
  return Write(results);
}

Status Fp16Runner::Write(const std::vector<std::vector<float>>& results) {
  if (results.empty()) return Error("FP16 benchmark has no completed results");
  const auto expected = state_->trajectory.tokens.size() * std::uint64_t{kVocabulary};
  for (const auto& result : results) {
    if (result.size() != expected) return Error("FP16 comparison has incomplete results");
    for (std::size_t i = 0; i < result.size(); ++i)
      if (std::bit_cast<std::uint32_t>(result[i]) !=
          std::bit_cast<std::uint32_t>(results.front()[i]))
        return Error("same shared FP16 implementation changed across evaluations");
  }
  base::Sha256 hash;
  hash.Update(std::as_bytes(std::span(results.front())));
  std::error_code error;
  std::filesystem::create_directories(state_->options.out, error);
  if (error) return Error("FP16 result directory is unavailable");
  std::ofstream raw(state_->options.out / "logits.f32le", std::ios::binary);
  raw.write(reinterpret_cast<const char*>(results.front().data()),
            static_cast<std::streamsize>(results.front().size() * sizeof(float)));
  if (!raw) return Error("FP16 results could not be written");
  std::ofstream summary(state_->options.out / "summary.json");
  summary << std::format(
      "{{\"execution\":\"shared-components\",\"trajectory\":\"{}\","
      "\"artifact\":\"{}\",\"tokens\":{},\"evaluations\":{},"
      "\"logits_sha256\":\"{}\",\"legacy_protocol\":false}}\n",
      state_->trajectory.name, state_->artifact_id, state_->trajectory.tokens.size(),
      results.size(), base::ToHex(hash.Finish()));
  return summary ? Status{} : Error("FP16 summary could not be written");
}

std::uint64_t Fp16Runner::activations_needed() const {
  return state_->runner ? state_->runner->activations_needed() : 0;
}
std::uint64_t Fp16Runner::pool_needed() const {
  return state_->runner ? state_->runner->pool_needed() : 0;
}
std::chrono::steady_clock::time_point Fp16Runner::first_chunk_done() const {
  return state_->first_done;
}
const catalog::Closure& Fp16Runner::everything() const {
  return state_->runner ? state_->runner->slots().execution() : state_->nothing;
}
std::vector<catalog::ExtentId> Fp16Runner::weights() const {
  return state_->runner ? state_->runner->weights() : std::vector<catalog::ExtentId>{};
}
std::uint64_t Fp16Runner::weight_read_bytes() const {
  return state_->runner ? state_->runner->weight_read_bytes() : 0;
}
const Trajectory& Fp16Runner::trajectory() const { return state_->trajectory; }
std::uint32_t Fp16Runner::stream() const { return state_->stream; }
const catalog::Closure& Fp16Runner::fence_closure() const {
  return state_->runner ? state_->runner->fence_closure() : state_->nothing;
}
std::vector<catalog::ExtentId> Fp16Runner::managed_extents() const {
  return state_->runner ? state_->runner->managed_extents() : std::vector<catalog::ExtentId>{};
}
Status Fp16Runner::Release() {
  auto& s = *state_;
  if (s.released) return {};
  if (s.runner) {
    if (auto released = s.runner->Release(); !released) return released;
  }
  s.backend.reset();
  s.cublas.reset();
  s.released = true;
  return {};
}

}  // namespace llmp::benchmarks
