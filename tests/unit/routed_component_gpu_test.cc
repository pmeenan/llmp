// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// One actual shared component step: no family runner or model admission.
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <expected>
#include <filesystem>
#include <map>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "artifact_writer.h"
#include "components/moe.h"
#include "components/step.h"
#include "engine/component_source.h"
#include "engine/execution_contract.h"
#include "engine/model_runner.h"
#include "engine/runner_resources.h"
#include "engine/support.h"
#include "execution/registry.h"
#include "expected_error.h"
#include "kernels/common/routed_program.h"
#include "kernels/decode/selected_grouped.h"
#include "kernels/ggml/step_program.h"
#include "platform/host_probe.h"
#include "providers/device_runtime.h"
#include "spec/stored.h"

namespace llmp::test_support {
class RoutedComponentInspection {
 public:
  static auto Completed(const engine::ModelRunner& runner, const kernels::RoutedProgram& program,
                        std::size_t group) {
    return runner.CompletedRoute(program, group);
  }
};
}  // namespace llmp::test_support

namespace {
namespace en = llmp::engine;
namespace co = llmp::components;
namespace sp = llmp::spec;
namespace md = llmp::model;
namespace ir = llmp::ir;
namespace pl = llmp::planner;
namespace kg = llmp::kernels::ggml;
namespace kn = llmp::kernels;
namespace de = llmp::kernels::decode;
namespace fmt = llmp::format;
namespace ca = llmp::catalog;
namespace ts = llmp::test;
using Role = fmt::PartRole;
using llmp::base::Bytes;
constexpr std::uint32_t kWidth = 32, kHead = 64, kExperts = 4, kUsed = 2, kRows = 3, kContext = 256;
constexpr std::uint64_t kMiB = 1ULL << 20, kDomain = 512 * kMiB, kMetadata = 384 * kMiB;
constexpr std::uint64_t kCompiler = 256 * kMiB, kOwner = 64 * kMiB;
// Fixed two-component graph emitter, not global IR maxima. Its compiler,
// temporary copies and native descriptors are prepaid before construction.
constexpr std::uint64_t kNodes = 2048, kTensors = 8192;
constexpr std::uint64_t kDescriptors = kTensors + 16 + 16 * kNodes;
constexpr std::uint64_t kGraphBytes = 4 * 3 * (kNodes * 480 + kTensors * 192);
constexpr std::uint64_t kWorkBytes = 4096 * (kNodes + kTensors);
constexpr std::uint64_t kNativeBytes = kDescriptors * (512 + 1024 + 32);
static_assert(kGraphBytes + kWorkBytes + kNativeBytes + 64 * kMiB <= kCompiler);
static_assert(sizeof(ir::Node) + 96 <= 480 && sizeof(ir::TensorDesc) <= 192);
constexpr std::uint64_t kProgram = 8 * kMiB, kLaunches = 512;
static_assert(kCompiler + kOwner + 2 * (kProgram + kLaunches * en::kGraphNodeHostBytes) <
              kMetadata);

en::Status Error(std::string s) { return std::unexpected(std::move(s)); }
pl::StepShape Shape(std::uint32_t past = 0) {
  return {.phase = pl::Phase::kScore,
          .segments =
              {{.slot = 0, .rows = kRows, .past = past, .read_cells = kContext, .outputs = kRows}},
          .read_align = 256};
}
fmt::TensorFormat Dense(std::uint64_t a, std::uint64_t b = 0) {
  return {.encoding = fmt::Encoding::kF32, .shape = b ? ir::ShapeOf({a, b}) : ir::ShapeOf({a})};
}
fmt::TensorFormat Expert(bool gate) {
  return {.encoding = fmt::Encoding::kNvfp4,
          .params = {.tensor_scale = true},
          .layout = fmt::Layout::kSplit,
          .shape = ir::ShapeOf({kWidth, gate ? 2 * kWidth : kWidth}),
          .stacked = static_cast<std::uint8_t>(gate ? 2 : 1)};
}
sp::Param P(std::string name, sp::ParamValue value) { return {std::move(name), std::move(value)}; }
sp::ComponentRef C(std::string name, std::vector<sp::Param> params) {
  return {.kind = std::move(name), .version = 1, .params = std::move(params), .weights = {}};
}
sp::ModelSpec Spec(bool ordered = false) {
  auto attention = C(
      "attention.dense",
      {P("placement", std::string("pre")), P("eps", 1e-6), P("heads", std::int64_t{1}),
       P("kv_heads", std::int64_t{1}), P("key_dim", std::int64_t{kHead}),
       P("value_dim", std::int64_t{kHead}), P("scale", 1.0 / std::sqrt(static_cast<double>(kHead))),
       P("window", std::int64_t{0}), P("rotation", std::string("neox")),
       P("rope_dims", std::int64_t{kHead}), P("rope_base", 10000.0), P("rope_scale", 1.0)});
  for (const std::string name : {"norm", "q", "k", "v", "output"})
    attention.weights.push_back({.role = name, .tensor = "attention." + name});
  auto moe =
      C("ffn.moe",
        {P("placement", std::string("none")), P("eps", 1e-6), P("activation", std::string("silu")),
         P("hidden", std::int64_t{kWidth}), P("experts", std::int64_t{kExperts}),
         P("used", std::int64_t{kUsed}), P("expert_hidden", std::int64_t{kWidth}),
         P("router", std::string("softmax_topk")), P("router_input", std::string("stream")),
         P("normalize", true), P("expert_scales", false), P("stacked", true)});
  for (const std::string name : {"gate", "up", "down", "router"})
    moe.weights.push_back({.role = name, .tensor = "moe." + name});
  const std::array roles{Role::kCodes, Role::kBlockScales, Role::kTensorScale};
  const std::array suffix{"codes", "blocks", "scale"};
  for (const bool gate : {true, false})
    for (std::size_t i = 0; i != roles.size(); ++i)
      moe.weights.push_back(
          {.role = gate ? "gate_up_experts" : "down_experts",
           .tensor = std::string(gate ? "experts.gate." : "experts.down.") + suffix[i],
           .part = roles[i],
           .experts = true});
  sp::Stage stage{.name = "target", .context = kContext};
  stage.stream.width = kWidth;
  stage.embedding = C("embedding.token", {P("scale", 1.0)});
  stage.embedding->weights = {{.role = "table", .tensor = "embedding"}};
  auto second = moe;
  if (ordered)
    for (auto& weight : second.weights) weight.tensor = "second." + weight.tensor;
  stage.layers = {{.components = {std::move(attention), std::move(moe)}}};
  if (ordered) stage.layers.push_back({.components = {std::move(second)}});
  stage.heads.output = C("head.linear", {});
  stage.heads.output->weights = {{.role = "weight", .tensor = "head"}};
  sp::ModelSpec spec{.stages = {std::move(stage)}};
  sp::Canonicalize(spec);
  return spec;
}
float Router(std::size_t expert, std::size_t col) {
  return col == expert ? 4.F : col == (expert ^ 1U) ? 2.F : 0.F;
}
std::filesystem::path Artifact(const std::filesystem::path& root, const sp::ModelSpec& spec,
                               bool ordered = false) {
  ts::V1Plan plan;
  plan.expert_count = kExperts;
  plan.spec = sp::ToJson(spec);
  ts::V1Group base{.kind = llmp::artifact::GroupKind::kGlobal};
  std::uint64_t offset = 0;
  std::vector<std::string> names{"embedding",   "head",        "attention.norm",   "attention.q",
                                 "attention.k", "attention.v", "attention.output", "moe.router",
                                 "moe.gate",    "moe.up",      "moe.down"};
  if (ordered)
    for (const auto* name : {"moe.router", "moe.gate", "moe.up", "moe.down"})
      names.push_back(std::string("second.") + name);
  for (const std::string& name : names) {
    const auto format = name == "attention.norm" ? Dense(kWidth)
                        : (name == "moe.router" || name == "second.moe.router")
                            ? Dense(kWidth, kExperts)
                        : name == "attention.output" ? Dense(kHead, kWidth)
                        : name == "attention.q" || name == "attention.k" || name == "attention.v"
                            ? Dense(kWidth, kHead)
                            : Dense(kWidth, kWidth);
    const auto parts = fmt::PartsOf(format);
    if (!parts || parts->size() != 1 || (*parts)[0].role != Role::kData ||
        (*parts)[0].readable == 0 || (*parts)[0].readable > 8192)
      return {};
    base.resources.push_back({.name = name, .format = format, .offset = offset});
    const auto stride = en::support::Round((*parts)[0].readable, 4096);
    if (__builtin_add_overflow(offset, stride, &offset) || offset > (ordered ? 128 : 64) * 1024)
      return {};
  }
  plan.groups.push_back(base);
  plan.groups.push_back({.kind = llmp::artifact::GroupKind::kExpert,
                         .layer = 0,
                         .arrays = {{.name = "experts.gate.codes",
                                     .format = Expert(true),
                                     .part = Role::kCodes,
                                     .offset = 0},
                                    {.name = "experts.gate.blocks",
                                     .format = Expert(true),
                                     .part = Role::kBlockScales,
                                     .offset = 1024},
                                    {.name = "experts.gate.scale",
                                     .format = Expert(true),
                                     .part = Role::kTensorScale,
                                     .offset = 1280},
                                    {.name = "experts.down.codes",
                                     .format = Expert(false),
                                     .part = Role::kCodes,
                                     .offset = 1536},
                                    {.name = "experts.down.blocks",
                                     .format = Expert(false),
                                     .part = Role::kBlockScales,
                                     .offset = 2048},
                                    {.name = "experts.down.scale",
                                     .format = Expert(false),
                                     .part = Role::kTensorScale,
                                     .offset = 2304},
                                    {.name = "unused.padding",
                                     .format = Dense(64),
                                     .offset = en::kPagedExtent - 256}},
                         .new_shard = true});
  if (ordered) {
    auto second = plan.groups.back();
    second.layer = 1;
    for (auto& array : second.arrays) array.name = "second." + array.name;
    plan.groups.push_back(std::move(second));
  }
  auto files = ts::Render(plan);
  if (files.shards.size() != (ordered ? 3U : 2U)) return {};
  std::size_t chunk = 0;
  for (std::size_t shard = 0; shard != files.shards.size(); ++shard) {
    auto& bytes = files.shards[shard];
    std::uint64_t header = 0;
    if (bytes.size() < sizeof(header)) return {};
    std::memcpy(&header, bytes.data(), sizeof(header));
    if (header > bytes.size() - 8) return {};
    const auto begin = header + 8, size = bytes.size() - begin;
    std::fill(bytes.begin() + static_cast<std::ptrdiff_t>(begin), bytes.end(), '\0');
    if (shard == 0) {
      if (size != offset) return {};
      for (const auto& resource : base.resources) {
        const auto at = begin + *resource.offset;
        const auto cols = resource.format.shape.ne[0];
        const auto rows = resource.format.shape.rank == 1 ? 1 : resource.format.shape.ne[1];
        const auto parts = fmt::PartsOf(resource.format);
        if (!parts || parts->size() != 1 || at > bytes.size() ||
            (*parts)[0].readable > bytes.size() - at ||
            cols * rows * sizeof(float) != (*parts)[0].readable)
          return {};
        for (std::size_t n = 0; n != rows; ++n)
          for (std::size_t col = 0; col != cols; ++col) {
            float value = 0.F;
            if (resource.name == "attention.norm") value = 1.F;
            if (resource.name == "head" || resource.name == "embedding" ||
                resource.name == "attention.k" || resource.name == "attention.v")
              value = n == col ? 1.F : 0.F;
            if (resource.name == "moe.router") value = Router(n, col);
            if (resource.name == "second.moe.router") value = col == (n + 2) % 4 ? 65536.F : 0.F;
            std::memcpy(bytes.data() + at + (n * cols + col) * sizeof(float), &value,
                        sizeof(value));
          }
      }
    } else {
      if (size != kExperts * en::kPagedExtent) return {};
      for (std::size_t e = 0; e != kExperts; ++e) {
        auto* data = bytes.data() + begin + e * en::kPagedExtent;
        std::fill_n(data, 1024, static_cast<char>(0x11));
        std::fill_n(data + 1024, 128, static_cast<char>(0x38));
        const std::array<float, 2> scale =
            ordered
                ? (shard == 1 ? std::array{1.F / 16.F, 1.F / 32.F} : std::array{1024.F, 512.F})
                : std::array{static_cast<float>(e + 1) / 16.F, static_cast<float>(e + 1) / 32.F};
        std::memcpy(data + 1280, scale.data(), sizeof(scale));
        if (ordered)
          std::fill_n(data + 1536 + e * 16, 16, static_cast<char>(0x11));
        else
          std::fill_n(data + 1536, 512, static_cast<char>(0x11));
        std::fill_n(data + 2048, 64, static_cast<char>(0x38));
        const float down = ordered && shard == 2 ? 16.F : 1.F / 32.F;
        std::memcpy(data + 2304, &down, 4);
      }
    }
    for (std::uint64_t at = 0; at != size;) {
      const auto count = std::min(en::kPagedExtent, size - at);
      llmp::base::Sha256 hash;
      hash.Update(std::string_view(bytes.data() + begin + at, count));
      files.index["chunk_sha256"].at(chunk++) = llmp::base::ToHex(hash.Finish());
      at += count;
    }
  }
  return ts::Publish(root, files);
}

class KernelOwner final : public en::PagedModel {
 public:
  explicit KernelOwner(en::PagedNode& node) : resources(node, 0, 0) {}
  std::uint32_t stream() const override { return 0; }
  const ca::Closure& fence_closure() const override { return empty; }
  std::vector<ca::ExtentId> managed_extents() const override { return {}; }
  en::Status Release() override {
    backend.reset();
    std::vector<std::string> errors;
    resources.Release(errors);
    return en::support::Joined(errors);
  }
  en::RunnerResources resources;
  std::unique_ptr<kg::GgmlBackend> backend;
  ca::Closure empty;
};
struct FixtureLimits {
  std::uint64_t domain = kDomain, metadata = kMetadata, compiler = kCompiler, owner = kOwner;
  std::uint64_t nodes = kNodes, tensors = kTensors, program = kProgram, launches = kLaunches;
  std::uint64_t workspace = 32 * kMiB, weight_reads = 8 * en::kPagedExtent;
  std::size_t weights = 16;
};
constexpr FixtureLimits kOrderedLimits{.domain = 1024 * kMiB,
                                       .metadata = 576 * kMiB,
                                       .compiler = 384 * kMiB,
                                       .owner = 96 * kMiB,
                                       .nodes = 4096,
                                       .tensors = 16384,
                                       .program = 16 * kMiB,
                                       .launches = 1024,
                                       .workspace = 64 * kMiB,
                                       .weight_reads = 16 * en::kPagedExtent,
                                       .weights = 32};
static_assert(12 * (4096 * 480ULL + 16384 * 192ULL) + 4096 * (4096 + 16384ULL) +
                  (16384 + 16 + 16 * 4096ULL) * (512 + 1024 + 32) + 64 * kMiB <
              kOrderedLimits.compiler);
static_assert(kOrderedLimits.compiler + kOrderedLimits.owner +
                  2 * (kOrderedLimits.program + kOrderedLimits.launches * en::kGraphNodeHostBytes) <
              kOrderedLimits.metadata);
struct Lifetime {
  bool ordered = false;
  FixtureLimits limits;
  std::uint32_t route_groups = 1;
  en::PagedNode node{{.compute_streams = 1, .slots = 1, .slot_bytes = en::kSlabSlotBytes}};
  std::shared_ptr<en::ExecutionMetadataAllowance> paid;
  std::unique_ptr<KernelOwner> kernel;
  std::unique_ptr<en::ModelRunner> runner;
  std::vector<en::PagedModel*> entered;
  sp::ModelSpec spec;
  sp::Source source;
  md::StateLayoutDerived layout;
  std::map<std::pair<ir::WeightRef, Role>, std::uint32_t> resources;
  std::array<ca::ExtentId, 2 * kExperts> experts{};
  ca::ExtentId base{};
  const kn::StepProgram* program = nullptr;
  kn::StepLaunch* launch = nullptr;
  en::PagedWeights* weights = nullptr;
  std::filesystem::path root;
  bool opened = false, retirement_attempted = false;
  std::size_t builds = 0, measured = 0;
};
// One unknown retirement is terminal for this process's finite serial budget.
// An owned-flight cycle must not defer the first explicit teardown until after
// the next GTest case opens a second domain.
bool retirement_blocked = false;
en::Status RetireLifetime(Lifetime& life) {
  if (!life.opened) return {};
  if (life.retirement_attempted) return Error("retirement was already attempted without proof");
  life.retirement_attempted = true;
  auto done = life.node.TearDown(life.entered);
  if (done)
    life.opened = false;
  else
    retirement_blocked = true;
  return done;
}
struct TeardownGuard {
  const std::shared_ptr<Lifetime>& life;
  ~TeardownGuard() {
    if (!life || !life->opened || life->retirement_attempted) return;
    const auto done = RetireLifetime(*life);
    EXPECT_TRUE(done) << llmp::test_support::Failed(done).value_or("");
  }
};
struct Retire {
  void operator()(Lifetime* life) const {
    if (!life) return;
    if (life->opened) {
      if (life->retirement_attempted) return;
      const auto done = RetireLifetime(*life);
      EXPECT_TRUE(done) << llmp::test_support::Failed(done).value_or("");
      if (!done) return;  // retain the complete owner on unknown completion
    }
    std::error_code error;
    if (!life->root.empty()) std::filesystem::remove_all(life->root, error);
    EXPECT_FALSE(error) << error.message();
    delete life;
  }
};

en::Status SetupFixture(const std::shared_ptr<Lifetime>& life, int malformed = 0,
                        bool ordered = false) {
  life->ordered = ordered;
  life->route_groups = ordered ? 2U : 1U;
  if (ordered) life->limits = kOrderedLimits;
  const auto& limits = life->limits;
  if (retirement_blocked) return Error("a prior complete owner has unknown retirement");
  auto& node = life->node;
  const auto available = llmp::platform::AvailableMemoryBytes();
  if (!available || *available < (10ULL << 30) + (6ULL << 30)) return Error("physical margin");
  if (auto r = node.Open(); !r) return r;
  life->opened = true;
  node.SetHostFloor(0);
  auto paid = en::ExecutionMetadataAllowance::Admit(node.catalog(), node.domain(), limits.metadata,
                                                    *available, 6ULL << 30);
  if (!paid) return Error(paid.error());
  life->paid = *paid;
  if (!life->paid->Charge(limits.compiler + limits.owner)) return Error("preconstruction payment");
  // One directory per test: CTest runs this file's tests in parallel from
  // one working directory, and each removes its artifact when it ends.
  life->root = std::filesystem::path("scratch") /
               (std::string("routed-component-") +
                ::testing::UnitTest::GetInstance()->current_test_info()->name());
  life->spec = Spec(ordered);
  if (auto r = sp::Validate(life->spec); !r) return Error(r.error());
  auto weights = std::make_unique<en::PagedWeights>();
  const auto path = Artifact(life->root, life->spec, ordered);
  if (path.empty()) return Error("artifact construction geometry");
  if (auto r = weights->Open(path); !r) return r;
  life->weights = weights.get();
  auto source = sp::SourceOf(weights->artifact());
  if (!source) return Error(source.error());
  life->source = std::move(*source);
  auto layout = co::StageLayout(life->spec, 0, kContext, kRows);
  if (!layout) return Error(layout.error());
  if (layout->items.size() != 2 || layout->bytes > 128 * 1024) return Error("state inventory");
  for (const auto& item : layout->items)
    if (item.contract.encoding != fmt::Encoding::kF16 || item.contract.cell_bytes != 2 * kHead ||
        item.cells != kContext || item.bytes != 2 * std::uint64_t{kHead} * kContext)
      return Error("D64 attention state geometry");
  life->layout = std::move(*layout);
  auto step = co::BuildStep(life->spec, 0, Shape(),
                            {.tensors = life->source.tensors, .state = &life->layout});
  if (!step) return Error(step.error());
  if (step->graph.nodes().size() > limits.nodes || step->graph.tensor_count() > limits.tensors ||
      step->weights.size() > limits.weights)
    return Error("fixed primitive inventory");
  for (const auto& weight : step->weights) {
    if (weight.ref.expert_array) continue;
    for (const auto& [name, role] : weight.parts) {
      const auto resource = weights->artifact().FindResource(name);
      if (!resource || !life->resources.emplace(std::pair{weight.ref, role}, *resource).second)
        return Error("ordinary source bindings");
    }
  }
  life->kernel = std::make_unique<KernelOwner>(node);
  life->entered.push_back(life->kernel.get());
  auto selected = de::GroupedSelectedSource();
  if (!selected) return Error(selected.error());
  auto backend = kg::GgmlBackend::Create(0, node.execution(), node.stream(0), nullptr,
                                         {.selected = *selected});
  if (!backend) return Error(backend.error());
  life->kernel->backend = std::move(*backend);
  const auto facts = llmp::providers::QueryDeviceFacts(0);
  if (!facts) return Error("device facts");
  const auto workspace = kg::CublasHandle::UpstreamWorkspace(static_cast<int>(facts->architecture));
  if (workspace.value() > 64 * kMiB) return Error("handle workspace bound");
  auto* held = life.get();
  if (auto r = life->kernel->backend->Declare(
          {}, workspace.value(), [held](std::uint64_t bytes) { return held->paid->Charge(bytes); },
          [held](std::uint64_t bytes) { held->paid->Release(bytes); });
      !r)
    return Error(r.error());
  if (auto r = life->kernel->resources.OpenCublas("routed shared handle"); !r) return r;
  if (auto r = life->kernel->backend->LendCublas(&life->kernel->resources.cublas()); !r)
    return Error(r.error());
  auto domain = en::NumericalDomain::Create({{.stage = 0,
                                              .name = "target",
                                              .uses = {en::NumericalUse::kScore},
                                              .phases = {{.phase = pl::Phase::kScore,
                                                          .max_owners = 1,
                                                          .max_owner_rows = kRows,
                                                          .max_total_rows = kRows}}}});
  if (!domain) return Error(domain.error());
  const std::array<std::string_view, 1> artifacts{weights->artifact().id()};
  const auto digest = sp::BoundDigest(life->spec, artifacts);
  if (!digest) return Error(digest.error());
  auto component = en::ComponentStepSource::Create(
      {.spec = &life->spec,
       .bindings = {.tensors = life->source.tensors, .state = &life->layout},
       .bound_digest = *digest,
       .slots = 1,
       .policy = pl::Policy{},
       .backend = life->kernel->backend->Backend(),
       .execution_domain = std::make_shared<const en::NumericalDomain>(std::move(*domain)),
       .validate_program = [held](const kn::LowerInput& input,
                                  const kn::StepProgram& program) -> en::Status {
         if (input.plan.graph.nodes().size() > held->limits.nodes ||
             input.plan.graph.tensor_count() > held->limits.tensors || !program.routed() ||
             !program.weight_reads() || program.host_bytes() > held->limits.program ||
             program.launches() > held->limits.launches ||
             program.routed()->groups() != held->route_groups ||
             program.routed()->pinned_bytes() > held->route_groups * 16 * 1024)
           return Error("actual lowered program exceeded paid routed domain");
         std::size_t attention = 0;
         for (const auto& op : input.token->step.graph.nodes()) {
           if (op.key != ir::Primitive(ir::Op::kAttention)) continue;
           if (op.nsrc < 3) return Error("attention operand count");
           const auto& query = input.token->step.graph.tensor(op.src[0]);
           const auto& key = input.token->step.graph.tensor(op.src[1]);
           const auto& value = input.token->step.graph.tensor(op.src[2]);
           if (query.ne(0) != kHead || key.ne(0) != kHead || value.ne(0) != kHead ||
               query.format.encoding != fmt::Encoding::kF32 ||
               key.format.encoding != fmt::Encoding::kF16 ||
               value.format.encoding != fmt::Encoding::kF16)
             return Error("actual attention differs from supported D64 geometry");
           ++attention;
         }
         if (attention != 1) return Error("actual attention operation inventory");
         for (std::size_t group = 0; group != held->route_groups; ++group) {
           const auto& route = program.routed()->layout(group).shape;
           if (route.experts != kExperts || route.used != kUsed || route.tokens != kRows)
             return Error("actual route geometry differs");
           const auto& readback = program.routed()->layout(group).readback;
           if (readback.size() != 5 || readback[0].rows != kRows ||
               readback[0].bytes != kUsed * sizeof(std::int32_t) ||
               readback[0].device_stride != kExperts * sizeof(std::int32_t))
             return Error("actual top-k IDs did not retain their source row pitch");
         }
         std::size_t selected = 0;
         for (const auto& op : input.plan.steps)
           if (held->kernel->backend->registry().at(op.implementation).name ==
               "llmp.decode.grouped.nvfp4.split")
             ++selected;
         if (selected != 2 * held->route_groups)
           return Error("real plan did not select both native split products");
         std::size_t deferred = 0;
         for (const auto& read : *program.weight_reads()) {
           if (!read.dynamic) continue;
           for (const auto& [name, role] : read.parts) {
             const auto part = input.addresses.expert_part(read.ref.stage, name);
             if (!part || part->role != role) return Error("actual deferred source part missing");
             for (const auto& use : program.tensors())
               if (use.role == ir::Role::kWeight && use.address == part->address &&
                   use.bytes == part->bytes)
                 return Error("whole pitched descriptor incorrectly requires eager residency");
             ++deferred;
           }
         }
         if (deferred != 6 * held->route_groups ||
             std::ranges::none_of(program.tensors(),
                                  [](const auto& use) { return use.role == ir::Role::kWeight; }))
           return Error("deferred versus ordinary coverage inventory differs");
         if (input.addresses.measuring())
           ++held->measured;
         else {
           ++held->builds;
           held->program = &program;
         }
         return {};
       }});
  if (!component) return Error(component.error());
  en::RunnerConfig config;
  config.slots = {.model = "routed-components",
                  .slots = 1,
                  .stages = {{.name = "target", .layout = life->layout}},
                  .spec = *digest,
                  .read_align = 256,
                  .spill_place = {},
                  .spill_directory = life->root,
                  .invalidate = {},
                  .drop_plans = {}};
  const auto groups = weights->artifact().groups();
  if (groups.size() != 1 + life->route_groups * kExperts ||
      groups[0].kind != llmp::artifact::GroupKind::kGlobal || groups[0].chunks != 1)
    return Error("fixed ordinary/expert group inventory");
  std::vector<en::GroupPlace> places(groups.size(), en::GroupPlace::kNone);
  places[0] = en::GroupPlace::kDevice;
  for (std::uint32_t e = 0; e < life->route_groups * kExperts; ++e)
    if (groups[1 + e].kind != llmp::artifact::GroupKind::kExpert ||
        groups[1 + e].stored.value() != en::kPagedExtent || groups[1 + e].chunks != 1)
      return Error("fixed expert slab stored geometry");
  // One source-derived slab, not dense placement of expert groups. Each actual
  // stored group is exactly one page; all consumed part offsets are 256-aligned.
  std::vector<en::SlabSpec> slabs;
  slabs.reserve(life->route_groups);
  for (std::uint32_t group = 0; group != life->route_groups; ++group)
    slabs.push_back({.first_group = 1 + group * kExperts,
                     .count = kExperts,
                     .stride = groups[1 + group * kExperts].stored.value(),
                     .alignment = 256});
  config.weights.push_back(
      {.weights = std::move(weights), .places = std::move(places), .slabs = std::move(slabs)});
  config.setup_shapes = {Shape()};
  config.routed_metadata = life->paid;
  if (malformed != 1) config.routed_owner = life;
  config.weight_address = [held, malformed](const ir::WeightRef& ref, Role role) -> std::uint64_t {
    const auto found = held->resources.find({ref, role});
    if (found == held->resources.end()) return 0;
    if (malformed == 2 && held->weights->artifact().resources()[found->second].name == "moe.router")
      return held->weights->group_address(1);
    return held->weights->resource_address(found->second);
  };
  config.launch =
      [held](std::uint64_t) -> std::expected<std::unique_ptr<kn::StepLaunch>, std::string> {
    auto launch = held->kernel->backend->Launch(
        {.base = held->node.pool().base, .size = Bytes(held->node.pool().bytes)});
    if (launch) held->launch = launch->get();
    return launch;
  };
  config.retire = [held] { held->kernel->backend->LaunchReleased(); };
  auto runner = en::ModelRunner::Create(node, std::move(config), std::move(*component), 0, 0);
  if (!runner) return Error(runner.error());
  life->runner = std::move(*runner);
  life->entered.insert(life->entered.begin(), life->runner.get());
  if (auto r = life->runner->Setup(); !r) return r;
  if (malformed != 0) return Error("malformed source unexpectedly admitted");
  const auto activation = life->runner->activations_needed();
  const auto pool = std::max(en::kPagedExtent, life->runner->pool_needed());
  const auto fixed = node.catalog().OccupancyOf(node.domain()).Total().value();
  if (activation > limits.workspace || pool > limits.workspace || fixed > limits.domain ||
      activation + pool + 4 * node.StateCapacity() + limits.weight_reads +
              2 * (limits.program + limits.launches * en::kGraphNodeHostBytes) >
          limits.domain - fixed)
    return Error("complete workspace/weights/state/recording budget");
  if (auto r = node.MapWorkspace(activation, pool); !r) return r;
  std::array<ca::ExtentId, 16> pages{};
  auto count = life->weights->ExtentsTouching(life->weights->group_address(0), 4096, pages);
  if (!count || *count != 1) return Error("base page inventory");
  life->base = pages[0];
  for (std::uint32_t e = 0; e != life->route_groups * kExperts; ++e) {
    count = life->weights->ExtentsTouching(life->weights->group_address(1 + e), en::kPagedExtent,
                                           pages);
    if (!count || *count != 1) return Error("separate expert page inventory");
    life->experts[e] = pages[0];
  }
  if (auto r = node.Start(Bytes(limits.domain)); !r) return r;
  if (auto r = life->runner->Register(); !r) return r;
  if (auto r = life->runner->Bind(); !r) return r;
  node.Run();
  const std::array<std::uint32_t, 1> slot{0};
  return life->runner->slots().SelectSlots(slot);
}

std::expected<std::vector<double>, std::string> Oracle(std::span<const std::int32_t> tokens) {
  if (tokens.size() != kRows) return std::unexpected(std::string("oracle token extent"));
  for (const auto token : tokens)
    if (token < 0 || token >= static_cast<std::int32_t>(kWidth))
      return std::unexpected(std::string("oracle token outside vocabulary"));
  std::vector<double> x(tokens.size() * kWidth);
  for (std::size_t row = 0; row != tokens.size(); ++row)
    x[row * kWidth + static_cast<std::uint32_t>(tokens[row])] = 1;
  std::vector<float> router(kWidth * kExperts), gate(kWidth * 2 * kWidth * kExperts),
      down(kWidth * kWidth * kExperts);
  for (std::size_t e = 0; e != kExperts; ++e) {
    for (std::size_t k = 0; k != kWidth; ++k) router[e * kWidth + k] = Router(e, k);
    for (std::size_t n = 0; n != 2 * kWidth; ++n)
      for (std::size_t k = 0; k != kWidth; ++k)
        gate[(e * 2 * kWidth + n) * kWidth + k] =
            0.5F * static_cast<float>(e + 1) / (n < kWidth ? 16.F : 32.F);
    std::fill_n(down.begin() + static_cast<std::ptrdiff_t>(e * kWidth * kWidth), kWidth * kWidth,
                0.5F / 32.F);
  }
  const auto route = co::moe::RouterReference(
      x, kWidth, router, {},
      {.experts = kExperts, .used = kUsed, .normalize = true, .rms_input = false});
  return co::moe::ExpertsReference(
      x, route, gate, {}, down, {},
      {.width = kWidth, .hidden = kWidth, .stacked = true, .kind = ir::GluKind::kSwiGlu});
}

TEST(RoutedComponentGpu, RealSharedStepDiscoversOnlySelectedExpertsAndReplaysChangedRoutes) {
  std::shared_ptr<Lifetime> life(new Lifetime, Retire{});
  const TeardownGuard retirement{life};
  const auto setup = SetupFixture(life);
  ASSERT_TRUE(setup) << llmp::test_support::Failed(setup).value_or("");
  const auto completed = life->node.WithRequest(
      0, life->runner->slots().execution(), "routed shared component", [&]() -> en::Status {
        const std::array<std::array<std::int32_t, kRows>, 3> input{
            {{0, 0, 1}, {2, 2, 3}, {1, 0, 1}}};
        constexpr std::array path{en::RunPath::kEager, en::RunPath::kCaptured,
                                  en::RunPath::kReplayed};
        for (std::size_t turn = 0; turn != input.size(); ++turn) {
          const en::SegmentWork work{.tokens = input[turn]};
          auto ran =
              life->runner->RunStep({.shape = Shape(static_cast<std::uint32_t>(turn * kRows)),
                                     .work = std::span(&work, 1),
                                     .ahead = {}});
          if (!ran) return Error(ran.error().what);
          if (ran->path != path[turn] || ran->outputs.size() != 1 ||
              ran->outputs[0].size() != kRows * kWidth * sizeof(float))
            return Error("step path/coverage");
          const auto expected = Oracle(input[turn]);
          if (!expected) return Error(expected.error());
          if (expected->size() != kRows * kWidth) return Error("oracle output extent");
          for (std::size_t i = 0; i != expected->size(); ++i) {
            float actual = 0;
            std::memcpy(&actual, ran->outputs[0].data() + i * sizeof(float), sizeof(float));
            if (!std::isfinite(actual) ||
                std::abs(actual - (*expected)[i]) > 2e-5 + 2e-3 * std::abs((*expected)[i]))
              return Error("complete native output differs from scalar component oracle");
          }
          // RunStep completed both segments; no device job remains active.
          auto residency = life->node.Call(
              [&]() -> en::Status {
                const auto base = life->node.catalog().Describe(life->base);
                if (!base || base->state != ca::ExtentState::kResident || base->leases == 0)
                  return Error("base lease lost");
                const auto refuse = life->node.catalog().BeginEvict(life->base);
                if (refuse || refuse.error() != ca::CatalogError::kHeld)
                  return Error("held base page was eligible for eviction");
                for (std::size_t e = 0; e != kExperts; ++e) {
                  const bool selected = turn == 1 ? e >= 2 : e < 2;
                  const auto page = life->node.catalog().Describe(life->experts[e]);
                  if (!page || (page->state == ca::ExtentState::kResident) != selected)
                    return Error("selected-only expert residency differs");
                }
                return {};
              },
              "actual selected residency");
          if (!residency) return residency;
          // The supplemental lease has retired, but the base request remains.
          // Removing these pages forces fresh backing under retained captures.
          if (auto evicted = life->node.Evict(
                  std::vector(life->experts.begin(), life->experts.begin() + kExperts));
              !evicted)
            return evicted;
          if (!life->node.InRequest(0)) return Error("selected eviction ended base request");
        }
        if (life->runner->slots().positions(0) != 9 || life->measured != 1 || life->builds != 1 ||
            life->runner->plans().size() != 1)
          return Error("state/cache frontier");
        auto native = life->node.execution().Submission(life->node.stream(0));
        if (!native || !life->program || !life->launch) return Error("actual source owner");
        if (life->program->Run(*life->launch, *native)) return Error("whole Run bypassed routing");
        bool body_called = false;
        auto recorded = life->program->Record(*life->launch, *native,
                                              [&]() -> std::expected<void, kn::StepFailure> {
                                                body_called = true;
                                                return {};
                                              });
        if (recorded || body_called) return Error("whole Record bypassed routing");
        return {};
      });
  ASSERT_TRUE(completed) << llmp::test_support::Failed(completed).value_or("");
  RecordProperty("attention_head_dimension", kHead);
  RecordProperty("full_output_values", 288);
  RecordProperty("completed_steps", 3);
  RecordProperty("selected_experts_per_step", 2);
  RecordProperty("retained_programs", 1);
  RecordProperty("state_frontier", 9);
  RecordProperty("deferred_part_descriptors", 6);
  // Explicitly break the complete-owner retention cycle only on proven teardown.
  const auto retired = RetireLifetime(*life);
  ASSERT_TRUE(retired) << llmp::test_support::Failed(retired).value_or("");
  life->opened = false;
}

std::expected<std::vector<double>, std::string> OrderedOracle(
    std::span<const std::int32_t> tokens) {
  if (tokens.size() != kRows) return std::unexpected("ordered oracle token extent");
  std::vector<double> x(tokens.size() * kWidth);
  for (std::size_t row = 0; row != tokens.size(); ++row) {
    if (tokens[row] < 0 || tokens[row] >= 4) return std::unexpected("ordered oracle token range");
    x[row * kWidth + static_cast<std::uint32_t>(tokens[row])] = 1;
  }
  for (unsigned group = 0; group != 2; ++group) {
    std::vector<float> router(kWidth * kExperts), gate(kWidth * 2 * kWidth * kExperts),
        down(kWidth * kWidth * kExperts);
    for (std::size_t e = 0; e != kExperts; ++e) {
      for (std::size_t k = 0; k != kWidth; ++k)
        router[e * kWidth + k] = group == 0 ? Router(e, k) : k == (e + 2) % 4 ? 65536.F : 0.F;
      for (std::size_t n = 0; n != 2 * kWidth; ++n)
        for (std::size_t k = 0; k != kWidth; ++k)
          gate[(e * 2 * kWidth + n) * kWidth + k] =
              group == 0 ? 0.5F / (n < kWidth ? 16.F : 32.F) : 0.5F * (n < kWidth ? 1024.F : 512.F);
      for (std::size_t k = 0; k != kWidth; ++k)
        down[(e * kWidth + e) * kWidth + k] = group == 0 ? 0.5F / 32.F : 8.F;
    }
    const auto route = co::moe::RouterReference(
        x, kWidth, router, {},
        {.experts = kExperts, .used = kUsed, .normalize = true, .rms_input = false});
    x = co::moe::ExpertsReference(
        x, route, gate, {}, down, {},
        {.width = kWidth, .hidden = kWidth, .stacked = true, .kind = ir::GluKind::kSwiGlu});
  }
  return x;
}

TEST(RoutedComponentGpu, TwoOrderedGroupsReacquireChangedRoutesBeforeFinalOutputs) {
  std::shared_ptr<Lifetime> life(new Lifetime, Retire{});
  const TeardownGuard retirement{life};
  const auto setup = SetupFixture(life, 0, true);
  ASSERT_TRUE(setup) << llmp::test_support::Failed(setup).value_or("");
  const auto completed = life->node.WithRequest(
      0, life->runner->slots().execution(), "two ordered routed groups", [&]() -> en::Status {
        const std::array<std::array<std::int32_t, kRows>, 3> input{
            {{0, 0, 1}, {2, 2, 3}, {1, 0, 1}}};
        constexpr std::array path{en::RunPath::kEager, en::RunPath::kCaptured,
                                  en::RunPath::kReplayed};
        constexpr std::int32_t wanted[3][2][6] = {{{0, 1, 0, 1, 1, 0}, {2, 3, 2, 3, 3, 2}},
                                                  {{2, 3, 2, 3, 3, 2}, {0, 1, 0, 1, 1, 0}},
                                                  {{1, 0, 0, 1, 1, 0}, {3, 2, 2, 3, 3, 2}}};
        for (std::size_t turn = 0; turn != input.size(); ++turn) {
          const en::SegmentWork work{.tokens = input[turn]};
          auto ran =
              life->runner->RunStep({.shape = Shape(static_cast<std::uint32_t>(turn * kRows)),
                                     .work = std::span(&work, 1),
                                     .ahead = {}});
          if (!ran) return Error(ran.error().what);
          if (ran->path != path[turn] || ran->outputs.size() != 1 ||
              ran->outputs[0].size() != kRows * kWidth * sizeof(float))
            return Error("ordered step path or final output coverage");
          if (!life->program || !life->program->routed() || life->program->routed()->groups() != 2)
            return Error("actual two-group program disappeared");
          for (std::size_t group = 0; group != 2; ++group) {
            // No device read after scratch reuse: the runner copied each actual
            // route at its completed boundary into its stable, retained slice.
            const auto actual = llmp::test_support::RoutedComponentInspection::Completed(
                *life->runner, *life->program->routed(), group);
            if (!actual || actual->first != turn + 1 ||
                !std::ranges::equal(actual->second, wanted[turn][group]))
              return Error("completed actual group IDs or epoch differ");
          }
          if (llmp::test_support::RoutedComponentInspection::Completed(*life->runner,
                                                                       *life->program->routed(), 2))
            return Error("out-of-range group inspection accepted");
          const auto expected = OrderedOracle(input[turn]);
          if (!expected || expected->size() != kRows * kWidth)
            return Error("ordered scalar oracle extent");
          for (std::size_t i = 0; i != expected->size(); ++i) {
            float actual = 0;
            std::memcpy(&actual, ran->outputs[0].data() + i * sizeof(float), sizeof(float));
            if (!std::isfinite(actual) ||
                std::abs(actual - (*expected)[i]) > 2e-5 + 2e-3 * std::abs((*expected)[i]))
              return Error("complete ordered output differs from scalar component oracle");
          }
          const auto residency = life->node.Call(
              [&]() -> en::Status {
                const auto base = life->node.catalog().Describe(life->base);
                if (!base || base->state != ca::ExtentState::kResident || base->leases == 0)
                  return Error("ordered base lease lost");
                const auto evict_base = life->node.catalog().BeginEvict(life->base);
                if (evict_base || evict_base.error() != ca::CatalogError::kHeld)
                  return Error("ordered base became evictable");
                for (std::size_t group = 0; group != 2; ++group)
                  for (std::size_t expert = 0; expert != kExperts; ++expert) {
                    const bool lower_pair = (turn != 1) == (group == 0);
                    const bool selected = lower_pair ? expert < 2 : expert >= 2;
                    const auto page =
                        life->node.catalog().Describe(life->experts[group * kExperts + expert]);
                    if (!page || (page->state == ca::ExtentState::kResident) != selected)
                      return Error("per-group selected-only residency differs");
                  }
                return {};
              },
              "ordered selected residency");
          if (!residency) return residency;
          if (auto evicted =
                  life->node.Evict(std::vector(life->experts.begin(), life->experts.end()));
              !evicted)
            return evicted;
          if (!life->node.InRequest(0)) return Error("ordered eviction ended base request");
        }
        if (life->runner->slots().positions(0) != 9 || life->measured != 1 || life->builds != 1 ||
            life->runner->plans().size() != 1)
          return Error("ordered state/cache frontier");
        return {};
      });
  ASSERT_TRUE(completed) << llmp::test_support::Failed(completed).value_or("");
  RecordProperty("full_output_values", 288);
  RecordProperty("completed_steps", 3);
  RecordProperty("ordered_groups_per_step", 2);
  RecordProperty("completed_segments", 9);
  RecordProperty("actual_route_ids_compared", 36);
  RecordProperty("selected_experts_per_group", 2);
  RecordProperty("retained_programs", 1);
  RecordProperty("state_frontier", 9);
  RecordProperty("deferred_part_descriptors", 12);
  const auto retired = RetireLifetime(*life);
  ASSERT_TRUE(retired) << llmp::test_support::Failed(retired).value_or("");
}

TEST(RoutedComponentGpu, MissingOwnerAndMisboundOrdinaryReaderRefuseBeforeAdmission) {
  for (const int malformed : {1, 2}) {
    SCOPED_TRACE(malformed);
    std::shared_ptr<Lifetime> life(new Lifetime, Retire{});
    const TeardownGuard retirement{life};
    const auto setup = SetupFixture(life, malformed);
    ASSERT_FALSE(setup);
    ASSERT_EQ(setup.error(),
              malformed == 1
                  ? "routed source needs complete declared reads, execution identity and owner"
                  : "ordinary callback aliases a different source resource");
    EXPECT_EQ(life->builds, 0U);
    for (const auto id : life->weights->extents()) {
      const auto page = life->node.catalog().Describe(id);
      ASSERT_TRUE(page);
      EXPECT_NE(page->state, ca::ExtentState::kResident);
    }
    life.reset();
    if (::testing::Test::HasFailure()) return;  // no second domain after unknown retirement
  }
}

// A declaration-only test adapter: it never supplies or launches a program.
// Its mutable identity deliberately violates the production immutable-source
// contract so the real backend's readiness barrier can be challenged.
class DeclarationSource final : public kg::SelectedSource {
 public:
  explicit DeclarationSource(std::shared_ptr<const kg::SelectedSource> base)
      : base_(std::move(base)) {}
  std::optional<llmp::base::Sha256Digest> identity{};
  std::uint64_t persistent = 0;
  std::vector<llmp::execution::Implementation> Implementations() const override {
    return base_->Implementations();
  }
  std::vector<llmp::execution::TunedShape> Tuning(std::uint32_t) const override { return {}; }
  bool Owns(std::string_view name) const override { return base_->Owns(name); }
  std::optional<llmp::base::Sha256Digest> ExecutionIdentity() const override { return identity; }
  std::uint64_t persistent_bytes() const override { return persistent; }
  std::expected<std::unique_ptr<kg::SelectedCall>, kg::KernelFailure> Lower(
      const llmp::execution::Implementation&, const ir::Node&, std::span<const kg::SelectedOperand>,
      std::span<const kg::SelectedOperand>) const override {
    return std::unexpected(kg::KernelFailure{.detail = "declaration-only control"});
  }
  std::expected<std::unique_ptr<kg::SelectedLaunch>, kg::KernelFailure> Launch(
      int, llmp::providers::DeviceExecution&, llmp::providers::StreamId,
      kg::SelectedWorkspace) const override {
    return std::unexpected(kg::KernelFailure{.detail = "declaration-only control"});
  }

 private:
  std::shared_ptr<const kg::SelectedSource> base_;
};
TEST(RoutedComponentGpu, OwnedBackendSealsAdapterIdentityAndRejectsLaterSubstitution) {
  std::shared_ptr<Lifetime> life(new Lifetime, Retire{});
  const TeardownGuard retirement{life};
  ASSERT_FALSE(retirement_blocked);
  const auto available = llmp::platform::AvailableMemoryBytes();
  ASSERT_TRUE(available);
  ASSERT_GE(*available, (10ULL << 30) + (6ULL << 30));
  ASSERT_TRUE(life->node.Open());
  life->opened = true;
  life->node.SetHostFloor(0);
  auto paid = en::ExecutionMetadataAllowance::Admit(life->node.catalog(), life->node.domain(),
                                                    kMetadata, *available, 6ULL << 30);
  ASSERT_TRUE(paid) << llmp::test_support::Failed(paid).value_or("");
  life->paid = *paid;
  ASSERT_TRUE(life->paid->Charge(kOwner));
  life->kernel = std::make_unique<KernelOwner>(life->node);
  life->entered.push_back(life->kernel.get());
  const auto facts = llmp::providers::QueryDeviceFacts(0);
  ASSERT_TRUE(facts);
  const auto workspace = kg::CublasHandle::UpstreamWorkspace(static_cast<int>(facts->architecture));
  ASSERT_LE(workspace.value(), 64 * kMiB);
  ASSERT_TRUE(life->kernel->resources.OpenCublas("declared native adapter control"));
  const auto real = de::GroupedSelectedSource();
  ASSERT_TRUE(real);
  auto source = std::make_shared<DeclarationSource>(*real);
  auto backend = kg::GgmlBackend::Create(0, life->node.execution(), life->node.stream(0), nullptr,
                                         {.selected = source});
  ASSERT_TRUE(backend);
  life->kernel->backend = std::move(*backend);
  const auto declare = [&] {
    return life->kernel->backend->Declare(
        {}, workspace.value(),
        [held = life.get()](std::uint64_t n) { return held->paid->Charge(n); },
        [held = life.get()](std::uint64_t n) { held->paid->Release(n); });
  };
  EXPECT_FALSE(declare());  // Default/absent selected identity remains refused.
  source->identity = (*real)->ExecutionIdentity();
  ASSERT_TRUE(declare());
  ASSERT_TRUE(life->kernel->backend->LendCublas(&life->kernel->resources.cublas()));
  const auto owner = life->kernel->backend->Backend().owner;
  ASSERT_TRUE(owner);
  const auto& device = owner->backend().device;
  ASSERT_EQ(device.arch, facts->architecture / 10);
  ASSERT_GE(device.arch, 80U);
  ASSERT_NE(device.features & llmp::execution::kBf16Mma, 0U);
  // Query both actual routed products through the declared backend registry.
  // Their K32 geometry is supported; a missing BF16 MMA fact must still refuse.
  namespace ex = llmp::execution;
  for (const std::uint64_t outputs : {64ULL, 32ULL}) {
    const std::uint64_t lanes = outputs == 64 ? 1 : kUsed;
    const std::array operands{ex::OperandFormat{.operand = 0,
                                                .encoding = fmt::Encoding::kNvfp4,
                                                .layout = fmt::Layout::kSplit,
                                                .ne = {kWidth, outputs, kExperts, 1}},
                              ex::OperandFormat{.operand = 1,
                                                .encoding = fmt::Encoding::kF32,
                                                .layout = fmt::Layout::kDense,
                                                .ne = {kWidth, lanes, kRows, 1}},
                              ex::OperandFormat{.operand = 2,
                                                .encoding = fmt::Encoding::kI32,
                                                .layout = fmt::Layout::kDense,
                                                .ne = {kUsed, kRows, 1, 1}}};
    ex::CapabilityQuery query{.op = ir::Primitive(ir::Op::kMatMulId),
                              .operands = operands,
                              .rows = kRows,
                              .k = kWidth,
                              .n = outputs,
                              .arch = device.arch,
                              .features = device.features};
    const auto selected = ex::Candidates(life->kernel->backend->registry(), query);
    ASSERT_EQ(selected.size(), 1U);
    EXPECT_EQ(life->kernel->backend->registry().at(selected[0]).name,
              "llmp.decode.grouped.nvfp4.split");
    query.features &= ~static_cast<std::uint32_t>(ex::kBf16Mma);
    EXPECT_TRUE(ex::Candidates(life->kernel->backend->registry(), query).empty());
    query.features = device.features;
    query.k = kWidth - 1;
    EXPECT_TRUE(ex::Candidates(life->kernel->backend->registry(), query).empty());
  }
  auto first = owner->Ready();
  ASSERT_TRUE(first);
  source->identity->at(0) ^= 1U;
  EXPECT_FALSE(owner->Ready());
  source->identity = (*real)->ExecutionIdentity();
  EXPECT_TRUE(owner->Ready());
  source->persistent = 256;
  EXPECT_FALSE(owner->Ready());
  source->persistent = 0;
  EXPECT_TRUE(owner->Ready());
  // Same registry declarations, different explicit adapter identity: it must
  // produce a different execution recipe, not collide on implementation names.
  source->identity->at(0) ^= 1U;
  auto other = kg::GgmlBackend::Create(0, life->node.execution(), life->node.stream(0), nullptr,
                                       {.selected = source});
  ASSERT_TRUE(other);
  ASSERT_TRUE((*other)->Declare(
      {}, workspace.value(), [held = life.get()](std::uint64_t n) { return held->paid->Charge(n); },
      [held = life.get()](std::uint64_t n) { held->paid->Release(n); }));
  ASSERT_TRUE((*other)->LendCublas(&life->kernel->resources.cublas()));
  auto second = (*other)->Backend().owner->Ready();
  ASSERT_TRUE(second);
  EXPECT_NE(first->recipe, second->recipe);
  EXPECT_NE(first->selection, second->selection);
}
}  // namespace
