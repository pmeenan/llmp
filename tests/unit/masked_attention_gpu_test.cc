// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Opt-in masked D128 attention: the same IR, owned plan and completed ports.
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <format>
#include <limits>
#include <memory>
#include <optional>
#include <vector>

#include "base/sha256.h"
#include "engine/coverage.h"
#include "engine/execution_contract.h"
#include "engine/graph_plan.h"
#include "engine/plan_cache.h"
#include "engine/runner_resources.h"
#include "engine/step_runs.h"
#include "engine/support.h"
#include "expected_error.h"
#include "ir/builder.h"
#include "ir/decompose.h"
#include "kernels/ggml/attention_f32.h"
#include "kernels/ggml/step_program.h"
#include "platform/host_probe.h"
#include "providers/device_runtime.h"

namespace {
namespace en = llmp::engine;
namespace ir = llmp::ir;
namespace kg = llmp::kernels::ggml;
namespace kn = llmp::kernels;
namespace ca = llmp::catalog;
namespace fm = llmp::format;
namespace sc = llmp::scheduler;
using llmp::base::Bytes;
using llmp::test_support::Failed;
constexpr std::uint64_t kMiB = 1ULL << 20U;
constexpr std::uint64_t kDomain = 512 * kMiB, kMetadata = 96 * kMiB;
constexpr std::uint64_t kConstruction = 64 * kMiB, kActivation = 128 * kMiB;
constexpr std::uint64_t kPool = 16 * kMiB, kPinned = 32 * kMiB;
constexpr std::uint64_t kOutputOffset = 16 * kMiB, kGuard = 256;
constexpr std::uint64_t kProgram = 4 * kMiB, kMaxNodes = 32, kMaxTensors = 64;
// Two retained plans, up to four live graph generations during each build,
// native descriptors and capture metadata, plus 16 MiB deterministic inputs.
static_assert(4 * 2 * (kMaxNodes * 512 + kMaxTensors * 512) + 4096 * (kMaxNodes + kMaxTensors) +
                  2 * kProgram + 2 * 32 * en::kGraphNodeHostBytes + 16 * kMiB + 16 * kMiB <
              kConstruction);
static_assert(kMetadata + kActivation + kPool + kPinned + 64 * kMiB + 4 * kMiB < kDomain);
en::Status Error(std::string message) { return std::unexpected(std::move(message)); }
std::string Hash(std::span<const float> values) {
  llmp::base::Sha256 h;
  h.Update({reinterpret_cast<const char*>(values.data()), values.size_bytes()});
  return llmp::base::ToHex(h.Finish());
}
struct Arm {
  std::optional<ir::Graph> graph;
  std::unique_ptr<kn::StepProgram> program;
  en::StepPlanRuns runs;
};
struct Lifetime final : en::PagedModel {
  en::PagedNode node{{.compute_streams = 1, .slots = 1}};
  en::RunnerResources resources{node, 0, 0};
  std::shared_ptr<en::ExecutionMetadataAllowance> paid;
  std::unique_ptr<kg::GgmlBackend> backend;
  std::unique_ptr<kn::StepLaunch> launch;
  std::array<Arm, 2> arms;
  std::array<std::vector<float>, 4> inputs;
  std::size_t rows = 0, heads = 0;
  std::byte* host = nullptr;
  en::StepRuns step_runs{true};
  en::GraphStats stats;
  ca::Closure closure, empty;
  bool opened = false, attempted = false, retired = false;
  std::uint32_t stream() const override { return 0; }
  const ca::Closure& fence_closure() const override { return empty; }
  std::vector<ca::ExtentId> managed_extents() const override { return {}; }
  en::Status Release() override {
    for (auto& arm : arms) {
      arm.runs.DropGraph();
      arm.program.reset();
      arm.graph.reset();
    }
    step_runs.SetLaunch(nullptr);
    launch.reset();
    if (backend) backend->LaunchReleased();
    backend.reset();
    std::vector<std::string> errors;
    resources.Release(errors);
    return en::support::Joined(errors);
  }
  en::Status Retire() {
    if (!opened) return {};
    if (attempted) return retired ? en::Status{} : Error("first retirement unproven");
    attempted = true;
    const std::array<en::PagedModel*, 1> models{this};
    auto result = node.TearDown(models);
    retired = result.has_value();
    return result;
  }
};
struct Retire {
  void operator()(Lifetime* life) const {
    if (!life) return;
    auto done = life->Retire();
    EXPECT_TRUE(done) << Failed(done).value_or("");
    if (done) delete life;  // Unknown completion retains node, source, plans and all borrowers.
  }
};
void Fill(Lifetime& life, bool changed, bool wide = false) {
  const auto rows = life.rows, heads = life.heads;
  for (std::size_t source = 0; source < 3; ++source) {
    auto& values = life.inputs[source];
    values.resize(128 * rows * heads);
    for (std::size_t n = 0; n < values.size(); ++n)
      values[n] =
          static_cast<float>(static_cast<int>((n * 17 + source * 29 + 7) % 257) - 128) / 256.F;
  }
  if (wide) std::ranges::fill(life.inputs[2], 70000.F);
  auto& mask = life.inputs[3];
  mask.resize(rows * rows);
  const auto text = rows == 7 ? 3U : 25U;
  for (std::size_t row = 0; row < rows; ++row)
    for (std::size_t key = 0; key < rows; ++key) {
      // Full source semantics: text queries causal within text; image queries
      // see all text/image keys. The final row is deliberately fully masked.
      float value = (row < text && key > row) || row == rows - 1
                        ? -std::numeric_limits<float>::infinity()
                        : 0.F;
      if (std::isfinite(value) && row >= text && (key % 7) == 2) value = -0.137F;
      if (changed && row == text && key != 0) value = -std::numeric_limits<float>::infinity();
      mask[key + rows * row] = value;
    }
  if (changed) life.inputs[1][0] += .25F;
}
std::expected<ir::Graph, std::string> Graph(std::size_t rows, std::size_t heads, bool direct) {
  ir::Builder b;
  b.SetOwner(0);
  const auto q = b.Input(ir::Dense(fm::Encoding::kF32, ir::ShapeOf({128, rows, heads, 1})));
  const auto k = b.Input(ir::Dense(fm::Encoding::kF32, ir::ShapeOf({128, rows, heads, 1})));
  // A real transposed V source exercises strided conversion rather than
  // constructing only the easiest packed descriptor.
  const auto raw = b.Input(ir::Dense(fm::Encoding::kF32, ir::ShapeOf({rows, 128, heads, 1})));
  const auto v = b.Transpose(raw);
  const auto mask = b.Input(ir::Dense(fm::Encoding::kF32, ir::ShapeOf({rows, rows, 1, 1})));
  b.Output(b.Attention(q, k, v, mask, {}, {.scale = 0.08838834764831845F}));
  auto graph = std::move(b).Finish();
  if (!graph) return std::unexpected(graph.error().detail);
  if (direct) {
    auto expanded = ir::Expand(*graph, true);
    if (!expanded) return std::unexpected(expanded.error().detail);
    return std::move(*expanded);
  }
  return std::move(*graph);
}
kn::Addresses Addresses(Lifetime& life, bool measuring) {
  kn::Addresses a;
  a.activations = measuring ? 0 : life.node.activations().base + kGuard;
  a.activation_bytes = kActivation;
  a.activation_alignment = 256;
  return a;
}
en::Status Check(const Lifetime& life, const kn::StepProgram& program, std::size_t arm) {
  if (program.inputs().size() != 4 || program.outputs().size() != 1 ||
      program.outputs()[0].bytes != 128 * life.rows * life.heads * 4 ||
      program.scratch_bytes() > kActivation || program.pool_bytes() > kPool ||
      program.host_bytes() > kProgram || program.launches() > 32 || !program.capturable() ||
      program.routed())
    return Error("attention program exceeds paid typed envelope");
  for (std::size_t i = 0; i < 4; ++i)
    if (program.inputs()[i].kind != kn::InputKind::kStepBytes ||
        program.inputs()[i].bytes != life.inputs[i].size() * 4)
      return Error("complete attention input ports");
  if (program.bound()) {
    std::size_t count = 0;
    for (std::size_t i = 0; i < program.bound()->size(); ++i)
      count += program.bound()->at(i).name == kg::kAttentionF32 ? 1U : 0U;
    if (count != (arm == 1 ? 1U : 0U)) return Error("actual selected attention implementation");
  }
  return {};
}
en::Status SetupFixture(Lifetime& life, bool small) {
  life.rows = small ? 7 : 1057;
  life.heads = small ? 2 : 4;
  auto available = llmp::platform::AvailableMemoryBytes();
  if (!available || *available < 8ULL * 1024 * kMiB) return Error("physical headroom");
  if (auto r = life.node.Open(); !r) return r;
  life.opened = true;
  life.node.SetHostFloor(0);
  auto paid = en::ExecutionMetadataAllowance::Admit(life.node.catalog(), life.node.domain(),
                                                    kMetadata, *available, 6ULL << 30U);
  if (!paid) return Error(paid.error());
  life.paid = *paid;
  if (!life.paid->Charge(kConstruction)) return Error("attention preconstruction payment");
  Fill(life, false);
  for (std::size_t i = 0; i < 2; ++i) {
    auto graph = Graph(life.rows, life.heads, i == 0);
    if (!graph) return Error(graph.error());
    if (graph->nodes().size() > kMaxNodes || graph->tensor_count() > kMaxTensors)
      return Error("fixed graph inventory");
    life.arms[i].graph.emplace(std::move(*graph));
  }
  auto backend = kg::GgmlBackend::Create(0, life.node.execution(), life.node.stream(0), nullptr);
  if (!backend) return Error(backend.error());
  life.backend = std::move(*backend);
  auto facts = llmp::providers::QueryDeviceFacts(0);
  if (!facts) return Error("actual device facts");
  const auto workspace = kg::CublasHandle::UpstreamWorkspace(static_cast<int>(facts->architecture));
  if (workspace.value() > 64 * kMiB) return Error("handle workspace");
  const std::array<llmp::execution::TunedShape, 1> tuning{
      {{.implementation = kg::kAttentionF32,
        .rows = life.rows,
        .k = 128,
        .n = 128,
        .output_encoding = fm::Encoding::kF32}}};
  if (auto r = life.backend->Declare(
          tuning, workspace.value(), [p = life.paid](auto n) { return p->Charge(n); },
          [p = life.paid](auto n) { p->Release(n); });
      !r)
    return Error(r.error());
  if (auto r = life.resources.OpenCublas("masked attention handle"); !r) return r;
  if (auto r = life.backend->LendCublas(&life.resources.cublas()); !r) return Error(r.error());
  const auto owned = life.backend->Backend();
  if (!owned.owner || !owned.owner->Ready()) return Error("backend ownership");
  const auto& table = owned.owner->backend();
  for (bool measuring : {true, false}) {
    if (!measuring) {
      if (auto r = life.node.MapWorkspace(kActivation + 2 * kGuard, kPool + 2 * kGuard); !r)
        return r;
      auto launch = life.backend->Launch({life.node.pool().base + kGuard, Bytes(0)});
      if (!launch) return Error(launch.error());
      life.launch = std::move(*launch);
    }
    for (std::size_t i = 0; i < 2; ++i) {
      auto& arm = life.arms[i];
      en::GraphBuildIO io{*arm.graph, arm.graph->inputs(), arm.graph->outputs(), {}};
      auto program = en::BuildComponentProgram(
          io, table, {}, Addresses(life, measuring),
          {.staged = std::vector(io.inputs.begin(), io.inputs.end()),
           .written = table.written,
           .within = table.within,
           .alignment_staged = std::vector(io.inputs.begin(), io.inputs.end())});
      if (!program) return Error(program.error());
      if (auto r = Check(life, **program, i); !r) return r;
      if (!measuring) {
        arm.program = std::move(*program);
        if (i == 1) {
          const auto refused = arm.program->Bind(*life.launch);
          if (refused || refused.error().detail.find("exceeds the pool") == std::string::npos)
            return Error("underpaid attention was not refused before launch");
        }
      }
    }
  }
  life.launch.reset();
  life.backend->LaunchReleased();
  if (auto r = life.backend->LendCublas(&life.resources.cublas()); !r) return Error(r.error());
  auto launch = life.backend->Launch({life.node.pool().base + kGuard, Bytes(kPool)});
  if (!launch) return Error(launch.error());
  life.launch = std::move(*launch);
  for (std::size_t i = 0; i < 2; ++i) {
    auto& program = *life.arms[i].program;
    if (auto r = program.Bind(*life.launch); !r) return Error(r.error().detail);
    if (!program.bound()) return Error("unbound attention");
    if (auto r = Check(life, program, i); !r) return r;
  }
  ::testing::Test::RecordProperty(
      small ? "small_underpayment_refusals" : "representative_underpayment_refusals", "1");
  auto pinned = life.resources.Pinned(kPinned);
  if (!pinned) return Error(pinned.error());
  life.host = static_cast<std::byte*>(*pinned);
  life.step_runs.SetStaging(life.host, 12 * kMiB);
  life.step_runs.SetLaunch(life.launch.get());
  if (auto r = life.node.Start(Bytes(kDomain)); !r) return r;
  auto extents = life.resources.extents();
  for (const auto& more : {life.node.activations().extents, life.node.pool().extents})
    extents.insert(extents.end(), more.begin(), more.end());
  auto closure = life.node.catalog().ClosureOfExtents(extents);
  if (!closure) return Error("complete attention closure");
  life.closure = std::move(*closure);
  life.node.Run();
  return {};
}
// Independent scalar F64 softmax/value formula. No candidate packing or kernel
// code is used. Small case checks every output; large case all128 channels for
// text/image, changed-mask and padded-tail query boundaries, all heads.
struct Sample {
  std::size_t index;
  double value;
};
std::vector<Sample> Reference(const Lifetime& life, bool small) {
  std::vector<Sample> samples;
  const auto rows = life.rows, heads = life.heads;
  std::vector<double> probabilities(rows);
  for (std::size_t head = 0; head < heads; ++head)
    for (std::size_t query = 0; query < rows; ++query) {
      if (!small && query != 0 && query != 24 && query != 25 && query != 1023 && query != 1024 &&
          query != rows - 2 && query != rows - 1)
        continue;
      double maximum = -std::numeric_limits<double>::infinity();
      for (std::size_t key = 0; key < rows; ++key) {
        double dot = 0;
        for (std::size_t d = 0; d < 128; ++d)
          dot += static_cast<double>(life.inputs[0][d + 128 * (query + rows * head)]) *
                 life.inputs[1][d + 128 * (key + rows * head)];
        probabilities[key] =
            dot * static_cast<double>(0.08838834764831845F) + life.inputs[3][key + rows * query];
        maximum = std::max(maximum, probabilities[key]);
      }
      double sum = 0;
      for (auto& p : probabilities) {
        p = std::isfinite(maximum) ? std::exp(p - maximum) : 0;
        sum += p;
      }
      for (std::size_t d = 0; d < 128; ++d) {
        double value = 0;
        for (std::size_t key = 0; key < rows; ++key)
          value += probabilities[key] * life.inputs[2][key + rows * (d + 128 * head)];
        samples.push_back({d + 128 * (head + heads * query), sum > 0 ? value / sum : 0});
      }
    }
  return samples;
}
en::Status Canaries(Lifetime& life, bool initialize) {
  const std::array addresses{life.node.activations().base,
                             life.node.activations().base + kGuard + kActivation,
                             life.node.pool().base, life.node.pool().base + kGuard + kPool};
  for (std::size_t i = 0; i < addresses.size(); ++i) {
    auto* host = life.host + 28 * kMiB + i * kGuard;
    if (initialize) std::memset(host, 0xa5, kGuard);
    if (auto r = life.node.Copy(0, life.closure, addresses[i], host, kGuard, !initialize,
                                "attention boundary canary");
        !r)
      return r;
    if (!std::ranges::all_of(std::span(host, kGuard), [](auto b) { return b == std::byte{0xa5}; }))
      return Error("attention canary changed");
  }
  return {};
}
en::Status Execute(Lifetime& life, bool small) {
  return life.node.WithRequest(
      0, life.closure, "masked attention representative", [&]() -> en::Status {
        for (std::size_t arm_index = 0; arm_index < 2; ++arm_index) {
          auto& arm = life.arms[arm_index];
          const auto& program = *arm.program;
          const auto prefix = std::string(small ? "small_" : "representative_") +
                              (arm_index == 0 ? "primitive_" : "flash_");
          auto coverage = life.node.Call(
              [&]() -> en::Status {
                en::Coverage result;
                en::CheckCoverage(life.node, 0, program.tensors(), "attention", result);
                return result.violations ? Error(result.first_violation) : en::Status{};
              },
              "attention coverage");
          if (!coverage) return coverage;
          if (auto r = Canaries(life, true); !r) return r;
          std::string original, changed;
          std::uint64_t finite = 0;
          double worst = 0;
          for (std::size_t repeat = 0; repeat < (small ? 10U : 9U); ++repeat) {
            const bool wide = repeat == 9;
            Fill(life, repeat == 7, wide);
            const auto reference = Reference(life, small);
            const auto output_bytes = program.outputs()[0].bytes;
            std::memset(life.host + kOutputOffset - kGuard, 0x5a, kGuard);
            std::memset(life.host + kOutputOffset + output_bytes, 0x5a, kGuard);
            auto staged = life.step_runs.Stage(
                program.inputs(),
                [&](std::size_t slot, std::span<std::byte> out) -> en::Status {
                  if (slot >= 4 || out.size() != life.inputs[slot].size() * 4)
                    return Error("attention staged port");
                  std::memcpy(out.data(), life.inputs[slot].data(), out.size());
                  return {};
                },
                0);
            if (!staged) return Error(staged.error());
            const std::array<en::RunCopy, 1> outputs{
                {{en::support::Address(life.host + kOutputOffset),
                  program.outputs()[0].device_address, output_bytes}}};
            en::StepQueued queued;
            const auto begin = std::chrono::steady_clock::now();
            auto done = life.node.Job(
                life.closure,
                [&](llmp::providers::NativeStream native) {
                  queued = life.step_runs.Queue(arm.runs, *staged, {}, program, outputs,
                                                arm.runs.CaptureDue(true), life.stats, native);
                  if (queued.result) return sc::JobResult::kQueued;
                  return queued.before || queued.result.error().error == kn::StepError::kUnknown
                             ? sc::JobResult::kUnknown
                             : sc::JobResult::kNotStarted;
                },
                "completed attention inputs/compute/output", 0);
            const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now() - begin)
                                .count();
            if (!done) return done;
            if (!queued.result) return Error(queued.result.error().detail);
            const auto expected = repeat == 0   ? en::RunPath::kEager
                                  : repeat == 1 ? en::RunPath::kCaptured
                                                : en::RunPath::kReplayed;
            if (queued.path != expected) return Error("required attention capture/replay path");
            en::Count(life.stats, queued.path);
            const std::span actual(reinterpret_cast<const float*>(life.host + kOutputOffset),
                                   static_cast<std::size_t>(output_bytes / 4));
            const auto nonfinite =
                std::ranges::count_if(actual, [](float value) { return !std::isfinite(value); });
            if (wide && arm_index == 1) {
              // Negative numerical-domain witness, not a qualified output.
              if (nonfinite == 0) return Error("wide-value witness did not expose F16 overflow");
              ::testing::Test::RecordProperty(prefix + "wide_nonfinite", std::to_string(nonfinite));
            } else if (nonfinite != 0)
              return Error("nonfinite qualified attention output");
            if (!wide) finite += actual.size();
            // Inputs <=.5, score dot <=32, scaled <=2.83. F16/TF32
            // one-ULP operand errors perturb scores <.006; finite mask bias
            // adds <.00014. Softmax L1 Lipschitz <=2 and |V|<=.5 give
            // <.0062 output, plus value-round/FP32 reduction allowance.
            // 0.01 is frozen before GPU, independent of observed differences.
            for (const auto& sample : reference) {
              if (wide && arm_index == 1) continue;
              const auto error = std::abs(static_cast<double>(actual[sample.index]) - sample.value);
              if (error > (wide ? 256.0 : 0.01))
                return Error(
                    std::format("F64 attention bound index{} error{}", sample.index, error));
              if (!wide) worst = std::max(worst, error);
            }
            const auto digest = Hash(actual);
            if (wide) {
              if (arm_index == 0)
                ::testing::Test::RecordProperty(prefix + "wide_finite",
                                                std::to_string(actual.size()));
            } else if (repeat == 0)
              original = digest;
            else if (repeat == 7) {
              if (digest == original) return Error("changed mask not observed");
              changed = digest;
            } else if (digest != original)
              return Error("own replay/restoration differs");
            for (auto* guard :
                 {life.host + kOutputOffset - kGuard, life.host + kOutputOffset + output_bytes})
              if (!std::ranges::all_of(std::span(guard, kGuard),
                                       [](auto b) { return b == std::byte{0x5a}; }))
                return Error("output copy canary changed");
            if (auto r = Canaries(life, false); !r) return r;
            ::testing::Test::RecordProperty(prefix + "step" + std::to_string(repeat) + "_ns",
                                            std::to_string(ns));
          }
          ::testing::Test::RecordProperty(prefix + "sha", original);
          ::testing::Test::RecordProperty(prefix + "changed_sha", changed);
          ::testing::Test::RecordProperty(prefix + "finite", std::to_string(finite));
          ::testing::Test::RecordProperty(prefix + "max_error", std::format("{:.17g}", worst));
          ::testing::Test::RecordProperty(prefix + "activation",
                                          std::to_string(program.scratch_bytes()));
          ::testing::Test::RecordProperty(prefix + "pool", std::to_string(program.pool_bytes()));
          ::testing::Test::RecordProperty(prefix + "launch_bound",
                                          std::to_string(program.launches()));
          ::testing::Test::RecordProperty(prefix + "plan",
                                          llmp::base::ToHex(program.bound()->identity()));
        }
        return {};
      });
}
TEST(MaskedAttentionGpuTest, PrimitiveAndPackedMaskPreserveCompletedChangedReplay) {
  for (bool small : {true, false}) {
    std::unique_ptr<Lifetime, Retire> life(new Lifetime);
    const auto setup = SetupFixture(*life, small);
    ASSERT_TRUE(setup) << Failed(setup).value_or("");
    const auto result = Execute(*life, small);
    ASSERT_TRUE(result) << Failed(result).value_or("");
    const auto retired = life->Retire();
    ASSERT_TRUE(retired) << Failed(retired).value_or("");
  }
}
}  // namespace
