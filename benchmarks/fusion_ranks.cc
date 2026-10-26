// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Fused-versus-decomposed screen for the coarse fusion rows
// (execution/rank_data.h FusionRank; docs/experiments/fusion-ranks/README.md).
// One case's graph is its fused node's own decomposition, planned twice with
// the GGML backend's ordinary planner: once as the policy stands (the pattern
// rewrites it back into the fusion) and once with `disable <pattern>`. Both
// plans are lowered and bound normally, run once and compared, captured as
// graphs of repeated runs and timed alternately with CUDA events. The result
// is a FusionRank initializer to review and paste into
// src/kernels/ggml/measured_ranks.h, with the measured point; its row and
// width bands are the reviewer's, from the cases measured. No checkpoints.
//
//   llmp_fusion_ranks rms_norm_mul WIDTH ROWS
//   llmp_fusion_ranks spatial_channel_norm W H C [silu]

#include "execution/fusion_ranks.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <expected>
#include <format>
#include <memory>
#include <optional>
#include <print>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "base/bytes.h"
#include "engine/graph_plan.h"
#include "ir/builder.h"
#include "ir/decompose.h"
#include "kernels/common/device_backend.h"
#include "planner/select.h"
#include "providers/cuda/cuda_device_execution.h"

namespace {
namespace ex = llmp::execution;
namespace ir = llmp::ir;
namespace pl = llmp::planner;
namespace kn = llmp::kernels;
namespace en = llmp::engine;
using llmp::base::Bytes;
using llmp::format::Encoding;
constexpr std::uint64_t kMiB = 1ULL << 20U;
constexpr std::uint64_t kPool = 256 * kMiB;
constexpr std::uint64_t kMetadata = 256 * kMiB;
constexpr int kSamples = 9, kWarmups = 3;

[[noreturn]] void Fail(std::string_view message) {
  std::println(stderr, "llmp_fusion_ranks: {}", message);
  std::fflush(stderr);
  std::abort();  // process exit retains every owner until context teardown
}
void Check(cudaError_t error) {
  if (error != cudaSuccess) Fail(cudaGetErrorString(error));
}
template <class T, class E>
T Must(std::expected<T, E> result) {
  if (!result) {
    if constexpr (requires { result.error().detail; })
      Fail(result.error().detail);
    else if constexpr (requires { std::string_view(result.error()); })
      Fail(result.error());
    else
      Fail("provider operation refused");
  }
  if constexpr (!std::is_void_v<T>) return std::move(*result);
}

// A case: the fused node in a graph of F32 inputs, and its pattern's name.
struct Case {
  std::string label;
  std::string pattern;
  ir::OpKey op{};
  ir::Graph graph;
};

Case RmsNormMul(std::uint64_t width, std::uint64_t rows) {
  ir::Builder b;
  b.SetOwner(0);
  const auto x = b.Input(ir::Dense(Encoding::kF32, ir::ShapeOf({width, rows})));
  const auto w = b.Input(ir::Dense(Encoding::kF32, ir::ShapeOf({width})));
  const std::array sources{x, w};
  b.Output(b.Fused(ir::FusedKind::kRmsNormMul, sources, ir::NormParams{.eps = 1e-6F})[0]);
  return {std::format("rms_norm_mul.f32.width{}.rows{}", width, rows), "rms_norm_mul",
          ir::Fused(ir::FusedKind::kRmsNormMul), Must(std::move(b).Finish())};
}

Case SpatialChannelNorm(std::uint64_t w, std::uint64_t h, std::uint64_t c, bool silu) {
  ir::Builder b;
  b.SetOwner(0);
  const auto x = b.Input(ir::Dense(Encoding::kF32, ir::ShapeOf({w, h, c, 1})));
  const auto gamma = b.Input(ir::Dense(Encoding::kF32, ir::ShapeOf({c})));
  const std::array sources{x, gamma};
  b.Output(b.Fused(ir::FusedKind::kSpatialChannelNorm, sources,
                   ir::SpatialNormParams{.eps = 1e-12F, .scale = 1.0F, .silu = silu})[0]);
  return {std::format("spatial_channel_norm.f32.w{}.h{}.c{}{}", w, h, c, silu ? ".silu" : ""),
          "spatial_channel_norm", ir::Fused(ir::FusedKind::kSpatialChannelNorm),
          Must(std::move(b).Finish())};
}

struct Arm {
  std::string name;  // what the plan chose for the case's fused node, or "decomposition"
  std::uint32_t version = 0;
  std::uint64_t rows = 0, k = 0;  // the fused node's selection query, when selected
  std::size_t steps = 0;
  std::unique_ptr<kn::StepProgram> program;
  std::unique_ptr<kn::Recorded> recorded;
  std::vector<float> values;
  std::vector<double> samples;
};

class Screen {
 public:
  Screen() {
    execution_ = Must(llmp::providers::cuda::OpenDeviceExecution(0));
    stream_ = Must(execution_->CreateStream());
    native_ = Must(execution_->Submission(stream_));
    backend_ = Must(kn::DeviceBackend::Create(0, *execution_, stream_));
    const auto charge = [this](std::uint64_t bytes) {
      if (bytes > kMetadata - metadata_used_) return false;
      metadata_used_ += bytes;
      return true;
    };
    const auto refund = [this](std::uint64_t bytes) {
      if (bytes > metadata_used_) Fail("metadata accounting underflow");
      metadata_used_ -= bytes;
    };
    Must(backend_->FundConstruction(charge, refund));
    Must(backend_->Declare({}, charge, refund));
    std::vector<std::uint64_t> workspaces{Allocate(backend_->library_workspace_bytes())};
    if (backend_->persistent_bytes() != 0)
      workspaces.push_back(Allocate(backend_->persistent_bytes()));
    Must(backend_->BindWorkspaces(workspaces));
    launch_ = Must(backend_->Launch({Allocate(kPool), Bytes(kPool)}, /*allow_lanes=*/false));
    Check(cudaEventCreate(&start_));
    Check(cudaEventCreate(&stop_));
  }
  ~Screen() {
    Finish();
    for (auto& arm : arms_) {
      arm.recorded.reset();
      arm.program.reset();
    }
    launch_.reset();
    backend_->ReleaseWorkspaces();
    backend_.reset();
    Check(cudaEventDestroy(start_));
    Check(cudaEventDestroy(stop_));
    Must(execution_->DestroyStream(stream_));
    for (void* allocation : allocations_) Check(cudaFree(allocation));
  }

  void Run(const Case& c, int replays) {
    const auto decomposition = Must(ir::Expand(c.graph, false));
    const pl::Policy fused_policy;
    pl::Policy decomposed_policy = fused_policy;
    decomposed_policy.disabled = {c.pattern};
    const auto owned = backend_->Backend();
    if (!owned.owner || !owned.owner->Ready()) Fail("backend has no admitted owner");
    const auto& table = owned.owner->backend();
    Prepare(arms_[0], c, decomposition, fused_policy, table, replays);
    Prepare(arms_[1], c, decomposition, decomposed_policy, table, replays);
    if (arms_[0].name == "decomposition") Fail("the policy did not select a fusion for " + c.label);
    if (arms_[1].name != "decomposition") Fail("`disable " + c.pattern + "` still fused");

    double sum_error = 0, sum_reference = 0, max_error = 0;
    if (arms_[0].values.size() != arms_[1].values.size()) Fail("output sizes differ");
    for (std::size_t i = 0; i < arms_[0].values.size(); ++i) {
      const double a = arms_[0].values[i], b = arms_[1].values[i];
      if (!std::isfinite(a) || !std::isfinite(b)) Fail("nonfinite output");
      sum_error += (a - b) * (a - b);
      sum_reference += b * b;
      max_error = std::max(max_error, std::abs(a - b));
    }
    const double relative_rms = std::sqrt(sum_error / std::max(sum_reference, 1e-30));
    if (relative_rms > 1e-5)
      Fail(std::format("fused and decomposed outputs differ ({})", relative_rms));

    for (int sample = -kWarmups; sample < kSamples; ++sample) {
      for (std::size_t order = 0; order < arms_.size(); ++order) {
        auto& arm = arms_[(order + static_cast<std::size_t>(sample + kWarmups)) % arms_.size()];
        const double ns = Time(arm, replays);
        if (sample >= 0) arm.samples.push_back(ns);
      }
    }
    const auto median = [](std::vector<double> raw) {
      std::ranges::sort(raw);
      return static_cast<std::uint64_t>(std::llround(raw.at(raw.size() / 2)));
    };
    const auto fused = median(arms_[0].samples), decomposed = median(arms_[1].samples);
    std::string_view preference = "unmeasured";
    switch (ex::ClassifyFusionLatency(fused, decomposed)) {
      case ex::FusionPreference::kDecomposition:
        preference = "decomposition";
        break;
      case ex::FusionPreference::kFusion:
        preference = "fusion";
        break;
      case ex::FusionPreference::kNeutral:
        preference = "neutral";
        break;
      default:
        break;
    }
    int driver = 0, runtime = 0;
    Check(cudaDriverGetVersion(&driver));
    Check(cudaRuntimeGetVersion(&runtime));
    std::println(
        "{{\"case\":\"{}\",\"device\":\"{}\",\"device_class\":\"{}\",\"driver\":{},"
        "\"runtime\":{},\"fusion\":\"{}\",\"version\":{},\"query_rows\":{},\"query_k\":{},"
        "\"fused_ns\":{},\"decomposition_ns\":{},\"samples\":{},\"replays\":{},"
        "\"fused_launches\":{},\"decomposition_launches\":{},\"relative_rms\":{},"
        "\"max_absolute\":{},\"preference\":\"{}\"}}",
        c.label, table.device.device_model, table.device.device_class, driver, runtime,
        arms_[0].name, arms_[0].version, arms_[0].rows, arms_[0].k, fused, decomposed, kSamples,
        replays, arms_[0].steps, arms_[1].steps, relative_rms, max_error, preference);
    std::println(
        "    // {}: fused {} ns, decomposed {} ns ({})\n"
        "    {{.device_class = \"{}\", .op = ir::Fused(ir::FusedKind::k...), "
        ".implementation = \"{}\", .version = {}, .encoding = format::Encoding::kF32, "
        ".layout = format::Layout::kDense, .rows_min = {}, .rows_max = {}, .width_min = {}, "
        ".width_max = {}, .fused_ns = {}, .decomposition_ns = {}, .samples = {}, "
        ".provenance = \"docs/experiments/fusion-ranks/README.md\"}},",
        c.label, fused, decomposed, preference, table.device.device_class, arms_[0].name,
        arms_[0].version, arms_[0].rows, arms_[0].rows, arms_[0].k, arms_[0].k, fused, decomposed,
        kSamples);
    std::fflush(stdout);
  }

 private:
  std::uint64_t Allocate(std::uint64_t bytes) {
    void* allocation = nullptr;
    Check(cudaMalloc(&allocation, std::max<std::uint64_t>(bytes, 256)));
    allocations_.push_back(allocation);
    return reinterpret_cast<std::uintptr_t>(allocation);
  }
  cudaStream_t CudaStream() const { return static_cast<cudaStream_t>(native_.handle); }
  void Finish() {
    Check(cudaStreamSynchronize(CudaStream()));
    const auto fence = Must(execution_->Record(stream_));
    Check(cudaStreamSynchronize(CudaStream()));
    if (Must(execution_->Query(fence)) != llmp::providers::FenceState::kComplete)
      Fail("completion unproven");
    Must(execution_->Release(fence));
  }
  void Prepare(Arm& arm, const Case& c, const ir::Graph& graph, const pl::Policy& policy,
               const kn::StepBackend& table, int replays) {
    const en::GraphBuildIO io{graph, graph.inputs(), graph.outputs(), {}};
    kn::Addresses measure;
    measure.activation_alignment = 256;
    // Size the arena as the engine does: a measuring plan and lowering first.
    const auto sized = Must(en::PlanComponentGraph(io, table, policy, measure,
                                                   {.capturable = true, .within = table.within}));
    if (!table.lower) Fail("the backend has no lowerer");
    const auto measured = Must(table.lower({sized, io, measure}));
    kn::Addresses addresses;
    addresses.activation_alignment = 256;
    addresses.activation_bytes =
        std::max<std::uint64_t>({sized.placement.extent, measured->scratch_bytes(), 256});
    addresses.activations = Allocate(addresses.activation_bytes);
    const auto plan = Must(en::PlanComponentGraph(io, table, policy, addresses,
                                                  {.capturable = true, .within = table.within}));
    if (plan.placement.extent > addresses.activation_bytes) Fail("activations exceed the arena");
    arm.name = "decomposition";
    arm.steps = plan.steps.size();
    const pl::SelectionView view(*table.registry, table.device, policy, false, true);
    for (const auto& step : plan.steps) {
      for (const auto node : step.nodes) {
        const auto& selected = plan.graph.node(node);
        if (selected.key != c.op) continue;
        const auto& implementation = table.registry->at(step.implementation);
        arm.name = implementation.name;
        arm.version = implementation.version;
        view.Inspect(plan.graph, selected, [&](const ex::CapabilityQuery& q) {
          arm.rows = q.rows;
          arm.k = q.k;
        });
      }
    }
    arm.program = Must(table.lower({plan, io, addresses}));
    if (arm.program->scratch_bytes() > addresses.activation_bytes ||
        arm.program->pool_bytes() > kPool)
      Fail("lowered storage exceeds the fixture's");
    Must(arm.program->Bind(*launch_));
    for (std::size_t input = 0; input < arm.program->inputs().size(); ++input) {
      const auto& slot = arm.program->inputs()[input];
      if (slot.bytes % sizeof(float) != 0) Fail("a non-F32 fixture input");
      std::vector<float> values(slot.bytes / sizeof(float));
      for (std::size_t i = 0; i < values.size(); ++i) {
        const auto v = static_cast<float>(static_cast<int>((i * 37 + input * 13) % 251) - 125);
        values[i] = input == 0 ? v / 67.0F : 1.0F + v / 512.0F;
      }
      Check(cudaMemcpyAsync(reinterpret_cast<void*>(slot.device_address), values.data(), slot.bytes,
                            cudaMemcpyHostToDevice, CudaStream()));
      Finish();
    }
    Must(arm.program->Run(*launch_, native_));
    Finish();
    if (arm.program->outputs().size() != 1) Fail("a case has one output");
    const auto& output = arm.program->outputs()[0];
    arm.values.resize(output.bytes / sizeof(float));
    Check(cudaMemcpy(arm.values.data(), reinterpret_cast<void*>(output.device_address),
                     output.bytes, cudaMemcpyDeviceToHost));
    arm.recorded =
        Must(arm.program->Record(*launch_, native_, [&]() -> std::expected<void, kn::StepFailure> {
          for (int i = 0; i < replays; ++i) {
            auto ran = arm.program->Run(*launch_, native_);
            if (!ran) return ran;
          }
          return {};
        }));
    Finish();
  }
  double Time(const Arm& arm, int replays) {
    Check(cudaEventRecord(start_, CudaStream()));
    Must(launch_->Launch(native_, *arm.recorded));
    Check(cudaEventRecord(stop_, CudaStream()));
    Check(cudaEventSynchronize(stop_));
    float ms = 0;
    Check(cudaEventElapsedTime(&ms, start_, stop_));
    if (!std::isfinite(ms) || ms <= 0) Fail("invalid elapsed time");
    Finish();
    return static_cast<double>(ms) * 1e6 / replays;
  }
  std::uint64_t metadata_used_ = 0;
  std::vector<void*> allocations_;
  std::unique_ptr<llmp::providers::DeviceExecution> execution_;
  llmp::providers::StreamId stream_{};
  llmp::providers::NativeStream native_{};
  std::unique_ptr<kn::DeviceBackend> backend_;
  std::unique_ptr<kn::StepLaunch> launch_;
  std::array<Arm, 2> arms_;
  cudaEvent_t start_ = nullptr, stop_ = nullptr;
};

bool Parse(std::string_view text, std::uint64_t& value) {
  const auto [end, ec] = std::from_chars(text.data(), text.data() + text.size(), value);
  return ec == std::errc{} && end == text.data() + text.size() && value > 0;
}
}  // namespace

int main(int argc, char** argv) {
  const std::vector<std::string_view> args(argv + 1, argv + argc);
  std::array<std::uint64_t, 3> n{};
  std::optional<Case> c;
  // Cap one input at 2 GiB of F32 so a typo cannot exhaust the device.
  constexpr std::uint64_t kMaxElements = (2ULL << 30U) / sizeof(float);
  const auto fits = [&](std::size_t count) {
    std::uint64_t total = 1;
    for (std::size_t i = 0; i < count; ++i)
      if (__builtin_mul_overflow(total, n[i], &total)) return false;
    return total <= kMaxElements;
  };
  if (args.size() == 3 && args[0] == "rms_norm_mul" && Parse(args[1], n[0]) &&
      Parse(args[2], n[1]) && fits(2)) {
    c = RmsNormMul(n[0], n[1]);
  } else if ((args.size() == 4 || (args.size() == 5 && args[4] == "silu")) &&
             args[0] == "spatial_channel_norm" && Parse(args[1], n[0]) && Parse(args[2], n[1]) &&
             Parse(args[3], n[2]) && fits(3)) {
    c = SpatialChannelNorm(n[0], n[1], n[2], args.size() == 5);
  }
  if (!c) {
    std::println(stderr,
                 "usage: llmp_fusion_ranks rms_norm_mul WIDTH ROWS\n"
                 "       llmp_fusion_ranks spatial_channel_norm W H C [silu]");
    return 2;
  }
  // Enough replays to time at least ~2 ms a graph at small shapes, few at large.
  std::uint64_t elements = 1;
  for (const auto v : n)
    if (v != 0) elements *= v;
  const int replays = elements >= (1ULL << 24U) ? 4 : elements >= (1ULL << 20U) ? 16 : 128;
  Screen screen;
  screen.Run(*c, replays);
  return 0;
}
