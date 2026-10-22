// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Own-decomposition paired screen. Protocol and prospective bounds:
// docs/experiments/fusion-ranks/README.md. No checkpoints or global rank pins.
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
#include <print>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "base/bytes.h"
#include "base/json.h"
#include "engine/graph_plan.h"
#include "ir/builder.h"
#include "ir/decompose.h"
#include "kernels/common/device_backend.h"
#include "planner/pattern.h"
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
constexpr std::uint64_t kPool = 16 * kMiB;
constexpr std::uint64_t kActivations = 32 * kMiB;
constexpr std::uint64_t kMetadata = 64 * kMiB;
constexpr int kSamples = 9, kWarmups = 4, kReplays = 128;

std::string Quote(std::string_view text);
std::string_view phase = "setup";
void Receipt(std::string_view event, std::string_view fields) {
  std::println("{{\"event\":{},\"phase\":{},{} }}", Quote(event), Quote(phase), fields);
  std::fflush(stdout);
}
[[noreturn]] void Fail(std::string_view message) {
  Receipt("refused", std::format("\"reason\":{}", Quote(message)));
  std::println(stderr, "fusion screen failed: {}", message);
  std::abort();  // supervised process exit retains every owner until context teardown
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
std::string Quote(std::string_view text) {
  std::string out;
  llmp::base::json::AppendQuoted(text, out);
  return out;
}
std::string Identity(const ex::MeasuredImplementation& value) {
  return std::format("{{\"name\":{},\"identity\":{}}}", Quote(value.name),
                     Quote(llmp::base::ToHex(value.identity)));
}

ir::Graph Norm(std::uint64_t width, std::uint64_t rows) {
  ir::Builder b;
  b.SetOwner(0);  // An owned graph port, as required by spatial component lowering.
  const auto x = b.Input(ir::Dense(Encoding::kF32, ir::ShapeOf({width, rows})));
  const auto w = b.Input(ir::Dense(Encoding::kF32, ir::ShapeOf({width})));
  const std::array sources{x, w};
  b.Output(b.Fused(ir::FusedKind::kRmsNormMul, sources, ir::NormParams{.eps = 1e-6F})[0]);
  return Must(std::move(b).Finish());
}
struct Arm {
  std::unique_ptr<kn::StepProgram> program;
  std::unique_ptr<kn::Recorded> recorded;
  std::vector<ex::MeasuredImplementation> implementations;
  std::vector<ex::MeasuredImplementation> physical_recipes;
  std::vector<float> values;
  std::vector<double> samples;
  std::optional<llmp::base::Sha256Digest> final_query;
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
    // Use the serving backend's declaration and workspace ownership sequence:
    // its library sizing is a device fact, and binding seals the declaration.
    Must(backend_->Declare({}, charge, refund));
    std::vector<std::uint64_t> workspaces{Allocate(backend_->library_workspace_bytes())};
    if (backend_->persistent_bytes() != 0)
      workspaces.push_back(Allocate(backend_->persistent_bytes()));
    Must(backend_->BindWorkspaces(workspaces));
    launch_ = Must(backend_->Launch({Allocate(kPool), Bytes(kPool)}, /*allow_lanes=*/false));
    Receipt("backend_ready",
            std::format("\"library_workspace_bytes\":{},\"persistent_bytes\":{},\"pool_bytes\":{}",
                        backend_->library_workspace_bytes(), backend_->persistent_bytes(), kPool));
  }
  ~Screen() {
    Finish();
    arms_ = {};
    launch_.reset();
    backend_->ReleaseWorkspaces();
    backend_.reset();
    if (start_) Check(cudaEventDestroy(start_));
    if (stop_) Check(cudaEventDestroy(stop_));
    Must(execution_->DestroyStream(stream_));
    for (void* allocation : allocations_) Check(cudaFree(allocation));
  }
  void Run(std::uint64_t width, std::uint64_t rows, bool validation_only) {
    auto graph = Norm(width, rows);
    auto decomposition = Must(ir::Expand(graph, false));
    const pl::Policy fused_policy;
    auto primitive_policy = fused_policy;
    // This arm pays exactly its own IR decomposition, with no rewrite back
    // into the fusion being measured. RMSNormMul has only primitive children.
    primitive_policy.disabled = {"rms_norm_mul"};
    const auto owned = backend_->Backend();
    if (!owned.owner || !owned.owner->Ready()) Fail("backend has no admitted owner");
    const auto& table = owned.owner->backend();
    // GGML's fused recipe retains the production rewrite's primitive nodes.
    // Start both arms from this typed fusion's own decomposition; the ordinary
    // matcher supplies that provenance only for the fused-policy arm.
    phase = "fused_prepare";
    Prepare(arms_[0], decomposition, fused_policy, table);
    phase = "decomposition_prepare";
    Prepare(arms_[1], decomposition, primitive_policy, table);
    phase = "chain_proof";
    if (arms_[0].implementations.size() != 1 ||
        arms_[0].implementations[0].name != "ggml.rms_norm_mul.fused" ||
        arms_[1].implementations.size() != 2 ||
        arms_[1].implementations[0].name != "ggml.rms_norm" ||
        arms_[1].implementations[1].name != "ggml.mul")
      Fail("screen did not select the requested fusion and its own primitive chain");
    if (arms_[0].physical_recipes.size() != 1 ||
        arms_[0].physical_recipes[0].name != "ggml.rms_norm_mul.fused" ||
        arms_[1].physical_recipes.size() != 1 ||
        arms_[1].physical_recipes[0].name != "ggml.rms_norm_mul.unfused")
      Fail("screen did not bind the audited one-launch and ordered two-launch recipes");
    // Capture the same query the production planner uses, with real aligned
    // activation backing; no activation addresses enter the key.
    pl::SelectionView view(*table.registry, table.device, fused_policy, false, true, {}, {},
                           {.base = 256, .offset = 256, .within = table.within});
    std::optional<llmp::base::Sha256Digest> query;
    bool proved = false;
    view.Inspect(graph, graph.nodes().back(), [&](const ex::CapabilityQuery& q) {
      query = ex::FusionQueryKey(q);
      proved = q.terminal_decomposition.size() == arms_[1].implementations.size();
      for (std::size_t i = 0; proved && i < q.terminal_decomposition.size(); ++i) {
        const auto index = q.terminal_decomposition[i];
        proved = table.registry->at(index).name == arms_[1].implementations[i].name &&
                 table.registry->identity(index) == arms_[1].implementations[i].identity;
      }
    });
    if (!query || !proved) Fail("unqualified or different own-decomposition launch chain");
    if (!arms_[0].final_query || arms_[0].final_query != query)
      Fail("typed fixture differs from the actual final fused selection query");
    int driver = 0, runtime = 0;
    Check(cudaDriverGetVersion(&driver));
    Check(cudaRuntimeGetVersion(&runtime));
    Receipt(
        "identity",
        std::format(
            "\"query\":{},\"selection_environment\":{},\"device\":{},\"driver\":{},\"runtime\":{}",
            Quote(llmp::base::ToHex(*query)),
            Quote(llmp::base::ToHex(table.registry->SelectionDigest())),
            Quote(table.device.device_model), driver, runtime));
    phase = "correctness";

    double sum_error = 0, sum_reference = 0, max_error = 0;
    if (arms_[0].values.size() != arms_[1].values.size()) Fail("output size mismatch");
    for (std::size_t i = 0; i < arms_[0].values.size(); ++i) {
      const auto a = arms_[0].values[i], b = arms_[1].values[i];
      if (!std::isfinite(a) || !std::isfinite(b)) Fail("nonfinite output");
      const double error = static_cast<double>(a) - b;
      sum_error += error * error;
      sum_reference += static_cast<double>(b) * b;
      max_error = std::max(max_error, std::abs(error));
    }
    const double relative_rms = std::sqrt(sum_error / std::max(sum_reference, 1e-30));
    if (relative_rms > 1e-6 || max_error > 2e-5) Fail("own-decomposition correctness bound");
    Receipt("correctness",
            std::format("\"relative_rms\":{},\"max_absolute\":{}", relative_rms, max_error));
    if (validation_only) {
      phase = "validated";
      Receipt("validated",
              std::format("\"fusion_recipe\":{},\"decomposition_recipe\":{},"
                          "\"fusion_nodes\":{},\"decomposition_nodes\":{},"
                          "\"fusion_activation_bytes\":{},\"decomposition_activation_bytes\":{},"
                          "\"replayed_outputs_equal\":true,\"timing_performed\":false",
                          Identity(arms_[0].physical_recipes[0]),
                          Identity(arms_[1].physical_recipes[0]), arms_[0].recorded->nodes(),
                          arms_[1].recorded->nodes(), arms_[0].program->scratch_bytes(),
                          arms_[1].program->scratch_bytes()));
      return;
    }
    Check(cudaEventCreate(&start_));
    Check(cudaEventCreate(&stop_));
    phase = "warmup";
    for (int sample = 0; sample < kWarmups; ++sample)
      for (std::size_t arm = 0; arm < arms_.size(); ++arm)
        Receipt("sample", std::format("\"pair\":{},\"order\":{},\"arm\":{},\"ns\":{}", sample, arm,
                                      arm, Time(arms_[arm])));
    phase = "bookend_before";
    const double first = Time(arms_[1]);
    Receipt("bookend", std::format("\"arm\":1,\"ns\":{}", first));
    std::string pairs;
    for (int sample = -kWarmups; sample < kSamples; ++sample) {
      phase = sample < 0 ? "paired_warmup" : "paired_timing";
      std::array<double, 2> times{};
      const auto first_arm = static_cast<std::size_t>((sample + kWarmups) % 2);
      for (std::size_t order = 0; order < 2; ++order) {
        const auto index = (first_arm + order) % 2;
        const auto ns = Time(arms_[index]);
        times[index] = ns;
        // Flush each completed arm immediately, including a partial pair if
        // the next arm or a later bookend refuses the run.
        Receipt("sample", std::format("\"pair\":{},\"order\":{},\"arm\":{},\"ns\":{}", sample,
                                      order, index, ns));
        if (sample >= 0) arms_[index].samples.push_back(ns);
      }
      if (sample >= 0) {
        if (!pairs.empty()) pairs += ',';
        pairs +=
            std::format("{{\"pair\":{},\"first_arm\":{},\"fused_ns\":{},\"decomposition_ns\":{}}}",
                        sample, first_arm, times[0], times[1]);
      }
    }
    phase = "bookend_after";
    const double last = Time(arms_[1]);
    Receipt("bookend", std::format("\"arm\":1,\"ns\":{}", last));
    if (std::max(first, last) > 1.1 * std::min(first, last)) Fail("bookend drift exceeds 10%");
    const auto median = [](const std::vector<double>& raw) {
      auto ordered = raw;
      std::ranges::sort(ordered);
      return static_cast<std::uint64_t>(std::llround(ordered.at(kSamples / 2)));
    };
    const auto fused = median(arms_[0].samples), primitive = median(arms_[1].samples);
    phase = "accepted";
    std::string parts;
    for (const auto& implementation : arms_[1].implementations) {
      if (!parts.empty()) parts += ',';
      parts += Identity(implementation);
    }
    const auto label = std::format("rms_norm_mul.f32.width{}.rows{}", width, rows);
    std::string_view preference;
    switch (ex::ClassifyFusionLatency(fused, primitive)) {
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
        Fail("timings do not qualify for latency classification");
    }
    // The measurement record can be copied verbatim into data.json only after
    // its provenance placeholder is replaced by the reviewed report receipt.
    std::println(
        "{{\"measurement\":{{\"shape_class\":{},\"query\":{},\"selection_environment\":{},"
        "\"fusion\":{},"
        "\"decomposition\":[{}],\"fused_ns\":{},\"decomposition_ns\":{},"
        "\"samples\":{},\"provenance\":\"UNREVIEWED\"}},"
        "\"device\":{},\"class\":{},\"arch\":{},\"multiprocessors\":{},"
        "\"driver\":{},\"runtime\":{},\"relative_rms\":{},\"max_absolute\":{},"
        "\"bookend_before_ns\":{},\"bookend_after_ns\":{},\"decision\":{},"
        "\"pairs\":[{}],\"fusion_recipe\":{},\"decomposition_recipe\":{},"
        "\"fusion_nodes\":{},\"decomposition_nodes\":{},"
        "\"fusion_activation_bytes\":{},\"decomposition_activation_bytes\":{}}}",
        Quote(label), Quote(llmp::base::ToHex(*query)),
        Quote(llmp::base::ToHex(table.registry->SelectionDigest())),
        Identity(arms_[0].implementations[0]), parts, fused, primitive, kSamples,
        Quote(table.device.device_model), Quote(table.device.device_class), table.device.arch,
        table.device.multiprocessors, driver, runtime, relative_rms, max_error, first, last,
        Quote(preference), pairs, Identity(arms_[0].physical_recipes[0]),
        Identity(arms_[1].physical_recipes[0]), arms_[0].recorded->nodes(),
        arms_[1].recorded->nodes(), arms_[0].program->scratch_bytes(),
        arms_[1].program->scratch_bytes());
    std::fflush(stdout);
  }

 private:
  std::uint64_t Allocate(std::uint64_t bytes) {
    void* allocation = nullptr;
    Check(cudaMalloc(&allocation, bytes));
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
  void Prepare(Arm& arm, const ir::Graph& graph, const pl::Policy& policy,
               const kn::StepBackend& table) {
    const en::GraphBuildIO io{graph, graph.inputs(), graph.outputs(), {}};
    const kn::Addresses addresses{.weight = {},
                                  .weight_rows = {},
                                  .state = {},
                                  .save = {},
                                  .activations = Allocate(kActivations),
                                  .activation_bytes = kActivations,
                                  .activation_alignment = 256};
    auto plan = Must(en::PlanComponentGraph(io, table, policy, addresses,
                                            {.capturable = true, .within = table.within}));
    Receipt("planned_storage", std::format("\"activation_bytes\":{},\"activation_capacity\":{}",
                                           plan.placement.extent, kActivations));
    if (plan.placement.extent > kActivations) Fail("planned activations exceed fixture storage");
    pl::SelectionView final_view(
        *table.registry, table.device, policy, false, true, {}, {},
        {.base = addresses.activation_alignment, .offset = 256, .within = table.within});
    for (const auto& step : plan.steps) {
      if (step.nodes.size() != 1 || !step.group.empty() || !step.batch.empty())
        Fail("screen selected a grouped or batched logical step");
      arm.implementations.push_back({table.registry->at(step.implementation).name,
                                     table.registry->identity(step.implementation)});
      const auto& selected = plan.graph.node(step.nodes[0]);
      if (selected.key.op == ir::Op::kFused) {
        if (arm.final_query) Fail("screen selected more than one fused query");
        final_view.Inspect(plan.graph, selected, [&](const ex::CapabilityQuery& q) {
          arm.final_query = ex::FusionQueryKey(q);
        });
      }
    }
    if (!table.lower) Fail("backend omitted its canonical lowerer");
    arm.program = Must(table.lower({plan, io, addresses}));
    Receipt("lowered_storage", std::format("\"activation_bytes\":{},\"activation_capacity\":{},"
                                           "\"pool_bytes\":{},\"pool_capacity\":{}",
                                           arm.program->scratch_bytes(), kActivations,
                                           arm.program->pool_bytes(), kPool));
    if (arm.program->scratch_bytes() > kActivations || arm.program->pool_bytes() > kPool)
      Fail("lowered storage exceeds fixture capacity");
    Must(arm.program->Bind(*launch_));
    const auto* bound = arm.program->bound();
    if (!bound) Fail("program omitted bound implementation inventory");
    for (std::size_t i = 0; i < bound->size(); ++i) {
      const auto index = table.registry->Find(bound->at(i).name);
      if (!index) Fail("bound implementation absent from registry");
      arm.physical_recipes.push_back({bound->at(i).name, table.registry->identity(*index)});
    }
    for (std::size_t input = 0; input < arm.program->inputs().size(); ++input) {
      const auto& slot = arm.program->inputs()[input];
      if (slot.bytes % sizeof(float)) Fail("non-F32 fixture input");
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
    if (arm.program->outputs().size() != 1) Fail("unexpected output count");
    const auto& output = arm.program->outputs()[0];
    arm.values.resize(output.bytes / sizeof(float));
    Check(cudaMemcpy(arm.values.data(), reinterpret_cast<void*>(output.device_address),
                     output.bytes, cudaMemcpyDeviceToHost));
    arm.recorded =
        Must(arm.program->Record(*launch_, native_, [&]() -> std::expected<void, kn::StepFailure> {
          for (int i = 0; i < kReplays; ++i) {
            auto ran = arm.program->Run(*launch_, native_);
            if (!ran) return ran;
          }
          return {};
        }));
    Finish();
    if (arm.recorded->nodes() != kReplays * arm.implementations.size())
      Fail("captured graph does not contain the complete repeated chain");
    Check(cudaMemsetAsync(reinterpret_cast<void*>(output.device_address), 0, output.bytes,
                          CudaStream()));
    Must(launch_->Launch(native_, *arm.recorded));
    Finish();
    std::vector<float> replayed(arm.values.size());
    Check(cudaMemcpy(replayed.data(), reinterpret_cast<void*>(output.device_address), output.bytes,
                     cudaMemcpyDeviceToHost));
    if (replayed != arm.values) Fail("captured replay differs from checked eager output");
  }
  double Time(const Arm& arm) {
    Check(cudaEventRecord(start_, CudaStream()));
    Must(launch_->Launch(native_, *arm.recorded));
    Check(cudaEventRecord(stop_, CudaStream()));
    Check(cudaEventSynchronize(stop_));
    float ms = 0;
    Check(cudaEventElapsedTime(&ms, start_, stop_));
    if (!std::isfinite(ms) || ms <= 0) Fail("invalid elapsed time");
    Finish();
    return static_cast<double>(ms) * 1e6 / kReplays;
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
}  // namespace

int main(int argc, char** argv) {
  // First screen by default; expanded classes are individually supervised.
  std::uint64_t width = 1024, rows = 1;
  const auto parse = [](const char* text, std::uint64_t& value) {
    const std::string_view s(text);
    const auto [end, ec] = std::from_chars(s.data(), s.data() + s.size(), value);
    return ec == std::errc{} && end == s.data() + s.size() && value > 0;
  };
  const bool validation_only = argc > 1 && std::string_view(argv[1]) == "--validate-only";
  const int first = validation_only ? 2 : 1;
  if (argc > first + 2 || (argc > first && !parse(argv[first], width)) ||
      (argc > first + 1 && !parse(argv[first + 1], rows)) ||
      // Bound each F32 input before graph construction. Division and short-circuit
      // evaluation keep the extent product overflow-safe; the actual planner
      // and lowerer below account for simultaneous live tensors and scratch.
      width > kActivations / sizeof(float) || rows > kActivations / (sizeof(float) * width)) {
    Receipt("refused", "\"reason\":\"invalid arguments or fixture storage bounds\"");
    std::println(
        stderr,
        "usage: llmp_fusion_ranks [--validate-only] [width [rows]] (must fit fixture storage)");
    return 2;
  }
  Receipt(
      "begin",
      std::format("\"width\":{},\"rows\":{},\"samples\":{},\"replays\":{},\"validation_only\":{}",
                  width, rows, kSamples, kReplays, validation_only));
  {
    Screen screen;
    screen.Run(width, rows, validation_only);
  }
  phase = "retired";
  Receipt("complete", std::format("\"validation_only\":{}", validation_only));
}
