// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The decode module's templates (kernels/decode/products.h, D-108) against
// the tuned kernels they would someday replace, one representative
// Qwen3.8 shape each, on a GB10 (the tuned kernels are sm_121a-only):
//
//   vector      MXFP8 K 2560 x N 10240 (blk.N.attn_qkv), rows 1 and 16,
//               against llmp.mxfp8.mul_mat_vec;
//   vector_glu  MXFP8 K 2560 x N 640 gate and up (the shared expert), rows
//               1 and 16, against two llmp.mxfp8.mul_mat_vec and SwiGLU;
//   matrix      the qkv shape, rows 512, against llmp.mxfp8.quantize,
//               .swizzle and .gemm (CUTLASS), and the swizzle alone;
//               matrix_fma beside it;
//   grouped     NVFP4 experts in the CUTLASS layout (512 experts, gate and
//               up K 2560 x N 1280), 10 of 512 experts a token: rows 1 and
//               16 against llmp.moe.gemv, rows 512 against llmp.moe.route,
//               .quantize and .gemm (CUTLASS), and the route alone;
//   rows        the n-gram table's row-interleaved NVFP4 rows (160 values),
//               16 ids a token at 1, 16 and 512 tokens, against
//               llmp.nvfp4.get_rows.
// The worklist the grouped template reads is built on the host here; the
// tuned path's routing is timed alone beside it. Weights are constant
// bytes (speed does not depend on their values), sized as the model's;
// decode-shaped products rotate over copies of their weights (4 of qkv,
// 16 of the shared expert) or over 8 routings and id sets, so successive
// launches do not find their weights in L2.
//
// Each candidate is one CUDA graph of --reps launches, replayed --batches
// times in alternation with its counterpart on one stream; the median
// replay's time per launch is reported (JSON lines), with the batches'
// range. Informational: nothing here selects an implementation.
//
//   llmp_decode_templates [--case all|mxfp8|glu|grouped|rows|dense] [--rows 1|16|512]
//                         [--reps 64] [--batches 7]

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <expected>
#include <format>
#include <functional>
#include <memory>
#include <print>
#include <random>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include "base/bytes.h"
#include "components/graph_io.h"
#include "execution/registry.h"
#include "format/encoding.h"
#include "format/layout.h"
#include "format/tensor_format.h"
#include "ggml.h"
#include "ir/builder.h"
#include "ir/op.h"
#include "kernels/decode/products.h"
#include "kernels/ggml/executor.h"
#include "kernels/ggml/fusion.h"
#include "kernels/ggml/implementations.h"
#include "kernels/ggml/launch.h"
#include "kernels/ggml/llmp_ops.h"
#include "kernels/ggml/lower.h"
#include "kernels/ggml/moe_layout.h"
#include "kernels/ggml/plan_step.h"
#include "kernels/ggml/tensors.h"
#include "planner/plan.h"
#include "providers/cuda/cuda_device_execution.h"
#include "providers/device_execution.h"

namespace {

namespace kg = llmp::kernels::ggml;
namespace decode = llmp::kernels::decode;
namespace format = llmp::format;
namespace pl = llmp::planner;
namespace co = llmp::components;
namespace kn = llmp::kernels;
namespace ir = llmp::ir;
using format::Encoding;
using format::Layout;
using format::PartRole;

using Status = std::expected<void, std::string>;

[[noreturn]] void Die(std::string_view what) {
  (void)std::fflush(stdout);
  std::println(stderr, "llmp_decode_templates: {}", what);
  (void)std::fflush(stderr);
  std::_Exit(1);  // no GPU-resource destructors after a failure
}

void Cuda(cudaError_t status, std::string_view what) {
  if (status != cudaSuccess) {
    Die(std::format("{}: {}", what, cudaGetErrorString(status)));
  }
}

template <typename T>
T Ok(std::expected<T, std::string> value, std::string_view what) {
  if (!value) {
    Die(std::format("{}: {}", what, value.error()));
  }
  return std::move(*value);
}
void Ok(const Status& status, std::string_view what) {
  if (!status) {
    Die(std::format("{}: {}", what, status.error()));
  }
}

template <typename T>
T* Pointer(std::uint64_t address) {
  return reinterpret_cast<T*>(address);  // NOLINT(performance-no-int-to-ptr)
}

constexpr std::uint64_t kWorkspace = std::uint64_t{512} << 20U;
constexpr std::int64_t kWidth = 2560;
constexpr std::int64_t kFfn = 640;
constexpr std::int64_t kQkv = 10240;
constexpr std::int64_t kExperts = 512;
constexpr std::int64_t kUsed = 10;
constexpr std::size_t kRoutings = 8;

class Bench {
 public:
  using Replay = std::function<void()>;

  Bench(int reps, int batches, int rows) : reps_(reps), batches_(batches), rows_(rows) {
    auto execution = llmp::providers::cuda::OpenDeviceExecution(0);
    if (!execution) {
      Die("no CUDA device execution");
    }
    execution_ = std::move(*execution);
    stream_ = execution_->CreateStream().value();
    native_ = static_cast<cudaStream_t>(execution_->Submission(stream_).value().handle);
    auto launch = kg::LaunchContext::Create(
        0, *execution_, stream_,
        {.base = Allocate(kWorkspace), .size = llmp::base::Bytes(kWorkspace)});
    if (!launch) {
      Die(launch.error().detail);
    }
    launch_ = std::move(*launch);
    arena_ = std::make_unique<kg::TensorArena>(kg::TensorArena::Create(4096).value());
    auto registry = llmp::execution::Registry::Create(kg::Implementations());
    if (!registry) {
      Die("the GGML module's registry");
    }
    registry_ = std::make_unique<llmp::execution::Registry>(std::move(*registry));
    Cuda(cudaEventCreate(&start_), "event");
    Cuda(cudaEventCreate(&end_), "event");
  }

  bool RowsEnabled(std::int64_t rows) const { return rows_ == 0 || rows_ == rows; }

  std::uint64_t Allocate(std::uint64_t bytes, int fill = 0) {
    void* p = nullptr;
    const auto allocated = std::max<std::uint64_t>(bytes, 256);
    Cuda(cudaMalloc(&p, allocated), "cudaMalloc");
    allocated_bytes_ += allocated;
    Cuda(cudaMemset(p, fill, bytes), "cudaMemset");
    return reinterpret_cast<std::uintptr_t>(p);
  }
  template <typename T>
  T* Upload(const std::vector<T>& host) {
    auto* p = Pointer<T>(Allocate(host.size() * sizeof(T)));
    Cuda(cudaMemcpy(p, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice), "upload");
    return p;
  }

  ggml_context* c() const { return arena_->context(); }
  ggml_tensor* Leaf(ggml_tensor* t, std::uint64_t address) {
    kg::TensorArena::Bind(t, address);
    return t;
  }
  ggml_tensor* Leaf(ggml_tensor* t, const void* address) {
    return Leaf(t, reinterpret_cast<std::uintptr_t>(address));
  }

  // Lowers an IR graph using the production planner and lowering engine.
  kg::LoweredStep Lower(
      const ir::Graph& graph, std::span<const co::StepWeight> weights = {},
      const std::function<std::uint64_t(const ir::WeightRef&, PartRole)>& weight_addr = {},
      const std::vector<const void*>& input_sources = {}, std::uint64_t activation_bytes = 0) {
    pl::DeviceFacts facts{
        .device_class = "sm_121", .arch = 121, .features = llmp::execution::kBf16Mma};
    auto plan = pl::Plan(graph.Clone(), *registry_, facts, {});
    if (!plan) Die(plan.error().detail);
    const std::uint64_t extent = std::max(plan->placement.extent, activation_bytes);
    const std::uint64_t act_base = Allocate(std::max<std::uint64_t>(extent, 256));
    kn::Addresses addresses{
        .weight = weight_addr,
        .weight_rows = {},
        .state = {},
        .save = {},
        .save_rows = 0,
        .activations = act_base,
        .activation_bytes = extent,
        .activation_alignment = 256,
        .state_alignment = {},
        .expert_part = {},
        .weight_alignment = {},
    };
    co::GraphBuildIO io{graph, graph.inputs(), graph.outputs(), weights};
    auto lowered = kg::LowerStep({*plan, io, addresses}, *registry_, kg::DeviceChoicesOf(*launch_));
    if (!lowered) Die(lowered.error().detail);
    for (std::size_t i = 0; i < lowered->inputs.size(); ++i) {
      if (i < input_sources.size() && input_sources[i] != nullptr) {
        Cuda(cudaMemcpyAsync(reinterpret_cast<void*>(lowered->inputs[i].device_address),
                             input_sources[i], lowered->inputs[i].bytes, cudaMemcpyDeviceToDevice,
                             native_),
             "input copy");
      }
    }
    Cuda(cudaStreamSynchronize(native_), "input sync");
    return std::move(*lowered);
  }

  // The lowered steps planned, bound and captured: launch r runs step r mod steps.
  Replay Tuned(const std::vector<kg::LoweredStep>& steps) {
    std::vector<std::shared_ptr<kg::BoundGraph>> bound;
    for (const auto& step : steps) {
      if (bound.empty()) {
        for (const auto& s : step.plan.steps)
          std::println(R"({{"kind":"resolved_route","implementation":"{}"}})", s.implementation);
      }
      auto one = kg::BoundGraph::Bind(*registry_, step.plan);
      if (!one) {
        Die(one.error().detail);
      }
      bound.push_back(std::make_shared<kg::BoundGraph>(std::move(*one)));
    }
    auto graph =
        launch_->Capture([&](kg::LaunchContext& l) -> std::expected<void, kg::KernelFailure> {
          for (int r = 0; r < reps_; ++r) {
            if (auto ran = bound[static_cast<std::size_t>(r) % bound.size()]->Run(l); !ran) {
              return ran;
            }
          }
          return {};
        });
    if (!graph) {
      Die(graph.error().detail);
    }
    auto captured = std::make_shared<kg::CapturedGraph>(std::move(*graph));
    return [this, bound, captured] {
      if (auto ran = launch_->Launch(*captured); !ran) {
        Die(ran.error().detail);
      }
    };
  }

  // The decode module's launches, captured on the same stream: launch r is
  // queue(stream, r).
  Replay Ours(const std::function<Status(void*, std::size_t)>& queue) {
    for (int r = 0; r < reps_; ++r) {  // checks and module loads, outside the capture
      Ok(queue(native_, static_cast<std::size_t>(r)), "a first launch");
    }
    Cuda(cudaStreamSynchronize(native_), "sync");
    (void)execution_->Submission(stream_);
    Cuda(cudaStreamBeginCapture(native_, cudaStreamCaptureModeThreadLocal), "capture");
    for (int r = 0; r < reps_; ++r) {
      Ok(queue(native_, static_cast<std::size_t>(r)), "capture a launch");
    }
    cudaGraph_t graph = nullptr;
    Cuda(cudaStreamEndCapture(native_, &graph), "end capture");
    cudaGraphExec_t exec = nullptr;
    Cuda(cudaGraphInstantiate(&exec, graph, 0), "instantiate");
    Cuda(cudaGraphDestroy(graph), "graph");
    return [this, exec] {
      (void)execution_->Submission(stream_);
      Cuda(cudaGraphLaunch(exec, native_), "graph launch");
    };
  }

  // Microseconds a launch: the median batch, and the batches' range.
  struct Time {
    double median = 0;
    double low = 0;
    double high = 0;
  };
  std::vector<Time> Measure(const std::vector<Replay>& candidates) {
    for (const Replay& r : candidates) {  // warm
      r();
      r();
    }
    Cuda(cudaStreamSynchronize(native_), "warm");
    std::vector<std::vector<double>> samples(candidates.size());
    for (int b = 0; b < batches_; ++b) {
      for (std::size_t i = 0; i < candidates.size(); ++i) {
        const std::size_t c = (b % 2 == 0) ? i : candidates.size() - 1 - i;  // ABBA
        Cuda(cudaEventRecord(start_, native_), "event");
        candidates[c]();
        Cuda(cudaEventRecord(end_, native_), "event");
        Cuda(cudaEventSynchronize(end_), "event wait");
        float ms = 0;
        Cuda(cudaEventElapsedTime(&ms, start_, end_), "event time");
        samples[c].push_back(static_cast<double>(ms) * 1000.0 / reps_);
      }
    }
    std::vector<Time> out;
    for (std::vector<double>& s : samples) {
      std::ranges::sort(s);
      out.push_back({.median = s[s.size() / 2], .low = s.front(), .high = s.back()});
    }
    return out;
  }

  void Report(std::string_view tmpl, std::string_view shape, std::int64_t rows,
              const std::vector<std::string>& names, const std::vector<Replay>& candidates,
              const float* baseline = nullptr, const float* candidate = nullptr,
              std::size_t elements = 0, double tolerance = 0.01) {
    if (baseline && candidate) {
      candidates[0]();
      candidates[1]();
      Cuda(cudaStreamSynchronize(native_), "correctness sync");
      std::vector<float> a(elements), z(elements);
      Cuda(cudaMemcpy(a.data(), baseline, elements * sizeof(float), cudaMemcpyDeviceToHost),
           "baseline result");
      Cuda(cudaMemcpy(z.data(), candidate, elements * sizeof(float), cudaMemcpyDeviceToHost),
           "candidate result");
      double squared = 0, reference = 0;
      for (std::size_t i = 0; i < elements; ++i) {
        if (!std::isfinite(a[i]) || !std::isfinite(z[i])) Die("non-finite comparison output");
        const double delta = static_cast<double>(a[i]) - z[i];
        squared += delta * delta;
        reference += static_cast<double>(a[i]) * a[i];
      }
      const double rms = std::sqrt(squared / std::max(reference, 1e-20));
      std::println(
          R"({{"kind":"correctness","template":"{}","rows":{},"relative_rms":{:.9g},"tolerance":{}}})",
          tmpl, rows, rms, tolerance);
      if (rms > tolerance)
        Die(std::format("{} relative RMS {} exceeds frozen tolerance {}; first output {} vs {}",
                        names[1], rms, tolerance, a.front(), z.front()));
    }
    std::size_t free = 0, total = 0;
    Cuda(cudaMemGetInfo(&free, &total), "memory accounting");
    std::println(
        R"({{"kind":"memory","template":"{}","rows":{},"harness_allocated_bytes":{},"device_used_bytes":{},"device_total_bytes":{}}})",
        tmpl, rows, allocated_bytes_, total - free, total);
    const std::vector<Time> times = Measure(candidates);
    for (std::size_t i = 0; i < names.size(); ++i) {
      std::println(
          R"({{"template": "{}", "shape": "{}", "rows": {}, "candidate": "{}", "us": {:.2f}, "low": {:.2f}, "high": {:.2f}}})",
          tmpl, shape, rows, names[i], times[i].median, times[i].low, times[i].high);
    }
    (void)std::fflush(stdout);
  }

 private:
  std::uint64_t allocated_bytes_ = 0;
  int reps_;
  int batches_;
  int rows_;
  std::unique_ptr<llmp::providers::DeviceExecution> execution_;
  llmp::providers::StreamId stream_;
  cudaStream_t native_ = nullptr;
  std::unique_ptr<kg::LaunchContext> launch_;
  std::unique_ptr<kg::TensorArena> arena_;
  std::unique_ptr<llmp::execution::Registry> registry_;
  cudaEvent_t start_ = nullptr;
  cudaEvent_t end_ = nullptr;
};

std::vector<float> Normal(std::size_t n, std::uint32_t seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> dist(0.0f, 1.0f);
  std::vector<float> out(n);
  for (float& v : out) {
    v = dist(rng);
  }
  return out;
}

format::TensorFormat Format(Encoding e, Layout l, std::int64_t k, std::int64_t n,
                            std::uint8_t stacked = 1) {
  format::TensorFormat f;
  f.encoding = e;
  f.layout = l;
  f.shape =
      format::ShapeOf(std::array{static_cast<std::uint64_t>(k), static_cast<std::uint64_t>(n)})
          .value();
  f.stacked = stacked;
  return f;
}

// An MXFP8 [k, n]: codes 0x38 (1.0), scales 127 (1.0).
struct Mxfp8 {
  std::uint64_t codes = 0;
  std::uint64_t scales = 0;
  std::int64_t k = 0;
  std::int64_t n = 0;
  std::array<decode::DevicePart, 2> parts{};
  decode::Weight weight() const {
    return {.format = Format(Encoding::kMxfp8, Layout::kSplit, k, n), .parts = parts};
  }
  ggml_tensor* Codes(Bench& b) const {
    return b.Leaf(ggml_new_tensor_2d(b.c(), GGML_TYPE_I8, k, n), codes);
  }
  ggml_tensor* Scales(Bench& b) const {
    return b.Leaf(ggml_new_tensor_2d(b.c(), GGML_TYPE_I8, k / 32, n), scales);
  }
};
std::vector<Mxfp8> MakeMxfp8(Bench& b, std::int64_t k, std::int64_t n, std::size_t copies) {
  std::vector<Mxfp8> out;
  for (std::size_t i = 0; i < copies; ++i) {
    Mxfp8 m;
    m.k = k;
    m.n = n;
    m.codes = b.Allocate(static_cast<std::uint64_t>(k * n), 0x38);
    m.scales = b.Allocate(static_cast<std::uint64_t>(k / 32 * n), 127);
    m.parts = {{{PartRole::kCodes, Pointer<const void>(m.codes), static_cast<std::uint64_t>(k * n)},
                {PartRole::kBlockScales, Pointer<const void>(m.scales),
                 static_cast<std::uint64_t>(k / 32 * n)}}};
    out.push_back(m);
  }
  return out;
}

void VectorAndMatrix(Bench& b) {
  const std::vector<Mxfp8> w = MakeMxfp8(b, kWidth, kQkv, 4);
  for (const std::int64_t rows : {std::int64_t{1}, std::int64_t{16}, std::int64_t{512}}) {
    if (!b.RowsEnabled(rows)) continue;
    float* x = b.Upload(Normal(static_cast<std::size_t>(kWidth * rows), 1));
    auto* y = Pointer<float>(b.Allocate(static_cast<std::uint64_t>(kQkv * rows * 4)));
    const auto product = [&](const Mxfp8& m) {
      return decode::Product{.weight = m.weight(),
                             .first = 0,
                             .outputs = kQkv,
                             .x = x,
                             .rows = static_cast<std::uint64_t>(rows),
                             .x_stride = kWidth,
                             .y = y,
                             .y_stride = kQkv};
    };
    std::vector<kg::LoweredStep> tuned;
    for (const Mxfp8& m : w) {
      ir::Builder bld;
      const auto xt = bld.Input(ir::Dense(
          Encoding::kF32,
          ir::ShapeOf({static_cast<std::uint64_t>(kWidth), static_cast<std::uint64_t>(rows)})));
      const auto wt = bld.Weight(m.weight().format, ir::WeightRef{});
      bld.Output(bld.MatMul(wt, xt));
      auto g = std::move(bld).Finish();
      if (!g) Die(g.error().detail);
      const std::array w_spec{co::StepWeight{
          .tensor = wt,
          .ref = ir::WeightRef{},
          .parts = {{"codes", PartRole::kCodes}, {"scales", PartRole::kBlockScales}}}};
      tuned.push_back(b.Lower(*g, w_spec,
                              [&](const ir::WeightRef&, PartRole role) {
                                return role == PartRole::kCodes ? m.codes : m.scales;
                              },
                              {x}));
    }
    if (rows <= 16) {
      const auto incumbent = b.Tuned(tuned);
      b.Report("vector", "mxfp8.split K2560 N10240", rows,
               {"llmp.mxfp8.mul_mat_vec", "llmp.decode.vector.mxfp8.split"},
               {incumbent, b.Ours([&](void* s, std::size_t r) {
                  return decode::RunVector(product(w[r % w.size()]), s);
                })},
               static_cast<const float*>(tuned.back().keep.front()->data), y,
               static_cast<std::size_t>(kQkv * rows));
      continue;
    }
    std::vector<kg::LoweredStep> swizzle;
    for (const Mxfp8& m : w) {
      ir::Builder bld;
      const auto sc = bld.Input(Format(Encoding::kMxfp8, Layout::kSplit, kWidth / 32, kQkv));
      bld.Output(bld.Convert(sc, Encoding::kMxfp8));
      auto g = std::move(bld).Finish();
      if (!g) Die(g.error().detail);
      swizzle.push_back(b.Lower(*g, {}, [&](const ir::WeightRef&, PartRole) { return 0ULL; },
                                {Pointer<const void>(m.scales)}));
    }
    b.Report("matrix", "mxfp8.split K2560 N10240", rows,
             {"llmp.mxfp8.quantize+swizzle+gemm", "llmp.mxfp8.swizzle alone",
              "llmp.decode.matrix.mxfp8.split", "llmp.decode.matrix_fma.mxfp8.split"},
             {b.Tuned(tuned), b.Tuned(swizzle), b.Ours([&](void* s, std::size_t r) {
                return decode::RunMatrix(product(w[r % w.size()]), decode::MatrixPath::kMma, s);
              }),
              b.Ours([&](void* s, std::size_t r) {
                return decode::RunMatrix(product(w[r % w.size()]), decode::MatrixPath::kFma, s);
              })});
  }
}

void Glu(Bench& b) {
  const std::vector<Mxfp8> gate = MakeMxfp8(b, kWidth, kFfn, 16);
  const std::vector<Mxfp8> up = MakeMxfp8(b, kWidth, kFfn, 16);
  for (const std::int64_t rows : {std::int64_t{1}, std::int64_t{16}}) {
    if (!b.RowsEnabled(rows)) continue;
    float* x = b.Upload(Normal(static_cast<std::size_t>(kWidth * rows), 2));
    auto* y = Pointer<float>(b.Allocate(static_cast<std::uint64_t>(kFfn * rows * 4)));
    std::vector<kg::LoweredStep> tuned;
    for (std::size_t i = 0; i < gate.size(); ++i) {
      ir::Builder bld;
      const auto xt = bld.Input(ir::Dense(
          Encoding::kF32,
          ir::ShapeOf({static_cast<std::uint64_t>(kWidth), static_cast<std::uint64_t>(rows)})));
      const auto wg =
          bld.Weight(gate[i].weight().format,
                     ir::WeightRef{.component = static_cast<std::uint16_t>(i), .layer = 0});
      const auto wu =
          bld.Weight(up[i].weight().format,
                     ir::WeightRef{.component = static_cast<std::uint16_t>(i), .layer = 1});
      const auto g = bld.MatMul(wg, xt);
      const auto u = bld.MatMul(wu, xt);
      bld.Output(bld.Glu(g, u, ir::GluParams{.kind = ir::GluKind::kSwiGlu}));
      auto graph = std::move(bld).Finish();
      if (!graph) Die(graph.error().detail);
      const std::array w_spec{
          co::StepWeight{
              .tensor = wg,
              .ref = ir::WeightRef{.component = static_cast<std::uint16_t>(i), .layer = 0},
              .parts = {{"codes", PartRole::kCodes}, {"scales", PartRole::kBlockScales}}},
          co::StepWeight{
              .tensor = wu,
              .ref = ir::WeightRef{.component = static_cast<std::uint16_t>(i), .layer = 1},
              .parts = {{"codes", PartRole::kCodes}, {"scales", PartRole::kBlockScales}}}};
      tuned.push_back(b.Lower(*graph, w_spec,
                              [&](const ir::WeightRef& ref, PartRole role) {
                                const auto& m = ref.layer == 0 ? gate[i] : up[i];
                                return role == PartRole::kCodes ? m.codes : m.scales;
                              },
                              {x}));
    }
    const auto incumbent = b.Tuned(tuned);
    b.Report("vector_glu", "mxfp8.split K2560 N640 gate and up", rows,
             {"2x llmp.mxfp8.mul_mat_vec+swiglu", "llmp.decode.vector_glu.mxfp8.split"},
             {incumbent, b.Ours([&](void* s, std::size_t r) {
                const decode::GluProduct p{.gate = gate[r % gate.size()].weight(),
                                           .up = up[r % up.size()].weight(),
                                           .outputs = kFfn,
                                           .glu = {.kind = llmp::ir::GluKind::kSwiGlu},
                                           .x = x,
                                           .rows = static_cast<std::uint64_t>(rows),
                                           .x_stride = kWidth,
                                           .y = y,
                                           .y_stride = kFfn};
                return decode::RunVectorGlu(p, s);
              })},
             static_cast<const float*>(tuned.back().keep.front()->data), y,
             static_cast<std::size_t>(kFfn * rows));
  }
}

// One routing of `tokens` tokens to kUsed of kExperts each, as the tuned
// path's ids and the decode module's worklist sorted by expert.
struct Routing {
  std::int32_t* ids = nullptr;
  std::int32_t* offsets = nullptr;
  std::int32_t* pair_rows = nullptr;
  std::int32_t* pair_out = nullptr;
  std::uint64_t pairs = 0;
};
Routing Route(Bench& b, std::int64_t tokens, std::uint32_t seed) {
  std::mt19937 rng(seed);
  std::vector<std::int32_t> ids;
  for (std::int64_t t = 0; t < tokens; ++t) {
    std::vector<std::int32_t> all(kExperts);
    for (std::int32_t i = 0; i < kExperts; ++i) {
      all[static_cast<std::size_t>(i)] = i;
    }
    std::shuffle(all.begin(), all.end(), rng);
    ids.insert(ids.end(), all.begin(), all.begin() + kUsed);
  }
  std::vector<std::int32_t> offsets;
  std::vector<std::int32_t> pair_rows;
  std::vector<std::int32_t> pair_out;
  for (std::int32_t e = 0; e < kExperts; ++e) {
    offsets.push_back(static_cast<std::int32_t>(pair_rows.size()));
    for (std::size_t s = 0; s < ids.size(); ++s) {
      if (ids[s] == e) {
        pair_rows.push_back(static_cast<std::int32_t>(s / kUsed));
        pair_out.push_back(static_cast<std::int32_t>(s));
      }
    }
  }
  offsets.push_back(static_cast<std::int32_t>(pair_rows.size()));
  return {.ids = b.Upload(ids),
          .offsets = b.Upload(offsets),
          .pair_rows = b.Upload(pair_rows),
          .pair_out = b.Upload(pair_out),
          .pairs = pair_rows.size()};
}

void Grouped(Bench& b) {
  namespace moe = kg::moe;
  const moe::ExpertLayout l{.ffn = kFfn, .width = kWidth};
  const std::uint64_t stride = l.bytes();
  const std::uint64_t slab = b.Allocate(stride * kExperts, 0x22);
  // The decode module's operands: each expert's gate and up codes and atoms.
  const format::TensorFormat f = Format(Encoding::kNvfp4, Layout::kScaleAtoms, kWidth, 2 * kFfn, 2);
  std::vector<decode::Operand> operands;
  for (std::int64_t e = 0; e < kExperts; ++e) {
    const std::uint64_t base = slab + (static_cast<std::uint64_t>(e) * stride);
    const std::array<decode::DevicePart, 2> parts = {
        {{PartRole::kCodes, Pointer<const void>(base), l.gate_up_scales()},
         {PartRole::kBlockScales, Pointer<const void>(base + l.gate_up_scales()),
          l.down_codes() - l.gate_up_scales()}}};
    operands.push_back(Ok(decode::OperandOf({.format = f, .parts = parts}), "expert operand"));
  }
  const decode::Operand* table = b.Upload(operands);
  const std::string shape = "nvfp4.scale_atoms K2560 N1280, 10 of 512 experts";
  for (const std::int64_t tokens : {std::int64_t{1}, std::int64_t{16}, std::int64_t{512}}) {
    if (!b.RowsEnabled(tokens)) continue;
    std::vector<Routing> routings;
    for (std::size_t i = 0; i < (tokens <= 16 ? kRoutings : 1); ++i) {
      routings.push_back(Route(
          b, tokens, static_cast<std::uint32_t>(tokens * 100) + static_cast<std::uint32_t>(i)));
    }
    float* x = b.Upload(Normal(static_cast<std::size_t>(kWidth * tokens), 3));
    auto* y = Pointer<float>(b.Allocate(static_cast<std::uint64_t>(2 * kFfn * kUsed * tokens * 4)));
    const auto ours = [&](void* s, std::size_t r) {
      const Routing& route = routings[r % routings.size()];
      const decode::GroupedProduct p{.format = f,
                                     .experts = table,
                                     .experts_count = kExperts,
                                     .first = 0,
                                     .outputs = 2 * kFfn,
                                     .offsets = route.offsets,
                                     .pair_rows = route.pair_rows,
                                     .pair_out = route.pair_out,
                                     .pairs = route.pairs,
                                     .x = x,
                                     .x_rows = static_cast<std::uint64_t>(tokens),
                                     .x_stride = kWidth,
                                     .y = y,
                                     .y_rows = static_cast<std::uint64_t>(kUsed * tokens),
                                     .y_stride = 2 * kFfn,
                                     .rows_hint = route.pairs / kExperts};
      return decode::RunGrouped(p, s);
    };
    std::vector<kg::LoweredStep> tuned;
    std::vector<kg::LoweredStep> route_only;
    for (const Routing& route : routings) {
      ir::Builder bld;
      const auto wt = bld.Weight(f, ir::WeightRef{});
      const auto idt = bld.Input(ir::Dense(
          Encoding::kI32,
          ir::ShapeOf({static_cast<std::uint64_t>(kUsed), static_cast<std::uint64_t>(tokens)})));
      if (tokens <= 16) {
        const auto x3 = bld.Input(
            ir::Dense(Encoding::kF32, ir::ShapeOf({static_cast<std::uint64_t>(kWidth), 1ULL,
                                                   static_cast<std::uint64_t>(tokens)})));
        bld.Output(bld.MatMulId(wt, x3, idt));
        auto g = std::move(bld).Finish();
        if (!g) Die(g.error().detail);
        const std::array w_spec{co::StepWeight{
            .tensor = wt, .ref = ir::WeightRef{}, .parts = {{"slab", PartRole::kData}}}};
        tuned.push_back(b.Lower(*g, w_spec, [&](const ir::WeightRef&, PartRole) { return slab; },
                                {route.ids, x}));
      } else {
        const auto xt = bld.Input(ir::Dense(
            Encoding::kF32,
            ir::ShapeOf({static_cast<std::uint64_t>(kWidth), static_cast<std::uint64_t>(tokens)})));
        bld.Output(bld.MatMulId(wt, xt, idt));
        auto g = std::move(bld).Finish();
        if (!g) Die(g.error().detail);
        const std::array w_spec{co::StepWeight{
            .tensor = wt, .ref = ir::WeightRef{}, .parts = {{"slab", PartRole::kData}}}};
        tuned.push_back(b.Lower(*g, w_spec, [&](const ir::WeightRef&, PartRole) { return slab; },
                                {route.ids, x}));
        ir::Builder rbld;
        const auto ridt = rbld.Input(ir::Dense(
            Encoding::kI32,
            ir::ShapeOf({static_cast<std::uint64_t>(kUsed), static_cast<std::uint64_t>(tokens)})));
        const auto routed = rbld.ExpertSort(ridt, static_cast<std::uint32_t>(kExperts));
        rbld.Output(routed[0]);
        rbld.Output(routed[1]);
        auto rg = std::move(rbld).Finish();
        if (!rg) Die(rg.error().detail);
        route_only.push_back(
            b.Lower(*rg, {}, [&](const ir::WeightRef&, PartRole) { return 0ULL; }, {route.ids}));
      }
    }
    if (tokens <= 16) {
      b.Report("grouped", shape, tokens, {"llmp.moe.gemv", "llmp.decode.grouped.nvfp4.scale_atoms"},
               {b.Tuned(tuned), b.Ours(ours)});
      continue;
    }
    b.Report("grouped", shape, tokens,
             {"llmp.moe.route+quantize+gemm", "llmp.moe.route alone",
              "llmp.decode.grouped.nvfp4.scale_atoms"},
             {b.Tuned(tuned), b.Tuned(route_only), b.Ours(ours)});
  }
}

void Rows(Bench& b) {
  constexpr std::int64_t kTableRows = std::int64_t{1} << 20U;
  constexpr std::int64_t kValues = 160;
  constexpr std::int64_t kRowBytes = (kValues / 2) + (kValues / 16);
  const std::uint64_t table = b.Allocate(static_cast<std::uint64_t>(kTableRows * kRowBytes), 0x22);
  float* scale = b.Upload(std::vector<float>{1.0f});
  format::TensorFormat f = Format(Encoding::kNvfp4, Layout::kRowInterleaved, kValues, kTableRows);
  f.params.tensor_scale = true;
  const std::array<decode::DevicePart, 2> parts = {
      {{PartRole::kData, Pointer<const void>(table),
        static_cast<std::uint64_t>(kTableRows * kRowBytes)},
       {PartRole::kTensorScale, scale, 4}}};
  for (const std::int64_t tokens : {std::int64_t{1}, std::int64_t{16}, std::int64_t{512}}) {
    if (!b.RowsEnabled(tokens)) continue;
    const std::int64_t count = 16 * tokens;
    std::vector<std::int32_t*> id_sets;
    std::mt19937 rng(static_cast<std::uint32_t>(tokens));
    for (std::size_t i = 0; i < kRoutings; ++i) {
      std::vector<std::int32_t> ids(static_cast<std::size_t>(count));
      for (std::int32_t& id : ids) {
        id = static_cast<std::int32_t>(rng() % kTableRows);
      }
      id_sets.push_back(b.Upload(ids));
    }
    auto* out = Pointer<float>(b.Allocate(static_cast<std::uint64_t>(count * kValues * 4)));
    std::vector<kg::LoweredStep> tuned;
    for (std::int32_t* ids : id_sets) {
      ir::Builder bld;
      const auto wt = bld.Weight(f, ir::WeightRef{});
      const auto it =
          bld.Input(ir::Dense(Encoding::kI32, ir::ShapeOf({static_cast<std::uint64_t>(count)})));
      bld.Output(bld.GetRows(wt, it));
      auto g = std::move(bld).Finish();
      if (!g) Die(g.error().detail);
      const std::array w_spec{
          co::StepWeight{.tensor = wt,
                         .ref = ir::WeightRef{},
                         .parts = {{"data", PartRole::kData}, {"scale", PartRole::kTensorScale}}}};
      tuned.push_back(b.Lower(*g, w_spec,
                              [&](const ir::WeightRef&, PartRole role) {
                                return role == PartRole::kData
                                           ? table
                                           : reinterpret_cast<std::uintptr_t>(scale);
                              },
                              {ids}));
    }
    const auto incumbent = b.Tuned(tuned);
    b.Report("rows", "nvfp4.row_interleaved 160 values, 16 ids a token", tokens,
             {"llmp.nvfp4.get_rows", "llmp.decode.rows.nvfp4.row_interleaved"},
             {incumbent, b.Ours([&](void* s, std::size_t r) {
                const decode::RowsGather g{.weight = {.format = f, .parts = parts},
                                           .ids = id_sets[r % id_sets.size()],
                                           .count = static_cast<std::uint64_t>(count),
                                           .out = out,
                                           .out_stride = kValues};
                return decode::RunRows(g, s);
              })},
             static_cast<const float*>(tuned.back().keep.front()->data), out,
             static_cast<std::size_t>(count * kValues), 0.0);
  }
}

// Gemma 2 2B's dense BF16 FFN projection (the frozen fixture profile).
void Dense(Bench& b) {
  constexpr std::int64_t k = 2304, n = 9216;
  std::vector<decode::DevicePart> weights;
  for (int i = 0; i < 4; ++i) {
    auto* data = b.Upload(std::vector<std::uint16_t>(k * n, 0x3f80));
    weights.push_back({PartRole::kData, data, k * n * 2});
  }
  for (const std::int64_t rows : {std::int64_t{1}, std::int64_t{16}, std::int64_t{512}}) {
    if (!b.RowsEnabled(rows)) continue;
    const auto inputs = Normal(k * static_cast<std::uint64_t>(rows), 7);
    auto* x = b.Upload(inputs);
    if (rows == 1) {
      double f32_sum = 0, bf16_sum = 0;
      for (const float value : inputs) {
        f32_sum += value;
        auto bits = std::bit_cast<std::uint32_t>(value);
        bits += 0x7fffU + ((bits >> 16U) & 1U);
        bf16_sum += std::bit_cast<float>(bits & 0xffff0000U);
      }
      // The constant unit weights make the expected activation-rounding
      // difference independently visible before any candidate can qualify.
      std::println(R"({{"kind":"dense_input_sums","f32":{:.9g},"bf16":{:.9g}}})", f32_sum,
                   bf16_sum);
    }
    auto* out = Pointer<float>(b.Allocate(n * static_cast<std::uint64_t>(rows) * 4));
    std::vector<kg::LoweredStep> tuned;
    for (std::size_t i = 0; i < weights.size(); ++i) {
      ir::Builder bld;
      const auto xt = bld.Input(ir::Dense(
          Encoding::kF32,
          ir::ShapeOf({static_cast<std::uint64_t>(k), static_cast<std::uint64_t>(rows)})));
      const auto wt = bld.Weight(Format(Encoding::kBF16, Layout::kDense, k, n),
                                 ir::WeightRef{.layer = static_cast<std::uint32_t>(i)});
      bld.Output(bld.MatMul(wt, xt));
      auto g = std::move(bld).Finish();
      if (!g) Die(g.error().detail);
      const std::array w_spec{
          co::StepWeight{.tensor = wt,
                         .ref = ir::WeightRef{.layer = static_cast<std::uint32_t>(i)},
                         .parts = {{"data", PartRole::kData}}}};
      tuned.push_back(b.Lower(*g, w_spec,
                              [&](const ir::WeightRef&, PartRole) {
                                return reinterpret_cast<std::uintptr_t>(weights[i].address);
                              },
                              {x}));
    }
    const auto product = [&](std::size_t r) {
      return decode::Product{.weight = {.format = Format(Encoding::kBF16, Layout::kDense, k, n),
                                        .parts = std::span(weights).subspan(r % weights.size(), 1)},
                             .outputs = static_cast<std::uint64_t>(n),
                             .x = x,
                             .rows = static_cast<std::uint64_t>(rows),
                             .x_stride = static_cast<std::uint64_t>(k),
                             .y = out,
                             .y_stride = static_cast<std::uint64_t>(n)};
    };
    std::vector<std::string> names{"ggml.mul_mat", "llmp.decode.matrix.bf16.dense",
                                   "llmp.decode.matrix_fma.bf16.dense"};
    std::vector<Bench::Replay> arms{
        b.Tuned(tuned), b.Ours([&](void* stream, std::size_t r) {
          return decode::RunMatrix(product(r), decode::MatrixPath::kMma, stream);
        }),
        b.Ours([&](void* stream, std::size_t r) {
          return decode::RunMatrix(product(r), decode::MatrixPath::kFma, stream);
        })};
    if (rows <= 16) {
      names.push_back("llmp.decode.vector.bf16.dense");
      arms.push_back(b.Ours(
          [&](void* stream, std::size_t r) { return decode::RunVector(product(r), stream); }));
    }
    for (std::size_t i = 1; i < arms.size(); ++i) {
      b.Report("dense", "bf16.dense K2304 N9216", rows, {names[0], names[i]}, {arms[0], arms[i]},
               static_cast<const float*>(tuned.back().keep.front()->data), out,
               static_cast<std::size_t>(n * rows));
    }
  }
}
std::string_view Case(int argc, char** argv) {
  for (int i = 1; i + 1 < argc; ++i)
    if (std::string_view(argv[i]) == "--case") return argv[i + 1];
  return "all";
}

int Flag(int argc, char** argv, std::string_view name, int fallback) {
  const std::span<char*> args(argv, static_cast<std::size_t>(argc));
  for (std::size_t i = 1; i + 1 < args.size(); ++i) {
    if (std::string_view(args[i]) == name) {
      int v = 0;
      const std::string_view text(args[i + 1]);
      if (std::from_chars(text.data(), text.data() + text.size(), v).ec == std::errc{} && v > 0) {
        return v;
      }
      Die(std::format("{} takes a positive integer", name));
    }
  }
  return fallback;
}

}  // namespace

int main(int argc, char** argv) {
  const auto selected = Case(argc, argv);
  if (selected != "all" && selected != "mxfp8" && selected != "glu" && selected != "grouped" &&
      selected != "rows" && selected != "dense")
    Die("unknown --case");
  const int reps = Flag(argc, argv, "--reps", 64), batches = Flag(argc, argv, "--batches", 7);
  const int rows = Flag(argc, argv, "--rows", 0);
  cudaDeviceProp device{};
  Cuda(cudaGetDeviceProperties(&device, 0), "device properties");
  int driver = 0, runtime = 0;
  Cuda(cudaDriverGetVersion(&driver), "driver version");
  Cuda(cudaRuntimeGetVersion(&runtime), "runtime version");
  std::println(
      R"({{"kind":"provenance","device":"{}","arch":{},"sms":{},"driver":{},"runtime":{},"reps":{},"batches":{},"case":"{}","rows_filter":{}}})",
      device.name, device.major * 10 + device.minor, device.multiProcessorCount, driver, runtime,
      reps, batches, selected, rows);
  Bench b(reps, batches, rows);
  if (selected == "all" || selected == "mxfp8") VectorAndMatrix(b);
  if (selected == "all" || selected == "glu") Glu(b);
  if (selected == "all" || selected == "grouped") Grouped(b);
  if (selected == "all" || selected == "rows") Rows(b);
  if (selected == "all" || selected == "dense") Dense(b);
  std::_Exit(0);  // the process ends; its GPU resources go with it
}
