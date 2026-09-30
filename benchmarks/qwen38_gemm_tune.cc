// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// Bounded cuBLASLt search for Qwen3.8's small-column BF16 products.
// Operands, accumulation and output types stay explicit. Mixed BF16 weights
// and F32 activations may have no library implementation; a refusal is a
// measured result, never permission to round the input.
//
//   jitllm_qwen38_gemm_tune --shape columns,outputs,inputs [...]
//     [--input bf16|f32] [--output bf16|f32] [--candidates 32]
//     [--reps 128] [--warmup-ms 100] [--workspace 32]
//     [--ggml-reference on|off]
//
// Reports JSON lines for cublasGemmEx and every heuristic candidate: exact
// types/shapes, nine algorithm attributes, workspace, median CUDA-event
// time over three CUDA-graph batches, repeatability and differences from
// GemmEx. At least 64 calls per graph remove per-call host dispatch gaps;
// the graph covers every allocated weight copy.
// Fixed generated inputs are not a model-quality check. The benchmark owns
// temporary cudaMalloc buffers; no model/artifact state or library pins
// change. Clean refusals are recorded and skipped; unknown completion
// terminates the process without running GPU-resource destructors.

#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <expected>
#include <format>
#include <memory>
#include <numeric>
#include <print>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

#include "base/bytes.h"
#include "ggml.h"
#include "kernels/ggml/jitllm_ops.h"
#include "kernels/ggml/launch.h"
#include "kernels/ggml/ops.h"
#include "kernels/ggml/tensors.h"
#include "providers/cuda/cuda_device_execution.h"

namespace {

namespace kg = jitllm::kernels::ggml;

using Status = std::expected<void, std::string>;
std::unexpected<std::string> Error(std::string message) {
  return std::unexpected(std::move(message));
}
Status Cuda(cudaError_t status, std::string_view operation) {
  if (status != cudaSuccess) {
    // Unknown completion quarantines every owner until process teardown.
    // _Exit queues no further work and runs no GPU-resource destructors.
    std::println(stderr, "terminal CUDA failure: {}: {}", operation, cudaGetErrorString(status));
    (void)std::fflush(stderr);
    std::_Exit(1);
  }
  return {};
}
Status Blas(cublasStatus_t status, std::string_view operation) {
  if (status == CUBLAS_STATUS_EXECUTION_FAILED || status == CUBLAS_STATUS_INTERNAL_ERROR) {
    std::println(stderr, "terminal library failure: {}: {}", operation,
                 cublasLtGetStatusString(status));
    (void)std::fflush(stderr);
    std::_Exit(1);
  }
  return status == CUBLAS_STATUS_SUCCESS
             ? Status{}
             : Error(std::format("{}: {}", operation, cublasLtGetStatusString(status)));
}

constexpr std::array<cublasLtMatmulAlgoConfigAttributes_t, 9> kAttributes{
    CUBLASLT_ALGO_CONFIG_ID,
    CUBLASLT_ALGO_CONFIG_TILE_ID,
    CUBLASLT_ALGO_CONFIG_SPLITK_NUM,
    CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,
    CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING,
    CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION,
    CUBLASLT_ALGO_CONFIG_STAGES_ID,
    CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID,
    CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID};

struct Shape {
  int columns = 0;
  int outputs = 0;
  int inputs = 0;
};
struct Options {
  cudaDataType_t input = CUDA_R_16BF;
  cudaDataType_t output = CUDA_R_32F;
  int candidates = 32;
  int reps = 128;
  int warmup_ms = 100;
  int workspace_mib = 32;
  bool ggml_reference = false;
  std::vector<Shape> shapes;
};

std::string_view TypeName(cudaDataType_t type) { return type == CUDA_R_16BF ? "bf16" : "f32"; }
std::size_t TypeBytes(cudaDataType_t type) { return type == CUDA_R_16BF ? 2U : 4U; }

bool Integer(std::string_view text, int& value) {
  const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
  return error == std::errc() && end == text.data() + text.size();
}
std::expected<Options, std::string> Parse(std::span<char*> args) {
  Options options;
  for (std::size_t i = 1; i < args.size(); ++i) {
    const std::string_view option(args[i]);
    if (i + 1 == args.size()) {
      return Error(std::format("{} needs a value", option));
    }
    const std::string_view value(args[++i]);
    bool ok = true;
    if (option == "--shape") {
      std::array<int, 3> dims{};
      std::size_t begin = 0;
      for (int& dim : dims) {
        const std::size_t end = std::min(value.find(',', begin), value.size());
        if (begin > value.size() || !Integer(value.substr(begin, end - begin), dim)) {
          ok = false;
          break;
        }
        begin = end + 1;
      }
      ok = ok && begin == value.size() + 1 && dims[0] >= 1 && dims[0] <= 8 && dims[1] >= 1 &&
           dims[1] <= 262144 && dims[2] >= 1 && dims[2] <= 20480;
      if (ok) {
        options.shapes.push_back({dims[0], dims[1], dims[2]});
      }
    } else if (option == "--input" || option == "--output") {
      ok = value == "bf16" || value == "f32";
      auto& type = option == "--input" ? options.input : options.output;
      type = value == "bf16" ? CUDA_R_16BF : CUDA_R_32F;
    } else if (option == "--candidates") {
      ok =
          Integer(value, options.candidates) && options.candidates >= 1 && options.candidates <= 64;
    } else if (option == "--reps") {
      ok = Integer(value, options.reps) && options.reps >= 1 && options.reps <= 1024;
    } else if (option == "--warmup-ms") {
      ok = Integer(value, options.warmup_ms) && options.warmup_ms >= 0 && options.warmup_ms <= 500;
    } else if (option == "--workspace") {
      ok = Integer(value, options.workspace_mib) && options.workspace_mib >= 0 &&
           options.workspace_mib <= 32;
    } else if (option == "--ggml-reference") {
      ok = value == "on" || value == "off";
      options.ggml_reference = value == "on";
    } else {
      return Error(std::format("unknown option {}", option));
    }
    if (!ok) {
      return Error(std::format("invalid {} value {}", option, value));
    }
  }
  if (options.shapes.empty() || options.shapes.size() > 32) {
    return Error("provide between 1 and 32 --shape columns,outputs,inputs cases");
  }
  for (const Shape& shape : options.shapes) {
    const std::uint64_t weights = std::uint64_t{2} * static_cast<unsigned>(shape.outputs) *
                                  static_cast<unsigned>(shape.inputs);
    if (weights > (std::uint64_t{2} << 30U)) {
      return Error("a case's BF16 weights exceed the 2 GiB benchmark bound");
    }
    if (options.ggml_reference &&
        (options.input != CUDA_R_16BF || options.output != CUDA_R_32F || shape.columns < 2)) {
      return Error("GGML MMF controls require 2–8 columns, BF16 input and F32 output");
    }
  }
  return options;
}

struct Descriptors {
  cublasLtMatmulDesc_t operation = nullptr;
  cublasLtMatrixLayout_t weights = nullptr;
  cublasLtMatrixLayout_t input = nullptr;
  cublasLtMatrixLayout_t output = nullptr;
  cublasLtMatmulPreference_t preference = nullptr;
  Descriptors() = default;
  Descriptors(const Descriptors&) = delete;
  Descriptors& operator=(const Descriptors&) = delete;
  Descriptors(Descriptors&&) = delete;
  Descriptors& operator=(Descriptors&&) = delete;
  ~Descriptors() {
    if (preference != nullptr) (void)cublasLtMatmulPreferenceDestroy(preference);
    if (output != nullptr) (void)cublasLtMatrixLayoutDestroy(output);
    if (input != nullptr) (void)cublasLtMatrixLayoutDestroy(input);
    if (weights != nullptr) (void)cublasLtMatrixLayoutDestroy(weights);
    if (operation != nullptr) (void)cublasLtMatmulDescDestroy(operation);
  }
};

struct Buffers {
  cudaStream_t stream = nullptr;
  void* weights = nullptr;
  void* input = nullptr;
  void* output = nullptr;
  void* reference = nullptr;
  void* repeat = nullptr;
  void* workspace = nullptr;
  void* float_input = nullptr;
  void* ggml_reference = nullptr;
  explicit Buffers(cudaStream_t owner) : stream(owner) {}
  Buffers(const Buffers&) = delete;
  Buffers& operator=(const Buffers&) = delete;
  Buffers(Buffers&&) = delete;
  Buffers& operator=(Buffers&&) = delete;
  ~Buffers() {
    // Errors may leave earlier launches in flight too.
    if (stream != nullptr && !Cuda(cudaStreamSynchronize(stream), "buffer completion"))
      std::_Exit(1);
    for (void* pointer :
         {weights, input, output, reference, repeat, workspace, float_input, ggml_reference}) {
      if (pointer != nullptr) (void)cudaFree(pointer);
    }
  }
};

Status Fill(void* device, std::size_t count, cudaDataType_t type, std::uint32_t seed) {
  std::vector<std::byte> values(count * TypeBytes(type));
  for (std::size_t i = 0; i < count; ++i) {
    std::uint32_t hash = (static_cast<std::uint32_t>(i) * 2654435761U) ^ seed;
    hash ^= hash >> 15U;
    hash *= 2246822519U;
    hash ^= hash >> 13U;
    const float value = (static_cast<float>(hash & 0xffffU) / 32768.0f) - 1.0f;
    if (type == CUDA_R_16BF) {
      const auto bits = static_cast<std::uint16_t>(std::bit_cast<std::uint32_t>(value) >> 16U);
      std::memcpy(values.data() + (i * 2), &bits, 2);
    } else {
      std::memcpy(values.data() + (i * 4), &value, 4);
    }
  }
  return Cuda(cudaMemcpy(device, values.data(), values.size(), cudaMemcpyHostToDevice), "fill");
}

struct Differences {
  std::uint64_t count = 0;
  std::uint64_t nonfinite = 0;
  double max_abs = 0;
  double max_rel = 0;
};
std::expected<Differences, std::string> Compare(const void* actual, const void* reference,
                                                std::size_t count, cudaDataType_t type) {
  const std::size_t bytes = TypeBytes(type);
  std::vector<std::byte> a(count * bytes);
  std::vector<std::byte> b(count * bytes);
  if (auto copied = Cuda(cudaMemcpy(a.data(), actual, a.size(), cudaMemcpyDeviceToHost), "compare");
      !copied) {
    return std::unexpected(copied.error());
  }
  if (auto copied =
          Cuda(cudaMemcpy(b.data(), reference, b.size(), cudaMemcpyDeviceToHost), "compare");
      !copied) {
    return std::unexpected(copied.error());
  }
  const auto value = [type, bytes](const std::byte* pointer) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, pointer, bytes);
    if (type == CUDA_R_16BF) bits <<= 16U;
    return std::bit_cast<float>(bits);
  };
  Differences result;
  for (std::size_t i = 0; i < count; ++i) {
    const std::byte* x = a.data() + (i * bytes);
    const std::byte* y = b.data() + (i * bytes);
    result.count += std::memcmp(x, y, bytes) != 0 ? 1U : 0U;
    const double av = value(x);
    const double bv = value(y);
    if (!std::isfinite(av) || !std::isfinite(bv)) {
      ++result.nonfinite;
      continue;
    }
    const double difference = std::abs(av - bv);
    result.max_abs = std::max(result.max_abs, difference);
    result.max_rel = std::max(result.max_rel, difference / std::max(std::abs(bv), 1e-3));
  }
  return result;
}

std::expected<std::array<std::uint64_t, 9>, std::string> Config(const cublasLtMatmulAlgo_t& algo) {
  std::array<std::uint64_t, 9> result{};
  for (std::size_t i = 0; i < kAttributes.size(); ++i) {
    std::size_t size = 0;
    if (auto got =
            Blas(cublasLtMatmulAlgoConfigGetAttribute(&algo, kAttributes[i], nullptr, 0, &size),
                 "config size");
        !got)
      return std::unexpected(got.error());
    if (size == 0 || size > sizeof(std::uint64_t)) return Error("invalid config attribute size");
    if (auto got = Blas(
            cublasLtMatmulAlgoConfigGetAttribute(&algo, kAttributes[i], &result[i], size, &size),
            "config value");
        !got)
      return std::unexpected(got.error());
  }
  return result;
}

struct Events {
  cudaEvent_t start = nullptr, end = nullptr;
  Events() = default;
  Events(const Events&) = delete;
  Events& operator=(const Events&) = delete;
  Events(Events&&) = delete;
  Events& operator=(Events&&) = delete;
  ~Events() {
    if (end != nullptr) (void)cudaEventDestroy(end);
    if (start != nullptr) (void)cudaEventDestroy(start);
  }
};
struct Captured {
  cudaStream_t stream = nullptr;
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t executable = nullptr;
  explicit Captured(cudaStream_t owner) : stream(owner) {}
  Captured(const Captured&) = delete;
  Captured& operator=(const Captured&) = delete;
  Captured(Captured&&) = delete;
  Captured& operator=(Captured&&) = delete;
  ~Captured() {
    if (stream != nullptr && !Cuda(cudaStreamSynchronize(stream), "graph completion"))
      std::_Exit(1);
    if (executable != nullptr) (void)cudaGraphExecDestroy(executable);
    if (graph != nullptr) (void)cudaGraphDestroy(graph);
  }
};
template <class Launch>
std::expected<double, std::string> Time(const Options& options, cudaStream_t stream, void* output,
                                        int capture_calls, Launch&& launch) {
  Events events;
  if (auto made = Cuda(cudaEventCreate(&events.start), "event"); !made) {
    return std::unexpected(made.error());
  }
  if (auto made = Cuda(cudaEventCreate(&events.end), "event"); !made) {
    return std::unexpected(made.error());
  }
  const auto batch = [&](int repeats) -> std::expected<double, std::string> {
    if (auto recorded = Cuda(cudaEventRecord(events.start, stream), "event"); !recorded) {
      return std::unexpected(recorded.error());
    }
    for (int i = 0; i < repeats; ++i) {
      if (auto run = launch(output); !run) return std::unexpected(run.error());
    }
    if (auto recorded = Cuda(cudaEventRecord(events.end, stream), "event"); !recorded) {
      return std::unexpected(recorded.error());
    }
    if (auto waited = Cuda(cudaEventSynchronize(events.end), "event wait"); !waited) {
      return std::unexpected(waited.error());
    }
    float elapsed = 0;
    if (auto read = Cuda(cudaEventElapsedTime(&elapsed, events.start, events.end), "event time");
        !read)
      return std::unexpected(read.error());
    return static_cast<double>(elapsed) / repeats;
  };
  auto first = batch(1);
  if (!first) return first;
  const int warm =
      std::clamp(static_cast<int>(options.warmup_ms / std::max(*first, 0.001)), 1, 10000);
  if (auto warmed = batch(warm); !warmed) return warmed;
  Captured captured(stream);
  if (auto begun =
          Cuda(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "capture");
      !begun)
    return std::unexpected(begun.error());
  Status queued;
  for (int i = 0; i < capture_calls && queued; ++i) queued = launch(output);
  const auto ended = cudaStreamEndCapture(stream, &captured.graph);
  if (!queued) return std::unexpected(queued.error());
  if (auto complete = Cuda(ended, "capture completion"); !complete) {
    return std::unexpected(complete.error());
  }
  if (auto instantiated =
          Cuda(cudaGraphInstantiate(&captured.executable, captured.graph, 0), "graph instantiate");
      !instantiated) {
    return std::unexpected(instantiated.error());
  }
  std::array<double, 3> values{};
  const int graph_repeats = (options.reps + capture_calls - 1) / capture_calls;
  for (double& value : values) {
    if (auto recorded = Cuda(cudaEventRecord(events.start, stream), "graph timing"); !recorded) {
      return std::unexpected(recorded.error());
    }
    for (int i = 0; i < graph_repeats; ++i) {
      if (auto launched = Cuda(cudaGraphLaunch(captured.executable, stream), "graph replay");
          !launched)
        return std::unexpected(launched.error());
    }
    if (auto recorded = Cuda(cudaEventRecord(events.end, stream), "graph timing"); !recorded) {
      return std::unexpected(recorded.error());
    }
    if (auto waited = Cuda(cudaEventSynchronize(events.end), "graph timing wait"); !waited) {
      return std::unexpected(waited.error());
    }
    float elapsed = 0;
    if (auto read = Cuda(cudaEventElapsedTime(&elapsed, events.start, events.end), "graph time");
        !read)
      return std::unexpected(read.error());
    value = static_cast<double>(elapsed) / (graph_repeats * capture_calls);
  }
  std::ranges::sort(values);
  return values[1];
}

Status Tune(cublasLtHandle_t lt, cublasHandle_t blas, cudaStream_t stream, const Options& options,
            const Shape& shape, kg::LaunchContext& kernels) {
  Descriptors descriptors;
  const cublasOperation_t transpose = CUBLAS_OP_T;
  const auto k = static_cast<std::uint64_t>(shape.inputs);
  const auto n = static_cast<std::uint64_t>(shape.outputs);
  const auto t = static_cast<std::uint64_t>(shape.columns);
  const std::uint64_t weight_bytes = k * n * 2;
  const std::uint64_t workspace = static_cast<std::uint64_t>(options.workspace_mib) << 20U;
  const auto required = [&](cublasStatus_t status, std::string_view operation) {
    return Blas(status, operation);
  };
  if (auto result =
          required(cublasLtMatmulDescCreate(&descriptors.operation, CUBLAS_COMPUTE_32F, CUDA_R_32F),
                   "operation");
      !result)
    return result;
  if (auto result = required(
          cublasLtMatmulDescSetAttribute(descriptors.operation, CUBLASLT_MATMUL_DESC_TRANSA,
                                         &transpose, sizeof transpose),
          "transpose");
      !result)
    return result;
  if (auto result = required(
          cublasLtMatrixLayoutCreate(&descriptors.weights, CUDA_R_16BF, k, n, shape.inputs),
          "weights");
      !result)
    return result;
  if (auto result = required(
          cublasLtMatrixLayoutCreate(&descriptors.input, options.input, k, t, shape.inputs),
          "input");
      !result)
    return result;
  if (auto result = required(
          cublasLtMatrixLayoutCreate(&descriptors.output, options.output, n, t, shape.outputs),
          "output");
      !result)
    return result;
  if (auto result = required(cublasLtMatmulPreferenceCreate(&descriptors.preference), "preference");
      !result)
    return result;
  if (auto result = required(cublasLtMatmulPreferenceSetAttribute(
                                 descriptors.preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                 &workspace, sizeof workspace),
                             "workspace preference");
      !result)
    return result;
  const auto weight_alignment =
      static_cast<std::uint32_t>(std::gcd(weight_bytes, std::uint64_t{256}));
  if (auto result = required(cublasLtMatmulPreferenceSetAttribute(
                                 descriptors.preference, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES,
                                 &weight_alignment, sizeof weight_alignment),
                             "weight alignment");
      !result)
    return result;
  std::vector<cublasLtMatmulHeuristicResult_t> candidates(
      static_cast<std::size_t>(options.candidates));
  int returned = 0;
  const auto heuristic = cublasLtMatmulAlgoGetHeuristic(
      lt, descriptors.operation, descriptors.weights, descriptors.input, descriptors.output,
      descriptors.output, descriptors.preference, options.candidates, candidates.data(), &returned);
  if (heuristic != CUBLAS_STATUS_SUCCESS || returned == 0) {
    std::println(
        R"({{"shape":[{},{},{}],"input":"{}","output":"{}","candidates":0,"status":"{}"}})",
        shape.columns, shape.outputs, shape.inputs, TypeName(options.input),
        TypeName(options.output), cublasLtGetStatusString(heuristic));
    return {};
  }
  Buffers buffers(stream);
  const auto allocate = [](void** pointer, std::uint64_t bytes) {
    return Cuda(cudaMalloc(pointer, static_cast<std::size_t>(std::max<std::uint64_t>(bytes, 256))),
                "allocation");
  };
  // Consecutive model mixes use different weights. Rotate identical
  // operands in a 256 MiB working set rather than crediting cuBLAS for
  // repeatedly reading one L2-resident matrix. Tiny diagnostic shapes cap
  // copy count at 1,024 and report their smaller actual working set.
  const std::uint64_t copies = std::clamp<std::uint64_t>(
      ((std::uint64_t{256} << 20U) + weight_bytes - 1) / weight_bytes, 1, 1024);
  const int capture_calls = std::max(64, static_cast<int>(copies));
  if (auto result = allocate(&buffers.weights, weight_bytes * copies); !result) return result;
  if (auto result = allocate(&buffers.input, k * t * TypeBytes(options.input)); !result)
    return result;
  for (void** output : {&buffers.output, &buffers.reference, &buffers.repeat}) {
    if (auto result = allocate(output, n * t * TypeBytes(options.output)); !result) return result;
  }
  if (auto result = allocate(&buffers.workspace, workspace); !result) return result;
  if (auto result = Fill(buffers.weights, static_cast<std::size_t>(k * n), CUDA_R_16BF, 1); !result)
    return result;
  for (std::uint64_t copy = 1; copy < copies; ++copy) {
    if (auto result =
            Cuda(cudaMemcpyAsync(static_cast<std::byte*>(buffers.weights) + (copy * weight_bytes),
                                 buffers.weights, static_cast<std::size_t>(weight_bytes),
                                 cudaMemcpyDeviceToDevice, stream),
                 "weight copies");
        !result)
      return result;
  }
  if (auto result = Fill(buffers.input, static_cast<std::size_t>(k * t), options.input, 2); !result)
    return result;
  auto arena = kg::TensorArena::Create(8);
  if (!arena) return Error(arena.error().detail);
  ggml_tensor* mmf = nullptr;
  ggml_tensor* convert = nullptr;
  if (options.ggml_reference) {
    if (auto result = allocate(&buffers.float_input, k * t * 4); !result) return result;
    if (auto result = allocate(&buffers.ggml_reference, n * t * 4); !result) return result;
    if (auto result = Fill(buffers.float_input, static_cast<std::size_t>(k * t), CUDA_R_32F, 2);
        !result)
      return result;
    auto* c = arena->context();
    ggml_tensor* w = ggml_new_tensor_2d(c, GGML_TYPE_BF16, shape.inputs, shape.outputs);
    ggml_tensor* x = ggml_new_tensor_2d(c, GGML_TYPE_F32, shape.inputs, shape.columns);
    kg::TensorArena::Bind(w, reinterpret_cast<std::uintptr_t>(buffers.weights));
    kg::TensorArena::Bind(x, reinterpret_cast<std::uintptr_t>(buffers.float_input));
    mmf = ggml_mul_mat(c, w, x);
    convert = kg::ToBf16(c, x);
    kg::TensorArena::Bind(mmf, reinterpret_cast<std::uintptr_t>(buffers.ggml_reference));
    kg::TensorArena::Bind(convert, reinterpret_cast<std::uintptr_t>(buffers.input));
  }
  if (auto result =
          Blas(cublasSetWorkspace(blas, buffers.workspace, workspace), "cuBLAS workspace");
      !result)
    return result;
  const float alpha = 1.0f;
  const float beta = 0.0f;
  std::uint64_t weight_index = 0;
  const auto weights = [&]() {
    return static_cast<const std::byte*>(buffers.weights) +
           (((weight_index++) % copies) * weight_bytes);
  };
  const auto kernel_status = [](const std::expected<void, kg::KernelFailure>& result) -> Status {
    if (!result && result.error().error == kg::KernelError::kUnknown) {
      std::println(stderr, "terminal GGML failure: {}", result.error().detail);
      (void)std::fflush(stderr);
      std::_Exit(1);
    }
    return result ? Status{} : Error(result.error().detail);
  };
  const auto converted = [&]() -> Status {
    return convert == nullptr ? Status{} : kernel_status(kg::RunBf16(kernels, convert));
  };
  std::println(
      R"({{"shape":[{},{},{}],"weight_copies":{},"weight_working_set_bytes":{},"captured_calls":{}}})",
      shape.columns, shape.outputs, shape.inputs, copies, weight_bytes * copies, capture_calls);
  const auto reference = [&](void* output) {
    if (auto result = converted(); !result) return result;
    return Blas(
        cublasGemmEx(blas, CUBLAS_OP_T, CUBLAS_OP_N, shape.outputs, shape.columns, shape.inputs,
                     &alpha, weights(), CUDA_R_16BF, shape.inputs, buffers.input, options.input,
                     shape.inputs, &beta, output, options.output, shape.outputs, CUBLAS_COMPUTE_32F,
                     CUBLAS_GEMM_DEFAULT_TENSOR_OP),
        "GemmEx");
  };
  bool has_reference = false;
  if (auto result = reference(buffers.reference); result) {
    has_reference = true;
    auto timed = Time(options, stream, buffers.output, capture_calls, reference);
    if (!timed) return std::unexpected(timed.error());
    std::println(R"({{"shape":[{},{},{}],"input":"{}","output":"{}","path":"GemmEx","ms":{:.6f}}})",
                 shape.columns, shape.outputs, shape.inputs, TypeName(options.input),
                 TypeName(options.output), *timed);
  }
  bool failed = false;
  if (mmf != nullptr) {
    const auto original = [&](void* output) {
      kg::TensorArena::Bind(mmf->src[0], reinterpret_cast<std::uintptr_t>(weights()));
      kg::TensorArena::Bind(mmf, reinterpret_cast<std::uintptr_t>(output));
      return kernel_status(kg::MulMatF(kernels, mmf));
    };
    if (auto result = original(buffers.ggml_reference); !result) return result;
    auto timed = Time(options, stream, buffers.output, capture_calls, original);
    if (!timed) return std::unexpected(timed.error());
    std::println(R"({{"shape":[{},{},{}],"path":"GGML-MMF","ms":{:.6f},"conversion_in_lt":true}})",
                 shape.columns, shape.outputs, shape.inputs, *timed);
  }
  for (int rank = 0; rank < returned; ++rank) {
    const auto& candidate = candidates[static_cast<std::size_t>(rank)];
    if (candidate.state != CUBLAS_STATUS_SUCCESS || candidate.workspaceSize > workspace) continue;
    const auto launch = [&](void* output) {
      if (auto result = converted(); !result) return result;
      return Blas(cublasLtMatmul(lt, descriptors.operation, &alpha, weights(), descriptors.weights,
                                 buffers.input, descriptors.input, &beta, output,
                                 descriptors.output, output, descriptors.output, &candidate.algo,
                                 buffers.workspace, workspace, stream),
                  "Lt");
    };
    auto timed = Time(options, stream, buffers.output, capture_calls, launch);
    if (!timed) {
      failed = true;
      std::println(R"({{"shape":[{},{},{}],"rank":{},"error":"{}"}})", shape.columns, shape.outputs,
                   shape.inputs, rank, timed.error());
      continue;
    }
    if (auto result = launch(buffers.repeat); !result) return result;
    if (auto result = Cuda(cudaStreamSynchronize(stream), "completion"); !result) return result;
    auto repeat =
        Compare(buffers.output, buffers.repeat, static_cast<std::size_t>(n * t), options.output);
    auto difference = has_reference ? Compare(buffers.output, buffers.reference,
                                              static_cast<std::size_t>(n * t), options.output)
                                    : std::expected<Differences, std::string>(Differences{});
    auto config = Config(candidate.algo);
    if (!repeat || !difference || !config) return Error("output/config comparison failed");
    failed = failed || repeat->count != 0 || repeat->nonfinite != 0 || difference->nonfinite != 0;
    const auto& c = *config;
    auto vs_mmf = mmf == nullptr ? std::expected<Differences, std::string>(Differences{})
                                 : Compare(buffers.output, buffers.ggml_reference,
                                           static_cast<std::size_t>(n * t), options.output);
    if (!vs_mmf) return Error(vs_mmf.error());
    failed = failed || vs_mmf->nonfinite != 0;
    std::println(
        R"({{"shape":[{},{},{}],"input":"{}","output":"{}","rank":{},"ms":{:.6f},)"
        R"("workspace":{},"repeat_differ":{},"nonfinite":{},"has_reference":{},)"
        R"("reference_differ":{},"max_abs":{:.8g},"max_rel":{:.8g},"mmf_differ":{},"mmf_max_abs":{:.8g},)"
        R"("config":[{},{},{},{},{},{},{},{},{}]}})",
        shape.columns, shape.outputs, shape.inputs, TypeName(options.input),
        TypeName(options.output), rank, *timed, candidate.workspaceSize, repeat->count,
        repeat->nonfinite + difference->nonfinite, has_reference, difference->count,
        difference->max_abs, difference->max_rel, vs_mmf->count, vs_mmf->max_abs, c[0], c[1], c[2],
        c[3], c[4], c[5], c[6], c[7], c[8]);
    (void)std::fflush(stdout);
  }
  if (auto result = Cuda(cudaStreamSynchronize(stream), "final completion"); !result) return result;
  return failed ? Error("a candidate failed or produced unstable/nonfinite output") : Status{};
}

Status Run(const Options& options) {
  if (cublasLtGetVersion() != CUBLAS_VERSION) return Error("cuBLASLt does not match the SDK pin");
  for (char** entry = environ; *entry != nullptr; ++entry) {
    const std::string_view variable(*entry);
    const std::string_view name = variable.substr(0, variable.find('='));
    const bool logging = name == "CUBLAS_LOGINFO_DBG" || name == "CUBLAS_LOGDEST_DBG" ||
                         name == "CUBLASLT_LOG_LEVEL" || name == "CUBLASLT_LOG_FILE" ||
                         name == "CUBLASLT_LOG_MASK" || name == "CUBLAS_NVTX_LEVEL" ||
                         name == "CUBLASLT_NVTX_LEVEL";
    if ((!logging && name.starts_with("CUBLAS")) || name == "NVIDIA_TF32_OVERRIDE") {
      return Error(std::format("{} changes library numerics", name));
    }
  }
  cudaDeviceProp device{};
  if (auto result = Cuda(cudaGetDeviceProperties(&device, 0), "device"); !result) return result;
  std::println(
      R"({{"device":"{}","cc":{}{},"sms":{},"cublaslt":{},"workspace_mib":{},"reps":{},"warmup_ms":{}}})",
      device.name, device.major, device.minor, device.multiProcessorCount, cublasLtGetVersion(),
      options.workspace_mib, options.reps, options.warmup_ms);
  auto execution = jitllm::providers::cuda::OpenDeviceExecution(0);
  if (!execution) return Error(execution.error().detail);
  auto stream_id = (*execution)->CreateStream();
  if (!stream_id) return Error(stream_id.error().detail);
  auto native = (*execution)->Submission(*stream_id);
  if (!native) {
    if (!(*execution)->DestroyStream(*stream_id)) std::_Exit(1);
    return Error(native.error().detail);
  }
  auto* const stream = static_cast<cudaStream_t>(native->handle);
  auto kernels = kg::LaunchContext::Create(0, **execution, *stream_id,
                                           {.base = 0, .size = jitllm::base::Bytes(0)});
  if (!kernels) {
    if (!(*execution)->DestroyStream(*stream_id)) std::_Exit(1);
    return Error(kernels.error().detail);
  }
  cublasHandle_t blas = nullptr;
  cublasLtHandle_t lt = nullptr;
  Status result = Blas(cublasCreate(&blas), "cuBLAS handle");
  if (result) result = Blas(cublasLtCreate(&lt), "cuBLASLt handle");
  if (result) result = Blas(cublasSetMathMode(blas, CUBLAS_TF32_TENSOR_OP_MATH), "math mode");
  if (result) result = Blas(cublasSetStream(blas, stream), "cuBLAS stream");
  bool failed = false;
  if (result) {
    for (const Shape& shape : options.shapes) {
      if (auto tuned = Tune(lt, blas, stream, options, shape, **kernels); !tuned) {
        failed = true;
        std::println(stderr, "{}x{}x{}: {}", shape.columns, shape.outputs, shape.inputs,
                     tuned.error());
      }
    }
  }
  if (auto completed = Cuda(cudaStreamSynchronize(stream), "handle completion"); !completed)
    return completed;
  auto fence = (*execution)->Record(*stream_id);
  if (!fence) {
    std::println(stderr, "terminal fence failure: {}", fence.error().detail);
    (void)std::fflush(stderr);
    std::_Exit(1);
  }
  if (auto completed = Cuda(cudaStreamSynchronize(stream), "fence completion"); !completed)
    return completed;
  auto state = (*execution)->Query(*fence);
  if (!state || *state != jitllm::providers::FenceState::kComplete ||
      !(*execution)->Release(*fence)) {
    std::println(stderr, "terminal fence retirement failure");
    (void)std::fflush(stderr);
    std::_Exit(1);
  }
  if (lt != nullptr) (void)cublasLtDestroy(lt);
  if (blas != nullptr) (void)cublasDestroy(blas);
  kernels->reset();
  if (auto destroyed = (*execution)->DestroyStream(*stream_id); !destroyed) {
    return Error(destroyed.error().detail);
  }
  return result && failed ? Error("some cases failed") : result;
}

}  // namespace

int main(int argc, char** argv) {
  auto options = Parse(std::span(argv, static_cast<std::size_t>(argc)));
  if (!options) {
    std::println(stderr, "{}", options.error());
    return 2;
  }
  if (auto result = Run(*options); !result) {
    std::println(stderr, "{}", result.error());
    return 1;
  }
  return 0;
}
