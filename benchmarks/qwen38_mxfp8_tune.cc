// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Bounded, benchmark-only MXFP8 vector scheduling sweep. The reference is
// the actual production launcher. Every candidate keeps its arithmetic,
// F32 input/output and E4M3/E8M0 weights; only work/load scheduling changes.
// Shapes are columns,outputs,inputs. --padding is F32 elements per column.
// --check-only on covers tails/strides without timing a small hot matrix.
// Timing captures every bank in a rotating ~256 MiB weight set, capped at
// 1,024 banks and explicitly reporting any smaller set. Three graph/event
// batches are reported as a median, with production controls at both ends.
// CUDA/unknown-completion failures terminate without resource destructors.

#include <cuda_runtime_api.h>

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
#include <initializer_list>
#include <memory>
#include <print>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include "base/bytes.h"
#include "ggml.h"
#include "kernels/ggml/launch.h"
#include "kernels/ggml/llmp_ops.h"
#include "kernels/ggml/tensors.h"
#include "providers/cuda/cuda_device_execution.h"
#include "qwen38_mxfp8_tune_kernels.h"

namespace {
namespace kg = llmp::kernels::ggml;
namespace diag = llmp::diag;

using Status = std::expected<void, std::string>;
std::unexpected<std::string> Error(std::string message) {
  return std::unexpected(std::move(message));
}
void Cuda(cudaError_t status, std::string_view operation) {
  if (status != cudaSuccess) {
    std::println(stderr, "terminal CUDA failure: {}: {}", operation, cudaGetErrorString(status));
    (void)std::fflush(stderr);
    std::_Exit(1);
  }
}
void Kernel(const std::expected<void, kg::KernelFailure>& result) {
  if (!result) {
    std::println(stderr, "kernel failure: {}", result.error().detail);
    (void)std::fflush(stderr);
    std::_Exit(1);
  }
}
struct Shape {
  int columns = 0;
  int outputs = 0;
  int inputs = 0;
};
struct Options {
  int padding = 0;
  int reps = 128;
  int warmup_ms = 20;
  bool check_only = false;
  std::vector<Shape> shapes;
};
bool Integer(std::string_view text, int& value) {
  const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
  return error == std::errc() && end == text.data() + text.size();
}
std::expected<Options, std::string> Parse(std::span<char*> args) {
  Options options;
  for (std::size_t i = 1; i < args.size(); ++i) {
    const std::string_view option(args[i]);
    if (i + 1 == args.size()) return Error(std::format("{} needs a value", option));
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
           dims[1] <= 262144 && dims[2] >= 32 && dims[2] <= 20480 && dims[2] % 32 == 0;
      if (ok) options.shapes.push_back({dims[0], dims[1], dims[2]});
    } else if (option == "--padding") {
      ok = Integer(value, options.padding) && options.padding >= 0 && options.padding <= 4096 &&
           options.padding % 4 == 0;
    } else if (option == "--reps") {
      ok = Integer(value, options.reps) && options.reps >= 1 && options.reps <= 16384;
    } else if (option == "--warmup-ms") {
      ok = Integer(value, options.warmup_ms) && options.warmup_ms >= 0 && options.warmup_ms <= 100;
    } else if (option == "--check-only") {
      ok = value == "on" || value == "off";
      options.check_only = value == "on";
    } else {
      return Error(std::format("unknown option {}", option));
    }
    if (!ok) return Error(std::format("invalid {} value {}", option, value));
  }
  if (options.shapes.empty() || options.shapes.size() > 32)
    return Error("provide between 1 and 32 --shape columns,outputs,inputs cases");
  for (const Shape& shape : options.shapes) {
    const auto weights =
        static_cast<std::uint64_t>(shape.outputs) * static_cast<std::uint64_t>(shape.inputs);
    if (weights > (std::uint64_t{512} << 20U))
      return Error("a case's code weights exceed the 512 MiB benchmark bound");
  }
  return options;
}

class Fixture {
 public:
  Fixture() {
    auto execution = llmp::providers::cuda::OpenDeviceExecution(0);
    if (!execution) Terminal(execution.error().detail);
    execution_ = std::move(*execution);
    auto stream = execution_->CreateStream();
    if (!stream) Terminal(stream.error().detail);
    stream_ = *stream;
    auto native = execution_->Submission(stream_);
    if (!native) Terminal(native.error().detail);
    native_ = static_cast<cudaStream_t>(native->handle);
    auto launch = kg::LaunchContext::Create(0, *execution_, stream_,
                                            {.base = 0, .size = llmp::base::Bytes(0)});
    if (!launch) Terminal(launch.error().detail);
    launch_ = std::move(*launch);
  }
  Fixture(const Fixture&) = delete;
  Fixture& operator=(const Fixture&) = delete;
  Fixture(Fixture&&) = delete;
  Fixture& operator=(Fixture&&) = delete;
  ~Fixture() {
    Cuda(cudaStreamSynchronize(native_), "fixture completion");
    auto fence = execution_->Record(stream_);
    if (!fence) Terminal(fence.error().detail);
    Cuda(cudaStreamSynchronize(native_), "fence completion");
    auto state = execution_->Query(*fence);
    if (!state || *state != llmp::providers::FenceState::kComplete || !execution_->Release(*fence))
      Terminal("fence retirement failed");
    launch_.reset();
    if (!execution_->DestroyStream(stream_)) Terminal("stream destruction failed");
  }
  cudaStream_t stream() const { return native_; }
  kg::LaunchContext& launch() { return *launch_; }

 private:
  [[noreturn]] static void Terminal(std::string_view message) {
    std::println(stderr, "terminal provider failure: {}", message);
    (void)std::fflush(stderr);
    std::_Exit(1);
  }
  std::unique_ptr<llmp::providers::DeviceExecution> execution_;
  llmp::providers::StreamId stream_;
  cudaStream_t native_ = nullptr;
  std::unique_ptr<kg::LaunchContext> launch_;
};
struct Buffers {
  cudaStream_t stream = nullptr;
  std::vector<void*> owners;
  explicit Buffers(cudaStream_t owner) : stream(owner) {}
  Buffers(const Buffers&) = delete;
  Buffers& operator=(const Buffers&) = delete;
  Buffers(Buffers&&) = delete;
  Buffers& operator=(Buffers&&) = delete;
  ~Buffers() {
    Cuda(cudaStreamSynchronize(stream), "buffer completion");
    for (void* owner : owners) Cuda(cudaFree(owner), "free buffer");
  }
  template <class T>
  T* Allocate(std::uint64_t count) {
    void* memory = nullptr;
    Cuda(cudaMalloc(&memory, static_cast<std::size_t>(count * sizeof(T))), "allocate");
    owners.push_back(memory);
    return static_cast<T*>(memory);
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
    Cuda(cudaStreamSynchronize(stream), "graph completion");
    if (executable != nullptr) Cuda(cudaGraphExecDestroy(executable), "destroy graph executable");
    if (graph != nullptr) Cuda(cudaGraphDestroy(graph), "destroy graph");
  }
};
struct Events {
  cudaEvent_t start = nullptr;
  cudaEvent_t end = nullptr;
  Events() {
    Cuda(cudaEventCreate(&start), "event");
    Cuda(cudaEventCreate(&end), "event");
  }
  Events(const Events&) = delete;
  Events& operator=(const Events&) = delete;
  Events(Events&&) = delete;
  Events& operator=(Events&&) = delete;
  ~Events() {
    Cuda(cudaEventDestroy(end), "destroy event");
    Cuda(cudaEventDestroy(start), "destroy event");
  }
};
template <class Launch>
double Time(const Options& options, cudaStream_t stream, int capture_calls, Launch&& launch) {
  Events events;
  const auto batch = [&](int calls) {
    Cuda(cudaEventRecord(events.start, stream), "timing event");
    for (int i = 0; i < calls; ++i) launch();
    Cuda(cudaEventRecord(events.end, stream), "timing event");
    Cuda(cudaEventSynchronize(events.end), "timing completion");
    float elapsed = 0;
    Cuda(cudaEventElapsedTime(&elapsed, events.start, events.end), "timing read");
    return static_cast<double>(elapsed) / calls;
  };
  const double first = batch(1);
  if (options.warmup_ms > 0) {
    const int warm =
        std::clamp(static_cast<int>(options.warmup_ms / std::max(first, 0.001)), 1, 10000);
    (void)batch(warm);
  }
  Captured captured(stream);
  Cuda(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "capture");
  for (int i = 0; i < capture_calls; ++i) launch();
  Cuda(cudaStreamEndCapture(stream, &captured.graph), "capture completion");
  Cuda(cudaGraphInstantiate(&captured.executable, captured.graph, 0), "graph instantiate");
  const int graphs = (options.reps + capture_calls - 1) / capture_calls;
  std::array<double, 3> values{};
  for (double& value : values) {
    Cuda(cudaEventRecord(events.start, stream), "graph timing");
    for (int i = 0; i < graphs; ++i) Cuda(cudaGraphLaunch(captured.executable, stream), "graph");
    Cuda(cudaEventRecord(events.end, stream), "graph timing");
    Cuda(cudaEventSynchronize(events.end), "graph completion");
    float elapsed = 0;
    Cuda(cudaEventElapsedTime(&elapsed, events.start, events.end), "graph timing read");
    value = static_cast<double>(elapsed) / (graphs * capture_calls);
  }
  std::ranges::sort(values);
  return values[1];
}
std::uint32_t Hash(std::size_t i) {
  std::uint32_t value = static_cast<std::uint32_t>(i) * 2654435761U;
  value ^= value >> 15U;
  value *= 2246822519U;
  return value ^ (value >> 13U);
}
struct Differences {
  std::uint64_t bits = 0;
  std::uint64_t nonfinite = 0;
};
Differences Compare(const float* actual, const float* reference, std::size_t count) {
  std::vector<float> a(count);
  std::vector<float> b(count);
  Cuda(cudaMemcpy(a.data(), actual, count * sizeof(float), cudaMemcpyDeviceToHost), "compare");
  Cuda(cudaMemcpy(b.data(), reference, count * sizeof(float), cudaMemcpyDeviceToHost), "compare");
  Differences difference;
  for (std::size_t i = 0; i < count; ++i) {
    difference.bits +=
        std::bit_cast<std::uint32_t>(a[i]) != std::bit_cast<std::uint32_t>(b[i]) ? 1U : 0U;
    difference.nonfinite += !std::isfinite(a[i]) || !std::isfinite(b[i]) ? 1U : 0U;
  }
  return difference;
}

Status Tune(Fixture& fixture, const Options& options, const Shape& shape) {
  const auto k = static_cast<std::uint64_t>(shape.inputs);
  const auto n = static_cast<std::uint64_t>(shape.outputs);
  const auto t = static_cast<std::uint64_t>(shape.columns);
  const std::uint64_t code_bytes = k * n;
  const std::uint64_t scale_bytes = code_bytes / 32;
  const std::uint64_t weight_bytes = code_bytes + scale_bytes;
  const std::uint64_t copies =
      options.check_only
          ? 1U
          : std::clamp<std::uint64_t>(
                ((std::uint64_t{256} << 20U) + weight_bytes - 1) / weight_bytes, 1, 1024);
  const int capture_calls = std::max(64, static_cast<int>(copies));
  const int stride = shape.inputs + options.padding;
  const auto outputs = static_cast<std::size_t>(n * t);
  Buffers buffers(fixture.stream());
  auto* codes = buffers.Allocate<std::uint8_t>(code_bytes * copies);
  auto* scales = buffers.Allocate<std::uint8_t>(scale_bytes * copies);
  auto* input = buffers.Allocate<float>(static_cast<std::uint64_t>(stride) * t);
  auto* output = buffers.Allocate<float>(n * t);
  auto* reference = buffers.Allocate<float>(n * t);
  auto* repeat = buffers.Allocate<float>(n * t);
  std::vector<std::uint8_t> host_codes(static_cast<std::size_t>(code_bytes));
  std::vector<std::uint8_t> host_scales(static_cast<std::size_t>(scale_bytes));
  std::vector<float> host_input(static_cast<std::size_t>(static_cast<std::uint64_t>(stride) * t));
  for (std::size_t i = 0; i < host_codes.size(); ++i) {
    const auto code = static_cast<std::uint8_t>(Hash(i) & 255U);
    host_codes[i] = (code & 127U) == 127U ? static_cast<std::uint8_t>(code - 1) : code;
  }
  for (std::size_t i = 0; i < host_scales.size(); ++i)
    host_scales[i] = i % 257 == 0 ? 0U : static_cast<std::uint8_t>(117 + (Hash(i) % 14));
  for (std::size_t i = 0; i < host_input.size(); ++i)
    host_input[i] = (static_cast<float>(Hash(i) & 65535U) / 32768.0f) - 1.0f;
  Cuda(cudaMemcpy(codes, host_codes.data(), host_codes.size(), cudaMemcpyHostToDevice), "codes");
  Cuda(cudaMemcpy(scales, host_scales.data(), host_scales.size(), cudaMemcpyHostToDevice),
       "scales");
  Cuda(cudaMemcpy(input, host_input.data(), host_input.size() * sizeof(float),
                  cudaMemcpyHostToDevice),
       "input");
  for (std::uint64_t copy = 1; copy < copies; ++copy) {
    Cuda(cudaMemcpyAsync(codes + (copy * code_bytes), codes, static_cast<std::size_t>(code_bytes),
                         cudaMemcpyDeviceToDevice, fixture.stream()),
         "code banks");
    Cuda(cudaMemcpyAsync(scales + (copy * scale_bytes), scales,
                         static_cast<std::size_t>(scale_bytes), cudaMemcpyDeviceToDevice,
                         fixture.stream()),
         "scale banks");
  }
  Cuda(cudaStreamSynchronize(fixture.stream()), "initialization");
  auto arena = kg::TensorArena::Create(8);
  if (!arena) return Error(arena.error().detail);
  auto* c = arena->context();
  auto* wc = ggml_new_tensor_2d(c, GGML_TYPE_I8, shape.inputs, shape.outputs);
  auto* ws = ggml_new_tensor_2d(c, GGML_TYPE_I8, shape.inputs / 32, shape.outputs);
  auto* x = ggml_new_tensor_2d(c, GGML_TYPE_F32, shape.inputs, shape.columns);
  x->nb[1] = static_cast<std::size_t>(stride) * sizeof(float);
  x->nb[2] = x->nb[1] * static_cast<std::size_t>(shape.columns);
  x->nb[3] = x->nb[2];
  kg::TensorArena::Bind(wc, reinterpret_cast<std::uintptr_t>(codes));
  kg::TensorArena::Bind(ws, reinterpret_cast<std::uintptr_t>(scales));
  kg::TensorArena::Bind(x, reinterpret_cast<std::uintptr_t>(input));
  auto* y = kg::Mxfp8MulMatVec(c, wc, ws, x);
  kg::TensorArena::Bind(y, reinterpret_cast<std::uintptr_t>(reference));
  Kernel(kg::RunMxfp8MulMatVec(fixture.launch(), y));
  Cuda(cudaStreamSynchronize(fixture.stream()), "reference completion");
  std::uint64_t index = 0;
  const auto production = [&]() {
    const auto copy = (index++) % copies;
    wc->data = codes + (copy * code_bytes);
    ws->data = scales + (copy * scale_bytes);
    y->data = output;
    Kernel(kg::RunMxfp8MulMatVec(fixture.launch(), y));
  };
  const auto print_control = [&](std::string_view arm) {
    const double ms = Time(options, fixture.stream(), capture_calls, production);
    std::println(R"({{"shape":[{},{},{}],"padding":{},"arm":"{}","ms":{:.8f},)"
                 R"("weight_bytes":{},"copies":{},"working_set":{},"capture_calls":{}}})",
                 shape.columns, shape.outputs, shape.inputs, options.padding, arm, ms, weight_bytes,
                 copies, weight_bytes * copies, capture_calls);
    (void)std::fflush(stdout);
  };
  if (!options.check_only) print_control("production-before");
  bool failed = false;
  for (bool column_at_a_time : {false, true}) {
    for (int warps : {4, 8}) {
      for (int rows : {1, 2, 4}) {
        const diag::Mxfp8Schedule schedule{rows, warps, column_at_a_time};
        const auto run = [&](float* destination, std::uint64_t copy) {
          Cuda(diag::Mxfp8Tune(fixture.stream(), codes + (copy * code_bytes),
                               scales + (copy * scale_bytes), input, destination, shape.columns,
                               shape.outputs, shape.inputs, stride, schedule),
               "candidate launch");
        };
        run(output, 0);
        run(repeat, 0);
        Cuda(cudaStreamSynchronize(fixture.stream()), "candidate completion");
        const Differences difference = Compare(output, reference, outputs);
        const Differences repetition = Compare(output, repeat, outputs);
        failed = failed || difference.bits != 0 || difference.nonfinite != 0 ||
                 repetition.bits != 0 || repetition.nonfinite != 0;
        const double ms = options.check_only ? 0
                                             : Time(options, fixture.stream(), capture_calls,
                                                    [&]() { run(output, (index++) % copies); });
        std::println(R"({{"shape":[{},{},{}],"padding":{},"arm":"candidate",)"
                     R"("rows":{},"warps":{},"column_at_a_time":{},"ms":{:.8f},)"
                     R"("reference_differ":{},"repeat_differ":{},"nonfinite":{},)"
                     R"("weight_bytes":{},"copies":{},"working_set":{},"capture_calls":{}}})",
                     shape.columns, shape.outputs, shape.inputs, options.padding, rows, warps,
                     column_at_a_time, ms, difference.bits, repetition.bits,
                     difference.nonfinite + repetition.nonfinite, weight_bytes, copies,
                     weight_bytes * copies, options.check_only ? 0 : capture_calls);
        (void)std::fflush(stdout);
      }
    }
  }
  if (!options.check_only) print_control("production-after");
  return failed ? Error("a candidate differed, repeated differently or produced nonfinite output")
                : Status{};
}

Status Run(const Options& options) {
  cudaDeviceProp device{};
  Cuda(cudaGetDeviceProperties(&device, 0), "device");
  if (device.major != 12 || device.minor != 1)
    return Error("this scheduling diagnostic requires GB10");
  std::println(R"({{"device":"{}","cc":{}{},"sms":{},"l2_bytes":{},)"
               R"("reps":{},"warmup_ms":{},"check_only":{}}})",
               device.name, device.major, device.minor, device.multiProcessorCount,
               device.l2CacheSize, options.reps, options.warmup_ms, options.check_only);
  Fixture fixture;
  bool failed = false;
  for (const Shape& shape : options.shapes) {
    auto result = Tune(fixture, options, shape);
    if (!result) {
      failed = true;
      std::println(stderr, "{}x{}x{}: {}", shape.columns, shape.outputs, shape.inputs,
                   result.error());
    }
  }
  return failed ? Error("some cases failed") : Status{};
}
}  // namespace

int main(int argc, char** argv) {
  auto options = Parse(std::span(argv, static_cast<std::size_t>(argc)));
  if (!options) {
    std::println(stderr, "{}", options.error());
    return 2;
  }
  auto result = Run(*options);
  if (!result) {
    std::println(stderr, "{}", result.error());
    return 1;
  }
}
