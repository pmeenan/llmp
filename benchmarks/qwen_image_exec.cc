// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// M3's third model slice (docs/experiments/qwen-image-native/README.md):
// the Qwen-Image-2.1 BF16 pipeline run natively on one Spark from its D-056
// component artifacts and their D-089 composition, for the comparison with
// diffusers and a coarse speed and memory report.
//
//   llmp_qwen_image_exec --store DIR --composition ID --out DIR
//                          [--prompt TEXT] [--size N] [--steps N] [--stop-after N]
//                          [--reference DIR] [--embeds native|reference]
//                          [--noise FILE] [--force-latents] [--decode-reference]
//                          [--no-vae] [--phases resident|released]
//                          [--plan fast|legacy] [--choose ROLE=IMPLEMENTATION]...
//                          [--graphs on|off] [--runs N]
//
// - The composition is opened as untrusted input (artifact/composition.h)
//   and each component's artifact by the ID it names; each is bound to the
//   compiled-in profile (model/qwen_image.h) and its groups read with
//   direct I/O through pinned staging into device memory (the VAE's F32
//   weights converted to BF16 on the way, as diffusers loads them, in the
//   layout the plan's convolution reads). Only the groups a phase reads are
//   loaded: the text encoder's table and layers (not its vision tower or
//   head), the denoiser, the VAE's decoder.
// - Phases, each over its component: encode (Qwen3-VL's text path, the
//   system turn's tokens dropped), denoise (the block-causal DiT with the
//   prefix K/V cache and the flow-matching Euler scheduler, no guidance),
//   decode (the VAE decoder). With --phases released, each component is
//   loaded at its phase's start and released at its end; resident loads
//   all three first (as the diffusers baseline holds them).
// - Kernels: the pipeline's (kernels/image/pipeline.h), dispatched through
//   a plan bound against the implementation registry (D-053): --plan fast
//   (the default) or legacy (the M3 slice's), --choose replacing one role's
//   implementation ("dit.linear=image.linear.cublas"), the A/B of one
//   lever. --graphs on (the default; off with --force-latents) replays
//   each denoising step from the third on as one CUDA graph, captured once
//   (the steps' inputs live on the device).
// - --stop-after N runs only the first N steps of the schedule;
//   --decode-reference decodes diffusers' final latents (the VAE alone).
// - --reference: the diffusers tensors reference.py wrote; each phase's
//   output is compared with them (relative RMS, cosine, max |diff|), and
//   --embeds reference / --force-latents feed diffusers' prompt embeddings
//   and each step's latents instead of the native ones (teacher forcing,
//   isolating one component). --noise takes the initial latents (default:
//   the reference's latents_init).
// - Writes the prompt embeddings, every step's noise prediction, the final
//   latents, the decoded tensor and the image's RGBA pixels as raw files,
//   and report.json with the plan, the timings (per phase, per step, per
//   full generation over --runs plain runs), peak memory (the drop in
//   MemAvailable) and the comparisons.

#include <cuda_runtime.h>
#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <expected>
#include <filesystem>
#include <format>
#include <fstream>
#include <map>
#include <memory>
#include <optional>
#include <print>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include "artifact/artifact.h"
#include "artifact/composition.h"
#include "base/bytes.h"
#include "base/sha256.h"
#include "chat/chat.h"
#include "execution/registry.h"
#include "kernels/ggml/cublas.h"
#include "kernels/image/gemm.h"
#include "kernels/image/implementations.h"
#include "kernels/image/ops.h"
#include "kernels/image/pipeline.h"
#include "model/qwen_image.h"
#include "providers/cuda/cuda_device_execution.h"
#include "providers/device_execution.h"
#include "tokenizer/hf.h"
#include "tokenizer/tokenizer.h"

namespace {

namespace ja = llmp::artifact;
namespace kg = llmp::kernels::ggml;
namespace ki = llmp::kernels::image;
namespace md = llmp::model;
using llmp::base::Bytes;
using Status = std::expected<void, std::string>;
using Clock = std::chrono::steady_clock;
using Bf16 = std::uint16_t;

std::unexpected<std::string> Error(std::string what) { return std::unexpected(std::move(what)); }

Status Cuda(cudaError_t result, std::string_view what) {
  if (result != cudaSuccess) {
    return Error(std::format("{}: {}", what, cudaGetErrorString(result)));
  }
  return {};
}

double Seconds(Clock::duration d) { return std::chrono::duration<double>(d).count(); }
std::uint64_t Round(std::uint64_t bytes, std::uint64_t to) { return (bytes + to - 1) / to * to; }

template <typename T>
T* At(std::uint64_t address) {
  return reinterpret_cast<T*>(address);  // NOLINT(performance-no-int-to-ptr)
}

std::uint64_t MemAvailable() {
  std::ifstream file("/proc/meminfo");
  std::string key;
  std::uint64_t value = 0;
  std::string unit;
  while (file >> key >> value) {
    std::getline(file, unit);
    if (key == "MemAvailable:") {
      return value * 1024;
    }
  }
  return 0;
}

// The lowest MemAvailable seen while it runs, sampled every 50 ms.
class MemorySampler {
 public:
  MemorySampler() : low_(MemAvailable()), thread_([this] { Loop(); }) {}
  MemorySampler(const MemorySampler&) = delete;
  MemorySampler& operator=(const MemorySampler&) = delete;
  MemorySampler(MemorySampler&&) = delete;
  MemorySampler& operator=(MemorySampler&&) = delete;
  ~MemorySampler() {
    stop_ = true;
    thread_.join();
  }
  std::uint64_t low() const { return low_.load(); }
  void Reset() { low_ = MemAvailable(); }

 private:
  void Loop() {
    while (!stop_) {
      const std::uint64_t now = MemAvailable();
      std::uint64_t seen = low_.load();
      while (now < seen && !low_.compare_exchange_weak(seen, now)) {
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
  }
  std::atomic<std::uint64_t> low_;
  std::atomic<bool> stop_{false};
  std::thread thread_;
};

// A cudaMalloc'd region, freed when it goes.
class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  DeviceBuffer(DeviceBuffer&& o) noexcept
      : address_(std::exchange(o.address_, 0)), bytes_(std::exchange(o.bytes_, 0)) {}
  DeviceBuffer& operator=(DeviceBuffer&& o) noexcept {
    std::swap(address_, o.address_);
    std::swap(bytes_, o.bytes_);
    return *this;
  }
  ~DeviceBuffer() { Free(); }
  Status Allocate(std::uint64_t bytes, std::string_view what) {
    Free();
    void* p = nullptr;
    if (auto r = Cuda(cudaMalloc(&p, std::max<std::uint64_t>(bytes, 256)), what); !r) {
      return r;
    }
    address_ = reinterpret_cast<std::uintptr_t>(p);
    bytes_ = bytes;
    return {};
  }
  void Free() {
    if (address_ != 0) {
      (void)cudaFree(At<void>(address_));
      address_ = 0;
      bytes_ = 0;
    }
  }
  std::uint64_t address() const { return address_; }
  std::uint64_t bytes() const { return bytes_; }
  template <typename T>
  T* as() const {
    return At<T>(address_);
  }

 private:
  std::uint64_t address_ = 0;
  std::uint64_t bytes_ = 0;
};

class Device {
 public:
  static std::expected<std::unique_ptr<Device>, std::string> Open() {
    if (auto set = Cuda(cudaSetDevice(0), "cudaSetDevice"); !set) {
      return std::unexpected(set.error());
    }
    if (auto context = Cuda(cudaFree(nullptr), "the CUDA context"); !context) {
      return std::unexpected(context.error());
    }
    auto execution = llmp::providers::cuda::OpenDeviceExecution(0);
    if (!execution) {
      return Error("OpenDeviceExecution failed");
    }
    auto stream = (*execution)->CreateStream();
    if (!stream) {
      return Error("CreateStream failed");
    }
    auto device = std::unique_ptr<Device>(new Device(std::move(*execution), *stream));  // NOLINT
    const auto native = device->execution_->Submission(device->stream_);
    if (!native) {
      return Error("Submission failed");
    }
    device->native_ = static_cast<cudaStream_t>(native->handle);
    return device;
  }
  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;
  Device(Device&&) = delete;
  Device& operator=(Device&&) = delete;
  ~Device() {
    if (!Finish() || !execution_->DestroyStream(stream_)) {
      std::println(stderr, "the stream could not be retired");
    }
  }
  llmp::providers::DeviceExecution& execution() { return *execution_; }
  llmp::providers::StreamId stream() const { return stream_; }
  cudaStream_t native() const { return native_; }
  Status Finish() {
    const auto fence = execution_->Record(stream_);
    if (!fence) {
      return Error("Record failed");
    }
    const auto deadline = Clock::now() + std::chrono::minutes(5);
    for (;;) {
      const auto state = execution_->Query(*fence);
      if (!state) {
        return Error("Query failed");
      }
      if (*state == llmp::providers::FenceState::kComplete) {
        break;
      }
      if (Clock::now() > deadline) {
        return Error("the stream did not complete");
      }
      std::this_thread::sleep_for(std::chrono::microseconds(20));
    }
    if (!execution_->Release(*fence)) {
      return Error("Release failed");
    }
    return Cuda(cudaGetLastError(), "the stream");
  }

 private:
  Device(std::unique_ptr<llmp::providers::DeviceExecution> execution,
         llmp::providers::StreamId stream)
      : execution_(std::move(execution)), stream_(stream) {}
  std::unique_ptr<llmp::providers::DeviceExecution> execution_;
  llmp::providers::StreamId stream_;
  cudaStream_t native_ = nullptr;
};

// Waits for the stream when it leaves scope. Declared after buffers that
// queued work uses, it runs before they are freed on every return path, so
// no buffer is freed under queued work (a free is not a completion).
class FinishFirst {
 public:
  explicit FinishFirst(Device& d) : d_(d) {}
  FinishFirst(const FinishFirst&) = delete;
  FinishFirst& operator=(const FinishFirst&) = delete;
  FinishFirst(FinishFirst&&) = delete;
  FinishFirst& operator=(FinishFirst&&) = delete;
  ~FinishFirst() {
    if (auto r = d_.Finish(); !r) {
      std::println(stderr, "queued work did not finish: {}", r.error());
    }
  }

 private:
  Device& d_;
};

// A captured denoising step, destroyed with its graph.
class StepGraph {
 public:
  StepGraph() = default;
  StepGraph(const StepGraph&) = delete;
  StepGraph& operator=(const StepGraph&) = delete;
  StepGraph(StepGraph&&) = delete;
  StepGraph& operator=(StepGraph&&) = delete;
  ~StepGraph() { Reset(); }
  void Reset() {
    if (exec_ != nullptr) {
      (void)cudaGraphExecDestroy(exec_);
      exec_ = nullptr;
    }
    if (graph_ != nullptr) {
      (void)cudaGraphDestroy(graph_);
      graph_ = nullptr;
    }
  }
  // Captures `queue` on `s` (thread-local capture: nothing else on this
  // thread may use CUDA meanwhile) and instantiates it.
  template <typename F>
  Status Capture(cudaStream_t s, F&& queue) {
    Reset();
    if (auto r = Cuda(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal), "capture"); !r) {
      return r;
    }
    const Status queued = queue();
    const cudaError_t ended = cudaStreamEndCapture(s, &graph_);
    if (!queued) {
      return queued;
    }
    if (auto r = Cuda(ended, "end capture"); !r) {
      return r;
    }
    return Cuda(cudaGraphInstantiate(&exec_, graph_, 0), "graph instantiate");
  }
  Status Launch(cudaStream_t s) const { return Cuda(cudaGraphLaunch(exec_, s), "graph launch"); }
  bool ready() const { return exec_ != nullptr; }

 private:
  cudaGraph_t graph_ = nullptr;
  cudaGraphExec_t exec_ = nullptr;
};

// ------------------------------------------------------------------ weights

struct Loaded {
  DeviceBuffer region;
  std::vector<std::uint64_t> group_address;         // 0: not loaded
  std::vector<std::vector<std::byte>> host_groups;  // host loads (the VAE)
  std::vector<std::uint64_t> tensor_address;        // by the component's tensor list
  std::uint64_t bytes_read = 0;
  double seconds = 0;
};

// Reads the groups holding `resources`, into one device region (or, with
// `host`, into host memory), with direct reads through pinned staging.
Status LoadGroups(const ja::Artifact& artifact, std::span<const std::uint32_t> resources, bool host,
                  Device& device, Loaded& w) {
  const auto start = Clock::now();
  const auto groups = artifact.groups();
  std::vector<bool> want(groups.size(), false);
  for (const std::uint32_t r : resources) {
    want[artifact.resources()[r].group] = true;
  }
  w.group_address.assign(groups.size(), 0);
  w.host_groups.assign(groups.size(), {});
  std::uint64_t total = 0;
  std::vector<std::uint64_t> offset(groups.size(), 0);
  for (std::size_t g = 0; g < groups.size(); ++g) {
    if (want[g]) {
      offset[g] = total;
      total += Round(groups[g].stored.value(), 256);
      if (host) {
        w.host_groups[g].assign(groups[g].stored.value(), std::byte{0});
      }
    }
  }
  if (!host) {
    if (auto r = w.region.Allocate(total, "a component's weights"); !r) {
      return r;
    }
    for (std::size_t g = 0; g < groups.size(); ++g) {
      if (want[g]) {
        w.group_address[g] = w.region.address() + offset[g];
      }
    }
  }
  std::vector<ja::ChunkKey> chunks;
  for (std::uint32_t g = 0; g < groups.size(); ++g) {
    for (std::uint32_t c = 0; want[g] && c < groups[g].chunks; ++c) {
      chunks.push_back({.group = g, .chunk = c});
    }
  }
  const ja::ReadLimits limits{};
  const auto runs = artifact.PlanReads(chunks, {}, limits);
  if (!runs) {
    return Error("the artifact's read plan was refused");
  }
  const std::uint64_t staging_bytes = limits.max_run.value();
  std::array<void*, 2> staging{};
  for (void*& s : staging) {
    if (auto r = Cuda(cudaMallocHost(&s, staging_bytes), "pinned staging"); !r) {
      return r;
    }
  }
  std::vector<ja::FileDescriptor> shards;
  for (std::uint32_t s = 0; s < artifact.shards().size(); ++s) {
    auto fd = artifact.OpenShardForDirectRead(s);
    if (!fd) {
      return Error(std::format("shard {} cannot be opened for direct reads", s));
    }
    shards.push_back(std::move(*fd));
  }
  std::array<cudaEvent_t, 2> copied{};
  for (cudaEvent_t& e : copied) {
    if (auto r = Cuda(cudaEventCreateWithFlags(&e, cudaEventDisableTiming), "an event"); !r) {
      return r;
    }
  }
  std::array<bool, 2> pending = {false, false};
  std::size_t which = 0;
  Status status;
  for (const auto& run : *runs) {
    if (pending[which]) {
      if (auto r = Cuda(cudaEventSynchronize(copied[which]), "a staging copy"); !r) {
        status = r;
        break;
      }
    }
    auto* bytes = static_cast<std::byte*>(staging[which]);
    std::uint64_t done = 0;
    while (done < run.length.value()) {
      const ssize_t got = pread(shards[run.shard].get(), bytes + done, run.length.value() - done,
                                static_cast<off_t>(run.file_offset.value() + done));
      if (got <= 0) {
        return Error(std::format("a direct read of shard {} failed", run.shard));
      }
      done += static_cast<std::uint64_t>(got);
    }
    w.bytes_read += done;
    std::uint64_t at = 0;
    for (const auto& segment : run.segments) {
      const std::uint64_t n = segment.length.value();
      if (host) {
        std::memcpy(w.host_groups[segment.chunk.group].data() + segment.group_offset.value(),
                    bytes + at, n);
      } else {
        const std::uint64_t dst =
            w.group_address[segment.chunk.group] + segment.group_offset.value();
        if (auto r = Cuda(cudaMemcpyAsync(At<void>(dst), bytes + at, n, cudaMemcpyHostToDevice,
                                          device.native()),
                          "a weight upload");
            !r) {
          return r;
        }
      }
      at += n;
    }
    if (!host) {
      if (auto r = Cuda(cudaEventRecord(copied[which], device.native()), "an event record"); !r) {
        return r;
      }
      pending[which] = true;
    }
    which ^= 1U;
  }
  if (auto r = device.Finish(); !r) {
    return r;
  }
  for (cudaEvent_t e : copied) {
    (void)cudaEventDestroy(e);
  }
  for (void* s : staging) {
    (void)cudaFreeHost(s);
  }
  if (!status) {
    return status;
  }
  w.seconds = Seconds(Clock::now() - start);
  return {};
}

// A BF16 component: device addresses of its bound tensors.
Status LoadBf16(const ja::Artifact& a, std::span<const std::uint32_t> bound, Device& d, Loaded& w) {
  if (auto r = LoadGroups(a, bound, false, d, w); !r) {
    return r;
  }
  w.tensor_address.clear();
  for (const std::uint32_t r : bound) {
    const auto& res = a.resources()[r];
    w.tensor_address.push_back(w.group_address[res.group] + res.offset.value());
  }
  return {};
}

// The VAE: its F32 tensors read to the host, rounded to BF16 (as
// from_pretrained(torch_dtype=bfloat16) casts them) into a device region,
// in the layout the plan's convolution reads (QwenImagePipeline::
// VaeWeightBf16).
Status LoadVae(const ja::Artifact& a, std::span<const std::uint32_t> bound,
               const ki::QwenImagePipeline& pipeline, Device& d, Loaded& w) {
  if (auto r = LoadGroups(a, bound, true, d, w); !r) {
    return r;
  }
  const auto start = Clock::now();
  std::uint64_t total = 0;
  std::vector<std::uint64_t> offsets;
  for (const std::uint32_t r : bound) {
    offsets.push_back(total);
    total += Round(a.resources()[r].bytes.value() / 2, 256);
  }
  if (auto r = w.region.Allocate(total, "the VAE's BF16 weights"); !r) {
    return r;
  }
  std::vector<Bf16> host(total / 2);
  w.tensor_address.clear();
  for (std::size_t i = 0; i < bound.size(); ++i) {
    const auto& res = a.resources()[bound[i]];
    const std::span<const std::byte> src(w.host_groups[res.group].data() + res.offset.value(),
                                         res.bytes.value());
    if (auto r = pipeline.VaeWeightBf16(
            i, src, std::span(host).subspan(offsets[i] / 2, res.bytes.value() / 4));
        !r) {
      return r;
    }
    w.tensor_address.push_back(w.region.address() + offsets[i]);
  }
  if (auto r = Cuda(cudaMemcpy(w.region.as<void>(), host.data(), total, cudaMemcpyHostToDevice),
                    "the VAE's weights");
      !r) {
    return r;
  }
  w.host_groups.clear();
  w.seconds += Seconds(Clock::now() - start);
  return {};
}

// ------------------------------------------------------------------ comparison

struct Compared {
  double rel_rms = 0;
  double cosine = 0;
  double max_abs = 0;
};

Compared CompareBf16(std::span<const Bf16> a, std::span<const Bf16> b) {
  double dd = 0;
  double bb = 0;
  double aa = 0;
  double ab = 0;
  double mx = 0;
  for (std::size_t i = 0; i < a.size() && i < b.size(); ++i) {
    const double x = md::FromBf16(a[i]);
    const double y = md::FromBf16(b[i]);
    dd += (x - y) * (x - y);
    bb += y * y;
    aa += x * x;
    ab += x * y;
    mx = std::max(mx, std::abs(x - y));
  }
  return {.rel_rms = std::sqrt(dd / std::max(bb, 1e-300)),
          .cosine = ab / std::sqrt(std::max(aa * bb, 1e-300)),
          .max_abs = mx};
}

std::string Json(const Compared& c) {
  return std::format(R"({{"rel_rms": {:.6g}, "cosine": {:.9f}, "max_abs": {:.6g}}})", c.rel_rms,
                     c.cosine, c.max_abs);
}

std::expected<std::vector<Bf16>, std::string> ReadBf16(const std::filesystem::path& p,
                                                       std::size_t count) {
  std::ifstream in(p, std::ios::binary);
  if (!in) {
    return Error(std::format("{}: cannot open", p.string()));
  }
  std::vector<Bf16> out(count);
  in.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(count * 2));
  if (static_cast<std::size_t>(in.gcount()) != count * 2 || in.peek() != EOF) {
    return Error(std::format("{}: not {} BF16 values", p.string(), count));
  }
  return out;
}

Status WriteBytes(const std::filesystem::path& p, const void* data, std::size_t bytes) {
  std::ofstream out(p, std::ios::binary | std::ios::trunc);
  out.write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes));
  return out ? Status{} : Error(std::format("{}: write failed", p.string()));
}

template <typename T>
std::expected<std::vector<T>, std::string> Download(std::uint64_t address, std::size_t count) {
  std::vector<T> out(count);
  if (auto r =
          Cuda(cudaMemcpy(out.data(), At<void>(address), count * sizeof(T), cudaMemcpyDeviceToHost),
               "a download");
      !r) {
    return std::unexpected(r.error());
  }
  return out;
}

// ------------------------------------------------------------------ main

struct Options {
  std::filesystem::path store;
  std::string composition;
  std::filesystem::path out;
  std::string prompt = "A red ceramic teapot on a plain wooden table, soft daylight, no text.";
  std::uint32_t size = 1024;
  std::uint32_t steps = 40;
  std::optional<std::filesystem::path> reference;
  bool reference_embeds = false;
  std::optional<std::filesystem::path> noise;
  bool force_latents = false;
  bool vae = true;
  bool released = false;
  std::uint32_t runs = 0;
  std::uint32_t stop_after = 0;  // 0: every step
  bool decode_reference = false;
  ki::PlanKind plan = ki::PlanKind::kFast;
  std::vector<std::string> choose;
  bool graphs = true;
};

std::expected<Options, std::string> Parse(std::span<char*> args) {
  Options o;
  for (std::size_t i = 1; i < args.size(); ++i) {
    const std::string_view a = args[i];
    const auto value = [&]() -> std::expected<std::string_view, std::string> {
      if (i + 1 >= args.size()) {
        return Error(std::format("{} needs a value", a));
      }
      return std::string_view(args[++i]);
    };
    const auto number = [&]() -> std::expected<std::uint32_t, std::string> {
      auto v = value();
      if (!v) {
        return std::unexpected(v.error());
      }
      std::uint32_t n = 0;
      for (const char c : *v) {
        if (c < '0' || c > '9' || n > 100000) {
          return Error(std::format("{}: not a number", a));
        }
        n = (n * 10) + static_cast<std::uint32_t>(c - '0');
      }
      return n;
    };
    if (a == "--store" || a == "--composition" || a == "--out" || a == "--prompt" ||
        a == "--reference" || a == "--embeds" || a == "--noise" || a == "--phases" ||
        a == "--plan" || a == "--choose") {
      auto v = value();
      if (!v) {
        return std::unexpected(v.error());
      }
      if (a == "--store") o.store = *v;
      if (a == "--composition") o.composition = *v;
      if (a == "--out") o.out = *v;
      if (a == "--prompt") o.prompt = *v;
      if (a == "--reference") o.reference = std::filesystem::path(*v);
      if (a == "--noise") o.noise = std::filesystem::path(*v);
      if (a == "--choose") o.choose.emplace_back(*v);
      if (a == "--embeds") {
        if (*v != "native" && *v != "reference") return Error("--embeds native|reference");
        o.reference_embeds = *v == "reference";
      }
      if (a == "--phases") {
        if (*v != "resident" && *v != "released") return Error("--phases resident|released");
        o.released = *v == "released";
      }
      if (a == "--plan") {
        if (*v != "fast" && *v != "legacy") return Error("--plan fast|legacy");
        o.plan = *v == "fast" ? ki::PlanKind::kFast : ki::PlanKind::kLegacy;
      }
    } else if (a == "--size" || a == "--steps" || a == "--runs" || a == "--stop-after") {
      auto n = number();
      if (!n) {
        return std::unexpected(n.error());
      }
      if (a == "--size") o.size = *n;
      if (a == "--steps") o.steps = *n;
      if (a == "--runs") o.runs = *n;
      if (a == "--stop-after") o.stop_after = *n;
    } else if (a == "--force-latents") {
      o.force_latents = true;
    } else if (a == "--no-vae") {
      o.vae = false;
    } else if (a == "--graphs") {
      auto v = value();
      if (!v || (*v != "on" && *v != "off")) {
        return Error("--graphs on|off");
      }
      o.graphs = *v == "on";
    } else if (a == "--decode-reference") {
      o.decode_reference = true;
    } else {
      return Error(std::format("unknown argument {}", a));
    }
  }
  if (o.store.empty() || o.composition.empty() || o.out.empty()) {
    return Error("--store, --composition and --out are required");
  }
  if (o.size % 32 != 0 || o.size < 64 || o.size > 2048 || o.steps < 2 || o.steps > 100) {
    return Error("--size a multiple of 32 in 64-2048, --steps 2-100");
  }
  if ((o.reference_embeds || o.force_latents || o.decode_reference) && !o.reference) {
    return Error("--embeds reference, --force-latents and --decode-reference need --reference");
  }
  // A replayed step reads the device's own latents; forced ones are
  // uploaded per step, which a graph would not see change.
  o.graphs = o.graphs && !o.force_latents;
  return o;
}

struct Component {
  std::optional<ja::Artifact> artifact;
  std::vector<std::uint32_t> bound;
  Loaded weights;
};

Status Run(const Options& o) {
  std::filesystem::create_directories(o.out);
  const md::QwenImageProfile& profile = md::QwenImage21();
  std::ostringstream report;
  report << "{\n";
  MemorySampler memory;
  const std::uint64_t baseline = MemAvailable();

  // The plan, bound against this build's registry (D-053).
  auto registry = llmp::execution::Registry::Create(ki::Implementations());
  if (!registry) {
    return Error("registry: " + registry.error().detail);
  }
  auto choices = ki::QwenImageChoices(o.plan, o.choose);
  if (!choices) {
    return Error("plan: " + choices.error());
  }
  auto plan = llmp::execution::Plan::Build(*registry, *choices);
  if (!plan) {
    return Error("plan: " + plan.error().detail);
  }
  auto pipeline = ki::QwenImagePipeline::Bind(*registry, *plan, profile);
  if (!pipeline) {
    return Error("plan: " + pipeline.error());
  }
  const ki::QwenImagePipeline& pipe = **pipeline;
  report << std::format("  \"plan\": {{\"identity\": \"{}\", \"roles\": {}}},\n",
                        llmp::base::ToHex(pipe.identity()), pipe.Describe());

  // The composition and its components, as untrusted input.
  auto composition = ja::OpenComposition(o.store / o.composition);
  if (!composition) {
    return Error("composition: " + composition.error().ToString());
  }
  if (composition->architecture() != profile.pipeline_architecture) {
    return Error("the composition is not a " + std::string(profile.pipeline_architecture));
  }
  std::map<std::string, Component> components;
  const std::array<std::tuple<std::string_view, std::string_view, std::vector<md::QwenImageTensor>>,
                   3>
      roles = {
          {{"text_encoder", profile.text_architecture, md::TextEncoderTensors(profile.text)},
           {"transformer", profile.denoiser_architecture, md::DenoiserTensors(profile.denoiser)},
           {"vae", profile.vae_architecture, md::VaeDecoderTensors(profile.vae)}}};
  for (const auto& [role, arch, tensors] : roles) {
    const ja::CompositionComponent* c = composition->Find(role);
    if (c == nullptr) {
      return Error(std::format("the composition has no {}", role));
    }
    ja::OpenOptions options;
    options.expected_id = c->artifact;
    auto a = ja::Artifact::Open(o.store / c->artifact, options);
    if (!a) {
      return Error(std::format("{}: {}", role, a.error().ToString()));
    }
    if (a->model().architecture != c->architecture) {
      return Error(
          std::format("{}: the artifact's architecture differs from the composition's", role));
    }
    auto bound = md::BindQwenImageComponent(tensors, arch, *a);
    if (!bound) {
      return Error(std::format("{}: {}", role, bound.error()));
    }
    Component& comp = components[std::string(role)];
    comp.artifact.emplace(std::move(*a));
    comp.bound = std::move(*bound);
  }

  // The prompt's tokens (the native tokenizer and renderer).
  const auto tokenizer_json = composition->Metadata("tokenizer.json");
  if (!tokenizer_json) {
    return Error("the composition keeps no tokenizer.json");
  }
  auto spec = llmp::tokenizer::ReadHfTokenizer(*tokenizer_json);
  if (!spec) {
    return Error("tokenizer: " + spec.error().ToString());
  }
  auto tokenizer = llmp::tokenizer::Tokenizer::Create(std::move(*spec));
  if (!tokenizer) {
    return Error("tokenizer: " + tokenizer.error().ToString());
  }
  auto rendered = llmp::chat::RenderQwenImagePrompt(o.prompt, *tokenizer);
  if (!rendered) {
    return Error("prompt: " + rendered.error().ToString());
  }
  std::vector<llmp::tokenizer::TokenId> ids;
  if (auto e =
          tokenizer->EncodeMarked(rendered->rendered.text, rendered->rendered.specials, {}, ids);
      !e) {
    return Error("prompt: " + e.error().ToString());
  }
  const std::size_t drop = rendered->drop_tokens;
  if (ids.size() <= drop || ids.size() > 1024) {
    return Error("a prompt of 1 to 1,024 tokens after the system turn");
  }
  const auto token_rows = static_cast<std::int64_t>(ids.size());
  const auto text_rows = static_cast<std::int64_t>(ids.size() - drop);
  report << std::format("  \"tokens\": {}, \"drop_tokens\": {}, \"text_rows\": {},\n", ids.size(),
                        drop, text_rows);
  if (o.reference) {
    std::ifstream in(*o.reference / "input_ids.i64", std::ios::binary);
    std::vector<std::int64_t> ref(ids.size() + 1);
    in.read(reinterpret_cast<char*>(ref.data()), static_cast<std::streamsize>(ref.size() * 8));
    const bool same = static_cast<std::size_t>(in.gcount()) == ids.size() * 8 &&
                      std::equal(ids.begin(), ids.end(), ref.begin());
    report << std::format("  \"tokens_match_reference\": {},\n", same);
    if (!same) {
      return Error("the native prompt tokens differ from the reference's");
    }
  }

  auto device = Device::Open();
  if (!device) {
    return std::unexpected(device.error());
  }
  Device& d = **device;
  // cuBLAS and cuBLASLt, sharing the workspace upstream sizes (one stream).
  DeviceBuffer blas_ws;
  cudaDeviceProp prop{};
  (void)cudaGetDeviceProperties(&prop, 0);
  const Bytes blas_bytes =
      kg::CublasHandle::UpstreamWorkspace((prop.major * 100) + (prop.minor * 10));
  if (auto r = blas_ws.Allocate(blas_bytes.value(), "cuBLAS workspace"); !r) {
    return r;
  }
  auto blas = kg::CublasHandle::Create(0, d.execution(), d.stream(),
                                       {.base = blas_ws.address(), .size = blas_bytes});
  if (!blas) {
    return Error("cuBLAS: " + blas.error().detail);
  }
  auto lt = ki::LtGemm::Create(blas_ws.address(), blas_bytes.value());
  if (!lt) {
    return Error("cuBLASLt: " + lt.error());
  }
  const ki::Handles handles{.blas = (*blas)->native(), .lt = lt->get()};

  const std::uint32_t grid = o.size / profile.vae.spatial_factor;
  const std::int64_t image = std::int64_t{grid} * grid;
  const std::int64_t latent_elems = image * profile.denoiser.in_channels;
  auto schedule =
      md::QwenImageSchedulerSigmas(profile.scheduler, o.steps, static_cast<std::uint32_t>(image));
  if (!schedule) {
    return Error(schedule.error());
  }
  report << std::format(R"(  "mu": {:.17g}, "sigmas": [)", schedule->mu);
  for (std::size_t i = 0; i < schedule->sigmas.size(); ++i) {
    report << std::format("{}{:.9g}", i ? ", " : "", schedule->sigmas[i]);
  }
  report << "],\n";

  // Reference tensors.
  std::vector<Bf16> ref_embeds;
  std::vector<Bf16> noise0;
  if (o.reference) {
    auto e = ReadBf16(*o.reference / "prompt_embeds.bf16",
                      static_cast<std::size_t>(text_rows * profile.denoiser.context));
    if (!e) {
      return std::unexpected(e.error());
    }
    ref_embeds = std::move(*e);
  }
  {
    std::filesystem::path path;
    if (o.noise) {
      path = *o.noise;
    } else if (o.reference) {
      path = *o.reference / "latents_init.bf16";
    }
    if (path.empty()) {
      return Error(
          "the initial latents: --noise or --reference (llmpalooza does not reproduce "
          "PyTorch's CUDA generator)");
    }
    auto n = ReadBf16(path, static_cast<std::size_t>(latent_elems));
    if (!n) {
      return std::unexpected(n.error());
    }
    noise0 = std::move(*n);
  }

  auto load = [&](std::string_view role) -> Status {
    Component& c = components[std::string(role)];
    if (c.weights.region.address() != 0) {
      return {};
    }
    return role == "vae" ? LoadVae(*c.artifact, c.bound, pipe, d, c.weights)
                         : LoadBf16(*c.artifact, c.bound, d, c.weights);
  };
  // Completion-aware: the phase's queued work finishes before its weights
  // are freed (a free is not proof that the kernels reading them are done).
  auto release = [&](std::string_view role) -> Status {
    if (auto r = d.Finish(); !r) {
      return r;
    }
    Component& c = components[std::string(role)];
    c.weights.region.Free();
    c.weights.tensor_address.clear();
    c.weights.group_address.clear();
    return {};
  };
  std::vector<std::string> load_lines;
  if (!o.released) {
    for (const char* role : {"text_encoder", "transformer", "vae"}) {
      if (auto r = load(role); !r) {
        return r;
      }
      const Loaded& w = components[role].weights;
      load_lines.push_back(
          std::format(R"("{}": {{"bytes_read": {}, "seconds": {:.3f}, "device_bytes": {}}})", role,
                      w.bytes_read, w.seconds, w.region.bytes()));
    }
  }

  // Working memory: the image's own (lives from step to step), and one
  // workspace the phases share in turn (the paged runner's arrangement);
  // with --phases released, each phase's own workspace, allocated at its
  // start and freed at its end as its weights are.
  std::uint64_t own_bytes = 0;
  (void)ki::OwnLayout(0, profile.denoiser, text_rows, image, own_bytes);
  std::uint64_t text_bytes = 0;
  std::uint64_t dit_bytes = 0;
  std::uint64_t vae_bytes = 0;
  std::vector<std::uint64_t> no_weights;
  (void)ki::TextLayout(0, profile.text, token_rows, text_bytes);
  (void)ki::DitLayout(0, profile.denoiser, text_rows + image, dit_bytes);
  (void)ki::VaeLayout(0, profile.vae, grid, {}, false, no_weights, vae_bytes);
  DeviceBuffer own;
  DeviceBuffer work;
  const FinishFirst finish(d);  // before these are freed
  if (auto r = own.Allocate(own_bytes, "the image's own memory"); !r) {
    return r;
  }
  if (!o.released) {
    if (auto r = work.Allocate(std::max({text_bytes, dit_bytes, vae_bytes}), "the workspace"); !r) {
      return r;
    }
  }
  // A phase's workspace when released (queued work finished first: a free
  // is not a completion), and its end.
  const auto phase_work = [&](std::uint64_t bytes) -> Status {
    if (!o.released) {
      return {};
    }
    if (auto f = d.Finish(); !f) {
      return f;
    }
    return work.Allocate(bytes, "a phase's workspace");
  };
  const auto phase_done = [&]() -> Status {
    if (!o.released) {
      return {};
    }
    if (auto f = d.Finish(); !f) {
      return f;
    }
    work.Free();
    return {};
  };
  report << std::format(
      "  \"memory\": {{\"own\": {}, \"workspace\": {}, \"text\": {}, \"dit\": {}, \"vae\": {}}},\n",
      own_bytes, std::max({text_bytes, dit_bytes, vae_bytes}), text_bytes, dit_bytes, vae_bytes);
  std::uint64_t unused = 0;
  const ki::DitMemory own_layout =
      ki::OwnLayout(own.address(), profile.denoiser, text_rows, image, unused);
  auto* const s = d.native();

  // One full generation; `record` keeps every tensor and the per-step
  // times (synchronized each step).
  struct Timing {
    double encode = 0, denoise = 0, decode = 0, total = 0, first_step = 0;
    std::string pixels_sha256;
    std::vector<double> steps;
    std::vector<std::string> loads;
  };
  std::vector<Compared> step_cmp;
  std::optional<Compared> embeds_cmp;
  std::optional<Compared> final_cmp;
  std::optional<Compared> decoded_cmp;
  StepGraph graph;
  const auto rot = md::QwenImageTextRotary(profile.text, static_cast<std::uint32_t>(token_rows));
  const auto freqs =
      md::QwenImageDenoiserRotary(profile.denoiser, static_cast<std::uint32_t>(text_rows), grid);
  std::vector<std::int32_t> ids32(ids.begin(), ids.end());

  auto generate = [&](bool record, Timing& t) -> Status {
    const auto start = Clock::now();
    // --- encode
    if (o.released) {
      const auto l0 = Clock::now();
      if (auto r = load("text_encoder"); !r) return r;
      t.loads.push_back(std::format(R"("text_encoder": {:.3f})", Seconds(Clock::now() - l0)));
    }
    if (o.reference_embeds) {
      if (auto r = Cuda(cudaMemcpy(At<void>(own_layout.embeds), ref_embeds.data(),
                                   ref_embeds.size() * 2, cudaMemcpyHostToDevice),
                        "embeddings");
          !r)
        return r;
    } else {
      if (auto w = phase_work(text_bytes); !w) return w;
      std::uint64_t bytes = 0;
      ki::TextMemory m = ki::TextLayout(work.address(), profile.text, token_rows, bytes);
      m.tensors = components["text_encoder"].weights.tensor_address;
      m.embeds = own_layout.embeds;
      m.drop = static_cast<std::int64_t>(drop);
      Status r = Cuda(cudaMemcpyAsync(At<void>(m.ids), ids32.data(), ids32.size() * 4,
                                      cudaMemcpyHostToDevice, s),
                      "ids");
      r = r ? Cuda(cudaMemcpyAsync(At<void>(m.cos), rot.cos.data(), rot.cos.size() * 2,
                                   cudaMemcpyHostToDevice, s),
                   "cos")
            : r;
      r = r ? Cuda(cudaMemcpyAsync(At<void>(m.sin), rot.sin.data(), rot.sin.size() * 2,
                                   cudaMemcpyHostToDevice, s),
                   "sin")
            : r;
      r = r ? Cuda(cudaMemsetAsync(At<void>(m.bad), 0, 4, s), "a flag") : r;
      r = r ? pipe.Encode(m, handles, s) : r;
      r = r ? d.Finish() : r;
      if (!r) return r;
      std::int32_t flag = 0;
      if (auto c = Cuda(cudaMemcpy(&flag, At<void>(m.bad), 4, cudaMemcpyDeviceToHost), "flag"); !c)
        return c;
      if (flag != 0) return Error("a token id outside the embedding table");
    }
    if (auto r = d.Finish(); !r) return r;
    if (o.released) {
      if (auto r = release("text_encoder"); !r) return r;
    }
    if (auto r = phase_done(); !r) return r;
    const auto encoded = Clock::now();
    t.encode = Seconds(encoded - start);
    if (record) {
      auto got = Download<Bf16>(own_layout.embeds,
                                static_cast<std::size_t>(text_rows * profile.denoiser.context));
      if (!got) return std::unexpected(got.error());
      if (auto w = WriteBytes(o.out / "prompt_embeds.bf16", got->data(), got->size() * 2); !w)
        return w;
      if (!ref_embeds.empty()) embeds_cmp = CompareBf16(*got, ref_embeds);
    }
    // --- denoise
    if (o.released) {
      const auto l0 = Clock::now();
      if (auto r = load("transformer"); !r) return r;
      t.loads.push_back(std::format(R"("transformer": {:.3f})", Seconds(Clock::now() - l0)));
    }
    const auto denoise_start = Clock::now();
    if (o.released) {
      graph.Reset();  // the weights and workspace moved: a graph holds their addresses
    }
    if (auto w = phase_work(dit_bytes); !w) return w;
    std::uint64_t bytes = 0;
    ki::DitMemory m = ki::DitLayout(work.address(), profile.denoiser, text_rows + image, bytes);
    m.tensors = components["transformer"].weights.tensor_address;
    m.embeds = own_layout.embeds;
    m.txt = own_layout.txt;
    m.pk = own_layout.pk;
    m.pv = own_layout.pv;
    m.freqs = own_layout.freqs;
    m.latents = own_layout.latents;
    m.noise = own_layout.noise;
    m.text = text_rows;
    m.image = image;
    Status r = Cuda(cudaMemcpyAsync(At<void>(m.freqs), freqs.data(), freqs.size() * 4,
                                    cudaMemcpyHostToDevice, s),
                    "rotary frequencies");
    r = r ? pipe.TextRows(m, handles, s) : r;
    r = r ? Cuda(cudaMemcpyAsync(At<void>(m.latents), noise0.data(), noise0.size() * 2,
                                 cudaMemcpyHostToDevice, s),
                 "latents")
          : r;
    if (!r) return r;
    auto step_start = Clock::now();
    const std::uint32_t last = o.stop_after != 0 ? std::min(o.stop_after, o.steps) : o.steps;
    for (std::uint32_t i = 0; i < last; ++i) {
      if (o.force_latents) {
        auto in = ReadBf16(*o.reference / std::format("step{:02d}_latents_in.bf16", i),
                           static_cast<std::size_t>(latent_elems));
        if (!in) return std::unexpected(in.error());
        if (auto c = Cuda(cudaMemcpyAsync(At<void>(m.latents), in->data(), in->size() * 2,
                                          cudaMemcpyHostToDevice, s),
                          "forced latents");
            !c)
          return c;
        if (auto f = d.Finish(); !f) return f;
      }
      // The step's inputs: the sinusoid of its timestep and dt. dt is a
      // 0-dim F32 tensor times the BF16 noise: PyTorch rounds it to BF16
      // first (checked against the reference's steps: 0 of 10.2M values
      // differ this way, 287,316 with an F32 dt).
      const auto sinus = md::QwenImageTimestepSinusoid(profile.denoiser, schedule->timesteps[i]);
      const float dt = ki::EulerStepDt(schedule->sigmas[i + 1] - schedule->sigmas[i], true);
      Status u = Cuda(cudaMemcpyAsync(At<void>(m.sin), sinus.data(), sinus.size() * 2,
                                      cudaMemcpyHostToDevice, s),
                      "the sinusoid");
      u = u ? Cuda(cudaMemcpyAsync(At<void>(m.dt), &dt, 4, cudaMemcpyHostToDevice, s), "dt") : u;
      if (!u) return u;
      // With --graphs, steps from the third replay one captured step (the
      // second has made every product's descriptors).
      if (o.graphs && i >= 2) {
        if (!graph.ready()) {
          if (auto c = graph.Capture(s, [&] { return pipe.Step(false, m, handles, s); }); !c)
            return c;
        }
        if (auto c = graph.Launch(s); !c) return c;
      } else {
        if (auto c = pipe.Step(i == 0, m, handles, s); !c) return c;
      }
      // (cudaMemcpyAsync stages pageable host memory before it returns, so
      // the sinusoid and dt may go out of scope.)
      if (record) {
        if (auto f = d.Finish(); !f) return f;
        const auto now = Clock::now();
        t.steps.push_back(Seconds(now - step_start));
        step_start = now;
        auto got = Download<Bf16>(m.noise, static_cast<std::size_t>(latent_elems));
        if (!got) return std::unexpected(got.error());
        if (auto w = WriteBytes(o.out / std::format("step{:02d}_noise_pred.bf16", i), got->data(),
                                got->size() * 2);
            !w)
          return w;
        if (o.reference) {
          auto ref = ReadBf16(*o.reference / std::format("step{:02d}_noise_pred.bf16", i),
                              static_cast<std::size_t>(latent_elems));
          if (ref) step_cmp.push_back(CompareBf16(*got, *ref));
        }
      }
    }
    if (auto f = d.Finish(); !f) return f;
    const auto denoised = Clock::now();
    t.denoise = Seconds(denoised - denoise_start);
    t.first_step = t.steps.empty() ? 0 : t.steps[0];
    if (o.released) {
      if (auto f = release("transformer"); !f) return f;
    }
    if (auto f = phase_done(); !f) return f;
    if (record) {
      auto got = Download<Bf16>(m.latents, static_cast<std::size_t>(latent_elems));
      if (!got) return std::unexpected(got.error());
      if (auto w = WriteBytes(o.out / "latents_final.bf16", got->data(), got->size() * 2); !w)
        return w;
      if (o.reference) {
        auto ref =
            ReadBf16(*o.reference / "latents_final.bf16", static_cast<std::size_t>(latent_elems));
        if (ref) final_cmp = CompareBf16(*got, *ref);
      }
    }
    // --- decode
    if (o.vae) {
      if (o.released) {
        const auto l0 = Clock::now();
        if (auto f = load("vae"); !f) return f;
        t.loads.push_back(std::format(R"("vae": {:.3f})", Seconds(Clock::now() - l0)));
      }
      if (o.decode_reference) {
        // The VAE alone: diffusers' final latents.
        auto ref =
            ReadBf16(*o.reference / "latents_final.bf16", static_cast<std::size_t>(latent_elems));
        if (!ref) return std::unexpected(ref.error());
        if (auto c = Cuda(cudaMemcpy(At<void>(m.latents), ref->data(), ref->size() * 2,
                                     cudaMemcpyHostToDevice),
                          "reference latents");
            !c)
          return c;
      }
      if (auto w = phase_work(vae_bytes); !w) return w;
      const auto decode_start = Clock::now();
      std::vector<std::uint64_t> weights;
      std::uint64_t vbytes = 0;
      ki::VaeMemory v =
          ki::VaeLayout(work.address(), profile.vae, grid, {}, false, weights, vbytes);
      v.weights = components["vae"].weights.tensor_address;
      v.latents = m.latents;
      std::vector<Bf16> stats(std::size_t{2} * 64);
      for (std::size_t c = 0; c < 64; ++c) {
        stats[c] = md::ToBf16(profile.vae.latents_std.at(c));
        stats[64 + c] = md::ToBf16(profile.vae.latents_mean.at(c));
      }
      Status dr = Cuda(cudaMemcpyAsync(At<void>(v.stats), stats.data(), stats.size() * 2,
                                       cudaMemcpyHostToDevice, s),
                       "latent statistics");
      dr = dr ? pipe.Decode(v, handles, s) : dr;
      dr = dr ? d.Finish() : dr;
      if (!dr) return dr;
      t.decode = Seconds(Clock::now() - decode_start);
      if (o.released) {
        if (auto f = release("vae"); !f) return f;
      }
      const std::size_t count = std::size_t{profile.vae.out_channels} * o.size * o.size;
      auto got = Download<Bf16>(v.buf.at(md::kVaeX), count);
      if (!got) return std::unexpected(got.error());
      if (record) {
        if (auto w = WriteBytes(o.out / "vae_decoded.bf16", got->data(), got->size() * 2); !w)
          return w;
        if (o.reference) {
          auto ref = ReadBf16(*o.reference / "vae_decoded.bf16", count);
          if (ref) decoded_cmp = CompareBf16(*got, *ref);
        }
      }
      const auto pixels = md::QwenImagePixels(*got, profile.vae.out_channels, o.size, o.size);
      t.pixels_sha256 =
          llmp::base::ToHex(llmp::base::Sha256().Update(std::as_bytes(std::span(pixels))).Finish());
      if (record) {
        if (auto w = WriteBytes(o.out / "image.rgba8", pixels.data(), pixels.size()); !w) return w;
      }
      if (auto f = phase_done(); !f) return f;
    }
    t.total = Seconds(Clock::now() - start);
    return {};
  };

  memory.Reset();
  Timing first;
  if (auto r = generate(true, first); !r) {
    return r;
  }
  const std::uint64_t low_first = memory.low();
  std::vector<double> plain;
  std::vector<std::string> plain_steps;
  bool same_pixels = true;
  for (std::uint32_t i = 0; i < o.runs; ++i) {
    Timing t;
    if (auto r = generate(false, t); !r) {
      return r;
    }
    plain.push_back(t.total);
    plain_steps.push_back(
        std::format(R"({{"encode_s": {:.4f}, "denoise_s": {:.4f}, "decode_s": {:.4f}}})", t.encode,
                    t.denoise, t.decode));
    same_pixels = same_pixels && t.pixels_sha256 == first.pixels_sha256;
  }
  report << std::format("  \"pixels_sha256\": \"{}\", \"plain_runs_same_pixels\": {},\n",
                        first.pixels_sha256, same_pixels);
  // The report.
  report << std::format("  \"composition\": \"{}\",\n", composition->id());
  report << "  \"loads\": {";
  for (std::size_t i = 0; i < load_lines.size(); ++i) {
    report << (i ? ", " : "") << load_lines[i];
  }
  report << "},\n  \"released_loads_s\": {";
  for (std::size_t i = 0; i < first.loads.size(); ++i) {
    report << (i ? ", " : "") << first.loads[i];
  }
  report << "},\n";
  std::vector<double> gaps(first.steps.begin() + (first.steps.size() > 1 ? 1 : 0),
                           first.steps.end());
  std::ranges::sort(gaps);
  report << std::format(
      "  \"first_run\": {{\"encode_s\": {:.4f}, \"denoise_s\": {:.4f}, \"decode_s\": {:.4f}, "
      "\"total_s\": {:.4f}, \"first_step_s\": {:.4f}, \"step_s_median\": {:.4f}, "
      "\"step_s_min\": {:.4f}, \"step_s_max\": {:.4f}}},\n",
      first.encode, first.denoise, first.decode, first.total, first.first_step,
      gaps.empty() ? 0.0 : gaps[gaps.size() / 2], gaps.empty() ? 0.0 : gaps.front(),
      gaps.empty() ? 0.0 : gaps.back());
  report << "  \"plain_runs_s\": [";
  for (std::size_t i = 0; i < plain.size(); ++i) {
    report << std::format("{}{:.4f}", i ? ", " : "", plain[i]);
  }
  report << "],\n  \"plain_runs_phases\": [";
  for (std::size_t i = 0; i < plain_steps.size(); ++i) {
    report << (i ? ", " : "") << plain_steps[i];
  }
  report << "],\n";
  report << std::format(
      "  \"peak_memavailable_drop_bytes\": {},\n  \"first_run_memavailable_drop_bytes\": {},\n",
      baseline - memory.low(), baseline - low_first);
  report << "  \"cublaslt\": " << (*lt)->Describe() << ",\n";
  if (embeds_cmp) {
    report << "  \"prompt_embeds\": " << Json(*embeds_cmp) << ",\n";
  }
  report << "  \"noise_pred\": [";
  for (std::size_t i = 0; i < step_cmp.size(); ++i) {
    report << (i ? ",\n    " : "\n    ") << Json(step_cmp[i]);
  }
  report << "],\n";
  if (final_cmp) {
    report << "  \"latents_final\": " << Json(*final_cmp) << ",\n";
  }
  if (decoded_cmp) {
    report << "  \"vae_decoded\": " << Json(*decoded_cmp) << ",\n";
  }
  report << std::format(
      "  \"options\": {{\"size\": {}, \"steps\": {}, \"embeds\": \"{}\", "
      "\"force_latents\": {}, \"phases\": \"{}\", \"decode_reference\": {}, "
      "\"plan\": \"{}\", \"graphs\": {}}}\n}}\n",
      o.size, o.steps, o.reference_embeds ? "reference" : "native", o.force_latents,
      o.released ? "released" : "resident", o.decode_reference,
      o.plan == ki::PlanKind::kFast ? "fast" : "legacy", o.graphs);
  const std::string text_report = report.str();
  if (auto r = WriteBytes(o.out / "report.json", text_report.data(), text_report.size()); !r) {
    return r;
  }
  std::print("{}", text_report);
  return {};
}

}  // namespace

int main(int argc, char** argv) {
  auto options = Parse(std::span(argv, static_cast<std::size_t>(argc)));
  if (!options) {
    std::println(stderr, "{}", options.error());
    return 2;
  }
  if (auto r = Run(*options); !r) {
    std::println(stderr, "error: {}", r.error());
    return 1;
  }
  return 0;
}
