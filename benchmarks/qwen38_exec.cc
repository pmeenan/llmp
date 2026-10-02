// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// M3's second model slice (docs/experiments/qwen38-native/README.md): Qwen3.8
// Flash Next in Mia's NVFP4 and MXFP8 checkpoint, run by jitLLM natively and
// resident on one Spark from its v0 prepared artifact, for the correctness
// comparison with Mia's vLLM on the same checkpoint and a coarse speed and
// memory report.
//
//   jitllm_qwen38_exec --artifact DIR --out DIR [--context N] [--max-rows N]
//                      [--prompts FILE --generate N [--force FILE]]
//                      [--ppl FILE] [--dump NAMES] [--layout-proof] [--ab]
//                      [--bench-prefill N --bench-decode N] [--unfused | --exact]
//                      [--convert-experts]
//
// - Weights: the artifact is opened as untrusted input (artifact.h), bound
//   to the compiled-in Qwen3.8 profile (model/qwen38.h) and read with direct
//   I/O, one coalesced read plan, through pinned staging into device memory:
//   every group but the experts' in one region (the n-gram table among them:
//   resident for this slice), and each layer's routed experts in a slab
//   where expert e's group sits at e·S, S the group's stored bytes rounded to
//   the arrays' element (GGML's NVFP4 block) and 16 bytes (the resident
//   expert layout, addressed at a uniform stride). The n-gram hash's
//   constants are read back and checked against the table (model/qwen38.h
//   CheckQwen38PleHash) before any chunk.
// - Each chunk: host-built inputs (model/qwen38.h Qwen38Chunk), the GGML
//   graph (kernels/ggml/qwen38_graph.h) built, bound, planned with fusion
//   off and placed by the path the paged runner shares
//   (engine/qwen38_plan.h),
//   bound to the registry's implementations and run on one stream; plans
//   are kept per chunk shape. The graph runs jitLLM's fusions (qwen38_graph.h
//   Qwen38GraphOptions::fused) in their fast form by default (D-085: speed
//   before bit exactness), judged against the oracle; --exact runs the
//   reference form, and --unfused builds the GGML nodes the reference form
//   repeats, whose logits must be the reference form's bit for bit.
// - The routed experts: an artifact in the CUTLASS layout (the importer's
//   since the prefill work) is read as it is, and the graph's grouped GEMM
//   and vector products read it. An artifact in GGML's layout (the first
//   import) runs GGML's mul_mat_id (MMVQ and MMQ), or with --convert-experts
//   each slab is rewritten in place into the CUTLASS layout (moe_layout.h)
//   after loading (and after the A/B's and layout proof's GGML side); with
//   --layout-proof every slab is then converted back and compared with the
//   loaded bytes. --unfused needs GGML's layout, unconverted.
// - --prompts: lines `name<TAB>ids...`, each from a cleared state in
//   prefill chunks of --max-rows (ChunkRows; one chunk when it fits), then
//   --generate tokens one at a time: greedy, or with
//   --force the given tokens fed while the argmax is still recorded. Every
//   step's logits are written (the last row of each chunk only). With
//   --stepwise the prompt too is fed one token at a time, so every product
//   takes decode's kernels (a measure of the kernels' own disagreement).
//   With --state-roundtrip each prompt runs again with the whole state
//   copied to the host, overwritten on the device and restored halfway
//   through its steps; every logit must be the first run's, bit for bit.
// - --ppl: one line of ids from a cleared state in chunks of --max-rows
//   (ChunkRows: from 1,024 rows, a multiple of 64);
//   every row's logits, and the NLL of every token after the first.
// - --layout-proof: every layer's routed products at 1, 5 and 64 tokens over
//   the slab and over the reference layout (the slices packed), bit for bit.
// - --ab: the kernel A/B's timings on layer 0's weights (the README's table).
// - --bench-prefill/--bench-decode: that many synthetic tokens (best of
//   three) and decode steps from an empty context after a warm-up (mean of
//   three).

#include <cuda.h>
#include <cuda_runtime.h>
#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <charconv>
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
#include <functional>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <print>
#include <random>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include "artifact/artifact.h"
#include "artifact/layout.h"
#include "base/bytes.h"
#include "engine_names.h"
#include "execution/registry.h"
#include "ggml.h"
#include "kernels/ggml/cublas.h"
#include "kernels/ggml/dsv4_graph.h"
#include "kernels/ggml/executor.h"
#include "kernels/ggml/graph_plan.h"
#include "kernels/ggml/implementations.h"
#include "kernels/ggml/jitllm_ops.h"
#include "kernels/ggml/launch.h"
#include "kernels/ggml/moe_layout.h"
#include "kernels/ggml/qwen38_graph.h"
#include "kernels/ggml/tensors.h"
#include "kernels/ggml/validate_ext.h"
#include "model/qwen38.h"
#include "providers/cuda/cuda_device_execution.h"
#include "providers/device_execution.h"

namespace {

namespace kg = jitllm::kernels::ggml;
namespace moe = jitllm::kernels::ggml::moe;
namespace md = jitllm::model;
using jitllm::base::Bytes;
using Status = std::expected<void, std::string>;
using Clock = std::chrono::steady_clock;

std::unexpected<std::string> Error(std::string what) { return std::unexpected(std::move(what)); }

Status Cuda(cudaError_t result, std::string_view what) {
  if (result != cudaSuccess) {
    return Error(std::format("{}: {}", what, cudaGetErrorString(result)));
  }
  return {};
}

void* Pointer(std::uint64_t address) {
  return reinterpret_cast<void*>(address);  // NOLINT(performance-no-int-to-ptr)
}

std::uint64_t Address(const void* pointer) { return reinterpret_cast<std::uintptr_t>(pointer); }

std::uint64_t Round(std::uint64_t bytes, std::uint64_t to) { return (bytes + to - 1) / to * to; }

double Seconds(Clock::duration d) { return std::chrono::duration<double>(d).count(); }

// /proc/meminfo's MemAvailable, in bytes.
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

class Device {
 public:
  static std::expected<std::unique_ptr<Device>, std::string> Open() {
    if (auto set = Cuda(cudaSetDevice(0), "cudaSetDevice"); !set) {
      return std::unexpected(set.error());
    }
    if (auto context = Cuda(cudaFree(nullptr), "the CUDA context"); !context) {
      return std::unexpected(context.error());
    }
    auto execution = jitllm::providers::cuda::OpenDeviceExecution(0);
    if (!execution) {
      return Error("OpenDeviceExecution failed");
    }
    auto stream = (*execution)->CreateStream();
    if (!stream) {
      return Error("CreateStream failed");
    }
    return std::unique_ptr<Device>(new Device(std::move(*execution), *stream));  // NOLINT
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
  jitllm::providers::DeviceExecution& execution() { return *execution_; }
  jitllm::providers::StreamId stream() const { return stream_; }
  std::expected<cudaStream_t, std::string> Stream() {
    const auto native = execution_->Submission(stream_);
    if (!native) {
      return Error("Submission failed");
    }
    return static_cast<cudaStream_t>(native->handle);
  }
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
      if (*state == jitllm::providers::FenceState::kComplete) {
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
    return {};
  }

 private:
  Device(std::unique_ptr<jitllm::providers::DeviceExecution> execution,
         jitllm::providers::StreamId stream)
      : execution_(std::move(execution)), stream_(stream) {}
  std::unique_ptr<jitllm::providers::DeviceExecution> execution_;
  jitllm::providers::StreamId stream_;
};

// ------------------------------------------------------------------ weights

struct Weights {
  std::uint64_t dense = 0;
  std::uint64_t dense_bytes = 0;
  std::vector<std::uint64_t> group_address;
  std::vector<std::uint64_t> slab;
  std::vector<std::uint64_t> stride;
  std::uint64_t slab_bytes = 0;
  std::uint64_t slab_padding = 0;
  std::uint64_t bytes_read = 0;
  double load_seconds = 0;
};

std::uint64_t ResourceAddress(const jitllm::artifact::Artifact& artifact, const Weights& w,
                              std::uint32_t resource) {
  const auto& r = artifact.resources()[resource];
  return w.group_address[r.group] + r.offset.value();
}

std::uint64_t ArrayAddress(const jitllm::artifact::Artifact& artifact, const Weights& w,
                           std::uint32_t array) {
  const auto& a = artifact.expert_arrays()[array];
  return w.group_address[a.first_group] + a.group_offset.value();
}

Status PlaceWeights(const jitllm::artifact::Artifact& artifact, const md::Qwen38Profile& profile,
                    const md::Qwen38Binding& binding, Weights& w) {
  const auto groups = artifact.groups();
  w.group_address.assign(groups.size(), 0);
  w.slab.assign(profile.layers, 0);
  w.stride.assign(profile.layers, 0);
  std::vector<std::uint32_t> first(profile.layers, 0);
  std::vector<std::uint64_t> group_bytes(profile.layers, 0);
  for (std::uint32_t il = 0; il < profile.layers; ++il) {
    const md::Qwen38Layer& l = binding.layers[il];
    std::uint64_t unit = 16;
    std::uint64_t stored = 0;
    std::optional<std::uint32_t> first_group;
    for (const md::Qwen38Tensor* t : l.expert_arrays(binding.cutlass())) {
      const auto& a = artifact.expert_arrays()[t->index];
      auto type = kg::GgmlTypeOf(t->type);
      if (!type) {
        return Error(type.error().detail);
      }
      unit = std::lcm(unit, static_cast<std::uint64_t>(ggml_type_size(*type)));
      if (first_group && *first_group != a.first_group) {
        return Error(std::format("layer {}'s expert arrays do not share their groups", il));
      }
      first_group = a.first_group;
      stored = groups[a.first_group].stored.value();
    }
    const std::uint32_t layer_first = first_group.value_or(0);
    for (std::uint32_t e = 0; e < profile.experts; ++e) {
      const auto& g = groups[layer_first + e];
      if (g.kind != jitllm::artifact::GroupKind::kExpert || g.stored.value() != stored) {
        return Error(std::format("layer {}'s expert groups are not uniform", il));
      }
    }
    first[il] = layer_first;
    group_bytes[il] = stored;
    w.stride[il] = Round(stored, unit);
    w.slab_padding += (w.stride[il] - stored) * profile.experts;
  }
  std::vector<bool> expert(groups.size(), false);
  for (std::uint32_t il = 0; il < profile.layers; ++il) {
    for (std::uint32_t e = 0; e < profile.experts; ++e) {
      expert[first[il] + e] = true;
    }
  }
  std::vector<std::uint64_t> offset(groups.size(), 0);
  for (std::size_t g = 0; g < groups.size(); ++g) {
    if (expert[g]) {
      continue;
    }
    if (groups[g].kind == jitllm::artifact::GroupKind::kExpert) {
      return Error(std::format("expert group {} belongs to no bound array", g));
    }
    offset[g] = w.dense_bytes;
    w.dense_bytes += Round(groups[g].stored.value(), 256);
  }
  void* dense = nullptr;
  if (auto r = Cuda(cudaMalloc(&dense, w.dense_bytes), "the dense weights"); !r) {
    return r;
  }
  w.dense = Address(dense);
  for (std::size_t g = 0; g < groups.size(); ++g) {
    if (!expert[g]) {
      w.group_address[g] = w.dense + offset[g];
    }
  }
  for (std::uint32_t il = 0; il < profile.layers; ++il) {
    const std::uint64_t bytes = w.stride[il] * profile.experts;
    void* slab = nullptr;
    if (auto r = Cuda(cudaMalloc(&slab, bytes), std::format("layer {}'s expert slab", il)); !r) {
      return r;
    }
    w.slab[il] = Address(slab);
    w.slab_bytes += bytes;
    for (std::uint32_t e = 0; e < profile.experts; ++e) {
      w.group_address[first[il] + e] = w.slab[il] + (std::uint64_t{e} * w.stride[il]);
    }
    if (w.stride[il] > group_bytes[il]) {
      if (auto r = Cuda(cudaMemset2D(Pointer(w.slab[il] + group_bytes[il]), w.stride[il], 0,
                                     w.stride[il] - group_bytes[il], profile.experts),
                        "zeroing the slab's padding");
          !r) {
        return r;
      }
    }
  }
  return Cuda(cudaDeviceSynchronize(), "zeroing the slabs' padding");
}

Status LoadWeights(const jitllm::artifact::Artifact& artifact, Device& device, Weights& w) {
  const auto start = Clock::now();
  const auto groups = artifact.groups();
  std::vector<jitllm::artifact::ChunkKey> all;
  for (std::uint32_t g = 0; g < groups.size(); ++g) {
    for (std::uint32_t c = 0; c < groups[g].chunks; ++c) {
      all.push_back({.group = g, .chunk = c});
    }
  }
  const jitllm::artifact::ReadLimits limits{};
  const auto runs = artifact.PlanReads(all, {}, limits);
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
  std::vector<jitllm::artifact::FileDescriptor> shards;
  for (std::uint32_t s = 0; s < artifact.shards().size(); ++s) {
    auto fd = artifact.OpenShardForDirectRead(s);
    if (!fd) {
      return Error(std::format("shard {} cannot be opened for direct reads", s));
    }
    shards.push_back(std::move(*fd));
  }
  auto stream = device.Stream();
  if (!stream) {
    return std::unexpected(stream.error());
  }
  std::array<cudaEvent_t, 2> copied{};
  for (cudaEvent_t& e : copied) {
    if (auto r = Cuda(cudaEventCreateWithFlags(&e, cudaEventDisableTiming), "an event"); !r) {
      return r;
    }
  }
  std::array<bool, 2> pending = {false, false};
  std::size_t which = 0;
  for (const auto& run : *runs) {
    if (run.length.value() > staging_bytes) {
      return Error("a read run exceeds the staging buffer");
    }
    if (pending[which]) {
      if (auto r = Cuda(cudaEventSynchronize(copied[which]), "a staging copy"); !r) {
        return r;
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
      const std::uint64_t dst = w.group_address[segment.chunk.group] + segment.group_offset.value();
      if (auto r = Cuda(cudaMemcpyAsync(Pointer(dst), bytes + at, segment.length.value(),
                                        cudaMemcpyHostToDevice, *stream),
                        "a weight upload");
          !r) {
        return r;
      }
      at += segment.length.value();
    }
    if (auto r = Cuda(cudaEventRecord(copied[which], *stream), "an event record"); !r) {
      return r;
    }
    pending[which] = true;
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
  w.load_seconds = Seconds(Clock::now() - start);
  return {};
}

// The n-gram hash's constants, read back from the loaded weights and checked
// against the table's rows.
std::expected<md::Qwen38PleHash, std::string> ReadPleHash(const jitllm::artifact::Artifact& a,
                                                          const Weights& w,
                                                          const md::Qwen38Profile& p,
                                                          const md::Qwen38Binding& b) {
  if (b.gguf()) {
    // A GGUF checkpoint's constants are its kept metadata's.
    return md::ReadQwen38GgufHash(p, a, b.ple_table.ne[1]);
  }
  const md::Qwen38Layer& l = b.layers[p.ple_layer];
  const auto read =
      [&](const md::Qwen38Tensor& t) -> std::expected<std::vector<std::int64_t>, std::string> {
    std::vector<std::int64_t> v(t.ne[0]);
    if (auto r = Cuda(cudaMemcpy(v.data(), Pointer(ResourceAddress(a, w, t.index)),
                                 v.size() * sizeof(std::int64_t), cudaMemcpyDeviceToHost),
                      "reading the n-gram hash");
        !r) {
      return std::unexpected(r.error());
    }
    return v;
  };
  auto m = read(l.ple_multipliers);
  auto o = read(l.ple_head_offsets);
  auto v = read(l.ple_head_vocab);
  if (!m || !o || !v) {
    return Error("the n-gram hash could not be read");
  }
  return md::CheckQwen38PleHash(p, *m, *o, *v, b.ple_table.ne[1]);
}

// ------------------------------------------------------------------ chunks

using Planned = jitllm::benchmarks::Qwen38Planned;

struct Model {
  const jitllm::artifact::Artifact* artifact = nullptr;
  const md::Qwen38Profile* profile = nullptr;
  const md::Qwen38Binding* binding = nullptr;
  const Weights* weights = nullptr;
  const md::Qwen38StateLayout* state = nullptr;
  const md::Qwen38PleHash* hash = nullptr;
  std::uint64_t state_base = 0;
  bool fused = true;
  bool exact = false;
  // The slots hold the CUTLASS layout: the artifact's, or GGML's converted
  // at load (--convert-experts).
  bool cutlass = true;
};

// The shared planning path's view of the model (engine/qwen38_plan.h): the
// weights and state at their cudaMalloc addresses.
jitllm::benchmarks::Qwen38Model CommonOf(const Model& m) {
  const jitllm::artifact::Artifact* a = m.artifact;
  const Weights* w = m.weights;
  return {.artifact = a,
          .profile = m.profile,
          .binding = m.binding,
          .state = m.state,
          .places = {.resource = [a, w](std::uint32_t r) { return ResourceAddress(*a, *w, r); },
                     .array = [a, w](std::uint32_t x) { return ArrayAddress(*a, *w, x); },
                     .stride = w->stride,
                     .state = m.state_base,
                     .ple_table = ResourceAddress(*a, *w, m.binding->ple_table.index)},
          .fused = m.fused,
          .exact = m.exact,
          .cutlass = m.cutlass};
}

std::expected<std::unique_ptr<Planned>, std::string> PlanChunk(
    const Model& m, const kg::Qwen38ChunkShape& shape, const kg::DeviceChoices& choices,
    std::span<const std::string> keep_names, std::uint64_t activations,
    std::uint64_t activation_bytes) {
  return jitllm::benchmarks::PlanQwen38Chunk(CommonOf(m), shape, choices, activations,
                                             activation_bytes, keep_names);
}

class Runner {
 public:
  Runner(Device& device, Model model, kg::LaunchContext& launch,
         const jitllm::execution::Registry& registry, std::uint64_t activations,
         std::uint64_t activation_bytes, void* staging, std::uint64_t staging_bytes)
      : d_(device),
        m_(model),
        launch_(launch),
        registry_(registry),
        activations_(activations),
        activation_bytes_(activation_bytes),
        staging_(static_cast<std::byte*>(staging)),
        staging_bytes_(staging_bytes) {}

  Status Clear() {
    auto stream = d_.Stream();
    if (!stream) {
      return std::unexpected(stream.error());
    }
    if (auto r = Cuda(cudaMemsetAsync(Pointer(m_.state_base), 0, m_.state->bytes, *stream),
                      "clearing the state");
        !r) {
      return r;
    }
    return d_.Finish();
  }

  // One chunk: history[n_past, end) after n_past; the logits of its last
  // `outputs` rows (outputs x vocab) in `logits`.
  Status Chunk(std::span<const std::int32_t> history, std::uint32_t n_past, std::uint32_t outputs,
               std::vector<float>& logits, std::span<const std::string> keep = {},
               std::map<std::string, std::vector<float>>* kept = nullptr) {
    const auto rows = static_cast<std::uint32_t>(history.size() - n_past);
    // The fast graph's QSA selection makes its masks on the device.
    auto in = md::Qwen38Chunk(*m_.profile, *m_.state, *m_.hash, history, n_past, rows,
                              !m_.fused || m_.exact);
    if (!in) {
      return std::unexpected(in.error());
    }
    const kg::Qwen38ChunkShape shape = kg::Qwen38ShapeOf(*m_.state, *in, outputs);
    Planned* p = nullptr;
    std::unique_ptr<Planned> once;
    if (keep.empty()) {
      auto found = std::ranges::find_if(cache_, [&](const auto& e) { return e.first == shape; });
      if (found == cache_.end()) {
        auto planned = Prepare(shape, {});
        if (!planned) {
          return std::unexpected(planned.error());
        }
        if (cache_.size() >= 8) {
          cache_.erase(cache_.begin());
        }
        cache_.emplace_back(shape, std::move(*planned));
        found = std::prev(cache_.end());
        ++plans_made_;
      }
      p = found->second.get();
    } else {
      auto planned = Prepare(shape, keep);
      if (!planned) {
        return std::unexpected(planned.error());
      }
      once = std::move(*planned);
      p = once.get();
    }
    const kg::Qwen38Graph& g = p->graph;
    if ((g.mask != nullptr && in->mask.empty()) ||
        (g.mask_f32 != nullptr && in->mask_f32.empty())) {
      // A selection the device does not make (past its blocks): the host's
      // masks after all.
      in = md::Qwen38Chunk(*m_.profile, *m_.state, *m_.hash, history, n_past, rows, true);
      if (!in) {
        return std::unexpected(in.error());
      }
    }
    jitllm::benchmarks::Qwen38HostInputs host;
    jitllm::benchmarks::Qwen38Sources(g, *in, outputs, {}, host);
    const auto& sources = host.sources;
    auto stream = d_.Stream();
    if (!stream) {
      return std::unexpected(stream.error());
    }
    std::uint64_t staged = 0;
    for (const auto& [tensor, source] : sources) {
      const std::uint64_t bytes = ggml_nbytes(tensor);
      if (staged + bytes > staging_bytes_) {
        return Error("the inputs exceed their staging");
      }
      std::memcpy(staging_ + staged, source, bytes);
      if (auto r = Cuda(cudaMemcpyAsync(tensor->data, staging_ + staged, bytes,
                                        cudaMemcpyHostToDevice, *stream),
                        "an input copy");
          !r) {
        return r;
      }
      staged += Round(bytes, 256);
    }
    if (!p->bound) {
      return Error(std::format("chunk at {}: no bound plan", n_past));
    }
    if (auto r = p->bound->Run(launch_); !r) {
      return Error(std::format("chunk at {}: {}", n_past, r.error().detail));
    }
    const std::uint64_t logit_bytes = std::uint64_t{outputs} * m_.profile->vocab * sizeof(float);
    logits.resize(std::size_t{outputs} * m_.profile->vocab);
    if (auto r = Cuda(cudaMemcpyAsync(logits.data(), g.logits->data, logit_bytes,
                                      cudaMemcpyDeviceToHost, *stream),
                      "the logits copy");
        !r) {
      return r;
    }
    if (kept != nullptr) {
      if (auto r = d_.Finish(); !r) {
        return r;
      }
      for (const std::string& name : keep) {
        const ggml_tensor* t = g.Named(name);
        std::vector<float>& to = (*kept)[name];
        const auto n = static_cast<std::size_t>(ggml_nelements(t));
        to.assign(n, 0.0f);
        if (t->type == GGML_TYPE_F32 && ggml_is_contiguous(t)) {
          (void)cudaMemcpy(to.data(), t->data, n * sizeof(float), cudaMemcpyDeviceToHost);
        } else if (t->type == GGML_TYPE_I32 && ggml_is_contiguous(t)) {
          std::vector<std::int32_t> ids(n);
          (void)cudaMemcpy(ids.data(), t->data, n * sizeof(std::int32_t), cudaMemcpyDeviceToHost);
          std::ranges::transform(ids, to.begin(),
                                 [](std::int32_t v) { return static_cast<float>(v); });
        }
      }
    }
    if (auto r = d_.Finish(); !r) {
      return r;
    }
    if (launch_.faulted()) {
      return Error(std::format("chunk at {}: the launch context faulted", n_past));
    }
    return {};
  }

  // The whole state region to the host, the device copy overwritten with a
  // pattern, then copied back: what a spill and restore does to it.
  Status RoundTripState() {
    std::vector<std::byte> host(m_.state->bytes);
    if (auto r = d_.Finish(); !r) {
      return r;
    }
    if (auto r = Cuda(
            cudaMemcpy(host.data(), Pointer(m_.state_base), host.size(), cudaMemcpyDeviceToHost),
            "spilling the state");
        !r) {
      return r;
    }
    if (auto r =
            Cuda(cudaMemset(Pointer(m_.state_base), 0x5A, host.size()), "clobbering the state");
        !r) {
      return r;
    }
    return Cuda(
        cudaMemcpy(Pointer(m_.state_base), host.data(), host.size(), cudaMemcpyHostToDevice),
        "restoring the state");
  }

  int plans_made() const { return plans_made_; }
  std::uint64_t most_scratch() const { return most_scratch_; }
  std::uint64_t most_activations() const { return most_activations_; }

 private:
  std::expected<std::unique_ptr<Planned>, std::string> Prepare(const kg::Qwen38ChunkShape& shape,
                                                               std::span<const std::string> keep) {
    auto planned =
        PlanChunk(m_, shape, kg::DeviceChoicesOf(launch_), keep, activations_, activation_bytes_);
    if (!planned) {
      return std::unexpected(planned.error());
    }
    auto scratch = kg::PlanScratch(launch_, (*planned)->plan);
    if (!scratch) {
      return Error(scratch.error().detail);
    }
    if (*scratch > launch_.workspace().size.value()) {
      return Error(std::format("the plan's scratch ({} bytes) exceeds the pool ({} bytes)",
                               *scratch, launch_.workspace().size.value()));
    }
    (*planned)->scratch = *scratch;
    auto bound = kg::BoundGraph::Bind(registry_, (*planned)->plan);
    if (!bound) {
      return Error(bound.error().detail);
    }
    (*planned)->bound.emplace(std::move(*bound));
    most_scratch_ = std::max(most_scratch_, *scratch);
    most_activations_ = std::max(most_activations_, (*planned)->placement.extent);
    return planned;
  }

  Device& d_;
  Model m_;
  kg::LaunchContext& launch_;
  const jitllm::execution::Registry& registry_;
  std::uint64_t activations_;
  std::uint64_t activation_bytes_;
  std::byte* staging_;
  std::uint64_t staging_bytes_;
  std::vector<std::pair<kg::Qwen38ChunkShape, std::unique_ptr<Planned>>> cache_;
  int plans_made_ = 0;
  std::uint64_t most_scratch_ = 0;
  std::uint64_t most_activations_ = 0;
};

// The next chunk's rows: at most `max_rows`, and from 1,024 rows on a
// multiple of 64 (a shorter remainder follows as its own chunk): the
// tensor-core flash attention's mask pre-pass, which runs from 1,024 query
// rows, reads whole column tiles (8 rows) of the mask, and the chunk's mask
// has exactly its rows (fattn_mma.cu PlanFlashAttnMma); 64 is the padding
// llama.cpp gives its masks (GGML_KQ_MASK_PAD).
std::uint32_t ChunkRows(std::size_t remaining, std::uint32_t max_rows) {
  auto rows = static_cast<std::uint32_t>(std::min<std::size_t>(max_rows, remaining));
  if (rows >= 1024 && rows % 64 != 0) {
    rows -= rows % 64;
  }
  return rows;
}

// ------------------------------------------------------------------ inputs

struct TokenLine {
  std::string name;
  std::vector<std::int32_t> ids;
};

// A line's name names output files, so it is a plain file name.
bool PlainName(std::string_view name) {
  return name.empty() || (!name.starts_with('.') && std::ranges::all_of(name, [](char c) {
           return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') ||
                  c == '_' || c == '-' || c == '.';
         }));
}

// Lines `[name<TAB>]ids...`; every field must be a token id in [0, vocab).
std::expected<std::vector<TokenLine>, std::string> ReadTokenLines(const std::filesystem::path& p,
                                                                  std::uint32_t vocab) {
  std::ifstream file(p);
  if (!file) {
    return Error(std::format("cannot read {}", p.string()));
  }
  std::vector<TokenLine> out;
  std::string line;
  while (std::getline(file, line)) {
    if (line.empty() || line.starts_with('#')) {
      continue;
    }
    TokenLine t;
    const auto tab = line.find('\t');
    std::string_view rest = line;
    if (tab != std::string::npos) {
      t.name = line.substr(0, tab);
      rest = std::string_view(line).substr(tab + 1);
    }
    if (!PlainName(t.name)) {
      return Error(std::format("{}: a line's name is not a plain file name", p.string()));
    }
    while (true) {
      const auto at = rest.find_first_not_of(" \t");
      if (at == std::string_view::npos) {
        break;
      }
      rest.remove_prefix(at);
      std::int32_t id = 0;
      const auto [end, ec] = std::from_chars(rest.data(), rest.data() + rest.size(), id);
      const auto used = static_cast<std::size_t>(end - rest.data());
      if (ec != std::errc() || id < 0 || std::cmp_greater_equal(id, vocab) ||
          (used < rest.size() && rest[used] != ' ' && rest[used] != '\t')) {
        return Error(
            std::format("{}: a field that is not a token id in [0, {})", p.string(), vocab));
      }
      t.ids.push_back(id);
      rest.remove_prefix(used);
    }
    out.push_back(std::move(t));
  }
  return out;
}

std::int32_t Argmax(std::span<const float> row) {
  return static_cast<std::int32_t>(std::ranges::max_element(row) - row.begin());
}

double Nll(std::span<const float> row, std::int32_t target) {
  if (target < 0 || std::cmp_greater_equal(target, row.size())) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  const double most = *std::ranges::max_element(row);
  double sum = 0;
  for (const float v : row) {
    sum += std::exp(static_cast<double>(v) - most);
  }
  return (most + std::log(sum)) - static_cast<double>(row[static_cast<std::size_t>(target)]);
}

Status WriteFloats(const std::filesystem::path& p, std::span<const float> v) {
  std::ofstream out(p, std::ios::binary);
  out.write(reinterpret_cast<const char*>(v.data()), static_cast<std::streamsize>(v.size_bytes()));
  return out ? Status{} : Error(std::format("cannot write {}", p.string()));
}

std::string Describe(const ggml_tensor* t) {
  if (t == nullptr) {
    return "null";
  }
  return std::format("{} {} ne [{}, {}, {}, {}] nb [{}, {}, {}, {}] at {:#x}", ggml_op_desc(t),
                     ggml_type_name(t->type), t->ne[0], t->ne[1], t->ne[2], t->ne[3], t->nb[0],
                     t->nb[1], t->nb[2], t->nb[3], Address(t->data));
}

// The first step whose scratch plan refuses, with its operands (a
// diagnostic for PlanScratch's refusals).
std::string FailingStep(const kg::LaunchContext& launch, const kg::GraphPlan& plan) {
  for (std::size_t i = 0; i < plan.steps.size(); ++i) {
    const auto& step = plan.steps[i];
    kg::GraphPlan one;
    one.steps.push_back(step);
    if (!kg::PlanScratch(launch, one)) {
      const ggml_tensor* n = step.nodes.front();
      return std::format(" (step {} {}: {}; src0 {}; src1 {})", i, step.implementation, Describe(n),
                         Describe(n->src[0]), Describe(n->src[1]));
    }
  }
  return "";
}

// ------------------------------------------------------------------ small graphs

// Plans, binds and runs `nodes` (their leaves bound) with every computed
// tensor in `buffer`; `reps` timed runs after one warm-up, the mean in ms.
std::expected<double, std::string> RunNodes(Device& d, kg::LaunchContext& launch,
                                            const jitllm::execution::Registry& registry,
                                            std::vector<ggml_tensor*> nodes, std::uint64_t buffer,
                                            std::uint64_t buffer_bytes, int reps,
                                            std::string* implementation = nullptr) {
  constexpr std::uint64_t kDistinct = std::uint64_t{1} << 46U;
  kg::BindDistinct(nodes, kDistinct);
  auto plan = kg::PlanGraph(nodes, false, kg::DeviceChoicesOf(launch));
  if (!plan) {
    return Error(plan.error().detail);
  }
  auto placement = kg::PlaceActivations(nodes, *plan, {}, 256, nodes);
  if (!placement) {
    return Error(placement.error().detail);
  }
  if (placement->extent > buffer_bytes) {
    return Error("a small graph's activations exceed its buffer");
  }
  for (const auto& [tensor, offset] : placement->offsets) {
    kg::TensorArena::Bind(tensor, buffer + offset);
  }
  kg::BindViews(nodes);
  auto again = kg::PlanGraph(nodes, false, kg::DeviceChoicesOf(launch));
  if (!again) {
    return Error(again.error().detail);
  }
  auto bound = kg::BoundGraph::Bind(registry, *again);
  if (!bound) {
    return Error(bound.error().detail);
  }
  if (implementation != nullptr) {
    *implementation = "";
    for (const auto& step : again->steps) {
      *implementation +=
          std::format("{}{}", implementation->empty() ? "" : "+", step.implementation);
    }
  }
  auto stream = d.Stream();
  if (!stream) {
    return std::unexpected(stream.error());
  }
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  (void)cudaEventCreate(&start);
  (void)cudaEventCreate(&stop);
  double ms = 0;
  for (int rep = -1; rep < reps; ++rep) {
    if (rep == 0) {
      (void)cudaEventRecord(start, *stream);
    }
    if (auto r = bound->Run(launch); !r) {
      return Error(r.error().detail);
    }
  }
  (void)cudaEventRecord(stop, *stream);
  if (auto r = d.Finish(); !r) {
    return std::unexpected(r.error());
  }
  float elapsed = 0;
  (void)cudaEventElapsedTime(&elapsed, start, stop);
  ms = reps > 0 ? static_cast<double>(elapsed) / reps : 0;
  (void)cudaEventDestroy(start);
  (void)cudaEventDestroy(stop);
  return ms;
}

// ------------------------------------------------------------------ layout proof and A/B

std::vector<std::int32_t> RandomRoutes(std::mt19937& rng, std::int64_t experts, std::int64_t used,
                                       std::int64_t tokens) {
  std::uniform_int_distribution<int> pick(0, static_cast<int>(experts) - 1);
  std::vector<std::int32_t> routes(static_cast<std::size_t>(used * tokens));
  for (std::int64_t t = 0; t < tokens; ++t) {
    std::vector<int> chosen;
    while (std::cmp_less(chosen.size(), used)) {
      const int e = pick(rng);
      if (std::ranges::find(chosen, e) == chosen.end()) {
        chosen.push_back(e);
      }
    }
    for (std::int64_t j = 0; j < used; ++j) {
      routes[static_cast<std::size_t>((t * used) + j)] = chosen[static_cast<std::size_t>(j)];
    }
  }
  return routes;
}

Status LayoutProof(const Model& m, Device& d, kg::LaunchContext& launch,
                   const jitllm::execution::Registry& registry, std::string& report) {
  const auto& a = *m.artifact;
  const auto& w = *m.weights;
  std::mt19937 rng(20260928);  // NOLINT(bugprone-random-generator-seed): reproducible
  std::normal_distribution<float> normal(0.0f, 1.0f);
  std::uint64_t compared = 0;
  std::uint64_t differing = 0;
  report += "[";
  bool first_entry = true;
  for (std::uint32_t il = 0; il < m.profile->layers; ++il) {
    const md::Qwen38Layer& l = m.binding->layers[il];
    for (const md::Qwen38Tensor* t : {&l.gate_exps, &l.up_exps, &l.down_exps}) {
      const auto& array = a.expert_arrays()[t->index];
      const std::uint64_t slice = array.slice_bytes.value();
      const std::uint64_t pad = array.readable.value() - slice;
      // The reference layout: the slices packed in expert order (a GGUF's
      // [k, n, experts] tensor), the last one's readable padding zeroed.
      void* reference = nullptr;
      const std::uint64_t packed_bytes = (slice * array.count) + pad;
      if (auto r = Cuda(cudaMalloc(&reference, packed_bytes), "the reference layout"); !r) {
        return r;
      }
      (void)cudaMemset(reference, 0, packed_bytes);
      for (std::uint32_t e = 0; e < array.count; ++e) {
        const std::uint64_t src =
            w.group_address[array.first_group + e] + array.group_offset.value();
        if (auto r = Cuda(cudaMemcpy(static_cast<std::byte*>(reference) + (e * slice), Pointer(src),
                                     slice, cudaMemcpyDeviceToDevice),
                          "a reference slice");
            !r) {
          return r;
        }
      }
      auto type = kg::GgmlTypeOf(t->type);
      if (!type) {
        return Error(type.error().detail);
      }
      const auto k = static_cast<std::int64_t>(t->ne[0]);
      const auto n = static_cast<std::int64_t>(t->ne[1]);
      for (const std::int64_t tokens : {1, 5, 64}) {
        const std::int64_t used = m.profile->experts_used;
        auto arena = kg::TensorArena::Create(32);
        if (!arena) {
          return Error(arena.error().detail);
        }
        ggml_context* c = arena->context();
        ggml_tensor* slab =
            ggml_new_tensor_3d(c, *type, k, n, static_cast<std::int64_t>(array.count));
        slab->nb[2] = w.stride[il];
        slab->nb[3] = w.stride[il] * array.count;
        ggml_tensor* packed =
            ggml_new_tensor_3d(c, *type, k, n, static_cast<std::int64_t>(array.count));
        ggml_tensor* x = ggml_new_tensor_3d(c, GGML_TYPE_F32, k, 1, tokens);
        ggml_tensor* ids = ggml_new_tensor_2d(c, GGML_TYPE_I32, used, tokens);
        kg::TensorArena::Bind(slab, ArrayAddress(a, w, t->index));
        kg::TensorArena::Bind(packed, Address(reference));
        kg::MarkRowPaddingReadable(slab);
        kg::MarkRowPaddingReadable(packed);
        ggml_tensor* out_slab = ggml_mul_mat_id(c, slab, x, ids);
        ggml_tensor* out_packed = ggml_mul_mat_id(c, packed, x, ids);
        std::vector<float> xs(static_cast<std::size_t>(k * tokens));
        for (float& v : xs) {
          v = normal(rng);
        }
        const auto routes = RandomRoutes(rng, m.profile->experts, used, tokens);
        const std::uint64_t out_bytes = ggml_nbytes(out_slab);
        void* buffer = nullptr;
        const std::uint64_t total =
            Round(ggml_nbytes(x), 256) + Round(ggml_nbytes(ids), 256) + (2 * Round(out_bytes, 256));
        if (auto r = Cuda(cudaMalloc(&buffer, total), "the proof's buffers"); !r) {
          return r;
        }
        std::uint64_t at = Address(buffer);
        kg::TensorArena::Bind(x, at);
        at += Round(ggml_nbytes(x), 256);
        kg::TensorArena::Bind(ids, at);
        at += Round(ggml_nbytes(ids), 256);
        kg::TensorArena::Bind(out_slab, at);
        at += Round(out_bytes, 256);
        kg::TensorArena::Bind(out_packed, at);
        (void)cudaMemcpy(x->data, xs.data(), ggml_nbytes(x), cudaMemcpyHostToDevice);
        (void)cudaMemcpy(ids->data, routes.data(), ggml_nbytes(ids), cudaMemcpyHostToDevice);
        const std::vector<ggml_tensor*> nodes = {out_slab, out_packed};
        auto plan = kg::PlanGraph(nodes, false, kg::DeviceChoicesOf(launch));
        if (!plan) {
          return Error(plan.error().detail);
        }
        if (plan->steps[0].implementation != plan->steps[1].implementation) {
          return Error("the two layouts took different kernel families");
        }
        auto bound = kg::BoundGraph::Bind(registry, *plan);
        if (!bound) {
          return Error(bound.error().detail);
        }
        if (auto r = bound->Run(launch); !r) {
          return Error(r.error().detail);
        }
        if (auto r = d.Finish(); !r) {
          return r;
        }
        std::vector<std::uint32_t> got_slab(out_bytes / 4);
        std::vector<std::uint32_t> got_packed(out_bytes / 4);
        (void)cudaMemcpy(got_slab.data(), out_slab->data, out_bytes, cudaMemcpyDeviceToHost);
        (void)cudaMemcpy(got_packed.data(), out_packed->data, out_bytes, cudaMemcpyDeviceToHost);
        std::uint64_t diff = 0;
        for (std::size_t i = 0; i < got_slab.size(); ++i) {
          diff += got_slab[i] != got_packed[i];
        }
        compared += got_slab.size();
        differing += diff;
        report += std::format(
            R"({}{{"layer":{},"tensor":"{}","tokens":{},"implementation":"{}","stride":{},"packed_stride":{},"elements":{},"differing":{}}})",
            first_entry ? "" : ",", il, array.name, tokens, plan->steps[0].implementation,
            w.stride[il], slice, got_slab.size(), diff);
        first_entry = false;
        (void)cudaFree(buffer);
      }
      (void)cudaFree(reference);
    }
  }
  report += "]";
  std::println("layout proof: {} outputs compared, {} differing", compared, differing);
  if (differing != 0) {
    return Error("the resident expert layout's products differ from the reference layout's");
  }
  return {};
}

// The kernel A/B on layer 0's (and layer 3's) weights: each candidate's mean
// time over 50 runs, with the bytes it must read.
Status KernelAb(const Model& m, Device& d, kg::LaunchContext& launch,
                const jitllm::execution::Registry& registry, std::string& report) {
  const auto& a = *m.artifact;
  const auto& w = *m.weights;
  const md::Qwen38Profile& p = *m.profile;
  std::mt19937 rng(7);  // NOLINT(bugprone-random-generator-seed): reproducible
  std::normal_distribution<float> normal(0.0f, 1.0f);
  constexpr std::uint64_t kBuffer = std::uint64_t{3} << 30U;
  void* buffer = nullptr;
  if (auto r = Cuda(cudaMalloc(&buffer, kBuffer), "the A/B buffer"); !r) {
    return r;
  }
  void* inputs = nullptr;
  if (auto r = Cuda(cudaMalloc(&inputs, std::uint64_t{64} << 20U), "the A/B inputs"); !r) {
    return r;
  }
  report += "[";
  bool first = true;
  const auto entry = [&](std::string_view what, std::int64_t tokens, std::string_view impl,
                         double ms, double bytes) {
    report += std::format(
        R"({}{{"case":"{}","tokens":{},"implementation":"{}","ms":{:.4f},"weight_gb_s":{:.1f}}})",
        first ? "" : ",", what, tokens, impl, ms, bytes / (ms * 1e6));
    first = false;
    std::println("A/B {:<34} t={:<4} {:<60} {:.4f} ms ({:.1f} GB/s of weights)", what, tokens, impl,
                 ms, bytes / (ms * 1e6));
  };
  const auto random_input = [&](ggml_tensor* x) {
    std::vector<float> xs(static_cast<std::size_t>(ggml_nelements(x)));
    for (float& v : xs) {
      v = normal(rng);
    }
    (void)cudaMemcpy(x->data, xs.data(), ggml_nbytes(x), cudaMemcpyHostToDevice);
  };
  // NVFP4 routed experts: layer 0's gate (k 2560, n 640) and down (k 640, n
  // 2560) through GGML's mul_mat_id (MMVQ up to 8 tokens, MMQ beyond), 10
  // routed experts a token.
  const md::Qwen38Layer& l0 = m.binding->layers[0];
  for (const auto& [name, t] : {std::pair{"nvfp4 experts gate 2560x640", &l0.gate_exps},
                                std::pair{"nvfp4 experts down 640x2560", &l0.down_exps}}) {
    for (const std::int64_t tokens : {1, 8, 64, 512, 2048}) {
      auto arena = kg::TensorArena::Create(16);
      ggml_context* c = arena->context();
      const auto k = static_cast<std::int64_t>(t->ne[0]);
      const auto n = static_cast<std::int64_t>(t->ne[1]);
      ggml_tensor* slab = ggml_new_tensor_3d(c, GGML_TYPE_NVFP4, k, n, p.experts);
      slab->nb[2] = w.stride[0];
      slab->nb[3] = w.stride[0] * p.experts;
      kg::TensorArena::Bind(slab, ArrayAddress(a, w, t->index));
      kg::MarkRowPaddingReadable(slab);
      ggml_tensor* x = ggml_new_tensor_3d(c, GGML_TYPE_F32, k, 1, tokens);
      ggml_tensor* ids = ggml_new_tensor_2d(c, GGML_TYPE_I32, p.experts_used, tokens);
      kg::TensorArena::Bind(x, Address(inputs));
      kg::TensorArena::Bind(ids, Address(inputs) + Round(ggml_nbytes(x), 256));
      random_input(x);
      const auto routes = RandomRoutes(rng, p.experts, p.experts_used, tokens);
      (void)cudaMemcpy(ids->data, routes.data(), ggml_nbytes(ids), cudaMemcpyHostToDevice);
      ggml_tensor* y = ggml_mul_mat_id(c, slab, x, ids);
      std::string impl;
      auto ms = RunNodes(d, launch, registry, {y}, Address(buffer), kBuffer, 50, &impl);
      if (!ms) {
        return Error(std::format("{}: {}", name, ms.error()));
      }
      // Distinct experts read (at most experts_used · tokens, at most all).
      std::vector<std::int32_t> distinct = routes;
      std::ranges::sort(distinct);
      const auto count =
          static_cast<double>(std::ranges::unique(distinct).begin() - distinct.begin());
      entry(
          name, tokens, impl, *ms,
          count * static_cast<double>(ggml_row_size(GGML_TYPE_NVFP4, k)) * static_cast<double>(n));
    }
  }
  // MXFP8 dense products: GDN layer 0's QKV (2560 -> 10240) and output
  // (6144 -> 2560), QSA layer 3's Q (2560 -> 12288): jitLLM's MXFP8 vector
  // product against GGML's BF16 products over the dequantized weights (the
  // candidate that needs no MXFP8 kernel), and at prefill widths the
  // dequantization plus GGML's product.
  const md::Qwen38Layer& l3 = m.binding->layers[3];
  for (const auto& [name, mxt] : {std::pair{"mxfp8 qkv 2560x10240", &l0.qkv},
                                  std::pair{"mxfp8 ssm_out 6144x2560", &l0.ssm_out},
                                  std::pair{"mxfp8 attn_q 2560x12288", &l3.q}}) {
    const auto k = static_cast<std::int64_t>(mxt->codes.ne[0]);
    const auto n = static_cast<std::int64_t>(mxt->codes.ne[1]);
    const double mx_bytes = static_cast<double>(k * n) * (1.0 + (1.0 / 32.0));
    for (const std::int64_t tokens : {1, 4, 8, 512}) {
      auto arena = kg::TensorArena::Create(32);
      ggml_context* c = arena->context();
      ggml_tensor* codes = ggml_new_tensor_2d(c, GGML_TYPE_I8, k, n);
      ggml_tensor* scales = ggml_new_tensor_2d(c, GGML_TYPE_I8, k / 32, n);
      kg::TensorArena::Bind(codes, ResourceAddress(a, w, mxt->codes.index));
      kg::TensorArena::Bind(scales, ResourceAddress(a, w, mxt->scales.index));
      ggml_tensor* x = ggml_new_tensor_2d(c, GGML_TYPE_F32, k, tokens);
      kg::TensorArena::Bind(x, Address(inputs));
      random_input(x);
      if (tokens <= kg::kMxfp8VecColumns) {
        std::string impl;
        auto ms = RunNodes(d, launch, registry, {kg::Mxfp8MulMatVec(c, codes, scales, x)},
                           Address(buffer), kBuffer, 50, &impl);
        if (!ms) {
          return Error(std::format("{}: {}", name, ms.error()));
        }
        entry(name, tokens, impl, *ms, mx_bytes);
        // The BF16 candidate: weights dequantized once (as an import-time
        // BF16 conversion would store them), then GGML's product alone.
        ggml_tensor* bf16 = kg::Mxfp8Dequant(c, codes, scales);
        auto made = RunNodes(d, launch, registry, {bf16}, Address(buffer), kBuffer, 1);
        if (!made) {
          return Error(made.error());
        }
        ggml_tensor* weights = ggml_new_tensor_2d(c, GGML_TYPE_BF16, k, n);
        kg::TensorArena::Bind(weights, Address(bf16->data));
        ggml_tensor* y = ggml_mul_mat(c, weights, x);
        // The product's output after the weights in the buffer.
        auto bms =
            RunNodes(d, launch, registry, {y}, Address(buffer) + Round(ggml_nbytes(bf16), 256),
                     kBuffer - Round(ggml_nbytes(bf16), 256), 50, &impl);
        if (!bms) {
          return Error(std::format("{} BF16: {}", name, bms.error()));
        }
        entry(std::format("{} (BF16 weights)", name), tokens, impl, *bms,
              static_cast<double>(k * n) * 2.0);
      } else {
        std::string impl;
        ggml_tensor* dq = kg::Mxfp8Dequant(c, codes, scales);
        ggml_tensor* y = ggml_mul_mat(c, dq, x);
        auto ms = RunNodes(d, launch, registry, {dq, y}, Address(buffer), kBuffer, 20, &impl);
        if (!ms) {
          return Error(std::format("{}: {}", name, ms.error()));
        }
        entry(name, tokens, impl, *ms, mx_bytes);
      }
    }
  }
  report += "]";
  (void)cudaFree(buffer);
  (void)cudaFree(inputs);
  return {};
}

// ------------------------------------------------------------------ the CUTLASS layout

moe::ExpertSlab SlabOf(const Model& m, std::uint32_t il) {
  const md::Qwen38Layer& l = m.binding->layers[il];
  const auto& a = *m.artifact;
  const auto offset = [&](const md::Qwen38Tensor& t) {
    return a.expert_arrays()[t.index].group_offset.value();
  };
  return {.base = Pointer(m.weights->slab[il]),
          .stride = m.weights->stride[il],
          .experts = m.profile->experts,
          .gate = offset(l.gate_exps),
          .up = offset(l.up_exps),
          .down = offset(l.down_exps),
          .layout = {.ffn = m.profile->expert_ffn, .width = m.profile->width}};
}

// Every layer's slab rewritten in place into the CUTLASS layout
// (moe_layout.h); with `proof`, each converted back and compared, slice by
// slice, with the GGML bytes it was loaded with.
Status ConvertExperts(const Model& m, Device& d, bool proof, std::string& report) {
  auto stream = d.Stream();
  if (!stream) {
    return std::unexpected(stream.error());
  }
  constexpr std::int64_t kBatch = 32;
  const std::uint64_t stride = *std::ranges::max_element(m.weights->stride);
  const std::uint64_t slab_bytes = stride * m.profile->experts;
  void* temp = nullptr;
  void* original = nullptr;
  void* back = nullptr;
  if (auto r = Cuda(cudaMalloc(&temp, stride * kBatch), "the conversion's staging"); !r) {
    return r;
  }
  if (proof) {
    if (auto r = Cuda(cudaMalloc(&original, slab_bytes), "the proof's copy"); !r) {
      return r;
    }
    if (auto r = Cuda(cudaMalloc(&back, slab_bytes), "the proof's inverse"); !r) {
      return r;
    }
  }
  const auto t0 = Clock::now();
  std::uint64_t compared = 0;
  std::uint64_t differing = 0;
  std::vector<std::uint8_t> host_original;
  std::vector<std::uint8_t> host_back;
  for (std::uint32_t il = 0; il < m.profile->layers; ++il) {
    const moe::ExpertSlab slab = SlabOf(m, il);
    const std::uint64_t bytes = slab.stride * static_cast<std::uint64_t>(slab.experts);
    if (proof) {
      (void)cudaMemcpyAsync(original, slab.base, bytes, cudaMemcpyDeviceToDevice, *stream);
    }
    if (!moe::ToCutlassLayout(slab, temp, kBatch, *stream)) {
      return Error(std::format("layer {}'s expert slab does not hold the CUTLASS layout", il));
    }
    if (!proof) {
      continue;
    }
    if (!moe::ToGgmlLayout(slab, back, *stream)) {
      return Error(std::format("layer {}'s slab could not be converted back", il));
    }
    if (auto r = d.Finish(); !r) {
      return r;
    }
    host_original.resize(bytes);
    host_back.resize(bytes);
    (void)cudaMemcpy(host_original.data(), original, bytes, cudaMemcpyDeviceToHost);
    (void)cudaMemcpy(host_back.data(), back, bytes, cudaMemcpyDeviceToHost);
    const md::Qwen38Layer& l = m.binding->layers[il];
    for (const md::Qwen38Tensor* t : {&l.gate_exps, &l.up_exps, &l.down_exps}) {
      const auto& array = m.artifact->expert_arrays()[t->index];
      for (std::int64_t e = 0; e < slab.experts; ++e) {
        const std::uint64_t at =
            (static_cast<std::uint64_t>(e) * slab.stride) + array.group_offset.value();
        const std::uint64_t n = array.slice_bytes.value();
        compared += n;
        for (std::uint64_t i = 0; i < n; ++i) {
          differing += host_original[at + i] != host_back[at + i];
        }
      }
    }
  }
  if (auto r = d.Finish(); !r) {
    return r;
  }
  const double seconds = Seconds(Clock::now() - t0);
  (void)cudaFree(temp);
  (void)cudaFree(original);
  (void)cudaFree(back);
  report = std::format(R"({{"seconds":{:.3f},"proof_bytes":{},"proof_differing":{}}})", seconds,
                       compared, differing);
  std::println(
      "experts to the CUTLASS layout in {:.2f} s{}", seconds,
      proof ? std::format("; proof: {} bytes back, {} differing", compared, differing) : "");
  if (proof && differing != 0) {
    return Error("the CUTLASS layout does not convert back to the loaded bytes");
  }
  return {};
}

// One MoE block's routed experts (products, SwiGLU, weighted sum; no shared
// expert) on layer 0's weights at prefill widths, in whichever layout the
// slab holds: GGML's (mul_mat_id: MMQ) or CUTLASS's (the grouped GEMM path).
// The mean of 10 runs after a warm-up.
Status MoeAb(const Model& m, Device& d, kg::LaunchContext& launch,
             const jitllm::execution::Registry& registry, bool cutlass, std::string& report) {
  const auto& a = *m.artifact;
  const auto& w = *m.weights;
  const md::Qwen38Profile& p = *m.profile;
  const md::Qwen38Layer& l0 = m.binding->layers[0];
  constexpr std::uint64_t kBuffer = std::uint64_t{3} << 30U;
  void* buffer = nullptr;
  if (auto r = Cuda(cudaMalloc(&buffer, kBuffer), "the A/B buffer"); !r) {
    return r;
  }
  void* inputs = nullptr;
  constexpr std::uint64_t kInputs = std::uint64_t{256} << 20U;
  if (auto r = Cuda(cudaMalloc(&inputs, kInputs), "the A/B inputs"); !r) {
    return r;
  }
  std::mt19937 rng(11);  // NOLINT(bugprone-random-generator-seed): reproducible
  std::normal_distribution<float> normal(0.0f, 1.0f);
  const auto n_embd = static_cast<std::int64_t>(p.width);
  const auto f = static_cast<std::int64_t>(p.expert_ffn);
  const auto used = static_cast<std::int64_t>(p.experts_used);
  const auto experts = static_cast<std::int64_t>(p.experts);
  for (const std::int64_t tokens : {512, 2048, 8192}) {
    auto arena = kg::TensorArena::Create(64);
    ggml_context* c = arena->context();
    std::uint64_t at = Address(inputs);
    const auto place = [&](ggml_tensor* t, const auto& values) {
      kg::TensorArena::Bind(t, at);
      (void)cudaMemcpy(t->data, values.data(), ggml_nbytes(t), cudaMemcpyHostToDevice);
      at += Round(ggml_nbytes(t), 256);
      return t;
    };
    std::vector<float> xs(static_cast<std::size_t>(n_embd * tokens));
    for (float& v : xs) {
      v = normal(rng);
    }
    const auto routes = RandomRoutes(rng, experts, used, tokens);
    ggml_tensor* x = place(ggml_new_tensor_2d(c, GGML_TYPE_F32, n_embd, tokens), xs);
    ggml_tensor* ids = place(ggml_new_tensor_2d(c, GGML_TYPE_I32, used, tokens), routes);
    ggml_tensor* weights = place(ggml_new_tensor_3d(c, GGML_TYPE_F32, 1, used, tokens),
                                 std::vector<float>(static_cast<std::size_t>(used * tokens), 0.1f));
    ggml_tensor* shared = place(ggml_new_tensor_2d(c, GGML_TYPE_F32, n_embd, tokens),
                                std::vector<float>(static_cast<std::size_t>(n_embd * tokens)));
    ggml_tensor* shared_gate = place(ggml_new_tensor_2d(c, GGML_TYPE_F32, 1, tokens),
                                     std::vector<float>(static_cast<std::size_t>(tokens)));
    const auto scale = [&](const md::Qwen38Tensor& t) {
      ggml_tensor* s = ggml_new_tensor_1d(c, GGML_TYPE_F32, experts);
      kg::TensorArena::Bind(s, ResourceAddress(a, w, t.index));
      return s;
    };
    ggml_tensor* gs = scale(l0.gate_exps_scale);
    ggml_tensor* us = scale(l0.up_exps_scale);
    ggml_tensor* ds = scale(l0.down_exps_scale);
    std::vector<ggml_tensor*> nodes;
    if (cutlass) {
      const moe::ExpertLayout layout{.ffn = p.expert_ffn, .width = p.width};
      ggml_tensor* slab =
          ggml_new_tensor_2d(c, GGML_TYPE_I8, static_cast<std::int64_t>(w.stride[0]), experts);
      kg::TensorArena::Bind(slab, w.slab[0]);
      ggml_tensor* route = kg::MoeRoute(c, ids, experts);
      ggml_tensor* a1 = kg::MoeQuantize(c, x, route);
      ggml_tensor* d1 = kg::MoeGemm(c, a1, route, slab, 2 * f, moe::ExpertLayout::gate_up_codes(),
                                    layout.gate_up_scales());
      ggml_tensor* a2 = kg::MoeGluQuantize(c, d1, a1, route, gs, us);
      ggml_tensor* d2 =
          kg::MoeGemm(c, a2, route, slab, n_embd, layout.down_codes(), layout.down_scales());
      nodes = {route, a1, d1,
               a2,    d2, kg::MoeCombineSorted(c, d2, a2, route, ds, weights, shared, shared_gate)};
    } else {
      const auto slice = [&](const md::Qwen38Tensor& t) {
        ggml_tensor* s = ggml_new_tensor_3d(c, GGML_TYPE_NVFP4, static_cast<std::int64_t>(t.ne[0]),
                                            static_cast<std::int64_t>(t.ne[1]), experts);
        s->nb[2] = w.stride[0];
        s->nb[3] = w.stride[0] * p.experts;
        kg::TensorArena::Bind(s, ArrayAddress(a, w, t.index));
        kg::MarkRowPaddingReadable(s);
        return s;
      };
      ggml_tensor* x3 = ggml_reshape_3d(c, x, n_embd, 1, tokens);
      ggml_tensor* up = ggml_mul_mat_id(c, slice(l0.up_exps), x3, ids);
      ggml_tensor* gate = ggml_mul_mat_id(c, slice(l0.gate_exps), x3, ids);
      ggml_tensor* act = kg::MoeGlu(c, gate, up, ids, gs, us);
      ggml_tensor* down = ggml_mul_mat_id(c, slice(l0.down_exps), act, ids);
      nodes = kg::GraphOrder(std::vector<ggml_tensor*>{
          kg::MoeCombine(c, down, ids, ds, weights, shared, shared_gate)});
    }
    std::string impl;
    auto ms = RunNodes(d, launch, registry, nodes, Address(buffer), kBuffer, 10, &impl);
    if (!ms) {
      return Error(std::format("the MoE A/B at {} tokens: {}", tokens, ms.error()));
    }
    report +=
        std::format(R"({}{{"layout":"{}","tokens":{},"ms":{:.4f},"steps":"{}"}})",
                    report.empty() ? "" : ",", cutlass ? "cutlass" : "ggml", tokens, *ms, impl);
    std::println("MoE A/B {:<8} t={:<5} {:.3f} ms  ({})", cutlass ? "cutlass" : "ggml", tokens, *ms,
                 impl);
  }
  (void)cudaFree(buffer);
  (void)cudaFree(inputs);
  return {};
}

// ------------------------------------------------------------------ main

struct Options {
  std::filesystem::path artifact;
  std::filesystem::path out;
  std::uint32_t context = 4096;
  std::uint32_t max_rows = 512;
  std::filesystem::path prompts;
  std::filesystem::path force;
  std::uint32_t generate = 0;
  std::filesystem::path ppl;
  std::vector<std::string> dump;
  bool layout_proof = false;
  bool ab = false;
  bool stepwise = false;
  bool state_roundtrip = false;
  bool unfused = false;
  bool exact = false;
  bool convert_experts = false;
  std::uint32_t bench_prefill = 0;
  std::uint32_t bench_decode = 0;
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
    const auto number = [&](std::uint32_t& into) -> Status {
      auto v = value();
      if (!v) {
        return std::unexpected(v.error());
      }
      const auto [end, ec] = std::from_chars(v->data(), v->data() + v->size(), into);
      if (ec != std::errc() || end != v->data() + v->size()) {
        return Error(std::format("{} takes a number", a));
      }
      return {};
    };
    Status ok;
    if (a == "--artifact" || a == "--out" || a == "--prompts" || a == "--force" || a == "--ppl" ||
        a == "--dump") {
      auto v = value();
      if (!v) {
        return std::unexpected(v.error());
      }
      if (a == "--artifact") {
        o.artifact = *v;
      } else if (a == "--out") {
        o.out = *v;
      } else if (a == "--prompts") {
        o.prompts = *v;
      } else if (a == "--force") {
        o.force = *v;
      } else if (a == "--ppl") {
        o.ppl = *v;
      } else {
        std::string names(*v);
        std::size_t start = 0;
        while (start <= names.size()) {
          const auto comma = names.find(',', start);
          const auto end = comma == std::string::npos ? names.size() : comma;
          if (end > start) {
            o.dump.push_back(names.substr(start, end - start));
          }
          start = end + 1;
        }
      }
    } else if (a == "--context") {
      ok = number(o.context);
    } else if (a == "--max-rows") {
      ok = number(o.max_rows);
    } else if (a == "--generate") {
      ok = number(o.generate);
    } else if (a == "--bench-prefill") {
      ok = number(o.bench_prefill);
    } else if (a == "--bench-decode") {
      ok = number(o.bench_decode);
    } else if (a == "--layout-proof") {
      o.layout_proof = true;
    } else if (a == "--ab") {
      o.ab = true;
    } else if (a == "--stepwise") {
      o.stepwise = true;
    } else if (a == "--state-roundtrip") {
      o.state_roundtrip = true;
    } else if (a == "--unfused") {
      o.unfused = true;
    } else if (a == "--exact") {
      o.exact = true;
    } else if (a == "--convert-experts") {
      o.convert_experts = true;
    } else {
      return Error(std::format("unknown argument {}", a));
    }
    if (!ok) {
      return std::unexpected(ok.error());
    }
  }
  if (o.artifact.empty() || o.out.empty()) {
    return Error("--artifact and --out are required");
  }
  return o;
}

Status Run(const Options& o) {
  const std::uint64_t available_before = MemAvailable();
  MemorySampler memory;
  auto device = Device::Open();
  if (!device) {
    return std::unexpected(device.error());
  }
  Device& d = **device;
  auto artifact = jitllm::artifact::Artifact::Open(o.artifact);
  if (!artifact) {
    return Error(std::format("the artifact was refused: {}", artifact.error().reason));
  }
  const md::Qwen38Profile& profile = md::Qwen38Flash();
  auto binding = md::BindQwen38(profile, *artifact);
  if (!binding) {
    return std::unexpected(binding.error());
  }
  // What each layout takes: GGML's products, the layout proof and the
  // kernel A/B's NVFP4 rows read GGML's layout; the conversion writes the
  // CUTLASS layout from it.
  if (binding->cutlass() && (o.convert_experts || o.unfused || o.layout_proof || o.ab)) {
    return Error(
        "the artifact's experts are in the CUTLASS layout: --convert-experts, "
        "--unfused, --layout-proof and --ab take an artifact in GGML's layout");
  }
  if (o.convert_experts && o.unfused) {
    return Error("--unfused reads GGML's layout, which --convert-experts replaces");
  }
  // A GGUF checkpoint's experts are GGML's types: no CUTLASS conversion,
  // and the kernel A/B's NVFP4 and MXFP8 operands are the ModelOpt
  // artifact's.
  if (binding->gguf() && (o.convert_experts || o.ab)) {
    return Error("--convert-experts and --ab take the ModelOpt artifact, not a GGUF one");
  }
  // (The fast graph builds no tensor of every cell by every row: RE-037's
  // chunk bound is the reference and unfused forms'.)
  auto state = md::Qwen38State(profile, o.context, o.max_rows, o.unfused || o.exact);
  if (!state) {
    return std::unexpected(state.error());
  }
  Weights weights;
  if (auto r = PlaceWeights(*artifact, profile, *binding, weights); !r) {
    return r;
  }
  if (auto r = LoadWeights(*artifact, d, weights); !r) {
    return r;
  }
  auto hash = ReadPleHash(*artifact, weights, profile, *binding);
  if (!hash) {
    return std::unexpected(hash.error());
  }
  std::println("loaded {} bytes in {:.2f} s ({:.2f} GB/s); dense {} B, slabs {} B (padding {} B)",
               weights.bytes_read, weights.load_seconds,
               static_cast<double>(weights.bytes_read) / weights.load_seconds / 1e9,
               weights.dense_bytes, weights.slab_bytes, weights.slab_padding);
  void* state_region = nullptr;
  if (auto r = Cuda(cudaMalloc(&state_region, state->bytes), "the state"); !r) {
    return r;
  }
  Model model{.artifact = &*artifact,
              .profile = &profile,
              .binding = &*binding,
              .weights = &weights,
              .state = &*state,
              .hash = &*hash,
              .state_base = Address(state_region),
              .fused = !o.unfused,
              .exact = o.exact,
              .cutlass = binding->cutlass() || o.convert_experts};

  int major = 0;
  int minor = 0;
  (void)cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0);
  (void)cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, 0);
  const std::uint64_t cublas_bytes =
      kg::CublasHandle::UpstreamWorkspace((100 * major) + (10 * minor)).value();
  void* cublas_workspace = nullptr;
  if (auto r = Cuda(cudaMalloc(&cublas_workspace, cublas_bytes), "the cuBLAS workspace"); !r) {
    return r;
  }
  auto cublas =
      kg::CublasHandle::Create(0, d.execution(), d.stream(),
                               {.base = Address(cublas_workspace), .size = Bytes(cublas_bytes)});
  if (!cublas) {
    return Error(cublas.error().detail);
  }

  // Size the activations and the pool by the largest shapes: a full chunk at
  // the start and at the end of the context, all rows out, and decode steps.
  std::uint64_t most_activations = 0;
  std::uint64_t most_scratch = 0;
  std::uint64_t most_inputs = 0;
  {
    auto measure = kg::LaunchContext::Create(0, d.execution(), d.stream(),
                                             {.base = 0, .size = Bytes(0)}, cublas->get());
    if (!measure) {
      return Error(measure.error().detail);
    }
    const kg::DeviceChoices choices = kg::DeviceChoicesOf(**measure);
    const std::array<std::pair<std::uint32_t, std::uint32_t>, 4> probes = {
        {{0, o.max_rows}, {o.context - o.max_rows, o.max_rows}, {o.context - 1, 1}, {0, 1}}};
    for (const auto& [n_past, rows] : probes) {
      std::vector<std::int32_t> history(std::size_t{n_past} + rows, 1000);
      auto in = md::Qwen38Chunk(profile, *state, *hash, history, n_past, rows);
      if (!in) {
        return std::unexpected(in.error());
      }
      auto planned = PlanChunk(model, kg::Qwen38ShapeOf(*state, *in, rows), choices, o.dump, 0, 0);
      if (!planned) {
        return Error(
            std::format("measuring a chunk of {} at {}: {}", rows, n_past, planned.error()));
      }
      most_activations = std::max(most_activations, (*planned)->placement.extent);
      auto scratch = kg::PlanScratch(**measure, (*planned)->plan);
      if (!scratch) {
        return Error(std::format("the scratch of a chunk of {} at {}: {}{}", rows, n_past,
                                 scratch.error().detail, FailingStep(**measure, (*planned)->plan)));
      }
      most_scratch = std::max(most_scratch, *scratch);
      most_inputs = std::max(most_inputs, (*planned)->inputs_bytes);
      std::println("shape rows {} at {}: {} steps, activations {} B, scratch {} B", rows, n_past,
                   (*planned)->plan.steps.size(), (*planned)->placement.extent, *scratch);
    }
  }
  const std::uint64_t activation_bytes = Round(most_activations + (most_activations / 4), 1 << 21);
  const std::uint64_t scratch_bytes = Round(most_scratch + (most_scratch / 4) + (1 << 20), 1 << 21);
  const std::uint64_t staging_bytes = Round((most_inputs * 2) + (1 << 20), 1 << 21);
  void* activations = nullptr;
  void* scratch = nullptr;
  void* staging = nullptr;
  if (auto r = Cuda(cudaMalloc(&activations, activation_bytes), "the activations"); !r) {
    return r;
  }
  if (auto r = Cuda(cudaMalloc(&scratch, scratch_bytes), "the pool"); !r) {
    return r;
  }
  if (auto r = Cuda(cudaMallocHost(&staging, staging_bytes), "the input staging"); !r) {
    return r;
  }
  auto launch = kg::LaunchContext::Create(0, d.execution(), d.stream(),
                                          {.base = Address(scratch), .size = Bytes(scratch_bytes)},
                                          cublas->get());
  if (!launch) {
    return Error(launch.error().detail);
  }
  auto registry = jitllm::execution::Registry::Create(kg::Implementations());
  if (!registry) {
    return Error(registry.error().detail);
  }
  Runner runner(d, model, **launch, *registry, Address(activations), activation_bytes, staging,
                staging_bytes);
  std::filesystem::create_directories(o.out);
  std::string summary = std::format(
      R"({{"artifact":"{}","context":{},"max_rows":{},"load_seconds":{:.3f},"bytes_read":{},"dense_bytes":{},"slab_bytes":{},"slab_padding":{},"expert_stride":{},"state_bytes":{},"activation_bytes":{},"scratch_bytes":{})",
      artifact->id(), o.context, o.max_rows, weights.load_seconds, weights.bytes_read,
      weights.dense_bytes, weights.slab_bytes, weights.slab_padding, weights.stride[0],
      state->bytes, activation_bytes, scratch_bytes);

  if (o.ab) {
    std::string report;
    if (auto r = KernelAb(model, d, **launch, *registry, report); !r) {
      return r;
    }
    summary += R"(,"ab":)" + report;
  }
  if (o.layout_proof) {
    std::string report;
    if (auto r = LayoutProof(model, d, **launch, *registry, report); !r) {
      return r;
    }
    summary += R"(,"layout_proof":)" + report;
  }
  if (o.convert_experts) {
    // The MoE A/B's GGML side while the slabs hold GGML's layout, then the
    // conversion (proved with --layout-proof) and the CUTLASS side.
    std::string moe_ab;
    if (o.ab) {
      if (auto r = MoeAb(model, d, **launch, *registry, false, moe_ab); !r) {
        return r;
      }
    }
    std::string report;
    if (auto r = ConvertExperts(model, d, o.layout_proof, report); !r) {
      return r;
    }
    summary += R"(,"cutlass_layout":)" + report;
    if (o.ab) {
      if (auto r = MoeAb(model, d, **launch, *registry, true, moe_ab); !r) {
        return r;
      }
      summary += R"(,"moe_ab":[)" + moe_ab + "]";
    }
  }

  const std::uint32_t vocab = profile.vocab;
  if (!o.prompts.empty()) {
    auto prompts = ReadTokenLines(o.prompts, vocab);
    if (!prompts) {
      return std::unexpected(prompts.error());
    }
    std::vector<TokenLine> forced;
    if (!o.force.empty()) {
      auto f = ReadTokenLines(o.force, vocab);
      if (!f) {
        return std::unexpected(f.error());
      }
      forced = std::move(*f);
      if (forced.size() != prompts->size()) {
        return Error("--force needs a line per prompt");
      }
    }
    summary += R"(,"prompts":[)";
    std::string state_roundtrip_report;
    std::vector<float> logits;
    for (std::size_t pi = 0; pi < prompts->size(); ++pi) {
      const TokenLine& prompt = (*prompts)[pi];
      if (!forced.empty() &&
          (forced[pi].name != prompt.name || forced[pi].ids.size() < o.generate)) {
        return Error(std::format("--force has no {} tokens for {}", o.generate, prompt.name));
      }
      if (auto r = runner.Clear(); !r) {
        return r;
      }
      std::vector<std::int32_t> history = prompt.ids;
      std::map<std::string, std::vector<float>> kept;
      const bool dump = pi == 0 && !o.dump.empty();
      const auto t0 = Clock::now();
      if (o.stepwise) {
        // The prompt one token at a time (decode's kernels throughout).
        for (std::uint32_t at = 0; at < history.size(); ++at) {
          if (auto r = runner.Chunk(std::span(history).first(std::size_t{at} + 1), at, 1, logits);
              !r) {
            return Error(std::format("{} (stepwise) at {}: {}", prompt.name, at, r.error()));
          }
        }
      } else {
        if (dump && history.size() > o.max_rows) {
          return Error("--dump needs a first prompt of one chunk");
        }
        // A prompt longer than --max-rows (long context) is prefilled in
        // chunks of ChunkRows; the last chunk's last row gives the first
        // token.
        for (std::uint32_t at = 0, rows = 0; at < history.size(); at += rows) {
          rows = ChunkRows(history.size() - at, o.max_rows);
          if (auto r = runner.Chunk(std::span(history).first(std::size_t{at} + rows), at, 1, logits,
                                    dump ? std::span(o.dump) : std::span<const std::string>{},
                                    dump ? &kept : nullptr);
              !r) {
            return Error(std::format("{} at {}: {}", prompt.name, at, r.error()));
          }
        }
      }
      const double prefill = Seconds(Clock::now() - t0);
      if (dump) {
        std::filesystem::create_directories(o.out / "dump");
        for (const auto& [name, values] : kept) {
          if (auto r = WriteFloats(o.out / "dump" / (name + ".f32"), values); !r) {
            return r;
          }
        }
      }
      std::vector<float> steps(logits.begin(), logits.end());
      std::vector<std::int32_t> argmax = {Argmax(logits)};
      double decode = 0;
      for (std::uint32_t k = 1; k < o.generate; ++k) {
        const std::int32_t next = forced.empty() ? argmax.back() : forced[pi].ids[k - 1];
        history.push_back(next);
        const auto t1 = Clock::now();
        if (auto r =
                runner.Chunk(history, static_cast<std::uint32_t>(history.size() - 1), 1, logits);
            !r) {
          return Error(std::format("{} step {}: {}", prompt.name, k, r.error()));
        }
        decode += Seconds(Clock::now() - t1);
        steps.insert(steps.end(), logits.begin(), logits.end());
        argmax.push_back(Argmax(logits));
      }
      std::uint64_t roundtrip_differing = 0;
      if (o.state_roundtrip && o.generate > 2) {
        // The same prompt and steps again, the whole state copied to the
        // host, the device copy overwritten, and restored halfway: every
        // logit must be the uninterrupted run's, bit for bit.
        if (auto r = runner.Clear(); !r) {
          return r;
        }
        std::vector<std::int32_t> again = prompt.ids;
        std::vector<float> steps2;
        if (auto r = runner.Chunk(again, 0, 1, logits); !r) {
          return r;
        }
        steps2.insert(steps2.end(), logits.begin(), logits.end());
        for (std::uint32_t k = 1; k < o.generate; ++k) {
          if (k == o.generate / 2) {
            if (auto r = runner.RoundTripState(); !r) {
              return r;
            }
          }
          again.push_back(history[prompt.ids.size() + k - 1]);
          if (auto r = runner.Chunk(again, static_cast<std::uint32_t>(again.size() - 1), 1, logits);
              !r) {
            return r;
          }
          steps2.insert(steps2.end(), logits.begin(), logits.end());
        }
        for (std::size_t i = 0; i < steps.size(); ++i) {
          roundtrip_differing +=
              std::bit_cast<std::uint32_t>(steps[i]) != std::bit_cast<std::uint32_t>(steps2[i]);
        }
        std::println("{}: state round trip at step {}: {} of {} logits differ", prompt.name,
                     o.generate / 2, roundtrip_differing, steps.size());
        state_roundtrip_report +=
            std::format(R"({}{{"name":"{}","logits":{},"differing":{}}})", pi == 0 ? "" : ",",
                        prompt.name, steps.size(), roundtrip_differing);
      }
      if (auto r = WriteFloats(o.out / (prompt.name + ".logits.f32"), steps); !r) {
        return r;
      }
      std::string ids;
      for (std::size_t i = 0; i < argmax.size(); ++i) {
        ids += std::format("{}{}", i == 0 ? "" : ",", argmax[i]);
      }
      summary += std::format(
          R"({}{{"name":"{}","prompt_tokens":{},"argmax":[{}],"prefill_seconds":{:.4f},"decode_seconds":{:.4f},"decode_steps":{}}})",
          pi == 0 ? "" : ",", prompt.name, prompt.ids.size(), ids, prefill, decode,
          o.generate > 0 ? o.generate - 1 : 0);
      std::println("{}: {} prompt tokens, prefill {:.3f} s, {} decode steps {:.3f} s", prompt.name,
                   prompt.ids.size(), prefill, o.generate > 0 ? o.generate - 1 : 0, decode);
    }
    summary += "]";
    if (o.state_roundtrip) {
      summary += R"(,"state_roundtrip":[)" + state_roundtrip_report + "]";
    }
  }

  if (!o.ppl.empty()) {
    auto lines = ReadTokenLines(o.ppl, vocab);
    if (!lines || lines->size() != 1) {
      return Error("--ppl takes one line of token ids");
    }
    const std::vector<std::int32_t>& ids = lines->front().ids;
    if (ids.size() < 2 || ids.size() > o.context) {
      return Error("the perplexity text does not fit the context");
    }
    if (auto r = runner.Clear(); !r) {
      return r;
    }
    std::vector<double> nll;
    std::vector<std::int32_t> top1;
    std::vector<float> logits;
    const auto t0 = Clock::now();
    for (std::uint32_t at = 0, rows = 0; at < ids.size(); at += rows) {
      rows = ChunkRows(ids.size() - at, o.max_rows);
      if (auto r = runner.Chunk(std::span(ids).first(std::size_t{at} + rows), at, rows, logits);
          !r) {
        return Error(std::format("perplexity chunk at {}: {}", at, r.error()));
      }
      for (std::uint32_t i = 0; i < rows; ++i) {
        const std::size_t next = std::size_t{at} + i + 1;
        if (next < ids.size()) {
          const auto row = std::span(logits).subspan(std::size_t{i} * vocab, vocab);
          nll.push_back(Nll(row, ids[next]));
          top1.push_back(Argmax(row));
        }
      }
    }
    const double seconds = Seconds(Clock::now() - t0);
    const double mean =
        std::accumulate(nll.begin(), nll.end(), 0.0) / static_cast<double>(nll.size());
    {
      std::ofstream out(o.out / "ppl.nll.f64", std::ios::binary);
      out.write(reinterpret_cast<const char*>(nll.data()),
                static_cast<std::streamsize>(nll.size() * sizeof(double)));
      std::ofstream top(o.out / "ppl.top1.i32", std::ios::binary);
      top.write(reinterpret_cast<const char*>(top1.data()),
                static_cast<std::streamsize>(top1.size() * sizeof(std::int32_t)));
    }
    summary += std::format(
        R"(,"ppl":{{"tokens":{},"scored":{},"mean_nll":{:.6f},"ppl":{:.6f},"seconds":{:.3f}}})",
        ids.size(), nll.size(), mean, std::exp(mean), seconds);
    std::println("perplexity {:.4f} over {} tokens ({:.2f} s)", std::exp(mean), nll.size(),
                 seconds);
  }

  if (o.bench_prefill > 0) {
    std::vector<std::int32_t> ids(o.bench_prefill);
    for (std::size_t i = 0; i < ids.size(); ++i) {
      ids[i] = static_cast<std::int32_t>(1000 + ((i * 7919) % 100000));
    }
    std::vector<float> logits;
    double prefill_best = 1e30;
    for (int rep = 0; rep < 3; ++rep) {
      if (auto r = runner.Clear(); !r) {
        return r;
      }
      const auto t0 = Clock::now();
      for (std::uint32_t at = 0, rows = 0; at < ids.size(); at += rows) {
        rows = ChunkRows(ids.size() - at, o.max_rows);
        if (auto r = runner.Chunk(std::span(ids).first(std::size_t{at} + rows), at, 1, logits);
            !r) {
          return r;
        }
      }
      prefill_best = std::min(prefill_best, Seconds(Clock::now() - t0));
    }
    double decode_total = 0;
    for (int rep = -1; rep < 3 && o.bench_decode > 0; ++rep) {
      if (auto r = runner.Clear(); !r) {
        return r;
      }
      std::vector<std::int32_t> history = {ids[0]};
      const auto t1 = Clock::now();
      for (std::uint32_t k = 0; k < o.bench_decode; ++k) {
        if (auto r = runner.Chunk(history, k, 1, logits); !r) {
          return r;
        }
        history.push_back(Argmax(logits));
      }
      if (rep >= 0) {
        decode_total += Seconds(Clock::now() - t1) / 3.0;
      }
    }
    summary += std::format(
        R"(,"bench":{{"prefill_tokens":{},"prefill_seconds_best_of_3":{:.4f},"prefill_tok_s":{:.2f},"decode_steps":{},"decode_seconds":{:.4f},"decode_tok_s":{:.3f}}})",
        ids.size(), prefill_best, static_cast<double>(ids.size()) / prefill_best, o.bench_decode,
        decode_total, o.bench_decode > 0 ? o.bench_decode / decode_total : 0.0);
    std::println(
        "bench: prefill {} tokens {:.3f} s ({:.1f} tok/s); decode {} steps {:.3f} s "
        "({:.2f} tok/s)",
        ids.size(), prefill_best, static_cast<double>(ids.size()) / prefill_best, o.bench_decode,
        decode_total, o.bench_decode > 0 ? o.bench_decode / decode_total : 0.0);
  }

  const std::uint64_t low = memory.low();
  summary += std::format(
      R"(,"plans_made":{},"most_activations":{},"most_scratch":{},"mem_available_before":{},"mem_available_low":{},"peak_bytes_by_mem_available":{}}})",
      runner.plans_made(), runner.most_activations(), runner.most_scratch(), available_before, low,
      available_before > low ? available_before - low : 0);
  std::ofstream(o.out / "summary.json") << summary << "\n";
  std::println(
      "peak by MemAvailable: {:.2f} GiB",
      static_cast<double>(available_before > low ? available_before - low : 0) / (1ULL << 30U));
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
    std::println(stderr, "jitllm_qwen38_exec: {}", r.error());
    return 1;
  }
  return 0;
}
