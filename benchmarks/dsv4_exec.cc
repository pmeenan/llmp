// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// M3's first model slice (docs/experiments/dsv4-native/README.md): DeepSeek
// V4 Flash run natively and resident on one Spark from its v0 prepared
// artifact, for the correctness comparison with llama.cpp on the same GGUF
// and a coarse speed and memory report.
//
//   jitllm_dsv4_exec --artifact DIR --out DIR [--context N] [--max-rows N]
//                    [--prompts FILE --generate N [--force FILE]]
//                    [--ppl FILE] [--dump NAMES] [--layout-proof]
//                    [--bench-prefill N --bench-decode N] [--exact on|off]
//                    [--compact-experts]
//                    [--probe-step N]
//
// - --exact on: the reference mode, the graph node for node as llama.cpp
//   builds it and planned unfused (its logits llama.cpp's with fusion off,
//   bit for bit); off (the default), jitLLM's fast plan (dsv4_graph.h
//   Dsv4GraphOptions::fused), judged coarsely against llama.cpp; its window
//   cache a ring (model/dsv4.h Dsv4Window::kRing), the reference mode's the
//   full cache.
// - --probe-step N (with --force): the first prompt's step N (the argmax at
//   index N) run from the state before it in both plans, the state put back
//   after each, every named tensor and the logits written under
//   OUT/probe/{fast,exact}/ (docs/experiments/dsv4-decode's probe method,
//   resident); the state a full window, which both plans read.
// - Weights: the artifact is opened as untrusted input (artifact.h), its
//   resources and expert arrays bound to the compiled-in DeepSeek V4 profile
//   (model/dsv4.h), and every chunk read with direct I/O, one coalesced read
//   plan, through pinned staging into device memory: the dense groups in one
//   region, and each layer's routed experts in a slab where expert e's group
//   sits at e·S, S the smallest multiple of the layer's expert block sizes
//   (and 16 bytes) that holds a group: the resident expert layout, which
//   GGML's mul_mat_id addresses at a uniform stride with its stock kernels.
//   The token table stays on the host, where the embedding rows are looked
//   up and dequantized with GGML's own row function, as llama.cpp's CPU
//   backend looks them up. Chunk digests are not recomputed (the import
//   verified them; page-in never hashes, D-056). The hash-routed layers'
//   token-to-expert tables, which the kernels index with unchecked, must
//   name only experts (model/dsv4.h CheckDsv4HashRouting) before any chunk.
// - Each chunk: the host builds its inputs (model/dsv4.h Dsv4Chunk), the
//   GGML graph is built as llama.cpp builds it (kernels/ggml/dsv4_graph.h),
//   planned with fusion off, its activations placed (graph_plan.h), bound
//   to the registry's implementations (executor.h) and run on one stream
//   through the K-C launch context. Graphs, plans and bindings are kept per
//   chunk shape, so decode steps of one shape reuse them.
// - --prompts: each line `name<TAB>ids...`; each prompt from a cleared state
//   in prefill chunks of --max-rows (one chunk when it fits; from 1,024
//   rows whole 8-row tiles, RE-036), then --generate tokens one at a time.
//   Without
//   --force the next token is the argmax (greedy); with --force (lines of
//   the same names) the given tokens are fed instead, and the argmax is
//   still recorded. Every step's logits are written.
// - --ppl: one line of ids, evaluated from a cleared state in chunks of
//   --max-rows; the negative log-likelihood of every token after the first.
// - --dump: comma-separated llama.cpp callback names ("l_last-0", ...) of
//   the first prompt's prefill, kept alive and written as F32.
// - --layout-proof: every layer's routed products at 1, 5 and 64 tokens
//   with random activations and routes, over the slab and over the
//   reference layout (the GGUF's [k, n, experts] tensor, rebuilt from the
//   artifact's slices, which the import copied byte for byte), compared bit
//   for bit.
// - --bench-prefill/--bench-decode: a prompt of that many synthetic tokens
//   (the best of three), and that many decode steps from an empty context
//   after a warm-up (the mean of three), as llama-bench's pp and tg run.

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
#include <cstdlib>
#include <cstring>
#include <expected>
#include <filesystem>
#include <format>
#include <fstream>
#include <iterator>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <print>
#include <random>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include "artifact/artifact.h"
#include "artifact/layout.h"
#include "base/bytes.h"
#include "execution/registry.h"
#include "ggml.h"
#include "kernels/ggml/cublas.h"
#include "kernels/ggml/dsv4_graph.h"
#include "kernels/ggml/executor.h"
#include "kernels/ggml/graph_plan.h"
#include "kernels/ggml/implementations.h"
#include "kernels/ggml/launch.h"
#include "kernels/ggml/ops_ext.h"
#include "kernels/ggml/tensors.h"
#include "model/dsv4.h"
#include "providers/cuda/cuda_device_execution.h"
#include "providers/device_execution.h"

namespace {

namespace kg = jitllm::kernels::ggml;
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

// A chunk's rows: at most `max_rows`, and from 1,024 rows whole 8-row tiles
// (RE-036: GGML's mask pre-pass reads whole tiles there), the rest left to
// the next chunk.
std::uint32_t ChunkRows(std::size_t remaining, std::uint32_t max_rows) {
  auto rows = static_cast<std::uint32_t>(std::min<std::size_t>(max_rows, remaining));
  if (rows >= 1024 && rows % 8 != 0) {
    rows -= rows % 8;
  }
  return rows;
}

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

std::uint64_t Lcm(std::uint64_t a, std::uint64_t b) { return a / std::gcd(a, b) * b; }

struct Weights {
  // Dense groups: one region; each group's offset in it.
  std::uint64_t dense = 0;
  std::uint64_t dense_bytes = 0;
  std::vector<std::uint64_t> group_address;  // every group's device address (0: host only)
  // Each layer's expert slab and stride.
  std::vector<std::uint64_t> slab;
  std::vector<std::uint64_t> stride;
  std::uint64_t slab_bytes = 0;
  std::uint64_t slab_padding = 0;  // bytes of stride beyond the groups' stored bytes
  // The token table, on the host.
  std::vector<std::byte> table;
  std::uint32_t table_group = 0;
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

Status PlaceWeights(const jitllm::artifact::Artifact& artifact, const md::Dsv4Profile& profile,
                    const md::Dsv4Binding& binding, Weights& w) {
  const auto groups = artifact.groups();
  w.group_address.assign(groups.size(), 0);
  w.table_group = artifact.resources()[binding.token_embd.index].group;
  // The table's group stays on the host: a device tensor bound into it
  // would have no device address.
  for (std::uint32_t r = 0; r < artifact.resources().size(); ++r) {
    if (r != binding.token_embd.index && artifact.resources()[r].group == w.table_group) {
      return Error(std::format("{} shares the token table's group", artifact.resources()[r].name));
    }
  }
  // Expert slabs: the layer's arrays share their groups (artifact.h).
  w.slab.assign(profile.layers, 0);
  w.stride.assign(profile.layers, 0);
  std::vector<std::uint32_t> first(profile.layers, 0);
  std::vector<std::uint64_t> group_bytes(profile.layers, 0);
  for (std::uint32_t il = 0; il < profile.layers; ++il) {
    const md::Dsv4Layer& l = binding.layers[il];
    std::uint64_t unit = 16;
    std::uint64_t stored = 0;
    std::optional<std::uint32_t> first_group;
    for (const md::Dsv4Tensor* t : {&l.gate_exps, &l.up_exps, &l.down_exps}) {
      const auto& a = artifact.expert_arrays()[t->index];
      auto type = kg::GgmlTypeOf(t->type);
      if (!type) {
        return Error(type.error().detail);
      }
      unit = Lcm(unit, ggml_type_size(*type));
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
  // Dense groups at 256-byte aligned offsets.
  std::vector<bool> expert(groups.size(), false);
  for (std::uint32_t il = 0; il < profile.layers; ++il) {
    for (std::uint32_t e = 0; e < profile.experts; ++e) {
      expert[first[il] + e] = true;
    }
  }
  if (expert[w.table_group]) {
    return Error("the token table's group is an expert group");
  }
  std::vector<std::uint64_t> offset(groups.size(), 0);
  for (std::size_t g = 0; g < groups.size(); ++g) {
    if (expert[g] || g == w.table_group) {
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
    if (!expert[g] && g != w.table_group) {
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
    // The stride's bytes past each group are never read as weights (every
    // slice and its readable bytes lie in its group); zeroed all the same.
    if (w.stride[il] > group_bytes[il]) {
      if (auto r = Cuda(cudaMemset2D(Pointer(w.slab[il] + group_bytes[il]), w.stride[il], 0,
                                     w.stride[il] - group_bytes[il], profile.experts),
                        "zeroing the slab's padding");
          !r) {
        return r;
      }
    }
  }
  if (auto r = Cuda(cudaDeviceSynchronize(), "zeroing the slabs' padding"); !r) {
    return r;
  }
  const auto& table = groups[w.table_group];
  w.table.assign(table.stored.value(), std::byte{0});
  return {};
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
    // Double-buffered: the buffer's last copies must have finished.
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
      const std::span<const std::byte> piece(bytes + at, segment.length.value());
      if (segment.chunk.group == w.table_group) {
        std::memcpy(w.table.data() + segment.group_offset.value(), piece.data(), piece.size());
      } else {
        const std::uint64_t dst =
            w.group_address[segment.chunk.group] + segment.group_offset.value();
        if (auto r = Cuda(cudaMemcpyAsync(Pointer(dst), piece.data(), piece.size(),
                                          cudaMemcpyHostToDevice, *stream),
                          "a weight upload");
            !r) {
          return r;
        }
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

// ------------------------------------------------------------------ chunks

struct Planned {
  std::optional<kg::TensorArena> arena;
  kg::Dsv4Graph graph;
  kg::GraphPlan plan;
  kg::Placement placement;
  std::optional<kg::BoundGraph> bound;
  std::uint64_t scratch = 0;
  std::uint64_t inputs_bytes = 0;
};

struct Model {
  const jitllm::artifact::Artifact* artifact = nullptr;
  const md::Dsv4Profile* profile = nullptr;
  const md::Dsv4Binding* binding = nullptr;
  const Weights* weights = nullptr;
  const md::Dsv4StateLayout* state = nullptr;
  std::uint64_t state_base = 0;
  std::vector<float> rot;  // the indexer's Hadamard matrix
  // The reference mode (--exact on): llama.cpp's graph planned unfused;
  // off, the fast plan (dsv4_graph.h Dsv4GraphOptions::fused).
  bool exact = false;
  bool compact_experts = false;
};

void BindWeights(const Model& m, kg::Dsv4Graph& g) {
  const auto& a = *m.artifact;
  const auto& w = *m.weights;
  const auto bind = [&](ggml_tensor* t, const md::Dsv4Tensor& r) {
    if (t != nullptr) {
      kg::TensorArena::Bind(t, ResourceAddress(a, w, r.index));
    }
  };
  const auto& b = *m.binding;
  bind(g.output_norm, b.output_norm);
  bind(g.output, b.output);
  bind(g.hc_head_fn, b.hc_head_fn);
  bind(g.hc_head_base, b.hc_head_base);
  bind(g.hc_head_scale, b.hc_head_scale);
  using K = md::Dsv4StateTensor::Kind;
  for (std::uint32_t il = 0; il < m.profile->layers; ++il) {
    const md::Dsv4Layer& r = b.layers[il];
    kg::Dsv4LayerTensors& l = g.layers[il];
    bind(l.attn_norm, r.attn_norm);
    bind(l.attn_sinks, r.attn_sinks);
    bind(l.q_a, r.q_a);
    bind(l.q_a_norm, r.q_a_norm);
    bind(l.q_b, r.q_b);
    bind(l.kv, r.kv);
    bind(l.kv_norm, r.kv_norm);
    bind(l.out_a, r.out_a);
    bind(l.out_b, r.out_b);
    bind(l.hc_attn_fn, r.hc_attn_fn);
    bind(l.hc_attn_base, r.hc_attn_base);
    bind(l.hc_attn_scale, r.hc_attn_scale);
    bind(l.hc_ffn_fn, r.hc_ffn_fn);
    bind(l.hc_ffn_base, r.hc_ffn_base);
    bind(l.hc_ffn_scale, r.hc_ffn_scale);
    bind(l.comp_kv, r.comp_kv);
    bind(l.comp_gate, r.comp_gate);
    bind(l.comp_ape, r.comp_ape);
    bind(l.comp_norm, r.comp_norm);
    bind(l.idx_q_b, r.idx_q_b);
    bind(l.idx_proj, r.idx_proj);
    bind(l.idx_comp_kv, r.idx_comp_kv);
    bind(l.idx_comp_gate, r.idx_comp_gate);
    bind(l.idx_comp_ape, r.idx_comp_ape);
    bind(l.idx_comp_norm, r.idx_comp_norm);
    bind(l.ffn_norm, r.ffn_norm);
    bind(l.router, r.router);
    bind(l.router_bias, r.router_bias);
    bind(l.tid2eid, r.tid2eid);
    bind(l.up_shexp, r.up_shexp);
    bind(l.gate_shexp, r.gate_shexp);
    bind(l.down_shexp, r.down_shexp);
    kg::TensorArena::Bind(l.up_exps, ArrayAddress(a, w, r.up_exps.index));
    kg::TensorArena::Bind(l.gate_exps, ArrayAddress(a, w, r.gate_exps.index));
    kg::TensorArena::Bind(l.down_exps, ArrayAddress(a, w, r.down_exps.index));
    const auto state = [&](ggml_tensor* t, K kind) {
      // A view (a ring's compressed cache) is bound with its window's tensor.
      if (t == nullptr || t->view_src != nullptr) {
        return;
      }
      const std::int64_t i = m.state->Find(il, kind);
      kg::TensorArena::Bind(t, m.state_base + m.state->tensors[static_cast<std::size_t>(i)].offset);
    };
    state(l.raw_k, K::kRawK);
    state(l.csa_k, K::kCsaK);
    state(l.csa_state_kv, K::kCsaStateKv);
    state(l.csa_state_score, K::kCsaStateScore);
    state(l.lid_k, K::kLidK);
    state(l.lid_state_kv, K::kLidStateKv);
    state(l.lid_state_score, K::kLidStateScore);
    state(l.hca_k, K::kHcaK);
    state(l.hca_state_kv, K::kHcaStateKv);
    state(l.hca_state_score, K::kHcaStateScore);
  }
}

// Builds, binds, plans and places one chunk shape's graph; `activations` is
// the region its computed tensors and inputs live in (0 to measure).
std::expected<std::unique_ptr<Planned>, std::string> PlanChunk(
    const Model& m, const kg::Dsv4ChunkShape& shape, const kg::DeviceChoices& choices_in,
    std::span<const std::string> keep_names, std::uint64_t activations,
    std::uint64_t activation_bytes) {
  auto out = std::make_unique<Planned>();
  auto arena = kg::TensorArena::Create(kg::Dsv4GraphTensors(*m.profile));
  if (!arena) {
    return Error(arena.error().detail);
  }
  out->arena.emplace(std::move(*arena));
  auto graph = kg::BuildDsv4Graph(*out->arena, *m.profile, *m.binding, shape,
                                  {.expert_stride = m.weights->stride, .fused = !m.exact});
  if (!graph) {
    return Error(graph.error().detail);
  }
  out->graph = std::move(*graph);
  kg::Dsv4Graph& g = out->graph;
  BindWeights(m, g);
  kg::DeviceChoices device = choices_in;
  device.fuse_norms = !m.exact;
  device.vector_floats = !m.exact;
  device.pair_experts = !m.exact;
  device.compact_experts = !m.exact && m.compact_experts;
  device.wide_sparse_attention = !m.exact;
  const kg::DeviceChoices& choices = device;
  std::vector<ggml_tensor*> keep;
  for (const std::string& name : keep_names) {
    if (name == "*") {
      for (const auto& [n, t] : g.named) {
        keep.push_back(t);
      }
    } else if (ggml_tensor* t = g.Named(name); t != nullptr) {
      keep.push_back(t);
    } else {
      return Error(std::format("the graph names no {}", name));
    }
  }
  // First pass: every computed tensor at its own address.
  constexpr std::uint64_t kDistinct = std::uint64_t{1} << 46U;
  std::uint64_t leaf = kDistinct - (std::uint64_t{1} << 40U);
  const auto inputs = g.inputs();
  for (ggml_tensor* input : inputs) {
    kg::TensorArena::Bind(input, leaf);
    leaf += Round(ggml_nbytes(input), 256) + 256;
  }
  kg::BindDistinct(g.nodes, kDistinct);
  auto first = kg::PlanGraph(g.nodes, /*fusion=*/false, choices);
  if (!first) {
    return Error(first.error().detail);
  }
  auto placement = kg::PlaceActivations(g.nodes, *first, inputs, 256, keep);
  if (!placement) {
    return Error(placement.error().detail);
  }
  out->placement = std::move(*placement);
  for (ggml_tensor* input : inputs) {
    out->inputs_bytes += ggml_nbytes(input);
  }
  if (activations == 0) {
    out->plan = std::move(*first);
    return out;
  }
  if (out->placement.extent > activation_bytes) {
    return Error(std::format("the activations ({} bytes) exceed their region ({} bytes)",
                             out->placement.extent, activation_bytes));
  }
  for (const auto& [tensor, offset] : out->placement.offsets) {
    kg::TensorArena::Bind(tensor, activations + offset);
  }
  kg::BindViews(g.nodes);
  auto second = kg::PlanGraph(g.nodes, false, choices);
  if (!second) {
    return Error(second.error().detail);
  }
  if (!kg::SamePlan(*first, *second)) {
    return Error("the plan changed once the activations were placed");
  }
  out->plan = std::move(*second);
  return out;
}

class Runner {
 public:
  Runner(Device& device, Model model, kg::LaunchContext& launch,
         const jitllm::execution::Registry& registry, std::uint64_t activations,
         std::uint64_t activation_bytes, void* staging, std::uint64_t staging_bytes)
      : d_(device),
        m_(std::move(model)),
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

  // Runs one chunk of `tokens` after n_past; its logits (rows x vocab) in
  // `logits`. With `keep`, the named tensors are kept (a separate plan).
  Status Chunk(std::uint32_t n_past, std::span<const std::int32_t> tokens,
               std::vector<float>& logits, std::span<const std::string> keep = {},
               std::map<std::string, std::vector<float>>* kept = nullptr) {
    const auto rows = static_cast<std::uint32_t>(tokens.size());
    auto in = md::Dsv4Chunk(*m_.profile, *m_.state, n_past, rows, m_.exact);
    if (!in) {
      return std::unexpected(in.error());
    }
    const kg::Dsv4ChunkShape shape = kg::Dsv4ShapeOf(*m_.state, *in);
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
    const kg::Dsv4Graph& g = p->graph;
    // The embedding rows, dequantized on the host.
    const md::Dsv4Tensor& table = m_.binding->token_embd;
    auto type = kg::GgmlTypeOf(table.type);
    if (!type) {
      return Error(type.error().detail);
    }
    const auto* traits = ggml_get_type_traits(*type);
    const std::uint64_t row_bytes = ggml_row_size(*type, m_.profile->width);
    const std::uint64_t table_offset = m_.artifact->resources()[table.index].offset.value();
    std::vector<float> embd(std::size_t{rows} * m_.profile->width);
    for (std::uint32_t i = 0; i < rows; ++i) {
      if (tokens[i] < 0 || std::cmp_greater_equal(tokens[i], m_.profile->vocab)) {
        return Error(std::format("token {} is outside the vocabulary", tokens[i]));
      }
      const std::byte* row = m_.weights->table.data() + table_offset +
                             (static_cast<std::uint64_t>(tokens[i]) * row_bytes);
      traits->to_float(row, embd.data() + (std::size_t{i} * m_.profile->width), m_.profile->width);
    }
    std::vector<std::int32_t> out_ids(rows);
    std::ranges::iota(out_ids, 0);
    const std::vector<float>& rot = m_.rot;
    // A ring's graph has no CSA or indexer mask, and reads the rows' visible
    // counts where the zero fill's source was (engine/dsv4_plan.cc).
    std::vector<std::uint16_t> zeros(
        g.top_k_zeros != nullptr ? static_cast<std::size_t>(ggml_nelements(g.top_k_zeros)) : 0, 0);
    std::vector<std::pair<ggml_tensor*, const void*>> sources;
    const auto comp = [&](const kg::Dsv4CompInputs& t, const md::Dsv4CompPlan& plan,
                          const std::vector<std::uint16_t>& mask) {
      sources.insert(sources.end(), {{t.state_pos, plan.state_pos.data()},
                                     {t.persist_src, plan.persist_src.data()},
                                     {t.persist_dst, plan.persist_dst.data()},
                                     {t.read_idxs, plan.read_idxs.data()},
                                     {t.write_idxs, plan.write_idxs.data()},
                                     {t.write_pos, plan.write_pos.data()}});
      if (t.mask != nullptr) {
        sources.emplace_back(t.mask, mask.data());
      }
    };
    sources = {{g.embd, embd.data()},
               {g.tokens, tokens.data()},
               {g.positions, in->positions.data()},
               {g.raw_k_idxs, in->raw_cells.data()},
               {g.raw_mask, in->raw_mask.data()},
               {g.out_ids, out_ids.data()}};
    comp(g.csa, in->csa, in->csa_mask);
    comp(g.hca, in->hca, in->hca_mask);
    comp(g.lid, in->lid, in->lid_mask);
    sources.emplace_back(g.lid_rot, rot.data());
    if (g.top_k_zeros != nullptr) {
      sources.emplace_back(g.top_k_zeros, zeros.data());
    } else {
      sources.emplace_back(g.csa_visible, in->csa.n_visible.data());
      sources.emplace_back(g.hca_visible, in->hca.n_visible.data());
    }
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
    const std::uint64_t logit_bytes = std::uint64_t{rows} * m_.profile->vocab * sizeof(float);
    logits.resize(std::size_t{rows} * m_.profile->vocab);
    if (auto r = Cuda(cudaMemcpyAsync(logits.data(), g.logits->data, logit_bytes,
                                      cudaMemcpyDeviceToHost, *stream),
                      "the logits copy");
        !r) {
      return r;
    }
    if (kept != nullptr) {
      std::vector<std::string> names(keep.begin(), keep.end());
      if (std::ranges::find(keep, std::string("*")) != keep.end()) {
        names.clear();
        for (const auto& [n, t] : g.named) {
          names.push_back(n);
        }
      }
      for (const std::string& name : names) {
        const ggml_tensor* t = g.Named(name);
        std::vector<float>& to = (*kept)[name];
        const auto n = static_cast<std::size_t>(ggml_nelements(t));
        to.assign(n, 0.0f);
        if (t->type == GGML_TYPE_F32 && ggml_is_contiguous(t)) {
          if (auto r = Cuda(cudaMemcpyAsync(to.data(), t->data, n * sizeof(float),
                                            cudaMemcpyDeviceToHost, *stream),
                            "a kept tensor");
              !r) {
            return r;
          }
        } else if (t->type == GGML_TYPE_I32 && ggml_is_contiguous(t)) {
          std::vector<std::int32_t> ids(n);
          if (auto r = Cuda(
                  cudaMemcpy(ids.data(), t->data, n * sizeof(std::int32_t), cudaMemcpyDeviceToHost),
                  "a kept tensor");
              !r) {
            return r;
          }
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

  // The reference mode on or off for the next chunks, every plan dropped
  // (--probe-step: both plans over one full-window state).
  void SetExact(bool on) {
    m_.exact = on;
    cache_.clear();
  }
  int plans_made() const { return plans_made_; }
  std::uint64_t most_scratch() const { return most_scratch_; }
  std::uint64_t most_activations() const { return most_activations_; }

 private:
  std::expected<std::unique_ptr<Planned>, std::string> Prepare(const kg::Dsv4ChunkShape& shape,
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
  std::vector<std::pair<kg::Dsv4ChunkShape, std::unique_ptr<Planned>>> cache_;
  int plans_made_ = 0;
  std::uint64_t most_scratch_ = 0;
  std::uint64_t most_activations_ = 0;
};

// ------------------------------------------------------------------ inputs

struct TokenLine {
  std::string name;
  std::vector<std::int32_t> ids;
};

std::expected<std::vector<TokenLine>, std::string> ReadTokenLines(const std::filesystem::path& p) {
  std::ifstream file(p);
  if (!file) {
    return Error(std::format("cannot read {}", p.string()));
  }
  std::vector<TokenLine> out;
  std::string line;
  while (std::getline(file, line)) {
    if (line.empty()) {
      continue;
    }
    TokenLine t;
    const auto tab = line.find('\t');
    std::string_view rest = line;
    if (tab != std::string::npos) {
      t.name = line.substr(0, tab);
      rest = std::string_view(line).substr(tab + 1);
    }
    std::istringstream ids{std::string(rest)};
    std::int64_t id = 0;
    while (ids >> id) {
      t.ids.push_back(static_cast<std::int32_t>(id));
    }
    out.push_back(std::move(t));
  }
  return out;
}

std::int32_t Argmax(std::span<const float> row) {
  return static_cast<std::int32_t>(std::ranges::max_element(row) - row.begin());
}

// -log softmax(row)[target], in double.
double Nll(std::span<const float> row, std::int32_t target) {
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

// ------------------------------------------------------------------ layout proof

// Every layer's routed products over the slab against the reference layout
// ([k, n, experts] packed), bit for bit, at 1, 5 and 64 tokens.
Status LayoutProof(const Model& m, Device& d, kg::LaunchContext& launch,
                   const jitllm::execution::Registry& registry, std::string& report) {
  const auto& a = *m.artifact;
  const auto& w = *m.weights;
  std::mt19937 rng(20260928);  // NOLINT(bugprone-random-generator-seed): reproducible
  std::normal_distribution<float> normal(0.0f, 1.0f);
  std::uniform_int_distribution<int> expert(0, static_cast<int>(m.profile->experts) - 1);
  std::uint64_t compared = 0;
  std::uint64_t differing = 0;
  report += "[";
  bool first_entry = true;
  for (std::uint32_t il = 0; il < m.profile->layers; ++il) {
    const md::Dsv4Layer& l = m.binding->layers[il];
    for (const md::Dsv4Tensor* t : {&l.gate_exps, &l.up_exps, &l.down_exps}) {
      const auto& array = a.expert_arrays()[t->index];
      const std::uint64_t slice = array.slice_bytes.value();
      const std::uint64_t packed_bytes = slice * array.count;
      // The reference layout: the slices packed in expert order, which is
      // the GGUF tensor's own layout (the import copied each slice from
      // offset e x slice of it).
      void* reference = nullptr;
      if (auto r = Cuda(cudaMalloc(&reference, packed_bytes), "the reference layout"); !r) {
        return r;
      }
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
        ggml_tensor* out_slab = ggml_mul_mat_id(c, slab, x, ids);
        ggml_tensor* out_packed = ggml_mul_mat_id(c, packed, x, ids);
        std::vector<float> xs(static_cast<std::size_t>(k * tokens));
        for (float& v : xs) {
          v = normal(rng);
        }
        std::vector<std::int32_t> routes(static_cast<std::size_t>(used * tokens));
        for (std::int64_t tok = 0; tok < tokens; ++tok) {
          std::vector<int> chosen;
          while (std::cmp_less(chosen.size(), used)) {
            const int e = expert(rng);
            if (std::ranges::find(chosen, e) == chosen.end()) {
              chosen.push_back(e);
            }
          }
          for (std::int64_t j = 0; j < used; ++j) {
            routes[static_cast<std::size_t>((tok * used) + j)] =
                chosen[static_cast<std::size_t>(j)];
          }
        }
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
        kg::TensorArena::Bind(slab, ArrayAddress(a, w, t->index));
        kg::TensorArena::Bind(packed, Address(reference));
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
            R"({}{{"layer":{},"tensor":"{}","type":"{}","tokens":{},"implementation":"{}","stride":{},"packed_stride":{},"elements":{},"differing":{}}})",
            first_entry ? "" : ",", il, array.name, t->type, tokens, plan->steps[0].implementation,
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
  std::uint32_t bench_prefill = 0;
  std::uint32_t bench_decode = 0;
  bool exact = false;            // --exact on: the reference mode (Model::exact)
  bool compact_experts = false;  // the experimental device-built expert tile list
  // --probe-step N: the first prompt's forced step N run twice from the
  // state before it, in the fast plan and in the reference mode, every
  // named tensor dumped (a full window, which both plans read).
  std::uint32_t probe_step = 0;
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
    } else if (a == "--probe-step") {
      ok = number(o.probe_step);
    } else if (a == "--layout-proof") {
      o.layout_proof = true;
    } else if (a == "--compact-experts") {
      o.compact_experts = true;
    } else if (a == "--exact") {
      auto v = value();
      if (!v || (*v != "on" && *v != "off")) {
        return Error("--exact takes on or off");
      }
      o.exact = *v == "on";
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
  if (o.probe_step != 0 && (o.force.empty() || o.probe_step >= o.generate)) {
    return Error("--probe-step needs --force and a step below --generate");
  }
  return o;
}

// --probe-step: the step at n_past (token `next`) from the state before it,
// once in the fast plan and once in the reference mode, the state put back
// after each; every named tensor and the logits under OUT/probe/{fast,exact}/
// (the dsv4-decode probe's method on the resident harness).
Status Probe(const Options& o, Runner& runner, void* state, std::uint64_t bytes,
             std::uint32_t n_past, std::int32_t next) {
  void* saved = nullptr;
  if (auto r = Cuda(cudaMalloc(&saved, bytes), "the probe's saved state"); !r) {
    return r;
  }
  Status result;
  if (auto r = Cuda(cudaMemcpy(saved, state, bytes, cudaMemcpyDeviceToDevice), "saving the state");
      !r) {
    result = r;
  }
  const std::array<std::string, 1> all = {"*"};
  for (const bool exact : {false, true}) {
    if (!result) {
      break;
    }
    runner.SetExact(exact);
    std::vector<float> logits;
    std::map<std::string, std::vector<float>> kept;
    if (auto r = runner.Chunk(n_past, std::span(&next, 1), logits, all, &kept); !r) {
      result = r;
      break;
    }
    const std::filesystem::path dir = o.out / "probe" / (exact ? "exact" : "fast");
    std::filesystem::create_directories(dir);
    std::string index = "{";
    for (const auto& [name, values] : kept) {
      if (auto r = WriteFloats(dir / (name + ".f32"), values); !r) {
        result = r;
      }
      index += std::format(R"({}"{}":{})", index.size() == 1 ? "" : ",", name, values.size());
    }
    std::ofstream(dir / "index.json") << index << "}\n";
    if (auto r = WriteFloats(dir / "logits.f32", logits); !r) {
      result = r;
    }
    if (auto r =
            Cuda(cudaMemcpy(state, saved, bytes, cudaMemcpyDeviceToDevice), "restoring the state");
        !r) {
      result = r;
    }
  }
  runner.SetExact(o.exact);
  (void)cudaFree(saved);
  std::println("probe: position {} (token {}) run in both plans from one state", n_past, next);
  return result;
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
  const md::Dsv4Profile& profile = md::Dsv4Flash();
  auto binding = md::BindDsv4(profile, *artifact);
  if (!binding) {
    return std::unexpected(binding.error());
  }
  // The fast plan's window is a ring; the reference mode's, llama.cpp's full
  // cache (model/dsv4.h Dsv4Window).
  auto state =
      md::Dsv4State(profile, o.context, o.max_rows,
                    o.exact || o.probe_step != 0 ? md::Dsv4Window::kFull : md::Dsv4Window::kRing);
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
  // The hash-routed layers' tables are data the kernels index with
  // unchecked: every entry must name an expert before anything runs.
  for (std::uint32_t il = 0; il < profile.layers; ++il) {
    const md::Dsv4Layer& l = binding->layers[il];
    if (!l.hash) {
      continue;
    }
    std::vector<std::int32_t> table(std::size_t{profile.experts_used} * profile.vocab);
    if (auto r = Cuda(
            cudaMemcpy(table.data(), Pointer(ResourceAddress(*artifact, weights, l.tid2eid.index)),
                       table.size() * sizeof(std::int32_t), cudaMemcpyDeviceToHost),
            "reading a hash-routing table");
        !r) {
      return r;
    }
    if (auto checked = md::CheckDsv4HashRouting(profile, table); !checked) {
      return Error(std::format("layer {}: {}", il, checked.error()));
    }
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
              .state_base = Address(state_region),
              .rot = kg::HadamardMatrix(profile.indexer_head_dim),
              .exact = o.exact,
              .compact_experts = o.compact_experts};

  // cuBLAS, with upstream's workspace for the device: the router's BF16
  // products run there at prefill widths.
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

  // Size the activations and the pool by the largest shapes: a full chunk
  // at the start and at the end of the context, and a decode step at the end.
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
        {{0, o.max_rows},
         {o.context - o.max_rows, o.max_rows},
         {o.context - 1, 1},
         {o.max_rows, 1}}};
    for (const auto& [n_past, rows] : probes) {
      auto in = md::Dsv4Chunk(profile, *state, n_past, rows, o.exact);
      if (!in) {
        return std::unexpected(in.error());
      }
      auto planned = PlanChunk(model, kg::Dsv4ShapeOf(*state, *in), choices, o.dump, 0, 0);
      if (!planned) {
        return Error(
            std::format("measuring a chunk of {} at {}: {}", rows, n_past, planned.error()));
      }
      most_activations = std::max(most_activations, (*planned)->placement.extent);
      auto scratch = kg::PlanScratch(**measure, (*planned)->plan);
      if (!scratch) {
        return Error(scratch.error().detail);
      }
      most_scratch = std::max(most_scratch, *scratch);
      most_inputs = std::max(most_inputs, (*planned)->inputs_bytes);
      std::println("shape rows {} at {}: {} steps, activations {} B, scratch {} B", rows, n_past,
                   (*planned)->plan.steps.size(), (*planned)->placement.extent, *scratch);
    }
  }
  // Margins: other shapes of these widths place a little differently.
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
      R"({{"artifact":"{}","context":{},"max_rows":{},"load_seconds":{:.3f},"bytes_read":{},"dense_bytes":{},"slab_bytes":{},"slab_padding":{},"expert_strides":[{}],"state_bytes":{},"activation_bytes":{},"scratch_bytes":{})",
      artifact->id(), o.context, o.max_rows, weights.load_seconds, weights.bytes_read,
      weights.dense_bytes, weights.slab_bytes, weights.slab_padding,
      [&] {
        std::string s;
        for (std::size_t i = 0; i < weights.stride.size(); ++i) {
          s += std::format("{}{}", i == 0 ? "" : ",", weights.stride[i]);
        }
        return s;
      }(),
      state->bytes, activation_bytes, scratch_bytes);

  if (o.layout_proof) {
    std::string report;
    if (auto r = LayoutProof(model, d, **launch, *registry, report); !r) {
      return r;
    }
    summary += R"(,"layout_proof":)" + report;
  }

  if (!o.prompts.empty()) {
    auto prompts = ReadTokenLines(o.prompts);
    if (!prompts) {
      return std::unexpected(prompts.error());
    }
    std::vector<TokenLine> forced;
    if (!o.force.empty()) {
      auto f = ReadTokenLines(o.force);
      if (!f) {
        return std::unexpected(f.error());
      }
      forced = std::move(*f);
      if (forced.size() != prompts->size()) {
        return Error("--force needs a line per prompt");
      }
    }
    summary += R"(,"prompts":[)";
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
      std::map<std::string, std::vector<float>> kept;
      const bool dump = pi == 0 && !o.dump.empty();
      if (dump && prompt.ids.size() > o.max_rows) {
        return Error("--dump needs a first prompt of one chunk");
      }
      const auto t0 = Clock::now();
      // A prompt longer than --max-rows (long context) is prefilled in chunks
      // of that many rows; the last chunk's last row gives the first token.
      for (std::size_t at = 0, rows = 0; at < prompt.ids.size(); at += rows) {
        rows = ChunkRows(prompt.ids.size() - at, o.max_rows);
        if (auto r = runner.Chunk(
                static_cast<std::uint32_t>(at), std::span(prompt.ids).subspan(at, rows), logits,
                dump ? std::span(o.dump) : std::span<const std::string>{}, dump ? &kept : nullptr);
            !r) {
          return Error(std::format("{} at {}: {}", prompt.name, at, r.error()));
        }
      }
      const double prefill = Seconds(Clock::now() - t0);
      if (dump) {
        std::filesystem::create_directories(o.out / "dump");
        std::string index = "{";
        bool first = true;
        for (const auto& [name, values] : kept) {
          if (auto r = WriteFloats(o.out / "dump" / (name + ".f32"), values); !r) {
            return r;
          }
          index += std::format(R"({}"{}":{})", first ? "" : ",", name, values.size());
          first = false;
        }
        std::ofstream(o.out / "dump" / "index.json") << index << "}\n";
      }
      const std::uint32_t vocab = profile.vocab;
      std::vector<float> steps;
      std::vector<std::int32_t> argmax;
      steps.insert(steps.end(), logits.end() - vocab, logits.end());
      argmax.push_back(Argmax(std::span(logits).subspan(logits.size() - vocab)));
      auto n_past = static_cast<std::uint32_t>(prompt.ids.size());
      double decode = 0;
      for (std::uint32_t k = 1; k < o.generate; ++k) {
        const std::int32_t next = forced.empty() ? argmax.back() : forced[pi].ids[k - 1];
        if (pi == 0 && k == o.probe_step) {
          if (auto r = Probe(o, runner, state_region, state->bytes, n_past, next); !r) {
            return r;
          }
        }
        const auto t1 = Clock::now();
        if (auto r = runner.Chunk(n_past, std::span(&next, 1), logits); !r) {
          return Error(std::format("{} step {}: {}", prompt.name, k, r.error()));
        }
        decode += Seconds(Clock::now() - t1);
        ++n_past;
        steps.insert(steps.end(), logits.begin(), logits.end());
        argmax.push_back(Argmax(logits));
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
  }

  if (!o.ppl.empty()) {
    auto lines = ReadTokenLines(o.ppl);
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
    std::vector<float> logits;
    const auto t0 = Clock::now();
    for (std::uint32_t at = 0, rows = 0; at < ids.size(); at += rows) {
      rows = ChunkRows(ids.size() - at, o.max_rows);
      if (auto r = runner.Chunk(at, std::span(ids).subspan(at, rows), logits); !r) {
        return Error(std::format("perplexity chunk at {}: {}", at, r.error()));
      }
      for (std::uint32_t i = 0; i < rows; ++i) {
        const std::size_t next = std::size_t{at} + i + 1;
        if (next < ids.size()) {
          nll.push_back(Nll(
              std::span(logits).subspan(std::size_t{i} * profile.vocab, profile.vocab), ids[next]));
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
    double decode_total = 0;
    std::uint32_t decode_steps = 0;
    for (int rep = 0; rep < 3; ++rep) {
      if (auto r = runner.Clear(); !r) {
        return r;
      }
      const auto t0 = Clock::now();
      for (std::uint32_t at = 0; at < ids.size(); at += o.max_rows) {
        const auto rows =
            static_cast<std::uint32_t>(std::min<std::size_t>(o.max_rows, ids.size() - at));
        if (auto r = runner.Chunk(at, std::span(ids).subspan(at, rows), logits); !r) {
          return r;
        }
      }
      prefill_best = std::min(prefill_best, Seconds(Clock::now() - t0));
    }
    // Decode as llama-bench's tg runs it: one token at a time from an empty
    // context, a warm-up (which plans the shape), then three repetitions,
    // the mean reported.
    for (int rep = -1; rep < 3 && o.bench_decode > 0; ++rep) {
      if (auto r = runner.Clear(); !r) {
        return r;
      }
      std::int32_t next = ids[0];
      const auto t1 = Clock::now();
      for (std::uint32_t k = 0; k < o.bench_decode; ++k) {
        if (auto r = runner.Chunk(k, std::span(&next, 1), logits); !r) {
          return r;
        }
        next = Argmax(logits);
      }
      if (rep >= 0) {
        decode_total += Seconds(Clock::now() - t1) / 3.0;
      }
      decode_steps = o.bench_decode;
    }
    summary += std::format(
        R"(,"bench":{{"prefill_tokens":{},"prefill_seconds_best_of_3":{:.4f},"prefill_tok_s":{:.2f},"decode_steps":{},"decode_seconds":{:.4f},"decode_tok_s":{:.3f}}})",
        ids.size(), prefill_best, static_cast<double>(ids.size()) / prefill_best, decode_steps,
        decode_total, decode_steps > 0 ? static_cast<double>(decode_steps) / decode_total : 0.0);
    std::println(
        "bench: prefill {} tokens {:.3f} s ({:.1f} tok/s); decode {} steps {:.3f} s "
        "({:.2f} tok/s)",
        ids.size(), prefill_best, static_cast<double>(ids.size()) / prefill_best, decode_steps,
        decode_total, decode_steps > 0 ? static_cast<double>(decode_steps) / decode_total : 0.0);
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
    std::println(stderr, "jitllm_dsv4_exec: {}", r.error());
    return 1;
  }
  return 0;
}
