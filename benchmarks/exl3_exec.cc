// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Backend-proof P3, the native EXL3 model (docs/backend-proof.md, Tier E
// "GGML-derived operations inside the EXL3 plan" and Tier C;
// docs/experiments/backend-proof-p3/README.md): an EXL3 fixture from its v0
// prepared artifact, run natively through the operation plan
// (model/qwen2_exl3.h) bound to the registry's implementations
// (kernels/exl3/qwen2.h), on the declared trajectories: for each prefix of
// the held-out IDs, the prefill of every row, then 16 single-token steps.
//
//   llmp_exl3_exec --artifact DIR --fixture 4.0bpw|4.5bpw --arm G|O
//                    --plan PLAN.txt --ids FILE --out DIR
//                    [--prefixes 32,144,145,1023,1024] [--evaluations N]
//                    [--record] [--capture] [--record-ops DIR]
//
// - Weights: the artifact is opened as untrusted input (artifact.h) and
//   bound to the plan's tensors (BindQwen2Exl3); every chunk of every group
//   is read with direct I/O (Artifact::PlanReads), its SHA-256 checked
//   against the index, and copied into one cudaMalloc region (rung 3). The
//   norms, BF16 in the artifact, are widened exactly into F32 copies (the
//   record's attn_norm.w, mlp_norm.w, final_norm.w: 49 × 3,584 bytes);
//   each layer's multi-GEMM tables are uploaded once.
// - PLAN.txt: the forced launch plans (docs/experiments/backend-proof-p3/
//   model_plan.py): `linear` lines, which must match the artifact's
//   linears, and `case ID ROWS PATH ...` lines, the per-linear sweep's
//   format; `#` lines are comments, and `# cache SHA256` names the frozen
//   tuning cache they were decoded from, which with the file's SHA-256
//   is the launch table's source in every plan identity.
// - Every phase of the run is planned first (PlanPhase refuses an
//   unrecorded phase kind or a missing plan), and bound (Qwen2Program::Bind
//   refuses a missing or stale implementation, and any operand its
//   implementation's host checks refuse), before anything runs. Then the
//   activation region is the largest plan's and the GGML pool the largest
//   attention scratch.
// - Each evaluation clears the cache and runs every prefix's trajectory
//   from an empty cache; the first evaluation's logits (the F16 logits,
//   widened exactly) are written as .npy files, and each later one must
//   equal them bit for bit.
// - --record: tests/support's launch recorder notes every launch, copy and
//   cuBLAS(Lt) call of the first evaluation, each phase between chunk
//   lines and each operation after an op line (plan_record.h), for
//   op_plan_compare.py with an nsys trace of the same run.
// - --capture (Tier C): the residual stream after every block and the
//   final norm's output during each prefill, and the cache's K and V after
//   the prefill and after the steps, as .npy files (pack_run.py makes the
//   reference's npz of them). Instrumented: compare its logits with an
//   uninstrumented run's (RE-010).
// - --record-ops DIR (Tier E, operation level): every operation's inputs
//   (SHA-256, hashed before it runs) and outputs (bytes and SHA-256, after
//   it has run) in the first evaluation, per phase (op_tier_e.py's format).
//   Weights are hashed once, at load (they are never written), and each
//   operation's weight is recorded by the hash of what lies at the address
//   the program bound for it: the embedding, norms and biases among its
//   inputs, and a linear's trellis and side vectors (for the multi-GEMM,
//   both linears', and what lies at the addresses its device tables hold)
//   as its "weights".
//
// Output: DIR/logits-P.prefill.npy (float32 [P, vocab]),
// DIR/logits-P.suffix.npy (float32 [16, vocab]), the captures, and
// DIR/manifest.json: the artifact, the plan file's SHA-256, each phase's
// plan identity, the logits' SHA-256s and the evaluations' comparison.

#include <cuda_runtime.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <expected>
#include <filesystem>
#include <format>
#include <fstream>
#include <functional>
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
#include "artifact/layout.h"
#include "base/bytes.h"
#include "base/sha256.h"
#include "execution/registry.h"
#include "exl3_common.h"
#include "kernels/exl3/implementations.h"
#include "kernels/exl3/launch.h"
#include "kernels/exl3/qwen2.h"
#include "kernels/exl3/recon_gemm.h"
#include "kernels/exl3/validate.h"
#include "kernels/ggml/implementations.h"
#include "kernels/ggml/launch.h"
#include "launch_recorder.h"
#include "model/qwen2.h"
#include "model/qwen2_exl3.h"
#include "plan_record.h"
#include "providers/cuda/cuda_device_execution.h"
#include "providers/device_execution.h"

namespace {

namespace exl3 = llmp::kernels::exl3;
namespace kg = llmp::kernels::ggml;
namespace model = llmp::model;
using Status = std::expected<void, std::string>;

using llmp::benchmarks::Bf16ToFloat;
using llmp::benchmarks::Hex;
using llmp::benchmarks::HexFile;
using llmp::benchmarks::kCells;
using llmp::benchmarks::kSuffix;
using llmp::benchmarks::LoadIds;
using llmp::benchmarks::LoadTable;
using llmp::benchmarks::WriteNpy;

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

// ------------------------------------------------------------------ device

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
    for (void* pointer : device_) {
      (void)cudaFree(pointer);
    }
    for (void* pointer : pinned_) {
      (void)cudaFreeHost(pointer);
    }
  }

  llmp::providers::DeviceExecution& execution() { return *execution_; }
  llmp::providers::StreamId stream() const { return stream_; }

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
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::minutes(2);
    for (;;) {
      const auto state = execution_->Query(*fence);
      if (!state) {
        return Error("Query failed");
      }
      if (*state == llmp::providers::FenceState::kComplete) {
        break;
      }
      if (std::chrono::steady_clock::now() > deadline) {
        return Error("the stream did not complete");
      }
      std::this_thread::sleep_for(std::chrono::microseconds(20));
    }
    return execution_->Release(*fence) ? Status() : Error("Release failed");
  }

  // Device memory (cudaMalloc, 256-byte aligned) and pinned host memory,
  // freed with the device.
  std::expected<std::uint64_t, std::string> Allocate(std::uint64_t bytes, std::string_view what) {
    void* pointer = nullptr;
    if (auto r = Cuda(cudaMalloc(&pointer, std::max<std::uint64_t>(bytes, 256)), what); !r) {
      return std::unexpected(r.error());
    }
    device_.push_back(pointer);
    return Address(pointer);
  }
  std::expected<std::byte*, std::string> Pinned(std::uint64_t bytes, std::string_view what) {
    void* pointer = nullptr;
    if (auto r = Cuda(cudaMallocHost(&pointer, std::max<std::uint64_t>(bytes, 256)), what); !r) {
      return std::unexpected(r.error());
    }
    pinned_.push_back(pointer);
    return static_cast<std::byte*>(pointer);
  }

  // Copies device bytes to the host once everything queued has run.
  Status Download(std::uint64_t address, std::span<std::byte> out) {
    if (auto r = Finish(); !r) {
      return r;
    }
    return Cuda(cudaMemcpy(out.data(), Pointer(address), out.size(), cudaMemcpyDeviceToHost),
                "a download");
  }

 private:
  Device(std::unique_ptr<llmp::providers::DeviceExecution> execution,
         llmp::providers::StreamId stream)
      : execution_(std::move(execution)), stream_(stream) {}

  std::unique_ptr<llmp::providers::DeviceExecution> execution_;
  llmp::providers::StreamId stream_;
  std::vector<void*> device_;
  std::vector<void*> pinned_;
};

// ------------------------------------------------------------------ inputs

struct Options {
  std::filesystem::path artifact;
  std::string fixture;
  model::Exl3Arm arm = model::Exl3Arm::kG;
  std::filesystem::path plan;
  std::filesystem::path ids;
  std::filesystem::path out;
  std::vector<int> prefixes = {32, 144, 145, 1023, 1024};
  int evaluations = 2;
  bool record = false;
  bool capture = false;
  std::filesystem::path record_ops;
};

std::expected<Options, std::string> Parse(std::span<char*> args) {
  Options o;
  bool arm = false;
  for (std::size_t i = 1; i < args.size(); ++i) {
    const std::string_view flag = args[i];
    const auto value = [&]() -> std::string_view {
      return i + 1 < args.size() ? std::string_view(args[++i]) : std::string_view();
    };
    if (flag == "--artifact") {
      o.artifact = value();
    } else if (flag == "--fixture") {
      o.fixture = value();
    } else if (flag == "--arm") {
      const std::string_view a = value();
      if (a != "G" && a != "O") {
        return Error("--arm is G or O");
      }
      o.arm = a == "G" ? model::Exl3Arm::kG : model::Exl3Arm::kO;
      arm = true;
    } else if (flag == "--plan") {
      o.plan = value();
    } else if (flag == "--ids") {
      o.ids = value();
    } else if (flag == "--out") {
      o.out = value();
    } else if (flag == "--prefixes") {
      o.prefixes.clear();
      std::istringstream list{std::string(value())};
      for (std::string item; std::getline(list, item, ',');) {
        int prefix = 0;
        const auto [end, error] = std::from_chars(item.data(), item.data() + item.size(), prefix);
        if (error != std::errc() || end != item.data() + item.size() || prefix <= 0) {
          return Error("--prefixes takes positive integers");
        }
        o.prefixes.push_back(prefix);
      }
    } else if (flag == "--evaluations") {
      const std::string_view text = value();
      const auto [end, error] =
          std::from_chars(text.data(), text.data() + text.size(), o.evaluations);
      if (error != std::errc() || end != text.data() + text.size()) {
        return Error("--evaluations takes an integer");
      }
    } else if (flag == "--record") {
      o.record = true;
    } else if (flag == "--capture") {
      o.capture = true;
    } else if (flag == "--record-ops") {
      o.record_ops = value();
    } else {
      return Error(std::format("unknown argument {}", flag));
    }
  }
  if (o.artifact.empty() || (o.fixture != "4.0bpw" && o.fixture != "4.5bpw") || !arm ||
      o.plan.empty() || o.ids.empty() || o.out.empty() || o.prefixes.empty() || o.evaluations < 1 ||
      o.evaluations > 8) {
    return Error(
        "usage: llmp_exl3_exec --artifact DIR --fixture 4.0bpw|4.5bpw --arm G|O "
        "--plan PLAN.txt --ids FILE --out DIR [--prefixes LIST] [--evaluations N] "
        "[--record] [--capture] [--record-ops DIR]");
  }
  return o;
}

// ------------------------------------------------------------------ weights

struct Weights {
  std::uint64_t base = 0;
  std::vector<std::uint64_t> group_base;
  std::uint64_t bytes = 0;
  std::uint64_t chunks_verified = 0;
};

Status LoadWeights(const llmp::artifact::Artifact& artifact, Device& device, Weights& weights) {
  const auto groups = artifact.groups();
  weights.group_base.resize(groups.size());
  for (std::size_t g = 0; g < groups.size(); ++g) {
    weights.group_base[g] = weights.bytes;
    weights.bytes += Round(groups[g].stored.value(), 256);
  }
  auto region = device.Allocate(weights.bytes, "the weights' region");
  if (!region) {
    return std::unexpected(region.error());
  }
  weights.base = *region;
  std::vector<llmp::artifact::ChunkKey> all;
  for (std::uint32_t g = 0; g < groups.size(); ++g) {
    for (std::uint32_t c = 0; c < groups[g].chunks; ++c) {
      all.push_back({.group = g, .chunk = c});
    }
  }
  const llmp::artifact::ReadLimits limits{};
  const auto runs = artifact.PlanReads(all, {}, limits);
  if (!runs) {
    return Error("the artifact's read plan was refused");
  }
  auto staging = device.Pinned(limits.max_run.value(), "pinned staging");
  if (!staging) {
    return std::unexpected(staging.error());
  }
  std::vector<llmp::artifact::FileDescriptor> shards;
  for (std::uint32_t s = 0; s < artifact.shards().size(); ++s) {
    auto fd = artifact.OpenShardForDirectRead(s);
    if (!fd) {
      return Error(std::format("shard {} cannot be opened for direct reads", s));
    }
    shards.push_back(std::move(*fd));
  }
  for (const auto& run : *runs) {
    std::uint64_t done = 0;
    while (done < run.length.value()) {
      const ssize_t got = pread(shards[run.shard].get(), *staging + done, run.length.value() - done,
                                static_cast<off_t>(run.file_offset.value() + done));
      if (got <= 0) {
        return Error(std::format("a direct read of shard {} failed", run.shard));
      }
      done += static_cast<std::uint64_t>(got);
    }
    auto stream = device.Stream();
    if (!stream) {
      return std::unexpected(stream.error());
    }
    std::uint64_t at = 0;
    for (const auto& segment : run.segments) {
      const auto& group = groups[segment.chunk.group];
      const std::span<const std::byte> piece(*staging + at, segment.length.value());
      llmp::base::Sha256 hash;
      hash.Update(piece);
      if (hash.Finish() != artifact.chunk_sha256()[group.first_chunk + segment.chunk.chunk]) {
        return Error(std::format("group {} chunk {} does not match its digest", segment.chunk.group,
                                 segment.chunk.chunk));
      }
      ++weights.chunks_verified;
      const std::uint64_t dst =
          weights.base + weights.group_base[segment.chunk.group] + segment.group_offset.value();
      if (auto r = Cuda(cudaMemcpyAsync(Pointer(dst), piece.data(), piece.size(),
                                        cudaMemcpyHostToDevice, *stream),
                        "a weight upload");
          !r) {
        return r;
      }
      at += segment.length.value();
    }
    if (auto r = device.Finish(); !r) {
      return r;
    }
  }
  if (weights.chunks_verified != artifact.chunk_sha256().size()) {
    return Error("not every chunk was read and verified");
  }
  return {};
}

// ------------------------------------------------------------------ the run

struct PhaseRun {
  int prefix = 0;
  int index = 0;  // 0: the prefill; 1..16: the steps
  model::Exl3PhasePlan plan;
  std::unique_ptr<exl3::Qwen2Program> program;
};

class Run {
 public:
  explicit Run(const Options& o) : o_(o) {}

  Status Execute();

 private:
  Status Setup();
  Status PlanAll();
  Status Evaluate(int evaluation);
  Status RunPhase(PhaseRun& phase, int evaluation);
  Status Write();
  std::uint64_t WeightAddress(std::uint32_t resource) const {
    const auto& r = artifact_->resources()[resource];
    return weights_.base + weights_.group_base[r.group] + r.offset.value();
  }
  exl3::Qwen2Linear Linear(const model::Exl3LinearBinding& l) const {
    return {.weights = {.trellis = WeightAddress(l.trellis),
                        .suh = WeightAddress(l.suh),
                        .svh = WeightAddress(l.svh),
                        .k = l.k,
                        .n = l.n,
                        .bits = l.bits},
            .bias = l.bias ? WeightAddress(*l.bias) : 0};
  }
  // Recording hooks.
  std::expected<void, exl3::KernelFailure> Before(const PhaseRun& phase, std::size_t op);
  std::expected<void, exl3::KernelFailure> After(const PhaseRun& phase, std::size_t op);
  void Flush();
  std::expected<std::string, std::string> HashDevice(std::uint64_t address, std::uint64_t bytes);

  const Options& o_;
  const model::Qwen2Profile& profile_ = model::Qwen25Instruct05BExl3();
  std::unique_ptr<Device> device_;
  std::unique_ptr<llmp::artifact::Artifact> artifact_;
  model::Exl3Binding binding_;
  std::optional<model::Exl3LaunchTable> table_;
  Weights weights_;
  exl3::Qwen2Memory memory_;
  std::vector<std::int32_t> ids_;
  std::unique_ptr<llmp::execution::Registry> registry_;
  std::unique_ptr<kg::LaunchContext> ggml_;
  std::unique_ptr<exl3::LaunchContext> launch_;
  std::unique_ptr<exl3::ReconGemm> gemm_;
  std::uint64_t kv_ = 0;
  std::uint64_t kv_bytes_ = 0;
  std::byte* host_inputs_ = nullptr;
  std::uint64_t host_inputs_bytes_ = 0;
  std::vector<PhaseRun> phases_;
  // Per prefix: the first evaluation's logits (F32, prefill then suffix).
  std::map<int, std::vector<float>> logits_;
  std::map<int, std::vector<float>> current_;
  int mismatches_ = 0;
  std::string evaluations_;
  // Recording.
  std::unique_ptr<llmp::test_support::Recording> recording_;
  std::string record_;
  // Capture.
  std::map<int, std::vector<std::byte>> blocks_, final_norm_, kv_prefill_, kv_suffix_;
  // Operation recording.
  std::map<std::uint64_t, std::string> weight_hash_;  // by device address
  std::string ops_json_;
  std::ofstream ops_bin_;
  std::uint64_t ops_at_ = 0;
  std::string op_inputs_;
  std::string op_weights_;
};

std::expected<std::string, std::string> Run::HashDevice(std::uint64_t address,
                                                        std::uint64_t bytes) {
  std::vector<std::byte> host(bytes);
  if (auto r = device_->Download(address, host); !r) {
    return std::unexpected(r.error());
  }
  return Hex(host);
}

Status Run::Setup() {
  auto device = Device::Open();
  if (!device) {
    return std::unexpected(device.error());
  }
  device_ = std::move(*device);
  auto ids = LoadIds(o_.ids);
  if (!ids) {
    return std::unexpected(ids.error());
  }
  ids_ = std::move(*ids);
  auto artifact = llmp::artifact::Artifact::Open(o_.artifact);
  if (!artifact) {
    return Error(std::format("the artifact was refused: {}", artifact.error().reason));
  }
  artifact_ = std::make_unique<llmp::artifact::Artifact>(std::move(*artifact));
  auto binding = model::BindQwen2Exl3(profile_, *artifact_);
  if (!binding) {
    return Error("the artifact does not bind: " + binding.error());
  }
  binding_ = std::move(*binding);
  auto table = LoadTable(o_.plan, binding_);
  if (!table) {
    return std::unexpected(table.error());
  }
  table_.emplace(std::move(*table));
  if (auto r = LoadWeights(*artifact_, *device_, weights_); !r) {
    return r;
  }

  // The norms, widened exactly to F32, and the multi-GEMM tables.
  const std::uint64_t norm_bytes = std::uint64_t{profile_.width} * 4;
  auto norms = device_->Allocate(norm_bytes * ((2 * profile_.layers) + 1), "the F32 norms");
  auto tables = device_->Allocate(std::uint64_t{48} * profile_.layers, "the multi-GEMM tables");
  if (!norms || !tables) {
    return std::unexpected(!norms ? norms.error() : tables.error());
  }
  std::vector<float> widened;
  std::vector<std::uint64_t> table_words;
  const auto widen = [&](std::uint32_t resource) -> std::expected<std::uint64_t, std::string> {
    std::vector<std::uint16_t> bf16(profile_.width);
    if (auto r =
            device_->Download(WeightAddress(resource), std::as_writable_bytes(std::span(bf16)));
        !r) {
      return std::unexpected(r.error());
    }
    const std::uint64_t address = *norms + (widened.size() * 4);
    for (const std::uint16_t v : bf16) {
      widened.push_back(Bf16ToFloat(v));
    }
    return address;
  };
  memory_.embed = WeightAddress(binding_.embed);
  auto final_norm = widen(binding_.final_norm);
  if (!final_norm) {
    return std::unexpected(final_norm.error());
  }
  memory_.final_norm = *final_norm;
  memory_.lm_head = Linear(binding_.lm_head);
  kv_bytes_ = std::uint64_t{profile_.layers} * 2 * kCells * profile_.kv_width() * 2;
  auto kv = device_->Allocate(kv_bytes_, "the cache");
  if (!kv) {
    return std::unexpected(kv.error());
  }
  kv_ = *kv;
  memory_.cells = kCells;
  for (std::uint32_t l = 0; l < profile_.layers; ++l) {
    const model::Exl3LayerBinding& b = binding_.layers[l];
    exl3::Qwen2Layer layer;
    auto attn = widen(b.attn_norm);
    auto mlp = widen(b.mlp_norm);
    if (!attn || !mlp) {
      return std::unexpected(!attn ? attn.error() : mlp.error());
    }
    layer.attn_norm = *attn;
    layer.mlp_norm = *mlp;
    layer.q = Linear(b.q);
    layer.k = Linear(b.k);
    layer.v = Linear(b.v);
    layer.o = Linear(b.o);
    layer.gate = Linear(b.gate);
    layer.up = Linear(b.up);
    layer.down = Linear(b.down);
    const std::uint64_t base = *tables + (std::uint64_t{48} * l);
    layer.trellis_table = base;
    layer.suh_table = base + 16;
    layer.svh_table = base + 32;
    layer.tables_written = exl3::MultiGemmTables(layer.gate.weights, layer.up.weights);
    table_words.insert(table_words.end(), layer.tables_written.begin(), layer.tables_written.end());
    const std::uint64_t per_layer = std::uint64_t{2} * kCells * profile_.kv_width() * 2;
    layer.k_cache = kv_ + (per_layer * l);
    layer.v_cache = layer.k_cache + (per_layer / 2);
    memory_.layers.push_back(layer);
  }
  if (auto r = Cuda(
          cudaMemcpy(Pointer(*norms), widened.data(), widened.size() * 4, cudaMemcpyHostToDevice),
          "the norms' upload");
      !r) {
    return r;
  }
  if (auto r = Cuda(cudaMemcpy(Pointer(*tables), table_words.data(), table_words.size() * 8,
                               cudaMemcpyHostToDevice),
                    "the tables' upload");
      !r) {
    return r;
  }

  std::vector<llmp::execution::Implementation> implementations = kg::Implementations();
  for (auto& implementation : exl3::Implementations()) {
    implementations.push_back(std::move(implementation));
  }
  auto registry = llmp::execution::Registry::Create(std::move(implementations));
  if (!registry) {
    return Error("the registry was refused: " + registry.error().detail);
  }
  registry_ = std::make_unique<llmp::execution::Registry>(std::move(*registry));
  auto locks = device_->Allocate(exl3::kLockBytes, "the lock area");
  if (!locks) {
    return std::unexpected(locks.error());
  }
  auto launch = exl3::LaunchContext::Create(0, device_->execution(), device_->stream(), *locks);
  if (!launch) {
    return Error("the EXL3 launch context was refused: " + launch.error().detail);
  }
  launch_ = std::move(*launch);
  auto gemm = exl3::ReconGemm::Create();
  if (!gemm) {
    return Error("the reconstruction GEMM was refused: " + gemm.error().detail);
  }
  gemm_ = std::move(*gemm);
  return {};
}

// Plans and binds every phase of the run before any runs.
Status Run::PlanAll() {
  if (!table_) {
    return Error("no launch table to plan with");
  }
  const model::Exl3LaunchTable& table = *table_;
  for (const int prefix : o_.prefixes) {
    if (prefix + kSuffix > static_cast<int>(ids_.size())) {
      return Error(std::format("prefix {} and its steps exceed the held-out IDs", prefix));
    }
    for (int index = 0; index <= kSuffix; ++index) {
      const model::Exl3Phase phase = index == 0
                                         ? model::Exl3Phase{.rows = prefix, .past = 0}
                                         : model::Exl3Phase{.rows = 1, .past = prefix + index - 1};
      auto plan = model::PlanPhase(profile_, binding_, table, o_.arm, phase);
      if (!plan) {
        return Error(std::format("prefix {} phase {}: {}", prefix, index, plan.error()));
      }
      phases_.push_back(
          {.prefix = prefix, .index = index, .plan = std::move(*plan), .program = {}});
    }
  }
  std::uint64_t region = 0;
  for (const PhaseRun& phase : phases_) {
    region = std::max(region, phase.plan.region);
    host_inputs_bytes_ = std::max(host_inputs_bytes_, exl3::HostInputsLayout(phase.plan).bytes);
  }
  auto activations = device_->Allocate(region, "the activations");
  auto inputs = device_->Pinned(host_inputs_bytes_, "the host inputs");
  if (!activations || !inputs) {
    return std::unexpected(!activations ? activations.error() : inputs.error());
  }
  memory_.region = *activations;
  memory_.region_bytes = region;
  host_inputs_ = *inputs;
  // Bind every phase with a planning context (no workspace), for the pool
  // scratch the largest attention draws; then the real context.
  auto planning = kg::LaunchContext::Create(0, device_->execution(), device_->stream(),
                                            {.base = 0, .size = llmp::base::Bytes(0)});
  if (!planning) {
    return Error("the GGML planning context was refused: " + planning.error().detail);
  }
  std::uint64_t scratch = 0;
  for (PhaseRun& phase : phases_) {
    auto program =
        exl3::Qwen2Program::Bind(*registry_, profile_, phase.plan, memory_, **planning, *launch_);
    if (!program) {
      return Error(std::format("prefix {} phase {} did not bind: {}", phase.prefix, phase.index,
                               program.error().detail));
    }
    scratch = std::max(scratch, (*program)->ggml_scratch());
    phase.program = std::move(*program);
  }
  planning->reset();
  auto pool = device_->Allocate(scratch, "the GGML pool");
  if (!pool) {
    return std::unexpected(pool.error());
  }
  auto ggml = kg::LaunchContext::Create(0, device_->execution(), device_->stream(),
                                        {.base = *pool, .size = llmp::base::Bytes(scratch)});
  if (!ggml) {
    return Error("the GGML launch context was refused: " + ggml.error().detail);
  }
  ggml_ = std::move(*ggml);
  std::println("planned {} phases: region {} bytes, GGML pool {} bytes", phases_.size(), region,
               scratch);
  return {};
}

void Run::Flush() {
  if (recording_) {
    for (const auto& event : recording_->Take()) {
      record_ += llmp::test_support::EventLine(event);
    }
  }
}

std::expected<void, exl3::KernelFailure> Run::Before(const PhaseRun& phase, std::size_t op) {
  const model::Exl3Op& o = phase.plan.ops[op];
  if (recording_) {
    Flush();
    record_ += llmp::test_support::OpLine(o.name, o.layer);
  }
  if (!o_.record_ops.empty() && ops_bin_.is_open()) {
    op_inputs_.clear();
    const auto fail = [](std::string detail) {
      return std::unexpected(
          exl3::KernelFailure{.error = exl3::KernelError::kRejected, .detail = std::move(detail)});
    };
    for (const std::string& name : o.inputs) {
      std::string sha;
      std::string dtype;
      std::string shape;
      if (const auto t = phase.plan.tensors.find(name); t != phase.plan.tensors.end()) {
        auto hash = HashDevice(phase.program->Address(op, name), t->second.bytes());
        if (!hash) {
          return fail(hash.error());
        }
        sha = *hash;
        dtype = model::Exl3DtypeName(t->second.dtype);
        for (const std::int64_t d : t->second.shape) {
          shape += std::format("{}{}", shape.empty() ? "" : ",", d);
        }
      } else if (name == "k_cache" || name == "v_cache") {
        const std::uint64_t bytes =
            static_cast<std::uint64_t>(phase.plan.padded) * profile_.kv_width() * 2;
        auto hash = HashDevice(phase.program->Address(op, name), bytes);
        if (!hash) {
          return fail(hash.error());
        }
        sha = *hash;
        dtype = "float16";
        shape = std::format("{},{},{}", phase.plan.padded, profile_.kv_heads, profile_.head_dim);
      } else {
        // The load-time hash of what lies at the address the program bound
        // for this weight: one bound to another layer's or tensor's bytes
        // records that tensor's hash, which op_tier_e.py holds to the
        // artifact's tensor of this name and layer.
        const std::uint64_t address = phase.program->Address(op, name);
        const auto found = weight_hash_.find(address);
        if (address == 0 || found == weight_hash_.end()) {
          return fail(std::format("no load-time hash of weight {} (layer {}) at its bound address",
                                  name, o.layer));
        }
        sha = found->second;
        if (name == "embed_table") {
          dtype = "bfloat16";
        } else {
          dtype = name.ends_with(".bias") ? "float16" : "float32";
        }
        shape = name == "embed_table"
                    ? std::format("{},{}", profile_.vocab, profile_.width)
                    : std::format("{}", name.ends_with(".bias") && !name.starts_with("q_")
                                            ? profile_.kv_width()
                                            : profile_.width);
      }
      op_inputs_ +=
          std::format(R"({}{{"name": "{}", "dtype": "{}", "shape": [{}], "sha256": "{}"}})",
                      op_inputs_.empty() ? "" : ", ", name, dtype, shape, sha);
    }
    // The linears' weights, by the load-time hash of what lies at each
    // address the program bound (an address no weight was loaded to
    // records "unknown"), and for the multi-GEMM, what lies at the two
    // addresses each of its device tables holds.
    op_weights_.clear();
    const auto hash_at = [&](std::uint64_t address) -> std::string {
      const auto found = weight_hash_.find(address);
      return found == weight_hash_.end() ? std::format("unknown 0x{:x}", address) : found->second;
    };
    const auto note = [&](std::string_view name, const std::string& sha) {
      op_weights_ += std::format(R"({}{{"name": "{}", "sha256": "{}"}})",
                                 op_weights_.empty() ? "" : ", ", name, sha);
    };
    for (const auto& [name, address] : phase.program->BoundWeights(op)) {
      if (std::ranges::find(o.inputs, name) != o.inputs.end()) {
        continue;  // an input, recorded above
      }
      if (!name.starts_with("table.")) {
        note(name, hash_at(address));
        continue;
      }
      std::array<std::uint64_t, 2> held{};
      if (auto r = device_->Download(address, std::as_writable_bytes(std::span(held))); !r) {
        return fail(r.error());
      }
      note(name + "[0]", hash_at(held[0]));
      note(name + "[1]", hash_at(held[1]));
    }
  }
  return {};
}

std::expected<void, exl3::KernelFailure> Run::After(const PhaseRun& phase, std::size_t op) {
  const model::Exl3Op& o = phase.plan.ops[op];
  const auto fail = [](std::string detail) {
    return std::unexpected(
        exl3::KernelFailure{.error = exl3::KernelError::kRejected, .detail = std::move(detail)});
  };
  // Tier C's captures, during the prefill.
  if (o_.capture && phase.index == 0) {
    const auto grab =
        [&](std::string_view tensor,
            std::vector<std::byte>& into) -> std::expected<void, exl3::KernelFailure> {
      const std::uint64_t bytes = phase.plan.tensors.at(std::string(tensor)).bytes();
      const std::size_t at = into.size();
      into.resize(at + bytes);
      if (auto r = device_->Download(phase.program->Address(op, tensor),
                                     std::span(into).subspan(at, bytes));
          !r) {
        return fail(r.error());
      }
      return {};
    };
    if (o.name == "mlp_residual_add") {
      if (auto r = grab("resid.out", blocks_[phase.prefix]); !r) {
        return r;
      }
    } else if (o.name == "final_norm.cast") {
      if (auto r = grab("final_norm.out", final_norm_[phase.prefix]); !r) {
        return r;
      }
    }
  }
  if (!o_.record_ops.empty() && ops_bin_.is_open()) {
    std::string outputs;
    for (const std::string& name : o.outputs) {
      std::uint64_t address = 0;
      std::uint64_t bytes = 0;
      std::string dtype;
      std::string shape;
      std::string extra;
      if (name == "k_cache" || name == "v_cache") {
        address = phase.program->Address(op, name);
        bytes = static_cast<std::uint64_t>(phase.plan.phase.rows) * profile_.kv_width() * 2;
        dtype = "float16";
        shape =
            std::format("{},{},{}", phase.plan.phase.rows, profile_.kv_heads, profile_.head_dim);
        extra = std::format(R"(, "cells": [{}, {}])", phase.plan.phase.past,
                            phase.plan.phase.past + phase.plan.phase.rows);
      } else {
        const model::Exl3Tensor& t = phase.plan.tensors.at(name);
        address = phase.program->Address(op, name);
        bytes = t.bytes();
        dtype = model::Exl3DtypeName(t.dtype);
        for (const std::int64_t d : t.shape) {
          shape += std::format("{}{}", shape.empty() ? "" : ",", d);
        }
      }
      std::vector<std::byte> host(bytes);
      if (auto r = device_->Download(address, host); !r) {
        return fail(r.error());
      }
      std::string stored;
      if (name != "logits") {
        ops_bin_.write(reinterpret_cast<const char*>(host.data()),
                       static_cast<std::streamsize>(host.size()));
        stored = std::format(R"(, "offset": {}, "bytes": {})", ops_at_, bytes);
        ops_at_ += bytes;
      }
      outputs +=
          std::format(R"({}{{"name": "{}", "dtype": "{}", "shape": [{}], "sha256": "{}"{}{}}})",
                      outputs.empty() ? "" : ", ", name, dtype, shape, Hex(host), stored, extra);
    }
    ops_json_ += std::format(
        R"({}    {{"op": "{}", "layer": {}, "inputs": [{}], "weights": [{}], "outputs": [{}]}})",
        ops_json_.empty() ? "" : ",\n", o.name, o.layer, op_inputs_, op_weights_, outputs);
  }
  return {};
}

Status Run::RunPhase(PhaseRun& phase, int evaluation) {
  const model::Exl3PhasePlan& plan = phase.plan;
  const int rows = plan.phase.rows;
  const std::span<const std::int32_t> tokens(ids_.data() + plan.phase.past,
                                             static_cast<std::size_t>(rows));
  if (auto r = device_->Finish(); !r) {  // the host inputs are free again
    return r;
  }
  if (auto written = exl3::WriteHostInputs(profile_, plan, tokens,
                                           std::span(host_inputs_, host_inputs_bytes_));
      !written) {
    return Error(written.error().detail);
  }
  const bool first = evaluation == 1;
  if (recording_ && first) {
    Flush();
    record_ += llmp::test_support::ChunkLine(
        {.evaluation = evaluation, .chunk = phase.index, .rows = rows, .n_past = plan.phase.past});
  }
  const bool ops = first && !o_.record_ops.empty();
  if (ops) {
    const auto dir = o_.record_ops / std::to_string(phase.prefix);
    std::filesystem::create_directories(dir);
    ops_bin_.open(dir / std::format("{}.bin", phase.index), std::ios::binary | std::ios::trunc);
    ops_json_.clear();
    ops_at_ = 0;
  }
  exl3::Qwen2Hooks hooks;
  if (first && (recording_ || o_.capture || ops)) {
    hooks.before = [&](std::size_t op) { return Before(phase, op); };
    hooks.after = [&](std::size_t op) { return After(phase, op); };
  }
  if (auto ran = phase.program->Run(*ggml_, *launch_, *gemm_, device_->execution(),
                                    device_->stream(), Address(host_inputs_), hooks);
      !ran) {
    return Error(
        std::format("prefix {} phase {}: {}", phase.prefix, phase.index, ran.error().detail));
  }
  if (recording_ && first) {
    Flush();
    record_ += llmp::test_support::OpLine("outputs", -1);
  }
  // The logits, F16, widened exactly.
  const std::uint64_t logits_bytes = plan.tensors.at("logits").bytes();
  std::vector<std::uint16_t> half(logits_bytes / 2);
  auto stream = device_->Stream();
  if (!stream) {
    return std::unexpected(stream.error());
  }
  if (auto r = Cuda(
          cudaMemcpyAsync(half.data(), Pointer(memory_.region + plan.slots.at("logits").offset),
                          logits_bytes, cudaMemcpyDeviceToHost, *stream),
          "the logits' download");
      !r) {
    return r;
  }
  if (auto r = device_->Finish(); !r) {
    return r;
  }
  if (recording_ && first) {
    Flush();
    record_ += llmp::test_support::EndChunkLine();
  }
  std::vector<float>& out = current_[phase.prefix];
  if (phase.index == 0) {
    out.clear();
  }
  for (const std::uint16_t v : half) {
    out.push_back(model::HalfToFloat(v));
  }
  if (ops) {
    ops_bin_.close();
    std::ofstream json(o_.record_ops / std::to_string(phase.prefix) /
                       std::format("{}.json", phase.index));
    json << std::format(
        "{{\"prefix\": {}, \"phase\": {}, \"rows\": {}, \"past\": {}, \"npad\": {},\n "
        "\"ops\": [\n{}\n]}}\n",
        phase.prefix, phase.index, rows, plan.phase.past, plan.padded, ops_json_);
  }
  // Tier C: the cache after the prefill and after the steps.
  if (first && o_.capture && (phase.index == 0 || phase.index == kSuffix)) {
    const int cells = phase.index == 0 ? phase.prefix : phase.prefix + kSuffix;
    const std::uint64_t cell = std::uint64_t{profile_.kv_width()} * 2;
    auto& into = phase.index == 0 ? kv_prefill_[phase.prefix] : kv_suffix_[phase.prefix];
    for (const exl3::Qwen2Layer& layer : memory_.layers) {
      for (const std::uint64_t base : {layer.k_cache, layer.v_cache}) {
        const std::size_t at = into.size();
        into.resize(at + (cell * static_cast<std::uint64_t>(cells)));
        if (auto r = device_->Download(base, std::span(into).subspan(at)); !r) {
          return r;
        }
      }
    }
  }
  return {};
}

Status Run::Evaluate(int evaluation) {
  current_.clear();
  for (PhaseRun& phase : phases_) {
    if (phase.index == 0) {
      // A fresh trajectory: the whole cache cleared, padded cells included.
      auto stream = device_->Stream();
      if (!stream) {
        return std::unexpected(stream.error());
      }
      if (auto r = Cuda(cudaMemsetAsync(Pointer(kv_), 0, kv_bytes_, *stream), "the cache's clear");
          !r) {
        return r;
      }
    }
    if (auto r = RunPhase(phase, evaluation); !r) {
      return r;
    }
  }
  if (evaluation == 1) {
    logits_ = current_;
    return {};
  }
  for (const auto& [prefix, values] : logits_) {
    const auto& other = current_.at(prefix);
    const bool same = values.size() == other.size() &&
                      std::memcmp(values.data(), other.data(), values.size() * 4) == 0;
    evaluations_ += std::format(R"({}{{"evaluation": {}, "prefix": {}, "identical": {}}})",
                                evaluations_.empty() ? "" : ", ", evaluation, prefix, same);
    if (!same) {
      ++mismatches_;
    }
  }
  return {};
}

Status Run::Write() {
  std::filesystem::create_directories(o_.out);
  std::string logits;
  const auto vocab = static_cast<std::int64_t>(profile_.vocab);
  for (const auto& [prefix, values] : logits_) {
    const std::size_t prefill = static_cast<std::size_t>(prefix) * profile_.vocab;
    const auto bytes = std::as_bytes(std::span(values));
    if (auto r = WriteNpy(o_.out / std::format("logits-{}.prefill.npy", prefix), "<f4",
                          {prefix, vocab}, bytes.first(prefill * 4));
        !r) {
      return r;
    }
    if (auto r = WriteNpy(o_.out / std::format("logits-{}.suffix.npy", prefix), "<f4",
                          {kSuffix, vocab}, bytes.subspan(prefill * 4));
        !r) {
      return r;
    }
    logits += std::format(R"({}"{}": {{"prefill_sha256": "{}", "suffix_sha256": "{}"}})",
                          logits.empty() ? "" : ", ", prefix, Hex(bytes.first(prefill * 4)),
                          Hex(bytes.subspan(prefill * 4)));
  }
  for (const auto& [prefix, bytes] : blocks_) {
    const auto width = static_cast<std::int64_t>(profile_.width);
    const auto layers = static_cast<std::int64_t>(profile_.layers);
    const auto kv_heads = static_cast<std::int64_t>(profile_.kv_heads);
    const auto head = static_cast<std::int64_t>(profile_.head_dim);
    if (auto r = WriteNpy(o_.out / std::format("capture-{}.blocks.npy", prefix), "<f4",
                          {layers, prefix, width}, bytes);
        !r) {
      return r;
    }
    if (auto r = WriteNpy(o_.out / std::format("capture-{}.final_norm.npy", prefix), "<f2",
                          {prefix, width}, final_norm_.at(prefix));
        !r) {
      return r;
    }
    if (auto r = WriteNpy(o_.out / std::format("capture-{}.kv.npy", prefix), "<f2",
                          {2 * layers, prefix, kv_heads, head}, kv_prefill_.at(prefix));
        !r) {
      return r;
    }
    if (auto r = WriteNpy(o_.out / std::format("capture-{}.kv_after_suffix.npy", prefix), "<f2",
                          {2 * layers, prefix + kSuffix, kv_heads, head}, kv_suffix_.at(prefix));
        !r) {
      return r;
    }
  }
  std::string identities;
  for (const PhaseRun& phase : phases_) {
    identities += std::format(
        R"({}{{"prefix": {}, "phase": {}, "rows": {}, "past": {}, "npad": {}, "plan": "{}", "launch": "{}", "region": {}, "ggml_scratch": {}}})",
        identities.empty() ? "" : ",\n  ", phase.prefix, phase.index, phase.plan.phase.rows,
        phase.plan.phase.past, phase.plan.padded, llmp::base::ToHex(phase.program->identity()),
        llmp::base::ToHex(phase.plan.launch_digest), phase.plan.region,
        phase.program->ggml_scratch());
  }
  if (!table_) {
    return Error("no launch table to record");
  }
  std::ofstream manifest(o_.out / "manifest.json");
  manifest << std::format(
      "{{\"harness\": \"llmp_exl3_exec\", \"fixture\": \"{}\", \"arm\": \"{}\", \"artifact\": "
      "\"{}\", \"plan_file_sha256\": \"{}\", \"launch_table\": \"{}\", \"memory\": \"cudaMalloc\", "
      "\"capture\": {}, \"record_ops\": {}, \"evaluations\": {}, \"mismatches\": {},\n "
      "\"comparisons\": [{}],\n \"logits\": {{{}}},\n \"phases\": [\n  {}]}}\n",
      o_.fixture, o_.arm == model::Exl3Arm::kG ? "G" : "O", artifact_->id(), HexFile(o_.plan),
      table_->source(), o_.capture, !o_.record_ops.empty(), o_.evaluations, mismatches_,
      evaluations_, logits, identities);
  if (o_.record) {
    Flush();
    std::ofstream file(o_.out / "record.jsonl");
    file << llmp::test_support::HeaderLine(
                std::format("llmp_exl3_exec {} {}", o_.fixture,
                            o_.arm == model::Exl3Arm::kG ? "EXL3-G" : "EXL3-O"),
                llmp::test_support::LoadedCublas())
         << record_;
  }
  if (!o_.record_ops.empty()) {
    std::string prefixes;
    for (const int prefix : o_.prefixes) {
      prefixes += std::format("{}{}", prefixes.empty() ? "" : ", ", prefix);
    }
    std::ofstream file(o_.record_ops / "manifest.json");
    file << std::format(
        "{{\"format\": \"llmp-exl3-ops/1\", \"fixture\": \"{}\", \"arm\": \"{}\", \"artifact\": "
        "\"{}\", \"prefixes\": [{}], \"suffix\": {}}}\n",
        o_.fixture, o_.arm == model::Exl3Arm::kG ? "G" : "O", artifact_->id(), prefixes, kSuffix);
  }
  return mismatches_ == 0 ? Status() : Error(std::format("{} evaluations differ", mismatches_));
}

Status Run::Execute() {
  if (auto r = Setup(); !r) {
    return r;
  }
  if (auto r = PlanAll(); !r) {
    return r;
  }
  if (!o_.record_ops.empty()) {
    // The weights each operation reads, hashed once (they are never
    // written), by device address; two weights at one address are refused.
    std::string failed;
    const auto note = [&](const std::string& key, std::uint64_t address, std::uint64_t bytes) {
      auto hash = HashDevice(address, bytes);
      if (!hash || !weight_hash_.emplace(address, *hash).second) {
        failed = key;
        return false;
      }
      return true;
    };
    bool ok =
        note("embed_table", memory_.embed, std::uint64_t{profile_.vocab} * profile_.width * 2) &&
        note("final_norm.w", memory_.final_norm, std::uint64_t{profile_.width} * 4);
    for (std::uint32_t l = 0; l < profile_.layers && ok; ++l) {
      const exl3::Qwen2Layer& layer = memory_.layers[l];
      ok =
          note(std::format("{}:attn_norm.w", l), layer.attn_norm,
               std::uint64_t{profile_.width} * 4) &&
          note(std::format("{}:mlp_norm.w", l), layer.mlp_norm,
               std::uint64_t{profile_.width} * 4) &&
          note(std::format("{}:q_proj.bias", l), layer.q.bias, std::uint64_t{profile_.width} * 2) &&
          note(std::format("{}:k_proj.bias", l), layer.k.bias,
               std::uint64_t{profile_.kv_width()} * 2) &&
          note(std::format("{}:v_proj.bias", l), layer.v.bias,
               std::uint64_t{profile_.kv_width()} * 2);
    }
    // Every linear's trellis and side vectors, the artifact's resource
    // bytes at the address each was loaded to.
    std::vector<const model::Exl3LinearBinding*> linears = {&binding_.lm_head};
    for (const model::Exl3LayerBinding& b : binding_.layers) {
      linears.insert(linears.end(), {&b.q, &b.k, &b.v, &b.o, &b.gate, &b.up, &b.down});
    }
    for (const model::Exl3LinearBinding* l : linears) {
      for (const auto& [part, resource] :
           {std::pair{"trellis", l->trellis}, std::pair{"suh", l->suh}, std::pair{"svh", l->svh}}) {
        ok = ok && note(std::format("{}.{}", l->name, part), WeightAddress(resource),
                        artifact_->resources()[resource].bytes.value());
      }
    }
    if (!ok) {
      return Error(std::format("the weight {} could not be hashed, or shares an address", failed));
    }
  }
  if (o_.record) {
    recording_ = std::make_unique<llmp::test_support::Recording>();
  }
  for (int evaluation = 1; evaluation <= o_.evaluations; ++evaluation) {
    if (auto r = Evaluate(evaluation); !r) {
      return r;
    }
    if (evaluation == 1) {
      recording_.reset();  // the first evaluation's record only
    }
  }
  return Write();
}

}  // namespace

int main(int argc, char** argv) {
  auto options = Parse(std::span(argv, static_cast<std::size_t>(argc)));
  if (!options) {
    std::println(stderr, "{}", options.error());
    return 2;
  }
  Run run(*options);
  if (auto r = run.Execute(); !r) {
    std::println(stderr, "FAILED: {}", r.error());
    return 1;
  }
  std::println("DONE {}", options->out.string());
  return 0;
}
