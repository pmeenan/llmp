// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Backend-proof P2, oracle rung 3 (docs/backend-proof.md): native
// execution of the Qwen2.5-0.5B FP16 fixture from its v0 prepared artifact
// on cudaMalloc memory, for the FP16 Tier E gate and the allocation census
// (docs/experiments/backend-proof-p2/README.md).
//
//   llmp_fp16_exec --artifact DIR --trajectory control|heldout --tokens FILE
//                    --fusion on|off --out DIR [--evaluations N]
//                    [--record | --census]
//
// - Weights: the artifact is opened as untrusted input (artifact.h), its
//   resources bound to the compiled-in Qwen2 profile (model/qwen2.h), and
//   every chunk of every group read with direct I/O, one coalesced read plan
//   (Artifact::PlanReads), into pinned staging; each chunk's SHA-256 must
//   equal the index's before it is copied to one cudaMalloc region, group by
//   group. The token table is also kept on the host, where the embedding
//   rows are looked up, as llama.cpp does.
// - Each chunk: the host builds its inputs (positions, K and V cells, mask,
//   output rows, embedding rows), the GGML graph is built as llama.cpp
//   builds it (kernels/ggml/qwen2_graph.h), planned with fusion on (FP16-F)
//   or off (FP16-U) and its activations placed (graph_plan.h), bound to the
//   registry's implementations (executor.h), and run on one stream through
//   the K-C launch context. The inputs are copied in the bridge's order and
//   the logits come back to pinned memory.
// - The cuBLAS handle and its 32 MiB workspace are created where the bridge
//   creates them: before the first chunk whose plan calls cuBLAS.
// - --record: tests/support's launch recorder runs from the first line of
//   main on this, the only launching thread, and every chunk is marked, so
//   an nsys trace of the whole process lines up with it (plan_compare.py).
// - --census: the census readings and controls, as fp16_census.cc takes
//   them on the bridge, with this harness's own buffers declared, and six
//   tail readings after the last evaluation. A census run records nothing,
//   and a recorded run takes no readings. Since D-085 the census rules are
//   gone; run_native.sh peak runs this mode for the coarse memory check.
//
// Every evaluation clears the cache and runs the trajectory from the start;
// the first one's logits are written, and each later one must equal them
// bit for bit. Token IDs: `control`, the bridge's tokens.txt (one per
// line, 76); `heldout`, the declared held-out IDs (1,040 little-endian
// int64, SHA-256 6dd8da89..., the first 577 used).

#include <cuda.h>
#include <cuda_runtime.h>
#include <fcntl.h>
#include <malloc.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
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
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

#include "artifact/artifact.h"
#include "artifact/layout.h"
#include "base/bytes.h"
#include "base/sha256.h"
#include "execution/registry.h"
#include "ggml.h"
#include "kernels/ggml/cublas.h"
#include "kernels/ggml/executor.h"
#include "kernels/ggml/graph_plan.h"
#include "kernels/ggml/implementations.h"
#include "kernels/ggml/launch.h"
#include "kernels/ggml/qwen2_graph.h"
#include "kernels/ggml/tensors.h"
#include "launch_recorder.h"
#include "model/qwen2.h"
#include "plan_record.h"
#include "providers/cuda/cuda_device_execution.h"
#include "providers/device_execution.h"

namespace {

namespace kg = llmp::kernels::ggml;
using llmp::base::Bytes;
using Status = std::expected<void, std::string>;

std::unexpected<std::string> Error(std::string what) { return std::unexpected(std::move(what)); }

Status Cuda(cudaError_t result, std::string_view what) {
  if (result != cudaSuccess) {
    return Error(std::format("{}: {}", what, cudaGetErrorString(result)));
  }
  return {};
}

Status Driver(CUresult result, std::string_view what) {
  if (result != CUDA_SUCCESS) {
    return Error(std::format("{}: CUresult {}", what, static_cast<int>(result)));
  }
  return {};
}

void* Pointer(std::uint64_t address) {
  return reinterpret_cast<void*>(address);  // NOLINT(performance-no-int-to-ptr)
}

std::uint64_t Address(const void* pointer) { return reinterpret_cast<std::uintptr_t>(pointer); }

std::uint64_t Round(std::uint64_t bytes, std::uint64_t to) { return (bytes + to - 1) / to * to; }

// ------------------------------------------------------------------ census

constexpr int kTailReadings = 6;

// fp16_census.cc's readings: after a fixed settle (CENSUS_SETTLE_MS,
// 2,500 ms by default), /proc/meminfo's MemAvailable and SUnreclaim with the
// per-CPU page lists' pages (/proc/zoneinfo), three times back to back;
// RssAnon and RssFile; mallinfo2; every other process's RssAnon, summed; and
// CLOCK_REALTIME and CLOCK_MONOTONIC_RAW, which place the reading (and the
// start of its settle) on an nsys trace's timeline.
std::string Slurp(const char* path) {
  std::ifstream file(path);
  std::stringstream text;
  text << file.rdbuf();
  return text.str();
}

std::uint64_t FieldKib(const std::string& text, std::string_view key) {
  const std::string needle = std::format("\n{}:", key);
  const std::string padded = "\n" + text;
  const std::size_t at = padded.find(needle);
  if (at == std::string::npos) {
    return 0;
  }
  return std::strtoull(padded.c_str() + at + needle.size(), nullptr, 10) * 1024;
}

// Every other process's RssAnon, summed, and how many there are.
std::pair<std::uint64_t, std::uint64_t> OtherProcesses() {
  std::uint64_t rss_anon = 0;
  std::uint64_t processes = 0;
  const std::string self = std::to_string(getpid());
  std::error_code error;
  for (const auto& entry : std::filesystem::directory_iterator("/proc", error)) {
    const std::string name = entry.path().filename().string();
    if (name.empty() || !std::ranges::all_of(name, [](char c) { return c >= '0' && c <= '9'; }) ||
        name == self) {
      continue;
    }
    const std::string status = Slurp((entry.path() / "status").c_str());
    if (status.empty()) {
      continue;  // exited meanwhile
    }
    ++processes;
    rss_anon += FieldKib(status, "RssAnon");
  }
  return {rss_anon, processes};
}

std::int64_t ClockNs(clockid_t clock) {
  timespec now{};
  clock_gettime(clock, &now);
  return (static_cast<std::int64_t>(now.tv_sec) * 1'000'000'000) + now.tv_nsec;
}

std::uint64_t PerCpuListBytes() {
  std::ifstream file("/proc/zoneinfo");
  std::string line;
  std::uint64_t pages = 0;
  while (std::getline(file, line)) {
    const std::size_t at = line.find("count:");
    if (at != std::string::npos && line.find_first_not_of(" \t") == at) {
      pages += std::strtoull(line.c_str() + at + 6, nullptr, 10);
    }
  }
  return pages * static_cast<std::uint64_t>(sysconf(_SC_PAGESIZE));
}

class Census {
 public:
  explicit Census(bool on) : on_(on) {
    const char* settle = std::getenv("CENSUS_SETTLE_MS");  // NOLINT(concurrency-mt-unsafe)
    if (settle != nullptr) {
      const std::string_view text(settle);
      int value = 0;
      if (std::from_chars(text.data(), text.data() + text.size(), value).ec == std::errc{} &&
          value >= 0) {
        settle_ms_ = value;
      }
    }
  }
  bool on() const { return on_; }

  void Declare(const std::string& name, std::uint64_t bytes) { declared_[name] = bytes; }

  int settle_ms() const { return settle_ms_; }

  // A reading as a JSON object.
  std::string Take(std::string_view step, int evaluation = 0, int chunk = -1) const {
    const std::int64_t settle_from = ClockNs(CLOCK_MONOTONIC_RAW);
    std::this_thread::sleep_for(std::chrono::milliseconds(settle_ms_));
    const std::int64_t realtime = ClockNs(CLOCK_REALTIME);
    const std::int64_t raw = ClockNs(CLOCK_MONOTONIC_RAW);
    std::array<std::uint64_t, 3> available{};
    std::array<std::uint64_t, 3> lists{};
    std::array<std::uint64_t, 3> slab{};
    for (std::size_t i = 0; i < available.size(); ++i) {
      const std::string meminfo = Slurp("/proc/meminfo");
      available.at(i) = FieldKib(meminfo, "MemAvailable");
      lists.at(i) = PerCpuListBytes();
      slab.at(i) = FieldKib(meminfo, "SUnreclaim");
    }
    const std::string status = Slurp("/proc/self/status");
    const struct mallinfo2 m = mallinfo2();
    const auto [others_rss_anon, others_processes] = OtherProcesses();
    const auto triple = [](const std::array<std::uint64_t, 3>& v) {
      return std::format("[{},{},{}]", v[0], v[1], v[2]);
    };
    std::string out = std::format(
        R"({{"step":"{}","evaluation":{},"chunk":{},"settle_from_raw_ns":{},"realtime_ns":{},)"
        R"("raw_ns":{},"mem_available":{},"pcp":{},"sunreclaim":{},)"
        R"("rss_anon":{},"rss_file":{},"malloc_arena":{},"malloc_hblkhd":{},"malloc_uordblks":{},)"
        R"("others_rss_anon":{},"others_processes":{},"declared":{{)",
        step, evaluation, chunk, settle_from, realtime, raw, triple(available), triple(lists),
        triple(slab), FieldKib(status, "RssAnon"), FieldKib(status, "RssFile"), m.arena, m.hblkhd,
        m.uordblks, others_rss_anon, others_processes);
    bool first = true;
    for (const auto& [name, bytes] : declared_) {
      out += std::format(R"({}"{}":{})", first ? "" : ",", name, bytes);
      first = false;
    }
    return out + "}}";
  }

  void Read(std::string_view step, int evaluation = 0, int chunk = -1) {
    if (on_) {
      readings_.push_back(Take(step, evaluation, chunk));
    }
  }

  std::vector<std::string>& readings() { return readings_; }
  std::vector<std::string>& controls() { return controls_; }
  std::vector<std::string>& phases() { return phases_; }

 private:
  bool on_;
  int settle_ms_ = 2500;
  std::map<std::string, std::uint64_t> declared_;
  std::vector<std::string> readings_;
  std::vector<std::string> controls_;
  std::vector<std::string> phases_;
};

// ------------------------------------------------------------------ the device

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
  }

  llmp::providers::DeviceExecution& execution() { return *execution_; }
  llmp::providers::StreamId stream() const { return stream_; }

  // The stream's handle, noting that work is about to be queued on it.
  std::expected<cudaStream_t, std::string> Stream() {
    const auto native = execution_->Submission(stream_);
    if (!native) {
      return Error("Submission failed");
    }
    return static_cast<cudaStream_t>(native->handle);
  }

  // Everything queued on the stream has completed.
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
    if (!execution_->Release(*fence)) {
      return Error("Release failed");
    }
    return {};
  }

 private:
  Device(std::unique_ptr<llmp::providers::DeviceExecution> execution,
         llmp::providers::StreamId stream)
      : execution_(std::move(execution)), stream_(stream) {}

  std::unique_ptr<llmp::providers::DeviceExecution> execution_;
  llmp::providers::StreamId stream_;
};

// The census rule's controls (fp16_census.cc's, with the device clear on
// the stream so the recorder would see it).
Status Controls(Census& census, Device& device, std::string_view where, bool probe) {
  constexpr std::size_t kControl = std::size_t{64} << 20;
  constexpr std::size_t kExtent = std::size_t{2} << 20;
  const auto record = [&](std::string_view kind, int repeat, const std::string& before,
                          const std::string& held, const std::string& after) {
    census.controls().push_back(
        std::format(R"({{"kind":"{}","repeat":{},"where":"{}","before":{},"held":{},"after":{}}})",
                    kind, repeat, where, before, held, after));
  };
  for (int repeat = 0; repeat < 3; ++repeat) {
    {
      const std::string before = census.Take("control");
      void* p = nullptr;
      if (auto r = Cuda(cudaMalloc(&p, kControl), "cudaMalloc control"); !r) {
        return r;
      }
      auto stream = device.Stream();
      if (!stream) {
        return std::unexpected(stream.error());
      }
      if (auto r = Cuda(cudaMemsetAsync(p, 1, kControl, *stream), "control clear"); !r) {
        return r;
      }
      if (auto r = device.Finish(); !r) {
        return r;
      }
      const std::string held = census.Take("control");
      if (auto r = Cuda(cudaFree(p), "cudaFree control"); !r) {
        return r;
      }
      record("cudaMalloc", repeat, before, held, census.Take("control"));
    }
    {
      const std::string before = census.Take("control");
      CUmemAllocationProp prop{};
      prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
      prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
      prop.location.id = 0;
      CUdeviceptr base = 0;
      if (auto r = Driver(cuMemAddressReserve(&base, kControl, kExtent, 0, 0), "reserve"); !r) {
        return r;
      }
      std::vector<CUmemGenericAllocationHandle> handles(kControl / kExtent);
      for (std::size_t i = 0; i < handles.size(); ++i) {
        if (auto r = Driver(cuMemCreate(&handles[i], kExtent, &prop, 0), "create"); !r) {
          return r;
        }
        if (auto r = Driver(cuMemMap(base + (i * kExtent), kExtent, 0, handles[i], 0), "map"); !r) {
          return r;
        }
      }
      CUmemAccessDesc access{};
      access.location = prop.location;
      access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
      if (auto r = Driver(cuMemSetAccess(base, kControl, &access, 1), "access"); !r) {
        return r;
      }
      const std::string held = census.Take("control");
      (void)cuMemUnmap(base, kControl);
      for (const auto handle : handles) {
        (void)cuMemRelease(handle);
      }
      (void)cuMemAddressFree(base, kControl);
      record("vmm", repeat, before, held, census.Take("control"));
    }
    {
      const std::string before = census.Take("control");
      std::vector<unsigned char> host(kControl);
      unsigned char* p = host.data();
      std::memset(p, 1, kControl);
      asm volatile("" : : "r"(p) : "memory");
      const std::string held = census.Take("control");
      asm volatile("" : : "r"(p) : "memory");
      std::vector<unsigned char>().swap(host);  // freed, not kept as capacity
      record("host", repeat, before, held, census.Take("control"));
    }
    if (probe) {
      const std::string before = census.Take("control");
      void* p = nullptr;
      if (auto r = Cuda(cudaMallocHost(&p, kControl), "cudaMallocHost probe"); !r) {
        return r;
      }
      std::memset(p, 1, kControl);
      const std::string held = census.Take("control");
      (void)cudaFreeHost(p);
      record("pinned-probe", repeat, before, held, census.Take("control"));
    }
  }
  return {};
}

// ------------------------------------------------------------------ trajectories

struct Trajectory {
  std::string name;
  std::vector<std::int32_t> tokens;
  std::vector<std::uint32_t> chunks;
  std::uint32_t cells = 0;
};

std::expected<Trajectory, std::string> LoadTrajectory(std::string_view name,
                                                      const std::filesystem::path& tokens) {
  Trajectory t;
  t.name = std::string(name);
  std::ifstream file(tokens, std::ios::binary);
  if (!file) {
    return Error(std::format("cannot read {}", tokens.string()));
  }
  const std::string bytes((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
  if (name == "control") {
    std::istringstream in(bytes);
    std::int64_t id = 0;
    while (in >> id) {
      t.tokens.push_back(static_cast<std::int32_t>(id));
    }
    if (t.tokens.size() != 76) {
      return Error(std::format("control has 76 tokens, not {}", t.tokens.size()));
    }
    t.chunks.push_back(32);
    t.chunks.insert(t.chunks.end(), 44, 1);
    t.cells = 512;
  } else if (name == "heldout") {
    llmp::base::Sha256 hash;
    hash.Update(std::as_bytes(std::span(bytes)));
    const std::string digest = llmp::base::ToHex(hash.Finish());
    if (bytes.size() != std::size_t{1040} * 8 ||
        digest != "6dd8da897821f5615e7796f4d882795faf306a941f050e2eb68b619096e3fb0c") {
      return Error("not the declared held-out IDs");
    }
    t.chunks = {16, 17};
    t.chunks.insert(t.chunks.end(), 16, 1);
    t.chunks.push_back(512);
    t.chunks.insert(t.chunks.end(), 16, 1);
    std::size_t total = 0;
    for (const std::uint32_t c : t.chunks) {
      total += c;
    }
    for (std::size_t i = 0; i < total; ++i) {
      std::uint64_t id = 0;
      for (int b = 7; b >= 0; --b) {
        id = (id << 8U) | static_cast<unsigned char>(bytes[(i * 8) + static_cast<std::size_t>(b)]);
      }
      t.tokens.push_back(static_cast<std::int32_t>(id));
    }
    t.cells = 1024;
  } else {
    return Error("the trajectory is control or heldout");
  }
  return t;
}

// ------------------------------------------------------------------ weights

struct Weights {
  std::uint64_t base = 0;                  // one cudaMalloc region
  std::vector<std::uint64_t> group_base;   // each group's offset in it
  std::uint64_t bytes = 0;                 // the region
  std::vector<std::uint16_t> token_table;  // the F16 token table, on the host
  std::uint64_t chunks_verified = 0;
  std::uint64_t bytes_read = 0;
  std::uint64_t resource_bytes = 0;  // the tensors' own bytes, without the groups' pads
  void* staging = nullptr;           // pinned, for the loads
};

std::uint64_t ResourceAddress(const llmp::artifact::Artifact& artifact, const Weights& weights,
                              std::uint32_t resource) {
  const auto& r = artifact.resources()[resource];
  return weights.base + weights.group_base[r.group] + r.offset.value();
}

Status LoadWeights(const llmp::artifact::Artifact& artifact,
                   const llmp::model::Qwen2Binding& binding, Device& device, Census& census,
                   Weights& weights) {
  const auto groups = artifact.groups();
  weights.group_base.resize(groups.size());
  for (std::size_t g = 0; g < groups.size(); ++g) {
    weights.group_base[g] = weights.bytes;
    weights.bytes += Round(groups[g].stored.value(), 256);
  }
  void* region = nullptr;
  if (auto r = Cuda(cudaMalloc(&region, weights.bytes), "the weights' region"); !r) {
    return r;
  }
  weights.base = Address(region);
  census.Declare("device:weights", weights.bytes);

  for (const auto& resource : artifact.resources()) {
    weights.resource_bytes += resource.bytes.value();
  }
  const auto& table = artifact.resources()[binding.token_embd];
  weights.token_table.assign(table.bytes.value() / 2, 0);
  census.Declare("host:token-table", table.bytes.value());

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
  const std::uint64_t staging_bytes = limits.max_run.value();
  void* staging = nullptr;
  if (auto r = Cuda(cudaMallocHost(&staging, staging_bytes), "pinned staging"); !r) {
    return r;
  }
  census.Declare("pinned:staging", staging_bytes);
  std::vector<llmp::artifact::FileDescriptor> shards;
  for (std::uint32_t s = 0; s < artifact.shards().size(); ++s) {
    auto fd = artifact.OpenShardForDirectRead(s);
    if (!fd) {
      return Error(std::format("shard {} cannot be opened for direct reads", s));
    }
    shards.push_back(std::move(*fd));
  }
  auto* bytes = static_cast<std::byte*>(staging);
  const std::uint64_t table_offset = table.offset.value();
  const std::uint64_t table_end = table_offset + table.bytes.value();
  for (const auto& run : *runs) {
    if (run.length.value() > staging_bytes) {
      return Error("a read run exceeds the staging buffer");
    }
    std::uint64_t done = 0;
    while (done < run.length.value()) {
      const ssize_t got = pread(shards[run.shard].get(), bytes + done, run.length.value() - done,
                                static_cast<off_t>(run.file_offset.value() + done));
      if (got <= 0) {
        return Error(std::format("a direct read of shard {} failed", run.shard));
      }
      done += static_cast<std::uint64_t>(got);
    }
    weights.bytes_read += done;
    auto stream = device.Stream();
    if (!stream) {
      return std::unexpected(stream.error());
    }
    std::uint64_t at = 0;
    for (const auto& segment : run.segments) {
      const auto& group = groups[segment.chunk.group];
      const std::span<const std::byte> piece(bytes + at, segment.length.value());
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
      if (segment.chunk.group == table.group) {
        const std::uint64_t lo = std::max(table_offset, segment.group_offset.value());
        const std::uint64_t hi = std::min(table_end, segment.group_offset.value() + piece.size());
        if (lo < hi) {
          std::memcpy(
              reinterpret_cast<std::byte*>(weights.token_table.data()) + (lo - table_offset),
              piece.data() + (lo - segment.group_offset.value()), hi - lo);
        }
      }
      at += segment.length.value();
    }
    if (auto r = device.Finish(); !r) {
      return r;
    }
  }
  std::uint64_t chunks = 0;
  for (const auto& group : groups) {
    chunks += group.chunks;
  }
  if (weights.chunks_verified != chunks || chunks != artifact.chunk_sha256().size()) {
    return Error("not every chunk was read and verified");
  }
  // The staging stays allocated, and declared, to the end: the runtime
  // keeps freed pinned memory rather than returning it (census rule).
  weights.staging = staging;
  return {};
}

// ------------------------------------------------------------------ chunks

struct ChunkGraph {
  std::optional<kg::TensorArena> arena;
  kg::Qwen2Graph graph;
  kg::GraphPlan plan;
  kg::Placement placement;
};

// Builds, binds, plans and places one chunk's graph; `activations` is the
// region its computed tensors and device inputs live in (0 to measure).
std::expected<ChunkGraph, std::string> PlanChunk(
    const llmp::model::Qwen2Profile& profile, const llmp::artifact::Artifact& artifact,
    const llmp::model::Qwen2Binding& binding, const Weights& weights, std::uint64_t kv_base,
    std::uint32_t cells, std::uint32_t rows, std::uint32_t n_kv, bool fusion,
    const kg::DeviceChoices& choices, std::uint64_t activations, std::uint64_t activation_bytes) {
  ChunkGraph out;
  auto arena = kg::TensorArena::Create(kg::Qwen2GraphTensors(profile));
  if (!arena) {
    return Error(arena.error().detail);
  }
  out.arena.emplace(std::move(*arena));
  auto graph =
      kg::BuildQwen2Graph(*out.arena, profile, {.rows = rows, .n_kv = n_kv, .cells = cells});
  if (!graph) {
    return Error(graph.error().detail);
  }
  out.graph = std::move(*graph);
  kg::Qwen2Graph& g = out.graph;
  const auto bind = [&](ggml_tensor* t, std::uint32_t resource) {
    kg::TensorArena::Bind(t, ResourceAddress(artifact, weights, resource));
  };
  const std::uint64_t layer_cache = std::uint64_t{profile.kv_width()} * cells * 2;
  for (std::uint32_t il = 0; il < profile.layers; ++il) {
    const auto& b = binding.layers[il];
    auto& l = g.layers[il];
    bind(l.attn_norm, b.attn_norm);
    bind(l.q, b.q);
    bind(l.q_bias, b.q_bias);
    bind(l.k, b.k);
    bind(l.k_bias, b.k_bias);
    bind(l.v, b.v);
    bind(l.v_bias, b.v_bias);
    bind(l.out, b.out);
    bind(l.ffn_norm, b.ffn_norm);
    bind(l.gate, b.gate);
    bind(l.up, b.up);
    bind(l.down, b.down);
    kg::TensorArena::Bind(l.k_cache, kv_base + (std::uint64_t{2} * il * layer_cache));
    kg::TensorArena::Bind(l.v_cache, kv_base + (((std::uint64_t{2} * il) + 1) * layer_cache));
  }
  bind(g.output_norm, binding.output_norm);
  bind(g.output, binding.output);

  // First pass: every computed tensor at its own address.
  constexpr std::uint64_t kDistinct = std::uint64_t{1} << 46U;
  for (ggml_tensor* input : g.inputs()) {
    kg::TensorArena::Bind(input, kDistinct - (std::uint64_t{1} << 40U));
  }
  kg::BindDistinct(g.nodes, kDistinct);
  auto first = kg::PlanGraph(g.nodes, fusion, choices);
  if (!first) {
    return Error(first.error().detail);
  }
  auto inputs = g.inputs();
  auto placement = kg::PlaceActivations(g.nodes, *first, inputs, 128);
  if (!placement) {
    return Error(placement.error().detail);
  }
  out.placement = std::move(*placement);
  if (activations == 0) {  // measuring only
    out.plan = std::move(*first);
    return out;
  }
  if (out.placement.extent > activation_bytes) {
    return Error("the activations exceed their region");
  }
  for (const auto& [tensor, offset] : out.placement.offsets) {
    kg::TensorArena::Bind(tensor, activations + offset);
  }
  kg::BindViews(g.nodes);
  auto second = kg::PlanGraph(g.nodes, fusion, choices);
  if (!second) {
    return Error(second.error().detail);
  }
  if (!kg::SamePlan(*first, *second)) {
    return Error("the plan changed once the activations were placed");
  }
  out.plan = std::move(*second);
  // 128-byte alignment of every tensor bound (the recording sees no
  // addresses).
  for (const ggml_tensor* t : g.nodes) {
    for (const ggml_tensor* src : t->src) {
      if (src != nullptr && src->view_src == nullptr && Address(src->data) % 128 != 0) {
        return Error(std::format("{} is not 128-byte aligned", src->name));
      }
    }
    if (t->view_src == nullptr && Address(t->data) % 128 != 0) {
      return Error(std::format("node {} is not 128-byte aligned", ggml_op_desc(t)));
    }
  }
  return out;
}

struct Options {
  std::filesystem::path artifact;
  std::string trajectory;
  std::filesystem::path tokens;
  bool fusion = true;
  std::filesystem::path out;
  int evaluations = 2;
  bool record = false;
  bool census = false;
};

std::expected<Options, std::string> Parse(std::span<char*> args) {
  Options o;
  bool fusion_set = false;
  for (std::size_t i = 1; i < args.size(); ++i) {
    const std::string_view a = args[i];
    const auto value = [&]() -> std::expected<std::string_view, std::string> {
      if (i + 1 >= args.size()) {
        return Error(std::format("{} needs a value", a));
      }
      return std::string_view(args[++i]);
    };
    if (a == "--record") {
      o.record = true;
      continue;
    }
    if (a == "--census") {
      o.census = true;
      continue;
    }
    const auto v = value();
    if (!v) {
      return std::unexpected(v.error());
    }
    if (a == "--artifact") {
      o.artifact = *v;
    } else if (a == "--trajectory") {
      o.trajectory = *v;
    } else if (a == "--tokens") {
      o.tokens = *v;
    } else if (a == "--fusion") {
      o.fusion = *v == "on";
      fusion_set = *v == "on" || *v == "off";
    } else if (a == "--out") {
      o.out = *v;
    } else if (a == "--evaluations") {
      if (std::from_chars(v->data(), v->data() + v->size(), o.evaluations).ec != std::errc{}) {
        return Error("--evaluations takes a number");
      }
    } else {
      return Error(std::format("unknown argument {}", a));
    }
  }
  if (o.artifact.empty() || o.trajectory.empty() || o.tokens.empty() || o.out.empty() ||
      !fusion_set || o.evaluations < 1 || (o.record && o.census)) {
    return Error(
        "usage: llmp_fp16_exec --artifact DIR --trajectory control|heldout --tokens FILE "
        "--fusion on|off --out DIR [--evaluations N] [--record | --census]");
  }
  return o;
}

Status Run(const Options& o, llmp::test_support::Recording* recording, std::string& record) {
  Census census(o.census);
  census.Read("start");
  auto device = Device::Open();
  if (!device) {
    return std::unexpected(device.error());
  }
  Device& d = **device;
  census.Read("context");
  if (census.on()) {
    if (auto r = Controls(census, d, "after context", false); !r) {
      return r;
    }
    census.Read("controls");
  }

  auto trajectory = LoadTrajectory(o.trajectory, o.tokens);
  if (!trajectory) {
    return std::unexpected(trajectory.error());
  }
  const Trajectory& t = *trajectory;
  auto artifact = llmp::artifact::Artifact::Open(o.artifact);
  if (!artifact) {
    return Error(std::format("the artifact was refused: {}", artifact.error().reason));
  }
  const llmp::model::Qwen2Profile& profile = llmp::model::Qwen25Instruct05B();
  auto binding = llmp::model::BindQwen2(profile, *artifact);
  if (!binding) {
    return std::unexpected(binding.error());
  }
  Weights weights;
  if (auto r = LoadWeights(*artifact, *binding, d, census, weights); !r) {
    return r;
  }

  // The cache: every layer's K, then V, [kv_width, cells] F16, cleared.
  const std::uint64_t layer_cache = std::uint64_t{profile.kv_width()} * t.cells * 2;
  const std::uint64_t kv_bytes = layer_cache * 2 * profile.layers;
  void* kv = nullptr;
  if (auto r = Cuda(cudaMalloc(&kv, kv_bytes), "the cache"); !r) {
    return r;
  }
  census.Declare("device:kv", kv_bytes);

  // Measure every chunk's activations and scratch on a launch context with
  // no workspace, then size the regions for the largest.
  auto measure =
      kg::LaunchContext::Create(0, d.execution(), d.stream(), {.base = 0, .size = Bytes(0)});
  if (!measure) {
    return Error(measure.error().detail);
  }
  std::uint64_t most_activations = 0;
  std::uint64_t most_scratch = 0;
  std::uint64_t most_inputs = 0;
  std::uint32_t most_rows = 0;
  {
    const kg::DeviceChoices choices = kg::DeviceChoicesOf(**measure);
    std::uint32_t n_past = 0;
    for (const std::uint32_t rows : t.chunks) {
      const std::uint32_t n_kv = llmp::model::PaddedKv(n_past + rows, t.cells);
      auto planned = PlanChunk(profile, *artifact, *binding, weights, Address(kv), t.cells, rows,
                               n_kv, o.fusion, choices, 0, 0);
      if (!planned) {
        return Error(std::format("chunk at {}: {}", n_past, planned.error()));
      }
      most_activations = std::max(most_activations, planned->placement.extent);
      auto scratch = kg::PlanScratch(**measure, planned->plan);
      if (!scratch) {
        return Error(scratch.error().detail);
      }
      most_scratch = std::max(most_scratch, *scratch);
      std::uint64_t inputs = 0;
      for (const ggml_tensor* input : planned->graph.inputs()) {
        inputs += Round(ggml_nbytes(input), 128);
      }
      most_inputs = std::max(most_inputs, inputs);
      most_rows = std::max(most_rows, rows);
      n_past += rows;
    }
  }
  measure->reset();
  void* activations = nullptr;
  void* scratch = nullptr;
  void* input_staging = nullptr;
  void* logits = nullptr;
  const std::uint64_t logits_bytes = std::uint64_t{most_rows} * profile.vocab * sizeof(float);
  if (auto r = Cuda(cudaMalloc(&activations, most_activations), "the activations"); !r) {
    return r;
  }
  if (most_scratch > 0) {
    if (auto r = Cuda(cudaMalloc(&scratch, most_scratch), "the pool scratch"); !r) {
      return r;
    }
  }
  if (auto r = Cuda(cudaMallocHost(&input_staging, most_inputs), "the input staging"); !r) {
    return r;
  }
  if (auto r = Cuda(cudaMallocHost(&logits, logits_bytes), "the logits"); !r) {
    return r;
  }
  census.Declare("device:activations", most_activations);
  census.Declare("device:scratch", most_scratch);
  census.Declare("pinned:inputs", most_inputs);
  census.Declare("pinned:logits", logits_bytes);
  auto launch = kg::LaunchContext::Create(0, d.execution(), d.stream(),
                                          {.base = Address(scratch), .size = Bytes(most_scratch)});
  if (!launch) {
    return Error(launch.error().detail);
  }
  auto registry = llmp::execution::Registry::Create(kg::Implementations());
  if (!registry) {
    return Error(registry.error().detail);
  }
  // Every evaluation's logits, kept for the comparison: allocated and
  // touched before the first phase, as the bridge's census harness does, so
  // no reading interval faults them in.
  std::vector<std::vector<float>> results(static_cast<std::size_t>(o.evaluations),
                                          std::vector<float>(t.tokens.size() * profile.vocab));
  census.Read("setup");

  std::unique_ptr<kg::CublasHandle> cublas;
  void* cublas_workspace = nullptr;
  std::uint64_t cublas_bytes = 0;
  for (int e = 1; e <= o.evaluations; ++e) {
    // A fresh cache for every evaluation, as the bridge's new context has.
    {
      auto stream = d.Stream();
      if (!stream) {
        return std::unexpected(stream.error());
      }
      if (auto r = Cuda(cudaMemsetAsync(kv, 0, kv_bytes, *stream), "clearing the cache"); !r) {
        return r;
      }
      if (auto r = d.Finish(); !r) {
        return r;
      }
    }
    std::vector<float>& result = results[static_cast<std::size_t>(e - 1)];
    std::uint32_t n_past = 0;
    for (std::size_t k = 0; k < t.chunks.size(); ++k) {
      const std::uint32_t rows = t.chunks[k];
      const int chunk = static_cast<int>(k);
      auto inputs = llmp::model::Qwen2ChunkInputs(profile, t.cells, n_past, rows);
      if (!inputs) {
        return std::unexpected(inputs.error());
      }
      std::vector<float> embd(std::size_t{rows} * profile.width);
      if (auto r = llmp::model::EmbedRows(weights.token_table, profile.width, profile.vocab,
                                          std::span(t.tokens).subspan(n_past, rows), embd);
          !r) {
        return r;
      }
      census.Read("before", e, chunk);
      auto planned = PlanChunk(profile, *artifact, *binding, weights, Address(kv), t.cells, rows,
                               inputs->n_kv, o.fusion, kg::DeviceChoicesOf(**launch),
                               Address(activations), most_activations);
      if (!planned) {
        return Error(std::format("chunk {}: {}", k, planned.error()));
      }
      const bool calls_cublas = std::ranges::any_of(
          planned->plan.steps, [](const auto& s) { return s.implementation == kg::kMulMatCublas; });
      if (calls_cublas && !cublas) {
        // The bridge creates its handle and workspace here, lazily, with
        // upstream's size for the device.
        int major = 0;
        int minor = 0;
        (void)cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0);
        (void)cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, 0);
        cublas_bytes = kg::CublasHandle::UpstreamWorkspace((100 * major) + (10 * minor)).value();
        if (auto r = Cuda(cudaMalloc(&cublas_workspace, cublas_bytes), "the cuBLAS workspace");
            !r) {
          return r;
        }
        launch->reset();
        auto handle = kg::CublasHandle::Create(
            0, d.execution(), d.stream(),
            {.base = Address(cublas_workspace), .size = Bytes(cublas_bytes)});
        if (!handle) {
          return Error(handle.error().detail);
        }
        cublas = std::move(*handle);
        auto relaunch = kg::LaunchContext::Create(
            0, d.execution(), d.stream(), {.base = Address(scratch), .size = Bytes(most_scratch)},
            cublas.get());
        if (!relaunch) {
          return Error(relaunch.error().detail);
        }
        launch = std::move(relaunch);
        census.Declare("device:cublas-workspace", cublas_bytes);
        census.Read("cublas", e, chunk);
        // The plan's device choices follow the new context.
        planned = PlanChunk(profile, *artifact, *binding, weights, Address(kv), t.cells, rows,
                            inputs->n_kv, o.fusion, kg::DeviceChoicesOf(**launch),
                            Address(activations), most_activations);
        if (!planned) {
          return Error(std::format("chunk {}: {}", k, planned.error()));
        }
      }
      auto bound = kg::BoundGraph::Bind(*registry, planned->plan);
      if (!bound) {
        return Error(std::format("chunk {}: {}", k, bound.error().detail));
      }
      auto step_scratch = kg::PlanScratch(**launch, planned->plan);
      if (!step_scratch) {
        return Error(step_scratch.error().detail);
      }

      if (recording != nullptr) {
        for (const auto& event : recording->Take()) {
          record += llmp::test_support::EventLine(event);
        }
        record += llmp::test_support::ChunkLine(
            {.evaluation = e, .chunk = chunk, .rows = rows, .n_past = n_past});
      }
      // The inputs, staged and copied in the bridge's order.
      auto stream = d.Stream();
      if (!stream) {
        return std::unexpected(stream.error());
      }
      const kg::Qwen2Graph& g = planned->graph;
      const std::array<std::pair<ggml_tensor*, const void*>, 6> sources = {
          {{g.embd, embd.data()},
           {g.positions, inputs->positions.data()},
           {g.k_idxs, inputs->k_idxs.data()},
           {g.v_idxs, inputs->v_idxs.data()},
           {g.mask, inputs->mask.data()},
           {g.out_ids, inputs->out_ids.data()}}};
      std::uint64_t staged = 0;
      std::uint64_t input_bytes = 0;
      for (const auto& [tensor, source] : sources) {
        input_bytes += ggml_nbytes(tensor);
        auto* at = static_cast<std::byte*>(input_staging) + staged;
        std::memcpy(at, source, ggml_nbytes(tensor));
        if (auto r = Cuda(cudaMemcpyAsync(tensor->data, at, ggml_nbytes(tensor),
                                          cudaMemcpyHostToDevice, *stream),
                          "an input copy");
            !r) {
          return r;
        }
        staged += Round(ggml_nbytes(tensor), 128);
      }
      if (auto r = bound->Run(**launch); !r) {
        return Error(std::format("chunk {}: {}", k, r.error().detail));
      }
      const std::uint64_t chunk_logits = std::uint64_t{rows} * profile.vocab * sizeof(float);
      if (auto r = Cuda(cudaMemcpyAsync(logits, g.logits->data, chunk_logits,
                                        cudaMemcpyDeviceToHost, *stream),
                        "the logits copy");
          !r) {
        return r;
      }
      if (auto r = d.Finish(); !r) {
        return r;
      }
      if ((*launch)->faulted()) {
        return Error(std::format("chunk {}: the launch context faulted", k));
      }
      if (recording != nullptr) {
        for (const auto& event : recording->Take()) {
          record += llmp::test_support::EventLine(event);
        }
        record += llmp::test_support::EndChunkLine();
      }
      std::memcpy(result.data() + (std::size_t{n_past} * profile.vocab), logits, chunk_logits);
      census.Read("chunk", e, chunk);
      census.phases().push_back(std::format(
          R"({{"evaluation":{},"chunk":{},"rows":{},"n_past":{},"n_kv":{},"A":{},"S":{},"I":{},"L":{},"steps":{}}})",
          e, chunk, rows, n_past, inputs->n_kv, planned->placement.extent, *step_scratch,
          input_bytes, chunk_logits, planned->plan.steps.size()));
      n_past += rows;
    }
  }
  if (census.on()) {
    // Readings with no work between them, as the bridge's harness takes.
    for (int i = 0; i < kTailReadings; ++i) {
      census.Read("tail", 0, i);
    }
    if (auto r = Controls(census, d, "after the evaluations", true); !r) {
      return r;
    }
  }

  // Outputs: the first evaluation's logits, and every later one equal to it.
  const std::vector<float>& first = results.front();
  std::size_t differing = 0;
  for (std::size_t e = 1; e < results.size(); ++e) {
    for (std::size_t i = 0; i < first.size(); ++i) {
      differing +=
          std::bit_cast<std::uint32_t>(first[i]) != std::bit_cast<std::uint32_t>(results[e][i]);
    }
  }
  llmp::base::Sha256 hash;
  hash.Update(std::as_bytes(std::span(first)));
  const std::string digest = llmp::base::ToHex(hash.Finish());
  std::filesystem::create_directories(o.out);
  {
    std::ofstream raw(o.out / "logits.f32le", std::ios::binary);
    raw.write(reinterpret_cast<const char*>(first.data()),
              static_cast<std::streamsize>(first.size() * sizeof(float)));
    if (!raw) {
      return Error("the logits could not be written");
    }
  }
  std::string chunks;
  for (const std::uint32_t c : t.chunks) {
    chunks += std::format("{}{}", chunks.empty() ? "" : ",", c);
  }
  const std::string summary = std::format(
      R"({{"trajectory":"{}","fusion":{},"tokens":{},"chunks":[{}],"cells":{},"evaluations":{},)"
      R"("logits_sha256":"{}","repeat_bit_differences":{},"artifact":"{}","weights_region":{},)"
      R"("weights_resource_bytes":{},"chunks_verified":{},"bytes_read":{},"kv_bytes":{},)"
      R"("activations":{},"scratch":{},"cublas_workspace":{}}})",
      t.name, o.fusion ? "true" : "false", t.tokens.size(), chunks, t.cells, o.evaluations, digest,
      differing, artifact->id(), weights.bytes, weights.resource_bytes, weights.chunks_verified,
      weights.bytes_read, kv_bytes, most_activations, most_scratch, cublas_bytes);
  {
    std::ofstream file(o.out / "summary.json");
    file << summary << "\n";
  }
  // The logits' identity is in the summary, not on the console: the FP16
  // gate compares them only once the plan has matched.
  std::println("wrote {}", o.out.string());
  if (census.on()) {
    std::ofstream file(o.out / "census.json");
    file << std::format(
        R"j({{"format":"llmp-census/2","source":"llmp_fp16_exec (native, rung 3)","trajectory":"{}","fusion":{},"settle_ms":{},"evaluations":{},"repeat_bit_differences":{},"logits_sha256":"{}","chunks":[{}],"readings":[)j",
        t.name, o.fusion ? "true" : "false", census.settle_ms(), o.evaluations, differing, digest,
        chunks);
    for (std::size_t i = 0; i < census.readings().size(); ++i) {
      file << (i == 0 ? "\n" : ",\n") << census.readings()[i];
    }
    file << "],\"controls\":[";
    for (std::size_t i = 0; i < census.controls().size(); ++i) {
      file << (i == 0 ? "\n" : ",\n") << census.controls()[i];
    }
    file << "],\"phases\":[";
    for (std::size_t i = 0; i < census.phases().size(); ++i) {
      file << (i == 0 ? "\n" : ",\n") << census.phases()[i];
    }
    file << "]}\n";
    if (!file) {
      return Error("the census could not be written");
    }
  }
  if (differing != 0) {
    return Error(std::format("a repeated evaluation differs in {} logits", differing));
  }
  launch->reset();
  cublas.reset();
  (void)cudaFree(cublas_workspace);
  (void)cudaFree(activations);
  (void)cudaFree(scratch);
  (void)cudaFree(kv);
  (void)cudaFree(Pointer(weights.base));
  (void)cudaFreeHost(input_staging);
  (void)cudaFreeHost(logits);
  (void)cudaFreeHost(weights.staging);
  return {};
}

}  // namespace

int main(int argc, char** argv) {
  const std::span<char*> args(argv, static_cast<std::size_t>(argc));
  // The recording starts before anything touches CUDA (--record).
  const bool record =
      std::ranges::any_of(args, [](const char* a) { return std::string_view(a) == "--record"; });
  std::unique_ptr<llmp::test_support::Recording> recording;
  if (record) {
    recording = std::make_unique<llmp::test_support::Recording>();
  }
  const auto options = Parse(args);
  if (!options) {
    std::println(stderr, "{}", options.error());
    return 2;
  }
  std::string lines;
  const Status ran = Run(*options, recording.get(), lines);
  if (recording) {
    for (const auto& event : recording->Take()) {
      lines += llmp::test_support::EventLine(event);
    }
    std::filesystem::create_directories(options->out);
    std::ofstream file(options->out / "recording.jsonl");
    file << llmp::test_support::HeaderLine(
                std::format("llmp_fp16_exec {} fusion {}", options->trajectory,
                            options->fusion ? "on" : "off"),
                llmp::test_support::LoadedCublas())
         << lines;
  }
  if (!ran) {
    std::println(stderr, "{}", ran.error());
    return 1;
  }
  return 0;
}
