// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Backend-proof P3, the per-linear sweep (BP-N5; docs/backend-proof.md,
// Tier E "EXL3 packed linears" and "EXL3 reconstruction-path linears";
// docs/experiments/backend-proof-p3/README.md): every real projection of
// an EXL3 fixture, from its v0 prepared artifact on cudaMalloc memory,
// through llmpalooza's launchers at the forced plans upstream's frozen tuning
// cache gives, on the inputs upstream's run used.
//
//   llmp_exl3_linear_sweep --artifact DIR --fixture 4.0bpw|4.5bpw
//                            --plan PLAN.txt --out OUT.jsonl
//                            [--placement malloc|minimal|flush-end|flush-start]
//
// PLAN.txt is native_plan.py's: each linear's shape, rate, output type and
// bias, and each case's path and plan. Weights are read from the artifact
// (opened as untrusted input, artifact.h) with pread and copied to the
// device; their SHA-256s are written, for comparison with the reference's.
// For each packed case it also records where upstream's EXL3-O would
// launch the GEMV, by llmpalooza's copy of upstream's choice (upstream_gemv.h).
// Each case runs twice: once step by step through the launch context
// (launch.h), hashing every buffer a step writes, as the reference hashes
// upstream's (linear_reference.py), and once through the implementation the
// registry binds for its path (implementations.h), whose output must equal
// the steps'. Per linear it also hashes the full reconstructed weights,
// rotated and fused. --placement puts every operand, weights and
// activations alike, at cudaMalloc's 256-byte alignment (malloc, the
// default); at the smallest alignment the host checks accept (validate.h;
// minimal, the alignment probe); or in device VMM flush against an
// unmapped granule, ending at it or starting after it (flush-end,
// flush-start: the over-read probe, where any access past an operand
// faults). Output: JSON lines of SHA-256s; raw tensors are never written.

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <chrono>
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
#include <tuple>
#include <utility>
#include <vector>

#include "artifact/artifact.h"
#include "base/bytes.h"
#include "base/sha256.h"
#include "execution/registry.h"
#include "kernels/exl3/implementations.h"
#include "kernels/exl3/launch.h"
#include "kernels/exl3/linear.h"
#include "kernels/exl3/recon_gemm.h"
#include "kernels/exl3/validate.h"
#include "providers/cuda/cuda_device_execution.h"
#include "providers/cuda/cuda_device_memory.h"
#include "providers/device_execution.h"
#include "providers/device_memory.h"

namespace {

namespace exl3 = llmp::kernels::exl3;
using exl3::Output;

std::unexpected<std::string> Error(std::string text) { return std::unexpected(std::move(text)); }

std::expected<void, std::string> Cuda(cudaError_t error, std::string_view what) {
  if (error != cudaSuccess) {
    return Error(std::format("{}: {}", what, cudaGetErrorString(error)));
  }
  return {};
}

// NOLINTBEGIN(performance-no-int-to-ptr): device addresses.
void* Pointer(std::uint64_t address) { return reinterpret_cast<void*>(address); }
// NOLINTEND(performance-no-int-to-ptr)

std::string Hex(std::span<const std::byte> bytes) {
  llmp::base::Sha256 hash;
  hash.Update(bytes);
  return llmp::base::ToHex(hash.Finish());
}

// Where operands go (--placement): cudaMalloc at its 256-byte alignment; at
// the smallest alignment the host checks accept (validate.h), the
// alignment probe; or in device VMM ending flush against an unmapped
// granule, or starting flush after one, the over-read probe.
enum class Mode : std::uint8_t { kMalloc, kMinimal, kFlushEnd, kFlushStart };

// A buffer: its usable range, from which an operand of a case is placed.
struct Region {
  std::uint64_t start = 0;
  std::uint64_t end = 0;
};

// Device memory the harness owns.
class Arena {
 public:
  explicit Arena(Mode mode) : mode_(mode) {}
  Arena(const Arena&) = delete;
  Arena& operator=(const Arena&) = delete;
  Arena(Arena&&) = delete;
  Arena& operator=(Arena&&) = delete;
  ~Arena() {
    for (void* region : regions_) {
      (void)cudaFree(region);
    }
    // A failure leaves nothing to do at exit: the process is ending.
    for (const Mapped& m : mapped_) {
      if (memory_->Unmap(m.reservation, llmp::base::Bytes(m.offset), llmp::base::Bytes(m.size)) &&
          memory_->Release(m.backing)) {
        std::ignore = memory_->Free(m.reservation);
      }
    }
  }

  // A buffer of at least `bytes` for operands aligned to `alignment` (a
  // power of two up to 256).
  std::expected<Region, std::string> Allocate(std::uint64_t bytes, std::uint64_t alignment) {
    if (mode_ == Mode::kMalloc || mode_ == Mode::kMinimal) {
      void* region = nullptr;
      if (auto r = Cuda(cudaMalloc(&region, bytes + 512), "cudaMalloc"); !r) {
        return std::unexpected(r.error());
      }
      regions_.push_back(region);
      const auto base = reinterpret_cast<std::uint64_t>(region);  // 256-byte aligned
      const std::uint64_t start =
          mode_ == Mode::kMinimal && alignment < 256 ? base + alignment : base;
      return Region{.start = start, .end = start + bytes};
    }
    if (!memory_) {
      auto memory = llmp::providers::cuda::OpenDeviceMemory(0);
      if (!memory) {
        return Error("device VMM did not open: " + memory.error().detail);
      }
      memory_ = std::move(*memory);
      for (std::size_t i = 0; i < memory_->Classes().size(); ++i) {
        if (memory_->Classes()[i].kind == llmp::providers::BackingKind::kDevice) {
          device_class_ = i;
        }
      }
    }
    const std::uint64_t granule = memory_->Granularity().value();
    const std::uint64_t size =
        (std::max<std::uint64_t>(bytes, 1) + granule - 1) / granule * granule;
    auto reservation = memory_->Reserve(llmp::base::Bytes(size + granule));
    if (!reservation) {
      return Error("a VMM reservation failed: " + reservation.error().detail);
    }
    auto backing = memory_->Create(device_class_, llmp::base::Bytes(size));
    if (!backing) {
      return Error("a VMM backing failed: " + backing.error().detail);
    }
    // The unmapped granule follows the mapping (flush at the end) or
    // precedes it (flush at the start).
    const std::uint64_t offset = mode_ == Mode::kFlushStart ? granule : 0;
    if (!memory_->Map(*reservation, llmp::base::Bytes(offset), *backing) ||
        !memory_->SetAccess(*reservation, llmp::base::Bytes(offset), llmp::base::Bytes(size),
                            llmp::providers::Access::kReadWrite)) {
      return Error("a VMM mapping failed");
    }
    mapped_.push_back({*reservation, *backing, offset, size});
    const std::uint64_t base = memory_->RangeOf(*reservation).value().base + offset;
    return Region{.start = base, .end = base + size};
  }

  // An operand of `bytes` in the region: flush with its end in kFlushEnd,
  // at its start otherwise.
  std::uint64_t At(const Region& region, std::uint64_t bytes) const {
    return mode_ == Mode::kFlushEnd ? region.end - bytes : region.start;
  }

 private:
  struct Mapped {
    llmp::providers::ReservationId reservation;
    llmp::providers::BackingId backing;
    std::uint64_t offset;
    std::uint64_t size;
  };
  Mode mode_;
  std::vector<void*> regions_;
  std::unique_ptr<llmp::providers::VmmProvider> memory_;
  std::size_t device_class_ = 0;
  std::vector<Mapped> mapped_;
};

// A pageable cudaMemcpy may return before its DMA lands (it runs on the
// legacy stream, not the provider's), so the upload waits for the device.
std::expected<void, std::string> Upload(std::uint64_t address, const void* data, std::size_t size) {
  if (auto r = Cuda(cudaMemcpy(Pointer(address), data, size, cudaMemcpyHostToDevice), "an upload");
      !r) {
    return r;
  }
  return Cuda(cudaDeviceSynchronize(), "an upload");
}

std::expected<std::string, std::string> HashDevice(std::uint64_t address, std::size_t size) {
  std::vector<std::byte> host(size);
  if (auto r = Cuda(cudaMemcpy(host.data(), Pointer(address), size, cudaMemcpyDeviceToHost),
                    "a download");
      !r) {
    return std::unexpected(r.error());
  }
  return Hex(host);
}

// cases.py's inputs: splitmix64 over seed + i, the top 11 bits less 1,024,
// over 1,024, which F16 holds exactly.
std::uint64_t SplitMix64(std::uint64_t x) {
  std::uint64_t z = x + 0x9E3779B97F4A7C15ULL;
  z = (z ^ (z >> 30U)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27U)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31U);
}

std::vector<__half> Inputs(std::string_view fixture, std::string_view key, int rows, int k) {
  llmp::base::Sha256 hash;
  hash.Update(std::format("p3a|{}|{}|{}", fixture, key, rows));
  const auto digest = hash.Finish();
  std::uint64_t seed = 0;
  for (std::size_t i = 0; i < 8; ++i) {
    seed |= static_cast<std::uint64_t>(digest[i]) << (8 * i);
  }
  std::vector<__half> values(static_cast<std::size_t>(rows) * static_cast<std::size_t>(k));
  for (std::size_t i = 0; i < values.size(); ++i) {
    const auto bits = static_cast<std::int64_t>(SplitMix64(seed + i) >> 53U);
    values[i] = __float2half_rn(static_cast<float>(bits - 1024) / 1024.0F);
  }
  return values;
}

struct Linear {
  std::string name;
  exl3::Weights weights;
  Output output = Output::kF16;
  std::uint64_t bias = 0;
};

struct Case {
  std::string id;
  std::string key;
  int rows = 0;
  std::string path;  // gemm, gemv, multi, recon, fused
  std::vector<std::string> linears;
  exl3::GemmPlan gemm;
  exl3::GemvPlan gemv;
  exl3::MultiGemmPlan multi;
  std::vector<exl3::LtAlgorithm> algorithms;
};

struct Plan {
  std::map<std::string, std::vector<std::string>> linears;  // name -> fields
  std::vector<Case> cases;
};

std::expected<Plan, std::string> ReadPlan(const std::filesystem::path& path) {
  std::ifstream in(path);
  if (!in) {
    return Error(std::format("cannot read {}", path.string()));
  }
  Plan plan;
  std::string line;
  while (std::getline(in, line)) {
    std::istringstream words(line);
    std::string kind;
    words >> kind;
    if (kind == "linear") {
      std::string name;
      words >> name;
      std::vector<std::string> fields;
      for (std::string field; words >> field;) {
        fields.push_back(field);
      }
      plan.linears[name] = fields;
    } else if (kind == "case") {
      Case c;
      words >> c.id >> c.rows >> c.path;
      c.key = c.id.substr(0, c.id.rfind('@'));
      if (c.path == "gemm") {
        c.linears.resize(1);
        words >> c.linears[0] >> c.gemm.shape >> c.gemm.blocks;
      } else if (c.path == "gemv") {
        c.linears.resize(1);
        words >> c.linears[0] >> c.gemv.config >> c.gemv.blocks;
      } else if (c.path == "multi") {
        c.linears.resize(2);
        words >> c.linears[0] >> c.linears[1] >> c.multi.shape >> c.multi.blocks >>
            c.multi.concurrency;
      } else if (c.path == "recon" || c.path == "fused") {
        c.linears.resize(1);
        int slices = 0;
        words >> c.linears[0] >> slices;
        for (int s = 0; s < slices; ++s) {
          // The GEMM the pin was recorded for, then its nine attributes.
          exl3::LtAlgorithm algorithm;
          std::string gemm;
          words >> gemm >> algorithm.m >> algorithm.k >> algorithm.n >> algorithm.ldc;
          if (gemm != "HSH" && gemm != "HSS") {
            return Error(std::format("a pin of kind {} in: {}", gemm, line));
          }
          algorithm.output = gemm == "HSS" ? exl3::Output::kF32 : exl3::Output::kF16;
          for (std::uint64_t& value : algorithm.config) {
            words >> value;
          }
          c.algorithms.push_back(algorithm);
        }
      } else {
        return Error(std::format("unknown path in: {}", line));
      }
      if (!words && !words.eof()) {
        return Error(std::format("malformed: {}", line));
      }
      plan.cases.push_back(std::move(c));
    } else if (!kind.empty()) {
      return Error(std::format("unknown line: {}", line));
    }
  }
  return plan;
}

// Reads one resource's bytes from its shard.
std::expected<std::vector<std::byte>, std::string> ReadResource(
    const llmp::artifact::Artifact& artifact, const std::filesystem::path& root,
    std::string_view role) {
  const auto index = artifact.FindResource(role);
  if (!index) {
    return Error(std::format("the artifact has no {}", role));
  }
  const auto& resource = artifact.resources()[*index];
  const auto& group = artifact.groups()[resource.group];
  const auto& shard = artifact.shards()[group.shard];
  const std::uint64_t offset =
      shard.data_offset.value() + group.offset.value() + resource.offset.value();
  std::vector<std::byte> bytes(resource.bytes.value());
  const int fd = open((root / shard.path).c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0) {
    return Error(std::format("cannot open {}", shard.path));
  }
  std::uint64_t done = 0;
  while (done < bytes.size()) {
    const ssize_t got =
        pread(fd, bytes.data() + done, bytes.size() - done, static_cast<off_t>(offset + done));
    if (got <= 0) {
      close(fd);
      return Error(std::format("reading {} failed", role));
    }
    done += static_cast<std::uint64_t>(got);
  }
  close(fd);
  return bytes;
}

class Sweep {
 public:
  Sweep(std::string fixture, Mode mode) : fixture_(std::move(fixture)), arena_(mode) {}

  std::expected<void, std::string> Setup() {
    if (auto r = Cuda(cudaSetDevice(0), "cudaSetDevice"); !r) {
      return r;
    }
    auto execution = llmp::providers::cuda::OpenDeviceExecution(0);
    if (!execution) {
      return Error("the CUDA provider did not open: " + execution.error().detail);
    }
    execution_ = std::move(*execution);
    auto stream = execution_->CreateStream();
    if (!stream) {
      return Error("no stream: " + stream.error().detail);
    }
    stream_ = *stream;
    auto locks = arena_.Allocate(exl3::kLockBytes, 256);
    if (!locks) {
      return std::unexpected(locks.error());
    }
    // The lock area is workspace, not an operand: at its region's start,
    // 256-byte aligned, in every mode.
    auto launch = exl3::LaunchContext::Create(0, *execution_, stream_, locks->start);
    if (!launch) {
      return Error("the launch context was refused: " + launch.error().detail);
    }
    launch_ = std::move(*launch);
    auto gemm = exl3::ReconGemm::Create();
    if (!gemm) {
      return Error("the reconstruction GEMM was refused: " + gemm.error().detail);
    }
    gemm_ = std::move(*gemm);
    auto registry = llmp::execution::Registry::Create(exl3::Implementations());
    if (!registry) {
      return Error("the registry was refused: " + registry.error().detail);
    }
    for (const std::string_view name :
         {"exl3.linear.gemm", "exl3.linear.gemv", "exl3.linear.reconstruct",
          "exl3.linear.reconstruct_fused", "exl3.multi_linear.mgemm"}) {
      const auto index = registry->Find(name);
      if (!index) {
        return Error(std::format("the registry has no {}", name));
      }
      auto kernel = exl3::Kernel::Bind(registry->at(*index));
      if (!kernel) {
        return Error("binding failed: " + kernel.error().detail);
      }
      kernels_.emplace(std::string(name), *kernel);
    }
    return {};
  }

  // Everything queued so far has run.
  std::expected<void, std::string> Finish() {
    auto fence = execution_->Record(stream_);
    if (!fence) {
      return Error("no fence: " + fence.error().detail);
    }
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
    for (;;) {
      auto state = execution_->Query(*fence);
      if (!state) {
        return Error("the fence query failed: " + state.error().detail);
      }
      if (*state == llmp::providers::FenceState::kComplete) {
        break;
      }
      if (std::chrono::steady_clock::now() > deadline) {
        return Error("the stream did not finish in 60 s");
      }
      std::this_thread::sleep_for(std::chrono::microseconds(20));
    }
    if (!execution_->Release(*fence)) {
      return Error("the fence release failed");
    }
    return {};
  }

  std::expected<void, std::string> LoadWeights(const std::filesystem::path& root, const Plan& plan,
                                               std::FILE* out) {
    auto artifact = llmp::artifact::Artifact::Open(root);
    if (!artifact) {
      return Error("the artifact did not open");
    }
    for (const auto& [name, fields] : plan.linears) {
      if (fields.size() != 5) {
        return Error("a malformed linear line for " + name);
      }
      Linear linear;
      linear.name = name;
      linear.weights.k = std::stoi(fields[0]);
      linear.weights.n = std::stoi(fields[1]);
      linear.weights.bits = std::stoi(fields[2]);
      linear.output = fields[3] == "F32" ? Output::kF32 : Output::kF16;
      const bool bias = fields[4] == "1";
      std::string record = std::format(R"({{"linear": "{}")", name);
      struct Part {
        std::string_view suffix;
        std::uint64_t* address;
        std::uint64_t alignment;
      };
      std::vector<Part> parts{{"trellis", &linear.weights.trellis, 16},
                              {"suh", &linear.weights.suh, 8},
                              {"svh", &linear.weights.svh, 8}};
      if (bias) {
        parts.push_back({"bias", &linear.bias, 2});
      }
      for (const Part& part : parts) {
        auto bytes = ReadResource(*artifact, root, std::format("{}.{}", name, part.suffix));
        if (!bytes) {
          return std::unexpected(bytes.error());
        }
        auto region = arena_.Allocate(bytes->size(), part.alignment);
        if (!region) {
          return std::unexpected(region.error());
        }
        const std::uint64_t address = arena_.At(*region, bytes->size());
        if (auto r = Upload(address, bytes->data(), bytes->size()); !r) {
          return r;
        }
        *part.address = address;
        record += std::format(R"(, "{}": "{}")", part.suffix, Hex(*bytes));
      }
      if (auto checked = exl3::CheckWeights(linear.weights); !checked) {
        return Error(name + ": " + checked.error().detail);
      }
      // The full reconstructed weights, rotated and fused, as one launch
      // each (upstream's reconstruct and reconstruct_had_slice at offset 0).
      const std::uint64_t full_bytes = static_cast<std::uint64_t>(linear.weights.k) *
                                       static_cast<std::uint64_t>(linear.weights.n) * 2;
      auto region = arena_.Allocate(full_bytes, 16);
      if (!region) {
        return std::unexpected(region.error());
      }
      const std::uint64_t full = arena_.At(*region, full_bytes);
      for (const bool fused : {false, true}) {
        if (auto r = launch_->Reconstruct({.weights = linear.weights,
                                           .w = full,
                                           .column = 0,
                                           .columns = linear.weights.n,
                                           .fused = fused});
            !r) {
          return Error(name + ": " + r.error().detail);
        }
        if (auto r = Finish(); !r) {
          return r;
        }
        auto hash = HashDevice(full, full_bytes);
        if (!hash) {
          return std::unexpected(hash.error());
        }
        record += std::format(R"(, "{}": "{}")", fused ? "W_fused" : "W", *hash);
      }
      std::println(out, "{}}}", record);
      linears_.emplace(name, linear);
    }
    return {};
  }

  // Buffers big enough for every case.
  std::expected<void, std::string> AllocateScratch(const Plan& plan) {
    std::uint64_t x = 0;
    std::uint64_t a_had = 0;
    std::uint64_t y = 0;
    std::uint64_t w = 0;
    for (const Case& c : plan.cases) {
      const Linear& l = linears_.at(c.linears[0]);
      const auto rows = static_cast<std::uint64_t>(c.rows);
      const auto k = static_cast<std::uint64_t>(l.weights.k);
      const auto n = static_cast<std::uint64_t>(l.weights.n);
      const auto matrices = static_cast<std::uint64_t>(c.linears.size());
      x = std::max(x, rows * k * 2);
      a_had = std::max(a_had, matrices * rows * k * 2);
      y = std::max(y,
                   matrices * rows * n * static_cast<std::uint64_t>(exl3::OutputBytes(l.output)));
      w = std::max(w, exl3::ReconstructScratchBytes(l.weights));
    }
    // The packed paths read x as 8-byte vectors and write an F16 y as 8-byte
    // ones (x8_, y8_); the reconstruction GEMM needs 16 for its operands,
    // and an F32 y needs 16 everywhere (x_, y_).
    for (const auto& [bytes, alignment, address] :
         {std::tuple{x, std::uint64_t{16}, &x_}, std::tuple{x, std::uint64_t{8}, &x8_},
          std::tuple{a_had, std::uint64_t{16}, &a_had_}, std::tuple{y, std::uint64_t{16}, &y_},
          std::tuple{y, std::uint64_t{8}, &y8_}, std::tuple{y, std::uint64_t{16}, &y2_},
          std::tuple{a_had, std::uint64_t{16}, &xh_}, std::tuple{w, std::uint64_t{16}, &w_}}) {
      auto allocated = arena_.Allocate(bytes, alignment);
      if (!allocated) {
        return std::unexpected(allocated.error());
      }
      *address = *allocated;
    }
    // Multi-GEMM tables: two addresses each.
    auto tables = arena_.Allocate(48, 16);
    if (!tables) {
      return std::unexpected(tables.error());
    }
    tables_ = *tables;
    return {};
  }

  std::expected<void, std::string> Run(const Case& c, std::FILE* out) {
    const Linear& first = linears_.at(c.linears[0]);
    const int k = first.weights.k;
    const int n = first.weights.n;
    const auto rows = static_cast<std::uint64_t>(c.rows);
    const std::vector<__half> x = Inputs(fixture_, c.key, c.rows, k);
    const bool packed = c.path == "gemm" || c.path == "gemv" || c.path == "multi";
    const std::uint64_t matrices = c.linears.size();
    const std::uint64_t x_bytes = x.size() * sizeof(__half);
    const std::uint64_t y_bytes = rows * static_cast<std::uint64_t>(n) *
                                  static_cast<std::uint64_t>(exl3::OutputBytes(first.output));
    const std::uint64_t out_bytes = matrices * y_bytes;
    // This case's operands in the scratch regions (Arena::At).
    const std::uint64_t x_at = arena_.At(packed ? x8_ : x_, x_bytes);
    const std::uint64_t y_at =
        arena_.At(packed && first.output == Output::kF16 ? y8_ : y_, out_bytes);
    const std::uint64_t a_had = arena_.At(a_had_, matrices * x_bytes);
    const std::uint64_t xh = arena_.At(xh_, x_bytes);
    const std::uint64_t y2 = arena_.At(y2_, out_bytes);
    const std::uint64_t tables = arena_.At(tables_, 48);
    if (auto r = Upload(x_at, x.data(), x.size() * sizeof(__half)); !r) {
      return r;
    }
    std::vector<std::pair<std::string, std::string>> stages;
    auto stage = [&](std::string name, std::uint64_t address,
                     std::uint64_t bytes) -> std::expected<void, std::string> {
      if (auto r = Finish(); !r) {
        return r;
      }
      auto hash = HashDevice(address, bytes);
      if (!hash) {
        return std::unexpected(hash.error());
      }
      stages.emplace_back(std::move(name), *hash);
      return {};
    };
    auto launched =
        [&](std::expected<void, exl3::KernelFailure> r) -> std::expected<void, std::string> {
      if (!r) {
        return Error(std::format("{}: {}", c.id, r.error().detail));
      }
      return {};
    };
    bool composed_equal = false;
    std::string composed_hash;
    if (c.path == "gemm" || c.path == "gemv") {
      const exl3::LinearOperands o{.weights = first.weights,
                                   .x = x_at,
                                   .a_had = a_had,
                                   .y = y_at,
                                   .output = first.output,
                                   .m = c.rows};
      if (auto r = launched(c.path == "gemm" ? launch_->Gemm(o, c.gemm) : launch_->Gemv(o, c.gemv));
          !r) {
        return r;
      }
      if (auto r = stage("a_had", a_had, x_bytes); !r) {
        return r;
      }
      if (auto r = stage("y", y_at, y_bytes); !r) {
        return r;
      }
      if (first.bias != 0) {
        if (auto r = launched(launch_->Bias(
                {.x = y_at, .bias = first.bias, .y = y_at, .rows = c.rows, .columns = n}));
            !r) {
          return r;
        }
      }
      if (auto r = stage("out", y_at, y_bytes); !r) {
        return r;
      }
      exl3::LinearOperands composed = o;
      composed.y = y2;
      const exl3::Kernel& kernel =
          kernels_.at(c.path == "gemm" ? "exl3.linear.gemm" : "exl3.linear.gemv");
      if (auto r = launched(c.path == "gemm" ? kernel.Run(*launch_, composed, c.gemm, first.bias)
                                             : kernel.Run(*launch_, composed, c.gemv, first.bias));
          !r) {
        return r;
      }
    } else if (c.path == "multi") {
      const Linear& second = linears_.at(c.linears[1]);
      const std::array<std::uint64_t, 6> table =
          exl3::MultiGemmTables(first.weights, second.weights);
      if (auto r = Upload(tables, table.data(), sizeof table); !r) {
        return r;
      }
      const exl3::MultiLinearOperands o{.first = first.weights,
                                        .second = second.weights,
                                        .trellis_table = tables,
                                        .suh_table = tables + 16,
                                        .svh_table = tables + 32,
                                        .written = table,
                                        .x = x_at,
                                        .a_had = a_had,
                                        .y = y_at,
                                        .output = first.output,
                                        .m = c.rows};
      if (auto r = launched(launch_->MultiGemm(o, c.multi)); !r) {
        return r;
      }
      if (auto r = stage("a_had", a_had, 2 * x_bytes); !r) {
        return r;
      }
      if (auto r = stage("y", y_at, out_bytes); !r) {
        return r;
      }
      if (auto r = stage("out", y_at, out_bytes); !r) {
        return r;
      }
      exl3::MultiLinearOperands composed = o;
      composed.y = y2;
      if (auto r =
              launched(kernels_.at("exl3.multi_linear.mgemm").Run(*launch_, composed, c.multi));
          !r) {
        return r;
      }
    } else {
      const bool fused = c.path == "fused";
      if (!fused) {
        if (auto r = launched(launch_->Hadamard({.x = x_at,
                                                 .y = xh,
                                                 .scale = first.weights.suh,
                                                 .rows = c.rows,
                                                 .columns = k,
                                                 .type = Output::kF16,
                                                 .input_scale = true}));
            !r) {
          return r;
        }
        if (auto r = stage("xh", xh, x_bytes); !r) {
          return r;
        }
      }
      const std::vector<int> slices = exl3::ReconstructSlices(n);
      if (slices.size() != c.algorithms.size()) {
        return Error(c.id + ": the plan's algorithms do not match the slices");
      }
      int column = 0;
      for (std::size_t s = 0; s < slices.size(); ++s) {
        const int columns = slices[s];
        const std::uint64_t w_bytes =
            static_cast<std::uint64_t>(k) * static_cast<std::uint64_t>(columns) * 2;
        const std::uint64_t w = arena_.At(w_, w_bytes);
        if (auto r = launched(launch_->Reconstruct({.weights = first.weights,
                                                    .w = w,
                                                    .column = column,
                                                    .columns = columns,
                                                    .fused = fused}));
            !r) {
          return r;
        }
        if (auto r = stage(std::format("w{}", s), w, w_bytes); !r) {
          return r;
        }
        const exl3::ReconGemmOperands product{
            .w = w,
            .x = fused ? x_at : xh,
            .y = y_at + (static_cast<std::uint64_t>(column) *
                         static_cast<std::uint64_t>(exl3::OutputBytes(first.output))),
            .m = c.rows,
            .k = k,
            .n = columns,
            .ldc = n,
            .output = first.output};
        if (auto r = launched(gemm_->Run(*launch_, product, c.algorithms[s], launch_->sm_count()));
            !r) {
          return r;
        }
        column += columns;
      }
      if (!fused) {
        if (auto r = stage("gemm", y_at, y_bytes); !r) {
          return r;
        }
        if (auto r = launched(launch_->Hadamard({.x = y_at,
                                                 .y = y_at,
                                                 .scale = first.weights.svh,
                                                 .rows = c.rows,
                                                 .columns = n,
                                                 .type = first.output,
                                                 .input_scale = false}));
            !r) {
          return r;
        }
      }
      if (auto r = stage("y", y_at, y_bytes); !r) {
        return r;
      }
      if (first.bias != 0) {
        if (auto r = launched(launch_->Bias(
                {.x = y_at, .bias = first.bias, .y = y_at, .rows = c.rows, .columns = n}));
            !r) {
          return r;
        }
      }
      if (auto r = stage("out", y_at, y_bytes); !r) {
        return r;
      }
      const exl3::ReconstructedOperands composed{
          .weights = first.weights,
          .x = x_at,
          .xh = xh,
          .w = arena_.At(w_, exl3::ReconstructScratchBytes(first.weights)),
          .y = y2,
          .output = first.output,
          .m = c.rows,
          .bias = first.bias};
      const exl3::Kernel& kernel =
          kernels_.at(fused ? "exl3.linear.reconstruct_fused" : "exl3.linear.reconstruct");
      if (auto r = launched(kernel.Run(*launch_, *gemm_, composed, c.algorithms)); !r) {
        return r;
      }
    }
    if (auto r = Finish(); !r) {
      return r;
    }
    auto composed = HashDevice(y2, out_bytes);
    if (!composed) {
      return std::unexpected(composed.error());
    }
    composed_hash = *composed;
    composed_equal = composed_hash == stages.back().second;
    // Where upstream's EXL3-O profile would take the GEMV (upstream_gemv.h),
    // as this device's occupancy gives it, for comparison with the plan.
    std::string gemv_choice = "null";
    if (c.path == "gemm" || c.path == "gemv") {
      auto choice = launch_->UpstreamGemv(first.weights, first.output, c.rows);
      if (!choice) {
        return Error(c.id + ": " + choice.error().detail);
      }
      if (*choice) {
        gemv_choice = std::format("[{}, {}]", (*choice)->config, (*choice)->blocks);
      }
    }
    std::string record =
        std::format(R"({{"case": "{}", "path": "{}", "upstream_gemv": {}, "x": "{}", "stages": {{)",
                    c.id, c.path, gemv_choice, Hex(std::as_bytes(std::span(x))));
    for (std::size_t i = 0; i < stages.size(); ++i) {
      record +=
          std::format(R"({}"{}": "{}")", i == 0 ? "" : ", ", stages[i].first, stages[i].second);
    }
    std::println(out, R"({}}}, "composed_equal": {}}})", record, composed_equal ? "true" : "false");
    return {};
  }

  std::expected<void, std::string> Teardown() {
    if (auto r = Finish(); !r) {
      return r;
    }
    launch_.reset();
    gemm_.reset();
    if (!execution_->DestroyStream(stream_)) {
      return Error("the stream was not destroyed");
    }
    return {};
  }

 private:
  std::string fixture_;
  Arena arena_;
  std::unique_ptr<llmp::providers::DeviceExecution> execution_;
  llmp::providers::StreamId stream_;
  std::unique_ptr<exl3::LaunchContext> launch_;
  std::unique_ptr<exl3::ReconGemm> gemm_;
  std::map<std::string, exl3::Kernel> kernels_;
  std::map<std::string, Linear> linears_;
  Region x_;
  Region x8_;
  Region a_had_;
  Region y_;
  Region y8_;
  Region y2_;
  Region xh_;
  Region w_;
  Region tables_;
};

std::expected<void, std::string> Main(int argc, char** argv) {
  std::filesystem::path artifact;
  std::filesystem::path plan_path;
  std::filesystem::path out_path;
  std::string fixture;
  Mode mode = Mode::kMalloc;
  for (int i = 1; i < argc; ++i) {
    const std::string_view arg(argv[i]);
    auto value = [&]() -> std::string {
      return i + 1 < argc ? std::string(argv[++i]) : std::string();
    };
    if (arg == "--artifact") {
      artifact = value();
    } else if (arg == "--plan") {
      plan_path = value();
    } else if (arg == "--out") {
      out_path = value();
    } else if (arg == "--fixture") {
      fixture = value();
    } else if (arg == "--placement") {
      const std::string placement = value();
      if (placement == "malloc") {
        mode = Mode::kMalloc;
      } else if (placement == "minimal") {
        mode = Mode::kMinimal;
      } else if (placement == "flush-end") {
        mode = Mode::kFlushEnd;
      } else if (placement == "flush-start") {
        mode = Mode::kFlushStart;
      } else {
        return Error("--placement is malloc, minimal, flush-end or flush-start");
      }
    } else {
      return Error(std::format("unknown argument {}", arg));
    }
  }
  if (artifact.empty() || plan_path.empty() || out_path.empty() || fixture.empty()) {
    return Error(
        "usage: --artifact DIR --fixture NAME --plan PLAN.txt --out OUT.jsonl "
        "[--placement malloc|minimal|flush-end|flush-start]");
  }
  auto plan = ReadPlan(plan_path);
  if (!plan) {
    return std::unexpected(plan.error());
  }
  std::FILE* out = std::fopen(out_path.c_str(), "w");
  if (out == nullptr) {
    return Error(std::format("cannot write {}", out_path.string()));
  }
  Sweep sweep(fixture, mode);
  auto result = [&]() -> std::expected<void, std::string> {
    if (auto r = sweep.Setup(); !r) {
      return r;
    }
    if (auto r = sweep.LoadWeights(artifact, *plan, out); !r) {
      return r;
    }
    if (auto r = sweep.AllocateScratch(*plan); !r) {
      return r;
    }
    const auto started = std::chrono::steady_clock::now();
    for (const Case& c : plan->cases) {
      if (auto r = sweep.Run(c, out); !r) {
        return r;
      }
    }
    std::println(stderr, "{} cases in {:.1f} s", plan->cases.size(),
                 std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count());
    return sweep.Teardown();
  }();
  if (std::fclose(out) != 0 && result) {
    return Error(std::format("writing {} failed", out_path.string()));
  }
  return result;
}

}  // namespace

int main(int argc, char** argv) {
  if (auto r = Main(argc, argv); !r) {
    std::println(stderr, "exl3_linear_sweep: {}", r.error());
    return 1;
  }
  return 0;
}
