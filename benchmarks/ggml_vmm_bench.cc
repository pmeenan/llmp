// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// BP-F1 (docs/backend-proof.md, "Performance protocol"): llmpalooza's GGML
// kernels at the FP16 fixture's held-out shapes, timed with every operand
// in one kind of memory: cudaMalloc, host VMM (D-034; rule v1) or device
// VMM (D-081; rule v2), the VMM from llmpalooza's provider. One process is one
// block of a timing session; the session driver
// (docs/experiments/backend-proof-p1/bpf1_session.py) runs the blocks in
// the protocol's order and records the host's state between them.
//
// The cases come from a case file (bpf1_cases.py derives it from the
// bridge's recorded plan, fp16-plan.json): each names an operation, its
// shape, the implementation (MMVF, MMF or GGML's cuBLAS path) and the
// launches one invocation must make. For each case, in order:
//
// - A ring of operand sets in the block's memory kind, each set holding all
//   of the case's tensors (weights, activations, output), with more sets
//   than fit four times in L2 (and at least two), all with the same
//   contents. Invocation k uses set k mod N, so consecutive invocations
//   never read the same set, and a set is read again only after the ring's
//   N - 1 others, which with it total more than 4 x L2. GGML's scratch and
//   the cuBLAS workspace are in the same memory kind; they are written
//   before they are read.
// - One eager invocation on set 0, whose output is hashed.
// - 36 CUDA graphs, each of ten invocations on consecutive sets bracketed
//   by two captured timing events. Every graph's launches are read back
//   from the graph and must equal the case file's, ten times over, in
//   order: kernel name (with a compiler-internal namespace tag
//   normalized), grid, block, static plus dynamic shared memory and
//   registers per thread; or a memset's size and value. A mismatch ends the
//   block before anything is timed.
// - Five warm replays, then 31 samples: each replays one graph and records
//   the events' interval divided by ten.
// - The stream-launched arm (reported, not gated): the same, with the ten
//   invocations launched on the stream between two events, on the next 360
//   sets of the ring.
// - The output of set 0 is hashed again and must equal the eager hash.
//
// A host- or device-VMM block needs --calibration-sha256: the driver passes
// the SHA-256 of the calibration of BP-F1's rule for that memory kind only
// once it matches the one pre-registered in docs/backend-proof.md, and the
// block records it. The setup staging buffer is host VMM in every kind; no
// timed kernel touches it.
//
//   llmp_ggml_vmm_bench --cases FILE --memory cuda-malloc|host-vmm|device-vmm --output FILE
//                         [--calibration-sha256 HEX] [--case NAME@ROWS]...

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <expected>
#include <format>
#include <fstream>
#include <memory>
#include <print>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "base/bytes.h"
#include "ggml.h"
#include "kernels/ggml/cublas.h"
#include "kernels/ggml/launch.h"
#include "kernels/ggml/ops.h"
#include "kernels/ggml/tensors.h"
#include "providers/cuda/cuda_device_execution.h"
#include "providers/cuda/cuda_device_memory.h"
#include "providers/device_execution.h"
#include "providers/device_memory.h"

namespace {

using llmp::base::Bytes;
using llmp::kernels::ggml::CublasHandle;
using llmp::kernels::ggml::KernelFailure;
using llmp::kernels::ggml::LaunchContext;
using llmp::kernels::ggml::TensorArena;
using llmp::providers::Access;
using llmp::providers::BackingKind;
using llmp::providers::DeviceExecution;
using llmp::providers::FenceState;
using llmp::providers::StreamId;
using llmp::providers::VmmProvider;

constexpr int kInvocations = 10;  // per graph and per stream sample
constexpr int kWarmups = 5;
constexpr int kSamples = 31;
constexpr int kReplays = kWarmups + kSamples;
constexpr std::uint64_t kAlign = 256;
constexpr std::uint64_t kStaging = std::uint64_t{64} << 20U;
constexpr float kEps = 1e-6f;  // Qwen2's RMSNorm epsilon

template <typename... Args>
std::unexpected<std::string> Fail(std::format_string<Args...> format, Args&&... args) {
  return std::unexpected(std::format(format, std::forward<Args>(args)...));
}

std::uint64_t RoundUp(std::uint64_t value, std::uint64_t to) { return (value + to - 1) / to * to; }

// Cleanup whose failure changes no result: reported, not fatal.
template <typename T, typename E>
void Report(const std::expected<T, E>& result, std::string_view what) {
  if (!result) {
    std::println(stderr, "warning: {} failed", what);
  }
}

// ---- The case file (bpf1_cases.py) ----

// One launch of an invocation: a kernel or a memset.
struct Launch {
  bool memset = false;
  std::string name;
  std::array<unsigned, 3> grid{};
  std::array<unsigned, 3> block{};
  std::uint64_t shared = 0;  // static plus dynamic bytes
  int registers = 0;
  std::uint64_t bytes = 0;  // a memset's
  unsigned value = 0;

  bool operator==(const Launch&) const = default;

  std::string ToString() const {
    if (memset) {
      return std::format("memset {} {}", bytes, value);
    }
    return std::format("kernel {} grid {}x{}x{} block {}x{}x{} shared {} registers {}", name,
                       grid[0], grid[1], grid[2], block[0], block[1], block[2], shared, registers);
  }
};

enum class Op : std::uint8_t { kRmsNorm, kRmsNormMul, kMul, kAddBias, kAdd, kLinear, kKq, kKqv };
enum class Impl : std::uint8_t { kNone, kMmvf, kMmf, kCublas };

struct Case {
  std::string name;
  std::int64_t rows = 0;
  Op op = Op::kRmsNorm;
  Impl impl = Impl::kNone;
  std::vector<std::int64_t> params;
  std::vector<Launch> launches;
};

struct CaseFile {
  std::string set;
  std::vector<Case> cases;
};

std::vector<std::string_view> Words(std::string_view line) {
  std::vector<std::string_view> words;
  std::size_t at = 0;
  while (at < line.size()) {
    const std::size_t start = line.find_first_not_of(' ', at);
    if (start == std::string_view::npos) {
      break;
    }
    const std::size_t end = std::min(line.find(' ', start), line.size());
    words.push_back(line.substr(start, end - start));
    at = end;
  }
  return words;
}

template <typename T>
bool Number(std::string_view text, T& value) {
  const char* begin = text.data();
  const char* end = begin + text.size();
  const auto parsed = std::from_chars(begin, end, value);
  return parsed.ec == std::errc{} && parsed.ptr == end;
}

std::expected<Op, std::string> OpNamed(std::string_view name) {
  static constexpr std::array<std::pair<std::string_view, Op>, 8> kOps{{
      {"rms_norm", Op::kRmsNorm},
      {"rms_norm_mul", Op::kRmsNormMul},
      {"mul", Op::kMul},
      {"add_bias", Op::kAddBias},
      {"add", Op::kAdd},
      {"linear", Op::kLinear},
      {"kq", Op::kKq},
      {"kqv", Op::kKqv},
  }};
  for (const auto& [text, op] : kOps) {
    if (text == name) {
      return op;
    }
  }
  return Fail("unknown operation {}", name);
}

// The parameters each operation takes, and whether it is a product.
std::size_t ParamCount(Op op) {
  switch (op) {
    case Op::kLinear:
      return 2;  // input width K, output width N
    case Op::kKq:
    case Op::kKqv:
      return 5;  // KV cells in use, cache size, head size, heads, KV heads
    default:
      return 1;  // row width
  }
}
bool Product(Op op) { return op == Op::kLinear || op == Op::kKq || op == Op::kKqv; }

std::expected<CaseFile, std::string> ReadCases(const std::string& path) {
  std::ifstream in(path);
  if (!in) {
    return Fail("cannot read {}", path);
  }
  CaseFile file;
  Case* open = nullptr;
  std::string line;
  for (int number = 1; std::getline(in, line); ++number) {
    const auto words = Words(line);
    if (words.empty() || words[0].starts_with('#')) {
      continue;
    }
    const auto bad = [&](std::string_view why) { return Fail("{}:{}: {}", path, number, why); };
    if (words[0] == "set" && words.size() == 2 && file.set.empty()) {
      file.set = words[1];
    } else if (words[0] == "case" && words.size() >= 5 && open == nullptr) {
      Case c;
      c.name = words[1];
      auto op = OpNamed(words[3]);
      if (!Number(words[2], c.rows) || c.rows <= 0 || !op) {
        return bad("a case needs positive rows and a known operation");
      }
      c.op = *op;
      if (words[4] == "mmvf") {
        c.impl = Impl::kMmvf;
      } else if (words[4] == "mmf") {
        c.impl = Impl::kMmf;
      } else if (words[4] == "cublas") {
        c.impl = Impl::kCublas;
      } else if (words[4] != "-") {
        return bad("unknown implementation");
      }
      if (Product(c.op) != (c.impl != Impl::kNone) || words.size() != 5 + ParamCount(c.op)) {
        return bad("the implementation or parameters do not fit the operation");
      }
      for (std::size_t i = 5; i < words.size(); ++i) {
        std::int64_t value = 0;
        if (!Number(words[i], value) || value <= 0) {
          return bad("parameters are positive integers");
        }
        c.params.push_back(value);
      }
      file.cases.push_back(std::move(c));
      open = &file.cases.back();
    } else if (words[0] == "kernel" && words.size() == 10 && open != nullptr) {
      Launch launch;
      launch.name = words[1];
      bool ok = true;
      for (std::size_t i = 0; i < 3; ++i) {
        ok = ok && Number(words[2 + i], launch.grid[i]) && Number(words[5 + i], launch.block[i]);
      }
      if (!ok || !Number(words[8], launch.shared) || !Number(words[9], launch.registers)) {
        return bad("a kernel line needs a name, grid, block, shared bytes and registers");
      }
      open->launches.push_back(std::move(launch));
    } else if (words[0] == "memset" && words.size() == 3 && open != nullptr) {
      Launch launch;
      launch.memset = true;
      if (!Number(words[1], launch.bytes) || !Number(words[2], launch.value)) {
        return bad("a memset line needs its size and value");
      }
      open->launches.push_back(std::move(launch));
    } else if (words[0] == "end" && words.size() == 1 && open != nullptr) {
      if (open->launches.empty()) {
        return bad("a case needs launches");
      }
      open = nullptr;
    } else {
      return bad("unexpected line");
    }
  }
  if (open != nullptr || file.cases.empty() || file.set.empty()) {
    return Fail("{}: needs a set and complete cases", path);
  }
  return file;
}

// Replaces a length-prefixed compiler-internal namespace name (NVCC tags
// internal-linkage names per translation unit) with `_INTERNAL_`, as
// bpf1_cases.py does.
std::string Normalize(std::string_view mangled) {
  constexpr std::string_view kTag = "_INTERNAL_";
  std::string out;
  std::size_t at = 0;
  for (std::size_t found = mangled.find(kTag); found != std::string_view::npos;
       found = mangled.find(kTag, found + 1)) {
    std::size_t digits = found;
    while (digits > at && mangled[digits - 1] >= '0' && mangled[digits - 1] <= '9') {
      --digits;
    }
    bool replaced = false;
    for (std::size_t cut = digits; cut < found && !replaced; ++cut) {
      std::size_t length = 0;
      if (Number(mangled.substr(cut, found - cut), length) && length >= kTag.size() &&
          found + length <= mangled.size()) {
        out += mangled.substr(at, cut - at);
        out += kTag;
        at = found + length;
        replaced = true;
      }
    }
  }
  out += mangled.substr(at);
  return out;
}

// ---- Device memory of one kind ----

enum class Memory : std::uint8_t { kCudaMalloc, kHostVmm, kDeviceVmm };

struct Allocation {
  std::uint64_t base = 0;
  std::uint64_t size = 0;
  void* malloced = nullptr;
  llmp::providers::ReservationId reservation;
  llmp::providers::BackingId backing;
};

class Device {
 public:
  static std::expected<std::unique_ptr<Device>, std::string> Open() {
    auto memory = llmp::providers::cuda::OpenDeviceMemory(0);
    if (!memory) {
      return Fail("no device memory: {}", memory.error().detail);
    }
    auto execution = llmp::providers::cuda::OpenDeviceExecution(0);
    if (!execution) {
      return Fail("no device execution: {}", execution.error().detail);
    }
    auto stream = (*execution)->CreateStream();
    if (!stream) {
      return Fail("no stream: {}", stream.error().detail);
    }
    auto device = std::unique_ptr<Device>(
        new Device(std::move(*memory), std::move(*execution), *stream));  // NOLINT
    auto staging = device->Vmm(BackingKind::kHost, kStaging);
    if (!staging) {
      return std::unexpected(staging.error());
    }
    device->staging_ = *staging;
    return device;
  }

  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;
  Device(Device&&) = delete;
  Device& operator=(Device&&) = delete;
  ~Device() {
    Report(Finish(), "the final fence");
    Free(staging_);
    Report(execution_->DestroyStream(stream_), "destroying the stream");
  }

  // `size` bytes of `kind`, 256-byte aligned (VMM: granule-aligned).
  std::expected<Allocation, std::string> Allocate(Memory kind, std::uint64_t size) {
    if (kind == Memory::kHostVmm) {
      return Vmm(BackingKind::kHost, size);
    }
    if (kind == Memory::kDeviceVmm) {
      return Vmm(BackingKind::kDevice, size);
    }
    Allocation allocation;
    allocation.size = size;
    if (const cudaError_t error = cudaMalloc(&allocation.malloced, size); error != cudaSuccess) {
      return Fail("cudaMalloc of {} bytes: {}", size, cudaGetErrorString(error));
    }
    allocation.base = reinterpret_cast<std::uintptr_t>(allocation.malloced);
    return allocation;
  }

  void Free(const Allocation& allocation) {
    if (allocation.size == 0) {
      return;
    }
    if (allocation.malloced != nullptr) {
      (void)cudaFree(allocation.malloced);
      return;
    }
    const Bytes rounded(RoundUp(allocation.size, memory_->Granularity().value()));
    Report(memory_->Unmap(allocation.reservation, Bytes(0), rounded), "unmap");
    Report(memory_->Release(allocation.backing), "release");
    Report(memory_->Free(allocation.reservation), "free");
  }

  // Everything queued so far has run.
  std::expected<void, std::string> Finish() {
    auto fence = execution_->Record(stream_);
    if (!fence) {
      return Fail("fence: {}", fence.error().detail);
    }
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
    FenceState state = FenceState::kPending;
    while (true) {
      auto queried = execution_->Query(*fence);
      if (!queried) {
        return Fail("fence query: {}", queried.error().detail);
      }
      state = *queried;
      if (state == FenceState::kComplete || std::chrono::steady_clock::now() > deadline) {
        break;
      }
      std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
    Report(execution_->Release(*fence), "releasing a fence");
    if (state != FenceState::kComplete) {
      return Fail("the stream did not finish within a minute");
    }
    return {};
  }

  // Copies host bytes to `address` through the staging region.
  std::expected<void, std::string> Upload(std::uint64_t address, std::span<const std::byte> data) {
    for (std::size_t done = 0; done < data.size();) {
      const std::size_t part = std::min<std::size_t>(data.size() - done, kStaging);
      std::memcpy(reinterpret_cast<void*>(staging_.base),  // NOLINT(performance-no-int-to-ptr)
                  data.subspan(done).data(), part);
      if (auto copied = Copy(address + done, staging_.base, part); !copied) {
        return copied;
      }
      if (auto finished = Finish(); !finished) {
        return finished;
      }
      done += part;
    }
    return {};
  }

  std::expected<std::vector<std::byte>, std::string> Download(std::uint64_t address,
                                                              std::uint64_t size) {
    std::vector<std::byte> out(size);
    for (std::uint64_t done = 0; done < size;) {
      const std::uint64_t part = std::min(size - done, kStaging);
      if (auto copied = Copy(staging_.base, address + done, part); !copied) {
        return std::unexpected(copied.error());
      }
      if (auto finished = Finish(); !finished) {
        return std::unexpected(finished.error());
      }
      std::memcpy(
          out.data() + done,
          reinterpret_cast<const void*>(staging_.base),  // NOLINT(performance-no-int-to-ptr)
          part);
      done += part;
    }
    return out;
  }

  std::expected<void, std::string> Copy(std::uint64_t to, std::uint64_t from, std::uint64_t size) {
    if (auto copied = execution_->Copy(stream_, to, from, Bytes(size)); !copied) {
      return Fail("copy: {}", copied.error().detail);
    }
    return {};
  }

  // The native stream, taken as queued work (a later Finish releases it).
  std::expected<cudaStream_t, std::string> Stream() {
    auto native = execution_->Submission(stream_);
    if (!native) {
      return Fail("stream: {}", native.error().detail);
    }
    return static_cast<cudaStream_t>(native->handle);
  }

  DeviceExecution& execution() { return *execution_; }
  StreamId stream() const { return stream_; }

 private:
  Device(std::unique_ptr<VmmProvider> memory, std::unique_ptr<DeviceExecution> execution,
         StreamId stream)
      : memory_(std::move(memory)), execution_(std::move(execution)), stream_(stream) {}

  std::expected<Allocation, std::string> Vmm(BackingKind kind, std::uint64_t size) {
    const std::uint64_t rounded = RoundUp(size, memory_->Granularity().value());
    std::size_t kind_class = memory_->Classes().size();
    for (std::size_t i = 0; i < memory_->Classes().size(); ++i) {
      if (memory_->Classes()[i].kind == kind) {
        kind_class = i;
      }
    }
    if (kind_class == memory_->Classes().size()) {
      return Fail("no allocation class of that kind");
    }
    auto reservation = memory_->Reserve(Bytes(rounded));
    if (!reservation) {
      return Fail("reserve {} bytes: {}", rounded, reservation.error().detail);
    }
    auto backing = memory_->Create(kind_class, Bytes(rounded));
    if (!backing) {
      Report(memory_->Free(*reservation), "free");
      return Fail("create {} bytes: {}", rounded, backing.error().detail);
    }
    if (auto mapped = memory_->Map(*reservation, Bytes(0), *backing); !mapped) {
      return Fail("map: {}", mapped.error().detail);
    }
    if (auto access =
            memory_->SetAccess(*reservation, Bytes(0), Bytes(rounded), Access::kReadWrite);
        !access) {
      return Fail("access: {}", access.error().detail);
    }
    return Allocation{.base = memory_->RangeOf(*reservation).value().base,
                      .size = size,
                      .malloced = nullptr,
                      .reservation = *reservation,
                      .backing = *backing};
  }

  std::unique_ptr<VmmProvider> memory_;
  std::unique_ptr<DeviceExecution> execution_;
  StreamId stream_;
  Allocation staging_;
};

// ---- Operands ----

// Deterministic values in [-1, 1).
float Value(std::uint64_t seed, std::uint64_t i) {
  std::uint64_t x = (seed * 0x9E3779B97F4A7C15ULL) ^ (i + 0x632BE59BD9B4E019ULL);
  x ^= x >> 31U;
  x *= 0xBF58476D1CE4E5B9ULL;
  x ^= x >> 29U;
  return static_cast<float>(static_cast<double>(x >> 40U) / static_cast<double>(1ULL << 23U)) -
         1.0f;
}

// One tensor of an operand set: its type, extents, offset in the set and
// how its values are made (Value(seed, i) * scale + shift). The output is
// not filled.
struct Operand {
  ggml_type type = GGML_TYPE_F32;
  std::array<std::int64_t, 4> ne{1, 1, 1, 1};
  std::uint64_t offset = 0;
  std::uint64_t bytes = 0;
  float scale = 1.0f;
  float shift = 0.0f;
};

struct Layout {
  std::vector<Operand> operands;  // the last is the output
  std::uint64_t set_bytes = 0;
};

Layout LayoutOf(const Case& c) {
  const std::int64_t r = c.rows;
  const auto f32 = [](std::array<std::int64_t, 4> ne, float scale = 1.0f, float shift = 0.0f) {
    return Operand{
        .type = GGML_TYPE_F32, .ne = ne, .offset = 0, .bytes = 0, .scale = scale, .shift = shift};
  };
  const auto f16 = [](std::array<std::int64_t, 4> ne, float scale) {
    return Operand{
        .type = GGML_TYPE_F16, .ne = ne, .offset = 0, .bytes = 0, .scale = scale, .shift = 0.0f};
  };
  Layout layout;
  const std::int64_t w = c.params[0];
  switch (c.op) {
    case Op::kRmsNorm:
      layout.operands = {f32({w, r, 1, 1}, 4.0f), f32({w, r, 1, 1})};
      break;
    case Op::kRmsNormMul:
    case Op::kMul:
    case Op::kAddBias:
      layout.operands = {f32({w, r, 1, 1}, 4.0f), f32({w, 1, 1, 1}, 1.0f, 1.0f), f32({w, r, 1, 1})};
      break;
    case Op::kAdd:
      layout.operands = {f32({w, r, 1, 1}), f32({w, r, 1, 1}), f32({w, r, 1, 1})};
      break;
    case Op::kLinear: {
      const std::int64_t k = c.params[0];
      const std::int64_t n = c.params[1];
      layout.operands = {f16({k, n, 1, 1}, 0.1f), f32({k, r, 1, 1}), f32({n, r, 1, 1})};
      break;
    }
    case Op::kKq: {
      const auto [kv, size, head, heads, kv_heads] =
          std::array{c.params[0], c.params[1], c.params[2], c.params[3], c.params[4]};
      layout.operands = {f16({head * kv_heads, size, 1, 1}, 1.0f), f32({head, heads, r, 1}, 0.125f),
                         f32({kv, r, heads, 1})};
      break;
    }
    case Op::kKqv: {
      const auto [kv, size, head, heads, kv_heads] =
          std::array{c.params[0], c.params[1], c.params[2], c.params[3], c.params[4]};
      // Softmax output: positive, about 1 / kv.
      layout.operands = {
          f16({size, head * kv_heads, 1, 1}, 1.0f),
          f32({kv, r, heads, 1}, 1.0f / static_cast<float>(kv), 1.0f / static_cast<float>(kv)),
          f32({head, r, heads, 1})};
      break;
    }
  }
  for (Operand& operand : layout.operands) {
    const std::int64_t count = operand.ne[0] * operand.ne[1] * operand.ne[2] * operand.ne[3];
    operand.bytes = static_cast<std::uint64_t>(count) * (operand.type == GGML_TYPE_F16 ? 2U : 4U);
    operand.offset = layout.set_bytes;
    layout.set_bytes += RoundUp(operand.bytes, kAlign);
  }
  return layout;
}

std::vector<std::byte> SetContents(const Layout& layout) {
  std::vector<std::byte> bytes(layout.set_bytes);
  for (std::size_t t = 0; t + 1 < layout.operands.size(); ++t) {
    const Operand& operand = layout.operands[t];
    const std::uint64_t count = operand.bytes / (operand.type == GGML_TYPE_F16 ? 2U : 4U);
    std::byte* at = bytes.data() + operand.offset;
    for (std::uint64_t i = 0; i < count; ++i) {
      const float value = (Value(t + 1, i) * operand.scale) + operand.shift;
      if (operand.type == GGML_TYPE_F16) {
        const ggml_fp16_t half = ggml_fp32_to_fp16(value);
        std::memcpy(at + (i * sizeof half), &half, sizeof half);
      } else {
        std::memcpy(at + (i * sizeof value), &value, sizeof value);
      }
    }
  }
  return bytes;
}

// One invocation's nodes, bound to one operand set.
struct Nodes {
  ggml_tensor* node = nullptr;
  ggml_tensor* norm = nullptr;  // the fused launcher's norm (never written)
};

constexpr std::size_t kTensorsPerSet = 8;

std::expected<Nodes, std::string> Build(TensorArena& arena, const Case& c, const Layout& layout,
                                        std::uint64_t base) {
  if (!arena.Reserve(kTensorsPerSet)) {
    return Fail("the tensor arena is full");
  }
  ggml_context* context = arena.context();
  std::vector<ggml_tensor*> t;
  for (std::size_t i = 0; i + 1 < layout.operands.size(); ++i) {
    const Operand& o = layout.operands[i];
    ggml_tensor* tensor = ggml_new_tensor_4d(context, o.type, o.ne[0], o.ne[1], o.ne[2], o.ne[3]);
    TensorArena::Bind(tensor, base + o.offset);
    t.push_back(tensor);
  }
  Nodes nodes;
  switch (c.op) {
    case Op::kRmsNorm:
      nodes.node = ggml_rms_norm(context, t[0], kEps);
      break;
    case Op::kRmsNormMul:
      nodes.norm = ggml_rms_norm(context, t[0], kEps);
      nodes.node = ggml_mul(context, nodes.norm, t[1]);
      break;
    case Op::kMul:
      nodes.node = ggml_mul(context, t[0], t[1]);
      break;
    case Op::kAddBias:
    case Op::kAdd:
      nodes.node = ggml_add(context, t[0], t[1]);
      break;
    case Op::kLinear:
      nodes.node = ggml_mul_mat(context, t[0], t[1]);
      break;
    case Op::kKq: {
      // llama.cpp's attention without flash attention: K a view of the F16
      // cache, grouped over the query heads, with F32 precision.
      const std::int64_t kv = c.params[0];
      const std::int64_t head = c.params[2];
      const std::int64_t kv_heads = c.params[4];
      ggml_tensor* k = ggml_permute(context,
                                    ggml_view_3d(context, t[0], head, kv_heads, kv,
                                                 ggml_row_size(t[0]->type, head), t[0]->nb[1], 0),
                                    0, 2, 1, 3);
      ggml_tensor* q = ggml_permute(context, t[1], 0, 2, 1, 3);
      nodes.node = ggml_mul_mat(context, k, q);
      if (!ggml_prec_set_acc(nodes.node, GGML_PREC_F32)) {
        return Fail("cannot set F32 precision");
      }
      break;
    }
    case Op::kKqv: {
      // V a view of the transposed F16 cache.
      const std::int64_t kv = c.params[0];
      const std::int64_t head = c.params[2];
      const std::int64_t kv_heads = c.params[4];
      ggml_tensor* v = ggml_view_3d(context, t[0], kv, head, kv_heads, t[0]->nb[1],
                                    t[0]->nb[1] * static_cast<std::size_t>(head), 0);
      nodes.node = ggml_mul_mat(context, v, t[1]);
      break;
    }
  }
  const Operand& out = layout.operands.back();
  if (ggml_nbytes(nodes.node) != out.bytes || !ggml_is_contiguous(nodes.node)) {
    return Fail("{}: the output is not the {} bytes laid out", c.name, out.bytes);
  }
  TensorArena::Bind(nodes.node, base + out.offset);
  return nodes;
}

std::expected<void, KernelFailure> Invoke(const Case& c, const Nodes& nodes,
                                          LaunchContext& launch) {
  namespace ops = llmp::kernels::ggml;
  switch (c.op) {
    case Op::kRmsNorm:
      return ops::RmsNorm(launch, nodes.node);
    case Op::kRmsNormMul:
      return ops::RmsNormMul(launch, nodes.norm, nodes.node);
    case Op::kMul:
      return ops::Mul(launch, nodes.node);
    case Op::kAddBias:
    case Op::kAdd:
      return ops::Add(launch, nodes.node);
    default:
      break;
  }
  switch (c.impl) {
    case Impl::kMmvf:
      return ops::MulMatVecF(launch, nodes.node);
    case Impl::kMmf:
      return ops::MulMatF(launch, nodes.node);
    case Impl::kCublas:
      return ops::MulMatCublas(launch, nodes.node);
    case Impl::kNone:
      break;
  }
  return std::unexpected(KernelFailure{.detail = "no implementation"});
}

std::uint64_t Fnv(std::span<const std::byte> bytes) {
  std::uint64_t hash = 0xcbf29ce484222325ULL;
  for (const std::byte b : bytes) {
    hash = (hash ^ std::to_integer<std::uint64_t>(b)) * 0x100000001b3ULL;
  }
  return hash;
}

// ---- Graph inspection ----

std::expected<void, std::string> Driver(CUresult result, std::string_view what) {
  if (result == CUDA_SUCCESS) {
    return {};
  }
  const char* name = nullptr;
  (void)cuGetErrorName(result, &name);
  return Fail("{}: {}", what, name == nullptr ? "unknown error" : name);
}

std::expected<Launch, std::string> KernelLaunch(CUgraphNode node, CUdevice device) {
  CUDA_KERNEL_NODE_PARAMS params{};
  if (auto got = Driver(cuGraphKernelNodeGetParams(node, &params), "kernel node"); !got) {
    return std::unexpected(got.error());
  }
  Launch launch;
  launch.grid = {params.gridDimX, params.gridDimY, params.gridDimZ};
  launch.block = {params.blockDimX, params.blockDimY, params.blockDimZ};
  const char* name = nullptr;
  int registers = 0;
  int shared = 0;
  if (params.func != nullptr) {
    if (auto got = Driver(cuFuncGetName(&name, params.func), "kernel name"); !got) {
      return std::unexpected(got.error());
    }
    if (auto got = Driver(cuFuncGetAttribute(&registers, CU_FUNC_ATTRIBUTE_NUM_REGS, params.func),
                          "registers");
        !got) {
      return std::unexpected(got.error());
    }
    if (auto got =
            Driver(cuFuncGetAttribute(&shared, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, params.func),
                   "static shared memory");
        !got) {
      return std::unexpected(got.error());
    }
  } else if (params.kern != nullptr) {
    if (auto got = Driver(cuKernelGetName(&name, params.kern), "kernel name"); !got) {
      return std::unexpected(got.error());
    }
    if (auto got = Driver(
            cuKernelGetAttribute(&registers, CU_FUNC_ATTRIBUTE_NUM_REGS, params.kern, device),
            "registers");
        !got) {
      return std::unexpected(got.error());
    }
    if (auto got = Driver(
            cuKernelGetAttribute(&shared, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, params.kern, device),
            "static shared memory");
        !got) {
      return std::unexpected(got.error());
    }
  } else {
    return Fail("a kernel node with neither a function nor a kernel");
  }
  launch.name = Normalize(name == nullptr ? "" : name);
  launch.registers = registers;
  launch.shared = params.sharedMemBytes + static_cast<std::uint64_t>(shared);
  return launch;
}

// The graph's work in order, less its first and last nodes, which must be
// the timing events. A graph captured from one stream is a chain.
std::expected<std::vector<Launch>, std::string> GraphLaunches(cudaGraph_t graph, CUdevice device) {
  CUgraph g = graph;  // the same type
  std::size_t count = 0;
  if (auto got = Driver(cuGraphGetNodes(g, nullptr, &count), "graph nodes"); !got) {
    return std::unexpected(got.error());
  }
  std::vector<CUgraphNode> nodes(count);
  if (auto got = Driver(cuGraphGetNodes(g, nodes.data(), &count), "graph nodes"); !got) {
    return std::unexpected(got.error());
  }
  std::size_t edges = 0;
  if (auto got = Driver(cuGraphGetEdges(g, nullptr, nullptr, nullptr, &edges), "graph edges");
      !got) {
    return std::unexpected(got.error());
  }
  std::vector<CUgraphNode> from(edges);
  std::vector<CUgraphNode> to(edges);
  std::vector<CUgraphEdgeData> data(edges);
  if (auto got =
          Driver(cuGraphGetEdges(g, from.data(), to.data(), data.data(), &edges), "graph edges");
      !got) {
    return std::unexpected(got.error());
  }
  std::unordered_map<CUgraphNode, std::size_t> incoming;
  std::unordered_map<CUgraphNode, CUgraphNode> next;
  for (std::size_t e = 0; e < edges; ++e) {
    ++incoming[to[e]];
    if (!next.emplace(from[e], to[e]).second) {
      return Fail("the graph branches");
    }
  }
  CUgraphNode at = nullptr;
  for (CUgraphNode node : nodes) {
    if (!incoming.contains(node)) {
      if (at != nullptr) {
        return Fail("the graph has more than one root");
      }
      at = node;
    }
  }
  std::vector<CUgraphNode> order;
  while (at != nullptr) {
    if (incoming.contains(at) && incoming[at] != 1) {
      return Fail("a node has more than one dependency");
    }
    order.push_back(at);
    const auto following = next.find(at);
    at = following == next.end() ? nullptr : following->second;
  }
  if (order.size() != nodes.size() || order.size() < 2) {
    return Fail("the graph is not one chain of work between two events");
  }
  std::vector<Launch> launches;
  for (std::size_t i = 0; i < order.size(); ++i) {
    CUgraphNodeType type{};
    if (auto got = Driver(cuGraphNodeGetType(order[i], &type), "node type"); !got) {
      return std::unexpected(got.error());
    }
    const bool edge = i == 0 || i + 1 == order.size();
    if (edge != (type == CU_GRAPH_NODE_TYPE_EVENT_RECORD)) {
      return Fail("the timing events are not the graph's first and last nodes");
    }
    if (edge) {
      continue;
    }
    if (type == CU_GRAPH_NODE_TYPE_KERNEL) {
      auto launch = KernelLaunch(order[i], device);
      if (!launch) {
        return std::unexpected(launch.error());
      }
      launches.push_back(std::move(*launch));
    } else if (type == CU_GRAPH_NODE_TYPE_MEMSET) {
      CUDA_MEMSET_NODE_PARAMS params{};
      if (auto got = Driver(cuGraphMemsetNodeGetParams(order[i], &params), "memset node"); !got) {
        return std::unexpected(got.error());
      }
      Launch memset;
      memset.memset = true;
      memset.bytes = params.width * params.elementSize * params.height;
      memset.value = params.value;
      launches.push_back(std::move(memset));
    } else {
      return Fail("an unexpected graph node of type {}", static_cast<int>(type));
    }
  }
  return launches;
}

// ---- One case ----

struct Result {
  std::string name;
  std::int64_t rows = 0;
  std::uint64_t set_bytes = 0;
  std::uint64_t sets = 0;
  std::uint64_t scratch = 0;
  std::size_t launches = 0;
  std::uint64_t eager_hash = 0;
  std::uint64_t final_hash = 0;
  std::vector<double> graph_us;
  std::vector<double> stream_us;
};

class Events {
 public:
  Events() {
    (void)cudaEventCreate(&start_);
    (void)cudaEventCreate(&stop_);
  }
  Events(const Events&) = delete;
  Events& operator=(const Events&) = delete;
  Events(Events&&) = delete;
  Events& operator=(Events&&) = delete;
  ~Events() {
    (void)cudaEventDestroy(start_);
    (void)cudaEventDestroy(stop_);
  }
  cudaEvent_t start() const { return start_; }
  cudaEvent_t stop() const { return stop_; }

  // Microseconds per invocation between the two events, once they ran.
  std::expected<double, std::string> PerInvocation() const {
    if (const cudaError_t error = cudaEventSynchronize(stop_); error != cudaSuccess) {
      return Fail("event: {}", cudaGetErrorString(error));
    }
    float ms = 0;
    if (const cudaError_t error = cudaEventElapsedTime(&ms, start_, stop_); error != cudaSuccess) {
      return Fail("elapsed time: {}", cudaGetErrorString(error));
    }
    return static_cast<double>(ms) * 1000.0 / kInvocations;
  }

 private:
  cudaEvent_t start_ = nullptr;
  cudaEvent_t stop_ = nullptr;
};

class Graphs {
 public:
  Graphs() = default;
  Graphs(const Graphs&) = delete;
  Graphs& operator=(const Graphs&) = delete;
  Graphs(Graphs&&) = delete;
  Graphs& operator=(Graphs&&) = delete;
  ~Graphs() {
    for (cudaGraphExec_t exec : execs) {
      (void)cudaGraphExecDestroy(exec);
    }
    for (cudaGraph_t graph : graphs) {
      (void)cudaGraphDestroy(graph);
    }
  }
  std::vector<cudaGraph_t> graphs;
  std::vector<cudaGraphExec_t> execs;
};

// Frees a case's allocations when it goes out of scope.
class Held {
 public:
  explicit Held(Device& device) : device_(device) {}
  Held(const Held&) = delete;
  Held& operator=(const Held&) = delete;
  Held(Held&&) = delete;
  Held& operator=(Held&&) = delete;
  ~Held() {
    Report(device_.Finish(), "a case's final fence");
    for (const Allocation& a : held_) {
      device_.Free(a);
    }
  }
  std::expected<std::uint64_t, std::string> Allocate(Memory kind, std::uint64_t size) {
    auto a = device_.Allocate(kind, size);
    if (!a) {
      return std::unexpected(a.error());
    }
    held_.push_back(*a);
    return a->base;
  }

 private:
  Device& device_;
  std::vector<Allocation> held_;
};

struct Context {
  Device& device;
  CublasHandle& cublas;
  Memory memory = Memory::kCudaMalloc;
  std::uint64_t l2 = 0;
  CUdevice cu_device = 0;
};

// Returns the case's result; a verification mismatch sets `mismatch`.
std::expected<Result, std::string> RunCase(Context& ctx, const Case& c, bool& mismatch) {
  Device& device = ctx.device;
  const Layout layout = LayoutOf(c);
  Result result;
  result.name = c.name;
  result.rows = c.rows;
  result.set_bytes = layout.set_bytes;
  // More sets than fit four times in L2, and at least two.
  const std::uint64_t sets = std::max<std::uint64_t>(2, (4 * ctx.l2 / layout.set_bytes) + 1);
  result.sets = sets;
  constexpr std::uint64_t kUsed = 2ULL * kReplays * kInvocations;  // graph and stream arms
  const std::uint64_t built = std::min(sets, kUsed);

  Held held(device);
  auto ring = held.Allocate(ctx.memory, sets * layout.set_bytes);
  if (!ring) {
    return std::unexpected(ring.error());
  }
  const std::vector<std::byte> contents = SetContents(layout);
  if (auto uploaded = device.Upload(*ring, contents); !uploaded) {
    return std::unexpected(uploaded.error());
  }
  // Every set holds the same bytes: double the filled prefix until done.
  for (std::uint64_t filled = 1; filled < sets;) {
    const std::uint64_t more = std::min(filled, sets - filled);
    if (auto copied =
            device.Copy(*ring + (filled * layout.set_bytes), *ring, more * layout.set_bytes);
        !copied) {
      return std::unexpected(copied.error());
    }
    filled += more;
  }
  if (auto finished = device.Finish(); !finished) {
    return std::unexpected(finished.error());
  }

  auto arena = TensorArena::Create(built * kTensorsPerSet);
  if (!arena) {
    return Fail("tensor arena: {}", arena.error().detail);
  }
  std::vector<Nodes> nodes;
  for (std::uint64_t s = 0; s < built; ++s) {
    auto n = Build(*arena, c, layout, *ring + (s * layout.set_bytes));
    if (!n) {
      return std::unexpected(n.error());
    }
    nodes.push_back(*n);
  }
  const auto set_of = [&](std::uint64_t invocation) -> const Nodes& {
    return nodes[(invocation % sets) % built];
  };

  // The scratch a product draws, in the block's memory kind.
  std::uint64_t scratch = 0;
  if (c.impl == Impl::kCublas) {
    auto planner = LaunchContext::Create(0, device.execution(), device.stream(),
                                         {.base = 0, .size = Bytes(0)}, &ctx.cublas);
    if (!planner) {
      return Fail("launch context: {}", planner.error().detail);
    }
    auto plan = llmp::kernels::ggml::PlanMulMatCublas(**planner, nodes[0].node);
    if (!plan) {
      return Fail("{} at {} rows: {}", c.name, c.rows, plan.error().detail);
    }
    scratch = plan->scratch;
  }
  result.scratch = scratch;
  std::uint64_t scratch_base = 0;
  if (scratch > 0) {
    auto allocated = held.Allocate(ctx.memory, scratch);
    if (!allocated) {
      return std::unexpected(allocated.error());
    }
    scratch_base = *allocated;
  }
  auto launch = LaunchContext::Create(0, device.execution(), device.stream(),
                                      {.base = scratch_base, .size = Bytes(scratch)}, &ctx.cublas);
  if (!launch) {
    return Fail("launch context: {}", launch.error().detail);
  }
  const auto invoke = [&](std::uint64_t invocation) -> std::expected<void, std::string> {
    if (auto ran = Invoke(c, set_of(invocation), **launch); !ran) {
      return Fail("{} at {} rows: {}", c.name, c.rows, ran.error().detail);
    }
    return {};
  };

  // The eager output.
  const Operand& out = layout.operands.back();
  if (auto ran = invoke(0); !ran) {
    return std::unexpected(ran.error());
  }
  auto eager = device.Download(*ring + out.offset, out.bytes);
  if (!eager) {
    return std::unexpected(eager.error());
  }
  result.eager_hash = Fnv(*eager);

  auto stream = device.Stream();
  if (!stream) {
    return std::unexpected(stream.error());
  }
  Events events;
  Graphs graphs;
  for (int replay = 0; replay < kReplays; ++replay) {
    if (cudaStreamBeginCapture(*stream, cudaStreamCaptureModeThreadLocal) != cudaSuccess) {
      return Fail("cannot begin a capture");
    }
    std::expected<void, std::string> captured =
        cudaEventRecordWithFlags(events.start(), *stream, cudaEventRecordExternal) == cudaSuccess
            ? std::expected<void, std::string>{}
            : Fail("cannot capture the start event");
    for (int i = 0; i < kInvocations && captured; ++i) {
      captured = invoke((static_cast<std::uint64_t>(replay) * kInvocations) +
                        static_cast<std::uint64_t>(i));
    }
    if (captured &&
        cudaEventRecordWithFlags(events.stop(), *stream, cudaEventRecordExternal) != cudaSuccess) {
      captured = Fail("cannot capture the stop event");
    }
    cudaGraph_t graph = nullptr;
    const cudaError_t ended = cudaStreamEndCapture(*stream, &graph);
    if (graph != nullptr) {
      graphs.graphs.push_back(graph);
    }
    if (!captured) {
      return std::unexpected(captured.error());
    }
    if (ended != cudaSuccess) {
      return Fail("capture: {}", cudaGetErrorString(ended));
    }
    auto observed = GraphLaunches(graph, ctx.cu_device);
    if (!observed) {
      return Fail("{} at {} rows: {}", c.name, c.rows, observed.error());
    }
    std::vector<Launch> expected;
    for (int i = 0; i < kInvocations; ++i) {
      expected.insert(expected.end(), c.launches.begin(), c.launches.end());
    }
    if (*observed != expected) {
      mismatch = true;
      std::println(stderr, "{} at {} rows: the launches differ from the recorded plan", c.name,
                   c.rows);
      std::println(stderr, "  recorded, per invocation:");
      for (const Launch& l : c.launches) {
        std::println(stderr, "    {}", l.ToString());
      }
      std::println(stderr, "  captured, {} launches for {} invocations:", observed->size(),
                   kInvocations);
      for (std::size_t i = 0; i < std::min<std::size_t>(observed->size(), c.launches.size() * 2);
           ++i) {
        std::println(stderr, "    {}", (*observed)[i].ToString());
      }
      return Fail("launch verification failed");
    }
    result.launches += observed->size();
    cudaGraphExec_t exec = nullptr;
    if (const cudaError_t error = cudaGraphInstantiate(&exec, graph, 0); error != cudaSuccess) {
      return Fail("instantiate: {}", cudaGetErrorString(error));
    }
    graphs.execs.push_back(exec);
  }

  // Graph samples: five warm replays, then 31.
  for (int replay = 0; replay < kReplays; ++replay) {
    if (const cudaError_t error =
            cudaGraphLaunch(graphs.execs[static_cast<std::size_t>(replay)], *stream);
        error != cudaSuccess) {
      return Fail("graph launch: {}", cudaGetErrorString(error));
    }
    auto us = events.PerInvocation();
    if (!us) {
      return std::unexpected(us.error());
    }
    if (replay >= kWarmups) {
      result.graph_us.push_back(*us);
    }
  }
  // Stream samples, on the next sets of the ring.
  for (int replay = 0; replay < kReplays; ++replay) {
    const std::uint64_t first =
        (static_cast<std::uint64_t>(kReplays + replay)) * static_cast<std::uint64_t>(kInvocations);
    if (cudaEventRecord(events.start(), *stream) != cudaSuccess) {
      return Fail("cannot record the start event");
    }
    for (std::uint64_t i = 0; i < kInvocations; ++i) {
      if (auto ran = invoke(first + i); !ran) {
        return std::unexpected(ran.error());
      }
    }
    if (cudaEventRecord(events.stop(), *stream) != cudaSuccess) {
      return Fail("cannot record the stop event");
    }
    auto us = events.PerInvocation();
    if (!us) {
      return std::unexpected(us.error());
    }
    if (replay >= kWarmups) {
      result.stream_us.push_back(*us);
    }
  }
  auto final_output = device.Download(*ring + out.offset, out.bytes);
  if (!final_output) {
    return std::unexpected(final_output.error());
  }
  result.final_hash = Fnv(*final_output);
  return result;
}

std::string Samples(const std::vector<double>& samples) {
  std::string out = "[";
  for (std::size_t i = 0; i < samples.size(); ++i) {
    out += std::format("{}{:.4f}", i == 0 ? "" : ", ", samples[i]);
  }
  return out + "]";
}

bool Hex64(std::string_view text) {
  return text.size() == 64 && std::ranges::all_of(text, [](char ch) {
           return (ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f');
         });
}

}  // namespace

int main(int argc, char** argv) {
  const std::vector<std::string> args(
      argv, argv + argc);  // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  std::string cases_path;
  std::string output_path;
  std::string calibration;
  std::string memory_name;
  std::vector<std::string> only;
  for (std::size_t i = 1; i + 1 < args.size(); i += 2) {
    const std::string& flag = args[i];
    const std::string& value = args[i + 1];
    if (flag == "--cases") {
      cases_path = value;
    } else if (flag == "--output") {
      output_path = value;
    } else if (flag == "--memory") {
      memory_name = value;
    } else if (flag == "--calibration-sha256") {
      calibration = value;
    } else if (flag == "--case") {
      only.push_back(value);
    } else {
      cases_path.clear();
      break;
    }
  }
  if (args.size() % 2 == 0 || cases_path.empty() || output_path.empty() ||
      (memory_name != "cuda-malloc" && memory_name != "host-vmm" && memory_name != "device-vmm")) {
    std::println(stderr,
                 "usage: llmp_ggml_vmm_bench --cases FILE "
                 "--memory cuda-malloc|host-vmm|device-vmm --output FILE "
                 "[--calibration-sha256 HEX] [--case NAME@ROWS]...");
    return 2;
  }
  Memory memory = Memory::kCudaMalloc;
  if (memory_name == "host-vmm") {
    memory = Memory::kHostVmm;
  } else if (memory_name == "device-vmm") {
    memory = Memory::kDeviceVmm;
  }
  if (!calibration.empty() && !Hex64(calibration)) {
    std::println(stderr, "--calibration-sha256 takes 64 lowercase hex digits");
    return 2;
  }
  if (memory != Memory::kCudaMalloc && calibration.empty()) {
    std::println(stderr,
                 "a {} block needs --calibration-sha256: BP-F1 compares VMM only under its "
                 "pre-registered calibration (docs/backend-proof.md)",
                 memory_name);
    return 2;
  }
  auto file = ReadCases(cases_path);
  if (!file) {
    std::println(stderr, "{}", file.error());
    return 2;
  }
  std::vector<const Case*> selected;
  for (const Case& c : file->cases) {
    const std::string key = std::format("{}@{}", c.name, c.rows);
    if (only.empty() || std::ranges::find(only, key) != only.end()) {
      selected.push_back(&c);
    }
  }
  if (selected.empty()) {
    std::println(stderr, "no case selected");
    return 2;
  }

  auto device = Device::Open();
  if (!device) {
    std::println(stderr, "{}", device.error());
    return 1;
  }
  int l2 = 0;
  int major = 0;
  int minor = 0;
  int sms = 0;
  int driver = 0;
  int runtime = 0;
  cudaDeviceProp properties{};
  CUdevice cu_device = 0;
  if (cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, 0) != cudaSuccess || l2 <= 0 ||
      cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0) != cudaSuccess ||
      cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, 0) != cudaSuccess ||
      cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0) != cudaSuccess ||
      cudaDriverGetVersion(&driver) != cudaSuccess ||
      cudaRuntimeGetVersion(&runtime) != cudaSuccess ||
      cudaGetDeviceProperties(&properties, 0) != cudaSuccess ||
      cuDeviceGet(&cu_device, 0) != CUDA_SUCCESS) {
    std::println(stderr, "cannot query the device");
    return 1;
  }
  const Bytes workspace_size = CublasHandle::UpstreamWorkspace((100 * major) + (10 * minor));
  auto workspace = (*device)->Allocate(memory, workspace_size.value());
  if (!workspace) {
    std::println(stderr, "{}", workspace.error());
    return 1;
  }
  int status = 0;
  std::vector<Result> results;
  {
    auto cublas = CublasHandle::Create(0, (*device)->execution(), (*device)->stream(),
                                       {.base = workspace->base, .size = workspace_size});
    if (!cublas) {
      std::println(stderr, "cuBLAS handle: {}", cublas.error().detail);
      return 1;
    }
    Context ctx{.device = **device,
                .cublas = **cublas,
                .memory = memory,
                .l2 = static_cast<std::uint64_t>(l2),
                .cu_device = cu_device};
    for (const Case* c : selected) {
      bool mismatch = false;
      auto result = RunCase(ctx, *c, mismatch);
      if (!result) {
        std::println(stderr, "{}", result.error());
        status = mismatch ? 3 : 1;
        break;
      }
      std::vector<double> sorted = result->graph_us;
      std::ranges::sort(sorted);
      std::println(stderr, "{} at {} rows: {} sets of {} bytes, graph median {:.3f} us", c->name,
                   c->rows, result->sets, result->set_bytes, sorted[sorted.size() / 2]);
      results.push_back(std::move(*result));
    }
    Report((*device)->Finish(), "the final fence");
  }
  (*device)->Free(*workspace);
  if (status != 0) {
    return status;
  }

  std::FILE* out =
      std::fopen(output_path.c_str(), "wx");  // NOLINT(cppcoreguidelines-owning-memory)
  if (out == nullptr) {
    std::println(stderr, "cannot create {} (it must not exist)", output_path);
    return 1;
  }
  std::println(out, "{{");
  std::println(out, " \"schema\": 1,");
  std::println(out, R"( "harness": "llmp_ggml_vmm_bench",)");
  std::println(out, R"( "memory": "{}",)", memory_name);
  std::println(out, " \"calibration_sha256\": {},",
               calibration.empty() ? std::string("null") : std::format("\"{}\"", calibration));
  std::println(out,
               " \"device\": {{\"name\": \"{}\", \"cc\": {}, \"sms\": {}, \"l2_bytes\": {}, "
               "\"driver_api\": {}, \"runtime_api\": {}}},",
               static_cast<const char*>(properties.name), (100 * major) + (10 * minor), sms, l2,
               driver, runtime);
  std::println(out,
               " \"protocol\": {{\"invocations_per_sample\": {}, \"warm_replays\": {}, "
               "\"samples\": {}, \"ring_bytes_over\": \"4 x L2\"}},",
               kInvocations, kWarmups, kSamples);
  std::println(out, R"( "set": "{}",)", file->set);
  std::println(out, " \"cases\": [");
  for (std::size_t i = 0; i < results.size(); ++i) {
    const Result& r = results[i];
    std::println(out,
                 "  {{\"name\": \"{}\", \"rows\": {}, \"set_bytes\": {}, \"sets\": {}, "
                 "\"scratch_bytes\": {}, \"graph_launches_verified\": {}, "
                 "\"eager_output_fnv1a64\": \"{:016x}\", \"final_output_fnv1a64\": \"{:016x}\", "
                 "\"graph_replay_us\": {}, \"stream_us\": {}}}{}",
                 r.name, r.rows, r.set_bytes, r.sets, r.scratch, r.launches, r.eager_hash,
                 r.final_hash, Samples(r.graph_us), Samples(r.stream_us),
                 i + 1 == results.size() ? "" : ",");
  }
  std::println(out, " ]");
  std::println(out, "}}");
  const bool written = std::fclose(out) == 0;  // NOLINT(cppcoreguidelines-owning-memory)
  return written ? 0 : 1;
}
