// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Llmpalooza's EXL3 launchers on a GB10 (label `gpu`; all but the pinned
// reconstruction GEMM's tests also on a discrete GPU the build targets,
// `gpu-discrete`, D-082; docs/backend-proof.md, P3), on a synthetic linear
// shaped like Qwen2.5-0.5B's q_proj (896 × 896, K = 4, mcg; random trellis
// words and ±1 side vectors, as P0's synthetic kernel cases):
// - every path of the linear (packed through the GEMM and the GEMV, the
//   fused gate/up multi-GEMM, the reconstruction path and its fused form)
//   gives identical bits with its operands in cudaMalloc memory and in
//   device VMM (BP-N3), and the paths agree with each other to within F16
//   rounding;
// - the over-read probe: with every operand ending flush against an
//   unmapped VMM granule, and then starting flush after one, every path
//   runs without a fault and gives the same bits, so no kernel reads or
//   writes outside its operands;
// - what does not fit is refused before launch: a cooperative grid larger
//   than the device holds at once, a lock area another live context uses,
//   and anything after a fault, which returns as a fault;
// - the registry binds each implementation to its own path only, and a
//   stale or foreign declaration binds nothing (BP-S2, BP-S4);
// - the over-read probe at every linear shape and rate of both fixtures,
//   with random weights: the GEMV in both configurations at one to eight
//   rows, the GEMM at every tile shape the shape takes (and at tile shape 4
//   on a synthetic 896 × 1,024, which no fixture shape takes) and the
//   multi-GEMM at the gate/up shape;
// - two contexts on two streams, each launching cooperative grids at the
//   device's co-resident limit, both complete: cooperative grids from
//   different streams do not deadlock each other;
// - a reconstruction GEMM runs only its own pin, and the multi-GEMM refuses
//   tables written for tensors that have moved (BP-P5).
// The per-linear exactness against upstream is the sweep's
// (docs/experiments/backend-proof-p3/), not this test's.

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <expected>
#include <format>
#include <map>
#include <memory>
#include <print>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include "base/bytes.h"
#include "execution/registry.h"
#include "expected_error.h"
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
using llmp::base::Bytes;
using llmp::providers::Access;
using llmp::providers::BackingKind;
using llmp::providers::DeviceExecution;
using llmp::providers::FenceState;
using llmp::providers::StreamId;
using llmp::test_support::FailedCode;

constexpr int kK = 896;
constexpr int kN = 896;
constexpr int kBits = 4;

// exl3-recon-pin.json's algorithms for 896 × 896 at 145 rows with an F32
// output (HSS) and at 1,024 rows with an F16 output (HSH), each with the
// GEMM it was pinned for.
constexpr exl3::LtAlgorithm kPin145{.config = {67, 316, 1, 0, 0, 66, 35, 0, 0},
                                    .m = 145,
                                    .k = kK,
                                    .n = kN,
                                    .ldc = kN,
                                    .output = Output::kF32};
constexpr exl3::LtAlgorithm kPin1024{.config = {67, 409, 1, 0, 0, 30, 35, 0, 0},
                                     .m = 1024,
                                     .k = kK,
                                     .n = kN,
                                     .ldc = kN,
                                     .output = Output::kF16};

std::uint64_t Mix(std::uint64_t x) {
  x += 0x9E3779B97F4A7C15ULL;
  x = (x ^ (x >> 30U)) * 0xBF58476D1CE4E5B9ULL;
  x = (x ^ (x >> 27U)) * 0x94D049BB133111EBULL;
  return x ^ (x >> 31U);
}

std::vector<__half> Halves(std::size_t count, std::uint64_t seed, float scale) {
  std::vector<__half> values(count);
  for (std::size_t i = 0; i < count; ++i) {
    const auto bits = static_cast<std::int64_t>(Mix(seed + i) >> 53U);
    values[i] = __float2half_rn(static_cast<float>(bits - 1024) / 1024.0F * scale);
  }
  return values;
}

std::vector<__half> Signs(std::size_t count, std::uint64_t seed, float scale) {
  std::vector<__half> values(count);
  for (std::size_t i = 0; i < count; ++i) {
    values[i] = __float2half_rn((Mix(seed + i) & 1U) != 0 ? scale : -scale);
  }
  return values;
}

std::vector<std::uint16_t> Words(std::size_t count, std::uint64_t seed) {
  std::vector<std::uint16_t> values(count);
  for (std::size_t i = 0; i < count; ++i) {
    values[i] = static_cast<std::uint16_t>(Mix(seed + i) >> 48U);
  }
  return values;
}

// Where operands go: cudaMalloc, device VMM, or device VMM with each operand
// flush against an unmapped granule, after it or before it.
enum class Placement : std::uint8_t { kCudaMalloc, kDeviceVmm, kFlushEnd, kFlushStart };

// The outputs of every path, as bytes.
struct Results {
  std::vector<std::byte> gemm;            // 16 rows, F16, with bias
  std::vector<std::byte> gemv;            // 1 row, F16, with bias
  std::vector<std::byte> gemm1;           // the same row through the GEMM
  std::vector<std::byte> multi;           // 8 rows, F32, two outputs
  std::vector<std::byte> recon;           // 145 rows, F32
  std::vector<std::byte> fused;           // 1,024 rows, F16, with bias
  std::vector<std::byte> fused_as_recon;  // the same 1,024 rows unfused (reported, not pinned)
};

class Exl3LinearTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    memory_ = std::move(llmp::providers::cuda::OpenDeviceMemory(0).value());
    execution_ = std::move(llmp::providers::cuda::OpenDeviceExecution(0).value());
    stream_ = execution_->CreateStream().value();
    for (std::size_t i = 0; i < memory_->Classes().size(); ++i) {
      if (memory_->Classes()[i].kind == BackingKind::kDevice) {
        device_class_ = i;
      }
    }
    granule_ = memory_->Granularity().value();
  }

  void TearDown() override {
    Finish();
    ASSERT_TRUE(execution_->DestroyStream(stream_).has_value());
    FreeAll();
  }

  void FreeAll() {
    for (void* pointer : malloced_) {
      EXPECT_EQ(cudaFree(pointer), cudaSuccess);
    }
    malloced_.clear();
    for (const Mapped& m : mapped_) {
      ASSERT_TRUE(memory_->Unmap(m.reservation, Bytes(m.offset), Bytes(m.size)).has_value());
      ASSERT_TRUE(memory_->Release(m.backing).has_value());
      ASSERT_TRUE(memory_->Free(m.reservation).has_value());
    }
    mapped_.clear();
  }

  // `bytes` at an address aligned to 256 bytes (cudaMalloc, device VMM),
  // or, flush, ending exactly at the end of its mapping (the next granule
  // unmapped) or starting exactly at its start (the one before unmapped).
  std::uint64_t Allocate(Placement placement, std::uint64_t bytes) {
    if (placement == Placement::kCudaMalloc) {
      void* pointer = nullptr;
      EXPECT_EQ(cudaMalloc(&pointer, bytes), cudaSuccess);
      malloced_.push_back(pointer);
      return reinterpret_cast<std::uint64_t>(pointer);
    }
    const std::uint64_t mapped = (bytes + granule_ - 1) / granule_ * granule_;
    const bool guarded = placement == Placement::kFlushEnd || placement == Placement::kFlushStart;
    const auto reservation = memory_->Reserve(Bytes(mapped + (guarded ? granule_ : 0))).value();
    const auto backing = memory_->Create(device_class_, Bytes(mapped)).value();
    const std::uint64_t offset = placement == Placement::kFlushStart ? granule_ : 0;
    EXPECT_TRUE(memory_->Map(reservation, Bytes(offset), backing).has_value());
    EXPECT_TRUE(memory_->SetAccess(reservation, Bytes(offset), Bytes(mapped), Access::kReadWrite)
                    .has_value());
    mapped_.push_back({reservation, backing, offset, mapped});
    const std::uint64_t base = memory_->RangeOf(reservation).value().base + offset;
    return placement == Placement::kFlushEnd ? base + mapped - bytes : base;
  }

  void Finish() { Finish(stream_); }

  void Finish(StreamId stream) {
    const auto fence = execution_->Record(stream).value();
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    FenceState state = FenceState::kPending;
    while ((state = execution_->Query(fence).value()) == FenceState::kPending &&
           std::chrono::steady_clock::now() < deadline) {
      std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
    ASSERT_EQ(state, FenceState::kComplete);
    ASSERT_TRUE(execution_->Release(fence).has_value());
  }

  // Waits for the device, since a pageable copy may return before its DMA
  // lands and runs on the legacy stream, not the provider's.
  template <typename T>
  std::uint64_t Upload(Placement placement, const std::vector<T>& values) {
    const std::uint64_t bytes = values.size() * sizeof(T);
    const std::uint64_t address = Allocate(placement, bytes);
    EXPECT_EQ(cudaMemcpy(reinterpret_cast<void*>(address), values.data(), bytes,  // NOLINT
                         cudaMemcpyHostToDevice),
              cudaSuccess);
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    return address;
  }

  std::vector<std::byte> Download(std::uint64_t address, std::uint64_t bytes) {
    Finish();
    std::vector<std::byte> host(bytes);
    EXPECT_EQ(cudaMemcpy(host.data(), reinterpret_cast<const void*>(address), bytes,  // NOLINT
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    return host;
  }

  std::unique_ptr<exl3::LaunchContext> Context(Placement placement) {
    const std::uint64_t locks = Allocate(
        placement == Placement::kCudaMalloc ? Placement::kCudaMalloc : Placement::kDeviceVmm,
        exl3::kLockBytes);
    auto launch = exl3::LaunchContext::Create(0, *execution_, stream_, locks);
    EXPECT_TRUE(launch.has_value()) << (launch ? "" : launch.error().detail);
    return launch ? std::move(*launch) : nullptr;
  }

  exl3::Weights Linear(Placement placement, std::uint64_t seed, int k = kK, int n = kN,
                       int bits = kBits) {
    exl3::Weights w{.trellis = 0, .suh = 0, .svh = 0, .k = k, .n = n, .bits = bits};
    w.trellis = Upload(placement, Words(exl3::TrellisBytes(w) / 2, seed));
    w.suh = Upload(placement, Signs(static_cast<std::size_t>(k), seed + 1,
                                    1.0F / std::sqrt(static_cast<float>(k))));
    w.svh = Upload(placement, Signs(static_cast<std::size_t>(n), seed + 2, 1.0F));
    return w;
  }

  static std::uint64_t Bytes16(int rows, int columns) {
    return static_cast<std::uint64_t>(rows) * static_cast<std::uint64_t>(columns) * 2;
  }

  // Runs every path with every operand placed as asked.
  Results Run(Placement placement) {
    Results r;
    auto launch = Context(placement);
    auto gemm = exl3::ReconGemm::Create();
    EXPECT_TRUE(gemm.has_value()) << (gemm ? "" : gemm.error().detail);
    if (!launch || !gemm) {
      return r;
    }
    const exl3::Weights w = Linear(placement, 100);
    const std::uint64_t bias = Upload(placement, Halves(kN, 7, 0.5F));
    auto check = [](const std::expected<void, exl3::KernelFailure>& result) {
      EXPECT_TRUE(result.has_value()) << (result ? "" : result.error().detail);
    };

    // Packed through the GEMM: 16 rows at q_proj's tuned plan.
    {
      const std::uint64_t x = Upload(placement, Halves(Bytes16(16, kK) / 2, 1, 1.0F));
      const exl3::LinearOperands o{.weights = w,
                                   .x = x,
                                   .a_had = Allocate(placement, Bytes16(16, kK)),
                                   .y = Allocate(placement, Bytes16(16, kN)),
                                   .output = Output::kF16,
                                   .m = 16};
      check(exl3::PackedGemmLinear(*launch, o, {.shape = 2, .blocks = 14}, bias));
      r.gemm = Download(o.y, Bytes16(16, kN));
    }
    // One row through the GEMV, at upstream's choice, and through the GEMM.
    {
      const std::uint64_t x = Upload(placement, Halves(kK, 2, 1.0F));
      exl3::LinearOperands o{.weights = w,
                             .x = x,
                             .a_had = Allocate(placement, Bytes16(1, kK)),
                             .y = Allocate(placement, Bytes16(1, kN)),
                             .output = Output::kF16,
                             .m = 1};
      const auto plan = launch->UpstreamGemv(w, Output::kF16, 1);
      EXPECT_TRUE(plan.has_value() && plan->has_value());
      if (plan && *plan) {
        EXPECT_EQ((*plan)->config, 0);
        check(exl3::PackedGemvLinear(*launch, o, **plan, bias));
      }
      r.gemv = Download(o.y, Bytes16(1, kN));
      o.y = Allocate(placement, Bytes16(1, kN));
      check(exl3::PackedGemmLinear(*launch, o, {.shape = 2, .blocks = 20}, bias));
      r.gemm1 = Download(o.y, Bytes16(1, kN));
    }
    // The fused gate/up multi-GEMM: two linears of one input, F32 outputs.
    {
      const exl3::Weights second = Linear(placement, 200);
      const auto written = exl3::MultiGemmTables(w, second);
      const std::uint64_t tables =
          Upload(placement, std::vector<std::uint64_t>(written.begin(), written.end()));
      const std::uint64_t x = Upload(placement, Halves(Bytes16(8, kK) / 2, 3, 1.0F));
      const exl3::MultiLinearOperands o{.first = w,
                                        .second = second,
                                        .trellis_table = tables,
                                        .suh_table = tables + 16,
                                        .svh_table = tables + 32,
                                        .written = written,
                                        .x = x,
                                        .a_had = Allocate(placement, 2 * Bytes16(8, kK)),
                                        .y = Allocate(placement, 4 * Bytes16(8, kN)),
                                        .output = Output::kF32,
                                        .m = 8};
      check(exl3::MultiLinear(*launch, o, {.shape = 2, .blocks = 14, .concurrency = 2}));
      r.multi = Download(o.y, 4 * Bytes16(8, kN));
    }
    // The reconstruction path at 145 rows (F32 output) and the fused one at
    // 1,024 rows (F16 output, with bias), each with its pinned algorithm.
    {
      const std::uint64_t x = Upload(placement, Halves(Bytes16(145, kK) / 2, 4, 1.0F));
      const exl3::ReconstructedOperands o{
          .weights = w,
          .x = x,
          .xh = Allocate(placement, Bytes16(145, kK)),
          .w = Allocate(placement, exl3::ReconstructScratchBytes(w)),
          .y = Allocate(placement, 2 * Bytes16(145, kN)),
          .output = Output::kF32,
          .m = 145,
          .bias = 0};
      const std::vector<exl3::LtAlgorithm> pins{kPin145};
      check(exl3::ReconstructedLinear(*launch, **gemm, o, false, pins));
      r.recon = Download(o.y, 2 * Bytes16(145, kN));
    }
    {
      const std::uint64_t x = Upload(placement, Halves(Bytes16(1024, kK) / 2, 5, 1.0F));
      exl3::ReconstructedOperands o{.weights = w,
                                    .x = x,
                                    .xh = Allocate(placement, Bytes16(1024, kK)),
                                    .w = Allocate(placement, exl3::ReconstructScratchBytes(w)),
                                    .y = Allocate(placement, Bytes16(1024, kN)),
                                    .output = Output::kF16,
                                    .m = 1024,
                                    .bias = bias};
      const std::vector<exl3::LtAlgorithm> pins{kPin1024};
      check(exl3::ReconstructedLinear(*launch, **gemm, o, true, pins));
      r.fused = Download(o.y, Bytes16(1024, kN));
      o.y = Allocate(placement, Bytes16(1024, kN));
      check(exl3::ReconstructedLinear(*launch, **gemm, o, false, pins));
      r.fused_as_recon = Download(o.y, Bytes16(1024, kN));
    }
    Finish();
    EXPECT_FALSE(launch->faulted());
    return r;
  }

  // The over-read probe at the fixtures' shapes: every packed launch the
  // shape and rate take, each result hashed by its case. Operands are
  // placed afresh for each shape and row count, so under kFlushEnd or
  // kFlushStart each launch's x, a_had and y sit flush against an unmapped
  // granule, as its weights do.
  std::map<std::string, std::uint64_t> Probe(Placement placement) {
    std::map<std::string, std::uint64_t> hashes;
    auto check = [](const std::expected<void, exl3::KernelFailure>& result, const std::string& id) {
      EXPECT_TRUE(result.has_value()) << id << ": " << (result ? "" : result.error().detail);
      return result.has_value();
    };
    for (const ProbeShape& shape : kProbeShapes) {
      auto launch = Context(placement);
      if (!launch) {
        return hashes;
      }
      const std::uint64_t seed = (static_cast<std::uint64_t>(shape.k) * 1000003U) +
                                 (static_cast<std::uint64_t>(shape.n) * 101U) +
                                 static_cast<std::uint64_t>(shape.bits);
      const exl3::Weights w = Linear(placement, seed, shape.k, shape.n, shape.bits);
      const bool gate_up = shape.k == 896 && shape.n == 4864;
      const exl3::Weights second =
          gate_up ? Linear(placement, seed + 7, shape.k, shape.n, shape.bits) : exl3::Weights{};
      std::uint64_t tables = 0;
      if (gate_up) {
        const auto written = exl3::MultiGemmTables(w, second);
        tables = Upload(placement, std::vector<std::uint64_t>(written.begin(), written.end()));
      }
      for (const Output output : {Output::kF16, Output::kF32}) {
        const int out_bytes = exl3::OutputBytes(output);
        for (int m = 1; m <= 16; ++m) {
          const bool gemv_rows = m <= exl3::kGemvMaxRows && shape.bits == 4;
          const bool gemm_rows = m == 1 || m == 3 || m == 8 || m == 16;
          if (!gemv_rows && !gemm_rows) {
            continue;
          }
          const std::uint64_t x = Upload(
              placement, Halves(static_cast<std::size_t>(m) * static_cast<std::size_t>(shape.k),
                                seed + static_cast<std::uint64_t>(m), 1.0F));
          const std::uint64_t y_bytes = static_cast<std::uint64_t>(m) *
                                        static_cast<std::uint64_t>(shape.n) *
                                        static_cast<std::uint64_t>(out_bytes);
          const exl3::LinearOperands o{.weights = w,
                                       .x = x,
                                       .a_had = Allocate(placement, Bytes16(m, shape.k)),
                                       .y = Allocate(placement, y_bytes),
                                       .output = output,
                                       .m = m};
          const std::string at = std::format("{}x{} K{} {} m{}", shape.k, shape.n, shape.bits,
                                             output == Output::kF32 ? "F32" : "F16", m);
          if (gemv_rows) {
            for (const int config : {0, 1}) {
              const auto coresident = launch->GemvCoresident(shape.bits, output, m, config);
              if (!coresident) {
                ADD_FAILURE() << at << ": " << coresident.error().detail;
                return hashes;
              }
              const int blocks = std::min(shape.n / exl3::GemvColumns(config), *coresident);
              const std::string id = std::format("{} gemv {} {}", at, config, blocks);
              if (check(launch->Gemv(o, {.config = config, .blocks = blocks}), id)) {
                hashes[id] = Hash(Download(o.y, y_bytes));
              }
            }
          }
          if (!gemm_rows) {
            continue;
          }
          for (int tile = 1; tile <= exl3::kShapes; ++tile) {
            const auto index = static_cast<std::size_t>(tile);
            if (shape.k % exl3::kTileK.at(index) != 0 || shape.n % exl3::kTileN.at(index) != 0) {
              continue;
            }
            const int slices =
                (shape.k / exl3::kTileK.at(index)) * (shape.n / exl3::kTileN.at(index));
            const auto coresident = launch->GemmCoresident(shape.bits, tile, output);
            if (!coresident) {
              ADD_FAILURE() << at << ": " << coresident.error().detail;
              return hashes;
            }
            for (const int blocks : {std::min(slices, *coresident), std::min(slices, 7)}) {
              const std::string id = std::format("{} gemm {} {}", at, tile, blocks);
              if (check(launch->Gemm(o, {.shape = tile, .blocks = blocks}), id)) {
                hashes[id] = Hash(Download(o.y, y_bytes));
              }
            }
            if (!gate_up || m == 3) {
              continue;
            }
            const exl3::MultiLinearOperands mo{
                .first = w,
                .second = second,
                .trellis_table = tables,
                .suh_table = tables + 16,
                .svh_table = tables + 32,
                .written = exl3::MultiGemmTables(w, second),
                .x = x,
                .a_had = Allocate(placement, 2 * Bytes16(m, shape.k)),
                .y = Allocate(placement, 2 * y_bytes),
                .output = output,
                .m = m};
            const auto multi = launch->MultiGemmCoresident(shape.bits, tile, output);
            if (!multi) {
              ADD_FAILURE() << at << ": " << multi.error().detail;
              return hashes;
            }
            for (const int concurrency : {1, 2}) {
              const int blocks = std::min(slices, *multi / concurrency);
              const std::string id =
                  std::format("{} multi {} {} {}", at, tile, blocks, concurrency);
              if (check(launch->MultiGemm(
                            mo, {.shape = tile, .blocks = blocks, .concurrency = concurrency}),
                        id)) {
                hashes[id] = Hash(Download(mo.y, 2 * y_bytes));
              }
            }
          }
        }
      }
      Finish();
      EXPECT_FALSE(launch->faulted()) << shape.k << "x" << shape.n;
      launch.reset();
      FreeAll();
    }
    return hashes;
  }

  static std::uint64_t Hash(const std::vector<std::byte>& bytes) {
    std::uint64_t hash = 0xCBF29CE484222325ULL;  // FNV-1a
    for (const std::byte b : bytes) {
      hash = (hash ^ static_cast<std::uint64_t>(b)) * 0x100000001B3ULL;
    }
    return hash;
  }

  // Each linear shape and rate of the two fixtures (results.json's
  // weights: q_proj and o_proj, k_proj and v_proj, gate and up, down, and
  // the head), and a synthetic 896 × 1,024, the smallest q_proj-wide shape
  // that tile shape 4 (n a multiple of 512) takes.
  struct ProbeShape {
    int k;
    int n;
    int bits;
  };
  static constexpr std::array<ProbeShape, 12> kProbeShapes{{{896, 128, 4},
                                                            {896, 896, 4},
                                                            {896, 4864, 4},
                                                            {4864, 896, 4},
                                                            {896, 151936, 8},
                                                            {896, 128, 5},
                                                            {896, 128, 6},
                                                            {896, 896, 5},
                                                            {896, 896, 6},
                                                            {896, 4864, 5},
                                                            {4864, 896, 5},
                                                            {896, 1024, 4}}};

  std::unique_ptr<llmp::providers::VmmProvider> memory_;
  std::unique_ptr<DeviceExecution> execution_;
  StreamId stream_;
  std::size_t device_class_ = 0;
  std::uint64_t granule_ = 0;
  std::vector<void*> malloced_;
  struct Mapped {
    llmp::providers::ReservationId reservation;
    llmp::providers::BackingId backing;
    std::uint64_t offset;
    std::uint64_t size;
  };
  std::vector<Mapped> mapped_;
};

// Values of F16 or F32 bytes, for closeness checks.
std::vector<float> Floats(const std::vector<std::byte>& bytes, bool fp32) {
  std::vector<float> values;
  if (fp32) {
    values.resize(bytes.size() / 4);
    std::memcpy(values.data(), bytes.data(), bytes.size());
  } else {
    std::vector<__half> halves(bytes.size() / 2);
    std::memcpy(halves.data(), bytes.data(), bytes.size());
    for (const __half h : halves) {
      values.push_back(__half2float(h));
    }
  }
  return values;
}

// Root-mean-square difference relative to the reference's RMS.
double RelativeRms(const std::vector<float>& a, const std::vector<float>& reference) {
  double error = 0;
  double norm = 0;
  for (std::size_t i = 0; i < a.size() && i < reference.size(); ++i) {
    const double d = static_cast<double>(a[i]) - static_cast<double>(reference[i]);
    error += d * d;
    norm += static_cast<double>(reference[i]) * static_cast<double>(reference[i]);
  }
  return norm == 0 ? 1.0 : std::sqrt(error / norm);
}

TEST_F(Exl3LinearTest, EveryPathIsExactAcrossMemoryKindsAndThePathsAgree) {
  const Results malloced = Run(Placement::kCudaMalloc);
  const Results vmm = Run(Placement::kDeviceVmm);
  ASSERT_FALSE(malloced.gemm.empty());
  EXPECT_EQ(malloced.gemm, vmm.gemm);
  EXPECT_EQ(malloced.gemv, vmm.gemv);
  EXPECT_EQ(malloced.gemm1, vmm.gemm1);
  EXPECT_EQ(malloced.multi, vmm.multi);
  EXPECT_EQ(malloced.recon, vmm.recon);
  EXPECT_EQ(malloced.fused, vmm.fused);
  EXPECT_EQ(malloced.fused_as_recon, vmm.fused_as_recon);
  // Different kernels for the same product: close, not equal.
  const double gemv = RelativeRms(Floats(malloced.gemv, false), Floats(malloced.gemm1, false));
  const double fused =
      RelativeRms(Floats(malloced.fused, false), Floats(malloced.fused_as_recon, false));
  EXPECT_LT(gemv, 1e-2);
  EXPECT_LT(fused, 1e-2);
  std::println(
      "GEMV against GEMM at one row: relative RMS {:.3g}; fused against unfused "
      "reconstruction at 1,024 rows: {:.3g}",
      gemv, fused);
}

TEST_F(Exl3LinearTest, TheRegistryBindsEachPathToItsOwnCalls) {
  const auto declared = exl3::Implementations();
  ASSERT_EQ(declared.size(), 6U);
  auto registry = llmp::execution::Registry::Create(declared);
  ASSERT_TRUE(registry.has_value());
  for (const auto& implementation : declared) {
    auto kernel = exl3::Kernel::Bind(implementation);
    ASSERT_TRUE(kernel.has_value()) << implementation.name;
    EXPECT_EQ(kernel->name(), implementation.name);
    EXPECT_EQ(kernel->operation(), implementation.operation);
    EXPECT_EQ(implementation.source, "exllamav3");
    EXPECT_EQ(implementation.revision.size(), 64U);
  }
  auto stale = declared[0];
  stale.variant += " (changed)";
  EXPECT_FALSE(exl3::Kernel::Bind(stale).has_value());
  auto foreign = declared[0];
  foreign.name = "exl3.linear.unknown";
  EXPECT_FALSE(exl3::Kernel::Bind(foreign).has_value());

  // A GEMV implementation refuses a GEMM plan, before anything is queued.
  auto launch = Context(Placement::kCudaMalloc);
  ASSERT_NE(launch, nullptr);
  const exl3::Kernel gemv = exl3::Kernel::Bind(declared[1]).value();
  ASSERT_EQ(gemv.name(), "exl3.linear.gemv");
  const exl3::LinearOperands o{.weights = Linear(Placement::kCudaMalloc, 1),
                               .x = Allocate(Placement::kCudaMalloc, Bytes16(1, kK)),
                               .a_had = Allocate(Placement::kCudaMalloc, Bytes16(1, kK)),
                               .y = Allocate(Placement::kCudaMalloc, Bytes16(1, kN)),
                               .output = Output::kF16,
                               .m = 1};
  const auto refused = gemv.Run(*launch, o, exl3::GemmPlan{.shape = 2, .blocks = 14}, 0);
  ASSERT_FALSE(refused.has_value());
  EXPECT_EQ(refused.error().error, exl3::KernelError::kRejected);
  EXPECT_FALSE(launch->faulted());
}

TEST_F(Exl3LinearTest, RefusesGridsThatAreNotCoresident) {
  auto launch = Context(Placement::kCudaMalloc);
  ASSERT_NE(launch, nullptr);
  const auto coresident = launch->GemmCoresident(kBits, 2, Output::kF16);
  ASSERT_TRUE(coresident.has_value());
  EXPECT_GE(*coresident, launch->sm_count());
  const exl3::LinearOperands o{.weights = Linear(Placement::kCudaMalloc, 1),
                               .x = Allocate(Placement::kCudaMalloc, Bytes16(16, kK)),
                               .a_had = Allocate(Placement::kCudaMalloc, Bytes16(16, kK)),
                               .y = Allocate(Placement::kCudaMalloc, Bytes16(16, kN)),
                               .output = Output::kF16,
                               .m = 16};
  // q_proj at shape 2 has 196 split-K slices, more than fit at once.
  const auto refused = launch->Gemm(o, {.shape = 2, .blocks = *coresident + 1});
  ASSERT_FALSE(refused.has_value());
  EXPECT_EQ(refused.error().error, exl3::KernelError::kRejected);
  EXPECT_TRUE(launch->Gemm(o, {.shape = 2, .blocks = *coresident}).has_value());
  const auto multi = launch->MultiGemmCoresident(kBits, 2, Output::kF32);
  ASSERT_TRUE(multi.has_value());
  EXPECT_EQ(*multi, *coresident);
  Finish();
  EXPECT_FALSE(launch->faulted());
}

TEST_F(Exl3LinearTest, NoTwoLiveContextsShareLockSlots) {
  const std::uint64_t locks = Allocate(Placement::kCudaMalloc, exl3::kLockBytes + 4096);
  auto first = exl3::LaunchContext::Create(0, *execution_, stream_, locks);
  ASSERT_TRUE(first.has_value());
  EXPECT_FALSE(exl3::LaunchContext::Create(0, *execution_, stream_, locks).has_value());
  EXPECT_FALSE(exl3::LaunchContext::Create(0, *execution_, stream_, locks + 4096).has_value());
  EXPECT_FALSE(exl3::LaunchContext::Create(0, *execution_, stream_, locks + 8).has_value());
  first->reset();
  Finish();
  auto again = exl3::LaunchContext::Create(0, *execution_, stream_, locks);
  EXPECT_TRUE(again.has_value());
}

// Forwards to the CUDA provider, and reports its next Submission as an
// unknown outcome: a device fault.
class FaultingExecution final : public DeviceExecution {
 public:
  explicit FaultingExecution(DeviceExecution& inner) : inner_(inner) {}
  bool fault_next = false;

  std::expected<StreamId, llmp::providers::Failure> CreateStream() override {
    return inner_.CreateStream();
  }
  std::expected<void, llmp::providers::Failure> DestroyStream(StreamId stream) override {
    return inner_.DestroyStream(stream);
  }
  std::expected<void, llmp::providers::Failure> Copy(StreamId stream, std::uint64_t destination,
                                                     std::uint64_t source, Bytes size) override {
    return inner_.Copy(stream, destination, source, size);
  }
  std::expected<void, llmp::providers::Failure> Zero(StreamId stream, std::uint64_t destination,
                                                     Bytes size) override {
    return inner_.Zero(stream, destination, size);
  }
  std::expected<llmp::providers::NativeStream, llmp::providers::Failure> Submission(
      StreamId stream) override {
    if (std::exchange(fault_next, false)) {
      return std::unexpected(llmp::providers::Failure{
          .error = llmp::providers::ProviderError::kUnknown, .detail = "a scripted fault"});
    }
    return inner_.Submission(stream);
  }
  std::expected<void, llmp::providers::Failure> Wait(StreamId stream,
                                                     llmp::providers::FenceId fence) override {
    return inner_.Wait(stream, fence);
  }
  std::expected<llmp::providers::FenceId, llmp::providers::Failure> Record(
      StreamId stream) override {
    return inner_.Record(stream);
  }
  std::expected<FenceState, llmp::providers::Failure> Query(
      llmp::providers::FenceId fence) override {
    return inner_.Query(fence);
  }
  std::expected<void, llmp::providers::Failure> Release(llmp::providers::FenceId fence) override {
    return inner_.Release(fence);
  }

 private:
  DeviceExecution& inner_;
};

TEST_F(Exl3LinearTest, AFaultReturnsAsAFaultAndStopsTheContext) {
  FaultingExecution faulting(*execution_);
  const std::uint64_t locks = Allocate(Placement::kCudaMalloc, exl3::kLockBytes);
  auto launch = exl3::LaunchContext::Create(0, faulting, stream_, locks);
  ASSERT_TRUE(launch.has_value());
  const exl3::LinearOperands o{.weights = Linear(Placement::kCudaMalloc, 1),
                               .x = Allocate(Placement::kCudaMalloc, Bytes16(1, kK)),
                               .a_had = Allocate(Placement::kCudaMalloc, Bytes16(1, kK)),
                               .y = Allocate(Placement::kCudaMalloc, Bytes16(1, kN)),
                               .output = Output::kF16,
                               .m = 1};
  faulting.fault_next = true;
  const auto faulted = (*launch)->Gemm(o, {.shape = 2, .blocks = 14});
  ASSERT_FALSE(faulted.has_value());
  EXPECT_EQ(faulted.error().error, exl3::KernelError::kUnknown);
  EXPECT_TRUE((*launch)->faulted());
  // Nothing more runs on the context: its lock slots are undetermined.
  const auto refused = (*launch)->Gemm(o, {.shape = 2, .blocks = 14});
  ASSERT_FALSE(refused.has_value());
  EXPECT_EQ(refused.error().error, exl3::KernelError::kRejected);
}

// The over-read probe (docs/artifact-format.md leaves EXL3's open): every
// operand ends flush against an unmapped granule, then starts flush after
// one. A read or write past either edge would fault; none does, and the
// results are the cudaMalloc run's bit for bit.
TEST_F(Exl3LinearTest, NoKernelReadsOrWritesOutsideItsOperands) {
  const Results malloced = Run(Placement::kCudaMalloc);
  const Results end = Run(Placement::kFlushEnd);
  EXPECT_EQ(cudaGetLastError(), cudaSuccess);
  EXPECT_EQ(malloced.gemm, end.gemm);
  EXPECT_EQ(malloced.gemv, end.gemv);
  EXPECT_EQ(malloced.gemm1, end.gemm1);
  EXPECT_EQ(malloced.multi, end.multi);
  EXPECT_EQ(malloced.recon, end.recon);
  EXPECT_EQ(malloced.fused, end.fused);
  EXPECT_EQ(malloced.fused_as_recon, end.fused_as_recon);
  FreeAll();
  const Results start = Run(Placement::kFlushStart);
  EXPECT_EQ(cudaGetLastError(), cudaSuccess);
  EXPECT_EQ(malloced.gemm, start.gemm);
  EXPECT_EQ(malloced.gemv, start.gemv);
  EXPECT_EQ(malloced.gemm1, start.gemm1);
  EXPECT_EQ(malloced.multi, start.multi);
  EXPECT_EQ(malloced.recon, start.recon);
  EXPECT_EQ(malloced.fused, start.fused);
  EXPECT_EQ(malloced.fused_as_recon, start.fused_as_recon);
}

}  // namespace

namespace {

// The over-read probe at the fixtures' shapes (see Probe): the kernels and
// variants the sweep never launched there (the GEMV's wide configuration,
// its row-guarded mode at two to seven rows, the GEMM at tile shapes 3 and
// 4, the multi-GEMM at tile shapes 1 and 4, grids below the co-resident
// limit) stay inside their operands, and give the cudaMalloc run's bits.
TEST_F(Exl3LinearTest, NoPackedKernelReadsOutsideItsOperandsAtTheFixturesShapes) {
  const auto malloced = Probe(Placement::kCudaMalloc);
  ASSERT_FALSE(malloced.empty());
  std::size_t gemv_mode1 = 0;
  std::size_t tile4 = 0;
  for (const auto& [id, hash] : malloced) {
    gemv_mode1 += id.contains(" gemv ") && !id.contains(" m1 ") ? 1 : 0;
    tile4 += id.contains(" gemm 4 ") ? 1 : 0;
  }
  EXPECT_GT(gemv_mode1, 0U);
  EXPECT_GT(tile4, 0U);
  for (const Placement placement : {Placement::kFlushEnd, Placement::kFlushStart}) {
    const auto flush = Probe(placement);
    EXPECT_EQ(cudaGetLastError(), cudaSuccess);
    EXPECT_EQ(flush.size(), malloced.size());
    for (const auto& [id, hash] : malloced) {
      const auto found = flush.find(id);
      ASSERT_NE(found, flush.end()) << id;
      EXPECT_EQ(found->second, hash) << id;
    }
  }
  std::println("{} packed launches, {} of the GEMV at 2 to 8 rows and {} at tile shape 4",
               malloced.size(), gemv_mode1, tile4);
}

// Cooperative grids need every block co-resident; the limit each launch is
// held to is the whole device's. Two contexts on two streams, each
// launching grids at that limit back to back, must both complete: the
// device admits a cooperative grid whole, so neither can hold half the
// device while waiting on the other.
TEST_F(Exl3LinearTest, TwoContextsAtTheCoresidentLimitBothComplete) {
  // Gate/up-shaped linears at 16 rows, so that each launch runs long enough
  // for the two streams' launches to meet on the device.
  constexpr int kGateN = 4864;
  constexpr int kRows = 16;
  const StreamId other = execution_->CreateStream().value();
  const exl3::Weights w = Linear(Placement::kCudaMalloc, 1, kK, kGateN);
  const exl3::Weights second = Linear(Placement::kCudaMalloc, 2, kK, kGateN);
  const auto written = exl3::MultiGemmTables(w, second);
  const std::uint64_t tables =
      Upload(Placement::kCudaMalloc, std::vector<std::uint64_t>(written.begin(), written.end()));
  const std::uint64_t x = Upload(Placement::kCudaMalloc, Halves(Bytes16(kRows, kK) / 2, 3, 1.0F));
  struct Lane {
    std::unique_ptr<exl3::LaunchContext> launch;
    exl3::LinearOperands gemm;
    exl3::MultiLinearOperands multi;
  };
  std::array<Lane, 2> lanes;
  for (std::size_t i = 0; i < lanes.size(); ++i) {
    auto created = exl3::LaunchContext::Create(0, *execution_, i == 0 ? stream_ : other,
                                               Allocate(Placement::kCudaMalloc, exl3::kLockBytes));
    ASSERT_TRUE(created.has_value());
    lanes[i].launch = std::move(*created);
    lanes[i].gemm = {.weights = w,
                     .x = x,
                     .a_had = Allocate(Placement::kCudaMalloc, Bytes16(kRows, kK)),
                     .y = Allocate(Placement::kCudaMalloc, Bytes16(kRows, kGateN)),
                     .output = Output::kF16,
                     .m = kRows};
    lanes[i].multi = {.first = w,
                      .second = second,
                      .trellis_table = tables,
                      .suh_table = tables + 16,
                      .svh_table = tables + 32,
                      .written = written,
                      .x = x,
                      .a_had = Allocate(Placement::kCudaMalloc, 2 * Bytes16(kRows, kK)),
                      .y = Allocate(Placement::kCudaMalloc, 2 * Bytes16(kRows, kGateN)),
                      .output = Output::kF16,
                      .m = kRows};
  }
  exl3::LaunchContext& first = *lanes[0].launch;
  const int gemm_limit = first.GemmCoresident(kBits, 2, Output::kF16).value();
  const int multi_limit = first.MultiGemmCoresident(kBits, 2, Output::kF16).value();
  const int gemv_limit = first.GemvCoresident(kBits, Output::kF16, 8, 1).value();
  // 896 × 4,864 at tile shape 2 has 1,064 split-K slices and the GEMV's
  // wide configuration 76 column groups: each grid is the most the device
  // holds at once, or the most the problem takes.
  const exl3::GemmPlan gemm{.shape = 2, .blocks = gemm_limit};
  const exl3::MultiGemmPlan multi{.shape = 2, .blocks = multi_limit / 2, .concurrency = 2};
  const exl3::GemvPlan gemv{.config = 1, .blocks = std::min(gemv_limit, kGateN / 64)};
  ASSERT_LE(gemm.blocks, 1064);
  exl3::LinearOperands gemv_operands = lanes[0].gemm;
  constexpr int kRounds = 100;
  auto run = [&](std::span<Lane> running) {
    const auto started = std::chrono::steady_clock::now();
    for (int round = 0; round < kRounds; ++round) {
      for (Lane& lane : running) {
        gemv_operands.a_had = lane.gemm.a_had;
        gemv_operands.y = lane.gemm.y;
        gemv_operands.m = 8;
        EXPECT_TRUE(lane.launch->Gemm(lane.gemm, gemm).has_value());
        EXPECT_TRUE(lane.launch->MultiGemm(lane.multi, multi).has_value());
        EXPECT_TRUE(lane.launch->Gemv(gemv_operands, gemv).has_value());
      }
    }
    Finish(stream_);
    Finish(other);
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started)
        .count();
  };
  const double alone = run(std::span<Lane>(lanes).first(1));
  const double both = run(lanes);
  EXPECT_EQ(Download(lanes[0].gemm.y, Bytes16(8, kGateN)),
            Download(lanes[1].gemm.y, Bytes16(8, kGateN)));
  EXPECT_EQ(Download(lanes[0].multi.y, 2 * Bytes16(kRows, kGateN)),
            Download(lanes[1].multi.y, 2 * Bytes16(kRows, kGateN)));
  for (Lane& lane : lanes) {
    EXPECT_FALSE(lane.launch->faulted());
    lane.launch.reset();
  }
  EXPECT_TRUE(execution_->DestroyStream(other).has_value());
  std::println(
      "{} rounds of GEMM ({} blocks), multi-GEMM ({} x 2) and GEMV ({}): {:.1f} ms on one "
      "stream, {:.1f} ms on two",
      kRounds, gemm.blocks, multi.blocks, gemv.blocks, alone, both);
}

// A pin runs only the GEMM exl3-recon-pin.json recorded it for: the same
// attributes named for another row count, output or width are refused
// before anything is queued, and the context stays usable.
TEST_F(Exl3LinearTest, AReconstructionGemmRunsOnlyItsOwnPin) {
  auto launch = Context(Placement::kCudaMalloc);
  ASSERT_NE(launch, nullptr);
  auto gemm = exl3::ReconGemm::Create();
  ASSERT_TRUE(gemm.has_value());
  const exl3::ReconGemmOperands o{.w = Allocate(Placement::kCudaMalloc, Bytes16(kK, kN)),
                                  .x = Allocate(Placement::kCudaMalloc, Bytes16(145, kK)),
                                  .y = Allocate(Placement::kCudaMalloc, 2 * Bytes16(145, kN)),
                                  .m = 145,
                                  .k = kK,
                                  .n = kN,
                                  .ldc = kN,
                                  .output = Output::kF32};
  const int sms = launch->sm_count();
  EXPECT_TRUE((*gemm)->Check(o, kPin145, sms).has_value());
  // 1,024 rows' pin, a valid algorithm for this GEMM too, at 145 rows.
  exl3::LtAlgorithm other = kPin1024;
  other.output = Output::kF32;
  EXPECT_EQ(FailedCode((*gemm)->Check(o, other, sms)), exl3::KernelError::kRejected);
  EXPECT_EQ(FailedCode((*gemm)->Run(*launch, o, other, sms)), exl3::KernelError::kRejected);
  // 145 rows' pin at an unpinned 146 rows, and for the F16 output.
  exl3::ReconGemmOperands unpinned = o;
  unpinned.m = 146;
  EXPECT_EQ(FailedCode((*gemm)->Check(unpinned, kPin145, sms)), exl3::KernelError::kRejected);
  other = kPin145;
  other.output = Output::kF16;
  EXPECT_EQ(FailedCode((*gemm)->Check(o, other, sms)), exl3::KernelError::kRejected);
  // A slice of a wider output: its pin names the slice's width and the
  // output's row stride.
  other = kPin145;
  other.ldc = 2 * kN;
  EXPECT_EQ(FailedCode((*gemm)->Check(o, other, sms)), exl3::KernelError::kRejected);
  EXPECT_FALSE(launch->faulted());
  EXPECT_TRUE((*gemm)->Run(*launch, o, kPin145, sms).has_value());
  Finish();
  EXPECT_FALSE(launch->faulted());
}

// Each identity records whether the build checks libstdc++'s preconditions
// (D-083), as the GGML module's do.
TEST(Exl3ImplementationsTest, IdentitiesRecordTheLibraryAssertions) {
#ifdef _GLIBCXX_ASSERTIONS
  constexpr std::string_view kExpected = ", libstdc++ assertions)";
#else
  constexpr std::string_view kExpected = ", no libstdc++ assertions)";
#endif
  for (const auto& implementation : exl3::Implementations()) {
    EXPECT_NE(implementation.build.find(kExpected), std::string::npos)
        << implementation.name << ": " << implementation.build;
  }
}

}  // namespace
