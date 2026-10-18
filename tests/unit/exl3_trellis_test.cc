// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The EXL3 trellis through the format layer (D-108): the module's
// capabilities, ExLlamaV3's kernels against the CPU reference
// (kernels/reference/decode.h), and the Qwen2.5 EXL3 fixtures' products
// through the registry's candidates. Label `gpu` (and `models`: the
// fixtures' test reads their published v1 artifacts where the store holds
// them, on spark and spark-b); the selection and synthetic tests also run on a
// discrete GPU the build targets (`gpu-discrete`, D-082).
// - Candidates(): at decode rows (1 to 16) the GEMV (K = 4, up to 8 rows)
//   and the packed GEMM, never a reconstruction, whatever is measured; at
//   prefill rows the packed GEMM and, on the GB10, both reconstruction
//   paths where the device's tuning data pins the shape's GEMMs (none
//   otherwise); nothing for a rate, codebook, input encoding, shape or device
//   without kernels; the multi-GEMM for two trellises of one rate.
// - reconstruct_kernel's rotated weights equal the reference's bit for bit
//   at K = 4, 5, 6 and 8 on random trellises, and reconstruct_had_kernel's
//   original-basis weights agree with DecodeRows within its F16
//   arithmetic.
// - On both fixtures (4.0 and 4.5 bpw): every linear's rotated weights
//   bit for bit (the head's first and last 1,024 columns), layer 0's
//   original-basis weights through the fused kernel, and layer 0's q_proj,
//   gate_proj and down_proj products at 1, 8, 145 and 1,024 rows through
//   every candidate, and gate with up through the multi-GEMM, against an
//   F64 product of the reference's weights.

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <format>
#include <fstream>
#include <memory>
#include <optional>
#include <print>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include "artifact/artifact.h"
#include "execution/registry.h"
#include "format/encoding.h"
#include "format/layout.h"
#include "format/tensor_format.h"
#include "ir/op.h"
#include "kernels/exl3/implementations.h"
#include "kernels/exl3/launch.h"
#include "kernels/exl3/recon_gemm.h"
#include "kernels/exl3/validate.h"
#include "kernels/reference/decode.h"
#include "providers/cuda/cuda_device_execution.h"
#include "providers/device_execution.h"
#include "v1_frozen.h"

namespace {

namespace exl3 = llmp::kernels::exl3;
namespace ref = llmp::kernels::reference;
namespace ex = llmp::execution;
namespace format = llmp::format;
namespace ar = llmp::artifact;
namespace ir = llmp::ir;
namespace fs = std::filesystem;
using exl3::Output;
using format::Encoding;
using format::Exl3Codebook;
using format::Layout;
using format::PartRole;
using format::TensorFormat;

// ---------------------------------------------------------------- selection

ex::OperandFormat Trellis(std::uint8_t operand, std::uint8_t bits,
                          Exl3Codebook codebook = Exl3Codebook::kMcg) {
  return {.operand = operand,
          .encoding = Encoding::kExl3Trellis,
          .layout = Layout::kExl3Tiles,
          .params = {.k_bits = bits, .codebook = codebook, .tensor_scale = false}};
}

ex::OperandFormat Dense(std::uint8_t operand, Encoding encoding) {
  return {.operand = operand, .encoding = encoding, .layout = Layout::kDense, .params = {}};
}

ex::CapabilityQuery Product(std::span<const ex::OperandFormat> operands, std::uint64_t rows,
                            std::uint64_t k = 896, std::uint64_t n = 896,
                            std::uint32_t arch = 121) {
  ex::CapabilityQuery q;
  q.op = ir::Primitive(ir::Op::kMatMul);
  q.operands = operands;
  q.rows = rows;
  q.k = k;
  q.n = n;
  q.arch = arch;
  return q;
}

std::vector<std::string> Names(const ex::Registry& registry, const ex::CapabilityQuery& q) {
  std::vector<std::string> out;
  for (const std::size_t i : ex::Candidates(registry, q)) {
    out.push_back(registry.at(i).name);
  }
  return out;
}

using Strings = std::vector<std::string>;

constexpr std::array<std::string_view, 2> kReconstructions = {"exl3.linear.reconstruct",
                                                              "exl3.linear.reconstruct_fused"};

// Tuning data for both reconstruction paths at each (rows, k, n): what a
// device whose pins cover those GEMMs reports (registry.h TunedShape).
std::vector<ex::TunedShape> TunedFor(std::span<const std::array<std::uint64_t, 3>> shapes) {
  std::vector<ex::TunedShape> out;
  for (const auto& [rows, k, n] : shapes) {
    for (const std::string_view name : kReconstructions) {
      out.push_back({.implementation = name, .rows = rows, .k = k, .n = n});
    }
  }
  return out;
}

ex::CapabilityQuery Tuned(ex::CapabilityQuery q, std::span<const ex::TunedShape> tuned) {
  q.tuned = tuned;
  return q;
}

TEST(Exl3TrellisSelectionTest, CandidatesFollowRowsRatesShapesAndDevices) {
  const auto registry = ex::Registry::Create(exl3::Implementations());
  ASSERT_TRUE(registry.has_value());
  const ex::Registry& r = *registry;
  const std::array k4 = {Trellis(0, 4), Dense(1, Encoding::kF16)};
  const Strings packed = {"exl3.linear.gemv", "exl3.linear.gemm"};
  const Strings prefill = {"exl3.linear.gemm", "exl3.linear.reconstruct",
                           "exl3.linear.reconstruct_fused"};
  constexpr std::array<std::array<std::uint64_t, 3>, 5> kTunedShapes = {
      {{17, 896, 896}, {1024, 896, 896}, {65536, 896, 128}, {145, 896, 896}, {16, 896, 896}}};
  const std::vector<ex::TunedShape> tuned = TunedFor(kTunedShapes);
  // Decode rows: the GEMV through 8 rows, the GEMM through 16, and never
  // a reconstruction (D-108), tuned or not.
  EXPECT_EQ(Names(r, Product(k4, 1)), packed);
  EXPECT_EQ(Names(r, Product(k4, 8)), packed);
  EXPECT_EQ(Names(r, Product(k4, 9)), Strings{"exl3.linear.gemm"});
  EXPECT_EQ(Names(r, Tuned(Product(k4, 16), tuned)), Strings{"exl3.linear.gemm"});
  EXPECT_EQ(Names(r, Tuned(Product(k4, 17), tuned)), prefill);
  EXPECT_EQ(Names(r, Tuned(Product(k4, 1024), tuned)), prefill);
  EXPECT_EQ(Names(r, Tuned(Product(k4, 65536, 896, 128), tuned)), prefill);
  // A reconstruction needs its GEMMs' pins for exactly the shape: none
  // without tuning data, nor for a shape the data does not list.
  EXPECT_EQ(Names(r, Product(k4, 1024)), Strings{"exl3.linear.gemm"});
  EXPECT_EQ(Names(r, Tuned(Product(k4, 1023), tuned)), Strings{"exl3.linear.gemm"});
  EXPECT_EQ(Names(r, Tuned(Product(k4, 1024, 896, 4864), tuned)), Strings{"exl3.linear.gemm"});
  const std::array<ex::TunedShape, 1> plain_only = {
      {{.implementation = "exl3.linear.reconstruct", .rows = 1024, .k = 896, .n = 896}}};
  EXPECT_EQ(Names(r, Tuned(Product(k4, 1024), plain_only)),
            (Strings{"exl3.linear.gemm", "exl3.linear.reconstruct"}));
  // A measurement may put a reconstruction first at prefill rows, never
  // in a decode-shaped product.
  const std::array<ex::MeasuredRank, 1> fused_first = {
      {{.implementation = "exl3.linear.reconstruct_fused", .rank = 0}}};
  ex::CapabilityQuery measured = Tuned(Product(k4, 1024), tuned);
  measured.measured = fused_first;
  EXPECT_EQ(Names(r, measured).front(), "exl3.linear.reconstruct_fused");
  measured.rows = 16;
  EXPECT_EQ(Names(r, measured), Strings{"exl3.linear.gemm"});
  // The GEMV is compiled for K = 4 only.
  for (const int bits : {5, 6, 8}) {
    const std::array kb = {Trellis(0, static_cast<std::uint8_t>(bits)), Dense(1, Encoding::kF16)};
    EXPECT_EQ(Names(r, Product(kb, 1)), Strings{"exl3.linear.gemm"}) << bits;
    EXPECT_EQ(Names(r, Tuned(Product(kb, 145), tuned)), prefill) << bits;
  }
  // Rates and codebooks without kernels, no rate at all, an F32 input.
  for (const int bits : {1, 2, 3, 7}) {
    const std::array kb = {Trellis(0, static_cast<std::uint8_t>(bits)), Dense(1, Encoding::kF16)};
    EXPECT_TRUE(Names(r, Product(kb, 1)).empty()) << bits;
    EXPECT_TRUE(Names(r, Product(kb, 1024)).empty()) << bits;
  }
  for (const Exl3Codebook codebook : {Exl3Codebook::kMul1, Exl3Codebook::k3Inst}) {
    const std::array other = {Trellis(0, 4, codebook), Dense(1, Encoding::kF16)};
    EXPECT_TRUE(Names(r, Product(other, 1)).empty());
  }
  const std::array unrated = {Trellis(0, 0, Exl3Codebook::kNone), Dense(1, Encoding::kF16)};
  EXPECT_TRUE(Names(r, Product(unrated, 1)).empty());
  const std::array f32 = {Trellis(0, 4), Dense(1, Encoding::kF32)};
  EXPECT_TRUE(Names(r, Product(f32, 1)).empty());
  const std::array dense = {Dense(0, Encoding::kF16), Dense(1, Encoding::kF16)};
  EXPECT_TRUE(Names(r, Product(dense, 1)).empty());
  // Shapes: sides off 128 or past 2^19, more than 2^16 rows, rows times a
  // side past int.
  EXPECT_TRUE(Names(r, Product(k4, 1, 896, 960)).empty());
  EXPECT_TRUE(Names(r, Product(k4, 1, 896, std::uint64_t{1} << 20)).empty());
  EXPECT_TRUE(Names(r, Product(k4, 65537)).empty());
  EXPECT_TRUE(Names(r, Product(k4, 65536, 896, 65536)).empty());
  // Devices: nothing below sm_80, a reconstruction only on the GB10 (the
  // packed kernels wherever the build has SASS for the device).
  EXPECT_TRUE(Names(r, Product(k4, 1024, 896, 896, 75)).empty());
  for (const std::string& name : Names(r, Tuned(Product(k4, 1024, 896, 896, 86), tuned))) {
    EXPECT_EQ(name, "exl3.linear.gemm");
  }
  // The multi-GEMM: two trellises of one rate and codebook, one F16 input.
  const std::array two = {Trellis(0, 5), Trellis(1, 5), Dense(2, Encoding::kF16)};
  ex::CapabilityQuery multi = Product(two, 8);
  multi.op = ir::Fused(ir::FusedKind::kPreparedProducts);
  EXPECT_EQ(Names(r, multi), Strings{"exl3.multi_linear.mgemm"});
  const std::array mixed = {Trellis(0, 4), Trellis(1, 5), Dense(2, Encoding::kF16)};
  multi.operands = mixed;
  EXPECT_TRUE(Names(r, multi).empty());
  // Trellis products decode to F16 at rung 2. Reconstruction carries the
  // same rank with explicit transient workspace; packed paths decode on-chip.
  // The bias add has no product capability.
  for (std::size_t i = 0; i < r.size(); ++i) {
    const ex::Implementation& impl = r.at(i);
    EXPECT_EQ(impl.capability.has_value(), impl.name != "exl3.bias_add") << impl.name;
    if (impl.capability) {
      EXPECT_EQ(impl.capability->rung, ex::Rung::kOnChipDecode) << impl.name;
      EXPECT_EQ(impl.capability->expansion == ex::Expansion::kTransientWorkspace,
                impl.name.starts_with("exl3.linear.reconstruct"))
          << impl.name;
      EXPECT_EQ(impl.capability->needs_tuning, impl.name.starts_with("exl3.linear.reconstruct"))
          << impl.name;
    }
  }
}

// ---------------------------------------------------------------- device

class Buffer {
 public:
  explicit Buffer(std::size_t bytes) : bytes_(bytes) {
    EXPECT_EQ(cudaMalloc(&p_, std::max<std::size_t>(bytes, 16)), cudaSuccess);
  }
  explicit Buffer(std::span<const std::byte> host) : Buffer(host.size()) {
    EXPECT_EQ(cudaMemcpy(p_, host.data(), bytes_, cudaMemcpyHostToDevice), cudaSuccess);
  }
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
  Buffer(Buffer&&) = delete;
  Buffer& operator=(Buffer&&) = delete;
  ~Buffer() { (void)cudaFree(p_); }
  std::uint64_t address() const { return reinterpret_cast<std::uint64_t>(p_); }
  template <typename T>
  std::vector<T> Get() const {
    std::vector<T> out(bytes_ / sizeof(T));
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    EXPECT_EQ(cudaMemcpy(out.data(), p_, out.size() * sizeof(T), cudaMemcpyDeviceToHost),
              cudaSuccess);
    return out;
  }

 private:
  void* p_ = nullptr;
  std::size_t bytes_;
};

std::uint64_t Mix(std::uint64_t x) {
  x += 0x9E3779B97F4A7C15ULL;
  x = (x ^ (x >> 30U)) * 0xBF58476D1CE4E5B9ULL;
  x = (x ^ (x >> 27U)) * 0x94D049BB133111EBULL;
  return x ^ (x >> 31U);
}

// F16 values in [-scale, scale), as bytes.
std::vector<std::byte> Halves(std::size_t count, std::uint64_t seed, float scale) {
  std::vector<std::byte> out(2 * count);
  for (std::size_t i = 0; i < count; ++i) {
    const auto bits = static_cast<std::int64_t>(Mix(seed + i) >> 53U);
    const std::uint16_t h = ref::F32ToF16(static_cast<float>(bits - 1024) / 1024.0F * scale);
    std::memcpy(out.data() + (2 * i), &h, 2);
  }
  return out;
}

std::vector<float> Values(std::span<const std::byte> halves) {
  std::vector<float> out(halves.size() / 2);
  for (std::size_t i = 0; i < out.size(); ++i) {
    std::uint16_t h = 0;
    std::memcpy(&h, halves.data() + (2 * i), 2);
    out[i] = ref::F16ToF32(h);
  }
  return out;
}

// One EXL3 linear on the host and the device.
struct Linear {
  std::string name;
  TensorFormat format;
  std::vector<std::byte> codes, suh, svh, marker;
  std::unique_ptr<Buffer> d_codes, d_suh, d_svh;

  void Upload() {
    d_codes = std::make_unique<Buffer>(codes);
    d_suh = std::make_unique<Buffer>(suh);
    d_svh = std::make_unique<Buffer>(svh);
  }
  int k() const { return static_cast<int>(format.shape.ne[0]); }
  int n() const { return static_cast<int>(format.shape.ne[1]); }
  int bits() const { return format.params.k_bits; }
  exl3::Weights weights() const {
    return {.trellis = d_codes->address(),
            .suh = d_suh->address(),
            .svh = d_svh->address(),
            .k = k(),
            .n = n(),
            .bits = bits()};
  }
  std::vector<ref::PartView> parts() const {
    return {{PartRole::kCodes, codes},
            {PartRole::kInputSigns, suh},
            {PartRole::kOutputSigns, svh},
            {PartRole::kCodebookMarker, marker}};
  }
  // Rows (outputs) [0, n) of the original-basis weights, n × k.
  std::vector<float> Rows() const {
    std::vector<float> out(static_cast<std::size_t>(k()) * static_cast<std::size_t>(n()));
    const auto ok = ref::DecodeRows(format, parts(), 0, 0, static_cast<std::uint64_t>(n()), out);
    EXPECT_TRUE(ok.has_value()) << name;
    return out;
  }
};

std::unique_ptr<Linear> RandomLinear(std::uint64_t seed, int k, int n, int bits) {
  auto l = std::make_unique<Linear>();
  l->name = std::format("random {}x{} K{}", k, n, bits);
  l->format = {.encoding = Encoding::kExl3Trellis,
               .params = {.k_bits = static_cast<std::uint8_t>(bits),
                          .codebook = Exl3Codebook::kMcg,
                          .tensor_scale = false},
               .layout = Layout::kExl3Tiles,
               .shape = {.ne = {static_cast<std::uint64_t>(k), static_cast<std::uint64_t>(n), 1, 1},
                         .rank = 2},
               .stacked = 1};
  const auto parts = format::PartsOf(l->format);
  EXPECT_TRUE(parts.has_value());
  l->codes.resize((*parts)[0].bytes);
  for (std::size_t i = 0; i < l->codes.size(); ++i) {
    l->codes[i] = static_cast<std::byte>(Mix(seed + i) >> 56U);
  }
  // Side vectors of mixed signs and magnitudes, as the fixtures' are.
  l->suh = Halves(static_cast<std::size_t>(k), seed + 1, 0.5F);
  l->svh = Halves(static_cast<std::size_t>(n), seed + 2, 2.0F);
  l->marker.resize(4);
  l->Upload();
  return l;
}

double Rms(std::span<const double> v) {
  double s = 0;
  for (const double x : v) {
    s += x * x;
  }
  return v.empty() ? 0 : std::sqrt(s / static_cast<double>(v.size()));
}

// The pinned algorithms (docs/experiments/backend-proof-p0/exl3-recon-pin.json)
// of the reconstruction GEMMs this test runs: F32 outputs (HSS), at the
// fixtures' q_proj, gate_proj and down_proj shapes and 145 and 1,024 rows.
struct Pin {
  int k, n, m;
  std::array<std::uint64_t, 9> config;
};
constexpr std::array<Pin, 6> kPins = {{
    {896, 896, 145, {67, 316, 1, 0, 0, 66, 35, 0, 0}},
    {896, 896, 1024, {67, 409, 1, 0, 0, 30, 35, 0, 0}},
    {896, 4864, 145, {67, 193, 1, 0, 0, 18, 35, 0, 0}},
    {896, 4864, 1024, {21, 20, 1, 0, 0, 0, 11, 0, 0}},
    {4864, 896, 145, {67, 316, 1, 0, 0, 29, 36, 0, 0}},
    {4864, 896, 1024, {67, 30, 1, 0, 0, 23, 35, 0, 0}},
}};

std::optional<exl3::LtAlgorithm> PinFor(int k, int n, int m) {
  for (const Pin& p : kPins) {
    if (p.k == k && p.n == n && p.m == m) {
      return exl3::LtAlgorithm{
          .config = p.config, .m = m, .k = k, .n = n, .ldc = n, .output = Output::kF32};
    }
  }
  return std::nullopt;
}

class Exl3TrellisTest : public ::testing::Test {
 protected:
  void SetUp() override {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
      GTEST_SKIP() << "no CUDA device";
    }
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    cudaDeviceProp prop{};
    ASSERT_EQ(cudaGetDeviceProperties(&prop, 0), cudaSuccess);
    arch_ = static_cast<std::uint32_t>((prop.major * 10) + prop.minor);
    execution_ = std::move(llmp::providers::cuda::OpenDeviceExecution(0).value());
    stream_ = execution_->CreateStream().value();
    locks_ = std::make_unique<Buffer>(exl3::kLockBytes);
    auto launch = exl3::LaunchContext::Create(0, *execution_, stream_, locks_->address());
    ASSERT_TRUE(launch.has_value()) << launch.error().detail;
    launch_ = std::move(*launch);
    auto registry = ex::Registry::Create(exl3::Implementations());
    ASSERT_TRUE(registry.has_value());
    registry_ = std::make_unique<ex::Registry>(std::move(*registry));
  }

  void TearDown() override {
    if (execution_ == nullptr) {
      return;
    }
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    // The stream's work behind a completed, released fence, as the
    // provider requires before destroying it.
    const auto fence = execution_->Record(stream_);
    ASSERT_TRUE(fence.has_value());
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    auto state = execution_->Query(*fence);
    while (state && *state == llmp::providers::FenceState::kPending &&
           std::chrono::steady_clock::now() < deadline) {
      std::this_thread::sleep_for(std::chrono::microseconds(50));
      state = execution_->Query(*fence);
    }
    ASSERT_TRUE(state.has_value() && *state == llmp::providers::FenceState::kComplete);
    EXPECT_TRUE(execution_->Release(*fence).has_value());
    if (launch_ != nullptr) {
      EXPECT_FALSE(launch_->faulted());
    }
    launch_.reset();
    gemm_.reset();
    EXPECT_TRUE(execution_->DestroyStream(stream_).has_value());
  }

  static void Check(const std::expected<void, exl3::KernelFailure>& r, std::string_view what) {
    ASSERT_TRUE(r.has_value()) << what << ": " << r.error().detail;
  }

  // reconstruct_kernel's columns [column, column + columns) against the
  // reference's rotated weights, bit for bit.
  void CheckRotated(const Linear& l, int column, int columns) {
    SCOPED_TRACE(std::format("{} rotated columns [{}, {})", l.name, column, column + columns));
    const auto values = static_cast<std::size_t>(l.k()) * static_cast<std::size_t>(columns);
    const Buffer w(2 * values);
    Check(launch_->Reconstruct({.weights = l.weights(),
                                .w = w.address(),
                                .column = column,
                                .columns = columns,
                                .fused = false}),
          "reconstruct");
    const std::vector<std::uint16_t> got = w.Get<std::uint16_t>();
    std::vector<std::uint16_t> want(values);
    ASSERT_TRUE(ref::DecodeExl3Rotated(l.format, l.parts(), 0, static_cast<std::uint64_t>(column),
                                       static_cast<std::uint64_t>(columns), want)
                    .has_value());
    const auto first = std::ranges::mismatch(got, want);
    ASSERT_TRUE(first.in1 == got.end())
        << "first difference at row " << (first.in1 - got.begin()) / columns << " column "
        << (first.in1 - got.begin()) % columns << ": " << *first.in1 << " against " << *first.in2;
  }

  // reconstruct_had_kernel's original-basis weights against DecodeRows. Its
  // transforms run in F16 butterflies: each of a value's 14 butterfly
  // stages, two scalings and two sign products rounds to F16 (relative
  // 2^-11), on intermediates whose norm stays within twice the result's,
  // so a 128 × 128 block's relative L2 error is bounded near 2^-6 at worst
  // and is about 2^-10 for independent roundings; the check is 2^-7 per
  // block, and any value within 2^-4 of its block's RMS. A misplaced tile
  // or value differs by the block's RMS.
  void CheckFused(const Linear& l) {
    SCOPED_TRACE(std::format("{} fused", l.name));
    const int k = l.k();
    const int n = l.n();
    const Buffer w(std::size_t{2} * static_cast<std::size_t>(k) * static_cast<std::size_t>(n));
    Check(launch_->Reconstruct(
              {.weights = l.weights(), .w = w.address(), .column = 0, .columns = n, .fused = true}),
          "reconstruct_had");
    const std::vector<std::uint16_t> got = w.Get<std::uint16_t>();
    const std::vector<float> rows = l.Rows();
    double worst_l2 = 0;
    double worst_abs = 0;
    for (int kb = 0; kb < k; kb += 128) {
      for (int nb = 0; nb < n; nb += 128) {
        std::vector<double> want;
        std::vector<double> diff;
        for (int i = kb; i < kb + 128; ++i) {
          for (int j = nb; j < nb + 128; ++j) {
            const double v = rows[(static_cast<std::size_t>(j) * static_cast<std::size_t>(k)) +
                                  static_cast<std::size_t>(i)];
            want.push_back(v);
            diff.push_back(
                ref::F16ToF32(got[(static_cast<std::size_t>(i) * static_cast<std::size_t>(n)) +
                                  static_cast<std::size_t>(j)]) -
                v);
          }
        }
        const double rms = Rms(want);
        const double l2 = Rms(diff) / rms;
        double largest = 0;
        for (const double d : diff) {
          largest = std::max(largest, std::abs(d) / rms);
        }
        worst_l2 = std::max(worst_l2, l2);
        worst_abs = std::max(worst_abs, largest);
        ASSERT_LE(l2, 0x1p-7) << "block " << kb << ", " << nb;
        ASSERT_LE(largest, 0x1p-4) << "block " << kb << ", " << nb;
      }
    }
    std::println(
        "{}: fused reconstruction against the reference: relative L2 {:.3g}, largest {:.3g}",
        l.name, worst_l2, worst_abs);
  }

  // A product's checked rows against y = x · W in F64 from the reference's
  // W (n × k) and the F16 inputs. The kernels round the transformed input
  // (or, fused, the weights) to F16 and accumulate in F32: a few roundings
  // of 2^-11 per value, so the check is 2^-8 relative RMS per row and any
  // output within 2^-5 of the row's RMS. A misdecoded tile is off by O(1).
  static void CheckOutputs(const std::string& what, std::span<const float> w,
                           std::span<const float> x, std::span<const float> y, int rows, int k,
                           int n) {
    std::vector<int> checked;
    if (rows <= 16) {
      for (int r = 0; r < rows; ++r) {
        checked.push_back(r);
      }
    } else {
      checked = {0, 1, rows / 2, rows - 1};
    }
    double worst = 0;
    for (const int r : checked) {
      std::vector<double> want(static_cast<std::size_t>(n));
      std::vector<double> diff(static_cast<std::size_t>(n));
      for (int j = 0; j < n; ++j) {
        double sum = 0;
        for (int i = 0; i < k; ++i) {
          sum += static_cast<double>(x[(static_cast<std::size_t>(r) * static_cast<std::size_t>(k)) +
                                       static_cast<std::size_t>(i)]) *
                 w[(static_cast<std::size_t>(j) * static_cast<std::size_t>(k)) +
                   static_cast<std::size_t>(i)];
        }
        want[static_cast<std::size_t>(j)] = sum;
        diff[static_cast<std::size_t>(j)] =
            y[(static_cast<std::size_t>(r) * static_cast<std::size_t>(n)) +
              static_cast<std::size_t>(j)] -
            sum;
      }
      const double rms = Rms(want);
      const double l2 = Rms(diff) / rms;
      double largest = 0;
      for (const double d : diff) {
        largest = std::max(largest, std::abs(d) / rms);
      }
      worst = std::max(worst, l2);
      ASSERT_LE(l2, 0x1p-8) << what << " row " << r;
      ASSERT_LE(largest, 0x1p-5) << what << " row " << r;
    }
    std::println("{}: relative RMS error {:.3g}", what, worst);
  }

  // Every candidate for `rows` rows of the linear, run through its bound
  // implementation, against the reference.
  void CheckProducts(const Linear& l, std::span<const float> w, int rows) {
    const int k = l.k();
    const int n = l.n();
    const std::array operands = {Trellis(0, static_cast<std::uint8_t>(l.bits())),
                                 Dense(1, Encoding::kF16)};
    ex::CapabilityQuery q =
        Product(operands, static_cast<std::uint64_t>(rows), static_cast<std::uint64_t>(k),
                static_cast<std::uint64_t>(n), arch_);
    // The device's tuning data: the pinned GEMMs.
    std::vector<ex::TunedShape> tuned;
    for (const Pin& p : kPins) {
      for (const std::string_view name : kReconstructions) {
        tuned.push_back({.implementation = name,
                         .rows = static_cast<std::uint64_t>(p.m),
                         .k = static_cast<std::uint64_t>(p.k),
                         .n = static_cast<std::uint64_t>(p.n)});
      }
    }
    q.tuned = tuned;
    const std::vector<std::size_t> candidates = ex::Candidates(*registry_, q);
    ASSERT_FALSE(candidates.empty()) << l.name << " at " << rows << " rows";
    const auto xs = Halves(static_cast<std::size_t>(rows) * static_cast<std::size_t>(k),
                           static_cast<std::uint64_t>(rows) * 7919U, 1.0F);
    const std::vector<float> x = Values(xs);
    const Buffer dx{std::span<const std::byte>(xs)};
    const std::size_t x_bytes = xs.size();
    const std::size_t y_bytes =
        std::size_t{4} * static_cast<std::size_t>(rows) * static_cast<std::size_t>(n);
    bool reconstructed = false;
    for (const std::size_t i : candidates) {
      const ex::Implementation& impl = registry_->at(i);
      const std::string what = std::format("{} at {} rows through {}", l.name, rows, impl.name);
      SCOPED_TRACE(what);
      ASSERT_TRUE(rows > 16 || !impl.name.starts_with("exl3.linear.reconstruct"));
      const auto kernel = exl3::Kernel::Bind(impl);
      ASSERT_TRUE(kernel.has_value()) << kernel.error().detail;
      const Buffer y(y_bytes);
      const Buffer scratch(x_bytes);
      const exl3::LinearOperands o{.weights = l.weights(),
                                   .x = dx.address(),
                                   .a_had = scratch.address(),
                                   .y = y.address(),
                                   .output = Output::kF32,
                                   .m = rows};
      if (impl.name == "exl3.linear.gemv") {
        auto plan = launch_->UpstreamGemv(o.weights, Output::kF32, rows);
        ASSERT_TRUE(plan.has_value()) << plan.error().detail;
        exl3::GemvPlan p{.config = 0, .blocks = 0};
        if (*plan) {
          p = **plan;
        } else {
          const auto coresident = launch_->GemvCoresident(l.bits(), Output::kF32, rows, 0);
          ASSERT_TRUE(coresident.has_value()) << coresident.error().detail;
          p.blocks = std::min(n / exl3::GemvColumns(0), *coresident);
        }
        Check(kernel->Run(*launch_, o, p, 0), what);
      } else if (impl.name == "exl3.linear.gemm") {
        const int shape = k % exl3::kTileK[2] == 0 && n % exl3::kTileN[2] == 0 ? 2 : 1;
        const auto coresident = launch_->GemmCoresident(l.bits(), shape, Output::kF32);
        ASSERT_TRUE(coresident.has_value()) << coresident.error().detail;
        const int slices = (k / exl3::kTileK.at(static_cast<std::size_t>(shape))) *
                           (n / exl3::kTileN.at(static_cast<std::size_t>(shape)));
        Check(
            kernel->Run(*launch_, o,
                        exl3::GemmPlan{.shape = shape, .blocks = std::min(slices, *coresident)}, 0),
            what);
      } else {
        const auto pin = PinFor(k, n, rows);
        ASSERT_TRUE(pin.has_value()) << "no pinned algorithm for " << what;
        if (gemm_ == nullptr) {
          auto created = exl3::ReconGemm::Create();
          ASSERT_TRUE(created.has_value()) << created.error().detail;
          gemm_ = std::move(*created);
        }
        const Buffer weights(exl3::ReconstructScratchBytes(o.weights));
        const exl3::ReconstructedOperands ro{.weights = o.weights,
                                             .x = dx.address(),
                                             .xh = scratch.address(),
                                             .w = weights.address(),
                                             .y = y.address(),
                                             .output = Output::kF32,
                                             .m = rows,
                                             .bias = 0};
        const std::array pins = {*pin};
        Check(kernel->Run(*launch_, *gemm_, ro, pins), what);
        reconstructed = true;
      }
      const std::vector<float> got = y.Get<float>();
      CheckOutputs(what, w, x, got, rows, k, n);
    }
    // At prefill rows on the GB10 the reconstruction paths are candidates.
    if (rows > 16 && arch_ == 121) {
      EXPECT_TRUE(reconstructed) << l.name << " at " << rows << " rows";
    }
  }

  // gate and up through the multi-GEMM, against each one's reference.
  void CheckMulti(const Linear& gate, std::span<const float> gate_w, const Linear& up,
                  std::span<const float> up_w, int rows) {
    const int k = gate.k();
    const int n = gate.n();
    const std::array operands = {Trellis(0, static_cast<std::uint8_t>(gate.bits())),
                                 Trellis(1, static_cast<std::uint8_t>(up.bits())),
                                 Dense(2, Encoding::kF16)};
    ex::CapabilityQuery q =
        Product(operands, static_cast<std::uint64_t>(rows), static_cast<std::uint64_t>(k),
                static_cast<std::uint64_t>(n), arch_);
    q.op = ir::Fused(ir::FusedKind::kPreparedProducts);
    const std::vector<std::size_t> candidates = ex::Candidates(*registry_, q);
    if (gate.bits() != up.bits()) {
      EXPECT_TRUE(candidates.empty());
      return;
    }
    ASSERT_EQ(candidates.size(), 1U);
    const auto kernel = exl3::Kernel::Bind(registry_->at(candidates[0]));
    ASSERT_TRUE(kernel.has_value());
    const auto xs =
        Halves(static_cast<std::size_t>(rows) * static_cast<std::size_t>(k), 4241, 1.0F);
    const std::vector<float> x = Values(xs);
    const Buffer dx{std::span<const std::byte>(xs)};
    const auto written = exl3::MultiGemmTables(gate.weights(), up.weights());
    const Buffer tables{std::as_bytes(std::span(written))};
    const Buffer scratch(2 * xs.size());
    const std::size_t y_values = 2 * static_cast<std::size_t>(rows) * static_cast<std::size_t>(n);
    const Buffer y(4 * y_values);
    const exl3::MultiLinearOperands o{.first = gate.weights(),
                                      .second = up.weights(),
                                      .trellis_table = tables.address(),
                                      .suh_table = tables.address() + 16,
                                      .svh_table = tables.address() + 32,
                                      .written = written,
                                      .x = dx.address(),
                                      .a_had = scratch.address(),
                                      .y = y.address(),
                                      .output = Output::kF32,
                                      .m = rows};
    const auto coresident = launch_->MultiGemmCoresident(gate.bits(), 2, Output::kF32);
    ASSERT_TRUE(coresident.has_value()) << coresident.error().detail;
    const int slices = (k / exl3::kTileK[2]) * (n / exl3::kTileN[2]);
    Check(
        kernel->Run(*launch_, o,
                    exl3::MultiGemmPlan{
                        .shape = 2, .blocks = std::min(slices, *coresident / 2), .concurrency = 2}),
        "multi-GEMM");
    const std::vector<float> got = y.Get<float>();
    const std::size_t half = y_values / 2;
    CheckOutputs(std::format("{} through the multi-GEMM", gate.name), gate_w, x,
                 std::span(got).first(half), rows, k, n);
    CheckOutputs(std::format("{} through the multi-GEMM", up.name), up_w, x,
                 std::span(got).subspan(half), rows, k, n);
  }

  std::uint32_t arch_ = 0;
  std::unique_ptr<llmp::providers::DeviceExecution> execution_;
  llmp::providers::StreamId stream_{};
  std::unique_ptr<Buffer> locks_;
  std::unique_ptr<exl3::LaunchContext> launch_;
  std::unique_ptr<exl3::ReconGemm> gemm_;
  std::unique_ptr<ex::Registry> registry_;
};

// Random trellises at every compiled rate: the reconstruction kernels
// against the reference (no pinned GEMM, so on a discrete GPU too).
TEST_F(Exl3TrellisTest, ReconstructionIsTheReferenceDecode) {
  for (const int bits : {4, 5, 6, 8}) {
    const auto l = RandomLinear(static_cast<std::uint64_t>(bits) * 1000003U, 256, 384, bits);
    CheckRotated(*l, 0, 384);
    CheckRotated(*l, 128, 256);
    CheckFused(*l);
  }
}

// ---------------------------------------------------------------- fixtures

// The published v1 EXL3 fixtures (the store's v1-ids.json names; the v1
// import of docs/experiments/artifact-layout's checkpoints).
constexpr std::array<std::pair<std::string_view, std::string_view>, 2> kFixtures = {{
    {"qwen2-exl3-4.0", "EXL3 4.0 bpw"},
    {"qwen2-exl3-4.5", "EXL3 4.5 bpw"},
}};

std::vector<std::byte> ReadResource(const fs::path& root, const ar::Artifact& a,
                                    std::uint32_t resource) {
  const ar::Placement p = a.ResourcePlacement(resource).value();
  const ar::Group& g = a.groups()[p.group];
  const ar::Shard& shard = a.shards()[g.shard];
  std::ifstream in(root / shard.path, std::ios::binary);
  in.seekg(
      static_cast<std::streamoff>(shard.data_offset.value() + g.offset.value() + p.offset.value()));
  std::vector<std::byte> out(p.bytes.value());
  in.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(out.size()));
  EXPECT_TRUE(in.good());
  return out;
}

// The linear `name` ("model.layers.0.mlp.down_proj", "lm_head"): its four
// parts, the resources <name>.trellis, .suh, .svh and .mcg, each recording
// the one trellis format the linear has.
std::unique_ptr<Linear> Load(const fs::path& root, const ar::Artifact& a, const std::string& name) {
  constexpr std::array<std::pair<PartRole, std::string_view>, 4> kParts = {{
      {PartRole::kCodes, ".trellis"},
      {PartRole::kInputSigns, ".suh"},
      {PartRole::kOutputSigns, ".svh"},
      {PartRole::kCodebookMarker, ".mcg"},
  }};
  std::array<std::uint32_t, 4> resources{};
  for (std::size_t i = 0; i < kParts.size(); ++i) {
    const auto found = a.FindResource(name + std::string(kParts[i].second));
    EXPECT_TRUE(found.has_value()) << name << kParts[i].second;
    if (!found) return nullptr;
    const ar::Resource& r = a.resources()[*found];
    EXPECT_EQ(r.part, kParts[i].first) << r.name;
    EXPECT_TRUE(r.format.has_value()) << r.name;
    if (r.part != kParts[i].first || !r.format) return nullptr;
    if (i != 0) EXPECT_EQ(*r.format, *a.resources()[resources[0]].format) << r.name;
    resources[i] = *found;
  }
  auto l = std::make_unique<Linear>();
  l->name = name;
  l->format = *a.resources()[resources[0]].format;
  EXPECT_EQ(l->format.encoding, Encoding::kExl3Trellis) << name;
  EXPECT_EQ(l->format.layout, Layout::kExl3Tiles) << name;
  if (l->format.encoding != Encoding::kExl3Trellis || l->format.layout != Layout::kExl3Tiles) {
    return nullptr;
  }
  l->codes = ReadResource(root, a, resources[0]);
  l->suh = ReadResource(root, a, resources[1]);
  l->svh = ReadResource(root, a, resources[2]);
  l->marker = ReadResource(root, a, resources[3]);
  l->Upload();
  return l;
}

TEST_F(Exl3TrellisTest, FixturesDecodeAsUpstreamAndTheirCandidatesComputeTheirProducts) {
  std::size_t found = 0;
  for (const auto& [fixture, what] : kFixtures) {
    const auto id = llmp::test_support::V1IdNamed(fixture);
    if (!id) {
      continue;
    }
    const fs::path root = fs::path(llmp::test_support::V1Store()) / *id;
    ++found;
    SCOPED_TRACE(what);
    const auto opened = ar::Artifact::Open(root);
    ASSERT_TRUE(opened.has_value()) << root;
    // Every linear's rotated weights, bit for bit: each resource holding a
    // trellis's codes names one.
    std::size_t linears = 0;
    for (const ar::Resource& r : opened->resources()) {
      if (r.part != PartRole::kCodes || !r.name.ends_with(".trellis")) {
        continue;
      }
      const auto l = Load(root, *opened, r.name.substr(0, r.name.size() - 8));
      ASSERT_NE(l, nullptr) << r.name;
      if (l->name == "lm_head") {
        CheckRotated(*l, 0, 1024);
        CheckRotated(*l, l->n() - 1024, 1024);
      } else {
        CheckRotated(*l, 0, l->n());
      }
      ++linears;
    }
    EXPECT_EQ(linears, 169U);
    // Layer 0: the original-basis weights, and the products.
    const std::string attention = "model.layers.0.self_attn.";
    const std::string mlp = "model.layers.0.mlp.";
    for (const std::string& linear :
         {attention + "q_proj", attention + "k_proj", attention + "v_proj", attention + "o_proj",
          mlp + "gate_proj", mlp + "up_proj", mlp + "down_proj"}) {
      const auto l = Load(root, *opened, linear);
      ASSERT_NE(l, nullptr);
      CheckFused(*l);
    }
    for (const std::string& linear : {attention + "q_proj", mlp + "gate_proj", mlp + "down_proj"}) {
      const auto l = Load(root, *opened, linear);
      ASSERT_NE(l, nullptr);
      const std::vector<float> w = l->Rows();
      for (const int rows : {1, 8, 145, 1024}) {
        CheckProducts(*l, w, rows);
      }
    }
    const auto gate = Load(root, *opened, mlp + "gate_proj");
    const auto up = Load(root, *opened, mlp + "up_proj");
    ASSERT_NE(gate, nullptr);
    ASSERT_NE(up, nullptr);
    CheckMulti(*gate, gate->Rows(), *up, up->Rows(), 8);
  }
  if (found == 0) {
    GTEST_FAIL() << "no published v1 EXL3 fixture in " << llmp::test_support::V1Store();
  }
}

}  // namespace
