// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The block decoders under the product templates (kernels/decode/products.h,
// D-108) on a GPU (label `gpu`, and `gpu-discrete`: the GB10 and the
// discrete sm_86 target). Every decoder under every template, at input
// rows {1, 2, 3, 8, 16} (the vector templates) and {1, 2, 3, 8, 16, 64,
// 100, 512, 2048} (matrix, matrix_fma, grouped, rows), against the CPU
// reference decode (kernels/reference/decode.h) and an F64 product of its
// values; weights of several matrices, a grouped product's empty and
// mismatched experts, edge codes (NaN, E8M0 0 and 0xFF, subnormal and
// zero scales), and every template captured into a graph.
//
// Weights are generated in each format: random codes and valid scales
// (E4M3 block scales 2^-3 to 7.5, E8M0 2^-5 to 2^3, NaN codes avoided),
// NVFP4 with a tensor scale per stacked half, dense and Q8_0 through the
// reference encoders. K is 256 plus the decoder's K multiple (a K tile
// half full where the multiple is 16), N 136 (no whole tile or scale
// atom), and every product reads an output range [4, 131).
//
// Tolerances, as bounds on |device - F64| for each output:
//   vector, vector_glu, matrix_fma: (K + 4) · 2^-24 · Σ|w x| (inputs and
//     decoded weights exact in F32; F32 sums in any order, and the tensor
//     scale's rounding);
//   matrix, grouped (mma.sync): that plus 2^-8 · Σ|w x| (inputs rounded to
//     BF16) or 2^-11 · Σ|w x| (to F16), plus 2^-11 · Σ|w x| for Q8_0 (its
//     d · q rounded to F16), plus 2^-24 · Σ|w| (F16 subnormal inputs);
//   vector_glu: the gate's and up's bounds through the GLU (slopes below
//     1.2) plus 2^-20 of the result (F32 exp, erf and tanh);
//   rows: bit-exact (the reference's arithmetic).
// Each product is also row-invariant: an input row's outputs at 16 or 64
// rows equal, bit for bit, its own one-row product.

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <format>
#include <memory>
#include <numbers>
#include <random>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "execution/registry.h"
#include "expected_error.h"
#include "format/encoding.h"
#include "format/layout.h"
#include "format/tensor_format.h"
#include "ir/builder.h"
#include "ir/op.h"
#include "kernels/decode/grouped_route.h"
#include "kernels/decode/implementations.h"
#include "kernels/decode/products.h"
#include "kernels/decode/selected_products.h"
#include "kernels/decode/selected_rows.h"
#include "kernels/decode/selected_source.h"
#include "kernels/ggml/implementations.h"
#include "kernels/ggml/lower.h"
#include "kernels/ggml/tensors.h"
#include "kernels/reference/decode.h"
#include "planner/plan.h"
#include "providers/cuda/cuda_device_execution.h"

namespace {

namespace format = llmp::format;
namespace decode = llmp::kernels::decode;
namespace ref = llmp::kernels::reference;
namespace execution = llmp::execution;
namespace ir = llmp::ir;
using decode::DecoderId;
using format::Encoding;
using format::Layout;
using format::PartRole;
using format::TensorFormat;
using llmp::test_support::Failed;

constexpr auto kDecoders = [] {
  std::array<DecoderId, static_cast<std::size_t>(DecoderId::kCount)> ids{};
  for (std::size_t i = 0; i < ids.size(); ++i) ids[i] = static_cast<DecoderId>(i);
  return ids;
}();
static_assert(kDecoders.size() == static_cast<std::size_t>(DecoderId::kCount));
constexpr std::array<std::uint64_t, 5> kVectorRows = {1, 2, 3, 8, 16};
constexpr std::array<std::uint64_t, 9> kRows = {1, 2, 3, 8, 16, 64, 100, 512, 2048};
constexpr std::uint64_t kN = 136;
constexpr std::uint64_t kFirst = 4;
constexpr std::uint64_t kOutputs = kN - 9;

std::uint64_t KFor(DecoderId id) {
  if (id == DecoderId::kAffineSplit) return 256 + 32;
  return 256 + decode::DecoderKMultiple(id);
}

// Device memory, freed on scope exit.
class Buffer {
 public:
  explicit Buffer(std::size_t bytes) : bytes_(bytes) {
    EXPECT_EQ(cudaMalloc(&p_, std::max<std::size_t>(bytes, 16)), cudaSuccess);
  }
  template <typename T>
  explicit Buffer(const std::vector<T>& host) : Buffer(host.size() * sizeof(T)) {
    EXPECT_EQ(cudaMemcpy(p_, host.data(), bytes_, cudaMemcpyHostToDevice), cudaSuccess);
  }
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
  Buffer(Buffer&&) = delete;
  Buffer& operator=(Buffer&&) = delete;
  ~Buffer() { (void)cudaFree(p_); }
  template <typename T>
  T* as() const {
    return static_cast<T*>(p_);
  }
  template <typename T>
  std::vector<T> Get() const {
    std::vector<T> out(bytes_ / sizeof(T));
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    EXPECT_EQ(cudaMemcpy(out.data(), p_, out.size() * sizeof(T), cudaMemcpyDeviceToHost),
              cudaSuccess);
    return out;
  }
  void Fill(int byte) const { EXPECT_EQ(cudaMemset(p_, byte, bytes_), cudaSuccess); }

 private:
  void* p_ = nullptr;
  std::size_t bytes_;
};

std::uint8_t Scale4m3(std::mt19937& rng) {  // 2^-3 · 1 to 2^2 · 1.875
  const auto e = static_cast<std::uint8_t>(4 + (rng() % 6));
  const auto m = static_cast<std::uint8_t>(rng() % 8);
  return static_cast<std::uint8_t>((e << 3U) | m);
}
std::uint8_t ScaleE8m0(std::mt19937& rng) { return static_cast<std::uint8_t>(122 + (rng() % 9)); }
std::uint8_t CodeE4m3(std::mt19937& rng) {
  auto b = static_cast<std::uint8_t>(rng() & 0xFFU);
  return (b & 0x7FU) == 0x7FU ? static_cast<std::uint8_t>(b ^ 1U) : b;  // never NaN
}
std::byte B(std::uint8_t v) { return std::byte{v}; }

std::vector<float> Normal(std::size_t n, std::mt19937& rng) {
  std::normal_distribution<float> dist(0.0f, 1.0f);
  std::vector<float> out(n);
  for (float& v : out) {
    v = dist(rng);
  }
  return out;
}

struct HostWeight {
  TensorFormat format;
  std::vector<PartRole> roles;
  std::vector<std::vector<std::byte>> bytes;
};

// A weight of `matrices` matrices of [k, n] in the decoder's format, NVFP4
// with a tensor scale per stacked half.
HostWeight MakeWeight(DecoderId id, std::uint64_t k, std::uint64_t n, std::uint64_t matrices,
                      std::mt19937& rng, bool tensor_scale = true, std::uint8_t stacked = 2) {
  HostWeight w;
  TensorFormat& f = w.format;
  f.encoding = decode::DecoderEncoding(id);
  f.layout = decode::DecoderLayout(id);
  f.shape = format::ShapeOf(std::array{k, n, matrices}).value();
  if (f.encoding == Encoding::kNvfp4) {
    f.params.tensor_scale = tensor_scale;
    f.stacked = stacked;
  } else if (f.encoding == Encoding::kAffine) {
    f.params.bits = 4;
    f.params.group = 32;
    f.params.scale = format::AffineScale::kBF16;
  }
  const format::PartList extents = format::PartsOf(f).value();
  for (const format::PartExtent& e : extents.parts()) {
    w.roles.push_back(e.role);
    std::vector<std::byte> bytes(e.bytes);
    switch (e.role) {
      case PartRole::kData:
        if (f.layout == Layout::kDense || f.layout == Layout::kGgmlBlocks) {
          for (std::uint64_t m = 0; m < matrices; ++m) {
            const std::vector<float> values = Normal(n * k, rng);
            EXPECT_TRUE(ref::EncodeRows(f, m, 0, values, bytes).has_value());
          }
        } else if (f.layout == Layout::kGgmlNvfp4Blocks) {
          for (std::size_t b = 0; b < bytes.size(); b += 36) {
            for (std::size_t i = 0; i < 4; ++i) {
              bytes[b + i] = B(Scale4m3(rng));
            }
            for (std::size_t i = 4; i < 36; ++i) {
              bytes[b + i] = B(static_cast<std::uint8_t>(rng()));
            }
          }
        } else {  // row-interleaved NVFP4: codes, then scales
          const std::uint64_t row = (k / 2) + (k / 16);
          for (std::size_t r = 0; r < bytes.size(); r += row) {
            for (std::size_t i = 0; i < row; ++i) {
              bytes[r + i] = i < k / 2 ? B(static_cast<std::uint8_t>(rng())) : B(Scale4m3(rng));
            }
          }
        }
        break;
      case PartRole::kCodes:
        for (std::byte& b : bytes) {
          b = f.encoding == Encoding::kMxfp8 ? B(CodeE4m3(rng))
                                             : B(static_cast<std::uint8_t>(rng()));
        }
        break;
      case PartRole::kBlockScales:
        if (f.encoding == Encoding::kAffine) {
          for (std::size_t i = 0; i < bytes.size(); i += 2) {
            const std::uint16_t s = static_cast<std::uint16_t>(0x3c00U + (rng() % 0x0380U));
            std::memcpy(bytes.data() + i, &s, 2);
          }
        } else {
          for (std::byte& b : bytes) {
            b = (f.encoding == Encoding::kMxfp8 || f.encoding == Encoding::kMxfp4)
                    ? B(ScaleE8m0(rng))
                    : B(Scale4m3(rng));
          }
        }
        break;
      case PartRole::kBiases:
        for (std::size_t i = 0; i < bytes.size(); i += 2) {
          const std::uint16_t b = static_cast<std::uint16_t>(0x3a00U + (rng() % 0x0400U));
          std::memcpy(bytes.data() + i, &b, 2);
        }
        break;
      case PartRole::kTensorScale: {
        std::uniform_real_distribution<float> dist(0.5f, 2.0f);
        for (std::size_t i = 0; i < bytes.size(); i += 4) {
          const float s = dist(rng);
          std::memcpy(bytes.data() + i, &s, 4);
        }
        break;
      }
      default:
        ADD_FAILURE() << "unexpected part";
    }
    w.bytes.push_back(std::move(bytes));
  }
  return w;
}

// A weight on the device, with its reference values [matrices][n][k].
struct DeviceWeight {
  TensorFormat format;
  std::vector<std::unique_ptr<Buffer>> buffers;
  std::vector<decode::DevicePart> parts;
  std::vector<float> values;
  decode::Weight weight(std::uint64_t matrix = 0) const {
    return {.format = format, .parts = parts, .matrix = matrix};
  }
  const float* row(std::uint64_t matrix, std::uint64_t r) const {
    return values.data() + (((matrix * format.shape.ne[1]) + r) * format.shape.ne[0]);
  }
};

std::unique_ptr<DeviceWeight> Upload(const HostWeight& host) {
  auto d = std::make_unique<DeviceWeight>();
  d->format = host.format;
  std::vector<ref::PartView> views;
  for (std::size_t i = 0; i < host.roles.size(); ++i) {
    d->buffers.push_back(std::make_unique<Buffer>(host.bytes[i]));
    d->parts.push_back({.role = host.roles[i],
                        .address = d->buffers.back()->as<void>(),
                        .bytes = host.bytes[i].size()});
    views.push_back({.role = host.roles[i], .bytes = host.bytes[i]});
  }
  const std::uint64_t k = host.format.shape.ne[0];
  const std::uint64_t n = host.format.shape.ne[1];
  const std::uint64_t matrices = format::Matrices(host.format);
  d->values.resize(matrices * n * k);
  for (std::uint64_t m = 0; m < matrices; ++m) {
    const auto ok = ref::DecodeRows(host.format, views, m, 0, n,
                                    std::span(d->values).subspan(m * n * k, n * k));
    EXPECT_TRUE(ok.has_value());
  }
  return d;
}

// Σ w x, Σ|w x| and Σ|w|, in F64.
struct Dot {
  double sum = 0;
  double abs = 0;
  double wabs = 0;
};
Dot DotOf(const float* w, const float* x, std::uint64_t k) {
  Dot d;
  for (std::uint64_t i = 0; i < k; ++i) {
    const double p = static_cast<double>(w[i]) * static_cast<double>(x[i]);
    d.sum += p;
    d.abs += std::abs(p);
    d.wabs += std::abs(static_cast<double>(w[i]));
  }
  return d;
}

double Tolerance(DecoderId id, bool mma, std::uint64_t k) {
  double t = static_cast<double>(k + 4) * 0x1p-24;
  if (mma) {
    t += decode::DecoderMmaElement(id) == "f16" ? 0x1p-11 : 0x1p-8;
    if (id == DecoderId::kQ8_0Blocks) {
      t += 0x1p-11;
    }
  }
  return t;
}

double Bound(const Dot& d, double tolerance) { return (tolerance * d.abs) + (0x1p-24 * d.wabs); }

struct Inputs {
  std::uint64_t rows = 0;
  std::uint64_t stride = 0;
  std::vector<float> host;
  std::unique_ptr<Buffer> device;
  const float* row(std::uint64_t t) const { return host.data() + (t * stride); }
};
Inputs MakeInputs(std::uint64_t rows, std::uint64_t k, std::mt19937& rng) {
  Inputs x;
  x.rows = rows;
  x.stride = k + 4;
  x.host = Normal(rows * x.stride, rng);
  x.device = std::make_unique<Buffer>(x.host);
  return x;
}

// A product's outputs against the F64 product of the reference values.
// Returns the largest error as a fraction of its bound.
double ExpectProduct(const std::vector<float>& y, std::uint64_t y_stride, const Inputs& x,
                     const DeviceWeight& w, std::uint64_t matrix, double tolerance,
                     const std::string& what) {
  const std::uint64_t k = w.format.shape.ne[0];
  double worst = 0;
  for (std::uint64_t t = 0; t < x.rows; ++t) {
    for (std::uint64_t j = 0; j < kOutputs; ++j) {
      const Dot d = DotOf(w.row(matrix, kFirst + j), x.row(t), k);
      const double got = y[(t * y_stride) + j];
      const double error = std::abs(got - d.sum);
      const double bound = Bound(d, tolerance);
      worst = std::max(worst, error / bound);
      EXPECT_LE(error, bound) << what << ": row " << t << " output " << j << " got " << got
                              << " want " << d.sum;
      if (error > bound) {
        return worst;
      }
    }
  }
  return worst;
}

decode::Status RunProduct(decode::Template t, const DeviceWeight& w, const Inputs& x,
                          const Buffer& y, std::uint64_t y_stride) {
  const decode::Product p{.weight = w.weight(),
                          .first = kFirst,
                          .outputs = kOutputs,
                          .x = x.device->as<float>(),
                          .rows = x.rows,
                          .x_stride = x.stride,
                          .y = y.as<float>(),
                          .y_stride = y_stride};
  switch (t) {
    case decode::Template::kVector:
      return decode::RunVector(p, nullptr);
    case decode::Template::kMatrix:
      return decode::RunMatrix(p, decode::MatrixPath::kMma, nullptr);
    default:
      return decode::RunMatrix(p, decode::MatrixPath::kFma, nullptr);
  }
}

class DecodeGpuTest : public ::testing::Test {
 protected:
  void SetUp() override {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
      GTEST_SKIP() << "no CUDA device";
    }
    cudaDeviceProp prop{};
    ASSERT_EQ(cudaGetDeviceProperties(&prop, 0), cudaSuccess);
    arch_ = static_cast<std::uint32_t>((prop.major * 10) + prop.minor);
    ASSERT_TRUE(decode::CompiledFor(arch_)) << "no SASS for sm_" << arch_;
  }
  std::uint32_t arch_ = 0;
  std::mt19937 rng_{20261008};
};

TEST_F(DecodeGpuTest, VectorAndMatrixProductsOfEveryDecoder) {
  for (const DecoderId id : kDecoders) {
    const std::uint64_t k = KFor(id);
    const auto w = Upload(MakeWeight(id, k, kN, 1, rng_));
    for (const decode::Template t :
         {decode::Template::kVector, decode::Template::kMatrix, decode::Template::kMatrixFma}) {
      double worst = 0;
      for (const std::uint64_t rows : kRows) {
        if (t == decode::Template::kVector && rows > decode::kMaxVectorRows) {
          continue;
        }
        const std::string what =
            std::format("{}.{} rows {}", decode::TemplateName(t), decode::DecoderName(id), rows);
        SCOPED_TRACE(what);
        const Inputs x = MakeInputs(rows, k, rng_);
        const std::uint64_t y_stride = kOutputs + 1;
        const Buffer y(rows * y_stride * sizeof(float));
        const auto ran = RunProduct(t, *w, x, y, y_stride);
        ASSERT_TRUE(ran.has_value()) << ran.error();
        worst =
            std::max(worst, ExpectProduct(y.Get<float>(), y_stride, x, *w, 0,
                                          Tolerance(id, t == decode::Template::kMatrix, k), what));
      }
      std::printf("%s.%s: largest error %.3f of its bound\n",
                  std::string(decode::TemplateName(t)).c_str(),
                  std::string(decode::DecoderName(id)).c_str(), worst);
    }
  }
}

// An input row's outputs do not depend on the other rows (D-092).
TEST_F(DecodeGpuTest, ProductsAreRowInvariant) {
  for (const DecoderId id : kDecoders) {
    const std::uint64_t k = KFor(id);
    const auto w = Upload(MakeWeight(id, k, kN, 1, rng_));
    for (const decode::Template t :
         {decode::Template::kVector, decode::Template::kMatrix, decode::Template::kMatrixFma}) {
      const std::uint64_t rows = t == decode::Template::kVector ? 16 : 64;
      SCOPED_TRACE(std::format("{}.{}", decode::TemplateName(t), decode::DecoderName(id)));
      const Inputs x = MakeInputs(rows, k, rng_);
      const std::uint64_t y_stride = kOutputs;
      const Buffer all(rows * y_stride * sizeof(float));
      ASSERT_TRUE(RunProduct(t, *w, x, all, y_stride).has_value());
      const std::vector<float> together = all.Get<float>();
      for (const std::uint64_t r : {std::uint64_t{0}, std::uint64_t{5}, rows - 1}) {
        Inputs one;
        one.rows = 1;
        one.stride = x.stride;
        one.host.assign(x.row(r), x.row(r) + x.stride);
        one.device = std::make_unique<Buffer>(one.host);
        const Buffer alone(y_stride * sizeof(float));
        ASSERT_TRUE(RunProduct(t, *w, one, alone, y_stride).has_value());
        const std::vector<float> got = alone.Get<float>();
        for (std::uint64_t j = 0; j < kOutputs; ++j) {
          ASSERT_EQ(std::bit_cast<std::uint32_t>(got[j]),
                    std::bit_cast<std::uint32_t>(together[(r * y_stride) + j]))
              << "row " << r << " output " << j;
        }
      }
    }
  }
}

double GluOf(ir::GluKind kind, double g, double u, double limit) {
  const auto silu = [](double v) { return v / (1.0 + std::exp(-v)); };
  switch (kind) {
    case ir::GluKind::kGeGluErf:
      return 0.5 * g * (1.0 + std::erf(g / std::sqrt(2.0))) * u;
    case ir::GluKind::kGeGluTanh:
      return 0.5 * g *
             (1.0 + std::tanh(std::sqrt(2.0 / std::numbers::pi) * (g + (0.044715 * g * g * g)))) *
             u;
    case ir::GluKind::kSwiGluClamp:
      return silu(std::min(g, limit)) * std::clamp(u, -limit, limit);
    case ir::GluKind::kSwiGlu:
      break;
  }
  return silu(g) * u;
}

void ExpectGlu(const std::vector<float>& y, std::uint64_t y_stride, const Inputs& x,
               const DeviceWeight& gate, std::uint64_t gate_first, DecoderId gate_id,
               const DeviceWeight& up, std::uint64_t up_first, DecoderId up_id,
               std::uint64_t outputs, ir::GluParams glu, const std::string& what) {
  const std::uint64_t k = gate.format.shape.ne[0];
  for (std::uint64_t t = 0; t < x.rows; ++t) {
    for (std::uint64_t j = 0; j < outputs; ++j) {
      const Dot g = DotOf(gate.row(0, gate_first + j), x.row(t), k);
      const Dot u = DotOf(up.row(0, up_first + j), x.row(t), k);
      const double bg = Bound(g, Tolerance(gate_id, false, k));
      const double bu = Bound(u, Tolerance(up_id, false, k));
      const double want = GluOf(glu.kind, g.sum, u.sum, glu.limit);
      const double act = GluOf(glu.kind, g.sum, 1.0, glu.limit);
      const double bound = (1.2 * bg * (std::abs(u.sum) + bu)) + (std::abs(act) * bu) +
                           (0x1p-20 * std::abs(want)) + 1e-30;
      const double got = y[(t * y_stride) + j];
      ASSERT_LE(std::abs(got - want), bound)
          << what << ": row " << t << " output " << j << " got " << got << " want " << want;
    }
  }
}

TEST_F(DecodeGpuTest, GluOfEveryDecoderAndMixedPairs) {
  struct Case {
    DecoderId gate;
    DecoderId up;
  };
  std::vector<Case> cases;
  for (const DecoderId id : kDecoders) {
    cases.push_back({id, id});
  }
  cases.push_back({DecoderId::kMxfp8Split, DecoderId::kBf16Dense});
  cases.push_back({DecoderId::kNvfp4Split, DecoderId::kQ8_0Blocks});
  for (const Case& c : cases) {
    std::uint64_t k =
        256 + std::max(decode::DecoderKMultiple(c.gate), decode::DecoderKMultiple(c.up));
    if (c.gate == DecoderId::kAffineSplit || c.up == DecoderId::kAffineSplit) {
      k = 256 + 32;
    }
    // One weight is both (gate rows [0, 68), up rows [68, 136), as a
    // stacked gate and up); two weights for a mixed pair.
    const auto gate = Upload(MakeWeight(c.gate, k, kN, 1, rng_));
    const auto up_own = c.gate == c.up ? nullptr : Upload(MakeWeight(c.up, k, kN, 1, rng_));
    const DeviceWeight& up = up_own ? *up_own : *gate;
    const std::uint64_t gate_first = c.gate == c.up ? 0 : 4;
    const std::uint64_t up_first = c.gate == c.up ? kN / 2 : 10;
    const std::uint64_t outputs = c.gate == c.up ? kN / 2 : 100;
    std::vector<ir::GluParams> glus = {{.kind = ir::GluKind::kSwiGlu}};
    if (c.gate == DecoderId::kNvfp4Atoms) {
      glus.push_back({.kind = ir::GluKind::kGeGluErf});
      glus.push_back({.kind = ir::GluKind::kGeGluTanh});
      glus.push_back({.kind = ir::GluKind::kSwiGluClamp, .limit = 7.0f});
    }
    for (const ir::GluParams& glu : glus) {
      for (const std::uint64_t rows : kVectorRows) {
        const std::string what =
            std::format("vector_glu.{}+{} glu {} rows {}", decode::DecoderName(c.gate),
                        decode::DecoderName(c.up), static_cast<int>(glu.kind), rows);
        SCOPED_TRACE(what);
        const Inputs x = MakeInputs(rows, k, rng_);
        const std::uint64_t y_stride = outputs + 2;
        const Buffer y(rows * y_stride * sizeof(float));
        const decode::GluProduct p{.gate = gate->weight(),
                                   .gate_first = gate_first,
                                   .up = up.weight(),
                                   .up_first = up_first,
                                   .outputs = outputs,
                                   .glu = glu,
                                   .x = x.device->as<float>(),
                                   .rows = rows,
                                   .x_stride = x.stride,
                                   .y = y.as<float>(),
                                   .y_stride = y_stride};
        const auto ran = decode::RunVectorGlu(p, nullptr);
        ASSERT_TRUE(ran.has_value()) << ran.error();
        ExpectGlu(y.Get<float>(), y_stride, x, *gate, gate_first, c.gate, up, up_first, c.up,
                  outputs, glu, what);
      }
    }
  }
}

// Six experts, each its own allocation; every token routed to two.
TEST_F(DecodeGpuTest, GroupedProductsOfEveryDecoder) {
  constexpr std::uint64_t kExperts = 6;
  constexpr std::uint64_t kUsed = 2;
  for (const DecoderId id : kDecoders) {
    const std::uint64_t k = KFor(id);
    std::vector<std::unique_ptr<DeviceWeight>> experts;
    std::vector<decode::Operand> operands;
    for (std::uint64_t e = 0; e < kExperts; ++e) {
      experts.push_back(Upload(MakeWeight(id, k, kN, 1, rng_)));
      const auto op = decode::OperandOf(experts.back()->weight());
      ASSERT_TRUE(op.has_value()) << op.error();
      operands.push_back(*op);
    }
    const Buffer table(operands);
    for (const std::uint64_t tokens : kRows) {
      const std::string what = std::format("grouped.{} tokens {}", decode::DecoderName(id), tokens);
      SCOPED_TRACE(what);
      // Each token's experts, then the pairs sorted by expert.
      std::vector<std::array<std::uint64_t, kUsed>> routes(tokens);
      for (auto& r : routes) {
        r[0] = rng_() % kExperts;
        r[1] = (r[0] + 1 + (rng_() % (kExperts - 1))) % kExperts;
      }
      std::vector<std::int32_t> offsets(kExperts + 1, 0);
      std::vector<std::int32_t> pair_rows;
      std::vector<std::int32_t> pair_out;
      for (std::uint64_t e = 0; e < kExperts; ++e) {
        offsets[e] = static_cast<std::int32_t>(pair_rows.size());
        for (std::uint64_t t = 0; t < tokens; ++t) {
          for (std::uint64_t s = 0; s < kUsed; ++s) {
            if (routes[t][s] == e) {
              pair_rows.push_back(static_cast<std::int32_t>(t));
              pair_out.push_back(static_cast<std::int32_t>((t * kUsed) + s));
            }
          }
        }
      }
      offsets[kExperts] = static_cast<std::int32_t>(pair_rows.size());
      const Inputs x = MakeInputs(tokens, k, rng_);
      const std::uint64_t y_stride = kOutputs + 3;
      const std::uint64_t y_rows = tokens * kUsed;
      const Buffer y(y_rows * y_stride * sizeof(float));
      const Buffer d_offsets(offsets);
      const Buffer d_rows(pair_rows);
      const Buffer d_out(pair_out);
      const decode::GroupedProduct p{.format = experts[0]->format,
                                     .experts = table.as<decode::Operand>(),
                                     .experts_count = kExperts,
                                     .first = kFirst,
                                     .outputs = kOutputs,
                                     .offsets = d_offsets.as<std::int32_t>(),
                                     .pair_rows = d_rows.as<std::int32_t>(),
                                     .pair_out = d_out.as<std::int32_t>(),
                                     .pairs = pair_rows.size(),
                                     .x = x.device->as<float>(),
                                     .x_rows = tokens,
                                     .x_stride = x.stride,
                                     .y = y.as<float>(),
                                     .y_rows = y_rows,
                                     .y_stride = y_stride,
                                     .rows_hint = pair_rows.size() / kExperts};
      const auto ran = decode::RunGrouped(p, nullptr);
      ASSERT_TRUE(ran.has_value()) << ran.error();
      const std::vector<float> got = y.Get<float>();
      const double tolerance = Tolerance(id, true, k);
      for (std::uint64_t t = 0; t < tokens; ++t) {
        for (std::uint64_t s = 0; s < kUsed; ++s) {
          const DeviceWeight& w = *experts[routes[t][s]];
          for (std::uint64_t j = 0; j < kOutputs; ++j) {
            const Dot d = DotOf(w.row(0, kFirst + j), x.row(t), k);
            const double value = got[(((t * kUsed) + s) * y_stride) + j];
            ASSERT_LE(std::abs(value - d.sum), Bound(d, tolerance))
                << "token " << t << " slot " << s << " output " << j;
          }
        }
      }
    }
  }
}

// Resident experts at one pitch (one stacked weight): the device builds the
// worklist from the router's IDs (grouped_route.h) and the grouped product
// reads it; IDs outside the experts drop their pairs. Every routed slot's
// output against the reference, inputs a row a token and a row a slot.
// The route's compaction past one expert a thread (each thread a run of
// experts, scanned over the block): groups in ascending expert order, each
// its expert's pointer, runs in order, every routed pair in its run.
TEST_F(DecodeGpuTest, ResidentRoutesCompactManyExpertsInOrder) {
  for (const std::uint32_t experts : {1500U, 4096U}) {
    SCOPED_TRACE(experts);
    constexpr std::uint32_t kUsed = 8, kTokens = 300;
    const std::uint32_t pairs = kUsed * kTokens;
    std::vector<std::int32_t> ids(pairs);
    for (auto& i : ids) i = static_cast<std::int32_t>(rng_() % (experts + 2)) - 1;
    const std::uint32_t groups = std::min(pairs, experts);
    const Buffer d_ids(ids);
    const Buffer table(groups * sizeof(decode::Operand));
    const Buffer offsets((groups + 1) * sizeof(std::int32_t));
    const Buffer rows(pairs * sizeof(std::int32_t));
    const Buffer out_rows(pairs * sizeof(std::int32_t));
    decode::Operand first{};
    first.data = reinterpret_cast<const std::uint8_t*>(std::uintptr_t{1} << 20U);
    first.k = 64;
    const decode::ResidentRoute route{.first = first,
                                      .data_pitch = 256,
                                      .experts = experts,
                                      .ids = d_ids.as<std::int32_t>(),
                                      .ids_stride = kUsed,
                                      .used = kUsed,
                                      .tokens = kTokens,
                                      .groups = groups,
                                      .descriptors = table.as<decode::Operand>(),
                                      .offsets = offsets.as<std::int32_t>(),
                                      .pair_rows = rows.as<std::int32_t>(),
                                      .pair_out = out_rows.as<std::int32_t>()};
    ASSERT_TRUE(decode::PrepareResidentRoute(route, nullptr).has_value());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    std::vector<std::uint32_t> count(experts, 0);
    for (const auto e : ids)
      if (e >= 0 && static_cast<std::uint32_t>(e) < experts) ++count[static_cast<std::size_t>(e)];
    const auto got_table = table.Get<decode::Operand>();
    const auto got_offsets = offsets.Get<std::int32_t>();
    const auto got_out = out_rows.Get<std::int32_t>();
    std::uint32_t c = 0;
    std::int32_t at = 0;
    for (std::uint32_t e = 0; e < experts; ++e) {
      if (count[e] == 0) continue;
      ASSERT_EQ(got_offsets[c], at) << e;
      ASSERT_EQ(reinterpret_cast<std::uintptr_t>(got_table[c].data),
                (std::uintptr_t{1} << 20U) + (std::uintptr_t{e} * 256))
          << e;
      for (std::uint32_t i = 0; i < count[e]; ++i)
        ASSERT_EQ(ids[static_cast<std::size_t>(got_out[static_cast<std::size_t>(at) + i])],
                  static_cast<std::int32_t>(e));
      at += static_cast<std::int32_t>(count[e]);
      ++c;
    }
    for (; c <= groups; ++c) ASSERT_EQ(got_offsets[c], at) << c;
  }
}

TEST_F(DecodeGpuTest, ResidentRoutesBuildTheirWorklistOnTheDevice) {
  constexpr std::uint64_t kExperts = 7;
  constexpr std::uint64_t kUsed = 3;
  // The scale-atom tables are the importer's relaid-out experts.
  for (const DecoderId id :
       {DecoderId::kNvfp4Split, DecoderId::kNvfp4Atoms, DecoderId::kAffineSplit,
        DecoderId::kBf16Dense, DecoderId::kMxfp8Split}) {
    const std::uint64_t k = KFor(id);
    const auto w = Upload(MakeWeight(id, k, kN, kExperts, rng_));
    std::vector<decode::Operand> operands;
    for (std::uint64_t e = 0; e < kExperts; ++e) {
      const auto op = decode::OperandOf(w->weight(e));
      ASSERT_TRUE(op.has_value()) << op.error();
      operands.push_back(*op);
    }
    const auto pitch = [&](auto member) -> std::uint64_t {
      const auto a = reinterpret_cast<std::uintptr_t>(operands[0].*member);
      const auto b = reinterpret_cast<std::uintptr_t>(operands[1].*member);
      return a == 0 ? 0 : b - a;
    };
    auto one = w->format;
    one.shape.rank = 2;
    one.shape.ne[2] = 1;
    for (const bool per_slot : {false, true}) {
      for (const std::uint64_t tokens : {1ULL, 3ULL, 64ULL, 517ULL}) {
        const std::string what = std::format("resident.{} tokens {} {}", decode::DecoderName(id),
                                             tokens, per_slot ? "a row a slot" : "a row a token");
        SCOPED_TRACE(what);
        std::vector<std::int32_t> ids(tokens * kUsed);
        for (auto& i : ids) {
          // Mostly in range; some one past the experts, some negative.
          i = static_cast<std::int32_t>(rng_() % (kExperts + 1));
          if (rng_() % 9 == 0) i = -1;
        }
        const std::uint64_t pairs = tokens * kUsed;
        const std::uint64_t groups = std::min(pairs, kExperts);
        const Inputs x = MakeInputs(per_slot ? pairs : tokens, k, rng_);
        const Buffer d_ids(ids);
        const Buffer table(groups * sizeof(decode::Operand));
        const Buffer offsets((groups + 1) * sizeof(std::int32_t));
        const Buffer rows(pairs * sizeof(std::int32_t));
        const Buffer out_rows(pairs * sizeof(std::int32_t));
        const std::uint64_t y_stride = kOutputs + 3;
        const Buffer y(pairs * y_stride * sizeof(float));
        y.Fill(0xFF);
        const decode::ResidentRoute route{
            .first = operands[0],
            .data_pitch = pitch(&decode::Operand::data),
            .scales_pitch = pitch(&decode::Operand::scales),
            .tensor_scale_pitch = pitch(&decode::Operand::tensor_scale),
            .biases_pitch = pitch(&decode::Operand::biases),
            .experts = kExperts,
            .ids = d_ids.as<std::int32_t>(),
            .ids_stride = kUsed,
            .used = kUsed,
            .tokens = static_cast<std::uint32_t>(tokens),
            .per_slot_input = per_slot,
            .groups = static_cast<std::uint32_t>(groups),
            .descriptors = table.as<decode::Operand>(),
            .offsets = offsets.as<std::int32_t>(),
            .pair_rows = rows.as<std::int32_t>(),
            .pair_out = out_rows.as<std::int32_t>(),
            .y = y.as<float>(),
            .y_stride = y_stride,
            .y_outputs = kOutputs};
        ASSERT_TRUE(decode::PrepareResidentRoute(route, nullptr).has_value());
        const auto ran = decode::RunGrouped({.format = one,
                                             .experts = table.as<decode::Operand>(),
                                             .experts_count = groups,
                                             .first = kFirst,
                                             .outputs = kOutputs,
                                             .offsets = offsets.as<std::int32_t>(),
                                             .pair_rows = rows.as<std::int32_t>(),
                                             .pair_out = out_rows.as<std::int32_t>(),
                                             .pairs = pairs,
                                             .x = x.device->as<float>(),
                                             .x_rows = per_slot ? pairs : tokens,
                                             .x_stride = x.stride,
                                             .y = y.as<float>(),
                                             .y_rows = pairs,
                                             .y_stride = y_stride,
                                             .rows_hint = tokens},
                                            nullptr);
        ASSERT_TRUE(ran.has_value()) << ran.error();
        const std::vector<float> got = y.Get<float>();
        const double tolerance = Tolerance(id, true, k);
        for (std::uint64_t p = 0; p < pairs; ++p) {
          const std::int32_t e = ids[p];
          if (e < 0 || static_cast<std::uint64_t>(e) >= kExperts) {
            // A dropped pair's row reads as zeros.
            for (std::uint64_t j = 0; j < kOutputs; ++j)
              ASSERT_EQ(got[(p * y_stride) + j], 0.0F) << "dropped pair " << p << " output " << j;
            continue;
          }
          const float* input = x.row(per_slot ? p : p / kUsed);
          for (std::uint64_t j = 0; j < kOutputs; ++j) {
            const Dot d = DotOf(w->row(static_cast<std::uint64_t>(e), kFirst + j), input, k);
            const double value = got[(p * y_stride) + j];
            ASSERT_LE(std::abs(value - d.sum), Bound(d, tolerance))
                << "pair " << p << " output " << j;
          }
        }
      }
    }
  }
}

TEST_F(DecodeGpuTest, RowsGatherExactly) {
  for (const DecoderId id : kDecoders) {
    const std::uint64_t k = KFor(id);
    const auto w = Upload(MakeWeight(id, k, kN, 1, rng_));
    for (const std::uint64_t count : kRows) {
      SCOPED_TRACE(std::format("rows.{} count {}", decode::DecoderName(id), count));
      std::vector<std::int32_t> ids(count);
      for (std::int32_t& i : ids) {
        i = static_cast<std::int32_t>(rng_() % kN);
      }
      const Buffer d_ids(ids);
      const std::uint64_t stride = k + 3;
      const Buffer out(count * stride * sizeof(float));
      const Buffer bad(sizeof(std::int32_t));
      bad.Fill(0);
      const decode::RowsGather g{.weight = w->weight(),
                                 .ids = d_ids.as<std::int32_t>(),
                                 .count = count,
                                 .out = out.as<float>(),
                                 .out_stride = stride,
                                 .bad = bad.as<std::int32_t>()};
      const auto ran = decode::RunRows(g, nullptr);
      ASSERT_TRUE(ran.has_value()) << ran.error();
      const std::vector<float> got = out.Get<float>();
      for (std::uint64_t i = 0; i < count; ++i) {
        const float* want = w->row(0, static_cast<std::uint64_t>(ids[i]));
        for (std::uint64_t j = 0; j < k; ++j) {
          ASSERT_EQ(std::bit_cast<std::uint32_t>(got[(i * stride) + j]),
                    std::bit_cast<std::uint32_t>(want[j]))
              << "id " << ids[i] << " value " << j;
        }
      }
      EXPECT_EQ(bad.Get<std::int32_t>()[0], 0);
    }
  }
  // An id outside the matrix: zeros, and the flag.
  const auto w = Upload(MakeWeight(DecoderId::kNvfp4Split, 272, kN, 1, rng_));
  const std::vector<std::int32_t> ids = {3, -1, static_cast<std::int32_t>(kN)};
  const Buffer d_ids(ids);
  const Buffer out(3 * 272 * sizeof(float));
  out.Fill(0xFF);
  const Buffer bad(sizeof(std::int32_t));
  bad.Fill(0);
  ASSERT_TRUE(decode::RunRows({.weight = w->weight(),
                               .ids = d_ids.as<std::int32_t>(),
                               .count = 3,
                               .out = out.as<float>(),
                               .out_stride = 272,
                               .bad = bad.as<std::int32_t>()},
                              nullptr)
                  .has_value());
  const std::vector<float> got = out.Get<float>();
  for (std::uint64_t j = 272; j < got.size(); ++j) {
    ASSERT_EQ(got[j], 0.0f) << j;
  }
  EXPECT_EQ(bad.Get<std::int32_t>()[0], 1);
}

TEST_F(DecodeGpuTest, RefusesWhatTheDecodersCannotRead) {
  const auto w = Upload(MakeWeight(DecoderId::kMxfp8Split, 288, kN, 1, rng_));
  const Inputs x = MakeInputs(1, 288, rng_);
  const Buffer y(kN * sizeof(float));
  decode::Product p{.weight = w->weight(),
                    .first = 0,
                    .outputs = kN,
                    .x = x.device->as<float>(),
                    .rows = 1,
                    .x_stride = x.stride,
                    .y = y.as<float>(),
                    .y_stride = kN};
  ASSERT_TRUE(decode::RunVector(p, nullptr).has_value());

  // Q4_0 is served by GGML, rather than a decoder in this module.
  TensorFormat unsupported = w->format;
  unsupported.encoding = Encoding::kQ4_0;
  unsupported.layout = Layout::kGgmlBlocks;
  EXPECT_FALSE(decode::DecoderOf(unsupported).has_value());
  EXPECT_TRUE(Failed(decode::OperandOf({.format = unsupported, .parts = w->parts})).has_value());
  // A part missing, short, or given twice; a matrix past the format's.
  std::vector<decode::DevicePart> parts = w->parts;
  EXPECT_TRUE(Failed(decode::OperandOf({.format = w->format, .parts = std::span(parts).first(1)}))
                  .has_value());
  parts[1].bytes -= 1;
  EXPECT_TRUE(Failed(decode::OperandOf({.format = w->format, .parts = parts})).has_value());
  parts[1] = parts[0];
  EXPECT_TRUE(Failed(decode::OperandOf({.format = w->format, .parts = parts})).has_value());
  EXPECT_TRUE(
      Failed(decode::OperandOf({.format = w->format, .parts = w->parts, .matrix = 1})).has_value());
  // Misaligned codes.
  parts = w->parts;
  parts[0].address = static_cast<const std::byte*>(parts[0].address) + 8;
  EXPECT_TRUE(Failed(decode::OperandOf({.format = w->format, .parts = parts})).has_value());
  // Rows past the vector template's, outputs past the matrix, a
  // misaligned input or a short stride.
  decode::Product bad = p;
  bad.rows = 17;
  bad.x_stride = 300;
  EXPECT_TRUE(Failed(decode::RunVector(bad, nullptr)).has_value());
  bad = p;
  bad.first = 1;
  EXPECT_TRUE(Failed(decode::RunVector(bad, nullptr)).has_value());
  EXPECT_TRUE(Failed(decode::RunMatrix(bad, decode::MatrixPath::kMma, nullptr)).has_value());
  bad = p;
  bad.x = x.device->as<float>() + 1;
  EXPECT_TRUE(Failed(decode::RunMatrix(bad, decode::MatrixPath::kFma, nullptr)).has_value());
  bad = p;
  bad.rows = 2;
  bad.x_stride = 284;
  EXPECT_TRUE(Failed(decode::RunMatrix(bad, decode::MatrixPath::kMma, nullptr)).has_value());
  // A GLU of different K, or of a pair with no instantiation.
  const auto other = Upload(MakeWeight(DecoderId::kBf16Dense, 272, kN, 1, rng_));
  decode::GluProduct glu{.gate = w->weight(),
                         .up = other->weight(),
                         .outputs = 8,
                         .x = x.device->as<float>(),
                         .rows = 1,
                         .x_stride = x.stride,
                         .y = y.as<float>(),
                         .y_stride = 8};
  EXPECT_TRUE(Failed(decode::RunVectorGlu(glu, nullptr)).has_value());
  const auto f16 = Upload(MakeWeight(DecoderId::kF16Dense, 288, kN, 1, rng_));
  glu.up = f16->weight();
  EXPECT_TRUE(Failed(decode::RunVectorGlu(glu, nullptr)).has_value());
  // A swapped GLU (the caller's to resolve).
  glu.up = w->weight();
  ASSERT_TRUE(decode::RunVectorGlu(glu, nullptr).has_value());
  glu.glu.swapped = true;
  EXPECT_TRUE(Failed(decode::RunVectorGlu(glu, nullptr)).has_value());
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
}

TEST_F(DecodeGpuTest, DeclarationsBindAndSelect) {
  const std::vector<execution::Implementation> declared = decode::Implementations();
  EXPECT_EQ(declared.size(), (kDecoders.size() * 6) + 6);
  const auto registry = execution::Registry::Create(declared);
  ASSERT_TRUE(registry.has_value()) << registry.error().detail;
  for (const execution::Implementation& impl : declared) {
    const auto bound = decode::Bind(impl);
    ASSERT_TRUE(bound.has_value()) << bound.error();
    EXPECT_EQ(decode::ImplementationName(*bound), impl.name);
    ASSERT_TRUE(impl.capability.has_value());
    EXPECT_EQ(impl.capability->expansion, execution::Expansion::kNone);
    EXPECT_EQ(impl.capability->rung,
              (bound->gate == DecoderId::kBf16Dense || bound->gate == DecoderId::kF16Dense) &&
                      (bound->up == DecoderId::kBf16Dense || bound->up == DecoderId::kF16Dense)
                  ? execution::Rung::kNative
                  : execution::Rung::kOnChipDecode);
  }
  execution::Implementation stale = declared.front();
  stale.variant += " (another)";
  EXPECT_TRUE(Failed(decode::Bind(stale)).value_or("").contains("is not this build's"));
  stale.name = "llmp.decode.vector.unknown.split";
  EXPECT_FALSE(decode::Bind(stale).has_value());

  // A decode-shaped NVFP4 product on this device: the vector template
  // first, then the matrix paths; past 16 rows, no vector template.
  const std::array operands = {
      execution::OperandFormat{
          .operand = 0, .encoding = Encoding::kNvfp4, .layout = Layout::kSplit},
      execution::OperandFormat{.operand = 1, .encoding = Encoding::kF32, .layout = Layout::kDense}};
  execution::CapabilityQuery q{.op = ir::Primitive(ir::Op::kMatMul),
                               .operands = operands,
                               .rows = 1,
                               .k = 2560,
                               .n = 10240,
                               .arch = arch_,
                               .features = execution::kBf16Mma};
  auto names = [&](const std::vector<std::size_t>& indices) {
    std::vector<std::string> out;
    for (const std::size_t i : indices) {
      out.push_back(registry->at(i).name);
    }
    return out;
  };
  EXPECT_EQ(
      names(execution::Candidates(*registry, q)),
      (std::vector<std::string>{"llmp.decode.vector.nvfp4.split", "llmp.decode.matrix.nvfp4.split",
                                "llmp.decode.matrix_fma.nvfp4.split"}));
  q.rows = 512;
  EXPECT_EQ(names(execution::Candidates(*registry, q)),
            (std::vector<std::string>{"llmp.decode.matrix.nvfp4.split",
                                      "llmp.decode.matrix_fma.nvfp4.split"}));
  q.rows = std::uint64_t{decode::kMaxMatrixRows} + 1;  // past the matrix templates' grid
  EXPECT_TRUE(execution::Candidates(*registry, q).empty());
  q.rows = 512;
  q.k = 2568;  // not whole blocks
  EXPECT_TRUE(execution::Candidates(*registry, q).empty());
  q.k = 2560;
  q.arch = 75;  // no SASS for it in any profile
  EXPECT_TRUE(execution::Candidates(*registry, q).empty());
}

// ---------------------------------------------------------------- edges

decode::Product ProductOf(const DeviceWeight& w, std::uint64_t matrix, const Inputs& x,
                          const Buffer& y, std::uint64_t y_stride) {
  return {.weight = w.weight(matrix),
          .first = kFirst,
          .outputs = kOutputs,
          .x = x.device->as<float>(),
          .rows = x.rows,
          .x_stride = x.stride,
          .y = y.as<float>(),
          .y_stride = y_stride};
}

// A worklist sorted by expert: experts[e] lists expert e's input rows,
// written to output rows first_out[e], first_out[e] + 1, ...
struct Worklist {
  std::unique_ptr<Buffer> offsets, rows, out;
  std::uint64_t pairs = 0;
};
Worklist WorklistOf(const std::vector<std::vector<std::int32_t>>& experts,
                    const std::vector<std::int32_t>& first_out) {
  std::vector<std::int32_t> offsets;
  std::vector<std::int32_t> rows;
  std::vector<std::int32_t> out;
  for (std::size_t e = 0; e < experts.size(); ++e) {
    offsets.push_back(static_cast<std::int32_t>(rows.size()));
    for (std::size_t i = 0; i < experts[e].size(); ++i) {
      rows.push_back(experts[e][i]);
      out.push_back(first_out[e] + static_cast<std::int32_t>(i));
    }
  }
  offsets.push_back(static_cast<std::int32_t>(rows.size()));
  Worklist w;
  w.pairs = rows.size();
  w.offsets = std::make_unique<Buffer>(offsets);
  w.rows = std::make_unique<Buffer>(rows);
  w.out = std::make_unique<Buffer>(out);
  return w;
}

std::vector<std::int32_t> Iota(std::int32_t count) {
  std::vector<std::int32_t> out(static_cast<std::size_t>(count));
  for (std::int32_t i = 0; i < count; ++i) {
    out[static_cast<std::size_t>(i)] = i;
  }
  return out;
}

// Matrices past the first of one weight (each part's share: codes, scale
// atoms and tensor scales), and grouped products over a weight's matrices:
// experts without pairs, one expert with every pair (a part-full tile of
// 64), a table of one expert, and an operand of another K, whose pairs are
// skipped; and the grouped products equal the matrix template's bit for
// bit (row invariance across tiles and worklists).
TEST_F(DecodeGpuTest, MatricesOfOneWeightAndGroupedEdges) {
  constexpr std::uint64_t kMatrices = 3;
  constexpr std::int32_t kTokens = 37;
  for (const DecoderId id :
       {DecoderId::kNvfp4Atoms, DecoderId::kNvfp4Split, DecoderId::kNvfp4GgmlBlocks,
        DecoderId::kNvfp4Interleaved, DecoderId::kMxfp8Atoms, DecoderId::kQ8_0Blocks,
        DecoderId::kBf16Dense}) {
    SCOPED_TRACE(decode::DecoderName(id));
    const std::uint64_t k = KFor(id);
    const auto w = Upload(MakeWeight(id, k, kN, kMatrices, rng_));
    const Inputs x3 = MakeInputs(3, k, rng_);
    const Inputs x = MakeInputs(kTokens, k, rng_);
    std::vector<std::vector<float>> matrix_out(kMatrices);
    for (std::uint64_t m = 1; m < kMatrices; ++m) {
      const Buffer y3(3 * kOutputs * sizeof(float));
      ASSERT_TRUE(decode::RunVector(ProductOf(*w, m, x3, y3, kOutputs), nullptr).has_value());
      ExpectProduct(y3.Get<float>(), kOutputs, x3, *w, m, Tolerance(id, false, k),
                    std::format("vector matrix {}", m));
      const Buffer y(kTokens * kOutputs * sizeof(float));
      ASSERT_TRUE(
          decode::RunMatrix(ProductOf(*w, m, x, y, kOutputs), decode::MatrixPath::kMma, nullptr)
              .has_value());
      matrix_out[m] = y.Get<float>();
      ExpectProduct(matrix_out[m], kOutputs, x, *w, m, Tolerance(id, true, k),
                    std::format("matrix matrix {}", m));
    }
    // The grouped template's arithmetic is the matrix template's: an
    // expert's outputs equal the matrix product's bit for bit, whatever
    // the tile (64 rows here, 16 for the table of one below).
    const auto expect_matrix = [&](const std::vector<float>& got, std::uint64_t m,
                                   std::uint64_t rows) {
      for (std::uint64_t i = 0; i < rows * kOutputs; ++i) {
        ASSERT_EQ(std::bit_cast<std::uint32_t>(got[i]),
                  std::bit_cast<std::uint32_t>(matrix_out[m][i]))
            << "matrix " << m << " value " << i;
      }
    };

    std::vector<decode::Operand> operands;
    const auto other = Upload(MakeWeight(id, k + decode::DecoderKMultiple(id), kN, 1, rng_));
    operands.push_back(decode::OperandOf(other->weight()).value());
    for (std::uint64_t m = 0; m < kMatrices; ++m) {
      const auto op = decode::OperandOf(w->weight(m));
      ASSERT_TRUE(op.has_value()) << op.error();
      operands.push_back(*op);
    }
    const Buffer table(operands);
    // Expert 0 (another K) takes tokens 0 to 4 into rows 37 to 41, which
    // stay as filled; matrix 1 takes every token; matrices 0 and 2 none.
    const Worklist work = WorklistOf({{0, 1, 2, 3, 4}, {}, Iota(kTokens), {}}, {kTokens, 0, 0, 0});
    const std::uint64_t y_rows = kTokens + 5;
    const Buffer y(y_rows * kOutputs * sizeof(float));
    y.Fill(0xFF);
    decode::GroupedProduct p{.format = w->format,
                             .experts = table.as<decode::Operand>(),
                             .experts_count = operands.size(),
                             .first = kFirst,
                             .outputs = kOutputs,
                             .offsets = work.offsets->as<std::int32_t>(),
                             .pair_rows = work.rows->as<std::int32_t>(),
                             .pair_out = work.out->as<std::int32_t>(),
                             .pairs = work.pairs,
                             .x = x.device->as<float>(),
                             .x_rows = kTokens,
                             .x_stride = x.stride,
                             .y = y.as<float>(),
                             .y_rows = y_rows,
                             .y_stride = kOutputs,
                             .rows_hint = kTokens};
    p.format.shape = format::ShapeOf(std::array{k, kN}).value();
    ASSERT_TRUE(decode::RunGrouped(p, nullptr).has_value());
    const std::vector<float> got = y.Get<float>();
    ExpectProduct(got, kOutputs, x, *w, 1, Tolerance(id, true, k), "grouped, one expert's rows");
    expect_matrix(got, 1, kTokens);
    for (std::size_t i = kTokens * kOutputs; i < got.size(); ++i) {
      ASSERT_EQ(std::bit_cast<std::uint32_t>(got[i]), 0xFFFFFFFFU) << "skipped expert, value " << i;
    }

    // A table of one expert (matrix 2), five tokens: a part-full tile of 16.
    const Buffer one(std::vector<decode::Operand>{operands[3]});
    const Worklist five = WorklistOf({Iota(5)}, {0});
    const Buffer y5(5 * kOutputs * sizeof(float));
    p.experts = one.as<decode::Operand>();
    p.experts_count = 1;
    p.offsets = five.offsets->as<std::int32_t>();
    p.pair_rows = five.rows->as<std::int32_t>();
    p.pair_out = five.out->as<std::int32_t>();
    p.pairs = five.pairs;
    p.y = y5.as<float>();
    p.y_rows = 5;
    p.rows_hint = 5;
    ASSERT_TRUE(decode::RunGrouped(p, nullptr).has_value());
    Inputs first5;
    first5.rows = 5;
    first5.stride = x.stride;
    first5.host = x.host;
    const std::vector<float> got5 = y5.Get<float>();
    ExpectProduct(got5, kOutputs, first5, *w, 2, Tolerance(id, true, k),
                  "grouped, a table of one expert");
    expect_matrix(got5, 2, 5);
  }
}

// Codes and scales at the edges of their element formats: E4M3 NaN codes
// and scales (0x7F, 0xFF), zero, negative-zero and subnormal E4M3 scales,
// E8M0 0xFF (NaN), 0 (2^-127, an F32 subnormal) and 1. Every few scales
// (and a few MXFP8 codes) of a weight take one.
void SetEdges(HostWeight& h) {
  constexpr std::array<std::uint8_t, 5> kE4m3 = {0x7F, 0xFF, 0x00, 0x01, 0x80};
  constexpr std::array<std::uint8_t, 3> kE8m0 = {0xFF, 0x00, 0x01};
  const bool mxfp8 = h.format.encoding == Encoding::kMxfp8;
  const std::uint64_t k = h.format.shape.ne[0];
  std::size_t next = 0;
  const auto edge = [&] {
    ++next;
    return B(mxfp8 ? kE8m0.at(next % kE8m0.size()) : kE4m3.at(next % kE4m3.size()));
  };
  for (std::size_t p = 0; p < h.roles.size(); ++p) {
    std::vector<std::byte>& b = h.bytes[p];
    if (h.roles[p] == PartRole::kBlockScales) {
      for (std::size_t i = 3; i < b.size(); i += 23) {
        b[i] = edge();
      }
    } else if (h.roles[p] == PartRole::kCodes && mxfp8) {
      for (std::size_t i = 5; i < b.size(); i += 997) {
        b[i] = B((i % 2) != 0 ? 0x7F : 0xFF);
      }
    } else if (h.roles[p] != PartRole::kData) {
      continue;  // tensor scales stay valid
    } else if (h.format.layout == Layout::kRowInterleaved) {
      const std::uint64_t row = (k / 2) + (k / 16);
      for (std::size_t r = 0; r < b.size(); r += 2 * row) {
        b[r + (k / 2) + (r / row % (k / 16))] = edge();
      }
    } else if (h.format.layout == Layout::kGgmlNvfp4Blocks) {
      for (std::size_t block = 0; block * 36 < b.size(); block += 7) {
        b[(block * 36) + (block % 4)] = edge();
      }
    }
  }
}

TEST_F(DecodeGpuTest, EdgeCodesDecodeAsTheReference) {
  for (const DecoderId id :
       {DecoderId::kNvfp4Split, DecoderId::kNvfp4Atoms, DecoderId::kNvfp4Interleaved,
        DecoderId::kNvfp4GgmlBlocks, DecoderId::kMxfp8Split, DecoderId::kMxfp8Atoms}) {
    SCOPED_TRACE(decode::DecoderName(id));
    const std::uint64_t k = KFor(id);
    HostWeight host = MakeWeight(id, k, kN, 1, rng_);
    SetEdges(host);
    const auto w = Upload(host);
    std::uint64_t nan_rows = 0;
    for (std::uint64_t r = kFirst; r < kFirst + kOutputs; ++r) {
      nan_rows +=
          std::ranges::any_of(std::span(w->row(0, r), k), [](float v) { return std::isnan(v); })
              ? 1
              : 0;
    }
    EXPECT_GT(nan_rows, 4U);
    EXPECT_LT(nan_rows, kOutputs - 32);

    // Every row gathered: the reference's values bit for bit, NaN where
    // it has NaN.
    const std::vector<std::int32_t> ids = Iota(static_cast<std::int32_t>(kN));
    const Buffer d_ids(ids);
    const Buffer out(kN * k * sizeof(float));
    ASSERT_TRUE(decode::RunRows({.weight = w->weight(),
                                 .ids = d_ids.as<std::int32_t>(),
                                 .count = kN,
                                 .out = out.as<float>(),
                                 .out_stride = k},
                                nullptr)
                    .has_value());
    const std::vector<float> rows = out.Get<float>();
    for (std::uint64_t i = 0; i < kN * k; ++i) {
      const float want = w->values[i];
      if (std::isnan(want)) {
        ASSERT_TRUE(std::isnan(rows[i])) << "value " << i;
      } else {
        ASSERT_EQ(std::bit_cast<std::uint32_t>(rows[i]), std::bit_cast<std::uint32_t>(want))
            << "value " << i;
      }
    }

    // The products: NaN where the row holds a NaN, within the bounds
    // elsewhere.
    for (const decode::Template t :
         {decode::Template::kVector, decode::Template::kMatrix, decode::Template::kMatrixFma}) {
      SCOPED_TRACE(decode::TemplateName(t));
      const Inputs x = MakeInputs(3, k, rng_);
      const Buffer y(3 * kOutputs * sizeof(float));
      ASSERT_TRUE(RunProduct(t, *w, x, y, kOutputs).has_value());
      const std::vector<float> got = y.Get<float>();
      const double tolerance = Tolerance(id, t == decode::Template::kMatrix, k);
      for (std::uint64_t r = 0; r < x.rows; ++r) {
        for (std::uint64_t j = 0; j < kOutputs; ++j) {
          const Dot d = DotOf(w->row(0, kFirst + j), x.row(r), k);
          const double value = got[(r * kOutputs) + j];
          if (std::isnan(d.sum)) {
            ASSERT_TRUE(std::isnan(value)) << "row " << r << " output " << j;
          } else {
            ASSERT_LE(std::abs(value - d.sum), Bound(d, tolerance))
                << "row " << r << " output " << j << " got " << value << " want " << d.sum;
          }
        }
      }
    }
  }
}

// ---------------------------------------------------------------- capture

// Releases a capture's stream, graph and executable.
struct Captured {
  cudaStream_t stream = nullptr;
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t exec = nullptr;
  Captured() = default;
  Captured(const Captured&) = delete;
  Captured& operator=(const Captured&) = delete;
  Captured(Captured&&) = delete;
  Captured& operator=(Captured&&) = delete;
  ~Captured() {
    if (exec != nullptr) {
      (void)cudaGraphExecDestroy(exec);
    }
    if (graph != nullptr) {
      (void)cudaGraphDestroy(graph);
    }
    if (stream != nullptr) {
      (void)cudaStreamDestroy(stream);
    }
  }
};

// `run(stream, y)` eagerly, then captured in global mode (where CUDA
// refuses cudaMalloc and other unsafe calls) into a graph that must be one
// kernel node (no allocation or copy), replayed: the same outputs, bit for
// bit.
template <typename Run>
void ExpectCaptured(const Run& run, std::size_t floats) {
  const Buffer eager(floats * sizeof(float));
  const Buffer replayed(floats * sizeof(float));
  eager.Fill(0);
  replayed.Fill(0);
  const decode::Status direct = run(nullptr, eager.as<float>());
  ASSERT_TRUE(direct.has_value()) << direct.error();
  Captured c;
  ASSERT_EQ(cudaStreamCreateWithFlags(&c.stream, cudaStreamNonBlocking), cudaSuccess);
  ASSERT_EQ(cudaStreamBeginCapture(c.stream, cudaStreamCaptureModeGlobal), cudaSuccess);
  const decode::Status queued = run(c.stream, replayed.as<float>());
  const cudaError_t ended = cudaStreamEndCapture(c.stream, &c.graph);
  ASSERT_TRUE(queued.has_value()) << queued.error();
  ASSERT_EQ(ended, cudaSuccess) << cudaGetErrorString(ended);
  std::size_t count = 0;
  ASSERT_EQ(cudaGraphGetNodes(c.graph, nullptr, &count), cudaSuccess);
  ASSERT_EQ(count, 1U);
  cudaGraphNode_t node = nullptr;
  ASSERT_EQ(cudaGraphGetNodes(c.graph, &node, &count), cudaSuccess);
  cudaGraphNodeType type{};
  ASSERT_EQ(cudaGraphNodeGetType(node, &type), cudaSuccess);
  EXPECT_EQ(type, cudaGraphNodeTypeKernel);
  ASSERT_EQ(cudaGraphInstantiate(&c.exec, c.graph, 0), cudaSuccess);
  ASSERT_EQ(cudaGraphLaunch(c.exec, c.stream), cudaSuccess);
  ASSERT_EQ(cudaStreamSynchronize(c.stream), cudaSuccess);
  const std::vector<float> want = eager.Get<float>();
  const std::vector<float> got = replayed.Get<float>();
  for (std::size_t i = 0; i < floats; ++i) {
    ASSERT_EQ(std::bit_cast<std::uint32_t>(got[i]), std::bit_cast<std::uint32_t>(want[i])) << i;
  }
}

TEST_F(DecodeGpuTest, EveryTemplateCapturesAsOneKernel) {
  for (const DecoderId id : kDecoders) {
    const std::uint64_t k = KFor(id);
    const auto w = Upload(MakeWeight(id, k, kN, 1, rng_));
    const Inputs x2 = MakeInputs(2, k, rng_);
    const Inputs x = MakeInputs(37, k, rng_);
    const auto product = [&](const Inputs& in, float* y) {
      return decode::Product{.weight = w->weight(),
                             .first = kFirst,
                             .outputs = kOutputs,
                             .x = in.device->as<float>(),
                             .rows = in.rows,
                             .x_stride = in.stride,
                             .y = y,
                             .y_stride = kOutputs};
    };
    const auto check = [&](std::string_view what, const auto& run, std::size_t floats) {
      SCOPED_TRACE(std::format("{}.{}", what, decode::DecoderName(id)));
      ExpectCaptured(run, floats);
    };
    check(
        "vector", [&](decode::Stream s, float* y) { return decode::RunVector(product(x2, y), s); },
        2 * kOutputs);
    check(
        "matrix",
        [&](decode::Stream s, float* y) {
          return decode::RunMatrix(product(x, y), decode::MatrixPath::kMma, s);
        },
        x.rows * kOutputs);
    check(
        "matrix_fma",
        [&](decode::Stream s, float* y) {
          return decode::RunMatrix(product(x, y), decode::MatrixPath::kFma, s);
        },
        x.rows * kOutputs);
    check(
        "vector_glu",
        [&](decode::Stream s, float* y) {
          return decode::RunVectorGlu({.gate = w->weight(),
                                       .gate_first = 0,
                                       .up = w->weight(),
                                       .up_first = kN / 2,
                                       .outputs = kN / 2,
                                       .x = x2.device->as<float>(),
                                       .rows = 2,
                                       .x_stride = x2.stride,
                                       .y = y,
                                       .y_stride = kN / 2},
                                      s);
        },
        2 * (kN / 2));
    const decode::Operand op = decode::OperandOf(w->weight()).value();
    const Buffer table(std::vector<decode::Operand>{op, op});
    const Worklist work = WorklistOf({Iota(37), {0, 1, 2}}, {0, 37});
    check(
        "grouped",
        [&](decode::Stream s, float* y) {
          return decode::RunGrouped({.format = w->format,
                                     .experts = table.as<decode::Operand>(),
                                     .experts_count = 2,
                                     .first = kFirst,
                                     .outputs = kOutputs,
                                     .offsets = work.offsets->as<std::int32_t>(),
                                     .pair_rows = work.rows->as<std::int32_t>(),
                                     .pair_out = work.out->as<std::int32_t>(),
                                     .pairs = work.pairs,
                                     .x = x.device->as<float>(),
                                     .x_rows = x.rows,
                                     .x_stride = x.stride,
                                     .y = y,
                                     .y_rows = 40,
                                     .y_stride = kOutputs,
                                     .rows_hint = 20},
                                    s);
        },
        40 * kOutputs);
    const Buffer ids(Iota(7));
    check(
        "rows",
        [&](decode::Stream s, float* y) {
          return decode::RunRows({.weight = w->weight(),
                                  .ids = ids.as<std::int32_t>(),
                                  .count = 7,
                                  .out = y,
                                  .out_stride = k},
                                 s);
        },
        7 * k);
  }
}

// End-to-end selected adapters, rather than direct template launches: every
// decoder over physical parts, batched broadcasting and pitched activations.
llmp::kernels::ggml::SelectedOperand SelectedWeight(llmp::kernels::ggml::TensorArena& arena,
                                                    const DeviceWeight& weight) {
  namespace kg = llmp::kernels::ggml;
  kg::SelectedOperand operand;
  operand.desc.format = weight.format;
  operand.desc.nb = ir::NaturalStrides(weight.format);
  for (const auto& part : weight.parts) {
    auto* tensor =
        ggml_new_tensor_1d(arena.context(), GGML_TYPE_I8, static_cast<std::int64_t>(part.bytes));
    kg::TensorArena::Bind(tensor, reinterpret_cast<std::uintptr_t>(part.address));
    operand.parts.push_back({part.role, tensor});
  }
  operand.tensor = operand.parts.front().tensor;
  return operand;
}
llmp::kernels::ggml::SelectedOperand SelectedActivation(llmp::kernels::ggml::TensorArena& arena,
                                                        format::Shape shape, Encoding encoding,
                                                        ggml_type type, void* address) {
  llmp::kernels::ggml::SelectedOperand operand;
  operand.desc.format = ir::Dense(encoding, shape);
  operand.desc.nb = ir::NaturalStrides(operand.desc.format);
  std::array<std::int64_t, 4> ne{};
  for (std::size_t i = 0; i < 4; ++i) ne[i] = static_cast<std::int64_t>(shape.ne[i]);
  operand.tensor = ggml_new_tensor(arena.context(), type, shape.rank, ne.data());
  llmp::kernels::ggml::TensorArena::Bind(operand.tensor, reinterpret_cast<std::uintptr_t>(address));
  return operand;
}

TEST_F(DecodeGpuTest, SelectedProductsExecuteBatchedBroadcastAndPitchesForEveryDecoder) {
  namespace kg = llmp::kernels::ggml;
  auto execution = llmp::providers::cuda::OpenDeviceExecution(0).value();
  const auto stream = execution->CreateStream().value();
  const auto native = execution->Submission(stream).value();
  auto source = decode::ProductsSelectedSource().value();
  auto launch = source->Launch(0, *execution, stream, {}).value();
  const auto implementations = source->Implementations();
  constexpr std::uint64_t rows = 3, batch2 = 4, batch3 = 2;
  for (const auto decoder : kDecoders) {
    SCOPED_TRACE(decode::DecoderName(decoder));
    const auto k = KFor(decoder);
    const auto weight = Upload(MakeWeight(decoder, k, kN, 2, rng_));
    const auto x = MakeInputs(rows * batch2 * batch3, k, rng_);
    Buffer y(kN * rows * batch2 * batch3 * sizeof(float));
    auto arena = kg::TensorArena::Create(16).value();
    std::array inputs{SelectedWeight(arena, *weight),
                      SelectedActivation(arena, ir::ShapeOf({k, rows, batch2, batch3}),
                                         Encoding::kF32, GGML_TYPE_F32, x.device->as<void>())};
    inputs[1].desc.nb = {4, x.stride * 4, x.stride * rows * 4, x.stride * rows * batch2 * 4};
    std::ranges::copy(inputs[1].desc.nb, inputs[1].tensor->nb);
    std::array outputs{SelectedActivation(arena, ir::ShapeOf({kN, rows, batch2, batch3}),
                                          Encoding::kF32, GGML_TYPE_F32, y.as<void>())};
    ir::Node node;
    node.key = ir::Primitive(ir::Op::kMatMul);
    node.nsrc = 2;
    node.nout = 1;
    for (const auto tmpl :
         {decode::Template::kVector, decode::Template::kMatrix, decode::Template::kMatrixFma}) {
      const auto name = decode::ImplementationName({.tmpl = tmpl, .gate = decoder, .up = decoder});
      const auto at = std::ranges::find(implementations, name, &execution::Implementation::name);
      ASSERT_NE(at, implementations.end());
      auto call = source->Lower(*at, node, inputs, outputs);
      ASSERT_TRUE(call) << Failed(call)->detail;
      // Pitched outputs are not supported; mutation must be caught at Bind.
      outputs[0].tensor->nb[1] += 4;
      EXPECT_FALSE((*call)->Bind(*at, *launch, {}));
      outputs[0].tensor->nb[1] -= 4;
      auto bound = (*call)->Bind(*at, *launch, {});
      ASSERT_TRUE(bound) << Failed(bound)->detail;
      ASSERT_TRUE((*bound)->Run(native));
      const auto got = y.Get<float>();
      for (std::uint64_t b3 = 0; b3 < batch3; ++b3) {
        for (std::uint64_t b2 = 0; b2 < batch2; ++b2) {
          const auto matrix = b2 / 2;
          for (std::uint64_t r = 0; r < rows; ++r) {
            const auto index = ((b3 * batch2 + b2) * rows) + r;
            for (std::uint64_t j = 0; j < kN; ++j) {
              const auto dot = DotOf(weight->row(matrix, j), x.row(index), k);
              EXPECT_LE(std::abs(got[index * kN + j] - dot.sum),
                        Bound(dot, Tolerance(decoder, tmpl == decode::Template::kMatrix, k)));
            }
          }
        }
      }
    }
  }
  const auto fence = execution->Record(stream).value();
  ASSERT_EQ(cudaStreamSynchronize(static_cast<cudaStream_t>(native.handle)), cudaSuccess);
  ASSERT_EQ(execution->Query(fence).value(), llmp::providers::FenceState::kComplete);
  EXPECT_TRUE(execution->Release(fence));
  launch.reset();
  EXPECT_TRUE(execution->DestroyStream(stream));
}

TEST_F(DecodeGpuTest, SelectedResidentRowsExecuteBatchedTablesForEveryDecoder) {
  namespace kg = llmp::kernels::ggml;
  auto execution = llmp::providers::cuda::OpenDeviceExecution(0).value();
  const auto stream = execution->CreateStream().value();
  const auto native = execution->Submission(stream).value();
  auto source = decode::RowsSelectedSource().value();
  auto launch = source->Launch(0, *execution, stream, {}).value();
  const auto implementations = source->Implementations();
  const std::vector<std::int32_t> ids{-1, 0, 135, 5, 5, 0, 136, -1, 7, 2};
  Buffer id_buffer(ids);
  for (const auto decoder : kDecoders) {
    SCOPED_TRACE(decode::DecoderName(decoder));
    const auto k = KFor(decoder);
    const auto weight = Upload(MakeWeight(decoder, k, kN, 2, rng_));
    Buffer y(k * ids.size() * sizeof(float));
    auto arena = kg::TensorArena::Create(16).value();
    std::array inputs{SelectedWeight(arena, *weight),
                      SelectedActivation(arena, ir::ShapeOf({5, 2}), Encoding::kI32, GGML_TYPE_I32,
                                         id_buffer.as<void>())};
    std::array outputs{SelectedActivation(arena, ir::ShapeOf({k, 5, 2}), Encoding::kF32,
                                          GGML_TYPE_F32, y.as<void>())};
    ir::Node node;
    node.key = ir::Primitive(ir::Op::kGetRows);
    node.nsrc = 2;
    node.nout = 1;
    const auto name = decode::ImplementationName(
        {.tmpl = decode::Template::kRows, .gate = decoder, .up = decoder});
    const auto at = std::ranges::find(implementations, name, &execution::Implementation::name);
    ASSERT_NE(at, implementations.end());
    auto call = source->Lower(*at, node, inputs, outputs);
    ASSERT_TRUE(call) << Failed(call)->detail;
    inputs[1].tensor->ne[0] = 10;
    EXPECT_FALSE((*call)->Bind(*at, *launch, {}));
    inputs[1].tensor->ne[0] = 5;
    auto bound = (*call)->Bind(*at, *launch, {});
    ASSERT_TRUE(bound) << Failed(bound)->detail;
    ASSERT_TRUE((*bound)->Run(native));
    const auto got = y.Get<float>();
    for (std::size_t i = 0; i < ids.size(); ++i)
      for (std::uint64_t j = 0; j < k; ++j)
        EXPECT_EQ(got[i * k + j], ids[i] < 0 || ids[i] >= static_cast<std::int32_t>(kN)
                                      ? 0.0f
                                      : weight->row(i / 5, static_cast<std::uint64_t>(ids[i]))[j]);
  }
  const auto fence = execution->Record(stream).value();
  ASSERT_EQ(cudaStreamSynchronize(static_cast<cudaStream_t>(native.handle)), cudaSuccess);
  ASSERT_EQ(execution->Query(fence).value(), llmp::providers::FenceState::kComplete);
  EXPECT_TRUE(execution->Release(fence));
  launch.reset();
  EXPECT_TRUE(execution->DestroyStream(stream));
}

TEST_F(DecodeGpuTest, SelectedGluCapturesSameAndMixedFormatsInBothOperandOrders) {
  namespace kg = llmp::kernels::ggml;
  auto execution = llmp::providers::cuda::OpenDeviceExecution(0).value();
  const auto stream = execution->CreateStream().value();
  const auto native = execution->Submission(stream).value();
  auto source = decode::ProductsSelectedSource().value();
  auto launch = source->Launch(0, *execution, stream, {}).value();
  constexpr std::uint64_t k = 256, rows = 3, batches = 4;
  const auto x = MakeInputs(rows * batches, k, rng_);
  Buffer y(kN * rows * batches * sizeof(float));
  for (const auto& impl : source->Implementations()) {
    const auto declaration = decode::Bind(impl);
    if (!declaration || declaration->tmpl != decode::Template::kVectorGlu) continue;
    SCOPED_TRACE(impl.name);
    const auto gate = Upload(MakeWeight(declaration->gate, k, kN, 2, rng_));
    const auto up = Upload(MakeWeight(declaration->up, k, kN, 2, rng_));
    auto arena = kg::TensorArena::Create(16).value();
    const auto gate_operand = SelectedWeight(arena, *gate);
    const auto up_operand = SelectedWeight(arena, *up);
    auto input = SelectedActivation(arena, ir::ShapeOf({k, rows, batches}), Encoding::kF32,
                                    GGML_TYPE_F32, x.device->as<void>());
    input.desc.nb = {4, x.stride * 4, x.stride * rows * 4, x.stride * rows * batches * 4};
    std::ranges::copy(input.desc.nb, input.tensor->nb);
    std::array outputs{SelectedActivation(arena, ir::ShapeOf({kN, rows, batches}), Encoding::kF32,
                                          GGML_TYPE_F32, y.as<void>())};
    for (const bool swapped : {false, true}) {
      const std::array inputs{swapped ? up_operand : gate_operand,
                              swapped ? gate_operand : up_operand, input};
      ir::Node node;
      node.key = ir::Fused(ir::FusedKind::kMatMulGlu);
      node.nsrc = 3;
      node.nout = 1;
      node.SetParams(ir::GluParams{.kind = ir::GluKind::kSwiGlu, .swapped = swapped});
      auto call = source->Lower(impl, node, inputs, outputs);
      ASSERT_TRUE(call) << Failed(call)->detail;
      auto bound = (*call)->Bind(impl, *launch, {});
      ASSERT_TRUE(bound) << Failed(bound)->detail;
      ASSERT_TRUE((*bound)->Run(native));  // load modules outside capture
      ASSERT_EQ(cudaStreamSynchronize(static_cast<cudaStream_t>(native.handle)), cudaSuccess);
      ASSERT_TRUE(launch->capturable());
      ASSERT_EQ(cudaStreamBeginCapture(static_cast<cudaStream_t>(native.handle),
                                       cudaStreamCaptureModeThreadLocal),
                cudaSuccess);
      ASSERT_TRUE((*bound)->Run(native));
      cudaGraph_t graph = nullptr;
      cudaGraphExec_t captured = nullptr;
      ASSERT_EQ(cudaStreamEndCapture(static_cast<cudaStream_t>(native.handle), &graph),
                cudaSuccess);
      ASSERT_EQ(cudaGraphInstantiate(&captured, graph, 0), cudaSuccess);
      ASSERT_EQ(cudaGraphLaunch(captured, static_cast<cudaStream_t>(native.handle)), cudaSuccess);
      const auto got = y.Get<float>();
      for (std::uint64_t b = 0; b < batches; ++b)
        for (std::uint64_t r = 0; r < rows; ++r)
          for (std::uint64_t j = 0; j < kN; ++j) {
            const auto index = b * rows + r;
            const auto g = DotOf(gate->row(b / 2, j), x.row(index), k);
            const auto u = DotOf(up->row(b / 2, j), x.row(index), k);
            const auto bg = Bound(g, Tolerance(declaration->gate, false, k));
            const auto bu = Bound(u, Tolerance(declaration->up, false, k));
            const auto want = GluOf(ir::GluKind::kSwiGlu, g.sum, u.sum, 0);
            const auto act = GluOf(ir::GluKind::kSwiGlu, g.sum, 1, 0);
            const auto tolerance = (1.2 * bg * (std::abs(u.sum) + bu)) + std::abs(act) * bu +
                                   0x1p-20 * std::abs(want) + 1e-30;
            EXPECT_LE(std::abs(got[index * kN + j] - want), tolerance);
          }
      EXPECT_EQ(cudaGraphExecDestroy(captured), cudaSuccess);
      EXPECT_EQ(cudaGraphDestroy(graph), cudaSuccess);
    }
  }
  const auto fence = execution->Record(stream).value();
  ASSERT_EQ(cudaStreamSynchronize(static_cast<cudaStream_t>(native.handle)), cudaSuccess);
  ASSERT_EQ(execution->Query(fence).value(), llmp::providers::FenceState::kComplete);
  EXPECT_TRUE(execution->Release(fence));
  launch.reset();
  EXPECT_TRUE(execution->DestroyStream(stream));
}

TEST_F(DecodeGpuTest, SharedLoweringExecutesOptionalAndStackedRowInterleavedScales) {
  namespace kg = llmp::kernels::ggml;
  namespace co = llmp::components;
  namespace pl = llmp::planner;
  namespace kn = llmp::kernels;
  auto execution = llmp::providers::cuda::OpenDeviceExecution(0).value();
  const auto stream = execution->CreateStream().value();
  const auto native = execution->Submission(stream).value();
  auto source = decode::DecodeResidentSelectedSource().value();
  auto launch = source->Launch(0, *execution, stream, {}).value();
  auto registry = execution::Registry::Create(source->Implementations()).value();
  constexpr std::uint64_t k = 256, rows = 3, activation_bytes = 1U << 20;
  const ir::WeightRef ref_weight{.stage = 0, .component = 7, .role = 0};
  const pl::DeviceFacts device{
      .device_class = "sm_121", .arch = arch_, .features = execution::kBf16Mma};
  for (const bool scale : {false, true}) {
    for (const std::uint8_t stacked : {std::uint8_t{1}, std::uint8_t{2}}) {
      for (const std::uint64_t batches : {1U, 2U}) {
        const auto weight =
            Upload(MakeWeight(DecoderId::kNvfp4Interleaved, k, kN, batches, rng_, scale, stacked));
        const auto x = Normal(k * rows * batches, rng_);
        std::vector<std::int32_t> ids(rows * batches);
        for (std::size_t i = 0; i < ids.size(); ++i)
          ids[i] = static_cast<std::int32_t>(i % 3 == 0 ? 0 : i % 3 == 1 ? kN / 2 : kN - 1);
        for (const bool gather : {false, true}) {
          SCOPED_TRACE(
              std::format("scale{} stacks{} batches{} gather{}", scale, stacked, batches, gather));
          ir::Builder builder;
          builder.SetOwner(0);
          const auto w = builder.Weight(weight->format, ref_weight);
          const auto input =
              builder.Input(gather ? ir::Dense(Encoding::kI32, ir::ShapeOf({rows, batches}))
                                   : ir::Dense(Encoding::kF32, ir::ShapeOf({k, rows, batches})));
          builder.Output(gather ? builder.GetRows(w, input) : builder.MatMul(w, input));
          auto graph = std::move(builder).Finish();
          ASSERT_TRUE(graph) << graph.error().detail;
          std::vector<std::pair<std::string, PartRole>> parts;
          for (const auto& part : weight->parts)
            parts.emplace_back(std::format("part{}", parts.size()), part.role);
          const std::array weights{co::StepWeight{.tensor = w, .ref = ref_weight, .parts = parts}};
          auto plan = pl::Plan(graph->Clone(), registry, device, {},
                               {.staged = std::vector<ir::TensorId>{input}});
          ASSERT_TRUE(plan) << plan.error().detail;
          Buffer activations(activation_bytes);
          const kn::Addresses addresses{
              .weight = [&](const ir::WeightRef& ref, PartRole role) -> std::uint64_t {
                if (ref != ref_weight) return 0;
                const auto at = std::ranges::find(weight->parts, role, &decode::DevicePart::role);
                return at == weight->parts.end() ? 0
                                                 : reinterpret_cast<std::uintptr_t>(at->address);
              },
              .weight_rows = {},
              .state = {},
              .save = {},
              .activations = reinterpret_cast<std::uintptr_t>(activations.as<void>()),
              .activation_bytes = activation_bytes};
          auto lowered = kg::LowerStep(
              {*plan, co::GraphBuildIO{*graph, graph->inputs(), graph->outputs(), weights},
               addresses},
              registry, {}, {.selected_source = source.get()});
          ASSERT_TRUE(lowered) << lowered.error().detail;
          ASSERT_EQ(lowered->selected.size(), 1U);
          ASSERT_EQ(lowered->inputs.size(), 1U);
          const auto input_bytes =
              gather ? ids.size() * sizeof(std::int32_t) : x.size() * sizeof(float);
          ASSERT_EQ(lowered->inputs[0].bytes, input_bytes);
          ASSERT_EQ(cudaMemcpyAsync(reinterpret_cast<void*>(lowered->inputs[0].device_address),
                                    gather ? static_cast<const void*>(ids.data())
                                           : static_cast<const void*>(x.data()),
                                    input_bytes, cudaMemcpyHostToDevice,
                                    static_cast<cudaStream_t>(native.handle)),
                    cudaSuccess);
          const auto& selected = lowered->selected[0];
          auto bound = selected.call->Bind(*selected.implementation, *launch, {});
          ASSERT_TRUE(bound) << Failed(bound)->detail;
          ASSERT_TRUE((*bound)->Run(native));
          ASSERT_EQ(cudaStreamSynchronize(static_cast<cudaStream_t>(native.handle)), cudaSuccess);
          const auto width = gather ? k : kN;
          std::vector<float> got(width * rows * batches);
          ASSERT_EQ(cudaMemcpy(got.data(), selected.tensor->data, got.size() * sizeof(float),
                               cudaMemcpyDeviceToHost),
                    cudaSuccess);
          for (std::uint64_t b = 0; b < batches; ++b)
            for (std::uint64_t r = 0; r < rows; ++r)
              for (std::uint64_t j = 0; j < width; ++j) {
                const auto at = b * rows + r;
                if (gather) {
                  EXPECT_EQ(got[at * width + j],
                            weight->row(b, static_cast<std::uint64_t>(ids[at]))[j]);
                } else {
                  const auto dot = DotOf(weight->row(b, j), x.data() + at * k, k);
                  EXPECT_LE(std::abs(got[at * width + j] - dot.sum),
                            Bound(dot, Tolerance(DecoderId::kNvfp4Interleaved, false, k)));
                }
              }
        }
      }
    }
  }
  const auto fence = execution->Record(stream).value();
  ASSERT_EQ(cudaStreamSynchronize(static_cast<cudaStream_t>(native.handle)), cudaSuccess);
  ASSERT_EQ(execution->Query(fence).value(), llmp::providers::FenceState::kComplete);
  EXPECT_TRUE(execution->Release(fence));
  launch.reset();
  EXPECT_TRUE(execution->DestroyStream(stream));
}

TEST_F(DecodeGpuTest, CombinedRegistryExecutesBatchedNativeFallbacks) {
  namespace kg = llmp::kernels::ggml;
  namespace co = llmp::components;
  namespace pl = llmp::planner;
  namespace kn = llmp::kernels;
  auto execution = llmp::providers::cuda::OpenDeviceExecution(0).value();
  const auto stream = execution->CreateStream().value();
  const auto native = execution->Submission(stream).value();
  auto source = decode::DecodeResidentSelectedSource().value();
  auto launch = source->Launch(0, *execution, stream, {}).value();
  auto implementations = kg::Implementations();
  auto decoders = source->Implementations();
  implementations.insert(implementations.end(), decoders.begin(), decoders.end());
  auto registry = execution::Registry::Create(std::move(implementations)).value();
  const pl::DeviceFacts device{
      .device_class = "sm_121", .arch = arch_, .features = execution::kBf16Mma};
  constexpr std::uint64_t rows = 3, batches = 2, activation_bytes = 1U << 20;
  const ir::WeightRef ref{.stage = 0, .component = 7, .role = 0};
  for (const bool gather : {false, true}) {
    for (const std::uint64_t weight_batches : {1U, 2U}) {
      if (gather && weight_batches == 1) continue;
      const auto decoder = gather ? DecoderId::kQ8_0Blocks : DecoderId::kMxfp8Split;
      const std::uint64_t k = gather ? 160 : 256;
      const auto weight = Upload(MakeWeight(decoder, k, kN, weight_batches, rng_));
      const auto x = Normal(k * rows * batches, rng_);
      const std::vector<std::int32_t> ids{0, 68, 135, 135, 0, 68};
      ir::Builder builder;
      builder.SetOwner(0);
      const auto w = builder.Weight(weight->format, ref);
      const auto input =
          builder.Input(gather ? ir::Dense(Encoding::kI32, ir::ShapeOf({rows, batches}))
                               : ir::Dense(Encoding::kF32, ir::ShapeOf({k, rows, batches})));
      builder.Output(gather ? builder.GetRows(w, input) : builder.MatMul(w, input));
      auto graph = std::move(builder).Finish();
      ASSERT_TRUE(graph) << graph.error().detail;
      std::vector<std::pair<std::string, PartRole>> parts;
      for (const auto& part : weight->parts)
        parts.emplace_back(std::format("part{}", parts.size()), part.role);
      const std::array weights{co::StepWeight{.tensor = w, .ref = ref, .parts = parts}};
      auto plan = pl::Plan(graph->Clone(), registry, device, {},
                           {.staged = std::vector<ir::TensorId>{input}});
      ASSERT_TRUE(plan) << plan.error().detail;
      ASSERT_EQ(plan->steps.size(), 1U);
      EXPECT_EQ(registry.at(plan->steps[0].implementation).name,
                gather ? "llmp.decode.rows.q8_0.ggml_blocks" : "llmp.decode.vector.mxfp8.split");
      Buffer activations(activation_bytes);
      const kn::Addresses addresses{
          .weight = [&](const ir::WeightRef& key, PartRole role) -> std::uint64_t {
            if (key != ref) return 0;
            const auto at = std::ranges::find(weight->parts, role, &decode::DevicePart::role);
            return at == weight->parts.end() ? 0 : reinterpret_cast<std::uintptr_t>(at->address);
          },
          .weight_rows = {},
          .state = {},
          .save = {},
          .activations = reinterpret_cast<std::uintptr_t>(activations.as<void>()),
          .activation_bytes = activation_bytes};
      auto lowered = kg::LowerStep(
          {*plan, co::GraphBuildIO{*graph, graph->inputs(), graph->outputs(), weights}, addresses},
          registry, {}, {.selected_source = source.get()});
      ASSERT_TRUE(lowered) << lowered.error().detail;
      ASSERT_EQ(lowered->selected.size(), 1U);
      ASSERT_EQ(lowered->inputs.size(), 1U);
      const auto bytes = gather ? ids.size() * sizeof(std::int32_t) : x.size() * sizeof(float);
      ASSERT_EQ(
          cudaMemcpyAsync(
              reinterpret_cast<void*>(lowered->inputs[0].device_address),
              gather ? static_cast<const void*>(ids.data()) : static_cast<const void*>(x.data()),
              bytes, cudaMemcpyHostToDevice, static_cast<cudaStream_t>(native.handle)),
          cudaSuccess);
      const auto& selected = lowered->selected[0];
      auto bound = selected.call->Bind(*selected.implementation, *launch, {});
      ASSERT_TRUE(bound) << Failed(bound)->detail;
      ASSERT_TRUE((*bound)->Run(native));
      ASSERT_EQ(cudaStreamSynchronize(static_cast<cudaStream_t>(native.handle)), cudaSuccess);
      const auto width = gather ? k : kN;
      std::vector<float> got(width * rows * batches);
      ASSERT_EQ(cudaMemcpy(got.data(), selected.tensor->data, got.size() * sizeof(float),
                           cudaMemcpyDeviceToHost),
                cudaSuccess);
      for (std::uint64_t b = 0; b < batches; ++b)
        for (std::uint64_t r = 0; r < rows; ++r)
          for (std::uint64_t j = 0; j < width; ++j) {
            const auto at = b * rows + r;
            const auto matrix = weight_batches == 1 ? 0 : b;
            if (gather) {
              EXPECT_EQ(got[at * width + j],
                        weight->row(matrix, static_cast<std::uint64_t>(ids[at]))[j]);
            } else {
              const auto dot = DotOf(weight->row(matrix, j), x.data() + at * k, k);
              EXPECT_LE(std::abs(got[at * width + j] - dot.sum),
                        Bound(dot, Tolerance(decoder, false, k)));
            }
          }
    }
  }
  const auto fence = execution->Record(stream).value();
  ASSERT_EQ(cudaStreamSynchronize(static_cast<cudaStream_t>(native.handle)), cudaSuccess);
  ASSERT_EQ(execution->Query(fence).value(), llmp::providers::FenceState::kComplete);
  EXPECT_TRUE(execution->Release(fence));
  launch.reset();
  EXPECT_TRUE(execution->DestroyStream(stream));
}

}  // namespace
