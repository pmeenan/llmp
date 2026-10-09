// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The format layer's first proof (D-108; docs/engine-components.md, "First
// proof"): Qwen3.8 Flash Next's native NVFP4 and MXFP8 tensors, read from
// its prepared artifacts, through on-chip decode (kernels/decode) product by
// product against the CPU reference decode and F64 products. Labels `gpu`,
// `models` and `gpu-discrete`: it is meant for the discrete sm_86 target,
// which has no FP4 or FP8 tensor cores, and runs on a GB10 too. The whole
// model does not fit a 3080 Ti, so it takes one layer's worth of each
// product shape:
//   layer 1 (linear attention and the n-gram layer): the MXFP8 projections
//     (qkv, z, beta, alpha, out), the shared expert (gate, up, down; and
//     gate with up as one GLU), its BF16 products (hyper-connection down,
//     up and inject, the n-gram key and value, the router and the shared
//     expert's gate);
//   layer 3 (QSA): the MXFP8 q, k, v, o and indexer projections;
//   layer 1's routed experts, NVFP4 in CUTLASS's scale atoms with gate and
//     up stacked, as grouped products over 30 of the 512 experts (17 i, i
//     < 30) and as one expert's vector GLU and down; where the first
//     import (GGML's block_nvfp4) is also given, its experts' grouped
//     products, which must equal the CUTLASS layout's bit for bit;
//   the n-gram table's row-interleaved NVFP4 rows ([0, 1024), [160000000,
//     160001024) and the last 1024), gathered exactly;
//   the output head (BF16, 248,320 rows) and its hyper-connection products.
// Each product runs at 1 and 16 input rows (vector templates) and 512 (the
// matrix and grouped templates; 1 and 16 for grouped too), random N(0, 1)
// inputs, with decode_gpu_test's tolerances on sampled outputs. Device
// memory: what the products' runs take beyond the allocations (compressed
// weights, inputs, outputs) must stay within 8 MiB (no expansion: the
// templates' tiles live in shared memory), with every module loaded
// before the baseline (CUDA_MODULE_LOADING=EAGER).
//
// The artifacts: LLMP_QWEN38_PROOF_ARTIFACT, the CUTLASS-layout import
// (c4fb47a9...), and optionally LLMP_QWEN38_PROOF_GGML_ARTIFACT, the GGML
// one (67617f87...); by default the M3 store under ~/.local/share/llmp
// (a Spark). A partial copy serves: manifest, index and meta whole, each
// shard at full size with its header and the groups and table rows above
// present (the rest may be holes); the test reads nothing else. Without
// the CUTLASS import the test skips, naming the variable.
//
// Known gap (v0): the CUTLASS import keeps gate's and up's per-expert
// tensor scales as two resources ([512] each), while a stacked
// TensorFormat has one tensor-scale part ordered [matrix][stacked]; the
// test gathers them into that order in a device buffer, which is what a
// launch path does until the importer writes the part itself.

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <format>
#include <fstream>
#include <initializer_list>
#include <memory>
#include <optional>
#include <random>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "artifact/artifact.h"
#include "artifact/format_map.h"
#include "format/encoding.h"
#include "format/layout.h"
#include "format/tensor_format.h"
#include "ir/op.h"
#include "kernels/decode/products.h"
#include "kernels/reference/decode.h"
#include "model/qwen38.h"

namespace {

namespace fs = std::filesystem;
namespace ar = llmp::artifact;
namespace format = llmp::format;
namespace decode = llmp::kernels::decode;
namespace ref = llmp::kernels::reference;
namespace model = llmp::model;
namespace ir = llmp::ir;
using decode::DecoderId;
using format::Encoding;
using format::Layout;
using format::PartRole;
using format::TensorFormat;

constexpr std::string_view kCutlassId =
    "c4fb47a911207c11f935f932d05196dc1701aa0d886eac1b5e91934e554b5a93";
constexpr std::string_view kGgmlId =
    "67617f87f2199408faa6a3663ca86f6d7aafb258a687a8d0a11092ffd74741b6";
constexpr std::uint32_t kExpertLayer = 1;
constexpr std::uint32_t kUsed = 10;
constexpr std::uint64_t kSlack = std::uint64_t{8} << 20U;
constexpr std::uint64_t kGranule = std::uint64_t{2} << 20U;

std::vector<std::uint32_t> Experts() {
  std::vector<std::uint32_t> out;
  for (std::uint32_t i = 0; i < 30; ++i) {
    out.push_back(17 * i);
  }
  return out;
}

fs::path Root(const char* variable, std::string_view id) {
  if (const char* p = std::getenv(variable);
      p != nullptr && *p != '\0') {  // NOLINT(concurrency-mt-unsafe)
    return p;
  }
  const char* home = std::getenv("HOME");  // NOLINT(concurrency-mt-unsafe)
  const fs::path store =
      fs::path(home != nullptr ? home : "/") / ".local/share/llmp/m3-artifacts" / id;
  std::error_code error;
  return fs::is_directory(store, error) ? store : fs::path();
}

// ---------------------------------------------------------------- device memory

// Bytes of every live Buffer, each rounded up to the driver's 2 MiB.
std::uint64_t allocated_bytes = 0;

class Buffer {
 public:
  explicit Buffer(std::size_t bytes) : bytes_(bytes) {
    EXPECT_EQ(cudaMalloc(&p_, std::max<std::size_t>(bytes, 16)), cudaSuccess);
    allocated_bytes += Rounded();
  }
  template <typename T>
  explicit Buffer(std::span<const T> host) : Buffer(host.size_bytes()) {
    EXPECT_EQ(cudaMemcpy(p_, host.data(), bytes_, cudaMemcpyHostToDevice), cudaSuccess);
  }
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
  Buffer(Buffer&&) = delete;
  Buffer& operator=(Buffer&&) = delete;
  ~Buffer() {
    (void)cudaFree(p_);
    allocated_bytes -= Rounded();
  }
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

 private:
  std::uint64_t Rounded() const {
    return (std::max<std::uint64_t>(bytes_, 1) + kGranule - 1) / kGranule * kGranule;
  }
  void* p_ = nullptr;
  std::size_t bytes_;
};

std::uint64_t FreeBytes() {
  std::size_t free = 0;
  std::size_t total = 0;
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  EXPECT_EQ(cudaMemGetInfo(&free, &total), cudaSuccess);
  return free;
}

// Device memory over one test: what each product's run took (the free
// memory just before its launch, its operands allocated, less the free
// memory after it completed), and the most the test held beyond its own
// allocations. Judged on a discrete GPU only: an integrated one's (the
// GB10's) free memory is the host's, which the test's own host buffers
// and the page cache move.
class Memory {
 public:
  Memory() : before_(FreeBytes()), base_(allocated_bytes) {
    cudaDeviceProp prop{};
    EXPECT_EQ(cudaGetDeviceProperties(&prop, 0), cudaSuccess);
    judged_ = prop.integrated == 0;
  }
  void Launching() { launching_ = FreeBytes(); }
  void Ran() {
    const std::uint64_t free = FreeBytes();
    run_growth_ = std::max(run_growth_, launching_ > free ? launching_ - free : 0);
    const std::uint64_t taken = before_ > free ? before_ - free : 0;
    const std::uint64_t ours = allocated_bytes - base_;
    peak_ = std::max(peak_, taken);
    excess_ = std::max(excess_, taken > ours ? taken - ours : 0);
  }
  void Check(std::uint64_t compressed, std::uint64_t expanded, const std::string& what) const {
    std::printf(
        "%s: weights %.1f MiB as stored (%.1f MiB as BF16); peak device memory %.1f MiB, %.1f "
        "MiB beyond its allocations (each rounded to 2 MiB); a product's run took at most %.1f "
        "MiB\n",
        what.c_str(), static_cast<double>(compressed) / 1048576.0,
        static_cast<double>(expanded) / 1048576.0, static_cast<double>(peak_) / 1048576.0,
        static_cast<double>(excess_) / 1048576.0, static_cast<double>(run_growth_) / 1048576.0);
    if (!judged_) {
      std::printf("%s: integrated GPU, device memory not judged\n", what.c_str());
      return;
    }
    EXPECT_LE(run_growth_, kSlack) << what << ": a product's run took device memory";
    EXPECT_LE(excess_, kSlack) << what << ": more device memory than the allocations";
  }

 private:
  bool judged_ = true;
  std::uint64_t before_ = 0;
  std::uint64_t base_ = 0;
  std::uint64_t launching_ = 0;
  std::uint64_t run_growth_ = 0;
  std::uint64_t peak_ = 0;
  std::uint64_t excess_ = 0;
};

// ---------------------------------------------------------------- tensors

std::vector<std::byte> ReadRange(const fs::path& root, const ar::Artifact& a, std::uint32_t group,
                                 std::uint64_t offset, std::uint64_t bytes) {
  const ar::Group& g = a.groups()[group];
  const ar::Shard& shard = a.shards()[g.shard];
  std::ifstream in(root / shard.path, std::ios::binary);
  in.seekg(static_cast<std::streamoff>(shard.data_offset.value() + g.offset.value() + offset));
  std::vector<std::byte> out(bytes);
  in.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(out.size()));
  EXPECT_TRUE(in.good());
  return out;
}

std::vector<std::byte> ReadPlaced(const fs::path& root, const ar::Artifact& a,
                                  const ar::Placement& p) {
  return ReadRange(root, a, p.group, p.offset.value(), p.bytes.value());
}

std::vector<std::byte> ReadResource(const fs::path& root, const ar::Artifact& a,
                                    std::uint32_t resource) {
  return ReadPlaced(root, a, a.ResourcePlacement(resource).value());
}

// A tensor on the host and the device: its format, parts and their views.
struct Tensor {
  std::string name;
  TensorFormat format;
  std::vector<PartRole> roles;
  std::vector<std::vector<std::byte>> host;
  std::vector<std::unique_ptr<Buffer>> device;
  std::vector<decode::DevicePart> parts;

  void Add(PartRole role, std::vector<std::byte> bytes) {
    roles.push_back(role);
    host.push_back(std::move(bytes));
  }
  void Upload() {
    for (std::size_t i = 0; i < host.size(); ++i) {
      device.push_back(std::make_unique<Buffer>(std::span<const std::byte>(host[i])));
      parts.push_back(
          {.role = roles[i], .address = device.back()->as<void>(), .bytes = host[i].size()});
    }
  }
  decode::Weight weight() const { return {.format = format, .parts = parts}; }
  std::uint64_t k() const { return format.shape.ne[0]; }
  std::uint64_t n() const { return format.shape.ne[1]; }
  std::uint64_t stored() const {
    std::uint64_t total = 0;
    for (const auto& h : host) {
      total += h.size();
    }
    return total;
  }
  std::uint64_t expanded() const { return k() * n() * 2; }
  // Row r as the reference decodes it, scaled.
  std::vector<float> Row(std::uint64_t r) const {
    std::vector<ref::PartView> views;
    for (std::size_t i = 0; i < host.size(); ++i) {
      views.push_back({.role = roles[i], .bytes = host[i]});
    }
    std::vector<float> out(k());
    const auto ok = ref::DecodeRows(format, views, 0, r, 1, out);
    EXPECT_TRUE(ok.has_value()) << name;
    return out;
  }
};

// An MXFP8 linear (codes and scales), or a GGML resource as it is (BF16).
std::unique_ptr<Tensor> LoadLinear(const fs::path& root, const ar::Artifact& a,
                                   const std::string& name, const model::Qwen38Mxfp8& m) {
  auto t = std::make_unique<Tensor>();
  t->name = name;
  const auto& codes = a.resources()[m.codes.index];
  const auto& scales = a.resources()[m.scales.index];
  const std::vector<ar::V0Part> parts = {{PartRole::kCodes, &codes.repr},
                                         {PartRole::kBlockScales, &scales.repr}};
  const auto f = ar::Assemble(Encoding::kMxfp8, Layout::kSplit, parts);
  EXPECT_TRUE(f.has_value()) << name;
  t->format = f.value_or(TensorFormat{});
  t->Add(PartRole::kCodes, ReadResource(root, a, m.codes.index));
  t->Add(PartRole::kBlockScales, ReadResource(root, a, m.scales.index));
  t->Upload();
  return t;
}

std::unique_ptr<Tensor> LoadDense(const fs::path& root, const ar::Artifact& a,
                                  const std::string& name, const model::Qwen38Tensor& r) {
  auto t = std::make_unique<Tensor>();
  t->name = name;
  const auto f = ar::FormatOf(a.resources()[r.index].repr);
  EXPECT_TRUE(f.has_value()) << name;
  t->format = f.value_or(TensorFormat{});
  if (t->format.shape.rank == 1) {  // a vector: one output row
    t->format.shape.rank = 2;
  }
  t->Add(PartRole::kData, ReadResource(root, a, r.index));
  t->Upload();
  return t;
}

// ---------------------------------------------------------------- checks

std::vector<float> Normal(std::size_t n, std::mt19937& rng) {
  std::normal_distribution<float> dist(0.0f, 1.0f);
  std::vector<float> out(n);
  for (float& v : out) {
    v = dist(rng);
  }
  return out;
}

struct Dot {
  double sum = 0;
  double abs = 0;
  double wabs = 0;
};
Dot DotOf(std::span<const float> w, const float* x) {
  Dot d;
  for (std::size_t i = 0; i < w.size(); ++i) {
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
  }
  return t;
}
double Bound(const Dot& d, double tolerance) { return (tolerance * d.abs) + (0x1p-24 * d.wabs); }

// The outputs checked: all of a product of at most 16,384 rows, else every
// 31st and the last; the input rows checked: all of 16 or fewer, else 8.
std::vector<std::uint64_t> CheckedOutputs(std::uint64_t n) {
  std::vector<std::uint64_t> out;
  const std::uint64_t step = n <= 16384 ? 1 : 31;
  for (std::uint64_t j = 0; j < n; j += step) {
    out.push_back(j);
  }
  if (out.back() != n - 1) {
    out.push_back(n - 1);
  }
  return out;
}
std::vector<std::uint64_t> CheckedRows(std::uint64_t rows) {
  if (rows <= 16) {
    std::vector<std::uint64_t> out(rows);
    for (std::uint64_t t = 0; t < rows; ++t) {
      out[t] = t;
    }
    return out;
  }
  return {0, 1, 2, 100, 255, 256, 400, rows - 1};
}

// The tensor's products at 1 and 16 rows (vector) and 512 (matrix, mma).
void CheckLinear(const Tensor& t, Memory& memory, std::mt19937& rng) {
  const DecoderId id = decode::DecoderOf(t.format).value();
  const std::uint64_t k = t.k();
  const std::uint64_t n = t.n();
  const std::vector<std::uint64_t> outputs = CheckedOutputs(n);
  std::vector<std::vector<float>> rows_of(outputs.size());
  for (std::size_t i = 0; i < outputs.size(); ++i) {
    rows_of[i] = t.Row(outputs[i]);
  }
  for (const std::uint64_t rows : {std::uint64_t{1}, std::uint64_t{16}, std::uint64_t{512}}) {
    const bool vector = rows <= decode::kMaxVectorRows;
    const std::string what =
        std::format("{} ({}, K {}, N {}) {} rows {}", t.name, decode::DecoderName(id), k, n,
                    vector ? "vector" : "matrix", rows);
    SCOPED_TRACE(what);
    const std::vector<float> x = Normal(rows * k, rng);
    const Buffer dx{std::span<const float>(x)};
    const Buffer dy(rows * n * sizeof(float));
    const decode::Product p{.weight = t.weight(),
                            .first = 0,
                            .outputs = n,
                            .x = dx.as<float>(),
                            .rows = rows,
                            .x_stride = k,
                            .y = dy.as<float>(),
                            .y_stride = n};
    memory.Launching();
    const auto ran = vector ? decode::RunVector(p, nullptr)
                            : decode::RunMatrix(p, decode::MatrixPath::kMma, nullptr);
    ASSERT_TRUE(ran.has_value()) << ran.error();
    memory.Ran();
    const std::vector<float> y = dy.Get<float>();
    const double tolerance = Tolerance(id, !vector, k);
    double worst = 0;
    for (const std::uint64_t r : CheckedRows(rows)) {
      for (std::size_t i = 0; i < outputs.size(); ++i) {
        const Dot d = DotOf(rows_of[i], x.data() + (r * k));
        const double got = y[(r * n) + outputs[i]];
        const double bound = Bound(d, tolerance);
        worst = std::max(worst, std::abs(got - d.sum) / bound);
        ASSERT_LE(std::abs(got - d.sum), bound)
            << "row " << r << " output " << outputs[i] << " got " << got << " want " << d.sum;
      }
    }
    std::printf("%s: largest error %.3f of its bound\n", what.c_str(), worst);
  }
}

double Silu(double v) { return v / (1.0 + std::exp(-v)); }

// A vector SwiGLU of two weights' rows (gate_first + j, up_first + j).
void CheckGlu(const Tensor& gate, std::uint64_t gate_first, const Tensor& up,
              std::uint64_t up_first, std::uint64_t outputs, Memory& memory, std::mt19937& rng) {
  const DecoderId gid = decode::DecoderOf(gate.format).value();
  const DecoderId uid = decode::DecoderOf(up.format).value();
  const std::uint64_t k = gate.k();
  for (const std::uint64_t rows : {std::uint64_t{1}, std::uint64_t{16}}) {
    const std::string what =
        std::format("{} with {}: vector_glu rows {}", gate.name, up.name, rows);
    SCOPED_TRACE(what);
    const std::vector<float> x = Normal(rows * k, rng);
    const Buffer dx{std::span<const float>(x)};
    const Buffer dy(rows * outputs * sizeof(float));
    const decode::GluProduct p{.gate = gate.weight(),
                               .gate_first = gate_first,
                               .up = up.weight(),
                               .up_first = up_first,
                               .outputs = outputs,
                               .glu = {.kind = ir::GluKind::kSwiGlu},
                               .x = dx.as<float>(),
                               .rows = rows,
                               .x_stride = k,
                               .y = dy.as<float>(),
                               .y_stride = outputs};
    memory.Launching();
    const auto ran = decode::RunVectorGlu(p, nullptr);
    ASSERT_TRUE(ran.has_value()) << ran.error();
    memory.Ran();
    const std::vector<float> y = dy.Get<float>();
    for (std::uint64_t j = 0; j < outputs; ++j) {
      const std::vector<float> gw = gate.Row(gate_first + j);
      const std::vector<float> uw = up.Row(up_first + j);
      for (std::uint64_t t = 0; t < rows; ++t) {
        const Dot g = DotOf(gw, x.data() + (t * k));
        const Dot u = DotOf(uw, x.data() + (t * k));
        const double bg = Bound(g, Tolerance(gid, false, k));
        const double bu = Bound(u, Tolerance(uid, false, k));
        const double want = Silu(g.sum) * u.sum;
        const double bound = (1.2 * bg * (std::abs(u.sum) + bu)) + (std::abs(Silu(g.sum)) * bu) +
                             (0x1p-20 * std::abs(want)) + 1e-30;
        const double got = y[(t * outputs) + j];
        ASSERT_LE(std::abs(got - want), bound)
            << "row " << t << " output " << j << " got " << got << " want " << want;
      }
    }
  }
}

class Qwen38DecodeProofTest : public ::testing::Test {
 protected:
  void SetUp() override {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
      GTEST_SKIP() << "no CUDA device";
    }
    root_ = Root("LLMP_QWEN38_PROOF_ARTIFACT", kCutlassId);
    if (root_.empty()) {
      GTEST_SKIP() << "no Qwen3.8 native import (" << kCutlassId
                   << "): set LLMP_QWEN38_PROOF_ARTIFACT to its root, or a partial copy of it";
    }
    auto opened = ar::Artifact::Open(root_);
    ASSERT_TRUE(opened.has_value()) << root_;
    artifact_ = std::make_unique<ar::Artifact>(std::move(*opened));
    auto bound = model::BindQwen38(model::Qwen38Flash(), *artifact_);
    ASSERT_TRUE(bound.has_value()) << bound.error();
    binding_ = std::move(*bound);
    ASSERT_TRUE(binding_.cutlass());
  }

  fs::path root_;
  std::unique_ptr<ar::Artifact> artifact_;
  model::Qwen38Binding binding_;
  std::mt19937 rng_{38};
};

TEST_F(Qwen38DecodeProofTest, LayerProductsMatchTheReference) {
  const ar::Artifact& a = *artifact_;
  const model::Qwen38Layer& linear = binding_.layers.at(1);
  const model::Qwen38Layer& qsa = binding_.layers.at(3);
  ASSERT_TRUE(linear.linear);
  ASSERT_FALSE(qsa.linear);
  Memory memory;
  std::vector<std::unique_ptr<Tensor>> tensors;
  for (const auto& [name, m] :
       std::initializer_list<std::pair<const char*, const model::Qwen38Mxfp8*>>{
           {"blk.1.attn_qkv", &linear.qkv},
           {"blk.1.attn_gate", &linear.z},
           {"blk.1.ssm_beta", &linear.beta},
           {"blk.1.ssm_alpha", &linear.alpha},
           {"blk.1.ssm_out", &linear.ssm_out},
           {"blk.1.ffn_gate_shexp", &linear.gate_shexp},
           {"blk.1.ffn_up_shexp", &linear.up_shexp},
           {"blk.1.ffn_down_shexp", &linear.down_shexp},
           {"blk.3.attn_q", &qsa.q},
           {"blk.3.attn_k", &qsa.k},
           {"blk.3.attn_v", &qsa.v},
           {"blk.3.attn_output", &qsa.o},
           {"blk.3.indexer.qk_proj", &qsa.idx_qk}}) {
    tensors.push_back(LoadLinear(root_, a, name, *m));
  }
  for (const auto& [name, r] :
       std::initializer_list<std::pair<const char*, const model::Qwen38Tensor*>>{
           {"blk.1.hc_attn_down", &linear.hc_attn_down},
           {"blk.1.hc_attn_up", &linear.hc_attn_up},
           {"blk.1.hc_attn_inject", &linear.hc_attn_inject},
           {"blk.1.hc_ffn_down", &linear.hc_ffn_down},
           {"blk.1.hc_ffn_up", &linear.hc_ffn_up},
           {"blk.1.hc_ffn_inject", &linear.hc_ffn_inject},
           {"blk.1.ple_key", &linear.ple_key},
           {"blk.1.ple_value", &linear.ple_value},
           {"blk.1.ffn_gate_inp", &linear.router},
           {"blk.1.ffn_gate_inp_shexp", &linear.shared_gate}}) {
    tensors.push_back(LoadDense(root_, a, name, *r));
  }
  std::uint64_t stored = 0;
  std::uint64_t expanded = 0;
  for (const auto& t : tensors) {
    ASSERT_TRUE(decode::DecoderOf(t->format).has_value()) << t->name;
    stored += t->stored();
    expanded += t->expanded();
    CheckLinear(*t, memory, rng_);
  }
  // The shared expert's gate and up as one GLU.
  CheckGlu(*tensors[5], 0, *tensors[6], 0, tensors[5]->n(), memory, rng_);
  memory.Check(stored, expanded, "layers 1 and 3");
}

TEST_F(Qwen38DecodeProofTest, HeadProductsMatchTheReference) {
  Memory memory;
  std::vector<std::unique_ptr<Tensor>> tensors;
  tensors.push_back(LoadDense(root_, *artifact_, "output", binding_.output));
  tensors.push_back(LoadDense(root_, *artifact_, "output_hc_down", binding_.output_hc_down));
  tensors.push_back(LoadDense(root_, *artifact_, "output_hc_up", binding_.output_hc_up));
  std::uint64_t stored = 0;
  for (const auto& t : tensors) {
    stored += t->stored();
    CheckLinear(*t, memory, rng_);
  }
  memory.Check(stored, stored, "the output head");
}

TEST_F(Qwen38DecodeProofTest, NgramTableRowsDecodeExactly) {
  const ar::Artifact& a = *artifact_;
  const auto& table = a.resources()[binding_.ple_table.index];
  const auto& scale = a.resources()[binding_.ple_table_scale.index];
  const std::vector<ar::V0Part> parts = {{PartRole::kData, &table.repr},
                                         {PartRole::kTensorScale, &scale.repr}};
  const auto whole = ar::Assemble(Encoding::kNvfp4, Layout::kRowInterleaved, parts);
  ASSERT_TRUE(whole.has_value());
  ASSERT_EQ(whole->shape.ne[0], 160U);
  const std::uint64_t rows = whole->shape.ne[1];
  const std::uint64_t row_bytes = format::RowBytes(*whole, PartRole::kData).value();
  ASSERT_EQ(row_bytes, 90U);
  // The rows the proof reads, as one table of 3,072 rows.
  Memory memory;
  Tensor t;
  t.name = "per_layer_token_embd";
  t.format = *whole;
  t.format.shape.ne[1] = 3072;
  const ar::Placement placed = a.ResourcePlacement(binding_.ple_table.index).value();
  std::vector<std::byte> bytes;
  for (const std::uint64_t first : {std::uint64_t{0}, std::uint64_t{160000000}, rows - 1024}) {
    const std::vector<std::byte> range = ReadRange(
        root_, a, placed.group, placed.offset.value() + (first * row_bytes), 1024 * row_bytes);
    bytes.insert(bytes.end(), range.begin(), range.end());
  }
  t.Add(PartRole::kData, std::move(bytes));
  t.Add(PartRole::kTensorScale, ReadResource(root_, a, binding_.ple_table_scale.index));
  t.Upload();
  for (const std::uint64_t count :
       {std::uint64_t{1}, std::uint64_t{16}, std::uint64_t{512}, std::uint64_t{3072}}) {
    SCOPED_TRACE(count);
    std::vector<std::int32_t> ids(count);
    for (std::int32_t& id : ids) {
      id = static_cast<std::int32_t>(rng_() % 3072);
    }
    const Buffer d_ids{std::span<const std::int32_t>(ids)};
    const Buffer out(count * 160 * sizeof(float));
    memory.Launching();
    const auto ran = decode::RunRows({.weight = t.weight(),
                                      .ids = d_ids.as<std::int32_t>(),
                                      .count = count,
                                      .out = out.as<float>(),
                                      .out_stride = 160},
                                     nullptr);
    ASSERT_TRUE(ran.has_value()) << ran.error();
    memory.Ran();
    const std::vector<float> got = out.Get<float>();
    for (std::uint64_t i = 0; i < count; ++i) {
      const std::vector<float> want = t.Row(static_cast<std::uint64_t>(ids[i]));
      for (std::uint64_t j = 0; j < 160; ++j) {
        ASSERT_EQ(std::bit_cast<std::uint32_t>(got[(i * 160) + j]),
                  std::bit_cast<std::uint32_t>(want[j]))
            << "id " << ids[i] << " value " << j;
      }
    }
  }
  memory.Check(t.stored(), t.expanded(), "the n-gram table's rows");
}

// One expert's tensors, and the grouped products over a set of experts.
struct ExpertSet {
  std::vector<std::unique_ptr<Tensor>> gate_up;  // or gate, in GGML's layout
  std::vector<std::unique_ptr<Tensor>> up;       // GGML's layout only
  std::vector<std::unique_ptr<Tensor>> down;
  std::unique_ptr<Buffer> gate_up_table;
  std::unique_ptr<Buffer> up_table;
  std::unique_ptr<Buffer> down_table;
  std::vector<std::unique_ptr<Buffer>> scales;
};

std::unique_ptr<Buffer> Table(const std::vector<std::unique_ptr<Tensor>>& tensors) {
  std::vector<decode::Operand> operands;
  for (const auto& t : tensors) {
    const auto op = decode::OperandOf(t->weight());
    EXPECT_TRUE(op.has_value()) << t->name << ": " << op.error();
    operands.push_back(op.value_or(decode::Operand{}));
  }
  return std::make_unique<Buffer>(std::span<const decode::Operand>(operands));
}

std::vector<float> ScalesOf(const fs::path& root, const ar::Artifact& a, const std::string& name) {
  const auto index = a.FindResource(name);
  EXPECT_TRUE(index.has_value()) << name;
  const std::vector<std::byte> bytes = ReadResource(root, a, index.value_or(0));
  std::vector<float> out(bytes.size() / 4);
  std::memcpy(out.data(), bytes.data(), out.size() * 4);
  return out;
}

std::vector<std::byte> AsBytes(std::span<const float> v) {
  const auto b = std::as_bytes(v);
  return {b.begin(), b.end()};
}

// The CUTLASS import's experts: gate and up stacked (codes, atoms), down,
// each with its tensor scales gathered from the per-projection resources
// into [stacked] order.
ExpertSet LoadCutlassExperts(const fs::path& root, const ar::Artifact& a) {
  ExpertSet set;
  const std::string blk = std::format("blk.{}.", kExpertLayer);
  const std::vector<float> gate_s = ScalesOf(root, a, blk + "ffn_gate_exps.weight_scale_2");
  const std::vector<float> up_s = ScalesOf(root, a, blk + "ffn_up_exps.weight_scale_2");
  const std::vector<float> down_s = ScalesOf(root, a, blk + "ffn_down_exps.weight_scale_2");
  const auto array = [&](const std::string& name) { return a.FindExpertArray(blk + name).value(); };
  const std::uint32_t gu_codes = array("ffn_gate_up_exps.codes");
  const std::uint32_t gu_scales = array("ffn_gate_up_exps.scales");
  const std::uint32_t d_codes = array("ffn_down_exps.codes");
  const std::uint32_t d_scales = array("ffn_down_exps.scales");
  const auto assemble = [&](std::uint32_t codes, std::uint32_t scales, std::uint8_t stacked) {
    const std::vector<ar::V0Part> parts = {
        {PartRole::kCodes, &a.expert_arrays()[codes].repr},
        {PartRole::kBlockScales, &a.expert_arrays()[scales].repr}};
    TensorFormat f =
        ar::Assemble(Encoding::kNvfp4, Layout::kCutlassScaleAtoms, parts, stacked).value();
    f.params.tensor_scale = true;  // gathered below
    return f;
  };
  const TensorFormat gu_format = assemble(gu_codes, gu_scales, 2);
  const TensorFormat d_format = assemble(d_codes, d_scales, 1);
  for (const std::uint32_t e : Experts()) {
    auto gu = std::make_unique<Tensor>();
    gu->name = std::format("{}ffn_gate_up_exps[{}]", blk, e);
    gu->format = gu_format;
    gu->Add(PartRole::kCodes, ReadPlaced(root, a, a.SlicePlacement(gu_codes, e).value()));
    gu->Add(PartRole::kBlockScales, ReadPlaced(root, a, a.SlicePlacement(gu_scales, e).value()));
    gu->Add(PartRole::kTensorScale, AsBytes(std::array{gate_s.at(e), up_s.at(e)}));
    gu->Upload();
    set.gate_up.push_back(std::move(gu));
    auto d = std::make_unique<Tensor>();
    d->name = std::format("{}ffn_down_exps[{}]", blk, e);
    d->format = d_format;
    d->Add(PartRole::kCodes, ReadPlaced(root, a, a.SlicePlacement(d_codes, e).value()));
    d->Add(PartRole::kBlockScales, ReadPlaced(root, a, a.SlicePlacement(d_scales, e).value()));
    d->Add(PartRole::kTensorScale, AsBytes(std::array{down_s.at(e)}));
    d->Upload();
    set.down.push_back(std::move(d));
  }
  set.gate_up_table = Table(set.gate_up);
  set.down_table = Table(set.down);
  return set;
}

// The GGML import's experts: gate, up and down in block_nvfp4, each with
// its own tensor scale.
ExpertSet LoadGgmlExperts(const fs::path& root, const ar::Artifact& a) {
  ExpertSet set;
  const std::string blk = std::format("blk.{}.", kExpertLayer);
  const auto load = [&](const std::string& proj, std::vector<std::unique_ptr<Tensor>>& out) {
    const std::vector<float> s = ScalesOf(root, a, blk + proj + ".weight_scale_2");
    const std::uint32_t index = a.FindExpertArray(blk + proj + ".weight").value();
    TensorFormat f = ar::FormatOf(a.expert_arrays()[index].repr).value();
    f.params.tensor_scale = true;
    for (const std::uint32_t e : Experts()) {
      auto t = std::make_unique<Tensor>();
      t->name = std::format("{}{}[{}]", blk, proj, e);
      t->format = f;
      t->Add(PartRole::kData, ReadPlaced(root, a, a.SlicePlacement(index, e).value()));
      t->Add(PartRole::kTensorScale, AsBytes(std::array{s.at(e)}));
      t->Upload();
      out.push_back(std::move(t));
    }
  };
  load("ffn_gate_exps", set.gate_up);
  load("ffn_up_exps", set.up);
  load("ffn_down_exps", set.down);
  set.gate_up_table = Table(set.gate_up);
  set.up_table = Table(set.up);
  set.down_table = Table(set.down);
  return set;
}

// A routing of `tokens` tokens to kUsed of the experts each, as a worklist
// sorted by expert: pair p reads row pair_rows[p] (token · kUsed + slot
// for the down product's inputs) into row pair_out[p].
struct Routing {
  std::vector<std::array<std::uint32_t, kUsed>> experts;  // per token: indices into Experts()
  std::vector<std::int32_t> offsets;
  std::vector<std::int32_t> token_rows;
  std::vector<std::int32_t> slot_rows;
};
Routing Route(std::uint64_t tokens, std::uint32_t count, std::mt19937& rng) {
  Routing r;
  r.experts.resize(tokens);
  for (auto& chosen : r.experts) {
    std::vector<std::uint32_t> all(count);
    for (std::uint32_t i = 0; i < count; ++i) {
      all[i] = i;
    }
    std::shuffle(all.begin(), all.end(), rng);
    std::copy_n(all.begin(), kUsed, chosen.begin());
  }
  for (std::uint32_t e = 0; e < count; ++e) {
    r.offsets.push_back(static_cast<std::int32_t>(r.token_rows.size()));
    for (std::uint64_t t = 0; t < tokens; ++t) {
      for (std::uint32_t s = 0; s < kUsed; ++s) {
        if (r.experts[t][s] == e) {
          r.token_rows.push_back(static_cast<std::int32_t>(t));
          r.slot_rows.push_back(static_cast<std::int32_t>((t * kUsed) + s));
        }
      }
    }
  }
  r.offsets.push_back(static_cast<std::int32_t>(r.token_rows.size()));
  return r;
}

// One grouped product: y[slot row] = W_e[first + j] x[input row].
std::vector<float> Grouped(const std::vector<std::unique_ptr<Tensor>>& experts, const Buffer& table,
                           std::uint64_t first, std::uint64_t outputs, const Routing& r,
                           bool from_slots, const Buffer& x, std::uint64_t x_rows,
                           std::uint64_t tokens, Memory& memory) {
  const Buffer offsets{std::span<const std::int32_t>(r.offsets)};
  const Buffer rows{std::span<const std::int32_t>(from_slots ? r.slot_rows : r.token_rows)};
  const Buffer out_rows{std::span<const std::int32_t>(r.slot_rows)};
  const std::uint64_t y_rows = tokens * kUsed;
  const Buffer y(y_rows * outputs * sizeof(float));
  const std::uint64_t k = experts[0]->k();
  const decode::GroupedProduct p{.format = experts[0]->format,
                                 .experts = table.as<decode::Operand>(),
                                 .experts_count = experts.size(),
                                 .first = first,
                                 .outputs = outputs,
                                 .offsets = offsets.as<std::int32_t>(),
                                 .pair_rows = rows.as<std::int32_t>(),
                                 .pair_out = out_rows.as<std::int32_t>(),
                                 .pairs = r.token_rows.size(),
                                 .x = x.as<float>(),
                                 .x_rows = x_rows,
                                 .x_stride = k,
                                 .y = y.as<float>(),
                                 .y_rows = y_rows,
                                 .y_stride = outputs,
                                 .rows_hint = r.token_rows.size() / experts.size()};
  memory.Launching();
  const auto ran = decode::RunGrouped(p, nullptr);
  EXPECT_TRUE(ran.has_value()) << ran.error();
  memory.Ran();
  return y.Get<float>();
}

// The grouped output of each checked slot against the F64 product.
void ExpectGrouped(const std::vector<float>& y, const std::vector<std::unique_ptr<Tensor>>& experts,
                   std::uint64_t first, std::uint64_t outputs, const Routing& r,
                   const std::vector<float>& x, bool from_slots, std::uint64_t tokens,
                   std::mt19937& rng, const std::string& what) {
  const std::uint64_t k = experts[0]->k();
  const DecoderId id = decode::DecoderOf(experts[0]->format).value();
  const double tolerance = Tolerance(id, true, k);
  std::vector<std::uint64_t> slots;
  for (std::uint64_t s = 0; s < tokens * kUsed; ++s) {
    slots.push_back(s);
  }
  if (slots.size() > 48) {
    std::shuffle(slots.begin(), slots.end(), rng);
    slots.resize(48);
  }
  double worst = 0;
  for (const std::uint64_t slot : slots) {
    const std::uint64_t t = slot / kUsed;
    const Tensor& w = *experts[r.experts[t][slot % kUsed]];
    const float* in = x.data() + ((from_slots ? slot : t) * k);
    for (std::uint64_t j = 0; j < outputs; ++j) {
      const Dot d = DotOf(w.Row(first + j), in);
      const double got = y[(slot * outputs) + j];
      const double bound = Bound(d, tolerance);
      worst = std::max(worst, std::abs(got - d.sum) / bound);
      ASSERT_LE(std::abs(got - d.sum), bound)
          << what << ": slot " << slot << " output " << j << " got " << got << " want " << d.sum;
    }
  }
  std::printf("%s: largest error %.3f of its bound\n", what.c_str(), worst);
}

TEST_F(Qwen38DecodeProofTest, RoutedExpertsMatchTheReferenceInBothLayouts) {
  Memory memory;
  ExpertSet cutlass = LoadCutlassExperts(root_, *artifact_);
  std::unique_ptr<ar::Artifact> ggml_artifact;
  ExpertSet ggml;
  const fs::path ggml_root = Root("LLMP_QWEN38_PROOF_GGML_ARTIFACT", kGgmlId);
  if (!ggml_root.empty()) {
    auto opened = ar::Artifact::Open(ggml_root);
    ASSERT_TRUE(opened.has_value()) << ggml_root;
    ggml_artifact = std::make_unique<ar::Artifact>(std::move(*opened));
    ggml = LoadGgmlExperts(ggml_root, *ggml_artifact);
  } else {
    std::printf("no GGML-layout import (%s): its experts are not compared\n",
                std::string(kGgmlId).c_str());
  }
  const std::uint64_t f = model::Qwen38Flash().expert_ffn;
  const std::uint64_t w = model::Qwen38Flash().width;
  const auto count = static_cast<std::uint32_t>(Experts().size());
  for (const std::uint64_t tokens : {std::uint64_t{1}, std::uint64_t{16}, std::uint64_t{512}}) {
    SCOPED_TRACE(std::format("tokens {}", tokens));
    const Routing r = Route(tokens, count, rng_);
    const std::vector<float> x = Normal(tokens * w, rng_);
    const Buffer dx{std::span<const float>(x)};
    const std::vector<float> h = Normal(tokens * kUsed * f, rng_);
    const Buffer dh{std::span<const float>(h)};
    const std::vector<float> gate_up = Grouped(cutlass.gate_up, *cutlass.gate_up_table, 0, 2 * f, r,
                                               false, dx, tokens, tokens, memory);
    ExpectGrouped(gate_up, cutlass.gate_up, 0, 2 * f, r, x, false, tokens, rng_,
                  std::format("grouped gate_up (cutlass_scale_atoms) tokens {}", tokens));
    const std::vector<float> down = Grouped(cutlass.down, *cutlass.down_table, 0, w, r, true, dh,
                                            tokens * kUsed, tokens, memory);
    ExpectGrouped(down, cutlass.down, 0, w, r, h, true, tokens, rng_,
                  std::format("grouped down (cutlass_scale_atoms) tokens {}", tokens));
    if (ggml_artifact == nullptr) {
      continue;
    }
    // The first import's experts, each projection its own tensor: the
    // same values, so the same outputs bit for bit.
    const std::vector<float> gate =
        Grouped(ggml.gate_up, *ggml.gate_up_table, 0, f, r, false, dx, tokens, tokens, memory);
    const std::vector<float> up =
        Grouped(ggml.up, *ggml.up_table, 0, f, r, false, dx, tokens, tokens, memory);
    const std::vector<float> ggml_down =
        Grouped(ggml.down, *ggml.down_table, 0, w, r, true, dh, tokens * kUsed, tokens, memory);
    for (std::uint64_t s = 0; s < tokens * kUsed; ++s) {
      for (std::uint64_t j = 0; j < f; ++j) {
        ASSERT_EQ(std::bit_cast<std::uint32_t>(gate[(s * f) + j]),
                  std::bit_cast<std::uint32_t>(gate_up[(s * 2 * f) + j]))
            << "gate slot " << s << " output " << j;
        ASSERT_EQ(std::bit_cast<std::uint32_t>(up[(s * f) + j]),
                  std::bit_cast<std::uint32_t>(gate_up[(s * 2 * f) + f + j]))
            << "up slot " << s << " output " << j;
      }
      for (std::uint64_t j = 0; j < w; ++j) {
        ASSERT_EQ(std::bit_cast<std::uint32_t>(ggml_down[(s * w) + j]),
                  std::bit_cast<std::uint32_t>(down[(s * w) + j]))
            << "down slot " << s << " output " << j;
      }
    }
    std::printf(
        "tokens %llu: the GGML import's grouped gate, up and down equal the CUTLASS "
        "import's bit for bit\n",
        static_cast<unsigned long long>(tokens));
  }
  // One expert as a decode step: its gate and up as a vector GLU, its down
  // as a vector product.
  CheckGlu(*cutlass.gate_up[3], 0, *cutlass.gate_up[3], f, f, memory, rng_);
  CheckLinear(*cutlass.down[3], memory, rng_);
  std::uint64_t stored = 0;
  std::uint64_t expanded = 0;
  for (const ExpertSet* set : {&cutlass, &ggml}) {
    for (const auto* list : {&set->gate_up, &set->up, &set->down}) {
      for (const auto& t : *list) {
        stored += t->stored();
        expanded += t->expanded();
      }
    }
  }
  memory.Check(stored, expanded, std::format("{} routed experts", count));
}

}  // namespace
