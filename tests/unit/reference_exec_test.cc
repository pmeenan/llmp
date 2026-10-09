// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The reference executor (kernels/reference/exec.h): every op of op set v1
// against hand-computed values or an independent formula (the FP64
// references tests/unit/ggml_ext_ops_test.cc holds GGML's CUDA kernels
// to, ported here, and execution/sampling.h); each composite op against
// its primitive decomposition; fused kinds against the primitives they
// name; collectives over simulated devices.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <format>
#include <initializer_list>
#include <limits>
#include <map>
#include <memory>
#include <numbers>
#include <numeric>
#include <random>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "execution/sampling.h"
#include "expected_error.h"
#include "format/encoding.h"
#include "format/layout.h"
#include "format/tensor_format.h"
#include "ir/builder.h"
#include "ir/decompose.h"
#include "ir/graph.h"
#include "ir/op.h"
#include "kernels/reference/decode.h"
#include "kernels/reference/exec.h"

namespace llmp::kernels::reference {
namespace {

using format::Encoding;
using format::Layout;
using ir::TensorId;

constexpr double kInf = std::numeric_limits<double>::infinity();

std::vector<float> Normal(std::uint32_t seed, std::size_t n, float sd = 1.0f) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> d(0.0f, sd);
  std::vector<float> v(n);
  for (float& x : v) {
    x = d(rng);
  }
  return v;
}

std::vector<float> Uniform(std::uint32_t seed, std::size_t n, float lo, float hi) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> d(lo, hi);
  std::vector<float> v(n);
  for (float& x : v) {
    x = d(rng);
  }
  return v;
}

std::size_t N(std::uint64_t n) { return static_cast<std::size_t>(n); }

void ExpectNear(const std::vector<float>& got, const std::vector<double>& want, double tol,
                const std::string& what) {
  ASSERT_EQ(got.size(), want.size()) << what;
  for (std::size_t i = 0; i < got.size(); ++i) {
    if (std::isinf(want[i])) {
      EXPECT_EQ(static_cast<double>(got[i]), want[i]) << what << " at " << i;
      continue;
    }
    EXPECT_NEAR(got[i], want[i], tol * (1.0 + std::fabs(want[i]))) << what << " at " << i;
  }
}

std::vector<double> D(std::initializer_list<double> v) { return v; }

// Bytes of `values` in a single-part format, as the reference encoder writes them.
std::vector<std::byte> Encode(const format::TensorFormat& f, const std::vector<float>& values) {
  const auto parts = format::PartsOf(f);
  EXPECT_TRUE(parts.has_value());
  std::vector<std::byte> bytes(N((*parts)[0].bytes));
  const std::uint64_t per = f.shape.ne[0] * f.shape.ne[1];
  for (std::uint64_t m = 0; m < format::Matrices(f); ++m) {
    const auto e = EncodeRows(f, m, 0, std::span(values).subspan(N(m * per), N(per)), bytes);
    EXPECT_TRUE(e.has_value());
  }
  return bytes;
}

std::vector<float> Decode(const format::TensorFormat& f, const std::vector<std::byte>& bytes) {
  const PartView part{.role = format::PartRole::kData, .bytes = bytes};
  const std::uint64_t per = f.shape.ne[0] * f.shape.ne[1];
  std::vector<float> out(N(per * format::Matrices(f)));
  for (std::uint64_t m = 0; m < format::Matrices(f); ++m) {
    EXPECT_TRUE(DecodeRows(f, std::span(&part, 1), m, 0, f.shape.ne[1],
                           std::span(out).subspan(N(m * per), N(per)))
                    .has_value());
  }
  return out;
}

class ExecTest : public testing::Test {
 protected:
  ir::Builder b;
  std::unique_ptr<ir::Graph> graph;
  std::unique_ptr<Executor> exec;

  TensorId In(Encoding e, std::initializer_list<std::uint64_t> ne) {
    return b.Input(ir::Dense(e, ir::ShapeOf(ne)));
  }
  TensorId F(std::initializer_list<std::uint64_t> ne) { return In(Encoding::kF32, ne); }
  TensorId I(std::initializer_list<std::uint64_t> ne) { return In(Encoding::kI32, ne); }

  bool Finish() {
    auto g = std::move(b).Finish();
    if (!g) {
      ADD_FAILURE() << "graph: " << g.error().detail;
      return false;
    }
    graph = std::make_unique<ir::Graph>(std::move(*g));
    auto e = Executor::Create(*graph);
    if (!e) {
      ADD_FAILURE() << "executor: " << e.error().detail;
      return false;
    }
    exec = std::make_unique<Executor>(std::move(*e));
    return true;
  }
  void Set(TensorId id, const std::vector<float>& v) {
    const auto r = exec->BindValues(id, v);
    EXPECT_TRUE(r.has_value()) << (r ? "" : r.error().detail);
  }
  void SetInts(TensorId id, const std::vector<std::int64_t>& v) {
    const auto r = exec->BindInts(id, v);
    EXPECT_TRUE(r.has_value()) << (r ? "" : r.error().detail);
  }
  void SetBytes(TensorId id, const std::vector<std::byte>& bytes) {
    const PartView part{.role = format::PartRole::kData, .bytes = bytes};
    const auto r = exec->Bind(id, std::span(&part, 1));
    EXPECT_TRUE(r.has_value()) << (r ? "" : r.error().detail);
  }
  bool Run() {
    const auto r = exec->Run();
    if (!r) {
      ADD_FAILURE() << "run: " << r.error().detail;
    }
    return r.has_value();
  }
  std::vector<float> Get(TensorId id) {
    auto v = exec->Values(id);
    EXPECT_TRUE(v.has_value());
    return v ? *v : std::vector<float>{};
  }
  std::vector<std::int64_t> GetInts(TensorId id) {
    auto v = exec->Ints(id);
    EXPECT_TRUE(v.has_value());
    return v ? *v : std::vector<std::int64_t>{};
  }
};

// ---------------------------------------------------------------- structure

TEST_F(ExecTest, ViewsPermutesCopiesJoinsAndTiles) {
  const TensorId x = F({3, 4});
  const TensorId t = b.Cont(b.Transpose(x));
  const TensorId v = b.View(x, ir::ShapeOf({2, 2}), {4, 12, 24, 24}, 4);  // rows 0-1, cols 1-2
  const TensorId c0 = F({2, 2});
  const TensorId cat = b.Concat(c0, b.Cont(v), 1);
  const TensorId rep =
      b.Repeat(b.View(x, ir::ShapeOf({1, 2}), {4, 12, 24, 24}, 0), ir::ShapeOf({3, 4}));
  const TensorId half = b.Convert(x, Encoding::kF16);
  const TensorId cont = b.Cont(x, ir::ShapeOf({6, 2}));
  for (const TensorId o : {t, cat, rep, half, cont}) {
    b.Output(o);
  }
  ASSERT_TRUE(Finish());
  std::vector<float> xs(12);
  for (std::size_t i = 0; i < xs.size(); ++i) {
    xs[i] = static_cast<float>(i) / 3.0f;
  }
  Set(x, xs);
  Set(c0, {100, 101, 102, 103});
  ASSERT_TRUE(Run());
  // t[a, c] = x[c, a]: logical order a + 4c.
  std::vector<double> t_logical(12);
  for (int c = 0; c < 3; ++c) {
    for (int a = 0; a < 4; ++a) {
      t_logical[N(static_cast<std::uint64_t>(a + (4 * c)))] =
          xs[N(static_cast<std::uint64_t>(c + (3 * a)))];
    }
  }
  ExpectNear(Get(t), t_logical, 1e-7, "transpose");
  ExpectNear(Get(cat), D({100, 101, 102, 103, xs[1], xs[2], xs[4], xs[5]}), 1e-7, "concat");
  ExpectNear(
      Get(rep),
      D({xs[0], xs[0], xs[0], xs[3], xs[3], xs[3], xs[0], xs[0], xs[0], xs[3], xs[3], xs[3]}), 1e-7,
      "repeat");
  std::vector<double> want_half;
  for (const float f : xs) {
    want_half.push_back(F16ToF32(F32ToF16(f)));
  }
  ExpectNear(Get(half), want_half, 0.0, "convert");
  ExpectNear(Get(cont), std::vector<double>(xs.begin(), xs.end()), 0.0, "cont");
}

TEST_F(ExecTest, GetRowsReadsDenseAndQuantizedTables) {
  const format::TensorFormat q8{
      .encoding = Encoding::kQ8_0, .layout = Layout::kGgmlBlocks, .shape = ir::ShapeOf({32, 4})};
  const TensorId table = b.Weight(q8, ir::WeightRef{.role = 1});
  const TensorId dense = F({3, 5});
  const TensorId ints = In(Encoding::kI32, {2, 3});
  const TensorId ids = I({3});
  const TensorId small = I({2});
  const TensorId q = b.GetRows(table, ids);
  const TensorId d = b.GetRows(dense, ids);
  const TensorId i = b.GetRows(ints, small);
  b.Output(q);
  b.Output(d);
  b.Output(i);
  ASSERT_TRUE(Finish());
  const std::vector<float> w = Normal(1, 128);
  const std::vector<std::byte> bytes = Encode(q8, w);
  const std::vector<float> decoded = Decode(q8, bytes);
  SetBytes(table, bytes);
  std::vector<float> dv(15);
  std::iota(dv.begin(), dv.end(), 0.0f);
  Set(dense, dv);
  SetInts(ints, {7, 8, 9, 10, 11, 12});
  SetInts(ids, {2, 0, 3});
  SetInts(small, {2, 1});
  ASSERT_TRUE(Run());
  std::vector<double> want;
  for (const int r : {2, 0, 3}) {
    for (int c = 0; c < 32; ++c) {
      want.push_back(decoded[N(static_cast<std::uint64_t>((r * 32) + c))]);
    }
  }
  ExpectNear(Get(q), want, 0.0, "q8_0 rows");
  EXPECT_EQ(graph->tensor(d).format.encoding, Encoding::kF32);
  EXPECT_EQ(exec->Values(d)->at(0), 6.0f);
  EXPECT_EQ(GetInts(i), (std::vector<std::int64_t>{11, 12, 9, 10}));

  SetInts(ids, {2, 5, 3});
  const auto bad = exec->Run();
  ASSERT_FALSE(bad.has_value());
  EXPECT_EQ(bad.error().error, ExecError::kIndex);
}

// ---------------------------------------------------------------- products

TEST_F(ExecTest, ProductsBroadcastAndReadQuantizedWeights) {
  const TensorId a = F({3, 2, 1, 2});
  const TensorId x = F({3, 2, 2, 2});
  const format::TensorFormat q4{
      .encoding = Encoding::kQ4_0, .layout = Layout::kGgmlBlocks, .shape = ir::ShapeOf({32, 3})};
  const TensorId wq = b.Weight(q4, ir::WeightRef{.role = 1});
  const TensorId xq = F({32, 2});
  const TensorId y = b.MatMul(a, x);
  const TensorId yq = b.MatMul(wq, xq);
  b.Output(y);
  b.Output(yq);
  ASSERT_TRUE(Finish());
  const std::vector<float> av = Normal(2, 12);
  const std::vector<float> xv = Normal(3, 24);
  const std::vector<float> wv = Normal(4, 96);
  const std::vector<float> xqv = Normal(5, 64);
  Set(a, av);
  Set(x, xv);
  const std::vector<std::byte> wb = Encode(q4, wv);
  SetBytes(wq, wb);
  Set(xq, xqv);
  ASSERT_TRUE(Run());
  // out[n, m, i2, i3] = Σ_k a[k, n, 0, i3] x[k, m, i2, i3] (a broadcast over i2).
  std::vector<double> want(16);
  for (int i3 = 0; i3 < 2; ++i3) {
    for (int i2 = 0; i2 < 2; ++i2) {
      for (int m = 0; m < 2; ++m) {
        for (int n = 0; n < 2; ++n) {
          double s = 0;
          for (int k = 0; k < 3; ++k) {
            s += static_cast<double>(av[N(static_cast<std::uint64_t>(k + (3 * n) + (6 * i3)))]) *
                 xv[N(static_cast<std::uint64_t>(k + (3 * m) + (6 * i2) + (12 * i3)))];
          }
          want[N(static_cast<std::uint64_t>(n + (2 * m) + (4 * i2) + (8 * i3)))] = s;
        }
      }
    }
  }
  ExpectNear(Get(y), want, 1e-6, "mul_mat");
  const std::vector<float> wd = Decode(q4, wb);
  std::vector<double> want_q(6);
  for (int m = 0; m < 2; ++m) {
    for (int n = 0; n < 3; ++n) {
      double s = 0;
      for (int k = 0; k < 32; ++k) {
        s += static_cast<double>(wd[N(static_cast<std::uint64_t>(k + (32 * n)))]) *
             xqv[N(static_cast<std::uint64_t>(k + (32 * m)))];
      }
      want_q[N(static_cast<std::uint64_t>(n + (3 * m)))] = s;
    }
  }
  ExpectNear(Get(yq), want_q, 1e-6, "q4_0 product");
}

TEST_F(ExecTest, ViewsOfBlockLayoutsReadTheirRowsBlocks) {
  // Expert 1 of a Q8_0 expert array as a matrix, and its last two rows.
  const format::TensorFormat q8{
      .encoding = Encoding::kQ8_0, .layout = Layout::kGgmlBlocks, .shape = ir::ShapeOf({64, 3, 2})};
  const TensorId experts = b.Weight(q8, ir::WeightRef{.role = 1, .expert_array = true});
  const auto nb = b.tensor(experts).nb;
  const TensorId e1 = b.View(experts, ir::ShapeOf({64, 3}), {nb[0], nb[1], nb[2], nb[2]}, nb[2]);
  const TensorId rows = b.View(e1, ir::ShapeOf({64, 2}), {nb[0], nb[1], nb[2], nb[2]}, nb[1]);
  const TensorId x = F({64, 1});
  const TensorId y = b.MatMul(e1, x);
  b.Output(y);
  b.Output(b.Cont(b.Convert(b.Scale(rows, 1.0f), Encoding::kF32)));
  ASSERT_TRUE(Finish());
  const std::vector<float> w = Normal(7, 384);
  const std::vector<std::byte> bytes = Encode(q8, w);
  const std::vector<float> decoded = Decode(q8, bytes);
  SetBytes(experts, bytes);
  const std::vector<float> xv = Normal(8, 64);
  Set(x, xv);
  ASSERT_TRUE(Run());
  std::vector<double> want(3);
  for (std::size_t n = 0; n < 3; ++n) {
    for (std::size_t k = 0; k < 64; ++k) {
      want[n] += static_cast<double>(decoded[192 + (n * 64) + k]) * xv[k];
    }
  }
  ExpectNear(Get(y), want, 1e-6, "expert view product");
  ExpectNear(Get(rows), std::vector<double>(decoded.begin() + 256, decoded.end()), 0, "row view");
}

TEST_F(ExecTest, GroupedProductsSelectEachRowsExperts) {
  const TensorId as = b.Weight(ir::Dense(Encoding::kF32, ir::ShapeOf({2, 3, 3})),
                               ir::WeightRef{.role = 1, .expert_array = true});
  const TensorId x = F({2, 1, 2});
  const TensorId ids = I({2, 2});
  const TensorId y = b.MatMulId(as, x, ids);
  const ir::Outputs sorted = b.ExpertSort(ids, 3);
  b.Output(y);
  b.Output(sorted[0]);
  b.Output(sorted[1]);
  ASSERT_TRUE(Finish());
  std::vector<float> w(18);
  std::iota(w.begin(), w.end(), 1.0f);  // expert e, row n: 6e + 2n + {1, 2}
  Set(as, w);
  Set(x, {1, 10, 2, 20});
  SetInts(ids, {2, 0, 0, 2});
  ASSERT_TRUE(Run());
  // token 0 (1, 10) by experts 2 then 0; token 1 (2, 20) by 0 then 2.
  const auto row = [&](int e, int n, double a, double c) {
    return (w[N(static_cast<std::uint64_t>((6 * e) + (2 * n)))] * a) +
           (w[N(static_cast<std::uint64_t>((6 * e) + (2 * n) + 1))] * c);
  };
  std::vector<double> want;
  for (const auto& [e, a, c] : std::array<std::tuple<int, double, double>, 4>{
           {{2, 1, 10}, {0, 1, 10}, {0, 2, 20}, {2, 2, 20}}}) {
    for (int n = 0; n < 3; ++n) {
      want.push_back(row(e, n, a, c));
    }
  }
  ExpectNear(Get(y), want, 1e-7, "mul_mat_id");
  // Slots t · 2 + u in expert order: expert 0 has (t0, u1) and (t1, u0).
  EXPECT_EQ(GetInts(sorted[0]), (std::vector<std::int64_t>{1, 2, 0, 3}));
  EXPECT_EQ(GetInts(sorted[1]), (std::vector<std::int64_t>{0, 2, 2, 4}));
}

TEST_F(ExecTest, ProductsBroadcastByDivisionAndReadEachSlotsColumn) {
  // A2 = 2 under B2 = 4: b's matrix i reads a's i / 2 (ggml_mul_mat), not
  // i % 2 as an elementwise broadcast would.
  const TensorId a = F({3, 2, 2});
  const TensorId x = F({3, 1, 4});
  // mul_mat_id with B1 = U (the down projection's form): slot u reads b's
  // column u.
  const TensorId as = b.Weight(ir::Dense(Encoding::kF32, ir::ShapeOf({2, 3, 3})),
                               ir::WeightRef{.role = 1, .expert_array = true});
  const TensorId xs = F({2, 2, 2});
  const TensorId ids = I({2, 2});
  const TensorId y = b.MatMul(a, x);
  const TensorId yd = b.MatMulId(as, xs, ids);
  b.Output(y);
  b.Output(yd);
  ASSERT_TRUE(Finish());
  const std::vector<float> av = Normal(12, 12);
  const std::vector<float> xv = Normal(13, 12);
  const std::vector<float> wv = Normal(14, 18);
  const std::vector<float> xsv = Normal(15, 8);
  const std::vector<std::int64_t> iv = {2, 0, 1, 2};
  Set(a, av);
  Set(x, xv);
  Set(as, wv);
  Set(xs, xsv);
  SetInts(ids, iv);
  ASSERT_TRUE(Run());
  std::vector<double> want(8);
  for (std::uint64_t i2 = 0; i2 < 4; ++i2) {
    for (std::uint64_t n = 0; n < 2; ++n) {
      double s = 0;
      for (std::uint64_t k = 0; k < 3; ++k) {
        s += static_cast<double>(av[N(k + (3 * n) + (6 * (i2 / 2)))]) * xv[N(k + (3 * i2))];
      }
      want[N(n + (2 * i2))] = s;
    }
  }
  ExpectNear(Get(y), want, 1e-6, "mul_mat broadcast");
  std::vector<double> want_id(12);
  for (std::uint64_t t = 0; t < 2; ++t) {
    for (std::uint64_t u = 0; u < 2; ++u) {
      const auto e = static_cast<std::uint64_t>(iv[N(u + (2 * t))]);
      for (std::uint64_t n = 0; n < 3; ++n) {
        double s = 0;
        for (std::uint64_t k = 0; k < 2; ++k) {
          s += static_cast<double>(wv[N(k + (2 * n) + (6 * e))]) * xsv[N(k + (2 * u) + (4 * t))];
        }
        want_id[N(n + (3 * u) + (6 * t))] = s;
      }
    }
  }
  ExpectNear(Get(yd), want_id, 1e-6, "mul_mat_id per slot");
}

TEST_F(ExecTest, PreparationQuantizesAsTheEncoderDoes) {
  const TensorId x = F({64, 2});
  const TensorId q = b.Prepare(x, Encoding::kQ8_0);
  b.Output(q);
  ASSERT_TRUE(Finish());
  const std::vector<float> xv = Normal(6, 128);
  Set(x, xv);
  ASSERT_TRUE(Run());
  const format::TensorFormat q8{
      .encoding = Encoding::kQ8_0, .layout = Layout::kGgmlBlocks, .shape = ir::ShapeOf({64, 2})};
  const std::vector<float> want = Decode(q8, Encode(q8, xv));
  ExpectNear(Get(q), std::vector<double>(want.begin(), want.end()), 0.0, "prepare");
}

// ---------------------------------------------------------------- attention

// Attention as op.h defines it, written out.
std::vector<double> NaiveAttention(const std::vector<float>& q, const std::vector<float>& k,
                                   const std::vector<float>& v, const std::vector<float>& mask,
                                   const std::vector<float>& sinks, int d, int nq, int heads,
                                   int kv_heads, int cells, int dv, double scale, double softcap,
                                   double max_bias) {
  std::vector<double> out(N(static_cast<std::uint64_t>(dv * heads * nq)));
  const int group = heads / kv_heads;
  const double n2 = std::exp2(std::floor(std::log2(heads)));
  for (int h = 0; h < heads; ++h) {
    double slope = 1.0;
    if (max_bias > 0) {
      slope = h < n2 ? std::pow(std::exp2(-max_bias / n2), h + 1)
                     : std::pow(std::exp2(-max_bias / 2 / n2), (2 * (h - n2)) + 1);
    }
    for (int i = 0; i < nq; ++i) {
      std::vector<double> s(N(static_cast<std::uint64_t>(cells)));
      double mx = sinks.empty() ? -kInf : sinks[N(static_cast<std::uint64_t>(h))];
      for (int c = 0; c < cells; ++c) {
        double dot = 0;
        for (int j = 0; j < d; ++j) {
          dot += static_cast<double>(q[N(static_cast<std::uint64_t>(j + (d * i) + (d * nq * h)))]) *
                 k[N(static_cast<std::uint64_t>(j + (d * c) + (d * cells * (h / group))))];
        }
        double sc = dot * scale;
        if (softcap > 0) {
          sc = softcap * std::tanh(sc / softcap);
        }
        if (!mask.empty()) {
          sc += slope * mask[N(static_cast<std::uint64_t>(c + (cells * i)))];
        }
        s[N(static_cast<std::uint64_t>(c))] = sc;
        mx = std::max(mx, sc);
      }
      double sum = sinks.empty() ? 0.0 : std::exp(sinks[N(static_cast<std::uint64_t>(h))] - mx);
      for (double& x : s) {
        x = std::isfinite(mx) ? std::exp(x - mx) : 0.0;
        sum += x;
      }
      for (int e = 0; e < dv; ++e) {
        double acc = 0;
        for (int c = 0; c < cells; ++c) {
          acc += s[N(static_cast<std::uint64_t>(c))] *
                 v[N(static_cast<std::uint64_t>(e + (dv * c) + (dv * cells * (h / group))))];
        }
        out[N(static_cast<std::uint64_t>(e + (dv * h) + (dv * heads * i)))] =
            sum > 0 ? acc / sum : 0.0;
      }
    }
  }
  return out;
}

TEST_F(ExecTest, AttentionMatchesItsDefinition) {
  constexpr int kD = 8;
  constexpr int kNq = 3;
  constexpr int kHeads = 4;
  constexpr int kKv = 2;
  constexpr int kCells = 5;
  constexpr int kDv = 6;
  const TensorId q = F({kD, kNq, kHeads});
  const TensorId k = In(Encoding::kF16, {kD, kCells, kKv});
  const TensorId v = In(Encoding::kF16, {kDv, kCells, kKv});
  const TensorId mask = In(Encoding::kF16, {kCells, 4});
  const TensorId sinks = F({kHeads});
  const TensorId plain = b.Attention(q, k, v, mask, {}, ir::AttnParams{.scale = 0.35f});
  const TensorId capped = b.Attention(
      q, k, v, mask, sinks, ir::AttnParams{.scale = 0.35f, .softcap = 2.0f, .sinks = true});
  const TensorId alibi =
      b.Attention(q, k, v, mask, {}, ir::AttnParams{.scale = 0.35f, .max_bias = 8.0f});
  for (const TensorId o : {plain, capped, alibi}) {
    b.Output(o);
  }
  ASSERT_TRUE(Finish());
  const auto half = [](std::vector<float> x) {
    for (float& f : x) {
      f = F16ToF32(F32ToF16(f));
    }
    return x;
  };
  const std::vector<float> qv = Normal(10, kD * kNq * kHeads);
  const std::vector<float> kv = half(Normal(11, kD * kCells * kKv));
  const std::vector<float> vv = half(Normal(12, kDv * kCells * kKv));
  const float ninf = -std::numeric_limits<float>::infinity();
  // Row 2 sees nothing; row 1 sees cells 0-2 with a soft bias on cell 1.
  std::vector<float> mv = {0,    0,    0,    0,    0,    0, -1.5f, 0, ninf, ninf,
                           ninf, ninf, ninf, ninf, ninf, 0, 0,     0, 0,    0};
  const std::vector<float> sv = {0.5f, -1.0f, 2.0f, 0.0f};
  Set(q, qv);
  Set(k, kv);
  Set(v, vv);
  Set(mask, mv);
  Set(sinks, sv);
  ASSERT_TRUE(Run());
  const std::vector<float> m3(mv.begin(), mv.begin() + 15);
  ExpectNear(Get(plain),
             NaiveAttention(qv, kv, vv, m3, {}, kD, kNq, kHeads, kKv, kCells, kDv, 0.35, 0, 0),
             1e-6, "attention");
  ExpectNear(Get(capped),
             NaiveAttention(qv, kv, vv, m3, sv, kD, kNq, kHeads, kKv, kCells, kDv, 0.35, 2.0, 0),
             1e-6, "softcap and sinks");
  ExpectNear(Get(alibi),
             NaiveAttention(qv, kv, vv, m3, {}, kD, kNq, kHeads, kKv, kCells, kDv, 0.35, 0, 8.0),
             1e-6, "alibi");
  // The fully masked row, without a sink, is zero.
  const std::vector<float> got = Get(plain);
  for (int i = 0; i < kDv * kHeads; ++i) {
    EXPECT_EQ(got[N(static_cast<std::uint64_t>((2 * kDv * kHeads) + i))], 0.0f);
  }
}

TEST_F(ExecTest, MasksSeeWhatTheirKindAllows) {
  const TensorId qp = I({2});
  const TensorId cp = I({6});
  const TensorId causal = b.MaskBuild(qp, cp, 3, ir::MaskParams{}, Encoding::kF32);
  const TensorId window = b.MaskBuild(
      qp, cp, 2, ir::MaskParams{.kind = ir::MaskKind::kWindow, .window = 2}, Encoding::kF16);
  const TensorId block =
      b.MaskBuild(qp, cp, 2, ir::MaskParams{.kind = ir::MaskKind::kBlock, .block = 4});
  for (const TensorId o : {causal, window, block}) {
    b.Output(o);
  }
  ASSERT_TRUE(Finish());
  SetInts(qp, {2, 5});
  SetInts(cp, {0, 1, 2, 3, -1, 5});
  ASSERT_TRUE(Run());
  const double x = -kInf;
  ExpectNear(Get(causal), D({0, 0, 0, x, x, x, 0, 0, 0, 0, x, 0, x, x, x, x, x, x}), 0, "causal");
  ExpectNear(Get(window), D({x, 0, 0, x, x, x, x, x, x, x, x, 0}), 0, "window");
  // Block 4: position 2 sees cells to 3; position 5 to 7.
  ExpectNear(Get(block), D({0, 0, 0, 0, x, x, 0, 0, 0, 0, x, 0}), 0, "block");
}

TEST_F(ExecTest, CacheStoresWriteOnlyTheirRows) {
  const TensorId cache =
      b.State(ir::Dense(Encoding::kF16, ir::ShapeOf({4, 5})), ir::StateRef{.tensor = 0});
  const format::TensorFormat q8{
      .encoding = Encoding::kQ8_0, .layout = Layout::kGgmlBlocks, .shape = ir::ShapeOf({32, 4})};
  const TensorId qcache = b.State(q8, ir::StateRef{.tensor = 1});
  const TensorId src = F({4, 2});
  const TensorId qsrc = F({32, 1});
  const TensorId idx = In(Encoding::kI64, {2});
  const TensorId qidx = I({1});
  const TensorId stored = b.CacheStore(cache, src, idx);
  const TensorId qstored = b.CacheStore(qcache, qsrc, qidx);
  // A reader after the store sees it.
  const TensorId sum = b.SumRows(stored);
  b.Output(sum);
  b.Output(qstored);
  ASSERT_TRUE(Finish());
  std::vector<float> cv(20);
  std::iota(cv.begin(), cv.end(), 100.0f);
  Set(cache, cv);
  const std::vector<float> qv = Normal(20, 128);
  const std::vector<std::byte> qbytes = Encode(q8, qv);
  SetBytes(qcache, qbytes);
  Set(src, {1, 2, 3, 4, 5, 6, 7, 8});
  SetInts(idx, {3, 1});
  const std::vector<float> qsv = Normal(21, 32);
  Set(qsrc, qsv);
  SetInts(qidx, {2});
  ASSERT_TRUE(Run());
  std::vector<double> want(cv.begin(), cv.end());
  for (int i = 0; i < 4; ++i) {
    want[N(static_cast<std::uint64_t>(12 + i))] = i + 1;
    want[N(static_cast<std::uint64_t>(4 + i))] = i + 5;
  }
  ExpectNear(Get(stored), want, 0, "stored rows");
  // Rows of the quantized cache not stored keep their bytes exactly.
  const std::span<const std::byte> after = exec->Bytes(qstored);
  ASSERT_EQ(after.size(), qbytes.size());
  const std::size_t row = qbytes.size() / 4;
  for (std::size_t i = 0; i < qbytes.size(); ++i) {
    if (i / row != 2) {
      EXPECT_EQ(after[i], qbytes[i]) << i;
    }
  }
  const format::TensorFormat one{
      .encoding = Encoding::kQ8_0, .layout = Layout::kGgmlBlocks, .shape = ir::ShapeOf({32, 1})};
  const std::vector<std::byte> want_row = Encode(one, qsv);
  for (std::size_t i = 0; i < row; ++i) {
    EXPECT_EQ(after[(2 * row) + i], want_row[i]) << i;
  }
  // The sum of each cell's row after the store.
  const std::vector<float> sums = Get(sum);
  EXPECT_EQ(sums[3], 10.0f);
  EXPECT_EQ(sums[1], 26.0f);
  EXPECT_EQ(sums[0], 100.0f + 101 + 102 + 103);

  SetInts(qidx, {4});
  const auto bad = exec->Run();
  ASSERT_FALSE(bad.has_value());
  EXPECT_EQ(bad.error().error, ExecError::kIndex);
}

TEST_F(ExecTest, IndexScoresSumEachHeadsReluScores) {
  // ggml_ext_ops_test LightningIndexerMatchesTheReference's reference.
  constexpr int kEmbd = 16;
  constexpr int kHeads = 4;
  constexpr int kBatch = 3;
  constexpr int kCells = 7;
  const TensorId q = F({kEmbd, kHeads, kBatch});
  const TensorId k = In(Encoding::kF16, {kEmbd, 1, kCells});
  const TensorId w = F({kHeads, kBatch});
  const TensorId mask = In(Encoding::kF16, {kCells, kBatch});
  const TensorId s = b.IndexScore(q, k, w, mask);
  b.Output(s);
  ASSERT_TRUE(Finish());
  const std::vector<float> qv = Normal(30, kEmbd * kHeads * kBatch);
  std::vector<float> kv = Normal(31, kEmbd * kCells);
  for (float& f : kv) {
    f = F16ToF32(F32ToF16(f));
  }
  const std::vector<float> wv = Normal(32, kHeads * kBatch, 0.25f);
  std::vector<float> mv(kCells * kBatch, 0.0f);
  mv[N(kCells - 1)] = -std::numeric_limits<float>::infinity();
  Set(q, qv);
  Set(k, kv);
  Set(w, wv);
  Set(mask, mv);
  ASSERT_TRUE(Run());
  std::vector<double> want;
  for (int bt = 0; bt < kBatch; ++bt) {
    for (int c = 0; c < kCells; ++c) {
      double score = 0;
      for (int h = 0; h < kHeads; ++h) {
        double dot = 0;
        for (int i = 0; i < kEmbd; ++i) {
          dot += static_cast<double>(
                     qv[N(static_cast<std::uint64_t>((((bt * kHeads) + h) * kEmbd) + i))]) *
                 kv[N(static_cast<std::uint64_t>((c * kEmbd) + i))];
        }
        score += std::max(dot, 0.0) * wv[N(static_cast<std::uint64_t>((bt * kHeads) + h))];
      }
      want.push_back(score + mv[N(static_cast<std::uint64_t>((bt * kCells) + c))]);
    }
  }
  ExpectNear(Get(s), want, 1e-6, "index scores");
}

TEST_F(ExecTest, PoolingMeansOrSoftmaxWeightsEachRun) {
  const TensorId x = F({2, 4});
  const TensorId sc = F({2, 4});
  const TensorId mean = b.Pool(x, {}, ir::PoolParams{.ratio = 2});
  const TensorId soft = b.Pool(x, sc, ir::PoolParams{.ratio = 4, .kind = ir::PoolKind::kSoftmax});
  b.Output(mean);
  b.Output(soft);
  ASSERT_TRUE(Finish());
  Set(x, {1, 10, 3, 30, 5, 50, 7, 70});
  const float ninf = -std::numeric_limits<float>::infinity();
  Set(sc, {0, ninf, 0, ninf, std::log(2.0f), ninf, ninf, ninf});
  ASSERT_TRUE(Run());
  ExpectNear(Get(mean), D({2, 20, 6, 60}), 1e-7, "mean");
  // Channel 0: weights 1, 1, 2, 0 over 4; channel 1 sees nothing: 0.
  ExpectNear(Get(soft), D({(1 + 3 + (2.0 * 5)) / 4.0, 0}), 1e-6, "softmax");
}

// ---------------------------------------------------------------- norms

TEST_F(ExecTest, NormsMatchTheirDefinitions) {
  const TensorId x = F({4, 2});
  const TensorId rms = b.RmsNorm(x, 1e-5f);
  const TensorId l2 = b.L2Norm(x, 1e-5f);
  const TensorId ln = b.LayerNorm(x, 1e-5f);
  const TensorId ch = b.ChannelNorm(x, 1e-12f);
  for (const TensorId o : {rms, l2, ln, ch}) {
    b.Output(o);
  }
  ASSERT_TRUE(Finish());
  const std::vector<float> xv = {1, 2, 3, 4, -2, 0, 2, 4};
  Set(x, xv);
  ASSERT_TRUE(Run());
  std::vector<double> want_rms;
  std::vector<double> want_l2;
  std::vector<double> want_ln;
  for (int r = 0; r < 2; ++r) {
    double sq = 0;
    double mean = 0;
    for (int i = 0; i < 4; ++i) {
      sq += xv[N(static_cast<std::uint64_t>((4 * r) + i))] *
            xv[N(static_cast<std::uint64_t>((4 * r) + i))];
      mean += xv[N(static_cast<std::uint64_t>((4 * r) + i))] / 4.0;
    }
    double var = 0;
    for (int i = 0; i < 4; ++i) {
      var += std::pow(xv[N(static_cast<std::uint64_t>((4 * r) + i))] - mean, 2) / 4.0;
    }
    for (int i = 0; i < 4; ++i) {
      const double e = xv[N(static_cast<std::uint64_t>((4 * r) + i))];
      want_rms.push_back(e / std::sqrt((sq / 4) + 1e-5));
      want_l2.push_back(e / std::sqrt(sq));
      want_ln.push_back((e - mean) / std::sqrt(var + 1e-5));
    }
  }
  ExpectNear(Get(rms), want_rms, 1e-6, "rms");
  ExpectNear(Get(l2), want_l2, 1e-6, "l2");
  ExpectNear(Get(ln), want_ln, 1e-6, "layer");
  // Over ne[1]: each column's two values.
  std::vector<double> want_ch(8);
  for (int i = 0; i < 4; ++i) {
    const double a = xv[N(static_cast<std::uint64_t>(i))];
    const double c = xv[N(static_cast<std::uint64_t>(4 + i))];
    const double n = std::sqrt((a * a) + (c * c));
    want_ch[N(static_cast<std::uint64_t>(i))] = a / n;
    want_ch[N(static_cast<std::uint64_t>(4 + i))] = c / n;
  }
  ExpectNear(Get(ch), want_ch, 1e-6, "channel");
}

// ggml's rotation of one pair (rope.cu:15-44): op.h's angle, each step in
// F32 (one statement each, so nothing contracts into a fused multiply-add),
// then the exact cosine and sine of it.
std::pair<double, double> RopeAngle(float theta_extrap, int pair, float freq_scale,
                                    const std::array<float, 2>& corr, float ext_factor,
                                    float attn_factor, bool forward) {
  const float theta_interp = freq_scale * theta_extrap;
  float theta = theta_interp;
  float mscale = attn_factor;
  if (ext_factor != 0.0f) {
    const float y = (static_cast<float>(pair) - corr[0]) / std::max(0.001f, corr[1] - corr[0]);
    const float ramp = (1.0f - std::min(1.0f, std::max(0.0f, y))) * ext_factor;
    const float kept = theta_interp * (1.0f - ramp);
    const float mixed = theta_extrap * ramp;
    theta = kept + mixed;
    const float magnitude = 0.1f * std::log(1.0f / freq_scale);
    mscale *= 1.0f + magnitude;
  }
  const double m = mscale;
  const double t = theta;
  return {std::cos(t) * m, std::sin(t) * m * (forward ? 1.0 : -1.0)};
}

// op.h's F32 base angle of a pair: (float) pos · (float) s^e, s = (float)
// base^(-2 / dims).
float RopeBase(std::int64_t pos, double exponent, float base, std::uint32_t dims) {
  const auto s = static_cast<float>(
      std::pow(static_cast<double>(base), static_cast<double>(-2.0f / static_cast<float>(dims))));
  const auto f = static_cast<float>(std::pow(static_cast<double>(s), exponent));
  return static_cast<float>(pos) * f;
}

std::array<float, 2> YarnCorr(int n_dims, int ctx, float base, float fast, float slow) {
  const auto dim = [&](float rot) {
    return static_cast<float>(n_dims) *
           std::log(static_cast<float>(ctx) / (rot * 2 * std::numbers::pi_v<float>)) /
           (2 * std::log(base));
  };
  return {std::max(0.0f, std::floor(dim(fast))),
          std::min(static_cast<float>(n_dims) - 1, std::ceil(dim(slow)))};
}

TEST_F(ExecTest, RotationsMatchGgmlsDefinition) {
  // ggml_ext_ops_test RopeWithOffsetsYarnAndSectionsMatchesTheReference's
  // cases, and NEOX with frequency factors.
  struct Case {
    std::uint64_t head;
    std::uint64_t heads;
    std::uint32_t dims;
    std::uint32_t offset;
    ir::RopeMode mode;
    bool forward;
    float base;
    float scale;
    float ext;
    bool factors;
  };
  const std::array<Case, 7> cases = {{
      // NEOX over part of the head from an offset, inverse, with factors;
      // and GGML's MROPE (sections in turn).
      {64, 2, 16, 8, ir::RopeMode::kNeox, false, 10000.0f, 1.0f, 0.0f, true},
      {32, 2, 16, 0, ir::RopeMode::kMulti, true, 1e6f, 1.0f, 0.0f, false},
      {64, 2, 16, 48, ir::RopeMode::kNormal, true, 10000.0f, 1.0f / 16.0f, 1.0f, false},
      {64, 2, 16, 48, ir::RopeMode::kNormal, false, 10000.0f, 1.0f / 16.0f, 1.0f, false},
      {32, 3, 16, 8, ir::RopeMode::kNormal, true, 160000.0f, 1.0f / 16.0f, 1.0f, false},
      {64, 3, 32, 0, ir::RopeMode::kInterleavedMulti, true, 1e7f, 1.0f, 0.0f, false},
      {32, 2, 32, 0, ir::RopeMode::kNeox, true, 10000.0f, 1.0f, 0.0f, true},
  }};
  constexpr std::uint64_t kTokens = 5;
  constexpr int kContext = 65536;
  const std::array<std::int64_t, kTokens> positions = {0, 1, 17, 300, 999};
  const std::array<std::uint32_t, 4> sections = {3, 3, 2, 0};
  std::vector<TensorId> xs;
  std::vector<TensorId> ps;
  std::vector<TensorId> outs;
  std::vector<TensorId> ffs;
  for (const Case& c : cases) {
    const bool multi = c.mode == ir::RopeMode::kInterleavedMulti || c.mode == ir::RopeMode::kMulti;
    xs.push_back(F({c.head, c.heads, kTokens}));
    ps.push_back(I({multi ? 4 * kTokens : kTokens}));
    ffs.push_back(c.factors ? F({c.dims / 2}) : TensorId{});
    outs.push_back(
        b.Rope(xs.back(), ps.back(), ffs.back(),
               ir::RopeParams{.mode = c.mode,
                              .inverse = !c.forward,
                              .dims = c.dims,
                              .offset = c.offset,
                              .original_context = kContext,
                              .base = c.base,
                              .scale = c.scale,
                              .ext_factor = c.ext,
                              .sections = multi ? sections : std::array<std::uint32_t, 4>{}}));
    b.Output(outs.back());
  }
  ASSERT_TRUE(Finish());
  std::vector<std::vector<float>> xv;
  std::vector<std::vector<std::int64_t>> pv;
  std::vector<std::vector<float>> fv;
  for (std::size_t i = 0; i < cases.size(); ++i) {
    const Case& c = cases[i];
    const bool multi = c.mode == ir::RopeMode::kInterleavedMulti || c.mode == ir::RopeMode::kMulti;
    xv.push_back(Normal(static_cast<std::uint32_t>(40 + i), N(c.head * c.heads * kTokens)));
    Set(xs[i], xv.back());
    std::vector<std::int64_t> pos;
    for (int s = 0; s < (multi ? 4 : 1); ++s) {
      for (const std::int64_t p : positions) {
        pos.push_back(s == 0 ? p : p + (7 * s));
      }
    }
    pv.push_back(pos);
    SetInts(ps[i], pos);
    fv.push_back(Uniform(static_cast<std::uint32_t>(50 + i), c.dims / 2, 1.0f, 4.0f));
    if (c.factors) {
      Set(ffs[i], fv.back());
    }
  }
  ASSERT_TRUE(Run());
  for (std::size_t ci = 0; ci < cases.size(); ++ci) {
    const Case& c = cases[ci];
    const bool multi = c.mode == ir::RopeMode::kInterleavedMulti;
    const bool mrope = c.mode == ir::RopeMode::kMulti;
    const bool neox = c.mode != ir::RopeMode::kNormal;
    const std::array<float, 2> corr =
        YarnCorr(static_cast<int>(c.dims), kContext, c.base, 32.0f, 1.0f);
    const int sect = 8;
    std::vector<double> want(xv[ci].begin(), xv[ci].end());
    for (std::uint64_t t = 0; t < kTokens; ++t) {
      for (std::uint64_t h = 0; h < c.heads; ++h) {
        const std::size_t row = N(((t * c.heads) + h) * c.head);
        for (int pair = 0; pair < static_cast<int>(c.dims / 2); ++pair) {
          int component = 0;
          if (mrope) {
            const int sector = pair % sect;  // sections {3, 3, 2, 0} in turn
            component = sector < 3 ? 0 : (sector < 6 ? 1 : 2);
          }
          if (multi) {
            const int sector = pair % sect;
            if (sector % 3 == 1 && sector < 3 * 3) {
              component = 1;
            } else if (sector % 3 == 2 && sector < 3 * 2) {
              component = 2;
            } else if (sector % 3 == 0 && sector < 3 * 3) {
              component = 0;
            } else {
              component = 3;
            }
          }
          float extrap = RopeBase(pv[ci][N(static_cast<std::uint64_t>(component) * kTokens + t)],
                                  pair, c.base, c.dims);
          if (c.factors) {
            extrap /= fv[ci][N(static_cast<std::uint64_t>(pair))];
          }
          const auto [cos_t, sin_t] =
              RopeAngle(extrap, pair, c.scale, corr, c.ext, 1.0f, c.forward);
          const std::size_t i0 =
              row + c.offset + N(static_cast<std::uint64_t>(neox ? pair : 2 * pair));
          const std::size_t i1 = neox ? i0 + (c.dims / 2) : i0 + 1;
          const double x0 = xv[ci][i0];
          const double x1 = xv[ci][i1];
          want[i0] = (x0 * cos_t) - (x1 * sin_t);
          want[i1] = (x0 * sin_t) + (x1 * cos_t);
        }
      }
    }
    ExpectNear(Get(outs[ci]), want, 2e-5, "rope case " + std::to_string(ci));
  }
}

TEST_F(ExecTest, RotationAnglesAreFormedInF32NearAMillion) {
  // op.h's F32 angle at positions near 2^20 (NEOX, 128 dims, base 10000),
  // where its rounding is hundredths of a radian: pairs 1, 2 and 5 of one
  // head rotate (1, 0), so the results are the cosine and sine of the angle
  // formed. The angles are pinned: (float) pos · (float) s^e by hand.
  constexpr std::uint64_t kDims = 128;
  const TensorId x = F({kDims, 1, 2});
  const TensorId pos = I({2});
  const TensorId out =
      b.Rope(x, pos, {}, ir::RopeParams{.mode = ir::RopeMode::kNeox, .dims = kDims});
  b.Output(out);
  ASSERT_TRUE(Finish());
  const std::array<std::int64_t, 2> positions = {1048573, 999983};
  const std::array<int, 3> pairs = {1, 2, 5};
  // Python with struct-packed F32 roundings of each step.
  const std::array<std::array<double, 3>, 2> angles = {{
      {908026.8125, 786318.875, 510621.09375},
      {865949.625, 749881.5, 486959.34375},
  }};
  std::vector<float> xv(N(kDims * 2), 0.0f);
  for (std::size_t t = 0; t < 2; ++t) {
    for (const int pair : pairs) {
      xv[(t * kDims) + N(static_cast<std::uint64_t>(pair))] = 1.0f;
    }
  }
  Set(x, xv);
  SetInts(pos, {positions[0], positions[1]});
  ASSERT_TRUE(Run());
  const std::vector<float> got = Get(out);
  for (std::size_t t = 0; t < 2; ++t) {
    for (std::size_t i = 0; i < pairs.size(); ++i) {
      const int pair = pairs.at(i);
      const double theta = angles.at(t).at(i);
      EXPECT_EQ(static_cast<double>(RopeBase(positions.at(t), pair, 10000.0f, kDims)), theta);
      const std::size_t i0 = (t * kDims) + N(static_cast<std::uint64_t>(pair));
      const std::size_t i1 = i0 + (kDims / 2);
      EXPECT_NEAR(got[i0], std::cos(theta), 1e-6) << t << " pair " << pair;
      EXPECT_NEAR(got[i1], std::sin(theta), 1e-6) << t << " pair " << pair;
      // The exact angle is another rotation: by 0.004 to 0.1 radians here.
      const double exact =
          static_cast<double>(positions.at(t)) * std::pow(10000.0, -2.0 * pair / kDims);
      EXPECT_GT(std::fabs(theta - exact), 3e-3) << t << " pair " << pair;
      if (pair == 5) {
        EXPECT_GT(
            std::max(std::fabs(got[i0] - std::cos(exact)), std::fabs(got[i1] - std::sin(exact))),
            1e-2)
            << t;
      }
    }
  }
}

TEST_F(ExecTest, RotationTablesAndInversesRoundTrip) {
  const TensorId x = F({8, 2, 3});
  const TensorId table = F({2, 4, 3});
  const TensorId pos = I({3});
  const ir::RopeParams neox{.mode = ir::RopeMode::kNeox, .dims = 8, .base = 500.0f};
  ir::RopeParams back = neox;
  back.inverse = true;
  const TensorId round = b.Rope(b.Rope(x, pos, {}, neox), pos, {}, back);
  const TensorId tab =
      b.Rope(x, table, {}, ir::RopeParams{.mode = ir::RopeMode::kNormal, .table = true, .dims = 8});
  b.Output(round);
  b.Output(tab);
  ASSERT_TRUE(Finish());
  const std::vector<float> xv = Normal(60, 48);
  Set(x, xv);
  SetInts(pos, {3, 40, 700});
  std::vector<float> tv;
  for (int t = 0; t < 3; ++t) {
    for (int p = 0; p < 4; ++p) {
      const double a = 0.1 * (t + 1) * (p + 1);
      tv.push_back(static_cast<float>(std::cos(a)));
      tv.push_back(static_cast<float>(std::sin(a)));
    }
  }
  Set(table, tv);
  ASSERT_TRUE(Run());
  ExpectNear(Get(round), std::vector<double>(xv.begin(), xv.end()), 1e-5, "inverse");
  std::vector<double> want(48);
  for (int t = 0; t < 3; ++t) {
    for (int h = 0; h < 2; ++h) {
      for (int p = 0; p < 4; ++p) {
        const std::size_t i = N(static_cast<std::uint64_t>((((t * 2) + h) * 8) + (2 * p)));
        const double c = tv[N(static_cast<std::uint64_t>((((t * 4) + p) * 2)))];
        const double s = tv[N(static_cast<std::uint64_t>((((t * 4) + p) * 2) + 1))];
        want[i] = (xv[i] * c) - (xv[i + 1] * s);
        want[i + 1] = (xv[i] * s) + (xv[i + 1] * c);
      }
    }
  }
  ExpectNear(Get(tab), want, 1e-6, "table");
}

// ---------------------------------------------------------------- elementwise

TEST_F(ExecTest, ElementwiseOpsBroadcastAndApplyTheirFunctions) {
  const TensorId a = F({3, 2});
  const TensorId row = F({3});
  const TensorId col = F({1, 2});
  const TensorId add = b.Add(a, row);
  const TensorId sub = b.Sub(a, col);
  const TensorId mul = b.Mul(a, row);
  const TensorId div = b.Div(a, col);
  const TensorId scale = b.Scale(a, 2.0f, 0.5f);
  const TensorId clamp = b.Clamp(a, -1.0f, 2.0f);
  const TensorId cap = b.Softcap(a, 2.0f);
  const TensorId fill = b.Fill(ir::ShapeOf({2, 2}), 7.5f, Encoding::kBF16);
  for (const TensorId o : {add, sub, mul, div, scale, clamp, cap, fill}) {
    b.Output(o);
  }
  const TensorId u = F({1});
  std::vector<TensorId> unary;
  for (int f = 0; f <= static_cast<int>(ir::UnaryFn::kSgn); ++f) {
    unary.push_back(b.Unary(u, static_cast<ir::UnaryFn>(f)));
    b.Output(unary.back());
  }
  ASSERT_TRUE(Finish());
  Set(a, {1, -2, 3, 4, 0.5f, -6});
  Set(row, {10, 20, 30});
  Set(col, {2, 4});
  Set(u, {1.0f});
  ASSERT_TRUE(Run());
  ExpectNear(Get(add), D({11, 18, 33, 14, 20.5, 24}), 1e-7, "add");
  ExpectNear(Get(sub), D({-1, -4, 1, 0, -3.5, -10}), 1e-7, "sub");
  ExpectNear(Get(mul), D({10, -40, 90, 40, 10, -180}), 1e-7, "mul");
  ExpectNear(Get(div), D({0.5, -1, 1.5, 1, 0.125, -1.5}), 1e-7, "div");
  ExpectNear(Get(scale), D({2.5, -3.5, 6.5, 8.5, 1.5, -11.5}), 1e-7, "scale");
  ExpectNear(Get(clamp), D({1, -1, 2, 2, 0.5, -1}), 1e-7, "clamp");
  std::vector<double> want_cap;
  for (const double v : {1.0, -2.0, 3.0, 4.0, 0.5, -6.0}) {
    want_cap.push_back(2.0 * std::tanh(v / 2.0));
  }
  ExpectNear(Get(cap), want_cap, 1e-6, "softcap");
  ExpectNear(Get(fill), D({7.5, 7.5, 7.5, 7.5}), 0, "fill");
  // silu, gelu (erf), gelu (tanh), sigmoid, tanh, relu, exp, softplus,
  // sqrt, sqr, neg, abs, sgn at 1.
  const std::vector<double> want_u = {0.7310585786300049,
                                      0.8413447460685429,
                                      0.8411919906082768,
                                      0.7310585786300049,
                                      0.7615941559557649,
                                      1.0,
                                      2.718281828459045,
                                      1.3132616875182228,
                                      1.0,
                                      1.0,
                                      -1.0,
                                      1.0,
                                      1.0};
  for (std::size_t f = 0; f < unary.size(); ++f) {
    ExpectNear(Get(unary[f]), {want_u[f]}, 1e-6, "unary " + std::to_string(f));
  }
}

TEST_F(ExecTest, GatedUnitsTakeSplitOrHalvedSources) {
  const TensorId g = F({2, 1});
  const TensorId u = F({2, 1});
  const TensorId both = F({4, 1});
  const TensorId swi = b.Glu(g, u, ir::GluParams{});
  const TensorId geglu = b.Glu(g, u, ir::GluParams{.kind = ir::GluKind::kGeGluTanh});
  const TensorId halves = b.Glu(both, {}, ir::GluParams{});
  const TensorId swapped = b.Glu(both, {}, ir::GluParams{.swapped = true});
  const TensorId clamped =
      b.Glu(g, u, ir::GluParams{.kind = ir::GluKind::kSwiGluClamp, .limit = 1.5f});
  for (const TensorId o : {swi, geglu, halves, swapped, clamped}) {
    b.Output(o);
  }
  ASSERT_TRUE(Finish());
  Set(g, {1.0f, 3.0f});
  Set(u, {2.0f, -4.0f});
  Set(both, {1.0f, 3.0f, 2.0f, -4.0f});
  ASSERT_TRUE(Run());
  const auto silu = [](double x) { return x / (1 + std::exp(-x)); };
  const auto gelu_tanh = [](double x) {
    return 0.5 * x *
           (1 + std::tanh(std::sqrt(2 / std::numbers::pi) * x * (1 + (0.044715 * x * x))));
  };
  ExpectNear(Get(swi), D({silu(1) * 2, silu(3) * -4}), 1e-6, "swiglu");
  ExpectNear(Get(geglu), D({gelu_tanh(1) * 2, gelu_tanh(3) * -4}), 1e-6, "geglu");
  ExpectNear(Get(halves), D({silu(1) * 2, silu(3) * -4}), 1e-6, "halves");
  ExpectNear(Get(swapped), D({silu(2) * 1, silu(-4) * 3}), 1e-6, "swapped");
  ExpectNear(Get(clamped), D({silu(1) * 1.5, silu(1.5) * -1.5}), 1e-6, "clamped");
}

// ---------------------------------------------------------------- reductions

TEST_F(ExecTest, ReductionsAndSelectionsTakeTheLowerIndexAmongEquals) {
  const TensorId x = F({4, 3});
  const TensorId sum = b.SumRows(x);
  const TensorId amax = b.Argmax(x);
  const TensorId greedy = b.Argmax(x, ir::ArgmaxNan::kFirstColumn);
  const TensorId asc = b.Argsort(x, false);
  const TensorId desc = b.Argsort(x, true);
  const TensorId top = b.TopK(x, 2);
  for (const TensorId o : {sum, amax, greedy, asc, desc, top}) {
    b.Output(o);
  }
  ASSERT_TRUE(Finish());
  const float nan = std::numeric_limits<float>::quiet_NaN();
  Set(x, {3, 1, 3, 2, nan, 5, 1, 5, nan, nan, nan, nan});
  ASSERT_TRUE(Run());
  const std::vector<float> s = Get(sum);
  EXPECT_EQ(s[0], 9.0f);
  EXPECT_TRUE(std::isnan(s[1]));
  EXPECT_EQ(GetInts(amax), (std::vector<std::int64_t>{0, 1, 0}));
  // Host greedy (std::max_element): row 1's leading NaN keeps index 0.
  EXPECT_EQ(GetInts(greedy), (std::vector<std::int64_t>{0, 0, 0}));
  EXPECT_EQ(GetInts(asc), (std::vector<std::int64_t>{1, 3, 0, 2, 2, 1, 3, 0, 0, 1, 2, 3}));
  EXPECT_EQ(GetInts(desc), (std::vector<std::int64_t>{0, 2, 3, 1, 1, 3, 2, 0, 0, 1, 2, 3}));
  EXPECT_EQ(GetInts(top), (std::vector<std::int64_t>{0, 2, 1, 3, 0, 1}));
}

TEST_F(ExecTest, SoftmaxScalesMasksSinksAndAlibi) {
  const TensorId x = F({3, 2, 2});
  const TensorId mask = F({3, 2});
  const TensorId sinks = F({2});
  const TensorId plain = b.SoftMax(x, {}, {}, ir::SoftMaxParams{.scale = 0.5f});
  const TensorId masked = b.SoftMax(x, mask, sinks, ir::SoftMaxParams{});
  const TensorId alibi = b.SoftMax(x, mask, {}, ir::SoftMaxParams{.max_bias = 4.0f});
  for (const TensorId o : {plain, masked, alibi}) {
    b.Output(o);
  }
  ASSERT_TRUE(Finish());
  const std::vector<float> xv = {1, 2, 3, 0, 0, 0, -1, 4, 2, 1, 1, 1};
  const float ninf = -std::numeric_limits<float>::infinity();
  const std::vector<float> mv = {0, -1, ninf, ninf, ninf, ninf};
  Set(x, xv);
  Set(mask, mv);
  Set(sinks, {0.5f, -0.5f});
  ASSERT_TRUE(Run());
  const auto softmax = [](std::vector<double> v, double sink, bool has_sink) {
    double mx = has_sink ? sink : -kInf;
    for (const double e : v) {
      mx = std::max(mx, e);
    }
    double sum = has_sink ? std::exp(sink - mx) : 0;
    for (double& e : v) {
      e = std::isfinite(mx) ? std::exp(e - mx) : 0;
      sum += e;
    }
    for (double& e : v) {
      e = sum > 0 ? e / sum : 0;
    }
    return v;
  };
  std::vector<double> want_plain;
  std::vector<double> want_masked;
  std::vector<double> want_alibi;
  // ALiBi over two heads (GGML's: a power of two of heads, 2): slopes
  // 2^(-max_bias / 2 · (h + 1)).
  const std::array<double, 2> slopes = {std::pow(2.0, -4.0 / 2.0), std::pow(2.0, -8.0 / 2.0)};
  for (int h = 0; h < 2; ++h) {
    for (int r = 0; r < 2; ++r) {
      std::vector<double> p;
      std::vector<double> m;
      std::vector<double> a;
      for (int i = 0; i < 3; ++i) {
        const double v = xv[N(static_cast<std::uint64_t>(i + (3 * r) + (6 * h)))];
        const double mk = mv[N(static_cast<std::uint64_t>(i + (3 * r)))];
        p.push_back(v * 0.5);
        m.push_back(v + mk);
        a.push_back(v + (slopes[N(static_cast<std::uint64_t>(h))] * mk));
      }
      for (const double e : softmax(p, 0, false)) {
        want_plain.push_back(e);
      }
      for (const double e : softmax(m, h == 0 ? 0.5 : -0.5, true)) {
        want_masked.push_back(e);
      }
      for (const double e : softmax(a, 0, false)) {
        want_alibi.push_back(e);
      }
    }
  }
  ExpectNear(Get(plain), want_plain, 1e-6, "plain");
  ExpectNear(Get(masked), want_masked, 1e-6, "masked");
  ExpectNear(Get(alibi), want_alibi, 1e-6, "alibi");
}

// ---------------------------------------------------------------- routing

TEST_F(ExecTest, RoutesSelectAndWeightExperts) {
  const TensorId logits = F({4, 2});
  const TensorId bias = F({4});
  const TensorId table = I({2, 3});
  const TensorId tokens = I({2});
  const ir::Outputs soft =
      b.Route(logits, {}, {}, {},
              ir::RouteParams{.normalize = true, .experts = 4, .used = 2, .scale = 2.0f});
  const ir::Outputs sig =
      b.Route(logits, bias, {}, {},
              ir::RouteParams{.kind = ir::RouteKind::kSigmoidTopK, .experts = 4, .used = 2});
  const ir::Outputs ssp = b.Route(
      logits, bias, {}, {},
      ir::RouteParams{
          .kind = ir::RouteKind::kSqrtSoftplusBias, .normalize = true, .experts = 4, .used = 2});
  const ir::Outputs hash = b.Route(
      logits, {}, table, tokens,
      ir::RouteParams{.kind = ir::RouteKind::kHash, .normalize = true, .experts = 4, .used = 2});
  for (const ir::Outputs& o : {soft, sig, ssp, hash}) {
    b.Output(o[0]);
    b.Output(o[1]);
  }
  ASSERT_TRUE(Finish());
  const std::vector<float> lv = {1, 3, 2, 3, 0, -1, 0.5f, 0.25f};
  Set(logits, lv);
  Set(bias, {0, 0, 5, 0});
  SetInts(table, {3, 0, 1, 2, 2, 1});
  SetInts(tokens, {2, 0});
  ASSERT_TRUE(Run());
  // Softmax: token 0's top two are experts 1 and 3 (equal; the lower
  // first), normalized to halves, times 2.
  EXPECT_EQ(GetInts(soft[0]), (std::vector<std::int64_t>{1, 3, 2, 3}));
  const double e2 = std::exp(0.5);
  const double e3 = std::exp(0.25);
  ExpectNear(Get(soft[1]), D({1, 1, 2 * e2 / (e2 + e3), 2 * e3 / (e2 + e3)}), 1e-6, "softmax");
  // Sigmoid with a bias: expert 2 is selected first, weighted unbiased.
  EXPECT_EQ(GetInts(sig[0]), (std::vector<std::int64_t>{2, 1, 2, 3}));
  const auto sigm = [](double x) { return 1 / (1 + std::exp(-x)); };
  ExpectNear(Get(sig[1]), D({sigm(2), sigm(3), sigm(0.5), sigm(0.25)}), 1e-6, "sigmoid");
  const auto ssq = [](double x) { return std::sqrt(std::log1p(std::exp(x))); };
  EXPECT_EQ(GetInts(ssp[0]), (std::vector<std::int64_t>{2, 1, 2, 3}));
  ExpectNear(Get(ssp[1]),
             D({ssq(2) / (ssq(2) + ssq(3)), ssq(3) / (ssq(2) + ssq(3)),
                ssq(0.5) / (ssq(0.5) + ssq(0.25)), ssq(0.25) / (ssq(0.5) + ssq(0.25))}),
             1e-6, "sqrt softplus");
  // Table rows: token 2's (2, 1), token 0's (3, 0).
  EXPECT_EQ(GetInts(hash[0]), (std::vector<std::int64_t>{2, 1, 3, 0}));
  ExpectNear(Get(hash[1]),
             D({ssq(2) / (ssq(2) + ssq(3)), ssq(3) / (ssq(2) + ssq(3)),
                ssq(0.25) / (ssq(0.25) + ssq(0)), ssq(0) / (ssq(0.25) + ssq(0))}),
             1e-6, "table");
}

TEST_F(ExecTest, ExpertReductionScalesWeighsAndSumsSlots) {
  const TensorId x = F({2, 2, 1});
  const TensorId w = F({2, 1});
  const TensorId ids = I({2, 1});
  const TensorId scale = F({3});
  const TensorId plain = b.ExpertReduce(x, w, {}, {});
  const TensorId scaled = b.ExpertReduce(x, w, ids, scale);
  b.Output(plain);
  b.Output(scaled);
  ASSERT_TRUE(Finish());
  Set(x, {1, 2, 10, 20});
  Set(w, {0.5f, 0.25f});
  SetInts(ids, {2, 0});
  Set(scale, {3, 5, 7});
  ASSERT_TRUE(Run());
  ExpectNear(Get(plain), D({0.5 + 2.5, 1 + 5}), 1e-7, "plain");
  ExpectNear(Get(scaled), D({(1 * 7 * 0.5) + (10 * 3 * 0.25), (2 * 7 * 0.5) + (20 * 3 * 0.25)}),
             1e-7, "scaled");
}

// ---------------------------------------------------------------- recurrences

TEST_F(ExecTest, CausalConvolutionMatchesItsDefinition) {
  constexpr std::uint64_t kChannels = 6;
  constexpr std::uint64_t kConv = 4;
  constexpr std::uint64_t kTokens = 5;
  const TensorId sx = F({kConv - 1 + kTokens, kChannels, 1});
  const TensorId c = F({kConv, kChannels});
  const TensorId y = b.SsmConv(sx, c);
  b.Output(y);
  ASSERT_TRUE(Finish());
  const std::uint64_t window = kConv - 1 + kTokens;
  const std::vector<float> xv = Normal(70, N(window * kChannels));
  const std::vector<float> wv = Normal(71, N(kConv * kChannels), 0.5f);
  Set(sx, xv);
  Set(c, wv);
  ASSERT_TRUE(Run());
  std::vector<double> want(N(kChannels * kTokens));
  for (std::uint64_t t = 0; t < kTokens; ++t) {
    for (std::uint64_t ch = 0; ch < kChannels; ++ch) {
      double sum = 0;
      for (std::uint64_t j = 0; j < kConv; ++j) {
        sum += static_cast<double>(xv[N((ch * window) + t + j)]) * wv[N((ch * kConv) + j)];
      }
      want[N((t * kChannels) + ch)] = sum;
    }
  }
  ExpectNear(Get(y), want, 1e-6, "ssm_conv");
}

// ggml_ext_ops_test GatedDeltaRuleMatchesTheReference's recurrence, with a
// gate per row for the KDA form.
std::pair<std::vector<double>, std::vector<double>> DeltaRule(
    const std::vector<float>& q, const std::vector<float>& k, const std::vector<float>& v,
    const std::vector<float>& g, const std::vector<float>& beta, const std::vector<float>& state,
    std::uint64_t s, std::uint64_t hk, std::uint64_t hv, std::uint64_t tokens, bool kda) {
  std::vector<double> attn(N(s * hv * tokens));
  std::vector<double> final_state(N(s * s * hv));
  const double scale = 1.0 / std::sqrt(static_cast<double>(s));
  for (std::uint64_t h = 0; h < hv; ++h) {
    std::vector<double> st(N(s * s));  // st[col * s + i] = S[i][col]
    for (std::size_t i = 0; i < st.size(); ++i) {
      st[i] = state[N(h * s * s) + i];
    }
    const std::uint64_t qh = h % hk;
    for (std::uint64_t t = 0; t < tokens; ++t) {
      const double b = beta[N((t * hv) + h)];
      const std::size_t qk = N(((t * hk) + qh) * s);
      const std::size_t vr = N(((t * hv) + h) * s);
      for (std::uint64_t col = 0; col < s; ++col) {
        double kv = 0;
        for (std::uint64_t i = 0; i < s; ++i) {
          const double d = std::exp(
              static_cast<double>(kda ? g[N((((t * hv) + h) * s) + i)] : g[N((t * hv) + h)]));
          st[N((col * s) + i)] *= d;
          kv += st[N((col * s) + i)] * k[qk + N(i)];
        }
        const double delta = (v[vr + N(col)] - kv) * b;
        double a = 0;
        for (std::uint64_t i = 0; i < s; ++i) {
          double& cell = st[N((col * s) + i)];
          cell += k[qk + N(i)] * delta;
          a += cell * q[qk + N(i)];
        }
        attn[vr + N(col)] = a * scale;
      }
    }
    for (std::size_t i = 0; i < st.size(); ++i) {
      final_state[N(h * s * s) + i] = st[i];
    }
  }
  return {attn, final_state};
}

TEST_F(ExecTest, GatedDeltaRuleMatchesTheRecurrence) {
  constexpr std::uint64_t kS = 8;
  constexpr std::uint64_t kHk = 2;
  constexpr std::uint64_t kHv = 4;
  constexpr std::uint64_t kTokens = 3;
  const TensorId q = F({kS, kHk, kTokens, 1});
  const TensorId k = F({kS, kHk, kTokens, 1});
  const TensorId v = F({kS, kHv, kTokens, 1});
  const TensorId g = F({1, kHv, kTokens, 1});
  const TensorId gk = F({kS, kHv, kTokens, 1});
  const TensorId beta = F({1, kHv, kTokens, 1});
  const TensorId state = F({kS, kS, kHv, 1});
  const ir::Outputs scalar = b.GatedDeltaNet(q, k, v, g, beta, state);
  const ir::Outputs kda = b.GatedDeltaNet(q, k, v, gk, beta, state);
  for (const ir::Outputs& o : {scalar, kda}) {
    b.Output(o[0]);
    b.Output(o[1]);
  }
  ASSERT_TRUE(Finish());
  const std::vector<float> qv = Normal(80, N(kS * kHk * kTokens), 0.3f);
  const std::vector<float> kv = Normal(81, N(kS * kHk * kTokens), 0.3f);
  const std::vector<float> vv = Normal(82, N(kS * kHv * kTokens));
  const std::vector<float> gv = Uniform(83, N(kHv * kTokens), -2.0f, -0.01f);
  const std::vector<float> gkv = Uniform(84, N(kS * kHv * kTokens), -2.0f, -0.01f);
  const std::vector<float> bv = Uniform(85, N(kHv * kTokens), 0.0f, 1.0f);
  const std::vector<float> sv = Normal(86, N(kS * kS * kHv), 0.1f);
  Set(q, qv);
  Set(k, kv);
  Set(v, vv);
  Set(g, gv);
  Set(gk, gkv);
  Set(beta, bv);
  Set(state, sv);
  ASSERT_TRUE(Run());
  const auto want = DeltaRule(qv, kv, vv, gv, bv, sv, kS, kHk, kHv, kTokens, false);
  ExpectNear(Get(scalar[0]), want.first, 1e-5, "attention");
  ExpectNear(Get(scalar[1]), want.second, 1e-5, "state");
  const auto want_kda = DeltaRule(qv, kv, vv, gkv, bv, sv, kS, kHk, kHv, kTokens, true);
  ExpectNear(Get(kda[0]), want_kda.first, 1e-5, "kda attention");
  ExpectNear(Get(kda[1]), want_kda.second, 1e-5, "kda state");
}

TEST_F(ExecTest, ConvolutionsAndUpsamplingMatchHandValues) {
  const TensorId x = F({3, 3, 1, 1});
  const TensorId k = F({2, 2, 1, 2});
  const TensorId bias = F({2});
  const TensorId ones = F({2, 2, 1, 1});
  const TensorId k3 = F({3, 3, 1, 1});
  const TensorId conv = b.Conv2d(k, x, bias, ir::Conv2dParams{});
  const TensorId padded = b.Conv2d(k3, ones, {}, ir::Conv2dParams{.padding = 1});
  const TensorId strided = b.Conv2d(k3, x, {}, ir::Conv2dParams{.stride = 2, .padding = 1});
  const TensorId up = b.Upsample(ones, 2);
  for (const TensorId o : {conv, padded, strided, up}) {
    b.Output(o);
  }
  ASSERT_TRUE(Finish());
  Set(x, {1, 2, 3, 4, 5, 6, 7, 8, 9});
  Set(k, {1, 1, 1, 1, 1, 0, 0, -1});
  Set(bias, {0.5f, -0.5f});
  Set(ones, {1, 2, 3, 4});
  Set(k3, std::vector<float>(9, 1.0f));
  ASSERT_TRUE(Run());
  ExpectNear(Get(conv), D({12.5, 16.5, 24.5, 28.5, -4.5, -4.5, -4.5, -4.5}), 1e-7, "conv");
  ExpectNear(Get(padded), D({10, 10, 10, 10}), 1e-7, "padded");
  // Stride 2, padding 1 over 3x3: outputs at (0,0), (2,0), (0,2), (2,2).
  ExpectNear(Get(strided), D({1 + 2 + 4 + 5, 2 + 3 + 5 + 6, 4 + 5 + 7 + 8, 5 + 6 + 8 + 9}), 1e-7,
             "strided");
  ExpectNear(Get(up), D({1, 1, 2, 2, 1, 1, 2, 2, 3, 3, 4, 4, 3, 3, 4, 4}), 0, "upsample");
}

// ---------------------------------------------------------------- hyper-connections

TEST_F(ExecTest, HyperConnectionsMatchDeepSeeksAndQwensDefinitions) {
  constexpr std::uint64_t kHc = 4;
  constexpr std::uint64_t kW = 8;
  constexpr std::uint64_t kT = 3;
  constexpr float kEps = 1e-6f;
  constexpr std::uint32_t kIter = 20;
  const TensorId mixes = F({24, kT});
  const TensorId scale = F({3});
  const TensorId base = F({24});
  const TensorId x = F({kW, kHc, kT});
  const TensorId w = F({kHc, kT});
  const TensorId gate = F({kW, kHc, kT});
  const TensorId y = F({kW, kT});
  const TensorId post = F({kHc, kT});
  const TensorId mix = F({kHc, kHc, kT});
  const ir::HcParams hc{.streams = kHc};
  const TensorId comb = b.HcMix(
      mixes, scale, base,
      ir::HcParams{
          .mix = ir::HcMixKind::kSinkhorn, .streams = kHc, .iterations = kIter, .eps = kEps});
  const TensorId weights = b.HcMix(mixes, scale, base, ir::HcParams{.streams = kHc, .eps = kEps});
  const TensorId pre = b.HcPre(x, w, hc);
  const TensorId gated =
      b.HcPre(x, gate, ir::HcParams{.fold = ir::HcFold::kGated, .streams = kHc, .scale = 0.25f});
  const TensorId spread = b.HcPost(y, x, post, mix, hc);
  const TensorId added = b.HcPost(y, x, post, {}, hc);
  for (const TensorId o : {comb, weights, pre, gated, spread, added}) {
    b.Output(o);
  }
  ASSERT_TRUE(Finish());
  const std::vector<float> mv = Normal(90, 24 * kT);
  const std::vector<float> sv = {0.5f, 0.7f, 0.9f};
  const std::vector<float> bv = Normal(91, 24, 0.2f);
  const std::vector<float> xv = Normal(92, N(kW * kHc * kT));
  const std::vector<float> wv = Normal(93, N(kHc * kT));
  const std::vector<float> gv = Normal(94, N(kW * kHc * kT));
  const std::vector<float> yv = Normal(95, N(kW * kT));
  const std::vector<float> pv = Normal(96, N(kHc * kT));
  const std::vector<float> cv = Normal(97, N(kHc * kHc * kT));
  Set(mixes, mv);
  Set(scale, sv);
  Set(base, bv);
  Set(x, xv);
  Set(w, wv);
  Set(gate, gv);
  Set(y, yv);
  Set(post, pv);
  Set(mix, cv);
  ASSERT_TRUE(Run());
  // ggml_ext_ops_test HyperConnectionsMatchTheReference's Sinkhorn.
  std::vector<double> want_comb;
  std::vector<double> want_weights;
  for (std::uint64_t t = 0; t < kT; ++t) {
    std::array<double, 16> m{};
    for (std::size_t src = 0; src < 4; ++src) {
      double max = -kInf;
      for (std::size_t dst = 0; dst < 4; ++dst) {
        const std::size_t idx = dst + (4 * src);
        m[idx] = (mv[N(t * 24) + 8 + idx] * static_cast<double>(sv[2])) + bv[8 + idx];
        max = std::max(max, m[idx]);
      }
      double sum = 0;
      for (std::size_t dst = 0; dst < 4; ++dst) {
        m[dst + (4 * src)] = std::exp(m[dst + (4 * src)] - max);
        sum += m[dst + (4 * src)];
      }
      for (std::size_t dst = 0; dst < 4; ++dst) {
        m[dst + (4 * src)] = (m[dst + (4 * src)] / sum) + kEps;
      }
    }
    const auto cols = [&m] {
      for (std::size_t dst = 0; dst < 4; ++dst) {
        double sum = kEps;
        for (std::size_t src = 0; src < 4; ++src) {
          sum += m[dst + (4 * src)];
        }
        for (std::size_t src = 0; src < 4; ++src) {
          m[dst + (4 * src)] /= sum;
        }
      }
    };
    const auto rows = [&m] {
      for (std::size_t src = 0; src < 4; ++src) {
        double sum = kEps;
        for (std::size_t dst = 0; dst < 4; ++dst) {
          sum += m[dst + (4 * src)];
        }
        for (std::size_t dst = 0; dst < 4; ++dst) {
          m[dst + (4 * src)] /= sum;
        }
      }
    };
    cols();
    for (std::uint32_t i = 1; i < kIter; ++i) {
      rows();
      cols();
    }
    want_comb.insert(want_comb.end(), m.begin(), m.end());
    const auto sig = [](double v) { return 1 / (1 + std::exp(-v)); };
    for (std::size_t d = 0; d < 4; ++d) {
      want_weights.push_back(sig((mv[N(t * 24) + d] * static_cast<double>(sv[0])) + bv[d]) + kEps);
    }
    for (std::size_t d = 0; d < 4; ++d) {
      want_weights.push_back(2 *
                             sig((mv[N(t * 24) + 4 + d] * static_cast<double>(sv[1])) + bv[4 + d]));
    }
  }
  ExpectNear(Get(comb), want_comb, 1e-6, "sinkhorn");
  ExpectNear(Get(weights), want_weights, 1e-6, "pre and post");
  std::vector<double> want_pre;
  std::vector<double> want_gated;
  for (std::uint64_t t = 0; t < kT; ++t) {
    for (std::uint64_t i = 0; i < kW; ++i) {
      double s = 0;
      double gs = 0;
      for (std::uint64_t h = 0; h < kHc; ++h) {
        const double xe = xv[N((((t * kHc) + h) * kW) + i)];
        s += xe * wv[N((t * kHc) + h)];
        gs += xe / (1 + std::exp(-static_cast<double>(gv[N((((t * kHc) + h) * kW) + i)])));
      }
      want_pre.push_back(s);
      want_gated.push_back(0.25 * gs);
    }
  }
  ExpectNear(Get(pre), want_pre, 1e-6, "pre");
  ExpectNear(Get(gated), want_gated, 1e-6, "gated pre");
  std::vector<double> want_post(N(kW * kHc * kT));
  std::vector<double> want_added(N(kW * kHc * kT));
  for (std::uint64_t t = 0; t < kT; ++t) {
    for (std::uint64_t dst = 0; dst < kHc; ++dst) {
      for (std::uint64_t i = 0; i < kW; ++i) {
        const double base_v = static_cast<double>(yv[N((t * kW) + i)]) * pv[N((t * kHc) + dst)];
        double sum = base_v;
        for (std::uint64_t src = 0; src < kHc; ++src) {
          sum += static_cast<double>(xv[N((((t * kHc) + src) * kW) + i)]) *
                 cv[N((t * 16) + dst + (4 * src))];
        }
        want_post[N((((t * kHc) + dst) * kW) + i)] = sum;
        want_added[N((((t * kHc) + dst) * kW) + i)] = base_v + xv[N((((t * kHc) + dst) * kW) + i)];
      }
    }
  }
  ExpectNear(Get(spread), want_post, 1e-6, "post");
  ExpectNear(Get(added), want_added, 1e-6, "post without comb");
}

// ---------------------------------------------------------------- sampling

// DeepSeek V4's hc_split_sinkhorn in torch's terms, written independently
// of the executor: comb [src, dst] (dst innermost, as unflatten(-1, (hc,
// hc)) lays it out), softmax over dst, + eps, then divided by the sum over
// src (+ eps), then iterations - 1 times over dst and over src. A few
// iterations pin the softmax's axis, the order and the count, which 20
// iterations' converged result does not.
std::vector<double> SinkhornReference(const std::vector<float>& mixes,
                                      const std::vector<float>& scale,
                                      const std::vector<float>& base, std::uint64_t hc,
                                      std::uint64_t t_count, std::uint32_t iterations, double eps) {
  const std::uint64_t m = (2 + hc) * hc;
  std::vector<double> out;
  for (std::uint64_t t = 0; t < t_count; ++t) {
    std::vector<std::vector<double>> c(N(hc), std::vector<double>(N(hc)));
    for (std::uint64_t src = 0; src < hc; ++src) {
      double max = -std::numeric_limits<double>::infinity();
      for (std::uint64_t dst = 0; dst < hc; ++dst) {
        const std::uint64_t at = (2 * hc) + (src * hc) + dst;
        c[src][dst] = (static_cast<double>(mixes[N((t * m) + at)]) * scale[2]) + base[N(at)];
        max = std::max(max, c[src][dst]);
      }
      double sum = 0.0;
      for (double& v : c[src]) {
        v = std::exp(v - max);
        sum += v;
      }
      for (double& v : c[src]) {
        v = (v / sum) + eps;
      }
    }
    const auto over_src = [&] {  // comb.sum(-2)
      for (std::uint64_t dst = 0; dst < hc; ++dst) {
        double sum = eps;
        for (std::uint64_t src = 0; src < hc; ++src) {
          sum += c[src][dst];
        }
        for (std::uint64_t src = 0; src < hc; ++src) {
          c[src][dst] /= sum;
        }
      }
    };
    const auto over_dst = [&] {  // comb.sum(-1)
      for (std::uint64_t src = 0; src < hc; ++src) {
        double sum = eps;
        for (const double v : c[src]) {
          sum += v;
        }
        for (double& v : c[src]) {
          v /= sum;
        }
      }
    };
    over_src();
    for (std::uint32_t i = 1; i < iterations; ++i) {
      over_dst();
      over_src();
    }
    for (std::uint64_t src = 0; src < hc; ++src) {
      for (std::uint64_t dst = 0; dst < hc; ++dst) {
        out.push_back(c[src][dst]);  // IR [dst, src, t]: dst innermost
      }
    }
  }
  return out;
}

TEST_F(ExecTest, SinkhornAtFewIterationsAndTheHcNormsMatchTheirDefinitions) {
  constexpr std::uint64_t kHc = 4;
  constexpr std::uint64_t kW = 8;
  constexpr std::uint64_t kT = 3;
  constexpr double kEps = 1e-2;  // large enough that its placement shows
  const TensorId mixes = F({24, kT});
  const TensorId scale = F({3});
  const TensorId base = F({24});
  const TensorId x = F({kW, kHc, kT});
  const TensorId wn = F({kW * kHc});
  const TensorId y = F({kW, kT});
  const TensorId post = F({kHc, kT});
  const TensorId mix = F({kHc, kHc, kT});
  std::array<TensorId, 3> combs{};
  for (std::uint32_t i = 0; i < 3; ++i) {
    combs.at(i) = b.HcMix(mixes, scale, base,
                          ir::HcParams{.mix = ir::HcMixKind::kSinkhorn,
                                       .streams = kHc,
                                       .iterations = i + 1,
                                       .eps = static_cast<float>(kEps)});
    b.Output(combs.at(i));
  }
  const TensorId norm =
      b.Fused(ir::FusedKind::kHcNorm, std::array{x, wn}, ir::NormParams{.eps = 1e-6f})[0];
  const ir::Outputs pn = b.Fused(ir::FusedKind::kHcNormPost, std::array{y, x, post, mix},
                                 ir::HcNormPostParams{.eps = 1e-6f, .streams = kHc});
  const TensorId spread = b.HcPost(y, x, post, mix, ir::HcParams{.streams = kHc});
  b.Output(norm);
  b.Output(pn[0]);
  b.Output(pn[1]);
  b.Output(spread);
  ASSERT_TRUE(Finish());
  const std::vector<float> mv = Normal(190, 24 * kT);
  const std::vector<float> sv = {0.5f, 0.7f, 0.9f};
  const std::vector<float> bv = Normal(191, 24, 0.5f);
  const std::vector<float> xv = Normal(192, N(kW * kHc * kT));
  const std::vector<float> wv = Normal(193, N(kW * kHc));
  const std::vector<float> yv = Normal(194, N(kW * kT));
  const std::vector<float> pv = Normal(195, N(kHc * kT));
  const std::vector<float> cv = Normal(196, N(kHc * kHc * kT));
  Set(mixes, mv);
  Set(scale, sv);
  Set(base, bv);
  Set(x, xv);
  Set(wn, wv);
  Set(y, yv);
  Set(post, pv);
  Set(mix, cv);
  ASSERT_TRUE(Run());
  for (std::uint32_t i = 0; i < 3; ++i) {
    ExpectNear(Get(combs.at(i)), SinkhornReference(mv, sv, bv, kHc, kT, i + 1, kEps), 1e-6,
               std::format("sinkhorn at {} iterations", i + 1));
  }
  // Qwen3.8's norm: each stream over W, then the weight over W · hc.
  std::vector<double> want(N(kW * kHc * kT));
  for (std::uint64_t t = 0; t < kT; ++t) {
    for (std::uint64_t h = 0; h < kHc; ++h) {
      double sq = 0.0;
      for (std::uint64_t i = 0; i < kW; ++i) {
        const double v = xv[N(i + (kW * (h + (kHc * t))))];
        sq += v * v;
      }
      const double r = 1.0 / std::sqrt((sq / static_cast<double>(kW)) + 1e-6);
      for (std::uint64_t i = 0; i < kW; ++i) {
        const std::size_t at = N(i + (kW * (h + (kHc * t))));
        want[at] = xv[at] * r * wv[N(i + (kW * h))];
      }
    }
  }
  ExpectNear(Get(norm), want, 1e-6, "hc_norm");
  // DeepSeek V4's post and flat norm: over all W · hc of a token, no weight.
  const std::vector<float> p = Get(spread);
  ExpectNear(Get(pn[0]), std::vector<double>(p.begin(), p.end()), 1e-6, "hc_norm_post's post");
  std::vector<double> flat(p.size());
  for (std::uint64_t t = 0; t < kT; ++t) {
    double sq = 0.0;
    for (std::uint64_t i = 0; i < kW * kHc; ++i) {
      sq += static_cast<double>(p[N((t * kW * kHc) + i)]) * p[N((t * kW * kHc) + i)];
    }
    const double r = 1.0 / std::sqrt((sq / static_cast<double>(kW * kHc)) + 1e-6);
    for (std::uint64_t i = 0; i < kW * kHc; ++i) {
      flat[N((t * kW * kHc) + i)] = p[N((t * kW * kHc) + i)] * r;
    }
  }
  ExpectNear(Get(pn[1]), flat, 1e-6, "hc_norm_post's flat norm");
}

TEST_F(ExecTest, SamplingIsTheHostSamplers) {
  constexpr std::uint64_t kVocab = 50;
  const TensorId logits = F({kVocab, 3});
  const TensorId keys = In(Encoding::kI64, {3, 3});
  const TensorId greedy = b.Sample(logits, {}, ir::SampleParams{});
  const ir::SampleParams sp{.kind = ir::SampleKind::kSeeded,
                            .temperature = 0.8f,
                            .top_k = 20,
                            .top_p = 0.9f,
                            .min_p = 0.01f};
  const TensorId seeded = b.Sample(logits, keys, sp);
  b.Output(greedy);
  b.Output(seeded);
  ASSERT_TRUE(Finish());
  std::vector<float> lv = Normal(100, N(kVocab * 3));
  lv[7] = lv[9] = 10.0f;  // a tie in row 0: the lower id
  Set(logits, lv);
  // Full 64-bit seeds (past double's 2^53, and one negative as I64): the
  // keys reach the sampler exactly.
  const std::vector<std::int64_t> kv = {
      1, 2, 3, static_cast<std::int64_t>(0x9E3779B97F4A7C15ULL), 2, 4, (std::int64_t{1} << 60) + 1,
      0, 7};
  SetInts(keys, kv);
  ASSERT_TRUE(Run());
  const std::vector<std::int64_t> g = GetInts(greedy);
  const std::vector<std::int64_t> s = GetInts(seeded);
  EXPECT_EQ(g[0], 7);
  std::vector<execution::SamplingCandidate> scratch;
  for (std::size_t r = 0; r < 3; ++r) {
    const std::span<const float> row(&lv[r * kVocab], N(kVocab));
    EXPECT_EQ(g[r], *execution::Greedy(row));
    const execution::SamplingKey key{.seed = static_cast<std::uint64_t>(kv[r * 3]),
                                     .stream = static_cast<std::uint64_t>(kv[(r * 3) + 1]),
                                     .position = static_cast<std::uint64_t>(kv[(r * 3) + 2])};
    EXPECT_EQ(s[r], *execution::Sample(
                        row, {.temperature = 0.8f, .top_k = 20, .top_p = 0.9f, .min_p = 0.01f}, key,
                        scratch));
  }
}

// ---------------------------------------------------------------- decompositions

// Runs `graph` and its full primitive expansion on the same leaves and
// compares every output.
void ExpectExpansionAgrees(const ir::Graph& graph,
                           const std::map<std::uint32_t, std::vector<float>>& values,
                           const std::map<std::uint32_t, std::vector<std::int64_t>>& ints,
                           double tol) {
  std::vector<TensorId> map;
  auto expanded = ir::Expand(graph, true, &map);
  ASSERT_TRUE(expanded.has_value()) << expanded.error().detail;
  for (const ir::Node& n : expanded->nodes()) {
    ASSERT_FALSE(ir::HasDecomposition(n.key)) << ir::OpKeyName(n.key);
  }
  auto direct = Executor::Create(graph);
  auto split = Executor::Create(*expanded);
  ASSERT_TRUE(direct.has_value()) << direct.error().detail;
  ASSERT_TRUE(split.has_value()) << split.error().detail;
  for (const auto& [id, v] : values) {
    ASSERT_TRUE(direct->BindValues(TensorId{id}, v).has_value());
    ASSERT_TRUE(split->BindValues(map.at(id), v).has_value());
  }
  for (const auto& [id, v] : ints) {
    ASSERT_TRUE(direct->BindInts(TensorId{id}, v).has_value());
    ASSERT_TRUE(split->BindInts(map.at(id), v).has_value());
  }
  const auto r1 = direct->Run();
  ASSERT_TRUE(r1.has_value()) << r1.error().detail;
  const auto r2 = split->Run();
  ASSERT_TRUE(r2.has_value()) << r2.error().detail;
  for (const TensorId o : graph.outputs()) {
    const std::string what = ir::OpKeyName(graph.node(graph.producer(o)).key);
    if (graph.tensor(o).format.encoding == Encoding::kI32) {
      EXPECT_EQ(*direct->Ints(o), *split->Ints(map.at(o.v))) << what;
      continue;
    }
    const std::vector<float> a = *direct->Values(o);
    const std::vector<float> c = *split->Values(map.at(o.v));
    ExpectNear(c, std::vector<double>(a.begin(), a.end()), tol, what);
  }
}

TEST(ReferenceDecompositionTest, CompositeOpsEqualTheirDecompositions) {
  ir::Builder b;
  std::map<std::uint32_t, std::vector<float>> values;
  std::map<std::uint32_t, std::vector<std::int64_t>> ints;
  std::uint32_t seed = 200;
  const auto in = [&](std::initializer_list<std::uint64_t> ne, float sd = 1.0f) {
    const TensorId id = b.Input(ir::Dense(Encoding::kF32, ir::ShapeOf(ne)));
    values[id.v] = Normal(seed++, N(ir::Elements(ir::ShapeOf(ne))), sd);
    return id;
  };
  const auto index = [&](std::initializer_list<std::uint64_t> ne, std::vector<std::int64_t> v) {
    const TensorId id = b.Input(ir::Dense(Encoding::kI32, ir::ShapeOf(ne)));
    ints[id.v] = std::move(v);
    return id;
  };
  // Hyper-connections.
  const TensorId mixes = in({24, 3});
  const TensorId scale = in({3});
  const TensorId base = in({24});
  const TensorId x = in({8, 4, 3});
  b.Output(b.HcMix(
      mixes, scale, base,
      ir::HcParams{.mix = ir::HcMixKind::kSinkhorn, .streams = 4, .iterations = 5, .eps = 1e-6f}));
  b.Output(b.HcMix(mixes, scale, base, ir::HcParams{.streams = 4, .eps = 1e-6f}));
  b.Output(b.HcPre(x, in({4, 3}), ir::HcParams{.streams = 4}));
  b.Output(b.HcPre(x, in({8, 4, 3}),
                   ir::HcParams{.fold = ir::HcFold::kGated, .streams = 4, .scale = 0.25f}));
  b.Output(b.HcPost(in({8, 3}), x, in({4, 3}), in({4, 4, 3}), ir::HcParams{.streams = 4}));
  b.Output(b.HcPost(in({8, 3}), x, in({4, 3}), {}, ir::HcParams{.streams = 4}));
  // The indexer's scores, pooling, routing, the reduction.
  b.Output(b.IndexScore(in({8, 3, 2, 1}), in({8, 1, 5, 1}), in({3, 2, 1, 1}), in({5, 2})));
  const TensorId px = in({3, 8, 2});
  b.Output(b.Pool(px, {}, ir::PoolParams{.ratio = 4}));
  b.Output(b.Pool(px, in({3, 8, 2}), ir::PoolParams{.ratio = 2, .kind = ir::PoolKind::kSoftmax}));
  // Several matrices along dim 3 too.
  const TensorId px4 = in({3, 8, 2, 3});
  b.Output(b.Pool(px4, {}, ir::PoolParams{.ratio = 2}));
  b.Output(
      b.Pool(px4, in({3, 8, 2, 3}), ir::PoolParams{.ratio = 4, .kind = ir::PoolKind::kSoftmax}));
  const TensorId logits = in({6, 3});
  for (const ir::Outputs& r :
       {b.Route(logits, {}, {}, {},
                ir::RouteParams{.normalize = true, .experts = 6, .used = 2, .scale = 1.5f}),
        b.Route(logits, in({6}), {}, {},
                ir::RouteParams{.kind = ir::RouteKind::kSigmoidTopK, .experts = 6, .used = 3}),
        b.Route(logits, in({6}), {}, {},
                ir::RouteParams{.kind = ir::RouteKind::kSqrtSoftplusBias,
                                .normalize = true,
                                .experts = 6,
                                .used = 2}),
        b.Route(logits, {}, index({2, 4}, {0, 5, 1, 2, 3, 3, 4, 0}), index({3}, {3, 0, 2}),
                ir::RouteParams{
                    .kind = ir::RouteKind::kHash, .normalize = true, .experts = 6, .used = 2})}) {
    b.Output(r[0]);
    b.Output(r[1]);
  }
  b.Output(b.ExpertReduce(in({5, 2, 3}), in({2, 3}), index({2, 3}, {1, 0, 2, 2, 0, 1}), in({3})));
  // The convolution and the recurrence.
  b.Output(b.SsmConv(in({3 + 4, 5, 1}), in({4, 5})));
  const ir::Outputs gdn =
      b.GatedDeltaNet(in({4, 2, 3, 1}, 0.4f), in({4, 2, 3, 1}, 0.4f), in({4, 4, 3, 1}),
                      in({1, 4, 3, 1}, 0.1f), in({1, 4, 3, 1}, 0.5f), in({4, 4, 4, 1}, 0.2f));
  b.Output(gdn[0]);
  b.Output(gdn[1]);
  // Four sequences reading two q and k sequences (sequence s reads s / 2),
  // and each its own.
  for (const std::uint64_t bq : {2U, 4U}) {
    const ir::Outputs many =
        b.GatedDeltaNet(in({4, 2, 3, bq}, 0.4f), in({4, 2, 3, bq}, 0.4f), in({3, 4, 3, 4}),
                        in({4, 4, 3, 4}, 0.1f), in({1, 4, 3, 4}, 0.5f), in({4, 3, 4, 4}, 0.2f));
    b.Output(many[0]);
    b.Output(many[1]);
  }
  auto g = std::move(b).Finish();
  ASSERT_TRUE(g.has_value()) << g.error().detail;
  ExpectExpansionAgrees(*g, values, ints, 1e-5);
}

TEST(ReferenceDecompositionTest, FusedKindsRunAsThePrimitivesTheyName) {
  // Each fused node beside the primitives built by hand, in one graph.
  ir::Builder b;
  std::map<std::uint32_t, std::vector<float>> values;
  std::map<std::uint32_t, std::vector<std::int64_t>> ints;
  std::uint32_t seed = 300;
  const auto in = [&](std::initializer_list<std::uint64_t> ne) {
    const TensorId id = b.Input(ir::Dense(Encoding::kF32, ir::ShapeOf(ne)));
    values[id.v] = Normal(seed++, N(ir::Elements(ir::ShapeOf(ne))));
    return id;
  };
  const auto index = [&](std::initializer_list<std::uint64_t> ne, std::vector<std::int64_t> v) {
    const TensorId id = b.Input(ir::Dense(Encoding::kI32, ir::ShapeOf(ne)));
    ints[id.v] = std::move(v);
    return id;
  };
  std::vector<std::pair<TensorId, TensorId>> pairs;  // fused, by hand
  const auto pair = [&](TensorId fused, TensorId hand) {
    pairs.emplace_back(fused, hand);
    b.Output(fused);
    b.Output(hand);
  };
  const TensorId x = in({8, 3});
  const TensorId w = in({8});
  const TensorId r = in({8, 3});
  const TensorId pos = index({3}, {4, 9, 30});
  const ir::RopeParams rope{.mode = ir::RopeMode::kNeox, .dims = 4, .base = 100.0f};
  pair(b.Fused(ir::FusedKind::kRmsNormMul, std::array{x, w}, ir::NormParams{1e-6f})[0],
       b.Mul(b.RmsNorm(x, 1e-6f), w));
  pair(b.Fused(ir::FusedKind::kRmsNormMulAdd, std::array{x, w, r}, ir::NormParams{1e-6f})[0],
       b.Add(b.Mul(b.RmsNorm(x, 1e-6f), w), r));
  const TensorId heads = in({4, 2, 3});
  const TensorId hw = in({4});
  pair(b.Fused(ir::FusedKind::kRmsNormMulRope, std::array{heads, hw, pos},
               ir::NormRopeParams{.eps = 1e-6f, .rope = rope})[0],
       b.Rope(b.Mul(b.RmsNorm(heads, 1e-6f), hw), pos, {}, rope));
  const TensorId k = in({4, 2, 3});
  const TensorId kw = in({4});
  const ir::Outputs qk = b.Fused(ir::FusedKind::kQkPrep, std::array{heads, hw, k, kw, pos},
                                 ir::NormRopeParams{.eps = 1e-6f, .rope = rope});
  pair(qk[1], b.Rope(b.Mul(b.RmsNorm(k, 1e-6f), kw), pos, {}, rope));
  const TensorId wg = in({8, 5});
  const TensorId wu = in({8, 5});
  const ir::GluParams glu{.kind = ir::GluKind::kGeGluTanh};
  pair(b.Fused(ir::FusedKind::kMatMulGlu, std::array{wg, wu, x}, glu)[0],
       b.Glu(b.MatMul(wg, x), b.MatMul(wu, x), glu));
  // A product's accumulation and hint leave its value.
  const TensorId had = in({8, 8});
  pair(b.MatMul(wg, x, ir::ProductParams{.accumulate = ir::Accumulate::kF32}), b.MatMul(wg, x));
  pair(b.MatMul(had, x, ir::ProductParams{.hint = ir::ProductHint::kHadamard}), b.MatMul(had, x));
  const TensorId bias = in({5, 3});
  pair(b.Fused(ir::FusedKind::kMatMulAdd, std::array{wg, x, bias})[0],
       b.Add(b.MatMul(wg, x), bias));
  const ir::Outputs prods = b.Fused(ir::FusedKind::kPreparedProducts, std::array{wg, wu, x});
  pair(prods[0], b.MatMul(wg, x));
  pair(prods[1], b.MatMul(wu, x));
  const TensorId z = in({8, 3});
  pair(b.Fused(ir::FusedKind::kNormGate, std::array{x, w, z}, ir::NormParams{1e-5f})[0],
       b.Mul(b.Mul(b.RmsNorm(x, 1e-5f), w), b.Unary(z, ir::UnaryFn::kSigmoid)));
  const TensorId mod = in({8, 1});
  pair(b.Fused(ir::FusedKind::kLayerNormModulate, std::array{x, mod}, ir::NormParams{1e-6f})[0],
       b.Mul(b.LayerNorm(x, 1e-6f), b.Scale(mod, 1.0f, 1.0f)));
  const ir::Outputs grn =
      b.Fused(ir::FusedKind::kGatedResidualNorm, std::array{x, mod, r, mod}, ir::NormParams{1e-6f});
  const TensorId gres = b.Fused(ir::FusedKind::kGatedResidual, std::array{x, mod, r})[0];
  pair(grn[0], gres);
  pair(gres, b.Add(x, b.Mul(r, b.Unary(mod, ir::UnaryFn::kTanh))));
  const TensorId dt = in({1});
  pair(b.Fused(ir::FusedKind::kEulerStep, std::array{x, r, dt})[0], b.Add(x, b.Mul(r, dt)));
  // Routed experts' scales and combine.
  const TensorId gate = in({5, 2, 3});
  const TensorId up = in({5, 2, 3});
  const TensorId ids = index({2, 3}, {0, 3, 2, 1, 3, 3});
  const TensorId sg = in({4});
  const TensorId su = in({4});
  const ir::Outputs moe =
      b.Fused(ir::FusedKind::kMoeGlu, std::array{gate, up, ids, sg, su}, ir::GluParams{});
  const TensorId weights = in({2, 3});
  const TensorId shared = in({5, 3});
  const TensorId shared_gate = in({1, 3});
  pair(b.Fused(ir::FusedKind::kMoeCombine,
               std::array{moe[0], ids, weights, shared, shared_gate, sg})[0],
       b.Add(b.ExpertReduce(moe[0], weights, ids, sg),
             b.Mul(shared, b.Unary(shared_gate, ir::UnaryFn::kSigmoid))));
  // DeepSeek V4's shared expert has no gate.
  pair(b.Fused(ir::FusedKind::kMoeCombine, std::array{moe[0], ids, weights, shared, TensorId{}})[0],
       b.Add(b.ExpertReduce(moe[0], weights, ids, {}), shared));
  auto g = std::move(b).Finish();
  ASSERT_TRUE(g.has_value()) << g.error().detail;
  auto e = Executor::Create(*g);
  ASSERT_TRUE(e.has_value()) << e.error().detail;
  for (const auto& [id, v] : values) {
    ASSERT_TRUE(e->BindValues(TensorId{id}, v).has_value());
  }
  for (const auto& [id, v] : ints) {
    ASSERT_TRUE(e->BindInts(TensorId{id}, v).has_value());
  }
  const auto ran = e->Run();
  ASSERT_TRUE(ran.has_value()) << ran.error().detail;
  for (const auto& [fused, hand] : pairs) {
    EXPECT_EQ(*e->Values(fused), *e->Values(hand))
        << ir::FusedKindName(g->node(g->producer(fused)).key.fused);
  }
  // The MoE GLU by its definition.
  const std::vector<float> got = *e->Values(moe[0]);
  const std::vector<float>& gv = values.at(gate.v);
  const std::vector<float>& uv = values.at(up.v);
  const std::vector<std::int64_t>& iv = ints.at(ids.v);
  for (std::size_t i = 0; i < got.size(); ++i) {
    const auto slot = static_cast<std::size_t>(i / 5);
    const auto ex = static_cast<std::size_t>(iv[slot]);
    const double a = static_cast<double>(gv[i]) * values.at(sg.v)[ex];
    const double c = static_cast<double>(uv[i]) * values.at(su.v)[ex];
    EXPECT_NEAR(got[i], (a / (1 + std::exp(-a))) * c, 1e-5) << i;
  }
}

TEST(ReferenceDecompositionTest, FusedStoresAndOwnerAttention) {
  ir::Builder b;
  const auto f32 = [&](std::initializer_list<std::uint64_t> ne) {
    return b.Input(ir::Dense(Encoding::kF32, ir::ShapeOf(ne)));
  };
  const auto f16_state = [&](std::initializer_list<std::uint64_t> ne, std::uint16_t tensor) {
    return b.State(ir::Dense(Encoding::kF16, ir::ShapeOf(ne)), ir::StateRef{.tensor = tensor});
  };
  const TensorId x = f32({4, 2, 3});
  const TensorId pos = b.Input(ir::Dense(Encoding::kI32, ir::ShapeOf({3})));
  const TensorId idx = b.Input(ir::Dense(Encoding::kI32, ir::ShapeOf({3})));
  const TensorId cache_a = f16_state({8, 6}, 0);
  const TensorId cache_b = f16_state({8, 6}, 1);
  const ir::RopeParams rope{.mode = ir::RopeMode::kNeox, .dims = 4, .base = 100.0f};
  const TensorId fused =
      b.Fused(ir::FusedKind::kRopeStore, std::array{x, pos, cache_a, idx}, rope)[0];
  const TensorId hand =
      b.CacheStore(cache_b, b.Reshape(b.Rope(x, pos, {}, rope), ir::ShapeOf({8, 3})), idx);
  b.Output(fused);
  b.Output(hand);
  // Two owners of two rows each, caches of 5 and 3 cells, on dim 3: each
  // owner's mask its own matrix of 3 rows (padded past its 2).
  const TensorId q = f32({4, 2, 2, 2});
  const TensorId mask = f32({5, 3, 1, 2});
  const TensorId k0 = f32({4, 5, 1, 1});
  const TensorId k1 = f32({4, 3, 1, 1});
  const TensorId v0 = f32({4, 5, 1, 1});
  const TensorId v1 = f32({4, 3, 1, 1});
  const TensorId owners =
      b.Fused(ir::FusedKind::kOwnerAttention, std::array{q, mask, k0, k1, v0, v1},
              ir::OwnerAttnParams{.attn = ir::AttnParams{.scale = 0.5f}, .owners = 2})[0];
  b.Output(owners);
  auto g = std::move(b).Finish();
  ASSERT_TRUE(g.has_value()) << g.error().detail;
  auto e = Executor::Create(*g);
  ASSERT_TRUE(e.has_value()) << e.error().detail;
  std::uint32_t seed = 400;
  std::map<std::uint32_t, std::vector<float>> values;
  for (const TensorId t : {x, q, mask, k0, k1, v0, v1}) {
    values[t.v] = Normal(seed++, N(ir::Elements(g->tensor(t).shape())));
    ASSERT_TRUE(e->BindValues(t, values[t.v]).has_value());
  }
  ASSERT_TRUE(e->BindInts(pos, std::vector<std::int64_t>{1, 2, 3}).has_value());
  ASSERT_TRUE(e->BindInts(idx, std::vector<std::int64_t>{5, 0, 2}).has_value());
  for (const TensorId c : {cache_a, cache_b}) {
    ASSERT_TRUE(e->BindValues(c, std::vector<float>(48, 1.0f)).has_value());
  }
  const auto ran = e->Run();
  ASSERT_TRUE(ran.has_value()) << ran.error().detail;
  // The fused rope store wrote what the hand-built one did.
  const std::vector<float> fa = *e->Values(fused);
  EXPECT_EQ(fa, *e->Values(hand));
  EXPECT_NE(fa, std::vector<float>(48, 1.0f));
  EXPECT_EQ(fa[8], 1.0f);  // cell 1 not stored
  // Owner o's rows attend to its own cache through its own mask matrix's
  // first cells and rows: [Dv, H, R, owners].
  EXPECT_EQ(g->tensor(owners).shape(), ir::ShapeOf({4, 2, 2, 2}));
  const std::vector<float> got = *e->Values(owners);
  const std::vector<float>& qv = values.at(q.v);
  const std::vector<float>& mv = values.at(mask.v);
  for (int o = 0; o < 2; ++o) {
    const int cells = o == 0 ? 5 : 3;
    std::vector<float> qo;
    std::vector<float> mo;
    for (int h = 0; h < 2; ++h) {
      for (int r = 0; r < 2; ++r) {
        for (int d = 0; d < 4; ++d) {
          qo.push_back(qv[N(static_cast<std::uint64_t>(d + (4 * r) + (8 * h) + (16 * o)))]);
        }
      }
    }
    for (int r = 0; r < 2; ++r) {
      for (int c = 0; c < cells; ++c) {
        mo.push_back(mv[N(static_cast<std::uint64_t>(c + (5 * r) + (15 * o)))]);
      }
    }
    const std::vector<double> want =
        NaiveAttention(qo, values.at(o == 0 ? k0.v : k1.v), values.at(o == 0 ? v0.v : v1.v), mo, {},
                       4, 2, 2, 1, cells, 4, 0.5, 0, 0);
    for (std::size_t i = 0; i < want.size(); ++i) {
      EXPECT_NEAR(got[N(static_cast<std::uint64_t>(o) * 16) + i], want[i], 1e-5) << o << " " << i;
    }
  }
  // A mask narrower than an owner's cache, or of fewer rows than its
  // queries, is refused (it would read another row's or owner's cells).
  for (const auto& ne :
       {std::array<std::uint64_t, 4>{4, 3, 1, 2}, std::array<std::uint64_t, 4>{5, 1, 1, 2}}) {
    ir::Builder c;
    const auto in = [&](std::initializer_list<std::uint64_t> shape) {
      return c.Input(ir::Dense(Encoding::kF32, ir::ShapeOf(shape)));
    };
    const TensorId cq = in({4, 2, 2, 2});
    const TensorId cm = in({ne[0], ne[1], ne[2], ne[3]});
    c.Fused(
        ir::FusedKind::kOwnerAttention,
        std::array{cq, cm, in({4, 5, 1, 1}), in({4, 3, 1, 1}), in({4, 5, 1, 1}), in({4, 3, 1, 1})},
        ir::OwnerAttnParams{.attn = ir::AttnParams{.scale = 0.5f}, .owners = 2});
    EXPECT_FALSE(c.ok()) << ne[0] << " " << ne[1];
  }
}

TEST(ReferenceDecompositionTest, DupUpAddSpreadsTheShortcutsChannels) {
  // kernels/image/ops.h AddDupUp: output channel o's sub-pixel (b, d) reads
  // input channel (((o · ft + ft - 1) · 2 + b) · 2 + d) / repeats.
  constexpr std::uint64_t kW = 3;
  constexpr std::uint64_t kH = 2;
  constexpr std::uint64_t kCi = 6;
  constexpr std::uint64_t kCo = 3;
  constexpr std::uint64_t kFt = 2;
  ir::Builder b;
  const TensorId x = b.Input(ir::Dense(Encoding::kF32, ir::ShapeOf({2 * kW, 2 * kH, kCo})));
  const TensorId sc = b.Input(ir::Dense(Encoding::kF32, ir::ShapeOf({kW, kH, kCi})));
  const TensorId y =
      b.Fused(ir::FusedKind::kDupUpAdd, std::array{x, sc}, ir::DupUpParams{.factor_t = kFt})[0];
  b.Output(y);
  auto g = std::move(b).Finish();
  ASSERT_TRUE(g.has_value()) << g.error().detail;
  auto e = Executor::Create(*g);
  ASSERT_TRUE(e.has_value()) << e.error().detail;
  const std::vector<float> xv = Normal(500, N(4 * kW * kH * kCo));
  const std::vector<float> sv = Normal(501, N(kW * kH * kCi));
  ASSERT_TRUE(e->BindValues(x, xv).has_value());
  ASSERT_TRUE(e->BindValues(sc, sv).has_value());
  const auto ran = e->Run();
  ASSERT_TRUE(ran.has_value()) << ran.error().detail;
  const std::uint64_t repeats = kCo * kFt * 4 / kCi;
  std::vector<double> want(xv.begin(), xv.end());
  for (std::uint64_t o = 0; o < kCo; ++o) {
    for (std::uint64_t yy = 0; yy < 2 * kH; ++yy) {
      for (std::uint64_t xx = 0; xx < 2 * kW; ++xx) {
        const std::uint64_t bsub = yy % 2;
        const std::uint64_t dsub = xx % 2;
        const std::uint64_t ci = ((((o * kFt) + kFt - 1) * 2 + bsub) * 2 + dsub) / repeats;
        want[N(xx + (2 * kW * (yy + (2 * kH * o))))] +=
            sv[N((xx / 2) + (kW * ((yy / 2) + (kH * ci))))];
      }
    }
  }
  ExpectNear(*e->Values(y), want, 1e-6, "dup up");
}

// ---------------------------------------------------------------- devices

TEST(ReferenceDevicesTest, CollectivesCompleteAcrossSimulatedDevices) {
  constexpr std::uint32_t kDevices = 4;
  std::vector<std::unique_ptr<ir::Graph>> graphs;
  std::vector<Executor> execs;
  std::vector<std::array<TensorId, 6>> ids;
  for (std::uint32_t d = 0; d < kDevices; ++d) {
    ir::Builder b;
    const TensorId x = b.Input(ir::Dense(Encoding::kF32, ir::ShapeOf({2, 4})));
    const ir::CollectiveParams group{.group = 7};
    const TensorId sum = b.AllReduce(x, group);
    const TensorId all =
        b.AllGather(x, ir::ShapeOf({2, 16}), ir::CollectiveParams{.group = 7, .dim = 1});
    const TensorId swap = b.AllToAll(x, ir::CollectiveParams{.group = 7, .dim = 1});
    // A ring: each sends to the next and receives from the previous.
    b.Send(b.Scale(x, 10.0f), ir::CollectiveParams{.group = 7, .peer = (d + 1) % kDevices});
    const TensorId got =
        b.Recv(ir::Dense(Encoding::kF32, ir::ShapeOf({2, 4})),
               ir::CollectiveParams{.group = 7, .peer = (d + kDevices - 1) % kDevices});
    const TensorId after = b.Add(got, sum);
    for (const TensorId o : {sum, all, swap, got, after}) {
      b.Output(o);
    }
    auto g = std::move(b).Finish();
    ASSERT_TRUE(g.has_value()) << g.error().detail;
    graphs.push_back(std::make_unique<ir::Graph>(std::move(*g)));
    auto e = Executor::Create(*graphs.back());
    ASSERT_TRUE(e.has_value()) << e.error().detail;
    execs.push_back(std::move(*e));
    ids.push_back({x, sum, all, swap, got, after});
  }
  std::vector<std::vector<float>> xs;
  for (std::uint32_t d = 0; d < kDevices; ++d) {
    std::vector<float> v(8);
    for (std::size_t i = 0; i < 8; ++i) {
      v[i] = static_cast<float>((d * 100) + i);
    }
    xs.push_back(v);
    ASSERT_TRUE(execs[d].BindValues(ids[d][0], v).has_value());
  }
  std::vector<Executor*> devices;
  for (Executor& e : execs) {
    devices.push_back(&e);
  }
  const DeviceGroup group{.group = 7, .devices = {0, 1, 2, 3}};
  const auto ran = RunDevices(devices, std::span(&group, 1));
  ASSERT_TRUE(ran.has_value()) << ran.error().detail;
  for (std::uint32_t d = 0; d < kDevices; ++d) {
    std::vector<double> sum(8);
    std::vector<double> all;
    std::vector<double> swap;
    for (std::uint32_t m = 0; m < kDevices; ++m) {
      for (std::size_t i = 0; i < 8; ++i) {
        sum[i] += xs[m][i];
        all.push_back(xs[m][i]);
      }
      // Rows d of each member (one row of two values each).
      swap.push_back(xs[m][2 * d]);
      swap.push_back(xs[m][(2 * d) + 1]);
    }
    ExpectNear(*execs[d].Values(ids[d][1]), sum, 0, "all-reduce");
    ExpectNear(*execs[d].Values(ids[d][2]), all, 0, "all-gather");
    ExpectNear(*execs[d].Values(ids[d][3]), swap, 0, "all-to-all");
    const std::vector<float>& prev = xs[(d + kDevices - 1) % kDevices];
    std::vector<double> got;
    std::vector<double> after;
    for (std::size_t i = 0; i < 8; ++i) {
      got.push_back(10.0 * prev[i]);
      after.push_back((10.0 * prev[i]) + sum[i]);
    }
    ExpectNear(*execs[d].Values(ids[d][4]), got, 0, "recv");
    ExpectNear(*execs[d].Values(ids[d][5]), after, 0, "after");
  }
}

TEST(ReferenceDevicesTest, UnpairedCollectivesAreRefused) {
  // Two devices each waiting to receive first.
  std::vector<std::unique_ptr<ir::Graph>> graphs;
  std::vector<Executor> execs;
  for (std::uint32_t d = 0; d < 2; ++d) {
    ir::Builder b;
    const TensorId got = b.Recv(ir::Dense(Encoding::kF32, ir::ShapeOf({2})),
                                ir::CollectiveParams{.group = 1, .peer = 1 - d});
    b.Send(got, ir::CollectiveParams{.group = 1, .peer = 1 - d});
    b.Output(got);
    auto g = std::move(b).Finish();
    ASSERT_TRUE(g.has_value()) << g.error().detail;
    graphs.push_back(std::make_unique<ir::Graph>(std::move(*g)));
    execs.push_back(std::move(*Executor::Create(*graphs.back())));
  }
  std::vector<Executor*> devices{&execs[0], &execs[1]};
  const DeviceGroup group{.group = 1, .devices = {0, 1}};
  const auto ran = RunDevices(devices, std::span(&group, 1));
  ASSERT_FALSE(ran.has_value());
  EXPECT_EQ(ran.error().error, ExecError::kCollective);
}

// ---------------------------------------------------------------- a layer

// A decoder layer of Gemma's shape at small widths with Q8_0 and F16
// weights, F16 caches and a prompt of several rows: built from primitives,
// or with its norms, rotations, stores and feed-forward as fused kinds.
struct SmallLayer {
  std::unique_ptr<ir::Graph> graph;
  TensorId tokens, positions, cells, store_at, logits, cache_k, cache_v;
  std::vector<std::pair<TensorId, std::vector<std::byte>>> weights;
};

SmallLayer BuildSmallLayer(bool fused) {
  constexpr std::uint64_t kWidth = 256;
  constexpr std::uint64_t kHead = 64;
  constexpr std::uint64_t kHeads = 4;
  constexpr std::uint64_t kKv = 2;
  constexpr std::uint64_t kRows = 6;
  constexpr std::uint64_t kCells = 32;
  constexpr std::uint64_t kVocab = 512;
  constexpr std::uint64_t kFfn = 512;
  ir::Builder b;
  SmallLayer l;
  std::uint32_t seed = 600;
  const auto weight = [&](std::uint16_t role, Encoding e, std::uint64_t k, std::uint64_t n) {
    const format::TensorFormat f{
        .encoding = e,
        .layout = e == Encoding::kQ8_0 ? Layout::kGgmlBlocks : Layout::kDense,
        .shape = ir::ShapeOf({k, n})};
    const TensorId id = b.Weight(f, ir::WeightRef{.component = 1, .role = role});
    std::vector<float> v = Normal(seed++, N(k * n), 1.0f / std::sqrt(static_cast<float>(k)));
    std::vector<std::byte> bytes;
    if (e == Encoding::kQ8_0) {
      bytes = Encode(f, v);
    } else {
      const format::TensorFormat dense = ir::Dense(Encoding::kF16, ir::ShapeOf({k, n}));
      bytes = Encode(dense, v);
    }
    l.weights.emplace_back(id, std::move(bytes));
    return id;
  };
  l.tokens = b.Input(ir::Dense(Encoding::kI32, ir::ShapeOf({kRows})));
  l.positions = b.Input(ir::Dense(Encoding::kI32, ir::ShapeOf({kRows})));
  l.cells = b.Input(ir::Dense(Encoding::kI32, ir::ShapeOf({kCells})));
  l.store_at = b.Input(ir::Dense(Encoding::kI32, ir::ShapeOf({kRows})));
  const TensorId embed = weight(0, Encoding::kQ8_0, kWidth, kVocab);
  const TensorId norm = weight(1, Encoding::kF16, kWidth, 1);
  const TensorId wq = weight(2, Encoding::kQ8_0, kWidth, kHeads * kHead);
  const TensorId wk = weight(3, Encoding::kQ8_0, kWidth, kKv * kHead);
  const TensorId wv = weight(4, Encoding::kQ8_0, kWidth, kKv * kHead);
  const TensorId wo = weight(5, Encoding::kQ8_0, kHeads * kHead, kWidth);
  const TensorId wg = weight(6, Encoding::kQ8_0, kWidth, kFfn);
  const TensorId wu = weight(7, Encoding::kQ8_0, kWidth, kFfn);
  const TensorId wd = weight(8, Encoding::kQ8_0, kFfn, kWidth);
  const TensorId ffn_norm = weight(9, Encoding::kF16, kWidth, 1);
  l.cache_k = b.State(ir::Dense(Encoding::kF16, ir::ShapeOf({kKv * kHead, kCells})),
                      ir::StateRef{.tensor = 0});
  l.cache_v = b.State(ir::Dense(Encoding::kF16, ir::ShapeOf({kKv * kHead, kCells})),
                      ir::StateRef{.tensor = 1});
  const ir::RopeParams rope{.mode = ir::RopeMode::kNeox, .dims = kHead, .base = 10000.0f};
  const auto norm_mul = [&](TensorId x, TensorId w) {
    const TensorId wv1 = b.Reshape(w, ir::ShapeOf({kWidth}));
    return fused ? b.Fused(ir::FusedKind::kRmsNormMul, std::array{x, wv1}, ir::NormParams{1e-6f})[0]
                 : b.Mul(b.RmsNorm(x, 1e-6f), wv1);
  };
  TensorId x = b.Scale(b.GetRows(embed, l.tokens), 16.0f);
  const TensorId n = norm_mul(x, norm);
  const TensorId q = b.Rope(b.Reshape(b.MatMul(wq, n), ir::ShapeOf({kHead, kHeads, kRows})),
                            l.positions, {}, rope);
  const TensorId kr = b.Reshape(b.MatMul(wk, n), ir::ShapeOf({kHead, kKv, kRows}));
  const TensorId kc = fused ? b.Fused(ir::FusedKind::kRopeStore,
                                      std::array{kr, l.positions, l.cache_k, l.store_at}, rope)[0]
                            : b.CacheStore(l.cache_k,
                                           b.Reshape(b.Rope(kr, l.positions, {}, rope),
                                                     ir::ShapeOf({kKv * kHead, kRows})),
                                           l.store_at);
  const TensorId vc = b.CacheStore(l.cache_v, b.MatMul(wv, n), l.store_at);
  const TensorId mask = b.MaskBuild(l.positions, l.cells, kRows, ir::MaskParams{});
  const auto heads = [&](TensorId c) {
    return b.Permute(b.Reshape(c, ir::ShapeOf({kHead, kKv, kCells})), {0, 2, 1, 3});
  };
  const TensorId attn = b.Attention(b.Permute(q, {0, 2, 1, 3}), heads(kc), heads(vc), mask, {},
                                    ir::AttnParams{.scale = 0.125f, .softcap = 50.0f});
  x = b.Add(x, b.MatMul(wo, b.Reshape(b.Cont(attn), ir::ShapeOf({kHeads * kHead, kRows}))));
  const TensorId fn = norm_mul(x, ffn_norm);
  const ir::GluParams glu{.kind = ir::GluKind::kGeGluTanh};
  const TensorId ff = fused ? b.Fused(ir::FusedKind::kMatMulGlu, std::array{wg, wu, fn}, glu)[0]
                            : b.Glu(b.MatMul(wg, fn), b.MatMul(wu, fn), glu);
  x = b.Add(x, b.MatMul(wd, ff));
  l.logits = b.Softcap(b.MatMul(embed, x), 30.0f);
  b.Output(l.logits);
  auto g = std::move(b).Finish();
  EXPECT_TRUE(g.has_value()) << (g ? "" : g.error().detail);
  if (g) {
    l.graph = std::make_unique<ir::Graph>(std::move(*g));
  }
  return l;
}

TEST(ReferenceLayerTest, ASmallLayerRunsAndItsFusedFormAgrees) {
  std::array<std::vector<float>, 2> logits;
  std::array<std::vector<std::byte>, 2> caches;
  for (const bool fused : {false, true}) {
    SmallLayer l = BuildSmallLayer(fused);
    ASSERT_NE(l.graph, nullptr);
    auto e = Executor::Create(*l.graph);
    ASSERT_TRUE(e.has_value()) << e.error().detail;
    for (const auto& [id, bytes] : l.weights) {
      const PartView part{.role = format::PartRole::kData, .bytes = bytes};
      ASSERT_TRUE(e->Bind(id, std::span(&part, 1)).has_value());
    }
    for (const TensorId c : {l.cache_k, l.cache_v}) {
      ASSERT_TRUE(e->BindValues(c, std::vector<float>(128 * 32, 0.0f)).has_value());
    }
    ASSERT_TRUE(
        e->BindInts(l.tokens, std::vector<std::int64_t>{3, 77, 500, 12, 12, 9}).has_value());
    ASSERT_TRUE(e->BindInts(l.positions, std::vector<std::int64_t>{0, 1, 2, 3, 4, 5}).has_value());
    std::vector<std::int64_t> cells(32, -1);
    for (std::size_t i = 0; i < 6; ++i) {
      cells[i] = static_cast<std::int64_t>(i);
    }
    ASSERT_TRUE(e->BindInts(l.cells, cells).has_value());
    ASSERT_TRUE(e->BindInts(l.store_at, std::vector<std::int64_t>{0, 1, 2, 3, 4, 5}).has_value());
    const auto ran = e->Run();
    ASSERT_TRUE(ran.has_value()) << ran.error().detail;
    logits.at(fused ? 1 : 0) = *e->Values(l.logits);
    const std::span<const std::byte> k = e->Bytes(l.cache_k);
    caches.at(fused ? 1 : 0).assign(k.begin(), k.end());
  }
  ASSERT_EQ(logits[0].size(), 512U * 6);
  for (const float v : logits[0]) {
    ASSERT_TRUE(std::isfinite(v));
    ASSERT_LE(std::fabs(v), 30.0f);
  }
  // The fused kinds are their primitives: the same values, the same cache.
  EXPECT_EQ(logits[0], logits[1]);
  EXPECT_EQ(caches[0], caches[1]);
}

// ---------------------------------------------------------------- bindings

TEST_F(ExecTest, BindingsAreCheckedAndStateKeepsItsStores) {
  const TensorId x = F({4});
  const TensorId cache =
      b.State(ir::Dense(Encoding::kF32, ir::ShapeOf({4, 3})), ir::StateRef{.tensor = 0});
  const TensorId idx = I({1});
  const TensorId stored = b.CacheStore(cache, b.Reshape(x, ir::ShapeOf({4, 1})), idx);
  const TensorId total = b.SumRows(b.Reshape(b.Cont(stored), ir::ShapeOf({12})));
  b.Output(total);
  ASSERT_TRUE(Finish());
  // Unbound leaves.
  const auto unbound = exec->Run();
  ASSERT_FALSE(unbound.has_value());
  EXPECT_EQ(unbound.error().error, ExecError::kUnbound);
  // A result is not a leaf; bytes of the wrong size.
  EXPECT_EQ(test_support::FailedCode(exec->BindValues(total, std::vector<float>{1})),
            ExecError::kBinding);
  const std::vector<std::byte> short_bytes(8);
  const PartView part{.role = format::PartRole::kData, .bytes = short_bytes};
  EXPECT_EQ(test_support::FailedCode(exec->Bind(x, std::span(&part, 1))), ExecError::kBinding);
  Set(x, {1, 2, 3, 4});
  Set(cache, std::vector<float>(12, 0.0f));
  SetInts(idx, {0});
  ASSERT_TRUE(Run());
  EXPECT_EQ(Get(total)[0], 10.0f);
  // The state keeps the first run's store.
  SetInts(idx, {2});
  ASSERT_TRUE(Run());
  EXPECT_EQ(Get(total)[0], 20.0f);
}

}  // namespace
}  // namespace llmp::kernels::reference
