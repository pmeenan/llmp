// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Layout sinking (ir/layout.h) on the image pipeline's actual graphs
// at small widths, a flow block, the decoder and the text encoder, and on
// small hand-built graphs: the sunk graph computes the original's outputs
// bit for bit on the reference executor, with fewer copies.

#include <gtest/gtest.h>

#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <format>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "components/spatial.h"
#include "expected_error.h"
#include "format/tensor_format.h"
#include "ir/builder.h"
#include "ir/graph.h"
#include "ir/layout.h"
#include "ir/op.h"
#include "kernels/reference/exec.h"
#include "model/qwen_image.h"
#include "spec/image_spatial.h"

namespace {

namespace co = llmp::components;
namespace fm = llmp::format;
namespace ir = llmp::ir;
namespace mo = llmp::model;
namespace rf = llmp::kernels::reference;
namespace sp = llmp::spec;
using ir::TensorId;
using llmp::test_support::Failed;

struct Draw {
  std::uint64_t state;
  float operator()() {
    state ^= state << 13U;
    state ^= state >> 7U;
    state ^= state << 17U;
    return static_cast<float>(static_cast<double>(state >> 11U) * 0x1p-52 - 1.0);
  }
};

std::uint64_t Elements(const ir::TensorDesc& t) { return ir::Elements(t.shape()); }

std::size_t Count(const ir::Graph& g, ir::Op op) {
  std::size_t n = 0;
  for (const ir::Node& node : g.nodes()) n += node.key == ir::Primitive(op);
  return n;
}

// Leaves' values by the original graph's leaf: inputs by a filler, weights
// drawn (BF16 ones rounded first, so both graphs read the same values).
struct Values {
  std::vector<std::pair<TensorId, std::vector<float>>> floats;
  std::vector<std::pair<TensorId, std::vector<std::int64_t>>> ints;
};

using Filler = void (*)(const ir::Graph&, TensorId, std::vector<float>&,
                        std::vector<std::int64_t>&);

Values Leaves(const ir::Graph& g, Filler fill) {
  Values v;
  Draw draw{29};
  for (std::uint32_t i = 0; i < g.nodes().size(); ++i) {
    const ir::Node& n = g.nodes()[i];
    if (n.key != ir::Primitive(ir::Op::kInput) && n.key != ir::Primitive(ir::Op::kWeight)) continue;
    const TensorId t = n.out[0];
    const ir::TensorDesc& d = g.tensor(t);
    if (d.format.encoding == fm::Encoding::kI32) {
      std::vector<float> unused;
      std::vector<std::int64_t> ints(Elements(d));
      fill(g, t, unused, ints);
      v.ints.emplace_back(t, std::move(ints));
      continue;
    }
    std::vector<float> values(Elements(d));
    std::vector<std::int64_t> unused;
    if (n.key == ir::Primitive(ir::Op::kInput)) {
      fill(g, t, values, unused);
    } else {
      const double fan = d.shape().rank >= 2 ? static_cast<double>(d.ne(0)) : 0.0;
      const float scale = fan > 0 ? static_cast<float>(std::sqrt(3.0 / fan)) : 0.2F;
      for (auto& x : values) x = (fan > 0 ? 0.0F : 1.0F) + scale * draw();
    }
    if (d.format.encoding == fm::Encoding::kBF16) {
      for (auto& x : values) {
        const auto u = std::bit_cast<std::uint32_t>(x);
        x = std::bit_cast<float>(((u + 0x7fffU + ((u >> 16U) & 1U)) >> 16U) << 16U);
      }
    }
    v.floats.emplace_back(t, std::move(values));
  }
  return v;
}

std::vector<std::vector<float>> Run(const ir::Graph& g, const Values& v,
                                    std::span<const TensorId> map) {
  auto e = rf::Executor::Create(g);
  EXPECT_TRUE(e) << (e ? "" : e.error().detail);
  if (!e) return {};
  for (const auto& [t, values] : v.floats) {
    const TensorId at = map.empty() ? t : map[t.v];
    EXPECT_TRUE(at.valid());
    auto bound = e->BindValues(at, values);
    EXPECT_TRUE(bound) << (bound ? "" : bound.error().detail);
  }
  for (const auto& [t, ints] : v.ints) {
    const TensorId at = map.empty() ? t : map[t.v];
    EXPECT_TRUE(at.valid());
    auto bound = e->BindInts(at, ints);
    EXPECT_TRUE(bound) << (bound ? "" : bound.error().detail);
  }
  auto ran = e->Run();
  EXPECT_TRUE(ran) << (ran ? "" : ran.error().detail);
  std::vector<std::vector<float>> out;
  for (const TensorId o : g.outputs()) {
    auto values = e->Values(o);
    EXPECT_TRUE(values) << (values ? "" : values.error().detail);
    out.push_back(values.value_or(std::vector<float>{}));
  }
  return out;
}

// The sunk graph computes `g`'s outputs exactly, with at most `conts`
// copies (Cont nodes) left.
void ExpectSunkExact(const ir::Graph& g, Filler fill, std::size_t conts, const char* what) {
  SCOPED_TRACE(what);
  auto sunk = ir::SinkLayouts(g);
  ASSERT_TRUE(sunk) << Failed(sunk, &ir::VerifyError::detail).value_or("");
  const std::size_t before = Count(g, ir::Op::kCont), after = Count(sunk->graph, ir::Op::kCont);
  std::printf("%s: %zu nodes, %zu copies -> %zu nodes, %zu copies\n", what, g.nodes().size(),
              before, sunk->graph.nodes().size(), after);
  EXPECT_LE(after, conts);
  if (after > conts || std::getenv("LLMP_LAYOUT_DUMP") != nullptr) {
    for (const ir::Node& n : sunk->graph.nodes()) {
      std::printf("  %s", std::string(ir::OpKeyName(n.key)).c_str());
      for (const TensorId s : n.sources())
        if (s.valid()) {
          const auto& t = sunk->graph.tensor(s);
          std::printf(" [%s|%s]",
                      std::format("{} {} {} {}", t.ne(0), t.ne(1), t.ne(2), t.ne(3)).c_str(),
                      std::format("{} {} {} {}", t.nb[0], t.nb[1], t.nb[2], t.nb[3]).c_str());
        }
      std::printf("\n");
    }
  }
  ASSERT_EQ(sunk->graph.outputs().size(), g.outputs().size());
  for (std::size_t i = 0; i < g.outputs().size(); ++i) {
    const auto& want = g.tensor(g.outputs()[i]);
    const auto& got = sunk->graph.tensor(sunk->graph.outputs()[i]);
    EXPECT_EQ(got.shape().ne, want.shape().ne) << i;
    EXPECT_EQ(got.nb, want.nb) << i;
  }
  ASSERT_EQ(sunk->graph.inputs().size(), g.inputs().size());
  const Values v = Leaves(g, fill);
  const auto want = Run(g, v, {});
  const auto got = Run(sunk->graph, v, sunk->tensors);
  ASSERT_EQ(got.size(), want.size());
  for (std::size_t i = 0; i < want.size(); ++i) {
    ASSERT_EQ(got[i].size(), want[i].size()) << i;
    for (std::size_t j = 0; j < want[i].size(); ++j)
      ASSERT_EQ(std::bit_cast<std::uint32_t>(got[i][j]), std::bit_cast<std::uint32_t>(want[i][j]))
          << what << " output " << i << " value " << j;
  }
}

co::SpatialPayment Unbounded() {
  return {[](std::uint64_t) { return true; }, [](std::uint64_t) {}};
}
std::vector<co::BoundTensor> Bound(std::span<const mo::QwenImageTensor> source, fm::Encoding e) {
  std::vector<co::BoundTensor> tensors;
  for (const auto& tensor : source) {
    fm::Shape shape;
    shape.rank = static_cast<std::uint8_t>(tensor.shape.size());
    for (std::size_t d = 0; d < tensor.shape.size(); ++d)
      shape.ne[d] = tensor.shape[tensor.shape.size() - 1 - d];
    tensors.push_back({tensor.name, ir::Dense(e, shape)});
  }
  return tensors;
}

constexpr std::uint32_t kText = 8, kGrid = 4, kRows = kText + kGrid * kGrid;

void FlowFill(const ir::Graph& g, TensorId t, std::vector<float>& values,
              std::vector<std::int64_t>& ints) {
  Draw draw{7 + t.v};
  if (!ints.empty()) {
    for (std::uint32_t r = 0; r < kRows; ++r) {
      const auto position = mo::QwenImageDenoiserPosition(kText, kGrid, kGrid, r);
      for (std::uint32_t a = 0; a < 3 && r + kRows * a < ints.size(); ++a)
        ints[r + std::size_t{kRows} * a] = position[a];
    }
    return;
  }
  if (values.size() == std::size_t{kRows} * kRows) {
    for (std::uint32_t q = 0; q < kText; ++q)
      for (std::uint32_t k = q + 1; k < kRows; ++k) values[k + std::size_t{kRows} * q] = -INFINITY;
    return;
  }
  if (values.size() == 1) {
    values[0] = -0.05F;
    return;
  }
  (void)g;
  for (auto& x : values) x = draw();
}

TEST(IrLayoutTest, FlowBlockSinksExactlyWithFewerCopies) {
  mo::QwenImageDenoiserProfile p;
  p.blocks = 2;
  p.heads = 2;
  p.width = p.heads * p.head_dim;
  p.mlp = 512;
  p.context = p.width;
  const auto source = mo::DenoiserTensors(p);
  auto mapped = sp::ImageFlowSpec(p, kText, kGrid * kGrid, source);
  ASSERT_TRUE(mapped) << Failed(mapped).value_or("");
  const auto tensors = Bound(source, fm::Encoding::kBF16);
  const std::array<std::span<const co::BoundTensor>, 1> tables{tensors};
  const std::array owners{co::SpatialOwner{0, {1, kText, p.context, 1}}};
  const std::array ports{co::SpatialInput{0, 0, {1, kGrid * kGrid, p.in_channels, 1}},
                         co::SpatialInput{0, 1, {1, 2, p.timestep_dim, 1}},
                         co::SpatialInput{0, 2, {kRows, 3, 1, 1}, fm::Encoding::kI32},
                         co::SpatialInput{0, 3, {kRows, kRows, 1, 1}},
                         co::SpatialInput{0, 4, {1, 1, 1, 1}}};
  auto graph = co::BuildSpatialStep(mapped->spec, tables, owners, Unbounded(),
                                    {.links = mapped->links,
                                     .reference_buffers = false,
                                     .exports = mapped->exports,
                                     .inputs = ports});
  ASSERT_TRUE(graph) << Failed(graph).value_or("");
  // Two blocks: at most a third of the copies remain.
  ExpectSunkExact(graph->graph(), &FlowFill, Count(graph->graph(), ir::Op::kCont) / 3, "flow");
}

void PlainFill(const ir::Graph&, TensorId t, std::vector<float>& values,
               std::vector<std::int64_t>& ints) {
  Draw draw{13 + t.v};
  for (auto& x : values) x = draw();
  for (std::size_t i = 0; i < ints.size(); ++i) ints[i] = static_cast<std::int64_t>(i % 7);
}

TEST(IrLayoutTest, DecoderSinksExactly) {
  mo::QwenImageVaeProfile profile;
  profile.z_dim = 16;
  profile.base = 16;
  profile.dim_mult = {1, 2, 4, 4, 4};
  profile.res_blocks = 1;
  constexpr std::uint32_t kLatent = 2;
  const auto source = mo::VaeDecoderTensors(profile);
  auto mapped = sp::ImageDecodeSpec(profile, kLatent, source);
  ASSERT_TRUE(mapped) << Failed(mapped).value_or("");
  const auto tensors = Bound(source, fm::Encoding::kF32);
  const std::array<std::span<const co::BoundTensor>, 1> tables{tensors};
  const std::array owners{co::SpatialOwner{0, {1, kLatent * kLatent, profile.z_dim, 1}}};
  const std::array ports{co::SpatialInput{0, 0, {1, 2, profile.z_dim, 1}}};
  auto graph = co::BuildSpatialStep(mapped->spec, tables, owners, Unbounded(),
                                    {.links = mapped->links,
                                     .reference_buffers = false,
                                     .exports = mapped->exports,
                                     .inputs = ports});
  ASSERT_TRUE(graph) << Failed(graph).value_or("");
  // The decoder's convolutions keep the spatial layout: no more copies.
  ExpectSunkExact(graph->graph(), &PlainFill, Count(graph->graph(), ir::Op::kCont), "decoder");
}

void TextFill(const ir::Graph& g, TensorId t, std::vector<float>& values,
              std::vector<std::int64_t>& ints) {
  Draw draw{17 + t.v};
  (void)g;
  for (std::size_t i = 0; i < ints.size(); ++i) ints[i] = static_cast<std::int64_t>((i * 37) % 64);
  const auto rows = static_cast<std::size_t>(std::sqrt(static_cast<double>(values.size())));
  if (!values.empty() && rows * rows == values.size()) {
    for (std::size_t q = 0; q < rows; ++q)
      for (std::size_t k = 0; k < rows; ++k) values[k + rows * q] = k > q ? -INFINITY : 0.0F;
    return;
  }
  for (auto& x : values) x = draw();
}

TEST(IrLayoutTest, TextEncoderSinksExactlyWithFewerCopies) {
  mo::QwenImageTextProfile p;
  p.layers = 2;
  p.width = 256;
  p.heads = 2;
  p.kv_heads = 1;
  p.head_dim = 128;
  p.ffn = 512;
  p.vocab = 64;
  constexpr std::uint32_t kTokens = 9, kDrop = 3;
  const auto source = mo::TextEncoderTensors(p);
  auto mapping = sp::ImageTextSpec(p, kTokens, kDrop, source);
  ASSERT_TRUE(mapping) << Failed(mapping).value_or("");
  const auto tensors = Bound(source, fm::Encoding::kBF16);
  const std::array<std::span<const co::BoundTensor>, 1> tables{tensors};
  const std::array owners{co::SpatialOwner{0, {1, kTokens, 1, 1}}};
  const std::array ports{co::SpatialInput{0, 0, {kTokens, 1, 1, 1}, fm::Encoding::kI32},
                         co::SpatialInput{0, 1, {kTokens, kTokens, 1, 1}}};
  auto graph = co::BuildSpatialStep(mapping->spec, tables, owners, Unbounded(),
                                    {.links = mapping->links,
                                     .reference_buffers = false,
                                     .input_encoding = fm::Encoding::kI32,
                                     .exports = mapping->exports,
                                     .inputs = ports});
  ASSERT_TRUE(graph) << Failed(graph).value_or("");
  ExpectSunkExact(graph->graph(), &TextFill, Count(graph->graph(), ir::Op::kCont) / 2, "text");
}

// A row-wise op between two transposes: the round trip goes, the result is
// packed once at the output.
TEST(IrLayoutTest, RoundTripsCancelAndOutputsKeepTheirLayout) {
  ir::Builder b;
  const TensorId x = b.Input(ir::Dense(fm::Encoding::kF32, ir::ShapeOf({1, 6, 4, 1})));
  const TensorId w = b.Weight(ir::Dense(fm::Encoding::kF32, ir::ShapeOf({4, 3})),
                              {.stage = 0, .component = 1, .role = 0, .layer = 0});
  // Spatial [1, 6, 4] to channel rows [4, 6], a product, and back.
  const auto rows = [&](TensorId s, std::uint64_t c) {
    return b.Reshape(b.Cont(b.Permute(s, {1, 2, 0, 3})), ir::ShapeOf({c, 6}));
  };
  const auto spatial = [&](TensorId r, std::uint64_t c) {
    return b.Cont(b.Permute(b.Reshape(r, ir::ShapeOf({c, 1, 6, 1})), {2, 0, 1, 3}));
  };
  const TensorId y = spatial(b.MatMul(w, rows(x, 4)), 3);
  const TensorId z = spatial(b.Unary(b.Scale(rows(y, 3), 2.0F), ir::UnaryFn::kSilu), 3);
  b.Output(b.Add(z, y));
  auto g = std::move(b).Finish();
  ASSERT_TRUE(g) << g.error().detail;
  // One copy for the input's rows and one for the output's layout.
  ExpectSunkExact(*g, &PlainFill, 2, "round trip");
}

TEST(IrLayoutTest, LeavesAGraphWithStateUnchanged) {
  ir::Builder b;
  const TensorId cache =
      b.State(ir::Dense(fm::Encoding::kF32, ir::ShapeOf({4, 8})), {.stage = 0, .tensor = 0});
  const TensorId x = b.Input(ir::Dense(fm::Encoding::kF32, ir::ShapeOf({2, 4})));
  const TensorId idx = b.Input(ir::Dense(fm::Encoding::kI32, ir::ShapeOf({2})));
  b.Output(b.CacheStore(cache, b.Cont(b.Transpose(x)), idx));
  auto g = std::move(b).Finish();
  ASSERT_TRUE(g) << g.error().detail;
  auto sunk = ir::SinkLayouts(*g);
  ASSERT_TRUE(sunk);
  EXPECT_EQ(sunk->rewritten, 0U);
  EXPECT_EQ(sunk->graph.nodes().size(), g->nodes().size());
}

}  // namespace
