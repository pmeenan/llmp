// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The importer's routed-expert relayout (import/expert_relayout.h): which
// tables it takes, the assemblies it makes, and that the published bytes
// decode (kernels/reference) to exactly the split source's values.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <expected>
#include <format>
#include <map>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "format/layout.h"
#include "format/tensor_format.h"
#include "import/expert_relayout.h"
#include "import/indexed_parts.h"
#include "kernels/reference/decode.h"
#include "spec/spec.h"

namespace llmp::import {
namespace {

using format::Encoding;
using format::Layout;
using format::PartRole;
using format::TensorFormat;

constexpr std::uint32_t kExperts = 3;

TensorFormat Split(std::uint64_t k, std::uint64_t n, std::uint64_t experts = 0) {
  TensorFormat f{.encoding = Encoding::kNvfp4,
                 .layout = Layout::kSplit,
                 .shape = {.ne = {k, n, 1, 1}, .rank = 2}};
  f.params.tensor_scale = true;
  if (experts != 0) {
    f.shape.ne[2] = experts;
    f.shape.rank = 3;
  }
  return f;
}

TEST(ImportExpertRelayoutTest, RelaysOutOnlyLosslessBlockScaledTables) {
  EXPECT_TRUE(RelaysOut(Split(2560, 640, 512)));
  EXPECT_TRUE(RelaysOut(Split(640, 2560, 512)));
  EXPECT_FALSE(RelaysOut(Split(2560, 640)));       // not a table of experts
  EXPECT_FALSE(RelaysOut(Split(2560, 600, 512)));  // rows not whole atoms: would grow
  EXPECT_FALSE(RelaysOut(Split(2576, 640, 512)));  // 161 scales a row: not whole atoms
  EXPECT_FALSE(RelaysOut(Split(8192, 640, 512)));  // 512 scales a row: past the route's band
  auto mx = Split(2560, 640, 512);
  mx.encoding = Encoding::kMxfp8;
  mx.params.tensor_scale = false;
  EXPECT_TRUE(RelaysOut(mx));
  auto mx4 = Split(2560, 640, 512);
  mx4.encoding = Encoding::kMxfp4;
  mx4.params.tensor_scale = false;
  EXPECT_TRUE(RelaysOut(mx4));
  auto atoms = Split(2560, 640, 512);
  atoms.layout = Layout::kScaleAtoms;
  EXPECT_FALSE(RelaysOut(atoms));
}

// A synthetic checkpoint of three experts' gate, up and down matrices,
// their bytes in one file.
struct Source {
  Checkpoint checkpoint{};
  std::vector<std::byte> file{};
  std::map<std::string, std::uint32_t> index{};
  void Add(const std::string& name, const TensorFormat& f, PartRole role, std::uint32_t seed) {
    const auto parts = format::PartsOf(f);
    const auto bytes = parts->Find(role)->bytes;
    const std::uint64_t at = file.size();
    for (std::uint64_t i = 0; i < bytes; ++i) {
      auto v = static_cast<std::uint8_t>((i * 37 + seed * 101 + 11) % 251);
      // Finite positive E4M3 block scales; F32 tensor scales of 0.5.
      if (role == PartRole::kBlockScales) v = static_cast<std::uint8_t>(0x28 + (v % 0x30));
      if (role == PartRole::kTensorScale) v = std::array<std::uint8_t, 4>{0, 0, 0, 0x3F}[i];
      file.push_back(static_cast<std::byte>(v));
    }
    index[name] = static_cast<std::uint32_t>(checkpoint.tensors.size());
    checkpoint.tensors.push_back({.name = name,
                                  .format = f,
                                  .part = role,
                                  .experts = false,
                                  .extent = {.file = 0, .offset = at, .bytes = bytes}});
  }
};

// Runs every member's route into its output part.
std::map<std::pair<std::uint32_t, PartRole>, std::vector<std::byte>> Publish(
    const IndexedPartPlan& plan, std::span<const std::byte> file) {
  std::map<std::pair<std::uint32_t, PartRole>, std::vector<std::byte>> out;
  for (std::uint32_t o = 0; o < plan.outputs().size(); ++o)
    for (const auto& part : plan.outputs()[o].parts.parts())
      out[{o, part.role}].resize(static_cast<std::size_t>(part.bytes));
  std::vector<std::byte> workspace(65536);
  for (const auto& member : plan.members()) {
    auto run = member.route.Start(workspace);
    EXPECT_TRUE(run);
    if (!run) continue;
    auto& target = out[{member.output, member.part}];
    const PartSink sink = [&](std::uint64_t at,
                              std::span<const std::byte> bytes) -> std::expected<void, Error> {
      std::memcpy(target.data() + member.destination + at, bytes.data(), bytes.size());
      return {};
    };
    const auto source = file.subspan(static_cast<std::size_t>(member.extent.offset),
                                     static_cast<std::size_t>(member.extent.bytes));
    // Odd fragments, as a stream delivers them.
    for (std::size_t at = 0; at < source.size();) {
      const auto count = std::min<std::size_t>(997, source.size() - at);
      EXPECT_TRUE(run->Consume(at, source.subspan(at, count), sink));
      at += count;
    }
    EXPECT_TRUE(run->Finish());
  }
  return out;
}

std::vector<float> Decode(const TensorFormat& f,
                          std::span<const kernels::reference::PartView> parts) {
  std::vector<float> out(f.shape.ne[0] * f.shape.ne[1]);
  const auto ok = kernels::reference::DecodeRows(f, parts, 0, 0, f.shape.ne[1], out);
  EXPECT_TRUE(ok) << (ok ? "" : kernels::reference::DecodeErrorName(ok.error()));
  return out;
}

TEST(ImportExpertRelayoutTest, RelaidTablesDecodeToTheSplitSourcesValues) {
  // Gate and up [K 64, N 128], down [K 128, N 128]: whole atoms.
  const std::array<std::pair<std::string, TensorFormat>, 3> projections{
      {{"gate", Split(64, 128)}, {"up", Split(64, 128)}, {"down", Split(128, 128)}}};
  Source s;
  for (std::uint32_t e = 0; e < kExperts; ++e)
    for (std::size_t p = 0; p < projections.size(); ++p)
      for (const auto role : {PartRole::kCodes, PartRole::kBlockScales, PartRole::kTensorScale})
        s.Add(std::format("{}.{}.{}", projections[p].first, e, static_cast<int>(role)),
              projections[p].second, role, static_cast<std::uint32_t>((e * 3) + p));
  s.checkpoint.container = "safetensors";
  s.checkpoint.files.push_back({.name = "model.safetensors", .bytes = s.file.size()});

  std::vector<TensorAssembly> outputs;
  std::vector<ExpertTable> tables;
  for (std::size_t p = 0; p < projections.size(); ++p) {
    const auto& [name, f] = projections[p];
    TensorAssembly a{.name = "blk.0.ffn_" + name + "_exps.weight",
                     .format = Split(f.shape.ne[0], f.shape.ne[1], kExperts)};
    for (const auto role : {PartRole::kCodes, PartRole::kBlockScales, PartRole::kTensorScale}) {
      PartAssembly part{.part = role, .kind = PartAssemblyKind::kMatrices, .sources = {}};
      for (std::uint32_t e = 0; e < kExperts; ++e)
        part.sources.push_back(
            s.index.at(std::format("{}.{}.{}", name, e, static_cast<int>(role))));
      a.parts.push_back(std::move(part));
    }
    tables.push_back({.output = static_cast<std::uint32_t>(p),
                      .projection = p == 0   ? ExpertProjection::kGate
                                    : p == 1 ? ExpertProjection::kUp
                                             : ExpertProjection::kDown,
                      .moe = 0,
                      .name = a.name,
                      .stacked = p == 0 ? "blk.0.ffn_gate_up_exps.weight" : "",
                      .scales = a.name + "_scale_2"});
    outputs.push_back(std::move(a));
  }
  auto summary = RelayoutExpertTables(outputs, tables);
  ASSERT_TRUE(summary);
  EXPECT_EQ(summary->tables, 2U);
  EXPECT_EQ(summary->stacked, 1U);
  EXPECT_EQ(summary->scale_vectors, 3U);
  ASSERT_EQ(outputs.size(), 5U);
  EXPECT_EQ(outputs[0].name, "blk.0.ffn_gate_up_exps.weight");
  EXPECT_EQ(outputs[0].format.layout, Layout::kScaleAtoms);
  EXPECT_EQ(outputs[0].format.stacked, 2U);
  EXPECT_EQ(outputs[0].format.shape.ne[1], 256U);
  EXPECT_FALSE(outputs[0].format.params.tensor_scale);
  EXPECT_EQ(outputs[1].name, "blk.0.ffn_down_exps.weight");
  EXPECT_EQ(outputs[2].name, "blk.0.ffn_gate_exps.weight_scale_2");
  EXPECT_EQ(outputs[2].format.encoding, Encoding::kF32);
  EXPECT_EQ(outputs[2].format.shape.ne[0], kExperts);

  // The relaid form is no larger than the source.
  std::uint64_t before = 0, after = 0;
  for (const auto& t : s.checkpoint.tensors) before += t.extent.bytes;
  std::vector<SourceDisposition> dispositions(s.checkpoint.tensors.size());
  auto cost = IndexedPartPlan::Bound(s.checkpoint, outputs);
  ASSERT_TRUE(cost);
  auto plan = IndexedPartPlan::Make(std::move(s.checkpoint), outputs, dispositions, *cost);
  ASSERT_TRUE(plan) << plan.error().reason;
  for (const auto& out : plan->outputs())
    for (const auto& part : out.parts.parts()) after += part.bytes;
  EXPECT_EQ(after, before);
  EXPECT_EQ(plan->outputs()[0].relayout_from, Layout::kSplit);
  EXPECT_EQ(plan->outputs()[1].relayout_from, Layout::kSplit);
  EXPECT_FALSE(plan->outputs()[2].relayout_from);

  const auto published = Publish(*plan, s.file);
  const auto& src = plan->source();
  const auto bytes_of = [&](const std::string& name) {
    const auto& t = src.tensors[s.index.at(name)];
    return std::span<const std::byte>(s.file).subspan(static_cast<std::size_t>(t.extent.offset),
                                                      static_cast<std::size_t>(t.extent.bytes));
  };
  for (std::uint32_t e = 0; e < kExperts; ++e) {
    SCOPED_TRACE(e);
    for (std::uint32_t o = 0; o < 2; ++o) {
      // The relaid table's expert e, one slice of each part.
      auto slice = plan->outputs()[o].format;
      slice.shape.ne[2] = 1;
      slice.shape.rank = 2;
      const auto slice_parts = format::PartsOf(slice);
      ASSERT_TRUE(slice_parts);
      const auto codes_bytes = slice_parts->Find(PartRole::kCodes)->bytes;
      const auto scale_bytes = slice_parts->Find(PartRole::kBlockScales)->bytes;
      const std::array<kernels::reference::PartView, 2> relaid{
          {{PartRole::kCodes, std::span(published.at({o, PartRole::kCodes}))
                                  .subspan(static_cast<std::size_t>(e * codes_bytes),
                                           static_cast<std::size_t>(codes_bytes))},
           {PartRole::kBlockScales, std::span(published.at({o, PartRole::kBlockScales}))
                                        .subspan(static_cast<std::size_t>(e * scale_bytes),
                                                 static_cast<std::size_t>(scale_bytes))}}};
      const auto now = Decode(slice, relaid);
      // The split sources' values (without their tensor scales, which the
      // vectors carry), gate's rows then up's.
      std::vector<float> was;
      for (const std::string& name :
           o == 0 ? std::vector<std::string>{"gate", "up"} : std::vector<std::string>{"down"}) {
        auto f = projections[name == "gate" ? 0 : name == "up" ? 1 : 2].second;
        f.params.tensor_scale = false;
        const std::array<kernels::reference::PartView, 2> split{
            {{PartRole::kCodes,
              bytes_of(std::format("{}.{}.{}", name, e, static_cast<int>(PartRole::kCodes)))},
             {PartRole::kBlockScales,
              bytes_of(
                  std::format("{}.{}.{}", name, e, static_cast<int>(PartRole::kBlockScales)))}}};
        const auto values = Decode(f, split);
        was.insert(was.end(), values.begin(), values.end());
      }
      ASSERT_EQ(now.size(), was.size());
      EXPECT_EQ(std::memcmp(now.data(), was.data(), now.size() * sizeof(float)), 0);
    }
    // Each projection's tensor scale, gathered in expert order.
    for (std::uint32_t v = 0; v < 3; ++v) {
      const auto& vector = published.at({2 + v, PartRole::kData});
      const std::string name = std::array{"gate", "up", "down"}[v];
      const auto scale =
          bytes_of(std::format("{}.{}.{}", name, e, static_cast<int>(PartRole::kTensorScale)));
      EXPECT_EQ(std::memcmp(vector.data() + (4 * e), scale.data(), 4), 0);
    }
  }
}

// A row table gathered from split row blocks, row-interleaved: each row's
// codes then its scales, decoding to the blocks' values in order.
TEST(ImportExpertRelayoutTest, RowTablesInterleaveToTheSplitBlocksValues) {
  constexpr std::uint64_t kK = 64;
  const std::array<std::uint64_t, 2> blocks{5, 3};
  Source s;
  for (std::size_t b = 0; b < blocks.size(); ++b)
    for (const auto role : {PartRole::kCodes, PartRole::kBlockScales})
      s.Add(std::format("rows.{}.{}", b, static_cast<int>(role)), Split(kK, blocks[b]), role,
            static_cast<std::uint32_t>(b + 7));
  s.Add("scale", Split(kK, blocks[0]), PartRole::kTensorScale, 0);
  s.checkpoint.container = "safetensors";
  s.checkpoint.files.push_back({.name = "model.safetensors", .bytes = s.file.size()});
  TensorAssembly table{.name = "table.weight", .format = Split(kK, 8)};
  for (const auto role : {PartRole::kCodes, PartRole::kBlockScales}) {
    PartAssembly part{.part = role, .kind = PartAssemblyKind::kRows, .sources = {}};
    for (std::size_t b = 0; b < blocks.size(); ++b)
      part.sources.push_back(s.index.at(std::format("rows.{}.{}", b, static_cast<int>(role))));
    table.parts.push_back(std::move(part));
  }
  table.parts.push_back({.part = PartRole::kTensorScale,
                         .kind = PartAssemblyKind::kSharedTensorScale,
                         .sources = {s.index.at("scale")}});
  auto relaid = RelayoutRowTable(table);
  ASSERT_TRUE(relaid && *relaid);
  EXPECT_EQ(table.format.layout, Layout::kRowInterleaved);
  EXPECT_EQ(table.relayout_from, Layout::kSplit);
  ASSERT_EQ(table.parts.size(), 2U);
  EXPECT_EQ(table.parts[0].kind, PartAssemblyKind::kInterleavedRows);
  // Not a split row table: unchanged.
  auto again = table;
  EXPECT_FALSE(*RelayoutRowTable(again));
  std::vector<TensorAssembly> outputs{table};
  std::vector<SourceDisposition> dispositions(s.checkpoint.tensors.size());
  auto cost = IndexedPartPlan::Bound(s.checkpoint, outputs);
  ASSERT_TRUE(cost);
  auto plan = IndexedPartPlan::Make(std::move(s.checkpoint), outputs, dispositions, *cost);
  ASSERT_TRUE(plan) << plan.error().reason;
  const auto published = Publish(*plan, s.file);
  const std::array<kernels::reference::PartView, 2> interleaved{
      {{PartRole::kData, published.at({0, PartRole::kData})},
       {PartRole::kTensorScale, published.at({0, PartRole::kTensorScale})}}};
  const auto now = Decode(table.format, interleaved);
  std::vector<float> was;
  const auto& src = plan->source();
  const auto bytes_of = [&](const std::string& name) {
    const auto& t = src.tensors[s.index.at(name)];
    return std::span<const std::byte>(s.file).subspan(static_cast<std::size_t>(t.extent.offset),
                                                      static_cast<std::size_t>(t.extent.bytes));
  };
  for (std::size_t b = 0; b < blocks.size(); ++b) {
    const std::array<kernels::reference::PartView, 3> split{
        {{PartRole::kCodes,
          bytes_of(std::format("rows.{}.{}", b, static_cast<int>(PartRole::kCodes)))},
         {PartRole::kBlockScales,
          bytes_of(std::format("rows.{}.{}", b, static_cast<int>(PartRole::kBlockScales)))},
         {PartRole::kTensorScale, bytes_of("scale")}}};
    const auto values = Decode(Split(kK, blocks[b]), split);
    was.insert(was.end(), values.begin(), values.end());
  }
  ASSERT_EQ(now.size(), was.size());
  EXPECT_EQ(std::memcmp(now.data(), was.data(), now.size() * sizeof(float)), 0);
}

// Canonical expert-table names by role; anything else, a malformed layer
// index among them, names none.
// Only an encoding a device kernel reads row-interleaved (NVFP4) is relaid
// out; an MXFP8 or MXFP4 row table keeps its split blocks, which its row
// lookups read.
TEST(ImportExpertRelayoutTest, RowTablesRelayOutOnlyWhatAKernelReadsInterleaved) {
  for (const auto encoding : {Encoding::kNvfp4, Encoding::kMxfp8, Encoding::kMxfp4}) {
    SCOPED_TRACE(static_cast<int>(encoding));
    TensorFormat f{
        .encoding = encoding, .layout = Layout::kSplit, .shape = {.ne = {64, 8, 1, 1}, .rank = 2}};
    TensorAssembly table{.name = "table.weight", .format = f};
    for (const auto role : {PartRole::kCodes, PartRole::kBlockScales})
      table.parts.push_back({.part = role, .kind = PartAssemblyKind::kRows, .sources = {0}});
    const auto before = table.format;
    auto relaid = RelayoutRowTable(table);
    ASSERT_TRUE(relaid);
    EXPECT_EQ(*relaid, encoding == Encoding::kNvfp4);
    if (encoding != Encoding::kNvfp4) {
      EXPECT_EQ(table.format, before);
      EXPECT_FALSE(table.relayout_from);
      EXPECT_EQ(table.parts.size(), 2U);
    }
  }
}

TEST(ImportExpertRelayoutTest, NamesExpertTablesByCanonicalRole) {
  const auto gate = ExpertTableOf("blk.12.ffn_gate_exps.weight", 3);
  ASSERT_TRUE(gate);
  EXPECT_EQ(gate->output, 3U);
  EXPECT_EQ(gate->moe, 12U);
  EXPECT_EQ(gate->projection, ExpertProjection::kGate);
  EXPECT_EQ(gate->name, "blk.12.ffn_gate_exps.weight");
  EXPECT_EQ(gate->stacked, "blk.12.ffn_gate_up_exps.weight");
  EXPECT_EQ(gate->scales, "blk.12.ffn_gate_exps.weight_scale_2");
  const auto down = ExpertTableOf("blk.0.ffn_down_exps.weight", 0);
  ASSERT_TRUE(down);
  EXPECT_EQ(down->projection, ExpertProjection::kDown);
  EXPECT_TRUE(down->stacked.empty());
  EXPECT_EQ(ExpertTableOf("blk.0.ffn_up_exps.weight", 0)->projection, ExpertProjection::kUp);
  for (const std::string_view bad :
       {"blk..ffn_gate_exps.weight", "blk.01.ffn_gate_exps.weight", "blk.x.ffn_gate_exps.weight",
        "blk.-1.ffn_gate_exps.weight", "blk.65536.ffn_gate_exps.weight",
        "blk.123456.ffn_gate_exps.weight", "blk.1ffn_gate_exps.weight", "blk_exps.weight",
        "blk.1.ffn_gate_up_exps.weight", "blk.1.ffn_gate_exps.weight_scale",
        "blk.1.ffn_gate_shexp.weight", "blk.1.2.ffn_gate_exps.weight", "token_embd.weight"})
    EXPECT_FALSE(ExpertTableOf(bad, 0)) << bad;
}

// A gate without its MoE's up is relaid out alone, under its own name; a
// table missing a part is refused, not dereferenced.
TEST(ImportExpertRelayoutTest, LoneGatesKeepTheirNameAndPartlessTablesAreRefused) {
  std::vector<TensorAssembly> outputs(1);
  outputs[0] = {.name = "blk.0.ffn_gate_exps.weight", .format = Split(64, 128, 2)};
  for (const auto role : {PartRole::kCodes, PartRole::kBlockScales, PartRole::kTensorScale})
    outputs[0].parts.push_back(
        {.part = role, .kind = PartAssemblyKind::kMatrices, .sources = {0, 1}});
  const auto gate = ExpertTableOf(outputs[0].name, 0);
  ASSERT_TRUE(gate);
  auto lone = outputs;
  auto relaid = RelayoutExpertTables(lone, std::span(&*gate, 1));
  ASSERT_TRUE(relaid);
  EXPECT_EQ(relaid->stacked, 0U);
  ASSERT_EQ(lone.size(), 2U);
  EXPECT_EQ(lone[0].name, "blk.0.ffn_gate_exps.weight");
  EXPECT_EQ(lone[0].format.stacked, 1U);
  EXPECT_EQ(lone[1].name, "blk.0.ffn_gate_exps.weight_scale_2");
  auto partless = outputs;
  partless[0].parts.erase(partless[0].parts.begin() + 1);  // its block scales
  EXPECT_FALSE(RelayoutExpertTables(partless, std::span(&*gate, 1)));
}

TEST(ImportExpertRelayoutTest, KeepsTablesItCannotRelayLosslesslyAndRefusesBadLists) {
  std::vector<TensorAssembly> outputs(1);
  outputs[0] = {.name = "blk.0.ffn_down_exps.weight", .format = Split(64, 100, 2)};
  for (const auto role : {PartRole::kCodes, PartRole::kBlockScales, PartRole::kTensorScale})
    outputs[0].parts.push_back(
        {.part = role, .kind = PartAssemblyKind::kMatrices, .sources = {0, 1}});
  const std::array<ExpertTable, 1> down{{{.output = 0,
                                          .projection = ExpertProjection::kDown,
                                          .name = "blk.0.ffn_down_exps.weight",
                                          .scales = "s"}}};
  auto kept = RelayoutExpertTables(outputs, down);
  ASSERT_TRUE(kept);
  EXPECT_EQ(kept->tables, 0U);
  ASSERT_EQ(outputs.size(), 1U);
  EXPECT_EQ(outputs[0].format.layout, Layout::kSplit);
  const std::array<ExpertTable, 2> twice{down[0], down[0]};
  EXPECT_FALSE(RelayoutExpertTables(outputs, twice));
  const std::array<ExpertTable, 1> outside{{{.output = 1, .name = "x"}}};
  EXPECT_FALSE(RelayoutExpertTables(outputs, outside));
}

TEST(ImportExpertRelayoutTest, RelayoutExpertTablesWithNullSpec) {
  std::vector<TensorAssembly> outputs(1);
  outputs[0] = {.name = "blk.0.ffn_gate_exps.weight", .format = Split(64, 128, 2)};
  for (const auto role : {PartRole::kCodes, PartRole::kBlockScales, PartRole::kTensorScale})
    outputs[0].parts.push_back(
        {.part = role, .kind = PartAssemblyKind::kMatrices, .sources = {0, 1}});
  auto relaid = RelayoutExpertTables(outputs, nullptr);
  ASSERT_TRUE(relaid);
  EXPECT_EQ(relaid->tables, 1U);
}

TEST(ImportExpertRelayoutTest, RelayoutExpertTablesWithSpec) {
  std::vector<TensorAssembly> outputs(1);
  outputs[0] = {.name = "custom.gate.weight", .format = Split(64, 128, 2)};
  for (const auto role : {PartRole::kCodes, PartRole::kBlockScales, PartRole::kTensorScale})
    outputs[0].parts.push_back(
        {.part = role, .kind = PartAssemblyKind::kMatrices, .sources = {0, 1}});
  spec::ModelSpec spec;
  spec::Stage stage;
  spec::Layer layer;
  spec::ComponentRef comp;
  comp.kind = "ffn.moe";
  comp.weights.push_back({.role = "gate_experts", .tensor = "custom.gate.weight"});
  layer.components.push_back(std::move(comp));
  stage.layers.push_back(std::move(layer));
  spec.stages.push_back(std::move(stage));

  auto relaid = RelayoutExpertTables(outputs, &spec);
  ASSERT_TRUE(relaid);
  EXPECT_EQ(relaid->tables, 1U);
  EXPECT_EQ(outputs[0].format.layout, Layout::kScaleAtoms);
}

}  // namespace
}  // namespace llmp::import
