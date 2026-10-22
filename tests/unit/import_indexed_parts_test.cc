// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <map>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "import/indexed_parts.h"

namespace llmp::import {
namespace {
using format::Encoding;
using format::Layout;
using format::PartRole;
using Key = std::pair<std::uint32_t, PartRole>;
using Payloads = std::map<Key, std::vector<std::byte>>;
format::TensorFormat Fmt(Encoding encoding, std::initializer_list<std::uint64_t> dims) {
  auto shape = format::ShapeOf(std::span(dims.begin(), dims.size()));
  EXPECT_TRUE(shape);
  format::EncodingParams params{.tensor_scale = encoding == Encoding::kNvfp4};
  if (encoding == Encoding::kAffine)
    params = {.bits = 4, .group = 32, .scale = format::AffineScale::kBF16};
  return {.encoding = encoding,
          .params = params,
          .layout = encoding == Encoding::kNvfp4 || encoding == Encoding::kMxfp8 ||
                            encoding == Encoding::kAffine
                        ? Layout::kSplit
                        : Layout::kDense,
          .shape = shape.value_or(format::Shape{})};
}
struct Fixture {
  Checkpoint source{};
  std::vector<std::vector<std::byte>> payloads{};
  std::vector<TensorAssembly> outputs{};
  std::vector<SourceDisposition> dispositions{};
  Fixture() { source.files.push_back({.name = "source", .bytes = 16}); }
  std::uint32_t Add(std::string name, format::TensorFormat format, PartRole part, unsigned seed) {
    auto parts = format::PartsOf(format);
    EXPECT_TRUE(parts);
    const auto bytes = parts->Find(part)->bytes;
    const auto index = static_cast<std::uint32_t>(source.tensors.size());
    source.tensors.push_back(
        {.name = std::move(name),
         .format = format,
         .part = part,
         .extent = {.file = 0, .offset = source.files[0].bytes, .bytes = bytes}});
    source.files[0].bytes += bytes + 13;
    payloads.emplace_back(static_cast<std::size_t>(bytes));
    for (std::size_t i = 0; i < payloads.back().size(); ++i)
      payloads.back()[i] = static_cast<std::byte>((seed + i * 29) % 251);
    dispositions.push_back({});
    return index;
  }
  auto Build() const {
    auto cost = IndexedPartPlan::Bound(source, outputs);
    if (!cost) return std::expected<IndexedPartPlan, Error>(std::unexpected(cost.error()));
    auto owned_source = source;  // Fixture-owned, explicitly allocated before admission.
    return IndexedPartPlan::Make(std::move(owned_source), outputs, dispositions,
                                 cost.value_or(IndexedPartCost{}));
  }
};

void Reconstruct(const IndexedPartPlan& plan, const Fixture& fixture, const Payloads& expected) {
  for (std::size_t chunk : {1U, 3U, 17U, 4096U}) {
    Payloads actual;
    std::map<Key, std::vector<unsigned>> counts;
    for (std::size_t oi = 0; oi < plan.outputs().size(); ++oi) {
      for (const auto& part : plan.outputs()[oi].parts.parts()) {
        const Key key{static_cast<std::uint32_t>(oi), part.role};
        actual[key].resize(static_cast<std::size_t>(part.bytes) + 2, std::byte{0xCE});
        counts[key].resize(static_cast<std::size_t>(part.bytes));
      }
    }
    std::uint64_t previous_end = 0;
    for (auto mi : plan.physical_order()) {
      const auto& member = plan.members()[mi];
      EXPECT_GE(member.extent.offset, previous_end);
      previous_end = member.extent.offset + member.extent.bytes;
      const Key key{member.output, member.part};
      auto& bytes = actual.at(key);
      auto& written = counts.at(key);
      std::array<std::byte, 65536> workspace{};
      auto run = member.route.Start(workspace);
      ASSERT_TRUE(run);
      EXPECT_FALSE(run->Finish());
      const PartSink sink = [&](std::uint64_t offset,
                                std::span<const std::byte> input) -> std::expected<void, Error> {
        const auto at = member.destination + offset;
        if (at > written.size() || input.size() > written.size() - at)
          return std::unexpected(Error{.rule = Rule::kBounds, .reason = "test sink outside part"});
        for (std::size_t i = 0; i < input.size(); ++i) {
          const auto dest = static_cast<std::size_t>(at) + i;
          bytes[dest + 1] = input[i];
          ++written[dest];
        }
        return {};
      };
      const auto input = std::span(fixture.payloads[member.source]);
      for (std::size_t at = 0; at < input.size();) {
        const auto take = std::min(chunk, input.size() - at);
        ASSERT_TRUE(run->Consume(at, input.subspan(at, take), sink));
        at += take;
      }
      ASSERT_TRUE(run->Finish());
    }
    ASSERT_EQ(actual.size(), expected.size());
    for (const auto& [key, bytes] : actual) {
      EXPECT_EQ(bytes.front(), std::byte{0xCE});
      EXPECT_EQ(bytes.back(), std::byte{0xCE});
      EXPECT_TRUE(
          std::ranges::equal(std::span(bytes).subspan(1, bytes.size() - 2), expected.at(key)));
      EXPECT_TRUE(std::ranges::all_of(counts.at(key), [](auto n) { return n == 1; }));
    }
  }
}

TEST(ImportIndexedPartsTest, AllExpertPartsGatherInNumericOrderFromPhysicalOrder) {
  Fixture f;
  const auto matrix = Fmt(Encoding::kNvfp4, {32, 3});
  const auto gathered = Fmt(Encoding::kNvfp4, {32, 3, 3});
  std::array<std::array<std::uint32_t, 3>, 3> ids{};
  constexpr std::array roles = {PartRole::kCodes, PartRole::kBlockScales, PartRole::kTensorScale};
  // Files deliberately store expert2, expert0, expert1 and scale before code.
  for (std::size_t expert : {2U, 0U, 1U})
    for (std::size_t p : {2U, 1U, 0U})
      ids[p][expert] = f.Add("expert" + std::to_string(expert) + ".part" + std::to_string(p),
                             matrix, roles[p], static_cast<unsigned>(expert * 53 + p * 17));
  TensorAssembly out{.name = "experts", .format = gathered};
  Payloads expected;
  for (std::size_t p = 0; p < roles.size(); ++p) {
    out.parts.push_back({.part = roles[p],
                         .kind = PartAssemblyKind::kMatrices,
                         .sources = {ids[p][0], ids[p][1], ids[p][2]}});
    auto& bytes = expected[{0, roles[p]}];
    for (auto id : ids[p]) bytes.insert(bytes.end(), f.payloads[id].begin(), f.payloads[id].end());
  }
  f.outputs.push_back(std::move(out));
  auto plan = f.Build();
  ASSERT_TRUE(plan);
  EXPECT_EQ(plan->source().tensors.size(), 9U);
  EXPECT_EQ(plan->members().size(), 9U);
  EXPECT_EQ(plan->FindSource("expert1.part0"), ids[0][1]);
  EXPECT_FALSE(plan->FindSource("expert01.part0"));
  Reconstruct(*plan, f, expected);
}

TEST(ImportIndexedPartsTest, NarrowTensorScaleConversionPreservesPackedWeightParts) {
  for (const auto storage : {Encoding::kBF16, Encoding::kF16}) {
    Fixture fixture;
    const auto format = Fmt(Encoding::kNvfp4, {32, 3});
    TensorAssembly output{.name = "weight", .format = format};
    Payloads expected;
    for (const auto role : {PartRole::kCodes, PartRole::kBlockScales, PartRole::kTensorScale}) {
      const auto source = fixture.Add(std::string(format::PartRoleName(role)), format, role, 9);
      output.parts.push_back({.part = role, .sources = {source}});
      if (role == PartRole::kTensorScale) {
        auto& tensor = fixture.source.tensors[source];
        tensor.scalar_storage = storage;
        tensor.extent.bytes = 2;
        fixture.payloads[source] = storage == Encoding::kBF16
                                       ? std::vector{std::byte{0x80}, std::byte{0x3F}}
                                       : std::vector{std::byte{0}, std::byte{0x3C}};
        expected[{0, role}] = {std::byte{0}, std::byte{0}, std::byte{0x80}, std::byte{0x3F}};
      } else {
        expected[{0, role}] = fixture.payloads[source];
      }
    }
    fixture.outputs.push_back(std::move(output));
    const auto plan = fixture.Build();
    ASSERT_TRUE(plan) << plan.error().ToString();
    EXPECT_EQ(plan->outputs().front().format, format);
    Reconstruct(*plan, fixture, expected);
    auto& scale = fixture.source.tensors.back();
    scale.scalar_storage = Encoding::kF32;
    EXPECT_FALSE(SourcePartBytes(scale));
    EXPECT_FALSE(fixture.Build());
    scale.scalar_storage = storage;
    scale.extent.bytes = 4;
    EXPECT_FALSE(fixture.Build());
    scale.extent.bytes = 2;
    scale.part = PartRole::kCodes;
    EXPECT_FALSE(SourcePartBytes(scale));
    EXPECT_FALSE(fixture.Build());
  }
}

TEST(ImportIndexedPartsTest, MixedScalarStorageGathersCanonicalF32InDeclaredOrder) {
  Fixture fixture;
  const auto format = Fmt(Encoding::kNvfp4, {32, 3});
  std::vector<std::uint32_t> sources;
  for (const auto storage : {Encoding::kBF16, Encoding::kF16, Encoding::kF32}) {
    const auto source =
        fixture.Add(std::to_string(sources.size()), format, PartRole::kTensorScale, 0);
    sources.push_back(source);
    if (storage != Encoding::kF32) {
      fixture.source.tensors[source].scalar_storage = storage;
      fixture.source.tensors[source].extent.bytes = 2;
      fixture.payloads[source] = storage == Encoding::kBF16
                                     ? std::vector{std::byte{0x80}, std::byte{0x3F}}
                                     : std::vector{std::byte{0}, std::byte{0x3C}};
    } else {
      fixture.payloads[source] = {std::byte{0}, std::byte{0}, std::byte{0x80}, std::byte{0x3F}};
    }
  }
  fixture.outputs.push_back(
      {.name = "scales",
       .format = Fmt(Encoding::kF32, {3}),
       .parts = {
           {.part = PartRole::kData, .kind = PartAssemblyKind::kScalars, .sources = sources}}});
  Payloads expected;
  for (std::size_t i = 0; i < sources.size(); ++i)
    for (const auto byte : {std::byte{0}, std::byte{0}, std::byte{0x80}, std::byte{0x3F}})
      expected[{0, PartRole::kData}].push_back(byte);
  const auto plan = fixture.Build();
  ASSERT_TRUE(plan) << plan.error().ToString();
  Reconstruct(*plan, fixture, expected);
}

TEST(ImportIndexedPartsTest, SourceInventoryAcceptsReaderShardCountsAbove256) {
  Fixture fixture;
  const auto format = Fmt(Encoding::kF32, {4});
  const auto source = fixture.Add("weight", format, PartRole::kData, 7);
  fixture.outputs.push_back({.name = "weight",
                             .format = format,
                             .parts = {{.part = PartRole::kData, .sources = {source}}}});
  for (std::size_t i = fixture.source.files.size(); i < 300; ++i)
    fixture.source.files.push_back({.name = "shard-" + std::to_string(i)});
  const auto plan = fixture.Build();
  ASSERT_TRUE(plan) << plan.error().ToString();
  EXPECT_EQ(plan->source().files.size(), 300U);
  Reconstruct(*plan, fixture, {{{0, PartRole::kData}, fixture.payloads[source]}});
  fixture.source.files.back().name = fixture.source.files.front().name;
  EXPECT_FALSE(fixture.Build());
}

TEST(ImportIndexedPartsTest, RowShardsUseOneInheritedScaleWithoutReplication) {
  Fixture f;
  const auto shard = Fmt(Encoding::kNvfp4, {32, 2});
  const auto table = Fmt(Encoding::kNvfp4, {32, 6});
  std::array<std::array<std::uint32_t, 3>, 2> ids{};
  constexpr std::array roles = {PartRole::kCodes, PartRole::kBlockScales};
  for (std::size_t s : {2U, 1U, 0U})
    for (std::size_t p = 0; p < roles.size(); ++p)
      ids[p][s] = f.Add("shard" + std::to_string(s) + ".part" + std::to_string(p), shard, roles[p],
                        static_cast<unsigned>(s * 31 + p));
  const auto scale = f.Add("shared_scale", shard, PartRole::kTensorScale, 93);
  TensorAssembly out{.name = "table", .format = table};
  Payloads expected;
  for (std::size_t p = 0; p < roles.size(); ++p) {
    out.parts.push_back({.part = roles[p],
                         .kind = PartAssemblyKind::kRows,
                         .sources = {ids[p][0], ids[p][1], ids[p][2]}});
    for (auto id : ids[p])
      expected[{0, roles[p]}].insert(expected[{0, roles[p]}].end(), f.payloads[id].begin(),
                                     f.payloads[id].end());
  }
  out.parts.push_back({.part = PartRole::kTensorScale,
                       .kind = PartAssemblyKind::kSharedTensorScale,
                       .sources = {scale}});
  expected[{0, PartRole::kTensorScale}] = f.payloads[scale];
  f.outputs.push_back(std::move(out));
  auto plan = f.Build();
  ASSERT_TRUE(plan);
  EXPECT_EQ(plan->members().size(), 7U);
  Reconstruct(*plan, f, expected);
}

TEST(ImportIndexedPartsTest, Bf16SingletonReshapeAndAllMxfp8PartsSharePermutation) {
  for (auto encoding : {Encoding::kBF16, Encoding::kMxfp8, Encoding::kAffine}) {
    Fixture f;
    const auto destination = Fmt(encoding, {32, 6});
    const auto source = encoding == Encoding::kBF16 ? Fmt(encoding, {32, 1, 6}) : destination;
    auto permutation = HeadPermutationPlan::Make(destination, HeadAxis::kRows, 2, 6, 1);
    ASSERT_TRUE(permutation);
    TensorAssembly out{.name = "heads", .format = destination, .permutation = *permutation};
    Payloads expected;
    auto parts = format::PartsOf(destination);
    ASSERT_TRUE(parts);
    for (const auto& part : parts->parts()) {
      auto id = f.Add(std::string(format::PartRoleName(part.role)), source, part.role, 9);
      out.parts.push_back({.part = part.role,
                           .kind = encoding == Encoding::kBF16 ? PartAssemblyKind::kDenseReshape
                                                               : PartAssemblyKind::kCopy,
                           .sources = {id}});
      const auto row_bytes = static_cast<std::size_t>(part.bytes / 6);
      auto& bytes = expected[{0, part.role}];
      // Independent whole-row oracle: grouped 0,1,2 / 3,4,5 -> 0,3,1,4,2,5.
      for (std::size_t row : {0U, 3U, 1U, 4U, 2U, 5U}) {
        const auto begin = f.payloads[id].begin() + static_cast<std::ptrdiff_t>(row * row_bytes);
        bytes.insert(bytes.end(), begin, begin + static_cast<std::ptrdiff_t>(row_bytes));
      }
    }
    f.outputs.push_back(std::move(out));
    auto plan = f.Build();
    ASSERT_TRUE(plan);
    Reconstruct(*plan, f, expected);
  }
}

TEST(ImportIndexedPartsTest, AffineSingletonReshapeSucceeds) {
  Fixture f;
  const auto destination = Fmt(Encoding::kAffine, {32});
  const auto source = Fmt(Encoding::kAffine, {32, 1});
  TensorAssembly out{.name = "gate", .format = destination};
  Payloads expected;
  auto parts = format::PartsOf(destination);
  ASSERT_TRUE(parts);
  for (const auto& part : parts->parts()) {
    auto id = f.Add(std::string(format::PartRoleName(part.role)), source, part.role, 9);
    out.parts.push_back(
        {.part = part.role, .kind = PartAssemblyKind::kDenseReshape, .sources = {id}});
    expected[{0, part.role}] = f.payloads[id];
  }
  f.outputs.push_back(std::move(out));
  auto plan = f.Build();
  ASSERT_TRUE(plan);
  Reconstruct(*plan, f, expected);
}

Fixture Dense() {
  Fixture f;
  auto format = Fmt(Encoding::kBF16, {4, 2});
  const auto id = f.Add("weight", format, PartRole::kData, 3);
  f.outputs.push_back(
      {.name = "out", .format = format, .parts = {{.part = PartRole::kData, .sources = {id}}}});
  return f;
}
TEST(ImportIndexedPartsTest, RequiresEverySourceDispositionAndEveryOutputPart) {
  auto f = Dense();
  const auto extra = f.Add("semantic", Fmt(Encoding::kI64, {3}), PartRole::kData, 11);
  EXPECT_FALSE(f.Build());
  f.dispositions[extra] = {.kind = SourceDispositionKind::kSemantic, .rule = 7};
  ASSERT_TRUE(f.Build());
  f.dispositions[extra].rule = 0;
  EXPECT_FALSE(f.Build());
  f.dispositions[extra] = {.kind = SourceDispositionKind::kAuxiliary, .rule = 8};
  ASSERT_TRUE(f.Build());
  f.dispositions[0] = {.kind = SourceDispositionKind::kAuxiliary, .rule = 9};
  EXPECT_FALSE(f.Build());
  f = Dense();
  f.outputs[0].format = Fmt(Encoding::kMxfp8, {32, 2});
  EXPECT_FALSE(f.Build());
  f = Dense();
  f.outputs[0].parts.push_back(f.outputs[0].parts[0]);
  EXPECT_FALSE(f.Build());
}
TEST(ImportIndexedPartsTest, RefusesAliasesMutationOverflowAndUnfundedConstruction) {
  auto f = Dense();
  auto cost = IndexedPartPlan::Bound(f.source, f.outputs);
  ASSERT_TRUE(cost);
  const auto* original_tensors = f.source.tensors.data();
  const auto* original_files = f.source.files.data();
  auto short_cost = *cost;
  --short_cost.metadata_bytes;
  EXPECT_FALSE(IndexedPartPlan::Make(std::move(f.source), f.outputs, f.dispositions, short_cost));
  EXPECT_EQ(f.source.tensors.data(), original_tensors);
  EXPECT_EQ(f.source.files.data(), original_files);
  ASSERT_EQ(f.source.tensors.size(), 1U);
  EXPECT_EQ(f.source.tensors[0].name, "weight");
  short_cost = *cost;
  --short_cost.work_units;
  EXPECT_FALSE(IndexedPartPlan::Make(std::move(f.source), f.outputs, f.dispositions, short_cost));
  f.outputs.push_back(f.outputs[0]);
  EXPECT_FALSE(IndexedPartPlan::Bound(f.source, f.outputs));
  f = Dense();
  const auto extra = f.Add("other", f.source.tensors[0].format, PartRole::kData, 8);
  f.dispositions[extra] = {.kind = SourceDispositionKind::kAuxiliary, .rule = 1};
  f.source.tensors[extra].extent.offset = f.source.tensors[0].extent.offset;
  EXPECT_FALSE(f.Build());
  f.source.tensors[extra].extent.offset = f.source.files[0].bytes;
  EXPECT_FALSE(f.Build());
  f = Dense();
  f.outputs[0].format.shape.ne[0] = std::numeric_limits<std::uint64_t>::max();
  EXPECT_FALSE(f.Build());
  f = Dense();
  f.outputs[0].parts[0].kind = PartAssemblyKind::kDenseReshape;
  f.outputs[0].format = Fmt(Encoding::kBF16, {2, 4});
  EXPECT_FALSE(f.Build());
  f = Dense();
  f.outputs[0].format.encoding = Encoding::kF16;
  EXPECT_FALSE(f.Build());
  f = Dense();
  f.source.tensors[0].extent.bytes -= 1;
  EXPECT_FALSE(f.Build());
}
TEST(ImportIndexedPartsTest, RefusesDuplicateNamesIndicesAndUnknownOperations) {
  auto f = Dense();
  auto extra = f.Add("other", f.source.tensors[0].format, PartRole::kData, 8);
  f.dispositions[extra] = {.kind = SourceDispositionKind::kAuxiliary, .rule = 1};
  f.outputs.push_back(f.outputs[0]);
  EXPECT_FALSE(f.Build());  // duplicate canonical name
  f.outputs[1].name = "different";
  EXPECT_FALSE(f.Build());  // same physical source selected twice
  f.outputs.pop_back();
  f.source.tensors[extra].name = "weight";
  EXPECT_FALSE(f.Build());  // duplicate source name
  f = Dense();
  f.outputs[0].parts[0].sources[0] = 17;
  EXPECT_FALSE(f.Build());
  f = Dense();
  f.outputs[0].parts[0].kind = static_cast<PartAssemblyKind>(255);
  EXPECT_FALSE(f.Build());
  f = Dense();
  f.dispositions[0].kind = static_cast<SourceDispositionKind>(255);
  EXPECT_FALSE(f.Build());
}

TEST(ImportIndexedPartsTest, RefusesDifferentRowPartitionsAcrossCodesAndScales) {
  Fixture f;
  const auto small = Fmt(Encoding::kMxfp8, {32, 1});
  const auto large = Fmt(Encoding::kMxfp8, {32, 2});
  const auto all = Fmt(Encoding::kMxfp8, {32, 3});
  auto c0 = f.Add("code0", small, PartRole::kCodes, 1);
  auto c1 = f.Add("code1", large, PartRole::kCodes, 2);
  auto s0 = f.Add("scale0", small, PartRole::kBlockScales, 3);
  auto s1 = f.Add("scale1", large, PartRole::kBlockScales, 4);
  f.outputs.push_back(
      {.name = "table",
       .format = all,
       .parts = {{.part = PartRole::kCodes, .kind = PartAssemblyKind::kRows, .sources = {c0, c1}},
                 {.part = PartRole::kBlockScales,
                  .kind = PartAssemblyKind::kRows,
                  .sources = {s0, s1}}}});
  ASSERT_TRUE(f.Build());
  std::ranges::reverse(f.outputs[0].parts[1].sources);
  EXPECT_FALSE(f.Build());  // aggregate bytes match; row membership does not
}

TEST(ImportIndexedPartsTest, BoundMembersRetainTerminalSourceCursorRefusal) {
  auto f = Dense();
  auto plan = f.Build();
  ASSERT_TRUE(plan);
  auto run = plan->members()[0].route.Start();
  ASSERT_TRUE(run);
  unsigned calls = 0;
  const PartSink sink = [&](std::uint64_t,
                            std::span<const std::byte>) -> std::expected<void, Error> {
    ++calls;
    return {};
  };
  EXPECT_FALSE(run->Consume(1, f.payloads[0], sink));
  EXPECT_FALSE(run->Consume(0, f.payloads[0], sink));
  EXPECT_FALSE(run->Finish());
  EXPECT_EQ(calls, 0U);
}
}  // namespace
}  // namespace llmp::import
