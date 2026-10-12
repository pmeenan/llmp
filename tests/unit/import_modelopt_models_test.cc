// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
// Metadata-only native normalization. The runner must create a confined
// process before launch: an 8 GiB combined memory.max pays process AND file
// cache, with no assumption about buffered-reader automatic readahead.
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include "base/sha256.h"
#include "components/step.h"
#include "import/modelopt_qwen38.h"
#include "import/reader.h"
#include "platform/memory_pressure.h"

namespace llmp::import {
namespace {
constexpr std::uint64_t kGiB = std::uint64_t{1} << 30U;
constexpr std::uint64_t kCombined = 8 * kGiB, kMargin = 6 * kGiB;
std::string SmallFile(const std::filesystem::path& path) {
  std::ifstream file(path);
  std::array<char, 8192> text{};
  file.read(text.data(), static_cast<std::streamsize>(text.size()));
  if (file.gcount() == static_cast<std::streamsize>(text.size()) || file.bad()) return {};
  return std::string(text.data(), static_cast<std::size_t>(file.gcount()));
}
std::uint64_t OwnMemoryMax() {
  auto text = SmallFile("/proc/self/cgroup");
  if (!text.starts_with("0::") || text.back() != '\n') return 0;
  text.erase(0, 3);
  text.pop_back();
  if (text.empty() || text[0] != '/' || text.find("..") != std::string::npos ||
      text.find('\n') != std::string::npos)
    return 0;
  auto maximum = SmallFile(std::filesystem::path("/sys/fs/cgroup" + text) / "memory.max");
  if (maximum.empty()) return 0;
  std::uint64_t bytes = 0;
  const auto parsed = std::from_chars(maximum.data(), maximum.data() + maximum.size() - 1, bytes);
  return parsed.ec == std::errc{} && parsed.ptr == maximum.data() + maximum.size() - 1 ? bytes : 0;
}
TEST(ImportModelOptModelsTest, ClosedNativeTargetOwnsEverySourceAndPreservesFormats) {
  // NOLINTNEXTLINE(concurrency-mt-unsafe): read-only test environment.
  const char* configured = std::getenv("LLMP_TEST_MODELOPT_SOURCE");
  if (configured == nullptr) GTEST_SKIP() << "no explicit pinned ModelOpt source";
  ASSERT_EQ(OwnMemoryMax(), kCombined) << "run in prepaid 8 GiB combined memory.max cgroup";
  const auto memory = platform::ReadMemoryPressure();
  ASSERT_TRUE(memory.available);
  ASSERT_GE(*memory.available, kCombined + kMargin);
  auto source = ReadCheckpoint(configured);
  ASSERT_TRUE(source) << source.error().ToString();
  ASSERT_EQ(source->tensors.size(), 301730U);
  ASSERT_EQ(source->files.size(), 38U);
  auto bound = ModelOptTargetBound(*source);
  ASSERT_TRUE(bound) << bound.error().ToString();
  // The diagnostic's 8 GiB hard domain includes reader, retained Checkpoint,
  // planner and file cache. This assertion constrains the closed planner's
  // separately derived inventory; the domain cap is not measured occupancy.
  ASSERT_LT(bound->metadata_bytes, kGiB);
  const auto* retained = source->tensors.data();
  ASSERT_FALSE(
      NormalizeModelOptTarget(std::move(*source), {bound->metadata_bytes - 1, bound->work_units}));
  ASSERT_EQ(source->tensors.data(), retained);
  // These fields determine the cache/operand/decoder contract, independently
  // of weight shapes. Every refusal precedes ownership transfer and semantics.
  for (const auto& [key, replacement] : std::array<std::pair<std::string_view, MetaValue>, 5>{
           {{"text_config.mamba_ssm_dtype", std::string("bfloat16")},
            {"text_config.dtype", std::string("float32")},
            {"dtype", std::string("float32")},
            {"text_config.is_encoder_decoder", true},
            {"is_encoder_decoder", true}}}) {
    auto entry = std::ranges::find(source->metadata, key, &MetaEntry::key);
    ASSERT_NE(entry, source->metadata.end());
    auto original = std::move(entry->value);
    entry->value = replacement;
    ASSERT_FALSE(ModelOptTargetBound(*source));
    ASSERT_FALSE(
        NormalizeModelOptTarget(std::move(*source), {bound->metadata_bytes, bound->work_units}));
    ASSERT_EQ(source->tensors.data(), retained);
    entry->value = std::move(original);
    const auto index = static_cast<std::size_t>(entry - source->metadata.begin());
    auto required = std::move(*entry);
    source->metadata.erase(entry);
    ASSERT_FALSE(ModelOptTargetBound(*source));
    ASSERT_FALSE(
        NormalizeModelOptTarget(std::move(*source), {bound->metadata_bytes, bound->work_units}));
    ASSERT_EQ(source->tensors.data(), retained);
    source->metadata.insert(source->metadata.begin() + static_cast<std::ptrdiff_t>(index),
                            std::move(required));
  }
  source->metadata.push_back({"zz_unknown_semantic_setting", true});
  ASSERT_FALSE(ModelOptTargetBound(*source));
  ASSERT_FALSE(
      NormalizeModelOptTarget(std::move(*source), {bound->metadata_bytes, bound->work_units}));
  ASSERT_EQ(source->tensors.data(), retained);
  source->metadata.pop_back();
  const auto original_name = source->tensors.front().name;
  source->tensors.front().name = "unclassified.target.weight";
  ASSERT_FALSE(
      NormalizeModelOptTarget(std::move(*source), {bound->metadata_bytes, bound->work_units}));
  ASSERT_EQ(source->tensors.data(), retained);
  source->tensors.front().name = original_name;
  auto plan =
      NormalizeModelOptTarget(std::move(*source), {bound->metadata_bytes, bound->work_units});
  ASSERT_TRUE(plan) << plan.error().ToString();
  ASSERT_EQ(plan->parts().source().tensors.size(), 301730U);
  ASSERT_EQ(plan->parts().members().size(), 222892U);
  std::array<std::uint64_t, 3> classified{};
  for (const auto& d : plan->parts().dispositions()) ++classified[static_cast<std::size_t>(d.kind)];
  EXPECT_EQ(classified[0], 222892U);
  EXPECT_EQ(classified[1], 3U);
  EXPECT_EQ(classified[2], 78835U);
  std::uint64_t input_bytes = 0, output_bytes = 0, bf16_bytes = 0;
  for (const auto& member : plan->parts().members()) {
    const auto& original = plan->parts().source().tensors[member.source];
    EXPECT_EQ(original.extent, member.extent);
    EXPECT_EQ(member.route.input_bytes(), member.route.output_bytes());
    EXPECT_EQ(member.route.input_bytes(), original.extent.bytes);
    input_bytes += original.extent.bytes;
  }
  for (const auto& output : plan->parts().outputs()) {
    auto parts = format::PartsOf(output.format);
    ASSERT_TRUE(parts);
    for (const auto& part : parts->parts()) output_bytes += part.bytes;
    if (output.format.encoding == format::Encoding::kBF16)
      for (const auto& part : parts->parts()) bf16_bytes += part.bytes;
  }
  EXPECT_EQ(input_bytes, output_bytes);
  ASSERT_EQ(plan->bindings().size(), plan->canonical().tensors.size());
  for (std::size_t i = 0; i < plan->bindings().size(); ++i) {
    const auto& binding = plan->bindings()[i];
    ASSERT_LT(binding.output, plan->parts().outputs().size());
    EXPECT_EQ(binding.part, plan->canonical().tensors[i].part);
    EXPECT_EQ(plan->parts().outputs()[binding.output].format, plan->canonical().tensors[i].format);
  }
  ASSERT_EQ(plan->planned_folds().size(), 1U);
  const auto& fold = plan->planned_folds().front();
  EXPECT_EQ(fold.fold, "gdn.value_heads_tiled");
  ASSERT_EQ(fold.outputs.size(), 288U);
  EXPECT_TRUE(std::ranges::is_sorted(fold.outputs));
  EXPECT_EQ(std::ranges::adjacent_find(fold.outputs), fold.outputs.end());
  for (const auto output : fold.outputs) ASSERT_LT(output, plan->parts().outputs().size());
  EXPECT_EQ(std::ranges::count_if(plan->bindings(),
                                  [&](const auto& binding) {
                                    return std::ranges::binary_search(fold.outputs, binding.output);
                                  }),
            468);
  ASSERT_EQ(plan->semantic().size(), 3U);
  std::uint64_t semantic_bytes = 0;
  for (const auto& r : plan->semantic()) semantic_bytes += r.extent.bytes;
  EXPECT_EQ(semantic_bytes, 280U);
  ASSERT_EQ(plan->spec().stages.size(), 1U);
  EXPECT_EQ(plan->spec().stages[0].layers.size(), 48U);
  EXPECT_EQ(plan->spec().stages[0].folds, std::vector<std::string>{"gdn.value_heads_tiled"});
  EXPECT_TRUE(plan->spec().stages[0].reads.empty());
  EXPECT_TRUE(plan->spec().stages[0].heads.mtp.empty());
  auto valid = components::ValidateSpec(plan->spec(), plan->canonical().tensors);
  ASSERT_TRUE(valid) << valid.error();
  RecordProperty("normalization_metadata_bound", std::to_string(bound->metadata_bytes));
  RecordProperty("combined_memory_max", std::to_string(kCombined));
  RecordProperty("mem_available_before", std::to_string(*memory.available));
  RecordProperty("canonical_parts", std::to_string(plan->canonical().tensors.size()));
  RecordProperty("logical_outputs", std::to_string(plan->parts().outputs().size()));
  RecordProperty("preserved_weight_bytes", std::to_string(output_bytes));
  RecordProperty("bf16_weight_bytes", std::to_string(bf16_bytes));
  RecordProperty("spec_sha256", base::ToHex(plan->spec().Digest()));
}
}  // namespace
}  // namespace llmp::import
