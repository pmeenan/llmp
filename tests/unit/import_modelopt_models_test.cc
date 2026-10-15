// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
// Native normalization/publication controls. The runner must create a confined
// process before launch: an 8 GiB combined memory.max pays process AND file
// cache, with no assumption about buffered-reader automatic readahead.
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include "artifact/artifact.h"
#include "base/sha256.h"
#include "cli/cli.h"
#include "components/step.h"
#include "import/modelopt_qwen38.h"
#include "import/reader.h"
#include "import/writer.h"
#include "platform/memory_pressure.h"
#include "spec/stored.h"

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
  // Exercise the real dispatcher without a redundant whole-payload write.
  // A tiny budget must refuse normalization; a larger one reaches the owned
  // plan but refuses the independently prepaid writer before store creation.
  // NOLINTNEXTLINE(concurrency-mt-unsafe): explicit read-only test configuration.
  const char* cli_store = std::getenv("LLMP_TEST_MODELOPT_CLI_STORE");
  ASSERT_NE(cli_store, nullptr);
  ASSERT_FALSE(std::filesystem::exists(cli_store));
  for (const auto& [mib, message] : std::array<std::pair<std::string_view, std::string_view>, 2>{
           {{"1", "normalization exceeds the requested metadata budget"},
            {"512", "artifact metadata has not been funded"}}}) {
    const std::array<std::string_view, 6> args{"import",  configured,       "--store",
                                               cli_store, "--metadata-mib", mib};
    std::FILE* output = std::tmpfile();
    std::FILE* error = std::tmpfile();
    ASSERT_NE(output, nullptr);
    ASSERT_NE(error, nullptr);
    EXPECT_EQ(cli::Run(args, output, error), cli::kExitFailure);
    EXPECT_EQ(std::ftell(output), 0);
    std::rewind(error);
    std::array<char, 1024> text{};
    const auto count = std::fread(text.data(), 1, text.size(), error);
    EXPECT_NE(std::string_view(text.data(), count).find(message), std::string_view::npos);
    std::fclose(output);
    std::fclose(error);
    EXPECT_FALSE(std::filesystem::exists(cli_store));
  }
  RecordProperty("cli_normalization_unfunded_refused", true);
  RecordProperty("cli_writer_unfunded_refused", true);
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
struct PublicationPin {
  std::string_view name, sha256;
  std::uint64_t bytes;
};
constexpr std::array<PublicationPin, 38> kPublicationPins{{
    {"config.json", "2d655dfa625c3c018f9685031367bc83f8532d3faf3333d7c838b2bdc13b8a7e", 109897ULL},
    {"hf_quant_config.json", "d60c54caa65317fd8f1338bdeb81b9014512ac8f21ec18da22541cb88c6784b7",
     99088ULL},
    {"model-00001-of-00034.safetensors",
     "3f870511a20ad170c997517765f03d475bcd364ae1a6011491085cab620d3c39", 2841505880ULL},
    {"model-00002-of-00034.safetensors",
     "fe8f4bdbce28f4911f10460b75581b6c88a8ade9b836d2f1ec22c6bee6f976a8", 1575009800ULL},
    {"model-00003-of-00034.safetensors",
     "2b7d1312a660cf17bb05c34e1ac79570820f86fb1e1115108ee40562e48acb5e", 1575009808ULL},
    {"model-00004-of-00034.safetensors",
     "25ba4508c209a5cc54c04edb1edb8d0a0b5efa80e5f99637e2263fa78d6a25e8", 1575009808ULL},
    {"model-00005-of-00034.safetensors",
     "a9848067bb49f44b8339bcdca9d60ca74d801a6df874c979c6f901c8da6ff545", 1575009808ULL},
    {"model-00006-of-00034.safetensors",
     "66fc496b58675a90b94d9f36765fe67692dd8a4c966ec0692e076d751f835c8d", 1575009808ULL},
    {"model-00007-of-00034.safetensors",
     "68f89e84be13069bd9f2928b44a9f9638d628fd62be5abb6a4fc956c1af25ea9", 1575009808ULL},
    {"model-00008-of-00034.safetensors",
     "b8bdaa0a7b027dc0fa38c2e43bb31e8ddf9ba9288d3d0f28c422e6947241d96e", 1575009808ULL},
    {"model-00009-of-00034.safetensors",
     "6c5a85990b16a93f0e7e28fb76f56713d5cbc05bef494bb15360188f575746e9", 1575009808ULL},
    {"model-00010-of-00034.safetensors",
     "8734a1fc3b5857391fbf2252d4305391a9f77894b70577949f472fc75386582f", 1575009808ULL},
    {"model-00011-of-00034.safetensors",
     "29792378d43c9f70d9f3fbdd07d0ee24b38f81fe27aff25a02ba0d3cde8df13c", 1575009808ULL},
    {"model-00012-of-00034.safetensors",
     "b23e615ebefff4753d982d9ec6a8be6ad57353a8900146a3ec4e217f81c10e61", 1575009808ULL},
    {"model-00013-of-00034.safetensors",
     "9da7b4570fc3f13cb781e5aca5e79d0edfc10be5555a18bc7bda8f47459d5acf", 1575009808ULL},
    {"model-00014-of-00034.safetensors",
     "53e770d0bdf6261e30abaae2477aa0b50f6686511e0240514ded0fd65edbfeba", 1575009808ULL},
    {"model-00015-of-00034.safetensors",
     "03440de56d127ff58e72963aafd0de0574b825667330c50b4441bb300dae9f06", 1575009816ULL},
    {"model-00016-of-00034.safetensors",
     "ce2e0474ccb34c56003a61608b46d3a8f668c218a9993c1f861239df83afe77b", 1575009824ULL},
    {"model-00017-of-00034.safetensors",
     "57f84845e0b8bcab738e70f0f880e941f1baa8da14663293d7d40481288c73a0", 1575009824ULL},
    {"model-00018-of-00034.safetensors",
     "3477df0be60910576fc5a1f8159d8dbbf7847a03872b9cb53664808e1d5dfac8", 1575009824ULL},
    {"model-00019-of-00034.safetensors",
     "43c22213b2fe0990c7fc2aa54c0ea65062eb93253b4ea419914d1a655e64b23a", 2438722732ULL},
    {"model-00020-of-00034.safetensors",
     "b7143694bf8698c5b704ae9b3d492a0e9b8ef57bdda8ae6f3f29f264c183ad17", 5133924592ULL},
    {"model-00021-of-00034.safetensors",
     "522bbbc894183865726e06ca44021d275d03cc0e8dcd60a2c7074d870e811d37", 5194891408ULL},
    {"model-00022-of-00034.safetensors",
     "5a4789f6da6b386b09cb9093b0315fe12a816142958e4d8a4ae83e4fbaa066e7", 5133933936ULL},
    {"model-00023-of-00034.safetensors",
     "cd0aaef1c8688cb4e35b488048ede2a6b653e1d558b20d4657567df24dc8be13", 5194908864ULL},
    {"model-00024-of-00034.safetensors",
     "d07198edd38ae8740d556487729ae87be1e21e31a88cf6624c1b3f9d0fc6de76", 5188755712ULL},
    {"model-00025-of-00034.safetensors",
     "c7d5e642968679ca14d07bdb1e8c2187c59898611d05a6909c209738642c9519", 5133940040ULL},
    {"model-00026-of-00034.safetensors",
     "47a7c7a70188cb3dc318071a5278260c4b05d7ce1b69c3b99e7bf1c0e42c353b", 5194907416ULL},
    {"model-00027-of-00034.safetensors",
     "a81fedf160434806e81f36731c853fe4e3f973dea2f0817043987f32b4915a54", 5133940216ULL},
    {"model-00028-of-00034.safetensors",
     "91c5d479fe4de1035c430ad7fd3de9d15214a46fd786ca7909d1f2fa40ef603a", 5194907672ULL},
    {"model-00029-of-00034.safetensors",
     "f31c6f61968f6f9e85d469f31c55c82ab6c5f2bc54c1d7a0cd6b23ac45f0ce15", 5133939992ULL},
    {"model-00030-of-00034.safetensors",
     "df8627f5b8b0711b2ebf415c1f2549ef0b50f964e0babb560469b191b33c6a14", 5194908960ULL},
    {"model-00031-of-00034.safetensors",
     "654c8f9ae1ab5de393cbcdd46a13bd0d3d1c1005e575879e0e660d40ad9ef425", 5188755536ULL},
    {"model-00032-of-00034.safetensors",
     "7de2e680802ad92a0e019f875ebc07f2d5d82c8bca1c64b0f817d547bc3c9de7", 5133940208ULL},
    {"model-00033-of-00034.safetensors",
     "0fbcdf16c13cd830447efdcd3be19b82ec0e76b7f57f0e96952c1f2b62002b11", 5021547836ULL},
    {"model-00034-of-00034.safetensors",
     "fe0a0a227e39c4d02dc66c726d3e1ad87ef082e90a6b023ac90b2bb301672203", 1597270800ULL},
    {"model-inputscales.safetensors",
     "49c7d04606ff7c8f103b4d1fc9742b44649a2937b25ecc896a5450ab1020f308", 9669936ULL},
    {"model.safetensors.index.json",
     "435eef76fc10fc6e932a208a1f85a86bc9c7ffc389b27b34ba83bb6a5d0371e9", 33074286ULL},
}};

// Independent destination-order oracle: no route/cursor/transform API is used.
// The closed source has16 key heads and48 value heads; destination j*16+g
// receives source g*3+j. Every byte of every recorded permuted part is read.
void ComparePermutedPart(const ModelOptTargetPlan& plan, const IndexedMember& member,
                         const artifact::Artifact& stored, const std::filesystem::path& root,
                         std::string_view canonical_name, std::uint64_t& compared) {
  const auto& output = plan.parts().outputs()[member.output];
  const auto& source = plan.parts().source().files[member.extent.file];
  const auto resource = stored.FindResource(canonical_name);
  ASSERT_TRUE(resource);
  const auto placement = stored.ResourcePlacement(*resource);
  ASSERT_TRUE(placement);
  ASSERT_EQ(placement->bytes.value(), member.extent.bytes);
  ASSERT_EQ(member.destination, 0U);
  const auto& group = stored.groups()[placement->group];
  const auto& shard = stored.shards()[group.shard];
  auto input = OpenCheckpointFile(source.path, member.extent.file);
  ASSERT_TRUE(input) << input.error().ToString();
  ASSERT_EQ(input->bytes, source.bytes);
  ASSERT_EQ(input->device, source.device);
  ASSERT_EQ(input->inode, source.inode);
  auto actual = OpenCheckpointFile(root / shard.path, 0);
  ASSERT_TRUE(actual) << actual.error().ToString();
  const auto base = shard.data_offset.value() + group.offset.value() + placement->offset.value();
  ASSERT_LE(base, actual->bytes);
  ASSERT_LE(member.extent.bytes, actual->bytes - base);
  ASSERT_LE(member.extent.offset, input->bytes);
  ASSERT_LE(member.extent.bytes, input->bytes - member.extent.offset);
  const auto& format = output.format;
  ASSERT_LE(format.shape.rank, 2U);
  ASSERT_TRUE(
      (format.encoding == format::Encoding::kBF16 && format.layout == format::Layout::kDense) ||
      (format.encoding == format::Encoding::kMxfp8 && format.layout == format::Layout::kSplit));
  const auto rows = format.shape.ne[1];
  ASSERT_NE(rows, 0U);
  ASSERT_EQ(member.extent.bytes % rows, 0U);
  const auto row_bytes = member.extent.bytes / rows;
  ASSERT_GT(row_bytes, 0U);
  ASSERT_LE(row_bytes, 65536U);
  const bool scalar = output.name.ends_with(".ssm_a") || output.name.ends_with(".ssm_dt.bias");
  const bool columns = scalar || output.name.ends_with(".ssm_out.weight");
  const bool row_scalar =
      output.name.ends_with(".ssm_alpha.weight") || output.name.ends_with(".ssm_beta.weight");
  const bool qkv =
      output.name.ends_with(".attn_qkv.weight") || output.name.ends_with(".ssm_conv1d.weight");
  ASSERT_TRUE(columns || row_scalar || qkv || output.name.ends_with(".attn_gate.weight"));
  std::array<std::byte, 65536> expected{}, got{};
  if (columns) {
    ASSERT_EQ(row_bytes % 48, 0U);
    const auto head_bytes = row_bytes / 48;
    for (std::uint64_t row = 0; row < rows; ++row) {
      ASSERT_TRUE(ReadExactly(input->fd.get(), member.extent.offset + row * row_bytes,
                              std::span(expected).first(static_cast<std::size_t>(row_bytes)),
                              member.extent.file));
      ASSERT_TRUE(ReadExactly(actual->fd.get(), base + row * row_bytes,
                              std::span(got).first(static_cast<std::size_t>(row_bytes)), 0));
      for (std::uint64_t destination = 0; destination < 48; ++destination) {
        const auto original = (destination % 16) * 3 + destination / 16;
        ASSERT_EQ(std::memcmp(got.data() + destination * head_bytes,
                              expected.data() + original * head_bytes,
                              static_cast<std::size_t>(head_bytes)),
                  0)
            << canonical_name << " row " << row << " head " << destination;
      }
      compared += row_bytes;
    }
    return;
  }
  const std::uint64_t first = qkv ? 4096 : 0, head_rows = row_scalar ? 1 : 128;
  ASSERT_GE(rows, first + 48 * head_rows);
  for (std::uint64_t row = 0; row < rows;) {
    std::uint64_t original = row, available = rows - row;
    if (row < first) {
      available = first - row;
    } else if (row < first + 48 * head_rows) {
      const auto head = (row - first) / head_rows, within = (row - first) % head_rows;
      original = first + ((head % 16) * 3 + head / 16) * head_rows + within;
      available = head_rows - within;
    }
    const auto count = std::min(available, 65536 / row_bytes);
    const auto bytes = static_cast<std::size_t>(count * row_bytes);
    ASSERT_GT(bytes, 0U);
    ASSERT_TRUE(ReadExactly(input->fd.get(), member.extent.offset + original * row_bytes,
                            std::span(expected).first(bytes), member.extent.file));
    ASSERT_TRUE(
        ReadExactly(actual->fd.get(), base + row * row_bytes, std::span(got).first(bytes), 0));
    ASSERT_EQ(std::memcmp(expected.data(), got.data(), bytes), 0)
        << canonical_name << " row " << row;
    compared += bytes;
    row += count;
  }
}

TEST(ImportModelOptModelsTest, PublishesPinnedNativeTargetWithCompleteSourceAuthentication) {
  // No implicit opt-in: this large publication case needs its explicit confined recipe.
  // NOLINTNEXTLINE(concurrency-mt-unsafe): read-only test environment.
  const char* configured = std::getenv("LLMP_TEST_MODELOPT_SOURCE");
  // NOLINTNEXTLINE(concurrency-mt-unsafe): read-only test environment.
  const char* destination = std::getenv("LLMP_TEST_MODELOPT_PUBLICATION_STORE");
  if (configured == nullptr || destination == nullptr)
    GTEST_SKIP() << "the confined publication recipe needs LLMP_TEST_MODELOPT_SOURCE and "
                    "LLMP_TEST_MODELOPT_PUBLICATION_STORE";
  ASSERT_EQ(OwnMemoryMax(), kCombined);
  const auto memory = platform::ReadMemoryPressure();
  ASSERT_TRUE(memory.available);
  ASSERT_GE(*memory.available, kCombined + kMargin);
  const std::filesystem::path store(destination);
  ASSERT_TRUE(store.is_absolute());
  ASSERT_FALSE(std::filesystem::exists(store));
  ASSERT_TRUE(std::filesystem::is_directory(store.parent_path()));
  auto source = ReadCheckpoint(configured);
  ASSERT_TRUE(source) << source.error().ToString();
  ASSERT_EQ(source->files.size(), kPublicationPins.size());
  ASSERT_EQ(source->tensors.size(), 301730U);
  std::uint64_t authenticated_source_bytes = 0;
  for (std::size_t i = 0; i < source->files.size(); ++i) {
    const auto& file = source->files[i];
    const auto pin = std::ranges::find(kPublicationPins, file.name, &PublicationPin::name);
    ASSERT_NE(pin, kPublicationPins.end());
    ASSERT_EQ(file.bytes, pin->bytes);
    authenticated_source_bytes += file.bytes;
  }
  ASSERT_EQ(authenticated_source_bytes, 105872821791ULL);
  auto bound = ModelOptTargetBound(*source);
  ASSERT_TRUE(bound) << bound.error().ToString();
  ASSERT_LT(bound->metadata_bytes, kGiB);
  auto plan =
      NormalizeModelOptTarget(std::move(*source), {bound->metadata_bytes, bound->work_units});
  ASSERT_TRUE(plan) << plan.error().ToString();
  ASSERT_EQ(plan->canonical().tensors.size(), 1886U);
  ASSERT_EQ(plan->parts().outputs().size(), 1212U);
  ASSERT_EQ(plan->parts().members().size(), 222892U);
  ASSERT_EQ(plan->planned_folds().size(), 1U);
  ASSERT_EQ(plan->planned_folds()[0].outputs.size(), 288U);
  ASSERT_EQ(plan->semantic().size(), 3U);
  std::uint64_t expanded = 0, bytes = 0;
  for (const auto& tensor : plan->canonical().tensors) {
    ASSERT_LE(tensor.format.shape.ne[2], 512U);
    expanded += tensor.experts ? tensor.format.shape.ne[2] : 1;
    const auto parts = format::PartsOf(tensor.format);
    ASSERT_TRUE(parts);
    const auto part = parts->Find(tensor.part);
    ASSERT_TRUE(part);
    bytes += part->bytes;
  }
  ASSERT_EQ(bytes, 103797070596ULL);
  ASSERT_EQ(expanded, 222638U);
  // Same conservative writer inventory, evaluated before any store mutation.
  // Every multiplier is bounded by the authenticated fixed domain above.
  const auto chunks =
      std::max(plan->cost().payload_bytes, bytes) / artifact::kChunkBytes + 2 * expanded;
  const auto writer_bound =
      (256ULL << 20U) + expanded * 8192 + chunks * 512 +
      plan->parts().source().files.size() * 1024 + plan->parts().members().size() * 512 +
      plan->parts().source().tensors.size() * 32 + 3 * 256 + 288 * 8192 + 65536;
  ASSERT_LT(writer_bound, 3 * kGiB);
  ASSERT_FALSE(std::filesystem::exists(store));
  // Qualify the actual ordinary-source CLI path with its default 4 GiB
  // metadata budget. The independent oracle plan remains owned above; both
  // plans fit the same prepaid combined process-and-file-cache domain.
  ASSERT_LT(bound->metadata_bytes + writer_bound, 4 * kGiB);
  const auto source_path = std::filesystem::path(configured).string();
  const auto store_path = store.string();
  const std::array<std::string_view, 4> args{"import", source_path, "--store", store_path};
  std::FILE* output = std::tmpfile();
  std::FILE* error = std::tmpfile();
  ASSERT_NE(output, nullptr);
  ASSERT_NE(error, nullptr);
  const auto code = cli::Run(args, output, error);
  std::rewind(output);
  std::rewind(error);
  std::array<char, 512> text{}, diagnostic{};
  const auto count = std::fread(text.data(), 1, text.size(), output);
  const auto errors = std::fread(diagnostic.data(), 1, diagnostic.size(), error);
  std::fclose(output);
  std::fclose(error);
  ASSERT_EQ(code, cli::kExitOk) << std::string_view(diagnostic.data(), errors);
  ASSERT_EQ(errors, 0U);
  const std::string_view cli_receipt(text.data(), count);
  ASSERT_EQ(cli_receipt.size(), 75U);
  ASSERT_TRUE(cli_receipt.starts_with("published "));
  ASSERT_EQ(cli_receipt.back(), '\n');
  const std::string id(cli_receipt.substr(10, 64));
  ASSERT_TRUE(std::ranges::all_of(
      id, [](char c) { return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'); }));
  const auto written = std::expected<WriteResult, Error>(WriteResult{id, store / id, false});
  RecordProperty("cli_funded_publication", true);
  std::size_t directories = 0;
  for (const auto& entry : std::filesystem::directory_iterator(store)) {
    ASSERT_TRUE(entry.is_directory());
    ASSERT_TRUE(entry.path().filename() == written->id || entry.path().filename() == ".staging");
    ++directories;
  }
  ASSERT_EQ(directories, 2U);
  for (const auto& entry : std::filesystem::directory_iterator(store / ".staging"))
    ASSERT_EQ(entry.path().filename(), "native-import.lock");
  ASSERT_EQ(written->path.parent_path(), store);
  auto stored = artifact::Artifact::Open(written->path, {.expected_id = written->id});
  ASSERT_TRUE(stored);
  ASSERT_EQ(stored->format_version(), 1U);
  auto spec = spec::StoredSpec(*stored);
  ASSERT_TRUE(spec) << spec.error();
  EXPECT_EQ(spec->Digest(), plan->spec().Digest());
  EXPECT_EQ(base::ToHex(spec->Digest()),
            "b7fdd2300c7690a17437490ec8b41110cb19169962435c20be128434b8544108");
  ASSERT_EQ(stored->sources().size(), 38U);
  for (const auto& receipt : stored->sources()) {
    const auto pin = std::ranges::find(kPublicationPins, receipt.name, &PublicationPin::name);
    ASSERT_NE(pin, kPublicationPins.end());
    EXPECT_EQ(receipt.bytes.value(), pin->bytes);
    EXPECT_EQ(base::ToHex(receipt.sha256), pin->sha256);
  }
  std::vector<std::string> folded;
  folded.reserve(468);
  std::uint64_t permuted_bytes = 0, permuted_parts = 0, expert_arrays = 0;
  for (std::size_t i = 0; i < plan->canonical().tensors.size(); ++i) {
    const auto& tensor = plan->canonical().tensors[i];
    const auto& binding = plan->bindings()[i];
    const auto parts = format::PartsOf(tensor.format);
    ASSERT_TRUE(parts);
    const auto part = parts->Find(tensor.part);
    ASSERT_TRUE(part);
    if (tensor.experts) {
      const auto index = stored->FindExpertArray(tensor.name);
      ASSERT_TRUE(index) << tensor.name;
      const auto& array = stored->expert_arrays()[*index];
      auto format = tensor.format;
      format.shape.rank = 2;
      format.shape.ne[2] = 1;
      ASSERT_EQ(array.count, 512U);
      EXPECT_EQ(array.format, format);
      EXPECT_EQ(array.part, tensor.part);
      EXPECT_EQ(array.slice_bytes.value() * array.count, part->bytes);
      for (std::uint32_t expert = 0; expert < array.count; ++expert) {
        const auto slice = stored->SlicePlacement(*index, expert);
        ASSERT_TRUE(slice);
        EXPECT_EQ(slice->bytes, array.slice_bytes);
        EXPECT_EQ(stored->groups()[slice->group].expert, expert);
      }
      ++expert_arrays;
    } else {
      const auto index = stored->FindResource(tensor.name);
      ASSERT_TRUE(index) << tensor.name;
      const auto& resource = stored->resources()[*index];
      EXPECT_EQ(resource.format, tensor.format);
      EXPECT_EQ(resource.part, tensor.part);
      EXPECT_EQ(resource.bytes.value(), part->bytes);
    }
    if (!std::ranges::binary_search(plan->planned_folds()[0].outputs, binding.output)) continue;
    ASSERT_FALSE(tensor.experts);
    const auto member = std::ranges::find_if(plan->parts().members(), [&](const auto& m) {
      return m.output == binding.output && m.part == binding.part;
    });
    ASSERT_NE(member, plan->parts().members().end());
    ASSERT_EQ(std::ranges::count_if(plan->parts().members(),
                                    [&](const auto& m) {
                                      return m.output == binding.output && m.part == binding.part;
                                    }),
              1);
    ASSERT_NO_FATAL_FAILURE(
        ComparePermutedPart(*plan, *member, *stored, written->path, tensor.name, permuted_bytes));
    folded.push_back(tensor.name);
    ++permuted_parts;
  }
  ASSERT_EQ(expert_arrays, 432U);
  ASSERT_EQ(permuted_parts, 468U);
  std::ranges::sort(folded);
  ASSERT_EQ(stored->folds().size(), 1U);
  EXPECT_EQ(stored->folds()[0].name, "gdn.value_heads_tiled");
  EXPECT_EQ(stored->folds()[0].tensors, folded);
  // Successful CLI publication already performed full file and chunk Deep
  // readback plus StoredSpec/ValidateSpec before atomic publication. Do not
  // duplicate those multi-100GB passes here, or automatically republish for reuse.
  std::uint64_t published_bytes = 0;
  for (const auto& file : stored->files()) published_bytes += file.bytes.value();
  RecordProperty("artifact_id", written->id);
  RecordProperty("artifact_path", written->path.string());
  RecordProperty("normalization_metadata_bound", std::to_string(bound->metadata_bytes));
  RecordProperty("writer_metadata_bound", std::to_string(writer_bound));
  RecordProperty("combined_memory_max", std::to_string(kCombined));
  RecordProperty("mem_available_before", std::to_string(*memory.available));
  RecordProperty("source_files_authenticated", "38");
  RecordProperty("source_bytes_authenticated", std::to_string(authenticated_source_bytes));
  RecordProperty("canonical_parts", "1886");
  RecordProperty("logical_outputs", "1212");
  RecordProperty("preserved_weight_bytes", std::to_string(bytes));
  RecordProperty("published_file_bytes", std::to_string(published_bytes));
  RecordProperty("artifact_files", std::to_string(stored->files().size()));
  RecordProperty("artifact_chunks", std::to_string(stored->chunk_sha256().size()));
  RecordProperty("expert_arrays", std::to_string(expert_arrays));
  RecordProperty("permuted_parts_compared", std::to_string(permuted_parts));
  RecordProperty("permuted_bytes_compared", std::to_string(permuted_bytes));
  RecordProperty("spec_sha256", base::ToHex(spec->Digest()));
}

}  // namespace
}  // namespace llmp::import
