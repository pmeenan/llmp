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
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include "artifact/artifact.h"
#include "base/json.h"
#include "base/sha256.h"
#include "cli/cli.h"
#include "components/step.h"
#include "import/normalize.h"
#include "import/reader.h"
#include "import/transforms.h"
#include "import/writer.h"
#include "platform/memory_pressure.h"
#include "qwen38_modelopt_pins.h"
#include "spec/stored.h"

namespace llmp::import {
namespace {
constexpr std::uint64_t kGiB = std::uint64_t{1} << 30U;
constexpr std::uint64_t kCombined = 8 * kGiB, kMargin = 6 * kGiB;
constexpr std::uint64_t kNormalizationBudget = 3 * kGiB, kDefaultMetadataBudget = 4 * kGiB;
bool KnownCliWarning(std::string_view line) {
  constexpr std::string_view prefix = "llmp import: warning: ";
  if (!line.starts_with(prefix)) return false;
  line.remove_prefix(prefix.size());
  for (const std::string_view reason :
       {": component excluded by this text-only architecture mapping",
        ": unread metadata retained as source provenance",
        ": trailing unimported layer labels retained as source provenance"}) {
    if (!line.ends_with(reason)) continue;
    const auto key = base::json::Parse(
        line.substr(0, line.size() - reason.size()),
        {.max_bytes = 8192, .max_depth = 1, .max_values = 1, .max_string_bytes = 8192});
    return key && key->root().is_string() && !key->root().string().empty();
  }
  return false;
}
struct CliDiagnostics {
  bool complete = false;
  std::size_t warnings = 0;
  std::string terminal;
};
CliDiagnostics ReadCliDiagnostics(std::FILE* file) {
  CliDiagnostics result;
  if (std::fseek(file, 0, SEEK_SET) != 0) return result;
  // Stream all records in fixed storage; never mistake a truncated prefix of
  // provenance warnings for the final refusal. Reject oversized/partial lines.
  std::array<char, 8192> buffer{};
  while (std::fgets(buffer.data(), static_cast<int>(buffer.size()), file)) {
    std::string_view line(buffer.data());
    if (!line.ends_with('\n') || !result.terminal.empty()) return result;
    line.remove_suffix(1);
    if (line.empty()) return result;
    if (KnownCliWarning(line))
      ++result.warnings;
    else
      result.terminal = line;
  }
  result.complete = std::feof(file) != 0 && std::ferror(file) == 0;
  return result;
}
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
  if (configured == nullptr)
    GTEST_SKIP() << "opt-in confined proof: needs a prepaid 8 GiB memory.max cgroup "
                    "(tools/job-proof) and LLMP_TEST_MODELOPT_SOURCE, LLMP_TEST_MODELOPT_CLI_STORE "
                    "(docs/experiments/m35-f12-frozen/README.md, \"Opt-in tests\")";
  ASSERT_EQ(OwnMemoryMax(), kCombined) << "run in prepaid 8 GiB combined memory.max cgroup";
  const auto memory = platform::ReadMemoryPressure();
  ASSERT_TRUE(memory.available);
  ASSERT_GE(*memory.available, kCombined + kMargin);
  // Exercise the real dispatcher without a redundant whole-payload write.
  // A tiny budget must refuse normalization; a larger one reaches the owned
  // plan but refuses the independently prepaid writer before publication.
  // NOLINTNEXTLINE(concurrency-mt-unsafe): explicit read-only test configuration.
  const char* cli_store = std::getenv("LLMP_TEST_MODELOPT_CLI_STORE");
  ASSERT_NE(cli_store, nullptr);
  ASSERT_FALSE(std::filesystem::exists(cli_store));
  for (const auto& [mib, message] : std::array<std::pair<std::string_view, std::string_view>, 2>{
           {{"1", "normalization exceeds the requested metadata budget"},
            {"3072", "artifact metadata has not been funded"}}}) {
    const std::array<std::string_view, 6> args{"import",  configured,       "--store",
                                               cli_store, "--metadata-mib", mib};
    std::FILE* output = std::tmpfile();
    std::FILE* error = std::tmpfile();
    ASSERT_NE(output, nullptr);
    ASSERT_NE(error, nullptr);
    EXPECT_EQ(cli::Run(args, output, error), cli::kExitFailure);
    EXPECT_EQ(std::ftell(output), 0);
    const auto diagnostics = ReadCliDiagnostics(error);
    EXPECT_TRUE(diagnostics.complete) << "metadata MiB=" << mib;
    EXPECT_GT(diagnostics.warnings, 0U);
    EXPECT_EQ(diagnostics.terminal, "llmp import: bounds: " + std::string(message))
        << "metadata MiB=" << mib;
    if (mib == "3072" && diagnostics.complete)
      RecordProperty("cli_writer_warning_records", std::to_string(diagnostics.warnings));
    std::fclose(output);
    std::fclose(error);
    // The import took the store's lock before planning: the store holds its
    // staging directory with the lock alone, nothing published.
    std::error_code ignored;
    for (const auto& entry : std::filesystem::directory_iterator(cli_store, ignored))
      EXPECT_EQ(entry.path().filename(), ".staging");
    for (const auto& entry : std::filesystem::directory_iterator(
             std::filesystem::path(cli_store) / ".staging", ignored))
      EXPECT_EQ(entry.path().filename(), "native-import.lock");
    std::filesystem::remove_all(cli_store, ignored);
  }
  ASSERT_FALSE(HasFailure());
  RecordProperty("cli_normalization_unfunded_refused", true);
  RecordProperty("cli_writer_unfunded_refused", true);
  auto source = ReadCheckpoint(configured);
  ASSERT_TRUE(source) << source.error().ToString();
  ASSERT_EQ(source->tensors.size(), 301730U);
  ASSERT_EQ(source->files.size(), 42U);
  auto bound = NormalizationBound(*source);
  ASSERT_TRUE(bound) << bound.error().ToString();
  // The diagnostic's 8 GiB hard domain includes reader, retained Checkpoint,
  // planner and file cache. Generic normalization's owned canonical/synthesis
  // ledger replaces the old indexed-plan sub-GiB inventory. The 3 GiB CLI
  // refusal budget must fund normalization; the default 4 GiB must also fund
  // the independent writer, checked through its actual planner below.
  // Neither admission bound is measured occupancy.
  ASSERT_LT(bound->metadata_bytes, kNormalizationBudget);
  const auto* retained = source->tensors.data();
  ASSERT_FALSE(Normalize(std::move(*source), {bound->metadata_bytes - 1, bound->work_units}));
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
    ASSERT_FALSE(NormalizationBound(*source));
    ASSERT_FALSE(Normalize(std::move(*source), {bound->metadata_bytes, bound->work_units}));
    ASSERT_EQ(source->tensors.data(), retained);
    entry->value = std::move(original);
  }
  for (const std::string_view key : {"text_config.num_hidden_layers", "text_config.hidden_size"}) {
    auto entry = std::ranges::find(source->metadata, key, &MetaEntry::key);
    ASSERT_NE(entry, source->metadata.end());
    const auto index = static_cast<std::size_t>(entry - source->metadata.begin());
    auto required = std::move(*entry);
    source->metadata.erase(entry);
    ASSERT_FALSE(NormalizationBound(*source));
    ASSERT_FALSE(Normalize(std::move(*source), {bound->metadata_bytes, bound->work_units}));
    ASSERT_EQ(source->tensors.data(), retained);
    source->metadata.insert(source->metadata.begin() + static_cast<std::ptrdiff_t>(index),
                            std::move(required));
  }
  source->metadata.push_back({"zz_unknown_semantic_setting", true});
  ASSERT_FALSE(NormalizationBound(*source));
  ASSERT_FALSE(Normalize(std::move(*source), {bound->metadata_bytes, bound->work_units}));
  ASSERT_EQ(source->tensors.data(), retained);
  source->metadata.pop_back();
  const auto original_name = source->tensors.front().name;
  source->tensors.front().name = "unclassified.target.weight";
  ASSERT_FALSE(Normalize(std::move(*source), {bound->metadata_bytes, bound->work_units}));
  ASSERT_EQ(source->tensors.data(), retained);
  source->tensors.front().name = original_name;
  auto plan = Normalize(std::move(*source), {bound->metadata_bytes, bound->work_units});
  ASSERT_TRUE(plan) << plan.error().ToString();
  const auto placement =
      publication_detail::InspectPlacement(*plan, kDefaultMetadataBudget - bound->metadata_bytes);
  ASSERT_TRUE(placement) << placement.error().ToString();
  ASSERT_GT(placement->metadata_bytes, kNormalizationBudget - bound->metadata_bytes);
  ASSERT_LE(bound->metadata_bytes + placement->metadata_bytes, kDefaultMetadataBudget);
  const spec::SamplingDefaults expected_sampling{1.0, 0.95, 20, std::nullopt};
  EXPECT_EQ(plan->spec().stages.front().sampling, expected_sampling);
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
    // Copies and relayouts keep their bytes; the BF16 folds write F32.
    const bool folded =
        original.format.encoding == format::Encoding::kBF16 &&
        plan->parts().outputs()[member.output].format.encoding == format::Encoding::kF32;
    EXPECT_EQ(member.route.output_bytes(), member.route.input_bytes() * (folded ? 2 : 1));
    EXPECT_EQ(member.route.input_bytes(), original.extent.bytes);
    input_bytes += member.route.output_bytes();
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
  // The head reorder, then the parameter folds (by role): the zero-centered
  // norms' 1 + w and A's -exp(A_log), in F32.
  ASSERT_EQ(plan->planned_folds().size(), 3U);
  EXPECT_EQ(plan->planned_folds()[1].fold, "norm.plus_one");
  EXPECT_EQ(plan->planned_folds()[1].outputs.size(), 148U);
  EXPECT_EQ(plan->planned_folds()[2].fold, "ssm.a_neg_exp");
  EXPECT_EQ(plan->planned_folds()[2].outputs.size(), 36U);
  const auto& fold = plan->planned_folds().front();
  EXPECT_EQ(fold.fold, "gdn.value_heads_tiled");
  ASSERT_EQ(fold.outputs.size(), 288U);
  // Generic planning preserves component traversal order. Membership and
  // uniqueness are the contract; do not impose numeric ordering on the plan.
  auto fold_outputs = fold.outputs;
  std::ranges::sort(fold_outputs);
  EXPECT_EQ(std::ranges::adjacent_find(fold_outputs), fold_outputs.end());
  for (const auto output : fold.outputs) ASSERT_LT(output, plan->parts().outputs().size());
  EXPECT_EQ(std::ranges::count_if(plan->bindings(),
                                  [&](const auto& binding) {
                                    return std::ranges::binary_search(fold_outputs, binding.output);
                                  }),
            468);
  ASSERT_EQ(plan->semantic().size(), 3U);
  std::uint64_t semantic_bytes = 0;
  for (const auto& r : plan->semantic()) semantic_bytes += r.extent.bytes;
  EXPECT_EQ(semantic_bytes, 280U);
  ASSERT_EQ(plan->spec().stages.size(), 1U);
  EXPECT_EQ(plan->spec().stages[0].layers.size(), 48U);
  EXPECT_EQ(plan->spec().stages[0].folds,
            (std::vector<std::string>{"gdn.value_heads_tiled", "norm.plus_one", "ssm.a_neg_exp"}));
  // The routed experts' 96 tables and the n-gram table, relaid out.
  EXPECT_EQ(std::ranges::count_if(plan->parts().outputs(),
                                  [](const auto& o) { return o.relayout_from.has_value(); }),
            97);
  EXPECT_TRUE(plan->spec().stages[0].reads.empty());
  EXPECT_TRUE(plan->spec().stages[0].heads.mtp.empty());
  auto valid = components::ValidateSpec(plan->spec(), plan->canonical().tensors);
  ASSERT_TRUE(valid) << valid.error();
  RecordProperty("normalization_metadata_bound", std::to_string(bound->metadata_bytes));
  RecordProperty("writer_metadata_bound", std::to_string(placement->metadata_bytes));
  RecordProperty("default_metadata_budget", std::to_string(kDefaultMetadataBudget));
  RecordProperty("combined_memory_max", std::to_string(kCombined));
  RecordProperty("mem_available_before", std::to_string(*memory.available));
  RecordProperty("canonical_parts", std::to_string(plan->canonical().tensors.size()));
  RecordProperty("logical_outputs", std::to_string(plan->parts().outputs().size()));
  RecordProperty("preserved_weight_bytes", std::to_string(output_bytes));
  RecordProperty("bf16_weight_bytes", std::to_string(bf16_bytes));
  RecordProperty("spec_sha256", base::ToHex(plan->spec().Digest()));
}
// The checkpoint's pinned files: the fixture's table (qwen38_modelopt_pins.h)
// and the tokenizer and generation files this test also authenticates.
struct PublicationPin {
  std::string_view name, sha256;
  std::uint64_t bytes;
};
constexpr std::array<PublicationPin, 4> kExtraPins{{
    {"tokenizer.json", "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3",
     12809320ULL},
    {"tokenizer_config.json", "b11349aafa7cdc6a320767cf7ceb29ed82f7eda5d65e8e0819e76f0ce947bf27",
     17928ULL},
    {"generation_config.json", "735ec63e625c2086fbec18327233a03a7b060fddabfd7915af4e6fdcec3187a2",
     2097ULL},
    {"chat_template.jinja", "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041",
     8952ULL},
}};
constexpr std::size_t kPublicationFiles =
    test_support::qwen38::kModelOptPins.size() + kExtraPins.size();
std::optional<PublicationPin> PublicationPinFor(std::string_view name) {
  for (const auto& [pinned, bytes, sha256] : test_support::qwen38::kModelOptPins) {
    if (pinned == name) return PublicationPin{pinned, sha256, bytes};
  }
  for (const PublicationPin& pin : kExtraPins) {
    if (pin.name == name) return pin;
  }
  return std::nullopt;
}

// Independent destination-order oracle: no route/cursor/transform API is used.
// The closed source has16 key heads and48 value heads; destination j*16+g
// receives source g*3+j. Every byte of every recorded permuted part is read.
void ComparePermutedPart(const NormalizedPlan& plan, const IndexedMember& member,
                         const artifact::Artifact& stored, const std::filesystem::path& root,
                         std::string_view canonical_name, std::uint64_t& compared) {
  const auto& output = plan.parts().outputs()[member.output];
  const auto& source = plan.parts().source().files[member.extent.file];
  const auto resource = stored.FindResource(canonical_name);
  ASSERT_TRUE(resource);
  const auto placement = stored.ResourcePlacement(*resource);
  ASSERT_TRUE(placement);
  // A, folded to -exp(A_log) in F32 under the reorder (ssm.a_neg_exp).
  const bool folded = output.format.encoding == format::Encoding::kF32;
  ASSERT_EQ(placement->bytes.value(), member.extent.bytes * (folded ? 2 : 1));
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
  ASSERT_LE(member.extent.bytes * (folded ? 2 : 1), actual->bytes - base);
  ASSERT_LE(member.extent.offset, input->bytes);
  ASSERT_LE(member.extent.bytes, input->bytes - member.extent.offset);
  const auto& format = output.format;
  ASSERT_LE(format.shape.rank, 2U);
  ASSERT_TRUE(
      (format.encoding == format::Encoding::kBF16 && format.layout == format::Layout::kDense) ||
      (folded && output.name.ends_with(".ssm_a") && format.layout == format::Layout::kDense) ||
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
      const auto out_row = row_bytes * (folded ? 2 : 1), out_head = head_bytes * (folded ? 2 : 1);
      ASSERT_TRUE(ReadExactly(actual->fd.get(), base + row * out_row,
                              std::span(got).first(static_cast<std::size_t>(out_row)), 0));
      std::array<std::byte, 64> value{};
      for (std::uint64_t destination = 0; destination < 48; ++destination) {
        const auto original = (destination % 16) * 3 + destination / 16;
        auto want = std::span<const std::byte>(expected).subspan(
            static_cast<std::size_t>(original * head_bytes), static_cast<std::size_t>(head_bytes));
        if (folded) {
          ASSERT_LE(out_head, value.size());
          ASSERT_TRUE(FoldBf16(ScalarFold::kANegExp, want,
                               std::span(value).first(static_cast<std::size_t>(out_head))));
          want = std::span<const std::byte>(value).first(static_cast<std::size_t>(out_head));
        }
        ASSERT_EQ(std::memcmp(got.data() + destination * out_head, want.data(),
                              static_cast<std::size_t>(out_head)),
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
    GTEST_SKIP() << "opt-in confined proof: needs a prepaid 8 GiB memory.max cgroup "
                    "(tools/job-proof) and LLMP_TEST_MODELOPT_SOURCE, "
                    "LLMP_TEST_MODELOPT_PUBLICATION_STORE (docs/experiments/m35-f12-frozen/"
                    "README.md, \"Opt-in tests\")";
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
  ASSERT_EQ(source->files.size(), kPublicationFiles);
  ASSERT_EQ(source->tensors.size(), 301730U);
  std::uint64_t authenticated_source_bytes = 0;
  for (std::size_t i = 0; i < source->files.size(); ++i) {
    const auto& file = source->files[i];
    const auto pin = PublicationPinFor(file.name);
    ASSERT_TRUE(pin);
    ASSERT_EQ(file.bytes, pin->bytes);
    authenticated_source_bytes += file.bytes;
  }
  ASSERT_EQ(authenticated_source_bytes, 105885660088ULL);
  auto bound = NormalizationBound(*source);
  ASSERT_TRUE(bound) << bound.error().ToString();
  ASSERT_LT(bound->metadata_bytes, kNormalizationBudget);
  auto plan = Normalize(std::move(*source), {bound->metadata_bytes, bound->work_units});
  ASSERT_TRUE(plan) << plan.error().ToString();
  const spec::SamplingDefaults expected_sampling{1.0, 0.95, 20, std::nullopt};
  EXPECT_EQ(plan->spec().stages.front().sampling, expected_sampling);
  ASSERT_EQ(plan->canonical().tensors.size(), 1789U);
  ASSERT_EQ(plan->parts().outputs().size(), 1308U);
  ASSERT_EQ(plan->parts().members().size(), 222892U);
  ASSERT_EQ(plan->planned_folds().size(), 3U);
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
  // The source's weight bytes, but its folded BF16 norms and A in F32.
  ASSERT_EQ(bytes, 103797070596ULL + 2069888ULL);
  ASSERT_EQ(expanded, 99901U);
  // Same conservative writer inventory, evaluated before any store mutation.
  // Every multiplier is bounded by the authenticated fixed domain above.
  const auto chunks =
      std::max(plan->cost().payload_bytes, bytes) / artifact::kChunkBytes + 2 * expanded;
  const auto writer_bound =
      (256ULL << 20U) + expanded * 8192 + chunks * 512 +
      plan->parts().source().files.size() * 1024 + plan->parts().members().size() * 512 +
      plan->parts().source().tensors.size() * 32 + 3 * 256 + (288 + 148 + 36) * 8192 + 65536;
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
  std::array<char, 512> text{};
  const auto count = std::fread(text.data(), 1, text.size(), output);
  const auto diagnostics = ReadCliDiagnostics(error);
  std::fclose(output);
  std::fclose(error);
  ASSERT_EQ(code, cli::kExitOk) << diagnostics.terminal;
  ASSERT_TRUE(diagnostics.complete);
  ASSERT_GT(diagnostics.warnings, 0U);
  ASSERT_TRUE(diagnostics.terminal.empty()) << diagnostics.terminal;
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
            "02d130bce7462794251cc29e5554565a0ad83a398ce2776fd8078ef4b683618a");
  EXPECT_EQ(spec->stages.front().sampling, expected_sampling);
  ASSERT_EQ(stored->sources().size(), kPublicationFiles);
  for (const auto& receipt : stored->sources()) {
    const auto pin = PublicationPinFor(receipt.name);
    ASSERT_TRUE(pin);
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
    if (!std::ranges::contains(plan->planned_folds()[0].outputs, binding.output)) continue;
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
  ASSERT_EQ(expert_arrays, 192U);
  ASSERT_EQ(permuted_parts, 468U);
  std::ranges::sort(folded);
  ASSERT_EQ(stored->folds().size(), 3U);
  EXPECT_EQ(stored->folds()[0].name, "gdn.value_heads_tiled");
  EXPECT_EQ(stored->folds()[0].tensors, folded);
  EXPECT_EQ(stored->folds()[1].tensors.size(), 148U);
  EXPECT_EQ(stored->folds()[2].tensors.size(), 36U);
  ASSERT_EQ(stored->relayouts().size(), 1U);
  EXPECT_EQ(stored->relayouts()[0].from, format::Layout::kSplit);
  EXPECT_EQ(stored->relayouts()[0].tensors.size(), 194U);
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
  RecordProperty("source_files_authenticated", std::to_string(kPublicationFiles));
  RecordProperty("source_bytes_authenticated", std::to_string(authenticated_source_bytes));
  RecordProperty("canonical_parts", "1789");
  RecordProperty("logical_outputs", "1308");
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
