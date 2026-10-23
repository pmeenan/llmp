// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <array>
#include <charconv>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <fstream>
#include <span>
#include <string>
#include <system_error>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include "artifact/artifact.h"
#include "artifact/gguf_metadata.h"
#include "artifact_writer.h"
#include "base/check.h"
#include "base/json.h"
#include "components/step.h"
#include "format/encoding.h"
#include "format/layout.h"
#include "format/tensor_format.h"
#include "gemma4_fixture.h"
#include "import/checkpoint.h"
#include "import/normalize.h"
#include "import/reader.h"
#include "import/readers/gguf.h"
#include "import/writer.h"
#include "import_target_fixture.h"
#include "qwen38_fixture.h"
#include "runtime/component_admission.h"
#include "runtime/model_settings.h"
#include "spec/spec.h"
#include "spec/stored.h"
#include "tokenizer_fixtures.h"
#include "v1_frozen.h"

namespace llmp::import {
namespace {
namespace rt = llmp::runtime;
namespace fm = llmp::format;

artifact::Source Origin(std::string_view name, std::uint64_t bytes, std::string_view hex) {
  base::Sha256Digest digest{};
  for (std::size_t i = 0; i < digest.size(); ++i) {
    unsigned byte = 0;
    const auto read = std::from_chars(hex.data() + 2 * i, hex.data() + 2 * i + 2, byte, 16);
    EXPECT_EQ(read.ec, std::errc{});
    digest[i] = static_cast<std::uint8_t>(byte);
  }
  return artifact::Source{std::string(name), base::Bytes(bytes), digest};
}

std::filesystem::path ModelsRoot() { return std::filesystem::path(test_support::ModelsDir()); }

std::expected<artifact::Artifact, std::string> PublishedTarget(std::string_view name) {
  const auto id = test_support::V1IdNamed(name);
  if (!id) return std::unexpected("no published target ID for " + std::string(name));
  return rt::OpenInstalled(test_support::V1Store(), *id);
}
std::expected<NormalizationTarget, std::string> TargetOf(const artifact::Artifact& artifact) {
  auto admitted = rt::ReadComponentAdmission(artifact);
  if (!admitted) return std::unexpected(admitted.error());
  const auto* mapping =
      spec::FindMapping(admitted->source.architecture, admitted->source.container);
  if (!mapping) return std::unexpected("published target has no exact container mapping");
  return NormalizationTarget{mapping, std::move(admitted->model), std::move(admitted->source)};
}

std::filesystem::path TestDataDir() {
  const char* dir = std::getenv("LLMP_TEST_DATA");
  if (dir && *dir) return dir;
  return "tests/unit/data";
}

void Ordered(Checkpoint& s) { std::ranges::sort(s.metadata, {}, &MetaEntry::key); }

void Value(Checkpoint& s, std::string key, MetaValue value) {
  const auto old = std::ranges::find(s.metadata, key, &MetaEntry::key);
  if (old == s.metadata.end())
    s.metadata.push_back({std::move(key), std::move(value)});
  else
    old->value = std::move(value);
  Ordered(s);
}

void AddTensor(Checkpoint& s, std::string name, std::span<const std::uint64_t> dims,
               fm::Encoding encoding = fm::Encoding::kF32) {
  const auto shape = fm::ShapeOf(dims);
  base::Check(shape.has_value(), "test shape");
  const auto& info = fm::Info(encoding);
  fm::TensorFormat f{.encoding = encoding,
                     .layout = info.quantized ? fm::Layout::kGgmlBlocks : fm::Layout::kDense,
                     .shape = *shape};
  const auto parts = fm::PartsOf(f);
  base::Check(parts.has_value(), "test format");
  const auto offset = (s.files[0].bytes + 31) / 32 * 32;
  s.tensors.push_back({.name = std::move(name),
                       .format = f,
                       .extent = {.offset = offset, .bytes = (*parts)[0].bytes},
                       .dtype = info.name,
                       .dims = std::vector<std::uint64_t>(dims.begin(), dims.end())});
  s.files[0].bytes = offset + (*parts)[0].bytes;
}

// The kept GGUF metadata (meta/<stem>.kv.gguf) the importer writes for a
// GGUF drafter, its header's key/value section: the checkpoint's metadata
// (GGUF v3; integers I64, reals F32 as the checkpoint stores them, lists
// of I64, F32 or strings). A stored drafter's composition values are read from it.
std::string KeptGgufHeader(const Checkpoint& s) {
  std::string out;
  const auto raw = [&](const void* p, std::size_t n) {
    out.append(static_cast<const char*>(p), n);
  };
  const auto u32 = [&](std::uint32_t v) { raw(&v, 4); };
  const auto u64 = [&](std::uint64_t v) { raw(&v, 8); };
  const auto str = [&](std::string_view v) {
    u64(v.size());
    out.append(v);
  };
  constexpr std::uint32_t kF32 = 6, kBool = 7, kString = 8, kArray = 9, kI64 = 11;
  std::vector<const MetaEntry*> kept;
  for (const MetaEntry& e : s.metadata)
    if (std::holds_alternative<std::int64_t>(e.value) || std::holds_alternative<double>(e.value) ||
        std::holds_alternative<bool>(e.value) || std::holds_alternative<std::string>(e.value) ||
        std::holds_alternative<std::vector<std::int64_t>>(e.value) ||
        std::holds_alternative<std::vector<double>>(e.value) ||
        std::holds_alternative<std::vector<std::string>>(e.value))
      kept.push_back(&e);
  out.append("GGUF");
  u32(3);
  u64(0);  // tensors: a kept header holds none
  u64(kept.size());
  for (const MetaEntry* e : kept) {
    str(e->key);
    if (const auto* i = std::get_if<std::int64_t>(&e->value)) {
      u32(kI64);
      raw(i, 8);
    } else if (const auto* d = std::get_if<double>(&e->value)) {
      u32(kF32);
      const auto f = static_cast<float>(*d);
      raw(&f, 4);
    } else if (const auto* b = std::get_if<bool>(&e->value)) {
      u32(kBool);
      const auto byte = static_cast<std::uint8_t>(*b ? 1 : 0);
      raw(&byte, 1);
    } else if (const auto* t = std::get_if<std::string>(&e->value)) {
      u32(kString);
      str(*t);
    } else if (const auto* list = std::get_if<std::vector<std::int64_t>>(&e->value)) {
      u32(kArray);
      u32(kI64);
      u64(list->size());
      for (const std::int64_t v : *list) raw(&v, 8);
    } else if (const auto* reals = std::get_if<std::vector<double>>(&e->value)) {
      u32(kArray);
      u32(kF32);
      u64(reals->size());
      for (const double v : *reals) {
        const auto f = static_cast<float>(v);
        raw(&f, 4);
      }
    } else {
      const auto& texts = std::get<std::vector<std::string>>(e->value);
      u32(kArray);
      u32(kString);
      u64(texts.size());
      for (const auto& text : texts) str(text);
    }
  }
  return out;
}

std::optional<Checkpoint> SyntheticAssistantCheckpoint(std::uint32_t size) {
  const auto file_path =
      TestDataDir() / ("gemma4-assistant/assistant" + std::to_string(size) + ".json");
  std::error_code ec;
  if (!std::filesystem::exists(file_path, ec)) return std::nullopt;
  std::ifstream file(file_path);
  std::string text(std::istreambuf_iterator<char>{file}, {});
  auto doc = base::json::Parse(text);
  if (!doc.has_value()) return std::nullopt;
  Checkpoint s{.container = "gguf",
               .architecture = "gemma4-assistant",
               .files = {{.name = "mtp-gemma-4-" + std::to_string(size) +
                                  (size == 26 ? "B-A4B-it.gguf" : "B-it.gguf"),
                          .bytes = 0}}};
  const auto meta = test_support::gemma4::Get(doc->root(), "metadata");
  for (std::size_t i = 0; i < meta.size(); ++i) {
    const auto key = meta.key(i);
    if (key != "general.architecture" && !key.starts_with("gemma4-assistant.")) continue;
    const auto val = meta.member(i);
    if (val.is_bool()) {
      Value(s, std::string(key), val.boolean());
    } else if (val.is_number()) {
      if (val.is_integer()) {
        Value(s, std::string(key), *val.int64());
      } else {
        Value(s, std::string(key), *val.float64());
      }
    } else if (val.is_string()) {
      Value(s, std::string(key), std::string(val.string()));
    } else if (val.is_array()) {
      if (val.size() > 0 && val.at(0).is_bool()) {
        std::vector<std::int64_t> list;
        for (std::size_t j = 0; j < val.size(); ++j) list.push_back(val.at(j).boolean() ? 1 : 0);
        Value(s, std::string(key), list);
      } else if (val.size() > 0 && val.at(0).is_number() && val.at(0).is_integer()) {
        std::vector<std::int64_t> list;
        for (std::size_t j = 0; j < val.size(); ++j) list.push_back(*val.at(j).int64());
        Value(s, std::string(key), list);
      }
    }
  }
  Ordered(s);
  const auto tensors = test_support::gemma4::Get(doc->root(), "tensors");
  for (std::size_t i = 0; i < tensors.size(); ++i) {
    const auto t = tensors.at(i);
    const std::string name = std::string(test_support::gemma4::Get(t, "name").string());
    const std::string type_name = std::string(test_support::gemma4::Get(t, "type").string());
    const auto ne = test_support::gemma4::Get(t, "ne");
    std::vector<std::uint64_t> dims;
    for (std::size_t j = 0; j < ne.size(); ++j)
      dims.push_back(static_cast<std::uint64_t>(*ne.at(j).int64()));
    fm::Encoding enc = fm::Encoding::kF32;
    if (type_name == "Q8_0") enc = fm::Encoding::kQ8_0;
    AddTensor(s, name, dims, enc);
  }
  return s;
}

std::expected<artifact::Artifact, artifact::Error> MakeTargetV1(
    const std::filesystem::path& target_scratch, std::string arch, std::string source_name,
    std::uint64_t bytes, std::string sha256, std::string container = "gguf",
    std::string stored_spec = {}) {
  llmp::test::V1Plan p;
  p.container = std::move(container);
  p.architecture = std::move(arch);
  if (!stored_spec.empty()) p.spec = std::move(stored_spec);
  p.groups = {
      llmp::test::V1Group{.kind = llmp::artifact::GroupKind::kTable,
                          .resources = {{.name = "token_embd.weight",
                                         .format = {.encoding = fm::Encoding::kBF16,
                                                    .layout = fm::Layout::kDense,
                                                    .shape = {.ne = {64, 64, 1, 1}, .rank = 2}}}}}};
  auto rendered = llmp::test::Render(p);
  llmp::test::Json::Array source_list;
  source_list.emplace_back(llmp::test::Json::Object{{"bytes", static_cast<std::int64_t>(bytes)},
                                                    {"name", std::move(source_name)},
                                                    {"sha256", std::move(sha256)}});
  rendered.manifest["source"] = llmp::test::Json(std::move(source_list));
  return llmp::artifact::Artifact::Open(llmp::test::Publish(target_scratch, rendered));
}

std::expected<artifact::Artifact, artifact::Error> MakeDrafterV1(
    const std::filesystem::path& drafter_scratch, std::string arch,
    std::span<const spec::SourceTensor> tensors, std::string source_name, std::uint64_t bytes,
    std::string sha256, std::string kept_gguf = {}, std::string container = "gguf",
    std::string stored_spec = {}) {
  llmp::test::V1Plan p;
  p.architecture = std::move(arch);
  p.container = std::move(container);
  p.sources.clear();
  if (!stored_spec.empty()) p.spec = std::move(stored_spec);
  // Kept as the importer keeps a GGUF drafter's header: meta/<stem>.kv.gguf.
  if (!kept_gguf.empty())
    p.meta.emplace_back(
        source_name.substr(0, source_name.size() - std::string_view(".gguf").size()) + ".kv.gguf",
        std::move(kept_gguf));
  llmp::test::V1Group g;
  g.kind = artifact::GroupKind::kGlobal;
  for (const auto& t : tensors) {
    g.resources.push_back({
        .name = t.name,
        .format = t.format,
        .part = t.part,
    });
  }
  p.groups.push_back(std::move(g));
  auto rendered = llmp::test::Render(p);
  llmp::test::Json::Array source_list;
  source_list.emplace_back(llmp::test::Json::Object{{"bytes", static_cast<std::int64_t>(bytes)},
                                                    {"name", std::move(source_name)},
                                                    {"sha256", std::move(sha256)}});
  rendered.manifest["source"] = llmp::test::Json(std::move(source_list));
  return llmp::artifact::Artifact::Open(llmp::test::Publish(drafter_scratch, rendered));
}

TEST(ImportDrafterNormalizeTest,
     SyntheticGemma4AssistantNormalizesPublishesAndAdmitsWithTargetComposition) {
  auto cp = SyntheticAssistantCheckpoint(26);
  if (!cp.has_value()) {
    GTEST_FAIL() << "no synthetic assistant fixture in " << TestDataDir();
  }
  const std::string kept_gguf = KeptGgufHeader(*cp);

  auto cost = NormalizationBound(*cp);
  ASSERT_TRUE(cost) << cost.error().ToString();
  NormalizationLimits limits{cost->metadata_bytes, cost->work_units};
  auto plan = test_support::NormalizeWithFixtureTarget(std::move(*cp), limits);
  ASSERT_TRUE(plan) << plan.error().ToString();

  ASSERT_EQ(plan->spec().stages.size(), 2U);
  EXPECT_EQ(plan->spec().stages[0].role, spec::StageRole::kTarget);
  EXPECT_EQ(plan->spec().stages[1].role, spec::StageRole::kDrafter);
  EXPECT_EQ(plan->spec().stages[1].artifact, 1U);
  EXPECT_EQ(plan->spec().stages[0].stream.width, 2816U);
  EXPECT_EQ(plan->spec().stages[1].stream.width, 1024U);
  EXPECT_EQ(plan->routes().size(), 49U);

  const std::filesystem::path scratch =
      std::filesystem::path(testing::TempDir()) / "import-drafter-synthetic-gemma4-asst";
  std::error_code ec;
  std::filesystem::remove_all(scratch, ec);
  std::filesystem::create_directories(scratch, ec);

  auto drafter_art =
      MakeDrafterV1(scratch / "drafter", "gemma4-assistant", plan->canonical().tensors,
                    "mtp-gemma-4-26B-A4B-it.gguf", 461766816ULL,
                    "6326fb9f5e487aa8dcdd313a091e3c67724cb2a666ec3b7d2895b5b26d93ed1b", kept_gguf,
                    "gguf", spec::ToJson(plan->spec()));
  ASSERT_TRUE(drafter_art) << (drafter_art ? "" : drafter_art.error().ToString());
  EXPECT_EQ(drafter_art->model().architecture, "gemma4-assistant");

  ASSERT_TRUE(drafter_art->spec_json().has_value());

  auto target_art = MakeTargetV1(
      scratch / "target", "gemma4", "gemma-4-26B-A4B-it-UD-Q4_K_M.gguf", 16947541728ULL,
      "f2c28b3dc4776931ac6f879e11f203dec637ea0f14267a86ec8f6165f63f293f");
  ASSERT_TRUE(target_art) << (target_art ? "" : target_art.error().ToString());

  auto drafter_source = spec::SourceOf(*drafter_art);
  ASSERT_TRUE(drafter_source) << drafter_source.error();
  // Drafter admission check with matching pinned origin
  std::array<artifact::Source, 1> origins{
      Origin("mtp-gemma-4-26B-A4B-it.gguf", 461766816ULL,
             "6326fb9f5e487aa8dcdd313a091e3c67724cb2a666ec3b7d2895b5b26d93ed1b")};
  auto admission = rt::CheckComponentDrafterAdmission(plan->spec(), *drafter_source, origins);
  ASSERT_TRUE(admission) << (admission ? "" : admission.error());

  auto stored = spec::StoredSpec(*drafter_art);
  ASSERT_TRUE(stored) << stored.error();
  EXPECT_EQ(*stored, plan->spec());
  auto admitted = rt::ReadComponentDrafterAdmission(*drafter_art);
  ASSERT_TRUE(admitted) << admitted.error();
  EXPECT_EQ(*admitted->model, plan->spec());
  EXPECT_TRUE(std::holds_alternative<spec::AnchoredDraft>(*stored->stages.back().drafting));

  std::filesystem::remove_all(scratch, ec);
}

TEST(ImportDrafterNormalizeTest, UsableKeptMetadataCannotReplaceMissingOrMalformedStoredExecution) {
  auto checkpoint = SyntheticAssistantCheckpoint(26);
  ASSERT_TRUE(checkpoint);
  const auto kept = KeptGgufHeader(*checkpoint);
  const std::array<std::string_view, 1> wanted{"gemma4-assistant.embedding_length"};
  const auto metadata = artifact::ReadGgufMetadata(std::as_bytes(std::span(kept)), wanted);
  ASSERT_TRUE(metadata) << metadata.error().ToString();
  ASSERT_TRUE(metadata->contains(wanted.front()));
  const std::array tensors{
      spec::SourceTensor{.name = "tiny.weight",
                         .format = {.encoding = fm::Encoding::kF32,
                                    .layout = fm::Layout::kDense,
                                    .shape = {.ne = {1, 1, 1, 1}, .rank = 1}}}};
  const auto scratch = std::filesystem::path(testing::TempDir()) / "drafter-stored-refusal";
  for (const std::string& stored : {std::string(), std::string("{}\n")}) {
    const auto artifact = MakeDrafterV1(scratch, "gemma4-assistant", tensors, "drafter.gguf", 1,
                                        std::string(64, '1'), kept, "gguf", stored);
    ASSERT_TRUE(artifact) << artifact.error().ToString();
    const auto admitted = rt::ReadComponentDrafterAdmission(*artifact);
    EXPECT_FALSE(admitted);
    if (stored.empty()) EXPECT_NE(admitted.error().find("carries no spec"), std::string::npos);
  }
  std::error_code error;
  std::filesystem::remove_all(scratch, error);
}

TEST(ImportDrafterNormalizeTest, NativeExtractionUsesCanonicalTargetAcrossSourceAliases) {
  const std::filesystem::path scratch =
      std::filesystem::path(testing::TempDir()) / "import-drafter-source-alias";
  std::error_code ec;
  std::filesystem::remove_all(scratch, ec);
  // The legacy native MTP artifact has no spec.json: its recipe extracts
  // architecture facts from the target spec and layer/head counts from its
  // own tensors. Tiny placeholders suffice for this metadata-only check.
  const std::array tensors{
      spec::SourceTensor{.name = "blk.0.attn_q.weight",
                         .format = {.encoding = fm::Encoding::kF32,
                                    .layout = fm::Layout::kDense,
                                    .shape = {.ne = {1, 1, 1, 1}, .rank = 2}}},
      spec::SourceTensor{.name = "draft_output.weight",
                         .format = {.encoding = fm::Encoding::kF32,
                                    .layout = fm::Layout::kDense,
                                    .shape = {.ne = {1, 64, 1, 1}, .rank = 2}}}};
  auto target_model = test_support::qwen38::NativeModelOpt().model;
  // These metadata-only placeholders represent unprocessed weights and carry no
  // preprocessing receipts; only their structural facts are used by this test.
  for (auto& stage : target_model.stages) stage.folds.clear();
  const auto target_spec = spec::ToJson(target_model);
  auto drafter = MakeDrafterV1(scratch / "drafter", "qwen4exp-mtp", tensors, "drafter.safetensors",
                               1, std::string(64, '2'), {}, "safetensors");
  ASSERT_TRUE(drafter) << drafter.error().ToString();
  ASSERT_FALSE(drafter->spec_json());
  for (const std::string container : {"safetensors", "mlx"}) {
    for (const std::string architecture : {"qwen4exp", "qwen3_8_flash_next", "qwen4_exp"}) {
      SCOPED_TRACE(container + ":" + architecture);
      auto target = MakeTargetV1(scratch / (container + architecture), architecture, "target.bin",
                                 1, std::string(64, '1'), container, target_spec);
      ASSERT_TRUE(target) << target.error().ToString();
      auto target_source = spec::SourceOf(*target);
      auto draft_source = spec::SourceOf(*drafter);
      ASSERT_TRUE(target_source && draft_source);
      const auto values = spec::ExtractValues(spec::mappings::Qwen38MtpNativeMapping(),
                                              target_model, *target_source, *draft_source);
      ASSERT_TRUE(values) << values.error();
      for (const auto& [key, expected] :
           {std::pair{"qwen4exp-mtp.block_count", std::int64_t{1}},
            std::pair{"qwen4exp-mtp.draft_vocab", std::int64_t{64}},
            std::pair{"qwen4exp-mtp.embedding_length", std::int64_t{2560}}}) {
        const auto found = std::ranges::find(*values, key, &spec::SourceValue::key);
        ASSERT_NE(found, values->end()) << key;
        EXPECT_EQ(std::get<std::int64_t>(found->value), expected) << key;
      }
    }
  }
  // Extraction is import-only. A complete source inventory without stored
  // execution structure cannot be admitted, regardless of a known registry key.
  EXPECT_FALSE(rt::ReadComponentDrafterAdmission(*drafter));
  std::filesystem::remove_all(scratch, ec);
}

TEST(ImportDrafterNormalizeTest, InvalidSourceListsCannotEraseAnOptionalRequirement) {
  for (const auto count : {spec::kMaxList, spec::kMaxList + 1}) {
    auto cp = SyntheticAssistantCheckpoint(26);
    ASSERT_TRUE(cp);
    Value(*cp, "gemma4-assistant.attention.shared_kv_layers", std::vector<std::int64_t>(count, 4));
    auto cost = NormalizationBound(*cp);
    if (!cost) continue;
    auto plan = test_support::NormalizeWithFixtureTarget(std::move(*cp),
                                                         {cost->metadata_bytes, cost->work_units});
    EXPECT_FALSE(plan);
  }
}

TEST(ImportDrafterNormalizeTest, RefusedTargetDiagnosticOutlivesValidation) {
  const auto refused = [] {
    auto target = test_support::GemmaImportTarget(26);
    target.model.stages.front().stream.width = 1;
    auto valid = ValidateTarget(target, spec::mappings::Gemma4AssistantMapping());
    EXPECT_FALSE(valid);
    return valid.error();
  }();
  EXPECT_EQ(refused.reason, "drafter target spec does not satisfy its components");
  EXPECT_EQ(refused.ToString(), "container: drafter target spec does not satisfy its components");
}

// A supplied target must satisfy the assistant's parent dimensions. A new
// structurally compatible size does not need an approved checkpoint profile.
TEST(ImportDrafterNormalizeTest, AssistantParentDimensionsMustMatchTheSuppliedTarget) {
  for (const auto& [out, parent, refused] :
       {std::tuple<std::int64_t, std::int64_t, bool>{5376, 0, false},
        std::tuple<std::int64_t, std::int64_t, bool>{1234, 0, true},
        std::tuple<std::int64_t, std::int64_t, bool>{2816, 5376, true}}) {
    SCOPED_TRACE(std::to_string(out) + "/" + std::to_string(parent));
    auto cp = SyntheticAssistantCheckpoint(26);
    if (!cp.has_value()) GTEST_FAIL() << "no synthetic assistant fixture in " << TestDataDir();
    Value(*cp, "gemma4-assistant.embedding_length_out", out);
    if (parent != 0) Value(*cp, "gemma4-assistant.Parent.embedding_length", parent);
    auto cost = NormalizationBound(*cp);
    if (!cost) {
      // A key the source mapping does not read is refused before selection.
      EXPECT_TRUE(refused) << cost.error().ToString();
      continue;
    }
    auto plan = test_support::NormalizeWithFixtureTarget(std::move(*cp),
                                                         {cost->metadata_bytes, cost->work_units});
    if (refused) {
      ASSERT_FALSE(plan);
      if (parent == 0)
        EXPECT_TRUE(
            plan.error().reason.starts_with("normalized tensors do not satisfy their components"));
      continue;
    }
    // The 26B assistant's tensors against the 31B target: selected, then
    // refused by the composition's own checks, never by the selection.
    if (!plan) {
      EXPECT_EQ(plan.error().ToString().find("unsupported drafter target"), std::string::npos)
          << plan.error().ToString();
    }
  }
}

TEST(ImportDrafterNormalizeModelsTest,
     Gemma4AssistantRealCheckpointsNormalizePublishAndAdmitWithTargetComposition) {
  bool unpublished = false;
  for (const std::uint32_t size : {26U, 31U}) {
    const char* env_var =
        size == 26 ? "LLMP_GEMMA4_26B_ASSISTANT_GGUF" : "LLMP_GEMMA4_31B_ASSISTANT_GGUF";
    const char* configured = std::getenv(env_var);
    const std::filesystem::path path =
        configured != nullptr
            ? std::filesystem::path(configured)
            : ModelsRoot() / (size == 26 ? "reference-models/mtp-gemma-4-26B-A4B-it.gguf"
                                         : "reference-models/mtp-gemma-4-31B-it.gguf");
    std::error_code ec;
    if (!std::filesystem::exists(path, ec)) {
      GTEST_FAIL() << "no Gemma 4 " << size << "B assistant GGUF: " << path;
    }

    auto checkpoint = ReadGguf(path);
    ASSERT_TRUE(checkpoint) << checkpoint.error().ToString();
    EXPECT_EQ(checkpoint->container, "gguf");
    EXPECT_EQ(checkpoint->architecture, "gemma4-assistant");
    EXPECT_EQ(checkpoint->tensors.size(), 49U);

    auto bound = NormalizationBound(*checkpoint);
    ASSERT_TRUE(bound) << bound.error().ToString();

    NormalizationLimits limits{bound->metadata_bytes, bound->work_units};
    auto target_art = PublishedTarget(size == 26 ? "gemma26" : "gemma31");
    ASSERT_TRUE(target_art) << target_art.error();
    auto target = TargetOf(*target_art);
    ASSERT_TRUE(target) << target.error();
    auto plan = Normalize(std::move(*checkpoint), limits, &*target);
    ASSERT_TRUE(plan) << plan.error().ToString();
    auto bad_target_art = PublishedTarget(size == 26 ? "gemma31" : "gemma26");
    ASSERT_TRUE(bad_target_art) << bad_target_art.error();
    auto bad_target = TargetOf(*bad_target_art);
    ASSERT_TRUE(bad_target) << bad_target.error();
    // Refuse the complete mismatched composition before any publication.
    EXPECT_FALSE(Normalize(plan->source(), limits, &*bad_target));

    // Composed spec has stage 0 target, stage 1 drafter
    ASSERT_EQ(plan->spec().stages.size(), 2U);
    EXPECT_EQ(plan->spec().stages[0].role, spec::StageRole::kTarget);
    EXPECT_EQ(plan->spec().stages[0].artifact, 0U);
    EXPECT_EQ(plan->spec().stages[1].role, spec::StageRole::kDrafter);
    EXPECT_EQ(plan->spec().stages[1].artifact, 1U);
    const std::uint32_t expected_target_width = size == 26 ? 2816 : 5376;
    EXPECT_EQ(plan->spec().stages[0].stream.width, expected_target_width);
    EXPECT_EQ(plan->spec().stages[1].stream.width, 1024U);
    EXPECT_EQ(plan->routes().size(), 49U);

    const char* store_root = std::getenv("LLMP_TEST_PUBLICATION_STORE");
    if (store_root == nullptr || *store_root == '\0') {
      unpublished = true;
      continue;
    }
    const std::filesystem::path scratch =
        std::filesystem::path(store_root) /
        ("import-gemma4-assistant-" + std::to_string(size) + "-models-test");
    std::filesystem::remove_all(scratch, ec);
    std::filesystem::create_directories(scratch, ec);

    WriteOptions options{
        .store = scratch / ("gemma4-asst-" + std::to_string(size) + "-v1"),
        .build = "test",
        .metadata_bytes = std::max(bound->metadata_bytes, std::uint64_t{1} << 30U),
    };
    auto written = WriteArtifact(*plan, options);
    ASSERT_TRUE(written) << written.error().ToString();
    EXPECT_FALSE(written->reused);

    // Opening the published drafter artifact
    auto drafter_art = artifact::Artifact::Open(written->path, {.expected_id = written->id});
    ASSERT_TRUE(drafter_art) << drafter_art.error().ToString();
    EXPECT_EQ(drafter_art->model().architecture, "gemma4-assistant");

    // Drafter artifact carries its composed spec.json
    EXPECT_TRUE(std::filesystem::exists(written->path / "spec.json"));
    EXPECT_TRUE(drafter_art->spec_json().has_value());

    // Verify drafter admission
    auto drafter_source = spec::SourceOf(*drafter_art);
    ASSERT_TRUE(drafter_source) << drafter_source.error();
    auto admission =
        rt::CheckComponentDrafterAdmission(plan->spec(), *drafter_source, drafter_art->sources());
    ASSERT_TRUE(admission) << (admission ? "" : admission.error());

    auto stored = spec::StoredSpec(*drafter_art);
    ASSERT_TRUE(stored) << stored.error();
    EXPECT_EQ(*stored, plan->spec());
    auto composed = spec::ComposeStoredDrafter(target->model, *stored);
    ASSERT_TRUE(composed) << composed.error();
    const std::array<std::span<const spec::SourceTensor>, 2> tables{target->source.tensors,
                                                                    drafter_source->tensors};
    EXPECT_TRUE(components::ValidateSpec(*composed, tables));
    EXPECT_TRUE(rt::CheckComposedAdmission(*composed, *target_art, *drafter_art));

    RecordProperty("drafter_id", written->id);
    RecordProperty("drafter_path", written->path.string());
    RecordProperty("routes", "49");

    std::filesystem::remove_all(scratch, ec);
  }
  if (unpublished) RecordProperty("published", "false");
}

TEST(ImportDrafterNormalizeModelsTest,
     DsparkRealCheckpointNormalizesPublishesAndAdmitsWithTargetComposition) {
  const char* configured = std::getenv("LLMP_DSPARK_GGUF");
  const std::filesystem::path path =
      configured != nullptr ? std::filesystem::path(configured)
                            : ModelsRoot() /
                                  "models/unsloth/DeepSeek-V4-Flash-0731-GGUF@fbbb5b93/"
                                  "dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf";
  std::error_code ec;
  if (!std::filesystem::exists(path, ec)) {
    GTEST_FAIL() << "no DSpark GGUF: " << path;
  }

  auto checkpoint = ReadGguf(path);
  ASSERT_TRUE(checkpoint) << checkpoint.error().ToString();
  EXPECT_EQ(checkpoint->container, "gguf");
  EXPECT_EQ(checkpoint->architecture, "dflash");

  auto bound = NormalizationBound(*checkpoint);
  ASSERT_TRUE(bound) << bound.error().ToString();

  NormalizationLimits limits{bound->metadata_bytes, bound->work_units};
  auto target_art = PublishedTarget("deepseek");
  ASSERT_TRUE(target_art) << target_art.error();
  auto target = TargetOf(*target_art);
  ASSERT_TRUE(target) << target.error();
  auto plan = Normalize(std::move(*checkpoint), limits, &*target);
  ASSERT_TRUE(plan) << plan.error().ToString();
  auto gemma_target_art = PublishedTarget("gemma26");
  ASSERT_TRUE(gemma_target_art) << gemma_target_art.error();
  auto gemma_target = TargetOf(*gemma_target_art);
  ASSERT_TRUE(gemma_target) << gemma_target.error();
  EXPECT_FALSE(Normalize(plan->source(), limits, &*gemma_target));

  ASSERT_EQ(plan->spec().stages.size(), 2U);
  EXPECT_EQ(plan->spec().stages[0].role, spec::StageRole::kTarget);
  EXPECT_EQ(plan->spec().stages[0].artifact, 0U);
  EXPECT_EQ(plan->spec().stages[1].role, spec::StageRole::kDrafter);
  EXPECT_EQ(plan->spec().stages[1].artifact, 1U);

  const char* store_root = std::getenv("LLMP_TEST_PUBLICATION_STORE");
  if (store_root == nullptr || *store_root == '\0') {
    RecordProperty("published", "false");
    return;
  }
  const std::filesystem::path scratch =
      std::filesystem::path(store_root) / "import-dspark-models-test";
  std::filesystem::remove_all(scratch, ec);
  std::filesystem::create_directories(scratch, ec);

  WriteOptions options{
      .store = scratch / "dspark-v1",
      .build = "test",
      .metadata_bytes = std::max(bound->metadata_bytes, std::uint64_t{1} << 30U),
  };
  auto written = WriteArtifact(*plan, options);
  ASSERT_TRUE(written) << written.error().ToString();
  EXPECT_FALSE(written->reused);

  auto drafter_art = artifact::Artifact::Open(written->path, {.expected_id = written->id});
  ASSERT_TRUE(drafter_art) << drafter_art.error().ToString();
  EXPECT_EQ(drafter_art->model().architecture, "dflash");

  // Drafter artifact carries its composed spec.json
  EXPECT_TRUE(std::filesystem::exists(written->path / "spec.json"));
  EXPECT_TRUE(drafter_art->spec_json().has_value());

  // Verify drafter admission
  auto drafter_source = spec::SourceOf(*drafter_art);
  ASSERT_TRUE(drafter_source) << drafter_source.error();
  auto admission =
      rt::CheckComponentDrafterAdmission(plan->spec(), *drafter_source, drafter_art->sources());
  ASSERT_TRUE(admission) << (admission ? "" : admission.error());

  auto stored = spec::StoredSpec(*drafter_art);
  ASSERT_TRUE(stored) << stored.error();
  EXPECT_EQ(*stored, plan->spec());
  auto composed = spec::ComposeStoredDrafter(target->model, *stored);
  ASSERT_TRUE(composed) << composed.error();
  const std::array<std::span<const spec::SourceTensor>, 2> tables{target->source.tensors,
                                                                  drafter_source->tensors};
  EXPECT_TRUE(components::ValidateSpec(*composed, tables));
  EXPECT_TRUE(rt::CheckComposedAdmission(*composed, *target_art, *drafter_art));
  auto wrong_target = spec::StoredSpec(*gemma_target_art);
  ASSERT_TRUE(wrong_target) << wrong_target.error();
  auto wrong = spec::ComposeStoredDrafter(*wrong_target, *stored);
  EXPECT_TRUE(!wrong || !rt::CheckComposedAdmission(*wrong, *gemma_target_art, *drafter_art));

  RecordProperty("drafter_id", written->id);
  RecordProperty("drafter_path", written->path.string());

  std::filesystem::remove_all(scratch, ec);
}

}  // namespace
}  // namespace llmp::import
