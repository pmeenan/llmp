// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The composition reader (artifact/composition.h, D-089, D-113): a valid
// v1 composition built here, and each rule broken one at a time. The real
// Qwen-Image-2.1 composition is opened where it is installed (a Spark) and
// skipped elsewhere.

#include "artifact/composition.h"

#include <gtest/gtest.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdlib>
#include <filesystem>
#include <format>
#include <fstream>
#include <functional>
#include <iterator>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include "artifact/error.h"
#include "base/sha256.h"

namespace {

namespace fs = std::filesystem;
using llmp::artifact::OpenComposition;
using llmp::artifact::RuleName;

std::string Hex(std::string_view bytes) {
  return llmp::base::ToHex(llmp::base::Sha256().Update(bytes).Finish());
}

void Write(const fs::path& path, std::string_view bytes) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  out << bytes;
}

// The pieces of a composition's manifest; Manifest() writes them in
// canonical form (sorted keys, no whitespace, one newline).
struct Spec {
  std::string format = "llmp-composition";
  std::string version = "1";
  std::string experimental = "true";
  std::string architecture = "QwenImage21Pipeline";
  std::vector<std::pair<std::string, std::string>> components = {
      {"text_encoder", std::string(64, 'a')}, {"transformer", std::string(64, 'b')}};
  std::vector<std::pair<std::string, std::string>> kept = {
      {"model_index.json", R"({"_class_name": "QwenImage21Pipeline"})"},
      {"scheduler_config.json", R"({"shift": 1})"}};
  std::string extra_key;  // inserted verbatim, in order, before "files"

  std::string Manifest() const {
    std::string components_json;
    for (const auto& [role, id] : components) {
      components_json +=
          std::format(R"({}{{"architecture":"Arch_{}","artifact":"{}","role":"{}"}})",
                      components_json.empty() ? "" : ",", role, id, role);
    }
    std::string files;
    std::string sources;
    for (const auto& [name, bytes] : kept) {
      files +=
          std::format(R"({}{{"bytes":{},"path":"meta/{}","role":"source-metadata","sha256":"{}"}})",
                      files.empty() ? "" : ",", bytes.size(), name, Hex(bytes));
      sources += std::format(R"({}{{"bytes":{},"name":"{}","sha256":"{}"}})",
                             sources.empty() ? "" : ",", bytes.size(), name, Hex(bytes));
    }
    return std::format(
        R"({{"components":[{}],"experimental":{},{}"files":[{}],"format":"{}","format_version":{},"importer":{{"build":"unknown","name":"llmp-import","options_sha256":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","version":"0.0.0"}},"model":{{"architecture":"{}"}},"source":[{}]}})"
        "\n",
        components_json, experimental, extra_key, files, format, version, architecture, sources);
  }
};

class CompositionTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const char* scratch = std::getenv("LLMP_TEST_SCRATCH");  // NOLINT(concurrency-mt-unsafe)
    base_ = fs::path(scratch != nullptr ? scratch : fs::temp_directory_path().string()) /
            std::format("composition-{}", ::getpid());
    fs::remove_all(base_);
    fs::create_directories(base_);
  }
  void TearDown() override { fs::remove_all(base_); }

  // Writes the spec as a composition directory named by its manifest's
  // digest, then lets `mutate` change it; returns the directory.
  fs::path Build(const Spec& spec, const std::function<void(const fs::path&)>& mutate = {}) {
    const std::string manifest = spec.Manifest();
    const fs::path root = base_ / Hex(manifest);
    fs::remove_all(root);
    fs::create_directories(root / "meta");
    Write(root / "manifest.json", manifest);
    for (const auto& [name, bytes] : spec.kept) {
      Write(root / "meta" / name, bytes);
    }
    if (mutate) {
      mutate(root);
    }
    return root;
  }

  static std::string RuleOf(const fs::path& root,
                            const std::optional<std::string>& id = std::nullopt) {
    auto opened = OpenComposition(root, id);
    return opened ? "ok" : std::string(RuleName(opened.error().rule));
  }

  fs::path base_;
};

TEST_F(CompositionTest, OpensAValidCompositionAndItsKeptFiles) {
  const Spec spec;
  const fs::path root = Build(spec);
  auto c = OpenComposition(root);
  ASSERT_TRUE(c.has_value()) << c.error().ToString();
  EXPECT_EQ(c->id(), root.filename().string());
  EXPECT_EQ(c->architecture(), "QwenImage21Pipeline");
  ASSERT_EQ(c->components().size(), 2U);
  const auto* te = c->Find("text_encoder");
  ASSERT_NE(te, nullptr);
  EXPECT_EQ(te->artifact, std::string(64, 'a'));
  EXPECT_EQ(te->architecture, "Arch_text_encoder");
  EXPECT_EQ(c->Find("vae"), nullptr);
  EXPECT_EQ(c->Metadata("scheduler_config.json").value_or(""), R"({"shift": 1})");
  EXPECT_FALSE(c->Metadata("manifest.json").has_value());
  EXPECT_EQ(c->converter().name, "llmp-import");
  EXPECT_EQ(c->converter().version, "0.0.0");
}

TEST_F(CompositionTest, RefusesV0CompositionFormatAndVersion) {
  Spec v0_format;
  v0_format.format = "jitllm-composition";
  EXPECT_EQ(RuleOf(Build(v0_format)), "format");
  Spec v0_version;
  v0_version.version = "0";
  EXPECT_EQ(RuleOf(Build(v0_version)), "unsupported-version");
}

TEST_F(CompositionTest, RefusesEachBrokenRuleByName) {
  // The directory.
  EXPECT_EQ(RuleOf(base_ / "missing"), "file-type");
  EXPECT_EQ(RuleOf(Build({}, [](const fs::path& r) { fs::create_directory(r / "data"); })),
            "file-set");
  EXPECT_EQ(RuleOf(Build({}, [](const fs::path& r) { fs::create_directory(r / "other"); })),
            "file-set");
  EXPECT_EQ(RuleOf(Build({}, [](const fs::path& r) { Write(r / "meta" / "extra.json", "{}"); })),
            "file-set");
  EXPECT_EQ(RuleOf(Build({},
                         [](const fs::path& r) {
                           fs::create_symlink(r / "meta" / "model_index.json",
                                              r / "meta" / "link.json");
                         })),
            "file-type");
  EXPECT_EQ(RuleOf(Build({},
                         [this](const fs::path& r) {
                           Write(base_ / "external_manifest.json", "{}");
                           fs::remove(r / "manifest.json");
                           fs::create_symlink(base_ / "external_manifest.json",
                                              r / "manifest.json");
                         })),
            "file-type");
  // A hard-linked kept file is read as any file, and the root may be a
  // link (D-056 as amended on 2026-10-09).
  EXPECT_EQ(RuleOf(Build({},
                         [this](const fs::path& r) {
                           fs::create_hard_link(r / "meta" / "model_index.json",
                                                base_ / "second-name");
                         })),
            "ok");
  {
    const fs::path root = Build({});
    fs::create_directory(base_ / "store");
    fs::create_directory_symlink(root, base_ / "store" / root.filename());
    EXPECT_EQ(RuleOf(base_ / "store" / root.filename()), "ok");
  }
  // The manifest.
  Spec wrong_format;
  wrong_format.format = "jitllm-artifact";
  EXPECT_EQ(RuleOf(Build(wrong_format)), "format");
  Spec version;
  version.version = "0";
  EXPECT_EQ(RuleOf(Build(version)), "unsupported-version");
  EXPECT_EQ(RuleOf(Build({},
                         [](const fs::path& r) {
                           std::ifstream in(r / "manifest.json");
                           std::string text((std::istreambuf_iterator<char>(in)), {});
                           Write(r / "manifest.json", " " + text);
                         })),
            "canonical");
  EXPECT_EQ(RuleOf(Build({}), std::string(64, '0')), "identity");
  Spec experimental;
  experimental.experimental = "false";
  EXPECT_EQ(RuleOf(Build(experimental)), "schema");
  Spec extra;
  extra.extra_key = R"("extra":1,)";
  EXPECT_EQ(RuleOf(Build(extra)), "schema");
  Spec role;
  role.components = {{"Transformer", std::string(64, 'b')}};
  EXPECT_EQ(RuleOf(Build(role)), "schema");
  Spec id;
  id.components = {{"transformer", std::string(63, 'b')}};
  EXPECT_EQ(RuleOf(Build(id)), "schema");
  Spec order;
  order.components = {{"transformer", std::string(64, 'b')},
                      {"text_encoder", std::string(64, 'a')}};
  EXPECT_EQ(RuleOf(Build(order)), "canonical");
  Spec duplicate;
  duplicate.components = {{"transformer", std::string(64, 'b')},
                          {"transformer", std::string(64, 'c')}};
  EXPECT_EQ(RuleOf(Build(duplicate)), "canonical");
  // Components are IDs, never paths.
  Spec traversal;
  traversal.components = {{"transformer", "../" + std::string(61, 'b')}};
  EXPECT_EQ(RuleOf(Build(traversal)), "schema");
  Spec upper;
  upper.components = {{"transformer", std::string(64, 'B')}};
  EXPECT_EQ(RuleOf(Build(upper)), "schema");
  // A kept file named as a path out of meta/ (Build writes it there, so
  // only the name rule refuses it).
  Spec escape;
  escape.kept.insert(escape.kept.begin(), {"../x.json", "{}"});
  EXPECT_EQ(RuleOf(Build(escape)), "schema");
  // Nesting past the parser's depth cap.
  Spec deep;
  deep.extra_key = R"("deep":[[[[[[[[[[1]]]]]]]]]],)";
  EXPECT_NE(RuleOf(Build(deep)), "ok");
  Spec none;
  none.components = {};
  EXPECT_EQ(RuleOf(Build(none)), "schema");
  Spec many;
  many.components.clear();
  for (char c = 'a'; c < 'a' + 17; ++c) {
    many.components.emplace_back(std::string("r") + c, std::string(64, 'c'));
  }
  EXPECT_EQ(RuleOf(Build(many)), "schema");
  // The kept files.
  EXPECT_EQ(RuleOf(Build({},
                         [](const fs::path& r) {
                           Write(r / "meta" / "scheduler_config.json", R"({"shift": 2})");
                         })),
            "hash");
  EXPECT_EQ(RuleOf(Build({},
                         [](const fs::path& r) {
                           Write(r / "meta" / "scheduler_config.json", R"({"shift": 22})");
                         })),
            "file-size");
  EXPECT_EQ(
      RuleOf(Build({}, [](const fs::path& r) { fs::remove(r / "meta" / "model_index.json"); })),
      "file-set");
}

}  // namespace
