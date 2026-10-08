// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The v0 composition reader (artifact/composition.h, D-089): a valid
// composition built here as the prototype writes it (import_m3.py compose),
// and each rule the prototype's verify_composition names, broken one at a
// time. The real Qwen-Image-2.1 composition is opened where it is installed
// (a Spark) and skipped elsewhere.

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
  std::string format = "jitllm-composition";
  std::string version = "0";
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
        R"({{"components":[{}],"converter":{{"name":"artifact-layout/import_m3.py","version":"m3-2+layout-a0d1980a9eddd1ad"}},"experimental":{},{}"files":[{}],"format":"{}","format_version":{},"model":{{"architecture":"{}"}},"source":[{}]}})"
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
  EXPECT_EQ(c->converter().version, "m3-2+layout-a0d1980a9eddd1ad");
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
                         [](const fs::path& r) {
                           fs::create_hard_link(r / "meta" / "model_index.json", r / "hard.json");
                         })),
            "file-type");
  // The manifest.
  Spec wrong_format;
  wrong_format.format = "jitllm-artifact";
  EXPECT_EQ(RuleOf(Build(wrong_format)), "format");
  Spec version;
  version.version = "1";
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
  Spec no_index;
  no_index.kept = {{"scheduler_config.json", "{}"}};
  EXPECT_EQ(RuleOf(Build(no_index)), "file-set");
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

// The installed Qwen-Image-2.1 composition and its three components, where
// the M3 store is (docs/experiments/qwen-image-native).
TEST(InstalledComposition, QwenImage) {
  const char* home = std::getenv("HOME");  // NOLINT(concurrency-mt-unsafe)
  const fs::path store = fs::path(home != nullptr ? home : "/") / ".local/share/llmp/m3-artifacts";
  std::error_code error;
  std::optional<fs::path> found;
  for (const auto& entry : fs::directory_iterator(store, error)) {
    if (entry.path().filename().string().size() == 64 &&
        fs::exists(entry.path() / "meta" / "model_index.json") &&
        !fs::exists(entry.path() / "index.json")) {
      found = entry.path();
    }
  }
  if (!found) {
    GTEST_SKIP() << "no installed composition in " << store;
  }
  auto c = OpenComposition(*found);
  ASSERT_TRUE(c.has_value()) << c.error().ToString();
  EXPECT_EQ(c->architecture(), "QwenImage21Pipeline");
  for (const auto& component : c->components()) {
    auto a = llmp::artifact::Artifact::Open(store / component.artifact);
    ASSERT_TRUE(a.has_value()) << component.role << ": " << a.error().ToString();
    EXPECT_EQ(a->model().architecture, component.architecture);
  }
}

}  // namespace
