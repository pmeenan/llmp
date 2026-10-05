// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// Actual approved prepared metadata/weights directories, without model work.
#include <gtest/gtest.h>

#include <array>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <span>
#include <string>

#include "artifact/artifact.h"
#include "artifact/gguf_metadata.h"
#include "model/gemma4_assistant.h"
#include "tokenizer/gguf.h"

namespace md = jitllm::model;
namespace ar = jitllm::artifact;
namespace tk = jitllm::tokenizer;
namespace {
constexpr std::array targets{"4ddb360c9ce08f1e984ab304b6af918be44246d52346734066b06443f7c249d3",
                             "32c92e077a6816b54aa988e2dee61a3639c958fd510ea99e25f3621f10b2aa08"};
constexpr std::array assistants{"1040a0299a459e00ad0a77efd77bd319ac593986ba2c9ef29eb03d07ce97db42",
                                "447a5c20a0a25632bf35e118d5dde1867a182cd209b9ccd3afe93866c3696120"};
constexpr std::array target_files{"gemma-4-26B-A4B-it-UD-Q4_K_M.kv.gguf",
                                  "gemma-4-31B-it-UD-Q4_K_XL.kv.gguf"};
constexpr std::array assistant_files{"mtp-gemma-4-26B-A4B-it.kv.gguf",
                                     "mtp-gemma-4-31B-it.kv.gguf"};
std::filesystem::path Store() {
  const auto* home = std::getenv("HOME");  // NOLINT(concurrency-mt-unsafe)
  return std::filesystem::path(home ? home : "") / ".local/share/jitllm/m3-artifacts";
}
std::filesystem::path AssistantMetadata() {
  const auto* root = std::getenv("JITLLM_ASSISTANT_METADATA");  // NOLINT(concurrency-mt-unsafe)
  return root ? std::filesystem::path(root) : Store();
}
std::expected<std::string, std::string> Read(const std::filesystem::path& path) {
  std::error_code error;
  const auto bytes = std::filesystem::file_size(path, error);
  if (error || bytes == 0 || bytes > (16ULL << 20))
    return std::unexpected("actual metadata missing or outside the byte cap: " + path.string());
  std::ifstream file(path, std::ios::binary);
  std::string result(static_cast<std::size_t>(bytes), '\0');
  file.read(result.data(), static_cast<std::streamsize>(bytes));
  if (!file || file.peek() != std::char_traits<char>::eof())
    return std::unexpected("actual metadata read failed");
  return result;
}
struct Vocabulary {
  tk::GgufTokenizer tokenizer;
  ar::GgufMetadata raw;
  md::Gemma4AssistantVocabulary view() const {
    return {.tokens = tokenizer.spec.tokens,
            .merges = tokenizer.spec.merges,
            .scores = raw.at("tokenizer.ggml.scores").reals,
            .types = raw.at("tokenizer.ggml.token_type").integers};
  }
};
std::expected<Vocabulary, std::string> Decode(std::string_view bytes) {
  const auto input = std::as_bytes(std::span(bytes.data(), bytes.size()));
  auto tokenizer = tk::ReadGgufTokenizer(input);
  if (!tokenizer) return std::unexpected("native actual tokenizer decode failed");
  constexpr std::array<std::string_view, 2> keys{"tokenizer.ggml.scores",
                                                 "tokenizer.ggml.token_type"};
  auto raw = ar::ReadGgufMetadata(input, keys);
  if (!raw || !raw->contains(keys[0]) || !raw->contains(keys[1]) ||
      raw->at(std::string(keys[0])).kind != ar::GgufValue::Kind::kFloats ||
      raw->at(std::string(keys[1])).kind != ar::GgufValue::Kind::kIntegers)
    return std::unexpected("actual raw score/type arrays missing");
  return Vocabulary{.tokenizer = std::move(*tokenizer), .raw = std::move(*raw)};
}
class Gemma4AssistantArtifact : public ::testing::TestWithParam<unsigned> {};
TEST_P(Gemma4AssistantArtifact, ActualNativeVocabularyAndKeptSemanticsMatchPairedTarget) {
  const auto i = GetParam();
  const auto& profile = i == 0 ? md::Gemma4Assistant26() : md::Gemma4Assistant31();
  const auto target = Read(Store() / targets[i] / "meta" / target_files[i]);
  const auto assistant = Read(AssistantMetadata() / assistants[i] / "meta" / assistant_files[i]);
  ASSERT_TRUE(target) << (target ? "" : target.error());
  ASSERT_TRUE(assistant) << (assistant ? "" : assistant.error());
  const auto semantic = md::CheckGemma4AssistantMetadata(
      profile, std::as_bytes(std::span(assistant->data(), assistant->size())));
  ASSERT_TRUE(semantic) << (semantic ? "" : semantic.error());
  const auto t = Decode(*target), a = Decode(*assistant);
  ASSERT_TRUE(t) << (t ? "" : t.error());
  ASSERT_TRUE(a) << (a ? "" : a.error());
  const auto compatible = md::CheckGemma4AssistantVocabulary(t->view(), a->view());
  ASSERT_TRUE(compatible) << (compatible ? "" : compatible.error());
  unsigned differences = 0;
  for (std::size_t id = 0; id < t->view().types.size(); ++id)
    if (t->view().types[id] != a->view().types[id]) {
      EXPECT_TRUE(id == 1 || id == 258884);
      ++differences;
    }
  EXPECT_EQ(differences, 2);
}
TEST_P(Gemma4AssistantArtifact, ActualCompleteArtifactBindsOnlyToItsClosedTarget) {
  const auto i = GetParam();
  const auto& profile = i == 0 ? md::Gemma4Assistant26() : md::Gemma4Assistant31();
  const auto target = ar::Artifact::Open(Store() / targets[i]);
  const auto assistant = ar::Artifact::Open(Store() / assistants[i]);
  ASSERT_TRUE(target);
  ASSERT_TRUE(assistant);
  const auto& tp = i == 0 ? md::Gemma4_26BA4B() : md::Gemma4_31B();
  const auto ab = md::BindGemma4Assistant(profile, *assistant);
  const auto tb = md::BindGemma4(tp, *target);
  ASSERT_TRUE(ab) << (ab ? "" : ab.error());
  ASSERT_TRUE(tb) << (tb ? "" : tb.error());
  const auto shared = md::CheckGemma4AssistantTarget(profile, *ab, tp, *tb);
  ASSERT_TRUE(shared) << (shared ? "" : shared.error());
  EXPECT_EQ(ab->embedding, ab->head);
  EXPECT_FALSE(md::BindGemma4Assistant(i == 0 ? md::Gemma4Assistant31() : md::Gemma4Assistant26(),
                                       *assistant));
  EXPECT_FALSE(md::BindGemma4Assistant(profile, *target));
}
INSTANTIATE_TEST_SUITE_P(Paired, Gemma4AssistantArtifact, ::testing::Values(0U, 1U));
}  // namespace
