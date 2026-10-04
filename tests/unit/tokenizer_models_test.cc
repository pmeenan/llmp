// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// The native tokenizer and renderers against the reference outputs stored
// in tests/unit/data, on the real vocabularies. Label `models`: the model
// files live on the Sparks (under JITLLM_TEST_MODELS, by default
// ~/.local/share/jitllm, as docs/tokenizer.md lists), and a test whose files
// are absent skips. A file that is present but differs from the one the
// fixture was generated from fails.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <map>
#include <memory>
#include <print>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "base/json.h"
#include "base/sha256.h"
#include "chat/chat.h"
#include "tokenizer/gguf.h"
#include "tokenizer/hf.h"
#include "tokenizer/tokenizer.h"
#include "tokenizer/unicode.h"
#include "tokenizer_fixtures.h"

namespace {

namespace chat = jitllm::chat;
namespace json = jitllm::base::json;
namespace tok = jitllm::tokenizer;
namespace uni = jitllm::tokenizer::unicode;
using jitllm::test_support::ConversationFrom;
using jitllm::test_support::Get;
using jitllm::test_support::Ids;
using jitllm::test_support::Int;
using jitllm::test_support::LoadJson;
using jitllm::test_support::ModelsDir;
using jitllm::test_support::ReadFile;
using tok::TokenId;

std::string Sha256(std::string_view bytes) {
  jitllm::base::Sha256 h;
  h.Update(bytes);
  return jitllm::base::ToHex(h.Finish());
}

std::string FromHex(std::string_view hex) {
  std::string out;
  for (std::size_t i = 0; i + 1 < hex.size(); i += 2) {
    unsigned value = 0;
    std::from_chars(hex.data() + i, hex.data() + i + 2, value, 16);
    out.push_back(static_cast<char>(value));
  }
  return out;
}

// The corpus items' names and bytes, in order.
std::vector<std::pair<std::string, std::string>> Corpus() {
  const json::Document doc = LoadJson("tokenizer/corpus.json");
  const json::Value items = Get(doc.root(), "items");
  std::vector<std::pair<std::string, std::string>> out;
  for (std::size_t i = 0; i < items.size(); ++i) {
    const json::Value item = items.at(i);
    std::string data;
    if (const auto hex = item.find("hex")) {
      data = FromHex(hex->string());
    } else if (const auto repeat = item.find("repeat")) {
      for (std::int64_t k = 0; k < Int(Get(item, "count")); ++k) {
        data += repeat->string();
      }
    } else {
      data = std::string(Get(item, "text").string());
    }
    out.emplace_back(std::string(Get(item, "name").string()), std::move(data));
  }
  return out;
}

struct Loaded {
  std::unique_ptr<tok::Tokenizer> tokenizer;  // null when not loaded
  std::string path;
  std::string chat_template;  // GGUF only
  std::string skip;           // why not loaded, when the file is absent
};

// Loads a tokenizer configuration's model file, as the fixture names it.
Loaded Load(std::string_view config, std::string_view kind) {
  const json::Document fixture = LoadJson("tokenizer/" + std::string(config) + ".json");
  const json::Value source = Get(fixture.root(), "source");
  Loaded out;
  out.path = ModelsDir() + "/" + std::string(Get(source, "path").string());
  const auto bytes = ReadFile(out.path, std::size_t{512} << 20U);
  if (!bytes) {
    out.skip = "no " + out.path;
    return out;
  }
  EXPECT_EQ(Sha256(*bytes), Get(source, "sha256").string())
      << out.path << " is not the file the fixture was generated from";
  std::expected<tok::TokenizerSpec, tok::Error> spec;
  if (kind == "gguf") {
    auto g = tok::ReadGgufTokenizer(std::as_bytes(std::span(*bytes)));
    if (!g) {
      ADD_FAILURE() << out.path << ": " << g.error().ToString();
      return out;
    }
    out.chat_template = g->chat_template;
    spec = std::move(g->spec);
  } else {
    spec = tok::ReadHfTokenizer(*bytes);
  }
  if (!spec) {
    ADD_FAILURE() << out.path << ": " << spec.error().ToString();
    return out;
  }
  auto t = tok::Tokenizer::Create(std::move(*spec));
  if (!t) {
    ADD_FAILURE() << out.path << ": " << t.error().ToString();
    return out;
  }
  out.tokenizer = std::make_unique<tok::Tokenizer>(std::move(*t));
  return out;
}

// Items where the native tokenizer agrees with llama.cpp and deliberately
// not with Hugging Face tokenizers, and why (docs/tokenizer.md).
const std::map<std::pair<std::string_view, std::string_view>, std::string_view> kKnownDivergences =
    {
        {{"deepseek-v4-0731-hf", "emoji-recent"},
         "Unicode 16.0 emoji: unassigned (Cn) in UCD 15.1, as in llama.cpp; symbols (So) in HF's "
         "newer Oniguruma"},
};

struct Config {
  std::string_view name;
  std::string_view kind;        // gguf or hf
  std::string_view llama_twin;  // the llama.cpp configuration a divergence must match instead
};

constexpr std::array<Config, 9> kConfigs = {{
    {"gemma-4-26b-gguf", "gguf", ""},
    {"gemma-2-2b-gguf", "gguf", ""},
    {"gemma-3-4b-gguf", "gguf", ""},
    {"phi-3.5-gguf", "gguf", ""},
    {"deepseek-v4-0731-gguf", "gguf", ""},
    {"qwen3.8-gguf", "gguf", ""},
    {"qwen3.8-nvfp4", "hf", ""},
    {"qwen-image-2.1", "hf", ""},
    {"deepseek-v4-0731-hf", "hf", "deepseek-v4-0731-gguf"},
}};

class Agreement : public testing::TestWithParam<Config> {};

TEST_P(Agreement, TokenForTokenWithTheReference) {
  const Config& config = GetParam();
  const Loaded loaded = Load(config.name, config.kind);
  if (!loaded.skip.empty()) {
    GTEST_SKIP() << loaded.skip;
  }
  ASSERT_NE(loaded.tokenizer, nullptr);
  const tok::Tokenizer& t = *loaded.tokenizer;
  const json::Document fixture = LoadJson("tokenizer/" + std::string(config.name) + ".json");
  const json::Value items = Get(fixture.root(), "items");
  std::unique_ptr<json::Document> twin;
  if (!config.llama_twin.empty()) {
    twin = std::make_unique<json::Document>(
        LoadJson("tokenizer/" + std::string(config.llama_twin) + ".json"));
  }

  const auto corpus = Corpus();
  ASSERT_EQ(items.size(), corpus.size());
  std::size_t compared = 0;
  std::size_t tokens = 0;
  for (std::size_t i = 0; i < corpus.size(); ++i) {
    const auto& [name, bytes] = corpus[i];
    const json::Value item = items.at(i);
    ASSERT_EQ(Get(item, "name").string(), name);
    std::string text = bytes;
    if (!Get(item, "well_formed").boolean()) {
      // Refused, at the first ill-formed byte; the lossy replacement then
      // matches what the reference made of Python's "replace" decoding.
      std::vector<TokenId> ids;
      const auto r = t.Encode(bytes, {}, ids);
      ASSERT_FALSE(r.has_value()) << name;
      EXPECT_EQ(r.error().rule, tok::Rule::kInvalidUtf8) << name;
      EXPECT_EQ(r.error().item, json::FirstInvalidUtf8(bytes).value_or(0)) << name;
      text = uni::ReplaceInvalidUtf8(bytes);
    }
    for (const auto& [mode, special] : {std::pair{"parse", tok::SpecialTokens::kParse},
                                        std::pair{"plain", tok::SpecialTokens::kUserDefinedOnly}}) {
      std::vector<TokenId> ids;
      tok::EncodeOptions options;
      options.special = special;
      const auto r = t.Encode(text, options, ids);
      ASSERT_TRUE(r.has_value()) << name << ": " << r.error().ToString();
      const std::vector<TokenId> expected = Ids(Get(item, mode));
      if (const auto it = kKnownDivergences.find({config.name, name});
          it != kKnownDivergences.end()) {
        ASSERT_NE(twin, nullptr);
        EXPECT_NE(ids, expected) << name << " no longer diverges: " << it->second;
        EXPECT_EQ(ids, Ids(Get(Get(twin->root(), "items").at(i), mode)))
            << name << " (llama.cpp's)";
        continue;
      }
      EXPECT_EQ(ids, expected) << config.name << " " << mode << " " << name;
      ++compared;
      tokens += ids.size();
      // Decoding gives the text back, where the tokenizer does not
      // normalize it.
      if (t.normalization() == tok::Normalization::kNone &&
          t.pre_tokenizer() != tok::PreTokenizer::kSentencePiece) {
        std::string back;
        ASSERT_TRUE(t.Decode(ids, {.control_tokens = true}, back).has_value());
        if (t.pre_tokenizer() == tok::PreTokenizer::kGemma4) {
          std::string normalized;
          for (std::size_t at = 0; at < text.size();) {
            if (std::string_view(text).substr(at).starts_with("▁")) {
              normalized += ' ';
              at += 3;
            } else {
              normalized += text[at++];
            }
          }
          EXPECT_EQ(back, normalized) << name;
        } else {
          EXPECT_EQ(back, text) << name;
        }
      }
    }
  }
  std::println("{}: {} encodings ({} tokens) equal to the reference's", config.name, compared,
               tokens);
}

INSTANTIATE_TEST_SUITE_P(Models, Agreement, testing::ValuesIn(kConfigs),
                         [](const testing::TestParamInfo<Config>& info) {
                           std::string name(info.param.name);
                           std::ranges::replace(name, '-', '_');
                           std::ranges::replace(name, '.', '_');
                           return name;
                         });

// Leftmost-longest special-token matching (Hugging Face's) equals
// llama.cpp's passes, longest token first, unless some special token a ends
// with the start of another b that is at least as long: then text holding a
// overlapping b can split differently. Check no vocabulary has such a pair.
// Homogeneous whitespace runs have overlapping suffixes but both matchers
// consume their longest run from the left; Gemma2/3 declare those as added tokens.
// (DeepSeek's "<｜/table>｜" does end with the start of "｜DSML｜", which is
// shorter: both references take "<｜/table>｜"; the corpus holds that case.)
TEST(Vocabularies, SpecialTokensCannotSplitDifferently) {
  std::size_t checked = 0;
  for (const Config& config : kConfigs) {
    const Loaded loaded = Load(config.name, config.kind);
    if (loaded.tokenizer == nullptr) {
      continue;
    }
    const tok::Tokenizer& t = *loaded.tokenizer;
    std::vector<std::string_view> special;
    for (std::size_t id = 0; id < t.size(); ++id) {
      const auto k = t.Kind(static_cast<TokenId>(id));
      if (k == tok::TokenKind::kControl || k == tok::TokenKind::kUserDefined) {
        special.push_back(t.Text(static_cast<TokenId>(id)));
      }
    }
    for (const std::string_view a : special) {
      for (const std::string_view b : special) {
        if (a == b || b.size() < a.size()) {
          continue;
        }
        const auto uniform = [](std::string_view text) {
          return text.find_first_not_of(text.front()) == std::string_view::npos;
        };
        if (a.front() == b.front() && uniform(a) && uniform(b)) {
          continue;
        }
        for (std::size_t start = a.find(b.front(), 1); start != std::string_view::npos;
             start = a.find(b.front(), start + 1)) {
          const auto n = a.size() - start;
          EXPECT_FALSE(a.substr(start) == b.substr(0, n)) << config.name << ": " << a << " / " << b;
        }
      }
    }
    ++checked;
  }
  if (checked == 0) {
    GTEST_SKIP() << "no model files under " << ModelsDir();
  }
}

// The 0731 GGUF replaced revision e3aa0d6a on the Sparks: its tokenizer is
// the same (only the chat template changed).
TEST(Vocabularies, DeepSeek0731TokenizerEqualsE3aa0d6a) {
  const Loaded now = Load("deepseek-v4-0731-gguf", "gguf");
  const auto before =
      ReadFile(ModelsDir() +
               "/reference-models/D/UD-Q2_K_XL/DeepSeek-V4-Flash-UD-Q2_K_XL-00001-of-00003.gguf");
  const auto current = ReadFile(now.path);
  if (!before || !current) {
    GTEST_SKIP() << "no DeepSeek GGUFs under " << ModelsDir();
  }
  const auto a = tok::ReadGgufTokenizer(std::as_bytes(std::span(*current)));
  const auto b = tok::ReadGgufTokenizer(std::as_bytes(std::span(*before)));
  ASSERT_TRUE(a.has_value() && b.has_value());
  EXPECT_EQ(a->spec.tokens, b->spec.tokens);
  EXPECT_EQ(a->spec.kinds, b->spec.kinds);
  EXPECT_EQ(a->spec.merges, b->spec.merges);
  EXPECT_EQ(a->pre, b->pre);
  EXPECT_EQ(a->spec.bos, b->spec.bos);
  EXPECT_EQ(a->spec.eos, b->spec.eos);
  EXPECT_NE(a->chat_template, b->chat_template);
  EXPECT_EQ(Sha256(b->chat_template),
            "d05566ebe26667ec54f4ef7a3dbc114ce2e00aefc40e0c62286374eee0e22080");
}

// The community IQ2_XXS GGUF (antirez/deepseek-v4-gguf@f71f23d5, artifact
// cd39d504…) as its imported artifact keeps its metadata: the 0731 GGUF's
// tokenizer with another chat template, chat-v2.
constexpr std::string_view kCommunityKv =
    "m3-artifacts/cd39d504dc2dbfe911a4a521fa8efc8053dc3e80e99738a9b25fa6b70c97a1ac/meta/"
    "DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.kv.gguf";

TEST(Vocabularies, DeepSeekCommunityTokenizerEquals0731) {
  const Loaded now = Load("deepseek-v4-0731-gguf", "gguf");
  const auto community = ReadFile(ModelsDir() + "/" + std::string(kCommunityKv));
  const auto current = ReadFile(now.path);
  if (!community || !current) {
    GTEST_SKIP() << "no community artifact or 0731 GGUF under " << ModelsDir();
  }
  const auto a = tok::ReadGgufTokenizer(std::as_bytes(std::span(*current)));
  const auto b = tok::ReadGgufTokenizer(std::as_bytes(std::span(*community)));
  ASSERT_TRUE(a.has_value() && b.has_value());
  EXPECT_EQ(a->spec.tokens, b->spec.tokens);
  EXPECT_EQ(a->spec.kinds, b->spec.kinds);
  EXPECT_EQ(a->spec.merges, b->spec.merges);
  EXPECT_EQ(a->pre, b->pre);
  EXPECT_EQ(a->spec.bos, b->spec.bos);
  EXPECT_EQ(a->spec.eos, b->spec.eos);
  EXPECT_EQ(Sha256(b->chat_template),
            "872492071c22c8d2025238120309ffbddddb666b49f4433f55c19b69bf51af27");
}

// --- chat templates, as token IDs ---------------------------------------------------

// Renders each case and encodes it with EncodeMarked; `ids_key` names the
// reference IDs for this tokenizer.
void CheckChatTokens(const tok::Tokenizer& t, std::string_view fixture_name,
                     std::string_view ids_key) {
  const json::Document fixture = LoadJson("chat/" + std::string(fixture_name) + ".json");
  const json::Value root = fixture.root();
  const chat::Template* tmpl = chat::FindTemplate(Get(root, "template_sha256").string());
  ASSERT_NE(tmpl, nullptr);
  const auto stops = chat::StopTokens(tmpl->stop, t);
  ASSERT_TRUE(stops.has_value()) << stops.error().ToString();
  const json::Value cases = Get(root, "cases");
  std::size_t compared = 0;
  for (std::size_t i = 0; i < cases.size(); ++i) {
    const json::Value c = cases.at(i);
    if (c.find("error")) {
      continue;
    }
    std::deque<json::Document> arguments;
    const auto rendered = tmpl->render(ConversationFrom(c, &arguments));
    ASSERT_TRUE(rendered.has_value());
    std::vector<TokenId> ids;
    const auto r = t.EncodeMarked(rendered->text, rendered->specials, {}, ids);
    ASSERT_TRUE(r.has_value()) << r.error().ToString();
    EXPECT_EQ(ids, Ids(Get(c, ids_key))) << fixture_name << " " << Get(c, "name").string();
    ++compared;
  }
  EXPECT_GT(compared, 10U);
}

TEST(ChatTokens, DeepSeekV4OnTheGguf) {
  const Loaded loaded = Load("deepseek-v4-0731-gguf", "gguf");
  if (!loaded.skip.empty()) {
    GTEST_SKIP() << loaded.skip;
  }
  ASSERT_NE(loaded.tokenizer, nullptr);
  // The GGUF's own template is the one whose hash selects the renderer.
  EXPECT_EQ(Sha256(loaded.chat_template),
            "e643c31fcec17f342f72296e02c46d35846bf4c70f6a0271f23bad73fd4eb645");
  CheckChatTokens(*loaded.tokenizer, "deepseek-v4-0731", "gguf_ids");
}

// chat-v2's fixture against llama.cpp on the community GGUF itself, here
// through the tokenizer its artifact keeps.
TEST(ChatTokens, DeepSeekV4ChatV2OnTheCommunityGguf) {
  const auto bytes = ReadFile(ModelsDir() + "/" + std::string(kCommunityKv));
  if (!bytes) {
    GTEST_SKIP() << "no community artifact under " << ModelsDir();
  }
  auto g = tok::ReadGgufTokenizer(std::as_bytes(std::span(*bytes)));
  ASSERT_TRUE(g.has_value()) << g.error().ToString();
  EXPECT_EQ(Sha256(g->chat_template),
            "872492071c22c8d2025238120309ffbddddb666b49f4433f55c19b69bf51af27");
  auto t = tok::Tokenizer::Create(std::move(g->spec));
  ASSERT_TRUE(t.has_value()) << t.error().ToString();
  CheckChatTokens(*t, "deepseek-v4-chat-v2", "gguf_ids");
}

TEST(ChatTokens, DeepSeekV4ChatV2OnItsTokenizerJson) {
  const Loaded loaded = Load("deepseek-v4-0731-hf", "hf");
  if (!loaded.skip.empty()) {
    GTEST_SKIP() << loaded.skip;
  }
  ASSERT_NE(loaded.tokenizer, nullptr);
  CheckChatTokens(*loaded.tokenizer, "deepseek-v4-chat-v2", "hf_ids");
}

TEST(ChatTokens, DeepSeekV4OnItsTokenizerJson) {
  const Loaded loaded = Load("deepseek-v4-0731-hf", "hf");
  if (!loaded.skip.empty()) {
    GTEST_SKIP() << loaded.skip;
  }
  ASSERT_NE(loaded.tokenizer, nullptr);
  CheckChatTokens(*loaded.tokenizer, "deepseek-v4-0731", "hf_ids");
}

TEST(ChatTokens, Qwen38OnTheNvfp4TokenizerJson) {
  const Loaded loaded = Load("qwen3.8-nvfp4", "hf");
  if (!loaded.skip.empty()) {
    GTEST_SKIP() << loaded.skip;
  }
  ASSERT_NE(loaded.tokenizer, nullptr);
  // The checkpoint's template is the one whose hash selects the renderer.
  const std::string dir = loaded.path.substr(0, loaded.path.rfind('/'));
  const auto template_bytes = ReadFile(dir + "/chat_template.jinja");
  ASSERT_TRUE(template_bytes.has_value());
  EXPECT_EQ(Sha256(template_bytes.value_or("")),
            "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041");
  CheckChatTokens(*loaded.tokenizer, "qwen3.8", "hf_ids");
}

TEST(ChatTokens, QwenImagePrompt) {
  const Loaded loaded = Load("qwen-image-2.1", "hf");
  if (!loaded.skip.empty()) {
    GTEST_SKIP() << loaded.skip;
  }
  ASSERT_NE(loaded.tokenizer, nullptr);
  const json::Document image = LoadJson("chat/qwen-image-2.1.json");
  const json::Value cases = Get(image.root(), "cases");
  const std::int64_t drop = Int(Get(image.root(), "drop_tokens"));
  ASSERT_GT(cases.size(), 3U);
  for (std::size_t i = 0; i < cases.size(); ++i) {
    const json::Value c = cases.at(i);
    const auto prompt = chat::RenderQwenImagePrompt(Get(c, "prompt").string(), *loaded.tokenizer);
    ASSERT_TRUE(prompt.has_value());
    EXPECT_EQ(prompt->rendered.text, Get(c, "text").string());
    EXPECT_EQ(static_cast<std::int64_t>(prompt->drop_tokens), drop);
    std::vector<TokenId> ids;
    ASSERT_TRUE(
        loaded.tokenizer->EncodeMarked(prompt->rendered.text, prompt->rendered.specials, {}, ids)
            .has_value());
    EXPECT_EQ(ids, Ids(Get(c, "hf_ids"))) << Get(c, "name").string();
  }
}

// --- NFC conformance -------------------------------------------------------------

std::vector<char32_t> CodePoints(std::string_view field) {
  std::vector<char32_t> out;
  std::size_t at = 0;
  while (at < field.size()) {
    while (at < field.size() && field[at] == ' ') {
      ++at;
    }
    const std::size_t end = std::min(field.find(' ', at), field.size());
    if (end > at) {
      std::uint32_t value = 0;
      std::from_chars(field.data() + at, field.data() + end, value, 16);
      out.push_back(value);
    }
    at = end;
  }
  return out;
}

// UCD 15.1.0's NormalizationTest.txt, NFC columns: c2 == NFC(c1) == NFC(c2)
// == NFC(c3) and c4 == NFC(c4) == NFC(c5); every code point not in part 1
// is its own NFC.
TEST(Nfc, ConformsToNormalizationTest) {
  const std::string path = ModelsDir() + "/ucd/15.1.0/NormalizationTest.txt";
  const auto file = ReadFile(path);
  if (!file) {
    GTEST_SKIP() << "no " << path;
  }
  const std::string& bytes = *file;
  ASSERT_EQ(Sha256(bytes), "871238e37e3be0696ec2bd0891119a041b052da1a84485eda05a5438724b223e");
  std::size_t lines = 0;
  bool part1 = false;
  std::vector<bool> listed(0x110000, false);
  std::string_view rest = bytes;
  while (!rest.empty()) {
    const std::size_t nl = std::min(rest.find('\n'), rest.size());
    std::string_view line = rest.substr(0, nl);
    rest.remove_prefix(std::min(nl + 1, rest.size()));
    if (line.starts_with("@Part")) {
      part1 = line.starts_with("@Part1");
      continue;
    }
    line = line.substr(0, line.find('#'));
    if (line.empty()) {
      continue;
    }
    std::vector<std::vector<char32_t>> c;
    for (int k = 0; k < 5; ++k) {
      const std::size_t semi = line.find(';');
      c.push_back(CodePoints(line.substr(0, semi)));
      line.remove_prefix(std::min(semi + 1, line.size()));
    }
    if (part1 && c[0].size() == 1) {
      listed[c[0][0]] = true;
    }
    for (const std::size_t k : {0U, 1U, 2U}) {
      std::vector<char32_t> v = c[k];
      uni::ToNfc(v);
      EXPECT_EQ(v, c[1]) << "line " << lines << " column " << k + 1;
    }
    for (const std::size_t k : {3U, 4U}) {
      std::vector<char32_t> v = c[k];
      uni::ToNfc(v);
      EXPECT_EQ(v, c[3]) << "line " << lines << " column " << k + 1;
    }
    ++lines;
  }
  EXPECT_GT(lines, 19000U);
  std::size_t singles = 0;
  for (char32_t cp = 0; cp < 0x110000; ++cp) {
    if (listed[cp] || (cp >= 0xD800 && cp <= 0xDFFF)) {
      continue;
    }
    std::vector<char32_t> v = {cp};
    uni::ToNfc(v);
    ASSERT_EQ(v, std::vector<char32_t>{cp}) << static_cast<std::uint32_t>(cp);
    ++singles;
  }
  EXPECT_GT(singles, 1000000U);
}

}  // namespace
