// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

#include "artifact/artifact.h"
#include "artifact_writer.h"
#include "config/node_config.h"
#include "expected_error.h"
#include "runtime/model_settings.h"
#include "tokenizer/tokenizer.h"

namespace {

// An independent small ModelProto writer: one unknown, BOS/EOS, byte
// fallback and two normal pieces. Omitting scores supplies protobuf's 0.
std::string Varint(std::uint64_t value) {
  std::string bytes;
  do {
    const auto low = value & 127U;
    value >>= 7U;
    bytes.push_back(static_cast<char>(low | (value != 0 ? 128U : 0U)));
  } while (value != 0);
  return bytes;
}
std::string Scalar(unsigned field, std::uint64_t value) {
  return Varint(std::uint64_t{field} << 3U) + Varint(value);
}
std::string Bytes(unsigned field, std::string_view bytes) {
  return Varint((std::uint64_t{field} << 3U) | 2U) + Varint(bytes.size()) + std::string(bytes);
}
std::string Model() {
  std::string model;
  const auto piece = [&](std::string_view text, unsigned kind) {
    model += Bytes(1, Bytes(1, text) + Scalar(3, kind));
  };
  piece("<unk>", 2);
  piece("<bos>", 3);
  piece("<eos>", 3);
  constexpr std::string_view hex = "0123456789ABCDEF";
  for (unsigned b = 0; b < 256; ++b) {
    std::string text = "<0x00>";
    text[3] = hex[b >> 4U];
    text[4] = hex[b & 15U];
    piece(text, 6);
  }
  piece("▁", 1);
  piece("a", 1);
  model += Bytes(2, Scalar(3, 2) + Scalar(35, 1) + Scalar(40, 0) + Scalar(41, 1) + Scalar(42, 2));
  model += Bytes(3, Bytes(1, "identity") + Scalar(3, 0) + Scalar(4, 0));
  return model;
}

class TokenizerAssets : public testing::Test {
 protected:
  void SetUp() override {
    const char* scratch = std::getenv("LLMP_TEST_SCRATCH");
    ASSERT_NE(scratch, nullptr);
    path_ = std::filesystem::path(scratch) / "tokenizer-assets" /
            testing::UnitTest::GetInstance()->current_test_info()->name();
    std::filesystem::remove_all(path_);
    std::filesystem::create_directories(path_);
  }
  void TearDown() override { std::filesystem::remove_all(path_); }
  std::filesystem::path path_;
};

TEST_F(TokenizerAssets, KeptSentencePieceLoadsAndAppliesTheKeptConfig) {
  llmp::test::V1Plan plan;
  plan.container = "safetensors";
  plan.groups = {{.kind = llmp::artifact::GroupKind::kGlobal,
                  .resources = {{.name = "weight",
                                 .format = {.encoding = llmp::format::Encoding::kBF16,
                                            .layout = llmp::format::Layout::kDense,
                                            .shape = {.ne = {1, 1, 1, 1}, .rank = 1}}}}}};
  plan.meta = {{"tokenizer.model", Model()},
               {"tokenizer_config.json",
                R"({"bos_token":"<bos>","eos_token":"<eos>","add_bos_token":true,)"
                R"("add_eos_token":false,"chat_template":"{{ bos_token }}{{ messages }}"})"}};
  const auto artifact =
      llmp::artifact::Artifact::Open(llmp::test::Publish(path_, llmp::test::Render(plan)));
  ASSERT_TRUE(artifact)
      << llmp::test_support::Failed(artifact, &llmp::artifact::Error::ToString).value_or("");
  auto assets = llmp::runtime::ReadChatAssets(*artifact, {});
  ASSERT_TRUE(assets) << llmp::test_support::Failed(assets).value_or("");
  EXPECT_EQ(assets->from, "tokenizer.model");
  EXPECT_TRUE(assets->chat_template);
  auto tokenizer = llmp::tokenizer::Tokenizer::Create(std::move(assets->spec));
  ASSERT_TRUE(tokenizer)
      << llmp::test_support::Failed(tokenizer, &llmp::tokenizer::Error::ToString).value_or("");
  std::vector<llmp::tokenizer::TokenId> ids;
  ASSERT_TRUE(tokenizer->Encode("a", {.add_bos_eos = true}, ids));
  EXPECT_EQ(ids, (std::vector<llmp::tokenizer::TokenId>{1, 260}));
}

TEST_F(TokenizerAssets, AKeptMalformedJsonDoesNotFallBackToTheModel) {
  llmp::test::V1Plan plan;
  plan.container = "safetensors";
  plan.groups = {{.kind = llmp::artifact::GroupKind::kGlobal,
                  .resources = {{.name = "weight",
                                 .format = {.encoding = llmp::format::Encoding::kBF16,
                                            .layout = llmp::format::Layout::kDense,
                                            .shape = {.ne = {1, 1, 1, 1}, .rank = 1}}}}}};
  plan.meta = {{"tokenizer.model", Model()}, {"tokenizer.json", "{ malformed"}};
  const auto artifact =
      llmp::artifact::Artifact::Open(llmp::test::Publish(path_, llmp::test::Render(plan)));
  ASSERT_TRUE(artifact)
      << llmp::test_support::Failed(artifact, &llmp::artifact::Error::ToString).value_or("");
  const auto assets = llmp::runtime::ReadChatAssets(*artifact, {});
  ASSERT_FALSE(assets);
  EXPECT_NE(assets.error().find("tokenizer.json:"), std::string::npos);
}

}  // namespace
