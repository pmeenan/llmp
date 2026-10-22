// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <unistd.h>

#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "artifact_writer.h"
#include "chat_tokenizer.h"
#include "expected_error.h"
#include "llm_fixture.h"
#include "runtime/memory_guard.h"

namespace {
namespace rt = llmp::runtime;
namespace en = llmp::engine;
namespace fs = std::filesystem;
using Access = rt::LlmTestAccess;
using llmp::test_support::Failed;
constexpr std::uint64_t kCap = std::uint64_t{32} << 30U;
constexpr std::uint64_t kAvailable = std::uint64_t{128} << 30U;
constexpr std::string_view kTemplate = "{{ messages[0].content }}";

class ChatStartup : public testing::Test {
 protected:
  void SetUp() override {
    path = fs::path(testing::TempDir()) /
           ("llmp-chat-startup-" + std::to_string(::getpid()) + "-" +
            testing::UnitTest::GetInstance()->current_test_info()->name());
    fs::remove_all(path);
    fs::create_directories(path);
    entry.name = "chat";
    entry.chat_template = path / "template.jinja";
    settings.name = entry.name;
    settings.context.value = 64;
    settings.max_slots.value = 1;
    Write(kTemplate);
    node = std::make_unique<en::PagedNode>(
        en::NodeSettings{.compute_streams = 1, .slots = 1, .slot_bytes = en::kPagedExtent});
    ASSERT_TRUE(node->Open());
  }
  void TearDown() override {
    model.reset();
    if (node) EXPECT_TRUE(node->TearDown({}));
    node.reset();
    fs::remove_all(path);
  }
  void Write(std::string_view text) { std::ofstream(*entry.chat_template) << text; }
  llmp::artifact::Artifact Artifact(std::string_view piece = "a", bool keep_tokenizer = true) {
    llmp::test::V1Plan plan;
    plan.groups = {{.kind = llmp::artifact::GroupKind::kGlobal,
                    .resources = {{.name = "weight",
                                   .format = {.encoding = llmp::format::Encoding::kBF16,
                                              .layout = llmp::format::Layout::kDense,
                                              .shape = {.ne = {1, 1, 1, 1}, .rank = 1}}}}}};
    if (keep_tokenizer)
      plan.meta = {{"tokenizer.model", llmp::test_support::chat_tokenizer::Model(piece)},
                   {"tokenizer_config.json",
                    R"({"bos_token":"<bos>","eos_token":"<eos>","add_bos_token":true})"}};
    auto opened =
        llmp::artifact::Artifact::Open(llmp::test::Publish(path, llmp::test::Render(plan)));
    EXPECT_TRUE(opened) << Failed(opened, &llmp::artifact::Error::ToString).value_or("");
    return std::move(*opened);
  }
  void Make(rt::StartupReaderLoan loan = {}) {
    model = std::make_unique<rt::Llm>(*node, entry, settings, llmp::config::RuntimeRoles{}, 0,
                                      rt::LlmBuild{}, std::move(loan));
  }
  rt::Status Use(const llmp::artifact::Artifact& artifact, std::uint64_t cap = kCap) {
    return Access::UseChatAssets(*model, artifact, entry, cap);
  }
  std::uint64_t Occupied() const {
    return node->catalog().OccupancyOf(node->domain()).Total().value();
  }
  std::unique_ptr<rt::StartupArtifactPayment> Reader() {
    const auto bound = rt::ChatReaderBytes();
    EXPECT_TRUE(bound);
    auto paid = rt::StartupArtifactPayment::AdmitReader(node->catalog(), node->domain(), *bound,
                                                        kCap, kAvailable);
    EXPECT_TRUE(paid) << Failed(paid).value_or("");
    return std::move(*paid);
  }
  std::vector<std::int32_t> Encode(std::string_view text = "a") const {
    std::vector<std::int32_t> ids;
    EXPECT_TRUE(model->tokenizer().Encode(text, {}, ids));
    return ids;
  }
  std::string Render() const {
    llmp::chat::Conversation conversation;
    conversation.messages = {{.role = llmp::chat::Role::kUser, .content = "a"}};
    auto rendered = Access::ChatTemplate(*model).Render(conversation);
    EXPECT_TRUE(rendered);
    return rendered ? rendered->text : "";
  }
  fs::path path;
  llmp::config::ModelEntry entry;
  rt::ModelSettings settings;
  std::unique_ptr<en::PagedNode> node;
  std::unique_ptr<rt::Llm> model;
};

TEST_F(ChatStartup, FailedConstructionReusesOnePoolAndRepairPublishesRealAssets) {
  auto artifact = Artifact();
  Make();
  Write("{% if " + std::string(128, ' '));
  auto first = Use(artifact);
  ASSERT_FALSE(first);
  EXPECT_NE(first.error().find("interpreter does not accept"), std::string::npos);
  auto pool = Access::ChatPool(*model);
  ASSERT_TRUE(pool);
  EXPECT_GT(pool->capacity(), 0U);
  EXPECT_EQ(pool->used(), 0U);
  EXPECT_EQ(model->startup_host_bytes(), 0U);
  EXPECT_FALSE(Access::HasChat(*model));
  const auto warmed = Occupied();
  for (int retry = 0; retry < 3; ++retry) {
    EXPECT_FALSE(Use(artifact));
    EXPECT_EQ(Occupied(), warmed);
    EXPECT_EQ(pool->used(), 0U);
    EXPECT_EQ(Access::ChatPool(*model).get(), pool.get());
  }
  Write(kTemplate);
  ASSERT_TRUE(Use(artifact));
  EXPECT_EQ(Occupied(), warmed);
  EXPECT_EQ(Encode(), (std::vector<std::int32_t>{260}));
  EXPECT_EQ(Render(), "a");
  EXPECT_EQ(std::vector(model->StopTokens().begin(), model->StopTokens().end()),
            (std::vector<std::int32_t>{2}));
  EXPECT_EQ(model->startup_host_bytes(),
            pool->used() + en::ExecutionMetadataAllowance::OverheadBytes());
}

TEST_F(ChatStartup, FailedReplacementPreservesAllPublishedAssetsAndSettings) {
  auto old = Artifact();
  auto replacement = Artifact("longer");
  auto reader = Reader();
  Make(reader->reader_loan());
  entry.overrides["reasoning_start"] = std::string("<bos>");
  entry.overrides["reasoning_end"] = std::string("<eos>");
  ASSERT_TRUE(Use(old));
  const auto original = Encode();
  const auto rendered = Render();
  const auto hash = std::string(Access::ChatTemplate(*model).sha256());
  const auto stops = std::vector(model->StopTokens().begin(), model->StopTokens().end());
  const auto start = model->settings().reasoning_start;
  const auto end = model->settings().reasoning_end;
  const auto render_bytes = model->render_bytes();
  const auto longest = Access::LongestToken(*model);
  const auto live = model->startup_host_bytes();
  auto pool = Access::ChatPool(*model);
  const auto charged = pool->used();
  const auto preserved = [&] {
    EXPECT_EQ(Encode(), original);
    EXPECT_EQ(Render(), rendered);
    EXPECT_EQ(Access::ChatTemplate(*model).sha256(), hash);
    EXPECT_EQ(std::vector(model->StopTokens().begin(), model->StopTokens().end()), stops);
    EXPECT_EQ(model->settings().reasoning_start.value, start.value);
    EXPECT_EQ(model->settings().reasoning_start.basis, start.basis);
    EXPECT_EQ(model->settings().reasoning_start.source, start.source);
    EXPECT_EQ(model->settings().reasoning_end.value, end.value);
    EXPECT_EQ(model->settings().reasoning_end.basis, end.basis);
    EXPECT_EQ(model->settings().reasoning_end.source, end.source);
    EXPECT_EQ(model->think_start(), 1);
    EXPECT_EQ(model->think_end(), 2);
    EXPECT_EQ(model->render_bytes(), render_bytes);
    EXPECT_EQ(Access::LongestToken(*model), longest);
    EXPECT_EQ(model->startup_host_bytes(), live);
    EXPECT_EQ(pool->used(), charged);
  };
  entry.overrides["reasoning_start"] = std::string("<eos>");
  entry.overrides["reasoning_end"] = std::string("absent");
  auto reasoning_failure = Use(replacement);
  ASSERT_FALSE(reasoning_failure);
  EXPECT_NE(reasoning_failure.error().find("reasoning_end"), std::string::npos);
  preserved();
  entry.overrides["reasoning_end"] = std::string("<bos>");
  const char* data = std::getenv("LLMP_TEST_DATA");
  ASSERT_NE(data, nullptr);
  std::ifstream supported(fs::path(data) / "chat/templates/qwen3.8.jinja");
  ASSERT_TRUE(supported);
  Write(std::string(std::istreambuf_iterator<char>(supported), std::istreambuf_iterator<char>()));
  auto stop_failure = Use(replacement);
  ASSERT_FALSE(stop_failure);
  EXPECT_NE(stop_failure.error().find("a stop token is not in the vocabulary"), std::string::npos);
  preserved();
  Write("{% if " + std::string(128, ' '));
  auto template_failure = Use(replacement);
  ASSERT_FALSE(template_failure);
  EXPECT_NE(template_failure.error().find("interpreter does not accept"), std::string::npos);
  preserved();
  EXPECT_GT(pool->peak(), charged);
  Write(kTemplate);
  ASSERT_TRUE(Use(replacement));
  EXPECT_NE(Encode(), original);
  EXPECT_EQ(model->tokenizer().Find("longer"), 260);
  EXPECT_EQ(Encode(), (std::vector<std::int32_t>{100}));
  EXPECT_EQ(Render(), "a");
  EXPECT_EQ(model->think_start(), 2);
  EXPECT_EQ(model->think_end(), 1);
  const auto overlap_capacity = Occupied();
  ASSERT_TRUE(Use(replacement));
  EXPECT_EQ(Occupied(), overlap_capacity);
  EXPECT_GE(pool->peak(), 2 * pool->used());
  EXPECT_LT(model->startup_host_bytes(), pool->capacity());
  model.reset();  // the borrowed reader may retire only after this attempt
}

TEST_F(ChatStartup, OldAndNewMustFitTogetherAndIndependentBlockerCanBeReleased) {
  auto artifact = Artifact();
  auto reader = Reader();
  Make(reader->reader_loan());
  ASSERT_TRUE(Use(artifact));
  auto pool = Access::ChatPool(*model);
  const auto used = pool->used();
  const auto live = model->startup_host_bytes();
  const auto hash = std::string(Access::ChatTemplate(*model).sha256());
  // The same artifact fits by itself, but less than a second live charge is
  // left beside an independently owned payment. Never loosen this cap.
  auto blocker = rt::StartupArtifactPayment::Admit(
      node->catalog(), node->domain(), kCap - Occupied() - 2 * en::kPagedExtent, kCap, kAvailable);
  ASSERT_TRUE(blocker);
  auto refused = Use(artifact);
  ASSERT_FALSE(refused);
  EXPECT_NE(refused.error().find("chat asset construction exceeds"), std::string::npos);
  EXPECT_EQ(pool->used(), used);
  EXPECT_EQ(model->startup_host_bytes(), live);
  EXPECT_EQ(Access::ChatTemplate(*model).sha256(), hash);
  EXPECT_EQ(Encode(), (std::vector<std::int32_t>{260}));
  blocker->reset();
  ASSERT_TRUE(Use(artifact));
  EXPECT_EQ(pool->used(), used);
  EXPECT_EQ(pool->peak(), 2 * used);
  model.reset();
}

TEST_F(ChatStartup, BorrowedReaderProofAvoidsDuplicateChargeAndExpires) {
  auto artifact = Artifact("a", false);
  entry.tokenizer = path / "missing.model";
  const auto baseline = Occupied();
  auto reader = Reader();
  auto loan = reader->reader_loan();
  Make(loan);
  // The cap has room for the existing reader, but cannot pay a second one.
  auto missing = Use(artifact, Occupied());
  ASSERT_FALSE(missing);
  EXPECT_EQ(missing.error().find("startup artifacts exceed"), std::string::npos);
  EXPECT_NE(missing.error().find("missing.model"), std::string::npos);
  EXPECT_EQ(Occupied(), baseline + reader->bytes());
  model.reset();
  reader.reset();
  Make(loan);  // stale weak proof, even if a slot/generation is reused
  auto refused = Use(artifact, baseline + en::kPagedExtent);
  ASSERT_FALSE(refused);
  EXPECT_NE(refused.error().find("startup artifacts exceed"), std::string::npos);
  EXPECT_EQ(Occupied(), baseline);
  auto failed_read = Use(artifact);
  ASSERT_FALSE(failed_read);
  EXPECT_NE(failed_read.error().find("missing.model"), std::string::npos);
  EXPECT_EQ(Occupied(), baseline);  // local reader loan refunded on read error
  std::ofstream(*entry.tokenizer) << llmp::test_support::chat_tokenizer::Model();
  ASSERT_TRUE(Use(artifact));
  EXPECT_EQ(Encode(), (std::vector<std::int32_t>{260}));
}

TEST_F(ChatStartup, ForeignAndUndersizedReaderProofsCannotAuthorizeReads) {
  auto artifact = Artifact("a", false);
  entry.tokenizer = path / "missing.model";
  llmp::catalog::Catalog foreign;
  const auto domain = foreign.AddDomain("other-node");
  const auto bytes = rt::ChatReaderBytes();
  ASSERT_TRUE(bytes);
  auto foreign_reader =
      rt::StartupArtifactPayment::AdmitReader(foreign, domain, *bytes, kCap, kAvailable);
  ASSERT_TRUE(foreign_reader);
  auto small =
      rt::StartupArtifactPayment::AdmitReader(node->catalog(), node->domain(), 1, kCap, kAvailable);
  ASSERT_TRUE(small);
  for (const auto& loan : {(*foreign_reader)->reader_loan(), (*small)->reader_loan()}) {
    Make(loan);
    auto refused = Use(artifact, Occupied());
    ASSERT_FALSE(refused);
    EXPECT_NE(refused.error().find("startup artifacts exceed"), std::string::npos);
  }
  model.reset();
}

TEST_F(ChatStartup, DiagnosticCallerFundedModeRemainsTransactional) {
  auto artifact = Artifact();
  Make();
  ASSERT_TRUE(Use(artifact, 0));
  EXPECT_FALSE(Access::ChatPool(*model));
  EXPECT_EQ(model->startup_host_bytes(), 0U);
  Write("{% if");
  EXPECT_FALSE(Use(artifact, 0));
  EXPECT_EQ(Encode(), (std::vector<std::int32_t>{260}));
  EXPECT_EQ(Render(), "a");
}
}  // namespace
