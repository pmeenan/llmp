// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <filesystem>
#include <memory>
#include <tuple>
#include <vector>

#include "base/sha256.h"
#include "engine/dsv4_runner.h"
#include "engine/qwen38_runner.h"
#include "plain_token_serving_checks.h"
#include "runtime/serving.h"

namespace rt = jitllm::runtime;
namespace en = jitllm::engine;
namespace cfg = jitllm::config;

class PlainTokensServingGpu : public ::testing::TestWithParam<std::uint32_t> {
 protected:
  struct Lifetime {
    cfg::NodeConfig config;
    cfg::RuntimeRoles roles;
    rt::ServingOptions options;
    std::unique_ptr<rt::Server> server;
  };
  std::unique_ptr<Lifetime> life = std::make_unique<Lifetime>();
  rt::Llm* model = nullptr;
  std::filesystem::path scratch;
  void SetUp() override {
    std::string name = "/home/pmeenan/.cache/jitllm-plain-tokens-XXXXXX";
    ASSERT_NE(mkdtemp(name.data()), nullptr);
    scratch = name;
    const bool gguf = GetParam() == 1, deepseek = GetParam() == 2;
    life->roles.installed = gguf ? "/home/pmeenan/.local/share/jitllm/qgguf-artifacts"
                                 : "/home/pmeenan/.local/share/jitllm/m3-artifacts";
    life->roles.spill = scratch / "spill";
    life->roles.state = scratch / "state";
    for (const auto& path : {life->roles.spill, life->roles.state}) {
      ASSERT_TRUE(std::filesystem::create_directory(path));
      ASSERT_EQ(chmod(path.c_str(), 0700), 0);
    }
    cfg::ModelEntry entry;
    entry.name = "target";
    entry.artifact = deepseek ? "8a355bfb27c90e1150fbd7fa62ea6e63f6bf34fcca33934e52d22773f1508234"
                     : gguf   ? "5356b5b05fd93d06419cd842c4946e0df9d57ec816916109af5c724cadf25b78"
                              : "c4fb47a911207c11f935f932d05196dc1701aa0d886eac1b5e91934e554b5a93";
    if (!deepseek) {
      const std::filesystem::path checkpoint =
          "/home/pmeenan/.local/share/jitllm/models/Mia-AiLab/"
          "Qwen3.8-Flash-Next-NVFP4@925d7be6";
      entry.tokenizer = checkpoint / "tokenizer.json";
      entry.chat_template = checkpoint / "chat_template.jinja";
    }
    entry.overrides["context"] = std::int64_t{deepseek ? 8704 : 2048};
    entry.overrides["prefill_chunk"] = std::int64_t{deepseek ? 4096 : 512};
    entry.overrides["max_slots"] = std::int64_t{2};
    life->config.models.push_back(entry);
    life->options.plain = true;
    ASSERT_FALSE(life->options.diagnostic_plain_device_tokens.has_value());
    life->server = std::make_unique<rt::Server>(life->config, life->roles, life->options, stderr);
    const auto started = life->server->Start(true);
    ASSERT_TRUE(started) << (started ? "" : started.error());
    model = dynamic_cast<rt::Llm*>(life->server->Find("target"));
    ASSERT_NE(model, nullptr);
    rt::SwapParts parts;
    const auto activated = life->server->Activate(*model, parts);
    ASSERT_TRUE(activated) << (activated ? "" : activated.error());
  }
  void TearDown() override {
    if (life->server) {
      const auto retired = life->server->TearDown();
      EXPECT_TRUE(retired) << (retired ? "" : retired.error());
      if (!retired) {
        std::ignore = life.release();
        return;
      }
      life->server.reset();
    }
    std::filesystem::remove_all(scratch);
  }
  template <class Runner>
  std::expected<jitllm::base::Sha256Digest, std::string> StateHash(Runner& runner,
                                                                   std::uint32_t id) {
    auto slot = runner.request_slot(id);
    if (!slot) return std::unexpected(slot.error());
    auto& node = life->server->node();
    constexpr std::uint64_t capacity = 1ULL << 20U;
    std::vector<jitllm::catalog::ExtentId> staging;
    auto buffer = node.Pinned(capacity, 0, staging);
    if (!buffer) return std::unexpected(buffer.error());
    jitllm::base::Sha256 hash;
    std::uint64_t bytes = 0;
    for (const auto& range : (*slot)->used_state_ranges()) {
      bytes += range.bytes;
      for (std::uint64_t at = 0; at < range.bytes; at += capacity) {
        const en::LiveState::Range part{range.region, range.offset + at,
                                        std::min(capacity, range.bytes - at)};
        if (auto r = (*slot)->SaveUsedState(*buffer, std::span(&part, 1)); !r) {
          // An error is not completion proof; keep its host destination.
          node.KeepPinned(*buffer);
          return std::unexpected(r.error());
        }
        hash.Update(std::span(static_cast<const std::byte*>(*buffer), part.bytes));
      }
    }
    if (auto r = node.FreePinned(*buffer); !r) return std::unexpected(r.error());
    if (bytes == 0) return std::unexpected("complete initialized state required");
    return hash.Finish();
  }
};

TEST_P(PlainTokensServingGpu, ScalarJoinedMixedAndSpilledRowsMatchDeviceTokens) {
  const auto exercise = [&](auto& runner) {
    return jitllm::test::CheckPlainServing(
        *life->server, *model, runner.vocab(),
        [&](std::uint32_t id) { return StateHash(runner, id); },
        [&] { return runner.device_token_outputs(); });
  };
  const auto r = GetParam() == 2 ? exercise(dynamic_cast<en::Dsv4Runner&>(model->paged()))
                                 : exercise(dynamic_cast<en::Qwen38Runner&>(model->paged()));
  ASSERT_TRUE(r) << (r ? "" : r.error());
}

INSTANTIATE_TEST_SUITE_P(ApprovedTargets, PlainTokensServingGpu, ::testing::Values(0U, 1U, 2U));
