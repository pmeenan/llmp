// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Synthetic source construction only: no checkpoint, generation or timing.
#include <gtest/gtest.h>
#include <unistd.h>

#include <filesystem>
#include <memory>
#include <string>

#include "artifact_writer.h"
#include "expected_error.h"
#include "runtime/component_build.h"
#include "runtime/memory_guard.h"
#include "runtime/serving.h"
#include "spec/stored.h"

namespace {
namespace rt = llmp::runtime;
namespace sp = llmp::spec;
namespace en = llmp::engine;
namespace fs = std::filesystem;
using llmp::test_support::Failed;

class ComponentStartupGpu : public testing::Test {
 protected:
  fs::path store;
  llmp::test::V1Plan plan;
  llmp::config::ModelEntry entry;
  rt::ModelSettings settings;
  void SetUp() override {
    store = fs::path(testing::TempDir()) /
            ("llmp-component-startup-" + std::to_string(::getpid()) + "-" +
             testing::UnitTest::GetInstance()->current_test_info()->name());
    fs::remove_all(store);
    sp::Stage stage{
        .name = "target", .role = sp::StageRole::kTarget, .artifact = 0, .context = 256};
    stage.stream.width = 32;
    stage.embedding = sp::ComponentRef{.kind = "embedding.token",
                                       .version = 1,
                                       .params = {{"scale", 1.0}},
                                       .weights = {{.role = "table", .tensor = "table"}}};
    stage.layers = {{{sp::ComponentRef{.kind = "attention.dense",
                                       .version = 1,
                                       .params = {{"placement", std::string("pre")},
                                                  {"eps", 0.00001},
                                                  {"heads", std::int64_t{1}},
                                                  {"kv_heads", std::int64_t{1}},
                                                  {"key_dim", std::int64_t{32}},
                                                  {"value_dim", std::int64_t{32}},
                                                  {"scale", 0.1767766952966369},
                                                  {"window", std::int64_t{0}},
                                                  {"rotation", std::string("neox")},
                                                  {"rope_dims", std::int64_t{32}},
                                                  {"rope_base", 10000.0},
                                                  {"rope_scale", 1.0}},
                                       .weights = {{.role = "norm", .tensor = "norm"},
                                                   {.role = "q", .tensor = "table"},
                                                   {.role = "k", .tensor = "table"},
                                                   {.role = "v", .tensor = "table"},
                                                   {.role = "output", .tensor = "table"}}}}}};
    stage.heads.output = sp::ComponentRef{.kind = "head.linear",
                                          .version = 1,
                                          .params = {{"rows", std::int64_t{32}}},
                                          .weights = {{.role = "weight", .tensor = "head"}}};
    sp::ModelSpec model{.pipeline = sp::PipelineKind::kLlm, .stages = {stage}};
    sp::Canonicalize(model);
    plan.architecture = "qwen2";
    plan.spec = sp::ToJson(model);
    plan.groups = {{.kind = llmp::artifact::GroupKind::kGlobal,
                    .resources = {{.name = "table",
                                   .aliases = {"head"},
                                   .format = {.encoding = llmp::format::Encoding::kF32,
                                              .layout = llmp::format::Layout::kDense,
                                              .shape = {.ne = {32, 32, 1, 1}, .rank = 2}}},
                                  {.name = "norm",
                                   .format = {.encoding = llmp::format::Encoding::kF32,
                                              .layout = llmp::format::Layout::kDense,
                                              .shape = {.ne = {32, 1, 1, 1}, .rank = 1}}}}}};
    const auto path = llmp::test::Publish(store, llmp::test::Render(plan));
    entry.name = "prepared";
    entry.artifact = path.filename().string();
    settings.name = entry.name;
    settings.architecture = "qwen2";
    settings.container = plan.container;
    settings.context.value = 64;
    settings.max_slots.value = 1;
    settings.prefill_chunk.value = 32;
    settings.wave_read_align.value = 256;
    settings.kv_cache_encoding.value = "f16";
  }
  void TearDown() override { fs::remove_all(store); }
};

TEST_F(ComponentStartupGpu, FailedPreparationRetriesAndFactoryCopiesReusePaidArtifacts) {
  en::PagedNode node({.compute_streams = 1, .slots = 1, .slot_bytes = en::kPagedExtent});
  ASSERT_TRUE(node.Open());
  const auto baseline = node.catalog().OccupancyOf(node.domain()).Total().value();
  const auto root = store / *entry.artifact;
  fs::rename(root, store / "temporarily-unavailable");
  std::shared_ptr<const llmp::artifact::Artifact> retained_view;
  std::weak_ptr<const llmp::artifact::Artifact> lifetime;
  {
    auto factory = rt::ComponentLlmBuild(node, store, entry, 0);
    EXPECT_FALSE(factory(settings));  // transient I/O failure must not be memoized
    EXPECT_EQ(node.catalog().OccupancyOf(node.domain()).Total().value(), baseline);
    fs::rename(store / "temporarily-unavailable", root);
    auto invalid = settings;
    invalid.context.value = 257;
    const auto failed_build = factory(invalid);  // admitted preparation, then envelope refusal
    ASSERT_FALSE(failed_build);
    EXPECT_NE(failed_build.error().find("resolved context exceeds"), std::string::npos);
    const auto retained = node.catalog().OccupancyOf(node.domain()).Total().value();
    EXPECT_GT(retained, baseline);
    for (int retry = 0; retry < 3; ++retry) {
      EXPECT_FALSE(factory(invalid));
      EXPECT_EQ(node.catalog().OccupancyOf(node.domain()).Total().value(), retained);
    }
    // Force every subsequent Artifact::Open to fail while retained descriptors
    // and payload shard paths stay valid. A factory copy must share preparation.
    ASSERT_TRUE(fs::remove(root / "manifest.json"));
    auto another = factory;
    auto first = factory(settings);
    ASSERT_TRUE(first) << Failed(first).value_or("");
    const auto one_live = node.catalog().OccupancyOf(node.domain()).Total().value();
    auto second = another(settings);
    ASSERT_TRUE(second) << Failed(second).value_or("");
    EXPECT_GT(node.catalog().OccupancyOf(node.domain()).Total().value(), one_live);
    EXPECT_NE(first->source.get(), second->source.get());
    EXPECT_NE(first->keep.get(), second->keep.get());
    retained_view = first->target_artifact;
    lifetime = retained_view;
    ASSERT_TRUE(first->target_artifact);
    EXPECT_EQ(first->target_artifact.get(), second->target_artifact.get());
    EXPECT_EQ(&first->config.weights.front().weights->artifact(), first->target_artifact.get());
    EXPECT_EQ(&second->config.weights.front().weights->artifact(), first->target_artifact.get());
    const auto two_live = node.catalog().OccupancyOf(node.domain()).Total().value();
    first = std::unexpected(std::string("released fixture build"));
    auto replacement = factory(settings);
    ASSERT_TRUE(replacement) << Failed(replacement).value_or("");
    EXPECT_EQ(node.catalog().OccupancyOf(node.domain()).Total().value(), two_live);
    auto wrong_kv = settings;
    wrong_kv.kv_cache_encoding.value = "unsupported";
    EXPECT_FALSE(another(wrong_kv));  // preparation supplies no backend/KV capability bypass
    auto wrong_bound = settings;
    wrong_bound.bound_spec = llmp::base::Sha256Digest{};
    EXPECT_FALSE(another(wrong_bound));
    factory = {};
    another = {};
    EXPECT_EQ(second->target_artifact->id(), *entry.artifact);  // builds retain the paid owner
  }
  // The node keeps backend/binding high-water reservations. Isolate the paid
  // prepared-artifact owner: all build/factory owners are gone, and releasing
  // this final view must refund its distinct StartupArtifactPayment.
  EXPECT_FALSE(lifetime.expired());
  const auto last_view_held = node.catalog().OccupancyOf(node.domain()).Total().value();
  retained_view.reset();
  EXPECT_TRUE(lifetime.expired());
  EXPECT_LT(node.catalog().OccupancyOf(node.domain()).Total().value(), last_view_held);
  ASSERT_TRUE(node.TearDown({}));
}
TEST_F(ComponentStartupGpu, FailedSetupRetriesWithTheSamePreparedOwner) {
  en::PagedNode node({.compute_streams = 1, .slots = 1, .slot_bytes = en::kPagedExtent});
  ASSERT_TRUE(node.Open());
  auto factory = rt::ComponentLlmBuild(node, store, entry, 0);
  std::weak_ptr<const llmp::artifact::Artifact> prepared;
  bool first = true;
  auto build =
      [&](const rt::ModelSettings& resolved) -> std::expected<rt::LlmBuildParts, std::string> {
    auto parts = factory(resolved);
    if (!parts) return std::unexpected(parts.error());
    if (first) {
      prepared = parts->target_artifact;
      parts->logit_width = 0;  // fail Setup after the real factory succeeded
      first = false;
    } else {
      EXPECT_EQ(prepared.lock().get(), parts->target_artifact.get());
    }
    return parts;
  };
  auto without_chat = entry;
  without_chat.artifact.reset();  // synthetic fixture has no tokenizer
  llmp::config::RuntimeRoles roles;
  roles.installed = store;
  rt::Llm model(node, without_chat, settings, roles, 0, build);
  auto failed = model.Setup();
  ASSERT_FALSE(failed);
  EXPECT_NE(failed.error().find("its head has no logits"), std::string::npos);
  ASSERT_TRUE(fs::remove(store / *entry.artifact / "manifest.json"));
  auto retried = model.Setup();
  EXPECT_TRUE(retried) << Failed(retried).value_or("");
  // Retirement is explicit even when a later Setup step refuses construction.
  ASSERT_TRUE(node.TearDown({}));
}

TEST_F(ComponentStartupGpu, LazyParserFundingRefusesBeforeOpenAndCanRetry) {
  en::PagedNode node({.compute_streams = 1, .slots = 1, .slot_bytes = en::kPagedExtent});
  ASSERT_TRUE(node.Open());
  const auto baseline = node.catalog().OccupancyOf(node.domain()).Total().value();
  auto bytes = rt::StartupBootstrapBytes(1);
  ASSERT_TRUE(bytes);
  const auto cap = baseline + *bytes + 2 * en::kPagedExtent;
  auto blocker = rt::StartupArtifactPayment::Admit(
      node.catalog(), node.domain(), 4 * en::kPagedExtent, cap, cap + rt::kUncountedMargin);
  ASSERT_TRUE(blocker);
  rt::ComponentBuildOptions options;
  options.automatic_startup_cap = cap;
  auto factory = rt::ComponentLlmBuild(node, store, entry, 0, std::move(options));
  fs::rename(store / *entry.artifact, store / "temporarily-unavailable");
  auto refused = factory(settings);
  ASSERT_FALSE(refused);
  EXPECT_NE(refused.error().find("startup artifacts exceed"), std::string::npos);
  blocker->reset();
  EXPECT_EQ(node.catalog().OccupancyOf(node.domain()).Total().value(), baseline);
  auto retry = factory(settings);
  ASSERT_FALSE(retry);
  // Funding now succeeds and reaches the real missing artifact; neither the
  // funding failure nor the new I/O failure is memoized or leaks its loan.
  EXPECT_NE(retry.error().find(*entry.artifact), std::string::npos);
  EXPECT_EQ(retry.error().find("startup artifacts exceed"), std::string::npos);
  EXPECT_EQ(node.catalog().OccupancyOf(node.domain()).Total().value(), baseline);
  fs::rename(store / "temporarily-unavailable", store / *entry.artifact);
  ASSERT_TRUE(node.TearDown({}));
}

TEST_F(ComponentStartupGpu, AutomaticConstructionReusesFailedAndReleasedLiveCharges) {
  en::PagedNode node({.compute_streams = 1, .slots = 1, .slot_bytes = en::kPagedExtent});
  ASSERT_TRUE(node.Open());
  auto parser = rt::StartupBootstrapBytes(1);
  ASSERT_TRUE(parser);
  rt::ComponentBuildOptions options;
  options.automatic_startup_cap = *parser + (std::uint64_t{2} << 30U);
  options.shared_compiler = std::make_shared<en::SharedCompilerAllowance>();
  auto factory = rt::ComponentLlmBuild(node, store, entry, 0, std::move(options));
  auto wrong_bound = settings;
  wrong_bound.bound_spec = llmp::base::Sha256Digest{};
  auto failed = factory(wrong_bound);  // fails after automatic construction admission
  ASSERT_FALSE(failed);
  EXPECT_NE(failed.error().find("resolved calibration spec differs"), std::string::npos);
  const auto warmed = node.catalog().OccupancyOf(node.domain()).Total().value();
  for (int retry = 0; retry < 3; ++retry) {
    EXPECT_FALSE(factory(wrong_bound));
    EXPECT_EQ(node.catalog().OccupancyOf(node.domain()).Total().value(), warmed);
  }
  auto first = factory(settings);
  ASSERT_TRUE(first) << Failed(first).value_or("");
  const auto one_live = node.catalog().OccupancyOf(node.domain()).Total().value();
  auto copy = factory;
  auto second = copy(settings);
  ASSERT_TRUE(second) << Failed(second).value_or("");
  const auto two_live = node.catalog().OccupancyOf(node.domain()).Total().value();
  EXPECT_GT(two_live, one_live);
  EXPECT_GT(first->startup_host_bytes, 0U);
  EXPECT_EQ(first->startup_host_bytes, second->startup_host_bytes);
  const auto footprint = first->startup_host_bytes;
  first = std::unexpected(std::string("released fixture build"));
  auto replacement = copy(settings);
  ASSERT_TRUE(replacement) << Failed(replacement).value_or("");
  EXPECT_EQ(replacement->startup_host_bytes, footprint);
  EXPECT_EQ(node.catalog().OccupancyOf(node.domain()).Total().value(), two_live);
  ASSERT_TRUE(node.TearDown({}));
}

TEST_F(ComponentStartupGpu, LastShardCallbackRefundsBeforeTheNodeCatalogDies) {
  auto node = std::make_unique<en::PagedNode>(
      en::NodeSettings{.compute_streams = 1, .slots = 1, .slot_bytes = en::kPagedExtent});
  ASSERT_TRUE(node->Open());
  constexpr std::uint64_t cap = std::uint64_t{1} << 30U;
  auto payment = rt::StartupArtifactPayment::Admit(node->catalog(), node->domain(), 4096, cap,
                                                   cap + rt::kUncountedMargin);
  ASSERT_TRUE(payment) << Failed(payment).value_or("");
  std::shared_ptr<rt::StartupArtifactPayment> shared(std::move(*payment));
  std::weak_ptr<rt::StartupArtifactPayment> last = shared;
  ASSERT_TRUE(node->Start(llmp::base::Bytes(cap)));
  auto opened = llmp::artifact::Artifact::Open(store / *entry.artifact);
  ASSERT_TRUE(opened);
  auto artifact = std::make_shared<const llmp::artifact::Artifact>(std::move(*opened));
  ASSERT_TRUE(
      node->AddWeightFile(0, [owner = shared, artifact]() -> std::expected<int, std::string> {
        auto shard = artifact->OpenShardForDirectRead(0);
        if (!shard) return std::unexpected(shard.error().ToString());
        return shard->release();
      }));
  artifact.reset();
  node->Run();
  shared.reset();
  EXPECT_FALSE(last.expired());
  const auto retired = node->TearDown({});  // stop/drain first, independently of destruction
  ASSERT_TRUE(retired) << Failed(retired).value_or("");
  EXPECT_FALSE(last.expired());  // registered source callback still owns its metadata
  node.reset();                  // actual StartupArtifactPayment refund checks its catalog extent
  EXPECT_TRUE(last.expired());
}
}  // namespace
