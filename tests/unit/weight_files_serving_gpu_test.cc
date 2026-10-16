// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Weight files across real switches (label `models`): Gemma 2 and Gemma 3
// on one server, their published v1 artifacts linked into one installed store,
// the process's descriptors on their shards counted from /proc/self/fd.
// Registered, neither model's shards are open; each activation opens the
// incoming model's and, once its swap is done, leaves the outgoing model's
// closed, so after each switch exactly the resident model's shards are
// open (engine/paged_node.h SetActive; docs/runtime-serving.md).

#include <gtest/gtest.h>
#include <sys/stat.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <string>
#include <system_error>
#include <tuple>
#include <vector>

#include "artifact/artifact.h"
#include "runtime/serving.h"
#include "v1_frozen.h"

namespace {

namespace rt = llmp::runtime;
namespace cfg = llmp::config;
namespace fs = std::filesystem;

// The descriptors of this process open on a file beneath `root`'s data/.
std::size_t OpenBeneath(const fs::path& root) {
  const std::string prefix = fs::canonical(root).string() + "/data/";
  std::size_t count = 0;
  for (const auto& entry : fs::directory_iterator("/proc/self/fd")) {
    std::error_code error;
    const fs::path target = fs::read_symlink(entry.path(), error);
    if (!error && target.string().starts_with(prefix)) {
      ++count;
    }
  }
  return count;
}

std::size_t Shards(const fs::path& root) {
  auto opened = llmp::artifact::Artifact::Open(root);
  return opened ? opened->shards().size() : 0;
}

TEST(WeightFilesServingGpu, OnlyTheResidentModelsShardsStayOpenAcrossSwitches) {
  const fs::path models = llmp::test_support::V1Store();
  const auto id2 = llmp::test_support::V1IdNamed("gemma2");
  const auto id3 = llmp::test_support::V1IdNamed("gemma3");
  if (!id2 || !id3) GTEST_FAIL() << "no published v1 Gemma 2 and Gemma 3 in " << models;
  const fs::path gemma2 = models / *id2;
  const fs::path gemma3 = models / *id3;
  std::error_code error;
  // NOLINTNEXTLINE(concurrency-mt-unsafe): test environment is immutable
  const char* base = std::getenv("LLMP_TEST_SCRATCH");
  const fs::path root = base != nullptr ? base : ::testing::TempDir();
  fs::create_directories(root, error);
  std::string name = (root / "weight-files-XXXXXX").string();
  ASSERT_NE(mkdtemp(name.data()), nullptr);
  const fs::path scratch = name;
  // One store with both artifacts, each linked from where it is installed
  // (an artifact's root may be a link, D-056 as amended).
  const fs::path store = scratch / "store";
  ASSERT_TRUE(fs::create_directory(store));
  fs::create_directory_symlink(gemma2, store / *id2, error);
  ASSERT_FALSE(error) << error.message();
  fs::create_directory_symlink(gemma3, store / *id3, error);
  ASSERT_FALSE(error) << error.message();
  const std::size_t shards2 = Shards(gemma2);
  const std::size_t shards3 = Shards(gemma3);
  ASSERT_GT(shards2, 0U);
  ASSERT_GT(shards3, 0U);

  cfg::NodeConfig config;
  cfg::RuntimeRoles roles;
  roles.installed = store;
  roles.spill = scratch / "spill";
  roles.state = scratch / "state";
  for (const auto& path : {roles.spill, roles.state}) {
    ASSERT_TRUE(fs::create_directory(path));
    ASSERT_EQ(chmod(path.c_str(), 0700), 0);
  }
  for (const auto& [model, artifact, context] :
       {std::tuple{"gemma2", *id2, 8192}, std::tuple{"gemma3", *id3, 4096}}) {
    cfg::ModelEntry entry;
    entry.name = model;
    entry.artifact = artifact;
    entry.overrides["context"] = std::int64_t{context};
    entry.overrides["prefill_chunk"] = std::int64_t{128};
    entry.overrides["max_slots"] = std::int64_t{1};
    config.models.push_back(entry);
  }
  rt::ServingOptions options;
  std::FILE* log = std::tmpfile();
  ASSERT_NE(log, nullptr);
  {
    auto server = std::make_unique<rt::Server>(config, roles, options, log);
    const auto started = server->Start(false);
    ASSERT_TRUE(started) << (started ? "" : started.error());
    rt::Served* g2 = server->Find("gemma2");
    rt::Served* g3 = server->Find("gemma3");
    ASSERT_NE(g2, nullptr);
    ASSERT_NE(g3, nullptr);
    // Registered and set up: nothing of either is open.
    EXPECT_EQ(OpenBeneath(gemma2), 0U);
    EXPECT_EQ(OpenBeneath(gemma3), 0U);
    EXPECT_EQ(server->node().open_weight_files(), 0U);
    const std::vector<rt::Served*> order = {g2, g3, g2, g3};
    for (std::size_t i = 0; i < order.size(); ++i) {
      rt::SwapParts parts;
      const auto activated = server->Activate(*order[i], parts);
      ASSERT_TRUE(activated) << (activated ? "" : activated.error());
      const auto finished = server->FinishSwap(parts);
      ASSERT_TRUE(finished) << (finished ? "" : finished.error());
      const bool two = order[i] == g2;
      std::fprintf(
          stderr, "after activating %s: gemma2 %zu of %zu shards open, gemma3 %zu of %zu\n",
          order[i]->name().c_str(), OpenBeneath(gemma2), shards2, OpenBeneath(gemma3), shards3);
      EXPECT_EQ(OpenBeneath(gemma2), two ? shards2 : 0U) << "switch " << i;
      EXPECT_EQ(OpenBeneath(gemma3), two ? 0U : shards3) << "switch " << i;
      EXPECT_EQ(server->node().open_weight_files(), two ? shards2 : shards3) << "switch " << i;
    }
    // No page-in found its file closed: each one ran while its model was
    // active.
    const auto stats = server->node().Stats();
    ASSERT_TRUE(stats) << (stats ? "" : stats.error());
    EXPECT_EQ(stats->closed_file_reads, 0U);
    const auto down = server->TearDown();
    EXPECT_TRUE(down) << (down ? "" : down.error());
    EXPECT_EQ(OpenBeneath(gemma2), 0U);
    EXPECT_EQ(OpenBeneath(gemma3), 0U);
  }
  (void)std::fclose(log);
  fs::remove_all(scratch, error);
}

}  // namespace
