// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Weight files open only while their model is active (engine/paged_node.h
// SetActive, engine/paged_weights.h, scheduler/files.h), on the fake device
// backend over copies of the artifact corpus's tiny golden, counting the
// process's own descriptors (/proc/self/fd):
//
// - the file table's rules: a descriptor installed once, taken back only
//   once no read that began on it is in flight;
// - nothing open after registration; a model's shards open at its first
//   page-in, and close when it is deactivated; a page-in of an inactive
//   model reads nothing and fails;
// - a reopen checks each shard's size against the installed manifest, and
//   one of another size refuses the activation with none left open;
// - a file a read began on stays open (and the read lands) while its model
//   is deactivated, and closes once the read has concluded;
// - two models swapped back and forth leave only the active one's open.

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <expected>
#include <filesystem>
#include <format>
#include <fstream>
#include <memory>
#include <span>
#include <string>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

#include "artifact/artifact.h"
#include "artifact/layout.h"
#include "artifact_writer.h"
#include "base/bytes.h"
#include "catalog/catalog.h"
#include "engine/paged_node.h"
#include "engine/paged_weights.h"
#include "expected_error.h"
#include "format/encoding.h"
#include "format/layout.h"
#include "format/tensor_format.h"
#include "scheduler/files.h"
#include "scheduler/programs.h"
#include "scheduler/scheduler.h"

namespace {

namespace en = llmp::engine;
namespace sc = llmp::scheduler;
namespace fs = std::filesystem;
using llmp::test_support::Failed;
using Status = std::expected<void, std::string>;

constexpr std::uint64_t kExtent = en::kPagedExtent;

// ------------------------------------------------------------ the table

TEST(FileTable, InstallsOnceAndTakesBackOnlyWithNoReadInFlight) {
  sc::FileTable table;
  const sc::FileId file = table.Add();
  EXPECT_TRUE(table.Known(file));
  EXPECT_FALSE(table.Known(sc::FileId{}));
  EXPECT_FALSE(table.Known(sc::FileId(7, 1)));
  EXPECT_EQ(table.Descriptor(file), -1);
  // Closed: a read gets no descriptor and is not counted.
  EXPECT_EQ(table.BeginRead(file), -1);
  EXPECT_EQ(table.Reads(file), 0U);
  EXPECT_EQ(Failed(table.Install(file, -1)), sc::FileRefusal::kInvalid);
  EXPECT_EQ(Failed(table.Install(sc::FileId(7, 1), 3)), sc::FileRefusal::kUnknown);
  ASSERT_TRUE(table.Install(file, 42).has_value());
  EXPECT_EQ(Failed(table.Install(file, 43)), sc::FileRefusal::kOpen);
  EXPECT_EQ(table.Descriptor(file), 42);
  EXPECT_EQ(table.open(), 1U);
  // Two reads begin on it: it is not taken back until both have ended.
  EXPECT_EQ(table.BeginRead(file), 42);
  EXPECT_EQ(table.BeginRead(file), 42);
  EXPECT_EQ(table.Reads(file), 2U);
  EXPECT_EQ(Failed(table.Take(file)), sc::FileRefusal::kReading);
  EXPECT_EQ(table.Descriptor(file), 42);
  table.EndRead(file);
  EXPECT_EQ(Failed(table.Take(file)), sc::FileRefusal::kReading);
  table.EndRead(file);
  const auto taken = table.Take(file);
  ASSERT_TRUE(taken.has_value());
  EXPECT_EQ(*taken, 42);
  // Closed again: the next read gets none, and taking it gives none.
  EXPECT_EQ(table.BeginRead(file), -1);
  EXPECT_EQ(table.Take(file).value_or(0), -1);
  EXPECT_EQ(table.open(), 0U);
}

// ------------------------------------------------------------ the node

llmp::test::V1Plan TwoShardPlan() {
  namespace fm = llmp::format;
  llmp::test::V1Plan p;
  p.architecture = "toy";
  p.container = "safetensors";
  p.groups = {
      llmp::test::V1Group{.kind = llmp::artifact::GroupKind::kTable,
                          .resources = {{.name = "token_embd.weight",
                                         .aliases = {"output.weight"},
                                         .format = {.encoding = fm::Encoding::kQ8_0,
                                                    .layout = fm::Layout::kGgmlBlocks,
                                                    .shape = {.ne = {64, 512, 1, 1}, .rank = 2}},
                                         .rows = true}}},
      llmp::test::V1Group{.kind = llmp::artifact::GroupKind::kLayer,
                          .layer = 0,
                          .resources = {{.name = "blk.0.attn_q.weight",
                                         .format = {.encoding = fm::Encoding::kQ8_0,
                                                    .layout = fm::Layout::kGgmlBlocks,
                                                    .shape = {.ne = {64, 512, 1, 1}, .rank = 2}}}},
                          .new_shard = true},
  };
  return p;
}

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

void Append(const fs::path& path, std::size_t bytes) {
  std::ofstream out(path, std::ios::binary | std::ios::app);
  const std::string zeros(bytes, '\0');
  out.write(zeros.data(), static_cast<std::streamsize>(zeros.size()));
}

// One model: its weights, every group but the experts' on the device.
class Model final : public en::PagedModel {
 public:
  Model(en::PagedNode& node, std::uint32_t stream) : node_(node), stream_(stream) {}
  std::uint32_t stream() const override { return stream_; }
  const llmp::catalog::Closure& fence_closure() const override { return none_; }
  std::vector<llmp::catalog::ExtentId> managed_extents() const override {
    return weights.extents();
  }
  Status Release() override { return weights.Release(node_.memory()); }

  Status Reserve(const fs::path& root) {
    if (auto opened = weights.Open(root); !opened) return opened;
    std::vector<en::GroupPlace> places;
    for (const auto& group : weights.artifact().groups()) {
      places.push_back(group.kind == llmp::artifact::GroupKind::kExpert ? en::GroupPlace::kNone
                                                                        : en::GroupPlace::kDevice);
    }
    return weights.Reserve(node_, places, {});
  }

  en::PagedWeights weights;

 private:
  en::PagedNode& node_;
  std::uint32_t stream_;
  llmp::catalog::Closure none_;
};

class WeightFilesTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // NOLINTNEXTLINE(concurrency-mt-unsafe): test environment is immutable
    const char* base = std::getenv("LLMP_TEST_SCRATCH");
    ASSERT_NE(base, nullptr);
    scratch_ = fs::path(base) / "weight-files" /
               ::testing::UnitTest::GetInstance()->current_test_info()->name();
    std::error_code error;
    fs::remove_all(scratch_, error);
    const auto rendered = llmp::test::Render(TwoShardPlan());
    for (const char* name : {"a", "b"}) {
      fs::create_directories(scratch_ / name);
      roots_.push_back(llmp::test::Publish(scratch_ / name, rendered));
    }
    auto probe = llmp::artifact::Artifact::Open(roots_[0]);
    ASSERT_TRUE(probe.has_value()) << probe.error().ToString();
    shards_ = probe->shards().size();
    ASSERT_GE(shards_, 2U);
    if (auto fd = probe->OpenShardForDirectRead(0);
        !fd && fd.error().rule == llmp::artifact::Rule::kIo) {
      GTEST_SKIP() << "no direct I/O beneath " << scratch_;
    }
  }

  void TearDown() override {
    if (node_ != nullptr) {
      std::vector<en::PagedModel*> models;
      for (auto& m : models_) models.push_back(m.get());
      const auto closed = node_->TearDown(models);
      EXPECT_TRUE(closed) << Failed(closed).value_or("");
      for (const fs::path& root : roots_) EXPECT_EQ(OpenBeneath(root), 0U) << root;
    }
    models_.clear();
    node_.reset();
    std::error_code error;
    fs::remove_all(scratch_, error);
  }

  // A node with `count` models (owner i on root i), registered and running.
  void Start(std::size_t count, const std::atomic<bool>* hold = nullptr) {
    node_ = std::make_unique<en::PagedNode>(en::NodeSettings{.compute_streams = count,
                                                             .slots = 2,
                                                             .copy_lane = true,
                                                             .slot_bytes = kExtent,
                                                             .hold_reads = hold});
    ASSERT_TRUE(node_->Open());
    for (std::size_t i = 0; i < count; ++i) {
      models_.push_back(std::make_unique<Model>(*node_, static_cast<std::uint32_t>(i)));
      const auto reserved = models_.back()->Reserve(roots_[i]);
      ASSERT_TRUE(reserved) << Failed(reserved).value_or("");
    }
    ASSERT_TRUE(node_->Start(llmp::base::Bytes(256 * kExtent)));
    for (std::size_t i = 0; i < count; ++i) {
      const auto registered = models_[i]->weights.Register(*node_, static_cast<int>(i));
      ASSERT_TRUE(registered) << Failed(registered).value_or("");
    }
    node_->Run();
  }

  std::vector<llmp::catalog::ExtentId> Extents(std::size_t model) const {
    return models_[model]->weights.extents();
  }
  Status Load(std::size_t model) {
    std::vector<en::LoadStats> log;
    return node_->Load(Extents(model), "a model's weights", log);
  }
  std::uint64_t ClosedReads() {
    const auto stats = node_->Stats();
    return stats ? stats->closed_file_reads : 0;
  }

  fs::path scratch_;
  std::vector<fs::path> roots_;
  std::size_t shards_ = 0;
  std::unique_ptr<en::PagedNode> node_;
  std::vector<std::unique_ptr<Model>> models_;
};

TEST_F(WeightFilesTest, OpenOnFirstNeedAndClosedWhenInactive) {
  Start(1);
  ASSERT_FALSE(models_.empty());
  // Registered, nothing paged in: nothing open.
  EXPECT_EQ(node_->open_weight_files(), 0U);
  EXPECT_EQ(OpenBeneath(roots_[0]), 0U);
  EXPECT_EQ(models_[0]->weights.shard_fd(0), -1);
  // The first page-in opens every shard of the model's.
  const auto loaded = Load(0);
  ASSERT_TRUE(loaded) << Failed(loaded).value_or("");
  EXPECT_EQ(node_->open_weight_files(0), shards_);
  EXPECT_EQ(OpenBeneath(roots_[0]), shards_);
  EXPECT_GE(models_[0]->weights.shard_fd(0), 0);
  // Deactivated: closed, and a page-in reads nothing and fails.
  ASSERT_TRUE(node_->SetActive(0, false));
  EXPECT_FALSE(node_->active(0));
  EXPECT_EQ(node_->open_weight_files(), 0U);
  EXPECT_EQ(OpenBeneath(roots_[0]), 0U);
  EXPECT_EQ(models_[0]->weights.shard_fd(0), -1);
  ASSERT_TRUE(node_->Evict(Extents(0)));
  const std::uint64_t refused = ClosedReads();
  EXPECT_FALSE(Load(0));
  EXPECT_GT(ClosedReads(), refused);
  EXPECT_EQ(OpenBeneath(roots_[0]), 0U);
  // Active again: reopened, and the page-in reads.
  ASSERT_TRUE(node_->SetActive(0, true));
  EXPECT_EQ(OpenBeneath(roots_[0]), shards_);
  const auto again = Load(0);
  EXPECT_TRUE(again) << Failed(again).value_or("");
}

TEST_F(WeightFilesTest, AReopenChecksEachShardsSize) {
  Start(1);
  ASSERT_FALSE(models_.empty());
  ASSERT_TRUE(Load(0));
  ASSERT_TRUE(node_->SetActive(0, false));
  // A shard grown while the model was inactive refuses its activation,
  // with none of its shards left open.
  const auto& shard = models_[0]->weights.artifact().shards()[1];
  const fs::path path = roots_[0] / shard.path;
  Append(path, 4096);
  const auto refused = node_->SetActive(0, true);
  ASSERT_FALSE(refused);
  const std::string why = Failed(refused).value_or("");
  EXPECT_NE(why.find(shard.path), std::string::npos) << why;
  EXPECT_NE(why.find(std::format("its installed manifest lists {} bytes",
                                 shard.data_offset.value() + shard.data_bytes.value())),
            std::string::npos)
      << why;
  EXPECT_FALSE(node_->active(0));
  EXPECT_EQ(node_->open_weight_files(), 0U);
  EXPECT_EQ(OpenBeneath(roots_[0]), 0U);
  // Its size restored, it activates and pages in again.
  fs::resize_file(path, shard.data_offset.value() + shard.data_bytes.value());
  ASSERT_TRUE(node_->SetActive(0, true));
  ASSERT_TRUE(node_->Evict(Extents(0)));
  const auto loaded = Load(0);
  EXPECT_TRUE(loaded) << Failed(loaded).value_or("");
}

TEST_F(WeightFilesTest, AFileAReadUsesStaysOpenUntilTheReadEnds) {
  std::atomic<bool> hold{false};
  Start(1, &hold);
  ASSERT_FALSE(models_.empty());
  ASSERT_TRUE(Load(0));  // opens every shard
  ASSERT_TRUE(node_->Evict(Extents(0)));
  // One extent's page-in, its read held in the drive.
  const llmp::catalog::ExtentId one = Extents(0).front();
  llmp::catalog::Closure closure;
  ASSERT_TRUE(node_->Call(
      [&]() -> Status {
        auto c = node_->catalog().ClosureOfExtents(std::span(&one, 1));
        if (!c) return std::unexpected("no closure");
        closure = *c;
        return {};
      },
      "a closure"));
  hold = true;
  sc::ProgramDone done;
  const std::uint64_t request =
      node_->Submit(std::make_unique<sc::RunProgram>(done, closure, sc::DeviceJob{}));
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
  while (node_->counting().held.load() == 0 && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  ASSERT_GT(node_->counting().held.load(), 0U);
  // Deactivated meanwhile: every other shard closes; the one the read
  // began on stays open, its descriptor still valid.
  ASSERT_TRUE(node_->SetActive(0, false));
  EXPECT_EQ(node_->open_weight_files(0), 1U);
  EXPECT_EQ(OpenBeneath(roots_[0]), 1U);
  // The read lands, through the descriptor it began on.
  hold = false;
  const auto finished = node_->Await(done, "a held page-in", request);
  EXPECT_TRUE(finished) << Failed(finished).value_or("");
  // Now nothing uses it: the next sync closes it.
  ASSERT_TRUE(node_->SyncWeightFiles());
  EXPECT_EQ(node_->open_weight_files(), 0U);
  EXPECT_EQ(OpenBeneath(roots_[0]), 0U);
}

TEST_F(WeightFilesTest, ASwapCycleLeavesOneModelsFilesOpen) {
  Start(2);
  ASSERT_EQ(models_.size(), 2U);
  // As the runtime does: no model active until it is activated.
  ASSERT_TRUE(node_->SetActive(0, false));
  ASSERT_TRUE(node_->SetActive(1, false));
  ASSERT_TRUE(node_->SetActive(0, true));
  ASSERT_TRUE(Load(0));
  EXPECT_EQ(OpenBeneath(roots_[0]), shards_);
  EXPECT_EQ(OpenBeneath(roots_[1]), 0U);
  for (std::size_t cycle = 0; cycle < 4; ++cycle) {
    const std::size_t out = cycle % 2;
    const std::size_t in = 1 - out;
    ASSERT_TRUE(node_->SetActive(static_cast<int>(in), true));
    llmp::catalog::Closure closure;
    ASSERT_TRUE(node_->Call(
        [&]() -> Status {
          auto c = node_->catalog().ClosureOfExtents(Extents(in));
          if (!c) return std::unexpected("no closure");
          closure = *c;
          return {};
        },
        "a closure"));
    sc::SwapReport report;
    const auto swapped = node_->Swap(Extents(out), closure, true, report);
    ASSERT_TRUE(swapped) << Failed(swapped).value_or("");
    ASSERT_TRUE(node_->SetActive(static_cast<int>(out), false));
    EXPECT_EQ(node_->open_weight_files(), shards_) << "cycle " << cycle;
    EXPECT_EQ(OpenBeneath(roots_[in]), shards_) << "cycle " << cycle;
    EXPECT_EQ(OpenBeneath(roots_[out]), 0U) << "cycle " << cycle;
  }
}

}  // namespace
