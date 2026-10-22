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

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <memory>
#include <string>
#include <system_error>
#include <thread>
#include <tuple>
#include <type_traits>
#include <vector>

#include "artifact/artifact.h"
#include "expected_error.h"
#include "runtime/service_backend.h"
#include "runtime/serving.h"
#include "v1_frozen.h"

namespace {

namespace rt = llmp::runtime;
namespace cfg = llmp::config;
namespace fs = std::filesystem;
namespace en = llmp::engine;

TEST(SwapObserverGpu, NamesArePrepaidMoveOnlyAndOutliveTheNode) {
  static_assert(!std::is_copy_constructible_v<rt::SwapParts>);
  static_assert(!std::is_copy_assignable_v<rt::SwapParts>);
  en::HostPayment probe;
  std::uint64_t baseline = 0;
  std::uint64_t replacement_paid = 0;
  std::vector<rt::SwapParts> rows;
  {
    en::PagedNode node(
        en::NodeSettings{.compute_streams = 1, .slots = 1, .slot_bytes = en::kPagedExtent});
    ASSERT_TRUE(node.Open());
    const auto fixed = node.catalog().OccupancyOf(node.domain()).Total().value();
    node.SetHostFloor(0);
    ASSERT_TRUE(node.Start(llmp::base::Bytes(fixed + 16 * en::kPagedExtent)));
    node.Run();
    auto tracking = node.PrepayHost(1);
    ASSERT_TRUE(tracking);
    probe = std::move(*tracking);
    baseline = probe.accounted_bytes();
    std::string from(128 * 1024, 'a');
    std::string to(256 * 1024, 'b');
    auto first = rt::SwapParts::Create(node, from, to);
    ASSERT_TRUE(first) << llmp::test_support::Failed(first).value_or("");
    const auto first_paid = probe.accounted_bytes() - baseline;
    EXPECT_GE(first_paid, from.size() + to.size() + sizeof(rt::SwapParts));
    const auto* original = first->from.data();
    rows.push_back(std::move(*first));
    rows.reserve(8);  // table growth transfers ownership without duplicating strings
    EXPECT_EQ(rows.front().from.data(), original);
    EXPECT_EQ(probe.accounted_bytes(), baseline + first_paid);
    from.assign(from.size(), 'x');
    to.assign(to.size(), 'y');
    EXPECT_EQ(rows.front().from.front(), 'a');
    EXPECT_EQ(rows.front().to.back(), 'b');
    auto second = rt::SwapParts::Create(node, from, to);
    ASSERT_TRUE(second);
    replacement_paid = probe.accounted_bytes() - baseline - first_paid;
    rows.front() = std::move(*second);  // retires old payload before refunding it
    EXPECT_EQ(probe.accounted_bytes(), baseline + replacement_paid);
    auto& self = rows.front();
    rows.front() = std::move(self);
    EXPECT_EQ(probe.accounted_bytes(), baseline + replacement_paid);
    ASSERT_TRUE(node.TearDown({}));
  }
  EXPECT_EQ(rows.front().from.front(), 'x');
  EXPECT_EQ(rows.front().to.back(), 'y');
  EXPECT_EQ(probe.accounted_bytes(), baseline + replacement_paid);
  std::thread release([retained = std::move(rows)]() mutable { retained.clear(); });
  release.join();
  EXPECT_EQ(probe.accounted_bytes(), baseline);
}

TEST(SwapObserverGpu, RefusedNamesLeaveNoUnfundedObserverOrCharge) {
  en::PagedNode node(
      en::NodeSettings{.compute_streams = 1, .slots = 1, .slot_bytes = en::kPagedExtent});
  ASSERT_TRUE(node.Open());
  const auto fixed = node.catalog().OccupancyOf(node.domain()).Total().value();
  node.SetHostFloor(0);
  ASSERT_TRUE(node.Start(llmp::base::Bytes(fixed + 2 * en::kPagedExtent)));
  node.Run();
  const auto before = node.host_counted();
  const std::string name(3 * en::kPagedExtent, 'z');
  auto refused = rt::SwapParts::Create(node, name, "incoming");
  EXPECT_FALSE(refused);
  EXPECT_EQ(node.host_counted(), before);
  auto retry = rt::SwapParts::Create(node, "outgoing", "incoming");
  ASSERT_TRUE(retry) << llmp::test_support::Failed(retry).value_or("");
  EXPECT_EQ(retry->from, "outgoing");
  EXPECT_EQ(retry->to, "incoming");
  ASSERT_TRUE(node.TearDown({}));
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

std::size_t Shards(const fs::path& root) {
  auto opened = llmp::artifact::Artifact::Open(root);
  return opened ? opened->shards().size() : 0;
}

class WeightFilesServingGpu : public ::testing::Test {
 protected:
  cfg::NodeConfig config;
  cfg::RuntimeRoles roles;
  rt::ServingOptions options;
  fs::path scratch, gemma2, gemma3;
  std::size_t shards2 = 0, shards3 = 0;
  std::FILE* log = nullptr;
  std::unique_ptr<rt::Server> server;
  rt::Served* g2 = nullptr;
  rt::Served* g3 = nullptr;

  void SetUp() override {
    const fs::path models = llmp::test_support::V1Store();
    const auto id2 = llmp::test_support::V1IdNamed("gemma2");
    const auto id3 = llmp::test_support::V1IdNamed("gemma3");
    if (!id2 || !id3) GTEST_FAIL() << "no published v1 Gemma 2 and Gemma 3 in " << models;
    gemma2 = models / *id2;
    gemma3 = models / *id3;
    std::error_code error;
    // NOLINTNEXTLINE(concurrency-mt-unsafe): test environment is immutable
    const char* base = std::getenv("LLMP_TEST_SCRATCH");
    const fs::path root = base != nullptr ? base : ::testing::TempDir();
    fs::create_directories(root, error);
    std::string name = (root / "weight-files-XXXXXX").string();
    ASSERT_NE(mkdtemp(name.data()), nullptr);
    scratch = name;
    // One store with both artifacts, each linked from where it is installed
    // (an artifact's root may be a link, D-056 as amended).
    const fs::path store = scratch / "store";
    ASSERT_TRUE(fs::create_directory(store));
    fs::create_directory_symlink(gemma2, store / *id2, error);
    ASSERT_FALSE(error) << error.message();
    fs::create_directory_symlink(gemma3, store / *id3, error);
    ASSERT_FALSE(error) << error.message();
    shards2 = Shards(gemma2);
    shards3 = Shards(gemma3);
    ASSERT_GT(shards2, 0U);
    ASSERT_GT(shards3, 0U);

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
    log = std::tmpfile();
    ASSERT_NE(log, nullptr);
    server = std::make_unique<rt::Server>(config, roles, options, log);
    const auto started = server->Start(false);
    ASSERT_TRUE(started) << (started ? "" : started.error());
    g2 = server->Find("gemma2");
    g3 = server->Find("gemma3");
    ASSERT_NE(g2, nullptr);
    ASSERT_NE(g3, nullptr);
  }
  void TearDown() override {
    if (server) {
      const auto down = server->TearDown();
      EXPECT_TRUE(down) << llmp::test_support::Failed(down).value_or("");
      server.reset();
    }
    if (log) (void)std::fclose(log);
    std::error_code error;
    if (!scratch.empty()) fs::remove_all(scratch, error);
  }
  std::string Log() const {
    (void)std::fflush(log);
    std::rewind(log);
    std::string text;
    std::array<char, 4096> buffer{};
    for (std::size_t n = 0; (n = std::fread(buffer.data(), 1, buffer.size(), log)) > 0;)
      text.append(buffer.data(), n);
    (void)std::fseek(log, 0, SEEK_END);
    return text;
  }
  std::size_t Reports(std::string_view from, std::string_view to) const {
    const std::string text = Log();
    const std::string prefix = std::format("swap {} -> {}: ready in ", from, to);
    std::size_t count = 0;
    for (std::size_t at = 0; (at = text.find(prefix, at)) != std::string::npos; at += prefix.size())
      ++count;
    return count;
  }
  void AwaitReport(rt::service_detail::NodeBackend& backend, std::string_view from,
                   std::string_view to) const {
    const auto until = rt::Clock::now() + std::chrono::seconds(10);
    while (Reports(from, to) == 0 && backend.healthy() && rt::Clock::now() < until) {
      backend.AfterResponse();
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    EXPECT_TRUE(backend.healthy()) << backend.failure();
    EXPECT_EQ(Reports(from, to), 1U);
  }
  static rt::api::CompletionRequest Request(std::string name) {
    rt::api::CompletionRequest request;
    request.options.model = std::move(name);
    request.options.max_tokens = 1;
    request.token_ids = {2, 818};
    return request;
  }
};

// The service returns directly after activation: no prompt content or
// generation is required to exercise the production swap owner.
class ActivationExchange final : public rt::api::Exchange {
 public:
  bool allow_swap = true;
  bool Admit(const Admission&) override { return true; }
  bool Next(rt::Phase phase, std::uint64_t) override {
    return phase != rt::Phase::kSwap || allow_swap;
  }
  bool Reasoning(std::string_view) override { return true; }
  bool Content(std::string_view) override { return true; }
  bool Continue() override { return false; }
};

TEST_F(WeightFilesServingGpu, OnlyTheResidentModelsShardsStayOpenAcrossSwitches) {
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
    std::fprintf(stderr, "after activating %s: gemma2 %zu of %zu shards open, gemma3 %zu of %zu\n",
                 order[i]->name().c_str(), OpenBeneath(gemma2), shards2, OpenBeneath(gemma3),
                 shards3);
    EXPECT_EQ(OpenBeneath(gemma2), two ? shards2 : 0U) << "switch " << i;
    EXPECT_EQ(OpenBeneath(gemma3), two ? 0U : shards3) << "switch " << i;
    EXPECT_EQ(server->node().open_weight_files(), two ? shards2 : shards3) << "switch " << i;
  }
  const auto stats = server->node().Stats();
  ASSERT_TRUE(stats) << llmp::test_support::Failed(stats).value_or("");
  EXPECT_EQ(stats->closed_file_reads, 0U);
  ASSERT_TRUE(server->TearDown());
  EXPECT_EQ(OpenBeneath(gemma2), 0U);
  EXPECT_EQ(OpenBeneath(gemma3), 0U);
}

TEST_F(WeightFilesServingGpu, SerialNoOpAndCancelledSwitchPreservePendingMeasurement) {
  auto memory = std::make_shared<rt::RequestMemory>(rt::kRequestMemoryFloor);
  rt::service_detail::NodeBackend backend(*server, config, memory, log);
  ActivationExchange exchange;
  auto first = backend.Complete(Request("gemma2"), exchange);
  ASSERT_TRUE(first) << llmp::test_support::Failed(first, &rt::api::Error::message).value_or("");
  EXPECT_EQ(Reports("(nothing)", "gemma2"), 0U);
  // A second same-model completion reaches ActivateFor without a swap.
  ASSERT_TRUE(backend.Complete(Request("gemma2"), exchange));
  // A different model's cancelled swap must not erase the pending one.
  exchange.allow_swap = false;
  ASSERT_TRUE(backend.Complete(Request("gemma3"), exchange));
  EXPECT_EQ(server->resident(), g2);
  backend.AfterResponse();
  backend.AfterResponse();
  EXPECT_TRUE(backend.healthy()) << backend.failure();
  EXPECT_EQ(Reports("(nothing)", "gemma2"), 1U);
  EXPECT_EQ(Reports("gemma2", "gemma3"), 0U);
}

TEST_F(WeightFilesServingGpu, CooperativeSwitchKeepsIndependentMeasurementsBeforeAnyResponse) {
  auto memory = std::make_shared<rt::RequestMemory>(rt::kRequestMemoryFloor);
  rt::service_detail::NodeBackend backend(*server, config, memory, log);
  ActivationExchange exchange;
  // Drive actual cooperative swap units, retiring each request without
  // AfterResponse, as a yielded request does while another model runs.
  const auto activate = [&](std::string name) -> rt::Status {
    const auto request = Request(std::move(name));
    auto started = backend.Start({.options = request.options, .literal = &request}, exchange);
    if (!started) return std::unexpected(started.error().message);
    if (!*started) return std::unexpected("the cooperative request was deferred");
    auto work = std::move(*started);
    const std::array<rt::api::CooperativeBackend::Work*, 1> owners{work.get()};
    const auto unit = backend.NextUnit(owners);
    rt::Status advanced;
    if (!unit)
      advanced = std::unexpected(unit.error());
    else if (unit->phase != rt::Phase::kSwap)
      advanced = std::unexpected("the cooperative unit did not select a swap");
    else
      advanced = backend.Advance(owners);
    work->Cancel();
    const auto retired = backend.Retire(*work);
    // Never free a frame whose native references might still exist.
    if (!retired.references_retired) std::abort();
    if (!retired.result) return std::unexpected(retired.result.error().message);
    return advanced;
  };
  const auto first = activate("gemma2");
  ASSERT_TRUE(first) << llmp::test_support::Failed(first).value_or("");
  EXPECT_EQ(Reports("(nothing)", "gemma2"), 0U);
  const auto second = activate("gemma3");
  ASSERT_TRUE(second) << llmp::test_support::Failed(second).value_or("");
  EXPECT_EQ(Reports("(nothing)", "gemma2"), 1U);
  EXPECT_EQ(Reports("gemma2", "gemma3"), 0U);
  // A response from an earlier request may arrive now; it closes the
  // current activation once, without relabelling the preceding swap.
  AwaitReport(backend, "gemma2", "gemma3");
  backend.AfterResponse();
  EXPECT_TRUE(backend.healthy()) << backend.failure();
  EXPECT_EQ(Reports("(nothing)", "gemma2"), 1U);
  EXPECT_EQ(Reports("gemma2", "gemma3"), 1U);
}

TEST_F(WeightFilesServingGpu, FailedActivationClosesPriorSwapAndDoesNotPublishSuccess) {
  auto memory = std::make_shared<rt::RequestMemory>(rt::kRequestMemoryFloor);
  rt::service_detail::NodeBackend backend(*server, config, memory, log);
  ActivationExchange exchange;
  ASSERT_TRUE(backend.Complete(Request("gemma2"), exchange));
  ASSERT_FALSE(g3->weights().empty());
  const auto missing = g3->weights().back();
  llmp::scheduler::PageSource original;
  ASSERT_TRUE(server->node().Call(
      [&]() -> rt::Status {
        original = *server->node().scheduler().SourceOf(missing);
        auto bad = original;
        bad.read.offset = std::uint64_t{1} << 60U;  // a clean short read
        if (!server->node().scheduler().SetSource(missing, bad))
          return std::unexpected("the failing weight source was refused");
        return {};
      },
      "injecting a swap measurement read failure"));
  const auto failed = backend.Complete(Request("gemma3"), exchange);
  // Repair before assertions that may end the test.
  ASSERT_TRUE(server->node().Call(
      [&]() -> rt::Status {
        if (!server->node().scheduler().SetSource(missing, original))
          return std::unexpected("the original weight source could not be restored");
        return {};
      },
      "restoring the swap measurement read source"));
  EXPECT_EQ(llmp::test_support::Failed(failed, &rt::api::Error::status), 503);
  EXPECT_TRUE(backend.healthy()) << backend.failure();
  EXPECT_EQ(server->resident(), g2);
  backend.AfterResponse();
  EXPECT_EQ(Reports("(nothing)", "gemma2"), 1U);
  EXPECT_EQ(Reports("gemma2", "gemma3"), 0U);
  // Paid work in a failed swap is not reported as a successful activation;
  // the retry starts a new measurement and reports its own success once.
  const auto retried = backend.Complete(Request("gemma3"), exchange);
  ASSERT_TRUE(retried)
      << llmp::test_support::Failed(retried, &rt::api::Error::message).value_or("");
  AwaitReport(backend, "gemma2", "gemma3");
  EXPECT_EQ(Reports("gemma2", "gemma3"), 1U);
  EXPECT_TRUE(backend.healthy()) << backend.failure();
}

TEST_F(WeightFilesServingGpu, FinishedMeasurementKeepsItsEndpointAcrossLaterSwaps) {
  rt::SwapParts first;
  ASSERT_TRUE(server->Activate(*g2, first));
  auto probe = server->node().PrepayHost(1);
  ASSERT_TRUE(probe);
  const auto paid_before_noop = probe->accounted_bytes();
  rt::SwapParts noop;
  ASSERT_TRUE(server->Activate(*g2, noop));
  EXPECT_TRUE(noop.from.empty());
  EXPECT_TRUE(noop.to.empty());
  EXPECT_EQ(probe->accounted_bytes(), paid_before_noop);
  ASSERT_TRUE(server->FinishSwap(noop));
  EXPECT_EQ(noop.release, 0);
  ASSERT_TRUE(server->FinishSwap(first));
  const auto release = first.release;
  const auto handed_off = first.handed_off;
  const auto released_unused = first.released_unused;
  const auto occupancy = first.occupancy_released;
  const auto before_second = server->node().Stats();
  ASSERT_TRUE(before_second) << llmp::test_support::Failed(before_second).value_or("");
  rt::SwapParts second;
  ASSERT_TRUE(server->Activate(*g3, second));
  ASSERT_TRUE(server->FinishSwap(second));
  const auto after_second = server->node().Stats();
  ASSERT_TRUE(after_second) << llmp::test_support::Failed(after_second).value_or("");
  EXPECT_EQ(second.handed_off, after_second->handed_off - before_second->handed_off);
  EXPECT_EQ(second.released_unused, after_second->released_unused - before_second->released_unused);
  ASSERT_TRUE(server->FinishSwap(first));
  EXPECT_EQ(first.release, release);
  EXPECT_EQ(first.handed_off, handed_off);
  EXPECT_EQ(first.released_unused, released_unused);
  EXPECT_EQ(first.occupancy_released, occupancy);
}

}  // namespace
