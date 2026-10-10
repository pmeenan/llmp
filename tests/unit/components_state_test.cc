// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The state contracts of the components (components/step.h StageLayout)
// as the engine's shared slot lifecycle (engine/slot_lifecycle.h) runs
// them on the fake device backend: a small Gemma-like stage (a sliding
// layer's ring and a global layer's cache, each K and V) grows, spills and
// restores its bytes exactly; and what the contracts declare: the ring
// cannot be truncated or written by a verify, the global cache can; and the
// kept layout's identity changes with the window.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "base/bytes.h"
#include "components/step.h"
#include "engine/paged_node.h"
#include "engine/slot_lifecycle.h"
#include "engine/support.h"
#include "expected_error.h"
#include "model/state.h"
#include "model/state_contract.h"
#include "providers/device_runtime.h"
#include "scheduler/commands.h"
#include "spec/spec.h"

namespace {

namespace co = llmp::components;
namespace en = llmp::engine;
namespace md = llmp::model;
namespace pr = llmp::providers;
namespace sp = llmp::spec;
using llmp::base::Bytes;
using Range = en::LiveState::Range;

#define ASSERT_OK(expr)                                                       \
  do {                                                                        \
    const auto status_ = (expr);                                              \
    ASSERT_TRUE(status_) << llmp::test_support::Failed(status_).value_or(""); \
  } while (false)
constexpr std::uint64_t kExtent = en::kPagedExtent;
constexpr std::uint32_t kContext = 4096;
constexpr std::uint32_t kRows = 64;

sp::ComponentRef C(std::string kind, std::vector<sp::Param> params) {
  std::ranges::sort(params, {}, &sp::Param::name);
  return {.kind = std::move(kind), .version = 1, .params = std::move(params), .weights = {}};
}
sp::Param P(std::string name, sp::ParamValue v) {
  return {.name = std::move(name), .value = std::move(v)};
}

// Two layers: a sliding layer of `window` positions, then a global layer;
// one KV head of 256 values (512 bytes a cell).
sp::ModelSpec Spec(std::int64_t window) {
  const auto attention = [](std::int64_t w) {
    return C("attention.dense",
             {P("placement", std::string("sandwich")), P("eps", 1e-6), P("heads", std::int64_t{2}),
              P("kv_heads", std::int64_t{1}), P("key_dim", std::int64_t{256}),
              P("value_dim", std::int64_t{256}), P("scale", 0.0625), P("window", w),
              P("rotation", std::string("neox")), P("rope_dims", std::int64_t{256}),
              P("rope_base", 10000.0), P("rope_scale", 1.0)});
  };
  const auto ffn =
      C("ffn.glu", {P("placement", std::string("sandwich")), P("eps", 1e-6),
                    P("activation", std::string("gelu_tanh")), P("hidden", std::int64_t{1024})});
  sp::Stage s{.name = "target", .context = kContext};
  s.embedding = C("embedding.token", {P("scale", 16.0)});
  s.stream.width = 256;
  s.layers = {{.components = {attention(window), ffn}}, {.components = {attention(0), ffn}}};
  s.heads.final_norm = C("norm.rms", {P("eps", 1e-6)});
  s.heads.output = C("head.linear", {});
  return {.stages = {std::move(s)}};
}

TEST(ComponentsStateTest, ContractsDeclareTheRingsLimitsAndTheLayoutIdentity) {
  const auto layout = co::StageLayout(Spec(128), 0, kContext, kRows);
  ASSERT_TRUE(layout.has_value()) << layout.error();
  ASSERT_EQ(layout->items.size(), 4U);
  // The ring holds pad(window + chunk) cells, the global cache the context.
  EXPECT_EQ(layout->items[0].cells, 256U);
  EXPECT_EQ(layout->items[0].contract.cells, md::CellRule::kRing);
  EXPECT_EQ(layout->items[0].contract.encoding, md::Encoding::kF16);
  EXPECT_EQ(layout->items[0].contract.cell_bytes, 512U);
  EXPECT_EQ(layout->items[2].cells, kContext);
  EXPECT_EQ(layout->items[2].contract.cells, md::CellRule::kPerPosition);
  // A ring truncates as the global cache does: it holds a chunk's rows
  // beside its window, so a later query never reads a rejected row's cell
  // (nor the older position it overwrote). A verify saves nothing in it.
  const auto reps = md::Representations(*layout, "target", true);
  ASSERT_EQ(reps.size(), 2U);
  EXPECT_TRUE(reps[0].Can(md::StateCapability::kTruncate));
  EXPECT_TRUE(reps[1].Can(md::StateCapability::kTruncate));
  const auto writes = md::WritesFor(*layout, 10, 4);
  ASSERT_TRUE(writes.has_value()) << writes.error();
  for (const auto& row : writes->rows) EXPECT_TRUE(row.empty());
  EXPECT_TRUE(writes->scratch.empty());
  // So does a global-only stage.
  const auto global = co::StageLayout(Spec(0), 0, kContext, kRows);
  ASSERT_TRUE(global.has_value());
  EXPECT_TRUE(md::WritesFor(*global, 10, 4).has_value());
  // The kept layout's identity follows the window.
  const auto other = co::StageLayout(Spec(64), 0, kContext, kRows);
  ASSERT_TRUE(other.has_value());
  EXPECT_NE(md::KeptLayoutId(Spec(128).Digest(), std::span(&*layout, 1)),
            md::KeptLayoutId(Spec(64).Digest(), std::span(&*other, 1)));
}

class Runner final : public en::PagedModel {
 public:
  Runner(en::PagedNode& node, en::SlotLifecycleOptions options)
      : slots(node, std::move(options), 0, 0), node_(node) {}
  en::SlotLifecycle slots;
  std::uint32_t stream() const override { return 0; }
  const llmp::catalog::Closure& fence_closure() const override { return slots.fence(); }
  std::vector<llmp::catalog::ExtentId> managed_extents() const override {
    auto all = slots.state();
    const auto kept = slots.kept_state();
    all.insert(all.end(), kept.begin(), kept.end());
    return all;
  }
  en::Status Release() override {
    std::vector<std::string> problems;
    slots.Release(node_.memory(), problems);
    if (!problems.empty()) return std::unexpected(problems.front());
    return {};
  }

 private:
  en::PagedNode& node_;
};

TEST(ComponentsStateTest, TheSlotLifecycleSpillsAndRestoresTheContractsState) {
  // NOLINTNEXTLINE(concurrency-mt-unsafe): the test environment is immutable
  const char* scratch = std::getenv("LLMP_TEST_SCRATCH");
  ASSERT_NE(scratch, nullptr);
  std::filesystem::create_directories(scratch);
  const auto spec = Spec(128);
  const auto layout = co::StageLayout(spec, 0, kContext, kRows);
  ASSERT_TRUE(layout.has_value());
  en::PagedNode node{
      en::NodeSettings{.compute_streams = 1, .slots = 2, .copy_lane = true, .slot_bytes = kExtent}};
  ASSERT_TRUE(node.Open());
  const std::uint64_t fixed = node.catalog().OccupancyOf(node.domain()).Total().value();
  en::SlotLifecycleOptions options{.model = "Components",
                                   .slots = 2,
                                   .stages = {{.name = "target", .layout = *layout}},
                                   .spec = spec.Digest(),
                                   .read_align = 256,
                                   .spill_place = {},
                                   .spill_directory = scratch,
                                   .invalidate = {},
                                   .drop_plans = {}};
  auto runner = std::make_unique<Runner>(node, std::move(options));
  en::SlotLifecycle& slots = runner->slots;
  ASSERT_OK(slots.Setup());
  ASSERT_TRUE(node.Start(Bytes(fixed + (32 * kExtent))));
  ASSERT_OK(slots.Register());
  ASSERT_TRUE(node.scheduler().PinPlaces(slots.reserved_extents()));
  node.Run();
  ASSERT_OK(slots.Bind({}, {}));
  std::vector<llmp::catalog::ExtentId> staging;
  auto host = node.Pinned(8 * kExtent, 0, staging);
  ASSERT_TRUE(host.has_value()) << host.error();
  const auto selected = slots.SelectSlots(std::vector<std::uint32_t>{0});
  ASSERT_TRUE(selected) << llmp::test_support::Failed(selected).value_or("");

  // Slot 0 holds 300 positions: the ring whole, the global cache's first 512
  // cells (its read alignment).
  ASSERT_OK(slots.ReserveStateThrough(0, 300));
  ASSERT_OK(slots.SetPositions(0, 300));
  std::vector<Range> used = slots.live(0, 0).used_ranges();
  std::uint64_t bytes = 0;
  for (Range& r : used) {
    r.region = 0;
    bytes += r.bytes;
  }
  const auto expected = md::UsedThrough(*layout, 300, 256);
  ASSERT_TRUE(expected.has_value());
  std::uint64_t contract_bytes = 0;
  for (const auto& r : *expected) contract_bytes += r.bytes;
  EXPECT_EQ(contract_bytes, (256 * 512 * 2) + (512 * 512 * 2));
  ASSERT_LE(bytes, 8 * kExtent);
  // Each used range filled with its own value on the device (one job),
  // spilled, restored and read back.
  std::vector<std::pair<std::uint64_t, std::uint64_t>> at;
  for (const Range& r : used) at.emplace_back(slots.live(0, 0).base(0) + r.offset, r.bytes);
  const auto filled = node.Job(
      slots.slot_fence(0),
      [at](pr::NativeStream native) {
        for (std::size_t i = 0; i < at.size(); ++i) {
          const auto value = static_cast<std::uint8_t>(0xA0 + i);
          if (!pr::FillAsync(native, en::support::Pointer(at[i].first), value, at[i].second).ok()) {
            return llmp::scheduler::JobResult::kUnknown;
          }
        }
        return llmp::scheduler::JobResult::kQueued;
      },
      "test fill", 0);
  ASSERT_OK(filled);
  ASSERT_OK(slots.Spill(0));
  EXPECT_TRUE(slots.spilled(0));
  ASSERT_OK(slots.Restore(0));
  auto* p = static_cast<std::uint8_t*>(*host);
  std::fill(p, p + bytes, std::uint8_t{0});
  ASSERT_OK(slots.CopyState(0, *host, used, true));
  std::uint64_t offset = 0;
  for (std::size_t i = 0; i < used.size(); ++i) {
    const auto value = static_cast<std::uint8_t>(0xA0 + i);
    EXPECT_TRUE(std::all_of(p + offset, p + offset + used[i].bytes,
                            [value](std::uint8_t b) { return b == value; }))
        << "range " << i;
    offset += used[i].bytes;
  }

  std::array<en::PagedModel*, 1> models = {runner.get()};
  const auto closed = node.TearDown(models);
  EXPECT_TRUE(closed) << llmp::test_support::Failed(closed).value_or("");
}

}  // namespace
