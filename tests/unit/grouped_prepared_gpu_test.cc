// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Operation-level proof only. Real route -> completed host validation ->
// selected artifact closure -> existing RunGrouped, without an engine bridge.
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "artifact_writer.h"
#include "base/bytes.h"
#include "base/sha256.h"
#include "engine/execution_contract.h"
#include "engine/paged_node.h"
#include "engine/paged_weights.h"
#include "expected_error.h"
#include "ir/graph.h"
#include "kernels/common/grouped_worklist.h"
#include "kernels/decode/grouped_binding.h"
#include "kernels/decode/products.h"
#include "kernels/ggml/launch.h"
#include "kernels/ggml/llmp_ops.h"
#include "kernels/ggml/moe_layout.h"
#include "kernels/ggml/tensors.h"
#include "platform/host_probe.h"

namespace {
namespace en = llmp::engine;
namespace kn = llmp::kernels;
namespace de = llmp::kernels::decode;
namespace kg = llmp::kernels::ggml;
namespace fmt = llmp::format;
namespace ca = llmp::catalog;
namespace sc = llmp::scheduler;
namespace pr = llmp::providers;
namespace ts = llmp::test;
using llmp::base::Bytes;
using Role = fmt::PartRole;
constexpr std::size_t kE = 4, kU = 2, kT = 3, kP = kU * kT, kK = 32, kN = 16;
constexpr std::uint64_t kDomain = 256ULL << 20, kMetadata = 16ULL << 20;
constexpr std::uint64_t kProcessAndCacheCap = 10ULL << 30, kPhysicalMargin = 6ULL << 30;
constexpr std::uint64_t kIds = 0, kRoute = 256, kOffsets = 512, kRows = 768, kSlots = 1024,
                        kExperts = 1280, kX = 1536, kY = 2304;
constexpr kg::moe::RouteLayout kRouteLayout{kE, kP, 1};
static_assert(kRouteLayout.ints() == 32);
static_assert(sizeof(de::Operand) * kE <= kX - kExperts);
static_assert(kX + kP * kK * sizeof(float) == kY);
static_assert(kY + kP * kN * sizeof(float) < 4096);
bool retirement_failed = false;

fmt::TensorFormat Matrix() {
  return {.encoding = fmt::Encoding::kNvfp4,
          .params = {.tensor_scale = true},
          .layout = fmt::Layout::kSplit,
          .shape = llmp::ir::ShapeOf({kK, kN}),
          .stacked = 2};
}
fmt::TensorFormat AllExperts() {
  auto f = Matrix();
  f.shape = llmp::ir::ShapeOf({kK, kN, kE});
  return f;
}
en::Status Error(std::string message) { return std::unexpected(std::move(message)); }

// Two small independent values per block and two distinct scale halves.
float Weight(std::size_t expert, std::size_t row, std::size_t k) {
  const float code = ((expert + row + k) % 2 == 0) ? 0.5F : 1.F;
  return code * (k < 16 ? 1.F : 2.F) * static_cast<float>(expert + 1) * (row < 8 ? 1.F : 2.F);
}
std::filesystem::path Artifact(const std::filesystem::path& root) {
  ts::V1Plan plan;
  plan.expert_count = kE;
  plan.spec = "{\"pipeline\":\"llm\",\"stages\":[],\"version\":1}\n";
  plan.groups = {
      {.kind = llmp::artifact::GroupKind::kExpert,
       .layer = 0,
       .resources = {},
       .arrays = {{.name = "codes", .format = Matrix(), .part = Role::kCodes, .offset = 0},
                  {.name = "scales", .format = Matrix(), .part = Role::kBlockScales, .offset = 256},
                  {.name = "scale", .format = Matrix(), .part = Role::kTensorScale, .offset = 512}},
       .new_shard = false}};
  auto files = ts::Render(plan);
  if (files.shards.size() != 1) return {};
  auto& shard = files.shards.front();
  std::uint64_t header = 0;
  std::memcpy(&header, shard.data(), sizeof(header));
  if (header + 8 > shard.size() || shard.size() - header - 8 != kE * 4096) return {};
  const auto start = header + 8;
  for (std::size_t e = 0; e != kE; ++e) {
    auto* data = shard.data() + start + e * 4096;
    for (std::size_t n = 0; n != kN; ++n)
      for (std::size_t k = 0; k != kK; k += 2)
        data[n * (kK / 2) + k / 2] = static_cast<char>((e + n) % 2 == 0 ? 0x21 : 0x12);
    for (std::size_t n = 0; n != kN; ++n) {
      data[256 + n * 2] = 0x38;
      data[256 + n * 2 + 1] = 0x40;
    }
    const std::array<float, 2> scales{static_cast<float>(e + 1), 2.F * static_cast<float>(e + 1)};
    std::memcpy(data + 512, scales.data(), sizeof(scales));
    llmp::base::Sha256 hash;
    hash.Update(std::string_view(data, 4096));
    files.index["chunk_sha256"].at(e) = llmp::base::ToHex(hash.Finish());
  }
  return ts::Publish(root, files);
}

// Complete owner, never a production API. Only a successfully prepared token
// of this owner can launch; a token does not create a catalog lease.
class PreparedGrouped final : public en::PagedModel {
 public:
  struct Token {
   private:
    friend class PreparedGrouped;
    const PreparedGrouped* owner = nullptr;
    std::uint64_t epoch = 0;
  };
  // Inline scheduler lanes keep catalog validation on its sole writer thread;
  // actual CUDA/storage submissions and completed scheduler leases are unchanged.
  en::PagedNode node{
      {.compute_streams = 1, .slots = 1, .inline_lanes = true, .slot_bytes = en::kSlabSlotBytes}};
  bool opened = false;
  std::filesystem::path root;
  std::shared_ptr<en::ExecutionMetadataAllowance> payment;
  en::PagedWeights weights;
  std::unique_ptr<kg::LaunchContext> launch;
  std::optional<kg::TensorArena> arena;
  std::optional<kg::CapturedGraph> graph;
  ggml_tensor* ids_tensor = nullptr;
  ggml_tensor* route_tensor = nullptr;
  void* host = nullptr;
  ca::Closure runtime, current;
  std::vector<ca::ExtentId> staging;
  std::array<ca::ExtentId, kE> expert_extents{};
  std::array<std::int32_t, kP> ids{}, rows{}, slots{};
  std::array<std::int32_t, kE + 1> offsets{};
  std::array<std::uint8_t, kP> seen{};
  std::array<std::int32_t, 32> route{};
  std::array<de::Operand, kE> operands{};
  std::array<de::PitchedExpertPart, 3> parts{};
  std::array<float, kP * kK> x{};
  std::array<float, kP * kN> y{};
  std::uint64_t epoch = 0, basis = 0, route_jobs = 0, uploads = 0, product_jobs = 0, captures = 0,
                replays = 0;
  std::array<std::uint64_t, 8> prepared_backing{};
  const kg::LaunchContext* prepared_launch = nullptr;
  std::uint64_t prepared_activation = 0, prepared_pool = 0;
  bool ready = false, faulted = false;

  bool Current() const {
    if (!ready || faulted || prepared_launch != launch.get() ||
        prepared_activation != node.activations().base || prepared_pool != node.pool().base ||
        current.extents.size() > prepared_backing.size())
      return false;
    for (std::size_t i = 0; i != current.extents.size(); ++i) {
      const auto [id, content] = current.extents[i];
      const auto actual = node.catalog().Describe(id);
      if (!actual || actual->content_generation != content ||
          actual->backing_generation != prepared_backing[i] ||
          actual->state != ca::ExtentState::kResident)
        return false;
    }
    return true;
  }
  en::Status Fault(en::Status status) {
    if (!status) {
      ready = false;
      faulted = true;
    }
    return status;
  }

  std::uint32_t stream() const override { return 0; }
  const ca::Closure& fence_closure() const override { return runtime; }
  std::vector<ca::ExtentId> managed_extents() const override { return weights.extents(); }
  en::Status Release() override {
    graph.reset();
    launch.reset();
    arena.reset();
    return weights.Release(node.memory());
  }
  template <class T>
  T* Device(std::uint64_t at) const {
    return reinterpret_cast<T*>(node.activations().base + at);
  }
  en::Status Upload(std::uint64_t at, const void* p, std::size_t bytes) {
    if (bytes > 4096) return Error("upload exceeds paid staging");
    ++uploads;
    std::memcpy(host, p, bytes);
    return Fault(
        node.Copy(0, runtime, node.activations().base + at, host, bytes, false, "grouped upload"));
  }
  en::Status Download(std::uint64_t at, void* p, std::size_t bytes) {
    if (bytes > 4096) return Error("readback exceeds paid staging");
    if (auto done = node.Copy(0, runtime, node.activations().base + at, host, bytes, true,
                              "grouped readback");
        !done)
      return Fault(std::move(done));
    std::memcpy(p, host, bytes);
    return {};
  }
  en::Status Setup(std::uint64_t input_basis) {
    basis = input_basis;
    const auto bound = kn::GroupedRouteBound({kE, kU, kT, basis});
    const auto descriptor_bound = de::GroupedBindingBound(AllExperts());
    if (!bound || !descriptor_bound || bound->inputs.value() != 116 ||
        bound->outputs.value() != 68 || bound->scratch.value() != 6 ||
        descriptor_bound->descriptors.value() != sizeof(operands))
      return Error("fixed helper bound");
    const auto available = llmp::platform::AvailableMemoryBytes();
    if (!available || *available < kProcessAndCacheCap + kPhysicalMargin)
      return Error("physical margin");
    if (auto r = node.Open(); !r) return r;
    opened = true;
    node.SetHostFloor(0);
    auto paid = en::ExecutionMetadataAllowance::Admit(node.catalog(), node.domain(), kMetadata,
                                                      *available, 6ULL << 30);
    if (!paid) return Error(paid.error());
    payment = *paid;
    if (!payment->Charge(8ULL << 20)) return Error("fixed owner/metadata payment");
    if (payment->Charge(kMetadata)) return Error("metadata allowance accepted overpayment");
    root = std::filesystem::path("scratch") / (basis == 1 ? "grouped-token" : "grouped-routed");
    const auto path = Artifact(root);
    if (path.empty()) return Error("synthetic artifact geometry");
    if (auto r = weights.Open(path); !r) return r;
    const auto& artifact = weights.artifact();
    if (artifact.groups().size() != kE || artifact.expert_arrays().size() != 3)
      return Error("complete source inventory");
    constexpr std::array names{"codes", "scales", "scale"};
    constexpr std::array roles{Role::kCodes, Role::kBlockScales, Role::kTensorScale};
    constexpr std::array<std::uint64_t, 3> at{0, 256, 512}, bytes{256, 32, 8};
    for (std::size_t p = 0; p != 3; ++p) {
      const auto index = artifact.FindExpertArray(names[p]);
      if (!index) return Error("missing source part");
      const auto& a = artifact.expert_arrays()[*index];
      if (!a.format || *a.format != Matrix() || a.part != roles[p] || a.count != kE ||
          a.first_group != 0 || a.layer != 0 || a.group_offset.value() != at[p] ||
          a.slice_bytes.value() != bytes[p] || a.readable.value() != bytes[p])
        return Error("source part identity/geometry");
    }
    std::array<en::GroupPlace, kE> place{};
    const std::array<en::SlabSpec, 1> slab{
        {{.first_group = 0, .count = kE, .stride = en::kPagedExtent}}};
    if (auto r = weights.Reserve(node, place, slab); !r) return r;
    if (weights.extents().size() != kE || weights.bytes() != kE * en::kPagedExtent ||
        weights.group_address(0) % en::kPagedExtent != 0)
      return Error("source slab extent map");
    for (std::size_t e = 0; e != kE; ++e) {
      expert_extents[e] = weights.extents()[e];
      const auto described = node.catalog().Describe(expert_extents[e]);
      if (!described || described->descriptor.content.group != 0 ||
          described->descriptor.content.chunk != e ||
          llmp::base::ToHex(described->descriptor.content.artifact) != artifact.id() ||
          weights.group_address(static_cast<std::uint32_t>(e)) !=
              weights.group_address(0) + e * en::kPagedExtent)
        return Error("source extent identity");
    }
    for (std::size_t p = 0; p != 3; ++p)
      parts[p] = {roles[p], weights.group_address(0) + at[p],
                  (kE - 1) * en::kPagedExtent + bytes[p], en::kPagedExtent};
    if (!de::BindGroupedInto(AllExperts(), parts, operands))
      return Error("owned multipart binding");
    if (auto r = node.MapWorkspace(en::kPagedExtent, en::kPagedExtent); !r) return r;
    if (ggml_tensor_overhead() > (4096 - 4 * sizeof(ggml_tensor*)) / 2)
      return Error("two-descriptor allocation bound");
    auto tensors = kg::TensorArena::Create(2);
    if (!tensors) return Error(tensors.error().detail);
    arena.emplace(std::move(*tensors));
    if (arena->bytes() > 4096) return Error("two-descriptor bound");
    if (auto room = arena->Reserve(2); !room) return Error(room.error().detail);
    ids_tensor = ggml_new_tensor_2d(arena->context(), GGML_TYPE_I32, kU, kT);
    kg::TensorArena::Bind(ids_tensor, node.activations().base + kIds);
    route_tensor = kg::MoeRoute(arena->context(), ids_tensor, kE);
    kg::TensorArena::Bind(route_tensor, node.activations().base + kRoute);
    auto l = kg::LaunchContext::Create(0, node.execution(), node.stream(0),
                                       {node.pool().base, Bytes(node.pool().bytes)});
    if (!l) return Error(l.error().detail);
    launch = std::move(*l);
    auto pin = node.Pinned(4096, 0, staging);
    if (!pin) return Error(pin.error());
    host = *pin;
    if (auto r = node.Start(Bytes(kDomain)); !r) return r;
    if (auto r = weights.Register(node, 0); !r) return r;
    node.Run();
    std::vector<ca::ExtentId> extents = node.activations().extents;
    extents.insert(extents.end(), node.pool().extents.begin(), node.pool().extents.end());
    extents.insert(extents.end(), staging.begin(), staging.end());
    auto closure = node.catalog().ClosureOfExtents(extents);
    if (!closure) return Error("runtime closure");
    runtime = *closure;
    return Upload(kExperts, operands.data(), sizeof(operands));
  }

  std::expected<Token, std::string> Prepare(const std::array<std::int32_t, kP>& next) {
    if (faulted) return std::unexpected("prepared owner has uncertain completion");
    // Invalid IDs refuse before route kernels, I/O, descriptor uploads or epoch changes.
    for (auto id : next)
      if (id < 0 || id >= static_cast<std::int32_t>(kE))
        return std::unexpected("invalid selected expert");
    if (epoch == std::numeric_limits<std::uint64_t>::max())
      return std::unexpected("epoch exhausted");
    ready = false;
    ids = next;
    if (auto r = Upload(kIds, ids.data(), sizeof(ids)); !r) return std::unexpected(r.error());
    auto routed = node.Job(
        runtime,
        [&](pr::NativeStream) {
          return kg::RunMoeRoute(*launch, route_tensor) ? sc::JobResult::kQueued
                                                        : sc::JobResult::kUnknown;
        },
        "actual route", 0);
    if (!routed) {
      faulted = true;
      return std::unexpected(routed.error());
    }
    ++route_jobs;
    if (auto r = Download(kRoute, route.data(), sizeof(route)); !r)
      return std::unexpected(r.error());
    const auto span = std::span<const std::int32_t>(route);
    auto valid = kn::ValidateGroupedRouteInto(
        {kE, kU, kT, basis},
        {ids, span.first(kE + 1), span.subspan(kRouteLayout.position(), kP),
         span.subspan(kRouteLayout.token(), kP), span.subspan(kRouteLayout.expert(), kP)},
        {offsets, rows, slots, seen});
    if (!valid) return std::unexpected("completed route validation");
    std::vector<ca::ExtentId> selected;
    for (const auto& [id, generation] : runtime.extents) {
      (void)generation;
      selected.push_back(id);
    }
    for (std::size_t e = 0; e != kE; ++e)
      if (offsets[e] != offsets[e + 1]) selected.push_back(expert_extents[e]);
    auto closure = node.catalog().ClosureOfExtents(selected);
    if (!closure) return std::unexpected("selected closure");
    if (closure->extents.size() > prepared_backing.size())
      return std::unexpected("selected closure exceeds fixed inventory");
    current = *closure;
    sc::AcquireReport report;
    if (auto r = node.Acquire(current, report, "selected experts only"); !r) {
      faulted = true;
      return std::unexpected(r.error());
    }
    if (auto r = Upload(kOffsets, offsets.data(), sizeof(offsets)); !r)
      return std::unexpected(r.error());
    if (auto r = Upload(kRows, rows.data(), sizeof(rows)); !r) return std::unexpected(r.error());
    if (auto r = Upload(kSlots, slots.data(), sizeof(slots)); !r) return std::unexpected(r.error());
    for (std::size_t row = 0; row != kP; ++row)
      for (std::size_t k = 0; k != kK; ++k)
        x[row * kK + k] = static_cast<float>((row + k + epoch) % 5) - 2.F;
    if (auto r = Upload(kX, x.data(), sizeof(x)); !r) return std::unexpected(r.error());
    for (std::size_t i = 0; i != current.extents.size(); ++i) {
      const auto [id, content] = current.extents[i];
      const auto actual = node.catalog().Describe(id);
      if (!actual || actual->content_generation != content ||
          actual->state != ca::ExtentState::kResident)
        return std::unexpected("completed preparation lost its source closure");
      prepared_backing[i] = actual->backing_generation;
    }
    prepared_activation = node.activations().base;
    prepared_pool = node.pool().base;
    prepared_launch = launch.get();
    ++epoch;
    ready = true;
    Token token;
    token.owner = this;
    token.epoch = epoch;
    return token;
  }

  en::Status Run(Token token, bool record) {
    if (token.owner != this || token.epoch != epoch || !Current())
      return Error("stale prepared owner/epoch/source/backing");
    const auto product = [&](pr::NativeStream stream) {
      return de::RunGrouped({.format = Matrix(),
                             .experts = Device<de::Operand>(kExperts),
                             .experts_count = kE,
                             .first = 0,
                             .outputs = kN,
                             .offsets = Device<std::int32_t>(kOffsets),
                             .pair_rows = Device<std::int32_t>(kRows),
                             .pair_out = Device<std::int32_t>(kSlots),
                             .pairs = kP,
                             .x = Device<float>(kX),
                             .x_rows = kT * basis,
                             .x_stride = kK,
                             .y = Device<float>(kY),
                             .y_rows = kP,
                             .y_stride = kN,
                             .rows_hint = kP},
                            stream.handle);
    };
    // Job reacquires this exact generation-bound closure and retains its lease
    // through its completed fence. Acquire alone above was not a lease.
    auto done = node.Job(
        current,
        [&](pr::NativeStream stream) {
          // Recheck after the scheduler acquires the closure; a backing change
          // between preparation and admission cannot authorize a stale token.
          if (!Current()) return sc::JobResult::kUnknown;
          if (record) {
            if (graph) return sc::JobResult::kUnknown;
            auto captured =
                launch->Capture([&](kg::LaunchContext&) -> std::expected<void, kg::KernelFailure> {
                  auto ran = product(stream);
                  if (!ran)
                    return std::unexpected(kg::KernelFailure{.error = kg::KernelError::kUnknown,
                                                             .detail = ran.error()});
                  return {};
                });
            if (!captured) return sc::JobResult::kUnknown;
            graph.emplace(std::move(*captured));
            ++captures;
          }
          if (graph) {
            if (!launch->Launch(*graph)) return sc::JobResult::kUnknown;
            ++replays;
          } else if (!product(stream))
            return sc::JobResult::kUnknown;
          return sc::JobResult::kQueued;
        },
        "prepared grouped product", 0);
    if (!done) return Fault(std::move(done));
    ++product_jobs;
    if (auto r = Download(kY, y.data(), sizeof(y)); !r) return r;
    for (std::size_t slot = 0; slot != kP; ++slot)
      for (std::size_t n = 0; n != kN; ++n) {
        float expected = 0;
        const auto input_row = basis == 1 ? slot / kU : slot;
        for (std::size_t k = 0; k != kK; ++k)
          expected += Weight(static_cast<std::size_t>(ids[slot]), n, k) * x[input_row * kK + k];
        if (std::bit_cast<std::uint32_t>(expected) !=
            std::bit_cast<std::uint32_t>(y[slot * kN + n]))
          return Error("grouped output differs from complete independent representable oracle");
      }
    return {};
  }
};
static_assert(sizeof(PreparedGrouped) < 65536);
struct Retire {
  void operator()(PreparedGrouped* p) const {
    if (!p) return;
    if (p->opened) {
      std::array<en::PagedModel*, 1> models{p};
      const auto retired = p->node.TearDown(models);
      EXPECT_TRUE(retired) << llmp::test_support::Failed(retired).value_or("");
      if (!retired) {
        retirement_failed = true;
        return;
      }
    }
    std::error_code error;
    if (!p->root.empty()) std::filesystem::remove_all(p->root, error);
    EXPECT_FALSE(error) << error.message();
    delete p;
  }
};
void RunCase(std::uint64_t basis) {
  ASSERT_FALSE(retirement_failed);
  const auto available = llmp::platform::AvailableMemoryBytes();
  ASSERT_TRUE(available);
  ASSERT_GE(*available, kProcessAndCacheCap + kPhysicalMargin);
  std::unique_ptr<PreparedGrouped, Retire> owner(new PreparedGrouped);
  const auto setup = owner->Setup(basis);
  ASSERT_TRUE(setup) << llmp::test_support::Failed(setup).value_or("");
  const std::array<std::array<std::int32_t, kP>, 3> ids{
      {{2, 2, 0, 3, 2, 0}, {1, 0, 1, 2, 0, 1}, {3, 1, 3, 0, 1, 0}}};
  std::optional<PreparedGrouped::Token> old;
  for (std::size_t turn = 0; turn != ids.size(); ++turn) {
    const auto requests = owner->node.requests();
    const auto epoch = owner->epoch;
    const auto routes = owner->route_jobs, uploads = owner->uploads;
    auto invalid = ids[turn];
    invalid[2] = -1;
    EXPECT_FALSE(owner->Prepare(invalid));
    invalid[2] = kE;
    EXPECT_FALSE(owner->Prepare(invalid));
    ASSERT_EQ(owner->epoch, epoch);
    ASSERT_EQ(owner->node.requests(), requests);
    ASSERT_EQ(owner->route_jobs, routes);
    ASSERT_EQ(owner->uploads, uploads);
    const auto token = owner->Prepare(ids[turn]);
    ASSERT_TRUE(token) << llmp::test_support::Failed(token).value_or("");
    if (old) EXPECT_FALSE(owner->Run(*old, false));
    // A same-address descriptor/closure is insufficient if its backing epoch
    // differs. Corrupt only the local receipt; no catalog mutation or work.
    const auto backing = owner->prepared_backing[0];
    owner->prepared_backing[0] ^= 1;
    EXPECT_FALSE(owner->Run(*token, false));
    owner->prepared_backing[0] = backing;
    const auto ran = owner->Run(*token, turn == 1);
    ASSERT_TRUE(ran) << llmp::test_support::Failed(ran).value_or("");
    old = *token;
    if (turn == 0) {
      const auto unused = owner->node.catalog().Describe(owner->expert_extents[1]);
      ASSERT_TRUE(unused);
      EXPECT_EQ(unused->state, ca::ExtentState::kNonresident);
    }
  }
  EXPECT_EQ(owner->product_jobs, 3U);
  EXPECT_EQ(owner->captures, 1U);
  EXPECT_EQ(owner->replays, 2U);
  ::testing::Test::RecordProperty("checked_output_values", 3 * kP * kN);
  ::testing::Test::RecordProperty("route_jobs", static_cast<int>(owner->route_jobs));
  ::testing::Test::RecordProperty("product_jobs", static_cast<int>(owner->product_jobs));
  ::testing::Test::RecordProperty("captures", static_cast<int>(owner->captures));
  ::testing::Test::RecordProperty("replays", static_cast<int>(owner->replays));
}
TEST(PreparedGroupedGpu, TokenRowsUseOnlySelectedExpertsAndReplayChangedRouting) { RunCase(1); }
TEST(PreparedGroupedGpu, RoutedRowsPreserveOriginalSlotsAndRepeatedContributions) { RunCase(kU); }
}  // namespace
