// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "artifact_writer.h"
#include "base/sha256.h"
#include "engine/batch_resources.h"
#include "engine/draft_companions.h"
#include "engine/execution_contract.h"
#include "engine/runner_owner.h"
#include "engine/support.h"
#include "expected_error.h"
#include "fake_step_source.h"
#include "kernels/common/lower.h"
#include "kernels/common/recording_launch.h"
#include "kernels/common/routed_program.h"
#include "providers/fake/fake_device_execution.h"
#include "providers/fake/fake_device_runtime.h"

namespace {
namespace en = llmp::engine;
namespace md = llmp::model;
namespace kn = llmp::kernels;
namespace pr = llmp::providers;
namespace pl = llmp::planner;
namespace ir = llmp::ir;
namespace fm = llmp::format;
namespace ar = llmp::artifact;
namespace ts = llmp::test_support;
using Status = std::expected<void, kn::StepFailure>;
constexpr auto kExtent = en::kPagedExtent;
constexpr std::string_view kArray = "expert.checksum";

std::unexpected<kn::StepFailure> Rejected(std::string text) {
  return std::unexpected(kn::StepFailure{.detail = std::move(text)});
}
std::unexpected<kn::StepFailure> Unknown(std::string text) {
  return std::unexpected(
      kn::StepFailure{.error = kn::StepError::kUnknown, .detail = std::move(text)});
}
struct RouteFacts {
  bool fail_final_complete = false;
  bool refuse_future_build = false, change_future_identity = false, refuse_recording = false;
  bool base_leased_route = false;
  std::uint32_t numeric = 0, source_reads = 0, prefix_completions = 0, final_completions = 0;
  std::uint32_t builds = 0, companion_builds = 0, recordings = 0, replays = 0;
  std::uint32_t private_builds = 0, private_views = 0;
  std::uint64_t measured_host = 0, future_host = 0;
  float consumed = 0;
  bool source_valid = false;
  std::function<void(bool)> after_retirement;
};

class TailRecording final : public kn::RoutedRecording {
 public:
  TailRecording(const kn::RoutedProgram* owner, std::size_t segment,
                std::unique_ptr<kn::Recorded> graph)
      : owner_(owner), segment_(segment), graph_(std::move(graph)) {}
  const kn::RoutedProgram* owner() const override { return owner_; }
  std::size_t segment() const override { return segment_; }
  std::size_t nodes() const override { return graph_->nodes(); }
  double capture_seconds() const override { return graph_->capture_seconds(); }
  double instantiate_seconds() const override { return graph_->instantiate_seconds(); }
  Status Replay(const kn::RoutedAccess& access, kn::StepLaunch& launch,
                pr::NativeStream stream) const override {
    if (!access.Names(owner_, access.epoch()) || access.segment() != segment_)
      return Rejected("recording access differs");
    return launch.Launch(stream, *graph_);
  }

 private:
  const kn::RoutedProgram* owner_;
  std::size_t segment_;
  std::unique_ptr<kn::Recorded> graph_;
};

// An actual two-segment native program. Prefix writes the route readback;
// Prepare checks its complete bijection; the selected-closure continuation
// writes target state/taps and consumes F32 data from the selected artifact slice.
// No caller fabricates RoutedAccess or a retirement certificate.
class TailProgram final : public kn::StepProgram, public kn::RoutedProgram {
 public:
  TailProgram(std::unique_ptr<kn::StepProgram> inner, kn::ExpertPartBinding part,
              std::uint64_t activations, std::shared_ptr<RouteFacts> facts,
              llmp::base::Sha256Digest identity, bool routed)
      : inner_(std::move(inner)),
        part_(part),
        facts_(std::move(facts)),
        identity_(identity),
        routed_(routed) {
    const auto end = en::support::Round(inner_->scratch_bytes(), 256);
    scratch_ = end + 256;
    route_device_ = activations == 0 ? 0 : activations + end;
    // Measuring has no activation address, but owns the same descriptors as
    // execution. Reserve the complete inventory before filling it in either arm.
    uses_.reserve(inner_->tensors().size() + (routed_ ? 1U : 0U));
    uses_.assign(inner_->tensors().begin(), inner_->tensors().end());
    if (routed_) uses_.push_back({.address = route_device_, .bytes = 28});
    // The reader is the opened matrix format with the actual expert axis,
    // preserving encoding/layout/part identity instead of an unrelated shape.
    auto array_format = part_.format;
    array_format.shape.rank = 3;
    array_format.shape.ne[2] = 2;
    reads_.reserve(2);
    reads_.push_back({.ref = {.expert_array = true},
                      .format = array_format,
                      .parts = {{std::string(kArray), part_.role}},
                      .dynamic = true});
    if (facts_->base_leased_route) {
      reads_.push_back({.ref = {.component = 1, .expert_array = true},
                        .format = array_format,
                        .parts = {{std::string(kArray), part_.role}},
                        .dynamic = false});
    }
    layout_.shape = {.experts = 2, .used = 1, .tokens = 1, .input_lanes = 1};
    layout_.ids = 0;
    layout_.offsets = 4;
    layout_.position = 16;
    layout_.token = 20;
    layout_.expert = 24;
    layout_.work_offsets = 28;
    layout_.work_tokens = 40;
    layout_.work_slots = 44;
    layout_.seen = 48;
    layout_.host_bytes = 256;
    layout_.readback = {{.device = route_device_, .host_offset = 0, .bytes = 28}};
  }
  std::uint64_t scratch_bytes() const override { return scratch_; }
  std::uint64_t host_bytes() const override {
    // Inner descriptors, concrete wrapper/control, vector payloads, names and
    // seven allocations' malloc/control metadata. No payload-sized guess.
    std::uint64_t weight_bytes = 0;
    for (const auto& w : reads_) {
      weight_bytes += w.parts.capacity() * sizeof(w.parts[0]) + kArray.size() + 1;
    }
    return inner_->host_bytes() + sizeof(TailProgram) + uses_.capacity() * sizeof(kn::TensorUse) +
           layout_.readback.capacity() * sizeof(kn::RoutedCopy) +
           reads_.capacity() * sizeof(kn::RoutedWeight) + weight_bytes + 7 * 128;
  }
  std::uint64_t launches() const override { return inner_->launches() + 2; }
  std::span<const kn::InputSlot> inputs() const override { return inner_->inputs(); }
  std::span<const kn::OutputSlot> outputs() const override { return inner_->outputs(); }
  std::span<const kn::TensorUse> tensors() const override { return uses_; }
  std::optional<std::span<const kn::RoutedWeight>> weight_reads() const override {
    return routed_ ? std::span<const kn::RoutedWeight>(reads_)
                   : std::span<const kn::RoutedWeight>{};
  }
  kn::RoutedProgram* routed() override { return routed_ ? this : nullptr; }
  const kn::RoutedProgram* routed() const override { return routed_ ? this : nullptr; }
  Status Bind(kn::StepLaunch& launch) override { return inner_->Bind(launch); }
  Status Run(kn::StepLaunch& launch, pr::NativeStream native) const override {
    return routed_ ? Status{Rejected("route required")} : inner_->Run(launch, native);
  }
  std::expected<std::unique_ptr<kn::Recorded>, kn::StepFailure> Record(
      kn::StepLaunch& launch, pr::NativeStream native,
      const kn::StepLaunch::Body& body) const override {
    if (routed_) return Rejected("route required");
    return inner_->Record(launch, native, body);
  }
  std::span<const kn::RoutedWeight> weights() const override { return reads_; }
  std::size_t groups() const override { return 1; }
  const kn::RoutedLayout& layout(std::size_t = 0) const override { return layout_; }
  std::span<const std::size_t> members(std::size_t) const override { return member_; }
  std::uint64_t pinned_bytes() const override { return layout_.host_bytes; }
  const llmp::base::Sha256Digest& routing_identity() const override { return identity_; }
  const llmp::base::Sha256Digest& graph_identity() const override { return identity_; }
  Status RunSegment(const kn::RoutedAccess& access, kn::StepLaunch& launch,
                    pr::NativeStream native) override {
    if (!access.Names(this, access.epoch())) return Rejected("foreign route capability");
    if (access.segment() == 0 && access.epoch() != progress_.epoch() &&
        !progress_.Start(access.epoch(), 1))
      return Rejected("route start differs");
    if (!progress_.Ready(access.epoch(), access.segment()) ||
        (!recording_ && !progress_.Submit(access.epoch(), access.segment())))
      return Rejected("route order differs");
    if (access.segment() == 0) {
      const auto queued = pr::fake::FakeDeviceExecution::Launch(native, [address = route_device_] {
        // Expert 1, one sorted token at original slot 0; expert 0 is empty.
        constexpr std::array<std::int32_t, 7> route{1, 0, 0, 1, 0, 0, 1};
        std::memcpy(en::support::Pointer(address), route.data(), sizeof(route));
      });
      return queued ? Status{} : Status{Unknown("prefix native queue refused")};
    }
    if (auto numeric = inner_->Run(launch, native); !numeric) return numeric;
    const auto output = inner_->outputs().front().device_address;
    const auto queued =
        pr::fake::FakeDeviceExecution::Launch(native, [part = part_, output, facts = facts_] {
          const auto* source =
              static_cast<const float*>(en::support::Pointer(part.address + part.pitch));
          facts->consumed = source[0];  // Consume the opened F32 matrix's first element.
          ++facts->source_reads;
          ++facts->numeric;
          // The fixed published fixture has a finite positive integral first
          // weight. Check the conversion before using it in this integer head.
          facts->source_valid =
              std::isfinite(facts->consumed) && facts->consumed >= 0 &&
              static_cast<double>(facts->consumed) <=
                  static_cast<double>(std::numeric_limits<std::int32_t>::max()) - 73;
          if (!facts->source_valid) return;
          auto* row = static_cast<ts::FakeRow*>(en::support::Pointer(output));
          row->token += static_cast<std::int32_t>(facts->consumed);
        });
    return queued ? Status{} : Status{Unknown("selected native queue refused")};
  }
  Status CompleteSegment(const kn::RoutedAccess& access) override {
    if (!access.Names(this, access.epoch()) ||
        !progress_.Complete(access.epoch(), access.segment()))
      return Rejected("completion route order differs");
    const bool final = access.segment() == 1;
    if (final)
      ++facts_->final_completions;
    else
      ++facts_->prefix_completions;
    // Invoked only by the real runner after its node reports retirement.
    if (facts_->after_retirement) facts_->after_retirement(final);
    if (final && facts_->fail_final_complete) return Rejected("retired final host refusal");
    return {};
  }
  Status Prepare(const kn::RoutedAccess& access, std::span<std::byte> pinned) override {
    if (!access.Names(this, access.epoch()) || !progress_.CanPrepare(access.epoch(), 0) ||
        pinned.size() != layout_.host_bytes)
      return Rejected("prepare capability or storage differs");
    const auto integers = [&](std::uint64_t offset, std::size_t count) {
      return std::span(reinterpret_cast<std::int32_t*>(pinned.data() + offset), count);
    };
    const auto valid = kn::ValidateGroupedRouteInto(
        layout_.shape,
        {.ids = integers(layout_.ids, 1),
         .offsets = integers(layout_.offsets, 3),
         .position = integers(layout_.position, 1),
         .token = integers(layout_.token, 1),
         .expert = integers(layout_.expert, 1)},
        {.offsets = integers(layout_.work_offsets, 3),
         .rows = integers(layout_.work_tokens, 1),
         .original_slots = integers(layout_.work_slots, 1),
         .seen = {reinterpret_cast<std::uint8_t*>(pinned.data() + layout_.seen), 1}});
    if (!valid || !progress_.Prepared(access.epoch(), 0)) return Rejected("invalid real worklist");
    return {};
  }
  std::expected<std::unique_ptr<kn::RoutedRecording>, kn::StepFailure> RecordSegment(
      const kn::RoutedAccess& access, kn::StepLaunch& launch, pr::NativeStream native,
      const kn::StepLaunch::Body& body) override {
    if (!access.Names(this, access.epoch())) return Rejected("foreign recording access");
    if (access.segment() == 0 && access.epoch() != progress_.epoch() &&
        !progress_.Start(access.epoch(), 1))
      return Rejected("recording route start differs");
    if (!progress_.Ready(access.epoch(), access.segment()))
      return Rejected("recording requires current prepared route");
    if (facts_->refuse_recording) return Rejected("current routed recording refused");
    recording_ = true;
    auto graph = launch.Capture(native, body);
    recording_ = false;
    if (!graph) return std::unexpected(graph.error());
    ++facts_->recordings;
    return std::make_unique<TailRecording>(this, access.segment(), std::move(*graph));
  }
  Status ReplaySegment(const kn::RoutedAccess& access, kn::StepLaunch& launch,
                       pr::NativeStream native, const kn::RoutedRecording& graph) override {
    if (!access.Names(this, access.epoch()) || graph.owner() != this ||
        graph.segment() != access.segment())
      return Rejected("foreign replay access");
    if (access.segment() == 0 && access.epoch() != progress_.epoch() &&
        !progress_.Start(access.epoch(), 1))
      return Rejected("replay route start differs");
    if (!progress_.Submit(access.epoch(), access.segment()))
      return Rejected("replay order differs");
    auto queued = graph.Replay(access, launch, native);
    if (queued) ++facts_->replays;
    return queued;
  }

 private:
  std::unique_ptr<kn::StepProgram> inner_;
  kn::ExpertPartBinding part_;
  std::shared_ptr<RouteFacts> facts_;
  llmp::base::Sha256Digest identity_;
  std::uint64_t route_device_ = 0, scratch_ = 0;
  std::vector<kn::TensorUse> uses_;
  std::vector<kn::RoutedWeight> reads_;
  std::array<std::size_t, 1> member_{0};
  kn::RoutedLayout layout_;
  kn::RoutedProgress progress_;
  bool recording_ = false;
  bool routed_;
};

class RoutedBatchBackend final : public kn::BackendOwner {
 public:
  RoutedBatchBackend(std::shared_ptr<en::ExecutionMetadataAllowance> paid,
                     std::shared_ptr<RouteFacts> facts)
      : paid_(std::move(paid)), facts_(std::move(facts)) {}
  const kn::StepBackend& backend() const override { return table_; }
  std::expected<llmp::execution::BackendDeclaration, std::string> Ready() const override {
    return llmp::execution::BackendDeclaration{};
  }
  bool Charge(std::uint64_t bytes) const override { return paid_->Charge(bytes); }
  void Release(std::uint64_t bytes) const override { paid_->Release(bytes); }

 private:
  class View final : public kn::ResourceInstance {
   public:
    explicit View(kn::ResourceOrigin origin) : ResourceInstance(std::move(origin)) {}
    kn::StepLaunch& launch() const override { return launch_; }
    std::expected<std::unique_ptr<kn::StepProgram>, std::string> Lower(
        const kn::LowerInput&) const override {
      return std::unexpected("actual source provides routed private lowering");
    }

   private:
    mutable kn::RecordingLaunch launch_;
  };
  std::expected<std::uint64_t, kn::ResourceFailure> ResourceBytes(
      kn::ResourceWorkspace) const override {
    return sizeof(View) + 64;
  }
  std::expected<kn::ResourceInstance*, kn::ResourceFailure> MakeResources(
      kn::ResourceOrigin origin, kn::ResourceWorkspace) const override {
    ++facts_->private_views;
    return new View(std::move(origin));
  }
  std::shared_ptr<en::ExecutionMetadataAllowance> paid_;
  std::shared_ptr<RouteFacts> facts_;
  kn::StepBackend table_{};
};

class TailSource final : public en::StepSource {
 public:
  TailSource(ts::FakeSourceOptions options, std::shared_ptr<RouteFacts> facts,
             std::shared_ptr<en::ExecutionMetadataAllowance> paid,
             llmp::base::Sha256Digest identity)
      : inner_(std::move(options)),
        facts_(std::move(facts)),
        paid_(std::move(paid)),
        identity_(identity) {}
  pl::PlanKey KeyBase() const override { return inner_.KeyBase(); }
  std::expected<std::optional<llmp::base::Sha256Digest>, std::string> ExecutionIdentity()
      const override {
    return std::optional{identity_};
  }
  en::Status CheckExecutionShape(const pl::StepShape& shape) const override {
    if (shape.segments.size() != 1 || shape.segments[0].slot != 0 || shape.segments[0].rows != 1 ||
        shape.segments[0].past >= 256 || shape.segments[0].read_cells != 256 ||
        shape.read_align != 256 || shape.segments[0].greedy || shape.segments[0].group != 0)
      return std::unexpected("source admits its measured one-row, one-owner geometry");
    const auto& segment = shape.segments[0];
    const bool target = shape.stage == 0 &&
                        (shape.phase == pl::Phase::kPrefill || shape.phase == pl::Phase::kScore) &&
                        segment.want_head && segment.outputs == 1;
    // TokenAfterCache requires exactly one real post companion after each
    // subsequent one-row target. The shared schedule supplies its ledger unit,
    // complete token history and preceding target tap; public calls cannot.
    const bool companion = shape.stage == 1 && shape.phase == pl::Phase::kDraft &&
                           !shape.row_invariant && !segment.want_head && segment.outputs == 0;
    if (!target && !companion)
      return std::unexpected("source admits only its measured target or draft companion");
    return {};
  }
  std::shared_ptr<const kn::BackendOwner> ResourceBackend(std::uint16_t stage) const override {
    return inner_.ResourceBackend(stage);
  }
  std::expected<std::unique_ptr<kn::StepProgram>, std::string> BuildPrivate(
      const pl::StepShape& shape, const kn::Addresses& addresses,
      const kn::ExecutionResources& resources) override {
    if (!ResourceBackend(shape.stage) || resources.parent() != ResourceBackend(shape.stage).get() ||
        !resources.Ready())
      return std::unexpected("private route lacks its actual source producer");
    auto built = Build(shape, addresses);
    if (!built) return built;
    if (!*built || !(**built).HoldResources(resources))
      return std::unexpected("private route cannot retain its producer-minted view");
    ++facts_->private_builds;
    return built;
  }
  std::expected<std::unique_ptr<kn::StepProgram>, std::string> Build(
      const pl::StepShape& shape, const kn::Addresses& addresses) override {
    if (auto valid = CheckExecutionShape(shape); !valid) return std::unexpected(valid.error());
    if (!addresses.measuring()) {
      if (shape.stage == 0)
        ++facts_->builds;
      else
        ++facts_->companion_builds;
      if (facts_->refuse_future_build && shape.row_invariant)
        return std::unexpected("optional future build refused");
    }
    const auto part = addresses.expert_part ? addresses.expert_part(0, kArray) : std::nullopt;
    if (!part) return std::unexpected("real expert source binding absent");
    const fm::TensorFormat expected_slice{.encoding = fm::Encoding::kF32,
                                          .shape = ir::ShapeOf({2, 2})};
    const auto extents = fm::PartsOf(expected_slice);
    if (!extents || part->format != expected_slice || part->role != fm::PartRole::kData ||
        part->pitch < (*extents)[0].readable || part->bytes != part->pitch + (*extents)[0].readable)
      return std::unexpected("opened two-expert matrix/part inventory differs");
    // One measured row: inner's declared4096, wrapper, exact maximum tensor
    // inventory (target cache + tap + token/position/output + route), containers,
    // and construction/shared-control allocator metadata, prepaid before Build.
    constexpr auto construction = 4096 + sizeof(TailProgram) + 8 * sizeof(kn::TensorUse) +
                                  4 * sizeof(kn::RoutedCopy) + 16 * 128;
    if (!paid_->Charge(construction)) return std::unexpected("test construction not paid");
    auto grant = std::shared_ptr<void>(new std::byte, [paid = paid_](void* allocation) {
      delete static_cast<std::byte*>(allocation);
      paid->Release(construction);
    });
    auto built = inner_.Build(shape, addresses);
    if (!built) return std::unexpected(built.error());
    const auto identity = facts_->change_future_identity && shape.row_invariant
                              ? llmp::base::Sha256().Update("unmeasured future route").Finish()
                              : identity_;
    auto program = std::make_unique<TailProgram>(
        std::move(*built), *part, addresses.activations, facts_, identity,
        shape.stage == 0 && shape.phase == pl::Phase::kPrefill);
    if (program->host_bytes() > construction || !program->HoldConstruction(std::move(grant)))
      return std::unexpected("test concrete source exceeded its construction payment");
    if (addresses.measuring())
      facts_->measured_host = std::max(facts_->measured_host, program->host_bytes());
    else if (shape.row_invariant)
      facts_->future_host = program->host_bytes();
    return program;
  }

 private:
  ts::FakeStepSource inner_;
  std::shared_ptr<RouteFacts> facts_;
  std::shared_ptr<en::ExecutionMetadataAllowance> paid_;
  llmp::base::Sha256Digest identity_;
};

struct TailEnvironment {
  en::PagedNode node{{.compute_streams = 1, .slots = 1, .slot_bytes = kExtent}};
  en::RunnerOwner runner;
  std::shared_ptr<en::ExecutionMetadataAllowance> paid;
  std::shared_ptr<RouteFacts> facts;
  std::filesystem::path directory;
  std::uint64_t owner_payment = 0;
};

class CompoundRoutedTest : public ::testing::Test {
 protected:
  std::shared_ptr<TailEnvironment> life_ = std::make_shared<TailEnvironment>();
  bool request_success_ = true;
  virtual bool Graphs() const { return false; }
  virtual bool PrivateResources() const { return false; }
  virtual std::uint32_t CapturePhases() const { return 0x1F; }
  virtual std::uint32_t AutomaticCapturePhases() const { return 0x1F; }
  virtual std::uint32_t FirstCapturePhases() const { return 0; }
  virtual bool BaseLeasedRoute() const { return false; }
  void SetUp() override {
    auto& life = *life_;
    ASSERT_TRUE(life.node.Open());
    life.node.SetHostFloor(0);
    // Exact source cardinality:2 artifact pages,3 shapes/1 signature,1 route,
    // 1 flight,1 wrapper plus inner. Add the production retained descriptor
    // bounds and constructor inventory, instead of a whole compiler allowance.
    // Two4KiB expert groups plus one4KiB safetensors header. Canonical
    // manifest/index publication and reopened descriptor copies have separate
    // 8KiB inventories, checked below before handing the source to the runner.
    constexpr std::uint64_t artifact_publication = 3 * 4096 + 2 * 8192;
    const std::uint64_t owner_bytes = sizeof(TailEnvironment) + sizeof(TailSource) +
                                      sizeof(RouteFacts) + artifact_publication + 2048 +
                                      (PrivateResources() ? sizeof(RoutedBatchBackend) + 128 : 0);
    constexpr std::uint64_t construction_bytes = 4096 + sizeof(TailProgram) +
                                                 8 * sizeof(kn::TensorUse) +
                                                 4 * sizeof(kn::RoutedCopy) + 16 * 128;
    const std::uint64_t source_bytes = owner_bytes + construction_bytes;
    constexpr std::uint64_t routing_bytes = 32768 + 2 * 256 + 8 * sizeof(kn::RoutedWeight) + 512 +
                                            8 * kArray.size() +
                                            12 * sizeof(llmp::base::Sha256Digest);
    constexpr std::uint64_t flight_bytes = 4096 + 8 * 128 + 2 * 128;
    constexpr std::uint64_t supplemental_bytes = 4096 + 4096 + 2 * 256;
    const auto allowance = en::ExecutionMetadataAllowance::Admit(
        life.node.catalog(), life.node.domain(),
        source_bytes + routing_bytes + flight_bytes + supplemental_bytes + 4096, 128 * kExtent,
        kExtent);
    ASSERT_TRUE(allowance) << ts::Failed(allowance).value_or("");
    life.paid = *allowance;
    life.owner_payment = owner_bytes;
    ASSERT_TRUE(life.paid->Charge(life.owner_payment));
    life.facts = std::make_shared<RouteFacts>();
    life.facts->base_leased_route = BaseLeasedRoute();
    // A deterministic shape-specific graph identity exists before setup. The
    // invariant variant is deliberately absent from ordinary route setup; its
    // actual successful private compilation must own its preparation authority.
    life.facts->change_future_identity = PrivateResources();
    const char* scratch = std::getenv("LLMP_TEST_SCRATCH");  // NOLINT(concurrency-mt-unsafe)
    ASSERT_NE(scratch, nullptr);
    life.directory = std::filesystem::path(scratch) /
                     ::testing::UnitTest::GetInstance()->current_test_info()->name();
    // ShapeOf canonicalizes1x1 to rank0. Use a genuine2x2 matrix because
    // production expert admission checks a rank2 slice against a rank3 array.
    const fm::TensorFormat slice{.encoding = fm::Encoding::kF32, .shape = ir::ShapeOf({2, 2})};
    const llmp::test::V1Plan artifact{
        .expert_count = 2,
        .groups = {{.kind = ar::GroupKind::kExpert,
                    .layer = 0,
                    .arrays = {{.name = std::string(kArray), .format = slice}}}}};
    auto weights = std::make_unique<en::PagedWeights>();
    const auto rendered = llmp::test::Render(artifact);
    ASSERT_EQ(rendered.shards.size(), 1U);
    ASSERT_LE(rendered.shards[0].size(), 3 * 4096U);
    ASSERT_LE(rendered.index.Text().size() + rendered.manifest.Text().size(), 8192U);
    ASSERT_TRUE(weights->Open(llmp::test::Publish(life.directory, rendered)));
    const auto identity = llmp::base::Sha256().Update(weights->artifact().id()).Finish();
    const std::array target{std::pair{
        0U, md::StateContract{.tensors = {{.name = "cache", .cell_bytes = sizeof(ts::FakeCell)}}}}};
    const std::array draft{std::pair{
        0U, md::StateContract{
                .tensors = {
                    {.name = "cache", .cell_bytes = sizeof(ts::FakeCell), .replayable = true},
                    {.name = "tap",
                     .kind = md::StateKind::kFeatures,
                     .encoding = fm::Encoding::kF32,
                     .cell_bytes = sizeof(ts::FakeCell),
                     .cells = md::CellRule::kPerRow,
                     .row_carry = 1,
                     .rollback = md::Rollback::kSnapshot,
                     .target_writes = true},
                    md::DraftProgressControl("progress", md::DraftRelation::kTokenAfterCache)}}}};
    auto target_layout = md::Layout(target, 256, 1), draft_layout = md::Layout(draft, 256, 1);
    ASSERT_TRUE(target_layout);
    ASSERT_TRUE(draft_layout);
    ts::FakeSourceOptions source_options;
    source_options.layouts = {*target_layout, *draft_layout};
    source_options.spec = identity;
    if (PrivateResources())
      source_options.private_backend = std::make_shared<RoutedBatchBackend>(life.paid, life.facts);
    auto source =
        std::make_unique<TailSource>(std::move(source_options), life.facts, life.paid, identity);
    en::RunnerConfig config;
    config.slots = {.model = "routed-tail",
                    .slots = 1,
                    .stages = {{.name = "target", .layout = *target_layout},
                               {.name = "next", .layout = *draft_layout}},
                    .spec = identity,
                    .spill_place = {},
                    .spill_directory = life.directory,
                    .invalidate = {},
                    .drop_plans = {}};
    config.weights.push_back({.weights = std::move(weights),
                              .places = {en::GroupPlace::kNone, en::GroupPlace::kNone},
                              .slabs = {{.first_group = 0, .count = 2, .stride = kExtent}}});
    config.setup_shapes = {
        {.phase = pl::Phase::kPrefill,
         .segments = {{.slot = 0, .rows = 1, .past = 0, .read_cells = 256, .outputs = 1}},
         .read_align = 256},
        {.phase = pl::Phase::kScore,
         .segments = {{.slot = 0, .rows = 1, .past = 0, .read_cells = 256, .outputs = 1}},
         .read_align = 256},
        {.stage = 1,
         .phase = pl::Phase::kDraft,
         .segments = {{.slot = 0,
                       .rows = 1,
                       .past = 0,
                       .read_cells = 256,
                       .outputs = 0,
                       .want_head = false}},
         .read_align = 256}};
    config.graphs = Graphs();
    config.capture_phases = CapturePhases();
    config.automatic_capture_phases = AutomaticCapturePhases();
    config.capture_first_phases = FirstCapturePhases();
    config.routed_metadata = life.paid;
    config.routed_owner = life_;
    config.launch =
        [](std::uint64_t) -> std::expected<std::unique_ptr<kn::StepLaunch>, std::string> {
      return std::make_unique<kn::RecordingLaunch>();
    };
    auto runner = en::ModelRunner::Create(life.node, std::move(config), std::move(source), 0, 0);
    ASSERT_TRUE(runner) << ts::Failed(runner).value_or("");
    auto retained = en::RunnerOwner::Adopt(life.node, *runner);
    ASSERT_TRUE(retained) << ts::Failed(retained).value_or("");
    life.runner = std::move(*retained);
    const auto setup = life.runner->Setup();
    ASSERT_TRUE(setup) << ts::Failed(setup).value_or("");
    ASSERT_TRUE(life.node.MapWorkspace(life.runner->activations_needed(),
                                       std::max(kExtent, life.runner->pool_needed())));
    const auto fixed = life.node.catalog().OccupancyOf(life.node.domain()).Total().value();
    ASSERT_TRUE(life.node.Start(llmp::base::Bytes(fixed + 32 * kExtent)));
    ASSERT_TRUE(life.runner->Register());
    life.node.Run();
    ASSERT_TRUE(life.runner->Bind());
    ASSERT_TRUE(life.runner->slots().SelectSlots(std::array{0U}));
    ASSERT_TRUE(life.node.BeginRequest(0, life.runner->slots().execution(), "real routed tail"));
  }
  void TearDown() override {
    if (!life_) return;
    auto& life = *life_;
    pr::fake::ClearCopyFailure();
    if (life.node.InRequest(0)) {
      EXPECT_EQ(life.node.EndRequest(0).has_value(), request_success_);
      EXPECT_FALSE(life.node.InRequest(0));
    }
    if (life.runner) {
      std::array<en::PagedModel*, 1> models{life.runner.get()};
      const auto ended = life.node.TearDown(models);
      EXPECT_TRUE(ended) << ts::Failed(ended).value_or("");
      if (!ended) {
        (void)new std::shared_ptr<TailEnvironment>(life_);
        return;
      }
      life.runner.reset();
    } else
      EXPECT_TRUE(life.node.TearDown({}));
    life.facts.reset();
    if (life.paid && life.owner_payment) life.paid->Release(life.owner_payment);
    std::error_code error;
    std::filesystem::remove_all(life.directory, error);
    EXPECT_FALSE(error);
  }
  en::WaveReport Prompt(std::uint32_t past = 0, bool invariant = false, en::StepAhead ahead = {},
                        pl::Phase phase = pl::Phase::kPrefill) {
    auto& life = *life_;
    auto composer = en::WaveComposer::For({}, life.runner->slots());
    EXPECT_TRUE(composer);
    if (!composer) return {};
    const std::array<std::int32_t, 4> history{73, 74, 75, 76};
    const std::array pending{en::PendingWork{.slot = 0,
                                             .phase = phase,
                                             .past = past,
                                             .rows = 1,
                                             .outputs = 1,
                                             .row_invariant = invariant}};
    const std::array work{en::SegmentWork{.tokens = std::span(history).subspan(past, 1)}};
    const std::array histories{std::span<const std::int32_t>(history).first(past + 1)};
    return en::RunPromptWave(*life.runner, *composer, pending, work, histories,
                             [&](const auto&, auto) { return ahead; });
  }
  en::StepAhead Future(std::uint32_t past, bool invariant, std::uint32_t builds = 1) {
    const std::array<std::optional<pl::StepShape>, 1> shapes{pl::StepShape{
        .phase = pl::Phase::kPrefill,
        .segments = {{.slot = 0, .rows = 1, .past = past, .read_cells = 256, .outputs = 1}},
        .row_invariant = invariant,
        .read_align = 256}};
    auto paid = life_->runner->Predict(shapes, builds, true, true);
    EXPECT_TRUE(paid) << ts::Failed(paid).value_or("");
    return paid ? std::move(*paid) : en::StepAhead{};
  }
};

TEST_F(CompoundRoutedTest, TailFollowsActualSelectedContinuationAndPublishesAtItsRetirement) {
  auto& life = *life_;
  bool prefix_old = false, final_native = false, final_mirror_old = false;
  life.facts->after_retirement = [&](bool final) {
    auto& slots = life.runner->slots();
    if (!final) {
      prefix_old = slots.positions(0) == 0 && life.facts->numeric == 0;
      return;
    }
    const auto& control = slots.layout(1).items.back();
    md::DraftProgressWords words;
    std::memcpy(&words, en::support::Pointer(slots.live(0, 1).base(0) + control.offset),
                sizeof(words));
    final_native = words[1] == 1 && life.facts->numeric == 1 && life.facts->source_reads == 1;
    final_mirror_old =
        slots.positions(0) == 0 && slots.progress(0, 1)->completed() == md::DraftProgress{};
  };
  auto report = Prompt(0, false, Future(1, true));
  life.facts->after_retirement = {};
  ASSERT_EQ(report.outcomes.size(), 1U);
  ASSERT_FALSE(report.outcomes[0].refusal) << report.outcomes[0].refusal->what;
  EXPECT_TRUE(prefix_old);
  EXPECT_TRUE(final_native);
  EXPECT_TRUE(final_mirror_old);
  EXPECT_EQ(life.facts->prefix_completions, 1U);
  EXPECT_EQ(life.facts->final_completions, 1U);
  EXPECT_EQ(life.runner->slots().positions(0), 1U);
  EXPECT_EQ(life.runner->slots().progress(0, 1)->completed(), (md::DraftProgress{1, 0, 0, 1, 0}));
  ASSERT_EQ(report.outcomes[0].outputs.size(), sizeof(ts::FakeRow));
  ts::FakeRow row;
  std::memcpy(&row, report.outcomes[0].outputs.data(), sizeof(row));
  std::uint8_t salt = 0;
  for (const char c : std::string(kArray) + "#1")
    salt = static_cast<std::uint8_t>(salt * 31U + static_cast<std::uint8_t>(c));
  std::array<std::uint8_t, sizeof(float)> first_weight{};
  for (std::size_t i = 0; i < first_weight.size(); ++i)
    first_weight[i] = static_cast<std::uint8_t>(salt + i * 7U + 1U);
  float expected_weight = 0;
  std::memcpy(&expected_weight, first_weight.data(), sizeof(expected_weight));
  ASSERT_TRUE(life.facts->source_valid);
  ASSERT_TRUE(std::isfinite(expected_weight));
  ASSERT_GE(expected_weight, 0);
  ASSERT_LE(static_cast<double>(expected_weight),
            static_cast<double>(std::numeric_limits<std::int32_t>::max()) - 73);
  EXPECT_EQ(life.facts->consumed, expected_weight);
  EXPECT_EQ(row.token, 73 + static_cast<std::int32_t>(expected_weight));
  const auto expert_pages = life.runner->weights();
  ASSERT_EQ(expert_pages.size(), 2U);
  EXPECT_TRUE(life.node.Call(
      [&]() -> en::Status {
        const auto unselected = life.node.catalog().Describe(expert_pages[0]);
        const auto selected = life.node.catalog().Describe(expert_pages[1]);
        if (!unselected || !selected ||
            unselected->state != llmp::catalog::ExtentState::kNonresident ||
            selected->state != llmp::catalog::ExtentState::kResident)
          return std::unexpected("selected expert was not the sole materialized source");
        return {};
      },
      "inspect actual selected expert residency"));
  EXPECT_EQ(life.node.kept_pinned(), 0U);
  EXPECT_EQ(life.runner->lookahead_stats().cached, 1U);
}

class BaseLeasedRoutedTest : public CompoundRoutedTest {
 protected:
  bool BaseLeasedRoute() const override { return true; }
};

TEST_F(BaseLeasedRoutedTest, AllSelectedExpertsLeasedSkipsCatalogClosureRpcAndPageIn) {
  auto& life = *life_;
  const auto expert_pages = life.runner->weights();
  ASSERT_EQ(expert_pages.size(), 2U);
  EXPECT_EQ(life.runner->base_weights().size(), expert_pages.size());
  auto report = Prompt(0, false, Future(1, true));
  ASSERT_EQ(report.outcomes.size(), 1U);
  ASSERT_FALSE(report.outcomes[0].refusal) << report.outcomes[0].refusal->what;
  EXPECT_EQ(life.facts->prefix_completions, 1U);
  EXPECT_EQ(life.facts->final_completions, 1U);
  EXPECT_EQ(life.runner->slots().positions(0), 1U);
  EXPECT_EQ(life.node.direct_steps(), 2U);
  EXPECT_TRUE(life.node.Call(
      [&]() -> en::Status {
        const auto exp0 = life.node.catalog().Describe(expert_pages[0]);
        const auto exp1 = life.node.catalog().Describe(expert_pages[1]);
        if (!exp0 || !exp1 || exp0->state != llmp::catalog::ExtentState::kResident ||
            exp1->state != llmp::catalog::ExtentState::kResident)
          return std::unexpected("base-leased experts were unexpectedly nonresident");
        return {};
      },
      "inspect base-leased expert residency"));
}

TEST_F(CompoundRoutedTest, GraphsOffStillReusesThePaidFutureAndRunsItsActualRoute) {
  auto& life = *life_;
  const auto first = Prompt(0, false, Future(1, true));
  ASSERT_FALSE(first.outcomes[0].refusal) << first.outcomes[0].refusal->what;
  EXPECT_EQ(life.runner->lookahead_stats().cached, 1U);
  const auto next = Prompt(1, true);
  ASSERT_FALSE(next.outcomes[0].refusal) << next.outcomes[0].refusal->what;
  EXPECT_EQ(next.outcomes[0].path, en::RunPath::kEager);
  EXPECT_EQ(life.facts->builds, 2U);
  EXPECT_EQ(life.facts->recordings, 0U);
  EXPECT_EQ(life.facts->source_reads, 2U);
  EXPECT_EQ(life.runner->slots().positions(0), 2U);
}

TEST_F(CompoundRoutedTest, RetiredFinalTailPublishesEvenWhenLaterHostCompletionRefuses) {
  auto& life = *life_;
  life.facts->fail_final_complete = true;
  bool observed_retired_native = false;
  life.facts->after_retirement = [&](bool final) {
    if (!final) return;
    auto& slots = life.runner->slots();
    const auto& control = slots.layout(1).items.back();
    md::DraftProgressWords words;
    std::memcpy(&words, en::support::Pointer(slots.live(0, 1).base(0) + control.offset),
                sizeof(words));
    observed_retired_native = words[1] == 1 && slots.positions(0) == 0 && life.facts->numeric == 1;
  };
  auto report = Prompt(0, false, Future(1, true));
  life.facts->after_retirement = {};
  ASSERT_EQ(report.outcomes.size(), 1U);
  ASSERT_TRUE(report.outcomes[0].refusal);
  EXPECT_EQ(report.outcomes[0].refusal->what, "retired final host refusal");
  EXPECT_TRUE(observed_retired_native);
  EXPECT_EQ(life.runner->slots().positions(0), 1U);
  EXPECT_EQ(life.runner->slots().progress(0, 1)->completed(), (md::DraftProgress{1, 0, 0, 1, 0}));
  EXPECT_FALSE(life.runner->slots().usable(0));
  EXPECT_EQ(life.node.kept_pinned(), 0U);  // Actual final report was positively retired.
  EXPECT_EQ(life.runner->lookahead_stats().built, 1U);
  EXPECT_EQ(life.runner->lookahead_stats().cached, 0U);
  std::weak_ptr<TailEnvironment> owned = life_;
  life_.reset();
  EXPECT_FALSE(owned.expired());  // Only the retained flight owns the complete environment.
  life_ = owned.lock();
  ASSERT_TRUE(life_);
}

TEST_F(CompoundRoutedTest, PartialFinalTailNeverPublishesAndRetainsItsSource) {
  auto& life = *life_;
  request_success_ = false;
  pr::fake::FailNextCopy(pr::CopyKind::kHostToDevice, md::kDraftProgressBytes, true);
  const auto report = Prompt(0, false, Future(1, true));
  ASSERT_EQ(report.outcomes.size(), 1U);
  ASSERT_TRUE(report.outcomes[0].refusal);
  EXPECT_EQ(life.facts->numeric, 1U);
  EXPECT_EQ(life.runner->slots().positions(0), 0U);
  EXPECT_EQ(life.runner->slots().progress(0, 1)->completed(), md::DraftProgress{});
  EXPECT_FALSE(life.runner->slots().usable(0));
  EXPECT_EQ(life.node.kept_pinned(), 1U);
  EXPECT_EQ(life.runner->lookahead_stats().built, 1U);
  EXPECT_EQ(life.runner->lookahead_stats().cached, 0U);
}

class LookaheadRoutedTest : public CompoundRoutedTest {
 protected:
  bool Graphs() const override { return true; }
  std::size_t RecordingIntents() const {
    std::size_t count = 0;
    life_->runner->plans().ForEach(
        [&](const auto& entry) { count += entry.runs[0].capture_on_use ? 1U : 0U; });
    return count;
  }
};

TEST_F(LookaheadRoutedTest, PaidFutureBuildInstallsAfterFinalRetirementAndRecordsOnlyAtRealUse) {
  auto& life = *life_;
  bool cached_before_final = false;
  life.facts->after_retirement = [&](bool) {
    cached_before_final |= life.runner->lookahead_stats().cached != 0;
  };
  const auto first = Prompt(0, false, Future(1, true));
  life.facts->after_retirement = {};
  ASSERT_FALSE(first.outcomes[0].refusal) << first.outcomes[0].refusal->what;
  EXPECT_FALSE(cached_before_final);
  EXPECT_EQ(life.runner->lookahead_stats().built, 1U);
  EXPECT_EQ(life.runner->lookahead_stats().cached, 1U);
  EXPECT_EQ(life.facts->builds, 2U);
  EXPECT_GT(life.facts->future_host, 0U);
  EXPECT_LE(life.facts->future_host, life.facts->measured_host);
  EXPECT_EQ(life.facts->measured_host, life.runner->plan_floor_bytes());
  EXPECT_EQ(life.facts->recordings, 0U);  // No predicted route token or selected lease exists.
  EXPECT_EQ(RecordingIntents(), 1U);
  const auto second = Prompt(1, true, Future(2, false, 0));
  ASSERT_FALSE(second.outcomes[0].refusal) << second.outcomes[0].refusal->what;
  EXPECT_EQ(second.outcomes[0].path, en::RunPath::kCaptured);
  EXPECT_EQ(life.facts->builds, 2U);      // Reused the exact paid future.
  EXPECT_EQ(life.facts->recordings, 2U);  // Real prefix and selected continuation.
  EXPECT_EQ(life.runner->lookahead_stats().captured_ahead, 0U);
  EXPECT_EQ(RecordingIntents(), 1U);  // Consumed current; deferred the other cached plan.
  const auto third = Prompt(2, false);
  ASSERT_FALSE(third.outcomes[0].refusal) << third.outcomes[0].refusal->what;
  EXPECT_EQ(third.outcomes[0].path, en::RunPath::kCaptured);
  EXPECT_EQ(RecordingIntents(), 0U);
  const auto fourth = Prompt(3, true);
  ASSERT_FALSE(fourth.outcomes[0].refusal) << fourth.outcomes[0].refusal->what;
  EXPECT_EQ(fourth.outcomes[0].path, en::RunPath::kReplayed);
  EXPECT_EQ(life.facts->recordings, 4U);
  EXPECT_EQ(life.facts->replays, 6U);
  EXPECT_EQ(life.facts->source_reads, 4U);
  EXPECT_EQ(life.facts->numeric, 4U);
  EXPECT_EQ(life.runner->slots().positions(0), 4U);
  EXPECT_EQ(life.facts->companion_builds, 1U);
  ASSERT_TRUE(life.runner->slots().progress(0, 1)->settled());
  EXPECT_EQ(life.runner->slots().progress(0, 1)->completed().positions, 4U);
  // Three actual TokenAfterCache companions consume the following tokens,
  // with their authorized cache coordinates. Reuse cannot skip that work.
  const auto& draft = life.runner->slots().layout(1).items.front();
  const auto cache = life.runner->slots().live(0, 1).base(0) + draft.offset;
  for (std::uint32_t position = 0; position < 3; ++position) {
    ts::FakeCell cell;
    std::memcpy(&cell, en::support::Pointer(cache + position * draft.contract.cell_bytes),
                sizeof(cell));
    EXPECT_EQ(cell.magic, ts::kFakeMagic);
    EXPECT_EQ(cell.position, static_cast<std::int32_t>(position));
    EXPECT_EQ(cell.token, 74 + static_cast<std::int32_t>(position));
  }
  ts::FakeRow row;
  ASSERT_EQ(fourth.outcomes[0].outputs.size(), sizeof(row));
  std::memcpy(&row, fourth.outcomes[0].outputs.data(), sizeof(row));
  EXPECT_EQ(row.position, 3);
  EXPECT_EQ(row.token, 76 + static_cast<std::int32_t>(life.facts->consumed));
}

TEST_F(LookaheadRoutedTest, RefusedFutureBuildCannotSuppressTheActualSelectedStep) {
  auto& life = *life_;
  life.facts->refuse_future_build = true;
  const auto first = Prompt(0, false, Future(1, true));
  ASSERT_FALSE(first.outcomes[0].refusal) << first.outcomes[0].refusal->what;
  EXPECT_EQ(life.runner->lookahead_stats().built, 0U);
  EXPECT_EQ(life.runner->lookahead_stats().cached, 0U);
  EXPECT_EQ(life.runner->plans().size(), 1U);
  EXPECT_EQ(life.facts->source_reads, 1U);
  life.facts->refuse_future_build = false;
  const auto next = Prompt(1, true);
  ASSERT_FALSE(next.outcomes[0].refusal) << next.outcomes[0].refusal->what;
  EXPECT_EQ(life.facts->source_reads, 2U);
  EXPECT_EQ(life.runner->slots().positions(0), 2U);
}

TEST_F(LookaheadRoutedTest, UnmeasuredFutureRoutingIdentityIsDiscardedBeforeCachePublication) {
  auto& life = *life_;
  life.facts->change_future_identity = true;
  const auto first = Prompt(0, false, Future(1, true));
  ASSERT_FALSE(first.outcomes[0].refusal) << first.outcomes[0].refusal->what;
  EXPECT_EQ(life.runner->lookahead_stats().built, 1U);
  EXPECT_EQ(life.runner->lookahead_stats().cached, 0U);
  EXPECT_EQ(life.runner->plans().size(), 1U);
  EXPECT_EQ(life.facts->source_reads, 1U);
  life.facts->change_future_identity = false;
  const auto next = Prompt(1, true);
  ASSERT_FALSE(next.outcomes[0].refusal) << next.outcomes[0].refusal->what;
  EXPECT_EQ(life.facts->source_reads, 2U);
}

TEST_F(LookaheadRoutedTest, ARealSegmentRecordingRefusalKeepsTheSelectedStepEager) {
  auto& life = *life_;
  const auto first = Prompt(0, false, Future(1, true));
  ASSERT_FALSE(first.outcomes[0].refusal);
  life.facts->refuse_recording = true;
  const auto next = Prompt(1, true);
  ASSERT_FALSE(next.outcomes[0].refusal) << next.outcomes[0].refusal->what;
  EXPECT_EQ(next.outcomes[0].path, en::RunPath::kEager);
  EXPECT_EQ(life.facts->recordings, 0U);
  EXPECT_EQ(RecordingIntents(), 0U);  // A proven capture refusal consumes the request.
  EXPECT_EQ(life.facts->source_reads, 2U);
  EXPECT_EQ(life.runner->slots().positions(0), 2U);
}

class HintedCaptureRoutedTest : public LookaheadRoutedTest {
 protected:
  std::uint32_t AutomaticCapturePhases() const override { return 0; }
};

TEST_F(HintedCaptureRoutedTest, ExplicitHintsRecordWhileUnhintedReuseStaysEager) {
  auto& life = *life_;
  const auto first = Prompt(0, false, Future(1, true));
  ASSERT_FALSE(first.outcomes[0].refusal) << first.outcomes[0].refusal->what;
  EXPECT_EQ(life.runner->lookahead_stats().cached, 1U);
  const auto second = Prompt(1, true);
  ASSERT_FALSE(second.outcomes[0].refusal) << second.outcomes[0].refusal->what;
  EXPECT_EQ(second.outcomes[0].path, en::RunPath::kCaptured);
  EXPECT_EQ(life.facts->recordings, 2U);
  const auto third = Prompt(2, false);
  ASSERT_FALSE(third.outcomes[0].refusal) << third.outcomes[0].refusal->what;
  EXPECT_EQ(third.outcomes[0].path, en::RunPath::kEager);
  EXPECT_EQ(life.facts->recordings, 2U);  // Reuse alone does not request recording.
  const auto fourth = Prompt(3, false, Future(4, false, 0));
  ASSERT_FALSE(fourth.outcomes[0].refusal) << fourth.outcomes[0].refusal->what;
  EXPECT_EQ(fourth.outcomes[0].path, en::RunPath::kCaptured);
  EXPECT_EQ(life.facts->recordings, 4U);  // Explicit repeat hint uses real selected work.
  EXPECT_EQ(life.facts->source_reads, 4U);
  EXPECT_EQ(life.runner->slots().positions(0), 4U);
  EXPECT_TRUE(life.runner->slots().progress(0, 1)->settled());
  EXPECT_EQ(life.runner->lookahead_stats().captured_ahead, 0U);
}

class FirstCaptureRoutedTest : public LookaheadRoutedTest {
 protected:
  std::uint32_t FirstCapturePhases() const override {
    return std::uint32_t{1} << static_cast<std::uint32_t>(pl::Phase::kPrefill);
  }
};

TEST_F(FirstCaptureRoutedTest, FirstUseRecordingRequiresTheRealSelectedContinuation) {
  auto& life = *life_;
  const auto first = Prompt();
  ASSERT_FALSE(first.outcomes[0].refusal) << first.outcomes[0].refusal->what;
  EXPECT_EQ(first.outcomes[0].path, en::RunPath::kCaptured);
  EXPECT_EQ(life.facts->recordings, 2U);
  EXPECT_EQ(life.facts->replays, 2U);
  EXPECT_EQ(life.facts->source_reads, 1U);
  EXPECT_EQ(life.runner->slots().positions(0), 1U);
  EXPECT_EQ(life.runner->lookahead_stats().captured_ahead, 0U);
}

class CaptureDisabledRoutedTest : public LookaheadRoutedTest {
 protected:
  std::uint32_t CapturePhases() const override { return 0; }
};

TEST_F(CaptureDisabledRoutedTest, DeferredIntentCannotOverrideTheActualPhasePolicy) {
  auto& life = *life_;
  const auto first = Prompt(0, false, Future(1, true));
  ASSERT_FALSE(first.outcomes[0].refusal) << first.outcomes[0].refusal->what;
  EXPECT_EQ(life.runner->lookahead_stats().cached, 1U);
  const auto next = Prompt(1, true);
  ASSERT_FALSE(next.outcomes[0].refusal) << next.outcomes[0].refusal->what;
  EXPECT_EQ(next.outcomes[0].path, en::RunPath::kEager);
  EXPECT_EQ(life.facts->recordings, 0U);
  EXPECT_EQ(life.facts->source_reads, 2U);
  EXPECT_EQ(RecordingIntents(), 1U);  // Policy did not attempt or consume it.
  EXPECT_EQ(life.runner->lookahead_stats().captured_ahead, 0U);
}

TEST_F(LookaheadRoutedTest, AnOrdinaryFutureInARoutedSourceUsesItsRealWholeGraphCapability) {
  auto& life = *life_;
  // Score owns the same target/drafter history as a prompt. Warm and replay
  // through the real compound schedule, including its actual post companion.
  const auto warm = Prompt(0, false, {}, pl::Phase::kScore);
  ASSERT_FALSE(warm.outcomes[0].refusal) << warm.outcomes[0].refusal->what;
  const std::array<std::optional<pl::StepShape>, 1> shapes{pl::StepShape{
      .phase = pl::Phase::kScore,
      .segments = {{.slot = 0, .rows = 1, .past = 2, .read_cells = 256, .outputs = 1}}}};
  auto ahead = life.runner->Predict(shapes, 0, true, false);
  ASSERT_TRUE(ahead);
  const auto first = Prompt(1, false, std::move(*ahead));
  ASSERT_FALSE(first.outcomes[0].refusal) << first.outcomes[0].refusal->what;
  EXPECT_EQ(life.runner->lookahead_stats().captured_ahead, 1U);
  EXPECT_EQ(life.facts->recordings, 0U);
  const auto actual = Prompt(2, false, {}, pl::Phase::kScore);
  ASSERT_FALSE(actual.outcomes[0].refusal) << actual.outcomes[0].refusal->what;
  EXPECT_EQ(actual.outcomes[0].path, en::RunPath::kReplayed);
  EXPECT_EQ(life.facts->source_reads, 1U);  // Only routed work consumed expert1.
  EXPECT_EQ(life.runner->slots().positions(0), 3U);
  EXPECT_TRUE(life.runner->slots().progress(0, 1)->settled());
  EXPECT_EQ(life.runner->slots().progress(0, 1)->completed().positions, 3U);
}

}  // namespace

namespace {

class PrivateCompoundRoutedTest : public CompoundRoutedTest {
 protected:
  bool PrivateResources() const override { return true; }
};

TEST_F(PrivateCompoundRoutedTest, ProducerMintedPrivateRouteConsumesExactExpertAndPublishesTail) {
  auto& life = *life_;
  bool final_native = false, mirrors_old = false;
  life.facts->after_retirement = [&](bool final) {
    if (!final) return;
    const auto& control = life.runner->slots().layout(1).items.back();
    md::DraftProgressWords words;
    std::memcpy(&words,
                en::support::Pointer(life.runner->slots().live(0, 1).base(0) + control.offset),
                sizeof(words));
    final_native = words[1] == 1 && life.facts->source_reads == 1;
    mirrors_old = life.runner->slots().positions(0) == 0;
  };
  const auto report = Prompt(0, true);
  life.facts->after_retirement = {};
  ASSERT_EQ(report.outcomes.size(), 1U);
  ASSERT_FALSE(report.outcomes[0].refusal) << report.outcomes[0].refusal->what;
  EXPECT_TRUE(final_native);
  EXPECT_TRUE(mirrors_old);
  EXPECT_GT(life.facts->private_views, 0U);
  EXPECT_GT(life.facts->private_builds, 0U);
  EXPECT_EQ(report.outcomes[0].path, en::RunPath::kEager);
  EXPECT_EQ(life.facts->source_reads, 1U);
  ASSERT_TRUE(life.facts->source_valid);
  EXPECT_EQ(life.runner->slots().positions(0), 1U);
  EXPECT_TRUE(life.runner->slots().progress(0, 1)->settled());
  EXPECT_EQ(life.node.kept_pinned(), 0U);
  ASSERT_EQ(report.outcomes[0].outputs.size(), sizeof(ts::FakeRow));
  ts::FakeRow row;
  std::memcpy(&row, report.outcomes[0].outputs.data(), sizeof(row));
  EXPECT_EQ(row.token, 73 + static_cast<std::int32_t>(life.facts->consumed));
}

TEST_F(PrivateCompoundRoutedTest, PrivateFinalRetirementPublishesBeforeLaterHostRefusal) {
  auto& life = *life_;
  life.facts->fail_final_complete = true;
  const auto report = Prompt(0, true);
  ASSERT_EQ(report.outcomes.size(), 1U);
  ASSERT_TRUE(report.outcomes[0].refusal);
  EXPECT_GT(life.facts->private_builds, 0U);
  EXPECT_EQ(life.facts->source_reads, 1U);
  EXPECT_EQ(life.facts->final_completions, 1U);
  EXPECT_EQ(life.runner->slots().positions(0), 1U);
  EXPECT_EQ(life.runner->slots().progress(0, 1)->completed(), (md::DraftProgress{1, 0, 0, 1, 0}));
  EXPECT_FALSE(life.runner->slots().usable(0));
  EXPECT_FALSE(life.runner->slots().owed(0));
  EXPECT_FALSE(life.runner->slots().faulted());
  EXPECT_EQ(life.node.kept_pinned(), 0U);
}

TEST_F(PrivateCompoundRoutedTest, UncertainPrivateTailRetainsItsCompleteBindingAndBacking) {
  auto& life = *life_;
  request_success_ = false;
  pr::fake::FailNextCopy(pr::CopyKind::kHostToDevice, md::kDraftProgressBytes, true);
  const auto report = Prompt(0, true);
  ASSERT_EQ(report.outcomes.size(), 1U);
  ASSERT_TRUE(report.outcomes[0].refusal);
  EXPECT_FALSE(report.outcomes[0].refusal->clean);
  EXPECT_GT(life.facts->private_builds, 0U);
  EXPECT_GT(life.facts->private_views, 0U);
  EXPECT_EQ(life.facts->source_reads, 1U);
  EXPECT_EQ(life.runner->slots().positions(0), 0U);
  EXPECT_FALSE(life.runner->slots().usable(0));
  EXPECT_GT(life.node.kept_pinned(), 0U);
  EXPECT_FALSE(life.runner->Release());
  std::weak_ptr<TailEnvironment> held = life_;
  life_.reset();
  EXPECT_FALSE(held.expired());
  life_ = held.lock();
  ASSERT_TRUE(life_);
  // The normal fixture drains through the real node before freeing anything.
}

}  // namespace
