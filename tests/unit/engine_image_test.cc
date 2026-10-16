// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
#include "runtime/engine_image.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "artifact_writer.h"
#include "base/sha256.h"
#include "engine/program_lifecycle.h"
#include "expected_error.h"
#include "kernels/common/recording_launch.h"
#include "providers/fake/fake_device_execution.h"

namespace {
namespace en = llmp::engine;
namespace rt = llmp::runtime;
namespace kn = llmp::kernels;
namespace co = llmp::components;
namespace ir = llmp::ir;
using en::Status;
using llmp::test_support::Failed;
struct Trace {
  bool fail_second = false, capturable = true, fail_position = false, backend_changed = false;
  std::size_t positioned = 0, prepared_requests = 0;
  std::size_t queued = 0, executed = 0;
  std::size_t live_programs = 0;
  float value = 1;
  kn::RecordingLaunch* launch = nullptr;
};
class TypedCopy final : public kn::StepProgram {
 public:
  TypedCopy(const kn::LowerInput& input, std::shared_ptr<Trace> trace)
      : trace_(std::move(trace)), measuring_(input.addresses.measuring()) {
    ++trace_->live_programs;
    const auto& graph = input.io.graph;
    const auto in = input.io.inputs[0], out = input.io.outputs[0];
    bytes_ = 4;
    for (auto dimension : graph.tensor(in).shape().ne) bytes_ *= dimension;
    const auto output_offset = (bytes_ + 255) / 256 * 256;
    scratch_ = std::max<std::uint64_t>(512, output_offset + bytes_);
    input_[0] = {.device_address = input.addresses.activations,
                 .bytes = bytes_,
                 .tensor = in,
                 .segment = graph.tensor(in).segment,
                 .kind = kn::InputKind::kStepBytes};
    output_[0] = {.device_address = input.addresses.activations + output_offset,
                  .bytes = bytes_,
                  .tensor = out,
                  .segment = graph.tensor(out).segment};
    uses_[0] = {.address = input.addresses.activations, .bytes = scratch_};
    second_ = graph.tensor(out).shape().ne[0] == 1;
  }
  ~TypedCopy() override { --trace_->live_programs; }
  bool capturable() const override { return trace_->capturable; }
  std::expected<void, kn::StepFailure> Position(std::span<const std::uint32_t> past) override {
    ++trace_->positioned;
    if (!past.empty() || trace_->fail_position)
      return std::unexpected(kn::StepFailure{kn::StepError::kRejected, "typed eager positioning"});
    positioned_ = true;
    return {};
  }
  std::uint64_t scratch_bytes() const override { return scratch_; }
  std::uint64_t host_bytes() const override { return 1024; }
  std::uint64_t launches() const override { return 1; }
  std::span<const kn::InputSlot> inputs() const override { return input_; }
  std::span<const kn::OutputSlot> outputs() const override { return output_; }
  std::span<const kn::TensorUse> tensors() const override { return uses_; }
  std::expected<void, kn::StepFailure> Bind(kn::StepLaunch&) override {
    if (measuring_)
      return std::unexpected(kn::StepFailure{kn::StepError::kRejected, "measurement"});
    bound_ = true;
    return {};
  }
  std::expected<void, kn::StepFailure> Run(kn::StepLaunch&,
                                           llmp::providers::NativeStream stream) const override {
    if (!bound_) return std::unexpected(kn::StepFailure{kn::StepError::kRejected, "unbound"});
    if (!capturable() && !positioned_)
      return std::unexpected(kn::StepFailure{kn::StepError::kRejected, "eager hook missing"});
    positioned_ = false;
    ++trace_->queued;
    const auto from = input_[0].device_address, to = output_[0].device_address;
    auto queued = llmp::providers::fake::FakeDeviceExecution::Launch(
        stream, [from, to, bytes = bytes_, trace = trace_] {
          // Fake device payload, deliberately only the typed identity copy.
          std::memcpy(reinterpret_cast<void*>(to), reinterpret_cast<const void*>(from), bytes);
          ++trace->executed;
        });
    if (!queued || (second_ && trace_->fail_second))
      return std::unexpected(kn::StepFailure{kn::StepError::kUnknown, "queued image fault"});
    return {};
  }

 private:
  std::shared_ptr<Trace> trace_;
  std::uint64_t bytes_ = 0, scratch_ = 0;
  bool measuring_, bound_ = false, second_ = false;
  mutable bool positioned_ = false;
  std::array<kn::InputSlot, 1> input_{};
  std::array<kn::OutputSlot, 1> output_{};
  std::array<kn::TensorUse, 1> uses_{};
};
class Owner final : public kn::BackendOwner {
 public:
  Owner(kn::StepBackend backend, std::shared_ptr<en::ExecutionMetadataAllowance> payment,
        std::shared_ptr<Trace> trace)
      : backend_(std::move(backend)), payment_(std::move(payment)), trace_(std::move(trace)) {}
  const kn::StepBackend& backend() const override { return backend_; }
  bool Charge(std::uint64_t n) const override { return payment_->Charge(n); }
  void Release(std::uint64_t n) const override { payment_->Release(n); }
  std::expected<llmp::execution::BackendDeclaration, std::string> Ready() const override {
    const auto id = llmp::base::Sha256{}
                        .Update(trace_->backend_changed ? "typed-copy-changed"
                                                        : "typed-copy-lifecycle-only")
                        .Finish();
    return llmp::execution::BackendDeclaration{id, id, id};
  }

 private:
  const kn::StepBackend backend_;
  std::shared_ptr<en::ExecutionMetadataAllowance> payment_;
  std::shared_ptr<Trace> trace_;
};
struct Life {
  en::PagedNode node{{.compute_streams = 1, .slots = 1, .slot_bytes = en::kPagedExtent}};
  std::shared_ptr<en::ExecutionMetadataAllowance> payment;
  std::optional<llmp::execution::Registry> registry;
  std::shared_ptr<Trace> trace = std::make_shared<Trace>();
  std::unique_ptr<rt::EngineImage> image;
  std::vector<en::PagedModel*> entered;
  bool attempted = false;
  Status Start(const std::filesystem::path& scratch, bool bad_port = false,
               bool mapped_weight = false) {
    if (auto opened = node.Open(); !opened) return opened;
    auto admitted = en::ExecutionMetadataAllowance::Admit(node.catalog(), node.domain(), 8ULL << 20,
                                                          1ULL << 30, 1ULL << 20);
    if (!admitted) return std::unexpected(admitted.error());
    payment = *admitted;
    if (!payment->Charge(4ULL << 20)) return std::unexpected("fixture inventory payment");
    auto declared = llmp::execution::Registry::Create({});
    if (!declared) return std::unexpected("empty identity registry");
    registry.emplace(std::move(*declared));
    rt::EngineImageParts parts;
    kn::StepBackend backend{.registry = &*registry,
                            .lower = [trace = trace](const kn::LowerInput& input)
                                -> std::expected<std::unique_ptr<kn::StepProgram>, std::string> {
                              if (input.token) return std::unexpected("unexpected token metadata");
                              return std::make_unique<TypedCopy>(input, trace);
                            }};
    auto owner = std::make_shared<Owner>(backend, payment, trace);
    backend.owner = owner;
    // A copied table is not the immutable owner's execution authority.
    backend.lower =
        [](const kn::LowerInput&) -> std::expected<std::unique_ptr<kn::StepProgram>, std::string> {
      return std::unexpected("caller-substituted lowerer must not execute");
    };
    parts.backend = std::make_shared<const kn::StepBackend>(std::move(backend));
    parts.keep = owner;
    parts.architecture = "typed-identity-control";
    parts.identity = llmp::base::Sha256{}.Update("two-typed-copy-stages").Finish();
    parts.request.keep = owner;
    parts.request.identity = parts.identity;
    parts.request.layout = parts.identity;
    llmp::test::V1Plan artifact;
    artifact.groups = {
        {.layer = 0,
         .resources = {{.name = "unused",
                        .format = ir::Dense(llmp::format::Encoding::kF32, ir::ShapeOf({4}))}}}};
    auto weights = std::make_unique<en::PagedWeights>();
    if (auto opened = weights->Open(llmp::test::Publish(scratch, llmp::test::Render(artifact)));
        !opened)
      return opened;
    parts.weights.push_back({std::move(weights),
                             {mapped_weight ? en::GroupPlace::kDevice : en::GroupPlace::kNone},
                             {}});
    for (std::size_t stage_id = 0; stage_id < 2; ++stage_id) {
      llmp::spec::Stage stage{.name = "copy", .role = llmp::spec::StageRole::kVae, .artifact = 0};
      stage.stream.width = 1;
      llmp::spec::Layer layer;
      layer.components.push_back({.kind = "reshape.spatial",
                                  .version = 1,
                                  .params = {{"width", std::int64_t{stage_id == 0 ? 2 : 1}},
                                             {"height", std::int64_t{stage_id == 0 ? 2 : 4}}}});
      stage.layers.push_back(std::move(layer));
      llmp::spec::ModelSpec spec{.pipeline = llmp::spec::PipelineKind::kImage,
                                 .stages = {std::move(stage)}};
      llmp::spec::Canonicalize(spec);
      const std::array<std::span<const co::BoundTensor>, 1> tables{};
      const std::array owners{co::SpatialOwner{
          0, stage_id == 0 ? co::SpatialShape{1, 4, 1, 1} : co::SpatialShape{2, 2, 1, 1}}};
      auto graph =
          co::BuildSpatialStep(spec, tables, owners,
                               {[payment = payment](std::uint64_t n) { return payment->Charge(n); },
                                [payment = payment](std::uint64_t n) { payment->Release(n); }},
                               {.reference_buffers = false});
      if (!graph) return std::unexpected(graph.error());
      parts.stages.push_back(
          {.graph = std::make_shared<const co::SpatialGraph>(std::move(*graph)),
           .addresses = {},
           .inputs = {stage_id == 0 ? std::nullopt : std::optional<std::uint32_t>(0)},
           .outputs = {stage_id == 0 ? std::optional<std::uint32_t>(0) : std::nullopt}});
    }
    parts.request.ports.push_back(
        {.format = ir::Dense(llmp::format::Encoding::kF32, ir::ShapeOf({2, 2, 1, 1})),
         .owner = static_cast<std::uint8_t>(bad_port ? 1 : 0),
         .initial = {}});
    parts.request.sequence = {{0, 0}, {1, 0}};
    parts.request.first_output = 0;
    parts.launch = [trace =
                        trace]() -> std::expected<std::unique_ptr<kn::StepLaunch>, std::string> {
      auto launch = std::make_unique<kn::RecordingLaunch>();
      trace->launch = launch.get();
      return launch;
    };
    parts.request.fill = [trace = trace](const rt::ImageInvocation&, std::size_t,
                                         std::span<std::byte> data) -> Status {
      if (data.size() != 16) return std::unexpected("fake typed size");
      const std::array<float, 4> values{trace->value, trace->value + 1, trace->value + 2,
                                        trace->value + 3};
      std::memcpy(data.data(), values.data(), data.size());
      return {};
    };
    parts.prepare_request = [this](const rt::ImageInput& input, const llmp::base::Sha256Digest&)
        -> std::expected<std::variant<rt::ImageRequest, rt::ImageGeometry>, std::string> {
      ++trace->prepared_requests;
      if (input.prompt == "refuse") return std::unexpected("paid source refused input");
      auto request = Request(input.prompt == "changed" ? 9.F : 1.F);
      if (!input.output.empty())
        request.finish = [](std::span<const std::byte>) -> std::expected<std::string, std::string> {
          return std::unexpected("completed host export refused");
        };
      return std::variant<rt::ImageRequest, rt::ImageGeometry>(std::move(request));
    };
    auto captured = std::make_shared<std::optional<rt::EngineImageParts>>(std::move(parts));
    llmp::config::ModelEntry entry;
    entry.name = "typed-copy";
    image = std::make_unique<rt::EngineImage>(
        node, entry, rt::ModelSettings{}, 0,
        [captured](const rt::ModelSettings&) -> std::expected<rt::EngineImageParts, std::string> {
          if (!*captured) return std::unexpected("already consumed");
          auto result = std::move(**captured);
          captured->reset();
          return result;
        });
    auto setup = image->Setup();
    if (image->has_paged()) entered.push_back(&image->paged());
    if (!setup) return setup;
    if (auto mapped = node.MapWorkspace(image->activations_needed(), en::kPagedExtent, 64ULL << 20);
        !mapped)
      return mapped;
    if (auto started = node.Start(llmp::base::Bytes(64ULL << 20)); !started) return started;
    if (auto registered = image->Register(); !registered) return registered;
    if (auto bound = image->Bind(); !bound) return bound;
    node.Run();
    node.SetHostFloor(0);
    return {};
  }
  rt::ImageRequest Request(float value, std::uint32_t repeats = 1) {
    rt::ImageRequest request;
    request.keep = std::make_shared<float>(value);  // within the paid fake fixture owner
    request.layout = llmp::base::Sha256{}.Update("two-typed-copy-stages").Finish();
    const std::array identity{value, static_cast<float>(repeats)};
    request.identity = llmp::base::Sha256{}.Update(std::as_bytes(std::span(identity))).Finish();
    request.ports.push_back(
        {.format = ir::Dense(llmp::format::Encoding::kF32, ir::ShapeOf({2, 2, 1, 1})),
         .initial = {}});
    for (std::uint32_t i = 0; i < repeats; ++i) request.sequence.push_back({0, i});
    request.sequence.push_back({1, 0});
    request.first_output = repeats - 1;
    request.fill = [value](const rt::ImageInvocation& invocation, std::size_t,
                           std::span<std::byte> data) -> Status {
      if (data.size() != 16) return std::unexpected("fake replacement size");
      const auto start = value + static_cast<float>(invocation.iteration);
      const std::array<float, 4> values{start, start + 1, start + 2, start + 3};
      std::memcpy(data.data(), values.data(), data.size());
      return {};
    };
    return request;
  }
  std::expected<rt::ImageGeometry, std::string> Geometry(std::uint32_t width, std::uint32_t height,
                                                         float value) {
    rt::ImageGeometry next;
    next.keep = std::make_shared<float>(value);
    const std::array dims{width, height};
    next.identity = llmp::base::Sha256{}.Update(std::as_bytes(std::span(dims))).Finish();
    next.request.keep = next.keep;
    next.request.layout = next.request.identity = next.identity;
    next.request.ports.push_back(
        {.format = ir::Dense(llmp::format::Encoding::kF32, ir::ShapeOf({width, height, 1, 1})),
         .initial = {}});
    next.request.sequence = {{0, 0}, {1, 0}};
    next.request.fill = [value](const rt::ImageInvocation&, std::size_t,
                                std::span<std::byte> data) -> Status {
      if (data.size() % sizeof(float)) return std::unexpected("fake geometry element bytes");
      for (std::size_t at = 0; at < data.size(); at += sizeof(float))
        std::memcpy(data.data() + at, &value, sizeof(value));
      return {};
    };
    for (std::uint32_t i = 0; i < 2; ++i) {
      llmp::spec::Stage stage{.name = "copy", .role = llmp::spec::StageRole::kVae, .artifact = 0};
      stage.stream.width = 1;
      llmp::spec::Layer layer;
      layer.components.push_back(
          {.kind = "reshape.spatial",
           .version = 1,
           .params = {{"width", std::int64_t{i == 0 ? width : 1}},
                      {"height", std::int64_t{i == 0 ? height : width * height}}}});
      stage.layers.push_back(std::move(layer));
      llmp::spec::ModelSpec spec{.pipeline = llmp::spec::PipelineKind::kImage,
                                 .stages = {std::move(stage)}};
      llmp::spec::Canonicalize(spec);
      const std::array<std::span<const co::BoundTensor>, 1> tables{};
      const std::array owners{co::SpatialOwner{0, {width, height, 1, 1}}};
      auto graph =
          co::BuildSpatialStep(spec, tables, owners,
                               {[payment = payment](std::uint64_t n) { return payment->Charge(n); },
                                [payment = payment](std::uint64_t n) { payment->Release(n); }},
                               {.reference_buffers = false});
      if (!graph) return std::unexpected(graph.error());
      next.stages.push_back({.graph = std::make_shared<const co::SpatialGraph>(std::move(*graph)),
                             .addresses = {},
                             .inputs = {i == 0 ? std::nullopt : std::optional<std::uint32_t>(0)},
                             .outputs = {i == 0 ? std::optional<std::uint32_t>(0) : std::nullopt}});
    }
    return next;
  }
  Status Retire() {
    if (attempted) return std::unexpected("already attempted");
    attempted = true;
    return node.TearDown(entered);
  }
};
class EngineImageTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const char* root = std::getenv("LLMP_TEST_SCRATCH");
    ASSERT_NE(root, nullptr);
    path =
        std::filesystem::path(root) / testing::UnitTest::GetInstance()->current_test_info()->name();
    std::filesystem::create_directories(path);
    life = std::make_unique<Life>();
  }
  void TearDown() override {
    auto retired = life->Retire();
    if (!retired) {
      std::ignore = life.release();
      ADD_FAILURE() << Failed(retired).value_or("");
      return;
    }
    life.reset();
    std::filesystem::remove_all(path);
  }
  std::filesystem::path path;
  std::unique_ptr<Life> life;
};
TEST_F(EngineImageTest, DeviceHandoffAndChangedInputsSurviveRepeatedGenerations) {
  const auto started = life->Start(path);
  ASSERT_TRUE(started) << Failed(started).value_or("");
  std::array<std::string, 3> output;
  for (std::size_t i = 0; i < output.size(); ++i) {
    life->trace->value = i == 1 ? 9.F : 1.F;
    std::string first;
    ASSERT_TRUE(life->image->FirstOutput(first));
    ASSERT_TRUE(life->image->Finish(output[i]));
    EXPECT_EQ(first, output[i]);
  }
  EXPECT_NE(output[0], output[1]);
  EXPECT_EQ(output[0], output[2]);
  EXPECT_EQ(life->trace->launch->captures(), 2U);
  EXPECT_EQ(life->trace->launch->launches(), 4U);
  EXPECT_EQ(life->trace->executed, 6U);
  EXPECT_EQ(life->trace->live_programs, 2U);
}
// A generation stopped after its first output (the swap table's B) is
// refused another first output until it is abandoned; then it starts again,
// takes a new request and finishes as an unstopped one does.
TEST_F(EngineImageTest, AnAbandonedFirstOutputStartsAgain) {
  const auto started = life->Start(path);
  ASSERT_TRUE(started) << Failed(started).value_or("");
  std::string first, again, output, control;
  ASSERT_TRUE(life->image->Abandon());  // unstarted: unchanged
  ASSERT_TRUE(life->image->FirstOutput(first));
  const auto refused = life->image->FirstOutput(again);
  ASSERT_FALSE(refused);
  EXPECT_EQ(refused.error(), "image first output order");
  EXPECT_FALSE(life->image->PrepareRequest({.prompt = "changed"}));
  for (int cycle = 0; cycle < 2; ++cycle) {
    ASSERT_TRUE(life->image->Abandon());
    ASSERT_TRUE(life->image->FirstOutput(again));
    EXPECT_EQ(again, first);
  }
  ASSERT_TRUE(life->image->Finish(output));
  EXPECT_EQ(output, first);
  ASSERT_TRUE(life->image->Abandon());  // completed: unchanged
  EXPECT_FALSE(life->image->Finish(control));
  ASSERT_TRUE(life->image->Abandon());
  ASSERT_TRUE(life->image->PrepareRequest({.prompt = "changed"}));
  ASSERT_TRUE(life->image->FirstOutput(again));
  ASSERT_TRUE(life->image->Finish(control));
  EXPECT_NE(control, output);  // the changed request's content
  EXPECT_EQ(life->trace->executed, 6U);
}
TEST_F(EngineImageTest, GeometryGrowthRetiresOnlyOwnedPortsAndKeepsWorkspaceAddresses) {
  ASSERT_TRUE(life->Start(path, false, true));
  const auto managed = life->image->managed_extents();
  ASSERT_EQ(managed.size(), 1U);
  const auto check_places = [&]() -> Status {
    auto& scheduler = life->node.scheduler();
    EXPECT_TRUE(scheduler.PlacePinned(managed.front()));
    const auto* source = scheduler.SourceOf(managed.front());
    if (!source) return std::unexpected("registered source is absent");
    auto relocated = *source;
    relocated.destination += en::kPagedExtent;
    const auto refused = scheduler.SetSource(managed.front(), relocated);
    EXPECT_FALSE(refused);
    if (!refused) EXPECT_EQ(refused.error(), llmp::scheduler::WorkError::kBusy);
    return life->image->CheckPlaces();
  };
  ASSERT_TRUE(life->node.Call(check_places, "checking image registered places"));
  std::string first, output;
  for (int i = 0; i < 2; ++i) {
    ASSERT_TRUE(life->image->FirstOutput(first));
    ASSERT_TRUE(life->image->Finish(output));
  }
  const auto activation_base = life->node.activations().base;
  const auto pool_base = life->node.pool().base;
  const auto old_extent = life->node.activations().extents.front();
  const auto old_generation = life->node.catalog().Describe(old_extent)->backing_generation;
  auto* const launch = life->trace->launch;
  auto geometry = life->Geometry(1024, 512, 7.F);
  ASSERT_TRUE(geometry) << Failed(geometry).value_or("");
  std::weak_ptr<void> old_owner = geometry->keep;
  const auto replaced = life->image->ReplaceGeometry(std::move(*geometry));
  ASSERT_TRUE(replaced) << Failed(replaced).value_or("");
  EXPECT_EQ(life->node.activations().base, activation_base);
  EXPECT_EQ(life->node.pool().base, pool_base);
  EXPECT_EQ(life->node.activations().extents.front(), old_extent);
  EXPECT_EQ(life->node.catalog().Describe(old_extent)->backing_generation, old_generation);
  EXPECT_EQ(life->node.activations().bytes, 6ULL << 20);
  EXPECT_EQ(life->trace->launch, launch);
  EXPECT_EQ(life->trace->live_programs, 0U);  // old geometry retired; candidate builds lazily
  for (int i = 0; i < 2; ++i) {
    ASSERT_TRUE(life->image->FirstOutput(first));
    ASSERT_TRUE(life->image->Finish(output));
    EXPECT_EQ(first, output);
  }
  const auto staging = life->image->host_input_bytes();
  const auto occupancy = life->node.catalog().OccupancyOf(life->node.domain()).Total();
  auto smaller = life->Geometry(2, 2, 3.F);
  ASSERT_TRUE(smaller);
  ASSERT_TRUE(life->image->ReplaceGeometry(std::move(*smaller)));
  EXPECT_TRUE(old_owner.expired());
  EXPECT_EQ(life->image->managed_extents(), managed);
  ASSERT_TRUE(life->node.Call(check_places, "checking retained image registered places"));
  EXPECT_EQ(life->node.activations().bytes, 6ULL << 20);  // shared high-water backing
  EXPECT_EQ(life->trace->launch, launch);
  EXPECT_LT(life->image->host_input_bytes(), staging);
  EXPECT_LT(life->node.catalog().OccupancyOf(life->node.domain()).Total(), occupancy);
  ASSERT_TRUE(life->image->FirstOutput(first));
  ASSERT_TRUE(life->image->Finish(output));
}
TEST_F(EngineImageTest, InvalidGeometryAndGrowthRefusalLeaveCompletedRequestUsable) {
  ASSERT_TRUE(life->Start(path));
  std::string first, original, output;
  ASSERT_TRUE(life->image->FirstOutput(first));
  ASSERT_TRUE(life->image->Finish(original));
  const auto base = life->node.activations().base;
  const auto extents = life->node.activations().extents;
  auto bad = life->Geometry(8, 4, 9.F);
  ASSERT_TRUE(bad);
  bad->request.ports[0].owner = 1;
  EXPECT_FALSE(life->image->ReplaceGeometry(std::move(*bad)));
  EXPECT_FALSE(life->node.GrowWorkspace(65ULL << 20, 0));
  const auto staging = life->image->host_input_bytes();
  auto unpaid = life->Geometry(8192, 1024, 9.F);
  ASSERT_TRUE(unpaid);
  std::weak_ptr<void> candidate_owner = unpaid->keep;
  EXPECT_FALSE(life->image->ReplaceGeometry(std::move(*unpaid)));
  EXPECT_TRUE(candidate_owner.expired());
  EXPECT_EQ(life->image->host_input_bytes(), staging);
  EXPECT_LE(life->node.catalog().OccupancyOf(life->node.domain()).Total(), life->node.budget());
  EXPECT_EQ(life->node.activations().base, base);
  EXPECT_EQ(life->node.activations().extents, extents);
  ASSERT_TRUE(life->image->FirstOutput(first));
  ASSERT_TRUE(life->image->Finish(output));
  EXPECT_EQ(output, original);
}
TEST_F(EngineImageTest, ServingRequestPreparationIsIdleAtomicAndReusesCompletedOwners) {
  ASSERT_TRUE(life->Start(path));
  EXPECT_FALSE(life->image->PrepareRequest({.prompt = "refuse"}));
  ASSERT_TRUE(life->image->PrepareRequest({.prompt = "original"}));
  std::string first, original, changed;
  ASSERT_TRUE(life->image->FirstOutput(first));
  const auto calls = life->trace->prepared_requests;
  EXPECT_FALSE(life->image->PrepareRequest({.prompt = "changed"}));
  EXPECT_EQ(life->trace->prepared_requests, calls);
  ASSERT_TRUE(life->image->Finish(original));
  ASSERT_TRUE(life->image->PrepareRequest({.prompt = "changed"}));
  ASSERT_TRUE(life->image->FirstOutput(first));
  ASSERT_TRUE(life->image->Finish(changed));
  EXPECT_NE(original, changed);
  const auto captures = life->trace->launch->captures();
  const auto occupied = life->node.catalog().OccupancyOf(life->node.domain()).Total();
  EXPECT_FALSE(life->image->PrepareRequest({.prompt = "refuse"}));
  ASSERT_TRUE(life->image->FirstOutput(first));
  ASSERT_TRUE(life->image->Finish(first));
  EXPECT_EQ(first, changed);
  EXPECT_EQ(life->trace->launch->captures(), captures);
  EXPECT_EQ(life->node.catalog().OccupancyOf(life->node.domain()).Total(), occupied);
  ASSERT_TRUE(life->image->PrepareRequest({.prompt = "original", .output = "refused"}));
  ASSERT_TRUE(life->image->FirstOutput(first));
  EXPECT_FALSE(life->image->Finish(first));
  // GPU completion is known even when the optional host output observer fails.
  ASSERT_TRUE(life->image->PrepareRequest({.prompt = "original"}));
  ASSERT_TRUE(life->image->FirstOutput(first));
  ASSERT_TRUE(life->image->Finish(first));
  EXPECT_EQ(first, original);
}

TEST_F(EngineImageTest, CompletedRequestReplacementReusesCapturesAndReleasesContent) {
  ASSERT_TRUE(life->Start(path));
  std::string first, original, output;
  ASSERT_TRUE(life->image->ReplaceRequest(life->Request(1)));
  ASSERT_TRUE(life->image->FirstOutput(first));
  EXPECT_FALSE(life->image->ReplaceRequest(life->Request(9)));
  ASSERT_TRUE(life->image->Finish(original));
  const auto occupancy = life->node.catalog().OccupancyOf(life->node.domain()).Total();
  const auto staging = life->image->host_input_bytes();
  const auto plans = life->trace->live_programs;
  std::weak_ptr<void> retired;
  for (std::uint32_t i = 0; i < 33; ++i) {
    auto request = life->Request(i % 2 ? 1.F : 9.F, i % 2 ? 1U : 2U);
    auto next_owner = std::weak_ptr<void>(request.keep);
    ASSERT_TRUE(life->image->ReplaceRequest(std::move(request)));
    EXPECT_TRUE(retired.expired());
    retired = next_owner;
    ASSERT_TRUE(life->image->FirstOutput(first));
    ASSERT_TRUE(life->image->Finish(output));
    EXPECT_EQ(first, output);
    if (i % 2)
      EXPECT_EQ(output, original);
    else
      EXPECT_NE(output, original);
    EXPECT_EQ(life->image->host_input_bytes(), staging);
    EXPECT_EQ(life->node.catalog().OccupancyOf(life->node.domain()).Total(), occupancy);
    EXPECT_EQ(life->trace->live_programs, plans);
    EXPECT_EQ(life->trace->launch->captures(), 2U);
  }
}
TEST_F(EngineImageTest, ReplacementRefusalIsAtomicAndRequiresReleasedRequestLease) {
  ASSERT_TRUE(life->Start(path));
  std::string output, original;
  ASSERT_TRUE(life->image->FirstOutput(output));
  ASSERT_TRUE(life->image->Finish(original));
  const auto rejected = life->node.WithRequest(
      0, life->image->request_closure(), "replacement lease control", [&]() -> Status {
        EXPECT_FALSE(life->image->ReplaceRequest(life->Request(9)));
        return {};
      });
  ASSERT_TRUE(rejected);
  for (int fault = 0; fault < 7; ++fault) {
    auto request = life->Request(9);
    if (fault == 0) request.layout = {};
    if (fault == 1) request.ports[0].owner = 1;
    if (fault == 2) request.ports[0].initial.resize(1);
    if (fault == 3) request.sequence[0].stage = 99;
    if (fault == 4) request.keep.reset();
    if (fault == 5) request.first_output = static_cast<std::uint32_t>(request.sequence.size() - 1);
    if (fault == 6) request.fill = {};
    EXPECT_FALSE(life->image->ReplaceRequest(std::move(request)));
    ASSERT_TRUE(life->image->FirstOutput(output));
    ASSERT_TRUE(life->image->Finish(output));
    EXPECT_EQ(output, original);
  }
  EXPECT_EQ(life->trace->launch->captures(), 2U);
}
TEST_F(EngineImageTest, NoncapturableProgramsPositionEveryRunAndRefuseBeforeSubmission) {
  life->trace->capturable = false;
  const auto started = life->Start(path);
  ASSERT_TRUE(started) << Failed(started).value_or("");
  std::string value;
  ASSERT_TRUE(life->image->FirstOutput(value));
  ASSERT_TRUE(life->image->Finish(value));
  EXPECT_EQ(life->trace->positioned, 2U);
  EXPECT_EQ(life->trace->queued, 2U);
  life->trace->fail_position = true;
  const auto refused = life->image->FirstOutput(value);
  ASSERT_FALSE(refused);
  EXPECT_EQ(refused.error(), "typed eager positioning");
  EXPECT_EQ(life->trace->queued, 2U);
  life->trace->fail_position = false;
  ASSERT_TRUE(life->image->FirstOutput(value));
  ASSERT_TRUE(life->image->Finish(value));
  EXPECT_EQ(life->trace->positioned, 5U);
  EXPECT_EQ(life->trace->queued, 4U);
  EXPECT_EQ(life->trace->executed, 4U);
  EXPECT_EQ(life->trace->launch->captures(), 0U);
}
TEST_F(EngineImageTest, QueuedFailureQuarantinesTheWholeFlightUntilPositiveDrain) {
  const auto started = life->Start(path);
  ASSERT_TRUE(started) << Failed(started).value_or("");
  life->trace->fail_second = true;
  std::string value;
  ASSERT_TRUE(life->image->FirstOutput(value));
  const auto failed = life->image->Finish(value);
  ASSERT_FALSE(failed);
  EXPECT_FALSE(failed.error().empty());
  EXPECT_EQ(life->trace->queued, 2U);
  const auto count = life->trace->queued;
  EXPECT_FALSE(life->image->FirstOutput(value));
  EXPECT_FALSE(life->image->Finish(value));
  EXPECT_FALSE(life->image->ReplaceRequest(life->Request(9)));
  const auto prepared = life->trace->prepared_requests;
  EXPECT_FALSE(life->image->PrepareRequest({.prompt = "changed"}));
  EXPECT_EQ(life->trace->prepared_requests, prepared);
  auto next = life->Geometry(8, 4, 9.F);
  ASSERT_TRUE(next);
  EXPECT_FALSE(life->image->ReplaceGeometry(std::move(*next)));
  life->image->DropPlans();
  std::vector<llmp::memory::ReclaimCandidate> candidates;
  life->image->ReclaimCandidates(0, false, candidates);
  EXPECT_TRUE(candidates.empty());
  EXPECT_EQ(life->trace->queued, count);
  EXPECT_EQ(life->trace->live_programs, 2U);
  // TearDown owns the positive drain in the common fixture cleanup. No failed
  // completion or destruction is interpreted as a retirement certificate.
}
TEST_F(EngineImageTest, ARefusalBetweenStagesReleasesTheLeaseItHeld) {
  ASSERT_TRUE(life->Start(path));
  std::string output, original;
  ASSERT_TRUE(life->image->FirstOutput(output));
  ASSERT_TRUE(life->image->Finish(original));
  // Three first-stage runs in one call: the lease is held between them. The
  // first run's input fill changes the backend, so the second refuses before
  // queuing anything, with the call's lease still open.
  auto request = life->Request(1, 3);
  auto fill = request.fill;
  request.fill = [fill, trace = life->trace, armed = std::make_shared<bool>(true)](
                     const rt::ImageInvocation& invocation, std::size_t slot,
                     std::span<std::byte> data) {
    if (invocation.stage == 0 && invocation.iteration == 0 && *armed) {
      trace->backend_changed = true;
      *armed = false;
    }
    return fill(invocation, slot, data);
  };
  ASSERT_TRUE(life->image->ReplaceRequest(std::move(request)));
  const auto refused = life->image->FirstOutput(output);
  ASSERT_FALSE(refused);
  EXPECT_EQ(refused.error(), "image backend identity changed");
  EXPECT_FALSE(life->node.InRequest(0));
  // The request is not unknown: the generation runs again from its start
  // once the backend is back.
  life->trace->backend_changed = false;
  const auto resumed = life->image->FirstOutput(output);
  ASSERT_TRUE(resumed) << Failed(resumed).value_or("");
  const auto finished = life->image->Finish(output);
  ASSERT_TRUE(finished) << Failed(finished).value_or("");
  EXPECT_FALSE(life->node.InRequest(0));
  ASSERT_TRUE(life->image->PrepareRequest({.prompt = "changed"}));
}
TEST_F(EngineImageTest, TypedOwnerMismatchRefusesBeforeAnyProgramLaunch) {
  const auto result = life->Start(path, true);
  ASSERT_FALSE(result);
  EXPECT_NE(result.error().find("owner differs"), std::string::npos) << result.error();
  EXPECT_EQ(life->trace->queued, 0U);
}
TEST(EngineImageInputs, MissingWeightOwnerRefusesBeforeCleanupPublication) {
  en::PagedNode node{{.slot_bytes = en::kPagedExtent}};
  llmp::config::ModelEntry entry;
  rt::EngineImage image(
      node, entry, {}, 0,
      [](const rt::ModelSettings&) -> std::expected<rt::EngineImageParts, std::string> {
        rt::EngineImageParts parts;
        parts.weights.emplace_back();
        return parts;
      });
  const auto result = image.Setup();
  ASSERT_FALSE(result);
  EXPECT_EQ(result.error(), "image source weights are missing");
  EXPECT_FALSE(image.has_paged());
  EXPECT_TRUE(image.managed_extents().empty());
  EXPECT_TRUE(image.Release());
}
}  // namespace
