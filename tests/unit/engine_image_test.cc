// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
#include "runtime/engine_image.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "artifact_writer.h"
#include "base/json.h"
#include "base/sha256.h"
#include "engine/program_lifecycle.h"
#include "expected_error.h"
#include "kernels/common/recording_launch.h"
#include "plan_report.h"
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
  // Warm runs: the first stage's programs declare a warm form and its input
  // invariant; the warm runs made and the host fills of request inputs.
  bool fail_bind = false, expansion_only = false, missing_qualification_only = false;
  bool numerical_candidate = false;
  std::uint64_t report_charge_calls = 0, report_release_calls = 0, live_report_bytes = 0;
  std::uint64_t construction_paid = 0;
  std::uint64_t report_emissions = 0;
  bool warm_capable = false;
  std::size_t warm_runs = 0, fills = 0;
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
    second_ = graph.tensor(out).shape().ne[0] == 1;
    const auto output_offset = (bytes_ + 255) / 256 * 256;
    scratch_ = std::max<std::uint64_t>(512, output_offset + bytes_);
    input_[0] = {.device_address = input.addresses.activations,
                 .bytes = bytes_,
                 .tensor = in,
                 .segment = graph.tensor(in).segment,
                 .kind = kn::InputKind::kStepBytes,
                 .invariant = trace_->warm_capable && !second_};
    output_[0] = {.device_address = input.addresses.activations + output_offset,
                  .bytes = bytes_,
                  .tensor = out,
                  .segment = graph.tensor(out).segment};
    uses_[0] = {.address = input.addresses.activations, .bytes = scratch_};
  }
  ~TypedCopy() override { --trace_->live_programs; }
  bool capturable() const override { return trace_->capturable; }
  bool warm() const override { return trace_->warm_capable && !second_; }
  std::expected<void, kn::StepFailure> RunWarm(
      kn::StepLaunch& launch, llmp::providers::NativeStream stream) const override {
    ++trace_->warm_runs;
    return Run(launch, stream);
  }
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
  void StreamPlanReport(const kn::PlanReportSink& sink) const override {
    if (!sink) return;
    ++trace_->report_emissions;
    sink(R"({"kind":"image_backend","stage":)");
    sink(second_ ? "1" : "0");
    sink("}\n");
  }
  kn::PlanCoverage plan_coverage() const override {
    if (trace_->numerical_candidate)
      return {.missing_numerical_qualifications = 7, .unqualified_numerical_candidates = 2};
    if (trace_->expansion_only) return {.decode_expansion_rules = 1};
    if (trace_->missing_qualification_only) return {.missing_numerical_qualifications = 7};
    return {.primitive_fallbacks = second_ ? 3U : 1U,
            .available_fast_fallbacks = second_ ? 2U : 0U,
            .logical_activation_bytes = second_ ? 1024U : 512U,
            .physical_activation_bytes = second_ ? 512U : 256U,
            .unmaterialized_logical_bytes = second_ ? 512U : 256U,
            .unmaterialized_roots = second_ ? 2U : 1U};
  }
  void ConstructionPaid() override {
    ++trace_->construction_paid;
    kn::StepProgram::ConstructionPaid();
  }
  std::span<const kn::InputSlot> inputs() const override { return input_; }
  std::span<const kn::OutputSlot> outputs() const override { return output_; }
  std::span<const kn::TensorUse> tensors() const override { return uses_; }
  std::expected<void, kn::StepFailure> Bind(kn::StepLaunch&) override {
    if (measuring_)
      return std::unexpected(kn::StepFailure{kn::StepError::kRejected, "measurement"});
    if (trace_->fail_bind)
      return std::unexpected(kn::StepFailure{kn::StepError::kRejected, "image bind refused"});
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
  bool Charge(std::uint64_t n) const override {
    if (!payment_->Charge(n)) return false;
    ++trace_->report_charge_calls;
    trace_->live_report_bytes += n;
    return true;
  }
  void Release(std::uint64_t n) const override {
    ++trace_->report_release_calls;
    trace_->live_report_bytes -= n;
    payment_->Release(n);
  }
  std::expected<llmp::execution::BackendDeclaration, std::string> Ready() const override {
    const auto id =
        llmp::base::Sha256{}
            .Update(trace_->backend_changed ? "typed-copy-changed" : "typed-copy-lifecycle-only")
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
  kn::PlanRecordSink report_sink;
  Status Start(const std::filesystem::path& scratch, bool bad_port = false,
               bool mapped_weight = false, bool pageable = false) {
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
           .outputs = {stage_id == 0 ? std::optional<std::uint32_t>(0) : std::nullopt},
           .invariant = stage_id == 0 && trace->warm_capable ? std::vector<std::uint32_t>{0}
                                                             : std::vector<std::uint32_t>{}});
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
    parts.prepare_request = [this](const rt::ImageInput& input,
                                   const llmp::base::Sha256Digest& layout)
        -> std::expected<std::variant<rt::ImageRequest, rt::ImageGeometry>, std::string> {
      ++trace->prepared_requests;
      if (input.prompt == "refuse") return std::unexpected("paid source refused input");
      auto request = Request(input.prompt == "changed" ? 9.F : 1.F);
      if (request.layout != layout) return std::unexpected("request layout digest mismatch");
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
    image->set_plan_report_sink(report_sink);
    auto setup = image->Setup();
    if (image->has_paged()) entered.push_back(&image->paged());
    if (!setup) return setup;
    if (auto mapped =
            node.MapWorkspace(image->activations_needed(), en::kPagedExtent, 64ULL << 20, pageable);
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
    request.fill = [value, trace = trace](const rt::ImageInvocation& invocation, std::size_t,
                                          std::span<std::byte> data) -> Status {
      if (data.size() != 16) return std::unexpected("fake replacement size");
      ++trace->fills;
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
TEST_F(EngineImageTest, RegistrationWarnsForExpansionWithoutPrimitiveFallbacks) {
  life->trace->expansion_only = true;
  ASSERT_TRUE(life->Start(path));
  auto report = llmp::base::json::Parse(life->image->plan_report());
  ASSERT_TRUE(report);
  EXPECT_EQ(report->root().find("max_available_fast_fallbacks")->int64(), 0);
  EXPECT_EQ(report->root().find("max_decode_expansion_rules")->int64(), 1);
  EXPECT_FALSE(report->root().find("warning")->string().empty());
}
TEST_F(EngineImageTest, RegistrationWarnsForMissingQualificationWithoutOtherFallbacks) {
  life->trace->missing_qualification_only = true;
  ASSERT_TRUE(life->Start(path));
  auto report = llmp::base::json::Parse(life->image->plan_report());
  ASSERT_TRUE(report);
  EXPECT_EQ(report->root().find("max_available_fast_fallbacks")->int64(), 0);
  EXPECT_EQ(report->root().find("max_decode_expansion_rules")->int64(), 0);
  EXPECT_EQ(report->root().find("max_missing_numerical_qualifications")->int64(), 7);
  EXPECT_NE(report->root().find("warning")->string().find("qualification"), std::string_view::npos);
}
TEST_F(EngineImageTest, RegistrationIdentifiesActiveCandidatesBeforeMissingEvidence) {
  life->trace->numerical_candidate = true;
  ASSERT_TRUE(life->Start(path));
  auto report = llmp::base::json::Parse(life->image->plan_report());
  ASSERT_TRUE(report);
  EXPECT_EQ(report->root().find("max_unqualified_numerical_candidates")->int64(), 2);
  EXPECT_EQ(report->root().find("max_missing_numerical_qualifications")->int64(), 7);
  EXPECT_NE(report->root().find("warning")->string().find("UNQUALIFIED CANDIDATE"),
            std::string_view::npos);
}
TEST_F(EngineImageTest, DebugReportsInstallOnceAndOnDemandUsesCompleteRetainedRecords) {
  std::vector<std::string> records;
  life->report_sink = llmp::test_support::CollectPlanRecords(records);
  const auto started = life->Start(path);
  ASSERT_TRUE(started) << Failed(started).value_or("");
  // The sink was set before Setup, but measurement is not installation.
  EXPECT_TRUE(records.empty());
  const auto summary = llmp::base::json::Parse(life->image->plan_report());
  ASSERT_TRUE(summary);
  EXPECT_EQ(summary->root().find("measured_shapes")->int64(), 2);
  EXPECT_EQ(summary->root().find("max_primitive_fallbacks")->int64(), 3);
  EXPECT_EQ(summary->root().find("max_available_fast_fallbacks")->int64(), 2);
  EXPECT_EQ(summary->root().find("max_logical_activation_bytes")->int64(), 1024);
  EXPECT_EQ(summary->root().find("max_physical_activation_bytes")->int64(), 512);
  EXPECT_EQ(summary->root().find("max_omitted_logical_bytes")->int64(), 512);
  EXPECT_EQ(summary->root().find("max_omitted_roots")->int64(), 2);
  EXPECT_FALSE(summary->root().find("warning")->string().empty());
  std::string output;
  ASSERT_TRUE(life->image->FirstOutput(output));
  ASSERT_TRUE(life->image->Finish(output));
  ASSERT_FALSE(records.empty());
  std::size_t installed = 0, compiler = 0, backend = 0;
  std::vector<std::int64_t> serials;
  for (const auto& record : records) {
    EXPECT_EQ(record.find('\n'), std::string::npos);
    auto parsed = llmp::base::json::Parse(record);
    ASSERT_TRUE(parsed) << record;
    const auto kind = parsed->root().find("kind");
    ASSERT_TRUE(kind);
    if (kind->string() == "installed_plan") {
      ++installed;
      serials.push_back(parsed->root().find("id")->int64().value_or(-1));
      EXPECT_EQ(parsed->root().find("pipeline")->string(), "image");
    }
    compiler += kind->string() == "plan";
    backend += kind->string() == "image_backend";
    EXPECT_EQ(record.find("device_address"), std::string::npos);
  }
  ASSERT_EQ(installed, 2U);
  EXPECT_EQ(compiler, 2U);
  EXPECT_EQ(backend, 2U);
  std::vector<std::string> queried;
  life->image->ReportPlans(llmp::test_support::CollectPlanRecords(queried));
  EXPECT_EQ(queried, records);
  life->image->ReportPlans({});
  const auto initial = records;
  ASSERT_TRUE(life->image->FirstOutput(output));
  ASSERT_TRUE(life->image->Finish(output));
  EXPECT_EQ(records, initial);                   // Cache hits stay quiet.
  EXPECT_EQ(life->trace->report_emissions, 4U);  // Two installs, two explicit queries.
  life->image->DropPlans();
  queried.clear();
  life->image->ReportPlans(llmp::test_support::CollectPlanRecords(queried));
  EXPECT_TRUE(queried.empty());
  ASSERT_TRUE(life->image->FirstOutput(output));
  ASSERT_TRUE(life->image->Finish(output));
  EXPECT_EQ(records.size(), initial.size() * 2);
  std::size_t reinstalled = 0;
  for (std::size_t i = initial.size(); i < records.size(); ++i) {
    const auto parsed = llmp::base::json::Parse(records[i]);
    ASSERT_TRUE(parsed);
    if (parsed->root().find("kind")->string() != "installed_plan") continue;
    EXPECT_GT(parsed->root().find("id")->int64(), serials.back());
    ++reinstalled;
  }
  EXPECT_EQ(reinstalled, 2U);
}
TEST_F(EngineImageTest, DefaultReportsAreQueryableAndConstructionPaymentTransfersToTheCache) {
  const auto started = life->Start(path);
  ASSERT_TRUE(started) << Failed(started).value_or("");
  const auto before = life->payment->used();
  const auto host_before = life->node.host_counted();
  EXPECT_EQ(life->trace->live_report_bytes, 0U);
  EXPECT_EQ(life->trace->report_emissions, 0U);
  EXPECT_EQ(life->trace->report_charge_calls, 2U);
  EXPECT_EQ(life->trace->report_release_calls, 2U);
  EXPECT_EQ(life->trace->construction_paid, 0U);
  EXPECT_EQ(life->trace->live_programs, 0U);
  std::string output;
  ASSERT_TRUE(life->image->FirstOutput(output));
  ASSERT_TRUE(life->image->Finish(output));
  EXPECT_EQ(life->trace->construction_paid, 2U);
  EXPECT_EQ(life->trace->report_emissions, 0U);  // Debug off never serializes.
  EXPECT_EQ(life->trace->live_programs, 2U);
  EXPECT_EQ(life->trace->live_report_bytes, 0U);
  EXPECT_EQ(life->trace->report_charge_calls, life->trace->report_release_calls);
  EXPECT_EQ(life->payment->used(), before);
  EXPECT_GT(life->node.host_counted(), host_before);
  std::size_t records = 0;
  life->image->ReportPlans([&](kn::PlanReportEvent event, std::string_view) {
    records += event == kn::PlanReportEvent::kEnd;
  });
  EXPECT_GT(records, 2U);
  life->image->DropPlans();
  EXPECT_EQ(life->trace->live_programs, 0U);
  EXPECT_EQ(life->node.host_counted(), host_before);
  // Enabling reporting after setup affects subsequent cache installations.
  life->image->set_plan_report_sink([&](kn::PlanReportEvent event, std::string_view) {
    records += event == kn::PlanReportEvent::kEnd;
  });
  const auto previous_records = records;
  ASSERT_TRUE(life->image->FirstOutput(output));
  ASSERT_TRUE(life->image->Finish(output));
  EXPECT_GT(records, previous_records);
  life->image->set_plan_report_sink({});
  life->image->DropPlans();
  const auto disabled_records = records;
  ASSERT_TRUE(life->image->FirstOutput(output));
  ASSERT_TRUE(life->image->Finish(output));
  EXPECT_EQ(records, disabled_records);
}
TEST_F(EngineImageTest, FailedBindingReleasesReportSubgrantWithoutPublishingOrTransferring) {
  std::vector<std::string> records;
  life->report_sink = llmp::test_support::CollectPlanRecords(records);
  const auto started = life->Start(path);
  ASSERT_TRUE(started) << Failed(started).value_or("");
  const auto before = life->payment->used();
  const auto host_before = life->node.host_counted();
  life->trace->fail_bind = true;
  std::string output;
  const auto refused = life->image->FirstOutput(output);
  ASSERT_FALSE(refused);
  EXPECT_NE(refused.error().find("image bind refused"), std::string::npos);
  EXPECT_TRUE(records.empty());
  EXPECT_EQ(life->trace->construction_paid, 0U);
  EXPECT_EQ(life->trace->live_programs, 0U);
  EXPECT_EQ(life->trace->live_report_bytes, 0U);
  EXPECT_EQ(life->trace->report_charge_calls, 3U);
  EXPECT_EQ(life->trace->report_charge_calls, life->trace->report_release_calls);
  EXPECT_EQ(life->payment->used(), before);
  EXPECT_EQ(life->node.host_counted(), host_before);
  EXPECT_EQ(life->trace->queued, 0U);
  life->image->ReportPlans(llmp::test_support::CollectPlanRecords(records));
  EXPECT_TRUE(records.empty());
  life->trace->fail_bind = false;
  ASSERT_TRUE(life->image->FirstOutput(output));
  ASSERT_TRUE(life->image->Finish(output));
  EXPECT_FALSE(records.empty());
  EXPECT_EQ(life->trace->construction_paid, 2U);
  EXPECT_EQ(life->trace->live_report_bytes, 0U);
  EXPECT_EQ(life->payment->used(), before);
}
TEST_F(EngineImageTest, GeometryReplacementMeasuresQuietlyAndUsesTheInstanceReportSink) {
  std::vector<std::string> records;
  life->report_sink = llmp::test_support::CollectPlanRecords(records);
  const auto started = life->Start(path);
  ASSERT_TRUE(started) << Failed(started).value_or("");
  std::string output;
  ASSERT_TRUE(life->image->FirstOutput(output));
  ASSERT_TRUE(life->image->Finish(output));
  const auto count = records.size();
  auto geometry = life->Geometry(4, 4, 7.F);
  ASSERT_TRUE(geometry) << Failed(geometry).value_or("");
  ASSERT_TRUE(life->image->ReplaceGeometry(std::move(*geometry)));
  EXPECT_EQ(records.size(), count);
  EXPECT_EQ(life->trace->live_programs, 0U);
  EXPECT_EQ(life->trace->live_report_bytes, 0U);
  std::size_t queried = 0;
  life->image->ReportPlans([&](kn::PlanReportEvent event, std::string_view) {
    queried += event == kn::PlanReportEvent::kEnd;
  });
  EXPECT_EQ(queried, 0U);
  ASSERT_TRUE(life->image->FirstOutput(output));
  ASSERT_TRUE(life->image->Finish(output));
  EXPECT_GT(records.size(), count);
  queried = 0;
  std::vector<std::string> requested;
  life->image->ReportPlans(llmp::test_support::CollectPlanRecords(requested));
  for (const auto& record : requested) {
    auto parsed = llmp::base::json::Parse(record);
    EXPECT_TRUE(parsed) << record;
    ++queried;
  }
  EXPECT_EQ(queried, records.size() - count);
  EXPECT_EQ(life->trace->construction_paid, 4U);
  EXPECT_EQ(life->trace->live_report_bytes, 0U);
}
TEST_F(EngineImageTest, FailedMeasurementReleasesReportSubgrantWithoutInstalling) {
  std::vector<std::string> records;
  life->report_sink = llmp::test_support::CollectPlanRecords(records);
  EXPECT_FALSE(life->Start(path, true));
  EXPECT_TRUE(records.empty());
  EXPECT_GT(life->trace->report_charge_calls, 0U);
  EXPECT_EQ(life->trace->report_charge_calls, life->trace->report_release_calls);
  EXPECT_EQ(life->trace->live_report_bytes, 0U);
  EXPECT_EQ(life->trace->live_programs, 0U);
  EXPECT_EQ(life->trace->construction_paid, 0U);
}
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
// A stage run again right after itself runs warm: its invariant input is
// neither filled nor copied again. Each generation, and a new request at the
// same layout, starts cold.
TEST_F(EngineImageTest, AWarmRunSkipsItsInvariantInputAndEachGenerationStartsCold) {
  life->trace->warm_capable = true;
  const auto started = life->Start(path);
  ASSERT_TRUE(started) << Failed(started).value_or("");
  ASSERT_TRUE(life->image->ReplaceRequest(life->Request(1.F, 3)));
  std::array<std::string, 3> output;
  for (std::size_t g = 0; g < output.size(); ++g) {
    if (g == 2) ASSERT_TRUE(life->image->ReplaceRequest(life->Request(5.F, 3)));
    life->trace->fills = 0;
    const auto warm = life->trace->warm_runs;
    std::string first;
    ASSERT_TRUE(life->image->FirstOutput(first));
    // Three runs of the first stage: the cold one fills its input, the
    // warm ones leave it (their program's warm form, or its recording).
    EXPECT_EQ(life->trace->fills, 1U) << g;
    if (g == 0) EXPECT_GT(life->trace->warm_runs, warm);
    ASSERT_TRUE(life->image->Finish(output[g]));
    EXPECT_EQ(life->trace->fills, 1U) << g;  // the second stage reads a port
  }
  EXPECT_EQ(output[0], output[1]);
  EXPECT_NE(output[1], output[2]);  // the new request's input, filled cold
}
// After a load every stage is planned and recorded ahead: the first
// generation replays both, with the same outputs, and nothing records again.
TEST_F(EngineImageTest, AfterLoadRecordsEveryStageAheadOfTheFirstGeneration) {
  const auto started = life->Start(path);
  ASSERT_TRUE(started) << Failed(started).value_or("");
  const auto loaded = life->image->AfterLoad();
  ASSERT_TRUE(loaded) << Failed(loaded).value_or("");
  EXPECT_EQ(life->trace->launch->captures(), 2U);
  EXPECT_EQ(life->trace->executed, 0U);
  ASSERT_TRUE(life->image->AfterLoad());  // planned and recorded: nothing more
  EXPECT_EQ(life->trace->launch->captures(), 2U);
  std::array<std::string, 2> output;
  for (auto& o : output) {
    std::string first;
    ASSERT_TRUE(life->image->FirstOutput(first));
    ASSERT_TRUE(life->image->Finish(o));
    EXPECT_EQ(first, o);
  }
  EXPECT_EQ(output[0], output[1]);
  EXPECT_EQ(life->trace->launch->captures(), 2U);
  EXPECT_EQ(life->trace->launch->launches(), 4U);
  EXPECT_EQ(life->trace->executed, 4U);
  EXPECT_EQ(life->trace->live_programs, 2U);
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
// The runtime's pageable workspace: the image's closures page its prefix in;
// a geometry growth pages its grown prefix in before the previous geometry
// retires, and one whose room is taken is refused with that geometry still
// usable; a stage whose closure cannot be admitted (no lease, nothing
// moved) leaves the generation known and runnable once there is room.
TEST_F(EngineImageTest, PageableGrowthPagesInOrIsRefusedWithTheModelUsable) {
  ASSERT_TRUE(life->Start(path, false, false, true));
  ASSERT_TRUE(life->node.workspace_pageable());
  auto& node = life->node;
  const auto state_of = [&](llmp::catalog::ExtentId extent) {
    auto state = llmp::catalog::ExtentState::kNonresident;
    EXPECT_TRUE(node.Call(
        [&]() -> Status {
          state = node.catalog().Describe(extent).value().state;
          return {};
        },
        "reading a workspace extent"));
    return state;
  };
  const auto occupancy = [&]() {
    std::uint64_t total = 0;
    EXPECT_TRUE(node.Call(
        [&]() -> Status {
          total = node.catalog().OccupancyOf(node.domain()).Total().value();
          return {};
        },
        "reading the occupancy"));
    return total;
  };
  using llmp::catalog::ExtentState;
  EXPECT_EQ(state_of(node.activations().extents.front()), ExtentState::kNonresident);
  std::string first, original, output;
  ASSERT_TRUE(life->image->FirstOutput(first));
  ASSERT_TRUE(life->image->Finish(original));
  EXPECT_EQ(state_of(node.activations().extents.front()), ExtentState::kResident);
  // A growth pages its new extents in as it replaces the geometry.
  auto big = life->Geometry(1024, 512, 7.F);
  ASSERT_TRUE(big);
  ASSERT_TRUE(life->image->ReplaceGeometry(std::move(*big)));
  const std::vector<llmp::catalog::ExtentId> grown = node.activations().extents;
  ASSERT_EQ(grown.size(), 3U);
  for (const auto extent : grown) EXPECT_EQ(state_of(extent), ExtentState::kResident);
  const std::uint64_t with_big = occupancy();
  auto small = life->Geometry(2, 2, 3.F);
  ASSERT_TRUE(small);
  ASSERT_TRUE(life->image->ReplaceGeometry(std::move(*small)));
  // The smaller geometry of the active image gave back the tail only the
  // larger one used.
  const std::vector<llmp::catalog::ExtentId> tail(grown.begin() + 1, grown.end());
  for (const auto extent : tail) EXPECT_EQ(state_of(extent), ExtentState::kNonresident);
  // What the larger geometry's ports took beyond the smaller's: the
  // occupancy it had less the tail it gave back, less what is held now.
  const std::uint64_t tail_bytes = tail.size() * en::kPagedExtent;
  ASSERT_GT(with_big, tail_bytes + occupancy());
  const std::uint64_t ports = with_big - tail_bytes - occupancy();
  // The room left is less than the tail, though enough for the larger
  // geometry's ports.
  std::vector<llmp::catalog::ExtentId> staging;
  const std::uint64_t filler = node.budget().value() - occupancy() - ports - en::kPagedExtent;
  auto held = node.Pinned(filler, 0, staging);
  ASSERT_TRUE(held) << Failed(held).value_or("");
  auto refused = life->Geometry(1024, 512, 7.F);
  ASSERT_TRUE(refused);
  std::weak_ptr<void> candidate_owner = refused->keep;
  EXPECT_FALSE(life->image->ReplaceGeometry(std::move(*refused)));
  EXPECT_TRUE(candidate_owner.expired());
  for (const auto extent : tail) EXPECT_EQ(state_of(extent), ExtentState::kNonresident);
  EXPECT_LE(occupancy(), node.budget().value());
  ASSERT_TRUE(life->image->FirstOutput(first));  // the smaller geometry still runs
  ASSERT_TRUE(life->image->Finish(output));
  ASSERT_TRUE(node.FreePinned(*held));
  auto again = life->Geometry(1024, 512, 7.F);
  ASSERT_TRUE(again);
  ASSERT_TRUE(life->image->ReplaceGeometry(std::move(*again)));
  for (const auto extent : tail) EXPECT_EQ(state_of(extent), ExtentState::kResident);
  ASSERT_TRUE(life->image->FirstOutput(first));
  ASSERT_TRUE(life->image->Finish(output));
  // A stage refused its closure: no lease, nothing moved, nothing unknown.
  ASSERT_TRUE(node.Evict(tail));
  staging.clear();
  auto full = node.Pinned(node.budget().value() - occupancy() - en::kPagedExtent, 0, staging);
  ASSERT_TRUE(full) << Failed(full).value_or("");
  std::string late;
  EXPECT_FALSE(life->image->FirstOutput(late));
  EXPECT_FALSE(node.InRequest(0));
  ASSERT_TRUE(node.FreePinned(*full));
  ASSERT_TRUE(life->image->FirstOutput(late));
  ASSERT_TRUE(life->image->Finish(output));
  EXPECT_EQ(late, output);
  // Prepared while another model is resident (the image inactive): the
  // growth is only cataloged, and the image's own activation, which the
  // swap makes room for, brings it in with the rest of its closure.
  ASSERT_TRUE(node.SetActive(0, false));
  auto wider = life->Geometry(2048, 512, 5.F);
  ASSERT_TRUE(wider);
  ASSERT_TRUE(life->image->ReplaceGeometry(std::move(*wider)));
  const std::vector<llmp::catalog::ExtentId> wide = node.activations().extents;
  ASSERT_GT(wide.size(), grown.size());
  for (std::size_t i = grown.size(); i < wide.size(); ++i)
    EXPECT_EQ(state_of(wide[i]), ExtentState::kNonresident);
  ASSERT_TRUE(node.SetActive(0, true));
  ASSERT_TRUE(life->image->FirstOutput(first));
  ASSERT_TRUE(life->image->Finish(output));
  for (const auto extent : wide) EXPECT_EQ(state_of(extent), ExtentState::kResident);
  // A smaller geometry of the active image gives back what only the larger
  // one used.
  auto narrow = life->Geometry(2, 2, 3.F);
  ASSERT_TRUE(narrow);
  ASSERT_TRUE(life->image->ReplaceGeometry(std::move(*narrow)));
  EXPECT_EQ(state_of(wide.front()), ExtentState::kResident);
  EXPECT_EQ(state_of(wide.back()), ExtentState::kNonresident);
  ASSERT_TRUE(life->image->FirstOutput(first));
  ASSERT_TRUE(life->image->Finish(output));
}
// A pageable workspace: what is recorded ahead at a load replays after a
// swap out (the workspace evicted) and back in, with the same pixels and no
// new recording; a load seen while another model is active records nothing
// and leaves the workspace out.
TEST_F(EngineImageTest, PageableRecordingsAheadReplayAfterASwapOutAndIn) {
  ASSERT_TRUE(life->Start(path, false, false, true));
  auto& node = life->node;
  ASSERT_TRUE(node.workspace_pageable());
  const auto resident = [&](llmp::catalog::ExtentId extent) {
    auto state = llmp::catalog::ExtentState::kNonresident;
    EXPECT_TRUE(node.Call(
        [&]() -> Status {
          state = node.catalog().Describe(extent).value().state;
          return {};
        },
        "reading a workspace extent"));
    return state == llmp::catalog::ExtentState::kResident;
  };
  // A swap's activation pages the model's whole closure in before its
  // AfterLoad (Server::Activate); here, by hand.
  const auto activate = [&] {
    llmp::scheduler::AcquireReport report;
    return node.Acquire(life->image->everything(), report, "the image's activation");
  };
  ASSERT_TRUE(activate());
  const auto loaded = life->image->AfterLoad();
  ASSERT_TRUE(loaded) << Failed(loaded).value_or("");
  EXPECT_EQ(life->trace->launch->captures(), 2U);
  std::string first, before, after;
  ASSERT_TRUE(life->image->FirstOutput(first));
  ASSERT_TRUE(life->image->Finish(before));
  EXPECT_EQ(life->trace->launch->launches(), 2U);
  // Swapped out: another model active, the workspace evicted.
  ASSERT_TRUE(node.SetActive(0, false));
  std::vector<llmp::catalog::ExtentId> workspace;
  for (const auto* region : {&node.activations(), &node.pool()})
    for (const auto extent : region->extents)
      if (resident(extent)) workspace.push_back(extent);
  ASSERT_FALSE(workspace.empty());
  ASSERT_TRUE(node.Evict(workspace));
  EXPECT_FALSE(resident(node.activations().extents.front()));
  ASSERT_TRUE(life->image->AfterLoad());  // inactive: nothing recorded or paged in
  EXPECT_EQ(life->trace->launch->captures(), 2U);
  EXPECT_FALSE(resident(node.activations().extents.front()));
  // Back in: active, its closure paged in again, and its recordings replay.
  ASSERT_TRUE(node.SetActive(0, true));
  ASSERT_TRUE(activate());
  EXPECT_TRUE(resident(node.activations().extents.front()));
  ASSERT_TRUE(life->image->AfterLoad());  // recorded already: nothing more
  ASSERT_TRUE(life->image->FirstOutput(first));
  ASSERT_TRUE(life->image->Finish(after));
  EXPECT_EQ(after, before);
  EXPECT_EQ(life->trace->launch->captures(), 2U);
  EXPECT_EQ(life->trace->launch->launches(), 4U);
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
  EXPECT_EQ(life->trace->construction_paid, 2U);
  EXPECT_EQ(life->trace->live_programs, 2U);
  ASSERT_TRUE(life->image->PrepareRequest({.prompt = "changed"}));
  EXPECT_EQ(life->trace->construction_paid, 2U);
  EXPECT_EQ(life->trace->live_programs, 2U);
  ASSERT_TRUE(life->image->FirstOutput(first));
  ASSERT_TRUE(life->image->Finish(changed));
  EXPECT_NE(original, changed);
  EXPECT_EQ(life->trace->construction_paid, 2U);
  EXPECT_EQ(life->trace->live_programs, 2U);
  EXPECT_EQ(life->trace->launch->captures(), 2U);
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
TEST_F(EngineImageTest, NodeRetainsUnknownStagesAfterTheRuntimeAdapterIsDropped) {
  ASSERT_TRUE(life->Start(path));
  life->trace->fail_second = true;
  std::string value;
  ASSERT_TRUE(life->image->FirstOutput(value));
  ASSERT_FALSE(life->image->Finish(value));
  EXPECT_FALSE(life->image->Release());  // Public status is no drain certificate.
  EXPECT_EQ(life->trace->live_programs, 2U);
  life->image.reset();
  life->entered.clear();  // Node retains the actual executor, independently.
  EXPECT_EQ(life->trace->live_programs, 2U);
  ASSERT_TRUE(life->node.TearDown({}));
  EXPECT_EQ(life->trace->live_programs, 0U);
}
TEST_F(EngineImageTest, NodeRetainsIdlePlansAfterTheRuntimeAdapterIsDropped) {
  ASSERT_TRUE(life->Start(path));
  std::string value;
  ASSERT_TRUE(life->image->FirstOutput(value));
  ASSERT_TRUE(life->image->Finish(value));
  EXPECT_FALSE(life->image->Release());
  life->image.reset();
  life->entered.clear();
  EXPECT_EQ(life->trace->live_programs, 2U);
  ASSERT_TRUE(life->node.TearDown({}));
  EXPECT_EQ(life->trace->live_programs, 0U);
}
TEST_F(EngineImageTest, BufferNoiseOwnsItsPaymentAndValidatesEncodingLayoutExtentAndFiniteness) {
  ASSERT_TRUE(life->Start(path));
  std::array<std::uint16_t, 2> values{0x3F80, 0xBF80};
  auto made = en::HostBuffer::Copy(life->node, std::as_bytes(std::span(values)));
  ASSERT_TRUE(made) << Failed(made).value_or("");
  rt::ImageNoise noise{.bytes = std::move(*made)};
  auto checked = rt::CheckedImageNoise(noise, 2);
  ASSERT_TRUE(checked);
  values[0] = 0;  // The caller's temporary storage is independent.
  std::uint16_t retained = 0;
  std::memcpy(&retained, checked->data(), sizeof(retained));
  EXPECT_EQ(retained, 0x3F80);
  EXPECT_GT(noise.bytes.paid_bytes(), noise.bytes.size());
  EXPECT_FALSE(rt::CheckedImageNoise(noise, 1));
  EXPECT_FALSE(rt::CheckedImageNoise(noise, UINT64_MAX));
  auto wrong = noise;
  wrong.encoding = llmp::format::Encoding::kF16;
  EXPECT_FALSE(rt::CheckedImageNoise(wrong, 2));
  wrong = noise;
  wrong.layout = static_cast<rt::ImageNoiseLayout>(255);
  EXPECT_FALSE(rt::CheckedImageNoise(wrong, 2));
  EXPECT_FALSE(rt::CheckedImageNoise({}, 2));
  for (const auto nonfinite : {std::uint16_t{0x7F80}, std::uint16_t{0x7FC1}}) {
    values[0] = nonfinite;
    auto bytes = en::HostBuffer::Copy(life->node, std::as_bytes(std::span(values)));
    ASSERT_TRUE(bytes);
    EXPECT_FALSE(rt::CheckedImageNoise({.bytes = std::move(*bytes)}, 2));
  }
  const auto counted = life->node.host_counted();
  const auto payment = noise.bytes.paid_bytes();
  wrong = {};
  auto last = std::move(noise.bytes);
  EXPECT_EQ(life->node.host_counted(), counted);
  last = {};
  EXPECT_EQ(life->node.host_counted(), counted - payment);
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
