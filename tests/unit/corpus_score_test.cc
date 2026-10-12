// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "corpus_score.h"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <memory>
#include <vector>

#include "expected_error.h"
#include "fake_step_source.h"
#include "ir/builder.h"
#include "kernels/common/recording_launch.h"
#include "model/state_contract.h"

namespace {
namespace en = llmp::engine;
namespace kn = llmp::kernels;
namespace md = llmp::model;
namespace pl = llmp::planner;
namespace ts = llmp::test_support;
using llmp::base::Bytes;
constexpr std::uint32_t kVocabulary = 17;
constexpr std::uint64_t kExtent = en::kPagedExtent;

TEST(CorpusScore, StableFullVocabularyNllAndNonfiniteRefusal) {
  const std::array<float, 4> uniform{10000, 10000, 10000, 10000};
  auto score = ts::CorpusScore::Nll(uniform, 3);
  ASSERT_TRUE(score);
  EXPECT_DOUBLE_EQ(*score, std::log(4.0));
  const std::array<float, 3> skew{10000, 9999, 9998};
  score = ts::CorpusScore::Nll(skew, 2);
  ASSERT_TRUE(score);
  EXPECT_NEAR(*score, 2.0 + std::log(1.0 + std::exp(-1.0) + std::exp(-2.0)), 1e-14);
  EXPECT_FALSE(ts::CorpusScore::Nll(skew, 3));
  for (auto v : {std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(),
                 std::numeric_limits<float>::quiet_NaN()}) {
    const std::array<float, 3> bad{0, v, 1};
    EXPECT_FALSE(ts::CorpusScore::Nll(bad, 0));
  }
}

TEST(CorpusScore, BoundRefusesInvalidCorpusAndOversizedOutputs) {
  const std::array<std::int32_t, 3> ids{0, 1, 2};
  const std::array docs{ts::CorpusDocument{.slot = 0, .tokens = ids},
                        ts::CorpusDocument{.slot = 0, .tokens = ids}};
  EXPECT_FALSE(ts::CorpusScore::Bound(docs, {17, 2}));
  EXPECT_FALSE(ts::CorpusScore::Bound(std::span(docs).first(1), {2, 2}));
  EXPECT_FALSE(ts::CorpusScore::Bound(std::span(docs).first(1), {17, 0}));
  auto distinct = docs;
  distinct[1].slot = 1;
  const auto small = ts::CorpusScore::Bound(distinct, {17, 2});
  ASSERT_TRUE(small);
  EXPECT_GT(*small, 1024U * 1024U);
  std::array<std::int32_t, 32> wide{};
  const std::array large{ts::CorpusDocument{.slot = 0, .tokens = wide}};
  EXPECT_FALSE(ts::CorpusScore::Bound(large, {1048576, 32}));
}

// No execution: exercise the mandatory component lowering boundary itself.
class HeadProgram final : public kn::StepProgram {
 public:
  std::array<kn::OutputSlot, 2> slots{};
  std::uint64_t scratch_bytes() const override { return 0; }
  std::uint64_t host_bytes() const override { return sizeof(*this); }
  std::uint64_t launches() const override { return 0; }
  std::span<const kn::InputSlot> inputs() const override { return {}; }
  std::span<const kn::OutputSlot> outputs() const override { return slots; }
  std::span<const kn::TensorUse> tensors() const override { return {}; }
  std::expected<void, kn::StepFailure> Bind(kn::StepLaunch&) override { return {}; }
  std::expected<void, kn::StepFailure> Run(kn::StepLaunch&,
                                           llmp::providers::NativeStream) const override {
    return {};
  }
};

TEST(CorpusScore, GuardedLowerRejectsTypedNarrowedAndMisownedHeadsBeforeExecution) {
  namespace ir = llmp::ir;
  namespace fm = llmp::format;
  auto empty = ir::Builder().Finish();
  ASSERT_TRUE(empty);
  llmp::components::StepGraph step{
      .graph = std::move(*empty), .inputs = {}, .weights = {}, .state = {}};
  auto make_graph = [&](fm::Encoding encoding, std::uint64_t vocabulary) {
    ir::Builder builder;
    step.logits = builder.Input(ir::Dense(encoding, ir::ShapeOf({vocabulary, 2})));
    (void)builder.Input(ir::Dense(fm::Encoding::kF32, ir::ShapeOf({kVocabulary, 2})));
    builder.Output(step.logits);
    auto graph = std::move(builder).Finish();
    ASSERT_TRUE(graph);
    step.graph = std::move(*graph);
  };
  make_graph(fm::Encoding::kF32, kVocabulary);
  auto planned = ir::Builder().Finish();
  ASSERT_TRUE(planned);
  pl::StepPlan plan{.graph = std::move(*planned),
                    .steps = {},
                    .regions = {},
                    .placement = {},
                    .applied = {},
                    .from = {},
                    .tensors = {},
                    .decomposed = {}};
  pl::StepShape shape{
      .stage = 0,
      .phase = pl::Phase::kScore,
      .segments = {{.slot = 0, .rows = 1, .outputs = 1}, {.slot = 2, .rows = 1, .outputs = 1}}};
  kn::Addresses addresses;
  const kn::LowerInput input{plan, step, shape, addresses};
  int calls = 0, fault = 0;
  auto lower = ts::GuardCorpusLower(
      [&](const kn::LowerInput& in)
          -> std::expected<std::unique_ptr<kn::StepProgram>, std::string> {
        ++calls;
        auto program = std::make_unique<HeadProgram>();
        constexpr std::uint64_t bytes = kVocabulary * sizeof(float);
        program->slots = {
            {{.device_address = 4096, .bytes = bytes, .tensor = in.step.logits, .segment = 0},
             {.device_address = 4096 + bytes,
              .bytes = bytes,
              .tensor = in.step.logits,
              .segment = 1}}};
        switch (fault) {
          case 1:
            program->slots[1].bytes -= sizeof(float);
            break;
          case 2:
            program->slots[1].segment = 0;
            break;
          case 3:
            program->slots[1].features = true;
            break;
          case 4:
            program->slots[1].tensor = ir::TensorId{1};
            break;
          case 5:
            std::swap(program->slots[0].device_address, program->slots[1].device_address);
            break;
          case 6:
            program->slots[1].segment = ir::kJoinedSegment;
            break;
          default:
            break;
        }
        return program;
      },
      kVocabulary);
  EXPECT_TRUE(lower(input));
  EXPECT_EQ(calls, 1);
  for (fault = 1; fault <= 6; ++fault) EXPECT_FALSE(lower(input)) << fault;
  fault = 0;
  const int before = calls;
  make_graph(fm::Encoding::kF16, kVocabulary);
  EXPECT_FALSE(lower(input));
  make_graph(fm::Encoding::kF32, kVocabulary - 1);
  EXPECT_FALSE(lower(input));
  make_graph(fm::Encoding::kF32, kVocabulary);
  shape.segments[1].outputs = 0;
  EXPECT_FALSE(lower(input));
  shape.segments[1].outputs = 1;
  step.features = step.logits;
  EXPECT_FALSE(lower(input));
  step.features = {};
  step.logits = ir::TensorId{10000};
  EXPECT_FALSE(lower(input));
  EXPECT_EQ(calls, before);  // Invalid semantic heads never reach the backend.
}

class CorpusRunner : public ::testing::Test {
 protected:
  struct Lifetime {
    en::PagedNode node{en::NodeSettings{
        .compute_streams = 1, .slots = 2, .copy_lane = true, .slot_bytes = kExtent}};
    std::unique_ptr<en::ModelRunner> runner;
    ts::FakeStepSource* source = nullptr;
    std::optional<en::WaveComposer> composer;
    std::unique_ptr<ts::CorpusScore> scorer;
  };
  std::unique_ptr<Lifetime> life;
  std::array<std::int32_t, 7> a{1, 4, 3, 7, 2, 11, 5};
  std::array<std::int32_t, 4> b{9, 2, 6, 1};
  std::array<ts::CorpusDocument, 2> Documents() { return {{{2, a}, {0, b}}}; }

  void SetUp() override {
    life = std::make_unique<Lifetime>();
    auto& node = life->node;
    // NOLINTNEXTLINE(concurrency-mt-unsafe): immutable test environment.
    const char* scratch = std::getenv("LLMP_TEST_SCRATCH");
    ASSERT_NE(scratch, nullptr);
    std::filesystem::create_directories(scratch);
    ASSERT_TRUE(node.Open());
    const auto fixed = node.catalog().OccupancyOf(node.domain()).Total().value();
    const std::vector<std::pair<std::uint32_t, md::StateContract>> layers{
        {0,
         {.tensors = {{.name = "kv",
                       .kind = md::StateKind::kKv,
                       .cell_bytes = 256,
                       .rollback = md::Rollback::kTruncate}}}}};
    auto layout = md::Layout(layers, 64, 8);
    ASSERT_TRUE(layout);
    ts::FakeSourceOptions options;
    options.layouts = {*layout};
    options.logits = kVocabulary;
    auto source = std::make_unique<ts::FakeStepSource>(std::move(options));
    life->source = source.get();
    en::RunnerConfig config;
    config.slots = {.model = "corpus",
                    .slots = 3,
                    .stages = {{.name = "target", .layout = *layout, .read_align = 0}},
                    .spec = {},
                    .read_align = 256,
                    .spill_place = {},
                    .spill_directory = scratch,
                    .invalidate = {},
                    .drop_plans = {}};
    config.setup_shapes = {{.stage = 0,
                            .phase = pl::Phase::kScore,
                            .segments = {{.slot = 0, .rows = 8, .read_cells = 64, .outputs = 8},
                                         {.slot = 2, .rows = 8, .read_cells = 64, .outputs = 8}},
                            .read_align = 256}};
    config.launch = []() -> std::expected<std::unique_ptr<kn::StepLaunch>, std::string> {
      return std::make_unique<kn::RecordingLaunch>();
    };
    auto runner = en::ModelRunner::Create(node, std::move(config), std::move(source), 0, 0);
    ASSERT_TRUE(runner) << ts::Failed(runner).value_or("");
    life->runner = std::move(*runner);
    auto setup = life->runner->Setup();
    ASSERT_TRUE(setup) << ts::Failed(setup).value_or("");
    ASSERT_TRUE(node.MapWorkspace(life->runner->activations_needed(),
                                  std::max(life->runner->pool_needed(), kExtent)));
    ASSERT_TRUE(node.Start(Bytes(fixed + 64 * kExtent)));
    node.SetHostFloor(0);
    ASSERT_TRUE(life->runner->Register());
    node.Run();
    ASSERT_TRUE(life->runner->Bind());
    ASSERT_TRUE(life->runner->slots().SelectSlots(std::vector<std::uint32_t>{0, 2}));
    ASSERT_TRUE(node.BeginRequest(0, life->runner->slots().execution(), "corpus test"));
    md::Batching caps;
    caps.max_segments = 3;
    auto composer = en::WaveComposer::For(caps, life->runner->slots());
    ASSERT_TRUE(composer);
    life->composer.emplace(std::move(*composer));
  }
  void TearDown() override {
    if (!life || !life->runner) return;
    if (life->node.InRequest(0)) EXPECT_TRUE(life->node.EndRequest(0));
    std::array<en::PagedModel*, 1> models{life->runner.get()};
    auto retired = life->node.TearDown(models);
    EXPECT_TRUE(retired) << ts::Failed(retired).value_or("");
    // Every borrowed source/layout/stream/plan/scoring owner survives an
    // unproven teardown. A destructor is not a completion proof.
    if (!retired) (void)life.release();
  }
  void Make(std::uint32_t chunk = 3) {
    auto score = ts::CorpusScore::Make(life->node, *life->runner, *life->composer, Documents(),
                                       {kVocabulary, chunk});
    ASSERT_TRUE(score) << ts::Failed(score).value_or("");
    life->scorer = std::move(*score);
  }
};

TEST_F(CorpusRunner, RaggedJoinedOwnersScoreEveryNextTargetAcrossChunkBoundaries) {
  Make();
  ASSERT_NE(life->scorer, nullptr);
  const auto charged = life->node.host_charged();
  EXPECT_GE(charged, life->scorer->charged_bytes());
  for (int i = 0; i < 3; ++i) {
    auto step = life->scorer->Step();
    ASSERT_TRUE(step) << ts::Failed(step).value_or("");
    EXPECT_EQ(*step, i == 2);
  }
  const auto& result = life->scorer->result();
  EXPECT_EQ(result.inputs, 11U);
  EXPECT_EQ(result.targets, 9U);
  EXPECT_EQ(result.steps, 3U);  // Real joined waves, not two scalar loops.
  double expected = 0;
  for (const auto& d : Documents())
    for (std::size_t i = 0; i + 1 < d.tokens.size(); ++i) {
      const auto chosen = ts::FakeLogit(d.tokens[i], static_cast<std::int32_t>(i), kVocabulary);
      expected += std::log(std::exp(1.0) + 16.0) - (d.tokens[i + 1] == chosen ? 1.0 : 0.0);
    }
  EXPECT_NEAR(result.total_nll, expected, 1e-13);
  EXPECT_NEAR(result.perplexity(), std::exp(expected / 9), 1e-13);
  EXPECT_EQ(life->runner->slots().positions(2), a.size());
  EXPECT_EQ(life->runner->slots().positions(0), b.size());
  ASSERT_TRUE(life->scorer->Step());
  EXPECT_EQ(life->scorer->result().targets, 9U);
}

TEST_F(CorpusRunner, CancellationDoesNotAdvanceOrReportCompletion) {
  Make();
  ASSERT_NE(life->scorer, nullptr);
  ASSERT_TRUE(life->scorer->Step());
  life->scorer->Cancel();
  EXPECT_FALSE(life->scorer->Step());
  EXPECT_FALSE(life->scorer->complete());
  EXPECT_EQ(life->runner->slots().positions(2), 3U);
  EXPECT_EQ(life->runner->slots().positions(0), 3U);
}

TEST_F(CorpusRunner, UnknownRunIsTerminalAndWholeOwnerRetires) {
  Make();
  ASSERT_NE(life->scorer, nullptr);
  life->source->options().fail_run = kn::StepError::kUnknown;
  life->source->options().fail_phase = pl::Phase::kScore;
  EXPECT_FALSE(life->scorer->Step());
  EXPECT_FALSE(life->scorer->complete());
  const auto count = life->source->built().size();
  EXPECT_FALSE(life->scorer->Step());
  EXPECT_EQ(life->source->built().size(), count);
}

TEST_F(CorpusRunner, AdmissionRefusalLeavesSlotsAndCallerIdsUntouched) {
  const auto before = life->node.host_charged();
  auto score =
      ts::CorpusScore::Make(life->node, *life->runner, *life->composer, Documents(), {1048576, 8});
  EXPECT_FALSE(score);
  EXPECT_EQ(life->node.host_charged(), before);
  EXPECT_EQ(life->runner->slots().positions(2), 0U);
  EXPECT_EQ(a[0], 1);
}

TEST_F(CorpusRunner, CorpusOwnsIdsAndRejectsExternallyAdvancedSlot) {
  Make();
  ASSERT_NE(life->scorer, nullptr);
  a.fill(-1);  // No borrowed caller token buffer remains.
  ASSERT_TRUE(life->scorer->Step());
  ASSERT_TRUE(life->runner->slots().SetPositions(2, 4));
  EXPECT_FALSE(life->scorer->Step());
  EXPECT_FALSE(life->scorer->complete());
}
}  // namespace
