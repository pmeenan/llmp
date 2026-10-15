// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <format>
#include <memory>
#include <span>
#include <string>
#include <system_error>
#include <vector>

#include "artifact_writer.h"
#include "base/bytes.h"
#include "base/sha256.h"
#include "components/step.h"
#include "engine/execution_contract.h"
#include "engine/model_runner.h"
#include "expected_error.h"
#include "ir/builder.h"
#include "kernels/decode/selected_rows.h"
#include "kernels/ggml/implementations.h"
#include "kernels/ggml/step_program.h"
#include "kernels/ggml/tensors.h"
#include "model/state_contract.h"
#include "planner/plan.h"
#include "planner/select.h"
#include "platform/host_probe.h"
#include "providers/device_runtime.h"

namespace {
namespace de = llmp::kernels::decode;
namespace kg = llmp::kernels::ggml;
namespace kn = llmp::kernels;
namespace en = llmp::engine;
namespace ir = llmp::ir;
namespace co = llmp::components;
namespace pl = llmp::planner;
namespace fmt = llmp::format;
namespace md = llmp::model;
namespace ts = llmp::test;
using Role = fmt::PartRole;
using llmp::base::Bytes;
constexpr ir::WeightRef kWeight{.stage = 0, .component = 7, .role = 0};
constexpr std::uint32_t kWidth = 32, kTableRows = 9, kRows = 4, kContext = 256;
constexpr std::uint64_t kDomain = 256ULL << 20;
constexpr std::uint64_t kMetadata = 16ULL << 20;

fmt::TensorFormat Format(bool scale = true) {
  return {.encoding = fmt::Encoding::kNvfp4,
          .params = {.tensor_scale = scale},
          .layout = fmt::Layout::kSplit,
          .shape = ir::ShapeOf({kWidth, kTableRows})};
}
pl::StepShape Shape(std::uint32_t past = 0) {
  return {.stage = 0,
          .phase = pl::Phase::kPrefill,
          .segments = {{.slot = 0,
                        .rows = kRows,
                        .past = past,
                        .read_cells = kContext,
                        .outputs = kRows,
                        .want_head = true}},
          .read_align = 256};
}
struct Descriptors {
  kg::TensorArena arena;
  std::array<kg::SelectedOperand, 2> inputs;
  std::array<kg::SelectedOperand, 1> outputs;
  ir::Node node;
  explicit Descriptors(bool scale) : arena(std::move(*kg::TensorArena::Create(8))) {
    auto* ctx = arena.context();
    auto& w = inputs[0];
    w.desc.format = Format(scale);
    w.paged_weight = kWeight;
    w.parts = {{Role::kCodes, ggml_new_tensor_2d(ctx, GGML_TYPE_I8, kWidth / 2, kRows)},
               {Role::kBlockScales, ggml_new_tensor_2d(ctx, GGML_TYPE_I8, kWidth / 16, kRows)}};
    if (scale) w.parts.push_back({Role::kTensorScale, ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1)});
    w.tensor = w.parts.front().tensor;
    auto& ids = inputs[1];
    ids.desc.format = ir::Dense(fmt::Encoding::kI32, ir::ShapeOf({kRows}));
    ids.desc.nb = ir::NaturalStrides(ids.desc.format);
    ids.tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, kRows);
    ids.side_rows_component = kWeight.component;
    auto& out = outputs[0];
    out.desc.format = ir::Dense(fmt::Encoding::kF32, ir::ShapeOf({kWidth, kRows}));
    out.desc.nb = ir::NaturalStrides(out.desc.format);
    out.tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kWidth, kRows);
    node.key = ir::Primitive(ir::Op::kGetRows);
    node.nsrc = 2;
    node.nout = 1;
  }
};
TEST(DecodeSelectedRows, CompletePartsKeepOriginalLogicalShapeAndBoundedPhysicalSlots) {
  auto source = de::RowsSelectedSource();
  ASSERT_TRUE(source) << source.error();
  const auto impl = (*source)->Implementations();
  ASSERT_EQ(impl.size(), 2U);
  const auto at = std::ranges::find(impl, "llmp.decode.rows.nvfp4.split",
                                    &llmp::execution::Implementation::name);
  ASSERT_NE(at, impl.end());
  const auto& nvfp4_impl = *at;
  for (bool scale : {false, true}) {
    Descriptors d(scale);
    auto call = (*source)->Lower(nvfp4_impl, d.node, d.inputs, d.outputs);
    ASSERT_TRUE(call) << call.error().detail;
    EXPECT_EQ((*call)->launches(), 1U);
    EXPECT_EQ((*call)->scratch_bytes(), 0U);
    EXPECT_LE((*call)->host_bytes(), 2048U);
    EXPECT_EQ(d.inputs[0].desc.format.shape.ne[1], kTableRows);
    EXPECT_EQ(d.inputs[0].parts[0].tensor->ne[1], kRows);
  }
}
TEST(DecodeSelectedRows, MissingDuplicateGeometryAndUncheckedIdsRefuse) {
  auto source = de::RowsSelectedSource();
  ASSERT_TRUE(source);
  const auto impl = (*source)->Implementations();
  const auto at = std::ranges::find(impl, "llmp.decode.rows.nvfp4.split",
                                    &llmp::execution::Implementation::name);
  ASSERT_NE(at, impl.end());
  const auto& nvfp4_impl = *at;
  for (int bad = 0; bad != 9; ++bad) {
    SCOPED_TRACE(bad);
    Descriptors d(true);
    switch (bad) {
      case 0:
        d.inputs[0].parts.pop_back();
        break;
      case 1:
        d.inputs[0].parts[1].role = Role::kCodes;
        break;
      case 2:
        d.inputs[0].parts[1].tensor->ne[1] = kRows + 1;
        break;
      case 3:
        d.inputs[0].paged_weight.reset();
        break;
      case 4:
        d.inputs[1].side_rows_component.reset();
        break;
      case 5:
        d.inputs[1].side_rows_component = kWeight.component + 1;
        break;
      case 6:
        d.inputs[0].paged_weight->expert_array = true;
        break;
      case 7:
        d.outputs[0].desc.format.encoding = fmt::Encoding::kF16;
        break;
      case 8:
        d.inputs[0].parts[0].tensor->nb[1] += 8;
        break;
    }
    EXPECT_FALSE((*source)->Lower(nvfp4_impl, d.node, d.inputs, d.outputs));
  }
}

TEST(DecodeSelectedRows, Nvfp4ImplementationRefusesAffineEncodingAndViceVersa) {
  auto source = de::RowsSelectedSource();
  ASSERT_TRUE(source);
  const auto impl = (*source)->Implementations();
  const auto nvfp4 = std::ranges::find(impl, "llmp.decode.rows.nvfp4.split",
                                       &llmp::execution::Implementation::name);
  ASSERT_NE(nvfp4, impl.end());
  const auto affine = std::ranges::find(impl, "llmp.decode.rows.affine.split",
                                        &llmp::execution::Implementation::name);
  ASSERT_NE(affine, impl.end());

  Descriptors d(false);
  d.inputs[0].desc.format.encoding = fmt::Encoding::kAffine;
  d.inputs[0].desc.format.params = {.bits = 4, .group = 32, .scale = fmt::AffineScale::kBF16};
  // nvfp4 implementation refuses affine weight
  EXPECT_FALSE((*source)->Lower(*nvfp4, d.node, d.inputs, d.outputs));

  Descriptors nvfp4_d(false);
  // affine implementation refuses nvfp4 weight
  EXPECT_FALSE((*source)->Lower(*affine, nvfp4_d.node, nvfp4_d.inputs, nvfp4_d.outputs));
}

TEST(DecodeSelectedRows, AffineSplitRowsLowerSuccessfully) {
  auto source = de::RowsSelectedSource();
  ASSERT_TRUE(source);
  const auto impl = (*source)->Implementations();
  const auto affine = std::ranges::find(impl, "llmp.decode.rows.affine.split",
                                        &llmp::execution::Implementation::name);
  ASSERT_NE(affine, impl.end());

  kg::TensorArena arena(std::move(*kg::TensorArena::Create(8)));
  auto* ctx = arena.context();
  std::array<kg::SelectedOperand, 2> inputs;
  std::array<kg::SelectedOperand, 1> outputs;
  ir::Node node;

  auto& w = inputs[0];
  w.desc.format = {.encoding = fmt::Encoding::kAffine,
                   .params = {.bits = 4, .group = 32, .scale = fmt::AffineScale::kBF16},
                   .layout = fmt::Layout::kSplit,
                   .shape = ir::ShapeOf({kWidth, kTableRows})};
  w.paged_weight = kWeight;
  w.parts = {{Role::kCodes, ggml_new_tensor_2d(ctx, GGML_TYPE_I8, kWidth / 2, kRows)},
             {Role::kBlockScales, ggml_new_tensor_2d(ctx, GGML_TYPE_I8, 2, kRows)},
             {Role::kBiases, ggml_new_tensor_2d(ctx, GGML_TYPE_I8, 2, kRows)}};
  for (std::size_t i = 0; i < w.parts.size(); ++i) {
    kg::TensorArena::Bind(w.parts[i].tensor, 0x100000ULL + i * 0x1000ULL);
  }
  w.tensor = w.parts.front().tensor;

  auto& ids = inputs[1];
  ids.desc.format = ir::Dense(fmt::Encoding::kI32, ir::ShapeOf({kRows}));
  ids.desc.nb = ir::NaturalStrides(ids.desc.format);
  ids.tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, kRows);
  ids.side_rows_component = kWeight.component;
  kg::TensorArena::Bind(ids.tensor, 0x200000ULL);

  auto& out = outputs[0];
  out.desc.format = ir::Dense(fmt::Encoding::kF32, ir::ShapeOf({kWidth, kRows}));
  out.desc.nb = ir::NaturalStrides(out.desc.format);
  out.tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kWidth, kRows);
  kg::TensorArena::Bind(out.tensor, 0x300000ULL);

  node.key = ir::Primitive(ir::Op::kGetRows);
  node.nsrc = 2;
  node.nout = 1;

  auto call = (*source)->Lower(*affine, node, inputs, outputs);
  ASSERT_TRUE(call) << call.error().detail;
  EXPECT_EQ((*call)->launches(), 1U);
  EXPECT_EQ((*call)->scratch_bytes(), 0U);
  EXPECT_LE((*call)->host_bytes(), 2048U);
}

TEST(DecodeSelectedRows, LoweringRequiresTheDirectSideInputAndPagedComponent) {
  auto source = de::RowsSelectedSource();
  ASSERT_TRUE(source);
  auto implementations = kg::Implementations();
  const auto native = (*source)->Implementations();
  implementations.insert(implementations.end(), native.begin(), native.end());
  auto registry = llmp::execution::Registry::Create(implementations);
  ASSERT_TRUE(registry) << registry.error().detail;
  const pl::DeviceFacts mixed_device{.device_class = "sm_121", .arch = 121};
  const pl::Policy policy;
  const pl::SelectionView view(*registry, mixed_device, policy, false, false);
  for (const auto encoding : {fmt::Encoding::kQ8_0, fmt::Encoding::kI32}) {
    for (const auto width : {32U, 256U}) {
      SCOPED_TRACE(width);
      ir::Builder builder;
      auto format = ir::Dense(encoding, ir::ShapeOf({width, kTableRows}));
      if (encoding == fmt::Encoding::kQ8_0) format.layout = fmt::Layout::kGgmlBlocks;
      const auto weight = builder.Weight(format, kWeight);
      const auto ids = builder.Input(ir::Dense(fmt::Encoding::kI32, ir::ShapeOf({kRows})));
      const auto rows = builder.GetRows(weight, ids);
      builder.Output(rows);
      auto graph = std::move(builder).Finish();
      ASSERT_TRUE(graph) << graph.error().detail;
      const auto candidates = view.Candidates(*graph, graph->node(graph->producer(rows)));
      EXPECT_EQ(
          std::ranges::any_of(candidates,
                              [&](auto i) { return registry->at(i).name == "ggml.get_rows.ext"; }),
          encoding == fmt::Encoding::kI32 || width == 256);
    }
  }
  for (int bad = 0; bad != 4; ++bad) {
    SCOPED_TRACE(bad);
    ir::Builder builder;
    const auto weight = builder.Weight(Format(), kWeight);
    const auto ids = builder.Input(ir::Dense(fmt::Encoding::kI32, ir::ShapeOf({kRows})));
    const auto output = builder.GetRows(weight, ids);
    builder.Output(output);
    auto graph = std::move(builder).Finish();
    ASSERT_TRUE(graph) << graph.error().detail;
    co::StepGraph step{
        .graph = std::move(*graph),
        .inputs = {{.tensor = ids,
                    .kind = bad == 1 ? co::InputKind::kTokens : co::InputKind::kSideRows,
                    .component =
                        static_cast<std::uint16_t>(kWeight.component + (bad == 2 ? 1 : 0))}},
        .weights = {{.tensor = weight,
                     .ref = kWeight,
                     .parts = {{"codes", Role::kCodes},
                               {"scales", Role::kBlockScales},
                               {"scale", Role::kTensorScale}}}},
        .state = {},
        .logits = output};
    const pl::DeviceFacts device{.device_class = "sm_121", .arch = 121};
    auto plan = pl::Plan(step.graph.Clone(), *registry, device, {});
    ASSERT_TRUE(plan) << plan.error().detail;
    ASSERT_EQ(plan->steps.size(), 1U);
    EXPECT_EQ(registry->at(plan->steps.front().implementation).name,
              "llmp.decode.rows.nvfp4.split");
    const auto shape = Shape();
    const kn::Addresses addresses{.weight = {},
                                  .weight_rows = [bad](const ir::WeightRef& ref) -> std::uint64_t {
                                    return bad != 3 && ref == kWeight ? kRows : 0;
                                  },
                                  .state = {},
                                  .save = {}};
    auto lowered = kg::LowerStep({*plan, step, shape, addresses, kContext}, *registry, {},
                                 {.selected_source = source->get()});
    if (bad == 0) {
      ASSERT_TRUE(lowered) << lowered.error().detail;
      ASSERT_EQ(lowered->selected.size(), 1U);
      ASSERT_EQ(lowered->inputs.size(), 1U);
      EXPECT_EQ(lowered->inputs[0].kind, kn::InputKind::kSideRows);
      EXPECT_EQ(lowered->inputs[0].component, kWeight.component);
    } else {
      ASSERT_FALSE(lowered);
      EXPECT_EQ(lowered.error().detail,
                "selected rows require a complete split table and its checked side-row input");
    }
  }
}

// Bind's address validation queues nothing; these deliberately invalid
// bindings must refuse before touching the launch implementation.
TEST(DecodeSelectedRows, MisalignedOverlappingAndChangedPhysicalBindingsRefuse) {
  class RefusalLaunch final : public kg::SelectedLaunch {
   public:
    explicit RefusalLaunch(const kg::SelectedSource* owner) : kg::SelectedLaunch(owner, {}) {}
    bool faulted() const override { return false; }
  };
  auto source = de::RowsSelectedSource();
  ASSERT_TRUE(source);
  const auto impl = (*source)->Implementations();
  const auto at = std::ranges::find(impl, "llmp.decode.rows.nvfp4.split",
                                    &llmp::execution::Implementation::name);
  ASSERT_NE(at, impl.end());
  const auto& nvfp4_impl = *at;
  RefusalLaunch launch(source->get());
  EXPECT_FALSE(launch.capturable());
  for (int bad = 0; bad != 7; ++bad) {
    SCOPED_TRACE(bad);
    Descriptors d(true);
    auto call = (*source)->Lower(nvfp4_impl, d.node, d.inputs, d.outputs);
    ASSERT_TRUE(call);
    std::array<ggml_tensor*, 5> tensors{d.inputs[0].parts[0].tensor, d.inputs[0].parts[1].tensor,
                                        d.inputs[0].parts[2].tensor, d.inputs[1].tensor,
                                        d.outputs[0].tensor};
    for (std::size_t i = 0; i != tensors.size(); ++i)
      tensors[i]->data = reinterpret_cast<void*>(0x100000ULL + i * 0x10000ULL);
    if (bad < 5) {
      tensors[static_cast<std::size_t>(bad)]->data =
          reinterpret_cast<void*>(0x100001ULL + static_cast<unsigned>(bad) * 0x10000ULL);
      // Byte block scales have no alignment requirement; exercise their overlap.
      if (bad == 1) tensors[1]->data = tensors[0]->data;
    } else if (bad == 5) {
      tensors[4]->data = tensors[3]->data;
    } else {
      tensors[1]->ne[1] += 1;
    }
    EXPECT_FALSE((*call)->Bind(nvfp4_impl, launch, {}));
  }
}

// This test source only supplies a three-node generic IR step. The production
// runner supplies every physical row slot, checked ID remap and launch.
class RowsSource final : public en::StepSource {
 public:
  explicit RowsSource(kg::GgmlBackend& backend) : backend_(backend) {}
  pl::PlanKey KeyBase() const override { return {}; }
  std::expected<std::unique_ptr<kn::StepProgram>, std::string> Build(
      const pl::StepShape& shape, const kn::Addresses& addresses) override {
    if (shape != Shape() || shape.rows() != kRows)
      return std::unexpected("synthetic row shape differs");
    ir::Builder b;
    const auto weight = b.Weight(Format(), kWeight);
    const auto ids = b.Input(ir::Dense(fmt::Encoding::kI32, ir::ShapeOf({kRows})));
    const auto output = b.GetRows(weight, ids);
    b.Output(output);
    auto graph = std::move(b).Finish();
    if (!graph) return std::unexpected(graph.error().detail);
    co::StepGraph step{.graph = std::move(*graph),
                       .inputs = {{.tensor = ids,
                                   .kind = co::InputKind::kSideRows,
                                   .component = kWeight.component}},
                       .weights = {{.tensor = weight,
                                    .ref = kWeight,
                                    .parts = {{"codes", Role::kCodes},
                                              {"scales", Role::kBlockScales},
                                              {"scale", Role::kTensorScale}}}},
                       .state = {},
                       .logits = output};
    const auto backend = backend_.Backend();
    auto plan = pl::Plan(step.graph.Clone(), *backend.registry, backend.device, {},
                         {.capturable = true,
                          .staged = std::vector{ids},
                          .on_stream = backend.on_stream,
                          .written = backend.written,
                          .within = backend.within,
                          .shape = &shape});
    if (!plan) return std::unexpected(plan.error().detail);
    if (step.graph.tensor_count() != 3 || plan->graph.tensor_count() != 3 ||
        step.graph.nodes().size() != 3 || plan->graph.nodes().size() != 3 ||
        plan->steps.size() != 1 ||
        backend.registry->at(plan->steps[0].implementation).name != "llmp.decode.rows.nvfp4.split")
      return std::unexpected(std::format(
          "synthetic row plan escaped its paid one-operation domain: tensors {}/{}, nodes {}/{}, "
          "steps {}, first {}",
          step.graph.tensor_count(), plan->graph.tensor_count(), step.graph.nodes().size(),
          plan->graph.nodes().size(), plan->steps.size(),
          plan->steps.empty() ? std::string_view{"none"}
                              : backend.registry->at(plan->steps.front().implementation).name));
    ++builds;
    return backend_.Lower({*plan, step, shape, addresses, kContext});
  }
  std::size_t builds = 0;

 private:
  kg::GgmlBackend& backend_;
};

struct Lifetime {
  en::PagedNode node{{.compute_streams = 1, .slots = 1, .slot_bytes = en::kPagedExtent}};
  std::shared_ptr<en::ExecutionMetadataAllowance> payment;
  std::shared_ptr<const kg::SelectedSource> selected;
  std::unique_ptr<kg::GgmlBackend> backend;
  std::unique_ptr<en::ModelRunner> runner;
  std::vector<en::PagedModel*> entered;
  std::filesystem::path root;
  bool opened = false;
};
struct Retire {
  void operator()(Lifetime* life) const {
    if (!life) return;
    if (life->opened) {
      const auto retired = life->node.TearDown(life->entered);
      EXPECT_TRUE(retired) << (retired ? "" : retired.error());
      if (!retired) return;  // retain every borrower and source until process exit
    }
    if (!life->root.empty()) {
      std::error_code error;
      std::filesystem::remove_all(life->root, error);
      EXPECT_FALSE(error) << error.message();
    }
    delete life;
  }
};
using Owner = std::unique_ptr<Lifetime, Retire>;

// A small independent exact decoder oracle: code1=+0.5, code2=+1,
// E4M3 scales encode row+1 and twice that value; global scale=2. Every
// source row and scale block is distinct, and all oracle products are exact.
float Expected(std::int32_t id, std::size_t column) {
  const float code = ((static_cast<std::size_t>(id) + column) % 2 == 0) ? 0.5F : 1.F;
  return code * static_cast<float>(id + 1) * (column < 16 ? 1.F : 2.F) * 2.F;
}
std::filesystem::path MakeArtifact(const std::filesystem::path& root) {
  ts::V1Plan plan;
  plan.spec = R"({"pipeline":"llm","stages":[],"version":1})"
              "\n";
  const std::array roles{Role::kCodes, Role::kBlockScales, Role::kTensorScale};
  const std::array names{"codes", "scales", "scale"};
  for (std::size_t i = 0; i != roles.size(); ++i)
    plan.groups.push_back(
        {.kind = llmp::artifact::GroupKind::kGlobal,
         .resources = {{.name = names[i], .format = Format(), .part = roles[i], .rows = i < 2}},
         .new_shard = i != 0});
  auto files = ts::Render(plan);
  // Each part is the only resource at offset0 of its own shard/group.
  // Update the corresponding chunk identity after filling its actual bytes.
  EXPECT_EQ(files.shards.size(), 3U);
  for (std::size_t p = 0; p != 3; ++p) {
    auto& shard = files.shards[p];
    std::uint64_t header = 0;
    std::memcpy(&header, shard.data(), sizeof(header));
    const auto offset = header + 8;
    if (p == 0) {
      for (std::size_t row = 0; row != kTableRows; ++row)
        for (std::size_t col = 0; col != kWidth; col += 2)
          shard[offset + row * (kWidth / 2) + col / 2] =
              static_cast<char>(row % 2 == 0 ? 0x21 : 0x12);
    } else if (p == 1) {
      constexpr std::array<unsigned, kTableRows> scales{0x38, 0x40, 0x44, 0x48, 0x4a,
                                                        0x4c, 0x4e, 0x50, 0x51};
      for (std::size_t row = 0; row != kTableRows; ++row)
        for (std::size_t block = 0; block != kWidth / 16; ++block)
          shard[offset + row * (kWidth / 16) + block] = static_cast<char>(scales[row] + block * 8);
    } else {
      const float scale = 2.F;
      std::memcpy(shard.data() + offset, &scale, sizeof(scale));
    }
    llmp::base::Sha256 hash;
    hash.Update(std::string_view(shard).substr(offset));
    files.index["chunk_sha256"].at(p) = llmp::base::ToHex(hash.Finish());
  }
  return ts::Publish(root, files);
}

void PagedCase(int malformed) {
  const auto available = llmp::platform::AvailableMemoryBytes();
  ASSERT_TRUE(available);
  ASSERT_GE(*available, kDomain + (6ULL << 30));
  Owner life(new Lifetime);
  ASSERT_TRUE(life->node.Open());
  life->opened = true;
  // Fixed tiny domain paid before artifact/parser/compiler/result allocation.
  // 16MiB covers their simultaneous inventory; native/row slots are separately
  // cataloged. The wrapper pays process+file-cache and driver margins as well.
  life->node.SetHostFloor(0);
  auto payment = en::ExecutionMetadataAllowance::Admit(life->node.catalog(), life->node.domain(),
                                                       kMetadata, *available, 6ULL << 30);
  ASSERT_TRUE(payment) << payment.error();
  life->payment = *payment;
  ASSERT_TRUE(life->payment->Charge(8ULL << 20));
  // One directory per test: CTest runs this file's tests in parallel from
  // one working directory, and each removes its artifact when it ends.
  life->root = std::filesystem::path("scratch") /
               (std::string("native-split-rows-") +
                ::testing::UnitTest::GetInstance()->current_test_info()->name());
  auto path = MakeArtifact(life->root);
  auto selected = de::RowsSelectedSource();
  ASSERT_TRUE(selected) << selected.error();
  life->selected = *selected;
  auto backend = kg::GgmlBackend::Create(0, life->node.execution(), life->node.stream(0), nullptr,
                                         {.selected = life->selected});
  ASSERT_TRUE(backend) << backend.error();
  life->backend = std::move(*backend);
  auto weights = std::make_unique<en::PagedWeights>();
  const auto opened = weights->Open(path);
  ASSERT_TRUE(opened) << llmp::test_support::Failed(opened).value_or("");
  auto* weight_owner = weights.get();
  const std::array<std::pair<std::uint32_t, md::StateContract>, 1> state{
      {{0,
        {.tensors = {{.name = "fixture-state",
                      .kind = md::StateKind::kRecurrent,
                      .encoding = md::Encoding::kF32,
                      .cell_bytes = 4,
                      .cells = md::CellRule::kFixed,
                      .window_or_ratio = 1,
                      .rollback = md::Rollback::kSnapshot}}}}}};
  auto layout = md::Layout(state, kContext, kRows);
  ASSERT_TRUE(layout) << layout.error();
  en::RunnerConfig config;
  config.slots = {.model = "split rows",
                  .slots = 1,
                  .stages = {{.name = "target", .layout = *layout}},
                  .read_align = 256,
                  .spill_place = {},
                  .spill_directory = life->root,
                  .invalidate = {},
                  .drop_plans = {}};
  config.weights.push_back(
      {.weights = std::move(weights),
       .places = {en::GroupPlace::kNone, en::GroupPlace::kNone, en::GroupPlace::kDevice},
       .slabs = {}});
  config.setup_shapes = {Shape()};
  for (std::uint32_t p = 0; p != 2; ++p)
    config.row_tables.push_back({.weight = kWeight,
                                 .resource = p,
                                 .format = Format(),
                                 .part = p == 0 ? Role::kCodes : Role::kBlockScales,
                                 .rows_per_token = 1});
  config.weight_address = [weight_owner](const ir::WeightRef& ref, Role part) {
    return ref == kWeight && part == Role::kTensorScale ? weight_owner->resource_address(2) : 0;
  };
  config.launch = [owner = life.get()]() {
    return owner->backend->Launch(
        {.base = owner->node.pool().base, .size = Bytes(owner->node.pool().bytes)});
  };
  switch (malformed) {
    case 1:
      config.row_tables.pop_back();
      break;
    case 2:
      config.row_tables.push_back(config.row_tables.front());
      break;
    case 3:
      config.row_tables[1].rows_per_token = 2;
      break;
    case 4:
      config.row_tables[1].format.shape.ne[1] -= 1;
      break;
    case 5:
      config.row_tables[1].resource = 0;
      break;
    case 6:
      for (auto& part : config.row_tables) part.rows_per_token = UINT64_MAX;
      break;
    default:
      break;
  }
  config.retire = [owner = life.get()] { owner->backend->LaunchReleased(); };
  auto source = std::make_unique<RowsSource>(*life->backend);
  auto* source_owner = source.get();
  auto runner = en::ModelRunner::Create(life->node, std::move(config), std::move(source), 0, 0);
  ASSERT_TRUE(runner) << runner.error();
  life->runner = std::move(*runner);
  life->entered.push_back(life->runner.get());
  const auto setup = life->runner->Setup();
  if (malformed != 0) {
    ASSERT_FALSE(setup);
    constexpr std::array<const char*, 6> reasons{
        "a row table has missing or duplicate row parts",
        "a row table has missing or duplicate row parts",
        "a row table's parts disagree on logical geometry",
        "a row table's parts disagree on logical geometry",
        "a row table's format or part differs from its source resource",
        "a row table's capacity overflows"};
    EXPECT_EQ(setup.error(), reasons[static_cast<std::size_t>(malformed - 1)]);
    EXPECT_EQ(source_owner->builds, 0U);
    EXPECT_EQ(weight_owner->bytes(), 0U);
    return;
  }
  ASSERT_TRUE(setup) << setup.error();
  ASSERT_TRUE(life->node.MapWorkspace(life->runner->activations_needed(),
                                      std::max(en::kPagedExtent, life->runner->pool_needed())));
  ASSERT_TRUE(life->node.Start(Bytes(kDomain)));
  EXPECT_EQ(weight_owner->group_address(0), 0U);
  EXPECT_EQ(weight_owner->group_address(1), 0U);
  EXPECT_NE(weight_owner->group_address(2), 0U);
  ASSERT_TRUE(life->runner->Register());
  ASSERT_TRUE(life->runner->Bind());
  life->node.Run();
  std::vector<en::LoadStats> loaded;
  ASSERT_TRUE(life->node.Load(life->runner->weights(), "tensor scale", loaded));
  const std::array<std::uint32_t, 1> owner{0};
  ASSERT_TRUE(life->runner->slots().SelectSlots(owner));
  const auto done = life->node.WithRequest(
      0, life->runner->slots().execution(), "split rows", [&]() -> en::Status {
        const std::array<std::int32_t, kRows> tokens{1, 2, 3, 4};
        const en::SegmentWork work{.tokens = tokens};
        std::uint32_t past = 0;
        const std::array<std::array<std::int32_t, kRows>, 3> inputs{
            {{8, 1, 8, 0}, {3, 5, 3, 2}, {0, 8, 4, 0}}};
        for (std::size_t turn = 0; turn != inputs.size(); ++turn) {
          const auto shape = Shape(past);
          for (const auto bad : {std::array<std::int32_t, kRows>{0, 1, 2, -1},
                                 std::array<std::int32_t, kRows>{0, 1, 2, kTableRows}}) {
            const auto before = life->runner->slots().positions(0);
            const auto refused = life->runner->RunStep(
                {.shape = shape, .work = std::span(&work, 1), .side_rows = bad, .ahead = {}});
            if (refused || !refused.error().clean || before != life->runner->slots().positions(0))
              return std::unexpected("invalid IDs escaped pre-effect refusal");
          }
          auto ran = life->runner->RunStep({.shape = shape,
                                            .work = std::span(&work, 1),
                                            .side_rows = inputs[turn],
                                            .ahead = {}});
          if (!ran) return std::unexpected(ran.error().what);
          if (ran->outputs.size() != 1 || ran->outputs[0].size() != kRows * kWidth * sizeof(float))
            return std::unexpected("row output coverage differs");
          for (std::size_t row = 0; row != kRows; ++row)
            for (std::size_t col = 0; col != kWidth; ++col) {
              float value = 0;
              std::memcpy(&value, ran->outputs[0].data() + (row * kWidth + col) * sizeof(float),
                          sizeof(float));
              if (std::bit_cast<std::uint32_t>(value) !=
                  std::bit_cast<std::uint32_t>(Expected(inputs[turn][row], col)))
                return std::unexpected("paged row differs from independent code/scale oracle");
            }
          constexpr std::array paths{en::RunPath::kEager, en::RunPath::kCaptured,
                                     en::RunPath::kReplayed};
          if (ran->path != paths[turn])
            return std::unexpected(
                std::format("row wave {} did not use expected eager/capture/replay path: got {}",
                            turn, static_cast<int>(ran->path)));
          past += kRows;
        }
        if (source_owner->builds != 2) return std::unexpected("paged IDs caused extra planning");
        return {};
      });
  ASSERT_TRUE(done) << done.error();
}
TEST(DecodeSelectedRowsGpu, PagedPartsRemapAndReplayChangedIdsWithoutGrowingWeights) {
  PagedCase(0);
}
TEST(DecodeSelectedRowsGpu, IncompleteOrInconsistentTablePartsRefuseBeforePlanningOrReads) {
  for (int malformed = 1; malformed != 7; ++malformed) {
    SCOPED_TRACE(malformed);
    PagedCase(malformed);
    if (::testing::Test::HasFailure()) return;
  }
}
}  // namespace
