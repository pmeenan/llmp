// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The shared-Q8 comparison is a validation-only physical/numerical witness.
// It records no latency, timing rank, or product winner.
#include <cuda_runtime.h>
#include <ggml-common.h>
#include <ggml-quants.h>
#include <ggml.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "components/comparison_boundary.h"
#include "engine/runner_resources.h"
#include "format/encoding.h"
#include "ir/builder.h"
#include "kernels/common/comparison.h"
#include "kernels/ggml/comparison.h"
#include "kernels/ggml/llmp_ops.h"
#include "kernels/ggml/ops_ext.h"
#include "kernels/ggml/step_program.h"
#include "kernels/ggml/validate_ext.h"
#include "kernels/reference/decode.h"
#include "kernels/reference/exec.h"
#include "planner/alternative.h"
#include "platform/host_probe.h"
#include "providers/device_execution.h"

namespace {
namespace co = llmp::components;
namespace en = llmp::engine;
namespace fmt = llmp::format;
namespace ir = llmp::ir;
namespace kn = llmp::kernels;
namespace kg = llmp::kernels::ggml;
namespace pl = llmp::planner;
namespace ref = llmp::kernels::reference;

constexpr std::uint64_t kMiB = 1ULL << 20U;
constexpr std::uint64_t kK = 2304;
constexpr std::uint64_t kN = 64;
constexpr std::uint64_t kArenaBytes = 8 * kMiB;
constexpr std::uint64_t kPoolBytes = 4 * kMiB;

struct CudaBuffer {
  void* data = nullptr;
  ~CudaBuffer() {
    if (data) EXPECT_EQ(cudaFree(data), cudaSuccess);
  }
  bool Allocate(std::size_t bytes) { return cudaMalloc(&data, bytes) == cudaSuccess; }
  std::uint64_t address() const { return reinterpret_cast<std::uintptr_t>(data); }
};

class PaidBoundaryFunder final : public ir::StorageFunder {
 public:
  explicit PaidBoundaryFunder(std::shared_ptr<en::ExecutionMetadataAllowance> paid)
      : paid_(std::move(paid)) {}
  bool Charge(std::uint64_t bytes) const override { return paid_->Charge(bytes); }
  void Release(std::uint64_t bytes) const override { paid_->Release(bytes); }

 private:
  std::shared_ptr<en::ExecutionMetadataAllowance> paid_;
};

struct GpuLife final : en::PagedModel {
  en::PagedNode node{{.compute_streams = 1, .slots = 1}};
  en::RunnerResources resources{node, 0, 0};
  std::unique_ptr<kg::GgmlBackend> backend;
  std::shared_ptr<en::ExecutionMetadataAllowance> paid;
  bool opened = false;
  llmp::catalog::Closure empty;

  std::uint32_t stream() const override { return 0; }
  const llmp::catalog::Closure& fence_closure() const override { return empty; }
  std::vector<llmp::catalog::ExtentId> managed_extents() const override { return {}; }
  en::Status Release() override {
    backend.reset();
    std::vector<std::string> errors;
    resources.Release(errors);
    if (!errors.empty()) return std::unexpected(errors.front());
    opened = false;
    return {};
  }
};

class GgmlComparisonPreparationGpu : public testing::TestWithParam<int> {
 protected:
  void SetUp() override {
    life_ = std::make_unique<GpuLife>();
    auto opened = life_->node.Open();
    ASSERT_TRUE(opened) << (opened ? "" : opened.error());
    life_->opened = true;
    const auto available = llmp::platform::AvailableMemoryBytes();
    ASSERT_TRUE(available);
    auto admitted = en::ExecutionMetadataAllowance::Admit(
        life_->node.catalog(), life_->node.domain(), 16 * kMiB, *available, 6ULL << 30U);
    ASSERT_TRUE(admitted) << (admitted ? "" : admitted.error());
    life_->paid = std::move(*admitted);
    auto backend =
        kg::GgmlBackend::Create(0, life_->node.execution(), life_->node.stream(0), nullptr);
    ASSERT_TRUE(backend) << (backend ? "" : backend.error());
    life_->backend = std::move(*backend);
    ASSERT_TRUE(life_->resources.OpenLibrary("shared Q8 comparison declaration"));
    const auto charge = [paid = life_->paid](std::uint64_t bytes) { return paid->Charge(bytes); };
    const auto release = [paid = life_->paid](std::uint64_t bytes) { paid->Release(bytes); };
    ASSERT_TRUE(life_->backend->FundConstruction(charge, release));
    ASSERT_TRUE(life_->backend->Declare({}, life_->resources.library_bytes(), charge, release));
    ASSERT_TRUE(life_->backend->LendLibrary(&life_->resources.library()));
  }

  void TearDown() override {
    if (!life_ || !life_->opened) return;
    const std::array<en::PagedModel*, 1> models{life_.get()};
    auto retired = life_->node.TearDown(models);
    EXPECT_TRUE(retired) << (retired ? "" : retired.error());
    if (retired) life_->opened = false;
  }

  std::unique_ptr<GpuLife> life_;
};

std::uint64_t RootOffset(const pl::StepPlan& plan, ir::TensorId tensor) {
  const auto root = plan.graph.root(tensor);
  const auto found = std::ranges::find_if(plan.placement.offsets, [&](const auto& entry) {
    return plan.graph.root(entry.first) == root;
  });
  return found == plan.placement.offsets.end() ? UINT64_MAX : found->second;
}

void EncodeExactQ4KOneHot(std::uint64_t column, std::uint64_t output,
                          std::vector<std::byte>& encoded) {
  const std::size_t blocks_per_row = static_cast<std::size_t>(kK / QK_K);
  auto* blocks = reinterpret_cast<block_q4_K*>(encoded.data());
  auto& block = blocks[static_cast<std::size_t>(output) * blocks_per_row + column / QK_K];
  block.d = ggml_fp32_to_fp16(0.5F);
  block.dmin = ggml_fp32_to_fp16(0.0F);
  // Every scale is exactly one and every minimum is zero in GGML's packed
  // six-bit scale/min representation.
  for (std::size_t group = 0; group < 4; ++group) {
    block.scales[group] = 1;
    block.scales[group + 8] = 1;
  }
  const std::size_t local = static_cast<std::size_t>(column % QK_K);
  const std::size_t group = local / 64;
  const std::size_t within = local % 64;
  const std::size_t pair = group * 32 + within % 32;
  if (within < 32)
    block.qs[pair] = static_cast<std::uint8_t>((block.qs[pair] & 0xf0U) | 1U);
  else
    block.qs[pair] = static_cast<std::uint8_t>((block.qs[pair] & 0x0fU) | 0x10U);
}

void EncodeExactQ6KOneHot(std::uint64_t column, std::uint64_t output,
                          std::vector<std::byte>& encoded) {
  const std::size_t blocks_per_row = static_cast<std::size_t>(kK / QK_K);
  auto* blocks = reinterpret_cast<block_q6_K*>(encoded.data());
  auto& block = blocks[static_cast<std::size_t>(output) * blocks_per_row + column / QK_K];
  block.d = ggml_fp32_to_fp16(0.5F);
  std::ranges::fill(block.scales, std::int8_t{1});
  std::ranges::fill(block.qh, std::uint8_t{0xaa});
  const std::size_t local = static_cast<std::size_t>(column % QK_K);
  const std::size_t chunk = local / 128;
  const std::size_t within = local % 128;
  const std::size_t lane = within % 32;
  const std::size_t low_index = chunk * 64 + ((within / 32) % 2) * 32 + lane;
  const std::size_t nibble_shift = within >= 64 ? 4U : 0U;
  block.ql[low_index] = static_cast<std::uint8_t>(
      (block.ql[low_index] & (nibble_shift == 0 ? 0xf0U : 0x0fU)) | (1U << nibble_shift));
}

void EncodeExactOneHotWeights(const fmt::TensorFormat& format, bool q6,
                              std::vector<std::byte>& encoded) {
  const auto parts = fmt::PartsOf(format);
  ASSERT_TRUE(parts && parts->size() == 1);
  encoded.assign(static_cast<std::size_t>(parts->parts()[0].bytes), std::byte{0});
  for (std::uint64_t row = 0; row < kN; ++row) {
    const std::uint64_t column = (row * 37 + (q6 ? 17 : 3)) % kK;
    if (q6)
      EncodeExactQ6KOneHot(column, row, encoded);
    else
      EncodeExactQ4KOneHot(column, row, encoded);
  }
}

TEST_P(GgmlComparisonPreparationGpu,
       SharedQ8PreparationOwnAlternativeHasBoundRecipeAndExactOutputs) {
  const std::uint64_t columns = static_cast<std::uint64_t>(GetParam());
  ir::Builder builder;
  builder.SetComponent(1);
  builder.SetOwner(0);
  const auto input = builder.Input(ir::Dense(fmt::Encoding::kF32, ir::ShapeOf({kK, columns})), "x");
  const ir::WeightRef q4_ref{.component = 1, .role = 0};
  const ir::WeightRef q6_ref{.component = 1, .role = 1};
  const auto q4 = builder.Weight({.encoding = fmt::Encoding::kQ4_K,
                                  .layout = fmt::Layout::kGgmlBlocks,
                                  .shape = ir::ShapeOf({kK, kN})},
                                 q4_ref, "q4");
  const auto q6 = builder.Weight({.encoding = fmt::Encoding::kQ6_K,
                                  .layout = fmt::Layout::kGgmlBlocks,
                                  .shape = ir::ShapeOf({kK, kN})},
                                 q6_ref, "q6");
  const auto out4 = builder.MatMul(q4, input);
  const auto out6 = builder.MatMul(q6, input);
  // Keep the shape-changing suffix, followed by an intentionally unsupported
  // comparison unary schema. Ordinary execution still supports this suffix.
  const auto suffix = builder.Unary(builder.Concat(out4, out6, 0), ir::UnaryFn::kRelu);
  builder.Output(out4);
  builder.Output(out6);
  builder.Output(suffix);
  auto graph = std::move(builder).Finish();
  ASSERT_TRUE(graph) << (graph ? "" : graph.error().detail);
  const std::array weights{
      co::StepWeight{.tensor = q4, .ref = q4_ref, .parts = {{"q4", fmt::PartRole::kData}}},
      co::StepWeight{.tensor = q6, .ref = q6_ref, .parts = {{"q6", fmt::PartRole::kData}}}};
  const co::GraphBuildIO io{*graph, graph->inputs(), graph->outputs(), weights};

  auto backend = life_->backend->Backend();
  const auto& recipe = backend.owner ? backend.owner->backend() : backend;
  const auto* pattern = pl::FindPattern("prepared_inputs");
  ASSERT_NE(pattern, nullptr);
  std::vector<ir::NodeId> products;
  for (std::uint32_t i = 0; i < graph->nodes().size(); ++i) {
    if (graph->node(ir::NodeId{i}).key == ir::Primitive(ir::Op::kMatMul)) products.emplace_back(i);
  }
  ASSERT_EQ(products.size(), 2U);
  const pl::ComparisonSiteSelector selector{.source_graph_identity = graph->identity(),
                                            .kind = pl::ComparisonSiteKind::kRewrite,
                                            .pattern = "prepared_inputs",
                                            .version = pattern->version,
                                            .member_origins = {{products[0]}, {products[1]}},
                                            .member_owner_segments = {},
                                            .boundary_ports = {}};
  const auto input_bytes = ir::StorageBytes(graph->tensor(input));
  const auto q4_parts = fmt::PartsOf(graph->tensor(q4).format);
  const auto q6_parts = fmt::PartsOf(graph->tensor(q6).format);
  ASSERT_TRUE(input_bytes && q4_parts && q4_parts->size() == 1 && q6_parts &&
              q6_parts->size() == 1);
  const std::array external{
      co::ComparisonExternalBinding{.original_root = input,
                                    .allocation_class = 1,
                                    .allocation_bytes = *input_bytes,
                                    .bytes = *input_bytes,
                                    .alignment = 256},
      co::ComparisonExternalBinding{.original_root = q4,
                                    .part = fmt::PartRole::kData,
                                    .allocation_class = 2,
                                    .allocation_bytes = q4_parts->parts()[0].readable,
                                    .bytes = q4_parts->parts()[0].readable,
                                    .alignment = 256},
      co::ComparisonExternalBinding{.original_root = q6,
                                    .part = fmt::PartRole::kData,
                                    .allocation_class = 3,
                                    .allocation_bytes = q6_parts->parts()[0].readable,
                                    .bytes = q6_parts->parts()[0].readable,
                                    .alignment = 256}};
  const co::ComparisonBoundaryView boundary_view{io, {}, 0, external};
  auto funder = std::make_shared<PaidBoundaryFunder>(life_->paid);
  const pl::Policy policy{};
  const pl::PlanOptions options{.capturable = true, .within = recipe.within};
  auto inventory = pl::BoundComparison(boundary_view, selector, funder, *recipe.registry,
                                       recipe.device, policy, options);
  ASSERT_TRUE(inventory) << (inventory ? "" : inventory.error().detail);
  ASSERT_TRUE(std::ranges::any_of(inventory->incumbent.plan.applied, [](const auto& applied) {
    return applied.pattern == "prepared_inputs";
  }));
  auto alternative = pl::MakeAlternative(*inventory);
  ASSERT_TRUE(alternative) << (alternative ? "" : alternative.error().detail);
  auto planned = pl::PlanAlternative(*inventory, std::move(*alternative), *recipe.registry,
                                     recipe.device, policy, options);
  ASSERT_TRUE(planned) << (planned ? "" : planned.error().detail);

  const std::array<ir::NodeId, 2> region_members{products[0], products[1]};
  const std::array<std::uint32_t, 3> region_groups{0, 1, 2};
  auto region_boundary = co::MakeComparisonRegionBoundary(
      {.source = boundary_view, .member_offsets = region_groups, .members = region_members},
      funder);
  ASSERT_TRUE(region_boundary) << (region_boundary ? "" : region_boundary.error().detail);
  const std::array<ir::NodeId, 1> partial_members{products[0]};
  const std::array<std::uint32_t, 2> partial_groups{0, 1};
  auto partial_boundary = co::MakeComparisonRegionBoundary(
      {.source = boundary_view, .member_offsets = partial_groups, .members = partial_members},
      funder);
  ASSERT_TRUE(partial_boundary) << (partial_boundary ? "" : partial_boundary.error().detail);

  CudaBuffer activation, q4_device, q6_device, pool, persistent;
  ASSERT_TRUE(activation.Allocate(kArenaBytes));
  ASSERT_TRUE(pool.Allocate(kPoolBytes));
  const auto q4_readable = q4_parts->parts()[0].readable;
  const auto q6_readable = q6_parts->parts()[0].readable;
  // Keep enough real backing after the readable weight extent to exercise a
  // scratch-pool overlap with a weight tail at Bind time.
  ASSERT_TRUE(q4_device.Allocate(static_cast<std::size_t>(q4_readable + kPoolBytes)));
  ASSERT_TRUE(q6_device.Allocate(static_cast<std::size_t>(q6_readable)));
  const auto persistent_bytes = life_->backend->persistent_bytes();
  if (persistent_bytes != 0) ASSERT_TRUE(persistent.Allocate(persistent_bytes));

  std::vector<std::byte> q4_host, q6_host;
  EncodeExactOneHotWeights(graph->tensor(q4).format, false, q4_host);
  EncodeExactOneHotWeights(graph->tensor(q6).format, true, q6_host);
  std::vector<float> q4_decoded(kK * kN), q6_decoded(kK * kN);
  const std::array q4_host_part{ref::PartView{.role = fmt::PartRole::kData, .bytes = q4_host}};
  const std::array q6_host_part{ref::PartView{.role = fmt::PartRole::kData, .bytes = q6_host}};
  ASSERT_TRUE(ref::DecodeRows(graph->tensor(q4).format, q4_host_part, 0, 0, kN, q4_decoded));
  ASSERT_TRUE(ref::DecodeRows(graph->tensor(q6).format, q6_host_part, 0, 0, kN, q6_decoded));
  for (std::uint64_t row = 0; row < kN; ++row) {
    const auto q4_nonzero =
        std::ranges::count_if(std::span(q4_decoded).subspan(static_cast<std::size_t>(row * kK), kK),
                              [](float value) { return value != 0.0F; });
    const auto q6_nonzero =
        std::ranges::count_if(std::span(q6_decoded).subspan(static_cast<std::size_t>(row * kK), kK),
                              [](float value) { return value != 0.0F; });
    ASSERT_EQ(q4_nonzero, 1);
    ASSERT_EQ(q6_nonzero, 1);
    EXPECT_EQ(q4_decoded[static_cast<std::size_t>(row * kK + (row * 37 + 3) % kK)], 0.5F);
    EXPECT_EQ(q6_decoded[static_cast<std::size_t>(row * kK + (row * 37 + 17) % kK)], 0.5F);
  }
  ASSERT_LE(q4_host.size(), q4_readable);
  ASSERT_LT(q4_host.size(), q4_readable);
  ASSERT_EQ(q4_host.size() % 256, 0U);
  ASSERT_LE(q6_host.size(), q6_readable);
  std::vector<std::byte> q4_padded(static_cast<std::size_t>(q4_readable), std::byte{0});
  std::vector<std::byte> q6_padded(static_cast<std::size_t>(q6_readable), std::byte{0});
  std::ranges::copy(q4_host, q4_padded.begin());
  std::ranges::copy(q6_host, q6_padded.begin());
  ASSERT_EQ(cudaMemcpy(q4_device.data, q4_padded.data(), q4_padded.size(), cudaMemcpyHostToDevice),
            cudaSuccess);
  ASSERT_EQ(cudaMemcpy(q6_device.data, q6_padded.data(), q6_padded.size(), cudaMemcpyHostToDevice),
            cudaSuccess);

  kn::Addresses addresses{};
  addresses.weight = [q4_ref, q6_ref, q4_address = q4_device.address(),
                      q6_address = q6_device.address()](const ir::WeightRef& ref,
                                                        fmt::PartRole part) -> std::uint64_t {
    if (part != fmt::PartRole::kData) return 0;
    if (ref == q4_ref) return q4_address;
    if (ref == q6_ref) return q6_address;
    return 0;
  };
  addresses.activations = activation.address();
  addresses.activation_bytes = kArenaBytes;
  addresses.activation_alignment = 256;

  const auto incumbent_input_offset =
      RootOffset(inventory->incumbent.plan, inventory->incumbent.plan.tensors[input.v]);
  const auto alternative_input_offset = RootOffset(planned->plan, planned->plan.tensors[input.v]);
  ASSERT_NE(incumbent_input_offset, UINT64_MAX);
  ASSERT_NE(alternative_input_offset, UINT64_MAX);
  const auto make_bindings = [&](std::uint64_t input_offset) {
    return std::array{kn::ComparisonBindingRange{.original_root = input,
                                                 .allocation_class = 1,
                                                 .allocation_bytes = *input_bytes,
                                                 .bytes = *input_bytes,
                                                 .alignment = 256,
                                                 .base = activation.address() + input_offset},
                      kn::ComparisonBindingRange{.original_root = q4,
                                                 .part = fmt::PartRole::kData,
                                                 .allocation_class = 2,
                                                 .allocation_bytes = q4_readable,
                                                 .bytes = q4_readable,
                                                 .alignment = 256,
                                                 .base = q4_device.address()},
                      kn::ComparisonBindingRange{.original_root = q6,
                                                 .part = fmt::PartRole::kData,
                                                 .allocation_class = 3,
                                                 .allocation_bytes = q6_readable,
                                                 .bytes = q6_readable,
                                                 .alignment = 256,
                                                 .base = q6_device.address()}};
  };
  auto incumbent_bindings = make_bindings(incumbent_input_offset);
  auto alternative_bindings = make_bindings(alternative_input_offset);
  kn::ComparisonRequest incumbent_request{.mode = kn::ComparisonRunMode::kFull,
                                          .bindings = incumbent_bindings,
                                          .expected_boundary = &inventory->source_boundary};
  kn::ComparisonRequest alternative_request{.mode = kn::ComparisonRunMode::kFull,
                                            .bindings = alternative_bindings,
                                            .expected_boundary = &inventory->source_boundary};
  ASSERT_TRUE(inventory->incumbent.final_origins.has_value());
  ASSERT_TRUE(planned->final_origins.has_value());
  kn::ComparisonRegionCapture incumbent_capture{
      .seed = &*region_boundary,
      .origins = &*inventory->incumbent.final_origins,
      .final_nodes = &inventory->incumbent.final_node_origins};
  kn::ComparisonRegionCapture alternative_capture{.seed = &*region_boundary,
                                                  .origins = &*planned->final_origins,
                                                  .final_nodes = &planned->final_node_origins};
  kn::LowerInput incumbent_input(inventory->incumbent.plan, io, addresses, std::nullopt,
                                 &incumbent_request, &incumbent_capture);
  kn::LowerInput alternative_input(planned->plan, io, addresses, std::nullopt, &alternative_request,
                                   &alternative_capture);
  auto launch =
      life_->backend->Launch({.base = pool.address(), .size = llmp::base::Bytes(kPoolBytes)},
                             {.base = persistent.address(), .bytes = persistent_bytes});
  ASSERT_TRUE(launch) << (launch ? "" : launch.error());

  // The descriptor endpoint must return without any device data pointers and
  // must report the actual selected MMVQ and Q8 quantizer dimensions.
  struct GgmlContext {
    ggml_context* value =
        ggml_init({.mem_size = 1U << 20U, .mem_buffer = nullptr, .no_alloc = true});
    ~GgmlContext() {
      if (value) ggml_free(value);
    }
  } ggml;
  ASSERT_NE(ggml.value, nullptr);
  alignas(256) std::array<std::byte, 2U << 20U> fake_storage{};
  auto* descriptor_weights = ggml_new_tensor_2d(
      ggml.value, GGML_TYPE_Q4_K, static_cast<std::int64_t>(kK), static_cast<std::int64_t>(kN));
  auto* descriptor_input = ggml_new_tensor_2d(
      ggml.value, GGML_TYPE_F32, static_cast<std::int64_t>(kK), static_cast<std::int64_t>(columns));
  ASSERT_NE(descriptor_weights, nullptr);
  ASSERT_NE(descriptor_input, nullptr);
  kg::MarkRowPaddingReadable(descriptor_weights);
  descriptor_weights->data = fake_storage.data();
  descriptor_input->data = fake_storage.data() + (1U << 20U);
  auto* descriptor_product = ggml_mul_mat(ggml.value, descriptor_weights, descriptor_input);
  ASSERT_NE(descriptor_product, nullptr);
  descriptor_product->data = fake_storage.data() + (1536U << 10U);
  auto* native_launch = static_cast<kg::GgmlStepLaunch*>(launch->get());
  auto mmvq_description = kg::DescribeMmvqDispatch(native_launch->context(), descriptor_product);
  ASSERT_TRUE(mmvq_description) << (mmvq_description ? "" : mmvq_description.error().detail);
  EXPECT_EQ(mmvq_description->device, native_launch->context().device());
  const auto local_dispatch = kg::ComparisonLocalMmvqDispatchDigest(*mmvq_description);
  const auto whole_dispatch = kg::ComparisonMmvqDispatchDigest(*mmvq_description);
  auto another_ordinal = *mmvq_description;
  ++another_ordinal.device;
  EXPECT_EQ(local_dispatch, kg::ComparisonLocalMmvqDispatchDigest(another_ordinal));
  EXPECT_NE(whole_dispatch, kg::ComparisonMmvqDispatchDigest(another_ordinal));
  EXPECT_EQ(mmvq_description->weight_type, GGML_TYPE_Q4_K);
  EXPECT_EQ(mmvq_description->columns, columns);
  EXPECT_EQ(mmvq_description->k, kK);
  EXPECT_EQ(mmvq_description->padded_k, 2560);
  EXPECT_EQ(mmvq_description->stride_col_dst, kN);
  EXPECT_GT(mmvq_description->rows_per_block, 0);
  EXPECT_GT(mmvq_description->static_shared_bytes, 0);
  EXPECT_EQ(mmvq_description->dynamic_shared_bytes, 0);
  EXPECT_FALSE(mmvq_description->immutable_weights);
  auto* unsupported_weights = ggml_new_tensor_2d(
      ggml.value, GGML_TYPE_Q8_0, static_cast<std::int64_t>(kK), static_cast<std::int64_t>(kN));
  ASSERT_NE(unsupported_weights, nullptr);
  kg::MarkRowPaddingReadable(unsupported_weights);
  unsupported_weights->data = fake_storage.data() + (256U << 10U);
  auto* unsupported_product = ggml_mul_mat(ggml.value, unsupported_weights, descriptor_input);
  ASSERT_NE(unsupported_product, nullptr);
  unsupported_product->data = fake_storage.data() + (1536U << 10U);
  EXPECT_FALSE(kg::DescribeMmvqDispatch(native_launch->context(), unsupported_product));
  auto* q8 = kg::QuantizeQ8(ggml.value, descriptor_input);
  ASSERT_NE(q8, nullptr);
  q8->data = fake_storage.data() + (1280U << 10U);
  q8->op_params[8 + 2] = 1;
  EXPECT_FALSE(kg::DescribeQuantizeQ8(q8));
  q8->op_params[8 + 2] = 0;
  auto quantize_description = kg::DescribeQuantizeQ8(q8);
  ASSERT_TRUE(quantize_description)
      << (quantize_description ? "" : quantize_description.error().detail);
  EXPECT_EQ(quantize_description->source_type, GGML_TYPE_Q8_0);
  EXPECT_EQ(quantize_description->padded_k, 2560);
  EXPECT_EQ(quantize_description->rows, columns);
  EXPECT_EQ(quantize_description->grid_x, 10);
  EXPECT_EQ(quantize_description->block_x, 256);
  EXPECT_EQ(quantize_description->output_bytes,
            kg::Q8Bytes(kK, static_cast<std::int64_t>(columns)));
  auto* unsupported_input =
      ggml_new_tensor_2d(ggml.value, GGML_TYPE_F32, static_cast<std::int64_t>(kK), 9);
  ASSERT_NE(unsupported_input, nullptr);
  unsupported_input->data = fake_storage.data() + (768U << 10U);
  auto* unsupported_q8 = kg::QuantizeQ8(ggml.value, unsupported_input);
  ASSERT_NE(unsupported_q8, nullptr);
  unsupported_q8->data = fake_storage.data() + (1408U << 10U);
  EXPECT_FALSE(kg::DescribeQuantizeQ8(unsupported_q8));
  auto* prepared = kg::MmvqPrepared(ggml.value, descriptor_weights, q8, descriptor_input);
  ASSERT_NE(prepared, nullptr);
  prepared->data = fake_storage.data() + (1792U << 10U);
  auto prepared_description = kg::DescribeMmvqDispatch(native_launch->context(), prepared, true);
  ASSERT_TRUE(prepared_description)
      << (prepared_description ? "" : prepared_description.error().detail);
  EXPECT_TRUE(prepared_description->immutable_weights);
  EXPECT_EQ(prepared_description->device, native_launch->context().device());
  EXPECT_EQ(prepared_description->columns, columns);
  EXPECT_EQ(prepared_description->padded_k, 2560);
  EXPECT_GT(prepared_description->rows_per_block, 0);
  EXPECT_EQ(prepared_description->immutable_prefetch, prepared_description->pdl_enabled);
  EXPECT_EQ(prepared_description->streaming_prefetch, mmvq_description->streaming_prefetch);

  auto* wrong_producer_input = ggml_new_tensor_2d(
      ggml.value, GGML_TYPE_F32, static_cast<std::int64_t>(kK), static_cast<std::int64_t>(columns));
  ASSERT_NE(wrong_producer_input, nullptr);
  wrong_producer_input->data = fake_storage.data() + (512U << 10U);
  auto* wrong_producer_q8 = kg::QuantizeQ8(ggml.value, wrong_producer_input);
  ASSERT_NE(wrong_producer_q8, nullptr);
  wrong_producer_q8->data = fake_storage.data() + (1280U << 10U);
  auto* wrong_producer =
      kg::MmvqPrepared(ggml.value, descriptor_weights, wrong_producer_q8, descriptor_input);
  ASSERT_NE(wrong_producer, nullptr);
  wrong_producer->data = fake_storage.data() + (1792U << 10U);
  EXPECT_FALSE(kg::DescribeMmvqDispatch(native_launch->context(), wrong_producer, true));

  auto* raw_q8 = ggml_new_tensor_2d(
      ggml.value, GGML_TYPE_I8,
      static_cast<std::int64_t>(kg::Q8Bytes(kK, static_cast<std::int64_t>(columns))), 1);
  ASSERT_NE(raw_q8, nullptr);
  raw_q8->data = fake_storage.data() + (1024U << 10U);
  auto* raw_prepared = kg::MmvqPrepared(ggml.value, descriptor_weights, raw_q8, descriptor_input);
  ASSERT_NE(raw_prepared, nullptr);
  raw_prepared->data = fake_storage.data() + (1792U << 10U);
  EXPECT_FALSE(kg::DescribeMmvqDispatch(native_launch->context(), raw_prepared, true));

  auto incumbent_program = life_->backend->Lower(incumbent_input);
  ASSERT_TRUE(incumbent_program) << (incumbent_program ? "" : incumbent_program.error());
  auto alternative_program = life_->backend->Lower(alternative_input);
  ASSERT_TRUE(alternative_program) << (alternative_program ? "" : alternative_program.error());

  std::array<kn::ComparisonRegionRecipeSeal, 2> descriptor_region_seals{};
  std::array<std::optional<kn::ComparisonRegionProjection>, 2> region_projections;
  std::array<kn::StepProgram*, 2> region_programs{incumbent_program->get(),
                                                  alternative_program->get()};
  for (std::size_t arm = 0; arm < region_programs.size(); ++arm) {
    const auto partial = region_programs[arm]->ProjectComparisonRegionDescriptor(*partial_boundary);
    ASSERT_EQ(partial.status, kn::ComparisonRegionStatus::kExpand) << partial.reason;
    ASSERT_EQ(partial.required_members.size(), 1U);
    EXPECT_EQ(partial.required_members.front(), products[1]);
    const auto descriptor =
        region_programs[arm]->ProjectComparisonRegionDescriptor(*region_boundary);
    ASSERT_EQ(descriptor.status, kn::ComparisonRegionStatus::kClosed) << descriptor.reason;
    ASSERT_EQ(descriptor.seal.state, kn::ComparisonProofState::kDescriptorVerified);
    EXPECT_TRUE(descriptor.required_members.empty());
    descriptor_region_seals[arm] = descriptor.seal;
  }

  auto incumbent_seal = (*incumbent_program)->comparison_recipe();
  auto alternative_seal = (*alternative_program)->comparison_recipe();
  const auto input_address = [&](const kn::StepProgram& program) {
    const auto slots = program.inputs();
    const auto slot = std::ranges::find(slots, input, &kn::InputSlot::tensor);
    return slot == slots.end() ? std::uint64_t{0} : slot->device_address;
  };
  const auto output_slot = [](const kn::StepProgram& program,
                              ir::TensorId tensor) -> const kn::OutputSlot* {
    const auto slots = program.outputs();
    const auto slot = std::ranges::find(slots, tensor, &kn::OutputSlot::tensor);
    return slot == slots.end() ? nullptr : &*slot;
  };
  const std::uint64_t output_bytes = kN * columns * sizeof(float);
  const auto* incumbent_out4 = output_slot(**incumbent_program, out4);
  const auto* incumbent_out6 = output_slot(**incumbent_program, out6);
  const auto* alternative_out4 = output_slot(**alternative_program, out4);
  const auto* alternative_out6 = output_slot(**alternative_program, out6);
  ASSERT_NE(incumbent_out4, nullptr);
  ASSERT_NE(incumbent_out6, nullptr);
  ASSERT_NE(alternative_out4, nullptr);
  ASSERT_NE(alternative_out6, nullptr);
  EXPECT_EQ(incumbent_out4->bytes, output_bytes);
  EXPECT_EQ(incumbent_out6->bytes, output_bytes);
  EXPECT_EQ(alternative_out4->bytes, output_bytes);
  EXPECT_EQ(alternative_out6->bytes, output_bytes);
  ASSERT_TRUE(incumbent_seal && alternative_seal);
  const auto schedule = [&](const pl::StepPlan& plan) {
    std::string text;
    for (const auto& step : plan.steps) {
      if (!text.empty()) text.append(", ");
      text.append(recipe.registry->at(step.implementation).name);
      text.push_back('(');
      text.append(std::to_string(step.nodes.size()));
      text.push_back(')');
    }
    return text;
  };
  ASSERT_EQ(incumbent_seal->state, kn::ComparisonProofState::kUnproved)
      << "missing=" << static_cast<int>(incumbent_seal->missing)
      << " schedule=" << schedule(inventory->incumbent.plan);
  ASSERT_EQ(alternative_seal->state, kn::ComparisonProofState::kUnproved)
      << "missing=" << static_cast<int>(alternative_seal->missing)
      << " schedule=" << schedule(planned->plan);
  EXPECT_EQ(incumbent_seal->missing, kn::ComparisonMissing::kOperationSchema);
  EXPECT_EQ(alternative_seal->missing, kn::ComparisonMissing::kOperationSchema);
  EXPECT_EQ(
      schedule(inventory->incumbent.plan),
      "llmp.q8_1(1), llmp.mmvq.prepared(1), llmp.mmvq.prepared(1), ggml.concat(1), ggml.unary(1)");
  EXPECT_EQ(
      schedule(planned->plan),
      "ggml.mul_mat.mmvq(1), llmp.q8_1(1), llmp.mmvq.prepared(1), ggml.concat(1), ggml.unary(1)");
  EXPECT_EQ(incumbent_seal->subject, alternative_seal->subject);
  EXPECT_EQ(incumbent_seal->boundary, alternative_seal->boundary);

  // The physical boundary must fail closed for malformed weight-part identity,
  // truncated readable tails, and a real address alias between immutable parts.
  const auto rejects_lower = [&](const pl::StepPlan& plan,
                                 std::span<const kn::ComparisonBindingRange> bindings,
                                 const kn::Addresses& candidate_addresses) {
    kn::ComparisonRequest request{.mode = kn::ComparisonRunMode::kFull,
                                  .bindings = bindings,
                                  .expected_boundary = &inventory->source_boundary};
    kn::LowerInput candidate(plan, io, candidate_addresses, std::nullopt, &request);
    return !life_->backend->Lower(candidate);
  };
  auto wrong_part_bindings = incumbent_bindings;
  wrong_part_bindings[1].part = fmt::PartRole::kBlockScales;
  EXPECT_TRUE(rejects_lower(inventory->incumbent.plan, wrong_part_bindings, addresses));
  auto short_tail_bindings = incumbent_bindings;
  ASSERT_GT(short_tail_bindings[1].bytes, 0U);
  --short_tail_bindings[1].bytes;
  EXPECT_TRUE(rejects_lower(inventory->incumbent.plan, short_tail_bindings, addresses));
  kn::Addresses aliased_addresses = addresses;
  aliased_addresses.weight = [q4_ref, q6_ref, q4_address = q4_device.address()](
                                 const ir::WeightRef& ref, fmt::PartRole part) -> std::uint64_t {
    if (part != fmt::PartRole::kData) return 0;
    if (ref == q4_ref || ref == q6_ref) return q4_address;
    return 0;
  };
  auto alias_bindings = incumbent_bindings;
  alias_bindings[2].base = q4_device.address();
  alias_bindings[2].allocation_class = alias_bindings[1].allocation_class;
  EXPECT_TRUE(rejects_lower(inventory->incumbent.plan, alias_bindings, aliased_addresses));
  const auto rejects_scratch_alias = [&](std::uint64_t scratch_base) {
    auto bad_backend_result =
        kg::GgmlBackend::Create(0, life_->node.execution(), life_->node.stream(0), nullptr);
    if (!bad_backend_result) {
      ADD_FAILURE() << bad_backend_result.error();
      return false;
    }
    auto bad_backend = std::move(*bad_backend_result);
    const auto charge = [paid = life_->paid](std::uint64_t bytes) { return paid->Charge(bytes); };
    const auto release = [paid = life_->paid](std::uint64_t bytes) { paid->Release(bytes); };
    if (!bad_backend->FundConstruction(charge, release)) {
      ADD_FAILURE() << "could not fund isolated alias-check backend construction";
      return false;
    }
    if (!bad_backend->Declare({}, life_->resources.library_bytes(), charge, release)) {
      ADD_FAILURE() << "could not declare isolated alias-check backend";
      return false;
    }
    if (!bad_backend->LendLibrary(&life_->resources.library())) {
      ADD_FAILURE() << "could not lend the fixture library to alias-check backend";
      return false;
    }
    auto bad_launch_result =
        bad_backend->Launch({.base = scratch_base, .size = llmp::base::Bytes(kPoolBytes)},
                            {.base = persistent.address(), .bytes = persistent_bytes});
    if (!bad_launch_result) {
      ADD_FAILURE() << bad_launch_result.error();
      return false;
    }
    auto bad_launch = std::move(*bad_launch_result);
    auto bad_program_result = bad_backend->Lower(alternative_input);
    if (!bad_program_result) {
      ADD_FAILURE() << bad_program_result.error();
      return false;
    }
    auto bad_program = std::move(*bad_program_result);
    const auto whole = bad_program->comparison_recipe();
    if (!whole || whole->state != kn::ComparisonProofState::kUnproved ||
        whole->missing != kn::ComparisonMissing::kOperationSchema) {
      ADD_FAILURE() << "Relu suffix did not make the whole recipe unproved by its missing schema";
      return false;
    }
    const auto local = bad_program->ProjectComparisonRegionDescriptor(*region_boundary);
    if (local.status != kn::ComparisonRegionStatus::kUnproved ||
        local.reason != "Q8 private pool aliases a local activation or external readable range") {
      ADD_FAILURE() << "local descriptor projection did not reject the aliased pool: "
                    << local.reason;
      return false;
    }
    auto bound = bad_program->Bind(*bad_launch);
    if (bound) {
      ADD_FAILURE() << "shared-Q8 comparison accepted an aliased scratch pool";
      return false;
    }
    const auto seal = bad_program->comparison_recipe();
    return seal && seal->state == kn::ComparisonProofState::kUnproved &&
           seal->missing == kn::ComparisonMissing::kBindings;
  };
  EXPECT_TRUE(rejects_scratch_alias(activation.address()));
  EXPECT_TRUE(rejects_scratch_alias(q4_device.address() + q4_host.size()));
  ASSERT_TRUE((*incumbent_program)->Bind(**launch));
  ASSERT_TRUE((*alternative_program)->Bind(**launch));
  for (std::size_t arm = 0; arm < region_programs.size(); ++arm) {
    auto projected = region_programs[arm]->ProjectComparisonRegion(*region_boundary);
    ASSERT_EQ(projected.status, kn::ComparisonRegionStatus::kClosed) << projected.reason;
    ASSERT_TRUE(projected.projection);
    ASSERT_EQ(projected.projection->seal().state, kn::ComparisonProofState::kBoundVerified);
    const auto& before = descriptor_region_seals[arm];
    const auto& after = projected.projection->seal();
    EXPECT_EQ(after.subject, before.subject);
    EXPECT_EQ(after.boundary, before.boundary);
    EXPECT_EQ(after.planned_region, before.planned_region);
    EXPECT_EQ(after.physical_recipe, before.physical_recipe);
    EXPECT_EQ(after.policy, before.policy);
    EXPECT_EQ(after.registry, before.registry);
    EXPECT_EQ(after.backend_recipe, before.backend_recipe);
    EXPECT_EQ(after.backend_build, before.backend_build);
    EXPECT_EQ(after.backend_selection, before.backend_selection);
    EXPECT_EQ(after.backend_device, before.backend_device);
    region_projections[arm].emplace(std::move(*projected.projection));
  }
  incumbent_seal = (*incumbent_program)->comparison_recipe();
  alternative_seal = (*alternative_program)->comparison_recipe();
  ASSERT_EQ(incumbent_seal->state, kn::ComparisonProofState::kUnproved);
  ASSERT_EQ(alternative_seal->state, kn::ComparisonProofState::kUnproved);

  auto reference = ref::Executor::Create(*graph);
  ASSERT_TRUE(reference) << (reference ? "" : reference.error().detail);
  auto native = life_->node.execution().Submission(life_->node.stream(0));
  ASSERT_TRUE(native) << native.error().detail;
  auto incumbent_recording =
      (*incumbent_program)
          ->Record(**launch, *native, [&]() -> std::expected<void, kn::StepFailure> {
            return (*incumbent_program)
                ->RunComparisonRegion(*region_projections[0], **launch, *native);
          });
  ASSERT_TRUE(incumbent_recording)
      << (incumbent_recording ? "" : incumbent_recording.error().detail);
  auto alternative_recording =
      (*alternative_program)
          ->Record(**launch, *native, [&]() -> std::expected<void, kn::StepFailure> {
            return (*alternative_program)
                ->RunComparisonRegion(*region_projections[1], **launch, *native);
          });
  ASSERT_TRUE(alternative_recording)
      << (alternative_recording ? "" : alternative_recording.error().detail);
  EXPECT_EQ((*incumbent_recording)->nodes(), 3U);
  EXPECT_EQ((*alternative_recording)->nodes(), 4U);
  const auto run_case = [&](std::uint32_t phase) {
    std::vector<float> x(kK * columns);
    for (std::uint64_t i = 0; i < kK * columns; ++i) {
      const bool positive = (((i % kK) + (i / kK) * 16) % 32) < 16;
      const float value = positive ? 127.0F : -127.0F;
      x[static_cast<std::size_t>(i)] = phase == 0 ? value : -value;
    }
    ASSERT_TRUE(reference->BindValues(input, x));
    ASSERT_TRUE(reference->Bind(
        q4, std::array{ref::PartView{.role = fmt::PartRole::kData, .bytes = q4_host}}));
    ASSERT_TRUE(reference->Bind(
        q6, std::array{ref::PartView{.role = fmt::PartRole::kData, .bytes = q6_host}}));
    ASSERT_TRUE(reference->Run());
    const auto expected4 = reference->Values(out4);
    const auto expected6 = reference->Values(out6);
    ASSERT_TRUE(expected4 && expected6);
    EXPECT_TRUE(std::ranges::any_of(*expected4, [](float v) { return v != 0.0F; }));
    EXPECT_TRUE(std::ranges::any_of(*expected6, [](float v) { return v != 0.0F; }));
    EXPECT_FALSE(std::ranges::equal(*expected4, *expected6));

    std::vector<float> poison(kN * columns, 12345.0F);
    for (const bool alternative : {false, true}) {
      const auto& program = alternative ? **alternative_program : **incumbent_program;
      const auto in_address = input_address(program);
      ASSERT_NE(in_address, 0U);
      const auto* slot4 = alternative ? alternative_out4 : incumbent_out4;
      const auto* slot6 = alternative ? alternative_out6 : incumbent_out6;
      const auto out4_address = slot4->device_address;
      const auto out6_address = slot6->device_address;
      ASSERT_NE(out4_address, 0U);
      ASSERT_NE(out6_address, 0U);
      ASSERT_EQ(cudaMemcpy(reinterpret_cast<void*>(out4_address), poison.data(),
                           poison.size() * sizeof(float), cudaMemcpyHostToDevice),
                cudaSuccess);
      ASSERT_EQ(cudaMemcpy(reinterpret_cast<void*>(out6_address), poison.data(),
                           poison.size() * sizeof(float), cudaMemcpyHostToDevice),
                cudaSuccess);
      // The planner may reuse the input root for a result after its last
      // consumer. Poisoning outputs must therefore precede input staging.
      ASSERT_EQ(cudaMemcpy(reinterpret_cast<void*>(in_address), x.data(), x.size() * sizeof(float),
                           cudaMemcpyHostToDevice),
                cudaSuccess);
      std::vector<float> staged(x.size());
      ASSERT_EQ(cudaMemcpy(staged.data(), reinterpret_cast<void*>(in_address),
                           staged.size() * sizeof(float), cudaMemcpyDeviceToHost),
                cudaSuccess);
      ASSERT_EQ(staged, x) << "input address=" << in_address << " out4=" << out4_address
                           << " out6=" << out6_address;
      const auto& recording = alternative ? *alternative_recording : *incumbent_recording;
      ASSERT_TRUE((*launch)->Launch(*native, *recording));
      ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
      std::vector<float> got4(kN * columns), got6(kN * columns);
      ASSERT_EQ(cudaMemcpy(got4.data(), reinterpret_cast<void*>(out4_address),
                           got4.size() * sizeof(float), cudaMemcpyDeviceToHost),
                cudaSuccess);
      ASSERT_EQ(cudaMemcpy(got6.data(), reinterpret_cast<void*>(out6_address),
                           got6.size() * sizeof(float), cudaMemcpyDeviceToHost),
                cudaSuccess);
      EXPECT_TRUE(std::ranges::equal(got4, *expected4))
          << "arm=" << alternative << " phase=" << phase << " addresses=" << in_address << ','
          << out4_address << ',' << out6_address << " q4[0]=" << got4.front() << '/'
          << expected4->front();
      EXPECT_TRUE(std::ranges::equal(got6, *expected6))
          << "arm=" << alternative << " phase=" << phase << " addresses=" << in_address << ','
          << out4_address << ',' << out6_address << " q6[0]=" << got6.front() << '/'
          << expected6->front();
    }
  };
  run_case(0);
  run_case(1);  // changed-input poisoned-output replay through both owned programs
}

INSTANTIATE_TEST_SUITE_P(SharedQ8Rows, GgmlComparisonPreparationGpu, testing::Values(1, 2, 8));

}  // namespace
