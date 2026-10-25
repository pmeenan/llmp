// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#ifndef LLMP_KERNELS_GGML_COMPARISON_H_
#define LLMP_KERNELS_GGML_COMPARISON_H_

#include <array>
#include <expected>
#include <optional>
#include <span>
#include <string_view>

#include "execution/registry.h"
#include "kernels/common/comparison.h"
#include "kernels/common/lower.h"
#include "kernels/ggml/lower.h"
#include "kernels/ggml/ops_ext.h"
#include "kernels/ggml/validate.h"

namespace llmp::kernels::ggml {

class LaunchContext;

enum class ComparisonBf16Product : std::uint8_t { kVector, kGemmEx };

struct ComparisonRemainderCallFacts {
  std::uint64_t scratch_bytes = 0;
  std::uint64_t pool_base = 0;
  std::uint64_t pool_bytes = 0;
  std::uint32_t pool_lanes = 0;
  bool pool_faulted = false;
  int attention_columns_per_block = 0;
  int attention_parallel_blocks = 0;
  std::uint64_t attention_scratch_bytes = 0;
  bool has_mul_mat_selection = false;
  bool mul_mat_vector = false;
  std::optional<MulMatPath> float_selection;
  // Complete native programs may use the actual pinned quantized selector
  // with their sealed descriptors/device/build, without local access authority.
  std::optional<QuantMulMatPath> quantized_selection;
  std::optional<CublasMulMat> cublas_plan;
  std::optional<ComparisonBf16Product> bf16_product;
  base::Sha256Digest cublas_configuration{};
  std::uint64_t library_workspace_base = 0;
  std::uint64_t library_workspace_bytes = 0;
  bool has_attention_plan = false;
  bool attention_mask_prepass = false;
};
using ComparisonRemainderCallFactsFn = bool (*)(const LaunchContext&, const PlanStep&,
                                                ComparisonRemainderCallFacts&);

// The two registered deterministic compressor cell recipes, joined to normal
// staging metadata and this actual indexed destination. This is recipe/bounds
// validation, not authority for arbitrary host callbacks or device indices.
std::optional<std::string_view> ComparisonHostIndexRecipe(const components::StepInput&,
                                                          const InputSlot&,
                                                          const planner::StepShape&,
                                                          std::uint64_t destination_capacity);

// Backend-internal whole-program bounds check for native TopK -> indexed copy.
// This is not a seal or authority for arbitrary device indices/local regions.
bool CheckComparisonTopKIndices(const LowerInput&, const LoweredStep&, const execution::Registry&,
                                const ggml_tensor* store);

// Exact prepared NVFP4 aggregate: all four original parts remain distinct.
bool CheckComparisonExpertSlab(const LoweredLeaf&, std::span<const ComparisonBindingRange>);
std::optional<ComparisonExpertSlab> DescribeComparisonExpertSlab(const LowerInput&,
                                                                 const LoweredStep&,
                                                                 const LoweredLeaf&);

// Native whole only: authenticate a source Input's backend-selected physical
// slice without changing its logical source boundary or granting local authority.
std::optional<ComparisonStagedInput> DescribeComparisonStagedInput(const LowerInput&,
                                                                   const LoweredStep&,
                                                                   const LoweredLeaf&,
                                                                   std::size_t binding);
bool CheckComparisonStagedInput(const LoweredStep&, const LoweredLeaf&,
                                const ComparisonBindingRange& actual);

// Actual direct F32 SGEMM domain; the backend callback supplies the selected
// plan and currently lent library identity. It does not grant access closure.
bool ComparisonSgemmCall(const PlanStep&);
bool CheckComparisonSgemm(const PlanStep&, const ComparisonRemainderCallFacts&);
// Whole programs only: the actual checked cuBLAS plan, including conversion,
// batched strides and pointer-array scratch. No local cut authority or Lt path.
bool CheckComparisonWholeCublas(const PlanStep&, const ComparisonRemainderCallFacts&);
bool ComparisonLibraryWorkspaceDisjoint(const LoweredStep&, const ComparisonRemainderCallFacts&);
// Exact native MMF selected by the owned CUDA context, with its zero-scratch
// launcher. This supplies no local access/cut authority.
bool CheckComparisonMmf(const PlanStep&, const ComparisonRemainderCallFacts&);
// Complete programs also admit checked BF16 vector/GemmEx calls. Potential
// pinned-Lt shapes remain outside this protocol until actual Lt facts exist.
bool ComparisonBf16Call(const PlanStep&);
bool CheckComparisonBf16(const PlanStep&, const ComparisonRemainderCallFacts&);
std::optional<base::Sha256Digest> ComparisonProductScheduleIdentity(const LoweredStep&,
                                                                    const LaunchContext&,
                                                                    ComparisonRemainderCallFactsFn);

template <class Hash>
void HashComparisonCublasFacts(Hash& hash, const ComparisonRemainderCallFacts& facts) {
  hash.Text("ggml.cuda.checked-cublas-plan.v2");
  const auto& plan = *facts.cublas_plan;
  hash.U32(static_cast<std::uint32_t>(plan.compute));
  hash.U8(plan.f32_output ? 1 : 0);
  hash.U32(static_cast<std::uint32_t>(plan.weights));
  hash.U32(static_cast<std::uint32_t>(plan.input));
  hash.U32(static_cast<std::uint32_t>(plan.gemm));
  hash.U64(static_cast<std::uint64_t>(plan.s01));
  hash.U64(static_cast<std::uint64_t>(plan.s02));
  hash.U64(static_cast<std::uint64_t>(plan.s03));
  hash.U64(static_cast<std::uint64_t>(plan.s11));
  hash.U64(static_cast<std::uint64_t>(plan.s12));
  hash.U64(static_cast<std::uint64_t>(plan.s13));
  hash.U64(plan.alignment);
  hash.U64(plan.scratch);
  for (const auto byte : facts.cublas_configuration) hash.U8(byte);
  hash.U64(facts.library_workspace_bytes);
}

// Exact split F16 product edges and the actual SwiGLU descriptor. Product
// dispatch/access checking remains independent; split output may reuse dead input.
bool CheckComparisonF16SwiGlu(const ggml_tensor* gate, const ggml_tensor* up,
                              const ggml_tensor* glu);

bool CheckComparisonMatMulAdd(const ggml_tensor* product, const ggml_tensor* add);
bool CheckComparisonNormAdd(const ggml_tensor* norm, const ggml_tensor* mul,
                            const ggml_tensor* add);

struct ComparisonMetrics {
  std::uint64_t pool_bytes = 0;
  std::uint64_t private_pool_bytes = 0;
  std::uint64_t lanes = 0;
  std::uint64_t lane_scratch_bytes = 0;
};

using ComparisonMulMatSelector = std::expected<MulMatPath, KernelFailure> (*)(const LaunchContext&,
                                                                              const ggml_tensor*);

// Optional backend-owned access to the exact quantized dispatch descriptions
// and scratch planners. The all-profile comparison library can authenticate
// these records without linking CUDA-only implementation symbols.
struct ComparisonQ8Dispatch {
  decltype(&DescribeQ8MainPool) describe_main_pool = nullptr;
  decltype(&DescribeQuantizeQ8) describe_quantize_q8 = nullptr;
  decltype(&DescribeMmvqDispatch) describe_mmvq = nullptr;
  decltype(&PlanMulMatVecQ) plan_mmvq = nullptr;
  decltype(&PlanMmvqPrepared) plan_prepared_mmvq = nullptr;

  bool complete() const {
    return describe_quantize_q8 && describe_mmvq && plan_mmvq && plan_prepared_mmvq;
  }
  bool has_main_pool() const { return describe_main_pool != nullptr; }
};

// Canonical typed field serializers shared by whole-program and local-region
// recipe hashing. Whole-program identity includes the CUDA ordinal; local
// identity binds the producer-owned device facts separately. Hash must expose
// Text/U32/U64/U8 in little-endian order.
template <class Hash>
void HashComparisonMmvqDispatchFields(Hash& hash, const MmvqDispatchDescriptor& d) {
  hash.U32(static_cast<std::uint32_t>(d.compute_capability));
  hash.U32(static_cast<std::uint32_t>(d.parameter_table));
  hash.U32(static_cast<std::uint32_t>(d.warp_size));
  hash.U32(static_cast<std::uint32_t>(d.weight_type));
  hash.U32(static_cast<std::uint32_t>(d.columns));
  hash.U32(static_cast<std::uint32_t>(d.k));
  hash.U32(static_cast<std::uint32_t>(d.padded_k));
  hash.U32(static_cast<std::uint32_t>(d.row_low));
  hash.U32(static_cast<std::uint32_t>(d.row_high));
  hash.U32(static_cast<std::uint32_t>(d.stride_col_dst));
  hash.U32(static_cast<std::uint32_t>(d.rows_per_block));
  hash.U32(static_cast<std::uint32_t>(d.nwarps));
  hash.U32(static_cast<std::uint32_t>(d.grid_x));
  hash.U32(static_cast<std::uint32_t>(d.grid_y));
  hash.U32(static_cast<std::uint32_t>(d.grid_z));
  hash.U32(static_cast<std::uint32_t>(d.block_x));
  hash.U32(static_cast<std::uint32_t>(d.block_y));
  hash.U32(static_cast<std::uint32_t>(d.block_z));
  hash.U32(static_cast<std::uint32_t>(d.dynamic_shared_bytes));
  hash.U32(static_cast<std::uint32_t>(d.static_shared_bytes));
  hash.U8(d.small_k ? 1 : 0);
  hash.U8(d.halve_iters ? 1 : 0);
  hash.U8(d.immutable_weights ? 1 : 0);
  hash.U8(d.pdl_enabled ? 1 : 0);
  hash.U8(d.immutable_prefetch ? 1 : 0);
  hash.U8(d.streaming_prefetch ? 1 : 0);
  hash.U8(d.fused ? 1 : 0);
}

template <class Hash>
void HashComparisonMmvqDispatch(Hash& hash, const MmvqDispatchDescriptor& d) {
  hash.Text("ggml.cuda.mmvq.dispatch.v1");
  hash.U32(static_cast<std::uint32_t>(d.device));
  HashComparisonMmvqDispatchFields(hash, d);
}

// Local physical recipes already bind the actual device through the
// producer-owned backend_device identity. Keep all typed dispatch fields but
// do not persist the process-local CUDA ordinal.
template <class Hash>
void HashComparisonLocalMmvqDispatch(Hash& hash, const MmvqDispatchDescriptor& d) {
  hash.Text("ggml.cuda.mmvq.local-dispatch.v1");
  HashComparisonMmvqDispatchFields(hash, d);
}

template <class Hash>
void HashComparisonQ8QuantizeDispatch(Hash& hash, const Q8QuantizeDispatchDescriptor& d) {
  hash.Text("ggml.cuda.q8_1.quantize.dispatch.v1");
  hash.U32(static_cast<std::uint32_t>(d.source_type));
  hash.U64(static_cast<std::uint64_t>(d.k));
  hash.U64(static_cast<std::uint64_t>(d.padded_k));
  hash.U64(static_cast<std::uint64_t>(d.stride_row));
  hash.U64(static_cast<std::uint64_t>(d.stride_channel));
  hash.U64(static_cast<std::uint64_t>(d.stride_sample));
  hash.U64(static_cast<std::uint64_t>(d.rows));
  hash.U64(static_cast<std::uint64_t>(d.channels));
  hash.U64(static_cast<std::uint64_t>(d.samples));
  hash.U32(static_cast<std::uint32_t>(d.grid_x));
  hash.U32(static_cast<std::uint32_t>(d.grid_y));
  hash.U32(static_cast<std::uint32_t>(d.grid_z));
  hash.U32(static_cast<std::uint32_t>(d.block_x));
  hash.U32(static_cast<std::uint32_t>(d.block_y));
  hash.U32(static_cast<std::uint32_t>(d.block_z));
  hash.U32(static_cast<std::uint32_t>(d.dynamic_shared_bytes));
  hash.U64(d.output_bytes);
}

enum class ComparisonQ8CallKind : std::uint8_t { kQuantize, kMmvq, kMmvqPrepared };

// Synchronous description of one actual Q8 physical call. These records
// borrow only the current lowered graph and LaunchContext; they never escape
// the analyzer or become caller authority.
struct ComparisonQ8CallRecipe {
  ComparisonQ8CallKind kind = ComparisonQ8CallKind::kQuantize;
  const PlanStep* step = nullptr;
  std::size_t step_ordinal = 0;
  const ggml_tensor* node = nullptr;
  const ggml_tensor* input = nullptr;
  const ggml_tensor* weight = nullptr;
  const ggml_tensor* packed = nullptr;
  MmvqDispatchDescriptor mmvq{};
  Q8QuantizeDispatchDescriptor quantizer{};
  std::uint64_t scratch_bytes = 0;
};

struct ComparisonQ8StepRef {
  const PlanStep* step = nullptr;
  std::size_t ordinal = 0;
};

struct ComparisonQ8SpanRecipe {
  std::array<ComparisonQ8CallRecipe, 3> calls{};
  std::size_t call_count = 0;
  std::array<const ggml_tensor*, 2> products{};
  std::array<const ggml_tensor*, 2> weights{};
  std::size_t product_count = 0;
  const ggml_tensor* input = nullptr;
  const ggml_tensor* prepared = nullptr;
};

// Validates only the supported actual shared-Q8 call topology and obtains all
// dispatch/scratch facts from the backend-owned callbacks. Dimensions are
// derived from those descriptors; fixture shapes are not proof predicates.
std::expected<ComparisonQ8SpanRecipe, std::string_view> DescribeComparisonQ8Span(
    std::span<const ComparisonQ8StepRef> steps, const LaunchContext& launch,
    const ComparisonQ8Dispatch& dispatch);

// Shared, allocation-free serializers keep whole-program and local-region
// dispatch identities byte-for-byte consistent.
base::Sha256Digest ComparisonMmvqDispatchDigest(const MmvqDispatchDescriptor& descriptor);
base::Sha256Digest ComparisonLocalMmvqDispatchDigest(const MmvqDispatchDescriptor& descriptor);
base::Sha256Digest ComparisonQ8QuantizeDispatchDigest(
    const Q8QuantizeDispatchDescriptor& descriptor);

// Exact readable backing extent for the MMVQ weight operand: logical dense
// storage plus the one admitted final quantized-block tail. `padded_k` is the
// activation's Q8 width and must never be multiplied once per weight row.
std::optional<std::uint64_t> ComparisonMmvqWeightReadableBytes(
    const ggml_tensor& weight, const MmvqDispatchDescriptor& dispatch);

// Checks the real current main-pool interval against every actual activation
// root and authenticated external range. Used independently of whole-seal
// status by local proof and by whole-program Bind.
bool ComparisonPoolDisjoint(const LoweredStep& lowered, const Q8MainPoolDescriptor& pool,
                            std::uint64_t bytes);

// Supplied only by the concrete backend; uses its ordinary registered native
// wrapper checks. The CPU-linkable seal builder never links native executors.
using ComparisonNativeCallChecker = bool (*)(const execution::Registry&, const PlanStep&);

// Resident selected calls only. Joins each owned marker's complete operands
// and outputs to the retained planned IR; hashes immutable source identity and
// call resources. Ordinary SelectedCall::Bind remains the final launch check.
std::optional<base::Sha256Digest> ComparisonSelectedScheduleIdentity(const LoweredStep&,
                                                                     const execution::Registry&,
                                                                     const SelectedSource&,
                                                                     std::uint64_t pool_bytes);

// The result is allocation-free apart from the fixed SHA implementation.
// Unknown operation schemas and unsupported physical helper records produce
// an owned unproved seal with a typed reason.
ComparisonRecipeSeal DescribeComparison(
    const LowerInput& input, const LoweredStep& lowered, const execution::Registry& registry,
    const execution::BackendDeclaration* declaration, ComparisonMetrics metrics,
    std::optional<components::ComparisonBoundaryDigest> observed_boundary = std::nullopt,
    const LaunchContext* launch_context = nullptr,
    ComparisonMulMatSelector mul_mat_selector = nullptr,
    const ComparisonQ8Dispatch* q8_dispatch = nullptr,
    ComparisonRemainderCallFactsFn describe_call_facts = nullptr,
    ComparisonNativeCallChecker check_native_call = nullptr,
    const SelectedSource* selected_source = nullptr);

// Typed parameter/access schema shared with local region projection. This
// checks actual GGML parameters and reserved tails without making a whole-graph
// proof claim.
bool HasTypedDescriptorSchema(const ggml_tensor& tensor);

// Builds the one shared logical boundary from the actual LowerInput and the
// concrete request ranges. The caller pays conversion/control storage and
// supplies the backend's retained comparison funder.
std::expected<components::ComparisonBoundaryDigest, std::string> ObserveComparisonBoundary(
    const LowerInput& input, std::span<const ComparisonBindingRange> bindings,
    std::shared_ptr<const ir::StorageFunder> funder);

// Checks actual external declarations against original roots and checked
// intervals. Address values are consulted only for ownership/overlap checks.
std::expected<void, std::string> ValidateComparisonBindings(
    const LowerInput& input, std::span<const ComparisonBindingRange> bindings);

}  // namespace llmp::kernels::ggml

#endif  // LLMP_KERNELS_GGML_COMPARISON_H_
