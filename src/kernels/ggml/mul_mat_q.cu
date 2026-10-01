// SPDX-FileCopyrightText: 2023-2026 The ggml authors
// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: MIT AND Apache-2.0

// GGML's quantized matrix products (ops_ext.h): the vector (MMVQ) and tile
// (MMQ) kernel families under jitLLM's dispatch. SelectMulMatQ repeats
// upstream's routing for quantized weights (ggml-cuda.cu:1864-1871 and
// 1924-1942 at llama.cpp b29c606e2) with upstream's own predicates; the
// scratch plans are recorded copies of the host arithmetic by which
// ggml_cuda_mul_mat_vec_q (mmvq.cu:1484-1486) and ggml_cuda_mul_mat_q
// (mmq.cu:199-278), mul_mat_q_switch_J and launch_mul_mat_q
// (mmq.cuh:1396-1558) size what they draw from the pool.

#include <algorithm>
#include <array>
#include <climits>
#include <cstdint>
#include <expected>
#include <initializer_list>
#include <string>
#include <utility>

#include "base/bytes.h"
#include "common.cuh"
#include "kernels/ggml/cublas.h"
#include "kernels/ggml/mul_mat_q_borrowed.cuh"
#include "kernels/ggml/ops_ext.h"
#include "kernels/ggml/validate_ext.h"
#include "kernels/ggml/validate_util.h"
#include "mmid.cuh"
#include "mmq.cuh"
#include "mmvq.cuh"
#include "quantize.cuh"

// The weight types CheckMulMatQ admits must be those whose MMQ instance
// units the build compiles (third_party/patches/ggml/0002), or mmq.cu's
// switch would reach its abort.
#if !defined(GGML_JITLLM_MMQ_Q2_K) || !defined(GGML_JITLLM_MMQ_IQ2_XXS) ||   \
    !defined(GGML_JITLLM_MMQ_Q8_0) || !defined(GGML_JITLLM_MMQ_Q4_K) ||      \
    !defined(GGML_JITLLM_MMQ_Q5_K) || !defined(GGML_JITLLM_MMQ_Q6_K) ||      \
    !defined(GGML_JITLLM_MMQ_IQ2_XS) || !defined(GGML_JITLLM_MMQ_IQ3_XXS) || \
    !defined(GGML_JITLLM_MMQ_MXFP4) || !defined(GGML_JITLLM_MMQ_NVFP4)
#error "validate_ext.h's quantized weight types need their MMQ instance units"
#endif

namespace jitllm::kernels::ggml {
namespace {

std::unexpected<KernelFailure> Rejected(std::string detail) {
  return std::unexpected(
      KernelFailure{.error = KernelError::kRejected, .detail = std::move(detail)});
}

const ggml_cuda_device_info::cuda_device_info& Device(const LaunchContext& launch) {
  return ggml_cuda_info().devices[launch.device()];
}

constexpr std::uint64_t kBlock = 256;  // the pool's block boundary (launch.h)

// The pool's high-water mark for draws in this order, none freed between.
class Draws {
 public:
  void Add(std::uint64_t bytes) {
    if (bytes == 0) {
      return;  // an unallocated ggml_cuda_pool_alloc draws nothing
    }
    total_ = ((total_ + kBlock - 1) / kBlock * kBlock) + bytes;
  }
  std::uint64_t total() const { return total_; }

 private:
  std::uint64_t total_ = 0;
};

std::expected<void, KernelFailure> CheckNode(const ggml_tensor* node) {
  if (node != nullptr && node->op == GGML_OP_MUL_MAT_ID) {
    return CheckMulMatIdQ(node);
  }
  return CheckMulMatQ(node);
}

// What launch_mul_mat_q draws for the tile size mul_mat_q_switch_J picks,
// and the grid limits of its non-stream-k launch.
std::expected<std::uint64_t, KernelFailure> TileFixup(const LaunchContext& launch, ggml_type type,
                                                      std::int64_t rows, std::int64_t k,
                                                      std::int64_t columns, std::int64_t planes,
                                                      std::int64_t assignments = 0,
                                                      bool compact_experts = false) {
  const auto& device = Device(launch);
  const int cc = device.cc;
  const bool fallback = rows % 128 != 0;
  int best_j = 0;
  std::int64_t best_tiles = INT_MAX;
  for (int j = 8; j <= 128 && best_tiles > 1; j += 8) {
    const ggml_cuda_mmq_config config = ggml_cuda_mmq_get_config(type, j, fallback, cc);
    if (config.type == GGML_TYPE_COUNT || mmq_get_nbytes_shared(config, cc) > device.smpbo) {
      continue;
    }
    const std::int64_t tiles = (columns + config.J - 1) / config.J;
    if (tiles < best_tiles) {
      best_j = j;
      best_tiles = tiles;
    }
  }
  if (best_j == 0) {
    return Rejected("no MMQ tile size fits the device's shared memory");
  }
  const ggml_cuda_mmq_config config = ggml_cuda_mmq_get_config(type, best_j, fallback, cc);
  const std::int64_t nty = (rows + config.I - 1) / config.I;
  const std::int64_t ntx = (columns + config.J - 1) / config.J;
  const std::int64_t ntiles = ntx * nty * planes;
  const std::int64_t blocks_per_row = k / ggml_blck_size(type);
  if (config.nthreads % device.warp_size != 0 || ntiles * blocks_per_row >= (1LL << 30)) {
    return Rejected("an MMQ launch beyond its tile counters");
  }
  const auto capacity =
      compact_experts && GGML_CUDA_CC_IS_NVIDIA(cc)
          ? mmq_compact_expert_capacity(type, rows, columns, assignments, planes, config.J)
          : 0;
  if (capacity != 0) {
    return static_cast<std::uint64_t>(capacity) * sizeof(int2);
  }
  if (!config.stream_k) {
    if (ntx > 65535 || planes > 65535 || nty > INT_MAX) {
      return Rejected("an MMQ launch beyond its grid");
    }
    return 0;
  }
  const int nsm = device.nsm;
  const std::int64_t waves = (ntiles + nsm - 1) / nsm;
  const std::int64_t efficiency = 100 * ntiles / (nsm * waves);
  const std::int64_t blocks = GGML_CUDA_CC_IS_NVIDIA(cc) && efficiency >= 90 ? ntiles : nsm;
  if (blocks > INT_MAX) {
    return Rejected("an MMQ launch beyond its grid");
  }
  if (ntiles % blocks == 0) {
    return 0;  // no fixup
  }
  return static_cast<std::uint64_t>(blocks) * static_cast<std::uint64_t>(config.J) *
         static_cast<std::uint64_t>(config.I) * sizeof(float);
}

}  // namespace

std::expected<QuantMulMatPath, KernelFailure> SelectMulMatQ(const LaunchContext& launch,
                                                            const ggml_tensor* node) {
  if (auto checked = CheckNode(node); !checked) {
    return std::unexpected(checked.error());
  }
  const ggml_tensor* weights = node->src[0];
  const int cc = Device(launch).cc;
  if (node->op == GGML_OP_MUL_MAT_ID) {
    // ggml_cuda_mul_mat_id: the vector kernel up to the type's batch, then
    // the tile kernel; for quantized weights never the float families.
    const std::int64_t tokens = node->ne[2];
    if (tokens <= MMVQ_MAX_BATCH_SIZE && tokens <= get_mmvq_mmid_max_batch(weights->type, cc)) {
      return QuantMulMatPath::kVector;
    }
    if (ggml_cuda_should_use_mmq(weights->type, cc, node->src[1]->ne[2], weights->ne[2])) {
      return QuantMulMatPath::kTile;
    }
    return Rejected("upstream takes the sorting fallback for these experts, not implemented");
  }
  // ggml_cuda_mul_mat: MMVF and MMF take no quantized weights, so MMVQ, then
  // MMQ, then cuBLAS over dequantized weights.
  const std::int64_t columns = node->src[1]->ne[1];
  if (ggml_cuda_should_use_mmvq(weights->type, cc, columns)) {
    return QuantMulMatPath::kVector;
  }
  if (ggml_cuda_should_use_mmq(weights->type, cc, columns, /*n_experts=*/0)) {
    return QuantMulMatPath::kTile;
  }
  return Rejected("upstream takes cuBLAS over dequantized weights here, which is not implemented");
}

std::expected<std::uint64_t, KernelFailure> PlanMulMatVecQ(const LaunchContext& launch,
                                                           const ggml_tensor* node) {
  auto path = SelectMulMatQ(launch, node);
  if (!path) {
    return std::unexpected(path.error());
  }
  if (*path != QuantMulMatPath::kVector) {
    return Rejected("upstream does not select MMVQ for these operands");
  }
  const ggml_tensor* input = node->src[1];
  // The launcher's grid: row blocks, then output channels and samples, or
  // for experts the selected experts and tokens (mmvq.cu:983-992).
  if (node->op == GGML_OP_MUL_MAT && (node->ne[2] > 65535 || node->ne[3] > 65535)) {
    return Rejected("MMVQ beyond its grid");
  }
  if (node->op == GGML_OP_MUL_MAT_ID && (node->ne[1] > 65535 || input->ne[1] > 65535)) {
    return Rejected("MMVQ beyond its grid: more selected experts than a grid column holds");
  }
  if (input->ne[2] * input->ne[3] > 65535) {
    return Rejected("MMVQ's activation quantization beyond its grid");
  }
  const std::int64_t padded = GGML_PAD(input->ne[0], MATRIX_ROW_PADDING);
  Draws draws;
  draws.Add(static_cast<std::uint64_t>(input->ne[3] * input->ne[2] * input->ne[1] * padded) *
            sizeof(block_q8_1) / QK8_1);
  return draws.total();
}

static std::expected<std::uint64_t, KernelFailure> PlanMulMatQPrepared(const LaunchContext& launch,
                                                                       const ggml_tensor* node,
                                                                       bool compact_experts) {
  auto path = SelectMulMatQ(launch, node);
  if (!path) {
    return std::unexpected(path.error());
  }
  if (*path != QuantMulMatPath::kTile) {
    return Rejected("upstream does not select MMQ for these operands");
  }
  const ggml_tensor* weights = node->src[0];
  const ggml_tensor* input = node->src[1];
  const int cc = Device(launch).cc;
  const bool fallback = weights->ne[1] % 128 != 0;
  // Blackwell's FP4 tensor cores take FP4 activations for MXFP4 and NVFP4
  // weights; NVFP4's also carry one F32 scale per activation row
  // (mmq.cu:131-140, 207-213).
  const bool native_fp4 = blackwell_mma_available(cc) &&
                          (weights->type == GGML_TYPE_MXFP4 || weights->type == GGML_TYPE_NVFP4);
  const bool row_scales = native_fp4 && weights->type == GGML_TYPE_NVFP4;
  const std::uint64_t y_block = native_fp4 ? sizeof(block_fp4_mmq) : sizeof(block_q8_1_mmq);
  const auto y_values = static_cast<std::uint64_t>(native_fp4 ? QK_FP4_MMQ : QK8_1_MMQ);
  const auto padded = static_cast<std::uint64_t>(GGML_PAD(input->ne[0], MATRIX_ROW_PADDING));
  // A routed activation's slot axis (usually 1 or 6) does not bound J:
  // the tile spans sorted token assignments and loads all its columns.
  const auto pad_columns = node->op == GGML_OP_MUL_MAT_ID ? 128 : input->ne[1];
  const std::uint64_t j_max =
      static_cast<std::uint64_t>(ggml_cuda_mmq_get_J_max(weights->type, fallback, cc, pad_columns));
  Draws draws;
  if (node->op == GGML_OP_MUL_MAT) {
    const auto columns = static_cast<std::uint64_t>(input->ne[3] * input->ne[2] * input->ne[1]);
    // The activations' quantization grid: columns, then channels and
    // samples (quantize.cu:575-605).
    if (input->ne[2] * input->ne[3] > 65535) {
      return Rejected("MMQ's activation quantization beyond its grid");
    }
    draws.Add((columns * padded * y_block / y_values) + (j_max * sizeof(block_q8_1_mmq)));
    if (row_scales) {
      draws.Add((columns + j_max) * sizeof(float));
    }
    auto fixup = TileFixup(launch, weights->type, weights->ne[1], weights->ne[0], node->ne[1],
                           input->ne[2] * input->ne[3]);
    if (!fixup) {
      return std::unexpected(fixup.error());
    }
    draws.Add(*fixup);
    return draws.total();
  }
  // mul_mat_id: the expert maps, then the activations quantized per
  // selected expert and token (mmq.cu:207-262). The maps' helper packs a
  // token and a slot into 22 and 10 bits and holds 4 bytes per token in
  // shared memory (mmid.cu:5-20, 136-147).
  const ggml_tensor* ids = node->src[2];
  if (input->ne[2] >= (1 << 22) || ids->ne[0] >= (1 << 10) ||
      static_cast<std::uint64_t>(input->ne[2]) * sizeof(std::uint32_t) > Device(launch).smpbo) {
    return Rejected("more tokens or selected experts than MMQ's expert maps hold");
  }
  const auto rows = static_cast<std::uint64_t>(input->ne[2] * ids->ne[0]);
  draws.Add(rows * sizeof(std::int32_t));
  draws.Add((rows + j_max) * sizeof(std::int32_t));
  draws.Add(static_cast<std::uint64_t>(weights->ne[2] + 1) * sizeof(std::int32_t));
  draws.Add((rows * padded * y_block / y_values) + (j_max * sizeof(block_q8_1_mmq)));
  if (row_scales) {
    draws.Add((rows + j_max) * sizeof(float));
  }
  // The tile grid spans every token for each expert (ncols_max: tokens).
  auto fixup = TileFixup(launch, weights->type, weights->ne[1], weights->ne[0], input->ne[2],
                         weights->ne[2], input->ne[2] * ids->ne[0], compact_experts);
  if (!fixup) {
    return std::unexpected(fixup.error());
  }
  draws.Add(*fixup);
  return draws.total();
}

std::expected<std::uint64_t, KernelFailure> PlanMulMatQ(const LaunchContext& launch,
                                                        const ggml_tensor* node) {
  return PlanMulMatQPrepared(launch, node, false);
}

std::expected<std::uint64_t, KernelFailure> PlanMulMatIdQCompact(const LaunchContext& launch,
                                                                 const ggml_tensor* node) {
  if (auto checked = CheckMulMatIdQCompact(node); !checked) {
    return std::unexpected(checked.error());
  }
  return PlanMulMatQPrepared(launch, node, true);
}

std::expected<void, KernelFailure> MulMatVecQ(LaunchContext& launch, ggml_tensor* node) {
  auto scratch = PlanMulMatVecQ(launch, node);
  if (!scratch) {
    return std::unexpected(scratch.error());
  }
  return launch.Run(base::Bytes(*scratch), [node](ggml_backend_cuda_context& context) {
    const ggml_tensor* ids = node->op == GGML_OP_MUL_MAT_ID ? node->src[2] : nullptr;
    ggml_cuda_mul_mat_vec_q(context, node->src[0], node->src[1], ids, node);
  });
}

std::expected<void, KernelFailure> MulMatQ(LaunchContext& launch, ggml_tensor* node) {
  auto scratch = PlanMulMatQ(launch, node);
  if (!scratch) {
    return std::unexpected(scratch.error());
  }
  return launch.Run(base::Bytes(*scratch), [node](ggml_backend_cuda_context& context) {
    const ggml_tensor* ids = node->op == GGML_OP_MUL_MAT_ID ? node->src[2] : nullptr;
    ggml_cuda_mul_mat_q(context, node->src[0], node->src[1], ids, node);
  });
}

std::expected<std::uint64_t, KernelFailure> PlanMulMatQBorrowedD4(const LaunchContext& launch,
                                                                  const ggml_tensor* node,
                                                                  const BorrowedMmqD4& input,
                                                                  std::uint64_t generation) {
  if (auto checked = CheckMulMatQBorrowedD4(node, input, generation); !checked)
    return std::unexpected(checked.error());
  auto path = SelectMulMatQ(launch, node);
  if (!path || *path != QuantMulMatPath::kTile)
    return path ? Rejected("native MMQ does not select these borrowed dense operands")
                : std::unexpected(path.error());
  const auto* weights = node->src[0];
  const auto* source = node->src[1];
  const auto j = ggml_cuda_mmq_get_J_max(GGML_TYPE_Q8_0, weights->ne[1] % 128 != 0,
                                         Device(launch).cc, source->ne[1]);
  const auto payload = static_cast<std::uint64_t>(source->ne[0] / 128) *
                       static_cast<std::uint64_t>(source->ne[1]) * 144;
  if (j <= 0 || j > 128 || input.bytes - payload < static_cast<std::uint64_t>(j) * 144)
    return Rejected("borrowed D4 does not cover the actual native MMQ guard");
  const auto weight_bytes = detail::Extent(weights);
  const auto source_bytes = detail::Extent(source);
  const auto output_bytes = detail::Extent(node);
  if (!weight_bytes || !source_bytes || !output_bytes)
    return Rejected("borrowed MMQ has unmeasurable logical operands");
  const std::array operands{std::pair{input.data, input.bytes},
                            std::pair{static_cast<const void*>(weights->data), *weight_bytes},
                            std::pair{static_cast<const void*>(source->data), *source_bytes},
                            std::pair{static_cast<const void*>(node->data), *output_bytes}};
  const auto disjoint = [&](LaunchContext::Workspace workspace) {
    std::uint64_t end = 0;
    if (__builtin_add_overflow(workspace.base, workspace.size.value(), &end)) return false;
    if (workspace.size.value() == 0) return true;
    return std::ranges::all_of(operands, [&](const auto& operand) {
      const auto begin = reinterpret_cast<std::uintptr_t>(operand.first);
      return end <= begin || begin + operand.second <= workspace.base;
    });
  };
  if (!disjoint(launch.workspace()) ||
      (launch.cublas() != nullptr && !disjoint(launch.cublas()->workspace())))
    return Rejected("borrowed MMQ operands overlap a native workspace");
  return TileFixup(launch, GGML_TYPE_Q8_0, weights->ne[1], weights->ne[0], node->ne[1], 1);
}

void internal::LaunchMulMatQBorrowedD4(ggml_backend_cuda_context& context, const ggml_tensor* node,
                                       const void* quantized) {
  const auto* weights = node->src[0];
  const auto* source = node->src[1];
  const auto sy = source->ne[1] * source->ne[0] *
                  static_cast<std::int64_t>(sizeof(block_q8_1_mmq)) /
                  (QK8_1_MMQ * static_cast<std::int64_t>(sizeof(int)));
  const auto type_bytes = static_cast<std::int64_t>(ggml_type_size(GGML_TYPE_Q8_0));
  const mmq_args args{static_cast<const char*>(weights->data),
                      GGML_TYPE_Q8_0,
                      static_cast<const int*>(quantized),
                      nullptr,
                      nullptr,
                      static_cast<float*>(node->data),
                      nullptr,
                      weights->ne[0],
                      weights->ne[1],
                      node->ne[1],
                      static_cast<std::int64_t>(weights->nb[1]) / type_bytes,
                      source->ne[1],
                      static_cast<std::int64_t>(node->nb[1] / sizeof(float)),
                      1,
                      1,
                      static_cast<std::int64_t>(weights->nb[2]) / type_bytes,
                      sy,
                      static_cast<std::int64_t>(node->nb[2] / sizeof(float)),
                      1,
                      1,
                      static_cast<std::int64_t>(weights->nb[3]) / type_bytes,
                      sy,
                      static_cast<std::int64_t>(node->nb[3] / sizeof(float)),
                      node->ne[1],
                      node->ne[1],
                      false};
  mul_mat_q_case<GGML_TYPE_Q8_0>(context, args, context.stream());
  CUDA_CHECK(cudaGetLastError());
}

std::expected<void, KernelFailure> MulMatQBorrowedD4(LaunchContext& launch, ggml_tensor* node,
                                                     const BorrowedMmqD4& input,
                                                     std::uint64_t generation) {
  auto scratch = PlanMulMatQBorrowedD4(launch, node, input, generation);
  if (!scratch) return std::unexpected(scratch.error());
  return launch.Run(base::Bytes(*scratch), [node, input](ggml_backend_cuda_context& context) {
    internal::LaunchMulMatQBorrowedD4(context, node, input.data);
  });
}

std::expected<std::uint64_t, KernelFailure> PlanMulMatIdQPair(const LaunchContext& launch,
                                                              const ggml_tensor* first,
                                                              const ggml_tensor* second,
                                                              bool compact_experts) {
  if (auto checked = CheckMulMatIdQPair(first, second); !checked) {
    return std::unexpected(checked.error());
  }
  auto a = PlanMulMatQPrepared(launch, first, compact_experts);
  if (!a) {
    return std::unexpected(a.error());
  }
  auto b = PlanMulMatQPrepared(launch, second, compact_experts);
  if (!b) {
    return std::unexpected(b.error());
  }
  // Maps and quantization match; each product returns its fixup or compact
  // tile list before the next draws it. Their high-water marks do not add.
  return std::max(*a, *b);
}

std::expected<ExpertMmqLayout, KernelFailure> DescribeMulMatIdQPairPrepared(
    const LaunchContext& launch, const ggml_tensor* first, const ggml_tensor* second) {
  if (auto checked = CheckMulMatIdQPair(first, second); !checked)
    return std::unexpected(checked.error());
  const auto* weights = first->src[0];
  const auto* source = first->src[1];
  const auto* ids = first->src[2];
  constexpr std::array<std::int64_t, 4> weight_shape{4096, 2048, 256, 1};
  constexpr std::array<std::int64_t, 4> input_shape{4096, 1, 4096, 1};
  constexpr std::array<std::int64_t, 4> ids_shape{6, 4096, 1, 1};
  constexpr std::array<std::int64_t, 4> output_shape{2048, 6, 4096, 1};
  if (Device(launch).cc != 1210 || weights->type != GGML_TYPE_IQ2_XXS ||
      ggml_type_size(GGML_TYPE_IQ2_XXS) != 66 || ggml_blck_size(GGML_TYPE_IQ2_XXS) != 256 ||
      sizeof(block_q8_1_mmq) != 144 || QK8_1_MMQ != 128 ||
      !std::ranges::equal(weights->ne, weight_shape) ||
      !std::ranges::equal(source->ne, input_shape) ||
      !std::ranges::equal(ids->ne, ids_shape) ||
      !std::ranges::equal(first->ne, output_shape) ||
      !std::ranges::all_of(std::array<const ggml_tensor*, 6>{weights, second->src[0], source,
                                                           ids, first, second},
                           [](const ggml_tensor* tensor) { return detail::Packed(tensor); }))
    return Rejected("prepared IQ2 controls require the fixed packed GB10 capture geometry");
  auto selected = SelectMulMatQ(launch, first);
  if (!selected || *selected != QuantMulMatPath::kTile)
    return selected ? Rejected("prepared IQ2 controls require the native MMQ tile path")
                    : std::unexpected(selected.error());
  // The same J-max call as RunExpertProducts, not a presumed128-column guard.
  const auto j = ggml_cuda_mmq_get_J_max(GGML_TYPE_IQ2_XXS, false, Device(launch).cc, 128);
  if (j <= 0 || j > 128)
    return Rejected("prepared IQ2 native J guard exceeds the bounded captured tier");
  constexpr std::uint64_t pairs = 4096ULL * 6;
  constexpr std::uint64_t payload = (4096ULL / 128) * pairs * 144;
  return ExpertMmqLayout{.activation_payload_bytes = payload,
                        .activation_bytes = payload + static_cast<std::uint64_t>(j) * 144,
                        .source_ids_bytes = pairs * 4,
                        .destination_ids_payload_bytes = pairs * 4,
                        .destination_ids_bytes = (pairs + static_cast<std::uint64_t>(j)) * 4,
                        .bounds_bytes = 257ULL * 4,
                        .native_j_max = static_cast<std::uint64_t>(j)};
}

namespace {
struct ExpertRange {
  std::uint64_t begin = 0;
  std::uint64_t end = 0;
};
bool ExpertExtent(const void* data, std::uint64_t bytes, ExpertRange& result) {
  const auto begin = static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(data));
  std::uint64_t end = 0;
  if (data == nullptr || bytes == 0 || __builtin_add_overflow(begin, bytes, &end)) return false;
  result = {.begin = begin, .end = end};
  return true;
}
bool ExpertDisjoint(ExpertRange a, ExpertRange b) {
  return a.end <= b.begin || b.end <= a.begin;
}

std::expected<ExpertMmqLayout, KernelFailure> CheckExpertPrepared(
    const LaunchContext& launch, const ggml_tensor* first, const ggml_tensor* second,
    const ExpertMmqPrepared& input, std::uint64_t generation) {
  auto layout = DescribeMulMatIdQPairPrepared(launch, first, second);
  if (!layout) return std::unexpected(layout.error());
  if (generation == 0 || input.generation != generation ||
      input.source != first->src[1]->data || input.selected != first->src[2]->data)
    return Rejected("prepared IQ2 source, selected IDs or generation is stale");
  const std::array buffers{input.activation, input.source_ids, input.destination_ids, input.bounds};
  const std::array required{layout->activation_bytes, layout->source_ids_bytes,
                            layout->destination_ids_bytes, layout->bounds_bytes};
  std::array<ExpertRange, 4> prepared{};
  for (std::size_t i = 0; i < buffers.size(); ++i) {
    if (buffers[i].bytes < required[i] || buffers[i].bytes % 4 != 0 ||
        reinterpret_cast<std::uintptr_t>(buffers[i].data) % (i == 0 ? 16 : 4) != 0 ||
        !ExpertExtent(buffers[i].data, buffers[i].bytes, prepared[i]))
      return Rejected("prepared IQ2 buffer capacity, alignment or address extent is invalid");
    for (std::size_t previous = 0; previous < i; ++previous)
      if (!ExpertDisjoint(prepared[i], prepared[previous]))
        return Rejected("prepared IQ2 activation/map buffers overlap");
  }
  const std::array<const ggml_tensor*, 6> operands{first->src[0], second->src[0], first->src[1],
                                                 first->src[2], first, second};
  std::array<ExpertRange, 6> logical{};
  for (std::size_t i = 0; i < operands.size(); ++i) {
    const auto bytes = detail::Extent(operands[i]);
    if (!bytes || !ExpertExtent(operands[i]->data, *bytes, logical[i]))
      return Rejected("prepared IQ2 logical operand extent is invalid");
    for (const auto range : prepared)
      if (!ExpertDisjoint(range, logical[i]))
        return Rejected("prepared IQ2 buffers overlap a logical product operand");
  }
  for (const auto buffer : input.retained) {
    if (buffer.data == nullptr && buffer.bytes == 0) continue;
    ExpertRange retained{};
    if (!ExpertExtent(buffer.data, buffer.bytes, retained))
      return Rejected("prepared IQ2 retained extent is invalid");
    for (const auto range : prepared)
      if (!ExpertDisjoint(range, retained))
        return Rejected("prepared IQ2 buffers overlap retained original controls");
    if (!ExpertDisjoint(logical[4], retained) || !ExpertDisjoint(logical[5], retained))
      return Rejected("prepared IQ2 outputs overlap retained original controls");
  }
  const auto disjoint_workspace = [&](LaunchContext::Workspace workspace) {
    if (workspace.size.value() == 0) return true;
    ExpertRange range{};
    if (!ExpertExtent(reinterpret_cast<const void*>(workspace.base), workspace.size.value(), range))
      return false;
    if (!std::ranges::all_of(prepared, [&](ExpertRange p) { return ExpertDisjoint(p, range); }) ||
        !std::ranges::all_of(logical, [&](ExpertRange p) { return ExpertDisjoint(p, range); }))
      return false;
    return std::ranges::all_of(input.retained, [&](ExpertMmqBuffer b) {
      if (b.data == nullptr && b.bytes == 0) return true;
      ExpertRange retained{};
      return ExpertExtent(b.data, b.bytes, retained) && ExpertDisjoint(retained, range);
    });
  };
  if (!disjoint_workspace(launch.workspace()) ||
      (launch.cublas() != nullptr && !disjoint_workspace(launch.cublas()->workspace())))
    return Rejected("prepared IQ2 live operands overlap a native workspace");
  return *layout;
}
}  // namespace

static std::expected<void, KernelFailure> RunExpertProducts(LaunchContext& launch,
                                                            ggml_tensor* first, ggml_tensor* second,
                                                            bool compact_experts,
                                                            const ExpertMmqPrepared* capture = nullptr) {
  auto scratch = second != nullptr ? PlanMulMatIdQPair(launch, first, second, compact_experts)
                                   : PlanMulMatIdQCompact(launch, first);
  if (!scratch) {
    return std::unexpected(scratch.error());
  }
  return launch.Run(
      base::Bytes(*scratch), [first, second, compact_experts, capture](ggml_backend_cuda_context& context) {
        // Host preparation from GGML mmq.cu: the same inverse broadcast map
        // and scatter quantization feed two ordinary MMQ launches. This is
        // ds4's paired-preparation technique without its SoA repack or fused
        // early route weighting.
        const ggml_tensor* weights = first->src[0];
        const ggml_tensor* input = first->src[1];
        const ggml_tensor* ids = first->src[2];
        const auto type = weights->type;
        const auto used = ids->ne[0];
        const auto tokens = input->ne[2];
        const auto rows = tokens * used;
        const auto padded = GGML_PAD(input->ne[0], MATRIX_ROW_PADDING);
        const int cc = ggml_cuda_info().devices[context.device].cc;
        const auto j_max = ggml_cuda_mmq_get_J_max(type, weights->ne[1] % 128 != 0, cc, 128);
        cudaStream_t stream = context.stream();
        ggml_cuda_pool_alloc<std::int32_t> ids_src(context.pool(), static_cast<std::size_t>(rows));
        ggml_cuda_pool_alloc<std::int32_t> ids_dst(context.pool(),
                                                   static_cast<std::size_t>(rows + j_max));
        ggml_cuda_pool_alloc<std::int32_t> bounds(context.pool(),
                                                  static_cast<std::size_t>(weights->ne[2] + 1));
        const bool broadcast = input->ne[1] == 1 && used > 1;
        ggml_cuda_launch_mm_ids_helper(
            static_cast<const std::int32_t*>(ids->data), ids_src.get(), ids_dst.get(), bounds.get(),
            static_cast<int>(weights->ne[2]), static_cast<int>(tokens), static_cast<int>(used),
            static_cast<int>(input->ne[1]), static_cast<int>(ids->nb[1] / sizeof(std::int32_t)),
            static_cast<int>(input->nb[2] / input->nb[1]), broadcast, stream);
        CUDA_CHECK(cudaGetLastError());
        const auto bytes =
            static_cast<std::size_t>(rows * padded) * sizeof(block_q8_1_mmq) / QK8_1_MMQ +
            static_cast<std::size_t>(j_max) * sizeof(block_q8_1_mmq);
        ggml_cuda_pool_alloc<char> quantized(context.pool(), bytes);
        const auto* x = static_cast<const float*>(input->data);
        const auto s11 = static_cast<std::int64_t>(input->nb[1] / sizeof(float));
        const auto s12 = static_cast<std::int64_t>(input->nb[2] / sizeof(float));
        const auto s13 = static_cast<std::int64_t>(input->nb[3] / sizeof(float));
        if (broadcast) {
          quantize_scatter_mmq_q8_1_cuda(x, ids_src.get(), quantized.get(), type, input->ne[0], s12,
                                         padded, tokens, rows, static_cast<int>(used), stream);
        } else {
          quantize_mmq_q8_1_cuda(x, ids_src.get(), quantized.get(), type, input->ne[0], s11, s12,
                                 s13, padded, rows, 1, 1, stream);
        }
        CUDA_CHECK(cudaGetLastError());
        if (capture != nullptr) {
          // Diagnostic copies preserve every private guard byte without
          // modifying or assuming initialization of its source buffer.
          CUDA_CHECK(cudaMemcpyAsync(capture->activation.data, quantized.get(), bytes,
                                     cudaMemcpyDeviceToDevice, stream));
          CUDA_CHECK(cudaMemcpyAsync(capture->source_ids.data, ids_src.get(),
                                     static_cast<std::size_t>(rows) * sizeof(std::int32_t),
                                     cudaMemcpyDeviceToDevice, stream));
          CUDA_CHECK(cudaMemcpyAsync(capture->destination_ids.data, ids_dst.get(),
                                     static_cast<std::size_t>(rows + j_max) * sizeof(std::int32_t),
                                     cudaMemcpyDeviceToDevice, stream));
          CUDA_CHECK(cudaMemcpyAsync(capture->bounds.data, bounds.get(),
                                     static_cast<std::size_t>(weights->ne[2] + 1) * sizeof(std::int32_t),
                                     cudaMemcpyDeviceToDevice, stream));
        }
        const auto sy2 = input->ne[1] * padded * static_cast<std::int64_t>(sizeof(block_q8_1)) /
                         (QK8_1 * static_cast<std::int64_t>(sizeof(int)));
        const auto sy3 = tokens * sy2;
        const auto ncols_opt = GGML_CUDA_CC_IS_RDNA3_0(cc) || GGML_CUDA_CC_IS_RDNA4(cc)
                                   ? (rows + weights->ne[2] - 1) / weights->ne[2]
                                   : tokens;
        for (ggml_tensor* output : {first, second}) {
          if (output == nullptr) {
            continue;
          }
          const ggml_tensor* w = output->src[0];
          const auto ts = static_cast<std::int64_t>(ggml_type_size(type));
          const mmq_args args = {static_cast<const char*>(w->data),
                                 type,
                                 reinterpret_cast<const int*>(quantized.get()),
                                 ids_dst.get(),
                                 bounds.get(),
                                 static_cast<float*>(output->data),
                                 nullptr,
                                 w->ne[0],
                                 w->ne[1],
                                 rows,
                                 static_cast<std::int64_t>(w->nb[1]) / ts,
                                 rows,
                                 static_cast<std::int64_t>(output->nb[1] / sizeof(float)),
                                 w->ne[2],
                                 w->ne[2],
                                 static_cast<std::int64_t>(w->nb[2]) / ts,
                                 sy2,
                                 static_cast<std::int64_t>(output->nb[2] / sizeof(float)),
                                 w->ne[3],
                                 input->ne[3],
                                 static_cast<std::int64_t>(w->nb[3]) / ts,
                                 sy3,
                                 static_cast<std::int64_t>(output->nb[3] / sizeof(float)),
                                 tokens,
                                 ncols_opt,
                                 compact_experts};
          switch (type) {
            case GGML_TYPE_Q8_0:
              mul_mat_q_case<GGML_TYPE_Q8_0>(context, args, stream);
              break;
            case GGML_TYPE_Q2_K:
              mul_mat_q_case<GGML_TYPE_Q2_K>(context, args, stream);
              break;
            case GGML_TYPE_Q4_K:
              mul_mat_q_case<GGML_TYPE_Q4_K>(context, args, stream);
              break;
            case GGML_TYPE_Q5_K:
              mul_mat_q_case<GGML_TYPE_Q5_K>(context, args, stream);
              break;
            case GGML_TYPE_Q6_K:
              mul_mat_q_case<GGML_TYPE_Q6_K>(context, args, stream);
              break;
            case GGML_TYPE_IQ2_XXS:
              mul_mat_q_case<GGML_TYPE_IQ2_XXS>(context, args, stream);
              break;
            case GGML_TYPE_IQ2_XS:
              mul_mat_q_case<GGML_TYPE_IQ2_XS>(context, args, stream);
              break;
            case GGML_TYPE_IQ3_XXS:
              mul_mat_q_case<GGML_TYPE_IQ3_XXS>(context, args, stream);
              break;
            default:
              GGML_ABORT("paired MMQ type passed validation without its case");
          }
        }
      });
}

std::expected<void, KernelFailure> MulMatIdQPair(LaunchContext& launch, ggml_tensor* first,
                                                 ggml_tensor* second, bool compact_experts) {
  return RunExpertProducts(launch, first, second, compact_experts);
}

std::expected<void, KernelFailure> MulMatIdQCompact(LaunchContext& launch, ggml_tensor* node) {
  return RunExpertProducts(launch, node, nullptr, true);
}

std::expected<void, KernelFailure> MulMatIdQPairCapture(
    LaunchContext& launch, ggml_tensor* first, ggml_tensor* second,
    const ExpertMmqPrepared& capture, std::uint64_t generation) {
  if (auto checked = CheckExpertPrepared(launch, first, second, capture, generation); !checked)
    return std::unexpected(checked.error());
  return RunExpertProducts(launch, first, second, true, &capture);
}

std::expected<std::uint64_t, KernelFailure> PlanMulMatIdQPairBorrowed(
    const LaunchContext& launch, const ggml_tensor* first, const ggml_tensor* second,
    const ExpertMmqPrepared& input, std::uint64_t generation) {
  if (auto checked = CheckExpertPrepared(launch, first, second, input, generation); !checked)
    return std::unexpected(checked.error());
  const auto* weights = first->src[0];
  return TileFixup(launch, GGML_TYPE_IQ2_XXS, weights->ne[1], weights->ne[0],
                   first->src[1]->ne[2] * first->src[2]->ne[0], weights->ne[2],
                   first->src[1]->ne[2] * first->src[2]->ne[0], true);
}

std::expected<void, KernelFailure> MulMatIdQPairBorrowed(
    LaunchContext& launch, ggml_tensor* first, ggml_tensor* second,
    const ExpertMmqPrepared& input, std::uint64_t generation) {
  auto scratch = PlanMulMatIdQPairBorrowed(launch, first, second, input, generation);
  if (!scratch) return std::unexpected(scratch.error());
  return launch.Run(base::Bytes(*scratch), [first, second, input](ggml_backend_cuda_context& context) {
    const auto* source = first->src[1];
    const auto* ids = first->src[2];
    const auto tokens = source->ne[2];
    const auto rows = tokens * ids->ne[0];
    const auto padded = GGML_PAD(source->ne[0], MATRIX_ROW_PADDING);
    const auto sy2 = source->ne[1] * padded * static_cast<std::int64_t>(sizeof(block_q8_1)) /
                     (QK8_1 * static_cast<std::int64_t>(sizeof(int)));
    const auto sy3 = tokens * sy2;
    for (ggml_tensor* output : {first, second}) {
      const auto* weights = output->src[0];
      const auto ts = static_cast<std::int64_t>(ggml_type_size(GGML_TYPE_IQ2_XXS));
      const mmq_args args{static_cast<const char*>(weights->data),
                         GGML_TYPE_IQ2_XXS,
                         static_cast<const int*>(input.activation.data),
                         static_cast<const std::int32_t*>(input.destination_ids.data),
                         static_cast<const std::int32_t*>(input.bounds.data),
                         static_cast<float*>(output->data), nullptr,
                         weights->ne[0], weights->ne[1], rows,
                         static_cast<std::int64_t>(weights->nb[1]) / ts, rows,
                         static_cast<std::int64_t>(output->nb[1] / sizeof(float)),
                         weights->ne[2], weights->ne[2],
                         static_cast<std::int64_t>(weights->nb[2]) / ts, sy2,
                         static_cast<std::int64_t>(output->nb[2] / sizeof(float)),
                         weights->ne[3], source->ne[3],
                         static_cast<std::int64_t>(weights->nb[3]) / ts, sy3,
                         static_cast<std::int64_t>(output->nb[3] / sizeof(float)),
                         tokens, tokens, true};
      mul_mat_q_case<GGML_TYPE_IQ2_XXS>(context, args, context.stream());
      CUDA_CHECK(cudaGetLastError());
    }
  });
}

}  // namespace jitllm::kernels::ggml
