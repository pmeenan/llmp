// SPDX-FileCopyrightText: 2023-2026 The ggml authors
// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: MIT AND Apache-2.0

// GGML's quantized matrix products (ops_ext.h): the vector (MMVQ) and tile
// (MMQ) kernel families under llmpalooza's dispatch. SelectMulMatQ repeats
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
#include "kernels/ggml/llmp_ops.h"
#include "kernels/ggml/mul_mat_q_glu.cuh"
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
#if !defined(GGML_LLMP_MMQ_Q2_K) || !defined(GGML_LLMP_MMQ_IQ2_XXS) ||   \
    !defined(GGML_LLMP_MMQ_Q8_0) || !defined(GGML_LLMP_MMQ_Q4_K) ||      \
    !defined(GGML_LLMP_MMQ_Q5_K) || !defined(GGML_LLMP_MMQ_Q6_K) ||      \
    !defined(GGML_LLMP_MMQ_IQ2_XS) || !defined(GGML_LLMP_MMQ_IQ3_XXS) || \
    !defined(GGML_LLMP_MMQ_MXFP4) || !defined(GGML_LLMP_MMQ_NVFP4) ||    \
    !defined(GGML_LLMP_MMQ_Q4_1) || !defined(GGML_LLMP_MMQ_Q5_0) ||      \
    !defined(GGML_LLMP_MMQ_Q5_1) || !defined(GGML_LLMP_MMQ_Q4_0) ||      \
    !defined(GGML_LLMP_MMQ_Q2_0) || !defined(GGML_LLMP_MMQ_Q3_K) ||      \
    !defined(GGML_LLMP_MMQ_IQ1_S) || !defined(GGML_LLMP_MMQ_IQ2_S) ||    \
    !defined(GGML_LLMP_MMQ_IQ3_S) || !defined(GGML_LLMP_MMQ_IQ4_NL) ||   \
    !defined(GGML_LLMP_MMQ_IQ4_XS)
#error "validate_ext.h's quantized weight types need their MMQ instance units"
#endif

namespace llmp::kernels::ggml {
namespace {

std::unexpected<KernelFailure> Rejected(std::string detail) {
  return std::unexpected(
      KernelFailure{.error = KernelError::kRejected, .detail = std::move(detail)});
}

const ggml_cuda_device_info::cuda_device_info& Device(const LaunchContext& launch) {
  return ggml_cuda_info().devices[launch.device()];
}

// mmq.cu's dispatch of one MMQ launch by weight type, over the types
// CheckMulMatQ admits (the build's instance units).
void MmqCase(ggml_type type, ggml_backend_cuda_context& context, const mmq_args& args,
             cudaStream_t stream) {
  switch (type) {
    case GGML_TYPE_Q8_0:
      mul_mat_q_case<GGML_TYPE_Q8_0>(context, args, stream);
      break;
    case GGML_TYPE_Q2_K:
      mul_mat_q_case<GGML_TYPE_Q2_K>(context, args, stream);
      break;
    case GGML_TYPE_Q3_K:
      mul_mat_q_case<GGML_TYPE_Q3_K>(context, args, stream);
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
    case GGML_TYPE_Q4_1:
      mul_mat_q_case<GGML_TYPE_Q4_1>(context, args, stream);
      break;
    case GGML_TYPE_Q5_0:
      mul_mat_q_case<GGML_TYPE_Q5_0>(context, args, stream);
      break;
    case GGML_TYPE_Q5_1:
      mul_mat_q_case<GGML_TYPE_Q5_1>(context, args, stream);
      break;
    case GGML_TYPE_Q4_0:
      mul_mat_q_case<GGML_TYPE_Q4_0>(context, args, stream);
      break;
    case GGML_TYPE_Q2_0:
      mul_mat_q_case<GGML_TYPE_Q2_0>(context, args, stream);
      break;
    case GGML_TYPE_IQ1_S:
      mul_mat_q_case<GGML_TYPE_IQ1_S>(context, args, stream);
      break;
    case GGML_TYPE_IQ2_XXS:
      mul_mat_q_case<GGML_TYPE_IQ2_XXS>(context, args, stream);
      break;
    case GGML_TYPE_IQ2_XS:
      mul_mat_q_case<GGML_TYPE_IQ2_XS>(context, args, stream);
      break;
    case GGML_TYPE_IQ2_S:
      mul_mat_q_case<GGML_TYPE_IQ2_S>(context, args, stream);
      break;
    case GGML_TYPE_IQ3_XXS:
      mul_mat_q_case<GGML_TYPE_IQ3_XXS>(context, args, stream);
      break;
    case GGML_TYPE_IQ3_S:
      mul_mat_q_case<GGML_TYPE_IQ3_S>(context, args, stream);
      break;
    case GGML_TYPE_IQ4_NL:
      mul_mat_q_case<GGML_TYPE_IQ4_NL>(context, args, stream);
      break;
    case GGML_TYPE_IQ4_XS:
      mul_mat_q_case<GGML_TYPE_IQ4_XS>(context, args, stream);
      break;
    default:
      GGML_ABORT("an MMQ type passed validation without its case");
  }
}

bool Iq2Occ2Pair(const LaunchContext& launch, const ggml_tensor* first, const ggml_tensor* second,
                 bool compact_experts) {
  return compact_experts && Device(launch).cc == 1210 && IsMulMatIdQPairIq2Occ2(first, second);
}

// The pairs whose up product writes the activation (mul_mat_q_glu.cuh).
bool GluPair(const LaunchContext& launch, const ggml_tensor* first, const ggml_tensor* second,
             bool compact_experts) {
  return compact_experts && Device(launch).cc == 1210 && IsMulMatIdQPairGluPair(first, second);
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

ggml_prec PackingPrecision(ggml_type type, int cc) {
  return blackwell_mma_available(cc) && (type == GGML_TYPE_MXFP4 || type == GGML_TYPE_NVFP4)
             ? GGML_PREC_Q4
             : GGML_PREC_Q8;
}

// Same tile selection as mul_mat_q_switch_J. Shared with the fixed-fixture
// description so diagnostics cannot invent a different selection heuristic.
std::expected<ggml_cuda_mmq_config, KernelFailure> ChosenTile(const LaunchContext& launch,
                                                              ggml_type type, std::int64_t rows,
                                                              std::int64_t columns) {
  const auto& device = Device(launch);
  const int cc = device.cc;
  const bool fallback = rows % 128 != 0;
  int best_j = 0;
  std::int64_t best_tiles = INT_MAX;
  for (int j = 8; j <= 128 && best_tiles > 1; j += 8) {
    const ggml_cuda_mmq_config config =
        ggml_cuda_mmq_get_config(type, j, fallback, cc, PackingPrecision(type, cc));
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
  return ggml_cuda_mmq_get_config(type, best_j, fallback, cc, PackingPrecision(type, cc));
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
  const auto chosen = ChosenTile(launch, type, rows, columns);
  if (!chosen) return std::unexpected(chosen.error());
  const auto& config = *chosen;
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

bool DenseMmvqShapeSelected(const LaunchContext& launch, ggml_type type, std::int64_t columns) {
  return columns > 0 && columns <= MMVQ_MAX_BATCH_SIZE &&
         ggml_cuda_should_use_mmvq(type, Device(launch).cc, columns);
}

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
  if (auto rows = CheckMmvqRowFootprint(node, MmvqRowsPerBlock(launch, node)); !rows) {
    return std::unexpected(rows.error());
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

namespace {

std::expected<std::uint64_t, KernelFailure> PlanMulMatQPrepared(const LaunchContext& launch,
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
  const auto j_max = static_cast<std::uint64_t>(ggml_cuda_mmq_get_J_max(
      weights->type, fallback, cc, pad_columns, PackingPrecision(weights->type, cc)));
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

}  // namespace

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

std::expected<std::uint64_t, KernelFailure> PlanMmvqPrepared(const LaunchContext& launch,
                                                             const ggml_tensor* node) {
  auto original = MmvqPreparedOriginal(node);
  if (!original) return std::unexpected(original.error());
  auto planned = PlanMulMatVecQ(launch, &*original);
  if (!planned) return std::unexpected(planned.error());
  const auto* input = original->src[1];
  const auto bytes = static_cast<std::uint64_t>(Q8Bytes(input->ne[0], input->ne[1]));
  if (*planned != bytes)
    return Rejected("ordinary MMVQ preparation differs from its original padded draw");
  return 0;
}

std::expected<void, KernelFailure> RunMmvqPrepared(LaunchContext& launch, ggml_tensor* node) {
  if (auto checked = PlanMmvqPrepared(launch, node); !checked)
    return std::unexpected(checked.error());
  auto original = MmvqPreparedOriginal(node);
  if (!original) return std::unexpected(original.error());
  const auto padded = GGML_PAD(original->src[1]->ne[0], MATRIX_ROW_PADDING);
  // The descriptor is copied into this synchronous host callback. No cached
  // plan or deferred submission retains a pointer to a stack descriptor.
  return launch.Run(base::Bytes(0), [original = *original, q8 = node->src[1],
                                     padded](ggml_backend_cuda_context& context) mutable {
    ggml_cuda_op_mul_mat_vec_q(context, original.src[0], original.src[1], &original,
                               static_cast<const char*>(original.src[0]->data),
                               static_cast<const float*>(original.src[1]->data),
                               static_cast<const char*>(q8->data),
                               static_cast<float*>(original.data), 0, original.ne[0],
                               original.ne[1], padded, context.stream());
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

bool MulMatVecQGluFusible(const LaunchContext& launch, const ggml_tensor* gate,
                          const ggml_tensor* up, const ggml_tensor* glu) {
  if (gate == nullptr || up == nullptr || glu == nullptr || up->op != GGML_OP_MUL_MAT ||
      gate->op != GGML_OP_MUL_MAT || glu->op != GGML_OP_GLU || glu->src[0] != gate ||
      glu->src[1] != up || glu->op_params[1] != 0 || !ggml_is_quantized(up->src[0]->type) ||
      up->src[0]->type != gate->src[0]->type || !ggml_are_same_shape(up->src[0], gate->src[0]) ||
      !ggml_are_same_stride(up->src[0], gate->src[0]) || up->src[1] != gate->src[1] ||
      up->src[1]->type != GGML_TYPE_F32 || up->type != GGML_TYPE_F32 ||
      glu->type != GGML_TYPE_F32 || up->ne[1] != 1 || !ggml_are_same_shape(up, glu) ||
      !ggml_is_contiguous(glu)) {
    return false;
  }
  switch (ggml_get_glu_op(glu)) {
    case GGML_GLU_OP_SWIGLU:
    case GGML_GLU_OP_GEGLU:
    case GGML_GLU_OP_SWIGLU_OAI:
    case GGML_GLU_OP_SWIGLU_CLAMP:
      break;
    default:
      return false;
  }
  if (Device(launch).cc <= GGML_CUDA_CC_PASCAL || !CheckMulMatQ(gate) || !CheckMulMatQ(up)) {
    return false;
  }
  auto path = SelectMulMatQ(launch, up);
  return path && *path == QuantMulMatPath::kVector && up->src[1]->ne[1] <= MMVQ_MAX_BATCH_SIZE;
}

std::expected<std::uint64_t, KernelFailure> PlanMulMatVecQGlu(const LaunchContext& launch,
                                                              const ggml_tensor* gate,
                                                              const ggml_tensor* up,
                                                              const ggml_tensor* glu) {
  if (!MulMatVecQGluFusible(launch, gate, up, glu)) {
    return Rejected("upstream does not fuse these gate/up products and GLU in MMVQ");
  }
  return PlanMulMatVecQ(launch, up);
}

std::expected<void, KernelFailure> MulMatVecQGlu(LaunchContext& launch, ggml_tensor* gate,
                                                 ggml_tensor* up, ggml_tensor* glu) {
  auto scratch = PlanMulMatVecQGlu(launch, gate, up, glu);
  if (!scratch) {
    return std::unexpected(scratch.error());
  }
  return launch.Run(base::Bytes(*scratch), [gate, up, glu](ggml_backend_cuda_context& context) {
    ggml_cuda_mm_fusion_args_host fusion{};
    fusion.gate = gate->src[0];
    fusion.glu_op = ggml_get_glu_op(glu);
    fusion.glu_limit = ggml_get_op_params_f32(glu, 3);
    ggml_cuda_mul_mat_vec_q(context, up->src[0], up->src[1], nullptr, glu, &fusion);
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
  const std::uint64_t ordinary = std::max(*a, *b);
  if (!Iq2Occ2Pair(launch, first, second, compact_experts)) {
    return ordinary;
  }
  // The producer's original J128 Q8/ID guards stay intact. Only the
  // sequential compact worklist grows (at 4,096 tokens from 448 to 640 int2
  // entries), by what J64 tiles add over J128 for this chunk's tokens.
  const auto tokens = first->src[1]->ne[2];
  const auto rows = tokens * first->src[2]->ne[0];
  const ggml_type type = first->src[0]->type;
  const auto before = mmq_compact_expert_capacity(type, 2048, tokens, rows, 256, 128);
  const auto after = mmq_compact_expert_capacity(type, 2048, tokens, rows, 256, 64);
  return ordinary +
         (static_cast<std::uint64_t>(std::max<std::int64_t>(after - before, 0)) * sizeof(int2));
}

namespace {
std::expected<void, KernelFailure> RunExpertProducts(LaunchContext& launch, ggml_tensor* first,
                                                     ggml_tensor* second, bool compact_experts,
                                                     ggml_tensor* glu = nullptr,
                                                     bool glu_q8 = false,
                                                     const ggml_tensor* prequantized = nullptr) {
  auto scratch = second != nullptr ? PlanMulMatIdQPair(launch, first, second, compact_experts)
                                   : PlanMulMatIdQCompact(launch, first);
  if (!scratch) {
    return std::unexpected(scratch.error());
  }
  const bool iq2_occ2 = Iq2Occ2Pair(launch, first, second, compact_experts);
  if (glu != nullptr && !GluPair(launch, first, second, compact_experts)) {
    return Rejected("the pair activation write-back takes the GB10 IQ2_XXS and IQ2_XS pairs only");
  }
  return launch.Run(base::Bytes(*scratch), [first, second, compact_experts, iq2_occ2, glu, glu_q8,
                                            prequantized](ggml_backend_cuda_context& context) {
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
        (static_cast<std::size_t>(rows * padded) * sizeof(block_q8_1_mmq) / QK8_1_MMQ) +
        (static_cast<std::size_t>(j_max) * sizeof(block_q8_1_mmq));
    ggml_cuda_pool_alloc<char> quantized(context.pool(), bytes);
    const auto* x = static_cast<const float*>(input->data);
    const auto s11 = static_cast<std::int64_t>(input->nb[1] / sizeof(float));
    const auto s12 = static_cast<std::int64_t>(input->nb[2] / sizeof(float));
    const auto s13 = static_cast<std::int64_t>(input->nb[3] / sizeof(float));
    if (prequantized != nullptr) {
      // The pair's activation write-back already holds this D2S6 input at
      // the same sorted columns (the same maps of the same routes).
    } else if (broadcast) {
      quantize_scatter_mmq_q8_1_cuda(x, ids_src.get(), quantized.get(), type, input->ne[0], s12,
                                     padded, tokens, rows, static_cast<int>(used), stream);
    } else {
      quantize_mmq_q8_1_cuda(x, ids_src.get(), quantized.get(), type, input->ne[0], s11, s12, s13,
                             padded, rows, 1, 1, stream);
    }
    CUDA_CHECK(cudaGetLastError());
    const auto sy2 = input->ne[1] * padded * static_cast<std::int64_t>(sizeof(block_q8_1)) /
                     (QK8_1 * static_cast<std::int64_t>(sizeof(int)));
    const auto sy3 = tokens * sy2;
    const auto ncols_opt = GGML_CUDA_CC_IS_RDNA3_0(cc) || GGML_CUDA_CC_IS_RDNA4(cc)
                               ? (rows + weights->ne[2] - 1) / weights->ne[2]
                               : tokens;
    // With the activation write-back, gate (second) is written first and
    // the up product (first) stores swiglu_clamp(gate, up) into `glu`.
    const std::array<ggml_tensor*, 2> order = glu != nullptr
                                                  ? std::array<ggml_tensor*, 2>{second, first}
                                                  : std::array<ggml_tensor*, 2>{first, second};
    for (ggml_tensor* output : order) {
      if (output == nullptr) {
        continue;
      }
      const bool activation = glu != nullptr && output == first;
      float* const destination = static_cast<float*>(activation ? glu->data : output->data);
      const ggml_tensor* w = output->src[0];
      const auto ts = static_cast<std::int64_t>(ggml_type_size(type));
      const mmq_args args = {static_cast<const char*>(w->data),
                             type,
                             reinterpret_cast<const int*>(
                                 prequantized != nullptr ? prequantized->data : quantized.get()),
                             ids_dst.get(),
                             bounds.get(),
                             destination,
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
          if (activation) {
            CUDA_CHECK(LaunchIq2PairGluUp(context, args, static_cast<const float*>(second->data),
                                          MulMatIdQPairGluLimit(glu), glu_q8 ? glu->data : nullptr,
                                          stream));
          } else if (iq2_occ2) {
            CUDA_CHECK(ggml_cuda_mul_mat_iq2_occ2_pair_product(context, args, stream));
          } else {
            mul_mat_q_case<GGML_TYPE_IQ2_XXS>(context, args, stream);
          }
          break;
        case GGML_TYPE_IQ2_XS:
          if (activation) {
            CUDA_CHECK(LaunchIq2PairGluUp(context, args, static_cast<const float*>(second->data),
                                          MulMatIdQPairGluLimit(glu), glu_q8 ? glu->data : nullptr,
                                          stream));
          } else {
            mul_mat_q_case<GGML_TYPE_IQ2_XS>(context, args, stream);
          }
          break;
        case GGML_TYPE_IQ3_XXS:
          mul_mat_q_case<GGML_TYPE_IQ3_XXS>(context, args, stream);
          break;
        // The other types Qwen3.8's GGUF quantizations mix.
        case GGML_TYPE_Q4_1:
          mul_mat_q_case<GGML_TYPE_Q4_1>(context, args, stream);
          break;
        case GGML_TYPE_Q5_0:
          mul_mat_q_case<GGML_TYPE_Q5_0>(context, args, stream);
          break;
        case GGML_TYPE_Q5_1:
          mul_mat_q_case<GGML_TYPE_Q5_1>(context, args, stream);
          break;
        case GGML_TYPE_Q4_0:
          mul_mat_q_case<GGML_TYPE_Q4_0>(context, args, stream);
          break;
        case GGML_TYPE_Q2_0:
          mul_mat_q_case<GGML_TYPE_Q2_0>(context, args, stream);
          break;
        case GGML_TYPE_Q3_K:
          mul_mat_q_case<GGML_TYPE_Q3_K>(context, args, stream);
          break;
        case GGML_TYPE_IQ1_S:
          mul_mat_q_case<GGML_TYPE_IQ1_S>(context, args, stream);
          break;
        case GGML_TYPE_IQ2_S:
          mul_mat_q_case<GGML_TYPE_IQ2_S>(context, args, stream);
          break;
        case GGML_TYPE_IQ3_S:
          mul_mat_q_case<GGML_TYPE_IQ3_S>(context, args, stream);
          break;
        case GGML_TYPE_IQ4_NL:
          mul_mat_q_case<GGML_TYPE_IQ4_NL>(context, args, stream);
          break;
        case GGML_TYPE_IQ4_XS:
          mul_mat_q_case<GGML_TYPE_IQ4_XS>(context, args, stream);
          break;
        default:
          GGML_ABORT("paired MMQ type passed validation without its case");
      }
    }
  });
}

}  // namespace

std::expected<void, KernelFailure> MulMatIdQPair(LaunchContext& launch, ggml_tensor* first,
                                                 ggml_tensor* second, bool compact_experts) {
  return RunExpertProducts(launch, first, second, compact_experts);
}

std::expected<void, KernelFailure> MulMatIdQPairGlu(LaunchContext& launch, ggml_tensor* up,
                                                    ggml_tensor* gate, ggml_tensor* glu) {
  if (auto checked = CheckMulMatIdQPairGlu(up, gate, glu); !checked) {
    return checked;
  }
  return RunExpertProducts(launch, up, gate, /*compact_experts=*/true, glu);
}

bool MulMatIdQPairGluSupported(const LaunchContext& launch, const ggml_tensor* up,
                               const ggml_tensor* gate) {
  return GluPair(launch, up, gate, /*compact_experts=*/true);
}

std::expected<void, KernelFailure> MulMatIdQCompact(LaunchContext& launch, ggml_tensor* node) {
  return RunExpertProducts(launch, node, nullptr, true);
}

std::expected<std::uint64_t, KernelFailure> PlanMulMatQPairDense(const LaunchContext& launch,
                                                                 const ggml_tensor* a,
                                                                 const ggml_tensor* b) {
  if (auto checked = CheckMulMatQPairDense(a, b); !checked) {
    return std::unexpected(checked.error());
  }
  auto pa = PlanMulMatQ(launch, a);
  if (!pa) {
    return std::unexpected(pa.error());
  }
  auto pb = PlanMulMatQ(launch, b);
  if (!pb) {
    return std::unexpected(pb.error());
  }
  // One activation buffer and each product's stream-k fixup, generously.
  return *pa + *pb;
}

std::expected<void, KernelFailure> MulMatQPairDense(LaunchContext& launch, ggml_tensor* a,
                                                    ggml_tensor* b) {
  auto scratch = PlanMulMatQPairDense(launch, a, b);
  if (!scratch) {
    return std::unexpected(scratch.error());
  }
  return launch.Run(base::Bytes(*scratch), [a, b](ggml_backend_cuda_context& context) {
    // ggml_cuda_mul_mat_q's dense branch (mmq.cu), quantizing the shared
    // activation once for both products of the one type (whose rows are
    // whole 128-row tiles: no fallback configuration).
    const ggml_tensor* src1 = a->src[1];
    const ggml_type type = a->src[0]->type;
    cudaStream_t stream = context.stream();
    const int cc = ggml_cuda_info().devices[context.device].cc;
    const std::int64_t ne10 = src1->ne[0];
    const std::int64_t ne11 = src1->ne[1];
    const std::int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);
    const int j_max = ggml_cuda_mmq_get_J_max(type, false, cc, ne11);
    const auto nbytes =
        (static_cast<std::size_t>(ne11 * ne10_padded) * sizeof(block_q8_1_mmq) / QK8_1_MMQ) +
        (static_cast<std::size_t>(j_max) * sizeof(block_q8_1_mmq));
    ggml_cuda_pool_alloc<char> q8(context.pool(), nbytes);
    const auto s11 = static_cast<std::int64_t>(src1->nb[1] / sizeof(float));
    const auto s12 = static_cast<std::int64_t>(src1->nb[2] / sizeof(float));
    const auto s13 = static_cast<std::int64_t>(src1->nb[3] / sizeof(float));
    quantize_mmq_q8_1_cuda(static_cast<const float*>(src1->data), nullptr, q8.get(), type, ne10,
                           s11, s12, s13, ne10_padded, ne11, 1, 1, stream);
    CUDA_CHECK(cudaGetLastError());
    const std::int64_t y12 = ne11 * ne10_padded * static_cast<std::int64_t>(sizeof(block_q8_1)) /
                             (QK8_1 * static_cast<std::int64_t>(sizeof(int)));
    for (ggml_tensor* node : {a, b}) {
      const ggml_tensor* src0 = node->src[0];
      const auto ts0 = static_cast<std::int64_t>(ggml_type_size(src0->type));
      const mmq_args args = {static_cast<const char*>(src0->data),
                             src0->type,
                             reinterpret_cast<const int*>(q8.get()),
                             nullptr,
                             nullptr,
                             static_cast<float*>(node->data),
                             nullptr,
                             src0->ne[0],
                             src0->ne[1],
                             node->ne[1],
                             static_cast<std::int64_t>(src0->nb[1]) / ts0,
                             ne11,
                             static_cast<std::int64_t>(node->nb[1] / sizeof(float)),
                             src0->ne[2],
                             1,
                             static_cast<std::int64_t>(src0->nb[2]) / ts0,
                             y12,
                             static_cast<std::int64_t>(node->nb[2] / sizeof(float)),
                             src0->ne[3],
                             1,
                             static_cast<std::int64_t>(src0->nb[3]) / ts0,
                             y12,
                             static_cast<std::int64_t>(node->nb[3] / sizeof(float)),
                             node->ne[1],
                             node->ne[1]};
      MmqCase(type, context, args, stream);
    }
  });
}

std::expected<void, KernelFailure> MulMatIdQPairGluQ8(LaunchContext& launch, ggml_tensor* up,
                                                      ggml_tensor* gate, ggml_tensor* glu) {
  if (auto checked = CheckMulMatIdQPairGlu(up, gate, glu); !checked) {
    return checked;
  }
  // Each sorted column's D2S6 blocks (one per 128 values) and the down
  // product's J guard (at most 128 columns) within the F32 activation's
  // bytes: the chunk's own columns, not a full 4,096-token chunk's.
  const auto columns = static_cast<std::size_t>(glu->ne[1] * glu->ne[2]);
  const auto blocks = static_cast<std::size_t>(glu->ne[0] / QK8_1_MMQ);
  if (ggml_nbytes(glu) < ((columns * blocks) + 128) * sizeof(block_q8_1_mmq)) {
    return Rejected("the pair activation cannot hold its quantized form");
  }
  return RunExpertProducts(launch, up, gate, /*compact_experts=*/true, glu, true);
}

std::expected<void, KernelFailure> MulMatIdQCompactPrequant(LaunchContext& launch,
                                                            ggml_tensor* down) {
  if (auto checked = CheckMulMatIdQCompactPrequant(down); !checked) {
    return checked;
  }
  if (mmq_get_q8_1_ds_layout(down->src[0]->type) != MMQ_Q8_1_DS_LAYOUT_D2S6) {
    return Rejected("the prequantized down input is D2S6");
  }
  return RunExpertProducts(launch, down, nullptr, true, nullptr, false, down->src[1]);
}

}  // namespace llmp::kernels::ggml
