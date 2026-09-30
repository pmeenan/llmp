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
#include <climits>
#include <cstdint>
#include <expected>
#include <initializer_list>
#include <string>
#include <utility>

#include "base/bytes.h"
#include "common.cuh"
#include "kernels/ggml/ops_ext.h"
#include "kernels/ggml/validate_ext.h"
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
                                                      std::int64_t columns, std::int64_t planes) {
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

std::expected<std::uint64_t, KernelFailure> PlanMulMatQ(const LaunchContext& launch,
                                                        const ggml_tensor* node) {
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
  const std::uint64_t j_max = static_cast<std::uint64_t>(
      ggml_cuda_mmq_get_J_max(weights->type, fallback, cc, input->ne[1]));
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
      draws.Add(columns * sizeof(float));
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
  draws.Add(rows * sizeof(std::int32_t));
  draws.Add(static_cast<std::uint64_t>(weights->ne[2] + 1) * sizeof(std::int32_t));
  draws.Add((rows * padded * y_block / y_values) + (j_max * sizeof(block_q8_1_mmq)));
  if (row_scales) {
    draws.Add(rows * sizeof(float));
  }
  // The tile grid spans every token for each expert (ncols_max: tokens).
  auto fixup = TileFixup(launch, weights->type, weights->ne[1], weights->ne[0], input->ne[2],
                         weights->ne[2]);
  if (!fixup) {
    return std::unexpected(fixup.error());
  }
  draws.Add(*fixup);
  return draws.total();
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

std::expected<std::uint64_t, KernelFailure> PlanMulMatIdQPair(const LaunchContext& launch,
                                                              const ggml_tensor* first,
                                                              const ggml_tensor* second) {
  if (auto checked = CheckMulMatIdQPair(first, second); !checked) {
    return std::unexpected(checked.error());
  }
  auto a = PlanMulMatQ(launch, first);
  if (!a) {
    return std::unexpected(a.error());
  }
  auto b = PlanMulMatQ(launch, second);
  if (!b) {
    return std::unexpected(b.error());
  }
  // Maps and quantization match; each ordinary product returns its fixup
  // allocation before the next draws it. The high-water mark is unchanged.
  return std::max(*a, *b);
}

std::expected<void, KernelFailure> MulMatIdQPair(LaunchContext& launch, ggml_tensor* first,
                                                 ggml_tensor* second) {
  auto scratch = PlanMulMatIdQPair(launch, first, second);
  if (!scratch) {
    return std::unexpected(scratch.error());
  }
  return launch.Run(base::Bytes(*scratch), [first, second](ggml_backend_cuda_context& context) {
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
    cudaStream_t stream = context.stream();
    ggml_cuda_pool_alloc<std::int32_t> ids_src(context.pool(), static_cast<std::size_t>(rows));
    ggml_cuda_pool_alloc<std::int32_t> ids_dst(context.pool(), static_cast<std::size_t>(rows));
    ggml_cuda_pool_alloc<std::int32_t> bounds(context.pool(),
                                              static_cast<std::size_t>(weights->ne[2] + 1));
    const bool broadcast = input->ne[1] == 1 && used > 1;
    ggml_cuda_launch_mm_ids_helper(
        static_cast<const std::int32_t*>(ids->data), ids_src.get(), ids_dst.get(), bounds.get(),
        static_cast<int>(weights->ne[2]), static_cast<int>(tokens), static_cast<int>(used),
        static_cast<int>(input->ne[1]), static_cast<int>(ids->nb[1] / sizeof(std::int32_t)),
        static_cast<int>(input->nb[2] / input->nb[1]), broadcast, stream);
    CUDA_CHECK(cudaGetLastError());
    const int cc = ggml_cuda_info().devices[context.device].cc;
    const auto j_max = ggml_cuda_mmq_get_J_max(type, weights->ne[1] % 128 != 0, cc, input->ne[1]);
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
    for (ggml_tensor* output : {first, second}) {
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
                             ncols_opt};
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

}  // namespace jitllm::kernels::ggml
