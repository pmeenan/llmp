// SPDX-FileCopyrightText: 2023-2026 The ggml authors
// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: MIT AND Apache-2.0

// The elementwise, row, routing, recurrent and DeepSeek V4 operations of
// ops_ext.h over GGML's launchers. PlanTopK is a recorded copy of what
// ggml_cuda_op_top_k draws from the pool in jitLLM's build, which has no
// CUB (third_party/patches/ggml/0001): top_k_radix_cuda's state and
// histograms for rows over 1,024, else the bitonic argsort's full indices
// (top-k.cu:175-181 and 262-268 at llama.cpp b29c606e2).

#include <algorithm>
#include <cstdint>
#include <expected>
#include <string>
#include <utility>

#include "argsort.cuh"
#include "base/bytes.h"
#include "binbcast.cuh"
#include "clamp.cuh"
#include "common.cuh"
#include "concat.cuh"
#include "dsv4-hc.cuh"
#include "fill.cuh"
#include "fwht.cuh"
#include "gated_delta_net.cuh"
#include "getrows.cuh"
#include "kernels/ggml/ops_ext.h"
#include "kernels/ggml/validate.h"
#include "kernels/ggml/validate_ext.h"
#include "lightning-indexer.cuh"
#include "rope.cuh"
#include "scale.cuh"
#include "set-rows.cuh"
#include "ssm-conv.cuh"
#include "sumrows.cuh"
#include "top-k.cuh"
#include "unary.cuh"

// PlanTopK sizes the paths of a build without CUB (patches/ggml/0001).
#if defined(GGML_CUDA_USE_CUB)
#error "jitLLM's GGML build takes no CUB; PlanTopK sizes the radix and bitonic paths"
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

using Launcher = void (*)(ggml_backend_cuda_context&, ggml_tensor*);

// The upstream launcher for a node CheckUnary accepts.
Launcher UnaryLauncher(const ggml_tensor* node) {
  if (node->op == GGML_OP_SQRT) {
    return &ggml_cuda_op_sqrt;
  }
  switch (ggml_get_unary_op(node)) {
    case GGML_UNARY_OP_ABS:
      return &ggml_cuda_op_abs;
    case GGML_UNARY_OP_SGN:
      return &ggml_cuda_op_sgn;
    case GGML_UNARY_OP_NEG:
      return &ggml_cuda_op_neg;
    case GGML_UNARY_OP_SILU:
      return &ggml_cuda_op_silu;
    case GGML_UNARY_OP_GELU:
      return &ggml_cuda_op_gelu;
    case GGML_UNARY_OP_TANH:
      return &ggml_cuda_op_tanh;
    case GGML_UNARY_OP_RELU:
      return &ggml_cuda_op_relu;
    case GGML_UNARY_OP_SIGMOID:
      return &ggml_cuda_op_sigmoid;
    case GGML_UNARY_OP_EXP:
      return &ggml_cuda_op_exp;
    case GGML_UNARY_OP_SOFTPLUS:
      return &ggml_cuda_op_softplus;
    default:
      return nullptr;
  }
}

// Checks, then runs one scratch-free launcher over the node.
std::expected<void, KernelFailure> RunChecked(LaunchContext& launch, ggml_tensor* node,
                                              std::expected<void, KernelFailure> checked,
                                              Launcher launcher) {
  if (!checked) {
    return checked;
  }
  return launch.Run(base::Bytes(0), [node, launcher](ggml_backend_cuda_context& context) {
    launcher(context, node);
  });
}

}  // namespace

std::expected<void, KernelFailure> MulMatHadamard(LaunchContext& launch, ggml_tensor* node) {
  if (auto checked = CheckMulMatHadamard(node); !checked) {
    return checked;
  }
  return launch.Run(base::Bytes(0), [node](ggml_backend_cuda_context& context) {
    // CheckMulMatHadamard admits only what the transform takes, so it
    // always reports that it ran (fwht.cu:61-101).
    const bool ran = ggml_cuda_op_fwht(context, node->src[1], node);
    if (!ran) {
      ggml_cuda_error("ggml_cuda_op_fwht", __func__, __FILE__, __LINE__,
                      "the Hadamard transform refused a node its check admitted");
    }
  });
}

std::expected<void, KernelFailure> Unary(LaunchContext& launch, ggml_tensor* node) {
  auto checked = CheckUnary(node);
  return RunChecked(launch, node, checked, checked ? UnaryLauncher(node) : nullptr);
}

std::expected<void, KernelFailure> Scale(LaunchContext& launch, ggml_tensor* node) {
  return RunChecked(launch, node, CheckScale(node), &ggml_cuda_op_scale);
}

std::expected<void, KernelFailure> Clamp(LaunchContext& launch, ggml_tensor* node) {
  return RunChecked(launch, node, CheckClamp(node), &ggml_cuda_op_clamp);
}

std::expected<void, KernelFailure> Fill(LaunchContext& launch, ggml_tensor* node) {
  return RunChecked(launch, node, CheckFill(node), &ggml_cuda_op_fill);
}

std::expected<void, KernelFailure> Repeat(LaunchContext& launch, ggml_tensor* node) {
  return RunChecked(launch, node, CheckRepeat(node), &ggml_cuda_op_repeat);
}

std::expected<void, KernelFailure> Sub(LaunchContext& launch, ggml_tensor* node) {
  return RunChecked(launch, node, CheckBinary(node, GGML_OP_SUB), &ggml_cuda_op_sub);
}

std::expected<void, KernelFailure> Div(LaunchContext& launch, ggml_tensor* node) {
  return RunChecked(launch, node, CheckBinary(node, GGML_OP_DIV), &ggml_cuda_op_div);
}

std::expected<void, KernelFailure> Concat(LaunchContext& launch, ggml_tensor* node) {
  return RunChecked(launch, node, CheckConcat(node), &ggml_cuda_op_concat);
}

std::expected<void, KernelFailure> SumRows(LaunchContext& launch, ggml_tensor* node) {
  return RunChecked(launch, node, CheckSumRows(node), &ggml_cuda_op_sum_rows);
}

std::expected<void, KernelFailure> SwiGluClamp(LaunchContext& launch, ggml_tensor* node) {
  return RunChecked(launch, node, CheckSwiGluClamp(node), &ggml_cuda_op_swiglu_clamp);
}

std::expected<void, KernelFailure> RopeExt(LaunchContext& launch, ggml_tensor* node) {
  auto checked = CheckRopeExt(node);
  return RunChecked(
      launch, node, checked,
      checked && node->op == GGML_OP_ROPE_BACK ? &ggml_cuda_op_rope_back : &ggml_cuda_op_rope);
}

std::expected<void, KernelFailure> GetRowsExt(LaunchContext& launch, ggml_tensor* node) {
  return RunChecked(launch, node, CheckGetRowsExt(node), &ggml_cuda_op_get_rows);
}

std::expected<void, KernelFailure> SetRowsExt(LaunchContext& launch, ggml_tensor* node) {
  return RunChecked(launch, node, CheckSetRowsExt(node), &ggml_cuda_op_set_rows);
}

std::expected<void, KernelFailure> Argsort(LaunchContext& launch, ggml_tensor* node) {
  if (auto checked = CheckArgsort(node); !checked) {
    return checked;
  }
  // Upstream sorts in CUB beyond the shared memory the bitonic kernel may
  // use (argsort.cu:272-278); jitLLM's build has no CUB, and the bitonic
  // launcher would assert.
  if (ArgsortSharedBytes(node) > Device(launch).smpb) {
    return Rejected("an argsort row beyond the device's shared memory");
  }
  return launch.Run(base::Bytes(0), [node](ggml_backend_cuda_context& context) {
    ggml_cuda_op_argsort(context, node);
  });
}

std::expected<std::uint64_t, KernelFailure> PlanTopK(const LaunchContext& launch,
                                                     const ggml_tensor* node) {
  if (auto checked = CheckTopK(node); !checked) {
    return std::unexpected(checked.error());
  }
  const auto columns = static_cast<std::uint64_t>(node->src[0]->ne[0]);
  const auto rows = static_cast<std::uint64_t>(ggml_nrows(node->src[0]));
  if (columns <= 1024) {
    // The bitonic argsort of every row into scratch, then the first k of
    // each copied out; its padded row must fit shared memory
    // (argsort.cu:233-239).
    std::uint64_t padded = 1;
    while (padded < columns) {
      padded *= 2;
    }
    if (padded * sizeof(int) > Device(launch).smpb) {
      return Rejected("a top_k row beyond the device's shared memory");
    }
    // The bitonic kernel offsets each row in an int (argsort.cu:166).
    if (columns * rows > static_cast<std::uint64_t>(INT32_MAX)) {
      return Rejected("top_k rows beyond the bitonic kernel's 32-bit indexing");
    }
    return columns * rows * sizeof(int);
  }
  // top_k_radix_cuda: a state per row (top_k_radix_state: two uint32 and
  // three int), then 256 bins per row block, both held at once; one block
  // per row block, at most 64 per row.
  constexpr std::uint64_t kState = 5 * sizeof(std::int32_t);
  constexpr std::uint64_t kBins = 256;
  const std::uint64_t blocks_per_row = std::min<std::uint64_t>((columns + 1023) / 1024, 64);
  if (rows * blocks_per_row > static_cast<std::uint64_t>(INT32_MAX)) {
    return Rejected("top_k beyond the radix select's grid");
  }
  const std::uint64_t states = rows * kState;
  return ((states + 255) / 256 * 256) + (rows * blocks_per_row * kBins * sizeof(int));
}

std::expected<void, KernelFailure> TopK(LaunchContext& launch, ggml_tensor* node) {
  auto scratch = PlanTopK(launch, node);
  if (!scratch) {
    return std::unexpected(scratch.error());
  }
  return launch.Run(base::Bytes(*scratch), [node](ggml_backend_cuda_context& context) {
    ggml_cuda_op_top_k(context, node);
  });
}

std::expected<void, KernelFailure> SsmConv(LaunchContext& launch, ggml_tensor* node) {
  if (auto checked = CheckSsmConv(node); !checked) {
    return checked;
  }
  return launch.Run(base::Bytes(0), [node](ggml_backend_cuda_context& context) {
    ggml_cuda_op_ssm_conv(context, node);
  });
}

std::expected<void, KernelFailure> GatedDeltaNet(LaunchContext& launch, ggml_tensor* node) {
  return RunChecked(launch, node, CheckGatedDeltaNet(node), &ggml_cuda_op_gated_delta_net);
}

std::expected<void, KernelFailure> LightningIndexer(LaunchContext& launch, ggml_tensor* node) {
  if (auto checked = CheckLightningIndexer(node); !checked) {
    return checked;
  }
  // The launcher's tensor-core kernel (lightning-indexer.cu:447-465), which
  // needs Turing's matrix instructions; elsewhere it would take the vector
  // kernel, which is the same computation but not this implementation.
  if (!GGML_CUDA_CC_IS_NVIDIA(Device(launch).cc) || !turing_mma_available(Device(launch).cc)) {
    return Rejected("the lightning indexer's tensor-core kernel needs Turing or later");
  }
  return launch.Run(base::Bytes(0), [node](ggml_backend_cuda_context& context) {
    ggml_cuda_lightning_indexer(context, node);
  });
}

std::expected<void, KernelFailure> HcComb(LaunchContext& launch, ggml_tensor* node) {
  return RunChecked(launch, node, CheckHcComb(node), &ggml_cuda_op_dsv4_hc_comb);
}

std::expected<void, KernelFailure> HcPre(LaunchContext& launch, ggml_tensor* node) {
  return RunChecked(launch, node, CheckHcPre(node), &ggml_cuda_op_dsv4_hc_pre);
}

std::expected<void, KernelFailure> HcPost(LaunchContext& launch, ggml_tensor* node) {
  return RunChecked(launch, node, CheckHcPost(node), &ggml_cuda_op_dsv4_hc_post);
}

}  // namespace jitllm::kernels::ggml
