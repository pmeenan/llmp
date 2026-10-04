// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <cstring>
#include <expected>
#include <string>
#include <utility>

#include "base/bytes.h"
#include "binbcast.cuh"
#include "common.cuh"
#include "cpy.cuh"
#include "getrows.cuh"
#include "kernels/ggml/ops.h"
#include "kernels/ggml/validate.h"
#include "mmf.cuh"
#include "mmvf.cuh"
#include "norm.cuh"
#include "rope.cuh"
#include "set-rows.cuh"
#include "softmax.cuh"
#include "unary.cuh"

namespace jitllm::kernels::ggml {
namespace {

std::unexpected<KernelFailure> Rejected(std::string detail) {
  return std::unexpected(
      KernelFailure{.error = KernelError::kRejected, .detail = std::move(detail)});
}

// The launch context's device, read without a CUDA call: its table was
// read when the context was created.
const ggml_cuda_device_info::cuda_device_info& Device(const LaunchContext& launch) {
  return ggml_cuda_info().devices[launch.device()];
}

}  // namespace

std::expected<void, KernelFailure> RmsNorm(LaunchContext& launch, ggml_tensor* norm) {
  if (auto checked = CheckRmsNorm(norm); !checked) {
    return checked;
  }
  return launch.Run(base::Bytes(0), [norm](ggml_backend_cuda_context& context) {
    ggml_cuda_op_rms_norm(context, norm);
  });
}

std::expected<void, KernelFailure> RmsNormMul(LaunchContext& launch, ggml_tensor* norm,
                                              ggml_tensor* mul) {
  if (auto checked = CheckRmsNormMul(norm, mul); !checked) {
    return checked;
  }
  return launch.Run(base::Bytes(0), [norm, mul](ggml_backend_cuda_context& context) {
    ggml_cuda_op_rms_norm_fused(context, norm, mul);
  });
}

std::expected<void, KernelFailure> RmsNormThenMul(LaunchContext& launch, ggml_tensor* norm,
                                                  ggml_tensor* mul) {
  if (auto checked = CheckRmsNormThenMul(norm, mul); !checked) {
    return checked;
  }
  return launch.Run(base::Bytes(0), [norm, mul](ggml_backend_cuda_context& context) {
    ggml_cuda_op_rms_norm(context, norm);
    ggml_cuda_op_mul(context, mul);
  });
}

std::expected<void, KernelFailure> Add(LaunchContext& launch, ggml_tensor* node) {
  if (auto checked = CheckBinary(node, GGML_OP_ADD); !checked) {
    return checked;
  }
  return launch.Run(base::Bytes(0), [node](ggml_backend_cuda_context& context) {
    ggml_cuda_op_add(context, node);
  });
}

std::expected<void, KernelFailure> Mul(LaunchContext& launch, ggml_tensor* node) {
  if (auto checked = CheckBinary(node, GGML_OP_MUL); !checked) {
    return checked;
  }
  return launch.Run(base::Bytes(0), [node](ggml_backend_cuda_context& context) {
    ggml_cuda_op_mul(context, node);
  });
}

std::expected<void, KernelFailure> MulMatVecF(LaunchContext& launch, ggml_tensor* node) {
  if (auto checked = CheckMulMat(node); !checked) {
    return checked;
  }
  const ggml_tensor* weights = node->src[0];
  if (!ggml_cuda_should_use_mmvf(weights->type, Device(launch).cc, weights->ne, weights->nb,
                                 node->src[1]->ne[1])) {
    return Rejected("upstream does not select MMVF for these operands");
  }
  // The launcher's even column-stride assertion holds: CheckMulMat requires
  // every activation stride to be a multiple of 8 bytes.
  return launch.Run(base::Bytes(0), [node](ggml_backend_cuda_context& context) {
    ggml_cuda_mul_mat_vec_f(context, node->src[0], node->src[1], nullptr, node);
  });
}

std::expected<void, KernelFailure> MulMatF(LaunchContext& launch, ggml_tensor* node) {
  if (auto checked = CheckMulMatF(node); !checked) {
    return checked;
  }
  const ggml_tensor* weights = node->src[0];
  const auto& device = Device(launch);
  if (!ggml_cuda_should_use_mmf(weights->type, device.cc, device.warp_size, weights->ne,
                                weights->nb, static_cast<int>(node->src[1]->ne[1]),
                                /*mul_mat_id=*/false)) {
    return Rejected("upstream does not select MMF for these operands");
  }
  return launch.Run(base::Bytes(0), [node](ggml_backend_cuda_context& context) {
    ggml_cuda_mul_mat_f(context, node->src[0], node->src[1], nullptr, node);
  });
}

std::expected<MulMatPath, KernelFailure> SelectMulMat(const LaunchContext& launch,
                                                      const ggml_tensor* node) {
  if (node == nullptr || node->op != GGML_OP_MUL_MAT || node->src[0] == nullptr ||
      node->src[1] == nullptr) {
    return Rejected("not a ggml_mul_mat node");
  }
  const ggml_tensor* src0 = node->src[0];
  const ggml_tensor* src1 = node->src[1];
  // Only a quantized tensor is padded in a CUDA buffer, so upstream's
  // bad_padding_clear never holds for the others; quantized weights take
  // MMVQ or MMQ, which ops_ext.h's SelectMulMatQ chooses between.
  if (ggml_is_quantized(src0->type)) {
    return Rejected("quantized weights take MMVQ or MMQ: ops_ext.h SelectMulMatQ");
  }
  if (src1->type != GGML_TYPE_F32 || node->type != GGML_TYPE_F32) {
    return MulMatPath::kCublas;
  }
  const auto& device = Device(launch);
  const std::int64_t ne11 = src1->ne[1];
  if (ggml_cuda_should_use_mmvf(src0->type, device.cc, src0->ne, src0->nb, ne11)) {
    return MulMatPath::kVector;
  }
  if (src0->ne[1] == 1 && ne11 > MMVF_MAX_BATCH_SIZE && node->ne[2] == 1 && node->ne[3] == 1 &&
      src0->type == GGML_TYPE_F32 && ggml_is_contiguous(src0) && ggml_is_contiguous(src1) &&
      ggml_is_contiguous(node) &&
      ggml_cuda_should_use_mmvf(src1->type, device.cc, src1->ne, src1->nb, /*ne11=*/1)) {
    return Rejected("upstream takes the transposed vector product, which is not implemented");
  }
  if (ggml_cuda_should_use_mmf(src0->type, device.cc, device.warp_size, src0->ne, src0->nb,
                               static_cast<int>(ne11), /*mul_mat_id=*/false)) {
    return MulMatPath::kTensorCore;
  }
  return MulMatPath::kCublas;
}

bool MulMatVecFusible(const LaunchContext& launch, const ggml_tensor* mul_mat) {
  if (mul_mat == nullptr || mul_mat->op != GGML_OP_MUL_MAT || mul_mat->src[0] == nullptr ||
      mul_mat->src[1] == nullptr) {
    return false;
  }
  const ggml_tensor* weights = mul_mat->src[0];
  const bool types = (weights->type == GGML_TYPE_F32 || weights->type == GGML_TYPE_F16 ||
                      weights->type == GGML_TYPE_BF16) &&
                     mul_mat->src[1]->type == GGML_TYPE_F32 && mul_mat->type == GGML_TYPE_F32;
  return types && mul_mat->ne[1] == 1 &&
         ggml_cuda_should_use_mmvf(weights->type, Device(launch).cc, weights->ne, weights->nb,
                                   mul_mat->src[1]->ne[1]);
}

std::expected<void, KernelFailure> GetRows(LaunchContext& launch, ggml_tensor* node) {
  if (auto checked = CheckGetRows(node); !checked) {
    return checked;
  }
  return launch.Run(base::Bytes(0), [node](ggml_backend_cuda_context& context) {
    ggml_cuda_op_get_rows(context, node);
  });
}

std::expected<void, KernelFailure> SetRows(LaunchContext& launch, ggml_tensor* node) {
  if (auto checked = CheckSetRows(node); !checked) {
    return checked;
  }
  return launch.Run(base::Bytes(0), [node](ggml_backend_cuda_context& context) {
    ggml_cuda_op_set_rows(context, node);
  });
}

std::expected<void, KernelFailure> Rope(LaunchContext& launch, ggml_tensor* rope) {
  if (auto checked = CheckRope(rope); !checked) {
    return checked;
  }
  return launch.Run(base::Bytes(0), [rope](ggml_backend_cuda_context& context) {
    ggml_cuda_op_rope(context, rope);
  });
}

std::expected<void, KernelFailure> RopeSetRows(LaunchContext& launch, ggml_tensor* rope,
                                               ggml_tensor* set_rows) {
  if (auto checked = CheckRopeSetRows(rope, set_rows); !checked) {
    return checked;
  }
  return launch.Run(base::Bytes(0), [rope, set_rows](ggml_backend_cuda_context& context) {
    ggml_cuda_op_rope_fused(context, rope, set_rows);
  });
}

std::expected<void, KernelFailure> SoftMax(LaunchContext& launch, ggml_tensor* node) {
  if (auto checked = CheckSoftMax(node); !checked) {
    return checked;
  }
  // Beyond the opt-in limit upstream takes other kernels (softmax.cu:349-375),
  // one of them drawing scratch; none is implemented here.
  if (SoftMaxSharedBytes(node) > Device(launch).smpbo) {
    return Rejected("a soft_max row larger than the device's shared memory");
  }
  return launch.Run(base::Bytes(0), [node](ggml_backend_cuda_context& context) {
    ggml_cuda_op_soft_max(context, node);
  });
}

std::expected<void, KernelFailure> Cont(LaunchContext& launch, ggml_tensor* node) {
  if (auto checked = CheckCont(node); !checked) {
    return std::unexpected(checked.error());
  }
  return launch.Run(base::Bytes(0),
                    [node](ggml_backend_cuda_context& context) { ggml_cuda_dup(context, node); });
}

std::expected<void, KernelFailure> Convert(LaunchContext& launch, ggml_tensor* node) {
  if (auto checked = CheckConvert(node); !checked) {
    return checked;
  }
  return launch.Run(base::Bytes(0), [node](ggml_backend_cuda_context& context) {
    ggml_cuda_cpy(context, node->src[0], node->src[1]);
  });
}

std::expected<void, KernelFailure> SwiGlu(LaunchContext& launch, ggml_tensor* node) {
  if (auto checked = CheckSwiGlu(node); !checked) {
    return checked;
  }
  return launch.Run(base::Bytes(0), [node](ggml_backend_cuda_context& context) {
    ggml_cuda_op_swiglu(context, node);
  });
}

std::expected<void, KernelFailure> GeGlu(LaunchContext& launch, ggml_tensor* node) {
  if (auto checked = CheckGeGlu(node); !checked) {
    return checked;
  }
  return launch.Run(base::Bytes(0), [node](ggml_backend_cuda_context& context) {
    ggml_cuda_op_geglu(context, node);
  });
}

std::expected<void, KernelFailure> MulMatVecBias(LaunchContext& launch, ggml_tensor* mul_mat,
                                                 ggml_tensor* add) {
  if (auto checked = CheckMulMatVecBias(mul_mat, add); !checked) {
    return checked;
  }
  if (!MulMatVecFusible(launch, mul_mat)) {
    return Rejected("upstream does not select MMVF for this product");
  }
  const ggml_tensor* bias = add->src[0] == mul_mat ? add->src[1] : add->src[0];
  // As ggml_cuda_try_fuse calls it (ggml-cuda.cu:4113-4118): the add is the
  // node written, and its parameters give the accumulation precision.
  return launch.Run(base::Bytes(0), [mul_mat, add, bias](ggml_backend_cuda_context& context) {
    ggml_cuda_mm_fusion_args_host fusion{};
    fusion.x_bias = bias;
    ggml_cuda_mul_mat_vec_f(context, mul_mat->src[0], mul_mat->src[1], nullptr, add, &fusion);
  });
}

std::expected<void, KernelFailure> MulMatVecGlu(LaunchContext& launch, ggml_tensor* gate,
                                                ggml_tensor* up, ggml_tensor* glu) {
  if (auto checked = CheckMulMatVecGlu(gate, up, glu); !checked) {
    return checked;
  }
  if (!MulMatVecFusible(launch, up)) {
    return Rejected("upstream does not select MMVF for the up product");
  }
  // As ggml_cuda_try_fuse calls it (ggml-cuda.cu:3950-3957): the up
  // product's operands, the gate's weights, and the GLU as the node written,
  // whose first parameter (its GLU operation) the launcher reads as the
  // precision.
  return launch.Run(base::Bytes(0), [gate, up, glu](ggml_backend_cuda_context& context) {
    ggml_cuda_mm_fusion_args_host fusion{};
    fusion.gate = gate->src[0];
    fusion.glu_op = ggml_get_glu_op(glu);
    float limit = 0.0f;
    std::memcpy(&limit, &glu->op_params[3], sizeof(limit));
    fusion.glu_limit = limit;
    ggml_cuda_mul_mat_vec_f(context, up->src[0], up->src[1], nullptr, glu, &fusion);
  });
}

std::expected<void, KernelFailure> MulMatVecGeGlu(LaunchContext& launch, ggml_tensor* gate,
                                                  ggml_tensor* up, ggml_tensor* glu) {
  if (auto checked = CheckMulMatVecGeGlu(gate, up, glu); !checked) {
    return checked;
  }
  if (!MulMatVecFusible(launch, up)) {
    return Rejected("upstream does not select MMVF for the up product");
  }
  // As ggml_cuda_try_fuse calls it (ggml-cuda.cu:3950-3957): the up
  // product's operands, the gate's weights, and the GLU as the node written,
  // whose first parameter (its GLU operation) the launcher reads as the
  // precision.
  return launch.Run(base::Bytes(0), [gate, up, glu](ggml_backend_cuda_context& context) {
    ggml_cuda_mm_fusion_args_host fusion{};
    fusion.gate = gate->src[0];
    fusion.glu_op = ggml_get_glu_op(glu);
    float limit = 0.0f;
    std::memcpy(&limit, &glu->op_params[3], sizeof(limit));
    fusion.glu_limit = limit;
    ggml_cuda_mul_mat_vec_f(context, up->src[0], up->src[1], nullptr, glu, &fusion);
  });
}

}  // namespace jitllm::kernels::ggml
