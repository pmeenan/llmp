// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// GGML's ordinary MMVQ on a dense product, for vecq_bench's comparison:
// the pinned mmvq.cu dispatcher, over the same Q8_1 activations, with the
// immutable-weights prefetch llmp.mmvq.prepared uses; and the pinned
// SwiGLU (unary.cu) a gated product's decomposition adds.
#include "common.cuh"
#include "mmvq.cuh"
#include "unary.cuh"
#include "vecq_upstream.h"

namespace llmp::benchmarks::vecq {

bool UpstreamMmvq(ggml_type type, const void* w, const void* q8, float* dst, int k, int n,
                  int columns, cudaStream_t stream) {
  if (columns < 1 || columns > MMVQ_MAX_BATCH_SIZE || k % QK8_1 != 0) return false;
  ggml_tensor src0{};
  ggml_tensor src1{};
  ggml_tensor out{};
  src0.type = type;
  src0.ne[0] = k;
  src0.ne[1] = n;
  src0.ne[2] = src0.ne[3] = 1;
  src0.nb[0] = ggml_type_size(type);
  src0.nb[1] = ggml_row_size(type, k);
  src0.nb[2] = src0.nb[3] = src0.nb[1] * static_cast<std::size_t>(n);
  src1.type = GGML_TYPE_F32;
  src1.ne[0] = k;
  src1.ne[1] = columns;
  src1.ne[2] = src1.ne[3] = 1;
  src1.nb[0] = sizeof(float);
  src1.nb[1] = sizeof(float) * static_cast<std::size_t>(k);
  src1.nb[2] = src1.nb[3] = src1.nb[1] * static_cast<std::size_t>(columns);
  out.type = GGML_TYPE_F32;
  out.ne[0] = n;
  out.ne[1] = columns;
  out.ne[2] = out.ne[3] = 1;
  out.nb[0] = sizeof(float);
  out.nb[1] = sizeof(float) * static_cast<std::size_t>(n);
  out.nb[2] = out.nb[3] = out.nb[1] * static_cast<std::size_t>(columns);
  const std::int64_t padded = GGML_PAD(k, MATRIX_ROW_PADDING);
  ggml_cuda_op_mul_mat_vec_q_dispatch(
      ggml_cuda_get_device(), &src0, &src1, &out, static_cast<const char*>(w), nullptr,
      static_cast<const char*>(q8), dst, 0, n, columns, n, padded, stream, true, true, nullptr);
  return cudaGetLastError() == cudaSuccess;
}

bool UpstreamSwiGlu(const float* gate, const float* up, float* dst, int n, int columns,
                    cudaStream_t stream) {
  if (n < 1 || columns < 1) return false;
  const auto matrix = [&](ggml_tensor& t, const float* data) {
    t.type = GGML_TYPE_F32;
    t.ne[0] = n;
    t.ne[1] = columns;
    t.ne[2] = t.ne[3] = 1;
    t.nb[0] = sizeof(float);
    t.nb[1] = sizeof(float) * static_cast<std::size_t>(n);
    t.nb[2] = t.nb[3] = t.nb[1] * static_cast<std::size_t>(columns);
    t.data = const_cast<float*>(data);
  };
  ggml_tensor g{};
  ggml_tensor u{};
  ggml_tensor out{};
  matrix(g, gate);
  matrix(u, up);
  matrix(out, dst);
  out.op = GGML_OP_GLU;
  out.src[0] = &g;
  out.src[1] = &u;
  out.op_params[0] = GGML_GLU_OP_SWIGLU;
  out.op_params[1] = 0;  // not swapped
  // The stream lent for the launch and taken back, as the launch context
  // does (kernels/ggml/launch.cu): GGML creates nothing of its own.
  static ggml_backend_cuda_context context(ggml_cuda_get_device());
  context.streams[context.device][0] = stream;
  ggml_cuda_op_swiglu(context, &out);
  context.streams[context.device][0] = nullptr;
  return cudaGetLastError() == cudaSuccess;
}

}  // namespace llmp::benchmarks::vecq
