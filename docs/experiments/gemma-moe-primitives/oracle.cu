// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// External diagnostic only: directly call the pinned image's original math.
// Compile against exact b29 headers; never link into jitLLM production.
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

#include "ggml-backend-impl.h"
#include "ggml-cuda.h"
#include "moe-weighted-reduction.cuh"
#include "topk-moe.cuh"

static void Check(cudaError_t status) {
  if (status != cudaSuccess) {
    std::fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(status));
    std::abort();
  }
}
template <class T>
static std::vector<T> Read(const std::filesystem::path& p, std::size_t n) {
  if (std::filesystem::file_size(p) != n * sizeof(T)) std::abort();
  std::vector<T> v(n);
  std::ifstream f(p, std::ios::binary);
  f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(n * sizeof(T)));
  if (!f) std::abort();
  return v;
}
template <class T>
static void Write(const std::filesystem::path& p, const std::vector<T>& v) {
  if (std::filesystem::exists(p)) std::abort();
  std::ofstream f(p, std::ios::binary);
  f.write(reinterpret_cast<const char*>(v.data()),
          static_cast<std::streamsize>(v.size() * sizeof(T)));
  if (!f) std::abort();
}
template <class T>
static void Place(ggml_tensor* t, const std::vector<T>& v) {
  if (ggml_nbytes(t) != v.size() * sizeof(T)) std::abort();
  Check(cudaMalloc(&t->data, ggml_nbytes(t)));
  Check(cudaMemcpy(t->data, v.data(), ggml_nbytes(t), cudaMemcpyHostToDevice));
}
int main(int argc, char** argv) {
  if (argc != 3 || !std::filesystem::is_directory(argv[1]) ||
      !std::filesystem::create_directory(argv[2]))
    return 2;
  const std::filesystem::path in(argv[1]), out(argv[2]);
  auto backend = ggml_backend_cuda_init(0);
  if (!backend) return 2;
  // Original backend context is acceptable only in this external oracle.
  // Its original launchers and destruction use the same image ABI.
  auto& cuda = *static_cast<ggml_backend_cuda_context*>(backend->context);
  for (std::size_t rows : {1U, 2U, 4U, 8U, 128U}) {
    const auto name = "rows" + std::to_string(rows);
    auto logits = Read<float>(in / (name + "-logits.bin"), rows * 128);
    auto experts = Read<float>(in / (name + "-experts.bin"), rows * 8 * 2816);
    auto scales = Read<float>(in / (name + "-scales.bin"), rows * 8);
    auto weights = Read<float>(in / (name + "-weights.bin"), rows * 8);
    auto* c = ggml_init(
        {.mem_size = 32 * ggml_tensor_overhead(), .mem_buffer = nullptr, .no_alloc = true});
    if (!c) return 2;
    const auto r = static_cast<std::int64_t>(rows);
    auto* x = ggml_new_tensor_2d(c, GGML_TYPE_F32, 128, r);
    auto* w = ggml_new_tensor_3d(c, GGML_TYPE_F32, 1, 8, r);
    auto* root = ggml_argsort(c, x, GGML_SORT_ORDER_DESC);
    std::vector<float> route(rows * 8, 0), values(rows * 2816, 0);
    std::vector<std::int32_t> ids(rows * 128, -777777);
    Place(x, logits);
    Place(w, route);
    Place(root, ids);
    auto* iv = ggml_view_2d(c, root, 8, r, root->nb[1], 0);
    auto* clamp = ggml_clamp(c, w, 0x1p-14f, std::numeric_limits<float>::infinity());
    auto* e = ggml_new_tensor_3d(c, GGML_TYPE_F32, 2816, 8, r);
    auto* s = ggml_new_tensor_3d(c, GGML_TYPE_F32, 1, 8, r);
    auto* rw = ggml_new_tensor_3d(c, GGML_TYPE_F32, 1, 8, r);
    auto* dst = ggml_new_tensor_2d(c, GGML_TYPE_F32, 2816, r);
    Place(e, experts);
    Place(s, scales);
    Place(rw, weights);
    Place(dst, values);
    Check(cudaDeviceSynchronize());  // All pageable staging complete.
    const ggml_cuda_topk_moe_args args{.softmax = true, .norm = true};
    ggml_cuda_op_topk_moe(cuda, x, w, iv, clamp, nullptr, nullptr, args);
    ggml_cuda_op_moe_weighted_reduction(cuda, e, s, rw, dst);
    Check(cudaDeviceSynchronize());
    Check(cudaMemcpy(route.data(), w->data, route.size() * 4, cudaMemcpyDeviceToHost));
    Check(cudaMemcpy(ids.data(), root->data, ids.size() * 4, cudaMemcpyDeviceToHost));
    Check(cudaMemcpy(values.data(), dst->data, values.size() * 4, cudaMemcpyDeviceToHost));
    Write(out / (name + "-route.bin"), route);
    Write(out / (name + "-ids.bin"), ids);
    Write(out / (name + "-values.bin"), values);
    for (auto* t : {x, w, root, e, s, rw, dst}) Check(cudaFree(t->data));
    ggml_free(c);
    std::printf("PINNED_GEMMA_MOE rows=%zu complete\n", rows);
  }
  ggml_backend_free(backend);
  return 0;
}
