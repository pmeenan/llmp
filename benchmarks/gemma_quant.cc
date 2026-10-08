// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Bounded Gemma Q5_1 down screen: original GGML preparation/product versus
// llmpalooza shared Q8_1 preparation/joined products, identical actual geometry.
// Each launch changes expert IDs; A/B/A bookends include input preparation.
// llmp_gemma_quant_bench [launches, default 64] [shared experts per row: 0|2|8]

#include <cuda_runtime.h>

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <print>
#include <random>
#include <vector>

#include "base/bytes.h"
#include "kernels/ggml/dsv4_fast.h"
#include "kernels/ggml/launch.h"
#include "kernels/ggml/llmp_ops.h"
#include "kernels/ggml/ops_ext.h"
#include "providers/cuda/cuda_device_execution.h"

namespace kg = llmp::kernels::ggml;

int main(int argc, char** argv) {
  int launches = 64;
  int shared = 0;
  if (argc == 2 || argc == 3) {
    const char* end = argv[1] + std::strlen(argv[1]);
    const auto parsed = std::from_chars(argv[1], end, launches);
    if (parsed.ec != std::errc{} || parsed.ptr != end || launches < 1 || launches > 256) {
      return 1;
    }
  } else if (argc != 1) {
    return 1;
  }
  if (argc == 3) {
    const char* end = argv[2] + std::strlen(argv[2]);
    const auto parsed = std::from_chars(argv[2], end, shared);
    if (parsed.ec != std::errc{} || parsed.ptr != end ||
        (shared != 0 && shared != 2 && shared != 8)) {
      return 1;
    }
  }
  constexpr std::int64_t k = 704, n = 2816, experts = 128, used = 8;
  constexpr std::size_t workspace_bytes = 4U << 20U;
  auto opened = llmp::providers::cuda::OpenDeviceExecution(0);
  if (!opened) {
    return 1;
  }
  auto execution = std::move(*opened);
  auto made = execution->CreateStream();
  if (!made) {
    return 1;
  }
  const auto stream_id = *made;
  std::vector<void*> memory;
  const auto allocate = [&](std::size_t bytes) -> void* {
    void* p = nullptr;
    if (cudaMalloc(&p, bytes) != cudaSuccess) {
      return nullptr;
    }
    if (cudaMemset(p, 0, bytes) != cudaSuccess) {
      cudaFree(p);
      return nullptr;
    }
    memory.push_back(p);
    return p;
  };
  void* scratch = allocate(workspace_bytes);
  auto context = kg::LaunchContext::Create(0, *execution, stream_id,
                                           {.base = reinterpret_cast<std::uintptr_t>(scratch),
                                            .size = llmp::base::Bytes(workspace_bytes)});
  if (!scratch || !context) {
    return 1;
  }
  auto launch = std::move(*context);
  auto arena = kg::TensorArena::Create(128).value();
  const auto place = [&](ggml_tensor* t, const void* data = nullptr) {
    void* p = allocate(ggml_nbytes(t) + ggml_row_size(GGML_TYPE_Q5_1, 512));
    if (p == nullptr) {
      return static_cast<ggml_tensor*>(nullptr);
    }
    kg::TensorArena::Bind(t, reinterpret_cast<std::uintptr_t>(p));
    if (data && cudaMemcpy(p, data, ggml_nbytes(t), cudaMemcpyHostToDevice) != cudaSuccess) {
      return static_cast<ggml_tensor*>(nullptr);
    }
    return t;
  };
  auto* weights = place(ggml_new_tensor_3d(arena.context(), GGML_TYPE_Q5_1, k, n, experts));
  if (!weights) {
    return 1;
  }
  kg::MarkRowPaddingReadable(weights);
  std::mt19937 rng(704);
  std::normal_distribution<float> normal(0.0f, 0.05f);
  std::vector<float> values(static_cast<std::size_t>(k * n));
  std::vector<std::uint8_t> packed(ggml_row_size(GGML_TYPE_Q5_1, k) * static_cast<std::size_t>(n));
  ggml_quantize_init(GGML_TYPE_Q5_1);
  for (std::int64_t e = 0; e < experts; ++e) {
    for (auto& value : values) {
      value = normal(rng);
    }
    if (ggml_quantize_chunk(GGML_TYPE_Q5_1, values.data(), packed.data(), 0, n, k, nullptr) !=
            packed.size() ||
        cudaMemcpy(static_cast<char*>(weights->data) + static_cast<std::size_t>(e) * weights->nb[2],
                   packed.data(), packed.size(), cudaMemcpyHostToDevice) != cudaSuccess) {
      return 1;
    }
  }
  for (const std::int64_t tokens : {1, 4}) {
    values.resize(static_cast<std::size_t>(k * used * tokens));
    for (auto& value : values) {
      value = normal(rng);
    }
    auto* input =
        place(ggml_new_tensor_3d(arena.context(), GGML_TYPE_F32, k, used, tokens), values.data());
    std::vector<std::int32_t> route(static_cast<std::size_t>(used * tokens * launches));
    for (int l = 0; l < launches; ++l) {
      for (std::int64_t t = 0; t < tokens; ++t) {
        for (std::int64_t u = 0; u < used; ++u) {
          route[static_cast<std::size_t>((l * tokens + t) * used + u)] =
              static_cast<std::int32_t>((l * 17 + (u < shared ? 0 : t * 3) + u * 7) % experts);
        }
      }
    }
    void* route_data = allocate(route.size() * sizeof(std::int32_t));
    if (!input || !route_data ||
        cudaMemcpy(route_data, route.data(), route.size() * sizeof(std::int32_t),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
      return 1;
    }
    auto* ids = ggml_new_tensor_2d(arena.context(), GGML_TYPE_I32, used, tokens);
    kg::TensorArena::Bind(ids, reinterpret_cast<std::uintptr_t>(route_data));
    auto* reference = place(ggml_mul_mat_id(arena.context(), weights, input, ids));
    auto* q8 = place(kg::QuantizeQ8(arena.context(), input));
    auto* candidate = place(kg::VecQ(arena.context(), weights, q8, ids, tokens, true));
    if (!reference || !q8 || !candidate) {
      return 1;
    }
    kg::SetVecQOneToken(candidate);
    const auto ref_scratch = kg::PlanMulMatVecQ(*launch, reference);
    if (!ref_scratch || !kg::CheckVecQ(candidate)) {
      return 1;
    }
    std::println(
        "Q5_1 K704 N2816 experts128 top8 rows{} launches{} weights{} q8{} reference_workspace{} "
        "candidate_workspace{}",
        tokens, launches, ggml_nbytes(weights), ggml_nbytes(q8), *ref_scratch, 0);
    std::println("  synthetic shared experts per later row: {}/8", shared);
    std::vector<float> reference_outputs(static_cast<std::size_t>(n * used * tokens));
    std::vector<float> scalar_outputs(static_cast<std::size_t>(n * used * tokens));
    for (const int stage : {0, 1, 2, 0}) {
      if (stage == 2 && tokens == 1) {
        continue;
      }
      const auto native = execution->Submission(stream_id).value();
      const auto stream = reinterpret_cast<cudaStream_t>(native.handle);
      const auto run = [&]() {
        if (stage == 0) {
          return kg::MulMatVecQ(*launch, reference).has_value();
        }
        if (!kg::RunQuantizeQ8(*launch, q8)) {
          return false;
        }
        if (stage == 1) {
          return kg::RunVecQ(*launch, candidate).has_value();
        }
        kg::VecQDesc d;
        d.w = weights->data;
        d.y = q8->data;
        d.ids = static_cast<const std::int32_t*>(ids->data);
        d.dst = static_cast<float*>(candidate->data);
        d.ncols_x = static_cast<int>(k);
        d.nrows = static_cast<int>(n);
        d.stride_row = static_cast<int>(weights->nb[1] / ggml_type_size(weights->type));
        d.stride_expert = static_cast<int>(weights->nb[2] / ggml_type_size(weights->type));
        d.y_token = 32 * static_cast<int>(used);
        d.y_slot = 32;
        d.ids_stride = static_cast<int>(used);
        d.used = static_cast<int>(used);
        d.tokens = static_cast<int>(tokens);
        d.dst_token = static_cast<int>(n * used);
        d.dst_slot = static_cast<int>(n);
        bool accepted = false;
        return launch
                   ->Run(llmp::base::Bytes(0),
                         [&](auto&) { accepted = kg::LaunchVecQ(GGML_TYPE_Q5_1, d, 7, stream); })
                   .has_value() &&
               accepted;
      };
      kg::TensorArena::Bind(ids, reinterpret_cast<std::uintptr_t>(route_data));
      if (!run() || cudaDeviceSynchronize() != cudaSuccess) {
        return 1;
      }
      cudaEvent_t start = nullptr, stop = nullptr;
      if (cudaEventCreate(&start) != cudaSuccess || cudaEventCreate(&stop) != cudaSuccess) {
        return 1;
      }
      if (stage == 0 && cudaMemcpy(reference_outputs.data(), reference->data,
                                   reference_outputs.size() * sizeof(float),
                                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        return 1;
      }
      if (stage == 1 &&
          cudaMemcpy(scalar_outputs.data(), candidate->data, scalar_outputs.size() * sizeof(float),
                     cudaMemcpyDeviceToHost) != cudaSuccess) {
        return 1;
      }
      if (stage == 1) {
        double error = 0, norm = 0;
        for (std::size_t i = 0; i < scalar_outputs.size(); ++i) {
          const double d = static_cast<double>(scalar_outputs[i]) - reference_outputs[i];
          error += d * d;
          norm += static_cast<double>(reference_outputs[i]) * reference_outputs[i];
          if (!std::isfinite(scalar_outputs[i]) || !std::isfinite(reference_outputs[i])) {
            return 1;
          }
        }
        const double nmse = norm > 0 ? error / norm : error;
        if (nmse > 1e-10) {
          std::println(stderr, "P1 differs from reference: NMSE {}", nmse);
          return 1;
        }
        std::println("  P1 versus original warm control NMSE {}", nmse);
      }
      if (stage == 2) {
        std::vector<float> joined_outputs(scalar_outputs.size());
        if (cudaMemcpy(joined_outputs.data(), candidate->data,
                       joined_outputs.size() * sizeof(float),
                       cudaMemcpyDeviceToHost) != cudaSuccess ||
            std::memcmp(joined_outputs.data(), scalar_outputs.data(),
                        joined_outputs.size() * sizeof(float)) != 0) {
          std::println(stderr, "P4 differs from one-column arithmetic");
          return 1;
        }
        std::println("  P4 warm control equals one-column arithmetic bit for bit");
      }
      if (cudaEventRecord(start, stream) != cudaSuccess) {
        return 1;
      }
      for (int l = 0; l < launches; ++l) {
        kg::TensorArena::Bind(
            ids, reinterpret_cast<std::uintptr_t>(route_data) +
                     static_cast<std::uintptr_t>(l * tokens * used) * sizeof(std::int32_t));
        if (!run()) {
          return 1;
        }
      }
      if (cudaEventRecord(stop, stream) != cudaSuccess ||
          cudaEventSynchronize(stop) != cudaSuccess) {
        return 1;
      }
      float ms = 0;
      if (cudaEventElapsedTime(&ms, start, stop) != cudaSuccess || !std::isfinite(ms) || ms <= 0) {
        return 1;
      }
      std::println("  {} {:.3f} us including Q8 preparation",
                   stage == 0   ? "reference"
                   : stage == 1 ? "candidate P1"
                                : "candidate P4",
                   ms * 1000.0f / static_cast<float>(launches));
      if (cudaEventDestroy(start) != cudaSuccess || cudaEventDestroy(stop) != cudaSuccess) {
        return 1;
      }
    }
  }
  const auto fence = execution->Record(stream_id).value();
  if (cudaDeviceSynchronize() != cudaSuccess ||
      execution->Query(fence).value() != llmp::providers::FenceState::kComplete ||
      !execution->Release(fence)) {
    return 1;
  }
  launch.reset();
  if (!execution->DestroyStream(stream_id)) {
    return 1;
  }
  for (void* p : memory) {
    if (cudaFree(p) != cudaSuccess) {
      return 1;
    }
  }
  return 0;
}
