// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// Bounded pinned original versus registered Gemma local attention.
// jitllm_gemma_attention_bench [heads16|32] [rows1|2|4|8|16|33|1025] [segments1|2|4]
#include <cuda_runtime.h>

#include <algorithm>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <print>
#include <random>
#include <vector>

#include "base/bytes.h"
#include "kernels/ggml/fattn_mma.h"
#include "kernels/ggml/launch.h"
#include "kernels/ggml/ops.h"
#include "kernels/ggml/ops_ext.h"
#include "providers/cuda/cuda_device_execution.h"

// The unchanged instance is compiled in fattn.cu. Avoid importing CUDA
// device templates into this host benchmark translation unit.
template <int D, ggml_type K, ggml_type V>
void ggml_cuda_flash_attn_ext_vec_case(ggml_backend_cuda_context&, ggml_tensor*);
namespace kg = jitllm::kernels::ggml;
int main(int argc, char** argv) {
  int heads = 16, rows = 1, segments = 1;
  if (argc == 3 || argc == 4) {
    const auto parse = [](const char* s, int& value) {
      const char* end = s + std::strlen(s);
      const auto result = std::from_chars(s, end, value);
      return result.ec == std::errc{} && result.ptr == end;
    };
    if (!parse(argv[1], heads) || !parse(argv[2], rows) || (argc == 4 && !parse(argv[3], segments)))
      return 1;
  } else if (argc != 1)
    return 1;
  if ((heads != 16 && heads != 32) ||
      (rows != 1 && rows != 2 && rows != 4 && rows != 8 && rows != 16 && rows != 33 &&
       rows != 1025) ||
      (segments != 1 && segments != 2 && segments != 4))
    return 1;
  constexpr int d = 256, cells = 1280, launches = 128;
  auto opened = jitllm::providers::cuda::OpenDeviceExecution(0);
  if (!opened) return 1;
  auto execution = std::move(*opened);
  auto made = execution->CreateStream();
  if (!made) return 1;
  const auto stream_id = *made;
  std::vector<void*> memory;
  const auto allocate = [&](std::size_t bytes) -> void* {
    void* p = nullptr;
    if (cudaMalloc(&p, bytes) != cudaSuccess) return nullptr;
    memory.push_back(p);
    if (cudaMemset(p, 0, bytes) != cudaSuccess) return nullptr;
    return p;
  };
  constexpr std::size_t reserve = 16U << 20U;
  void* workspace = allocate(reserve);
  if (!workspace) return 1;
  auto context = kg::LaunchContext::Create(
      0, *execution, stream_id,
      {.base = reinterpret_cast<std::uintptr_t>(workspace), .size = jitllm::base::Bytes(reserve)});
  if (!context) return 1;
  auto owner = std::move(*context);
  auto& launch = *owner;
  auto arena = kg::TensorArena::Create(256).value();
  const auto place = [&](ggml_tensor* t) {
    void* p = allocate(ggml_nbytes(t));
    if (!p) return static_cast<ggml_tensor*>(nullptr);
    kg::TensorArena::Bind(t, reinterpret_cast<std::uintptr_t>(p));
    return t;
  };
  std::mt19937 random(714);
  std::normal_distribution<float> normal(0, 0.25f);
  std::vector<ggml_tensor*> nodes;
  for (int segment = 0; segment < segments; ++segment) {
    auto* q = place(ggml_new_tensor_4d(arena.context(), GGML_TYPE_F32, d, rows, heads, 1));
    auto* k = place(ggml_new_tensor_4d(arena.context(), GGML_TYPE_F16, d, cells, heads / 2, 1));
    auto* v = place(ggml_new_tensor_4d(arena.context(), GGML_TYPE_F16, d, cells, heads / 2, 1));
    const int mask_rows = rows >= 1024 ? (rows + 31) / 32 * 32 : rows;
    auto* mask = place(ggml_new_tensor_2d(arena.context(), GGML_TYPE_F16, cells, mask_rows));
    if (!q || !k || !v || !mask) return 1;
    std::vector<float> inputs(static_cast<std::size_t>(ggml_nelements(q)));
    for (float& value : inputs) value = normal(random);
    if (cudaMemcpy(q->data, inputs.data(), ggml_nbytes(q), cudaMemcpyHostToDevice) != cudaSuccess)
      return 1;
    for (auto* t : {k, v}) {
      std::vector<ggml_fp16_t> values(static_cast<std::size_t>(ggml_nelements(t)));
      for (auto& value : values) value = ggml_fp32_to_fp16(normal(random));
      if (cudaMemcpy(t->data, values.data(), ggml_nbytes(t), cudaMemcpyHostToDevice) != cudaSuccess)
        return 1;
    }
    std::vector<ggml_fp16_t> visibility(static_cast<std::size_t>(cells * mask_rows),
                                        ggml_fp32_to_fp16(-std::numeric_limits<float>::infinity()));
    for (int r = 0; r < rows; ++r) {
      const int position = cells - 2 + r + segment * 17;
      for (int p = position - 1023; p <= position; ++p)
        visibility[static_cast<std::size_t>(r * cells + p % cells)] = ggml_fp32_to_fp16(0);
    }
    if (cudaMemcpy(mask->data, visibility.data(), ggml_nbytes(mask), cudaMemcpyHostToDevice) !=
            cudaSuccess ||
        cudaDeviceSynchronize() != cudaSuccess)
      return 1;
    auto* node = place(ggml_flash_attn_ext(arena.context(), q, k, v, mask, 1, 0, 0));
    if (!node) return 1;
    ggml_prec_set_acc(node, GGML_PREC_F32);
    nodes.push_back(node);
  }
  auto* node = nodes.front();
  const bool vector = kg::FlashAttnVec256Selected(launch, node);
  std::uint64_t scratch = 0;
  int columns = 1;
  if (vector) {
    auto plan = kg::PlanFlashAttnVec256(launch, node);
    if (!plan) return 1;
    scratch = plan->scratch;
  } else {
    auto plan = kg::PlanFlashAttnMmaGqa2(launch, node);
    if (!plan) return 1;
    scratch = plan->scratch;
    columns = plan->columns;
  }
  // Actual pinned overall selector chooses vector only at one row and one
  // sequence on GB10; all other local cases choose group2 MMA.
  if (vector != (rows == 1)) return 1;
  const auto original = [&](kg::LaunchContext& l) {
    for (auto* segment : nodes) {
      auto result = l.Run(jitllm::base::Bytes(scratch), [=](ggml_backend_cuda_context& c) {
        if (vector)
          ggml_cuda_flash_attn_ext_vec_case<256, GGML_TYPE_F16, GGML_TYPE_F16>(c, segment);
        else
          kg::detail::FlashAttnMmaCaseGqa2(columns)(c, segment);
      });
      if (!result) return result;
    }
    return std::expected<void, kg::KernelFailure>{};
  };
  const auto selected = [&](kg::LaunchContext& l) {
    for (auto* segment : nodes) {
      auto result = vector ? kg::FlashAttnVec256(l, segment) : kg::FlashAttnMmaGqa2(l, segment);
      if (!result) return result;
    }
    return std::expected<void, kg::KernelFailure>{};
  };
  if (!original(launch) || cudaDeviceSynchronize() != cudaSuccess) return 1;
  const auto elements = static_cast<std::size_t>(ggml_nelements(node));
  const auto download = [&](std::vector<float>& values) {
    for (std::size_t segment = 0; segment < nodes.size(); ++segment)
      if (cudaMemcpy(values.data() + segment * elements, nodes[segment]->data,
                     ggml_nbytes(nodes[segment]), cudaMemcpyDeviceToHost) != cudaSuccess)
        return false;
    return true;
  };
  std::vector<float> reference(elements * nodes.size());
  if (!download(reference)) return 1;
  launch.ResetScratchPeak();
  if (!selected(launch) || cudaDeviceSynchronize() != cudaSuccess) return 1;
  std::vector<float> candidate(reference.size());
  if (!download(candidate) ||
      std::memcmp(candidate.data(), reference.data(), candidate.size() * sizeof(float)) != 0 ||
      launch.scratch_peak().value() > scratch)
    return 1;
  const auto peak = launch.scratch_peak().value();
  auto captured_a = launch.Capture(original);
  auto captured_b = launch.Capture(selected);
  if (!captured_a || !captured_b) return 1;
  std::optional<kg::CapturedGraph> a(std::move(*captured_a));
  std::optional<kg::CapturedGraph> b(std::move(*captured_b));
  for (auto* graph : {&*a, &*b}) {
    if (!launch.Launch(*graph) || cudaDeviceSynchronize() != cudaSuccess || !download(candidate) ||
        std::memcmp(candidate.data(), reference.data(), candidate.size() * sizeof(float)) != 0)
      return 1;
  }
  std::println(
      "D256 heads{} KV{} rows{} segments{} scale1 cells{} selected{} tile{} scratch{} peak{} "
      "graph_nodes{}/{} "
      "exact",
      heads, heads / 2, rows, segments, cells, vector ? "vector" : "group2", columns, scratch, peak,
      a->nodes(), b->nodes());
  const auto submission = execution->Submission(stream_id);
  if (!submission) return 1;
  const auto stream = reinterpret_cast<cudaStream_t>(submission->handle);
  for (const bool captured : {false, true}) {
    for (int stage = 0; stage < 3; ++stage) {
      const bool use_candidate = stage == 1;
      cudaEvent_t start = nullptr, end = nullptr;
      if (cudaEventCreate(&start) != cudaSuccess || cudaEventCreate(&end) != cudaSuccess) return 1;
      if (cudaEventRecord(start, stream) != cudaSuccess) return 1;
      for (int i = 0; i < launches; ++i) {
        const auto result = captured        ? launch.Launch(use_candidate ? *b : *a)
                            : use_candidate ? selected(launch)
                                            : original(launch);
        if (!result) return 1;
      }
      if (cudaEventRecord(end, stream) != cudaSuccess || cudaEventSynchronize(end) != cudaSuccess)
        return 1;
      float elapsed = 0;
      if (cudaEventElapsedTime(&elapsed, start, end) != cudaSuccess ||
          cudaEventDestroy(start) != cudaSuccess || cudaEventDestroy(end) != cudaSuccess)
        return 1;
      std::println("{} {} {:.3f} us paid mask/fixup", captured ? "graph" : "launch",
                   use_candidate ? "selected" : "original", elapsed * 1000 / launches);
    }
  }
  if (cudaDeviceSynchronize() != cudaSuccess) return 1;
  a.reset();
  b.reset();
  const auto fence = execution->Record(stream_id);
  if (!fence || cudaDeviceSynchronize() != cudaSuccess ||
      execution->Query(*fence).value() != jitllm::providers::FenceState::kComplete ||
      !execution->Release(*fence))
    return 1;
  owner.reset();
  if (!execution->DestroyStream(stream_id)) return 1;
  for (void* p : memory)
    if (cudaFree(p) != cudaSuccess) return 1;
  return 0;
}
