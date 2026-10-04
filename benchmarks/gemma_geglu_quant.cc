// SPDX-FileCopyrightText: 2026 jitllm contributors
// SPDX-License-Identifier: Apache-2.0

// Paid synthetic Gemma expert chain, original / native unfused / native
// GeGLU writer / original. No model dispatch or router qualification.
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <numeric>
#include <print>
#include <random>
#include <vector>

#include "base/bytes.h"
#include "kernels/ggml/jitllm_ops.h"
#include "kernels/ggml/launch.h"
#include "kernels/ggml/ops.h"
#include "kernels/ggml/ops_ext.h"
#include "providers/cuda/cuda_device_execution.h"

namespace kg = jitllm::kernels::ggml;

int main(int argc, char** argv) {
  std::array<int, 3> args{1, 0, 32};
  if (argc > 4) return 1;
  for (int i = 1; i < argc; ++i) {
    const char* end = argv[i] + std::strlen(argv[i]);
    auto parsed = std::from_chars(argv[i], end, args[static_cast<std::size_t>(i - 1)]);
    if (parsed.ec != std::errc{} || parsed.ptr != end) return 1;
  }
  const int tokens = args[0], shared = args[1], repeats = args[2];
  if ((tokens != 1 && tokens != 2 && tokens != 4) || (shared != 0 && shared != 2 && shared != 8) ||
      repeats < 1 || repeats > 256)
    return 1;
  constexpr std::int64_t k = 2816, ffn = 704, experts = 128, used = 8;
  constexpr std::size_t workspace = 4U << 20U;
  const auto round = [](std::size_t n, std::size_t unit) { return ((n + unit - 1) / unit) * unit; };
  const auto gate_row = ggml_row_size(GGML_TYPE_Q4_K, k);
  const auto down_row = ggml_row_size(GGML_TYPE_Q5_1, ffn);
  const auto gate_bytes = gate_row * static_cast<std::size_t>(2 * ffn);
  const auto down_bytes = down_row * static_cast<std::size_t>(k);
  const auto down_offset = round(gate_bytes + ggml_row_size(GGML_TYPE_Q4_K, 512), 256);
  // Disk member starts/padding are 256 aligned. Runtime ExpertSlab pitch
  // rounds group stored bytes to LCM(16,144,24), not to 256.
  const auto pitch =
      round(round(down_offset + down_bytes + ggml_row_size(GGML_TYPE_Q5_1, 512), 4096),
            std::lcm(std::lcm(std::size_t{16}, ggml_type_size(GGML_TYPE_Q4_K)),
                     ggml_type_size(GGML_TYPE_Q5_1)));
  auto opened = jitllm::providers::cuda::OpenDeviceExecution(0);
  if (!opened) return 1;
  auto execution = std::move(*opened);
  auto made = execution->CreateStream();
  if (!made) return 1;
  const auto stream_id = *made;
  const auto submission = execution->Submission(stream_id);
  if (!submission) return 1;
  const auto stream = reinterpret_cast<cudaStream_t>(submission->handle);
  std::vector<void*> memory;
  std::size_t allocated_bytes = 0;
  const auto allocate = [&](std::size_t bytes) -> void* {
    void* p = nullptr;
    if (cudaMalloc(&p, bytes) != cudaSuccess || !p) return nullptr;
    memory.push_back(p);
    allocated_bytes += bytes;
    if (cudaMemset(p, 0, bytes) != cudaSuccess) return nullptr;
    return p;
  };
  void* scratch = allocate(workspace);
  if (!scratch) return 1;
  auto context = kg::LaunchContext::Create(
      0, *execution, stream_id,
      {.base = reinterpret_cast<std::uintptr_t>(scratch), .size = jitllm::base::Bytes(workspace)});
  if (!context) return 1;
  auto launch = std::move(*context);
  auto made_arena = kg::TensorArena::Create(256);
  if (!made_arena) return 1;
  auto arena = std::move(*made_arena);
  auto* c = arena.context();
  const auto place = [&](ggml_tensor* t, const void* data = nullptr) {
    void* p = allocate(ggml_nbytes(t));
    if (!p) return static_cast<ggml_tensor*>(nullptr);
    kg::TensorArena::Bind(t, reinterpret_cast<std::uintptr_t>(p));
    if (data && cudaMemcpy(p, data, ggml_nbytes(t), cudaMemcpyHostToDevice) != cudaSuccess)
      return static_cast<ggml_tensor*>(nullptr);
    return t;
  };
  void* slab = allocate(pitch * static_cast<std::size_t>(experts));
  if (!slab) return 1;
  auto* fused = ggml_new_tensor_3d(c, GGML_TYPE_Q4_K, k, 2 * ffn, experts);
  fused->nb[2] = pitch;
  fused->nb[3] = pitch * static_cast<std::size_t>(experts);
  kg::TensorArena::Bind(fused, reinterpret_cast<std::uintptr_t>(slab));
  kg::MarkRowPaddingReadable(fused);
  const auto view = [&](ggml_tensor* t, std::int64_t n, std::size_t offset) {
    auto* v = ggml_view_3d(c, t, t->ne[0], n, t->ne[2], t->nb[1], t->nb[2], offset);
    kg::TensorArena::Bind(v, reinterpret_cast<std::uintptr_t>(t->data) + offset);
    kg::MarkRowPaddingReadable(v);
    return v;
  };
  auto* wg = view(fused, ffn, 0);
  auto* wu = view(fused, ffn, gate_row * static_cast<std::size_t>(ffn));
  auto* wd = ggml_new_tensor_3d(c, GGML_TYPE_Q5_1, ffn, k, experts);
  wd->nb[2] = pitch;
  wd->nb[3] = pitch * static_cast<std::size_t>(experts);
  kg::TensorArena::Bind(wd, reinterpret_cast<std::uintptr_t>(slab) + down_offset);
  kg::MarkRowPaddingReadable(wd);
  std::mt19937 random(1704);
  std::normal_distribution<float> normal(0.0f, 0.05f);
  // Eight seeded matrices repeat across 128 physically distinct experts.
  // This bounded screen tests geometry and declared routing, not model weights.
  for (int pattern = 0; pattern < 8; ++pattern) {
    for (auto* w : {fused, wd}) {
      const auto bytes = ggml_row_size(w->type, w->ne[0]) * static_cast<std::size_t>(w->ne[1]);
      std::vector<float> values(static_cast<std::size_t>(w->ne[0] * w->ne[1]));
      for (auto& x : values) x = normal(random);
      std::vector<std::uint8_t> packed(bytes);
      ggml_quantize_init(w->type);
      if (ggml_quantize_chunk(w->type, values.data(), packed.data(), 0, w->ne[1], w->ne[0],
                              nullptr) != bytes)
        return 1;
      for (std::int64_t e = pattern; e < experts; e += 8)
        if (cudaMemcpy(static_cast<char*>(w->data) + static_cast<std::size_t>(e) * pitch,
                       packed.data(), bytes, cudaMemcpyHostToDevice) != cudaSuccess)
          return 1;
    }
  }
  std::vector<float> values(static_cast<std::size_t>(k * tokens));
  for (auto& x : values) x = normal(random) * 10.0f;
  auto* input = place(ggml_new_tensor_3d(c, GGML_TYPE_F32, k, 1, tokens), values.data());
  std::vector<std::int32_t> routes(static_cast<std::size_t>(used * tokens));
  for (int t = 0; t < tokens; ++t)
    for (std::int64_t u = 0; u < used; ++u)
      routes[static_cast<std::size_t>(t * used + u)] =
          static_cast<std::int32_t>((u * 7 + (u < shared ? 0 : t * 3)) % experts);
  auto* ids = place(ggml_new_tensor_2d(c, GGML_TYPE_I32, used, tokens), routes.data());
  if (!input || !ids) return 1;
  auto* q8 = place(kg::QuantizeQ8(c, input));
  if (!q8) return 1;
  struct Arm {
    const char* name;
    ggml_tensor* activation;
    ggml_tensor* down;
    ggml_tensor* output;
    std::vector<std::function<bool()>> steps;
    std::size_t peak;
    std::size_t storage;
  };
  std::array<Arm, 3> arms{};
  for (int a = 0; a < 3; ++a) {
    auto& arm = arms[static_cast<std::size_t>(a)];
    const auto storage_start = allocated_bytes;
    arm.name = a == 0 ? "A original" : a == 1 ? "B0 native unfused" : "B1 GeGLU writer";
    if (a == 0) {
      auto* product = place(ggml_mul_mat_id(c, fused, input, ids));
      if (!product) return 1;
      auto* g = ggml_view_3d(c, product, ffn, used, tokens, product->nb[1], product->nb[2], 0);
      auto* u = ggml_view_3d(c, product, ffn, used, tokens, product->nb[1], product->nb[2],
                             static_cast<std::size_t>(ffn) * sizeof(float));
      kg::TensorArena::Bind(g, reinterpret_cast<std::uintptr_t>(product->data));
      kg::TensorArena::Bind(u, reinterpret_cast<std::uintptr_t>(product->data) +
                                   static_cast<std::size_t>(ffn) * sizeof(float));
      arm.activation = place(ggml_geglu_split(c, g, u));
      const auto peak = kg::PlanMulMatVecQ(*launch, product);
      if (!peak || !arm.activation) return 1;
      arm.peak = *peak;
      arm.steps.push_back([&, product] { return kg::MulMatVecQ(*launch, product).has_value(); });
      arm.steps.push_back(
          [&, node = arm.activation] { return kg::GeGlu(*launch, node).has_value(); });
    } else {
      arm.steps.push_back([&, q8] { return kg::RunQuantizeQ8(*launch, q8).has_value(); });
      if (a == 1) {
        auto* g = place(kg::VecQ(c, wg, q8, ids, tokens, false));
        auto* u = place(kg::VecQ(c, wu, q8, ids, tokens, false));
        if (!g || !u) return 1;
        kg::SetVecQOneToken(g);
        kg::SetVecQOneToken(u);
        arm.activation = place(ggml_geglu_split(c, g, u));
        arm.steps.push_back([&, g] { return kg::RunVecQ(*launch, g).has_value(); });
        arm.steps.push_back([&, u] { return kg::RunVecQ(*launch, u).has_value(); });
        arm.steps.push_back(
            [&, node = arm.activation] { return kg::GeGlu(*launch, node).has_value(); });
      } else {
        arm.activation = place(kg::VecQ(c, wu, q8, ids, tokens, false, wg, kg::VecQGlu::kGeGlu));
        if (!arm.activation) return 1;
        kg::SetVecQOneToken(arm.activation);
        arm.steps.push_back(
            [&, node = arm.activation] { return kg::RunVecQ(*launch, node).has_value(); });
      }
    }
    if (!arm.activation) return 1;
    arm.down = place(ggml_mul_mat_id(c, wd, arm.activation, ids));
    if (!arm.down) return 1;
    const auto peak = kg::PlanMulMatVecQ(*launch, arm.down);
    if (!peak) return 1;
    arm.peak = std::max(arm.peak, *peak);
    arm.steps.push_back([&, node = arm.down] { return kg::MulMatVecQ(*launch, node).has_value(); });
    // Preserve expert scale, normalized positive route weights and slot order.
    const std::vector<float> scales(static_cast<std::size_t>(used * tokens), 0.75f);
    std::vector<float> probabilities(static_cast<std::size_t>(used * tokens));
    for (int t = 0; t < tokens; ++t)
      for (std::int64_t u = 0; u < used; ++u)
        probabilities[static_cast<std::size_t>(t * used + u)] = static_cast<float>(u + 1) / 36.0f;
    auto* scale = place(ggml_new_tensor_3d(c, GGML_TYPE_F32, 1, used, tokens), scales.data());
    auto* weights =
        place(ggml_new_tensor_3d(c, GGML_TYPE_F32, 1, used, tokens), probabilities.data());
    if (!scale || !weights) return 1;
    auto* scaled = place(ggml_mul(c, arm.down, scale));
    if (!scaled) return 1;
    auto* weighted = place(ggml_mul(c, scaled, weights));
    if (!weighted) return 1;
    arm.steps.push_back([&, scaled] { return kg::Mul(*launch, scaled).has_value(); });
    arm.steps.push_back([&, weighted] { return kg::Mul(*launch, weighted).has_value(); });
    ggml_tensor* sum = nullptr;
    for (std::int64_t u = 0; u < used; ++u) {
      auto* v = ggml_view_2d(c, weighted, k, tokens, weighted->nb[2],
                             static_cast<std::size_t>(u) * weighted->nb[1]);
      kg::TensorArena::Bind(v, reinterpret_cast<std::uintptr_t>(weighted->data) +
                                   static_cast<std::size_t>(u) * weighted->nb[1]);
      if (!sum)
        sum = v;
      else {
        sum = place(ggml_add(c, sum, v));
        if (!sum) return 1;
        arm.steps.push_back([&, node = sum] { return kg::Add(*launch, node).has_value(); });
      }
    }
    arm.output = sum;
    arm.storage = allocated_bytes - storage_start + (a == 0 ? 0 : ggml_nbytes(q8));
  }
  const auto run = [](Arm& arm) {
    for (auto& step : arm.steps)
      if (!step()) return false;
    return true;
  };
  const auto download = [&](ggml_tensor* node) {
    std::vector<float> out(static_cast<std::size_t>(ggml_nelements(node)));
    if (cudaMemcpy(out.data(), node->data, out.size() * sizeof(float), cudaMemcpyDeviceToHost) !=
        cudaSuccess)
      out.clear();
    return out;
  };
  // Initialization used the default stream; execution owns a nonblocking
  // provider stream. Retire every setup write before either timing or capture.
  if (cudaDeviceSynchronize() != cudaSuccess) return 1;
  std::array<std::vector<float>, 3> outputs, activations;
  for (int a = 0; a < 3; ++a) {
    if (!run(arms[static_cast<std::size_t>(a)]) || cudaStreamSynchronize(stream) != cudaSuccess)
      return 1;
    outputs[static_cast<std::size_t>(a)] = download(arms[static_cast<std::size_t>(a)].output);
    activations[static_cast<std::size_t>(a)] =
        download(arms[static_cast<std::size_t>(a)].activation);
    if (outputs[static_cast<std::size_t>(a)].empty()) return 1;
  }
  // Independent ordered weighting over the actual downloaded down output.
  // Router probabilities are synthetic fixed inputs, not a DeepSeek combine.
  const auto down_values = download(arms[0].down);
  if (down_values.size() != static_cast<std::size_t>(k * used * tokens)) return 1;
  double combine_error = 0, combine_norm = 0;
  for (int t = 0; t < tokens; ++t) {
    for (std::int64_t row = 0; row < k; ++row) {
      double scalar = 0;
      for (std::int64_t u = 0; u < used; ++u)
        scalar +=
            static_cast<double>(down_values[static_cast<std::size_t>((t * used + u) * k + row)]) *
            0.75 * (static_cast<double>(u + 1) / 36.0);
      const double value = outputs[0][static_cast<std::size_t>(t * k + row)];
      combine_error += (value - scalar) * (value - scalar);
      combine_norm += scalar * scalar;
    }
  }
  const double combine_nmse = combine_norm > 0 ? combine_error / combine_norm : combine_error;
  if (combine_nmse > 1e-12) return 1;
  std::println("independent ordered scale/route-weight/sum NMSE {:.8g}", combine_nmse);
  const auto same_bits = [](const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() && !a.empty() &&
           std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
  };
  if (!same_bits(outputs[1], outputs[2]) || !same_bits(activations[1], activations[2])) {
    std::println("FAIL native unfused/writer exact agreement");
    return 1;
  }
  double error = 0, norm = 0;
  for (std::size_t i = 0; i < outputs[0].size(); ++i) {
    if (!std::isfinite(outputs[0][i]) || !std::isfinite(outputs[2][i])) return 1;
    const double d = static_cast<double>(outputs[0][i]) - outputs[2][i];
    error += d * d;
    norm += static_cast<double>(outputs[0][i]) * outputs[0][i];
  }
  const double nmse = norm > 0 ? error / norm : error;
  if (nmse > 5e-4) return 1;
  std::println(
      "Q4_K fused [2816,1408,128] GeGLU704 Q5_1 down [704,2816,128] rows{} synthetic overlap{}/8 "
      "repeats{} pitch{} pitch_mod256{} Q8_input{} original/native NMSE{:.8g} B0/B1 "
      "activation+output exact",
      tokens, shared, repeats, pitch, pitch % 256, ggml_nbytes(q8), nmse);
  for (const int index : {0, 1, 2, 0}) {
    auto& arm = arms[static_cast<std::size_t>(index)];
    for (int warm = 0; warm < 8; ++warm)
      if (!run(arm)) return 1;
    if (cudaStreamSynchronize(stream) != cudaSuccess) return 1;
    launch->ResetScratchPeak();
    cudaEvent_t begin = nullptr, end = nullptr;
    if (cudaEventCreate(&begin) != cudaSuccess || cudaEventCreate(&end) != cudaSuccess) return 1;
    const auto timed = [&](const std::function<bool()>& work) {
      if (cudaEventRecord(begin, stream) != cudaSuccess) return -1.0f;
      for (int r = 0; r < repeats; ++r)
        if (!work()) return -1.0f;
      if (cudaEventRecord(end, stream) != cudaSuccess || cudaEventSynchronize(end) != cudaSuccess)
        return -1.0f;
      float ms = 0;
      if (cudaEventElapsedTime(&ms, begin, end) != cudaSuccess || !std::isfinite(ms) || ms <= 0)
        return -1.0f;
      return ms * 1000.0f / static_cast<float>(repeats);
    };
    const float ordinary = timed([&] { return run(arm); });
    if (ordinary < 0 ||
        cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal) != cudaSuccess)
      return 1;
    if (!run(arm)) return 1;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    if (cudaStreamEndCapture(stream, &graph) != cudaSuccess ||
        cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0) != cudaSuccess ||
        cudaGraphLaunch(executable, stream) != cudaSuccess ||
        cudaStreamSynchronize(stream) != cudaSuccess)
      return 1;
    for (int warm = 0; warm < 8; ++warm)
      if (cudaGraphLaunch(executable, stream) != cudaSuccess) return 1;
    if (cudaStreamSynchronize(stream) != cudaSuccess) return 1;
    const float captured =
        timed([&] { return cudaGraphLaunch(executable, stream) == cudaSuccess; });
    if (captured < 0 ||
        !same_bits(download(arm.output), outputs[static_cast<std::size_t>(index)]) ||
        !same_bits(download(arm.activation), activations[static_cast<std::size_t>(index)]) ||
        launch->scratch_peak().value() != arm.peak)
      return 1;
    std::println(
        "{} paid {:.3f} us captured {:.3f} us plan_peak{} context_peak{} nodes{} "
        "activation_storage{} capture exact",
        arm.name, ordinary, captured, arm.peak, launch->scratch_peak().value(), arm.steps.size(),
        arm.storage);
    if (cudaGraphExecDestroy(executable) != cudaSuccess || cudaGraphDestroy(graph) != cudaSuccess ||
        cudaEventDestroy(begin) != cudaSuccess || cudaEventDestroy(end) != cudaSuccess)
      return 1;
  }
  std::println("physical cudaMalloc bytes across all simultaneously retained arms {}",
               allocated_bytes);
  const auto fence = execution->Record(stream_id);
  if (!fence || cudaStreamSynchronize(stream) != cudaSuccess ||
      execution->Query(*fence).value() != jitllm::providers::FenceState::kComplete ||
      !execution->Release(*fence))
    return 1;
  launch.reset();
  if (!execution->DestroyStream(stream_id)) return 1;
  for (void* p : memory)
    if (cudaFree(p) != cudaSuccess) return 1;
  return 0;
}
