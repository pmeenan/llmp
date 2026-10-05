// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0
// Original public backend scalar quantized FFN; fusion ON versus split graphs.
#include <sys/stat.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "replay_inputs.h"
namespace ar = ffn_replay;
namespace {
void Require(bool ok, const char* why) {
  if (!ok) {
    std::cerr << "REFUSAL " << why << '\n';
    std::exit(1);
  }
}
struct Lifetime {
  ar::Inputs input;
  ggml_context* ctx = nullptr;
  ggml_context* weights_ctx = nullptr;
  ggml_backend_t backend = nullptr;
  ggml_backend_buffer_t buffer = nullptr, weights_buffer = nullptr;
  std::vector<ggml_cgraph*> graphs;
  ggml_tensor* activation = nullptr;
  ggml_tensor* output = nullptr;
  std::vector<float> values, first;
};
}  // namespace
int main(int argc, char** argv) {
  umask(0077);
  std::cout << std::setprecision(12);
  if (argc != 4 || (std::string_view(argv[3]) != "A" && std::string_view(argv[3]) != "A0"))
    return 2;
  const bool split = std::string_view(argv[3]) == "A0";
  auto lifetime = std::make_unique<Lifetime>();
  auto& run = *lifetime;
  const std::filesystem::path out = argv[2];
  std::error_code error;
  Require(std::filesystem::create_directory(out, error) && !error, "new output required");
  Require(run.input.Read(argv[1]), "closed same-input packed operands");
  run.ctx = ggml_init({.mem_size = 2U << 20U, .mem_buffer = nullptr, .no_alloc = true});
  Require(run.ctx, "metadata arena");
  run.backend = ggml_backend_cuda_init(0);
  Require(run.backend, "original CUDA backend");
  run.weights_ctx = ggml_init({.mem_size = 2U << 20U, .mem_buffer = nullptr, .no_alloc = true});
  Require(run.weights_ctx, "weight metadata arena");
  auto* x = ggml_new_tensor_2d(run.ctx, GGML_TYPE_F32, ar::kWidth, 1);
  // Public roots have one zero-filled extra quant row. Public tensor_alloc binds
  // exact-shape leaf weight descriptors at their bases; no VIEW graph node
  // interrupts the adjacent products and no private padding is assumed.
  std::array<ggml_tensor*, 3> weight_roots{
      ggml_new_tensor_2d(run.weights_ctx, GGML_TYPE_Q4_K, ar::kWidth, ar::kHidden + 1),
      ggml_new_tensor_2d(run.weights_ctx, GGML_TYPE_Q4_K, ar::kWidth, ar::kHidden + 1),
      ggml_new_tensor_2d(run.weights_ctx, GGML_TYPE_Q6_K, ar::kHidden, ar::kWidth + 1)};
  std::array<ggml_tensor*, 3> weights{};
  for (std::size_t i = 0; i < 3; ++i)
    weights[i] =
        ggml_new_tensor_2d(run.ctx, i == 2 ? GGML_TYPE_Q6_K : GGML_TYPE_Q4_K,
                           i == 2 ? ar::kHidden : ar::kWidth, i == 2 ? ar::kWidth : ar::kHidden);
  auto* gate = ggml_mul_mat(run.ctx, weights[0], x);
  auto* up = ggml_mul_mat(run.ctx, weights[1], x);
  run.activation = ggml_geglu_split(run.ctx, gate, up);
  // A0 materializes gate/up in separate graphs. GLU and down refer to duplicate
  // descriptors bound to those exact GPU bytes, so forward expansion cannot
  // accidentally reinsert/fuse their producers.
  auto* glu_node = run.activation;
  auto* down_input = ggml_dup_tensor(run.ctx, run.activation);
  if (split) {
    auto* gate_input = ggml_dup_tensor(run.ctx, gate);
    auto* up_input = ggml_dup_tensor(run.ctx, up);
    run.activation = ggml_geglu_split(run.ctx, gate_input, up_input);
    glu_node = run.activation;
  }
  run.output = ggml_mul_mat(run.ctx, weights[2], down_input);
  Require(run.input.Matches({gate, up, run.output}, glu_node), "captured product/GeGLU parameters");
  const auto graph = [&](ggml_tensor* output) {
    auto* g = ggml_new_graph_custom(run.ctx, 32, false);
    ggml_build_forward_expand(g, output);
    run.graphs.push_back(g);
  };
  if (split) {
    graph(gate);
    graph(up);
  }
  graph(run.activation);
  graph(run.output);
  run.weights_buffer = ggml_backend_alloc_ctx_tensors(run.weights_ctx, run.backend);
  Require(run.weights_buffer, "weight allocation");
  ggml_backend_buffer_set_usage(run.weights_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
  ggml_backend_buffer_clear(run.weights_buffer, 0);
  for (std::size_t i = 0; i < 3; ++i) {
    Require(ggml_backend_buffer_get_alloc_size(run.weights_buffer, weights[i]) <=
                ggml_backend_buffer_get_alloc_size(run.weights_buffer, weight_roots[i]),
            "weight leaf fits public root allocation");
    Require(ggml_backend_tensor_alloc(run.weights_buffer, weights[i], weight_roots[i]->data) ==
                GGML_STATUS_SUCCESS,
            "public weight leaf binding");
  }
  run.buffer = ggml_backend_alloc_ctx_tensors(run.ctx, run.backend);
  Require(run.buffer, "original public compute allocation");
  // Rebind only descriptor-only materialized input aliases; no computation or
  // memory allocation changes during a step or capture.
  if (split) {
    glu_node->src[0]->data = gate->data;
    glu_node->src[0]->buffer = gate->buffer;
    glu_node->src[1]->data = up->data;
    glu_node->src[1]->buffer = up->buffer;
  }
  down_input->data = run.activation->data;
  down_input->buffer = run.activation->buffer;
  ggml_backend_tensor_set(x, run.input.data[0].data(), 0, ar::kLogical[0]);
  for (std::size_t i = 0; i < 3; ++i) {
    Require(weight_roots[i]->nb[1] == (i == 2 ? 17640U : 3024U), "original quant row pitch");
    ggml_backend_tensor_set(weight_roots[i], run.input.data[i + 1].data(), 0, ar::kLogical[i + 1]);
    std::cout << "ORIGINAL_WEIGHT root_rows=" << weight_roots[i]->ne[1]
              << " actual_rows=" << weights[i]->ne[1] << " logical=" << ar::kLogical[i + 1]
              << " root_bytes=" << ggml_nbytes(weight_roots[i]) << " public_allocated="
              << ggml_backend_buffer_get_alloc_size(run.weights_buffer, weight_roots[i]) << '\n';
  }
  ggml_backend_synchronize(run.backend);
  run.values.resize(ar::kOutputElements);
  const auto step = [&] {
    for (auto* graph : run.graphs) {
      const auto status = ggml_backend_graph_compute(run.backend, graph);
      Require(status == GGML_STATUS_SUCCESS, "original complete scalar FFN compute");
    }
    ggml_backend_synchronize(run.backend);
    ggml_backend_tensor_get(run.activation, run.values.data(), 0, ar::kHidden * 4);
    ggml_backend_tensor_get(run.output, run.values.data() + ar::kHidden, 0, ar::kWidth * 4);
    ggml_backend_synchronize(run.backend);
    Require(ar::Finite(run.values), "original full output finite");
  };
  const auto witness = [&](std::span<const std::byte> expected_q, const char* phase) {
    {
      std::vector<std::byte> got(ar::kBytes[0]);
      ggml_backend_tensor_get(x, got.data(), 0, got.size());
      ggml_backend_synchronize(run.backend);
      Require(got.size() == expected_q.size() &&
                  std::memcmp(got.data(), expected_q.data(), got.size()) == 0,
              "original input witness");
    }
    for (std::size_t i = 0; i < 3; ++i) {
      std::vector<std::byte> got(ggml_nbytes(weight_roots[i]));
      ggml_backend_tensor_get(weight_roots[i], got.data(), 0, got.size());
      ggml_backend_synchronize(run.backend);
      Require(got.size() >= ar::kBytes[i + 1] &&
                  std::memcmp(got.data(), run.input.data[i + 1].data(), ar::kBytes[i + 1]) == 0,
              "original weight witness");
      Require(std::all_of(got.begin() + static_cast<std::ptrdiff_t>(ar::kBytes[i + 1]), got.end(),
                          [](std::byte b) { return b == std::byte{0}; }),
              "constructed root padding changed");
    }
    std::cout << "ORIGINAL_GPU_OPERAND_WITNESS phase=" << phase
              << " complete_input_quant_weights_byte_exact=1\n";
  };
  witness(run.input.data[0], "before");
  step();
  run.first = run.values;
  for (int i = 0; i < 3; ++i) step();
  Require(run.values == run.first &&
              std::memcmp(run.values.data(), run.first.data(), ar::kOutputBytes) == 0,
          "original warm own byte identity");
  Require(ar::Write(out / "first.f32", run.first), "original first output write");
  auto fresh_q = run.input.data[0];
  for (std::size_t i = 0; i < fresh_q.size(); i += sizeof(float)) {
    float value = 0;
    std::memcpy(&value, fresh_q.data() + i, sizeof(value));
    value *= -0.5f;
    std::memcpy(fresh_q.data() + i, &value, sizeof(value));
  }
  ggml_backend_tensor_set(x, fresh_q.data(), 0, fresh_q.size());
  ggml_backend_synchronize(run.backend);
  step();
  auto fresh = run.values;
  Require(std::memcmp(fresh.data(), run.first.data(), ar::kOutputBytes) != 0,
          "original fresh input did not change output");
  step();
  Require(std::memcmp(fresh.data(), run.values.data(), ar::kOutputBytes) == 0,
          "original replay stale fresh input");
  witness(fresh_q, "fresh");
  Require(ar::Write(out / "fresh.f32", fresh), "original fresh output write");
  ggml_backend_tensor_set(x, run.input.data[0].data(), 0, run.input.data[0].size());
  ggml_backend_synchronize(run.backend);
  step();
  Require(std::memcmp(run.first.data(), run.values.data(), ar::kOutputBytes) == 0,
          "original restored input movement");
  witness(run.input.data[0], "restored");
  const auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < 32; ++i) {
    step();
    volatile std::size_t winners = 0;
    winners = static_cast<std::size_t>(
        std::max_element(run.values.begin(), run.values.begin() + ar::kHidden) -
        run.values.begin());
    winners = winners + static_cast<std::size_t>(
                            std::max_element(run.values.begin() + ar::kHidden, run.values.end()) -
                            (run.values.begin() + ar::kHidden));
    (void)winners;
  }
  const double seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  Require(std::memcmp(run.values.data(), run.first.data(), ar::kOutputBytes) == 0,
          "original paid own byte identity");
  Require(ar::Write(out / "repeat.f32", run.values), "original repeat output write");
  witness(run.input.data[0], "after-paid");
  std::cout << "ORIGINAL_FFN_REPLAY arm=" << (split ? "A0" : "A") << " seconds=" << seconds
            << " completed_chains=32 api_groups=" << 32 * run.graphs.size()
            << " output_bytes_per_chain=" << ar::kOutputBytes
            << " public_compute_buffer=" << ggml_backend_buffer_get_size(run.buffer)
            << " public_weight_buffer=" << ggml_backend_buffer_get_size(run.weights_buffer)
            << " known_host_allowance=" << ar::kHostAllowance
            << " selector_observation=requires_launch_trace\n";
  ggml_backend_synchronize(run.backend);
  ggml_backend_buffer_free(run.buffer);
  run.buffer = nullptr;
  ggml_backend_buffer_free(run.weights_buffer);
  run.weights_buffer = nullptr;
  ggml_free(run.weights_ctx);
  run.weights_ctx = nullptr;
  ggml_backend_free(run.backend);
  run.backend = nullptr;
  ggml_free(run.ctx);
  run.ctx = nullptr;
  std::ofstream completion(out / "completion.json", std::ios::noreplace);
  completion << "{\"phase\":\"completed_after_backend_release\",\"paid_chains\":32}\n";
  completion.flush();
  Require(bool(completion), "exclusive final completion receipt");
  return 0;
}
