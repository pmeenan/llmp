// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0
// Original public GGML CUDA backend, one reconstructed four-stream attention.
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
namespace ar = attention_replay;
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
  ggml_backend_t backend = nullptr;
  ggml_backend_buffer_t buffer = nullptr;
  ggml_cgraph* graph = nullptr;
  ggml_tensor* output = nullptr;
  std::vector<float> values, first;
};
}  // namespace
int main(int argc, char** argv) {
  umask(0077);
  std::cout << std::setprecision(12);
  if (argc != 3) return 2;
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
  auto* q = ggml_new_tensor_4d(run.ctx, GGML_TYPE_F32, 256, 32, 1, 4);
  auto* k = ggml_new_tensor_4d(run.ctx, GGML_TYPE_F16, 256, 16, 256, 4);
  auto* v = ggml_new_tensor_4d(run.ctx, GGML_TYPE_F16, 256, 16, 256, 4);
  auto* mask = ggml_new_tensor_4d(run.ctx, GGML_TYPE_F16, 256, 32, 1, 4);
  std::array<ggml_tensor*, 4> roots{q, k, v, mask};
  run.output = ar::Attention(run.ctx, q, k, v, mask, run.input);
  Require(ggml_nbytes(run.output) == ar::kOutputBytes, "canonical output layout");
  ggml_set_input(q);
  ggml_set_input(k);
  ggml_set_input(v);
  ggml_set_input(mask);
  ggml_set_output(run.output);
  Require(ggml_backend_supports_op(run.backend, run.output), "original backend flash support");
  run.graph = ggml_new_graph_custom(run.ctx, 32, false);
  ggml_build_forward_expand(run.graph, run.output);
  run.buffer = ggml_backend_alloc_ctx_tensors(run.ctx, run.backend);
  Require(run.buffer, "original backend operand buffer");
  for (std::size_t i = 0; i < roots.size(); ++i)
    ggml_backend_tensor_set(roots[i], run.input.data[i].data(), 0, run.input.data[i].size());
  ggml_backend_synchronize(run.backend);
  run.values.resize(ar::kOutputElements);
  const auto step = [&] {
    const auto status = ggml_backend_graph_compute(run.backend, run.graph);
    ggml_backend_synchronize(run.backend);
    Require(status == GGML_STATUS_SUCCESS, "original complete attention compute");
    ggml_backend_tensor_get(run.output, run.values.data(), 0, ar::kOutputBytes);
    ggml_backend_synchronize(run.backend);
    Require(ar::Finite(run.values), "original full output finite");
  };
  const auto witness = [&](std::span<const std::byte> expected_q, const char* phase) {
    for (std::size_t i = 0; i < roots.size(); ++i) {
      std::vector<std::byte> got(ar::kBytes[i]);
      ggml_backend_tensor_get(roots[i], got.data(), 0, got.size());
      ggml_backend_synchronize(run.backend);
      const auto expected = i == 0 ? expected_q : std::span<const std::byte>(run.input.data[i]);
      Require(got.size() == expected.size() &&
                  std::memcmp(got.data(), expected.data(), got.size()) == 0,
              "original GPU operand changed");
    }
    std::cout << "ORIGINAL_GPU_OPERAND_WITNESS phase=" << phase
              << " complete_qkv_mask_byte_exact=1\n";
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
  ggml_backend_tensor_set(q, fresh_q.data(), 0, fresh_q.size());
  ggml_backend_synchronize(run.backend);
  step();
  auto fresh = run.values;
  Require(std::memcmp(fresh.data(), run.first.data(), ar::kOutputBytes) != 0,
          "original fresh query did not change output");
  step();
  Require(std::memcmp(fresh.data(), run.values.data(), ar::kOutputBytes) == 0,
          "original replay stale fresh query");
  witness(fresh_q, "fresh");
  Require(ar::Write(out / "fresh.f32", fresh), "original fresh output write");
  ggml_backend_tensor_set(q, run.input.data[0].data(), 0, run.input.data[0].size());
  ggml_backend_synchronize(run.backend);
  step();
  Require(std::memcmp(run.first.data(), run.values.data(), ar::kOutputBytes) == 0,
          "original restored query movement");
  witness(run.input.data[0], "restored");
  const auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < 32; ++i) {
    step();
    volatile std::size_t winners = 0;
    for (std::size_t owner = 0; owner < 4; ++owner) {
      auto* begin = run.values.data() + owner * 8192;
      winners = winners + static_cast<std::size_t>(std::max_element(begin, begin + 8192) - begin);
    }
    (void)winners;
  }
  const double seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  Require(std::memcmp(run.values.data(), run.first.data(), ar::kOutputBytes) == 0,
          "original paid own byte identity");
  Require(ar::Write(out / "repeat.f32", run.values), "original repeat output write");
  witness(run.input.data[0], "after-paid");
  std::cout << "ORIGINAL_ATTENTION_REPLAY seconds=" << seconds
            << " completed_units=128 api_groups=32"
            << " constructed_q_ne=256,1,32,4 constructed_k_ne=256,256,16,4 mask_rows=32"
            << " output_bytes_per_wave=" << ar::kOutputBytes
            << " public_backend_buffer=" << ggml_backend_buffer_get_size(run.buffer)
            << " known_host_allowance=" << ar::kHostAllowance
            << " selector_observation=requires_launch_trace\n";
  ggml_backend_synchronize(run.backend);
  ggml_backend_buffer_free(run.buffer);
  run.buffer = nullptr;
  ggml_backend_free(run.backend);
  run.backend = nullptr;
  ggml_free(run.ctx);
  run.ctx = nullptr;
  std::ofstream completion(out / "completion.json", std::ios::noreplace);
  completion << "{\"phase\":\"completed_after_backend_release\",\"owners\":4,\"paid_waves\":32,"
                "\"completed_units\":128}\n";
  completion.flush();
  Require(bool(completion), "exclusive final completion receipt");
  return 0;
}
