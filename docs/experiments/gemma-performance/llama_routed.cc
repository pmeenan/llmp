// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// Pinned-reference diagnostic: same uploaded inputs, IDs and actual GGUF
// weights across fusion policies. Six-row merged gate/up -> split GeGLU ->
// expert down/scales follows the pinned public GGML API arithmetic. This
// does not implement a model runtime or measure competitive performance.
#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "gguf.h"

void Require(bool good, const char* text) {
  if (!good) throw std::runtime_error(text);
}
std::vector<char> Read(const std::filesystem::path& path, std::size_t bytes) {
  Require(std::filesystem::file_size(path) == bytes, "unexpected diagnostic input length");
  std::vector<char> result(bytes);
  std::ifstream file(path, std::ios::binary);
  file.read(result.data(), static_cast<std::streamsize>(bytes));
  Require(bool(file), "reading diagnostic input failed");
  return result;
}
int main(int argc, char** argv) {
  try {
    Require(argc == 5, "usage: MODEL TAPDIR OUTDIR fused|unfused");
    const bool unfused = std::string(argv[4]) == "unfused";
    Require(unfused || std::string(argv[4]) == "fused", "unknown policy");
    const char* disabled = std::getenv("GGML_CUDA_DISABLE_FUSION");
    Require(unfused ? disabled && std::string(disabled) == "1" : !disabled,
            "fusion-disable environment disagrees with policy");
    Require(std::filesystem::file_size(argv[1]) == 16947541728ULL, "wrong approved GGUF length");
    const std::filesystem::path input(argv[2]), out(argv[3]);
    Require(std::filesystem::create_directory(out), "output must be new");
    std::unique_ptr<gguf_context, decltype(&gguf_free)> meta(
        gguf_init_from_file(argv[1], {.no_alloc = true, .ctx = nullptr}), gguf_free);
    Require(bool(meta), "reading GGUF metadata failed");
    const auto backend = ggml_backend_cuda_init(0);
    Require(backend != nullptr, "CUDA backend creation failed");
    const auto wc = ggml_init({.mem_size = 1048576, .mem_buffer = nullptr, .no_alloc = true});
    const auto gc = ggml_init({.mem_size = 1048576, .mem_buffer = nullptr, .no_alloc = true});
    Require(wc && gc, "metadata arena allocation failed");
    const auto weight = [&](const char* name, ggml_type type, std::array<std::int64_t, 4> ne) {
      const auto id = gguf_find_tensor(meta.get(), name);
      Require(id >= 0 && gguf_get_tensor_type(meta.get(), id) == type,
              "wrong approved tensor type");
      const auto* dims = gguf_get_tensor_ne(meta.get(), id);
      Require(std::equal(ne.begin(), ne.end(), dims), "wrong approved tensor dimensions");
      auto* tensor = ggml_new_tensor(wc, type, 4, ne.data());
      ggml_set_name(tensor, name);
      return tensor;
    };
    auto* gate_up = weight("blk.0.ffn_gate_up_exps.weight", GGML_TYPE_Q4_K, {2816, 1408, 128, 1});
    auto* down = weight("blk.0.ffn_down_exps.weight", GGML_TYPE_Q5_1, {704, 2816, 128, 1});
    auto* scale = weight("blk.0.ffn_down_exps.scale", GGML_TYPE_F32, {128, 1, 1, 1});
    auto* wb = ggml_backend_alloc_ctx_tensors(wc, backend);
    Require(wb != nullptr, "weight allocation failed");
    ggml_backend_buffer_set_usage(wb, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    std::ifstream model(argv[1], std::ios::binary);
    std::vector<char> staging(16777216);
    for (auto* tensor : {gate_up, down, scale}) {
      const auto id = gguf_find_tensor(meta.get(), tensor->name);
      const auto offset = gguf_get_data_offset(meta.get()) + gguf_get_tensor_offset(meta.get(), id);
      model.seekg(static_cast<std::streamoff>(offset));
      for (std::size_t at = 0; at < ggml_nbytes(tensor);) {
        const auto count = std::min(staging.size(), ggml_nbytes(tensor) - at);
        model.read(staging.data(), static_cast<std::streamsize>(count));
        Require(bool(model), "reading weight payload failed");
        ggml_backend_tensor_set(tensor, staging.data(), at, count);
        at += count;
      }
      std::cout << "ROUTED_WEIGHT " << tensor->name << " bytes=" << ggml_nbytes(tensor)
                << " offset=" << offset << '\n';
    }
    auto* x = ggml_new_tensor_3d(gc, GGML_TYPE_F32, 2816, 1, 6);
    auto* ids = ggml_new_tensor_2d(gc, GGML_TYPE_I32, 8, 6);
    auto* product = ggml_mul_mat_id(gc, gate_up, x, ids);
    auto* gate = ggml_view_3d(gc, product, 704, 8, 6, product->nb[1], product->nb[2], 0);
    auto* up = ggml_view_3d(gc, product, 704, 8, 6, product->nb[1], product->nb[2], 704 * 4);
    auto* glu = ggml_geglu_split(gc, gate, up);
    auto* projected = ggml_mul_mat_id(gc, down, glu, ids);
    auto* scales = ggml_reshape_3d(gc, scale, 1, 128, 1);
    scales = ggml_repeat_4d(gc, scales, 1, 128, 6, 1);
    scales = ggml_get_rows(gc, scales, ids);
    auto* scaled = ggml_mul(gc, projected, scales);
    auto* graph = ggml_new_graph_custom(gc, 256, false);
    ggml_build_forward_expand(graph, scaled);
    auto* gb = ggml_backend_alloc_ctx_tensors(gc, backend);
    Require(gb != nullptr, "graph allocation failed");
    const auto values = Read(input / "ffn_norm_2-0.bin", 2816 * 6 * 4);
    const auto raw_ids = Read(input / "ffn_moe_topk-0.bin", 2592);
    std::array<std::int32_t, 48> compact{};
    for (int row = 0; row < 6; ++row)
      for (int expert = 0; expert < 8; ++expert) {
        std::memcpy(&compact[row * 8 + expert], raw_ids.data() + row * 512 + expert * 4, 4);
        Require(compact[row * 8 + expert] >= 0 && compact[row * 8 + expert] < 128, "bad expert ID");
      }
    ggml_backend_tensor_set(x, values.data(), 0, values.size());
    ggml_backend_tensor_set(ids, compact.data(), 0, sizeof(compact));
    Require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "compute failed");
    for (const auto& [tensor, name] :
         {std::pair{glu, "geglu.f32"}, std::pair{scaled, "down-scaled.f32"}}) {
      std::vector<char> raw(ggml_nbytes(tensor));
      ggml_backend_tensor_get(tensor, raw.data(), 0, raw.size());
      std::ofstream file(out / name, std::ios::binary);
      file.write(raw.data(), static_cast<std::streamsize>(raw.size()));
      file.flush();
      Require(bool(file), "writing isolated output failed");
    }
    ggml_backend_buffer_free(gb);
    ggml_backend_buffer_free(wb);
    ggml_free(gc);
    ggml_free(wc);
    ggml_backend_free(backend);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
