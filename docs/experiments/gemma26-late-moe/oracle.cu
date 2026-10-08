// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
// External original-image operator client. Defines no floating kernel.
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <sys/stat.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "argsort.cuh"
#include "binbcast.cuh"
#include "clamp.cuh"
#include "getrows.cuh"
#include "ggml-alloc.h"
#include "ggml-backend-impl.h"
#include "ggml-cuda.h"
#include "graphs.h"
#include "inputs.h"
#include "moe-weighted-reduction.cuh"
#include "softmax.cuh"
#include "sumrows.cuh"
#include "topk-moe.cuh"

static void Require(bool condition, const char* detail) {
  if (!condition) {
    std::fprintf(stderr, "original late operator: %s\n", detail);
    std::abort();  // Retain every device/context/host owner on unknown completion.
  }
}
static void Check(cudaError_t status) {
  Require(status == cudaSuccess, cudaGetErrorString(status));
}
static std::size_t Primitives(ggml_backend_cuda_context& cuda, const late_moe::Graph& graph) {
  std::size_t calls = 0;
  for (auto* node : graph.nodes) {
    if (node->view_src) continue;
    using Function = void (*)(ggml_backend_cuda_context&, ggml_tensor*);
    Function function = nullptr;
    const char* name = nullptr;
    switch (node->op) {
      case GGML_OP_SOFT_MAX:
        function = ggml_cuda_op_soft_max;
        name = "ggml_cuda_op_soft_max";
        break;
      case GGML_OP_ARGSORT:
        function = ggml_cuda_op_argsort;
        name = "ggml_cuda_op_argsort";
        break;
      case GGML_OP_GET_ROWS:
        function = ggml_cuda_op_get_rows;
        name = "ggml_cuda_op_get_rows";
        break;
      case GGML_OP_SUM_ROWS:
        function = ggml_cuda_op_sum_rows;
        name = "ggml_cuda_op_sum_rows";
        break;
      case GGML_OP_CLAMP:
        function = ggml_cuda_op_clamp;
        name = "ggml_cuda_op_clamp";
        break;
      case GGML_OP_DIV:
        function = ggml_cuda_op_div;
        name = "ggml_cuda_op_div";
        break;
      case GGML_OP_MUL:
        function = ggml_cuda_op_mul;
        name = "ggml_cuda_op_mul";
        break;
      case GGML_OP_ADD:
        function = ggml_cuda_op_add;
        name = "ggml_cuda_op_add";
        break;
      default:
        Require(false, "outside closed primitive operation");
    }
    Dl_info info{};
    Require(dladdr(reinterpret_cast<void*>(function), &info) && info.dli_fname &&
                std::strcmp(info.dli_fname, "/app/libggml-cuda.so") == 0,
            "actual primitive defining library");
    function(cuda, node);
    ++calls;
    std::printf("LATE_ORIGINAL_IMPLEMENTATION step=%zu name=%s library=%s\n", calls - 1, name,
                info.dli_fname);
  }
  return calls;
}
int main(int argc, char** argv) {
  umask(0077);
  if (argc != 6 ||
      (std::string_view(argv[4]) != "route" && std::string_view(argv[4]) != "reduce") ||
      (std::string_view(argv[5]) != "primitive" && std::string_view(argv[5]) != "fused") ||
      !late_moe::Hex(argv[2]) || std::getenv("LD_PRELOAD") ||
      std::getenv("GGML_CUDA_DISABLE_GRAPHS"))
    return 2;
  const bool route = std::string_view(argv[4]) == "route";
  const bool fused = std::string_view(argv[5]) == "fused";
  if (std::getenv("GGML_CUDA_DISABLE_FUSION")) return 2;
  for (void* symbol : {reinterpret_cast<void*>(&ggml_cuda_op_topk_moe),
                       reinterpret_cast<void*>(&ggml_cuda_op_moe_weighted_reduction)}) {
    Dl_info info{};
    Require(dladdr(symbol, &info) != 0 && info.dli_fname &&
                std::strcmp(info.dli_fname, "/app/libggml-cuda.so") == 0,
            "original defining library");
    std::fprintf(stderr, "LATE_ORIGINAL_SYMBOL name=%s library=%s address=%p\n",
                 info.dli_sname ? info.dli_sname : "unknown", info.dli_fname, symbol);
  }
  late_moe::Inputs inputs;
  Require(inputs.Load(argv[1], argv[2], route), "authenticated finite common inputs");
  const std::filesystem::path out(argv[3]);
  std::error_code error;
  Require(std::filesystem::create_directory(out, error) && !error, "exclusive output");
  auto backend = ggml_backend_cuda_init(0);
  Require(backend, "original backend");
  // Original context is confined to this external diagnostic. No native engine
  // includes or calls the foreign backend.
  auto& cuda = *static_cast<ggml_backend_cuda_context*>(backend->context);
  auto* context =
      ggml_init({.mem_size = 128 * ggml_tensor_overhead() + ggml_graph_overhead_custom(128, false),
                 .mem_buffer = nullptr,
                 .no_alloc = true});
  Require(context, "fixed descriptor/graph budget");
  auto model = route ? late_moe::Routing(context) : late_moe::Reduction(context);
  auto* graph = ggml_new_graph_custom(context, 128, false);
  // Expand every declared dependency in the same order, including all eight
  // reduction views before the seven ordered adds. Counts are measured below.
  for (auto* node : model.nodes) ggml_build_forward_expand(graph, node);
  Require(graph->n_nodes == static_cast<int>(model.nodes.size()), "actual graph count");
  for (int i = 0; i < graph->n_nodes; ++i) {
    Require(graph->nodes[i] == model.nodes[static_cast<std::size_t>(i)], "actual graph order");
    std::printf("LATE_ORIGINAL_DESCRIPTOR index=%d op=%s view=%d\n", i,
                ggml_op_name(graph->nodes[i]->op), graph->nodes[i]->view_src != nullptr);
  }
  auto buffer = ggml_backend_alloc_ctx_tensors(context, backend);
  Require(buffer, "original complete roots");
  Require(ggml_backend_buffer_get_size(buffer) <= (32U << 20U),
          "fixed original device backing cap");
  ggml_backend_buffer_clear(buffer, 0);  // Tail backing exists; never exported/consumed.
  const std::size_t output_bytes = route ? 4096 : 720896;
  std::vector<std::byte> result(output_bytes), first(output_bytes), witness(5767168);
  for (std::size_t layer = 0; layer < 2; ++layer) {
    for (std::size_t i = 0; i < model.inputs.size(); ++i)
      ggml_backend_tensor_set(model.inputs[i], inputs.layers[layer][i].data(), 0,
                              inputs.layers[layer][i].size());
    ggml_backend_synchronize(backend);
    std::size_t primitive_calls = 0;
    const auto execute = [&] {
      if (!fused) {
        // Exactly the original controller's single-node launchers, one per
        // computed dependency. Views are metadata; no generic fusion runs.
        primitive_calls = Primitives(cuda, model);
      } else if (route) {
        const ggml_cuda_topk_moe_args args{.softmax = true, .norm = true};
        ggml_cuda_op_topk_moe(cuda, model.inputs[0], model.outputs[1], model.outputs[0],
                              model.nodes[7], nullptr, nullptr, args);
      } else {
        ggml_cuda_op_moe_weighted_reduction(cuda, model.inputs[0], model.inputs[1], model.inputs[2],
                                            model.outputs[0]);
      }
      Check(cudaDeviceSynchronize());
      std::size_t offset = 0;
      for (auto* tensor : model.outputs) {
        const bool ids = tensor->type == GGML_TYPE_I32;
        for (std::size_t row = 0; row < (ids ? 64U : 1U); ++row) {
          const auto bytes = ids ? 32U : ggml_nbytes(tensor);
          ggml_backend_tensor_get(tensor, result.data() + offset, ids ? row * tensor->nb[1] : 0,
                                  bytes);
          offset += bytes;
        }
      }
      Require(offset == result.size(), "complete output publication");
      std::span<const std::byte> finite(result);
      if (route) {
        for (std::size_t i = 0; i < 2048; i += 4) {
          std::int32_t value = -1;
          std::memcpy(&value, result.data() + i, 4);
          Require(value >= 0 && value < 128, "canonical selected IDs");
        }
        finite = finite.subspan(2048);
      }
      Require(late_moe::Finite(finite), "finite whole output");
    };
    execute();
    first = result;
    execute();
    Require(first == result, "own complete repeat");
    for (std::size_t i = 0; i < model.inputs.size(); ++i) {
      const auto& expected = inputs.layers[layer][i];
      ggml_backend_tensor_get(model.inputs[i], witness.data(), 0, expected.size());
      Require(std::memcmp(witness.data(), expected.data(), expected.size()) == 0,
              "original immutable operand witness");
    }
    const auto name = "layer-" + std::to_string(layer + 28);
    Require(late_moe::Write(out / (name + "-first.bin"), first) &&
                late_moe::Write(out / (name + "-repeat.bin"), result),
            "exclusive completed outputs");
    std::printf(
        "LATE_ORIGINAL layer=%zu kind=%s policy=%s actual_descriptors=%d executed_control=%s "
        "primitive_calls=%zu"
        " own_repeat=1 immutable_inputs=1 output_bytes=%zu output_sha=%s\n",
        layer + 28, route ? "route" : "reduce", fused ? "fused" : "primitive", graph->n_nodes,
        fused ? "exported_original_launcher" : "exported_original_primitive_chain", primitive_calls,
        result.size(), late_moe::Hash(result).c_str());
  }
  ggml_backend_synchronize(backend);
  Check(cudaDeviceSynchronize());
  ggml_backend_buffer_free(buffer);
  ggml_free(context);
  ggml_backend_free(backend);
  Check(cudaDeviceSynchronize());
  std::printf("LATE_ORIGINAL_RETIRED complete=1\n");
  return 0;
}
