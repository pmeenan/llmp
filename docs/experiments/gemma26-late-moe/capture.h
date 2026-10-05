// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0
// External controller-only diagnostic. No floating operator is defined here.
#include <dlfcn.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

struct jitllm_late_item {
  const char* role;
  size_t bytes;
  void* pinned;
  bool queued;
  ggml_tensor descriptor;
};
struct jitllm_late_layer {
  jitllm_late_item items[7];
  const ggml_tensor* ids;
  const ggml_tensor* weights;
};
struct jitllm_late_probe {
  char directory[4096];
  jitllm_late_layer layers[2];
  unsigned calls64;
  bool selected;
  bool initialized;
};
// Intentionally no owner destructor: failures retain pinned storage to process exit.
static jitllm_late_probe* jitllm_late = nullptr;
static void jitllm_late_require(bool condition, const char* detail) {
  if (!condition) GGML_ABORT("late MoE probe: %s", detail);
}
extern "C" int jitllm_late_prepare(const char* directory) {
  jitllm_late_require(
      jitllm_late == nullptr && directory != nullptr && strnlen(directory, 4096) < 4000,
      "prepare bound");
  jitllm_late_require(mkdir(directory, 0700) == 0, "exclusive capture directory");
  auto* p = static_cast<jitllm_late_probe*>(calloc(1, sizeof(jitllm_late_probe)));
  jitllm_late_require(p != nullptr, "fixed metadata funding");
  jitllm_late = p;
  strcpy(p->directory, directory);
  CUDA_CHECK(cudaSetDevice(0));
  for (void* address : {reinterpret_cast<void*>(&ggml_cuda_op_topk_moe),
                        reinterpret_cast<void*>(&ggml_cuda_op_moe_weighted_reduction)}) {
    Dl_info info{};
    jitllm_late_require(dladdr(address, &info) != 0 && info.dli_fname &&
                            strcmp(info.dli_fname, "/app/libggml-cuda.so") == 0,
                        "original defining math library");
    fprintf(stderr, "JITLLM_LATE symbol=%s library=%s address=%p\n",
            info.dli_sname ? info.dli_sname : "unknown", info.dli_fname, address);
  }
  const char* roles[] = {"logits", "ids", "route-weights", "experts", "scales", "weights", "sum"};
  const size_t sizes[] = {128U * 64U * 4U, 8U * 64U * 4U, 8U * 64U * 4U,   2816U * 8U * 64U * 4U,
                          8U * 64U * 4U,   8U * 64U * 4U, 2816U * 64U * 4U};
  size_t total = 0;
  for (auto& layer : p->layers) {
    for (int i = 0; i < 7; ++i) {
      auto& item = layer.items[i];
      item.role = roles[i];
      item.bytes = sizes[i];
      total += sizes[i];
      jitllm_late_require(total <= 16U * 1024U * 1024U, "pinned budget");
      CUDA_CHECK(cudaHostAlloc(&item.pinned, item.bytes, cudaHostAllocDefault));
      const uint32_t sentinel = i == 1 ? UINT32_MAX : UINT32_C(0x7fc00001);
      for (size_t j = 0; j < item.bytes; j += 4)
        memcpy(static_cast<char*>(item.pinned) + j, &sentinel, 4);
    }
  }
  p->initialized = true;
  fprintf(stderr,
          "JITLLM_LATE prepared_pinned_bytes=%llu context=original_device0 "
          "lifetime=through_backend_retirement\n",
          (unsigned long long)total);
  return 0;
}
static void jitllm_late_graph(const ggml_cgraph* graph, bool use_graph, bool update) {
  jitllm_late_require(jitllm_late && jitllm_late->initialized, "prepare before graph");
  jitllm_late_require(graph && graph->n_nodes > 0 && graph->n_nodes <= 65536,
                      "bounded graph nodes");
  bool rows64 = false;
  for (int i = 0; i < graph->n_nodes; ++i) {
    const auto* t = graph->nodes[i];
    if (strcmp(t->name, "ffn_moe_probs-0") == 0 && t->op == GGML_OP_SOFT_MAX &&
        t->type == GGML_TYPE_F32 && t->ne[0] == 128 && t->ne[1] == 64 && t->ne[2] == 1 &&
        t->ne[3] == 1)
      rows64 = true;
  }
  if (rows64) ++jitllm_late->calls64;
  jitllm_late_require(jitllm_late->calls64 <= 2, "unexpected later64 graph replay");
  jitllm_late->selected = rows64 && jitllm_late->calls64 == 2;
  jitllm_late_require(!jitllm_late->selected || !use_graph || update,
                      "selected post-reset call must evaluate or capture");
  fprintf(stderr, "JITLLM_LATE graph64=%d occurrence64=%u selected=%d use_graph=%d update=%d\n",
          (int)rows64, jitllm_late->calls64, (int)jitllm_late->selected, (int)use_graph,
          (int)update);
}
static bool jitllm_late_packed(const ggml_tensor* t, int64_t a, int64_t b, int64_t c) {
  return t && t->data && t->type == GGML_TYPE_F32 && t->ne[0] == a && t->ne[1] == b &&
         t->ne[2] == c && t->ne[3] == 1 && t->nb[0] == 4 && t->nb[1] == size_t(a) * 4 &&
         t->nb[2] == size_t(a * b) * 4 && t->nb[3] == size_t(a * b * c) * 4;
}
static const ggml_tensor* jitllm_late_root(const ggml_tensor* t) {
  const ggml_tensor* seen[8]{};
  for (int i = 0; t && i < 8; ++i) {
    for (int j = 0; j < i; ++j) jitllm_late_require(t != seen[j], "view cycle");
    seen[i] = t;
    if (!t->view_src) return t;
    t = t->view_src;
  }
  jitllm_late_require(false, "bounded view chain");
  return nullptr;
}
static void jitllm_late_copy(ggml_backend_cuda_context& ctx, int layer, int role,
                             const ggml_tensor* tensor) {
  auto& item = jitllm_late->layers[layer].items[role];
  jitllm_late_require(ctx.device == 0 && tensor && tensor->data && !item.queued,
                      "one selected descriptor copy/device0");
  const auto start = reinterpret_cast<uintptr_t>(tensor->data);
  const size_t span = role == 1 ? 63U * 512U + 32U : item.bytes;
  jitllm_late_require(start <= UINTPTR_MAX - span, "source address bound");
  cudaStreamCaptureStatus capture;
  CUDA_CHECK(cudaStreamIsCapturing(ctx.stream(), &capture));
  jitllm_late_require(
      capture == cudaStreamCaptureStatusNone || capture == cudaStreamCaptureStatusActive,
      "legal original capture status");
  // With active capture these become stable D2H graph nodes. Never synchronize here.
  if (role == 1) {
    jitllm_late_require(tensor->type == GGML_TYPE_I32 && tensor->ne[0] == 8 &&
                            tensor->ne[1] == 64 && tensor->ne[2] == 1 && tensor->ne[3] == 1 &&
                            tensor->nb[0] == 4 && tensor->nb[1] == 512 && tensor->nb[2] == 32768 &&
                            tensor->nb[3] == 32768,
                        "selected IDs preserve full128 physical pitch");
    const auto* root = jitllm_late_root(tensor);
    jitllm_late_require(root->type == GGML_TYPE_I32 && root->ne[0] == 128 && root->ne[1] == 64 &&
                            root->ne[2] == 1 && root->ne[3] == 1 && root->data == tensor->data &&
                            root->nb[0] == 4 && root->nb[1] == 512 && root->nb[2] == 32768 &&
                            root->nb[3] == 32768,
                        "owned full128 ID root");
    CUDA_CHECK(cudaMemcpy2DAsync(item.pinned, 32, tensor->data, 512, 32, 64, cudaMemcpyDeviceToHost,
                                 ctx.stream()));
  } else {
    CUDA_CHECK(cudaMemcpyAsync(item.pinned, tensor->data, item.bytes, cudaMemcpyDeviceToHost,
                               ctx.stream()));
  }
  item.descriptor = *tensor;
  item.queued = true;
  fprintf(stderr, "JITLLM_LATE copy layer=%d role=%s capture=%d bytes=%llu\n", layer + 28,
          item.role, (int)capture, (unsigned long long)item.bytes);
}
static void jitllm_late_route(ggml_backend_cuda_context& ctx, const ggml_tensor* node,
                              const ggml_tensor* logits, const ggml_tensor* weights,
                              const ggml_tensor* ids, bool structural, bool shape, bool memory) {
  if (!jitllm_late->selected) return;
  int layer = strcmp(node->name, "ffn_moe_probs-28") == 0   ? 0
              : strcmp(node->name, "ffn_moe_probs-29") == 0 ? 1
                                                            : -1;
  if (layer < 0) return;
  jitllm_late_require(structural && shape && !memory && jitllm_late_packed(logits, 128, 64, 1) &&
                          jitllm_late_packed(weights, 1, 8, 64),
                      "actual late64 primitive route inputs");
  auto& l = jitllm_late->layers[layer];
  l.ids = ids;
  l.weights = weights;
  jitllm_late_copy(ctx, layer, 0, logits);
}
static void jitllm_late_after_node(ggml_backend_cuda_context& ctx, const ggml_tensor* node) {
  if (!jitllm_late->selected) return;
  for (int layer = 0; layer < 2; ++layer) {
    auto& l = jitllm_late->layers[layer];
    if (l.weights && node == jitllm_late_root(l.weights)) {
      jitllm_late_require(node->op == GGML_OP_DIV, "actual normalized route producer");
      jitllm_late_copy(ctx, layer, 1, l.ids);
      jitllm_late_copy(ctx, layer, 2, l.weights);
    }
  }
}
static int jitllm_late_reduction_layer(const ggml_tensor* node) {
  if (!jitllm_late->selected) return -1;
  return strcmp(node->name, "ffn_moe_down_scaled-28") == 0   ? 0
         : strcmp(node->name, "ffn_moe_down_scaled-29") == 0 ? 1
                                                             : -1;
}
static void jitllm_late_reduction_before(ggml_backend_cuda_context& ctx, int layer,
                                         const ggml_tensor* experts, const ggml_tensor* scale,
                                         const ggml_tensor* weights, const ggml_tensor* dst) {
  jitllm_late_require(
      jitllm_late_packed(experts, 2816, 8, 64) && jitllm_late_packed(scale, 1, 8, 64) &&
          jitllm_late_packed(weights, 1, 8, 64) && jitllm_late_packed(dst, 2816, 64, 1),
      "actual late64 scaled-reduction operands");
  jitllm_late_copy(ctx, layer, 3, experts);
  jitllm_late_copy(ctx, layer, 4, scale);
  jitllm_late_copy(ctx, layer, 5, weights);
}
static void jitllm_late_reduction_after(ggml_backend_cuda_context& ctx, int layer,
                                        const ggml_tensor* dst) {
  jitllm_late_copy(ctx, layer, 6, dst);
}
extern "C" int jitllm_late_finish() {
  jitllm_late_require(jitllm_late && jitllm_late->calls64 == 2, "exact warm/reset chronology");
  // Called after unchanged C API main explicitly retires context/model/backends.
  // Pinned destinations have outlived every recorded graph and potential replay.
  CUDA_CHECK(cudaDeviceSynchronize());
  char path[4096];
  snprintf(path, sizeof(path), "%s/descriptors.tsv", jitllm_late->directory);
  FILE* metadata = fopen(path, "wx");
  jitllm_late_require(metadata, "exclusive descriptors");
  for (int layer = 0; layer < 2; ++layer) {
    for (auto& item : jitllm_late->layers[layer].items) {
      jitllm_late_require(item.queued, "all selected role copies queued");
      for (size_t i = 0; i < item.bytes / 4; ++i) {
        if (strcmp(item.role, "ids") == 0) {
          const auto value = static_cast<int32_t*>(item.pinned)[i];
          jitllm_late_require(value >= 0 && value < 128, "initialized selected IDs");
        } else
          jitllm_late_require(std::isfinite(static_cast<float*>(item.pinned)[i]),
                              "finite complete snapshot/no unexecuted sentinel");
      }
      const auto& t = item.descriptor;
      fprintf(metadata, "%d\t%s\t%llu\t%d\t%p\t%lld:%lld:%lld:%lld\t%llu:%llu:%llu:%llu\n",
              layer + 28, item.role, (unsigned long long)item.bytes, (int)t.type, t.data,
              (long long)t.ne[0], (long long)t.ne[1], (long long)t.ne[2], (long long)t.ne[3],
              (unsigned long long)t.nb[0], (unsigned long long)t.nb[1], (unsigned long long)t.nb[2],
              (unsigned long long)t.nb[3]);
      snprintf(path, sizeof(path), "%s/layer-%d-%s.bin", jitllm_late->directory, layer + 28,
               item.role);
      FILE* output = fopen(path, "wx");
      jitllm_late_require(output, "exclusive snapshot");
      jitllm_late_require(fwrite(item.pinned, 1, item.bytes, output) == item.bytes &&
                              fflush(output) == 0 && fsync(fileno(output)) == 0,
                          "complete flushed snapshot");
      jitllm_late_require(fclose(output) == 0, "snapshot close");
    }
  }
  jitllm_late_require(!ferror(metadata) && fflush(metadata) == 0 && fsync(fileno(metadata)) == 0 &&
                          fclose(metadata) == 0,
                      "complete descriptor publication");
  for (auto& layer : jitllm_late->layers)
    for (auto& item : layer.items) CUDA_CHECK(cudaFreeHost(item.pinned));
  fprintf(stderr, "JITLLM_LATE retired_complete_roles=14 actual64calls=2\n");
  return 0;
}
