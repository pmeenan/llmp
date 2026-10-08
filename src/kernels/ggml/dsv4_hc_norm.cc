// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "kernels/ggml/dsv4_hc_norm.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <initializer_list>
#include <optional>
#include <utility>

#include "kernels/ggml/llmp_ops.h"
#include "kernels/ggml/validate_ext.h"
#include "kernels/ggml/validate_util.h"

namespace llmp::kernels::ggml {
namespace {

constexpr std::int64_t kWidth = 4096;
constexpr std::int64_t kStreams = 4;
constexpr std::int64_t kFlat = kWidth * kStreams;
constexpr std::int64_t kMaxRows = 65535;

bool OnlySources(const ggml_tensor* tensor, std::size_t count) {
  for (std::size_t i = count; i < GGML_MAX_SRC; ++i) {
    if (tensor->src[i] != nullptr) {
      return false;
    }
  }
  return true;
}

bool Shaped(const ggml_tensor* t, ggml_type type, std::int64_t n0, std::int64_t n1,
            std::int64_t n2) {
  return t != nullptr && t->type == type && t->ne[0] == n0 && t->ne[1] == n1 && t->ne[2] == n2 &&
         t->ne[3] == 1 && detail::Packed(t);
}

bool Rows(std::int64_t rows) { return rows >= 1 && rows <= kMaxRows; }

}  // namespace

bool Dsv4HcNormF16Fits(const ggml_tensor* flat, float eps) {
  return flat != nullptr && Rows(flat->ne[1]) &&
         Shaped(flat, GGML_TYPE_F32, kFlat, flat->ne[1], 1) && std::isfinite(eps) && eps > 0;
}

std::expected<void, KernelFailure> CheckDsv4HcNormF16(const ggml_tensor* norm) {
  if (LlmpOpOf(norm) != LlmpOp::kDsv4HcNormF16 || !detail::Bound(norm) ||
      !detail::Bound(norm->src[0]) || !OnlySources(norm, 1)) {
    return detail::Rejected("not a bound DeepSeek HC F16 norm node");
  }
  const ggml_tensor* flat = norm->src[0];
  if (!Dsv4HcNormF16Fits(flat, LlmpOpEps(norm)) ||
      (!Shaped(norm, GGML_TYPE_F16, kFlat, flat->ne[1], 1) &&
       !Shaped(norm, GGML_TYPE_F32, kFlat, flat->ne[1], 1)) ||
      !detail::AllSane({norm, flat}) || !detail::AllCurrent({norm, flat}) ||
      !detail::Aligned(norm, 16) || !detail::Aligned(flat, 16) ||
      !detail::Disjoint(norm, flat, false)) {
    return detail::Rejected(
        "DeepSeek HC F16 norm requires canonical F32 [16384, rows] input and a disjoint F16 or "
        "F32 output");
  }
  return {};
}

std::optional<Dsv4HcPostNormF16Nodes> Dsv4HcPostNormF16At(std::span<ggml_tensor* const> graph,
                                                          std::size_t index) {
  if (index >= graph.size() || graph.size() - index < 3) {
    return std::nullopt;
  }
  ggml_tensor* post = graph[index];
  ggml_tensor* flat = graph[index + 1];
  ggml_tensor* norm = graph[index + 2];
  if (post->op != GGML_OP_DSV4_HC_POST || post->view_src != nullptr ||
      flat->op != GGML_OP_RESHAPE || flat->src[0] != post || flat->view_src != post ||
      flat->view_offs != 0 || LlmpOpOf(norm) != LlmpOp::kDsv4HcNormF16 || norm->src[0] != flat) {
    return std::nullopt;
  }
  const std::int64_t rows = post->ne[2];
  if (!Rows(rows) || !Shaped(post, GGML_TYPE_F32, kWidth, kStreams, rows) ||
      !Shaped(post->src[0], GGML_TYPE_F32, kWidth, rows, 1) ||
      !Shaped(post->src[1], GGML_TYPE_F32, kWidth, kStreams, rows) ||
      !Shaped(post->src[2], GGML_TYPE_F32, kStreams, rows, 1) ||
      !Shaped(post->src[3], GGML_TYPE_F32, kStreams, kStreams, rows) || !OnlySources(post, 4) ||
      !Dsv4HcNormF16Fits(flat, LlmpOpEps(norm))) {
    return std::nullopt;
  }
  return Dsv4HcPostNormF16Nodes{.post = post, .norm = norm};
}

std::expected<void, KernelFailure> CheckDsv4HcPostNormF16(const ggml_tensor* post,
                                                          const ggml_tensor* norm) {
  if (post == nullptr || norm == nullptr || norm->src[0] == nullptr ||
      norm->src[0]->src[0] != post) {
    return detail::Rejected("HC post/F16 norm requires the norm of the post's flat reshape");
  }
  if (auto checked = CheckHcPost(post); !checked) {
    return checked;
  }
  if (auto checked = CheckDsv4HcNormF16(norm); !checked) {
    return checked;
  }
  const ggml_tensor* flat = norm->src[0];
  if (flat->data != post->data || !detail::Disjoint(norm, post, false)) {
    return detail::Rejected("HC post/F16 norm requires separate outputs over the post");
  }
  for (std::size_t i = 0; i < 4; ++i) {
    // Both outputs are written while every HC input is still read.
    if (post->src[i] == nullptr || !detail::Disjoint(post, post->src[i], false) ||
        !detail::Disjoint(norm, post->src[i], false)) {
      return detail::Rejected("HC post/F16 norm outputs overlap an HC input");
    }
  }
  return {};
}

namespace {

// Whether any node other than `except` reads `tensor` directly or through a view.
bool ReadElsewhere(std::span<ggml_tensor* const> graph, const ggml_tensor* tensor,
                   const ggml_tensor* except) {
  for (const ggml_tensor* node : graph) {
    if (node == except || node == tensor) {
      continue;
    }
    if (node->view_src == tensor) {
      return true;
    }
    for (const ggml_tensor* src : node->src) {
      if (src == tensor) {
        return true;
      }
    }
  }
  return false;
}

bool ExpertsShape(const ggml_tensor* reduce, const ggml_tensor* add, std::int64_t rows) {
  const ggml_tensor* down = reduce->src[0];
  const ggml_tensor* weights = reduce->src[1];
  const ggml_tensor* shared = add->src[0] == reduce ? add->src[1] : add->src[0];
  return OnlySources(reduce, 2) && OnlySources(add, 2) &&
         Shaped(reduce, GGML_TYPE_F32, kWidth, rows, 1) &&
         Shaped(down, GGML_TYPE_F32, kWidth, 6, rows) && weights != nullptr &&
         weights->type == GGML_TYPE_F32 && ggml_nelements(weights) == 6 * rows &&
         detail::Packed(weights) && Shaped(add, GGML_TYPE_F32, kWidth, rows, 1) &&
         Shaped(shared, GGML_TYPE_F32, kWidth, rows, 1);
}

}  // namespace

std::optional<Dsv4HcPostExpertsNodes> Dsv4HcPostExpertsAt(std::span<ggml_tensor* const> graph,
                                                          std::size_t index,
                                                          std::size_t* add_index) {
  if (index >= graph.size()) {
    return std::nullopt;
  }
  ggml_tensor* reduce = graph[index];
  if (LlmpOpOf(reduce) != LlmpOp::kDsv4WeightedReduce || reduce->view_src != nullptr) {
    return std::nullopt;
  }
  for (std::size_t a = index + 1; a < graph.size(); ++a) {
    ggml_tensor* add = graph[a];
    if (add->op != GGML_OP_ADD || (add->src[0] != reduce && add->src[1] != reduce)) {
      continue;
    }
    const auto f = Dsv4HcPostNormF16At(graph, a + 1);
    if (!f || f->post->src[0] != add || add->view_src != nullptr ||
        !ExpertsShape(reduce, add, f->post->ne[2]) || ReadElsewhere(graph, reduce, add) ||
        ReadElsewhere(graph, add, f->post)) {
      return std::nullopt;
    }
    *add_index = a;
    return Dsv4HcPostExpertsNodes{.reduce = reduce, .add = add, .post = f->post, .norm = f->norm};
  }
  return std::nullopt;
}

std::expected<void, KernelFailure> CheckDsv4HcPostExpertsNormF16(const ggml_tensor* reduce,
                                                                 const ggml_tensor* add,
                                                                 const ggml_tensor* post,
                                                                 const ggml_tensor* norm) {
  if (reduce == nullptr || add == nullptr || post == nullptr || post->src[0] != add ||
      (add->src[0] != reduce && add->src[1] != reduce) ||
      LlmpOpOf(reduce) != LlmpOp::kDsv4WeightedReduce || !ExpertsShape(reduce, add, post->ne[2])) {
    return detail::Rejected("not the ordered expert reduction and shared add of an HC post");
  }
  if (auto checked = CheckDsv4HcPostNormF16(post, norm); !checked) {
    return checked;
  }
  const ggml_tensor* shared = add->src[0] == reduce ? add->src[1] : add->src[0];
  for (const ggml_tensor* input :
       std::initializer_list<const ggml_tensor*>{reduce->src[0], reduce->src[1], shared}) {
    if (!detail::Bound(input) || !detail::AllSane({input}) || !detail::AllCurrent({input}) ||
        !detail::Disjoint(post, input, false) || !detail::Disjoint(norm, input, false)) {
      return detail::Rejected("the expert sum's inputs overlap the post's outputs");
    }
  }
  return {};
}

bool Dsv4F16CopyFits(const ggml_tensor* x) {
  return x != nullptr && x->type == GGML_TYPE_F32 && x->ne[0] % 8 == 0 && x->ne[1] >= 1 &&
         x->ne[2] == 1 && x->ne[3] == 1 && detail::Packed(x) &&
         std::cmp_less_equal(x->ne[0] * x->ne[1], detail::kInt32Max);
}

std::expected<void, KernelFailure> CheckDsv4F16Copy(const ggml_tensor* copy) {
  if (LlmpOpOf(copy) != LlmpOp::kDsv4F16Copy || !detail::Bound(copy) ||
      !detail::Bound(copy->src[0]) || !OnlySources(copy, 1)) {
    return detail::Rejected("not a bound DeepSeek F16 copy node");
  }
  const ggml_tensor* x = copy->src[0];
  if (!Dsv4F16CopyFits(x) || !Shaped(copy, GGML_TYPE_F16, x->ne[0], x->ne[1], 1) ||
      !detail::AllSane({copy, x}) || !detail::AllCurrent({copy, x}) || !detail::Aligned(copy, 16) ||
      !detail::Aligned(x, 16) || !detail::Disjoint(copy, x, false)) {
    return detail::Rejected(
        "DeepSeek F16 copy requires packed F32 input and a disjoint F16 output");
  }
  return {};
}

}  // namespace llmp::kernels::ggml
