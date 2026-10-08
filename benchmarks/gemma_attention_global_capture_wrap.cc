// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
#include <algorithm>
#include <cstring>
#include <fstream>
#include <string>

#include "engine/support.h"
#include "gemma_attention_global_capture.h"
#include "kernels/ggml/validate.h"
#include "kernels/ggml/validate_ext.h"
#include "providers/device_runtime.h"

namespace llmp::benchmark {
namespace kg = kernels::ggml;
GemmaAttentionGlobalCapture* GemmaAttentionGlobalCapture::active = nullptr;
engine::Status GemmaAttentionGlobalCapture::Setup(engine::PagedNode& node, bool enabled) {
  enabled_ = enabled;
  node_ = &node;
  constexpr std::array<std::uint64_t, 4> sizes{65536, 1048576, 1048576, 16384};
  for (auto& owner : tensors_)
    for (std::size_t i = 0; i < owner.size(); ++i) {
      std::vector<catalog::ExtentId> extents;
      auto made = node.Pinned(sizes[i], 0, extents);
      if (!made) return engine::support::Error(made.error());
      owner[i].pinned = *made;
    }
  return {};
}
std::expected<void, kg::KernelFailure> GemmaAttentionGlobalCapture::Before(
    kg::LaunchContext& launch, ggml_tensor* node) {
  const auto fail = [](std::string message) {
    return std::unexpected(
        kg::KernelFailure{.error = kg::KernelError::kRejected, .detail = std::move(message)});
  };
  if (!enabled_) return {};
  // The generic MMA entry also executes all scalar prefill queries. Skip
  // those without consuming an owner index; first one-query D512 calls are
  // global layer 5 in the checked profile. BeginWave resets the four owners.
  if (!node || !node->src[0]) return fail("global capture missing query");
  if (node->src[0]->ne[0] != 512 || node->src[0]->ne[1] != 1) return {};
  const auto index = calls_++;
  if (index >= 4) return {};
  if (!node_ || !launch.UsesStream(node_->execution(), node_->stream(0)))
    return fail("capture is not on the authenticated target stream");
  if (auto checked = kg::CheckFlashAttnMma(node); !checked) return checked;
  auto* q = node->src[0];
  auto* k = node->src[1];
  auto* v = node->src[2];
  auto* mask = node->src[3];
  if (!mask) return fail("global capture missing mask");
  bool first_global = false;
  const auto* lineage = q;
  for (unsigned depth = 0; lineage != nullptr && depth < 32; ++depth) {
    if (std::strncmp(lineage->name, "blk.5.q_rope", 13) == 0) first_global = true;
    lineage = lineage->view_src;
  }
  if (!first_global) return fail("first one-query global source is not layer5 Q");
  if (q->type != GGML_TYPE_F32 || k->type != GGML_TYPE_F16 || v->type != GGML_TYPE_F16 ||
      mask->type != GGML_TYPE_F16 || q->ne[3] != 1 || k->ne[3] != 1 || v->ne[3] != 1 ||
      v->ne[1] != 256 || v->ne[2] != 4 || mask->ne[0] != 256 || mask->ne[2] != 1 ||
      mask->ne[3] != 1 || q->nb[0] != 4 || k->nb[0] != 2 || v->nb[0] != 2 || mask->nb[0] != 2 ||
      q->ne[1] != 1 || q->ne[2] != 32 || q->nb[2] != 512 * sizeof(float) || k->ne[1] != 256 ||
      k->ne[2] != 4 || k->nb[1] != 4096 || k->nb[2] != 1024 || v->nb[1] != 4096 ||
      v->nb[2] != 1024 || mask->ne[1] < 1 || mask->ne[1] > 32 || mask->nb[1] != 512)
    return fail("capture is not the closed dense31 natural C4 first-global descriptor");
  const std::array<ggml_tensor*, 4> inputs{q, k, v, mask};
  const std::array<std::uint64_t, 4> bytes{65536, 1048576, 1048576,
                                           static_cast<std::uint64_t>(mask->ne[1]) * 512};
  for (std::size_t i = 0; i < inputs.size(); ++i) {
    auto& saved = tensors_[index][i];
    std::array<std::int64_t, 4> ne{};
    std::array<std::uint64_t, 4> nb{};
    std::copy_n(inputs[i]->ne, 4, ne.begin());
    std::copy_n(inputs[i]->nb, 4, nb.begin());
    if (saved.recorded && (saved.ne != ne || saved.nb != nb || saved.bytes != bytes[i]))
      return fail("capture descriptor changed between eager and capture");
    saved.ne = ne;
    saved.nb = nb;
    saved.type = static_cast<std::int32_t>(inputs[i]->type);
    saved.bytes = bytes[i];
    auto* root = inputs[i];
    std::uint64_t offset = 0;
    std::uint32_t depth = 0;
    for (; root->view_src != nullptr; root = root->view_src) {
      if (++depth > 32 || root->view_offs > UINT64_MAX - offset)
        return fail("capture view chain exceeds its fixed bound");
      offset += root->view_offs;
    }
    const auto root_bytes = ggml_nbytes(root);
    if (offset > root_bytes || bytes[i] > root_bytes - offset ||
        reinterpret_cast<std::uintptr_t>(inputs[i]->data) !=
            reinterpret_cast<std::uintptr_t>(root->data) + offset)
      return fail("capture payload exceeds its actual bound root");
    saved.relative_offset = offset;
    saved.root_bytes = root_bytes;
    saved.view_depth = depth;
    std::copy_n(root->ne, 4, saved.root_ne.begin());
    std::copy_n(root->nb, 4, saved.root_nb.begin());
    saved.recorded = true;
  }
  std::memcpy(parameters_[index].data(), node->op_params, GGML_MAX_OP_PARAMS);
  bool unknown = false;
  const auto stream = node_->execution().Submission(node_->stream(0));
  if (!stream) return fail("capture stream submission refused");
  auto queued = launch.Run(base::Bytes(0), [&](ggml_backend_cuda_context&) {
    for (std::size_t i = 0; i < inputs.size(); ++i)
      if (!providers::CopyAsync(*stream, tensors_[index][i].pinned, inputs[i]->data, bytes[i],
                                providers::CopyKind::kDeviceToHost)
               .ok()) {
        unknown = true;
        break;
      }
  });
  if (!queued) return queued;
  if (unknown)
    return std::unexpected(kg::KernelFailure{.error = kg::KernelError::kUnknown,
                                             .detail = "capture operand copy is uncertain"});
  return {};
}
engine::Status GemmaAttentionGlobalCapture::Save(const std::filesystem::path& out) const {
  if (!enabled_) return {};
  std::ofstream metadata(out / "operands.json", std::ios::noreplace);
  metadata << "{\"origin\":\"native dense31 first-global natural C4\",\"layer\":5,\"owners\":[";
  constexpr std::array<const char*, 4> names{"q", "k", "v", "mask"};
  for (std::size_t owner = 0; owner < tensors_.size(); ++owner) {
    if (owner != 0) metadata << ',';
    metadata << "{\"owner\":" << owner << ",\"position\":" << 67 + owner << ",\"tensors\":{ ";
    for (std::size_t i = 0; i < names.size(); ++i) {
      const auto& t = tensors_[owner][i];
      if (!t.recorded) return engine::support::Error("first-global capture hook was not observed");
      if (i != 0) metadata << ',';
      metadata << '"' << names[i] << "\":{\"type\":" << t.type << ",\"bytes\":" << t.bytes;
      for (bool shape : {true, false}) {
        metadata << (shape ? ",\"ne\":[" : ",\"nb\":[");
        for (unsigned d = 0; d < 4; ++d) {
          if (d != 0) metadata << ',';
          if (shape)
            metadata << t.ne[d];
          else
            metadata << t.nb[d];
        }
        metadata << ']';
      }
      metadata << ",\"relative_offset\":" << t.relative_offset << ",\"view_depth\":" << t.view_depth
               << ",\"root_bytes\":" << t.root_bytes;
      for (bool shape : {true, false}) {
        metadata << (shape ? ",\"root_ne\":[" : ",\"root_nb\":[");
        for (unsigned d = 0; d < 4; ++d) {
          if (d != 0) metadata << ',';
          if (shape)
            metadata << t.root_ne[d];
          else
            metadata << t.root_nb[d];
        }
        metadata << ']';
      }
      metadata << '}';
      std::ofstream file(out / ("owner-" + std::to_string(owner) + "-" + names[i] + ".bin"),
                         std::ios::binary | std::ios::noreplace);
      file.write(static_cast<const char*>(t.pinned), static_cast<std::streamsize>(t.bytes));
      file.flush();
      if (!file) return engine::support::Error("capture operand write failed");
    }
    metadata << "},\"op_params\":[";
    for (std::size_t i = 0; i < parameters_[owner].size(); ++i) {
      if (i != 0) metadata << ',';
      metadata << parameters_[owner][i];
    }
    metadata << "]}";
  }
  metadata << "]}\n";
  metadata.flush();
  if (!metadata) return engine::support::Error("capture metadata write failed");
  return {};
}
}  // namespace llmp::benchmark

namespace kg = llmp::kernels::ggml;
std::expected<void, kg::KernelFailure> RealMma(kg::LaunchContext&, ggml_tensor*, bool) asm(
    "__real__ZN4llmp7kernels4ggml12FlashAttnMmaERNS1_13LaunchContextEP11ggml_tensorb");
std::expected<void, kg::KernelFailure> WrappedMma(kg::LaunchContext&, ggml_tensor*, bool) asm(
    "__wrap__ZN4llmp7kernels4ggml12FlashAttnMmaERNS1_13LaunchContextEP11ggml_tensorb");
std::expected<void, kg::KernelFailure> WrappedMma(kg::LaunchContext& launch, ggml_tensor* node,
                                                  bool wide_sparse) {
  if (auto* capture = llmp::benchmark::GemmaAttentionGlobalCapture::active)
    if (auto copied = capture->Before(launch, node); !copied) return copied;
  return RealMma(launch, node, wide_sparse);
}
