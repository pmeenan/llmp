// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0
#include <algorithm>
#include <cstring>
#include <fstream>
#include <string>

#include "engine/support.h"
#include "gemma_attention_capture.h"
#include "kernels/ggml/validate.h"
#include "providers/device_runtime.h"

namespace jitllm::benchmark {
namespace kg = kernels::ggml;
GemmaAttentionCapture* GemmaAttentionCapture::active = nullptr;
engine::Status GemmaAttentionCapture::Setup(engine::PagedNode& node, bool enabled) {
  enabled_ = enabled;
  node_ = &node;
  constexpr std::array<std::uint64_t, 4> sizes{32768, 2097152, 2097152, 16384};
  for (auto& owner : tensors_)
    for (std::size_t i = 0; i < owner.size(); ++i) {
      std::vector<catalog::ExtentId> extents;
      auto made = node.Pinned(sizes[i], 0, extents);
      if (!made) return engine::support::Error(made.error());
      owner[i].pinned = *made;
    }
  return {};
}
std::expected<void, kg::KernelFailure> GemmaAttentionCapture::Before(kg::LaunchContext& launch,
                                                                     ggml_tensor* node) {
  const auto fail = [](std::string message) {
    return std::unexpected(
        kg::KernelFailure{.error = kg::KernelError::kRejected, .detail = std::move(message)});
  };
  const auto index = calls_++;
  if (!enabled_ || index >= 4) return {};
  if (!node_ || !launch.UsesStream(node_->execution(), node_->stream(0)))
    return fail("capture is not on the authenticated target stream");
  if (auto checked = kg::CheckFlashAttnVec256(node); !checked) return checked;
  auto* q = node->src[0];
  auto* k = node->src[1];
  auto* v = node->src[2];
  auto* mask = node->src[3];
  if (q->ne[1] != 1 || q->ne[2] != 32 || q->nb[2] != 256 * sizeof(float) || k->ne[1] != 256 ||
      k->ne[2] != 16 || k->nb[1] != 8192 || k->nb[2] != 512 || v->nb[1] != 8192 ||
      v->nb[2] != 512 || mask->ne[1] < 1 || mask->ne[1] > 32 || mask->nb[1] != 512)
    return fail("capture is not the closed dense31 natural C4 first-local descriptor");
  const std::array<ggml_tensor*, 4> inputs{q, k, v, mask};
  const std::array<std::uint64_t, 4> bytes{32768, 2097152, 2097152,
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
engine::Status GemmaAttentionCapture::Save(const std::filesystem::path& out) const {
  if (!enabled_) return {};
  std::ofstream metadata(out / "operands.json", std::ios::noreplace);
  metadata << "{\"origin\":\"native dense31 first-local natural C4\",\"owners\":[";
  constexpr std::array<const char*, 4> names{"q", "k", "v", "mask"};
  for (std::size_t owner = 0; owner < tensors_.size(); ++owner) {
    if (owner != 0) metadata << ',';
    metadata << "{\"owner\":" << owner << ",\"position\":" << 67 + owner << ",\"tensors\":{ ";
    for (std::size_t i = 0; i < names.size(); ++i) {
      const auto& t = tensors_[owner][i];
      if (!t.recorded) return engine::support::Error("first-local capture hook was not observed");
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
}  // namespace jitllm::benchmark

namespace kg = jitllm::kernels::ggml;
std::expected<void, kg::KernelFailure> RealVec(kg::LaunchContext&, ggml_tensor*) asm(
    "__real__ZN6jitllm7kernels4ggml15FlashAttnVec256ERNS1_13LaunchContextEP11ggml_tensor");
std::expected<void, kg::KernelFailure> WrappedVec(kg::LaunchContext&, ggml_tensor*) asm(
    "__wrap__ZN6jitllm7kernels4ggml15FlashAttnVec256ERNS1_13LaunchContextEP11ggml_tensor");
std::expected<void, kg::KernelFailure> WrappedVec(kg::LaunchContext& launch, ggml_tensor* node) {
  if (auto* capture = jitllm::benchmark::GemmaAttentionCapture::active)
    if (auto copied = capture->Before(launch, node); !copied) return copied;
  return RealVec(launch, node);
}
