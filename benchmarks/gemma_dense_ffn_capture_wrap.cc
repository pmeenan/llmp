// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
#include <algorithm>
#include <cstring>
#include <fstream>

#include "engine/support.h"
#include "gemma_dense_ffn_capture.h"
#include "kernels/ggml/validate.h"
#include "model/gemma4.h"
#include "providers/device_runtime.h"
namespace llmp::benchmark {
namespace kg = kernels::ggml;
namespace fr = ffn_replay;
DenseFfnCapture* DenseFfnCapture::active = nullptr;
namespace {
auto Fail(std::string why, bool unknown = false) {
  return std::unexpected(
      kg::KernelFailure{.error = unknown ? kg::KernelError::kUnknown : kg::KernelError::kRejected,
                        .detail = std::move(why)});
}
}  // namespace
engine::Status DenseFfnCapture::Setup(engine::PagedNode& node, const std::filesystem::path& path,
                                      bool enabled) {
  node_ = &node;
  enabled_ = enabled;
  auto artifact = artifact::Artifact::Open(path);
  if (!artifact) return engine::support::Error(artifact.error().ToString());
  if (artifact->id() != "32c92e077a6816b54aa988e2dee61a3639c958fd510ea99e25f3621f10b2aa08")
    return engine::support::Error("wrong approved artifact identity");
  auto binding = model::BindGemma4(model::Gemma4_31B(), *artifact);
  if (!binding) return engine::support::Error(binding.error());
  const auto& layer = binding->layers[0];
  const std::array<const model::Gemma4Tensor*, 3> descriptors{&layer.gate, &layer.up, &layer.down};
  for (std::size_t i = 0; i < 3; ++i) {
    const auto& d = *descriptors[i];
    const auto& r = artifact->resources()[d.index];
    if (d.expert_array || r.bytes.value() != fr::kLogical[i + 1] ||
        r.readable.value() != fr::kBytes[i + 1])
      return engine::support::Error("closed layer0 quant metadata mismatch");
    indices_[i] = d.index;
  }
  constexpr std::array<std::uint64_t, 6> sizes{fr::kBytes[0], fr::kBytes[1],   fr::kBytes[2],
                                               fr::kBytes[3], fr::kHidden * 4, fr::kWidth * 4};
  for (std::size_t i = 0; i < sizes.size(); ++i) {
    std::vector<catalog::ExtentId> extents;
    auto made = node.Pinned(sizes[i], 0, extents);
    if (!made) return engine::support::Error(made.error());
    tensors_[i].pinned = *made;
  }
  return {};
}
void DenseFfnCapture::Observe(const engine::PagedWeights& weights, std::uint32_t resource,
                              std::uint64_t address) {
  if (weights.artifact().id() != "32c92e077a6816b54aa988e2dee61a3639c958fd510ea99e25f3621f10b2aa08")
    return;
  for (std::size_t i = 0; i < indices_.size(); ++i)
    if (resource == indices_[i]) weights_[i] = address;
}
std::expected<void, kg::KernelFailure> DenseFfnCapture::Copy(kg::LaunchContext& launch,
                                                             ggml_tensor* t, std::size_t index,
                                                             std::uint64_t bytes) {
  if (!node_ || !launch.UsesStream(node_->execution(), node_->stream(0)))
    return Fail("capture stream mismatch");
  auto* root = t;
  std::uint64_t relative = 0;
  unsigned depth = 0;
  while (root && root->view_src) {
    if (++depth > 32 || root->view_offs > UINT64_MAX - relative)
      return Fail("capture view chain bound");
    relative += root->view_offs;
    root = root->view_src;
  }
  if (!root || !root->data || !t->data) return Fail("capture unbound root");
  const auto root_bytes = ggml_nbytes(root);
  const bool weight = index >= 1 && index <= 3;
  const auto tail = weight ? fr::kBytes[index] - fr::kLogical[index] : 0;
  if (relative > root_bytes || bytes > root_bytes - relative + tail ||
      reinterpret_cast<std::uintptr_t>(t->data) !=
          reinterpret_cast<std::uintptr_t>(root->data) + relative ||
      (weight && (relative != 0 || root->view_src ||
                  reinterpret_cast<std::uintptr_t>(t->data) != weights_[index - 1] ||
                  root_bytes != fr::kLogical[index])))
    return Fail("capture extent does not match actual root/authenticated readable weight");
  auto& s = tensors_[index];
  std::copy_n(t->ne, 4, s.ne.begin());
  std::copy_n(t->nb, 4, s.nb.begin());
  std::copy_n(root->ne, 4, s.root_ne.begin());
  std::copy_n(root->nb, 4, s.root_nb.begin());
  s.bytes = bytes;
  s.type = static_cast<std::int32_t>(t->type);
  s.root_bytes = root_bytes;
  s.relative = relative;
  s.recorded = true;
  const auto stream = node_->execution().Submission(node_->stream(0));
  if (!stream) return Fail("capture submission refused");
  bool unknown = false;
  auto queued = launch.Run(base::Bytes(0), [&](ggml_backend_cuda_context&) {
    if (!providers::CopyAsync(*stream, s.pinned, t->data, bytes, providers::CopyKind::kDeviceToHost)
             .ok())
      unknown = true;
  });
  if (!queued) return queued;
  if (unknown) return Fail("capture copy uncertain", true);
  return {};
}
std::expected<void, kg::KernelFailure> DenseFfnCapture::Before(kg::LaunchContext& launch,
                                                               ggml_tensor* t) {
  if (!armed_ || !t || !t->src[0]) return {};
  std::size_t w = 0;
  while (w < weights_.size() && reinterpret_cast<std::uintptr_t>(t->src[0]->data) != weights_[w])
    ++w;
  if (w == weights_.size()) return {};
  if (t->op != GGML_OP_MUL_MAT || t->ne[1] != 1 || t->ne[2] != 1 || t->ne[3] != 1 || !t->src[1] ||
      t->src[1]->type != GGML_TYPE_F32 || !ggml_is_contiguous(t->src[1]) ||
      t->src[0]->type != (w == 2 ? GGML_TYPE_Q6_K : GGML_TYPE_Q4_K) ||
      t->src[0]->ne[0] != static_cast<std::int64_t>(w == 2 ? fr::kHidden : fr::kWidth) ||
      t->src[0]->ne[1] != static_cast<std::int64_t>(w == 2 ? fr::kWidth : fr::kHidden) ||
      t->src[0]->nb[1] != (w == 2 ? 17640U : 3024U))
    return Fail("capture is not exact scalar layer0 descriptor");
  if (seen_[w]) return Fail("layer0 product repeated inside final query");
  seen_[w] = true;
  std::memcpy(parameters_[w].data(), t->op_params, GGML_MAX_OP_PARAMS);
  if (w < 2) {
    if (shared_ && shared_ != t->src[1]->data) return Fail("gate/up input differs");
    shared_ = t->src[1]->data;
    if (auto r = Copy(launch, t->src[1], 0, fr::kBytes[0]); !r) return r;
  } else {
    auto* glu = t->src[1];
    if (!seen_[0] || !seen_[1]) return Fail("down precedes captured gate/up");
    if (auto r = kg::CheckGeGlu(glu); !r) return r;
    std::memcpy(parameters_[3].data(), glu->op_params, GGML_MAX_OP_PARAMS);
    if (!glu->src[0]->src[0] || !glu->src[1]->src[0] ||
        reinterpret_cast<std::uintptr_t>(glu->src[0]->src[0]->data) != weights_[0] ||
        reinterpret_cast<std::uintptr_t>(glu->src[1]->src[0]->data) != weights_[1])
      return Fail("GeGLU dependencies differ from exact gate/up");
    if (auto r = Copy(launch, glu, 4, fr::kHidden * 4); !r) return r;
  }
  return Copy(launch, t->src[0], w + 1, fr::kBytes[w + 1]);
}
std::expected<void, kg::KernelFailure> DenseFfnCapture::After(kg::LaunchContext& launch,
                                                              ggml_tensor* t) {
  if (armed_ && t && t->src[0] && reinterpret_cast<std::uintptr_t>(t->src[0]->data) == weights_[2])
    return Copy(launch, t, 5, fr::kWidth * 4);
  return {};
}
engine::Status DenseFfnCapture::Save(const std::filesystem::path& out) const {
  if (!enabled_) return {};
  if (!std::ranges::all_of(seen_, [](bool v) { return v; }))
    return engine::support::Error("final scalar layer0 not fully observed");
  constexpr std::array<const char*, 6> names{"input", "gate",       "up",
                                             "down",  "activation", "down-output"};
  std::ofstream metadata(out / "operands.json", std::ios::noreplace);
  metadata << "{\"origin\":\"native dense31 scalar "
              "layer0\",\"position\":67,\"graphs\":false,\"resource_indices\":["
           << indices_[0] << "," << indices_[1] << "," << indices_[2]
           << "],\"addresses_observed\":true,\"tensors\":{";
  for (std::size_t i = 0; i < tensors_.size(); ++i) {
    const auto& t = tensors_[i];
    if (!t.recorded) return engine::support::Error("capture missing payload");
    if (i == 0 || i >= 4)
      if (!fr::Finite({static_cast<float*>(t.pinned), static_cast<std::size_t>(t.bytes / 4)}))
        return engine::support::Error("nonfinite captured F32 payload");
    if (i) metadata << ',';
    metadata << '"' << names[i] << "\":{\"bytes\":" << t.bytes << ",\"type\":" << t.type
             << ",\"root_bytes\":" << t.root_bytes << ",\"relative_offset\":" << t.relative;
    for (unsigned a = 0; a < 4; ++a) {
      metadata << (a == 0   ? ",\"ne\":["
                   : a == 1 ? ",\"nb\":["
                   : a == 2 ? ",\"root_ne\":["
                            : ",\"root_nb\":[");
      for (unsigned d = 0; d < 4; ++d) {
        if (d) metadata << ',';
        if (a == 0) metadata << t.ne[d];
        if (a == 1) metadata << t.nb[d];
        if (a == 2) metadata << t.root_ne[d];
        if (a == 3) metadata << t.root_nb[d];
      }
      metadata << ']';
    }
    metadata << '}';
    std::ofstream f(out / (std::string(names[i]) + ".bin"), std::ios::binary | std::ios::noreplace);
    f.write(static_cast<const char*>(t.pinned), static_cast<std::streamsize>(t.bytes));
    f.flush();
    if (!f) return engine::support::Error("capture payload write");
  }
  metadata << "},\"product_order\":[\"gate\",\"up\",\"down\",\"geglu\"],\"op_params\":[";
  for (std::size_t i = 0; i < parameters_.size(); ++i) {
    if (i) metadata << ',';
    metadata << '[';
    for (std::size_t j = 0; j < parameters_[i].size(); ++j) {
      if (j) metadata << ',';
      metadata << parameters_[i][j];
    }
    metadata << ']';
  }
  metadata << "]}\n";
  std::ofstream params(out / "params.bin", std::ios::binary | std::ios::noreplace);
  params.write(reinterpret_cast<const char*>(parameters_.data()), sizeof(parameters_));
  params.flush();
  if (!params) return engine::support::Error("parameter write");
  metadata.flush();
  if (!metadata) return engine::support::Error("capture metadata write");
  return {};
}
}  // namespace llmp::benchmark
namespace kg = llmp::kernels::ggml;
std::expected<void, kg::KernelFailure> RealProduct(kg::LaunchContext&, ggml_tensor*) asm(
    "__real__ZN4llmp7kernels4ggml10MulMatVecQERNS1_13LaunchContextEP11ggml_tensor");
std::expected<void, kg::KernelFailure> WrappedProduct(kg::LaunchContext&, ggml_tensor*) asm(
    "__wrap__ZN4llmp7kernels4ggml10MulMatVecQERNS1_13LaunchContextEP11ggml_tensor");
std::expected<void, kg::KernelFailure> WrappedProduct(kg::LaunchContext& launch, ggml_tensor* t) {
  auto* capture = llmp::benchmark::DenseFfnCapture::active;
  if (capture)
    if (auto r = capture->Before(launch, t); !r) return r;
  if (auto r = RealProduct(launch, t); !r) return r;
  if (capture)
    if (auto r = capture->After(launch, t); !r) return r;
  return {};
}

std::uint64_t RealAddress(const llmp::engine::PagedWeights*, std::uint32_t) asm(
    "__real__ZNK4llmp6engine12PagedWeights16resource_addressEj");
std::uint64_t WrappedAddress(const llmp::engine::PagedWeights*, std::uint32_t) asm(
    "__wrap__ZNK4llmp6engine12PagedWeights16resource_addressEj");
std::uint64_t WrappedAddress(const llmp::engine::PagedWeights* weights, std::uint32_t resource) {
  auto address = RealAddress(weights, resource);
  if (auto* capture = llmp::benchmark::DenseFfnCapture::active)
    capture->Observe(*weights, resource, address);
  return address;
}
