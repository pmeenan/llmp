// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
// Private first model screen, not a production serving/restore qualification.
#include <fcntl.h>
#include <gtest/gtest.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "artifact/artifact.h"
#include "base/bytes.h"
#include "base/sha256.h"
#include "components/step.h"
#include "dsv4_screen_source.h"
#include "engine/dsv4_runner.h"
#include "engine/expert_slab.h"
#include "engine/model_runner.h"
#include "engine/paged_node.h"
#include "engine/runner_resources.h"
#include "engine/support.h"
#include "format/tensor_format.h"
#include "kernels/ggml/dsv4_hc_norm.h"
#include "kernels/ggml/executor.h"
#include "kernels/ggml/llmp_ops.h"
#include "kernels/ggml/ops_ext.h"
#include "kernels/ggml/step_program.h"
#include "kernels/ggml/tensors.h"
#include "model/dsv4.h"
#include "planner/policy.h"
#include "platform/host_probe.h"
#include "providers/device_runtime.h"
#include "spec/mappings/dsv4.h"
#include "spec/stored.h"
#include "tokenizer_fixtures.h"

namespace llmp::engine {
// No mutation or independent address authority: the exact owner BindSlot uses.
struct Dsv4RunnerTestAccess {
  static const PagedWeights& TargetWeights(const Dsv4Runner& runner) { return runner.weights_; }
};
}  // namespace llmp::engine

namespace {
namespace ar = llmp::artifact;
namespace co = llmp::components;
namespace en = llmp::engine;
namespace kg = llmp::kernels::ggml;
namespace md = llmp::model;
namespace pl = llmp::planner;
namespace sp = llmp::spec;
namespace kn = llmp::kernels;
namespace ds = llmp::test_support::dsv4_screen;
namespace fs = std::filesystem;
using Error = std::unexpected<std::string>;
constexpr std::uint32_t kContext = 3072, kRows = 512, kPrefix = 2560, kDecode = 8;
constexpr std::uint64_t kWindow = 4U << 20U;
constexpr std::uint64_t kCsaWindow = 5U << 20U;
// Complete layer-2 Q/output (128 MiB), two HC-post boundaries (144 MiB),
// K/V/mask/sinks (< 4 MiB), and the existing complete CSA records (< 5 MiB).
// The pinned window is admitted before artifact reads. A separate extra
// 1152 MiB host payment covers all three arms' <= 384 MiB output-file pages.
constexpr std::uint64_t kLayerTwoWindow = 384U << 20U;
constexpr std::uint64_t kLayerTwoFiles = std::uint64_t{1152} << 20U;
static_assert(3 * kLayerTwoWindow <= kLayerTwoFiles);
// Includes the separately written complete half-Q diagnostic (32 MiB).
// Expert-post fusion leaves its sum unwritten: capture the actual complete
// down/route/shared inputs instead (at most 57 MiB), with no reconstruction.
static_assert((128U + 145U + 4U + 5U + 32U + 57U) * (1U << 20U) <= kLayerTwoWindow);
// Two whole current/ring planes, reads, three complete F32 outputs, norm
// weights, positions, worst-case I64 writes and 128 stored F16 rows.
constexpr std::uint64_t kCsaObservationBytes =
    (2U * 1024U * (512U + 8U) + 1024U + 3U * 512U * 128U + 512U + 128U) * 4U + 128U * 8U +
    128U * 512U * 2U;
static_assert(kCsaObservationBytes == 5185024 && kCsaObservationBytes <= kCsaWindow);
// The already-reviewed common compiler inventory is below 2,560 MiB at
// Graph's hard limits. This screen pays that entire allowance, plus a
// separate 1 GiB for artifact/spec metadata and 512 MiB for its original
// graph, closed-mask rewrite, inventories and full-head/file observations.
// The ordinary runtime separately charges retained plans, native graphs,
// activations, workspaces, weights and state. No diagnostic payload is a
// second unaccounted device allocation.
constexpr std::uint64_t kCompiler = std::uint64_t{4096} << 20U;
static_assert(llmp::ir::Graph::kMaxNodes == 65536 && llmp::ir::Graph::kMaxTensors == 131072);
static_assert(sizeof(llmp::ir::Node) <= 160 && sizeof(llmp::ir::TensorDesc) <= 160);
static_assert(sizeof(void*) == 8 && sizeof(std::string) <= 32);
static_assert(sizeof(llmp::base::Sha256) <= 256 && sizeof(en::LiveState::Range) <= 32);
constexpr std::string_view kArtifact =
    "m3-artifacts/8a355bfb27c90e1150fbd7fa62ea6e63f6bf34fcca33934e52d22773f1508234";
struct Target {
  fs::path root;
  ar::Artifact artifact;
  sp::Source source;
  llmp::base::Sha256Digest bound{};
  sp::ModelSpec spec;
  md::Dsv4Binding binding;
  md::StateLayoutDerived layout;
  std::vector<sp::SourceValue> values;
};
std::expected<std::shared_ptr<Target>, std::string> Open(const fs::path& root) {
  // Fixed-artifact screen: refuse expansion before either JSON document is
  // allocated. The captured artifact inventory is 14,492 and 4,994,976 bytes.
  std::error_code ec;
  const auto manifest_bytes = fs::file_size(root / "manifest.json", ec);
  if (ec || manifest_bytes > (16U << 10U))
    return Error("screen manifest exceeds paid metadata bound");
  const auto index_bytes = fs::file_size(root / "index.json", ec);
  if (ec || index_bytes > (5U << 20U)) return Error("screen index exceeds paid metadata bound");
  auto a = ar::Artifact::Open(root, {.expected_id = {}});
  if (!a) return Error(a.error().ToString());
  const auto& mapping = sp::mappings::Dsv4Mapping();
  auto src = sp::SourceOf(*a, &mapping);
  if (!src) return Error(src.error());
  auto values = sp::mappings::Dsv4V0Values(md::Dsv4Flash());
  if (!values) return Error(values.error());
  auto spec = sp::SynthesizeV0(*a, mapping, *values);
  if (!spec) return Error(spec.error());
  auto binding = md::BindDsv4(md::Dsv4Flash(), *a);
  if (!binding) return Error(binding.error());
  const std::array<std::string_view, 1> artifacts{a->id()};
  auto bound = sp::BoundDigest(*spec, artifacts);
  if (!bound) return Error(bound.error());
  if (spec->stages.size() != 1 || spec->stages[0].layers.size() != 43 || src->tensors.size() > 4096)
    return Error("screen source exceeds fixed paid profile");
  const auto& stage = spec->stages[0];
  std::uint64_t components = 0, parameters = 0, weights = 0;
  const auto include = [&](const sp::ComponentRef& c) {
    ++components;
    parameters += c.params.size();
    weights += c.weights.size();
  };
  if (stage.embedding) include(*stage.embedding);
  for (const auto& side : stage.side) include(side.source);
  if (stage.stream.mixing) include(*stage.stream.mixing);
  for (const auto& layer : stage.layers)
    for (const auto& c : layer.components) include(c);
  if (stage.heads.final_norm) include(*stage.heads.final_norm);
  if (stage.heads.output) include(*stage.heads.output);
  if (components > 320 || parameters > 8192 || weights > 4096 || !stage.heads.mtp.empty())
    return Error("screen component metadata exceeds common compiler bounds");
  if (auto checked = co::ValidateSpec(*spec, src->tensors); !checked) return Error(checked.error());
  auto layout = co::StageLayout(*spec, 0, kContext, kRows);
  if (!layout) return Error(layout.error());
  auto old = md::Dsv4State(md::Dsv4Flash(), kContext, kRows, md::Dsv4Window::kRing);
  if (!old || old->bytes != layout->bytes || old->tensors.size() != layout->items.size())
    return Error("legacy/shared state dimensions differ");
  for (std::size_t i = 0; i < layout->items.size(); ++i)
    if (old->tensors[i].offset != layout->items[i].offset ||
        old->tensors[i].bytes != layout->items[i].bytes)
      return Error("legacy/shared state item placement differs");
  return std::make_shared<Target>(Target{.root = root,
                                         .artifact = std::move(*a),
                                         .source = std::move(*src),
                                         .bound = *bound,
                                         .spec = std::move(*spec),
                                         .binding = std::move(*binding),
                                         .layout = std::move(*layout),
                                         .values = std::move(*values)});
}
std::vector<pl::StepShape> Script() {
  std::vector<pl::StepShape> out;
  for (std::uint32_t past = 0; past < kPrefix + kDecode;) {
    const std::uint32_t rows = past < kPrefix ? kRows : 1;
    out.push_back(
        {.stage = 0,
         .phase = rows == 1 ? pl::Phase::kDecode : pl::Phase::kPrefill,
         .segments = {{.slot = 0,
                       .rows = rows,
                       .past = past,
                       .read_cells = std::max(((md::Dsv4Flash().window + kRows + 255) / 256) * 256,
                                              ((past + rows + 255) / 256) * 256),
                       .outputs = 1}}});
    past += rows;
  }
  return out;
}
std::vector<std::int32_t> Tokens(std::uint32_t past, std::uint32_t rows) {
  std::vector<std::int32_t> result;
  for (std::uint32_t r = 0; r < rows; ++r)
    result.push_back(static_cast<std::int32_t>(100 + ((past + r) * 7919U + 17U) % 30000U));
  return result;
}
class KernelSource final : public en::PagedModel {
 public:
  explicit KernelSource(en::PagedNode& n) : resources(n, 0, 0) {}
  std::uint32_t stream() const override { return 0; }
  const llmp::catalog::Closure& fence_closure() const override { return empty_; }
  std::vector<llmp::catalog::ExtentId> managed_extents() const override { return {}; }
  en::Status Release() override {
    backend.reset();
    std::vector<std::string> errors;
    resources.Release(errors);
    return en::support::Joined(errors);
  }
  en::RunnerResources resources;
  std::unique_ptr<kg::GgmlBackend> backend;

 private:
  llmp::catalog::Closure empty_;
};
// Match Lowering::Run's descriptor reservation before TensorArena::Create.
// These native arrays/arenas are the fixed 65,536/32 MiB row of the paid
// common compiler inventory, independently of retained plan accounting.
en::Status NativeReservation(const kn::LowerInput& input) {
  const auto& p = input.plan.graph;
  const auto& g = input.token->step.graph;
  if (input.token->shape.stage != 0 || input.token->step.inputs.size() > 256 ||
      input.token->step.weights.size() > 4096 || input.token->step.state.size() > 4096 ||
      !input.token->step.saves.empty())
    return Error("screen bindings exceed paid single-stage inventory");
  std::uint64_t descriptors = p.tensor_count() + 16;
  for (const auto& applied : input.plan.applied)
    for (const auto replaced : applied.replaced) {
      if (replaced.v >= g.nodes().size()) return Error("screen malformed original replacement map");
      descriptors += g.node(replaced).nout;
    }
  using O = llmp::ir::Op;
  for (const auto& n : p.nodes()) {
    if (n.key == llmp::ir::Primitive(O::kRoute)) descriptors += 10;
    if (n.key == llmp::ir::Primitive(O::kFill) || n.key == llmp::ir::Primitive(O::kConvert))
      ++descriptors;
    if (n.key == llmp::ir::Primitive(O::kExpertReduce)) {
      if (n.nsrc == 0 || p.tensor(n.src[0]).ne(1) > 10)
        return Error("screen expert reduction bound");
      descriptors += 3 + 2 * p.tensor(n.src[0]).ne(1);
    }
    if (n.key.op == O::kFused || n.key.op == O::kMatMul || n.key.op == O::kWeight ||
        n.key.op == O::kGatedDeltaNet)
      descriptors += 8;
  }
  if (descriptors > 65536) return Error("screen native descriptors exceed paid bound");
  std::uint64_t traversal = 2;
  while (traversal / 2 < descriptors) traversal *= 2;
  if (descriptors * ggml_tensor_overhead() + traversal * sizeof(void*) + 4096 > (32U << 20U))
    return Error("screen native arena exceeds paid bound");
  return {};
}
struct Place {
  bool array = false;
  std::uint32_t index = 0;
};
class ScreenProbe : public kg::FirstAttentionObserver {
 public:
  virtual en::Status Save(const fs::path& dir) const = 0;
};
// This observer retains only three complete query/output rows, but every
// K/V cell, the complete mask and all sinks that those queries can read.
// Copies are queued immediately before/after the actual first attention on
// that same stream, then the first HC-post's four inputs and output at
// the same three rows. All 3,588,592 bytes fit the existing 4 MiB window.
// The enclosing completed RunStep/Chunk retires them.
class AttentionProbe final : public ScreenProbe {
 public:
  explicit AttentionProbe(void* host) : host_(static_cast<std::byte*>(host)) {}
  bool active() const override { return !done_; }
  std::expected<void, kg::KernelFailure> Observe(kg::LaunchContext& launch,
                                                 std::span<ggml_tensor* const> nodes,
                                                 std::string_view implementation,
                                                 bool after) override {
    if (done_) return {};
    if (attention_done_) return ObservePost(launch, nodes, implementation, after);
    if (node_ == nullptr) {
      if (after) return {};
      for (auto* node : nodes)
        if (node->op == GGML_OP_FLASH_ATTN_EXT) {
          if (node_ != nullptr) return Fail("multiple attention nodes in one step");
          node_ = node;
        }
      if (node_ == nullptr) return {};
      if (implementation != kg::kFlashAttnMmaName && implementation != kg::kFlashAttnMmaWideName)
        return Fail("unexpected first attention implementation");
      const auto* q = node_->src[0];
      const auto* k = node_->src[1];
      const auto* v = node_->src[2];
      const auto* mask = node_->src[3];
      const auto* sinks = node_->src[4];
      const auto shape = [](const ggml_tensor* t, ggml_type type, std::array<std::int64_t, 4> ne) {
        return t && t->data && t->type == type && std::equal(ne.begin(), ne.end(), t->ne);
      };
      if (!q || (q->type != GGML_TYPE_F32 && q->type != GGML_TYPE_F16) ||
          !shape(q, q->type, {512, 512, 64, 1}) ||
          !shape(node_, GGML_TYPE_F32, {512, 64, 512, 1}) || !k || !v || k->ne[1] < 512 ||
          k->ne[1] > 768 || !shape(k, GGML_TYPE_F16, {512, k->ne[1], 1, 1}) ||
          !shape(v, GGML_TYPE_F16, {512, k->ne[1], 1, 1}) ||
          !shape(mask, GGML_TYPE_F16, {k->ne[1], 512, 1, 1}) ||
          !shape(sinks, GGML_TYPE_F32, {64, 1, 1, 1}) || !ggml_is_contiguous(k) ||
          !ggml_is_contiguous(v) || !ggml_is_contiguous(mask) || !ggml_is_contiguous(sinks) ||
          !ggml_is_contiguous(node_) || q->nb[0] != ggml_type_size(q->type) ||
          q->nb[2] != 512 * q->nb[0] || q->nb[1] != 512 * 64 * q->nb[0])
        return Fail("first attention descriptor outside the bounded observation");
      auto plan = kg::PlanFlashAttnMma(launch, node_, implementation == kg::kFlashAttnMmaWideName);
      if (!plan) return std::unexpected(plan.error());
      plan_ = *plan;
      std::memcpy(params_.data(), node_->op_params, sizeof(node_->op_params));
      implementation_ = implementation;
      stream_ = launch.native_stream();
      // Reserve all destinations before submitting any copy. No device
      // allocation or graph use is introduced; each byte is in kWindow.
      const std::array<const ggml_tensor*, 6> tensors{q, k, v, mask, sinks, node_};
      constexpr std::array<const char*, 6> names{"q", "k", "v", "mask", "sinks", "out"};
      for (std::size_t i = 0; i < 6; ++i) {
        const std::size_t bytes = (i == 0 || i == 5)
                                      ? kSampleValues * ggml_type_size(tensors[i]->type)
                                      : ggml_nbytes(tensors[i]);
        if (bytes > kWindow - used_) return Fail("observation exceeds paid 4 MiB window");
        records_[i] = {.name = names[i],
                       .tensor = tensors[i],
                       .offset = used_,
                       .bytes = bytes,
                       .type = tensors[i]->type};
        std::copy_n(tensors[i]->ne, 4, records_[i].ne.begin());
        std::copy_n(tensors[i]->nb, 4, records_[i].nb.begin());
        used_ += bytes;
      }
      for (std::size_t i = 0; i < 5; ++i)
        if (auto copied = Copy(i); !copied) return copied;
      return {};
    }
    if (!after) return Fail("first attention before/after pairing changed");
    if (std::ranges::find(nodes, node_) == nodes.end() ||
        stream_.handle != launch.native_stream().handle || implementation_ != implementation)
      return Fail("first attention after callback does not match its original launch");
    if (auto copied = Copy(5); !copied) return copied;
    attention_done_ = true;
    return {};
  }
  en::Status Save(const fs::path& dir) const override {
    if (!done_) return Error("the first attention/HC-post was not observed completely");
    std::ofstream meta(dir / "attention.txt");
    meta << "implementation=" << implementation_ << " sparse=" << plan_.sparse
         << " columns=" << plan_.columns << " group=" << plan_.group << " blocks=" << plan_.blocks
         << " scratch=" << plan_.scratch << " sampled_query_rows=0,127,511\nparams";
    for (const auto value : params_) meta << ' ' << value;
    meta << '\n'
         << "hc_post_implementation=" << post_implementation_ << " sampled_token_rows=0,127,511\n";
    for (const auto& r : records_) {
      const auto bytes = std::span<const std::byte>(host_ + r.offset, r.bytes);
      llmp::base::Sha256 hash;
      hash.Update(bytes);
      meta << r.name << " type=" << ggml_type_name(r.type) << " ne";
      for (const auto n : r.ne) meta << ' ' << n;
      meta << " nb";
      for (const auto n : r.nb) meta << ' ' << n;
      meta << " bytes=" << r.bytes << " sha256=" << llmp::base::ToHex(hash.Finish()) << '\n';
      std::ofstream data(dir / (std::string("attention-") + r.name + ".bin"), std::ios::binary);
      data.write(reinterpret_cast<const char*>(bytes.data()),
                 static_cast<std::streamsize>(bytes.size()));
      if (!data) return Error("attention observation write failed");
    }
    // The existing MMA attention consumes half queries even when its
    // descriptor is F32. Preserve original bytes above as well as this
    // explicit, separately named diagnostic round-trip (no tensor mutation).
    std::ofstream effective(dir / "attention-q-half.bin", std::ios::binary);
    const auto& q = records_[0];
    std::array<ggml_fp16_t, 1024> half{};
    for (std::size_t at = 0; at < kSampleValues; at += half.size()) {
      const auto n = std::min(half.size(), kSampleValues - at);
      for (std::size_t j = 0; j < n; ++j) {
        if (q.type == GGML_TYPE_F16) {
          std::memcpy(&half[j], host_ + q.offset + (at + j) * sizeof(ggml_fp16_t),
                      sizeof(ggml_fp16_t));
        } else {
          float f = 0;
          std::memcpy(&f, host_ + q.offset + (at + j) * sizeof(float), sizeof(float));
          half[j] = ggml_fp32_to_fp16(f);
        }
      }
      effective.write(reinterpret_cast<const char*>(half.data()),
                      static_cast<std::streamsize>(n * sizeof(ggml_fp16_t)));
    }
    return meta && effective ? en::Status{}
                             : en::Status(Error("attention metadata/effective query write failed"));
  }

 private:
  static constexpr std::size_t kSampleValues = 3U * 512U * 64U;
  struct Record {
    const char* name = nullptr;
    const ggml_tensor* tensor = nullptr;
    std::size_t offset = 0, bytes = 0;
    ggml_type type = GGML_TYPE_F32;
    std::array<std::int64_t, 4> ne{};
    std::array<std::size_t, 4> nb{};
  };
  static std::unexpected<kg::KernelFailure> Fail(std::string text) {
    return std::unexpected(
        kg::KernelFailure{.error = kg::KernelError::kUnknown, .detail = std::move(text)});
  }
  std::expected<void, kg::KernelFailure> ObservePost(kg::LaunchContext& launch,
                                                     std::span<ggml_tensor* const> nodes,
                                                     std::string_view implementation, bool after) {
    if (post_node_ == nullptr) {
      if (after) return {};
      for (auto* node : nodes)
        if (node->op == GGML_OP_DSV4_HC_POST) {
          if (post_node_ != nullptr) return Fail("multiple HC-post nodes in one step");
          post_node_ = node;
        }
      if (post_node_ == nullptr) return {};
      if (implementation != kg::kHcPostName && implementation != kg::kDsv4HcPostNormF16Name)
        return Fail("unexpected first attention HC-post implementation");
      if (stream_.handle != launch.native_stream().handle)
        return Fail("first attention/HC-post changed streams");
      constexpr std::array<std::array<std::int64_t, 4>, 5> shapes{{{4096, 512, 1, 1},
                                                                   {4096, 4, 512, 1},
                                                                   {4, 512, 1, 1},
                                                                   {4, 4, 512, 1},
                                                                   {4096, 4, 512, 1}}};
      constexpr std::array<int, 5> row_axes{1, 2, 1, 2, 2};
      constexpr std::array<const char*, 5> names{"hc-cur", "hc-residual", "hc-post", "hc-comb",
                                                 "hc-out"};
      // Reserve and authenticate every source/destination before any copy.
      // Check the actual row strides, including fused packed carry tails.
      for (std::size_t j = 0; j < 5; ++j) {
        const auto* t = j == 4 ? post_node_ : post_node_->src[j];
        if (!t || !t->data || t->type != GGML_TYPE_F32 ||
            !std::equal(shapes[j].begin(), shapes[j].end(), t->ne) || t->nb[0] != sizeof(float))
          return Fail("first HC-post descriptor outside the bounded observation");
        const auto axis = static_cast<std::size_t>(row_axes[j]);
        std::size_t row_bytes = static_cast<std::size_t>(t->ne[0]) * sizeof(float);
        if (axis == 2) {
          if (t->nb[1] != row_bytes) return Fail("first HC-post has a pitched inner row");
          row_bytes *= static_cast<std::size_t>(t->ne[1]);
        }
        const auto bytes = 3 * row_bytes;
        if (t->nb[axis] < row_bytes || bytes > kWindow - used_)
          return Fail("first HC-post stride/window bound");
        auto& r = records_[j + 6];
        r = {.name = names[j], .tensor = t, .offset = used_, .bytes = bytes, .type = t->type};
        std::copy_n(t->ne, 4, r.ne.begin());
        std::copy_n(t->nb, 4, r.nb.begin());
        used_ += bytes;
      }
      post_implementation_ = implementation;
      for (std::size_t i = 6; i < 10; ++i)
        if (auto copied = Copy(i); !copied) return copied;
      return {};
    }
    if (!after || std::ranges::find(nodes, post_node_) == nodes.end() ||
        stream_.handle != launch.native_stream().handle || post_implementation_ != implementation)
      return Fail("first HC-post before/after pairing changed");
    if (auto copied = Copy(10); !copied) return copied;
    done_ = true;
    return {};
  }
  std::expected<void, kg::KernelFailure> Copy(std::size_t i) {
    const auto& r = records_[i];
    const auto copy = [&](std::size_t dst, std::size_t src, std::size_t bytes) {
      const auto copied = llmp::providers::CopyAsync(
          stream_, host_ + r.offset + dst, static_cast<const std::byte*>(r.tensor->data) + src,
          bytes, llmp::providers::CopyKind::kDeviceToHost);
      return copied.ok() ? std::expected<void, kg::KernelFailure>{}
                         : std::expected<void, kg::KernelFailure>{Fail(copied.text())};
    };
    if (i > 0 && i < 5) return copy(0, 0, r.bytes);
    constexpr std::array<std::size_t, 11> row_axes{1, 0, 0, 0, 0, 2, 1, 2, 1, 2, 2};
    constexpr std::array<std::size_t, 3> rows{0, 127, 511};
    for (std::size_t j = 0; j < rows.size(); ++j)
      if (auto copied = copy(j * r.bytes / 3, rows[j] * r.tensor->nb[row_axes[i]], r.bytes / 3);
          !copied)
        return copied;
    return {};
  }
  std::byte* host_;
  ggml_tensor* node_ = nullptr;
  ggml_tensor* post_node_ = nullptr;
  llmp::providers::NativeStream stream_;
  kg::FlashAttnMmaPlan plan_;
  std::array<std::int32_t, GGML_MAX_OP_PARAMS / sizeof(std::int32_t)> params_{};
  std::string_view implementation_, post_implementation_;
  std::array<Record, 11> records_{};
  std::size_t used_ = 0;
  bool attention_done_ = false, done_ = false;
};

// Mutually exclusive with AttentionProbe: the same paid window observes
// the first flat normalization, its actual mix product and the HC fold.
// Full F32 weights plus seven records occupy 2,212,176 bytes. Capturing
// normalized output and the actual product input separately also detects
// an intervening mutation. No new allocation or synchronization is used.
class InitialHcProbe final : public ScreenProbe {
 public:
  explicit InitialHcProbe(void* host) : host_(static_cast<std::byte*>(host)) {}
  bool active() const override { return stage_ != 3; }
  std::expected<void, kg::KernelFailure> Observe(kg::LaunchContext& launch,
                                                 std::span<ggml_tensor* const> nodes,
                                                 std::string_view implementation,
                                                 bool after) override {
    if (!active()) return {};
    if (stage_ == 2 && (primitive_fold_ ||
                        std::ranges::any_of(nodes, [&](const auto* n) { return FoldTerm(n, 0); })))
      return PrimitiveFold(launch, nodes, implementation, after);
    if (pending_ == nullptr) {
      if (after) return {};
      ggml_tensor* found = nullptr;
      for (auto* n : nodes) {
        const bool match = stage_ == 0   ? (n->op == GGML_OP_RMS_NORM ||
                                            kg::LlmpOpOf(n) == kg::LlmpOp::kDsv4HcNormF16) &&
                                               Shape(n, {16384, 512, 1, 1})
                           : stage_ == 1 ? n->op == GGML_OP_MUL_MAT && n->src[1] == norm_
                                         : n->op == GGML_OP_DSV4_HC_PRE;
        if (!match) continue;
        if (found) return Fail("multiple initial HC nodes in a step");
        found = n;
      }
      if (!found) return {};
      pending_ = found;
      implementations_[stage_] = implementation;
      std::memcpy(params_[stage_].data(), found->op_params, sizeof(found->op_params));
      if (stage_ == 0) {
        if (implementation != kg::kRmsNormName && implementation != kg::kDsv4HcNormF16Name)
          return Fail("initial HC normalization implementation");
        if (!Shape(found->src[0], {16384, 512, 1, 1}))
          return Fail("initial HC normalization source geometry");
        stream_ = launch.native_stream();
        flat_ = found->src[0];
        norm_ = found;
        norm_eps_ = found->op == GGML_OP_RMS_NORM ? 0.0F : kg::LlmpOpEps(found);
        if (found->op == GGML_OP_RMS_NORM)
          std::memcpy(&norm_eps_, found->op_params, sizeof(norm_eps_));
        if (auto r = RecordAt(0, "flat", flat_, false); !r) return r;
        if (auto r = RecordAt(1, "normalized", found, false); !r) return r;
        return Copy(0);
      }
      if (stream_.handle != launch.native_stream().handle)
        return Fail("initial HC observation changed stream");
      if (stage_ == 1) {
        if (!Shape(found, {24, 512, 1, 1}) || !Shape(found->src[0], {16384, 24, 1, 1}) ||
            !Shape(found->src[1], {16384, 512, 1, 1}))
          return Fail("initial HC product geometry");
        if (auto r = RecordAt(2, "product-input", found->src[1], false); !r) return r;
        if (auto r = RecordAt(3, "fn", found->src[0], true); !r) return r;
        if (auto r = RecordAt(4, "mixes", found, false); !r) return r;
        if (auto r = Copy(2); !r) return r;
        return Copy(3);
      }
      if (implementation != kg::kHcPreName || !Shape(found, {4096, 512, 1, 1}) ||
          !Shape(found->src[0], {4096, 4, 512, 1}) || !Shape(found->src[1], {4, 512, 1, 1}) ||
          found->src[0]->data != flat_->data)
        return Fail("initial HC fold geometry or residual identity");
      if (auto r = RecordAt(5, "pre", found->src[1], false); !r) return r;
      if (auto r = RecordAt(6, "folded", found, false); !r) return r;
      return Copy(5);
    }
    if (!after || std::ranges::find(nodes, pending_) == nodes.end() ||
        implementations_[stage_] != implementation ||
        stream_.handle != launch.native_stream().handle)
      return Fail("initial HC before/after pairing changed");
    constexpr std::array<std::size_t, 3> outputs{1, 4, 6};
    if (auto copied = Copy(outputs[stage_]); !copied) return copied;
    pending_ = nullptr;
    ++stage_;
    return {};
  }
  en::Status Save(const fs::path& dir) const override {
    std::ofstream meta(dir / "initial-hc.txt");
    meta << "completed_stages=" << stage_ << " primitive_fold_steps=" << fold_step_ << '\n';
    meta << "sampled_token_rows=0,127,511 full_weight=1 bytes=" << used_
         << " norm_epsilon=" << norm_eps_ << '\n';
    constexpr std::array<const char*, 3> stages{"normalize", "product", "fold"};
    for (std::size_t i = 0; i < stages.size(); ++i) {
      meta << stages[i] << " implementation=" << implementations_[i] << " params";
      for (const auto value : params_[i]) meta << ' ' << value;
      meta << '\n';
    }
    for (std::size_t i = 0; i < fold_step_; ++i) {
      meta << "fold_step=" << i << " implementation=" << fold_implementations_[i] << " params";
      for (const auto value : fold_params_[i]) meta << ' ' << value;
      meta << '\n';
    }
    for (const auto& r : records_) {
      if (!r.copied) continue;
      const auto bytes = std::span<const std::byte>(host_ + r.offset, r.bytes);
      llmp::base::Sha256 hash;
      hash.Update(bytes);
      meta << r.name << " type=f32 ne";
      for (const auto n : r.ne) meta << ' ' << n;
      meta << " nb";
      for (const auto n : r.nb) meta << ' ' << n;
      meta << " bytes=" << r.bytes << " sha256=" << llmp::base::ToHex(hash.Finish()) << '\n';
      std::ofstream data(dir / (std::string("initial-hc-") + r.name + ".bin"), std::ios::binary);
      data.write(reinterpret_cast<const char*>(bytes.data()),
                 static_cast<std::streamsize>(bytes.size()));
      if (!data) return Error("initial HC observation write failed");
    }
    if (!meta) return Error("initial HC metadata write failed");
    if (active())
      return Error("initial HC observation incomplete at stage " + std::to_string(stage_) +
                   " fold step " + std::to_string(fold_step_));
    return {};
  }

 private:
  struct Record {
    const char* name = nullptr;
    const ggml_tensor* tensor = nullptr;
    std::size_t offset = 0, bytes = 0;
    bool full = false, copied = false;
    std::array<std::int64_t, 4> ne{};
    std::array<std::size_t, 4> nb{};
  };
  static bool Shape(const ggml_tensor* t, std::array<std::int64_t, 4> ne) {
    return t && t->data && t->type == GGML_TYPE_F32 && ggml_is_contiguous(t) &&
           std::equal(ne.begin(), ne.end(), t->ne);
  }
  // The unfused shared HcPre decomposition is four weighted stream views
  // and three left-associated Adds (ir/decompose.cc). Authenticate its exact
  // descriptors/ordering; capture gates before the first Mul can consume them.
  bool FoldTerm(const ggml_tensor* n, std::size_t h) const {
    if (!Shape(n, {4096, 512, 1, 1}) || n->op != GGML_OP_MUL || n->view_src || !flat_ ||
        flat_->op != GGML_OP_RESHAPE)
      return false;
    const auto* x = n->src[0];
    const auto* w = n->src[1];
    if (!x || !w || x->op != GGML_OP_VIEW || w->op != GGML_OP_VIEW || x->src[0] != flat_->src[0] ||
        !Shape(x->src[0], {4096, 4, 512, 1}) || x->src[0]->data != flat_->data ||
        !Shape(w->src[0], {4, 512, 1, 1}) || (pre_ && w->src[0] != pre_))
      return false;
    const auto view = [](const ggml_tensor* t, const ggml_tensor* src,
                         std::array<std::int64_t, 4> ne, std::array<std::size_t, 4> nb,
                         std::size_t offset) {
      const auto* root = src->view_src ? src->view_src : src;
      const auto root_offset = src->view_src ? src->view_offs : 0;
      std::size_t parameter = 0;
      std::memcpy(&parameter, t->op_params, sizeof(parameter));
      return t->type == GGML_TYPE_F32 && std::equal(ne.begin(), ne.end(), t->ne) &&
             std::equal(nb.begin(), nb.end(), t->nb) && parameter == offset &&
             t->view_src == root && t->view_offs == root_offset + offset &&
             t->data == static_cast<const std::byte*>(src->data) + offset;
    };
    return view(x, x->src[0], {4096, 512, 1, 1}, {4, 65536, 33554432, 33554432}, h * 16384) &&
           view(w, w->src[0], {1, 512, 1, 1}, {4, 16, 8192, 8192}, h * 4);
  }
  std::expected<void, kg::KernelFailure> PrimitiveFold(kg::LaunchContext& launch,
                                                       std::span<ggml_tensor* const> nodes,
                                                       std::string_view implementation,
                                                       bool after) {
    if (stream_.handle != launch.native_stream().handle)
      return Fail("initial HC primitive fold changed stream");
    // Mul0, Mul1, Add1, Mul2, Add2, Mul3, Add3.
    const bool multiply = fold_step_ == 0 || fold_step_ % 2 == 1;
    const std::size_t h = (fold_step_ + 1) / 2;
    if (!pending_) {
      if (after) return {};
      ggml_tensor* found = nullptr;
      for (auto* n : nodes) {
        const bool match = multiply ? FoldTerm(n, h)
                                    : Shape(n, {4096, 512, 1, 1}) && n->op == GGML_OP_ADD &&
                                          !n->view_src && n->src[0] == accumulator_ &&
                                          n->src[1] == term_;
        if (!match) continue;
        if (found) return Fail("multiple initial HC primitive fold nodes");
        found = n;
      }
      if (!found) return {};
      if (implementation != (multiply ? kg::kMulName : kg::kAddName))
        return Fail("initial HC primitive fold implementation");
      primitive_fold_ = true;
      pending_ = found;
      fold_implementations_[fold_step_] = implementation;
      std::memcpy(fold_params_[fold_step_].data(), found->op_params, sizeof(found->op_params));
      if (fold_step_ == 0) {
        pre_ = found->src[1]->src[0];
        if (auto r = RecordAt(5, "pre", pre_, false); !r) return r;
        return Copy(5);
      }
      if (fold_step_ == 6) {
        implementations_[2] = implementation;
        params_[2] = fold_params_[6];
        return RecordAt(6, "folded", found, false);
      }
      return {};
    }
    if (!after || std::ranges::find(nodes, pending_) == nodes.end() ||
        fold_implementations_[fold_step_] != implementation)
      return Fail("initial HC primitive fold before/after pairing changed");
    if (fold_step_ == 0 || !multiply)
      accumulator_ = pending_;
    else
      term_ = pending_;
    if (fold_step_ == 6) {
      if (auto r = Copy(6); !r) return r;
      ++stage_;
    }
    pending_ = nullptr;
    ++fold_step_;
    return {};
  }
  static std::unexpected<kg::KernelFailure> Fail(std::string text) {
    return std::unexpected(
        kg::KernelFailure{.error = kg::KernelError::kUnknown, .detail = std::move(text)});
  }
  std::expected<void, kg::KernelFailure> RecordAt(std::size_t at, const char* name,
                                                  const ggml_tensor* t, bool full) {
    const auto bytes =
        full ? ggml_nbytes(t) : 3 * static_cast<std::size_t>(t->ne[0]) * sizeof(float);
    if (bytes > kWindow - used_) return Fail("initial HC observation exceeds paid window");
    auto& r = records_[at];
    r = {.name = name, .tensor = t, .offset = used_, .bytes = bytes, .full = full};
    std::copy_n(t->ne, 4, r.ne.begin());
    std::copy_n(t->nb, 4, r.nb.begin());
    used_ += bytes;
    return {};
  }
  std::expected<void, kg::KernelFailure> Copy(std::size_t at) {
    auto& r = records_[at];
    const auto copy = [&](std::size_t dst, std::size_t src, std::size_t bytes) {
      const auto copied = llmp::providers::CopyAsync(
          stream_, host_ + r.offset + dst, static_cast<const std::byte*>(r.tensor->data) + src,
          bytes, llmp::providers::CopyKind::kDeviceToHost);
      return copied.ok() ? std::expected<void, kg::KernelFailure>{}
                         : std::expected<void, kg::KernelFailure>{Fail(copied.text())};
    };
    if (r.full) {
      if (auto copied = copy(0, 0, r.bytes); !copied) return copied;
      r.copied = true;
      return {};
    }
    constexpr std::array<std::size_t, 3> rows{0, 127, 511};
    for (std::size_t j = 0; j < rows.size(); ++j)
      if (auto copied = copy(j * r.bytes / 3, rows[j] * r.nb[1], r.bytes / 3); !copied)
        return copied;
    r.copied = true;
    return {};
  }
  std::byte* host_;
  ggml_tensor *pending_ = nullptr, *norm_ = nullptr, *flat_ = nullptr;
  llmp::providers::NativeStream stream_;
  std::array<Record, 7> records_{};
  std::array<std::string_view, 3> implementations_{};
  std::array<std::array<std::int32_t, GGML_MAX_OP_PARAMS / sizeof(std::int32_t)>, 3> params_{};
  std::size_t used_ = 0, stage_ = 0;
  ggml_tensor *pre_ = nullptr, *accumulator_ = nullptr, *term_ = nullptr;
  std::array<std::string_view, 7> fold_implementations_{};
  std::array<std::array<std::int32_t, GGML_MAX_OP_PARAMS / sizeof(std::int32_t)>, 7> fold_params_{};
  std::size_t fold_step_ = 0;
  bool primitive_fold_ = false;
  float norm_eps_ = 0;
};
static_assert(sizeof(InitialHcProbe) <= 4096);  // Same existing paid compiler reservation.

static_assert(sizeof(AttentionProbe) <= 4096);  // Inside the existing 4 GiB compiler allowance.

// The first CSA compressor of the 512-row step: every source operand and
// every output at each arithmetic boundary. Its one paid 5 MiB window
// replaces the ordinary 4 MiB window; no extra synchronization or allocation.
class CsaProbe final : public ScreenProbe {
 public:
  CsaProbe(void* host, const Target& target, std::array<std::uint64_t, 3> state,
           std::array<std::uint64_t, 4> weights)
      : host_(static_cast<std::byte*>(host)), target_(target), state_(state), weights_(weights) {}

  std::uint64_t norm_address() const { return Address(weight_nodes_[3]); }
  bool active() const override { return stage_ != 4; }
  std::expected<void, kg::KernelFailure> Observe(kg::LaunchContext& launch,
                                                 std::span<ggml_tensor* const> nodes,
                                                 std::string_view implementation,
                                                 bool after) override {
    if (!active()) return {};
    if (!pending_) {
      if (after) return {};
      ggml_tensor* found = nullptr;
      for (auto* n : nodes) {
        const bool compressor = kg::LlmpOpOf(n) == kg::LlmpOp::kDsv4Compress && n->ne[0] == 512;
        if (stage_ == 0 && compressor && n->src[0] && n->src[1]) {
          if (seen_ < candidates_.size())
            candidates_[seen_] = {Address(n->src[0]), Address(n->src[1]), Address(n)};
          ++seen_;
        }
        const bool match =
            stage_ == 0
                ? compressor && Address(n->src[0]) == state_[0] && Address(n->src[1]) == state_[1]
            : stage_ == 1 ? n->op == GGML_OP_RMS_NORM && n->src[0] == pooled_
            : stage_ == 2 ? n->op == GGML_OP_ROPE && n->src[0] == normalized_
                          : n->op == GGML_OP_SET_ROWS && n->src[0] &&
                                n->src[0]->op == GGML_OP_VIEW && n->src[0]->src[0] == rotated_;
        if (!match) continue;
        if (found) return Fail("multiple CSA boundary nodes in one launch");
        found = n;
      }
      if (!found) return {};
      if (stage_ != 0 && stream_.handle != launch.native_stream().handle)
        return Fail("CSA observation changed stream");
      pending_ = found;
      implementations_[stage_] = implementation;
      std::memcpy(params_[stage_].data(), found->op_params, sizeof(found->op_params));
      if (stage_ == 0) {
        if (implementation != kg::kDsv4CompressName || kg::LlmpOpInt(found, 0) != 4 ||
            kg::LlmpOpInt(found, 1) != 1 || !Shape(found, GGML_TYPE_F32, {512, 1, 128, 1}) ||
            !Shape(found->src[0], GGML_TYPE_F32, {1024, 8, 1, 1}) ||
            !Shape(found->src[1], GGML_TYPE_F32, {1024, 8, 1, 1}) ||
            !Shape(found->src[2], GGML_TYPE_F32, {1024, 512, 1, 1}) ||
            !Shape(found->src[3], GGML_TYPE_F32, {1024, 512, 1, 1}) ||
            !Shape(found->src[4], GGML_TYPE_I32, {1024, 1, 1, 1}))
          return Fail("first CSA compressor geometry/recipe differs");
        stream_ = launch.native_stream();
        pooled_ = found;
        auto* kv = Unwrap(found->src[2]);
        auto* score = Unwrap(found->src[3]);
        if (!kv || kv->op != GGML_OP_MUL_MAT || !score || score->op != GGML_OP_ADD)
          return Fail("layer-2 compressor projection producer differs");
        auto* gate = Unwrap(score->src[0]);
        auto* ape = Unwrap(score->src[1]);
        if (!gate || gate->op != GGML_OP_MUL_MAT || !ape || ape->op != GGML_OP_GET_ROWS)
          return Fail("layer-2 compressor gate/APE producers differ");
        weight_nodes_[0] = Unwrap(kv->src[0]);
        weight_nodes_[1] = Unwrap(gate->src[0]);
        weight_nodes_[2] = Unwrap(ape->src[0]);
        constexpr std::array<const char*, 6> names{"ring-kv",       "ring-score", "current-kv",
                                                   "current-score", "read-ids",   "pooled"};
        for (std::size_t i = 0; i < names.size(); ++i)
          if (auto r = RecordAt(i, names[i], i == 5 ? found : found->src[i]); !r) return r;
        for (std::size_t i = 0; i < 5; ++i)
          if (auto r = Copy(i); !r) return r;
        return {};
      }
      if (stage_ == 1) {
        if (implementation != kg::kRmsNormMulFused && implementation != kg::kRmsNormMulUnfused)
          return Fail("first CSA normalization is not the complete RMS/multiply recipe");
        for (auto* n : nodes)
          if (n->op == GGML_OP_MUL && n->src[0] == found) {
            if (normalized_) return Fail("multiple CSA weighted normalizations");
            normalized_ = n;
          }
        if (!Shape(found, GGML_TYPE_F32, {512, 1, 128, 1}, false) ||
            !Shape(normalized_, GGML_TYPE_F32, {512, 1, 128, 1}) ||
            !Shape(normalized_->src[1], GGML_TYPE_F32, {512, 1, 1, 1}))
          return Fail("first CSA normalization geometry differs");
        weight_nodes_[3] = Unwrap(normalized_->src[1]);
        const auto& layer = target_.binding.layers.at(2);
        const std::array<std::uint32_t, 4> ids{layer.comp_kv.index, layer.comp_gate.index,
                                               layer.comp_ape.index, layer.comp_norm.index};
        const auto& norm = target_.artifact.resources()[ids[3]];
        const auto norm_address = Address(weight_nodes_[3]);
        if (norm_address < norm.offset.value())
          return Fail("layer-2 norm group address underflows");
        const auto group_base = norm_address - norm.offset.value();
        for (std::size_t i = 0; i < ids.size(); ++i) {
          const auto& resource = target_.artifact.resources()[ids[i]];
          if (!weight_nodes_[i] || weight_nodes_[i]->op != GGML_OP_NONE ||
              resource.group != norm.group ||
              Address(weight_nodes_[i]) != group_base + resource.offset.value() ||
              ggml_nbytes(weight_nodes_[i]) != resource.bytes.value() ||
              (weights_[i] != 0 && Address(weight_nodes_[i]) != weights_[i]))
            return Fail("layer-2 compressor actual weight binding differs from the artifact");
        }
        if (auto r = RecordAt(6, "norm-weight", normalized_->src[1]); !r) return r;
        if (auto r = RecordAt(7, "normalized", normalized_); !r) return r;
        return Copy(6);
      }
      if (stage_ == 2) {
        if (implementation != kg::kRopeExtName || !Shape(found, GGML_TYPE_F32, {512, 1, 128, 1}) ||
            !Shape(found->src[1], GGML_TYPE_I32, {128, 1, 1, 1}))
          return Fail("first CSA rotation geometry/recipe differs");
        rotated_ = found;
        if (auto r = RecordAt(8, "positions", found->src[1]); !r) return r;
        if (auto r = RecordAt(9, "rotated", found); !r) return r;
        return Copy(8);
      }
      if (implementation != kg::kSetRowsName || Address(found) != state_[2] ||
          !Shape(found, GGML_TYPE_F16, {512, 768, 1, 1}) ||
          !Shape(found->src[0], GGML_TYPE_F32, {512, 128, 1, 1}) ||
          found->src[0]->data != rotated_->data || !found->src[1] ||
          (found->src[1]->type != GGML_TYPE_I32 && found->src[1]->type != GGML_TYPE_I64) ||
          !Shape(found->src[1], found->src[1]->type, {128, 1, 1, 1}))
        return Fail("first CSA store geometry/recipe differs");
      if (auto r = RecordAt(10, "write-ids", found->src[1]); !r) return r;
      if (auto r = RecordAt(11, "stored", found, 128U * 512U * sizeof(ggml_fp16_t)); !r) return r;
      return Copy(10);
    }
    if (!after || std::ranges::find(nodes, pending_) == nodes.end() ||
        implementations_[stage_] != implementation ||
        stream_.handle != launch.native_stream().handle)
      return Fail("CSA before/after pairing changed");
    constexpr std::array<std::size_t, 4> outputs{5, 7, 9, 11};
    if (auto r = Copy(outputs[stage_]); !r) return r;
    pending_ = nullptr;
    ++stage_;
    return {};
  }
  en::Status Save(const fs::path& dir) const override {
    std::ofstream meta(dir / "csa.txt");
    meta << "completed_stages=" << stage_ << " bytes=" << used_
         << " current_rows=all_512 ring_rows=all_8 complete_overlap_blocks=all_128\n";
    meta << "architecture_layer=2 expected_ring_kv=" << state_[0]
         << " expected_ring_score=" << state_[1] << " expected_destination=" << state_[2]
         << " native_candidates_seen=" << seen_ << '\n';
    for (std::size_t i = 0; i < std::min(seen_, candidates_.size()); ++i)
      meta << "native_candidate=" << i << " ring_kv=" << candidates_[i][0]
           << " ring_score=" << candidates_[i][1] << " pooled=" << candidates_[i][2] << '\n';
    const auto& layer = target_.binding.layers.at(2);
    const std::array<std::uint32_t, 4> ids{layer.comp_kv.index, layer.comp_gate.index,
                                           layer.comp_ape.index, layer.comp_norm.index};
    for (std::size_t i = 0; i < ids.size(); ++i) {
      const auto& r = target_.artifact.resources()[ids[i]];
      meta << "weight=" << r.name << " resource=" << ids[i] << " group=" << r.group
           << " offset=" << r.offset.value() << " bytes=" << r.bytes.value()
           << " actual_address=" << Address(weight_nodes_[i])
           << " shared_bound_address=" << weights_[i] << '\n';
    }
    for (std::size_t i = 0; i < implementations_.size(); ++i) {
      meta << "stage=" << i << " implementation=" << implementations_[i] << " params";
      for (const auto value : params_[i]) meta << ' ' << value;
      meta << '\n';
    }
    for (const auto& r : records_) {
      if (!r.copied) continue;
      const auto bytes = std::span<const std::byte>(host_ + r.offset, r.bytes);
      llmp::base::Sha256 hash;
      hash.Update(bytes);
      meta << r.name << " type=" << ggml_type_name(r.type) << " ne";
      for (const auto n : r.ne) meta << ' ' << n;
      meta << " nb";
      for (const auto n : r.nb) meta << ' ' << n;
      meta << " bytes=" << r.bytes << " sha256=" << llmp::base::ToHex(hash.Finish()) << '\n';
      std::ofstream data(dir / (std::string("csa-") + r.name + ".bin"), std::ios::binary);
      data.write(reinterpret_cast<const char*>(bytes.data()),
                 static_cast<std::streamsize>(bytes.size()));
      if (!data) return Error("CSA observation write failed");
    }
    if (!meta) return Error("CSA metadata write failed");
    if (active())
      return Error("authoritative layer-2 CSA observation incomplete at stage " +
                   std::to_string(stage_));
    // Independently read from this fixed prepared artifact in DS19. Its full
    // F32 norm payload identifies blk.2, unlike DS19's shared blk.4 capture.
    llmp::base::Sha256 norm_hash;
    norm_hash.Update(std::span<const std::byte>(host_ + records_[6].offset, records_[6].bytes));
    if (llmp::base::ToHex(norm_hash.Finish()) !=
        "519556ec61b92644fe0f240adcede224b21325fb527bacfd361fe2b640d8becb")
      return Error("captured layer-2 norm bytes differ from the authenticated artifact resource");
    // The complete current/ring rows cover these exact overlap tuples. Check
    // only after the enclosing execution has completed all operand copies.
    const auto integer = [&](std::size_t record, std::size_t index) -> std::int64_t {
      const auto& r = records_[record];
      if (r.type == GGML_TYPE_I64) {
        std::int64_t value = 0;
        std::memcpy(&value, host_ + r.offset + index * sizeof(value), sizeof(value));
        return value;
      }
      std::int32_t value = 0;
      std::memcpy(&value, host_ + r.offset + index * sizeof(value), sizeof(value));
      return value;
    };
    for (std::int64_t b = 0; b < 128; ++b) {
      const auto block = static_cast<std::size_t>(b);
      if (integer(8, block) != 4 * b || integer(10, block) != b)
        return Error("CSA positions/write IDs differ from the first-step coordinate contract");
      for (std::int64_t j = 0; j < 4; ++j) {
        const auto at = static_cast<std::size_t>(4 * b + j);
        if (integer(4, at) != (b == 0 ? 520 : 8 + 4 * (b - 1) + j) ||
            integer(4, 512 + at) != 8 + 4 * b + j)
          return Error("CSA reads differ from the captured overlap coordinate contract");
      }
    }
    return {};
  }

 private:
  struct Record {
    const char* name = nullptr;
    const ggml_tensor* tensor = nullptr;
    std::size_t offset = 0, bytes = 0;
    ggml_type type = GGML_TYPE_F32;
    bool copied = false;
    std::array<std::int64_t, 4> ne{};
    std::array<std::size_t, 4> nb{};
  };
  static bool Shape(const ggml_tensor* t, ggml_type type, std::array<std::int64_t, 4> ne,
                    bool bound = true) {
    // A fused RMS/multiply descriptor need not materialize the intermediate
    // RMS output. Every copied operand/result still requires bound storage.
    return t && (!bound || t->data) && t->type == type && ggml_is_contiguous(t) &&
           std::equal(ne.begin(), ne.end(), t->ne);
  }
  static std::unexpected<kg::KernelFailure> Fail(std::string text) {
    return std::unexpected(
        kg::KernelFailure{.error = kg::KernelError::kUnknown, .detail = std::move(text)});
  }
  std::expected<void, kg::KernelFailure> RecordAt(std::size_t at, const char* name,
                                                  const ggml_tensor* t, std::size_t prefix = 0) {
    const auto bytes = prefix ? prefix : ggml_nbytes(t);
    if (used_ > kCsaWindow || bytes > kCsaWindow - used_ || bytes > ggml_nbytes(t))
      return Fail("CSA observation exceeds its paid window or source extent");
    auto& r = records_[at];
    r = {.name = name, .tensor = t, .offset = used_, .bytes = bytes, .type = t->type};
    std::copy_n(t->ne, 4, r.ne.begin());
    std::copy_n(t->nb, 4, r.nb.begin());
    used_ += bytes;
    return {};
  }
  std::expected<void, kg::KernelFailure> Copy(std::size_t at) {
    auto& r = records_[at];
    const auto copy = [&](std::size_t dst, std::size_t src, std::size_t bytes) {
      const auto copied = llmp::providers::CopyAsync(
          stream_, host_ + r.offset + dst, static_cast<const std::byte*>(r.tensor->data) + src,
          bytes, llmp::providers::CopyKind::kDeviceToHost);
      return copied.ok() ? std::expected<void, kg::KernelFailure>{}
                         : std::expected<void, kg::KernelFailure>{Fail(copied.text())};
    };
    if (auto copied = copy(0, 0, r.bytes); !copied) return copied;
    r.copied = true;
    return {};
  }
  static std::uint64_t Address(const ggml_tensor* t) {
    return t ? reinterpret_cast<std::uintptr_t>(t->data) : 0;
  }
  static ggml_tensor* Unwrap(ggml_tensor* t) {
    for (unsigned depth = 0; t && depth < 16; ++depth) {
      if (t->op != GGML_OP_VIEW && t->op != GGML_OP_RESHAPE && t->op != GGML_OP_CONT) return t;
      t = t->src[0];
    }
    return nullptr;
  }
  std::byte* host_;
  const Target& target_;
  std::array<std::uint64_t, 3> state_;
  std::array<std::uint64_t, 4> weights_;
  std::array<ggml_tensor*, 4> weight_nodes_{};
  std::array<std::array<std::uint64_t, 3>, 8> candidates_{};
  std::size_t seen_ = 0;
  ggml_tensor *pending_ = nullptr, *pooled_ = nullptr, *normalized_ = nullptr, *rotated_ = nullptr;
  llmp::providers::NativeStream stream_;
  std::array<Record, 12> records_{};
  std::array<std::string_view, 4> implementations_{};
  std::array<std::array<std::int32_t, GGML_MAX_OP_PARAMS / sizeof(std::int32_t)>, 4> params_{};
  std::size_t used_ = 0, stage_ = 0;
};
static_assert(sizeof(CsaProbe) <= 4096);  // Same existing paid compiler reservation.

// Full layer boundary capture. Layer2 keeps its CSA resource witness.
// Layer3 is selected by both actual raw/HCA state addresses and its complete
// sink payload is authenticated against the frozen artifact after the step and its copies complete.
class LayerBoundaryProbe final : public ScreenProbe {
 public:
  LayerBoundaryProbe(void* host, const Target& target, std::array<std::uint64_t, 3> state,
                     std::array<std::uint64_t, 4> weights, std::uint64_t raw,
                     std::uint32_t layer = 2, std::uint64_t sink = 0)
      : host_(static_cast<std::byte*>(host)),
        target_(target),
        state_(state),
        raw_(raw),
        layer_(layer),
        sink_(sink),
        csa_(host, target, state, weights) {}
  bool active() const override { return !done_; }
  std::expected<void, kg::KernelFailure> Observe(kg::LaunchContext& launch,
                                                 std::span<ggml_tensor* const> nodes,
                                                 std::string_view implementation,
                                                 bool after) override {
    if (done_) return {};
    if (layer_ == 2 && csa_.active()) return csa_.Observe(launch, nodes, implementation, after);
    if (attention_done_) return ObservePost(launch, nodes, implementation, after);
    if (node_ == nullptr) {
      if (after) return {};
      for (auto* node : nodes)
        if (node->op == GGML_OP_FLASH_ATTN_EXT &&
            (layer_ == 2 ? Address(node->src[4]) ==
                               Weight(target_.binding.layers.at(layer_).attn_sinks.index)
                         : MatchesBank(node))) {
          if (node_ != nullptr) return Fail("multiple attention nodes in one step");
          node_ = node;
        }
      if (node_ == nullptr) return {};
      if (auto authenticated = AuthenticateAttention(); !authenticated) return authenticated;
      if (implementation != kg::kFlashAttnMmaName && implementation != kg::kFlashAttnMmaWideName)
        return Fail("unexpected observed-layer attention implementation");
      const auto* q = node_->src[0];
      const auto* k = node_->src[1];
      const auto* v = node_->src[2];
      const auto* mask = node_->src[3];
      const auto* sinks = node_->src[4];
      const auto shape = [](const ggml_tensor* t, ggml_type type, std::array<std::int64_t, 4> ne) {
        return t && t->data && t->type == type && std::equal(ne.begin(), ne.end(), t->ne);
      };
      if (!q || (q->type != GGML_TYPE_F32 && q->type != GGML_TYPE_F16) ||
          !shape(q, q->type, {512, 512, 64, 1}) ||
          !shape(node_, GGML_TYPE_F32, {512, 64, 512, 1}) || !k || !v || k->ne[1] != 1024 ||
          !shape(k, GGML_TYPE_F16, {512, k->ne[1], 1, 1}) ||
          !shape(v, GGML_TYPE_F16, {512, k->ne[1], 1, 1}) ||
          !shape(mask, GGML_TYPE_F16, {k->ne[1], 512, 1, 1}) ||
          !shape(sinks, GGML_TYPE_F32, {64, 1, 1, 1}) || !ggml_is_contiguous(k) ||
          !ggml_is_contiguous(v) || !ggml_is_contiguous(mask) || !ggml_is_contiguous(sinks) ||
          !ggml_is_contiguous(node_) || q->nb[0] != ggml_type_size(q->type) ||
          q->nb[2] != 512 * q->nb[0] || q->nb[1] != 512 * 64 * q->nb[0])
        return Fail("observed-layer attention descriptor outside the bounded observation");
      auto plan = kg::PlanFlashAttnMma(launch, node_, implementation == kg::kFlashAttnMmaWideName);
      if (!plan) return std::unexpected(plan.error());
      plan_ = *plan;
      std::memcpy(params_.data(), node_->op_params, sizeof(node_->op_params));
      implementation_ = implementation;
      stream_ = launch.native_stream();
      // Reserve all destinations before submitting any copy. No device
      // allocation or graph use is introduced; each byte is in kLayerTwoWindow.
      const std::array<const ggml_tensor*, 6> tensors{q, k, v, mask, sinks, node_};
      constexpr std::array<const char*, 6> names{"q", "k", "v", "mask", "sinks", "out"};
      for (std::size_t i = 0; i < 6; ++i) {
        const std::size_t bytes = (i == 0 || i == 5)
                                      ? kFullValues * ggml_type_size(tensors[i]->type)
                                      : ggml_nbytes(tensors[i]);
        if (bytes > kLayerTwoWindow - used_)
          return Fail("observation exceeds paid observed-layer window");
        records_[i] = {.name = names[i],
                       .tensor = tensors[i],
                       .offset = used_,
                       .bytes = bytes,
                       .type = tensors[i]->type};
        std::copy_n(tensors[i]->ne, 4, records_[i].ne.begin());
        std::copy_n(tensors[i]->nb, 4, records_[i].nb.begin());
        used_ += bytes;
      }
      for (std::size_t i = 0; i < 5; ++i)
        if (auto copied = Copy(i); !copied) return copied;
      return {};
    }
    if (!after) return Fail("observed-layer attention before/after pairing changed");
    if (std::ranges::find(nodes, node_) == nodes.end() ||
        stream_.handle != launch.native_stream().handle || implementation_ != implementation)
      return Fail("observed-layer attention after callback does not match its original launch");
    if (auto copied = Copy(5); !copied) return copied;
    attention_done_ = true;
    return {};
  }
  en::Status Save(const fs::path& dir) const override {
    if (layer_ == 2) {
      if (auto saved = csa_.Save(dir); !saved) return saved;
    } else {
      // This 256-byte resource was independently read from the authenticated
      // artifact shard at file offset 2816159744 before this diagnostic.
      const auto& sink = records_[4];
      llmp::base::Sha256 hash;
      hash.Update(std::span<const std::byte>(host_ + sink.offset, sink.bytes));
      if (sink.bytes != 256 ||
          llmp::base::ToHex(hash.Finish()) !=
              "f7589f284a13d175fab5e1f346f52ae1e57cd3bbfbc433d1b2c5fdba874f8097")
        return Error("layer-3 sink payload differs from its authenticated artifact resource");
    }
    if (!done_)
      return Error(
          "the observed-layer attention/two HC-post boundaries was not observed completely");
    std::ofstream meta(dir / (Prefix() + ".txt"));
    meta << "implementation=" << implementation_ << " sparse=" << plan_.sparse
         << " columns=" << plan_.columns << " group=" << plan_.group << " blocks=" << plan_.blocks
         << " scratch=" << plan_.scratch << " architecture_layer=" << layer_
         << " query_rows=all_512\nparams";
    for (const auto value : params_) meta << ' ' << value;
    meta << '\n'
         << "hc_post_implementation=" << post_implementations_[0]
         << " ffn_post_implementation=" << post_implementations_[1] << " token_rows=all_512\n"
         << "raw_address=" << raw_ << " compressed_address=" << state_[2]
         << " sink_address=" << Address(node_->src[4])
         << " expected_sink_address=" << Weight(target_.binding.layers.at(layer_).attn_sinks.index)
         << " first_post_address=" << Address(first_post_)
         << " ffn_cur_materialized=" << !experts_elided_ << '\n';
    if (experts_elided_) {
      std::ofstream elided(dir / (Prefix() + "-ffn-cur-elided.txt"));
      elided << "llmp.dsv4.hc_post_experts_norm_f16 forms the sum in registers; "
                "ffn-cur is not a readable GPU output. Actual down/route/shared are retained.\n";
      if (!elided) return Error("expert post elision marker write failed");
    }
    for (const auto& r : records_) {
      if (!r.tensor) continue;
      const auto bytes = std::span<const std::byte>(host_ + r.offset, r.bytes);
      llmp::base::Sha256 hash;
      hash.Update(bytes);
      meta << r.name << " type=" << ggml_type_name(r.type) << " ne";
      for (const auto n : r.ne) meta << ' ' << n;
      meta << " nb";
      for (const auto n : r.nb) meta << ' ' << n;
      meta << " bytes=" << r.bytes << " sha256=" << llmp::base::ToHex(hash.Finish()) << '\n';
      std::ofstream data(dir / ((Prefix() + "-") + r.name + ".bin"), std::ios::binary);
      data.write(reinterpret_cast<const char*>(bytes.data()),
                 static_cast<std::streamsize>(bytes.size()));
      if (!data) return Error("attention observation write failed");
    }
    // The existing MMA attention consumes half queries even when its
    // descriptor is F32. Preserve original bytes above as well as this
    // explicit, separately named diagnostic round-trip (no tensor mutation).
    std::ofstream effective(dir / (Prefix() + "-q-half.bin"), std::ios::binary);
    const auto& q = records_[0];
    std::array<ggml_fp16_t, 1024> half{};
    for (std::size_t at = 0; at < kFullValues; at += half.size()) {
      const auto n = std::min(half.size(), kFullValues - at);
      for (std::size_t j = 0; j < n; ++j) {
        if (q.type == GGML_TYPE_F16) {
          std::memcpy(&half[j], host_ + q.offset + (at + j) * sizeof(ggml_fp16_t),
                      sizeof(ggml_fp16_t));
        } else {
          float f = 0;
          std::memcpy(&f, host_ + q.offset + (at + j) * sizeof(float), sizeof(float));
          half[j] = ggml_fp32_to_fp16(f);
        }
      }
      effective.write(reinterpret_cast<const char*>(half.data()),
                      static_cast<std::streamsize>(n * sizeof(ggml_fp16_t)));
    }
    return meta && effective ? en::Status{}
                             : en::Status(Error("attention metadata/effective query write failed"));
  }

 private:
  static constexpr std::size_t kFullValues = 512U * 512U * 64U;
  struct Record {
    const char* name = nullptr;
    const ggml_tensor* tensor = nullptr;
    std::size_t offset = 0, bytes = 0;
    ggml_type type = GGML_TYPE_F32;
    std::array<std::int64_t, 4> ne{};
    std::array<std::size_t, 4> nb{};
  };
  static std::unexpected<kg::KernelFailure> Fail(std::string text) {
    return std::unexpected(
        kg::KernelFailure{.error = kg::KernelError::kUnknown, .detail = std::move(text)});
  }
  std::expected<void, kg::KernelFailure> ObservePost(kg::LaunchContext& launch,
                                                     std::span<ggml_tensor* const> nodes,
                                                     std::string_view implementation, bool after) {
    if (post_node_ == nullptr) {
      if (after) return {};
      for (auto* node : nodes)
        if (node->op == GGML_OP_DSV4_HC_POST) {
          if (post_node_ != nullptr) return Fail("multiple HC-post nodes in one step");
          post_node_ = node;
        }
      if (post_node_ == nullptr) return {};
      if (post_stage_ == 0) {
        const auto* product = Unwrap(post_node_->src[0]);
        if (!product || product->op != GGML_OP_MUL_MAT ||
            Address(Unwrap(product->src[0])) !=
                Weight(target_.binding.layers.at(layer_).out_b.index))
          return Fail(
              "observed-layer first HC-post does not consume the authenticated output-B product");
      } else if (Address(post_node_->src[1]) != Address(first_post_)) {
        return Fail("observed-layer FFN HC-post residual is not the observed attention HC-post");
      }
      if (implementation == kg::kDsv4HcPostExpertsNormF16Name) {
        // The selected launch consumes all four descriptors. The ADD and
        // reduction have bound aliases but this kernel never writes either.
        // Authenticate its exact semantic tuple before reading any payload.
        if (post_stage_ != 1 || nodes.size() != 4 || nodes[2] != post_node_)
          return Fail("observed-layer expert HC-post tuple or boundary differs");
        if (auto checked =
                kg::CheckDsv4HcPostExpertsNormF16(nodes[0], nodes[1], nodes[2], nodes[3]);
            !checked)
          return checked;
        experts_elided_ = true;
        const auto* reduce = nodes[0];
        const auto* add = nodes[1];
        const auto* shared = add->src[0] == reduce ? add->src[1] : add->src[0];
        const std::array<const ggml_tensor*, 3> inputs{reduce->src[0], reduce->src[1], shared};
        constexpr std::array<const char*, 3> names{"ffn-down", "ffn-route", "ffn-shared"};
        constexpr std::array<std::size_t, 3> sizes{4096U * 6U * 512U * sizeof(float),
                                                   6U * 512U * sizeof(float),
                                                   4096U * 512U * sizeof(float)};
        for (std::size_t i = 0; i < inputs.size(); ++i) {
          const auto* input = inputs[i];
          if (!input || !input->data || input->type != GGML_TYPE_F32 ||
              !ggml_is_contiguous(input) || ggml_nbytes(input) != sizes[i] ||
              sizes[i] > kLayerTwoWindow - used_)
            return Fail("observed-layer expert HC-post input exceeds its exact paid descriptor");
          auto& record = records_[16 + i];
          record = {.name = names[i],
                    .tensor = input,
                    .offset = used_,
                    .bytes = sizes[i],
                    .type = input->type};
          std::copy_n(input->ne, 4, record.ne.begin());
          std::copy_n(input->nb, 4, record.nb.begin());
          used_ += record.bytes;
        }
      } else if (implementation != kg::kHcPostName &&
                 implementation != kg::kDsv4HcPostNormF16Name) {
        return Fail("unexpected observed-layer HC-post implementation: " +
                    std::string(implementation));
      }
      if (stream_.handle != launch.native_stream().handle)
        return Fail("observed-layer attention/HC-post changed streams");
      constexpr std::array<std::array<std::int64_t, 4>, 5> shapes{{{4096, 512, 1, 1},
                                                                   {4096, 4, 512, 1},
                                                                   {4, 512, 1, 1},
                                                                   {4, 4, 512, 1},
                                                                   {4096, 4, 512, 1}}};
      constexpr std::array<int, 5> row_axes{1, 2, 1, 2, 2};
      constexpr std::array<std::array<const char*, 5>, 2> names{
          {{"hc-cur", "hc-residual", "hc-post", "hc-comb", "hc-out"},
           {"ffn-cur", "ffn-residual", "ffn-post", "ffn-comb", "ffn-out"}}};
      // Reserve and authenticate every source/destination before any copy.
      // Check the actual row strides, including fused packed carry tails.
      for (std::size_t j = 0; j < 5; ++j) {
        if (experts_elided_ && post_stage_ == 1 && j == 0) continue;
        const auto* t = j == 4 ? post_node_ : post_node_->src[j];
        if (!t || !t->data || t->type != GGML_TYPE_F32 ||
            !std::equal(shapes[j].begin(), shapes[j].end(), t->ne) || t->nb[0] != sizeof(float))
          return Fail("observed-layer HC-post descriptor outside the bounded observation");
        const auto axis = static_cast<std::size_t>(row_axes[j]);
        std::size_t row_bytes = static_cast<std::size_t>(t->ne[0]) * sizeof(float);
        if (axis == 2) {
          if (t->nb[1] != row_bytes) return Fail("observed-layer HC-post has a pitched inner row");
          row_bytes *= static_cast<std::size_t>(t->ne[1]);
        }
        const auto bytes = 512 * row_bytes;
        if (t->nb[axis] < row_bytes || bytes > kLayerTwoWindow - used_)
          return Fail("observed-layer HC-post stride/window bound");
        auto& r = records_[j + 6 + 5 * post_stage_];
        r = {.name = names[post_stage_][j],
             .tensor = t,
             .offset = used_,
             .bytes = bytes,
             .type = t->type};
        std::copy_n(t->ne, 4, r.ne.begin());
        std::copy_n(t->nb, 4, r.nb.begin());
        used_ += bytes;
      }
      post_implementations_[post_stage_] = implementation;
      if (experts_elided_)
        for (std::size_t i = 16; i < 19; ++i)
          if (auto copied = Copy(i); !copied) return copied;
      for (std::size_t i = 6 + 5 * post_stage_; i < 10 + 5 * post_stage_; ++i) {
        if (!records_[i].tensor) continue;
        if (auto copied = Copy(i); !copied) return copied;
      }
      return {};
    }
    if (!after || std::ranges::find(nodes, post_node_) == nodes.end() ||
        stream_.handle != launch.native_stream().handle ||
        post_implementations_[post_stage_] != implementation)
      return Fail("observed-layer HC-post before/after pairing changed");
    if (auto copied = Copy(10 + 5 * post_stage_); !copied) return copied;
    if (post_stage_ == 0) first_post_ = post_node_;
    post_node_ = nullptr;
    done_ = ++post_stage_ == 2;
    return {};
  }
  std::expected<void, kg::KernelFailure> Copy(std::size_t i) {
    const auto& r = records_[i];
    const auto copy = [&](std::size_t dst, std::size_t src, std::size_t bytes) {
      const auto copied = llmp::providers::CopyAsync(
          stream_, host_ + r.offset + dst, static_cast<const std::byte*>(r.tensor->data) + src,
          bytes, llmp::providers::CopyKind::kDeviceToHost);
      return copied.ok() ? std::expected<void, kg::KernelFailure>{}
                         : std::expected<void, kg::KernelFailure>{Fail(copied.text())};
    };
    if ((i > 0 && i < 5) || i >= 16) return copy(0, 0, r.bytes);
    constexpr std::array<std::size_t, 16> row_axes{1, 0, 0, 0, 0, 2, 1, 2, 1, 2, 2, 1, 2, 1, 2, 2};
    const auto row_bytes = r.bytes / 512;
    if (r.tensor->nb[row_axes[i]] == row_bytes) return copy(0, 0, r.bytes);
    for (std::size_t j = 0; j < 512; ++j)
      if (auto copied = copy(j * row_bytes, j * r.tensor->nb[row_axes[i]], row_bytes); !copied)
        return copied;
    return {};
  }
  static std::uint64_t Address(const ggml_tensor* t) {
    return t ? reinterpret_cast<std::uintptr_t>(t->data) : 0;
  }
  static const ggml_tensor* Unwrap(const ggml_tensor* t) {
    for (unsigned depth = 0; t && depth < 16; ++depth) {
      if (t->op != GGML_OP_VIEW && t->op != GGML_OP_RESHAPE && t->op != GGML_OP_PERMUTE) return t;
      t = t->src[0];
    }
    return nullptr;
  }
  std::string Prefix() const { return "layer" + std::to_string(layer_); }
  bool MatchesBank(const ggml_tensor* node) const {
    if (!node || node->op != GGML_OP_FLASH_ATTN_EXT) return false;
    for (std::size_t i : {1U, 2U}) {
      const auto* bank = Unwrap(node->src[i]);
      if (!bank) return false;
      if (bank->op == GGML_OP_CONCAT) {
        if (Address(bank->src[0]) != raw_ || Address(bank->src[1]) != state_[2]) return false;
      } else if (bank->op != GGML_OP_NONE || Address(bank) != raw_ ||
                 state_[2] != raw_ + 768U * 512U * sizeof(ggml_fp16_t)) {
        return false;
      }
    }
    return true;
  }
  std::uint64_t Weight(std::uint32_t id) const {
    const auto& layer = target_.binding.layers.at(layer_);
    const auto& norm =
        target_.artifact.resources()[layer_ == 2 ? layer.comp_norm.index : layer.attn_sinks.index];
    const auto& wanted = target_.artifact.resources()[id];
    const auto address = layer_ == 2 ? csa_.norm_address() : (node_ ? Address(node_->src[4]) : 0);
    if (wanted.group != norm.group || address < norm.offset.value()) return 0;
    const auto base = address - norm.offset.value();
    std::uint64_t result = 0;
    return __builtin_add_overflow(base, wanted.offset.value(), &result) ? 0 : result;
  }
  std::expected<void, kg::KernelFailure> AuthenticateAttention() const {
    const auto* sink = Unwrap(node_->src[4]);
    const auto expected = Weight(target_.binding.layers.at(layer_).attn_sinks.index);
    if (!expected || !sink || sink->op != GGML_OP_NONE || Address(sink) != expected ||
        (sink_ != 0 && Address(sink) != sink_))
      return Fail("observed-layer sink weight binding differs");
    // Shared lowering may expose the proved contiguous state bank directly;
    // legacy computes Concat(raw, compressed). Authenticate both state roots,
    // rather than trusting the ordinal number or a tensor's display name.
    for (std::size_t i : {1U, 2U}) {
      const auto* bank = Unwrap(node_->src[i]);
      if (!bank) return Fail("observed-layer attention bank view depth");
      if (bank->op == GGML_OP_CONCAT) {
        if (Address(bank->src[0]) != raw_ || Address(bank->src[1]) != state_[2])
          return Fail(
              "observed-layer attention concatenation is not the bound raw/compressed state");
      } else if (bank->op != GGML_OP_NONE || Address(bank) != raw_ ||
                 state_[2] != raw_ + 768U * 512U * sizeof(ggml_fp16_t)) {
        return Fail("observed-layer attention direct bank is not the bound contiguous state");
      }
    }
    return {};
  }
  std::byte* host_;
  const Target& target_;
  std::array<std::uint64_t, 3> state_;
  std::uint64_t raw_;
  std::uint32_t layer_;
  std::uint64_t sink_;
  CsaProbe csa_;
  ggml_tensor* first_post_ = nullptr;
  ggml_tensor* node_ = nullptr;
  ggml_tensor* post_node_ = nullptr;
  llmp::providers::NativeStream stream_;
  kg::FlashAttnMmaPlan plan_;
  std::array<std::int32_t, GGML_MAX_OP_PARAMS / sizeof(std::int32_t)> params_{};
  std::string_view implementation_;
  std::array<std::string_view, 2> post_implementations_{};
  std::array<Record, 19> records_{};
  std::size_t used_ = kCsaWindow, post_stage_ = 0;
  bool attention_done_ = false, done_ = false, experts_elided_ = false;
};

static_assert(sizeof(LayerBoundaryProbe) <= 8192);

// Fixed rows=1 complete payload inventory: two HC-pre corridors, attention,
// and two materialized HC-post tuples. F16 original Q uses fewer bytes.
constexpr std::uint64_t kFirstDecodePayload =
    2ULL * (65536 + 1572864 + 25600 + 25600 + 65536 + 12 + 96 + 16384 + 16384 + 16 + 64) + 131072 +
    2ULL * 786432 + 1536 + 256 + 131072 + 2ULL * (16384 + 65536 + 16 + 64 + 65536) +
    768ULL * 32 * sizeof(std::uint16_t) + sizeof(std::int32_t) + 131072 + 32768 + 32768 + 16384 +
    41984 + 147508;  // dense corridor plus token/IDs/weights/routed activation/down
// Includes the full shared padded mask plus its actual position; the common
// one-query mask row above is a separate copy, never fabricated padding.
// Half-Q, all six full heads, both state text inventories and metadata also
// remain below one existing 384 MiB file allowance per arm; three are paid.
static_assert(kFirstDecodePayload + 65536 + 6ULL * 517120 + (1U << 20U) < kLayerTwoWindow);
// At most64 trace calls, each one header plus two nodes/four operands (11
// lines);64 partial receipt rows and64 other metadata lines, each below1KiB.
// All stay inside the existing per-arm1MiB metadata/file inventory above.
constexpr std::size_t kFirstDecodeTraceCalls = 64;
static_assert((kFirstDecodeTraceCalls * 11 + 64 + 64) * 1024 < (1U << 20U));
// This receipt is derived from the actual owning PagedWeights, not a selected
// descriptor's address. All expert groups must agree before any capture starts.
struct RoutedArrayReceipt {
  std::uint64_t address = 0, stride = 0, slice = 0;
  ggml_type type = GGML_TYPE_F32;
  std::array<std::int64_t, 4> ne{};
  std::array<std::size_t, 4> nb{};
};
bool MatchesRoutedArray(const ggml_tensor* t, const RoutedArrayReceipt& r) {
  return t && t->data && r.address != 0 && r.stride != 0 && r.stride <= (64ULL << 20U) &&
         r.slice != 0 && r.slice <= r.stride && t->op == GGML_OP_NONE && !t->view_src &&
         t->type == r.type && reinterpret_cast<std::uintptr_t>(t->data) == r.address &&
         std::equal(r.ne.begin(), r.ne.end(), t->ne) &&
         std::equal(r.nb.begin(), r.nb.end(), t->nb) && ggml_nbytes(t) == 255 * r.stride + r.slice;
}
TEST(Dsv4RoutedObservation, ExactArrayAddressTypeShapeAndPitchAreRequired) {
  RoutedArrayReceipt r{.address = 0x100000,
                       .stride = 8064224,
                       .slice = 2424832,
                       .type = GGML_TYPE_IQ2_XS,
                       .ne = {4096, 2048, 256, 1},
                       .nb = {74, 1184, 8064224, 8064224ULL * 256}};
  ggml_tensor t{};
  t.data = reinterpret_cast<void*>(r.address);
  t.type = r.type;
  std::copy(r.ne.begin(), r.ne.end(), t.ne);
  std::copy(r.nb.begin(), r.nb.end(), t.nb);
  ASSERT_TRUE(MatchesRoutedArray(&t, r));
  for (unsigned change = 0; change < 6; ++change) {
    auto bad = t;
    if (change == 0) bad.data = reinterpret_cast<void*>(r.address + 256);
    if (change == 1) bad.type = GGML_TYPE_IQ3_XXS;
    if (change == 2) --bad.ne[2];
    if (change == 3) bad.nb[2] += 256;
    if (change == 4) bad.view_src = &t;
    if (change == 5) bad.op = GGML_OP_CONT;
    EXPECT_FALSE(MatchesRoutedArray(&bad, r)) << change;
  }
  EXPECT_FALSE(MatchesRoutedArray(nullptr, r));
  r.stride = UINT64_MAX;
  EXPECT_FALSE(MatchesRoutedArray(&t, r));
}

// Exact encodings from the pinned artifact index, not a shape-based widening.
// Layers0/3/8 retain their original domain; layer26 has different source blocks.
ggml_type ObservedRoutedEncoding(std::uint32_t layer, std::size_t part) {
  if (part >= 3) return GGML_TYPE_COUNT;
  if (layer == 0 || layer == 3 || layer == 8)
    return part == 2 ? GGML_TYPE_IQ3_XXS : GGML_TYPE_IQ2_XS;
  if (layer == 26) return part == 2 ? GGML_TYPE_MXFP4 : GGML_TYPE_IQ3_XXS;
  return GGML_TYPE_COUNT;
}
ggml_type ObservedSharedDownEncoding(std::uint32_t layer) {
  if (layer == 3 || layer == 8) return GGML_TYPE_Q6_K;
  return layer == 26 ? GGML_TYPE_Q8_0 : GGML_TYPE_COUNT;
}
TEST(Dsv4RoutedObservation, PinnedLayer26FormatsDoNotWidenEarlierSources) {
  for (const auto layer : {0U, 3U, 8U}) {
    EXPECT_EQ(ObservedRoutedEncoding(layer, 0), GGML_TYPE_IQ2_XS);
    EXPECT_EQ(ObservedRoutedEncoding(layer, 1), GGML_TYPE_IQ2_XS);
    EXPECT_EQ(ObservedRoutedEncoding(layer, 2), GGML_TYPE_IQ3_XXS);
  }
  EXPECT_EQ(ObservedSharedDownEncoding(3), GGML_TYPE_Q6_K);
  EXPECT_EQ(ObservedSharedDownEncoding(8), GGML_TYPE_Q6_K);
  EXPECT_EQ(ObservedSharedDownEncoding(26), GGML_TYPE_Q8_0);
  for (const auto layer : {1U, 2U, 9U, 25U, 27U, 43U}) {
    EXPECT_EQ(ObservedRoutedEncoding(layer, 0), GGML_TYPE_COUNT);
    EXPECT_EQ(ObservedSharedDownEncoding(layer), GGML_TYPE_COUNT);
  }
  EXPECT_EQ(ObservedRoutedEncoding(26, 3), GGML_TYPE_COUNT);
  for (std::size_t i = 0; i < 3; ++i) {
    const auto type = ObservedRoutedEncoding(26, i);
    EXPECT_EQ(type, i == 2 ? GGML_TYPE_MXFP4 : GGML_TYPE_IQ3_XXS);
    const auto k = i == 2 ? 2048 : 4096, n = i == 2 ? 4096 : 2048;
    const auto row = ggml_row_size(type, k);
    RoutedArrayReceipt receipt{.address = 0x100000,
                               .stride = 10888976,
                               .slice = row * static_cast<std::size_t>(n),
                               .type = type,
                               .ne = {k, n, 256, 1},
                               .nb = {ggml_type_size(type), row, 10888976, 10888976ULL * 256}};
    EXPECT_EQ(receipt.slice, i == 2 ? 4456448U : 3211264U);
    ggml_tensor tensor{};
    tensor.type = type;
    tensor.data = reinterpret_cast<void*>(receipt.address);
    std::copy(receipt.ne.begin(), receipt.ne.end(), tensor.ne);
    std::copy(receipt.nb.begin(), receipt.nb.end(), tensor.nb);
    ASSERT_TRUE(MatchesRoutedArray(&tensor, receipt));
    tensor.type = ObservedRoutedEncoding(8, i);
    EXPECT_FALSE(MatchesRoutedArray(&tensor, receipt));
    tensor.type = type;
    tensor.nb[2] = 8064224;
    EXPECT_FALSE(MatchesRoutedArray(&tensor, receipt));
  }
}

std::expected<std::array<RoutedArrayReceipt, 3>, std::string> RoutedReceipts(
    const Target& target, const en::PagedWeights& owner, std::uint32_t layer = 0) {
  const auto& a = owner.artifact();
  if (a.id() != target.artifact.id() || a.format_version() != 0)
    return Error("routed observation weight owner differs from pinned target");
  const auto& l = target.binding.layers.at(layer);
  const std::array bindings{l.gate_exps, l.up_exps, l.down_exps};
  const std::array<std::pair<std::uint32_t, std::string_view>, 3> arrays{
      {{l.gate_exps.index, l.gate_exps.type},
       {l.up_exps.index, l.up_exps.type},
       {l.down_exps.index, l.down_exps.type}}};
  for (const auto& binding : bindings)
    if (binding.index >= a.expert_arrays().size()) return Error("routed array index absent");
  auto slab = en::ExpertSlab(a, arrays, 256, 256, layer);
  if (!slab || slab->stride == 0 || slab->stride > (64ULL << 20U) ||
      slab->first_group >= a.groups().size() || 256 > a.groups().size() - slab->first_group)
    return Error("routed observed-layer slab outside fixed descriptor domain");
  const auto base = owner.group_address(slab->first_group);
  if (base == 0 || base > UINT64_MAX - 256 * slab->stride)
    return Error("routed slab address overflow");
  for (std::uint32_t e = 0; e < 256; ++e)
    if (owner.group_address(slab->first_group + e) != base + e * slab->stride ||
        a.groups()[slab->first_group + e].kind != ar::GroupKind::kExpert)
      return Error("routed physical group placement differs from actual slab");
  std::array<RoutedArrayReceipt, 3> out{};
  for (std::size_t i = 0; i < out.size(); ++i) {
    const auto& binding = bindings[i];
    const auto& array = a.expert_arrays()[binding.index];
    auto type = kg::GgmlTypeOf(binding.type);
    const auto expected_type = ObservedRoutedEncoding(layer, i);
    const auto k = i == 2 ? 2048 : 4096, n = i == 2 ? 4096 : 2048;
    if (!type || *type != expected_type || array.layer != layer || array.count != 256 ||
        array.first_group != slab->first_group || array.part != llmp::format::PartRole::kData ||
        !array.format || array.format->shape.rank != 2 || array.format->stacked != 1 ||
        array.format->layout != llmp::format::Layout::kGgmlBlocks ||
        binding.ne != std::vector<std::uint64_t>{static_cast<std::uint64_t>(k),
                                                 static_cast<std::uint64_t>(n)} ||
        array.format->shape.ne[0] != static_cast<std::uint64_t>(k) ||
        array.format->shape.ne[1] != static_cast<std::uint64_t>(n))
      return Error("routed array no longer has its complete approved data slice");
    const auto parts = llmp::format::PartsOf(*array.format);
    if (!parts || parts->size() != 1 || (*parts)[0].role != llmp::format::PartRole::kData)
      return Error("routed array source is not a single immutable data part");
    const auto& info = llmp::format::Info(array.format->encoding);
    if (!info.ggml || info.ggml->type != static_cast<std::uint32_t>(*type))
      return Error("routed source encoding differs from native descriptor");
    const auto row = ggml_row_size(*type, k);
    const auto slice = row * static_cast<std::size_t>(n);
    if (array.slice_bytes.value() != slice || array.group_offset.value() > slab->stride ||
        slice > slab->stride - array.group_offset.value())
      return Error("routed slice extent differs from checked slab");
    for (std::uint32_t e = 0; e < 256; ++e) {
      const auto stored = a.groups()[slab->first_group + e].stored.value();
      if (array.group_offset.value() > stored || slice > stored - array.group_offset.value())
        return Error("routed data slice reaches an unwritten group gap");
    }
    const auto address = owner.array_address(binding.index);
    if (address != base + array.group_offset.value())
      return Error("routed array callback differs from actual owning groups");
    out[i] = {.address = address,
              .stride = slab->stride,
              .slice = slice,
              .type = *type,
              .ne = {k, n, 256, 1},
              .nb = {ggml_type_size(*type), row, slab->stride, slab->stride * 256}};
  }
  return out;
}

// A separate fixed scalar-decode observation. No 512-row probe contract changes.
// It is installed only after all five original prefill units have completed.
class FirstDecodeProbe final : public ScreenProbe {
 public:
  FirstDecodeProbe(void* host, const Target& target, std::uint64_t raw,
                   std::array<std::uint64_t, 9> weights, std::array<RoutedArrayReceipt, 3> routed,
                   std::int32_t token, bool shared)
      : host_(static_cast<std::byte*>(host)),
        target_(target),
        raw_(raw),
        weights_(weights),
        routed_(routed),
        token_(token),
        shared_(shared) {}
  bool active() const override { return stage_ != 7; }
  std::expected<void, kg::KernelFailure> Observe(kg::LaunchContext& launch,
                                                 std::span<ggml_tensor* const> nodes,
                                                 std::string_view implementation,
                                                 bool after) override {
    if (!active()) return {};
    if (shared_) {
      const auto selected = SelectedRouteRead(launch, nodes, implementation, after);
      if (!selected) return std::unexpected(selected.error());
      if (*selected) return {};
      const auto hash = HashRead(launch, nodes, implementation, after);
      if (!hash) return std::unexpected(hash.error());
      if (*hash) return {};
    }
    if (stage_ == 6) {
      const auto routed = RoutedProducts(launch, nodes, implementation, after);
      if (!routed) return std::unexpected(routed.error());
      if (*routed) return {};
      const auto ffn = FfnProducts(launch, nodes, implementation, after);
      if (!ffn) return std::unexpected(ffn.error());
      if (*ffn) return {};
    }
    if (stage_ == 3) {
      const auto projected = Projection(launch, nodes, implementation, after);
      if (!projected) {
        TraceSelected(nodes, implementation);
        return std::unexpected(projected.error());
      }
      if (*projected) return {};
    }
    if (shared_ && (stage_ == 3 || stage_ == 6)) {
      const auto post = PrimitivePost(launch, nodes, implementation, after);
      if (!post) {
        TraceSelected(nodes, implementation);
        return std::unexpected(post.error());
      }
      if (*post) return {};
    }
    if (!after && (stage_ == 3 || stage_ == 6)) TraceSelected(nodes, implementation);
    if (!pending_) {
      if (after) return {};
      ggml_tensor* found = nullptr;
      for (auto* n : nodes) {
        const auto custom = kg::LlmpOpOf(n);
        const bool match = (stage_ == 0 || stage_ == 4)   ? custom == kg::LlmpOp::kDsv4HcMix
                           : (stage_ == 1 || stage_ == 5) ? custom == kg::LlmpOp::kDsv4HcPre
                           : stage_ == 2                  ? n->op == GGML_OP_FLASH_ATTN_EXT
                                                          : n->op == GGML_OP_DSV4_HC_POST;
        if (!match) continue;
        if (found) return Fail("multiple nodes at the fixed decode boundary");
        found = n;
      }
      if (!found) return {};
      if (stage_ == 0) {
        if (stream_.handle && stream_.handle != launch.native_stream().handle)
          return Fail("early hash and decode corridor changed streams");
        stream_ = launch.native_stream();
      }
      if (stream_.handle != launch.native_stream().handle)
        return Fail("decode probe changed streams");
      pending_ = found;
      first_ = count_;
      implementations_[stage_] = implementation;
      std::memcpy(params_[stage_].data(), found->op_params, sizeof(found->op_params));
      if (auto admitted = Boundary(launch, found, implementation); !admitted) {
        FailureGeometry(found, implementation);
        return admitted;
      }
      for (std::size_t i = first_; i < count_; ++i)
        if (!records_[i].output)
          if (auto copied = Copy(i); !copied) return copied;
      return {};
    }
    if (!after || std::ranges::find(nodes, pending_) == nodes.end() ||
        implementation != implementations_[stage_] ||
        stream_.handle != launch.native_stream().handle)
      return Fail("decode boundary before/after pairing changed");
    for (std::size_t i = first_; i < count_; ++i)
      if (records_[i].output)
        if (auto copied = Copy(i); !copied) return copied;
    pending_ = nullptr;
    ++stage_;
    return {};
  }
  en::Status Save(const fs::path& dir) const override {
    if (active()) {
      // Save is called only after the complete RunStep/Chunk fence. This
      // receipt diagnoses completed earlier boundaries; it never satisfies
      // the mandatory seven-boundary oracle or creates missing payloads.
      if (pending_ || projection_pending_ || ffn_pending_ || routed_pending_ || hash_pending_ ||
          selected_pending_ || primitive_pending_ || count_ > records_.size() ||
          used_ > kFirstDecodePayload)
        return Error("incomplete decode observation has an unfinished boundary");
      std::ofstream partial(dir / "decode-layer0-incomplete.txt");
      partial << "completed_fence=1 shared=" << shared_ << " completed_boundaries=" << stage_
              << " records=" << count_ << " bytes=" << used_ << " raw_address=" << raw_
              << " trace_calls=" << trace_calls_ << " trace_capped=" << trace_capped_ << '\n';
      partial << "attention head=" << plan_.head << " mask_prepass=" << plan_.mask_prepass
              << " sparse=" << plan_.sparse << " columns=" << plan_.columns
              << " group=" << plan_.group << " blocks=" << plan_.blocks
              << " scratch=" << plan_.scratch << '\n';
      for (std::size_t i = 0; i < stage_; ++i) {
        partial << "stage=" << i << " implementation=" << implementations_[i] << " params";
        for (const auto value : params_[i]) partial << ' ' << value;
        partial << '\n';
      }
      for (std::size_t i = 0; i < count_; ++i) {
        const auto& r = records_[i];
        if (r.offset > used_ || r.bytes > used_ - r.offset)
          return Error("incomplete decode observation range exceeds completed copies");
        const auto bytes = std::span<const std::byte>(host_ + r.offset, r.bytes);
        const auto hash = llmp::base::ToHex(llmp::base::Sha256().Update(bytes).Finish());
        partial << r.name << " type=" << ggml_type_name(r.type) << " ne";
        for (auto n : r.ne) partial << ' ' << n;
        partial << " nb";
        for (auto n : r.nb) partial << ' ' << n;
        partial << " address=" << reinterpret_cast<std::uintptr_t>(r.address)
                << " tensor_offset=" << r.tensor_offset << " bytes=" << r.bytes
                << " sha256=" << hash << " expected=" << r.expected
                << " expected_match=" << (r.expected.empty() || hash == r.expected) << '\n';
      }
      if (!partial) return Error("incomplete decode receipt write failed");
      return Error("first decode observation incomplete at stage " + std::to_string(stage_));
    }
    if ((shared_ && !selected_route_) || count_ != 53U || projection_count_ != 2 ||
        ffn_count_ != 2 || routed_count_ != 2 || !weights_copied_ || !combine_output_ ||
        routed_pending_ || hash_pending_ || selected_pending_ || (shared_ && !hash_token_) ||
        used_ > kFirstDecodePayload)
      return Error("first decode complete payload inventory differs");
    // Only after the complete RunStep/Chunk fence. Check the full initialized
    // mask, including every shared padding element, and its actual position.
    std::int32_t token = -1;
    std::memcpy(&token, host_ + records_[token_record_].offset, sizeof(token));
    if (token != token_) return Error("routed hash token differs from actual first decode input");
    for (std::size_t e = 0; e < 6; ++e) {
      std::int32_t id = -1;
      std::memcpy(&id, host_ + records_[ids_record_].offset + e * sizeof(id), sizeof(id));
      if (id < 0 || id >= 256) return Error("routed selected ID outside authenticated array");
    }
    const auto& mask = records_[mask_record_];
    for (std::size_t i = 0; i < mask.bytes / sizeof(std::uint16_t); ++i) {
      std::uint16_t bits = 0;
      std::memcpy(&bits, host_ + mask.offset + i * sizeof(bits), sizeof(bits));
      const auto row = i / 768, cell = i % 768;
      const bool visible = row == 0 && (kPrefix % 768 + 768 - cell) % 768 < 128;
      if (bits != (visible ? 0x0000U : 0xFC00U))
        return Error("first decode mask differs from actual fixed query/window or padding");
    }
    std::int32_t position = -1;
    std::memcpy(&position, host_ + records_[positions_record_].offset, sizeof(position));
    if (position != static_cast<std::int32_t>(kPrefix))
      return Error("first decode actual mask position changed");
    std::ofstream meta(dir / "decode-layer0.txt");
    meta << "layer=0 past=2560 rows=all_1 raw_address=" << raw_ << " bytes=" << used_ << '\n';
    meta << "attention head=" << plan_.head << " mask_prepass=" << plan_.mask_prepass
         << " sparse=" << plan_.sparse << " columns=" << plan_.columns << " group=" << plan_.group
         << " blocks=" << plan_.blocks << " scratch=" << plan_.scratch << '\n';
    for (std::size_t i = 0; i < routed_.size(); ++i) {
      const auto& r = routed_[i];
      meta << "routed_array=" << i << " address=" << r.address << " type=" << r.type
           << " stride=" << r.stride << " slice=" << r.slice << " ne";
      for (auto value : r.ne) meta << ' ' << value;
      meta << " nb";
      for (auto value : r.nb) meta << ' ' << value;
      meta << '\n';
    }
    for (std::size_t i = 0; i < routed_implementations_.size(); ++i) {
      if (routed_implementations_[i].empty()) continue;
      meta << "routed_boundary=" << i << " implementation=" << routed_implementations_[i]
           << " params";
      for (auto value : routed_params_[i]) meta << ' ' << value;
      meta << '\n';
    }
    for (std::size_t i = 0; i < projection_count_; ++i)
      meta << "projection=" << i << " implementation=" << projection_impl_[i] << '\n';
    for (std::size_t i = 0; i < stage_; ++i) {
      meta << "stage=" << i << " implementation=" << implementations_[i] << " params";
      for (auto v : params_[i]) meta << ' ' << v;
      meta << '\n';
    }
    for (std::size_t i = 0; i < count_; ++i) {
      const auto& r = records_[i];
      const auto bytes = std::span<const std::byte>(host_ + r.offset, r.bytes);
      const auto hash = llmp::base::ToHex(llmp::base::Sha256().Update(bytes).Finish());
      if (!r.expected.empty() && hash != r.expected)
        return Error("decode layer0 resource identity differs: " + r.name);
      meta << r.name << " type=" << ggml_type_name(r.type) << " ne";
      for (auto n : r.ne) meta << ' ' << n;
      meta << " nb";
      for (auto n : r.nb) meta << ' ' << n;
      meta << " tensor_offset=" << r.tensor_offset
           << " address=" << reinterpret_cast<std::uintptr_t>(r.address) << " bytes=" << r.bytes
           << " sha256=" << hash << '\n';
      std::ofstream file(dir / ("decode-layer0-" + r.name + ".bin"), std::ios::binary);
      file.write(reinterpret_cast<const char*>(bytes.data()),
                 static_cast<std::streamsize>(bytes.size()));
      if (!file) return Error("decode observation output failed");
    }
    const auto& q = records_[query_record_];
    std::ofstream half(dir / "decode-layer0-q-half.bin", std::ios::binary);
    std::array<ggml_fp16_t, 1024> converted{};
    for (std::size_t at = 0; at < 512U * 64U; at += converted.size()) {
      for (std::size_t j = 0; j < converted.size(); ++j) {
        if (q.type == GGML_TYPE_F16)
          std::memcpy(&converted[j], host_ + q.offset + (at + j) * 2, 2);
        else {
          float value = 0;
          std::memcpy(&value, host_ + q.offset + (at + j) * 4, 4);
          converted[j] = ggml_fp32_to_fp16(value);
        }
      }
      half.write(reinterpret_cast<const char*>(converted.data()), sizeof(converted));
    }
    return meta && half ? en::Status{} : Error("decode metadata/effective-query output failed");
  }

 private:
  struct Record {
    std::string name;
    const std::byte* address = nullptr;
    std::size_t offset = 0, bytes = 0, tensor_offset = 0;
    ggml_type type = GGML_TYPE_F32;
    std::array<std::int64_t, 4> ne{};
    std::array<std::size_t, 4> nb{};
    bool output = false;
    std::string_view expected;
  };
  static std::unexpected<kg::KernelFailure> Fail(std::string reason) {
    return std::unexpected(
        kg::KernelFailure{.error = kg::KernelError::kUnknown, .detail = std::move(reason)});
  }
  void TraceSelected(std::span<ggml_tensor* const> nodes, std::string_view implementation) {
    // Actual BoundGraph RunStep callback: host descriptors only, never GPU
    // payloads. Bound64 calls, two selected nodes and four direct operands;
    // report truncation explicitly. No graph clone or retained tensor list.
    if (trace_calls_ == kFirstDecodeTraceCalls) {
      trace_capped_ = true;
      return;
    }
    const auto call = trace_calls_++;
    std::printf(
        "DECODE_SELECTED shared=%d stage=%zu call=%zu nodes=%zu truncated=%d impl=%.*s "
        "raw=%llu residual=%llu pre=%llu fn=%llu\n",
        shared_, stage_, call, nodes.size(), nodes.size() > 2,
        static_cast<int>(std::min<std::size_t>(implementation.size(), 192)), implementation.data(),
        static_cast<unsigned long long>(raw_), static_cast<unsigned long long>(Address(residual_)),
        static_cast<unsigned long long>(Address(pre_)),
        static_cast<unsigned long long>(Address(fn_)));
    for (std::size_t i = 0; i < std::min<std::size_t>(nodes.size(), 2); ++i) {
      const auto* n = nodes[i];
      if (!n) continue;
      const std::array<const ggml_tensor*, 5> tensors{n, n->src[0], n->src[1], n->src[2],
                                                      n->src[3]};
      for (std::size_t operand = 0; operand < tensors.size(); ++operand) {
        const auto* t = tensors[operand];
        if (!t) continue;
        std::printf(
            "DECODE_SELECTED_DESC call=%zu node=%zu operand=%zu tensor=%p op=%d custom=%d "
            "type=%d address=%llu view=%p view_offset=%zu ne=%lld,%lld,%lld,%lld "
            "nb=%zu,%zu,%zu,%zu src=%p,%p,%p,%p\n",
            call, i, operand, static_cast<const void*>(t), static_cast<int>(t->op),
            static_cast<int>(kg::LlmpOpOf(t)), static_cast<int>(t->type),
            static_cast<unsigned long long>(Address(t)), static_cast<const void*>(t->view_src),
            t->view_offs, static_cast<long long>(t->ne[0]), static_cast<long long>(t->ne[1]),
            static_cast<long long>(t->ne[2]), static_cast<long long>(t->ne[3]), t->nb[0], t->nb[1],
            t->nb[2], t->nb[3], static_cast<const void*>(t->src[0]),
            static_cast<const void*>(t->src[1]), static_cast<const void*>(t->src[2]),
            static_cast<const void*>(t->src[3]));
      }
    }
    std::fflush(stdout);
  }
  void FailureGeometry(const ggml_tensor* n, std::string_view implementation) const {
    // Descriptor fields only: no GPU bytes, staged values or unfinished copies.
    std::printf(
        "DECODE_GEOMETRY shared=%d stage=%zu past=2560 rows=1 raw=%llu "
        "raw_bytes=786432 impl=%.*s\n",
        shared_, stage_, static_cast<unsigned long long>(raw_),
        static_cast<int>(std::min<std::size_t>(implementation.size(), 192)), implementation.data());
    const std::array<const ggml_tensor*, 7> tensors{n,
                                                    n->src[0],
                                                    n->src[1],
                                                    n->src[2],
                                                    n->src[3],
                                                    n->src[4],
                                                    n->src[3] ? n->src[3]->src[0] : nullptr};
    for (std::size_t i = 0; i < tensors.size(); ++i) {
      auto* t = tensors[i];
      for (unsigned depth = 0; t && depth < 16; ++depth) {
        const auto address = Address(t);
        std::printf(
            "DECODE_DESC tensor=%zu depth=%u op=%d type=%d address=%llu "
            "raw_relative=%llu raw_inside=%d view_offset=%zu ne=%lld,%lld,%lld,%lld "
            "nb=%zu,%zu,%zu,%zu\n",
            i, depth, static_cast<int>(t->op), static_cast<int>(t->type),
            static_cast<unsigned long long>(address),
            static_cast<unsigned long long>(address >= raw_ ? address - raw_ : 0),
            address >= raw_ && address - raw_ < 786432, t->view_offs,
            static_cast<long long>(t->ne[0]), static_cast<long long>(t->ne[1]),
            static_cast<long long>(t->ne[2]), static_cast<long long>(t->ne[3]), t->nb[0], t->nb[1],
            t->nb[2], t->nb[3]);
        std::printf("DECODE_PARAMS tensor=%zu depth=%u", i, depth);
        for (const auto value : t->op_params) std::printf(" %d", value);
        std::printf("\n");
        if (t->op != GGML_OP_VIEW && t->op != GGML_OP_RESHAPE && t->op != GGML_OP_PERMUTE) break;
        t = t->src[0];
      }
    }
    std::fflush(stdout);
  }
  static std::uint64_t Address(const ggml_tensor* t) {
    return t ? reinterpret_cast<std::uintptr_t>(t->data) : 0;
  }
  static const ggml_tensor* Unwrap(const ggml_tensor* t) {
    for (unsigned depth = 0; t && depth < 16; ++depth) {
      if (t->op != GGML_OP_VIEW && t->op != GGML_OP_RESHAPE && t->op != GGML_OP_PERMUTE) return t;
      t = t->src[0];
    }
    return nullptr;
  }
  static bool Shape(const ggml_tensor* t, ggml_type type, std::array<std::int64_t, 4> ne) {
    return t && t->data && t->type == type && std::equal(ne.begin(), ne.end(), t->ne);
  }
  std::uint32_t Resource(std::size_t index) const {
    const auto& l = target_.binding.layers.at(0);
    const std::array ids{l.hc_attn_fn.index,  l.hc_attn_scale.index, l.hc_attn_base.index,
                         l.attn_norm.index,   l.hc_ffn_fn.index,     l.hc_ffn_scale.index,
                         l.hc_ffn_base.index, l.ffn_norm.index,      l.attn_sinks.index};
    return ids.at(index);
  }
  std::uint64_t Weight(std::uint32_t id) const {
    const auto& anchor = target_.artifact.resources()[Resource(0)];
    const auto& r = target_.artifact.resources()[id];
    // Shared's early hash lookup may precede the first HC-mix callback. Its
    // anchor was obtained from the same actual owning PagedWeights before
    // execution; later HC callbacks still verify their exact bound addresses.
    const auto address = shared_ ? weights_[0] : Address(fn_);
    if (r.group != anchor.group || address < anchor.offset.value() || address == 0) return 0;
    std::uint64_t result = 0;
    return __builtin_add_overflow(address - anchor.offset.value(), r.offset.value(), &result)
               ? 0
               : result;
  }
  std::expected<void, kg::KernelFailure> Add(std::string name, const ggml_tensor* t,
                                             bool output = false, std::size_t offset = 0,
                                             std::size_t bytes = 0,
                                             std::string_view expected = {}) {
    if (!t || !t->data || count_ == records_.size() || name.size() > 64)
      return Fail("decode record identity/count bound");
    const auto extent = ggml_nbytes(t);
    if (extent == 0 || extent > (16U << 20U) || offset > extent)
      return Fail("decode descriptor extent bound");
    if (bytes == 0) bytes = extent - offset;
    if (bytes > extent - offset || used_ > kFirstDecodePayload ||
        bytes > kFirstDecodePayload - used_)
      return Fail("decode record byte bound");
    if (offset == 0 && bytes == extent &&
        static_cast<std::size_t>(ggml_nelements(t)) * ggml_type_size(t->type) != extent)
      return Fail("decode record includes a pitched or padded payload");
    auto& r = records_[count_++];
    r = {.name = std::move(name),
         .address = static_cast<const std::byte*>(t->data) + offset,
         .offset = used_,
         .bytes = bytes,
         .tensor_offset = offset,
         .type = t->type,
         .output = output,
         .expected = expected};
    std::copy_n(t->ne, 4, r.ne.begin());
    std::copy_n(t->nb, 4, r.nb.begin());
    used_ += bytes;
    return {};
  }
  std::expected<void, kg::KernelFailure> WeightRecord(std::string name, const ggml_tensor* t,
                                                      std::size_t index) {
    static constexpr std::array<std::string_view, 9> hashes{
        "f5c1ffdfb92df2c04ac17e9a31e38701f2b7a5cac0cd427a2df2aa3e239987fc",
        "0b0e327d2f4d1a104c53d6e0a9172cf532028383e82cdf6d70537cb83092c63f",
        "edaa695cf5de59f919415f6e71dcb35be5ad817a06fa7222ee3021d9f388adda",
        "010f343569ec11134ae6f156a99d846a0528fd5c80625d474faa79440caff16a",
        "7e65400d5ac74d31e2dc99df3560cc6e32b7c041e13c55fa71e77cfb8c413542",
        "877b3018c5c861fd57195b173629f69324576b9521b13048bfc06502c7a06ddd",
        "c1e6c055605909517f5738b4e0107a4d8b916e15b82a3a03f9f7273efa03b761",
        "aadb2b3223c48f1672db6c67de0b6c9df9d5fd8482866f8300844ed1a1c37b86",
        "59910cf05f8e44b6ea2cf9b4df8f81f2ab8ae8956211c830220e07f0e1d05ee5"};
    const auto* root = Unwrap(t);
    const auto& r = target_.artifact.resources()[Resource(index)];
    if (!root || root->op != GGML_OP_NONE || Address(root) != Weight(Resource(index)) ||
        (weights_[index] && weights_[index] != Address(root)) || ggml_nbytes(t) != r.bytes.value())
      return Fail("decode weight does not bind its exact layer0 resource");
    return Add(std::move(name), t, false, 0, 0, hashes[index]);
  }
  std::expected<void, kg::KernelFailure> Boundary(kg::LaunchContext& launch, ggml_tensor* n,
                                                  std::string_view impl) {
    const bool ffn = stage_ >= 4;
    const std::string prefix = ffn ? "ffn-" : "attn-";
    if (stage_ == 0 || stage_ == 4) {
      if (impl != kg::kDsv4HcMixName) return Fail("decode HC mix implementation differs");
      if (auto valid = kg::CheckDsv4HcMix(n); !valid) return valid;
      if (!Shape(n->src[0], GGML_TYPE_F32, {4096, 4, 1, 1}) ||
          !Shape(n->src[1], GGML_TYPE_F32, {16384, 24, 1, 1}) ||
          !Shape(n, GGML_TYPE_F32, {25, 256, 1, 1}))
        return Fail("decode HC mix geometry");
      if (!ffn)
        fn_ = n->src[1];
      else if (Address(n->src[0]) != Address(post_))
        return Fail("FFN HC mix residual is not attention HC-post");
      mix_ = n;
      residual_ = n->src[0];
      if (auto r = Add(prefix + "mix-x", n->src[0]); !r) return r;
      if (auto r = WeightRecord(prefix + "mix-fn", n->src[1], ffn ? 4 : 0); !r) return r;
      return Add(prefix + "mix-partials", n, true);
    }
    if (stage_ == 1 || stage_ == 5) {
      if (impl != kg::kDsv4HcPreName) return Fail("decode HC-pre implementation differs");
      if (auto valid = kg::CheckDsv4HcPre(n); !valid) return valid;
      if (n->src[0] != mix_ || n->src[1] != residual_ ||
          ggml_nbytes(n) != (4096 + 32) * sizeof(float))
        return Fail("decode HC-pre source/packed output identity");
      pre_ = n;
      if (auto r = Add(prefix + "pre-partials", n->src[0]); !r) return r;
      if (auto r = Add(prefix + "pre-x", n->src[1]); !r) return r;
      constexpr std::array names{"scale", "base", "norm"};
      for (std::size_t i = 0; i < 3; ++i)
        if (auto r = WeightRecord(prefix + "pre-" + names[i], n->src[i + 2], (ffn ? 4 : 0) + i + 1);
            !r)
          return r;
      // The tail's unused12 floats are not outputs and must never be hashed.
      if (auto r = Add(prefix + "pre-normalized", n, true, 0, 4096 * sizeof(float)); !r) return r;
      if (auto r = Add(prefix + "pre-post", n, true, 4096 * sizeof(float), 4 * sizeof(float)); !r)
        return r;
      return Add(prefix + "pre-comb", n, true, 4100 * sizeof(float), 16 * sizeof(float));
    }
    if (stage_ == 2) {
      if (impl != kg::kFlashAttnMmaName && impl != kg::kFlashAttnMmaWideName)
        return Fail("decode attention implementation differs");
      const auto* q = n->src[0];
      const auto* k = n->src[1];
      const auto* v = n->src[2];
      if (!q || (q->type != GGML_TYPE_F32 && q->type != GGML_TYPE_F16) ||
          !Shape(q, q->type, {512, 1, 64, 1}) || !Shape(k, GGML_TYPE_F16, {512, 768, 1, 1}) ||
          !Shape(v, GGML_TYPE_F16, {512, 768, 1, 1}) ||
          !Shape(n->src[3], GGML_TYPE_F16, {768, shared_ ? 32 : 1, 1, 1}) ||
          !Shape(n->src[4], GGML_TYPE_F32, {64, 1, 1, 1}) ||
          !Shape(n, GGML_TYPE_F32, {512, 64, 1, 1}) || Address(k) != raw_ || Address(v) != raw_)
        return Fail("decode raw-only attention geometry/state address");
      for (const auto* t : {k, v}) {
        const auto* root = Unwrap(t);
        if (!root || (root->op != GGML_OP_NONE && root->op != GGML_OP_SET_ROWS) ||
            Address(root) != raw_)
          return Fail("decode raw attention has an unrelated state root");
      }
      const auto* mask = n->src[3];
      if (!ggml_is_contiguous(mask)) return Fail("decode mask is not its complete packed extent");
      // Both actual arms build their causal mask on device from positions.
      // Legacy Dsv4Graph uses exact rows; shared DeclareMasks pads to32.
      if (auto valid = kg::CheckGemma4Mask(mask); !valid) return valid;
      const auto row_layout = shared_ ? kg::CausalMaskRows::kPad32 : kg::CausalMaskRows::kExact;
      if (kg::LlmpOpInt(mask, 0) != 0 || kg::LlmpOpInt(mask, 1) != 1 ||
          kg::LlmpOpInt(mask, 2) != 768 || kg::LlmpOpInt(mask, 3) != 128 ||
          kg::LlmpOpInt(mask, 4) != static_cast<std::int32_t>(kContext) ||
          kg::LlmpOpInt(mask, 5) != static_cast<int>(row_layout) ||
          kg::LlmpOpInt(mask, 6) != static_cast<int>(kg::MaskPolicy::kCausal) ||
          !Shape(mask->src[0], GGML_TYPE_I32, {1, 1, 1, 1}))
        return Fail("decode mask is not its arm's exact fixed-position recipe");
      auto plan = kg::PlanFlashAttnMma(launch, n, impl == kg::kFlashAttnMmaWideName);
      if (!plan) return std::unexpected(plan.error());
      plan_ = *plan;
      attention_ = n;
      query_record_ = count_;
      if (auto r = Add("q", q); !r) return r;
      if (auto r = Add("k", k); !r) return r;
      if (auto r = Add("v", v); !r) return r;
      mask_record_ = count_;
      if (auto r = Add("mask-full", mask); !r) return r;
      if (auto r = Add("mask", mask, false, 0, 768 * sizeof(std::uint16_t)); !r) return r;
      positions_record_ = count_;
      if (auto r = Add("mask-positions", mask->src[0]); !r) return r;
      if (auto r = WeightRecord("sinks", n->src[4], 8); !r) return r;
      return Add("attention-out", n, true);
    }
    if (impl != kg::kHcPostName)
      return Fail("decode HC-post implementation differs: " + std::string(impl));
    if (!Shape(n, GGML_TYPE_F32, {4096, 4, 1, 1}) ||
        !Shape(n->src[0], GGML_TYPE_F32, {4096, 1, 1, 1}) ||
        !Shape(n->src[1], GGML_TYPE_F32, {4096, 4, 1, 1}) ||
        !Shape(n->src[2], GGML_TYPE_F32, {4, 1, 1, 1}) ||
        !Shape(n->src[3], GGML_TYPE_F32, {4, 4, 1, 1}) || Address(n->src[1]) != Address(residual_))
      return Fail("decode HC-post exact residual/post/comb tuple");
    // Shared HC-pre lowers its two carry outputs as packed CONT copies of
    // the authenticated tail views. Legacy consumes those tail views directly.
    if (shared_) {
      if (!PackedTail(n->src[2], false) || !PackedTail(n->src[3], true))
        return Fail("decode shared native HC-post lacks its exact packed HC-pre tails");
    } else if (Address(n->src[2]) != Address(pre_) + 4096 * sizeof(float) ||
               Address(n->src[3]) != Address(pre_) + 4100 * sizeof(float)) {
      return Fail("decode legacy HC-post exact HC-pre tail addresses");
    }
    if (auto valid = kg::CheckHcPost(n); !valid) return valid;
    if (!ffn) {
      if (projection_count_ != 2 || n->src[0] != projection_output_[1])
        return Fail("decode native post does not consume both observed projections");
      const auto* product = Unwrap(n->src[0]);
      if (!product ||
          (product->op != GGML_OP_MUL_MAT && kg::LlmpOpOf(product) != kg::LlmpOp::kVecQ) ||
          Address(product->src[0]) != Weight(target_.binding.layers.at(0).out_b.index))
        return Fail("decode attention post does not consume layer0 output-B");
      post_ = n;
    }
    if (ffn && (!combine_output_ || n->src[0] != combine_output_))
      return Fail("decode FFN post does not consume authenticated routed/shared combine");
    constexpr std::array names{"cur", "residual", "post", "comb"};
    for (std::size_t i = 0; i < 4; ++i)
      if (auto r = Add(prefix + "post-" + names[i], n->src[i]); !r) return r;
    return Add(prefix + "post-out", n, true);
  }
  // Only zero-offset reshapes/permutations preserve the selected producer;
  // the one CONT in shared output-A assembly is separately admitted below.
  // Only dense layer0 FFN sources are admitted here. Routed experts occupy
  // separately paged slabs and need their own address authority before they
  // can be added to this probe; shape/type coincidence is not such authority.
  bool DenseFfnWeight(const ggml_tensor* t, const md::Dsv4Tensor& binding, ggml_type type,
                      std::array<std::int64_t, 4> shape) const {
    return Shape(t, type, shape) && t->op == GGML_OP_NONE && !t->view_src &&
           ggml_is_contiguous(t) && Address(t) == Weight(binding.index) &&
           ggml_nbytes(t) == target_.artifact.resources()[binding.index].bytes.value();
  }
  bool FfnNormalized(const ggml_tensor* t) const {
    return Shape(t, GGML_TYPE_F32, {4096, 1, 1, 1}) && ggml_is_contiguous(t) &&
           t->op == GGML_OP_VIEW && t->src[0] == pre_ && t->view_src == pre_ && t->view_offs == 0 &&
           Address(t) == Address(pre_);
  }
  bool SharedFfnProduct(const ggml_tensor* n, const md::Dsv4Tensor& weight, bool glu) const {
    const auto& l = target_.binding.layers.at(0);
    if (!n || kg::LlmpOpOf(n) != kg::LlmpOp::kVecQ || !kg::CheckVecQ(n) ||
        !DenseFfnWeight(n->src[0], weight, GGML_TYPE_Q5_K, {4096, 2048, 1, 1}) ||
        kg::LlmpOpInt(n, 0) != 1 || kg::LlmpOpInt(n, 1) != 0 ||
        kg::LlmpOpInt(n, 2) != (glu ? static_cast<int>(kg::VecQGlu::kSwigluClamp) : 0) ||
        !Shape(n, GGML_TYPE_F32, {2048, 1, 1, 1}) || !ggml_is_contiguous(n))
      return false;
    const auto* q = n->src[1];
    if (!q || kg::LlmpOpOf(q) != kg::LlmpOp::kQuantizeQ8 || !kg::CheckQuantizeQ8(q) ||
        !FfnNormalized(q->src[0]))
      return false;
    if (!glu) return n->src[2] == nullptr;
    return DenseFfnWeight(n->src[2], l.gate_shexp, GGML_TYPE_Q5_K, {4096, 2048, 1, 1}) &&
           kg::LlmpOpFloat(n, 3) == md::Dsv4Flash().swiglu_limit_shared;
  }
  std::expected<bool, kg::KernelFailure> FfnProducts(kg::LaunchContext& launch,
                                                     std::span<ggml_tensor* const> nodes,
                                                     std::string_view impl, bool after) {
    if (ffn_pending_) {
      if (!after || nodes.size() != 1 || nodes[0] != ffn_pending_ || impl != ffn_impl_ ||
          stream_.handle != launch.native_stream().handle)
        return Fail("decode FFN product before/after pairing changed");
      if (auto copied = Copy(count_ - 1); !copied) return std::unexpected(copied.error());
      if (ffn_count_ == 0)
        router_output_ = ffn_pending_;
      else
        shared_output_ = ffn_pending_;
      ++ffn_count_;
      ffn_pending_ = nullptr;
      return true;
    }
    if (after || ffn_count_ == 2 || nodes.size() != 1) return false;
    auto* n = nodes[0];
    if (!n || (n->op != GGML_OP_MUL_MAT && kg::LlmpOpOf(n) != kg::LlmpOp::kVecQ)) return false;
    const auto& l = target_.binding.layers.at(0);
    const bool router = ffn_count_ == 0;
    const auto& binding = router ? l.router : l.down_shexp;
    if (Address(n->src[0]) != Weight(binding.index)) return false;
    if (stream_.handle != launch.native_stream().handle) return Fail("decode FFN stream changed");
    const ggml_tensor* input = n->src[1];
    if (router) {
      if (n->op != GGML_OP_MUL_MAT ||
          impl != (shared_ ? "ggml.mul_mat.mmvf" : "llmp.mul_mat.mmvf_rows") ||
          !DenseFfnWeight(n->src[0], binding, GGML_TYPE_BF16, {4096, 256, 1, 1}) ||
          !FfnNormalized(input) || n->op_params[0] != GGML_PREC_F32 ||
          !Shape(n, GGML_TYPE_F32, {256, 1, 1, 1}) || !ggml_is_contiguous(n))
        return Fail("decode FFN router lost source/input/output authority");
    } else {
      if (!DenseFfnWeight(n->src[0], binding, GGML_TYPE_Q6_K, {2048, 4096, 1, 1}) ||
          !Shape(n, GGML_TYPE_F32, {4096, 1, 1, 1}) || !ggml_is_contiguous(n))
        return Fail("decode shared-expert down resource/output differs");
      if (!shared_) {
        if (impl != "llmp.vecq" || kg::LlmpOpOf(n) != kg::LlmpOp::kVecQ || !kg::CheckVecQ(n) ||
            kg::LlmpOpInt(n, 0) != 1 || kg::LlmpOpInt(n, 1) != 0 || kg::LlmpOpInt(n, 2) != 0 ||
            !input || kg::LlmpOpOf(input) != kg::LlmpOp::kQuantizeQ8 || !kg::CheckQuantizeQ8(input))
          return Fail("decode legacy shared down recipe differs");
        input = input->src[0];
        if (!SharedFfnProduct(input, l.up_shexp, true))
          return Fail("decode legacy shared GLU source differs");
      } else {
        if (impl != "ggml.mul_mat.mmvq" || n->op != GGML_OP_MUL_MAT || !input ||
            input->op != GGML_OP_GLU || ggml_get_glu_op(input) != GGML_GLU_OP_SWIGLU_CLAMP ||
            input->op_params[1] != 0 ||
            std::bit_cast<float>(input->op_params[3]) != md::Dsv4Flash().swiglu_limit_shared ||
            !SharedFfnProduct(input->src[0], l.gate_shexp, false) ||
            !SharedFfnProduct(input->src[1], l.up_shexp, false))
          return Fail("decode shared GLU/down producer recipe differs");
      }
      if (!Shape(input, GGML_TYPE_F32, {2048, 1, 1, 1}) || !ggml_is_contiguous(input))
        return Fail("decode shared GLU complete payload differs");
    }
    const auto first = count_;
    if (auto added = Add(router ? "ffn-router-in" : "ffn-shared-act", input); !added)
      return std::unexpected(added.error());
    if (auto added = Add(router ? "ffn-router-out" : "ffn-shared-down", n, true); !added)
      return std::unexpected(added.error());
    if (auto copied = Copy(first); !copied) return std::unexpected(copied.error());
    ffn_pending_ = n;
    ffn_impl_ = impl;
    return true;
  }
  // Observe the actual selected-ID recipe before and after its launch. Its
  // IDs came from the already observed, resource-authenticated hash lookup;
  // its logits are the completed router projection. No shape-only authority.
  std::expected<bool, kg::KernelFailure> SelectedRouteRead(kg::LaunchContext& launch,
                                                           std::span<ggml_tensor* const> nodes,
                                                           std::string_view impl, bool after) {
    if (selected_pending_) {
      if (!after || nodes.size() != 1 || nodes[0] != selected_pending_ ||
          impl != kg::kDsv4RouteName || stream_.handle != launch.native_stream().handle)
        return Fail("selected route before/after producer or stream changed");
      selected_route_ = selected_pending_;
      selected_pending_ = nullptr;
      return true;
    }
    if (after || nodes.size() != 1 || !nodes[0] || kg::LlmpOpOf(nodes[0]) != kg::LlmpOp::kDsv4Route)
      return false;
    const auto* n = nodes[0];
    if (stage_ != 6 || selected_route_ || impl != kg::kDsv4RouteName || !kg::CheckDsv4Route(n) ||
        !Shape(n, GGML_TYPE_I32, {12, 1, 1, 1}) || !ggml_is_contiguous(n) || !router_output_ ||
        n->src[0] != router_output_ || !hash_ids_ || hash_pending_ || n->src[1] != hash_ids_ ||
        n->src[2] || kg::LlmpOpInt(n, 0) != 6 || kg::LlmpOpInt(n, 1) != 1 ||
        kg::LlmpOpInt(n, 4) != 1 || kg::LlmpOpFloat(n, 2) != 0x1p-14F ||
        kg::LlmpOpFloat(n, 3) != md::Dsv4Flash().expert_weights_scale ||
        stream_.handle != launch.native_stream().handle)
      return Fail("selected route lost observed router/hash input or recipe authority");
    selected_pending_ = n;
    return true;
  }
  // Staged inputs have planned lifetimes ending at their actual last reader.
  // Copy this input before GET_ROWS, not later when GLU merely consumes IDs:
  // the staged token's address is allowed to hold another activation by then.
  std::expected<bool, kg::KernelFailure> HashRead(kg::LaunchContext& launch,
                                                  std::span<ggml_tensor* const> nodes,
                                                  std::string_view impl, bool after) {
    if (hash_pending_) {
      if (!after || nodes.size() != 1 || nodes[0] != hash_pending_ || impl != "ggml.get_rows.ext" ||
          stream_.handle != launch.native_stream().handle)
        return Fail("early hash lookup before/after producer or stream changed");
      if (auto copied = Copy(ids_record_); !copied) return std::unexpected(copied.error());
      hash_ids_ = hash_pending_;
      hash_pending_ = nullptr;
      return true;
    }
    if (after || nodes.size() != 1 || !nodes[0]) return false;
    const auto* n = nodes[0];
    const auto& table = target_.binding.layers.at(0).tid2eid;
    if (n->op != GGML_OP_GET_ROWS || Address(n->src[0]) != Weight(table.index)) return false;
    const ggml_tensor* token = nullptr;
    if (hash_token_ || impl != "ggml.get_rows.ext" || !HashIds(n, token) ||
        !ggml_is_contiguous(n) ||
        (stream_.handle && stream_.handle != launch.native_stream().handle))
      return Fail("early layer0 hash lookup source, identity or stream differs");
    stream_ = launch.native_stream();
    token_record_ = count_;
    if (auto added = Add("ffn-hash-token", token); !added) return std::unexpected(added.error());
    ids_record_ = count_;
    if (auto added = Add("ffn-routed-ids", n, true); !added) return std::unexpected(added.error());
    if (auto copied = Copy(token_record_); !copied) return std::unexpected(copied.error());
    hash_token_ = token;
    hash_pending_ = n;
    return true;
  }
  bool HashIds(const ggml_tensor* ids, const ggml_tensor*& token) {
    if (!Shape(ids, GGML_TYPE_I32, {6, 1, 1, 1})) return false;
    const auto& table = target_.binding.layers.at(0).tid2eid;
    if (shared_) {
      if (ids->op == GGML_OP_VIEW) {
        if (!selected_route_ || selected_pending_ || ids->src[0] != selected_route_ ||
            ids->view_src != selected_route_ || ids->view_offs != 0 ||
            Address(ids) != Address(selected_route_) || !ggml_is_contiguous(ids) ||
            selected_route_->src[1] != hash_ids_)
          return false;
        return HashIds(hash_ids_, token);
      }
      if (ids->op != GGML_OP_GET_ROWS ||
          !DenseFfnWeight(ids->src[0], table, GGML_TYPE_I32, {6, 129280, 1, 1}))
        return false;
      token = ids->src[1];
    } else {
      const auto* route = ids->src[0];
      if (!route || !kg::CheckDsv4Route(route) || ids->op != GGML_OP_VIEW ||
          ids->view_src != route || ids->view_offs != 0 || Address(ids) != Address(route) ||
          route->src[0] != router_output_ ||
          !DenseFfnWeight(route->src[1], table, GGML_TYPE_I32, {6, 129280, 1, 1}) ||
          kg::LlmpOpInt(route, 0) != 6 || kg::LlmpOpInt(route, 1) != 1 ||
          kg::LlmpOpFloat(route, 2) != 0x1p-14F ||
          kg::LlmpOpFloat(route, 3) != md::Dsv4Flash().expert_weights_scale)
        return false;
      token = route->src[2];
      route_ = route;
    }
    return Shape(token, GGML_TYPE_I32, {1, 1, 1, 1}) && token->op == GGML_OP_NONE &&
           !token->view_src && ggml_is_contiguous(token);
  }
  bool RoutedInput(const ggml_tensor* t) const {
    if (FfnNormalized(t)) return true;
    return Shape(t, GGML_TYPE_F32, {4096, 1, 1, 1}) && t->op == GGML_OP_RESHAPE &&
           ggml_is_contiguous(t) && t->view_src == pre_ && t->view_offs == 0 &&
           Address(t) == Address(pre_) && FfnNormalized(t->src[0]);
  }
  bool RoutedGateUp(const ggml_tensor* t, std::size_t index, const ggml_tensor* ids) const {
    return t && t->op == GGML_OP_MUL_MAT_ID && MatchesRoutedArray(t->src[0], routed_[index]) &&
           RoutedInput(t->src[1]) && t->src[2] == ids && Shape(t, GGML_TYPE_F32, {2048, 6, 1, 1}) &&
           ggml_is_contiguous(t);
  }
  bool RouteWeights(const ggml_tensor* view) const {
    // ggml flattens view_src/offset to the blob, but src[0] preserves the
    // exact typed-plane -> broadcast-view producer chain.
    if (!selected_route_ || selected_pending_ || !Shape(view, GGML_TYPE_F32, {1, 6, 1, 1}) ||
        view->op != GGML_OP_VIEW || !view->src[0] || view->view_src != selected_route_ ||
        view->view_offs != 24 || Address(view) != Address(selected_route_) + 24 ||
        !ggml_is_contiguous(view))
      return false;
    const auto* weights = view->src[0];
    return Shape(weights, GGML_TYPE_F32, {6, 1, 1, 1}) && weights->op == GGML_OP_VIEW &&
           weights->src[0] == selected_route_ && weights->view_src == selected_route_ &&
           weights->view_offs == 24 && Address(weights) == Address(selected_route_) + 24 &&
           ggml_is_contiguous(weights) && ids_ && ids_->src[0] == selected_route_;
  }

  bool WeightedSlot(const ggml_tensor* t, std::size_t slot) const {
    return Shape(t, GGML_TYPE_F32, {4096, 1, 1, 1}) && t->op == GGML_OP_VIEW &&
           t->src[0] == weighted_ && t->view_src == weighted_ && t->view_offs == slot * 16384 &&
           Address(t) == Address(weighted_) + slot * 16384 && t->nb[0] == 4;
  }
  bool OrderedSlots(const ggml_tensor* t) const {
    // Exact six-slot order, not a same-shaped reduction chosen by a name.
    for (std::size_t slot = 5; slot > 0; --slot) {
      if (!Shape(t, GGML_TYPE_F32, {4096, 1, 1, 1}) || t->op != GGML_OP_ADD ||
          !WeightedSlot(t->src[1], slot))
        return false;
      t = t->src[0];
    }
    return WeightedSlot(t, 0);
  }
  std::expected<bool, kg::KernelFailure> RoutedProducts(kg::LaunchContext& launch,
                                                        std::span<ggml_tensor* const> nodes,
                                                        std::string_view impl, bool after) {
    if (stream_.handle != launch.native_stream().handle)
      return Fail("routed probe stream changed before capture");
    if (routed_pending_) {
      if (!after || nodes.size() != 1 || nodes[0] != routed_pending_ || impl != routed_impl_ ||
          stream_.handle != launch.native_stream().handle)
        return Fail("routed boundary before/after identity changed");
      if (routed_kind_ < 2) {
        if (auto copied = Copy(routed_output_record_); !copied)
          return std::unexpected(copied.error());
        if (routed_kind_ == 0)
          routed_act_ = routed_pending_;
        else
          routed_down_ = routed_pending_;
        ++routed_count_;
      } else if (routed_kind_ == 2) {
        weighted_ = routed_pending_;
        weights_copied_ = true;
      } else {
        combine_output_ = routed_pending_;
        weights_copied_ = true;
      }
      routed_pending_ = nullptr;
      return true;
    }
    if (after || nodes.size() != 1 || !nodes[0]) return false;
    auto* n = nodes[0];
    const auto custom = kg::LlmpOpOf(n);
    bool matched = false;
    if (routed_count_ == 0 && router_output_) {
      const ggml_tensor* ids = nullptr;
      if (!shared_ && custom == kg::LlmpOp::kVecQ && Address(n->src[0]) == routed_[1].address) {
        if (impl != "llmp.vecq" || !kg::CheckVecQ(n) ||
            !MatchesRoutedArray(n->src[0], routed_[1]) ||
            !MatchesRoutedArray(n->src[3], routed_[0]) || kg::LlmpOpInt(n, 0) != 1 ||
            kg::LlmpOpInt(n, 1) != 0 ||
            kg::LlmpOpInt(n, 2) != static_cast<int>(kg::VecQGlu::kSwigluClamp) ||
            kg::LlmpOpFloat(n, 3) != md::Dsv4Flash().swiglu_limit || !n->src[1] ||
            !kg::CheckQuantizeQ8(n->src[1]) || !FfnNormalized(n->src[1]->src[0]))
          return Fail("legacy routed GLU lost array/preparation authority");
        ids = n->src[2];
        matched = true;
      } else if (shared_ && n->op == GGML_OP_GLU && n->src[0] && n->src[1] &&
                 n->src[0]->op == GGML_OP_MUL_MAT_ID &&
                 Address(n->src[0]->src[0]) == routed_[0].address) {
        ids = n->src[0]->src[2];
        if (impl != "ggml.swiglu_clamp" || ggml_get_glu_op(n) != GGML_GLU_OP_SWIGLU_CLAMP ||
            n->op_params[1] != 0 ||
            std::bit_cast<float>(n->op_params[3]) != md::Dsv4Flash().swiglu_limit ||
            !RoutedGateUp(n->src[0], 0, ids) || !RoutedGateUp(n->src[1], 1, ids))
          return Fail("shared routed GLU lost complete producer authority");
        matched = true;
      }
      if (matched) {
        const ggml_tensor* token = nullptr;
        if (!HashIds(ids, token) || !Shape(n, GGML_TYPE_F32, {2048, 6, 1, 1}) ||
            !ggml_is_contiguous(n))
          return Fail("routed GLU hash IDs or output differ");
        ids_ = ids;
        if (shared_) {
          if (hash_pending_ || selected_pending_ || !selected_route_ || !hash_token_ ||
              ids->src[0] != selected_route_ || token != hash_token_)
            return Fail("routed GLU does not consume the completed observed hash lookup");
        } else {
          token_record_ = count_;
          if (auto r = Add("ffn-hash-token", token); !r) return std::unexpected(r.error());
          ids_record_ = count_;
          if (auto r = Add("ffn-routed-ids", ids); !r) return std::unexpected(r.error());
          for (const auto record : {token_record_, ids_record_})
            if (auto r = Copy(record); !r) return std::unexpected(r.error());
        }
        routed_kind_ = 0;
        routed_output_record_ = count_;
        if (auto r = Add("ffn-routed-act", n, true); !r) return std::unexpected(r.error());
      }
    } else if (routed_count_ == 1 && Address(n->src[0]) == routed_[2].address) {
      if (!MatchesRoutedArray(n->src[0], routed_[2]) || n->src[2] != ids_ ||
          !Shape(n, GGML_TYPE_F32, {4096, 6, 1, 1}) || !ggml_is_contiguous(n))
        return Fail("routed down array/IDs/output differ");
      if (shared_) {
        if (impl != "ggml.mul_mat_id.mmvq" || n->op != GGML_OP_MUL_MAT_ID ||
            n->src[1] != routed_act_)
          return Fail("shared routed down does not consume observed GLU");
      } else if (impl != "llmp.vecq" || custom != kg::LlmpOp::kVecQ || !kg::CheckVecQ(n) ||
                 kg::LlmpOpInt(n, 0) != 1 || kg::LlmpOpInt(n, 1) != 1 || kg::LlmpOpInt(n, 2) != 0 ||
                 !n->src[1] || !kg::CheckQuantizeQ8(n->src[1]) ||
                 n->src[1]->src[0] != routed_act_) {
        return Fail("legacy routed down does not consume observed GLU");
      }
      matched = true;
      routed_kind_ = 1;
      routed_output_record_ = count_;
      if (auto r = Add("ffn-routed-down", n, true); !r) return std::unexpected(r.error());
    } else if (routed_count_ == 2) {
      if (shared_ && !weights_copied_ && n->op == GGML_OP_MUL && n->src[0] == routed_down_) {
        if (impl != "ggml.mul" || !RouteWeights(n->src[1]) ||
            !Shape(n, GGML_TYPE_F32, {4096, 6, 1, 1}) || !ggml_is_contiguous(n))
          return Fail("shared weighted expert product differs from actual route");
        const auto record = count_;
        if (auto r = Add("ffn-route-weights", n->src[1]); !r) return std::unexpected(r.error());
        if (auto r = Copy(record); !r) return std::unexpected(r.error());
        matched = true;
        routed_kind_ = 2;
      } else if (shared_output_ && !combine_output_) {
        if (shared_ && custom == kg::LlmpOp::kDsv4Combine) {
          const auto* weights = n->src[1];
          if (impl != kg::kDsv4CombineName || !kg::CheckDsv4Combine(n) ||
              kg::LlmpOpInt(n, 0) != 1 || weights_copied_ || n->src[0] != routed_down_ ||
              n->src[2] != shared_output_ || !selected_route_ || selected_pending_ ||
              !Shape(weights, GGML_TYPE_F32, {6, 1, 1, 1}) || weights->op != GGML_OP_VIEW ||
              weights->src[0] != selected_route_ || weights->view_src != selected_route_ ||
              weights->view_offs != 24 || Address(weights) != Address(selected_route_) + 24 ||
              !ggml_is_contiguous(weights) || !ids_ || ids_->src[0] != selected_route_)
            return Fail("native shared combine lost observed down/weights/shared authority");
          const auto record = count_;
          if (auto r = Add("ffn-route-weights", weights); !r) return std::unexpected(r.error());
          if (auto r = Copy(record); !r) return std::unexpected(r.error());
          matched = true;
        } else if (!shared_ && custom == kg::LlmpOp::kDsv4Combine) {
          if (impl != "llmp.dsv4.combine" || !kg::CheckDsv4Combine(n) ||
              n->src[0] != routed_down_ || n->src[1] != route_ || n->src[2] != shared_output_)
            return Fail("legacy combine does not consume observed complete operands");
          const auto record = count_;
          // The actual route's second six words are F32 weights in its I32 blob.
          if (auto r = Add("ffn-route-weights", route_, false, 24, 24); !r)
            return std::unexpected(r.error());
          if (auto r = Copy(record); !r) return std::unexpected(r.error());
          matched = true;
        } else if (shared_ && n->op == GGML_OP_ADD && n->src[1] == shared_output_) {
          if (impl != "ggml.add" || !weights_copied_ || !OrderedSlots(n->src[0]) ||
              !Shape(n, GGML_TYPE_F32, {4096, 1, 1, 1}) || !ggml_is_contiguous(n))
            return Fail("shared combine lost exact six-slot reduction ancestry");
          matched = true;
        }
        if (matched) routed_kind_ = 3;
      }
    }
    if (!matched) return false;
    if (used_ > kFirstDecodePayload)
      return Fail("routed probe exceeded complete paid payload bound");
    if (stream_.handle != launch.native_stream().handle) return Fail("routed probe stream changed");
    routed_pending_ = n;
    routed_impl_ = impl;
    routed_implementations_[routed_kind_] = impl;
    std::copy_n(n->op_params, GGML_MAX_OP_PARAMS / sizeof(std::int32_t),
                routed_params_[routed_kind_].begin());
    return true;
  }
  static const ggml_tensor* ProjectionRoot(const ggml_tensor* t) {
    for (unsigned i = 0; t && i < 8; ++i) {
      if (t->op != GGML_OP_RESHAPE && t->op != GGML_OP_PERMUTE) return t;
      if (t->view_offs != 0) return nullptr;
      t = t->src[0];
    }
    return nullptr;
  }
  std::expected<bool, kg::KernelFailure> Projection(kg::LaunchContext& launch,
                                                    std::span<ggml_tensor* const> nodes,
                                                    std::string_view impl, bool after) {
    if (projection_pending_) {
      if (!after || nodes.size() != 1 || nodes[0] != projection_pending_ ||
          impl != projection_impl_[projection_count_] ||
          stream_.handle != launch.native_stream().handle)
        return Fail("decode projection before/after pairing changed");
      if (auto copied = Copy(count_ - 1); !copied) return std::unexpected(copied.error());
      projection_output_[projection_count_++] = projection_pending_;
      projection_pending_ = nullptr;
      return true;
    }
    if (after || projection_count_ == 2 || nodes.size() != 1) return false;
    auto* n = nodes[0];
    if (!n || (n->op != GGML_OP_MUL_MAT && kg::LlmpOpOf(n) != kg::LlmpOp::kVecQ)) return false;
    const auto& layer = target_.binding.layers.at(0);
    const auto id = projection_count_ == 0 ? layer.out_a.index : layer.out_b.index;
    const auto* weight = ProjectionRoot(n->src[0]);
    if (!weight || Address(weight) != Weight(id)) return false;
    const auto* input = n->src[1];
    const bool first = projection_count_ == 0;
    const bool prepared_recipe = !shared_ || first;
    if (!prepared_recipe) {
      if (impl != "ggml.mul_mat.mmvq" || n->op != GGML_OP_MUL_MAT)
        return Fail("decode shared output-B implementation changed");
    } else {
      if (impl != "llmp.vecq" || kg::LlmpOpOf(n) != kg::LlmpOp::kVecQ)
        return Fail("decode prepared projection implementation changed");
      if (auto valid = kg::CheckVecQ(n); !valid) return std::unexpected(valid.error());
      if (!input || kg::LlmpOpOf(input) != kg::LlmpOp::kQuantizeQ8)
        return Fail("decode prepared projection lacks its actual Q8 producer");
      if (auto valid = kg::CheckQuantizeQ8(input); !valid) return std::unexpected(valid.error());
      input = input->src[0];
      if (shared_ && (!first || kg::LlmpOpInt(n, 0) != 1 || kg::LlmpOpInt(n, 1) != 1 ||
                      kg::LlmpOpInt(n, 2) != 0 || kg::LlmpOpInt(n, 4) != 0 ||
                      !Shape(input, GGML_TYPE_F32, {4096, 8, 1, 1}) || !ggml_is_contiguous(input)))
        return Fail("decode shared grouped output-A lost its exact prepared layout");
    }
    const std::array<std::int64_t, 4> weight_shape =
        first ? std::array<std::int64_t, 4>{4096, 1024, 8, 1}
              : std::array<std::int64_t, 4>{8192, 4096, 1, 1};
    if (stream_.handle != launch.native_stream().handle || weight->op != GGML_OP_NONE ||
        weight->type != GGML_TYPE_Q8_0 || !Shape(n->src[0], GGML_TYPE_Q8_0, weight_shape) ||
        ggml_nbytes(weight) != target_.artifact.resources()[id].bytes.value() || !input ||
        input->type != GGML_TYPE_F32 || !input->data || n->type != GGML_TYPE_F32 ||
        ggml_nbytes(input) != (first ? 131072U : 32768U) ||
        ggml_nbytes(n) != (first ? 32768U : 16384U))
      return Fail("decode layer0 projection resource/extent differs");
    const auto* producer = ProjectionRoot(input);
    if (first) {
      if (!producer || producer->op != GGML_OP_ROPE_BACK ||
          ProjectionRoot(producer->src[0]) != attention_)
        return Fail("decode output-A does not consume this attention's rotated output");
    } else {
      if (shared_) {
        if (!producer || producer->op != GGML_OP_CONT)
          return Fail("decode shared output-B lacks its exact packed output-A assembly");
        producer = ProjectionRoot(producer->src[0]);
      }
      if (producer != projection_output_[0])
        return Fail("decode output-B does not consume the observed output-A");
    }
    const std::string prefix = first ? "projection-a-" : "projection-b-";
    if (auto r = Add(prefix + "in", input); !r) return std::unexpected(r.error());
    if (auto copied = Copy(count_ - 1); !copied) return std::unexpected(copied.error());
    if (auto r = Add(prefix + "out", n, true); !r) return std::unexpected(r.error());
    projection_pending_ = n;
    projection_impl_[projection_count_] = impl;
    return true;
  }
  bool PackedTail(const ggml_tensor* packed, bool combination) const {
    const auto shape = combination ? std::array<std::int64_t, 4>{4, 4, 1, 1}
                                   : std::array<std::int64_t, 4>{4, 1, 1, 1};
    if (!Shape(packed, GGML_TYPE_F32, shape) || packed->op != GGML_OP_CONT ||
        !ggml_is_contiguous(packed) || !packed->src[0])
      return false;
    const auto* view = packed->src[0];
    const auto tail = (combination ? 4100U : 4096U) * sizeof(float);
    return view->op == GGML_OP_VIEW && view->src[0] == pre_ && view->view_offs == tail &&
           Shape(view, GGML_TYPE_F32, shape) && Address(view) == Address(pre_) + tail &&
           ggml_nbytes(view) == ggml_nbytes(packed);
  }
  const ggml_tensor* TailScalar(const ggml_tensor* scalar, bool combination,
                                std::size_t offset) const {
    if (!Shape(scalar, GGML_TYPE_F32, {1, 1, 1, 1}) || scalar->op != GGML_OP_VIEW ||
        scalar->view_offs != offset || !PackedTail(scalar->src[0], combination) ||
        Address(scalar) != Address(scalar->src[0]) + offset)
      return nullptr;
    return scalar->src[0];
  }
  static bool PostReshape(const ggml_tensor* t, const ggml_tensor* producer) {
    return t == producer ||
           (t && t->op == GGML_OP_RESHAPE && t->src[0] == producer && t->view_offs == 0 &&
            Shape(t, GGML_TYPE_F32, {4096, 1, 1, 1}) && Address(t) == Address(producer));
  }
  std::expected<bool, kg::KernelFailure> PrimitivePost(kg::LaunchContext& launch,
                                                       std::span<ggml_tensor* const> nodes,
                                                       std::string_view impl, bool after) {
    if (after && !primitive_pending_) return false;
    if (!after && primitive_phase_ == 0) {
      if (nodes.size() != 1 || !nodes[0] || nodes[0]->op != GGML_OP_MUL ||
          !TailScalar(nodes[0]->src[1], false, 0))
        return false;
      if (stage_ == 3 && projection_count_ != 2)
        return Fail("decode primitive post precedes complete output projections");
      primitive_phase_ = 1;
      primitive_dst_ = primitive_source_ = 0;
      primitive_current_ = nodes[0]->src[0];
      if (!Shape(primitive_current_, GGML_TYPE_F32, {4096, 1, 1, 1}) ||
          (stage_ == 3 && primitive_current_ != projection_output_[1]) ||
          (stage_ == 6 && (!combine_output_ || primitive_current_ != combine_output_)))
        return Fail("decode primitive post current is not its complete observed product");
    }
    if (primitive_phase_ == 0) return false;
    if (nodes.size() != 1 || !nodes[0] || stream_.handle != launch.native_stream().handle)
      return Fail("decode primitive post selected-node/stream bound");
    auto* n = nodes[0];
    const std::string prefix = stage_ == 3 ? "attn-" : "ffn-";
    if (after) {
      if (n != primitive_pending_ || impl != primitive_impl_)
        return Fail("decode primitive post before/after pairing changed");
      primitive_pending_ = nullptr;
      ++primitive_calls_;
      if (primitive_phase_ == 1) {
        primitive_acc_ = n;
        primitive_source_ = 0;
        primitive_phase_ = 2;
      } else if (primitive_phase_ == 2) {
        primitive_term_ = n;
        primitive_phase_ = 3;
      } else if (primitive_phase_ == 3) {
        primitive_acc_ = n;
        if (++primitive_source_ < 4) {
          primitive_phase_ = 2;
        } else if (primitive_dst_ == 0) {
          primitive_joined_ = n;
          ++primitive_dst_;
          primitive_phase_ = 1;
        } else {
          primitive_phase_ = 4;
        }
      } else {
        primitive_joined_ = n;
        if (primitive_dst_ == 3) {
          if (primitive_calls_ != 39) return Fail("decode primitive post operation count changed");
          if (auto copied = Copy(count_ - 1); !copied) return std::unexpected(copied.error());
          if (stage_ == 3) post_ = n;
          implementations_[stage_] = "ggml.mul+ggml.add+ggml.concat:39";
          primitive_phase_ = primitive_calls_ = 0;
          ++stage_;
        } else {
          ++primitive_dst_;
          primitive_phase_ = 1;
        }
      }
      return true;
    }
    if (primitive_pending_) return Fail("decode primitive post missed its after callback");
    if (primitive_phase_ == 1) {
      const auto* post = TailScalar(n->src[1], false, primitive_dst_ * sizeof(float));
      if (impl != "ggml.mul" || n->op != GGML_OP_MUL || n->src[0] != primitive_current_ || !post ||
          !Shape(n, GGML_TYPE_F32, {4096, 1, 1, 1}))
        return Fail("decode primitive post current/gate product differs");
      if (primitive_dst_ == 0) {
        primitive_post_ = post;
        for (const auto& [name, tensor] : {std::pair{prefix + "post-cur", primitive_current_},
                                           std::pair{prefix + "post-residual", residual_},
                                           std::pair{prefix + "post-post", post}}) {
          if (auto r = Add(name, tensor); !r) return std::unexpected(r.error());
          if (auto copied = Copy(count_ - 1); !copied) return std::unexpected(copied.error());
        }
      } else if (post != primitive_post_) {
        return Fail("decode primitive post changed gate parent");
      }
    } else if (primitive_phase_ == 2) {
      const auto* slice = n->src[0];
      const auto* comb =
          TailScalar(n->src[1], true, (primitive_dst_ + primitive_source_ * 4) * sizeof(float));
      if (impl != "ggml.mul" || n->op != GGML_OP_MUL || !comb ||
          !Shape(n, GGML_TYPE_F32, {4096, 1, 1, 1}) ||
          !Shape(slice, GGML_TYPE_F32, {4096, 1, 1, 1}) || slice->op != GGML_OP_VIEW ||
          slice->src[0] != residual_ ||
          slice->view_offs != primitive_source_ * 4096 * sizeof(float) ||
          Address(slice) != Address(residual_) + slice->view_offs)
        return Fail("decode primitive post residual/combination product differs");
      if (primitive_dst_ == 0 && primitive_source_ == 0) {
        primitive_comb_ = comb;
        if (auto r = Add(prefix + "post-comb", comb); !r) return std::unexpected(r.error());
        if (auto copied = Copy(count_ - 1); !copied) return std::unexpected(copied.error());
      } else if (comb != primitive_comb_) {
        return Fail("decode primitive post changed combination parent");
      }
    } else if (primitive_phase_ == 3) {
      if (impl != "ggml.add" || n->op != GGML_OP_ADD || n->src[0] != primitive_acc_ ||
          n->src[1] != primitive_term_ || !Shape(n, GGML_TYPE_F32, {4096, 1, 1, 1}))
        return Fail("decode primitive post ordered accumulation differs");
    } else {
      std::int32_t axis = -1;
      std::memcpy(&axis, n->op_params, sizeof(axis));
      if (impl != "ggml.concat" || n->op != GGML_OP_CONCAT || axis != 1 ||
          !PostReshape(n->src[0], primitive_joined_) || !PostReshape(n->src[1], primitive_acc_) ||
          !Shape(n, GGML_TYPE_F32, {4096, static_cast<std::int64_t>(primitive_dst_ + 1), 1, 1}))
        return Fail("decode primitive post final stream assembly differs");
      if (primitive_dst_ == 3)
        if (auto r = Add(prefix + "post-out", n, true); !r) return std::unexpected(r.error());
    }
    primitive_pending_ = n;
    primitive_impl_ = impl;
    return true;
  }
  std::expected<void, kg::KernelFailure> Copy(std::size_t i) {
    const auto& r = records_[i];
    const auto copy = llmp::providers::CopyAsync(stream_, host_ + r.offset, r.address, r.bytes,
                                                 llmp::providers::CopyKind::kDeviceToHost);
    return copy.ok() ? std::expected<void, kg::KernelFailure>{} : Fail(copy.text());
  }
  std::byte* host_;
  const Target& target_;
  std::uint64_t raw_;
  std::array<std::uint64_t, 9> weights_;
  std::array<RoutedArrayReceipt, 3> routed_;
  std::int32_t token_;
  const ggml_tensor *router_output_ = nullptr, *shared_output_ = nullptr, *route_ = nullptr,
                    *ids_ = nullptr, *routed_act_ = nullptr, *routed_down_ = nullptr,
                    *weighted_ = nullptr, *combine_output_ = nullptr;
  ggml_tensor* routed_pending_ = nullptr;
  const ggml_tensor *selected_pending_ = nullptr, *selected_route_ = nullptr;
  const ggml_tensor *hash_pending_ = nullptr, *hash_ids_ = nullptr, *hash_token_ = nullptr;
  std::string_view routed_impl_;
  std::array<std::string_view, 4> routed_implementations_{};
  std::array<std::array<std::int32_t, GGML_MAX_OP_PARAMS / sizeof(std::int32_t)>, 4>
      routed_params_{};
  std::size_t routed_count_ = 0, routed_kind_ = 0, routed_output_record_ = 0, token_record_ = 0,
              ids_record_ = 0;
  bool weights_copied_ = false;
  bool shared_;
  std::size_t trace_calls_ = 0;
  bool trace_capped_ = false;
  const ggml_tensor *fn_ = nullptr, *mix_ = nullptr, *residual_ = nullptr, *pre_ = nullptr,
                    *post_ = nullptr;
  ggml_tensor* pending_ = nullptr;
  const ggml_tensor* attention_ = nullptr;
  std::array<const ggml_tensor*, 2> projection_output_{};
  std::array<std::string_view, 2> projection_impl_{};
  ggml_tensor* ffn_pending_ = nullptr;
  std::size_t ffn_count_ = 0;
  std::string_view ffn_impl_;
  ggml_tensor* projection_pending_ = nullptr;
  std::size_t projection_count_ = 0;
  const ggml_tensor *primitive_current_ = nullptr, *primitive_post_ = nullptr,
                    *primitive_comb_ = nullptr, *primitive_acc_ = nullptr,
                    *primitive_term_ = nullptr, *primitive_joined_ = nullptr;
  ggml_tensor* primitive_pending_ = nullptr;
  std::string_view primitive_impl_;
  std::size_t primitive_phase_ = 0, primitive_dst_ = 0, primitive_source_ = 0, primitive_calls_ = 0;
  llmp::providers::NativeStream stream_{};
  kg::FlashAttnMmaPlan plan_{};
  std::array<Record, 64> records_{};
  std::array<std::string_view, 7> implementations_{};
  std::array<std::array<std::int32_t, GGML_MAX_OP_PARAMS / sizeof(std::int32_t)>, 7> params_{};
  std::size_t count_ = 0, first_ = 0, used_ = 0, stage_ = 0, query_record_ = 0;
  std::size_t mask_record_ = 0, positions_record_ = 0;
};
static_assert(sizeof(FirstDecodeProbe) < (32U << 10U));

// Every resource receipt comes from the actual PagedWeights owner after loading.
// This fixed diagnostic's profile/spec/binding must all agree on 43 layers.
constexpr std::size_t kDecodeLayerEnds = 43;
constexpr std::size_t kLayerEndBytes = 4096U * 4U * sizeof(float);
constexpr std::size_t kLayerEndsOffset = static_cast<std::size_t>(kFirstDecodePayload);
static_assert(kFirstDecodePayload + kDecodeLayerEnds * kLayerEndBytes < (16U << 20U));
static_assert(kFirstDecodePayload + kDecodeLayerEnds * kLayerEndBytes + 65536 + 6ULL * 517120 +
                  (1U << 20U) <
              kLayerTwoWindow);
static_assert(3 * (kFirstDecodePayload + kDecodeLayerEnds * kLayerEndBytes + 65536 + 6ULL * 517120 +
                   (1U << 20U)) <
              (64U << 20U));
static_assert((kFirstDecodeTraceCalls * 11 + 64 + 64) * 1024 + kDecodeLayerEnds * 2048 <
              (1U << 20U));
struct LayerEndResource {
  std::uint32_t index = 0;
  std::uint64_t address = 0, bytes = 0;
};
using LayerEndResources = std::array<std::array<LayerEndResource, 4>, kDecodeLayerEnds>;
bool LayerEndShape(const ggml_tensor* t, std::array<std::int64_t, 4> ne) {
  return t && t->data && t->type == GGML_TYPE_F32 && std::equal(ne.begin(), ne.end(), t->ne);
}
std::uint64_t LayerEndAddress(const ggml_tensor* t) {
  return t ? reinterpret_cast<std::uintptr_t>(t->data) : 0;
}
bool LayerEndWeight(const ggml_tensor* t, const LayerEndResource& resource,
                    std::array<std::int64_t, 4> shape) {
  return resource.address != 0 && LayerEndShape(t, shape) && t->op == GGML_OP_NONE &&
         !t->view_src && ggml_is_contiguous(t) && LayerEndAddress(t) == resource.address &&
         ggml_nbytes(t) == resource.bytes;
}
bool LayerEndTail(const ggml_tensor* t, const ggml_tensor* pre, bool shared, bool combination) {
  const auto shape = combination ? std::array<std::int64_t, 4>{4, 4, 1, 1}
                                 : std::array<std::int64_t, 4>{4, 1, 1, 1};
  if (!LayerEndShape(t, shape)) return false;
  if (shared) {
    if (t->op != GGML_OP_CONT || !ggml_is_contiguous(t) || !t->src[0]) return false;
    t = t->src[0];
  }
  const auto offset = (combination ? 4100U : 4096U) * sizeof(float);
  return LayerEndShape(t, shape) && t->op == GGML_OP_VIEW && t->src[0] == pre &&
         t->view_src == pre && t->view_offs == offset &&
         LayerEndAddress(t) == LayerEndAddress(pre) + offset &&
         ggml_nbytes(t) == (combination ? 16U : 4U) * sizeof(float);
}

TEST(Dsv4LayerEndObservation, OwnedWeightIdentityAndExactTailProducerAreRequired) {
  ggml_tensor weight{};
  weight.type = GGML_TYPE_F32;
  weight.op = GGML_OP_NONE;
  weight.data = reinterpret_cast<void*>(0x100000);
  weight.ne[0] = 3;
  weight.ne[1] = weight.ne[2] = weight.ne[3] = 1;
  weight.nb[0] = 4;
  weight.nb[1] = weight.nb[2] = weight.nb[3] = 12;
  const LayerEndResource r{.index = 7, .address = 0x100000, .bytes = 12};
  ASSERT_TRUE(LayerEndWeight(&weight, r, {3, 1, 1, 1}));
  for (unsigned mode = 0; mode < 4; ++mode) {
    auto bad = weight;
    if (mode == 0) bad.data = reinterpret_cast<void*>(0x100004);
    if (mode == 1) bad.op = GGML_OP_DUP;
    if (mode == 2) bad.view_src = &weight;
    if (mode == 3) bad.nb[0] = 8;
    EXPECT_FALSE(LayerEndWeight(&bad, r, {3, 1, 1, 1}));
  }
  ggml_tensor pre{}, view{}, packed{};
  pre.data = reinterpret_cast<void*>(0x200000);
  view.type = GGML_TYPE_F32;
  view.op = GGML_OP_VIEW;
  view.src[0] = &pre;
  view.view_src = &pre;
  view.view_offs = 4096U * 4U;
  view.data = reinterpret_cast<void*>(0x204000);
  view.ne[0] = 4;
  view.ne[1] = view.ne[2] = view.ne[3] = 1;
  view.nb[0] = 4;
  view.nb[1] = view.nb[2] = view.nb[3] = 128;
  packed = view;
  packed.op = GGML_OP_CONT;
  packed.src[0] = &view;
  packed.view_src = nullptr;
  packed.view_offs = 0;
  packed.data = reinterpret_cast<void*>(0x300000);
  packed.nb[1] = packed.nb[2] = packed.nb[3] = 16;
  ASSERT_TRUE(LayerEndTail(&view, &pre, false, false));
  ASSERT_TRUE(LayerEndTail(&packed, &pre, true, false));
  EXPECT_FALSE(LayerEndTail(&view, &pre, true, false));
  EXPECT_FALSE(LayerEndTail(&packed, &pre, false, false));
  view.view_offs += 4;
  EXPECT_FALSE(LayerEndTail(&packed, &pre, true, false));
  view.view_offs -= 4;
  view.src[0] = &weight;
  EXPECT_FALSE(LayerEndTail(&packed, &pre, true, false));
}

constexpr std::uint32_t kLearnedLayer = 3;
constexpr std::size_t kLearnedOffset = kLayerEndsOffset + kDecodeLayerEnds * kLayerEndBytes;
constexpr std::array<std::size_t, 9> kLearnedSizes{65536, 16384, 1024,  1024, 24,
                                                   24,    16384, 98304, 16384};
constexpr std::array<std::string_view, 9> kLearnedNames{"residual", "normalized",  "logits",
                                                        "bias",     "ids",         "weights",
                                                        "current",  "routed-down", "shared-down"};
constexpr std::size_t kLearnedBytes = 215088;
static_assert(kLearnedSizes[0] + kLearnedSizes[1] + kLearnedSizes[2] + kLearnedSizes[3] +
                  kLearnedSizes[4] + kLearnedSizes[5] + kLearnedSizes[6] + kLearnedSizes[7] +
                  kLearnedSizes[8] ==
              kLearnedBytes);
static_assert(kLearnedOffset + kLearnedBytes < (16U << 20U));
static_assert(3 * (kLearnedOffset + kLearnedBytes + 65536 + 6ULL * 517120 + (1U << 20U)) <
              (64U << 20U));
static_assert((kFirstDecodeTraceCalls * 11 + 64 + 64) * 1024 + kDecodeLayerEnds * 2048 + 8192 <
              (1U << 20U));
// All attention outputs are copied at their actual producer completion, before
// any following FFN can reuse workspace. The existing pinned/cache payments
// cover these 43 extra records; no allocation or cap increases.
constexpr std::size_t kAttentionOffset = kLearnedOffset + kLearnedBytes;
constexpr std::size_t kAttentionBytes = kDecodeLayerEnds * kLayerEndBytes;
static_assert(kAttentionOffset + kAttentionBytes == 12010880);
static_assert(kAttentionOffset + kAttentionBytes < (16U << 20U));
static_assert(3 * (kAttentionOffset + kAttentionBytes + 65536 + 6ULL * 517120 + (1U << 20U)) ==
              48683136);
static_assert(3 * (kAttentionOffset + kAttentionBytes + 65536 + 6ULL * 517120 + (1U << 20U)) <
              (64U << 20U));
static_assert((kFirstDecodeTraceCalls * 11 + 64 + 64) * 1024 + 2 * kDecodeLayerEnds * 2048 + 8192 <
              (1U << 20U));

constexpr std::uint32_t kSecondLearnedLayer = 8;
constexpr std::uint32_t kThirdLearnedLayer = 26;
constexpr std::array<std::uint32_t, 3> kLearnedLayers{kLearnedLayer, kSecondLearnedLayer,
                                                      kThirdLearnedLayer};
constexpr std::size_t kSecondLearnedOffset = kAttentionOffset + kAttentionBytes;
static_assert(kSecondLearnedOffset + kLearnedBytes == 12225968);
static_assert(kSecondLearnedOffset + kLearnedBytes < (16U << 20U));
static_assert(3 * (kSecondLearnedOffset + kLearnedBytes + 65536 + 6ULL * 517120 + (1U << 20U)) ==
              49328400);
static_assert(3 * (kSecondLearnedOffset + kLearnedBytes + 65536 + 6ULL * 517120 + (1U << 20U)) <
              (64U << 20U));
static_assert((kFirstDecodeTraceCalls * 11 + 64 + 64) * 1024 + 2 * kDecodeLayerEnds * 2048 +
                  2 * 8192 <
              (1U << 20U));
constexpr std::size_t kThirdLearnedOffset = kSecondLearnedOffset + kLearnedBytes;
constexpr std::array<std::size_t, 3> kLearnedOffsets{kLearnedOffset, kSecondLearnedOffset,
                                                     kThirdLearnedOffset};
static_assert(kThirdLearnedOffset + kLearnedBytes == 12441056);
static_assert(kThirdLearnedOffset + kLearnedBytes < (16U << 20U));
static_assert(3 * (kThirdLearnedOffset + kLearnedBytes + 65536 + 6ULL * 517120 + (1U << 20U)) ==
              49973664);
static_assert(49973664 < (64U << 20U));
// Fixed 16-line learned metadata: header<=256; three owned resource lines<=320
// (source names<=128); three array lines<=128; nine descriptor lines<=448
// (fixed accepted producer names/shapes/pitches and 64-bit addresses).
// These source-proved maxima total5632B;6144B per set leaves the same
// whole-arm1MiB allowance for the additional actual shared-input receipt.
constexpr std::size_t kLearnedMetadataBytes = 6144;
constexpr std::size_t kSharedInputMetadataBytes = 1536;
static_assert(256 + 3 * 320 + 3 * 128 + 9 * 448 <= kLearnedMetadataBytes);
static_assert((kFirstDecodeTraceCalls * 11 + 64 + 64) * 1024 + 2 * kDecodeLayerEnds * 2048 +
                  kLearnedLayers.size() * kLearnedMetadataBytes + kSharedInputMetadataBytes ==
              1048064);
static_assert(1048064 < (1U << 20U));
constexpr std::size_t kSharedInputOffset = kThirdLearnedOffset + kLearnedBytes;
constexpr std::size_t kSharedInputBytes = 2048 * sizeof(float);
static_assert(kSharedInputOffset + kSharedInputBytes == 12449248);
static_assert(kSharedInputOffset + kSharedInputBytes < (16U << 20U));
static_assert(3 * (kSharedInputOffset + kSharedInputBytes + 65536 + 6ULL * 517120 + (1U << 20U)) ==
              49998240);
static_assert(49998240 < (64U << 20U));
static_assert(256 + 2 * 320 + 512 <= kSharedInputMetadataBytes);

constexpr std::size_t LearnedSlot(std::size_t layer) {
  for (std::size_t i = 0; i < kLearnedLayers.size(); ++i)
    if (kLearnedLayers[i] == layer) return i;
  return kLearnedLayers.size();
}
bool IsLearnedLayer(std::size_t layer) { return LearnedSlot(layer) < kLearnedLayers.size(); }

TEST(Dsv4LearnedRouteObservation, ThreeLayersKeepDistinctPaidCaptureDomains) {
  for (std::size_t i = 0; i < kLearnedLayers.size(); ++i) {
    EXPECT_EQ(LearnedSlot(kLearnedLayers[i]), i);
    EXPECT_TRUE(IsLearnedLayer(kLearnedLayers[i]));
    EXPECT_LT(kLearnedOffsets[i] + kLearnedBytes, 16U << 20U);
    for (std::size_t j = 0; j < i; ++j)
      EXPECT_LE(kLearnedOffsets[j] + kLearnedBytes, kLearnedOffsets[i]);
  }
  for (const auto layer : {0U, 2U, 4U, 7U, 9U, 25U, 27U, 43U}) {
    EXPECT_FALSE(IsLearnedLayer(layer));
    EXPECT_EQ(LearnedSlot(layer), kLearnedLayers.size());
  }
}

// This is source/edge authentication, not an inference from matching geometry.
// The post must use the exact tails and residual of the owned attention pre.
const ggml_tensor* AttentionPostSource(const ggml_tensor* post, bool shared,
                                       const std::array<LayerEndResource, 4>& r) {
  if (!post || post->op != GGML_OP_DSV4_HC_POST || !LayerEndShape(post, {4096, 4, 1, 1}) ||
      !ggml_is_contiguous(post) || ggml_nbytes(post) != kLayerEndBytes || !post->src[2])
    return nullptr;
  const auto* tail = post->src[2];
  if (shared) {
    if (tail->op != GGML_OP_CONT || !tail->src[0]) return nullptr;
    tail = tail->src[0];
  }
  const auto* pre = tail->src[0];
  if (!pre || kg::LlmpOpOf(pre) != kg::LlmpOp::kDsv4HcPre || !kg::CheckDsv4HcPre(pre) ||
      !pre->src[0] || kg::LlmpOpOf(pre->src[0]) != kg::LlmpOp::kDsv4HcMix ||
      !kg::CheckDsv4HcMix(pre->src[0]) || pre->src[0]->src[0] != pre->src[1] ||
      post->src[1] != pre->src[1] ||
      !LayerEndWeight(pre->src[0]->src[1], r[0], {16384, 24, 1, 1}) ||
      !LayerEndWeight(pre->src[2], r[1], {3, 1, 1, 1}) ||
      !LayerEndWeight(pre->src[3], r[2], {24, 1, 1, 1}) ||
      !LayerEndWeight(pre->src[4], r[3], {4096, 1, 1, 1}) ||
      !LayerEndTail(post->src[2], pre, shared, false) ||
      !LayerEndTail(post->src[3], pre, shared, true))
    return nullptr;
  const auto& profile = md::Dsv4Flash();
  if (kg::LlmpOpFloat(pre, 0) != profile.rms_eps || kg::LlmpOpFloat(pre, 1) != profile.hc_eps ||
      kg::LlmpOpInt(pre, 2) != static_cast<std::int32_t>(profile.sinkhorn_iterations))
    return nullptr;
  return pre;
}

TEST(Dsv4LayerEndObservation, AttentionPostRequiresOwnedPreMixAndExactTails) {
  auto arena = kg::TensorArena::Create(32);
  ASSERT_TRUE(arena);
  auto* c = arena->context();
  std::uint64_t next = 0x100000;
  const auto bind = [&](ggml_tensor* t) {
    kg::TensorArena::Bind(t, next);
    next += ((ggml_nbytes(t) + 255U) / 256U) * 256U;
    return t;
  };
  auto* x = bind(ggml_new_tensor_2d(c, GGML_TYPE_F32, 4096, 4));
  auto* fn = bind(ggml_new_tensor_2d(c, GGML_TYPE_F32, 16384, 24));
  auto* scale = bind(ggml_new_tensor_1d(c, GGML_TYPE_F32, 3));
  auto* base = bind(ggml_new_tensor_1d(c, GGML_TYPE_F32, 24));
  auto* norm = bind(ggml_new_tensor_1d(c, GGML_TYPE_F32, 4096));
  const auto& profile = md::Dsv4Flash();
  auto* mix = bind(kg::Dsv4HcMix(c, x, fn));
  auto* pre = bind(kg::Dsv4HcPre(c, mix, x, scale, base, norm, profile.rms_eps, profile.hc_eps,
                                 static_cast<std::int32_t>(profile.sinkhorn_iterations)));
  auto* post = ggml_view_1d(c, pre, 4, 4096U * 4U);
  auto* comb = ggml_view_2d(c, pre, 4, 4, 16, 4100U * 4U);
  auto* current = bind(ggml_new_tensor_1d(c, GGML_TYPE_F32, 4096));
  auto* output = bind(ggml_dsv4_hc_post(c, current, x, post, comb));
  std::array<LayerEndResource, 4> receipts{};
  const std::array weights{fn, scale, base, norm};
  for (std::size_t i = 0; i < weights.size(); ++i)
    receipts[i] = {.index = static_cast<std::uint32_t>(i),
                   .address = LayerEndAddress(weights[i]),
                   .bytes = ggml_nbytes(weights[i])};
  ASSERT_EQ(AttentionPostSource(output, false, receipts), pre);
  EXPECT_EQ(AttentionPostSource(output, true, receipts), nullptr);
  for (std::size_t i = 0; i < receipts.size(); ++i) {
    auto wrong = receipts;
    wrong[i].address += 256;
    EXPECT_EQ(AttentionPostSource(output, false, wrong), nullptr);
  }
  output->src[1] = current;
  EXPECT_EQ(AttentionPostSource(output, false, receipts), nullptr);
  output->src[1] = x;
  post->view_offs += 4;
  EXPECT_EQ(AttentionPostSource(output, false, receipts), nullptr);
  post->view_offs -= 4;
  auto* packed_post = bind(ggml_cont(c, post));
  auto* packed_comb = bind(ggml_cont(c, comb));
  output->src[2] = packed_post;
  output->src[3] = packed_comb;
  ASSERT_EQ(AttentionPostSource(output, true, receipts), pre);
  EXPECT_EQ(AttentionPostSource(output, false, receipts), nullptr);
  packed_comb->src[0] = post;
  EXPECT_EQ(AttentionPostSource(output, true, receipts), nullptr);
}

bool AttentionConsumer(const ggml_tensor* mix, const ggml_tensor* completed,
                       std::uint64_t captured_address) {
  return mix && completed && mix->src[0] == completed && captured_address != 0 &&
         LayerEndAddress(completed) == captured_address;
}
TEST(Dsv4LayerEndObservation, AttentionConsumerRequiresExactCompletedProducer) {
  ggml_tensor producer{}, alias{}, mix{};
  producer.data = reinterpret_cast<void*>(0x100000);
  alias.data = producer.data;
  mix.src[0] = &producer;
  EXPECT_TRUE(AttentionConsumer(&mix, &producer, 0x100000));
  EXPECT_FALSE(AttentionConsumer(&mix, nullptr, 0x100000));
  EXPECT_FALSE(AttentionConsumer(&mix, &alias, 0x100000));
  EXPECT_FALSE(AttentionConsumer(&mix, &producer, 0x100004));
  EXPECT_FALSE(AttentionPostSource(&producer, false, {}));
  EXPECT_FALSE(AttentionPostSource(&producer, true, {}));
}

// Descriptor-only proof of the exact ordinary learned route. Actual launch
// completion is recorded separately before these tensor outputs are copied.
bool LearnedShape(const ggml_tensor* t, ggml_type type, std::int64_t width) {
  return t && t->data && t->type == type && t->ne[0] == width && t->ne[1] == 1 && t->ne[2] == 1 &&
         t->ne[3] == 1 && ggml_is_contiguous(t);
}
bool LearnedWeight(const ggml_tensor* t, const LayerEndResource& r, ggml_type type,
                   std::array<std::int64_t, 4> shape) {
  return t && t->data && t->type == type && t->op == GGML_OP_NONE && !t->view_src &&
         std::equal(shape.begin(), shape.end(), t->ne) && ggml_is_contiguous(t) &&
         LayerEndAddress(t) == r.address && r.address != 0 && ggml_nbytes(t) == r.bytes;
}
// Exact completed source chain of the shared-expert GLU. The legacy operation
// fuses the two products; the component engine materializes both. Neither form
// licenses a synthetic separate legacy product or a reread after scratch reuse.
bool LearnedSharedVector(const ggml_tensor* n, const ggml_tensor* pre,
                         const LayerEndResource& weight, const LayerEndResource* gate = nullptr) {
  if (!n || kg::LlmpOpOf(n) != kg::LlmpOp::kVecQ || !kg::CheckVecQ(n) || kg::LlmpOpInt(n, 0) != 1 ||
      kg::LlmpOpInt(n, 1) != 0 ||
      kg::LlmpOpInt(n, 2) != (gate ? static_cast<int>(kg::VecQGlu::kSwigluClamp) : 0) ||
      !LearnedShape(n, GGML_TYPE_F32, 2048) ||
      !LearnedWeight(n->src[0], weight, GGML_TYPE_Q6_K, {4096, 2048, 1, 1}))
    return false;
  const auto* q = n->src[1];
  if (!q || kg::LlmpOpOf(q) != kg::LlmpOp::kQuantizeQ8 || !kg::CheckQuantizeQ8(q) || !q->src[0])
    return false;
  const auto* input = q->src[0];
  if (!LearnedShape(input, GGML_TYPE_F32, 4096) || input->op != GGML_OP_VIEW ||
      input->src[0] != pre || input->view_src != pre || input->view_offs != 0 ||
      LayerEndAddress(input) != LayerEndAddress(pre))
    return false;
  return gate ? (LearnedWeight(n->src[2], *gate, GGML_TYPE_Q6_K, {4096, 2048, 1, 1}) &&
                 kg::LlmpOpFloat(n, 3) == md::Dsv4Flash().swiglu_limit_shared)
              : n->src[2] == nullptr;
}
bool LearnedSharedInput(const ggml_tensor* n, const ggml_tensor* pre, bool shared,
                        const std::array<LayerEndResource, 2>& sources) {
  if (!LearnedShape(n, GGML_TYPE_F32, 2048)) return false;
  if (!shared) return LearnedSharedVector(n, pre, sources[1], &sources[0]);
  return n->op == GGML_OP_GLU && ggml_get_glu_op(n) == GGML_GLU_OP_SWIGLU_CLAMP &&
         n->op_params[1] == 0 &&
         std::bit_cast<float>(n->op_params[3]) == md::Dsv4Flash().swiglu_limit_shared &&
         LearnedSharedVector(n->src[0], pre, sources[0]) &&
         LearnedSharedVector(n->src[1], pre, sources[1]);
}
TEST(Dsv4LearnedRouteObservation, SharedInputRequiresOwnedGateUpAndNormalizedProducer) {
  auto arena = kg::TensorArena::Create(32);
  ASSERT_TRUE(arena);
  auto* c = arena->context();
  std::uint64_t next = 0x100000;
  const auto bind = [&](ggml_tensor* t) {
    kg::TensorArena::Bind(t, next);
    next += ((ggml_nbytes(t) + 255U) / 256U) * 256U;
    return t;
  };
  auto* pre = bind(ggml_new_tensor_1d(c, GGML_TYPE_F32, 4128));
  auto* input = ggml_view_1d(c, pre, 4096, 0);
  auto* q = bind(kg::QuantizeQ8(c, input));
  auto* gate = bind(ggml_new_tensor_2d(c, GGML_TYPE_Q6_K, 4096, 2048));
  auto* up = bind(ggml_new_tensor_2d(c, GGML_TYPE_Q6_K, 4096, 2048));
  const std::array<LayerEndResource, 2> sources{
      LayerEndResource{.index = 0, .address = LayerEndAddress(gate), .bytes = ggml_nbytes(gate)},
      LayerEndResource{.index = 1, .address = LayerEndAddress(up), .bytes = ggml_nbytes(up)}};
  auto* legacy = bind(kg::VecQ(c, up, q, nullptr, 1, false, gate, kg::VecQGlu::kSwigluClamp,
                               md::Dsv4Flash().swiglu_limit_shared));
  auto* gp = bind(kg::VecQ(c, gate, q, nullptr, 1, false));
  auto* upv = bind(kg::VecQ(c, up, q, nullptr, 1, false));
  auto* shared = bind(ggml_swiglu_clamp(c, gp, upv, md::Dsv4Flash().swiglu_limit_shared));
  ASSERT_TRUE(LearnedSharedInput(legacy, pre, false, sources));
  ASSERT_TRUE(LearnedSharedInput(shared, pre, true, sources));
  EXPECT_FALSE(LearnedSharedInput(legacy, pre, true, sources));
  EXPECT_FALSE(LearnedSharedInput(shared, pre, false, sources));
  auto bad = sources;
  bad[0].address += 256;
  EXPECT_FALSE(LearnedSharedInput(legacy, pre, false, bad));
  EXPECT_FALSE(LearnedSharedInput(shared, pre, true, bad));
  bad = sources;
  bad[1].bytes -= 256;
  EXPECT_FALSE(LearnedSharedInput(shared, pre, true, bad));
  std::swap(shared->src[0], shared->src[1]);
  EXPECT_FALSE(LearnedSharedInput(shared, pre, true, sources));
  std::swap(shared->src[0], shared->src[1]);
  input->view_offs = 4;
  EXPECT_FALSE(LearnedSharedInput(legacy, pre, false, sources));
  EXPECT_FALSE(LearnedSharedInput(shared, pre, true, sources));
  input->view_offs = 0;
  EXPECT_FALSE(LearnedSharedInput(shared, upv, true, sources));
}

bool LearnedSort(const ggml_tensor* sort, const ggml_tensor* logits, const LayerEndResource& bias) {
  if (!LearnedShape(sort, GGML_TYPE_I32, 256) || sort->op != GGML_OP_ARGSORT ||
      sort->op_params[0] != GGML_SORT_ORDER_DESC)
    return false;
  const auto* add = sort->src[0];
  if (!LearnedShape(add, GGML_TYPE_F32, 256) || add->op != GGML_OP_ADD ||
      !LearnedWeight(add->src[1], bias, GGML_TYPE_F32, {256, 1, 1, 1}))
    return false;
  const auto* root = add->src[0];
  const auto* soft = root ? root->src[0] : nullptr;
  return LearnedShape(root, GGML_TYPE_F32, 256) && root->op == GGML_OP_SQRT &&
         LearnedShape(soft, GGML_TYPE_F32, 256) && soft->op == GGML_OP_UNARY &&
         ggml_get_unary_op(soft) == GGML_UNARY_OP_SOFTPLUS && soft->src[0] == logits;
}
bool LearnedIds(const ggml_tensor* ids, const ggml_tensor* sort) {
  return ids && ids->type == GGML_TYPE_I32 && ids->ne[0] == 6 && ids->ne[1] == 1 &&
         ids->ne[2] == 1 && ids->ne[3] == 1 && ids->nb[0] == 4 && ids->op == GGML_OP_VIEW &&
         ids->src[0] == sort && ids->view_src == sort && ids->view_offs == 0 &&
         LayerEndAddress(ids) == LayerEndAddress(sort) && ggml_nbytes(ids) == 24;
}
bool LearnedWeights(const ggml_tensor* scale, const ggml_tensor* sort, float factor) {
  if (!LearnedShape(scale, GGML_TYPE_F32, 6) || scale->op != GGML_OP_SCALE ||
      std::bit_cast<float>(scale->op_params[0]) != factor ||
      std::bit_cast<float>(scale->op_params[1]) != 0.0F)
    return false;
  const auto* div = scale->src[0];
  if (!LearnedShape(div, GGML_TYPE_F32, 6) || div->op != GGML_OP_DIV) return false;
  const auto* weights = div->src[0];
  const auto* clamp = div->src[1];
  if (!LearnedShape(weights, GGML_TYPE_F32, 6) || weights->op != GGML_OP_RESHAPE ||
      !LearnedShape(clamp, GGML_TYPE_F32, 1) || clamp->op != GGML_OP_CLAMP ||
      std::bit_cast<float>(clamp->op_params[0]) != 0x1p-14F ||
      std::bit_cast<float>(clamp->op_params[1]) != std::numeric_limits<float>::infinity())
    return false;
  const auto* sum = clamp->src[0];
  const auto* gather = weights->src[0];
  if (!LearnedShape(sum, GGML_TYPE_F32, 1) || sum->op != GGML_OP_SUM_ROWS ||
      sum->src[0] != weights || !LayerEndShape(gather, {1, 6, 1, 1}) ||
      !ggml_is_contiguous(gather) || gather->op != GGML_OP_GET_ROWS ||
      !LearnedIds(gather->src[1], sort) || weights->view_src != gather || weights->view_offs != 0 ||
      LayerEndAddress(weights) != LayerEndAddress(gather))
    return false;
  const auto* rows = gather->src[0];
  const auto* probabilities = sort && sort->src[0] ? sort->src[0]->src[0] : nullptr;
  return LayerEndShape(rows, {1, 256, 1, 1}) && rows->op == GGML_OP_RESHAPE &&
         ggml_is_contiguous(rows) && rows->src[0] == probabilities &&
         rows->view_src == probabilities && rows->view_offs == 0 &&
         LayerEndAddress(rows) == LayerEndAddress(probabilities);
}

// The ordinary reduction is five ordered Adds of six views of one completed
// weighted expert tensor, followed by the actual completed shared product.
// A prefix of the reduction is never the FFN current consumed by HC post.
bool LearnedOrdinaryCurrent(const ggml_tensor* current, const ggml_tensor* weighted,
                            const ggml_tensor* shared) {
  if (!LearnedShape(current, GGML_TYPE_F32, 4096) || current->op != GGML_OP_ADD ||
      !LayerEndShape(weighted, {4096, 6, 1, 1}) || !ggml_is_contiguous(weighted) ||
      weighted->op != GGML_OP_MUL || !LearnedShape(shared, GGML_TYPE_F32, 4096) ||
      current->src[1] != shared)
    return false;
  const auto slot = [weighted](const ggml_tensor* t, std::size_t i) {
    return LayerEndShape(t, {4096, 1, 1, 1}) && t->op == GGML_OP_VIEW && t->src[0] == weighted &&
           t->view_src == weighted && t->view_offs == i * 16384 &&
           LayerEndAddress(t) == LayerEndAddress(weighted) + i * 16384 && t->nb[0] == 4 &&
           t->nb[1] == 98304 && t->nb[2] == 98304 && t->nb[3] == 98304;
  };
  const auto* acc = current->src[0];
  for (std::size_t i = 5; i > 0; --i) {
    if (!LearnedShape(acc, GGML_TYPE_F32, 4096) || acc->op != GGML_OP_ADD || !slot(acc->src[1], i))
      return false;
    acc = acc->src[0];
  }
  return slot(acc, 0);
}
// The completed planar native producer owns both typed six-slot views.
bool LearnedNativeWeights(const ggml_tensor* weights, const ggml_tensor* route) {
  return route && kg::LlmpOpOf(route) == kg::LlmpOp::kDsv4Route && kg::LlmpOpInt(route, 4) == 2 &&
         LearnedShape(route, GGML_TYPE_I32, 12) && LearnedShape(weights, GGML_TYPE_F32, 6) &&
         weights->op == GGML_OP_VIEW && weights->src[0] == route && weights->view_src == route &&
         weights->view_offs == 24 && LayerEndAddress(weights) == LayerEndAddress(route) + 24;
}
bool LearnedBroadcastWeights(const ggml_tensor* view, const ggml_tensor* weights) {
  return LearnedShape(weights, GGML_TYPE_F32, 6) && LayerEndShape(view, {1, 6, 1, 1}) &&
         view->op == GGML_OP_VIEW && view->src[0] == weights && view->view_src == weights &&
         view->view_offs == 0 && LayerEndAddress(view) == LayerEndAddress(weights) &&
         view->nb[0] == 4 && view->nb[1] == 4 && view->nb[2] == 24 && view->nb[3] == 24;
}

TEST(Dsv4LearnedRouteObservation, RequiresExactCompletedNativeWeightPlane) {
  auto arena = kg::TensorArena::Create(16);
  ASSERT_TRUE(arena);
  auto* c = arena->context();
  auto* logits = ggml_new_tensor_1d(c, GGML_TYPE_F32, 256);
  auto* bias = ggml_new_tensor_1d(c, GGML_TYPE_F32, 256);
  auto* route = kg::RouteBiasedTopK(c, logits, bias, 6, true, 1.5F);
  kg::TensorArena::Bind(logits, 0x100000);
  kg::TensorArena::Bind(bias, 0x200000);
  kg::TensorArena::Bind(route, 0x300000);
  auto weights = *ggml_new_tensor_1d(c, GGML_TYPE_F32, 6);
  weights.op = GGML_OP_VIEW;
  weights.src[0] = weights.view_src = route;
  weights.view_offs = 24;
  weights.data = reinterpret_cast<void*>(0x300018);
  ASSERT_TRUE(LearnedNativeWeights(&weights, route));
  weights.view_offs = 0;
  EXPECT_FALSE(LearnedNativeWeights(&weights, route));
  weights.view_offs = 24;
  weights.src[0] = logits;
  EXPECT_FALSE(LearnedNativeWeights(&weights, route));
  weights.src[0] = route;
  weights.data = reinterpret_cast<void*>(0x30001c);
  EXPECT_FALSE(LearnedNativeWeights(&weights, route));
  weights.data = reinterpret_cast<void*>(0x300018);
  weights.type = GGML_TYPE_I32;
  EXPECT_FALSE(LearnedNativeWeights(&weights, route));
}

TEST(Dsv4LearnedRouteObservation, RequiresExactBiasSortAndNormalizationAncestry) {
  std::array<ggml_tensor, 14> a{};
  for (std::size_t i = 0; i < a.size(); ++i) {
    auto& t = a[i];
    t.type = GGML_TYPE_F32;
    t.data = reinterpret_cast<void*>(0x100000 + i * 0x10000);
    t.ne[0] = 256;
    t.ne[1] = t.ne[2] = t.ne[3] = 1;
    t.nb[0] = 4;
    t.nb[1] = t.nb[2] = t.nb[3] = 1024;
  }
  auto& logits = a[0];
  auto& bias = a[1];
  auto& soft = a[2];
  auto& root = a[3];
  auto& add = a[4];
  auto& sort = a[5];
  auto& ids = a[6];
  auto& rows = a[7];
  auto& gather = a[8];
  auto& weights = a[9];
  auto& sum = a[10];
  auto& clamp = a[11];
  auto& div = a[12];
  auto& scale = a[13];
  soft.op = GGML_OP_UNARY;
  soft.op_params[0] = GGML_UNARY_OP_SOFTPLUS;
  soft.src[0] = &logits;
  root.op = GGML_OP_SQRT;
  root.src[0] = &soft;
  add.op = GGML_OP_ADD;
  add.src[0] = &root;
  add.src[1] = &bias;
  sort.type = GGML_TYPE_I32;
  sort.op = GGML_OP_ARGSORT;
  sort.op_params[0] = GGML_SORT_ORDER_DESC;
  sort.src[0] = &add;
  ids.type = GGML_TYPE_I32;
  ids.op = GGML_OP_VIEW;
  ids.src[0] = ids.view_src = &sort;
  ids.data = sort.data;
  ids.ne[0] = 6;
  rows.op = GGML_OP_RESHAPE;
  rows.src[0] = rows.view_src = &root;
  rows.data = root.data;
  rows.ne[0] = 1;
  rows.ne[1] = 256;
  rows.nb[1] = 4;
  gather.op = GGML_OP_GET_ROWS;
  gather.src[0] = &rows;
  gather.src[1] = &ids;
  gather.ne[0] = 1;
  gather.ne[1] = 6;
  gather.nb[1] = 4;
  gather.nb[2] = gather.nb[3] = 24;
  weights.op = GGML_OP_RESHAPE;
  weights.src[0] = weights.view_src = &gather;
  weights.data = gather.data;
  weights.ne[0] = 6;
  weights.nb[1] = weights.nb[2] = weights.nb[3] = 24;
  sum.op = GGML_OP_SUM_ROWS;
  sum.src[0] = &weights;
  sum.ne[0] = 1;
  sum.nb[1] = sum.nb[2] = sum.nb[3] = 4;
  clamp.op = GGML_OP_CLAMP;
  clamp.src[0] = &sum;
  clamp.ne[0] = 1;
  clamp.nb[1] = clamp.nb[2] = clamp.nb[3] = 4;
  clamp.op_params[0] = std::bit_cast<std::int32_t>(0x1p-14F);
  clamp.op_params[1] = std::bit_cast<std::int32_t>(std::numeric_limits<float>::infinity());
  div.op = GGML_OP_DIV;
  div.src[0] = &weights;
  div.src[1] = &clamp;
  div.ne[0] = 6;
  div.nb[1] = div.nb[2] = div.nb[3] = 24;
  scale.op = GGML_OP_SCALE;
  scale.src[0] = &div;
  scale.op_params[0] = std::bit_cast<std::int32_t>(1.5F);
  scale.ne[0] = 6;
  scale.nb[1] = scale.nb[2] = scale.nb[3] = 24;
  const LayerEndResource owned{.index = 1, .address = LayerEndAddress(&bias), .bytes = 1024};
  ASSERT_TRUE(LearnedSort(&sort, &logits, owned));
  ASSERT_TRUE(LearnedWeights(&scale, &sort, 1.5F));
  auto original = sum.src[0];
  sum.src[0] = &logits;
  EXPECT_FALSE(LearnedWeights(&scale, &sort, 1.5F));
  sum.src[0] = original;
  ids.view_offs = 4;
  EXPECT_FALSE(LearnedWeights(&scale, &sort, 1.5F));
  ids.view_offs = 0;
  scale.op_params[1] = std::bit_cast<std::int32_t>(1.0F);
  EXPECT_FALSE(LearnedWeights(&scale, &sort, 1.5F));
  scale.op_params[1] = 0;
  sort.op_params[0] = GGML_SORT_ORDER_ASC;
  EXPECT_FALSE(LearnedSort(&sort, &logits, owned));
  sort.op_params[0] = GGML_SORT_ORDER_DESC;
  soft.src[0] = &bias;
  EXPECT_FALSE(LearnedSort(&sort, &logits, owned));
  soft.src[0] = &logits;
  auto shifted = owned;
  shifted.address += 4;
  EXPECT_FALSE(LearnedSort(&sort, &logits, shifted));
  EXPECT_TRUE(LearnedSort(&sort, &logits, owned));
  EXPECT_TRUE(LearnedWeights(&scale, &sort, 1.5F));
}

TEST(Dsv4LearnedRouteObservation, OrdinaryCurrentRequiresAllOrderedSlotsAndCompletedOperands) {
  std::array<ggml_tensor, 16> a{};
  for (std::size_t i = 0; i < a.size(); ++i) {
    auto& t = a[i];
    t.type = GGML_TYPE_F32;
    t.data = reinterpret_cast<void*>(0x100000 + i * 0x20000);
    t.ne[0] = 4096;
    t.ne[1] = t.ne[2] = t.ne[3] = 1;
    t.nb[0] = 4;
    t.nb[1] = t.nb[2] = t.nb[3] = 16384;
  }
  auto& weighted = a[0];
  weighted.op = GGML_OP_MUL;
  weighted.ne[1] = 6;
  weighted.nb[2] = weighted.nb[3] = 98304;
  auto& shared = a[1];
  for (std::size_t i = 0; i < 6; ++i) {
    auto& slot = a[2 + i];
    slot.op = GGML_OP_VIEW;
    slot.src[0] = slot.view_src = &weighted;
    slot.view_offs = i * 16384;
    slot.data = reinterpret_cast<void*>(LayerEndAddress(&weighted) + slot.view_offs);
    slot.nb[1] = slot.nb[2] = slot.nb[3] = 98304;
  }
  for (std::size_t i = 0; i < 5; ++i) {
    auto& sum = a[8 + i];
    sum.op = GGML_OP_ADD;
    sum.src[0] = i == 0 ? &a[2] : &a[7 + i];
    sum.src[1] = &a[3 + i];
  }
  auto& current = a[13];
  current.op = GGML_OP_ADD;
  current.src[0] = &a[12];
  current.src[1] = &shared;
  ASSERT_TRUE(LearnedOrdinaryCurrent(&current, &weighted, &shared));
  EXPECT_FALSE(LearnedOrdinaryCurrent(&a[12], &weighted, &shared));
  EXPECT_FALSE(LearnedOrdinaryCurrent(&current, nullptr, &shared));
  EXPECT_FALSE(LearnedOrdinaryCurrent(&current, &weighted, nullptr));
  for (std::size_t i = 0; i < 6; ++i) {
    auto& slot = a[2 + i];
    const auto original = slot;
    slot.view_offs = ((i + 1) % 6) * 16384;
    slot.data = reinterpret_cast<void*>(LayerEndAddress(&weighted) + slot.view_offs);
    EXPECT_FALSE(LearnedOrdinaryCurrent(&current, &weighted, &shared)) << i;
    slot = original;
    slot.src[0] = &shared;
    EXPECT_FALSE(LearnedOrdinaryCurrent(&current, &weighted, &shared)) << i;
    slot = original;
    slot.nb[1] = 16384;
    EXPECT_FALSE(LearnedOrdinaryCurrent(&current, &weighted, &shared)) << i;
    slot = original;
  }
  current.src[0] = &a[11];
  EXPECT_FALSE(LearnedOrdinaryCurrent(&current, &weighted, &shared));
  current.src[0] = &a[12];
  current.src[1] = &a[2];
  EXPECT_FALSE(LearnedOrdinaryCurrent(&current, &weighted, &shared));
  current.src[1] = &shared;
  a[10].op = GGML_OP_MUL;
  EXPECT_FALSE(LearnedOrdinaryCurrent(&current, &weighted, &shared));
  a[10].op = GGML_OP_ADD;
  EXPECT_TRUE(LearnedOrdinaryCurrent(&current, &weighted, &shared));

  auto& weights = a[14];
  weights.ne[0] = 6;
  weights.nb[1] = weights.nb[2] = weights.nb[3] = 24;
  auto& view = a[15];
  view.op = GGML_OP_VIEW;
  view.src[0] = view.view_src = &weights;
  view.data = weights.data;
  view.ne[0] = 1;
  view.ne[1] = 6;
  view.nb[1] = 4;
  view.nb[2] = view.nb[3] = 24;
  ASSERT_TRUE(LearnedBroadcastWeights(&view, &weights));
  view.src[0] = &shared;
  EXPECT_FALSE(LearnedBroadcastWeights(&view, &weights));
  view.src[0] = &weights;
  view.view_offs = 4;
  EXPECT_FALSE(LearnedBroadcastWeights(&view, &weights));
  view.view_offs = 0;
  view.data = shared.data;
  EXPECT_FALSE(LearnedBroadcastWeights(&view, &weights));
}

// The original 53-record observer remains intact. This wrapper adds only each
// completed FFN HC-post, using distinct paid host bytes after its payload bound.
class LayerEndsProbe final : public ScreenProbe {
  struct LearnedRecord {
    std::uint64_t address = 0;
    ggml_type type = GGML_TYPE_F32;
    ggml_op op = GGML_OP_NONE;
    std::array<std::int64_t, 4> ne{};
    std::array<std::size_t, 4> nb{};
    std::size_t offset = 0;
    std::string_view implementation{};
  };
  struct LearnedState {
    std::array<LayerEndResource, 3> resources{};
    std::array<RoutedArrayReceipt, 3> routed{};
    std::array<LearnedRecord, 9> records{};
    const ggml_tensor *logits = nullptr, *sort = nullptr, *weights = nullptr, *legacy = nullptr,
                      *current = nullptr, *pending = nullptr, *shared_down = nullptr,
                      *down = nullptr, *weighted = nullptr;
    std::string_view implementation{};
    unsigned kind = 0;
  };
  LearnedState& Learned() { return learned_.at(LearnedSlot(layer_)); }

 public:
  LayerEndsProbe(
      void* host, const Target& target, std::uint64_t raw, std::array<std::uint64_t, 9> weights,
      std::array<RoutedArrayReceipt, 3> routed, std::int32_t token, bool shared,
      LayerEndResources resources,
      std::array<std::array<LayerEndResource, 3>, kLearnedLayers.size()> learned_resources,
      std::array<std::array<RoutedArrayReceipt, 3>, kLearnedLayers.size()> learned_routed,
      LayerEndResources attention_resources, std::array<LayerEndResource, 2> shared_input_resources)
      : first_(host, target, raw, weights, routed, token, shared),
        host_(static_cast<std::byte*>(host)),
        target_(target),
        shared_(shared),
        resources_(resources),
        shared_input_resources_(shared_input_resources),
        attention_resources_(attention_resources) {
    for (std::size_t i = 0; i < learned_.size(); ++i) {
      learned_[i].resources = learned_resources[i];
      learned_[i].routed = learned_routed[i];
    }
  }
  bool active() const override {
    return first_.active() || layer_ != kDecodeLayerEnds || pending_ || attention_pending_ ||
           attention_completed_;
  }
  std::expected<void, kg::KernelFailure> Observe(kg::LaunchContext& launch,
                                                 std::span<ggml_tensor* const> nodes,
                                                 std::string_view impl, bool after) override {
    if (auto observed = first_.Observe(launch, nodes, impl, after); !observed) return observed;
    if (auto observed = LearnedRoute(launch, nodes, impl, after); !observed) return observed;
    if (auto observed = AttentionOutput(launch, nodes, impl, after); !observed) return observed;
    if (layer_ == kDecodeLayerEnds) return {};
    if (pending_) {
      if (!after || nodes.size() != pending_count_ || impl != pending_impl_ ||
          !std::equal(nodes.begin(), nodes.end(), pending_nodes_.begin()) ||
          stream_.handle != launch.native_stream().handle)
        return Fail("layer-end before/after selected recipe or stream changed");
      if (phase_ == 0)
        mix_ = pending_;
      else if (phase_ == 1) {
        pre_ = pending_;
        if (IsLearnedLayer(layer_)) {
          if (auto copied = CopyLearned(0, pre_->src[1], impl); !copied) return copied;
          if (auto copied = CopyLearned(1, pre_, impl); !copied) return copied;
        }
      } else {
        const auto copied = llmp::providers::CopyAsync(
            stream_, host_ + kLayerEndsOffset + layer_ * kLayerEndBytes, pending_->data,
            kLayerEndBytes, llmp::providers::CopyKind::kDeviceToHost);
        if (!copied.ok()) return Fail(copied.text());
        auto& record = records_[layer_];
        record.address = LayerEndAddress(pending_);
        record.implementation = impl;
        attention_completed_ = nullptr;
        ++layer_;
      }
      pending_ = nullptr;
      phase_ = (phase_ + 1) % 3;
      return {};
    }
    if (after) return {};
    ggml_tensor* found = nullptr;
    for (auto* n : nodes) {
      if (!n) continue;
      bool match = false;
      if (phase_ == 0 && kg::LlmpOpOf(n) == kg::LlmpOp::kDsv4HcMix) {
        // Classify by source-owned resource, not a name, ordinal or shape.
        for (std::size_t layer = 0; layer < resources_.size(); ++layer)
          if (LayerEndAddress(n->src[1]) == resources_[layer][0].address) {
            if (layer != layer_) return Fail("FFN layer-end source order changed");
            match = true;
          }
      } else if (phase_ == 1 && kg::LlmpOpOf(n) == kg::LlmpOp::kDsv4HcPre)
        match = true;
      else if (phase_ == 2 && n->op == GGML_OP_DSV4_HC_POST)
        match = true;
      if (match) {
        if (found) return Fail("multiple layer-end boundary nodes");
        found = n;
      }
    }
    if (!found) return {};
    if (nodes.empty() || nodes.size() > pending_nodes_.size())
      return Fail("layer-end selected node bound");
    if (!stream_.handle) stream_ = launch.native_stream();
    if (stream_.handle != launch.native_stream().handle) return Fail("layer-end stream changed");
    const auto& r = resources_[layer_];
    if (phase_ == 0) {
      if (impl != kg::kDsv4HcMixName || nodes.size() != 1 || !kg::CheckDsv4HcMix(found) ||
          !LayerEndWeight(found->src[1], r[0], {16384, 24, 1, 1}) ||
          !LayerEndShape(found->src[0], {4096, 4, 1, 1}) || !ggml_is_contiguous(found->src[0]))
        return Fail("layer-end FFN mixing source/geometry differs");
      if (!AttentionConsumer(found, attention_completed_, attention_records_[layer_].address))
        return Fail("FFN does not consume the exact completed attention output");
      attention_records_[layer_].consumer = LayerEndAddress(found);
      records_[layer_].residual = LayerEndAddress(found->src[0]);
    } else if (phase_ == 1) {
      const auto& profile = md::Dsv4Flash();
      if (impl != kg::kDsv4HcPreName || nodes.size() != 1 || !kg::CheckDsv4HcPre(found) ||
          found->src[0] != mix_ || found->src[1] != mix_->src[0] ||
          !LayerEndWeight(found->src[2], r[1], {3, 1, 1, 1}) ||
          !LayerEndWeight(found->src[3], r[2], {24, 1, 1, 1}) ||
          !LayerEndWeight(found->src[4], r[3], {4096, 1, 1, 1}) ||
          kg::LlmpOpFloat(found, 0) != profile.rms_eps ||
          kg::LlmpOpFloat(found, 1) != profile.hc_eps ||
          kg::LlmpOpInt(found, 2) != static_cast<std::int32_t>(profile.sinkhorn_iterations) ||
          !LayerEndShape(found, {4128, 1, 1, 1}) || !ggml_is_contiguous(found))
        return Fail("layer-end FFN pre-normalization source/parameters differ");
      records_[layer_].pre = LayerEndAddress(found);
    } else {
      if (!LayerEndShape(found, {4096, 4, 1, 1}) || !ggml_is_contiguous(found) ||
          ggml_nbytes(found) != kLayerEndBytes || found->src[1] != pre_->src[1] ||
          !LayerEndTail(found->src[2], pre_, shared_, false) ||
          !LayerEndTail(found->src[3], pre_, shared_, true))
        return Fail("layer-end post does not consume this FFN pre's residual and exact tails");
      if (IsLearnedLayer(layer_) &&
          (!Learned().current || Learned().pending || found->src[0] != Learned().current))
        return Fail("observed layer post does not consume its completed FFN current");
      if (impl == kg::kHcPostName) {
        if (nodes.size() != 1 || !kg::CheckHcPost(found))
          return Fail("layer-end native post contract");
      } else if (impl == kg::kDsv4HcPostNormF16Name) {
        if (nodes.size() != 2 || nodes[0] != found ||
            !kg::CheckDsv4HcPostNormF16(nodes[0], nodes[1]))
          return Fail("layer-end post/norm selected output contract");
      } else if (impl == kg::kDsv4HcPostExpertsNormF16Name) {
        if (nodes.size() != 4 || nodes[2] != found ||
            !kg::CheckDsv4HcPostExpertsNormF16(nodes[0], nodes[1], nodes[2], nodes[3]))
          return Fail("layer-end experts/post/norm selected output contract");
      } else
        return Fail("unknown layer-end output producer");
    }
    pending_ = found;
    pending_impl_ = impl;
    pending_count_ = nodes.size();
    std::copy(nodes.begin(), nodes.end(), pending_nodes_.begin());
    return {};
  }
  en::Status Save(const fs::path& dir) const override {
    if (auto saved = first_.Save(dir); !saved) return saved;
    if (active() || phase_ != 0) return Error("not every source-authenticated layer end completed");
    if (auto saved = SaveLearned(dir); !saved) return saved;
    if (auto saved = SaveAttention(dir); !saved) return saved;
    std::ofstream meta(dir / "decode-layer-ends.txt");
    meta << "layers=" << layer_ << " rows=1 past=" << kPrefix
         << " artifact=" << target_.artifact.id()
         << " bound_spec=" << llmp::base::ToHex(target_.bound)
         << " bytes=" << layer_ * kLayerEndBytes << '\n';
    for (std::size_t i = 0; i < layer_; ++i) {
      const auto bytes =
          std::span<const std::byte>(host_ + kLayerEndsOffset + i * kLayerEndBytes, kLayerEndBytes);
      const auto hash = llmp::base::ToHex(llmp::base::Sha256().Update(bytes).Finish());
      for (std::size_t w = 0; w < 4; ++w) {
        const auto& r = resources_[i][w];
        const auto& source = target_.artifact.resources()[r.index];
        meta << "layer=" << i << " role=" << w << " resource=" << r.index << " name=" << source.name
             << " group=" << source.group << " offset=" << source.offset.value()
             << " bytes=" << r.bytes << " address=" << r.address << '\n';
      }
      meta << "layer=" << i << " implementation=" << records_[i].implementation
           << " residual=" << records_[i].residual << " pre=" << records_[i].pre
           << " output=" << records_[i].address << " type=f32 ne=4096,4,1,1 nb=4,16384,65536,65536"
           << " bytes=" << bytes.size() << " sha256=" << hash << '\n';
      std::ofstream file(dir / ("decode-layer-end-" + std::to_string(i) + ".bin"),
                         std::ios::binary);
      file.write(reinterpret_cast<const char*>(bytes.data()),
                 static_cast<std::streamsize>(bytes.size()));
      if (!file) return Error("layer-end payload output failed");
    }
    return meta ? en::Status{} : Error("layer-end metadata output failed");
  }

 private:
  static std::unexpected<kg::KernelFailure> Fail(std::string text) {
    return std::unexpected(
        kg::KernelFailure{.error = kg::KernelError::kUnknown, .detail = std::move(text)});
  }
  std::expected<void, kg::KernelFailure> AttentionOutput(kg::LaunchContext& launch,
                                                         std::span<ggml_tensor* const> nodes,
                                                         std::string_view impl, bool after) {
    if (attention_pending_) {
      if (!after || nodes.size() != attention_count_ || impl != attention_impl_ ||
          !std::equal(nodes.begin(), nodes.end(), attention_nodes_.begin()) ||
          stream_.handle != launch.native_stream().handle)
        return Fail("attention output selected before/after recipe or stream changed");
      if (LayerEndAddress(attention_pending_) != attention_records_[layer_].address)
        return Fail("attention output address changed during production");
      const auto copied = llmp::providers::CopyAsync(
          stream_, host_ + kAttentionOffset + layer_ * kLayerEndBytes, attention_pending_->data,
          kLayerEndBytes, llmp::providers::CopyKind::kDeviceToHost);
      if (!copied.ok()) return Fail(copied.text());
      attention_completed_ = attention_pending_;
      attention_pending_ = nullptr;
      return {};
    }
    if (after || layer_ == kDecodeLayerEnds || phase_ != 0) return {};
    ggml_tensor* post = nullptr;
    for (auto* n : nodes) {
      if (!n || n->op != GGML_OP_DSV4_HC_POST) continue;
      if (post) return Fail("multiple attention output producers");
      post = n;
    }
    if (!post) return {};
    if (attention_completed_ || attention_records_[layer_].address || nodes.empty() ||
        nodes.size() > attention_nodes_.size())
      return Fail("duplicate attention output or selected node bound");
    const auto* pre = AttentionPostSource(post, shared_, attention_resources_[layer_]);
    if (!pre) return Fail("attention output owned source/pre/residual/tail authority differs");
    if (impl == kg::kHcPostName) {
      if (nodes.size() != 1 || !kg::CheckHcPost(post))
        return Fail("attention native post contract");
    } else if (impl == kg::kDsv4HcPostNormF16Name) {
      if (nodes.size() != 2 || nodes[0] != post || !kg::CheckDsv4HcPostNormF16(nodes[0], nodes[1]))
        return Fail("attention post/norm output contract");
    } else {
      return Fail("unknown actual attention output recipe");
    }
    if (!stream_.handle) stream_ = launch.native_stream();
    if (stream_.handle != launch.native_stream().handle)
      return Fail("attention output stream changed");
    attention_records_[layer_] = {.residual = LayerEndAddress(post->src[1]),
                                  .pre = LayerEndAddress(pre),
                                  .address = LayerEndAddress(post),
                                  .implementation = impl};
    attention_pending_ = post;
    attention_impl_ = impl;
    attention_count_ = nodes.size();
    std::copy(nodes.begin(), nodes.end(), attention_nodes_.begin());
    return {};
  }
  en::Status SaveAttention(const fs::path& dir) const {
    std::ofstream meta(dir / "decode-attention-ends.txt");
    meta << "layers=" << layer_ << " rows=1 past=" << kPrefix
         << " artifact=" << target_.artifact.id()
         << " bound_spec=" << llmp::base::ToHex(target_.bound)
         << " bytes=" << layer_ * kLayerEndBytes << '\n';
    for (std::size_t i = 0; i < layer_; ++i) {
      const auto& record = attention_records_[i];
      if (!record.address || !record.consumer || record.address != records_[i].residual)
        return Error("attention output lacks completed FFN consumer linkage");
      for (std::size_t w = 0; w < 4; ++w) {
        const auto& r = attention_resources_[i][w];
        const auto& source = target_.artifact.resources()[r.index];
        meta << "layer=" << i << " role=" << w << " resource=" << r.index << " name=" << source.name
             << " group=" << source.group << " offset=" << source.offset.value()
             << " bytes=" << r.bytes << " address=" << r.address << '\n';
      }
      const auto bytes =
          std::span<const std::byte>(host_ + kAttentionOffset + i * kLayerEndBytes, kLayerEndBytes);
      meta << "layer=" << i << " implementation=" << record.implementation
           << " residual=" << record.residual << " pre=" << record.pre
           << " output=" << record.address << " consumer=" << record.consumer
           << " type=f32 ne=4096,4,1,1 nb=4,16384,65536,65536 bytes=" << bytes.size()
           << " sha256=" << llmp::base::ToHex(llmp::base::Sha256().Update(bytes).Finish()) << '\n';
      std::ofstream file(dir / ("decode-attention-end-" + std::to_string(i) + ".bin"),
                         std::ios::binary);
      file.write(reinterpret_cast<const char*>(bytes.data()),
                 static_cast<std::streamsize>(bytes.size()));
      if (!file) return Error("attention output payload write failed");
    }
    return meta ? en::Status{} : Error("attention output metadata write failed");
  }
  std::expected<void, kg::KernelFailure> CopyLearned(std::size_t index, const ggml_tensor* t,
                                                     std::string_view impl,
                                                     std::size_t offset = 0) {
    if (!IsLearnedLayer(layer_)) return Fail("copy outside authenticated observed layer");
    auto& observation = Learned();
    if (!t || !t->data || index >= observation.records.size() ||
        observation.records[index].address || offset > ggml_nbytes(t) ||
        kLearnedSizes[index] > ggml_nbytes(t) - offset)
      return Fail("learned route bounded payload already copied or outside actual tensor");
    std::size_t at = kLearnedOffsets.at(LearnedSlot(layer_));
    for (std::size_t i = 0; i < index; ++i) at += kLearnedSizes[i];
    auto copied = llmp::providers::CopyAsync(
        stream_, host_ + at, static_cast<const std::byte*>(t->data) + offset, kLearnedSizes[index],
        llmp::providers::CopyKind::kDeviceToHost);
    if (!copied.ok()) return Fail(copied.text());
    auto& r = observation.records[index];
    r.address = LayerEndAddress(t);
    r.type = t->type;
    r.op = t->op;
    r.offset = offset;
    r.implementation = impl;
    std::copy_n(t->ne, 4, r.ne.begin());
    std::copy_n(t->nb, 4, r.nb.begin());
    return {};
  }
  std::expected<void, kg::KernelFailure> LearnedRoute(kg::LaunchContext& launch,
                                                      std::span<ggml_tensor* const> nodes,
                                                      std::string_view impl, bool after) {
    if (!IsLearnedLayer(layer_) || phase_ != 2) return {};
    auto& observation = Learned();
    if (stream_.handle != launch.native_stream().handle)
      return Fail("learned route stream changed");
    if (observation.pending) {
      if (!after || nodes.size() != 1 || nodes[0] != observation.pending ||
          impl != observation.implementation)
        return Fail("learned route before/after actual producer changed");
      const auto* n = observation.pending;
      if (observation.kind == 0) {
        observation.logits = n;
        if (auto r = CopyLearned(2, n, impl); !r) return r;
      } else if (observation.kind == 1) {
        observation.sort = n;
        if (auto r = CopyLearned(4, n, impl); !r) return r;
      } else if (observation.kind == 2) {
        observation.weights = n;
        if (auto r = CopyLearned(5, n, impl); !r) return r;
      } else if (observation.kind == 3) {
        if (shared_)
          observation.sort = n;
        else
          observation.legacy = n;
        if (auto r = CopyLearned(4, n, impl); !r) return r;
        if (auto r = CopyLearned(5, n, impl, 24); !r) return r;
      } else if (observation.kind == 8) {
        const auto copied =
            llmp::providers::CopyAsync(stream_, host_ + kSharedInputOffset, n->data,
                                       kSharedInputBytes, llmp::providers::CopyKind::kDeviceToHost);
        if (!copied.ok()) return Fail(copied.text());
        shared_input_producer_ = n;
        shared_input_record_ = {.address = LayerEndAddress(n),
                                .type = n->type,
                                .op = n->op,
                                .offset = 0,
                                .implementation = impl};
        std::copy_n(n->ne, 4, shared_input_record_.ne.begin());
        std::copy_n(n->nb, 4, shared_input_record_.nb.begin());
        shared_input_pre_ = LayerEndAddress(pre_);
      } else if (observation.kind == 5) {
        observation.shared_down = n;
        if (auto r = CopyLearned(8, n, impl); !r) return r;
      } else if (observation.kind == 6) {
        observation.down = n;
        if (auto r = CopyLearned(7, n, impl); !r) return r;
      } else if (observation.kind == 7) {
        observation.weighted = n;
      } else {
        observation.current = n;
        if (auto r = CopyLearned(6, n, impl); !r) return r;
      }
      observation.pending = nullptr;
      return {};
    }
    if (after || nodes.size() != 1 || !nodes[0]) return {};
    auto* n = nodes[0];
    // Bind the actual consumed typed weight plane only after its native producer
    // completed. Capture itself happened in that producer's AFTER callback.
    if (shared_ && observation.sort && kg::LlmpOpOf(observation.sort) == kg::LlmpOp::kDsv4Route &&
        ((kg::LlmpOpOf(n) == kg::LlmpOp::kDsv4Combine) ||
         (n->op == GGML_OP_MUL && n->src[0] == observation.down))) {
      const auto* weights = kg::LlmpOpOf(n) == kg::LlmpOp::kDsv4Combine
                                ? n->src[1]
                                : (n->src[1] ? n->src[1]->src[0] : nullptr);
      if (!LearnedNativeWeights(weights, observation.sort) ||
          (observation.weights && observation.weights != weights))
        return Fail("learned native weights lost completed planar producer");
      observation.weights = weights;
    }
    if (layer_ == kThirdLearnedLayer &&
        LearnedSharedInput(n, pre_, shared_, shared_input_resources_)) {
      if (shared_input_producer_ || impl != (shared_ ? "ggml.swiglu_clamp" : "llmp.vecq"))
        return Fail("layer26 shared input duplicate or actual producer implementation changed");
      observation.kind = 8;
    } else if (n->op == GGML_OP_MUL_MAT &&
               LayerEndAddress(n->src[0]) == observation.resources[0].address) {
      if (observation.logits ||
          impl != (shared_ ? "ggml.mul_mat.mmvf" : "llmp.mul_mat.mmvf_rows") ||
          !LearnedWeight(n->src[0], observation.resources[0], GGML_TYPE_BF16, {4096, 256, 1, 1}) ||
          !LearnedShape(n, GGML_TYPE_F32, 256) || n->op_params[0] != GGML_PREC_F32 ||
          !LearnedShape(n->src[1], GGML_TYPE_F32, 4096) || n->src[1]->op != GGML_OP_VIEW ||
          n->src[1]->src[0] != pre_ || n->src[1]->view_src != pre_ || n->src[1]->view_offs != 0 ||
          LayerEndAddress(n->src[1]) != LayerEndAddress(pre_))
        return Fail("learned router lost owned BF16 source/completed normalized input");
      observation.kind = 0;
    } else if (shared_ && observation.logits && n->op == GGML_OP_ARGSORT) {
      if (observation.sort || impl != "ggml.argsort.bitonic" ||
          !LearnedSort(n, observation.logits, observation.resources[1]))
        return Fail("learned selected IDs lost exact biased sqrt-softplus producer");
      if (auto r = CopyLearned(3, n->src[0]->src[1], impl); !r) return r;
      observation.kind = 1;
    } else if (shared_ && observation.sort && n->op == GGML_OP_SCALE && n->ne[0] == 6) {
      if (observation.weights || impl != "ggml.scale" ||
          !LearnedWeights(n, observation.sort, md::Dsv4Flash().expert_weights_scale))
        return Fail("learned weights lost exact observed sort/gather/normalization order");
      observation.kind = 2;
    } else if (observation.logits && kg::LlmpOpOf(n) == kg::LlmpOp::kDsv4Route) {
      if (observation.legacy || observation.sort || observation.weights ||
          impl != kg::kDsv4RouteName || !kg::CheckDsv4Route(n) || n->src[0] != observation.logits ||
          n->src[2] ||
          !LearnedWeight(n->src[1], observation.resources[1], GGML_TYPE_F32, {256, 1, 1, 1}) ||
          !LearnedShape(n, GGML_TYPE_I32, 12) || kg::LlmpOpInt(n, 0) != 6 ||
          kg::LlmpOpInt(n, 1) != 1 || kg::LlmpOpFloat(n, 2) != 0x1p-14F ||
          kg::LlmpOpFloat(n, 3) != md::Dsv4Flash().expert_weights_scale ||
          kg::LlmpOpInt(n, 4) != (shared_ ? 2 : 0))
        return Fail("learned route lost actual router/bias/native addressing mode");
      if (auto r = CopyLearned(3, n->src[1], impl); !r) return r;
      observation.kind = 3;
    } else if ((n->op == GGML_OP_MUL_MAT || kg::LlmpOpOf(n) == kg::LlmpOp::kVecQ) &&
               LayerEndAddress(n->src[0]) == observation.resources[2].address) {
      if (observation.shared_down ||
          (shared_ ? (impl != "ggml.mul_mat.mmvq" || n->op != GGML_OP_MUL_MAT)
                   : (impl != "llmp.vecq" || !kg::CheckVecQ(n) || kg::LlmpOpInt(n, 0) != 1 ||
                      kg::LlmpOpInt(n, 1) != 0 || kg::LlmpOpInt(n, 2) != 0 || !n->src[1] ||
                      !kg::CheckQuantizeQ8(n->src[1]))) ||
          !LearnedWeight(n->src[0], observation.resources[2],
                         ObservedSharedDownEncoding(static_cast<std::uint32_t>(layer_)),
                         {2048, 4096, 1, 1}) ||
          !LearnedShape(n, GGML_TYPE_F32, 4096))
        return Fail("learned shared down lost owned source/actual completed product");
      if (layer_ == kThirdLearnedLayer) {
        if (!shared_ && kg::LlmpOpOf(n->src[1]) != kg::LlmpOp::kQuantizeQ8)
          return Fail("layer26 legacy shared down lost real Q8 input preparation");
        const auto* consumed = shared_ ? n->src[1] : n->src[1]->src[0];
        if (!shared_input_producer_ || shared_input_consumer_ || consumed != shared_input_producer_)
          return Fail("layer26 shared down does not consume completed captured GLU");
        shared_input_consumer_ = LayerEndAddress(n);
      }
      observation.kind = 5;
    } else if ((shared_ ? observation.sort != nullptr : observation.legacy != nullptr) &&
               (n->op == GGML_OP_MUL_MAT_ID || kg::LlmpOpOf(n) == kg::LlmpOp::kVecQ) &&
               MatchesRoutedArray(n->src[0], observation.routed[2])) {
      if (observation.down ||
          (shared_ ? (impl != "ggml.mul_mat_id.mmvq" || n->op != GGML_OP_MUL_MAT_ID)
                   : (impl != "llmp.vecq" || !kg::CheckVecQ(n) || kg::LlmpOpInt(n, 0) != 1 ||
                      kg::LlmpOpInt(n, 1) != 1 || kg::LlmpOpInt(n, 2) != 0 || !n->src[1] ||
                      !kg::CheckQuantizeQ8(n->src[1]))) ||
          !LearnedIds(n->src[2], shared_ ? observation.sort : observation.legacy) ||
          !LayerEndShape(n, {4096, 6, 1, 1}) || !ggml_is_contiguous(n))
        return Fail("learned routed down lost owned array/actual selected IDs");
      observation.kind = 6;
    } else if (shared_ && observation.down && n->op == GGML_OP_MUL &&
               n->src[0] == observation.down) {
      if (observation.weighted || impl != "ggml.mul" ||
          !LearnedBroadcastWeights(n->src[1], observation.weights) ||
          !LayerEndShape(n, {4096, 6, 1, 1}) || !ggml_is_contiguous(n))
        return Fail("learned weighted experts lost completed down/route weights");
      observation.kind = 7;
    } else if (shared_ && observation.shared_down && n->op == GGML_OP_ADD &&
               n->src[1] == observation.shared_down) {
      if (observation.current || impl != "ggml.add" ||
          !LearnedOrdinaryCurrent(n, observation.weighted, observation.shared_down))
        return Fail("learned FFN current lost complete ordered six-slot plus shared sum");
      observation.kind = 4;
    } else if (kg::LlmpOpOf(n) == kg::LlmpOp::kDsv4Combine &&
               (shared_ ? observation.weights != nullptr : observation.legacy != nullptr)) {
      if (observation.current || impl != kg::kDsv4CombineName || !kg::CheckDsv4Combine(n) ||
          n->src[1] != (shared_ ? observation.weights : observation.legacy) || !observation.down ||
          !observation.shared_down || n->src[0] != observation.down ||
          n->src[2] != observation.shared_down ||
          !MatchesRoutedArray(n->src[0]->src[0], observation.routed[2]) ||
          !LearnedIds(n->src[0]->src[2], shared_ ? observation.sort : observation.legacy) ||
          (shared_ ? n->src[0]->op != GGML_OP_MUL_MAT_ID
                   : (kg::LlmpOpOf(n->src[0]) != kg::LlmpOp::kVecQ || !kg::CheckVecQ(n->src[0]))) ||
          kg::LlmpOpInt(n, 0) != (shared_ ? 1 : 0) || !LearnedShape(n, GGML_TYPE_F32, 4096))
        return Fail("learned FFN current lost completed routing weights/selected output");
      observation.kind = 4;
    } else
      return {};
    observation.pending = n;
    observation.implementation = impl;
    return {};
  }
  en::Status SaveLearned(const fs::path& dir) const {
    for (std::size_t i = 0; i < learned_.size(); ++i)
      if (auto saved = SaveLearnedOne(dir, i); !saved) return saved;
    return SaveSharedInput(dir);
  }
  en::Status SaveLearnedOne(const fs::path& dir, std::size_t index) const {
    const auto& observation = learned_[index];
    const auto layer = kLearnedLayers[index];
    const auto prefix = "decode-layer" + std::to_string(layer) + "-route";
    if (observation.pending || !observation.logits || !observation.current || !observation.down ||
        !observation.shared_down ||
        (shared_ ? (!observation.sort || !observation.weights) : !observation.legacy))
      return Error("learned route observation incomplete");
    std::ofstream meta(dir / (prefix + ".txt"));
    meta << "layer=" << layer << " rows=1 past=" << kPrefix << " artifact=" << target_.artifact.id()
         << " bound_spec=" << llmp::base::ToHex(target_.bound) << " bytes=" << kLearnedBytes
         << '\n';
    for (std::size_t role = 0; role < observation.resources.size(); ++role) {
      const auto& r = observation.resources[role];
      const auto& source = target_.artifact.resources()[r.index];
      meta << "role=" << role << " resource=" << r.index << " name=" << source.name
           << " group=" << source.group << " offset=" << source.offset.value()
           << " bytes=" << r.bytes << " address=" << r.address << '\n';
    }
    for (std::size_t i = 0; i < observation.routed.size(); ++i) {
      const auto& r = observation.routed[i];
      meta << "routed=" << i << " address=" << r.address << " stride=" << r.stride
           << " slice=" << r.slice << " type=" << static_cast<int>(r.type) << '\n';
    }
    std::size_t at = kLearnedOffsets.at(index);
    for (std::size_t i = 0; i < observation.records.size(); ++i) {
      const auto& r = observation.records[i];
      if (!r.address) return Error("learned payload record absent");
      const auto bytes = std::span<const std::byte>(host_ + at, kLearnedSizes[i]);
      at += bytes.size();
      meta << "name=" << kLearnedNames[i] << " logical=" << (i == 4 ? "i32" : "f32")
           << " implementation=" << r.implementation << " stored_type=" << static_cast<int>(r.type)
           << " stored_op=" << static_cast<int>(r.op) << " address=" << r.address
           << " offset=" << r.offset << " bytes=" << bytes.size() << " ne=" << r.ne[0] << ','
           << r.ne[1] << ',' << r.ne[2] << ',' << r.ne[3] << " nb=" << r.nb[0] << ',' << r.nb[1]
           << ',' << r.nb[2] << ',' << r.nb[3]
           << " sha256=" << llmp::base::ToHex(llmp::base::Sha256().Update(bytes).Finish()) << '\n';
      std::ofstream file(dir / (prefix + "-" + std::string(kLearnedNames[i]) + ".bin"),
                         std::ios::binary);
      file.write(reinterpret_cast<const char*>(bytes.data()),
                 static_cast<std::streamsize>(bytes.size()));
      if (!file) return Error("learned payload write failed");
    }
    if (!meta || meta.tellp() < 0 ||
        meta.tellp() > static_cast<std::streamoff>(kLearnedMetadataBytes))
      return Error("learned metadata exceeds unchanged paid bound or write failed");
    return {};
  }

  en::Status SaveSharedInput(const fs::path& dir) const {
    if (!shared_input_producer_ || !shared_input_consumer_ || !shared_input_record_.address ||
        shared_input_pre_ != records_[kThirdLearnedLayer].pre)
      return Error("layer26 shared-input source/completion/consumer absent");
    const auto bytes = std::span<const std::byte>(host_ + kSharedInputOffset, kSharedInputBytes);
    std::ofstream meta(dir / "decode-layer26-shared-input.txt");
    meta << "layer=26 rows=1 past=" << kPrefix << " artifact=" << target_.artifact.id()
         << " bound_spec=" << llmp::base::ToHex(target_.bound) << " bytes=" << bytes.size() << '\n';
    for (std::size_t i = 0; i < shared_input_resources_.size(); ++i) {
      const auto& r = shared_input_resources_[i];
      const auto& source = target_.artifact.resources()[r.index];
      meta << "role=" << i << " resource=" << r.index << " name=" << source.name
           << " group=" << source.group << " offset=" << source.offset.value()
           << " bytes=" << r.bytes << " address=" << r.address << '\n';
    }
    const auto& r = shared_input_record_;
    meta << "name=shared-input logical=f32 implementation=" << r.implementation
         << " stored_type=" << static_cast<int>(r.type) << " stored_op=" << static_cast<int>(r.op)
         << " address=" << r.address << " pre=" << shared_input_pre_
         << " consumer=" << shared_input_consumer_ << " bytes=" << bytes.size() << " ne=" << r.ne[0]
         << ',' << r.ne[1] << ',' << r.ne[2] << ',' << r.ne[3] << " nb=" << r.nb[0] << ','
         << r.nb[1] << ',' << r.nb[2] << ',' << r.nb[3]
         << " sha256=" << llmp::base::ToHex(llmp::base::Sha256().Update(bytes).Finish()) << '\n';
    if (!meta || meta.tellp() < 0 ||
        meta.tellp() > static_cast<std::streamoff>(kSharedInputMetadataBytes))
      return Error("shared-input metadata exceeds paid bound or write failed");
    std::ofstream file(dir / "decode-layer26-shared-input.bin", std::ios::binary);
    file.write(reinterpret_cast<const char*>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
    return file ? en::Status{} : Error("shared-input payload write failed");
  }

  struct Record {
    std::uint64_t residual = 0, pre = 0, address = 0;
    std::string_view implementation{};
  };
  FirstDecodeProbe first_;
  std::byte* host_;
  const Target& target_;
  bool shared_;
  LayerEndResources resources_;
  std::array<Record, kDecodeLayerEnds> records_{};
  std::size_t layer_ = 0, phase_ = 0, pending_count_ = 0;
  ggml_tensor* pending_ = nullptr;
  const ggml_tensor *mix_ = nullptr, *pre_ = nullptr;
  std::array<ggml_tensor*, 4> pending_nodes_{};
  std::string_view pending_impl_;
  llmp::providers::NativeStream stream_{};
  std::array<LearnedState, kLearnedLayers.size()> learned_{};
  std::array<LayerEndResource, 2> shared_input_resources_{};
  LearnedRecord shared_input_record_{};
  const ggml_tensor* shared_input_producer_ = nullptr;
  std::uint64_t shared_input_pre_ = 0, shared_input_consumer_ = 0;
  struct AttentionRecord {
    std::uint64_t residual = 0, pre = 0, address = 0, consumer = 0;
    std::string_view implementation{};
  };
  LayerEndResources attention_resources_;
  std::array<AttentionRecord, kDecodeLayerEnds> attention_records_{};
  ggml_tensor* attention_pending_ = nullptr;
  const ggml_tensor* attention_completed_ = nullptr;
  std::size_t attention_count_ = 0;
  std::array<ggml_tensor*, 4> attention_nodes_{};
  std::string_view attention_impl_{};
};
static_assert(sizeof(LayerEndsProbe) < (64U << 10U));

// The whole owner survives any uncertain teardown, including callbacks' maps,
// the artifact/spec/layout, node, catalog payment and kernel resources.
struct Arm {
  Arm() : node({.slot_bytes = en::kSlabSlotBytes}) {}
  std::shared_ptr<Target> target;
  en::PagedNode node;
  std::unique_ptr<KernelSource> kernel;
  std::unique_ptr<en::ModelRunner> shared;
  // Dsv4Runner borrows these settings through every submitted chunk.
  en::Dsv4Options legacy_options;
  std::unique_ptr<en::Dsv4Runner> legacy;
  std::map<std::pair<llmp::ir::WeightRef, llmp::format::PartRole>, Place> places;
  std::map<llmp::ir::WeightRef, std::uint64_t> pitch;
  ds::Source* source = nullptr;
  en::PagedWeights* weights = nullptr;
  std::vector<en::PagedModel*> entered;
  std::vector<std::int32_t> history;
  void* readback = nullptr;
  std::unique_ptr<ScreenProbe> probe;
  bool csa_inventory = false;
  bool probe_installed = false;
  bool charged = false, dispatched = false;
  std::uint64_t replayed = 0;
};
en::Status AdmitNode(Arm& arm, bool layer_boundary, bool csa) {
  if (auto s = arm.node.Start(llmp::base::Bytes(std::uint64_t{110} << 30)); !s) return s;
  const auto before = arm.node.catalog().OccupancyOf(arm.node.domain());
  const auto compiler_charge = kCompiler + (layer_boundary ? kLayerTwoFiles : 0);
  if (!arm.node.TryChargeHost(compiler_charge)) return Error("screen compiler accounting refused");
  arm.charged = true;
  const auto after = arm.node.catalog().OccupancyOf(arm.node.domain());
  const auto paid = (compiler_charge + en::kPagedExtent - 1) / en::kPagedExtent * en::kPagedExtent;
  const auto runtime = static_cast<std::size_t>(llmp::catalog::MemoryClass::kRuntime);
  if (after.Total().value() - before.Total().value() != paid ||
      after.pinned.value() - before.pinned.value() != paid ||
      after.by_class[runtime].value() - before.by_class[runtime].value() != paid)
    return Error("compiler allowance has no physical catalog occupancy");
  std::vector<llmp::catalog::ExtentId> staging;
  auto host =
      arm.node.Pinned(layer_boundary ? kLayerTwoWindow : (csa ? kCsaWindow : kWindow), 0, staging);
  if (!host) return Error(host.error());
  arm.readback = *host;
  return {};
}

en::Status StartLegacy(Arm& arm, const fs::path& dir, bool graphs, bool reduced_outa_hca,
                       bool layer_boundary, bool csa) {
  auto& options = arm.legacy_options;
  options.artifact = arm.target->root;
  options.out = dir;
  options.context = kContext;
  options.max_rows = kRows;
  options.graphs = graphs;
  options.frontier_head = true;
  en::SetDsv4ServedPrefill(options);
  if (reduced_outa_hca) {
    // Diagnostic attribution only: the shared screen has not transferred
    // the served output-A/HCA combination yet. Other serving flags stay set.
    options.prefill_outa_hca = false;
    options.prefill_outa_hca_partial = false;
  }
  arm.legacy = std::make_unique<en::Dsv4Runner>(arm.node, options, 0, 0);
  arm.entered.push_back(arm.legacy.get());
  if (auto s = arm.legacy->Setup(); !s) return s;
  if (auto s = arm.node.MapWorkspace(
          arm.legacy->activations_needed(),
          std::max<std::uint64_t>(arm.legacy->pool_needed(), en::kPagedExtent));
      !s)
    return s;
  if (auto s = AdmitNode(arm, layer_boundary, csa); !s) return s;
  if (auto s = arm.legacy->Register(); !s) return s;
  if (auto s = arm.legacy->Bind(); !s) return s;
  arm.node.Run();
  std::vector<en::LoadStats> loads;
  return arm.node.Load(arm.legacy->weights(), "legacy weights", loads);
}
en::Status StartShared(Arm& arm, const fs::path& dir, bool graphs,
                       const std::vector<pl::StepShape>& script, bool layer_boundary, bool csa) {
  const auto& target = *arm.target;
  std::vector<en::GroupPlace> places(target.artifact.groups().size(), en::GroupPlace::kNone);
  for (std::size_t i = 0; i < places.size(); ++i)
    if (target.artifact.groups()[i].kind != ar::GroupKind::kExpert)
      places[i] = en::GroupPlace::kDevice;
  std::vector<en::SlabSpec> slabs;
  std::map<std::uint32_t, std::uint64_t> pitches;
  for (std::uint32_t i = 0; i < target.binding.layers.size(); ++i) {
    const auto& l = target.binding.layers[i];
    const std::array<std::pair<std::uint32_t, std::string_view>, 3> arrays = {
        {{l.gate_exps.index, l.gate_exps.type},
         {l.up_exps.index, l.up_exps.type},
         {l.down_exps.index, l.down_exps.type}}};
    auto slab = en::ExpertSlab(target.artifact, arrays, md::Dsv4Flash().experts, 256, i);
    if (!slab) return Error(slab.error());
    for (const auto& [id, type] : arrays) pitches[id] = slab->stride;
    slabs.push_back(*slab);
  }
  // One actual graph suffices for the weight binding inventory; all shapes
  // have the same spec/component weight leaves. The source validates the
  // full immutable shape list when each program is built.
  auto step = co::BuildStep(target.spec, 0, script.front(),
                            {.tensors = target.source.tensors, .state = &target.layout});
  if (!step) return Error(step.error());
  for (const auto& w : step->weights)
    for (const auto& [name, part] : w.parts) {
      const auto found = w.ref.expert_array ? target.artifact.FindExpertArray(name)
                                            : target.artifact.FindResource(name);
      if (!found) return Error("a component weight is absent from its real artifact");
      arm.places[{w.ref, part}] = {.array = w.ref.expert_array, .index = *found};
      if (w.ref.expert_array) arm.pitch[w.ref] = pitches.at(*found);
    }
  arm.kernel = std::make_unique<KernelSource>(arm.node);
  if (auto s = arm.kernel->resources.OpenCublas("shared screen cuBLAS workspace"); !s) return s;
  auto backend = kg::GgmlBackend::Create(
      0, arm.node.execution(), arm.node.stream(0), &arm.kernel->resources.cublas(),
      {.lower = {.expert_stride = [&arm](const llmp::ir::WeightRef& r) {
         const auto at = arm.pitch.find(r);
         return at == arm.pitch.end() ? 0 : at->second;
       }}});
  if (!backend) return Error(backend.error());
  arm.kernel->backend = std::move(*backend);
  pl::Policy policy;
  policy.enabled = {"biased_route",  "moe_combine",         "hash_route",  "grouped_vecq",
                    "dsv4_compress", "dsv4_hc_post",        "dsv4_hc_pre", "dsv4_lid_mask",
                    "dsv4_qhead",    "state_bank_attention"};
  pl::Canonicalize(policy);
  auto backend_contract = arm.kernel->backend->Backend();
  backend_contract.lower = [native = backend_contract.lower, registry = backend_contract.registry,
                            &arm, dir](const kn::LowerInput& input)
      -> std::expected<std::unique_ptr<kn::StepProgram>, std::string> {
    if (auto paid = NativeReservation(input); !paid) return Error(paid.error());
    if (arm.csa_inventory && !input.addresses.measuring()) {
      std::array<std::uint32_t, 320> layers;
      layers.fill(~std::uint32_t{0});
      for (const auto& w : input.token->step.weights) {
        if (w.ref.component >= layers.size()) return Error("CSA inventory component bound");
        auto& layer = layers[w.ref.component];
        if (layer != ~std::uint32_t{0} && layer != w.ref.layer)
          return Error("CSA inventory component has inconsistent layer bindings");
        layer = w.ref.layer;
      }
      std::ofstream out(dir / "csa-bound-plan.txt");
      std::size_t count = 0;
      for (const auto& selected : input.plan.steps)
        for (const auto id : selected.nodes) {
          const auto& n = input.plan.graph.node(id);
          const auto layer = n.component < layers.size() ? layers[n.component] : ~std::uint32_t{0};
          const auto key = llmp::ir::OpKeyName(n.key);
          if (layer != 2 && key.find("compress") == std::string::npos &&
              n.key.op != llmp::ir::Op::kPool)
            continue;
          if (++count > 1024 || key.size() > 128) return Error("CSA inventory node/name bound");
          out << "node=" << id.v << " component=" << n.component << " layer=" << layer
              << " op=" << key << " implementation=" << registry->at(selected.implementation).name;
          out << " src";
          for (const auto t : n.sources()) out << ' ' << t.v;
          out << " out";
          for (const auto t : n.outputs()) out << ' ' << t.v;
          out << '\n';
        }
      count = 0;
      for (const auto& w : input.token->step.weights) {
        if (w.ref.layer != 2) continue;
        for (const auto& [name, part] : w.parts) {
          if (++count > 256 || name.size() > 256) return Error("CSA inventory weight/name bound");
          const auto at = arm.places.find({w.ref, part});
          if (at == arm.places.end()) return Error("CSA inventory missing weight binding");
          out << "weight_ref=" << w.ref.stage << ',' << w.ref.component << ',' << w.ref.role << ','
              << w.ref.layer << ',' << w.ref.expert_array << " part=" << static_cast<int>(part)
              << " resource=" << at->second.index << " name=" << name
              << " address=" << input.addresses.weight(w.ref, part) << '\n';
        }
      }
      if (!out) return Error("CSA bound-plan inventory write failed");
    }
    return native(input);
  };
  auto source =
      ds::Source::Create({.spec = &target.spec,
                          .stage = 0,
                          .bindings = {.tensors = target.source.tensors, .state = &target.layout},
                          .bound_digest = target.bound,
                          .mapping = &sp::mappings::Dsv4Mapping(),
                          .values = target.values,
                          .slots = 1,
                          .policy = policy,
                          .backend = std::move(backend_contract)},
                         script);
  if (!source) return Error(source.error());
  arm.source = source->get();
  auto weights = std::make_unique<en::PagedWeights>();
  if (auto s = weights->Open(target.root); !s) return s;
  arm.weights = weights.get();
  en::RunnerConfig config;
  config.slots = {.model = "private-dsv4-screen",
                  .slots = 1,
                  .stages = {{.name = "target", .layout = target.layout}},
                  .spec = target.bound,
                  .read_align = 256,
                  .spill_place = {},
                  .spill_directory = dir,
                  .invalidate = {},
                  .drop_plans = {}};
  config.weights.push_back({.weights = std::move(weights), .places = places, .slabs = slabs});
  config.setup_shapes = script;
  config.graphs = graphs;
  config.max_verify_rows = 0;
  config.launch = [&arm]() {
    return arm.kernel->backend->Launch(
        {.base = arm.node.pool().base, .size = llmp::base::Bytes(arm.node.pool().bytes)});
  };
  config.weight_address = [&arm](const llmp::ir::WeightRef& r,
                                 llmp::format::PartRole role) -> std::uint64_t {
    const auto at = arm.places.find({r, role});
    if (at == arm.places.end()) return 0;
    return at->second.array ? arm.weights->array_address(at->second.index)
                            : arm.weights->resource_address(at->second.index);
  };
  auto runner = en::ModelRunner::Create(arm.node, std::move(config), std::move(*source), 0, 0);
  if (!runner) return Error(runner.error());
  arm.shared = std::move(*runner);
  arm.entered = {arm.shared.get(), arm.kernel.get()};
  if (auto s = arm.shared->Setup(); !s) return s;
  if (auto s = arm.node.MapWorkspace(
          arm.shared->activations_needed(),
          std::max<std::uint64_t>(arm.shared->pool_needed(), en::kPagedExtent));
      !s)
    return s;
  if (auto s = AdmitNode(arm, layer_boundary, csa); !s) return s;
  if (auto s = arm.shared->Register(); !s) return s;
  if (auto s = arm.shared->Bind(); !s) return s;
  arm.node.Run();
  std::vector<en::LoadStats> loads;
  if (auto s = arm.node.Load(arm.shared->weights(), "shared weights", loads); !s) return s;
  const std::array<std::uint32_t, 1> owners{0};
  return arm.shared->slots().SelectSlots(owners);
}
en::Status SaveState(Arm& arm, const fs::path& path, std::uint32_t positions) {
  auto ranges = md::UsedThrough(arm.target->layout, positions, 256);
  if (!ranges) return Error(ranges.error());
  const auto base = arm.legacy ? arm.legacy->state_base() : arm.shared->slots().live(0, 0).base(0);
  const auto& closure = arm.legacy ? arm.legacy->fence_closure() : arm.shared->slots().execution();
  auto initialized =
      arm.legacy ? arm.legacy->used_state_ranges() : arm.shared->slots().live(0, 0).used_ranges();
  std::ranges::sort(initialized, {}, &en::LiveState::Range::offset);
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output) return Error("state output open failed");
  for (const auto& r : *ranges) {
    // Do not read absent/uninitialized backing even when a virtual region
    // exists. The read's exact closure is held until node.Copy completes.
    std::uint64_t cursor = r.offset;
    const auto end = r.offset + r.bytes;
    for (const auto& u : initialized)
      if (u.region == 0 && u.offset <= cursor && u.offset + u.bytes > cursor)
        cursor = std::min(end, u.offset + u.bytes);
    if (cursor != end) return Error("future-readable state interval was not initialized");
    llmp::base::Sha256 observed;
    for (std::uint64_t offset = 0; offset < r.bytes; offset += kWindow) {
      const auto n = std::min(kWindow, r.bytes - offset);
      if (auto s = arm.node.Copy(0, closure, base + r.offset + offset, arm.readback, n, true,
                                 "screen state read");
          !s)
        return s;
      observed.Update(
          std::span(static_cast<const std::byte*>(arm.readback), static_cast<std::size_t>(n)));
    }
    output << r.offset << " " << r.bytes << " " << llmp::base::ToHex(observed.Finish()) << "\n";
    if (!output) return Error("complete state digest output failed");
  }
  return {};
}
en::Status CheckBudget(Arm& arm) {
  // Observe scheduler-owned occupancy on its lane after completed work.
  return arm.node.Call(
      [&]() -> en::Status {
        if (arm.node.host_overcharges() != 0)
          return Error("screen exceeded its admitted host accounting");
        if (arm.node.catalog().OccupancyOf(arm.node.domain()).Total() > arm.node.budget())
          return Error("screen catalog occupancy exceeds its physical budget");
        return {};
      },
      "screen budget check");
}
struct Result {
  std::array<double, kPrefix / kRows + kDecode> seconds{};
  std::uint64_t eager = 0, captured = 0, replayed = 0, native_masks = 0;
};
std::expected<Result, std::string> RunArm(const fs::path& root, const fs::path& dir, bool shared,
                                          bool graphs, std::unique_ptr<Arm>& retired,
                                          bool reduced_outa_hca = false,
                                          bool first_attention = false, bool initial_hc = false,
                                          bool csa = false, bool layer_two = false,
                                          bool layer_three = false, bool first_decode = false) {
  if (static_cast<int>(first_attention) + static_cast<int>(initial_hc) + static_cast<int>(csa) +
          static_cast<int>(layer_two) + static_cast<int>(layer_three) +
          static_cast<int>(first_decode) >
      1)
    return Error("diagnostic modes are mutually exclusive");
  const bool observed =
      first_attention || initial_hc || csa || layer_two || layer_three || first_decode;
  const bool layer_boundary = layer_two || layer_three || first_decode;
  auto owned = std::make_unique<Arm>();
  Arm& arm = *owned;
  Result result;
  if (first_decode) fs::remove_all(dir);
  fs::create_directories(dir);
  if (layer_two || layer_three) {
    const auto prefix = layer_three ? "layer3-" : "layer2-";
    fs::remove(dir / (std::string(prefix) + "ffn-cur.bin"));
    fs::remove(dir / (std::string(prefix) + "ffn-cur-elided.txt"));
  }
  const auto operation = [&]() -> en::Status {
    const auto available = llmp::platform::AvailableMemoryBytes();
    if (!available || *available < (std::uint64_t{112} << 30U))
      return Error("screen needs 110 GiB node budget and 2 GiB physical headroom");
    if (auto s = arm.node.Open(); !s) return s;
    arm.node.SetHostFloor(0);
    // Previous outputs/page-cache observations remain paid until this exact
    // replacement admission. The stopped old node owns no device borrowers.
    retired.reset();
    auto target = Open(root);
    if (!target) return Error(target.error());
    arm.target = std::move(*target);
    arm.csa_inventory = csa;
    auto script = Script();
    if (observed) script.resize(first_decode ? kPrefix / kRows + 1 : 1);
    arm.history = Tokens(0, kPrefix + kDecode);
    if (auto s = shared ? StartShared(arm, dir, graphs, script, layer_boundary, csa)
                        : StartLegacy(arm, dir, graphs, reduced_outa_hca, layer_boundary, csa);
        !s)
      return s;
    if (auto s = CheckBudget(arm); !s) return s;
    if (observed) {
      if (first_decode) {
        auto layout = md::Dsv4State(md::Dsv4Flash(), kContext, kRows, md::Dsv4Window::kRing);
        if (!layout) return Error(layout.error());
        const auto at = layout->Find(0, md::Dsv4StateTensor::Kind::kRawK);
        const auto base =
            arm.legacy ? arm.legacy->state_base() : arm.shared->slots().live(0, 0).base(0);
        if (at < 0 || base == 0) return Error("first decode raw layer0 binding absent");
        const auto& raw = layout->tensors[static_cast<std::size_t>(at)];
        if (raw.offset != 0 || raw.bytes != 786432)
          return Error("first decode raw ring geometry changed");
        const auto& l = arm.target->binding.layers.at(0);
        const std::array resources{l.hc_attn_fn.index,  l.hc_attn_scale.index, l.hc_attn_base.index,
                                   l.attn_norm.index,   l.hc_ffn_fn.index,     l.hc_ffn_scale.index,
                                   l.hc_ffn_base.index, l.ffn_norm.index,      l.attn_sinks.index};
        std::array<std::uint64_t, 9> addresses{};
        if (arm.weights)
          for (std::size_t i = 0; i < resources.size(); ++i) {
            addresses[i] = arm.weights->resource_address(resources[i]);
            if (addresses[i] == 0) return Error("first decode layer0 resource address absent");
          }
        const auto& weight_owner =
            arm.weights ? *arm.weights : en::Dsv4RunnerTestAccess::TargetWeights(*arm.legacy);
        auto routed = RoutedReceipts(*arm.target, weight_owner);
        if (!routed) return Error(routed.error());
        const auto& profile = md::Dsv4Flash();
        if (profile.layers != kDecodeLayerEnds || profile.width != 4096 || profile.hc != 4 ||
            arm.target->binding.layers.size() != kDecodeLayerEnds ||
            arm.target->spec.stages.size() != 1 ||
            arm.target->spec.stages[0].layers.size() != kDecodeLayerEnds)
          return Error("layer-end fixed profile/spec/binding domain changed");
        LayerEndResources layer_resources{};
        const std::array<std::uint64_t, 4> expected_bytes{1572864, 12, 96, 16384};
        for (std::size_t layer = 0; layer < layer_resources.size(); ++layer) {
          const auto& b = arm.target->binding.layers[layer];
          const std::array bindings{&b.hc_ffn_fn, &b.hc_ffn_scale, &b.hc_ffn_base, &b.ffn_norm};
          for (std::size_t role = 0; role < bindings.size(); ++role) {
            const auto& binding = *bindings[role];
            if (binding.index >= arm.target->artifact.resources().size())
              return Error("layer-end source resource index absent");
            const auto& resource = arm.target->artifact.resources()[binding.index];
            const auto address = weight_owner.resource_address(binding.index);
            auto type = kg::GgmlTypeOf(binding.type);
            if (!type || *type != GGML_TYPE_F32 || address == 0 ||
                resource.bytes.value() != expected_bytes[role] || resource.name.size() > 128 ||
                address > UINT64_MAX - expected_bytes[role])
              return Error("layer-end source encoding/extent/address outside fixed domain");
            layer_resources[layer][role] = {
                .index = binding.index, .address = address, .bytes = expected_bytes[role]};
          }
          for (std::size_t prior = 0; prior < layer; ++prior)
            if (layer_resources[prior][0].address == layer_resources[layer][0].address)
              return Error("layer-end source mixing resources alias");
        }
        LayerEndResources attention_resources{};
        for (std::size_t layer = 0; layer < attention_resources.size(); ++layer) {
          const auto& b = arm.target->binding.layers[layer];
          const std::array bindings{&b.hc_attn_fn, &b.hc_attn_scale, &b.hc_attn_base, &b.attn_norm};
          for (std::size_t role = 0; role < bindings.size(); ++role) {
            const auto& binding = *bindings[role];
            if (binding.index >= arm.target->artifact.resources().size())
              return Error("attention source resource index absent");
            const auto& resource = arm.target->artifact.resources()[binding.index];
            const auto address = weight_owner.resource_address(binding.index);
            const auto type = kg::GgmlTypeOf(binding.type);
            if (!type || *type != GGML_TYPE_F32 || address == 0 ||
                resource.bytes.value() != expected_bytes[role] || resource.name.size() > 128 ||
                address > UINT64_MAX - expected_bytes[role])
              return Error("attention source encoding/extent/address outside fixed domain");
            attention_resources[layer][role] = {
                .index = binding.index, .address = address, .bytes = expected_bytes[role]};
          }
          for (std::size_t prior = 0; prior < layer; ++prior)
            if (attention_resources[prior][0].address == attention_resources[layer][0].address)
              return Error("attention source mixing resources alias");
          for (const auto& ffn : layer_resources)
            if (ffn[0].address == attention_resources[layer][0].address)
              return Error("attention and FFN source mixing resources alias");
        }
        std::array<std::array<RoutedArrayReceipt, 3>, kLearnedLayers.size()> learned_routed{};
        std::array<std::array<LayerEndResource, 3>, kLearnedLayers.size()> learned_resources{};
        if (profile.hash_layers != kLearnedLayer || profile.experts != 256 ||
            profile.experts_used != 6 || !profile.expert_weights_norm ||
            profile.expert_weights_scale == 1.0F)
          return Error("layer3 learned route fixed semantic profile changed");
        for (std::size_t slot = 0; slot < kLearnedLayers.size(); ++slot) {
          const auto observed_layer = kLearnedLayers[slot];
          if (observed_layer < profile.hash_layers || observed_layer >= profile.layers)
            return Error("observed layer is outside actual learned routing domain");
          auto routed_receipts = RoutedReceipts(*arm.target, weight_owner, observed_layer);
          if (!routed_receipts) return Error(routed_receipts.error());
          learned_routed[slot] = *routed_receipts;
          const auto& learned = arm.target->binding.layers[observed_layer];
          const std::array bindings{&learned.router, &learned.router_bias, &learned.down_shexp};
          const auto shared_down_type = ObservedSharedDownEncoding(observed_layer);
          if (shared_down_type == GGML_TYPE_COUNT)
            return Error("unapproved learned shared-down encoding");
          const std::array kinds{GGML_TYPE_BF16, GGML_TYPE_F32, shared_down_type};
          const std::array<std::uint64_t, 3> sizes{
              2097152, 1024,
              static_cast<std::uint64_t>(ggml_row_size(shared_down_type, 2048)) * 4096};
          for (std::size_t i = 0; i < bindings.size(); ++i) {
            const auto& binding = *bindings[i];
            if (binding.index >= arm.target->artifact.resources().size())
              return Error("learned source index absent");
            const auto& source = arm.target->artifact.resources()[binding.index];
            const auto address = weight_owner.resource_address(binding.index);
            const auto type = kg::GgmlTypeOf(binding.type);
            if (!type || *type != kinds[i] || source.bytes.value() != sizes[i] ||
                source.name.size() > 128 || address == 0 || address > UINT64_MAX - sizes[i])
              return Error("learned source encoding/extent/address differs");
            learned_resources[slot][i] = {
                .index = binding.index, .address = address, .bytes = sizes[i]};
          }
        }
        std::array<LayerEndResource, 2> shared_input_resources{};
        const auto& input_layer = arm.target->binding.layers[kThirdLearnedLayer];
        const std::array input_bindings{&input_layer.gate_shexp, &input_layer.up_shexp};
        for (std::size_t i = 0; i < input_bindings.size(); ++i) {
          const auto& binding = *input_bindings[i];
          if (binding.index >= arm.target->artifact.resources().size())
            return Error("shared-input source index absent");
          const auto& source = arm.target->artifact.resources()[binding.index];
          const auto type = kg::GgmlTypeOf(binding.type);
          const auto address = weight_owner.resource_address(binding.index);
          if (!type || *type != GGML_TYPE_Q6_K || source.bytes.value() != 6881280 ||
              source.name.size() > 128 || address == 0 || address > UINT64_MAX - 6881280)
            return Error("shared-input source encoding/extent/address differs");
          shared_input_resources[i] = {
              .index = binding.index, .address = address, .bytes = 6881280};
        }
        if (shared_input_resources[0].address == shared_input_resources[1].address)
          return Error("shared-input gate/up resources alias");
        arm.probe = std::make_unique<LayerEndsProbe>(
            arm.readback, *arm.target, base, addresses, *routed, arm.history.at(kPrefix), shared,
            layer_resources, learned_resources, learned_routed, attention_resources,
            shared_input_resources);
      } else if (csa || layer_boundary) {
        const std::uint32_t observed_layer = layer_three ? 3 : 2;
        auto layout = md::Dsv4State(md::Dsv4Flash(), kContext, kRows, md::Dsv4Window::kRing);
        if (!layout) return Error(layout.error());
        const auto base =
            arm.legacy ? arm.legacy->state_base() : arm.shared->slots().live(0, 0).base(0);
        using K = md::Dsv4StateTensor::Kind;
        const std::array<K, 3> kinds =
            layer_three ? std::array{K::kHcaStateKv, K::kHcaStateScore, K::kHcaK}
                        : std::array{K::kCsaStateKv, K::kCsaStateScore, K::kCsaK};
        std::array<std::uint64_t, 3> addresses{};
        for (std::size_t i = 0; i < kinds.size(); ++i) {
          const auto at = layout->Find(observed_layer, kinds[i]);
          if (at < 0 || base == 0) return Error("observed compressed state binding absent");
          addresses[i] = base + layout->tensors[static_cast<std::size_t>(at)].offset;
        }
        std::array<std::uint64_t, 4> weights{};
        if (arm.weights) {
          const auto& layer = arm.target->binding.layers.at(observed_layer);
          weights = {arm.weights->resource_address(layer.comp_kv.index),
                     arm.weights->resource_address(layer.comp_gate.index),
                     arm.weights->resource_address(layer.comp_ape.index),
                     arm.weights->resource_address(layer.comp_norm.index)};
          if (std::ranges::find(weights, 0) != weights.end())
            return Error("observed compressor weight binding absent");
        }
        if (layer_boundary) {
          const auto raw = layout->Find(observed_layer, K::kRawK);
          if (raw < 0) return Error("observed raw state binding absent");
          arm.probe = std::make_unique<LayerBoundaryProbe>(
              arm.readback, *arm.target, addresses, weights,
              base + layout->tensors[static_cast<std::size_t>(raw)].offset, observed_layer,
              arm.weights ? arm.weights->resource_address(
                                arm.target->binding.layers.at(observed_layer).attn_sinks.index)
                          : 0);
        } else {
          arm.probe = std::make_unique<CsaProbe>(arm.readback, *arm.target, addresses, weights);
        }
      } else if (initial_hc)
        arm.probe = std::make_unique<InitialHcProbe>(arm.readback);
      else
        arm.probe = std::make_unique<AttentionProbe>(arm.readback);
      if (!first_decode) {
        if (!kg::SetFirstAttentionObserver(nullptr, arm.probe.get()))
          return Error("another first attention observer is already installed");
        arm.probe_installed = true;
      }
    }
    const auto& closure =
        arm.legacy ? arm.legacy->fence_closure() : arm.shared->slots().execution();
    return arm.node.WithRequest(0, closure, "private DeepSeek paired screen", [&]() -> en::Status {
      for (std::size_t i = 0; i < script.size(); ++i) {
        const auto& shape = script[i];
        const auto& seg = shape.segments[0];
        if (first_decode && i == kPrefix / kRows) {
          if (seg.past != kPrefix || seg.rows != 1 ||
              !kg::SetFirstAttentionObserver(nullptr, arm.probe.get()))
            return Error("first decode observer unit/ownership differs");
          arm.probe_installed = true;
        }
        const auto tokens = std::span(arm.history).subspan(seg.past, seg.rows);
        std::vector<std::byte> logits;
        const auto start = std::chrono::steady_clock::now();
        arm.dispatched = true;
        if (shared) {
          const std::array<en::SegmentWork, 1> work{
              {{.tokens = tokens,
                .bytes = {},
                .history = std::span(arm.history).first(seg.past + seg.rows)}}};
          auto ran =
              arm.shared->RunStep({.shape = shape, .work = work, .advance = true, .ahead = {}});
          if (!ran) return Error(ran.error().what);
          if (ran->outputs.size() != 1) return Error("shared full head count");
          logits = std::move(ran->outputs[0]);
          result.eager += ran->path == en::RunPath::kEager;
          result.captured += ran->path == en::RunPath::kCaptured;
          result.replayed += ran->path == en::RunPath::kReplayed;
        } else {
          std::vector<float> values;
          if (auto s = arm.legacy->Chunk(seg.past, tokens, values); !s) return s;
          const auto bytes = std::as_bytes(std::span(values));
          logits.assign(bytes.begin(), bytes.end());
        }
        // RunStep/Chunk returned only after this submission and all its
        // observation copies completed. Save before another use of readback.
        if (observed && (!first_decode || i == kPrefix / kRows))
          if (auto saved = arm.probe->Save(dir); !saved) return saved;
        const double seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        result.seconds[i] = seconds;
        if (auto s = CheckBudget(arm); !s) return s;
        if (logits.size() != md::Dsv4Flash().vocab * sizeof(float))
          return Error("full head byte extent");
        std::ofstream out(dir / ("head-" + std::to_string(i) + ".bin"),
                          std::ios::binary | std::ios::trunc);
        out.write(reinterpret_cast<const char*>(logits.data()),
                  static_cast<std::streamsize>(logits.size()));
        if (!out) return Error("full head output failed");
        if (seg.past + seg.rows == kPrefix)
          if (auto state = SaveState(arm, dir / "state-prefill.txt", kPrefix); !state) return state;
        std::printf(
            "DS_SCREEN shared=%d graphs=%d step=%zu past=%u rows=%u visible_last=%u top=512 "
            "seconds=%.9f\n",
            shared, graphs, i, seg.past, seg.rows, (seg.past + seg.rows) / 4, seconds);
      }
      if (shared) {
        result.native_masks = arm.source->selected_masks();
        if (result.native_masks == 0 || arm.shared->coverage().violations != 0)
          return Error("native masks/coverage");
        if (graphs && (result.eager == 0 || result.captured == 0 || result.replayed < 2))
          return Error("eager/capture/changed-input replay modes were not all exercised");
      } else if (arm.legacy->coverage_violations() != 0)
        return Error(arm.legacy->first_violation());
      return SaveState(arm, dir / "state-final.txt",
                       first_decode ? kPrefix + 1
                       : observed   ? kRows
                                    : kPrefix + kDecode);
    });
  }();
  if (!operation && arm.dispatched) {
    const auto reason = operation.error();
    std::ignore = owned.release();
    return Error("submitted arm failed; whole owner retained and remaining arms forbidden: " +
                 reason);
  }
  // Unknown submitted failure returned above with the entire arm and
  // installed callback retained. Only completed/no-submission paths detach.
  if (arm.probe_installed) {
    if (!kg::SetFirstAttentionObserver(arm.probe.get(), nullptr)) {
      std::ignore = owned.release();
      return Error("observer ownership changed; whole arm retained");
    }
    arm.probe_installed = false;
  }
  const auto ended = arm.node.TearDown(arm.entered);
  if (!ended) {
    std::ignore = owned.release();
    return Error("uncertain terminal completion: whole arm retained; remaining arms forbidden: " +
                 ended.error());
  }
  // Destroy every borrower and its model metadata before releasing the
  // accounting owner. A failed teardown above retains this entire bundle.
  arm.shared.reset();
  arm.legacy.reset();
  arm.kernel.reset();
  arm.target.reset();
  arm.places.clear();
  arm.pitch.clear();
  std::vector<std::int32_t>().swap(arm.history);
  if (!operation) return Error(operation.error());
  // Keep this stopped catalog owner through the next admitted arm, or
  // through all final file comparisons. Teardown is not an uncharge proof.
  retired = std::move(owned);
  return result;
}
struct Comparison {
  std::uint64_t bytes = 0, mismatched = 0, first = 0;
};
std::expected<Comparison, std::string> CompareFiles(const fs::path& a, const fs::path& b) {
  struct File {
    int fd = -1;
    ~File() {
      if (fd >= 0) (void)::close(fd);
    }
  } left{::open(a.c_str(), O_RDONLY | O_CLOEXEC | O_NONBLOCK)},
      right{::open(b.c_str(), O_RDONLY | O_CLOEXEC | O_NONBLOCK)};
  struct stat ls{}, rs{};
  if (left.fd < 0 || right.fd < 0 || ::fstat(left.fd, &ls) != 0 || ::fstat(right.fd, &rs) != 0 ||
      !S_ISREG(ls.st_mode) || !S_ISREG(rs.st_mode))
    return Error("comparison requires two opened regular files");
  // Every emitted observation is nonempty and smaller than its paid window.
  if (ls.st_size <= 0 || rs.st_size <= 0 || ls.st_size != rs.st_size ||
      static_cast<std::uint64_t>(ls.st_size) > kLayerTwoWindow)
    return Error("comparison file lengths differ or exceed their nonempty bound");
  const auto length = static_cast<std::uint64_t>(ls.st_size);
  const auto read_exact = [](int fd, std::span<char> bytes) -> en::Status {
    std::size_t consumed = 0;
    while (consumed < bytes.size()) {
      const auto got = ::read(fd, bytes.data() + consumed, bytes.size() - consumed);
      if (got < 0 && errno == EINTR) continue;
      if (got <= 0) return Error("comparison read failed or ended before its advertised length");
      consumed += static_cast<std::size_t>(got);
    }
    return {};
  };
  Comparison result;
  std::array<char, 65536> x{}, y{};
  while (result.bytes < length) {
    const auto n =
        static_cast<std::size_t>(std::min<std::uint64_t>(x.size(), length - result.bytes));
    if (auto r = read_exact(left.fd, std::span(x).first(n)); !r) return Error(r.error());
    if (auto r = read_exact(right.fd, std::span(y).first(n)); !r) return Error(r.error());
    for (std::size_t i = 0; i < n; ++i)
      if (x[i] != y[i]) {
        if (result.mismatched++ == 0) result.first = result.bytes + i;
      }
    result.bytes += n;
  }
  // Both files must end exactly here. Read failures are not EOF, and growth
  // cannot turn a matching advertised prefix into a complete comparison.
  for (const int fd : {left.fd, right.fd}) {
    char extra = 0;
    ssize_t got = -1;
    do {
      got = ::read(fd, &extra, 1);
    } while (got < 0 && errno == EINTR);
    struct stat after{};
    if (got != 0 || ::fstat(fd, &after) != 0 || after.st_size != ls.st_size)
      return Error("comparison has an extra byte, read error or changed length");
  }
  return result;
}
void Compare(const fs::path& a, const fs::path& b) {
  const auto compared = CompareFiles(a, b);
  ASSERT_TRUE(compared.has_value()) << a << " " << b << ": " << compared.error();
  EXPECT_EQ(compared->mismatched, 0U)
      << a << " first_byte=" << compared->first << " bytes=" << compared->bytes;
}

struct ComparisonFiles {
  std::string path = testing::TempDir() + "/dsv4-compare-XXXXXX";
  bool made = ::mkdtemp(path.data()) != nullptr;
  ~ComparisonFiles() {
    if (made) {
      std::error_code ignored;
      fs::remove_all(path, ignored);
    }
  }
  fs::path file(const char* name) const { return fs::path(path) / name; }
  bool Write(const char* name, std::span<const char> bytes) const {
    std::ofstream out(file(name), std::ios::binary | std::ios::trunc);
    out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    out.close();
    return !out.fail();
  }
};
TEST(Dsv4ScreenComparison, RequiresOpenedNonemptyRegularFilesAndEqualCompleteLengths) {
  ComparisonFiles files;
  ASSERT_TRUE(files.made);
  const std::array<char, 3> bytes{'a', 'b', 'c'};
  ASSERT_TRUE(files.Write("a", bytes));
  ASSERT_TRUE(files.Write("empty", {}));
  EXPECT_FALSE(CompareFiles(files.file("absent"), files.file("absent")));
  EXPECT_FALSE(CompareFiles(files.file("a"), files.file("absent")));
  EXPECT_FALSE(CompareFiles(files.file("absent"), files.file("a")));
  EXPECT_FALSE(CompareFiles(files.file("a"), files.path));
  EXPECT_FALSE(CompareFiles(files.file("empty"), files.file("empty")));
  ASSERT_TRUE(files.Write("b", std::span(bytes).first(2)));
  EXPECT_FALSE(CompareFiles(files.file("a"), files.file("b")));
  EXPECT_FALSE(CompareFiles(files.file("b"), files.file("a")));
}
TEST(Dsv4ScreenComparison, ReadsTheFinalPartialBlockAndReportsExactByteDifferences) {
  ComparisonFiles files;
  ASSERT_TRUE(files.made);
  std::array<char, 65539> bytes{};
  ASSERT_TRUE(files.Write("a", bytes));
  ASSERT_TRUE(files.Write("b", bytes));
  const auto equal = CompareFiles(files.file("a"), files.file("b"));
  ASSERT_TRUE(equal.has_value()) << equal.error();
  EXPECT_EQ(equal->bytes, bytes.size());
  EXPECT_EQ(equal->mismatched, 0U);
  bytes.back() = 'x';
  ASSERT_TRUE(files.Write("b", bytes));
  const auto different = CompareFiles(files.file("a"), files.file("b"));
  ASSERT_TRUE(different.has_value()) << different.error();
  EXPECT_EQ(different->bytes, bytes.size());
  EXPECT_EQ(different->mismatched, 1U);
  EXPECT_EQ(different->first, bytes.size() - 1);
}
void Screen(bool reduced_outa_hca) {
  const fs::path root = fs::path(llmp::test_support::ModelsDir()) / kArtifact;
  ASSERT_TRUE(fs::exists(root)) << "This explicitly selected diagnostic requires its artifact";
  const char* scratch = std::getenv("LLMP_TEST_SCRATCH");
  const fs::path out =
      fs::path(scratch ? scratch : testing::TempDir()) /
      (reduced_outa_hca ? "dsv4-component-screen-reduced-outa-hca" : "dsv4-component-screen");
  std::printf("DS_SCREEN_RECIPE legacy=%s qualified=0\n",
              reduced_outa_hca ? "reduced-output-A-HCA" : "current-serving");
  std::printf(
      "DS_SCREEN_BUDGET compiler=%llu pinned=%llu context=%u max_rows=%u prefix=%u decode=%u\n",
      static_cast<unsigned long long>(kCompiler), static_cast<unsigned long long>(kWindow),
      kContext, kRows, kPrefix, kDecode);
  // Paired current-serving control, shared execution (prefill/eager decode,
  // capture, then changed-input replay), and a final serving bookend. A failed/unknown arm stops
  // all subsequent work.
  std::unique_ptr<Arm> retired;
  auto first = RunArm(root, out / "legacy-before", false, true, retired, reduced_outa_hca);
  ASSERT_TRUE(first) << (first ? "" : first.error());
  auto captured = RunArm(root, out / "shared-captured", true, true, retired);
  ASSERT_TRUE(captured) << (captured ? "" : captured.error());
  auto last = RunArm(root, out / "legacy-after", false, true, retired, reduced_outa_hca);
  ASSERT_TRUE(last) << (last ? "" : last.error());
  for (const auto& name : {"legacy-after", "shared-captured"}) {
    for (std::size_t i = 0; i < kPrefix / kRows + kDecode; ++i)
      Compare(out / "legacy-before" / ("head-" + std::to_string(i) + ".bin"),
              out / name / ("head-" + std::to_string(i) + ".bin"));
    for (const auto state : {"state-prefill.txt", "state-final.txt"})
      Compare(out / "legacy-before" / state, out / name / state);
  }
  // Timing is emitted only. The matched 1% family gate is unchanged and is
  // not asserted from this single representative first screen.
}
TEST(ComponentRunnerDsv4Screen, RoundedNativeRecipeSingleOwnerFullModel) { Screen(false); }
TEST(ComponentRunnerDsv4Screen, UnqualifiedReducedOutAHcaRecipeControl) { Screen(true); }
// One 512-row step per arm. No capture or timing qualification: operands
// are observed in-place around the existing first attention launch.
TEST(ComponentRunnerDsv4Screen, FirstAttentionSameOperandsDiagnostic) {
  const fs::path root = fs::path(llmp::test_support::ModelsDir()) / kArtifact;
  ASSERT_TRUE(fs::exists(root));
  const char* scratch = std::getenv("LLMP_TEST_SCRATCH");
  const fs::path out = fs::path(scratch ? scratch : testing::TempDir()) / "dsv4-first-attention";
  std::unique_ptr<Arm> retired;
  for (const auto& [name, shared] : {std::pair{"legacy-before", false}, std::pair{"shared", true},
                                     std::pair{"legacy-after", false}}) {
    auto arm = RunArm(root, out / name, shared, false, retired, true, true);
    ASSERT_TRUE(arm) << (arm ? "" : arm.error());
  }
  Compare(out / "legacy-before" / "attention-q.bin", out / "legacy-after" / "attention-q.bin");
  for (const auto* name : {"q-half", "k", "v", "mask", "sinks", "out", "hc-cur", "hc-residual",
                           "hc-post", "hc-comb", "hc-out"}) {
    const auto file = std::string("attention-") + name + ".bin";
    Compare(out / "legacy-before" / file, out / "legacy-after" / file);
    Compare(out / "legacy-before" / file, out / "shared" / file);
  }
}
TEST(ComponentRunnerDsv4Screen, InitialHcMixSameOperandsDiagnostic) {
  const fs::path root = fs::path(llmp::test_support::ModelsDir()) / kArtifact;
  ASSERT_TRUE(fs::exists(root));
  const char* scratch = std::getenv("LLMP_TEST_SCRATCH");
  const fs::path out = fs::path(scratch ? scratch : testing::TempDir()) / "dsv4-initial-hc";
  std::unique_ptr<Arm> retired;
  for (const auto& [name, shared] : {std::pair{"legacy-before", false}, std::pair{"shared", true},
                                     std::pair{"legacy-after", false}}) {
    auto arm = RunArm(root, out / name, shared, false, retired, true, false, true);
    ASSERT_TRUE(arm) << (arm ? "" : arm.error());
  }
  for (const auto* name : {"flat", "normalized", "product-input", "fn", "mixes", "pre", "folded"}) {
    const auto file = std::string("initial-hc-") + name + ".bin";
    Compare(out / "legacy-before" / file, out / "legacy-after" / file);
    Compare(out / "legacy-before" / file, out / "shared" / file);
  }
  for (const auto* name : {"legacy-before", "shared", "legacy-after"})
    Compare(out / name / "initial-hc-normalized.bin", out / name / "initial-hc-product-input.bin");
}
TEST(ComponentRunnerDsv4Screen, FirstCsaSameOperandsDiagnostic) {
  const fs::path root = fs::path(llmp::test_support::ModelsDir()) / kArtifact;
  ASSERT_TRUE(fs::exists(root));
  const char* scratch = std::getenv("LLMP_TEST_SCRATCH");
  const fs::path out = fs::path(scratch ? scratch : testing::TempDir()) / "dsv4-first-csa";
  std::unique_ptr<Arm> retired;
  for (const auto& [name, shared] : {std::pair{"legacy-before", false}, std::pair{"shared", true},
                                     std::pair{"legacy-after", false}}) {
    auto arm = RunArm(root, out / name, shared, false, retired, true, false, false, true);
    ASSERT_TRUE(arm) << (arm ? "" : arm.error());
  }
  for (const auto* name :
       {"ring-kv", "ring-score", "current-kv", "current-score", "read-ids", "pooled", "norm-weight",
        "normalized", "positions", "rotated", "write-ids", "stored"}) {
    const auto file = std::string("csa-") + name + ".bin";
    Compare(out / "legacy-before" / file, out / "legacy-after" / file);
    Compare(out / "legacy-before" / file, out / "shared" / file);
  }
}
TEST(ComponentRunnerDsv4Screen, LayerTwoCompleteBoundariesDiagnostic) {
  const fs::path root = fs::path(llmp::test_support::ModelsDir()) / kArtifact;
  ASSERT_TRUE(fs::exists(root));
  const char* scratch = std::getenv("LLMP_TEST_SCRATCH");
  const fs::path out = fs::path(scratch ? scratch : testing::TempDir()) / "dsv4-layer2";
  std::unique_ptr<Arm> retired;
  for (const auto& [name, shared] : {std::pair{"legacy-before", false}, std::pair{"shared", true},
                                     std::pair{"legacy-after", false}}) {
    auto arm = RunArm(root, out / name, shared, false, retired, true, false, false, false, true);
    ASSERT_TRUE(arm) << (arm ? "" : arm.error());
  }
  // Original Q bytes are retained; only the separately emitted effective
  // half values compare across F32/F16 query recipes. All 512 rows participate.
  Compare(out / "legacy-before" / "layer2-q.bin", out / "legacy-after" / "layer2-q.bin");
  for (const auto* name :
       {"q-half", "k", "v", "mask", "sinks", "out", "hc-cur", "hc-residual", "hc-post", "hc-comb",
        "hc-out", "ffn-residual", "ffn-post", "ffn-comb", "ffn-out"}) {
    const auto file = std::string("layer2-") + name + ".bin";
    Compare(out / "legacy-before" / file, out / "legacy-after" / file);
    Compare(out / "legacy-before" / file, out / "shared" / file);
  }
  // The exact legacy fusion has no resident ffn-cur. Assert that fact
  // rather than reading its aliased/unwritten intermediate or fabricating
  // an operand. Shared's materialized sum remains a diagnostic artifact;
  // the actual full HC-post result still compares strictly across all arms.
  for (const auto* name : {"legacy-before", "legacy-after"}) {
    ASSERT_TRUE(fs::exists(out / name / "layer2-ffn-cur-elided.txt"));
    ASSERT_FALSE(fs::exists(out / name / "layer2-ffn-cur.bin"));
  }
  ASSERT_FALSE(fs::exists(out / "shared" / "layer2-ffn-cur-elided.txt"));
  ASSERT_TRUE(fs::exists(out / "shared" / "layer2-ffn-cur.bin"));
  for (const auto* name : {"ffn-down", "ffn-route", "ffn-shared"}) {
    const auto file = std::string("layer2-") + name + ".bin";
    Compare(out / "legacy-before" / file, out / "legacy-after" / file);
  }
  for (const auto* name :
       {"ring-kv", "ring-score", "current-kv", "current-score", "read-ids", "pooled", "norm-weight",
        "normalized", "positions", "rotated", "write-ids", "stored"}) {
    const auto file = std::string("csa-") + name + ".bin";
    Compare(out / "legacy-before" / file, out / "legacy-after" / file);
    Compare(out / "legacy-before" / file, out / "shared" / file);
  }
}
TEST(ComponentRunnerDsv4Screen, LayerThreeCompleteBoundariesDiagnostic) {
  const fs::path root = fs::path(llmp::test_support::ModelsDir()) / kArtifact;
  ASSERT_TRUE(fs::exists(root));
  const char* scratch = std::getenv("LLMP_TEST_SCRATCH");
  const fs::path out = fs::path(scratch ? scratch : testing::TempDir()) / "dsv4-layer3";
  std::unique_ptr<Arm> retired;
  for (const auto& [name, shared] : {std::pair{"legacy-before", false}, std::pair{"shared", true},
                                     std::pair{"legacy-after", false}}) {
    auto arm =
        RunArm(root, out / name, shared, false, retired, true, false, false, false, false, true);
    ASSERT_TRUE(arm) << (arm ? "" : arm.error());
  }
  // Original Q bytes are retained; only the separately emitted effective
  // half values compare across F32/F16 query recipes. All 512 rows participate.
  Compare(out / "legacy-before" / "layer3-q.bin", out / "legacy-after" / "layer3-q.bin");
  for (const auto* name :
       {"q-half", "k", "v", "mask", "sinks", "out", "hc-cur", "hc-residual", "hc-post", "hc-comb",
        "hc-out", "ffn-residual", "ffn-post", "ffn-comb", "ffn-out"}) {
    const auto file = std::string("layer3-") + name + ".bin";
    Compare(out / "legacy-before" / file, out / "legacy-after" / file);
    Compare(out / "legacy-before" / file, out / "shared" / file);
  }
  // The exact legacy fusion has no resident ffn-cur. Assert that fact
  // rather than reading its aliased/unwritten intermediate or fabricating
  // an operand. Shared's materialized sum remains a diagnostic artifact;
  // the actual full HC-post result still compares strictly across all arms.
  for (const auto* name : {"legacy-before", "legacy-after"}) {
    ASSERT_TRUE(fs::exists(out / name / "layer3-ffn-cur-elided.txt"));
    ASSERT_FALSE(fs::exists(out / name / "layer3-ffn-cur.bin"));
  }
  ASSERT_FALSE(fs::exists(out / "shared" / "layer3-ffn-cur-elided.txt"));
  ASSERT_TRUE(fs::exists(out / "shared" / "layer3-ffn-cur.bin"));
  for (const auto* name : {"ffn-down", "ffn-route", "ffn-shared"}) {
    const auto file = std::string("layer3-") + name + ".bin";
    Compare(out / "legacy-before" / file, out / "legacy-after" / file);
  }
}
TEST(ComponentRunnerDsv4Screen, FirstDecodeLayerZeroCompleteBoundariesDiagnostic) {
  const fs::path root = fs::path(llmp::test_support::ModelsDir()) / kArtifact;
  ASSERT_TRUE(fs::exists(root));
  const char* scratch = std::getenv("LLMP_TEST_SCRATCH");
  const fs::path out = fs::path(scratch ? scratch : testing::TempDir()) / "dsv4-first-decode";
  std::unique_ptr<Arm> retired;
  for (const auto& [name, shared] : {std::pair{"legacy-before", false}, std::pair{"shared", true},
                                     std::pair{"legacy-after", false}}) {
    auto arm = RunArm(root, out / name, shared, false, retired, true, false, false, false, false,
                      false, true);
    ASSERT_TRUE(arm) << (arm ? "" : arm.error());
  }
  // Full physical masks match the legacy bookend. Shared has31 additional
  // initialized padding rows; Save validates every one as -inf, while the
  // actual complete query row is compared below without invented padding.
  Compare(out / "legacy-before" / "decode-layer0-mask-full.bin",
          out / "legacy-after" / "decode-layer0-mask-full.bin");
  for (const auto* arm : {"legacy-after", "shared"}) {
    // The exact original five prefill heads and state remain mandatory.
    // Only this diagnostic stops after the first decode unit.
    for (std::size_t i = 0; i < kPrefix / kRows + 1; ++i)
      Compare(out / "legacy-before" / ("head-" + std::to_string(i) + ".bin"),
              out / arm / ("head-" + std::to_string(i) + ".bin"));
    for (const auto* file : {"state-prefill.txt", "state-final.txt"})
      Compare(out / "legacy-before" / file, out / arm / file);
    for (const auto* name : {"attn-mix-x",
                             "attn-mix-fn",
                             "attn-mix-partials",
                             "attn-pre-partials",
                             "attn-pre-x",
                             "attn-pre-scale",
                             "attn-pre-base",
                             "attn-pre-norm",
                             "attn-pre-normalized",
                             "attn-pre-post",
                             "attn-pre-comb",
                             "q",
                             "q-half",
                             "k",
                             "v",
                             "mask",
                             "mask-positions",
                             "sinks",
                             "attention-out",
                             "projection-a-in",
                             "projection-a-out",
                             "projection-b-in",
                             "projection-b-out",
                             "attn-post-cur",
                             "attn-post-residual",
                             "attn-post-post",
                             "attn-post-comb",
                             "attn-post-out",
                             "ffn-mix-x",
                             "ffn-mix-fn",
                             "ffn-mix-partials",
                             "ffn-pre-partials",
                             "ffn-pre-x",
                             "ffn-pre-scale",
                             "ffn-pre-base",
                             "ffn-pre-norm",
                             "ffn-pre-normalized",
                             "ffn-pre-post",
                             "ffn-pre-comb",
                             "ffn-router-in",
                             "ffn-router-out",
                             "ffn-shared-act",
                             "ffn-shared-down",
                             "ffn-hash-token",
                             "ffn-routed-ids",
                             "ffn-route-weights",
                             "ffn-routed-act",
                             "ffn-routed-down",
                             "ffn-post-cur",
                             "ffn-post-residual",
                             "ffn-post-post",
                             "ffn-post-comb",
                             "ffn-post-out"}) {
      const auto file = std::string("decode-layer0-") + name + ".bin";
      Compare(out / "legacy-before" / file, out / arm / file);
    }
    for (const auto layer : kLearnedLayers)
      for (const auto name : kLearnedNames) {
        const auto file =
            "decode-layer" + std::to_string(layer) + "-route-" + std::string(name) + ".bin";
        Compare(out / "legacy-before" / file, out / arm / file);
      }
    Compare(out / "legacy-before" / "decode-layer26-shared-input.bin",
            out / arm / "decode-layer26-shared-input.bin");
    for (std::size_t layer = 0; layer < kDecodeLayerEnds; ++layer) {
      const auto file = "decode-layer-end-" + std::to_string(layer) + ".bin";
      Compare(out / "legacy-before" / file, out / arm / file);
      const auto attention = "decode-attention-end-" + std::to_string(layer) + ".bin";
      Compare(out / "legacy-before" / attention, out / arm / attention);
    }
  }
}

}  // namespace
