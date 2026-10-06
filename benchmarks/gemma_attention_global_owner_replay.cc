// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0
// PACKED_OPERANDS NEW_OUTPUT packed|owners. Closed D512 four-stream proof.
// --metadata checks descriptors without opening a device. No model changes.
#include <sys/stat.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <string_view>

#include "engine/paged_node.h"
#include "engine/paged_weights.h"
#include "engine/runner_resources.h"
#include "engine/support.h"
#include "gemma_attention_global_inputs.h"
#include "gemma_attention_owner_metadata.h"
#include "kernels/ggml/fattn_owner.h"
#include "kernels/ggml/ops.h"
#include "kernels/ggml/ops_ext.h"
#include "providers/device_runtime.h"
namespace en = jitllm::engine;
namespace kg = jitllm::kernels::ggml;
namespace ar = attention_global_replay;
using en::support::Error;
class Replay final : public en::PagedModel {
 public:
  explicit Replay(en::PagedNode& node) : node_(node), resources_(node, 0, 0) {}
  std::uint32_t stream() const override { return 0; }
  const jitllm::catalog::Closure& fence_closure() const override { return closure_; }
  std::vector<jitllm::catalog::ExtentId> managed_extents() const override { return {}; }
  en::Status Release() override {
    graph_.reset();
    arena_.reset();
    std::vector<std::string> problems;
    resources_.Release(problems);
    inputs_ = {};
    if (host_charged_) {
      node_.UnchargeHost(ar::kHostAllowance);
      host_charged_ = false;
    }
    return en::support::Joined(problems);
  }
  en::Status Setup(const std::filesystem::path& input, bool owners) {
    owners_ = owners;
    if (!node_.ChargeHost(ar::kHostAllowance, false)) return Error("replay host funding");
    host_charged_ = true;
    if (!inputs_.Read(input)) return Error("closed replay inputs/parameters");
    auto arena = kg::TensorArena::Create(128);
    if (!arena) return Error(arena.error().detail);
    arena_.emplace(std::move(*arena));
    constexpr std::uint64_t device_bytes = 64U << 20U;
    if (auto r = resources_.Map(storage_, "attention replay immutable operands/output",
                                device_bytes, jitllm::catalog::MemoryClass::kRuntime);
        !r)
      return r;
    std::uint64_t offset = 0;
    const auto root = [&](ggml_type type, int a, int b, int c, int d) {
      auto* tensor = ggml_new_tensor_4d(arena_->context(), type, a, b, c, d);
      kg::TensorArena::Bind(tensor, storage_.base + offset);
      offset += en::support::Round(ggml_nbytes(tensor), 256);
      return tensor;
    };
    std::array<ggml_tensor*, 4> packed{
        root(GGML_TYPE_F32, 512, 32, 1, 4), root(GGML_TYPE_F16, 512, 4, 256, 4),
        root(GGML_TYPE_F16, 512, 4, 256, 4), root(GGML_TYPE_F16, 256, 32, 1, 4)};
    roots_ = packed;
    auto* out =
        ar::Attention(arena_->context(), packed[0], packed[1], packed[2], packed[3], inputs_);
    kg::TensorArena::Bind(out, storage_.base + offset);
    offset += en::support::Round(ggml_nbytes(out), 256);
    nodes_.push_back(out);
    owner_inputs_.q = nodes_[0]->src[0];
    owner_inputs_.mask = nodes_[0]->src[3];
    owner_inputs_.output = nodes_[0];
    // Independent real descriptors at deliberately unequal pitches. This
    // catches accidentally reintroducing a common inter-owner VA stride.
    for (std::size_t owner = 0; owner < 4; ++owner) {
      offset += (owner + 1) * 256;
      auto* k = root(GGML_TYPE_F16, 512, 4, 256, 1);
      offset += (4 - owner) * 256;
      auto* v = root(GGML_TYPE_F16, 512, 4, 256, 1);
      owner_raw_k_[owner] = k;
      owner_raw_v_[owner] = v;
      owner_inputs_.k[owner] = ggml_permute(arena_->context(), k, 0, 2, 1, 3);
      owner_inputs_.v[owner] = ggml_permute(arena_->context(), v, 0, 2, 1, 3);
    }
    if (auto checked = kg::CheckFlashAttnOwners(owner_inputs_); !checked)
      return Error(checked.error().detail);
    if (offset > device_bytes) return Error("closed replay storage bound");
    if (auto r = node_.MapWorkspace(en::kPagedExtent, 16U << 20U); !r) return r;
    auto upload = resources_.Pinned(4194304), output = resources_.Pinned(ar::kOutputBytes);
    if (!upload || !output) return Error("replay staging funding");
    upload_ = *upload;
    output_ = *output;
    node_.SetHostFloor(ar::kHostAllowance);
    const auto fixed = node_.catalog().OccupancyOf(node_.domain()).Total().value();
    if (auto r = node_.Start(jitllm::base::Bytes(fixed + ar::kHostAllowance)); !r) return r;
    auto extents = resources_.extents();
    for (const auto* workspace : {&node_.activations(), &node_.pool()})
      extents.insert(extents.end(), workspace->extents.begin(), workspace->extents.end());
    auto closure = node_.catalog().ClosureOfExtents(extents);
    if (!closure) return Error(jitllm::catalog::ToString(closure.error()));
    closure_ = std::move(*closure);
    if (auto r = resources_.BindLaunch(16U << 20U); !r) return r;
    node_.Run();
    return {};
  }
  en::Status Run(const std::filesystem::path& out) {
    return node_.WithRequest(0, closure_, "attention replay protected operands",
                             [&] { return RunHeld(out); });
  }
  en::Status RunHeld(const std::filesystem::path& out) {
    for (std::size_t i = 0; i < roots_.size(); ++i) {
      std::memcpy(upload_, inputs_.data[i].data(), inputs_.data[i].size());
      if (auto r = node_.Job(
              closure_,
              [&, i](jitllm::providers::NativeStream stream) {
                return jitllm::providers::CopyAsync(stream, roots_[i]->data, upload_,
                                                    inputs_.data[i].size(),
                                                    jitllm::providers::CopyKind::kHostToDevice)
                               .ok()
                           ? jitllm::scheduler::JobResult::kQueued
                           : jitllm::scheduler::JobResult::kUnknown;
              },
              "replay staged immutable operand", 0);
          !r)
        return r;
    }
    for (std::size_t owner = 0; owner < 4; ++owner) {
      for (std::size_t which = 0; which < 2; ++which) {
        const auto& data = inputs_.data[which + 1];
        constexpr std::size_t part = 1048576;
        std::memcpy(upload_, data.data() + owner * part, part);
        auto* tensor = which == 0 ? owner_raw_k_[owner] : owner_raw_v_[owner];
        if (auto r = node_.Job(
                closure_,
                [&](jitllm::providers::NativeStream stream) {
                  return jitllm::providers::CopyAsync(stream, tensor->data, upload_, part,
                                                      jitllm::providers::CopyKind::kHostToDevice)
                                 .ok()
                             ? jitllm::scheduler::JobResult::kQueued
                             : jitllm::scheduler::JobResult::kUnknown;
                },
                "immutable independent owner upload", 0);
            !r)
          return r;
      }
    }
    auto planning = node_.Job(
        closure_,
        [&](jitllm::providers::NativeStream) {
          for (auto* node : nodes_) {
            if (owners_) {
              auto plan = kg::PlanFlashAttnOwners(resources_.launch(), owner_inputs_);
              if (!plan) return jitllm::scheduler::JobResult::kUnknown;
              std::cout << "REPLAY_PLAN owner_roots head=" << plan->original.head
                        << " blocks=" << plan->original.blocks
                        << " original_blocks_per_sm=" << plan->original_blocks_per_sm
                        << " owner_blocks_per_sm=" << plan->owner_blocks_per_sm
                        << " scratch=" << plan->original.scratch << '\n';
            } else {
              auto plan = kg::PlanFlashAttnMma(resources_.launch(), node);
              if (!plan) return jitllm::scheduler::JobResult::kUnknown;
              std::cout << "REPLAY_PLAN packed head=" << plan->head << " blocks=" << plan->blocks
                        << " scratch=" << plan->scratch << '\n';
            }
          }
          return jitllm::scheduler::JobResult::kQueued;
        },
        "attention replay checked launch plans", 0);
    if (!planning) return planning;
    const auto witness = [&](std::span<const std::byte> expected_q,
                             std::string_view phase) -> en::Status {
      for (std::size_t i = 0; i < roots_.size(); ++i) {
        auto copied = node_.Job(
            closure_,
            [&, i](jitllm::providers::NativeStream stream) {
              return jitllm::providers::CopyAsync(stream, upload_, roots_[i]->data,
                                                  inputs_.data[i].size(),
                                                  jitllm::providers::CopyKind::kDeviceToHost)
                             .ok()
                         ? jitllm::scheduler::JobResult::kQueued
                         : jitllm::scheduler::JobResult::kUnknown;
            },
            "complete GPU operand witness", 0);
        if (!copied) return copied;
        const auto expected = i == 0 ? expected_q : std::span<const std::byte>(inputs_.data[i]);
        if (std::memcmp(upload_, expected.data(), expected.size()) != 0)
          return Error("GPU operand witness changed");
      }
      for (std::size_t owner = 0; owner < 4; ++owner) {
        for (std::size_t which = 0; which < 2; ++which) {
          constexpr std::size_t part = 1048576;
          auto* tensor = which == 0 ? owner_raw_k_[owner] : owner_raw_v_[owner];
          if (auto r = node_.Job(
                  closure_,
                  [&](jitllm::providers::NativeStream stream) {
                    return jitllm::providers::CopyAsync(stream, upload_, tensor->data, part,
                                                        jitllm::providers::CopyKind::kDeviceToHost)
                                   .ok()
                               ? jitllm::scheduler::JobResult::kQueued
                               : jitllm::scheduler::JobResult::kUnknown;
                  },
                  "independent owner byte witness", 0);
              !r)
            return r;
          if (std::memcmp(upload_, inputs_.data[which + 1].data() + owner * part, part) != 0)
            return Error("independent owner cache bytes changed");
        }
      }
      std::cout << "GPU_OPERAND_WITNESS phase=" << phase << " complete_qkv_mask_byte_exact=1\n";
      return {};
    };
    if (auto r = witness(inputs_.data[0], "before"); !r) return r;

    const auto kernels = [&](kg::LaunchContext& launch) -> std::expected<void, kg::KernelFailure> {
      for (auto* node : nodes_) {
        auto r = owners_ ? kg::FlashAttnOwnerRoots(launch, owner_inputs_)
                         : kg::FlashAttnMma(launch, node);
        if (!r) return r;
      }
      return {};
    };
    const auto step = [&](bool captured) -> en::Status {
      auto r = node_.Job(
          closure_,
          [&](jitllm::providers::NativeStream stream) {
            auto launched =
                captured ? resources_.launch().Launch(*graph_) : kernels(resources_.launch());
            if (!launched) return jitllm::scheduler::JobResult::kUnknown;
            std::size_t offset = 0;
            for (auto* tensor : nodes_) {
              if (!jitllm::providers::CopyAsync(stream, static_cast<std::byte*>(output_) + offset,
                                                tensor->data, ggml_nbytes(tensor),
                                                jitllm::providers::CopyKind::kDeviceToHost)
                       .ok())
                return jitllm::scheduler::JobResult::kUnknown;
              offset += ggml_nbytes(tensor);
            }
            return jitllm::scheduler::JobResult::kQueued;
          },
          "complete attention operands replay and full publication", 0);
      if (!r) return r;
      if (!ar::Finite({static_cast<float*>(output_), ar::kOutputElements}))
        return Error("nonfinite attention replay");
      return {};
    };
    if (auto r = step(false); !r) return r;
    std::vector<float> first(static_cast<float*>(output_),
                             static_cast<float*>(output_) + ar::kOutputElements);
    auto capture_job = node_.Job(
        closure_,
        [&](jitllm::providers::NativeStream) {
          auto captured = resources_.launch().Capture(kernels);
          if (!captured) return jitllm::scheduler::JobResult::kUnknown;
          graph_.emplace(std::move(*captured));
          return jitllm::scheduler::JobResult::kQueued;
        },
        "completion-aware attention graph capture", 0);
    if (!capture_job) return capture_job;
    for (int i = 0; i < 3; ++i)
      if (auto r = step(true); !r) return r;
    if (std::memcmp(first.data(), output_, ar::kOutputBytes) != 0)
      return Error("eager/captured attention movement");
    if (!ar::Write(out / "first.f32", first)) return Error("first output write");
    std::vector<std::byte> fresh_q = inputs_.data[0];
    for (std::size_t i = 0; i < fresh_q.size(); i += sizeof(float)) {
      float v = 0;
      std::memcpy(&v, fresh_q.data() + i, sizeof(v));
      v *= -0.5f;
      std::memcpy(fresh_q.data() + i, &v, sizeof(v));
    }
    const auto upload_q = [&](std::span<const std::byte> q) -> en::Status {
      std::memcpy(upload_, q.data(), q.size());
      return node_.Job(
          closure_,
          [&](jitllm::providers::NativeStream stream) {
            return jitllm::providers::CopyAsync(stream, roots_[0]->data, upload_, q.size(),
                                                jitllm::providers::CopyKind::kHostToDevice)
                           .ok()
                       ? jitllm::scheduler::JobResult::kQueued
                       : jitllm::scheduler::JobResult::kUnknown;
          },
          "fresh query control upload", 0);
    };
    if (auto r = upload_q(fresh_q); !r) return r;
    if (auto r = step(false); !r) return r;
    std::vector<float> fresh(static_cast<float*>(output_),
                             static_cast<float*>(output_) + ar::kOutputElements);
    if (std::memcmp(first.data(), fresh.data(), ar::kOutputBytes) == 0)
      return Error("fresh query failed to change output");
    if (auto r = step(true); !r) return r;
    if (std::memcmp(fresh.data(), output_, ar::kOutputBytes) != 0)
      return Error("captured replay stale fresh query");
    if (auto r = witness(fresh_q, "fresh"); !r) return r;
    if (!ar::Write(out / "fresh.f32", fresh)) return Error("fresh output write");
    if (auto r = upload_q(inputs_.data[0]); !r) return r;
    if (auto r = step(true); !r) return r;
    if (std::memcmp(first.data(), output_, ar::kOutputBytes) != 0)
      return Error("restored query control movement");
    if (auto r = witness(inputs_.data[0], "restored"); !r) return r;
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < 32; ++i) {
      if (auto r = step(true); !r) return r;
      volatile std::size_t winners = 0;
      auto* values = static_cast<float*>(output_);
      for (std::size_t owner = 0; owner < 4; ++owner) {
        auto* begin = values + owner * 16384;
        winners =
            winners + static_cast<std::size_t>(std::max_element(begin, begin + 16384) - begin);
      }
      (void)winners;
    }
    const auto seconds = en::support::Seconds(std::chrono::steady_clock::now() - start);
    if (std::memcmp(first.data(), output_, ar::kOutputBytes) != 0)
      return Error("paid attention own-repeat movement");
    if (!ar::Write(out / "repeat.f32", {static_cast<float*>(output_), ar::kOutputElements}))
      return Error("repeat output write");
    if (auto r = witness(inputs_.data[0], "after-paid"); !r) return r;
    std::cout << "ATTENTION_REPLAY seconds=" << seconds << " completed_units=128 api_groups=" << 32
              << " native_captured_nodes=" << graph_->nodes()
              << " output_bytes_per_wave=" << ar::kOutputBytes
              << " scratch_peak=" << resources_.launch().scratch_peak().value()
              << " known_host_allowance=" << ar::kHostAllowance
              << " device_mapped=" << resources_.mapped_bytes()
              << " pinned=" << resources_.pinned_bytes() << '\n';
    return {};
  }

 private:
  en::PagedNode& node_;
  en::RunnerResources resources_;
  en::Mapped storage_;
  ar::Inputs inputs_;
  std::optional<kg::TensorArena> arena_;
  std::optional<kg::CapturedGraph> graph_;
  std::array<ggml_tensor*, 4> roots_{};
  std::vector<ggml_tensor*> nodes_;
  jitllm::catalog::Closure closure_;
  void* upload_ = nullptr;
  void* output_ = nullptr;
  kg::FlashAttnOwners owner_inputs_;
  std::array<ggml_tensor*, 4> owner_raw_k_{}, owner_raw_v_{};
  bool owners_ = false, host_charged_ = false;
};
int main(int argc, char** argv) {
  umask(0077);
  std::cout << std::setprecision(12);
  if (argc == 2 && std::string_view(argv[1]) == "--metadata")
    return attention_owner_metadata::Controls() ? 0 : 1;
  if (argc != 4) return 2;
  const std::string_view arm = argv[3];
  if (arm != "packed" && arm != "owners") return 2;
  const std::filesystem::path out = argv[2];
  std::error_code error;
  if (!std::filesystem::create_directory(out, error) || error) return 2;
  struct Lifetime {
    en::PagedNode node{{.slot_bytes = en::kSlabSlotBytes}};
    Replay replay{node};
  };
  auto lifetime = std::make_unique<Lifetime>();
  auto result = lifetime->node.Open();
  if (result) result = lifetime->replay.Setup(argv[1], arm == "owners");
  if (result) result = lifetime->replay.Run(out);
  std::array<en::PagedModel*, 1> models{&lifetime->replay};
  auto retired = lifetime->node.TearDown(models);
  if (!retired) {
    std::cerr << retired.error() << '\n';
    (void)lifetime.release();
    return 1;
  }
  if (!result) {
    std::cerr << result.error() << '\n';
    return 1;
  }
  return 0;
}
