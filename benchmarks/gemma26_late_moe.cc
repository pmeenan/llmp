// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
// INPUT_DIRECTORY INPUT_MANIFEST_SHA NEW_OUTPUT route|reduce primitive|fused.
// Common-input diagnostic only; no model, backend runtime or policy changes.
#include <sys/stat.h>

#include <array>
#include <cstring>
#include <iostream>
#include <memory>
#include <optional>
#include <string_view>

#include "../docs/experiments/gemma26-late-moe/graphs.h"
#include "../docs/experiments/gemma26-late-moe/inputs.h"
#include "engine/paged_node.h"
#include "engine/paged_weights.h"
#include "engine/runner_resources.h"
#include "engine/support.h"
#include "kernels/ggml/executor.h"
#include "kernels/ggml/gemma_moe_fusion.h"
#include "providers/device_runtime.h"
namespace en = llmp::engine;
namespace kg = llmp::kernels::ggml;
namespace lm = late_moe;
using en::support::Error;
class Replay final : public en::PagedModel {
 public:
  explicit Replay(en::PagedNode& node) : node_(node), resources_(node, 0, 0) {}
  std::uint32_t stream() const override { return 0; }
  const llmp::catalog::Closure& fence_closure() const override { return closure_; }
  std::vector<llmp::catalog::ExtentId> managed_extents() const override { return {}; }
  en::Status Release() override {
    captured_.reset();
    bound_.reset();
    registry_.reset();
    arena_.reset();
    std::vector<std::string> problems;
    resources_.Release(problems);
    inputs_ = {};
    if (host_charged_) {
      node_.UnchargeHost(lm::kHostAllowance);
      host_charged_ = false;
    }
    return en::support::Joined(problems);
  }
  en::Status Setup(const std::filesystem::path& input, std::string_view manifest, bool route,
                   bool fused) {
    route_ = route;
    fused_ = fused;
    if (!node_.ChargeHost(lm::kHostAllowance, false)) return Error("operator host funding");
    host_charged_ = true;
    if (!inputs_.Load(input, manifest, route)) return Error("authenticated finite closed operands");
    auto arena = kg::TensorArena::Create(128);
    if (!arena) return Error(arena.error().detail);
    arena_.emplace(std::move(*arena));
    graph_ = route ? lm::Routing(arena_->context()) : lm::Reduction(arena_->context());
    constexpr std::uint64_t device_bytes = 32U << 20U;
    if (auto r = resources_.Map(storage_, "late MoE complete roots and immutable operands",
                                device_bytes, llmp::catalog::MemoryClass::kRuntime);
        !r)
      return r;
    std::uint64_t offset = 0;
    const auto bind = [&](ggml_tensor* tensor) {
      kg::TensorArena::Bind(tensor, storage_.base + offset);
      offset += en::support::Round(ggml_nbytes(tensor), 256);
    };
    for (auto* tensor : graph_.inputs) bind(tensor);
    for (auto* tensor : graph_.nodes)
      if (!tensor->view_src) bind(tensor);
    if (offset > device_bytes) return Error("closed complete-root storage bound");
    kg::BindViews(graph_.nodes);
    kg::DeviceChoices choices;
    choices.fuse_gemma_route = route && fused;
    choices.fuse_gemma_reduce = !route && fused;
    auto plan = kg::PlanGraph(graph_.nodes, false, choices);
    if (!plan) return Error(plan.error().detail);
    plan_ = std::move(*plan);
    if (graph_.nodes.size() != (route ? 10U : 17U) ||
        (fused &&
         (plan_.steps.size() != 1 || plan_.steps[0].nodes != graph_.nodes ||
          plan_.steps[0].implementation != (route ? kg::kGemmaRouteName : kg::kGemmaReduceName))))
      return Error("actual constructed descriptors/fused selector mismatch");
    if (!fused)
      for (const auto& step : plan_.steps)
        if (step.implementation == kg::kGemmaRouteName ||
            step.implementation == kg::kGemmaReduceName)
          return Error("primitive policy unexpectedly fused");
    auto registry = llmp::execution::Registry::Create(kg::Implementations());
    if (!registry) return Error("registry construction");
    registry_.emplace(std::move(*registry));
    auto bound = kg::BoundGraph::Bind(*registry_, plan_);
    if (!bound) return Error(bound.error().detail);
    bound_.emplace(std::move(*bound));
    for (std::size_t i = 0; i < graph_.nodes.size(); ++i)
      std::cout << "LATE_DESCRIPTOR index=" << i << " op=" << ggml_op_name(graph_.nodes[i]->op)
                << " view=" << bool(graph_.nodes[i]->view_src) << '\n';
    for (std::size_t i = 0; i < plan_.steps.size(); ++i)
      std::cout << "LATE_IMPLEMENTATION step=" << i << " name=" << plan_.steps[i].implementation
                << " descriptors=" << plan_.steps[i].nodes.size() << '\n';
    if (auto r = node_.MapWorkspace(en::kPagedExtent, 16U << 20U); !r) return r;
    auto upload = resources_.Pinned(8U << 20U), output = resources_.Pinned(1U << 20U);
    if (!upload || !output) return Error("operator pinned staging/publication funding");
    upload_ = *upload;
    output_ = *output;
    output_bytes_ = route ? 4096U : 720896U;
    node_.SetHostFloor(lm::kHostAllowance);
    const auto fixed = node_.catalog().OccupancyOf(node_.domain()).Total().value();
    if (auto r = node_.Start(llmp::base::Bytes(fixed + lm::kHostAllowance)); !r) return r;
    auto extents = resources_.extents();
    for (const auto* workspace : {&node_.activations(), &node_.pool()})
      extents.insert(extents.end(), workspace->extents.begin(), workspace->extents.end());
    auto closure = node_.catalog().ClosureOfExtents(extents);
    if (!closure) return Error(llmp::catalog::ToString(closure.error()));
    closure_ = std::move(*closure);
    if (auto r = resources_.BindLaunch(16U << 20U); !r) return r;
    node_.Run();
    return {};
  }
  en::Status Run(const std::filesystem::path& out) {
    return node_.WithRequest(0, closure_, "late operator complete protected owners",
                             [&] { return RunHeld(out); });
  }
  en::Status RunHeld(const std::filesystem::path& out) {
    auto r = node_.Job(
        closure_,
        [&](llmp::providers::NativeStream stream) {
          return llmp::providers::FillAsync(stream, reinterpret_cast<void*>(storage_.base), 0,
                                            32U << 20U)
                         .ok()
                     ? llmp::scheduler::JobResult::kQueued
                     : llmp::scheduler::JobResult::kUnknown;
        },
        "initialize complete mapped roots including unwritten sort tails", 0);
    if (!r) return r;
    auto scratch = kg::PlanScratch(resources_.launch(), plan_);
    if (!scratch || *scratch > (16U << 20U)) return Error("bounded operator launch scratch");
    const auto kernels = [&](kg::LaunchContext& launch) { return bound_->Run(launch); };
    const auto step = [&](bool capture) -> en::Status {
      auto result = node_.Job(
          closure_,
          [&](llmp::providers::NativeStream stream) {
            auto queued =
                capture ? resources_.launch().Launch(*captured_) : kernels(resources_.launch());
            if (!queued) return llmp::scheduler::JobResult::kUnknown;
            std::size_t offset = 0;
            for (auto* tensor : graph_.outputs) {
              const bool ids = tensor->type == GGML_TYPE_I32;
              const std::size_t rows = ids ? 64U : 1U;
              const std::size_t bytes = ids ? 32U : ggml_nbytes(tensor);
              for (std::size_t row = 0; row < rows; ++row) {
                auto* source =
                    static_cast<const std::byte*>(tensor->data) + (ids ? row * tensor->nb[1] : 0);
                if (!llmp::providers::CopyAsync(stream, static_cast<std::byte*>(output_) + offset,
                                                source, bytes,
                                                llmp::providers::CopyKind::kDeviceToHost)
                         .ok())
                  return llmp::scheduler::JobResult::kUnknown;
                offset += bytes;
              }
            }
            return offset == output_bytes_ ? llmp::scheduler::JobResult::kQueued
                                           : llmp::scheduler::JobResult::kUnknown;
          },
          "complete operator execution and selected/full publication", 0);
      if (!result) return result;
      auto bytes = std::span(static_cast<const std::byte*>(output_), output_bytes_);
      if (route_) {
        for (std::size_t i = 0; i < 2048; i += 4) {
          std::int32_t id = -1;
          std::memcpy(&id, bytes.data() + i, 4);
          if (id < 0 || id >= 128) return Error("noncanonical selected expert");
        }
        bytes = bytes.subspan(2048);
      }
      return lm::Finite(bytes) ? en::Status{} : Error("nonfinite complete operator output");
    };
    for (std::size_t layer = 0; layer < inputs_.layers.size(); ++layer) {
      const auto& inputs = inputs_.layers[layer];
      for (std::size_t i = 0; i < graph_.inputs.size(); ++i) {
        std::memcpy(upload_, inputs[i].data(), inputs[i].size());
        auto uploaded = node_.Job(
            closure_,
            [&, i](llmp::providers::NativeStream stream) {
              return llmp::providers::CopyAsync(stream, graph_.inputs[i]->data, upload_,
                                                inputs[i].size(),
                                                llmp::providers::CopyKind::kHostToDevice)
                             .ok()
                         ? llmp::scheduler::JobResult::kQueued
                         : llmp::scheduler::JobResult::kUnknown;
            },
            "retired pinned common-input upload", 0);
        if (!uploaded) return uploaded;
      }
      const auto witness = [&]() -> en::Status {
        for (std::size_t i = 0; i < graph_.inputs.size(); ++i) {
          auto copied = node_.Job(
              closure_,
              [&, i](llmp::providers::NativeStream stream) {
                return llmp::providers::CopyAsync(stream, upload_, graph_.inputs[i]->data,
                                                  inputs[i].size(),
                                                  llmp::providers::CopyKind::kDeviceToHost)
                               .ok()
                           ? llmp::scheduler::JobResult::kQueued
                           : llmp::scheduler::JobResult::kUnknown;
              },
              "complete immutable operand byte witness", 0);
          if (!copied) return copied;
          if (std::memcmp(upload_, inputs[i].data(), inputs[i].size()) != 0)
            return Error("common input changed on device");
        }
        return {};
      };
      if (auto checked = witness(); !checked) return checked;
      if (auto executed = step(false); !executed) return executed;
      std::vector<std::byte> first(static_cast<std::byte*>(output_),
                                   static_cast<std::byte*>(output_) + output_bytes_);
      if (!captured_) {
        auto captured = node_.Job(
            closure_,
            [&](llmp::providers::NativeStream) {
              auto graph = resources_.launch().Capture(kernels);
              if (!graph) return llmp::scheduler::JobResult::kUnknown;
              captured_.emplace(std::move(*graph));
              return llmp::scheduler::JobResult::kQueued;
            },
            "complete stable-address operator graph capture", 0);
        if (!captured) return captured;
      }
      for (int repeat = 0; repeat < 3; ++repeat) {
        if (auto executed = step(true); !executed) return executed;
        if (std::memcmp(first.data(), output_, output_bytes_) != 0)
          return Error("eager/captured own output movement");
      }
      if (auto checked = witness(); !checked) return checked;
      const auto name = "layer-" + std::to_string(layer + 28);
      if (!lm::Write(out / (name + "-first.bin"), first) ||
          !lm::Write(out / (name + "-repeat.bin"),
                     {static_cast<const std::byte*>(output_), output_bytes_}))
        return Error("exclusive full output write");
      std::cout << "LATE_NATIVE layer=" << layer + 28 << " kind=" << (route_ ? "route" : "reduce")
                << " policy=" << (fused_ ? "fused" : "primitive")
                << " constructed_descriptors=" << graph_.nodes.size()
                << " executed_plan_steps=" << plan_.steps.size()
                << " captured_nodes=" << captured_->nodes() << " own_repeat=1 immutable_inputs=1"
                << " output_bytes=" << output_bytes_ << " output_sha=" << lm::Hash(first) << '\n';
    }
    return {};
  }

 private:
  en::PagedNode& node_;
  en::RunnerResources resources_;
  en::Mapped storage_;
  lm::Inputs inputs_;
  lm::Graph graph_;
  kg::GraphPlan plan_;
  std::optional<kg::TensorArena> arena_;
  std::optional<llmp::execution::Registry> registry_;
  std::optional<kg::BoundGraph> bound_;
  std::optional<kg::CapturedGraph> captured_;
  llmp::catalog::Closure closure_;
  void* upload_ = nullptr;
  void* output_ = nullptr;
  std::size_t output_bytes_ = 0;
  bool route_ = false, fused_ = false, host_charged_ = false;
};
int main(int argc, char** argv) {
  umask(0077);
  if (argc == 2 && std::string_view(argv[1]) == "--metadata") {
    for (bool route : {true, false}) {
      for (bool fused : {false, true}) {
        auto arena = kg::TensorArena::Create(128);
        if (!arena) return 1;
        auto graph = route ? lm::Routing(arena->context()) : lm::Reduction(arena->context());
        std::uint64_t address = UINT64_C(0x100000000);
        const auto bind = [&](ggml_tensor* tensor) {
          kg::TensorArena::Bind(tensor, address);
          address += en::support::Round(ggml_nbytes(tensor), 256);
        };
        for (auto* tensor : graph.inputs) bind(tensor);
        for (auto* tensor : graph.nodes)
          if (!tensor->view_src) bind(tensor);
        kg::BindViews(graph.nodes);
        kg::DeviceChoices choices;
        choices.fuse_gemma_route = route && fused;
        choices.fuse_gemma_reduce = !route && fused;
        auto planned = kg::PlanGraph(graph.nodes, false, choices);
        if (!planned || graph.nodes.size() != (route ? 10U : 17U) ||
            planned->steps.size() != (fused   ? 1U
                                      : route ? 6U
                                              : 9U))
          return 1;
        auto registry = llmp::execution::Registry::Create(kg::Implementations());
        if (!registry || !kg::BoundGraph::Bind(*registry, *planned)) return 1;
        std::cout << "LATE_METADATA kind=" << (route ? "route" : "reduce")
                  << " policy=" << (fused ? "fused" : "primitive")
                  << " actual_descriptors=" << graph.nodes.size()
                  << " actual_plan_steps=" << planned->steps.size() << " launches=0\n";
      }
    }
    return 0;
  }
  if (argc != 6 ||
      (std::string_view(argv[4]) != "route" && std::string_view(argv[4]) != "reduce") ||
      (std::string_view(argv[5]) != "primitive" && std::string_view(argv[5]) != "fused") ||
      !lm::Hex(argv[2]))
    return 2;
  std::error_code error;
  const std::filesystem::path out = argv[3];
  if (!std::filesystem::create_directory(out, error) || error) return 2;
  struct Lifetime {
    en::PagedNode node{{.slot_bytes = en::kSlabSlotBytes}};
    Replay replay{node};
  };
  auto lifetime = std::make_unique<Lifetime>();
  auto result = lifetime->node.Open();
  if (result)
    result = lifetime->replay.Setup(argv[1], argv[2], std::string_view(argv[4]) == "route",
                                    std::string_view(argv[5]) == "fused");
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
  std::cout << "LATE_NATIVE_RETIRED complete=1\n";
  return 0;
}
