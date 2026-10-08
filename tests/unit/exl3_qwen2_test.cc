// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// One phase of the native EXL3 plan under jitLLM's dispatch
// (kernels/exl3/qwen2.h) on a GB10 (label `gpu`; all but the GEMV-choice
// test also on a discrete GPU the build targets, `gpu-discrete`, D-082),
// over synthetic weights of the model's shapes (all zero, which every
// kernel accepts):
// - a prefill and a single-token step bind and run, every operation
//   launching on the one stream, with finite (zero) logits;
// - Bind refuses a registry without an implementation the plan names
//   (BP-S4) or with another identity for it (BP-S2), memory smaller than
//   the plan's, and in EXL3-O a table whose packed plan is not upstream's
//   GEMV choice on this device; nothing is substituted;
// - every weight an operation reads, a linear's included, is named at the
//   address bound for it (what the Tier E recording hashes);
// - tables whose rewrite failed partway leave no record, so the
//   multi-GEMM is refused until they are written again (BP-P5).

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <format>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include "execution/registry.h"
#include "expected_error.h"
#include "kernels/exl3/implementations.h"
#include "kernels/exl3/launch.h"
#include "kernels/exl3/qwen2.h"
#include "kernels/exl3/recon_gemm.h"
#include "kernels/exl3/validate.h"
#include "kernels/ggml/implementations.h"
#include "kernels/ggml/launch.h"
#include "model/qwen2_exl3.h"
#include "providers/cuda/cuda_device_execution.h"
#include "providers/device_execution.h"

namespace {

namespace exl3 = jitllm::kernels::exl3;
namespace kg = jitllm::kernels::ggml;
namespace model = jitllm::model;
using jitllm::test_support::Failed;

const model::Qwen2Profile& Profile() { return model::Qwen25Instruct05BExl3(); }

// Resource indices are irrelevant here: the memory map gives addresses.
model::Exl3LinearBinding Linear(std::string name, int k, int n, int bits) {
  model::Exl3LinearBinding l;
  l.name = std::move(name);
  l.k = k;
  l.n = n;
  l.bits = bits;
  return l;
}

model::Exl3Binding Binding(int bits) {
  const auto& p = Profile();
  model::Exl3Binding b;
  const auto width = static_cast<int>(p.width);
  const auto kv = static_cast<int>(p.kv_width());
  const auto ffn = static_cast<int>(p.ffn);
  b.lm_head = Linear("lm_head", width, static_cast<int>(p.vocab), 8);
  for (std::uint32_t l = 0; l < p.layers; ++l) {
    const std::string n = std::format("model.layers.{}.", l);
    model::Exl3LayerBinding layer;
    layer.q = Linear(n + "self_attn.q_proj", width, width, bits);
    layer.k = Linear(n + "self_attn.k_proj", width, kv, bits);
    layer.v = Linear(n + "self_attn.v_proj", width, kv, bits);
    layer.o = Linear(n + "self_attn.o_proj", width, width, bits);
    layer.gate = Linear(n + "mlp.gate_proj", width, ffn, bits);
    layer.up = Linear(n + "mlp.up_proj", width, ffn, bits);
    layer.down = Linear(n + "mlp.down_proj", ffn, width, bits);
    b.layers.push_back(layer);
  }
  return b;
}

model::Exl3LinearPlan Packed(model::Exl3Path path, int shape, int blocks, int concurrency = 0,
                             int config = 0) {
  model::Exl3LinearPlan plan;
  plan.path = path;
  plan.shape = shape;
  plan.blocks = blocks;
  plan.concurrency = concurrency;
  plan.config = config;
  return plan;
}

class Exl3Qwen2Test : public ::testing::Test {
 protected:
  void SetUp() override {
    execution_ = std::move(jitllm::providers::cuda::OpenDeviceExecution(0).value());
    stream_ = execution_->CreateStream().value();
    // Zeroed weights shared by every linear (read only): the largest
    // trellis (the head's, K = 8), side vectors and biases, the embedding,
    // the norms, the cache.
    const auto& p = Profile();
    trellis_ = Zeroed(std::uint64_t{p.width} * p.vocab * 8 / 8);
    vectors_ = Zeroed(std::uint64_t{p.vocab} * 2);
    embed_ = Zeroed(std::uint64_t{p.vocab} * p.width * 2);
    norms_ = Zeroed(std::uint64_t{p.width} * 4);
    kv_ = Zeroed(std::uint64_t{p.layers} * 2 * 4096 * p.kv_width() * 2);
    tables_ = Zeroed(std::uint64_t{48} * p.layers);
    locks_ = Zeroed(exl3::kLockBytes);
    auto launch = exl3::LaunchContext::Create(0, *execution_, stream_, locks_);
    ASSERT_TRUE(launch.has_value()) << launch.error().detail;
    launch_ = std::move(*launch);
    auto planning = kg::LaunchContext::Create(0, *execution_, stream_, {.base = 0, .size = {}});
    ASSERT_TRUE(planning.has_value());
    planning_ = std::move(*planning);
    std::vector<jitllm::execution::Implementation> all = kg::Implementations();
    for (auto& i : exl3::Implementations()) {
      all.push_back(std::move(i));
    }
    declared_ = all;
  }

  void TearDown() override {
    Finish();
    launch_.reset();
    planning_.reset();
    ASSERT_TRUE(execution_->DestroyStream(stream_).has_value());
    for (void* pointer : device_) {
      EXPECT_EQ(cudaFree(pointer), cudaSuccess);
    }
  }

  void Finish() {
    const auto fence = execution_->Record(stream_).value();
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
    auto state = jitllm::providers::FenceState::kPending;
    while ((state = execution_->Query(fence).value()) == jitllm::providers::FenceState::kPending &&
           std::chrono::steady_clock::now() < deadline) {
      std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
    ASSERT_EQ(state, jitllm::providers::FenceState::kComplete);
    ASSERT_TRUE(execution_->Release(fence).has_value());
  }

  std::uint64_t Zeroed(std::uint64_t bytes) {
    void* pointer = nullptr;
    EXPECT_EQ(cudaMalloc(&pointer, bytes), cudaSuccess);
    EXPECT_EQ(cudaMemset(pointer, 0, bytes), cudaSuccess);
    device_.push_back(pointer);
    return reinterpret_cast<std::uintptr_t>(pointer);
  }

  exl3::Qwen2Memory Memory(const model::Exl3Binding& b, std::uint64_t region_bytes) {
    const auto& p = Profile();
    exl3::Qwen2Memory m;
    const auto linear = [&](const model::Exl3LinearBinding& l, bool bias) {
      return exl3::Qwen2Linear{.weights = {.trellis = trellis_,
                                           .suh = vectors_,
                                           .svh = vectors_,
                                           .k = l.k,
                                           .n = l.n,
                                           .bits = l.bits},
                               .bias = bias ? vectors_ : 0};
    };
    m.embed = embed_;
    m.final_norm = norms_;
    m.lm_head = linear(b.lm_head, false);
    m.cells = 4096;
    m.region = Zeroed(region_bytes);
    m.region_bytes = region_bytes;
    std::vector<std::uint64_t> words;
    const std::uint64_t per_layer = std::uint64_t{2} * 4096 * p.kv_width() * 2;
    for (std::uint32_t l = 0; l < p.layers; ++l) {
      exl3::Qwen2Layer layer;
      layer.attn_norm = norms_;
      layer.mlp_norm = norms_;
      layer.q = linear(b.layers[l].q, true);
      layer.k = linear(b.layers[l].k, true);
      layer.v = linear(b.layers[l].v, true);
      // Each bias at its own place in the zeroed vectors, so a bias read
      // from another layer or projection shows in Address.
      layer.q.bias = vectors_ + (256ULL * ((3ULL * l) + 0));
      layer.k.bias = vectors_ + (256ULL * ((3ULL * l) + 1));
      layer.v.bias = vectors_ + (256ULL * ((3ULL * l) + 2));
      layer.o = linear(b.layers[l].o, false);
      layer.gate = linear(b.layers[l].gate, false);
      layer.up = linear(b.layers[l].up, false);
      layer.down = linear(b.layers[l].down, false);
      layer.trellis_table = tables_ + (48ULL * l);
      layer.suh_table = layer.trellis_table + 16;
      layer.svh_table = layer.trellis_table + 32;
      layer.tables_written = exl3::MultiGemmTables(layer.gate.weights, layer.up.weights);
      words.insert(words.end(), layer.tables_written.begin(), layer.tables_written.end());
      layer.k_cache = kv_ + (per_layer * l);
      layer.v_cache = layer.k_cache + (per_layer / 2);
      m.layers.push_back(layer);
    }
    EXPECT_EQ(
        cudaMemcpy(reinterpret_cast<void*>(tables_), words.data(), words.size() * 8,  // NOLINT
                   cudaMemcpyHostToDevice),
        cudaSuccess);
    // A pageable copy may return before its DMA lands, on the legacy stream,
    // which the provider's non-blocking stream does not wait for.
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    return m;
  }

  // A table forcing every case at `rows` onto upstream's path, with the
  // GEMM's shape 1 at a grid of 16 and the multi-GEMM's shape 2 at 16 × 2
  // (valid on any device); EXL3-O's GEMV where `gemv` says, at upstream's
  // choice on this device.
  model::Exl3LaunchTable Table(const model::Exl3Binding& b, int rows, model::Exl3Arm arm) {
    model::Exl3LaunchTable table("test");
    const auto add = [&](const model::Exl3LinearBinding& l, bool f32) {
      model::Exl3LinearPlan plan = Packed(model::Exl3Path::kGemm, 1, 16);
      const exl3::Weights w{.k = l.k, .n = l.n, .bits = l.bits};
      if (arm == model::Exl3Arm::kO) {
        auto upstream =
            launch_->UpstreamGemv(w, f32 ? exl3::Output::kF32 : exl3::Output::kF16, rows);
        if (upstream && upstream->has_value()) {
          plan = Packed(model::Exl3Path::kGemv, 0, (*upstream)->blocks, 0, (*upstream)->config);
        }
      }
      EXPECT_TRUE(table.Add(l.name, rows, plan).has_value());
    };
    add(b.lm_head, false);
    for (const auto& layer : b.layers) {
      add(layer.q, false);
      add(layer.k, false);
      add(layer.v, false);
      add(layer.o, true);
      add(layer.down, true);
      EXPECT_TRUE(
          table.Add(layer.gate.name, rows, Packed(model::Exl3Path::kMulti, 2, 16, 2), layer.up.name)
              .has_value());
    }
    return table;
  }

  std::unique_ptr<jitllm::providers::DeviceExecution> execution_;
  jitllm::providers::StreamId stream_;
  std::vector<void*> device_;
  std::uint64_t trellis_ = 0, vectors_ = 0, embed_ = 0, norms_ = 0, kv_ = 0, tables_ = 0,
                locks_ = 0;
  std::unique_ptr<exl3::LaunchContext> launch_;
  std::unique_ptr<kg::LaunchContext> planning_;
  std::vector<jitllm::execution::Implementation> declared_;
};

TEST_F(Exl3Qwen2Test, APrefillAndAStepBindAndRunOnOneStream) {
  const auto registry = jitllm::execution::Registry::Create(declared_).value();
  const auto binding = Binding(4);
  auto gemm = exl3::ReconGemm::Create();
  ASSERT_TRUE(gemm.has_value()) << gemm.error().detail;
  for (const model::Exl3Phase phase :
       {model::Exl3Phase{.rows = 32, .past = 0}, model::Exl3Phase{.rows = 1, .past = 32}}) {
    const auto table = Table(binding, phase.rows, model::Exl3Arm::kG);
    auto plan = model::PlanPhase(Profile(), binding, table, model::Exl3Arm::kG, phase);
    ASSERT_TRUE(plan.has_value()) << plan.error();
    const auto memory = Memory(binding, plan->region);
    auto program =
        exl3::Qwen2Program::Bind(registry, Profile(), *plan, memory, *planning_, *launch_);
    ASSERT_TRUE(program.has_value()) << program.error().detail;
    EXPECT_EQ((*program)->bound().size(),
              static_cast<std::size_t>(std::ranges::count_if(
                  plan->ops, [](const model::Exl3Op& o) { return !o.implementation.empty(); })));
    // The same plan binds to the same identity; the attention's scratch
    // is what its launcher draws.
    auto again = exl3::Qwen2Program::Bind(registry, Profile(), *plan, memory, *planning_, *launch_);
    ASSERT_TRUE(again.has_value());
    EXPECT_EQ((*again)->identity(), (*program)->identity());
    EXPECT_GT((*program)->ggml_scratch(), 0U);
    // Every weight an operation reads by name is at the address bound for
    // its layer: what --record-ops hashes, and op_tier_e.py holds to the
    // artifact's tensor of that name and layer.
    std::size_t weights = 0;
    for (std::size_t i = 0; i < plan->ops.size(); ++i) {
      const model::Exl3Op& o = plan->ops[i];
      const exl3::Qwen2Layer* layer =
          o.layer >= 0 ? &memory.layers[static_cast<std::size_t>(o.layer)] : nullptr;
      for (const std::string& in : o.inputs) {
        std::uint64_t want = 0;
        if (in == "embed_table") {
          want = memory.embed;
        } else if (in == "final_norm.w") {
          want = memory.final_norm;
        } else if (layer != nullptr && in == "attn_norm.w") {
          want = layer->attn_norm;
        } else if (layer != nullptr && in == "mlp_norm.w") {
          want = layer->mlp_norm;
        } else if (layer != nullptr && in == "q_proj.bias") {
          want = layer->q.bias;
        } else if (layer != nullptr && in == "k_proj.bias") {
          want = layer->k.bias;
        } else if (layer != nullptr && in == "v_proj.bias") {
          want = layer->v.bias;
        } else {
          continue;
        }
        ++weights;
        EXPECT_NE(want, 0U) << o.name << " " << in;
        EXPECT_EQ((*program)->Address(i, in), want) << o.name << " layer " << o.layer << " " << in;
      }
    }
    EXPECT_EQ(weights, 2 + (5 * std::size_t{Profile().layers}));
    // And every linear's trellis and side vectors, the multi-GEMM's for
    // both its linears and its three tables: a linear bound to another
    // layer's weights shows in the recording too.
    std::size_t linears = 0;
    for (std::size_t i = 0; i < plan->ops.size(); ++i) {
      const model::Exl3Op& o = plan->ops[i];
      if (o.owner != model::Exl3Owner::kExl3 || o.name.ends_with(".bias_add")) {
        continue;
      }
      const exl3::Qwen2Layer* layer =
          o.layer >= 0 ? &memory.layers[static_cast<std::size_t>(o.layer)] : nullptr;
      const auto of = [&](std::string_view name) -> const exl3::Qwen2Linear& {
        if (layer == nullptr) {
          return memory.lm_head;
        }
        for (const auto& [suffix, l] :
             {std::pair{"q_proj", &layer->q}, std::pair{"k_proj", &layer->k},
              std::pair{"v_proj", &layer->v}, std::pair{"o_proj", &layer->o},
              std::pair{"gate_proj", &layer->gate}, std::pair{"up_proj", &layer->up},
              std::pair{"down_proj", &layer->down}}) {
          if (name.ends_with(suffix)) {
            return *l;
          }
        }
        ADD_FAILURE() << "no linear " << name;
        return memory.lm_head;
      };
      std::vector<std::pair<std::string, std::uint64_t>> want;
      if (o.launch.path == model::Exl3Path::kMulti) {
        for (const auto& [prefix, l] : {std::pair{"gate_proj.", &layer->gate.weights},
                                        std::pair{"up_proj.", &layer->up.weights}}) {
          want.emplace_back(std::string(prefix) + "trellis", l->trellis);
          want.emplace_back(std::string(prefix) + "suh", l->suh);
          want.emplace_back(std::string(prefix) + "svh", l->svh);
        }
        want.emplace_back("table.trellis", layer->trellis_table);
        want.emplace_back("table.suh", layer->suh_table);
        want.emplace_back("table.svh", layer->svh_table);
      } else {
        const exl3::Weights& w = of(o.linear).weights;
        want = {{"trellis", w.trellis}, {"suh", w.suh}, {"svh", w.svh}};
      }
      const auto bound = (*program)->BoundWeights(i);
      EXPECT_EQ(std::vector(bound.begin(), bound.end()), want) << o.name << " layer " << o.layer;
      for (const auto& [name, address] : want) {
        EXPECT_EQ((*program)->Address(i, name), address) << o.name << " " << name;
      }
      ++linears;
    }
    EXPECT_EQ(linears, 1 + (6 * std::size_t{Profile().layers}));

    void* pool = nullptr;
    ASSERT_EQ(cudaMalloc(&pool, (*program)->ggml_scratch()), cudaSuccess);
    device_.push_back(pool);
    auto ggml =
        kg::LaunchContext::Create(0, *execution_, stream_,
                                  {.base = reinterpret_cast<std::uintptr_t>(pool),
                                   .size = jitllm::base::Bytes((*program)->ggml_scratch())});
    ASSERT_TRUE(ggml.has_value());
    void* host = nullptr;
    const exl3::HostInputs layout = exl3::HostInputsLayout(*plan);
    ASSERT_EQ(cudaMallocHost(&host, layout.bytes), cudaSuccess);
    std::vector<std::int32_t> tokens(static_cast<std::size_t>(phase.rows), 7);
    ASSERT_TRUE(exl3::WriteHostInputs(Profile(), *plan, tokens,
                                      std::span(static_cast<std::byte*>(host), layout.bytes))
                    .has_value());
    std::size_t before = 0;
    std::size_t after = 0;
    exl3::Qwen2Hooks hooks{.before = [&](std::size_t) -> std::expected<void, exl3::KernelFailure> {
                             ++before;
                             return {};
                           },
                           .after = [&](std::size_t) -> std::expected<void, exl3::KernelFailure> {
                             ++after;
                             return {};
                           }};
    auto ran = (*program)->Run(**ggml, *launch_, **gemm, *execution_, stream_,
                               reinterpret_cast<std::uintptr_t>(host), hooks);
    ASSERT_TRUE(ran.has_value()) << ran.error().detail;
    Finish();
    EXPECT_EQ(before, plan->ops.size());
    EXPECT_EQ(after, plan->ops.size());
    std::vector<std::uint16_t> logits(plan->tensors.at("logits").bytes() / 2);
    const std::uint64_t logits_at = memory.region + plan->slots.at("logits").offset;
    // NOLINTNEXTLINE(performance-no-int-to-ptr): a device address.
    const auto* source = reinterpret_cast<const void*>(logits_at);
    ASSERT_EQ(cudaMemcpy(logits.data(), source, logits.size() * 2, cudaMemcpyDeviceToHost),
              cudaSuccess);
    EXPECT_TRUE(std::ranges::all_of(logits, [](std::uint16_t v) { return (v & 0x7FFFU) == 0; }));
    ASSERT_EQ(cudaFreeHost(host), cudaSuccess);
    ggml->reset();
  }
}

TEST_F(Exl3Qwen2Test, BindRefusesAMissingOrStaleImplementationAndShortMemory) {
  const auto binding = Binding(4);
  const auto table = Table(binding, 32, model::Exl3Arm::kG);
  const auto plan =
      model::PlanPhase(Profile(), binding, table, model::Exl3Arm::kG, {.rows = 32, .past = 0})
          .value();
  const auto memory = Memory(binding, plan.region);
  {
    // A build without ExLlamaV3's bias add: the plan is unsupported.
    auto without = declared_;
    std::erase_if(without, [](const auto& i) { return i.name == "exl3.bias_add"; });
    const auto registry = jitllm::execution::Registry::Create(without).value();
    const auto refused =
        Failed(exl3::Qwen2Program::Bind(registry, Profile(), plan, memory, *planning_, *launch_),
               &exl3::KernelFailure::detail);
    EXPECT_NE(refused.value_or("").find("exl3.bias_add"), std::string::npos)
        << refused.value_or("");
  }
  {
    // Another identity for the vector attention: the plan binds the
    // registry's declaration, which no kernel of this build accepts.
    auto stale = declared_;
    for (auto& i : stale) {
      if (i.name == "ggml.flash_attn_ext.vec") {
        i.variant += " (another build)";
      }
    }
    const auto registry = jitllm::execution::Registry::Create(stale).value();
    EXPECT_TRUE(
        Failed(exl3::Qwen2Program::Bind(registry, Profile(), plan, memory, *planning_, *launch_))
            .has_value());
  }
  {
    const auto registry = jitllm::execution::Registry::Create(declared_).value();
    auto short_memory = memory;
    short_memory.region_bytes = plan.region - 256;
    EXPECT_TRUE(Failed(exl3::Qwen2Program::Bind(registry, Profile(), plan, short_memory, *planning_,
                                                *launch_))
                    .has_value());
  }
}

TEST_F(Exl3Qwen2Test, InExl3OTheTablesPackedPlanMustBeUpstreamsGemvChoice) {
  const auto registry = jitllm::execution::Registry::Create(declared_).value();
  const model::Exl3Phase step{.rows = 1, .past = 32};
  // At K = 4 upstream takes the GEMV at one row (on GB10, where the
  // per-linear sweep saw it); the table's GEMV plan binds.
  const auto k4 = Binding(4);
  const auto o_table = Table(k4, 1, model::Exl3Arm::kO);
  const auto plan = model::PlanPhase(Profile(), k4, o_table, model::Exl3Arm::kO, step);
  ASSERT_TRUE(plan.has_value()) << plan.error();
  const auto memory = Memory(k4, plan->region);
  EXPECT_TRUE(exl3::Qwen2Program::Bind(registry, Profile(), *plan, memory, *planning_, *launch_)
                  .has_value());
  EXPECT_TRUE(std::ranges::any_of(
      plan->ops, [](const model::Exl3Op& o) { return o.implementation == "exl3.linear.gemv"; }));
  // The GEMM where upstream takes the GEMV is refused.
  const auto g_table = Table(k4, 1, model::Exl3Arm::kG);
  const auto gemm_plan = model::PlanPhase(Profile(), k4, g_table, model::Exl3Arm::kO, step);
  ASSERT_TRUE(gemm_plan.has_value());
  const auto refused = Failed(
      exl3::Qwen2Program::Bind(registry, Profile(), *gemm_plan, memory, *planning_, *launch_),
      &exl3::KernelFailure::detail);
  EXPECT_NE(refused.value_or("").find("upstream takes the GEMV"), std::string::npos)
      << refused.value_or("");
}

// The CUDA provider, but the `fail_at`th copy (from 0) fails.
class FailingCopies final : public jitllm::providers::DeviceExecution {
 public:
  FailingCopies(DeviceExecution& inner, int fail_at) : inner_(inner), fail_at_(fail_at) {}
  std::expected<jitllm::providers::StreamId, jitllm::providers::Failure> CreateStream() override {
    return inner_.CreateStream();
  }
  std::expected<void, jitllm::providers::Failure> DestroyStream(
      jitllm::providers::StreamId stream) override {
    return inner_.DestroyStream(stream);
  }
  std::expected<void, jitllm::providers::Failure> Copy(jitllm::providers::StreamId stream,
                                                       std::uint64_t destination,
                                                       std::uint64_t source,
                                                       jitllm::base::Bytes size) override {
    if (copies_++ == fail_at_) {
      return std::unexpected(jitllm::providers::Failure{.detail = "an injected copy failure"});
    }
    return inner_.Copy(stream, destination, source, size);
  }
  std::expected<void, jitllm::providers::Failure> Zero(jitllm::providers::StreamId stream,
                                                       std::uint64_t destination,
                                                       jitllm::base::Bytes size) override {
    return inner_.Zero(stream, destination, size);
  }
  std::expected<jitllm::providers::NativeStream, jitllm::providers::Failure> Submission(
      jitllm::providers::StreamId stream) override {
    return inner_.Submission(stream);
  }
  std::expected<void, jitllm::providers::Failure> Wait(jitllm::providers::StreamId stream,
                                                       jitllm::providers::FenceId fence) override {
    return inner_.Wait(stream, fence);
  }
  std::expected<jitllm::providers::FenceId, jitllm::providers::Failure> Record(
      jitllm::providers::StreamId stream) override {
    return inner_.Record(stream);
  }
  std::expected<jitllm::providers::FenceState, jitllm::providers::Failure> Query(
      jitllm::providers::FenceId fence) override {
    return inner_.Query(fence);
  }
  std::expected<void, jitllm::providers::Failure> Release(
      jitllm::providers::FenceId fence) override {
    return inner_.Release(fence);
  }

 private:
  DeviceExecution& inner_;
  int fail_at_;
  int copies_ = 0;
};

// BP-P5 through a relocation: gate and up of the last layer move, and their
// tables' rewrite fails partway (at that layer's second table). The
// multi-GEMM of that layer is refused at
// Bind until the tables are written again; nothing launches with tables
// that may still hold the old addresses or half of the new ones.
TEST_F(Exl3Qwen2Test, TablesWhoseRewriteFailedPartwayRefuseTheMultiGemm) {
  const auto registry = jitllm::execution::Registry::Create(declared_).value();
  const auto binding = Binding(4);
  const auto table = Table(binding, 32, model::Exl3Arm::kG);
  const auto plan =
      model::PlanPhase(Profile(), binding, table, model::Exl3Arm::kG, {.rows = 32, .past = 0})
          .value();
  auto memory = Memory(binding, plan.region);
  ASSERT_TRUE(exl3::Qwen2Program::Bind(registry, Profile(), plan, memory, *planning_, *launch_)
                  .has_value());

  // The last layer's gate and up come back elsewhere (the last, so no
  // later layer's missing record hides a stale one at Bind).
  const auto& p = Profile();
  const std::size_t moved_at = p.layers - 1;
  exl3::Qwen2Layer& moved = memory.layers[moved_at];
  for (exl3::Weights* w : {&moved.gate.weights, &moved.up.weights}) {
    w->trellis = Zeroed(exl3::TrellisBytes(*w));
    w->suh = Zeroed(std::uint64_t{p.width} * 2);
    w->svh = Zeroed(std::uint64_t{p.ffn} * 2);
  }
  // Stale tables whose record was not updated are refused (validate.h).
  EXPECT_TRUE(
      Failed(exl3::Qwen2Program::Bind(registry, Profile(), plan, memory, *planning_, *launch_))
          .has_value());

  void* staging = nullptr;
  const std::size_t staging_bytes = std::size_t{48} * p.layers;
  ASSERT_EQ(cudaMallocHost(&staging, staging_bytes), cudaSuccess);
  const std::span<std::byte> stage(static_cast<std::byte*>(staging), staging_bytes);
  {
    // Three copies per layer: the rewrite fails at that layer's suh table.
    FailingCopies failing(*execution_, static_cast<int>((3 * moved_at) + 1));
    EXPECT_TRUE(Failed(exl3::WriteMultiGemmTables(failing, stream_, stage, memory)).has_value());
    Finish();
  }
  for (std::size_t l = 0; l < memory.layers.size(); ++l) {
    const auto written =
        exl3::MultiGemmTables(memory.layers[l].gate.weights, memory.layers[l].up.weights);
    EXPECT_EQ(memory.layers[l].tables_written == written, l < moved_at) << "layer " << l;
  }
  const auto refused =
      Failed(exl3::Qwen2Program::Bind(registry, Profile(), plan, memory, *planning_, *launch_),
             &exl3::KernelFailure::detail);
  EXPECT_NE(refused.value_or("").find("rebuild them"), std::string::npos) << refused.value_or("");

  // Written again, in full: the tables hold the moved weights and Bind
  // takes them.
  ASSERT_TRUE(exl3::WriteMultiGemmTables(*execution_, stream_, stage, memory).has_value());
  Finish();
  std::vector<std::uint64_t> words(std::size_t{6} * p.layers);
  ASSERT_EQ(cudaMemcpy(words.data(), reinterpret_cast<const void*>(tables_),  // NOLINT
                       words.size() * 8, cudaMemcpyDeviceToHost),
            cudaSuccess);
  for (std::size_t l = 0; l < memory.layers.size(); ++l) {
    const auto want =
        exl3::MultiGemmTables(memory.layers[l].gate.weights, memory.layers[l].up.weights);
    EXPECT_TRUE(
        std::equal(want.begin(), want.end(), words.begin() + static_cast<std::ptrdiff_t>(6 * l)))
        << "layer " << l;
    EXPECT_EQ(memory.layers[l].tables_written, want);
  }
  EXPECT_TRUE(exl3::Qwen2Program::Bind(registry, Profile(), plan, memory, *planning_, *launch_)
                  .has_value());
  ASSERT_EQ(cudaFreeHost(staging), cudaSuccess);
}

}  // namespace
