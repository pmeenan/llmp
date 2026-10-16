// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The image's planned fusions (spec/image_spatial.h kImagePlanningPolicy;
// docs/experiments/m36-runtime-fusion/README.md) on small synthetic
// Qwen-Image-shaped graphs built by the production spatial components with
// the production layout sinking and lowered through the production GGML
// backend: every output against the reference executor's double-precision
// oracle (at the source geometry, out of the oracle's reach, against the
// same graph planned without patterns), and each fused implementation
// shown in a plan. Synthetic weights; no model files, no speed claim.
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <expected>
#include <format>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include "base/bytes.h"
#include "components/spatial.h"
#include "engine/graph_plan.h"
#include "expected_error.h"
#include "kernels/ggml/attention_f32.h"
#include "kernels/ggml/conv2d_tiled.h"
#include "kernels/ggml/cublas.h"
#include "kernels/ggml/executor.h"
#include "kernels/ggml/launch.h"
#include "kernels/ggml/step_program.h"
#include "kernels/reference/exec.h"
#include "model/qwen_image.h"
#include "planner/policy.h"
#include "providers/cuda/cuda_device_execution.h"
#include "providers/device_runtime.h"
#include "spec/image_spatial.h"

namespace {
namespace co = llmp::components;
namespace en = llmp::engine;
namespace fm = llmp::format;
namespace ir = llmp::ir;
namespace kg = llmp::kernels::ggml;
namespace kn = llmp::kernels;
namespace mo = llmp::model;
namespace rf = llmp::kernels::reference;
namespace sp = llmp::spec;
using llmp::test_support::Failed;

constexpr std::uint64_t kMiB = 1ULL << 20U;

float FromBf16(std::uint16_t bits) {
  return std::bit_cast<float>(static_cast<std::uint32_t>(bits) << 16U);
}
std::uint16_t ToBf16(float value) {
  const auto u = std::bit_cast<std::uint32_t>(value);
  return static_cast<std::uint16_t>((u + 0x7fffU + ((u >> 16U) & 1U)) >> 16U);
}
// A fixed sequence in [-1, 1): every run and the oracle see the same values.
struct Draw {
  std::uint64_t state;
  float operator()() {
    state ^= state << 13U;
    state ^= state >> 7U;
    state ^= state << 17U;
    return static_cast<float>(static_cast<double>(state >> 11U) * 0x1p-52 - 1.0);
  }
};

struct Errors {
  double rel_rms = 0, max_rel = 0;
};
Errors Compare(std::span<const float> got, std::span<const float> want) {
  double diff = 0, norm = 0, max_abs = 0;
  for (std::size_t i = 0; i < want.size(); ++i) {
    const double d = static_cast<double>(got[i]) - want[i];
    diff += d * d;
    norm += static_cast<double>(want[i]) * want[i];
    max_abs = std::max(max_abs, std::abs(d));
  }
  const double rms = std::sqrt(norm / static_cast<double>(std::max<std::size_t>(want.size(), 1)));
  return {.rel_rms = std::sqrt(diff / std::max(norm, 1e-300)), .max_rel = max_abs / rms};
}

co::SpatialPayment Unbounded() {
  return {[](std::uint64_t) { return true; }, [](std::uint64_t) {}};
}

std::vector<co::BoundTensor> Bound(std::span<const mo::QwenImageTensor> source, fm::Encoding e) {
  std::vector<co::BoundTensor> tensors;
  for (const auto& tensor : source) {
    fm::Shape shape;
    shape.rank = static_cast<std::uint8_t>(tensor.shape.size());
    for (std::size_t d = 0; d < tensor.shape.size(); ++d)
      shape.ne[d] = tensor.shape[tensor.shape.size() - 1 - d];
    tensors.push_back({tensor.name, ir::Dense(e, shape)});
  }
  return tensors;
}

// A denoiser of the source's head geometry (head_dim 128, its three rotary
// axes) at a small width: two heads, one block.
mo::QwenImageDenoiserProfile SmallDenoiser(std::uint32_t heads, std::uint32_t mlp,
                                           std::uint32_t blocks = 1) {
  mo::QwenImageDenoiserProfile p;
  p.blocks = blocks;
  p.heads = heads;
  p.width = heads * p.head_dim;
  p.mlp = mlp;
  p.context = p.width;  // the source's context_in_dim is its width
  return p;
}

// Flow inputs: text rows, latent, sinusoid, positions, mask, dt.
void FlowInputs(const co::SpatialGraph& graph, std::uint32_t text, std::uint32_t grid,
                std::vector<std::vector<float>>& inputs, std::uint64_t seed) {
  Draw draw{seed};
  const std::uint32_t rows = text + grid * grid;
  for (const auto id : graph.graph().inputs()) {
    const auto& format = graph.graph().tensor(id).format;
    std::uint64_t count = 1;
    for (std::size_t d = 0; d < format.shape.rank; ++d) count *= format.shape.ne[d];
    std::vector<float> values(count);
    if (format.encoding == fm::Encoding::kI32) {
      for (std::uint32_t r = 0; r < rows; ++r) {
        const auto position = mo::QwenImageDenoiserPosition(text, grid, grid, r);
        for (std::uint32_t a = 0; a < 3; ++a)
          values[r + std::size_t{rows} * a] = static_cast<float>(position[a]);
      }
    } else if (count == std::uint64_t{rows} * rows) {
      for (std::uint32_t q = 0; q < text; ++q)
        for (std::uint32_t k = q + 1; k < rows; ++k) values[k + std::size_t{rows} * q] = -INFINITY;
    } else if (count == 1) {
      values[0] = -0.05F;
    } else {
      for (auto& v : values) v = draw();
    }
    inputs.push_back(std::move(values));
  }
}

// A decoder of the source's structure at base 16: a 4×4 latent to 64×64.
constexpr std::uint32_t kDecoderGrid = 4;
mo::QwenImageVaeProfile SmallVae() {
  mo::QwenImageVaeProfile profile;
  profile.z_dim = 16;
  profile.base = 16;
  profile.dim_mult = {1, 2, 4, 4, 4};
  profile.res_blocks = 1;
  return profile;
}

struct FlowShape {
  std::string name;
  std::uint32_t heads, mlp, text, grid;
  bool oracle = true;
  std::uint32_t blocks = 1;
  double bound = 5e-3;  // relative RMS against the reference
};
// The image's fused implementations, each in some case's plan (a gated
// residual alone, kGatedResidualSplitName, is not: every flow joins each
// with the modulation of its sum).
constexpr std::array<std::string_view, 12> kFused{kg::kSpatialChannelNormName,
                                                  kg::kNormRopeSectionsName,
                                                  kg::kQkAttentionName,
                                                  kg::kAttentionMaskPackName,
                                                  kg::kRopeTableName,
                                                  kg::kSwiGluBf16Name,
                                                  kg::kLayerNormModulateSplitName,
                                                  kg::kGatedResidualModulateSplitName,
                                                  kg::kConv2dTiledF32,
                                                  kg::kDupUpAddName,
                                                  kg::kConv1x1Name,
                                                  kg::kGemmBf16LtName};

// The image's planning policy, as the served pipeline plans it.
llmp::planner::Policy ImagePolicy() {
  auto policy = llmp::planner::ParsePolicy(sp::kImagePlanningPolicy);
  EXPECT_TRUE(policy) << (policy ? "" : policy.error());
  return policy.value_or(llmp::planner::Policy{});
}

class ImageFusionGpu : public ::testing::Test {
 protected:
  struct Case {
    std::string name;
    std::optional<co::SpatialGraph> graph;
    std::vector<std::vector<float>> inputs;  // per graph input, logical order
    std::vector<std::vector<float>> reference;
    // False: too large for the oracle; the same graph planned without
    // patterns or layout sinking is the reference.
    bool oracle = true;
    std::map<std::string, std::size_t> plan;  // implementation: steps
    std::map<ir::WeightRef, std::uint64_t> weight_addresses;
    std::unique_ptr<kn::StepProgram> program;
    // The image's planning policy, as the served pipeline plans it.
    llmp::planner::Policy policy = ImagePolicy();
    std::uint64_t activations = 0, activation_bytes = 0;
  };

  void SetUp() override {
    auto opened = llmp::providers::cuda::OpenDeviceExecution(0);
    ASSERT_TRUE(opened);
    execution = std::move(*opened);
    auto created = execution->CreateStream();
    ASSERT_TRUE(created);
    stream = *created;
    stream_ready = true;
    const auto facts = llmp::providers::QueryDeviceFacts(0);
    ASSERT_TRUE(facts);
    const auto workspace =
        kg::CublasHandle::UpstreamWorkspace(static_cast<int>(facts->architecture));
    auto handle = kg::CublasHandle::Create(
        0, *execution, stream, {.base = Allocate(workspace.value()), .size = workspace});
    ASSERT_TRUE(handle) << Failed(handle, &kg::KernelFailure::detail).value_or("");
    cublas = std::move(*handle);
    auto made = kg::GgmlBackend::Create(0, *execution, stream, nullptr);
    ASSERT_TRUE(made) << Failed(made).value_or("");
    backend = std::move(*made);
    // As the image source declares its backend (component_image_build.cc):
    // the masked F32 attention tuned for each flow's joint rows.
    std::vector<llmp::execution::TunedShape> tuning;
    for (const std::uint64_t rows : kFlowRows)
      tuning.push_back({.implementation = kg::kAttentionF32,
                        .rows = rows,
                        .k = 128,
                        .n = 128,
                        .output_encoding = fm::Encoding::kF32});
    // And the tiled convolution the 1024 decoder takes by default, for the
    // small decoder's 3×3 shapes.
    for (const auto& step : mo::VaeDecoderPlan(SmallVae(), kDecoderGrid, kDecoderGrid)) {
      if (step.kind != mo::VaeStep::Kind::kConv3x3) continue;
      const llmp::execution::TunedShape key{.implementation = kg::kConv2dTiledF32,
                                            .rows = std::uint64_t{step.width} * step.height,
                                            .k = 9ULL * step.in_channels,
                                            .n = step.out_channels,
                                            .output_encoding = fm::Encoding::kF32};
      if (std::ranges::none_of(tuning, [&](const auto& old) {
            return old.implementation == key.implementation && old.rows == key.rows &&
                   old.k == key.k && old.n == key.n;
          }))
        tuning.push_back(key);
    }
    auto declared = backend->Declare(
        tuning, workspace.value(), [](std::uint64_t) { return true; }, [](std::uint64_t) {});
    ASSERT_TRUE(declared) << Failed(declared).value_or("");
    auto lent = backend->LendCublas(cublas.get());
    ASSERT_TRUE(lent) << Failed(lent).value_or("");
    auto launched = backend->Launch({.base = Allocate(kPool), .size = llmp::base::Bytes(kPool)});
    ASSERT_TRUE(launched) << Failed(launched).value_or("");
    launch = std::move(*launched);
  }
  void TearDown() override {
    if (!Finish()) {
      // Unproven retirement: every borrower stays alive.
      std::ignore = new decltype(cases)(std::move(cases));
      std::ignore = launch.release();
      std::ignore = backend.release();
      std::ignore = cublas.release();
      std::ignore = execution.release();
      return;
    }
    cases.clear();
    launch.reset();
    if (backend) backend->LaunchReleased();
    backend.reset();
    cublas.reset();
    if (execution && stream_ready) EXPECT_TRUE(execution->DestroyStream(stream));
    for (void* p : allocations) EXPECT_EQ(cudaFree(p), cudaSuccess);
  }
  bool Finish() {
    if (retirement_failed) return false;
    if (!execution || !stream_ready) return true;
    auto fence = execution->Record(stream);
    if (!fence) {
      retirement_failed = true;
      return false;
    }
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
    for (;;) {
      auto state = execution->Query(*fence);
      if (!state || std::chrono::steady_clock::now() >= deadline) {
        retirement_failed = true;
        ADD_FAILURE() << "unproven retirement; retain fixture owners";
        return false;
      }
      if (*state == llmp::providers::FenceState::kComplete) break;
      std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
    if (!execution->Release(*fence)) {
      retirement_failed = true;
      return false;
    }
    return true;
  }
  std::uint64_t Allocate(std::uint64_t size) {
    void* p = nullptr;
    EXPECT_EQ(cudaMalloc(&p, std::max<std::uint64_t>(size, 256)), cudaSuccess);
    allocations.push_back(p);
    return reinterpret_cast<std::uintptr_t>(p);
  }
  void Copy(std::uint64_t device, const void* host, std::size_t bytes) {
    auto native = execution->Submission(stream);
    ASSERT_TRUE(native);
    const auto copied =
        cudaMemcpyAsync(reinterpret_cast<void*>(device), host, bytes, cudaMemcpyHostToDevice,
                        static_cast<cudaStream_t>(native->handle));
    if (copied != cudaSuccess) retirement_failed = true;
    ASSERT_EQ(copied, cudaSuccess);
    ASSERT_TRUE(Finish());  // the host bytes are the caller's
  }

  // The graph's weights, drawn once, uploaded and bound into the oracle.
  void Weights(Case& c, rf::Executor* oracle, std::uint64_t seed) {
    const auto& graph = c.graph->graph();
    Draw draw{seed};
    for (const auto& weight : c.graph->weights()) {
      const auto& format = graph.tensor(weight.tensor).format;
      ASSERT_EQ(format.layout, fm::Layout::kDense);
      const auto& ne = format.shape.ne;
      std::uint64_t count = 1;
      for (std::size_t d = 0; d < format.shape.rank; ++d) count *= ne[d];
      // Unit-variance products: scale by the fan-in (a convolution's
      // [kw, kh, ci, co] reads kw · kh · ci); vectors (norm weights,
      // biases, modulation scales) near one.
      const double fan_in = format.shape.rank >= 4   ? static_cast<double>(ne[0] * ne[1] * ne[2])
                            : format.shape.rank >= 2 ? static_cast<double>(ne[0])
                                                     : 0.0;
      const float scale = fan_in > 0 ? static_cast<float>(std::sqrt(3.0 / fan_in)) : 0.2F;
      std::vector<float> values(count);
      for (auto& v : values) v = (fan_in > 0 ? 0.0F : 1.0F) + scale * draw();
      std::vector<std::byte> bytes;
      if (format.encoding == fm::Encoding::kBF16) {
        std::vector<std::uint16_t> bits(count);
        for (std::size_t i = 0; i < count; ++i) {
          bits[i] = ToBf16(values[i]);
          values[i] = FromBf16(bits[i]);
        }
        bytes.resize(count * 2);
        std::memcpy(bytes.data(), bits.data(), bytes.size());
      } else {
        ASSERT_EQ(format.encoding, fm::Encoding::kF32);
        bytes.resize(count * 4);
        std::memcpy(bytes.data(), values.data(), bytes.size());
      }
      const auto address = Allocate(bytes.size());
      Copy(address, bytes.data(), bytes.size());
      ASSERT_TRUE(c.weight_addresses.emplace(weight.ref, address).second);
      if (oracle == nullptr) continue;
      auto bound = oracle->BindValues(weight.tensor, values);
      ASSERT_TRUE(bound) << bound.error().detail;
    }
  }

  // Lowers the case's graph through the production backend, binds it to
  // the launch and computes its oracle.
  void Prepare(Case& c, std::uint64_t seed) {
    const auto& graph = c.graph->graph();
    std::optional<rf::Executor> oracle;
    if (c.oracle) {
      auto created = rf::Executor::Create(graph);
      ASSERT_TRUE(created) << created.error().detail;
      oracle.emplace(std::move(*created));
    }
    Weights(c, oracle ? &*oracle : nullptr, seed);
    ASSERT_FALSE(HasFatalFailure());
    ASSERT_EQ(c.inputs.size(), graph.inputs().size());
    if (oracle) {
      for (std::size_t i = 0; i < c.inputs.size(); ++i) {
        const auto id = graph.inputs()[i];
        if (graph.tensor(id).format.encoding == fm::Encoding::kI32) {
          std::vector<std::int64_t> ints(c.inputs[i].begin(), c.inputs[i].end());
          auto bound = oracle->BindInts(id, ints);
          ASSERT_TRUE(bound) << bound.error().detail;
        } else {
          auto bound = oracle->BindValues(id, c.inputs[i]);
          ASSERT_TRUE(bound) << bound.error().detail;
        }
      }
      const auto began = std::chrono::steady_clock::now();
      auto ran = oracle->Run();
      ASSERT_TRUE(ran) << ran.error().detail;
      std::printf("%s oracle %.1f s\n", c.name.c_str(),
                  std::chrono::duration<double>(std::chrono::steady_clock::now() - began).count());
      for (const auto id : graph.outputs()) {
        auto values = oracle->Values(id);
        ASSERT_TRUE(values) << values.error().detail;
        c.reference.push_back(std::move(*values));
      }
    }
    const en::GraphBuildIO io{graph, graph.inputs(), graph.outputs(), c.graph->weights()};
    c.activation_bytes = c.oracle ? 512 * kMiB : 6144 * kMiB;
    c.activations = Allocate(c.activation_bytes);
    kn::Addresses addresses;
    addresses.weight = [&c](const ir::WeightRef& ref, fm::PartRole) {
      const auto found = c.weight_addresses.find(ref);
      return found == c.weight_addresses.end() ? 0ULL : found->second;
    };
    addresses.weight_alignment = [](const ir::WeightRef&, fm::PartRole) { return 256; };
    addresses.activations = c.activations;
    addresses.activation_bytes = c.activation_bytes;
    addresses.activation_alignment = 256;
    const auto owned = backend->Backend();
    ASSERT_TRUE(owned.owner && owned.owner->Ready());
    const auto& table = owned.owner->backend();
    auto program = en::BuildComponentProgram(
        io, table, c.policy, addresses,
        {.staged = std::vector(io.inputs.begin(), io.inputs.end()),
         .written = table.written,
         .within = table.within,
         .alignment_staged = std::vector(io.inputs.begin(), io.inputs.end())});
    ASSERT_TRUE(program) << Failed(program).value_or("");
    c.program = std::move(*program);
    auto bound = c.program->Bind(*launch);
    ASSERT_TRUE(bound) << bound.error().detail;
    ASSERT_EQ(c.program->inputs().size(), c.inputs.size());
    if (c.oracle) ASSERT_EQ(c.program->outputs().size(), c.reference.size());
    if (const auto* plan = c.program->bound()) {
      auto& histogram = c.plan;
      for (std::size_t i = 0; i < plan->size(); ++i) ++histogram[std::string(plan->at(i).name)];
      std::string text;
      for (const auto& [name, n] : histogram) text += std::format(" {}={}", name, n);
      std::printf("%s plan:%s\n", c.name.c_str(), text.c_str());
      RecordProperty(c.name + ".plan", text);
    }
  }

  // One run; a run the launch refused returns its refusal and the planned
  // steps around it.
  std::expected<std::vector<std::vector<float>>, std::string> Run(Case& c) {
    std::vector<std::vector<float>> out;
    const auto& graph = c.graph->graph();
    for (std::size_t i = 0; i < c.inputs.size(); ++i) {
      const auto& slot = c.program->inputs()[i];
      std::vector<std::byte> bytes;
      if (graph.tensor(graph.inputs()[i]).format.encoding == fm::Encoding::kI32) {
        std::vector<std::int32_t> ints(c.inputs[i].begin(), c.inputs[i].end());
        bytes.resize(ints.size() * 4);
        std::memcpy(bytes.data(), ints.data(), bytes.size());
      } else {
        bytes.resize(c.inputs[i].size() * 4);
        std::memcpy(bytes.data(), c.inputs[i].data(), bytes.size());
      }
      EXPECT_EQ(slot.bytes, bytes.size()) << c.name << " input " << i;
      if (slot.bytes != bytes.size()) return out;
      Copy(slot.device_address, bytes.data(), bytes.size());
      if (HasFatalFailure()) return out;
    }
    auto native = execution->Submission(stream);
    EXPECT_TRUE(native);
    if (!native) return out;
    const auto ran = c.program->Run(*launch, *native);
    if (!Finish()) return out;
    if (!ran) {
      std::string around;
      if (const auto* plan = c.program->bound()) {
        std::size_t at = 0;
        std::sscanf(ran.error().detail.c_str(), "step %zu", &at);
        for (std::size_t i = at > 6 ? at - 6 : 0; i < std::min(plan->size(), at + 4); ++i)
          around += std::format("\n  {}{}: {}", i == at ? "> " : "  ", i, plan->at(i).name);
      }
      return std::unexpected(ran.error().detail + around);
    }
    for (const auto& slot : c.program->outputs()) {
      std::vector<float> values(slot.bytes / 4);
      EXPECT_EQ(cudaMemcpy(values.data(), reinterpret_cast<const void*>(slot.device_address),
                           slot.bytes, cudaMemcpyDeviceToHost),
                cudaSuccess);
      out.push_back(std::move(values));
    }
    return out;
  }

  // The case's run against its reference (`reference`: the primitive
  // plan's run, where the oracle is out of reach).
  void Check(Case& c, double bound, Case* reference = nullptr) {
    SCOPED_TRACE(c.name);
    if (reference != nullptr) {
      const auto want = Run(*reference);
      ASSERT_TRUE(want) << want.error();
      c.reference = *want;
    }
    const auto got = Run(c);
    if (HasFatalFailure()) return;
    ASSERT_TRUE(got) << got.error();
    ASSERT_EQ(got->size(), c.reference.size());
    std::string summary;
    for (std::size_t o = 0; o < got->size(); ++o) {
      ASSERT_EQ((*got)[o].size(), c.reference[o].size()) << "output " << o;
      const auto e = Compare((*got)[o], c.reference[o]);
      EXPECT_LE(e.rel_rms, bound) << "output " << o;
      EXPECT_TRUE(std::isfinite(e.rel_rms)) << "output " << o;
      summary +=
          std::format("{}out{} rel_rms {:.3e} max {:.3e}", o ? "; " : "", o, e.rel_rms, e.max_rel);
    }
    RecordProperty(c.name + ".error", summary);
    std::printf("%s: %s\n", c.name.c_str(), summary.c_str());
  }

  // A flow of `shape.blocks` blocks: text rows then a square latent grid,
  // as the image plans it (`primitive`: without patterns or layout
  // sinking).
  Case* FlowCase(const FlowShape& shape, bool primitive = false) {
    auto& c = *cases.emplace_back(std::make_unique<Case>());
    c.name = primitive ? shape.name + "_primitive" : shape.name;
    c.oracle = shape.oracle;
    if (primitive) c.policy = llmp::planner::Policy{};
    const auto profile = SmallDenoiser(shape.heads, shape.mlp, shape.blocks);
    const auto source = mo::DenoiserTensors(profile);
    const auto image_rows = shape.grid * shape.grid;
    auto mapped = sp::ImageFlowSpec(profile, shape.text, image_rows, source);
    EXPECT_TRUE(mapped) << Failed(mapped).value_or("");
    if (!mapped) return nullptr;
    const auto tensors = Bound(source, fm::Encoding::kBF16);
    const std::array<std::span<const co::BoundTensor>, 1> tables{tensors};
    const std::uint32_t rows = shape.text + image_rows;
    const std::array owners{co::SpatialOwner{0, {1, shape.text, profile.context, 1}}};
    const std::array ports{co::SpatialInput{0, 0, {1, image_rows, profile.in_channels, 1}},
                           co::SpatialInput{0, 1, {1, 2, profile.timestep_dim, 1}},
                           co::SpatialInput{0, 2, {rows, 3, 1, 1}, fm::Encoding::kI32},
                           co::SpatialInput{0, 3, {rows, rows, 1, 1}},
                           co::SpatialInput{0, 4, {1, 1, 1, 1}}};
    auto graph = co::BuildSpatialStep(mapped->spec, tables, owners, Unbounded(),
                                      {.links = mapped->links,
                                       .reference_buffers = false,
                                       .exports = mapped->exports,
                                       .inputs = ports,
                                       .sink_layouts = !primitive});
    EXPECT_TRUE(graph) << Failed(graph).value_or("");
    if (!graph) return nullptr;
    c.graph.emplace(std::move(*graph));
    FlowInputs(*c.graph, shape.text, shape.grid, c.inputs, 7);
    Prepare(c, 11);
    if (HasFatalFailure()) return nullptr;
    return &c;
  }
  void Flow(const FlowShape& shape) {
    Case* c = FlowCase(shape);
    if (c == nullptr || HasFatalFailure()) return;
    Case* reference = nullptr;
    if (!shape.oracle) {
      reference = FlowCase(shape, true);
      if (reference == nullptr || HasFatalFailure()) return;
    }
    Check(*c, shape.bound, reference);
  }

  static constexpr std::uint64_t kPool = 256 * kMiB;
  // The flows' joint rows (text + latent grid²), as FlowShape lists them.
  static constexpr std::array<std::uint64_t, 3> kFlowRows{8 + 16 * 16, 25 + 64 * 64, 8 + 6 * 6};
  std::unique_ptr<llmp::providers::DeviceExecution> execution;
  llmp::providers::StreamId stream{};
  bool stream_ready = false, retirement_failed = false;
  std::unique_ptr<kg::CublasHandle> cublas;
  std::unique_ptr<kg::GgmlBackend> backend;
  std::unique_ptr<kn::StepLaunch> launch;
  std::vector<void*> allocations;
  std::vector<std::unique_ptr<Case>> cases;
  // The decoder: the source's structure at a small base width and grid,
  // against the oracle.
  void Decoder() {
    auto& c = *cases.emplace_back(std::make_unique<Case>());
    c.name = "decoder";
    // The tiled convolution the 1024 decoder selects by default
    // (ggml_spatial_test); these small shapes would take the direct one.
    c.policy.ranks.push_back({"sm_121", std::string(kg::kConv2dTiledF32), 0});
    llmp::planner::Canonicalize(c.policy);
    const auto profile = SmallVae();
    const std::uint32_t grid = kDecoderGrid;
    const auto source = mo::VaeDecoderTensors(profile);
    auto mapped = sp::ImageDecodeSpec(profile, grid, source);
    EXPECT_TRUE(mapped) << Failed(mapped).value_or("");
    if (!mapped) return;
    const auto tensors = Bound(source, fm::Encoding::kF32);
    const std::array<std::span<const co::BoundTensor>, 1> tables{tensors};
    const std::array owners{co::SpatialOwner{0, {1, grid * grid, profile.z_dim, 1}}};
    const std::array ports{co::SpatialInput{0, 0, {1, 2, profile.z_dim, 1}}};
    auto graph = co::BuildSpatialStep(mapped->spec, tables, owners, Unbounded(),
                                      {.links = mapped->links,
                                       .reference_buffers = false,
                                       .exports = mapped->exports,
                                       .inputs = ports,
                                       .sink_layouts = true});
    EXPECT_TRUE(graph) << Failed(graph).value_or("");
    if (!graph) return;
    c.graph.emplace(std::move(*graph));
    Draw draw{13};
    for (const auto id : c.graph->graph().inputs()) {
      const auto& format = c.graph->graph().tensor(id).format;
      std::uint64_t count = 1;
      for (std::size_t d = 0; d < format.shape.rank; ++d) count *= format.shape.ne[d];
      std::vector<float> values(count);
      if (count == 2ULL * profile.z_dim) {
        for (std::uint32_t z = 0; z < profile.z_dim; ++z) {
          values[2 * z] = 1.0F + 0.2F * draw();  // std
          values[2 * z + 1] = 0.1F * draw();     // mean
        }
      } else {
        for (auto& v : values) v = draw();
      }
      c.inputs.push_back(std::move(values));
    }
    Prepare(c, 17);
    if (HasFatalFailure()) return;
    // The tiled convolution rounds its operands to BF16.
    Check(c, 5e-3);
  }
};

TEST_F(ImageFusionGpu, FlowsAndDecoderMatchTheOracle) {
  // The flow: one transformer block of thirty-two heads (the source width),
  // text rows then a square latent grid.
  Flow({"flow_w4096", 32, 1024, 8, 16});
  ASSERT_FALSE(HasFatalFailure());
  // Two of the source's blocks at its 1024² geometry: the oracle is out of
  // reach, so the same graph planned without patterns is the reference.
  Flow({"flow_source", 32, 12288, 25, 64, false, 2, 6e-3});
  ASSERT_FALSE(HasFatalFailure());
  Decoder();
  ASSERT_FALSE(HasFatalFailure());
  // Every fused implementation in some plan.
  for (const std::string_view implementation : kFused)
    EXPECT_TRUE(std::ranges::any_of(cases, [&](const auto& c) {
      return !c->name.ends_with("_primitive") && c->plan.contains(std::string(implementation));
    })) << implementation;
}

// Width 256 (two heads): where the run-time path's split broadcast and
// gated residual miscomputed by 21% (README, "Findings" 2), the plan
// matches the oracle.
TEST_F(ImageFusionGpu, SmallWidthFlowMatchesTheOracle) {
  Flow({"flow_w256", 2, 512, 8, 6});  // joint rows in kFlowRows
}
}  // namespace
