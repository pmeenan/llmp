// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Every ComponentDef in Catalog() (token-stage and spatial-stage) and its
// supported weight encodings executed on the GPU through planner::Plan,
// kernels::ggml::LowerStep and kernels::ggml::BoundGraph::Run against the
// component's independent scalar CPU reference (Def::reference /
// SpatialOps::reference) and the IR reference executor on small synthetic
// fixtures (label `gpu`).

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <optional>
#include <random>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include "base/bytes.h"
#include "component_catalog_cases.h"
#include "components/component.h"
#include "components/spatial.h"
#include "components/step.h"
#include "execution/registry.h"
#include "format/encoding.h"
#include "ggml.h"
#include "ir/graph.h"
#include "kernels/common/step_program.h"
#include "kernels/ggml/cublas.h"
#include "kernels/ggml/executor.h"
#include "kernels/ggml/implementations.h"
#include "kernels/ggml/launch.h"
#include "kernels/ggml/lower.h"
#include "kernels/reference/decode.h"
#include "kernels/reference/exec.h"
#include "planner/plan.h"
#include "providers/cuda/cuda_device_execution.h"
#include "providers/device_execution.h"

namespace llmp::components {
namespace {

using base::Bytes;
using format::Encoding;
using providers::DeviceExecution;
using providers::FenceState;
using providers::StreamId;
using test_catalog::CatalogSpatialCase;
using test_catalog::CatalogSpatialRecipe;
using test_catalog::CatalogSpatialRecipes;
using test_catalog::CatalogTokenCase;
using test_catalog::CatalogTokenRecipe;
using test_catalog::CatalogTokenRecipes;
using test_catalog::Gap;
using test_catalog::MakeSingleSpatialSpec;
using test_catalog::UnboundedSpatialPayment;

namespace ex = llmp::execution;
namespace kg = llmp::kernels::ggml;
namespace kn = llmp::kernels;
namespace pl = llmp::planner;
namespace ref = llmp::kernels::reference;

constexpr std::uint64_t kPoolBytes = 128ULL << 20;
constexpr std::uint64_t kCublasBytes = 32ULL << 20;

struct DeviceBufferScope {
  DeviceExecution& execution;
  StreamId stream;
  std::vector<void*> ptrs;

  ~DeviceBufferScope() {
    for (void* p : ptrs) {
      EXPECT_EQ(cudaFree(p), cudaSuccess);
    }
  }

  std::uint64_t Allocate(std::size_t bytes) {
    const std::size_t rounded = std::max<std::size_t>(((bytes + 255) / 256) * 256, 256);
    void* ptr = nullptr;
    EXPECT_EQ(cudaMalloc(&ptr, rounded), cudaSuccess);
    EXPECT_EQ(cudaMemset(ptr, 0, rounded), cudaSuccess);
    ptrs.push_back(ptr);
    return reinterpret_cast<std::uintptr_t>(ptr);
  }

  void Upload(std::uint64_t dst, std::span<const std::byte> src) {
    if (src.empty()) return;
    EXPECT_EQ(
        cudaMemcpy(reinterpret_cast<void*>(dst), src.data(), src.size(), cudaMemcpyHostToDevice),
        cudaSuccess);
  }

  std::vector<std::byte> DownloadBytes(std::uint64_t src, std::size_t bytes) {
    std::vector<std::byte> out(bytes);
    if (bytes != 0) {
      EXPECT_EQ(
          cudaMemcpy(out.data(), reinterpret_cast<const void*>(src), bytes, cudaMemcpyDeviceToHost),
          cudaSuccess);
    }
    return out;
  }

  std::vector<float> ReadTensorValues(const ggml_tensor* t) {
    const std::size_t span = static_cast<std::size_t>(t->ne[0] - 1) * t->nb[0] +
                             static_cast<std::size_t>(t->ne[1] - 1) * t->nb[1] +
                             static_cast<std::size_t>(t->ne[2] - 1) * t->nb[2] +
                             static_cast<std::size_t>(t->ne[3] - 1) * t->nb[3] + t->nb[0];
    const auto raw = DownloadBytes(reinterpret_cast<std::uintptr_t>(t->data), span);
    std::vector<float> out;
    out.reserve(static_cast<std::size_t>(ggml_nelements(t)));
    for (std::int64_t i3 = 0; i3 < t->ne[3]; ++i3) {
      for (std::int64_t i2 = 0; i2 < t->ne[2]; ++i2) {
        for (std::int64_t i1 = 0; i1 < t->ne[1]; ++i1) {
          for (std::int64_t i0 = 0; i0 < t->ne[0]; ++i0) {
            const std::size_t off =
                static_cast<std::size_t>(i0) * t->nb[0] + static_cast<std::size_t>(i1) * t->nb[1] +
                static_cast<std::size_t>(i2) * t->nb[2] + static_cast<std::size_t>(i3) * t->nb[3];
            if (t->type == GGML_TYPE_F32) {
              float v = 0.0F;
              std::memcpy(&v, raw.data() + off, sizeof(float));
              out.push_back(v);
            } else if (t->type == GGML_TYPE_F16) {
              ggml_fp16_t v = 0;
              std::memcpy(&v, raw.data() + off, sizeof(ggml_fp16_t));
              out.push_back(ggml_fp16_to_fp32(v));
            } else if (t->type == GGML_TYPE_BF16) {
              ggml_bf16_t v{};
              std::memcpy(&v, raw.data() + off, sizeof(ggml_bf16_t));
              out.push_back(ggml_bf16_to_fp32(v));
            } else {
              ADD_FAILURE() << "Unsupported float ggml_type: " << t->type;
              return out;
            }
          }
        }
      }
    }
    return out;
  }

  std::vector<std::int64_t> ReadTensorInts(const ggml_tensor* t) {
    const std::size_t span = static_cast<std::size_t>(t->ne[0] - 1) * t->nb[0] +
                             static_cast<std::size_t>(t->ne[1] - 1) * t->nb[1] +
                             static_cast<std::size_t>(t->ne[2] - 1) * t->nb[2] +
                             static_cast<std::size_t>(t->ne[3] - 1) * t->nb[3] + t->nb[0];
    const auto raw = DownloadBytes(reinterpret_cast<std::uintptr_t>(t->data), span);
    std::vector<std::int64_t> out;
    out.reserve(static_cast<std::size_t>(ggml_nelements(t)));
    for (std::int64_t i3 = 0; i3 < t->ne[3]; ++i3) {
      for (std::int64_t i2 = 0; i2 < t->ne[2]; ++i2) {
        for (std::int64_t i1 = 0; i1 < t->ne[1]; ++i1) {
          for (std::int64_t i0 = 0; i0 < t->ne[0]; ++i0) {
            const std::size_t off =
                static_cast<std::size_t>(i0) * t->nb[0] + static_cast<std::size_t>(i1) * t->nb[1] +
                static_cast<std::size_t>(i2) * t->nb[2] + static_cast<std::size_t>(i3) * t->nb[3];
            if (t->type == GGML_TYPE_I32) {
              std::int32_t v = 0;
              std::memcpy(&v, raw.data() + off, sizeof(std::int32_t));
              out.push_back(static_cast<std::int64_t>(v));
            } else {
              ADD_FAILURE() << "Unsupported int ggml_type: " << t->type;
              return out;
            }
          }
        }
      }
    }
    return out;
  }
};

class CatalogGpuEnvironment {
 public:
  static CatalogGpuEnvironment& Instance() {
    static CatalogGpuEnvironment env;
    return env;
  }

  DeviceExecution& execution() { return *execution_; }
  StreamId stream() const { return stream_; }
  kg::LaunchContext& launch() { return *launch_; }
  const ex::Registry& registry() const { return *registry_; }

  void Finish() {
    const auto fence = execution_->Record(stream_).value();
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    FenceState state = FenceState::kPending;
    while ((state = execution_->Query(fence).value()) == FenceState::kPending &&
           std::chrono::steady_clock::now() < deadline) {
      std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
    ASSERT_EQ(state, FenceState::kComplete);
    ASSERT_TRUE(execution_->Release(fence).has_value());
  }

 private:
  CatalogGpuEnvironment() {
    execution_ = std::move(providers::cuda::OpenDeviceExecution(0).value());
    stream_ = execution_->CreateStream().value();
    EXPECT_EQ(cudaMalloc(&pool_ptr_, kPoolBytes), cudaSuccess);
    EXPECT_EQ(cudaMalloc(&cublas_ptr_, kCublasBytes), cudaSuccess);
    const auto pool_base = reinterpret_cast<std::uintptr_t>(pool_ptr_);
    const auto cublas_base = reinterpret_cast<std::uintptr_t>(cublas_ptr_);
    cublas_ = std::move(kg::CublasHandle::Create(0, *execution_, stream_,
                                                 {.base = cublas_base, .size = Bytes(kCublasBytes)})
                            .value());
    launch_ = std::move(kg::LaunchContext::Create(0, *execution_, stream_,
                                                  {.base = pool_base, .size = Bytes(kPoolBytes)},
                                                  cublas_.get())
                            .value());
    registry_ = std::make_unique<ex::Registry>(ex::Registry::Create(kg::Implementations()).value());
  }

  ~CatalogGpuEnvironment() {
    launch_.reset();
    cublas_.reset();
    if (cublas_ptr_ != nullptr) (void)cudaFree(cublas_ptr_);
    if (pool_ptr_ != nullptr) (void)cudaFree(pool_ptr_);
    if (execution_) (void)execution_->DestroyStream(stream_);
  }

  std::unique_ptr<DeviceExecution> execution_;
  StreamId stream_{};
  void* pool_ptr_ = nullptr;
  void* cublas_ptr_ = nullptr;
  std::unique_ptr<kg::CublasHandle> cublas_;
  std::unique_ptr<kg::LaunchContext> launch_;
  std::unique_ptr<ex::Registry> registry_;
};

void RunCatalogTokenGpuCase(const CatalogTokenCase& tc) {
  auto& env = CatalogGpuEnvironment::Instance();
  DeviceBufferScope mem{env.execution(), env.stream(), {}};

  auto in = MakeInstance(tc.ref, tc.stage, tc.site, 0, 1);
  ASSERT_TRUE(in.has_value()) << tc.ref.kind << ": " << in.error();
  const auto shapes = in->def->shapes(*in);
  ASSERT_TRUE(shapes.has_value()) << tc.ref.kind << ": " << shapes.error();

  std::mt19937 rng(77);
  std::uniform_real_distribution<float> u(-1.0F, 1.0F);
  RefIO io{.instance = &*in};
  io.weights.resize(in->def->roles.size());
  io.weight_shapes.resize(in->def->roles.size());
  std::vector<std::vector<std::byte>> encoded(in->def->roles.size());
  for (const RoleShape& r : *shapes) {
    const auto& f = tc.formats[r.role];
    std::vector<float> v(ir::Elements(f.shape));
    const bool norm = f.shape.rank == 1;
    const auto range = tc.ranges.find(r.role);
    for (float& x : v) {
      x = range != tc.ranges.end() ? range->second.first + (range->second.second * (1.0F + u(rng)))
          : norm                   ? 1.0F + (0.25F * u(rng))
                                   : 0.3F * u(rng);
    }
    const auto parts = format::PartsOf(f);
    ASSERT_TRUE(parts.has_value());
    encoded[r.role].resize((*parts)[0].bytes);
    if (f.encoding == Encoding::kI32) {
      io.weights[r.role].resize(v.size());
      for (std::size_t i = 0; i < v.size(); ++i) {
        const auto id = static_cast<std::int32_t>((i * 5) % 8);
        std::memcpy(encoded[r.role].data() + (i * 4), &id, 4);
        io.weights[r.role][i] = static_cast<float>(id);
      }
      io.weight_shapes[r.role] = f.shape;
      continue;
    }
    const std::uint64_t matrices = format::Matrices(f);
    const std::size_t per = v.size() / matrices;
    io.weights[r.role].resize(v.size());
    for (std::uint64_t m = 0; m < matrices; ++m) {
      ASSERT_TRUE(ref::EncodeRows(f, m, 0, std::span(v).subspan(m * per, per), encoded[r.role]));
    }
    const ref::PartView part{.role = format::PartRole::kData, .bytes = encoded[r.role]};
    for (std::uint64_t m = 0; m < matrices; ++m) {
      ASSERT_TRUE(ref::DecodeRows(f, std::span(&part, 1), m, 0,
                                  f.shape.ne[1] * f.shape.ne[2] / matrices,
                                  std::span(io.weights[r.role]).subspan(m * per, per)));
    }
    io.weight_shapes[r.role] = f.shape;
  }

  const model::StateContract contract =
      in->def->state != nullptr ? in->def->state(*in) : model::StateContract{};
  io.state_cells.assign(tc.cells.begin(), tc.cells.end());

  auto graph = BuildComponent(*in, tc.shape, tc.formats, tc.cells, tc.inputs);
  ASSERT_TRUE(graph.has_value()) << tc.ref.kind << " (" << tc.label << "): " << graph.error();
  auto exec = ref::Executor::Create(graph->step.graph);
  ASSERT_TRUE(exec.has_value()) << tc.ref.kind << " (" << tc.label << "): " << exec.error().detail;

  std::map<ir::WeightRef, std::uint64_t> weight_addrs;
  for (const StepWeight& w : graph->step.weights) {
    const ref::PartView part{.role = format::PartRole::kData, .bytes = encoded[w.ref.role]};
    ASSERT_TRUE(exec->Bind(w.tensor, std::span(&part, 1)));
    const std::uint64_t addr = mem.Allocate(encoded[w.ref.role].size() + 4096);
    mem.Upload(addr, encoded[w.ref.role]);
    weight_addrs[w.ref] = addr;
  }

  std::map<ir::StateRef, std::pair<std::uint64_t, std::size_t>> state_addrs;
  std::map<std::uint8_t, std::vector<std::vector<double>>> ref_state;
  for (const StepState& st : graph->step.state) {
    const std::uint8_t slot = tc.shape.segments.at(st.ref.segment).slot;
    auto& start = ref_state[slot];
    if (start.empty()) start.resize(contract.tensors.size());
    std::vector<float> values(ir::Elements(graph->step.graph.tensor(st.tensor).shape()), 0.0F);
    if (tc.random_state) {
      for (float& x : values) x = 0.25F * u(rng);
    }
    ASSERT_TRUE(exec->BindValues(st.tensor, values));
    auto held = exec->Values(st.tensor);
    ASSERT_TRUE(held.has_value());
    start[st.ref.tensor].assign(held->begin(), held->end());

    const auto state_bytes = exec->Bytes(st.tensor);
    ASSERT_FALSE(state_bytes.empty());
    const std::uint64_t addr = mem.Allocate(state_bytes.size());
    mem.Upload(addr, state_bytes);
    state_addrs[st.ref] = {addr, state_bytes.size()};
  }
  for (const planner::Segment& s : tc.shape.segments) {
    io.segments.push_back(
        {.rows = s.rows, .past = s.past, .read_cells = s.read_cells, .outputs = s.outputs});
    auto& st = ref_state[s.slot];
    if (st.empty()) st.resize(contract.tensors.size());
    io.state.push_back(st);
  }

  for (const StepInput& input : graph->step.inputs) {
    const auto random = [&](std::vector<double>& into, float scale) {
      std::vector<float> f(ir::Elements(graph->step.graph.tensor(input.tensor).shape()));
      for (float& x : f) x = scale * u(rng);
      EXPECT_TRUE(exec->BindValues(input.tensor, f));
      into.assign(f.begin(), f.end());
    };
    switch (input.kind) {
      case InputKind::kStream:
        random(io.stream, 1.0F);
        continue;
      case InputKind::kResidual:
        random(io.residual, 1.0F);
        continue;
      case InputKind::kCarry:
        random(input.tensor == graph->carry2_in ? io.carry2 : io.carry, 1.0F);
        continue;
      case InputKind::kSide:
        random(io.side, 1.0F);
        continue;
      case InputKind::kFeatures:
        random(io.features, 1.0F);
        continue;
      case InputKind::kSideRows: {
        std::uniform_int_distribution<std::int64_t> pick(
            0, static_cast<std::int64_t>(tc.table_rows) - 1);
        std::vector<std::int64_t> rows(
            ir::Elements(graph->step.graph.tensor(input.tensor).shape()));
        for (auto& r : rows) r = pick(rng);
        ASSERT_TRUE(exec->BindInts(input.tensor, rows));
        io.side_rows = rows;
        continue;
      }
      default:
        break;
    }
    auto values = InputValues(input, tc.shape, tc.tokens);
    ASSERT_TRUE(values.has_value()) << values.error();
    ASSERT_TRUE(exec->BindInts(input.tensor, *values));
  }

  // Capture initial input bytes before exec->Run() mutates state in place.
  std::map<std::uint32_t, std::vector<std::byte>> input_bytes;
  for (const StepInput& input : graph->step.inputs) {
    const auto bytes = exec->Bytes(input.tensor);
    ASSERT_FALSE(bytes.empty());
    input_bytes[input.tensor.v] = std::vector<std::byte>(bytes.begin(), bytes.end());
  }

  auto ran = exec->Run();
  ASSERT_TRUE(ran.has_value()) << tc.ref.kind << " (" << tc.label << "): " << ran.error().detail;
  // Also capture any host-computed mask inputs produced during exec->Run().
  for (std::uint32_t tid = 0; tid < graph->step.graph.tensor_count(); ++tid) {
    if (input_bytes.contains(tid)) continue;
    const auto bytes = exec->Bytes(ir::TensorId{tid});
    if (!bytes.empty()) {
      input_bytes[tid] = std::vector<std::byte>(bytes.begin(), bytes.end());
    }
  }

  io.tokens = tc.tokens;
  io.greedy = graph->step.greedy.valid();
  ASSERT_NE(in->def->reference, nullptr) << tc.ref.kind;
  auto r = in->def->reference(io);
  ASSERT_TRUE(r.has_value()) << tc.ref.kind << " (" << tc.label << "): " << r.error();

  StepGraph& step = graph->step;
  std::vector<ir::TensorId> staged;
  for (StepInput& input : step.inputs) {
    if (input.kind == InputKind::kStream || input.kind == InputKind::kResidual ||
        input.kind == InputKind::kCarry || input.kind == InputKind::kSide) {
      input.kind = InputKind::kFeatures;
    }
    if (input.kind != InputKind::kCellPositions) {
      staged.push_back(input.tensor);
    }
  }

  auto plan = pl::Plan(step.graph.Clone(), env.registry(), {.device_class = "sm_121", .arch = 121},
                       pl::Policy{},
                       {.staged = staged,
                        .written = kg::WrittenBytes,
                        .within = kg::WithinRoot,
                        .shape = &tc.shape,
                        .activation_alignment = 256,
                        .alignment_staged = staged});
  ASSERT_TRUE(plan.has_value()) << tc.ref.kind << " (" << tc.label << "): " << plan.error().detail;

  kn::Addresses measure_addrs;
  measure_addrs.weight = [&](const ir::WeightRef& ref, format::PartRole) -> std::uint64_t {
    const auto it = weight_addrs.find(ref);
    return it != weight_addrs.end() ? it->second : 0;
  };
  measure_addrs.state = [&](const ir::StateRef& ref) -> std::uint64_t {
    const auto it = state_addrs.find(ref);
    return it != state_addrs.end() ? it->second.first : 0;
  };
  measure_addrs.activations = 0;
  measure_addrs.activation_bytes = 0;
  measure_addrs.activation_alignment = 256;
  measure_addrs.state_alignment = [](const ir::StateRef&) { return 256ULL; };
  measure_addrs.weight_alignment = [](const ir::WeightRef&, format::PartRole) { return 256ULL; };

  const auto choices = kg::DeviceChoicesOf(env.launch());
  auto measured = kg::LowerStep(
      {*plan, step, tc.shape, measure_addrs, static_cast<std::uint32_t>(tc.stage.context)},
      env.registry(), choices);
  ASSERT_TRUE(measured.has_value())
      << tc.ref.kind << " (" << tc.label << "): " << measured.error().detail;

  const std::uint64_t act_bytes =
      std::max<std::uint64_t>(std::max(plan->placement.extent, measured->extent), 256);
  const std::uint64_t act_base = mem.Allocate(act_bytes);
  kn::Addresses bound_addrs = measure_addrs;
  bound_addrs.activations = act_base;
  bound_addrs.activation_bytes = act_bytes;

  auto lowered = kg::LowerStep(
      {*plan, step, tc.shape, bound_addrs, static_cast<std::uint32_t>(tc.stage.context)},
      env.registry(), choices);
  ASSERT_TRUE(lowered.has_value())
      << tc.ref.kind << " (" << tc.label << "): " << lowered.error().detail;

  for (const kn::InputSlot& slot : lowered->inputs) {
    const auto it = input_bytes.find(slot.tensor.v);
    ASSERT_NE(it, input_bytes.end())
        << tc.ref.kind << " (" << tc.label << ") missing input tensor " << slot.tensor.v;
    ASSERT_EQ(it->second.size(), slot.bytes)
        << tc.ref.kind << " (" << tc.label << ") input size mismatch";
    mem.Upload(slot.device_address, it->second);
  }

  auto bound = kg::BoundGraph::Bind(env.registry(), lowered->plan);
  ASSERT_TRUE(bound.has_value()) << tc.ref.kind << " (" << tc.label
                                 << "): " << bound.error().detail;
  auto gpu_ran = bound->Run(env.launch());
  ASSERT_TRUE(gpu_ran.has_value())
      << tc.ref.kind << " (" << tc.label << "): " << gpu_ran.error().detail;
  env.Finish();

  const auto find_kept = [&](ir::TensorId t) -> const ggml_tensor* {
    if (!t.valid()) return nullptr;
    const auto& outs = graph->step.graph.outputs();
    for (std::size_t i = 0; i < outs.size(); ++i) {
      if (outs[i] == t) return lowered->keep.at(i);
    }
    return nullptr;
  };

  const auto check_gpu = [&](ir::TensorId t, std::span<const double> want, std::string_view what) {
    if (!t.valid()) return;
    const ggml_tensor* gt = find_kept(t);
    ASSERT_NE(gt, nullptr) << tc.ref.kind << " (" << tc.label << ") " << what;
    const std::vector<float> got = mem.ReadTensorValues(gt);
    EXPECT_LE(Gap(got, want), tc.tolerance) << tc.ref.kind << " (" << tc.label << ") " << what;
  };

  if (graph->step.logits.valid()) {
    check_gpu(graph->step.logits, io.logits, "logits");
  } else if (tc.ref.kind == "side.hash") {
    const ggml_tensor* gt = find_kept(graph->stream_out);
    ASSERT_NE(gt, nullptr);
    const std::vector<std::int64_t> got = mem.ReadTensorInts(gt);
    EXPECT_EQ(got, io.side_ids) << tc.ref.kind << " (" << tc.label << ")";
  } else {
    check_gpu(graph->stream_out, io.stream, "stream");
    check_gpu(graph->residual_out, io.residual, "residual");
    check_gpu(graph->carry_out, io.carry, "carry");
    check_gpu(graph->carry2_out, io.carry2, "carry2");
  }
  check_gpu(graph->step.features, io.features_out, "features");

  for (const StepState& st : graph->step.state) {
    if (!st.written) continue;
    const auto& [addr, nbytes] = state_addrs.at(st.ref);
    const auto raw = mem.DownloadBytes(addr, nbytes);
    const ref::PartView part{.role = format::PartRole::kData, .bytes = raw};
    ASSERT_TRUE(exec->Bind(st.tensor, std::span(&part, 1)));
    auto values = exec->Values(st.tensor);
    ASSERT_TRUE(values.has_value());
    std::span<const float> got_span(*values);
    std::span<const double> want_span(io.state[st.ref.segment][st.ref.tensor]);
    if (tc.ref.kind == "attention.qsa" && st.ref.tensor == 3) {
      // llmp.qsa.pool writes only the blocks completed by this step (1 block * 64 dims).
      got_span = got_span.first(64);
      want_span = want_span.first(64);
    }
    EXPECT_LE(Gap(got_span, want_span), tc.state_tolerance)
        << tc.ref.kind << " (" << tc.label << ") state " << st.ref.tensor;
  }
}

void RunCatalogSpatialGpuCase(const CatalogSpatialCase& sc) {
  auto& env = CatalogGpuEnvironment::Instance();
  DeviceBufferScope mem{env.execution(), env.stream(), {}};

  const spec::ModelSpec model = MakeSingleSpatialSpec(sc);
  const std::array<std::span<const BoundTensor>, 1> tables{sc.tensors};
  const std::array<SpatialOwner, 1> owners{sc.owner};
  auto graph = BuildSpatialStep(model, tables, owners, UnboundedSpatialPayment(),
                                {.links = sc.links, .inputs = sc.inputs});
  ASSERT_TRUE(graph.has_value()) << sc.ref.kind << " (" << sc.label << "): " << graph.error();

  auto exec = ref::Executor::Create(graph->graph());
  ASSERT_TRUE(exec.has_value()) << sc.ref.kind << " (" << sc.label << "): " << exec.error().detail;

  std::mt19937 rng(99);
  std::uniform_real_distribution<float> u(-0.6F, 0.6F);

  std::map<ir::WeightRef, std::uint64_t> weight_addrs;
  std::vector<std::vector<float>> weights(graph->weights().size());
  std::array<std::span<const float>, 2> roles{};
  for (std::size_t i = 0; i < weights.size(); ++i) {
    const auto& w = graph->weights()[i];
    const auto count = ir::Elements(graph->graph().tensor(w.tensor).shape());
    weights[i].resize(static_cast<std::size_t>(count));
    for (float& x : weights[i]) x = u(rng);
    ASSERT_TRUE(exec->BindValues(w.tensor, weights[i]));
    roles.at(w.ref.role) = weights[i];
    const auto wbytes = exec->Bytes(w.tensor);
    ASSERT_FALSE(wbytes.empty());
    const std::uint64_t addr = mem.Allocate(wbytes.size());
    mem.Upload(addr, wbytes);
    weight_addrs[w.ref] = addr;
  }

  const auto in_count = ir::Elements(graph->graph().tensor(graph->graph().inputs()[0]).shape());
  std::vector<float> input_f(static_cast<std::size_t>(in_count));
  for (float& x : input_f) x = u(rng);
  ASSERT_TRUE(exec->BindValues(graph->graph().inputs()[0], input_f));
  std::vector<double> input_d(input_f.begin(), input_f.end());

  std::vector<double> aux_d;
  std::optional<SpatialShape> aux_shape;
  if (!sc.inputs.empty()) {
    ASSERT_GE(graph->graph().inputs().size(), 2U);
    const auto aux_id = graph->graph().inputs()[1];
    const auto aux_count = ir::Elements(graph->graph().tensor(aux_id).shape());
    std::vector<float> aux_f(static_cast<std::size_t>(aux_count), 0.0F);
    for (float& x : aux_f) x = u(rng);
    ASSERT_TRUE(exec->BindValues(aux_id, aux_f));
    aux_d.assign(aux_f.begin(), aux_f.end());
    aux_shape = sc.inputs[0].shape;
  }

  std::map<std::uint32_t, std::vector<std::byte>> input_bytes;
  for (const ir::TensorId in_id : graph->graph().inputs()) {
    const auto bytes = exec->Bytes(in_id);
    ASSERT_FALSE(bytes.empty());
    input_bytes[in_id.v] = std::vector<std::byte>(bytes.begin(), bytes.end());
  }

  auto ran = exec->Run();
  ASSERT_TRUE(ran.has_value()) << sc.ref.kind << " (" << sc.label << "): " << ran.error().detail;

  auto instances = InstancesOf(model, 0);
  ASSERT_TRUE(instances.has_value()) << instances.error();
  const Instance& instance = instances->layers[0][0];
  ASSERT_NE(instance.def->spatial, nullptr);
  ASSERT_NE(instance.def->spatial->reference, nullptr);

  SpatialRefIO oracle{
      .instance = instance,
      .shape = sc.owner.shape,
      .input = input_d,
      .kernel = roles[0],
      .bias = roles[1],
      .output = {},
      .auxiliary = aux_d,
      .auxiliary_shape = aux_shape,
  };
  auto ref_st = RunSpatialReference(oracle);
  ASSERT_TRUE(ref_st.has_value()) << sc.ref.kind << " (" << sc.label << "): " << ref_st.error();

  const std::vector<ir::TensorId> staged(graph->graph().inputs().begin(),
                                         graph->graph().inputs().end());
  auto plan = pl::Plan(graph->graph().Clone(), env.registry(),
                       {.device_class = "sm_121", .arch = 121}, pl::Policy{},
                       {.staged = staged,
                        .written = kg::WrittenBytes,
                        .within = kg::WithinRoot,
                        .activation_alignment = 256,
                        .alignment_staged = staged});
  ASSERT_TRUE(plan.has_value()) << sc.ref.kind << " (" << sc.label << "): " << plan.error().detail;

  kn::Addresses measure_addrs;
  measure_addrs.weight = [&](const ir::WeightRef& ref, format::PartRole) -> std::uint64_t {
    const auto it = weight_addrs.find(ref);
    return it != weight_addrs.end() ? it->second : 0;
  };
  measure_addrs.activations = 0;
  measure_addrs.activation_bytes = 0;
  measure_addrs.activation_alignment = 256;
  measure_addrs.weight_alignment = [](const ir::WeightRef&, format::PartRole) { return 256ULL; };

  const auto choices = kg::DeviceChoicesOf(env.launch());
  const GraphBuildIO step_io{
      .graph = graph->graph(),
      .inputs = graph->graph().inputs(),
      .outputs = graph->graph().outputs(),
      .weights = graph->weights(),
  };
  auto measured = kg::LowerStep({*plan, step_io, measure_addrs}, env.registry(), choices);
  ASSERT_TRUE(measured.has_value())
      << sc.ref.kind << " (" << sc.label << "): " << measured.error().detail;

  const std::uint64_t act_bytes =
      std::max<std::uint64_t>(std::max(plan->placement.extent, measured->extent), 256);
  const std::uint64_t act_base = mem.Allocate(act_bytes);
  kn::Addresses bound_addrs = measure_addrs;
  bound_addrs.activations = act_base;
  bound_addrs.activation_bytes = act_bytes;

  auto lowered = kg::LowerStep({*plan, step_io, bound_addrs}, env.registry(), choices);
  ASSERT_TRUE(lowered.has_value())
      << sc.ref.kind << " (" << sc.label << "): " << lowered.error().detail;

  for (const kn::InputSlot& slot : lowered->inputs) {
    const auto it = input_bytes.find(slot.tensor.v);
    ASSERT_NE(it, input_bytes.end());
    ASSERT_EQ(it->second.size(), slot.bytes);
    mem.Upload(slot.device_address, it->second);
  }

  if (!lowered->plan.steps.empty()) {
    auto bound = kg::BoundGraph::Bind(env.registry(), lowered->plan);
    ASSERT_TRUE(bound.has_value())
        << sc.ref.kind << " (" << sc.label << "): " << bound.error().detail;
    auto gpu_ran = bound->Run(env.launch());
    ASSERT_TRUE(gpu_ran.has_value())
        << sc.ref.kind << " (" << sc.label << "): " << gpu_ran.error().detail;
    env.Finish();
  }

  ASSERT_FALSE(lowered->keep.empty());
  const std::vector<float> got = mem.ReadTensorValues(lowered->keep[0]);
  EXPECT_LE(Gap(got, oracle.output), sc.tolerance) << sc.ref.kind << " (" << sc.label << ")";
}

TEST(ComponentsCatalogGpuTest, EveryCatalogComponentAndFormatMatchesCpuReferenceOnGpu) {
  const auto token_recipes = CatalogTokenRecipes();
  const auto spatial_recipes = CatalogSpatialRecipes();

  std::set<std::pair<std::string_view, std::uint32_t>> covered;
  for (const CatalogTokenRecipe& recipe : token_recipes) {
    ASSERT_FALSE(recipe.cases.empty()) << recipe.kind;
    covered.emplace(recipe.kind, recipe.version);
    for (const CatalogTokenCase& tc : recipe.cases) {
      SCOPED_TRACE(std::string(tc.ref.kind) + "/" + tc.label);
      RunCatalogTokenGpuCase(tc);
    }
  }
  for (const CatalogSpatialRecipe& recipe : spatial_recipes) {
    ASSERT_FALSE(recipe.cases.empty()) << recipe.kind;
    covered.emplace(recipe.kind, recipe.version);
    for (const CatalogSpatialCase& sc : recipe.cases) {
      SCOPED_TRACE(std::string(sc.ref.kind) + "/" + sc.label);
      RunCatalogSpatialGpuCase(sc);
    }
  }

  for (const ComponentDef* def : Catalog()) {
    ASSERT_NE(def, nullptr);
    EXPECT_TRUE(covered.contains({def->kind, def->version}))
        << "Component " << def->kind << " v" << def->version
        << " is missing from ComponentsCatalogGpuTest";
  }
}

}  // namespace
}  // namespace llmp::components
