// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// Decode graphs (D-090; kernels/ggml/launch.h Capture and Launch) on a GB10,
// over a small GGML plan on the paged node (tests/support/paged_node.h):
// the token's embedding row (get_rows), the unfused RMSNorm-mul, a Q8_0
// product (MMVQ, with pool scratch) and an F16 one (MMVF), their sum, RoPE
// and the write of the result into an F16 cache (set_rows). The token, its
// position and its cache cell vary per step as input data, copied from
// pinned staging inside the graph, as the DeepSeek runner's are.
// - A captured graph replays bit for bit what launching the plan step by
//   step computes, and the cache it writes is the same; so do the plan's
//   two products on concurrent lanes (graph_plan.h AssignLanes).
// - After full swaps A→B→A, with and without the handoff (A's cache written
//   back and restored, its weights paged in again into backing that is not
//   what they had), A's places are the pinned ones, and its graph, never
//   captured again, still replays bit for bit what a control that was never
//   swapped computes.
// - A capture that cannot be made is refused cleanly: the context is neither
//   faulted nor left capturing and runs as before; a graph replays only on
//   the context that captured it; a faulted context captures nothing.
// - RE-029: a graph of 1,500 kernels takes one entry of its stream's queue,
//   not 1,500 (the probe reports how many replays the stream holds).

#include <cuda.h>
#include <cuda_runtime.h>
#include <fcntl.h>
#include <gtest/gtest.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <expected>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <print>
#include <stop_token>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include "base/bytes.h"
#include "catalog/catalog.h"
#include "execution/registry.h"
#include "ggml.h"
#include "kernels/ggml/executor.h"
#include "kernels/ggml/graph_plan.h"
#include "kernels/ggml/implementations.h"
#include "kernels/ggml/launch.h"
#include "kernels/ggml/ops.h"
#include "kernels/ggml/tensors.h"
#include "paged_node.h"
#include "paged_programs.h"
#include "scheduler/commands.h"
#include "scheduler/scheduler.h"

// GGML's error hook (ggml_support.cu), to leave an error pending.
void ggml_cuda_error(const char* stmt, const char* func, const char* file, int line,
                     const char* msg);

namespace {

namespace kg = jitllm::kernels::ggml;
namespace sc = jitllm::scheduler;
namespace ts = jitllm::test_support;
using jitllm::base::Bytes;
using jitllm::catalog::ExtentId;
using jitllm::catalog::MemoryClass;
using jitllm::catalog::Recovery;

// Binding checks descriptors only; these aligned, distinct addresses are never
// dereferenced or submitted. Repeated declarations must not cache their operands.
class GgmlBindingTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto created = kg::TensorArena::Create(24);
    ASSERT_TRUE(created.has_value());
    arena_.emplace(std::move(*created));
  }
  ggml_tensor* Bind(ggml_tensor* tensor) {
    kg::TensorArena::Bind(tensor, next_);
    next_ += 65536;
    return tensor;
  }
  std::optional<kg::TensorArena> arena_;
  std::uint64_t next_ = 0x1000000000ULL;
};

TEST_F(GgmlBindingTest, RepeatedMixedWrappersCheckEveryOccurrenceAndEveryBind) {
  auto registry = jitllm::execution::Registry::Create(kg::Implementations());
  ASSERT_TRUE(registry.has_value());
  auto* c = arena_->context();
  kg::GraphPlan plan;
  ggml_tensor *last_mul = nullptr, *last_add = nullptr;
  for (int i = 0; i < 2; ++i) {
    auto* x = Bind(ggml_new_tensor_2d(c, GGML_TYPE_F32, 32, 2));
    auto* w = Bind(ggml_new_tensor_1d(c, GGML_TYPE_F32, 32));
    auto* norm = Bind(ggml_rms_norm(c, x, 1e-6f));
    auto* mul = Bind(ggml_mul(c, norm, w));
    auto* add = Bind(ggml_add(c, mul, x));
    plan.steps.push_back({.operation = jitllm::execution::Operation::kRmsNormMul,
                          .implementation = kg::kRmsNormMulFused,
                          .nodes = {norm, mul}});
    plan.steps.push_back({.operation = jitllm::execution::Operation::kAdd,
                          .implementation = kg::kAddName,
                          .nodes = {add}});
    last_mul = mul;
    last_add = add;
  }
  ASSERT_TRUE(kg::BoundGraph::Bind(*registry, plan).has_value());
  const auto rejected_at = [&](std::string_view step) {
    auto bound = kg::BoundGraph::Bind(*registry, plan);
    ASSERT_FALSE(bound.has_value());
    EXPECT_EQ(bound.error().error, kg::KernelError::kRejected);
    EXPECT_NE(bound.error().detail.find(step), std::string::npos);
  };
  last_mul->type = GGML_TYPE_I32;
  rejected_at("step 2");
  last_mul->type = GGML_TYPE_F32;
  plan.steps[2].nodes.pop_back();
  rejected_at("step 2: RMSNorm-mul takes two nodes");
  plan.steps[2].nodes.push_back(last_mul);
  last_add->type = GGML_TYPE_I32;
  rejected_at("step 3");
  last_add->type = GGML_TYPE_F32;
  plan.steps[3].nodes.clear();
  rejected_at("step 3");
  plan.steps[3].nodes.push_back(last_add);
  EXPECT_TRUE(kg::BoundGraph::Bind(*registry, plan).has_value());

  for (const auto name : {kg::kRmsNormMulFused, kg::kAddName}) {
    auto declarations = kg::Implementations();
    for (auto& declaration : declarations) {
      if (declaration.name == name) declaration.revision = "stale binding declaration";
    }
    auto stale = jitllm::execution::Registry::Create(std::move(declarations));
    ASSERT_TRUE(stale.has_value());
    auto rejected = kg::BoundGraph::Bind(*stale, plan);
    ASSERT_FALSE(rejected.has_value());
    EXPECT_EQ(rejected.error().error, kg::KernelError::kRejected);
    EXPECT_TRUE(kg::BoundGraph::Bind(*registry, plan).has_value());
  }
}

TEST_F(GgmlBindingTest, RepeatedCublasWrapperStillChecksTheLaterLaneAndOperands) {
  auto registry = jitllm::execution::Registry::Create(kg::Implementations());
  ASSERT_TRUE(registry.has_value());
  auto* c = arena_->context();
  kg::GraphPlan plan;
  ggml_tensor* last = nullptr;
  for (int i = 0; i < 2; ++i) {
    auto* w = Bind(ggml_new_tensor_2d(c, GGML_TYPE_F32, 32, 16));
    auto* x = Bind(ggml_new_tensor_2d(c, GGML_TYPE_F32, 32, 2));
    last = Bind(ggml_mul_mat(c, w, x));
    plan.steps.push_back({.operation = jitllm::execution::Operation::kMatMul,
                          .implementation = kg::kMulMatCublas,
                          .nodes = {last}});
  }
  ASSERT_TRUE(kg::BoundGraph::Bind(*registry, plan).has_value());
  plan.steps[1].lane = 1;
  auto lane = kg::BoundGraph::Bind(*registry, plan);
  ASSERT_FALSE(lane.has_value());
  EXPECT_EQ(lane.error().error, kg::KernelError::kRejected);
  EXPECT_NE(lane.error().detail.find("step 1"), std::string::npos);
  EXPECT_NE(lane.error().detail.find("borrows cuBLAS"), std::string::npos);
  plan.steps[1].lane = 0;
  last->type = GGML_TYPE_I32;
  auto operands = kg::BoundGraph::Bind(*registry, plan);
  ASSERT_FALSE(operands.has_value());
  EXPECT_EQ(operands.error().error, kg::KernelError::kRejected);
  EXPECT_NE(operands.error().detail.find("step 1"), std::string::npos);
  last->type = GGML_TYPE_F32;
  EXPECT_TRUE(kg::BoundGraph::Bind(*registry, plan).has_value());
}

constexpr std::uint64_t kExtent = ts::kPagedExtent;
constexpr std::int64_t kWidth = 512;  // quantized rows are whole 512-element steps
constexpr std::int64_t kVocab = 64;
constexpr std::int64_t kOut = 512;
constexpr std::int64_t kHead = 64;
constexpr std::int64_t kCells = 16;
constexpr int kSteps = 12;  // at most kCells: each step writes its own cell
constexpr float kEps = 1e-6f;

// The weights' file: the table, the norm and the Q8_0 matrix in its first
// extent, the F16 matrix in its second.
constexpr std::uint64_t kTableAt = 0;      // F32 [kWidth, kVocab]
constexpr std::uint64_t kNormAt = 131072;  // F32 [kWidth]
constexpr std::uint64_t kQ8At = 135168;    // Q8_0 [kWidth, kOut]
constexpr std::uint64_t kF16At = kExtent;  // F16 [kWidth, kOut]
constexpr std::uint64_t kWeightBytes = 2 * kExtent;
// The activations region: the three inputs, then the computed tensors.
constexpr std::uint64_t kComputedAt = 1024;

void* Pointer(std::uint64_t address) {
  return reinterpret_cast<void*>(address);  // NOLINT(performance-no-int-to-ptr)
}
std::uint64_t Address(const void* pointer) { return reinterpret_cast<std::uintptr_t>(pointer); }

// Deterministic values in [-1, 1).
float Value(std::uint64_t seed, std::uint64_t i) {
  std::uint64_t x = (seed * 0x9E3779B97F4A7C15ULL) ^ (i + 0x632BE59BD9B4E019ULL);
  x ^= x >> 31;
  x *= 0xBF58476D1CE4E5B9ULL;
  x ^= x >> 29;
  return static_cast<float>(static_cast<double>(x >> 40) / static_cast<double>(1ULL << 23)) - 1.0f;
}

std::filesystem::path Scratch() {
  const char* scratch = std::getenv("JITLLM_TEST_SCRATCH");  // NOLINT(concurrency-mt-unsafe)
  std::filesystem::path directory = scratch != nullptr
                                        ? std::filesystem::path(scratch)
                                        : std::filesystem::path(::testing::TempDir());
  std::filesystem::create_directories(directory);
  return directory;
}

// An unnamed direct-I/O file holding `bytes` (a multiple of 4 KiB).
int DirectFile(const std::vector<std::byte>& bytes) {
  const int fd = ::open(Scratch().c_str(), O_TMPFILE | O_RDWR | O_DIRECT | O_CLOEXEC, 0600);
  if (fd < 0 || bytes.empty()) {
    return fd;
  }
  auto* aligned =
      static_cast<std::byte*>(std::aligned_alloc(4096, bytes.size()));  // NOLINT(*-no-malloc)
  std::memcpy(aligned, bytes.data(), bytes.size());
  const ssize_t written = ::pwrite(fd, aligned, bytes.size(), 0);
  std::free(aligned);  // NOLINT(cppcoreguidelines-no-malloc)
  if (std::cmp_not_equal(written, bytes.size())) {
    (void)::close(fd);
    return -1;
  }
  return fd;
}

sc::PageSource Landed(int fd, std::uint64_t offset, std::uint64_t address,
                      jitllm::providers::ReservationId reservation, std::uint64_t at,
                      std::size_t allocation_class, bool write_back) {
  return sc::PageSource{.read = {.fd = fd, .offset = offset, .memory = nullptr, .length = kExtent},
                        .landed = true,
                        .destination = address,
                        .backing = sc::BackingPlace{.reservation = reservation,
                                                    .offset = Bytes(at),
                                                    .size = Bytes(kExtent),
                                                    .allocation_class = allocation_class},
                        .write_back = write_back};
}

enum class Mode : std::uint8_t { kEager, kCapture, kReplay };

// Step 0 launch by launch, step 1 captured, the later steps replayed.
Mode ModeOf(int k) {
  if (k == 0) {
    return Mode::kEager;
  }
  return k == 1 ? Mode::kCapture : Mode::kReplay;
}

// What a capture or a replay returned, kept past the job.
struct Outcome {
  bool ran = false;   // the call was made
  bool made = false;  // it succeeded
  kg::KernelFailure failure{};
};
template <typename T>
Outcome OutcomeOf(const std::expected<T, kg::KernelFailure>& result) {
  Outcome outcome{.ran = true, .made = result.has_value()};
  if (!result) {
    outcome.failure = result.error();
  }
  return outcome;
}

// Model A: the plan above over two weight extents and a one-extent cache,
// both at pinned places (D-090), on stream 0.
class GraphModel final : public ts::PagedModel {
 public:
  explicit GraphModel(ts::PagedNode& node) : node_(node) {}

  void Setup() {
    std::vector<std::byte> file(kWeightBytes);
    const auto put = [&](std::uint64_t at, const void* data, std::size_t bytes) {
      std::memcpy(file.data() + at, data, bytes);
    };
    std::vector<float> table(static_cast<std::size_t>(kWidth * kVocab));
    for (std::size_t i = 0; i < table.size(); ++i) {
      table[i] = Value(1, i) * 3.0f;
    }
    put(kTableAt, table.data(), table.size() * sizeof(float));
    std::vector<float> norm(static_cast<std::size_t>(kWidth));
    for (std::size_t i = 0; i < norm.size(); ++i) {
      norm[i] = Value(2, i) + 1.0f;
    }
    put(kNormAt, norm.data(), norm.size() * sizeof(float));
    std::vector<float> matrix(static_cast<std::size_t>(kWidth * kOut));
    for (std::size_t i = 0; i < matrix.size(); ++i) {
      matrix[i] = Value(3, i) * 0.1f;
    }
    std::vector<std::byte> q8(ggml_row_size(GGML_TYPE_Q8_0, kWidth) * kOut);
    ASSERT_EQ(
        ggml_quantize_chunk(GGML_TYPE_Q8_0, matrix.data(), q8.data(), 0, kOut, kWidth, nullptr),
        q8.size());
    ASSERT_LE(kQ8At + q8.size(), kExtent);
    put(kQ8At, q8.data(), q8.size());
    std::vector<ggml_fp16_t> f16(static_cast<std::size_t>(kWidth * kOut));
    for (std::size_t i = 0; i < f16.size(); ++i) {
      f16[i] = ggml_fp32_to_fp16(Value(4, i) * 0.1f);
    }
    put(kF16At, f16.data(), f16.size() * sizeof(ggml_fp16_t));
    fd_ = DirectFile(file);
    ASSERT_GE(fd_, 0);
    spill_ = DirectFile({});
    ASSERT_GE(spill_, 0);

    auto& memory = node_.memory();
    place_ = memory.Reserve(Bytes(kWeightBytes)).value();
    base_ = memory.RangeOf(place_).value().base;
    for (std::uint32_t i = 0; i < kWeightBytes / kExtent; ++i) {
      weights_.push_back(node_.catalog()
                             .AddExtent({.domain = node_.domain(),
                                         .memory_class = MemoryClass::kWeights,
                                         .recovery = Recovery::kFromArtifact,
                                         .size = Bytes(kExtent),
                                         .content = {.artifact = {7}, .group = 0, .chunk = i}})
                             .value());
    }
    ASSERT_TRUE(node_
                    .MapResident(cache_, "the cache", kOut * kCells * sizeof(ggml_fp16_t),
                                 jitllm::providers::BackingKind::kDevice, MemoryClass::kLiveState,
                                 Recovery::kPreserve, 0)
                    .has_value());
    auto staging = node_.Pinned(std::uint64_t{64} * 1024, 0, staging_);
    ASSERT_TRUE(staging.has_value());
    staging_bytes_ = static_cast<std::byte*>(*staging);
  }

  // Sources, the places pinned, the closure; after Start, before Run.
  void Register() {
    for (std::size_t i = 0; i < weights_.size(); ++i) {
      sources_.push_back(Landed(fd_, i * kExtent, base_ + (i * kExtent), place_, i * kExtent,
                                node_.device_class(), false));
      ASSERT_TRUE(node_.scheduler().SetSource(weights_[i], sources_.back()).has_value());
    }
    sources_.push_back(Landed(spill_, 0, cache_.base, cache_.reservation, 0, node_.device_class(),
                              /*write_back=*/true));
    ASSERT_TRUE(node_.scheduler().SetSource(cache_.extents.at(0), sources_.back()).has_value());
    cache_.backings.clear();  // the VMM lane's from now on
    ASSERT_TRUE(node_.scheduler().PinPlaces(managed_extents()).has_value());
    std::vector<ExtentId> all = managed_extents();
    for (const ts::Mapped* mapped : {&node_.activations(), &node_.pool()}) {
      all.insert(all.end(), mapped->extents.begin(), mapped->extents.end());
    }
    all.insert(all.end(), staging_.begin(), staging_.end());
    closure_ = node_.catalog().ClosureOfExtents(all).value();
    fence_ = node_.catalog().ClosureOfExtents(staging_).value();
  }

  // The launch context, the graph, its plan, bound; after Run.
  void Bind() {
    const std::uint64_t pool = node_.pool().base;
    const std::uint64_t pool_bytes = node_.pool().bytes;
    ASSERT_TRUE(node_
                    .Job(
                        fence_,
                        [&](jitllm::providers::NativeStream) {
                          auto launch =
                              kg::LaunchContext::Create(0, node_.execution(), node_.stream(0),
                                                        {.base = pool, .size = Bytes(pool_bytes)});
                          if (!launch) {
                            return sc::JobResult::kNotStarted;
                          }
                          launch_ = std::move(*launch);
                          return sc::JobResult::kQueued;
                        },
                        "making the launch context", 0)
                    .has_value());
    ASSERT_NE(launch_, nullptr);
    auto registry = jitllm::execution::Registry::Create(kg::Implementations());
    ASSERT_TRUE(registry.has_value());
    registry_ = std::make_unique<jitllm::execution::Registry>(std::move(*registry));
    auto arena = kg::TensorArena::Create(32);
    ASSERT_TRUE(arena.has_value());
    arena_.emplace(std::move(*arena));
    ggml_context* c = arena_->context();
    const std::uint64_t act = node_.activations().base;
    ids_ = ggml_new_tensor_1d(c, GGML_TYPE_I32, 1);
    pos_ = ggml_new_tensor_1d(c, GGML_TYPE_I32, 1);
    cell_ = ggml_new_tensor_1d(c, GGML_TYPE_I64, 1);
    kg::TensorArena::Bind(ids_, act);
    kg::TensorArena::Bind(pos_, act + 256);
    kg::TensorArena::Bind(cell_, act + 512);
    ggml_tensor* table = ggml_new_tensor_2d(c, GGML_TYPE_F32, kWidth, kVocab);
    ggml_tensor* norm = ggml_new_tensor_1d(c, GGML_TYPE_F32, kWidth);
    ggml_tensor* wq = ggml_new_tensor_2d(c, GGML_TYPE_Q8_0, kWidth, kOut);
    ggml_tensor* wf = ggml_new_tensor_2d(c, GGML_TYPE_F16, kWidth, kOut);
    ggml_tensor* cache = ggml_new_tensor_2d(c, GGML_TYPE_F16, kOut, kCells);
    kg::TensorArena::Bind(table, base_ + kTableAt);
    kg::TensorArena::Bind(norm, base_ + kNormAt);
    kg::TensorArena::Bind(wq, base_ + kQ8At);
    kg::TensorArena::Bind(wf, base_ + kF16At);
    kg::TensorArena::Bind(cache, cache_.base);
    ggml_tensor* x = ggml_get_rows(c, table, ids_);
    ggml_tensor* n = ggml_rms_norm(c, x, kEps);
    ggml_tensor* s = ggml_mul(c, n, norm);
    ggml_tensor* q = ggml_mul_mat(c, wq, s);
    ggml_tensor* f = ggml_mul_mat(c, wf, s);
    ggml_tensor* a = ggml_add(c, q, f);
    // The sum against every cell of the cache: what earlier steps wrote,
    // restored after a swap, is read.
    scores_ = ggml_mul_mat(c, cache, a);
    ggml_tensor* heads = ggml_reshape_3d(c, a, kHead, kOut / kHead, 1);
    out_ = ggml_rope(c, heads, pos_, static_cast<int>(kHead), GGML_ROPE_TYPE_NEOX);
    ggml_tensor* rows = ggml_reshape_2d(c, out_, kOut, 1);
    ggml_tensor* write = ggml_set_rows(c, cache, rows, cell_);
    nodes_ = {x, n, s, q, f, a, scores_, heads, out_, rows, write};
    q_ = q;
    f_ = f;
    sum_ = a;
    kg::BindDistinct(nodes_, act + kComputedAt);
    ASSERT_LT(Address(out_->data) + ggml_nbytes(out_), act + node_.activations().bytes);
    auto plan = kg::PlanGraph(nodes_, /*fusion=*/false, kg::DeviceChoicesOf(*launch_));
    ASSERT_TRUE(plan.has_value()) << plan.error().detail;
    steps_ = plan->steps.size();
    std::vector<std::string_view> names;
    for (const kg::PlanStep& step : plan->steps) {
      names.push_back(step.implementation);
    }
    EXPECT_EQ(names,
              (std::vector<std::string_view>{kg::kGetRowsName, kg::kRmsNormMulUnfused,
                                             kg::kMulMatVecQ, kg::kMulMatVector, kg::kAddName,
                                             kg::kMulMatVector, kg::kRopeName, kg::kSetRowsName}));
    auto scratch = kg::PlanScratch(*launch_, *plan);
    ASSERT_TRUE(scratch.has_value());
    EXPECT_GT(*scratch, 0U);  // MMVQ's quantized activations: pool addresses in the graph
    ASSERT_LE(*scratch, pool_bytes);
    auto bound = kg::BoundGraph::Bind(*registry_, *plan);
    ASSERT_TRUE(bound.has_value()) << bound.error().detail;
    bound_.emplace(std::move(*bound));
    plan_ = std::move(*plan);
  }

  // Concurrent lanes (graph_plan.h AssignLanes; executor.h BoundGraph::Run):
  // the Q8_0 product on lane 1, the F16 one on lane 2 and their sum on the
  // stream, one region; the context's lanes made, the plan bound again and
  // any graph dropped, so the next capture records the lanes' fork and join.
  void UseLanes() {
    if (!plan_.has_value()) {
      FAIL() << "the plan is bound first";
    }
    kg::GraphPlan lanes = *plan_;
    kg::AssignLanes(lanes, {{q_, {.lane = 1, .region = 1}},
                            {f_, {.lane = 2, .region = 1}},
                            {sum_, {.lane = 0, .region = 1}}});
    ASSERT_EQ(lanes.regions.size(), 1U);
    auto lane_scratch = kg::PlanLaneScratch(*launch_, lanes);
    ASSERT_TRUE(lane_scratch.has_value());
    EXPECT_GT(*lane_scratch, 0U);  // MMVQ's quantized activations, from lane 1's pool
    ASSERT_TRUE(node_
                    .Job(
                        fence_,
                        [&](jitllm::providers::NativeStream) {
                          return launch_->ConfigureLanes(kg::kMaxLanes, Bytes(*lane_scratch))
                                     ? sc::JobResult::kQueued
                                     : sc::JobResult::kNotStarted;
                        },
                        "making the lanes", 0)
                    .has_value());
    ASSERT_EQ(launch_->lanes(), kg::kMaxLanes);
    auto scratch = kg::PlanScratch(*launch_, lanes);
    ASSERT_TRUE(scratch.has_value());
    ASSERT_LE(*scratch, launch_->scratch_size(0).value());
    auto bound = kg::BoundGraph::Bind(*registry_, lanes);
    ASSERT_TRUE(bound.has_value()) << bound.error().detail;
    EXPECT_TRUE(bound->has_lanes());
    graph_.reset();
    bound_.emplace(std::move(*bound));
  }

  // One step: the token, position and cell of step k staged, then copied,
  // run and the output copied out, launch by launch, captured then
  // replayed, or replayed. The output, bit patterns.
  std::expected<std::vector<std::uint32_t>, std::string> Step(int k, Mode mode) {
    std::string failed;
    auto job = [&, k, mode](jitllm::providers::NativeStream native) -> sc::JobResult {
      auto* const stream = static_cast<cudaStream_t>(native.handle);
      const auto token = static_cast<std::int32_t>(((k * 7) + 3) % kVocab);
      const std::int32_t position = k;
      const std::int64_t cell = k % kCells;
      std::memcpy(staging_bytes_, &token, sizeof(token));
      std::memcpy(staging_bytes_ + 256, &position, sizeof(position));
      std::memcpy(staging_bytes_ + 512, &cell, sizeof(cell));
      const auto queue = [&](kg::LaunchContext& launch) -> std::expected<void, kg::KernelFailure> {
        for (const auto& [tensor, at] : std::array<std::pair<ggml_tensor*, std::uint64_t>, 3>{
                 {{ids_, 0}, {pos_, 256}, {cell_, 512}}}) {
          if (cudaMemcpyAsync(tensor->data, staging_bytes_ + at, ggml_nbytes(tensor),
                              cudaMemcpyHostToDevice, stream) != cudaSuccess) {
            return std::unexpected(
                kg::KernelFailure{.error = kg::KernelError::kUnknown, .detail = "an input copy"});
          }
        }
        if (auto r = bound_->Run(launch); !r) {
          return r;
        }
        if (cudaMemcpyAsync(staging_bytes_ + 4096, out_->data, ggml_nbytes(out_),
                            cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
            cudaMemcpyAsync(staging_bytes_ + 4096 + ggml_nbytes(out_), scores_->data,
                            ggml_nbytes(scores_), cudaMemcpyDeviceToHost, stream) != cudaSuccess) {
          return std::unexpected(
              kg::KernelFailure{.error = kg::KernelError::kUnknown, .detail = "the output copy"});
        }
        return {};
      };
      std::expected<void, kg::KernelFailure> ran;
      if (mode == Mode::kCapture) {
        auto captured = launch_->Capture(queue);
        if (!captured) {
          failed = "capture: " + captured.error().detail;
          return captured.error().error == kg::KernelError::kUnknown ? sc::JobResult::kUnknown
                                                                     : sc::JobResult::kNotStarted;
        }
        graph_.emplace(std::move(*captured));
        ++captures_;
      }
      ran = mode == Mode::kEager ? queue(*launch_) : launch_->Launch(graph_.value());
      if (!ran) {
        failed = ran.error().detail;
        return ran.error().error == kg::KernelError::kUnknown ? sc::JobResult::kUnknown
                                                              : sc::JobResult::kFailed;
      }
      return sc::JobResult::kQueued;
    };
    if (auto r = node_.Job(closure_, std::move(job), "a step", 0); !r || !failed.empty()) {
      return std::unexpected(failed.empty() ? r.error() : failed);
    }
    std::vector<std::uint32_t> bits(static_cast<std::size_t>(kOut + kCells));
    std::memcpy(bits.data(), staging_bytes_ + 4096, bits.size() * sizeof(float));
    return bits;
  }

  // The cache's bytes, read back.
  std::vector<std::uint16_t> Cache() {
    const std::uint64_t bytes = kOut * kCells * sizeof(ggml_fp16_t);
    std::byte* at = staging_bytes_ + 8192;
    const std::uint64_t base = cache_.base;
    EXPECT_TRUE(node_
                    .Job(
                        closure_,
                        [=](jitllm::providers::NativeStream native) {
                          return cudaMemcpyAsync(at, Pointer(base), bytes, cudaMemcpyDeviceToHost,
                                                 static_cast<cudaStream_t>(native.handle)) ==
                                         cudaSuccess
                                     ? sc::JobResult::kQueued
                                     : sc::JobResult::kUnknown;
                        },
                        "reading the cache", 0)
                    .has_value());
    std::vector<std::uint16_t> cache(bytes / sizeof(std::uint16_t));
    std::memcpy(cache.data(), at, bytes);
    return cache;
  }

  void ClearCache() {
    const std::uint64_t base = cache_.base;
    EXPECT_TRUE(node_
                    .Job(
                        closure_,
                        [=](jitllm::providers::NativeStream native) {
                          return cudaMemsetAsync(Pointer(base), 0, kExtent,
                                                 static_cast<cudaStream_t>(native.handle)) ==
                                         cudaSuccess
                                     ? sc::JobResult::kQueued
                                     : sc::JobResult::kUnknown;
                        },
                        "clearing the cache", 0)
                    .has_value());
  }

  // Every managed extent is at the place registered for it, and pinned.
  bool AtItsPlaces() {
    bool same = true;
    EXPECT_TRUE(node_
                    .Call(
                        [&]() -> ts::Status {
                          const std::vector<ExtentId> managed = managed_extents();
                          for (std::size_t i = 0; i < managed.size(); ++i) {
                            const sc::PageSource* now = node_.scheduler().SourceOf(managed[i]);
                            same = same && now != nullptr && sc::SamePlace(*now, sources_.at(i)) &&
                                   node_.scheduler().PlacePinned(managed[i]);
                          }
                          return {};
                        },
                        "checking the places")
                    .has_value());
    return same;
  }

  // What is mapped at each managed extent's place now (with the lanes idle).
  std::vector<std::optional<jitllm::providers::BackingId>> Backings() {
    std::vector<std::optional<jitllm::providers::BackingId>> out;
    out.reserve(weights_.size() + 1);
    for (std::size_t i = 0; i < weights_.size(); ++i) {
      out.push_back(node_.memory().MappedAt(place_, Bytes(i * kExtent)));
    }
    out.push_back(node_.memory().MappedAt(cache_.reservation, Bytes(0)));
    return out;
  }

  kg::LaunchContext& launch() { return *launch_; }
  const kg::CapturedGraph* graph() const { return graph_ ? &*graph_ : nullptr; }
  bool captured() const { return graph_.has_value(); }
  int captures() const { return captures_; }
  std::size_t steps() const { return steps_; }
  const jitllm::catalog::Closure& closure() const { return closure_; }
  std::vector<ExtentId> weights() const { return weights_; }
  std::vector<ExtentId> all_out() const { return managed_extents(); }

  std::uint32_t stream() const override { return 0; }
  const jitllm::catalog::Closure& fence_closure() const override { return fence_; }
  std::vector<ExtentId> managed_extents() const override {
    std::vector<ExtentId> all = weights_;
    all.push_back(cache_.extents.at(0));
    return all;
  }
  ts::Status Release() override {
    graph_.reset();  // before the context, and before its memory goes
    bound_.reset();
    launch_.reset();
    auto& memory = node_.memory();
    const bool freed =
        memory.Free(place_).has_value() && memory.Free(cache_.reservation).has_value();
    for (const int fd : {fd_, spill_}) {
      if (fd >= 0) {
        (void)::close(fd);
      }
    }
    return freed ? ts::Status{} : std::unexpected("a place still has mappings");
  }

 private:
  ts::PagedNode& node_;
  int fd_ = -1;
  int spill_ = -1;
  jitllm::providers::ReservationId place_;
  std::uint64_t base_ = 0;
  std::vector<ExtentId> weights_;
  ts::Mapped cache_;
  std::vector<sc::PageSource> sources_;  // by managed extent
  std::vector<ExtentId> staging_;
  std::byte* staging_bytes_ = nullptr;
  jitllm::catalog::Closure closure_;
  jitllm::catalog::Closure fence_;
  std::unique_ptr<jitllm::execution::Registry> registry_;
  std::optional<kg::TensorArena> arena_;
  std::vector<ggml_tensor*> nodes_;
  ggml_tensor* ids_ = nullptr;
  ggml_tensor* pos_ = nullptr;
  ggml_tensor* cell_ = nullptr;
  ggml_tensor* out_ = nullptr;
  ggml_tensor* scores_ = nullptr;
  ggml_tensor* q_ = nullptr;
  ggml_tensor* f_ = nullptr;
  ggml_tensor* sum_ = nullptr;
  std::optional<kg::GraphPlan> plan_;
  std::size_t steps_ = 0;
  std::unique_ptr<kg::LaunchContext> launch_;
  std::optional<kg::BoundGraph> bound_;
  std::optional<kg::CapturedGraph> graph_;  // destroyed before launch_
  int captures_ = 0;
};

// Model B: three weight extents, which a swap to it pages in.
class OtherModel final : public ts::PagedModel {
 public:
  explicit OtherModel(ts::PagedNode& node) : node_(node) {}

  void Setup() {
    std::vector<std::byte> file(3 * kExtent);
    for (std::size_t i = 0; i < file.size(); ++i) {
      file[i] = static_cast<std::byte>((i * 13) + 5);
    }
    fd_ = DirectFile(file);
    ASSERT_GE(fd_, 0);
    place_ = node_.memory().Reserve(Bytes(file.size())).value();
    base_ = node_.memory().RangeOf(place_).value().base;
    for (std::uint32_t i = 0; i < 3; ++i) {
      weights_.push_back(node_.catalog()
                             .AddExtent({.domain = node_.domain(),
                                         .memory_class = MemoryClass::kWeights,
                                         .recovery = Recovery::kFromArtifact,
                                         .size = Bytes(kExtent),
                                         .content = {.artifact = {8}, .group = 0, .chunk = i}})
                             .value());
    }
    auto staging = node_.Pinned(4096, 1, staging_);
    ASSERT_TRUE(staging.has_value());
  }
  void Register() {
    for (std::size_t i = 0; i < weights_.size(); ++i) {
      ASSERT_TRUE(
          node_.scheduler()
              .SetSource(weights_[i], Landed(fd_, i * kExtent, base_ + (i * kExtent), place_,
                                             i * kExtent, node_.device_class(), false))
              .has_value());
    }
    closure_ = node_.catalog().ClosureOfExtents(weights_).value();
    fence_ = node_.catalog().ClosureOfExtents(staging_).value();
  }
  const jitllm::catalog::Closure& closure() const { return closure_; }
  std::vector<ExtentId> weights() const { return weights_; }

  std::uint32_t stream() const override { return 1; }
  const jitllm::catalog::Closure& fence_closure() const override { return fence_; }
  std::vector<ExtentId> managed_extents() const override { return weights_; }
  ts::Status Release() override {
    const bool freed = node_.memory().Free(place_).has_value();
    if (fd_ >= 0) {
      (void)::close(fd_);
    }
    return freed ? ts::Status{} : std::unexpected("B's place still has mappings");
  }

 private:
  ts::PagedNode& node_;
  int fd_ = -1;
  jitllm::providers::ReservationId place_;
  std::uint64_t base_ = 0;
  std::vector<ExtentId> weights_;
  std::vector<ExtentId> staging_;
  jitllm::catalog::Closure closure_;
  jitllm::catalog::Closure fence_;
};

// A node with A and B, under a budget that holds only one of them.
class CudaGraphTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ASSERT_TRUE(node_.Open().has_value());
    a_.Setup();
    b_.Setup();
    ASSERT_TRUE(node_.MapWorkspace(kExtent, kExtent).has_value());
    const std::uint64_t fixed = node_.catalog().OccupancyOf(node_.domain()).Total().value();
    ASSERT_TRUE(node_.Start(Bytes(fixed + (3 * kExtent))).has_value());
    a_.Register();
    b_.Register();
    node_.Run();
    std::vector<ts::LoadStats> log;
    ASSERT_TRUE(node_.Load(a_.weights(), "A's weights", log).has_value());
    a_.Bind();
  }
  void TearDown() override {
    const std::array<ts::PagedModel*, 2> models = {&a_, &b_};
    const ts::Status finished = node_.TearDown(models);
    EXPECT_TRUE(finished.has_value()) << finished.error();
  }

  // Waits until no eviction is left (backing no load took, released).
  void Settle() {
    std::size_t left = 1;
    for (int i = 0; i < 5000 && left > 0; ++i) {
      ASSERT_TRUE(node_
                      .Call(
                          [&]() -> ts::Status {
                            left = node_.scheduler().evictions();
                            return {};
                          },
                          "counting evictions")
                      .has_value());
    }
    ASSERT_EQ(left, 0U);
  }

  // Every step launch by launch from a cleared cache: the outputs, and the
  // cache at the end.
  std::vector<std::vector<std::uint32_t>> Control(std::vector<std::uint16_t>& cache) {
    a_.ClearCache();
    std::vector<std::vector<std::uint32_t>> outputs;
    for (int k = 0; k < kSteps; ++k) {
      auto out = a_.Step(k, Mode::kEager);
      EXPECT_TRUE(out.has_value()) << out.error();
      outputs.push_back(out.value_or(std::vector<std::uint32_t>{}));
    }
    cache = a_.Cache();
    return outputs;
  }

  ts::PagedNode node_{{.compute_streams = 2, .slots = 4, .inline_lanes = false, .coalesce = false}};
  GraphModel a_{node_};
  OtherModel b_{node_};
};

// A captured graph replays bit for bit what launching the plan computes,
// step after step, as the token, position and cell change.
TEST_F(CudaGraphTest, AReplayEqualsTheStepLaunchedLaunchByLaunch) {
  std::vector<std::uint16_t> control_cache;
  const auto control = Control(control_cache);
  ASSERT_EQ(control.size(), static_cast<std::size_t>(kSteps));
  EXPECT_NE(control[1], control[2]);  // the inputs vary, and so do the outputs

  a_.ClearCache();
  for (int k = 0; k < kSteps; ++k) {
    auto out = a_.Step(k, ModeOf(k));
    ASSERT_TRUE(out.has_value()) << "step " << k << ": " << out.error();
    EXPECT_EQ(*out, control[static_cast<std::size_t>(k)]) << "step " << k;
  }
  EXPECT_EQ(a_.Cache(), control_cache);
  EXPECT_EQ(a_.captures(), 1);
  // The plan's kernels (MMVQ quantizes first) and the five copies, as one
  // graph.
  const kg::CapturedGraph* graph = a_.graph();
  ASSERT_NE(graph, nullptr);
  EXPECT_GE(graph->nodes(), a_.steps() + 5);
  std::println(
      "graph: {} nodes for {} plan steps; captured in {:.1f} us, instantiated and "
      "uploaded in {:.1f} us",
      graph->nodes(), a_.steps(), graph->capture_seconds() * 1e6,
      graph->instantiate_seconds() * 1e6);
}

// Concurrent lanes compute bit for bit what the stream alone computes,
// launch by launch, captured (the lanes' fork and join in the graph) and
// replayed, and the cache they write is the same.
TEST_F(CudaGraphTest, LanesComputeWhatTheStreamAloneComputes) {
  std::vector<std::uint16_t> control_cache;
  const auto control = Control(control_cache);
  ASSERT_EQ(control.size(), static_cast<std::size_t>(kSteps));
  a_.UseLanes();
  a_.ClearCache();
  for (int k = 0; k < kSteps; ++k) {
    auto out = a_.Step(k, ModeOf(k));
    ASSERT_TRUE(out.has_value()) << "step " << k << ": " << out.error();
    EXPECT_EQ(*out, control[static_cast<std::size_t>(k)]) << "step " << k;
  }
  EXPECT_EQ(a_.Cache(), control_cache);
  EXPECT_EQ(a_.captures(), 1);
  EXPECT_FALSE(a_.launch().faulted());
}

// A→B→A twice, with the handoff and without: A's cache written back and
// restored, its weights paged in again, the second time into backing
// created anew. A's places are the pinned ones, and the graph captured
// before the swaps, never captured again, replays bit for bit what the
// never-swapped control computes.
TEST_F(CudaGraphTest, AGraphReplaysAfterSwapsThatRestoreEveryAddress) {
  std::vector<std::uint16_t> control_cache;
  const auto control = Control(control_cache);
  a_.ClearCache();
  int k = 0;
  const auto steps = [&](int until) {
    for (; k < until; ++k) {
      auto out = a_.Step(k, ModeOf(k));
      ASSERT_TRUE(out.has_value()) << "step " << k << ": " << out.error();
      EXPECT_EQ(*out, control[static_cast<std::size_t>(k)]) << "step " << k;
    }
  };
  steps(4);
  for (const bool handoff : {true, false}) {
    Settle();
    const auto before = a_.Backings();
    for (const auto& backing : before) {
      ASSERT_TRUE(backing.has_value());
    }
    ts::SwapReport out;
    ASSERT_TRUE(node_.Swap(a_.all_out(), b_.closure(), handoff, out).has_value());
    Settle();
    for (const auto& backing : a_.Backings()) {
      EXPECT_FALSE(backing.has_value());  // A's places are empty while B is resident
    }
    ts::SwapReport back;
    ASSERT_TRUE(node_.Swap(b_.weights(), a_.closure(), handoff, back).has_value());
    Settle();
    EXPECT_TRUE(a_.AtItsPlaces()) << "handoff " << handoff;
    const auto after = a_.Backings();
    for (std::size_t i = 0; i < after.size(); ++i) {
      ASSERT_TRUE(after[i].has_value()) << i;  // mapped at its place again
      if (!handoff) {
        EXPECT_NE(after[i], before[i]) << i;  // over backing made anew
      }
    }
    steps(handoff ? 8 : kSteps);
  }
  EXPECT_EQ(a_.Cache(), control_cache);
  EXPECT_EQ(a_.captures(), 1);  // captured once, before the swaps
}

// A capture that cannot be made is refused, and the context is as it was:
// not faulted, not capturing, and it runs the step as before. A nested
// capture is refused and leaves the outer one to finish; a graph replays
// only on the context that captured it; a faulted context captures
// nothing.
TEST_F(CudaGraphTest, ACaptureThatCannotBeMadeIsRefusedCleanly) {
  std::vector<std::uint16_t> control_cache;
  const auto control = Control(control_cache);
  kg::LaunchContext& launch = a_.launch();
  Outcome synchronizing;
  Outcome refusing;
  Outcome nested;
  Outcome inner;
  bool faulted = true;
  bool capturing = true;
  cudaStreamCaptureStatus status = cudaStreamCaptureStatusActive;
  ASSERT_TRUE(node_
                  .Job(
                      a_.fence_closure(),
                      [&](jitllm::providers::NativeStream native) {
                        auto* const stream = static_cast<cudaStream_t>(native.handle);
                        // A synchronization inside the capture: refused, and
                        // the rest of the record fails.
                        synchronizing = OutcomeOf(launch.Capture(
                            [&](kg::LaunchContext& l) -> std::expected<void, kg::KernelFailure> {
                              (void)cudaStreamSynchronize(stream);
                              return l.Run(Bytes(0), [](ggml_backend_cuda_context&) {});
                            }));
                        // A record that fails on its own, before any launch.
                        refusing = OutcomeOf(launch.Capture(
                            [&](kg::LaunchContext& l) -> std::expected<void, kg::KernelFailure> {
                              return l.Run(Bytes(std::uint64_t{1} << 40U),
                                           [](ggml_backend_cuda_context&) {});
                            }));
                        // A capture inside a capture.
                        nested = OutcomeOf(launch.Capture(
                            [&](kg::LaunchContext& l) -> std::expected<void, kg::KernelFailure> {
                              inner = OutcomeOf(l.Capture([](kg::LaunchContext&) {
                                return std::expected<void, kg::KernelFailure>{};
                              }));
                              return l.Run(Bytes(0), [](ggml_backend_cuda_context&) {});
                            }));
                        faulted = launch.faulted();
                        capturing = launch.capturing();
                        (void)cudaStreamIsCapturing(stream, &status);
                        return sc::JobResult::kQueued;
                      },
                      "refused captures", 0)
                  .has_value());
  ASSERT_TRUE(synchronizing.ran && refusing.ran && nested.ran && inner.ran);
  EXPECT_FALSE(synchronizing.made);
  EXPECT_EQ(synchronizing.failure.error, kg::KernelError::kRejected)
      << synchronizing.failure.detail;
  EXPECT_FALSE(refusing.made);
  EXPECT_EQ(refusing.failure.error, kg::KernelError::kRejected);
  EXPECT_FALSE(inner.made);
  EXPECT_EQ(inner.failure.error, kg::KernelError::kRejected);
  EXPECT_TRUE(nested.made) << nested.failure.detail;
  EXPECT_FALSE(faulted);
  EXPECT_FALSE(capturing);
  EXPECT_EQ(status, cudaStreamCaptureStatusNone);

  // The context runs as before.
  a_.ClearCache();
  for (int k = 0; k < 3; ++k) {
    auto out = a_.Step(k, ModeOf(k));
    ASSERT_TRUE(out.has_value()) << out.error();
    EXPECT_EQ(*out, control[static_cast<std::size_t>(k)]) << k;
  }

  // Another context (on B's stream) replays none of A's graphs, and a
  // faulted one captures nothing.
  const kg::CapturedGraph* graph = a_.graph();
  ASSERT_NE(graph, nullptr);
  Outcome foreign;
  Outcome after_fault;
  bool other_faulted = false;
  ASSERT_TRUE(node_
                  .Job(
                      b_.fence_closure(),
                      [&](jitllm::providers::NativeStream) {
                        auto other =
                            kg::LaunchContext::Create(0, node_.execution(), node_.stream(1), {});
                        if (!other) {
                          return sc::JobResult::kNotStarted;
                        }
                        foreign = OutcomeOf((*other)->Launch(*graph));
                        other_faulted = (*other)->faulted();
                        ggml_cuda_error("scripted", "a test", __FILE__, __LINE__, "a fault");
                        EXPECT_FALSE(
                            (*other)->Run(Bytes(0), [](ggml_backend_cuda_context&) {}).has_value());
                        after_fault = OutcomeOf((*other)->Capture([](kg::LaunchContext&) {
                          return std::expected<void, kg::KernelFailure>{};
                        }));
                        return sc::JobResult::kQueued;
                      },
                      "another context", 1)
                  .has_value());
  ASSERT_TRUE(foreign.ran && after_fault.ran);
  EXPECT_FALSE(foreign.made);
  EXPECT_EQ(foreign.failure.error, kg::KernelError::kRejected);
  EXPECT_FALSE(other_faulted);
  EXPECT_FALSE(after_fault.made);
  EXPECT_EQ(after_fault.failure.error, kg::KernelError::kRejected);
  EXPECT_NE(after_fault.failure.detail.find("faulted"), std::string::npos)
      << after_fault.failure.detail;
}

// RE-029: a stream holds about 1,020 pending operations, and a launch into
// a full one blocks. A graph of 1,500 kernels, replayed behind a wait on a
// host flag, takes one entry: the stream holds many replays before one
// blocks (the probe prints how many).
TEST_F(CudaGraphTest, AGraphOfManyKernelsIsOneOperationInItsStream) {
  constexpr int kKernels = 1500;
  constexpr int kReplays = 1100;
  const std::uint64_t act = node_.activations().base;
  auto arena = kg::TensorArena::Create(kKernels + 4);
  ASSERT_TRUE(arena.has_value());
  ggml_context* c = arena->context();
  ggml_tensor* x = ggml_new_tensor_1d(c, GGML_TYPE_F32, 256);
  ggml_tensor* y = ggml_new_tensor_1d(c, GGML_TYPE_F32, 256);
  kg::TensorArena::Bind(x, act);
  kg::TensorArena::Bind(y, act + 1024);
  std::vector<ggml_tensor*> sums;
  for (int i = 0; i < kKernels; ++i) {
    sums.push_back(ggml_add(c, x, y));
    kg::TensorArena::Bind(sums.back(), act + 2048);
  }
  void* flag = nullptr;
  ASSERT_EQ(cudaHostAlloc(&flag, sizeof(std::uint32_t), cudaHostAllocMapped), cudaSuccess);
  std::atomic_ref<std::uint32_t>(*static_cast<std::uint32_t*>(flag)).store(0);
  void* device_flag = nullptr;
  ASSERT_EQ(cudaHostGetDevicePointer(&device_flag, flag, 0), cudaSuccess);
  kg::LaunchContext& launch = a_.launch();
  std::optional<kg::CapturedGraph> graph;
  std::size_t nodes = 0;
  std::atomic<int> replayed{0};
  std::string failed;
  ts::Done done;
  const std::uint64_t request = node_.Submit(std::make_unique<ts::RunProgram>(
      done, a_.closure(),
      [&](jitllm::providers::NativeStream native) {
        auto captured = launch.Capture([&](kg::LaunchContext& l) {
          for (ggml_tensor* sum : sums) {
            if (auto r = kg::Add(l, sum); !r) {
              return r;
            }
          }
          return std::expected<void, kg::KernelFailure>{};
        });
        if (!captured) {
          failed = captured.error().detail;
          return sc::JobResult::kNotStarted;
        }
        const kg::CapturedGraph& made = graph.emplace(std::move(*captured));
        nodes = made.nodes();
        if (cuStreamWaitValue32(static_cast<CUstream>(native.handle),
                                reinterpret_cast<CUdeviceptr>(device_flag), 1,
                                CU_STREAM_WAIT_VALUE_GEQ) != CUDA_SUCCESS) {
          return sc::JobResult::kUnknown;
        }
        for (int i = 0; i < kReplays; ++i) {
          if (auto r = launch.Launch(made); !r) {
            failed = r.error().detail;
            return sc::JobResult::kUnknown;
          }
          replayed.store(i + 1);
        }
        return sc::JobResult::kQueued;
      },
      0));
  // Until the first replay is queued (the job waits for its turn, and
  // captures first), then until the replays stop coming (a launch
  // blocked) or all are queued.
  for (int i = 0; i < 2000 && replayed.load() == 0; ++i) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  int seen = -1;
  for (int i = 0; i < 400 && replayed.load() != seen; ++i) {
    seen = replayed.load();
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
  const int held = replayed.load();
  std::atomic_ref<std::uint32_t>(*static_cast<std::uint32_t*>(flag)).store(1);
  const ts::Status job = node_.Await(done, "the gated replays", request);
  EXPECT_TRUE(job.has_value()) << job.error();
  EXPECT_TRUE(failed.empty()) << failed;
  ASSERT_TRUE(graph.has_value());
  EXPECT_EQ(nodes, static_cast<std::size_t>(kKernels));
  // Each replay one entry: far more than one 1,500-kernel replay fits.
  EXPECT_GE(held, 64);
  EXPECT_EQ(replayed.load(), kReplays);
  std::println(
      "behind a closed gate, the stream took {} of {} replays of a {}-kernel graph "
      "before a launch blocked",
      held, kReplays, kKernels);
  graph.reset();
  (void)cudaFreeHost(flag);
}

}  // namespace
