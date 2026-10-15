// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Block-quantized KV caches (kernels/ggml/kv_quant.h) under llmpalooza's
// launch context, on a GB10 (label `gpu`) and on a discrete GPU
// (`gpu-discrete`):
// - the quantizing stores (one store, a group, and NEOX RoPE fused with
//   one) write exactly the bytes of kernels/reference EncodeRows (GGML's
//   reference quantizers) for every stored row, I32 and I64 indices,
//   broadcast indices, and blocks at the quantizers' rounding boundaries,
//   and leave every other row's bytes as they were; the fused RoPE store's
//   rows are GGML's RoPE output (ggml.rope.neox) stored so, byte for byte;
//   upstream's own fast-math store (set-rows.cu) is run beside them and how
//   often its blocks differ is printed, not judged; in blocks below F32's
//   normal range the codes may differ, and only where both scales are F16
//   zeros;
// - the quantized vector attention at decode (1 and 2 query rows) and
//   prefill shapes (odd rows, 1,024 and 1,025 rows with the mask pre-pass),
//   D 64 and 256, Q8_0 and Q4_0, no GQA and GQA 2, 7 and 8, with and
//   without softcap, against
//   the reference executor (kernels/reference/exec.h: the same K and V
//   bytes decoded exactly, attention in F64) within upstream's own bound
//   for flash attention (test-backend-ops.cpp's NMSE 5e-4; the kernel
//   quantizes Q to Q8_1 for the K product and accumulates V in F16);
// - the checks refuse what the kernels would not take, before any launch;
// - the registry declares the five implementations with their capabilities,
//   and Candidates() finds each for its formats only.

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <expected>
#include <format>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <span>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "base/bytes.h"
#include "execution/registry.h"
#include "expected_error.h"
#include "format/encoding.h"
#include "format/layout.h"
#include "format/tensor_format.h"
#include "ggml.h"
#include "ir/builder.h"
#include "ir/graph.h"
#include "ir/op.h"
#include "kernels/ggml/implementations.h"
#include "kernels/ggml/kv_quant.h"
#include "kernels/ggml/launch.h"
#include "kernels/ggml/ops.h"
#include "kernels/ggml/tensors.h"
#include "kernels/reference/decode.h"
#include "kernels/reference/exec.h"
#include "providers/cuda/cuda_device_execution.h"
#include "providers/device_execution.h"

// Upstream's own quantizing store (set-rows.cu), for comparison only.
struct ggml_backend_cuda_context;
void ggml_cuda_op_set_rows(ggml_backend_cuda_context& ctx, ggml_tensor* dst);

namespace {

using llmp::base::Bytes;
using llmp::format::Encoding;
using llmp::format::Layout;
using llmp::kernels::ggml::KernelError;
using llmp::kernels::ggml::KernelFailure;
using llmp::kernels::ggml::LaunchContext;
using llmp::kernels::ggml::TensorArena;
using llmp::providers::DeviceExecution;
using llmp::providers::FenceState;
using llmp::providers::StreamId;
using llmp::test_support::FailedCode;
namespace kg = llmp::kernels::ggml;
namespace ref = llmp::kernels::reference;
namespace ir = llmp::ir;

constexpr double kFlashAttnNmse = 5e-4;  // test-backend-ops.cpp's bound for flash attention
constexpr std::uint64_t kWorkspace = 256ULL << 20U;
constexpr std::size_t kBlock = 32;

std::size_t N(std::int64_t n) { return static_cast<std::size_t>(n); }

Encoding EncodingOf(ggml_type type) {
  return type == GGML_TYPE_Q8_0 ? Encoding::kQ8_0 : Encoding::kQ4_0;
}

std::vector<float> Normal(std::uint64_t seed, std::size_t count, float sigma = 1.0f) {
  std::mt19937_64 random(seed);
  std::normal_distribution<float> normal(0.0f, sigma);
  std::vector<float> values(count);
  for (float& v : values) {
    v = normal(random);
  }
  return values;
}

// Values whose blocks reach the quantizers' edges: a zero block, ties in
// magnitude of opposite sign (Q4_0 takes the first), codes exactly at a
// rounding boundary (amax 127 or vmax -8: d is 1), tiny and huge blocks,
// and ordinary normal values of several scales elsewhere.
std::vector<float> StoreValues(std::uint64_t seed, std::size_t count) {
  std::vector<float> v = Normal(seed, count);
  std::mt19937_64 random(seed + 1);
  for (std::size_t b = 0; b + kBlock <= count; b += kBlock) {
    float* x = v.data() + b;
    switch ((b / kBlock) % 9) {
      case 0:
        std::fill(x, x + kBlock, 0.0f);
        break;
      case 1:
        x[3] = 3.0f;
        x[9] = -3.0f;
        x[20] = 3.0f;
        for (std::size_t j = 0; j < kBlock; ++j) {
          if (j != 3 && j != 9 && j != 20) x[j] = std::clamp(x[j], -2.9f, 2.9f);
        }
        break;
      case 2:  // Q8_0: d = 1, codes k + 0.5 round away from zero
        for (std::size_t j = 0; j < kBlock; ++j) {
          x[j] = static_cast<float>(static_cast<int>(random() % 252) - 126) + 0.5f;
        }
        x[7] = -127.0f;
        break;
      case 3:  // Q4_0: d = 1, x + 8.5 lands on integers and halves
        for (std::size_t j = 0; j < kBlock; ++j) {
          x[j] = static_cast<float>(static_cast<int>(random() % 15) - 7) +
                 ((random() % 2) != 0 ? 0.5f : 0.0f);
        }
        x[0] = -8.0f;
        break;
      case 4:
        for (std::size_t j = 0; j < kBlock; ++j) x[j] *= 1e-20f;
        break;
      case 5:
        for (std::size_t j = 0; j < kBlock; ++j) x[j] *= 1e20f;
        break;
      case 6:
        for (std::size_t j = 0; j < kBlock; ++j) x[j] *= 37.0f;
        break;
      default:
        break;
    }
  }
  return v;
}

// One row of `values` in a block encoding, as EncodeRows writes it.
std::vector<std::byte> EncodeRow(ggml_type type, std::span<const float> values) {
  const auto shape = llmp::format::ShapeOf(std::array{std::uint64_t{values.size()}});
  const llmp::format::TensorFormat f{
      .encoding = EncodingOf(type), .layout = Layout::kGgmlBlocks, .shape = *shape};
  std::vector<std::byte> bytes(ggml_row_size(type, static_cast<std::int64_t>(values.size())));
  EXPECT_TRUE(ref::EncodeRows(f, 0, 0, values, bytes).has_value());
  return bytes;
}

double Nmse(const std::vector<float>& got, const std::vector<float>& want) {
  EXPECT_EQ(got.size(), want.size());
  double error = 0.0;
  double norm = 0.0;
  for (std::size_t i = 0; i < std::min(got.size(), want.size()); ++i) {
    const double d = static_cast<double>(got[i]) - want[i];
    error += d * d;
    norm += static_cast<double>(want[i]) * want[i];
  }
  return norm > 0.0 ? error / norm : error;
}

// Blocks of `got` that differ from `want` (block_bytes each).
std::size_t DifferingBlocks(std::span<const std::byte> got, std::span<const std::byte> want,
                            std::size_t block_bytes) {
  std::size_t differing = 0;
  for (std::size_t b = 0; b + block_bytes <= std::min(got.size(), want.size()); b += block_bytes) {
    differing += std::memcmp(got.data() + b, want.data() + b, block_bytes) != 0 ? 1 : 0;
  }
  return differing;
}

void Launched(const std::expected<void, KernelFailure>& result, const std::string& what) {
  EXPECT_TRUE(result.has_value()) << what << ": " << (result ? "" : result.error().detail);
}

class KvQuantTest : public ::testing::Test {
 protected:
  void SetUp() override {
    execution_ = std::move(llmp::providers::cuda::OpenDeviceExecution(0).value());
    stream_ = execution_->CreateStream().value();
    const std::uint64_t workspace = Allocate(kWorkspace);
    auto launch = LaunchContext::Create(0, *execution_, stream_,
                                        {.base = workspace, .size = Bytes(kWorkspace)});
    ASSERT_TRUE(launch.has_value()) << (launch ? "" : launch.error().detail);
    launch_ = std::move(*launch);
    arena_ = std::make_unique<TensorArena>(TensorArena::Create(512).value());
  }

  void TearDown() override {
    Finish();
    launch_.reset();
    ASSERT_TRUE(execution_->DestroyStream(stream_).has_value());
    for (void* pointer : device_) {
      EXPECT_EQ(cudaFree(pointer), cudaSuccess);
    }
  }

  void Finish() {
    const auto fence = execution_->Record(stream_).value();
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(120);
    FenceState state = FenceState::kPending;
    while ((state = execution_->Query(fence).value()) == FenceState::kPending &&
           std::chrono::steady_clock::now() < deadline) {
      std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
    ASSERT_EQ(state, FenceState::kComplete);
    ASSERT_TRUE(execution_->Release(fence).has_value());
  }

  std::uint64_t Allocate(std::size_t bytes) {
    void* pointer = nullptr;
    EXPECT_EQ(cudaMalloc(&pointer, std::max<std::size_t>(bytes, 256)), cudaSuccess);
    device_.push_back(pointer);
    return reinterpret_cast<std::uintptr_t>(pointer);
  }

  ggml_context* c() const { return arena_->context(); }
  LaunchContext& launch() { return *launch_; }

  // Binds `tensor` to new device memory holding `data` (its bytes) if given.
  template <typename T = float>
  ggml_tensor* Place(ggml_tensor* tensor, const std::vector<T>& data = {}) {
    const std::size_t size = ggml_nbytes(tensor);
    const std::uint64_t address = Allocate(size);
    TensorArena::Bind(tensor, address);
    if (!data.empty()) {
      EXPECT_EQ(data.size() * sizeof(T), size);
      Upload(tensor, data);
    }
    return tensor;
  }

  template <typename T>
  void Upload(const ggml_tensor* tensor, const std::vector<T>& data) {
    Finish();
    EXPECT_EQ(cudaMemcpy(tensor->data, data.data(), data.size() * sizeof(T),  // NOLINT
                         cudaMemcpyHostToDevice),
              cudaSuccess);
    // A copy from pageable memory may return before its DMA lands, and the
    // provider's stream does not wait for the legacy stream.
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  }

  template <typename T = float>
  std::vector<T> Download(const ggml_tensor* tensor) {
    Finish();
    std::vector<T> values(ggml_nbytes(tensor) / sizeof(T));
    EXPECT_EQ(cudaMemcpy(values.data(), tensor->data, ggml_nbytes(tensor), cudaMemcpyDeviceToHost),
              cudaSuccess);
    return values;
  }

  static int ComputeCapability() {
    int major = 0;
    int minor = 0;
    EXPECT_EQ(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0), cudaSuccess);
    EXPECT_EQ(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, 0), cudaSuccess);
    return (10 * major) + minor;
  }

  std::unique_ptr<DeviceExecution> execution_;
  StreamId stream_;
  std::vector<void*> device_;
  std::unique_ptr<LaunchContext> launch_;
  std::unique_ptr<TensorArena> arena_;
};

// A cache's bytes before a store: a fixed pattern, so that untouched rows
// are seen untouched.
std::vector<std::byte> Pattern(std::size_t bytes) {
  std::vector<std::byte> out(bytes);
  for (std::size_t i = 0; i < bytes; ++i) {
    out[i] = static_cast<std::byte>((i * 131U + 7U) & 0xFFU);
  }
  return out;
}

struct StoreCase {
  std::int64_t width = 0;  // values a row
  std::int64_t rows = 0;
  std::int64_t planes = 1;  // values' ne[2]
  std::int64_t id_planes = 1;
  std::int64_t cells = 0;
};

TEST_F(KvQuantTest, StoresWriteExactlyTheReferenceEncodersBytes) {
  const std::array<StoreCase, 4> cases = {{
      {.width = 1024, .rows = 1, .cells = 512},   // decode: four heads of 256
      {.width = 1024, .rows = 77, .cells = 512},  // prefill
      {.width = 128, .rows = 33, .cells = 64},    // two heads of 64
      {.width = 64, .rows = 5, .planes = 6, .id_planes = 2, .cells = 16},  // broadcast ids
  }};
  std::size_t upstream_blocks = 0;
  std::size_t upstream_differing = 0;
  std::uint64_t seed = 100;
  for (const ggml_type type : {GGML_TYPE_Q8_0, GGML_TYPE_Q4_0}) {
    const std::size_t block_bytes = ggml_type_size(type);
    for (const ggml_type id_type : {GGML_TYPE_I32, GGML_TYPE_I64}) {
      for (const StoreCase& sc : cases) {
        SCOPED_TRACE(std::format("{} ids {} width {} rows {} planes {}", ggml_type_name(type),
                                 ggml_type_name(id_type), sc.width, sc.rows, sc.planes));
        const auto values = StoreValues(++seed, N(sc.width * sc.rows * sc.planes));
        // Distinct cells for each row of each index plane.
        std::vector<std::int64_t> ids(N(sc.rows * sc.id_planes));
        std::mt19937_64 random(seed);
        for (std::int64_t p = 0; p < sc.id_planes; ++p) {
          std::vector<std::int64_t> cells(N(sc.cells));
          for (std::int64_t i = 0; i < sc.cells; ++i) cells[N(i)] = i;
          std::shuffle(cells.begin(), cells.end(), random);
          std::copy_n(cells.begin(), sc.rows, ids.begin() + (p * sc.rows));
        }
        const auto row_bytes = ggml_row_size(type, sc.width);
        const std::size_t plane_bytes = row_bytes * N(sc.cells);
        const auto before = Pattern(plane_bytes * N(sc.planes));
        std::vector<std::byte> want = before;
        for (std::int64_t p = 0; p < sc.planes; ++p) {
          for (std::int64_t r = 0; r < sc.rows; ++r) {
            const auto cell = ids[N(((p % sc.id_planes) * sc.rows) + r)];
            const auto row = EncodeRow(
                type, std::span(values).subspan(N(((p * sc.rows) + r) * sc.width), N(sc.width)));
            std::ranges::copy(
                row, want.begin() +
                         static_cast<std::ptrdiff_t>((N(p) * plane_bytes) + (N(cell) * row_bytes)));
          }
        }
        auto* v =
            Place(ggml_new_tensor_3d(c(), GGML_TYPE_F32, sc.width, sc.rows, sc.planes), values);
        ggml_tensor* index = ggml_new_tensor_2d(c(), id_type, sc.rows, sc.id_planes);
        if (id_type == GGML_TYPE_I64) {
          Place(index, ids);
        } else {
          Place(index, std::vector<std::int32_t>(ids.begin(), ids.end()));
        }
        for (const bool upstream : {false, true}) {
          auto* cache = Place(ggml_new_tensor_3d(c(), type, sc.width, sc.cells, sc.planes), before);
          auto* store = ggml_set_rows(c(), cache, v, index);
          TensorArena::Bind(store, reinterpret_cast<std::uintptr_t>(cache->data));
          ASSERT_TRUE(kg::CheckSetRowsQuant(store));
          if (upstream) {
            Launched(launch().Run(Bytes(0),
                                  [store](ggml_backend_cuda_context& context) {
                                    ggml_cuda_op_set_rows(context, store);
                                  }),
                     "upstream's set_rows");
          } else {
            Launched(kg::SetRowsQuant(launch(), store), "the quantizing store");
          }
          const auto got = Download<std::byte>(cache);
          if (upstream) {
            upstream_blocks += got.size() / block_bytes;
            upstream_differing += DifferingBlocks(got, want, block_bytes);
          } else {
            ASSERT_EQ(got.size(), want.size());
            EXPECT_EQ(DifferingBlocks(got, want, block_bytes), 0U);
            EXPECT_TRUE(got == want);
          }
        }
      }
    }
  }
  std::cout << "upstream's fast-math set_rows: " << upstream_differing << " of " << upstream_blocks
            << " cache blocks differ from the reference encoder\n";
}

TEST_F(KvQuantTest, GroupedStoresWriteWhatEachStoreWrites) {
  constexpr std::int64_t kWidth = 512;
  constexpr std::int64_t kRows = 9;
  constexpr std::int64_t kCells = 256;
  struct One {
    ggml_type type;
    ggml_type ids;
  };
  const std::array<One, 3> stores = {{{GGML_TYPE_Q8_0, GGML_TYPE_I64},
                                      {GGML_TYPE_Q4_0, GGML_TYPE_I64},
                                      {GGML_TYPE_Q8_0, GGML_TYPE_I32}}};
  std::vector<ggml_tensor*> nodes;
  std::vector<ggml_tensor*> caches;
  std::vector<std::vector<std::byte>> wants;
  for (std::size_t s = 0; s < stores.size(); ++s) {
    const auto values = StoreValues(500 + s, N(kWidth * kRows));
    std::vector<std::int64_t> ids(N(kRows));
    for (std::int64_t r = 0; r < kRows; ++r) ids[N(r)] = (r * 37 + 11) % kCells;
    const auto row_bytes = ggml_row_size(stores[s].type, kWidth);
    const auto before = Pattern(row_bytes * N(kCells));
    auto want = before;
    for (std::int64_t r = 0; r < kRows; ++r) {
      const auto row =
          EncodeRow(stores[s].type, std::span(values).subspan(N(r * kWidth), N(kWidth)));
      std::ranges::copy(row, want.begin() + static_cast<std::ptrdiff_t>(N(ids[N(r)]) * row_bytes));
    }
    auto* v = Place(ggml_new_tensor_2d(c(), GGML_TYPE_F32, kWidth, kRows), values);
    ggml_tensor* index = ggml_new_tensor_1d(c(), stores[s].ids, kRows);
    if (stores[s].ids == GGML_TYPE_I64) {
      Place(index, ids);
    } else {
      Place(index, std::vector<std::int32_t>(ids.begin(), ids.end()));
    }
    auto* cache = Place(ggml_new_tensor_2d(c(), stores[s].type, kWidth, kCells), before);
    auto* store = ggml_set_rows(c(), cache, v, index);
    TensorArena::Bind(store, reinterpret_cast<std::uintptr_t>(cache->data));
    nodes.push_back(store);
    caches.push_back(cache);
    wants.push_back(std::move(want));
  }
  Launched(kg::SetRowsQuantGroup(launch(), nodes), "the grouped quantizing store");
  for (std::size_t s = 0; s < stores.size(); ++s) {
    EXPECT_TRUE(Download<std::byte>(caches[s]) == wants[s]) << "store " << s;
  }
  // A group of one, or of overlapping stores, is refused before any launch.
  EXPECT_EQ(FailedCode(kg::SetRowsQuantGroup(launch(), std::span(nodes).first(1))),
            KernelError::kRejected);
  const std::array<ggml_tensor*, 2> twice = {nodes[0], nodes[0]};
  EXPECT_EQ(FailedCode(kg::SetRowsQuantGroup(launch(), twice)), KernelError::kRejected);
}

struct RopeCase {
  std::int64_t head = 0;
  std::int64_t heads = 0;
  std::int64_t tokens = 0;
  int n_dims = 0;
  bool factors = false;
  float freq_scale = 1.0f;
  float ext_factor = 0.0f;
};

TEST_F(KvQuantTest, RopeStoreWritesGgmlsRopeRowsAsTheStoreWould) {
  const std::array<RopeCase, 4> cases = {{
      {.head = 256, .heads = 4, .tokens = 1, .n_dims = 256},
      {.head = 256, .heads = 4, .tokens = 37, .n_dims = 256, .factors = true},
      {.head = 64, .heads = 2, .tokens = 19, .n_dims = 64},
      // Partial rotation with YaRN: the rest of each head passes through.
      {.head = 64, .heads = 2, .tokens = 23, .n_dims = 32, .freq_scale = 0.25f, .ext_factor = 1.0f},
  }};
  constexpr std::int64_t kCells = 1024;
  std::uint64_t seed = 700;
  for (const ggml_type type : {GGML_TYPE_Q8_0, GGML_TYPE_Q4_0}) {
    for (const RopeCase& rc : cases) {
      SCOPED_TRACE(std::format("{} head {} tokens {} dims {}", ggml_type_name(type), rc.head,
                               rc.tokens, rc.n_dims));
      const std::int64_t width = rc.head * rc.heads;
      auto* x = Place(ggml_new_tensor_3d(c(), GGML_TYPE_F32, rc.head, rc.heads, rc.tokens),
                      Normal(++seed, N(width * rc.tokens), 3.0f));
      std::vector<std::int32_t> positions(N(rc.tokens));
      for (std::int64_t t = 0; t < rc.tokens; ++t) {
        positions[N(t)] = static_cast<std::int32_t>(100000 + (t * 977) % 30000);
      }
      auto* pos = Place(ggml_new_tensor_1d(c(), GGML_TYPE_I32, rc.tokens), positions);
      ggml_tensor* factors = nullptr;
      if (rc.factors) {
        std::vector<float> f(N(rc.n_dims / 2));
        for (std::size_t i = 0; i < f.size(); ++i) f[i] = 1.0f + (0.37f * static_cast<float>(i));
        factors = Place(ggml_new_tensor_1d(c(), GGML_TYPE_F32, rc.n_dims / 2), f);
      }
      auto* rope = ggml_rope_ext(c(), x, pos, factors, rc.n_dims, GGML_ROPE_TYPE_NEOX, 4096,
                                 1000000.0f, rc.freq_scale, rc.ext_factor, 1.0f, 32.0f, 1.0f);
      Place(rope);
      Launched(kg::Rope(launch(), rope), "GGML's RoPE");
      const auto rotated = Download(rope);
      std::vector<std::int64_t> ids(N(rc.tokens));
      for (std::int64_t t = 0; t < rc.tokens; ++t) ids[N(t)] = (t * 131 + 5) % kCells;
      auto* index = Place(ggml_new_tensor_1d(c(), GGML_TYPE_I64, rc.tokens), ids);
      const auto row_bytes = ggml_row_size(type, width);
      const auto before = Pattern(row_bytes * N(kCells));
      auto want = before;
      for (std::int64_t t = 0; t < rc.tokens; ++t) {
        const auto row = EncodeRow(type, std::span(rotated).subspan(N(t * width), N(width)));
        std::ranges::copy(row,
                          want.begin() + static_cast<std::ptrdiff_t>(N(ids[N(t)]) * row_bytes));
      }
      auto* cache = Place(ggml_new_tensor_2d(c(), type, width, kCells), before);
      auto* flat = ggml_view_2d(c(), rope, width, rc.tokens, rope->nb[2], 0);
      TensorArena::Bind(flat, reinterpret_cast<std::uintptr_t>(rope->data));
      auto* store = ggml_set_rows(c(), cache, flat, index);
      TensorArena::Bind(store, reinterpret_cast<std::uintptr_t>(cache->data));
      ASSERT_TRUE(kg::CheckRopeSetRowsQuant(rope, store));
      Launched(kg::RopeSetRowsQuant(launch(), rope, store), "the RoPE quantizing store");
      const auto got = Download<std::byte>(cache);
      EXPECT_EQ(DifferingBlocks(got, want, ggml_type_size(type)), 0U);
      EXPECT_TRUE(got == want);
    }
  }
}

// Blocks whose values or scale lie below F32's normal range (kv_quant.h):
// the reference's reciprocal scale may overflow (its integer conversion of
// an infinite product is then undefined in C) and the fast-math RoPE store
// flushes subnormal operands, so their codes may differ; every block that
// differs stores an F16 zero scale on both sides and decodes to zeros.
TEST_F(KvQuantTest, BlocksBelowTheNormalRangeDifferOnlyWithAZeroScale) {
  constexpr std::int64_t kHead = 64;
  constexpr std::int64_t kHeads = 2;
  constexpr std::int64_t kWidth = kHead * kHeads;
  constexpr std::int64_t kTokens = 14;
  constexpr std::int64_t kCells = 32;
  // Token t's values at about 2^-(96 + 4t): normal, scales at the edge of
  // the normal range, subnormal values, and blocks of zeros and subnormals.
  std::vector<float> values = Normal(1300, N(kWidth * kTokens));
  for (std::int64_t t = 0; t < kTokens; ++t) {
    for (std::int64_t i = 0; i < kWidth; ++i) {
      float& v = values[N((t * kWidth) + i)];
      v = std::ldexp(v, -static_cast<int>(96 + (4 * t)));
    }
  }
  std::vector<std::int64_t> ids(N(kTokens));
  for (std::int64_t t = 0; t < kTokens; ++t) ids[N(t)] = (t * 7 + 3) % kCells;
  const auto scale_is_zero = [](std::span<const std::byte> bytes, std::size_t at) {
    std::uint16_t scale = 0;
    std::memcpy(&scale, bytes.data() + at, sizeof(scale));
    return (scale & 0x7FFFU) == 0;
  };
  // Every differing block's scales are F16 zeros; returns how many differ.
  const auto zero_where_different = [&](std::span<const std::byte> got,
                                        std::span<const std::byte> want, std::size_t block_bytes) {
    std::size_t differing = 0;
    EXPECT_EQ(got.size(), want.size());
    for (std::size_t b = 0; b + block_bytes <= std::min(got.size(), want.size());
         b += block_bytes) {
      if (std::memcmp(got.data() + b, want.data() + b, block_bytes) == 0) continue;
      ++differing;
      EXPECT_TRUE(scale_is_zero(got, b) && scale_is_zero(want, b)) << "block at byte " << b;
    }
    return differing;
  };
  for (const ggml_type type : {GGML_TYPE_Q8_0, GGML_TYPE_Q4_0}) {
    SCOPED_TRACE(ggml_type_name(type));
    const std::size_t block_bytes = ggml_type_size(type);
    const auto row_bytes = ggml_row_size(type, kWidth);
    const auto before = Pattern(row_bytes * N(kCells));
    const auto expected = [&](const std::vector<float>& rows) {
      auto want = before;
      for (std::int64_t t = 0; t < kTokens; ++t) {
        const auto row = EncodeRow(type, std::span(rows).subspan(N(t * kWidth), N(kWidth)));
        std::ranges::copy(row,
                          want.begin() + static_cast<std::ptrdiff_t>(N(ids[N(t)]) * row_bytes));
      }
      return want;
    };
    auto* index = Place(ggml_new_tensor_1d(c(), GGML_TYPE_I64, kTokens), ids);

    // The plain store.
    auto* v = Place(ggml_new_tensor_2d(c(), GGML_TYPE_F32, kWidth, kTokens), values);
    auto* cache = Place(ggml_new_tensor_2d(c(), type, kWidth, kCells), before);
    auto* store = ggml_set_rows(c(), cache, v, index);
    TensorArena::Bind(store, reinterpret_cast<std::uintptr_t>(cache->data));
    Launched(kg::SetRowsQuant(launch(), store), "the quantizing store");
    const std::size_t plain =
        zero_where_different(Download<std::byte>(cache), expected(values), block_bytes);

    // The RoPE store, half of each head rotated (the rest passes through,
    // subnormal values unchanged), against GGML's RoPE and the reference.
    auto* x = Place(ggml_new_tensor_3d(c(), GGML_TYPE_F32, kHead, kHeads, kTokens), values);
    std::vector<std::int32_t> positions(N(kTokens));
    for (std::int64_t t = 0; t < kTokens; ++t) positions[N(t)] = static_cast<std::int32_t>(t * 31);
    auto* pos = Place(ggml_new_tensor_1d(c(), GGML_TYPE_I32, kTokens), positions);
    auto* rope = ggml_rope_ext(c(), x, pos, nullptr, kHead / 2, GGML_ROPE_TYPE_NEOX, 4096, 10000.0f,
                               1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    Place(rope);
    Launched(kg::Rope(launch(), rope), "GGML's RoPE");
    const auto rotated = Download(rope);
    auto* rope_cache = Place(ggml_new_tensor_2d(c(), type, kWidth, kCells), before);
    auto* flat = ggml_view_2d(c(), rope, kWidth, kTokens, rope->nb[2], 0);
    TensorArena::Bind(flat, reinterpret_cast<std::uintptr_t>(rope->data));
    auto* rope_store = ggml_set_rows(c(), rope_cache, flat, index);
    TensorArena::Bind(rope_store, reinterpret_cast<std::uintptr_t>(rope_cache->data));
    Launched(kg::RopeSetRowsQuant(launch(), rope, rope_store), "the RoPE quantizing store");
    const std::size_t fused =
        zero_where_different(Download<std::byte>(rope_cache), expected(rotated), block_bytes);
    std::cout << std::format(
        "{}: {} plain and {} RoPE-store blocks of {} differ, each a zero scale\n",
        ggml_type_name(type), plain, fused, kWidth / 32 * kTokens);
  }
}

struct AttentionCase {
  std::int64_t head = 0;
  std::int64_t heads = 0;
  std::int64_t kv_heads = 0;
  std::int64_t rows = 0;
  std::int64_t cells = 0;
  float softcap = 0.0f;
};

TEST_F(KvQuantTest, AttentionMatchesTheReferenceExecutorAtDecodeAndPrefillShapes) {
  const std::array<AttentionCase, 12> cases = {{
      {.head = 64, .heads = 14, .kv_heads = 2, .rows = 1, .cells = 512},
      // No GQA; and from 1,024 odd rows, the pre-pass reading a padding row.
      {.head = 64, .heads = 2, .kv_heads = 2, .rows = 1, .cells = 256},
      {.head = 64, .heads = 2, .kv_heads = 2, .rows = 1025, .cells = 1280},
      {.head = 256, .heads = 4, .kv_heads = 4, .rows = 3, .cells = 512, .softcap = 30.0f},
      {.head = 64, .heads = 14, .kv_heads = 2, .rows = 2, .cells = 512},
      {.head = 64, .heads = 14, .kv_heads = 2, .rows = 33, .cells = 768},
      {.head = 64, .heads = 4, .kv_heads = 2, .rows = 1024, .cells = 1280},  // the pre-pass
      {.head = 256, .heads = 8, .kv_heads = 4, .rows = 1, .cells = 1024},
      {.head = 256, .heads = 8, .kv_heads = 4, .rows = 2, .cells = 512, .softcap = 50.0f},
      {.head = 256, .heads = 8, .kv_heads = 4, .rows = 1, .cells = 512, .softcap = 30.0f},
      {.head = 256, .heads = 8, .kv_heads = 4, .rows = 17, .cells = 512},
      {.head = 256, .heads = 16, .kv_heads = 2, .rows = 1, .cells = 2048},
  }};
  std::uint64_t seed = 900;
  for (const ggml_type type : {GGML_TYPE_Q8_0, GGML_TYPE_Q4_0}) {
    for (const AttentionCase& ac : cases) {
      SCOPED_TRACE(std::format("{} D {} heads {}/{} rows {} cells {} softcap {}",
                               ggml_type_name(type), ac.head, ac.heads, ac.kv_heads, ac.rows,
                               ac.cells, ac.softcap));
      const std::int64_t d = ac.head;
      const auto row_bytes = ggml_row_size(type, d);
      // K and V: each (KV head, cell) row quantized once; the device cache
      // holds a cell's heads together ([D · KV heads, cells]), the
      // reference's tensor each head's cells together ([D, cells, heads]).
      std::vector<std::byte> k_cache(row_bytes * N(ac.kv_heads * ac.cells));
      std::vector<std::byte> v_cache(k_cache.size());
      std::vector<std::byte> k_ref(k_cache.size());
      std::vector<std::byte> v_ref(k_cache.size());
      const auto keys = Normal(++seed, N(d * ac.kv_heads * ac.cells));
      const auto vals = Normal(++seed, N(d * ac.kv_heads * ac.cells));
      for (std::int64_t h = 0; h < ac.kv_heads; ++h) {
        for (std::int64_t cell = 0; cell < ac.cells; ++cell) {
          const std::size_t at = N(((h * ac.cells) + cell) * d);
          const auto k = EncodeRow(type, std::span(keys).subspan(at, N(d)));
          const auto v = EncodeRow(type, std::span(vals).subspan(at, N(d)));
          const std::size_t device = ((N(cell) * N(ac.kv_heads)) + N(h)) * row_bytes;
          const std::size_t host = ((N(h) * N(ac.cells)) + N(cell)) * row_bytes;
          std::ranges::copy(k, k_cache.begin() + static_cast<std::ptrdiff_t>(device));
          std::ranges::copy(v, v_cache.begin() + static_cast<std::ptrdiff_t>(device));
          std::ranges::copy(k, k_ref.begin() + static_cast<std::ptrdiff_t>(host));
          std::ranges::copy(v, v_ref.begin() + static_cast<std::ptrdiff_t>(host));
        }
      }
      const auto queries = Normal(++seed, N(d * ac.rows * ac.heads));
      // Causal over the filled cells: row r at position filled - rows + r.
      const std::int64_t mask_rows = ac.rows + (ac.rows % 2);
      const std::int64_t filled = ac.cells - 37;
      std::vector<float> mask(N(ac.cells * mask_rows), -std::numeric_limits<float>::infinity());
      for (std::int64_t r = 0; r < mask_rows; ++r) {
        const std::int64_t position = std::min(filled - 1, filled - ac.rows + r);
        ASSERT_GE(position, 0);  // every row sees a cell (a row that sees none is 0 / 0)
        for (std::int64_t cell = 0; cell <= position; ++cell) {
          mask[N((r * ac.cells) + cell)] = (cell % 5 == 1) ? -0.75f : 0.0f;
        }
      }
      const float scale = 1.0f / std::sqrt(static_cast<float>(d));

      // The reference executor.
      ir::Builder b;
      const auto shape = [](std::initializer_list<std::uint64_t> ne) {
        return llmp::format::ShapeOf(std::vector<std::uint64_t>(ne)).value();
      };
      const ir::TensorId q_id = b.Input(
          ir::Dense(Encoding::kF32,
                    shape({std::uint64_t(d), std::uint64_t(ac.rows), std::uint64_t(ac.heads)})));
      const llmp::format::TensorFormat kv_format{
          .encoding = EncodingOf(type),
          .layout = Layout::kGgmlBlocks,
          .shape = shape({std::uint64_t(d), std::uint64_t(ac.cells), std::uint64_t(ac.kv_heads)})};
      const ir::TensorId k_id = b.Input(kv_format);
      const ir::TensorId v_id = b.Input(kv_format);
      const ir::TensorId mask_id = b.Input(
          ir::Dense(Encoding::kF16, shape({std::uint64_t(ac.cells), std::uint64_t(mask_rows)})));
      const ir::TensorId out_id = b.Attention(
          q_id, k_id, v_id, mask_id, {}, ir::AttnParams{.scale = scale, .softcap = ac.softcap});
      b.Output(out_id);
      auto graph = std::move(b).Finish();
      ASSERT_TRUE(graph.has_value());
      auto executor = ref::Executor::Create(*graph);
      ASSERT_TRUE(executor.has_value());
      ASSERT_TRUE(executor->BindValues(q_id, queries).has_value());
      const ref::PartView k_part{.bytes = k_ref};
      const ref::PartView v_part{.bytes = v_ref};
      ASSERT_TRUE(executor->Bind(k_id, std::span(&k_part, 1)).has_value());
      ASSERT_TRUE(executor->Bind(v_id, std::span(&v_part, 1)).has_value());
      ASSERT_TRUE(executor->BindValues(mask_id, mask).has_value());
      ASSERT_TRUE(executor->Run().has_value());
      const auto want = executor->Values(out_id).value();

      // The device.
      auto* q = Place(ggml_new_tensor_3d(c(), GGML_TYPE_F32, d, ac.rows, ac.heads), queries);
      auto* k_root = Place(ggml_new_tensor_2d(c(), type, d * ac.kv_heads, ac.cells), k_cache);
      auto* v_root = Place(ggml_new_tensor_2d(c(), type, d * ac.kv_heads, ac.cells), v_cache);
      auto* k = ggml_view_3d(c(), k_root, d, ac.cells, ac.kv_heads, k_root->nb[1], row_bytes, 0);
      auto* v = ggml_view_3d(c(), v_root, d, ac.cells, ac.kv_heads, v_root->nb[1], row_bytes, 0);
      TensorArena::Bind(k, reinterpret_cast<std::uintptr_t>(k_root->data));
      TensorArena::Bind(v, reinterpret_cast<std::uintptr_t>(v_root->data));
      std::vector<ggml_fp16_t> mask_bits(mask.size());
      for (std::size_t i = 0; i < mask.size(); ++i) mask_bits[i] = ggml_fp32_to_fp16(mask[i]);
      auto* m = Place(ggml_new_tensor_2d(c(), GGML_TYPE_F16, ac.cells, mask_rows), mask_bits);
      auto* node = ggml_flash_attn_ext(c(), q, k, v, m, scale, 0.0f, ac.softcap);
      ggml_prec_set_acc(node, GGML_PREC_F32);
      Place(node);
      ASSERT_TRUE(kg::CheckFlashAttnVecQuant(node));
      const auto plan = kg::PlanFlashAttnVecQuant(launch(), node);
      ASSERT_TRUE(plan.has_value());
      EXPECT_EQ(plan->columns_per_block, ac.rows == 1 ? 1 : 2);
      EXPECT_EQ(plan->mask_prepass, ac.rows >= 1024);
      Launched(kg::FlashAttnVecQuant(launch(), node), "the quantized vector attention");
      const auto got = Download(node);
      const double nmse = Nmse(got, want);
      EXPECT_LE(nmse, kFlashAttnNmse);
      EXPECT_TRUE(std::ranges::all_of(got, [](float x) { return std::isfinite(x); }));
      std::cout << std::format("{} D {} heads {}/{} rows {} cells {} softcap {}: NMSE {:.3e}\n",
                               ggml_type_name(type), ac.head, ac.heads, ac.kv_heads, ac.rows,
                               ac.cells, ac.softcap, nmse);
    }
  }
}

TEST_F(KvQuantTest, ChecksRefuseWhatTheKernelsWouldNotTake) {
  const auto attention = [&](ggml_type k_type, ggml_type v_type, std::int64_t d, std::int64_t cells,
                             float softcap) {
    auto* q = Place(ggml_new_tensor_3d(c(), GGML_TYPE_F32, d, 1, 4));
    auto* k = Place(ggml_new_tensor_3d(c(), k_type, d, cells, 2));
    auto* v = Place(ggml_new_tensor_3d(c(), v_type, d, cells, 2));
    auto* m = Place(ggml_new_tensor_2d(c(), GGML_TYPE_F16, cells, 1));
    auto* node = ggml_flash_attn_ext(c(), q, k, v, m, 0.125f, 0.0f, softcap);
    ggml_prec_set_acc(node, GGML_PREC_F32);
    return Place(node);
  };
  EXPECT_TRUE(kg::CheckFlashAttnVecQuant(attention(GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, 64, 256, 0)));
  EXPECT_TRUE(kg::CheckFlashAttnVecQuant(attention(GGML_TYPE_Q4_0, GGML_TYPE_Q4_0, 256, 256, 9)));
  // Mixed encodings, an F16 V, D 128, softcap at D 64, unpadded cells.
  for (ggml_tensor* node : {attention(GGML_TYPE_Q8_0, GGML_TYPE_Q4_0, 64, 256, 0),
                            attention(GGML_TYPE_Q8_0, GGML_TYPE_F16, 64, 256, 0),
                            attention(GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, 128, 256, 0),
                            attention(GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, 64, 256, 5),
                            attention(GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, 64, 320, 0)}) {
    EXPECT_FALSE(kg::CheckFlashAttnVecQuant(node));
    EXPECT_EQ(FailedCode(kg::FlashAttnVecQuant(launch(), node)), KernelError::kRejected);
  }
  // A K view whose head stride is not whole blocks.
  {
    auto* node = attention(GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, 64, 256, 0);
    auto* k = node->src[1];
    k->nb[2] += 2;
    EXPECT_FALSE(kg::CheckFlashAttnVecQuant(node));
    k->nb[2] -= 2;
  }
  // Sinks: one F32 sink per query head is taken (the vector kernel reads
  // them, fattn-vec.cuh); another count or encoding is not.
  {
    auto* node = attention(GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, 64, 256, 0);
    auto* sinks = Place(ggml_new_tensor_1d(c(), GGML_TYPE_F32, 4));
    ggml_flash_attn_ext_add_sinks(node, sinks);
    EXPECT_TRUE(kg::CheckFlashAttnVecQuant(node));
  }
  for (const auto& [type, count] :
       {std::pair{GGML_TYPE_F32, std::int64_t{3}}, std::pair{GGML_TYPE_F16, std::int64_t{4}}}) {
    auto* node = attention(GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, 64, 256, 0);
    // ggml_flash_attn_ext_add_sinks asserts both, so attach them directly.
    node->src[4] = Place(ggml_new_tensor_1d(c(), type, count));
    EXPECT_FALSE(kg::CheckFlashAttnVecQuant(node));
    EXPECT_EQ(FailedCode(kg::FlashAttnVecQuant(launch(), node)), KernelError::kRejected);
  }
  // D 512 is compiled for Q8_0 (Gemma 4's global layers), not for Q4_0.
  EXPECT_TRUE(kg::CheckFlashAttnVecQuant(attention(GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, 512, 256, 0)));
  {
    auto* node = attention(GGML_TYPE_Q4_0, GGML_TYPE_Q4_0, 512, 256, 0);
    EXPECT_FALSE(kg::CheckFlashAttnVecQuant(node));
    EXPECT_EQ(FailedCode(kg::FlashAttnVecQuant(launch(), node)), KernelError::kRejected);
  }

  const auto store = [&](ggml_type values, ggml_type dest, ggml_type ids) {
    auto* v = Place(ggml_new_tensor_2d(c(), values, 64, 2));
    auto* cache = Place(ggml_new_tensor_2d(c(), dest, 64, 16));
    auto* index = Place(ggml_new_tensor_1d(c(), ids, 2));
    auto* node = ggml_set_rows(c(), cache, v, index);
    TensorArena::Bind(node, reinterpret_cast<std::uintptr_t>(cache->data));
    return node;
  };
  EXPECT_TRUE(kg::CheckSetRowsQuant(store(GGML_TYPE_F32, GGML_TYPE_Q8_0, GGML_TYPE_I32)));
  for (ggml_tensor* node : {store(GGML_TYPE_F16, GGML_TYPE_Q8_0, GGML_TYPE_I64),
                            store(GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_I64),
                            store(GGML_TYPE_F32, GGML_TYPE_Q5_0, GGML_TYPE_I64)}) {
    EXPECT_FALSE(kg::CheckSetRowsQuant(node));
    EXPECT_EQ(FailedCode(kg::SetRowsQuant(launch(), node)), KernelError::kRejected);
  }
  // A destination overlapping its rows.
  {
    auto* node = store(GGML_TYPE_F32, GGML_TYPE_Q4_0, GGML_TYPE_I64);
    TensorArena::Bind(node->src[0], reinterpret_cast<std::uintptr_t>(node->data));
    EXPECT_FALSE(kg::CheckSetRowsQuant(node));
  }

  const auto rope_store = [&](int mode, ggml_type dest, std::int64_t head) {
    auto* x = Place(ggml_new_tensor_3d(c(), GGML_TYPE_F32, head, 2, 3));
    auto* pos = Place(ggml_new_tensor_1d(c(), GGML_TYPE_I32, 3));
    auto* rope = ggml_rope_ext(c(), x, pos, nullptr, static_cast<int>(head), mode, 4096, 10000.0f,
                               1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    Place(rope);
    auto* flat = ggml_view_2d(c(), rope, head * 2, 3, rope->nb[2], 0);
    TensorArena::Bind(flat, reinterpret_cast<std::uintptr_t>(rope->data));
    auto* cache = Place(ggml_new_tensor_2d(c(), dest, head * 2, 64));
    auto* index = Place(ggml_new_tensor_1d(c(), GGML_TYPE_I64, 3));
    auto* node = ggml_set_rows(c(), cache, flat, index);
    TensorArena::Bind(node, reinterpret_cast<std::uintptr_t>(cache->data));
    return std::pair{rope, node};
  };
  {
    const auto [rope, node] = rope_store(GGML_ROPE_TYPE_NEOX, GGML_TYPE_Q8_0, 64);
    EXPECT_TRUE(kg::CheckRopeSetRowsQuant(rope, node));
  }
  for (const auto& [rope, node] : {rope_store(GGML_ROPE_TYPE_NORMAL, GGML_TYPE_Q8_0, 64),
                                   rope_store(GGML_ROPE_TYPE_NEOX, GGML_TYPE_F16, 64),
                                   rope_store(GGML_ROPE_TYPE_NEOX, GGML_TYPE_Q8_0, 80)}) {
    EXPECT_FALSE(kg::CheckRopeSetRowsQuant(rope, node));
    EXPECT_EQ(FailedCode(kg::RopeSetRowsQuant(launch(), rope, node)), KernelError::kRejected);
  }
}

TEST_F(KvQuantTest, TheRegistryDeclaresEachWithItsCapability) {
  namespace ex = llmp::execution;
  const std::vector<ex::Implementation> declared = kg::Implementations();
  const auto registry = ex::Registry::Create(declared).value();
  for (const std::string_view name :
       {kg::kFlashAttnVecQ8Name, kg::kFlashAttnVecQ4Name, kg::kSetRowsQuantName,
        kg::kSetRowsQuantGroupName, kg::kRopeSetRowsQuantName}) {
    const std::size_t index = registry.Find(name).value_or(registry.size());
    ASSERT_LT(index, registry.size()) << name;
    const ex::Implementation& implementation = registry.at(index);
    ASSERT_TRUE(implementation.capability.has_value()) << name;
    EXPECT_EQ(implementation.capability->rung, ex::Rung::kNative);
    EXPECT_EQ(implementation.capability->expansion, ex::Expansion::kNone);
    EXPECT_FALSE(implementation.capability->lossy);
    const auto kernel = kg::Kernel::Bind(implementation);
    ASSERT_TRUE(kernel.has_value()) << name;
    EXPECT_EQ(kernel->name(), name);
  }
  const auto arch = static_cast<std::uint32_t>(ComputeCapability());
  const auto names = [&](const ex::CapabilityQuery& query) {
    std::vector<std::string> out;
    for (const std::size_t i : ex::Candidates(registry, query)) {
      out.push_back(registry.at(i).name);
    }
    return out;
  };
  using Of = std::vector<std::string>;
  const auto dense = [](std::uint8_t operand, Encoding e) {
    return ex::OperandFormat{.operand = operand, .encoding = e, .layout = Layout::kDense};
  };
  const auto blocks = [](std::uint8_t operand, Encoding e) {
    return ex::OperandFormat{.operand = operand, .encoding = e, .layout = Layout::kGgmlBlocks};
  };
  const auto attention = [&](Encoding kv, std::uint64_t d, std::uint64_t rows, std::uint32_t on) {
    const std::array<ex::OperandFormat, 4> operands = {dense(0, Encoding::kF32), blocks(1, kv),
                                                       blocks(2, kv), dense(3, Encoding::kF16)};
    return names({.op = ir::Primitive(ir::Op::kAttention),
                  .operands = operands,
                  .rows = rows,
                  .k = d,
                  .n = d,
                  .arch = on,
                  .capturable = true});
  };
  // Decode and prefill rows alike: nothing expands.
  for (const std::uint64_t rows : {1U, 2U, 512U}) {
    EXPECT_EQ(attention(Encoding::kQ8_0, 256, rows, arch),
              Of{std::string(kg::kFlashAttnVecQ8Name)});
    EXPECT_EQ(attention(Encoding::kQ4_0, 64, rows, arch), Of{std::string(kg::kFlashAttnVecQ4Name)});
  }
  EXPECT_EQ(attention(Encoding::kQ8_0, 128, 1, arch), Of{});
  // D 512 (Gemma 4's global layers) is compiled for Q8_0 alone.
  EXPECT_EQ(attention(Encoding::kQ8_0, 512, 1, arch), Of{std::string(kg::kFlashAttnVecQ8Name)});
  EXPECT_EQ(attention(Encoding::kQ4_0, 512, 1, arch), Of{});
  EXPECT_EQ(attention(Encoding::kQ5_0, 256, 1, arch), Of{});
  EXPECT_EQ(attention(Encoding::kQ8_0, 256, 1, 90), Of{});  // no SASS for sm_90
  // F32 sinks are taken; sinks in another encoding are not.
  for (const auto& [sink, want] :
       {std::pair{Encoding::kF32, Of{std::string(kg::kFlashAttnVecQ8Name)}},
        std::pair{Encoding::kF16, Of{}}}) {
    const std::array<ex::OperandFormat, 5> with_sinks = {
        dense(0, Encoding::kF32), blocks(1, Encoding::kQ8_0), blocks(2, Encoding::kQ8_0),
        dense(3, Encoding::kF16), dense(4, sink)};
    EXPECT_EQ(names({.op = ir::Primitive(ir::Op::kAttention),
                     .operands = with_sinks,
                     .rows = 1,
                     .k = 256,
                     .n = 256,
                     .arch = arch}),
              want);
  }
  const std::array<ex::OperandFormat, 3> store = {
      blocks(0, Encoding::kQ4_0), dense(1, Encoding::kF32), dense(2, Encoding::kI32)};
  EXPECT_EQ(names({.op = ir::Primitive(ir::Op::kCacheStore),
                   .operands = store,
                   .rows = 7,
                   .k = 1024,
                   .arch = arch,
                   .row_invariant = true}),
            Of{std::string(kg::kSetRowsQuantName)});
  EXPECT_EQ(names({.op = ir::Primitive(ir::Op::kCacheStore),
                   .operands = store,
                   .rows = 7,
                   .k = 48,
                   .arch = arch}),
            Of{});
  // The grouped store is a batch: a member store's query under the
  // grouped_stores key, two to sixteen of them (the planner's batch).
  const std::array<ex::OperandFormat, 3> member = {
      blocks(0, Encoding::kQ8_0), dense(1, Encoding::kF32), dense(2, Encoding::kI64)};
  EXPECT_EQ(names({.op = ir::Fused(ir::FusedKind::kGroupedStores),
                   .operands = member,
                   .rows = 3,
                   .k = 1024,
                   .arch = arch}),
            Of{std::string(kg::kSetRowsQuantGroupName)});
  EXPECT_EQ(registry.at(registry.Find(kg::kSetRowsQuantGroupName).value()).capability->batch, 16U);
  const std::array<ex::OperandFormat, 5> rope = {
      dense(0, Encoding::kF32), dense(1, Encoding::kI32), blocks(2, Encoding::kQ8_0),
      dense(3, Encoding::kI64), dense(4, Encoding::kF32)};
  EXPECT_EQ(names({.op = ir::Fused(ir::FusedKind::kRopeStore),
                   .operands = rope,
                   .rows = 1,
                   .k = 1024,
                   .arch = arch}),
            Of{std::string(kg::kRopeSetRowsQuantName)});
  const std::array<ex::OperandFormat, 4> f16_rope = {
      dense(0, Encoding::kF32), dense(1, Encoding::kI32), dense(2, Encoding::kF16),
      dense(3, Encoding::kI64)};
  EXPECT_EQ(names({.op = ir::Fused(ir::FusedKind::kRopeStore),
                   .operands = f16_rope,
                   .rows = 1,
                   .k = 1024,
                   .arch = arch}),
            Of{});
}

}  // namespace
