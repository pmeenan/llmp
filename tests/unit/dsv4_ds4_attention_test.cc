// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// Operator controls for a temporary complete original reference. These
// analytical operands are not a model quality/performance comparison.
#include "kernels/ggml/dsv4_ds4_attention.h"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <numeric>
#include <span>
#include <thread>
#include <utility>
#include <vector>

#include "base/bytes.h"
#include "kernels/ggml/launch.h"
#include "providers/cuda/cuda_device_execution.h"
#include "providers/device_execution.h"

namespace {
namespace kg = jitllm::kernels::ggml;
namespace pd = jitllm::providers;
constexpr std::size_t kGuardBytes = 256;
constexpr std::size_t kHeads = 64;
constexpr std::size_t kAttentionDim = 512;
constexpr std::size_t kIndexerDim = 128;

class Ds4AttentionTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto opened = pd::cuda::OpenDeviceExecution(0);
    ASSERT_TRUE(opened);
    execution_ = std::move(*opened);
    const auto stream = execution_->CreateStream();
    ASSERT_TRUE(stream);
    stream_ = *stream;
    auto created =
        kg::LaunchContext::Create(0, *execution_, stream_, {.size = jitllm::base::Bytes(0)});
    ASSERT_TRUE(created) << (created ? "" : created.error().detail);
    launch_ = std::move(*created);
    ASSERT_TRUE(kg::PrepareDs4Indexer(launch()));
    ASSERT_TRUE(kg::PrepareDs4Attention(launch()));
    Finish();
  }

  void TearDown() override {
    if (stream_.valid()) Finish();
    for (const auto& allocation : allocations_) Guard(allocation);
    if (launch_) {
      EXPECT_EQ(launch_->scratch_peak().value(), 0);
      launch_.reset();
    }
    if (stream_.valid()) EXPECT_TRUE(execution_->DestroyStream(stream_));
    for (const auto& allocation : allocations_)
      EXPECT_EQ(cudaFree(std::bit_cast<void*>(allocation.address)), cudaSuccess);
  }

  // Completion failure aborts while every operand still has an owner.
  void Finish() {
    const auto fence = execution_->Record(stream_);
    if (!fence) std::abort();
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    while (true) {
      const auto state = execution_->Query(*fence);
      if (!state || std::chrono::steady_clock::now() >= deadline) std::abort();
      if (*state == pd::FenceState::kComplete) break;
      std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
    if (!execution_->Release(*fence)) std::abort();
  }

  kg::Ds4CacheBuffer Allocate(std::size_t bytes) {
    void* pointer = nullptr;
    if (bytes == 0 || cudaMalloc(&pointer, bytes + kGuardBytes) != cudaSuccess) std::abort();
    const kg::Ds4CacheBuffer buffer{std::bit_cast<std::uintptr_t>(pointer), bytes};
    allocations_.push_back(buffer);
    if (cudaMemset(pointer, 0xa5, bytes + kGuardBytes) != cudaSuccess) std::abort();
    // Default-stream initialization must finish before the nonblocking consumer.
    if (cudaDeviceSynchronize() != cudaSuccess) std::abort();
    return buffer;
  }

  template <typename T>
  void Write(const kg::Ds4CacheBuffer& buffer, std::span<const T> values) {
    Finish();
    if (values.size_bytes() > buffer.bytes ||
        cudaMemcpy(std::bit_cast<void*>(buffer.address), values.data(), values.size_bytes(),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaDeviceSynchronize() != cudaSuccess)
      std::abort();
  }

  template <typename T>
  kg::Ds4CacheBuffer Upload(std::span<const T> values) {
    const auto buffer = Allocate(values.size_bytes());
    Write<T>(buffer, values);
    return buffer;
  }

  template <typename T>
  std::vector<T> Download(const kg::Ds4CacheBuffer& buffer) {
    Finish();
    std::vector<T> values(static_cast<std::size_t>(buffer.bytes) / sizeof(T));
    if (cudaMemcpy(values.data(), std::bit_cast<const void*>(buffer.address), buffer.bytes,
                   cudaMemcpyDeviceToHost) != cudaSuccess)
      std::abort();
    return values;
  }

  static void Guard(const kg::Ds4CacheBuffer& buffer) {
    std::array<std::uint8_t, kGuardBytes> guard{};
    ASSERT_EQ(cudaMemcpy(guard.data(), std::bit_cast<const void*>(buffer.address + buffer.bytes),
                         guard.size(), cudaMemcpyDeviceToHost),
              cudaSuccess);
    EXPECT_TRUE(std::ranges::all_of(guard, [](auto byte) { return byte == 0xa5; }));
  }

  static void Exact(const std::vector<float>& a, const std::vector<float>& b) {
    ASSERT_EQ(a.size(), b.size());
    EXPECT_TRUE(std::ranges::equal(a, b, [](float x, float y) {
      return std::bit_cast<std::uint32_t>(x) == std::bit_cast<std::uint32_t>(y);
    }));
  }

  template <typename T>
  kg::Ds4CacheBuffer Zeroed() {
    const std::array<T, 1> zero{};
    return Upload<T>(zero);
  }

  kg::LaunchContext& launch() { return *launch_; }

 private:
  std::unique_ptr<pd::DeviceExecution> execution_;
  pd::StreamId stream_;
  std::unique_ptr<kg::LaunchContext> launch_;
  std::vector<kg::Ds4CacheBuffer> allocations_;
};

std::vector<std::uint32_t> SelectReference(std::span<const float> row, std::uint32_t top_k = 512) {
  std::vector<std::uint32_t> ids(row.size());
  std::ranges::iota(ids, 0);
  std::ranges::sort(ids, [row](std::uint32_t a, std::uint32_t b) {
    return row[a] > row[b] || (row[a] == row[b] && a < b);
  });
  ids.resize(top_k, std::numeric_limits<std::uint32_t>::max());
  return ids;
}

TEST_F(Ds4AttentionTest, ExactSelectorsMatchCpuAcrossTiersTiesAndDeepStreaming) {
  constexpr std::array<std::uint32_t, 7> kCells = {513, 1025, 2049, 4096, 4097, 8192, 32769};
  for (const auto cells : kCells) {
    SCOPED_TRACE(cells);
    std::vector<float> scores(2ULL * cells);
    for (std::uint32_t t = 0; t < 2; ++t) {
      for (std::uint32_t c = 0; c < cells; ++c) {
        // Rising/falling rows, ties, and actual -infinity causal cells.
        const auto finite_score = static_cast<float>(t == 0 ? c % 53 : (cells - c) % 53);
        scores[(std::uint64_t{t} * cells) + c] =
            c % 11 == 0 ? -std::numeric_limits<float>::infinity() : finite_score;
      }
    }
    kg::Ds4IndexerSelect desc{.scores = Upload<float>(scores),
                              .selected = Allocate(2ULL * 512ULL * 4ULL),
                              .diagnostics = Zeroed<kg::Ds4IndexerDiagnostics>(),
                              .tokens = 2,
                              .cells = cells,
                              .score_band = cells};
    ASSERT_TRUE(kg::RunDs4IndexerSelect(launch(), desc));
    const auto first = Download<std::uint32_t>(desc.selected);
    ASSERT_TRUE(kg::RunDs4IndexerSelect(launch(), desc));
    EXPECT_EQ(first, Download<std::uint32_t>(desc.selected));
    for (std::uint32_t t = 0; t < 2; ++t) {
      const auto expected = SelectReference(
          std::span<const float>(scores).subspan(static_cast<std::size_t>(t) * cells, cells));
      EXPECT_TRUE(std::ranges::equal(std::span<const std::uint32_t>(first).subspan(t * 512ULL, 512),
                                     expected));
    }
    if (cells > 8192) {
      desc.kind = kg::Ds4IndexerSelectKind::kChunkTree;
      const auto bytes = kg::Ds4IndexerSelectScratchBytes(desc);
      ASSERT_TRUE(bytes);
      desc.scratch = Allocate(*bytes);
      ASSERT_TRUE(kg::RunDs4IndexerSelect(launch(), desc));
      EXPECT_EQ(first, Download<std::uint32_t>(desc.selected));
    }
    EXPECT_EQ(Download<kg::Ds4IndexerDiagnostics>(desc.diagnostics)[0].bound_count, 0);
  }
}

double ScoreReference(std::span<const float> query, std::span<const float> weights,
                      std::span<const float> keys, std::uint32_t t, std::uint32_t c,
                      std::uint32_t bank, std::uint32_t capacity, double scale) {
  double total = 0;
  for (std::uint32_t h = 0; h < kHeads; ++h) {
    double dot = 0;
    for (std::uint32_t d = 0; d < kIndexerDim; ++d) {
      dot += static_cast<double>(query[(((std::uint64_t{t} * kHeads) + h) * kIndexerDim) + d]) *
             keys[(((std::uint64_t{bank} * capacity) + c) * kIndexerDim) + d];
    }
    total += std::max(dot, 0.0) * weights[(std::uint64_t{t} * kHeads) + h];
  }
  return total * scale;
}

TEST_F(Ds4AttentionTest, ScalarAndPerBankScoresReadOriginalPackedQatOperands) {
  constexpr std::size_t kTokens = 4;
  constexpr std::size_t kCells = 33;
  std::vector<float> query(kTokens * kHeads * kIndexerDim);
  std::vector<float> keys(2ULL * kCells * kIndexerDim);
  std::vector<float> weights(kTokens * kHeads);
  for (std::size_t i = 0; i < query.size(); ++i)
    query[i] = static_cast<float>(static_cast<int>(i % 13) - 6) / 64;
  for (std::size_t i = 0; i < keys.size(); ++i)
    keys[i] = static_cast<float>(static_cast<int>((i * 7ULL) % 17) - 8) / 64;
  for (std::size_t i = 0; i < weights.size(); ++i)
    weights[i] = static_cast<float>((i % 7) + 1) / 64;
  const auto q = Upload<float>(query);
  const auto q_scales = Allocate(kTokens * kHeads * 16ULL);
  const auto key = Upload<float>(keys);
  const auto codes = Allocate(2ULL * kCells * 64ULL);
  const auto scales = Allocate(2ULL * kCells * 16ULL);
  ASSERT_TRUE(kg::RunDs4CacheQat(launch(), {.kind = kg::Ds4CacheKind::kIndexer128,
                                            .values = q,
                                            .codes = {},
                                            .scales = q_scales,
                                            .rows = kTokens * kHeads}));
  ASSERT_TRUE(kg::RunDs4CacheQat(launch(), {.kind = kg::Ds4CacheKind::kIndexer128,
                                            .values = key,
                                            .codes = codes,
                                            .scales = scales,
                                            .rows = 2ULL * kCells}));
  const auto rounded_q = Download<float>(q);
  const auto rounded_keys = Download<float>(key);
  kg::Ds4IndexerScores desc{.query = q,
                            .weights = Upload<float>(weights),
                            .keys = key,
                            .key_codes = codes,
                            .key_scales = scales,
                            .query_scales = q_scales,
                            .scores = Allocate(kTokens * kCells * 4ULL),
                            .diagnostics = Zeroed<kg::Ds4IndexerDiagnostics>(),
                            .tokens = kTokens,
                            .cells = kCells,
                            .cells_per_bank = kCells,
                            .banks = 2,
                            .first = 128,
                            .score_band = kCells,
                            .scale = 0.125f,
                            .kind = kg::Ds4IndexerScoreKind::kScalar};
  ASSERT_TRUE(kg::RunDs4IndexerScores(launch(), desc));
  const auto packed = Download<float>(desc.scores);
  auto dense = desc;
  dense.key_codes = {};
  dense.key_scales = {};
  ASSERT_TRUE(kg::RunDs4IndexerScores(launch(), dense));
  Exact(packed, Download<float>(desc.scores));
  for (std::uint32_t t = 0; t < kTokens; ++t) {
    for (std::uint32_t c = 0; c < kCells; ++c) {
      if (c >= (desc.first + t + 1) / desc.ratio) {
        EXPECT_EQ(packed[(t * kCells) + c], -std::numeric_limits<float>::infinity());
      } else {
        const auto expected =
            ScoreReference(rounded_q, weights, rounded_keys, t, c, 0, kCells, desc.scale);
        EXPECT_NEAR(packed[(t * kCells) + c], expected, 1e-7 + (std::abs(expected) * 1e-5));
      }
    }
  }
  const std::array<std::int32_t, kTokens> positions{140, 8, 141, 9};
  const std::array<std::int32_t, kTokens> banks{1, 0, 1, 0};
  desc.positions = Upload<std::int32_t>(positions);
  desc.bank_ids = Upload<std::int32_t>(banks);
  for (const auto kind :
       {kg::Ds4IndexerScoreKind::kMultisequenceScalar, kg::Ds4IndexerScoreKind::kMultisequenceV5d,
        kg::Ds4IndexerScoreKind::kMultisequenceV5e}) {
    SCOPED_TRACE(static_cast<int>(kind));
    desc.kind = kind;
    ASSERT_TRUE(kg::RunDs4IndexerScores(launch(), desc));
    const auto actual = Download<float>(desc.scores);
    for (std::uint32_t t = 0; t < kTokens; ++t) {
      for (std::uint32_t c = 0; c < kCells; ++c) {
        if (c >= static_cast<std::uint32_t>(positions[t]) / desc.ratio) {
          EXPECT_EQ(actual[(t * kCells) + c], -std::numeric_limits<float>::infinity());
        } else {
          const auto expected =
              ScoreReference(rounded_q, weights, rounded_keys, t, c,
                             static_cast<std::uint32_t>(banks[t]), kCells, desc.scale);
          EXPECT_NEAR(actual[(t * kCells) + c], expected, 1e-7 + (std::abs(expected) * 1e-5));
        }
      }
    }
  }
  EXPECT_GT(Download<kg::Ds4IndexerDiagnostics>(desc.diagnostics)[0].packed_reads, 0);
}

TEST_F(Ds4AttentionTest, CaptureScorerAndStreamingSelectorConsumeGrowingLiveBand) {
  constexpr std::size_t kCapacity = 9001;
  std::vector<float> q(kHeads * kIndexerDim, 0.0f);
  std::vector<float> keys(kCapacity * kIndexerDim, 0.0f);
  std::vector<float> weights(kHeads, 1.0f / 64);
  for (std::uint32_t h = 0; h < kHeads; ++h) q[h * kIndexerDim] = 1;
  for (std::uint32_t c = 0; c < kCapacity; ++c)
    keys[c * kIndexerDim] = static_cast<float>(c < 511 ? c % 17 : 32 + (c % 31)) / 8;
  const auto live = Zeroed<kg::Ds4IndexerLayerScalars>();
  kg::Ds4IndexerScores scores{.query = Upload<float>(q),
                              .weights = Upload<float>(weights),
                              .keys = Upload<float>(keys),
                              .layer_scalars = live,
                              .scores = Allocate(kCapacity * 4ULL),
                              .diagnostics = Zeroed<kg::Ds4IndexerDiagnostics>(),
                              .tokens = 1,
                              .cells = 511,
                              .cells_per_bank = kCapacity,
                              .score_band = kCapacity,
                              .causal = false};
  kg::Ds4IndexerSelect select{.scores = scores.scores,
                              .selected = Allocate(512ULL * 4ULL),
                              .layer_scalars = live,
                              .diagnostics = scores.diagnostics,
                              .tokens = 1,
                              .cells = 511,
                              .score_band = kCapacity};
  const auto described = kg::DescribeDs4IndexerDispatch(launch(), scores, select);
  ASSERT_TRUE(described);
  EXPECT_EQ(described->score_kind, kg::Ds4IndexerScoreKind::kDirectOne);
  EXPECT_EQ(described->select_kind, kg::Ds4IndexerSelectKind::kStream512);
  auto graph = launch().Capture([scores, select](auto& context) {
    return kg::RunDs4IndexerScoreSelect(context, scores, select);
  });
  ASSERT_TRUE(graph);
  std::vector<std::uint32_t> previous;
  for (const auto count : {std::uint32_t{511}, std::uint32_t{1025}}) {
    const std::array<kg::Ds4IndexerLayerScalars, 1> scalars{
        kg::Ds4IndexerLayerScalars{.n_index_comp = count}};
    Write<kg::Ds4IndexerLayerScalars>(live, scalars);
    ASSERT_TRUE(launch().Launch(*graph));
    const auto captured_ids = Download<std::uint32_t>(select.selected);
    const auto captured_scores = Download<float>(scores.scores);
    ASSERT_TRUE(kg::RunDs4IndexerScoreSelect(launch(), scores, select));
    EXPECT_EQ(captured_ids, Download<std::uint32_t>(select.selected));
    Exact(captured_scores, Download<float>(scores.scores));
    std::vector<float> reference(count);
    for (std::uint32_t c = 0; c < count; ++c) reference[c] = keys[c * kIndexerDim];
    EXPECT_EQ(captured_ids, SelectReference(reference));
    if (!previous.empty()) EXPECT_NE(previous, captured_ids);
    previous = captured_ids;
  }
  EXPECT_EQ(Download<kg::Ds4IndexerDiagnostics>(scores.diagnostics)[0].bound_count, 0);
  Finish();  // Graph and every owner remain alive until replay retirement.
}

TEST_F(Ds4AttentionTest, Mxf4ProducerMirrorAndRequantizerAgreeOnRaggedTiles) {
  constexpr std::size_t kTokens = 33;
  constexpr std::size_t kCells = 1025;
  std::vector<float> q(kTokens * kHeads * kIndexerDim, 0.0f);
  std::vector<float> keys(kCells * kIndexerDim, 0.0f);
  std::vector<float> weights(kTokens * kHeads, 1.0f / 64);
  for (std::size_t row = 0; row < kTokens * kHeads; ++row)
    q[(row * kIndexerDim) + (row % kIndexerDim)] = 0.125f;
  for (std::size_t row = 0; row < kCells; ++row)
    keys[(row * kIndexerDim) + (row % kIndexerDim)] = static_cast<float>((row % 7) + 1) / 64;
  const auto query = Upload<float>(q);
  const auto q_codes = Allocate(kTokens * kHeads * 64ULL);
  const auto q_scales = Allocate(kTokens * kHeads * 16ULL);
  const auto key = Upload<float>(keys);
  const auto k_codes = Allocate(kCells * 64ULL);
  const auto k_scales = Allocate(kCells * 16ULL);
  ASSERT_TRUE(kg::RunDs4CacheQat(launch(), {.kind = kg::Ds4CacheKind::kIndexer128,
                                            .values = query,
                                            .codes = q_codes,
                                            .scales = q_scales,
                                            .rows = kTokens * kHeads}));
  ASSERT_TRUE(kg::RunDs4CacheQat(launch(), {.kind = kg::Ds4CacheKind::kIndexer128,
                                            .values = key,
                                            .codes = k_codes,
                                            .scales = k_scales,
                                            .rows = kCells}));
  const auto rounded_q = Download<float>(query);
  const auto rounded_keys = Download<float>(key);
  kg::Ds4IndexerScores desc{.query = query,
                            .weights = Upload<float>(weights),
                            .keys = key,
                            .key_codes = k_codes,
                            .key_scales = k_scales,
                            .query_codes = q_codes,
                            .query_scales = q_scales,
                            .scores = Allocate(kTokens * kCells * 4ULL),
                            .diagnostics = Zeroed<kg::Ds4IndexerDiagnostics>(),
                            .tokens = kTokens,
                            .cells = kCells,
                            .cells_per_bank = kCells,
                            .first = 4096,
                            .score_band = kCells,
                            .kind = kg::Ds4IndexerScoreKind::kMxf4};
  const auto bytes = kg::Ds4IndexerScoreScratchBytes(desc);
  ASSERT_TRUE(bytes);
  desc.scratch = Allocate(*bytes);
  ASSERT_TRUE(kg::RunDs4IndexerScores(launch(), desc));
  const auto mirror = Download<float>(desc.scores);
  ASSERT_TRUE(kg::RunDs4IndexerScores(launch(), desc));
  Exact(mirror, Download<float>(desc.scores));
  // The original requantizer consumes the already-rounded Q.
  desc.query_codes = {};
  desc.query_scales = {};
  const auto requant_bytes = kg::Ds4IndexerScoreScratchBytes(desc);
  ASSERT_TRUE(requant_bytes);
  desc.scratch = Allocate(*requant_bytes);
  ASSERT_TRUE(kg::RunDs4IndexerScores(launch(), desc));
  Exact(mirror, Download<float>(desc.scores));
  for (const auto t : {std::uint32_t{0}, std::uint32_t{16}, std::uint32_t{32}}) {
    for (const auto c : {std::uint32_t{0}, std::uint32_t{255}, std::uint32_t{256},
                         std::uint32_t{1023}, std::uint32_t{1024}}) {
      SCOPED_TRACE(::testing::Message() << "token " << t << ", cell " << c);
      // Original MXF4 scores include the causal compressed-cell clamp.
      if (c >= (desc.first + t + 1) / desc.ratio) {
        EXPECT_EQ(mirror[(t * kCells) + c], -std::numeric_limits<float>::infinity());
        continue;
      }
      const auto expected = ScoreReference(rounded_q, weights, rounded_keys, t, c, 0, kCells, 1);
      EXPECT_NEAR(mirror[(t * kCells) + c], expected, 1e-7 + (std::abs(expected) * 1e-5));
    }
  }
}

std::vector<double> AttentionReference(std::span<const float> q, std::span<const float> raw,
                                       std::span<const float> comp, std::span<const float> sinks,
                                       std::span<const std::int32_t> positions,
                                       std::span<const std::int32_t> banks,
                                       std::span<const std::int32_t> selected,
                                       std::uint32_t raw_cells, std::uint32_t comp_cells,
                                       std::uint32_t comp_count, std::uint32_t window,
                                       std::uint32_t top_k) {
  std::vector<double> output(q.size(), 0);
  for (std::uint32_t t = 0; t < positions.size(); ++t) {
    const auto pos = static_cast<std::uint32_t>(positions[t]);
    const auto bank = static_cast<std::uint32_t>(banks[t]);
    const auto raw_count = std::min({pos + 1, window, raw_cells});
    const auto first = pos + 1 - raw_count;
    const auto visible = std::min(pos / 4, comp_count);
    std::vector<std::span<const float>> rows;
    rows.reserve(static_cast<std::size_t>(raw_count) + top_k);
    for (std::uint32_t r = 0; r < raw_count; ++r)
      rows.push_back(
          raw.subspan(((static_cast<std::size_t>(bank) * raw_cells) + ((first + r) % raw_cells)) *
                          kAttentionDim,
                      kAttentionDim));
    for (std::uint32_t slot = 0; slot < top_k; ++slot) {
      const auto id = selected[(t * top_k) + slot];
      if (id >= 0 && std::cmp_less(id, visible))
        rows.push_back(comp.subspan(
            ((static_cast<std::size_t>(bank) * comp_cells) + static_cast<std::uint32_t>(id)) *
                kAttentionDim,
            kAttentionDim));
    }
    for (std::uint32_t h = 0; h < kHeads; ++h) {
      std::vector<double> scores(rows.size());
      double maximum = sinks[h];
      const auto offset = ((static_cast<std::size_t>(t) * kHeads) + h) * kAttentionDim;
      for (std::size_t r = 0; r < rows.size(); ++r) {
        double dot = 0;
        for (std::uint32_t d = 0; d < kAttentionDim; ++d)
          dot += static_cast<double>(q[offset + d]) * rows[r][d];
        scores[r] = dot / std::sqrt(static_cast<double>(kAttentionDim));
        maximum = std::max(maximum, scores[r]);
      }
      double denominator = std::exp(static_cast<double>(sinks[h]) - maximum);
      for (auto& value : scores) {
        value = std::exp(value - maximum);
        denominator += value;
      }
      for (std::uint32_t d = 0; d < kAttentionDim; ++d) {
        double numerator = 0;
        for (std::size_t r = 0; r < rows.size(); ++r) numerator += scores[r] * rows[r][d];
        output[offset + d] = numerator / denominator;
      }
    }
  }
  return output;
}

TEST_F(Ds4AttentionTest, IndexedScalarAndHeadGroupsRespectBanksFutureRowsAndSink) {
  constexpr std::size_t kTokens = 3;
  constexpr std::size_t kRaw = 16;
  constexpr std::size_t kComp = 12;
  constexpr std::size_t kTop = 8;
  std::vector<float> query(kTokens * kHeads * kAttentionDim);
  std::vector<float> raw(2ULL * kRaw * kAttentionDim);
  std::vector<float> comp(2ULL * kComp * kAttentionDim);
  std::vector<float> sinks(kHeads);
  for (std::size_t i = 0; i < query.size(); ++i)
    query[i] = static_cast<float>(static_cast<int>(i % 7) - 3) / 32;
  for (std::size_t i = 0; i < raw.size(); ++i)
    raw[i] = static_cast<float>(static_cast<int>(i % 13) - 6) / 32;
  for (std::size_t i = 0; i < comp.size(); ++i)
    comp[i] = static_cast<float>(static_cast<int>(i % 19) - 9) / 32;
  for (std::size_t h = 0; h < sinks.size(); ++h) sinks[h] = static_cast<float>(h % 5) / 4;
  const std::array<std::int32_t, kTokens> positions{20, 8, 21};
  const std::array<std::int32_t, kTokens> banks{1, 0, 1};
  std::vector<std::int32_t> selected(kTokens * kTop);
  for (std::uint32_t t = 0; t < kTokens; ++t) {
    for (std::uint32_t slot = 0; slot < kTop; ++slot)
      selected[(t * kTop) + slot] = static_cast<std::int32_t>((slot + t) % kComp);
    selected[(t * kTop) + 6] = -1;
    selected[(t * kTop) + 7] = kComp + 7;
  }
  const auto compressed = Upload<float>(comp);
  const auto codes = Allocate(2ULL * kComp * 704ULL);
  const auto scales = Allocate(2ULL * kComp * 28ULL);
  ASSERT_TRUE(kg::RunDs4CacheQat(
      launch(), {.values = compressed, .codes = codes, .scales = scales, .rows = 2ULL * kComp}));
  const auto rounded = Download<float>(compressed);
  kg::Ds4Attention desc{.query = Upload<float>(query),
                        .output = Allocate(query.size() * sizeof(float)),
                        .sinks = Upload<float>(sinks),
                        .raw = Upload<float>(raw),
                        .compressed = compressed,
                        .compressed_codes = codes,
                        .compressed_scales = scales,
                        .decode_table = Upload<float>(kg::Ds4CacheDecodeTable()),
                        .selected = Upload<std::int32_t>(selected),
                        .positions = Upload<std::int32_t>(positions),
                        .bank_ids = Upload<std::int32_t>(banks),
                        .diagnostics = Zeroed<kg::Ds4AttentionDiagnostics>(),
                        .tokens = kTokens,
                        .first = 20,
                        .raw_cells = kRaw,
                        .raw_count = kRaw,
                        .compressed_cells = kComp,
                        .compressed_count = kComp,
                        .banks = 2,
                        .window = 8,
                        .top_k = kTop,
                        .domain = kg::Ds4AttentionDomain::kIndexedRing,
                        .kind = kg::Ds4AttentionKind::kScalar};
  // No paid predecode: original in-kernel packed reader is the fallback.
  ASSERT_TRUE(kg::RunDs4Attention(launch(), desc));
  const auto packed = Download<float>(desc.output);
  auto dense = desc;
  dense.compressed_codes = {};
  dense.compressed_scales = {};
  dense.decode_table = {};
  ASSERT_TRUE(kg::RunDs4Attention(launch(), dense));
  Exact(packed, Download<float>(desc.output));
  const auto reference = AttentionReference(query, raw, rounded, sinks, positions, banks, selected,
                                            kRaw, kComp, kComp, 8, kTop);
  ASSERT_EQ(reference.size(), packed.size());
  for (std::size_t i = 0; i < packed.size(); ++i) EXPECT_NEAR(packed[i], reference[i], 2e-6);
  const auto predecode = kg::PlanDs4AttentionScratch(desc, kg::Ds4AttentionKind::kScalar, 48);
  ASSERT_TRUE(predecode);
  ASSERT_TRUE(predecode->predecode);
  desc.scratch = Allocate(predecode->bytes);
  ASSERT_TRUE(kg::RunDs4Attention(launch(), desc));
  Exact(packed, Download<float>(desc.output));
  desc.kind = kg::Ds4AttentionKind::kHeadGroup;
  const auto hg = kg::PlanDs4AttentionScratch(desc, desc.kind, 48);
  ASSERT_TRUE(hg);
  desc.scratch = Allocate(hg->bytes);
  ASSERT_TRUE(kg::RunDs4Attention(launch(), desc));
  const auto grouped = Download<float>(desc.output);
  for (std::size_t i = 0; i < grouped.size(); ++i) EXPECT_NEAR(grouped[i], reference[i], 2e-6);
  EXPECT_GT(Download<kg::Ds4AttentionDiagnostics>(desc.diagnostics)[0].indexed_packed_reads, 0);
}

TEST_F(Ds4AttentionTest, TokenTileBitmapAndDeepSortChargeCompleteUnionMirrorChain) {
  constexpr std::size_t kTokens = 129;
  constexpr std::size_t kRawCells = 512;
  std::vector<float> query(kTokens * kHeads * kAttentionDim, 0.0f);
  std::vector<float> raw(kRawCells * kAttentionDim, 0.0f);
  std::vector<float> sinks(kHeads, 0.0f);
  for (const auto cells : {std::uint32_t{1024}, std::uint32_t{32769}}) {
    SCOPED_TRACE(cells);
    const auto first = cells == 1024 ? 128U : 131072U;
    std::vector<float> comp(static_cast<std::size_t>(cells) * kAttentionDim, 0.0f);
    for (std::uint32_t c = 0; c < cells; ++c)
      comp[c * kAttentionDim] = static_cast<float>((c % 7) + 1) / 16;
    std::vector<std::int32_t> selected(kTokens * 512ULL);
    for (std::uint32_t t = 0; t < kTokens; ++t) {
      std::vector<bool> used(cells, false);
      selected[t * 512ULL] = static_cast<std::int32_t>(cells - 1);
      used[cells - 1] = true;
      selected[(t * 512ULL) + 1] = -1;
      selected[(t * 512ULL) + 2] = static_cast<std::int32_t>(cells + 3);
      std::uint32_t walk = 0;
      for (std::uint32_t slot = 3; slot < 512; ++slot) {
        std::uint32_t id = 0;
        do {
          id = (cells - 600 + ((walk++) * 17ULL) + (t * 13ULL)) % cells;
        } while (used[id]);
        used[id] = true;
        selected[(t * 512ULL) + slot] = static_cast<std::int32_t>(id);
      }
    }
    const auto compressed = Upload<float>(comp);
    const auto codes = Allocate(static_cast<std::size_t>(cells) * 704ULL);
    const auto scales = Allocate(static_cast<std::size_t>(cells) * 28ULL);
    ASSERT_TRUE(kg::RunDs4CacheQat(
        launch(), {.values = compressed, .codes = codes, .scales = scales, .rows = cells}));
    const auto rounded = Download<float>(compressed);
    kg::Ds4Attention desc{
        .query = Upload<float>(query),
        .output = Allocate(query.size() * sizeof(float)),
        .sinks = Upload<float>(sinks),
        .raw = Upload<float>(raw),
        .compressed = compressed,
        .compressed_codes = codes,
        .compressed_scales = scales,
        .decode_table = Upload<float>(kg::Ds4CacheDecodeTable()),
        .selected = Upload<std::int32_t>(selected),
        .diagnostics = Zeroed<kg::Ds4AttentionDiagnostics>(),
        .tokens = kTokens,
        .first = first,
        .raw_cells = kRawCells,
        .raw_count = kTokens + 127,
        .raw_start = static_cast<std::uint32_t>((first + kTokens - (kTokens + 127)) % kRawCells),
        .compressed_cells = cells,
        .compressed_count = cells,
        .consecutive_first = first,
        .domain = kg::Ds4AttentionDomain::kIndexedRing,
        .kind = kg::Ds4AttentionKind::kTokenTile};
    const auto plan = kg::PlanDs4AttentionScratch(desc, desc.kind, 48);
    ASSERT_TRUE(plan);
    desc.scratch = Allocate(plan->bytes);
    const auto dispatch = kg::DescribeDs4AttentionDispatch(launch(), desc);
    ASSERT_TRUE(dispatch);
    EXPECT_EQ(dispatch->kind, kg::Ds4AttentionKind::kTokenTile);
    EXPECT_EQ(dispatch->scratch.bytes, plan->bytes);
    ASSERT_TRUE(kg::RunDs4Attention(launch(), desc));
    const auto actual = Download<float>(desc.output);
    ASSERT_TRUE(kg::RunDs4Attention(launch(), desc));
    Exact(actual, Download<float>(desc.output));
    double other_max = 0;
    for (std::uint32_t t = 0; t < kTokens; ++t) {
      const auto visible = std::min((first + t + 1) / 4, cells);
      double sum = 0;
      std::uint32_t count = 0;
      for (std::uint32_t slot = 0; slot < 512; ++slot) {
        const auto id = selected[(t * 512ULL) + slot];
        if (id >= 0 && std::cmp_less(id, visible)) {
          sum += rounded[static_cast<std::size_t>(id) * kAttentionDim];
          ++count;
        }
      }
      const auto expected = sum / (128 + count + 1.0);  // sink has a zero value.
      for (std::uint32_t h = 0; h < kHeads; ++h) {
        const auto offset = ((static_cast<std::size_t>(t) * kHeads) + h) * kAttentionDim;
        EXPECT_NEAR(actual[offset], expected, 2e-4);
        for (std::uint32_t d = 1; d < kAttentionDim; ++d)
          other_max = std::max(other_max, std::abs(static_cast<double>(actual[offset + d])));
      }
    }
    EXPECT_TRUE(std::ranges::all_of(actual, [](float value) { return std::isfinite(value); }));
    EXPECT_EQ(other_max, 0);
    // The eager-only chain must refuse capture before submitting its first stage.
    const auto rejected =
        launch().Capture([desc](auto& context) { return kg::RunDs4Attention(context, desc); });
    ASSERT_FALSE(rejected);
    EXPECT_EQ(rejected.error().error, kg::KernelError::kRejected);
    EXPECT_FALSE(launch().faulted());
  }
}

TEST_F(Ds4AttentionTest, IndexedCaptureReadsUpdatedPositionAndCompressedCount) {
  constexpr std::size_t kRaw = 16;
  constexpr std::size_t kComp = 12;
  std::vector<float> query(kHeads * kAttentionDim, 0.0f);
  std::vector<float> raw(kRaw * kAttentionDim, 0.0f);
  std::vector<float> comp(kComp * kAttentionDim, 0.0f);
  std::vector<float> sinks(kHeads, 0.0f);
  const std::array<std::int32_t, 8> selected{0, 1, 2, 3, 4, 5, 6, 7};
  for (std::uint32_t c = 0; c < kComp; ++c)
    comp[c * kAttentionDim] = (static_cast<float>(c) + 1.0f) / 16;
  const auto compressed = Upload<float>(comp);
  const auto codes = Allocate(kComp * 704ULL);
  const auto scales = Allocate(kComp * 28ULL);
  ASSERT_TRUE(kg::RunDs4CacheQat(
      launch(), {.values = compressed, .codes = codes, .scales = scales, .rows = kComp}));
  const std::array<std::int32_t, 1> initial_position{20};
  const std::array<std::int32_t, 1> bank{0};
  kg::Ds4Attention desc{.query = Upload<float>(query),
                        .output = Allocate(query.size() * sizeof(float)),
                        .sinks = Upload<float>(sinks),
                        .raw = Upload<float>(raw),
                        .compressed = compressed,
                        .compressed_codes = codes,
                        .compressed_scales = scales,
                        .decode_table = Upload<float>(kg::Ds4CacheDecodeTable()),
                        .selected = Upload<std::int32_t>(selected),
                        .positions = Upload<std::int32_t>(initial_position),
                        .bank_ids = Upload<std::int32_t>(bank),
                        .decode_scalars = Zeroed<kg::Ds4AttentionDecodeScalars>(),
                        .layer_scalars = Zeroed<kg::Ds4IndexerLayerScalars>(),
                        .diagnostics = Zeroed<kg::Ds4AttentionDiagnostics>(),
                        .tokens = 1,
                        .first = 20,
                        .raw_cells = kRaw,
                        .raw_count = kRaw,
                        .compressed_cells = kComp,
                        .compressed_count = 3,
                        .window = 8,
                        .top_k = 8,
                        .domain = kg::Ds4AttentionDomain::kIndexedRing,
                        .kind = kg::Ds4AttentionKind::kScalar};
  auto graph =
      launch().Capture([desc](auto& context) { return kg::RunDs4Attention(context, desc); });
  ASSERT_TRUE(graph);
  float previous = -1;
  for (const auto count : {std::uint32_t{3}, std::uint32_t{7}}) {
    const std::array<kg::Ds4IndexerLayerScalars, 1> layers{
        kg::Ds4IndexerLayerScalars{.n_comp = count}};
    const std::array<std::int32_t, 1> position{count == 3 ? 20 : 40};
    Write<kg::Ds4IndexerLayerScalars>(desc.layer_scalars, layers);
    Write<std::int32_t>(desc.positions, position);
    // Position-specific kernels deliberately supersede these decode fields.
    const std::array<kg::Ds4AttentionDecodeScalars, 1> scalars{
        kg::Ds4AttentionDecodeScalars{.raw_start = 1, .n_raw = 4}};
    Write<kg::Ds4AttentionDecodeScalars>(desc.decode_scalars, scalars);
    ASSERT_TRUE(launch().Launch(*graph));
    const auto captured = Download<float>(desc.output);
    ASSERT_TRUE(kg::RunDs4Attention(launch(), desc));
    Exact(captured, Download<float>(desc.output));
    EXPECT_GT(captured[0], previous);
    previous = captured[0];
    double sum = 0;
    for (std::uint32_t c = 0; c < count; ++c) sum += (c + 1.0) / 16;
    EXPECT_NEAR(captured[0], sum / (8 + count + 1.0), 2e-6);
  }
  Finish();
}
TEST_F(Ds4AttentionTest, OriginalSelectedMaskFeedsItsMaskedStaticFallback) {
  constexpr std::size_t kTokens = 4;
  constexpr std::size_t kCells = 7;
  const std::array<std::int32_t, 8> selected{3, -1, 1, 2, 2, 3, 0, 3};
  const auto ids = Upload<std::int32_t>(selected);
  const auto mask = Allocate(kTokens * kCells * 4ULL);
  ASSERT_TRUE(kg::RunDs4AttentionMask(
      launch(), {.selected = ids, .mask = mask, .tokens = kTokens, .cells = kCells, .top_k = 2}));
  const auto values = Download<float>(mask);
  for (std::uint32_t t = 0; t < kTokens; ++t) {
    for (std::uint32_t c = 0; c < kCells; ++c) {
      const bool included =
          std::cmp_equal(selected[t * 2ULL], c) || std::cmp_equal(selected[(t * 2ULL) + 1], c);
      EXPECT_EQ(values[(t * kCells) + c],
                included ? 0.0f : -std::numeric_limits<float>::infinity());
    }
  }
  std::vector<float> query(kTokens * kHeads * kAttentionDim, 0.0f);
  std::vector<float> raw(kTokens * kAttentionDim, 0.0f);
  std::vector<float> comp(kCells * kAttentionDim, 0.0f);
  std::vector<float> sinks(kHeads, 0.0f);
  comp[0] = 1;
  const kg::Ds4Attention desc{.query = Upload<float>(query),
                              .output = Allocate(query.size() * sizeof(float)),
                              .sinks = Upload<float>(sinks),
                              .raw = Upload<float>(raw),
                              .compressed = Upload<float>(comp),
                              .mask = mask,
                              .diagnostics = Zeroed<kg::Ds4AttentionDiagnostics>(),
                              .tokens = kTokens,
                              .raw_cells = kTokens,
                              .raw_count = kTokens,
                              .compressed_cells = kCells,
                              .compressed_count = kCells,
                              .domain = kg::Ds4AttentionDomain::kMaskedPrefill,
                              .kind = kg::Ds4AttentionKind::kScalar};
  ASSERT_TRUE(kg::RunDs4Attention(launch(), desc));
  const auto output = Download<float>(desc.output);
  for (std::uint32_t t = 0; t < kTokens; ++t) {
    for (std::uint32_t h = 0; h < kHeads; ++h) {
      const auto offset = ((static_cast<std::size_t>(t) * kHeads) + h) * kAttentionDim;
      EXPECT_NEAR(output[offset], t == 3 ? 1.0 / 6 : 0.0, 1e-7);
    }
  }
}

}  // namespace
