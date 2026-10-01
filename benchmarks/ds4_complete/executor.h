// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// Temporary matched ds4 reference, owned and scheduled by jitLLM. This is
// a concrete original-stage recipe, not a third-party runtime or callback
// dispatch table. Bounded scope: context8192/32768, T4096/all43/final-only head.
#ifndef JITLLM_BENCHMARKS_DS4_COMPLETE_EXECUTOR_H_
#define JITLLM_BENCHMARKS_DS4_COMPLETE_EXECUTOR_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "kernels/ggml/dsv4_ds4_attention.h"
#include "kernels/ggml/dsv4_ds4_comp.h"
#include "kernels/ggml/dsv4_ds4_hc.h"
#include "kernels/ggml/dsv4_ds4_moe.h"
#include "kernels/ggml/dsv4_ds4_product.h"
#include "model/dsv4.h"
#include "providers/device_runtime.h"

namespace jitllm::engine {
class PagedNode;
}
namespace jitllm::catalog {
struct Closure;
}

namespace jitllm::benchmarks::ds4_complete {
namespace kg = kernels::ggml;
using Result = std::expected<void, std::string>;
inline constexpr std::uint32_t kLayers = 43;
inline constexpr std::uint32_t kRows = 4096;
constexpr bool SupportedContext(std::uint32_t context) {
  return context == 8192 || context == 32768;
}
constexpr bool ChunkStart(std::uint32_t context, std::uint32_t first) {
  return SupportedContext(context) && first % kRows == 0 && first <= context - kRows;
}
// Exact pinned original default at the supported whole-chunk bands. Device
// eligibility is reported by Describe, not supplied by the benchmark caller.
constexpr bool OriginalSelectorChoice(std::uint32_t cells, kg::Ds4IndexerSelectKind kind,
                                      bool cub_available) {
  if (cells == 1024) return kind == kg::Ds4IndexerSelectKind::kBitonic1024;
  if (cells == 2048) return kind == kg::Ds4IndexerSelectKind::kBitonic2048;
  if (cells == 3072) return kind == kg::Ds4IndexerSelectKind::kBitonic4096;
  if (cells == 4096)
    return kind == (cub_available ? kg::Ds4IndexerSelectKind::kCub8192
                                  : kg::Ds4IndexerSelectKind::kBitonic4096);
  if (cells >= 5120 && cells <= 8192 && cells % 1024 == 0)
    return kind == (cub_available ? kg::Ds4IndexerSelectKind::kCub8192
                                  : kg::Ds4IndexerSelectKind::kBitonic8192);
  return false;
}
inline constexpr std::size_t kProfileChains = 8;
// Borrowed diagnostic handles. The runner owns them until native teardown
// has fenced every consumer, including a failed/unknown recording. No event
// is created, recorded, read or destroyed by an inactive profile.
struct ProfileMarks {
  std::array<providers::TimingMark, 2 * kProfileChains> marks{};
  std::array<float, kProfileChains> milliseconds{};
  std::array<bool, kProfileChains> active{};
};
inline constexpr std::array<std::string_view, kProfileChains> kProfileNames{
    "hc-attention-input", "q-kv-production", "compression-indexer", "attention",
    "attention-output",   "ffn-input-route", "routed-ffn",          "shared-ffn-expand"};
inline constexpr std::string_view kCommunityArtifact =
    "cd39d504dc2dbfe911a4a521fa8efc8053dc3e80e99738a9b25fa6b70c97a1ac";

// No allocation in Encode. Root binds these ordinary typed descriptors
// from checked model roles/native alternate weights and mapped scratch.
// Complete cross-stage wiring is validated before embedding can submit.
struct Compression {
  kg::Ds4F16Product kv_projection{}, score_projection{};
  kg::Ds4CompChunk chunk{};
  // Original ratio4 refresh uses F32 inputs and small split-K products.
  // Both are mandatory when chunk's plan marks refresh_required.
  std::optional<kg::Ds4F16Vector> refresh_kv, refresh_score;
  std::optional<kg::Ds4CompRefresh> refresh;
};
struct Indexer {
  Compression compression{};
  kg::Ds4F16Conversion query_conversion{};
  kg::Ds4F16Product query_projection{}, weight_projection{};
  kg::Ds4HeadRope query_rope{};
  kg::Ds4CacheQat query_qat{};
  kg::Ds4IndexerScores scores{};
  kg::Ds4IndexerSelect select{};
};
struct Layer {
  std::uint32_t index = 0, ratio = 0;
  // Original fresh-chunk invalidation means only layer0 does this RMS.
  kg::Ds4Rms hc_attention_rms{};
  kg::Ds4F16Product hc_attention_projection{};
  kg::Ds4HcPre hc_attention_pre{};
  kg::Ds4Rms attention_norm{};
  kg::Ds4Q8Product query_a{}, kv_projection{}, query_b{};
  kg::Ds4QkvNorm qkv_norm{};
  kg::Ds4HeadRope query_rope{}, kv_rope{};
  kg::Ds4CacheQat raw_qat{};
  kg::Ds4CacheRawStore raw_store{};
  std::optional<Compression> compression;
  std::optional<Indexer> indexer;
  kg::Ds4Attention attention{};
  kg::Ds4OutA output_a{};
  kg::Ds4Q8Product output_b{};
  kg::Ds4HcExpand attention_expand{};
  kg::Ds4F16Product hc_ffn_projection{};
  kg::Ds4HcPre hc_ffn_pre{};
  kg::Ds4Rms ffn_norm{};
  kg::Ds4F16Product router_projection{};
  kg::Ds4Router router{};
  kg::Ds4Moe routed{};
  // Optional fully paid Materialized probe on ORIGINAL norm/routes.
  // Every writable range is separate from the ordinary recipe.
  std::optional<kg::Ds4Moe> routed_control;
  kg::Ds4Q8Product shared_gate{}, shared_up{}, shared_down{};
  kg::Ds4SharedSwiglu shared_swiglu{};
  // Present on last layer only; preceding layers fuse this guarded sum.
  std::optional<kg::Ds4MoeSum> final_sum;
  kg::Ds4HcExpand ffn_expand{};
};
struct Frontier {
  kg::Ds4Rms hc_rms{};
  kg::Ds4F16Vector hc_projection{};
  kg::Ds4HcHeadWeights hc_weights{};
  kg::Ds4HcWeighted collapse{};
  kg::Ds4Rms norm{};
  kg::Ds4Q8Vector projection{};
};
struct ResolvedStage {
  std::uint32_t layer = 0;
  std::string_view stage, kind;
  std::uint64_t scratch_bytes = 0;
  std::uint32_t score_band = 0;
  std::uint64_t cub_temp_storage_bytes = 0, selector_dynamic_smem_bytes = 0, device_smem_bytes = 0;
  bool cub_available = false;
};
struct ResolvedDispatch {
  int device = -1;
  std::uint32_t architecture = 0, stream = 0, planned_sms = 0;
  std::uint64_t native_pool_bytes = 0, cublas_bytes = 0;
  std::vector<ResolvedStage> stages;
};
enum class OutputBConsumer : std::uint8_t { kOriginal, kNativeMmq };
enum class RoutedFfnTier : std::uint8_t { kDirect, kMaterialized };
constexpr bool CaptureRoutedFfnAt(std::uint32_t first, std::uint32_t layer) {
  return first == kRows && layer == 21;
}
// Bounded diagnostic calls on ORIGINAL upstream operands in the late chunk.
constexpr bool CaptureOutputBAt(std::uint32_t first, std::uint32_t layer) {
  return first == kRows && (layer == 0 || layer == 21 || layer == 42);
}
struct Chunk {
  std::uint32_t first = 0;
  std::uint32_t context = 0;
  std::uint32_t device_sms = 0;
  std::uint64_t storage_generation = 0, model_generation = 0;
  std::string artifact_id;
  OutputBConsumer output_b_consumer = OutputBConsumer::kOriginal;
  bool output_b_study = false;
  RoutedFfnTier routed_ffn_tier = RoutedFfnTier::kDirect;
  bool routed_ffn_study = false;
  // Symmetrically charged stored gate/up/activation in both study arms.
  std::array<kg::Ds4CacheBuffer, 3> routed_intermediates{};
  // Optional separate native result in an original-only diagnostic pass.
  // It is mapped/charged with the same lifetime as every recipe operand.
  kg::Ds4CacheBuffer output_b_control{};
  kg::Ds4Embedding embedding{};
  std::vector<Layer> layers;
  std::optional<Frontier> frontier;
  // Mapped, charged catalog ranges retained by the native Job closure.
  // Includes weights, state, all scratch and the logical packed-cache proxy.
  // This internal binder inventory is never accepted from an API client.
  std::vector<kg::Ds4CacheBuffer> paid_ranges;

 private:
  friend struct RuntimeAccess;
  // A resolve seal belongs to this exact context and operand binding.
  // Rebinding or reusing another context requires a new Resolve.
  const kg::LaunchContext* resolved_owner_ = nullptr;
  std::optional<model::Dsv4Profile> resolved_profile_;
  std::vector<std::uint64_t> resolved_identity_;
  std::optional<ResolvedDispatch> resolved_dispatch_;
};

// Caller-owned borrowed identity. Hash rows are HOST copies from the
// authenticated artifact, validated here in <=4096-row batches. No Boolean
// supplied by a client can authorize an unchecked token->expert table.
struct HashTable {
  std::uint32_t layer = 0;
  kg::Ds4CacheBuffer device{};
  std::span<const std::int32_t> entries;
};
Result CheckHashTables(const model::Dsv4Profile& profile, const Chunk& chunk,
                       std::span<const HashTable> tables);
// Exact original C YaRN parameter construction, independent of CUDA.
std::expected<kg::Ds4ProductRope, std::string> OriginalRope(const model::Dsv4Profile& profile,
                                                            std::uint32_t ratio,
                                                            std::uint32_t first,
                                                            bool inverse = false);
Result CheckChunk(const model::Dsv4Profile& profile, const Chunk& chunk);
// Host preflight of paid byte coverage and immutable/state alias contracts.
// This checks an inventory slice; only CheckChunk qualifies a full recipe.
Result CheckMappedOperands(const Chunk& chunk);
// Actual Describe/Plan results retained by Resolve, never requested enums
// alone. This host-only copy can be persisted outside the measured interval.
std::expected<ResolvedDispatch, std::string> ReadResolvedDispatch(const Chunk& chunk);

// Progress is a completion ledger, never a GPU rollback mechanism.
// Failure after possible submission poisons the request; counted state is
// unchanged but GPU owners must still be retained/fenced by the runner.
enum class Phase : std::uint8_t { kEmbedding, kLayers, kFrontier, kComplete, kPoisoned };
struct CompletionToken;
struct Progress {
  std::uint32_t first = 0, next_layer = 0, context = 0;
  std::uint64_t storage_generation = 0, model_generation = 0;
  Phase phase = Phase::kEmbedding;
  std::array<std::uint32_t, kLayers> compressed{}, indexed{};

 private:
  friend struct RuntimeAccess;
  friend void Poison(Progress& x);
  friend std::expected<Progress, std::string> Begin(const model::Dsv4Profile& p, const Chunk& c,
                                                    const Progress* previous);
  // CPU counting transitions cannot manufacture native completion. Run
  // keeps a private receipt of the ledger after each successful Job fence.
  const kg::LaunchContext* completion_owner_ = nullptr;
  std::vector<std::uint64_t> completion_identity_;
  // Copies share the latest completed native epoch. A stale saved ledger
  // cannot replay against newer GPU contents or revive a poisoned request.
  std::shared_ptr<CompletionToken> completion_token_;
  std::uint64_t completion_serial_ = 0;
};
// New generation means caller already completed original state setup/clear.
// No arbitrary seek into a partial/failed original state is accepted.
std::expected<Progress, std::string> Begin(const model::Dsv4Profile& p, const Chunk& c,
                                           const Progress* previous = nullptr);
Result CheckProgress(const Chunk& c, const Progress& x, Phase phase, std::uint32_t layer = 0);
// These CPU transitions are invoked by Run only after its native Job
// returns successful, proved completion. They never queue GPU work.
Result CompleteEmbedding(const Chunk& c, Progress& x);
Result CompleteLayer(const Chunk& c, std::uint32_t layer, Progress& x);
Result CompleteFrontier(const Chunk& c, Progress& x);
void Poison(Progress& x);

// Built only for the native benchmark/control with
// JITLLM_DS4_COMPLETE_CUDA. CPU build retains validation/ledger and returns
// an explicit refusal for runtime entry points. All methods use root's
// native node/closure/provider/stream, no private allocator or CUDA types.
// Resolve authenticates all actual original dispatch/workspace choices
// before the measured recipe, refusing any missing paid chain.
// Hash tables must be checked before Resolve/model math.
Result Resolve(engine::PagedNode& node, kg::LaunchContext& launch, const catalog::Closure& closure,
               std::uint32_t stream, const model::Dsv4Profile& profile, Chunk& chunk,
               std::span<const HashTable> hash_tables);
Result RunEmbedding(engine::PagedNode& node, kg::LaunchContext& launch,
                    const catalog::Closure& closure, std::uint32_t stream, const Chunk& c,
                    Progress& x, ProfileMarks* profile = nullptr);
Result RunLayer(engine::PagedNode& node, kg::LaunchContext& launch, const catalog::Closure& closure,
                std::uint32_t stream, const Chunk& c, std::uint32_t layer, Progress& x,
                ProfileMarks* profile = nullptr);
Result RunFrontier(engine::PagedNode& node, kg::LaunchContext& launch,
                   const catalog::Closure& closure, std::uint32_t stream, const Chunk& c,
                   Progress& x, ProfileMarks* profile = nullptr);

}  // namespace jitllm::benchmarks::ds4_complete
#endif  // JITLLM_BENCHMARKS_DS4_COMPLETE_EXECUTOR_H_
