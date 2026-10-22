// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// A model's settings in three layers (D-103; docs/runtime-serving.md#model-
// settings): every setting the runtime uses for a model resolves once, at
// registration and before the model is constructed, from
//
//   1. a default derived from the artifact itself (its architecture, its
//      kept metadata: the trained context, the checkpoint's sampling
//      defaults, the drafter's block, the vocabulary's reasoning markers),
//      never from a checkpoint's name or hash;
//   2. a calibration measured on this machine, for a measured speed trade
//      (runtime/calibration.h), when one is recorded for this artifact,
//      drafter, device, driver and build;
//   3. the owner's override in [models.<name>] (config/node_config.h
//      ModelKeys),
//
// later layers winning; a setting none of them gives takes its fallback,
// the constant measured on a GB10 that M3 shipped. Each value keeps its
// source and what it came from, logged at registration and listed by
// `llmp-runtime settings`. Vendor-free: the CPU tests drive it, and the
// settings command runs in any build.

#ifndef LLMP_RUNTIME_MODEL_SETTINGS_H_
#define LLMP_RUNTIME_MODEL_SETTINGS_H_

#include <sys/types.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <expected>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "artifact/artifact.h"
#include "artifact/gguf_metadata.h"
#include "base/sha256.h"
#include "config/node_config.h"
#include "runtime/component_admission.h"
#include "runtime/memory_guard.h"
#include "spec/mapping.h"
#include "spec/spec.h"
#include "tokenizer/tokenizer.h"

namespace llmp::spec {
struct ServingRecipe;
}
namespace llmp::runtime {

struct Approval;

enum class SettingSource : std::uint8_t { kDerived, kCalibrated, kOverride, kFallback };
std::string_view SourceName(SettingSource source);

template <typename T>
struct Setting {
  T value{};
  SettingSource source = SettingSource::kFallback;
  std::string basis;  // what it came from: a metadata key, a measurement, a constant
};

// ------------------------------------------------------------ fallbacks
// Each the constant M3 measured on one GB10 (spark-b), used until a
// calibration on the machine measures it or the owner overrides it.

// Prefill chunk rows (docs/runtime-serving.md#prefill-chunks-and-
// cancellation) are each architecture's serving recipe's
// (spec/mapping.h ServingRecipe): the fastest through the runtime at 8K and
// 32K within the memory bound (DeepSeek: 4,096 rows 12-15% faster than
// 2,048, 8,192 no faster; its attention mask is sized for the whole
// context, so above 262,144 tokens it takes 2,048 rows: at 1,048,576, 4,096
// would fix 2.14 GiB more and take that from the measured 1M
// conversation's state), as are a speculating model's most owners (DeepSeek
// with DSpark: eight, a verify of at least two of a wave's sixteen rows).
// Request slots: the knee where another slot stops raising the completed
// rate enough to pay for slowing every request (D-104; docs/experiments/
// request-slots).
inline constexpr std::uint32_t kDefaultSlots = 4;
// The reclaim order's cost of dropped conversation state, recomputed by
// prefill, a token of the conversation (Server::Reclaim): 1.4 ms measured
// for DeepSeek's prefill (16,384 tokens in 22.9 s on a GB10), the slower of
// the two models'. A token, not a byte: a conversation's state is mostly
// fixed rings and windows while short, so a cost a byte measured on short
// prefills would rank long conversations far too cheap to drop.
inline constexpr double kRecomputeMsPerToken = 1.4;
// The drafting and wave settings' measured defaults are each family's
// serving recipe's (spec::ServingRecipe's default_* fields).
// The image pipeline's generation (runtime/engine_image.h).
inline constexpr std::uint32_t kImageSize = 1024;
inline constexpr std::uint32_t kImageSteps = 40;
// The reasoning markers the runtime looks for in a vocabulary.
inline constexpr std::string_view kThinkStart = "<think>";
inline constexpr std::string_view kThinkEnd = "</think>";

// ------------------------------------------------------------ the artifact

// What a model's artifacts say, read before the model is constructed
// (ReadArtifactFacts), including the bound spec identity for shared serving.
struct ArtifactFacts {
  // Encoding used to compute target_spec; independent of speed calibration.
  std::string kv_cache_encoding = "f16";
  // Actual spec/artifact identities for current component serving. No parsed
  // model is retained. Legacy facts leave these absent.
  std::optional<base::Sha256Digest> target_spec{};
  std::optional<base::Sha256Digest> composed_spec{};
  std::optional<std::uint32_t> composed_draft_vocab{};
  std::string composed_spec_error;  // ignored when speculation is disabled
  // The target's approval (runtime/approvals.h), by its source identity:
  // its envelope's policy rules and label; null for an unapproved source.
  const Approval* approval = nullptr;
  // The drafter's approval, likewise (null: no drafter, or none approved).
  const Approval* drafter_approval = nullptr;
  // The pipeline its spec says (a stored spec's, else its mapping's).
  spec::PipelineKind pipeline = spec::PipelineKind::kLlm;
  // The artifact's architecture ("deepseek4", "qwen4exp"); a composition's
  // models are pipelines (composition is true, architecture empty).
  std::string architecture;
  std::string container;
  std::uint32_t vocabulary = 0;
  std::vector<spec::SourceValue> policy_gates;
  bool composition = false;
  // The checkpoint's trained context (GGUF <architecture>.context_length,
  // config.json max_position_embeddings, or a rope-scaled original), and
  // where it came from.
  std::optional<std::uint64_t> trained_context;
  std::string trained_context_from;
  // The checkpoint's sampling defaults (GGUF general.sampling.*,
  // generation_config.json), each where kept, and where they came from.
  std::optional<double> temperature;
  std::optional<double> top_p;
  std::optional<std::int64_t> top_k;
  std::optional<double> min_p;
  std::string sampling_from;
  // The drafter's architecture (empty: no drafter) and its block: the most
  // drafts it proposes a step (GGUF <architecture>.block_size), if kept.
  std::string drafter_architecture;
  std::optional<std::uint64_t> drafter_block;
  // Qwen MTP selected-head rows: the packed BF16 matrix and matching I32
  // token map limit every request, even a larger nominal draft_vocab.
  std::optional<std::uint32_t> drafter_selected_rows;
  // The runner's context ceiling derived from the spec's stage context /
  // position limits (0: no runner / unsupported).
  std::uint32_t context_ceiling = 0;
  // Whether the target stage contains any routed MoE layer (gates bounded
  // policy rules when unpinned).
  bool has_moe = false;
};

struct ModelSettings;

// Capability predicates over a model's facts or settings.
bool SupportsSpeculativeDrafting(const ArtifactFacts& facts);
bool SupportsWaveForm(const ArtifactFacts& facts);
bool SupportsWaveCosts(const ArtifactFacts& facts);
bool SupportsMtpDrafting(const ArtifactFacts& facts);

bool SupportsSpeculativeDrafting(const ModelSettings& settings);
bool SupportsWaveForm(const ModelSettings& settings);
bool SupportsWaveCosts(const ModelSettings& settings);
bool SupportsMtpDrafting(const ModelSettings& settings);

// Whether a setting key applies to a model with the given facts or settings.
bool KeyAppliesTo(std::string_view key, const ArtifactFacts& facts);
bool KeyAppliesTo(std::string_view key, const ModelSettings& settings);
bool KeyAppliesTo(const config::ModelKeySpec& spec, const ArtifactFacts& facts);
bool KeyAppliesTo(const config::ModelKeySpec& spec, const ModelSettings& settings);

// An installed artifact, opened and its ID checked (who may change the
// store is the filesystem's business, D-073 as amended on 2026-10-09).
std::expected<artifact::Artifact, std::string> OpenInstalled(const std::filesystem::path& store,
                                                             const std::string& id);

// A file the configuration names, or a record the runtime wrote: opened
// as any file is (links followed), a regular file of at most `limit`
// bytes.
std::expected<std::string, std::string> ReadRegularFile(const std::filesystem::path& path,
                                                        std::size_t limit);

// The opened artifacts and their verified admissions from ReadArtifactFacts /
// ArtifactFactsOf, retained by callers (Server::Start -> ComponentLlmBuild,
// PrintSettings) that continue with the same artifacts rather than opening or
// admitting them a second time.
struct OpenedModelArtifacts {
  OpenedModelArtifacts() = default;
  OpenedModelArtifacts(OpenedModelArtifacts&&) noexcept = default;
  OpenedModelArtifacts& operator=(OpenedModelArtifacts&&) = delete;
  // First: release only after every retained artifact/admission field dies.
  // Published as shared const storage; factories and artifact views retain it.
  std::unique_ptr<StartupArtifactPayment> payment;
  std::optional<artifact::Artifact> target;
  std::optional<ComponentAdmission> target_admission;
  std::optional<artifact::Artifact> drafter;
  std::optional<ComponentDrafterAdmission> drafter_admission;
  std::shared_ptr<const PreparedDrafterComposition> composition;
};

using StartupArtifactFunder =
    std::function<std::expected<std::unique_ptr<StartupArtifactPayment>, std::string>(
        const artifact::Artifact&, const artifact::Artifact*)>;

// The facts of a configured model's artifact and drafter: both opened as
// OpenInstalled does, their stored specs and admissions read once.
std::expected<ArtifactFacts, std::string> ReadArtifactFacts(
    const config::ModelEntry& entry, const std::filesystem::path& store,
    OpenedModelArtifacts* opened_out = nullptr, const StartupArtifactFunder& funder = {});
// The same from an opened artifact (and its drafter's, if any).
std::expected<ArtifactFacts, std::string> ArtifactFactsOf(
    const artifact::Artifact& target, const artifact::Artifact* drafter,
    std::string_view kv_cache_encoding = "f16",
    std::optional<std::uint32_t> draft_vocab = std::nullopt,
    OpenedModelArtifacts* opened_out = nullptr);

// A model's tokenizer and chat template: kept in its artifact's metadata
// (a GGUF's, the first shard's; or tokenizer.json and chat_template.jinja),
// or the checkpoint's files the configuration names (M3's Qwen3.8 import
// kept config.json only).
inline constexpr std::size_t kMaxTokenizerBytes = std::size_t{64} << 20U;
struct ChatAssets {
  tokenizer::TokenizerSpec spec;
  std::optional<std::string> chat_template;  // none: the model has no chat turns
  std::string from;                          // where they came from, for messages
};
std::expected<ChatAssets, std::string> ReadChatAssets(const artifact::Artifact& target,
                                                      const config::ModelEntry& entry);

// ------------------------------------------------------------ calibration

// Values measured on this machine for one artifact, device, driver and
// build (runtime/calibration.h reads and writes them); each absent until
// measured.
struct Calibration {
  std::optional<std::uint32_t> prefill_chunk;
  std::optional<std::uint32_t> max_slots;
  std::optional<std::uint32_t> prefill_floor_tok_s;
  std::optional<std::uint32_t> decode_floor_tok_s;
  std::optional<double> recompute_ms_per_token;
  // By width from 2, those measured; the rest keep their fallback.
  std::array<std::optional<double>, config::kMaxWaveCostWidths> wave_costs{};
  std::optional<double> depth_cost_ratio;
  std::string basis;  // where it was measured, for the log

  bool empty() const {
    return !prefill_chunk && !max_slots && !prefill_floor_tok_s && !decode_floor_tok_s &&
           !recompute_ms_per_token && !depth_cost_ratio &&
           std::ranges::none_of(wave_costs, [](const auto& c) { return c.has_value(); });
  }
};

// ------------------------------------------------------------ settings

struct ModelSettings {
  std::string name;
  std::string architecture;  // empty for a composition
  std::string container;
  bool composition = false;
  // The serving adapter's kind (a composition's: the image pipeline).
  spec::PipelineKind pipeline = spec::PipelineKind::kLlm;
  bool drafter = false;
  // Chosen before calibration from the unchanged speculation setting.
  std::optional<base::Sha256Digest> bound_spec{};

  // Every LLM.
  Setting<std::string> kv_cache_encoding{"f16", SettingSource::kDerived,
                                         "dense-attention state default"};
  Setting<std::uint32_t> context;
  Setting<bool> speculation;
  Setting<std::uint32_t> prefill_chunk;  // before the model caps it at its context
  Setting<std::uint32_t> max_slots;
  Setting<std::uint32_t> prefill_floor_tok_s;
  Setting<std::uint32_t> decode_floor_tok_s;
  Setting<double> recompute_ms_per_token;
  Setting<double> temperature;
  Setting<double> top_p;
  Setting<std::uint32_t> top_k;
  Setting<double> min_p;
  // The reasoning markers' texts, each a token of the vocabulary; empty:
  // none. Resolved when the tokenizer is read (ResolveReasoning).
  Setting<std::string> reasoning_start;
  Setting<std::string> reasoning_end;
  Setting<std::uint32_t> draft_rows;
  // Adaptive waves (ServingRecipe::adaptive_waves).
  Setting<config::WaveForm> wave_form;
  Setting<std::vector<double>> wave_costs;  // widths 2 up
  // How many leading widths the owner overrides. The resolved vector alone
  // cannot distinguish equal fallback-valued prefixes that hide different
  // calibrated widths. Internal dependency metadata, not another setting.
  std::size_t wave_costs_override_count = 0;
  // MTP waves (ServingRecipe::mtp_waves).
  Setting<std::uint32_t> draft_vocab;
  Setting<double> depth_cost_ratio;
  Setting<std::uint32_t> shared_wave_depth;
  Setting<std::uint32_t> draft_wave_max;
  Setting<bool> wave_lanes;
  Setting<std::uint32_t> wave_read_align;
  // A composition (the image pipeline).
  Setting<std::uint32_t> image_size;
  Setting<std::uint32_t> image_steps;

  // Overrides the model's architecture does not use: ignored, each with
  // why, for the log.
  std::vector<std::string> ignored;
  // Overrides past an approved checkpoint's qualified envelope: served,
  // each with why, for the log (D-119).
  std::vector<std::string> warnings;

  // One setting as listed: its key (ModelKeys'), value, source and basis.
  struct Line {
    std::string_view key;
    std::string value;
    SettingSource source = SettingSource::kFallback;
    std::string basis;
  };
  // Every setting the model uses, in ModelKeys' order.
  std::vector<Line> Lines() const;
  // The settings on one line, "key=value (source)" each, for the log (the
  // ignored overrides are logged on their own).
  std::string Summary() const;
};

// The settings of `entry`, whose artifacts say `facts`, with `calibration`
// (null: none recorded); `plain` (the commands' --plain) turns speculation
// off. Refused, saying why, when an override is outside what the artifact
// allows: a context beyond the checkpoint's ceiling, more drafts than the
// drafter proposes. The reasoning markers stay unresolved (their fallback,
// none) until ResolveReasoning.
// trained_max is an internal qualification hook: the runner's ceiling is the
// checkpoint's trained maximum, which its metadata must state (its recipe's
// owner bounds still apply).
std::expected<ModelSettings, std::string> ResolveSettings(const config::ModelEntry& entry,
                                                          const ArtifactFacts& facts,
                                                          const Calibration* calibration,
                                                          bool plain, bool trained_max = false);

// An architecture's serving recipe (its mapping's; spec/mapping.h); null:
// none is served.
const spec::ServingRecipe* RecipeOf(std::string_view architecture, std::string_view container = {});

// The reasoning markers once the vocabulary is known (`find`: a token's ID
// by its exact text): each override, which must be a token of it (empty:
// none); else derived, "<think>" and "</think>" where the vocabulary has
// both; else none.
using FindToken = std::function<std::optional<std::int32_t>(std::string_view)>;
std::expected<void, std::string> ResolveReasoning(ModelSettings& settings,
                                                  const config::ModelEntry& entry,
                                                  const FindToken& find);

// `llmp-runtime settings`: every configured model's settings resolved
// (its artifacts, tokenizer and calibration record read), printed to
// `out` as a table, or with `json` one JSON object; problems to `log`.
// It reads and never writes, takes no process lock (a
// running service's settings can be listed beside it) and creates no
// device context. Returns the process's exit status: 1 if any model's
// could not be.
int PrintSettings(const config::NodeConfig& config, bool json, std::FILE* out, std::FILE* log);

}  // namespace llmp::runtime

#endif  // LLMP_RUNTIME_MODEL_SETTINGS_H_
