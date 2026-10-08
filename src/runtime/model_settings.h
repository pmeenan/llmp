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
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "artifact/artifact.h"
#include "artifact/gguf_metadata.h"
#include "config/node_config.h"
#include "tokenizer/tokenizer.h"

namespace llmp::model {
struct Gemma4Profile;
}
namespace llmp::runtime {

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
// cancellation): the fastest through the runtime at 8K and 32K within the
// memory bound (DeepSeek: 4,096 rows 12-15% faster than 2,048, 8,192 no
// faster). DeepSeek's attention mask is sized for the whole context, so
// above 262,144 tokens it takes 2,048 rows: at 1,048,576, 4,096 would fix
// 2.14 GiB more and take that from the measured 1M conversation's state.
inline constexpr std::uint32_t kDsv4PrefillRows = 4096;
inline constexpr std::uint32_t kDsv4DeepPrefillRows = 2048;
inline constexpr std::uint32_t kDsv4WidePrefillContext = 262144;  // the widest at 4,096
inline constexpr std::uint32_t kQwen38PrefillRows = 4096;
// Request slots: the knee where another slot stops raising the completed
// rate enough to pay for slowing every request (D-104; docs/experiments/
// request-slots). DeepSeek with DSpark takes at most eight: a verify of at
// least two of a wave's sixteen rows (engine/dsv4_runner.h kWaveRows).
inline constexpr std::uint32_t kDefaultSlots = 4;
inline constexpr std::uint32_t kDsv4SpeculativeMostSlots = 8;
// The reclaim order's cost of dropped conversation state, recomputed by
// prefill, a token of the conversation (Server::Reclaim): 1.4 ms measured
// for DeepSeek's prefill (16,384 tokens in 22.9 s on a GB10), the slower of
// the two models'. A token, not a byte: a conversation's state is mostly
// fixed rings and windows while short, so a cost a byte measured on short
// prefills would rank long conversations far too cheap to drop.
inline constexpr double kRecomputeMsPerToken = 1.4;
// DeepSeek with DSpark: a draft-verify wave's time over a plain decode
// wave's, widths 2 to 8 (execution/adaptive_wave_mode.h): the median joined
// wave of `llmp_spec_runner --check wave --slots N --wave-mode
// verify|decode` (GB10, community GGUF, wave lanes on, 2026-10-03;
// docs/experiments/deepseek-batching, "Wave lanes" and "Joined draft
// blocks"): with joined drafts, 127.5 / 61.3, 180.0 / 71.5, 231.2 / 82.3
// and 229.0 / 95.6 ms at widths 2 to 5; with lanes, 224.7 / 106.1, 258.0 /
// 116.6 and 287.3 / 129.1 ms at widths 6 to 8 (D-104; past five a joined
// draft's rows exceed a wave's, so each block runs alone). A DSpark verify
// then takes three rows a request at width 5 and two past it, so past five
// a draft-verify wave (at most two tokens a request) never pays
// (docs/experiments/request-slots). To be measured again when a wave's
// cost changes.
inline constexpr std::array<double, config::kMaxWaveCostWidths> kDsv4WaveCosts = {
    2.08, 2.52, 2.81, 2.40, 2.12, 2.21, 2.23};
// DSpark's drafts a step (its verify takes one row more).
inline constexpr std::uint32_t kDsv4DraftRows = 3;
// Qwen3.8's MTP: drafts a step at most (depths 2-3 chosen by acceptance,
// AdaptiveDepth), the draft head's rows (the fastest of 32,768 rows to the
// whole vocabulary, docs/experiments/qwen38-mtp), depth 3's cost over
// depth 2's (execution/adaptive_depth.h), a wave's drafts a request (C2/C4
// HTTP screens: +8%/+15% over adaptive depth), how many requests draft in
// one joined wave (past two each drafts alone: 3.6% faster at C4), and the
// state a wave reads, rounded up (engine/qwen38_runner.cc).
inline constexpr std::uint32_t kQwen38DraftRows = 3;
inline constexpr std::uint32_t kQwen38DraftVocab = 65536;
inline constexpr double kQwen38DepthCostRatio = 1.16;
inline constexpr std::uint32_t kQwen38SharedWaveDepth = 2;
inline constexpr std::uint32_t kQwen38DraftWaveMax = 2;
inline constexpr std::uint32_t kQwen38WaveReadAlign = 2048;
// The image pipeline's generation (engine/qwen_image_runner.h).
inline constexpr std::uint32_t kImageSize = 1024;
inline constexpr std::uint32_t kImageSteps = 40;
// The reasoning markers the runtime looks for in a vocabulary.
inline constexpr std::string_view kThinkStart = "<think>";
inline constexpr std::string_view kThinkEnd = "</think>";

// ------------------------------------------------------------ the artifact

// What a model's artifacts say, read before the model is constructed
// (ReadArtifactFacts). Nothing here is a checkpoint's name or hash.
struct ArtifactFacts {
  // Non-null only after the complete approved Gemma binding was checked.
  const model::Gemma4Profile* gemma_profile = nullptr;
  // The artifact's architecture ("deepseek4", "qwen4exp"); a composition's
  // models are pipelines (composition is true, architecture empty).
  std::string architecture;
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
};

// The runner's own ceiling of an architecture's context (the position
// tables it supports), or 0 for an architecture without a runner.
std::uint32_t RunnerContextCeiling(std::string_view architecture);

// An installed artifact, opened under the store's trust rules (only root
// and the runtime's user may change it; D-056's integrity rests on them)
// and its ID checked.
std::expected<artifact::Artifact, std::string> OpenInstalled(const std::filesystem::path& store,
                                                             const std::string& id);

// A file the configuration names, under its trust rules (D-073): every
// directory on the way and the file itself root's or `trusted`'s and
// writable by nobody else, opened without following links, at most `limit`
// bytes.
std::expected<std::string, std::string> ReadTrustedFile(const std::filesystem::path& path,
                                                        uid_t trusted, std::size_t limit);

// The facts of a configured model's artifact and drafter: both opened as
// OpenInstalled does, their kept metadata read as untrusted input.
std::expected<ArtifactFacts, std::string> ReadArtifactFacts(const config::ModelEntry& entry,
                                                            const std::filesystem::path& store);
// The same from what an opened artifact keeps (and its drafter's, if any).
std::expected<ArtifactFacts, std::string> ArtifactFactsOf(const artifact::Artifact& target,
                                                          const artifact::Artifact* drafter);
// Its parts, each adding to `facts` (`facts.architecture` set): the kept
// GGUF metadata's keys (GgufFactKeys: the context, its rope scaling, the
// sampling defaults), config.json's (max_position_embeddings and
// rope_scaling, at the top or in text_config) and generation_config.json's
// sampling defaults. A sampling default outside a request's range is not
// one (ignored); the first source of each wins.
std::vector<std::string> GgufFactKeys(std::string_view architecture);
void FactsFromGguf(const artifact::GgufMetadata& metadata, ArtifactFacts& facts);
std::expected<void, std::string> FactsFromConfigJson(std::string_view text, ArtifactFacts& facts);
std::expected<void, std::string> FactsFromGenerationConfig(std::string_view text,
                                                           ArtifactFacts& facts);

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
                                                      const config::ModelEntry& entry,
                                                      uid_t trusted);

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
  bool composition = false;
  bool drafter = false;
  // Internal candidate recipe, keyed separately from ordinary calibration.
  bool gemma31_production = false;
  // The bounded Gemma26 production recipe (1024-row prefill, norm chains,
  // MoE route/reduce, joined owner attention), keyed separately too.
  bool gemma26_production = false;

  // Every LLM.
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
  // DeepSeek V4.
  Setting<config::WaveForm> wave_form;
  Setting<std::vector<double>> wave_costs;  // widths 2 up
  // How many leading widths the owner overrides. The resolved vector alone
  // cannot distinguish equal fallback-valued prefixes that hide different
  // calibrated widths. Internal dependency metadata, not another setting.
  std::size_t wave_costs_override_count = 0;
  Setting<bool> prefill_outa_hca;
  Setting<bool> prefill_outa_hca_partial;
  // Qwen3.8.
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
// The default enables the qualified ordinary Gemma31 recipe only within its
// checked context/owner bounds; false preserves internal diagnostic controls.
// gemma3_trained_max is an internal qualification hook: only the approved
// Gemma3 profile may resolve its trained maximum, still with one owner.
std::expected<ModelSettings, std::string> ResolveSettings(
    const config::ModelEntry& entry, const ArtifactFacts& facts, const Calibration* calibration,
    bool plain, bool gemma31_production = true, bool gemma3_trained_max = false);

// The reasoning markers once the vocabulary is known (`find`: a token's ID
// by its exact text): each override, which must be a token of it (empty:
// none); else derived, "<think>" and "</think>" where the vocabulary has
// both; else none.
using FindToken = std::function<std::optional<std::int32_t>(std::string_view)>;
std::expected<void, std::string> ResolveReasoning(ModelSettings& settings,
                                                  const config::ModelEntry& entry,
                                                  const FindToken& find);

// `llmp-runtime settings`: every configured model's settings resolved
// (its artifacts, tokenizer and calibration record read under their trust
// rules), printed to `out` as a table, or with `json` one JSON object;
// problems to `log`. It reads and never writes, takes no process lock (a
// running service's settings can be listed beside it) and creates no
// device context. Returns the process's exit status: 1 if any model's
// could not be.
int PrintSettings(const config::NodeConfig& config, bool json, std::FILE* out, std::FILE* log);

}  // namespace llmp::runtime

#endif  // LLMP_RUNTIME_MODEL_SETTINGS_H_
