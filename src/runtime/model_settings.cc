// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "runtime/model_settings.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <format>
#include <limits>
#include <span>
#include <utility>

#include "artifact/gguf_metadata.h"
#include "base/json.h"
#include "base/report.h"
#include "components/state_encoding.h"
#include "engine/request_cohort.h"
#include "runtime/approvals.h"
#include "runtime/calibration.h"
#include "runtime/component_admission.h"
#include "spec/mapping.h"
#include "spec/stored.h"
#include "tokenizer/gguf.h"
#include "tokenizer/hf.h"
#include "tokenizer/sentencepiece.h"

namespace llmp::runtime {
namespace {

namespace fs = std::filesystem;
namespace ja = llmp::artifact;
namespace json = llmp::base::json;

std::unexpected<std::string> Error(std::string what) { return std::unexpected(std::move(what)); }

std::string Errno(int error) { return std::strerror(error); }  // NOLINT(concurrency-mt-unsafe)

// A kept metadata file's name (`meta/<name>`), if the artifact lists it.
bool Keeps(const ja::Artifact& a, std::string_view name) {
  return std::ranges::any_of(a.files(), [&](const ja::ListedFile& f) {
    return f.role == ja::FileRole::kSourceMetadata && f.path.starts_with("meta/") &&
           std::string_view(f.path).substr(5) == name;
  });
}

std::expected<ja::GgufMetadata, std::string> ReadKeptGguf(const ja::Artifact& a,
                                                          const std::string& kv,
                                                          std::span<const std::string_view> keys) {
  auto bytes = a.ReadMetadata(kv);
  if (!bytes) {
    return Error(std::format("{}: {}", kv, bytes.error().ToString()));
  }
  auto read = ja::ReadGgufMetadata(std::as_bytes(std::span(*bytes)), keys);
  if (!read) {
    return Error(std::format("{}: {}", kv, read.error().ToString()));
  }
  return std::move(*read);
}

std::optional<std::int64_t> Integer(const ja::GgufMetadata& m, std::string_view key) {
  const auto it = m.find(key);
  if (it == m.end() || it->second.kind != ja::GgufValue::Kind::kInteger) {
    return std::nullopt;
  }
  return it->second.integer;
}

// The sampling defaults a checkpoint keeps, each within the range a request
// may ask for (api.h); one outside it is not a default the route can apply.
void Sampling(ArtifactFacts& facts, std::optional<double> temperature, std::optional<double> top_p,
              std::optional<std::int64_t> top_k, std::optional<double> min_p,
              std::string_view from) {
  bool any = false;
  if (!facts.temperature && temperature && std::isfinite(*temperature) && *temperature >= 0 &&
      *temperature <= 2) {
    facts.temperature = temperature;
    any = true;
  }
  if (!facts.top_p && top_p && std::isfinite(*top_p) && static_cast<float>(*top_p) > 0.0F &&
      *top_p <= 1) {
    facts.top_p = top_p;
    any = true;
  }
  if (!facts.top_k && top_k && *top_k >= 0 && *top_k < (std::int64_t{1} << 31)) {
    facts.top_k = top_k;
    any = true;
  }
  if (!facts.min_p && min_p && std::isfinite(*min_p) && *min_p >= 0 && *min_p <= 1) {
    facts.min_p = min_p;
    any = true;
  }
  if (any) {
    facts.sampling_from += (facts.sampling_from.empty() ? "" : ", ") + std::string(from);
  }
}

// A value's text as the settings command and the log show it: JSON.
std::string Text(bool v) { return v ? "true" : "false"; }
std::string Text(std::uint32_t v) { return std::format("{}", v); }
std::string Text(double v) { return std::format("{}", v); }
std::string Text(const std::string& v) {
  std::string out;
  json::AppendQuoted(v, out);
  return out;
}
std::string Text(config::WaveForm v) {
  switch (v) {
    case config::WaveForm::kSpeculative:
      return "\"speculative\"";
    case config::WaveForm::kPlain:
      return "\"plain\"";
    case config::WaveForm::kAuto:
      break;
  }
  return "\"auto\"";
}
std::string Text(const std::vector<double>& v) {
  std::string out = "[";
  for (std::size_t i = 0; i < v.size(); ++i) {
    out += std::format("{}{}", i == 0 ? "" : ", ", v[i]);
  }
  return out + "]";
}

std::string OverrideBasis(const config::ModelEntry& entry, std::string_view key) {
  return std::format("models.{}.{}", entry.name, key);
}

// A setting from its layers: the override, else the calibration, else the
// derived default, else the fallback.
template <typename T>
Setting<T> Pick(std::optional<T> override_value, std::string override_basis,
                std::optional<T> calibrated, const std::string& calibration_basis,
                std::optional<T> derived, std::string derived_basis, T fallback,
                std::string fallback_basis) {
  if (override_value) {
    return {std::move(*override_value), SettingSource::kOverride, std::move(override_basis)};
  }
  if (calibrated) {
    return {std::move(*calibrated), SettingSource::kCalibrated, calibration_basis};
  }
  if (derived) {
    return {std::move(*derived), SettingSource::kDerived, std::move(derived_basis)};
  }
  return {std::move(fallback), SettingSource::kFallback, std::move(fallback_basis)};
}

std::optional<std::uint32_t> U32(std::optional<std::int64_t> v) {
  return v ? std::optional(static_cast<std::uint32_t>(*v)) : std::nullopt;
}

}  // namespace

std::string_view SourceName(SettingSource source) {
  switch (source) {
    case SettingSource::kDerived:
      return "derived";
    case SettingSource::kCalibrated:
      return "calibrated";
    case SettingSource::kOverride:
      return "override";
    case SettingSource::kFallback:
      break;
  }
  return "fallback";
}

namespace {

// The runner's ceiling: its spec's stage context (0: no runner).
std::uint32_t ContextCeilingOf(const ArtifactFacts& facts) { return facts.context_ceiling; }

// The recipe's, or none's (every capability off).
const spec::ServingRecipe& Recipe(std::string_view architecture, std::string_view container = {}) {
  static constexpr spec::ServingRecipe kNone{};
  const spec::ServingRecipe* recipe = RecipeOf(architecture, container);
  return recipe != nullptr ? *recipe : kNone;
}

// A count as written in the bases ("262,144").
std::string Grouped(std::uint64_t n) {
  std::string digits = std::to_string(n);
  for (auto at = digits.size(); at > 3; at -= 3) digits.insert(at - 3, ",");
  return digits;
}

}  // namespace

const spec::ServingRecipe* RecipeOf(std::string_view architecture, std::string_view container) {
  if (!container.empty()) {
    const spec::Mapping* mapping = spec::FindMapping(architecture, container);
    return mapping ? spec::ServingRecipeOf(*mapping) : nullptr;
  }
  // Recipe discovery for architecture-only UI/synthetic facts does not select
  // a mapping variant. All variants must share the same serving recipe.
  const spec::ServingRecipe* recipe = nullptr;
  for (const auto* mapping : spec::Mappings()) {
    const spec::ServingRecipe* m_recipe = spec::ServingRecipeOf(*mapping);
    if (mapping->architecture != architecture || !m_recipe) continue;
    if (recipe && recipe != m_recipe) return nullptr;
    recipe = m_recipe;
  }
  return recipe;
}

bool SupportsSpeculativeDrafting(const ArtifactFacts& facts) {
  return !facts.drafter_architecture.empty() ||
         Recipe(facts.architecture, facts.container).draft_rows;
}

bool SupportsWaveForm(const ArtifactFacts& facts) { return SupportsSpeculativeDrafting(facts); }

bool SupportsWaveCosts(const ArtifactFacts& facts) { return SupportsSpeculativeDrafting(facts); }

bool SupportsMtpDrafting(const ArtifactFacts& facts) {
  return Recipe(facts.architecture, facts.container).mtp_waves;
}

bool SupportsSpeculativeDrafting(const ModelSettings& settings) {
  return settings.drafter || Recipe(settings.architecture, settings.container).draft_rows;
}

bool SupportsWaveForm(const ModelSettings& settings) {
  return SupportsSpeculativeDrafting(settings);
}

bool SupportsWaveCosts(const ModelSettings& settings) {
  return SupportsSpeculativeDrafting(settings);
}

bool SupportsMtpDrafting(const ModelSettings& settings) {
  return Recipe(settings.architecture, settings.container).mtp_waves;
}

bool KeyAppliesTo(std::string_view key, const ArtifactFacts& facts) {
  if (facts.composition) {
    return key == "image_size" || key == "image_steps";
  }
  if (key == "image_size" || key == "image_steps") {
    return false;
  }
  if (key == "draft_rows") {
    return SupportsSpeculativeDrafting(facts);
  }
  if (key == "wave_form") {
    return SupportsWaveForm(facts);
  }
  if (key == "wave_costs") {
    return SupportsWaveCosts(facts);
  }
  if (key == "draft_vocab" || key == "depth_cost_ratio" || key == "shared_wave_depth" ||
      key == "draft_wave_max") {
    return SupportsMtpDrafting(facts);
  }
  return true;
}

bool KeyAppliesTo(std::string_view key, const ModelSettings& settings) {
  if (settings.composition) {
    return key == "image_size" || key == "image_steps";
  }
  if (key == "image_size" || key == "image_steps") {
    return false;
  }
  if (key == "draft_rows") {
    return SupportsSpeculativeDrafting(settings);
  }
  if (key == "wave_form") {
    return SupportsWaveForm(settings);
  }
  if (key == "wave_costs") {
    return SupportsWaveCosts(settings);
  }
  if (key == "draft_vocab" || key == "depth_cost_ratio" || key == "shared_wave_depth" ||
      key == "draft_wave_max") {
    return SupportsMtpDrafting(settings);
  }
  return true;
}

bool KeyAppliesTo(const config::ModelKeySpec& spec, const ArtifactFacts& facts) {
  return KeyAppliesTo(spec.key, facts);
}

bool KeyAppliesTo(const config::ModelKeySpec& spec, const ModelSettings& settings) {
  return KeyAppliesTo(spec.key, settings);
}

std::expected<ja::Artifact, std::string> OpenInstalled(const fs::path& store,
                                                       const std::string& id) {
  ja::OpenOptions options;
  options.expected_id = id;
  auto opened = ja::Artifact::Open(store / id, options);
  if (!opened) {
    return Error(std::format("artifact {}: {}", id, opened.error().ToString()));
  }
  return std::move(*opened);
}

std::expected<std::string, std::string> ReadRegularFile(const fs::path& path, std::size_t limit) {
  const int fd = ::open(path.c_str(), O_RDONLY | O_NONBLOCK | O_CLOEXEC | O_NOCTTY);
  if (fd < 0) {
    return Error(std::format("{}: {}", path.string(), Errno(errno)));
  }
  struct stat status{};
  std::string bytes;
  std::string problem;
  if (::fstat(fd, &status) != 0) {
    problem = Errno(errno);
  } else if (!S_ISREG(status.st_mode)) {
    problem = "not a regular file";
  } else if (std::cmp_greater(status.st_size, limit)) {
    problem = std::format("larger than {} bytes", limit);
  } else {
    bytes.resize(static_cast<std::size_t>(status.st_size));
    std::size_t done = 0;
    while (done < bytes.size()) {
      const ssize_t n =
          ::pread(fd, bytes.data() + done, bytes.size() - done, static_cast<off_t>(done));
      if (n < 0 && errno == EINTR) {
        continue;
      }
      if (n <= 0) {
        problem = n < 0 ? Errno(errno) : "shorter than it was";
        break;
      }
      done += static_cast<std::size_t>(n);
    }
  }
  (void)::close(fd);
  if (!problem.empty()) {
    return Error(std::format("{}: {}", path.string(), problem));
  }
  return bytes;
}

std::expected<ArtifactFacts, std::string> ArtifactFactsOf(const ja::Artifact& target,
                                                          const ja::Artifact* drafter,
                                                          std::string_view kv_cache_encoding,
                                                          std::optional<std::uint32_t> draft_vocab,
                                                          OpenedModelArtifacts* opened_out) {
  if (opened_out != nullptr && (opened_out->target || opened_out->drafter ||
                                opened_out->target_admission || opened_out->drafter_admission))
    return Error("artifact facts require an empty retained owner");
  ArtifactFacts facts;
  if (kv_cache_encoding != "f16" && kv_cache_encoding != "q8_0")
    return Error("unknown KV cache encoding");
  facts.kv_cache_encoding = kv_cache_encoding;
  const bool quantized_kv = kv_cache_encoding == "q8_0";
  const auto shared_model = [](const ja::Artifact& artifact) {
    const auto* mapping =
        spec::FindMapping(artifact.model().architecture, artifact.model().container);
    return mapping && (spec::ServingRecipeOf(*mapping) != nullptr || mapping->draft);
  };
  if (quantized_kv && (!shared_model(target) || (drafter != nullptr && !shared_model(*drafter))))
    return Error("Q8_0 KV requires a supported shared model");
  const auto configure = [&](spec::ModelSpec& model) -> std::expected<void, std::string> {
    if (!quantized_kv) return components::ApplyKvCacheEncoding(model, format::Encoding::kF16, 0);
    auto bytes = components::KvCacheEncodingBytes(model, format::Encoding::kQ8_0);
    if (!bytes) return Error(bytes.error());
    // Bounded metadata-only inspection; the actual factory independently pays
    // and retains the same transform before constructing an executable source.
    return components::ApplyKvCacheEncoding(model, format::Encoding::kQ8_0, *bytes);
  };
  facts.architecture = target.model().architecture;
  facts.container = target.model().container;
  facts.approval = FindApproval(facts.architecture, target.sources());
  if (const spec::Mapping* mapping = spec::FindMapping(facts.architecture, facts.container);
      mapping != nullptr)
    facts.pipeline = mapping->pipeline;
  // The target's stored spec (a v1 artifact's alone: D-118) and, with a
  // drafter, its composition: their bound identities.
  if (auto served = RequireServedVersion(target); !served) return Error(served.error());
  if (drafter != nullptr)
    if (auto served = RequireServedVersion(*drafter); !served) return Error(served.error());
  auto admitted = ReadComponentAdmission(target);
  if (!admitted) return Error(admitted.error());
  const spec::Stage& target_stage = admitted->model.stages.front();
  facts.trained_context = target_stage.context;
  facts.trained_context_from = "the admitted stored spec";
  facts.pipeline = admitted->model.pipeline;
  const spec::SamplingDefaults& smp = target_stage.sampling;
  Sampling(facts, smp.temperature, smp.top_p, smp.top_k, smp.min_p, "the admitted stored spec");
  auto vocabulary = spec::Vocabulary(admitted->model, admitted->source);
  if (!vocabulary) return Error(vocabulary.error());
  facts.vocabulary = *vocabulary;
  if (const auto* mapping = spec::FindMapping(facts.architecture, facts.container))
    facts.policy_gates = spec::PolicyGates(*mapping, admitted->model, facts.container);
  // The drafter's admission, once: its source and optional stored spec as
  // checked, for its composition below.
  std::optional<ComponentDrafterAdmission> drafter_admitted;
  if (drafter != nullptr) {
    auto d_admitted = ReadComponentDrafterAdmission(*drafter);
    if (!d_admitted) return Error(d_admitted.error());
    drafter_admitted.emplace(std::move(*d_admitted));
  }
  {
    spec::ModelSpec model = admitted->model;
    facts.context_ceiling = model.stages.front().context;
    facts.has_moe = std::ranges::any_of(model.stages.front().layers, [](const spec::Layer& layer) {
      return std::ranges::any_of(layer.components,
                                 [](const spec::ComponentRef& c) { return c.kind == "ffn.moe"; });
    });
    if (drafter != nullptr) {
      const auto compose = [&]() -> std::expected<base::Sha256Digest, std::string> {
        const auto* target_mapping = spec::FindMapping(facts.architecture, facts.container);
        if (target_mapping == nullptr) return Error("component calibration has no target mapping");
        const auto* draft_mapping = ComposedDrafterMapping(*drafter);
        if (draft_mapping == nullptr) return Error("component calibration has no drafter mapping");
        std::expected<spec::ModelSpec, std::string> composed = std::unexpected(std::string());
        if (drafter_admitted->model) {
          if (draft_mapping->draft != nullptr &&
              draft_mapping->draft->target_architecture != target_mapping->architecture)
            return Error(std::format("drafter {} does not target architecture {}",
                                     drafter->model().architecture, facts.architecture));
          composed = spec::ComposeStoredDrafter(model, *drafter_admitted->model);
          if (!composed) return Error(composed.error());
          if (auto approved =
                  CheckComposedAdmission(*composed, admitted->source, drafter_admitted->source);
              !approved)
            return Error(approved.error());
          if (draft_mapping->draft != nullptr && !draft_mapping->draft->vocab_key.empty()) {
            const auto* serving_ptr = spec::ServingRecipeOf(*target_mapping);
            if (serving_ptr == nullptr)
              return Error("component calibration has no target serving recipe");
            const auto& serving = *serving_ptr;
            auto map = DraftHeadRows(*drafter, draft_mapping->draft->head_ids, facts.vocabulary);
            if (!map) return Error(map.error());
            const auto selected =
                *map ? std::optional(static_cast<std::uint32_t>((*map)->rows)) : std::nullopt;
            const std::uint32_t available = selected.value_or(facts.vocabulary);
            const std::uint32_t requested = draft_vocab.value_or(serving.default_draft_vocab);
            const std::uint32_t width = requested == 0 ? available : std::min(requested, available);
            if (width == 0 || available > facts.vocabulary)
              return Error("a draft head needs a positive row count within the target vocabulary");
            facts.composed_draft_vocab = width;
            composed = spec::ComposeStoredDrafter(model, *drafter_admitted->model, width);
            if (!composed) return Error(composed.error());
          }
        } else {
          auto draft_values = ComponentDrafterValues(target, *drafter);
          if (!draft_values) return Error(draft_values.error());
          const spec::Source* draft_source = &drafter_admitted->source;
          composed = spec::ComposeDrafter(model, *target_mapping, *draft_mapping, *draft_source,
                                          *draft_values);
          if (!composed) return Error(composed.error());
          if (auto approved =
                  CheckComposedAdmission(*composed, admitted->source, drafter_admitted->source);
              !approved)
            return Error(approved.error());
          if (draft_mapping->draft != nullptr && !draft_mapping->draft->vocab_key.empty()) {
            const auto* serving_ptr = spec::ServingRecipeOf(*target_mapping);
            if (serving_ptr == nullptr)
              return Error("component calibration has no target serving recipe");
            const auto& serving = *serving_ptr;
            auto map = DraftHeadRows(*drafter, draft_mapping->draft->head_ids, facts.vocabulary);
            if (!map) return Error(map.error());
            const auto selected =
                *map ? std::optional(static_cast<std::uint32_t>((*map)->rows)) : std::nullopt;
            auto width = ConfigureDraftVocabulary(*draft_values, *draft_mapping->draft,
                                                  draft_vocab.value_or(serving.default_draft_vocab),
                                                  facts.vocabulary, selected);
            if (!width) return Error(width.error());
            facts.composed_draft_vocab = *width;
            composed = spec::ComposeDrafter(model, *target_mapping, *draft_mapping, *draft_source,
                                            *draft_values);
            if (!composed) return Error(composed.error());
          }
        }
        if (auto configured = configure(*composed); !configured) return Error(configured.error());
        const std::array<std::string_view, 2> both{target.id(), drafter->id()};
        return spec::BoundDigest(*composed, both);
      };
      auto composed = compose();
      if (composed)
        facts.composed_spec = *composed;
      else
        facts.composed_spec_error = composed.error();
    }
    const std::array<std::string_view, 1> ids{target.id()};
    if (auto configured = configure(model); !configured) return Error(configured.error());
    auto digest = spec::BoundDigest(model, ids);
    if (!digest) return Error(digest.error());
    facts.target_spec = *digest;
  }
  if (drafter != nullptr) {
    facts.drafter_architecture = drafter->model().architecture;
    facts.drafter_approval = drafter_admitted->approval;
    const spec::Mapping* draft_mapping = ComposedDrafterMapping(*drafter);
    const spec::DraftRecipe* draft = draft_mapping != nullptr ? draft_mapping->draft : nullptr;
    if (draft != nullptr && draft->shape == spec::DraftRecipe::Shape::kMtp &&
        !draft->head_ids.empty()) {
      // A selected head: its token map's rows (its approval pins the whole
      // inventory).
      auto map =
          DraftHeadRows(*drafter, draft->head_ids, std::numeric_limits<std::uint32_t>::max());
      if (!map) return Error(map.error());
      if (*map) facts.drafter_selected_rows = static_cast<std::uint32_t>((*map)->rows);
    }
    if (const std::string kv = KeptGguf(*drafter); !kv.empty()) {
      const std::string block_key = std::format("{}.block_size", facts.drafter_architecture);
      const std::array<std::string_view, 1> keys = {block_key};
      auto m = ReadKeptGguf(*drafter, kv, keys);
      if (!m) {
        return Error(std::format("the drafter's {}", m.error()));
      }
      if (const auto block = Integer(*m, block_key); block && *block > 0) {
        facts.drafter_block = static_cast<std::uint64_t>(*block);
      }
    }
  }
  if (opened_out != nullptr) {
    opened_out->target_admission = std::move(*admitted);
    opened_out->drafter_admission = std::move(drafter_admitted);
  }
  return facts;
}

std::expected<ArtifactFacts, std::string> ReadArtifactFacts(const config::ModelEntry& entry,
                                                            const fs::path& store,
                                                            OpenedModelArtifacts* opened_out,
                                                            const StartupArtifactFunder& funder) {
  if (opened_out != nullptr && (opened_out->payment || opened_out->target || opened_out->drafter ||
                                opened_out->target_admission || opened_out->drafter_admission))
    return Error("artifact facts require an empty retained owner");
  if (entry.composition) {
    ArtifactFacts pipeline;
    pipeline.composition = true;
    return pipeline;
  }
  auto target = OpenInstalled(store, entry.artifact.value_or(""));
  if (!target) {
    return std::unexpected(target.error());
  }
  std::optional<ja::Artifact> drafter;
  if (entry.drafter) {
    auto opened = OpenInstalled(store, *entry.drafter);
    if (!opened) {
      return std::unexpected(opened.error());
    }
    drafter.emplace(std::move(*opened));
  }
  // Open's current pair is covered by the reusable parser loan. Before
  // admitting/copying its spec and source, give this retained alias a separate
  // payment, so the next alias can reuse that entire parser workspace.
  if (funder) {
    if (opened_out == nullptr) return Error("startup artifact funding needs a retained owner");
    auto paid = funder(*target, drafter ? &*drafter : nullptr);
    if (!paid) return Error(paid.error());
    opened_out->payment = std::move(*paid);
  }
  auto facts = ArtifactFactsOf(*target, drafter ? &*drafter : nullptr,
                               entry.Text("kv_cache_encoding").value_or("f16"),
                               U32(entry.Integer("draft_vocab")), opened_out);
  if (facts && opened_out != nullptr) {
    opened_out->target = std::move(*target);
    opened_out->drafter = std::move(drafter);
  }
  return facts;
}

std::expected<ChatAssets, std::string> ReadChatAssets(const ja::Artifact& target,
                                                      const config::ModelEntry& entry) {
  ChatAssets assets;
  // A kept file, or the one the configuration names, or none.
  const auto source = [&](std::string_view kept, const std::optional<fs::path>& named)
      -> std::expected<std::optional<std::string>, std::string> {
    if (Keeps(target, kept)) {
      auto meta = target.ReadMetadata(kept);
      if (!meta) {
        return Error(std::format("{}: {}", kept, meta.error().ToString()));
      }
      return std::optional(std::move(*meta));
    }
    if (named) {
      auto read = ReadRegularFile(*named, kMaxTokenizerBytes);
      if (!read) {
        return std::unexpected(read.error());
      }
      return std::optional(std::move(*read));
    }
    return std::optional<std::string>();
  };
  if (const std::string kv = KeptGguf(target); !kv.empty()) {
    auto bytes = target.ReadMetadata(kv);
    if (!bytes) {
      return Error(std::format("{}: {}", kv, bytes.error().ToString()));
    }
    auto read = tokenizer::ReadGgufTokenizer(std::as_bytes(std::span(*bytes)));
    if (!read) {
      return Error(std::format("{}: {}", kv, read.error().ToString()));
    }
    assets.spec = std::move(read->spec);
    assets.from = kv;
    if (read->has_chat_template) {
      assets.chat_template = std::move(read->chat_template);
      return assets;
    }
  } else {
    const bool model = !Keeps(target, "tokenizer.json") &&
                       (Keeps(target, "tokenizer.model") ||
                        (entry.tokenizer && entry.tokenizer->extension() == ".model"));
    const std::string_view name = model ? "tokenizer.model" : "tokenizer.json";
    auto data = source(name, entry.tokenizer);
    if (!data) return std::unexpected(data.error());
    if (!*data)
      return Error(
          std::format("the artifact keeps no GGUF metadata, tokenizer.json or "
                      "tokenizer.model and models.{}.tokenizer is not set",
                      entry.name));
    auto spec =
        model ? tokenizer::ReadSentencePieceTokenizer(**data) : tokenizer::ReadHfTokenizer(**data);
    if (!spec) return Error(std::format("{}: {}", name, spec.error().ToString()));
    if (auto config = source("tokenizer_config.json", std::nullopt); !config) {
      return std::unexpected(config.error());
    } else if (*config) {
      auto applied = tokenizer::ApplyHfTokenizerConfig(**config, *spec);
      if (!applied) return Error("tokenizer_config.json: " + applied.error().ToString());
    }
    assets.spec = std::move(*spec);
    assets.from =
        Keeps(target, name) ? std::string(name) : entry.tokenizer.value_or(fs::path()).string();
  }
  auto text = source("chat_template.jinja", entry.chat_template);
  if (!text) {
    return std::unexpected(text.error());
  }
  assets.chat_template = std::move(*text);
  if (!assets.chat_template) {
    for (const auto name : {"chat_template.json", "tokenizer_config.json"}) {
      auto kept = source(name, std::nullopt);
      if (!kept) return std::unexpected(kept.error());
      if (!*kept) continue;
      auto parsed = json::Parse(**kept);
      if (!parsed) return Error(std::format("{}: invalid JSON", name));
      const auto root = parsed->root();
      const auto value = root.is_object() ? root.find("chat_template") : std::optional(root);
      if (!value) continue;
      if (value->is_string())
        assets.chat_template = std::string(value->string());
      else if (value->is_array()) {
        // Transformers can store several named templates; select its default.
        for (std::size_t i = 0; i < value->size(); ++i) {
          const auto item = value->at(i);
          const auto key = item.is_object() ? item.find("name") : std::nullopt;
          const auto body = item.is_object() ? item.find("template") : std::nullopt;
          if (key && key->is_string() && key->string() == "default" && body && body->is_string())
            assets.chat_template = std::string(body->string());
        }
      }
      if (assets.chat_template) break;
    }
  }
  return assets;
}

std::expected<ModelSettings, std::string> ResolveSettings(const config::ModelEntry& entry,
                                                          const ArtifactFacts& facts,
                                                          const Calibration* calibration,
                                                          bool plain, bool trained_max) {
  ModelSettings s;
  s.name = entry.name;
  s.architecture = facts.architecture;
  s.container = facts.container;
  s.composition = facts.composition;
  s.pipeline = facts.composition ? spec::PipelineKind::kImage : facts.pipeline;
  s.drafter = entry.drafter.has_value();
  if (facts.composition && entry.Text("kv_cache_encoding"))
    return Error("KV cache encoding is not a composition setting");
  const auto kv = entry.Text("kv_cache_encoding").value_or("f16");
  if ((kv != "f16" && kv != "q8_0") || kv != facts.kv_cache_encoding)
    return Error("KV cache setting differs from the authenticated source facts");
  if (kv == "q8_0" &&
      (facts.composition || (entry.drafter ? !facts.composed_spec : !facts.target_spec)))
    return Error("Q8_0 KV requires an authenticated model");
  s.kv_cache_encoding = {
      kv, entry.Text("kv_cache_encoding") ? SettingSource::kOverride : SettingSource::kDerived,
      kv == "f16" ? "exact dense-attention contract encoding"
                  : "explicit lossy KV state override; never calibrated"};
  const Calibration none;
  const Calibration& cal = calibration != nullptr ? *calibration : none;
  const auto basis = [&](std::string_view key) { return OverrideBasis(entry, key); };

  if (s.composition) {
    s.image_size = Pick<std::uint32_t>(U32(entry.Integer("image_size")), basis("image_size"), {},
                                       {}, {}, {}, kImageSize, "the image runner's default");
    s.image_steps = Pick<std::uint32_t>(U32(entry.Integer("image_steps")), basis("image_steps"), {},
                                        {}, {}, {}, kImageSteps, "the image runner's default");
    return s;
  }
  const std::string& arch = facts.architecture;
  // The architecture's serving recipe (its mapping's data).
  const spec::ServingRecipe& recipe = Recipe(arch, facts.container);
  if (trained_max && !facts.trained_context)
    return Error(std::format("{} maximum-context qualification needs trained-context metadata",
                             recipe.name.empty() ? std::string_view(arch) : recipe.name));
  // The hook's runner is the checkpoint's trained maximum itself.
  const std::uint32_t runner =
      trained_max ? static_cast<std::uint32_t>(std::min<std::uint64_t>(
                        *facts.trained_context, std::numeric_limits<std::uint32_t>::max()))
                  : ContextCeilingOf(facts);
  if (runner == 0) {
    return Error(std::format("no runner for architecture {}", arch));
  }
  for (const auto& [key, value] : entry.overrides) {
    const config::ModelKeySpec* spec = config::FindModelKey(key);
    if (spec != nullptr && !KeyAppliesTo(key, facts)) {
      s.ignored.push_back(
          std::format("{} (a setting unsupported by this model's capabilities)", key));
    }
  }

  // The context: within the checkpoint's trained context and the runner's
  // position tables.
  std::uint32_t ceiling = runner;
  std::string ceiling_from = std::format("the {} runner's {}", arch, runner);
  if (facts.trained_context && *facts.trained_context >= config::kMinContext &&
      *facts.trained_context < runner) {
    ceiling = static_cast<std::uint32_t>(*facts.trained_context);
    ceiling_from = std::format("{} {}", facts.trained_context_from, ceiling);
  } else if (facts.trained_context) {
    ceiling_from = std::format("{} {}, the {} runner's {}", facts.trained_context_from,
                               *facts.trained_context, arch, runner);
  }
  if (const auto context = entry.Integer("context")) {
    if (std::cmp_greater(*context, ceiling)) {
      return Error(std::format("context {} is outside {}'s supported range {} to {} ({})", *context,
                               arch, config::kMinContext, ceiling, ceiling_from));
    }
    s.context = {static_cast<std::uint32_t>(*context), SettingSource::kOverride, basis("context")};
  } else if (recipe.default_context != 0 && ceiling >= recipe.default_context) {
    // The wider scalar envelope is explicit opt-in; retain the checked default.
    s.context = {recipe.default_context, SettingSource::kDerived,
                 std::format("bounded {} default; wider context opt-in", recipe.name)};
  } else if (ceiling < config::kDefaultContext) {
    s.context = {ceiling, SettingSource::kDerived, ceiling_from};
  } else {
    s.context = {config::kDefaultContext, SettingSource::kFallback,
                 std::format("the default, within {}", ceiling_from)};
  }
  // A default within the approved envelope (the target's approval's
  // default context, its joined rule domains', and its configured
  // drafter's): a default the approval refuses would never start, and a
  // wider one than its joined recipes is an explicit opt-in.
  if (s.context.source != SettingSource::kOverride) {
    const auto within = [&](const Approval* approval) {
      if (approval == nullptr || approval->max_context == 0) return;
      const std::uint32_t bound = ApprovedDefaultContext(*approval);
      if (s.context.value <= bound) return;
      s.context = {bound, SettingSource::kDerived,
                   std::format("the approved {} envelope's {}", approval->label, bound)};
    };
    within(facts.approval);
    if (entry.drafter) within(facts.drafter_approval);
    if (facts.approval == nullptr && facts.target_spec.has_value()) {
      const spec::Mapping* mapping = spec::FindMapping(arch, facts.container);
      if (mapping != nullptr) {
        const std::uint32_t bound = MappingDefaultContext(*mapping, facts.policy_gates);
        if (bound != 0 && s.context.value > bound) {
          s.context = {
              bound, SettingSource::kDerived,
              std::format("the {} mapping's joined envelope {}",
                          recipe.name.empty() ? std::string_view(arch) : recipe.name, bound)};
        }
      }
    }
  }
  const std::uint32_t context = s.context.value;

  // Speculation: with a drafter, unless turned off.
  if (plain) {
    s.speculation = {false, SettingSource::kOverride, "--plain"};
  } else if (!entry.drafter) {
    s.speculation = {false, SettingSource::kDerived, "no drafter"};
    if (entry.Bool("speculation").value_or(false)) {
      s.ignored.emplace_back("speculation (no drafter)");
    }
  } else if (const auto on = entry.Bool("speculation")) {
    s.speculation = {*on, SettingSource::kOverride, basis("speculation")};
  } else {
    s.speculation = {true, SettingSource::kDerived,
                     std::format("the drafter ({})", facts.drafter_architecture)};
  }
  // A recipe without drafters (Gemma 2's and 3's) has no speculative path.
  if (RecipeOf(arch, facts.container) != nullptr && !recipe.drafters &&
      (entry.drafter || entry.Bool("speculation").value_or(false)))
    return Error(
        std::format("{} serving has no qualified assistant or speculative path", recipe.family));
  const bool speculative = s.speculation.value;

  // Speed trades: calibrated on the machine, else M3's measured constants.
  const bool bounded = recipe.bounded;
  const bool deep = recipe.deep_context != 0 && context > recipe.deep_context;
  const std::uint32_t rows = deep ? recipe.deep_prefill_rows : recipe.prefill_rows;
  s.prefill_chunk = Pick<std::uint32_t>(
      U32(entry.Integer("prefill_chunk")), basis("prefill_chunk"), cal.prefill_chunk, cal.basis, {},
      {}, rows,
      bounded ? std::format("bounded {} runner envelope; uncalibrated", recipe.family)
      : deep  ? std::format("measured on a GB10 (above {} tokens: the mask's memory)",
                            Grouped(recipe.deep_context))
              : std::string("measured on a GB10"));
  s.max_slots = Pick<std::uint32_t>(
      U32(entry.Integer("max_slots")), basis("max_slots"), cal.max_slots, cal.basis, {}, {},
      kDefaultSlots,
      bounded ? "cooperative cohort default; uncalibrated" : "the knee measured on a GB10 (D-104)");
  // The approved envelopes (the target's approval's and its configured
  // drafter's; D-119's qualified envelope defaults): within them by
  // default, as the context is above, the basis saying so; an override past
  // one is served unqualified, with a warning (D-102: no refusal for want
  // of a qualification; real bounds, the stored context among them, hold).
  const auto envelope = [&](const Approval* approval) {
    if (approval == nullptr) return;
    const auto past = [&](std::string_view what) {
      s.warnings.push_back(std::format(
          "{} is past the approved {} envelope ({} tokens, {} owners{}): "
          "served, unqualified",
          what, approval->label, approval->max_context, approval->max_slots,
          approval->wide_context != 0 ? std::format(", one above {} tokens", approval->wide_context)
                                      : std::string()));
    };
    if (approval->max_context != 0 && context > approval->max_context &&
        s.context.source == SettingSource::kOverride)
      past(std::format("context {}", context));
    std::uint32_t owners = approval->max_slots;
    if (approval->wide_context != 0 && context > approval->wide_context) owners = 1;
    if (owners == 0 || s.max_slots.value <= owners) return;
    if (s.max_slots.source == SettingSource::kOverride) {
      past(std::format("max_slots {}", s.max_slots.value));
    } else {
      s.max_slots = {owners, SettingSource::kDerived,
                     owners == approval->max_slots
                         ? std::format("the approved {} envelope's {}", approval->label, owners)
                         : std::format("the approved {} envelope's one owner above {} tokens",
                                       approval->label, approval->wide_context)};
    }
  };
  envelope(facts.approval);
  if (entry.drafter) envelope(facts.drafter_approval);
  if (facts.approval == nullptr && recipe.wide_context != 0 && context > recipe.wide_context &&
      s.max_slots.value > 1) {
    if (s.max_slots.source != SettingSource::kOverride) {
      s.max_slots = {1, SettingSource::kDerived,
                     std::format("the {} recipe's one owner above {} tokens", recipe.name,
                                 recipe.wide_context)};
    }
  }
  if (bounded) {
    // Gemma 4's bounded recipes (a context of at most 8,192 and four
    // owners) were qualified at a prompt chunk, which its mapping's policy
    // rule for the envelope names (spec/mapping.h PolicyRule::prefill_chunk);
    // larger contexts and cohorts take the default.
    std::uint32_t recipe_chunk = 0;
    const std::string_view label = facts.approval != nullptr ? facts.approval->label : recipe.name;
    if (facts.approval != nullptr) {
      const auto rule =
          ApprovedRule(*facts.approval, {.context = context, .slots = s.max_slots.value});
      if (!rule) return Error(rule.error());
      if (*rule != nullptr) recipe_chunk = (*rule)->prefill_chunk;
    } else if (facts.target_spec.has_value()) {
      if (const spec::Mapping* mapping = spec::FindMapping(arch, facts.container);
          mapping != nullptr) {
        const auto rule = spec::DefaultRule(*mapping, facts.policy_gates,
                                            {.context = context, .slots = s.max_slots.value});
        if (!rule) return Error(rule.error());
        if (*rule != nullptr) recipe_chunk = (*rule)->prefill_chunk;
      }
    }
    if (recipe_chunk != 0 && s.prefill_chunk.source == SettingSource::kFallback) {
      s.prefill_chunk.value = recipe_chunk;
      s.prefill_chunk.basis = std::format("qualified bounded {} serving envelope", label);
    }
  }
  // Settings never assert component/backend capacity. The common serving
  // policy intersects those facts after source admission; this bound is the
  // actual request-slot representation, independent of model family.
  if (s.max_slots.value == 0 || s.max_slots.value > engine::kMaxRequestSlots)
    return Error("max_slots exceeds the request-slot representation");
  s.prefill_floor_tok_s = Pick<std::uint32_t>(
      U32(entry.Integer("prefill_floor_tok_s")), basis("prefill_floor_tok_s"),
      cal.prefill_floor_tok_s, cal.basis, {}, {}, config::kDefaultPrefillFloor, "the default");
  s.decode_floor_tok_s = Pick<std::uint32_t>(
      U32(entry.Integer("decode_floor_tok_s")), basis("decode_floor_tok_s"), cal.decode_floor_tok_s,
      cal.basis, {}, {}, config::kDefaultDecodeFloor, "the default");
  s.recompute_ms_per_token = Pick<double>(
      entry.Real("recompute_ms_per_token"), basis("recompute_ms_per_token"),
      cal.recompute_ms_per_token, cal.basis, {}, {}, kRecomputeMsPerToken,
      bounded ? "provisional recompute cost; uncalibrated"
              : "the slowest prefill measured on a GB10 (M3's; uncalibrated for this model)");

  // Sampling defaults: the checkpoint's own, else OpenAI's.
  const std::string& sampled = facts.sampling_from;
  s.temperature = Pick<double>(entry.Real("temperature"), basis("temperature"), {}, {},
                               facts.temperature, sampled, 1.0, "OpenAI's default");
  s.top_p = Pick<double>(entry.Real("top_p"), basis("top_p"), {}, {}, facts.top_p, sampled, 1.0,
                         "OpenAI's default");
  // The sampler takes a float, as a request's top_p (api.cc).
  if (static_cast<float>(s.top_p.value) <= 0.0F) {
    return Error(std::format("top_p {} is 0 as the sampler's float", s.top_p.value));
  }
  s.top_k = Pick<std::uint32_t>(U32(entry.Integer("top_k")), basis("top_k"), {}, {},
                                U32(facts.top_k), sampled, 0, "off");
  s.min_p =
      Pick<double>(entry.Real("min_p"), basis("min_p"), {}, {}, facts.min_p, sampled, 0.0, "off");
  s.reasoning_start = {std::string(), SettingSource::kFallback,
                       "none until the vocabulary is read"};
  s.reasoning_end = s.reasoning_start;

  // Drafting.
  const std::optional<std::int64_t> draft_rows = entry.Integer("draft_rows");
  if (recipe.block_drafts) {
    // The drafter's block (0: not kept) bounds its drafts.
    const std::uint64_t block = facts.drafter_block.value_or(0);
    const std::int64_t asked = draft_rows.value_or(0);
    if (block != 0 && std::cmp_greater(asked, block)) {
      return Error(
          std::format("draft_rows {} is more than the drafter's block of {} drafts", asked, block));
    }
    std::optional<std::uint32_t> derived;
    std::string derived_basis;
    if (block != 0 && block < recipe.default_draft_rows) {
      derived = static_cast<std::uint32_t>(block);
      derived_basis = std::format("{}.block_size {}", facts.drafter_architecture, block);
    }
    s.draft_rows =
        Pick<std::uint32_t>(U32(draft_rows), basis("draft_rows"), {}, {}, derived, derived_basis,
                            recipe.default_draft_rows, "measured on a GB10");
  } else {
    const std::int64_t asked = draft_rows.value_or(recipe.default_draft_rows);
    if (recipe.mtp_waves) {
      if (asked < 2 || asked > 3) {
        return Error(std::format(
            "draft_rows {} is outside {}'s 2 to 3 (its MTP block drafts two or three passes)",
            asked, arch));
      }
    } else {
      if (asked < 1 || asked > 16) {
        return Error(std::format("draft_rows {} is outside {}'s 1 to 16", asked, arch));
      }
    }
    s.draft_rows = Pick<std::uint32_t>(U32(draft_rows), basis("draft_rows"), {}, {}, {}, {},
                                       recipe.default_draft_rows, "measured on a GB10");
  }

  // DeepSeek V4's waves and prefill arithmetic.
  std::optional<config::WaveForm> form;
  if (const std::string text = entry.Text("wave_form").value_or(""); text == "plain") {
    form = config::WaveForm::kPlain;
  } else if (text == "speculative") {
    form = config::WaveForm::kSpeculative;
  } else if (text == "auto") {
    form = config::WaveForm::kAuto;
  }
  s.wave_form = Pick<config::WaveForm>(form, basis("wave_form"), {}, {}, {}, {},
                                       config::WaveForm::kAuto, "chosen by counted acceptance");
  // By width: the fallback, then the widths calibrated, then the widths an
  // override names; the source is the last layer that gave any.
  std::vector<double> costs(config::kMaxWaveCostWidths, 0.0);
  std::ranges::copy(
      recipe.default_wave_costs.first(std::min(recipe.default_wave_costs.size(), costs.size())),
      costs.begin());
  std::string measured_widths;
  for (std::size_t w = 0; w < cal.wave_costs.size() && w < costs.size(); ++w) {
    if (const std::optional<double> cost = cal.wave_costs[w]; cost.has_value()) {
      costs[w] = *cost;
      measured_widths += std::format("{}{}", measured_widths.empty() ? "" : ", ", w + 2);
    }
  }
  const std::vector<double> asked_costs = entry.Reals("wave_costs").value_or(std::vector<double>{});
  s.wave_costs_override_count = asked_costs.size();
  for (std::size_t w = 0; w < asked_costs.size() && w < costs.size(); ++w) {
    costs[w] = asked_costs[w];
  }
  const std::string measured = measured_widths.empty()
                                   ? std::string()
                                   : std::format("widths {} {}", measured_widths, cal.basis);
  if (!asked_costs.empty()) {
    s.wave_costs = {std::move(costs), SettingSource::kOverride,
                    std::format("{}, widths 2 to {}{}{}", basis("wave_costs"),
                                asked_costs.size() + 1, measured.empty() ? "" : "; ", measured)};
  } else if (!measured.empty()) {
    s.wave_costs = {std::move(costs), SettingSource::kCalibrated,
                    measured + "; the rest measured on a GB10"};
  } else {
    s.wave_costs = {std::move(costs), SettingSource::kFallback,
                    "measured on a GB10, widths 2 to 8"};
  }
  // Qwen3.8's MTP and waves.
  s.draft_vocab = Pick<std::uint32_t>(U32(entry.Integer("draft_vocab")), basis("draft_vocab"), {},
                                      {}, {}, {}, recipe.default_draft_vocab, "measured on a GB10");
  // Keep the runner's refusal of a cap larger than its target vocabulary.
  if (recipe.mtp_waves && speculative && facts.vocabulary != 0 &&
      s.draft_vocab.value > facts.vocabulary) {
    return Error(std::format("draft_vocab exceeds the {} target vocabulary", recipe.name));
  }
  // The runner already caps selected heads; report that effective value
  // before registration, preserving a smaller owner-selected limit.
  if (recipe.mtp_waves && speculative && facts.drafter_selected_rows &&
      (s.draft_vocab.value == 0 || s.draft_vocab.value > *facts.drafter_selected_rows)) {
    s.draft_vocab.basis += std::format(" ({} requested; selected draft head has {} rows)",
                                       s.draft_vocab.value, *facts.drafter_selected_rows);
    s.draft_vocab.value = *facts.drafter_selected_rows;
    if (s.draft_vocab.source == SettingSource::kFallback) {
      s.draft_vocab.source = SettingSource::kDerived;
    }
  }
  if (facts.target_spec) {
    s.bound_spec = speculative ? facts.composed_spec : facts.target_spec;
    if (!s.bound_spec)
      return Error(facts.composed_spec_error.empty()
                       ? "component calibration has no active composed spec identity"
                       : facts.composed_spec_error);
  }
  if (speculative && facts.composed_draft_vocab &&
      *facts.composed_draft_vocab !=
          (s.draft_vocab.value == 0 ? facts.vocabulary : s.draft_vocab.value))
    return Error("artifact facts were derived for a different draft vocabulary");

  s.depth_cost_ratio =
      Pick<double>(entry.Real("depth_cost_ratio"), basis("depth_cost_ratio"), cal.depth_cost_ratio,
                   cal.basis, {}, {}, recipe.default_depth_cost_ratio, "measured on a GB10");
  s.shared_wave_depth =
      Pick<std::uint32_t>(U32(entry.Integer("shared_wave_depth")), basis("shared_wave_depth"), {},
                          {}, {}, {}, recipe.default_shared_wave_depth, "measured on a GB10");
  // A wave's drafts never exceed a step's.
  if (s.shared_wave_depth.value > s.draft_rows.value) {
    s.shared_wave_depth.value = s.draft_rows.value;
    s.shared_wave_depth.basis += std::format(" (at most draft_rows, {})", s.draft_rows.value);
  }
  s.draft_wave_max =
      Pick<std::uint32_t>(U32(entry.Integer("draft_wave_max")), basis("draft_wave_max"), {}, {}, {},
                          {}, recipe.default_draft_wave_max, "measured on a GB10");
  s.wave_lanes = Pick<bool>(entry.Bool("wave_lanes"), basis("wave_lanes"), {}, {}, {}, {},
                            recipe.default_wave_lanes,
                            "enabled when admitted backend lanes and workspace permit");
  s.wave_read_align =
      Pick<std::uint32_t>(U32(entry.Integer("wave_read_align")), basis("wave_read_align"), {}, {},
                          {}, {}, recipe.default_wave_read_align, "the plans' reuse across depths");
  return s;
}

std::expected<void, std::string> ResolveReasoning(ModelSettings& settings,
                                                  const config::ModelEntry& entry,
                                                  const FindToken& find) {
  const auto one = [&](std::string_view key, std::string_view marker,
                       Setting<std::string>& into) -> std::expected<void, std::string> {
    if (const auto text = entry.Text(key)) {
      if (!text->empty() && !find(*text)) {
        return Error(std::format("{} {} is not a token of the model's vocabulary", key,
                                 base::Printable(*text)));
      }
      into = {*text, SettingSource::kOverride, OverrideBasis(entry, key)};
      return {};
    }
    if (find(marker) && find(marker == kThinkStart ? kThinkEnd : kThinkStart)) {
      into = {std::string(marker), SettingSource::kDerived, "a token of the vocabulary"};
    } else {
      into = {std::string(), SettingSource::kFallback, "the vocabulary has no <think> pair"};
    }
    return {};
  };
  if (auto r = one("reasoning_start", kThinkStart, settings.reasoning_start); !r) {
    return r;
  }
  return one("reasoning_end", kThinkEnd, settings.reasoning_end);
}

std::vector<ModelSettings::Line> ModelSettings::Lines() const {
  std::vector<Line> lines;
  const auto add = [&](std::string_view key, const auto& setting) {
    lines.push_back({.key = key,
                     .value = Text(setting.value),
                     .source = setting.source,
                     .basis = setting.basis});
  };
  for (const config::ModelKeySpec& spec : config::ModelKeys()) {
    if (!spec.setting) {
      continue;
    }
    const std::uint8_t kind = composition ? config::kCompositionModels : config::kArtifactModels;
    if ((spec.kinds & kind) == 0 || !KeyAppliesTo(spec, *this)) {
      continue;
    }
    const std::string_view k = spec.key;
    if (k == "kv_cache_encoding") {
      add(k, kv_cache_encoding);
    } else if (k == "context") {
      add(k, context);
    } else if (k == "speculation") {
      add(k, speculation);
    } else if (k == "prefill_chunk") {
      add(k, prefill_chunk);
    } else if (k == "max_slots") {
      add(k, max_slots);
    } else if (k == "prefill_floor_tok_s") {
      add(k, prefill_floor_tok_s);
    } else if (k == "decode_floor_tok_s") {
      add(k, decode_floor_tok_s);
    } else if (k == "recompute_ms_per_token") {
      add(k, recompute_ms_per_token);
    } else if (k == "temperature") {
      add(k, temperature);
    } else if (k == "top_p") {
      add(k, top_p);
    } else if (k == "top_k") {
      add(k, top_k);
    } else if (k == "min_p") {
      add(k, min_p);
    } else if (k == "reasoning_start") {
      add(k, reasoning_start);
    } else if (k == "reasoning_end") {
      add(k, reasoning_end);
    } else if (k == "draft_rows") {
      add(k, draft_rows);
    } else if (k == "wave_form") {
      add(k, wave_form);
    } else if (k == "wave_costs") {
      add(k, wave_costs);
    } else if (k == "draft_vocab") {
      add(k, draft_vocab);
    } else if (k == "depth_cost_ratio") {
      add(k, depth_cost_ratio);
    } else if (k == "shared_wave_depth") {
      add(k, shared_wave_depth);
    } else if (k == "draft_wave_max") {
      add(k, draft_wave_max);
    } else if (k == "wave_lanes") {
      add(k, wave_lanes);
    } else if (k == "wave_read_align") {
      add(k, wave_read_align);
    } else if (k == "image_size") {
      add(k, image_size);
    } else if (k == "image_steps") {
      add(k, image_steps);
    }
  }
  return lines;
}

std::string ModelSettings::Summary() const {
  std::string out;
  for (const Line& line : Lines()) {
    out += std::format("{}{}={} ({})", out.empty() ? "" : " ", line.key, line.value,
                       SourceName(line.source));
  }
  return out;
}

int PrintSettings(const config::NodeConfig& config, bool json_output, std::FILE* out,
                  std::FILE* log) {
  const auto say = [log](std::string_view text) {
    const std::string line = std::format("llmp-runtime: {}\n", base::Printable(text));
    (void)std::fwrite(line.data(), 1, line.size(), log);
  };
  const fs::path& store = config.storage.installed;
  bool failed = false;
  std::string text = json_output ? R"({"models":[)" : "";
  bool first = true;
  for (const config::ModelEntry& entry : config.models) {
    OpenedModelArtifacts opened;
    auto facts = ReadArtifactFacts(entry, store, &opened);
    if (!facts) {
      say(std::format("model {}: {}", entry.name, facts.error()));
      failed = true;
      continue;
    }
    // Its calibration as registration would read it (calibration.h).
    CalibrationRead calibration{.calibration = std::nullopt, .note = "none", .refused = false};
    if (!entry.composition) {
      if (auto uncalibrated = ResolveSettings(entry, *facts, nullptr, /*plain=*/false)) {
        calibration = ReadCalibration(config.storage.state, CalibrationKeyOf(entry, *uncalibrated));
      }
    }
    auto settings = ResolveSettings(entry, *facts,
                                    calibration.calibration ? &*calibration.calibration : nullptr,
                                    /*plain=*/false);
    if (!settings) {
      say(std::format("model {}: {}", entry.name, settings.error()));
      failed = true;
      continue;
    }
    if (!entry.composition) {
      auto assets = opened.target ? ReadChatAssets(*opened.target, entry)
                                  : std::expected<ChatAssets, std::string>(
                                        std::unexpected("opened target artifact is missing"));
      if (!assets) {
        say(std::format("model {}: {}", entry.name, assets.error()));
        failed = true;
        continue;
      }
      const auto& tokens = assets->spec.tokens;
      const FindToken find = [&](std::string_view t) -> std::optional<std::int32_t> {
        const auto it = std::ranges::find(tokens, t);
        return it == tokens.end() ? std::nullopt
                                  : std::optional(static_cast<std::int32_t>(it - tokens.begin()));
      };
      if (auto r = ResolveReasoning(*settings, entry, find); !r) {
        say(std::format("model {}: {}", entry.name, r.error()));
        failed = true;
        continue;
      }
    }
    if (json_output) {
      text += first ? "{\"name\":" : ",{\"name\":";
      json::AppendQuoted(entry.name, text);
      text += ",\"architecture\":";
      json::AppendQuoted(settings->composition ? "composition" : settings->architecture, text);
      text += ",\"settings\":{";
      bool first_line = true;
      for (const ModelSettings::Line& line : settings->Lines()) {
        text += std::format(R"({}"{}":{{"value":{},"source":"{}","basis":)", first_line ? "" : ",",
                            line.key, line.value, SourceName(line.source));
        json::AppendQuoted(line.basis, text);
        text += "}";
        first_line = false;
      }
      text += "},\"ignored\":[";
      for (std::size_t i = 0; i < settings->ignored.size(); ++i) {
        text += i == 0 ? "" : ",";
        json::AppendQuoted(settings->ignored[i], text);
      }
      text += "],\"warnings\":[";
      for (std::size_t i = 0; i < settings->warnings.size(); ++i) {
        text += i == 0 ? "" : ",";
        json::AppendQuoted(settings->warnings[i], text);
      }
      text += "],\"calibration\":";
      json::AppendQuoted(calibration.note, text);
      text += "}";
    } else {
      text += std::format("model {} ({}{}; calibration {})\n", entry.name,
                          settings->composition ? "composition" : settings->architecture,
                          settings->drafter ? ", with a drafter" : "",
                          base::Printable(calibration.note));
      for (const ModelSettings::Line& line : settings->Lines()) {
        text += std::format("  {:<26} {:<24} {:<10} {}\n", line.key, base::Printable(line.value),
                            SourceName(line.source), base::Printable(line.basis));
      }
      for (const std::string& ignored : settings->ignored) {
        text += std::format("  ignored: {}\n", ignored);
      }
      for (const std::string& warning : settings->warnings) {
        text += std::format("  warning: {}\n", warning);
      }
    }
    first = false;
  }
  if (json_output) {
    text += "]}\n";
  }
  (void)std::fwrite(text.data(), 1, text.size(), out);
  (void)std::fflush(out);
  (void)std::fflush(log);
  return failed ? 1 : 0;
}

}  // namespace llmp::runtime
