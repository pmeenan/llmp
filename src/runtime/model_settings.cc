// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "runtime/model_settings.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <charconv>
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
#include "model/dsv4.h"
#include "model/qwen38.h"
#include "platform/path_trust.h"
#include "runtime/calibration.h"
#include "tokenizer/gguf.h"
#include "tokenizer/hf.h"

namespace jitllm::runtime {
namespace {

namespace fs = std::filesystem;
namespace ja = jitllm::artifact;
namespace json = jitllm::base::json;

std::unexpected<std::string> Error(std::string what) { return std::unexpected(std::move(what)); }

std::string Errno(int error) { return std::strerror(error); }  // NOLINT(concurrency-mt-unsafe)

// A kept metadata file's name (`meta/<name>`), if the artifact lists it.
bool Keeps(const ja::Artifact& a, std::string_view name) {
  return std::ranges::any_of(a.files(), [&](const ja::ListedFile& f) {
    return f.role == ja::FileRole::kSourceMetadata && f.path.starts_with("meta/") &&
           std::string_view(f.path).substr(5) == name;
  });
}

// The kept GGUF metadata (import rule 7): a single file's, or the first
// shard's; empty when there is none.
std::string KeptGguf(const ja::Artifact& a) {
  std::string kv;
  for (const ja::ListedFile& f : a.files()) {
    if (f.role == ja::FileRole::kSourceMetadata && f.path.starts_with("meta/") &&
        f.path.ends_with(".kv.gguf") && (kv.empty() || f.path.contains("-00001-of-"))) {
      kv = f.path.substr(5);
    }
  }
  return kv;
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

// GGUF's F32 value type.
constexpr std::uint32_t kGgufF32 = 6;

// A float as the decimal it was written as: its shortest round-trip text
// read as a double (0.95f is 0.95, not 0.949999988...).
double Decimal(float value) {
  std::array<char, 32> text{};
  const auto written = std::to_chars(text.data(), text.data() + text.size(), value);
  double out = value;
  (void)std::from_chars(text.data(), written.ptr, out);
  return out;
}

std::optional<double> Real(const ja::GgufMetadata& m, std::string_view key) {
  const auto it = m.find(key);
  if (it == m.end()) {
    return std::nullopt;
  }
  if (it->second.kind == ja::GgufValue::Kind::kFloat) {
    return it->second.type == kGgufF32 ? Decimal(static_cast<float>(it->second.real))
                                       : it->second.real;
  }
  if (it->second.kind == ja::GgufValue::Kind::kInteger) {
    return static_cast<double>(it->second.integer);
  }
  return std::nullopt;
}

std::optional<std::int64_t> Integer(const ja::GgufMetadata& m, std::string_view key) {
  const auto it = m.find(key);
  if (it == m.end() || it->second.kind != ja::GgufValue::Kind::kInteger) {
    return std::nullopt;
  }
  return it->second.integer;
}

std::optional<double> JsonReal(const json::Value& object, std::string_view key) {
  if (!object.is_object()) {
    return std::nullopt;
  }
  const auto v = object.find(key);
  return v && v->is_number() ? v->float64() : std::nullopt;
}

std::optional<std::int64_t> JsonInteger(const json::Value& object, std::string_view key) {
  if (!object.is_object()) {
    return std::nullopt;
  }
  const auto v = object.find(key);
  return v && v->is_number() ? v->int64() : std::nullopt;
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

// A rope-scaled checkpoint's trained context (YaRN and the like): its
// original context times the scaling factor, where that is longer than the
// context its metadata states (GGUFs state DeepSeek V4 Flash's scaled
// 1,048,576 already; others state the original).
void TrainedByScaling(ArtifactFacts& facts, std::optional<double> factor,
                      std::optional<std::int64_t> original, const std::string& from) {
  if (!factor || !original || !std::isfinite(*factor) || *factor <= 1 || *factor > 1024 ||
      *original <= 0 || *original > (std::int64_t{1} << 32)) {
    return;
  }
  const auto scaled =
      static_cast<std::uint64_t>(std::floor(static_cast<double>(*original) * *factor));
  if (!facts.trained_context || scaled > *facts.trained_context) {
    facts.trained_context = scaled;
    facts.trained_context_from = from;
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

std::uint32_t RunnerContextCeiling(std::string_view architecture) {
  if (architecture == "deepseek4") {
    return model::kDsv4FlashContext;
  }
  if (architecture == "qwen4exp") {
    return model::kQwen38FlashContext;
  }
  return 0;
}

std::expected<ja::Artifact, std::string> OpenInstalled(const fs::path& store,
                                                       const std::string& id) {
  ja::OpenOptions options;
  options.expected_id = id;
  options.trusted_owner = ::geteuid();
  auto opened = ja::Artifact::Open(store / id, options);
  if (!opened) {
    return Error(std::format("artifact {}: {}", id, opened.error().ToString()));
  }
  return std::move(*opened);
}

std::expected<std::string, std::string> ReadTrustedFile(const fs::path& path, uid_t trusted,
                                                        std::size_t limit) {
  auto walked = platform::WalkTrusted(path, trusted, false);
  if (!walked) {
    return Error(std::format("{}: {}", path.string(), walked.error()));
  }
  if (!walked->exists) {
    return Error(std::format("{} does not exist", path.string()));
  }
  const int fd =
      ::open(walked->resolved.c_str(), O_RDONLY | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC | O_NOCTTY);
  if (fd < 0) {
    return Error(std::format("{}: {}", path.string(), Errno(errno)));
  }
  struct stat status{};
  std::string bytes;
  std::string problem;
  if (::fstat(fd, &status) != 0) {
    problem = Errno(errno);
  } else if (!S_ISREG(status.st_mode) || status.st_dev != walked->status.st_dev ||
             status.st_ino != walked->status.st_ino) {
    problem = "not the regular file the trust walk approved";
  } else if (platform::OthersCanWrite(status, trusted, fd)) {
    problem = "other users can write it";
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

std::vector<std::string> GgufFactKeys(std::string_view architecture) {
  return {std::format("{}.context_length", architecture),
          std::format("{}.rope.scaling.factor", architecture),
          std::format("{}.rope.scaling.original_context_length", architecture),
          "general.sampling.temp",
          "general.sampling.top_p",
          "general.sampling.top_k",
          "general.sampling.min_p"};
}

void FactsFromGguf(const ja::GgufMetadata& m, ArtifactFacts& facts) {
  const std::vector<std::string> keys = GgufFactKeys(facts.architecture);
  if (const auto context = Integer(m, keys[0]); context && *context > 0) {
    facts.trained_context = static_cast<std::uint64_t>(*context);
    facts.trained_context_from = keys[0];
  }
  TrainedByScaling(facts, Real(m, keys[1]), Integer(m, keys[2]),
                   std::format("{} times {}", keys[2], keys[1]));
  Sampling(facts, Real(m, keys[3]), Real(m, keys[4]), Integer(m, keys[5]), Real(m, keys[6]),
           "general.sampling");
}

std::expected<void, std::string> FactsFromConfigJson(std::string_view text, ArtifactFacts& facts) {
  auto doc = json::Parse(text);
  if (!doc) {
    return Error(std::format("config.json: {}", doc.error().ToString()));
  }
  // The language model's own keys: at the top, or in text_config.
  json::Value config = doc->root();
  std::string prefix = "config.json ";
  if (!JsonInteger(config, "max_position_embeddings") && config.is_object()) {
    if (const auto text_config = config.find("text_config")) {
      config = *text_config;
      prefix = "config.json text_config.";
    }
  }
  if (const auto context = JsonInteger(config, "max_position_embeddings");
      context && *context > 0) {
    facts.trained_context = static_cast<std::uint64_t>(*context);
    facts.trained_context_from = prefix + "max_position_embeddings";
  }
  if (const auto scaling = config.is_object() ? config.find("rope_scaling") : std::nullopt) {
    TrainedByScaling(facts, JsonReal(*scaling, "factor"),
                     JsonInteger(*scaling, "original_max_position_embeddings"),
                     prefix + "rope_scaling");
  }
  return {};
}

std::expected<void, std::string> FactsFromGenerationConfig(std::string_view text,
                                                           ArtifactFacts& facts) {
  auto doc = json::Parse(text);
  if (!doc) {
    return Error(std::format("generation_config.json: {}", doc.error().ToString()));
  }
  const json::Value root = doc->root();
  // do_sample false is greedy whatever the temperature says (as
  // transformers reads it).
  std::optional<double> temperature = JsonReal(root, "temperature");
  std::string_view from = "generation_config.json";
  if (const auto greedy = root.is_object() ? root.find("do_sample") : std::nullopt;
      greedy && greedy->is_bool() && !greedy->boolean()) {
    temperature = 0.0;
    from = "generation_config.json (do_sample false)";
  }
  Sampling(facts, temperature, JsonReal(root, "top_p"), JsonInteger(root, "top_k"),
           JsonReal(root, "min_p"), from);
  return {};
}

std::expected<ArtifactFacts, std::string> ArtifactFactsOf(const ja::Artifact& target,
                                                          const ja::Artifact* drafter) {
  ArtifactFacts facts;
  facts.architecture = target.model().architecture;
  // generation_config.json first: GGUF's general.sampling keys are copied
  // from it where a converter kept them.
  if (Keeps(target, "generation_config.json")) {
    auto text = target.ReadMetadata("generation_config.json");
    if (!text) {
      return Error(std::format("generation_config.json: {}", text.error().ToString()));
    }
    if (auto read = FactsFromGenerationConfig(*text, facts); !read) {
      return std::unexpected(read.error());
    }
  }
  if (const std::string kv = KeptGguf(target); !kv.empty()) {
    const std::vector<std::string> names = GgufFactKeys(facts.architecture);
    const std::vector<std::string_view> keys(names.begin(), names.end());
    auto m = ReadKeptGguf(target, kv, keys);
    if (!m) {
      return std::unexpected(m.error());
    }
    FactsFromGguf(*m, facts);
  } else if (Keeps(target, "config.json")) {
    auto text = target.ReadMetadata("config.json");
    if (!text) {
      return Error(std::format("config.json: {}", text.error().ToString()));
    }
    if (auto read = FactsFromConfigJson(*text, facts); !read) {
      return std::unexpected(read.error());
    }
  }
  if (drafter != nullptr) {
    facts.drafter_architecture = drafter->model().architecture;
    if (facts.drafter_architecture == "qwen4exp-mtp") {
      const auto head = drafter->FindResource("draft_output.weight");
      const auto ids = drafter->FindResource("draft_output.ids");
      if (head.has_value() != ids.has_value()) {
        return Error("the selected draft head requires its weight and token map");
      }
      if (head) {
        const auto& weight = drafter->resources()[*head].repr;
        const auto& map = drafter->resources()[*ids].repr;
        if (weight.family != ja::Family::kGgml || weight.type != "BF16" ||
            weight.dims.size() != 2 || weight.dims[1] == 0 ||
            weight.dims[1] > std::numeric_limits<std::uint32_t>::max() ||
            map.family != ja::Family::kGgml || map.type != "I32" ||
            map.dims != std::vector<std::uint64_t>{1, weight.dims[1]}) {
          return Error("the selected draft head requires BF16 rows and a matching I32 token map");
        }
        facts.drafter_selected_rows = static_cast<std::uint32_t>(weight.dims[1]);
      }
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
  return facts;
}

std::expected<ArtifactFacts, std::string> ReadArtifactFacts(const config::ModelEntry& entry,
                                                            const fs::path& store) {
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
  return ArtifactFactsOf(*target, drafter ? &*drafter : nullptr);
}

std::expected<ChatAssets, std::string> ReadChatAssets(const ja::Artifact& target,
                                                      const config::ModelEntry& entry,
                                                      uid_t trusted) {
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
      auto read = ReadTrustedFile(*named, trusted, kMaxTokenizerBytes);
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
    auto json_text = source("tokenizer.json", entry.tokenizer);
    if (!json_text) {
      return std::unexpected(json_text.error());
    }
    if (!*json_text) {
      return Error(
          std::format("the artifact keeps no GGUF metadata or tokenizer.json and "
                      "models.{}.tokenizer is not set",
                      entry.name));
    }
    auto spec = tokenizer::ReadHfTokenizer(**json_text);
    if (!spec) {
      return Error("tokenizer: " + spec.error().ToString());
    }
    assets.spec = std::move(*spec);
    assets.from = Keeps(target, "tokenizer.json") ? std::string("tokenizer.json")
                                                  : entry.tokenizer.value_or(fs::path()).string();
  }
  auto text = source("chat_template.jinja", entry.chat_template);
  if (!text) {
    return std::unexpected(text.error());
  }
  assets.chat_template = std::move(*text);
  return assets;
}

std::expected<ModelSettings, std::string> ResolveSettings(const config::ModelEntry& entry,
                                                          const ArtifactFacts& facts,
                                                          const Calibration* calibration,
                                                          bool plain) {
  ModelSettings s;
  s.name = entry.name;
  s.architecture = facts.architecture;
  s.composition = facts.composition;
  s.drafter = entry.drafter.has_value();
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
  const std::uint32_t runner = RunnerContextCeiling(arch);
  if (runner == 0) {
    return Error(std::format("no runner for architecture {}", arch));
  }
  for (const auto& [key, value] : entry.overrides) {
    const config::ModelKeySpec* spec = config::FindModelKey(key);
    if (spec != nullptr && !config::KeyAppliesTo(*spec, arch)) {
      s.ignored.push_back(std::format("{} (a setting of {} only)", key, spec->architectures));
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
  } else if (ceiling < config::kDefaultContext) {
    s.context = {ceiling, SettingSource::kDerived, ceiling_from};
  } else {
    s.context = {config::kDefaultContext, SettingSource::kFallback,
                 std::format("the default, within {}", ceiling_from)};
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
  const bool speculative = s.speculation.value;

  // Speed trades: calibrated on the machine, else M3's measured constants.
  const bool dsv4 = arch == "deepseek4";
  const bool deep = dsv4 && context > kDsv4WidePrefillContext;
  std::uint32_t rows = kQwen38PrefillRows;
  if (dsv4) {
    rows = deep ? kDsv4DeepPrefillRows : kDsv4PrefillRows;
  }
  s.prefill_chunk = Pick<std::uint32_t>(
      U32(entry.Integer("prefill_chunk")), basis("prefill_chunk"), cal.prefill_chunk, cal.basis, {},
      {}, rows,
      deep ? "measured on a GB10 (above 262,144 tokens: the mask's memory)" : "measured on a GB10");
  s.max_slots =
      Pick<std::uint32_t>(U32(entry.Integer("max_slots")), basis("max_slots"), cal.max_slots,
                          cal.basis, {}, {}, kDefaultSlots, "the knee measured on a GB10 (D-104)");
  if (dsv4 && speculative && s.max_slots.value > kDsv4SpeculativeMostSlots) {
    s.max_slots.basis +=
        std::format(", at most {}: a DSpark verify takes two of a wave's sixteen rows",
                    kDsv4SpeculativeMostSlots);
    s.max_slots.value = kDsv4SpeculativeMostSlots;
  }
  s.prefill_floor_tok_s = Pick<std::uint32_t>(
      U32(entry.Integer("prefill_floor_tok_s")), basis("prefill_floor_tok_s"),
      cal.prefill_floor_tok_s, cal.basis, {}, {}, config::kDefaultPrefillFloor, "the default");
  s.decode_floor_tok_s = Pick<std::uint32_t>(
      U32(entry.Integer("decode_floor_tok_s")), basis("decode_floor_tok_s"), cal.decode_floor_tok_s,
      cal.basis, {}, {}, config::kDefaultDecodeFloor, "the default");
  s.recompute_ms_per_token =
      Pick<double>(entry.Real("recompute_ms_per_token"), basis("recompute_ms_per_token"),
                   cal.recompute_ms_per_token, cal.basis, {}, {}, kRecomputeMsPerToken,
                   "DeepSeek's prefill measured on a GB10");

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
  if (dsv4) {
    // The drafter's block (0: not kept) bounds its drafts.
    const std::uint64_t block = facts.drafter_block.value_or(0);
    const std::int64_t asked = draft_rows.value_or(0);
    if (block != 0 && std::cmp_greater(asked, block)) {
      return Error(
          std::format("draft_rows {} is more than the drafter's block of {} drafts", asked, block));
    }
    std::optional<std::uint32_t> derived;
    std::string derived_basis;
    if (block != 0 && block < kDsv4DraftRows) {
      derived = static_cast<std::uint32_t>(block);
      derived_basis = std::format("{}.block_size {}", facts.drafter_architecture, block);
    }
    s.draft_rows = Pick<std::uint32_t>(U32(draft_rows), basis("draft_rows"), {}, {}, derived,
                                       derived_basis, kDsv4DraftRows, "measured on a GB10");
  } else {
    const std::int64_t asked = draft_rows.value_or(kQwen38DraftRows);
    if (asked < 2 || asked > 3) {
      return Error(std::format(
          "draft_rows {} is outside {}'s 2 to 3 (its MTP block drafts two or three passes)", asked,
          arch));
    }
    s.draft_rows = Pick<std::uint32_t>(U32(draft_rows), basis("draft_rows"), {}, {}, {}, {},
                                       kQwen38DraftRows, "measured on a GB10");
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
  std::vector<double> costs(kDsv4WaveCosts.begin(), kDsv4WaveCosts.end());
  std::string measured_widths;
  for (std::size_t w = 0; w < cal.wave_costs.size() && w < costs.size(); ++w) {
    if (const std::optional<double> cost = cal.wave_costs[w]; cost.has_value()) {
      costs[w] = *cost;
      measured_widths += std::format("{}{}", measured_widths.empty() ? "" : ", ", w + 2);
    }
  }
  const std::vector<double> asked_costs = entry.Reals("wave_costs").value_or(std::vector<double>{});
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
  s.prefill_outa_hca =
      Pick<bool>(entry.Bool("prefill_outa_hca"), basis("prefill_outa_hca"), {}, {}, {}, {}, true,
                 "the served default, qualified (docs/experiments/ds4-output-prefix)");
  s.prefill_outa_hca_partial =
      Pick<bool>(entry.Bool("prefill_outa_hca_partial"), basis("prefill_outa_hca_partial"), {}, {},
                 {}, {}, true, "the served default, qualified under the tie-aware rule");
  // In effect only with the full-chunk form (engine/dsv4_plan.h).
  if (!s.prefill_outa_hca.value && s.prefill_outa_hca_partial.value) {
    s.prefill_outa_hca_partial.value = false;
    s.prefill_outa_hca_partial.basis += " (off: prefill_outa_hca is off)";
  }

  // Qwen3.8's MTP and waves.
  s.draft_vocab = Pick<std::uint32_t>(U32(entry.Integer("draft_vocab")), basis("draft_vocab"), {},
                                      {}, {}, {}, kQwen38DraftVocab, "measured on a GB10");
  // Keep the runner's refusal of a cap larger than its target vocabulary.
  if (arch == "qwen4exp" && speculative && s.draft_vocab.value > model::kQwen38FlashVocab) {
    return Error("draft_vocab exceeds the Qwen3.8 target vocabulary");
  }
  // The runner already caps selected heads; report that effective value
  // before registration, preserving a smaller owner-selected limit.
  if (arch == "qwen4exp" && speculative && facts.drafter_selected_rows &&
      (s.draft_vocab.value == 0 || s.draft_vocab.value > *facts.drafter_selected_rows)) {
    s.draft_vocab.basis += std::format(" ({} requested; selected draft head has {} rows)",
                                       s.draft_vocab.value, *facts.drafter_selected_rows);
    s.draft_vocab.value = *facts.drafter_selected_rows;
    if (s.draft_vocab.source == SettingSource::kFallback) {
      s.draft_vocab.source = SettingSource::kDerived;
    }
  }
  s.depth_cost_ratio =
      Pick<double>(entry.Real("depth_cost_ratio"), basis("depth_cost_ratio"), cal.depth_cost_ratio,
                   cal.basis, {}, {}, kQwen38DepthCostRatio, "measured on a GB10");
  s.shared_wave_depth =
      Pick<std::uint32_t>(U32(entry.Integer("shared_wave_depth")), basis("shared_wave_depth"), {},
                          {}, {}, {}, kQwen38SharedWaveDepth, "measured on a GB10");
  // A wave's drafts never exceed a step's.
  if (s.shared_wave_depth.value > s.draft_rows.value) {
    s.shared_wave_depth.value = s.draft_rows.value;
    s.shared_wave_depth.basis += std::format(" (at most draft_rows, {})", s.draft_rows.value);
  }
  s.draft_wave_max =
      Pick<std::uint32_t>(U32(entry.Integer("draft_wave_max")), basis("draft_wave_max"), {}, {}, {},
                          {}, kQwen38DraftWaveMax, "measured on a GB10");
  s.wave_lanes = Pick<bool>(entry.Bool("wave_lanes"), basis("wave_lanes"), {}, {}, {}, {}, true,
                            "measured four-request concurrent streams on a GB10");
  s.wave_read_align =
      Pick<std::uint32_t>(U32(entry.Integer("wave_read_align")), basis("wave_read_align"), {}, {},
                          {}, {}, kQwen38WaveReadAlign, "the plans' reuse across depths");
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
    if ((spec.kinds & kind) == 0 || (!composition && !config::KeyAppliesTo(spec, architecture))) {
      continue;
    }
    const std::string_view k = spec.key;
    if (k == "context") {
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
    } else if (k == "prefill_outa_hca") {
      add(k, prefill_outa_hca);
    } else if (k == "prefill_outa_hca_partial") {
      add(k, prefill_outa_hca_partial);
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
    const std::string line = std::format("jitllm-runtime: {}\n", base::Printable(text));
    (void)std::fwrite(line.data(), 1, line.size(), log);
  };
  const fs::path& store = config.storage.installed;
  const uid_t self = ::geteuid();
  bool failed = false;
  std::string text = json_output ? R"({"models":[)" : "";
  bool first = true;
  for (const config::ModelEntry& entry : config.models) {
    auto facts = ReadArtifactFacts(entry, store);
    if (!facts) {
      say(std::format("model {}: {}", entry.name, facts.error()));
      failed = true;
      continue;
    }
    // Its calibration as registration would read it (calibration.h).
    CalibrationRead calibration{.calibration = std::nullopt, .note = "none", .refused = false};
    if (!entry.composition) {
      if (auto uncalibrated = ResolveSettings(entry, *facts, nullptr, /*plain=*/false)) {
        calibration =
            ReadCalibration(config.storage.state, CalibrationKeyOf(entry, *uncalibrated), self);
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
      auto target = OpenInstalled(store, entry.artifact.value_or(""));
      auto assets = target
                        ? ReadChatAssets(*target, entry, self)
                        : std::expected<ChatAssets, std::string>(std::unexpected(target.error()));
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

}  // namespace jitllm::runtime
