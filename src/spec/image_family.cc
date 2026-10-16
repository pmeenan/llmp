// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "spec/image_family.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <format>
#include <limits>
#include <optional>
#include <string>
#include <system_error>
#include <utility>
#include <variant>
#include <vector>

#include "spec/mappings/qwen_image.h"

namespace llmp::spec {
namespace {

std::unexpected<std::string> Refused(std::string why) {
  return std::unexpected("image family: " + std::move(why));
}

// ------------------------------------------------------------------ schema

enum class Type : std::uint8_t { kInt, kReal, kInts, kReals };

// A family parameter: its kind, name and type; a number's (and each list
// member's) range, and a list's length range.
struct Schema {
  std::string_view kind{};
  std::string_view param{};
  Type type = Type::kInt;
  double least = 0.0;
  double most = 0.0;
  std::size_t shortest = 0;
  std::size_t longest = 0;
  bool positive = false;  // a real above 0 (`least` 0)
};

constexpr double kBig = 1e12;
constexpr double kWide = static_cast<double>(std::uint32_t{1} << 24U);

constexpr std::array kSchema{
    // The text encoder's transformer.
    Schema{kImageTextKind, "layers", Type::kInt, 1, 4096},
    Schema{kImageTextKind, "width", Type::kInt, 1, kWide},
    Schema{kImageTextKind, "heads", Type::kInt, 1, 4096},
    Schema{kImageTextKind, "kv_heads", Type::kInt, 1, 4096},
    Schema{kImageTextKind, "head_dim", Type::kInt, 2, 4096},
    Schema{kImageTextKind, "ffn", Type::kInt, 1, kWide},
    Schema{kImageTextKind, "vocab", Type::kInt, 1, kWide},
    Schema{kImageTextKind, "rms_eps", Type::kReal, 0, 1, 0, 0, true},
    Schema{kImageTextKind, "rope_theta", Type::kReal, 1, kBig},
    // The flow transformer.
    Schema{kImageFlowKind, "blocks", Type::kInt, 1, 4096},
    Schema{kImageFlowKind, "heads", Type::kInt, 1, 4096},
    Schema{kImageFlowKind, "head_dim", Type::kInt, 2, 4096},
    Schema{kImageFlowKind, "mlp_ratio", Type::kInt, 1, 64},
    Schema{kImageFlowKind, "in_channels", Type::kInt, 1, 65536},
    Schema{kImageFlowKind, "out_channels", Type::kInt, 1, 65536},
    Schema{kImageFlowKind, "context", Type::kInt, 1, kWide},
    Schema{kImageFlowKind, "timestep_dim", Type::kInt, 2, 65536},
    Schema{kImageFlowKind, "rope_axes", Type::kInts, 2, 4096, 3, 3},
    Schema{kImageFlowKind, "rope_theta", Type::kReal, 1, kBig},
    Schema{kImageFlowKind, "eps", Type::kReal, 0, 1, 0, 0, true},
    Schema{kImageFlowKind, "timestep_divisor", Type::kReal, 1, kBig},
    Schema{kImageFlowKind, "timestep_scale", Type::kReal, 1, kBig},
    Schema{kImageFlowKind, "timestep_period", Type::kReal, 2, kBig},
    // The flow-matching schedule.
    Schema{kImageScheduleKind, "train_timesteps", Type::kInt, 1, kWide},
    Schema{kImageScheduleKind, "base_image_seq_len", Type::kInt, 1, kWide},
    Schema{kImageScheduleKind, "max_image_seq_len", Type::kInt, 1, kWide},
    Schema{kImageScheduleKind, "base_shift", Type::kReal, -1024, 1024},
    Schema{kImageScheduleKind, "max_shift", Type::kReal, -1024, 1024},
    Schema{kImageScheduleKind, "shift_terminal", Type::kReal, 0, 1, 0, 0, true},
    // The VAE's decoder.
    Schema{kImageVaeKind, "z_dim", Type::kInt, 1, 65536},
    Schema{kImageVaeKind, "base", Type::kInt, 1, 65536},
    Schema{kImageVaeKind, "dim_mult", Type::kInts, 1, 64, 2, 16},
    Schema{kImageVaeKind, "res_blocks", Type::kInt, 0, 64},
    Schema{kImageVaeKind, "temporal_upsample", Type::kInts, 0, 1, 1, 16},
    Schema{kImageVaeKind, "out_channels", Type::kInt, 1, 64},
    Schema{kImageVaeKind, "spatial_factor", Type::kInt, 1, 1024},
    Schema{kImageVaeKind, "latents_mean", Type::kReals, -kBig, kBig, 1, 65536},
    Schema{kImageVaeKind, "latents_std", Type::kReals, 0, kBig, 1, 65536, true},
};

bool KnownKind(std::string_view kind) {
  return std::ranges::contains(kSchema, kind, &Schema::kind);
}

// A number within the schema's range.
bool InRange(const Schema& s, double v) {
  return std::isfinite(v) && v >= s.least && v <= s.most && (!s.positive || v > 0);
}

// A value read as the schema's type, checked; nullopt where it is not.
std::optional<ParamValue> Typed(const Schema& s, const ParamValue& v, bool reverse) {
  switch (s.type) {
    case Type::kInt: {
      double d = 0;
      if (const auto* i = std::get_if<std::int64_t>(&v))
        d = static_cast<double>(*i);
      else if (const auto* r = std::get_if<double>(&v); r && std::trunc(*r) == *r)
        d = *r;
      else
        return std::nullopt;
      if (!InRange(s, d)) return std::nullopt;
      return ParamValue(static_cast<std::int64_t>(d));
    }
    case Type::kReal: {
      double d = 0;
      if (const auto* i = std::get_if<std::int64_t>(&v))
        d = static_cast<double>(*i);
      else if (const auto* r = std::get_if<double>(&v))
        d = *r;
      else
        return std::nullopt;
      if (!InRange(s, d)) return std::nullopt;
      return ParamValue(d);
    }
    case Type::kInts: {
      const auto* list = std::get_if<std::vector<std::int64_t>>(&v);
      if (!list || list->size() < s.shortest || list->size() > s.longest) return std::nullopt;
      std::vector<std::int64_t> out = *list;
      if (reverse) std::ranges::reverse(out);
      for (const auto i : out)
        if (!InRange(s, static_cast<double>(i))) return std::nullopt;
      return ParamValue(std::move(out));
    }
    case Type::kReals: {
      std::vector<double> out;
      if (const auto* reals = std::get_if<std::vector<double>>(&v))
        out = *reals;
      else if (const auto* ints = std::get_if<std::vector<std::int64_t>>(&v))
        for (const auto i : *ints) out.push_back(static_cast<double>(i));
      else
        return std::nullopt;
      if (out.size() < s.shortest || out.size() > s.longest) return std::nullopt;
      if (reverse) std::ranges::reverse(out);
      for (const auto d : out)
        if (!InRange(s, d)) return std::nullopt;
      return ParamValue(std::move(out));
    }
  }
  return std::nullopt;
}

const ParamValue* ValueOf(std::span<const SourceValue> values, std::string_view key) {
  const auto found = std::ranges::find(values, key, &SourceValue::key);
  return found == values.end() ? nullptr : &found->value;
}

// A real as the float its shortest decimal names (what a compiled float
// literal of that decimal is), not the double rounded again.
float ToFloat(double d) {
  std::array<char, 64> text{};
  const auto written = std::to_chars(text.data(), text.data() + text.size(), d);
  float out = 0.0f;
  const auto read = std::from_chars(text.data(), written.ptr, out);
  return read.ec == std::errc() ? out : static_cast<float>(d);
}

// A stage's stream width: its family's (the first kind's).
std::uint64_t WidthOf(const Stage& stage, const model::QwenImageProfile& p) {
  const auto& kind = stage.layers.front().components.front().kind;
  return kind == kImageTextKind   ? p.text.width
         : kind == kImageFlowKind ? p.denoiser.width
                                  : p.vae.z_dim;
}

std::string_view WhereOf(const ImageParamRule& r) {
  return !r.role.empty() ? r.role : (!r.file.empty() ? r.file : std::string_view("a constant"));
}

std::expected<model::QwenImageProfile, std::string> ProfileOf(const ModelSpec& spec,
                                                              const ImageMapping& mapping,
                                                              bool widths);

}  // namespace

// ----------------------------------------------------------------- mappings

std::span<const ImageMapping* const> ImageMappings() {
  static const std::array<const ImageMapping*, 1> kAll{&mappings::QwenImage21Mapping()};
  return kAll;
}

const ImageMapping* FindImageMapping(std::string_view architecture) {
  for (const ImageMapping* m : ImageMappings())
    if (m->architecture == architecture) return m;
  return nullptr;
}

std::expected<void, std::string> ValidateImageMapping(const ImageMapping& m) {
  if (m.architecture.empty() || m.components.empty() || m.components.size() > kMaxStages)
    return Refused("an architecture and one to sixteen components");
  for (std::size_t i = 0; i < m.components.size(); ++i) {
    const auto& c = m.components[i];
    if (c.role.empty() || !IsSourceKey(c.role) || c.stage.empty() || c.kinds.empty() ||
        c.architectures.empty() || (i > 0 && !(m.components[i - 1].role < c.role)))
      return Refused(
          std::format("component {}: a role, a stage, kinds and architectures, "
                      "sorted by role",
                      i));
    for (const auto kind : c.kinds)
      if (!KnownKind(kind)) return Refused(std::format("{}: an unknown kind {}", c.role, kind));
  }
  // Every schema parameter of a used kind: exactly one rule.
  for (const auto& s : kSchema) {
    const bool used = std::ranges::any_of(
        m.components, [&](const auto& c) { return std::ranges::contains(c.kinds, s.kind); });
    const auto rules = std::ranges::count_if(
        m.params, [&](const auto& r) { return r.kind == s.kind && r.param == s.param; });
    if (used != (rules == 1))
      return Refused(
          std::format("{}.{}: one rule exactly where its kind is used", s.kind, s.param));
  }
  for (const auto& r : m.params) {
    const auto schema = std::ranges::find_if(
        kSchema, [&](const auto& s) { return s.kind == r.kind && s.param == r.param; });
    if (schema == kSchema.end()) return Refused(std::format("{}.{}: unknown", r.kind, r.param));
    const bool keyed = !r.key.empty();
    if ((keyed && r.role.empty() == r.file.empty()) ||
        (!keyed && (!r.role.empty() || !r.file.empty())))
      return Refused(
          std::format("{}.{}: a key of one role or file, or a constant", r.kind, r.param));
    if (keyed && !IsSourceKey(r.key)) return Refused(std::format("{}: not a source key", r.key));
    if (!r.role.empty() && !std::ranges::contains(m.components, r.role, &ImageComponentRule::role))
      return Refused(std::format("{}.{}: no component {}", r.kind, r.param, r.role));
    if (!keyed && (schema->type == Type::kInts || schema->type == Type::kReals))
      return Refused(std::format("{}.{}: a list from a key only", r.kind, r.param));
    if (!keyed &&
        !Typed(*schema,
               schema->type == Type::kInt ? ParamValue(r.value.integer) : ParamValue(r.value.real),
               false))
      return Refused(std::format("{}.{}: a constant out of range", r.kind, r.param));
  }
  for (const auto& q : m.requirements) {
    if (q.key.empty() || !IsSourceKey(q.key) || q.role.empty() == q.file.empty() ||
        (!q.role.empty() &&
         !std::ranges::contains(m.components, q.role, &ImageComponentRule::role)))
      return Refused(std::format("requirement {}: a key of one role or file", q.key));
  }
  const auto& n = m.names;
  const auto present = [](std::string_view v) { return !v.empty(); };
  const auto field = [](std::string_view v) { return v.find("{layer}") != std::string_view::npos; };
  if (!present(n.text.embedding) || !std::ranges::all_of(n.text.layer, field) ||
      !std::ranges::all_of(n.denoiser.globals, present) ||
      !std::ranges::all_of(n.denoiser.block, field) ||
      !std::ranges::all_of(
          std::array{n.vae.post_quant_conv, n.vae.conv_in, n.vae.mid_attention, n.vae.upsampler,
                     n.vae.norm_out, n.vae.conv_out, n.vae.shortcut, n.vae.norm1, n.vae.conv1,
                     n.vae.norm2, n.vae.conv2, n.vae.attention_norm, n.vae.attention_qkv,
                     n.vae.attention_proj, n.vae.weight, n.vae.bias},
          present) ||
      !field(n.vae.mid_resnet) || !field(n.vae.up_block) || !field(n.vae.up_resnet))
    return Refused("every tensor name, a layer's and a block's with {layer}");
  // Every tensor its names make distinct from every other (two layers, two
  // blocks, two up blocks of a small family: each pattern and suffix
  // instantiated), so that no role binds another's tensor.
  {
    const model::QwenImageTextProfile text{
        .layers = 2, .width = 2, .heads = 1, .kv_heads = 1, .head_dim = 2, .ffn = 2, .vocab = 2};
    model::QwenImageDenoiserProfile flow;
    flow.blocks = 2;
    flow.heads = 1;
    flow.head_dim = 2;
    flow.width = 2;
    flow.mlp = 2;
    flow.in_channels = flow.out_channels = 2;
    flow.context = 2;
    flow.timestep_dim = 2;
    model::QwenImageVaeProfile vae;
    vae.z_dim = 1;
    vae.base = 1;
    vae.dim_mult = {1, 2, 2};
    vae.res_blocks = 1;
    vae.temporal_upsample = {false, false};
    vae.out_channels = 1;
    vae.spatial_factor = 4;
    vae.latents_mean = {0.0F};
    vae.latents_std = {1.0F};
    for (const auto& list :
         {model::TextEncoderTensors(text, n.text), model::DenoiserTensors(flow, n.denoiser),
          model::VaeDecoderTensors(vae, n.vae)}) {
      std::vector<std::string_view> names;
      names.reserve(list.size());
      for (const auto& t : list) names.push_back(t.name);
      std::ranges::sort(names);
      if (std::ranges::adjacent_find(names) != names.end())
        return Refused(std::format("two tensors are named {}", *std::ranges::adjacent_find(names)));
    }
  }
  if (m.serving == nullptr || m.tokenizer.empty() || m.prompt.empty())
    return Refused("a serving recipe, a tokenizer and a prompt renderer");
  const PolicyRule* rule = m.serving->serving_rule;
  if (rule == nullptr || rule->text.empty() || rule->text.size() > kMaxPolicyText ||
      rule->text.back() != '\n')
    return Refused("a serving rule's planning policy of printable lines");
  return {};
}

// ---------------------------------------------------------------- synthesis

std::expected<ModelSpec, std::string> SynthesizeImageFamily(
    const ImageMapping& mapping, std::span<const ImageComponentSource> components,
    std::span<const ImageSource> files) {
  if (auto valid = ValidateImageMapping(mapping); !valid) return std::unexpected(valid.error());
  const auto values_of = [&](std::string_view role,
                             std::string_view file) -> std::optional<std::span<const SourceValue>> {
    if (!role.empty()) {
      const auto c = std::ranges::find(components, role, &ImageComponentSource::role);
      if (c == components.end()) return std::nullopt;
      return c->values;
    }
    const auto f = std::ranges::find(files, file, &ImageSource::name);
    if (f == files.end()) return std::nullopt;
    return f->values;
  };
  for (const auto& q : mapping.requirements) {
    const auto values = values_of(q.role, q.file);
    if (!values) return Refused(std::format("no {}{} for requirement {}", q.role, q.file, q.key));
    const ParamValue* v = ValueOf(*values, q.key);
    bool holds = false;
    if (v != nullptr) switch (q.type) {
        case ValueType::kInt: {
          const auto* i = std::get_if<std::int64_t>(v);
          holds = i && *i == q.value.integer;
          break;
        }
        case ValueType::kReal: {
          const auto* r = std::get_if<double>(v);
          const auto* i = std::get_if<std::int64_t>(v);
          holds = (r && *r == q.value.real) || (i && static_cast<double>(*i) == q.value.real);
          break;
        }
        case ValueType::kBool: {
          const auto* b = std::get_if<bool>(v);
          holds = b && *b == q.value.flag;
          break;
        }
        case ValueType::kString: {
          const auto* t = std::get_if<std::string>(v);
          holds = t && *t == q.value.text;
          break;
        }
        case ValueType::kInts:
          break;
      }
    if (!holds)
      return Refused(
          std::format("{}{} does not hold {} as its mapping requires", q.role, q.file, q.key));
  }
  ModelSpec spec{.pipeline = PipelineKind::kImage};
  for (std::size_t i = 0; i < mapping.components.size(); ++i) {
    const auto& rule = mapping.components[i];
    const auto source = std::ranges::find(components, rule.role, &ImageComponentSource::role);
    if (source == components.end()) return Refused(std::format("no {} component", rule.role));
    if (!std::ranges::contains(rule.architectures, source->architecture))
      return Refused(std::format("{} is a {}, not an architecture its mapping accepts", rule.role,
                                 source->architecture));
    Stage stage{.name = std::string(rule.stage),
                .role = rule.stage_role,
                .artifact = static_cast<std::uint32_t>(i)};
    Layer layer;
    for (const auto kind : rule.kinds) {
      ComponentRef ref{.kind = std::string(kind), .version = 1};
      for (const auto& s : kSchema) {
        if (s.kind != kind) continue;
        const auto r = std::ranges::find_if(
            mapping.params, [&](const auto& p) { return p.kind == kind && p.param == s.param; });
        std::optional<ParamValue> typed;
        if (r->key.empty()) {
          typed = Typed(
              s, s.type == Type::kInt ? ParamValue(r->value.integer) : ParamValue(r->value.real),
              false);
        } else {
          const auto values = values_of(r->role, r->file);
          if (!values) return Refused(std::format("{}.{}: no {}", kind, s.param, WhereOf(*r)));
          const ParamValue* v = ValueOf(*values, r->key);
          if (v == nullptr)
            return Refused(std::format("{}.{}: {} has no {}", kind, s.param, WhereOf(*r), r->key));
          typed = Typed(s, *v, r->reverse);
        }
        if (!typed)
          return Refused(std::format("{}.{}: {} of another type or out of range", kind, s.param,
                                     r->key.empty() ? std::string_view("its constant") : r->key));
        ref.params.push_back({.name = std::string(s.param), .value = std::move(*typed)});
      }
      layer.components.push_back(std::move(ref));
    }
    stage.layers.push_back(std::move(layer));
    spec.stages.push_back(std::move(stage));
  }
  Canonicalize(spec);
  // The stream widths the profile derives.
  auto profile = ProfileOf(spec, mapping, false);
  if (!profile) return std::unexpected(profile.error());
  for (auto& stage : spec.stages) stage.stream.width = WidthOf(stage, *profile);
  if (auto valid = Validate(spec); !valid) return std::unexpected(valid.error());
  return spec;
}

// ------------------------------------------------------------------ reading

namespace {
std::expected<model::QwenImageProfile, std::string> ProfileOf(const ModelSpec& spec,
                                                              const ImageMapping& mapping,
                                                              bool widths) {
  if (spec.pipeline != PipelineKind::kImage || spec.stages.size() != mapping.components.size())
    return Refused("the stored spec is not the mapping's pipeline");
  model::QwenImageProfile p;
  // A component's parameters by name, checked against the schema exactly.
  const auto read = [&](const ComponentRef& ref) -> std::expected<std::vector<Param>, std::string> {
    std::vector<Param> out;
    for (const auto& s : kSchema) {
      if (s.kind != ref.kind) continue;
      const Param* given = ref.Find(s.param);
      if (given == nullptr) return Refused(std::format("{} has no {}", ref.kind, s.param));
      auto typed = Typed(s, given->value, false);
      if (!typed || *typed != given->value)
        return Refused(std::format("{}.{} of another type or out of range", ref.kind, s.param));
      out.push_back(*given);
    }
    if (out.size() != ref.params.size() || !ref.weights.empty() || ref.version != 1)
      return Refused(std::format("{} has parameters or weights its schema does not", ref.kind));
    return out;
  };
  const auto integer = [](const ComponentRef& ref, std::string_view name) {
    return static_cast<std::uint32_t>(std::get<std::int64_t>(ref.Find(name)->value));
  };
  const auto real = [](const ComponentRef& ref, std::string_view name) {
    return std::get<double>(ref.Find(name)->value);
  };
  const auto ints = [](const ComponentRef& ref, std::string_view name) {
    return std::get<std::vector<std::int64_t>>(ref.Find(name)->value);
  };
  const auto reals = [](const ComponentRef& ref, std::string_view name) {
    return std::get<std::vector<double>>(ref.Find(name)->value);
  };
  bool text = false, flow = false, schedule = false, vae = false;
  for (std::size_t i = 0; i < spec.stages.size(); ++i) {
    const Stage& stage = spec.stages[i];
    const auto& rule = mapping.components[i];
    if (stage.name != rule.stage || stage.role != rule.stage_role || stage.artifact != i ||
        stage.layers.size() != 1 || stage.layers.front().components.size() != rule.kinds.size() ||
        stage.embedding || stage.stream.streams != 1 || !stage.side.empty() ||
        !stage.taps.empty() || !stage.reads.empty() || !stage.folds.empty() ||
        stage.heads != Heads{})
      return Refused(std::format("stage {} is not the mapping's {}", i, rule.stage));
    for (std::size_t k = 0; k < rule.kinds.size(); ++k) {
      const ComponentRef& ref = stage.layers.front().components[k];
      if (ref.kind != rule.kinds[k]) return Refused(std::format("stage {}: another kind", i));
      if (auto ok = read(ref); !ok) return std::unexpected(ok.error());
      if (ref.kind == kImageTextKind) {
        text = true;
        p.text = {.layers = integer(ref, "layers"),
                  .width = integer(ref, "width"),
                  .heads = integer(ref, "heads"),
                  .kv_heads = integer(ref, "kv_heads"),
                  .head_dim = integer(ref, "head_dim"),
                  .ffn = integer(ref, "ffn"),
                  .vocab = integer(ref, "vocab"),
                  .rms_eps = ToFloat(real(ref, "rms_eps")),
                  .rope_theta = ToFloat(real(ref, "rope_theta"))};
      } else if (ref.kind == kImageFlowKind) {
        flow = true;
        auto& d = p.denoiser;
        d.blocks = integer(ref, "blocks");
        d.heads = integer(ref, "heads");
        d.head_dim = integer(ref, "head_dim");
        const std::uint64_t width = std::uint64_t{d.heads} * d.head_dim;
        const std::uint64_t mlp = width * integer(ref, "mlp_ratio");
        constexpr std::uint64_t kMostWidth = std::uint64_t{1} << 24U;
        if (width > kMostWidth || mlp > kMostWidth)
          return Refused("the flow transformer's width overflows");
        d.width = static_cast<std::uint32_t>(width);
        d.mlp = static_cast<std::uint32_t>(mlp);
        d.in_channels = integer(ref, "in_channels");
        d.out_channels = integer(ref, "out_channels");
        d.context = integer(ref, "context");
        d.timestep_dim = integer(ref, "timestep_dim");
        const auto axes = ints(ref, "rope_axes");
        for (std::size_t a = 0; a < 3; ++a) d.rope_axes[a] = static_cast<std::uint32_t>(axes[a]);
        d.rope_theta = ToFloat(real(ref, "rope_theta"));
        d.eps = ToFloat(real(ref, "eps"));
        d.timestep_divisor = ToFloat(real(ref, "timestep_divisor"));
        d.timestep_scale = ToFloat(real(ref, "timestep_scale"));
        d.timestep_period = real(ref, "timestep_period");
      } else if (ref.kind == kImageScheduleKind) {
        schedule = true;
        p.scheduler = {.train_timesteps = integer(ref, "train_timesteps"),
                       .base_image_seq_len = integer(ref, "base_image_seq_len"),
                       .max_image_seq_len = integer(ref, "max_image_seq_len"),
                       .base_shift = real(ref, "base_shift"),
                       .max_shift = real(ref, "max_shift"),
                       .shift_terminal = real(ref, "shift_terminal")};
      } else if (ref.kind == kImageVaeKind) {
        vae = true;
        auto& v = p.vae;
        v.z_dim = integer(ref, "z_dim");
        v.base = integer(ref, "base");
        for (const auto m : ints(ref, "dim_mult"))
          v.dim_mult.push_back(static_cast<std::uint32_t>(m));
        v.res_blocks = integer(ref, "res_blocks");
        for (const auto t : ints(ref, "temporal_upsample")) v.temporal_upsample.push_back(t != 0);
        v.out_channels = integer(ref, "out_channels");
        v.spatial_factor = integer(ref, "spatial_factor");
        for (const auto m : reals(ref, "latents_mean")) v.latents_mean.push_back(ToFloat(m));
        for (const auto s : reals(ref, "latents_std")) v.latents_std.push_back(ToFloat(s));
        if (v.latents_mean.size() != v.z_dim || v.latents_std.size() != v.z_dim ||
            v.temporal_upsample.size() + 1 < v.dim_mult.size())
          return Refused("the VAE's latent statistics or temporal steps do not match its shape");
      }
    }
  }
  if (!text || !flow || !schedule || !vae) return Refused("a family kind is missing");
  for (const Stage& stage : spec.stages)
    if (widths && stage.stream.width != WidthOf(stage, p))
      return Refused(std::format("stage {}'s width is not its family's", stage.name));
  return p;
}
}  // namespace

std::expected<model::QwenImageProfile, std::string> ImageProfileOf(const ModelSpec& spec,
                                                                   const ImageMapping& mapping) {
  return ProfileOf(spec, mapping, true);
}

}  // namespace llmp::spec
