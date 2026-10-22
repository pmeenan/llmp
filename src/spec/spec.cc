// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "spec/spec.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <format>
#include <functional>
#include <initializer_list>
#include <limits>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include "artifact/json.h"
#include "base/sha256.h"
#include "format/encoding.h"
#include "spec/names.h"

namespace llmp::spec {
namespace {

namespace json = artifact::json;
using Status = std::expected<void, std::string>;

std::unexpected<std::string> Refuse(std::string_view where, std::string_view what) {
  return std::unexpected(std::format("{}: {}", where, what));
}

constexpr std::array kFolds = {
    FoldInfo{.name = "norm.plus_one",
             .meaning = "RMS norm weights stored as 1 + w (Gemma's offset; llama.cpp's GGUF "
                        "converter adds it)"},
    FoldInfo{.name = "ssm.a_neg_exp",
             .meaning = "a gated delta rule's A stored as -exp(A_log) (Qwen3.8; llama.cpp's "
                        "converter)"},
    FoldInfo{.name = "gdn.value_heads_tiled",
             .meaning = "a gated delta rule's per-value-head tensors tiled to the value heads "
                        "(Qwen3.8; llama.cpp's converter)"},
};

constexpr std::array<std::string_view, 5> kPipelines = {"llm", "image", "embedding", "speech",
                                                        "decision"};
constexpr std::array<std::string_view, 6> kRoles = {"target",       "drafter",  "encoder",
                                                    "text_encoder", "denoiser", "vae"};
constexpr std::array<std::string_view, 4> kTapPoints = {"layer_out", "streams", "post_final_norm",
                                                        "stream_mean"};
constexpr std::array<std::string_view, 2> kReadKinds = {"tap", "state"};
constexpr std::array kParts = {format::PartRole::kData,           format::PartRole::kCodes,
                               format::PartRole::kBlockScales,    format::PartRole::kTensorScale,
                               format::PartRole::kInputSigns,     format::PartRole::kOutputSigns,
                               format::PartRole::kCodebookMarker, format::PartRole::kBiases,
                               format::PartRole::kZeros};

template <typename E, std::size_t N>
std::optional<E> ByName(const std::array<std::string_view, N>& names, std::string_view name) {
  for (std::size_t i = 0; i < N; ++i) {
    if (names.at(i) == name) {
      return static_cast<E>(i);
    }
  }
  return std::nullopt;
}

template <std::size_t N>
std::string_view NameOf(const std::array<std::string_view, N>& names, std::size_t i) {
  return i < N ? names.at(i) : std::string_view("unknown");
}

std::optional<format::PartRole> PartByName(std::string_view name) {
  for (const format::PartRole p : kParts) {
    if (format::PartRoleName(p) == name) {
      return p;
    }
  }
  return std::nullopt;
}

using names::Digit;
using names::IsKind;
using names::IsName;
using names::Lower;

// A tensor name: printable ASCII without space, '"' or '\\'.
bool IsTensor(std::string_view s) {
  if (s.empty() || s.size() > kMaxText) {
    return false;
  }
  return std::ranges::all_of(s,
                             [](char c) { return c > ' ' && c < 0x7F && c != '"' && c != '\\'; });
}

bool Finite(double v) { return std::isfinite(v); }

// A real a spec may hold: finite and not -0, which compares equal to 0 but
// writes "-0", so one spec would have two encodings and two digests.
bool IsReal(double v) { return Finite(v) && !(v == 0.0 && std::signbit(v)); }

// std::to_chars' shortest round-trip form.
std::string RealText(double v) {
  std::array<char, 64> buf{};
  const auto r = std::to_chars(buf.data(), buf.data() + buf.size(), v);
  return {buf.data(), r.ptr};
}

// A real's text: it parses whole, IsReal holds and it is RealText's form.
std::optional<double> ParseReal(std::string_view s) {
  if (s.empty() || !(Digit(s.front()) || s.front() == '-')) {
    return std::nullopt;
  }
  double v = 0.0;
  const auto r = std::from_chars(s.data(), s.data() + s.size(), v);
  if (r.ec != std::errc() || r.ptr != s.data() + s.size() || !IsReal(v) || RealText(v) != s) {
    return std::nullopt;
  }
  return v;
}

// ---------------------------------------------------------------- writing

void Quoted(std::string& out, std::string_view s) {
  out += '"';
  out += s;
  out += '"';
}

void Key(std::string& out, std::string_view key, bool& first) {
  if (!first) {
    out += ',';
  }
  first = false;
  Quoted(out, key);
  out += ':';
}

void WriteValue(std::string& out, const ParamValue& v) {
  if (const auto* i = std::get_if<std::int64_t>(&v)) {
    out += std::to_string(*i);
  } else if (const auto* r = std::get_if<double>(&v)) {
    Quoted(out, RealText(*r));
  } else if (const auto* b = std::get_if<bool>(&v)) {
    out += *b ? "true" : "false";
  } else if (const auto* s = std::get_if<std::string>(&v)) {
    Quoted(out, *s);
  } else if (const auto* is = std::get_if<std::vector<std::int64_t>>(&v)) {
    out += '[';
    for (std::size_t k = 0; k < is->size(); ++k) {
      out += k > 0 ? "," : "";
      out += std::to_string((*is)[k]);
    }
    out += ']';
  } else if (const auto* rs = std::get_if<std::vector<double>>(&v)) {
    out += '[';
    for (std::size_t k = 0; k < rs->size(); ++k) {
      out += k > 0 ? "," : "";
      Quoted(out, RealText((*rs)[k]));
    }
    out += ']';
  }
}

void WriteComponent(std::string& out, const ComponentRef& c) {
  bool first = true;
  out += '{';
  Key(out, "kind", first);
  Quoted(out, c.kind);
  Key(out, "params", first);
  out += '{';
  bool p_first = true;
  for (const Param& p : c.params) {
    Key(out, p.name, p_first);
    WriteValue(out, p.value);
  }
  out += '}';
  Key(out, "version", first);
  out += std::to_string(c.version);
  Key(out, "weights", first);
  out += '[';
  for (std::size_t i = 0; i < c.weights.size(); ++i) {
    const WeightBinding& w = c.weights[i];
    out += i > 0 ? ",{" : "{";
    bool w_first = true;
    if (w.experts) {
      Key(out, "experts", w_first);
      out += "true";
    }
    Key(out, "part", w_first);
    Quoted(out, format::PartRoleName(w.part));
    Key(out, "role", w_first);
    Quoted(out, w.role);
    if (!w.stage.empty()) {
      Key(out, "stage", w_first);
      Quoted(out, w.stage);
    }
    Key(out, "tensor", w_first);
    Quoted(out, w.tensor);
    out += '}';
  }
  out += "]}";
}

void WriteStage(std::string& out, const Stage& s) {
  bool first = true;
  out += '{';
  Key(out, "artifact", first);
  out += std::to_string(s.artifact);
  Key(out, "context", first);
  out += std::to_string(s.context);
  if (s.embedding) {
    Key(out, "embedding", first);
    WriteComponent(out, *s.embedding);
  }
  if (!s.exports.empty()) {
    Key(out, "exports", first);
    out += '[';
    for (std::size_t i = 0; i < s.exports.size(); ++i) {
      out += i > 0 ? "," : "";
      out += std::to_string(s.exports[i]);
    }
    out += ']';
  }
  Key(out, "folds", first);
  out += '[';
  for (std::size_t i = 0; i < s.folds.size(); ++i) {
    out += i > 0 ? "," : "";
    Quoted(out, s.folds[i]);
  }
  out += ']';
  Key(out, "heads", first);
  {
    out += '{';
    bool h_first = true;
    if (s.heads.final_norm) {
      Key(out, "final_norm", h_first);
      WriteComponent(out, *s.heads.final_norm);
    }
    Key(out, "mtp", h_first);
    out += '[';
    for (std::size_t i = 0; i < s.heads.mtp.size(); ++i) {
      out += i > 0 ? "," : "";
      WriteComponent(out, s.heads.mtp[i]);
    }
    out += ']';
    if (s.heads.output) {
      Key(out, "output", h_first);
      WriteComponent(out, *s.heads.output);
    }
    if (s.heads.softcap) {
      Key(out, "softcap", h_first);
      Quoted(out, RealText(*s.heads.softcap));
    }
    out += '}';
  }
  Key(out, "layers", first);
  out += '[';
  for (std::size_t l = 0; l < s.layers.size(); ++l) {
    out += l > 0 ? ",[" : "[";
    for (std::size_t i = 0; i < s.layers[l].components.size(); ++i) {
      out += i > 0 ? "," : "";
      WriteComponent(out, s.layers[l].components[i]);
    }
    out += ']';
  }
  out += ']';
  if (!s.links.empty()) {
    Key(out, "links", first);
    out += '[';
    for (std::size_t i = 0; i < s.links.size(); ++i) {
      const GraphLink& link = s.links[i];
      out += i > 0 ? ",{" : "{";
      bool l_first = true;
      if (link.auxiliary) {
        Key(out, "auxiliary", l_first);
        out += std::to_string(*link.auxiliary);
      }
      if (link.auxiliary_port) {
        Key(out, "auxiliary_port", l_first);
        out += std::to_string(*link.auxiliary_port);
      }
      if (link.carry) {
        Key(out, "carry", l_first);
        out += std::to_string(*link.carry);
      }
      if (link.carry_port) {
        Key(out, "carry_port", l_first);
        out += std::to_string(*link.carry_port);
      }
      Key(out, "input", l_first);
      out += std::to_string(link.input);
      if (link.input_port) {
        Key(out, "input_port", l_first);
        out += std::to_string(*link.input_port);
      }
      out += '}';
    }
    out += ']';
  }
  Key(out, "name", first);
  Quoted(out, s.name);
  Key(out, "reads", first);
  out += '[';
  for (std::size_t i = 0; i < s.reads.size(); ++i) {
    const Read& r = s.reads[i];
    out += i > 0 ? ",{" : "{";
    bool r_first = true;
    Key(out, "kind", r_first);
    Quoted(out, ReadKindName(r.kind));
    Key(out, "name", r_first);
    Quoted(out, r.name);
    Key(out, "stage", r_first);
    Quoted(out, r.stage);
    out += '}';
  }
  out += ']';
  Key(out, "role", first);
  Quoted(out, StageRoleName(s.role));
  if (!s.sampling.empty()) {
    Key(out, "sampling", first);
    out += '{';
    bool smp_first = true;
    if (s.sampling.min_p) {
      Key(out, "min_p", smp_first);
      Quoted(out, RealText(*s.sampling.min_p));
    }
    if (s.sampling.temperature) {
      Key(out, "temperature", smp_first);
      Quoted(out, RealText(*s.sampling.temperature));
    }
    if (s.sampling.top_k) {
      Key(out, "top_k", smp_first);
      out += std::to_string(*s.sampling.top_k);
    }
    if (s.sampling.top_p) {
      Key(out, "top_p", smp_first);
      Quoted(out, RealText(*s.sampling.top_p));
    }
    out += '}';
  }
  Key(out, "side", first);
  out += '[';
  for (std::size_t i = 0; i < s.side.size(); ++i) {
    const SideInput& side = s.side[i];
    out += i > 0 ? ",{" : "{";
    bool s_first = true;
    Key(out, "layers", s_first);
    WriteValue(out, std::vector<std::int64_t>(side.layers.begin(), side.layers.end()));
    Key(out, "name", s_first);
    Quoted(out, side.name);
    Key(out, "source", s_first);
    WriteComponent(out, side.source);
    out += '}';
  }
  out += ']';
  Key(out, "stream", first);
  {
    out += '{';
    bool st_first = true;
    if (s.stream.mixing) {
      Key(out, "mixing", st_first);
      WriteComponent(out, *s.stream.mixing);
    }
    Key(out, "streams", st_first);
    out += std::to_string(s.stream.streams);
    Key(out, "width", st_first);
    out += std::to_string(s.stream.width);
    out += '}';
  }
  Key(out, "taps", first);
  out += '[';
  for (std::size_t i = 0; i < s.taps.size(); ++i) {
    const FeatureTap& t = s.taps[i];
    out += i > 0 ? ",{" : "{";
    bool t_first = true;
    Key(out, "layer", t_first);
    out += std::to_string(t.layer);
    Key(out, "name", t_first);
    Quoted(out, t.name);
    Key(out, "point", t_first);
    Quoted(out, TapPointName(t.point));
    out += '}';
  }
  out += "]}";
}

// ---------------------------------------------------------------- reading

// An object whose keys are `required` plus any of `optional`.
Status Keys(json::Value v, std::string_view where, std::initializer_list<std::string_view> required,
            std::initializer_list<std::string_view> optional = {}) {
  if (!v.is_object()) {
    return Refuse(where, "not an object");
  }
  for (std::size_t i = 0; i < v.size(); ++i) {
    const std::string_view k = v.key(i);
    if (std::ranges::find(required, k) == required.end() &&
        std::ranges::find(optional, k) == optional.end()) {
      return Refuse(where, std::format("an unknown key \"{}\"", k));
    }
  }
  for (const std::string_view k : required) {
    if (!v.find(k)) {
      return Refuse(where, std::format("no \"{}\"", k));
    }
  }
  return {};
}

json::Value Member(json::Value v, std::string_view key) { return *v.find(key); }

std::expected<std::uint64_t, std::string> Unsigned(json::Value v, std::string_view where,
                                                   std::uint64_t most) {
  const std::optional<std::int64_t> i = v.integer();
  if (!v.is_integer() || !i || *i < 0 || static_cast<std::uint64_t>(*i) > most) {
    return Refuse(where, std::format("not an integer from 0 to {}", most));
  }
  return static_cast<std::uint64_t>(*i);
}

std::expected<std::string, std::string> Text(json::Value v, std::string_view where,
                                             bool (*valid)(std::string_view)) {
  if (!v.is_string() || !valid(v.string())) {
    return Refuse(where, "not a valid name");
  }
  return std::string(v.string());
}

std::expected<json::Value, std::string> Array(json::Value v, std::string_view where,
                                              std::size_t most) {
  if (!v.is_array() || v.size() > most) {
    return Refuse(where, std::format("not an array of at most {}", most));
  }
  return v;
}

std::expected<ParamValue, std::string> ReadValue(json::Value v, const std::string& where) {
  switch (v.kind()) {
    case json::Kind::kInteger: {
      const std::optional<std::int64_t> i = v.integer();
      if (!i) {
        return Refuse(where, "an integer outside 64 bits");
      }
      return ParamValue(*i);
    }
    case json::Kind::kTrue:
      return ParamValue(true);
    case json::Kind::kFalse:
      return ParamValue(false);
    case json::Kind::kString: {
      const std::string_view s = v.string();
      if (!s.empty() && (Digit(s.front()) || s.front() == '-')) {
        const std::optional<double> r = ParseReal(s);
        if (!r) {
          return Refuse(where, "not a finite real (nor -0) in its shortest round-trip form");
        }
        return ParamValue(*r);
      }
      if (!IsName(s)) {
        return Refuse(where, "a string that is neither a real nor a name");
      }
      return ParamValue(std::string(s));
    }
    case json::Kind::kArray: {
      if (v.size() > kMaxList) {
        return Refuse(where, std::format("a list past {} values", kMaxList));
      }
      if (v.size() == 0 || v.at(0).is_integer()) {
        std::vector<std::int64_t> out;
        for (std::size_t i = 0; i < v.size(); ++i) {
          const std::optional<std::int64_t> x = v.at(i).integer();
          if (!v.at(i).is_integer() || !x) {
            return Refuse(where, "a list of integers holds something else");
          }
          out.push_back(*x);
        }
        return ParamValue(std::move(out));
      }
      std::vector<double> out;
      for (std::size_t i = 0; i < v.size(); ++i) {
        const std::optional<double> x =
            v.at(i).is_string() ? ParseReal(v.at(i).string()) : std::nullopt;
        if (!x) {
          return Refuse(where, "a list of reals holds something else");
        }
        out.push_back(*x);
      }
      return ParamValue(std::move(out));
    }
    default:
      return Refuse(where, "not a parameter value");
  }
}

#define LLMP_SPEC_TRY(var, expr)                    \
  auto var = (expr);                                \
  if (!var) {                                       \
    return std::unexpected(std::move(var).error()); \
  }

std::expected<ComponentRef, std::string> ReadComponent(json::Value v, const std::string& where) {
  if (auto k = Keys(v, where, {"kind", "params", "version", "weights"}); !k) {
    return std::unexpected(k.error());
  }
  ComponentRef c;
  LLMP_SPEC_TRY(kind, Text(Member(v, "kind"), where + ".kind", &IsKind));
  c.kind = std::move(*kind);
  LLMP_SPEC_TRY(version, Unsigned(Member(v, "version"), where + ".version", 65535));
  c.version = static_cast<std::uint16_t>(*version);
  const json::Value params = Member(v, "params");
  if (!params.is_object() || params.size() > kMaxParams) {
    return Refuse(where + ".params", std::format("not an object of at most {}", kMaxParams));
  }
  for (std::size_t i = 0; i < params.size(); ++i) {
    const std::string name(params.key(i));
    const std::string at = std::format("{}.params.{}", where, name);
    if (!IsName(name)) {
      return Refuse(at, "not a valid name");
    }
    LLMP_SPEC_TRY(value, ReadValue(params.member(i), at));
    c.params.push_back({.name = name, .value = std::move(*value)});
  }
  LLMP_SPEC_TRY(weights, Array(Member(v, "weights"), where + ".weights", kMaxWeights));
  for (std::size_t i = 0; i < weights->size(); ++i) {
    const json::Value w = weights->at(i);
    const std::string at = std::format("{}.weights[{}]", where, i);
    if (auto k = Keys(w, at, {"part", "role", "tensor"}, {"experts", "stage"}); !k) {
      return std::unexpected(k.error());
    }
    WeightBinding b;
    LLMP_SPEC_TRY(role, Text(Member(w, "role"), at + ".role", &IsName));
    b.role = std::move(*role);
    LLMP_SPEC_TRY(tensor, Text(Member(w, "tensor"), at + ".tensor", &IsTensor));
    b.tensor = std::move(*tensor);
    const json::Value part = Member(w, "part");
    const std::optional<format::PartRole> p =
        part.is_string() ? PartByName(part.string()) : std::nullopt;
    if (!p) {
      return Refuse(at + ".part", "not a part role");
    }
    b.part = *p;
    if (const auto e = w.find("experts"); e) {
      // Written only when true.
      if (e->kind() != json::Kind::kTrue) {
        return Refuse(at + ".experts", "present only as true");
      }
      b.experts = true;
    }
    if (const auto st = w.find("stage"); st) {
      // Written only when not empty.
      LLMP_SPEC_TRY(stage, Text(*st, at + ".stage", &IsName));
      b.stage = std::move(*stage);
    }
    c.weights.push_back(std::move(b));
  }
  return c;
}

std::expected<std::optional<ComponentRef>, std::string> MaybeComponent(json::Value v,
                                                                       std::string_view key,
                                                                       const std::string& where) {
  const auto found = v.find(key);
  if (!found) {
    return std::optional<ComponentRef>();
  }
  LLMP_SPEC_TRY(c, ReadComponent(*found, std::format("{}.{}", where, key)));
  return std::optional<ComponentRef>(std::move(*c));
}

std::expected<Stage, std::string> ReadStage(json::Value v, const std::string& where) {
  if (auto k = Keys(v, where,
                    {"artifact", "context", "folds", "heads", "layers", "name", "reads", "role",
                     "side", "stream", "taps"},
                    {"embedding", "exports", "links", "sampling"});
      !k) {
    return std::unexpected(k.error());
  }
  Stage s;
  LLMP_SPEC_TRY(name, Text(Member(v, "name"), where + ".name", &IsName));
  s.name = std::move(*name);
  const json::Value role = Member(v, "role");
  const auto stage_role =
      role.is_string() ? ByName<StageRole>(kRoles, role.string()) : std::nullopt;
  if (!stage_role) {
    return Refuse(where + ".role", "not a stage role");
  }
  s.role = *stage_role;
  LLMP_SPEC_TRY(artifact, Unsigned(Member(v, "artifact"), where + ".artifact", kMaxArtifacts - 1));
  s.artifact = static_cast<std::uint32_t>(*artifact);
  LLMP_SPEC_TRY(context, Unsigned(Member(v, "context"), where + ".context", kMaxContext));
  s.context = static_cast<std::uint32_t>(*context);
  if (const auto smp = v.find("sampling"); smp) {
    const std::string sm = where + ".sampling";
    if (auto k = Keys(*smp, sm, {}, {"min_p", "temperature", "top_k", "top_p"}); !k) {
      return std::unexpected(k.error());
    }
    if (smp->size() == 0) {
      return Refuse(sm, "present only when non-empty");
    }
    const auto read_real =
        [&](std::string_view key) -> std::expected<std::optional<double>, std::string> {
      const auto f = smp->find(key);
      if (!f) return std::nullopt;
      const std::optional<double> r = f->is_string() ? ParseReal(f->string()) : std::nullopt;
      if (!r) {
        return Refuse(std::format("{}.{}", sm, key),
                      "not a finite real in its shortest round-trip form");
      }
      return *r;
    };
    LLMP_SPEC_TRY(min_p, read_real("min_p"));
    s.sampling.min_p = *min_p;
    LLMP_SPEC_TRY(temperature, read_real("temperature"));
    s.sampling.temperature = *temperature;
    if (const auto tk = smp->find("top_k"); tk) {
      LLMP_SPEC_TRY(top_k, Unsigned(*tk, sm + ".top_k", 2147483647));
      s.sampling.top_k = static_cast<std::int64_t>(*top_k);
    }
    LLMP_SPEC_TRY(top_p, read_real("top_p"));
    s.sampling.top_p = *top_p;
  }
  LLMP_SPEC_TRY(embedding, MaybeComponent(v, "embedding", where));
  s.embedding = std::move(*embedding);

  const json::Value stream = Member(v, "stream");
  const std::string st = where + ".stream";
  if (auto k = Keys(stream, st, {"streams", "width"}, {"mixing"}); !k) {
    return std::unexpected(k.error());
  }
  LLMP_SPEC_TRY(streams, Unsigned(Member(stream, "streams"), st + ".streams", kMaxStreams));
  s.stream.streams = static_cast<std::uint32_t>(*streams);
  LLMP_SPEC_TRY(width, Unsigned(Member(stream, "width"), st + ".width", kMaxWidth));
  s.stream.width = *width;
  LLMP_SPEC_TRY(mixing, MaybeComponent(stream, "mixing", st));
  s.stream.mixing = std::move(*mixing);

  LLMP_SPEC_TRY(side, Array(Member(v, "side"), where + ".side", kMaxNamed));
  for (std::size_t i = 0; i < side->size(); ++i) {
    const json::Value e = side->at(i);
    const std::string at = std::format("{}.side[{}]", where, i);
    if (auto k = Keys(e, at, {"layers", "name", "source"}); !k) {
      return std::unexpected(k.error());
    }
    SideInput in;
    LLMP_SPEC_TRY(n, Text(Member(e, "name"), at + ".name", &IsName));
    in.name = std::move(*n);
    LLMP_SPEC_TRY(src, ReadComponent(Member(e, "source"), at + ".source"));
    in.source = std::move(*src);
    LLMP_SPEC_TRY(layers, Array(Member(e, "layers"), at + ".layers", kMaxLayers));
    for (std::size_t l = 0; l < layers->size(); ++l) {
      LLMP_SPEC_TRY(layer,
                    Unsigned(layers->at(l), std::format("{}.layers[{}]", at, l), kMaxLayers - 1));
      in.layers.push_back(static_cast<std::uint32_t>(*layer));
    }
    s.side.push_back(std::move(in));
  }

  LLMP_SPEC_TRY(layers, Array(Member(v, "layers"), where + ".layers", kMaxLayers));
  for (std::size_t l = 0; l < layers->size(); ++l) {
    const std::string at = std::format("{}.layers[{}]", where, l);
    LLMP_SPEC_TRY(list, Array(layers->at(l), at, kMaxComponents));
    Layer layer;
    for (std::size_t i = 0; i < list->size(); ++i) {
      LLMP_SPEC_TRY(c, ReadComponent(list->at(i), std::format("{}[{}]", at, i)));
      layer.components.push_back(std::move(*c));
    }
    s.layers.push_back(std::move(layer));
  }

  if (const auto lk = v.find("links"); lk) {
    LLMP_SPEC_TRY(links, Array(*lk, where + ".links", kMaxComponents));
    if (links->size() == 0) {
      return Refuse(where + ".links", "present only when non-empty");
    }
    for (std::size_t i = 0; i < links->size(); ++i) {
      const json::Value e = links->at(i);
      const std::string at = std::format("{}.links[{}]", where, i);
      if (auto k = Keys(e, at, {"input"},
                        {"auxiliary", "auxiliary_port", "carry", "carry_port", "input_port"});
          !k) {
        return std::unexpected(k.error());
      }
      GraphLink link;
      LLMP_SPEC_TRY(in_id, Unsigned(Member(e, "input"), at + ".input", kMaxComponents));
      link.input = static_cast<std::uint16_t>(*in_id);
      if (const auto aux = e.find("auxiliary"); aux) {
        LLMP_SPEC_TRY(id, Unsigned(*aux, at + ".auxiliary", kMaxComponents));
        link.auxiliary = static_cast<std::uint16_t>(*id);
      }
      if (const auto aux_p = e.find("auxiliary_port"); aux_p) {
        LLMP_SPEC_TRY(id, Unsigned(*aux_p, at + ".auxiliary_port", kMaxNamed - 1));
        link.auxiliary_port = static_cast<std::uint16_t>(*id);
      }
      if (const auto carry = e.find("carry"); carry) {
        LLMP_SPEC_TRY(id, Unsigned(*carry, at + ".carry", kMaxComponents));
        link.carry = static_cast<std::uint16_t>(*id);
      }
      if (const auto carry_p = e.find("carry_port"); carry_p) {
        LLMP_SPEC_TRY(id, Unsigned(*carry_p, at + ".carry_port", kMaxNamed - 1));
        link.carry_port = static_cast<std::uint16_t>(*id);
      }
      if (const auto in_p = e.find("input_port"); in_p) {
        LLMP_SPEC_TRY(id, Unsigned(*in_p, at + ".input_port", kMaxNamed - 1));
        link.input_port = static_cast<std::uint16_t>(*id);
      }
      s.links.push_back(link);
    }
  }

  if (const auto ex = v.find("exports"); ex) {
    LLMP_SPEC_TRY(exports, Array(*ex, where + ".exports", kMaxNamed));
    if (exports->size() == 0) {
      return Refuse(where + ".exports", "present only when non-empty");
    }
    for (std::size_t i = 0; i < exports->size(); ++i) {
      LLMP_SPEC_TRY(
          id, Unsigned(exports->at(i), std::format("{}.exports[{}]", where, i), kMaxComponents));
      s.exports.push_back(static_cast<std::uint16_t>(*id));
    }
  }

  LLMP_SPEC_TRY(taps, Array(Member(v, "taps"), where + ".taps", kMaxNamed));
  for (std::size_t i = 0; i < taps->size(); ++i) {
    const json::Value e = taps->at(i);
    const std::string at = std::format("{}.taps[{}]", where, i);
    if (auto k = Keys(e, at, {"layer", "name", "point"}); !k) {
      return std::unexpected(k.error());
    }
    FeatureTap t;
    LLMP_SPEC_TRY(n, Text(Member(e, "name"), at + ".name", &IsName));
    t.name = std::move(*n);
    LLMP_SPEC_TRY(layer, Unsigned(Member(e, "layer"), at + ".layer", kMaxLayers - 1));
    t.layer = static_cast<std::uint32_t>(*layer);
    const json::Value point = Member(e, "point");
    const auto p = point.is_string() ? ByName<TapPoint>(kTapPoints, point.string()) : std::nullopt;
    if (!p) {
      return Refuse(at + ".point", "not a tap point");
    }
    t.point = *p;
    s.taps.push_back(std::move(t));
  }

  const json::Value heads = Member(v, "heads");
  const std::string hd = where + ".heads";
  if (auto k = Keys(heads, hd, {"mtp"}, {"final_norm", "output", "softcap"}); !k) {
    return std::unexpected(k.error());
  }
  LLMP_SPEC_TRY(final_norm, MaybeComponent(heads, "final_norm", hd));
  s.heads.final_norm = std::move(*final_norm);
  LLMP_SPEC_TRY(output, MaybeComponent(heads, "output", hd));
  s.heads.output = std::move(*output);
  if (const auto cap = heads.find("softcap"); cap) {
    const std::optional<double> r = cap->is_string() ? ParseReal(cap->string()) : std::nullopt;
    if (!r) {
      return Refuse(hd + ".softcap", "not a finite real in its shortest round-trip form");
    }
    s.heads.softcap = *r;
  }
  LLMP_SPEC_TRY(mtp, Array(Member(heads, "mtp"), hd + ".mtp", kMaxNamed));
  for (std::size_t i = 0; i < mtp->size(); ++i) {
    LLMP_SPEC_TRY(c, ReadComponent(mtp->at(i), std::format("{}.mtp[{}]", hd, i)));
    s.heads.mtp.push_back(std::move(*c));
  }

  LLMP_SPEC_TRY(reads, Array(Member(v, "reads"), where + ".reads", kMaxNamed));
  for (std::size_t i = 0; i < reads->size(); ++i) {
    const json::Value e = reads->at(i);
    const std::string at = std::format("{}.reads[{}]", where, i);
    if (auto k = Keys(e, at, {"kind", "name", "stage"}); !k) {
      return std::unexpected(k.error());
    }
    Read r;
    LLMP_SPEC_TRY(st_name, Text(Member(e, "stage"), at + ".stage", &IsName));
    r.stage = std::move(*st_name);
    LLMP_SPEC_TRY(n, Text(Member(e, "name"), at + ".name", &IsName));
    r.name = std::move(*n);
    const json::Value kind = Member(e, "kind");
    const auto k = kind.is_string() ? ByName<ReadKind>(kReadKinds, kind.string()) : std::nullopt;
    if (!k) {
      return Refuse(at + ".kind", "not a read kind");
    }
    r.kind = *k;
    s.reads.push_back(std::move(r));
  }

  LLMP_SPEC_TRY(folds, Array(Member(v, "folds"), where + ".folds", kMaxNamed));
  for (std::size_t i = 0; i < folds->size(); ++i) {
    const json::Value f = folds->at(i);
    if (!f.is_string() || FindFold(f.string()) == nullptr) {
      return Refuse(std::format("{}.folds[{}]", where, i), "not in the fold catalog");
    }
    s.folds.emplace_back(f.string());
  }
  return s;
}

#undef LLMP_SPEC_TRY

// ---------------------------------------------------------------- validation

template <typename T, typename Key>
bool StrictlySorted(const std::vector<T>& v, Key key) {
  for (std::size_t i = 1; i < v.size(); ++i) {
    if (!(key(v[i - 1]) < key(v[i]))) {
      return false;
    }
  }
  return true;
}

Status CheckValue(const ParamValue& v, const std::string& where) {
  if (const auto* r = std::get_if<double>(&v); r != nullptr && !IsReal(*r)) {
    return Refuse(where, "a real that is not finite, or -0");
  }
  if (const auto* s = std::get_if<std::string>(&v); s != nullptr && !IsName(*s)) {
    return Refuse(where, "a string that is not a name");
  }
  if (const auto* is = std::get_if<std::vector<std::int64_t>>(&v);
      is != nullptr && is->size() > kMaxList) {
    return Refuse(where, std::format("a list past {} values", kMaxList));
  }
  if (const auto* rs = std::get_if<std::vector<double>>(&v); rs != nullptr) {
    if (rs->empty()) {
      return Refuse(where, "an empty list of reals (an empty list is integers)");
    }
    if (rs->size() > kMaxList || !std::ranges::all_of(*rs, IsReal)) {
      return Refuse(where,
                    std::format("a list of reals past {} values, not finite or -0", kMaxList));
    }
  }
  return {};
}

Status CheckComponent(const ComponentRef& c, const std::string& where) {
  if (!IsKind(c.kind)) {
    return Refuse(where, "a kind that is not dotted names");
  }
  if (c.version == 0) {
    return Refuse(where, "version 0");
  }
  if (c.params.size() > kMaxParams || c.weights.size() > kMaxWeights) {
    return Refuse(where, "too many parameters or weights");
  }
  if (!StrictlySorted(c.params, [](const Param& p) -> std::string_view { return p.name; })) {
    return Refuse(where, "parameters not sorted by name, or one named twice");
  }
  for (const Param& p : c.params) {
    const std::string at = std::format("{}.params.{}", where, p.name);
    if (!IsName(p.name)) {
      return Refuse(at, "not a valid name");
    }
    if (auto ok = CheckValue(p.value, at); !ok) {
      return ok;
    }
  }
  if (!StrictlySorted(c.weights, [](const WeightBinding& w) {
        return std::pair<std::string_view, format::PartRole>(w.role, w.part);
      })) {
    return Refuse(where, "weights not sorted by role and part, or one bound twice");
  }
  for (std::size_t i = 0; i < c.weights.size(); ++i) {
    const WeightBinding& w = c.weights[i];
    if (!IsName(w.role) || !IsTensor(w.tensor) || !std::ranges::contains(kParts, w.part) ||
        (!w.stage.empty() && !IsName(w.stage))) {
      return Refuse(std::format("{}.weights[{}]", where, i),
                    "an invalid role, tensor, part or stage");
    }
  }
  return {};
}

Status CheckStage(const ModelSpec& spec, const Stage& s, const std::string& where) {
  if (!IsName(s.name)) {
    return Refuse(where, "a name that is not valid");
  }
  if (static_cast<std::size_t>(s.role) >= kRoles.size()) {
    return Refuse(where, "an unknown role");
  }
  if (s.artifact >= kMaxArtifacts || s.context > kMaxContext) {
    return Refuse(where, "an artifact index or context out of range");
  }
  if (s.sampling.temperature &&
      (!IsReal(*s.sampling.temperature) || *s.sampling.temperature < 0.0 ||
       *s.sampling.temperature > 100.0)) {
    return Refuse(where + ".sampling.temperature", "not a finite real in [0, 100]");
  }
  if (s.sampling.top_p &&
      (!IsReal(*s.sampling.top_p) || *s.sampling.top_p <= 0.0 || *s.sampling.top_p > 1.0)) {
    return Refuse(where + ".sampling.top_p", "not a finite real in (0, 1]");
  }
  if (s.sampling.top_k && (*s.sampling.top_k < 0 || *s.sampling.top_k > 2147483647)) {
    return Refuse(where + ".sampling.top_k", "not an integer in [0, INT32_MAX]");
  }
  if (s.sampling.min_p &&
      (!IsReal(*s.sampling.min_p) || *s.sampling.min_p < 0.0 || *s.sampling.min_p > 1.0)) {
    return Refuse(where + ".sampling.min_p", "not a finite real in [0, 1]");
  }
  if (s.layers.size() > kMaxLayers) {
    return Refuse(where, std::format("more than {} layers", kMaxLayers));
  }
  std::size_t components = 0;
  const auto count = [&](const ComponentRef& c, const std::string& at) -> Status {
    ++components;
    if (auto ok = CheckComponent(c, at); !ok) {
      return ok;
    }
    // A weight of another stage's artifact names another stage of the spec.
    for (const WeightBinding& w : c.weights) {
      if (!w.stage.empty() &&
          (w.stage == s.name || !std::ranges::contains(spec.stages, w.stage, &Stage::name))) {
        return Refuse(at, std::format("weight {} of stage {}, which is not another of the spec's",
                                      w.tensor, w.stage));
      }
    }
    return {};
  };
  Status component_status;
  EachComponent(s, [&](const ComponentRef& c, const std::string& path) {
    if (component_status) component_status = count(c, where + "." + path);
  });
  if (!component_status) return component_status;
  if (s.stream.streams == 0 || s.stream.streams > kMaxStreams || s.stream.width > kMaxWidth) {
    return Refuse(where + ".stream", "streams or width out of range");
  }
  if (s.stream.mixing.has_value() != (s.stream.streams > 1)) {
    return Refuse(where + ".stream", "a mixing component exactly when there are several streams");
  }
  if (s.side.size() > kMaxNamed || s.taps.size() > kMaxNamed || s.reads.size() > kMaxNamed ||
      s.heads.mtp.size() > kMaxNamed) {
    return Refuse(where,
                  std::format("more than {} side inputs, taps, reads or MTP heads", kMaxNamed));
  }
  if (!StrictlySorted(s.side, [](const SideInput& in) -> std::string_view { return in.name; })) {
    return Refuse(where + ".side", "not sorted by name, or a name twice");
  }
  for (std::size_t i = 0; i < s.side.size(); ++i) {
    const SideInput& in = s.side[i];
    const std::string at = std::format("{}.side[{}]", where, i);
    if (!IsName(in.name)) {
      return Refuse(at, "a name that is not valid");
    }
    if (!StrictlySorted(in.layers, [](std::uint32_t l) { return l; }) ||
        (!in.layers.empty() && in.layers.back() >= s.layers.size())) {
      return Refuse(at, "layers not ascending or past the stage's");
    }
  }
  if (s.heads.softcap && !(Finite(*s.heads.softcap) && *s.heads.softcap > 0.0)) {
    return Refuse(where + ".heads.softcap", "not a finite value above zero");
  }
  if (components > kMaxComponents) {
    return Refuse(where, std::format("more than {} components", kMaxComponents));
  }
  std::size_t layer_components = 0;
  for (const Layer& l : s.layers) {
    layer_components += l.components.size();
  }
  if (!s.links.empty()) {
    if (s.links.size() != layer_components) {
      return Refuse(where + ".links", "must match the stage's layer components");
    }
    for (std::size_t i = 0; i < s.links.size(); ++i) {
      const GraphLink& link = s.links[i];
      const std::string at = std::format("{}.links[{}]", where, i);
      if (link.input > i || (link.auxiliary && *link.auxiliary > i) ||
          (link.carry && *link.carry > i)) {
        return Refuse(at, "a link must name an earlier component or stage input");
      }
      if (link.input_port && (link.input != 0 || *link.input_port >= kMaxNamed)) {
        return Refuse(at, "input_port requires input 0 and a bounded port index");
      }
      if (link.auxiliary_port &&
          (link.auxiliary.has_value() || *link.auxiliary_port >= kMaxNamed)) {
        return Refuse(at, "auxiliary_port cannot combine with auxiliary or exceed port bounds");
      }
      if (link.carry_port && (link.carry.has_value() || *link.carry_port >= kMaxNamed)) {
        return Refuse(at, "carry_port cannot combine with carry or exceed port bounds");
      }
    }
  }
  if (!s.exports.empty()) {
    if (s.exports.size() > kMaxNamed) {
      return Refuse(where + ".exports", std::format("more than {} exports", kMaxNamed));
    }
    for (std::size_t i = 0; i < s.exports.size(); ++i) {
      const std::uint16_t id = s.exports[i];
      const std::string at = std::format("{}.exports[{}]", where, i);
      if (id == 0 || id > layer_components) {
        return Refuse(at, "an export out of range");
      }
      if (std::ranges::count(s.exports, id) != 1) {
        return Refuse(at, "an export named twice");
      }
    }
  }
  if (!StrictlySorted(s.taps, [](const FeatureTap& t) -> std::string_view { return t.name; })) {
    return Refuse(where + ".taps", "not sorted by name, or a name twice");
  }
  for (std::size_t i = 0; i < s.taps.size(); ++i) {
    const FeatureTap& t = s.taps[i];
    const std::string at = std::format("{}.taps[{}]", where, i);
    if (!IsName(t.name) || static_cast<std::size_t>(t.point) >= kTapPoints.size()) {
      return Refuse(at, "an invalid name or point");
    }
    if (t.point == TapPoint::kPostFinalNorm ? (t.layer != 0 || !s.heads.final_norm)
                                            : t.layer >= s.layers.size()) {
      return Refuse(at, "a layer past the stage's, or a final norm's tap without one");
    }
    if ((t.point == TapPoint::kStreams || t.point == TapPoint::kStreamMean) &&
        s.stream.streams < 2) {
      return Refuse(at, "the streams of a stage with one");
    }
  }
  if (!StrictlySorted(s.reads, [](const Read& r) {
        return std::tuple<std::string_view, ReadKind, std::string_view>(r.stage, r.kind, r.name);
      })) {
    return Refuse(where + ".reads", "not sorted by stage, kind and name, or one twice");
  }
  for (std::size_t i = 0; i < s.reads.size(); ++i) {
    const Read& r = s.reads[i];
    const std::string at = std::format("{}.reads[{}]", where, i);
    if (!IsName(r.name) || static_cast<std::size_t>(r.kind) >= kReadKinds.size()) {
      return Refuse(at, "an invalid name or kind");
    }
    const auto other = std::ranges::find(spec.stages, r.stage, &Stage::name);
    if (other == spec.stages.end() ||
        other >= std::ranges::find(spec.stages, s.name, &Stage::name)) {
      return Refuse(at, "a read must name an earlier stage (stage-read cycles are forbidden)");
    }
    // State is named by its component's contract, which the catalog checks.
    if (r.kind == ReadKind::kTap &&
        !std::ranges::contains(other->taps, r.name, &FeatureTap::name)) {
      return Refuse(at, "a tap the stage does not have");
    }
  }
  if (s.folds.size() > kMaxNamed || !StrictlySorted(s.folds, std::identity{})) {
    return Refuse(where + ".folds", "not sorted, or a fold twice");
  }
  for (const std::string& f : s.folds) {
    if (FindFold(f) == nullptr) {
      return Refuse(where + ".folds", std::format("\"{}\" is not in the fold catalog", f));
    }
  }
  return {};
}

}  // namespace

std::string_view PipelineName(PipelineKind kind) {
  return NameOf(kPipelines, static_cast<std::size_t>(kind));
}
std::string_view StageRoleName(StageRole role) {
  return NameOf(kRoles, static_cast<std::size_t>(role));
}
std::string_view TapPointName(TapPoint point) {
  return NameOf(kTapPoints, static_cast<std::size_t>(point));
}
std::string_view ReadKindName(ReadKind kind) {
  return NameOf(kReadKinds, static_cast<std::size_t>(kind));
}

const Param* ComponentRef::Find(std::string_view name) const {
  const auto found = std::ranges::find(params, name, &Param::name);
  return found == params.end() ? nullptr : &*found;
}

std::span<const FoldInfo> Folds() { return kFolds; }

const FoldInfo* FindFold(std::string_view name) {
  const auto found = std::ranges::find(kFolds, name, &FoldInfo::name);
  return found == kFolds.end() ? nullptr : &*found;
}

const Stage* HoldingStage(const ModelSpec& spec, const Stage& stage, const WeightBinding& binding) {
  if (binding.stage.empty()) {
    return &stage;
  }
  const auto found = std::ranges::find(spec.stages, binding.stage, &Stage::name);
  return found == spec.stages.end() ? nullptr : &*found;
}

void Canonicalize(ModelSpec& spec) {
  const auto component = [](ComponentRef& c) {
    std::ranges::sort(c.params, {}, &Param::name);
    std::ranges::sort(c.weights, {}, [](const WeightBinding& w) {
      return std::pair<std::string_view, format::PartRole>(w.role, w.part);
    });
  };
  for (Stage& s : spec.stages) {
    if (s.sampling.temperature && *s.sampling.temperature == 0.0) *s.sampling.temperature = 0.0;
    if (s.sampling.top_p && *s.sampling.top_p == 0.0) *s.sampling.top_p = 0.0;
    if (s.sampling.min_p && *s.sampling.min_p == 0.0) *s.sampling.min_p = 0.0;
    EachComponent(s, [&](ComponentRef& c, const std::string&) { component(c); });
    for (SideInput& in : s.side) std::ranges::sort(in.layers);
    std::ranges::sort(s.side, {}, &SideInput::name);
    std::ranges::sort(s.taps, {}, &FeatureTap::name);
    std::ranges::sort(s.reads, {}, [](const Read& r) {
      return std::tuple<std::string_view, ReadKind, std::string_view>(r.stage, r.kind, r.name);
    });
    std::ranges::sort(s.folds);
  }
}

Status Validate(const ModelSpec& spec) {
  if (spec.version != kSpecVersion) {
    return Refuse("version", std::format("{}, not this build's {}", spec.version, kSpecVersion));
  }
  if (static_cast<std::size_t>(spec.pipeline) >= kPipelines.size()) {
    return Refuse("pipeline", "an unknown pipeline kind");
  }
  if (spec.stages.empty() || spec.stages.size() > kMaxStages) {
    return Refuse("stages", std::format("one to {} stages", kMaxStages));
  }
  std::set<std::string_view> names;
  std::size_t targets = 0;
  for (std::size_t i = 0; i < spec.stages.size(); ++i) {
    const Stage& s = spec.stages[i];
    if (!names.insert(s.name).second) {
      return Refuse(std::format("stages[{}]", i), "a stage name twice");
    }
    targets += s.role == StageRole::kTarget ? 1 : 0;
    if (auto ok = CheckStage(spec, s, std::format("stages[{}]", i)); !ok) {
      return ok;
    }
    // A stage's folds are its artifact's tensors': stages that run from one
    // artifact list the same, so that none applies a fold another's
    // components take as done.
    for (std::size_t j = 0; j < i; ++j) {
      const Stage& o = spec.stages[j];
      if (o.artifact == s.artifact && o.folds != s.folds) {
        return Refuse(std::format("stages[{}].folds", i),
                      std::format("not those of stage {}, which runs from the same artifact {}",
                                  o.name, s.artifact));
      }
    }
  }
  if ((spec.pipeline == PipelineKind::kLlm || spec.pipeline == PipelineKind::kEmbedding) &&
      (targets != 1 || spec.stages.front().role != StageRole::kTarget)) {
    return Refuse("stages", "a language model's one target stage must come first");
  }
  return {};
}

std::string ToJson(const ModelSpec& spec) {
  std::string out;
  bool first = true;
  out += '{';
  Key(out, "pipeline", first);
  Quoted(out, PipelineName(spec.pipeline));
  Key(out, "stages", first);
  out += '[';
  for (std::size_t i = 0; i < spec.stages.size(); ++i) {
    out += i > 0 ? "," : "";
    WriteStage(out, spec.stages[i]);
  }
  out += ']';
  Key(out, "version", first);
  out += std::to_string(spec.version);
  out += "}\n";
  return out;
}

std::expected<ModelSpec, std::string> FromJson(std::string_view text) {
  if (text.size() > kMaxSpecBytes) {
    return Refuse("spec", std::format("more than {} bytes", kMaxSpecBytes));
  }
  const auto doc = json::Parse(text);
  if (!doc) {
    return Refuse("spec", std::format("not the strict JSON subset ({} at byte {})",
                                      doc.error().reason, doc.error().item));
  }
  if (!doc->canonical()) {
    return Refuse("spec", "not canonical JSON (sorted keys, no whitespace, one trailing newline)");
  }
  const json::Value root = doc->root();
  if (!root.is_object() || !root.find("version")) {
    return Refuse("spec", "not an object with a version");
  }
  ModelSpec spec;
  // The version first: a newer spec's other keys are not this build's.
  const auto version =
      Unsigned(Member(root, "version"), "version", std::numeric_limits<std::uint32_t>::max());
  if (!version) {
    return std::unexpected(version.error());
  }
  if (*version != kSpecVersion) {
    return Refuse("version", std::format("{}, not this build's {}", *version, kSpecVersion));
  }
  if (auto k = Keys(root, "spec", {"pipeline", "stages", "version"}); !k) {
    return std::unexpected(k.error());
  }
  spec.version = kSpecVersion;
  const json::Value pipeline = Member(root, "pipeline");
  const auto p =
      pipeline.is_string() ? ByName<PipelineKind>(kPipelines, pipeline.string()) : std::nullopt;
  if (!p) {
    return Refuse("pipeline", "not a pipeline kind");
  }
  spec.pipeline = *p;
  const auto stages = Array(Member(root, "stages"), "stages", kMaxStages);
  if (!stages) {
    return std::unexpected(stages.error());
  }
  for (std::size_t i = 0; i < stages->size(); ++i) {
    auto s = ReadStage(stages->at(i), std::format("stages[{}]", i));
    if (!s) {
      return std::unexpected(s.error());
    }
    spec.stages.push_back(std::move(*s));
  }
  if (auto ok = Validate(spec); !ok) {
    return std::unexpected(ok.error());
  }
  return spec;
}

base::Sha256Digest ModelSpec::Digest() const {
  return base::Sha256().Update(ToJson(*this)).Finish();
}

std::expected<base::Sha256Digest, std::string> BoundDigest(
    const ModelSpec& spec, std::span<const std::string_view> artifacts) {
  // A spec Validate refuses (unsorted lists among them) would bind a
  // digest its canonical twin does not share.
  if (auto ok = Validate(spec); !ok) {
    return std::unexpected(ok.error());
  }
  if (artifacts.size() > kMaxArtifacts) {
    return Refuse("artifacts", std::format("more than {}", kMaxArtifacts));
  }
  for (const Stage& s : spec.stages) {
    if (s.artifact >= artifacts.size()) {
      return Refuse(s.name, std::format("artifact {} of {} bound", s.artifact, artifacts.size()));
    }
  }
  base::Sha256 h;
  h.Update(std::string_view("llmp.spec.bound.v1"));
  const base::Sha256Digest d = spec.Digest();
  h.Update(std::as_bytes(std::span(d)));
  const auto u32 = [&](std::uint32_t v) {
    std::array<std::byte, 4> b{};
    for (std::size_t i = 0; i < 4; ++i) {
      b.at(i) = static_cast<std::byte>((v >> (8 * i)) & 0xFFU);
    }
    h.Update(b);
  };
  u32(static_cast<std::uint32_t>(artifacts.size()));
  for (const std::string_view id : artifacts) {
    if (id.empty() || id.size() > kMaxText) {
      return Refuse("artifacts", "an empty or overlong artifact ID");
    }
    u32(static_cast<std::uint32_t>(id.size()));
    h.Update(id);
  }
  return h.Finish();
}

}  // namespace llmp::spec
