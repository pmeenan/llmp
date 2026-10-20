// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The model spec (D-107, D-109; docs/engine-components.md#the-model-spec):
// a model as data. A pipeline kind and its stages (a target and its
// drafters, an encoder, a text encoder, a denoiser, a VAE); each stage's
// embedding, residual stream (one, or hyper-connection streams with their
// mixing), per-layer side inputs, layer list of component references with
// their parameters and weight bindings, feature taps, heads (final norm,
// output, logit softcap, MTP heads) and what it reads from another stage
// (a tap, or state it borrows) and the folds its artifact's tensors already
// carry (FoldInfo). The importer writes it into the artifact (spec/
// mapping.h synthesizes it) and the runtime validates it (spec/stored.h).
//
// Component kinds are strings with versions: the spec names them, the
// component catalog (src/components) knows them and validates parameters
// and weights against them. Validate here checks only what the spec itself
// can: structure, names, ranges, order and references between its parts.
//
// The JSON form is the canonical subset of the v0 manifest's JSON
// (artifact/json.h): integers, printable-ASCII strings without escapes,
// sorted keys, no whitespace and one trailing newline, and FromJson takes
// only canonical bytes, so ToJson(FromJson(t)) == t. It is untrusted input
// once an importer writes it: unknown and duplicate keys, out-of-range
// numbers, unknown names and dangling references are refused. Parameter
// values are typed by their JSON: an integer is an integer, true and false a
// flag, a string starting with a letter a name ([a-z][a-z0-9_]*), a string
// starting with a digit or '-' a real (its shortest round-trip decimal,
// std::to_chars' form, which is canonical; finite, and never -0, which
// would be a second encoding of 0), an array of integers a list of
// integers (an empty array too) and an array of real strings a list of
// reals. A layer is the array of its components; the nesting (eight
// levels: the root, stages, a stage, layers, a layer, a component, its
// parameters, a list) is the v0 JSON's limit, so a v1 artifact keeps the
// spec as a document of its own, spec.json (D-113; stored.h reads it).
//
// Digest() hashes the canonical JSON (the format version is in it). The
// kept-state layout and plan keys derive from BoundDigest: the spec's
// digest with the identities of the artifacts its stages bind, since two
// checkpoints of one spec share layouts but not state.

#ifndef LLMP_SPEC_SPEC_H_
#define LLMP_SPEC_SPEC_H_

#include <cstddef>
#include <cstdint>
#include <expected>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include "base/sha256.h"
#include "format/encoding.h"
#include "spec/graph_links.h"

namespace llmp::spec {

inline constexpr std::uint32_t kSpecVersion = 1;

inline constexpr std::size_t kMaxSpecBytes = std::size_t{4} << 20U;
inline constexpr std::size_t kMaxStages = 16;
inline constexpr std::size_t kMaxArtifacts = 16;  // a stage's artifact index is below
inline constexpr std::size_t kMaxLayers = 4096;
inline constexpr std::size_t kMaxComponents = 65535;  // a stage's (ir::WeightRef::component)
inline constexpr std::size_t kMaxParams = 64;         // a component's
inline constexpr std::size_t kMaxWeights = 256;       // a component's bindings
inline constexpr std::size_t kMaxList = 4096;         // a list parameter's values
inline constexpr std::size_t kMaxNamed = 256;         // taps, side inputs, reads, MTP heads, folds
inline constexpr std::size_t kMaxText = 256;          // a name's or tensor name's bytes
inline constexpr std::uint32_t kMaxStreams = 16;
inline constexpr std::uint64_t kMaxWidth = std::uint64_t{1} << 24U;
inline constexpr std::uint32_t kMaxContext = std::uint32_t{1} << 30U;

enum class PipelineKind : std::uint8_t { kLlm, kImage, kEmbedding, kSpeech, kDecision };
enum class StageRole : std::uint8_t { kTarget, kDrafter, kEncoder, kTextEncoder, kDenoiser, kVae };
enum class TapPoint : std::uint8_t {
  kLayerOut,       // the residual stream after layer `layer`
  kStreams,        // the hyper-connection streams after layer `layer`
  kPostFinalNorm,  // the final norm's output (layer 0)
  kStreamMean,     // the mean of the hyper-connection streams after layer `layer`
};
enum class ReadKind : std::uint8_t {
  kTap,    // another stage's feature tap
  kState,  // another stage's state, borrowed read-only (named by its component's contract)
};

std::string_view PipelineName(PipelineKind kind);
std::string_view StageRoleName(StageRole role);
std::string_view TapPointName(TapPoint point);
std::string_view ReadKindName(ReadKind kind);

// A component parameter: an integer, a real, a flag, a name, or a list of
// integers or reals.
using ParamValue = std::variant<std::int64_t, double, bool, std::string, std::vector<std::int64_t>,
                                std::vector<double>>;

struct Param {
  std::string name{};
  ParamValue value{};
  bool operator==(const Param&) const = default;
};

// A component's weight role bound to an artifact tensor: a resource's role
// (its name or a tied alias) or an expert array's name, and the part
// of its format the tensor holds. The tensor is of the artifact the
// component's stage runs from, or with `stage`, of the artifact another
// stage of the spec runs from: a weight the stages share, which that stage
// owns (a drafter reading its target's token table).
struct WeightBinding {
  std::string role{};
  std::string tensor{};
  format::PartRole part = format::PartRole::kData;
  bool experts = false;  // an expert array: every expert's matrix as one tensor
  std::string stage{};   // another stage's artifact holds it; empty: its own stage's
  bool operator==(const WeightBinding&) const = default;
};

struct ComponentRef {
  std::string kind{};  // "attention.dense"
  std::uint16_t version = 1;
  std::vector<Param> params{};           // sorted by name, names unique
  std::vector<WeightBinding> weights{};  // sorted by (role, part), each pair once
  const Param* Find(std::string_view name) const;
  bool operator==(const ComponentRef&) const = default;
};

struct ResidualStream {
  std::uint32_t streams = 1;  // more than one: hyper-connection streams
  std::uint64_t width = 0;
  std::optional<ComponentRef> mixing{};  // exactly when streams > 1
  bool operator==(const ResidualStream&) const = default;
};

// A per-layer input beside the stream (Qwen3.8's n-gram table, DeepSeek's
// token-hash routing table, image or audio features): the component that
// makes it and the layers that read it.
struct SideInput {
  std::string name{};
  ComponentRef source{};
  std::vector<std::uint32_t> layers{};  // ascending
  bool operator==(const SideInput&) const = default;
};

// A layer's components, in execution order.
struct Layer {
  std::vector<ComponentRef> components{};
  bool operator==(const Layer&) const = default;
};

struct FeatureTap {
  std::string name{};
  std::uint32_t layer = 0;
  TapPoint point = TapPoint::kLayerOut;
  bool operator==(const FeatureTap&) const = default;
};

struct Heads {
  std::optional<ComponentRef> final_norm{};
  std::optional<ComponentRef> output{};
  std::optional<double> softcap{};  // the logits' cap · tanh(logits / cap)
  std::vector<ComponentRef> mtp{};  // in-checkpoint multi-token-prediction heads, in depth order
  bool operator==(const Heads&) const = default;
};

struct Read {
  std::string stage{};
  ReadKind kind = ReadKind::kTap;
  std::string name{};
  bool operator==(const Read&) const = default;
};

struct Stage {
  std::string name{};
  StageRole role = StageRole::kTarget;
  std::uint32_t artifact = 0;  // its artifact: an index into the bound artifacts (a composition's)
  std::uint32_t context = 0;   // the trained context, 0 where none applies
  std::optional<ComponentRef> embedding{};
  ResidualStream stream{};
  std::vector<SideInput> side{};  // sorted by name
  std::vector<Layer> layers{};
  std::vector<FeatureTap> taps{};  // sorted by name
  Heads heads{};
  std::vector<Read> reads{};  // sorted by (stage, kind, name)
  // The folds the tensors of the stage's artifact carry (FoldInfo names,
  // sorted): its source container converter's and its importer's. A
  // stage's, not the model's: a composition's stages may come from
  // different containers. Stages that run from one artifact list the same
  // folds (Validate). A binding to another stage's tensor
  // (WeightBinding::stage) carries that stage's folds (HoldingStage).
  std::vector<std::string> folds{};
  // Optional non-sequential component wiring and multi-value stage exports
  // (1-based component value IDs across layers, 0 = stage input stream).
  // Empty for sequential stages; serialized in canonical JSON only when
  // non-empty (D-122).
  std::vector<GraphLink> links{};
  std::vector<std::uint16_t> exports{};
  bool operator==(const Stage&) const = default;
};

// Every component site in execution order. Shared by validation, canonicalization,
// stored-fold checks and metadata extraction; preserves constness.
template <class StageT, class F>
void EachComponent(StageT& stage, const F& visit) {
  const auto maybe = [&](auto& component, std::string_view path) {
    if (component) visit(*component, std::string(path));
  };
  maybe(stage.embedding, "embedding");
  maybe(stage.stream.mixing, "stream.mixing");
  for (std::size_t i = 0; i < stage.side.size(); ++i)
    visit(stage.side[i].source, "side[" + std::to_string(i) + "].source");
  for (std::size_t l = 0; l < stage.layers.size(); ++l)
    for (std::size_t i = 0; i < stage.layers[l].components.size(); ++i)
      visit(stage.layers[l].components[i],
            "layers[" + std::to_string(l) + "][" + std::to_string(i) + "]");
  maybe(stage.heads.final_norm, "heads.final_norm");
  maybe(stage.heads.output, "heads.output");
  for (std::size_t i = 0; i < stage.heads.mtp.size(); ++i)
    visit(stage.heads.mtp[i], "heads.mtp[" + std::to_string(i) + "]");
}

struct ModelSpec {
  std::uint32_t version = kSpecVersion;
  PipelineKind pipeline = PipelineKind::kLlm;
  std::vector<Stage> stages{};

  // SHA-256 of ToJson(*this).
  base::Sha256Digest Digest() const;
  bool operator==(const ModelSpec&) const = default;
};

// The fold catalog: transforms a checkpoint's converter or llmpalooza's
// importer may already have applied to its tensors, which a component then
// must not apply again (Stage::folds). A mapping cannot name any other.
struct FoldInfo {
  std::string_view name{};
  std::string_view meaning{};
};
std::span<const FoldInfo> Folds();
const FoldInfo* FindFold(std::string_view name);

// The stage whose artifact holds a binding's tensor: the one the binding's
// WeightBinding::stage names, or `stage` (the reading stage, one of the
// spec's) when it names none; null for a name the spec does not have
// (which Validate refuses). The tensor is that stage's artifact's and
// carries that stage's folds, never the reading stage's.
const Stage* HoldingStage(const ModelSpec& spec, const Stage& stage, const WeightBinding& binding);

// Sorts every list the spec keeps sorted (parameters, weights, taps, side
// inputs and their layers, reads, each stage's folds); it does not validate.
void Canonicalize(ModelSpec& spec);

// The spec's own rules (see the header comment); the first violation, with
// where it is.
std::expected<void, std::string> Validate(const ModelSpec& spec);

// Canonical JSON with one trailing newline. A spec that Validate refuses
// may write JSON that FromJson refuses.
std::string ToJson(const ModelSpec& spec);
// Canonical JSON only (at most kMaxSpecBytes), validated.
std::expected<ModelSpec, std::string> FromJson(std::string_view text);

// The spec bound to the artifacts its stages run from (each stage's
// `artifact` indexes `artifacts`, their IDs): what kept-state layout IDs
// and plan keys derive from. Refused when Validate refuses the spec, a
// stage's index is past the list or an ID is empty or longer than kMaxText.
std::expected<base::Sha256Digest, std::string> BoundDigest(
    const ModelSpec& spec, std::span<const std::string_view> artifacts);

}  // namespace llmp::spec

#endif  // LLMP_SPEC_SPEC_H_
