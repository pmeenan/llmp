// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Architecture mappings (D-109; docs/engine-components.md#the-importer): an
// architecture as declarative data, from which a model spec (spec.h) is
// made for a checkpoint of it. A mapping names
//
//   - its slots: the components a model of the architecture has, where
//     (the embedding, every layer or the layers of some types, the final
//     norm, the output head), of which kind and version;
//   - tensor rules: name patterns ("blk.{layer}.attn_q.weight") to a slot's
//     weight role and part, optional ones, and a fallback a tied tensor
//     binds instead (an output head reading the embedding);
//   - parameter rules: source keys (GGUF metadata, config.json) to a
//     slot's parameters, typed, with a transform from a fixed set, per
//     layer type, or a constant (llama.cpp's built-in defaults); several
//     rules for one parameter are tried in order. A rule without a slot is a
//     requirement on the source (a rotation scaling kind the components
//     implement);
//   - its layer types, one letter a layer repeating from layer 0 (Gemma 2's
//     local and global layers, "lg"), and optionally the source key that
//     lists them as a flag a layer (Gemma 4's sliding-window pattern),
//     which must agree;
//   - the folds (spec.h's catalog) a source container's converter applied,
//     which with the source's own (an importer's) are the stage's folds;
//   - its default planning policies (planner/policy.h's canonical text),
//     each gated by a source key as a slot is: the patterns and bounds a
//     model of the architecture is planned with, the first migrations'
//     recipes as data (an alias's settings may replace it). The spec
//     module does not parse them; the step source does;
//   - for a drafter's architecture (an assistant), its stage's role and
//     name and what it reads of its target (its KV state, a feature tap):
//     ComposeDrafter makes a two-stage spec of the target's and the
//     drafter's checkpoints, the drafter's tensor rules may bind its
//     target's tensors (the target's token table), and its parameters may
//     read the target's facts (kTargetLastLayer).
//
// Slots may be gated by a source key (a routed-expert block in a
// checkpoint that has experts, a dense one in one that has none); a
// parameter may read a key that lists one value a layer (per_layer:
// Gemma 4's KV head counts); a tensor rule of a layer slot may name a
// tensor without a layer, which every layer of its types binds (a shared
// rotation-factor table), and may apply to some layer types only.
//
// Mappings are compiled in as data and validated like untrusted input
// (ValidateMapping): a mapping runs no code. Synthesize applies one to a
// source: every tensor of the source matches exactly one rule, every
// required role is bound, every parameter resolves. SynthesizeV0 does that
// for a v0 artifact, whose tensors are its resources' roles and expert
// arrays and whose values are today's compiled profile's (no new on-disk
// parsing: spec/mappings/*.h make them); the native importer later applies
// the same mappings to checkpoints.

#ifndef LLMP_SPEC_MAPPING_H_
#define LLMP_SPEC_MAPPING_H_

#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "format/encoding.h"
#include "format/tensor_format.h"
#include "spec/spec.h"

namespace llmp::artifact {
class Artifact;
}

namespace llmp::spec {

inline constexpr std::size_t kMaxMappingRules = 1024;
inline constexpr std::size_t kMaxLayerTypes = 64;  // a layer-type pattern's letters
inline constexpr std::size_t kMaxSourceTensors = std::size_t{1} << 18U;
inline constexpr std::size_t kMaxSourceValues = 4096;

// The pseudo-slots parameter rules may name besides a mapping's slots: the
// stage (context), its stream (width, streams) and its heads (softcap).
inline constexpr std::string_view kStageSlot = "stage";
inline constexpr std::string_view kStreamSlot = "stream";
inline constexpr std::string_view kHeadsSlot = "heads";
// The layer number in a tensor pattern.
inline constexpr std::string_view kLayerField = "{layer}";

enum class Site : std::uint8_t { kEmbedding, kLayer, kFinalNorm, kOutput };

struct SlotRule {
  std::string_view slot{};  // a name, unique in the mapping
  Site site = Site::kLayer;
  std::string_view kind{};  // the component kind
  std::uint16_t version = 1;
  std::string_view layer_types{};  // kLayer: the layer types that have it; empty: every layer
  // A slot some checkpoints of the architecture have: present when the
  // source key `gate` holds an integer above zero, or (gate_absent) only
  // when it is absent or zero. Empty: always present.
  std::string_view gate{};
  bool gate_absent = false;
};

struct TensorRule {
  // kLayerField at most once, and only in a kLayer slot's: a kLayer slot's
  // pattern without it names one tensor that every layer of the rule's
  // types binds (a table the layers share).
  std::string_view pattern{};
  std::string_view slot{};
  std::string_view role{};
  format::PartRole part = format::PartRole::kData;
  bool experts = false;   // an expert array
  bool optional = false;  // may be absent
  // Bound instead when the pattern's tensor is absent (a pattern of its
  // own, with the layer where the rule's has one).
  std::string_view fallback{};
  // kLayer slots: the layer types whose layers bind it (a layer of another
  // type that has the tensor is refused); empty: every type the slot has.
  std::string_view layer_types{};
  // A drafter mapping's rule for a tensor of its target's checkpoint (bound
  // with the target stage's name, spec.h WeightBinding::stage): never
  // matched against the drafter's own source.
  bool target = false;
};

enum class ValueType : std::uint8_t { kInt, kReal, kBool, kString };
// Applied to a real value: 1 / x, sqrt(x), 1 / sqrt(x).
enum class Transform : std::uint8_t { kNone, kReciprocal, kSqrt, kRsqrt };

// A rule's constant (when it has no key) or a requirement's value.
struct Literal {
  std::int64_t integer = 0;
  double real = 0.0;
  bool flag = false;
  std::string_view text{};
};

struct ParamRule {
  std::string_view slot{};  // a slot, a pseudo-slot, or empty: a requirement
  std::string_view param{};
  // kLayer slots: the layer types the rule applies to; empty: every type.
  std::string_view layer_types{};
  std::string_view key{};  // the source key; empty: `value` is the parameter
  ValueType type = ValueType::kInt;
  Transform transform = Transform::kNone;
  Literal value{};
  // When no rule for the parameter resolves, it is left out (a
  // requirement: the key may be absent).
  bool optional = false;
  // kLayer slots, kInt or kReal: the key's value is a list with an entry a
  // layer, the layer's being the value, or one value for every layer
  // (llama.cpp's get_key_or_arr).
  bool per_layer = false;
};

struct FoldRule {
  std::string_view fold{};       // spec.h's catalog
  std::string_view container{};  // the source container whose converter applied it ("gguf")
};

// The planner's default policy for some checkpoints of the architecture,
// in its canonical text (planner/policy.h validates it; here, printable
// lines), gated by a source key as a slot is (SlotRule::gate; empty: every
// checkpoint).
// A rule may hold only within a serving envelope (a family's qualified
// recipe bounded by context and owners: Gemma 4's): a stage context of at
// most `max_context` and at most `max_slots` owners (0: no bound).
struct PolicyRule {
  std::string_view gate{};
  bool gate_absent = false;
  std::uint32_t max_context = 0;
  std::uint32_t max_slots = 0;
  std::string_view text{};
};

// What a drafter's stage reads of its target (spec.h Read): its state by
// its components' contract name, or a feature tap, which the composition
// gives the target (at `point`, layer 0) unless it has one of the name.
struct ReadRule {
  ReadKind kind = ReadKind::kState;
  std::string_view name{};
  TapPoint point = TapPoint::kPostFinalNorm;
};

struct Mapping {
  std::string_view architecture{};  // the checkpoint's (GGUF general.architecture)
  std::uint16_t version = 1;        // the mapping's own, for provenance
  PipelineKind pipeline = PipelineKind::kLlm;
  std::string_view layer_count_key{};
  std::string_view layer_types{};  // one letter a layer, repeating
  std::span<const SlotRule> slots{};
  std::span<const TensorRule> tensors{};
  std::span<const ParamRule> params{};
  std::span<const FoldRule> folds{};
  // A source key listing a flag a layer (1 or true where the layer is of
  // type `layer_flag_type`): when the source has it, it must agree with
  // layer_types layer for layer.
  std::string_view layer_flags_key{};
  char layer_flag_type = 0;
  // A drafter's architecture (role kDrafter): its stage's name and what it
  // reads of its target; see ComposeDrafter.
  StageRole role = StageRole::kTarget;
  std::string_view stage{};
  std::span<const ReadRule> reads{};
  // The default planning policy: the first rule whose gate holds (each
  // text at most kMaxPolicyText bytes of printable lines, each ending in
  // '\n'); none: the planner's default policy (every pattern).
  std::span<const PolicyRule> policies{};
};
inline constexpr std::size_t kMaxPolicyText = 4096;

// The target's facts a drafter mapping's parameters may read, as source
// keys ComposeDrafter adds: kTargetLastLayer followed by a layer-type
// letter of the target's mapping is the target's last layer of that type.
inline constexpr std::string_view kTargetLastLayer = "llmp.target.last_layer.";

// The compiled mappings, and the one for an architecture.
std::span<const Mapping* const> Mappings();
const Mapping* FindMapping(std::string_view architecture);
std::expected<void, std::string> ValidateMapping(const Mapping& mapping);

// A source's metadata value: an integer, a real, a flag or a string.
struct SourceValue {
  std::string key{};
  ParamValue value{};
};

// Whether `key` can name a source value (and a mapping's tensor name):
// 1 to kMaxText bytes of printable ASCII without space, quotes, backslash
// or braces. Synthesize refuses values under any other key.
bool IsSourceKey(std::string_view key);

// A source's tensors as the mapping sees them.
struct SourceTensor {
  std::string name{};
  format::TensorFormat format{};
  format::PartRole part = format::PartRole::kData;
  bool experts = false;
};
struct Source {
  std::string architecture{};
  std::string container{};  // v0: "gguf", "safetensors" or "other"; v1: its manifest's
  std::vector<SourceTensor> tensors{};
  // Folds the tensors carry beyond the container converter's (the mapping's
  // FoldRules): a v1 artifact's importer's (spec/stored.h SourceOf). The
  // stage made of the source lists them, so that no component applies one
  // again.
  std::vector<std::string> folds{};
};

// A v0 artifact as a source: every role of every resource (its name and
// tied aliases) and every expert array (its slice's format with the expert
// count as ne[2]); the container "gguf" when every source file is a GGUF.
// Any other format version is refused (stored.h SourceOf reads both).
std::expected<Source, std::string> SourceOfV0(const artifact::Artifact& artifact);

// The serving envelope a policy is chosen for: the stage's context and the
// owners it serves (0: unknown; an envelope-bounded rule then never holds).
struct Envelope {
  std::uint32_t context = 0;
  std::uint32_t slots = 0;
};

// The default policy text `mapping` gives a source with `values` (its
// first policy rule whose gate and envelope hold; empty when none does).
std::expected<std::string_view, std::string> DefaultPolicy(const Mapping& mapping,
                                                           std::span<const SourceValue> values,
                                                           const Envelope& envelope = {});

// The spec `mapping` makes of `source` with `values`: one target stage
// bound to artifact 0, validated. A drafter's mapping is refused (it
// composes with its target: ComposeDrafter).
std::expected<ModelSpec, std::string> Synthesize(const Mapping& mapping, const Source& source,
                                                 std::span<const SourceValue> values);
std::expected<ModelSpec, std::string> SynthesizeV0(const artifact::Artifact& artifact,
                                                   const Mapping& mapping,
                                                   std::span<const SourceValue> values);

// `target` (a spec whose first stage is its target, made by
// `target_mapping`) with a drafter stage the drafter's mapping makes of
// its source and values, bound to the next artifact index: the drafter's
// reads of the target, its target-tensor bindings (WeightBinding::stage),
// its own source's folds (never the target's), and the target's taps it
// reads added to the target stage. The values gain the target's facts
// (kTargetLastLayer). Validated.
std::expected<ModelSpec, std::string> ComposeDrafter(const ModelSpec& target,
                                                     const Mapping& target_mapping,
                                                     const Mapping& drafter, const Source& source,
                                                     std::span<const SourceValue> values);
std::expected<ModelSpec, std::string> ComposeDrafterV0(const ModelSpec& target,
                                                       const Mapping& target_mapping,
                                                       const artifact::Artifact& artifact,
                                                       const Mapping& drafter,
                                                       std::span<const SourceValue> values);

}  // namespace llmp::spec

#endif  // LLMP_SPEC_MAPPING_H_
