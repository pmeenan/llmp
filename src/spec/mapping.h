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
//     local and global layers, "lg");
//   - the folds (spec.h's catalog) a source container's converter applied.
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
};

struct TensorRule {
  std::string_view pattern{};  // kLayerField exactly once in a kLayer slot's, never elsewhere
  std::string_view slot{};
  std::string_view role{};
  format::PartRole part = format::PartRole::kData;
  bool experts = false;   // an expert array
  bool optional = false;  // may be absent
  // Bound instead when the pattern's tensor is absent (a pattern of its
  // own, with the layer where the rule's has one).
  std::string_view fallback{};
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
};

struct FoldRule {
  std::string_view fold{};       // spec.h's catalog
  std::string_view container{};  // the source container whose converter applied it ("gguf")
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
};

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
  std::string container{};  // "gguf", "safetensors", or "other"
  std::vector<SourceTensor> tensors{};
};

// A v0 artifact as a source: every role of every resource (its name and
// tied aliases) and every expert array (its slice's format with the expert
// count as ne[2]); the container "gguf" when every source file is a GGUF.
std::expected<Source, std::string> SourceOfV0(const artifact::Artifact& artifact);

// The spec `mapping` makes of `source` with `values`: one target stage
// bound to artifact 0, validated.
std::expected<ModelSpec, std::string> Synthesize(const Mapping& mapping, const Source& source,
                                                 std::span<const SourceValue> values);
std::expected<ModelSpec, std::string> SynthesizeV0(const artifact::Artifact& artifact,
                                                   const Mapping& mapping,
                                                   std::span<const SourceValue> values);

}  // namespace llmp::spec

#endif  // LLMP_SPEC_MAPPING_H_
