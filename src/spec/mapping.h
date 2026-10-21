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
//   - the folds (spec.h's catalog) a source container's converter
//     applied, which with the source's own (a v1 artifact's importer's) are
//     the stage's folds;
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
// (ValidateMapping): their slot, tensor, parameter, fold and policy rules
// run no code. Recipes and retained-metadata extraction are declarative
// data interpreted centrally. Synthesize applies one to a
// source: every tensor of the source matches exactly one rule, every
// required role is bound, every parameter resolves. The native importer
// applies them to checkpoints (import/normalize.h) and stores the spec in
// the v1 artifact it writes (D-113); serving reads that stored spec
// (spec/stored.h) and synthesizes none (D-118).

#ifndef LLMP_SPEC_MAPPING_H_
#define LLMP_SPEC_MAPPING_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "format/encoding.h"
#include "format/tensor_format.h"
#include "spec/spec.h"

namespace llmp::spec {

inline constexpr std::size_t kMaxMappingRules = 1024;
inline constexpr std::size_t kMaxLayerTypes = 256;  // a layer-type pattern's letters
// A source's bounds, as untrusted input: tensors (a ModelOpt checkpoint
// keeps every expert's matrices and scales as tensors of their own:
// Qwen3.8's has 301,730; the GGUF reader's own bound is 2^20) and values
// (a config.json flattened, a GGUF's metadata).
inline constexpr std::size_t kMaxSourceTensors = std::size_t{1} << 20U;
inline constexpr std::size_t kMaxSourceValues = std::size_t{1} << 16U;

// The pseudo-slots parameter rules may name besides a mapping's slots: the
// stage (context), its stream (width, streams) and its heads (softcap).
inline constexpr std::string_view kStageSlot = "stage";
inline constexpr std::string_view kStreamSlot = "stream";
inline constexpr std::string_view kHeadsSlot = "heads";
// The layer number in a tensor pattern.
inline constexpr std::string_view kLayerField = "{layer}";

// kSide: a side input's source (spec.h SideInput, named by its slot), at
// the layers its layers_key names; kMixing: a stage of several streams'
// mixing (spec.h ResidualStream).
enum class Site : std::uint8_t { kEmbedding, kLayer, kFinalNorm, kOutput, kSide, kMixing };

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
  // kLayer: present only at the layers the source key names (an integer,
  // or a list of integers: Qwen3.8's n-gram layer); kSide: the layers that
  // read the side input. Empty: every layer of its types (kLayer).
  std::string_view layers_key{};
  // kLayer, kSide: present only at the layers below the count the source
  // key holds (an integer; absent: 0), or (count_above) only at the others
  // (DeepSeek V4's hash-routed first layers). At most one of layers_key
  // and count_key.
  std::string_view count_key{};
  bool count_above = false;
  // kSide: a side input a layer it is at, named "<slot>_<layer>", its
  // tensor rules' {layer} that layer's, read by that layer alone (a table
  // each layer has of its own: DeepSeek V4's token-hash routing).
  bool per_layer = false;
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
  // A drafter's anchor tensor in its target's logical weight role. Composition
  // borrows that role's complete binding, preserving every part and holder
  // stage; never matched against the drafter's own source.
  bool target = false;
  // The tensor binds with the part it holds, whatever it is (a matrix one
  // container keeps whole, kData, and another as its codes beside its
  // scales, kCodes: the scales a rule of their own).
  bool any_part = false;
  // A tensor the spec does not bind (its role empty): its values are the
  // slot's parameters (the n-gram hash's constants, which a GGUF keeps in
  // its metadata and the ModelOpt import as tensors too).
  bool unbound = false;
};

// kInts: a list of integers (from a key only: a GGUF array, the n-gram
// hash's constants).
enum class ValueType : std::uint8_t { kInt, kReal, kBool, kString, kInts };
// Applied to a real value: 1 / x, sqrt(x), 1 / sqrt(x).
enum class Transform : std::uint8_t { kNone, kReciprocal, kSqrt, kRsqrt, kFloat32 };

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
  // A layer slot's constant name may hold kLayerField once, which is the
  // layer's number (the per-layer side input a layer reads: "hash_{layer}").
  // kLayer slots: the layer types the rule applies to; empty: every type.
  std::string_view layer_types{};
  std::string_view key{};  // the source key; empty: `value` is the parameter
  std::string_view numerator_key{};
  std::string_view divisor_key{};
  std::string_view
      tensor{};  // an optional kBool parameter set true when this source tensor is present
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
  // A slot rule that does not apply to a source carrying this fold (spec.h's
  // catalog): a convention the importer's fold already applied to the bytes
  // (a zero-centered norm's flag beside norm.plus_one; D-108 as amended).
  std::string_view unless_fold{};
};

struct FoldRule {
  std::string_view fold{};       // spec.h's catalog
  std::string_view container{};  // the source container
  bool imported = false;         // an allowed importer fold, never asserted automatically
};

// The planner's default policy for some checkpoints of the architecture,
// in its canonical text (planner/policy.h validates it; here, printable
// lines), gated by a source key as a slot is (SlotRule::gate; empty: every
// checkpoint).
// A rule may hold only within a serving envelope (a family's qualified
// recipe bounded by context and owners: Gemma 4's): a stage context of at
// most `max_context` and at most `max_slots` owners (0: no bound).
// With the policy, how serving composes the recipe's steps where it
// differs from the engine's defaults (engine/wave_composer.h,
// engine/prefill.h): prompt chunks each step alone (no joined prompt
// rows), every step one owner (no joined rows at all), how many of a
// prompt's next chunks are planned ahead (0 to 2) and whether their
// recordings are captured ahead of their runs (Gemma 4's recipe prefills
// alone with one future planned and none captured, and outside its
// envelope steps each owner alone, as its M3 serving did).
// `prefill_chunk` is the rows of a prompt chunk the recipe was qualified
// at, which serving takes where no setting overrides it (0: the family's
// default): Gemma 4's recipes, by profile.
struct PolicyRule {
  std::string_view gate{};
  bool gate_absent = false;
  std::uint32_t max_context = 0;
  std::uint32_t max_slots = 0;
  std::string_view text{};
  bool prompts_alone = false;
  bool owners_alone = false;
  // The admitted composition's joined read semantics, independent of which
  // backend mechanisms serving selects. Numerical grants remain separate.
  bool common_read = false;
  std::uint8_t prefill_lookahead = 2;
  bool capture_ahead = true;
  std::uint32_t prefill_chunk = 0;
};

// What a drafter's stage reads of its target (spec.h Read): its state by
// its components' contract name, or a feature tap, which the composition
// gives the target (at `point`, layer 0) unless it has one of the name.
// A tap at a layer's output or streams is its target's last layer's (the
// streams before the head's mix: an MTP drafter's).
struct ReadRule {
  ReadKind kind = ReadKind::kState;
  std::string_view name{};
  TapPoint point = TapPoint::kPostFinalNorm;
  // A layer tap at a layer the drafter's source names (rather than its
  // target's last layer): entry `index` of the list `layers_key` holds; with
  // `entering`, the streams entering that layer (the tap after the layer
  // before it; the target's layer count names those leaving its last:
  // DSpark's dflash.target_layers).
  std::string_view layers_key{};
  std::uint32_t index = 0;
  bool entering = false;
};

struct Mapping;
struct Source;
struct SourceValue;

// How serving runs a target architecture's models where the engine's
// defaults differ: the architecture's serving recipe as data (D-109: "the
// approved list is data, not a profile in code"), which the runtime's
// settings (runtime/model_settings.cc) and component build
// (runtime/component_build.cc) read instead of naming architectures. What one
// checkpoint is approved for (its envelope, its policy gates) is the
// approval table's (runtime/approvals.h), not this.
//
// An unqualified LLM target mapping (`Mapping::serving == nullptr`) receives
// `kDefaultServingRecipe` via `ServingRecipeOf(mapping)`: batching on
// (the mapping's own `policies` govern `max_decode_rows` and step composition),
// shared route, drafters allowed when configured, `default_draft_rows = 3`,
// `default_wave_read_align = 256`, `chat_thinking = true`, `chat_tools = true`,
// and `frozen_host_masks = true`.
//
// When adding a new family, consider overriding `Mapping::serving` only when:
// - `bounded`, `prefill_rows`, `deep_context` / `deep_prefill_rows`, or
//   `default_context`: the family has a measured context/chunking envelope or
//   attention mask memory bound (e.g., Gemma 3/4, Qwen 3.8, DeepSeek V4).
// - `drafters`, `draft_rows`, `block_drafts`, `adaptive_waves`, `mtp_waves`,
//   and the `default_*` wave priors: the family uses built-in MTP or block
//   speculative drafting with GB10-calibrated wave costs.
// - `chat_thinking`, `chat_tools`, `chat_stops`, `default_thinking`: the
//   family's chat template requires extra stop openers or a non-default
//   thinking channel policy.
// - `serving_rule`: serving uses a step-composition policy distinct from the
//   mapping's admitted default `policies`.
struct ServingRecipe {
  // Settings. `family` and `name` name its bounded envelope in the
  // settings' bases and refusals ("Gemma", "Gemma3").
  std::string_view family{};
  std::string_view name{};
  // Profile defaults may depend on the admitted envelope. Settings remain
  // preferences; actual component/backend and paid capacity bound execution.
  bool bounded = false;
  std::uint32_t prefill_rows = 0;
  // Above this context, `deep_prefill_rows` (the mask's memory; 0: none).
  std::uint32_t deep_context = 0;
  std::uint32_t deep_prefill_rows = 0;
  // A bounded default context, wider ones an explicit opt-in (0: none).
  std::uint32_t default_context = 0;
  // Whether a drafter may be configured at all.
  bool drafters = true;
  // draft_rows applies without a drafter; within a block drafter's block
  // (block_drafts), else an MTP block's two or three passes.
  bool draft_rows = false;
  bool block_drafts = false;
  // wave_form and wave_costs apply, and the shared route forms speculative
  // waves adaptively by them.
  bool adaptive_waves = false;
  // The MTP wave keys apply (draft_vocab, depth_cost_ratio,
  // shared_wave_depth, draft_wave_max, wave_lanes, wave_read_align).
  bool mtp_waves = false;
  // The measured defaults of the drafting and wave settings (their
  // fallbacks under calibration and overrides, runtime/model_settings.h;
  // each measured on a GB10 by the family that sets it; the others keep
  // these, which their serving never reads but draft_rows): drafts a step
  // at most (a block drafter's, at most its block; an MTP block's two or
  // three passes); a draft-verify wave's time over a plain wave's by
  // width from 2 (at most config::kMaxWaveCostWidths, D-104; widths past
  // the list 0); the draft head's rows a pass reads (0: all); a three-draft
  // step's time over a two-draft step's (execution/adaptive_depth.h; 0:
  // its own prior); a request's drafts in a wave of several; how many
  // requests draft in one joined wave (0: no bound); and the state a
  // joined wave reads, rounded up to this.
  std::uint32_t default_draft_rows = 3;
  std::span<const double> default_wave_costs{};
  std::uint32_t default_draft_vocab = 0;
  double default_depth_cost_ratio = 0;
  std::uint32_t default_shared_wave_depth = 0;
  std::uint32_t default_draft_wave_max = 0;
  bool default_wave_lanes = false;
  std::uint32_t default_wave_read_align = 256;
  // What of its chat template the chat route serves (its qualified slice,
  // whatever template the model renders: a native match or an owner's
  // override never widens it): thought-channel output allowed and tool
  // declarations accepted, control tokens that end a chat reply beside the
  // template's own stops (Gemma 4's channel and tool-call openers, which no
  // parser here reads), and a request's thinking when it names none
  // (DeepSeek V4's on, as llama-server's /apply-template rendered its
  // references' prompts; unset, the template's own default).
  bool chat_thinking = true;
  bool chat_tools = true;
  std::span<const std::string_view> chat_stops{};
  std::optional<bool> default_thinking{};
  // The runner stages a frozen read's host mask.
  bool frozen_host_masks = true;
  // The serving composition in place of the admitted one (its policy
  // text and how it composes steps); null: the admitted one.
  const PolicyRule* serving_rule = nullptr;
  std::uint32_t wide_context = 0;
};

// The default serving recipe an unqualified target LLM mapping gets when
// `Mapping::serving` is null: batching on (governed by the mapping's own
// `policies`), shared route, and measured defaults (4,096 prefill rows,
// `docs/runtime-serving.md#model-settings`).
inline constexpr ServingRecipe kDefaultServingRecipe{
    .family = "generic",
    .name = "default",
    .prefill_rows = 4096,
};

// How missing drafter metadata is extracted from the actual target spec and
// the drafter's own tensor inventory. No checkpoint dimensions live here.
enum class ValueOrigin : std::uint8_t {
  kLiteral,
  kContext,
  kWidth,
  kStreams,
  kVocabulary,
  kComponent,
  kTensor,
  kLayerCount,
  kTensorPresent  // a flag: whether any of the drafter's own tensors matches `tensor`
};
struct ValueRule {
  std::string_view key{};
  ValueOrigin origin = ValueOrigin::kLiteral;
  std::string_view kind{}, param{}, tensor{};
  std::uint8_t dimension = 0;
  Literal value{};
  ValueType type = ValueType::kInt;
  bool optional = false;
  bool list = false;          // a scalar component parameter as a one-entry integer list
  std::uint32_t at_most = 0;  // an integer policy default clamped to the actual value
};
struct GateRule {
  std::string_view key{}, kind{}, param{}, container{};
};

// How a drafter architecture's stage drafts (the component build's
// decoding mode): anchored on its target's tapped features and borrowing
// its state (an assistant, a fixed `depth` a unit), a block of proposals
// (at most as many drafts as the drafter's metadata under `depth_key`
// says, filled with the token under `fill_key`), or autoregressive MTP
// passes with catch-up and shared waves.
struct DraftRecipe {
  enum class Shape : std::uint8_t { kAnchored, kBlock, kMtp };
  Shape shape = Shape::kAnchored;
  std::uint32_t depth = 0;
  std::string_view depth_key{};
  std::string_view fill_key{};
  std::string_view target_architecture{};
  // Only used when the stored drafter carries no kept metadata (old native
  // artifacts). New imports use the same target-spec extraction rules.
  std::span<const ValueRule> values{};
  // A selected draft head's token map (I32 [1, rows], strictly ascending
  // target token IDs; empty: no selected head), and the value that names the
  // draft vocabulary's rows, which serving bounds by draft_vocab and the
  // head's rows (else the target's vocabulary).
  std::string_view head_ids{};
  std::string_view vocab_key{};
  double default_depth_cost = 1.16;
  std::uint32_t small_chunk_rows = 4;
  std::uint32_t small_chunk_max_depth = 2;
};

// A layer type by a source key's value (Mapping::layer_type_key).
struct LayerTypeCode {
  std::int64_t value = 0;
  char type = 0;
};

struct Mapping {
  std::string_view architecture{};  // the checkpoint's (GGUF general.architecture)
  std::uint16_t version = 1;        // the mapping's own, for provenance
  PipelineKind pipeline = PipelineKind::kLlm;
  std::string_view layer_count_key{};
  std::string_view layer_types{};  // one letter a layer, repeating
  // Or each layer's type by a source key listing an integer a layer (its
  // first `layers` entries; entries past them unread: a GGUF's MTP heads'),
  // each value's letter in `layer_type_codes` (DeepSeek V4's compression
  // ratios); `layer_types` then lists the letters. Required when set.
  std::string_view layer_type_key{};
  std::span<const LayerTypeCode> layer_type_codes{};
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
  // How serving runs its models (a target's) or drafts (a drafter's);
  // when `serving` is null on a target LLM mapping, `ServingRecipeOf`
  // supplies `kDefaultServingRecipe`.
  const ServingRecipe* serving = nullptr;
  const DraftRecipe* draft = nullptr;
  const Mapping* companion = nullptr;
  // Empty only for the container-neutral synthesis fixture; serving always
  // selects an explicit architecture/container/version entry.
  std::span<const std::string_view> containers{};
  std::span<const GateRule> gates{};
};

// The effective serving recipe for `mapping`: its explicit `serving` recipe
// when set, or `kDefaultServingRecipe` for any target LLM mapping.
constexpr const ServingRecipe* ServingRecipeOf(const Mapping& mapping) {
  if (mapping.serving != nullptr) return mapping.serving;
  if (mapping.role == StageRole::kTarget && mapping.pipeline == PipelineKind::kLlm) {
    return &kDefaultServingRecipe;
  }
  return nullptr;
}
inline constexpr std::size_t kMaxPolicyText = 4096;

// The target's facts a drafter mapping's parameters may read, as source
// keys ComposeDrafter adds: kTargetLastLayer followed by a layer-type
// letter of the target's mapping is the target's last layer of that type
// (a target whose layer types a key gives, layer_type_key, gives none).
inline constexpr std::string_view kTargetLastLayer = "llmp.target.last_layer.";

// The compiled mappings, and the one for an architecture.
std::span<const Mapping* const> Mappings();
const Mapping* FindMapping(std::string_view architecture);
const Mapping* FindMapping(std::string_view architecture, std::string_view container,
                           std::uint16_t version = 1);
std::expected<void, std::string> ValidateMapping(const Mapping& mapping);

// Every source key the mapping reads (its layer, slot, parameter, read and
// policy keys), each once: what a stored drafter's kept metadata is read
// for (its values are the source's, as the importer read them).
std::vector<std::string_view> SourceKeys(const Mapping& mapping);

// A source's metadata value: an integer, a real, a flag or a string.
struct SourceValue {
  std::string key{};
  ParamValue value{};
  bool operator==(const SourceValue&) const = default;
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
  std::string container{};  // a checkpoint's ("gguf", "safetensors"); a v1 artifact's manifest's
  std::vector<SourceTensor> tensors{};
  // Folds the tensors carry beyond the container converter's (the mapping's
  // FoldRules): a v1 artifact's importer's (spec/stored.h SourceOf). The
  // stage made of the source lists them, so that no component applies one
  // again.
  std::vector<std::string> folds{};
};

// The actual target head's rows (its stored tensor geometry).
std::expected<std::uint32_t, std::string> Vocabulary(const ModelSpec& model, const Source& source);

// Retained policy facts derive from components, never architecture names.
std::vector<SourceValue> PolicyGates(const Mapping& mapping, const ModelSpec& spec,
                                     std::string_view container);
std::expected<std::vector<SourceValue>, std::string> ExtractValues(const Mapping& mapping,
                                                                   const ModelSpec& target,
                                                                   const Source& target_source,
                                                                   const Source& drafter);

// The serving envelope a policy is chosen for: the stage's context and the
// owners it serves (0: unknown; an envelope-bounded rule then never holds).
struct Envelope {
  std::uint32_t context = 0;
  std::uint32_t slots = 0;
};

// The default policy rule `mapping` gives a source with `values` (its
// first policy rule whose gate and envelope hold; null when none does),
// and its text (empty when none does).
std::expected<const PolicyRule*, std::string> DefaultRule(const Mapping& mapping,
                                                          std::span<const SourceValue> values,
                                                          const Envelope& envelope = {});
std::expected<std::string_view, std::string> DefaultPolicy(const Mapping& mapping,
                                                           std::span<const SourceValue> values,
                                                           const Envelope& envelope = {});

// The spec `mapping` makes of `source` with `values`: one target stage
// bound to artifact 0, validated. A drafter's mapping is refused (it
// composes with its target: ComposeDrafter).
std::expected<ModelSpec, std::string> Synthesize(const Mapping& mapping, const Source& source,
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

}  // namespace llmp::spec

#endif  // LLMP_SPEC_MAPPING_H_
