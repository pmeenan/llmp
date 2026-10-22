// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Admission of stored component specs before settings or execution allocation.
// Stored specs and native tensor formats determine execution eligibility.
// Reference approvals provide performance defaults and numerical evidence;
// changing a valid stored spec does not require a new checkpoint allowlist.
#ifndef LLMP_RUNTIME_COMPONENT_ADMISSION_H_
#define LLMP_RUNTIME_COMPONENT_ADMISSION_H_

#include <cstddef>
#include <cstdint>
#include <expected>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "artifact/artifact.h"
#include "artifact/composition.h"
#include "ir/graph.h"
#include "model/image_pipeline.h"
#include "planner/policy.h"
#include "planner/select.h"
#include "runtime/approvals.h"
#include "spec/image_family.h"
#include "spec/mapping.h"
#include "spec/spec.h"

namespace llmp::kernels {
struct ConstructionFunder;
struct StepBackend;
}  // namespace llmp::kernels

namespace llmp::runtime {

// One authority for current production component serving: an architecture
// whose mapping carries a serving or drafting recipe (spec/mapping.h).
bool OnComponentEngine(std::string_view architecture);
// The mapping a composition uses for a drafter architecture (its drafting
// recipe's replay form, or its own); null when it has none.
const spec::Mapping* ComposedDrafterMapping(std::string_view architecture,
                                            std::string_view container);
// The same for a drafter artifact, using its stored architecture/container
// and the declared composed version. Unknown variants refuse.
const spec::Mapping* ComposedDrafterMapping(const artifact::Artifact& drafter);
// Serving reads format-v1 artifacts only (D-118): a v0 artifact (or any
// other version) is refused with what to do about it, re-importing its
// source with `llmp import`.
std::expected<void, std::string> RequireServedVersion(const artifact::Artifact& artifact);
// A drafter's values for its composition: the approved pairing of both
// sources' identities, and the drafter's values from the metadata it keeps,
// or where it keeps none, its drafting recipe's given its source
// (spec::DraftRecipe::values).
std::expected<std::vector<spec::SourceValue>, std::string> ComponentDrafterValues(
    const artifact::Artifact& target, const artifact::Artifact& drafter);
// A drafter's execution envelope: a context and owners. An approved
// drafter's envelope is its settings' default, not a bound (D-119).
std::expected<void, std::string> ComponentDrafterEnvelope(const spec::Envelope& envelope);
// A drafter's selected draft head by its token map `head_ids` (its drafting
// recipe's): the map's resource and rows, I32 [1, rows] with 0 < rows <=
// `vocab`; nullopt without one; refused for another shape.
struct DraftHeadMap {
  std::uint32_t resource = 0;
  std::uint64_t rows = 0;
};
std::expected<std::optional<DraftHeadMap>, std::string> DraftHeadRows(
    const artifact::Artifact& drafter, std::string_view head_ids, std::uint64_t vocab);
// Apply the effective draft vocabulary to composition values. Zero requests
// all available rows; a selected head bounds the available rows. Shared by
// settings identity derivation and the executable factory.
std::expected<std::uint32_t, std::string> ConfigureDraftVocabulary(
    std::span<spec::SourceValue> values, const spec::DraftRecipe& recipe, std::uint32_t requested,
    std::uint32_t target_vocab, std::optional<std::uint32_t> selected_rows = std::nullopt);
// The kept GGUF metadata file's name (a single file's, or the first
// shard's, beneath `meta/`); empty when the artifact keeps none.
std::string KeptGguf(const artifact::Artifact& artifact);
// [0, place.bytes) of `artifact`'s data at `place`, read directly through
// 4 KiB windows within its shard, handed to `each` a window's part at a
// time with its offset from the start; the first refusal (each's, or the
// read's) ends it.
std::expected<void, std::string> ReadPlaced(const artifact::Artifact& artifact,
                                            const artifact::Placement& place,
                                            const std::function<std::expected<void, std::string>(
                                                std::uint64_t, std::span<const std::byte>)>& each);
// An integer the artifact's kept GGUF metadata holds under `key`.
std::expected<std::int64_t, std::string> KeptInteger(const artifact::Artifact& artifact,
                                                     std::string_view key);
// A drafter's composition (before any per-alias KV encoding) with `target`:
// one with a drafter stage whose component bindings validate against the
// two artifacts' native tensor descriptions.
std::expected<void, std::string> CheckComposedAdmission(const spec::ModelSpec& composed,
                                                        const artifact::Artifact& target,
                                                        const artifact::Artifact& drafter);
std::expected<void, std::string> CheckComposedAdmission(const spec::ModelSpec& composed,
                                                        const spec::Source& target,
                                                        const spec::Source& drafter);

// `count` integers of a resource, read directly (a draft head's token map).
template <class T>
std::expected<std::vector<T>, std::string> ReadRecipeIntegers(const artifact::Artifact& artifact,
                                                              std::uint32_t resource,
                                                              std::uint64_t count);

// An admitted target (ReadComponentAdmission) or drafter
// (ReadComponentDrafterAdmission): what admission read, checked once, for
// what follows to use rather than check again.
struct ComponentAdmission {
  spec::ModelSpec model;
  spec::Source source;
  const Approval* approval = nullptr;
};
struct ComponentDrafterAdmission {
  std::optional<spec::ModelSpec> model{};
  spec::Source source;
  const Approval* approval = nullptr;
};

// The same structural check used by artifact admission, exposed over values
// for descriptor controls: valid provenance, complete stored spec, native
// tensor roles/formats and consistent source/importer folds.
std::expected<void, std::string> CheckComponentAdmission(const spec::ModelSpec& model,
                                                         const spec::Source& source,
                                                         std::span<const artifact::Source> origins);
std::expected<void, std::string> CheckComponentDrafterAdmission(
    const spec::Source& source, std::span<const artifact::Source> origins);

// The artifact-level prerequisites, before its spec or source is read: a v1
// artifact with well-formed source receipts, an architecture with a mapping,
// and valid scale-atom expert slots.
std::expected<void, std::string> CheckArtifactPrerequisites(const artifact::Artifact& artifact);
// The whole admission of a target or drafter artifact, by its mapping's
// role (ReadComponentAdmission or ReadComponentDrafterAdmission, results
// dropped): for callers that only ask whether it is admitted.
std::expected<void, std::string> CheckComponentArtifactAdmission(
    const artifact::Artifact& artifact);

// A v1 artifact's routed experts in the scale-atom layout (the importer's
// relayout, import/expert_relayout.h) as the block-scaled grouped GEMM reads
// them: in every expert group of a layer with a stacked scale-atom table,
// that table's codes and scales, then the layer's other tables' codes and
// scales, back to back from the group's start (kernels/ggml/moe_layout.h
// ExpertLayout); layers without one are not checked.
std::expected<void, std::string> CheckScaleAtomSlots(const artifact::Artifact& artifact);

// Native hash-routing tables retain I32 source values. Validate each table
// `model` binds (each side.hash input's, by its own `experts` and `used`
// rows a token) through a bounded direct-read window of `artifact` before
// any routed work: an I32 [used, tokens] table of consistent bytes whose
// every entry names an expert in [0, experts). No whole-table copy or
// uncharged payload allocation is introduced for stored targets.
std::expected<void, std::string> CheckStoredHashTables(const artifact::Artifact& artifact,
                                                       const spec::ModelSpec& model);

// A target's admission, once: its prerequisites, then its StoredSpec and
// SourceOf checked (CheckComponentAdmission).
std::expected<ComponentAdmission, std::string> ReadComponentAdmission(
    const artifact::Artifact& artifact);
// A drafter's admission, once: its prerequisites, then its SourceOf checked
// (CheckComponentDrafterAdmission).
std::expected<ComponentDrafterAdmission, std::string> ReadComponentDrafterAdmission(
    const artifact::Artifact& drafter);

// The default context a mapping's rules select across its non-owners-alone
// policies matching `gates`; 0 when none.
std::uint32_t MappingDefaultContext(const spec::Mapping& mapping,
                                    std::span<const spec::SourceValue> gates);

// The context a configuration that names none takes within an approval:
// the widest context of its mapping's rule domains that its gates select
// and that join owners (a rule bounded by context, not an owners-alone
// one), else its envelope's.
std::uint32_t ApprovedDefaultContext(const Approval& approval);

// An image pipeline's admission (spec/image_family.h): a v1 composition
// whose architecture has an image mapping and whose kept files are an
// approved pipeline's (ImageApproval), its components (opened in the
// composition's order) each its approved source, dense inventory and
// pinned component spec, and the family spec it stores the pinned one,
// read back as the model's profile.
struct ImageAdmission {
  const spec::ImageMapping* mapping = nullptr;
  const ImageApproval* approval = nullptr;
  spec::ModelSpec family;
  model::ImageProfile profile;
};
std::expected<ImageAdmission, std::string> ReadImageAdmission(
    const artifact::Composition& composition,
    std::span<const artifact::Artifact* const> components);

// Uses the backend's existing paid construction token, independently of
// execution-domain declarations, and retains its exact charge/refund lifetime.
// Called after ordinary admission, before per-alias state transformations or
// composition: exact verified source/spec identity is checked once here.
std::expected<std::shared_ptr<const planner::NumericalQualifications>, std::string>
AdmittedNumerics(const spec::ModelSpec& model, const spec::Source& source,
                 std::span<const artifact::Source> origins, const Approval* approval,
                 const base::Sha256Digest& admitted_semantic, const spec::Envelope& envelope,
                 std::uint16_t active_stage, const planner::DeviceFacts& device,
                 std::shared_ptr<const kernels::ConstructionFunder> funder);
// Serving boundary: declared owners supply the authoritative device facts and
// existing construction payer. Mutable token fields cannot replace either.
std::expected<std::shared_ptr<const planner::NumericalQualifications>, std::string>
AdmittedNumerics(const spec::ModelSpec& model, const spec::Source& source,
                 std::span<const artifact::Source> origins, const Approval* approval,
                 const base::Sha256Digest& admitted_semantic, const spec::Envelope& envelope,
                 std::uint16_t active_stage, const kernels::StepBackend& backend);
// Join the generated stage's declared component semantics with accepted
// component/device evidence, binding actual graph roles and query dimensions.
// Historical checkpoint approval membership supplies no numerical grants.
std::expected<std::shared_ptr<const planner::NumericalQualifications>, std::string>
AdmittedImageNumerics(const spec::ModelSpec& family, const ImageApproval* approval,
                      const artifact::Composition& composition, const ir::Graph& graph,
                      const spec::ModelSpec& stage_spec, std::uint16_t stage,
                      const planner::DeviceFacts& device,
                      std::shared_ptr<const kernels::ConstructionFunder> funder);

// The approval's planning policy rule for an envelope (its mapping's first
// rule whose gate, by the approval's gates, and envelope hold; null: none).
std::expected<const spec::PolicyRule*, std::string> ApprovedRule(const Approval& approval,
                                                                 const spec::Envelope& envelope);

// Whether an envelope lies within an approval's qualified one (its context
// and owners, and one owner above its wide context): its settings' default.
bool WithinApprovedEnvelope(const Approval& approval, const spec::Envelope& envelope);
// An admitted target's planning policy rule for an execution envelope
// within its stored stage context: an approved one's rule (refused where
// its kMapping policy has none within its qualified envelope; past that
// envelope, an override, the engine's default where none holds), an
// unpinned one's mapping's rule for its gates, else the engine's default.
std::expected<spec::PolicyRule, std::string> AdmittedPolicy(const spec::ModelSpec& model,
                                                            const spec::Source& source,
                                                            const Approval* approval,
                                                            const spec::Envelope& envelope);
// CheckComponentAdmission, then AdmittedPolicy (descriptor checks: the
// runtime admits once, ReadComponentAdmission, and asks AdmittedPolicy).
std::expected<spec::PolicyRule, std::string> ComponentAdmissionPolicy(
    const spec::ModelSpec& model, const spec::Source& source,
    std::span<const artifact::Source> origins, const spec::Envelope& envelope);

}  // namespace llmp::runtime
#endif  // LLMP_RUNTIME_COMPONENT_ADMISSION_H_
