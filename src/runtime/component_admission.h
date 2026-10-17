// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Admission of stored component specs before settings or execution allocation.
// Approval is data (runtime/approvals.h): an approved source identity, its
// native tensor inventory and the digest of the spec it executes. Admission
// neither fabricates legacy representations nor changes the stored spec.
#ifndef LLMP_RUNTIME_COMPONENT_ADMISSION_H_
#define LLMP_RUNTIME_COMPONENT_ADMISSION_H_

#include <cstdint>
#include <expected>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "artifact/artifact.h"
#include "artifact/composition.h"
#include "model/qwen_image.h"
#include "runtime/approvals.h"
#include "spec/image_family.h"
#include "spec/mapping.h"
#include "spec/spec.h"

namespace llmp::runtime {

// One authority for current production component serving: an architecture
// whose mapping carries a serving or drafting recipe (spec/mapping.h).
bool OnComponentEngine(std::string_view architecture);
// The mapping a composition uses for a drafter architecture (its drafting
// recipe's replay form, or its own); null when it has none.
const spec::Mapping* ComposedDrafterMapping(std::string_view architecture);
// The same for a drafter artifact: by its approval's mapping
// (Approval::mapping) where it names one, else by its architecture.
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
// A drafter's approved envelope (its approval's context and owners).
std::expected<void, std::string> ComponentDrafterEnvelope(const artifact::Artifact& drafter,
                                                          const spec::Envelope& envelope);
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
// An integer the artifact's kept GGUF metadata holds under `key`.
std::expected<std::int64_t, std::string> KeptInteger(const artifact::Artifact& artifact,
                                                     std::string_view key);
// A drafter's composition (before any per-alias KV encoding) is the one its
// approval pins.
std::expected<void, std::string> CheckComposedAdmission(const spec::ModelSpec& composed,
                                                        const artifact::Artifact& drafter);

// `count` integers of a resource, read directly (a draft head's token map).
template <class T>
std::expected<std::vector<T>, std::string> ReadRecipeIntegers(const artifact::Artifact& artifact,
                                                              std::uint32_t resource,
                                                              std::uint64_t count);

struct ComponentAdmission {
  spec::ModelSpec model;
  spec::Source source;
  const Approval* approval = nullptr;
};

// The same closed check used by artifact admission, exposed over values for
// descriptor controls. Source identity, complete spec, tensor roles/formats
// and source/importer folds must all agree with an approved declaration.
std::expected<void, std::string> CheckComponentAdmission(const spec::ModelSpec& model,
                                                         const spec::Source& source,
                                                         std::span<const artifact::Source> origins);
std::expected<void, std::string> CheckComponentDrafterAdmission(
    const spec::Source& source, std::span<const artifact::Source> origins);

// Artifact-only prerequisite, before owned spec/source construction.
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

// Reads StoredSpec and SourceOf of a v1 artifact.
std::expected<ComponentAdmission, std::string> ReadComponentAdmission(
    const artifact::Artifact& artifact);

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
  model::QwenImageProfile profile;
};
std::expected<ImageAdmission, std::string> ReadImageAdmission(
    const artifact::Composition& composition,
    std::span<const artifact::Artifact* const> components);

// The approval's planning policy rule for an envelope (its mapping's first
// rule whose gate, by the approval's gates, and envelope hold; null: none).
std::expected<const spec::PolicyRule*, std::string> ApprovedRule(const Approval& approval,
                                                                 const spec::Envelope& envelope);

// Recheck the complete declaration, then the approved envelope, before
// selecting the approved policy rule for the requested execution envelope.
std::expected<spec::PolicyRule, std::string> ComponentAdmissionPolicy(
    const spec::ModelSpec& model, const spec::Source& source,
    std::span<const artifact::Source> origins, const spec::Envelope& envelope);

}  // namespace llmp::runtime
#endif  // LLMP_RUNTIME_COMPONENT_ADMISSION_H_
