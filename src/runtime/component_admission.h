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
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "artifact/artifact.h"
#include "runtime/approvals.h"
#include "spec/mapping.h"
#include "spec/spec.h"

namespace llmp::model {
struct Qwen38Binding;
struct Qwen38PleHash;
struct Qwen38Tensor;
}  // namespace llmp::model
namespace llmp::runtime {

// One authority for current production component serving: an architecture
// whose mapping carries a serving or drafting recipe (spec/mapping.h).
bool OnComponentEngine(std::string_view architecture);
// The mapping a composition uses for a drafter architecture (its drafting
// recipe's replay form, or its own); null when it has none.
const spec::Mapping* ComposedDrafterMapping(std::string_view architecture);
// Existing v0 recipe values shared by registration facts and actual execution.
// These do not parse a spec or change admission/default policy.
std::expected<std::vector<spec::SourceValue>, std::string> ComponentV0Values(
    const artifact::Artifact& artifact);
// A v0 target's approved binding (Gemma's), before its synthesis is used.
std::expected<void, std::string> CheckComponentV0Admission(const artifact::Artifact& artifact);
std::expected<std::vector<spec::SourceValue>, std::string> ComponentDrafterV0Values(
    const artifact::Artifact& target, const artifact::Artifact& drafter);
// A drafter's values for its composition: v0's from its compiled profile;
// with either stored, the approved pairing of both sources' identities and
// a stored drafter's values from the metadata it keeps.
std::expected<std::vector<spec::SourceValue>, std::string> ComponentDrafterValues(
    const artifact::Artifact& target, const artifact::Artifact& drafter);
// A drafter's approved envelope (its approval's context and owners; one
// without an approval, a v0 MTP drafter, passes).
std::expected<void, std::string> ComponentDrafterEnvelope(const artifact::Artifact& drafter,
                                                          const spec::Envelope& envelope);
// An integer the artifact's kept GGUF metadata holds under `key`.
std::expected<std::int64_t, std::string> KeptInteger(const artifact::Artifact& artifact,
                                                     std::string_view key);
// A stored drafter's composition (before any per-alias KV encoding) is the
// one its approval pins; a v0 drafter's passes.
std::expected<void, std::string> CheckComposedAdmission(const spec::ModelSpec& composed,
                                                        const artifact::Artifact& drafter);

template <class T>
std::expected<std::vector<T>, std::string> ReadRecipeIntegers(const artifact::Artifact& artifact,
                                                              std::uint32_t resource,
                                                              std::uint64_t count);

template <class T>
std::expected<std::vector<T>, std::string> RecipeIntegers(const artifact::Artifact& artifact,
                                                          const model::Qwen38Tensor& tensor,
                                                          std::uint32_t maximum,
                                                          std::string_view type);

std::expected<model::Qwen38PleHash, std::string> RecipeHash(const artifact::Artifact& artifact,
                                                            const model::Qwen38Binding& binding);

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

// v1 only. Reads StoredSpec and SourceOf; v0 retains its existing synthesis.
std::expected<ComponentAdmission, std::string> ReadComponentAdmission(
    const artifact::Artifact& artifact);

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
