// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Admission of stored component specs before settings or execution allocation.
// Approval is a closed profile declaration over native tensor formats; it
// neither fabricates legacy representations nor changes the stored spec.
#ifndef LLMP_RUNTIME_COMPONENT_ADMISSION_H_
#define LLMP_RUNTIME_COMPONENT_ADMISSION_H_

#include <expected>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "artifact/artifact.h"
#include "spec/mapping.h"
#include "spec/spec.h"

namespace llmp::model {
struct Gemma4Profile;
}
namespace llmp::runtime {

// One authority for current production component serving. Diagnostic Qwen/DS
// paths remain separate until their serving migration is qualified.
bool OnComponentEngine(std::string_view architecture);
// Existing v0 recipe values shared by registration facts and actual execution.
// These do not parse a spec or change admission/default policy.
std::expected<std::vector<spec::SourceValue>, std::string> ComponentV0Values(
    const artifact::Artifact& artifact);
std::expected<std::vector<spec::SourceValue>, std::string> ComponentDrafterV0Values(
    const artifact::Artifact& target, const artifact::Artifact& drafter);
std::expected<std::vector<spec::SourceValue>, std::string> ComponentDrafterValues(
    const artifact::Artifact& target, const artifact::Artifact& drafter);

struct ComponentAdmission {
  spec::ModelSpec model;
  spec::Source source;
  const model::Gemma4Profile* gemma_profile = nullptr;
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

// Recheck the complete declaration before selecting the approved mapping's
// recipe from its proved profile facts and the requested execution envelope.
std::expected<spec::PolicyRule, std::string> ComponentAdmissionPolicy(
    const spec::ModelSpec& model, const spec::Source& source,
    std::span<const artifact::Source> origins, const spec::Envelope& envelope);

}  // namespace llmp::runtime
#endif  // LLMP_RUNTIME_COMPONENT_ADMISSION_H_
