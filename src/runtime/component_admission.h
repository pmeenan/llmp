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

#include "artifact/artifact.h"
#include "spec/mapping.h"
#include "spec/spec.h"

namespace llmp::runtime {

struct ComponentAdmission {
  spec::ModelSpec model;
  spec::Source source;
};

// The same closed check used by artifact admission, exposed over values for
// descriptor controls. Source identity, complete spec, tensor roles/formats
// and source/importer folds must all agree with an approved declaration.
std::expected<void, std::string> CheckComponentAdmission(const spec::ModelSpec& model,
                                                         const spec::Source& source,
                                                         std::span<const artifact::Source> origins);

// v1 only. Reads StoredSpec and SourceOf; v0 retains its existing synthesis.
std::expected<ComponentAdmission, std::string> ReadComponentAdmission(
    const artifact::Artifact& artifact);

// A stored spec has no legacy checkpoint-key bag. Only approved mappings
// whose policy is independent of that bag are supported by this first slice.
std::expected<spec::PolicyRule, std::string> ComponentAdmissionPolicy(
    const spec::Mapping& mapping, const spec::Envelope& envelope);

}  // namespace llmp::runtime
#endif  // LLMP_RUNTIME_COMPONENT_ADMISSION_H_
