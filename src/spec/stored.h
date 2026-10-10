// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The model spec as a v1 artifact or composition stores it (D-109, D-113;
// docs/artifact-format.md, "Version 1"), and a stored artifact's tensors as
// a mapping's and the component catalog's source.
//
// A v1 artifact keeps its spec as a document of its own, spec.json, which
// the artifact reader has hashed and checked as strict canonical JSON; its
// schema is read here (FromJson). A model that one artifact holds carries
// its spec in that artifact, every stage running from it (artifact index
// 0); a model of several artifacts (a pipeline's components, a target and
// its drafter) carries it in its composition, the stages' indexes counting
// the composition's components in their order. A component artifact may
// carry a spec of its own (a text encoder usable alone) or none (a drafter
// that reads its target).
//
// The folds the importer applied, which the artifact reader checks only by
// pattern and against its tensors, must be fold catalog names and must be
// among the folds of every stage that runs from that artifact (a stage's
// folds are its own artifact's: a composition's stages may come from
// different containers), so that no component applies one again; and each
// must name a tensor some binding of the spec reads from that artifact.

#ifndef LLMP_SPEC_STORED_H_
#define LLMP_SPEC_STORED_H_

#include <expected>
#include <span>
#include <string>

#include "spec/mapping.h"
#include "spec/spec.h"

namespace llmp::artifact {
class Artifact;
class Composition;
}  // namespace llmp::artifact

namespace llmp::spec {

// The spec a v1 artifact carries: FromJson's, its bytes exactly ToJson's
// (so the digest the manifest lists is the spec's Digest()), every stage
// of artifact 0, every applied fold the catalog's, every stage's and
// naming a tensor the stages bind. Refused for a v0 artifact (its spec is
// synthesized from its architecture's mapping: mapping.h SynthesizeV0)
// and for an artifact that carries none.
std::expected<ModelSpec, std::string> StoredSpec(const artifact::Artifact& artifact);

// The spec a v1 composition carries, with its components opened in the
// composition's order (components[i] the artifact components()[i] names,
// whose ID and architecture must be the recorded ones): every stage's
// artifact index below the component count, and every fold a component's
// importer applied the catalog's, among the folds of every stage that runs
// from that component and naming a tensor bound from it. Refused for a v0
// composition and for one that carries no spec.
std::expected<ModelSpec, std::string> StoredSpec(
    const artifact::Composition& composition,
    std::span<const artifact::Artifact* const> components);

// An artifact's tensors as a source (mapping.h): v0's as SourceOfV0 gives
// them (with `mapping`, the resources its v0_assembly pairs as one
// tensor's parts); v1's each resource role and expert array with the
// format and part its index records (an expert array's slice format with
// the expert count as ne[2]), its manifest's container, no converter, and
// its applied folds, checked against the catalog, as the source's folds
// (so that a spec synthesized from it lists them).
std::expected<Source, std::string> SourceOf(const artifact::Artifact& artifact,
                                            const Mapping* mapping = nullptr);

}  // namespace llmp::spec

#endif  // LLMP_SPEC_STORED_H_
