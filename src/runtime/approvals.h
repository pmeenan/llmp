// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The pinned reference checkpoints, as data (D-109, D-118, D-119;
// docs/engine-components.md#admitting-a-checkpoint).
//
// Each entry is one qualified reference checkpoint: its identity (the source
// files' receipts and the native tensor inventory's digest), the stored
// spec it must carry (for a drafter, the spec it composes with its approved
// target), the execution envelope it was qualified for and how its
// mapping's planning policy is selected there. When a v1 artifact's source
// receipts match an entry, admission verifies its inventory, transforms and
// stored spec digest against the entry; unpinned v1 artifacts of supported
// architectures are admitted directly by stored spec and mapping validation
// (D-119, runtime/component_admission.h).
#ifndef LLMP_RUNTIME_APPROVALS_H_
#define LLMP_RUNTIME_APPROVALS_H_

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

#include "artifact/artifact.h"
#include "planner/policy.h"

namespace llmp::spec {
struct Mapping;
}  // namespace llmp::spec

namespace llmp::runtime {

// A source file's receipt: its name, SHA-256 (lower-case hex) and bytes.
struct SourceReceipt {
  std::string_view name{};
  std::string_view sha256{};
  std::uint64_t bytes = 0;
};

// How a target entry's planning policy is selected for an envelope: its
// mapping's first rule whose gate (by `gates`, the source keys the
// checkpoint holds as flags) and envelope hold, refused where none does
// within its qualified envelope (kMapping) or the engine's default where
// none does (kMappingOrNone), or always the engine's default (kNone): the
// target's stage plans with exactly the rule admission chose
// (runtime/component_admission.h AdmittedPolicy), or with none. A drafter
// entry names no rule of its own (kNone): its stage plans with its drafting
// mapping's rule for its values, or a shared route's (component_build.cc).
enum class ApprovedPolicy : std::uint8_t { kMapping, kMappingOrNone, kNone };

// An importer transform the artifact records (D-113): an applied fold (by
// its name) or relayout (by the layout its sources held, format::Info's
// name), the tensors it covers and the SHA-256 of their names, sorted, each
// followed by a line feed.
struct ApprovedTransform {
  std::string_view name{};
  std::size_t tensors = 0;
  std::string_view digest{};
};

// Historical evidence for an actual rounding/reduction transformation.
// The enclosing approval binds exact receipts and semantic spec; matching
// component kinds/anchors identify semantic sites, never model-name branches.
struct NumericalEvidenceDomain {
  planner::NumericalChange change{};
  std::string_view evidence{};  // SHA-256 of the authoritative accepted report
  std::string_view component_kind{};
  std::span<const ir::OpKey> anchors{};
  std::uint32_t arch = 121;
  std::uint32_t max_context = 0, max_slots = 0;
  // Actual qualified source envelope; no inference for unrelated smaller
  // sources. Explicit rows lists constrain evidence established at those rows.
  std::uint64_t max_rows = 0;
  std::span<const std::uint64_t> rows{};
  // Retention of an accepted incumbent recipe, not a new shape-specific
  // quality claim. Chunk rows are multiplied by actual admitted owners and
  // the declared semantic multiplicity of this anchor's input.
  bool incumbent_recipe = true;
  bool stream_rows = false;
  std::string_view row_parameter{};
};

struct Approval {
  // The entry's stable name, and the label settings' bases use for its
  // qualified envelope ("Gemma26").
  std::string_view name{};
  std::string_view label{};
  std::string_view architecture{};
  bool drafter = false;
  // The source: its container (the v1 manifest's), the importer folds its
  // tensors carry, its files' receipts.
  std::string_view container{};
  std::span<const std::string_view> folds{};
  std::span<const SourceReceipt> sources{};
  // The applied folds' and relayouts' records exactly (each fold of `folds`
  // in its order); empty `fold_tensors`: the folds by name only.
  std::span<const ApprovedTransform> fold_tensors{};
  std::span<const ApprovedTransform> relayouts{};
  // The native tensor inventory: its tensors (resources' and expert arrays'
  // together), its expert arrays and its complete digest.
  std::size_t tensors = 0;
  std::size_t arrays = 0;
  std::string_view inventory{};
  // The SHA-256 of the spec it executes: a target's stored spec.json, a
  // drafter's composition with its approved target (before any per-alias
  // KV encoding). Every entry pins one.
  std::string_view spec{};
  // A target whose output head must read its embedding's tensor.
  bool tied_output = false;
  // The qualified envelope: a stage context and owners at most, and above
  // `wide_context` exactly one owner (0: none). Its settings' default, not
  // a bound: an override past it is served with a warning (D-119).
  std::uint32_t max_context = 0;
  std::uint32_t max_slots = 0;
  std::uint32_t wide_context = 0;
  ApprovedPolicy policy = ApprovedPolicy::kMapping;
  std::span<const std::string_view> gates{};
  // A drafter's approved target (its entry's name).
  std::string_view target{};
  // The mapping its spec was synthesized by (its policies; a drafter's
  // composition and drafting recipe); null: its architecture's.
  const spec::Mapping& (*mapping)() = nullptr;
  std::span<const NumericalEvidenceDomain> numerical{};
};

std::span<const Approval> Approvals();

// An approved image pipeline (spec/image_family.h): its composition's
// architecture and kept files' receipts, the digest of the family spec it
// stores (its spec.json), and its components' entries in the
// composition's (role) order, each an Approval of its own source, of form
// kDense, pinning the component spec it stores, with no envelope (a
// request's geometry is checked against the family) and policy kNone (the
// pipeline's policy is its mapping's).
struct ImageApproval {
  std::string_view name{};
  std::string_view label{};
  std::string_view architecture{};
  std::span<const SourceReceipt> sources{};
  std::string_view spec{};
  std::span<const Approval> components{};
  std::span<const NumericalEvidenceDomain> numerical{};
  std::uint32_t numerical_width = 0, numerical_height = 0, numerical_prompt_rows = 0;
};

std::span<const ImageApproval> ImageApprovals();

// Whether `sources` are exactly the receipts: every file, each once.
bool SourcesMatch(std::span<const SourceReceipt> receipts,
                  std::span<const artifact::Source> sources);

// The entry whose architecture and source receipts (every file, each once)
// are `architecture` and `sources`; null when none is.
const Approval* FindApproval(std::string_view architecture,
                             std::span<const artifact::Source> sources);
// By name; null when none.
const Approval* FindApprovalNamed(std::string_view name);
// The image pipeline entry of a composition's architecture and kept files;
// null when none is.
const ImageApproval* FindImageApproval(std::string_view architecture,
                                       std::span<const artifact::Source> sources);

}  // namespace llmp::runtime
#endif  // LLMP_RUNTIME_APPROVALS_H_
