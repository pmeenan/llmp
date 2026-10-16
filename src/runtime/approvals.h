// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The approved checkpoints, as data (D-109: "the approved list is data, not
// a profile in code"; docs/engine-components.md#admitting-a-checkpoint).
//
// Each entry is one approved source checkpoint: its identity (the source
// files' receipts and the native tensor inventory's digest), the stored
// spec it must carry (for a drafter, the spec it composes with its approved
// target), the execution envelope it was qualified for and how its
// mapping's planning policy is selected there. A v1 artifact is admitted
// when its source receipts name an entry, its tensors hash to the entry's
// inventory and its stored spec's digest is the entry's
// (runtime/component_admission.h); nothing is synthesized from a compiled
// profile. approvals.cc holds the table, reviewed as a lock file is: a new
// entry or digest is a new approval, whose evidence the owner accepts
// (AGENTS.md: agents never invent supported combinations).
#ifndef LLMP_RUNTIME_APPROVALS_H_
#define LLMP_RUNTIME_APPROVALS_H_

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

#include "artifact/artifact.h"

namespace llmp::runtime {

// A source file's receipt: its name, SHA-256 (lower-case hex) and bytes.
struct SourceReceipt {
  std::string_view name{};
  std::string_view sha256{};
  std::uint64_t bytes = 0;
};

// How the entry's planning policy is selected for an envelope: its
// architecture's mapping's first rule whose gate (by `gates`, the source
// keys the checkpoint holds as flags) and envelope hold, refused where none
// does (kMapping) or the engine's default where none does (kMappingOrNone),
// or always the engine's default (kNone).
enum class ApprovedPolicy : std::uint8_t { kMapping, kMappingOrNone, kNone };

// The tensor formats an inventory admits: GGUF's data parts (GGML blocks,
// dense), EXL3 trellises and their dense tensors, ModelOpt's NVFP4/MXFP8
// splits beside BF16, MLX's affine splits beside BF16. The inventory digest
// covers each tensor's sorted name, encoding and dimensions, and with
// kExl3 each part's role and K bits.
enum class InventoryForm : std::uint8_t { kGguf, kExl3, kModelOpt, kMlx };

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
  // The native tensor inventory: its tensors (resources' and expert arrays'
  // together), its expert arrays, its form and digest.
  std::size_t tensors = 0;
  std::size_t arrays = 0;
  InventoryForm form = InventoryForm::kGguf;
  std::string_view inventory{};
  // The SHA-256 of the spec it executes: a target's stored spec.json, a
  // drafter's composition with its approved target (before any per-alias
  // KV encoding). Empty: not yet pinned (a transitional entry whose
  // stored spec is still checked against its compiled profile; none may be
  // added).
  std::string_view spec{};
  // A target whose output head must read its embedding's tensor.
  bool tied_output = false;
  // The envelope: a stage context and owners at most, and above
  // `wide_context` exactly one owner (0: none).
  std::uint32_t max_context = 0;
  std::uint32_t max_slots = 0;
  std::uint32_t wide_context = 0;
  ApprovedPolicy policy = ApprovedPolicy::kMapping;
  std::span<const std::string_view> gates{};
  // A drafter's approved target (its entry's name).
  std::string_view target{};
};

std::span<const Approval> Approvals();

// The entry whose architecture and source receipts (every file, each once)
// are `architecture` and `sources`; null when none is.
const Approval* FindApproval(std::string_view architecture,
                             std::span<const artifact::Source> sources);
// By name; null when none.
const Approval* FindApprovalNamed(std::string_view name);

}  // namespace llmp::runtime
#endif  // LLMP_RUNTIME_APPROVALS_H_
