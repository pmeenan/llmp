// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The importer's routed-expert relayout (D-108, D-113; AGENTS.md rule 11,
// "platform-specific work happens at import"): a routed-expert table of a
// block-scaled encoding, gathered from a checkpoint's split matrices, is
// relaid out losslessly into format::Layout::kScaleAtoms, the tile order
// block-scaled tensor-core grouped products read, with a MoE's gate and up
// stacked into one table, and its per-matrix tensor scales gathered into an
// F32 [experts] vector per projection. Chosen by component weight role
// and encoding, never by architecture; only
// where the result is no larger than the source (whole 128-row atoms, whole
// 4-scale atom columns), else the table stays split with its per-expert
// tensor scales in separate projection vectors. Another block-scaled
// encoding of one-byte scales (MXFP8, MXFP4) joins by entering kEncodings
// once its products read the atoms. Row tables of the same encodings take
// the row-interleaved layout (RelayoutRowTable).

#ifndef LLMP_IMPORT_EXPERT_RELAYOUT_H_
#define LLMP_IMPORT_EXPERT_RELAYOUT_H_

#include <cstdint>
#include <expected>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "import/error.h"
#include "import/indexed_parts.h"

namespace llmp::spec {
struct Mapping;
struct ModelSpec;
}  // namespace llmp::spec

namespace llmp::import {

enum class ExpertProjection : std::uint8_t { kGate, kUp, kDown, kOther };

// One routed-expert table among a normalization's outputs: a kMatrices
// assembly of every part (a matrix a source), its role, and the names its
// relaid-out form takes.
struct ExpertTable {
  std::uint32_t output = 0;
  ExpertProjection projection = ExpertProjection::kOther;
  std::uint32_t moe = 0;  // the MoE it belongs to: gate and up of one stack
  // The relaid-out table's name alone, a gate's when stacked with its
  // MoE's up (the gate-up table's), and its tensor scales' vector's when
  // the encoding has them.
  std::string name{}, stacked{}, scales{};
};

// An output's routed-expert table, if the selected mapping (or its
// companion: a drafter's target) names it with a tensor rule whose role the
// component catalog declares an expert projection
// (components::ImportTransform): its MoE is the rule's layer (a drafter's
// apart from its target's), its stacked name the rule in the same slot of
// the role its gate stacks into. No other mapping is consulted.
std::expected<std::optional<ExpertTable>, Error> ExpertTableOf(const TensorAssembly& tensor,
                                                               std::uint32_t output,
                                                               const spec::Mapping& mapping);

struct ExpertRelayoutSummary {
  std::uint32_t tables = 0;         // relaid out
  std::uint32_t stacked = 0;        // gate-and-up stacks among them
  std::uint32_t scale_vectors = 0;  // F32 [experts] tensor-scale vectors
};

// Whether a table's format can be relaid out: a kSplit three-dimensional
// table of an encoding in the relayout's list, whose atoms would hold
// exactly its scales.
bool RelaysOut(const format::TensorFormat& table);

// Rewrites `outputs`: each listed table that RelaysOut becomes its
// kScaleAtoms form (a gate with its MoE's up of the same format stacked,
// the up's output removed), and its tensor scales an F32 [experts] kScalars
// output appended. Split NVFP4 tables that cannot form whole scale atoms
// retain their packed bytes and separate the same scalar vectors. Other
// tables stay as they are. Output indices after the
// first rewritten one change. Refuses a listed output that is not a whole
// expert table or one listed twice.
std::expected<ExpertRelayoutSummary, Error> RelayoutExpertTables(
    std::vector<TensorAssembly>& outputs, std::span<const ExpertTable> tables);

// Overload identifying the expert tables by the selected mapping (ExpertTableOf).
std::expected<ExpertRelayoutSummary, Error> RelayoutExpertTables(
    std::vector<TensorAssembly>& outputs, const spec::Mapping& mapping);

// A row-gathered table (an embedding or side table: the caller names it by
// role) of a block-scaled encoding a device kernel reads row-interleaved
// (NVFP4 alone today), gathered from split row blocks (kRows
// codes and block scales, a shared tensor scale), into
// format::Layout::kRowInterleaved: each row's codes then its scales, so a
// paged row is one read (the row lookups' kernels read it so). The same
// bytes; recorded as a relayout from kSplit. False, unchanged, for a table
// of another form.
std::expected<bool, Error> RelayoutRowTable(TensorAssembly& table);

}  // namespace llmp::import

#endif  // LLMP_IMPORT_EXPERT_RELAYOUT_H_
