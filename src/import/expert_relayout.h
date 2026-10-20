// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The importer's routed-expert relayout (D-108, D-113; AGENTS.md rule 11,
// "platform-specific work happens at import"): a routed-expert table of a
// block-scaled encoding, gathered from a checkpoint's split matrices, is
// relaid out losslessly into format::Layout::kScaleAtoms, the tile order
// block-scaled tensor-core grouped products read, with a MoE's gate and up
// stacked into one table, and its per-matrix tensor scales gathered into an
// F32 [experts] vector per projection. Chosen by canonical name (the
// converted naming's `blk.L.ffn_{gate,up,down}_exps.weight` tables, whose
// name says the projection: ExpertTableOf) and encoding, never by
// architecture; only
// where the result is no larger than the source (whole 128-row atoms, whole
// 4-scale atom columns), else the table stays split. Another block-scaled
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
struct ModelSpec;
}

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

// The routed-expert table a canonical name names,
// `blk.<layer>.ffn_<gate|up|down>_exps.weight` (the canonical tensor names
// normalizations publish, not an architecture's), with the names its
// relayout gives it: its own, the gate-up table's for a gate, and
// `<name>_scale_2` for its tensor scales. Nullopt for any other name, a
// layer index that is not a canonical decimal below 2^16 among them.
std::optional<ExpertTable> ExpertTableOf(std::string_view name, std::uint32_t output);

// An expert table identified from a component's role in spec when available,
// falling back to canonical name parsing.
std::expected<std::optional<ExpertTable>, Error> ExpertTableOf(
    const TensorAssembly& tensor, const spec::ModelSpec* spec = nullptr);
std::expected<std::optional<ExpertTable>, Error> ExpertTableOf(
    const TensorAssembly& tensor, std::uint32_t output, const spec::ModelSpec* spec = nullptr);

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
// output appended; other tables stay as they are. Output indices after the
// first rewritten one change. Refuses a listed output that is not a whole
// expert table or one listed twice.
std::expected<ExpertRelayoutSummary, Error> RelayoutExpertTables(
    std::vector<TensorAssembly>& outputs, std::span<const ExpertTable> tables);

// Overload identifying expert tables from spec or naming.
std::expected<ExpertRelayoutSummary, Error> RelayoutExpertTables(
    std::vector<TensorAssembly>& outputs, const spec::ModelSpec* spec = nullptr);

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
