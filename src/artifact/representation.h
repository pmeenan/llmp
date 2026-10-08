// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Representation descriptors of the v0 index (docs/artifact-format.md,
// "index.json"; D-052): the index, not the container dtype, says what a
// resource's bytes are, and every size the index records must be the one
// its representation implies.
//
//   - GGML {family, type, ne}: a known type, 1-4 positive `ne`, ne[0] a
//     multiple of the block. Pinned GGML CUDA may read row_size(512 -
//     ne[0] mod 512) bytes past a block-quantized tensor whose ne[0] is not
//     a multiple of 512; that over-read is part of the readable range.
//   - EXL3 {family, role, dtype, shape}: only what the fixtures cover. A
//     trellis is I16 [k/16, n/16, 16K] with K in {4, 5, 6, 8}, codebook
//     "mcg" and matching in/out features; suh/svh are 1-D F16; the mcg
//     marker is a scalar I32. Nothing is read past `bytes`.
//   - Plain {family, dtype, shape}.
//
// Sizes are computed with checked arithmetic; one that overflows 64 bits
// saturates, and so never equals a recorded size (the prototype's integers
// are exact; a saturated size is always beyond any size it could accept).

#ifndef LLMP_ARTIFACT_REPRESENTATION_H_
#define LLMP_ARTIFACT_REPRESENTATION_H_

#include <cstddef>
#include <cstdint>
#include <expected>
#include <optional>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

#include "artifact/error.h"
#include "artifact/json.h"

namespace llmp::artifact {

enum class Family : std::uint8_t { kGgml, kExl3, kPlain };
std::string_view FamilyName(Family family);

// A GGML type at the pinned revision: its enum value, block size in
// elements and in bytes.
struct GgmlType {
  std::string_view name;
  std::uint32_t id{};
  std::uint32_t block_elements{};
  std::uint32_t block_bytes{};
};
std::span<const GgmlType> GgmlTypes();
const GgmlType* FindGgmlType(std::string_view name);

// A safetensors container dtype and its element size.
struct Dtype {
  std::string_view name;
  std::uint32_t bytes{};
};
std::span<const Dtype> Dtypes();
const Dtype* FindDtype(std::string_view name);
// GGML types that safetensors expresses natively, keeping dtype and shape.
bool IsNativeContainerType(std::string_view ggml_type);

inline constexpr std::size_t kMaxGgmlDims = 4;
inline constexpr std::size_t kMaxShapeDims = 8;
inline constexpr std::uint64_t kGgmlRowPadding = 512;  // MATRIX_ROW_PADDING at the pin

// A validated descriptor. Its text fields point into static tables, never
// into the artifact's bytes.
struct Representation {
  Family family = Family::kPlain;
  std::string_view type;            // GGML type name, or the container dtype
  std::vector<std::uint64_t> dims;  // GGML `ne`, or `shape`
  std::string_view role;            // EXL3: "trellis", "suh", "svh" or "mcg"
  std::uint32_t k_bits = 0;         // EXL3 trellis
  std::uint64_t in_features = 0;    // EXL3 trellis (saturated)
  std::uint64_t out_features = 0;   // EXL3 trellis (saturated)
  std::string_view codebook;        // EXL3 trellis: "mcg"
  std::uint64_t bytes = 0;          // implied size (saturated)
  std::uint64_t readable = 0;       // implied readable size, over-read included (saturated)
};

// Validates a descriptor, as the prototype's repr_bytes does, rule for
// rule: kRepr for what is unknown, unsupported or inconsistent, kSchema for
// wrong keys or a missing list.
std::expected<Representation, Error> ParseRepresentation(json::Value v);

// The container entry's dtype and shape (informational; the index rules).
struct ContainerView {
  std::string_view dtype;
  std::vector<std::uint64_t> shape;
};
ContainerView ContainerViewOf(const Representation& rep);

// A row table's geometry, from its representation: 2-D only.
struct RowGeometry {
  std::uint64_t rows = 0;
  std::uint64_t row_bytes = 0;
};
std::optional<RowGeometry> RowGeometryOf(const Representation& rep);

// EXL3 closure over every bound name (resource roles and expert array
// names) and its representation: a name with an EXL3 part suffix has the
// exl3 family and that role; every trellis has its suh (in_features long),
// svh (out_features long) and codebook marker; no side vector or marker
// appears without its trellis.
bool Exl3ClosureHolds(std::span<const std::pair<std::string_view, const Representation*>> bound);

}  // namespace llmp::artifact

#endif  // LLMP_ARTIFACT_REPRESENTATION_H_
