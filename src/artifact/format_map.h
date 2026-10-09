// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// v0's representation descriptors as tensor formats (D-108), read when an
// artifact is opened: nothing on disk changes; the format version that
// writes encodings and layouts is the native importer's (D-109).
//
// A v0 resource names a kernel library's type ({ggml, type, ne}), a
// container dtype ({plain, dtype, shape}) or an EXL3 part. FormatOf gives
// each resource's own format: a GGML type is its encoding in GGML's
// layout, a dtype its dense encoding in GGML's dimension order, an EXL3
// trellis the whole trellis's format (its other parts are F16 and I32
// resources of their own). Several resources that hold one logical tensor
// (an MXFP8 matrix's codes and scales, the CUTLASS expert arrays, the
// n-gram table and its scale, a trellis with its sign vectors and marker)
// are paired by the model's mapping data, not by the artifact; Assemble
// checks such a pairing against the size rule.

#ifndef LLMP_ARTIFACT_FORMAT_MAP_H_
#define LLMP_ARTIFACT_FORMAT_MAP_H_

#include <cstdint>
#include <expected>
#include <optional>
#include <span>
#include <string_view>

#include "artifact/representation.h"
#include "format/encoding.h"
#include "format/layout.h"
#include "format/tensor_format.h"

namespace llmp::artifact {

// A GGML type name of the pinned table (GgmlTypes()), and a container
// dtype (Dtypes()), as encodings.
std::optional<format::Encoding> EncodingOfGgmlType(std::string_view type);
std::optional<format::Encoding> EncodingOfDtype(std::string_view dtype);

// The resource's own format, whose size rule gives exactly its recorded
// bytes and readable range (for a trellis: its kCodes part's). A plain
// shape of more than four dimensions folds its outer ones into ne[3].
std::expected<format::TensorFormat, format::FormatError> FormatOf(const Representation& rep);

// The part of its tensor's format a resource holds: an EXL3 part's role,
// otherwise kData.
format::PartRole V0PartRole(const Representation& rep);

struct V0Part {
  format::PartRole role = format::PartRole::kData;
  const Representation* rep = nullptr;
};

// The logical format of one 2-D tensor stored as these resources, each the
// part its role names, in `layout`: codes and scales of one-byte elements
// for kSplit and kCutlassScaleAtoms, one interleaved table for
// kRowInterleaved, a GGML resource for the GGML layouts and kDense, a
// trellis with its sign vectors and marker for kExl3Tiles; and for NVFP4 an
// optional F32 tensor scale of `stacked` values. Refused (kParts) unless
// every part the format has is given once and each holds exactly the bytes
// the size rule gives it.
std::expected<format::TensorFormat, format::FormatError> Assemble(format::Encoding encoding,
                                                                  format::Layout layout,
                                                                  std::span<const V0Part> parts,
                                                                  std::uint8_t stacked = 1);

}  // namespace llmp::artifact

#endif  // LLMP_ARTIFACT_FORMAT_MAP_H_
