// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Tensor formats as the v1 index writes them (docs/artifact-format.md,
// "Version 1"; D-108, D-113): every resource and expert array names its
// tensor's format and the part of it the entry holds, in place of v0's
// library-named representation descriptors.
//
//   {"encoding":"nvfp4","layout":"split","ne":[4096,1024],
//    "params":{"tensor_scale":true},"stacked":1}
//
// `encoding` and `layout` are format::Info's names; `ne` the logical shape
// in GGML's order (innermost first), 0 to 4 positive integers, its length
// the rank; `stacked` 1 to 255; `params` exactly the encoding parameters
// that differ from their zero defaults (`k_bits`, `codebook` "mcg", "mul1"
// or "3inst", `tensor_scale` true, `bits`, `group`, `scale` "f16", "bf16"
// or "f32", `zero_point` true), so that one format has one encoding. The
// size rule (format::PartsOf) decides whether the combination is a format
// at all. Every name read here is matched against static tables and never
// kept, so a parsed format holds nothing from the document.

#ifndef LLMP_ARTIFACT_FORMAT_JSON_H_
#define LLMP_ARTIFACT_FORMAT_JSON_H_

#include <cstdint>
#include <expected>
#include <optional>
#include <string>
#include <string_view>

#include "artifact/error.h"
#include "artifact/json.h"
#include "artifact/representation.h"
#include "format/encoding.h"
#include "format/tensor_format.h"

namespace llmp::artifact {

inline constexpr std::uint64_t kMaxStacked = 255;

// A `format` object: kSchema for missing, unknown or mistyped keys and
// values out of their ranges, kCanonical for a parameter written at its
// default, kRepr for an unknown encoding, layout or codebook and for a
// combination the size rule refuses (format::Validate).
std::expected<format::TensorFormat, Error> ParseFormat(json::Value v);

// The canonical JSON ParseFormat reads, appended to `out` (no newline). Of
// a format format::Validate accepts.
void AppendFormatJson(const format::TensorFormat& f, std::string& out);

// A part role by its format::PartRoleName.
std::optional<format::PartRole> PartRoleByName(std::string_view name);

// The safetensors entry a v1 entry is stored as (informational; the index
// rules): a dense tensor's data part with its dtype and its shape outermost
// first, where safetensors has the dtype; every other part U8 [bytes].
ContainerView ContainerViewOf(const format::TensorFormat& f, format::PartRole part,
                              std::uint64_t bytes);

}  // namespace llmp::artifact

#endif  // LLMP_ARTIFACT_FORMAT_JSON_H_
