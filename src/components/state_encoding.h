// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#ifndef LLMP_COMPONENTS_STATE_ENCODING_H_
#define LLMP_COMPONENTS_STATE_ENCODING_H_

#include <cstdint>
#include <expected>
#include <string>

#include "format/encoding.h"
#include "spec/spec.h"

namespace llmp::components {

// Additional retained parameter storage and serial validation scratch. No
// mutation or inventory allocation. The caller admits this before Apply.
std::expected<std::uint64_t, std::string> KvCacheEncodingBytes(const spec::ModelSpec& model,
                                                               format::Encoding encoding);

// An alias-wide choice for an already owned, validated source spec. This does
// not replace spec/component validation. F16 preserves the
// default spec bytes. Any component declaring a kv_encoding parameter receives
// "q8_0", while components without kv_encoding are left unchanged. Encoding
// eligibility precedes mutation. The caller retains the quoted metadata
// allowance for the transformed spec's lifetime.
std::expected<void, std::string> ApplyKvCacheEncoding(spec::ModelSpec& model,
                                                      format::Encoding encoding,
                                                      std::uint64_t metadata_bytes);

// The source spec a Q8_0 ApplyKvCacheEncoding was applied to: `model` with
// every component's "q8_0" kv_encoding parameter removed. Stored specs carry
// no kv_encoding (F16 is its absence), so this is the admitted spec whose
// identity numerical grants bind to (runtime AdmittedNumerics). A copy of
// the spec, within the caller's temporary spec allowance.
spec::ModelSpec WithoutKvCacheEncoding(const spec::ModelSpec& model);

}  // namespace llmp::components
#endif  // LLMP_COMPONENTS_STATE_ENCODING_H_
