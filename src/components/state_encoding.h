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
// default spec bytes. Q8_0 requires one target and exclusively owned dense
// attention state at D64/D256. Encoding eligibility precedes mutation; borrowed,
// specialized, mixed-width and explicit-score consumers refuse. The caller
// retains the quoted metadata allowance for the transformed spec's lifetime.
std::expected<void, std::string> ApplyKvCacheEncoding(spec::ModelSpec& model,
                                                      format::Encoding encoding,
                                                      std::uint64_t metadata_bytes);

}  // namespace llmp::components
#endif  // LLMP_COMPONENTS_STATE_ENCODING_H_
