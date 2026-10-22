// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
#ifndef LLMP_IMPORT_SAMPLING_DEFAULTS_H_
#define LLMP_IMPORT_SAMPLING_DEFAULTS_H_

#include <cstdint>
#include <expected>
#include <string_view>

#include "import/checkpoint.h"
#include "spec/spec.h"

namespace llmp::import {
struct SamplingDefaultsCost {
  std::uint64_t metadata_bytes = 0, work_units = 0;
};

// Bounds the simultaneous document, JSON arena and parser temporaries before
// allocation. Generation defaults are independent of model architecture.
std::expected<SamplingDefaultsCost, Error> SamplingDefaultsBound(const Checkpoint& source);
// Rechecks the retained document's identity and parsed receipt, and records
// every interpreted byte for publication authentication. Invalid individual
// defaults are absent, as in runtime settings; malformed JSON is refused.
std::expected<spec::SamplingDefaults, Error> ReadSamplingDefaults(
    Checkpoint& source, SamplingDefaultsCost prepaid, std::string_view nested_prefix = {});
}  // namespace llmp::import
#endif  // LLMP_IMPORT_SAMPLING_DEFAULTS_H_
