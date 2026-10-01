// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// Temporary matched-pipeline benchmark setup. The native paging node owns
// the bounded preparation stream, catalog and staging. Complete derived
// files outlive this setup node and remain open through inference teardown.
#ifndef JITLLM_BENCHMARKS_DS4_COMPLETE_WEIGHT_PREPARATION_H_
#define JITLLM_BENCHMARKS_DS4_COMPLETE_WEIGHT_PREPARATION_H_

#include <cstdint>
#include <expected>
#include <filesystem>
#include <string>
#include <vector>

#include "artifact/artifact.h"
#include "engine/dsv4_ds4_weights.h"
#include "engine/paged_node.h"

namespace jitllm::benchmarks::ds4_complete {

struct PreparedModelWeights {
  engine::Ds4PreparedWeightSet plan;
  std::vector<engine::Ds4PreparedWeightFile> files;
  std::uint64_t direct_read_bytes = 0;
  std::uint64_t stored_bytes = 0;
  std::uint64_t preparation_physical_bytes = 0;
  std::uint64_t preparation_jobs = 0;
  double setup_seconds = 0;
  double preparation_seconds = 0;
  double teardown_seconds = 0;
  engine::StepTimes steps;
};

// Validates and prepares the full original expert replacement and additive
// aligned dense set. One tensor-sized device output, one bounded raw input
// and one pinned host slot are reused only after native completion. No model
// process, raw full-model residency, external runtime or permanent file is
// created. This cost is reported separately from resident model inference.
//
// All returned files are owner-only, unnamed, direct-I/O files. Their hashes
// cover stored bytes including zero padding. A failure drains the native node
// before any captured file/memory owner is released; failed teardown terminates
// this benchmark process rather than pretending that retirement succeeded.
std::expected<PreparedModelWeights, std::string> PrepareModelWeights(
    const artifact::Artifact& artifact, const std::filesystem::path& scratch,
    std::uint64_t payload_limit = std::uint64_t{64} << 20U);

}  // namespace jitllm::benchmarks::ds4_complete

#endif  // JITLLM_BENCHMARKS_DS4_COMPLETE_WEIGHT_PREPARATION_H_
