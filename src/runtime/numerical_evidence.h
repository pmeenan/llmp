// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#ifndef LLMP_RUNTIME_NUMERICAL_EVIDENCE_H_
#define LLMP_RUNTIME_NUMERICAL_EVIDENCE_H_

#include <cstdint>
#include <span>
#include <string_view>

#include "components/numerics.h"
#include "planner/select.h"

namespace llmp::runtime {

// Accepted arithmetic evidence, separated from checkpoint approval/defaults.
// These are bounds on optimization evidence, never request admission limits.
enum class NumericalPipeline : std::uint8_t { kLlm, kImage };

struct ComponentNumericalEvidence {
  ir::NumericalChange change;
  std::string_view semantic, report, digest;
  std::string_view device_class, device_model;
  std::uint32_t arch, multiprocessors;
  std::span<const format::Encoding> encodings;
  std::span<const std::uint64_t> rows{};
  std::uint64_t rows_per_owner = 0;
  std::uint32_t owners = 0;
  bool stream_rows = false;
  std::string_view row_parameter{};
  bool incumbent_recipe = false;
  NumericalPipeline pipeline = NumericalPipeline::kLlm;
  // The D128 attention implementation's real head-size requirement, not a
  // benchmark shape or prompt/image geometry. Other checks stay in capability.
  std::uint64_t attention_head_dim = 0;
};

std::span<const ComponentNumericalEvidence> ComponentNumericalEvidenceTable();
bool HasImageAttentionEvidence(const planner::DeviceFacts& device, std::uint64_t head_dim);
bool EvidenceDeviceMatches(const ComponentNumericalEvidence& evidence,
                           const planner::DeviceFacts& device);

}  // namespace llmp::runtime

#endif  // LLMP_RUNTIME_NUMERICAL_EVIDENCE_H_
