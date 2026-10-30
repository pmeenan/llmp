// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// A digest of the activation layouts a lowered step was given: the plan's
// logical placement (each root's offset, the extent) and the lowering's
// physical one (each physical root's offset and bytes, the extent). Tests
// pin it for whole families of shapes, so a change to placement's speed
// (F16) shows that it left every layout where it was.

#ifndef LLMP_TESTS_SUPPORT_PLACEMENT_DIGEST_H_
#define LLMP_TESTS_SUPPORT_PLACEMENT_DIGEST_H_

#include <cstdint>
#include <span>
#include <string>
#include <string_view>

#include "base/sha256.h"
#include "kernels/ggml/lower.h"
#include "planner/plan.h"

namespace llmp::test_support {

class PlacementDigest {
 public:
  // One lowered step, under a label naming its case.
  void Add(std::string_view label, const planner::StepPlan& plan,
           const kernels::ggml::LoweredStep& lowered) {
    Text(label);
    Number(plan.placement.offsets.size());
    for (const auto& [root, offset] : plan.placement.offsets) {
      Number(root.v);
      Number(offset);
    }
    Number(plan.placement.extent);
    Number(lowered.physical_roots.size());
    for (const auto& r : lowered.physical_roots) {
      Number(r.root.valid() ? r.root.v : UINT32_MAX);
      Number(r.offset);
      Number(r.bytes);
    }
    Number(lowered.extent);
    Number(lowered.logical_extent);
  }
  std::string Hex() { return base::ToHex(sha_.Finish()); }

 private:
  void Number(std::uint64_t v) { sha_.Update(std::as_bytes(std::span(&v, 1))); }
  void Text(std::string_view text) {
    Number(text.size());
    sha_.Update(text);
  }

  base::Sha256 sha_;
};

}  // namespace llmp::test_support

#endif  // LLMP_TESTS_SUPPORT_PLACEMENT_DIGEST_H_
