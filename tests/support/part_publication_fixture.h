// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
#ifndef LLMP_TESTS_SUPPORT_PART_PUBLICATION_FIXTURE_H_
#define LLMP_TESTS_SUPPORT_PART_PUBLICATION_FIXTURE_H_
#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include "import/normalize.h"

namespace llmp::test_support {
// The production header grants this named fixture friendship, but its body is
// confined to test support. No generic mutable publication view is exposed.
class PartPublicationFixture {
 public:
  static import::NormalizedPlan Make(import::IndexedPartPlan&& parts, spec::Source canonical,
                                     spec::ModelSpec spec,
                                     std::vector<import::NormalizedPartBinding> bindings,
                                     std::array<import::SemanticReceipt, 3> semantic,
                                     std::vector<import::PlannedPartFold> folds) {
    import::NormalizedPlan plan;
    plan.parts_.emplace(std::move(parts));
    plan.canonical_ = std::move(canonical);
    plan.spec_ = std::move(spec);
    plan.bindings_ = std::move(bindings);
    plan.semantic_.assign(semantic.begin(), semantic.end());
    plan.planned_folds_ = std::move(folds);
    for (const auto& fold : plan.planned_folds_) plan.canonical_.folds.push_back(fold.fold);
    std::ranges::sort(plan.canonical_.folds);
    for (const auto& tensor : plan.canonical_.tensors)
      plan.cost_.payload_bytes += format::PartsOf(tensor.format)->Find(tensor.part)->bytes;
    return plan;
  }
  static void WrongBinding(import::NormalizedPlan& plan) { ++plan.bindings_[0].output; }
  static void MissingSemantic(import::NormalizedPlan& plan) {
    plan.semantic_[1] = plan.semantic_[0];
  }
  static void WrongFoldStage(import::NormalizedPlan& plan) {
    std::erase(plan.spec_.stages[0].folds, "gdn.value_heads_tiled");
  }
  static void DuplicateFoldOutput(import::NormalizedPlan& plan) {
    plan.planned_folds_[0].outputs.push_back(plan.planned_folds_[0].outputs[0]);
  }
  static void WrongSpec(import::NormalizedPlan& plan) { plan.spec_.stages.clear(); }
  // A drafter plan of any small parts, generated parts and spec (writer
  // controls; NormalizeModelOptDrafter's own domain is pinned).
  static import::NormalizedPlan MakeDrafter(import::IndexedPartPlan&& parts, spec::Source canonical,
                                            spec::ModelSpec spec,
                                            std::vector<import::NormalizedPartBinding> bindings,
                                            std::vector<import::GeneratedPart> generated,
                                            std::string options) {
    import::NormalizedPlan plan;
    plan.parts_.emplace(std::move(parts));
    plan.canonical_ = std::move(canonical);
    plan.spec_ = std::move(spec);
    plan.bindings_ = std::move(bindings);
    plan.generated_ = std::move(generated);
    plan.options_ = std::move(options);
    plan.has_options_ = true;
    for (const auto& tensor : plan.canonical_.tensors)
      plan.cost_.payload_bytes += format::PartsOf(tensor.format)->Find(tensor.part)->bytes;
    return plan;
  }
  static void ResizeGenerated(import::NormalizedPlan& plan, std::size_t bytes) {
    plan.generated_.at(0).bytes.resize(bytes);
  }
};
}  // namespace llmp::test_support
#endif  // LLMP_TESTS_SUPPORT_PART_PUBLICATION_FIXTURE_H_
