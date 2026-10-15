// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
#ifndef LLMP_TEST_SUPPORT_DSV4_SCREEN_SOURCE_H_
#define LLMP_TEST_SUPPORT_DSV4_SCREEN_SOURCE_H_
#include "engine/component_source.h"
namespace llmp::test_support::dsv4_screen {
// Private numerical experiment. Only its fixed one-owner plain script can
// construct this source; no runtime registry, keeps, restores or drafters.
std::expected<components::StepGraph, std::string> RoundedRecipe(
    components::StepGraph step, const planner::StepShape& shape,
    const model::StateLayoutDerived& layout);
// Private transfer of the legacy HCA mask's ordinary sparse MMA selection.
std::expected<void, std::string> SelectHcaRecipe(planner::StepPlan& plan,
                                                 const components::StepGraph& step,
                                                 const planner::StepShape& shape,
                                                 const model::StateLayoutDerived& layout,
                                                 const execution::Registry& registry,
                                                 const planner::DeviceFacts& device);
// The current solo serving builder takes its fused HC-pre producer only
// through eight rows. This private screen preserves that recipe choice.
planner::Policy ServedPolicy(const planner::Policy& policy, std::uint32_t rows);
class Source final : public engine::StepSource {
 public:
  static std::expected<std::unique_ptr<Source>, std::string> Create(
      engine::ComponentSourceOptions options, std::vector<planner::StepShape> script);
  planner::PlanKey KeyBase() const override;
  std::vector<std::uint64_t> ShapeParameters(const planner::StepShape& shape) const override;
  std::expected<std::unique_ptr<kernels::StepProgram>, std::string> Build(
      const planner::StepShape& shape, const kernels::Addresses& addresses) override;
  bool UsesSideRows(std::uint16_t stage) const override { return ordinary_->UsesSideRows(stage); }
  engine::Status SideRows(std::uint16_t stage, std::uint16_t component,
                          std::span<const std::int32_t> history, std::uint32_t position,
                          std::span<std::int32_t> out) const override {
    return ordinary_->SideRows(stage, component, history, position, out);
  }
  std::uint64_t selected_masks() const { return selected_masks_; }

 private:
  engine::ComponentSourceOptions options_;
  std::unique_ptr<engine::ComponentStepSource> ordinary_;
  std::vector<planner::StepShape> script_;
  std::uint64_t selected_masks_ = 0;
  planner::StateSpans StateSpans(const planner::StepShape&, const kernels::Addresses&) const;
};
}  // namespace llmp::test_support::dsv4_screen
#endif
