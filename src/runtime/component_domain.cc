// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "runtime/component_domain.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "engine/request_cohort.h"
#include "engine/support.h"
#include "runtime/llm_policy.h"

namespace llmp::runtime {
namespace {
using engine::support::Error;
struct ConstructionCharge {
  std::shared_ptr<engine::ExecutionMetadataAllowance> allowance;
  std::uint64_t bytes = 0;
  ~ConstructionCharge() { allowance->Release(bytes); }
};
// Domain data and its existing node allowance have one complete lifetime.
struct DerivedDomain {
  std::shared_ptr<ConstructionCharge> charge;
  engine::NumericalDomain domain;
};
}  // namespace

std::expected<std::vector<planner::StepShape>, std::string> ComponentSetupShapes(
    std::span<const model::StateLayoutDerived> layouts, std::span<const model::Batching> batching,
    std::uint32_t owners, std::uint32_t read_align, const engine::DecodingMode& mode,
    const engine::NumericalDomain* explicit_domain,
    std::span<const planner::StepShape> diagnostic_shapes, bool device_tokens) {
  if (layouts.empty() || layouts.size() != batching.size() || owners == 0 ||
      owners > engine::kMaxRequestSlots || read_align == 0 || read_align % 256 != 0)
    return Error("setup measurement needs the actual stage and owner capacities");
  if (explicit_domain) {
    if (auto valid = ValidateServingEnvelope(*explicit_domain, layouts, batching, owners); !valid)
      return Error(valid.error());
  }
  for (std::size_t i = 0; i < layouts.size(); ++i)
    if (layouts[i].context == 0 || layouts[i].max_rows == 0 ||
        layouts[i].max_rows > layouts[i].context || batching[i].max_segments == 0 ||
        batching[i].max_segments > engine::kMaxRequestSlots)
      return Error("setup measurement lacks an actual component row capacity");
  std::uint64_t verify = mode.drafters.empty() ? 0 : 1;
  for (const auto& draft : mode.drafters) {
    if (draft.stage == 0 || draft.stage >= layouts.size() || draft.max_depth == 0)
      return Error("setup measurement has an invalid drafter stage");
    if (mode.combine == engine::DecodingMode::Drafters::kCascade)
      verify += draft.max_depth;
    else
      verify = std::max(verify, std::uint64_t{draft.max_depth} + 1);
    if (verify > UINT32_MAX) return Error("setup verify rows overflow");
  }
  // Reserve the actual representation bound before filling: two owner
  // frontiers, one prompt/two output modes each, then complete draft frontiers.
  // The ordinary construction payment owns this schedule inventory. Explicit
  // diagnostic entries add exactly their separately prepaid vector capacity.
  std::size_t capacity = 6 + (mode.drafters.empty() ? 0U : 4U);
  for (const auto& draft : mode.drafters)
    if (__builtin_add_overflow(capacity, std::size_t{4} + (draft.catch_up ? 8U : 0U), &capacity))
      return Error("setup shape capacity overflows");
  if (__builtin_add_overflow(capacity, diagnostic_shapes.size(), &capacity))
    return Error("diagnostic setup shape capacity overflows");
  std::vector<planner::StepShape> shapes;
  shapes.reserve(capacity);
  const auto add = [&](std::uint16_t stage, planner::Phase phase, std::uint32_t wanted_owners,
                       std::uint32_t rows, std::uint32_t outputs,
                       bool greedy) -> std::expected<void, std::string> {
    const auto& state = layouts[stage];
    const auto& caps = batching[stage];
    if (rows == 0 || rows > state.max_rows ||
        (caps.max_segment_rows != 0 && rows > caps.max_segment_rows) ||
        (caps.max_step_rows != 0 && rows > caps.max_step_rows))
      return Error("a complete setup unit exceeds its actual component row capacity");
    auto width = std::min(wanted_owners, caps.max_segments);
    if (caps.max_step_rows != 0) width = std::min(width, caps.max_step_rows / rows);
    std::vector<engine::StageReads> reads;
    reads.reserve(layouts.size());
    for (const auto& layout : layouts)
      reads.push_back({.context = layout.context,
                       .read_align = read_align,
                       .frozen = model::ReadsFrozen(layout)});
    auto composer = engine::WaveComposer::Make(caps, std::move(reads));
    if (!composer) return Error(composer.error());
    std::vector<engine::PendingWork> work;
    work.reserve(width);
    for (std::uint32_t i = 0; i < width; ++i)
      work.push_back({.slot = i,
                      .stage = stage,
                      .phase = phase,
                      .past = state.context - rows,
                      .rows = rows,
                      .outputs = outputs,
                      .want_head = outputs != 0,
                      .greedy = greedy});
    auto composed = composer->Compose(work);
    if (!composed.refused.empty()) return Error(composed.refused.front().second);
    for (auto& step : composed.steps) {
      if (explicit_domain)
        if (auto checked = explicit_domain->CheckShape(step.shape); !checked)
          return Error(checked.error());
      if (std::ranges::find(shapes, step.shape) == shapes.end())
        shapes.push_back(std::move(step.shape));
    }
    return {};
  };
  const auto chunk = [&] {
    auto rows = layouts.front().max_rows;
    if (batching.front().max_segment_rows != 0)
      rows = std::min(rows, batching.front().max_segment_rows);
    if (batching.front().max_step_rows != 0) rows = std::min(rows, batching.front().max_step_rows);
    return rows;
  }();
  for (const auto width : {1U, std::min(owners, batching.front().max_segments)}) {
    if (width == 1 || batching.front().join_prefill)
      if (auto result = add(0, planner::Phase::kPrefill, width, chunk, 1, false); !result)
        return Error(result.error());
    for (const bool greedy : {false, true}) {
      if (greedy && !device_tokens) continue;
      if (auto result = add(0, planner::Phase::kDecode, width, 1, 1, greedy); !result)
        return Error(result.error());
      if (verify != 0) {
        const auto rows = static_cast<std::uint32_t>(verify);
        if (auto result = add(0, planner::Phase::kVerify, width, rows, rows, greedy); !result)
          return Error(result.error());
      }
      for (const auto& draft : mode.drafters) {
        const bool block = draft.shape == engine::DrafterSpec::Shape::kBlock;
        const auto rows = block ? draft.max_depth : 1U;
        if (auto result = add(draft.stage, planner::Phase::kDraft, width, rows, block ? rows : 1U,
                              draft.device_tokens && greedy);
            !result)
          return Error(result.error());
        if (draft.catch_up) {
          for (const auto caught :
               {static_cast<std::uint32_t>(verify), layouts[draft.stage].max_rows}) {
            for (const bool head : {false, true})
              if (auto result = add(draft.stage, planner::Phase::kDraft, width, caught,
                                    head ? 1U : 0U, head && draft.device_tokens);
                  !result)
                return Error(result.error());
          }
        }
      }
    }
  }
  if (!diagnostic_shapes.empty() && explicit_domain == nullptr)
    return Error("diagnostic setup shapes require their explicit execution domain");
  for (const auto& shape : diagnostic_shapes) {
    if (auto checked = explicit_domain->CheckShape(shape); !checked) return Error(checked.error());
    const auto& state = layouts[shape.stage];
    const auto& caps = batching[shape.stage];
    if (shape.read_align != read_align ||
        shape.segments.size() > std::min(owners, caps.max_segments) ||
        (caps.max_step_rows != 0 && shape.rows() > caps.max_step_rows))
      return Error("diagnostic setup exceeds the actual owner, read or total row capacity");
    for (const auto& segment : shape.segments)
      if (segment.slot >= owners || segment.rows > state.max_rows ||
          (caps.max_segment_rows != 0 && segment.rows > caps.max_segment_rows) ||
          segment.past > state.context || segment.rows > state.context - segment.past ||
          segment.read_cells > state.context || segment.read_cells < segment.past + segment.rows ||
          segment.outputs > segment.rows || segment.want_head != (segment.outputs != 0) ||
          (segment.greedy && !device_tokens))
        return Error("diagnostic setup exceeds actual state, output or backend capacity");
    const auto group_limit = shape.row_invariant && caps.invariant_group_segments != 0
                                 ? caps.invariant_group_segments
                                 : caps.group_segments;
    std::uint32_t group_owners = 0;
    std::uint64_t group_rows = 0;
    std::uint8_t group = 0;
    for (const auto& segment : shape.segments) {
      if (segment.group != group) {
        if (group_owners == 0 || segment.group != group + 1)
          return Error("diagnostic setup has invalid joined group order");
        group = segment.group;
        group_rows = group_owners = 0;
      }
      group_rows += segment.rows;
      ++group_owners;
      if ((group_limit != 0 && group_owners > group_limit) ||
          (caps.group_rows != 0 && group_owners > 1 && group_rows > caps.group_rows))
        return Error("diagnostic setup exceeds the actual joined group capacity");
    }
    if (std::ranges::find(shapes, shape) == shapes.end()) shapes.push_back(shape);
  }
  return shapes;
}

std::expected<std::shared_ptr<const engine::NumericalDomain>, std::string> ComponentServingDomain(
    const spec::ModelSpec& model_spec, std::span<const model::StateLayoutDerived> layouts,
    std::span<const model::Batching> batching, std::uint32_t slots,
    const engine::DecodingMode& mode,
    const std::shared_ptr<engine::ExecutionMetadataAllowance>& allowance,
    std::function<bool(std::uint64_t)> charge_metadata) {
  const bool composed = model_spec.stages.size() > 1;
  std::uint64_t verify = composed ? 1 : 0;
  for (const auto& draft : mode.drafters) {
    if (mode.combine == engine::DecodingMode::Drafters::kCascade)
      verify += draft.max_depth;
    else
      verify = std::max(verify, std::uint64_t{draft.max_depth} + 1);
    if (verify > UINT32_MAX) return Error("serving verify row bound overflows");
  }
  const auto verify_rows = static_cast<std::uint32_t>(verify);
  if (model_spec.pipeline != spec::PipelineKind::kLlm || model_spec.stages.empty() ||
      model_spec.stages.size() > std::uint64_t{UINT16_MAX} + 1 ||
      model_spec.stages.size() != layouts.size() || batching.size() != layouts.size() ||
      slots == 0 || slots > engine::kMaxRequestSlots ||
      (composed ? mode.kind != engine::DecodingMode::Kind::kSpeculative || verify_rows == 0 ||
                      mode.drafters.size() != layouts.size() - 1
                : mode.kind != engine::DecodingMode::Kind::kPlain || !mode.drafters.empty()) ||
      !allowance)
    return Error("a serving domain needs its complete state and owner envelope");
  std::uint64_t bytes = sizeof(DerivedDomain) + sizeof(ConstructionCharge) + 512;
  for (std::size_t i = 0; i < layouts.size(); ++i) {
    const auto& stage = model_spec.stages[i];
    if ((i == 0 ? stage.role != spec::StageRole::kTarget
                : stage.role != spec::StageRole::kDrafter) ||
        layouts[i].max_rows == 0 || (i == 0 && verify_rows > layouts[i].max_rows))
      return Error("a serving domain needs the actual target and drafter bounds");
    if (i != 0) {
      const auto& draft = mode.drafters[i - 1];
      if (draft.stage != i || draft.max_depth == 0 || draft.shared_wave_depth > draft.max_depth ||
          (draft.shared_wave_depth != 0 &&
           draft.shape != engine::DrafterSpec::Shape::kAutoregressive) ||
          verify_rows > layouts[i].max_rows ||
          (!layouts[i].borrowed.empty() && !draft.borrows_target) ||
          (draft.features && model::FeatureTap(layouts[i]) == nullptr))
        return Error("a serving domain disagrees with the actual drafter schedule");
    }
    // Input vectors, canonical domain, the retained shared owner and the
    // maximum temporary moved representation; no graph or tensor payload.
    const auto stage_bytes = sizeof(engine::NumericalStage) + stage.name.size() + 1 +
                             6 * sizeof(engine::NumericalUse) + 5 * sizeof(engine::NumericalPhase);
    std::uint64_t copy_bytes = 0;
    if (__builtin_mul_overflow(std::uint64_t{4}, static_cast<std::uint64_t>(stage_bytes),
                               &copy_bytes) ||
        __builtin_add_overflow(bytes, copy_bytes, &bytes))
      return Error("serving domain construction bound overflows");
  }
  if (!(charge_metadata ? charge_metadata(bytes) : allowance->Charge(bytes)))
    return Error("serving domain metadata allowance is insufficient");
  auto charge = std::make_shared<ConstructionCharge>();
  charge->allowance = allowance;
  charge->bytes = bytes;
  std::vector<engine::NumericalStage> stages;
  stages.reserve(layouts.size());
  for (std::size_t i = 0; i < layouts.size(); ++i) {
    engine::NumericalStage stage{
        .stage = static_cast<std::uint16_t>(i),
        .name = model_spec.stages[i].name,
        .uses = {engine::NumericalUse::kPrompt, engine::NumericalUse::kDecode,
                 engine::NumericalUse::kScore},
        .phases = {}};
    if (composed) {
      stage.uses.push_back(engine::NumericalUse::kSpeculative);
    }
    stage.phases.reserve(i == 0 ? (composed ? 5 : 3) : 1);
    const auto add = [&](planner::Phase phase, std::uint32_t rows) {
      if (batching[i].max_segment_rows != 0) rows = std::min(rows, batching[i].max_segment_rows);
      const auto owners = std::min(slots, batching[i].max_segments);
      auto total = std::uint64_t{owners} * rows;
      if (batching[i].max_step_rows != 0)
        total = std::min(total, std::uint64_t{batching[i].max_step_rows});
      stage.phases.push_back(
          {.phase = phase, .max_owners = owners, .max_owner_rows = rows, .max_total_rows = total});
    };
    if (i == 0) {
      add(planner::Phase::kPrefill, layouts[i].max_rows);
      add(planner::Phase::kDecode, 1);
      if (composed) {
        add(planner::Phase::kVerify, verify_rows);
      }
      add(planner::Phase::kScore, layouts[i].max_rows);
      if (composed) add(planner::Phase::kCommit, verify_rows);
    } else {
      const auto& draft = mode.drafters[i - 1];
      add(planner::Phase::kDraft, draft.shape == engine::DrafterSpec::Shape::kBlock
                                      ? draft.max_depth
                                      : layouts[i].max_rows);
    }
    stages.push_back(std::move(stage));
  }
  auto domain = engine::NumericalDomain::Create(std::move(stages));
  if (!domain) return Error(domain.error());
  if (auto checked = ValidateServingEnvelope(*domain, layouts, batching, slots); !checked)
    return Error(checked.error());
  auto owner = std::make_shared<DerivedDomain>(
      DerivedDomain{.charge = std::move(charge), .domain = std::move(*domain)});
  return std::shared_ptr<const engine::NumericalDomain>(owner, &owner->domain);
}

}  // namespace llmp::runtime
