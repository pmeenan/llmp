// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "engine/program_lifecycle.h"

#include <algorithm>
#include <format>
#include <limits>
#include <utility>

#include "base/check.h"
#include "engine/paged_node.h"
#include "engine/support.h"

namespace llmp::engine {
namespace {
using support::Error;
bool AddSlot(std::uint64_t bytes, std::uint64_t& total) {
  constexpr auto max = std::numeric_limits<std::uint64_t>::max();
  if (bytes > max - 255) return false;
  const auto aligned = (bytes + 255) & ~std::uint64_t{255};
  if (aligned > max - total) return false;
  total += aligned;
  return true;
}
}  // namespace

std::expected<ProgramEnvelope, std::string> ProgramEnvelope::Measure(
    const kernels::StepProgram& program, bool stage_features) {
  ProgramEnvelope result{.scratch = program.scratch_bytes(),
                         .pool = program.pool_bytes(),
                         .host = program.host_bytes(),
                         .launches = program.launches()};
  for (const auto& slot : program.inputs()) {
    if (!stage_features && slot.kind == kernels::InputKind::kFeatures) continue;
    if (!AddSlot(slot.bytes, result.inputs)) return Error("program input staging overflow");
  }
  for (const auto& slot : program.outputs()) {
    if (!AddSlot(slot.bytes, result.outputs)) return Error("program output staging overflow");
  }
  return result;
}
void ProgramEnvelope::Include(const ProgramEnvelope& other) {
  scratch = std::max(scratch, other.scratch);
  pool = std::max(pool, other.pool);
  host = std::max(host, other.host);
  launches = std::max(launches, other.launches);
  inputs = std::max(inputs, other.inputs);
  outputs = std::max(outputs, other.outputs);
}

ProgramLifecycle::ProgramLifecycle(PagedNode& node, int owner, bool graphs)
    : node_(node), owner_(owner), runs_(graphs) {}
ProgramLifecycle::~ProgramLifecycle() = default;

std::expected<void, std::string> ProgramLifecycle::BindLaunch(
    std::unique_ptr<kernels::StepLaunch> launch) {
  if (launch_ || !launch) return Error("program launch is missing or already bound");
  launch_ = std::move(launch);
  runs_.SetLaunch(launch_.get());
  account_.Bind(
      [this](std::uint64_t bytes, bool required) { return node_.ChargeHost(bytes, required); },
      [this](std::uint64_t bytes) { node_.UnchargeHost(bytes); });
  return {};
}

std::expected<Coverage, std::string> ProgramLifecycle::BindPrepared(StepPlanned& planned,
                                                                    const Validate& validate) {
  if (!launch_ || !planned.program) return Error("no launch or program to bind");
  auto& program = *planned.program;
  if (!planned.bound) {
    if (auto bound = program.Bind(*launch_); !bound)
      return Error(std::format("binding a step: {}", bound.error().detail));
    planned.bound = true;
  }
  if (validate) {
    if (auto checked = validate(program); !checked) return Error(checked.error());
  }
  Coverage checked;
  CheckCoverage(node_, owner_, program.tensors(), "a step's ", checked);
  if (checked.violations != 0)
    return Error(std::format("catalog coverage: {}", checked.first_violation));
  planned.positioned = !program.capturable();
  return checked;
}
void ProgramLifecycle::ReleaseAfterDrain() {
  base::Check(account_.bytes() == 0, "program caches must retire before their launch");
  runs_.SetLaunch(nullptr);
  launch_.reset();
}
}  // namespace llmp::engine
