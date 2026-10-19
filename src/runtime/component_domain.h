// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The component serving build's vendor-free mechanical execution-envelope
// phase. Numerical transformation grants remain separate source facts.
#ifndef LLMP_RUNTIME_COMPONENT_DOMAIN_H_
#define LLMP_RUNTIME_COMPONENT_DOMAIN_H_

#include <cstdint>
#include <expected>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "engine/decoding.h"
#include "engine/execution_contract.h"
#include "model/batching.h"
#include "model/state_contract.h"
#include "spec/spec.h"

namespace llmp::runtime {

// The paid internal serving domain, shared by ordinary and block/replay
// compositions. The retained domain owns its allowance charge; this does not
// approve a source or replace ComponentStepSource's complete sealing checks.
std::expected<std::shared_ptr<const engine::NumericalDomain>, std::string> ComponentServingDomain(
    const spec::ModelSpec& model, std::span<const model::StateLayoutDerived> layouts,
    std::span<const model::Batching> batching, std::uint32_t slots,
    const engine::DecodingMode& mode,
    const std::shared_ptr<engine::ExecutionMetadataAllowance>& allowance,
    std::function<bool(std::uint64_t)> charge = {});

// Setup measures the real scalar/joined scheduling frontiers. An explicit
// diagnostic supplies its own additional shapes within its explicit domain;
// automatic scoring remains scalar until its complete batch has been admitted.
// This is measurement only, never numerical authority or a physical grant.
std::expected<std::vector<planner::StepShape>, std::string> ComponentSetupShapes(
    std::span<const model::StateLayoutDerived> layouts, std::span<const model::Batching> batching,
    std::uint32_t owners, std::uint32_t read_align, const engine::DecodingMode& mode,
    const engine::NumericalDomain* explicit_domain = nullptr,
    std::span<const planner::StepShape> diagnostic_shapes = {}, bool device_tokens = true);

}  // namespace llmp::runtime

#endif  // LLMP_RUNTIME_COMPONENT_DOMAIN_H_
