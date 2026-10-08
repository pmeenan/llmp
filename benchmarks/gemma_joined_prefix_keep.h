// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
#ifndef LLMP_BENCHMARKS_GEMMA_JOINED_PREFIX_KEEP_H_
#define LLMP_BENCHMARKS_GEMMA_JOINED_PREFIX_KEEP_H_
#include "engine/gemma4_plan.h"
using GemmaPrefixPlanResult =
    std::expected<std::unique_ptr<llmp::engine::Gemma4Planned>, std::string>;
GemmaPrefixPlanResult
PrefixRealPlan(const llmp::engine::Gemma4Model&, const llmp::kernels::ggml::Gemma4ChunkShape&, const llmp::kernels::ggml::DeviceChoices&, std::uint64_t, std::uint64_t, std::span<const std::string>) asm(
    "__real__ZN4llmp6engine15PlanGemma4ChunkERKNS0_11Gemma4ModelERKNS_7kernels4ggml"
    "16Gemma4ChunkShapeERKNS5_13DeviceChoicesEmmSt4spanIKNSt7__cxx1112basic_stringIc"
    "St11char_traitsIcESaIcEEELm18446744073709551615EE");
GemmaPrefixPlanResult
PrefixWrapPlan(const llmp::engine::Gemma4Model&, const llmp::kernels::ggml::Gemma4ChunkShape&, const llmp::kernels::ggml::DeviceChoices&, std::uint64_t, std::uint64_t, std::span<const std::string>) asm(
    "__wrap__ZN4llmp6engine15PlanGemma4ChunkERKNS0_11Gemma4ModelERKNS_7kernels4ggml"
    "16Gemma4ChunkShapeERKNS5_13DeviceChoicesEmmSt4spanIKNSt7__cxx1112basic_stringIc"
    "St11char_traitsIcESaIcEEELm18446744073709551615EE");
#endif  // LLMP_BENCHMARKS_GEMMA_JOINED_PREFIX_KEEP_H_
