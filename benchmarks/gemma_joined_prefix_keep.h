// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0
#ifndef JITLLM_BENCHMARKS_GEMMA_JOINED_PREFIX_KEEP_H_
#define JITLLM_BENCHMARKS_GEMMA_JOINED_PREFIX_KEEP_H_
#include "engine/gemma4_plan.h"
using GemmaPrefixPlanResult =
    std::expected<std::unique_ptr<jitllm::engine::Gemma4Planned>, std::string>;
GemmaPrefixPlanResult
PrefixRealPlan(const jitllm::engine::Gemma4Model&, const jitllm::kernels::ggml::Gemma4ChunkShape&, const jitllm::kernels::ggml::DeviceChoices&, std::uint64_t, std::uint64_t, std::span<const std::string>) asm(
    "__real__ZN6jitllm6engine15PlanGemma4ChunkERKNS0_11Gemma4ModelERKNS_7kernels4ggml"
    "16Gemma4ChunkShapeERKNS5_13DeviceChoicesEmmSt4spanIKNSt7__cxx1112basic_stringIc"
    "St11char_traitsIcESaIcEEELm18446744073709551615EE");
GemmaPrefixPlanResult
PrefixWrapPlan(const jitllm::engine::Gemma4Model&, const jitllm::kernels::ggml::Gemma4ChunkShape&, const jitllm::kernels::ggml::DeviceChoices&, std::uint64_t, std::uint64_t, std::span<const std::string>) asm(
    "__wrap__ZN6jitllm6engine15PlanGemma4ChunkERKNS0_11Gemma4ModelERKNS_7kernels4ggml"
    "16Gemma4ChunkShapeERKNS5_13DeviceChoicesEmmSt4spanIKNSt7__cxx1112basic_stringIc"
    "St11char_traitsIcESaIcEEELm18446744073709551615EE");
#endif  // JITLLM_BENCHMARKS_GEMMA_JOINED_PREFIX_KEEP_H_
