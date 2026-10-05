// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "runtime/gemma_profile.h"

#include <gtest/gtest.h>

#include <algorithm>

#include "gemma4_fixture.h"

namespace {
namespace rt = jitllm::runtime;
namespace md = jitllm::model;
namespace fixture = jitllm::test_support::gemma4;
TEST(GemmaServingProfile, ClosedCandidateRequiresTheCompleteApprovedBinding) {
  for (const auto size : {26U, 31U}) {
    const auto& p = size == 26 ? md::Gemma4_26BA4B() : md::Gemma4_31B();
    auto resources = fixture::Resources(size);
    auto accepted = rt::ApprovedGemmaProfile("gemma4", p.experts, resources);
    ASSERT_TRUE(accepted) << (accepted ? "" : accepted.error());
    EXPECT_EQ(*accepted, &p);
    EXPECT_FALSE(rt::ApprovedGemmaProfile("qwen4exp", p.experts, resources));
    EXPECT_FALSE(rt::ApprovedGemmaProfile("gemma4", 1, resources));
    EXPECT_FALSE(rt::ApprovedGemmaProfile("gemma4", size == 26 ? 0U : 128U, resources));
    auto missing = resources;
    missing.pop_back();
    EXPECT_FALSE(rt::ApprovedGemmaProfile("gemma4", p.experts, missing));
    auto duplicate = resources;
    duplicate.push_back(resources.front());
    EXPECT_FALSE(rt::ApprovedGemmaProfile("gemma4", p.experts, duplicate));
    auto malformed = resources;
    ++malformed.front().ne.front();
    EXPECT_FALSE(rt::ApprovedGemmaProfile("gemma4", p.experts, malformed));
  }
}
}  // namespace
