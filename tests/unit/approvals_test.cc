// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The approved checkpoints as data (runtime/approvals.h, D-109): the table's
// own rules and its lookups, and the recipes that say what is served.

#include "runtime/approvals.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <optional>
#include <set>
#include <string>
#include <string_view>

#include "artifact/artifact.h"
#include "base/bytes.h"
#include "runtime/component_admission.h"
#include "spec/mapping.h"

namespace {
namespace rt = llmp::runtime;

bool Hex64(std::string_view text) {
  return text.size() == 64 && std::ranges::all_of(text, [](char c) {
           return std::isdigit(static_cast<unsigned char>(c)) || (c >= 'a' && c <= 'f');
         });
}

TEST(Approvals, EveryEntryIsCompleteUniqueAndPairedWithAnApprovedTarget) {
  std::set<std::string_view> names;
  for (const rt::Approval& a : rt::Approvals()) {
    SCOPED_TRACE(a.name);
    EXPECT_TRUE(names.insert(a.name).second) << "a name twice";
    EXPECT_FALSE(a.label.empty());
    const llmp::spec::Mapping* mapping = llmp::spec::FindMapping(a.architecture);
    ASSERT_NE(mapping, nullptr);
    EXPECT_EQ(mapping->role == llmp::spec::StageRole::kDrafter, a.drafter);
    EXPECT_FALSE(a.sources.empty());
    for (const rt::SourceReceipt& r : a.sources) {
      EXPECT_FALSE(r.name.empty());
      EXPECT_TRUE(Hex64(r.sha256)) << r.name;
      EXPECT_NE(r.bytes, 0U) << r.name;
    }
    EXPECT_TRUE(Hex64(a.inventory));
    EXPECT_GT(a.tensors, a.arrays);
    EXPECT_NE(a.max_context, 0U);
    EXPECT_NE(a.max_slots, 0U);
    // Only Qwen3.8's native imports are still transitional.
    if (a.spec.empty())
      EXPECT_TRUE(a.form == rt::InventoryForm::kModelOpt || a.form == rt::InventoryForm::kMlx);
    else
      EXPECT_TRUE(Hex64(a.spec));
    if (a.drafter) {
      const rt::Approval* target = rt::FindApprovalNamed(a.target);
      ASSERT_NE(target, nullptr) << a.target;
      EXPECT_FALSE(target->drafter);
      EXPECT_FALSE(a.spec.empty()) << "a stored drafter's composition is always pinned";
    } else {
      EXPECT_TRUE(a.target.empty());
    }
  }
}

TEST(Approvals, LookupNeedsTheWholeSourceIdentity) {
  const rt::Approval* gemma2 = rt::FindApprovalNamed("gemma-2-2b-it-q8_0");
  ASSERT_NE(gemma2, nullptr);
  llmp::artifact::Source source{.name = std::string(gemma2->sources[0].name),
                                .bytes = llmp::base::Bytes(gemma2->sources[0].bytes)};
  for (std::size_t i = 0; i < source.sha256.size(); ++i)
    source.sha256[i] = static_cast<std::uint8_t>(
        std::stoul(std::string(gemma2->sources[0].sha256.substr(2 * i, 2)), nullptr, 16));
  std::array<llmp::artifact::Source, 1> sources{source};
  EXPECT_EQ(rt::FindApproval("gemma2", sources), gemma2);
  EXPECT_EQ(rt::FindApproval("gemma3", sources), nullptr);
  sources[0].sha256[0] ^= 1;
  EXPECT_EQ(rt::FindApproval("gemma2", sources), nullptr);
  sources[0] = source;
  sources[0].bytes = llmp::base::Bytes(source.bytes.value() + 1);
  EXPECT_EQ(rt::FindApproval("gemma2", sources), nullptr);
  EXPECT_EQ(rt::FindApproval("gemma2", std::span<const llmp::artifact::Source>{}), nullptr);
}

TEST(Approvals, EveryServedArchitectureHasARecipeAndEveryDrafterADraftingOne) {
  for (const llmp::spec::Mapping* m : llmp::spec::Mappings()) {
    SCOPED_TRACE(m->architecture);
    if (m->role == llmp::spec::StageRole::kDrafter) {
      EXPECT_EQ(m->serving, nullptr);
      ASSERT_NE(m->draft, nullptr);
      if (m->draft->shape == llmp::spec::DraftRecipe::Shape::kBlock) {
        EXPECT_FALSE(m->draft->depth_key.empty());
        EXPECT_FALSE(m->draft->fill_key.empty());
      }
      if (m->draft->shape == llmp::spec::DraftRecipe::Shape::kAnchored)
        EXPECT_NE(m->draft->depth, 0U);
    } else if (m->architecture != "qwen2") {
      ASSERT_NE(m->serving, nullptr);
      EXPECT_NE(m->serving->prefill_rows, 0U);
      EXPECT_FALSE(m->serving->name.empty());
      // A shared route names exactly one qualification; others name none.
      if (m->serving->shared_route)
        EXPECT_NE(m->serving->qualified_envelopes, m->serving->closed_bridge);
      else
        EXPECT_FALSE(m->serving->qualified_envelopes || m->serving->closed_bridge);
    }
    EXPECT_EQ(rt::OnComponentEngine(m->architecture), m->architecture != "qwen2");
  }
}

// The chat route's slices are each recipe's, whatever template renders:
// Gemma 4's thought channel and tools off with their openers ending a reply,
// DeepSeek V4 thinking by default, everyone else's whole template.
TEST(Approvals, ChatSlicesAreTheRecipes) {
  const auto* gemma4 = llmp::spec::FindMapping("gemma4")->serving;
  EXPECT_FALSE(gemma4->chat_thinking);
  EXPECT_FALSE(gemma4->chat_tools);
  EXPECT_EQ(gemma4->chat_stops.size(), 2U);
  EXPECT_FALSE(gemma4->default_thinking.has_value());
  const auto* dsv4 = llmp::spec::FindMapping("deepseek4")->serving;
  EXPECT_TRUE(dsv4->chat_thinking && dsv4->chat_tools && dsv4->chat_stops.empty());
  EXPECT_EQ(dsv4->default_thinking, std::optional<bool>(true));
  for (const std::string_view arch : {"gemma2", "gemma3", "qwen4exp"}) {
    const auto* r = llmp::spec::FindMapping(arch)->serving;
    EXPECT_TRUE(r->chat_thinking && r->chat_tools && r->chat_stops.empty()) << arch;
    EXPECT_FALSE(r->default_thinking.has_value()) << arch;
  }
}

}  // namespace
