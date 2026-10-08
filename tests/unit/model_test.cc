// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The model layer's shape contracts (D-068) in every build profile: state
// representations, their capabilities and a request's live state
// (model/state.h), and model contexts composed of components
// (model/context.h). The scenarios that drive them through admission and
// the catalog are in shapes_test.cc.

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <expected>
#include <optional>
#include <vector>

#include "base/bytes.h"
#include "base/sha256.h"
#include "catalog/catalog.h"
#include "expected_error.h"
#include "model/context.h"
#include "model/state.h"

namespace {

using llmp::base::Bytes;
using llmp::base::Sha256Digest;
using llmp::catalog::Catalog;
using llmp::catalog::MemoryClass;
using llmp::catalog::Recovery;
using llmp::catalog::ResourceId;
using llmp::model::ComponentRole;
using llmp::model::ContextError;
using llmp::model::ModelContext;
using llmp::model::StateCapability;
using llmp::model::StateCursor;
using llmp::model::StateError;
using llmp::model::StateRepresentation;
using llmp::test_support::Failed;
using ::testing::ElementsAre;
using ::testing::IsEmpty;

// A paged KV cache: 16 positions per 1000-byte block, append and truncate.
StateRepresentation Paged() {
  return {.name = "kv",
          .block_positions = 16,
          .block_bytes = Bytes(1000),
          .capabilities = StateCapability::kAppend | StateCapability::kTruncate,
          .max_snapshots = 0,
          .snapshot_bytes = {}};
}

// Recurrent state: one fixed block, truncated only through snapshots.
StateRepresentation Recurrent(std::uint32_t snapshots) {
  return {.name = "recurrent",
          .block_positions = 0,
          .block_bytes = Bytes(4000),
          .capabilities = StateCapability::kAppend | StateCapability::kTruncateAtSnapshot,
          .max_snapshots = snapshots,
          .snapshot_bytes = Bytes(4000)};
}

TEST(StateCursorTest, AllowanceCoversTheBoundInWholeBlocks) {
  const StateCursor kv = StateCursor::Create(Paged(), 40).value();
  EXPECT_EQ(kv.allowance(), Bytes(3000));  // 40 positions: three blocks
  EXPECT_EQ(kv.blocks(), 0U);
  const StateCursor recurrent = StateCursor::Create(Recurrent(5), 1000).value();
  EXPECT_EQ(recurrent.allowance(), Bytes(4000 + (5 * 4000)));  // the state and five snapshots
  EXPECT_EQ(recurrent.blocks(), 1U);
  // Snapshots without their bytes, or without the capability, are malformed.
  StateRepresentation bad = Recurrent(2);
  bad.snapshot_bytes = Bytes();
  EXPECT_EQ(Failed(StateCursor::Create(bad, 10)), StateError::kInvalid);
  bad = Paged();
  bad.max_snapshots = 1;
  bad.snapshot_bytes = Bytes(1);
  EXPECT_EQ(Failed(StateCursor::Create(bad, 10)), StateError::kInvalid);
  EXPECT_EQ(Failed(StateCursor::Create(Paged(), UINT64_MAX)), StateError::kExceedsBound);
}

// A draft of four is verified (five positions written tentatively), two
// drafts are accepted with the verify's own token, and the rest rolled
// back. Rollback never goes below the committed prefix, and an append past
// the bound is refused before anything is allocated.
TEST(StateCursorTest, RejectedDraftRollsBackToTheCommittedPrefix) {
  StateCursor kv = StateCursor::Create(Paged(), 50).value();
  ASSERT_TRUE(kv.Append(30).has_value());
  ASSERT_TRUE(kv.Commit(30).has_value());
  ASSERT_TRUE(kv.Append(5).has_value());
  EXPECT_EQ(kv.written(), 35U);
  EXPECT_EQ(kv.blocks(), 3U);
  ASSERT_TRUE(kv.Commit(3).has_value());
  EXPECT_EQ(Failed(kv.Truncate(32)), StateError::kBelowCommitted);
  EXPECT_EQ(Failed(kv.Truncate(36)), StateError::kBeyondWritten);
  ASSERT_TRUE(kv.Truncate(33).has_value());
  EXPECT_EQ(kv.committed(), 33U);
  EXPECT_EQ(kv.written(), 33U);
  EXPECT_EQ(kv.blocks(), 3U);
  EXPECT_EQ(Failed(kv.Commit(1)), StateError::kBeyondWritten);
  EXPECT_EQ(Failed(kv.Append(18)), StateError::kExceedsBound);
  EXPECT_EQ(kv.written(), 33U);
  ASSERT_TRUE(kv.Append(17).has_value());
  EXPECT_EQ(kv.room(), 0U);
  // Truncating back across a block boundary frees the tail block.
  ASSERT_TRUE(kv.Truncate(40).has_value());
  EXPECT_EQ(kv.blocks(), 3U);
  ASSERT_TRUE(kv.Truncate(33).has_value());
  EXPECT_EQ(kv.blocks(), 3U);
}

TEST(StateCursorTest, SnapshotStateTruncatesOnlyToASnapshot) {
  StateCursor state = StateCursor::Create(Recurrent(3), 100).value();
  ASSERT_TRUE(state.Append(10).has_value());
  ASSERT_TRUE(state.Commit(10).has_value());
  ASSERT_TRUE(state.Snapshot().has_value());  // at the committed prefix
  for (int i = 0; i < 2; ++i) {
    ASSERT_TRUE(state.Append(1).has_value());
    ASSERT_TRUE(state.Snapshot().has_value());
  }
  ASSERT_TRUE(state.Append(1).has_value());
  EXPECT_EQ(Failed(state.Snapshot()), StateError::kSnapshotsFull);
  EXPECT_THAT(state.snapshots(), ElementsAre(10, 11, 12));
  ASSERT_TRUE(state.Commit(1).has_value());  // one draft accepted
  EXPECT_THAT(state.snapshots(), ElementsAre(11, 12));
  EXPECT_EQ(Failed(state.Truncate(10)), StateError::kBelowCommitted);
  ASSERT_TRUE(state.Truncate(12).has_value());
  EXPECT_THAT(state.snapshots(), ElementsAre(11, 12));
  ASSERT_TRUE(state.Truncate(11).has_value());
  EXPECT_THAT(state.snapshots(), ElementsAre(11));
  ASSERT_TRUE(state.Append(1).has_value());
  // Twelve holds no snapshot any more.
  StateCursor no_snapshot = state;
  ASSERT_TRUE(no_snapshot.Append(1).has_value());
  EXPECT_EQ(Failed(no_snapshot.Truncate(12)), StateError::kNoSnapshot);

  // A representation that cannot truncate, or append, says so.
  StateRepresentation fixed = Paged();
  fixed.capabilities = static_cast<std::uint8_t>(StateCapability::kAppend);
  StateCursor append_only = StateCursor::Create(fixed, 10).value();
  ASSERT_TRUE(append_only.Append(2).has_value());
  EXPECT_EQ(Failed(append_only.Truncate(1)), StateError::kUnsupported);
  EXPECT_EQ(Failed(append_only.Snapshot()), StateError::kUnsupported);
  fixed.capabilities = static_cast<std::uint8_t>(StateCapability::kTruncate);
  EXPECT_EQ(Failed(StateCursor::Create(fixed, 10).value().Append(1)), StateError::kUnsupported);
}

Sha256Digest Artifact(std::uint8_t tag) {
  Sha256Digest digest{};
  digest[0] = tag;
  return digest;
}

// A target (embeddings, trunk, head) and a companion drafter that shares
// the target's embeddings: a closure over both holds them once.
TEST(ModelContextTest, SharedResourcesAreChargedOnce) {
  Catalog catalog;
  const auto spark = catalog.AddDomain("spark");
  const auto resource = [&](std::uint64_t bytes) {
    const auto extent = catalog
                            .AddExtent({.domain = spark,
                                        .memory_class = MemoryClass::kWeights,
                                        .recovery = Recovery::kFromArtifact,
                                        .size = Bytes(bytes),
                                        .content = {}})
                            .value();
    const std::array ranges = {
        llmp::catalog::Range{.extent = extent, .offset = Bytes(), .length = Bytes(bytes)}};
    return catalog.AddResource(ranges).value();
  };
  const ResourceId embeddings = resource(400);
  const ResourceId trunk = resource(2000);
  const ResourceId head = resource(400);
  const ResourceId layers = resource(300);
  const auto context = ModelContext::Create({{.role = ComponentRole::kMain,
                                              .artifact = Artifact(1),
                                              .resources = {embeddings, trunk, head}},
                                             {.role = ComponentRole::kDrafter,
                                              .artifact = Artifact(2),
                                              .resources = {layers, embeddings}}})
                           .value();
  const std::array both = {ComponentRole::kMain, ComponentRole::kDrafter};
  const auto closure = context.ClosureOf(catalog, both).value();
  EXPECT_EQ(closure.extents.size(), 4U);
  EXPECT_EQ(closure.bytes_by_domain.at(spark), Bytes(3100));
  EXPECT_EQ(context.SharedBytes(catalog, spark).value(), Bytes(400));
  const std::array drafter = {ComponentRole::kDrafter};
  EXPECT_EQ(context.ClosureOf(catalog, drafter).value().bytes_by_domain.at(spark), Bytes(700));
  const std::array encoder = {ComponentRole::kEncoder};
  EXPECT_THAT(context.ResourcesOf(encoder), IsEmpty());

  // The identity covers every component, whatever order they are listed
  // in; another drafter artifact is another context.
  const auto reordered = ModelContext::Create({{.role = ComponentRole::kDrafter,
                                                .artifact = Artifact(2),
                                                .resources = {layers, embeddings}},
                                               {.role = ComponentRole::kMain,
                                                .artifact = Artifact(1),
                                                .resources = {embeddings, trunk, head}}})
                             .value();
  EXPECT_EQ(reordered.identity(), context.identity());
  const auto other = ModelContext::Create({{.role = ComponentRole::kMain,
                                            .artifact = Artifact(1),
                                            .resources = {embeddings, trunk, head}},
                                           {.role = ComponentRole::kDrafter,
                                            .artifact = Artifact(3),
                                            .resources = {layers, embeddings}}})
                         .value();
  EXPECT_NE(other.identity(), context.identity());
  const auto alone = ModelContext::Create({{.role = ComponentRole::kMain,
                                            .artifact = Artifact(1),
                                            .resources = {embeddings, trunk, head}}})
                         .value();
  EXPECT_NE(alone.identity(), context.identity());
}

TEST(ModelContextTest, RefusesMalformedCompositions) {
  const ResourceId some(0, 1);
  EXPECT_EQ(Failed(ModelContext::Create({})), ContextError::kNoMain);
  EXPECT_EQ(Failed(ModelContext::Create(
                {{.role = ComponentRole::kDrafter, .artifact = Artifact(1), .resources = {some}}})),
            ContextError::kNoMain);
  EXPECT_EQ(Failed(ModelContext::Create(
                {{.role = ComponentRole::kMain, .artifact = Artifact(1), .resources = {some}},
                 {.role = ComponentRole::kMain, .artifact = Artifact(2), .resources = {some}}})),
            ContextError::kNoMain);
  EXPECT_EQ(Failed(ModelContext::Create(
                {{.role = ComponentRole::kMain, .artifact = Artifact(1), .resources = {}}})),
            ContextError::kEmpty);
  EXPECT_EQ(
      Failed(ModelContext::Create(
          {{.role = ComponentRole::kMain, .artifact = Artifact(1), .resources = {ResourceId{}}}})),
      ContextError::kBadResource);
  // A stored drafter is a second role of the main artifact; the same role
  // twice is not.
  EXPECT_TRUE(ModelContext::Create(
                  {{.role = ComponentRole::kMain, .artifact = Artifact(1), .resources = {some}},
                   {.role = ComponentRole::kDrafter, .artifact = Artifact(1), .resources = {some}}})
                  .has_value());
  EXPECT_EQ(Failed(ModelContext::Create(
                {{.role = ComponentRole::kMain, .artifact = Artifact(1), .resources = {some}},
                 {.role = ComponentRole::kDrafter, .artifact = Artifact(2), .resources = {some}},
                 {.role = ComponentRole::kDrafter, .artifact = Artifact(2), .resources = {some}}})),
            ContextError::kDuplicate);
}

}  // namespace
