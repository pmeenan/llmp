// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The locked ExLlamaV3 headers' sizes for the GEMM kernels' device context
// (every profile; docs/backend-proof.md, "Extension state"): upstream
// allocates a 4,202,760-byte lock area and a 16 MiB workspace per device
// (exl3_devctx.cu, not kept). Their sum is the P0 declarations' limit on
// native's persistent library workspaces (20,979,976 B), so a new pin that
// changes them fails here. This header is the component's only use in a
// profile without CUDA.

#include <gtest/gtest.h>

#include <cstddef>

#include "quant/exl3_devctx.cuh"

namespace {

TEST(Exl3ContextTest, LockAreaAndWorkspaceSizes) {
  // The lock slots: one per output tile, two barrier counters per group and
  // the MoE scheduler's state, as exl3_devctx.cu sizes them.
  constexpr std::size_t kLockInts = MAX_TILES_C + (MAX_BARRIERS * 2) + MOE_SCHED_INTS;
  EXPECT_EQ(kLockInts * sizeof(int), 4'202'760U);
  // The barriers follow the tile locks, and the scheduler the barriers.
  EXPECT_EQ(BARRIER_LOCKS_OFFSET, MAX_TILES_C);
  EXPECT_EQ(MOE_SCHED_OFFSET, MAX_TILES_C + (2 * MAX_BARRIERS));
  EXPECT_EQ(WORKSPACE_SIZE, 16 * 1024 * 1024);
  // docs/backend-proof.md, "Persistent library workspaces".
  EXPECT_EQ((kLockInts * sizeof(int)) + WORKSPACE_SIZE, 20'979'976U);
}

}  // namespace
