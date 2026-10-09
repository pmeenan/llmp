// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// A model's batching capability (docs/engine-components.md, "Execution"):
// what its components allow the wave composer (engine/wave_composer.h) to
// join across requests in one step, as data. Components will derive it
// from their ops' row-joinability (products and routed experts join rows;
// attention over one request's cache and recurrences stay per owner) and
// from the limits of the implementations their plans select; until they
// exist, a caller states it. The composer applies it and nothing else: no
// family constant decides a join.
//
// Header-only; model layer, below the components that will supply it.

#ifndef LLMP_MODEL_BATCHING_H_
#define LLMP_MODEL_BATCHING_H_

#include <cstdint>

namespace llmp::model {

// A row-invariant step (an exact verify, D-092) never takes the joined
// alignment or a common read: its owners read what each reads alone, so no
// row's arithmetic follows a peer's reads (attention's split over the read
// cells). The composer applies it; the runner refuses such a step whose
// reads follow its peers.
struct Batching {
  // Owners one step may join (1: every owner steps alone); at most
  // kMaxRequestSlots.
  std::uint32_t max_segments = 1;
  // One owner's rows in a step (its prefill chunk bound; 0: unbounded).
  std::uint32_t max_segment_rows = 0;
  // Every owner's rows in one step (a wave's bound, DeepSeek's 16; 0:
  // unbounded).
  std::uint32_t max_step_rows = 0;
  // Joined groups within a step: consecutive owners while their rows fit
  // `group_rows` (the widest joined product, Qwen3.8's 16) and they number
  // at most `group_segments` (Gemma3's four-root attention carriers); 0
  // leaves that bound off. A row-invariant step (a verify, D-092) may cap
  // its groups lower (`invariant_group_segments`, Gemma4's 8 against 12).
  std::uint32_t group_rows = 0;
  std::uint32_t group_segments = 0;
  std::uint32_t invariant_group_segments = 0;
  // The read alignment of a step joining several owners, so its plan's key
  // repeats as they grow (Qwen3.8's 2,048); 0: each owner's own (the state
  // contract's). The runner backs each owner's state through it once every
  // owner's own rows are backed, so a capacity refusal is that owner's
  // alone, and one there steps alone at its own.
  std::uint32_t joined_read_align = 0;
  // Joined owners read the step's widest cells (Gemma's common read
  // maximum for owner attention).
  bool common_read = false;
  // Prompt rows join across owners (decode, verify, draft and score rows
  // join whenever max_segments allows). Joined prompt owners need equal
  // rows and equal reads when `equal_prefill` (Gemma2/3's packed owner
  // prefill).
  bool join_prefill = false;
  bool equal_prefill = false;
  // Owners whose head intent differs (a state-only cut beside a headed
  // chunk) may join; otherwise they step apart.
  bool mixed_heads = false;
};

}  // namespace llmp::model

#endif  // LLMP_MODEL_BATCHING_H_
