// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <expected>
#include <format>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "base/sha256.h"
#include "runtime/serving.h"

namespace llmp::test {

// What a model's kept-across-a-restart control varies (D-105, D-112;
// docs/runtime-serving.md, "Conversations kept across a restart").
struct KeptRestartPlan {
  std::uint32_t vocab = 0;
  // Two owners' first prompts: their lengths, and the stable boundary
  // whose turn checkpoint the first turn leaves (it must be shorter than
  // both prompts).
  std::array<std::uint32_t, 2> rows{257, 258};
  std::uint32_t boundary = 128;
  std::uint32_t first_tokens = 8;
  std::uint32_t second_tokens = 12;
  // The continuation chooses its tokens on the device (no logit rows); the
  // first turn always keeps its rows.
  bool device_tokens = false;
  // A lone owner speculates (a drafter is configured): its first turn must
  // draft, so the kept state includes the drafter's.
  bool expect_drafts = false;
  // `owner`'s `count` prompt tokens, within the vocabulary.
  std::function<std::vector<std::int32_t>(std::uint32_t owner, std::size_t count)> tokens;
};

// The conversation kept across a restart continues exactly. Two owners
// each have a prompt (a turn checkpoint at its boundary) and a first turn
// (lone generation); the uninterrupted control then runs the second turn
// (the same two owners joined, after a new user message) and a fork from
// the turn checkpoint. The first turn is run again, the conversations are
// spilled and kept, the server is stopped and a new one started on the
// same directories (`Harness::Restart`), which adopts them. The owners are
// restored, and the second turn and the fork run again. The restored
// state's complete hash, and the continuation's heads, tokens, drafts, every
// complete logit row, history and settled state hash, must equal the
// control's, bit for bit.
//
// Harness: server(), model(), Branch(id), StateHash(slot) (the used state of
// every stage, hashed), Log() and Restart() (persist the kept records, stop
// the server, start another on the same directories with both owners
// selected; the old model() and server() are invalid after it).
template <class Harness>
runtime::Status CheckKeptRestart(Harness& h, const KeptRestartPlan& plan) {
  namespace rt = runtime;
  using Digest = base::Sha256Digest;
  const auto fail = [](std::string why) { return std::unexpected(std::move(why)); };

  struct Funding {
    engine::PagedNode& node;
    bool funded;
    explicit Funding(engine::PagedNode& n) : node(n), funded(n.ChargeHost(64ULL << 20U, false)) {}
    Funding(const Funding&) = delete;
    Funding& operator=(const Funding&) = delete;
    ~Funding() {
      if (funded) node.UnchargeHost(64ULL << 20U);
    }
  };
  std::optional<Funding> funding;
  const auto fund = [&]() -> rt::Status {
    funding.emplace(h.server().node());
    if (!funding->funded) return fail("kept restart observation funding refused");
    return {};
  };
  const auto same = [](const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
  };
  const auto finite = [&](const std::vector<float>& row) {
    return row.size() == plan.vocab &&
           std::ranges::all_of(row, [](float v) { return std::isfinite(v); });
  };

  struct Turn {
    std::array<std::vector<std::int32_t>, 2> prompt, history, next, alt, final_history;
    std::array<std::vector<float>, 2> head0, head1, fork;
    std::array<rt::Generation, 2> turn1, turn2;
    std::array<Digest, 2> state1{}, state2{};
  };
  constexpr std::array<std::int32_t, 3> kMessage{563, 42, 818};
  constexpr std::array<std::int32_t, 5> kAlternative{5279, 529, 7001, 563, 2};

  // Each owner's prompt and its lone first turn, rows kept.
  const auto first_turn = [&](Turn& t) -> rt::Status {
    for (std::uint32_t id = 0; id < 2; ++id) {
      if (auto r = h.Branch(id).Clear(); !r) return r;
      t.prompt[id] = plan.tokens(id, plan.rows[id]);
      std::uint32_t reused = 0;
      if (auto r = h.Branch(id).PreparePrompt(t.prompt[id], plan.boundary, t.head0[id], reused); !r)
        return r;
      if (reused != 0) return fail("a first prompt reused state");
      if (h.Branch(id).turn_checkpoints() != 1)
        return fail(std::format("owner {} left {} turn checkpoints, not 1", id,
                                h.Branch(id).turn_checkpoints()));
      rt::GenerateOptions options;
      options.max_tokens = plan.first_tokens;
      options.stop = false;
      options.keep_logits = true;
      if (auto r = h.Branch(id).Generate(t.head0[id], options, t.turn1[id]); !r) return r;
      if (t.turn1[id].tokens.size() != plan.first_tokens ||
          t.turn1[id].logits.size() != plan.first_tokens)
        return fail("a first turn is incomplete");
      if (plan.expect_drafts && t.turn1[id].drafted == 0)
        return fail("a lone first turn did not draft");
      t.history[id] = h.Branch(id).history();
      auto state = h.StateHash(id);
      if (!state) return fail(state.error());
      t.state1[id] = *state;
    }
    return {};
  };

  // The second turn: each owner's history, its anchor token and a new
  // message, both owners joined; then a fork from the first turn's
  // checkpoint (the prompt's stable prefix and a different tail).
  const auto second_turn = [&](Turn& t) -> rt::Status {
    for (std::uint32_t id = 0; id < 2; ++id) {
      t.next[id] = t.history[id];
      t.next[id].push_back(t.turn1[id].tokens.back());
      t.next[id].insert(t.next[id].end(), kMessage.begin(), kMessage.end());
      std::uint32_t reused = 0;
      if (auto r = h.Branch(id).PreparePrompt(t.next[id], 0, t.head1[id], reused); !r) return r;
      if (reused != t.history[id].size())
        return fail(std::format("owner {} reused {} of its {} kept tokens", id, reused,
                                t.history[id].size()));
    }
    struct Sessions {
      std::array<std::unique_ptr<rt::Llm::GenerationSession>, 2> owned;
      ~Sessions() {
        for (auto& s : owned) {
          if (!s) continue;
          if (!s->done()) s->Cancel();
          const auto r = s->Finish();
          EXPECT_TRUE(r) << (r ? "" : r.error());
        }
      }
    } sessions;
    std::array<rt::GenerateOptions, 2> options;
    for (std::uint32_t id = 0; id < 2; ++id) {
      options[id].max_tokens = plan.second_tokens;
      options[id].stop = false;
      options[id].keep_logits = !plan.device_tokens;
      auto opened = h.Branch(id).BeginGeneration(t.head1[id], options[id], t.turn2[id]);
      if (!opened) return fail(opened.error());
      sessions.owned[id] = std::move(*opened);
    }
    for (unsigned wave = 0; wave < 256; ++wave) {
      std::array<rt::Llm::GenerationSession*, 2> active{};
      std::size_t count = 0;
      for (auto& s : sessions.owned)
        if (!s->done()) active[count++] = s.get();
      if (count == 0) break;
      if (auto r = h.model().RunGenerationWave(std::span(active).first(count)); !r) return r;
    }
    for (std::uint32_t id = 0; id < 2; ++id) {
      if (!sessions.owned[id]->done()) return fail("a second turn made no progress");
      if (auto r = sessions.owned[id]->Finish(); !r) return r;
      sessions.owned[id].reset();
      if (t.turn2[id].tokens.size() != plan.second_tokens ||
          t.turn2[id].logits.size() != (plan.device_tokens ? 0U : plan.second_tokens))
        return fail("a second turn is incomplete");
      auto state = h.StateHash(id);
      if (!state) return fail(state.error());
      t.state2[id] = *state;
      t.final_history[id] = h.Branch(id).history();
    }
    for (std::uint32_t id = 0; id < 2; ++id) {
      t.alt[id] = plan.tokens(id, plan.boundary);
      t.alt[id].insert(t.alt[id].end(), kAlternative.begin(), kAlternative.end());
      std::uint32_t reused = 0;
      if (auto r = h.Branch(id).PreparePrompt(t.alt[id], plan.boundary, t.fork[id], reused); !r)
        return r;
      if (reused != plan.boundary)
        return fail(std::format("owner {}'s fork reused {} tokens, not its checkpoint's {}", id,
                                reused, plan.boundary));
    }
    return {};
  };

  const auto compare = [&](const Turn& got, const Turn& want, const char* what) -> rt::Status {
    for (std::uint32_t id = 0; id < 2; ++id) {
      const auto at = [&](const char* field) {
        return std::format("{} owner {}: {}", what, id, field);
      };
      if (got.history[id] != want.history[id]) return fail(at("first-turn history"));
      if (got.turn1[id].tokens != want.turn1[id].tokens) return fail(at("first-turn tokens"));
      if (got.turn1[id].logits.size() != want.turn1[id].logits.size())
        return fail(at("first-turn row count"));
      for (std::size_t r = 0; r < want.turn1[id].logits.size(); ++r)
        if (!same(got.turn1[id].logits[r], want.turn1[id].logits[r]))
          return fail(at("first-turn row ") + std::to_string(r));
      if (got.state1[id] != want.state1[id]) return fail(at("first-turn state hash"));
      if (!same(got.head1[id], want.head1[id])) return fail(at("continuation prompt head"));
      if (!finite(want.head1[id])) return fail(at("non-finite head"));
      if (got.turn2[id].tokens != want.turn2[id].tokens) return fail(at("continuation tokens"));
      if (got.turn2[id].drafted != want.turn2[id].drafted ||
          got.turn2[id].accepted != want.turn2[id].accepted)
        return fail(at("continuation drafts"));
      if (got.turn2[id].logits.size() != want.turn2[id].logits.size())
        return fail(at("continuation row count"));
      for (std::size_t r = 0; r < want.turn2[id].logits.size(); ++r) {
        if (!finite(want.turn2[id].logits[r])) return fail(at("non-finite row"));
        if (!same(got.turn2[id].logits[r], want.turn2[id].logits[r]))
          return fail(at("continuation row ") + std::to_string(r));
      }
      if (got.state2[id] != want.state2[id]) return fail(at("continuation state hash"));
      if (!same(got.fork[id], want.fork[id])) return fail(at("checkpoint fork head"));
      if (got.final_history[id] != want.final_history[id]) return fail(at("continuation history"));
    }
    return {};
  };

  if (!plan.tokens) return fail("no prompt generator");
  if (auto r = fund(); !r) return r;
  Turn want, again, got;
  // The control: nothing leaves the process.
  if (auto r = first_turn(want); !r) return fail("control first turn: " + r.error());
  if (auto r = second_turn(want); !r) return fail("control second turn: " + r.error());
  // The same first turn again, to be kept.
  if (auto r = first_turn(again); !r) return fail("kept first turn: " + r.error());
  {
    for (std::uint32_t id = 0; id < 2; ++id) {
      if (again.history[id] != want.history[id] ||
          again.turn1[id].tokens != want.turn1[id].tokens || again.state1[id] != want.state1[id])
        return fail(std::format("owner {}'s first turn differs from its own repeat", id));
      for (std::size_t r = 0; r < want.turn1[id].logits.size(); ++r)
        if (!same(again.turn1[id].logits[r], want.turn1[id].logits[r]))
          return fail(std::format("owner {}'s first-turn row {} differs from its repeat", id, r));
    }
  }
  // Keep: end the requests, spill both, persist, stop.
  {
    const auto retired = h.server().RetireRequestBranches(h.model(), true);
    if (!retired.result) return retired.result;
    if (!retired.references_retired) return fail("request references were not retired");
    for (std::uint32_t id = 0; id < 2; ++id) {
      if (auto r = h.model().SpillIdle(h.Branch(id)); !r) return r;
      if (!h.Branch(id).spilled()) return fail("an idle conversation did not spill");
    }
  }
  funding.reset();
  if (auto r = h.Restart(); !r) return fail("restart: " + r.error());
  if (auto r = fund(); !r) return r;
  // Adopted: the history, the checkpoint, the state still on disk.
  for (std::uint32_t id = 0; id < 2; ++id) {
    if (h.Branch(id).history() != again.history[id])
      return fail(std::format("owner {}'s history was not adopted", id));
    if (!h.Branch(id).spilled()) return fail(std::format("owner {} was not adopted spilled", id));
    if (h.Branch(id).turn_checkpoints() != 1)
      return fail(std::format("owner {} adopted {} turn checkpoints, not 1", id,
                              h.Branch(id).turn_checkpoints()));
  }
  // Restored from disk with no recomputation: the complete state is the
  // first turn's.
  for (std::uint32_t id = 0; id < 2; ++id) {
    auto restoring = h.Branch(id).BeginPrompt(again.history[id], 0, false, true);
    if (!restoring) return fail(restoring.error());
    rt::Status run;
    while (!(*restoring)->done() && (run = (*restoring)->Advance())) {
    }
    if (!run) return run;
    if (auto r = (*restoring)->Finish(); !r) return r;
    if ((*restoring)->reused() != again.history[id].size())
      return fail(std::format("owner {}'s restore reused {} of {} tokens", id,
                              (*restoring)->reused(), again.history[id].size()));
    restoring->reset();
    auto state = h.StateHash(id);
    if (!state) return fail(state.error());
    if (*state != want.state1[id])
      return fail(std::format("owner {}'s restored state hash differs", id));
  }
  // The continuation, from the restored state.
  got.prompt = again.prompt;
  got.history = again.history;
  got.turn1 = again.turn1;
  got.state1 = again.state1;
  got.head0 = again.head0;
  if (auto r = second_turn(got); !r) return fail("restored second turn: " + r.error());
  if (auto r = compare(got, want, "restored against uninterrupted"); !r) return r;
  const std::string text = h.Log();
  if (text.find("kept conversation was refused") != std::string::npos)
    return fail("the start refused a kept conversation: " + text);
  return {};
}

}  // namespace llmp::test
