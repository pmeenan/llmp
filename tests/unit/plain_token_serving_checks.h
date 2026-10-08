// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <vector>

#include "base/sha256.h"
#include "runtime/serving.h"

namespace jitllm::test {
// Actual adapter oracle shared by the small-family and paged-LLM fixtures.
// StateHash observes every initialized range; TokenCount comes from the runner,
// rather than the requested serving option. All host observations are funded.
template <class StateHash, class TokenCount>
runtime::Status CheckPlainServing(runtime::Server& server, runtime::Llm& model, std::uint32_t vocab,
                                  StateHash state_hash, TokenCount token_count,
                                  std::array<std::uint64_t, 2> expected_device_counts = {3, 5}) {
  namespace rt = runtime;
  struct HostCopies {
    engine::PagedNode& node;
    bool funded;
    explicit HostCopies(engine::PagedNode& n)
        : node(n), funded(n.ChargeHost(64ULL << 20U, false)) {}
    ~HostCopies() {
      if (funded) node.UnchargeHost(64ULL << 20U);
    }
  } copies(server.node());
  if (!copies.funded) return std::unexpected("plain serving observation funding refused");
  const auto branch = [&](std::uint32_t id) -> rt::Llm::Branch& { return **model.branch(id); };
  const auto select = [&]() {
    const std::array<rt::Llm::Branch*, 2> branches{&branch(0), &branch(1)};
    return server.SelectRequestBranches(model, branches);
  };
  constexpr std::array<std::int32_t, 7> seed{2, 818, 5279, 529, 7001, 563, 42};
  const auto finite = [&](const std::vector<float>& row) {
    return row.size() == vocab &&
           std::ranges::all_of(row, [](float v) { return std::isfinite(v); });
  };
  for (const auto width : {1U, 2U}) {
    std::array<std::vector<std::int32_t>, 2> expected_tokens, expected_history;
    std::array<std::vector<float>, 2> expected_heads;
    std::array<base::Sha256Digest, 2> expected_state{}, expected_continued{};
    for (const bool rows : {true, false}) {
      if (auto r = select(); !r) return r;
      std::array<std::vector<float>, 2> last;
      std::array<rt::GenerateOptions, 2> options;
      std::array<rt::Generation, 2> outputs;
      for (std::uint32_t id = 0; id < 2; ++id) {
        if (auto r = branch(id).Clear(); !r) return r;
        if (auto r = branch(id).Prefill(std::span(seed).first(6 + id), last[id]); !r) return r;
        options[id].max_tokens = width == 2 && id == 1 ? 3 : 4;
        options[id].stop = false;
        options[id].keep_logits = rows;
      }
      auto peer = state_hash(1);
      if (!peer) return std::unexpected(peer.error());
      struct Sessions {
        std::array<std::unique_ptr<rt::Llm::GenerationSession>, 2> owned;
        ~Sessions() {
          for (auto& s : owned)
            if (s) {
              s->Cancel();
              const auto r = s->Finish();
              EXPECT_TRUE(r) << (r ? "" : r.error());
            }
        }
      } sessions;
      const auto before = token_count();
      for (std::uint32_t id = 0; id < width; ++id) {
        auto opened = branch(id).BeginGeneration(last[id], options[id], outputs[id]);
        if (!opened) return std::unexpected(opened.error());
        sessions.owned[id] = std::move(*opened);
      }
      bool departed = false;
      while (!sessions.owned[0]->done()) {
        std::array<rt::Llm::GenerationSession*, 2> active{};
        std::size_t count = 0;
        for (std::uint32_t id = 0; id < width; ++id)
          if (!sessions.owned[id]->done()) active[count++] = sessions.owned[id].get();
        const bool scalar_after_departure = width == 2 && count == 1;
        auto saved_peer = state_hash(1);
        if (!saved_peer) return std::unexpected(saved_peer.error());
        if (auto r = model.RunGenerationWave(std::span(active).first(count)); !r) return r;
        if (scalar_after_departure) {
          departed = true;
          auto untouched = state_hash(1);
          if (!untouched || *untouched != *saved_peer)
            return std::unexpected("departed owner changed during scalar continuation");
        }
      }
      if (width == 2 && !departed) return std::unexpected("joined owner departure not exercised");
      for (std::uint32_t id = 0; id < width; ++id) {
        if (!sessions.owned[id]->done()) return std::unexpected("joined owner did not finish");
        if (auto r = sessions.owned[id]->Finish(); !r) return r;
        sessions.owned[id].reset();
        const auto expected_count = options[id].max_tokens;
        if (outputs[id].tokens.size() != expected_count ||
            outputs[id].logits.size() != (rows ? expected_count : 0U))
          return std::unexpected("actual plain output mode not selected");
        if (rows)
          for (const auto& row : outputs[id].logits)
            if (!finite(row)) return std::unexpected("complete finite row baseline required");
        auto state = state_hash(id);
        if (!state) return std::unexpected(state.error());
        if (rows) {
          expected_tokens[id] = outputs[id].tokens;
          expected_history[id] = branch(id).history();
          expected_state[id] = *state;
        } else if (outputs[id].tokens != expected_tokens[id] ||
                   branch(id).history() != expected_history[id] || *state != expected_state[id])
          return std::unexpected("row/token complete state or history differs");
      }
      if (token_count() - before != (rows ? 0U : expected_device_counts[width - 1]))
        return std::unexpected("actual device-token selection count differs");
      if (width == 1) {
        const auto untouched = state_hash(1);
        if (!untouched || *untouched != *peer) return std::unexpected("scalar changed its peer");
      }
      // A completed joined device generation survives a real idle spill;
      // the following actual row restores it and consumes its pending anchor.
      if (!rows && width == 2) {
        const auto retired = server.RetireRequestBranches(model, true);
        if (!retired.result) return retired.result;
        if (!retired.references_retired) return std::unexpected("request references not retired");
        if (auto r = model.SpillIdle(branch(0)); !r) return r;
        if (!branch(0).spilled()) return std::unexpected("idle state did not spill");
        if (auto r = select(); !r) return r;
      }
      for (std::uint32_t id = 0; id < width; ++id) {
        auto history = branch(id).history();
        const auto stable = static_cast<std::uint32_t>(history.size());
        history.push_back(outputs[id].tokens.back());
        std::uint32_t reused = 0;
        std::vector<float> continued;
        if (auto r = branch(id).PreparePrompt(history, stable, continued, reused); !r) return r;
        if (reused != stable || !finite(continued))
          return std::unexpected("actual complete continuation or reuse missing");
        auto state = state_hash(id);
        if (!state) return std::unexpected(state.error());
        if (rows) {
          expected_heads[id] = std::move(continued);
          expected_continued[id] = *state;
        } else if (continued != expected_heads[id] || *state != expected_continued[id])
          return std::unexpected("restored token continuation differs from row baseline");
      }
    }
  }
  // A joined request for scores, or a non-greedy sampler, keeps the whole
  // homogeneous wave on its row path even when its peer could use tokens.
  for (const bool sampling : {false, true}) {
    std::array<std::vector<float>, 2> last;
    std::array<rt::GenerateOptions, 2> options;
    std::array<rt::Generation, 2> outputs;
    std::array<std::unique_ptr<rt::Llm::GenerationSession>, 2> sessions;
    struct Cleanup {
      decltype(sessions)& owned;
      ~Cleanup() {
        for (auto& s : owned)
          if (s) {
            s->Cancel();
            const auto r = s->Finish();
            EXPECT_TRUE(r) << (r ? "" : r.error());
          }
      }
    } cleanup{sessions};
    for (std::uint32_t id = 0; id < 2; ++id) {
      if (auto r = branch(id).Clear(); !r) return r;
      if (auto r = branch(id).Prefill(std::span(seed).first(6 + id), last[id]); !r) return r;
      options[id].max_tokens = 2;
      options[id].stop = false;
    }
    if (sampling)
      options[0].sampling = execution::SamplingParams{.temperature = 0.7F, .top_k = 16};
    else
      options[0].keep_logits = true;
    const auto before = token_count();
    for (std::uint32_t id = 0; id < 2; ++id) {
      auto opened = branch(id).BeginGeneration(last[id], options[id], outputs[id]);
      if (!opened) return std::unexpected(opened.error());
      sessions[id] = std::move(*opened);
    }
    const std::array<rt::Llm::GenerationSession*, 2> active{sessions[0].get(), sessions[1].get()};
    if (auto r = model.RunGenerationWave(active); !r) return r;
    if (token_count() != before) return std::unexpected("mixed row wave selected device tokens");
    for (auto& s : sessions) {
      if (!s->done()) return std::unexpected("mixed wave did not finish");
      if (auto r = s->Finish(); !r) return r;
      s.reset();
    }
    if (outputs[0].tokens.size() != 2 || outputs[1].tokens.size() != 2 ||
        (!sampling && outputs[0].logits.size() != 2))
      return std::unexpected("mixed row outputs incomplete");
  }
  return {};
}
}  // namespace jitllm::test
