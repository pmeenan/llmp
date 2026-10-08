// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Manual C1/P64/depth3 target+assistant transaction, never CTest or serving.
#include <sys/stat.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <span>
#include <string>
#include <tuple>
#include <vector>

#include "artifact/gguf_metadata.h"
#include "base/sha256.h"
#include "engine/gemma4_assistant.h"
#include "engine/support.h"
#include "execution/sampling.h"
#include "tokenizer/gguf.h"

namespace {
namespace en = llmp::engine;
namespace ar = llmp::artifact;
namespace md = llmp::model;
namespace tk = llmp::tokenizer;
using en::support::Error;
constexpr std::uint64_t kHost = 64ULL << 20U, kAdmission = 512ULL << 20U;
constexpr std::uint32_t kVocab = 262144, kPast = 64;

template <class T>
en::Status Write(const std::filesystem::path& path, std::span<const T> values) {
  std::ofstream file(path, std::ios::binary | std::ios::noreplace);
  if (!file) return Error("exclusive output creation failed");
  file.write(reinterpret_cast<const char*>(values.data()),
             static_cast<std::streamsize>(values.size_bytes()));
  file.flush();
  return file ? en::Status{} : Error("complete output write failed");
}
en::Status Text(const std::filesystem::path& path, const std::string& value) {
  return Write<char>(path, value);
}
bool Exact(std::span<const float> a, std::span<const float> b) {
  return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size_bytes()) == 0 &&
         std::ranges::all_of(a, [](float x) { return std::isfinite(x); });
}
struct Vocabulary {
  tk::GgufTokenizer tokenizer;
  ar::GgufMetadata raw;
  md::Gemma4AssistantVocabulary view() const {
    return {tokenizer.spec.tokens, tokenizer.spec.merges, raw.at("tokenizer.ggml.scores").reals,
            raw.at("tokenizer.ggml.token_type").integers};
  }
};
std::expected<Vocabulary, std::string> Decode(const ar::Artifact& artifact,
                                              std::string_view filename) {
  const auto listed = std::ranges::find_if(artifact.files(), [filename](const auto& file) {
    return file.role == ar::FileRole::kSourceMetadata &&
           file.path == "meta/" + std::string(filename);
  });
  if (listed == artifact.files().end() || listed->bytes.value() > (16ULL << 20U))
    return Error("vocabulary exceeds funded metadata envelope");
  auto bytes = artifact.ReadMetadata(filename);
  if (!bytes) return Error(bytes.error().ToString());
  auto input = std::as_bytes(std::span(bytes->data(), bytes->size()));
  auto tokenizer = tk::ReadGgufTokenizer(input);
  constexpr std::array<std::string_view, 2> keys{"tokenizer.ggml.scores",
                                                 "tokenizer.ggml.token_type"};
  auto raw = ar::ReadGgufMetadata(input, keys);
  if (!tokenizer || !raw || !raw->contains(keys[0]) || !raw->contains(keys[1]))
    return Error("vocabulary parse failed");
  return Vocabulary{std::move(*tokenizer), std::move(*raw)};
}
struct Lifetime {
  en::PagedNode node{{.slot_bytes = en::kSlabSlotBytes}};
  std::unique_ptr<en::Gemma4Runner> runner;
  std::vector<en::PagedModel*> entered;
};
class Screen {
 public:
  Screen(Lifetime& life, en::Gemma4Assistant* assistant, std::span<const std::int32_t> ids,
         std::filesystem::path out)
      : node_(life.node),
        runner_(*life.runner),
        assistant_(assistant),
        ids_(ids),
        out_(std::move(out)) {}
  std::expected<std::vector<en::LiveState::Range>, std::string> Writes(std::uint32_t past,
                                                                       std::uint32_t rows) {
    auto writes = md::Gemma4ChunkWrites(runner_.profile(), runner_.layout(), past, rows);
    if (!writes) return Error(writes.error());
    std::vector<en::LiveState::Range> result;
    for (const auto& r : *writes)
      result.push_back({.region = 0, .offset = r.offset, .bytes = r.bytes});
    return result;
  }
  std::expected<std::string, std::string> Hash(std::span<const en::LiveState::Range> ranges) {
    std::uint64_t bytes = 0;
    for (const auto& r : ranges) bytes += r.bytes;
    if (bytes == 0) return llmp::base::ToHex(llmp::base::Sha256{}.Finish());
    std::vector<llmp::catalog::ExtentId> extents;
    auto pinned = node_.Pinned(bytes, 0, extents);
    if (!pinned) return Error(pinned.error());
    if (auto r = runner_.CopyState(0, *pinned, ranges, true); !r) {
      node_.KeepPinned(*pinned);
      return Error(r.error());
    }
    const auto result =
        llmp::base::ToHex(llmp::base::Sha256{}
                              .Update(std::span(static_cast<const std::byte*>(*pinned), bytes))
                              .Finish());
    if (auto r = node_.FreePinned(*pinned); !r) return Error(r.error());
    return result;
  }
  en::Status Feature(std::uint32_t first, std::uint32_t rows, std::vector<float>& out) {
    const auto count = std::uint64_t{rows} * runner_.profile().width;
    std::vector<llmp::catalog::ExtentId> extents;
    auto pinned = node_.Pinned(count * sizeof(float), 0, extents);
    if (!pinned) return Error(pinned.error());
    if (auto r = runner_.CopyFeatures(0, first, rows, *pinned); !r) {
      node_.KeepPinned(*pinned);
      return r;
    }
    const auto* values = static_cast<const float*>(*pinned);
    out.assign(values, values + count);
    if (auto r = node_.FreePinned(*pinned); !r) return r;
    return std::ranges::all_of(out, [](float x) { return std::isfinite(x); })
               ? en::Status{}
               : Error("nonfinite completed feature");
  }
  en::Status Prefix(std::vector<float>& head) {
    if (auto r = runner_.Clear(0); !r) return r;
    const en::Gemma4Runner::Work work{0, 0, ids_.first(kPast), &head};
    return runner_.Wave(std::span(&work, 1));
  }
  en::Status Run(std::string_view mode) {
    for (unsigned repeat = 0; repeat < 2; ++repeat) {
      const auto dir = out_ / (repeat == 0 ? "first" : "repeat");
      std::error_code ec;
      if (!std::filesystem::create_directory(dir, ec) || ec) return Error("round must be new");
      std::vector<float> initial, feature, draft_head, draft_feature, draft_heads, draft_features,
          control, control_features, restored, restored_feature, selected;
      const auto width = runner_.profile().width;
      initial.reserve(kVocab);
      feature.reserve(width);
      draft_head.reserve(kVocab);
      draft_feature.reserve(width);
      draft_heads.reserve(3 * kVocab);
      draft_features.reserve(3 * width);
      control.reserve(4 * kVocab);
      control_features.reserve(4 * width);
      restored.reserve(kVocab);
      restored_feature.reserve(width);
      selected.reserve(width);
      en::Gemma4GreedyWorkspace workspace;
      en::Gemma4GreedyResult result;
      workspace.draft_head.reserve(kVocab);
      workspace.verify_heads.reserve(4 * kVocab);
      workspace.verify_features.reserve(4 * width);
      result.head.reserve(kVocab);
      result.feature.reserve(width);
      std::uint64_t capacity_bytes = 0;
      for (const auto* values :
           {&initial, &feature, &draft_head, &draft_feature, &draft_heads, &draft_features,
            &control, &control_features, &restored, &restored_feature, &selected,
            &workspace.draft_head, &workspace.verify_heads, &workspace.verify_features,
            &result.head, &result.feature})
        capacity_bytes += values->capacity() * sizeof(float);
      if (capacity_bytes > kHost - (1ULL << 20U))
        return Error("actual caller vector capacities exceed retained host grant");
      if (auto r = Prefix(initial); !r) return r;
      if (repeat == 0) {
        const auto& policy = runner_.last_built_policy();
        std::cout << "GREEDY_TARGET_POLICY phase=prefill last_built_rows=" << policy.rows
                  << " norm_rope=" << policy.norm_rope << " norm_add=" << policy.norm_add << '\n';
      }
      if (initial.size() != kVocab || !Exact(initial, initial))
        return Error("initial head differs");
      if (auto r = Feature(kPast - 1, 1, feature); !r) return r;
      if (auto r = runner_.ReserveStateThrough(0, kPast + 4); !r) return r;
      auto initialized = runner_.CheckpointRanges(kPast + 4);
      if (!initialized) return Error(initialized.error());
      auto protected_before = Hash(*initialized);
      if (!protected_before) return Error(protected_before.error());
      std::array<std::int32_t, 4> proposal{};
      auto anchor = llmp::execution::Greedy(initial);
      if (!anchor) return Error("initial anchor refused");
      proposal[0] = *anchor;
      if (mode == "unit") {
        {
          auto borrow = runner_.BorrowFrozen(0, *anchor);
          if (!borrow) return Error(borrow.error());
          const std::array<const en::Gemma4Runner::FrozenBorrow*, 1> borrows{&*borrow};
          const std::array<std::vector<float>*, 1> heads{&draft_head}, features{&draft_feature};
          for (std::uint32_t step = 0; step < 3; ++step) {
            const std::array<std::int32_t, 1> token{proposal[step]};
            if (auto r = assistant_->Step(borrows, token, step == 0, heads, features); !r) return r;
            if (draft_head.size() != kVocab || draft_feature.size() != width ||
                !Exact(draft_head, draft_head) || !Exact(draft_feature, draft_feature))
              return Error("draft output differs");
            draft_heads.insert(draft_heads.end(), draft_head.begin(), draft_head.end());
            draft_features.insert(draft_features.end(), draft_feature.begin(), draft_feature.end());
            auto next = llmp::execution::Greedy(draft_head);
            if (!next) return Error("draft anchor refused");
            proposal[step + 1] = *next;
            if (borrow->prefix() != kPast) return Error("draft query moved");
            if (auto r = runner_.CheckBorrow(*borrow); !r) return r;
            auto unchanged = Hash(*initialized);
            if (!unchanged || *unchanged != *protected_before)
              return Error("draft mutated protected initialized target state");
          }
        }
        if (auto r = Feature(kPast - 1, 1, restored_feature); !r) return r;
        if (!Exact(feature, restored_feature))
          return Error("draft mutated protected target feature");
      } else {
        std::copy_n(ids_.begin() + kPast, 4, proposal.begin());
      }
      // Independent ordinary four-query target. It uses the same input block,
      // but no Verify/Accept machinery. No scalar-width numerical oracle.
      const en::Gemma4Runner::Work manual{0, kPast, proposal, &control};
      if (auto r = runner_.Wave(std::span(&manual, 1), true, true); !r) return r;
      if (repeat == 0) {
        const auto& policy = runner_.last_built_policy();
        std::cout << "GREEDY_TARGET_POLICY phase=normal4 last_built_rows=" << policy.rows
                  << " norm_rope=" << policy.norm_rope << " norm_add=" << policy.norm_add << '\n';
      }
      if (auto r = Feature(kPast, 4, control_features); !r) return r;
      if (control.size() != 4 * kVocab || !Exact(control, control))
        return Error("normal four-query outputs differ");
      auto decision = en::JudgeGemma4Greedy(std::span(proposal).subspan(1), control, kVocab);
      if (!decision) return Error(decision.error());
      const auto keep = mode == "unit" ? decision->keep : 4U;
      auto next_anchor = decision->next_anchor;
      if (mode == "teacher") {
        auto next = llmp::execution::Greedy(std::span(control).last(kVocab));
        if (!next) return Error("selected pending target anchor refused");
        next_anchor = *next;
      }
      auto semantic = Writes(0, kPast + keep);
      if (!semantic) return Error(semantic.error());
      auto expected_state = Hash(*semantic);
      if (!expected_state) return Error(expected_state.error());
      if (auto r = Prefix(restored); !r) return r;
      if (auto r = Feature(kPast - 1, 1, restored_feature); !r) return r;
      if (!Exact(initial, restored) || !Exact(feature, restored_feature))
        return Error("same-query64 prefix reset differs");
      if (auto r = runner_.ReserveStateThrough(0, kPast + 4); !r) return r;
      auto reset_state = Hash(*initialized);
      if (!reset_state || *reset_state != *protected_before)
        return Error("normal control reset changed protected initialized state");
      std::vector<en::LiveState::Range> rejected;
      if (keep < 4) {
        auto writes = Writes(kPast + keep, 4 - keep);
        if (!writes) return Error(writes.error());
        rejected = std::move(*writes);
      }
      auto rejected_before = Hash(rejected);
      if (!rejected_before) return Error(rejected_before.error());
      if (mode == "unit") {
        if (auto r = assistant_->GreedyUnit(0, kPast, 3, initial, workspace, result); !r) return r;
        if (!Exact(workspace.draft_head, draft_head))
          return Error("real unit last draft recurrence differs from manual observation");
        if (result.count != keep || result.past != kPast + keep ||
            result.next_anchor != decision->next_anchor)
          return Error("greedy unit acceptance differs from independent target");
        for (std::uint32_t row = 0; row < 4; ++row)
          if (result.committed[row] != (row < keep ? proposal[row] : 0))
            return Error("committed prefix differs");
      } else {
        if (auto r = runner_.Verify(0, kPast, proposal, workspace.verify_heads,
                                    workspace.verify_features);
            !r)
          return r;
        if (auto r = runner_.AcceptVerify(0, 4); !r) return r;
        result.head.assign(workspace.verify_heads.end() - static_cast<std::ptrdiff_t>(kVocab),
                           workspace.verify_heads.end());
        result.feature.assign(workspace.verify_features.end() - static_cast<std::ptrdiff_t>(width),
                              workspace.verify_features.end());
        result.count = 4;
        result.past = kPast + 4;
        result.next_anchor = next_anchor;
        result.committed = proposal;
      }
      if (!Exact(workspace.verify_heads, control) ||
          !Exact(workspace.verify_features, control_features) ||
          !Exact(result.head, std::span(control).subspan((keep - 1) * kVocab, kVocab)) ||
          !Exact(result.feature, std::span(control_features).subspan((keep - 1) * width, width)))
        return Error("real unit full rows or selected carry differ from independent target");
      auto accepted_state = Hash(*semantic), rejected_after = Hash(rejected);
      if (!accepted_state || *accepted_state != *expected_state || !rejected_after ||
          *rejected_after != *rejected_before)
        return Error("accepted semantic state or exact rejected restoration differs");
      if (auto r = Feature(result.past - 1, 1, selected); !r) return r;
      auto slot = runner_.request_slot(0);
      if (!slot || !(*slot)->state_usable() || (*slot)->completed_positions() != result.past ||
          !Exact(result.feature, selected))
        return Error("retired selected target feature or cursor differs");
      for (const auto& [name, values] :
           std::array<std::pair<const char*, std::span<const float>>, 8>{
               {{"initial-head.f32", initial},
                {"initial-feature.f32", feature},
                {"draft-heads.f32", draft_heads},
                {"draft-features.f32", draft_features},
                {"verify-heads.f32", workspace.verify_heads},
                {"verify-features.f32", workspace.verify_features},
                {"selected-head.f32", result.head},
                {"selected-feature.f32", result.feature}}})
        if (auto r = Write<float>(dir / name, values); !r) return r;
      if (auto r = Write<std::int32_t>(dir / "proposal.i32", proposal); !r) return r;
      if (auto r = Write<std::int32_t>(dir / "committed.i32", result.committed); !r) return r;
      if (auto r = Text(dir / "completion.json",
                        "{\"profile\":\"" + std::string(width == 5376 ? "31" : "26") +
                            "\",\"feature_width\":" + std::to_string(width) +
                            ",\"vocab\":262144,\"mode\":\"" + std::string(mode) +
                            "\",\"past\":64,\"prefill_query_rows\":64,"
                            "\"verify_rows\":4,\"keep\":" +
                            std::to_string(keep) +
                            ",\"completed_endpoint\":" + std::to_string(result.past) +
                            ",\"next_anchor\":" + std::to_string(result.next_anchor) +
                            ",\"pending_anchor_committed\":false,\"protected_state_sha256\":\"" +
                            *protected_before + "\",\"accepted_semantic_sha256\":\"" +
                            *accepted_state + "\",\"rejected_before_sha256\":\"" +
                            *rejected_before + "\",\"rejected_after_sha256\":\"" + *rejected_after +
                            "\",\"same_four_query_control_exact\":true,"
                            "\"draft_protected_target\":true}\n");
          !r)
        return r;
    }
    return {};
  }

  en::Status Run32(bool speculate, std::int32_t eos) {
    if (eos < 0 || eos >= static_cast<std::int32_t>(kVocab) || (speculate && !assistant_))
      return Error("bounded32 mode needs authenticated EOS and optional assistant");
    constexpr std::uint32_t emitted = 32;
    struct Unit {
      std::uint32_t past = 0, depth = 0, rows = 0, keep = 0;
    };
    struct Record {
      std::array<std::int32_t, emitted> tokens{};
      std::array<Unit, emitted> units{};
      std::uint32_t count = 0, unit_count = 0, verified = 0, drafted = 0, eos_count = 0;
      double seconds = 0;
    };
    std::vector<float> frontier, initial_feature, prediction_heads, verify_heads, verify_features,
        last_draft_heads;
    en::Gemma4GreedyWorkspace workspace;
    en::Gemma4GreedyResult result;
    frontier.reserve(kVocab);
    prediction_heads.reserve(emitted * kVocab);
    if (speculate) {
      initial_feature.reserve(runner_.profile().width);
      verify_heads.reserve(emitted * 4 * kVocab);
      verify_features.reserve(emitted * 4 * runner_.profile().width);
      last_draft_heads.reserve(emitted * kVocab);
      workspace.draft_head.reserve(kVocab);
      workspace.verify_heads.reserve(4 * kVocab);
      workspace.verify_features.reserve(4 * runner_.profile().width);
      result.head.reserve(kVocab);
      result.feature.reserve(runner_.profile().width);
    }
    std::uint64_t capacities = 0;
    for (const auto* v :
         {&frontier, &initial_feature, &prediction_heads, &verify_heads, &verify_features,
          &last_draft_heads, &workspace.draft_head, &workspace.verify_heads,
          &workspace.verify_features, &result.head, &result.feature})
      capacities += v->capacity() * sizeof(float);
    const auto grant = speculate ? (256ULL << 20U) : kHost;
    if (capacities > grant - (1ULL << 20U))
      return Error("bounded32 actual vector capacities exceed caller funding");
    auto ranges = Writes(0, kPast + emitted);
    if (!ranges) return Error(ranges.error());
    std::vector<llmp::catalog::ExtentId> tail_extents;
    void* tail_pinned = nullptr;
    if (speculate) {
      auto pinned =
          node_.Pinned(std::uint64_t{runner_.profile().width} * sizeof(float), 0, tail_extents);
      if (!pinned) return Error(pinned.error());
      tail_pinned = *pinned;
    }
    // Retain the catalog owner on any refusal; only successful completion frees it.
    for (unsigned round = 0; round < 5; ++round) {
      const bool quality = round == 1 || round == 2, timed = round >= 3;
      const auto name = round == 0   ? "warm"
                        : round == 1 ? "quality-first"
                        : round == 2 ? "quality-repeat"
                        : round == 3 ? "timed-first"
                                     : "timed-repeat";
      const auto dir = out_ / name;
      std::error_code ec;
      if (!std::filesystem::create_directory(dir, ec) || ec)
        return Error("bounded32 chronology directory must be new");
      prediction_heads.clear();
      verify_heads.clear();
      verify_features.clear();
      last_draft_heads.clear();
      if (auto r = Prefix(frontier); !r) return r;
      if (auto r = runner_.ReserveStateThrough(0, kPast + emitted); !r) return r;
      if (round == 0) {
        const auto& policy = runner_.last_built_policy();
        std::cout << "GREEDY32_POLICY phase=prefill last_built_rows=" << policy.rows
                  << " norm_rope=" << policy.norm_rope << " norm_add=" << policy.norm_add << '\n';
      }
      if (speculate) {
        if (auto r = Feature(kPast - 1, 1, initial_feature); !r) return r;
        result.head.assign(frontier.begin(), frontier.end());
      }
      Record record;
      auto append = [](std::vector<float>& out, std::span<const float> values) {
        out.insert(out.end(), values.begin(), values.end());
      };
      const auto started = std::chrono::steady_clock::now();
      while (record.count < emitted) {
        const auto remaining = emitted - record.count;
        const auto past = kPast + record.count;
        auto& head = speculate ? result.head : frontier;
        if (quality) append(prediction_heads, head);
        const auto depth = speculate && remaining >= 2 ? std::min(3U, remaining - 1) : 0;
        std::uint32_t keep = 1;
        if (depth != 0) {
          if (auto r = assistant_->GreedyUnit(0, past, depth, head, workspace, result); !r)
            return r;
          keep = result.count;
          if (keep == 0 || keep > depth + 1 || keep > remaining || result.past != past + keep ||
              workspace.verify_heads.size() != (depth + 1) * kVocab ||
              workspace.verify_features.size() != (depth + 1) * runner_.profile().width)
            return Error("bounded32 completed unit envelope changed");
          if (quality) {
            append(verify_heads, workspace.verify_heads);
            append(verify_features, workspace.verify_features);
            append(last_draft_heads, workspace.draft_head);
            for (std::uint32_t row = 1; row < keep; ++row)
              append(prediction_heads,
                     std::span(workspace.verify_heads).subspan((row - 1) * kVocab, kVocab));
          }
          for (std::uint32_t row = 0; row < keep; ++row)
            record.tokens[record.count + row] = result.committed[row];
        } else {
          const auto anchor = llmp::execution::Greedy(head);
          if (!anchor) return Error("bounded32 target frontier refused");
          const std::array<std::int32_t, 1> input{*anchor};
          const en::Gemma4Runner::Work work{0, past, input, &head};
          if (auto r = runner_.Wave(std::span(&work, 1)); !r) return r;
          if (speculate) {
            if (auto r = runner_.CopyFeatures(0, past, 1, tail_pinned); !r) {
              node_.KeepPinned(tail_pinned);
              return r;
            }
            const auto* copied = static_cast<const float*>(tail_pinned);
            result.feature.assign(copied, copied + runner_.profile().width);
            if (quality) {
              append(verify_heads, result.head);
              append(verify_features, result.feature);
            }
          }
          record.tokens[record.count] = *anchor;
        }

        if (round == 0 && record.unit_count == 0) {
          const auto& policy = runner_.last_built_policy();
          std::cout << "GREEDY32_POLICY phase=first_decode last_built_rows=" << policy.rows
                    << " norm_rope=" << policy.norm_rope << " norm_add=" << policy.norm_add << '\n';
        }
        for (std::uint32_t row = 0; row < keep; ++row)
          record.eos_count += record.tokens[record.count + row] == eos;
        record.units[record.unit_count++] = {
            .past = past, .depth = depth, .rows = depth + 1, .keep = keep};
        record.count += keep;
        record.verified += depth + 1;
        record.drafted += depth;
      }
      // All model work, job retirement and normal result publication are paid.
      const auto finished = std::chrono::steady_clock::now();
      if (timed) record.seconds = en::support::Seconds(finished - started);
      if (auto live = runner_.request_slot(0);
          !live || (*live)->completed_positions() != kPast + emitted || !(*live)->state_usable())
        return Error("bounded32 completed endpoint was not published");
      const auto& final = speculate ? result.head : frontier;
      if (!std::ranges::all_of(final, [](float x) { return std::isfinite(x); }) ||
          (speculate &&
           !std::ranges::all_of(result.feature, [](float x) { return std::isfinite(x); })))
        return Error("bounded32 final outputs are not finite");
      auto state = Hash(*ranges);
      if (!state) return Error(state.error());
      // Witness hashing and archival copies/writes never enter the paid loop.
      if (round == 0) continue;
      if (auto r = Write<std::int32_t>(dir / "tokens.i32", record.tokens); !r) return r;
      if (auto r = Write<float>(dir / "final-head.f32", final); !r) return r;
      if (speculate)
        if (auto r = Write<float>(dir / "final-feature.f32", result.feature); !r) return r;
      if (quality) {
        if (auto r = Write<float>(dir / "prediction-heads.f32", prediction_heads); !r) return r;
        if (speculate) {
          if (auto r = Write<float>(dir / "initial-feature.f32", initial_feature); !r) return r;
          if (auto r = Write<float>(dir / "verify-heads.f32", verify_heads); !r) return r;
          if (auto r = Write<float>(dir / "verify-features.f32", verify_features); !r) return r;
          if (auto r = Write<float>(dir / "last-draft-heads.f32", last_draft_heads); !r) return r;
        }
      }
      std::string units = "[";
      for (std::uint32_t i = 0; i < record.unit_count; ++i) {
        const auto& u = record.units[i];
        if (i != 0) units += ',';
        units += "{\"past\":" + std::to_string(u.past) + ",\"depth\":" + std::to_string(u.depth) +
                 ",\"verified_rows\":" + std::to_string(u.rows) +
                 ",\"keep\":" + std::to_string(u.keep) + "}";
      }
      units += ']';
      const auto metadata =
          "{\"mode\":\"" + std::string(speculate ? "unit32" : "plain32") + "\",\"phase\":\"" +
          (quality ? "quality" : "timed") +
          "\",\"emitted\":32,\"endpoint\":96,\"prefill_queries\":64,\"max_rows\":128,"
          "\"context\":4096,\"eog_ignored\":true,\"eos_id\":" +
          std::to_string(eos) + ",\"eos_occurrences\":" + std::to_string(record.eos_count) +
          ",\"verified_rows\":" + std::to_string(record.verified) +
          ",\"drafted_rows\":" + std::to_string(record.drafted) +
          ",\"seconds\":" + std::to_string(record.seconds) +
          ",\"actual_vector_capacity_bytes\":" + std::to_string(capacities) +
          ",\"semantic_state_sha256\":\"" + *state + "\",\"units\":" + units + "}\n";
      if (auto r = Text(dir / "completion.json", metadata); !r) return r;
      std::cout << "GREEDY32_COMPLETED mode=" << (speculate ? "unit32" : "plain32")
                << " phase=" << name << " emitted=32 endpoint=96 verified=" << record.verified
                << " drafted=" << record.drafted << " seconds=" << record.seconds << '\n';
    }
    if (tail_pinned)
      if (auto r = node_.FreePinned(tail_pinned); !r) return r;
    return {};
  }

 private:
  en::PagedNode& node_;
  en::Gemma4Runner& runner_;
  en::Gemma4Assistant* assistant_;
  std::span<const std::int32_t> ids_;
  std::filesystem::path out_;
};
}  // namespace

int main(int argc, char** argv) {
  umask(0077);
  if (argc != 7 || (std::string_view(argv[4]) != "26" && std::string_view(argv[4]) != "31") ||
      (std::string_view(argv[6]) != "unit" && std::string_view(argv[6]) != "teacher" &&
       std::string_view(argv[6]) != "plain32" && std::string_view(argv[6]) != "unit32"))
    return 2;
  const bool profile31 = std::string_view(argv[4]) == "31";
  const bool repeated = std::string_view(argv[6]).ends_with("32");
  const bool plain = std::string_view(argv[6]) == "plain32";
  if (repeated && !profile31) return 2;
  const auto host = repeated && !plain ? (256ULL << 20U) : kHost;
  std::int32_t eos = -1;
  std::array<std::int32_t, 1024> ids{};
  std::ifstream input(argv[5], std::ios::binary);
  if (!input.read(reinterpret_cast<char*>(ids.data()), sizeof(ids)) ||
      input.peek() != std::char_traits<char>::eof() || ids[0] != 2 ||
      !std::ranges::all_of(ids, [](auto id) { return id >= 0 && id < static_cast<int>(kVocab); }) ||
      llmp::base::ToHex(llmp::base::Sha256{}.Update(std::as_bytes(std::span(ids))).Finish()) !=
          "b2d7aaf6aa2ef06d82591a3794f36640e192f429539ec934fd74bf4d81df1610")
    return 2;
  const std::filesystem::path out = argv[3];
  std::error_code ec;
  if (!std::filesystem::create_directory(out, ec) || ec) return 2;
  if (!Write<std::int32_t>(out / "inputs.i32", ids)) return 2;
  auto life = std::make_unique<Lifetime>();
  auto& node = life->node;
  life->runner = std::make_unique<en::Gemma4Runner>(
      node,
      en::Gemma4Options{.artifact = argv[1],
                        .out = out,
                        .variant = profile31 ? en::Gemma4Variant::k31B : en::Gemma4Variant::k26BA4B,
                        .context = 4096,
                        .max_rows = 128,
                        .slots = 1,
                        .max_head_rows = plain ? 1U : 4U,
                        .max_verify_rows = plain ? 0U : 4U,
                        .retain_features = !plain,
                        .fuse_norm_rope = profile31,
                        .fuse_norm_add = profile31},
      0, 0);
  auto& runner = *life->runner;
  const auto execute = [&]() -> en::Status {
    if (auto r = node.Open(); !r) return r;
    life->entered.push_back(&runner);
    if (auto r = runner.Setup(); !r) return r;
    en::Gemma4Assistant* assistant = nullptr;
    {
      if (!node.ChargeHost(kAdmission, false)) return Error("metadata admission refused");
      struct Grant {
        en::PagedNode& node;
        ~Grant() { node.UnchargeHost(kAdmission); }
      } grant{node};
      auto ta = ar::Artifact::Open(argv[1]);
      if (!ta) return Error("target artifact metadata open failed");
      auto tv = Decode(*ta, profile31 ? "gemma-4-31B-it-UD-Q4_K_XL.kv.gguf"
                                      : "gemma-4-26B-A4B-it-UD-Q4_K_M.kv.gguf");
      if (!tv) return Error("target vocabulary metadata refused");
      eos = tv->tokenizer.spec.eos.value_or(-1);
      if (repeated && eos != 106) return Error("approved target EOS differs");
      if (!plain) {
        auto aa = ar::Artifact::Open(argv[2]);
        if (!aa) return Error("assistant artifact metadata open failed");
        auto av = Decode(
            *aa, profile31 ? "mtp-gemma-4-31B-it.kv.gguf" : "mtp-gemma-4-26B-A4B-it.kv.gguf");
        if (!av) return Error("assistant vocabulary metadata refused");
        auto created = runner.SetupAssistant(argv[2], tv->view(), av->view());
        if (!created) return Error(created.error());
        assistant = *created;
      }
    }
    if (auto r = node.MapWorkspace(runner.activations_needed(), runner.pool_needed()); !r) return r;
    const auto fixed = node.catalog().OccupancyOf(node.domain()).Total().value();
    node.SetHostFloor(runner.host_input_bytes() + runner.plan_floor_bytes() + host);
    if (auto r = node.Start(llmp::base::Bytes(fixed + runner.weights().size() * en::kPagedExtent +
                                              2 * node.StateCapacity() + host));
        !r)
      return r;
    if (auto r = runner.Register(); !r) return r;
    if (auto r = runner.Bind(); !r) return r;
    node.Run();
    if (!node.ChargeHost(host, false)) return Error("caller publication funding refused");
    struct Grant {
      en::PagedNode& node;
      std::uint64_t bytes;
      ~Grant() { node.UnchargeHost(bytes); }
    } grant{node, host};
    if (auto r = runner.SelectSlots(std::array<std::uint32_t, 1>{0}); !r) return r;
    Screen screen(*life, assistant, ids, out);
    return node.WithRequest(0, runner.closure(), "Gemma C1/P64 greedy diagnostic", [&] {
      return repeated ? screen.Run32(!plain, eos) : screen.Run(argv[6]);
    });
  };
  const auto done = execute();
  const auto retired = node.TearDown(life->entered);
  if (!done || !retired) {
    std::cerr << "REFUSAL " << (!done ? done.error() : retired.error()) << '\n';
    if (!retired) std::ignore = life.release();
    return 1;
  }
  if (!Text(out / "retired.json", repeated ? "{\"successful_teardown\":true,\"warm\":1,\"quality_"
                                             "rounds\":2,\"timed_rounds\":2}\n"
                                           : "{\"successful_teardown\":true,\"rounds\":2}\n"))
    return 1;
  if (repeated)
    std::cout << "GREEDY_NATIVE_RETIRED profile=" << argv[4] << " mode=" << argv[6]
              << " C=1 P=64 prefill_query=64 emitted=32 endpoint=96 warm=1 quality=2 timed=2\n";
  else
    std::cout << "GREEDY_NATIVE_RETIRED profile=" << argv[4] << " mode=" << argv[6]
              << " C=1 P=64 prefill_query=64 depth=3 verify_rows=4 rounds=2\n";
}
