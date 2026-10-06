// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// Manual C1/P64/depth3 target+assistant transaction, never CTest or serving.
#include <sys/stat.h>

#include <algorithm>
#include <array>
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
namespace en = jitllm::engine;
namespace ar = jitllm::artifact;
namespace md = jitllm::model;
namespace tk = jitllm::tokenizer;
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
  Screen(Lifetime& life, en::Gemma4Assistant& assistant, std::span<const std::int32_t> ids,
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
    if (bytes == 0) return jitllm::base::ToHex(jitllm::base::Sha256{}.Finish());
    std::vector<jitllm::catalog::ExtentId> extents;
    auto pinned = node_.Pinned(bytes, 0, extents);
    if (!pinned) return Error(pinned.error());
    if (auto r = runner_.CopyState(0, *pinned, ranges, true); !r) {
      node_.KeepPinned(*pinned);
      return Error(r.error());
    }
    const auto result =
        jitllm::base::ToHex(jitllm::base::Sha256{}
                                .Update(std::span(static_cast<const std::byte*>(*pinned), bytes))
                                .Finish());
    if (auto r = node_.FreePinned(*pinned); !r) return Error(r.error());
    return result;
  }
  en::Status Feature(std::uint32_t first, std::uint32_t rows, std::vector<float>& out) {
    const auto count = std::uint64_t{rows} * runner_.profile().width;
    std::vector<jitllm::catalog::ExtentId> extents;
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
      auto anchor = jitllm::execution::Greedy(initial);
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
            if (auto r = assistant_.Step(borrows, token, step == 0, heads, features); !r) return r;
            if (draft_head.size() != kVocab || draft_feature.size() != width ||
                !Exact(draft_head, draft_head) || !Exact(draft_feature, draft_feature))
              return Error("draft output differs");
            draft_heads.insert(draft_heads.end(), draft_head.begin(), draft_head.end());
            draft_features.insert(draft_features.end(), draft_feature.begin(), draft_feature.end());
            auto next = jitllm::execution::Greedy(draft_head);
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
        auto next = jitllm::execution::Greedy(std::span(control).last(kVocab));
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
        if (auto r = assistant_.GreedyUnit(0, kPast, 3, initial, workspace, result); !r) return r;
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

 private:
  en::PagedNode& node_;
  en::Gemma4Runner& runner_;
  en::Gemma4Assistant& assistant_;
  std::span<const std::int32_t> ids_;
  std::filesystem::path out_;
};
}  // namespace

int main(int argc, char** argv) {
  umask(0077);
  if (argc != 7 || (std::string_view(argv[4]) != "26" && std::string_view(argv[4]) != "31") ||
      (std::string_view(argv[6]) != "unit" && std::string_view(argv[6]) != "teacher"))
    return 2;
  const bool profile31 = std::string_view(argv[4]) == "31";
  std::array<std::int32_t, 1024> ids{};
  std::ifstream input(argv[5], std::ios::binary);
  if (!input.read(reinterpret_cast<char*>(ids.data()), sizeof(ids)) ||
      input.peek() != std::char_traits<char>::eof() || ids[0] != 2 ||
      !std::ranges::all_of(ids, [](auto id) { return id >= 0 && id < static_cast<int>(kVocab); }) ||
      jitllm::base::ToHex(jitllm::base::Sha256{}.Update(std::as_bytes(std::span(ids))).Finish()) !=
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
                        .max_head_rows = 4,
                        .max_verify_rows = 4,
                        .retain_features = true,
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
      auto ta = ar::Artifact::Open(argv[1]), aa = ar::Artifact::Open(argv[2]);
      if (!ta || !aa) return Error("artifact metadata open failed");
      auto tv = Decode(*ta, profile31 ? "gemma-4-31B-it-UD-Q4_K_XL.kv.gguf"
                                      : "gemma-4-26B-A4B-it-UD-Q4_K_M.kv.gguf");
      auto av =
          Decode(*aa, profile31 ? "mtp-gemma-4-31B-it.kv.gguf" : "mtp-gemma-4-26B-A4B-it.kv.gguf");
      if (!tv || !av) return Error("vocabulary metadata refused");
      auto created = runner.SetupAssistant(argv[2], tv->view(), av->view());
      if (!created) return Error(created.error());
      assistant = *created;
    }
    if (auto r = node.MapWorkspace(runner.activations_needed(), runner.pool_needed()); !r) return r;
    const auto fixed = node.catalog().OccupancyOf(node.domain()).Total().value();
    node.SetHostFloor(runner.host_input_bytes() + runner.plan_floor_bytes() + kHost);
    if (auto r = node.Start(jitllm::base::Bytes(fixed + runner.weights().size() * en::kPagedExtent +
                                                2 * node.StateCapacity() + kHost));
        !r)
      return r;
    if (auto r = runner.Register(); !r) return r;
    if (auto r = runner.Bind(); !r) return r;
    node.Run();
    if (!node.ChargeHost(kHost, false)) return Error("caller publication funding refused");
    struct Grant {
      en::PagedNode& node;
      ~Grant() { node.UnchargeHost(kHost); }
    } grant{node};
    if (auto r = runner.SelectSlots(std::array<std::uint32_t, 1>{0}); !r) return r;
    Screen screen(*life, *assistant, ids, out);
    return node.WithRequest(0, runner.closure(), "Gemma C1/P64 greedy diagnostic",
                            [&] { return screen.Run(argv[6]); });
  };
  const auto done = execute();
  const auto retired = node.TearDown(life->entered);
  if (!done || !retired) {
    std::cerr << "REFUSAL " << (!done ? done.error() : retired.error()) << '\n';
    if (!retired) std::ignore = life.release();
    return 1;
  }
  if (!Text(out / "retired.json", "{\"successful_teardown\":true,\"rounds\":2}\n")) return 1;
  std::cout << "GREEDY_NATIVE_RETIRED profile=" << argv[4] << " mode=" << argv[6]
            << " C=1 P=64 prefill_query=64 depth=3 verify_rows=4 rounds=2\n";
}
