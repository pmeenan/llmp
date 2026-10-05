// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// Exact integer-token native Gemma control. No implicit BOS/template.
// ARTIFACT OUTPUT_DIR TOKENS_CSV STEPS [SLOTS] [ordinary|norm|q8|norm-q8]
// [teacher|warm|control] [host|device] [ordinary-prefix-checkpoint]
#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "base/bytes.h"
#include "engine/gemma4_runner.h"
#include "engine/support.h"

namespace en = jitllm::engine;
using en::support::Error;
std::expected<std::uint32_t, std::string> Number(std::string_view text) {
  std::uint32_t value = 0;
  const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
  if (error != std::errc{} || end != text.data() + text.size()) return Error("invalid number");
  return value;
}
int main(int argc, char** argv) {
  if (argc < 5 || argc > 10) return 2;
  auto steps = Number(argv[4]);
  auto slots = Number(argc >= 6 ? argv[5] : "1");
  if (!steps || !slots || *steps > 256) return 2;
  const std::string_view requested_policy = argc >= 7 ? argv[6] : "ordinary";
  if (requested_policy != "ordinary" && requested_policy != "norm" && requested_policy != "q8" &&
      requested_policy != "norm-q8")
    return 2;
  std::vector<std::int32_t> tokens;
  std::string_view text(argv[3]);
  while (!text.empty()) {
    const auto comma = text.find(',');
    auto id = Number(text.substr(0, comma));
    if (!id || *id > INT32_MAX) return 2;
    tokens.push_back(static_cast<std::int32_t>(*id));
    if (comma == std::string_view::npos) break;
    text.remove_prefix(comma + 1);
  }
  std::filesystem::path out(argv[2]);
  std::error_code error;
  std::filesystem::create_directories(out, error);
  if (error) return 2;
  en::PagedNode node({.slot_bytes = en::kSlabSlotBytes});
  en::Gemma4Runner runner(
      node,
      {.artifact = argv[1],
       .out = out,
       .slots = *slots,
       .reference_masks = argc >= 9 && std::string_view(argv[8]) == "host",
       .shared_q8 = requested_policy == "q8" || requested_policy == "norm-q8",
       .fuse_norms = requested_policy == "norm" || requested_policy == "norm-q8"},
      0, 0);
  const bool teacher = argc >= 8 && std::string_view(argv[7]) == "teacher";
  const bool warm = argc >= 8 && std::string_view(argv[7]) == "warm";
  const bool restore_prefix = argc == 10;
  // This diagnostic uses the already verified literal's first greedy ID.
  // Check its exact input domain instead of guessing a head after restore.
  if (restore_prefix &&
      (teacher || warm || tokens != std::vector<std::int32_t>{2, 818, 5279, 529, 7001, 563}))
    return 2;
  std::vector<en::PagedModel*> entered;
  const auto execute = [&]() -> en::Status {
    if (auto r = node.Open(); !r) return r;
    entered.push_back(&runner);
    if (auto r = runner.Setup(); !r) return r;
    std::cout << "GEMMA_ENVELOPE activations=" << runner.activations_needed()
              << " scratch=" << runner.pool_needed() << " host_inputs=" << runner.host_input_bytes()
              << " plans=" << runner.plan_floor_bytes() << " weights=" << runner.weight_bytes()
              << " slab_padding=" << runner.slab_padding()
              << " pitch_padding=" << runner.pitch_padding() << '\n';
    if (auto r = node.MapWorkspace(runner.activations_needed(), runner.pool_needed()); !r) return r;
    const auto fixed = node.catalog().OccupancyOf(node.domain()).Total().value();
    const auto budget =
        fixed + runner.weights().size() * en::kPagedExtent + 2 * node.StateCapacity();
    node.SetHostFloor(runner.plan_floor_bytes() + runner.host_input_bytes());
    if (auto r = node.Start(jitllm::base::Bytes(budget)); !r) return r;
    if (auto r = runner.Register(); !r) return r;
    if (auto r = runner.Bind(); !r) return r;
    node.Run();
    return node.WithRequest(
        0, runner.closure(), "Gemma4 representative control", [&]() -> en::Status {
          std::vector<float> logits;
          std::uint32_t past = static_cast<std::uint32_t>(tokens.size());
          const auto checkpoint = [&](const std::filesystem::path& path,
                                      bool restore) -> en::Status {
            const auto tag_path = std::filesystem::path(path.string() + ".layout");
            std::string source_layout;
            if (restore) {
              std::error_code tag_error;
              const auto tag_bytes = std::filesystem::file_size(tag_path, tag_error);
              if (tag_error || tag_bytes == 0 || tag_bytes > 128)
                return Error("prefix checkpoint source layout missing or unbounded");
              std::ifstream tag(tag_path, std::ios::binary);
              source_layout.resize(tag_bytes);
              tag.read(source_layout.data(), static_cast<std::streamsize>(tag_bytes));
              if (!tag || tag.peek() != std::ifstream::traits_type::eof() ||
                  source_layout != runner.CheckpointLayoutId())
                return Error("prefix checkpoint source layout differs");
            } else {
              source_layout = runner.CheckpointLayoutId();
            }
            auto ranges = runner.CheckpointRanges(past);
            if (!ranges) return Error(ranges.error());
            std::uint64_t count = 0;
            for (const auto& range : *ranges) count += range.bytes;
            if (restore) {
              std::error_code size_error;
              if (std::filesystem::file_size(path, size_error) != count || size_error)
                return Error("prefix checkpoint length differs");
            }
            std::vector<jitllm::catalog::ExtentId> staging;
            auto saved = node.Pinned(count, 0, staging);
            if (!saved) return Error(saved.error());
            if (restore) {
              std::ifstream file(path, std::ios::binary);
              file.read(static_cast<char*>(*saved), static_cast<std::streamsize>(count));
              if (!file || file.peek() != std::ifstream::traits_type::eof()) {
                auto freed = node.FreePinned(*saved);
                if (!freed) return freed;
                return Error("reading prefix checkpoint failed");
              }
            }
            en::LiveState::CopyRetirement retirement = en::LiveState::CopyRetirement::kProven;
            auto copied = restore ? runner.RestoreCheckpoint(0, past, *saved, *ranges,
                                                             source_layout, &retirement)
                                  : runner.CopyState(0, *saved, *ranges, true, &retirement);
            if (retirement == en::LiveState::CopyRetirement::kUnproven) {
              node.KeepPinned(*saved);
              return Error("initialized-state copy completion unproven");
            }
            bool written = restore;
            if (copied && !restore) {
              std::ofstream file(path, std::ios::binary);
              file.write(static_cast<const char*>(*saved), static_cast<std::streamsize>(count));
              file.flush();
              written = bool(file);
              std::ofstream tag(tag_path, std::ios::binary);
              tag.write(source_layout.data(), static_cast<std::streamsize>(source_layout.size()));
              tag.flush();
              written = written && bool(tag);
            }
            auto freed = node.FreePinned(*saved);
            if (!copied) return copied;
            if (!freed) return freed;
            if (!written) return Error("writing initialized state failed");
            std::cout << "GEMMA_STATE past=" << past << " bytes=" << count
                      << " restored=" << restore << '\n';
            return {};
          };
          if (restore_prefix) {
            if (auto r = checkpoint(argv[9], true); !r) return r;
            const std::int32_t first_gold = 45518;
            if (auto r = runner.Chunk(past++, std::span(&first_gold, 1), logits); !r) return r;
            std::cout << "GEMMA_RESTORED_PREFIX appended=" << first_gold << " past=" << past
                      << '\n';
          } else {
            if (auto r = runner.Chunk(0, tokens, logits, teacher); !r) return r;
            if (!teacher && !warm)
              if (auto r = checkpoint(out / "initialized-prefix.bin", false); !r) return r;
          }
          if (teacher) {
            std::ofstream file(out / "teacher.f32", std::ios::binary);
            file.write(reinterpret_cast<const char*>(logits.data()),
                       static_cast<std::streamsize>(logits.size() * sizeof(float)));
            return file ? en::Status{} : en::Status(Error("writing teacher logits failed"));
          }
          if (warm) {
            // Exercise stable-address capture/replay before timing, then
            // restore the literal input so every bookend has identical IDs.
            for (std::uint32_t i = 0; i < 8; ++i) {
              const auto next = static_cast<std::int32_t>(
                  std::max_element(logits.begin(), logits.end()) - logits.begin());
              if (auto r = runner.Chunk(past++, std::span(&next, 1), logits); !r) return r;
            }
            if (auto r = runner.Clear(); !r) return r;
            if (auto r = runner.Chunk(0, tokens, logits); !r) return r;
            past = static_cast<std::uint32_t>(tokens.size());
            // Reset prefill displaces the comparator's cached one-row graph.
            // CUDA also needs a second stable call to capture. Three common
            // untimed rows cover build, capture and replay before timing.
            for (std::uint32_t i = 0; i < 3; ++i) {
              const auto seed = static_cast<std::int32_t>(
                  std::max_element(logits.begin(), logits.end()) - logits.begin());
              if (auto r = runner.Chunk(past++, std::span(&seed, 1), logits); !r) return r;
              std::cout << "GEMMA_TIMED_PREFIX appended=" << seed << " past=" << past << '\n';
            }
          }
          std::array<std::int32_t, 256> chosen{};
          const auto started = std::chrono::steady_clock::now();
          for (std::uint32_t i = 0; i < *steps; ++i) {
            auto best = std::max_element(logits.begin(), logits.end());
            const auto next = static_cast<std::int32_t>(best - logits.begin());
            chosen[i] = next;
            if (warm) {
              if (auto r = runner.Chunk(past++, std::span(&next, 1), logits); !r) return r;
              continue;
            }
            std::cout << "GEMMA_TOKEN step=" << i << " id=" << next << " logit=" << *best << '\n';
            std::ofstream file(out / ("logits-" + std::to_string(i) + ".f32"), std::ios::binary);
            file.write(reinterpret_cast<const char*>(logits.data()),
                       static_cast<std::streamsize>(logits.size() * sizeof(float)));
            if (!file) return Error("writing logits failed");
            if (i + 1 != *steps) {
              if (auto r = runner.Chunk(past++, std::span(&next, 1), logits); !r) return r;
            }
          }
          const auto elapsed = en::support::Seconds(std::chrono::steady_clock::now() - started);
          std::cout << (warm ? "GEMMA_WARM" : "GEMMA_CONTROL") << " seconds=" << elapsed
                    << " completed_chunks=" << (warm ? *steps : (*steps ? *steps - 1 : 0))
                    << " captured=" << runner.graph_stats().captured
                    << " replayed=" << runner.graph_stats().replayed << '\n';
          if (warm) {
            for (std::uint32_t i = 0; i < *steps; ++i)
              std::cout << "GEMMA_WARM_TOKEN step=" << i << " id=" << chosen[i] << '\n';
          }
          if (!warm)
            if (auto r = checkpoint(out / "initialized-state.bin", false); !r) return r;
          const auto& policy = runner.last_built_policy();
          std::cout << "GEMMA_POLICY rows=" << policy.rows << " segments=" << policy.segments
                    << " norm_fused=" << policy.norm_fused << " rope_store=" << policy.rope_store
                    << " shared_vecq=" << policy.shared_vecq
                    << " row_products=" << policy.row_products
                    << " lane_steps=" << policy.lane_steps << '\n';
          return {};
        });
  };
  auto ran = execute();
  const auto retired = node.TearDown(entered);
  if (!ran) std::cerr << ran.error() << '\n';
  if (!retired) std::cerr << retired.error() << '\n';
  return ran && retired ? 0 : 1;
}
