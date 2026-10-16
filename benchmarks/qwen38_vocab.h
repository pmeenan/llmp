// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#ifndef LLMP_BENCHMARKS_QWEN38_VOCAB_H_
#define LLMP_BENCHMARKS_QWEN38_VOCAB_H_

#include <cstdint>
#include <expected>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace llmp::benchmarks::draft_vocab {

inline constexpr std::uint32_t kContext = 33792;
inline constexpr std::uint32_t kDepth = 3;
inline constexpr std::uint32_t kAnchors = 4;
// The state capacities the prepared manifests record at kContext with
// 8,192-row prompt chunks: Qwen3.8 Flash Next's (the ModelOpt target) and
// its MTP drafter's, without the serving composition's draft-progress
// record (as the M3 runner laid them out). The preparer opens no artifact:
// components_qwen38_models_test checks them against components::StageLayout
// of the published v1 target's stored spec composed with its drafter.
inline constexpr std::uint64_t kTargetStateBytes = 1182081024;
inline constexpr std::uint64_t kMtpStateBytes = 424255488;

// Private benchmark inputs. The continuation is frozen conditioning text;
// its IDs are never the candidate's target-frequency labels.
struct Example {
  std::string id;
  std::string domain;
  std::string split;
  std::uint32_t nominal_tokens = 0;
  std::uint32_t stable_boundary = 0;
  std::vector<std::int32_t> prompt;
  std::vector<std::int32_t> continuation;
  std::vector<std::uint32_t> anchors;
};

std::expected<void, std::string> CheckExamples(std::span<const Example> examples,
                                               std::uint32_t context, std::uint32_t vocab);
std::expected<std::vector<Example>, std::string> ReadExamples(std::string_view json,
                                                              std::uint32_t context,
                                                              std::uint32_t vocab);

struct Preparation {
  std::filesystem::path source;
  std::filesystem::path tokenizer;
  std::filesystem::path chat_template;
  std::filesystem::path stop_metadata;
  std::filesystem::path output;
};
// CPU-only native rendering/tokenization. No node, provider or model is
// constructed. Both split manifests freeze before any model output.
std::expected<void, std::string> Prepare(const Preparation& paths);

}  // namespace llmp::benchmarks::draft_vocab

#endif  // LLMP_BENCHMARKS_QWEN38_VOCAB_H_
