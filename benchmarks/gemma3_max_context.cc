// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// One internal trained-maximum teacher pass through the ordinary server's
// reclamation policy. No whole-state snapshot or public admission change.
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <format>
#include <fstream>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "base/sha256.h"
#include "config/node_config.h"
#include "config/storage_roles.h"
#include "runtime/serving.h"
#include "tokenizer/gguf.h"
#include "tokenizer/tokenizer.h"

namespace {
namespace rt = llmp::runtime;
namespace fs = std::filesystem;
constexpr std::uint32_t kContext = 131072, kPrefix = 131008, kSteps = 64, kVocab = 262208;
std::unexpected<std::string> Error(std::string text) { return std::unexpected(std::move(text)); }
std::expected<std::string, std::string> Read(const fs::path& path, std::uint64_t cap) {
  std::error_code error;
  const auto size = fs::file_size(path, error);
  if (error || size == 0 || size > cap) return Error("bounded input refused");
  std::string data(static_cast<std::size_t>(size), '\0');
  std::ifstream file(path, std::ios::binary);
  if (!file.read(data.data(), static_cast<std::streamsize>(size)) ||
      file.peek() != std::char_traits<char>::eof())
    return Error("complete input read failed");
  return data;
}
template <class T>
std::string Hash(std::span<const T> values) {
  return llmp::base::ToHex(llmp::base::Sha256{}.Update(std::as_bytes(values)).Finish());
}
template <class T>
rt::Status Write(const fs::path& path, std::span<const T> values) {
  std::ofstream file(path, std::ios::binary | std::ios::noreplace);
  file.write(reinterpret_cast<const char*>(values.data()),
             static_cast<std::streamsize>(values.size_bytes()));
  file.flush();
  return file ? rt::Status{} : Error("exclusive output failed");
}
std::expected<std::int32_t, std::string> Best(std::span<const float> row) {
  if (row.size() != kVocab || !std::ranges::all_of(row, [](float x) { return std::isfinite(x); }))
    return Error("full finite head required");
  return static_cast<std::int32_t>(std::max_element(row.begin(), row.end()) - row.begin());
}
rt::Status Prepare(std::span<const std::string_view> args) {
  auto metadata = Read(fs::path(args[0]), 32ULL << 20U);
  auto text = Read(fs::path(args[1]), 8ULL << 20U);
  if (!metadata || !text) return Error("preparation inputs unavailable");
  auto parsed = llmp::tokenizer::ReadGgufTokenizer(std::as_bytes(std::span(*metadata)));
  if (!parsed) return Error(parsed.error().ToString());
  if (parsed->spec.tokens.size() != kVocab || parsed->spec.bos != 2 || !parsed->spec.add_bos ||
      parsed->spec.add_eos)
    return Error("Gemma3 vocabulary/BOS contract differs");
  auto tokenizer = llmp::tokenizer::Tokenizer::Create(std::move(parsed->spec));
  if (!tokenizer) return Error(tokenizer.error().ToString());
  std::vector<std::int32_t> ids;
  if (auto encoded = tokenizer->Encode(*text, {.add_bos_eos = true, .max_tokens = 2000000}, ids);
      !encoded)
    return Error(encoded.error().ToString());
  if (ids.size() < kContext || ids.front() != 2) return Error("corpus supplies too few IDs");
  if (!fs::create_directory(fs::path(args[2]))) return Error("input output must be new");
  const auto selected = std::span<const std::int32_t>(ids).first(kContext);
  if (auto written = Write(fs::path(args[2]) / "ids.i32", selected); !written) return written;
  (void)std::printf("GEMMA3_MAX_INPUT rows=%u ids_sha256=%s text_sha256=%s\n", kContext,
                    Hash(selected).c_str(), Hash(std::span<const char>(*text)).c_str());
  return {};
}
rt::Status Run(std::span<const std::string_view> args) {
  const fs::path out(args[4]);
  if (!fs::create_directory(out)) return Error("output must be new");
  auto bytes = Read(fs::path(args[3]), kContext * sizeof(std::int32_t));
  if (!bytes || bytes->size() != kContext * sizeof(std::int32_t))
    return Error("exact maximum input required");
  std::vector<std::int32_t> ids(kContext);
  std::memcpy(ids.data(), bytes->data(), bytes->size());
  if (ids.front() != 2 || std::ranges::any_of(ids, [](auto id) {
        return id < 0 || id >= static_cast<std::int32_t>(kVocab);
      }))
    return Error("invalid corpus IDs");
  auto config = llmp::config::LoadNodeConfig({.main_file = fs::path(args[0]),
                                              .main_file_optional = false,
                                              .anchor = fs::path(args[1]),
                                              .trusted_uid = getuid()});
  if (!config) return Error("invalid private node configuration");
  if (config->models.size() != 1) return Error("one configured model required");
  auto roles = llmp::config::PrepareRuntimeRoles(config->storage, getuid(), fs::path(args[1]));
  if (!roles) return Error(roles.error().front());
  rt::ServingOptions options;
  options.plain = true;
  options.gemma3_trained_max = true;
  // Optional words after the input and output: device-masks; legacy (no
  // prefill lookahead or first-run capture) or lookahead-only; repeat (one
  // earlier timed traversal of the same prefix, then Clear: the measured
  // pass runs on its plans and graphs); no-zero (fresh state read from the
  // spill file's holes) and no-reserve (no handle reserve), the pager's
  // matched controls.
  bool repeat = false;
  options.gemma3_device_masks = false;
  for (const auto word : args.subspan(5)) {
    if (word == "device-masks") {
      options.gemma3_device_masks = true;
    } else if (word == "legacy") {
      options.gemma3_prefill_lookahead = options.gemma3_capture_ahead = false;
    } else if (word == "lookahead-only") {
      options.gemma3_capture_ahead = false;
    } else if (word == "repeat") {
      repeat = true;
    } else if (word == "no-zero") {
      options.zero_state = false;
    } else if (word == "no-reserve") {
      options.handle_reserve = 0;
    } else {
      return Error("unknown run option");
    }
  }
  rt::Server server(*config, *roles, options, stderr);
  if (auto started = server.Start(false); !started) return started;
  auto* target = server.Find(args[2]);
  if (target == nullptr || !target->llm()) return Error("configured Gemma3 missing");
  auto& llm = static_cast<rt::Llm&>(*target);
  if (llm.context() != kContext || llm.max_rows() != 128 || llm.speculative() ||
      config->models.front().artifact !=
          "8c7103418a6608022e5eda50a0dcc4b7688a0d59ef239813c9de0984161397fb")
    return Error("maximum recipe differs");
  rt::SwapParts load;
  if (auto active = server.Activate(llm, load); !active) return active;
  std::ofstream heads(out / "heads.f32", std::ios::binary | std::ios::noreplace);
  if (!heads) return Error("exclusive head stream failed");
  std::array<std::int32_t, kSteps> choices{};
  std::vector<float> row;
  rt::PrefillRun run;
  double prefill_seconds = 0, teacher_seconds = 0, first_prefill_seconds = 0;
  const auto started = rt::Clock::now();
  const auto seconds = [](auto elapsed) { return std::chrono::duration<double>(elapsed).count(); };
  const auto publish = [&]() -> rt::Status {
    if (auto best = Best(row); !best) return Error(best.error());
    heads.write(reinterpret_cast<const char*>(row.data()),
                static_cast<std::streamsize>(row.size() * sizeof(float)));
    return heads ? rt::Status{} : Error("head stream failed");
  };
  auto request = server.InRequest(llm, [&]() -> rt::Status {
    if (repeat) {
      if (auto cleared = llm.Clear(); !cleared) return cleared;
      rt::PrefillRun first;
      const auto first_start = rt::Clock::now();
      if (auto pref =
              llm.Prefill(std::span<const std::int32_t>(ids).first(kPrefix), row, {}, &first);
          !pref)
        return pref;
      first_prefill_seconds = seconds(rt::Clock::now() - first_start);
      if (first.end != kPrefix || first.chunks != 1024 || first.stopped)
        return Error("first traversal geometry differs");
    }
    if (auto cleared = llm.Clear(); !cleared) return cleared;
    std::uint32_t scheduled = 0, chunks = 0;
    const auto progress = [&](std::uint32_t rows) {
      if (chunks % 128 == 0) {
        (void)std::fprintf(stderr, "GEMMA3_MAX_PROGRESS completed_prefix=%u elapsed_s=%.6f\n",
                           scheduled, seconds(rt::Clock::now() - started));
        (void)std::fflush(stderr);
      }
      scheduled += rows;
      ++chunks;
      return true;
    };
    const auto prefill_start = rt::Clock::now();
    if (auto pref =
            llm.Prefill(std::span<const std::int32_t>(ids).first(kPrefix), row, progress, &run);
        !pref)
      return pref;
    prefill_seconds = seconds(rt::Clock::now() - prefill_start);
    if (run.end != kPrefix || run.chunks != 1024 || run.stopped || llm.history().size() != kPrefix)
      return Error("completed prefill geometry differs");
    const auto teacher_start = rt::Clock::now();
    for (std::uint32_t i = 0; i < kSteps; ++i) {
      auto best = Best(row);
      if (!best) return Error(best.error());
      choices[i] = *best;
      if (auto written = publish(); !written) return written;
      rt::PrefillRun scalar;
      if (auto pref = llm.Prefill(std::span<const std::int32_t>(ids).subspan(kPrefix + i, 1), row,
                                  {}, &scalar);
          !pref)
        return pref;
      if (scalar.chunks != 1 || scalar.end != kPrefix + i + 1 || scalar.stopped)
        return Error("teacher scalar geometry differs");
    }
    teacher_seconds = seconds(rt::Clock::now() - teacher_start);
    if (auto written = publish(); !written) return written;
    if (!std::ranges::equal(llm.history(), ids)) return Error("initialized history differs");
    std::vector<float> refused;
    if (llm.Prefill(std::span<const std::int32_t>(ids).last(1), refused) ||
        !std::ranges::equal(llm.history(), ids))
      return Error("context refusal changed initialized history");
    return {};
  });
  if (!request) return request;
  heads.flush();
  if (!heads) return Error("complete head stream failed");
  if (auto written = Write(out / "chosen.i32", std::span<const std::int32_t>(choices)); !written)
    return written;
  if (auto written = Write(out / "final.f32", std::span<const float>(row)); !written)
    return written;
  if (auto finished = server.FinishSwap(load); !finished) return finished;
  const auto graphs = llm.graphs();
  const std::string result = std::format(
      R"({{"context":{},"prefix":{},"teacher_writes":{},"full_heads":65,"scored_transitions":64,"initialized_positions":{},"natural_emitted_tokens":0,"initialized_history_sha256":"{}","prefill_chunks":{},"first_prefill_seconds":{:.6f},"prefill_seconds":{:.6f},"teacher_seconds":{:.6f},"longest_prefill_chunk_seconds":{:.6f},"budget_bytes":{},"fixed_bytes":{},"host_input_bytes":{},"sampled_memavailable_decrease_bytes":{},"minimum_available_bytes":{},"graphs":{{"eager":{},"captured":{},"replayed":{},"refused":{},"kept":{}}},"model":{}}})",
      kContext, kPrefix, kSteps, llm.history().size(),
      Hash(std::span<const std::int32_t>(llm.history())), run.chunks, first_prefill_seconds,
      prefill_seconds, teacher_seconds, run.longest, server.budget(), server.fixed_bytes(),
      server.host_input_bytes(), server.memory().peak(), server.memory().all(), graphs.eager,
      graphs.captured, graphs.replayed, graphs.refused, graphs.kept, llm.extra());
  if (auto written = Write(out / "result.json", std::span<const char>(result)); !written)
    return written;
  (void)std::printf("%s\n", result.c_str());
  return server.TearDown();
}
}  // namespace

int main(int argc, char** argv) {
  std::vector<std::string_view> args;
  for (int i = 2; i < argc; ++i) args.emplace_back(argv[i]);
  if (argc == 5 && std::string_view(argv[1]) == "prepare") {
    const auto result = Prepare(args);
    if (!result) (void)std::fprintf(stderr, "%s\n", result.error().c_str());
    return result ? 0 : 1;
  }
  if (argc < 7 || argc > 12 || std::string_view(argv[1]) != "run") return 2;
  const auto result = Run(args);
  if (!result) (void)std::fprintf(stderr, "%s\n", result.error().c_str());
  return result ? 0 : 1;
}
