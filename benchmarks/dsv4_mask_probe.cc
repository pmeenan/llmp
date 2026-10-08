// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// One production-width DeepSeek target mask factor: independent prefills,
// joined C2 natural decode, full head bytes and initialized-state hashes.
// Usage: ARTIFACT IDS0.i32 IDS1.i32 OUT host|device|tokens-host|tokens-device
//        ARTIFACT IDS0.i32 IDS1.i32 OUT injected-host|injected-device DRAFTER
#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <span>
#include <string>
#include <tuple>
#include <vector>

#include "base/sha256.h"
#include "engine/dsv4_runner.h"
#include "engine/support.h"
#include "platform/crash_policy.h"
#include "tokenizer/gguf.h"
#include "tokenizer/tokenizer.h"

namespace {
namespace en = jitllm::engine;
namespace fs = std::filesystem;
using en::support::Error;
constexpr std::uint32_t kSteps = 16;
constexpr std::uint64_t kCopy = 8ULL << 20U;
template <class T>
en::Status Save(const fs::path& path, std::span<const T> data) {
  std::ofstream file(path, std::ios::binary | std::ios::noreplace);
  file.write(reinterpret_cast<const char*>(data.data()),
             static_cast<std::streamsize>(data.size_bytes()));
  file.flush();
  return file ? en::Status{} : Error("exclusive complete output refused");
}
std::expected<std::int32_t, std::string> Best(const std::vector<float>& row, std::uint32_t vocab) {
  if (row.size() != vocab || !std::ranges::all_of(row, [](float x) { return std::isfinite(x); }))
    return Error("finite complete head required");
  return static_cast<std::int32_t>(std::max_element(row.begin(), row.end()) - row.begin());
}
en::Status Prepare(const char* metadata_path, const char* text_path, const char* rows_text,
                   const char* output) {
  std::uint32_t rows = 0;
  const std::string_view number = rows_text;
  const auto parsed = std::from_chars(number.data(), number.data() + number.size(), rows);
  if (parsed.ec != std::errc{} || parsed.ptr != number.data() + number.size() ||
      (rows != 4352 && rows != 4608))
    return Error("unsupported factor prefix length");
  const auto read = [](const char* path,
                       std::uint64_t limit) -> std::expected<std::string, std::string> {
    std::error_code ec;
    const auto size = fs::file_size(path, ec);
    if (ec || size > limit) return Error("bounded preparation input refused");
    std::string data(size, '\0');
    std::ifstream file(path, std::ios::binary);
    if (!file.read(data.data(), static_cast<std::streamsize>(size)) ||
        file.peek() != std::char_traits<char>::eof())
      return Error("complete preparation read failed");
    return data;
  };
  auto metadata = read(metadata_path, 32ULL << 20U);
  auto text = read(text_path, 1ULL << 20U);
  if (!metadata || !text) return Error("preparation input failed");
  auto gguf = jitllm::tokenizer::ReadGgufTokenizer(std::as_bytes(std::span(*metadata)));
  if (!gguf) return Error(gguf.error().ToString());
  if (gguf->spec.tokens.size() != 129280 || gguf->spec.bos != 0 || gguf->spec.add_bos ||
      gguf->spec.add_eos)
    return Error("DeepSeek vocabulary/BOS contract differs");
  auto tokenizer = jitllm::tokenizer::Tokenizer::Create(std::move(gguf->spec));
  if (!tokenizer) return Error(tokenizer.error().ToString());
  std::vector<jitllm::tokenizer::TokenId> ids;
  if (auto r = tokenizer->Encode(*text,
                                 {.special = jitllm::tokenizer::SpecialTokens::kUserDefinedOnly,
                                  .add_bos_eos = true,
                                  .max_tokens = 262144},
                                 ids);
      !r)
    return Error(r.error().ToString());
  // Match serving.cc: the checkpoint disables automatic BOS, while serving
  // explicitly supplies its BOS once before a new prompt.
  if (tokenizer->bos() && (ids.empty() || ids.front() != *tokenizer->bos()))
    ids.insert(ids.begin(), *tokenizer->bos());
  if (ids.size() < rows || ids.front() != 0) return Error("preparation has too few actual IDs");
  const auto kept = std::span(ids).first(rows);
  if (auto r = Save<std::int32_t>(output, kept); !r) return r;
  std::cout << "DSV4_MASK_PREPARED rows=" << rows << " sha256="
            << jitllm::base::ToHex(jitllm::base::Sha256{}.Update(std::as_bytes(kept)).Finish())
            << " text_sha256=" << jitllm::base::ToHex(jitllm::base::Sha256{}.Update(*text).Finish())
            << '\n';
  return {};
}
}  // namespace
int main(int argc, char** argv) {
  if ((argc != 6 && argc != 7) || !jitllm::platform::InstallCrashPolicy("dsv4-mask-probe"))
    return 2;
  if (std::string_view(argv[1]) == "prepare") {
    if (argc != 6) return 2;
    auto result = Prepare(argv[2], argv[3], argv[4], argv[5]);
    if (!result) std::cerr << result.error() << '\n';
    return result ? 0 : 1;
  }
  const std::string_view mode = argv[5];
  const bool injected = mode == "injected-host" || mode == "injected-device";
  const bool token_factor = injected || mode == "tokens-host" || mode == "tokens-device";
  const bool token_device = mode == "tokens-device" || mode == "injected-device";
  if ((!token_factor && mode != "host" && mode != "device") || argc != (injected ? 7 : 6)) return 2;
  const bool device = mode == "device";
  const fs::path out = argv[4];
  if (!fs::create_directory(out)) return 2;
  std::array<std::vector<std::int32_t>, 2> ids;
  const std::array<std::uint32_t, 2> prefix{4352, 4608};
  for (std::size_t slot = 0; slot < 2; ++slot) {
    std::error_code ec;
    if (fs::file_size(argv[slot + 2], ec) != prefix[slot] * 4 || ec) return 2;
    ids[slot].resize(prefix[slot]);
    std::ifstream file(argv[slot + 2], std::ios::binary);
    if (!file.read(reinterpret_cast<char*>(ids[slot].data()), prefix[slot] * 4) ||
        file.peek() != std::char_traits<char>::eof())
      return 2;
    std::cout << "DSV4_MASK_INPUT slot=" << slot << " rows=" << prefix[slot] << " sha256="
              << jitllm::base::ToHex(
                     jitllm::base::Sha256{}.Update(std::as_bytes(std::span(ids[slot]))).Finish())
              << '\n';
  }
  if (ids[0] == ids[1]) return 2;
  struct Lifetime {
    en::PagedNode node{{.slot_bytes = en::kSlabSlotBytes}};
    std::unique_ptr<en::Dsv4Runner> runner;
    std::vector<en::PagedModel*> entered;
  };
  auto life = std::make_unique<Lifetime>();
  auto& node = life->node;
  en::Dsv4Options options{.artifact = argv[1],
                          .out = out,
                          .context = 8704,
                          .max_rows = 4096,
                          .drafter = injected ? fs::path(argv[6]) : fs::path{},
                          .frontier_head = true,
                          .wave_slots = 2,
                          .device_raw_masks = token_factor || device,
                          .device_tokens = token_device,
                          .spill_place = {}};
  en::SetDsv4ServedPrefill(options);
  life->runner = std::make_unique<en::Dsv4Runner>(node, options, 0, 0);
  auto& runner = *life->runner;
  std::array<en::Dsv4Runner::Slot*, 2> slots{};
  std::array<std::vector<float>, 2> heads;
  std::array<std::uint32_t, 2> past{};
  std::vector<float> saved;
  std::vector<std::int32_t> choices;
  void* pinned = nullptr;
  std::array<std::string, 2> state_hashes;
  std::array<std::uint64_t, 2> state_bytes{};
  std::array<std::string, 2> settled_hashes;
  std::array<std::uint64_t, 2> settled_bytes{};
  const auto snapshot = [&](bool settled = false) -> en::Status {
    auto& hashes = settled ? settled_hashes : state_hashes;
    auto& bytes = settled ? settled_bytes : state_bytes;
    std::ofstream ranges_file(out / (settled ? "settled-ranges.txt" : "state-ranges.txt"),
                              std::ios::noreplace);
    for (std::uint32_t slot = 0; slot < 2; ++slot) {
      const auto ranges = slots[slot]->used_state_ranges();
      jitllm::base::Sha256 hash;
      for (const auto& range : ranges) {
        bytes[slot] += range.bytes;
        ranges_file << slot << ' ' << range.region << ' ' << range.offset << ' ' << range.bytes
                    << '\n';
        for (std::uint64_t at = 0; at < range.bytes;) {
          auto part = range;
          part.offset += at;
          part.bytes = std::min(kCopy, range.bytes - at);
          auto copied = slots[slot]->SaveUsedState(pinned, std::span(&part, 1));
          if (!copied) {
            node.KeepPinned(pinned);
            return copied;
          }
          hash.Update(std::span(static_cast<const std::byte*>(pinned), std::size_t(part.bytes)));
          at += part.bytes;
        }
      }
      hashes[slot] = jitllm::base::ToHex(hash.Finish());
    }
    ranges_file.flush();
    return ranges_file ? en::Status{} : Error("state range output failed");
  };
  const auto execute = [&]() -> en::Status {
    if (auto r = node.Open(); !r) return r;
    life->entered.push_back(&runner);
    if (auto r = runner.Setup(); !r) return r;
    if (runner.wave_capacity() != 2 || runner.state_layout().raw_cells != 4352 ||
        runner.speculative() != injected || (injected && runner.drafter_state_bytes() == 0))
      return Error("production target C2/ring/injection envelope differs");
    for (std::uint32_t slot = 0; slot < 2; ++slot) {
      auto found = runner.request_slot(slot);
      if (!found) return Error(found.error());
      slots[slot] = *found;
      if (!std::ranges::all_of(
              ids[slot], [&](auto id) { return id >= 0 && std::cmp_less(id, runner.vocab()); }))
        return Error("token IDs outside actual vocabulary");
    }
    std::vector<jitllm::catalog::ExtentId> extents;
    auto allocation = node.Pinned(kCopy, 0, extents);
    if (!allocation) return Error(allocation.error());
    pinned = *allocation;
    if (auto r = node.MapWorkspace(runner.activations_needed(), runner.pool_needed()); !r) return r;
    const auto fixed = node.catalog().OccupancyOf(node.domain()).Total().value();
    // Fund retained full heads/choices before allocating the diagnostic payload.
    constexpr std::uint64_t kDiagnosticHost = 64ULL << 20U;
    node.SetHostFloor(runner.host_input_bytes() + runner.plan_floor_bytes() + kDiagnosticHost);
    if (!token_factor) {
      saved.reserve(std::size_t{runner.vocab()} * 2 * (kSteps + 1));
      choices.reserve(kSteps * 2);
    }
    if (auto r = node.Start(jitllm::base::Bytes(fixed + runner.weights().size() * en::kPagedExtent +
                                                4 * node.StateCapacity()));
        !r)
      return r;
    if (auto r = runner.Register(); !r) return r;
    if (auto r = runner.Bind(); !r) return r;
    node.Run();
    if (auto r = runner.CheckHashRouting(); !r) return r;
    if (auto r = runner.SelectSlots(slots); !r) return r;
    if (token_factor) {
      const auto budget =
          fixed + runner.weights().size() * en::kPagedExtent + 4 * node.StateCapacity();
      std::cout << "PLAIN_TOKEN_SETUP budget=" << budget << " fixed=" << fixed
                << " activations=" << runner.activations_needed()
                << " scratch=" << runner.pool_needed()
                << " host_inputs=" << runner.host_input_bytes() << '\n';
      std::array<std::vector<std::int32_t>, 2> histories;
      for (std::size_t i = 0; i < 2; ++i) histories[i].reserve(prefix[i] + 34);
      const auto choose = [&](std::size_t i) -> std::expected<std::int32_t, std::string> {
        if (heads[i].size() != runner.vocab()) return Error("complete plain head required");
        return static_cast<std::int32_t>(std::ranges::max_element(heads[i]) - heads[i].begin());
      };
      const auto traversal = [&](double& prefill, double& decode) -> en::Status {
        const auto begin = std::chrono::steady_clock::now();
        for (std::size_t i = 0; i < 2; ++i) {
          if (auto r = slots[i]->Clear(); !r) return r;
          histories[i].assign(ids[i].begin(), ids[i].end());
          past[i] = 0;
          while (past[i] < prefix[i]) {
            const auto rows = std::min(4096U, prefix[i] - past[i]);
            if (auto r = slots[i]->Chunk(
                    past[i], std::span(histories[i]).subspan(past[i], rows), heads[i],
                    injected ? en::Dsv4ChunkKind::kInject : en::Dsv4ChunkKind::kPlain);
                !r)
              return r;
            past[i] += rows;
          }
          const auto anchor = choose(i);
          if (!anchor) return Error(anchor.error());
          histories[i].push_back(*anchor);
        }
        const auto prefill_end = std::chrono::steady_clock::now();
        for (std::uint32_t step = 0; step < 32; ++step) {
          std::array<std::int32_t, 2> tokens{};
          std::array<en::Dsv4Runner::WaveWork, 2> work;
          for (std::size_t i = 0; i < 2; ++i)
            work[i] = {.slot = slots[i],
                       .pos = past[i],
                       .anchor = histories[i].back(),
                       .logits = token_device ? nullptr : &heads[i],
                       .token = token_device ? &tokens[i] : nullptr};
          if (auto r = runner.DecodeWave(work); !r) return r;
          for (std::size_t i = 0; i < 2; ++i) {
            if (!token_device) {
              const auto selected = choose(i);
              if (!selected) return Error(selected.error());
              tokens[i] = *selected;
            }
            histories[i].push_back(tokens[i]);
            ++past[i];
          }
        }
        const auto end = std::chrono::steady_clock::now();
        prefill = en::support::Seconds(prefill_end - begin);
        decode = en::support::Seconds(end - prefill_end);
        return {};
      };
      return node.WithRequest(
          0, runner.execution_closure(), "DeepSeek plain-token factor", [&]() -> en::Status {
            double warm_prefill = 0, warm_decode = 0;
            if (auto r = traversal(warm_prefill, warm_decode); !r) return r;
            const auto target_before = runner.graph_stats();
            const auto wave_before = runner.wave_stats();
            const auto tokens_before = runner.device_token_outputs();
            double prefill = 0, decode = 0;
            if (auto r = traversal(prefill, decode); !r) return r;
            const auto target_after = runner.graph_stats();
            const auto wave_after = runner.wave_stats();
            const auto selected_tokens = runner.device_token_outputs() - tokens_before;
            const auto target_completed = target_after.eager - target_before.eager +
                                          target_after.captured - target_before.captured +
                                          target_after.replayed - target_before.replayed;
            const auto wave_completed = wave_after.eager - wave_before.eager + wave_after.captured -
                                        wave_before.captured + wave_after.replayed -
                                        wave_before.replayed;
            if (selected_tokens != (token_device ? 64U : 0U) || target_completed != 36 ||
                wave_completed != 32 || wave_before.captured == 0 ||
                wave_after.replayed <= wave_before.replayed || runner.bound_raw_masks() == 0 ||
                runner.coverage_violations() != 0 || !std::isfinite(prefill) ||
                !std::isfinite(decode) || prefill <= 0 || decode <= 0)
              return Error(
                  "actual plain-token, completed-work, capture/replay or coverage witness refused");
            if (auto r = snapshot(); !r) return r;
            for (std::size_t i = 0; i < 2; ++i) {
              if (state_bytes[i] == 0) return Error("initialized target state required");
              if (auto r = Save<std::int32_t>(out / ("history" + std::to_string(i) + ".i32"),
                                              histories[i]);
                  !r)
                return r;
              const std::array<std::int32_t, 1> anchor{histories[i].back()};
              if (auto r = slots[i]->Chunk(
                      past[i], anchor, heads[i],
                      injected ? en::Dsv4ChunkKind::kInject : en::Dsv4ChunkKind::kPlain);
                  !r)
                return r;
              if (!Best(heads[i], runner.vocab()))
                return Error("finite complete continuation head required");
              if (auto r = Save<float>(out / ("head" + std::to_string(i) + ".f32"), heads[i]); !r)
                return r;
              if (injected) {
                // A real draft consumes the ring fed by every plain decode.
                // Complete verify heads/proposals are observations outside paid work.
                const auto next = Best(heads[i], runner.vocab());
                if (!next) return Error(next.error());
                std::vector<std::int32_t> drafts;
                std::vector<float> verified;
                if (auto r = slots[i]->DraftVerify(past[i] + 1, *next, 3, drafts, verified); !r)
                  return r;
                if (drafts.size() != runner.draft_rows() ||
                    verified.size() != 3 * std::size_t{runner.vocab()} ||
                    !std::ranges::all_of(verified, [](float v) { return std::isfinite(v); }) ||
                    !std::ranges::all_of(drafts, [&](auto id) {
                      return id >= 0 && std::cmp_less(id, runner.vocab());
                    }))
                  return Error("complete finite draft/verify continuation required");
                if (auto r =
                        Save<std::int32_t>(out / ("draft" + std::to_string(i) + ".i32"), drafts);
                    !r)
                  return r;
                if (auto r = Save<float>(out / ("verify" + std::to_string(i) + ".f32"), verified);
                    !r)
                  return r;
                if (auto r = slots[i]->Accept(1); !r) return r;
                if (auto r = slots[i]->Rollback(); !r) return r;
              }
            }
            if (injected) {
              if (auto r = snapshot(true); !r) return r;
              std::cout << "PLAIN_TOKEN_INJECTION drafter_state_bytes="
                        << runner.drafter_state_bytes() << " continued_draft_verifies=2"
                        << " settled0_bytes=" << settled_bytes[0]
                        << " settled0=" << settled_hashes[0]
                        << " settled1_bytes=" << settled_bytes[1]
                        << " settled1=" << settled_hashes[1] << '\n';
            }
            std::cout
                << std::setprecision(17)
                << "PLAIN_TOKEN_FACTOR mode=" << (token_device ? "on" : "off")
                << " context=8704 chunk=4096 slots=2 format=gguf decode_units=32 device_tokens="
                << selected_tokens << " prefill_seconds=" << prefill << " decode_seconds=" << decode
                << " paid_seconds=" << prefill + decode << " warm_prefill_seconds=" << warm_prefill
                << " warm_decode_seconds=" << warm_decode
                << " eager=" << target_after.eager - target_before.eager
                << " captured=" << target_after.captured - target_before.captured
                << " replayed=" << target_after.replayed - target_before.replayed
                << " wave_replayed=" << wave_after.replayed - wave_before.replayed
                << " selected_raw_masks=" << runner.bound_raw_masks()
                << " state0_bytes=" << state_bytes[0] << " state0=" << state_hashes[0]
                << " state1_bytes=" << state_bytes[1] << " state1=" << state_hashes[1] << '\n';
            return {};
          });
    }
    return node.WithRequest(
        0, runner.execution_closure(), "DeepSeek target raw-mask factor", [&]() -> en::Status {
          if (!node.InRequest(0)) return Error("held direct request required");
          const auto begin = std::chrono::steady_clock::now();
          for (auto* slot : slots)
            if (auto r = slot->Clear(); !r) return r;
          for (std::uint32_t slot = 0; slot < 2; ++slot) {
            while (past[slot] < prefix[slot]) {
              const auto rows = std::min(4096U, prefix[slot] - past[slot]);
              if (auto r = slots[slot]->Chunk(
                      past[slot], std::span(ids[slot]).subspan(past[slot], rows), heads[slot]);
                  !r)
                return r;
              past[slot] += rows;
              if (!Best(heads[slot], runner.vocab())) return Error("bad prefill frontier");
            }
          }
          const auto prefill_end = std::chrono::steady_clock::now();
          const auto retain = [&] {
            for (const auto& head : heads) saved.insert(saved.end(), head.begin(), head.end());
          };
          retain();
          for (std::uint32_t step = 0; step < kSteps; ++step) {
            std::array<en::Dsv4Runner::WaveWork, 2> wave;
            for (std::uint32_t slot = 0; slot < 2; ++slot) {
              auto token = Best(heads[slot], runner.vocab());
              if (!token) return Error(token.error());
              choices.push_back(*token);
              wave[slot] = {
                  .slot = slots[slot], .pos = past[slot], .anchor = *token, .logits = &heads[slot]};
            }
            if (auto r = runner.DecodeWave(wave); !r) return r;
            for (std::uint32_t slot = 0; slot < 2; ++slot) {
              ++past[slot];
              if (!Best(heads[slot], runner.vocab())) return Error("bad joined decode head");
            }
            retain();
          }
          const auto end = std::chrono::steady_clock::now();
          if (auto r = snapshot(); !r) return r;
          if (auto r = Save<float>(out / "heads.f32", saved); !r) return r;
          if (auto r = Save<std::int32_t>(out / "choices.i32", choices); !r) return r;
          const auto& stats = runner.graph_stats();
          const auto& waves = runner.wave_stats();
          if ((device ? runner.bound_raw_masks() == 0 : runner.bound_raw_masks() != 0) ||
              waves.captured == 0 || waves.replayed == 0 || runner.coverage_violations() != 0)
            return Error("target mask selection, joined replay or memory coverage failed");
          std::cout << "DSV4_MASK_FACTOR mode=" << mode
                    << " chunk=4096 context=8704 ring_cells=4352 slots=2"
                    << " prompt_rows0=" << prefix[0] << " prompt_rows1=" << prefix[1]
                    << " decode_steps=" << kSteps << " generated_tokens=" << choices.size()
                    << " past0=" << past[0] << " past1=" << past[1]
                    << " prefill_seconds=" << en::support::Seconds(prefill_end - begin)
                    << " decode_seconds=" << en::support::Seconds(end - prefill_end)
                    << " paid_seconds=" << en::support::Seconds(end - begin)
                    << " selected_raw_masks=" << runner.bound_raw_masks()
                    << " eager=" << stats.eager << " captured=" << stats.captured
                    << " replayed=" << stats.replayed << " wave_eager=" << waves.eager
                    << " wave_captured=" << waves.captured << " wave_replayed=" << waves.replayed
                    << " state0=" << state_hashes[0] << " state1=" << state_hashes[1] << '\n';
          return {};
        });
  };
  auto ran = execute();
  auto retired = node.TearDown(life->entered);
  if (!ran) std::cerr << ran.error() << '\n';
  if (!retired) {
    std::cerr << retired.error() << '\n';
    std::ignore = life.release();
  }
  if (ran && retired)
    std::cout << (token_factor ? "PLAIN_TOKEN_FACTOR_RETIRED\n" : "DSV4_MASK_FACTOR_RETIRED\n");
  return ran && retired ? 0 : 1;
}
