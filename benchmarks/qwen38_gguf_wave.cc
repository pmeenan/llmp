// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Fixed-history GGUF wave qualification: every target logit and final state
// against solo steps at 256-cell read alignment, or unpaired waves at
// production's 2048-cell alignment, with eager/capture/replay.
#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <expected>
#include <filesystem>
#include <fstream>
#include <print>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

#include "base/bytes.h"
#include "base/sha256.h"
#include "engine/paged_node.h"
#include "engine/qwen38_runner.h"
#include "engine/support.h"

namespace {
namespace en = llmp::engine;
using en::support::Error;
using Slot = en::Qwen38Runner::Slot;
constexpr std::size_t kSteps = 32;
using Histories = std::array<std::vector<std::int32_t>, 4>;

std::string StateHash(Slot& slot, en::Status& status) {
  std::vector<std::byte> target;
  std::vector<std::byte> draft;
  status = slot.ReadState(target, draft);
  if (!status) {
    return {};
  }
  llmp::base::Sha256 sha;
  sha.Update(target);
  sha.Update(draft);
  return llmp::base::ToHex(sha.Finish());
}

std::int32_t Argmax(std::span<const float> row) {
  return static_cast<std::int32_t>(std::ranges::max_element(row) - row.begin());
}

bool Same(std::span<const float> a, std::span<const float> b) {
  return a.size() == b.size() && std::ranges::equal(a, b, [](float x, float y) {
           return std::bit_cast<std::uint32_t>(x) == std::bit_cast<std::uint32_t>(y);
         });
}

struct Reference {
  std::vector<float> first;
  std::array<std::vector<float>, kSteps> logits;
  std::string state;
};

en::Status Check(en::PagedNode& node, en::Qwen38Runner& runner, const Histories& prompts,
                 std::uint32_t read_align) {
  std::array<Slot*, 4> slots{};
  for (std::size_t i = 0; i < slots.size(); ++i) {
    auto slot = runner.request_slot(i);
    if (!slot) {
      return std::unexpected(slot.error());
    }
    slots[i] = *slot;
  }
  std::uint64_t identical = 0;
  llmp::base::Sha256 wave_logits;
  llmp::base::Sha256 wave_states;
  for (const bool long_context : {false, true}) {
    Histories histories = prompts;
    if (long_context) {
      for (auto& prompt : histories) {
        prompt.insert(prompt.begin(), 2050, 1000);
      }
    }
    const Histories inputs = histories;
    std::array<Reference, 4> references;
    const auto prefill = [&](Slot& slot, const std::vector<std::int32_t>& ids,
                             std::vector<float>& logits) -> en::Status {
      if (auto cleared = slot.Clear(); !cleared) {
        return cleared;
      }
      for (std::size_t at = 0; at < ids.size(); at += 512) {
        const auto end = std::min(at + 512, ids.size());
        if (auto ran =
                slot.Chunk(std::span(ids).first(end), static_cast<std::uint32_t>(at), logits);
            !ran) {
          return ran;
        }
      }
      return {};
    };
    runner.set_graphs(false);
    for (std::size_t s = 0; s < slots.size(); ++s) {
      const std::array<Slot*, 1> alone = {slots[s]};
      if (auto selected = runner.SelectSlots(alone); !selected) {
        return selected;
      }
      auto ran = node.WithRequest(
          0, runner.execution_closure(), "GGUF scalar reference", [&]() -> en::Status {
            auto& ref = references[s];
            auto& history = histories[s];
            if (auto filled = prefill(*slots[s], history, ref.first); !filled) {
              return filled;
            }
            history.push_back(Argmax(ref.first));
            for (std::size_t i = 0; i < kSteps; ++i) {
              if (auto step = slots[s]->Chunk(
                      history, static_cast<std::uint32_t>(history.size() - 1), ref.logits[i]);
                  !step) {
                return step;
              }
              history.push_back(Argmax(ref.logits[i]));
            }
            en::Status read;
            ref.state = StateHash(*slots[s], read);
            return read;
          });
      if (!ran) {
        return ran;
      }
    }
    for (const std::size_t width : {2U, 3U, 4U}) {
      auto expected = references;
      if (auto selected = runner.SelectSlots(std::span(slots).first(width)); !selected) {
        return selected;
      }
      for (const bool graphs : {false, true}) {
        runner.set_graphs(graphs);
        for (const bool paired : {false, true}) {
          const bool golden = read_align != 256 && !graphs && !paired;
          auto ran = node.WithRequest(
              0, runner.execution_closure(), "GGUF wave control", [&]() -> en::Status {
                for (std::size_t s = 0; s < width; ++s) {
                  std::vector<float> first;
                  if (auto filled = prefill(*slots[s], inputs[s], first); !filled) {
                    return filled;
                  }
                  if (!Same(first, references[s].first)) {
                    return Error("GGUF control prefill changed");
                  }
                }
                const auto before = runner.graph_stats();
                for (std::size_t i = 0; i < kSteps; ++i) {
                  std::array<std::vector<float>, 4> logits;
                  std::vector<en::Qwen38Runner::ChunkWork> work;
                  work.reserve(width);
                  for (std::size_t s = 0; s < width; ++s) {
                    const auto end = inputs[s].size() + i + 1;
                    work.push_back({slots[s], std::span(histories[s]).first(end),
                                    static_cast<std::uint32_t>(end - 1), &logits[s]});
                  }
                  if (auto step = runner.ChunkWave(work, paired); !step) {
                    return step;
                  }
                  if ((runner.last_wave().vecq_pairs != 0) != paired) {
                    return Error("GGUF wave did not execute its requested product form");
                  }
                  for (std::size_t s = 0; s < width; ++s) {
                    wave_logits.Update(std::as_bytes(std::span(logits[s])));
                    if (golden) {
                      expected[s].logits[i] = logits[s];
                      continue;
                    }
                    if (!Same(logits[s], expected[s].logits[i])) {
                      std::println(
                          stderr,
                          "logits differ: long={} width={} graphs={} paired={} slot={} step={}",
                          long_context, width, graphs, paired, s, i);
                      return Error("GGUF wave logits differ from the fixed reference");
                    }
                    ++identical;
                  }
                }
                const auto& after = runner.graph_stats();
                const auto captures = after.captured - before.captured;
                const auto replays = after.replayed - before.replayed;
                if (graphs ? (captures == 0 || replays == 0) : (captures != 0 || replays != 0)) {
                  return Error("GGUF wave cell did not execute its requested graph path");
                }
                std::println("paths: long={} width={} graphs={} paired={} captured={} replayed={}",
                             long_context, width, graphs, paired, captures, replays);
                for (std::size_t s = 0; s < width; ++s) {
                  en::Status read;
                  const auto hash = StateHash(*slots[s], read);
                  if (!read) {
                    return read;
                  }
                  wave_states.Update(hash);
                  if (golden) {
                    expected[s].state = hash;
                  } else if (hash != expected[s].state) {
                    return Error("GGUF wave state differs from the fixed reference");
                  }
                }
                return {};
              });
          if (!ran) {
            return ran;
          }
          std::println("pass: long={} width={} graphs={} paired={}", long_context, width, graphs,
                       paired);
          static_cast<void>(std::fflush(stdout));
        }
      }
    }
  }
  const auto& graphs = runner.graph_stats();
  if (graphs.captured == 0 || graphs.replayed == 0) {
    return Error("GGUF wave controls did not capture and replay graphs");
  }
  std::println(
      "reference={} read_align={} exact rows={} captured={} replayed={} coverage_violations={}",
      read_align == 256 ? "scalar" : "unpaired-wave", read_align, identical, graphs.captured,
      graphs.replayed, runner.coverage_violations());
  std::println("wave_logits_sha256={} wave_states_sha256={}",
               llmp::base::ToHex(wave_logits.Finish()), llmp::base::ToHex(wave_states.Finish()));
  return runner.coverage_violations() == 0 ? en::Status{} : Error("GGUF coverage violation");
}
}  // namespace

int main(int argc, char** argv) {
  if (argc < 4 || argc > 6) {
    std::println(
        stderr,
        "usage: llmp_qwen38_gguf_wave ARTIFACT PROMPTS.tsv NEW_STATE_DIR [256|2048] [on|off]");
    return 2;
  }
  const std::string_view alignment = argc >= 5 ? argv[4] : "2048";
  if (alignment != "256" && alignment != "2048") {
    return 2;
  }
  const std::uint32_t read_align = alignment == "256" ? 256 : 2048;
  const std::string_view lanes = argc == 6 ? argv[5] : "on";
  if (lanes != "on" && lanes != "off") {
    return 2;
  }
  Histories prompts;
  std::ifstream file(argv[2]);
  for (auto& ids : prompts) {
    std::string line;
    if (!std::getline(file, line) || !line.contains('\t')) {
      return 2;
    }
    std::istringstream values(line.substr(line.find('\t') + 1));
    std::int32_t id = 0;
    while (values >> id) {
      if (id < 0 || id >= 248320) {
        return 2;
      }
      ids.push_back(id);
    }
    if (ids.empty() || ids.size() > 512 || !values.eof()) {
      return 2;
    }
  }
  std::error_code ec;
  if (!std::filesystem::create_directory(argv[3], ec) || ec) {
    return 2;
  }
  en::PagedNode node({.slot_bytes = en::kSlabSlotBytes});
  const en::Qwen38Options options{.artifact = argv[1],
                                  .out = argv[3],
                                  .context = 4096,
                                  .max_rows = 512,
                                  .drafter = {},
                                  .wave_slots = 4,
                                  .wave_read_align = read_align,
                                  .spill_place = {},
                                  .wave_lanes = lanes == "on"};
  en::Qwen38Runner runner(node, options, 0, 0);
  std::vector<en::PagedModel*> entered;
  const auto execute = [&]() -> en::Status {
    if (auto opened = node.Open(); !opened) {
      return opened;
    }
    entered.push_back(&runner);
    if (auto setup = runner.Setup(); !setup) {
      return setup;
    }
    if (auto workspace = node.MapWorkspace(runner.activations_needed(), runner.pool_needed());
        !workspace) {
      return workspace;
    }
    if (auto started = node.Start(llmp::base::Bytes(std::uint64_t{100} << 30)); !started) {
      return started;
    }
    if (auto registered = runner.Register(); !registered) {
      return registered;
    }
    if (auto bound = runner.Bind(); !bound) {
      return bound;
    }
    node.Run();
    std::vector<en::LoadStats> loads;
    if (auto loaded = node.Load(runner.weights(), "GGUF wave weights", loads); !loaded) {
      return loaded;
    }
    if (auto hash = runner.ReadPleHash(); !hash) {
      return hash;
    }
    return Check(node, runner, prompts, read_align);
  };
  const auto ran = execute();
  const auto retired = node.TearDown(entered);
  std::println("complete={} retired={} error={}", ran.has_value(), retired.has_value(),
               ran ? "" : ran.error());
  if (!retired) {
    std::println(stderr, "retirement: {}", retired.error());
    std::abort();
  }
  return ran ? 0 : 1;
}
