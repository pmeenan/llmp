// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <format>
#include <fstream>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "base/sha256.h"
#include "engine/support.h"
#include "qwen38_batch.h"

namespace llmp::benchmarks::qwen_batch {
namespace {
namespace en = engine;
namespace dv = draft_vocab;
using en::support::Error;
using Clock = std::chrono::steady_clock;
constexpr std::uint64_t kPage = en::kPagedExtent;
constexpr std::uint32_t kOutputs = 256;
// The most graphs a natural run may keep: its shapes' (the runner's caches
// no longer cap them; the proof's own shapes are few).
constexpr std::size_t kMaxProofGraphs = 16;

template <typename T>
bool Exact(const std::vector<T>& a, const std::vector<T>& b) {
  return a.size() == b.size() &&
         (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0);
}

template <typename T>
en::Status Write(const std::filesystem::path& path, const std::vector<T>& values) {
  std::ofstream file(path, std::ios::binary);
  if (!values.empty()) {
    file.write(reinterpret_cast<const char*>(values.data()),
               static_cast<std::streamsize>(values.size() * sizeof(T)));
  }
  file.close();
  return file ? en::Status{} : Error("C2 natural raw output flush failed");
}

template <typename T>
std::string Sha(const std::vector<T>& values) {
  base::Sha256 sha;
  sha.Update(std::as_bytes(std::span(values)));
  return base::ToHex(sha.Finish());
}

template <typename T>
std::string Numbers(const T& values) {
  std::string out;
  for (auto value : values) {
    out += std::format("{}{}", out.empty() ? "" : ",", value);
  }
  return "[" + out + "]";
}

en::GraphStats Difference(const en::GraphStats& after, const en::GraphStats& before) {
  en::GraphStats out;
  out.eager = after.eager - before.eager;
  out.captured = after.captured - before.captured;
  out.replayed = after.replayed - before.replayed;
  out.refused = after.refused - before.refused;
  out.dropped = after.dropped - before.dropped;
  out.nodes = after.nodes - before.nodes;
  out.capture_seconds = after.capture_seconds - before.capture_seconds;
  out.instantiate_seconds = after.instantiate_seconds - before.instantiate_seconds;
  out.memory_bytes = after.memory_bytes - before.memory_bytes;
  return out;
}
}  // namespace

Proof::Status Proof::Judge(GenerationStep& step, const Output& draft, const Output& target,
                           Requests<std::vector<std::int32_t>>& histories,
                           Requests<std::vector<std::int32_t>>* tokens) {
  for (std::size_t s = 0; s < requests_; ++s) {
    if ((step.active & (1U << s)) == 0) {
      if (!draft.ids[s].empty() || !target.ids[s].empty()) {
        return Error("C2 inactive request returned tokens");
      }
      continue;
    }
    const auto rows = step.rows[s];
    if (rows == 0 || rows > 4 || draft.ids[s].size() != 3 || target.ids[s].size() != rows ||
        histories[s].size() != std::size_t{step.first[s]} + 1 ||
        (tokens != nullptr &&
         ((*tokens)[s].size() >= kOutputs || rows > kOutputs - (*tokens)[s].size()))) {
      return Error("C2 natural verdict has inconsistent bounded rows/history/output budget");
    }
    std::uint32_t accepted = 0;
    while (accepted + 1 < rows && draft.ids[s][accepted] == target.ids[s][accepted]) {
      ++accepted;
    }
    step.keep[s] = accepted + 1;
    step.drafts[s] = draft.ids[s];
    step.verdicts[s] = target.ids[s];
    if (auto kept = slots_[s].state.Accept(step.keep[s]); !kept) {
      return kept;
    }
    slots_[s].pending = step.keep[s];
    for (std::uint32_t row = 0; row < step.keep[s]; ++row) {
      const auto id = row < accepted ? draft.ids[s][row] : target.ids[s][accepted];
      histories[s].push_back(id);
      if (tokens != nullptr) {
        (*tokens)[s].push_back(id);
      }
    }
  }
  return Settle();
}

Proof::Status Proof::GenerationState(std::size_t arm, std::size_t index,
                                     const std::filesystem::path& out) {
  if (index >= references_.size()) {
    return Error("C2 reference index exceeds fixed catalog");
  }
  for (std::size_t s = 0; s < requests_; ++s) {
    const auto& slot = slots_[s];
    if (slot.state.owed() || slot.state.verify_rows() != 0 || slot.state.quarantined() ||
        slot.pending == 0 || slot.pending > 4 ||
        !std::ranges::equal(
            slot.ranges, slot.state.used_ranges(), [](const auto& a, const auto& b) {
              return a.region == b.region && a.offset == b.offset && a.bytes == b.bytes;
            })) {
      return Error("C2 natural final state is unsettled or initialized geometry differs");
    }
    std::vector<dv::PageRange> ranges;
    ranges.reserve(slot.ranges.size());
    for (const auto& r : slot.ranges) {
      ranges.push_back({.region = r.region, .offset = r.offset, .bytes = r.bytes});
    }
    if (arm == 0) {
      auto captured = dv::ReferencePages::Capture(out, ranges, slot.pending, {live_page_, kPage},
                                                  Reader(s), reference_stats_);
      if (!captured) {
        return Error(captured.error().detail);
      }
      references_[index][s] = std::move(*captured);
    } else {
      auto same = references_[index][s].Compare(ranges, slot.pending, {expected_page_, kPage},
                                                {live_page_, kPage}, Reader(s), reference_stats_);
      if (!same) {
        return Error(same.error().detail);
      }
    }
    ++state_controls_;
    if (index == (requests_ == 4 ? 4U : 3U)) {
      generation_state_sha_[s] = references_[index][s].sha256();
    } else {
      ++new_shape_state_controls_;
    }
  }
  return {};
}

Proof::Status Proof::GenerationShapeControls(const std::filesystem::path& out) {
  // Three C2 or four C4 short waves. Prior graph qualification authenticates
  // the full-head products; these controls cover new ordinary confidence-off,
  // IDs-only copies, single active requests and unequal partial verify tails.
  const std::vector<std::uint32_t> masks = requests_ == 4 ? std::vector<std::uint32_t>{12, 5, 7, 15}
                                                          : std::vector<std::uint32_t>{1, 2, 3};
  const std::vector<Requests<std::uint32_t>> widths =
      requests_ == 4
          ? std::vector<Requests<std::uint32_t>>{{0, 0, 1, 2},
                                                 {3, 0, 1, 0},
                                                 {4, 3, 1, 0},
                                                 {4, 3, 2, 1}}
          : std::vector<Requests<std::uint32_t>>{{1, 0, 0, 0}, {0, 2, 0, 0}, {3, 1, 0, 0}};
  for (std::size_t shape = 0; shape < masks.size(); ++shape) {
    Output expected_draft;
    Output expected_target;
    GenerationStep expected_step;
    for (std::size_t arm = 0; arm < 4; ++arm) {
      const bool batch = arm >= 2;
      const bool diagnostic = arm % 2 == 0;
      if (auto reset = Reset(); !reset) {
        return reset;
      }
      auto histories = Histories();
      GenerationStep step;
      step.active = masks[shape];
      step.rows = widths[shape];
      Output draft;
      Output target;
      if (auto drafted = Draft(histories, batch, false, draft, step.active, diagnostic); !drafted) {
        return drafted;
      }
      auto verify = histories;
      for (std::size_t s = 0; s < requests_; ++s) {
        if ((step.active & (1U << s)) == 0) {
          continue;
        }
        step.first[s] = static_cast<std::uint32_t>(histories[s].size() - 1);
        verify[s].insert(verify[s].end(), draft.ids[s].begin(),
                         draft.ids[s].begin() + step.rows[s] - 1);
      }
      if (auto verified = Verify(verify, step.first, batch, false, target, step.active, diagnostic);
          !verified) {
        return verified;
      }
      if (auto judged = Judge(step, draft, target, histories, nullptr); !judged) {
        return judged;
      }
      for (std::size_t s = 0; s < requests_; ++s) {
        const auto prefix = std::format("shape{}-arm{}-slot{}-", shape, arm, s);
        for (const auto& [name, data] : {std::pair{"draft-input", &draft.head_inputs[s]},
                                         std::pair{"draft-logits", &draft.head_logits[s]},
                                         std::pair{"target-logits", &target.values[s]}}) {
          if (!data->empty()) {
            if (auto written = Write(out / (prefix + name + ".f32"), *data); !written) {
              return written;
            }
          }
        }
        if (auto written = Write(out / (prefix + "draft-ids.i32"), draft.ids[s]); !written) {
          return written;
        }
        if (auto written = Write(out / (prefix + "target-ids.i32"), target.ids[s]); !written) {
          return written;
        }
        if (arm != 0 &&
            (!Exact(draft.ids[s], expected_draft.ids[s]) ||
             !Exact(target.ids[s], expected_target.ids[s]) ||
             (diagnostic && (!Exact(draft.head_inputs[s], expected_draft.head_inputs[s]) ||
                             !Exact(draft.head_logits[s], expected_draft.head_logits[s]) ||
                             !Exact(target.values[s], expected_target.values[s]))))) {
          return Error("C2 new shape differs between full-row and IDs-only serial/batch controls");
        }
        ++new_shape_output_controls_;
      }
      if (arm == 0) {
        expected_draft = std::move(draft);
        expected_target = std::move(target);
        expected_step = step;
      } else if (step != expected_step) {
        return Error("C2 new shape natural commit/rollback trace differs");
      }
      if (auto state = GenerationState(arm, shape, out); !state) {
        return state;
      }
    }
  }
  return {};
}

Proof::Status Proof::Generate(bool batch, GenerationRun& run) {
  auto histories = Histories();
  for (std::size_t s = 0; s < requests_; ++s) {
    run.tokens[s] = {histories[s].back()};
  }
  const std::size_t max_steps = std::size_t{requests_} * (kOutputs - 1);
  run.steps.reserve(max_steps);
  const auto graph_before = graph_stats_;
  const auto mx_before = mxfp8_pairs_;
  const auto routed_before = routed_pairs_;
  const auto pack_before = packed_bytes_;
  // Same cold private cache policy in all four arms. Destruction occurs after
  // the previous completed job; every new plan/capture is paid by this clock.
  const auto start = Clock::now();
  target_plans_.Clear();
  draft_plans_.Clear();
  const auto generated = [&]() -> Status {
    while (std::ranges::any_of(std::span(run.tokens).first(requests_),
                               [](const auto& ids) { return ids.size() < kOutputs; })) {
      if (run.steps.size() >= max_steps) {
        return Error("Natural loop exceeded finite output bound");
      }
      GenerationStep step;
      for (std::size_t s = 0; s < requests_; ++s) {
        if (run.tokens[s].size() >= kOutputs) {
          continue;
        }
        step.active |= 1U << s;
        step.first[s] = static_cast<std::uint32_t>(histories[s].size() - 1);
        step.rows[s] =
            static_cast<std::uint32_t>(std::min<std::size_t>(4, kOutputs - run.tokens[s].size()));
      }
      Output draft;
      Output target;
      auto at = Clock::now();
      auto drafted = Draft(histories, batch, true, draft, step.active, false);
      run.draft_seconds += en::support::Seconds(Clock::now() - at);
      if (!drafted) {
        return drafted;
      }
      auto verify = histories;
      for (std::size_t s = 0; s < requests_; ++s) {
        if ((step.active & (1U << s)) == 0) {
          continue;
        }
        verify[s].insert(verify[s].end(), draft.ids[s].begin(),
                         draft.ids[s].begin() + step.rows[s] - 1);
      }
      at = Clock::now();
      auto verified = Verify(verify, step.first, batch, true, target, step.active, false);
      run.verify_seconds += en::support::Seconds(Clock::now() - at);
      if (!verified) {
        return verified;
      }
      at = Clock::now();
      auto judged = Judge(step, draft, target, histories, &run.tokens);
      run.settle_seconds += en::support::Seconds(Clock::now() - at);
      if (!judged) {
        return judged;
      }
      // Host-observed completion after the wave's independent settlements;
      // no claim of per-kernel or first-token timestamps inside the graph.
      const auto elapsed = en::support::Seconds(Clock::now() - start);
      for (std::size_t s = 0; s < requests_; ++s) {
        if (run.tokens[s].size() == kOutputs && run.completed_seconds[s] == 0) {
          run.completed_seconds[s] = elapsed;
        }
      }
      run.steps.push_back(std::move(step));
    }
    return {};
  }();
  run.seconds = en::support::Seconds(Clock::now() - start);
  run.graphs = Difference(graph_stats_, graph_before);
  run.mxfp8_pairs = mxfp8_pairs_ - mx_before;
  run.routed_pairs = routed_pairs_ - routed_before;
  run.packed_bytes = packed_bytes_ - pack_before;
  for (std::size_t s = 0; s < requests_; ++s) {
    run.final_pending[s] = slots_[s].pending;
  }
  if (!generated) {
    return generated;
  }
  if (!std::isfinite(run.seconds) || run.seconds <= 0 ||
      !std::ranges::all_of(std::span(run.completed_seconds).first(requests_),
                           [](double v) { return std::isfinite(v) && v > 0; }) ||
      run.graphs.refused != 0 || run.graphs.captured == 0 || run.graphs.replayed == 0 ||
      run.graphs.eager + run.graphs.captured + run.graphs.replayed != 2 * run.steps.size() ||
      coverage_.violations != 0 ||
      target_plans_.graphs() + draft_plans_.graphs() > kMaxProofGraphs) {
    return Error("C2 natural timing lacks finite completion or actual bounded graph execution");
  }
  run.complete = true;
  return {};
}

Proof::Status Proof::GenerationControls(const Requests<std::vector<std::int32_t>>& prompts,
                                        const std::filesystem::path& out) {
  if (!natural_) {
    return Error("C2 generation requires the explicit private natural mode");
  }
  if (auto initialized = Initialize(prompts); !initialized) {
    return initialized;
  }
  if (auto begun = owner_.node_.BeginRequest(stream(), closure_, "C2 natural shared-weight A/B");
      !begun) {
    return begun;
  }
  const auto checked = [&]() -> Status {
    if (auto shapes = GenerationShapeControls(out); !shapes) {
      return shapes;
    }
    for (std::size_t arm = 0; arm < generation_runs_.size(); ++arm) {
      if (auto reset = Reset(); !reset) {
        return reset;
      }
      const bool batch = arm == 1 || arm == 2;
      auto& run = generation_runs_[arm];
      const auto generated = Generate(batch, run);
      // Preserve partial trajectories even if a later kernel/control failed.
      for (std::size_t s = 0; s < requests_; ++s) {
        if (auto written =
                Write(out / std::format("run{}-slot{}-tokens.i32", arm, s), run.tokens[s]);
            !written) {
          return written;
        }
      }
      if (!generated) {
        return generated;
      }
      if (arm != 0 &&
          (run.tokens != generation_runs_[0].tokens || run.steps != generation_runs_[0].steps)) {
        return Error(std::format("C2 natural IDs/acceptance/cursor trace mismatch in run{}", arm));
      }
      if (auto state = GenerationState(arm, requests_ == 4 ? 4U : 3U, out); !state) {
        return state;
      }
    }
    const std::uint64_t positions = requests_ == 4 ? 5 : 4;
    const auto small = (positions - 1) * 4 * requests_;
    if (new_shape_output_controls_ != small || new_shape_state_controls_ != small ||
        state_controls_ != positions * 4 * requests_ ||
        reference_stats_.captures != positions * requests_ ||
        reference_stats_.comparisons != positions * 3 * requests_) {
      return Error("C2 natural missing small-shape or full final-state controls");
    }
    return {};
  }();
  const auto ended = owner_.node_.EndRequest(stream());
  if (!checked || !ended) {
    std::string error = checked ? "" : checked.error();
    if (!ended) {
      error += (error.empty() ? "" : "; ") + ended.error();
    }
    return Error(std::move(error));
  }
  complete_ = true;
  return {};
}

std::string Proof::GenerationReceipt() const {
  const auto row = [this](const auto& values) {
    return Numbers(std::span(values).first(requests_));
  };
  const auto nested = [this](const auto& values) {
    std::string out;
    for (std::size_t s = 0; s < requests_; ++s) {
      out += (out.empty() ? "" : ",") + Numbers(values[s]);
    }
    return "[" + out + "]";
  };
  std::string arms;
  for (std::size_t arm = 0; arm < generation_runs_.size(); ++arm) {
    const auto& run = generation_runs_[arm];
    std::string trace;
    Requests<std::uint64_t> drafted{};
    Requests<std::uint64_t> offered{};
    Requests<std::uint64_t> accepted{};
    Requests<std::uint64_t> verifies{};
    Requests<std::array<std::uint64_t, 3>> accepted_positions{};
    std::array<std::uint64_t, 1U << kMaxRequests> active_counts{};
    for (const auto& step : run.steps) {
      ++active_counts[step.active];
      for (std::size_t s = 0; s < requests_; ++s) {
        if ((step.active & (1U << s)) == 0) {
          continue;
        }
        drafted[s] += 3;
        offered[s] += step.rows[s] - 1;
        accepted[s] += step.keep[s] - 1;
        ++verifies[s];
        for (std::uint32_t i = 0; i + 1 < step.keep[s]; ++i) {
          ++accepted_positions[s][i];
        }
      }
      trace += std::format(
          R"({}{{"active":{},"first":{},"rows":{},"keep":{},"drafts":{},"verdicts":{}}})",
          trace.empty() ? "" : ",", step.active, row(step.first), row(step.rows), row(step.keep),
          nested(step.drafts), nested(step.verdicts));
    }
    const auto& g = run.graphs;
    std::size_t actual_outputs = 0;
    Requests<double> rates{};
    Requests<std::size_t> counts{};
    std::string token_sha;
    for (std::size_t s = 0; s < requests_; ++s) {
      counts[s] = run.tokens[s].size();
      actual_outputs += run.tokens[s].empty() ? 0 : run.tokens[s].size() - 1;
      rates[s] = run.completed_seconds[s] > 0 ? 255 / run.completed_seconds[s] : 0;
      token_sha += std::format("{}\"{}\"", token_sha.empty() ? "" : ",", Sha(run.tokens[s]));
    }
    arms += std::format(
        R"({}{{"arm":{},"batch":{},"complete":{},"seconds":{},"aggregate_timed_outputs":{},"aggregate_tok_s":{},)"
        R"("completion_seconds":{},"request_decode_tok_s":{},"output_counts":{},"final_pending":{},)"
        R"("tokens_sha256":[{}],"draft_seconds":{},"verify_seconds":{},"settle_seconds":{},)"
        R"("drafted":{},"offered":{},"accepted":{},"verifies":{},"accepted_positions":{},)"
        R"("active_wave_counts":{},"trace":[{}],"mxfp8_pairs":{},"routed_launch_pairs":{},"packed_input_bytes":{},)"
        R"("eager":{},"captured":{},"replayed":{},"dropped":{},"refused":{},"nodes":{},)"
        R"("capture_seconds":{},"instantiate_seconds":{},"graph_free_memory_delta_bytes":{}}})",
        arms.empty() ? "" : ",", arm, arm == 1 || arm == 2 ? "true" : "false",
        run.complete ? "true" : "false", run.seconds, actual_outputs,
        run.seconds > 0 ? static_cast<double>(actual_outputs) / run.seconds : 0,
        row(run.completed_seconds), row(rates), row(counts), row(run.final_pending), token_sha,
        run.draft_seconds, run.verify_seconds, run.settle_seconds, row(drafted), row(offered),
        row(accepted), row(verifies), nested(accepted_positions),
        Numbers(std::span(active_counts).first(1U << requests_)), trace, run.mxfp8_pairs,
        run.routed_pairs, run.packed_bytes, g.eager, g.captured, g.replayed, g.dropped, g.refused,
        g.nodes, g.capture_seconds, g.instantiate_seconds, g.memory_bytes);
  }
  std::string initial;
  for (const auto& slot : std::span(slots_).first(requests_)) {
    initial += std::format("{}\"{}\"", initial.empty() ? "" : ",", Sha(slot.initial));
  }
  std::string references;
  for (std::size_t index = 0; index < (requests_ == 4 ? 5U : 4U); ++index) {
    for (const auto& state : std::span(references_[index]).first(requests_)) {
      references += std::format("{}\"{}\"", references.empty() ? "" : ",", state.sha256());
    }
  }
  std::string final_sha;
  for (std::size_t s = 0; s < requests_; ++s) {
    final_sha += std::format("{}\"{}\"", final_sha.empty() ? "" : ",", generation_state_sha_[s]);
  }
  Requests<std::uint32_t> prompt_counts = {8192, 8192, 8192, 8192};
  return std::format(
      R"({{"schema":"llmp-qwen-c{}-natural-shared-rows-v1","complete":{},"requests":{},)"
      R"("prompt_tokens":{},"output_cap":256,"fixed_depth":3,"draft_vocab":47172,)"
      R"("stops_literal":true,"timed_output_mode":"ids_only","cold_private_plans_each_arm":true,)"
      R"("shared_weight_owners":1,"live_request_states":{},"prefill_owner_state":1,)"
      R"("baseline_device_bytes":{},"copied_checkpoint_bytes":{},"initial_history_sha256":[{}],)"
      R"("new_shape_output_controls":{},"new_shape_state_controls":{},"state_controls":{},)"
      R"("state_captures":{},"state_comparisons":{},"state_read_bytes":{},)"
      R"("state_capture_seconds":{},"state_comparison_seconds":{},"final_state_sha256":[{}],)"
      R"("reference_state_sha256":[{}],)"
      R"("host_control_allowance_bytes":{},"private_plan_cap_per_phase":12,"graph_cap":16,)"
      R"("workspace_probes":{},"workspace_masks":{},"conservative_activation_bytes":{},"activation_bytes":{},"pool_bytes":{},)"
      R"("coverage_violations":{},"arms":[{}]}})",
      requests_, complete_ && !failed_ ? "true" : "false", requests_, row(prompt_counts), requests_,
      requests_ * checkpoint_bytes_, copied_checkpoint_bytes_, initial, new_shape_output_controls_,
      new_shape_state_controls_, state_controls_, reference_stats_.captures,
      reference_stats_.comparisons,
      reference_stats_.capture_bytes + reference_stats_.comparison_bytes,
      reference_stats_.capture_seconds, reference_stats_.comparison_seconds, final_sha, references,
      std::uint64_t{requests_} << 26, workspace_probes_, workspace_masks_,
      conservative_activation_bytes_, activations_, pool_, coverage_.violations, arms);
}
}  // namespace llmp::benchmarks::qwen_batch
