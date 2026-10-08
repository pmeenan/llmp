// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The serving commands (commands.h) on a started server (serving.h).

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <format>
#include <fstream>
#include <iterator>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

#include "base/json.h"
#include "base/report.h"
#include "base/sha256.h"
#include "runtime/commands.h"
#include "runtime/intake_limits.h"
#include "runtime/serving.h"

namespace llmp::runtime {
namespace {

constexpr std::uint32_t kShortContext = 16;  // an LLM A's prompt at 0 context

std::unexpected<std::string> Error(std::string what) { return std::unexpected(std::move(what)); }

double Seconds(Clock::duration d) { return std::chrono::duration<double>(d).count(); }

double GiB(std::uint64_t bytes) { return static_cast<double>(bytes) / (1ULL << 30U); }

void Print(std::FILE* out, std::string_view text) {
  (void)std::fwrite(text.data(), 1, text.size(), out);
  (void)std::fputc('\n', out);
  (void)std::fflush(out);
}

std::string Quoted(std::string_view text) {
  std::string out;
  base::json::AppendQuoted(text, out);
  return out;
}

std::string Sha256(std::span<const std::byte> bytes) {
  return base::ToHex(base::Sha256().Update(bytes).Finish());
}

bool SameBits(std::span<const float> a, std::span<const float> b) {
  return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size_bytes()) == 0;
}

std::string Tokens(std::span<const std::int32_t> tokens) {
  std::string out;
  for (const std::int32_t t : tokens) {
    out += std::format("{}{}", out.empty() ? "" : ",", t);
  }
  return out;
}

// Complete rows, framed by each uint64 row length (native little endian on
// supported targets), so differently partitioned row bytes cannot alias.
std::string LogitsDigest(const Generation& generation) {
  base::Sha256 hash;
  for (const auto& row : generation.logits) {
    const std::uint64_t length = row.size();
    hash.Update(std::as_bytes(std::span(&length, 1)));
    hash.Update(std::as_bytes(std::span(row)));
  }
  return base::ToHex(hash.Finish());
}

// A row's highest logits, highest first, as JSON pairs [token, logit]: the
// report's view of a prefill's last row, to compare prefills (chunk sizes).
std::string TopLogits(std::span<const float> row) {
  constexpr std::size_t kTop = 8;
  std::vector<std::int32_t> ids(row.size());
  for (std::size_t i = 0; i < ids.size(); ++i) {
    ids[i] = static_cast<std::int32_t>(i);
  }
  const std::size_t n = std::min(kTop, ids.size());
  std::partial_sort(ids.begin(), ids.begin() + static_cast<std::ptrdiff_t>(n), ids.end(),
                    [&](std::int32_t a, std::int32_t b) {
                      const float x = row[static_cast<std::size_t>(a)];
                      const float y = row[static_cast<std::size_t>(b)];
                      return x > y || (x == y && a < b);
                    });
  std::string out;
  for (std::size_t i = 0; i < n; ++i) {
    out += std::format("{}[{},{:.6g}]", i == 0 ? "" : ",", ids[i],
                       row[static_cast<std::size_t>(ids[i])]);
  }
  return out;
}

std::string HostPhasesJson(const SwapHostPhases& p) {
  return std::format(
      R"({{"room_seconds":{},"gather_seconds":{},"selection_bookkeeping_seconds":{},)"
      R"("take_seconds":{},"heap_release_seconds":{},"partition_seconds":{},)"
      R"("reclaim_calls":{},"heap_release_calls":{}}})",
      p.room_seconds, p.gather_seconds, p.selection_bookkeeping_seconds, p.take_seconds,
      p.heap_release_seconds, p.partition_seconds, p.reclaim_calls, p.heap_release_calls);
}

std::string TimingEventJson(const scheduler::BackingTimingEvent& e) {
  return std::format(R"({{"serial":{},"start_ns":{},"end_ns":{}}})", e.serial, e.start_ns,
                     e.end_ns);
}

std::string TimingTotalsJson(const scheduler::BackingTimingTotals& t) {
  return std::format(
      R"({{"started":{},"completed":{},"total_ns":{},"first":{},"last":{},"longest":{}}})",
      t.started, t.completed, t.total_ns, TimingEventJson(t.first), TimingEventJson(t.last),
      TimingEventJson(t.longest));
}

std::string ParkStageJson(const scheduler::BackingParkStageTotals& t) {
  return std::format(R"({{"completed":{},"total_ns":{},"longest":{}}})", t.completed, t.total_ns,
                     TimingEventJson(t.longest));
}

std::string BackingTimingJson(const scheduler::BackingTimingStats& t) {
  if (!t.enabled) return "null";
  return std::format(R"({{"create_serial":{},"current_create_reserve":{},"current_create":{},)"
                     R"("reserve":{},"ordinary":{},"current_park":{},"park":{},)"
                     R"("park_metadata":{},"park_stash":{},"park_publication":{}}})",
                     t.create_serial, t.current_create_reserve ? "true" : "false",
                     TimingEventJson(t.current_create), TimingTotalsJson(t.reserve),
                     TimingTotalsJson(t.ordinary), TimingEventJson(t.current_park),
                     TimingTotalsJson(t.park), ParkStageJson(t.park_metadata),
                     ParkStageJson(t.park_stash), ParkStageJson(t.park_publication));
}

std::string CreateCountsJson(const scheduler::BackingCreateStats& c) {
  return std::format(R"({{"reserve_attempts":{},"reserve_failures":{},"ordinary_attempts":{},)"
                     R"("ordinary_failures":{},"timing":{}}})",
                     c.reserve_attempts, c.reserve_failures, c.ordinary_attempts,
                     c.ordinary_failures, BackingTimingJson(c.timing));
}

std::string SwapDiagnosticJson(const SwapDiagnostic& d) {
  if (!d.enabled) return "null";
  return std::format(R"({{"eviction_split_available":{},"eviction_prepare_seconds":{},)"
                     R"("eviction_retire_seconds":{},"requested_ns":{},"scheduler_started_ns":{},)"
                     R"("scheduler_evicted_ns":{},"scheduler_loaded_ns":{},)"
                     R"("requested":{},"ready":{},"finished":{},"creates_requested":{},)"
                     R"("creates_loaded":{},"creates_ready":{},"creates_finished":{}}})",
                     d.eviction_split_available ? "true" : "false", d.eviction_prepare_seconds,
                     d.eviction_retire_seconds, d.requested_ns, d.scheduler_started_ns,
                     d.scheduler_evicted_ns, d.scheduler_loaded_ns, HostPhasesJson(d.requested),
                     HostPhasesJson(d.ready), HostPhasesJson(d.finished),
                     CreateCountsJson(d.creates_requested), CreateCountsJson(d.creates_loaded),
                     CreateCountsJson(d.creates_ready), CreateCountsJson(d.creates_finished));
}

std::string PartsJson(const SwapParts& p) {
  return std::format(
      R"({{"from":{},"to":{},"with_state":{},"evict":{:.6f},"restore":{:.6f},"page_in":{:.6f},)"
      R"("setup":{:.6f},"ready":{:.6f},"release_after":{:.6f},"diagnostic":{},"evicted":{},"loaded":{},)"
      R"("read_bytes":{},"read_submitted_bytes":{},"occupancy_requested":{},"occupancy_released":{},)"
      R"("retained_incoming_weight_bytes":{},)"
      R"("retained_outgoing_weight_bytes":{},"evicted_weight_bytes":{},"spilled_bytes":{},"handed_off":{},"released_unused":{}}})",
      Quoted(p.from), Quoted(p.to), p.with_state ? "true" : "false", p.evict, p.restore, p.page_in,
      p.setup, p.total, p.release, SwapDiagnosticJson(p.diagnostic), p.evicted, p.loaded,
      p.read_bytes, p.read_submitted_bytes, p.occupancy_requested, p.occupancy_released,
      p.retained_incoming_weight_bytes, p.retained_outgoing_weight_bytes, p.evicted_weight_bytes,
      p.spilled_bytes, p.handed_off, p.released_unused);
}

std::string PartsLine(const SwapParts& p) {
  const double gb = static_cast<double>(p.read_bytes) / 1e9;
  const double reading = p.page_in + p.restore;
  return std::format(
      "{} -> {}: ready in {:.3f} s = evict{} {:.3f} + restore {:.3f} + page-in {:.3f} ({:.2f} "
      "GB, {:.2f} GB/s) + setup {:.3f}; backing released {:.3f} s later",
      p.from.empty() ? "(nothing)" : p.from, p.to, p.total, p.with_state ? " and spill" : "",
      p.evict, p.restore, p.page_in, gb, reading > 0 ? gb / reading : 0.0, p.setup, p.release);
}

Status WriteReport(const std::filesystem::path& path, const std::string& json) {
  if (path.empty()) {
    return {};
  }
  std::ofstream file(path, std::ios::binary | std::ios::trunc);
  file << json;
  file.close();
  if (!file) {
    return Error(std::format("cannot write the report {}", path.string()));
  }
  return {};
}

}  // namespace

// ---------------------------------------------------------------- chat

Status RunChat(Server& server, const ChatOptions& o, const ServingOptions& serving,
               std::FILE* out) {
  std::map<std::string, std::vector<chat::Message>> conversations;
  std::string turns;
  for (std::size_t index = 0; index < o.turns.size(); ++index) {
    const Turn& turn = o.turns[index];
    Served* m = server.Find(turn.model);
    if (m == nullptr) {
      return Error(std::format("turn {}: no registered model is named {}", index + 1, turn.model));
    }
    if (!m->llm() && turn.text != serving.image_prompt) {
      return Error(
          std::format("turn {}: this build's image runner generates the prompt it registered with "
                      "(--image-prompt); another needs a new process",
                      index + 1));
    }
    server.memory().Reset();
    SwapParts parts;
    const bool swapped = server.resident() != m;
    if (auto r = server.Activate(*m, parts); !r) {
      return Error(std::format("turn {}: {}", index + 1, r.error()));
    }
    const Clock::time_point requested = swapped ? parts.requested : Clock::now();
    std::string json;
    if (m->llm()) {
      auto& l = static_cast<Llm&>(*m);
      std::vector<chat::Message>& messages = conversations[m->name()];
      if (o.fresh) {
        messages.clear();
      }
      messages.push_back({.role = chat::Role::kUser,
                          .content = turn.text,
                          .reasoning_content = std::nullopt,
                          .tool_calls = {}});
      chat::Conversation conversation;
      conversation.messages = messages;
      l.Defaults(conversation);
      conversation.max_render_bytes = l.render_bytes();
      std::uint32_t stable_boundary = 0;
      auto tokens = l.RenderChat(conversation, &stable_boundary);
      if (!tokens) {
        return Error(std::format("turn {}: {}", index + 1, tokens.error()));
      }
      // --max-tokens is bounded by the model's context, not a fixed cap
      // (D-102): the turn generates at most what the context has left.
      if (tokens->size() >= l.usable_context()) {
        return Error(std::format("turn {}: its {} tokens do not fit {}'s usable context of {}",
                                 index + 1, tokens->size(), l.name(), l.usable_context()));
      }
      const auto room = static_cast<std::uint32_t>(l.usable_context() - tokens->size());
      Generation generation;
      std::uint32_t reused = 0;
      double prefill = 0;
      PrefillRun chunks;
      std::string top;  // the prefill's last row: its highest logits
      Clock::time_point first;
      auto ran = server.InRequest(*m, [&]() -> Status {
        std::vector<float> last;
        const auto start = Clock::now();
        if (auto r = l.PreparePrompt(*tokens, stable_boundary, last, reused, {}, &chunks, o.fresh);
            !r) {
          return r;
        }
        first = Clock::now();
        prefill = Seconds(first - start);
        top = TopLogits(last);
        return l.Generate(last,
                          {.max_tokens = std::min(o.max_tokens, room),
                           .extra_stops = l.ChatStops(),
                           .stop = !o.ignore_stop,
                           .keep_logits = false,
                           .sampling = std::nullopt,
                           .seed = 0,
                           .on_tokens = {}},
                          generation);
      });
      if (!ran) {
        return Error(std::format("turn {}: {}", index + 1, ran.error()));
      }
      std::vector<std::int32_t> shown = generation.tokens;
      if (generation.stopped && !shown.empty()) {
        shown.pop_back();  // the stop token
      }
      const std::string text = l.Detokenize(shown);
      // The reply as the next turn's render takes it: with the template's
      // thinking on, the reasoning before "</think>" and the answer after
      // it (the chat route's output parser, M5, does this properly).
      chat::Message reply{.role = chat::Role::kAssistant,
                          .content = text,
                          .reasoning_content = std::nullopt,
                          .tool_calls = {}};
      if (const std::size_t end = text.find("</think>"); end != std::string::npos) {
        std::string_view answer = std::string_view(text).substr(end + 8);
        while (!answer.empty() && (answer.front() == '\n' || answer.front() == ' ')) {
          answer.remove_prefix(1);
        }
        reply.reasoning_content = text.substr(0, end);
        reply.content = std::string(answer);
      }
      messages.push_back(std::move(reply));
      const std::size_t prefilled = tokens->size() - reused;
      const double rate =
          generation.tokens.size() > 1 && generation.decode_seconds > 0
              ? static_cast<double>(generation.tokens.size() - 1) / generation.decode_seconds
              : 0.0;
      const double acceptance = generation.drafted > 0 ? static_cast<double>(generation.accepted) /
                                                             static_cast<double>(generation.drafted)
                                                       : 0.0;
      Print(
          out,
          std::format(
              "[{}] {} prompt tokens ({} reused), prefill {:.3f} s ({:.1f} tok/s; {} chunks of {} "
              "rows, the longest {:.3f} s); first "
              "token {:.3f} s after the request; {} tokens{} in {:.3f} s after it ({:.2f} "
              "tok/s{}); peak {:.1f} GiB",
              m->name(), tokens->size(), reused, prefill,
              prefill > 0 ? static_cast<double>(prefilled) / prefill : 0.0, chunks.chunks,
              l.max_rows(), chunks.longest, Seconds(first - requested), generation.tokens.size(),
              generation.stopped ? " (stopped)" : "", generation.decode_seconds, rate,
              l.speculative() ? std::format(", acceptance {:.3f}", acceptance) : "",
              GiB(server.memory().peak())));
      Print(out, text);
      json = std::format(
          R"({{"model":{},"kind":"llm","prompt_tokens":{},"reused":{},"prefill_seconds":{:.6f},)"
          R"("prefill_chunk":{},"prefill_chunks":{},"longest_chunk_seconds":{:.6f},)"
          R"("first_token_seconds":{:.6f},"generated":{},"stopped":{},"decode_seconds":{:.6f},)"
          R"("tokens_per_second":{:.3f},"speculative":{},"drafted":{},"accepted":{},"steps":{},)"
          R"("peak_bytes":{},"prefill_top":[{}],"prompt_ids":[{}],"tokens":[{}],"text":{})",
          Quoted(m->name()), tokens->size(), reused, prefill, l.max_rows(), chunks.chunks,
          chunks.longest, Seconds(first - requested), generation.tokens.size(),
          generation.stopped ? "true" : "false", generation.decode_seconds, rate,
          l.speculative() ? "true" : "false", generation.drafted, generation.accepted,
          generation.steps, server.memory().peak(), top, Tokens(*tokens), Tokens(generation.tokens),
          Quoted(text));
    } else {
      auto& image = static_cast<Image&>(*m);
      std::string step;
      std::string pixels;
      Clock::time_point first;
      Clock::time_point done;
      auto ran = server.InRequest(*m, [&]() -> Status {
        if (auto r = image.FirstOutput(step); !r) {
          return r;
        }
        first = Clock::now();
        auto finished = image.Finish(pixels);
        done = Clock::now();
        return finished;
      });
      if (!ran) {
        return Error(std::format("turn {}: {}", index + 1, ran.error()));
      }
      Print(out, std::format("[{}] first denoising step {:.3f} s after the request; generated in "
                             "{:.3f} s; pixels {}; peak {:.1f} GiB",
                             m->name(), Seconds(first - requested), Seconds(done - requested),
                             pixels, GiB(server.memory().peak())));
      json = std::format(
          R"({{"model":{},"kind":"image","first_step_seconds":{:.6f},"generation_seconds":{:.6f},)"
          R"("pixels_sha256":"{}","peak_bytes":{})",
          Quoted(m->name()), Seconds(first - requested), Seconds(done - requested), pixels,
          server.memory().peak());
    }
    if (swapped) {
      if (auto r = server.FinishSwap(parts); !r) {
        return r;
      }
      Print(out, std::format("[{}] {}", m->name(), PartsLine(parts)));
      json += std::format(R"(,"swap":{}}})", PartsJson(parts));
    } else {
      json += "}";
    }
    turns += (turns.empty() ? "" : ",\n  ") + json;
  }
  std::string models;
  for (const auto& m : server.models()) {
    models += std::format("{}{}:{}", models.empty() ? "" : ",\n  ", Quoted(m->name()), m->extra());
  }
  return WriteReport(
      serving.report,
      std::format("{{\"command\":\"chat\",\"budget\":{},\"fixed\":{},\"workspace\":{},\n"
                  " \"native_token_history_bytes\":{},\"native_token_catalog_bytes\":{},\n"
                  " \"models\":{{\n  {}}},\n \"turns\":[\n  {}]}}\n",
                  server.budget(), server.fixed_bytes(), server.workspace_bytes(),
                  server.token_history_bytes(), server.token_catalog_bytes(), models, turns));
}

// ---------------------------------------------------------------- swap-table

namespace {

// One row of the table: a swap and its endpoint.
struct Row {
  std::string pair;
  std::string name;
  SwapParts parts;
  double first = 0;   // from the end of setup to the endpoint
  double total = 0;   // from the request to the endpoint (A's state digest excluded)
  double digest = 0;  // hashing A's restored state, outside the parts
  std::uint64_t peak = 0;
  bool exact = true;
  bool state_exact = true;
  std::string output;
  std::string context_ids_sha256;  // actual int32 context IDs, not the source text
  std::uint64_t initialized_state_bytes = 0;
  std::string state_before_sha256;
  std::string state_restored_sha256;
  std::string reference_logits_sha256;
  std::string continued_logits_sha256;
  std::uint64_t continuation_vocabulary = 0;
  std::uint64_t continuation_logit_rows = 0;
  std::uint64_t continuation_logit_bytes = 0;
  GraphCounts graphs;  // an LLM A's steps after it (captured, replayed, launched)
  std::uint64_t graphs_kept = 0;
  double plan_seconds = 0;
  double demand_seconds = 0;
};

class Table {
 public:
  Table(Server& server, const SwapTableOptions& o, const ServingOptions& serving, std::FILE* out)
      : server_(server), o_(o), serving_(serving), out_(out) {}

  Status Run();

 private:
  Status Pair(Served& a, Served& b);
  // B's first output from a cleared state (an LLM's first token for the
  // short prompt, the image's first denoising step): its hash.
  Status FirstOutput(Served& b, std::string& hash);
  Status Context(Llm& a);
  // A row's end: its peak, the release after it, printed and kept.
  Status Finish(Row& row);
  std::string Json() const;

  Server& server_;
  const SwapTableOptions& o_;
  const ServingOptions& serving_;
  std::FILE* out_;
  std::unique_ptr<RequestMemory> memory_;  // the request memory, once a context is read
  MemoryCharge text_charge_;               // text_'s bytes in it
  std::string text_;
  std::map<std::string, std::vector<std::int32_t>> context_;  // by LLM
  std::map<std::string, std::string> first_outputs_;          // B's, by pair
  std::map<std::string, std::string> pixels_;                 // an image A's control
  std::vector<Row> rows_;
  std::vector<std::string> positioning_;
  std::vector<std::string> problems_;
};

Status Table::Context(Llm& a) {
  if (context_.contains(a.name())) {
    return {};
  }
  if (text_.empty()) {
    if (o_.context_text.empty()) {
      return Error("an LLM A needs --context-text");
    }
    // The file is read whole and tokenized, so the request memory bounds it
    // as it bounds the chat route's requests (D-102; runtime/intake_limits.h):
    // the text, then its tokenization's working set, charged to the pool
    // the start set apart and, past it, to the budget through the reclaim
    // order (this thread is the node's driver).
    if (memory_ == nullptr) {
      memory_ =
          std::make_unique<RequestMemory>(server_.request_memory(), server_.request_capacity());
      memory_->SetDriver(std::this_thread::get_id(),
                         [this](std::uint64_t to) { return server_.SetRequestMemory(to); });
    }
    const std::uint64_t most = memory_->capacity();
    std::error_code size_error;
    const auto size = std::filesystem::file_size(o_.context_text, size_error);
    if (size_error || size == 0 || size > most || !text_charge_.Add(*memory_, size)) {
      return Error(
          std::format("{} is empty, unreadable or over {} bytes (the request memory, "
                      "[client] request_memory_bytes)",
                      o_.context_text.string(), most));
    }
    std::ifstream file(o_.context_text, std::ios::binary);
    text_.assign(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
    if (text_.empty() || text_.size() > most) {
      return Error(
          std::format("{} is empty, unreadable or over {} bytes", o_.context_text.string(), most));
    }
  }
  auto ids = a.EncodeText(text_, memory_.get());
  if (!ids) {
    return std::unexpected(ids.error() + " ([client] request_memory_bytes bounds it)");
  }
  if (ids->size() < o_.context_tokens) {
    return Error(std::format("{} has {} of {}'s tokens, fewer than {}", o_.context_text.string(),
                             ids->size(), a.name(), o_.context_tokens));
  }
  if (o_.context_tokens + o_.continue_tokens + 1 >= a.context()) {
    return Error(std::format("{}'s context of {} does not hold {} tokens and {} more", a.name(),
                             a.context(), o_.context_tokens, o_.continue_tokens));
  }
  ids->resize(o_.context_tokens);
  context_[a.name()] = std::move(*ids);
  return {};
}

Status Table::FirstOutput(Served& b, std::string& hash) {
  return server_.InRequest(b, [&]() -> Status {
    if (!b.llm()) {
      return static_cast<Image&>(b).FirstOutput(hash);
    }
    auto& l = static_cast<Llm&>(b);
    chat::Conversation c;
    c.messages.push_back({.role = chat::Role::kUser,
                          .content = o_.short_prompt,
                          .reasoning_content = std::nullopt,
                          .tool_calls = {}});
    l.Defaults(c);
    c.max_render_bytes = l.render_bytes();
    auto tokens = l.RenderChat(c);
    if (!tokens) {
      return std::unexpected(tokens.error());
    }
    if (auto r = l.Clear(); !r) {
      return r;
    }
    std::vector<float> last;
    if (auto r = l.Prefill(*tokens, last); !r) {
      return r;
    }
    Generation first;
    if (auto r = l.Generate(last,
                            {.max_tokens = 1,
                             .stop = false,
                             .keep_logits = false,
                             .sampling = std::nullopt,
                             .seed = 0,
                             .on_tokens = {}},
                            first);
        !r) {
      return r;
    }
    if (first.tokens.size() != 1) {
      return Error("the swap endpoint did not generate its first token");
    }
    hash = Sha256(std::as_bytes(std::span(last)));
    // B's conversation ends here: nothing of it is spilled when A returns
    // (as the harness's B, whose state was cleared each time).
    l.Forget();
    return {};
  });
}

Status Table::Finish(Row& row) {
  row.peak = server_.memory().peak();
  if (auto r = server_.FinishSwap(row.parts); !r) {
    return r;
  }
  const SwapParts& p = row.parts;
  const double gb = static_cast<double>(p.read_bytes) / 1e9;
  std::string verdict = "exact";
  if (!row.exact) {
    verdict = "DIFFERS";
  } else if (!row.state_exact) {
    verdict = "STATE DIFFERS";
  }
  Print(out_, std::format("{} {}: total {:.3f} s = evict {:.3f} + restore {:.3f} + page-in {:.3f} "
                          "({:.2f} GB, {:.2f} GB/s) + setup {:.3f} + first output {:.3f} (planning "
                          "{:.3f}, demand reads {:.3f}); peak {:.1f} GiB; handed off {}; {}",
                          row.pair, row.name, row.total, p.evict, p.restore, p.page_in, gb,
                          p.page_in + p.restore > 0 ? gb / (p.page_in + p.restore) : 0.0, p.setup,
                          row.first, row.plan_seconds, row.demand_seconds, GiB(row.peak),
                          p.handed_off, verdict));
  rows_.push_back(row);
  return {};
}

Status Table::Pair(Served& a, Served& b) {
  const std::string pair = std::format("{} <-> {}", a.name(), b.name());
  Print(out_, std::format("== {} ({} as A)", pair, a.name()));
  // The other models' conversations from earlier pairs are done with: no
  // swap in this pair spills or restores them (B starts from a cleared
  // state, as the harness's did).
  for (const auto& m : server_.models()) {
    if (m.get() != &a && m->llm()) {
      static_cast<Llm&>(*m).Forget();
    }
  }
  if (server_.resident() != &a) {
    // Positioning: A resident before its pair (not a row of the table).
    SwapParts parts;
    if (auto r = server_.Activate(a, parts); !r) {
      return r;
    }
    if (auto r = server_.FinishSwap(parts); !r) {
      return r;
    }
    positioning_.push_back(PartsJson(parts));
    Print(out_, "positioning: " + PartsLine(parts));
  }
  std::vector<std::int32_t> context;
  if (a.llm()) {
    if (auto r = Context(static_cast<Llm&>(a)); !r) {
      return r;
    }
    context = context_.at(a.name());
  } else if (!pixels_.contains(a.name())) {
    // An image A's control: one generation, never swapped.
    std::string first;
    std::string pixels;
    if (auto r = server_.InRequest(a,
                                   [&]() -> Status {
                                     auto& image = static_cast<Image&>(a);
                                     if (auto s = image.FirstOutput(first); !s) {
                                       return s;
                                     }
                                     return image.Finish(pixels);
                                   });
        !r) {
      return r;
    }
    pixels_[a.name()] = pixels;
    Print(out_, std::format("control: {} generated, pixels {}", a.name(), pixels));
    if (!o_.image_expect.empty() && pixels != o_.image_expect) {
      problems_.push_back(std::format("{}'s pixels ({}) are not the expected {}", a.name(), pixels,
                                      o_.image_expect));
    }
  }
  const GenerateOptions continuation{.max_tokens = o_.continue_tokens,
                                     .stop = false,
                                     .keep_logits = true,
                                     .sampling = std::nullopt,
                                     .seed = 0,
                                     .on_tokens = {}};
  const auto cycle_run = [&](int cycle, bool with_context) -> Status {
    const std::string use = cycle == 0 ? "first use" : "prepared";
    std::string held = "image";
    if (a.llm()) {
      held = with_context ? std::format("{} context tokens", context.size()) : "0 context";
    }
    std::vector<float> last;
    Generation reference;
    std::string digest;
    if (a.llm() && with_context) {
      // A's context, then this state's own continuation, never swapped:
      // the state saved and hashed, the continuation run, the state put
      // back as it was.
      auto& l = static_cast<Llm&>(a);
      if (auto r = server_.InRequest(a,
                                     [&]() -> Status {
                                       if (auto s = l.Clear(); !s) {
                                         return s;
                                       }
                                       if (auto s = l.Prefill(context, last); !s) {
                                         return s;
                                       }
                                       if (auto s = server_.SaveSnapshot(l); !s) {
                                         return s;
                                       }
                                       digest = Sha256(std::span(
                                           static_cast<const std::byte*>(server_.snapshot()),
                                           l.state_snapshot_bytes()));
                                       if (auto s = l.Generate(last, continuation, reference); !s) {
                                         return s;
                                       }
                                       return server_.RestoreSnapshot(l);
                                     });
          !r) {
        return r;
      }
    } else if (a.llm()) {
      if (auto r = static_cast<Llm&>(a).Clear(); !r) {  // 0 context: nothing to spill
        return r;
      }
    }
    if (cycle == 0) {
      b.DropPlans();  // B: nothing prepared in this process
    }
    // A -> B: A's state spilled when it holds context, B's first output.
    Row ab;
    ab.pair = pair;
    ab.name = std::format("A->B, {} ({})", use, held);
    if (a.llm()) ab.context_ids_sha256 = Sha256(std::as_bytes(std::span(context)));
    server_.memory().Reset();
    if (auto r = server_.Activate(b, ab.parts); !r) {
      return r;
    }
    const double planned = b.plan_seconds();
    const double demand = b.demand_seconds();
    if (auto r = FirstOutput(b, ab.output); !r) {
      return r;
    }
    const auto done = Clock::now();
    ab.first = Seconds(done - ab.parts.ready);
    ab.total = Seconds(done - ab.parts.requested);
    ab.plan_seconds = b.plan_seconds() - planned;
    ab.demand_seconds = b.demand_seconds() - demand;
    const auto [known, inserted] = first_outputs_.try_emplace(pair + b.name(), ab.output);
    ab.exact = known->second == ab.output;
    if (!ab.exact) {
      problems_.push_back(std::format("{} {}: B's first output ({}) differs from its first run's",
                                      pair, ab.name, ab.output));
    }
    if (auto r = Finish(ab); !r) {
      return r;
    }
    if (cycle == 0) {
      a.DropPlans();  // A returns to nothing prepared
    }
    // B -> A: A's state restored, its next output.
    Row ba;
    ba.pair = pair;
    ba.name = std::format("B->A, {} ({})", use, held);
    ba.context_ids_sha256 = ab.context_ids_sha256;
    server_.memory().Reset();
    if (auto r = server_.Activate(a, ba.parts); !r) {
      return r;
    }
    const double a_planned = a.plan_seconds();
    const double a_demand = a.demand_seconds();
    const GraphCounts graphs_before = a.graphs();
    ba.graphs_kept = graphs_before.kept;
    if (!a.llm()) {
      std::string pixels;
      Clock::time_point first;
      if (auto r = server_.InRequest(a,
                                     [&]() -> Status {
                                       auto& image = static_cast<Image&>(a);
                                       if (auto s = image.FirstOutput(ba.output); !s) {
                                         return s;
                                       }
                                       first = Clock::now();
                                       return image.Finish(pixels);
                                     });
          !r) {
        return r;
      }
      ba.first = Seconds(first - ba.parts.ready);
      ba.total = Seconds(first - ba.parts.requested);
      ba.exact = pixels == pixels_.at(a.name());
      if (!ba.exact) {
        problems_.push_back(
            std::format("{} {}: the regenerated image ({}) differs from the "
                        "control's",
                        pair, ba.name, pixels));
      }
    } else if (with_context) {
      auto& l = static_cast<Llm&>(a);
      // The restored state against the state A left with, byte for byte;
      // outside the timed parts.
      const auto hashing = Clock::now();
      if (auto r = server_.InRequest(a, [&]() -> Status { return server_.SaveSnapshot(l); }); !r) {
        return r;
      }
      ba.initialized_state_bytes = l.state_snapshot_bytes();
      ba.state_before_sha256 = digest;
      ba.state_restored_sha256 = Sha256(
          std::span(static_cast<const std::byte*>(server_.snapshot()), l.state_snapshot_bytes()));
      ba.state_exact = ba.state_restored_sha256 == ba.state_before_sha256;
      if (!ba.state_exact) {
        problems_.push_back(
            std::format("{} {}: A's restored state differs from the state it "
                        "left with",
                        pair, ba.name));
      }
      ba.digest = Seconds(Clock::now() - hashing);
      // The continuation: its first step is the endpoint (the first token
      // generated after the swap); every token and logit must equal the
      // unswapped reference's.
      Generation continued;
      if (auto r = server_.InRequest(
              a, [&]() -> Status { return l.Generate(last, continuation, continued); });
          !r) {
        return r;
      }
      ba.first = Seconds(continued.first_step - ba.parts.ready) - ba.digest;
      ba.total = Seconds(continued.first_step - ba.parts.requested) - ba.digest;
      std::size_t differing = 0;
      for (std::size_t k = 0; k < continued.logits.size() && k < reference.logits.size(); ++k) {
        differing += SameBits(continued.logits[k], reference.logits[k]) ? 0 : 1;
      }
      ba.exact = differing == 0 && continued.tokens == reference.tokens &&
                 continued.logits.size() == reference.logits.size();
      ba.output = Tokens(continued.tokens);
      // Diagnostics after the endpoint was recorded; hashing does not enter
      // swap setup, first-output timing or the restored-state digest interval.
      const auto complete = [&](const Generation& generation) {
        return generation.logits.size() == o_.continue_tokens &&
               std::ranges::all_of(generation.logits, [&](const auto& row) {
                 return row.size() == last.size() &&
                        std::ranges::all_of(row, [](float v) { return std::isfinite(v); });
               });
      };
      if (last.empty() || last.size() != l.tokenizer().size() || !complete(reference) ||
          !complete(continued))
        return Error("the swap continuation needs complete finite target-vocabulary rows");
      ba.continuation_vocabulary = last.size();
      ba.reference_logits_sha256 = LogitsDigest(reference);
      ba.continued_logits_sha256 = LogitsDigest(continued);
      ba.continuation_logit_rows = continued.logits.size();
      for (const auto& row : continued.logits)
        ba.continuation_logit_bytes += row.size() * sizeof(float);
      if (!ba.exact) {
        problems_.push_back(
            std::format("{} {}: {} of {} continued steps' logits differ from the "
                        "unswapped continuation's",
                        pair, ba.name, differing, continued.logits.size()));
      }
    } else {
      // At 0 context: a short prompt from a cleared state.
      auto& l = static_cast<Llm&>(a);
      std::vector<std::int32_t> prompt(context.begin(), context.begin() + kShortContext);
      Clock::time_point first;
      if (auto r = server_.InRequest(a,
                                     [&]() -> Status {
                                       if (auto s = l.Clear(); !s) {
                                         return s;
                                       }
                                       std::vector<float> row;
                                       if (auto s = l.Prefill(prompt, row); !s) {
                                         return s;
                                       }
                                       Generation generated;
                                       if (auto s = l.Generate(row,
                                                               {.max_tokens = 1,
                                                                .stop = false,
                                                                .keep_logits = false,
                                                                .sampling = std::nullopt,
                                                                .seed = 0,
                                                                .on_tokens = {}},
                                                               generated);
                                           !s) {
                                         return s;
                                       }
                                       if (generated.tokens.size() != 1) {
                                         return Error(
                                             "the swap endpoint did not generate its first token");
                                       }
                                       first = Clock::now();
                                       ba.output = Sha256(std::as_bytes(std::span(row)));
                                       return {};
                                     });
          !r) {
        return r;
      }
      ba.first = Seconds(first - ba.parts.ready);
      ba.total = Seconds(first - ba.parts.requested);
    }
    ba.plan_seconds = a.plan_seconds() - a_planned;
    ba.demand_seconds = a.demand_seconds() - a_demand;
    const GraphCounts after = a.graphs();
    ba.graphs = {.eager = after.eager - graphs_before.eager,
                 .captured = after.captured - graphs_before.captured,
                 .replayed = after.replayed - graphs_before.replayed,
                 .refused = after.refused - graphs_before.refused,
                 .kept = after.kept};
    // A model keeps its plans and graphs through a swap (D-090 as amended
    // 2026-10-02), unless the reclaim order took them for the incoming
    // model's room: a prepared return that kept graphs replays those
    // captured before the swap (D-090's pins).
    if (cycle > 0 && a.llm() && with_context && ba.graphs_kept != 0 && ba.graphs.replayed == 0) {
      problems_.push_back(std::format(
          "{} {}: A replayed no decode graph captured before the swap ({} kept, {} replayed)", pair,
          ba.name, ba.graphs_kept, ba.graphs.replayed));
    }
    return Finish(ba);
  };
  for (std::uint32_t cycle = 0; cycle < o_.cycles; ++cycle) {
    if (auto r = cycle_run(static_cast<int>(cycle), true); !r) {
      return Error(std::format("{}: {}", pair, r.error()));
    }
  }
  if (a.llm() && o_.cycles > 0 && o_.zero_context) {
    if (auto r = cycle_run(1, false); !r) {
      return Error(std::format("{}: {}", pair, r.error()));
    }
  }
  return {};
}

Status Table::Run() {
  server_.set_handoff(o_.handoff);
  std::vector<std::pair<Served*, Served*>> pairs;
  if (o_.pairs.empty()) {
    // Every ordered pair, each A's pairs together, so a model is positioned
    // as A once.
    for (const auto& a : server_.models()) {
      for (const auto& b : server_.models()) {
        if (a != b) {
          pairs.emplace_back(a.get(), b.get());
        }
      }
    }
  } else {
    for (const auto& [a, b] : o_.pairs) {
      Served* sa = server_.Find(a);
      Served* sb = server_.Find(b);
      if (sa == nullptr || sb == nullptr) {
        return Error(std::format("no registered model is named {}", sa == nullptr ? a : b));
      }
      pairs.emplace_back(sa, sb);
    }
  }
  for (const auto& [a, b] : pairs) {
    if (auto r = Pair(*a, *b); !r) {
      return r;
    }
  }
  for (const auto& m : server_.models()) {
    if (const std::string v = m->violations(); !v.empty()) {
      problems_.push_back(std::format("{}: {}", m->name(), v));
    }
  }
  if (auto r = WriteReport(serving_.report, Json()); !r) {
    return r;
  }
  Print(out_, std::format("lowest MemAvailable: {:.2f} GiB of {:.2f} GiB at the start",
                          GiB(server_.memory().all()), GiB(server_.memory().start())));
  if (!problems_.empty()) {
    std::string all;
    for (const std::string& p : problems_) {
      Print(out_, "PROBLEM: " + p);
      all += (all.empty() ? "" : "; ") + p;
    }
    return Error(all);
  }
  return {};
}

std::string Table::Json() const {
  std::string rows;
  for (const Row& r : rows_) {
    rows += std::format(
        R"({}{{"pair":{},"name":{},"total":{:.6f},"first_output":{:.6f},"digest_seconds":{:.6f},)"
        R"("plan_seconds":{:.6f},"demand_seconds":{:.6f},"peak_bytes":{},"exact":{},)"
        R"("state_exact":{},"output":{},"context_ids_sha256":{},"initialized_state_bytes":{},)"
        R"("state_before_sha256":{},)"
        R"("state_restored_sha256":{},"reference_logits_sha256":{},"continued_logits_sha256":{},)"
        R"("continuation_vocabulary":{},"continuation_logit_rows":{},"continuation_logit_bytes":{},)"
        R"("graphs_kept":{},"graphs":{{"eager":{},"captured":{},)"
        R"("replayed":{}}},"parts":{}}})",
        rows.empty() ? "" : ",\n  ", Quoted(r.pair), Quoted(r.name), r.total, r.first, r.digest,
        r.plan_seconds, r.demand_seconds, r.peak, r.exact ? "true" : "false",
        r.state_exact ? "true" : "false", Quoted(r.output), Quoted(r.context_ids_sha256),
        r.initialized_state_bytes, Quoted(r.state_before_sha256), Quoted(r.state_restored_sha256),
        Quoted(r.reference_logits_sha256), Quoted(r.continued_logits_sha256),
        r.continuation_vocabulary, r.continuation_logit_rows, r.continuation_logit_bytes,
        r.graphs_kept, r.graphs.eager, r.graphs.captured, r.graphs.replayed, PartsJson(r.parts));
  }
  std::string problems;
  for (const std::string& p : problems_) {
    problems += (problems.empty() ? "" : ",") + Quoted(p);
  }
  std::string positioning;
  for (const std::string& p : positioning_) {
    positioning += (positioning.empty() ? "" : ",\n  ") + p;
  }
  std::string models;
  for (const auto& m : server_.models()) {
    models += std::format("{}{}:{}", models.empty() ? "" : ",\n  ", Quoted(m->name()), m->extra());
  }
  const auto stats = server_.node().Stats();
  std::string creates = "null";
  if (stats) {
    const auto& c = stats->backing_creates;
    creates = std::format(
        R"({{"reserve_attempts":{},"reserve_failures":{},"ordinary_attempts":{},)"
        R"("ordinary_failures":{},"last_failure_monotonic_ns":{},"last_failure_reserve":{},)"
        R"("last_failure_error":{},"last_failure_detail":{}}})",
        c.reserve_attempts, c.reserve_failures, c.ordinary_attempts, c.ordinary_failures,
        c.last_failure_monotonic_ns, c.last_failure_reserve ? "true" : "false",
        Quoted(providers::ToString(c.last_failure_error)), Quoted(c.last_failure_detail));
  }
  return std::format(
      "{{\"command\":\"swap-table\",\"handoff\":{},\"context_tokens\":{},\"continue\":{},"
      "\"cycles\":{},\"context_sha256\":\"{}\",\"budget\":{},\"dynamic_budget\":{},\"fixed\":{},"
      "\"native_token_history_bytes\":{},\"native_token_catalog_bytes\":{},"
      "\"backing_creates\":{},\"mem_available_start\":{},\"mem_available_low\":{},\n "
      "\"models\":{{\n  {}}},\n"
      " \"problems\":[{}],\n \"positioning\":[\n  {}],\n \"rows\":[\n  {}]}}\n",
      o_.handoff ? "true" : "false", o_.context_tokens, o_.continue_tokens, o_.cycles,
      Sha256(std::as_bytes(std::span(text_))), server_.budget(), server_.dynamic_budget(),
      server_.fixed_bytes(), server_.token_history_bytes(), server_.token_catalog_bytes(), creates,
      server_.memory().start(), server_.memory().all(), models, problems, positioning, rows);
}

}  // namespace

Status RunSwapTable(Server& server, const SwapTableOptions& options, const ServingOptions& serving,
                    std::FILE* out) {
  Table table(server, options, serving, out);
  return table.Run();
}

int RunServing(const config::NodeConfig& config, const config::RuntimeRoles& roles,
               const CommandOptions& command, std::FILE* out, std::FILE* log) {
  if (command.command == Command::kService) {
    return RunService(config, roles, log);
  }
  const auto say = [log](std::string_view text) {
    const std::string line = std::format("llmp-runtime: {}\n", base::Printable(text));
    (void)std::fwrite(line.data(), 1, line.size(), log);
    (void)std::fflush(log);
  };
  Status ran;
  {
    Server server(config, roles, command.serving, log);
    ran = server.Start(command.command == Command::kSwapTable);
    if (ran) {
      ran = command.command == Command::kChat
                ? RunChat(server, command.chat, command.serving, out)
                : RunSwapTable(server, command.table, command.serving, out);
    }
    if (auto stopped = server.TearDown(); !stopped) {
      say("stopping: " + stopped.error());
      if (ran) {
        ran = stopped;
      }
    }
  }
  if (!ran) {
    say(ran.error());
    return 1;  // kExitFailure
  }
  return 0;
}

}  // namespace llmp::runtime
