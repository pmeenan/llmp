// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// M3's swap runner (docs/plan.md, "The swap path" and "Swap runner";
// docs/experiments/fast-swap/swap.md): DeepSeek V4 Flash (A,
// dsv4_runner.h) and the Qwen2.5-0.5B FP16 fixture (B, fp16_runner.h) on
// one paged node, full swaps A→B→A in one process, each part of each swap
// timed. A harness binary, not llmp-runtime: it runs on the test
// harness's paged node (tests/support/paged_node.h).
//
//   llmp_swap_runner --dsv4-artifact DIR --fp16-artifact DIR --tokens FILE
//                      --out DIR [--text FILE] [--context-tokens N]
//                      [--continue N] [--cycles N] [--handoff on|off]
//                      [--copy-lane on|off] [--fp16-expect SHA256]
//                      [--context N] [--reload N] [--overlap]
//                      [--prompts FILE --expect DIR --generate N]
//                      [--graphs on|off] [--bench N]
//
// - A's context: --text tokenized by the native tokenizer from the
//   artifact's GGUF metadata, BOS first, its first --context-tokens
//   tokens (8,192). B's prompt: the FP16 fixture's `control` trajectory.
// - Control: A prefills its context in chunks of 512, then decodes
//   --continue tokens greedily. Each cycle then prefills it again (the
//   prefill logits must equal the control's), swaps to B (A's state is
//   written back through the zone to the spill file and A's weights
//   evicted; B's weights paged in, taking A's backing if --handoff), runs
//   B's trajectory (its logits must equal the first cycle's, and
//   --fp16-expect), swaps back (B evicted; A's state restored and weights
//   paged in, taking B's backing), and continues A: every continued
//   token's logits must equal the control's bit for bit. The first cycle
//   is first use (B never ran; A's plans dropped before its return); later
//   ones are previously prepared. A last swap pair runs at 0 context (A's
//   state is not spilled: it stays resident, and A's first token is a
//   16-token prompt's).
// - Each swap's parts: evict and spill (until every outgoing extent is
//   evicted or parked), restore (A's state resident), page-in (the
//   incoming weights resident: bytes and throughput), setup (A: the
//   hash-routing check), the first token (B: its cache cleared, its cuBLAS
//   handle made on first use, its prompt prefilled; A: one decode step,
//   planned on first use), and the handoff's counts. Backing no load took
//   is released after the swap, off the critical path, and timed.
// - --reload N: N DeepSeek → evict all (state and weights) → DeepSeek
//   reloads, the heaviest page-in, with the handoff as set.
// - --overlap: the RE-029 probe: B's weights paged in while one of A's
//   512-row prefill chunks (4,972 launches) is in flight on A's stream,
//   against B paged in alone.
// - --prompts/--expect: the paged model against the resident harness's
//   outputs (dsv4_exec.cc's `--prompts` directory): each prompt from a
//   cleared state, prefill then --generate - 1 greedy steps, every step's
//   logits compared bit for bit. Run it with the resident run's --context.
// - Peak memory: the lowest MemAvailable, sampled every 50 ms.
// - --graphs (on by default): A's decode steps as captured CUDA graphs
//   (dsv4_runner.h, D-090). The control then runs launch by launch, so every
//   continued step after a swap, replayed from a graph captured before it
//   in a prepared cycle, is compared with launch-by-launch logits. After
//   each swap back to A, A's places are checked to be the pinned ones every
//   graph names (CheckPlaces). Each swap reports how A's first token ran.
// - --bench N: decode speed as llama-bench's tg-N measures it, N one-token
//   steps from an empty context (BOS, then each step's greedy token): a
//   warm-up and three passes launch by launch, then a pass that captures
//   and three that replay; every pass's logits must equal the warm-up's
//   bit for bit. Then N replays of the first step's graph back to back in
//   one job: the step's device time without the host's part. Needs --text
//   (its BOS).
//
// Exit 1 on any failed check; swap.json in --out has every number.

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <expected>
#include <filesystem>
#include <format>
#include <fstream>
#include <iterator>
#include <memory>
#include <print>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include "base/bytes.h"
#include "base/sha256.h"
#include "catalog/catalog.h"
#include "dsv4_common.h"
#include "fp16_runner.h"
#include "paged_node.h"
#include "paged_programs.h"
#include "scheduler/scheduler.h"
#include "tokenizer/gguf.h"
#include "tokenizer/tokenizer.h"

namespace {

namespace ts = llmp::test_support;
namespace sc = llmp::scheduler;
namespace catalog = llmp::catalog;
namespace jb = llmp::benchmarks;
using llmp::base::Bytes;
using Clock = std::chrono::steady_clock;
using Status = ts::Status;

constexpr int kDsv4 = 0;  // owner and stream
constexpr int kFp16 = 1;
constexpr std::uint32_t kShortPrompt = 16;  // A's prompt at 0 context

std::unexpected<std::string> Error(std::string what) { return std::unexpected(std::move(what)); }

double Seconds(Clock::duration d) { return std::chrono::duration<double>(d).count(); }

std::string Sha256(std::span<const std::byte> bytes) {
  llmp::base::Sha256 hash;
  hash.Update(bytes);
  return llmp::base::ToHex(hash.Finish());
}

// /proc/meminfo's MemAvailable, in bytes.
std::uint64_t MemAvailable() {
  std::ifstream file("/proc/meminfo");
  std::string key;
  std::uint64_t value = 0;
  std::string unit;
  while (file >> key >> value) {
    std::getline(file, unit);
    if (key == "MemAvailable:") {
      return value * 1024;
    }
  }
  return 0;
}

// The lowest MemAvailable seen while it runs, sampled every 50 ms.
class MemorySampler {
 public:
  MemorySampler() : low_(MemAvailable()), thread_([this] { Loop(); }) {}
  MemorySampler(const MemorySampler&) = delete;
  MemorySampler& operator=(const MemorySampler&) = delete;
  MemorySampler(MemorySampler&&) = delete;
  MemorySampler& operator=(MemorySampler&&) = delete;
  ~MemorySampler() {
    stop_ = true;
    thread_.join();
  }
  std::uint64_t low() const { return low_.load(); }

 private:
  void Loop() {
    while (!stop_) {
      const std::uint64_t now = MemAvailable();
      std::uint64_t seen = low_.load();
      while (now < seen && !low_.compare_exchange_weak(seen, now)) {
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
  }
  std::atomic<std::uint64_t> low_;
  std::atomic<bool> stop_{false};
  std::thread thread_;
};

// When each extent last became resident: written on the scheduler's
// thread, read on the main thread once the program that paged it in is
// gone.
class ResidentTimes final : public sc::PageInObserver {
 public:
  explicit ResidentTimes(std::size_t extents) : at_(extents) {}
  void Staged(catalog::ExtentId extent, sc::PageInEvent event) override {
    if (event == sc::PageInEvent::kResident && extent.index() < at_.size()) {
      at_[extent.index()] = Clock::now();
    }
  }
  Clock::time_point Latest(std::span<const catalog::ExtentId> extents) const {
    Clock::time_point latest{};
    for (const catalog::ExtentId extent : extents) {
      if (extent.index() < at_.size()) {
        latest = std::max(latest, at_[extent.index()]);
      }
    }
    return latest;
  }

 private:
  std::vector<Clock::time_point> at_;
};

struct Options {
  jb::Dsv4Options dsv4;
  jb::Fp16Options fp16;
  std::filesystem::path out;
  std::filesystem::path text;
  std::uint32_t context_tokens = 8192;
  std::uint32_t continue_tokens = 16;
  int cycles = 2;
  bool handoff = true;
  bool copy_lane = true;
  std::string fp16_expect;
  int reload = 0;
  bool overlap = false;
  std::filesystem::path prompts;
  std::filesystem::path expect;
  std::uint32_t generate = 32;
  std::uint32_t bench = 0;
};

// One swap's parts, in seconds.
struct SwapTimes {
  std::string name;
  double evict = 0;    // the request to every outgoing extent evicted (or parked)
  double restore = 0;  // then A's state resident again (0 without state)
  double page_in = 0;  // then the incoming weights resident
  double setup = 0;    // then A's hash-routing check
  double first = 0;    // then the first token
  double total = 0;    // the request to the first token
  double release = 0;  // after the swap: backing no load took, released
  std::uint64_t evicted = 0;
  std::uint64_t loaded = 0;
  std::uint64_t read_bytes = 0;
  std::uint64_t spilled_bytes = 0;
  std::uint64_t handed_off = 0;
  std::uint64_t parked = 0;
  std::uint64_t released_unused = 0;
  double plan_seconds = 0;       // A's planning within its first token
  bool exact = true;             // what it then computed equals the control
  std::string first_path = "-";  // A's first token: eager, captured or replayed (D-090)
  std::uint64_t captured = 0;    // A's graphs captured within the swap's first token
};

std::expected<Options, std::string> Parse(std::span<char*> args) {
  Options o;
  o.fp16.trajectory = "control";
  o.fp16.fusion = true;
  for (std::size_t i = 1; i < args.size(); ++i) {
    const std::string_view a = args[i];
    if (a == "--overlap") {
      o.overlap = true;
      continue;
    }
    if (i + 1 >= args.size()) {
      return Error(std::format("{} needs a value", a));
    }
    const std::string_view v = args[++i];
    const auto number = [&](auto& into) -> bool {
      // NOLINTNEXTLINE(bugprone-suspicious-stringview-data-usage): bounded by its end
      const auto [end, ec] = std::from_chars(v.data(), v.data() + v.size(), into);
      return ec == std::errc() && end == v.data() + v.size();
    };
    bool ok = true;
    if (a == "--dsv4-artifact") {
      o.dsv4.artifact = v;
    } else if (a == "--fp16-artifact") {
      o.fp16.artifact = v;
    } else if (a == "--tokens") {
      o.fp16.tokens = v;
    } else if (a == "--out") {
      o.out = v;
    } else if (a == "--text") {
      o.text = v;
    } else if (a == "--context-tokens") {
      ok = number(o.context_tokens) && o.context_tokens > kShortPrompt;
    } else if (a == "--continue") {
      ok = number(o.continue_tokens) && o.continue_tokens >= 1;
    } else if (a == "--cycles") {
      ok = number(o.cycles) && o.cycles >= 0 && o.cycles <= 8;
    } else if (a == "--handoff") {
      o.handoff = v == "on";
      ok = v == "on" || v == "off";
    } else if (a == "--copy-lane") {
      o.copy_lane = v == "on";
      ok = v == "on" || v == "off";
    } else if (a == "--fp16-expect") {
      o.fp16_expect = v;
    } else if (a == "--context") {
      ok = number(o.dsv4.context);
    } else if (a == "--reload") {
      ok = number(o.reload) && o.reload >= 0 && o.reload <= 8;
    } else if (a == "--prompts") {
      o.prompts = v;
    } else if (a == "--expect") {
      o.expect = v;
    } else if (a == "--generate") {
      ok = number(o.generate) && o.generate >= 1;
    } else if (a == "--graphs") {
      o.dsv4.graphs = v == "on";
      ok = v == "on" || v == "off";
    } else if (a == "--bench") {
      ok = number(o.bench) && o.bench >= 1 && o.bench <= 1024;
    } else {
      return Error(std::format("unknown argument {}", a));
    }
    if (!ok) {
      return Error(std::format("{} does not take {}", a, v));
    }
  }
  if (o.dsv4.artifact.empty() || o.fp16.artifact.empty() || o.fp16.tokens.empty() ||
      o.out.empty() ||
      ((o.cycles > 0 || o.reload > 0 || o.overlap || o.bench > 0) && o.text.empty()) ||
      (o.prompts.empty() != o.expect.empty())) {
    return Error(
        "usage: llmp_swap_runner --dsv4-artifact DIR --fp16-artifact DIR --tokens FILE --out DIR "
        "[--text FILE] [--context-tokens N] [--continue N] [--cycles N] [--handoff on|off] "
        "[--copy-lane on|off] [--fp16-expect SHA256] [--context N] [--reload N] [--overlap] "
        "[--prompts FILE --expect DIR --generate N] [--graphs on|off] [--bench N]");
  }
  if (o.bench >= o.dsv4.context) {
    return Error("--context must hold the bench's steps");
  }
  if (o.cycles > 0 && o.context_tokens + o.continue_tokens + 1 > o.dsv4.context) {
    return Error("--context must hold the context and the continuation");
  }
  o.dsv4.out = o.out / "dsv4";
  o.fp16.out = o.out / "fp16";
  return o;
}

bool SameBits(std::span<const float> a, std::span<const float> b) {
  return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size_bytes()) == 0;
}

class Swapper {
 public:
  explicit Swapper(const Options& options)
      : o_(options),
        times_(std::size_t{1} << 17U),
        node_({.compute_streams = 2,
               .slots = ts::kPagedSlots,
               .inline_lanes = false,
               .coalesce = false,
               .copy_lane = options.copy_lane,
               .slot_bytes = jb::kSlabSlotBytes,
               .observer = &times_}),
        dsv4_(node_, o_.dsv4, kDsv4, kDsv4),
        fp16_(node_, o_.fp16, kFp16, kFp16, nullptr, record_) {}

  Status Run();
  Status TearDown() { return node_.TearDown(entered_models_); }

 private:
  Status Tokenize();
  Status Prompts();
  Status Prefill(std::vector<float>& logits);
  // A's greedy decode of `steps` tokens from `first` at position `from`:
  // each step's logits, and the tokens fed then chosen.
  Status Decode(std::uint32_t from, std::int32_t first, std::uint32_t steps,
                std::vector<std::vector<float>>& logits, std::vector<std::int32_t>& tokens);
  Status SwapToB(SwapTimes& t, bool with_state);
  Status SwapToA(SwapTimes& t, bool with_state, bool continuing);
  // After a swap: waits until no eviction is left (backing no load took,
  // being released) and adds the handoff's counts since `before`.
  Status Settle(SwapTimes& t, const sc::SchedulerStats& before, Clock::time_point swapped);
  Status Reload(int round);
  Status Overlap();
  Status Bench();
  // After a swap back to A: A's places are the pinned ones (D-090).
  void CheckPlaces(const SwapTimes& t);
  Status Write();

  const Options& o_;
  std::string record_;
  ResidentTimes times_;
  MemorySampler memory_;
  std::uint64_t available_before_ = MemAvailable();
  ts::PagedNode node_;
  jb::Dsv4Runner dsv4_;
  jb::Fp16Runner fp16_;
  std::vector<ts::PagedModel*> entered_models_;
  std::unique_ptr<llmp::tokenizer::Tokenizer> tokenizer_;

  std::vector<std::int32_t> context_;  // A's context tokens, BOS first
  std::string text_sha256_;
  std::string context_sha256_;
  std::vector<float> control_prefill_;
  std::vector<std::vector<float>> control_steps_;
  std::vector<std::int32_t> control_tokens_;
  std::string continuation_;
  std::uint32_t a_position_ = 0;  // where A's state ends
  std::int32_t a_next_ = 0;       // A's next token
  double initial_load_ = 0;
  double control_prefill_seconds_ = 0;
  double control_decode_seconds_ = 0;
  std::vector<std::vector<float>> fp16_results_;
  std::vector<SwapTimes> swaps_;
  std::vector<SwapTimes> reloads_;
  std::string overlap_;
  std::string prompts_;
  std::string bench_;
  std::vector<std::string> problems_;
};

std::string PathName(jb::Dsv4Path path) {
  switch (path) {
    case jb::Dsv4Path::kEager:
      return "eager";
    case jb::Dsv4Path::kCaptured:
      return "captured";
    case jb::Dsv4Path::kReplayed:
      return "replayed";
  }
  return "?";
}

void Swapper::CheckPlaces(const SwapTimes& t) {
  if (auto r = dsv4_.CheckPlaces(); !r) {
    problems_.push_back(std::format("{}: {}", t.name, r.error()));
  }
}

Status Swapper::Tokenize() {
  // The tokenizer from the artifact's GGUF metadata (import rule 7).
  std::filesystem::path meta;
  for (const auto& entry : std::filesystem::directory_iterator(o_.dsv4.artifact / "meta")) {
    if (entry.path().filename().string().contains("00001-of")) {
      meta = entry.path();
    }
  }
  std::ifstream gguf(meta, std::ios::binary);
  const std::vector<char> header{std::istreambuf_iterator<char>(gguf),
                                 std::istreambuf_iterator<char>()};
  auto read = llmp::tokenizer::ReadGgufTokenizer(std::as_bytes(std::span(header)));
  if (!read) {
    return Error(std::format("{}: {}", meta.string(), read.error().ToString()));
  }
  auto tokenizer = llmp::tokenizer::Tokenizer::Create(std::move(read->spec));
  if (!tokenizer) {
    return Error(tokenizer.error().ToString());
  }
  std::ifstream file(o_.text, std::ios::binary);
  const std::string text{std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
  text_sha256_ = Sha256(std::as_bytes(std::span(text)));
  std::vector<llmp::tokenizer::TokenId> ids;
  if (auto r = tokenizer->Encode(text, {.add_bos_eos = true}, ids); !r) {
    return Error(r.error().ToString());
  }
  if (tokenizer->bos() && (ids.empty() || ids.front() != *tokenizer->bos())) {
    ids.insert(ids.begin(), *tokenizer->bos());  // BOS first, as the dsv4-native runs fed it
  }
  if (ids.size() < o_.context_tokens) {
    return Error(std::format("{} has {} tokens, fewer than {}", o_.text.string(), ids.size(),
                             o_.context_tokens));
  }
  context_.assign(ids.begin(), ids.begin() + o_.context_tokens);
  context_sha256_ = Sha256(std::as_bytes(std::span(context_)));
  tokenizer_ = std::make_unique<llmp::tokenizer::Tokenizer>(std::move(*tokenizer));
  return {};
}

Status Swapper::Prefill(std::vector<float>& logits) {
  const std::uint32_t rows = o_.dsv4.max_rows;
  for (std::uint32_t at = 0; at < context_.size(); at += rows) {
    const auto n = static_cast<std::uint32_t>(std::min<std::size_t>(rows, context_.size() - at));
    if (auto r = dsv4_.Chunk(at, std::span(context_).subspan(at, n), logits); !r) {
      return Error(std::format("A's prefill at {}: {}", at, r.error()));
    }
  }
  a_position_ = static_cast<std::uint32_t>(context_.size());
  a_next_ = jb::Argmax(logits);
  return {};
}

Status Swapper::Decode(std::uint32_t from, std::int32_t first, std::uint32_t steps,
                       std::vector<std::vector<float>>& logits, std::vector<std::int32_t>& tokens) {
  tokens.push_back(first);
  for (std::uint32_t k = 0; k < steps; ++k) {
    std::vector<float>& row = logits.emplace_back();
    const std::int32_t token = tokens.back();
    if (auto r = dsv4_.Chunk(from + k, std::span(&token, 1), row); !r) {
      return Error(std::format("A's decode at {}: {}", from + k, r.error()));
    }
    tokens.push_back(jb::Argmax(row));
  }
  a_position_ = from + steps;
  a_next_ = tokens.back();
  return {};
}

Status Swapper::Settle(SwapTimes& t, const sc::SchedulerStats& before, Clock::time_point swapped) {
  const auto give_up = Clock::now() + std::chrono::minutes(2);
  for (;;) {
    std::size_t left = 0;
    if (auto r = node_.Call(
            [&]() -> Status {
              left = node_.scheduler().evictions();
              return {};
            },
            "counting evictions");
        !r) {
      return r;
    }
    if (left == 0) {
      break;
    }
    if (Clock::now() > give_up) {
      return Error("backing no load took was not released");
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }
  t.release = Seconds(Clock::now() - swapped);
  auto after = node_.Stats();
  if (!after) {
    return std::unexpected(after.error());
  }
  t.handed_off = after->handed_off - before.handed_off;
  t.parked = after->parked - before.parked;
  t.released_unused = after->released_unused - before.released_unused;
  return {};
}

Status Swapper::SwapToB(SwapTimes& t, bool with_state) {
  auto before = node_.Stats();
  if (!before) {
    return std::unexpected(before.error());
  }
  std::vector<catalog::ExtentId> out;
  if (with_state) {
    out = dsv4_.state();  // first: its write-backs start first
  }
  const std::vector<catalog::ExtentId> weights = dsv4_.weights();
  out.insert(out.end(), weights.begin(), weights.end());
  ts::SwapReport report;
  const auto requested = Clock::now();
  if (auto r = node_.Swap(std::move(out), fp16_.everything(), o_.handoff, report); !r) {
    return r;
  }
  t.evict = Seconds(report.evicted - requested);
  t.page_in = Seconds(report.loaded - report.evicted);
  t.evicted = report.evictions;
  t.loaded = report.loads;
  t.read_bytes = fp16_.weight_read_bytes();
  t.spilled_bytes = with_state ? dsv4_.state().size() * ts::kPagedExtent : 0;
  std::vector<float>& result = fp16_results_.emplace_back();
  if (auto r = fp16_.Evaluate(1, result); !r) {
    return r;
  }
  t.first = Seconds(fp16_.first_chunk_done() - report.loaded);
  t.total = Seconds(fp16_.first_chunk_done() - requested);
  const std::string hash = Sha256(std::as_bytes(std::span(result)));
  t.exact =
      SameBits(result, fp16_results_.front()) && (o_.fp16_expect.empty() || hash == o_.fp16_expect);
  if (!t.exact) {
    problems_.push_back(std::format("{}: B's logits ({}) differ from its first run's or {}", t.name,
                                    hash, o_.fp16_expect));
  }
  return Settle(t, *before, report.loaded);
}

Status Swapper::SwapToA(SwapTimes& t, bool with_state, bool continuing) {
  auto before = node_.Stats();
  if (!before) {
    return std::unexpected(before.error());
  }
  ts::SwapReport report;
  const double planned_before = dsv4_.plan_seconds();
  const auto requested = Clock::now();
  if (auto r = node_.Swap(fp16_.weights(), dsv4_.everything(), o_.handoff, report); !r) {
    return r;
  }
  t.evict = Seconds(report.evicted - requested);
  Clock::time_point restored = report.evicted;
  if (with_state) {
    restored = std::max(restored, times_.Latest(dsv4_.state()));
    t.restore = Seconds(restored - report.evicted);
  }
  t.page_in = Seconds(report.loaded - restored);
  t.evicted = report.evictions;
  t.loaded = report.loads;
  t.read_bytes =
      dsv4_.weight_read_bytes() + (with_state ? dsv4_.state().size() * ts::kPagedExtent : 0);
  if (auto r = dsv4_.CheckHashRouting(); !r) {
    return r;
  }
  const auto first = Clock::now();
  t.setup = Seconds(first - report.loaded);
  const std::uint64_t captured_before = dsv4_.graph_stats().captured;
  std::vector<std::vector<float>> steps;
  std::vector<std::int32_t> tokens;
  if (continuing) {
    // The token after the context, as the control fed it.
    if (auto r = Decode(static_cast<std::uint32_t>(context_.size()), control_tokens_.front(), 1,
                        steps, tokens);
        !r) {
      return r;
    }
  } else {
    // At 0 context: a short prompt from a cleared state.
    if (auto r = dsv4_.Clear(); !r) {
      return r;
    }
    std::vector<float>& row = steps.emplace_back();
    if (auto r = dsv4_.Chunk(0, std::span(context_).first(kShortPrompt), row); !r) {
      return r;
    }
    a_position_ = kShortPrompt;
    a_next_ = jb::Argmax(row);
  }
  const auto done = Clock::now();
  t.first = Seconds(done - first);
  t.total = Seconds(done - requested);
  t.plan_seconds = dsv4_.plan_seconds() - planned_before;
  t.first_path = PathName(dsv4_.last_path());
  t.captured = dsv4_.graph_stats().captured - captured_before;
  if (continuing) {
    // The rest of the continuation: every step equal to the control's.
    std::vector<std::vector<float>> rest;
    std::vector<std::int32_t> more;
    if (auto r = Decode(a_position_, a_next_, o_.continue_tokens - 1, rest, more); !r) {
      return r;
    }
    steps.insert(steps.end(), std::make_move_iterator(rest.begin()),
                 std::make_move_iterator(rest.end()));
    std::size_t differing = 0;
    for (std::size_t k = 0; k < steps.size(); ++k) {
      differing += SameBits(steps[k], control_steps_.at(k)) ? 0 : 1;
    }
    t.exact = differing == 0 && tokens.at(1) == control_tokens_.at(1) &&
              std::equal(more.begin(), more.end(), control_tokens_.begin() + 1);
    if (!t.exact) {
      problems_.push_back(
          std::format("{}: {} of {} continued steps' logits differ from the unswapped control's",
                      t.name, differing, steps.size()));
    }
  }
  if (auto r = Settle(t, *before, report.loaded); !r) {
    return r;
  }
  CheckPlaces(t);
  return {};
}

Status Swapper::Prompts() {
  auto lines = jb::ReadTokenLines(o_.prompts);
  if (!lines) {
    return std::unexpected(lines.error());
  }
  const std::uint32_t vocab = dsv4_.vocab();
  for (const jb::TokenLine& prompt : *lines) {
    std::ifstream file(o_.expect / (prompt.name + ".logits.f32"), std::ios::binary);
    std::vector<float> expected(std::size_t{o_.generate} * vocab);
    file.read(reinterpret_cast<char*>(expected.data()),
              static_cast<std::streamsize>(expected.size() * sizeof(float)));
    if (!file) {
      return Error(std::format("no {} steps of expected logits for {}", o_.generate, prompt.name));
    }
    if (auto r = dsv4_.Clear(); !r) {
      return r;
    }
    std::vector<float> row;
    std::uint64_t differing = 0;
    double most = 0;
    std::string ids;
    auto n_past = static_cast<std::uint32_t>(prompt.ids.size());
    for (std::uint32_t k = 0; k < o_.generate; ++k) {
      if (k == 0) {
        if (auto r = dsv4_.Chunk(0, prompt.ids, row); !r) {
          return Error(std::format("{}: {}", prompt.name, r.error()));
        }
      } else {
        const std::int32_t next = jb::Argmax(row);
        if (auto r = dsv4_.Chunk(n_past++, std::span(&next, 1), row); !r) {
          return Error(std::format("{} step {}: {}", prompt.name, k, r.error()));
        }
      }
      const std::span<const float> want =
          std::span(expected).subspan(std::size_t{k} * vocab, vocab);
      for (std::size_t i = 0; i < vocab; ++i) {
        if (std::bit_cast<std::uint32_t>(row[i]) != std::bit_cast<std::uint32_t>(want[i])) {
          ++differing;
          most = std::max(most, std::fabs(static_cast<double>(row[i]) - want[i]));
        }
      }
      ids += std::format("{}{}", k == 0 ? "" : ",", jb::Argmax(row));
    }
    prompts_ += std::format(R"({}{{"name":"{}","prompt_tokens":{},"steps":{},"differing":{},)"
                            R"("max_abs":{},"argmax":[{}]}})",
                            prompts_.empty() ? "" : ",", prompt.name, prompt.ids.size(),
                            o_.generate, differing, most, ids);
    std::println("{}: {} steps, {} logits differ from the resident run's (max abs {})", prompt.name,
                 o_.generate, differing, most);
    if (differing != 0) {
      problems_.push_back(
          std::format("{}: the paged logits differ from the resident run's", prompt.name));
    }
  }
  return {};
}

Status Swapper::Reload(int round) {
  SwapTimes t;
  t.name = std::format("A reload {} (evict all, reload)", round + 1);
  auto before = node_.Stats();
  if (!before) {
    return std::unexpected(before.error());
  }
  std::vector<catalog::ExtentId> out = dsv4_.state();
  const std::vector<catalog::ExtentId> weights = dsv4_.weights();
  out.insert(out.end(), weights.begin(), weights.end());
  ts::SwapReport report;
  const auto requested = Clock::now();
  if (auto r = node_.Swap(std::move(out), dsv4_.everything(), o_.handoff, report); !r) {
    return r;
  }
  t.evict = Seconds(report.evicted - requested);
  const Clock::time_point restored = std::max(report.evicted, times_.Latest(dsv4_.state()));
  t.restore = Seconds(restored - report.evicted);
  t.page_in = Seconds(report.loaded - restored);
  t.evicted = report.evictions;
  t.loaded = report.loads;
  t.read_bytes = dsv4_.weight_read_bytes() + (dsv4_.state().size() * ts::kPagedExtent);
  t.spilled_bytes = dsv4_.state().size() * ts::kPagedExtent;
  if (auto r = dsv4_.CheckHashRouting(); !r) {
    return r;
  }
  const auto first = Clock::now();
  t.setup = Seconds(first - report.loaded);
  std::vector<float> row;
  const std::int32_t next = a_next_;
  const std::uint64_t captured_before = dsv4_.graph_stats().captured;
  if (auto r = dsv4_.Chunk(a_position_, std::span(&next, 1), row); !r) {
    return r;
  }
  ++a_position_;
  a_next_ = jb::Argmax(row);
  const auto done = Clock::now();
  t.first = Seconds(done - first);
  t.total = Seconds(done - requested);
  t.first_path = PathName(dsv4_.last_path());
  t.captured = dsv4_.graph_stats().captured - captured_before;
  if (auto r = Settle(t, *before, report.loaded); !r) {
    return r;
  }
  CheckPlaces(t);
  reloads_.push_back(t);
  return {};
}

Status Swapper::Bench() {
  const std::uint32_t steps = o_.bench;
  const auto n = static_cast<double>(steps);
  std::vector<std::vector<float>> reference;
  struct Pass {
    double seconds = 0;
    double submit = 0;  // the jobs' host time, in all
  };
  const auto pass = [&](std::string_view name, bool graphs, Pass& out) -> Status {
    dsv4_.set_graphs(graphs);
    if (auto r = dsv4_.Clear(); !r) {
      return r;
    }
    const jb::Dsv4GraphStats before = dsv4_.graph_stats();
    std::vector<std::vector<float>> logits(steps);
    std::int32_t token = context_.front();  // BOS
    const auto start = Clock::now();
    for (std::uint32_t k = 0; k < steps; ++k) {
      if (auto r = dsv4_.Chunk(k, std::span(&token, 1), logits[k]); !r) {
        return Error(std::format("bench {} step {}: {}", name, k, r.error()));
      }
      out.submit += dsv4_.last_submit_seconds();
      token = jb::Argmax(logits[k]);
    }
    out.seconds = Seconds(Clock::now() - start);
    const jb::Dsv4GraphStats& after = dsv4_.graph_stats();
    std::size_t differing = 0;
    if (reference.empty()) {
      reference = std::move(logits);
    } else {
      for (std::uint32_t k = 0; k < steps; ++k) {
        differing += SameBits(logits[k], reference[k]) ? 0 : 1;
      }
    }
    if (differing != 0) {
      problems_.push_back(std::format("bench {}: {} of {} steps' logits differ from the warm-up's",
                                      name, differing, steps));
    }
    bench_ += std::format(
        R"({}{{"pass":"{}","graphs":{},"seconds":{:.6f},"tokens_per_second":{:.3f},)"
        R"("submit_ms_per_token":{:.4f},"eager":{},"captured":{},"replayed":{},)"
        R"("capture_seconds":{:.6f},"instantiate_seconds":{:.6f},"differing_steps":{}}})",
        bench_.empty() ? "" : ",\n  ", name, graphs ? "true" : "false", out.seconds,
        n / out.seconds, out.submit * 1e3 / n, after.eager - before.eager,
        after.captured - before.captured, after.replayed - before.replayed,
        after.capture_seconds - before.capture_seconds,
        after.instantiate_seconds - before.instantiate_seconds, differing);
    std::println(
        "bench {}: {} steps in {:.3f} s ({:.2f} tok/s), jobs' host time {:.3f} ms per token; "
        "eager {}, captured {}, replayed {}; {} steps differ",
        name, steps, out.seconds, n / out.seconds, out.submit * 1e3 / n, after.eager - before.eager,
        after.captured - before.captured, after.replayed - before.replayed, differing);
    return {};
  };
  Pass warm;
  if (auto r = pass("warm-up, eager", false, warm); !r) {
    return r;
  }
  std::array<Pass, 3> eager{};
  for (std::size_t i = 0; i < eager.size(); ++i) {
    if (auto r = pass(std::format("eager {}", i + 1), false, eager.at(i)); !r) {
      return r;
    }
  }
  Pass capturing;
  if (auto r = pass("capturing", true, capturing); !r) {
    return r;
  }
  std::array<Pass, 3> graphs{};
  for (std::size_t i = 0; i < graphs.size(); ++i) {
    if (auto r = pass(std::format("graphs {}", i + 1), true, graphs.at(i)); !r) {
      return r;
    }
  }
  // The step's device time alone: its graph replayed back to back in one
  // job, so that the host's part of a step (the scheduler's round trip, the
  // closure's lease and fence, the inputs) shows as the difference.
  auto back_to_back = dsv4_.TimeReplays(0, context_.front(), steps);
  if (!back_to_back) {
    return Error("bench back-to-back replays: " + back_to_back.error());
  }
  if (auto r = dsv4_.Clear(); !r) {
    return r;
  }
  std::println("bench: {} replays back to back in one job: {:.3f} ms each", steps,
               *back_to_back * 1e3);
  dsv4_.set_graphs(o_.dsv4.graphs);
  const auto mean = [&](const std::array<Pass, 3>& passes) {
    double tps = 0;
    double submit = 0;
    for (const Pass& p : passes) {
      tps += n / p.seconds;
      submit += p.submit * 1e3 / n;
    }
    const auto count = static_cast<double>(passes.size());
    return std::pair(tps / count, submit / count);
  };
  const auto [eager_tps, eager_submit] = mean(eager);
  const auto [graph_tps, graph_submit] = mean(graphs);
  std::println(
      "bench: {} steps, eager {:.2f} tok/s ({:.3f} ms host per token), graphs {:.2f} tok/s "
      "({:.3f} ms host per token): {:.3f}x",
      steps, eager_tps, eager_submit, graph_tps, graph_submit, graph_tps / eager_tps);
  bench_ = std::format(
      R"({{"steps":{},"eager_tokens_per_second":{:.3f},"graph_tokens_per_second":{:.3f},)"
      R"("eager_submit_ms_per_token":{:.4f},"graph_submit_ms_per_token":{:.4f},)"
      R"("back_to_back_ms_per_replay":{:.4f},)"
      "\n \"passes\":[\n  {}]}}",
      steps, eager_tps, graph_tps, eager_submit, graph_submit, *back_to_back * 1e3, bench_);
  a_position_ = steps;
  a_next_ = jb::Argmax(reference.back());
  return {};
}

Status Swapper::Overlap() {
  // B paged in alone, then beside one of A's 512-row prefill chunks.
  std::vector<ts::LoadStats> log;
  std::vector<catalog::ExtentId> weights = fp16_.weights();
  if (auto r = node_.Evict(weights); !r) {
    return r;
  }
  auto start = Clock::now();
  if (auto r = node_.Load(weights, "B alone", log); !r) {
    return r;
  }
  const double alone = Seconds(Clock::now() - start);
  if (auto r = node_.Evict(weights); !r) {
    return r;
  }
  if (auto r = dsv4_.Clear(); !r) {
    return r;
  }
  std::vector<float> row;
  double beside = 0;
  start = Clock::now();
  if (auto r = dsv4_.Chunk(0, std::span(context_).first(o_.dsv4.max_rows), row,
                           [&]() -> Status {
                             const auto loading = Clock::now();
                             auto loaded = node_.Load(weights, "B beside A's chunk", log);
                             beside = Seconds(Clock::now() - loading);
                             return loaded;
                           });
      !r) {
    return r;
  }
  const double chunk = Seconds(Clock::now() - start);
  a_position_ = o_.dsv4.max_rows;
  a_next_ = jb::Argmax(row);
  overlap_ = std::format(
      R"({{"copy_lane":{},"b_bytes":{},"b_alone_seconds":{:.6f},"b_beside_seconds":{:.6f},)"
      R"("a_chunk_seconds":{:.6f}}})",
      o_.copy_lane ? "true" : "false", fp16_.weight_read_bytes(), alone, beside, chunk);
  std::println(
      "overlap (copy lane {}): B alone {:.3f} s, beside A's chunk {:.3f} s (chunk {:.3f} s)",
      o_.copy_lane ? "on" : "off", alone, beside, chunk);
  return {};
}

Status Swapper::Run() {
  if (!o_.text.empty()) {
    if (auto r = Tokenize(); !r) {
      return r;
    }
  }
  if (auto r = node_.Open(); !r) {
    return r;
  }
  entered_models_.push_back(&dsv4_);
  if (auto r = dsv4_.Setup(); !r) {
    return r;
  }
  entered_models_.push_back(&fp16_);
  if (auto r = fp16_.Setup(); !r) {
    return r;
  }
  if (auto r = node_.MapWorkspace(std::max(dsv4_.activations_needed(), fp16_.activations_needed()),
                                  std::max(dsv4_.pool_needed(), fp16_.pool_needed()));
      !r) {
    return r;
  }
  // B: everything resident now (the zone, both models' own memory, the
  // workspace, the staging) and both models' weights: the full swaps below
  // evict by policy, not for want of room.
  const std::uint64_t fixed = node_.catalog().OccupancyOf(node_.domain()).Total().value();
  const std::uint64_t budget =
      fixed + node_.StateCapacity() +
      ((dsv4_.weights().size() + fp16_.weights().size()) * ts::kPagedExtent);
  if (auto r = node_.Start(Bytes(budget)); !r) {
    return r;
  }
  if (auto r = dsv4_.Register(); !r) {
    return r;
  }
  if (auto r = fp16_.Register(); !r) {
    return r;
  }
  if (auto r = dsv4_.Bind(); !r) {
    return r;
  }
  if (auto r = fp16_.Bind(); !r) {
    return r;
  }
  node_.Run();

  std::vector<ts::LoadStats> log;
  auto start = Clock::now();
  if (auto r = node_.Load(dsv4_.weights(), "A's first load", log); !r) {
    return r;
  }
  initial_load_ = Seconds(Clock::now() - start);
  std::println("A loaded: {} bytes in {:.2f} s ({:.2f} GB/s)", dsv4_.weight_read_bytes(),
               initial_load_, static_cast<double>(dsv4_.weight_read_bytes()) / initial_load_ / 1e9);
  if (auto r = dsv4_.CheckHashRouting(); !r) {
    return r;
  }
  if (!o_.prompts.empty()) {
    if (auto r = Prompts(); !r) {
      return r;
    }
  }
  if (o_.bench > 0) {
    if (auto r = Bench(); !r) {
      return r;
    }
  }
  if (o_.cycles > 0) {
    // The control: A's context, then its continuation, never swapped, and
    // launch by launch (the cycles' graphs are compared with it).
    dsv4_.set_graphs(false);
    if (auto r = dsv4_.Clear(); !r) {
      return r;
    }
    start = Clock::now();
    if (auto r = Prefill(control_prefill_); !r) {
      return r;
    }
    control_prefill_seconds_ = Seconds(Clock::now() - start);
    start = Clock::now();
    if (auto r = Decode(a_position_, a_next_, o_.continue_tokens, control_steps_, control_tokens_);
        !r) {
      return r;
    }
    control_decode_seconds_ = Seconds(Clock::now() - start);
    dsv4_.set_graphs(o_.dsv4.graphs);
    if (auto r = tokenizer_->Decode(control_tokens_, {}, continuation_); !r) {
      return Error(r.error().ToString());
    }
    std::println("control: {} context tokens prefilled in {:.2f} s, {} decoded in {:.2f} s: {}",
                 context_.size(), control_prefill_seconds_, o_.continue_tokens,
                 control_decode_seconds_, continuation_);
    for (int cycle = 0; cycle < o_.cycles; ++cycle) {
      const std::string use = cycle == 0 ? "first use" : "prepared";
      if (auto r = dsv4_.Clear(); !r) {
        return r;
      }
      std::vector<float> prefill;
      if (auto r = Prefill(prefill); !r) {
        return r;
      }
      if (!SameBits(prefill, control_prefill_)) {
        problems_.push_back(std::format("cycle {}: A's prefill differs from the control's", cycle));
      }
      SwapTimes ab;
      ab.name = std::format("A→B, {} ({} context tokens)", use, context_.size());
      if (auto r = SwapToB(ab, true); !r) {
        return r;
      }
      swaps_.push_back(ab);
      if (cycle == 0) {
        dsv4_.DropPlans();  // A returns to nothing prepared
      }
      SwapTimes ba;
      ba.name = std::format("B→A, {} ({} context tokens)", use, context_.size());
      if (auto r = SwapToA(ba, true, true); !r) {
        return r;
      }
      swaps_.push_back(ba);
    }
    // At 0 context: A's state is not spilled (it stays resident).
    if (auto r = dsv4_.Clear(); !r) {
      return r;
    }
    SwapTimes ab;
    ab.name = "A→B, prepared (0 context)";
    if (auto r = SwapToB(ab, false); !r) {
      return r;
    }
    swaps_.push_back(ab);
    SwapTimes ba;
    ba.name = "B→A, prepared (0 context)";
    if (auto r = SwapToA(ba, false, false); !r) {
      return r;
    }
    swaps_.push_back(ba);
  }
  if (o_.reload > 0 && a_position_ == 0) {
    // Something in A's state to spill and restore: a short prompt.
    if (auto r = dsv4_.Clear(); !r) {
      return r;
    }
    std::vector<float> row;
    if (auto r = dsv4_.Chunk(0, std::span(context_).first(kShortPrompt), row); !r) {
      return r;
    }
    a_position_ = kShortPrompt;
    a_next_ = jb::Argmax(row);
  }
  for (int round = 0; round < o_.reload; ++round) {
    if (auto r = Reload(round); !r) {
      return r;
    }
  }
  if (o_.overlap) {
    if (auto r = Overlap(); !r) {
      return r;
    }
  }
  if (!fp16_results_.empty()) {
    if (auto r = fp16_.Write(fp16_results_); !r) {
      problems_.push_back("B: " + r.error());
    }
  }
  if (dsv4_.coverage_violations() != 0) {
    problems_.push_back(
        std::format("A: {} bound tensors outside cataloged extents of their class; "
                    "first {}",
                    dsv4_.coverage_violations(), dsv4_.first_violation()));
  }
  return Write();
}

std::string SwapJson(const SwapTimes& t) {
  return std::format(
      R"({{"name":"{}","evict":{:.6f},"restore":{:.6f},"page_in":{:.6f},"setup":{:.6f},)"
      R"("first_token":{:.6f},"total":{:.6f},"release_after":{:.6f},"evicted":{},"loaded":{},)"
      R"("read_bytes":{},"spilled_bytes":{},"handed_off":{},"parked":{},"released_unused":{},)"
      R"("plan_seconds":{:.6f},"exact":{},"first_path":"{}","captured":{}}})",
      t.name, t.evict, t.restore, t.page_in, t.setup, t.first, t.total, t.release, t.evicted,
      t.loaded, t.read_bytes, t.spilled_bytes, t.handed_off, t.parked, t.released_unused,
      t.plan_seconds, t.exact ? "true" : "false", t.first_path, t.captured);
}

void Print(const SwapTimes& t) {
  const double weights_in = t.page_in > 0 ? static_cast<double>(t.read_bytes) / 1e9 : 0;
  std::println(
      "{}: total {:.3f} s = evict {:.3f} + restore {:.3f} + page-in {:.3f} ({:.2f} GB, "
      "{:.2f} GB/s) + setup {:.3f} + first token {:.3f} ({}); handed off {}, released after "
      "{:.3f} s; {}",
      t.name, t.total, t.evict, t.restore, t.page_in, weights_in,
      t.page_in + t.restore > 0 ? weights_in / (t.page_in + t.restore) : 0.0, t.setup, t.first,
      t.first_path, t.handed_off, t.release, t.exact ? "exact" : "DIFFERS");
}

Status Swapper::Write() {
  std::string swaps;
  for (const SwapTimes& t : swaps_) {
    swaps += (swaps.empty() ? "" : ",\n  ") + SwapJson(t);
    Print(t);
  }
  std::string reloads;
  for (const SwapTimes& t : reloads_) {
    reloads += (reloads.empty() ? "" : ",\n  ") + SwapJson(t);
    Print(t);
  }
  std::string tokens;
  for (const std::int32_t token : control_tokens_) {
    tokens += std::format("{}{}", tokens.empty() ? "" : ",", token);
  }
  std::string problems;
  for (const std::string& problem : problems_) {
    problems += std::format(R"({}"{}")", problems.empty() ? "" : ",", problem);
  }
  const std::uint64_t low = memory_.low();
  const jb::Dsv4GraphStats& g = dsv4_.graph_stats();
  const std::string graphs = std::format(
      R"({{"on":{},"eager":{},"captured":{},"replayed":{},"refused":{},"capture_seconds":{:.6f},)"
      R"("instantiate_seconds":{:.6f},"nodes":{},"memory_bytes":{},"graphs_kept":{},)"
      R"("graphs_dropped":{},"first_refusal":"{}"}})",
      o_.dsv4.graphs ? "true" : "false", g.eager, g.captured, g.replayed, g.refused,
      g.capture_seconds, g.instantiate_seconds, g.nodes, g.memory_bytes, dsv4_.graphs(), g.dropped,
      [&] {
        std::string text = g.first_refusal;  // as a JSON string's contents
        std::ranges::replace_if(
            text, [](char c) { return c == '"' || c == '\\' || c < ' '; }, '\'');
        return text;
      }());
  std::println(
      "graphs: {} captured ({} nodes, {:.3f} s capturing, {:.3f} s instantiating and uploading, "
      "{:.1f} MiB by free memory), {} replayed, {} eager, {} refused, {} dropped{}",
      g.captured, g.nodes, g.capture_seconds, g.instantiate_seconds,
      static_cast<double>(g.memory_bytes) / (1U << 20U), g.replayed, g.eager, g.refused, g.dropped,
      g.first_refusal.empty() ? "" : " (" + g.first_refusal + ")");
  std::filesystem::create_directories(o_.out);
  std::ofstream file(o_.out / "swap.json");
  file << std::format(
      "{{\"dsv4_artifact\":\"{}\",\"fp16_artifact\":\"{}\",\"handoff\":{},\"copy_lane\":{},"
      "\"context\":{},\"text_sha256\":\"{}\",\"context_tokens\":{},\"context_sha256\":\"{}\","
      "\"continue\":{},\"control_tokens\":[{}],\"initial_load_seconds\":{:.6f},"
      "\"dsv4_read_bytes\":{},\"dsv4_state_bytes\":{},\"dsv4_extents\":{},\"slab_padding\":{},"
      "\"control_prefill_seconds\":{:.6f},\"control_decode_seconds\":{:.6f},"
      "\"coverage_tensors\":{},\"coverage_violations\":{},\"mem_available_before\":{},"
      "\"mem_available_low\":{},\"problems\":[{}],\n \"prompts\":[{}],\n \"overlap\":{},\n"
      " \"graphs\":{},\n \"bench\":{},\n"
      " \"swaps\":[\n  {}],\n \"reloads\":[\n  {}]}}\n",
      o_.dsv4.artifact.filename().string(), o_.fp16.artifact.filename().string(),
      o_.handoff ? "true" : "false", o_.copy_lane ? "true" : "false", o_.dsv4.context, text_sha256_,
      context_.size(), context_sha256_, o_.continue_tokens, tokens, initial_load_,
      dsv4_.weight_read_bytes(), dsv4_.state_bytes(), dsv4_.weights().size(), dsv4_.slab_padding(),
      control_prefill_seconds_, control_decode_seconds_, dsv4_.coverage_tensors(),
      dsv4_.coverage_violations(), available_before_, low, problems, prompts_,
      overlap_.empty() ? "null" : overlap_, graphs, bench_.empty() ? "null" : bench_, swaps,
      reloads);
  std::println(
      "peak by MemAvailable: {:.2f} GiB; wrote {}",
      static_cast<double>(available_before_ > low ? available_before_ - low : 0) / (1ULL << 30U),
      (o_.out / "swap.json").string());
  if (!problems_.empty()) {
    std::string all;
    for (const std::string& problem : problems_) {
      all += (all.empty() ? "" : "; ") + problem;
    }
    return Error(all);
  }
  return {};
}

}  // namespace

int main(int argc, char** argv) {
  const auto options = Parse(std::span(argv, static_cast<std::size_t>(argc)));
  if (!options) {
    std::println(stderr, "{}", options.error());
    return 2;
  }
  Status ran;
  {
    Swapper swapper(*options);
    ran = swapper.Run();
    if (auto finished = swapper.TearDown(); !finished) {
      if (!ran) std::println(stderr, "FAILED: {}", ran.error());
      std::println(stderr, "retirement failed: {}", finished.error());
      std::abort();
    }
  }
  if (!ran) {
    std::println(stderr, "FAILED: {}", ran.error());
    return 1;
  }
  std::println("DONE {}", options->out.string());
  return 0;
}
