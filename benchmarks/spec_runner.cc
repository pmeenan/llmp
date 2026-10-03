// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// M3's speculation harness (docs/plan.md, "Speculative decoding in the
// core" and the exit's "Speculation correctness";
// docs/experiments/dspark/README.md): DeepSeek V4 Flash with its DSpark
// drafter on one paged node (dsv4_runner.h), greedy and sampled
// speculation, the forced-rejection checks and speed. A harness binary: it
// links the native tokenizer and chat renderer, which production binaries
// may not until D-088 is accepted.
//
//   jitllm_spec_runner --dsv4-artifact DIR --drafter DIR --prompts FILE --out DIR
//                      --check greedy|forced|swap|sampled-plain|sampled-spec|probe|frontier|sizing|
//                              wave|plan-memory|capacity
//                      [--tokens N] [--context N] [--max-rows N] [--graphs on|off] [--draft N]
//                      [--probe-step N]
//                      [--fp16-artifact DIR --fp16-tokens FILE --fp16-expect SHA256]
//                      [--seeds N] [--sampled FILE] [--poll-us N]
//                      [--exact on|off] [--frontier-head on|off] [--margin B]
//                      [--prefill-outa-hca on|off] (two qualified artifacts only)
//
// Two modes (docs/experiments/dsv4-decode/): by default DeepSeek's fast
// plan, a batched verify, and the checks coarse where the kernels differ
// (below); --exact on, the reference mode, D-092's row-invariant verify
// and the checks bit for bit as described. --margin is the fast plan's
// near-tie bound, from the engine's own kernel noise.
//
// Prompts are the fixed set's (docs/experiments/fast-swap/prompts.json),
// rendered by the native DeepSeek V4 renderer and tokenized by the native
// tokenizer from the target artifact's GGUF metadata; the `capital` chat
// prompt's IDs must equal those the llama.cpp reference recorded
// (reference-deepseek-v4-flash-0731-llamacpp.json, beside the prompts).
//
// - greedy: for the decode prompts (`prose`, `code`) and the chat prompts,
//   plain greedy decoding (one-row chunks, decode graphs on) against greedy
//   speculation (the drafter's blocks, row-invariant verifies): every
//   generated token, and the logits it was chosen from, bit for bit; the
//   prefill with the drafter's injection gives the plain prefill's logits
//   bit for bit. Decode speed, acceptance (accepted ÷ drafted) and tokens
//   per verify, per prompt, the speculative run three times. Each
//   generation is a request (D-093): one lease, every chunk under it. Fast
//   plan: the three speculative runs repeat each other bit for bit, and
//   every speculative token is the plain engine's argmax on its own prefix
//   (teacher-forced) or within --margin of it.
// - forced: rejections forced at chosen draft positions (all-reject, then
//   each partial acceptance, and at the rows that complete a CSA or an HCA
//   compressor block): after each verify's rollback the whole state (every
//   state tensor, and the drafter's ring) is hashed, and a control that
//   drafted exactly the accepted tokens must hash the same at every step;
//   the tokens and logits equal plain greedy decoding's. Fast plan: no
//   control (a verify's rows need not equal a shorter verify's); the whole
//   state is read before each verify and after its rollback and must be
//   unchanged but for the accepted rows' writes, and the tokens are held to
//   the near-tie rule.
// - swap: forced rejections, then A swapped out (the FP16 fixture as B)
//   and back with its state spilled and restored, and speculation
//   continued: tokens, logits and the state's hashes equal the unswapped
//   run's; B's logits equal --fp16-expect.
// - sampled-plain, sampled-spec: seeded sampling (temperature 1), plain or
//   speculative (execution/sampling.h VerifyDraft), for 4 prompts, --seeds
//   seeds and the first 8 generated tokens; each prompt's token counts to
//   --out/sampled-<mode>.json. With --sampled FILE (the other mode's), the
//   total-variation distance over each prompt's 16 most frequent tokens
//   plus "other" (the exit's bound: 0.1).
// - probe (fast plan): a diagnostic of one forced-run token (--probe-step);
//   see Probe below and docs/experiments/dsv4-decode/probe.py.
// - wave (--slots N, 2 to 16; docs/experiments/deepseek-batching/): N
//   request slots (engine/dsv4_runner.h), the first N decode and chat
//   prompts one a slot; without --drafter, plain decode waves. First each
//   slot alone (selected alone) prefills and generates --tokens: plain
//   one-row steps, or DSpark draft-and-verify steps of at most the wave's
//   share of rows a slot (8 / N). Then every slot in one request: each
//   cleared and prefilled in turn (its peers' states held), and waves of
//   every slot still running. Plain waves are teacher-forced on the solo
//   tokens: each row's logits against the solo step's (bit-identical rows,
//   the largest difference, argmax agreement, the top-two margin's move).
//   DSpark waves run each slot's solo step's rows: drafts and every verify
//   row of two or more rows bit for bit, the same accepted tokens, and the
//   states' fingerprints at the end equal. The last slot leaves the waves
//   halfway (a cancelled request): its state when it left equals its state
//   at the end. With DSpark, slot 0's first wave verify is discarded: its
//   state equals before the wave but for its draft block's own ring cells
//   (uncommitted positions), and the next wave's re-run of that step gives
//   the solo step bit for bit. Decode seconds of both phases. With a
//   drafter and --wave-mode decode, the waves are injected decode waves
//   instead, against injected one-row steps alone, compared as plain ones.
//   With --wave-mode alternate, each slot's even steps are injected decode
//   steps and its odd ones DSpark steps, alone and in waves alike (serving
//   mixes the two forms, execution/adaptive_wave_mode.h): every row, draft
//   and kept count equals alone's, so drafts after plain steps show the
//   drafter's ring fed by them. No step is discarded.
// - plan-memory (--slots N): the host and driver memory of the runner's
//   plans and graphs, one new shape at a time, against what it counts
//   (PlanMemory below).
// - capacity (--slots N, --state-budget-mib B): the typed capacity refusal
//   and the idle-state release a serving cohort waits on (Capacity below).
//
// The node runs the runtime's own wake (docs/experiments/runtime-wake/);
// --poll-us, a diagnostic, instead has the scheduler and the device lane
// poll that long after their last progress (100,000: the harness's old
// window, longer than a step).
//
// Exit 1 on any failed check; spec.json in --out has every number.

#include <cuda_runtime.h>
#include <malloc.h>

#include <algorithm>
#include <array>
#include <atomic>
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
#include <functional>
#include <iterator>
#include <map>
#include <memory>
#include <optional>
#include <print>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include "base/bytes.h"
#include "base/json.h"
#include "base/sha256.h"
#include "catalog/catalog.h"
#include "chat/chat.h"
#include "dsv4_common.h"
#include "engine/request_cohort.h"
#include "execution/sampling.h"
#include "fp16_runner.h"
#include "ggml.h"
#include "model/dsv4.h"
#include "paged_node.h"
#include "paged_programs.h"
#include "scheduler/scheduler.h"
#include "tokenizer/gguf.h"
#include "tokenizer/tokenizer.h"

namespace {

namespace ts = jitllm::test_support;
namespace jb = jitllm::benchmarks;
namespace md = jitllm::model;
namespace ex = jitllm::execution;
using jitllm::base::Bytes;
using Clock = std::chrono::steady_clock;
using Status = ts::Status;

constexpr int kDsv4 = 0;  // owner and stream
constexpr int kFp16 = 1;
constexpr std::uint32_t kSampledTokens = 8;
// The capacity check's prefill before its spills, and its chunks.
constexpr std::uint32_t kSpillPrefix = 8192;
constexpr std::uint32_t kSpillChunk = 2048;
constexpr std::size_t kHistogramTop = 16;
constexpr double kTvBound = 0.1;

std::unexpected<std::string> Error(std::string what) { return std::unexpected(std::move(what)); }

double Seconds(Clock::duration d) { return std::chrono::duration<double>(d).count(); }

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

bool SameBits(std::span<const float> a, std::span<const float> b) {
  return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size_bytes()) == 0;
}

// Every row's logits, in order, as one SHA-256: a fingerprint to compare
// two builds' generations bit for bit ("" for none kept).
std::string LogitsDigest(std::span<const std::vector<float>> rows) {
  if (rows.empty()) {
    return {};
  }
  jitllm::base::Sha256 hash;
  for (const std::vector<float>& row : rows) {
    hash.Update(std::as_bytes(std::span(row)));
  }
  return jitllm::base::ToHex(hash.Finish());
}

// FNV-1a over 64-bit words (and the tail's bytes): a fingerprint of state
// bytes, for comparing two runs' states without keeping either.
std::uint64_t Fingerprint(std::span<const std::byte> bytes) {
  std::uint64_t h = 0xcbf29ce484222325ULL;
  const std::size_t words = bytes.size() / 8;
  for (std::size_t i = 0; i < words; ++i) {
    std::uint64_t w = 0;
    std::memcpy(&w, bytes.data() + (i * 8), 8);
    h = (h ^ w) * 0x100000001b3ULL;
  }
  for (std::size_t i = words * 8; i < bytes.size(); ++i) {
    h = (h ^ static_cast<std::uint64_t>(bytes[i])) * 0x100000001b3ULL;
  }
  return h;
}

struct Options {
  jb::Dsv4Options dsv4;
  jb::Fp16Options fp16;
  std::filesystem::path out;
  std::filesystem::path prompts;
  std::string check;
  std::uint32_t tokens = 256;
  std::string fp16_expect;
  std::uint32_t seeds = 256;
  std::filesystem::path sampled;
  std::string only;  // greedy: this prompt alone
  // A diagnostic poll window (the paged node's NodeSettings); unset, the
  // runtime's own wake.
  std::optional<std::uint32_t> poll_us;
  // The fast plan's near-tie bound (docs/experiments/dsv4-decode/): a
  // speculative token may differ from the plain engine's argmax only where
  // that argmax leads it by less than this.
  double margin = 0.0;
  // probe: the generated token whose verify is replayed and dumped.
  std::uint32_t probe_step = 0;
  // wave: with a drafter, "verify" (DSpark waves) or "decode" (injected decode waves).
  std::string wave_mode = "verify";
  // capacity: the state's room in the budget (MiB), beside the fixed memory
  // and the weights; 0: room for every slot's whole state twice.
  std::uint64_t state_budget_mib = 0;
};

struct Prompt {
  std::string id;
  std::vector<std::int32_t> ids;
  bool supplied_ids = false;
};

// One speculative step's record.
struct Step {
  std::uint32_t pos = 0;     // the anchor's position
  std::uint32_t rows = 0;    // the verify's rows
  std::uint32_t kept = 0;    // rows kept: the anchor and the accepted drafts
  std::int32_t forced = -1;  // the draft position forced wrong, or -1
  std::vector<std::int32_t> drafts;
  std::vector<std::uint64_t> state;  // fingerprints after the rollback (forced checks)
  // Forcing::untouched: state bytes outside the accepted rows' writes that
  // differ from before the verify (0 if the rollback left nothing stale),
  // and the bytes compared.
  std::uint64_t stale = 0;
  std::uint64_t compared = 0;
};

// A generation: the tokens after the prompt (the first from the prefill)
// and the logits each was chosen from (the first's: the prefill's last row).
struct Generation {
  std::vector<std::int32_t> tokens;
  std::vector<std::vector<float>> logits;
  std::vector<Step> steps;
  std::uint64_t drafted = 0;
  std::uint64_t accepted = 0;
  std::uint64_t verifies = 0;
  double decode_seconds = 0;  // after the first token
  double draft_seconds = 0;   // in Draft
  double verify_seconds = 0;  // in the verifies' Chunk
};

// How a speculative run chooses its drafts.
struct Forcing {
  // Per step: the draft position made wrong (-1: none).
  // NOLINTNEXTLINE(readability-redundant-member-init): designated initializers name it
  std::function<std::int32_t(std::size_t step, std::uint32_t pos, std::uint32_t rows)> wrong = {};
  // A control: per step, exactly these drafts (the other run's accepted).
  const std::vector<Step>* control = nullptr;
  bool fingerprint = false;
  // Stop after this many steps, whatever the tokens left.
  std::size_t max_steps = SIZE_MAX;
  // The fast plan's rollback check: the whole state read before each
  // verify and after its rollback, which may differ only in the accepted
  // rows' writes (Untouched).
  bool untouched = false;
};

class Harness {
 public:
  explicit Harness(Options& options)
      : o_(options),
        node_({.compute_streams = 2,
               .slots = ts::kPagedSlots,
               .inline_lanes = false,
               .coalesce = false,
               .copy_lane = true,
               .slot_bytes = jb::kSlabSlotBytes,
               .observer = nullptr,
               .poll_window = o_.poll_us ? std::optional(std::chrono::microseconds(*o_.poll_us))
                                         : std::nullopt}),
        dsv4_(node_, o_.dsv4, kDsv4, kDsv4),
        fp16_(node_, o_.fp16, kFp16, kFp16, nullptr, record_) {}

  Status Run();
  Status TearDown() { return node_.TearDown(entered_models_); }

 private:
  bool with_fp16() const { return !o_.fp16.artifact.empty(); }
  Status Tokenize();
  Status Prefill(const Prompt& prompt, bool inject, std::vector<float>& last);
  Status Plain(const Prompt& prompt, std::uint32_t count, Generation& out);
  // `body` as one request (M3's lease per request, D-093): the model's
  // closure leased once, every job of `body` a step under it.
  Status InRequest(std::string_view what, const std::function<Status()>& body);
  // Speculation from a prefilled prompt, until `count` tokens.
  Status Speculate(const Prompt& prompt, std::uint32_t count, const std::vector<float>& first,
                   const Forcing& forcing, Generation& out,
                   const ex::SamplingParams* sampling = nullptr, std::uint64_t seed = 0);
  Status Judge(Step& step, const std::vector<float>& logits, std::uint32_t& pos,
               std::int32_t& anchor, const ex::SamplingParams* sampling, std::uint64_t seed,
               std::vector<ex::SamplingCandidate>& scratch, Generation& out);
  Status Fingerprints(std::vector<std::uint64_t>& out);
  // After a verify's Accept: its rollback run, the state read again and
  // compared with `pre_*` (read before the verify) everywhere but the
  // accepted rows' writes; the step's stale and compared bytes.
  Status Untouched(const std::vector<std::byte>& pre_target, const std::vector<std::byte>& pre_ring,
                   Step& step);
  Status Greedy();
  Status Frontier();
  // The forced check's rejections, per step (Forced); counts what each covers.
  static std::function<std::int32_t(std::size_t, std::uint32_t, std::uint32_t)> ForcedWrong(
      std::map<std::string, std::uint64_t>& covered);
  Status Forced();
  Status Probe();
  Status Swap();
  Status Sampled(bool speculative);
  Status Wave();
  Status PlanMemory();
  Status Capacity();
  Status Compare(const Generation& plain, const Generation& spec, std::string_view what);
  // Plain one-token decoding fed `tokens` (teacher-forced): each step's
  // logits, the first the prefill's.
  Status Teacher(const Prompt& prompt, std::span<const std::int32_t> tokens, Generation& out);
  // The fast plan's rule (--margin): every token of `spec` is the plain
  // engine's argmax on `spec`'s own prefix, or within the margin of it (a
  // near-tie). Teacher-forces the plain engine on `spec`'s tokens.
  Status NearTies(const Prompt& prompt, const Generation& plain, const Generation& spec);
  Status SwapOut();
  Status SwapIn();
  Status Write();

  Options& o_;
  std::string record_;
  MemorySampler memory_;
  std::uint64_t available_before_ = MemAvailable();
  ts::PagedNode node_;
  jb::Dsv4Runner dsv4_;
  jb::Fp16Runner fp16_;
  std::vector<ts::PagedModel*> entered_models_;
  std::unique_ptr<jitllm::tokenizer::Tokenizer> tokenizer_;
  std::vector<Prompt> decode_;
  std::vector<Prompt> chat_;
  std::vector<std::vector<float>> fp16_results_;
  std::vector<std::string> results_;  // JSON objects
  std::vector<std::string> problems_;
  double load_seconds_ = 0;
  bool weights_loaded_ = false;
};

// ------------------------------------------------------------------ prompts

std::expected<std::string, std::string> ReadFile(const std::filesystem::path& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    return Error(std::format("cannot read {}", path.string()));
  }
  return std::string{std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}

Status Harness::Tokenize() {
  // The tokenizer from the target artifact's GGUF metadata (import rule 7).
  std::filesystem::path meta;
  for (const auto& entry : std::filesystem::directory_iterator(o_.dsv4.artifact / "meta")) {
    const auto name = entry.path().filename().string();
    if (name.ends_with(".kv.gguf") && (meta.empty() || name.contains("00001-of"))) {
      meta = entry.path();
    }
  }
  auto header = ReadFile(meta);
  if (!header) {
    return std::unexpected(header.error());
  }
  auto read = jitllm::tokenizer::ReadGgufTokenizer(std::as_bytes(std::span(*header)));
  if (!read) {
    return Error(std::format("{}: {}", meta.string(), read.error().ToString()));
  }
  auto tokenizer = jitllm::tokenizer::Tokenizer::Create(std::move(read->spec));
  if (!tokenizer) {
    return Error(tokenizer.error().ToString());
  }
  tokenizer_ = std::make_unique<jitllm::tokenizer::Tokenizer>(std::move(*tokenizer));
  auto text = ReadFile(o_.prompts);
  if (!text) {
    return std::unexpected(text.error());
  }
  auto doc = jitllm::base::json::Parse(*text);
  if (!doc) {
    return Error(std::format("{} is not JSON", o_.prompts.string()));
  }
  const auto render = [&](jitllm::base::json::Value entry) -> std::expected<Prompt, std::string> {
    Prompt p;
    const auto id = entry.find("id");
    const auto messages = entry.find("messages");
    const auto supplied = entry.find("token_ids");
    if (!id || ((!messages || !messages->is_array()) && !supplied)) {
      return Error("a prompt without an id or messages");
    }
    p.id = std::string(id->string());
    if (supplied) {
      p.supplied_ids = true;
      if (!supplied->is_array() || supplied->size() == 0) {
        return Error("a token_ids prompt must be a nonempty array");
      }
      for (std::size_t i = 0; i < supplied->size(); ++i) {
        const auto token = supplied->at(i).int64().value_or(-1);
        if (token < 0 || std::cmp_greater_equal(token, md::Dsv4Flash().vocab)) {
          return Error("a supplied prompt token is outside the vocabulary");
        }
        p.ids.push_back(static_cast<std::int32_t>(token));
      }
      return p;
    }
    jitllm::chat::Conversation c;
    // As llama-server's /apply-template rendered the baselines' prompts: its
    // default turns the template's thinking on (the generation prompt ends
    // in <think>).
    c.enable_thinking = true;
    for (std::size_t i = 0; i < messages->size(); ++i) {
      const auto content = messages->at(i).find("content");
      c.messages.push_back({.role = jitllm::chat::Role::kUser,
                            .content = std::string(content ? content->string() : ""),
                            .reasoning_content = std::nullopt,
                            .tool_calls = {}});
    }
    auto rendered = jitllm::chat::RenderDeepSeekV4(c);
    if (!rendered) {
      return Error(rendered.error().ToString());
    }
    std::vector<jitllm::tokenizer::TokenId> ids;
    if (auto r = tokenizer_->EncodeMarked(rendered->text, rendered->specials, {}, ids); !r) {
      return Error(r.error().ToString());
    }
    p.ids.assign(ids.begin(), ids.end());
    return p;
  };
  for (const auto& [key, into] :
       std::initializer_list<std::pair<std::string_view, std::vector<Prompt>*>>{
           {"decode", &decode_}, {"chat", &chat_}}) {
    const auto list = doc->root().find(key);
    if (!list || !list->is_array()) {
      return Error(std::format("{} has no {} prompts", o_.prompts.string(), key));
    }
    for (std::size_t i = 0; i < list->size(); ++i) {
      auto p = render(list->at(i));
      if (!p) {
        return std::unexpected(p.error());
      }
      if (o_.check == "frontier" && p->ids.size() > o_.dsv4.context - o_.tokens) {
        return Error("the frontier diagnostic's prompt and outputs exceed its context");
      }
      into->push_back(std::move(*p));
    }
  }
  if (o_.check == "frontier" && decode_.empty() && chat_.empty()) {
    return Error("the frontier diagnostic needs at least one prompt");
  }
  const auto supplied_ids = [](const Prompt& p) { return p.supplied_ids; };
  if (std::ranges::all_of(decode_, supplied_ids) && std::ranges::all_of(chat_, supplied_ids)) {
    return {};
  }
  // The renderer and tokenizer against llama.cpp's own IDs for `capital`.
  const std::filesystem::path reference =
      o_.prompts.parent_path() / "reference-deepseek-v4-flash-0731-llamacpp.json";
  if (auto ref = ReadFile(reference); ref) {
    if (auto rdoc = jitllm::base::json::Parse(*ref); rdoc) {
      const auto prompts = rdoc->root().find("prompts");
      for (std::size_t i = 0; prompts && i < prompts->size(); ++i) {
        const auto entry = prompts->at(i);
        const auto id = entry.find("id");
        const auto ids = entry.find("prompt_token_ids");
        if (!id || !ids) {
          continue;
        }
        for (const Prompt& p : chat_) {
          if (p.supplied_ids || p.id != id->string()) {
            continue;
          }
          std::vector<std::int32_t> want;
          want.reserve(ids->size());
          for (std::size_t k = 0; k < ids->size(); ++k) {
            want.push_back(static_cast<std::int32_t>(ids->at(k).int64().value_or(-1)));
          }
          if (want != p.ids) {
            std::string ours;
            std::string theirs;
            for (const std::int32_t t : p.ids) {
              ours += std::format(" {}", t);
            }
            for (const std::int32_t t : want) {
              theirs += std::format(" {}", t);
            }
            problems_.push_back(std::format(
                "prompt {}: the native IDs [{}] differ from llama.cpp's [{}]", p.id, ours, theirs));
          }
        }
      }
    }
  } else {
    problems_.push_back(std::format("no reference IDs at {}", reference.string()));
  }
  return {};
}

// ------------------------------------------------------------------ runs

Status Harness::Prefill(const Prompt& prompt, bool inject, std::vector<float>& last) {
  if (auto r = dsv4_.Clear(); !r) {
    return r;
  }
  const std::uint32_t rows = o_.dsv4.max_rows;
  for (std::uint32_t at = 0; at < prompt.ids.size(); at += rows) {
    const auto n = static_cast<std::uint32_t>(std::min<std::size_t>(rows, prompt.ids.size() - at));
    if (auto r = dsv4_.Chunk(at, std::span(prompt.ids).subspan(at, n), last, {},
                             inject ? jb::Dsv4ChunkKind::kInject : jb::Dsv4ChunkKind::kPlain);
        !r) {
      return Error(std::format("{}'s prefill at {}: {}", prompt.id, at, r.error()));
    }
  }
  return {};
}

Status Harness::InRequest(std::string_view what, const std::function<Status()>& body) {
  return node_.WithRequest(kDsv4, dsv4_.everything(), what, body);
}

Status Harness::Plain(const Prompt& prompt, std::uint32_t count, Generation& out) {
  std::vector<float> last;
  if (auto r = Prefill(prompt, false, last); !r) {
    return r;
  }
  out.tokens = {jb::Argmax(last)};
  out.logits = {last};
  auto pos = static_cast<std::uint32_t>(prompt.ids.size());
  const auto start = Clock::now();
  while (out.tokens.size() < count) {
    std::vector<float> row;
    const std::int32_t token = out.tokens.back();
    if (auto r = dsv4_.Chunk(pos, std::span(&token, 1), row); !r) {
      return r;
    }
    ++pos;
    out.tokens.push_back(jb::Argmax(row));
    out.logits.push_back(std::move(row));
  }
  out.decode_seconds = Seconds(Clock::now() - start);
  return {};
}

Status Harness::Teacher(const Prompt& prompt, std::span<const std::int32_t> tokens,
                        Generation& out) {
  std::vector<float> last;
  if (auto r = Prefill(prompt, false, last); !r) {
    return r;
  }
  out.tokens.assign(tokens.begin(), tokens.end());
  out.logits = {last};
  auto pos = static_cast<std::uint32_t>(prompt.ids.size());
  for (std::size_t i = 1; i < tokens.size(); ++i) {
    std::vector<float> row;
    if (auto r = dsv4_.Chunk(pos, tokens.subspan(i - 1, 1), row); !r) {
      return r;
    }
    ++pos;
    out.logits.push_back(std::move(row));
  }
  return {};
}

Status Harness::NearTies(const Prompt& prompt, const Generation& plain, const Generation& spec) {
  Generation forced;
  if (auto r = InRequest(prompt.id, [&] { return Teacher(prompt, spec.tokens, forced); }); !r) {
    return r;
  }
  std::size_t equal = 0;
  std::size_t violations = 0;
  std::string ties;
  // The verify's own noise: at each step, how far the verify row that gave
  // the token moves the plain engine's top-two margin (its logit difference
  // between the plain row's two best tokens, against the plain row's).
  std::vector<double> moves;
  for (std::size_t i = 0; i < spec.tokens.size(); ++i) {
    const std::span<const float> row = forced.logits[i];
    const std::int32_t best = jb::Argmax(row);
    const std::int32_t got = spec.tokens[i];
    const auto at = [](std::int32_t t) { return static_cast<std::size_t>(t); };
    const bool comparable = i < spec.logits.size() && spec.logits[i].size() == row.size();
    if (comparable && i > 0) {
      std::size_t second = best == 0 ? 1 : 0;
      for (std::size_t j = 0; j < row.size(); ++j) {
        if (std::cmp_not_equal(j, best) && row[j] > row[second]) {
          second = j;
        }
      }
      const std::vector<float>& v = spec.logits[i];
      moves.push_back(std::abs((static_cast<double>(v[at(best)]) - v[second]) -
                               (static_cast<double>(row[at(best)]) - row[second])));
    }
    if (best == got) {
      ++equal;
      continue;
    }
    const double margin = static_cast<double>(row[at(best)]) - static_cast<double>(row[at(got)]);
    // How far the verify's row preferred its token over the plain argmax.
    const double verify_margin =
        comparable ? static_cast<double>(spec.logits[i][at(got)]) - spec.logits[i][at(best)] : 0.0;
    const bool tie = margin < o_.margin;
    violations += tie ? 0 : 1;
    ties += std::format(
        R"({}{{"step":{},"plain":{},"spec":{},"margin":{:.4f},"verify_margin":{:.4f},)"
        R"("near_tie":{}}})",
        ties.empty() ? "" : ",", i, best, got, margin, verify_margin, tie ? "true" : "false");
  }
  std::ranges::sort(moves);
  const auto quantile = [&](double q) {
    return moves.empty()
               ? 0.0
               : moves[std::min(moves.size() - 1,
                                static_cast<std::size_t>(
                                    std::ceil(q * static_cast<double>(moves.size())) - 1.0))];
  };
  const std::string noise =
      std::format(R"("verify_noise":{{"steps":{},"median":{:.4f},"p99":{:.4f},"max":{:.4f}}})",
                  moves.size(), quantile(0.5), quantile(0.99), moves.empty() ? 0.0 : moves.back());
  // Free-running: how long plain greedy decoding and speculation agree.
  std::size_t prefix = 0;
  while (prefix < plain.tokens.size() && prefix < spec.tokens.size() &&
         plain.tokens[prefix] == spec.tokens[prefix]) {
    ++prefix;
  }
  if (violations != 0) {
    problems_.push_back(
        std::format("{}: {} speculative tokens are not the plain engine's argmax "
                    "on their prefix, nor within {} of it",
                    prompt.id, violations, o_.margin));
  }
  std::println(
      "{}: {} of {} speculative tokens the plain engine's argmax on their prefix, {} "
      "near-ties, {} violations (margin {}); free-running prefix {} of {}; verify noise {}",
      prompt.id, equal, spec.tokens.size(), spec.tokens.size() - equal - violations, violations,
      o_.margin, prefix, plain.tokens.size(), noise);
  results_.push_back(std::format(
      R"({{"check":"near_ties","prompt":"{}","tokens":{},"equal":{},"violations":{},)"
      R"("margin":{},"free_running_prefix":{},{},"differing":[{}]}})",
      prompt.id, spec.tokens.size(), equal, violations, o_.margin, prefix, noise, ties));
  return {};
}

Status Harness::Untouched(const std::vector<std::byte>& pre_target,
                          const std::vector<std::byte>& pre_ring, Step& step) {
  if (auto r = dsv4_.Rollback(); !r) {
    return r;
  }
  std::vector<std::byte> target;
  std::vector<std::byte> ring;
  if (auto r = dsv4_.ReadState(target, ring); !r) {
    return r;
  }
  if (target.size() != pre_target.size() || ring.size() != pre_ring.size()) {
    return Error("the state's size changed across a verify");
  }
  // The accepted rows' writes, which may differ; everything else must not.
  std::vector<std::pair<std::uint64_t, std::uint64_t>> allowed_target;
  std::vector<std::pair<std::uint64_t, std::uint64_t>> allowed_ring;
  for (const jb::Dsv4Runner::VerifyWrite& w : dsv4_.last_verify_writes()) {
    if (w.row >= 0 && std::cmp_less(w.row, step.kept)) {
      (w.ring ? allowed_ring : allowed_target).emplace_back(w.offset, w.offset + w.bytes);
    }
  }
  const auto compare = [&](const std::vector<std::byte>& before,
                           const std::vector<std::byte>& after,
                           std::vector<std::pair<std::uint64_t, std::uint64_t>>& allowed) {
    std::ranges::sort(allowed);
    std::uint64_t at = 0;
    const auto span = [&](std::uint64_t from, std::uint64_t to) {
      to = std::min<std::uint64_t>(to, before.size());
      if (from >= to) {
        return;
      }
      step.compared += to - from;
      if (std::memcmp(before.data() + from, after.data() + from, to - from) == 0) {
        return;
      }
      for (std::uint64_t i = from; i < to; ++i) {
        step.stale += before[i] != after[i] ? 1 : 0;
      }
    };
    for (const auto& [from, to] : allowed) {
      span(at, from);
      at = std::max(at, to);
    }
    span(at, before.size());
  };
  compare(pre_target, target, allowed_target);
  compare(pre_ring, ring, allowed_ring);
  return {};
}

Status Harness::Fingerprints(std::vector<std::uint64_t>& out) {
  if (auto r = dsv4_.Rollback(); !r) {
    return r;
  }
  std::vector<std::byte> target;
  std::vector<std::byte> ring;
  if (auto r = dsv4_.ReadState(target, ring); !r) {
    return r;
  }
  out.clear();
  for (const md::Dsv4StateTensor& t : dsv4_.state_layout().tensors) {
    out.push_back(Fingerprint(std::span(target).subspan(t.offset, t.bytes)));
  }
  out.push_back(Fingerprint(ring));
  return {};
}

// A verify's verdict: accept drafts while the target agrees (greedy) or its
// speculative sampling accepts them; the first disagreement, or the row
// after the last draft, gives the next token. Then the rows kept, the
// tokens, their logits, the counts, and the next position and anchor.
Status Harness::Judge(Step& step, const std::vector<float>& logits, std::uint32_t& pos,
                      std::int32_t& anchor, const ex::SamplingParams* sampling, std::uint64_t seed,
                      std::vector<ex::SamplingCandidate>& scratch, Generation& out) {
  const std::uint32_t vocab = dsv4_.vocab();
  const std::uint32_t rows = step.rows;
  const auto row = [&](std::uint32_t i) {
    return std::span<const float>(logits).subspan(std::size_t{i} * vocab, vocab);
  };
  std::uint32_t m = 0;
  std::int32_t next = -1;
  for (; m < rows - 1; ++m) {
    if (sampling == nullptr) {
      const std::int32_t want = jb::Argmax(row(m));
      if (want != step.drafts[m]) {
        next = want;
        break;
      }
    } else {
      auto verdict = ex::VerifyDraft(row(m), step.drafts[m], *sampling,
                                     {.seed = seed, .stream = 0, .position = pos + m + 1}, scratch);
      if (!verdict) {
        return Error("a draft's verdict");
      }
      if (!verdict->accepted) {
        next = verdict->token;
        break;
      }
    }
  }
  if (next < 0) {
    if (sampling == nullptr) {
      next = jb::Argmax(row(m));
    } else {
      auto t = ex::Sample(row(m), *sampling, {.seed = seed, .stream = 0, .position = pos + m + 1},
                          scratch);
      if (!t) {
        return Error("a sample");
      }
      next = *t;
    }
  }
  step.kept = m + 1;
  if (auto r = dsv4_.Accept(step.kept); !r) {
    return r;
  }
  for (std::uint32_t i = 0; i <= m; ++i) {
    out.tokens.push_back(i < m ? step.drafts[i] : next);
    out.logits.emplace_back(row(i).begin(), row(i).end());
  }
  out.drafted += rows - 1;
  out.accepted += m;
  ++out.verifies;
  pos += m + 1;
  anchor = next;
  return {};
}

Status Harness::Speculate(const Prompt& prompt, std::uint32_t count,
                          const std::vector<float>& first, const Forcing& forcing, Generation& out,
                          const ex::SamplingParams* sampling, std::uint64_t seed) {
  std::vector<ex::SamplingCandidate> scratch;
  auto pos = static_cast<std::uint32_t>(prompt.ids.size());
  const std::uint32_t vocab = dsv4_.vocab();
  const auto choose = [&](std::span<const float> row, std::uint32_t at) -> std::int32_t {
    if (sampling == nullptr) {
      return jb::Argmax(row);
    }
    auto t = ex::Sample(row, *sampling, {.seed = seed, .stream = 0, .position = at}, scratch);
    return t.value_or(-1);
  };
  out.tokens = {choose(first, pos)};
  out.logits = {first};
  std::int32_t anchor = out.tokens.back();
  const auto start = Clock::now();
  while (out.tokens.size() < count && out.steps.size() < forcing.max_steps) {
    Step step;
    step.pos = pos;
    const std::size_t index = out.steps.size();
    const auto left = static_cast<std::uint32_t>(count - out.tokens.size());
    // Unforced, the draft and its verify run as one job (DraftVerify).
    if (!forcing.wrong && forcing.control == nullptr) {
      auto rows = std::min<std::uint32_t>(
          {o_.dsv4.draft_rows + 1, o_.dsv4.max_verify, left, o_.dsv4.context - pos});
      while (rows > 1 && !md::Dsv4SameWidths(dsv4_.state_layout(), pos, rows)) {
        --rows;
      }
      step.rows = rows;
      std::vector<float> logits;
      const auto verifying = Clock::now();
      if (auto r = dsv4_.DraftVerify(pos, anchor, rows, step.drafts, logits); !r) {
        return r;
      }
      out.verify_seconds += Seconds(Clock::now() - verifying);
      step.drafts.resize(rows - 1);
      if (auto r = Judge(step, logits, pos, anchor, sampling, seed, scratch, out); !r) {
        return r;
      }
      if (forcing.fingerprint) {
        if (auto r = Fingerprints(step.state); !r) {
          return r;
        }
      }
      out.steps.push_back(std::move(step));
      continue;
    }
    const auto drafting = Clock::now();
    if (auto r = dsv4_.Draft(pos, anchor, step.drafts); !r) {
      return r;
    }
    out.draft_seconds += Seconds(Clock::now() - drafting);
    if (forcing.control != nullptr) {
      if (index >= forcing.control->size()) {
        return Error("the control ran past the run it follows");
      }
      const Step& other = (*forcing.control)[index];
      step.drafts.assign(other.drafts.begin(), other.drafts.begin() + (other.kept - 1));
    }
    // The verify: the anchor and its drafts, within the tokens left, the
    // verify's bound, the context and the steps' mask widths (D-092).
    auto rows = std::min<std::uint32_t>({static_cast<std::uint32_t>(step.drafts.size()) + 1,
                                         o_.dsv4.max_verify, left, o_.dsv4.context - pos});
    while (rows > 1 && !md::Dsv4SameWidths(dsv4_.state_layout(), pos, rows)) {
      --rows;
    }
    step.rows = rows;
    step.drafts.resize(rows - 1);
    if (forcing.wrong) {
      step.forced = forcing.wrong(index, pos, rows);
      if (step.forced >= 0 && std::cmp_less(step.forced, step.drafts.size())) {
        auto& d = step.drafts[static_cast<std::size_t>(step.forced)];
        d = static_cast<std::int32_t>((static_cast<std::uint32_t>(d) + 1) % vocab);
      } else {
        step.forced = -1;
      }
    }
    std::vector<std::int32_t> input = {anchor};
    input.insert(input.end(), step.drafts.begin(), step.drafts.end());
    std::vector<std::byte> pre_target;
    std::vector<std::byte> pre_ring;
    if (forcing.untouched) {
      if (auto r = dsv4_.Rollback(); !r) {
        return r;
      }
      if (auto r = dsv4_.ReadState(pre_target, pre_ring); !r) {
        return r;
      }
    }
    std::vector<float> logits;
    const auto verifying = Clock::now();
    if (auto r = dsv4_.Chunk(pos, input, logits, {}, jb::Dsv4ChunkKind::kVerify); !r) {
      return r;
    }
    out.verify_seconds += Seconds(Clock::now() - verifying);
    if (auto r = Judge(step, logits, pos, anchor, sampling, seed, scratch, out); !r) {
      return r;
    }
    if (forcing.untouched) {
      if (auto r = Untouched(pre_target, pre_ring, step); !r) {
        return r;
      }
    }
    if (forcing.fingerprint) {
      if (auto r = Fingerprints(step.state); !r) {
        return r;
      }
    }
    out.steps.push_back(std::move(step));
  }
  if (auto r = dsv4_.Rollback(); !r) {
    return r;
  }
  out.decode_seconds = Seconds(Clock::now() - start);
  return {};
}

Status Harness::Compare(const Generation& plain, const Generation& spec, std::string_view what) {
  std::size_t tokens = 0;
  std::size_t logits = 0;
  std::optional<std::size_t> first;
  for (std::size_t i = 0; i < std::min(plain.tokens.size(), spec.tokens.size()); ++i) {
    const bool token_differs = plain.tokens[i] != spec.tokens[i];
    const bool logits_differ = !SameBits(plain.logits[i], spec.logits[i]);
    tokens += token_differs ? 1 : 0;
    logits += logits_differ ? 1 : 0;
    if ((token_differs || logits_differ) && !first) {
      first = i;
    }
  }
  if (plain.tokens.size() != spec.tokens.size() || tokens != 0 || logits != 0) {
    problems_.push_back(std::format(
        "{}: {} of {} tokens and {} logits rows differ from plain greedy decoding's (first at {})",
        what, tokens, plain.tokens.size(), logits, first.value_or(0)));
  }
  return {};
}

// ------------------------------------------------------------------ checks

// Production prefill copies only its last logit row in both arms. Compare
// raw target/DSpark state after prefill, then force the same continuation;
// cross-head differences are recorded, while each arm must repeat exactly.
Status Harness::Frontier() {
  if (o_.dsv4.exact) {
    return Error("the frontier control requires the fast plan");
  }
  std::vector<Prompt> prompts = decode_;
  prompts.insert(prompts.end(), chat_.begin(), chat_.end());
  if (!o_.only.empty()) {
    std::erase_if(prompts, [&](const Prompt& p) { return p.id != o_.only; });
  }
  if (prompts.empty()) {
    return Error("the frontier control has no prompt");
  }
  for (const Prompt& prompt : prompts) {
    for (const bool inject : {false, true}) {
      std::array<std::vector<std::byte>, 2> baseline_state;
      std::array<std::vector<float>, 2> own_first;
      std::array<Generation, 2> own;
      std::vector<std::int32_t> forced;
      for (std::uint32_t repeat = 0; repeat < 4; ++repeat) {
        const bool frontier = repeat == 1 || repeat == 2;
        const std::size_t arm = frontier ? 1 : 0;
        o_.dsv4.frontier_head = frontier;
        Generation generation;
        std::vector<float> first;
        double prefill = 0;
        std::array<std::vector<std::byte>, 2> state;
        if (auto r = InRequest("a frontier prefill control",
                               [&]() -> Status {
                                 const auto start = Clock::now();
                                 if (auto p = Prefill(prompt, inject, first); !p) {
                                   return p;
                                 }
                                 prefill = Seconds(Clock::now() - start);
                                 if (auto s = dsv4_.ReadState(state[0], state[1]); !s) {
                                   return s;
                                 }
                                 generation.tokens = {jb::Argmax(first)};
                                 generation.logits = {first};
                                 auto pos = static_cast<std::uint32_t>(prompt.ids.size());
                                 for (std::uint32_t i = 1; i < o_.tokens; ++i) {
                                   const std::int32_t token =
                                       repeat == 0 ? generation.tokens.back() : forced[i - 1];
                                   std::vector<float> row;
                                   if (auto c = dsv4_.Chunk(pos++, std::span(&token, 1), row); !c) {
                                     return c;
                                   }
                                   generation.tokens.push_back(jb::Argmax(row));
                                   generation.logits.push_back(std::move(row));
                                 }
                                 return {};
                               });
            !r) {
          return r;
        }
        if (repeat == 0) {
          baseline_state = std::move(state);
          forced = generation.tokens;
        } else if (state != baseline_state) {
          return Error("frontier prefill target or DSpark state differs from the all-head control");
        }
        if (repeat < 2) {
          own_first[arm] = first;
          own[arm] = generation;
        } else {
          if (!SameBits(own_first[arm], first) || own[arm].tokens != generation.tokens ||
              own[arm].logits.size() != generation.logits.size()) {
            return Error("the frontier control does not repeat its own trajectory exactly");
          }
          for (std::size_t i = 0; i < generation.logits.size(); ++i) {
            if (!SameBits(own[arm].logits[i], generation.logits[i])) {
              return Error("the frontier control does not repeat its own logits exactly");
            }
          }
        }
        for (std::size_t i = 1; i < generation.logits.size(); ++i) {
          if (!SameBits(own[0].logits[i], generation.logits[i])) {
            return Error("the frontier control's forced continuation logits differ");
          }
        }
        const std::string name =
            std::format("{}-{}-{}", prompt.id, inject ? "inject" : "plain", repeat);
        if (auto w = jb::WriteFloats(o_.out / (name + ".head.f32"), first); !w) {
          return w;
        }
        const auto hash = [](std::span<const std::byte> bytes) {
          return jitllm::base::ToHex(jitllm::base::Sha256().Update(bytes).Finish());
        };
        results_.push_back(std::format(
            R"({{"check":"frontier","prompt":"{}","prompt_tokens":{},"inject":{},"frontier":{},"repeat":{},"prefill_s":{:.6f},"prefill_tok_s":{:.3f},"target_sha256":"{}","ring_sha256":"{}","first_sha256":"{}","continuation_rows":{},"logits_sha256":"{}"}})",
            prompt.id, prompt.ids.size(), inject ? "true" : "false", frontier ? "true" : "false",
            repeat, prefill, static_cast<double>(prompt.ids.size()) / prefill,
            hash(baseline_state[0]), hash(baseline_state[1]), LogitsDigest(std::span(&first, 1)),
            generation.logits.size() - 1, LogitsDigest(generation.logits)));
        std::println(
            "{}: frontier {}, prefill {:.6f} s, target/DSpark state and {} continuation rows exact",
            name, frontier, prefill, generation.logits.size() - 1);
      }
    }
  }
  o_.dsv4.frontier_head = false;
  return {};
}

Status Harness::Greedy() {
  // The chained draft-verify looks the drafts' embedding rows up on the
  // device: every token's row must equal the host's lookup bit for bit.
  const auto embedding_start = Clock::now();
  auto embedding = dsv4_.CheckDeviceEmbedding();
  if (!embedding) {
    problems_.push_back(std::format("the device's embedding lookup: {}", embedding.error()));
  } else {
    const double seconds = Seconds(Clock::now() - embedding_start);
    std::println("device embedding lookup: {} tokens' rows equal the host's ({:.1f} s)", *embedding,
                 seconds);
    results_.push_back(std::format(R"({{"check":"device_embedding","tokens":{},"seconds":{:.3f}}})",
                                   *embedding, seconds));
  }
  std::vector<Prompt> prompts = decode_;
  prompts.insert(prompts.end(), chat_.begin(), chat_.end());
  if (!o_.only.empty()) {
    std::erase_if(prompts, [&](const Prompt& p) { return p.id != o_.only; });
  }
  for (const Prompt& prompt : prompts) {
    const bool decode =
        std::ranges::any_of(decode_, [&](const Prompt& p) { return p.id == prompt.id; });
    const std::uint32_t count = decode ? o_.tokens : std::min<std::uint32_t>(o_.tokens, 32);
    // Each generation is a request (D-093): its prefill and every step run
    // under one lease on DeepSeek's closure, as a turn does in the runtime.
    Generation plain;
    if (auto r = InRequest("a plain generation", [&] { return Plain(prompt, count, plain); }); !r) {
      return r;
    }
    const std::size_t repeats = decode ? 3 : 1;
    std::vector<double> rates;
    Generation spec;
    Generation first_run;
    for (std::size_t k = 0; k < repeats; ++k) {
      std::vector<float> first;
      if (k == 1) {
        first_run = std::move(spec);
      }
      spec = {};
      if (auto r = InRequest("a speculative generation",
                             [&]() -> Status {
                               if (auto p = Prefill(prompt, true, first); !p) {
                                 return p;
                               }
                               return Speculate(prompt, count, first, {}, spec);
                             });
          !r) {
        return r;
      }
      if (!SameBits(first, plain.logits.front())) {
        problems_.push_back(std::format(
            "{}: the prefill with the injection differs from the plain prefill", prompt.id));
      }
      // The reference mode's rule: bit for bit (D-092). The fast plan's: the
      // same engine repeats itself bit for bit, and its tokens are the plain
      // engine's up to near-ties (NearTies, below).
      if (o_.dsv4.exact) {
        if (auto r = Compare(plain, spec, prompt.id); !r) {
          return r;
        }
      } else if (k > 0) {
        bool same =
            first_run.tokens == spec.tokens && first_run.logits.size() == spec.logits.size();
        for (std::size_t i = 0; same && i < spec.logits.size(); ++i) {
          same = SameBits(first_run.logits[i], spec.logits[i]);
        }
        if (!same) {
          problems_.push_back(std::format(
              "{}: speculative run {} does not repeat the first bit for bit", prompt.id, k + 1));
        }
      }
      rates.push_back(static_cast<double>(count - 1) / spec.decode_seconds);
    }
    if (!o_.dsv4.exact) {
      if (auto r = NearTies(prompt, plain, spec); !r) {
        return r;
      }
    }
    std::string text;
    if (auto decoded = tokenizer_->Decode(
            std::vector<jitllm::tokenizer::TokenId>(spec.tokens.begin(), spec.tokens.end()), {},
            text);
        !decoded) {
      text = "(not decodable)";
    }
    const double plain_rate = static_cast<double>(count - 1) / plain.decode_seconds;
    const double acceptance =
        spec.drafted > 0 ? static_cast<double>(spec.accepted) / static_cast<double>(spec.drafted)
                         : 0.0;
    std::string rates_json;
    for (const double r : rates) {
      rates_json += std::format("{}{:.3f}", rates_json.empty() ? "" : ",", r);
    }
    const double per_step_ms = 1000.0 / static_cast<double>(spec.verifies);
    std::println(
        "{}: {} prompt tokens, {} generated; plain {:.2f} tok/s, speculative [{}] "
        "tok/s; acceptance {:.3f} ({} of {}), {:.2f} tokens a verify; a step: draft "
        "{:.2f} ms, verify {:.2f} ms, of {:.2f} ms",
        prompt.id, prompt.ids.size(), count, plain_rate, rates_json, acceptance, spec.accepted,
        spec.drafted, static_cast<double>(count - 1) / static_cast<double>(spec.verifies),
        spec.draft_seconds * per_step_ms, spec.verify_seconds * per_step_ms,
        spec.decode_seconds * per_step_ms);
    std::string escaped;
    jitllm::base::json::AppendQuoted(text.substr(0, 160), escaped);
    // Every token, for comparisons with other drivers of the same engine
    // (jitllm-runtime's chat).
    const auto ids = [](const std::vector<std::int32_t>& tokens) {
      std::string out;
      for (const std::int32_t t : tokens) {
        out += std::format("{}{}", out.empty() ? "" : ",", t);
      }
      return out;
    };
    results_.push_back(std::format(
        R"({{"check":"greedy","prompt":"{}","prompt_tokens":{},"generated":{},"plain_tok_s":{:.3f},)"
        R"("spec_tok_s":[{}],"drafted":{},"accepted":{},"acceptance":{:.4f},"verifies":{},)"
        R"("step_ms":{{"draft":{:.3f},"verify":{:.3f},"all":{:.3f}}},)"
        R"("draft_path":{{"eager":{},"captured":{},"replayed":{}}},"text":{},)"
        R"("prompt_ids":[{}],"plain_tokens":[{}],"spec_tokens":[{}],)"
        R"("plain_logits_sha256":"{}","spec_logits_sha256":"{}"}})",
        prompt.id, prompt.ids.size(), count, plain_rate, rates_json, spec.drafted, spec.accepted,
        acceptance, spec.verifies, spec.draft_seconds * per_step_ms,
        spec.verify_seconds * per_step_ms, spec.decode_seconds * per_step_ms,
        dsv4_.draft_stats().eager, dsv4_.draft_stats().captured, dsv4_.draft_stats().replayed,
        escaped, ids(prompt.ids), ids(plain.tokens), ids(spec.tokens), LogitsDigest(plain.logits),
        LogitsDigest(spec.logits)));
  }
  return {};
}

// Forced rejections: all-reject, each partial acceptance, and wherever a
// verify's rows reach the row completing a compressor block (a CSA block
// every 4 positions, an HCA block every 128), the rejection placed on that
// row or before it, so that its compressed row is written and restored.
std::function<std::int32_t(std::size_t, std::uint32_t, std::uint32_t)> Harness::ForcedWrong(
    std::map<std::string, std::uint64_t>& covered) {
  return [&covered](std::size_t step, std::uint32_t pos, std::uint32_t rows) -> std::int32_t {
    if (rows < 2) {
      return -1;
    }
    // An HCA block completes at a rejected row when one ends in the rows.
    for (std::uint32_t i = 1; i < rows; ++i) {
      if ((pos + i + 1) % md::kDsv4HcaRatio == 0) {
        ++covered["hca block rejected"];
        return static_cast<std::int32_t>(i - 1);
      }
    }
    switch (step % 6) {
      case 0:
        ++covered["all rejected"];
        return 0;
      case 1:
        ++covered["one accepted"];
        return 1;
      case 2:
        ++covered["two accepted"];
        return 2;
      case 3:
        return -1;  // as drafted
      default: {
        // A CSA block's row rejected: the first row completing one.
        for (std::uint32_t i = 1; i < rows; ++i) {
          if ((pos + i + 1) % md::kDsv4CsaRatio == 0) {
            ++covered["csa block rejected"];
            return static_cast<std::int32_t>(i - 1);
          }
        }
        return 0;
      }
    }
  };
}

Status Harness::Forced() {
  const Prompt& prompt = chat_.front();
  const std::uint32_t count = o_.tokens;
  Generation plain;
  if (auto r = Plain(prompt, count, plain); !r) {
    return r;
  }
  std::map<std::string, std::uint64_t> covered;
  const auto wrong = ForcedWrong(covered);
  std::vector<float> first;
  if (auto r = Prefill(prompt, true, first); !r) {
    return r;
  }
  Generation spec;
  if (!o_.dsv4.exact) {
    // The fast plan: a verify's rows need not equal a shorter verify's bit
    // for bit, so no control; instead every step's rollback must leave the
    // state exactly as before the verify but for the accepted rows' writes,
    // and the tokens are the plain engine's up to near-ties.
    if (auto r = Speculate(prompt, count, first, {.wrong = wrong, .untouched = true}, spec); !r) {
      return r;
    }
    if (auto r = NearTies(prompt, plain, spec); !r) {
      return r;
    }
    std::uint64_t stale = 0;
    std::uint64_t compared = 0;
    std::size_t rejected = 0;
    std::size_t stale_steps = 0;
    for (const Step& s : spec.steps) {
      stale += s.stale;
      compared += s.compared;
      rejected += s.kept < s.rows ? 1 : 0;
      stale_steps += s.stale != 0 ? 1 : 0;
    }
    if (stale != 0) {
      problems_.push_back(std::format(
          "forced: {} steps' rollbacks left {} state bytes differing from before their verify",
          stale_steps, stale));
    }
    std::string covered_json;
    for (const auto& [what, n] : covered) {
      covered_json += std::format("{}\"{}\":{}", covered_json.empty() ? "" : ",", what, n);
    }
    std::println(
        "forced: {} steps, {} with rejected rows; {} stale bytes after rollbacks in {} compared; "
        "covered {}",
        spec.steps.size(), rejected, stale, compared, covered_json);
    results_.push_back(std::format(
        R"({{"check":"forced","prompt":"{}","generated":{},"steps":{},"rejected_steps":{},)"
        R"("stale_bytes":{},"compared_bytes":{},"covered":{{{}}}}})",
        prompt.id, count, spec.steps.size(), rejected, stale, compared, covered_json));
    return {};
  }
  if (auto r = Speculate(prompt, count, first, {.wrong = wrong, .fingerprint = true}, spec); !r) {
    return r;
  }
  if (auto r = Compare(plain, spec, "forced rejections"); !r) {
    return r;
  }
  // The control: the same steps, each drafting exactly what the forced run
  // accepted.
  if (auto r = Prefill(prompt, true, first); !r) {
    return r;
  }
  Generation control;
  if (auto r =
          Speculate(prompt, count, first, {.control = &spec.steps, .fingerprint = true}, control);
      !r) {
    return r;
  }
  if (auto r = Compare(plain, control, "the forced run's control"); !r) {
    return r;
  }
  std::size_t differing = 0;
  std::size_t rejected = 0;
  std::string first_difference;
  for (std::size_t s = 0; s < spec.steps.size() && s < control.steps.size(); ++s) {
    const Step& a = spec.steps[s];
    const Step& b = control.steps[s];
    rejected += a.kept < a.rows ? 1 : 0;
    if (a.state != b.state || a.pos != b.pos || a.kept != b.kept) {
      if (differing++ == 0) {
        for (std::size_t t = 0; t < a.state.size() && t < b.state.size(); ++t) {
          if (a.state[t] != b.state[t]) {
            first_difference = t < dsv4_.state_layout().tensors.size()
                                   ? std::format("state tensor {} (layer {})", t,
                                                 dsv4_.state_layout().tensors[t].layer)
                                   : std::string("the drafter's ring");
            break;
          }
        }
      }
    }
  }
  if (spec.steps.size() != control.steps.size() || differing != 0) {
    problems_.push_back(
        std::format("forced: {} of {} steps' states differ from the control's (first: {})",
                    differing, spec.steps.size(), first_difference));
  }
  std::string steps_json;
  for (const Step& s : spec.steps) {
    steps_json += std::format("{}[{},{},{},{}]", steps_json.empty() ? "" : ",", s.pos, s.rows,
                              s.kept, s.forced);
  }
  std::string covered_json;
  for (const auto& [what, n] : covered) {
    covered_json += std::format("{}\"{}\":{}", covered_json.empty() ? "" : ",", what, n);
  }
  std::println(
      "forced: {} steps, {} with rejected rows, {} states differing from the control; "
      "covered {}",
      spec.steps.size(), rejected, differing, covered_json);
  results_.push_back(std::format(
      R"({{"check":"forced","prompt":"{}","generated":{},"steps":{},"rejected_steps":{},)"
      R"("state_differs":{},"covered":{{{}}},"pos_rows_kept_forced":[{}]}})",
      prompt.id, count, spec.steps.size(), rejected, differing, covered_json, steps_json));
  return {};
}

// A diagnostic (--check probe --probe-step N; docs/experiments/dsv4-decode/):
// the fast plan's forced run as Forced runs it, then from the state before
// the verify that gave token N (the same run repeated to that step) that
// verify again with every named intermediate kept, plain one-row decoding
// of the same rows, and the reference mode's verify and one-row decoding;
// and both plans teacher-forced from the prompt on the run's tokens. Each
// variant's dump and logits row go to --out/probe/VARIANT/ (index.json
// names each tensor's type, shape and offset in dump.bin).
Status Harness::Probe() {
  if (o_.dsv4.exact || o_.probe_step == 0 || o_.probe_step >= o_.tokens) {
    return Error(
        "the probe starts from the fast plan (no --exact on) at a --probe-step below --tokens");
  }
  const Prompt& prompt = chat_.front();
  const std::uint32_t count = o_.tokens;
  const std::uint32_t n = o_.probe_step;
  Generation plain;
  if (auto r = Plain(prompt, count, plain); !r) {
    return r;
  }
  std::map<std::string, std::uint64_t> covered;
  const auto wrong = ForcedWrong(covered);
  std::vector<float> first;
  if (auto r = Prefill(prompt, true, first); !r) {
    return r;
  }
  Generation spec;
  if (auto r = Speculate(prompt, count, first, {.wrong = wrong, .untouched = true}, spec); !r) {
    return r;
  }
  if (auto r = NearTies(prompt, plain, spec); !r) {
    return r;
  }
  // The step whose verify gave token n, and its row.
  std::size_t base = 1;
  std::size_t k = 0;
  for (; k < spec.steps.size() && n >= base + spec.steps[k].kept; ++k) {
    base += spec.steps[k].kept;
  }
  if (n >= spec.tokens.size() || k == spec.steps.size()) {
    return Error(std::format("no step gave token {}", n));
  }
  const Step& step = spec.steps[k];
  const auto r = static_cast<std::uint32_t>(n - base);
  std::vector<std::int32_t> input = {spec.tokens[base - 1]};
  input.insert(input.end(), step.drafts.begin(), step.drafts.end());
  std::string inputs;
  for (const std::int32_t t : input) {
    inputs += std::format(" {}", t);
  }
  std::println(
      "probe: token {} from step {} (pos {}, rows {}, kept {}, forced {}), row {}; input [{}]", n,
      k, step.pos, step.rows, step.kept, step.forced, r, inputs);
  // The plain engine's argmax on the run's prefix (the fast teacher's, set
  // by its variant, which runs first) and the run's token.
  std::int32_t plain_best = -1;
  const std::int32_t spec_token = spec.tokens[n];
  const std::filesystem::path root = o_.out / "probe";
  // Back to the state before step k: the same run repeated to it, which
  // must repeat the first run's tokens and logits bit for bit.
  const auto restore = [&]() -> Status {
    std::vector<float> f;
    if (auto p = Prefill(prompt, true, f); !p) {
      return p;
    }
    Generation g;
    if (auto s =
            Speculate(prompt, count, f, {.wrong = wrong, .max_steps = k, .untouched = true}, g);
        !s) {
      return s;
    }
    bool same = g.tokens.size() == base;
    for (std::size_t i = 0; same && i < base; ++i) {
      same = g.tokens[i] == spec.tokens[i] && SameBits(g.logits[i], spec.logits[i]);
    }
    if (!same) {
      return Error("the repeated run does not repeat the first to the probed step");
    }
    return {};
  };
  const auto record = [&](std::string_view variant, std::span<const float> row,
                          std::uint32_t rows) -> Status {
    std::vector<jb::Dsv4Runner::Dumped> dumped;
    if (auto d = dsv4_.DumpLast(dumped); !d) {
      return d;
    }
    const std::filesystem::path dir = root / variant;
    std::filesystem::create_directories(dir);
    std::ofstream bin(dir / "dump.bin", std::ios::binary);
    std::string index;
    std::uint64_t at = 0;
    for (const jb::Dsv4Runner::Dumped& d : dumped) {
      bin.write(reinterpret_cast<const char*>(d.bytes.data()),
                static_cast<std::streamsize>(d.bytes.size()));
      index += std::format(R"({}{{"name":"{}","type":"{}","ne":[{},{},{},{}],"at":{},"bytes":{}}})",
                           index.empty() ? "" : ",", d.name, ggml_type_name(d.type), d.ne[0],
                           d.ne[1], d.ne[2], d.ne[3], at, d.bytes.size());
      at += d.bytes.size();
    }
    std::ofstream(dir / "index.json")
        << std::format(R"({{"variant":"{}","rows":{},"row":{},"tensors":[{}]}})", variant, rows,
                       rows > 1 ? r : 0, index)
        << '\n';
    if (auto w = jb::WriteFloats(dir / "logits.f32", row); !w) {
      return w;
    }
    if (plain_best < 0) {
      plain_best = jb::Argmax(row);
    }
    const auto at_token = [&](std::int32_t t) { return row[static_cast<std::size_t>(t)]; };
    std::println(
        "probe {}: argmax {}; logit[{}] {:.4f}, logit[{}] {:.4f}: the plain argmax leads by {:.4f}",
        variant, jb::Argmax(row), plain_best, at_token(plain_best), spec_token,
        at_token(spec_token), at_token(plain_best) - at_token(spec_token));
    results_.push_back(std::format(
        R"({{"check":"probe","variant":"{}","argmax":{},"lead":{:.4f},"tensors":{}}})", variant,
        jb::Argmax(row), at_token(plain_best) - at_token(spec_token), dumped.size()));
    return {};
  };
  // The target's state (before the probed chunk) to --out/probe/state-WHAT.bin,
  // its layout and the run's steps to state.json.
  std::filesystem::create_directories(root);
  {
    std::string tensors;
    for (const md::Dsv4StateTensor& t : dsv4_.state_layout().tensors) {
      tensors += std::format(
          R"({}{{"kind":{},"layer":{},"f16":{},"ne0":{},"ne1":{},"offset":{},"bytes":{}}})",
          tensors.empty() ? "" : ",", static_cast<int>(t.kind), t.layer, t.f16 ? "true" : "false",
          t.ne0, t.ne1, t.offset, t.bytes);
    }
    std::string steps;
    for (const Step& s : spec.steps) {
      steps +=
          std::format("{}[{},{},{},{}]", steps.empty() ? "" : ",", s.pos, s.rows, s.kept, s.forced);
    }
    std::ofstream(root / "state.json")
        << std::format(R"({{"prompt_tokens":{},"probe_step":{},"tensors":[{}],"steps":[{}]}})",
                       prompt.ids.size(), n, tensors, steps)
        << '\n';
  }
  const auto save_state = [&](std::string_view what) -> Status {
    std::vector<std::byte> target;
    std::vector<std::byte> ring;
    if (auto s = dsv4_.ReadState(target, ring); !s) {
      return s;
    }
    std::ofstream(root / std::format("state-{}.bin", what), std::ios::binary)
        .write(reinterpret_cast<const char*>(target.data()),
               static_cast<std::streamsize>(target.size()));
    return {};
  };
  // P and E: each plan's one-row decoding teacher-forced from the prompt.
  for (const bool exact : {false, true}) {
    if (auto e = dsv4_.set_exact(exact); !e) {
      return e;
    }
    std::vector<float> row;
    if (auto p = Prefill(prompt, false, row); !p) {
      return p;
    }
    auto pos = static_cast<std::uint32_t>(prompt.ids.size());
    for (std::size_t i = 1; i <= n; ++i, ++pos) {
      if (i == n) {
        if (auto s = save_state(exact ? "exact-teacher" : "fast-teacher"); !s) {
          return s;
        }
        dsv4_.set_graphs(false);
        dsv4_.set_dump({"*"});
      }
      if (auto c = dsv4_.Chunk(pos, std::span(spec.tokens).subspan(i - 1, 1), row); !c) {
        return c;
      }
    }
    if (auto rec = record(exact ? "exact-teacher" : "fast-teacher", row, 1); !rec) {
      return rec;
    }
    dsv4_.set_dump({});
    if (auto e = dsv4_.set_exact(false); !e) {
      return e;
    }
    dsv4_.set_graphs(o_.dsv4.graphs);
  }
  const std::uint32_t vocab = dsv4_.vocab();
  const auto verify_row = [&](const std::vector<float>& logits) {
    return std::span<const float>(logits).subspan(std::size_t{r} * vocab, vocab);
  };
  // V and X: the fast plan's verify and the reference mode's, from the state.
  for (const bool exact : {false, true}) {
    if (auto s = restore(); !s) {
      return s;
    }
    if (!exact) {
      if (auto s = save_state("spec"); !s) {
        return s;
      }
    }
    dsv4_.set_graphs(false);
    if (auto e = dsv4_.set_exact(exact); !e) {
      return e;
    }
    dsv4_.set_dump({"*"});
    std::vector<float> logits;
    if (auto c = dsv4_.Chunk(step.pos, input, logits, {}, jb::Dsv4ChunkKind::kVerify); !c) {
      return c;
    }
    if (auto rec = record(exact ? "exact-verify" : "fast-verify", verify_row(logits), step.rows);
        !rec) {
      return rec;
    }
    if (!exact && !SameBits(verify_row(logits), spec.logits[n])) {
      problems_.emplace_back("probe: the replayed verify's row differs from the run's");
    }
    if (auto a = dsv4_.Accept(step.rows); !a) {
      return a;
    }
    if (auto rb = dsv4_.Rollback(); !rb) {
      return rb;
    }
    dsv4_.set_dump({});
    if (auto e = dsv4_.set_exact(false); !e) {
      return e;
    }
    dsv4_.set_graphs(o_.dsv4.graphs);
  }
  // D and XD: one-row decoding of the verify's rows from the state.
  for (const bool exact : {false, true}) {
    if (auto s = restore(); !s) {
      return s;
    }
    if (auto e = dsv4_.set_exact(exact); !e) {
      return e;
    }
    std::vector<float> row;
    for (std::uint32_t i = 0; i <= r; ++i) {
      if (i == r) {
        dsv4_.set_graphs(false);
        dsv4_.set_dump({"*"});
      }
      if (auto c = dsv4_.Chunk(step.pos + i, std::span(input).subspan(i, 1), row); !c) {
        return c;
      }
    }
    if (auto rec = record(exact ? "exact-decode" : "fast-decode", row, 1); !rec) {
      return rec;
    }
    dsv4_.set_dump({});
    if (auto e = dsv4_.set_exact(false); !e) {
      return e;
    }
    dsv4_.set_graphs(o_.dsv4.graphs);
  }
  return {};
}

Status Harness::SwapOut() {
  std::vector<jitllm::catalog::ExtentId> out = dsv4_.state();
  const auto weights = dsv4_.weights();
  out.insert(out.end(), weights.begin(), weights.end());
  ts::SwapReport report;
  if (auto r = node_.Swap(std::move(out), fp16_.everything(), true, report); !r) {
    return r;
  }
  std::vector<float>& result = fp16_results_.emplace_back();
  if (auto r = fp16_.Evaluate(1, result); !r) {
    return r;
  }
  jitllm::base::Sha256 hash;
  hash.Update(std::as_bytes(std::span(result)));
  const std::string digest = jitllm::base::ToHex(hash.Finish());
  if (!o_.fp16_expect.empty() && digest != o_.fp16_expect) {
    problems_.push_back(std::format("B's logits {} differ from {}", digest, o_.fp16_expect));
  }
  return {};
}

Status Harness::SwapIn() {
  ts::SwapReport report;
  if (auto r = node_.Swap(fp16_.weights(), dsv4_.everything(), true, report); !r) {
    return r;
  }
  if (auto r = dsv4_.CheckHashRouting(); !r) {
    return r;
  }
  return dsv4_.CheckPlaces();
}

// Rollback composes with swap: forced rejections, A out and back after a
// rejected step, and speculation continued; against the same run unswapped.
Status Harness::Swap() {
  if (!with_fp16()) {
    return Error("the swap check needs the FP16 fixture");
  }
  const Prompt& prompt = chat_.front();
  const std::uint32_t count = o_.tokens;
  const auto wrong = [](std::size_t step, std::uint32_t, std::uint32_t) -> std::int32_t {
    return static_cast<std::int32_t>(step % 3);
  };
  // The unswapped run.
  std::vector<float> first;
  if (auto r = Prefill(prompt, true, first); !r) {
    return r;
  }
  Generation control;
  if (auto r = Speculate(prompt, count, first, {.wrong = wrong, .fingerprint = true}, control);
      !r) {
    return r;
  }
  // The swapped run: the same, with A swapped out after the first half of
  // its steps (the last of which rejected rows) and back.
  if (auto r = Prefill(prompt, true, first); !r) {
    return r;
  }
  std::size_t steps = control.steps.size() / 2;
  while (steps > 1 && control.steps[steps - 1].kept == control.steps[steps - 1].rows) {
    --steps;  // end on a step that rejected rows
  }
  Generation before;
  if (auto r = Speculate(prompt, count, first,
                         {.wrong = wrong, .fingerprint = true, .max_steps = steps}, before);
      !r) {
    return r;
  }
  const bool rejected =
      !before.steps.empty() && before.steps.back().kept < before.steps.back().rows;
  if (auto r = SwapOut(); !r) {
    return r;
  }
  if (auto r = SwapIn(); !r) {
    return r;
  }
  // Continue from where the first half ended: the prompt and the tokens so
  // far are the context; the last token is the anchor.
  Prompt rest = prompt;
  rest.ids.insert(rest.ids.end(), before.tokens.begin(), before.tokens.end() - 1);
  const std::size_t done = before.steps.size();
  Generation after;
  {
    // Speculate continues from the anchor at the context's end; its first
    // token is the first half's last, whose logits it takes as given.
    std::vector<float> anchor_logits = before.logits.back();
    const auto wrong_after = [&](std::size_t step, std::uint32_t pos, std::uint32_t rows) {
      return wrong(step + done, pos, rows);
    };
    const auto remaining = static_cast<std::uint32_t>(count - before.tokens.size() + 1);
    if (auto r = Speculate(rest, remaining, anchor_logits,
                           {.wrong = wrong_after, .fingerprint = true}, after);
        !r) {
      return r;
    }
  }
  Generation swapped = before;
  swapped.tokens.insert(swapped.tokens.end(), after.tokens.begin() + 1, after.tokens.end());
  swapped.logits.insert(swapped.logits.end(), after.logits.begin() + 1, after.logits.end());
  swapped.steps.insert(swapped.steps.end(), after.steps.begin(), after.steps.end());
  if (auto r = Compare(control, swapped, "resumed after a swap"); !r) {
    return r;
  }
  std::size_t differing = 0;
  for (std::size_t s = 0; s < control.steps.size() && s < swapped.steps.size(); ++s) {
    differing += control.steps[s].state != swapped.steps[s].state ? 1 : 0;
  }
  if (control.steps.size() != swapped.steps.size() || differing != 0) {
    problems_.push_back(std::format("swap: {} of {} steps' states differ from the unswapped run's",
                                    differing, control.steps.size()));
  }
  std::println("swap: A out and back after step {} ({}), {} steps compared, {} states differ", done,
               rejected ? "rows rejected" : "all accepted", control.steps.size(), differing);
  results_.push_back(
      std::format(R"({{"check":"swap","prompt":"{}","generated":{},"swapped_after_step":{},)"
                  R"("last_step_rejected":{},"steps":{},"state_differs":{},"draft_replayed":{}}})",
                  prompt.id, count, done, rejected ? "true" : "false", control.steps.size(),
                  differing, dsv4_.draft_stats().replayed));
  if (!rejected) {
    problems_.emplace_back("swap: the step before the swap rejected no rows");
  }
  return {};
}

// Seeded sampling, plain or speculative: each prompt's token counts over
// --seeds seeds and the first 8 generated tokens.
Status Harness::Sampled(bool speculative) {
  const ex::SamplingParams params{.temperature = 1.0F, .top_k = 0, .top_p = 1.0F, .min_p = 0.0F};
  std::vector<Prompt> prompts(
      chat_.begin(),
      chat_.begin() + static_cast<std::ptrdiff_t>(std::min<std::size_t>(4, chat_.size())));
  std::string json;
  std::vector<std::map<std::int32_t, std::uint64_t>> counts(prompts.size());
  const auto start = Clock::now();
  std::vector<ex::SamplingCandidate> scratch;
  for (std::size_t p = 0; p < prompts.size(); ++p) {
    const Prompt& prompt = prompts[p];
    for (std::uint64_t seed = 0; seed < o_.seeds; ++seed) {
      std::vector<float> first;
      if (auto r = Prefill(prompt, speculative, first); !r) {
        return r;
      }
      Generation g;
      if (speculative) {
        if (auto r = Speculate(prompt, kSampledTokens, first, {}, g, &params, seed); !r) {
          return r;
        }
      } else {
        auto pos = static_cast<std::uint32_t>(prompt.ids.size());
        auto token =
            ex::Sample(first, params, {.seed = seed, .stream = 0, .position = pos}, scratch);
        g.tokens = {token.value_or(-1)};
        while (g.tokens.size() < kSampledTokens) {
          std::vector<float> row;
          const std::int32_t input = g.tokens.back();
          if (auto r = dsv4_.Chunk(pos, std::span(&input, 1), row); !r) {
            return r;
          }
          ++pos;
          token = ex::Sample(row, params, {.seed = seed, .stream = 0, .position = pos}, scratch);
          g.tokens.push_back(token.value_or(-1));
        }
      }
      for (const std::int32_t t : g.tokens) {
        ++counts[p][t];
      }
    }
    std::string entries;
    for (const auto& [token, n] : counts[p]) {
      entries += std::format("{}[{},{}]", entries.empty() ? "" : ",", token, n);
    }
    json += std::format(R"({}{{"prompt":"{}","counts":[{}]}})", json.empty() ? "" : ",", prompt.id,
                        entries);
    std::println("{} {}: {} seeds, {} distinct tokens, {:.1f} s so far",
                 speculative ? "speculative" : "plain", prompt.id, o_.seeds, counts[p].size(),
                 Seconds(Clock::now() - start));
  }
  const double seconds = Seconds(Clock::now() - start);
  const std::string mode = speculative ? "spec" : "plain";
  std::ofstream(o_.out / std::format("sampled-{}.json", mode))
      << std::format(R"({{"mode":"{}","seeds":{},"tokens":{},"seconds":{:.1f},"prompts":[{}]}})",
                     mode, o_.seeds, kSampledTokens, seconds, json)
      << '\n';
  results_.push_back(std::format(R"({{"check":"sampled-{}","seeds":{},"seconds":{:.1f}}})", mode,
                                 o_.seeds, seconds));
  if (o_.sampled.empty()) {
    return {};
  }
  // Against the other mode's counts: each prompt's histogram over its 16
  // most frequent tokens (in this mode's and the other's pooled counts)
  // plus "other".
  auto other_text = ReadFile(o_.sampled);
  if (!other_text) {
    return std::unexpected(other_text.error());
  }
  auto other = jitllm::base::json::Parse(*other_text);
  if (!other) {
    return Error(std::format("{} is not JSON", o_.sampled.string()));
  }
  const auto list = other->root().find("prompts");
  std::string tv_json;
  for (std::size_t p = 0; list && p < list->size() && p < prompts.size(); ++p) {
    std::map<std::int32_t, std::uint64_t> theirs;
    const auto entries = list->at(p).find("counts");
    for (std::size_t k = 0; entries && k < entries->size(); ++k) {
      theirs[static_cast<std::int32_t>(entries->at(k).at(0).int64().value_or(-1))] =
          static_cast<std::uint64_t>(entries->at(k).at(1).int64().value_or(0));
    }
    std::map<std::int32_t, std::uint64_t> pooled = counts[p];
    std::uint64_t ours_total = 0;
    std::uint64_t theirs_total = 0;
    for (const auto& [t, n] : counts[p]) {
      ours_total += n;
    }
    for (const auto& [t, n] : theirs) {
      pooled[t] += n;
      theirs_total += n;
    }
    std::vector<std::pair<std::uint64_t, std::int32_t>> order;
    order.reserve(pooled.size());
    for (const auto& [t, n] : pooled) {
      order.emplace_back(n, t);
    }
    std::ranges::sort(order, [](const auto& a, const auto& b) {
      return a.first > b.first || (a.first == b.first && a.second < b.second);
    });
    double tv = 0;
    double other_ours = 1;
    double other_theirs = 1;
    for (std::size_t k = 0; k < std::min(kHistogramTop, order.size()); ++k) {
      const std::int32_t t = order[k].second;
      const double a = static_cast<double>(counts[p].contains(t) ? counts[p][t] : 0) /
                       static_cast<double>(ours_total);
      const double b = static_cast<double>(theirs.contains(t) ? theirs[t] : 0) /
                       static_cast<double>(theirs_total);
      tv += std::abs(a - b);
      other_ours -= a;
      other_theirs -= b;
    }
    tv = 0.5 * (tv + std::abs(other_ours - other_theirs));
    tv_json += std::format(R"({}{{"prompt":"{}","tv":{:.4f}}})", tv_json.empty() ? "" : ",",
                           prompts[p].id, tv);
    std::println("{}: total variation {:.4f} (bound {})", prompts[p].id, tv, kTvBound);
    if (tv > kTvBound) {
      problems_.push_back(
          std::format("{}: total variation {:.4f} over {}", prompts[p].id, tv, kTvBound));
    }
  }
  results_.push_back(std::format(R"({{"check":"sampled-tv","against":"{}","prompts":[{}]}})",
                                 o_.sampled.string(), tv_json));
  return {};
}

// ------------------------------------------------------------------ run

// --check wave (the header): request slots, alone and in waves.
Status Harness::Wave() {
  using Slot = jb::Dsv4Runner::Slot;
  const std::uint32_t slots = o_.dsv4.wave_slots;
  // With a drafter, DSpark waves, or (--wave-mode decode) injected decode
  // waves, or (--wave-mode alternate) the two by turns: a slot's even steps
  // plain, its odd ones DSpark, alone as in waves, so a draft after plain
  // steps shows the drafter's ring fed by them.
  const bool alternate = dsv4_.speculative() && o_.wave_mode == "alternate";
  const bool spec = dsv4_.speculative() && o_.wave_mode != "decode";
  const auto step_spec = [&](std::size_t step) { return spec && (!alternate || step % 2 == 1); };
  const jb::Dsv4ChunkKind kind =
      dsv4_.speculative() ? jb::Dsv4ChunkKind::kInject : jb::Dsv4ChunkKind::kPlain;
  const std::uint32_t vocab = dsv4_.vocab();
  const std::uint32_t tokens = o_.tokens;
  std::vector<Prompt> prompts = decode_;
  prompts.insert(prompts.end(), chat_.begin(), chat_.end());
  if (prompts.size() < slots) {
    return Error(std::format("the wave check needs {} prompts", slots));
  }
  prompts.resize(slots);
  std::vector<Slot*> handles;
  for (std::uint32_t i = 0; i < slots; ++i) {
    auto slot = dsv4_.request_slot(i);
    if (!slot) {
      return std::unexpected(slot.error());
    }
    handles.push_back(*slot);
  }
  // Each slot's share of a wave's rows; a solo step runs at most that.
  const std::uint32_t share = std::min<std::uint32_t>(
      o_.dsv4.max_verify, static_cast<std::uint32_t>(jb::Dsv4Runner::kWaveRows) / slots);
  const auto rows_at = [&](std::uint32_t pos, std::uint32_t left) {
    auto rows =
        std::min<std::uint32_t>({o_.dsv4.draft_rows + 1, share, left, o_.dsv4.context - pos});
    while (rows > 1 && !md::Dsv4SameWidths(dsv4_.state_layout(), pos, rows)) {
      --rows;
    }
    return rows;
  };
  const auto greedy = [&](std::span<const float> verified, std::span<const std::int32_t> drafts,
                          std::uint32_t rows, std::int32_t& next) {
    std::uint32_t m = 0;
    for (; m < rows - 1; ++m) {
      const std::int32_t want = jb::Argmax(verified.subspan(std::size_t{m} * vocab, vocab));
      if (want != drafts[m]) {
        next = want;
        return m;
      }
    }
    next = jb::Argmax(verified.subspan(std::size_t{m} * vocab, vocab));
    return m;
  };
  const auto read = [&](Slot& slot, std::vector<std::byte>& target, std::vector<std::byte>& ring) {
    if (auto r = slot.Rollback(); !r) {
      return r;
    }
    return slot.ReadState(target, ring);
  };
  const auto fingerprint = [&](Slot& slot) -> std::expected<std::uint64_t, std::string> {
    std::vector<std::byte> target;
    std::vector<std::byte> ring;
    if (auto r = read(slot, target, ring); !r) {
      return std::unexpected(r.error());
    }
    return Fingerprint(target) ^ (Fingerprint(ring) * 31);
  };
  const auto prefill = [&](Slot& slot, const Prompt& prompt, std::vector<float>& last) -> Status {
    if (auto r = slot.Clear(); !r) {
      return r;
    }
    const std::uint32_t rows = o_.dsv4.max_rows;
    for (std::uint32_t at = 0; at < prompt.ids.size(); at += rows) {
      const auto n =
          static_cast<std::uint32_t>(std::min<std::size_t>(rows, prompt.ids.size() - at));
      if (auto r = slot.Chunk(at, std::span(prompt.ids).subspan(at, n), last, kind); !r) {
        return Error(std::format("{}'s prefill at {}: {}", prompt.id, at, r.error()));
      }
    }
    return {};
  };
  // What a slot's run records: its tokens, the prefill's row, and per step
  // the logits (plain: the one row; DSpark: every verify row), the rows,
  // the drafts and the rows kept.
  struct Run {
    std::vector<std::int32_t> tokens;
    std::vector<float> first;
    std::vector<std::vector<float>> logits;
    std::vector<std::uint32_t> rows;
    std::vector<std::vector<std::int32_t>> drafts;
    std::vector<std::uint32_t> kept;
    std::vector<std::size_t> next;  // each step's next anchor, in tokens
    std::uint64_t fingerprint = 0;
  };
  std::vector<Run> solo(slots);
  double solo_seconds = 0;
  std::uint64_t solo_tokens = 0;
  for (std::uint32_t i = 0; i < slots; ++i) {
    const std::array<Slot*, 1> alone = {handles[i]};
    if (auto r = dsv4_.SelectSlots(alone); !r) {
      return r;
    }
    Run& run = solo[i];
    auto ran = node_.WithRequest(kDsv4, dsv4_.execution_closure(), "a solo slot", [&]() -> Status {
      Slot& slot = *handles[i];
      if (auto r = prefill(slot, prompts[i], run.first); !r) {
        return r;
      }
      auto pos = static_cast<std::uint32_t>(prompts[i].ids.size());
      std::int32_t anchor = jb::Argmax(run.first);
      run.tokens = {anchor};
      const auto start = Clock::now();
      while (run.tokens.size() < tokens) {
        const auto left = static_cast<std::uint32_t>(tokens - run.tokens.size());
        if (!step_spec(run.rows.size())) {
          std::vector<float> row;
          if (auto r = slot.Chunk(pos, std::span(&anchor, 1), row, kind); !r) {
            return r;
          }
          anchor = jb::Argmax(row);
          run.tokens.push_back(anchor);
          run.logits.push_back(std::move(row));
          run.rows.push_back(1);
          run.drafts.emplace_back();
          run.kept.push_back(1);
          run.next.push_back(run.tokens.size() - 1);
          ++pos;
          continue;
        }
        const std::uint32_t rows = rows_at(pos, left);
        std::vector<std::int32_t> drafts;
        std::vector<float> verified;
        if (auto r = slot.DraftVerify(pos, anchor, rows, drafts, verified); !r) {
          return r;
        }
        std::int32_t next = -1;
        const std::uint32_t m = greedy(verified, drafts, rows, next);
        if (auto r = slot.Accept(m + 1); !r) {
          return r;
        }
        run.tokens.insert(run.tokens.end(), drafts.begin(), drafts.begin() + m);
        run.tokens.push_back(next);
        run.logits.push_back(std::move(verified));
        run.rows.push_back(rows);
        run.drafts.push_back(std::move(drafts));
        run.kept.push_back(m + 1);
        run.next.push_back(run.tokens.size() - 1);
        pos += m + 1;
        anchor = next;
      }
      solo_seconds += Seconds(Clock::now() - start);
      solo_tokens += run.tokens.size() - 1;
      auto print = fingerprint(slot);
      if (!print) {
        return std::unexpected(print.error());
      }
      run.fingerprint = *print;
      return {};
    });
    if (!ran) {
      return ran;
    }
  }

  // Every slot in one request, cleared and prefilled in turn, then waves.
  if (auto r = dsv4_.SelectSlots(handles); !r) {
    return r;
  }
  struct Cursor {
    std::size_t step = 0;  // the solo step this slot is at
    std::uint32_t pos = 0;
    std::int32_t anchor = 0;
    bool done = false;
    bool left = false;
    std::vector<std::byte> left_target;
    std::vector<std::byte> left_ring;
  };
  std::vector<Cursor> at(slots);
  std::uint64_t rows_compared = 0;
  std::uint64_t rows_identical = 0;
  std::uint64_t argmax_agree = 0;
  double largest = 0;
  std::vector<double> moves;
  std::uint64_t exact_mismatches = 0;
  std::uint64_t waves = 0;
  std::uint64_t wave_tokens = 0;
  std::map<std::size_t, std::uint64_t> widths;
  // Each width's joined waves' time (the wave form's per-width cost:
  // a DSpark wave's time over a plain one's, model_settings.h
  // kDsv4WaveCosts), and each call's: their median calibrates the cost (the
  // first waves of a width plan and capture, which the mean includes).
  std::map<std::size_t, double> width_seconds;
  std::map<std::size_t, std::vector<double>> width_times;
  double wave_seconds = 0;
  std::optional<std::uint64_t> discard_stale;  // none: nothing discarded
  bool discard_rerun_exact = false;
  bool left_unchanged = false;
  std::vector<std::uint64_t> fingerprints(slots, 0);
  auto ran = node_.WithRequest(kDsv4, dsv4_.execution_closure(), "a wave cohort", [&]() -> Status {
    for (std::uint32_t i = 0; i < slots; ++i) {
      std::vector<float> last;
      if (auto r = prefill(*handles[i], prompts[i], last); !r) {
        return r;
      }
      if (!SameBits(last, solo[i].first)) {
        problems_.push_back(
            std::format("slot {}'s prefill beside its peers differs from alone", i));
      }
      at[i].pos = static_cast<std::uint32_t>(prompts[i].ids.size());
      at[i].anchor = solo[i].tokens.front();
      at[i].done = solo[i].rows.empty();
    }
    const std::size_t half = solo[slots - 1].rows.size() / 2;
    bool discarded = false;
    const auto start = Clock::now();
    while (true) {
      // The last slot leaves halfway: its state then, kept to the end.
      Cursor& last = at[slots - 1];
      if (!last.left && !last.done && last.step >= half && half > 0) {
        last.left = true;
        if (auto r = read(*handles[slots - 1], last.left_target, last.left_ring); !r) {
          return r;
        }
      }
      std::vector<std::uint32_t> active;
      for (std::uint32_t i = 0; i < slots; ++i) {
        if (!at[i].done && !at[i].left) {
          active.push_back(i);
        }
      }
      if (active.empty()) {
        break;
      }
      // Every active slot is at the same step (alternate takes no discard).
      const bool wave_spec = step_spec(at[active.front()].step);
      for (const std::uint32_t i : active) {
        if (step_spec(at[i].step) != wave_spec) {
          return Error("an alternating wave's slots are at different forms");
        }
      }
      std::vector<jb::Dsv4Runner::WaveWork> work;
      std::vector<std::vector<float>> logits(active.size());
      std::vector<std::vector<std::int32_t>> drafts(active.size());
      for (std::size_t k = 0; k < active.size(); ++k) {
        const std::uint32_t i = active[k];
        work.push_back({.slot = handles[i],
                        .pos = at[i].pos,
                        .anchor = at[i].anchor,
                        .rows = solo[i].rows[at[i].step],
                        .drafts = wave_spec ? &drafts[k] : nullptr,
                        .logits = &logits[k]});
      }
      // Slot 0's first wave verify is discarded (below), to be re-run.
      std::vector<std::byte> pre_target;
      std::vector<std::byte> pre_ring;
      const bool discard = wave_spec && !alternate && !discarded && active.front() == 0;
      if (discard) {
        if (auto r = read(*handles[0], pre_target, pre_ring); !r) {
          return r;
        }
      }
      // As the serving policy: a verify of one row (a step ending at a
      // mask width or the last token) runs alone, since a wave of wider
      // verifies takes the multi-row launch; so does a lone slot.
      std::vector<jb::Dsv4Runner::WaveWork> joined;
      std::vector<std::size_t> joined_at;
      std::vector<std::size_t> alone;
      for (std::size_t k = 0; k < work.size(); ++k) {
        if (wave_spec && work[k].rows == 1) {
          alone.push_back(k);
        } else {
          joined.push_back(work[k]);
          joined_at.push_back(k);
        }
      }
      if (joined.size() == 1) {
        alone.push_back(joined_at.front());
        joined.clear();
      }
      for (const std::size_t k : alone) {
        const std::uint32_t i = active[k];
        if (!wave_spec) {
          if (auto r = handles[i]->Chunk(at[i].pos, std::span(&at[i].anchor, 1), logits[k], kind);
              !r) {
            return r;
          }
        } else if (auto r = handles[i]->DraftVerify(at[i].pos, at[i].anchor, work[k].rows,
                                                    drafts[k], logits[k]);
                   !r) {
          return r;
        }
      }
      if (!joined.empty()) {
        const auto wave = Clock::now();
        if (auto r = wave_spec ? dsv4_.DraftVerifyWave(joined) : dsv4_.DecodeWave(joined); !r) {
          return r;
        }
        const double took = Seconds(Clock::now() - wave);
        wave_seconds += took;
        width_seconds[joined.size()] += took;
        ++waves;
        ++widths[joined.size()];
        width_times[joined.size()].push_back(took);
      }
      for (std::size_t k = 0; k < active.size(); ++k) {
        const std::uint32_t i = active[k];
        Cursor& c = at[i];
        const std::size_t step = c.step;
        const std::vector<float>& want = solo[i].logits[step];
        const bool same = SameBits(logits[k], want);
        if (!wave_spec) {
          ++rows_compared;
          rows_identical += same ? 1 : 0;
          // Alternating, a plain row follows DSpark waves on the same
          // slots: it must equal the solo run's, or the drafter's ring
          // (or the state) diverged across forms.
          if (alternate && !same) {
            ++exact_mismatches;
            problems_.push_back(
                std::format("slot {} step {} (plain, wave of {}): the row differs from alone", i,
                            step, active.size()));
          }
          const std::int32_t mine = jb::Argmax(logits[k]);
          const std::int32_t theirs = jb::Argmax(want);
          argmax_agree += mine == theirs ? 1 : 0;
          double diff = 0;
          for (std::size_t v = 0; v < want.size(); ++v) {
            diff = std::max(diff, static_cast<double>(std::fabs(logits[k][v] - want[v])));
          }
          largest = std::max(largest, diff);
          // The solo row's top-two margin against the wave row's on the
          // same two tokens.
          const auto best = static_cast<std::size_t>(theirs);
          std::size_t second = best == 0 ? 1 : 0;
          for (std::size_t v = 0; v < vocab; ++v) {
            if (v != best && want[v] > want[second]) {
              second = v;
            }
          }
          moves.push_back(
              std::fabs((want[best] - want[second]) - (logits[k][best] - logits[k][second])));
          // Teacher-forced on the solo tokens.
          c.anchor = solo[i].tokens[solo[i].next[step]];
          ++c.pos;
          ++c.step;
          wave_tokens += 1;
          c.done = c.step >= solo[i].rows.size();
          continue;
        }
        const std::uint32_t rows = solo[i].rows[step];
        const bool drafts_same = drafts[k] == solo[i].drafts[step];
        if (!drafts_same || !same) {  // one-row verifies run alone (above)
          ++exact_mismatches;
          problems_.push_back(std::format("slot {} step {} ({} rows, wave of {}): {}", i, step,
                                          rows, active.size(),
                                          !drafts_same ? "drafts differ" : "verify rows differ"));
        }
        if (discard && i == 0) {
          // Undo its verify; its state must be back but for its draft
          // block's own ring cells (positions not yet committed).
          discarded = true;
          if (auto r = handles[0]->DiscardVerify(); !r) {
            return r;
          }
          std::vector<std::byte> target;
          std::vector<std::byte> ring;
          if (auto r = read(*handles[0], target, ring); !r) {
            return r;
          }
          std::uint64_t stale =
              target.size() != pre_target.size() ||
                      std::memcmp(target.data(), pre_target.data(), target.size()) != 0
                  ? 1
                  : 0;
          auto dlayout = md::DsparkState(dsv4_.dspark_profile(), o_.dsv4.draft_rows);
          if (!dlayout || ring.size() != pre_ring.size()) {
            return Error("the drafter's ring layout");
          }
          auto block = md::DsparkBlock(dsv4_.dspark_profile(), *dlayout, c.pos, c.anchor,
                                       o_.dsv4.draft_rows);
          if (!block) {
            return std::unexpected(block.error());
          }
          const std::uint64_t cell = std::uint64_t{dsv4_.dspark_profile().blocks.head_dim} * 2;
          std::vector<std::uint8_t> drafted(ring.size(), 0);
          for (const std::uint64_t offset : dlayout->offsets) {
            for (const std::int64_t at_cell : block->cells) {
              std::fill_n(
                  drafted.begin() + static_cast<std::ptrdiff_t>(
                                        offset + (static_cast<std::uint64_t>(at_cell) * cell)),
                  cell, 1);
            }
          }
          for (std::size_t b = 0; b < ring.size(); ++b) {
            stale += drafted[b] == 0 && ring[b] != pre_ring[b] ? 1 : 0;
          }
          discard_stale = stale;
          if (stale != 0) {
            problems_.push_back(
                std::format("a discarded wave verify left {} stale state bytes", stale));
          }
          continue;  // the step is re-run by the next wave
        }
        if (discarded && i == 0 && step == 0 && !discard_rerun_exact) {
          discard_rerun_exact = drafts_same && same;
        }
        std::int32_t next = -1;
        const std::uint32_t m = greedy(logits[k], drafts[k], rows, next);
        if (m + 1 != solo[i].kept[step]) {
          problems_.push_back(std::format("slot {} step {} kept {} rows, alone {}", i, step, m + 1,
                                          solo[i].kept[step]));
        }
        if (auto r = handles[i]->Accept(m + 1); !r) {
          return r;
        }
        c.pos += m + 1;
        c.anchor = next;
        ++c.step;
        wave_tokens += m + 1;
        c.done = c.step >= solo[i].rows.size();
      }
    }
    wave_seconds = Seconds(Clock::now() - start);
    // The slot that left: untouched since.
    Cursor& last = at[slots - 1];
    if (last.left) {
      std::vector<std::byte> target;
      std::vector<std::byte> ring;
      if (auto r = read(*handles[slots - 1], target, ring); !r) {
        return r;
      }
      left_unchanged = target == last.left_target && ring == last.left_ring;
      if (!left_unchanged) {
        problems_.emplace_back("the slot that left the waves changed after it left");
      }
    }
    for (std::uint32_t i = 0; i < slots; ++i) {
      auto print = fingerprint(*handles[i]);
      if (!print) {
        return std::unexpected(print.error());
      }
      fingerprints[i] = *print;
      if (spec && !at[i].left && *print != solo[i].fingerprint) {
        problems_.push_back(std::format("slot {}'s state after its waves differs from alone", i));
      }
    }
    return {};
  });
  if (!ran) {
    return ran;
  }
  std::ranges::sort(moves);
  const auto quantile = [&](double q) {
    return moves.empty()
               ? 0.0
               : moves[std::min(moves.size() - 1,
                                static_cast<std::size_t>(q * static_cast<double>(moves.size())))];
  };
  if (spec && !alternate && !discard_rerun_exact) {
    problems_.emplace_back("slot 0's discarded step did not re-run to its solo result");
  }
  std::string width_json;
  std::string width_ms_json;
  for (const auto& [width, count] : widths) {
    width_json += std::format("{}\"{}\":{}", width_json.empty() ? "" : ",", width, count);
    const double ms = 1e3 * width_seconds[width] / static_cast<double>(count);
    width_ms_json += std::format("{}\"{}\":{:.3f}", width_ms_json.empty() ? "" : ",", width, ms);
    std::println("wave check: width {}: {} waves, {:.3f} ms a wave", width, count, ms);
  }
  std::string width_ms;
  for (auto& [width, times] : width_times) {
    std::ranges::sort(times);
    width_ms += std::format("{}\"{}\":{:.3f}", width_ms.empty() ? "" : ",", width,
                            times[times.size() / 2] * 1000);
  }
  std::string_view form = spec ? "DSpark" : "plain";
  if (alternate) {
    form = "plain and DSpark by turns";
  }
  std::println(
      "wave check: {} slots, {}; solo {} tokens in {:.2f} s ({:.2f} tok/s); waves {} ({} tokens in "
      "{:.2f} s, {:.2f} tok/s)",
      slots, form, solo_tokens, solo_seconds, static_cast<double>(solo_tokens) / solo_seconds,
      waves, wave_tokens, wave_seconds, static_cast<double>(wave_tokens) / wave_seconds);
  results_.push_back(std::format(
      R"({{"check":"wave","slots":{},"speculative":{},"share":{},"solo_tokens":{},)"
      R"("solo_seconds":{:.4f},"wave_tokens":{},"wave_seconds":{:.4f},"waves":{},"widths":{{{}}},)"
      R"("width_ms":{{{}}},"width_median_ms":{{{}}},)"
      R"("rows_compared":{},"rows_identical":{},"argmax_agree":{},"largest_difference":{:.6f},)"
      R"("margin_move_p50":{:.6f},"margin_move_p99":{:.6f},"margin_move_max":{:.6f},)"
      R"("exact_mismatches":{},"discard_stale_bytes":{},"discard_rerun_exact":{},)"
      R"("left_unchanged":{}}})",
      slots, spec ? "true" : "false", share, solo_tokens, solo_seconds, wave_tokens, wave_seconds,
      waves, width_json, width_ms_json, width_ms, rows_compared, rows_identical, argmax_agree,
      largest, quantile(0.5), quantile(0.99), moves.empty() ? 0.0 : moves.back(), exact_mismatches,
      discard_stale ? std::format("{}", *discard_stale) : std::string("null"),
      discard_rerun_exact ? "true" : "false", left_unchanged ? "true" : "false"));
  return {};
}

// --check plan-memory (--slots N; docs/experiments/deepseek-batching/,
// "Plan memory"): what the runner's plans and graphs hold outside the
// catalog. Every plan dropped first; then, one new shape at a time, the
// process's heap in use (mallinfo2), its resident set and MemAvailable
// before and after: a one-row chunk at a new position (its plan; the
// state's growth is device memory), the same chunk again (its graph's
// capture) and a third time (a replay: nothing should grow); a prefill
// chunk of a new width; a decode wave of every slot at new positions (its
// plan, then its graph); with a drafter, a DSpark wave (each slot's draft
// plan and the wave's plan, then their graphs). Each kind's per-shape
// heap is held against what the runner counts for it (cached_plan_bytes),
// and the largest graph's growth against kGraphNodeHostBytes a node.
Status Harness::PlanMemory() {
  using Slot = jb::Dsv4Runner::Slot;
  const std::uint32_t slots = o_.dsv4.wave_slots;
  if (slots < 2) {
    return Error("the plan-memory check needs --slots 2 to 16");
  }
  std::vector<Slot*> handles;
  for (std::uint32_t i = 0; i < slots; ++i) {
    auto slot = dsv4_.request_slot(i);
    if (!slot) {
      return std::unexpected(slot.error());
    }
    handles.push_back(*slot);
  }
  struct Sample {
    std::int64_t heap = 0;
    std::int64_t rss = 0;
    std::int64_t available = 0;
    std::int64_t counted = 0;
    std::int64_t graph_device = 0;
    std::int64_t graph_counted = 0;
    // The reclaim costs (docs/experiments/memory-pressure): planning's and
    // capturing's seconds, and the kept plans' arenas (capacity and use).
    double plan_seconds = 0;
    double capture_seconds = 0;
    std::int64_t arena = 0;
    std::int64_t arena_used = 0;
  };
  const auto sample = [&]() {
    Sample s;
    const struct mallinfo2 m = mallinfo2();
    s.heap = static_cast<std::int64_t>(m.uordblks + m.hblkhd);
    std::ifstream statm("/proc/self/statm");
    std::int64_t pages = 0;
    std::int64_t resident = 0;
    statm >> pages >> resident;
    s.rss = resident * 4096;
    std::ifstream meminfo("/proc/meminfo");
    std::string key;
    std::int64_t kib = 0;
    std::string unit;
    while (meminfo >> key >> kib >> unit) {
      if (key == "MemAvailable:") {
        s.available = kib * 1024;
        break;
      }
    }
    s.counted = static_cast<std::int64_t>(dsv4_.cached_plan_bytes());
    s.graph_device = dsv4_.graph_stats().memory_bytes + dsv4_.draft_stats().memory_bytes;
    s.graph_counted = static_cast<std::int64_t>(dsv4_.cached_graph_bytes());
    s.plan_seconds = dsv4_.plan_seconds();
    for (const auto* g : {&dsv4_.graph_stats(), &dsv4_.draft_stats()}) {
      s.capture_seconds += g->capture_seconds + g->instantiate_seconds;
    }
    s.arena_used = static_cast<std::int64_t>(dsv4_.cached_arena_used());
    s.arena = static_cast<std::int64_t>(dsv4_.cached_arena_used(true));
    return s;
  };
  struct Kind {
    std::string name;
    std::vector<Sample> deltas;
    std::vector<std::int64_t> plans;     // new plans each
    std::vector<std::int64_t> graphs;    // new graphs each
    std::vector<std::uint64_t> dropped;  // graphs dropped to stay within the caps
  };
  std::map<std::string, Kind> kinds;
  const auto measure = [&](const std::string& name, const std::function<Status()>& run) -> Status {
    const Sample before = sample();
    const auto plans = static_cast<std::int64_t>(dsv4_.plans());
    const std::uint64_t dropped = dsv4_.graph_stats().dropped;
    const auto graphs = static_cast<std::int64_t>(dsv4_.graphs());
    if (auto r = run(); !r) {
      return Error(std::format("{}: {}", name, r.error()));
    }
    const Sample after = sample();
    Kind& k = kinds[name];
    k.name = name;
    k.deltas.push_back({.heap = after.heap - before.heap,
                        .rss = after.rss - before.rss,
                        .available = before.available - after.available,
                        .counted = after.counted - before.counted,
                        .graph_device = after.graph_device - before.graph_device,
                        .graph_counted = after.graph_counted - before.graph_counted,
                        .plan_seconds = after.plan_seconds - before.plan_seconds,
                        .capture_seconds = after.capture_seconds - before.capture_seconds,
                        .arena = after.arena - before.arena,
                        .arena_used = after.arena_used - before.arena_used});
    k.plans.push_back(static_cast<std::int64_t>(dsv4_.plans()) - plans);
    k.graphs.push_back(static_cast<std::int64_t>(dsv4_.graphs()) - graphs);
    k.dropped.push_back(dsv4_.graph_stats().dropped - dropped);
    return {};
  };
  dsv4_.DropPlans();
  dsv4_.set_graphs(true);
  if (auto r = dsv4_.SelectSlots(handles); !r) {
    return r;
  }
  const std::uint64_t bound = dsv4_.plan_floor_bytes();
  const std::int32_t token = 1000;
  constexpr std::uint32_t kShapes = 4;
  auto ran = node_.WithRequest(kDsv4, dsv4_.execution_closure(), "plan memory", [&]() -> Status {
    std::vector<float> row;
    // One-row chunks of slot 0 at new positions: plan, capture, replay.
    for (std::uint32_t k = 0; k < kShapes; ++k) {
      const std::uint32_t pos = 1000 + (1024 * k);
      const auto step = [&]() {
        return handles[0]->Chunk(pos, std::span(&token, 1), row, jb::Dsv4ChunkKind::kPlain);
      };
      for (const char* name : {"chunk-plan", "chunk-graph", "chunk-replay"}) {
        if (auto r = measure(name, step); !r) {
          return r;
        }
      }
    }
    // Prefill chunks of slot 1, each a new width.
    const std::uint32_t prefill = std::min<std::uint32_t>(o_.dsv4.max_rows, 2048);
    for (std::uint32_t k = 0; k < kShapes; ++k) {
      const std::vector<std::int32_t> tokens(prefill - (8 * k), token);
      if (auto r = measure("prefill-plan", [&]() { return handles[1]->Chunk(0, tokens, row); });
          !r) {
        return r;
      }
    }
    // Decode waves of every slot at new positions: plan, capture, replay.
    std::vector<std::vector<float>> outs(slots);
    std::vector<std::vector<std::int32_t>> drafts(slots);
    for (std::uint32_t k = 0; k < kShapes; ++k) {
      std::vector<jb::Dsv4Runner::WaveWork> work;
      work.reserve(slots);
      for (std::uint32_t i = 0; i < slots; ++i) {
        work.push_back({.slot = handles[i],
                        .pos = 6000 + (1024 * k) + (37 * i),
                        .anchor = token,
                        .rows = 1,
                        .drafts = nullptr,
                        .logits = &outs[i]});
      }
      for (const char* name : {"wave-plan", "wave-graph", "wave-replay"}) {
        if (auto r = measure(name, [&]() { return dsv4_.DecodeWave(work); }); !r) {
          return r;
        }
      }
    }
    if (!dsv4_.speculative()) {
      return {};
    }
    // DSpark waves: each slot's draft and one joined verify, then Accept.
    const std::uint32_t share = std::min<std::uint32_t>(
        o_.dsv4.max_verify, static_cast<std::uint32_t>(jb::Dsv4Runner::kWaveRows) / slots);
    for (std::uint32_t k = 0; k < kShapes; ++k) {
      std::vector<jb::Dsv4Runner::WaveWork> work;
      work.reserve(slots);
      for (std::uint32_t i = 0; i < slots; ++i) {
        std::uint32_t pos = 12000 + (1024 * k) + (64 * i);
        std::uint32_t rows = share;
        while (rows > 1 && !md::Dsv4SameWidths(dsv4_.state_layout(), pos, rows)) {
          --rows;
        }
        work.push_back({.slot = handles[i],
                        .pos = pos,
                        .anchor = token,
                        .rows = rows,
                        .drafts = &drafts[i],
                        .logits = &outs[i]});
      }
      const auto wave = [&]() -> Status {
        if (auto r = dsv4_.DraftVerifyWave(work); !r) {
          return r;
        }
        for (Slot* slot : handles) {
          if (auto r = slot->Accept(1); !r) {
            return r;
          }
        }
        return {};
      };
      for (const char* name : {"spec-wave-plan", "spec-wave-graph", "spec-wave-replay"}) {
        if (auto r = measure(name, wave); !r) {
          return r;
        }
      }
    }
    return {};
  });
  if (!ran) {
    return ran;
  }
  // Reclaim costs (docs/experiments/memory-pressure): the last slot's
  // prefill of four chunks of max_rows, timed with its plans kept (what
  // recomputing its state costs a byte), then every slot's state written
  // back to its spill file and read again (what spilling costs a byte).
  double prefill_seconds = 0;
  std::uint64_t prefill_tokens = 0;
  std::uint64_t prefill_state = 0;
  ran = node_.WithRequest(kDsv4, dsv4_.execution_closure(), "reclaim costs", [&]() -> Status {
    std::vector<float> row;
    Slot* slot = handles.back();
    const std::uint32_t rows = o_.dsv4.max_rows;
    const std::vector<std::int32_t> tokens(rows, token);
    for (int pass = 0; pass < 2; ++pass) {
      if (auto r = slot->Clear(); !r) {
        return r;
      }
      const auto start = std::chrono::steady_clock::now();
      for (std::uint32_t k = 0; k < 4 && (k + 1) * rows < o_.dsv4.context; ++k) {
        if (auto r = slot->Chunk(k * rows, tokens, row); !r) {
          return r;
        }
        prefill_tokens = std::uint64_t{k + 1} * rows;
      }
      prefill_seconds =
          std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
      prefill_state = slot->used_state_bytes();
    }
    return {};
  });
  if (!ran) {
    return ran;
  }
  const std::vector<jitllm::catalog::ExtentId> state = dsv4_.state();
  const std::uint64_t state_bytes = state.size() * (std::uint64_t{2} << 20U);
  const auto spill_start = std::chrono::steady_clock::now();
  if (auto r = node_.Evict(state); !r) {
    return r;
  }
  const auto spilled = std::chrono::steady_clock::now();
  std::vector<ts::LoadStats> restore_log;
  if (auto r = node_.Load(state, "restoring the state", restore_log); !r) {
    return r;
  }
  const auto restored = std::chrono::steady_clock::now();
  const double spill_seconds = std::chrono::duration<double>(spilled - spill_start).count();
  const double restore_seconds = std::chrono::duration<double>(restored - spilled).count();
  std::println(
      "reclaim costs: prefill {} tokens in {:.3f} s ({:.0f} tok/s), state {:.1f} MiB: {:.2f} s a "
      "GiB recomputed; spill {:.1f} MiB in {:.3f} s ({:.2f} GB/s), restore {:.3f} s ({:.2f} "
      "GB/s): {:.3f} s a GiB both ways",
      prefill_tokens, prefill_seconds, static_cast<double>(prefill_tokens) / prefill_seconds,
      static_cast<double>(prefill_state) / (1 << 20),
      prefill_seconds / (static_cast<double>(prefill_state) / (1 << 30)),
      static_cast<double>(state_bytes) / (1 << 20), spill_seconds,
      static_cast<double>(state_bytes) / spill_seconds / 1e9, restore_seconds,
      static_cast<double>(state_bytes) / restore_seconds / 1e9,
      (spill_seconds + restore_seconds) / (static_cast<double>(state_bytes) / (1 << 30)));
  results_.push_back(std::format(
      R"({{"check":"reclaim-costs","prefill_tokens":{},"prefill_seconds":{:.6f},"prefill_state_bytes":{},"spill_bytes":{},"spill_seconds":{:.6f},"restore_seconds":{:.6f}}})",
      prefill_tokens, prefill_seconds, prefill_state, state_bytes, spill_seconds, restore_seconds));
  std::string rows;
  for (const auto& [name, k] : kinds) {
    std::string deltas;
    for (std::size_t i = 0; i < k.deltas.size(); ++i) {
      const Sample& d = k.deltas[i];
      deltas += std::format(
          R"({}{{"plans":{},"graphs":{},"dropped":{},"heap":{},"rss":{},"available_drop":{},"counted":{},"graph_counted":{},"graph_device":{},"plan_seconds":{:.6f},"capture_seconds":{:.6f},"arena":{},"arena_used":{}}})",
          deltas.empty() ? "" : ",", k.plans[i], k.graphs[i], k.dropped[i], d.heap, d.rss,
          d.available, d.counted, d.graph_counted, d.graph_device, d.plan_seconds,
          d.capture_seconds, d.arena, d.arena_used);
      std::println(
          "plan memory: {} #{}: plans {:+}, graphs {:+} ({} dropped); heap {:+.2f} MiB, "
          "rss {:+.2f} MiB, "
          "MemAvailable -{:.2f} MiB, counted {:+.2f} MiB, graphs counted {:+.2f} MiB, "
          "graph device {:.2f} MiB; planning {:.1f} ms, capture {:.1f} ms; arena {:+.2f} MiB "
          "({:+.2f} used)",
          name, i, k.plans[i], k.graphs[i], k.dropped[i], static_cast<double>(d.heap) / (1 << 20),
          static_cast<double>(d.rss) / (1 << 20), static_cast<double>(d.available) / (1 << 20),
          static_cast<double>(d.counted) / (1 << 20),
          static_cast<double>(d.graph_counted) / (1 << 20),
          static_cast<double>(d.graph_device) / (1 << 20), d.plan_seconds * 1e3,
          d.capture_seconds * 1e3, static_cast<double>(d.arena) / (1 << 20),
          static_cast<double>(d.arena_used) / (1 << 20));
      // A kind's first shape also pays one-time costs (the driver loading
      // its kernels), and a shape that made the runner drop another plan
      // or graph nets them: reported, not held to the count. Otherwise a
      // plan's heap, or a graph's, past what the runner counts for it is a
      // defect. (MemAvailable moves with the rest of the system too: a
      // graph's device memory is reported beside it, not checked.)
      if (i == 0 || k.dropped[i] != 0 || k.plans[i] < 0 || k.graphs[i] < 0) {
        continue;
      }
      if (k.plans[i] > 0 && d.counted < d.heap) {
        problems_.push_back(
            std::format("{} #{}: heap +{} bytes but {} counted", name, i, d.heap, d.counted));
      }
      if (k.plans[i] == 0 && k.graphs[i] > 0 && d.graph_counted < d.heap) {
        problems_.push_back(std::format("{} #{}: a graph's heap +{} bytes but {} counted", name, i,
                                        d.heap, d.graph_counted));
      }
    }
    rows += std::format(R"({}"{}":[{}])", rows.empty() ? "" : ",", name, deltas);
  }
  std::println("plan memory: bound {:.2f} MiB; kept {} plans, {} graphs, counted {:.2f} MiB",
               static_cast<double>(bound) / (1 << 20), dsv4_.plans(), dsv4_.graphs(),
               static_cast<double>(dsv4_.cached_plan_bytes()) / (1 << 20));
  results_.push_back(std::format(
      R"({{"check":"plan-memory","slots":{},"speculative":{},"plan_floor_bytes":{},"kinds":{{{}}}}})",
      slots, dsv4_.speculative() ? "true" : "false", bound, rows));
  return {};
}

// --check capacity (--slots N, --state-budget-mib B; docs/runtime-serving.md,
// "State capacity in a cohort"): the runner's typed capacity refusal and
// the idle-state release the serving cohort waits on. With slots 0 and 1
// selected, slot 0 takes about 60% of the state's room; slot 1's growth to
// the same is then refused for capacity (Slot::state_refused, the state
// usable and unchanged), while a growth past the context is refused but
// not for capacity. With slot 1 selected alone, it cannot clear itself as
// idle; slot 0 (now idle) can, and slot 1's growth and a chunk then run.
Status Harness::Capacity() {
  using Slot = jb::Dsv4Runner::Slot;
  if (o_.dsv4.wave_slots < 2 || o_.state_budget_mib == 0) {
    return Error("the capacity check needs --slots 2 to 16 and --state-budget-mib");
  }
  std::array<Slot*, 2> slots{};
  for (std::uint32_t i = 0; i < 2; ++i) {
    auto slot = dsv4_.request_slot(i);
    if (!slot) {
      return std::unexpected(slot.error());
    }
    slots[i] = *slot;
  }
  // The positions whose state (and the DSpark ring) is about 60% of the room.
  const std::uint64_t room = o_.state_budget_mib << 20U;
  const auto bytes_through = [&](std::uint32_t positions) -> std::uint64_t {
    auto ranges = md::Dsv4UsedState(dsv4_.state_layout(), positions);
    std::uint64_t bytes = dsv4_.drafter_state_bytes();
    if (ranges) {
      for (const auto& r : *ranges) {
        bytes += r.bytes;
      }
    }
    return bytes;
  };
  const std::uint64_t target = (room * 6) / 10;
  std::uint32_t lo = 1;
  std::uint32_t hi = o_.dsv4.context - 1;
  while (lo < hi) {
    const std::uint32_t mid = lo + ((hi - lo + 1) / 2);
    if (bytes_through(mid) <= target) {
      lo = mid;
    } else {
      hi = mid - 1;
    }
  }
  const std::uint32_t positions = lo;
  if (bytes_through(positions) * 2 <= room || bytes_through(positions) > room) {
    return Error(std::format("no position fills 60% of a {} MiB room", o_.state_budget_mib));
  }
  std::string failures;
  const auto expect = [&](bool ok, std::string_view what) {
    if (!ok) {
      failures += std::format("{}{}", failures.empty() ? "" : "; ", what);
    }
  };
  std::uint64_t used_before = 0;
  std::vector<float> control;       // slot 0's next-token logits before any spill
  std::vector<float> control_then;  // and the token's after it
  std::uint64_t used_refused = 0;
  if (auto r = dsv4_.SelectSlots(slots); !r) {
    return r;
  }
  auto both = node_.WithRequest(kDsv4, dsv4_.execution_closure(), "two slots", [&]() -> Status {
    for (Slot* slot : slots) {
      if (auto r = slot->Clear(); !r) {
        return r;
      }
    }
    if (auto r = slots[0]->ReserveStateThrough(positions); !r) {
      return Error(std::format("slot 0's state through {}: {}", positions, r.error()));
    }
    expect(!slots[0]->state_refused(), "a growth that fit is not a refusal");
    {
      // Contents for the spill below to keep: kSpillPrefix tokens of a
      // prefill (enough that some pages lie wholly before the next step,
      // for the incremental spill); and its control, the next tokens'
      // logits from that state, before it is cleared and prefilled again
      // (the same chunks, the same state).
      const std::vector<std::int32_t> tokens(kSpillPrefix, 1000);
      const auto prefill = [&]() -> Status {
        std::vector<float> row;
        for (std::uint32_t at = 0; at < kSpillPrefix; at += kSpillChunk) {
          const auto part = std::span(tokens).subspan(at, std::min(kSpillChunk, kSpillPrefix - at));
          if (auto r = slots[0]->Chunk(at, part, row); !r) {
            return r;
          }
        }
        return {};
      };
      const std::int32_t next = 2000;
      if (auto r = prefill(); !r) {
        return Error(std::format("slot 0's prefill: {}", r.error()));
      }
      if (auto r = slots[0]->Chunk(kSpillPrefix, std::span(&next, 1), control); !r) {
        return Error(std::format("slot 0's control step: {}", r.error()));
      }
      const std::int32_t then = 2001;
      if (auto r = slots[0]->Chunk(kSpillPrefix + 1, std::span(&then, 1), control_then); !r) {
        return Error(std::format("slot 0's second control step: {}", r.error()));
      }
      if (auto r = slots[0]->Clear(); !r) {
        return r;
      }
      if (auto r = slots[0]->ReserveStateThrough(positions); !r) {
        return Error(std::format("slot 0's state through {} again: {}", positions, r.error()));
      }
      if (auto r = prefill(); !r) {
        return Error(std::format("slot 0's prefill again: {}", r.error()));
      }
    }
    used_before = slots[1]->used_state_bytes();
    auto refused = slots[1]->ReserveStateThrough(positions);
    expect(!refused.has_value(), "slot 1's growth beside slot 0's fit the room");
    expect(slots[1]->state_refused(), "the budget's refusal is typed as capacity");
    expect(slots[1]->state_usable(), "a capacity refusal leaves the state usable");
    used_refused = slots[1]->used_state_bytes();
    auto past = slots[1]->ReserveStateThrough(o_.dsv4.context + 8);
    expect(!past.has_value() && !slots[1]->state_refused(),
           "a growth past the context is refused, but not for capacity");
    return {};
  });
  if (!both) {
    return both;
  }
  // Spill, not clear (docs/runtime-serving.md#state-capacity-in-a-cohort):
  // idle slot 0's state is written to its spill file and its backing
  // released, so slot 1's growth now fits; slot 0's restore while slot 1
  // holds the room is refused for capacity (typed, still spilled); once
  // there is room again it comes back, and the next token's logits from it
  // equal the control's from the unspilled state, bit for bit.
  std::vector<float> after;
  std::vector<float> after_then;
  std::uint64_t spilled_bytes = 0;
  std::uint64_t incremental_bytes = 0;
  const std::array<Slot*, 1> zero = {slots[0]};
  const std::array<Slot*, 1> alone = {slots[1]};
  if (auto r = dsv4_.SelectSlots(alone); !r) {
    return r;
  }
  if (auto r = node_.WithRequest(
          kDsv4, dsv4_.execution_closure(), "slot 1 beside spilled slot 0",
          [&]() -> Status {
            if (auto s = slots[0]->Spill(); !s) {
              return Error(std::format("spilling idle slot 0: {}", s.error()));
            }
            expect(slots[0]->spilled(), "an idle slot spills");
            spilled_bytes = slots[0]->spilled_bytes();
            expect(spilled_bytes != 0, "a spilled slot's file holds its state");
            if (auto g = slots[1]->ReserveStateThrough(positions); !g) {
              return Error(std::format("slot 1's growth beside spilled slot 0: {}", g.error()));
            }
            return {};
          });
      !r) {
    return r;
  }
  if (auto r = dsv4_.SelectSlots(zero); !r) {
    return r;
  }
  if (auto r = node_.WithRequest(
          kDsv4, dsv4_.execution_closure(), "slot 0 restored",
          [&]() -> Status {
            const auto refused = slots[0]->Restore();
            expect(!refused.has_value() && slots[0]->state_refused() && slots[0]->spilled(),
                   "a restore past the room is refused for capacity, the slot still spilled");
            if (auto c = slots[1]->ClearIdle(); !c) {
              return Error(std::format("clearing idle slot 1: {}", c.error()));
            }
            if (auto s = slots[0]->Restore(); !s) {
              return Error(std::format("restoring slot 0: {}", s.error()));
            }
            expect(!slots[0]->spilled() && !slots[0]->state_refused(), "a restore with room runs");
            const std::int32_t next = 2000;
            if (auto c = slots[0]->Chunk(kSpillPrefix, std::span(&next, 1), after); !c) {
              return c;
            }
            // An incremental spill: only the pages the step from there may
            // have changed are written; the rest are released with the
            // file's copy kept. Restored, the state continues exactly.
            incremental_bytes = slots[0]->spill_write_bytes();
            if (auto s = slots[0]->Spill(); !s) {
              return Error(std::format("spilling slot 0 again: {}", s.error()));
            }
            expect(slots[0]->spill_write_bytes() == 0, "a spilled slot writes nothing more");
            if (auto s = slots[0]->Restore(); !s) {
              return Error(std::format("restoring slot 0 again: {}", s.error()));
            }
            expect(slots[0]->spill_write_bytes() == 0,
                   "a restored slot nothing wrote since has nothing to write");
            const std::int32_t then = 2001;
            return slots[0]->Chunk(kSpillPrefix + 1, std::span(&then, 1), after_then);
          });
      !r) {
    return r;
  }
  expect(!control.empty() && after == control,
         "the next token's logits from a restored state equal the unspilled control's");
  expect(incremental_bytes != 0 && incremental_bytes < spilled_bytes,
         "a second spill writes only what changed since the first");
  expect(!control_then.empty() && after_then == control_then,
         "the logits after an incremental spill and restore equal the unspilled control's");
  bool cleared_active = true;
  std::uint64_t idle_after = 0;
  if (auto r = dsv4_.SelectSlots(alone); !r) {
    return r;
  }
  auto one = node_.WithRequest(kDsv4, dsv4_.execution_closure(), "slot 1 alone", [&]() -> Status {
    cleared_active = slots[1]->ClearIdle().has_value();
    expect(!cleared_active, "a selected slot is not cleared as idle");
    if (auto r = slots[0]->ClearIdle(); !r) {
      return Error(std::format("clearing idle slot 0: {}", r.error()));
    }
    idle_after = slots[0]->used_state_bytes();
    expect(idle_after == 0, "an idle slot's cleared state holds nothing");
    if (auto r = slots[1]->ReserveStateThrough(positions); !r) {
      return Error(std::format("slot 1's growth after the idle clear: {}", r.error()));
    }
    std::vector<float> row;
    const std::int32_t token = 1000;
    return slots[1]->Chunk(positions - 1, std::span(&token, 1), row);
  });
  if (!one) {
    return one;
  }
  if (!failures.empty()) {
    problems_.push_back("capacity: " + failures);
  }
  std::println(
      "capacity: room {} MiB, {} positions ({:.1f} MiB a slot); slot 1 before {} bytes, "
      "after its refusal {} bytes; spilled {} bytes, then {} incrementally; {}",
      o_.state_budget_mib, positions, static_cast<double>(bytes_through(positions)) / (1 << 20),
      used_before, used_refused, spilled_bytes, incremental_bytes,
      failures.empty() ? "ok" : failures);
  results_.push_back(std::format(
      R"({{"check":"capacity","room_mib":{},"positions":{},"slot_bytes":{},"used_before":{},"used_after_refusal":{},"idle_after_clear":{},"spilled_bytes":{},"incremental_spill_bytes":{},"failures":{}}})",
      o_.state_budget_mib, positions, bytes_through(positions), used_before, used_refused,
      idle_after, spilled_bytes, incremental_bytes, failures.empty() ? 0 : 1));
  return {};
}

Status Harness::Run() {
  if (auto r = Tokenize(); !r) {
    return r;
  }
  if (auto r = node_.Open(); !r) {
    return r;
  }
  entered_models_.push_back(&dsv4_);
  if (auto r = dsv4_.Setup(); !r) {
    return r;
  }
  if (o_.dsv4.prefill_outa_hca &&
      dsv4_.artifact().id() != "8a355bfb27c90e1150fbd7fa62ea6e63f6bf34fcca33934e52d22773f1508234" &&
      dsv4_.artifact().id() != "cd39d504dc2dbfe911a4a521fa8efc8053dc3e80e99738a9b25fa6b70c97a1ac") {
    return Error("combined prefill runner control is limited to its two qualified artifacts");
  }
  if (with_fp16()) {
    entered_models_.push_back(&fp16_);
    if (auto r = fp16_.Setup(); !r) {
      return r;
    }
  }
  if (auto r = node_.MapWorkspace(std::max(dsv4_.activations_needed(), fp16_.activations_needed()),
                                  std::max(dsv4_.pool_needed(), fp16_.pool_needed()));
      !r) {
    return r;
  }
  const std::uint64_t fixed = node_.catalog().OccupancyOf(node_.domain()).Total().value();
  // State diagnostics keep a cataloged pinned copy alongside the device state.
  const std::uint64_t budget =
      fixed +
      (o_.state_budget_mib != 0 ? (o_.state_budget_mib << 20U) : 2 * node_.StateCapacity()) +
      ((dsv4_.weights().size() + (with_fp16() ? fp16_.weights().size() : 0)) * ts::kPagedExtent);
  if (auto r = node_.Start(Bytes(budget)); !r) {
    return r;
  }
  if (auto r = dsv4_.Register(); !r) {
    return r;
  }
  if (with_fp16()) {
    if (auto r = fp16_.Register(); !r) {
      return r;
    }
  }
  if (auto r = dsv4_.Bind(); !r) {
    return r;
  }
  if (with_fp16()) {
    if (auto r = fp16_.Bind(); !r) {
      return r;
    }
  }
  node_.Run();
  if (o_.check == "sizing") {
    return Write();
  }
  std::vector<ts::LoadStats> log;
  const auto start = Clock::now();
  if (auto r = node_.Load(dsv4_.weights(), "A's first load", log); !r) {
    return r;
  }
  weights_loaded_ = true;
  load_seconds_ = Seconds(Clock::now() - start);
  std::println("loaded: {} bytes ({} the drafter's) in {:.2f} s", dsv4_.weight_read_bytes(),
               dsv4_.drafter_read_bytes(), load_seconds_);
  if (auto r = dsv4_.CheckHashRouting(); !r) {
    return r;
  }
  // A standalone draft from the empty prefix must initialize its ring
  // before dispatch, including after Clear released growing state.
  if (auto r = dsv4_.Clear(); !r) {
    return r;
  }
  if (dsv4_.speculative()) {
    std::vector<std::int32_t> empty_drafts;
    if (auto r = dsv4_.Draft(0, 0, empty_drafts); !r) {
      return r;
    }
    if (empty_drafts.size() != o_.dsv4.draft_rows ||
        std::ranges::any_of(empty_drafts, [&](std::int32_t t) {
          return t < 0 || std::cmp_greater_equal(t, dsv4_.vocab());
        })) {
      return Error("a standalone empty-prefix draft returned invalid tokens");
    }
    std::println("empty-prefix standalone draft: {} valid tokens", empty_drafts.size());
    if (auto r = dsv4_.Clear(); !r) {
      return r;
    }
  }
  Status checked;
  if (o_.check == "frontier") {
    checked = Frontier();
  } else if (o_.check == "greedy") {
    checked = Greedy();
  } else if (o_.check == "forced") {
    checked = Forced();
  } else if (o_.check == "probe") {
    checked = Probe();
  } else if (o_.check == "swap") {
    checked = Swap();
  } else if (o_.check == "sampled-plain") {
    checked = Sampled(false);
  } else if (o_.check == "sampled-spec") {
    checked = Sampled(true);
  } else if (o_.check == "capacity") {
    checked = Capacity();
  } else if (o_.check == "plan-memory") {
    checked = PlanMemory();
  } else if (o_.check == "wave") {
    checked = Wave();
  } else {
    checked = Error(std::format("no check {}", o_.check));
  }
  if (!checked) {
    return checked;
  }
  if (dsv4_.coverage_violations() != 0) {
    problems_.push_back(
        std::format("{} bound tensors outside cataloged extents of their class; "
                    "first {}",
                    dsv4_.coverage_violations(), dsv4_.first_violation()));
  }
  return Write();
}

Status Harness::Write() {
  std::string all;
  for (const std::string& r : results_) {
    all += (all.empty() ? "" : ",") + r;
  }
  std::string problems;
  for (const std::string& p : problems_) {
    std::string quoted;
    jitllm::base::json::AppendQuoted(p, quoted);
    problems += (problems.empty() ? "" : ",") + quoted;
  }
  const std::uint64_t drop = available_before_ - std::min(available_before_, memory_.low());
  std::ofstream(o_.out / "spec.json")
      << std::format(
             R"({{"check":"{}","frontier_head":{},"prefill_outa_hca":{},"weights_loaded":{},"activations_bytes":{},"pool_bytes":{},"host_input_bytes":{},"load_seconds":{:.2f},"read_bytes":{},"drafter_read_bytes":{},)"
             R"("peak_memavailable_drop_bytes":{},"graphs":{{"eager":{},"captured":{},"replayed":{},)"
             R"("refused":{}}},"results":[{}],"problems":[{}]}})",
             o_.check, o_.dsv4.frontier_head ? "true" : "false",
             o_.dsv4.prefill_outa_hca ? "true" : "false", weights_loaded_ ? "true" : "false",
             dsv4_.activations_needed(), dsv4_.pool_needed(), dsv4_.host_input_bytes(),
             load_seconds_, dsv4_.weight_read_bytes(), dsv4_.drafter_read_bytes(), drop,
             dsv4_.graph_stats().eager, dsv4_.graph_stats().captured, dsv4_.graph_stats().replayed,
             dsv4_.graph_stats().refused, all, problems)
      << '\n';
  std::println("peak MemAvailable drop: {:.2f} GiB", static_cast<double>(drop) / (1U << 30U));
  if (!problems_.empty()) {
    std::string joined;
    for (const std::string& p : problems_) {
      joined += (joined.empty() ? "" : "; ") + p;
    }
    return Error(joined);
  }
  return {};
}

std::expected<Options, std::string> Parse(std::span<char*> args) {
  Options o;
  o.fp16.trajectory = "control";
  o.fp16.fusion = true;
  for (std::size_t i = 1; i < args.size(); ++i) {
    const std::string_view a = args[i];
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
    } else if (a == "--drafter") {
      o.dsv4.drafter = v;
    } else if (a == "--prompts") {
      o.prompts = v;
    } else if (a == "--out") {
      o.out = v;
    } else if (a == "--check") {
      o.check = v;
    } else if (a == "--tokens") {
      ok = number(o.tokens) && o.tokens >= 2;
    } else if (a == "--context") {
      ok = number(o.dsv4.context);
    } else if (a == "--max-rows") {
      ok = number(o.dsv4.max_rows) && o.dsv4.max_rows >= 1;
    } else if (a == "--graphs") {
      o.dsv4.graphs = v == "on";
      ok = v == "on" || v == "off";
    } else if (a == "--exact") {
      o.dsv4.exact = v == "on";
      ok = v == "on" || v == "off";
    } else if (a == "--frontier-head") {
      o.dsv4.frontier_head = v == "on";
      ok = v == "on" || v == "off";
    } else if (a == "--prefill-outa-hca") {
      o.dsv4.prefill_outa_hca = v == "on";
      ok = v == "on" || v == "off";
    } else if (a == "--margin") {
      const auto [end, ec] = std::from_chars(v.data(), v.data() + v.size(), o.margin);
      ok = ec == std::errc() && end == v.data() + v.size() && o.margin >= 0.0;
    } else if (a == "--draft") {
      ok = number(o.dsv4.draft_rows) && o.dsv4.draft_rows >= 1;
      o.dsv4.max_verify = o.dsv4.draft_rows + 1;
    } else if (a == "--fp16-artifact") {
      o.fp16.artifact = v;
    } else if (a == "--fp16-tokens") {
      o.fp16.tokens = v;
    } else if (a == "--fp16-expect") {
      o.fp16_expect = v;
    } else if (a == "--seeds") {
      ok = number(o.seeds) && o.seeds >= 1;
    } else if (a == "--poll-us") {
      std::uint32_t us = 0;
      ok = number(us) && us <= 1000000;
      o.poll_us = us;
    } else if (a == "--sampled") {
      o.sampled = v;
    } else if (a == "--only") {
      o.only = v;
    } else if (a == "--probe-step") {
      ok = number(o.probe_step) && o.probe_step >= 1;
    } else if (a == "--wave-mode") {
      o.wave_mode = v;
      ok = v == "decode" || v == "verify" || v == "alternate";
    } else if (a == "--state-budget-mib") {
      ok = number(o.state_budget_mib) && o.state_budget_mib > 0;
    } else if (a == "--slots") {
      ok = number(o.dsv4.wave_slots) && o.dsv4.wave_slots >= 2 &&
           o.dsv4.wave_slots <= jitllm::engine::kMaxRequestSlots;
    } else if (a == "--wave-lanes") {
      o.dsv4.wave_lanes = v == "on";
      ok = v == "on" || v == "off";
    } else if (a == "--joined-drafts") {
      o.dsv4.joined_drafts = v == "on";
      ok = v == "on" || v == "off";
    } else {
      return Error(std::format("unknown argument {}", a));
    }
    if (!ok) {
      return Error(std::format("{} does not take {}", a, v));
    }
  }
  // The slot checks (wave, plan-memory, capacity) run plain without a
  // drafter; every other check speculates.
  const bool slotted = o.check == "wave" || o.check == "plan-memory" || o.check == "capacity";
  if (o.dsv4.artifact.empty() || (o.dsv4.drafter.empty() && !slotted) || o.prompts.empty() ||
      o.out.empty() || o.check.empty() || (o.fp16.artifact.empty() != o.fp16.tokens.empty()) ||
      (slotted != (o.dsv4.wave_slots > 1))) {
    return Error(
        "usage: jitllm_spec_runner --dsv4-artifact DIR [--drafter DIR] --prompts FILE --out DIR "
        "--check greedy|forced|swap|sampled-plain|sampled-spec|probe|frontier|sizing|wave|"
        "plan-memory|capacity "
        "[--tokens N] [--context N] [--max-rows N] "
        "[--graphs on|off] [--exact on|off] [--margin B] [--draft N] "
        "[--prefill-outa-hca on|off] [--fp16-artifact DIR "
        "--fp16-tokens FILE "
        "--fp16-expect SHA256] [--seeds N] [--sampled FILE] [--probe-step N] "
        "[--slots N (wave, plan-memory, capacity: 2-16)] [--wave-mode verify|decode|alternate] "
        "[--wave-lanes on|off] [--joined-drafts on|off] "
        "[--state-budget-mib N]");
  }
  // The paired control needs the all-row workspace for its original arm.
  if (o.check == "frontier") {
    if (o.dsv4.exact || o.dsv4.context > 131072 || o.dsv4.max_rows > 4096 || o.tokens > 1024 ||
        o.tokens > o.dsv4.context) {
      return Error(
          "the frontier diagnostic needs fast mode, context <=131072, chunks <=4096 "
          "and outputs <=1024 within context");
    }
    o.dsv4.frontier_head = false;
  }
  if (o.dsv4.prefill_outa_hca &&
      (o.dsv4.exact || o.dsv4.max_rows != 4096 || o.dsv4.context > 131072 || o.check == "probe")) {
    return Error("combined prefill runner control needs fast4096 mode, context<=131072, no probe");
  }
  // The probe runs the fast plan and the reference mode over one state,
  // which needs the full window cache (engine/dsv4_runner.h set_exact).
  o.dsv4.full_window = o.check == "probe";
  std::filesystem::create_directories(o.out);
  o.dsv4.out = o.out / "dsv4";
  o.fp16.out = o.out / "fp16";
  return o;
}

}  // namespace

int main(int argc, char** argv) {
  auto options = Parse(std::span(argv, static_cast<std::size_t>(argc)));
  if (!options) {
    std::println(stderr, "{}", options.error());
    return 2;
  }
  Status ran;
  {
    Harness harness(*options);
    ran = harness.Run();
    if (auto finished = harness.TearDown(); !finished) {
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
