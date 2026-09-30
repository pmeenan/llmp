// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// M3's speculation harness for Qwen3.8 Flash Next with its MTP drafter
// (docs/plan.md, "Speculative decoding in the core" and the exit's
// "Speculation correctness"; docs/experiments/qwen38-mtp/): one paged node
// (qwen38_runner.h), greedy and sampled speculation, the forced-rejection
// checks and speed. A harness binary: it links the native tokenizer and
// chat renderer (D-088).
//
//   jitllm_qwen38_spec --qwen38-artifact DIR --drafter DIR --tokenizer FILE
//                      --prompts FILE --out DIR
//                      --check greedy|timing|forced|swap|sampled-plain|sampled-spec
//                      [--reference FILE] [--tokens N] [--context N]
//                      [--graphs on|off] [--draft N] [--draft-vocab N]
//                      [--adaptive-depth on|off]
//                      [--runtime-prefill on|off] [--profile-decode on|off]
//                      [--repeats N] [--margin B] [--seeds N] [--sampled FILE]
//                      [--only ID] [--poll-us N] [--window P]
//                      [--fp16-artifact DIR --fp16-tokens FILE --fp16-expect SHA256]
//
// --window P (0 to 1; default 0, off): TensorFold's adaptive window, the
// first draft always verified and each later one only while the drafter's
// probability for it is at least P (the checks' control runs keep every
// draft).
// --runtime-prefill on splits before the assistant opening and uses the
// runtime's tiled chunks, so draft-depth trials start from the route's
// state. --profile-decode on bounds cudaProfilerAPI capture to speculation.
// --check timing runs greedy speculation with argmax copies only, reporting
// acceptance and costs without the plain/teacher-forced correctness runs.
// Greedy/timing results include the actual step/depth trajectory and mean
// draft/verify costs by requested depth, excluding incomplete tail steps.
//
// Prompts are the fixed set's (docs/experiments/fast-swap/prompts.json),
// rendered by the native Qwen3.8 renderer with the template's defaults
// (thinking on, as the oracle's /tokenize rendered them) and tokenized by
// the native tokenizer from the checkpoint's tokenizer.json; with
// --reference (the oracle's recorded greedy reference, beside the prompts),
// each chat prompt's IDs must equal the oracle's.
//
// The default speculation is the fast graph's batched verify, so its rows
// need not equal one-row decode steps bit for bit (D-092's exact mode is
// DeepSeek's alone); the checks are the M3 exit's as amended (the owner,
// 2026-09-28, D-085's speed before bit exactness):
// - greedy: for the decode prompts (`prose`, `code`) and the chat prompts,
//   plain greedy decoding (one-row chunks, decode graphs) against greedy
//   speculation (the drafter's passes, a verify, the commit). The
//   speculative runs repeat each other bit for bit; every speculative token
//   is the plain engine's argmax on its own prefix (the plain engine
//   teacher-forced on the speculative tokens) or within --margin of it (a
//   near-tie); the prefill with the drafter's injection gives the plain
//   prefill's logits bit for bit. Decode speed, acceptance (accepted ÷
//   drafted) and tokens a verify, per prompt, the speculative run
//   --repeats times. Each generation is a request (D-093).
// - forced (the first chat prompt, or --only's): rejections forced at chosen
//   draft positions (all rejected, one and two accepted, and some as
//   drafted). After every step's commit the
//   whole state (every target state tensor, the drafter's caches and the
//   streams rows the next draft reads) is hashed; a control that ran the
//   same steps with different tokens after the accepted ones (the same
//   rows, so the same kernels) must hash the same at every step: nothing a
//   rejected row wrote survives. The tokens are held to the near-tie rule.
//   Each near-tie report also gives the verify's own noise (verify_noise):
//   at each step, how far the verify row that gave the token moves the
//   plain engine's top-two margin (docs/experiments/dsv4-decode/, "The
//   bound, going forward").
// - swap: rollback composes with swap. The forced run (the control, never
//   swapped), then the same run again stopped after a step that rejected
//   rows, its commit and restore still pending; Qwen3.8 (state and
//   weights) swapped out for the FP16 fixture (whose logits must hash to
//   --fp16-expect) and back, the pending commit run after the swap, and
//   speculation continued: every step's state, token and logits equal the
//   control's, and the drafts and verifies after the swap replay graphs
//   captured before it (D-090's pinned places).
// - sampled-plain, sampled-spec: seeded sampling (temperature 1), plain or
//   speculative (execution/sampling.h VerifyDraft, the greedy drafter's
//   q = δ), for 4 prompts, --seeds seeds and the first 8 generated tokens;
//   with --sampled FILE (the other mode's counts), the total-variation
//   distance over each prompt's 16 most frequent tokens plus "other"
//   (bound 0.1).
//
// Exit 1 on any failed check; spec.json in --out has every number.

#include <cuda.h>

#include <algorithm>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
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
#include "chat/chat.h"
#include "engine_names.h"
#include "execution/adaptive_depth.h"
#include "execution/sampling.h"
#include "fp16_runner.h"
#include "model/qwen38.h"
#include "paged_node.h"
#include "runtime/prefill.h"
#include "scheduler/scheduler.h"
#include "tokenizer/hf.h"
#include "tokenizer/tokenizer.h"

namespace {

namespace ts = jitllm::test_support;
namespace jb = jitllm::benchmarks;
namespace md = jitllm::model;
namespace ex = jitllm::execution;
using jitllm::base::Bytes;
using Clock = std::chrono::steady_clock;
using Status = ts::Status;

constexpr int kQwen = 0;  // owner and stream
constexpr int kFp16 = 1;  // the swap check's B
constexpr std::uint32_t kSampledTokens = 8;
constexpr std::size_t kHistogramTop = 16;
constexpr double kTvBound = 0.1;

std::unexpected<std::string> Error(std::string what) { return std::unexpected(std::move(what)); }

double Seconds(Clock::duration d) { return std::chrono::duration<double>(d).count(); }

Status ProfileDecode(bool start) {
  // The SDK exports the profiler entry points through the driver API but
  // does not include the optional profiler header.
  using Profiler = CUresult(CUDAAPI*)();
  void* address = nullptr;
  const char* name = start ? "cuProfilerStart" : "cuProfilerStop";
  const CUresult lookup =
      cuGetProcAddress(name, &address, 4000, CU_GET_PROC_ADDRESS_DEFAULT, nullptr);
  if (lookup != CUDA_SUCCESS || address == nullptr) {
    return Error(std::format("looking up {}: {}", name, static_cast<int>(lookup)));
  }
  if (const CUresult result = reinterpret_cast<Profiler>(address)(); result != CUDA_SUCCESS) {
    const char* message = nullptr;
    (void)cuGetErrorString(result, &message);
    return Error(std::format("{}: {}", name, message == nullptr ? "CUDA error" : message));
  }
  return {};
}

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
std::string LogitsDigest(const std::vector<std::vector<float>>& rows) {
  if (rows.empty()) {
    return {};
  }
  jitllm::base::Sha256 hash;
  for (const std::vector<float>& row : rows) {
    hash.Update(std::as_bytes(std::span(row)));
  }
  return jitllm::base::ToHex(hash.Finish());
}

// FNV-1a over 64-bit words (and the tail's bytes).
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

std::int32_t Argmax(std::span<const float> row) {
  std::size_t best = 0;
  for (std::size_t i = 1; i < row.size(); ++i) {
    if (row[i] > row[best]) {
      best = i;
    }
  }
  return static_cast<std::int32_t>(best);
}

struct Options {
  jb::Qwen38Options qwen;
  jb::Fp16Options fp16;     // the swap check's B
  std::string fp16_expect;  // its logits' SHA-256
  std::filesystem::path tokenizer;
  std::filesystem::path out;
  std::filesystem::path prompts;
  std::filesystem::path reference;
  std::string check;
  std::uint32_t tokens = 256;
  std::uint32_t repeats = 3;
  // The near-tie bound (logit units): a speculative token may differ from
  // the plain engine's argmax on its prefix only where that argmax leads it
  // by less (docs/experiments/qwen38-mtp/: jitLLM's own kernel noise,
  // qwen38-native's bound 1).
  double margin = 1.0;
  std::uint32_t seeds = 256;
  // TensorFold's adaptive window (0: off, every draft verified): the first
  // draft always, each later one only while the drafter's probability for
  // it is at least this (its own default 0.3).
  double window = 0.0;
  bool adaptive_depth = false;
  bool runtime_prefill = false;
  bool profile_decode = false;
  std::filesystem::path sampled;
  std::string only;
  std::optional<std::uint32_t> poll_us;
};

struct Prompt {
  std::string id;
  std::vector<std::int32_t> ids;
  std::uint32_t stable_boundary = 0;
};

// One speculative step's record.
struct Step {
  std::uint32_t depth = 0;   // passes actually run, before output/verify truncation
  std::uint32_t pos = 0;     // the anchor's position
  std::uint32_t rows = 0;    // the verify's rows
  std::uint32_t kept = 0;    // rows kept: the anchor and the accepted drafts
  std::int32_t forced = -1;  // the draft position forced wrong, or -1
  std::int32_t next = -1;    // the token after the kept rows (the verify's own)
  double draft_seconds = 0;
  double verify_seconds = 0;
  std::vector<std::int32_t> drafts;
  std::vector<std::uint64_t> state;  // fingerprints after the commit (forced checks)
};

// A generation: the tokens after the prompt (the first from the prefill)
// and the logits each was chosen from (the first's: the prefill's last row).
struct Generation {
  // NOLINTNEXTLINE(readability-redundant-member-init): aggregate resets need explicit construction
  ex::AdaptiveDepth depth{};
  std::vector<std::int32_t> tokens;
  std::vector<std::vector<float>> logits;
  std::vector<Step> steps;
  std::uint64_t drafted = 0;
  std::uint64_t accepted = 0;
  std::uint64_t verifies = 0;
  double decode_seconds = 0;  // after the first token
  double draft_seconds = 0;
  double verify_seconds = 0;
};

// How a speculative run chooses its drafts.
struct Forcing {
  // Per step: the draft position made wrong (-1: none).
  // NOLINTNEXTLINE(readability-redundant-member-init): designated initializers name it
  std::function<std::int32_t(std::size_t step, std::uint32_t pos, std::uint32_t rows)> wrong = {};
  // A control: per step, the other run's drafts, with its forced one made
  // wrong differently (the same rows; different tokens after the kept).
  const std::vector<Step>* control = nullptr;
  // Own repeats replay only the depth schedule; fresh drafts stay visible.
  const std::vector<Step>* depths = nullptr;
  bool fingerprint = false;
  // Keep every token's logits (a check's); otherwise a greedy run reads its
  // verdicts' argmaxes alone, as a greedy client would (a sampler always
  // reads the logits).
  bool logits = false;
  // Stop after this many steps, whatever the tokens left.
  std::size_t max_steps = SIZE_MAX;
  // Run the last step's commit before returning. Without, the last step's
  // commit and restore stay pending (and its state is not fingerprinted):
  // the swap check swaps with them owed.
  bool settle = true;
  const ex::AdaptiveDepth* initial_depth = nullptr;
};

class Harness {
 public:
  explicit Harness(const Options& options)
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
        qwen_(node_, o_.qwen, kQwen, kQwen),
        fp16_(node_, o_.fp16, kFp16, kFp16, nullptr, record_) {}

  Status Run();
  Status TearDown() {
    std::vector<ts::PagedModel*> models = {&qwen_};
    if (with_fp16()) {
      models.push_back(&fp16_);
    }
    return node_.TearDown(models);
  }

 private:
  bool with_fp16() const { return !o_.fp16.artifact.empty(); }
  Status Tokenize();
  Status InRequest(std::string_view what, const std::function<Status()>& body);
  Status Prefill(const Prompt& prompt, bool inject, std::vector<float>& last);
  Status Plain(const Prompt& prompt, std::uint32_t count, Generation& out);
  Status Teacher(const Prompt& prompt, std::span<const std::int32_t> tokens, Generation& out);
  // The near-tie rule on `spec`'s tokens, and the verify's own noise from
  // the logits of `rows` (a run whose tokens equal `spec`'s and that kept
  // its logits; `spec` itself by default).
  Status NearTies(const Prompt& prompt, const Generation& plain, const Generation& spec,
                  const Generation* rows = nullptr);
  Status Speculate(const Prompt& prompt, std::uint32_t count, const std::vector<float>& first,
                   const Forcing& forcing, Generation& out,
                   const ex::SamplingParams* sampling = nullptr, std::uint64_t seed = 0);
  Status Judge(Step& step, const std::vector<std::int32_t>& argmax,
               const std::vector<float>& logits, std::uint32_t& pos,
               std::vector<std::int32_t>& history, const ex::SamplingParams* sampling,
               std::uint64_t seed, std::vector<ex::SamplingCandidate>& scratch, Generation& out);
  Status Fingerprints(const Step& step, std::vector<std::uint64_t>& out);
  Status Greedy();
  Status Forced();
  Status Swap();
  Status SwapOut();
  Status SwapIn();
  Status Sampled(bool speculative);
  Status Write();

  const Options& o_;
  std::string record_;
  MemorySampler memory_;
  std::uint64_t available_before_ = MemAvailable();
  ts::PagedNode node_;
  jb::Qwen38Runner qwen_;
  jb::Fp16Runner fp16_;
  std::unique_ptr<jitllm::tokenizer::Tokenizer> tokenizer_;
  std::vector<Prompt> decode_;
  std::vector<Prompt> chat_;
  std::vector<std::string> results_;  // JSON objects
  std::vector<std::string> problems_;
  double load_seconds_ = 0;
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
  auto json = ReadFile(o_.tokenizer);
  if (!json) {
    return std::unexpected(json.error());
  }
  auto spec = jitllm::tokenizer::ReadHfTokenizer(*json);
  if (!spec) {
    return Error(spec.error().ToString());
  }
  auto tokenizer = jitllm::tokenizer::Tokenizer::Create(std::move(*spec));
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
    if (!id || !messages || !messages->is_array()) {
      return Error("a prompt without an id or messages");
    }
    p.id = std::string(id->string());
    jitllm::chat::Conversation c;  // the template's defaults: thinking on
    for (std::size_t i = 0; i < messages->size(); ++i) {
      const auto content = messages->at(i).find("content");
      c.messages.push_back({.role = jitllm::chat::Role::kUser,
                            .content = std::string(content ? content->string() : ""),
                            .reasoning_content = std::nullopt,
                            .tool_calls = {}});
    }
    auto rendered = jitllm::chat::RenderQwen38(c);
    if (!rendered) {
      return Error(rendered.error().ToString());
    }
    std::vector<jitllm::tokenizer::TokenId> ids;
    std::vector<std::size_t> span_tokens;
    if (auto r = tokenizer_->EncodeMarked(rendered->text, rendered->specials, {}, ids,
                                          o_.runtime_prefill ? &span_tokens : nullptr);
        !r) {
      return Error(r.error().ToString());
    }
    if (o_.runtime_prefill) {
      for (const jitllm::chat::Boundary& boundary : rendered->boundaries) {
        if (boundary.kind != jitllm::chat::BoundaryKind::kGenerationPrompt) {
          continue;
        }
        for (std::size_t i = 0; i < rendered->specials.size(); ++i) {
          if (rendered->specials[i].offset == boundary.offset && span_tokens[i] < ids.size()) {
            p.stable_boundary = static_cast<std::uint32_t>(span_tokens[i]);
            break;
          }
        }
      }
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
      into->push_back(std::move(*p));
    }
  }
  if (o_.reference.empty()) {
    return {};
  }
  // The renderer and tokenizer against the oracle's own IDs.
  auto ref = ReadFile(o_.reference);
  if (!ref) {
    return std::unexpected(ref.error());
  }
  auto rdoc = jitllm::base::json::Parse(*ref);
  if (!rdoc) {
    return Error(std::format("{} is not JSON", o_.reference.string()));
  }
  const auto prompts = rdoc->root().find("prompts");
  std::size_t checked = 0;
  for (std::size_t i = 0; prompts && i < prompts->size(); ++i) {
    const auto entry = prompts->at(i);
    const auto id = entry.find("id");
    const auto ids = entry.find("prompt_token_ids");
    if (!id || !ids) {
      continue;
    }
    for (const Prompt& p : chat_) {
      if (p.id != id->string()) {
        continue;
      }
      std::vector<std::int32_t> want;
      want.reserve(ids->size());
      for (std::size_t k = 0; k < ids->size(); ++k) {
        want.push_back(static_cast<std::int32_t>(ids->at(k).int64().value_or(-1)));
      }
      ++checked;
      if (want != p.ids) {
        problems_.push_back(
            std::format("prompt {}: the native IDs differ from the oracle's ({} against {} tokens)",
                        p.id, p.ids.size(), want.size()));
      }
    }
  }
  std::println("prompts: {} decode, {} chat; {} chat prompts' IDs checked against the oracle's",
               decode_.size(), chat_.size(), checked);
  return {};
}

// ------------------------------------------------------------------ runs

Status Harness::InRequest(std::string_view what, const std::function<Status()>& body) {
  return node_.WithRequest(qwen_.stream(), qwen_.everything(), what, body);
}

Status Harness::Prefill(const Prompt& prompt, bool inject, std::vector<float>& last) {
  if (auto r = qwen_.Clear(); !r) {
    return r;
  }
  const std::uint32_t rows = o_.qwen.max_rows;
  if (o_.runtime_prefill) {
    const auto chunk = [&](std::uint32_t at, std::uint32_t n) {
      return qwen_.Chunk(std::span(prompt.ids).first(at + n), at, last, inject);
    };
    std::uint32_t at = 0;
    if (prompt.stable_boundary != 0) {
      auto run = jitllm::runtime::RunPrefillChunks(0, prompt.stable_boundary, rows, chunk, {});
      if (!run) {
        return Error(std::format("{}'s prefill: {}", prompt.id, run.error()));
      }
      at = run->end;
    }
    auto run = jitllm::runtime::RunPrefillChunks(at, static_cast<std::uint32_t>(prompt.ids.size()),
                                                 rows, chunk, {});
    return run ? Status{} : Error(std::format("{}'s prefill: {}", prompt.id, run.error()));
  }
  for (std::uint32_t at = 0; at < prompt.ids.size(); at += rows) {
    const auto n = static_cast<std::uint32_t>(std::min<std::size_t>(rows, prompt.ids.size() - at));
    if (auto r = qwen_.Chunk(std::span(prompt.ids).first(at + n), at, last, inject); !r) {
      return Error(std::format("{}'s prefill at {}: {}", prompt.id, at, r.error()));
    }
  }
  return {};
}

Status Harness::Plain(const Prompt& prompt, std::uint32_t count, Generation& out) {
  std::vector<float> last;
  if (auto r = Prefill(prompt, false, last); !r) {
    return r;
  }
  out.tokens = {Argmax(last)};
  out.logits = {last};
  std::vector<std::int32_t> history = prompt.ids;
  const auto start = Clock::now();
  while (out.tokens.size() < count) {
    history.push_back(out.tokens.back());
    std::vector<float> row;
    if (auto r = qwen_.Chunk(history, static_cast<std::uint32_t>(history.size() - 1), row); !r) {
      return r;
    }
    out.tokens.push_back(Argmax(row));
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
  std::vector<std::int32_t> history = prompt.ids;
  for (std::size_t i = 1; i < tokens.size(); ++i) {
    history.push_back(tokens[i - 1]);
    std::vector<float> row;
    if (auto r = qwen_.Chunk(history, static_cast<std::uint32_t>(history.size() - 1), row); !r) {
      return r;
    }
    out.logits.push_back(std::move(row));
  }
  return {};
}

Status Harness::NearTies(const Prompt& prompt, const Generation& plain, const Generation& spec,
                         const Generation* rows) {
  Generation forced;
  if (auto r = InRequest(prompt.id, [&] { return Teacher(prompt, spec.tokens, forced); }); !r) {
    return r;
  }
  if (rows == nullptr || rows->tokens != spec.tokens) {
    rows = &spec;
  }
  std::size_t equal = 0;
  std::size_t violations = 0;
  double largest = 0;
  std::string ties;
  // The verify's own noise (dsv4-decode's "The bound, going forward"): at
  // each speculative step after the prefill's, how far the verify row that
  // gave the token moves the plain engine's top-two margin (its logit
  // difference between the plain row's two best tokens, against the plain
  // row's).
  std::vector<double> moves;
  for (std::size_t i = 0; i < spec.tokens.size(); ++i) {
    const std::span<const float> row = forced.logits[i];
    const std::int32_t best = Argmax(row);
    const std::int32_t got = spec.tokens[i];
    if (i > 0 && i < rows->logits.size() && rows->logits[i].size() == row.size()) {
      const auto at = [](std::int32_t t) { return static_cast<std::size_t>(t); };
      std::size_t second = best == 0 ? 1 : 0;
      for (std::size_t j = 0; j < row.size(); ++j) {
        if (std::cmp_not_equal(j, best) && row[j] > row[second]) {
          second = j;
        }
      }
      const std::vector<float>& v = rows->logits[i];
      moves.push_back(std::abs((static_cast<double>(v[at(best)]) - v[second]) -
                               (static_cast<double>(row[at(best)]) - row[second])));
    }
    if (best == got) {
      ++equal;
      continue;
    }
    const double margin = static_cast<double>(row[static_cast<std::size_t>(best)]) -
                          static_cast<double>(row[static_cast<std::size_t>(got)]);
    const bool tie = margin < o_.margin;
    violations += tie ? 0 : 1;
    largest = std::max(largest, margin);
    ties += std::format(R"({}{{"step":{},"plain":{},"spec":{},"margin":{:.4f},"near_tie":{}}})",
                        ties.empty() ? "" : ",", i, best, got, margin, tie ? "true" : "false");
  }
  std::size_t prefix = 0;
  while (prefix < plain.tokens.size() && prefix < spec.tokens.size() &&
         plain.tokens[prefix] == spec.tokens[prefix]) {
    ++prefix;
  }
  if (violations != 0) {
    problems_.push_back(
        std::format("{}: {} speculative tokens are not the plain engine's argmax on their "
                    "prefix, nor within {} of it",
                    prompt.id, violations, o_.margin));
  }
  std::ranges::sort(moves);
  const auto quantile = [&](double q) {
    return moves.empty()
               ? 0.0
               : moves[std::min(moves.size() - 1,
                                static_cast<std::size_t>(
                                    std::ceil(q * static_cast<double>(moves.size())) - 1.0))];
  };
  const std::string noise = std::format(
      R"("verify_noise":{{"steps":{},"median":{:.4f},"p95":{:.4f},"p99":{:.4f},"max":{:.4f}}})",
      moves.size(), quantile(0.5), quantile(0.95), quantile(0.99),
      moves.empty() ? 0.0 : moves.back());
  std::println(
      "{}: {} of {} speculative tokens the plain engine's argmax on their prefix, {} near-ties "
      "(largest margin {:.4f}), {} violations (margin {}); free-running prefix {} of {}; {}",
      prompt.id, equal, spec.tokens.size(), spec.tokens.size() - equal - violations, largest,
      violations, o_.margin, prefix, plain.tokens.size(), noise);
  results_.push_back(std::format(
      R"({{"check":"near_ties","prompt":"{}","tokens":{},"equal":{},"violations":{},)"
      R"("margin":{},"largest":{:.4f},"free_running_prefix":{},{},"differing":[{}]}})",
      prompt.id, spec.tokens.size(), equal, violations, o_.margin, largest, prefix, noise, ties));
  return {};
}

Status Harness::Fingerprints(const Step& step, std::vector<std::uint64_t>& out) {
  std::vector<std::byte> target;
  std::vector<std::byte> drafter;
  if (auto r = qwen_.ReadState(target, drafter); !r) {
    return r;
  }
  out.clear();
  for (const md::Qwen38StateTensor& t : qwen_.state_layout().tensors) {
    out.push_back(Fingerprint(std::span(target).subspan(t.offset, t.bytes)));
  }
  // The drafter's caches, and the streams rows the next draft reads (rows
  // 1 .. kept; the verify's later rows are never read).
  const md::Qwen38MtpState& m = qwen_.mtp_state();
  out.push_back(Fingerprint(std::span(drafter).subspan(m.k, m.hidden - m.k)));
  const std::uint64_t row = std::uint64_t{md::Qwen38Flash().hc_width()} * sizeof(float);
  out.push_back(Fingerprint(std::span(drafter).subspan(m.hidden + row, step.kept * row)));
  return {};
}

// A verify's verdict: accept drafts while the target agrees (greedy: the
// rows' argmaxes) or its speculative sampling accepts them; the first
// disagreement, or the row after the last draft, gives the next token.
Status Harness::Judge(Step& step, const std::vector<std::int32_t>& argmax,
                      const std::vector<float>& logits, std::uint32_t& pos,
                      std::vector<std::int32_t>& history, const ex::SamplingParams* sampling,
                      std::uint64_t seed, std::vector<ex::SamplingCandidate>& scratch,
                      Generation& out) {
  const std::uint32_t vocab = qwen_.vocab();
  const std::uint32_t rows = step.rows;
  const auto row = [&](std::uint32_t i) {
    return std::span<const float>(logits).subspan(std::size_t{i} * vocab, vocab);
  };
  std::uint32_t m = 0;
  std::int32_t next = -1;
  for (; m < rows - 1; ++m) {
    if (sampling == nullptr) {
      const std::int32_t want = argmax[m];
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
      next = argmax[m];
    } else {
      auto t = ex::Sample(row(m), *sampling, {.seed = seed, .stream = 0, .position = pos + m + 1},
                          scratch);
      if (!t) {
        return Error("a sample");
      }
      next = *t;
    }
  }
  step.next = next;
  step.kept = m + 1;
  if (auto r = qwen_.Accept(step.kept); !r) {
    return r;
  }
  for (std::uint32_t i = 0; i <= m; ++i) {
    const std::int32_t token = i < m ? step.drafts[i] : next;
    out.tokens.push_back(token);
    if (!logits.empty()) {
      out.logits.emplace_back(row(i).begin(), row(i).end());
    }
    history.push_back(token);
  }
  out.drafted += rows - 1;
  out.accepted += m;
  ++out.verifies;
  pos += m + 1;
  return {};
}

Status Harness::Speculate(const Prompt& prompt, std::uint32_t count,
                          const std::vector<float>& first, const Forcing& forcing, Generation& out,
                          const ex::SamplingParams* sampling, std::uint64_t seed) {
  std::vector<ex::SamplingCandidate> scratch;
  auto pos = static_cast<std::uint32_t>(prompt.ids.size());
  const std::uint32_t vocab = qwen_.vocab();
  const std::uint32_t k = qwen_.draft_rows();
  ex::AdaptiveDepth depth =
      forcing.initial_depth != nullptr ? *forcing.initial_depth : ex::AdaptiveDepth(k);
  const auto choose = [&](std::span<const float> row, std::uint32_t at) -> std::int32_t {
    if (sampling == nullptr) {
      return Argmax(row);
    }
    auto t = ex::Sample(row, *sampling, {.seed = seed, .stream = 0, .position = at}, scratch);
    return t.value_or(-1);
  };
  out.tokens = {choose(first, pos)};
  if (forcing.logits) {
    out.logits = {first};
  }
  // Every token through the anchor (at `pos`, not yet in the cache).
  std::vector<std::int32_t> history = prompt.ids;
  history.push_back(out.tokens.back());
  if (o_.profile_decode) {
    if (auto result = ProfileDecode(true); !result) {
      return result;
    }
  }
  const auto start = Clock::now();
  while (out.tokens.size() < count && out.steps.size() < forcing.max_steps) {
    if (pos + k > o_.qwen.context) {
      return Error("the drafts would pass the context");
    }
    Step step;
    step.pos = pos;
    const std::size_t index = out.steps.size();
    const auto left = static_cast<std::uint32_t>(count - out.tokens.size());
    const auto drafting = Clock::now();
    std::vector<float> probabilities;
    auto passes = k;
    if (forcing.depths != nullptr && index < forcing.depths->size()) {
      passes = (*forcing.depths)[index].depth;
    } else if (forcing.control != nullptr && index < forcing.control->size()) {
      passes = (*forcing.control)[index].depth;
    } else if (o_.adaptive_depth && sampling == nullptr) {
      passes = depth.Choose();
    }
    step.depth = passes;
    if (auto result =
            qwen_.Draft(history, step.drafts, o_.window > 0.0 ? &probabilities : nullptr, passes);
        !result) {
      return result;
    }
    step.draft_seconds = Seconds(Clock::now() - drafting);
    out.draft_seconds += step.draft_seconds;
    if (o_.window > 0.0 && forcing.control == nullptr) {
      // The adaptive window: a later draft is verified only while the
      // drafter is confident of it (a verify row costs ~5 ms, a plain step
      // ~37).
      std::size_t keep = 1;
      while (keep < step.drafts.size() && keep < probabilities.size() &&
             probabilities[keep] >= o_.window) {
        ++keep;
      }
      step.drafts.resize(std::min(keep, step.drafts.size()));
    }
    if (forcing.control != nullptr) {
      if (index >= forcing.control->size()) {
        // Kept fewer rows somewhere than the run it follows: the checks
        // report the first step that differs.
        break;
      }
      const Step& other = (*forcing.control)[index];
      step.drafts = other.drafts;
      if (other.forced >= 0 && std::cmp_less_equal(other.kept, other.forced + 1)) {
        // Wrong differently: the same rows, other tokens after the kept. A
        // forced draft the verify kept (it was right after all) stays, and
        // the other wrong token is never the verify's own at that row (the
        // token after the run's kept rows), which it would keep.
        auto& d = step.drafts[static_cast<std::size_t>(other.forced)];
        const auto after = [&](std::int32_t id) {
          return static_cast<std::int32_t>((static_cast<std::uint32_t>(id) + 1) % vocab);
        };
        d = after(d);
        if (d == other.next) {
          d = after(d);
        }
      }
    }
    // The verify: the anchor and its drafts, within the tokens left.
    const auto rows = std::min<std::uint32_t>(
        {static_cast<std::uint32_t>(step.drafts.size()) + 1, left, o_.qwen.context - pos});
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
    std::vector<std::int32_t> input = history;
    input.insert(input.end(), step.drafts.begin(), step.drafts.end());
    std::vector<std::int32_t> argmax;
    std::vector<float> logits;
    const bool read_logits = forcing.logits || sampling != nullptr;
    const auto verifying = Clock::now();
    if (auto r = qwen_.Verify(input, pos, argmax, read_logits ? &logits : nullptr); !r) {
      return r;
    }
    step.verify_seconds = Seconds(Clock::now() - verifying);
    out.verify_seconds += step.verify_seconds;
    if (auto r = Judge(step, argmax, logits, pos, history, sampling, seed, scratch, out); !r) {
      return r;
    }
    if (o_.adaptive_depth && sampling == nullptr && forcing.control == nullptr) {
      depth.Observe(passes, step.kept, rows == passes + 1 && o_.window == 0.0);
    }
    // Unsettled, the last step's commit stays owed (reading the state would
    // run it).
    const bool last = out.tokens.size() >= count || out.steps.size() + 1 >= forcing.max_steps;
    if (forcing.fingerprint && (forcing.settle || !last)) {
      if (auto r = Fingerprints(step, step.state); !r) {
        return r;
      }
    }
    out.steps.push_back(std::move(step));
  }
  if (forcing.settle) {
    if (auto r = qwen_.Rollback(); !r) {
      return r;
    }
  }
  out.decode_seconds = Seconds(Clock::now() - start);
  if (o_.profile_decode) {
    if (auto result = ProfileDecode(false); !result) {
      return result;
    }
  }
  out.depth = depth;
  // The last verify's tokens may run past `count` (a run stopped by
  // max_steps has fewer, and keeps them all).
  if (out.tokens.size() > count) {
    out.tokens.resize(count);
  }
  if (out.logits.size() > count) {
    out.logits.resize(count);
  }
  return {};
}

// ------------------------------------------------------------------ checks

Status Harness::Greedy() {
  const bool timing = o_.check == "timing";
  std::vector<Prompt> prompts = decode_;
  prompts.insert(prompts.end(), chat_.begin(), chat_.end());
  if (!o_.only.empty()) {
    std::erase_if(prompts, [&](const Prompt& p) { return p.id != o_.only; });
  }
  for (const Prompt& prompt : prompts) {
    const bool decode =
        std::ranges::any_of(decode_, [&](const Prompt& p) { return p.id == prompt.id; });
    const std::uint32_t count = decode ? o_.tokens : std::min<std::uint32_t>(o_.tokens, 32);
    Generation plain;
    if (!timing) {
      if (auto r = InRequest("a plain generation", [&] { return Plain(prompt, count, plain); });
          !r) {
        return r;
      }
    }
    const std::size_t repeats = decode ? o_.repeats : 1;
    std::vector<double> rates;
    Generation spec;
    Generation first_run;
    for (std::size_t r = 0; r < repeats; ++r) {
      if (r == 1) {
        first_run = std::move(spec);
      }
      spec = {};
      // The first two runs keep every logit (their repeat checked bit for
      // bit); later ones read the verdicts' argmaxes alone, as a greedy
      // client does (their tokens checked).
      const bool keep = !timing && r < 2;
      std::vector<float> first;
      if (auto run = InRequest(
              "a speculative generation",
              [&]() -> Status {
                if (auto p = Prefill(prompt, true, first); !p) {
                  return p;
                }
                return Speculate(prompt, count, first,
                                 {.depths = o_.adaptive_depth && r > 0 ? &first_run.steps : nullptr,
                                  .logits = keep},
                                 spec);
              });
          !run) {
        return run;
      }
      if (!timing && !SameBits(first, plain.logits.front())) {
        problems_.push_back(std::format(
            "{}: the prefill with the injection differs from the plain prefill", prompt.id));
      }
      if (r > 0) {
        bool same = first_run.tokens == spec.tokens;
        for (std::size_t i = 0; same && keep && i < spec.logits.size(); ++i) {
          same = i < first_run.logits.size() && SameBits(first_run.logits[i], spec.logits[i]);
        }
        if (!same) {
          problems_.push_back(std::format(
              "{}: speculative run {} does not repeat the first bit for bit", prompt.id, r + 1));
        }
      }
      rates.push_back(static_cast<double>(count - 1) / spec.decode_seconds);
    }
    // The verify's noise from the first run's rows (its tokens are checked
    // equal to the last's above; later runs keep no logits).
    if (!timing) {
      if (auto r = NearTies(prompt, plain, spec, repeats > 1 ? &first_run : &spec); !r) {
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
    const double plain_rate = timing ? 0.0 : static_cast<double>(count - 1) / plain.decode_seconds;
    const double acceptance =
        spec.drafted > 0 ? static_cast<double>(spec.accepted) / static_cast<double>(spec.drafted)
                         : 0.0;
    std::string rates_json;
    for (const double rate : rates) {
      rates_json += std::format("{}{:.3f}", rates_json.empty() ? "" : ",", rate);
    }
    // Acceptance by draft position: how often draft i was accepted.
    std::vector<std::uint64_t> by_position(qwen_.draft_rows(), 0);
    std::vector<std::uint64_t> offered(qwen_.draft_rows(), 0);
    std::map<std::uint32_t, std::uint64_t> depths;
    std::map<std::uint32_t, std::uint64_t> verify_rows;
    struct DepthCost {
      std::uint64_t steps = 0;
      double draft_seconds = 0;
      double verify_seconds = 0;
    };
    std::map<std::uint32_t, DepthCost> costs;
    // [anchor position, requested depth, verify rows, kept rows]. Timings
    // are reported separately and never change the deterministic policy.
    std::string trace;
    for (const Step& s : spec.steps) {
      ++depths[s.depth];
      ++verify_rows[s.rows];
      trace +=
          std::format("{}[{},{},{},{}]", trace.empty() ? "" : ",", s.pos, s.depth, s.rows, s.kept);
      // Partial tail verifies cannot calibrate a full-depth step.
      if (s.rows == s.depth + 1) {
        auto& cost = costs[s.depth];
        ++cost.steps;
        cost.draft_seconds += s.draft_seconds;
        cost.verify_seconds += s.verify_seconds;
      }
      for (std::uint32_t i = 0; i + 1 < s.rows; ++i) {
        ++offered[i];
        by_position[i] += i + 1 < s.kept ? 1 : 0;
      }
    }
    std::string positions_json;
    for (std::size_t i = 0; i < by_position.size(); ++i) {
      positions_json += std::format(
          "{}{:.4f}", positions_json.empty() ? "" : ",",
          offered[i] > 0 ? static_cast<double>(by_position[i]) / static_cast<double>(offered[i])
                         : 0.0);
    }
    const double per_step_ms = 1000.0 / static_cast<double>(spec.verifies);
    std::string depths_json;
    for (const auto& [depth, steps] : depths) {
      depths_json += std::format("{}\"{}\":{}", depths_json.empty() ? "" : ",", depth, steps);
    }
    std::string rows_json;
    for (const auto& [rows, steps] : verify_rows) {
      rows_json += std::format("{}\"{}\":{}", rows_json.empty() ? "" : ",", rows, steps);
    }
    std::string costs_json;
    for (const auto& [depth, cost] : costs) {
      const double factor = 1000.0 / static_cast<double>(cost.steps);
      costs_json += std::format(R"({}"{}":{{"steps":{},"draft":{:.3f},"verify":{:.3f}}})",
                                costs_json.empty() ? "" : ",", depth, cost.steps,
                                factor * cost.draft_seconds, factor * cost.verify_seconds);
    }
    std::println(
        "{}: {} prompt tokens, {} generated; plain {:.2f} tok/s, speculative [{}] tok/s; "
        "acceptance {:.3f} ({} of {}; by position [{}]), {:.2f} tokens a verify; a step: draft "
        "{:.2f} ms, verify {:.2f} ms, of {:.2f} ms",
        prompt.id, prompt.ids.size(), count, plain_rate, rates_json, acceptance, spec.accepted,
        spec.drafted, positions_json,
        static_cast<double>(count - 1) / static_cast<double>(spec.verifies),
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
        R"({{"check":"{}","prompt":"{}","prompt_tokens":{},"stable_boundary":{},"generated":{},"plain_tok_s":{:.3f},)"
        R"("spec_tok_s":[{}],"drafted":{},"accepted":{},"acceptance":{:.4f},)"
        R"("acceptance_by_position":[{}],"verifies":{},"draft_depths":{{{}}},"verify_rows":{{{}}},)"
        R"("step_trace":[{}],"depth_cost_ms":{{{}}},)"
        R"("step_ms":{{"draft":{:.3f},"verify":{:.3f},"all":{:.3f}}},"text":{},)"
        R"("prompt_ids":[{}],"plain_tokens":[{}],"spec_tokens":[{}],)"
        R"("plain_logits_sha256":"{}","spec_logits_sha256":"{}"}})",
        timing ? "timing" : "greedy", prompt.id, prompt.ids.size(), prompt.stable_boundary, count,
        plain_rate, rates_json, spec.drafted, spec.accepted, acceptance, positions_json,
        spec.verifies, depths_json, rows_json, trace, costs_json, spec.draft_seconds * per_step_ms,
        spec.verify_seconds * per_step_ms, spec.decode_seconds * per_step_ms, escaped,
        ids(prompt.ids), ids(plain.tokens), ids(spec.tokens), LogitsDigest(plain.logits),
        LogitsDigest(first_run.logits.empty() ? spec.logits : first_run.logits)));
  }
  return {};
}

// Forced rejections: all rejected, one and two accepted, and some as
// drafted; against a control that ran the same steps with other tokens
// after the kept ones.
Status Harness::Forced() {
  // The first chat prompt, or --only's (a chat or decode prompt).
  if (chat_.empty() && decode_.empty()) {
    return Error("the rejection check needs a prompt");
  }
  const Prompt* chosen = chat_.empty() ? &decode_.front() : &chat_.front();
  if (!o_.only.empty()) {
    chosen = nullptr;
    for (const std::vector<Prompt>* set : {&chat_, &decode_}) {
      for (const Prompt& p : *set) {
        if (chosen == nullptr && p.id == o_.only) {
          chosen = &p;
        }
      }
    }
    if (chosen == nullptr) {
      return Error(std::format("--only {}: no such prompt", o_.only));
    }
  }
  const Prompt& prompt = *chosen;
  const std::uint32_t count = o_.tokens;
  std::map<std::string, std::uint64_t> covered;
  const auto wrong = [&](std::size_t step, std::uint32_t, std::uint32_t rows) -> std::int32_t {
    if (rows < 2) {
      return -1;
    }
    switch (step % 5) {
      case 0:
        ++covered["all rejected"];
        return 0;
      case 1:
        if (rows < 3) {
          return -1;
        }
        ++covered["one accepted"];
        return 1;
      case 2:
        if (rows < 4) {
          return -1;
        }
        ++covered["two accepted"];
        return 2;
      default:
        return -1;  // as drafted
    }
  };
  Generation plain;
  if (auto r = InRequest("a plain generation", [&] { return Plain(prompt, count, plain); }); !r) {
    return r;
  }
  Generation spec;
  if (auto r = InRequest("a forced run",
                         [&]() -> Status {
                           std::vector<float> first;
                           if (auto p = Prefill(prompt, true, first); !p) {
                             return p;
                           }
                           return Speculate(prompt, count, first,
                                            {.wrong = wrong, .fingerprint = true, .logits = true},
                                            spec);
                         });
      !r) {
    return r;
  }
  if (auto r = NearTies(prompt, plain, spec); !r) {
    return r;
  }
  Generation control;
  if (auto r = InRequest("its control",
                         [&]() -> Status {
                           std::vector<float> first;
                           if (auto p = Prefill(prompt, true, first); !p) {
                             return p;
                           }
                           return Speculate(prompt, count, first,
                                            {.control = &spec.steps, .fingerprint = true}, control);
                         });
      !r) {
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
        const auto ids = [](const std::vector<std::int32_t>& v) {
          std::string text;
          for (const std::int32_t id : v) {
            text += std::format("{}{}", text.empty() ? "" : " ", id);
          }
          return text;
        };
        first_difference = std::format(
            "step {} (pos {}, kept {} of {}, control kept {}; forced {}, drafts [{}] and the "
            "control's [{}])",
            s, a.pos, a.kept, a.rows, b.kept, a.forced, ids(a.drafts), ids(b.drafts));
        for (std::size_t t = 0; t < a.state.size() && t < b.state.size(); ++t) {
          if (a.state[t] != b.state[t]) {
            const auto& tensors = qwen_.state_layout().tensors;
            first_difference +=
                t < tensors.size()
                    ? std::format(": state tensor {} (layer {}, kind {})", t, tensors[t].layer,
                                  static_cast<int>(tensors[t].kind))
                    : std::format(": the drafter's part {}", t - tensors.size());
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
      "forced: {} steps, {} with rejected rows, {} states differing from the control's; "
      "covered {}",
      spec.steps.size(), rejected, differing, covered_json);
  results_.push_back(std::format(
      R"({{"check":"forced","prompt":"{}","generated":{},"steps":{},"rejected_steps":{},)"
      R"("state_differs":{},"covered":{{{}}},"pos_rows_kept_forced":[{}]}})",
      prompt.id, count, spec.steps.size(), rejected, differing, covered_json, steps_json));
  return {};
}

// Qwen3.8 out (its state written back, its weights and the drafter's
// evicted) for the FP16 fixture, which runs once; its logits hashed.
Status Harness::SwapOut() {
  std::vector<jitllm::catalog::ExtentId> out = qwen_.state();
  const std::vector<jitllm::catalog::ExtentId> weights = qwen_.weights();
  out.insert(out.end(), weights.begin(), weights.end());
  ts::SwapReport report;
  if (auto r = node_.Swap(std::move(out), fp16_.everything(), true, report); !r) {
    return r;
  }
  std::vector<float> result;
  if (auto r = fp16_.Evaluate(1, result); !r) {
    return r;
  }
  jitllm::base::Sha256 hash;
  hash.Update(std::as_bytes(std::span(result)));
  const std::string digest = jitllm::base::ToHex(hash.Finish());
  std::println("swap: Qwen3.8 out, B's logits {}", digest);
  if (!o_.fp16_expect.empty() && digest != o_.fp16_expect) {
    problems_.push_back(std::format("swap: B's logits {} differ from {}", digest, o_.fp16_expect));
  }
  return {};
}

// Qwen3.8 back (its state restored, its weights paged in): the n-gram hash
// checked again and every weight and state extent still at its pinned
// place, where the graphs captured before the swap name it (D-090).
Status Harness::SwapIn() {
  ts::SwapReport report;
  if (auto r = node_.Swap(fp16_.weights(), qwen_.everything(), true, report); !r) {
    return r;
  }
  if (auto r = qwen_.ReadPleHash(); !r) {
    return r;
  }
  std::size_t unpinned = 0;
  const std::vector<jitllm::catalog::ExtentId> managed = qwen_.managed_extents();
  if (auto r = node_.Call(
          [&]() -> Status {
            for (const jitllm::catalog::ExtentId extent : managed) {
              unpinned += node_.scheduler().PlacePinned(extent) ? 0 : 1;
            }
            return {};
          },
          "checking places");
      !r) {
    return r;
  }
  if (unpinned != 0) {
    problems_.push_back(
        std::format("swap: {} of Qwen3.8's {} extents are not pinned at their places", unpinned,
                    managed.size()));
  }
  return {};
}

// Rollback composes with swap: the forced run unswapped (the control), then
// the same run stopped after a step that rejected rows, with that step's
// commit and restore still owed; Qwen3.8 out for the FP16 fixture and
// back; the owed commit run after the swap, and speculation continued.
// Every step's state, and every token and its logits, equal the control's.
Status Harness::Swap() {
  if (!with_fp16()) {
    return Error("the swap check needs the FP16 fixture (--fp16-artifact, --fp16-tokens)");
  }
  if (chat_.empty() && decode_.empty()) {
    return Error("the swap check needs a prompt");
  }
  const Prompt& prompt = chat_.empty() ? decode_.front() : chat_.front();
  const std::uint32_t count = o_.tokens;
  // All rejected, one accepted, as drafted (depth 2), in turn.
  const auto wrong = [](std::size_t step, std::uint32_t, std::uint32_t rows) -> std::int32_t {
    return rows < 2 ? -1 : static_cast<std::int32_t>(step % 3);
  };
  Generation control;
  if (auto r = InRequest("the unswapped run",
                         [&]() -> Status {
                           std::vector<float> first;
                           if (auto p = Prefill(prompt, true, first); !p) {
                             return p;
                           }
                           return Speculate(prompt, count, first,
                                            {.wrong = wrong, .fingerprint = true, .logits = true},
                                            control);
                         });
      !r) {
    return r;
  }
  // Half the steps, ending on one that rejected rows.
  std::size_t steps = control.steps.size() / 2;
  while (steps > 1 && control.steps[steps - 1].kept == control.steps[steps - 1].rows) {
    --steps;
  }
  Generation before;
  if (auto r = InRequest("the run before the swap",
                         [&]() -> Status {
                           std::vector<float> first;
                           if (auto p = Prefill(prompt, true, first); !p) {
                             return p;
                           }
                           return Speculate(prompt, count, first,
                                            {.wrong = wrong,
                                             .fingerprint = true,
                                             .logits = true,
                                             .max_steps = steps,
                                             .settle = false},
                                            before);
                         });
      !r) {
    return r;
  }
  if (before.steps.empty() || before.logits.size() != before.tokens.size()) {
    return Error("swap: no step before the swap");
  }
  const bool rejected = before.steps.back().kept < before.steps.back().rows;
  const std::size_t kept_graphs = qwen_.graphs();
  if (auto r = SwapOut(); !r) {
    return r;
  }
  if (auto r = SwapIn(); !r) {
    return r;
  }
  const jb::Dsv4GraphStats verify_before = qwen_.graph_stats();
  const jb::Dsv4GraphStats draft_before = qwen_.draft_stats();
  // Continued from the anchor: the prompt and the tokens before it are the
  // context; the first half's last token and its logits are taken as given.
  Prompt rest = prompt;
  rest.ids.insert(rest.ids.end(), before.tokens.begin(), before.tokens.end() - 1);
  if (Argmax(before.logits.back()) != before.tokens.back()) {
    return Error("swap: the anchor is not its logits' argmax");
  }
  const std::size_t done = before.steps.size();
  Generation after;
  if (auto r = InRequest(
          "the run after the swap",
          [&]() -> Status {
            // The owed commit and restore run first (reading the state runs
            // them): the last step's state, after the swap.
            Step& last = before.steps.back();
            if (auto f = Fingerprints(last, last.state); !f) {
              return f;
            }
            const auto wrong_after = [&](std::size_t step, std::uint32_t pos, std::uint32_t rows) {
              return wrong(step + done, pos, rows);
            };
            const auto remaining = static_cast<std::uint32_t>(count - before.tokens.size() + 1);
            return Speculate(rest, remaining, before.logits.back(),
                             {.wrong = wrong_after,
                              .fingerprint = true,
                              .logits = true,
                              .initial_depth = &before.depth},
                             after);
          });
      !r) {
    return r;
  }
  const jb::Dsv4GraphStats verify_after = qwen_.graph_stats();
  const jb::Dsv4GraphStats draft_after = qwen_.draft_stats();
  Generation swapped = before;
  swapped.tokens.insert(swapped.tokens.end(), after.tokens.begin() + 1, after.tokens.end());
  swapped.logits.insert(swapped.logits.end(), after.logits.begin() + 1, after.logits.end());
  swapped.steps.insert(swapped.steps.end(), after.steps.begin(), after.steps.end());
  std::size_t logits_differ = 0;
  for (std::size_t i = 0; i < control.logits.size() && i < swapped.logits.size(); ++i) {
    logits_differ += SameBits(control.logits[i], swapped.logits[i]) ? 0 : 1;
  }
  std::size_t differing = 0;
  for (std::size_t s = 0; s < control.steps.size() && s < swapped.steps.size(); ++s) {
    const Step& a = control.steps[s];
    const Step& b = swapped.steps[s];
    differing += a.state != b.state || a.pos != b.pos || a.kept != b.kept ? 1 : 0;
  }
  const bool tokens_equal = control.tokens == swapped.tokens;
  if (!tokens_equal || control.logits.size() != swapped.logits.size() || logits_differ != 0) {
    problems_.push_back(std::format(
        "swap: the resumed run's tokens {} the control's; {} of {} tokens' logits differ",
        tokens_equal ? "equal" : "differ from", logits_differ, control.logits.size()));
  }
  if (control.steps.size() != swapped.steps.size() || differing != 0) {
    problems_.push_back(std::format("swap: {} of {} steps' states differ from the unswapped run's",
                                    differing, control.steps.size()));
  }
  if (!rejected) {
    problems_.emplace_back("swap: the step before the swap rejected no rows");
  }
  const std::uint64_t verify_replayed = verify_after.replayed - verify_before.replayed;
  const std::uint64_t draft_replayed = draft_after.replayed - draft_before.replayed;
  const std::uint64_t captured = (verify_after.captured - verify_before.captured) +
                                 (draft_after.captured - draft_before.captured);
  if (o_.qwen.graphs && (kept_graphs == 0 || verify_replayed == 0 || draft_replayed == 0)) {
    problems_.push_back(std::format(
        "swap: no graph captured before the swap replayed after it ({} kept; {} verifies and {} "
        "drafts replayed)",
        kept_graphs, verify_replayed, draft_replayed));
  }
  std::println(
      "swap: out and back after step {} ({}, its commit owed across the swap), {} steps "
      "compared, {} states differ, {} of {} tokens' logits differ; after the swap {} verifies "
      "and {} drafts replayed graphs captured before it ({} kept), {} captured anew",
      done, rejected ? "rows rejected" : "all accepted", control.steps.size(), differing,
      logits_differ, control.logits.size(), verify_replayed, draft_replayed, kept_graphs, captured);
  results_.push_back(
      std::format(R"({{"check":"swap","prompt":"{}","generated":{},"swapped_after_step":{},)"
                  R"("last_step_rejected":{},"steps":{},"state_differs":{},"logits_differ":{},)"
                  R"("tokens_equal":{},"graphs_kept":{},"verify_replayed_after":{},)"
                  R"("draft_replayed_after":{},"captured_after":{}}})",
                  prompt.id, count, done, rejected ? "true" : "false", control.steps.size(),
                  differing, logits_differ, tokens_equal ? "true" : "false", kept_graphs,
                  verify_replayed, draft_replayed, captured));
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
      Generation g;
      if (auto r =
              InRequest("a sampled generation",
                        [&]() -> Status {
                          std::vector<float> first;
                          if (auto f = Prefill(prompt, speculative, first); !f) {
                            return f;
                          }
                          if (speculative) {
                            return Speculate(prompt, kSampledTokens, first, {}, g, &params, seed);
                          }
                          auto pos = static_cast<std::uint32_t>(prompt.ids.size());
                          auto token = ex::Sample(
                              first, params, {.seed = seed, .stream = 0, .position = pos}, scratch);
                          g.tokens = {token.value_or(-1)};
                          std::vector<std::int32_t> history = prompt.ids;
                          while (g.tokens.size() < kSampledTokens) {
                            history.push_back(g.tokens.back());
                            std::vector<float> row;
                            if (auto c = qwen_.Chunk(history, pos, row); !c) {
                              return c;
                            }
                            ++pos;
                            token = ex::Sample(
                                row, params, {.seed = seed, .stream = 0, .position = pos}, scratch);
                            g.tokens.push_back(token.value_or(-1));
                          }
                          return {};
                        });
          !r) {
        return r;
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
  // most frequent tokens (pooled over both modes) plus "other".
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
    for (std::size_t i = 0; entries && i < entries->size(); ++i) {
      theirs[static_cast<std::int32_t>(entries->at(i).at(0).int64().value_or(-1))] =
          static_cast<std::uint64_t>(entries->at(i).at(1).int64().value_or(0));
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
    for (std::size_t i = 0; i < std::min(kHistogramTop, order.size()); ++i) {
      const std::int32_t t = order[i].second;
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

Status Harness::Run() {
  if (auto r = Tokenize(); !r) {
    return r;
  }
  if (auto r = node_.Open(); !r) {
    return r;
  }
  if (auto r = qwen_.Setup(); !r) {
    return r;
  }
  if (with_fp16()) {
    if (auto r = fp16_.Setup(); !r) {
      return r;
    }
  }
  if (auto r = node_.MapWorkspace(
          std::max(qwen_.activations_needed(), with_fp16() ? fp16_.activations_needed() : 0),
          std::max(qwen_.pool_needed(), with_fp16() ? fp16_.pool_needed() : 0));
      !r) {
    return r;
  }
  const std::uint64_t fixed = node_.catalog().OccupancyOf(node_.domain()).Total().value();
  // State diagnostics keep a cataloged pinned copy alongside the device state.
  const std::uint64_t budget =
      fixed + (2 * node_.StateCapacity()) +
      ((qwen_.weights().size() + (with_fp16() ? fp16_.weights().size() : 0)) * ts::kPagedExtent);
  if (auto r = node_.Start(Bytes(budget)); !r) {
    return r;
  }
  if (auto r = qwen_.Register(); !r) {
    return r;
  }
  if (with_fp16()) {
    if (auto r = fp16_.Register(); !r) {
      return r;
    }
  }
  if (auto r = qwen_.Bind(); !r) {
    return r;
  }
  if (with_fp16()) {
    if (auto r = fp16_.Bind(); !r) {
      return r;
    }
  }
  node_.Run();
  std::vector<ts::LoadStats> log;
  const auto start = Clock::now();
  if (auto r = node_.Load(qwen_.weights(), "Qwen3.8's first load", log); !r) {
    return r;
  }
  load_seconds_ = Seconds(Clock::now() - start);
  std::println("loaded: {} bytes ({} the drafter's) in {:.2f} s", qwen_.weight_read_bytes(),
               qwen_.drafter_read_bytes(), load_seconds_);
  if (auto r = qwen_.ReadPleHash(); !r) {
    return r;
  }
  Status checked;
  if (o_.check == "greedy" || o_.check == "timing") {
    checked = Greedy();
  } else if (o_.check == "forced") {
    checked = Forced();
  } else if (o_.check == "swap") {
    checked = Swap();
  } else if (o_.check == "sampled-plain") {
    checked = Sampled(false);
  } else if (o_.check == "sampled-spec") {
    checked = Sampled(true);
  } else {
    checked = Error(std::format("no check {}", o_.check));
  }
  if (!checked) {
    return checked;
  }
  if (qwen_.coverage_violations() != 0) {
    problems_.push_back(
        std::format("{} bound tensors outside cataloged extents of their class; first {}",
                    qwen_.coverage_violations(), qwen_.first_violation()));
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
  const jb::Dsv4GraphStats& g = qwen_.graph_stats();
  const jb::Dsv4GraphStats& d = qwen_.draft_stats();
  std::ofstream(o_.out / "spec.json")
      << std::format(
             R"({{"check":"{}","draft_rows":{},"draft_vocab":{},"adaptive_depth":{},"window":{},"runtime_prefill":{},"load_seconds":{:.2f},)"
             R"("read_bytes":{},"drafter_read_bytes":{},"peak_memavailable_drop_bytes":{},)"
             R"("graphs":{{"eager":{},"captured":{},"replayed":{},"refused":{},"dropped":{}}},)"
             R"("draft_graphs":{{"eager":{},"captured":{},"replayed":{},"refused":{}}},)"
             R"("ple_seconds":{:.3f},"results":[{}],"problems":[{}]}})",
             o_.check, o_.qwen.draft_rows, o_.qwen.draft_vocab,
             o_.adaptive_depth ? "true" : "false", o_.window, o_.runtime_prefill ? "true" : "false",
             load_seconds_, qwen_.weight_read_bytes(), qwen_.drafter_read_bytes(), drop, g.eager,
             g.captured, g.replayed, g.refused, g.dropped, d.eager, d.captured, d.replayed,
             d.refused, qwen_.ple().seconds, all, problems)
      << '\n';
  std::println(
      "peak MemAvailable drop: {:.2f} GiB; graphs {} replayed, {} captured, {} refused ({})",
      static_cast<double>(drop) / (1U << 30U), g.replayed + d.replayed, g.captured + d.captured,
      g.refused + d.refused, g.first_refusal.empty() ? d.first_refusal : g.first_refusal);
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
    if (a == "--qwen38-artifact") {
      o.qwen.artifact = v;
    } else if (a == "--drafter") {
      o.qwen.drafter = v;
    } else if (a == "--tokenizer") {
      o.tokenizer = v;
    } else if (a == "--prompts") {
      o.prompts = v;
    } else if (a == "--reference") {
      o.reference = v;
    } else if (a == "--out") {
      o.out = v;
    } else if (a == "--check") {
      o.check = v;
    } else if (a == "--tokens") {
      ok = number(o.tokens) && o.tokens >= 2;
    } else if (a == "--context") {
      ok = number(o.qwen.context);
    } else if (a == "--prefill-chunk") {
      ok = number(o.qwen.max_rows) && o.qwen.max_rows >= 1;
    } else if (a == "--graphs") {
      o.qwen.graphs = v == "on";
      ok = v == "on" || v == "off";
    } else if (a == "--draft") {
      ok = number(o.qwen.draft_rows) && o.qwen.draft_rows >= 1;
    } else if (a == "--draft-vocab") {
      ok = number(o.qwen.draft_vocab);
    } else if (a == "--adaptive-depth") {
      o.adaptive_depth = v == "on";
      ok = v == "on" || v == "off";
    } else if (a == "--runtime-prefill") {
      o.runtime_prefill = v == "on";
      ok = v == "on" || v == "off";
    } else if (a == "--profile-decode") {
      o.profile_decode = v == "on";
      ok = v == "on" || v == "off";
    } else if (a == "--repeats") {
      ok = number(o.repeats) && o.repeats >= 1;
    } else if (a == "--margin") {
      ok = number(o.margin) && o.margin >= 0;
    } else if (a == "--window") {
      ok = number(o.window) && o.window >= 0 && o.window <= 1;
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
    } else if (a == "--fp16-artifact") {
      o.fp16.artifact = v;
    } else if (a == "--fp16-tokens") {
      o.fp16.tokens = v;
    } else if (a == "--fp16-expect") {
      o.fp16_expect = v;
    } else {
      return Error(std::format("unknown argument {}", a));
    }
    if (!ok) {
      return Error(std::format("{} does not take {}", a, v));
    }
  }
  if (o.qwen.artifact.empty() || o.qwen.drafter.empty() || o.tokenizer.empty() ||
      o.prompts.empty() || o.out.empty() || o.check.empty() ||
      o.fp16.artifact.empty() != o.fp16.tokens.empty()) {
    return Error(
        "usage: jitllm_qwen38_spec --qwen38-artifact DIR --drafter DIR --tokenizer FILE "
        "--prompts FILE --out DIR --check greedy|timing|forced|swap|sampled-plain|sampled-spec "
        "[--reference FILE] [--tokens N] [--context N] [--graphs on|off] [--draft N] "
        "[--draft-vocab N] [--adaptive-depth on|off] [--runtime-prefill on|off] "
        "[--profile-decode on|off] "
        "[--repeats N] [--margin B] [--seeds N] "
        "[--sampled FILE] [--only ID] "
        "[--poll-us N] [--window P] [--prefill-chunk N] [--fp16-artifact "
        "DIR --fp16-tokens FILE "
        "--fp16-expect "
        "SHA256]");
  }
  std::filesystem::create_directories(o.out);
  o.qwen.out = o.out / "qwen38";
  o.fp16.out = o.out / "fp16";
  return o;
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
    Harness harness(*options);
    ran = harness.Run();
    if (auto finished = harness.TearDown(); !finished && ran) {
      ran = finished;
    }
  }
  if (!ran) {
    std::println(stderr, "FAILED: {}", ran.error());
    return 1;
  }
  std::println("DONE {}", options->out.string());
  return 0;
}
