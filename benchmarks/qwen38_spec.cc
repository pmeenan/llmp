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
//                      --check greedy|timing|forced|swap|sampled-plain|sampled-spec|draft-head|wave
//                      [--reference FILE] [--tokens N] [--context N]
//                      [--graphs on|off] [--draft N] [--draft-vocab N]
//                      [--adaptive-depth on|off]
//                      [--runtime-prefill on|off] [--profile-decode on|off]
//                      [--repeats N] [--margin B] [--seeds N] [--sampled FILE]
//                      [--only ID] [--poll-us N] [--window P]
//                      [--slots N (wave: 2-4)] [--wave-lanes on|off]
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
// - wave: independent request slots decode at depth 2 in joined verify
//   waves, with --slots 2-4 and --wave-lanes on|off. Records full tokens,
//   every verify logit row and final target/drafter state SHA-256 per slot.
//   With the optional FP16 fixture, swaps out and back after a replayed
//   full-cohort verify rejected rows in every slot, each Accept's restore
//   still owed; the first verify after return must replay a kept graph.
//   Compare all three hash arrays against the identical no-fixture run.
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
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

#include "base/bytes.h"
#include "base/json.h"
#include "base/sha256.h"
#include "chat/chat.h"
#include "engine/checkpoint_file.h"
#include "engine/support.h"
#include "engine_names.h"
#include "execution/adaptive_depth.h"
#include "execution/sampling.h"
#include "fp16_runner.h"
#include "model/qwen38.h"
#include "paged_node.h"
#include "qwen38_reference.h"
#include "qwen38_vocab.h"
#include "runtime/prefill.h"
#include "scheduler/scheduler.h"
#include "tokenizer/hf.h"
#include "tokenizer/tokenizer.h"

namespace {

namespace ts = jitllm::test_support;
namespace jb = jitllm::benchmarks;
namespace md = jitllm::model;
namespace ex = jitllm::execution;
namespace dv = jitllm::benchmarks::draft_vocab;
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
  std::filesystem::path chat_template;
  std::filesystem::path stop_metadata;
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

// The private held-out control keeps complete target rows. Refuse a short
// generation or a nonfinite row before argmax/near-tie evidence is emitted.
Status VocabGeneration(const Generation& generation) {
  constexpr std::size_t count = 256;
  const auto vocab = md::Qwen38Flash().vocab;
  if (generation.tokens.size() != count || generation.logits.size() != count) {
    return Error("held-out generation did not retain all 256 token rows");
  }
  for (std::size_t i = 0; i < count; ++i) {
    if (generation.logits[i].size() != vocab || generation.tokens[i] < 0 ||
        std::cmp_greater_equal(generation.tokens[i], vocab) ||
        !std::ranges::all_of(
            generation.logits[i], [](float value) { return std::isfinite(value); })) {
      return Error("held-out generation has a malformed or nonfinite target row");
    }
  }
  return {};
}

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
  Status TearDown() { return node_.TearDown(entered_models_); }

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
  Status DraftHead(bool routed_down = false);
  Status VocabAnchors();
  std::expected<std::string, std::string> UsedStateDigest();
  Status Swap();
  Status SwapOut();
  Status SwapIn();
  Status Sampled(bool speculative);
  Status Wave();
  Status Write();

  const Options& o_;
  std::string record_;
  MemorySampler memory_;
  std::uint64_t available_before_ = MemAvailable();
  ts::PagedNode node_;
  jb::Qwen38Runner qwen_;
  jb::Fp16Runner fp16_;
  std::vector<ts::PagedModel*> entered_models_;
  std::unique_ptr<jitllm::tokenizer::Tokenizer> tokenizer_;
  std::vector<Prompt> decode_;
  std::vector<Prompt> chat_;
  std::vector<dv::Example> vocab_examples_;
  std::vector<std::string> results_;  // JSON objects
  std::vector<std::string> problems_;
  double load_seconds_ = 0;
  double vocab_digest_seconds_ = 0;
  double vocab_copy_seconds_ = 0;
  double vocab_sha_seconds_ = 0;
  double vocab_checkpoint_seconds_ = 0;
  double vocab_restore_seconds_ = 0;
  double vocab_reference_staging_seconds_ = 0;
  std::uint64_t vocab_digest_bytes_ = 0;
  std::uint64_t vocab_digest_calls_ = 0;
  dv::ReferenceStats vocab_reference_;
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
  if (o_.check == "vocab-anchors" || o_.check == "vocab-greedy") {
    auto examples = dv::ReadExamples(*text, o_.qwen.context, md::Qwen38Flash().vocab);
    if (!examples) {
      return Error(examples.error());
    }
    vocab_examples_ = std::move(*examples);
    if (o_.check == "vocab-greedy") {
      if (vocab_examples_.front().split != "held_out") {
        return Error("vocab-greedy requires the frozen held-out input manifest");
      }
      for (const auto& example : vocab_examples_) {
        decode_.push_back(
            {.id = example.id, .ids = example.prompt, .stable_boundary = example.stable_boundary});
      }
    }
    return {};
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
  if (o_.check == "vocab-greedy") {
    if (auto checked = VocabGeneration(forced); !checked) {
      return checked;
    }
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
      if (o_.check == "vocab-greedy") {
        if (auto checked = VocabGeneration(plain); !checked) {
          return checked;
        }
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
      if (o_.check == "vocab-greedy") {
        if (auto checked = VocabGeneration(spec); !checked) {
          return checked;
        }
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

// Two cataloged page views, held across this anchor's completed copies. A
// failed copy quarantines the whole allocation; no destructor can prove it.
class VocabReferenceBuffers {
 public:
  explicit VocabReferenceBuffers(ts::PagedNode& node) : node_(node) {}
  VocabReferenceBuffers(const VocabReferenceBuffers&) = delete;
  VocabReferenceBuffers& operator=(const VocabReferenceBuffers&) = delete;
  VocabReferenceBuffers(VocabReferenceBuffers&&) = delete;
  VocabReferenceBuffers& operator=(VocabReferenceBuffers&&) = delete;
  ~VocabReferenceBuffers() {
    if (base_ != nullptr) {
      if (unknown_ || !node_.FreePinned(base_)) {
        node_.KeepPinned(base_);
      }
    }
  }
  Status Open() {
    constexpr std::uint64_t alignment = 4096;
    std::vector<jitllm::catalog::ExtentId> extents;
    auto memory = node_.Pinned(2 * (dv::ReferencePages::kPage + alignment), kQwen, extents);
    if (!memory) {
      return Error(memory.error());
    }
    base_ = *memory;
    auto* first = static_cast<std::byte*>(jitllm::engine::support::Pointer(
        jitllm::engine::support::Round(reinterpret_cast<std::uintptr_t>(base_), alignment)));
    live_ = {first, static_cast<std::size_t>(dv::ReferencePages::kPage)};
    expected_ = {first + dv::ReferencePages::kPage,
                 static_cast<std::size_t>(dv::ReferencePages::kPage)};
    return {};
  }
  Status Close() {
    if (unknown_) {
      return Error("reference copy completion is unknown");
    }
    if (auto freed = node_.FreePinned(base_); !freed) {
      node_.KeepPinned(base_);
      base_ = nullptr;
      return freed;
    }
    base_ = nullptr;
    return {};
  }
  void Unknown() { unknown_ = true; }
  std::span<std::byte> live() const { return live_; }
  std::span<std::byte> expected() const { return expected_; }

 private:
  ts::PagedNode& node_;
  void* base_ = nullptr;
  bool unknown_ = false;
  std::span<std::byte> live_;
  std::span<std::byte> expected_;
};

std::expected<std::string, std::string> Harness::UsedStateDigest() {
  const auto start = Clock::now();
  const auto ranges = qwen_.used_state_ranges();
  if (ranges.empty()) {
    return Error("vocabulary control has no used state");
  }
  std::vector<jitllm::catalog::ExtentId> extents;
  auto staging = node_.Pinned(ts::kPagedExtent, kQwen, extents);
  if (!staging) {
    return Error(staging.error());
  }
  jitllm::base::Sha256 hash;
  for (const auto& range : ranges) {
    if (range.bytes == 0 || range.bytes > ts::kPagedExtent) {
      if (auto freed = node_.FreePinned(*staging); !freed) {
        node_.KeepPinned(*staging);
      }
      return Error("vocabulary control state exceeds one staging page");
    }
    const auto copy_start = Clock::now();
    if (auto copied = qwen_.CopyCheckpointState(*staging, std::span(&range, 1), true); !copied) {
      node_.KeepPinned(*staging);  // failed completion never authorizes a free
      return Error(copied.error());
    }
    vocab_copy_seconds_ += Seconds(Clock::now() - copy_start);
    const auto sha_start = Clock::now();
    hash.Update(std::format("{}:{}:{};", range.region, range.offset, range.bytes));
    hash.Update(
        std::span(static_cast<const std::byte*>(*staging), static_cast<std::size_t>(range.bytes)));
    vocab_digest_bytes_ += range.bytes;
    vocab_sha_seconds_ += Seconds(Clock::now() - sha_start);
  }
  if (auto freed = node_.FreePinned(*staging); !freed) {
    node_.KeepPinned(*staging);
    return Error(freed.error());
  }
  auto result = jitllm::base::ToHex(hash.Finish());
  vocab_digest_seconds_ += Seconds(Clock::now() - start);
  ++vocab_digest_calls_;
  return result;
}

Status Harness::VocabAnchors() {
  struct Arm {
    std::vector<std::int32_t> drafts;
    std::vector<std::int32_t> verdicts;
    std::vector<float> probabilities;
    std::string drafted_state;
    std::string committed_state;
    std::string verify_sha;
    std::uint32_t keep = 0;
  };
  const auto digest = [](std::span<const std::byte> bytes) {
    jitllm::base::Sha256 hash;
    hash.Update(bytes);
    return jitllm::base::ToHex(hash.Finish());
  };
  const auto write = [&](const std::string& name, std::span<const std::byte> bytes) {
    std::ofstream file(o_.out / name, std::ios::binary);
    file.write(reinterpret_cast<const char*>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
    file.close();
    return file.good();
  };
  for (const dv::Example& example : vocab_examples_) {
    if (!o_.only.empty() && example.id != o_.only) {
      continue;
    }
    const auto start = Clock::now();
    auto ran = InRequest("frozen own-vocabulary common history", [&]() -> Status {
      const Prompt prompt{
          .id = example.id, .ids = example.prompt, .stable_boundary = example.stable_boundary};
      std::vector<float> last;
      if (auto filled = Prefill(prompt, true, last); !filled) {
        return filled;
      }
      std::vector<std::int32_t> history = example.prompt;
      auto consumed = static_cast<std::uint32_t>(history.size());
      for (const std::uint32_t offset : std::span(example.anchors).first(o_.tokens)) {
        history.assign(example.prompt.begin(), example.prompt.end());
        history.insert(history.end(), example.continuation.begin(),
                       example.continuation.begin() + static_cast<std::ptrdiff_t>(offset) + 1);
        const auto position = static_cast<std::uint32_t>(history.size() - 1);
        // The anchor itself is uncached. Intervening authored tokens supply
        // conditioning only; their IDs are never the target-frequency labels.
        if (consumed < position) {
          if (auto advanced = qwen_.Chunk(std::span(history).first(position), consumed, last, true);
              !advanced) {
            return advanced;
          }
        }
        if (auto settled = qwen_.Rollback(); !settled) {
          return settled;
        }
        const auto pending = qwen_.pending_rows();
        const auto before_ranges = qwen_.used_state_ranges();
        VocabReferenceBuffers buffers(node_);
        const auto staging_start = Clock::now();
        if (auto opened = buffers.Open(); !opened) {
          return opened;
        }
        vocab_reference_staging_seconds_ += Seconds(Clock::now() - staging_start);
        const auto page_ranges = [&] {
          std::vector<dv::PageRange> pages;
          for (const auto& range : qwen_.used_state_ranges()) {
            pages.push_back({.region = range.region, .offset = range.offset, .bytes = range.bytes});
          }
          return pages;
        };
        const auto copy_page = [&](void* host, const dv::PageRange& page) {
          const jitllm::engine::LiveState::Range range{
              .region = page.region, .offset = page.offset, .bytes = page.bytes};
          return qwen_.CopyCheckpointState(host, std::span(&range, 1), true);
        };
        const auto capture_reference = [&]() -> std::expected<dv::ReferencePages, std::string> {
          auto reference = dv::ReferencePages::Capture(o_.out, page_ranges(), qwen_.pending_rows(),
                                                       buffers.live(), copy_page, vocab_reference_);
          if (!reference) {
            if (reference.error().unknown_copy) {
              buffers.Unknown();
            }
            return Error(reference.error().detail);
          }
          return std::move(*reference);
        };
        const auto compare_reference = [&](const dv::ReferencePages& reference) -> Status {
          auto compared = reference.Compare(page_ranges(), qwen_.pending_rows(), buffers.expected(),
                                            buffers.live(), copy_page, vocab_reference_);
          if (!compared) {
            if (compared.error().unknown_copy) {
              buffers.Unknown();
            }
            return Error(compared.error().detail);
          }
          return {};
        };
        auto before = capture_reference();
        if (!before) {
          return Error(before.error());
        }
        std::optional<dv::ReferencePages> drafted_reference;
        std::optional<dv::ReferencePages> committed_reference;
        const auto checkpoint_start = Clock::now();
        auto checkpoint = jitllm::engine::CheckpointFile::Capture(
            node_, o_.out, before_ranges,
            [&](void* host, std::span<const jitllm::engine::LiveState::Range> ranges) {
              return qwen_.CopyCheckpointState(host, ranges, true);
            });
        if (!checkpoint) {
          return Error(checkpoint.error().detail);
        }
        vocab_checkpoint_seconds_ += Seconds(Clock::now() - checkpoint_start);
        Arm baseline;
        jitllm::engine::Qwen38DraftHeadCapture captured;
        for (std::uint32_t arm = 0; arm < 4; ++arm) {
          const bool capture = arm == 1 || arm == 2;
          Arm current;
          jitllm::engine::Qwen38DraftHeadCapture head;
          if (auto drafted = qwen_.Draft(history, current.drafts, &current.probabilities,
                                         dv::kDepth, capture ? &head : nullptr);
              !drafted) {
            return drafted;
          }
          if (arm == 0) {
            auto reference = capture_reference();
            if (!reference) {
              return Error(reference.error());
            }
            drafted_reference.emplace(std::move(*reference));
          } else if (auto compared = compare_reference(*drafted_reference); !compared) {
            return compared;
          }
          current.drafted_state = drafted_reference->sha256();
          if (current.drafts.size() != dv::kDepth || current.probabilities.size() != dv::kDepth ||
              !std::ranges::all_of(current.probabilities,
                                   [](float p) { return std::isfinite(p) && p >= 0 && p <= 1; })) {
            return Error("vocabulary control returned incomplete drafts or confidence");
          }
          if (capture) {
            const std::size_t width = md::Qwen38Flash().width;
            if (head.inputs.size() != width * dv::kDepth || head.head_rows == 0 ||
                head.head_rows > 65536 ||
                head.logits.size() != static_cast<std::size_t>(head.head_rows) * dv::kDepth ||
                head.token_ids.size() != head.head_rows || head.catch_up_rows != pending ||
                !md::CheckQwen38DraftIds(head.token_ids, qwen_.vocab()) ||
                !std::ranges::all_of(head.inputs, [](float f) { return std::isfinite(f); }) ||
                !std::ranges::all_of(head.logits, [](float f) { return std::isfinite(f); })) {
              return Error("vocabulary control capture has invalid extents or values");
            }
            for (std::uint32_t pass = 0; pass < dv::kDepth; ++pass) {
              if (std::ranges::none_of(std::span(head.inputs).subspan(pass * width, width),
                                       [](float value) { return value != 0; })) {
                return Error("vocabulary control captured a zero head operand");
              }
            }
            if (arm == 1) {
              captured = std::move(head);
            } else if (head.catch_up_rows != captured.catch_up_rows ||
                       head.head_rows != captured.head_rows ||
                       head.token_ids != captured.token_ids ||
                       !SameBits(head.inputs, captured.inputs) ||
                       !SameBits(head.logits, captured.logits)) {
              return Error("vocabulary control retained operands do not repeat exactly");
            }
          }
          auto verification = history;
          verification.insert(verification.end(), current.drafts.begin(), current.drafts.end());
          std::vector<float> logits;
          if (auto verified = qwen_.Verify(verification, position, current.verdicts, &logits);
              !verified) {
            return verified;
          }
          if (current.verdicts.size() != dv::kDepth + 1 ||
              logits.size() != std::size_t{qwen_.vocab()} * (dv::kDepth + 1) ||
              !std::ranges::all_of(logits, [](float f) { return std::isfinite(f); })) {
            return Error("vocabulary control has incomplete or nonfinite verification rows");
          }
          current.verify_sha = digest(std::as_bytes(std::span(logits)));
          std::uint32_t accepted = 0;
          while (accepted < dv::kDepth && current.verdicts[accepted] == current.drafts[accepted]) {
            ++accepted;
          }
          current.keep = accepted + 1;
          if (auto kept = qwen_.Accept(current.keep); !kept) {
            return kept;
          }
          if (auto settled = qwen_.Rollback(); !settled) {
            return settled;
          }
          if (arm == 0) {
            auto reference = capture_reference();
            if (!reference) {
              return Error(reference.error());
            }
            committed_reference.emplace(std::move(*reference));
          } else if (auto compared = compare_reference(*committed_reference); !compared) {
            return compared;
          }
          current.committed_state = committed_reference->sha256();
          if (arm == 0) {
            baseline = current;
          } else if (current.drafts != baseline.drafts || current.verdicts != baseline.verdicts ||
                     current.keep != baseline.keep ||
                     !SameBits(current.probabilities, baseline.probabilities) ||
                     current.drafted_state != baseline.drafted_state ||
                     current.committed_state != baseline.committed_state ||
                     current.verify_sha != baseline.verify_sha) {
            return Error("capture off/on/on/off changed a natural proposal, verify or state");
          }
          const auto restore_start = Clock::now();
          auto restored = checkpoint->Restore(
              node_, [&] { return qwen_.PrepareRestoreState(checkpoint->ranges()); },
              [&](void* host, std::span<const jitllm::engine::LiveState::Range> ranges) {
                return qwen_.CopyCheckpointState(host, ranges, false);
              });
          if (!restored) {
            return Error(restored.error().detail);
          }
          vocab_restore_seconds_ += Seconds(Clock::now() - restore_start);
          qwen_.set_pending_rows(pending);
          if (auto compared = compare_reference(*before); !compared) {
            return compared;
          }
        }
        // Natural target argmax is computed on the original common history
        // after restoration, independently of any head's proposal or authored answer.
        if (auto controlled = qwen_.Chunk(history, position, last, true); !controlled) {
          return controlled;
        }
        if (last.size() != qwen_.vocab() ||
            !std::ranges::all_of(last, [](float f) { return std::isfinite(f); })) {
          return Error("vocabulary target-control row is incomplete or nonfinite");
        }
        consumed = position + 1;
        const auto stem = std::format("vocab-{}-{}", example.id, offset);
        if (!write(stem + ".inputs.f32", std::as_bytes(std::span(captured.inputs))) ||
            !write(stem + ".logits.f32", std::as_bytes(std::span(captured.logits))) ||
            !write(stem + ".history.i32", std::as_bytes(std::span(history))) ||
            !write(stem + ".target.f32", std::as_bytes(std::span(last))) ||
            !write("vocab-head.ids.i32", std::as_bytes(std::span(captured.token_ids)))) {
          return Error("writing authenticated common-anchor vocabulary captures");
        }
        std::string drafts;
        for (const auto token : baseline.drafts) {
          drafts += std::format("{}{}", drafts.empty() ? "" : ",", token);
        }
        auto advanced_sha = UsedStateDigest();
        if (!advanced_sha) {
          return Error(advanced_sha.error());
        }
        results_.push_back(std::format(
            R"({{"check":"vocab_anchor","example":"{}","domain":"{}","nominal_tokens":{},"prompt_tokens":{},"anchor_offset":{},"anchor_position":{},"passes":3,"arms":4,"catch_up_rows":{},"head_rows":{},"width":{},"drafts":[{}],"accepted":{},"target_signal":"native_control_natural_argmax","target_id":{},"history_sha256":"{}","first_input_sha256":"{}","capture_input_sha256":"{}","captured_logits_sha256":"{}","target_logits_sha256":"{}","confidence_sha256":"{}","verify_logits_sha256":"{}","drafted_state_sha256":"{}","committed_state_sha256":"{}","before_state_sha256":"{}","control_advanced_state_sha256":"{}","checkpoint_bytes":{},"state_sha_traversals":4,"state_reference_captures":3,"state_reference_comparisons":10,"used_range_geometry_exact":true,"pending_cursor_exact":true,"own_repeat_exact":true,"capture_off_on_exact":true,"full_restore_exact":true}})",
            example.id, example.domain, example.nominal_tokens, example.prompt.size(), offset,
            position, pending, captured.head_rows, md::Qwen38Flash().width, drafts,
            baseline.keep - 1, Argmax(last), digest(std::as_bytes(std::span(history))),
            digest(std::as_bytes(std::span(captured.inputs).first(md::Qwen38Flash().width))),
            digest(std::as_bytes(std::span(captured.inputs))),
            digest(std::as_bytes(std::span(captured.logits))),
            digest(std::as_bytes(std::span(last))),
            digest(std::as_bytes(std::span(baseline.probabilities))), baseline.verify_sha,
            baseline.drafted_state, baseline.committed_state, before->sha256(), *advanced_sha,
            checkpoint->bytes()));
        const auto staging_close = Clock::now();
        if (auto freed = buffers.Close(); !freed) {
          return freed;
        }
        vocab_reference_staging_seconds_ += Seconds(Clock::now() - staging_close);
      }
      return {};
    });
    if (!ran) {
      return ran;
    }
    results_.push_back(std::format(
        R"({{"check":"vocab_example","example":"{}","anchors":{},"diagnostic_seconds":{:.9f}}})",
        example.id, o_.tokens, Seconds(Clock::now() - start)));
  }
  const auto selected = std::ranges::count_if(
      vocab_examples_, [&](const dv::Example& e) { return o_.only.empty() || e.id == o_.only; });
  if (selected == 0) {
    return Error("vocabulary pilot did not select a frozen example");
  }
  results_.push_back(std::format(
      R"({{"check":"vocab_collection","split":"{}","examples":{},"anchors":{},"complete_collection":{},"diagnostic_only":true,"state_digest_calls":{},"state_digest_bytes":{},"state_digest_seconds":{:.9f},"state_copy_seconds":{:.9f},"state_sha_seconds":{:.9f},"checkpoint_capture_seconds":{:.9f},"checkpoint_restore_seconds":{:.9f},"reference_captures":{},"reference_comparisons":{},"reference_capture_bytes":{},"reference_comparison_bytes":{},"reference_write_bytes":{},"reference_read_bytes":{},"reference_capture_seconds":{:.9f},"reference_comparison_seconds":{:.9f},"reference_capture_copy_seconds":{:.9f},"reference_comparison_copy_seconds":{:.9f},"reference_write_seconds":{:.9f},"reference_read_seconds":{:.9f},"reference_byte_compare_seconds":{:.9f},"reference_staging_seconds":{:.9f},"advanced_digest_seconds":{:.9f}}})",
      vocab_examples_.front().split, selected, selected * o_.tokens,
      selected == 8 && o_.tokens == 4 ? "true" : "false",
      vocab_digest_calls_ + vocab_reference_.captures,
      vocab_digest_bytes_ + vocab_reference_.capture_bytes,
      vocab_digest_seconds_ + vocab_reference_.capture_copy_seconds + vocab_reference_.sha_seconds,
      vocab_copy_seconds_ + vocab_reference_.capture_copy_seconds,
      vocab_sha_seconds_ + vocab_reference_.sha_seconds, vocab_checkpoint_seconds_,
      vocab_restore_seconds_, vocab_reference_.captures, vocab_reference_.comparisons,
      vocab_reference_.capture_bytes, vocab_reference_.comparison_bytes,
      vocab_reference_.write_bytes, vocab_reference_.read_bytes, vocab_reference_.capture_seconds,
      vocab_reference_.comparison_seconds, vocab_reference_.capture_copy_seconds,
      vocab_reference_.comparison_copy_seconds, vocab_reference_.write_seconds,
      vocab_reference_.read_seconds, vocab_reference_.byte_compare_seconds,
      vocab_reference_staging_seconds_, vocab_digest_seconds_));
  return {};
}

Status Harness::DraftHead(bool routed_down) {
  const bool projection = o_.check == "mxfp8-projection";
  std::string_view stage = "draft-head";
  if (projection) {
    stage = "mxfp8-projection";
  } else if (routed_down) {
    stage = "routed-down";
  }
  const std::array<std::uint32_t, 3> all_layers{0, 23, 47};
  const std::span<const std::uint32_t> layers(all_layers);
  const auto selected_layers = projection ? layers.subspan(1) : layers;
  const Prompt* chosen = nullptr;
  for (const auto* set : {&chat_, &decode_}) {
    for (const Prompt& p : *set) {
      if (chosen == nullptr && (o_.only.empty() || p.id == o_.only)) {
        chosen = &p;
      }
    }
  }
  if (chosen == nullptr) {
    return Error("the draft-head control needs a selected prompt");
  }
  const Prompt& prompt = *chosen;
  const std::uint32_t depth = o_.qwen.draft_rows;
  if (std::uint64_t{prompt.ids.size()} + (std::uint64_t{o_.tokens} * depth) + depth + 1 >
      o_.qwen.context) {
    return Error("the prompt and bounded draft-head steps exceed the configured context");
  }
  struct Control {
    std::vector<std::int32_t> drafts;
    std::vector<float> probabilities;
    std::vector<std::uint64_t> draft_state;
    std::vector<std::uint64_t> kept_state;
    std::uint64_t verify_logits = 0;
    std::vector<std::int32_t> verdicts;
    std::int32_t next = -1;
  };
  std::vector<Control> baseline;
  std::vector<jitllm::engine::Qwen38DraftHeadCapture> captured;
  std::vector<jitllm::engine::Qwen38RoutedCapture> routed_records;
  std::vector<float> baseline_first;
  const auto same_floats = [](const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() &&
           std::ranges::equal(std::as_bytes(std::span(a)), std::as_bytes(std::span(b)));
  };
  const auto write = [&](const std::filesystem::path& name, std::span<const std::byte> bytes) {
    std::ofstream file(o_.out / name, std::ios::binary);
    file.write(reinterpret_cast<const char*>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
    file.close();
    return file.good();
  };
  if (!write(std::format("{}.prompt.i32", stage), std::as_bytes(std::span(prompt.ids)))) {
    return Error("writing the draft-head prompt IDs");
  }
  for (std::uint32_t arm = 0; arm < 4; ++arm) {
    const bool capture = arm == 1 || arm == 2;
    auto ran = InRequest(std::format("bounded {} capture control", stage), [&]() -> Status {
      std::vector<float> first;
      if (auto p = Prefill(prompt, true, first); !p) {
        return p;
      }
      if (arm == 0) {
        baseline_first = first;
      } else if (!same_floats(first, baseline_first)) {
        return Error("capture on/off prefills do not have identical first logits");
      }
      std::vector<std::int32_t> history = prompt.ids;
      history.push_back(Argmax(first));
      for (std::uint32_t s = 0; s < o_.tokens; ++s) {
        Control current;
        jitllm::engine::Qwen38DraftHeadCapture head;
        if (auto drafted = qwen_.Draft(history, current.drafts, &current.probabilities, depth,
                                       capture && !routed_down ? &head : nullptr);
            !drafted) {
          return drafted;
        }
        const std::uint32_t keep = 1 + (s % depth);
        Step stamp;
        stamp.kept = keep;
        if (auto read = Fingerprints(stamp, current.draft_state); !read) {
          return read;
        }
        if (capture && !routed_down) {
          const std::size_t width = md::Qwen38Flash().width;
          if (head.inputs.size() != width * depth || head.head_rows == 0 ||
              head.logits.size() != std::size_t{head.head_rows} * depth ||
              head.token_ids.size() != head.head_rows ||
              !md::CheckQwen38DraftIds(head.token_ids, qwen_.vocab()) ||
              !std::ranges::all_of(head.inputs, [](float v) { return std::isfinite(v); }) ||
              !std::ranges::all_of(head.logits, [](float v) { return std::isfinite(v); })) {
            return Error("the retained draft-head capture has invalid dimensions or values");
          }
          for (std::uint32_t pass = 0; pass < depth; ++pass) {
            const auto row = std::span(head.inputs).subspan(pass * width, width);
            if (std::ranges::none_of(row, [](float v) { return v != 0; })) {
              return Error("a retained draft-head input is all zero");
            }
          }
          if (arm == 1) {
            captured.push_back(head);
            const auto stem = std::format("draft-head-{}", s);
            if (!write(stem + ".inputs.f32", std::as_bytes(std::span(head.inputs))) ||
                !write(stem + ".logits.f32", std::as_bytes(std::span(head.logits))) ||
                !write(stem + ".history.i32",
                       std::as_bytes(std::span(history).subspan(prompt.ids.size()))) ||
                (s == 0 &&
                 !write("draft-head.ids.i32", std::as_bytes(std::span(head.token_ids))))) {
              return Error("writing the external draft-head capture");
            }
            results_.push_back(std::format(
                R"({{"check":"draft_head_capture","step":{},"anchor_position":{},"passes":{},"catch_up_rows":{},"head_rows":{},"width":{},"input_fingerprint":{},"logit_fingerprint":{},"ids_fingerprint":{}}})",
                s, history.size() - 1, depth, head.catch_up_rows, head.head_rows, width,
                Fingerprint(std::as_bytes(std::span(head.inputs))),
                Fingerprint(std::as_bytes(std::span(head.logits))),
                Fingerprint(std::as_bytes(std::span(head.token_ids)))));
          } else if (head.catch_up_rows != captured[s].catch_up_rows ||
                     head.head_rows != captured[s].head_rows ||
                     head.token_ids != captured[s].token_ids ||
                     !same_floats(head.inputs, captured[s].inputs) ||
                     !same_floats(head.logits, captured[s].logits)) {
            return Error(std::format("draft-head capture repeat differs at step {}", s));
          }
        }
        const auto pos = static_cast<std::uint32_t>(history.size() - 1);
        const auto& common = arm == 0 ? current.drafts : baseline[s].drafts;
        history.insert(history.end(), common.begin(), common.end());
        std::vector<float> logits;
        jitllm::engine::Qwen38RoutedCapture routed;
        if (auto verified = qwen_.Verify(history, pos, current.verdicts, &logits,
                                         capture && routed_down ? &routed : nullptr);
            !verified) {
          return verified;
        }
        if (capture && routed_down) {
          const auto& profile = md::Qwen38Flash();
          if (routed.rows != depth + 1 || routed.layers.size() != selected_layers.size()) {
            return Error("the routed capture is missing rows or layers");
          }
          std::size_t j = 0;
          for (const auto& layer : routed.layers) {
            const std::size_t slots = std::size_t{routed.rows} * profile.experts_used;
            const std::size_t projection_columns =
                layer.layer == 0 ? 0 : 2 * std::size_t{profile.heads} * profile.head_dim;
            const std::size_t attention_input =
                projection_columns == 0 ? 0 : std::size_t{routed.rows} * profile.width;
            if (layer.layer != selected_layers[j] ||
                layer.input.size() != std::size_t{routed.rows} * profile.width ||
                layer.activation.size() != slots * profile.expert_ffn ||
                layer.down.size() != slots * profile.width || layer.ids.size() != slots ||
                layer.weights.size() != slots || layer.gate.size() != routed.rows ||
                layer.shared.size() != std::size_t{routed.rows} * profile.width ||
                layer.combined.size() != layer.shared.size() ||
                layer.attention_input.size() != attention_input ||
                layer.attention_projection.size() != projection_columns * routed.rows) {
              return Error("the routed capture has invalid operand extents");
            }
            for (const auto* values : {&layer.input, &layer.activation, &layer.down, &layer.shared,
                                       &layer.gate, &layer.weights, &layer.combined,
                                       &layer.attention_input, &layer.attention_projection}) {
              if (!std::ranges::all_of(*values, [](float value) { return std::isfinite(value); })) {
                return Error("a routed capture operand is not finite");
              }
            }
            for (std::uint32_t t = 0; t < routed.rows; ++t) {
              const auto ids = std::span(layer.ids).subspan(std::size_t{t} * profile.experts_used,
                                                            profile.experts_used);
              for (std::size_t i = 0; i < ids.size(); ++i) {
                if (ids[i] < 0 || std::cmp_greater_equal(ids[i], profile.experts) ||
                    std::find(ids.begin(), ids.begin() + static_cast<std::ptrdiff_t>(i), ids[i]) !=
                        ids.begin() + static_cast<std::ptrdiff_t>(i)) {
                  return Error("routed capture IDs are not valid distinct experts per token");
                }
              }
            }
            if (std::ranges::none_of(layer.input, [](float value) { return value != 0; }) ||
                std::ranges::none_of(layer.activation, [](float value) { return value != 0; })) {
              return Error("a routed capture input or activation is all zero");
            }
            const auto stem = std::format("{}-{}-{}", stage, s, layer.layer);
            if (arm == 1) {
              for (const auto& [name, values] :
                   {std::pair{"input", &layer.input}, std::pair{"activation", &layer.activation},
                    std::pair{"down", &layer.down}, std::pair{"shared", &layer.shared},
                    std::pair{"gate", &layer.gate}, std::pair{"weights", &layer.weights},
                    std::pair{"combined", &layer.combined},
                    std::pair{"attention-input", &layer.attention_input},
                    std::pair{"attention-projection", &layer.attention_projection}}) {
                if (values->empty()) {
                  continue;
                }
                if (!write(stem + "." + name + ".f32", std::as_bytes(std::span(*values)))) {
                  return Error("writing the external routed operand");
                }
              }
              if (!write(stem + ".ids.i32", std::as_bytes(std::span(layer.ids))) ||
                  !write(stem + ".history.i32", std::as_bytes(std::span(history)))) {
                return Error("writing the external routed history and IDs");
              }
              results_.push_back(std::format(
                  R"({{"check":"routed_down_capture","step":{},"anchor_position":{},"layer":{},"rows":{},"ffn":{},"width":{},"experts_used":{},"input_fingerprint":{},"activation_fingerprint":{},"down_fingerprint":{},"combined_fingerprint":{},"ids_fingerprint":{},"projection_columns":{},"attention_input_fingerprint":{},"attention_projection_fingerprint":{}}})",
                  s, pos, layer.layer, routed.rows, profile.expert_ffn, profile.width,
                  profile.experts_used, Fingerprint(std::as_bytes(std::span(layer.input))),
                  Fingerprint(std::as_bytes(std::span(layer.activation))),
                  Fingerprint(std::as_bytes(std::span(layer.down))),
                  Fingerprint(std::as_bytes(std::span(layer.combined))),
                  Fingerprint(std::as_bytes(std::span(layer.ids))), projection_columns,
                  Fingerprint(std::as_bytes(std::span(layer.attention_input))),
                  Fingerprint(std::as_bytes(std::span(layer.attention_projection)))));
            } else {
              const auto& before = routed_records[s].layers[j];
              if (routed.rows != routed_records[s].rows || layer.layer != before.layer ||
                  layer.ids != before.ids || !same_floats(layer.activation, before.activation) ||
                  !same_floats(layer.input, before.input) ||
                  !same_floats(layer.down, before.down) ||
                  !same_floats(layer.shared, before.shared) ||
                  !same_floats(layer.gate, before.gate) ||
                  !same_floats(layer.weights, before.weights) ||
                  !same_floats(layer.combined, before.combined) ||
                  !same_floats(layer.attention_input, before.attention_input) ||
                  !same_floats(layer.attention_projection, before.attention_projection)) {
                return Error(std::format("routed operand repeat differs at step {} layer {}", s,
                                         layer.layer));
              }
            }
            ++j;
          }
          if (arm == 1) {
            routed_records.push_back(std::move(routed));
          }
        }
        if (current.verdicts.size() != depth + 1) {
          return Error("draft-head control verify returned an incomplete verdict");
        }
        current.verify_logits = Fingerprint(std::as_bytes(std::span(logits)));
        current.next = current.verdicts[keep - 1];
        if (auto accepted = qwen_.Accept(keep); !accepted) {
          return accepted;
        }
        if (auto settled = qwen_.Rollback(); !settled) {
          return settled;
        }
        if (auto read = Fingerprints(stamp, current.kept_state); !read) {
          return read;
        }
        if (arm == 0) {
          baseline.push_back(current);
        } else if (current.drafts != baseline[s].drafts ||
                   !same_floats(current.probabilities, baseline[s].probabilities) ||
                   current.draft_state != baseline[s].draft_state ||
                   current.kept_state != baseline[s].kept_state ||
                   current.verify_logits != baseline[s].verify_logits ||
                   current.verdicts != baseline[s].verdicts) {
          return Error(
              std::format("capture {}/off common-input control differs at step {}", arm, s));
        }
        history.resize(std::size_t{pos} + keep);
        history.push_back(baseline[s].next);
      }
      return {};
    });
    if (!ran) {
      return ran;
    }
    results_.push_back(std::format(
        R"({{"check":"{}_control","arm":{},"capture":{},"steps":{},"state_and_drafts_equal":true}})",
        stage, arm, capture ? "true" : "false", o_.tokens));
  }
  return {};
}

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
// --check wave (--slots N, 2 to 4; --wave-lanes on|off): the first N
// prompts (decode, then chat), each on a slot of its own, prefilled with the
// drafter's injection and decoded greedily in waves as serving runs them
// (serving.cc's Qwen RunPreparedGenerationWave): drafts at depth 2 (a draft
// wave of up to two slots, past two each slot drafting alone), then one
// verify wave, each slot's verdict its verify's argmaxes. Reports each
// slot's tokens and a SHA-256 of every verify row it computed, and the
// median draft and verify wave: two runs that differ only in their lanes
// must report the same digests. With the optional FP16 fixture, swaps out
// and back after a captured/replayed full-cohort verify has rejected rows
// in every slot, each Accept's restore still owed. The first verify after
// return must replay a kept graph. Runs with and without the fixture must
// report identical token, full-logit-row and final-state hashes.
Status Harness::Wave() {
  if (with_fp16() && !o_.qwen.graphs) {
    return Error("the wave swap check requires captured and replayed verify graphs");
  }
  const std::uint32_t n = o_.qwen.wave_slots;
  constexpr std::uint32_t kDepth = 2;
  std::vector<const Prompt*> prompts;
  for (const auto* set : {&decode_, &chat_}) {
    for (const Prompt& p : *set) {
      if (prompts.size() < n) {
        prompts.push_back(&p);
      }
    }
  }
  if (prompts.size() < n) {
    return Error("the wave check needs a prompt a slot");
  }
  for (const Prompt* prompt : prompts) {
    if (prompt->ids.empty() || prompt->ids.size() + o_.tokens + kDepth > o_.qwen.context) {
      return Error("the wave check needs room for the prompt, output and draft lookahead");
    }
  }
  std::vector<jb::Qwen38Runner::Slot*> slots;
  for (std::uint32_t i = 0; i < n; ++i) {
    auto slot = qwen_.request_slot(i);
    if (!slot) {
      return std::unexpected(slot.error());
    }
    slots.push_back(*slot);
  }
  struct Run {
    std::vector<std::int32_t> all;  // the prompt and its tokens (the anchor last)
    std::uint32_t generated = 0;
    jitllm::base::Sha256 rows;
  };
  std::vector<Run> runs(n);
  std::vector<double> draft_ms;
  std::vector<double> verify_ms;
  std::uint64_t waves = 0;
  std::uint64_t verify_captured = 0;
  std::uint64_t verify_replayed = 0;
  std::uint64_t swapped_after_wave = 0;
  std::size_t graphs_kept_at_swap = 0;
  std::uint64_t verify_replayed_after_swap = 0;
  std::uint32_t swap_owed_slots = 0;
  bool expect_swap_replay = false;
  const auto prefilled = InRequest("a wave cohort prefill", [&]() -> Status {
    if (auto r = qwen_.SelectSlots(slots); !r) {
      return r;
    }
    const std::uint32_t chunk = o_.qwen.max_rows;
    for (std::uint32_t i = 0; i < n; ++i) {
      const Prompt& p = *prompts[i];
      std::vector<float> last;
      for (std::uint32_t at = 0; at < p.ids.size(); at += chunk) {
        const auto rows =
            static_cast<std::uint32_t>(std::min<std::size_t>(chunk, p.ids.size() - at));
        if (auto r = slots[i]->Chunk(std::span(p.ids).first(at + rows), at, last, true); !r) {
          return r;
        }
      }
      runs[i].all = p.ids;
      runs[i].all.push_back(Argmax(last));
      runs[i].generated = 1;
    }
    return {};
  });
  if (!prefilled) {
    return prefilled;
  }
  while (true) {
    std::vector<std::uint32_t> active;
    for (std::uint32_t i = 0; i < n; ++i) {
      if (runs[i].generated < o_.tokens) {
        active.push_back(i);
      }
    }
    if (active.empty()) {
      break;
    }
    std::vector<jb::Qwen38Runner::Slot*> selected;
    selected.reserve(active.size());
    for (const std::uint32_t i : active) {
      selected.push_back(slots[i]);
    }
    const auto ran = InRequest("a wave cohort step", [&]() -> Status {
      if (auto r = qwen_.SelectSlots(selected); !r) {
        return r;
      }
      std::vector<std::vector<std::int32_t>> drafts(active.size());
      std::vector<jb::Qwen38Runner::DraftWork> dwork;
      dwork.reserve(active.size());
      for (std::size_t k = 0; k < active.size(); ++k) {
        dwork.push_back({.slot = slots[active[k]],
                         .history = runs[active[k]].all,
                         .drafts = &drafts[k],
                         .probabilities = nullptr,
                         .passes = kDepth});
      }
      auto at = Clock::now();
      if (active.size() > 2) {
        for (std::size_t k = 0; k < active.size(); ++k) {
          if (auto r = slots[active[k]]->Draft(runs[active[k]].all, drafts[k], nullptr, kDepth);
              !r) {
            return r;
          }
        }
      } else if (active.size() > 1) {
        if (auto r = qwen_.DraftWave(dwork); !r) {
          return r;
        }
      } else if (auto r = slots[active[0]]->Draft(runs[active[0]].all, drafts[0], nullptr, kDepth);
                 !r) {
        return r;
      }
      if (active.size() == n) {
        draft_ms.push_back(Seconds(Clock::now() - at) * 1e3);
      }
      std::vector<std::vector<std::int32_t>> inputs(active.size());
      std::vector<std::vector<std::int32_t>> argmax(active.size());
      std::vector<std::vector<float>> logits(active.size());
      std::vector<jb::Qwen38Runner::VerifyWork> vwork;
      for (std::size_t k = 0; k < active.size(); ++k) {
        Run& r = runs[active[k]];
        const auto rows = std::min<std::uint32_t>(
            {static_cast<std::uint32_t>(drafts[k].size()) + 1, o_.tokens - r.generated,
             o_.qwen.context - static_cast<std::uint32_t>(r.all.size() - 1)});
        drafts[k].resize(rows - 1);
        inputs[k] = r.all;
        inputs[k].insert(inputs[k].end(), drafts[k].begin(), drafts[k].end());
        vwork.push_back({.slot = slots[active[k]],
                         .history = inputs[k],
                         .n_past = static_cast<std::uint32_t>(r.all.size() - 1),
                         .argmax = &argmax[k],
                         .logits = &logits[k]});
      }
      const auto graph_before = qwen_.graph_stats();
      at = Clock::now();
      if (active.size() > 1) {
        if (auto r = qwen_.VerifyWave(vwork); !r) {
          return r;
        }
      } else if (auto r = slots[active[0]]->Verify(vwork[0].history, vwork[0].n_past, argmax[0],
                                                   logits.data());
                 !r) {
        return r;
      }
      if (active.size() == n) {
        verify_ms.push_back(Seconds(Clock::now() - at) * 1e3);
      }
      if (active.size() > 1) {
        verify_captured += qwen_.graph_stats().captured - graph_before.captured;
        verify_replayed += qwen_.graph_stats().replayed - graph_before.replayed;
      }
      if (expect_swap_replay) {
        verify_replayed_after_swap = qwen_.graph_stats().replayed - graph_before.replayed;
        if (verify_replayed_after_swap == 0 ||
            qwen_.graph_stats().captured != graph_before.captured) {
          return Error("the first verify wave after the swap did not replay its kept graph");
        }
        expect_swap_replay = false;
      } else if (swapped_after_wave != 0 && active.size() > 1) {
        verify_replayed_after_swap += qwen_.graph_stats().replayed - graph_before.replayed;
      }
      waves += active.size() > 1 ? 1 : 0;
      bool restores_owed = active.size() == n;
      for (std::size_t k = 0; k < active.size(); ++k) {
        Run& r = runs[active[k]];
        r.rows.Update(std::as_bytes(std::span(logits[k])));
        std::uint32_t m = 0;
        while (m < drafts[k].size() && argmax[k][m] == drafts[k][m]) {
          ++m;
        }
        if (auto a = slots[active[k]]->Accept(m + 1); !a) {
          return a;
        }
        r.all.insert(r.all.end(), drafts[k].begin(), drafts[k].begin() + m);
        r.all.push_back(argmax[k][m]);
        r.generated += m + 1;
        const auto& live = slots[active[k]]->live();
        const bool rejected_saved = std::ranges::any_of(live.saved(), [m](const auto& saved) {
          return saved.row >= 0 && std::cmp_greater_equal(saved.row, m + 1);
        });
        restores_owed = restores_owed && m + 1 < argmax[k].size() && rejected_saved &&
                        live.owed() && o_.tokens - r.generated >= kDepth + 1;
      }
      if (with_fp16() && swapped_after_wave == 0 && restores_owed && verify_captured != 0 &&
          qwen_.graph_stats().replayed > graph_before.replayed) {
        swap_owed_slots = n;
      }
      return {};
    });
    if (!ran) {
      return ran;
    }
    if (swap_owed_slots != 0 && swapped_after_wave == 0) {
      // Every completed Accept still owes rejected rows' restore. The
      // request lease is gone: the swap writes all initialized slot state,
      // while the snapshots and the captured graphs keep their places.
      if (node_.InRequest(qwen_.stream())) {
        return Error("the wave swap still holds its request lease");
      }
      const auto state = qwen_.state();
      for (const auto* slot : slots) {
        for (const auto extent : slot->live().extents()) {
          if (std::ranges::find(state, extent) == state.end() ||
              std::ranges::none_of(qwen_.everything().extents, [extent](const auto& member) {
                return member.first == extent;
              })) {
            return Error("the wave swap omits an active slot's state extent");
          }
        }
      }
      graphs_kept_at_swap = qwen_.graphs();
      if (auto out = SwapOut(); !out) {
        return out;
      }
      if (auto in = SwapIn(); !in) {
        return in;
      }
      if (qwen_.graphs() != graphs_kept_at_swap ||
          !std::ranges::all_of(slots, [](const auto* slot) { return slot->live().owed(); })) {
        return Error("the wave swap lost its graphs or settled an owed restore");
      }
      swapped_after_wave = waves;
      expect_swap_replay = true;
    }
  }
  if (with_fp16() && (swapped_after_wave == 0 || expect_swap_replay)) {
    return Error(
        "the wave swap needs a replayed full-cohort verify with rejected rows in every "
        "slot and a subsequent wave");
  }
  if (o_.qwen.graphs && (verify_captured == 0 || verify_replayed == 0)) {
    return Error("the wave check did not capture and replay its verify waves");
  }
  for (std::uint32_t i = 0; i < n; ++i) {
    if (runs[i].generated != o_.tokens ||
        runs[i].all.size() != prompts[i]->ids.size() + o_.tokens) {
      return Error("a wave slot did not produce its requested output budget");
    }
  }
  std::string states;
  if (auto read = InRequest("wave final states",
                            [&]() -> Status {
                              if (auto selected = qwen_.SelectSlots(slots); !selected) {
                                return selected;
                              }
                              for (std::uint32_t i = 0; i < n; ++i) {
                                std::vector<std::byte> target;
                                std::vector<std::byte> drafter;
                                if (auto state = slots[i]->ReadState(target, drafter); !state) {
                                  return state;
                                }
                                jitllm::base::Sha256 hash;
                                hash.Update(target);
                                hash.Update(drafter);
                                states += std::format("{}\"{}\"", i == 0 ? "" : ",",
                                                      jitllm::base::ToHex(hash.Finish()));
                              }
                              return {};
                            });
      !read) {
    return read;
  }
  const auto median = [](std::vector<double> v) {
    if (v.empty()) {
      return 0.0;
    }
    std::ranges::sort(v);
    return v[v.size() / 2];
  };
  std::string tokens;
  std::string rows;
  for (std::uint32_t i = 0; i < n; ++i) {
    jitllm::base::Sha256 hash;
    hash.Update(std::as_bytes(std::span(runs[i].all)));
    tokens += std::format("{}\"{}\"", i == 0 ? "" : ",", jitllm::base::ToHex(hash.Finish()));
    rows += std::format("{}\"{}\"", i == 0 ? "" : ",", jitllm::base::ToHex(runs[i].rows.Finish()));
  }
  std::println("wave check: {} slots, lanes {}: {} waves; draft {:.2f} ms, verify {:.2f} ms", n,
               o_.qwen.wave_lanes ? "on" : "off", waves, median(draft_ms), median(verify_ms));
  results_.push_back(std::format(
      R"({{"check":"wave","slots":{},"lanes":{},"waves":{},"draft_median_ms":{:.3f},)"
      R"("verify_median_ms":{:.3f},"tokens_sha256":[{}],"rows_sha256":[{}],"state_sha256":[{}],)"
      R"("verify_captured":{},"verify_replayed":{},"swapped_after_wave":{},)"
      R"("swap_owed_slots":{},"graphs_kept_at_swap":{},"verify_replayed_after_swap":{}}})",
      n, o_.qwen.wave_lanes ? "true" : "false", waves, median(draft_ms), median(verify_ms), tokens,
      rows, states, verify_captured, verify_replayed, swapped_after_wave, swap_owed_slots,
      graphs_kept_at_swap, verify_replayed_after_swap));
  return {};
}

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
  entered_models_.push_back(&qwen_);
  if (auto r = qwen_.Setup(); !r) {
    return r;
  }
  if (with_fp16()) {
    entered_models_.push_back(&fp16_);
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
  if (o_.check == "greedy" || o_.check == "timing" || o_.check == "vocab-greedy") {
    checked = Greedy();
  } else if (o_.check == "forced") {
    checked = Forced();
  } else if (o_.check == "draft-head") {
    checked = DraftHead();
  } else if (o_.check == "vocab-anchors") {
    checked = VocabAnchors();
  } else if (o_.check == "routed-down" || o_.check == "mxfp8-projection") {
    checked = DraftHead(true);
  } else if (o_.check == "swap") {
    checked = Swap();
  } else if (o_.check == "sampled-plain") {
    checked = Sampled(false);
  } else if (o_.check == "sampled-spec") {
    checked = Sampled(true);
  } else if (o_.check == "wave") {
    checked = Wave();
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
             R"({{"check":"{}","draft_rows":{},"draft_vocab":{},"draft_head_input":"f32","adaptive_depth":{},"window":{},"runtime_prefill":{},"load_seconds":{:.2f},)"
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
    } else if (a == "--template") {
      o.chat_template = v;
    } else if (a == "--stop-metadata") {
      o.stop_metadata = v;
    } else if (a == "--out") {
      o.out = v;
    } else if (a == "--check") {
      o.check = v;
    } else if (a == "--tokens") {
      ok = number(o.tokens) && o.tokens >= 1;
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
    } else if (a == "--slots") {
      ok = number(o.qwen.wave_slots) && o.qwen.wave_slots >= 2 && o.qwen.wave_slots <= 4;
      o.qwen.request_slots = std::max<std::uint32_t>(o.qwen.request_slots, o.qwen.wave_slots);
    } else if (a == "--wave-lanes") {
      o.qwen.wave_lanes = v == "on";
      ok = v == "on" || v == "off";
    } else {
      return Error(std::format("unknown argument {}", a));
    }
    if (!ok) {
      return Error(std::format("{} does not take {}", a, v));
    }
  }
  const bool preparing = o.check == "vocab-prepare";
  if (o.tokens < 2 && o.check != "vocab-anchors") {
    return Error("this check requires at least two tokens or steps");
  }
  if ((!preparing && (o.qwen.artifact.empty() || o.qwen.drafter.empty())) || o.tokenizer.empty() ||
      o.prompts.empty() || o.out.empty() || o.check.empty() ||
      o.fp16.artifact.empty() != o.fp16.tokens.empty()) {
    return Error(
        "usage: jitllm_qwen38_spec --qwen38-artifact DIR --drafter DIR --tokenizer FILE "
        "--prompts FILE --out DIR --check "
        "greedy|timing|forced|swap|sampled-plain|sampled-spec|draft-head|routed-down|mxfp8-"
        "projection|wave "
        "[--reference FILE] [--tokens N] [--context N] [--graphs on|off] [--draft N] "
        "[--draft-vocab N] [--adaptive-depth on|off] "
        "[--runtime-prefill on|off] "
        "[--profile-decode on|off] "
        "[--repeats N] [--margin B] [--seeds N] "
        "[--sampled FILE] [--only ID] "
        "[--poll-us N] [--window P] [--prefill-chunk N] [--slots N (wave: 2-4)] "
        "[--wave-lanes on|off] [--fp16-artifact "
        "DIR --fp16-tokens FILE "
        "--fp16-expect "
        "SHA256]");
  }
  if ((o.check == "wave") != (o.qwen.wave_slots > 1)) {
    return Error("the wave check, and only it, takes --slots 2-4");
  }
  if (preparing && (o.chat_template.empty() || o.stop_metadata.empty())) {
    return Error("vocab-prepare requires --template and --stop-metadata");
  }
  if (o.check == "vocab-greedy") {
    if (o.qwen.context != dv::kContext || o.qwen.max_rows != 8192 || o.tokens != 256 ||
        o.repeats != 2 || o.qwen.draft_rows != dv::kDepth ||
        (o.qwen.draft_vocab != 65536 && o.qwen.draft_vocab != 47172) || o.adaptive_depth ||
        o.window != 0 || o.margin != 1.0 || !o.runtime_prefill || !o.qwen.graphs ||
        o.profile_decode || !o.only.empty() || !o.reference.empty() || !o.fp16.artifact.empty()) {
      return Error(
          "vocab-greedy requires frozen held-out context33792/chunk8192, all eight cells, "
          "256 outputs, two repeats, fixed depth3, head65536/47172, margin1.0 and graphs");
    }
  }
  if (o.check == "vocab-anchors") {
    const bool pilot = o.only == "cal-code-8192" && o.tokens == 1;
    const bool complete = o.only.empty() && o.tokens == 4;
    if (o.qwen.context != dv::kContext || o.qwen.max_rows != 8192 || o.tokens > 4 ||
        o.qwen.draft_rows != dv::kDepth || o.qwen.draft_vocab == 0 || o.qwen.draft_vocab > 65536 ||
        o.adaptive_depth || o.window != 0 || !o.runtime_prefill || !o.reference.empty() ||
        !o.fp16.artifact.empty() || (!pilot && !complete)) {
      return Error(
          "vocab-anchors requires context33792, chunk8192, 1..4 anchors, fixed depth3, "
          "head1..65536, runtime prefill, all eight prepared input cells and no swap");
    }
    o.qwen.draft_head_capture = true;
  }
  if (o.check == "draft-head") {
    if (o.qwen.context > 131072 || o.qwen.max_rows > 8192 || o.tokens > 8 ||
        o.qwen.draft_rows > 3 || o.qwen.draft_vocab == 0 || o.qwen.draft_vocab > 65536 ||
        o.adaptive_depth || o.window != 0) {
      return Error(
          "draft-head requires context<=131072, chunk<=8192, 2..8 steps, depth<=3, "
          "head 1..65536 and fixed depth without a confidence window");
    }
    o.qwen.draft_head_capture = true;
  }
  if (o.check == "routed-down") {
    if (o.qwen.context > 131072 || o.qwen.max_rows > 8192 || o.tokens > 8 ||
        o.qwen.draft_rows != 3 || o.adaptive_depth || o.window != 0) {
      return Error(
          "routed-down requires context<=131072, chunk<=8192, 2..8 steps and fixed depth3");
    }
    o.qwen.routed_capture = 1 | (std::uint64_t{1} << 23) | (std::uint64_t{1} << 47);
  }
  if (o.check == "mxfp8-projection") {
    if (o.qwen.context != 33792 || o.qwen.max_rows != 4096 || o.tokens != 2 ||
        (o.qwen.draft_rows != 2 && o.qwen.draft_rows != 3) || o.adaptive_depth || o.window != 0 ||
        o.profile_decode || !o.runtime_prefill || !o.qwen.graphs || o.qwen.draft_vocab != 47172 ||
        !o.fp16.artifact.empty() || !o.reference.empty()) {
      return Error(
          "mxfp8-projection requires context33792/chunk4096, two steps, fixed depth2/3, "
          "head47172, runtime prefill/graphs and no profile/reference/swap");
    }
    o.qwen.routed_capture = (std::uint64_t{1} << 23) | (std::uint64_t{1} << 47);
  }
  std::error_code directory_error;
  std::filesystem::create_directories(o.out, directory_error);
  if (directory_error) {
    return Error("creating benchmark output directory: " + directory_error.message());
  }
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
  if (options->check == "vocab-prepare") {
    ran = dv::Prepare({.source = options->prompts,
                       .tokenizer = options->tokenizer,
                       .chat_template = options->chat_template,
                       .stop_metadata = options->stop_metadata,
                       .output = options->out});
  } else {
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
