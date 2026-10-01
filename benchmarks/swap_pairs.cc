// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// M3's swap pairs (docs/plan.md, the swap acceptance table;
// docs/experiments/fast-swap/swap.md): two of the three M3 models on one
// paged node — DeepSeek V4 Flash (dsv4_runner.h), Qwen3.8 Flash Next
// (qwen38_runner.h, its n-gram table paged by rows) and the Qwen-Image-2.1
// pipeline (qwen_image_runner.h) — and full swaps A→B→A between them in one
// process, each part of each swap timed. A harness binary, not
// jitllm-runtime (yet: D-088 now clears the native tokenizer it links).
//
//   jitllm_swap_pairs --a dsv4|qwen38|image --b dsv4|qwen38|image --out DIR
//                     [--dsv4-artifact DIR] [--qwen38-artifact DIR]
//                     [--image-store DIR --image-composition ID --image-noise FILE]
//                     [--text FILE] [--qwen38-tokenizer FILE]
//                     [--dsv4-prompt FILE] [--qwen38-prompt FILE]
//                     [--context-tokens N] [--continue N] [--cycles N]
//                     [--zero-context on|off] [--handoff on|off] [--context N]
//                     [--image-expect SHA256] [--release-first on|off]
//                     [--prompts FILE --expect DIR --generate N]
//                     [--poison-probe on|off] [--scrub-probe on|off]
//                     [--graphs on|off] [--bench N] [--poll-us N]
//                     [--spin-ahead-us N]
//
// - An LLM A holds --context-tokens tokens (8,192) of --text, tokenized by
//   the native tokenizer (DeepSeek's from its artifact's GGUF metadata,
//   Qwen3.8's from --qwen38-tokenizer, the checkpoint's tokenizer.json),
//   DeepSeek's BOS first. An LLM B's prompt is the first line of its
//   --*-prompt token file (`name<TAB>ids`); the image's is the fixed teapot
//   prompt at 1,024², 40 steps, from --image-noise's initial latents.
// - Control: A's context prefilled in chunks of its max rows, then
//   --continue tokens decoded greedily (an image A: one full generation,
//   its pixels hashed). Each cycle: A prefills again (compared with the
//   control's, chunk by chunk, and noted where it differs: a model's own
//   nondeterminism, RE-031), its state is saved to the host and hashed, its
//   unswapped continuation run from that state (the cycle's reference) and
//   the state put back; A swaps to B (A's state written back through the
//   zone and A's weights evicted, B's closure paged in, taking A's backing
//   with the handoff); B produces its first output; B swaps back (B
//   evicted; A's state restored, its weights paged in); A's restored state
//   must hash as it left (outside the timed parts), and every continued
//   step's logits must equal the reference's bit for bit (an image A: the
//   generation continues from its first step's output and its pixels must
//   equal the control's).
//   The first cycle is first use (B never ran in the process; A's plans are
//   dropped before it returns), later ones previously prepared. With
//   --zero-context on (an LLM A), a last pair runs at 0 context: A's state
//   is not spilled, and A's first token after it is a 16-token prompt's.
// - Endpoints: an LLM's first generated token (B: its prompt from a cleared
//   state; A: the next token after its context); the image's first
//   denoising step's output (the prompt encoded, step 0 run).
// - Parts of a swap, each from the end of the one before: evict and spill,
//   restore (A's state resident), page-in (the incoming weights: bytes and
//   throughput), setup (DeepSeek's hash-routing check, Qwen3.8's n-gram
//   hash), the first output (Qwen3.8's row reads counted apart). Backing no
//   load took is released after the swap, off the critical path, and timed.
//   Peak memory: the lowest MemAvailable from the swap request to its first
//   output, against the process's start. --release-first waits for that
//   release before the first output instead, and counts the wait.
// - Places (D-090): every model pins its weights' and state's places when
//   it registers them; after each swap the incoming model's are checked
//   still pinned (DeepSeek's also against their registered sources, as
//   its decode graphs name them). --graphs (on by default) runs DeepSeek's
//   and Qwen3.8's decode steps as captured graphs (dsv4_runner.h,
//   qwen38_runner.h); the image runs launch by launch. In a prepared
//   cycle an LLM A's return must replay graphs it captured before the
//   swap (its continuation's logits still equal the reference's).
// - Requests (a lease per request, paged_node.h): each turn is one request
//   that leases its model's whole closure once, and every chunk of it runs
//   as a step under that lease: the controls (an LLM A's prefill and
//   continuation; the image's whole generation), each cycle's prefill with
//   its reference continuation, B's first output, and A's return (the
//   image's rest of the generation, an LLM's continuation). A request's
//   lease is taken within the timed first output. The checks around them
//   (the state saved, restored and hashed; DeepSeek's hash-routing check,
//   Qwen3.8's n-gram hash) run as jobs of their own.
// - --bench N (an LLM A; needs --text): decode speed as llama-bench's tg-N
//   measures it, N one-token greedy steps from an empty context (the
//   context's first token), each step's lease taken per step (a job of its
//   own, as before requests) or held by the request; for DeepSeek and
//   Qwen3.8 each launch by launch and replayed from decode graphs. A warm-up pass per
//   arm, then three passes of each arm in turn; every pass's logits must
//   equal the first warm-up's bit for bit. Per step: the wall, the job's
//   host time, and the device's span of the step's work (CUDA events), so
//   the round trip a step adds is the wall less the device's span.
//   By default the node runs the runtime's own wake (its defaults,
//   docs/experiments/runtime-wake/). --poll-us, a diagnostic, instead has
//   the scheduler and the device lane poll that long after their last
//   progress (RE-017): the figures once measured at 100,000, longer than a
//   decode step, are the harness-polled ones.
// - Diagnostics: --poison-probe fills the shared workspace with 0x00, then
//   0xFF, before each of A's prefill chunks; --scrub-probe fills Qwen3.8's
//   weight extents' unwritten bytes so; a chunk whose logits then differ
//   read bytes it never wrote.
// - --prompts/--expect: the paged LLM A against its resident harness's
//   outputs (a `--prompts` directory of dsv4_exec.cc or qwen38_exec.cc):
//   each prompt from a cleared state, prefill then --generate - 1 greedy
//   steps, every step's logits compared bit for bit; --cycles 0 for this
//   alone. --image-expect: an image A's generation must hash to it.
//
// Exit 1 on any failed check; swap.json in --out has every number.

#include <cuda_runtime.h>
#include <sys/stat.h>

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
#include <ctime>
#include <expected>
#include <filesystem>
#include <format>
#include <fstream>
#include <iterator>
#include <memory>
#include <optional>
#include <print>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include "base/bytes.h"
#include "base/sha256.h"
#include "catalog/catalog.h"
#include "dsv4_common.h"
#include "paged_node.h"
#include "paged_programs.h"
#include "scheduler/scheduler.h"
#include "tokenizer/gguf.h"
#include "tokenizer/hf.h"
#include "tokenizer/tokenizer.h"

namespace {

namespace ts = jitllm::test_support;
namespace sc = jitllm::scheduler;
namespace catalog = jitllm::catalog;
namespace jb = jitllm::benchmarks;
using jitllm::base::Bytes;
using Clock = std::chrono::steady_clock;
using Status = ts::Status;

constexpr int kA = 0;  // owner and stream
constexpr int kB = 1;
constexpr std::uint32_t kShortPrompt = 16;  // A's prompt at 0 context

std::unexpected<std::string> Error(std::string what) { return std::unexpected(std::move(what)); }

double Seconds(Clock::duration d) { return std::chrono::duration<double>(d).count(); }

std::string Sha256(std::span<const std::byte> bytes) {
  jitllm::base::Sha256 hash;
  hash.Update(bytes);
  return jitllm::base::ToHex(hash.Finish());
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

// /proc/loadavg's first three fields (the host is shared: other work shows).
std::string LoadAverage() {
  std::ifstream file("/proc/loadavg");
  std::string one;
  std::string five;
  std::string fifteen;
  file >> one >> five >> fifteen;
  return std::format("{} {} {}", one, five, fifteen);
}

// The lowest MemAvailable seen since the last Reset, sampled every 20 ms,
// and the lowest over the whole run.
class MemorySampler {
 public:
  MemorySampler() : low_(MemAvailable()), all_(low_.load()), thread_([this] { Loop(); }) {}
  MemorySampler(const MemorySampler&) = delete;
  MemorySampler& operator=(const MemorySampler&) = delete;
  MemorySampler(MemorySampler&&) = delete;
  MemorySampler& operator=(MemorySampler&&) = delete;
  ~MemorySampler() {
    stop_ = true;
    thread_.join();
  }
  void Reset() { low_ = MemAvailable(); }
  std::uint64_t low() const { return low_.load(); }
  std::uint64_t all() const { return all_.load(); }

 private:
  static void Lower(std::atomic<std::uint64_t>& to, std::uint64_t now) {
    std::uint64_t seen = to.load();
    while (now < seen && !to.compare_exchange_weak(seen, now)) {
    }
  }
  void Loop() {
    while (!stop_) {
      const std::uint64_t now = MemAvailable();
      Lower(low_, now);
      Lower(all_, now);
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
  }
  std::atomic<std::uint64_t> low_;
  std::atomic<std::uint64_t> all_;
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

// ------------------------------------------------------------------ the models

// What the swaps need of a model, whichever it is.
class Model {
 public:
  Model() = default;
  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;
  Model(Model&&) = delete;
  Model& operator=(Model&&) = delete;
  virtual ~Model() = default;

  virtual std::string_view name() const = 0;
  virtual ts::PagedModel& paged() = 0;
  virtual Status Setup() = 0;
  virtual std::uint64_t activations_needed() const = 0;
  virtual std::uint64_t pool_needed() const = 0;
  virtual Status Register() = 0;
  virtual Status Bind() = 0;
  virtual std::vector<catalog::ExtentId> weights() const = 0;
  virtual std::vector<catalog::ExtentId> state() const { return {}; }
  virtual const catalog::Closure& everything() const = 0;
  virtual std::uint64_t weight_read_bytes() const = 0;
  // Its artifacts' data directories, for the files' ages.
  virtual std::vector<std::filesystem::path> data() const = 0;
  // After a full load: the checks of what the kernels index unchecked.
  virtual Status AfterLoad() { return {}; }
  virtual bool llm() const { return true; }
  // After a swap in: a model's own check of its places, beyond the pins
  // (DeepSeek's against the sources its graphs name).
  virtual Status CheckPlaces() { return {}; }
  virtual std::string violations() const { return {}; }
  virtual std::string extra() const { return "{}"; }  // the model's own counters, JSON
  // Decode graphs (DeepSeek's, D-090): whether it has them, on or off for
  // the next chunks, and how its chunks ran so far.
  virtual bool graphs() const { return false; }
  virtual void set_graphs(bool /*on*/) {}
  virtual jb::Dsv4GraphStats graph_stats() const { return {}; }
  virtual std::size_t graphs_kept() const { return 0; }  // captured and not yet destroyed

  // LLMs.
  virtual Status Clear() { return Error("not an LLM"); }
  virtual Status Chunk(std::span<const std::int32_t> /*history*/, std::uint32_t /*n_past*/,
                       std::vector<float>& /*logits*/) {
    return Error("not an LLM");
  }
  virtual std::uint32_t max_rows() const { return 0; }
  // The state region (an LLM's): where it is and its bytes.
  virtual std::uint64_t state_base() const { return 0; }
  virtual std::uint64_t state_bytes() const { return 0; }
  virtual void DropPlans() {}
  virtual double plan_seconds() const { return 0; }
  // Seconds and bytes the model reads on demand during chunks (Qwen3.8's
  // n-gram rows), so far.
  virtual double demand_seconds() const { return 0; }
  virtual std::uint64_t demand_bytes() const { return 0; }

  // Fills the weights' unwritten bytes (Qwen3.8 only, a diagnostic).
  virtual Status Scrub(std::uint8_t /*value*/, bool /*slabs*/, bool /*dense*/) {
    return Error("no scrub for this model");
  }

  // The image.
  virtual Status FirstOutput(std::string& /*sha*/) { return Error("not the image"); }
  virtual Status Finish(std::string& /*sha*/) { return Error("not the image"); }
};

class Dsv4 final : public Model {
 public:
  Dsv4(ts::PagedNode& node, const jb::Dsv4Options& o, int owner)
      : r_(node, o, owner, static_cast<std::uint32_t>(owner)), o_(o) {}
  std::string_view name() const override { return "DeepSeek V4 Flash"; }
  ts::PagedModel& paged() override { return r_; }
  Status Setup() override { return r_.Setup(); }
  std::uint64_t activations_needed() const override { return r_.activations_needed(); }
  std::uint64_t pool_needed() const override { return r_.pool_needed(); }
  Status Register() override { return r_.Register(); }
  Status Bind() override { return r_.Bind(); }
  std::vector<catalog::ExtentId> weights() const override { return r_.weights(); }
  std::vector<catalog::ExtentId> state() const override { return r_.state(); }
  const catalog::Closure& everything() const override { return r_.everything(); }
  std::uint64_t weight_read_bytes() const override { return r_.weight_read_bytes(); }
  std::vector<std::filesystem::path> data() const override { return {o_.artifact / "data"}; }
  Status AfterLoad() override { return r_.CheckHashRouting(); }
  Status CheckPlaces() override { return r_.CheckPlaces(); }
  bool graphs() const override { return true; }
  void set_graphs(bool on) override { r_.set_graphs(on); }
  jb::Dsv4GraphStats graph_stats() const override { return r_.graph_stats(); }
  std::size_t graphs_kept() const override { return r_.graphs(); }
  std::string violations() const override {
    return r_.coverage_violations() == 0
               ? ""
               : std::format("{} bound tensors outside cataloged extents of their class; first {}",
                             r_.coverage_violations(), r_.first_violation());
  }
  std::string extra() const override {
    const jb::Dsv4GraphStats& g = r_.graph_stats();
    return std::format(
        R"({{"coverage_tensors":{},"slab_padding":{},"state_bytes":{},)"
        R"("graphs":{{"on":{},"eager":{},"captured":{},"replayed":{},"refused":{},"kept":{}}}}})",
        r_.coverage_tensors(), r_.slab_padding(), r_.state_bytes(), o_.graphs ? "true" : "false",
        g.eager, g.captured, g.replayed, g.refused, r_.graphs());
  }
  Status Clear() override { return r_.Clear(); }
  Status Chunk(std::span<const std::int32_t> history, std::uint32_t n_past,
               std::vector<float>& logits) override {
    return r_.Chunk(n_past, history.subspan(n_past), logits);
  }
  std::uint32_t max_rows() const override { return o_.max_rows; }
  std::uint64_t state_base() const override { return r_.state_base(); }
  std::uint64_t state_bytes() const override { return r_.state_bytes(); }
  void DropPlans() override { r_.DropPlans(); }
  double plan_seconds() const override { return r_.plan_seconds(); }

 private:
  jb::Dsv4Runner r_;
  const jb::Dsv4Options& o_;
};

class Qwen38 final : public Model {
 public:
  Qwen38(ts::PagedNode& node, const jb::Qwen38Options& o, int owner)
      : r_(node, o, owner, static_cast<std::uint32_t>(owner)), o_(o) {}
  std::string_view name() const override { return "Qwen3.8 Flash Next"; }
  ts::PagedModel& paged() override { return r_; }
  Status Setup() override { return r_.Setup(); }
  std::uint64_t activations_needed() const override { return r_.activations_needed(); }
  std::uint64_t pool_needed() const override { return r_.pool_needed(); }
  Status Register() override { return r_.Register(); }
  Status Bind() override { return r_.Bind(); }
  std::vector<catalog::ExtentId> weights() const override { return r_.weights(); }
  std::vector<catalog::ExtentId> state() const override { return r_.state(); }
  const catalog::Closure& everything() const override { return r_.everything(); }
  std::uint64_t weight_read_bytes() const override { return r_.weight_read_bytes(); }
  std::vector<std::filesystem::path> data() const override { return {o_.artifact / "data"}; }
  Status AfterLoad() override { return r_.ReadPleHash(); }
  bool graphs() const override { return true; }
  void set_graphs(bool on) override { r_.set_graphs(on); }
  jb::Dsv4GraphStats graph_stats() const override { return r_.graph_stats(); }
  std::size_t graphs_kept() const override { return r_.graphs(); }
  std::string violations() const override {
    return r_.coverage_violations() == 0
               ? ""
               : std::format("{} bound tensors outside cataloged extents of their class; first {}",
                             r_.coverage_violations(), r_.first_violation());
  }
  std::string extra() const override {
    const jb::PleStats& p = r_.ple();
    const jb::Dsv4GraphStats& g = r_.graph_stats();
    return std::format(
        R"({{"coverage_tensors":{},"slab_padding":{},"state_bytes":{},"ple_table_bytes":{},)"
        R"("ple":{{"chunks":{},"lookups":{},"rows":{},"reads":{},"read_bytes":{},)"
        R"("useful_bytes":{},"whole_chunk_bytes":{},"seconds":{:.6f}}},)"
        R"("graphs":{{"on":{},"eager":{},"captured":{},"replayed":{},"refused":{},"kept":{}}}}})",
        r_.coverage_tensors(), r_.slab_padding(), r_.state_bytes(), r_.table_bytes(), p.chunks,
        p.lookups, p.rows, p.reads, p.read_bytes, p.useful_bytes, p.extent_bytes, p.seconds,
        o_.graphs ? "true" : "false", g.eager, g.captured, g.replayed, g.refused, r_.graphs());
  }
  Status Clear() override { return r_.Clear(); }
  Status Chunk(std::span<const std::int32_t> history, std::uint32_t n_past,
               std::vector<float>& logits) override {
    return r_.Chunk(history, n_past, logits);
  }
  std::uint32_t max_rows() const override { return o_.max_rows; }
  std::uint64_t state_base() const override { return r_.state_base(); }
  std::uint64_t state_bytes() const override { return r_.state_bytes(); }
  void DropPlans() override { r_.DropPlans(); }
  double plan_seconds() const override { return r_.plan_seconds(); }
  Status Scrub(std::uint8_t value, bool slabs, bool dense) override {
    return r_.Scrub(value, slabs, dense);
  }
  double demand_seconds() const override { return r_.ple().seconds; }
  std::uint64_t demand_bytes() const override { return r_.ple().read_bytes; }

 private:
  jb::Qwen38Runner r_;
  const jb::Qwen38Options& o_;
};

class Image final : public Model {
 public:
  Image(ts::PagedNode& node, const jb::QwenImageOptions& o, int owner)
      : r_(node, o, owner, static_cast<std::uint32_t>(owner)) {}
  std::string_view name() const override { return "Qwen-Image-2.1"; }
  ts::PagedModel& paged() override { return r_; }
  Status Setup() override { return r_.Setup(); }
  std::uint64_t activations_needed() const override { return r_.activations_needed(); }
  std::uint64_t pool_needed() const override { return jb::QwenImageRunner::pool_needed(); }
  Status Register() override { return r_.Register(); }
  Status Bind() override { return r_.Bind(); }
  std::vector<catalog::ExtentId> weights() const override { return r_.weights(); }
  const catalog::Closure& everything() const override { return r_.everything(); }
  std::uint64_t weight_read_bytes() const override { return r_.weight_read_bytes(); }
  std::vector<std::filesystem::path> data() const override { return r_.data(); }
  bool llm() const override { return false; }
  std::string extra() const override { return r_.Report(); }
  Status FirstOutput(std::string& sha) override { return r_.FirstOutput(sha); }
  Status Finish(std::string& sha) override { return r_.Finish(sha); }

 private:
  jb::QwenImageRunner r_;
};

// ------------------------------------------------------------------ options

struct Options {
  std::string a;
  std::string b;
  std::filesystem::path out;
  jb::Dsv4Options dsv4;
  jb::Qwen38Options qwen38;
  jb::QwenImageOptions image;
  std::filesystem::path text;
  std::filesystem::path qwen38_tokenizer;
  std::filesystem::path dsv4_prompt;
  std::filesystem::path qwen38_prompt;
  std::uint32_t context_tokens = 8192;
  std::uint32_t continue_tokens = 16;
  int cycles = 2;
  bool zero_context = true;
  bool handoff = true;
  std::uint32_t context = 8704;
  std::string image_expect;
  bool poison_probe = false;
  bool scrub_probe = false;
  bool release_first = false;
  std::filesystem::path prompts;
  std::filesystem::path expect;
  std::uint32_t generate = 32;
  std::uint32_t bench = 0;
  std::optional<std::uint32_t> poll_us;  // a diagnostic poll window (RE-017); unset: the runtime's
  // A diagnostic: how long before a likely end the device lane and the
  // driver spin (DeviceSettings::spin_ahead); unset: the runtime's.
  std::optional<std::uint32_t> spin_ahead_us;
};

std::expected<Options, std::string> Parse(std::span<char*> args) {
  Options o;
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
    const auto on_off = [&](bool& into) {
      into = v == "on";
      return v == "on" || v == "off";
    };
    bool ok = true;
    if (a == "--a") {
      o.a = v;
    } else if (a == "--b") {
      o.b = v;
    } else if (a == "--out") {
      o.out = v;
    } else if (a == "--dsv4-artifact") {
      o.dsv4.artifact = v;
    } else if (a == "--qwen38-artifact") {
      o.qwen38.artifact = v;
    } else if (a == "--image-store") {
      o.image.store = v;
    } else if (a == "--image-composition") {
      o.image.composition = v;
    } else if (a == "--image-noise") {
      o.image.noise = v;
    } else if (a == "--text") {
      o.text = v;
    } else if (a == "--qwen38-tokenizer") {
      o.qwen38_tokenizer = v;
    } else if (a == "--dsv4-prompt") {
      o.dsv4_prompt = v;
    } else if (a == "--qwen38-prompt") {
      o.qwen38_prompt = v;
    } else if (a == "--context-tokens") {
      ok = number(o.context_tokens) && o.context_tokens > kShortPrompt;
    } else if (a == "--continue") {
      ok = number(o.continue_tokens) && o.continue_tokens >= 1;
    } else if (a == "--cycles") {
      ok = number(o.cycles) && o.cycles >= 0 && o.cycles <= 8;
    } else if (a == "--zero-context") {
      ok = on_off(o.zero_context);
    } else if (a == "--handoff") {
      ok = on_off(o.handoff);
    } else if (a == "--context") {
      ok = number(o.context);
    } else if (a == "--poison-probe") {
      ok = on_off(o.poison_probe);
    } else if (a == "--scrub-probe") {
      ok = on_off(o.scrub_probe);
    } else if (a == "--release-first") {
      ok = on_off(o.release_first);
    } else if (a == "--graphs") {
      ok = on_off(o.dsv4.graphs);
    } else if (a == "--image-expect") {
      o.image_expect = v;
    } else if (a == "--prompts") {
      o.prompts = v;
    } else if (a == "--expect") {
      o.expect = v;
    } else if (a == "--generate") {
      ok = number(o.generate) && o.generate >= 1;
    } else if (a == "--bench") {
      ok = number(o.bench) && o.bench >= 1 && o.bench <= 1024;
    } else if (a == "--poll-us") {
      std::uint32_t us = 0;
      ok = number(us) && us <= 1000000;
      o.poll_us = us;
    } else if (a == "--spin-ahead-us") {
      std::uint32_t us = 0;
      ok = number(us) && us <= 10000;
      o.spin_ahead_us = us;
    } else {
      return Error(std::format("unknown argument {}", a));
    }
    if (!ok) {
      return Error(std::format("{} does not take {}", a, v));
    }
  }
  const auto known = [](const std::string& m) {
    return m == "dsv4" || m == "qwen38" || m == "image";
  };
  const auto needs = [&](const std::string& m) {
    return (m == "dsv4" && o.dsv4.artifact.empty()) ||
           (m == "qwen38" && o.qwen38.artifact.empty()) ||
           (m == "image" &&
            (o.image.store.empty() || o.image.composition.empty() || o.image.noise.empty()));
  };
  if (!known(o.a) || !known(o.b) || o.a == o.b || o.out.empty() || needs(o.a) || needs(o.b) ||
      (o.prompts.empty() != o.expect.empty())) {
    return Error(
        "usage: jitllm_swap_pairs --a dsv4|qwen38|image --b dsv4|qwen38|image --out DIR "
        "[--dsv4-artifact DIR] [--qwen38-artifact DIR] [--image-store DIR "
        "--image-composition ID --image-noise FILE] [--text FILE] [--qwen38-tokenizer FILE] "
        "[--dsv4-prompt FILE] [--qwen38-prompt FILE] [--context-tokens N] [--continue N] "
        "[--cycles N] [--zero-context on|off] [--handoff on|off] [--context N] "
        "[--image-expect SHA256] [--prompts FILE --expect DIR --generate N] "
        "[--graphs on|off] [--bench N] [--poll-us N] [--spin-ahead-us N]");
  }
  const bool a_llm = o.a != "image";
  if (a_llm && (o.cycles > 0 || o.bench > 0) &&
      (o.text.empty() || (o.a == "qwen38" && o.qwen38_tokenizer.empty()))) {
    return Error("an LLM A needs --text (and Qwen3.8 --qwen38-tokenizer)");
  }
  if (o.bench > 0 && (!a_llm || o.bench >= o.context)) {
    return Error("--bench needs an LLM A and a --context that holds its steps");
  }
  if ((o.b == "dsv4" && o.dsv4_prompt.empty() && o.cycles > 0) ||
      (o.b == "qwen38" && o.qwen38_prompt.empty() && o.cycles > 0)) {
    return Error("an LLM B needs its --*-prompt");
  }
  if (a_llm && o.cycles > 0 && o.context_tokens + o.continue_tokens + 1 > o.context) {
    return Error("--context must hold the context and the continuation");
  }
  o.dsv4.context = o.context;
  o.qwen38.context = o.context;
  o.qwen38.graphs = o.dsv4.graphs;
  o.dsv4.out = o.out / "dsv4";
  o.qwen38.out = o.out / "qwen38";
  o.image.out = o.out / "image";
  return o;
}

bool SameBits(std::span<const float> a, std::span<const float> b) {
  return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size_bytes()) == 0;
}

// The first line of a token file (`name<TAB>ids`).
std::expected<std::vector<std::int32_t>, std::string> FirstPrompt(const std::filesystem::path& p) {
  auto lines = jb::ReadTokenLines(p);
  if (!lines) {
    return std::unexpected(lines.error());
  }
  if (lines->empty() || lines->front().ids.empty()) {
    return Error(std::format("{} has no prompt", p.string()));
  }
  return lines->front().ids;
}

// The files of a directory: the oldest and newest change times, in hours
// before now (RE-027: the SSD reads recent writes faster).
std::string Ages(const std::vector<std::filesystem::path>& dirs) {
  const auto now = std::time(nullptr);
  double oldest = 0;
  double newest = 1e30;
  std::size_t files = 0;
  for (const std::filesystem::path& dir : dirs) {
    std::error_code ec;
    for (const auto& entry : std::filesystem::recursive_directory_iterator(dir, ec)) {
      struct stat st{};
      if (!entry.is_regular_file() || ::stat(entry.path().c_str(), &st) != 0) {
        continue;
      }
      const double hours = static_cast<double>(now - st.st_ctim.tv_sec) / 3600.0;
      oldest = std::max(oldest, hours);
      newest = std::min(newest, hours);
      ++files;
    }
  }
  return std::format(R"({{"files":{},"oldest_hours":{:.2f},"newest_hours":{:.2f}}})", files, oldest,
                     files == 0 ? 0.0 : newest);
}

// ------------------------------------------------------------------ the swaps

// One swap's parts, in seconds.
struct SwapTimes {
  std::string name;
  double evict = 0;         // the request to every outgoing extent evicted (or parked)
  double restore = 0;       // then A's state resident again (0 without state)
  double page_in = 0;       // then the incoming weights resident
  double setup = 0;         // then the incoming model's checks
  double first = 0;         // then the first output
  double total = 0;         // the request to the first output
  double release = 0;       // after the swap: backing no load took, released
  double demand = 0;        // within the first output: rows read on demand (Qwen3.8)
  double plan_seconds = 0;  // within the first output: planning
  std::uint64_t evicted = 0;
  std::uint64_t loaded = 0;
  std::uint64_t read_bytes = 0;  // weights and state paged in
  std::uint64_t demand_bytes = 0;
  std::uint64_t spilled_bytes = 0;
  std::uint64_t handed_off = 0;
  std::uint64_t parked = 0;
  std::uint64_t released_unused = 0;
  std::uint64_t peak_bytes = 0;  // in use at the swap's lowest MemAvailable
  bool exact = true;
  // An LLM A's return: its state after the restore equal byte for byte to
  // the state it left with (SHA-256), and its continuation equal to the
  // never-swapped control's (informational: the control's prefill may
  // differ where a model is not deterministic, RE-031).
  bool state_exact = true;
  bool control_exact = true;
  double digest = 0;  // hashing the restored state, outside the parts above
  // With --release-first: waiting, after the page-in, for backing no load
  // took to be released before the first output (on the critical path).
  double release_wait = 0;
  std::string output;  // B's first output's hash
  // Decode graphs (D-090) over an LLM A's return: the graphs A kept through
  // the swap, and its steps after it replayed from them, newly captured,
  // or launched step by step. In a prepared cycle every return must replay
  // graphs captured before the swap, with the continuation still exact.
  std::uint64_t graphs_kept = 0;
  std::uint64_t graph_replayed = 0;
  std::uint64_t graph_captured = 0;
  std::uint64_t graph_eager = 0;
};

class Swapper {
 public:
  explicit Swapper(const Options& options)
      : o_(options),
        times_(std::size_t{1} << 17U),
        node_({.compute_streams = 2,
               .slots = ts::kPagedSlots,
               .inline_lanes = false,
               .coalesce = false,
               .copy_lane = true,
               .slot_bytes = jb::kSlabSlotBytes,
               .observer = &times_,
               .poll_window = options.poll_us
                                  ? std::optional(std::chrono::microseconds(*options.poll_us))
                                  : std::nullopt,
               .spin_ahead = options.spin_ahead_us
                                 ? std::optional(std::chrono::microseconds(*options.spin_ahead_us))
                                 : std::nullopt}) {
    a_ = Make(o_.a, kA);
    b_ = Make(o_.b, kB);
  }

  Swapper(const Swapper&) = delete;
  Swapper& operator=(const Swapper&) = delete;
  Swapper(Swapper&&) = delete;
  Swapper& operator=(Swapper&&) = delete;
  ~Swapper() {
    if (snapshot_ != nullptr) {
      (void)cudaFreeHost(snapshot_);
    }
  }

  Status Run();
  Status TearDown() { return node_.TearDown(entered_models_); }

 private:
  std::unique_ptr<Model> Make(const std::string& which, int owner) {
    if (which == "dsv4") {
      return std::make_unique<Dsv4>(node_, o_.dsv4, owner);
    }
    if (which == "qwen38") {
      return std::make_unique<Qwen38>(node_, o_.qwen38, owner);
    }
    return std::make_unique<Image>(node_, o_.image, owner);
  }
  Status Tokenize();
  Status Prompts();
  // A request of `m` (a turn): its closure leased once, until End.
  Status Begin(Model& m) {
    return node_.BeginRequest(m.paged().stream(), m.everything(),
                              std::format("{}'s request", m.name()));
  }
  Status End(Model& m) { return node_.EndRequest(m.paged().stream()); }
  // --bench: decode, per-step leases against a request's (header).
  Status Bench();
  // A diagnostic: A's prefill twice, the shared workspace filled with 0x00
  // then 0xFF (NaN in every float type) before each chunk; each chunk's
  // logits compared. A difference is a read of workspace bytes the chunk
  // never wrote.
  Status PoisonProbe();
  Status Fill(std::uint8_t value);
  // A diagnostic: A's prefill with the weights' unwritten bytes zeroed,
  // then with the slab pages' or the dense chunks' filled with 0xFF; each
  // chunk's logits compared. A difference is a kernel reading them.
  Status ScrubProbe();
  Status ProbePrefill(std::vector<std::vector<float>>& chunks);
  // A's context in chunks of its max rows: the last chunk's logits, and
  // with `chunks` every chunk's.
  Status Prefill(std::vector<float>& logits, std::vector<std::vector<float>>* chunks = nullptr);
  Status Decode(std::uint32_t steps, std::vector<std::vector<float>>& logits);
  // A's whole state copied to (save) or from the host snapshot, by a job
  // leasing it; and the snapshot's SHA-256.
  Status CopyState(bool save);
  std::string SnapshotDigest() const;
  Status SwapToB(SwapTimes& t, bool with_state);
  Status SwapToA(SwapTimes& t, bool with_state, bool continuing);
  Status Settle(SwapTimes& t, const sc::SchedulerStats& before, Clock::time_point swapped);
  // After a swap into `m`: its places still pinned where it registered them
  // (D-090), and its own check; a failure is a problem, not an abort.
  Status CheckPlaces(Model& m, const SwapTimes& t);
  // Until no eviction is left: backing no load took, released.
  Status WaitReleased();
  Status Write();

  const Options& o_;
  ResidentTimes times_;
  MemorySampler memory_;
  std::uint64_t available_before_ = MemAvailable();
  std::string load_before_ = LoadAverage();
  ts::PagedNode node_;
  std::unique_ptr<Model> a_;
  std::unique_ptr<Model> b_;
  std::vector<ts::PagedModel*> entered_models_;

  std::vector<std::int32_t> context_;  // A's context tokens
  std::vector<std::int32_t> history_;  // A's sequence so far
  std::string text_sha256_;
  std::string context_sha256_;
  std::vector<std::int32_t> b_prompt_;
  std::vector<float> control_prefill_;
  std::vector<std::vector<float>> control_chunks_;
  std::vector<std::vector<float>> control_steps_;
  std::vector<std::int32_t> control_tokens_;
  std::string control_pixels_;
  std::string continuation_;
  double initial_load_ = 0;
  double control_prefill_seconds_ = 0;
  double control_decode_seconds_ = 0;
  std::vector<std::string> b_outputs_;
  std::vector<SwapTimes> swaps_;
  std::string prompts_;
  std::string bench_;
  std::vector<std::string> problems_;
  // Determinism notes: a cycle's prefill against the control's.
  std::vector<std::string> notes_;
  // Each cycle's own unswapped continuation, from its prefill's state
  // (snapshotted, continued, restored), and that state's digest.
  void* snapshot_ = nullptr;  // pinned, A's state bytes
  std::vector<std::vector<float>> ref_steps_;
  std::vector<std::int32_t> ref_tokens_;
  std::string ref_digest_;
};

Status Swapper::CopyState(bool save) {
  const std::uint64_t base = a_->state_base();
  const std::uint64_t bytes = a_->state_bytes();
  void* host = snapshot_;
  return node_.Job(
      a_->paged().fence_closure(),
      [base, bytes, host, save](jitllm::providers::NativeStream native) {
        auto* const s = static_cast<cudaStream_t>(native.handle);
        auto* device = reinterpret_cast<void*>(base);  // NOLINT(performance-no-int-to-ptr)
        const cudaError_t r = save
                                  ? cudaMemcpyAsync(host, device, bytes, cudaMemcpyDeviceToHost, s)
                                  : cudaMemcpyAsync(device, host, bytes, cudaMemcpyHostToDevice, s);
        return r == cudaSuccess ? sc::JobResult::kQueued : sc::JobResult::kUnknown;
      },
      save ? "saving A's state" : "restoring A's state", kA);
}

std::string Swapper::SnapshotDigest() const {
  return Sha256(std::span(static_cast<const std::byte*>(snapshot_), a_->state_bytes()));
}

Status Swapper::Tokenize() {
  std::ifstream file(o_.text, std::ios::binary);
  const std::string text{std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
  if (text.empty()) {
    return Error(std::format("{} is empty or unreadable", o_.text.string()));
  }
  text_sha256_ = Sha256(std::as_bytes(std::span(text)));
  std::expected<jitllm::tokenizer::TokenizerSpec, jitllm::tokenizer::Error> spec;
  if (o_.a == "dsv4") {
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
    auto read = jitllm::tokenizer::ReadGgufTokenizer(std::as_bytes(std::span(header)));
    if (!read) {
      return Error(std::format("{}: {}", meta.string(), read.error().ToString()));
    }
    spec = std::move(read->spec);
  } else {
    std::ifstream json(o_.qwen38_tokenizer, std::ios::binary);
    const std::string content{std::istreambuf_iterator<char>(json),
                              std::istreambuf_iterator<char>()};
    spec = jitllm::tokenizer::ReadHfTokenizer(content);
  }
  if (!spec) {
    return Error(spec.error().ToString());
  }
  auto tokenizer = jitllm::tokenizer::Tokenizer::Create(std::move(*spec));
  if (!tokenizer) {
    return Error(tokenizer.error().ToString());
  }
  std::vector<jitllm::tokenizer::TokenId> ids;
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
  return {};
}

Status Swapper::Prefill(std::vector<float>& logits, std::vector<std::vector<float>>* chunks) {
  const std::uint32_t rows = a_->max_rows();
  history_.clear();
  for (std::uint32_t at = 0; at < context_.size(); at += rows) {
    const auto n = static_cast<std::uint32_t>(std::min<std::size_t>(rows, context_.size() - at));
    history_.insert(history_.end(), context_.begin() + at, context_.begin() + at + n);
    if (auto r = a_->Chunk(history_, at, logits); !r) {
      return Error(std::format("A's prefill at {}: {}", at, r.error()));
    }
    if (chunks != nullptr) {
      chunks->push_back(logits);
    }
  }
  return {};
}

// A's greedy decode of `steps` tokens after history_, whose next token is
// the argmax of the last logits: each step's logits appended, history_
// grown.
Status Swapper::Decode(std::uint32_t steps, std::vector<std::vector<float>>& logits) {
  for (std::uint32_t k = 0; k < steps; ++k) {
    const std::int32_t next = jb::Argmax(logits.back());
    history_.push_back(next);
    std::vector<float> row;
    if (auto r = a_->Chunk(history_, static_cast<std::uint32_t>(history_.size() - 1), row); !r) {
      return Error(std::format("A's decode at {}: {}", history_.size() - 1, r.error()));
    }
    logits.push_back(std::move(row));
  }
  return {};
}

Status Swapper::Settle(SwapTimes& t, const sc::SchedulerStats& before, Clock::time_point swapped) {
  if (auto r = WaitReleased(); !r) {
    return r;
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

Status Swapper::CheckPlaces(Model& m, const SwapTimes& t) {
  std::size_t unpinned = 0;
  const std::vector<catalog::ExtentId> managed = m.paged().managed_extents();
  if (auto r = node_.Call(
          [&]() -> Status {
            for (const catalog::ExtentId extent : managed) {
              unpinned += node_.scheduler().PlacePinned(extent) ? 0 : 1;
            }
            return {};
          },
          "checking places");
      !r) {
    return r;
  }
  if (unpinned != 0) {
    problems_.push_back(std::format("{}: {} of {}'s {} extents are not pinned at their places",
                                    t.name, unpinned, m.name(), managed.size()));
  }
  if (auto r = m.CheckPlaces(); !r) {
    problems_.push_back(std::format("{}: {}", t.name, r.error()));
  }
  return {};
}

Status Swapper::WaitReleased() {
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
  return {};
}

Status Swapper::SwapToB(SwapTimes& t, bool with_state) {
  auto before = node_.Stats();
  if (!before) {
    return std::unexpected(before.error());
  }
  std::vector<catalog::ExtentId> out;
  if (with_state) {
    out = a_->state();  // first: its write-backs start first
  }
  const std::vector<catalog::ExtentId> weights = a_->weights();
  out.insert(out.end(), weights.begin(), weights.end());
  ts::SwapReport report;
  const double planned_before = b_->plan_seconds();
  const double demand_before = b_->demand_seconds();
  const std::uint64_t demand_bytes_before = b_->demand_bytes();
  memory_.Reset();
  const auto requested = Clock::now();
  if (auto r = node_.Swap(std::move(out), b_->everything(), o_.handoff, report); !r) {
    return r;
  }
  t.evict = Seconds(report.evicted - requested);
  t.page_in = Seconds(report.loaded - report.evicted);
  t.evicted = report.evictions;
  t.loaded = report.loads;
  t.read_bytes = b_->weight_read_bytes();
  t.spilled_bytes = with_state ? a_->state().size() * ts::kPagedExtent : 0;
  if (o_.release_first) {
    const auto waiting = Clock::now();
    if (auto r = WaitReleased(); !r) {
      return r;
    }
    t.release_wait = Seconds(Clock::now() - waiting);
  }
  if (auto r = b_->AfterLoad(); !r) {
    return r;
  }
  const auto first = Clock::now();
  t.setup = Seconds(first - report.loaded) - t.release_wait;
  // B's request: leased within its first output, ended after it.
  if (auto r = Begin(*b_); !r) {
    return r;
  }
  if (b_->llm()) {
    if (auto r = b_->Clear(); !r) {
      return r;
    }
    std::vector<float> row;
    if (auto r = b_->Chunk(b_prompt_, 0, row); !r) {
      return r;
    }
    t.output = Sha256(std::as_bytes(std::span(row)));
  } else if (auto r = b_->FirstOutput(t.output); !r) {
    return r;
  }
  const auto done = Clock::now();
  if (auto r = End(*b_); !r) {
    return r;
  }
  t.first = Seconds(done - first);
  t.total = Seconds(done - requested);
  t.peak_bytes = available_before_ > memory_.low() ? available_before_ - memory_.low() : 0;
  t.plan_seconds = b_->plan_seconds() - planned_before;
  t.demand = b_->demand_seconds() - demand_before;
  t.demand_bytes = b_->demand_bytes() - demand_bytes_before;
  b_outputs_.push_back(t.output);
  t.exact = t.output == b_outputs_.front();
  if (!t.exact) {
    problems_.push_back(std::format("{}: B's first output ({}) differs from its first run's ({})",
                                    t.name, t.output, b_outputs_.front()));
  }
  if (auto r = Settle(t, *before, report.loaded); !r) {
    return r;
  }
  return CheckPlaces(*b_, t);
}

Status Swapper::SwapToA(SwapTimes& t, bool with_state, bool continuing) {
  auto before = node_.Stats();
  if (!before) {
    return std::unexpected(before.error());
  }
  ts::SwapReport report;
  const double planned_before = a_->plan_seconds();
  const double demand_before = a_->demand_seconds();
  const std::uint64_t demand_bytes_before = a_->demand_bytes();
  memory_.Reset();
  const auto requested = Clock::now();
  if (auto r = node_.Swap(b_->weights(), a_->everything(), o_.handoff, report); !r) {
    return r;
  }
  t.evict = Seconds(report.evicted - requested);
  Clock::time_point restored = report.evicted;
  if (with_state) {
    restored = std::max(restored, times_.Latest(a_->state()));
    t.restore = Seconds(restored - report.evicted);
  }
  t.page_in = Seconds(report.loaded - restored);
  t.evicted = report.evictions;
  t.loaded = report.loads;
  t.read_bytes = a_->weight_read_bytes() + (with_state ? a_->state().size() * ts::kPagedExtent : 0);
  if (o_.release_first) {
    const auto waiting = Clock::now();
    if (auto r = WaitReleased(); !r) {
      return r;
    }
    t.release_wait = Seconds(Clock::now() - waiting);
  }
  if (auto r = a_->AfterLoad(); !r) {
    return r;
  }
  const auto set_up = Clock::now();
  t.setup = Seconds(set_up - report.loaded) - t.release_wait;
  if (a_->llm() && continuing && with_state) {
    // The restored state against the state A left with, byte for byte;
    // outside the timed parts.
    if (auto r = CopyState(true); !r) {
      return r;
    }
    t.state_exact = SnapshotDigest() == ref_digest_;
    if (!t.state_exact) {
      problems_.push_back(
          std::format("{}: A's restored state differs from the state it left with", t.name));
    }
  }
  // The graphs A brought through the swap, and how its steps run after it.
  const jb::Dsv4GraphStats graphs_before = a_->graph_stats();
  t.graphs_kept = a_->graphs_kept();
  const auto first = Clock::now();
  t.digest = Seconds(first - set_up);
  // A's request: leased within its first output, ended after the rest of
  // the turn (the generation, or the continuation).
  if (auto r = Begin(*a_); !r) {
    return r;
  }
  std::vector<std::vector<float>> steps;
  if (!a_->llm()) {
    if (auto r = a_->FirstOutput(t.output); !r) {
      return r;
    }
  } else if (continuing) {
    // The token after the context, as its unswapped continuation fed it.
    history_.resize(context_.size());
    history_.push_back(ref_tokens_.front());
    std::vector<float>& row = steps.emplace_back();
    if (auto r = a_->Chunk(history_, static_cast<std::uint32_t>(context_.size()), row); !r) {
      return r;
    }
  } else {
    // At 0 context: a short prompt from a cleared state.
    if (auto r = a_->Clear(); !r) {
      return r;
    }
    history_.assign(context_.begin(), context_.begin() + kShortPrompt);
    std::vector<float>& row = steps.emplace_back();
    if (auto r = a_->Chunk(history_, 0, row); !r) {
      return r;
    }
  }
  const auto done = Clock::now();
  t.first = Seconds(done - first);
  t.total = Seconds(done - requested) - t.digest;
  t.peak_bytes = available_before_ > memory_.low() ? available_before_ - memory_.low() : 0;
  t.plan_seconds = a_->plan_seconds() - planned_before;
  t.demand = a_->demand_seconds() - demand_before;
  t.demand_bytes = a_->demand_bytes() - demand_bytes_before;
  if (!a_->llm()) {
    // The rest of the generation: its pixels equal to the control's.
    std::string pixels;
    if (auto r = a_->Finish(pixels); !r) {
      return r;
    }
    t.exact = pixels == control_pixels_;
    if (!t.exact) {
      problems_.push_back(
          std::format("{}: the regenerated image ({}) differs from the control's", t.name, pixels));
    }
  } else if (continuing) {
    // The rest of the continuation: every step equal to the same state's
    // unswapped continuation's; against the process's control, noted.
    if (auto r = Decode(o_.continue_tokens - 1, steps); !r) {
      return r;
    }
    std::size_t differing = 0;
    std::size_t from_control = 0;
    for (std::size_t k = 0; k < steps.size(); ++k) {
      differing += SameBits(steps[k], ref_steps_.at(k)) ? 0 : 1;
      from_control += SameBits(steps[k], control_steps_.at(k)) ? 0 : 1;
    }
    const std::vector<std::int32_t> tokens(
        history_.begin() + static_cast<std::ptrdiff_t>(context_.size()), history_.end());
    t.exact = differing == 0 && tokens == ref_tokens_;
    t.control_exact = from_control == 0;
    if (!t.exact) {
      problems_.push_back(
          std::format("{}: {} of {} continued steps' logits differ from the unswapped "
                      "continuation's",
                      t.name, differing, steps.size()));
    }
    if (!t.control_exact) {
      notes_.push_back(std::format("{}: {} of {} continued steps differ from the control's", t.name,
                                   from_control, steps.size()));
    }
  }
  const jb::Dsv4GraphStats graphs_after = a_->graph_stats();
  t.graph_replayed = graphs_after.replayed - graphs_before.replayed;
  t.graph_captured = graphs_after.captured - graphs_before.captured;
  t.graph_eager = graphs_after.eager - graphs_before.eager;
  if (auto r = End(*a_); !r) {
    return r;
  }
  if (auto r = Settle(t, *before, report.loaded); !r) {
    return r;
  }
  return CheckPlaces(*a_, t);
}

Status Swapper::Fill(std::uint8_t value) {
  catalog::Closure closure;
  if (auto r = node_.Call(
          [&]() -> Status {
            std::vector<catalog::ExtentId> all = node_.activations().extents;
            all.insert(all.end(), node_.pool().extents.begin(), node_.pool().extents.end());
            auto c = node_.catalog().ClosureOfExtents(all);
            if (!c) {
              return Error("the workspace's closure");
            }
            closure = std::move(*c);
            return {};
          },
          "the workspace's closure");
      !r) {
    return r;
  }
  const ts::Mapped& a = node_.activations();
  const ts::Mapped& p = node_.pool();
  return node_.Job(
      closure,
      [&](jitllm::providers::NativeStream native) {
        auto* const s = static_cast<cudaStream_t>(native.handle);
        // NOLINTBEGIN(performance-no-int-to-ptr)
        const bool ok =
            cudaMemsetAsync(reinterpret_cast<void*>(a.base), value, a.bytes, s) == cudaSuccess &&
            cudaMemsetAsync(reinterpret_cast<void*>(p.base), value, p.bytes, s) == cudaSuccess;
        // NOLINTEND(performance-no-int-to-ptr)
        return ok ? sc::JobResult::kQueued : sc::JobResult::kUnknown;
      },
      "filling the workspace", kA);
}

Status Swapper::PoisonProbe() {
  if (context_.empty()) {
    if (auto r = Tokenize(); !r) {
      return r;
    }
  }
  std::array<std::vector<std::vector<float>>, 2> runs;
  for (std::size_t run = 0; run < runs.size(); ++run) {
    if (auto r = a_->Clear(); !r) {
      return r;
    }
    history_.clear();
    const std::uint32_t rows = a_->max_rows();
    for (std::uint32_t at = 0; at < context_.size(); at += rows) {
      if (auto r = Fill(run == 0 ? 0x00 : 0xFF); !r) {
        return r;
      }
      const auto n = static_cast<std::uint32_t>(std::min<std::size_t>(rows, context_.size() - at));
      history_.insert(history_.end(), context_.begin() + at, context_.begin() + at + n);
      std::vector<float>& logits = runs.at(run).emplace_back();
      if (auto r = a_->Chunk(history_, at, logits); !r) {
        return r;
      }
    }
  }
  std::size_t differing = 0;
  for (std::size_t k = 0; k < runs[0].size(); ++k) {
    const bool same = SameBits(runs[0][k], runs[1][k]);
    std::size_t nan = 0;
    for (const float v : runs[1][k]) {
      nan += std::isnan(v) ? 1 : 0;
    }
    std::println("poison probe: chunk {} at {}: {} ({} NaN logits)", k, k * a_->max_rows(),
                 same ? "same" : "DIFFERS", nan);
    differing += same ? 0 : 1;
  }
  if (differing != 0) {
    problems_.push_back(
        std::format("poison probe: {} of {} chunks read workspace bytes they never wrote",
                    differing, runs[0].size()));
  }
  return {};
}

Status Swapper::ProbePrefill(std::vector<std::vector<float>>& chunks) {
  if (auto r = a_->Clear(); !r) {
    return r;
  }
  history_.clear();
  const std::uint32_t rows = a_->max_rows();
  for (std::uint32_t at = 0; at < context_.size(); at += rows) {
    const auto n = static_cast<std::uint32_t>(std::min<std::size_t>(rows, context_.size() - at));
    history_.insert(history_.end(), context_.begin() + at, context_.begin() + at + n);
    if (auto r = a_->Chunk(history_, at, chunks.emplace_back()); !r) {
      return r;
    }
  }
  return {};
}

Status Swapper::ScrubProbe() {
  if (context_.empty()) {
    if (auto r = Tokenize(); !r) {
      return r;
    }
  }
  if (auto r = a_->Scrub(0x00, true, true); !r) {
    return r;
  }
  std::vector<std::vector<float>> base;
  if (auto r = ProbePrefill(base); !r) {
    return r;
  }
  for (const auto& [slabs, dense, what] :
       {std::tuple{true, false, "slab pages"}, std::tuple{false, true, "dense chunk tails"}}) {
    if (auto r = a_->Scrub(0xFF, slabs, dense); !r) {
      return r;
    }
    std::vector<std::vector<float>> poisoned;
    if (auto r = ProbePrefill(poisoned); !r) {
      return r;
    }
    if (auto r = a_->Scrub(0x00, true, true); !r) {
      return r;
    }
    std::size_t differing = 0;
    std::string first;
    for (std::size_t k = 0; k < base.size(); ++k) {
      if (!SameBits(base[k], poisoned[k])) {
        ++differing;
        if (first.empty()) {
          first = std::format("chunk {}", k);
        }
      }
    }
    std::println("scrub probe: {} filled with 0xFF: {} of {} chunks differ{}{}", what, differing,
                 base.size(), first.empty() ? "" : ", first ", first);
    if (differing != 0) {
      problems_.push_back(std::format("scrub probe: kernels read the {}' unwritten bytes", what));
    }
  }
  return {};
}

Status Swapper::Prompts() {
  auto lines = jb::ReadTokenLines(o_.prompts);
  if (!lines) {
    return std::unexpected(lines.error());
  }
  for (const jb::TokenLine& prompt : *lines) {
    std::ifstream file(o_.expect / (prompt.name + ".logits.f32"), std::ios::binary);
    std::vector<float> expected;
    std::vector<std::vector<float>> steps;
    if (auto r = Begin(*a_); !r) {  // a request per prompt
      return r;
    }
    if (auto r = a_->Clear(); !r) {
      return r;
    }
    history_ = prompt.ids;
    std::vector<float>& row = steps.emplace_back();
    if (auto r = a_->Chunk(history_, 0, row); !r) {
      return Error(std::format("{}: {}", prompt.name, r.error()));
    }
    if (auto r = Decode(o_.generate - 1, steps); !r) {
      return Error(std::format("{}: {}", prompt.name, r.error()));
    }
    if (auto r = End(*a_); !r) {
      return r;
    }
    const std::size_t vocab = steps.front().size();
    expected.resize(std::size_t{o_.generate} * vocab);
    file.read(reinterpret_cast<char*>(expected.data()),
              static_cast<std::streamsize>(expected.size() * sizeof(float)));
    if (!file) {
      return Error(std::format("no {} steps of expected logits for {}", o_.generate, prompt.name));
    }
    std::uint64_t differing = 0;
    double most = 0;
    std::string ids;
    for (std::uint32_t k = 0; k < o_.generate; ++k) {
      const std::span<const float> want =
          std::span(expected).subspan(std::size_t{k} * vocab, vocab);
      for (std::size_t i = 0; i < vocab; ++i) {
        if (std::bit_cast<std::uint32_t>(steps[k][i]) != std::bit_cast<std::uint32_t>(want[i])) {
          ++differing;
          most = std::max(most, std::fabs(static_cast<double>(steps[k][i]) - want[i]));
        }
      }
      ids += std::format("{}{}", k == 0 ? "" : ",", jb::Argmax(steps[k]));
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

Status Swapper::Bench() {
  if (context_.empty()) {
    if (auto r = Tokenize(); !r) {
      return r;
    }
  }
  const std::uint32_t steps = o_.bench;
  const auto n = static_cast<double>(steps);
  struct Arm {
    std::string name;
    bool request = false;
    bool graphs = false;
  };
  std::vector<Arm> arms = {{.name = "per-step lease", .request = false, .graphs = false},
                           {.name = "request lease", .request = true, .graphs = false}};
  if (a_->graphs()) {
    arms.push_back({.name = "per-step lease, graphs", .request = false, .graphs = true});
    arms.push_back({.name = "request lease, graphs", .request = true, .graphs = true});
  }
  struct Sum {
    double tps = 0;
    ts::StepTimes times;
    std::uint64_t passes = 0;
  };
  std::vector<Sum> sums(arms.size());
  std::vector<std::vector<float>> reference;
  std::string passes;
  const auto pass = [&](std::size_t arm_index, bool counted) -> Status {
    const Arm& arm = arms[arm_index];
    a_->set_graphs(arm.graphs);
    if (arm.request) {
      if (auto r = Begin(*a_); !r) {
        return r;
      }
    }
    if (auto r = a_->Clear(); !r) {
      return r;
    }
    (void)node_.TakeTimes(kA);
    const jb::Dsv4GraphStats before = a_->graph_stats();
    std::vector<std::vector<float>> logits(steps);
    std::vector<std::int32_t> history = {context_.front()};
    const auto start = Clock::now();
    for (std::uint32_t k = 0; k < steps; ++k) {
      if (auto r = a_->Chunk(history, k, logits[k]); !r) {
        return Error(std::format("bench {} step {}: {}", arm.name, k, r.error()));
      }
      history.push_back(jb::Argmax(logits[k]));
    }
    const double seconds = Seconds(Clock::now() - start);
    const ts::StepTimes t = node_.TakeTimes(kA);
    if (arm.request) {
      if (auto r = End(*a_); !r) {
        return r;
      }
    }
    const jb::Dsv4GraphStats after = a_->graph_stats();
    std::size_t differing = 0;
    if (reference.empty()) {
      reference = std::move(logits);
    } else {
      for (std::uint32_t k = 0; k < steps; ++k) {
        differing += SameBits(logits[k], reference[k]) ? 0 : 1;
      }
    }
    if (differing != 0) {
      problems_.push_back(std::format("bench {}: {} of {} steps' logits differ from the first's",
                                      arm.name, differing, steps));
    }
    const double per = t.steps > 0 ? 1e3 / static_cast<double>(t.steps) : 0.0;
    passes += std::format(
        R"({}{{"arm":"{}","counted":{},"seconds":{:.6f},"tokens_per_second":{:.3f},)"
        R"("wall_ms":{:.4f},"dispatch_ms":{:.4f},"job_ms":{:.4f},"after_ms":{:.4f},)"
        R"("device_ms":{:.4f},"round_trip_ms":{:.4f},"eager":{},"captured":{},"replayed":{},)"
        R"("differing_steps":{}}})",
        passes.empty() ? "" : ",\n  ", arm.name, counted ? "true" : "false", seconds, n / seconds,
        t.wall * per, t.dispatch * per, t.job * per, t.after * per, t.device * per,
        (t.wall - t.device) * per, after.eager - before.eager, after.captured - before.captured,
        after.replayed - before.replayed, differing);
    std::println(
        "bench {}{}: {} steps in {:.3f} s ({:.2f} tok/s); per step: wall {:.3f} ms, device "
        "{:.3f}, round trip {:.3f} (dispatch {:.3f}, job {:.3f}, after the job {:.3f}); {} "
        "differ",
        arm.name, counted ? "" : " (warm-up)", steps, seconds, n / seconds, t.wall * per,
        t.device * per, (t.wall - t.device) * per, t.dispatch * per, t.job * per, t.after * per,
        differing);
    if (counted) {
      Sum& sum = sums[arm_index];
      sum.tps += n / seconds;
      sum.times.steps += t.steps;
      sum.times.wall += t.wall;
      sum.times.dispatch += t.dispatch;
      sum.times.job += t.job;
      sum.times.after += t.after;
      sum.times.device += t.device;
      ++sum.passes;
    }
    return {};
  };
  for (std::size_t i = 0; i < arms.size(); ++i) {
    if (auto r = pass(i, false); !r) {
      return r;
    }
  }
  for (int repeat = 0; repeat < 3; ++repeat) {
    for (std::size_t i = 0; i < arms.size(); ++i) {
      if (auto r = pass(i, true); !r) {
        return r;
      }
    }
  }
  a_->set_graphs(o_.dsv4.graphs);  // the same flag for both LLMs
  std::string summary;
  for (std::size_t i = 0; i < arms.size(); ++i) {
    const Sum& s = sums[i];
    const double per = s.times.steps > 0 ? 1e3 / static_cast<double>(s.times.steps) : 0.0;
    const double tps = s.passes > 0 ? s.tps / static_cast<double>(s.passes) : 0.0;
    summary += std::format(
        R"({}{{"arm":"{}","tokens_per_second":{:.3f},"wall_ms":{:.4f},"device_ms":{:.4f},)"
        R"("round_trip_ms":{:.4f},"dispatch_ms":{:.4f},"job_ms":{:.4f},"after_ms":{:.4f}}})",
        summary.empty() ? "" : ",\n  ", arms[i].name, tps, s.times.wall * per, s.times.device * per,
        (s.times.wall - s.times.device) * per, s.times.dispatch * per, s.times.job * per,
        s.times.after * per);
    std::println(
        "bench {}: {:.2f} tok/s (mean of {}); per step wall {:.3f} ms, device {:.3f}, round "
        "trip {:.3f}, job {:.3f}",
        arms[i].name, tps, s.passes, s.times.wall * per, s.times.device * per,
        (s.times.wall - s.times.device) * per, s.times.job * per);
  }
  bench_ = std::format("{{\"steps\":{},\"arms\":[\n  {}],\n \"passes\":[\n  {}]}}", steps, summary,
                       passes);
  return {};
}

Status Swapper::Run() {
  if (a_->llm() && o_.cycles > 0) {
    if (auto r = Tokenize(); !r) {
      return r;
    }
  }
  if (b_->llm() && o_.cycles > 0) {
    auto prompt = FirstPrompt(o_.b == "dsv4" ? o_.dsv4_prompt : o_.qwen38_prompt);
    if (!prompt) {
      return std::unexpected(prompt.error());
    }
    b_prompt_ = std::move(*prompt);
  }
  if (auto r = node_.Open(); !r) {
    return r;
  }
  for (Model* m : {a_.get(), b_.get()}) {
    entered_models_.push_back(&m->paged());
    if (auto r = m->Setup(); !r) {
      return Error(std::format("{}: {}", m->name(), r.error()));
    }
  }
  if (a_->llm() && o_.cycles > 0 &&
      cudaMallocHost(&snapshot_, std::max<std::uint64_t>(a_->state_bytes(), 256)) != cudaSuccess) {
    // The harness's own check buffer (pinned, outside the catalog).
    return Error("pinned memory for A's state snapshot");
  }
  if (auto r = node_.MapWorkspace(std::max(a_->activations_needed(), b_->activations_needed()),
                                  std::max(a_->pool_needed(), b_->pool_needed()));
      !r) {
    return r;
  }
  // B: everything resident now (the zone, both models' own memory, the
  // workspace, the staging) and the larger model's weights. The two do not
  // fit together: a full swap is the only way in.
  const std::uint64_t fixed = node_.catalog().OccupancyOf(node_.domain()).Total().value();
  const std::uint64_t budget =
      fixed + node_.StateCapacity() +
      (std::max(a_->weights().size(), b_->weights().size()) * ts::kPagedExtent);
  if (auto r = node_.Start(Bytes(budget)); !r) {
    return r;
  }
  for (Model* m : {a_.get(), b_.get()}) {
    if (auto r = m->Register(); !r) {
      return Error(std::format("{}: {}", m->name(), r.error()));
    }
  }
  for (Model* m : {a_.get(), b_.get()}) {
    if (auto r = m->Bind(); !r) {
      return Error(std::format("{}: {}", m->name(), r.error()));
    }
  }
  node_.Run();

  std::vector<ts::LoadStats> log;
  auto start = Clock::now();
  if (auto r = node_.Load(a_->weights(), "A's first load", log); !r) {
    return r;
  }
  initial_load_ = Seconds(Clock::now() - start);
  std::println("A ({}) loaded: {} bytes in {:.2f} s ({:.2f} GB/s)", a_->name(),
               a_->weight_read_bytes(), initial_load_,
               static_cast<double>(a_->weight_read_bytes()) / initial_load_ / 1e9);
  if (auto r = a_->AfterLoad(); !r) {
    return r;
  }
  if (o_.poison_probe) {
    if (auto r = PoisonProbe(); !r) {
      return r;
    }
  }
  if (o_.scrub_probe) {
    if (auto r = ScrubProbe(); !r) {
      return r;
    }
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
  if (!a_->llm() && (o_.cycles > 0 || !o_.image_expect.empty())) {
    // The control: one full generation, one request.
    start = Clock::now();
    std::string first;
    if (auto r = Begin(*a_); !r) {
      return r;
    }
    if (auto r = a_->FirstOutput(first); !r) {
      return r;
    }
    if (auto r = a_->Finish(control_pixels_); !r) {
      return r;
    }
    if (auto r = End(*a_); !r) {
      return r;
    }
    control_prefill_seconds_ = Seconds(Clock::now() - start);
    std::println("control: the image generated in {:.2f} s, pixels {}", control_prefill_seconds_,
                 control_pixels_);
    if (!o_.image_expect.empty() && control_pixels_ != o_.image_expect) {
      problems_.push_back(std::format("the image's pixels ({}) are not the resident harness's ({})",
                                      control_pixels_, o_.image_expect));
    }
  }
  if (a_->llm() && o_.cycles > 0) {
    // The control: A's context, then its continuation, never swapped; one
    // request.
    if (auto r = Begin(*a_); !r) {
      return r;
    }
    if (auto r = a_->Clear(); !r) {
      return r;
    }
    start = Clock::now();
    if (auto r = Prefill(control_prefill_, &control_chunks_); !r) {
      return r;
    }
    control_prefill_seconds_ = Seconds(Clock::now() - start);
    start = Clock::now();
    control_steps_.push_back(control_prefill_);
    if (auto r = Decode(o_.continue_tokens, control_steps_); !r) {
      return r;
    }
    control_decode_seconds_ = Seconds(Clock::now() - start);
    if (auto r = End(*a_); !r) {
      return r;
    }
    // control_steps_[0] is the prefill's; the continued steps follow.
    control_steps_.erase(control_steps_.begin());
    control_tokens_.assign(history_.begin() + static_cast<std::ptrdiff_t>(context_.size()),
                           history_.end());
    std::println("control: {} context tokens prefilled in {:.2f} s, {} decoded in {:.2f} s",
                 context_.size(), control_prefill_seconds_, o_.continue_tokens,
                 control_decode_seconds_);
  }
  for (int cycle = 0; cycle < o_.cycles; ++cycle) {
    const std::string use = cycle == 0 ? "first use" : "prepared";
    const std::string held =
        a_->llm() ? std::format("{} context tokens", context_.size()) : std::string("image");
    if (a_->llm()) {
      // One request: the prefill, and this state's own continuation.
      if (auto r = Begin(*a_); !r) {
        return r;
      }
      if (auto r = a_->Clear(); !r) {
        return r;
      }
      std::vector<float> prefill;
      std::vector<std::vector<float>> chunks;
      if (auto r = Prefill(prefill, &chunks); !r) {
        return r;
      }
      std::size_t first = chunks.size();
      for (std::size_t k = 0; k < chunks.size() && k < control_chunks_.size(); ++k) {
        if (!SameBits(chunks[k], control_chunks_[k])) {
          first = std::min(first, k);
        }
      }
      if (!SameBits(prefill, control_prefill_) || first != chunks.size()) {
        notes_.push_back(
            std::format("cycle {}: A's prefill differs from the control's, from chunk {} of {}",
                        cycle, first, chunks.size()));
      }
      // This state's own continuation, never swapped: the state saved, the
      // continuation run, the state put back as it was.
      if (auto r = CopyState(true); !r) {
        return r;
      }
      ref_digest_ = SnapshotDigest();
      ref_steps_.assign(1, prefill);
      if (auto r = Decode(o_.continue_tokens, ref_steps_); !r) {
        return r;
      }
      ref_steps_.erase(ref_steps_.begin());
      ref_tokens_.assign(history_.begin() + static_cast<std::ptrdiff_t>(context_.size()),
                         history_.end());
      if (auto r = CopyState(false); !r) {
        return r;
      }
      if (auto r = End(*a_); !r) {
        return r;
      }
      history_.resize(context_.size());
    }
    SwapTimes ab;
    ab.name = std::format("A→B, {} ({})", use, held);
    if (auto r = SwapToB(ab, a_->llm()); !r) {
      return r;
    }
    swaps_.push_back(ab);
    if (cycle == 0) {
      a_->DropPlans();  // A returns to nothing prepared
    }
    SwapTimes ba;
    ba.name = std::format("B→A, {} ({})", use, held);
    if (auto r = SwapToA(ba, a_->llm(), true); !r) {
      return r;
    }
    // Prepared: A kept its plans and graphs through the swap (D-090's pins),
    // so its continuation replays graphs captured before it.
    if (cycle > 0 && a_->llm() && a_->graphs() && o_.dsv4.graphs &&
        (ba.graphs_kept == 0 || ba.graph_replayed == 0)) {
      problems_.push_back(std::format(
          "{}: A replayed no decode graph captured before the swap ({} kept, {} replayed)", ba.name,
          ba.graphs_kept, ba.graph_replayed));
    }
    swaps_.push_back(ba);
  }
  if (a_->llm() && o_.cycles > 0 && o_.zero_context) {
    // At 0 context: A's state is not spilled (it stays resident).
    if (auto r = a_->Clear(); !r) {
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
  for (Model* m : {a_.get(), b_.get()}) {
    if (const std::string v = m->violations(); !v.empty()) {
      problems_.push_back(std::format("{}: {}", m->name(), v));
    }
  }
  return Write();
}

std::string SwapJson(const SwapTimes& t) {
  return std::format(
      R"({{"name":"{}","evict":{:.6f},"restore":{:.6f},"page_in":{:.6f},"setup":{:.6f},)"
      R"("first_output":{:.6f},"total":{:.6f},"release_after":{:.6f},"demand":{:.6f},)"
      R"("plan_seconds":{:.6f},"evicted":{},"loaded":{},"read_bytes":{},"demand_bytes":{},)"
      R"("spilled_bytes":{},"handed_off":{},"parked":{},"released_unused":{},"peak_bytes":{},)"
      R"("exact":{},"state_exact":{},"control_exact":{},"digest_seconds":{:.6f},)"
      R"("release_wait":{:.6f},"output":"{}","graphs":{{"kept":{},"replayed":{},)"
      R"("captured":{},"eager":{}}}}})",
      t.name, t.evict, t.restore, t.page_in, t.setup, t.first, t.total, t.release, t.demand,
      t.plan_seconds, t.evicted, t.loaded, t.read_bytes, t.demand_bytes, t.spilled_bytes,
      t.handed_off, t.parked, t.released_unused, t.peak_bytes, t.exact ? "true" : "false",
      t.state_exact ? "true" : "false", t.control_exact ? "true" : "false", t.digest,
      t.release_wait, t.output, t.graphs_kept, t.graph_replayed, t.graph_captured, t.graph_eager);
}

void Print(const SwapTimes& t) {
  const double gb = static_cast<double>(t.read_bytes) / 1e9;
  std::string_view verdict = "exact";
  if (!t.exact) {
    verdict = "DIFFERS";
  } else if (!t.state_exact) {
    verdict = "STATE DIFFERS";
  }
  std::println(
      "{}: total {:.3f} s = evict {:.3f} + restore {:.3f} + page-in {:.3f} ({:.2f} GB, "
      "{:.2f} GB/s) + release wait {:.3f} + setup {:.3f} + first output {:.3f} (demand reads "
      "{:.3f}, planning {:.3f}); peak {:.1f} GiB; handed off {}, released after {:.3f} s; {}",
      t.name, t.total, t.evict, t.restore, t.page_in, gb,
      t.page_in + t.restore > 0 ? gb / (t.page_in + t.restore) : 0.0, t.release_wait, t.setup,
      t.first, t.demand, t.plan_seconds, static_cast<double>(t.peak_bytes) / (1ULL << 30U),
      t.handed_off, t.release, verdict);
  if (t.graphs_kept + t.graph_replayed + t.graph_captured != 0) {
    std::println(
        "  decode graphs after it: {} kept through the swap; steps {} replayed, {} "
        "captured, {} launch by launch",
        t.graphs_kept, t.graph_replayed, t.graph_captured, t.graph_eager);
  }
}

Status Swapper::Write() {
  std::string swaps;
  for (const SwapTimes& t : swaps_) {
    swaps += (swaps.empty() ? "" : ",\n  ") + SwapJson(t);
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
  std::string notes;
  for (const std::string& note : notes_) {
    notes += std::format(R"({}"{}")", notes.empty() ? "" : ",", note);
    std::println("note: {}", note);
  }
  const std::uint64_t low = memory_.all();
  std::filesystem::create_directories(o_.out);
  std::ofstream file(o_.out / "swap.json");
  file << std::format(
      "{{\"a\":\"{}\",\"b\":\"{}\",\"handoff\":{},\"context\":{},\"text_sha256\":\"{}\","
      "\"context_tokens\":{},\"context_sha256\":\"{}\",\"continue\":{},\"control_tokens\":[{}],"
      "\"control_pixels\":\"{}\",\"initial_load_seconds\":{:.6f},\"a_read_bytes\":{},"
      "\"b_read_bytes\":{},\"a_extents\":{},\"b_extents\":{},\"control_prefill_seconds\":{:.6f},"
      "\"control_decode_seconds\":{:.6f},\"mem_available_before\":{},\"mem_available_low\":{},"
      "\"load_before\":\"{}\",\"load_after\":\"{}\","
      "\"a_files\":{},\"b_files\":{},\n \"a_model\":{},\n \"b_model\":{},\n \"problems\":[{}],\n"
      " \"notes\":[{}],\n \"prompts\":[{}],\n \"bench\":{},\n \"swaps\":[\n  {}]}}\n",
      o_.a, o_.b, o_.handoff ? "true" : "false", o_.context, text_sha256_, context_.size(),
      context_sha256_, o_.continue_tokens, tokens, control_pixels_, initial_load_,
      a_->weight_read_bytes(), b_->weight_read_bytes(), a_->weights().size(), b_->weights().size(),
      control_prefill_seconds_, control_decode_seconds_, available_before_, low, load_before_,
      LoadAverage(), Ages(a_->data()), Ages(b_->data()), a_->extra(), b_->extra(), problems, notes,
      prompts_, bench_.empty() ? "null" : bench_, swaps);
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
