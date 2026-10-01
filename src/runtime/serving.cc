// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "runtime/serving.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <format>
#include <print>
#include <string>
#include <utility>

#include "artifact/artifact.h"
#include "artifact/composition.h"
#include "base/check.h"
#include "base/report.h"
#include "engine/dsv4_plan.h"
#include "engine/dsv4_runner.h"
#include "engine/qwen38_runner.h"
#include "engine/qwen_image_runner.h"
#include "execution/adaptive_depth.h"
#include "model/dsv4.h"
#include "model/qwen38.h"
#include "platform/crash_policy.h"
#include "platform/files.h"
#include "platform/host_probe.h"
#include "platform/path_trust.h"
#include "runtime/memory_guard.h"
#include "runtime/model_limits.h"
#include "runtime/prefill.h"
#include "scheduler/programs.h"
#include "tokenizer/gguf.h"
#include "tokenizer/hf.h"

namespace jitllm::runtime {
namespace {

namespace fs = std::filesystem;
namespace ja = jitllm::artifact;

constexpr std::uint64_t kExtent = engine::kPagedExtent;
// The most a tokenizer or template file may be.
constexpr std::size_t kMaxTokenizerBytes = std::size_t{64} << 20U;
// Extents the page-in observer tracks (the M3 models use about 100,000).
constexpr std::size_t kObservedExtents = std::size_t{1} << 18U;
// Each model's prefill chunk when its prefill_chunk is not configured
// (docs/runtime-serving.md#prefill-chunks-and-cancellation), measured
// through the runtime: doubled from 512 rows while that gained 10% or more
// prefill speed at 8K tokens and the longest chunk stayed within 5 s (a
// chunk is how soon a prefill notices a cancellation); then capped by the
// model at its context (prefill.h).
constexpr std::uint32_t kDsv4PrefillRows = 2048;
constexpr std::uint32_t kQwen38PrefillRows = 4096;

std::unexpected<std::string> Error(std::string what) { return std::unexpected(std::move(what)); }

double Seconds(Clock::duration d) { return std::chrono::duration<double>(d).count(); }

std::string Errno(int error) { return std::strerror(error); }  // NOLINT(concurrency-mt-unsafe)

// A file the configuration names, under its trust rules (D-073): every
// directory on the way and the file itself root's or the runtime user's
// and writable by nobody else; opened without following links, and the
// file the walk approved.
std::expected<std::string, std::string> ReadTrustedFile(const fs::path& path, uid_t trusted,
                                                        std::size_t limit) {
  auto walked = platform::WalkTrusted(path, trusted, false);
  if (!walked) {
    return Error(std::format("{}: {}", path.string(), walked.error()));
  }
  if (!walked->exists) {
    return Error(std::format("{} does not exist", path.string()));
  }
  const int fd =
      ::open(walked->resolved.c_str(), O_RDONLY | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC | O_NOCTTY);
  if (fd < 0) {
    return Error(std::format("{}: {}", path.string(), Errno(errno)));
  }
  struct stat status{};
  std::string bytes;
  std::string problem;
  if (::fstat(fd, &status) != 0) {
    problem = Errno(errno);
  } else if (!S_ISREG(status.st_mode) || status.st_dev != walked->status.st_dev ||
             status.st_ino != walked->status.st_ino) {
    problem = "not the regular file the trust walk approved";
  } else if (platform::OthersCanWrite(status, trusted, fd)) {
    problem = "other users can write it";
  } else if (std::cmp_greater(status.st_size, limit)) {
    problem = std::format("larger than {} bytes", limit);
  } else {
    bytes.resize(static_cast<std::size_t>(status.st_size));
    std::size_t done = 0;
    while (done < bytes.size()) {
      const ssize_t n =
          ::pread(fd, bytes.data() + done, bytes.size() - done, static_cast<off_t>(done));
      if (n < 0 && errno == EINTR) {
        continue;
      }
      if (n <= 0) {
        problem = n < 0 ? Errno(errno) : "shorter than it was";
        break;
      }
      done += static_cast<std::size_t>(n);
    }
  }
  (void)::close(fd);
  if (!problem.empty()) {
    return Error(std::format("{}: {}", path.string(), problem));
  }
  return bytes;
}

// An installed artifact, opened under the store's trust rules (D-056's
// integrity rests on them) and its ID checked; the runner opens it again.
std::expected<ja::Artifact, std::string> OpenTrusted(const fs::path& store, const std::string& id) {
  ja::OpenOptions options;
  options.expected_id = id;
  options.trusted_owner = ::geteuid();
  auto opened = ja::Artifact::Open(store / id, options);
  if (!opened) {
    return Error(std::format("artifact {}: {}", id, opened.error().ToString()));
  }
  return std::move(*opened);
}

std::string GraphJson(const GraphCounts& g) {
  return std::format(R"({{"eager":{},"captured":{},"replayed":{},"refused":{},"kept":{}}})",
                     g.eager, g.captured, g.replayed, g.refused, g.kept);
}

GraphCounts Sum(const engine::GraphStats& a, const engine::GraphStats& b, std::size_t kept) {
  return {.eager = a.eager + b.eager,
          .captured = a.captured + b.captured,
          .replayed = a.replayed + b.replayed,
          .refused = a.refused + b.refused,
          .kept = kept};
}

// ---------------------------------------------------------------- the models

// DeepSeek V4 Flash (engine/dsv4_runner.h), with DSpark as its drafter.
class Dsv4 final : public Llm {
 public:
  Dsv4(engine::PagedNode& node, const config::ModelEntry& entry, const config::RuntimeRoles& roles,
       bool plain, int index)
      : artifact_id_(entry.artifact.value_or("")),
        drafter_id_(entry.drafter.value_or("")),
        store_(roles.installed),
        runner_(node, options_, index, static_cast<std::uint32_t>(index)) {
    name_ = entry.name;
    node_ = &node;
    checkpoint_directory_ = roles.spill;
    speculate_ = !drafter_id_.empty() && entry.speculation && !plain;
    context_ = entry.context;
    options_.artifact = roles.installed / artifact_id_;
    options_.out = roles.spill;
    options_.context = entry.context;
    configured_rows_ = entry.prefill_chunk;
    max_rows_ = PrefillChunkRows(entry.context, entry.prefill_chunk, kDsv4PrefillRows,
                                 model::Dsv4MostRows(model::Dsv4Flash(), entry.context));
    options_.max_rows = max_rows_;
    options_.graphs = true;
    if (speculate_) {
      options_.drafter = roles.installed / drafter_id_;
    }
  }

  engine::PagedModel& paged() override { return runner_; }

  Status Setup() override {
    auto artifact = OpenTrusted(store_, artifact_id_);
    if (!artifact) {
      return std::unexpected(artifact.error());
    }
    if (artifact->model().architecture != "deepseek4") {
      return Error(std::format("artifact {} is a {}, not DeepSeek V4 (deepseek4)", artifact_id_,
                               artifact->model().architecture));
    }
    auto binding = model::BindDsv4(model::Dsv4Flash(), *artifact);
    if (!binding) {
      return std::unexpected(binding.error());
    }
    options_.frontier_head = Dsv4FrontierHeadForServing(*binding);
    if (speculate_) {
      if (auto drafter = OpenTrusted(store_, drafter_id_); !drafter) {
        return std::unexpected(drafter.error());
      }
    }
    // The tokenizer and template from the artifact's kept GGUF metadata
    // (import rule 7): the first shard's.
    std::string kv;
    for (const ja::ListedFile& f : artifact->files()) {
      if (f.role == ja::FileRole::kSourceMetadata && f.path.ends_with(".kv.gguf") &&
          (kv.empty() || f.path.contains("-00001-of-"))) {
        kv = f.path.substr(5);
      }
    }
    if (kv.empty()) {
      return Error("the artifact keeps no GGUF metadata, so no tokenizer");
    }
    auto bytes = artifact->ReadMetadata(kv);
    if (!bytes) {
      return Error(std::format("{}: {}", kv, bytes.error().ToString()));
    }
    auto read = tokenizer::ReadGgufTokenizer(std::as_bytes(std::span(*bytes)));
    if (!read) {
      return Error(std::format("{}: {}", kv, read.error().ToString()));
    }
    auto created = tokenizer::Tokenizer::Create(std::move(read->spec));
    if (!created) {
      return Error(std::format("{}: {}", kv, created.error().ToString()));
    }
    tokenizer_ = std::make_unique<tokenizer::Tokenizer>(std::move(*created));
    FindThinkTokens();
    // A model is refused here, like Qwen3.8, when it could serve no turn.
    if (!read->has_chat_template) {
      return Error(std::format("{} keeps no chat template, so it has no chat turns", kv));
    }
    auto found = chat::FindTemplateForText(read->chat_template);
    if (!found) {
      return std::unexpected(found.error());
    }
    template_ = *found;
    auto stops = chat::StopTokens(template_->stop, *tokenizer_);
    if (!stops) {
      return Error(stops.error().ToString());
    }
    stops_.assign(stops->begin(), stops->end());
    if (tokenizer_->eos() && std::ranges::find(stops_, *tokenizer_->eos()) == stops_.end()) {
      stops_.push_back(*tokenizer_->eos());
    }
    return runner_.Setup();
  }
  std::uint64_t activations_needed() const override { return runner_.activations_needed(); }
  std::uint64_t pool_needed() const override { return runner_.pool_needed(); }
  std::uint64_t host_input_bytes() const override { return runner_.host_input_bytes(); }
  Status Register() override { return runner_.Register(); }
  Status Bind() override { return runner_.Bind(); }
  std::vector<catalog::ExtentId> weights() const override { return runner_.weights(); }
  std::vector<catalog::ExtentId> state() const override { return runner_.state(); }
  const catalog::Closure& everything() const override { return runner_.everything(); }
  std::uint64_t weight_read_bytes() const override { return runner_.weight_read_bytes(); }
  Status AfterLoad() override { return runner_.CheckHashRouting(); }
  Status CheckPlaces() override { return runner_.CheckPlaces(); }
  void DropPlans() override { runner_.DropPlans(); }
  double plan_seconds() const override { return runner_.plan_seconds(); }
  GraphCounts graphs() const override {
    return Sum(runner_.graph_stats(), runner_.draft_stats(), runner_.graphs());
  }
  std::string violations() const override {
    return runner_.coverage_violations() == 0
               ? std::string()
               : std::format("{} bound tensors outside cataloged extents of their class; first {}",
                             runner_.coverage_violations(), runner_.first_violation());
  }
  std::string extra() const override {
    return std::format(
        R"({{"turn_checkpoints":{},"turn_checkpoint_bytes":{},"used_state_bytes":{},"architecture":"deepseek4","speculation":{},"coverage_tensors":{},)"
        R"("slab_padding":{},"state_bytes":{},"drafter_state_bytes":{},"graphs":{}}})",
        turn_checkpoints(), turn_checkpoint_bytes(), runner_.used_state_bytes(),
        speculate_ ? "\"dspark\"" : "null", runner_.coverage_tensors(), runner_.slab_padding(),
        runner_.state_bytes(), runner_.drafter_state_bytes(), GraphJson(graphs()));
  }
  void Defaults(chat::Conversation& c) const override {
    // As llama-server's /apply-template rendered the references' prompts:
    // its default turns the template's thinking on.
    if (!c.enable_thinking) {
      c.enable_thinking = true;
    }
  }

 protected:
  Status RunChunk(std::span<const std::int32_t> all, std::uint32_t n_past, bool inject,
                  std::vector<float>& logits) override {
    return runner_.Chunk(n_past, all.subspan(n_past), logits, {},
                         inject ? engine::Dsv4ChunkKind::kInject : engine::Dsv4ChunkKind::kPlain);
  }
  Status SpecStep(std::span<const std::int32_t> all, std::uint32_t pos, std::uint32_t left,
                  std::vector<std::int32_t>& kept, std::vector<std::vector<float>>* logits,
                  std::uint64_t& drafted) override {
    // The anchor and its drafts, within the tokens left, the verify's
    // bound, the context and the steps' mask widths (D-092).
    auto rows = std::min<std::uint32_t>(
        {options_.draft_rows + 1, options_.max_verify, left, options_.context - pos});
    while (rows > 1 && !model::Dsv4SameWidths(runner_.state_layout(), pos, rows)) {
      --rows;
    }
    std::vector<std::int32_t> drafts;
    std::vector<float> verified;
    if (auto r = runner_.DraftVerify(pos, all.back(), rows, drafts, verified); !r) {
      return r;
    }
    drafts.resize(rows - 1);
    const std::uint32_t vocab = runner_.vocab();
    const auto row = [&](std::uint32_t i) {
      return std::span<const float>(verified).subspan(std::size_t{i} * vocab, vocab);
    };
    // Accept drafts while the target agrees (greedy: its argmax; sampling:
    // speculative sampling's verdict); the first disagreement, or the row
    // after the last draft, gives the next token. Row m predicts the token
    // at pos + m + 1.
    std::uint32_t m = 0;
    std::int32_t next = -1;
    for (; m < rows - 1; ++m) {
      std::int32_t instead = -1;
      auto keep = Keep(row(m), drafts[m], std::uint64_t{pos} + m + 1, instead);
      if (!keep) {
        return std::unexpected(keep.error());
      }
      if (!*keep) {
        next = instead;
        break;
      }
    }
    if (next < 0) {
      auto chosen = Choose(row(m), std::uint64_t{pos} + m + 1);
      if (!chosen) {
        return std::unexpected(chosen.error());
      }
      next = *chosen;
    }
    if (auto r = runner_.Accept(m + 1); !r) {
      return r;
    }
    kept.assign(drafts.begin(), drafts.begin() + m);
    kept.push_back(next);
    if (logits != nullptr) {
      for (std::uint32_t i = 0; i <= m; ++i) {
        logits->emplace_back(row(i).begin(), row(i).end());
      }
    }
    drafted += rows - 1;
    return {};
  }
  Status Settle() override { return runner_.Rollback(); }
  Status ClearState() override { return runner_.Clear(); }
  std::uint64_t target_state_base() const override { return runner_.state_base(); }
  std::uint64_t used_state_bytes() const override { return runner_.used_state_bytes(); }
  bool StateUsable() const override { return runner_.state_usable(); }
  Status PrepareDecodeState(std::uint32_t pos, std::uint32_t left) override {
    auto rows = speculate_
                    ? std::min({options_.draft_rows + 1, options_.max_verify, left, context_ - pos})
                    : 1U;
    while (rows > 1 && !model::Dsv4SameWidths(runner_.state_layout(), pos, rows)) {
      --rows;
    }
    return runner_.ReserveStateThrough(pos + rows);
  }
  std::vector<engine::LiveState::Range> used_state_ranges() const override {
    return runner_.used_state_ranges();
  }
  Status SaveUsedState(void* host, std::span<const engine::LiveState::Range> ranges) override {
    return runner_.SaveUsedState(host, ranges);
  }
  Status RestoreUsedState(void* host, std::span<const engine::LiveState::Range> ranges) override {
    return runner_.RestoreUsedState(host, ranges);
  }
  std::expected<std::vector<engine::LiveState::Range>, std::string> CheckpointRanges(
      std::uint32_t positions) const override {
    return runner_.CheckpointRanges(positions);
  }
  Status PrepareRestoreState(std::span<const engine::LiveState::Range> footprint) override {
    return runner_.PrepareRestoreState(footprint);
  }
  Status CopyCheckpointState(void* host, std::span<const engine::LiveState::Range> ranges,
                             bool to_host) override {
    return runner_.CopyCheckpointState(host, ranges, to_host);
  }
  std::uint64_t target_state_bytes() const override { return runner_.state_bytes(); }
  std::uint64_t drafter_state_base() const override { return runner_.drafter_state_base(); }
  std::uint64_t drafter_state_bytes() const override { return runner_.drafter_state_bytes(); }

 private:
  std::string artifact_id_;
  std::string drafter_id_;  // empty: none
  fs::path store_;
  engine::Dsv4Options options_;  // before the runner, which keeps a reference
  engine::Dsv4Runner runner_;
};

// Qwen3.8 Flash Next (engine/qwen38_runner.h), with its MTP block as its
// drafter.
class Qwen38 final : public Llm {
 public:
  Qwen38(engine::PagedNode& node, const config::ModelEntry& entry,
         const config::RuntimeRoles& roles, bool plain, int index)
      : artifact_id_(entry.artifact.value_or("")),
        drafter_id_(entry.drafter.value_or("")),
        tokenizer_path_(entry.tokenizer),
        template_path_(entry.chat_template),
        store_(roles.installed),
        runner_(node, options_, index, static_cast<std::uint32_t>(index)) {
    name_ = entry.name;
    node_ = &node;
    speculate_ = !drafter_id_.empty() && entry.speculation && !plain;
    checkpoint_directory_ = roles.spill;
    context_ = entry.context;
    options_.artifact = roles.installed / artifact_id_;
    options_.out = roles.spill;
    options_.context = entry.context;
    configured_rows_ = entry.prefill_chunk;
    // (The runner's fast graph builds no mask of every cell by every row, so
    // RE-037's bound does not cap its chunks.)
    max_rows_ = PrefillChunkRows(entry.context, entry.prefill_chunk, kQwen38PrefillRows,
                                 model::Qwen38MostRows(entry.context, false));
    options_.max_rows = max_rows_;
    options_.graphs = true;
    if (speculate_) {
      options_.drafter = roles.installed / drafter_id_;
      options_.draft_rows = max_rows_ >= 4 ? 3 : 2;
    }
  }

  engine::PagedModel& paged() override { return runner_; }

  Status Setup() override {
    auto artifact = OpenTrusted(store_, artifact_id_);
    if (!artifact) {
      return std::unexpected(artifact.error());
    }
    if (artifact->model().architecture != "qwen4exp") {
      return Error(std::format("artifact {} is a {}, not Qwen3.8 Flash Next (qwen4exp)",
                               artifact_id_, artifact->model().architecture));
    }
    if (speculate_) {
      if (auto drafter = OpenTrusted(store_, drafter_id_); !drafter) {
        return std::unexpected(drafter.error());
      }
    }
    // The tokenizer and template: kept in the artifact's metadata, or (M3's
    // import kept config.json only) the checkpoint's, which the
    // configuration names.
    const auto source = [&](std::string_view kept, const std::optional<fs::path>& named,
                            std::string_view key) -> std::expected<std::string, std::string> {
      auto meta = artifact->ReadMetadata(kept);
      if (meta) {
        return std::move(*meta);
      }
      if (meta.error().rule != ja::Rule::kFileSet) {  // kept, but not as listed
        return Error(std::format("{}: {}", kept, meta.error().ToString()));
      }
      if (!named.has_value()) {
        return Error(
            std::format("the artifact keeps no {} and models.{}.{} is not set", kept, name_, key));
      }
      return ReadTrustedFile(named.value_or(fs::path()), ::geteuid(), kMaxTokenizerBytes);
    };
    auto json = source("tokenizer.json", tokenizer_path_, "tokenizer");
    if (!json) {
      return std::unexpected(json.error());
    }
    auto spec = tokenizer::ReadHfTokenizer(*json);
    if (!spec) {
      return Error("tokenizer: " + spec.error().ToString());
    }
    auto created = tokenizer::Tokenizer::Create(std::move(*spec));
    if (!created) {
      return Error("tokenizer: " + created.error().ToString());
    }
    tokenizer_ = std::make_unique<tokenizer::Tokenizer>(std::move(*created));
    FindThinkTokens();
    auto text = source("chat_template.jinja", template_path_, "chat_template");
    if (!text) {
      return std::unexpected(text.error());
    }
    auto found = chat::FindTemplateForText(*text);
    if (!found) {
      return std::unexpected(found.error());
    }
    template_ = *found;
    auto stops = chat::StopTokens(template_->stop, *tokenizer_);
    if (!stops) {
      return Error(stops.error().ToString());
    }
    stops_.assign(stops->begin(), stops->end());
    if (tokenizer_->eos() && std::ranges::find(stops_, *tokenizer_->eos()) == stops_.end()) {
      stops_.push_back(*tokenizer_->eos());
    }
    return runner_.Setup();
  }
  std::uint64_t activations_needed() const override { return runner_.activations_needed(); }
  std::uint64_t pool_needed() const override { return runner_.pool_needed(); }
  std::uint64_t host_input_bytes() const override { return runner_.host_input_bytes(); }
  Status Register() override { return runner_.Register(); }
  Status Bind() override { return runner_.Bind(); }
  std::vector<catalog::ExtentId> weights() const override { return runner_.weights(); }
  std::vector<catalog::ExtentId> state() const override { return runner_.state(); }
  const catalog::Closure& everything() const override { return runner_.everything(); }
  std::uint64_t weight_read_bytes() const override { return runner_.weight_read_bytes(); }
  Status AfterLoad() override { return runner_.ReadPleHash(); }
  Status CheckPlaces() override { return runner_.CheckPlaces(); }
  void DropPlans() override { runner_.DropPlans(); }
  double plan_seconds() const override { return runner_.plan_seconds(); }
  double demand_seconds() const override { return runner_.ple().seconds; }
  std::uint64_t demand_bytes() const override { return runner_.ple().read_bytes; }
  GraphCounts graphs() const override {
    return Sum(runner_.graph_stats(), runner_.draft_stats(), runner_.graphs());
  }
  std::string violations() const override {
    return runner_.coverage_violations() == 0
               ? std::string()
               : std::format("{} bound tensors outside cataloged extents of their class; first {}",
                             runner_.coverage_violations(), runner_.first_violation());
  }
  std::string extra() const override {
    const engine::PleStats& p = runner_.ple();
    return std::format(
        R"({{"turn_checkpoints":{},"turn_checkpoint_bytes":{},"used_state_bytes":{},"architecture":"qwen4exp","speculation":{},"coverage_tensors":{},)"
        R"("slab_padding":{},"state_bytes":{},"drafter_state_bytes":{},"ple_table_bytes":{},)"
        R"("ple":{{"chunks":{},"lookups":{},"rows":{},"reads":{},"read_bytes":{},)"
        R"("seconds":{:.6f}}},"graphs":{}}})",
        turn_checkpoints(), turn_checkpoint_bytes(), runner_.used_state_bytes(),
        speculate_ ? "\"mtp\"" : "null", runner_.coverage_tensors(), runner_.slab_padding(),
        runner_.state_bytes(), runner_.drafter_state_bytes(), runner_.table_bytes(), p.chunks,
        p.lookups, p.rows, p.reads, p.read_bytes, p.seconds, GraphJson(graphs()));
  }
  void Defaults(chat::Conversation& /*c*/) const override {
    // The template's own defaults (thinking on), as the oracle's /tokenize
    // rendered the references' prompts.
  }
  // A speculative step's drafts must stay inside the context (SpecStep).
  std::uint32_t usable_context() const override {
    return speculate_ ? context_ - runner_.draft_rows() : context_;
  }

 protected:
  Status RunChunk(std::span<const std::int32_t> all, std::uint32_t n_past, bool inject,
                  std::vector<float>& logits) override {
    return runner_.Chunk(all, n_past, logits, inject);
  }
  Status SpecStep(std::span<const std::int32_t> all, std::uint32_t pos, std::uint32_t left,
                  std::vector<std::int32_t>& kept, std::vector<std::vector<float>>* logits,
                  std::uint64_t& drafted) override {
    // Sampling's position keys keep their existing fixed draft schedule.
    // Greedy chooses its depth from deterministic acceptance observations.
    const auto depth = sampling() || runner_.draft_rows() < 3 ? 2U : depth_.Choose();
    if (depth > options_.context - pos) {
      return Error("the drafts would pass the context");
    }
    std::vector<std::int32_t> drafts;
    if (auto r = runner_.Draft(all, drafts, nullptr, depth); !r) {
      return r;
    }
    // The verify: the anchor and its drafts, within the tokens left.
    const auto rows = std::min<std::uint32_t>(
        {static_cast<std::uint32_t>(drafts.size()) + 1, left, options_.context - pos});
    drafts.resize(rows - 1);
    std::vector<std::int32_t> input(all.begin(), all.end());
    input.insert(input.end(), drafts.begin(), drafts.end());
    std::vector<std::int32_t> argmax;
    std::vector<float> verified;
    const bool rows_needed = logits != nullptr || sampling();
    if (auto r = runner_.Verify(input, pos, argmax, rows_needed ? &verified : nullptr); !r) {
      return r;
    }
    const std::uint32_t vocab = runner_.vocab();
    const auto row = [&](std::uint32_t i) {
      return std::span<const float>(verified).subspan(std::size_t{i} * vocab, vocab);
    };
    std::uint32_t m = 0;
    std::int32_t next = -1;
    if (!sampling()) {
      // Greedy: the verify's own argmaxes.
      for (; m < rows - 1; ++m) {
        if (argmax[m] != drafts[m]) {
          next = argmax[m];
          break;
        }
      }
      if (next < 0) {
        next = argmax[m];
      }
    } else {
      // Speculative sampling over the verify's rows; row m predicts the
      // token at pos + m + 1.
      for (; m < rows - 1; ++m) {
        std::int32_t instead = -1;
        auto keep = Keep(row(m), drafts[m], std::uint64_t{pos} + m + 1, instead);
        if (!keep) {
          return std::unexpected(keep.error());
        }
        if (!*keep) {
          next = instead;
          break;
        }
      }
      if (next < 0) {
        auto chosen = Choose(row(m), std::uint64_t{pos} + m + 1);
        if (!chosen) {
          return std::unexpected(chosen.error());
        }
        next = *chosen;
      }
    }
    if (auto r = runner_.Accept(m + 1); !r) {
      return r;
    }
    kept.assign(drafts.begin(), drafts.begin() + m);
    kept.push_back(next);
    if (logits != nullptr) {
      for (std::uint32_t i = 0; i <= m; ++i) {
        const std::span<const float> kept_row = row(i);
        logits->emplace_back(kept_row.begin(), kept_row.end());
      }
    }
    drafted += rows - 1;
    if (!sampling()) {
      depth_.Observe(depth, m + 1, rows == depth + 1);
    }
    return {};
  }
  Status Settle() override { return runner_.Rollback(); }
  Status ClearState() override {
    depth_ = execution::AdaptiveDepth(3);
    return runner_.Clear();
  }
  std::uint64_t target_state_base() const override { return runner_.state_base(); }
  std::uint64_t used_state_bytes() const override { return runner_.used_state_bytes(); }
  bool StateUsable() const override { return runner_.state_usable(); }
  Status PrepareDecodeState(std::uint32_t pos, std::uint32_t left) override {
    if (!speculate_) {
      return runner_.ReserveStateThrough(pos + 1);
    }
    const auto depth = sampling() || runner_.draft_rows() < 3 ? 2U : depth_.Choose();
    if (depth > context_ - pos) {
      return Error("the drafts would pass the context");
    }
    const auto verify_rows = std::min({depth + 1, left, context_ - pos});
    // Draft reprocesses tokens ending before the anchor; its final pass
    // ends at pos + depth - 1. Verify includes the anchor itself.
    return runner_.ReserveStateThrough(pos + std::max(depth - 1, verify_rows));
  }
  std::vector<engine::LiveState::Range> used_state_ranges() const override {
    return runner_.used_state_ranges();
  }
  Status SaveUsedState(void* host, std::span<const engine::LiveState::Range> ranges) override {
    return runner_.SaveUsedState(host, ranges);
  }
  Status RestoreUsedState(void* host, std::span<const engine::LiveState::Range> ranges) override {
    return runner_.RestoreUsedState(host, ranges);
  }
  std::uint64_t target_state_bytes() const override { return runner_.state_bytes(); }
  std::expected<std::vector<engine::LiveState::Range>, std::string> CheckpointRanges(
      std::uint32_t positions) const override {
    return runner_.CheckpointRanges(positions);
  }
  Status PrepareRestoreState(std::span<const engine::LiveState::Range> footprint) override {
    return runner_.PrepareRestoreState(footprint);
  }
  Status CopyCheckpointState(void* host, std::span<const engine::LiveState::Range> ranges,
                             bool to_host) override {
    return runner_.CopyCheckpointState(host, ranges, to_host);
  }
  std::uint64_t drafter_state_base() const override { return runner_.drafter_state_base(); }
  std::uint64_t drafter_state_bytes() const override { return runner_.drafter_state_bytes(); }
  std::uint32_t cursor() const override { return runner_.pending_rows(); }
  void set_cursor(std::uint32_t value) override { runner_.set_pending_rows(value); }
  void SaveDecodingState() override { saved_depth_ = depth_; }
  void RestoreDecodingState() override { depth_ = saved_depth_; }
  execution::AdaptiveDepth TurnDecodingState() const override { return depth_; }
  void RestoreTurnDecodingState(const execution::AdaptiveDepth& state) override { depth_ = state; }

 private:
  std::string artifact_id_;
  std::string drafter_id_;  // empty: none
  std::optional<fs::path> tokenizer_path_;
  std::optional<fs::path> template_path_;
  fs::path store_;
  engine::Qwen38Options options_;  // before the runner, which keeps a reference
  engine::Qwen38Runner runner_;
  execution::AdaptiveDepth depth_{3};
  execution::AdaptiveDepth saved_depth_{3};
};

// The Qwen-Image-2.1 pipeline (engine/qwen_image_runner.h).
class QwenImage final : public Image {
 public:
  QwenImage(engine::PagedNode& node, const config::ModelEntry& entry,
            const config::RuntimeRoles& roles, const ServingOptions& serving, int index)
      : composition_id_(entry.composition.value_or("")),
        store_(roles.installed),
        runner_(node, options_, index, static_cast<std::uint32_t>(index)) {
    name_ = entry.name;
    options_.store = roles.installed;
    options_.composition = composition_id_;
    options_.noise = serving.image_noise;
    options_.out = roles.spill;
    options_.prompt = serving.image_prompt;
  }

  engine::PagedModel& paged() override { return runner_; }
  Status Setup() override {
    // Every component under the store's trust rules first.
    const fs::path root = store_ / composition_id_;
    auto walked = platform::WalkTrusted(root, ::geteuid(), false);
    if (!walked || !walked->exists || !S_ISDIR(walked->status.st_mode) ||
        platform::OthersCanWrite(walked->status, ::geteuid(), root)) {
      return Error(
          std::format("composition {}: not a directory only root and this user can "
                      "change{}",
                      composition_id_, walked ? "" : ": " + walked.error()));
    }
    auto composition = ja::OpenComposition(root, composition_id_);
    if (!composition) {
      return Error(
          std::format("composition {}: {}", composition_id_, composition.error().ToString()));
    }
    for (const ja::CompositionComponent& c : composition->components()) {
      if (auto opened = OpenTrusted(store_, c.artifact); !opened) {
        return std::unexpected(opened.error());
      }
    }
    return runner_.Setup();
  }
  std::uint64_t activations_needed() const override { return runner_.activations_needed(); }
  std::uint64_t pool_needed() const override { return engine::QwenImageRunner::pool_needed(); }
  Status Register() override { return runner_.Register(); }
  Status Bind() override { return runner_.Bind(); }
  std::vector<catalog::ExtentId> weights() const override { return runner_.weights(); }
  const catalog::Closure& everything() const override { return runner_.everything(); }
  std::uint64_t weight_read_bytes() const override { return runner_.weight_read_bytes(); }
  std::string extra() const override { return runner_.Report(); }
  Status FirstOutput(std::string& sha) override { return runner_.FirstOutput(sha); }
  Status Finish(std::string& sha) override { return runner_.Finish(sha); }

 private:
  std::string composition_id_;
  fs::path store_;
  engine::QwenImageOptions options_;  // before the runner, which keeps a reference
  engine::QwenImageRunner runner_;
};

}  // namespace

// ---------------------------------------------------------------- memory

MemorySampler::MemorySampler() : start_(Available()) {
  low_ = start_;
  all_ = start_;
  thread_ = std::jthread([this] {
    if (auto stack = platform::InstallThreadSignalStack(); !stack) {
      std::println(stderr, "the memory sampler runs without a signal stack: {}", stack.error());
    }
    Loop();
  });
}

MemorySampler::~MemorySampler() {
  stop_ = true;
  if (thread_.joinable()) {
    thread_.join();
  }
}

std::uint64_t MemorySampler::Available() { return platform::AvailableMemoryBytes().value_or(0); }

void MemorySampler::Reset() { low_ = Available(); }

void MemorySampler::Loop() {
  const auto lower = [](std::atomic<std::uint64_t>& to, std::uint64_t now) {
    std::uint64_t seen = to.load();
    while (now < seen && !to.compare_exchange_weak(seen, now)) {
    }
  };
  while (!stop_) {
    const std::uint64_t now = Available();
    if (now != 0) {
      lower(low_, now);
      lower(all_, now);
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
}

void ResidentTimes::Staged(catalog::ExtentId extent, scheduler::PageInEvent event) {
  if (event == scheduler::PageInEvent::kResident && extent.index() < at_.size()) {
    at_[extent.index()] = Clock::now();
  }
}

Clock::time_point ResidentTimes::Latest(std::span<const catalog::ExtentId> extents) const {
  Clock::time_point latest{};
  for (const catalog::ExtentId extent : extents) {
    if (extent.index() < at_.size()) {
      latest = std::max(latest, at_[extent.index()]);
    }
  }
  return latest;
}

// ---------------------------------------------------------------- an LLM

Llm::Llm() : default_branch_(*this) {}

Llm::Branch::~Branch() {
  base::Check(!generation_active_, "a branch with an active generation was destroyed");
}

Status Llm::Branch::Clear() { return model_.Clear(*this); }

void Llm::Branch::Forget() { model_.Forget(*this); }

Status Llm::Branch::Prefill(std::span<const std::int32_t> tokens, std::vector<float>& last,
                            const PrefillGoOn& go_on, PrefillRun* run) {
  return model_.Prefill(*this, tokens, last, go_on, run);
}

Status Llm::Branch::ScorePrompt(
    std::span<const std::int32_t> tokens, std::vector<float>& last,
    const std::function<bool(std::int32_t, std::span<const float>)>& on_row,
    const PrefillGoOn& go_on, PrefillRun* run) {
  return model_.ScorePrompt(*this, tokens, last, on_row, go_on, run);
}

Status Llm::Branch::PreparePrompt(std::span<const std::int32_t> tokens,
                                  std::uint32_t stable_boundary, std::vector<float>& last,
                                  std::uint32_t& reused, const PrefillGoOn& go_on, PrefillRun* run,
                                  bool fresh) {
  return model_.PreparePrompt(*this, tokens, stable_boundary, last, reused, go_on, run, fresh);
}

std::expected<std::unique_ptr<Llm::GenerationSession>, std::string> Llm::Branch::BeginGeneration(
    const std::vector<float>& last, const GenerateOptions& options, Generation& out) & {
  return model_.BeginGeneration(*this, last, options, out);
}

Status Llm::Branch::Generate(const std::vector<float>& last, const GenerateOptions& options,
                             Generation& out) {
  return model_.Generate(*this, last, options, out);
}

std::uint64_t Llm::Branch::state_snapshot_bytes() const {
  model_.CheckBranch(*this);
  return model_.used_state_bytes();
}

Status Llm::Branch::SaveState(void* host) { return model_.SaveState(*this, host); }

Status Llm::Branch::RestoreState(void* host) { return model_.RestoreState(*this, host); }

Status Llm::Clear() { return default_branch_.Clear(); }

void Llm::Forget() { default_branch_.Forget(); }

Status Llm::Prefill(std::span<const std::int32_t> tokens, std::vector<float>& last,
                    const PrefillGoOn& go_on, PrefillRun* run) {
  return default_branch_.Prefill(tokens, last, go_on, run);
}

Status Llm::ScorePrompt(std::span<const std::int32_t> tokens, std::vector<float>& last,
                        const std::function<bool(std::int32_t, std::span<const float>)>& on_row,
                        const PrefillGoOn& go_on, PrefillRun* run) {
  return default_branch_.ScorePrompt(tokens, last, on_row, go_on, run);
}

Status Llm::PreparePrompt(std::span<const std::int32_t> tokens, std::uint32_t stable_boundary,
                          std::vector<float>& last, std::uint32_t& reused, const PrefillGoOn& go_on,
                          PrefillRun* run, bool fresh) {
  return default_branch_.PreparePrompt(tokens, stable_boundary, last, reused, go_on, run, fresh);
}

std::uint64_t Llm::turn_checkpoint_bytes() const { return default_branch_.turn_checkpoint_bytes(); }

std::expected<std::unique_ptr<Llm::GenerationSession>, std::string> Llm::BeginGeneration(
    const std::vector<float>& last, const GenerateOptions& options, Generation& out) & {
  return default_branch_.BeginGeneration(last, options, out);
}

Status Llm::Generate(const std::vector<float>& last, const GenerateOptions& options,
                     Generation& out) {
  return default_branch_.Generate(last, options, out);
}

std::uint64_t Llm::state_snapshot_bytes() const { return default_branch_.state_snapshot_bytes(); }

Status Llm::SaveState(void* host) { return default_branch_.SaveState(host); }

Status Llm::RestoreState(void* host) { return default_branch_.RestoreState(host); }

std::expected<std::vector<std::int32_t>, std::string> Llm::EncodeText(std::string_view text) const {
  std::vector<tokenizer::TokenId> ids;
  if (auto r = tokenizer_->Encode(text, {.add_bos_eos = true}, ids); !r) {
    return Error(r.error().ToString());
  }
  if (tokenizer_->bos() && (ids.empty() || ids.front() != *tokenizer_->bos())) {
    ids.insert(ids.begin(), *tokenizer_->bos());  // BOS first, as the references fed it
  }
  return std::vector<std::int32_t>(ids.begin(), ids.end());
}

std::expected<std::vector<std::int32_t>, std::string> Llm::RenderChat(
    const chat::Conversation& conversation, std::uint32_t* stable_boundary) const {
  if (stable_boundary != nullptr) {
    *stable_boundary = 0;
  }
  if (template_ == nullptr) {
    return Error(std::format("{} has no chat template a native renderer supports (D-067)", name_));
  }
  auto rendered = template_->render(conversation);
  if (!rendered) {
    return Error(rendered.error().ToString());
  }
  std::vector<tokenizer::TokenId> ids;
  std::vector<std::size_t> span_tokens;
  if (auto r = tokenizer_->EncodeMarked(rendered->text, rendered->specials, {}, ids,
                                        stable_boundary == nullptr ? nullptr : &span_tokens);
      !r) {
    return Error(r.error().ToString());
  }
  if (stable_boundary != nullptr) {
    for (const chat::Boundary& boundary : rendered->boundaries) {
      if (boundary.kind != chat::BoundaryKind::kGenerationPrompt) {
        continue;
      }
      for (std::size_t i = 0; i < rendered->specials.size(); ++i) {
        if (rendered->specials[i].offset == boundary.offset && span_tokens[i] < ids.size()) {
          *stable_boundary = static_cast<std::uint32_t>(span_tokens[i]);
          break;
        }
      }
    }
  }
  return std::vector<std::int32_t>(ids.begin(), ids.end());
}

std::string Llm::Detokenize(std::span<const std::int32_t> tokens) const {
  std::string text;
  if (auto r = tokenizer_->Decode(tokens, {}, text); !r) {
    return "(not decodable)";
  }
  return text;
}

std::expected<std::int32_t, std::string> Llm::Choose(std::span<const float> row,
                                                     std::uint64_t position) {
  return Choose(default_branch_, row, position);
}

std::expected<std::int32_t, std::string> Llm::Choose(Branch& branch, std::span<const float> row,
                                                     std::uint64_t position) {
  CheckBranch(branch);
  if (!branch.sampling_) {
    return engine::Argmax(row);
  }
  auto token =
      execution::Sample(row, *branch.sampling_,
                        {.seed = branch.seed_, .stream = 0, .position = position}, branch.scratch_);
  if (!token) {
    return Error(std::format("sampling: {}", execution::SamplingErrorName(token.error())));
  }
  return *token;
}

std::expected<bool, std::string> Llm::Keep(std::span<const float> row, std::int32_t draft,
                                           std::uint64_t position, std::int32_t& next) {
  return Keep(default_branch_, row, draft, position, next);
}

std::expected<bool, std::string> Llm::Keep(Branch& branch, std::span<const float> row,
                                           std::int32_t draft, std::uint64_t position,
                                           std::int32_t& next) {
  CheckBranch(branch);
  if (!branch.sampling_) {
    next = engine::Argmax(row);
    return next == draft;
  }
  auto verdict = execution::VerifyDraft(row, draft, *branch.sampling_,
                                        {.seed = branch.seed_, .stream = 0, .position = position},
                                        branch.scratch_);
  if (!verdict) {
    return Error(std::format("sampling: {}", execution::SamplingErrorName(verdict.error())));
  }
  next = verdict->token;
  return verdict->accepted;
}

void Llm::FindThinkTokens() {
  think_start_ = tokenizer_->Find("<think>");
  think_end_ = tokenizer_->Find("</think>");
}

void Llm::CheckBranch(const Branch& branch) const {
  base::Check(&branch.model_ == this && &branch == &default_branch_,
              "native slot forwarding supports only this model's default branch");
}

void Llm::CheckIdleGeneration(const Branch& branch) const {
  CheckBranch(branch);
  base::Check(!branch.generation_active_, "mutating a conversation with an active generation");
}

void Llm::Forget(Branch& branch) {
  CheckIdleGeneration(branch);
  branch.turn_checkpoints_.clear();
  branch.history_.clear();
  branch.needs_clear_ = true;
}

Status Llm::Clear(Branch& branch) {
  CheckIdleGeneration(branch);
  branch.turn_checkpoints_.clear();
  if (auto r = ClearState(); !r) {
    return r;
  }
  branch.history_.clear();
  branch.needs_clear_ = false;
  branch.history_used_ = Clock::now();
  return {};
}

std::uint64_t Llm::Branch::turn_checkpoint_bytes() const {
  std::uint64_t bytes = 0;
  for (const TurnCheckpoint& checkpoint : turn_checkpoints_) {
    bytes += checkpoint.file.bytes();
  }
  return bytes;
}

Status Llm::CaptureTurnCheckpoint(Branch& branch, const PrefillGoOn& go_on, bool& stopped) {
  if (branch.history_.empty()) {
    return {};
  }
  const std::size_t position = branch.history_.size();
  if (std::ranges::any_of(branch.turn_checkpoints_, [&](const TurnCheckpoint& c) {
        return c.boundary.position == position;
      })) {
    return {};  // looking it up does not renew its retention period
  }
  if (auto settled = Settle(); !settled) {
    return settled;
  }
  auto ranges = CheckpointRanges(static_cast<std::uint32_t>(position));
  if (!ranges) {
    return std::unexpected(ranges.error());
  }
  const engine::CheckpointFile::Continue progress =
      go_on ? engine::CheckpointFile::Continue([&]() { return go_on(0); })
            : engine::CheckpointFile::Continue{};
  auto file = engine::CheckpointFile::Capture(
      *node_, checkpoint_directory_, *ranges,
      [&](void* host, std::span<const engine::LiveState::Range> page) {
        return CopyCheckpointState(host, page, true);
      },
      progress);
  if (!file) {
    if (file.error().invalid_state) {
      Forget(branch);
      return Error("saving a turn checkpoint: " + file.error().detail);
    }
    stopped = file.error().cancelled;
    return {};  // optional cache: unavailable staging or disk does not fail the turn
  }
  TurnCheckpoint checkpoint{.boundary = {.position = position, .created = Clock::now()},
                            .file = std::move(*file),
                            .footprint = used_state_ranges(),
                            .cursor = cursor(),
                            .decoding = TurnDecodingState()};
  branch.turn_checkpoints_.push_back(std::move(checkpoint));
  if (branch.turn_checkpoints_.size() > kTurnCheckpointLimit) {
    branch.turn_checkpoints_.erase(branch.turn_checkpoints_.begin());
  }
  return {};
}

Status Llm::ReusePrompt(Branch& branch, std::span<const std::int32_t> tokens, std::uint32_t& reused,
                        bool fresh, const PrefillGoOn& go_on, bool& stopped) {
  reused = 0;
  const auto now = Clock::now();
  if (fresh || branch.needs_clear_ || !StateUsable() ||
      now - branch.history_used_ >= kTurnCheckpointRetention) {
    return Clear(branch);
  }
  std::erase_if(branch.turn_checkpoints_, [&](const TurnCheckpoint& checkpoint) {
    return now - checkpoint.boundary.created >= kTurnCheckpointRetention;
  });
  const std::size_t common = CommonPrefix(branch.history_, tokens);
  if (!branch.history_.empty() && common == branch.history_.size() && common < tokens.size()) {
    reused = static_cast<std::uint32_t>(common);
    return {};
  }
  std::vector<TurnBoundary> boundaries;
  boundaries.reserve(branch.turn_checkpoints_.size());
  for (const TurnCheckpoint& checkpoint : branch.turn_checkpoints_) {
    boundaries.push_back(checkpoint.boundary);
  }
  const auto match = MatchingTurnBoundary(boundaries, common, tokens.size(), now);
  if (!match) {
    return Clear(branch);
  }
  TurnCheckpoint& checkpoint = branch.turn_checkpoints_[*match];
  const auto position = checkpoint.boundary.position;
  const auto restored_cursor = checkpoint.cursor;
  const auto restored_decoding = checkpoint.decoding;
  if (auto settled = Settle(); !settled) {
    Forget(branch);
    return settled;
  }
  const engine::CheckpointFile::Continue progress =
      go_on ? engine::CheckpointFile::Continue([&]() { return go_on(0); })
            : engine::CheckpointFile::Continue{};
  auto restored = checkpoint.file.Restore(
      *node_, [&]() { return PrepareRestoreState(checkpoint.footprint); },
      [&](void* host, std::span<const engine::LiveState::Range> page) {
        return CopyCheckpointState(host, page, false);
      },
      progress);
  if (!restored) {
    if (restored.error().cancelled) {
      stopped = true;
      return restored.error().invalid_state ? Clear(branch) : Status{};
    }
    // Before mutation this is an optional cache miss. After mutation, Clear
    // drops the branch before any fresh prefill; an uncertain device copy
    // may make Clear fail, in which case the runtime stops normally.
    return Clear(branch);
  }
  branch.history_.resize(position);
  set_cursor(restored_cursor);
  RestoreTurnDecodingState(restored_decoding);
  std::erase_if(branch.turn_checkpoints_,
                [&](const TurnCheckpoint& c) { return c.boundary.position > position; });
  reused = static_cast<std::uint32_t>(position);
  branch.needs_clear_ = false;
  branch.history_used_ = now;
  return {};
}

Status Llm::PreparePrompt(Branch& branch, std::span<const std::int32_t> tokens,
                          std::uint32_t stable_boundary, std::vector<float>& last,
                          std::uint32_t& reused, const PrefillGoOn& go_on, PrefillRun* run,
                          bool fresh) {
  CheckIdleGeneration(branch);
  if (tokens.empty() || tokens.size() >= context_ || stable_boundary >= tokens.size()) {
    return Error("the prompt or its turn boundary is outside the context");
  }
  last.clear();
  reused = 0;
  bool stopped = go_on && !go_on(0);
  if (!stopped) {
    if (auto prepared = ReusePrompt(branch, tokens, reused, fresh, go_on, stopped); !prepared) {
      return prepared;
    }
  }
  PrefillRun total{.end = reused, .stopped = stopped};
  if (run != nullptr) {
    *run = total;
  }
  if (stopped) {
    return {};
  }
  const auto append = [&](std::uint32_t end) -> Status {
    const auto begin = static_cast<std::uint32_t>(branch.history_.size());
    if (end <= begin) {
      return {};
    }
    PrefillRun part;
    auto ran = Prefill(branch, tokens.subspan(begin, end - begin), last, go_on, &part);
    total.end = part.end;
    total.chunks += part.chunks;
    total.longest = std::max(total.longest, part.longest);
    total.stopped = part.stopped;
    if (run != nullptr) {
      *run = total;
    }
    return ran;
  };
  if (stable_boundary != 0 && stable_boundary >= branch.history_.size()) {
    if (auto ran = append(stable_boundary); !ran) {
      return ran;
    }
    if (total.stopped) {
      return {};
    }
    if (go_on && !go_on(0)) {
      total.stopped = true;
      last.clear();
      if (run != nullptr) {
        *run = total;
      }
      return {};
    }
    if (auto checkpointed = CaptureTurnCheckpoint(branch, go_on, total.stopped); !checkpointed) {
      return checkpointed;
    }
    if (total.stopped) {
      last.clear();
      if (run != nullptr) {
        *run = total;
      }
      return {};
    }
  }
  return append(static_cast<std::uint32_t>(tokens.size()));
}

Status Llm::Prefill(Branch& branch, std::span<const std::int32_t> tokens, std::vector<float>& last,
                    const PrefillGoOn& go_on, PrefillRun* run) {
  CheckIdleGeneration(branch);
  if (branch.needs_clear_) {
    if (auto r = Clear(branch); !r) {
      return r;
    }
  }
  if (tokens.empty()) {
    return Error("nothing to prefill");
  }
  std::vector<std::int32_t> all = branch.history_;
  all.insert(all.end(), tokens.begin(), tokens.end());
  if (all.size() >= context_) {
    return Error(
        std::format("{} tokens do not fit {}'s context of {}", all.size(), name_, context_));
  }
  last.clear();
  auto completed = static_cast<std::uint32_t>(branch.history_.size());
  auto ran = RunPrefillChunks(
      static_cast<std::uint32_t>(branch.history_.size()), static_cast<std::uint32_t>(all.size()),
      max_rows_,
      [&](std::uint32_t at, std::uint32_t n) {
        auto chunk = RunChunk(std::span(all).first(at + n), at, speculate_, last);
        if (chunk) {
          completed = at + n;
        }
        return chunk;
      },
      go_on);
  if (!ran) {
    branch.needs_clear_ = !StateUsable();
    if (branch.needs_clear_) {
      branch.history_.clear();
    } else {
      branch.history_.assign(all.begin(), all.begin() + completed);
    }
    last.clear();
    return Error(std::format("{}'s prefill: {}", name_, ran.error()));
  }
  if (run != nullptr) {
    *run = *ran;
  }
  // The state holds exactly the chunks that ran: stopped between chunks,
  // the history is their prefix (which the next prefill continues from,
  // through the same chunk boundaries), and there are no logits to
  // generate from.
  all.resize(ran->end);
  branch.history_ = std::move(all);
  branch.history_used_ = Clock::now();
  if (ran->stopped) {
    last.clear();
  }
  return {};
}

Status Llm::ScorePrompt(Branch& branch, std::span<const std::int32_t> tokens,
                        std::vector<float>& last,
                        const std::function<bool(std::int32_t, std::span<const float>)>& on_row,
                        const PrefillGoOn& go_on, PrefillRun* run) {
  CheckIdleGeneration(branch);
  if (!branch.history_.empty() || tokens.empty() || tokens.size() > context_) {
    return Error("literal scoring needs an empty history and a nonempty prompt within context");
  }
  if (branch.needs_clear_) {
    if (auto r = Clear(branch); !r) {
      return r;
    }
  }
  PrefillRun completed;
  last.clear();
  for (std::uint32_t at = 0; at < tokens.size(); ++at) {
    if (go_on && !go_on(1)) {
      completed.stopped = true;
      break;
    }
    const auto started = Clock::now();
    auto ran = RunChunk(tokens.first(std::size_t{at} + 1), at, speculate_, last);
    if (!ran) {
      branch.needs_clear_ = !StateUsable();
      if (branch.needs_clear_) {
        branch.history_.clear();
      }
      last.clear();
      return ran;
    }
    branch.history_.push_back(tokens[at]);
    completed.end = at + 1;
    ++completed.chunks;
    completed.longest = std::max(completed.longest, Seconds(Clock::now() - started));
    if (at + 1 < tokens.size() && on_row && !on_row(tokens[at + 1], last)) {
      completed.stopped = true;
      break;
    }
  }
  if (auto settled = Settle(); !settled) {
    branch.needs_clear_ = true;
    branch.history_.clear();
    last.clear();
    return settled;
  }
  branch.history_used_ = Clock::now();
  if (completed.stopped) {
    last.clear();
  }
  if (run != nullptr) {
    *run = completed;
  }
  return {};
}

Llm::GenerationSession::GenerationSession(Llm& model, Branch& branch,
                                          const GenerateOptions& options, Generation& out)
    : model_(model), branch_(branch), options_(options), out_(out) {}

Llm::GenerationSession::~GenerationSession() {
  base::Check(finished_, "an unfinished generation session was destroyed");
}

std::expected<std::unique_ptr<Llm::GenerationSession>, std::string> Llm::BeginGeneration(
    Branch& branch, const std::vector<float>& last, const GenerateOptions& options,
    Generation& out) {
  CheckBranch(branch);
  if (last.empty()) {
    return Error(std::format("{} has no prefill's logits to generate from", name_));
  }
  if (options.max_tokens == 0) {
    return Error("generation needs a positive token budget");
  }
  if (branch.generation_active_) {
    return Error("the default conversation already has an active generation");
  }
  branch.generation_active_ = true;
  auto session =
      std::unique_ptr<GenerationSession>(new GenerationSession(*this, branch, options, out));
  if (auto began = session->Begin(last); !began) {
    session->Close();
    return std::unexpected(began.error());
  }
  return session;
}

bool Llm::GenerationSession::IsStop(std::int32_t token) const {
  return options_.stop && std::ranges::find(model_.stops_, token) != model_.stops_.end();
}

bool Llm::GenerationSession::Report() {
  // The new tokens since the last call, to on_tokens: false ends the
  // generation.
  if (!options_.on_tokens) {
    return true;
  }
  std::size_t visible = std::min<std::size_t>(out_.tokens.size(), options_.max_tokens);
  if (out_.stopped && visible == out_.tokens.size() && visible > 0) {
    --visible;  // the stop token
  }
  const std::span<const std::int32_t> fresh =
      std::span<const std::int32_t>(out_.tokens).subspan(reported_, visible - reported_);
  reported_ = visible;
  return options_.on_tokens(fresh);
}

Status Llm::GenerationSession::Begin(const std::vector<float>& last) {
  branch_.sampling_.reset();
  if (options_.sampling && options_.sampling->temperature > 0) {
    branch_.sampling_ = options_.sampling;
    branch_.seed_ = options_.seed;
  }
  auto first = model_.Choose(branch_, last, branch_.history_.size());
  if (!first) {
    return std::unexpected(first.error());
  }
  out_.tokens = {*first};
  if (options_.keep_logits) {
    out_.logits = {last};
  }
  out_.stopped = IsStop(out_.tokens.back());
  out_.cancelled = options_.on_logits && !options_.on_logits(*first, last);
  out_.cancelled = !Report() || out_.cancelled;
  // Every token so far, the anchor (the last generated, not yet in the
  // state) last; pos: how many the state holds.
  all_ = branch_.history_;
  all_.push_back(out_.tokens.back());
  position_ = static_cast<std::uint32_t>(branch_.history_.size());
  start_ = Clock::now();
  return {};
}

bool Llm::GenerationSession::done() const {
  return finished_ || !ran_ || out_.stopped || out_.cancelled ||
         out_.tokens.size() >= options_.max_tokens;
}

std::expected<Llm::GenerationSession::Step, std::string> Llm::GenerationSession::PrepareStep() {
  base::Check(!prepared_ && !done(), "preparing a generation without a next step");
  if (position_ + 1 >= model_.context_) {
    ran_ = Error(
        std::format("{}'s conversation reached its context of {}", model_.name_, model_.context_));
    failed_prefix_valid_ = true;
    return std::unexpected(ran_.error());
  }
  left_ = static_cast<std::uint32_t>(options_.max_tokens - out_.tokens.size());
  if (auto prepared = model_.PrepareDecodeState(position_, left_); !prepared) {
    ran_ = prepared;
    failed_prefix_valid_ = model_.StateUsable();
    return std::unexpected(ran_.error());
  }
  prepared_ = true;
  return Step{.all = all_,
              .position = position_,
              .left = left_,
              .speculative = model_.speculate_,
              .need_logits = options_.keep_logits || static_cast<bool>(options_.on_logits)};
}

Status Llm::GenerationSession::FailStep(std::string error, bool prefix_valid) {
  base::Check(prepared_, "failing a generation without a prepared step");
  prepared_ = false;
  ran_ = Error(std::move(error));
  failed_prefix_valid_ = prefix_valid;
  return ran_;
}

Status Llm::GenerationSession::ApplyPlain(std::vector<float> row) {
  base::Check(prepared_ && !model_.speculate_, "applying an unprepared ordinary generation step");
  prepared_ = false;
  auto next = model_.Choose(branch_, row, position_ + 1);
  if (!next) {
    ran_ = std::unexpected(next.error());
    ++position_;  // the anchor was processed before sampling failed
    failed_prefix_valid_ = model_.StateUsable();
    return ran_;
  }
  std::vector<std::vector<float>> logits;
  if (options_.keep_logits || options_.on_logits) {
    logits.push_back(std::move(row));
  }
  return ApplyTokens({*next}, std::move(logits));
}

Status Llm::GenerationSession::ApplySpeculative(std::vector<std::int32_t> kept,
                                                std::vector<std::vector<float>> logits,
                                                std::uint64_t drafted) {
  base::Check(prepared_ && model_.speculate_, "applying an unprepared speculative generation step");
  prepared_ = false;
  out_.drafted += drafted;
  if (kept.empty() || kept.size() > left_ ||
      ((options_.keep_logits || options_.on_logits) && logits.size() != kept.size())) {
    // Refuse a broken runner contract before indexing borrowed rows.
    // The completed verify still retires its owed commit/restore.
    constexpr std::string_view reason =
        "a speculative step returned inconsistent token/logit counts";
    if (auto settled = model_.Settle(); !settled) {
      ran_ = Error(std::format("{}; settling failed: {}", reason, settled.error()));
    } else {
      ran_ = Error(std::string(reason));
    }
    return ran_;
  }
  out_.accepted += kept.size() - 1;
  return ApplyTokens(std::move(kept), std::move(logits));
}

Status Llm::GenerationSession::ApplyTokens(std::vector<std::int32_t> kept,
                                           std::vector<std::vector<float>> logits) {
  if (++out_.steps == 1) {
    out_.first_step = Clock::now();
  }
  // The anchor and the accepted drafts are in the state now; the last
  // kept token is the next anchor. The generation ends at a stop token,
  // though the state holds what was accepted after it.
  position_ += static_cast<std::uint32_t>(kept.size());
  all_.insert(all_.end(), kept.begin(), kept.end());
  for (std::size_t i = 0; i < kept.size() && out_.tokens.size() < options_.max_tokens; ++i) {
    const std::int32_t token = kept[i];
    out_.tokens.push_back(token);
    if (options_.keep_logits) {
      out_.logits.push_back(logits[i]);
    }
    if (options_.on_logits && !options_.on_logits(token, logits[i])) {
      out_.cancelled = true;
    }
    if (IsStop(token)) {
      out_.stopped = true;
    }
    // A scoring response associates each visible token with its row.
    // Deliver it now, so a stop string cannot collect later verify rows.
    // The entire accepted verify is already complete and is still settled.
    if (options_.on_logits) {
      out_.cancelled = !Report() || out_.cancelled;
    }
    if (out_.stopped || out_.cancelled) {
      break;
    }
  }
  if (!options_.on_logits) {
    out_.cancelled = !Report() || out_.cancelled;
  }
  return {};
}

Status Llm::GenerationSession::RunScalarStep() {
  auto step = PrepareStep();
  if (!step) {
    return std::unexpected(step.error());
  }
  if (step->speculative) {
    std::vector<std::int32_t> kept;
    std::vector<std::vector<float>> logits;
    auto ran = model_.SpecStep(step->all, step->position, step->left, kept,
                               step->need_logits ? &logits : nullptr, out_.drafted);
    if (!ran) {
      return FailStep(ran.error());
    }
    // SpecStep already updated the legacy counter, including partial failures.
    return ApplySpeculative(std::move(kept), std::move(logits), 0);
  }
  std::vector<float> row;
  if (auto ran = model_.RunChunk(step->all, step->position, false, row); !ran) {
    return FailStep(ran.error());
  }
  return ApplyPlain(std::move(row));
}

void Llm::GenerationSession::Cancel() {
  base::Check(!finished_ && !prepared_, "cancelling a generation away from a completed boundary");
  out_.cancelled = true;
}

void Llm::GenerationSession::Close() {
  branch_.sampling_.reset();
  branch_.generation_active_ = false;
  finished_ = true;
}

Status Llm::GenerationSession::Finish() {
  if (finished_) {
    return ran_;
  }
  base::Check(!prepared_, "finishing a generation before its prepared step was applied");
  base::Check(done(), "finishing an active generation without cancellation");
  out_.decode_seconds = Seconds(Clock::now() - start_);
  branch_.sampling_.reset();
  if (!ran_) {
    if (failed_prefix_valid_) {
      if (auto settled = model_.Settle(); !settled) {
        ran_ = Error(std::format("{}; settling failed: {}", ran_.error(), settled.error()));
      } else {
        branch_.history_.assign(all_.begin(), all_.begin() + position_);
        branch_.needs_clear_ = false;
        Close();
        return ran_;
      }
    }
    branch_.needs_clear_ = true;
    branch_.history_.clear();
    Close();
    return ran_;
  }
  branch_.history_.assign(all_.begin(), all_.begin() + position_);
  branch_.history_used_ = Clock::now();
  if (out_.tokens.size() > options_.max_tokens) {
    out_.tokens.resize(options_.max_tokens);
  }
  if (out_.logits.size() > out_.tokens.size()) {
    out_.logits.resize(out_.tokens.size());
  }
  if (auto settled = model_.Settle(); !settled) {
    branch_.needs_clear_ = true;
    branch_.history_.clear();
    ran_ = settled;
  }
  Close();
  return ran_;
}

Status Llm::Generate(Branch& branch, const std::vector<float>& last, const GenerateOptions& options,
                     Generation& out) {
  auto began = BeginGeneration(branch, last, options, out);
  if (!began) {
    return std::unexpected(began.error());
  }
  auto& session = **began;
  while (!session.done()) {
    if (auto ran = session.RunScalarStep(); !ran) {
      break;
    }
  }
  return session.Finish();
}

Status Llm::SaveState(Branch& branch, void* host) {
  CheckIdleGeneration(branch);
  branch.saved_valid_ = false;
  if (auto r = Settle(); !r) {
    return r;
  }
  auto ranges = used_state_ranges();
  if (auto r = SaveUsedState(host, ranges); !r) {
    return r;
  }
  branch.saved_ranges_ = std::move(ranges);
  branch.saved_history_ = branch.history_;
  branch.saved_cursor_ = cursor();
  SaveDecodingState();
  branch.saved_valid_ = true;
  return {};
}

Status Llm::RestoreState(Branch& branch, void* host) {
  CheckIdleGeneration(branch);
  if (!branch.saved_valid_) {
    return Error("no completed conversation snapshot to restore");
  }
  if (host == nullptr && !branch.saved_ranges_.empty()) {
    return Error("the conversation snapshot has no source buffer");
  }
  if (auto r = Settle(); !r) {
    return r;
  }
  if (auto r = RestoreUsedState(host, branch.saved_ranges_); !r) {
    branch.needs_clear_ = true;
    return r;
  }
  branch.history_ = branch.saved_history_;
  branch.turn_checkpoints_.clear();  // a full diagnostic restore may replace the branch
  branch.history_used_ = Clock::now();
  set_cursor(branch.saved_cursor_);
  RestoreDecodingState();
  branch.needs_clear_ = false;
  return {};
}

// ---------------------------------------------------------------- the server

Server::Server(const config::NodeConfig& config, const config::RuntimeRoles& roles,
               const ServingOptions& options, std::FILE* log)
    : config_(config),
      roles_(roles),
      options_(options),
      log_(log),
      times_(kObservedExtents),
      node_({.compute_streams = std::max<std::size_t>(config.models.size(), 1),
             .slots = engine::kPagedSlots,
             .inline_lanes = false,
             .coalesce = false,
             .copy_lane = true,
             .slot_bytes = engine::kSlabSlotBytes,
             .observer = &times_}) {}

Server::~Server() {
  if (started_ && !torn_down_) {
    if (auto stopped = TearDown(); !stopped) {
      Log("stopping: " + stopped.error());
    }
  }
}

void Server::Log(std::string_view text) {
  const std::string line = std::format("jitllm-runtime: {}\n", base::Printable(text));
  (void)std::fwrite(line.data(), 1, line.size(), log_);
  (void)std::fflush(log_);
}

Status Server::Make(const config::ModelEntry& entry, int index, std::string_view architecture) {
  if (entry.composition) {
    if (options_.image_noise.empty()) {
      Log(
          std::format("model {}: not registered: an image pipeline needs --image-noise (its "
                      "initial latents) in this build",
                      entry.name));
      return {};
    }
    models_.push_back(std::make_unique<QwenImage>(node_, entry, roles_, options_, index));
    return {};
  }
  if (architecture == "deepseek4") {
    models_.push_back(std::make_unique<Dsv4>(node_, entry, roles_, options_.plain, index));
  } else if (architecture == "qwen4exp") {
    models_.push_back(std::make_unique<Qwen38>(node_, entry, roles_, options_.plain, index));
  } else {
    return Error(std::format("model {}: no runner for architecture {}", entry.name, architecture));
  }
  return {};
}

Status Server::SaveSnapshot(Llm& model) {
  if (!snapshot_enabled_ || snapshot_unproven_) {
    return Error("the diagnostic snapshot is disabled or its completion is unproven");
  }
  if (snapshot_model_ != nullptr) {
    snapshot_model_->InvalidateStateSnapshot();
    snapshot_model_ = nullptr;
  }
  const auto bytes = std::max<std::uint64_t>(model.state_snapshot_bytes(), 256);
  if (bytes > snapshot_capacity_) {
    if (auto freed = node_.FreePinned(snapshot_); !freed) {
      return freed;
    }
    snapshot_ = nullptr;
    snapshot_capacity_ = 0;
    const auto available = MemorySampler::Available();
    if (available != 0 && (available < host_inputs_ + kUncountedMargin ||
                           bytes > available - host_inputs_ - kUncountedMargin)) {
      return Error("initialized state snapshot would exceed the memory guard");
    }
    std::vector<catalog::ExtentId> staging;
    auto allocated = node_.Pinned(bytes, engine::kShared, staging);
    if (!allocated) {
      return std::unexpected(allocated.error());
    }
    snapshot_ = *allocated;
    snapshot_capacity_ = bytes;
  }
  auto saved = model.SaveState(snapshot_);
  if (!saved) {
    snapshot_unproven_ = true;
    node_.KeepPinned(snapshot_);
  } else {
    snapshot_model_ = &model;
  }
  return saved;
}

Status Server::RestoreSnapshot(Llm& model) {
  if (!snapshot_enabled_ || snapshot_unproven_ || snapshot_model_ != &model) {
    return Error("the diagnostic snapshot is disabled or its completion is unproven");
  }
  auto restored = model.RestoreState(snapshot_);
  if (!restored) {
    snapshot_unproven_ = true;
    node_.KeepPinned(snapshot_);
  }
  return restored;
}

Status Server::Start(bool snapshot) {
  if (started_) {
    return Error("the server started already");
  }
  started_ = true;
  if (config_.models.empty()) {
    return Error("the configuration names no models ([models.<name>], D-096)");
  }
  // Check every LLM's checkpoint ceiling before opening the device node or
  // constructing runners. An invalid model later in the list must not
  // cause any earlier model's resources to be allocated either.
  std::vector<std::string> architectures;
  architectures.reserve(config_.models.size());
  for (const config::ModelEntry& entry : config_.models) {
    if (entry.composition) {
      architectures.emplace_back();
      continue;
    }
    auto artifact = OpenTrusted(roles_.installed, entry.artifact.value_or(""));
    if (!artifact) {
      return Error(std::format("model {}: {}", entry.name, artifact.error()));
    }
    const std::string& architecture = artifact->model().architecture;
    if (auto context = CheckModelContext(architecture, entry.context); !context) {
      return Error(std::format("model {}: {}", entry.name, context.error()));
    }
    architectures.push_back(architecture);
  }
  if (auto r = node_.Open(); !r) {
    return r;
  }
  // Each model's stream and owner: its place among the registered.
  std::size_t entry_index = 0;
  for (const config::ModelEntry& entry : config_.models) {
    if (auto r = Make(entry, static_cast<int>(models_.size()), architectures[entry_index]); !r) {
      return r;
    }
    ++entry_index;
  }
  if (models_.empty()) {
    return Error("no configured model could be registered");
  }
  std::uint64_t activations = 0;
  std::uint64_t pool = 0;
  std::uint64_t largest = 0;
  std::uint64_t host_inputs = 0;
  for (const auto& m : models_) {
    const auto started = Clock::now();
    if (auto r = m->Setup(); !r) {
      return Error(std::format("model {}: {}", m->name(), r.error()));
    }
    activations = std::max(activations, m->activations_needed());
    host_inputs = std::max(host_inputs, m->host_input_bytes());
    pool = std::max(pool, m->pool_needed());
    largest = std::max<std::uint64_t>(largest, m->weights().size() * kExtent);
    std::string chunks;
    if (m->llm()) {
      const auto& l = static_cast<Llm&>(*m);
      chunks = std::format("; prefill chunks of {} rows", l.max_rows());
      if (l.configured_rows() && *l.configured_rows() != l.max_rows()) {
        chunks += std::format(
            " (prefill_chunk {}: capped at what the model allows below its context of {}, in "
            "whole 8-row tiles)",
            *l.configured_rows(), l.context());
      }
    }
    Log(std::format("model {}: set up in {:.2f} s, {} weight extents ({:.2f} GB read a load){}",
                    m->name(), Seconds(Clock::now() - started), m->weights().size(),
                    static_cast<double>(m->weight_read_bytes()) / 1e9, chunks));
  }
  if (auto r = node_.MapWorkspace(activations, pool); !r) {
    return r;
  }
  workspace_ = activations + pool;
  snapshot_enabled_ = snapshot;
  // The budget (D-050's B): everything mapped for the node's life (the zone,
  // each model's own memory, the workspace, the staging) and the largest
  // model's weights, so a full swap is the only way in.
  fixed_ = node_.catalog().OccupancyOf(node_.domain()).Total().value();
  budget_ = fixed_ + largest;
  // A chunk's host-built inputs are the node's memory too, though allocated
  // per chunk and outside the catalog: counted here beside the margin, not
  // in the budget the catalog enforces. One model runs chunks at a time, so
  // the most any model's chunk builds, not their sum.
  host_inputs_ = host_inputs;
  const std::uint64_t available = MemorySampler::Available();
  if (auto guard = CheckMemoryGuard({.largest = largest,
                                     .host_inputs = host_inputs,
                                     .available = available,
                                     .fixed = fixed_});
      !guard) {
    return std::unexpected(guard.error());
  }
  // State is charged as it grows. Keep the physical execution cap below
  // the measured available memory, leaving host-built inputs and the
  // uncounted margin outside it. A virtual context ceiling need not fit
  // before it is used; a growth that cannot fit its active closure fails.
  if (available != 0) {
    budget_ = fixed_ + ((available - host_inputs - kUncountedMargin) / kExtent * kExtent);
  } else {
    budget_ += node_.StateCapacity();
  }
  if (auto r = node_.Start(base::Bytes(budget_)); !r) {
    return r;
  }
  for (const auto& m : models_) {
    if (auto r = m->Register(); !r) {
      return Error(std::format("model {}: {}", m->name(), r.error()));
    }
  }
  for (const auto& m : models_) {
    if (auto r = m->Bind(); !r) {
      return Error(std::format("model {}: {}", m->name(), r.error()));
    }
  }
  node_.Run();
  Log(std::format(
      "serving {} models; budget {:.2f} GiB ({:.2f} GiB fixed, the workspace {:.2f}; host-built "
      "chunk inputs {:.2f} GiB beside it); {:.2f} GiB available",
      models_.size(), static_cast<double>(budget_) / (1ULL << 30U),
      static_cast<double>(fixed_) / (1ULL << 30U), static_cast<double>(workspace_) / (1ULL << 30U),
      static_cast<double>(host_inputs_) / (1ULL << 30U),
      static_cast<double>(available) / (1ULL << 30U)));
  return {};
}

Status Server::TearDown() {
  if (torn_down_) {
    return {};
  }
  torn_down_ = true;
  std::vector<engine::PagedModel*> models;
  models.reserve(models_.size());
  for (const auto& m : models_) {
    models.push_back(&m->paged());
  }
  return node_.TearDown(models);
}

Served* Server::Find(std::string_view name) {
  for (const auto& m : models_) {
    if (m->name() == name) {
      return m.get();
    }
  }
  return nullptr;
}

Status Server::InRequest(Served& m, const std::function<Status()>& body) {
  return node_.WithRequest(m.paged().stream(), m.everything(),
                           std::format("{}'s request", m.name()), body);
}

Status Server::WaitReleased() {
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
      return {};
    }
    if (Clock::now() > give_up) {
      return Error("backing no load took was not released");
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }
}

Status Server::FinishSwap(SwapParts& parts) {
  if (auto r = WaitReleased(); !r) {
    return r;
  }
  parts.release = Seconds(Clock::now() - parts.ready);
  auto after = node_.Stats();
  if (!after) {
    return std::unexpected(after.error());
  }
  parts.handed_off = after->handed_off - swap_before_.handed_off;
  parts.released_unused = after->released_unused - swap_before_.released_unused;
  return {};
}

Status Server::Activate(Served& m, SwapParts& parts, std::optional<bool> spill_state) {
  parts = SwapParts{};
  parts.to = m.name();
  if (resident_ == &m) {
    return {};
  }
  auto before = node_.Stats();
  if (!before) {
    return std::unexpected(before.error());
  }
  swap_before_ = *before;
  const bool restoring = std::ranges::find(spilled_, &m) != spilled_.end();
  parts.requested = Clock::now();
  Clock::time_point evicted = parts.requested;
  Clock::time_point loaded;
  if (resident_ == nullptr) {
    // Nothing resident: the first load of its weights (its state is
    // resident since setup).
    std::vector<engine::LoadStats> log;
    if (auto r = node_.Load(m.weights(), std::format("{}'s first load", m.name()), log); !r) {
      return r;
    }
    loaded = Clock::now();
    parts.loaded = log.empty() ? 0 : log.front().extents;
  } else {
    Served& out = *resident_;
    parts.from = out.name();
    const bool conversation = out.llm() && out.HasRetainedState();
    parts.with_state = out.llm() && spill_state.value_or(conversation);
    std::vector<catalog::ExtentId> extents;
    if (parts.with_state) {
      extents = out.state();  // first: its write-backs start first
      parts.spilled_bytes = extents.size() * kExtent;
    }
    const std::vector<catalog::ExtentId> weights = out.weights();
    extents.insert(extents.end(), weights.begin(), weights.end());
    scheduler::SwapReport report;
    if (auto r = node_.Swap(std::move(extents), m.everything(), handoff_, report); !r) {
      return r;
    }
    if (parts.with_state) {
      spilled_.push_back(&out);
    }
    evicted = report.evicted;
    loaded = report.loaded;
    parts.evicted = report.evictions;
    parts.loaded = report.loads;
  }
  parts.evict = Seconds(evicted - parts.requested);
  Clock::time_point restored = evicted;
  if (restoring) {
    restored = std::max(restored, times_.Latest(m.state()));
    parts.restore = Seconds(restored - evicted);
    std::erase(spilled_, &m);
  }
  parts.page_in = Seconds(loaded - restored);
  parts.read_bytes = m.weight_read_bytes() + (restoring ? m.state().size() * kExtent : 0);
  resident_ = &m;
  if (auto r = m.AfterLoad(); !r) {
    return Error(std::format("{} after its load: {}", m.name(), r.error()));
  }
  parts.ready = Clock::now();
  parts.setup = Seconds(parts.ready - loaded);
  parts.total = Seconds(parts.ready - parts.requested);
  // Its places still pinned where it registered them (D-090).
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
    return Error(std::format("{} of {}'s {} extents are not pinned at their places", unpinned,
                             m.name(), managed.size()));
  }
  if (auto r = m.CheckPlaces(); !r) {
    return Error(std::format("{}'s places: {}", m.name(), r.error()));
  }
  return {};
}

}  // namespace jitllm::runtime
