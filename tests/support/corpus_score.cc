// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "corpus_score.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <format>
#include <limits>
#include <utility>

#include "execution/sampling.h"

namespace llmp::test_support {
namespace {
auto Error(const char* what) { return std::unexpected(std::string(what)); }
constexpr std::uint64_t kMiB = 1024 * 1024;
}  // namespace

double CorpusScoreResult::mean_nll() const {
  return targets ? total_nll / static_cast<double>(targets)
                 : std::numeric_limits<double>::quiet_NaN();
}
double CorpusScoreResult::perplexity() const { return std::exp(mean_nll()); }

std::expected<std::uint64_t, std::string> CorpusScore::Bound(
    std::span<const CorpusDocument> documents, CorpusScoreLimits limits) {
  if (documents.empty() || documents.size() > kMaxDocuments || !limits.vocabulary ||
      limits.vocabulary > 1048576 || !limits.chunk_rows || limits.chunk_rows > 1024)
    return Error("corpus dimensions exceed the finite scorer domain");
  std::uint64_t tokens = 0, rows = 0;
  for (std::size_t i = 0; i < documents.size(); ++i) {
    const auto& d = documents[i];
    if (d.tokens.size() < 2 || d.tokens.size() > 1048576 ||
        std::ranges::any_of(documents.first(i),
                            [&](const auto& old) { return old.slot == d.slot; }))
      return Error("corpus documents need distinct slots and 2..1048576 tokens");
    for (auto token : d.tokens)
      if (token < 0 || std::cmp_greater_equal(token, limits.vocabulary))
        return Error("corpus token is outside the full vocabulary");
    tokens += d.tokens.size();
    rows += std::min<std::uint64_t>(d.tokens.size(), limits.chunk_rows);
  }
  const std::uint64_t outputs = 4 * limits.vocabulary * rows;
  if (outputs > 64 * kMiB) return Error("corpus wave exceeds 64 MiB of full head outputs");
  // Finite dimensions above make every product/sum safe. Owned IDs plus the
  // scored orchestrator's old/new full-history reallocation peak: 16*T.
  // Four output planes cover simultaneous completed reports/vector growth;
  // staging/device outputs are already measured and paid by runner Setup.
  // One aligned row, and 1 MiB for fixed scorer, <=16 work/outcome containers,
  // tiny top-1 score allocations and allocator overhead. Not compiler funding.
  static_assert(sizeof(CorpusScore) < 16384);
  static_assert(sizeof(engine::DecodeWork) + sizeof(engine::DecodeOutcome) < 4096);
  return 16 * tokens + 4 * outputs + 4 * limits.vocabulary + kMiB;
}

std::expected<std::unique_ptr<CorpusScore>, std::string> CorpusScore::Make(
    engine::PagedNode& node, engine::ModelRunner& runner, const engine::WaveComposer& composer,
    std::span<const CorpusDocument> documents, CorpusScoreLimits limits, CorpusScoreSink sink,
    CorpusRowSink row_sink) {
  auto bound = Bound(documents, limits);
  if (!bound) return std::unexpected(bound.error());
  if (runner.slots().stages() != 1) return Error("corpus scorer requires a target-only runner");
  for (const auto& d : documents)
    if (d.slot >= runner.slots().slots() || runner.slots().positions(d.slot) != 0 ||
        d.tokens.size() > runner.slots().context(0))
      return Error("corpus slot is not empty or its context is too short");
  if (!node.TryChargeHost(*bound)) return Error("corpus host admission refused before allocation");
  auto out = std::unique_ptr<CorpusScore>(new CorpusScore(node, runner, limits, *bound));
  std::uint64_t count = 0;
  for (const auto& d : documents) count += d.tokens.size();
  out->tokens_ = std::make_unique<std::int32_t[]>(static_cast<std::size_t>(count));
  out->row_ = std::make_unique<float[]>(limits.vocabulary);
  std::uint64_t at = 0;
  for (const auto& d : documents) {
    std::copy(d.tokens.begin(), d.tokens.end(), out->tokens_.get() + at);
    out->documents_[out->count_++] = {.slot = d.slot,
                                      .size = static_cast<std::uint32_t>(d.tokens.size()),
                                      .cursor = 0,
                                      .offset = at};
    at += d.tokens.size();
  }
  auto decoder = engine::DecodingOrchestrator::Make(
      runner, composer, {.kind = engine::DecodingMode::Kind::kScored}, {});
  if (!decoder) return std::unexpected(decoder.error());
  out->decoder_.emplace(std::move(*decoder));
  out->sink_ = std::move(sink);
  out->row_sink_ = std::move(row_sink);
  return out;
}
CorpusScore::~CorpusScore() {
  decoder_.reset();
  row_.reset();
  tokens_.reset();
  node_.UnchargeHost(charge_);
}
void CorpusScore::Cancel() {
  if (!complete_) {
    canceled_ = true;
    failed_ = true;
  }
}

std::expected<double, std::string> CorpusScore::Nll(std::span<const float> logits,
                                                    std::int32_t target) {
  if (logits.empty() || logits.size() > 1048576 || target < 0 ||
      std::cmp_greater_equal(target, logits.size()) ||
      std::ranges::any_of(logits, [](float v) { return !std::isfinite(v); }))
    return Error("corpus score requires a finite complete row and valid target");
  auto score = execution::ScoreToken(logits, target, 0);
  if (!score || !std::isfinite(score->logprob)) return Error("corpus logsumexp failed");
  return -score->logprob;
}

std::expected<bool, std::string> CorpusScore::Step() {
  if (failed_) return Error("corpus scorer is terminal after failure or cancellation");
  if (complete_) return true;
  std::array<engine::DecodeWork, kMaxDocuments> work{};
  std::array<std::size_t, kMaxDocuments> indexes{};
  std::size_t count = 0;
  for (std::size_t i = 0; i < count_; ++i) {
    const auto& d = documents_[i];
    if (d.cursor == d.size) continue;
    if (runner_.slots().positions(d.slot) != d.cursor) {
      failed_ = true;
      return Error("corpus slot advanced outside its fixed teacher schedule");
    }
    const auto all = std::span(tokens_.get() + d.offset, d.size);
    indexes[count] = i;
    work[count++] = {
        .slot = d.slot,
        .history = all.first(d.cursor),
        .score = all.subspan(d.cursor, std::min(limits_.chunk_rows, d.size - d.cursor))};
  }
  // A failed outcome may have completed peers. Never retry this cohort or
  // publish a successful aggregate with missing rows.
  failed_ = true;
  auto report = decoder_->Step(std::span(work).first(count));
  if (report.outcomes.size() != count || report.speculated)
    return Error("corpus execution returned another outcome domain");
  result_.steps += report.steps;
  for (std::size_t i = 0; i < count; ++i) {
    const auto& out = report.outcomes[i];
    const auto& d = documents_[indexes[i]];
    const auto rows = work[i].score.size();
    if (out.refusal || out.discarded || !out.joined.empty() || !out.tokens.empty() ||
        out.kept != rows || out.outputs.size() != rows * limits_.vocabulary * sizeof(float) ||
        runner_.slots().positions(d.slot) != d.cursor + rows)
      return std::unexpected(
          std::format("corpus execution refused/incomplete: slot={} past={} rows={} "
                      "kept={} bytes={} "
                      "position={} discarded={} joined={} tokens={} refusal={}",
                      d.slot, d.cursor, rows, out.kept, out.outputs.size(),
                      runner_.slots().positions(d.slot), out.discarded, out.joined.size(),
                      out.tokens.size(), out.refusal ? out.refusal->what.substr(0, 1024) : "none"));
  }
  for (std::size_t i = 0; i < count; ++i) {
    auto& d = documents_[indexes[i]];
    const auto& output = report.outcomes[i].outputs;
    const auto rows = work[i].score.size();
    for (std::size_t r = 0; r < rows; ++r) {
      const std::size_t bytes = std::size_t{limits_.vocabulary} * sizeof(float);
      std::memcpy(row_.get(), output.data() + r * bytes, bytes);
      const auto row = std::span(row_.get(), limits_.vocabulary);
      // Validate even the unscored final row; every requested output is real.
      if (std::ranges::any_of(row, [](float v) { return !std::isfinite(v); }))
        return Error("corpus execution returned nonfinite vocabulary logits");
      const std::uint64_t next = d.cursor + r + 1;
      if (row_sink_) {
        const auto observed = row_sink_(static_cast<std::uint32_t>(indexes[i]), d.slot,
                                        static_cast<std::uint32_t>(next - 1), row);
        if (!observed) return std::unexpected(observed.error());
        if (canceled_) return Error("corpus scorer canceled by its synchronous row sink");
      }
      if (next < d.size) {
        auto nll = Nll(row, tokens_[d.offset + next]);
        if (!nll) return std::unexpected(nll.error());
        if (sink_) {
          const auto emitted = sink_({.document = static_cast<std::uint32_t>(indexes[i]),
                                      .slot = d.slot,
                                      .position = static_cast<std::uint32_t>(next),
                                      .target = tokens_[d.offset + next],
                                      .nll = *nll});
          if (!emitted) return std::unexpected(emitted.error());
          if (canceled_) return Error("corpus scorer canceled by its synchronous sink");
        }
        const double value = *nll - compensation_;
        const double sum = result_.total_nll + value;
        compensation_ = (sum - result_.total_nll) - value;
        result_.total_nll = sum;
        ++result_.targets;
      }
      ++result_.inputs;
    }
    d.cursor += static_cast<std::uint32_t>(rows);
  }
  const bool complete = std::ranges::all_of(std::span(documents_).first(count_),
                                            [](const auto& d) { return d.cursor == d.size; });
  if (complete && (!std::isfinite(result_.total_nll) || !std::isfinite(result_.perplexity())))
    return Error("corpus aggregate is nonfinite");
  complete_ = complete;
  failed_ = false;
  return complete_;
}

namespace {
engine::Status CheckCorpusStep(const kernels::LowerInput& input, std::uint32_t vocabulary) {
  const auto& shape = input.shape;
  const auto& step = input.step;
  if (shape.stage != 0 || shape.phase != planner::Phase::kScore || !vocabulary ||
      vocabulary > 1048576 || shape.segments.empty() ||
      shape.segments.size() > CorpusScore::kMaxDocuments || !step.logits.valid() ||
      step.logits.v >= step.graph.tensor_count() || step.greedy.valid() || step.features.valid() ||
      !step.saves.empty())
    return Error("corpus lowering is not a full target scoring step");
  for (const auto& segment : shape.segments)
    if (!segment.rows || segment.rows > 1024 || !segment.want_head || segment.greedy ||
        segment.outputs != segment.rows)
      return Error("corpus step narrowed or changed its head output");
  const auto& t = step.graph.tensor(step.logits);
  if (t.format.encoding != format::Encoding::kF32 ||
      t.shape() != ir::ShapeOf({vocabulary, shape.outputs()}) || !ir::IsContiguous(t))
    return Error("corpus IR head is not packed F32 full vocabulary");
  return {};
}
}  // namespace

engine::Status CheckCorpusHead(const kernels::LowerInput& input,
                               const kernels::StepProgram& program, std::uint32_t vocabulary) {
  if (auto valid = CheckCorpusStep(input, vocabulary); !valid) return valid;
  const auto& shape = input.shape;
  if (program.outputs().size() != shape.segments.size())
    return Error("corpus lower must return exactly one head per owner");
  std::array<const kernels::OutputSlot*, CorpusScore::kMaxDocuments> outputs{};
  for (const auto& out : program.outputs()) {
    if (out.segment >= shape.segments.size() || out.features || outputs[out.segment] ||
        out.tensor != input.step.logits)
      return Error("corpus lower returned another tensor/joined/duplicate/feature output");
    outputs[out.segment] = &out;
  }
  // Packed original head rows are partitioned in the shape's owner order.
  // Merely trusting equal byte counts would allow swapping equal-sized owners.
  std::uint64_t address = outputs[0]->device_address;
  for (std::size_t i = 0; i < shape.segments.size(); ++i) {
    const auto bytes = std::uint64_t{shape.segments[i].rows} * vocabulary * sizeof(float);
    const auto& out = *outputs[i];
    if (out.bytes != bytes || out.device_address != address ||
        address > std::numeric_limits<std::uint64_t>::max() - bytes)
      return Error("corpus lower narrowed or reordered packed owner heads");
    address += bytes;
  }
  return {};
}

kernels::LowerFunction GuardCorpusLower(kernels::LowerFunction lower, std::uint32_t vocabulary) {
  return [lower = std::move(lower), vocabulary](const kernels::LowerInput& input)
             -> std::expected<std::unique_ptr<kernels::StepProgram>, std::string> {
    if (auto valid = CheckCorpusStep(input, vocabulary); !valid)
      return std::unexpected(valid.error());
    if (!lower) return Error("corpus lower callback is absent");
    auto program = lower(input);
    if (!program) return std::unexpected(program.error());
    if (!*program) return Error("corpus lower returned no program");
    if (auto valid = CheckCorpusHead(input, **program, vocabulary); !valid)
      return std::unexpected(valid.error());
    return program;
  };
}
}  // namespace llmp::test_support
