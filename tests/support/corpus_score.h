// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#ifndef LLMP_TESTS_SUPPORT_CORPUS_SCORE_H_
#define LLMP_TESTS_SUPPORT_CORPUS_SCORE_H_

#include <array>
#include <cstdint>
#include <expected>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "engine/decoding.h"
#include "execution/registry.h"
#include "kernels/common/lower.h"

namespace llmp::test_support {

// Transparent test-only plan recorder. The caller prepays the record storage
// (up to4096 names of128 bytes) and retains it until device retirement.
struct CorpusPlanRecord {
  std::uint32_t rows = 0;
  llmp::base::Sha256Digest identity{};
  std::vector<std::string> implementations;
  std::uint64_t runs = 0;
};
class CorpusProgram final : public kernels::StepProgram {
 public:
  CorpusProgram(std::unique_ptr<kernels::StepProgram> inner, CorpusPlanRecord& record)
      : inner_(std::move(inner)), record_(record) {}
  void ConstructionPaid() override {
    if (construction_transferred_) return;
    construction_transferred_ = true;
    inner_->ConstructionPaid();
    kernels::StepProgram::ConstructionPaid();
  }
  std::uint64_t scratch_bytes() const override { return inner_->scratch_bytes(); }
  std::uint64_t pool_bytes() const override { return inner_->pool_bytes(); }
  kernels::PlanCoverage plan_coverage() const override { return inner_->plan_coverage(); }
  std::uint64_t host_bytes() const override { return inner_->host_bytes() + sizeof(*this); }
  std::uint64_t launches() const override { return inner_->launches(); }
  std::span<const kernels::InputSlot> inputs() const override { return inner_->inputs(); }
  std::span<const kernels::OutputSlot> outputs() const override { return inner_->outputs(); }
  std::span<const kernels::TensorUse> tensors() const override { return inner_->tensors(); }
  const llmp::execution::BoundPlan* bound() const override { return inner_->bound(); }
  void StreamPlanReport(const kernels::PlanReportSink& sink) const override {
    inner_->StreamPlanReport(sink);
  }
  bool capturable() const override { return inner_->capturable(); }
  std::expected<void, kernels::StepFailure> Position(std::span<const std::uint32_t> past) override {
    return inner_->Position(past);
  }
  std::expected<void, kernels::StepFailure> Bind(kernels::StepLaunch& launch) override {
    auto result = inner_->Bind(launch);
    if (!result) return result;
    const auto* selected = inner_->bound();
    if (!selected || !selected->size() || selected->size() > 4096)
      return std::unexpected(
          kernels::StepFailure{.detail = "corpus bound implementation inventory"});
    if (!record_.implementations.empty()) {
      if (record_.identity != selected->identity())
        return std::unexpected(
            kernels::StepFailure{.detail = "corpus same-shape plan identity changed"});
      return {};
    }
    record_.identity = selected->identity();
    record_.implementations.reserve(selected->size());
    for (std::size_t i = 0; i < selected->size(); ++i) {
      const auto& implementation = selected->at(i);
      if (implementation.name.size() > 128)
        return std::unexpected(kernels::StepFailure{.detail = "corpus implementation name bound"});
      record_.implementations.push_back(implementation.name);
    }
    return {};
  }
  std::expected<void, kernels::StepFailure> Run(
      kernels::StepLaunch& launch, llmp::providers::NativeStream stream) const override {
    auto result = inner_->Run(launch, stream);
    if (result) ++record_.runs;  // Includes recording; not a GPU timing/call counter.
    return result;
  }

 private:
  bool construction_transferred_ = false;
  std::unique_ptr<kernels::StepProgram> inner_;
  CorpusPlanRecord& record_;
};

// Fixed teacher IDs, one independent initially empty slot per document.
// Bound/Make borrow this input only during the call; Make copies it after payment.
struct CorpusDocument {
  std::uint32_t slot = 0;
  std::span<const std::int32_t> tokens{};
};
struct CorpusScoreLimits {
  std::uint32_t vocabulary = 0;
  std::uint32_t chunk_rows = 0;
};
struct CorpusTargetScore {
  std::uint32_t document = 0, slot = 0, position = 0;
  std::int32_t target = 0;
  double nll = 0;
};
// Synchronous sink; caller owns and prepays any retained report. Failure is
// terminal. Captured owners must outlive the scorer, including uncertain retirement.
using CorpusScoreSink = std::function<engine::Status(const CorpusTargetScore&)>;

// Optional same-plan state/replay witness over a completed finite host row,
// including the terminal row with no next target. Zero-based input position;
// spans expire on callback return. Retained hashes/report storage are caller-paid.
using CorpusRowSink = std::function<engine::Status(std::uint32_t document, std::uint32_t slot,
                                                   std::uint32_t position, std::span<const float>)>;

struct CorpusScoreResult {
  std::uint64_t inputs = 0, targets = 0, steps = 0;
  double total_nll = 0;
  double mean_nll() const;
  double perplexity() const;
};

// Mandatory wrapper for a component backend used by this scorer. Install it
// before runner Setup; it owns its callback and captures only the vocabulary.
kernels::LowerFunction GuardCorpusLower(kernels::LowerFunction, std::uint32_t vocabulary);

// The wrapper checks this before returning a program.
// The vocabulary comes from the validated full target spec/tokenizer, never a
// selected-head size. This verifies typed IR plus every physical output slot.
engine::Status CheckCorpusHead(const kernels::LowerInput&, const kernels::StepProgram&,
                               std::uint32_t vocabulary);

// Common target-only evaluator. No GPU ownership or asynchronous callback is
// introduced: Step uses the existing synchronous scored orchestrator. The node,
// runner and composer outlive this object; a harness must keep its entire owner
// alive on unknown device retirement, exactly as for ModelRunner itself.
// Make cannot authenticate an opaque runner: component callers MUST install
// GuardCorpusLower before Setup. Synthetic sources must enforce equivalent
// full-head semantics themselves. All fixture startup/compiler/state/device
// staging remains separately paid; Bound is not a whole-process/model limit.
class CorpusScore {
 public:
  static constexpr std::size_t kMaxDocuments = 16;
  static std::expected<std::uint64_t, std::string> Bound(std::span<const CorpusDocument>,
                                                         CorpusScoreLimits);
  static std::expected<std::unique_ptr<CorpusScore>, std::string> Make(
      engine::PagedNode&, engine::ModelRunner&, const engine::WaveComposer&,
      std::span<const CorpusDocument>, CorpusScoreLimits, CorpusScoreSink = {}, CorpusRowSink = {});
  ~CorpusScore();
  CorpusScore(const CorpusScore&) = delete;
  CorpusScore& operator=(const CorpusScore&) = delete;
  // Returns true only after all N inputs and N-1 targets per document. A
  // refusal/nonfinite output/cancellation is terminal, never a partial PASS.
  std::expected<bool, std::string> Step();
  void Cancel();
  bool complete() const { return complete_; }
  const CorpusScoreResult& result() const { return result_; }
  std::uint64_t charged_bytes() const { return charge_; }
  // Independent stable full-vocabulary score; rejects any nonfinite logit.
  static std::expected<double, std::string> Nll(std::span<const float>, std::int32_t);

 private:
  CorpusScore(engine::PagedNode& node, engine::ModelRunner& runner, CorpusScoreLimits limits,
              std::uint64_t charge)
      : node_(node), runner_(runner), limits_(limits), charge_(charge) {}
  struct Document {
    std::uint32_t slot = 0, size = 0, cursor = 0;
    std::uint64_t offset = 0;
  };
  engine::PagedNode& node_;
  engine::ModelRunner& runner_;
  CorpusScoreLimits limits_;
  std::uint64_t charge_;
  std::size_t count_ = 0;
  std::array<Document, kMaxDocuments> documents_{};
  std::unique_ptr<std::int32_t[]> tokens_;
  std::unique_ptr<float[]> row_;
  std::optional<engine::DecodingOrchestrator> decoder_;
  CorpusScoreResult result_{};
  CorpusScoreSink sink_;
  CorpusRowSink row_sink_;
  double compensation_ = 0;
  bool failed_ = false, complete_ = false, canceled_ = false;
};
}  // namespace llmp::test_support
#endif
