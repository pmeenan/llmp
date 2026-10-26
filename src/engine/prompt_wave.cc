// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "engine/prompt_wave.h"

#include <algorithm>
#include <vector>

#include "engine/batch_inventory.h"
#include "engine/model_runner.h"

namespace llmp::engine {

namespace pl = llmp::planner;

WaveReport RunPromptWave(ModelRunner& runner, const WaveComposer& composer,
                         std::span<const PendingWork> pending, std::span<const SegmentWork> work,
                         std::span<const std::span<const std::int32_t>> histories,
                         const AheadFor& ahead) {
  // Qualify every actual request phase before any state effect.
  for (const auto& request : pending) {
    NumericalUse use;
    switch (request.phase) {
      case pl::Phase::kPrefill:
        use = NumericalUse::kPrompt;
        break;
      case pl::Phase::kDecode:
        use = NumericalUse::kDecode;
        break;
      case pl::Phase::kScore:
        use = NumericalUse::kScore;
        break;
      default: {
        WaveReport report;
        report.outcomes.resize(pending.size());
        for (auto& out : report.outcomes) {
          out.refusal = StepRefusal{.what = "the target wave has no admitted semantic phase"};
        }
        return report;
      }
    }
    auto allowed = runner.CheckExecutionUse(use);
    if (!allowed) {
      WaveReport report;
      report.outcomes.resize(pending.size());
      for (auto& out : report.outcomes) {
        out.refusal = StepRefusal{.what = allowed.error()};
      }
      return report;
    }
  }

  const bool private_batch = PromptBatch::DeclaredFor(runner);
  HostPayment control_payment;
  if (private_batch) {
    const std::size_t control_bytes = std::max<std::size_t>(1, pending.size()) * 256;
    auto paid = runner.PayHostMetadata(control_bytes);
    if (!paid) {
      WaveReport report;
      report.outcomes.resize(pending.size());
      for (auto& out : report.outcomes) {
        out.refusal = StepRefusal{.what = paid.error(), .capacity_refused = true};
      }
      return report;
    }
    control_payment = std::move(*paid);
  }
  std::vector<SegmentWork> inputs(work.begin(), work.end());
  if (histories.size() == inputs.size()) {
    for (std::size_t i = 0; i < inputs.size(); ++i) {
      inputs[i].history = histories[i];
    }
  }

  const auto execute = [&](const PromptBatch& batch) -> WaveReport {
    return RunWave(runner, composer, pending, inputs, true, ahead, {}, batch);
  };

  if (!private_batch) return execute({});

  auto batch = PromptBatch::Prepare(runner, composer, pending, inputs, histories, {}, ahead);
  if (!batch) {
    WaveReport refused;
    (void)refused.HoldPayment(std::move(control_payment));
    refused.outcomes.resize(pending.size());
    for (auto& out : refused.outcomes) {
      out.refusal =
          StepRefusal{.what = batch.error().what,
                      .clean = true,
                      .capacity_refused = batch.error().error == kernels::ResourceError::kCapacity};
    }
    return refused;
  }
  auto report = execute(*batch);
  (void)report.HoldPayment(std::move(control_payment));
  if (auto closed = batch->Close(); !closed) {
    for (auto& out : report.outcomes) {
      if (!out.refusal) out.refusal = StepRefusal{.what = closed.error(), .clean = false};
    }
  }
  return report;
}

}  // namespace llmp::engine
