// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "engine/draft_companions.h"

#include <algorithm>
#include <optional>
#include <utility>
#include <vector>

namespace llmp::engine {

namespace pl = llmp::planner;

// Only this helper grants temporary access to compound-owned state. Its
// steps carry exact coordinates and owner/unit capabilities; arbitrary
// public RunStep calls cannot enter an unfinished prompt operation.
class PromptCompanions {
 public:
  PromptCompanions(ModelRunner& runner, const WaveComposer& composer,
                   std::span<const PendingWork> pending, std::span<const SegmentWork> work,
                   std::span<const std::span<const std::int32_t>> histories)
      : runner_(runner),
        slots_(runner.slots()),
        composer_(composer),
        pending_(pending),
        work_(work),
        histories_(histories),
        owners_(pending.size()) {
    report_.outcomes.resize(pending.size());
  }
  ~PromptCompanions() {
    if (!owns_schedule_) return;
    runner_.companion_steps_.clear();
    for (std::size_t i = 0; i < owners_.size(); ++i) {
      if (owners_[i].access) slots_.slots_[pending_[i].slot].compound_access = false;
    }
  }

  WaveReport Run(const AheadFor& ahead) {
    if (!runner_.companion_steps_.empty() || work_.size() != pending_.size()) {
      for (auto& out : report_.outcomes)
        out.refusal = StepRefusal{.what = "prompt companion work or ownership differs"};
      return std::move(report_);
    }
    owns_schedule_ = true;
    for (std::size_t i = 0; i < owners_.size(); ++i) Begin(i);
    for (std::size_t stage = 1; stage < slots_.stages(); ++stage) Companions(stage, false);

    std::vector<PendingWork> target;
    std::vector<SegmentWork> inputs;
    std::vector<std::size_t> of;
    runner_.companion_steps_.clear();
    for (std::size_t i = 0; i < owners_.size(); ++i) {
      if (!owners_[i].live) continue;
      target.push_back(pending_[i]);
      inputs.push_back(work_[i]);
      of.push_back(i);
      auto step = Step(i, 0, pending_[i].past, pending_[i].rows, Part::kTarget);
      for (const auto& consumer : owners_[i].consumers)
        step.units.emplace_back(consumer.stage, consumer.compound.unit);
      if (!step.units.empty()) runner_.companion_steps_.push_back(std::move(step));
    }
    const AheadFor mapped = [&](const ComposedStep& step, std::span<const std::size_t> indices) {
      std::vector<std::size_t> original;
      for (const auto i : indices) original.push_back(of[i]);
      return ahead ? ahead(step, original) : StepAhead{};
    };
    auto target_report = RunWave(runner_, composer_, target, inputs, false, mapped);
    report_.steps += target_report.steps;
    report_.recomposed += target_report.recomposed;
    for (std::size_t j = 0; j < of.size(); ++j) {
      const auto i = of[j];
      if (target_report.outcomes[j].refusal) {
        Refuse(i, std::move(*target_report.outcomes[j].refusal));
      } else {
        report_.outcomes[i] = std::move(target_report.outcomes[j]);
      }
    }
    runner_.companion_steps_.clear();
    for (std::size_t stage = 1; stage < slots_.stages(); ++stage) Companions(stage, true);
    for (std::size_t i = 0; i < owners_.size(); ++i) {
      Owner& owner = owners_[i];
      if (!owner.live) continue;
      for (const Consumer& consumer : owner.consumers) {
        auto finished = slots_.progress(pending_[i].slot, consumer.stage)
                            ->FinishCompound(consumer.compound.unit);
        if (!finished) {
          Refuse(i, {.what = finished.error(), .clean = false});
          break;
        }
      }
      if (!owner.live) continue;
      const auto slot = pending_[i].slot;
      // The completed control, then target positions, become visible as
      // one synchronous boundary. A publication failure quarantines it.
      auto settled = slots_.Settle(slot);
      auto published =
          settled ? slots_.SetPositions(slot, pending_[i].past + pending_[i].rows) : settled;
      if (!published) Refuse(i, {.what = published.error(), .clean = false});
    }
    return std::move(report_);
  }

 private:
  using Part = ModelRunner::CompanionPart;
  struct Consumer {
    std::size_t stage = 0;
    DraftProgressLedger::Compound compound;
  };
  struct Owner {
    bool live = false, access = false;
    std::vector<Consumer> consumers;
  };
  ModelRunner::CompanionStep Step(std::size_t i, std::uint16_t stage, std::uint32_t past,
                                  std::uint32_t rows, Part part) {
    return {.slot = pending_[i].slot,
            .stage = stage,
            .past = past,
            .rows = rows,
            .part = part,
            .units = {}};
  }
  void Refuse(std::size_t i, StepRefusal refusal) {
    Owner& owner = owners_[i];
    bool safe = refusal.clean;
    for (const Consumer& consumer : owner.consumers) {
      auto* progress = slots_.progress(pending_[i].slot, consumer.stage);
      if (!progress->RefuseCompound(consumer.compound.unit)) safe = false;
    }
    if (!safe && pending_[i].slot < slots_.slots())
      slots_.Quarantine(slots_.slots_[pending_[i].slot]);
    refusal.clean = safe;
    refusal.slot = pending_[i].slot;
    report_.outcomes[i].refusal = std::move(refusal);
    owner.live = false;
    if (owner.access) {
      slots_.slots_[pending_[i].slot].compound_access = false;
      owner.access = false;
    }
  }
  void Begin(std::size_t i) {
    const auto& p = pending_[i];
    if (p.slot >= slots_.slots() || p.stage != 0 || p.phase == planner::Phase::kVerify ||
        p.phase == planner::Phase::kCommit || p.past != slots_.positions(p.slot) || p.rows == 0 ||
        std::ranges::any_of(pending_.first(i), [&](const auto& q) { return q.slot == p.slot; })) {
      Refuse(i, {.what = "prompt does not continue one target owner"});
      return;
    }
    auto ready = slots_.CheckActive(p.slot);
    if (ready && slots_.owed(p.slot)) ready = slots_.Settle(p.slot);
    if (!ready) {
      Refuse(i, {.what = ready.error(), .clean = slots_.usable(p.slot)});
      return;
    }
    bool qualified = false;
    for (std::size_t s = 1; s < slots_.stages(); ++s)
      qualified |= slots_.progress(p.slot, s) != nullptr;
    if (qualified &&
        (histories_.size() != pending_.size() ||
         histories_[i].size() < std::uint64_t{p.past} + p.rows ||
         work_[i].tokens.size() != p.rows ||
         !std::ranges::equal(work_[i].tokens, histories_[i].subspan(p.past, p.rows)))) {
      Refuse(i, {.what = "prompt companion needs its complete token history"});
      return;
    }
    Owner& owner = owners_[i];
    for (std::size_t s = 1; s < slots_.stages(); ++s) {
      auto* progress = slots_.progress(p.slot, s);
      if (progress == nullptr) continue;
      auto compound = progress->BeginCompound(p.rows);
      if (!compound) {
        Refuse(i, {.what = compound.error()});
        return;
      }
      owner.consumers.push_back({.stage = s, .compound = *compound});
    }
    owner.live = true;
    owner.access = !owner.consumers.empty();
    slots_.slots_[p.slot].compound_access = owner.access;
  }
  void Companions(std::size_t stage, bool after) {
    std::vector<PendingWork> pending;
    std::vector<SegmentWork> work;
    std::vector<std::size_t> of;
    runner_.companion_steps_.clear();
    for (std::size_t i = 0; i < owners_.size(); ++i) {
      const Owner& owner = owners_[i];
      if (!owner.live) continue;
      for (const Consumer& consumer : owner.consumers) {
        if (consumer.stage != stage) continue;
        const auto& rows = after ? consumer.compound.shapes.after : consumer.compound.shapes.before;
        if (rows.count == 0) continue;
        const auto own_stage = static_cast<std::uint16_t>(stage);
        pending.push_back({.slot = pending_[i].slot,
                           .stage = own_stage,
                           .phase = planner::Phase::kDraft,
                           .past = rows.cache_first,
                           .rows = rows.count,
                           .outputs = 0,
                           .want_head = false});
        work.push_back(
            {.tokens = histories_[i].subspan(rows.token_first, rows.count),
             .feature = {.kind = FeatureFrom::Kind::kTap, .position = rows.feature_first},
             .history = histories_[i]});
        of.push_back(i);
        auto step =
            Step(i, own_stage, rows.cache_first, rows.count, after ? Part::kPost : Part::kPre);
        step.units.emplace_back(stage, consumer.compound.unit);
        runner_.companion_steps_.push_back(std::move(step));
      }
    }
    auto wave = RunWave(runner_, composer_, pending, work, false);
    report_.steps += wave.steps;
    report_.recomposed += wave.recomposed;
    for (std::size_t j = 0; j < of.size(); ++j)
      if (wave.outcomes[j].refusal) Refuse(of[j], std::move(*wave.outcomes[j].refusal));
    runner_.companion_steps_.clear();
  }

  ModelRunner& runner_;
  SlotLifecycle& slots_;
  const WaveComposer& composer_;
  std::span<const PendingWork> pending_;
  std::span<const SegmentWork> work_;
  bool owns_schedule_ = false;
  std::span<const std::span<const std::int32_t>> histories_;
  std::vector<Owner> owners_;
  WaveReport report_;
};

WaveReport RunPromptWave(ModelRunner& runner, const WaveComposer& composer,
                         std::span<const PendingWork> pending, std::span<const SegmentWork> work,
                         std::span<const std::span<const std::int32_t>> histories,
                         const AheadFor& ahead) {
  if (auto qualified = runner.CheckOwners(pending.size()); !qualified) {
    WaveReport report;
    report.outcomes.resize(pending.size());
    for (auto& out : report.outcomes) out.refusal = StepRefusal{.what = qualified.error()};
    return report;
  }
  bool companions = false;
  for (std::size_t s = 1; s < runner.slots().stages(); ++s)
    companions |= runner.slots().progress(0, s) != nullptr;
  // This shared transport serves prefill, ordinary decode and literal scoring.
  // Qualify every actual request phase before any companion or state effect.
  // Companion prefill retains its existing compound permission; plain/scored
  // decode also retains its own semantic permission at this direct entry.
  const auto qualify = [&]() -> Status {
    if (companions) {
      auto allowed = runner.CheckExecutionUse(NumericalUse::kPromptCompanions);
      if (!allowed) return allowed;
    }
    for (const auto& request : pending) {
      NumericalUse use;
      switch (request.phase) {
        case pl::Phase::kPrefill:
          if (companions) continue;
          use = NumericalUse::kPrompt;
          break;
        case pl::Phase::kDecode:
          use = NumericalUse::kDecode;
          break;
        case pl::Phase::kScore:
          use = NumericalUse::kScore;
          break;
        default:
          return std::unexpected("the target wave has no admitted semantic phase");
      }
      auto allowed = runner.CheckExecutionUse(use);
      if (!allowed) return allowed;
    }
    return {};
  };
  if (auto qualified = qualify(); !qualified) {
    WaveReport report;
    report.outcomes.resize(pending.size());
    for (auto& out : report.outcomes) out.refusal = StepRefusal{.what = qualified.error()};
    return report;
  }
  std::vector<SegmentWork> inputs(work.begin(), work.end());
  if (histories.size() == inputs.size())
    for (std::size_t i = 0; i < inputs.size(); ++i) inputs[i].history = histories[i];
  if (!companions) return RunWave(runner, composer, pending, inputs, true, ahead);
  if (ahead && !runner.CheckOwners(2)) {
    // Only one owner is admissible here. Freeze its target lookahead
    // before a pre companion can settle or overwrite state; checking the
    // eventual RunStep alone would be too late for the compound operation.
    const auto composition = composer.Compose(pending);
    if (composition.steps.size() == 1 && composition.refused.empty()) {
      const auto& step = composition.steps.front();
      const StepAhead predicted = ahead(step, step.work);
      for (const auto& future : predicted.shapes) {
        if (!future) continue;
        if (auto qualified = runner.CheckOwners(future->segments.size()); !qualified) {
          WaveReport report;
          report.outcomes.resize(pending.size());
          for (auto& out : report.outcomes) out.refusal = StepRefusal{.what = qualified.error()};
          return report;
        }
      }
      const AheadFor frozen = [&](const ComposedStep&, std::span<const std::size_t>) {
        return predicted;
      };
      return PromptCompanions(runner, composer, pending, inputs, histories).Run(frozen);
    }
  }
  return PromptCompanions(runner, composer, pending, inputs, histories).Run(ahead);
}

}  // namespace llmp::engine
