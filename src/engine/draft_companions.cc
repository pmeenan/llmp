// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "engine/draft_companions.h"

#include <algorithm>
#include <array>
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
  static std::expected<std::uint64_t, std::string> ControlBytes(std::size_t owners,
                                                                std::size_t stages);
  PromptCompanions(ModelRunner& runner, const WaveComposer& composer,
                   std::span<const PendingWork> pending, std::span<const SegmentWork> work,
                   std::span<const std::span<const std::int32_t>> histories,
                   const PromptBatch& batch = {})
      : runner_(runner),
        slots_(runner.slots()),
        composer_(composer),
        pending_(pending),
        work_(work),
        histories_(histories),
        batch_(batch),
        owners_(pending.size()) {
    report_.outcomes.resize(pending.size());
    for (auto& owner : owners_) {
      owner.consumers.reserve(slots_.stages());
      owner.witnesses.reserve(slots_.stages());
    }
  }
  ~PromptCompanions() {
    if (!owns_schedule_) return;
    for (std::size_t i = 0; i < owners_.size(); ++i) {
      if (owners_[i].access) slots_.slots_[pending_[i].slot].compound_access = false;
    }
  }

  WaveReport Run(const AheadFor& ahead) {
    if (work_.size() != pending_.size()) {
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
    std::vector<StepAuthorization> authorization;
    target.reserve(owners_.size());
    inputs.reserve(owners_.size());
    of.reserve(owners_.size());
    authorization.reserve(owners_.size());
    for (std::size_t i = 0; i < owners_.size(); ++i) {
      if (!owners_[i].live) continue;
      target.push_back(pending_[i]);
      inputs.push_back(work_[i]);
      of.push_back(i);
      auto step = Step(i, 0, pending_[i].past, pending_[i].rows, Part::kTarget);
      step.consumers_ = owners_[i].witnesses;
      step.compound_tail_ = owners_[i].closing_stage == 0;
      authorization.push_back(step);
    }
    const AheadFor mapped = [&](const ComposedStep& step, std::span<const std::size_t> indices) {
      std::vector<std::size_t> original;
      original.reserve(owners_.size());
      for (const auto i : indices) original.push_back(of[i]);
      return ahead ? ahead(step, original) : StepAhead{};
    };
    auto target_report =
        RunWave(runner_, composer_, target, inputs, false, mapped, authorization, batch_);
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
    for (std::size_t stage = 1; stage < slots_.stages(); ++stage) Companions(stage, true);
    for (std::size_t i = 0; i < owners_.size(); ++i) {
      if (!owners_[i].live) continue;
      const auto slot = pending_[i].slot;
      bool complete = slots_.positions(slot) == pending_[i].past + pending_[i].rows;
      for (const auto& consumer : owners_[i].consumers) {
        const auto* progress = slots_.progress(slot, consumer.stage);
        complete = complete && progress->settled() &&
                   progress->completed().positions == slots_.positions(slot);
      }
      if (!complete)
        Refuse(i, {.what = "compound final publication did not retire", .clean = false});
    }
    return std::move(report_);
  }

 private:
  using Part = StepAuthorization::Part;
  struct Consumer {
    std::size_t stage = 0;
    DraftProgressLedger::Compound compound;
  };
  struct Owner {
    bool live = false, access = false;
    std::size_t closing_stage = 0;
    std::vector<Consumer> consumers;
    std::vector<std::pair<std::size_t, DraftProgressLedger::Unit>> witnesses;
  };
  StepAuthorization Step(std::size_t i, std::uint16_t stage, std::uint32_t past, std::uint32_t rows,
                         Part part) {
    StepAuthorization permit;
    permit.runner_ = &runner_;
    permit.slot_ = pending_[i].slot;
    permit.stage_ = stage;
    permit.phase_ = part == Part::kTarget ? pending_[i].phase : planner::Phase::kDraft;
    permit.past_ = past;
    permit.rows_ = rows;
    permit.outputs_ = part == Part::kTarget ? pending_[i].outputs : 0;
    permit.want_head_ = part == Part::kTarget ? pending_[i].want_head : false;
    permit.greedy_ = part == Part::kTarget ? pending_[i].greedy : false;
    permit.part_ = part;
    return permit;
  }
  void Refuse(std::size_t i, StepRefusal refusal) {
    isolate_ = isolate_ || bool(batch_);
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
      if (compound->shapes.after.count != 0) owner.closing_stage = s;
      owner.witnesses.emplace_back(s, compound->unit);
    }
    owner.live = true;
    owner.access = !owner.consumers.empty();
    slots_.slots_[p.slot].compound_access = owner.access;
  }
  void Companions(std::size_t stage, bool after) {
    // A borrowed-target companion reads a completed target between the
    // surrounding synchronous target steps. Freeze that stage only for this
    // useful companion wave, then release before the target can write again.
    std::array<StateBorrow, kMaxRequestSlots> borrows;
    std::vector<PendingWork> pending;
    std::vector<SegmentWork> work;
    std::vector<std::size_t> of;
    std::vector<StepAuthorization> authorization;
    pending.reserve(owners_.size());
    work.reserve(owners_.size());
    of.reserve(owners_.size());
    authorization.reserve(owners_.size());
    for (std::size_t i = 0; i < owners_.size(); ++i) {
      const Owner& owner = owners_[i];
      if (!owner.live) continue;
      for (const Consumer& consumer : owner.consumers) {
        if (consumer.stage != stage) continue;
        const auto& rows = after ? consumer.compound.shapes.after : consumer.compound.shapes.before;
        if (rows.count == 0) continue;
        if (!slots_.layout(stage).borrowed.empty()) {
          const auto slot = pending_[i].slot;
          auto borrowed = slots_.owed(slot) ? slots_.BorrowAfterCompoundPrefix(
                                                  slot, stage, consumer.compound.unit,
                                                  rows.cache_first, rows.count)
                                            : slots_.Borrow(slot, 0);
          if (!borrowed) {
            Refuse(i, {.what = borrowed.error(), .clean = slots_.usable(pending_[i].slot)});
            break;
          }
          borrows[pending_[i].slot] = std::move(*borrowed);
        }
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
        step.feature_ = work.back().feature;
        step.borrowed_ = borrows[pending_[i].slot].permit();
        const auto witness = std::ranges::find(
            owner.witnesses, stage, &std::pair<std::size_t, DraftProgressLedger::Unit>::first);
        step.compound_tail_ = after && stage == owner.closing_stage;
        step.consumers_ =
            step.compound_tail_ ? std::span(owner.witnesses) : std::span(&*witness, 1);
        authorization.push_back(step);
      }
    }
    if (isolate_)
      for (auto& item : pending) item.alone = true;
    auto wave = RunWave(runner_, composer_, pending, work, false, {}, authorization, batch_);
    report_.steps += wave.steps;
    report_.recomposed += wave.recomposed;
    for (std::size_t j = 0; j < of.size(); ++j)
      if (wave.outcomes[j].refusal) Refuse(of[j], std::move(*wave.outcomes[j].refusal));
  }

  ModelRunner& runner_;
  SlotLifecycle& slots_;
  const WaveComposer& composer_;
  std::span<const PendingWork> pending_;
  std::span<const SegmentWork> work_;
  bool owns_schedule_ = false, isolate_ = false;
  std::span<const std::span<const std::int32_t>> histories_;
  PromptBatch batch_;
  std::vector<Owner> owners_;
  WaveReport report_;
};

std::expected<std::uint64_t, std::string> PromptCompanions::ControlBytes(std::size_t owners,
                                                                         std::size_t stages) {
  // Exact peak container capacities used by the companion and shared wave
  // orchestrators. Every stage is sequential; only its consumer descriptors
  // coexist. Compose reserves N segment/index slots per potential step.
  std::uint64_t bytes = 0, square = 0, consumers = 0;
  const auto add = [&](std::uint64_t n, std::uint64_t size) {
    std::uint64_t amount = 0;
    return !__builtin_mul_overflow(n, size, &amount) &&
           !__builtin_add_overflow(bytes, amount, &bytes) && bytes <= PTRDIFF_MAX;
  };
  if (__builtin_mul_overflow(std::uint64_t{owners}, std::uint64_t{owners}, &square) ||
      __builtin_mul_overflow(std::uint64_t{owners}, std::uint64_t{stages}, &consumers) ||
      !add(owners, sizeof(Owner) + 2 * sizeof(WaveOutcome) + 4 * sizeof(PendingWork) +
                       4 * sizeof(SegmentWork) + 4 * sizeof(StepAuthorization) +
                       8 * sizeof(std::size_t) + 2 * sizeof(pl::Segment) + sizeof(ComposedStep) +
                       sizeof(bool) + sizeof(std::pair<std::size_t, std::string>)) ||
      !add(consumers,
           sizeof(Consumer) + sizeof(std::pair<std::size_t, DraftProgressLedger::Unit>) + 2 * 64) ||
      !add(square, sizeof(pl::Segment) + sizeof(std::size_t) + 2 * 64) || !add(32, 64))
    return std::unexpected("batch orchestration metadata capacity overflows");
  return bytes;
}

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
  const bool private_batch = PromptBatch::DeclaredFor(runner);
  HostPayment control_payment;
  if (private_batch) {
    auto bytes = PromptCompanions::ControlBytes(pending.size(), runner.slots().stages());
    auto paid = bytes ? runner.PayHostMetadata(*bytes)
                      : std::expected<HostPayment, std::string>{std::unexpected(bytes.error())};
    if (!paid) {
      WaveReport refused;
      // Required work has not begun. The caller's already admitted public
      // outcome inventory owns this refusal transport, just as legacy waves.
      refused.outcomes.resize(pending.size());
      for (auto& out : refused.outcomes)
        out.refusal = StepRefusal{.what = paid.error(), .capacity_refused = true};
      return refused;
    }
    control_payment = std::move(*paid);
  }
  std::vector<SegmentWork> inputs(work.begin(), work.end());
  if (histories.size() == inputs.size())
    for (std::size_t i = 0; i < inputs.size(); ++i) inputs[i].history = histories[i];
  const auto execute = [&](const PromptBatch& batch) -> WaveReport {
    if (!companions) return RunWave(runner, composer, pending, inputs, true, ahead, {}, batch);
    if (!batch && ahead && !runner.CheckOwners(2)) {
      // Only one owner is admissible here. Freeze its target lookahead
      // before a pre companion can settle or overwrite state; checking the
      // eventual RunStep alone would be too late for the compound operation.
      const auto composition = composer.Compose(pending);
      if (composition.steps.size() == 1 && composition.refused.empty()) {
        const auto& step = composition.steps.front();
        const StepAhead predicted = ahead(step, step.work);
        for (const auto& future : predicted.shapes()) {
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
        return PromptCompanions(runner, composer, pending, inputs, histories, batch).Run(frozen);
      }
    }
    return PromptCompanions(runner, composer, pending, inputs, histories, batch).Run(ahead);
  };
  if (!private_batch) return execute({});
  auto batch = PromptBatch::Prepare(runner, composer, pending, inputs, histories, {}, ahead);
  if (!batch) {
    WaveReport refused;
    (void)refused.HoldPayment(std::move(control_payment));
    refused.outcomes.resize(pending.size());
    for (auto& out : refused.outcomes)
      out.refusal =
          StepRefusal{.what = batch.error().what,
                      .clean = true,
                      .capacity_refused = batch.error().error == kernels::ResourceError::kCapacity};
    return refused;
  }
  auto report = execute(*batch);
  (void)report.HoldPayment(std::move(control_payment));
  if (auto closed = batch->Close(); !closed) {
    for (auto& out : report.outcomes)
      if (!out.refusal) out.refusal = StepRefusal{.what = closed.error(), .clean = false};
  }
  return report;
}

}  // namespace llmp::engine
