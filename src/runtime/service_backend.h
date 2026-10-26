// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
// Internal model adapter: stable request owners and explicit cooperative
// selection/retirement state. Implementations are split at the driver seam.
#ifndef LLMP_RUNTIME_SERVICE_BACKEND_H_
#define LLMP_RUNTIME_SERVICE_BACKEND_H_
#include <array>
#include <list>
#include <optional>
#include <string_view>

#include "base/check.h"
#include "runtime/cohort_capacity.h"
#include "runtime/prefill.h"
#include "runtime/service_request.h"
#include "scheduler/cohort_schedule.h"
namespace llmp::runtime::service_detail {
class NodeBackend final : public api::Backend, public api::CooperativeBackend {
  class ChatWork final : public api::CooperativeBackend::Work {
   public:
    enum class Stage { kPrompt, kGeneration, kOutput, kDone };
    ChatWork(NodeBackend& owner, Llm& model, Llm::Branch& branch, std::size_t slot,
             const api::ChatRequest& request, api::Exchange& exchange, ChatPrompt rendered,
             Floors floors, std::uint64_t swap_bytes,
             std::shared_ptr<api::LiteralOutput> literal = nullptr)
        : owner(owner),
          model(model),
          branch(branch),
          slot(slot),
          request(request),
          exchange(exchange),
          rendered(std::move(rendered)),
          floors(floors),
          swap_bytes(swap_bytes),
          output(model, exchange, this->rendered.reasoning, this->rendered.tools,
                 this->rendered.output_charge),
          literal(std::move(literal)) {
      options.max_tokens = this->rendered.max_tokens;
      options.stop = true;
      if (this->rendered.tools) options.tool_sampling = this->rendered.tools->sampling;
      if (this->literal == nullptr) options.extra_stops = model.ChatStops();
      options.sampling = SamplingOf(request, model);
      if (options.sampling) {
        options.seed = request.seed.value_or(RandomSeed());
      }
      options.on_tokens = [this](std::span<const std::int32_t> fresh) {
        if (!FundTokens(true)) {
          return false;
        }
        if (this->literal == nullptr) {
          return output.Push(fresh);
        }
        bool said = false;
        const bool sent = this->literal->Push(fresh, [this, &said](std::string_view piece) {
          said = true;
          return this->exchange.Content(piece);
        });
        return sent && (said || this->exchange.Continue());
      };
      if (this->literal != nullptr && this->literal->score_generation()) {
        options.on_logits = [this](std::int32_t id, std::span<const float> row) {
          return this->literal->GeneratedRow(id, row) && this->exchange.Continue();
        };
      }
    }
    ~ChatWork() override {
      base::Check(retired, "a cooperative chat owner was freed before retirement");
    }
    ChatWork(const ChatWork&) = delete;
    ChatWork& operator=(const ChatWork&) = delete;
    ChatWork(ChatWork&&) = delete;
    ChatWork& operator=(ChatWork&&) = delete;
    bool terminal() const override {
      return cancelled || yielding || error.has_value() || stage == Stage::kDone ||
             (generation_session != nullptr && generation_session->done() &&
              (literal != nullptr || output.done()));
    }
    void Cancel() override {
      cancelled = true;
      yielding = false;  // cancelled, it ends; it does not continue
    }
    // Yields a generation under way that is not set aside or waiting for
    // capacity (its continuation is its generation so far), or output not
    // yet delivered (host-only: its continuation finishes it).
    bool Yield() override {
      if ((generation_session == nullptr && stage != Stage::kOutput) ||
          (generation_session && generation_session->done() && output.done()) ||
          generation.tokens.empty() || wait != CapacityWait::kNone || resume || error ||
          cancelled) {
        return false;
      }
      yielding = true;
      return true;
    }
    // Pauses for another model only while device work remains: a member
    // whose generation is complete has only its output to deliver, host-only
    // work that needs no residency, so it finishes in this cohort (or yields
    // its place to a slow reader, Yield) instead of carrying a continuation.
    bool YieldForSwitch() override {
      if (terminal() || stage == Stage::kOutput ||
          (generation_session != nullptr && generation_session->done())) {
        return false;
      }
      if (generation.tokens.empty() && prompt_session != nullptr &&
          prompt_session->remaining_rows() == 0) {
        // The final prefill head may already exist while its checkpoint
        // is still owed. Complete that boundary and choose its anchor
        // before pausing; otherwise cancelling would discard the head
        // and an equal-history resume could recompute a different one.
        return false;
      }
      yielding = true;
      model_paused = true;
      return true;
    }

    bool FundTokens(bool granted) {
      const std::uint64_t held =
          (generation.tokens.capacity() + resume_tokens.capacity()) * sizeof(std::int32_t);
      if (auto refused = FundHistory(token_charge, *owner.memory_, held, granted)) {
        error = std::move(refused);
        return false;
      }
      return true;
    }

    NodeBackend& owner;
    Llm& model;
    Llm::Branch& branch;
    const std::size_t slot;
    const api::ChatRequest& request;
    api::Exchange& exchange;
    const ChatPrompt rendered;
    const Floors floors;
    const std::uint64_t swap_bytes;
    ChatOutput output;
    std::shared_ptr<api::LiteralOutput> literal;
    GenerateOptions options;
    Generation generation;
    std::unique_ptr<Llm::PromptSession> prompt_session;
    std::unique_ptr<Llm::GenerationSession> generation_session;
    std::optional<api::Error> error;
    Stage stage = Stage::kPrompt;
    std::uint32_t reused = 0;
    bool cancelled = false;
    bool yielding = false;      // yields its place (Yield): retired with a continuation
    bool model_paused = false;  // prompt or generation paused for another model
    bool native_touched = false;
    bool retired = false;
    // State capacity (cohort_capacity.h): admission order, whether it
    // waits, the refusal it fails with if it cannot fit alone, and after a
    // preemption mid-generation, the tokens its state is rebuilt from (the
    // prompt and every generated token but the unprocessed anchor).
    std::uint64_t admitted = 0;
    // The schedule's counters (cohort_schedule.h): while it prefills, the
    // other prompt units since its own last; while it generates, the prompt
    // units since its last wave.
    std::uint32_t passed = 0;
    std::uint32_t waited = 0;
    CapacityWait wait = CapacityWait::kNone;
    std::string refusal;
    bool resume = false;
    bool continued = false;       // original cached-token usage survives a prompt pause too
    bool switch_restore = false;  // restore generating peers before resuming their wave
    std::vector<std::int32_t> resume_tokens;
    MemoryCharge token_charge;
  };
  static_assert(kCohortSlots == Llm::kMaxBranches);

 public:
  // `memory`: the request memory (intake_limits.h), which the chat route
  // shares; renderings, tokenizations and literal scores are charged to it.
  NodeBackend(Server& server, const config::NodeConfig& config,
              std::shared_ptr<RequestMemory> memory, std::FILE* log, HangLadder* ladder = nullptr);

  // Chat and literal requests share funded independent slots. Unsupported
  // models retain their ordinary entry points.
  api::CooperativeBackend* cooperative() override;

  bool Supports(const api::CooperativeBackend::Request& request) const override;

  std::expected<std::unique_ptr<api::CooperativeBackend::Work>, api::Error> Start(
      const api::CooperativeBackend::Request& request, api::Exchange& exchange) override;

  std::expected<api::CooperativeBackend::Unit, std::string> NextUnit(
      std::span<api::CooperativeBackend::Work* const> work) override;

  std::expected<void, std::string> Advance(
      std::span<api::CooperativeBackend::Work* const> work) override;

  api::CooperativeBackend::Retirement Retire(api::CooperativeBackend::Work& work) override;

  std::vector<api::ModelInfo> Models() const override;

  std::expected<api::Completion, api::Error> Complete(const api::ChatRequest& request,
                                                      api::Exchange& exchange) override;

  std::expected<api::Completion, api::Error> Resume(const api::ChatRequest& request,
                                                    api::Exchange& exchange,
                                                    const api::Yielded& from) override;

  // A chat request on the serial path, from its rendering, or continuing
  // from where it yielded its place (`resume`).
  std::expected<api::Completion, api::Error> SerialChat(Llm& l, const api::ChatRequest& request,
                                                        api::Exchange& exchange,
                                                        ChatPrompt rendered,
                                                        const ChatResume* resume);

  std::expected<api::Completion, api::Error> Complete(const api::CompletionRequest& request,
                                                      api::Exchange& exchange) override;

  void AfterResponse() override;

  // Between units and while idle: idle conversations past their retention
  // or the spill budget, and memory pressure from outside (Server::Maintain).
  void Maintain() override;
  bool healthy() const override;
  std::string failure() const override;
  std::expected<api::ReleaseReport, std::string> ReleaseWorkingSet(
      std::string_view reason = "manual") override;

 private:
  Status AdvancePrefillWave();

  std::uint64_t SwapBytes(Llm& model) const;

  Status CheckCohort(std::span<api::CooperativeBackend::Work* const> work) const;

  Status SelectCohort();

  static std::expected<std::unique_ptr<Llm::PromptSession>, std::string> WorkPrompt(
      ChatWork& frame);

  static Status BeginChatGeneration(ChatWork& frame);

  // State capacity (cohort_capacity.h; docs/runtime-serving.md#state-
  // capacity-in-a-cohort). The cohort as the policy sees it, and back.
  Cohort Snapshot() const;
  void Apply(const Cohort& cohort);

  // Within Advance, the cohort selected: members whose unit the state's
  // capacity refused (each with its `refusal`, its state usable and its
  // completed prefix kept) wait, fail if they cannot fit alone, or free
  // the youngest waiter's state when no member could otherwise go on.
  void WaitForCapacity(std::span<ChatWork* const> refused);

  // A serial request runs alone, as a cohort of one: refused for capacity,
  // it has the node's reclaim order free what its unit asked for (plans,
  // graphs, idle conversations spilled) and runs again.
  Llm::CapacityReclaim SerialReclaim(Llm& model);

  // Positions only (D-014).
  void RefusedAlone(const Llm& model);

  // Frees a waiting member's state between its completed units: its
  // sessions end at their completed prefix, the host keeps the tokens to
  // rebuild it from, and its state is discarded (Branch::Clear). It begins
  // again once no other member can run (Rebalance).
  void Preempt(ChatWork& frame);

  // After a unit or a retirement: when no member can go on, waiting
  // members retry, or the oldest preempted one begins its prompt again
  // (host-only; its first unit reuses or clears the branch as usual).
  void Rebalance();

  // What a prompt of `tokens` on `branch` has yet to take of the budget:
  // its state through its tokens and its first step, less what the branch
  // holds now (continued, or cleared for it).
  static std::expected<std::uint64_t, std::string> PromptStateToCome(const Llm& model,
                                                                     const Llm::Branch& branch,
                                                                     std::size_t tokens);

  void Fail(std::string what);

  // Collects ready activation receipts without draining outstanding work.
  void CollectSwaps();

  // Makes `m` resident for requests' work (Server::Activate; D-102,
  // recovery first): nothing, or the error those requests get. A swap whose
  // hung work was cancelled waits for the cancellation to drain (when it
  // does not, rung 3: a HungFailure); a node that cannot go on fails the
  // service (`what` names the model); otherwise only these requests fail,
  // with a 503 to retry.
  std::optional<api::Error> ActivateFor(Served& m, std::string_view what);
  // D-102's hang recovery, rung 2: a unit's work hung and its cancellation
  // drained (the node's waits flag it). The requests that needed it fail
  // with a 503 to retry, and the model is reset in place (Server::
  // RecoverModel); when that cannot be done, rung 3 (the ladder's last
  // resort restarts the process). `what` names the work.
  api::Error RecoverFromHang(Served& m, std::string_view what);
  // After a hang's cancellation (rung 1), the driver waits until it drained
  // (the ladder watches again: the driver moved and nothing the work
  // submitted is still in flight) before it frees or reuses anything the
  // work touched. False when it did not drain within its grace: the ladder
  // entered rung 3 (whose last resort exits; a test's returns).
  bool AwaitDrained();
  static api::Error HungFailure();
  // A cooperative unit's hang (rung 2 in a cohort): every member fails with
  // a 503; the model is reset once the last of them retires (Retire), after
  // a fence of its stream proves their native references retired.
  std::expected<void, std::string> HangCohort(std::string_view what);

  Server& server_;
  const config::NodeConfig& config_;
  std::shared_ptr<RequestMemory> memory_;  // the request memory (intake_limits.h)
  std::FILE* log_;
  HangLadder* ladder_ = nullptr;
  // The model a cohort's hang left to reset once its members retire.
  Llm* hang_reset_ = nullptr;
  // Each list entry is prepaid with its receipt; a pending entry survives
  // only while that activation owns unfinished work. Completed ones drain
  // between units, responses and activations, without a fixed count cap.
  std::list<SwapParts> pending_swaps_;
  std::string failure_;  // a node failure: the service stops
  std::array<ChatWork*, Llm::kMaxBranches> cohort_{};
  Llm* cohort_model_ = nullptr;
  ChatWork* selected_prompt_ = nullptr;
  std::array<ChatWork*, Llm::kMaxBranches> selected_prefill_{};
  std::size_t selected_prefill_count_ = 0;
  std::vector<ChatWork*> selected_decode_;
  std::vector<ChatWork*> selected_output_;
  std::uint64_t admissions_ = 0;
  bool selected_swap_ = false;
  bool cohort_native_touched_ = false;
  // The queue's head found no memory for a request slot: it waits until a
  // member retires (Start, Retire).
  MemoryWait memory_wait_;
};
}  // namespace llmp::runtime::service_detail
#endif  // LLMP_RUNTIME_SERVICE_BACKEND_H_
