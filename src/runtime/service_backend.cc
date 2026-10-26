// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
#include "runtime/service_backend.h"

#include <thread>

namespace llmp::runtime::service_detail {
namespace {
constexpr auto kDrainPoll = std::chrono::milliseconds(20);
}
NodeBackend::NodeBackend(Server& server, const config::NodeConfig& config,
                         std::shared_ptr<RequestMemory> memory, std::FILE* log, HangLadder* ladder)
    : server_(server), config_(config), memory_(std::move(memory)), log_(log), ladder_(ladder) {}

api::CooperativeBackend* NodeBackend::cooperative() { return this; }

std::vector<api::ModelInfo> NodeBackend::Models() const {
  std::vector<api::ModelInfo> models;
  for (const config::ModelEntry& entry : config_.models) {
    if (entry.composition) {
      models.push_back({.name = entry.name, .chat = false, .context = 0, .render_bytes = 0});
      continue;
    }
    Served* m = server_.Find(entry.name);
    if (m != nullptr && m->llm()) {
      const auto& l = static_cast<Llm&>(*m);
      models.push_back({.name = entry.name,
                        .chat = true,
                        .context = l.usable_context(),
                        .render_bytes = l.render_bytes()});
    }
  }
  return models;
}

std::expected<api::Completion, api::Error> NodeBackend::Complete(const api::ChatRequest& request,
                                                                 api::Exchange& exchange) {
  Served* m = server_.Find(request.model);
  if (m == nullptr || !m->llm()) {
    return std::unexpected(Failure(404, "The model does not exist", "model_not_found", "model"));
  }
  auto& l = static_cast<Llm&>(*m);
  auto rendered = PrepareChat(l, request, exchange, *memory_);
  if (!rendered) {
    return std::unexpected(rendered.error());
  }
  return SerialChat(l, request, exchange, std::move(*rendered), nullptr);
}

std::expected<api::Completion, api::Error> NodeBackend::Resume(const api::ChatRequest& request,
                                                               api::Exchange& exchange,
                                                               const api::Yielded& from) {
  Served* m = server_.Find(request.model);
  if (m == nullptr || !m->llm()) {
    return std::unexpected(Failure(404, "The model does not exist", "model_not_found", "model"));
  }
  // Only this backend yields, so only its continuations come back.
  const auto& resume = static_cast<const ChatResume&>(from);
  return SerialChat(static_cast<Llm&>(*m), request, exchange, resume.rendered, &resume);
}

std::expected<api::Completion, api::Error> NodeBackend::SerialChat(Llm& l,
                                                                   const api::ChatRequest& request,
                                                                   api::Exchange& exchange,
                                                                   ChatPrompt rendered,
                                                                   const ChatResume* resume) {
  server_.TouchActivity();
  Served* const m = &l;
  // Continuing: the prompt and everything generated so far (the state
  // holds most of it), and the generation's remaining tokens.
  std::vector<std::int32_t> continued;
  if (resume != nullptr) {
    continued = resume->history;
  }
  MemoryCharge token_charge;
  if (auto refused =
          FundHistory(token_charge, *memory_, continued.capacity() * sizeof(std::int32_t), false)) {
    return std::unexpected(*refused);
  }
  const std::vector<std::int32_t>& tokens = resume != nullptr ? continued : rendered.tokens();
  const auto prompt = static_cast<std::uint32_t>(rendered.tokens().size());
  const std::uint32_t done =
      resume != nullptr ? static_cast<std::uint32_t>(resume->generation.tokens.size()) : 0;
  const std::uint32_t max_tokens = rendered.max_tokens - done;
  // What making the model resident pages in: its weights and its state,
  // and the resident conversation's state written back (an upper bound;
  // the watchdog allows the swap for it, watchdog.h).
  std::uint64_t swap_bytes = 0;
  if (server_.resident() != m) {
    swap_bytes = m->weight_read_bytes() + l.state_snapshot_bytes();
    if (Served* out = server_.resident(); out != nullptr && out->llm()) {
      swap_bytes += static_cast<Llm&>(*out).state_snapshot_bytes();
    }
  }
  const Floors floors = l.floors();
  // Its usage: what so far it continues from.
  const auto so_far = [&](std::uint32_t cached) {
    return api::Completion{.completion_tokens = done,
                           .cached_tokens = resume != nullptr ? resume->reused : cached,
                           .stopped = false,
                           .literal = {},
                           .yielded = nullptr};
  };
  if (!exchange.Admit({.prompt_tokens = prompt,
                       .max_tokens = max_tokens,
                       .swap_bytes = swap_bytes,
                       .floors = floors})) {
    return so_far(0);
  }

  GenerateOptions options{.max_tokens = rendered.max_tokens,
                          .extra_stops = l.ChatStops(),
                          .stop = true,
                          .keep_logits = false,
                          .sampling = std::nullopt,
                          .seed = 0,
                          .on_tokens = {}};
  if (resume != nullptr) {
    options.sampling = resume->sampling;
    options.seed = resume->seed;
  } else if (auto sampling = SamplingOf(request, l)) {
    options.sampling = sampling;
    options.seed = request.seed.value_or(RandomSeed());
  }
  ChatOutput output(l, exchange, rendered.reasoning, rendered.tools, rendered.output_charge);
  if (rendered.tools) options.tool_sampling = rendered.tools->sampling;
  if (resume != nullptr) {
    output.Restore(resume->output);
  }
  Generation generation = resume != nullptr ? resume->generation : Generation{};
  if (resume != nullptr && !output.Drain()) {
    if (output.error()) return std::unexpected(*output.error());
    if (exchange.Yielding()) {
      auto kept =
          KeepContinuation(rendered, generation, options, output.Save(), resume->reused, *memory_);
      if (!kept) return std::unexpected(kept.error());
      return api::Completion{.completion_tokens = done,
                             .cached_tokens = resume->reused,
                             .stopped = generation.stopped,
                             .literal = {},
                             .yielded = std::move(*kept)};
    }
    return so_far(resume->reused);
  }
  std::optional<api::Error> token_problem;
  std::optional<api::Error> constraint_problem;
  options.on_tokens = [&](std::span<const std::int32_t> fresh) {
    server_.TouchActivity();
    const std::uint64_t held =
        (generation.tokens.capacity() + continued.capacity()) * sizeof(std::int32_t);
    token_problem = FundHistory(token_charge, *memory_, held, true);
    return !token_problem && output.Push(fresh);
  };
  std::uint32_t reused = 0;
  PrefillRun prefill;
  // Each chunk is a unit the watchdog allows for its rows (watchdog.h).
  const PrefillGoOn go_on = [&exchange](std::uint32_t rows) {
    return exchange.Next(Phase::kPrefill, rows);
  };
  const bool needs_model = max_tokens != 0 && !generation.stopped;
  if (needs_model) {
    if (server_.resident() != m && !exchange.Next(Phase::kSwap, swap_bytes)) {
      return so_far(0);
    }
    if (auto refused = ActivateFor(*m, m->name()); refused) {
      return std::unexpected(std::move(*refused));
    }
    // A swap is one program (about 10 s on a Spark), watched as one unit
    // at its bytes; whatever ended the request meanwhile (the client gone,
    // the backend stalled, the deadline, the runtime stopping) ends it
    // here, before any of its model work. The exchange answers for it.
    if (!exchange.Continue()) {
      return so_far(0);
    }
  }
  // Idle branches' retained state gives way to this request, the largest
  // first; a refusal it still meets fails the request alone.
  const ScopedReclaim reclaim(l, SerialReclaim(l));
  SerialCapacity capacity(l);
  const auto run = [&]() -> Status {
    // Between chunks, whatever ends the request (the client gone, the
    // backend stalled, the deadline, the runtime stopping) stops the
    // prefill: an ordinary end, the state holding the chunks that ran
    // (serving.h Llm::Prefill).
    if (max_tokens == 0 || generation.stopped) return {};
    std::vector<float> last;
    if (auto r = l.PreparePrompt(tokens, rendered.stable_boundary, last, reused, go_on, &prefill,
                                 false, resume != nullptr);
        !r) {
      return capacity(r, "the prompt could not be processed: ");
    }
    if (prefill.stopped || max_tokens == 0 || generation.stopped ||
        !exchange.Next(Phase::kDecode, 0)) {
      return {};  // the state holds what ran; the exchange answers for why
    }
    auto generated = l.Generate(last, options, generation, resume != nullptr);
    // Any refusal by the request's tool grammar (no allowed candidate, a token
    // or forced span outside it) is this request's alone, as in a cohort.
    if (!generated && options.tool_sampling && options.tool_sampling->refused() &&
        l.generation_cohort_usable()) {
      constraint_problem =
          Failure(500, "the generation was refused by its tool constraint: " + generated.error(),
                  "tool_output_invalid");
      return {};  // EndRequest still authenticates retirement before this is returned.
    }
    return capacity(std::move(generated), "the generation failed: ");
  };
  auto ran = needs_model ? server_.InRequest(*m, run) : Status{};
  if (!ran) {
    if (server_.node().TakeHangCancelled()) {
      return std::unexpected(RecoverFromHang(*m, std::format("{}'s request", m->name())));
    }
    Fail(std::format("{}'s request: {}", m->name(), ran.error()));
    return std::unexpected(Failure(500, "the generation failed; the runtime is stopping"));
  }
  if (server_.node().TakeHangCancelled()) {
    // A step ended after its request was cancelled for a hang: the reply
    // so far stands, and the model is reset for the next.
    (void)RecoverFromHang(*m, std::format("{}'s request", m->name()));
  }
  if (capacity.refusal()) {
    RefusedAlone(l);
    return std::unexpected(Failure(500, *capacity.refusal()));
  }
  if (token_problem) {
    return std::unexpected(*token_problem);
  }
  if (constraint_problem) return std::unexpected(*constraint_problem);
  if (prefill.stopped) {
    // Token counts and times only (D-014); the exchange answers for why.
    Say(log_, std::format("{}'s prefill stopped after {} chunks: the state holds {} of the "
                          "prompt's {} tokens",
                          m->name(), prefill.chunks, l.history().size(), tokens.size()));
    return so_far(reused);
  }
  // Its output finished once its generation is complete, also when it yields
  // its place (FinishOrCarry, as a cohort member's): a finish the yield
  // paused continues, as does a generation under way. There may be no
  // further GPU work, but accepted bytes must still reach the client exactly
  // once.
  const bool continues = !generation.stopped && generation.tokens.size() < rendered.max_tokens;
  const bool carry = FinishOrCarry(output, !continues, generation.stopped, exchange.Yielding());
  if (output.error()) return std::unexpected(*output.error());
  if (carry) {
    auto yielded = KeepContinuation(rendered, generation, options, output.Save(),
                                    resume != nullptr ? resume->reused : reused, *memory_);
    if (!yielded) return std::unexpected(yielded.error());
    return api::Completion{
        .completion_tokens = static_cast<std::uint32_t>(generation.tokens.size()),
        .cached_tokens = (*yielded)->reused,
        .stopped = generation.stopped,
        .literal = {},
        .yielded = std::move(*yielded)};
  }
  if (ladder_ != nullptr) {
    ladder_->Served(m->name());  // a later hang may reset it again
  }
  server_.TouchActivity();
  return api::Completion{.completion_tokens = static_cast<std::uint32_t>(generation.tokens.size()),
                         .cached_tokens = resume != nullptr ? resume->reused : reused,
                         .stopped = generation.stopped,
                         .literal = {},
                         .yielded = nullptr};
}

std::expected<api::Completion, api::Error> NodeBackend::Complete(
    const api::CompletionRequest& request, api::Exchange& exchange) {
  // A request's activity, as a chat turn's (shared mode's idle release).
  server_.TouchActivity();
  const auto& controls = request.options;
  Served* model = server_.Find(controls.model);
  if (model == nullptr || !model->llm()) {
    return std::unexpected(Failure(404, "The model does not exist", "model_not_found", "model"));
  }
  auto& llm = static_cast<Llm&>(*model);
  // A response's most: what the request memory holds (D-102); its score
  // rows are charged to it before any work (below).
  const auto response_bytes = static_cast<std::size_t>(memory_->capacity());
  auto prompt = api::PrepareLiteralPrompt(request, llm.tokenizer(), llm.usable_context(),
                                          response_bytes, memory_.get());
  if (!prompt) {
    return std::unexpected(prompt.error());
  }
  auto prompt_tokens = api::PromptTokens::Hold(std::move(prompt->tokens), *memory_);
  if (!prompt_tokens) {
    return std::unexpected(prompt_tokens.error());
  }
  const auto& tokens = (*prompt_tokens)->values();
  const auto max_tokens = controls.max_tokens.value_or(16);
  const bool score_prompt = request.prompt_logprobs || (request.echo && request.logprobs);
  api::Completion result;
  result.literal.prompt_text = std::move(prompt->text);
  std::size_t budget = 1024 + (6 * result.literal.prompt_text.size());
  if (budget > response_bytes) {
    return std::unexpected(Failure(413, "prompt echo exceeds the response size", {}, "prompt"));
  }
  // The rows it will hold, at most (PrepareLiteralPrompt figured them),
  // and its text: charged while it runs (the completed body is charged
  // in its place when it is made).
  MemoryCharge rows;
  if (const std::uint64_t held = prompt->score_bytes + budget; !rows.Add(*memory_, held)) {
    return std::unexpected(api::MemoryRefusal(*memory_, held, "the completion's scores"));
  }
  std::uint64_t swap_bytes = 0;
  if (server_.resident() != model) {
    swap_bytes = model->weight_read_bytes() + llm.state_snapshot_bytes();
    if (auto* out = server_.resident(); out != nullptr && out->llm()) {
      swap_bytes += static_cast<Llm&>(*out).state_snapshot_bytes();
    }
  }
  Floors floors = llm.floors();
  if (score_prompt) {
    // Pure scoring is one-row target work, not tiled prefill. Its scaled
    // deadline and per-row watchdog allowance must use the decode floor.
    floors.prefill = floors.decode;
  }
  if (!exchange.Admit({.prompt_tokens = static_cast<std::uint32_t>(tokens.size()),
                       .max_tokens = max_tokens,
                       .swap_bytes = swap_bytes,
                       .floors = floors})) {
    return api::Completion{};
  }
  if (server_.resident() != model && !exchange.Next(Phase::kSwap, swap_bytes)) {
    return api::Completion{};
  }
  if (auto refused = ActivateFor(*model, "the literal completion model"); refused) {
    return std::unexpected(std::move(*refused));
  }
  if (!exchange.Continue()) {
    return api::Completion{};
  }
  std::optional<api::Error> output_problem;
  api::LiteralRows prompt_rows(llm.tokenizer(), request.return_tokens_as_token_ids, true, 0, budget,
                               response_bytes);
  api::LiteralRows generated_rows(
      llm.tokenizer(), request.return_tokens_as_token_ids, false,
      request.echo ? api::TextCharacters(result.literal.prompt_text) : 0, budget, response_bytes);
  const std::uint32_t prompt_top = std::max(request.prompt_logprobs.value_or(0),
                                            request.echo ? request.logprobs.value_or(0) : 0);
  const auto add_prompt = [&](std::int32_t id, std::span<const float> row, bool first) {
    auto scored = prompt_rows.Add(id, row, prompt_top, first, first && prompt->added_bos);
    if (!scored) {
      output_problem = scored.error();
      return false;
    }
    if (request.echo && request.logprobs) {
      result.literal.logprobs.push_back(*scored);
    }
    if (request.prompt_logprobs) {
      result.literal.prompt_logprobs.push_back(std::move(*scored));
    }
    return exchange.Continue();
  };
  tokenizer::StreamDecoder decoder(llm.tokenizer(), {});
  GenerateOptions options{.max_tokens = max_tokens,
                          .stop = true,
                          .keep_logits = false,
                          .sampling = std::nullopt,
                          .seed = 0,
                          .on_tokens = {},
                          .on_logits = {}};
  if (auto sampling = SamplingOf(controls, llm)) {
    options.sampling = sampling;
    options.seed = controls.seed.value_or(RandomSeed());
  }
  if (request.logprobs) {
    options.on_logits = [&](std::int32_t id, std::span<const float> row) {
      auto scored = generated_rows.Add(id, row, *request.logprobs);
      if (!scored) {
        output_problem = scored.error();
        return false;
      }
      result.literal.logprobs.push_back(std::move(*scored));
      return exchange.Continue();
    };
  }
  Generation generation;
  MemoryCharge token_charge;
  options.on_tokens = [&](std::span<const std::int32_t> fresh) {
    server_.TouchActivity();
    if (auto refused = FundHistory(token_charge, *memory_,
                                   generation.tokens.capacity() * sizeof(std::int32_t), true)) {
      output_problem = std::move(refused);
      return false;
    }
    std::string piece;
    for (const auto id : fresh) {
      if (!decoder.Push(id, piece)) {
        output_problem = Failure(500, "a generated token cannot be decoded", "invalid_model_token");
        return false;
      }
    }
    if (budget > response_bytes || 6 * piece.size() > response_bytes - budget) {
      output_problem =
          Failure(413, "completion text exceeds the response size", "response_too_large");
      return false;
    }
    budget += 6 * piece.size();
    return piece.empty() ? exchange.Continue() : exchange.Content(piece);
  };
  PrefillRun prefill;
  // As chat's serial turn: idle branches' retained state gives way to this
  // request, and a refusal it still meets fails the request alone.
  const ScopedReclaim reclaim(llm, SerialReclaim(llm));
  SerialCapacity capacity(llm);
  auto ran = server_.InRequest(*model, [&]() -> Status {
    if (auto cleared = llm.Clear(); !cleared) {
      return cleared;
    }
    std::vector<float> last;
    const PrefillGoOn go_on = [&](std::uint32_t rows) {
      return exchange.Next(Phase::kPrefill, rows);
    };
    if (score_prompt) {
      if (!add_prompt(tokens.front(), {}, true)) {
        prefill.stopped = true;
        return {};
      }
      if (auto scored = llm.ScorePrompt(
              tokens, last,
              [&](std::int32_t id, std::span<const float> row) {
                return add_prompt(id, row, false);
              },
              go_on, &prefill);
          !scored) {
        return capacity(scored, "the prompt could not be processed: ");
      }
    } else if (auto filled = llm.Prefill(tokens, last, go_on, &prefill); !filled) {
      return capacity(filled, "the prompt could not be processed: ");
    }
    prompt_rows.Finish();  // same boundary flush as prompt_text, never carried into generation
    if (prefill.stopped || !exchange.Continue() || max_tokens == 0 ||
        !exchange.Next(Phase::kDecode, 0)) {
      return {};
    }
    return capacity(llm.Generate(last, options, generation), "the generation failed: ");
  });
  if (!ran) {
    if (server_.node().TakeHangCancelled()) {
      return std::unexpected(RecoverFromHang(*model, "a literal completion"));
    }
    Fail("the literal completion failed: " + ran.error());
    return std::unexpected(Failure(500, "the completion failed; the runtime is stopping"));
  }
  if (server_.node().TakeHangCancelled()) {
    (void)RecoverFromHang(*model, "a literal completion");
  }
  if (capacity.refusal()) {
    RefusedAlone(llm);
    return std::unexpected(Failure(500, *capacity.refusal()));
  }
  if (output_problem) {
    return std::unexpected(*output_problem);
  }
  std::string rest;
  decoder.Finish(rest);
  generated_rows.Finish();
  if (!rest.empty()) {
    if (budget > response_bytes || 6 * rest.size() > response_bytes - budget) {
      return std::unexpected(Failure(413, "completion text exceeds the response size",
                                     "response_unrepresentable", "logprobs"));
    }
    (void)exchange.Content(rest);
  }
  result.completion_tokens = static_cast<std::uint32_t>(generation.tokens.size());
  result.stopped = max_tokens == 0 || generation.stopped;
  if (ladder_ != nullptr) {
    ladder_->Served(model->name());  // a later hang may reset it again
  }
  server_.TouchActivity();
  return result;
}

void NodeBackend::CollectSwaps() {
  for (auto it = pending_swaps_.begin(); it != pending_swaps_.end();) {
    const auto finished = server_.PollSwap(*it);
    if (finished && !*finished) {
      ++it;
      continue;
    }
    if (!finished) {
      Fail("finishing the swap: " + finished.error());
    } else {
      const auto& parts = *it;
      if (parts.ready == Clock::time_point{}) {
        Say(log_, std::format("swap {} -> {} failed; its backing release settled {:.3f} s later "
                              "({} handoffs, {} unused releases)",
                              parts.from.empty() ? "(nothing)" : parts.from, parts.to,
                              parts.release, parts.handed_off, parts.released_unused));
      } else {
        Say(log_, std::format("swap {} -> {}: ready in {:.3f} s (evict {:.3f}, restore {:.3f}, "
                              "page-in {:.3f}, setup {:.3f}; {} graphs reclaimed for it); backing "
                              "released {:.3f} s "
                              "later",
                              parts.from.empty() ? "(nothing)" : parts.from, parts.to, parts.total,
                              parts.evict, parts.restore, parts.page_in, parts.setup,
                              parts.dropped_graphs, parts.release));
      }
    }
    it = pending_swaps_.erase(it);
  }
}

void NodeBackend::AfterResponse() { CollectSwaps(); }

void NodeBackend::Maintain() {
  CollectSwaps();
  if (failure_.empty()) {
    // The request memory grown to what the I/O and parse threads wanted
    // (or a denial counted), or given back toward what is used.
    memory_->Settle();
    server_.Maintain();
  }
}

bool NodeBackend::healthy() const { return failure_.empty(); }

std::string NodeBackend::failure() const { return failure_; }

Llm::CapacityReclaim NodeBackend::SerialReclaim(Llm& model) {
  return [this, &model](const Llm::Branch& refused) {
    const std::uint64_t needed = std::max<std::uint64_t>(model.RefusedBytes(refused), kExtentBytes);
    return server_.RelieveRefusedState(needed) >= needed;
  };
}

void NodeBackend::RefusedAlone(const Llm& model) {
  Say(log_, std::format("{}'s request does not fit the state's capacity alone at {} tokens: "
                        "refused",
                        model.name(), model.history().size()));
}

std::expected<std::uint64_t, std::string> NodeBackend::PromptStateToCome(const Llm& model,
                                                                         const Llm::Branch& branch,
                                                                         std::size_t tokens) {
  const auto positions =
      static_cast<std::uint32_t>(std::min<std::size_t>(tokens + 1, model.context()));
  auto total = model.StateBytesThrough(positions);
  if (!total) {
    return std::unexpected(total.error());
  }
  return StateToCome(*total, model.ResidentStateBytes(branch));
}

void NodeBackend::Fail(std::string what) {
  Say(log_, what);
  if (failure_.empty()) {
    failure_ = std::move(what);
  }
}

std::optional<api::Error> NodeBackend::ActivateFor(Served& m, std::string_view what) {
  if (server_.resident() == &m) {
    return std::nullopt;  // no swap: preserve the preceding activation's measurement
  }
  CollectSwaps();  // observation only: never wait for an earlier release tail
  if (!healthy()) return Failure(500, "the runtime is stopping after a swap release failure");
  (void)server_.node().TakeHangCancelled();  // asked after the work (D-102)
  SwapParts parts;
  auto activated = server_.Activate(m, parts);
  if (parts.requested != Clock::time_point{}) pending_swaps_.push_back(std::move(parts));
  if (activated) {
    return std::nullopt;
  }
  if (server_.node().TakeHangCancelled()) {
    Say(log_,
        "hang recovery: the swap's hung work was cancelled; the swap failed and only "
        "the requests that needed it fail");
    if (!AwaitDrained()) {  // a read still in the drive: rung 3
      return HungFailure();
    }
  }
  if (!server_.NodeHealthy()) {  // otherwise recovered (D-102): these requests only
    Fail(std::format("making {} resident: {}", what, activated.error()));
  }
  return Failure(503, "the model could not be made resident: " + activated.error());
}

api::Error NodeBackend::RecoverFromHang(Served& m, std::string_view what) {
  // Only once the cancellation drained: a read the drive still holds
  // would land in what the reset frees (a stuck read is rung 3's).
  if (!AwaitDrained()) {
    Fail(std::format("{}'s hung work did not drain after its cancellation", m.name()));
    return HungFailure();  // reached only when the last resort returns (a test's)
  }
  if (ladder_ != nullptr) {
    if (auto again = ladder_->Resetting(m.name()); again) {
      ladder_->Restart(*again);
      Fail(*again);  // reached only when the last resort returns (a test's)
      return HungFailure();
    }
  }
  Say(log_, std::format("hang recovery, rung 2 (reset the model in place): {} failed after its "
                        "hung work was cancelled and drained; {} is reset in place",
                        what, m.name()));
  if (auto recovered = server_.RecoverModel(m); !recovered) {
    const std::string why = std::format(
        "rung 2 could not reset model {} in place ({}): nothing less than a restart frees it",
        m.name(), recovered.error());
    if (ladder_ != nullptr) {
      ladder_->Restart(why);
    }
    Fail(why);  // reached only when the last resort returns (a test's): the service stops
  } else {
    Say(log_, std::format("hang recovery, rung 2: model {} was reset in place and evicted; its "
                          "next request loads it whole, and the service goes on",
                          m.name()));
  }
  return HungFailure();
}

bool NodeBackend::AwaitDrained() {
  if (ladder_ == nullptr) {
    return true;
  }
  while (ladder_->rung() == HangLadder::Rung::kCancel) {
    const auto now = Clock::now();
    ladder_->Activity(server_.progress(), now);
    (void)ladder_->Check(now);
    if (ladder_->rung() != HangLadder::Rung::kCancel) {
      break;
    }
    std::this_thread::sleep_for(kDrainPoll);
  }
  return ladder_->rung() != HangLadder::Rung::kRestart;
}

api::Error NodeBackend::HungFailure() {
  return Failure(503,
                 "the model's work hung and was cancelled, and the model was reset: retry the "
                 "request",
                 "backend_hung");
}

std::expected<api::ReleaseReport, std::string> NodeBackend::ReleaseWorkingSet(
    std::string_view reason) {
  for (const ChatWork* w : cohort_) {
    if (w != nullptr) {
      return std::unexpected("busy");
    }
  }
  return server_.ReleaseWorkingSet(reason);
}

}  // namespace llmp::runtime::service_detail
