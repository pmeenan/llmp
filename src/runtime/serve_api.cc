// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// The service with models configured (D-097; docs/runtime-serving.md#the-chat-route):
// the configured models registered on the node (serving.h), the chat
// route (api_server.h) over them, where [client] bind says (binding.h), and
// the runtime's signals watched as readiness (platform/event_loop.h) on the
// node's driver thread, which runs every request.

#include <algorithm>
#include <array>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <expected>
#include <format>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include "base/report.h"
#include "chat/chat.h"
#include "config/node_config.h"
#include "platform/event_loop.h"
#include "platform/interfaces.h"
#include "platform/job.h"
#include "platform/sd_notify.h"
#include "platform/sockets.h"
#include "runtime/api.h"
#include "runtime/api_server.h"
#include "runtime/binding.h"
#include "runtime/commands.h"
#include "runtime/prefill.h"
#include "runtime/runtime.h"
#include "runtime/serving.h"
#include "runtime/watchdog.h"
#include "tokenizer/tokenizer.h"

namespace jitllm::runtime {
namespace {

void Say(std::FILE* log, std::string_view text) {
  const std::string line = std::format("jitllm-runtime: {}\n", base::Printable(text));
  (void)std::fwrite(line.data(), 1, line.size(), log);
  (void)std::fflush(log);
}

api::Error Failure(int status, std::string message, std::string code = {}, std::string param = {}) {
  return api::Error{.status = status,
                    .type = status >= 500 ? "server_error" : "invalid_request_error",
                    .message = std::move(message),
                    .param = std::move(param),
                    .code = std::move(code)};
}

std::uint64_t RandomSeed() {
  std::uint64_t seed = 0;
  if (!platform::FillRandom(std::as_writable_bytes(std::span(&seed, 1)))) {
    seed = static_cast<std::uint64_t>(Clock::now().time_since_epoch().count());
  }
  return seed;
}

// The chat route's requests as turns on the node: one model resident, a
// swap when another is asked for, the conversation state reused when the
// rendered request extends it, and each request one lease (D-093).
class NodeBackend final : public api::Backend {
 public:
  NodeBackend(Server& server, const config::NodeConfig& config, std::FILE* log)
      : server_(server), config_(config), log_(log) {}

  std::vector<api::ModelInfo> Models() const override {
    std::vector<api::ModelInfo> models;
    for (const config::ModelEntry& entry : config_.models) {
      if (entry.composition) {
        models.push_back({.name = entry.name, .chat = false, .context = 0});
        continue;
      }
      Served* m = server_.Find(entry.name);
      if (m != nullptr && m->llm()) {
        models.push_back(
            {.name = entry.name, .chat = true, .context = static_cast<Llm&>(*m).usable_context()});
      }
    }
    return models;
  }

  std::expected<api::Completion, api::Error> Complete(const api::ChatRequest& request,
                                                      api::Exchange& exchange) override {
    Served* m = server_.Find(request.model);
    if (m == nullptr || !m->llm()) {
      return std::unexpected(Failure(404, "The model does not exist", "model_not_found", "model"));
    }
    auto& l = static_cast<Llm&>(*m);
    chat::Conversation conversation;
    for (const api::Message& message : request.messages) {
      chat::Role role = chat::Role::kUser;
      if (message.role == api::Role::kSystem) {
        role = chat::Role::kSystem;
      } else if (message.role == api::Role::kAssistant) {
        role = chat::Role::kAssistant;
      }
      conversation.messages.push_back({.role = role,
                                       .content = message.content,
                                       .reasoning_content = message.reasoning,
                                       .tool_calls = {}});
    }
    l.Defaults(conversation);
    std::uint32_t stable_boundary = 0;
    auto rendered = l.RenderChat(conversation, &stable_boundary);
    if (!rendered) {
      return std::unexpected(
          Failure(400, "the conversation cannot be rendered: " + rendered.error(), {}, "messages"));
    }
    const std::vector<std::int32_t>& tokens = *rendered;
    const std::uint32_t usable = l.usable_context();
    const auto prompt = static_cast<std::uint32_t>(tokens.size());
    if (prompt >= usable) {
      return std::unexpected(
          Failure(400,
                  std::format("This model's maximum context length is {} tokens. However, your "
                              "messages resulted in {} tokens. Please reduce the length of the "
                              "messages.",
                              usable, prompt),
                  "context_length_exceeded", "messages"));
    }
    const std::uint32_t max_tokens = request.max_tokens.value_or(usable - prompt);
    if (std::uint64_t{prompt} + max_tokens > usable) {
      return std::unexpected(Failure(
          400,
          std::format("This model's maximum context length is {} tokens. However, you requested "
                      "{} tokens ({} in the messages, {} in the completion). Please reduce the "
                      "length of the messages or completion.",
                      usable, std::uint64_t{prompt} + max_tokens, prompt, max_tokens),
          "context_length_exceeded", "messages"));
    }
    // Whether the rendered prompt leaves the model inside a reasoning
    // block: its last reasoning marker opens one.
    bool reasoning = false;
    if (l.think_start() && l.think_end()) {
      const auto marker = std::ranges::find_if(tokens.rbegin(), tokens.rend(), [&](std::int32_t t) {
        return t == *l.think_start() || t == *l.think_end();
      });
      reasoning = marker != tokens.rend() && *marker == *l.think_start();
    }
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
    Floors floors;
    if (const auto entry =
            std::ranges::find(config_.models, request.model, &config::ModelEntry::name);
        entry != config_.models.end()) {
      floors = {.prefill = entry->prefill_floor_tok_s, .decode = entry->decode_floor_tok_s};
    }
    if (!exchange.Admit({.prompt_tokens = prompt,
                         .max_tokens = max_tokens,
                         .swap_bytes = swap_bytes,
                         .floors = floors})) {
      return api::Completion{};
    }

    swapped_ = server_.resident() != m;
    if (swapped_ && !exchange.Next(Phase::kSwap, swap_bytes)) {
      swapped_ = false;
      return api::Completion{};
    }
    if (auto r = server_.Activate(*m, parts_); !r) {
      Fail(std::format("making {} resident: {}", m->name(), r.error()));
      swapped_ = false;
      return std::unexpected(Failure(503, "the model could not be made resident"));
    }
    // A swap is one program (about 10 s on a Spark), watched as one unit
    // at its bytes; whatever ended the request meanwhile (the client gone,
    // the backend stalled, the deadline, the runtime stopping) ends it
    // here, before any of its model work. The exchange answers for it.
    if (!exchange.Continue()) {
      return api::Completion{};
    }

    GenerateOptions options{.max_tokens = max_tokens,
                            .stop = true,
                            .keep_logits = false,
                            .sampling = std::nullopt,
                            .seed = 0,
                            .on_tokens = {}};
    if (request.temperature > 0) {
      options.sampling =
          execution::SamplingParams{.temperature = static_cast<float>(request.temperature),
                                    .top_k = request.top_k,
                                    .top_p = static_cast<float>(request.top_p),
                                    .min_p = static_cast<float>(request.min_p)};
      options.seed = request.seed.value_or(RandomSeed());
    }
    tokenizer::StreamDecoder decoder(l.tokenizer(), {});
    bool thinking = reasoning;
    bool any_content = false;
    options.on_tokens = [&](std::span<const std::int32_t> fresh) {
      std::string piece;
      bool go = true;
      bool said = false;
      const auto send = [&]() {
        if (!piece.empty()) {
          go = (thinking ? exchange.Reasoning(piece) : exchange.Content(piece)) && go;
          any_content = any_content || !thinking;
          said = true;
          piece.clear();
        }
      };
      for (const std::int32_t token : fresh) {
        if (thinking && l.think_end() && token == *l.think_end()) {
          decoder.Finish(piece);
          if (piece.empty()) {
            go = exchange.Reasoning({}) && go;  // an empty block: the answer is still trimmed
          }
          send();
          thinking = false;
          continue;
        }
        if (!thinking && !any_content && piece.empty() && l.think_start() &&
            token == *l.think_start()) {
          thinking = true;  // the model opened reasoning itself
          continue;
        }
        std::ignore = decoder.Push(token, piece);  // a token it cannot decode adds nothing
      }
      send();
      return said ? go : exchange.Continue();
    };

    Generation generation;
    std::uint32_t reused = 0;
    PrefillRun prefill;
    // Each chunk is a unit the watchdog allows for its rows (watchdog.h).
    const PrefillGoOn go_on = [&exchange](std::uint32_t rows) {
      return exchange.Next(Phase::kPrefill, rows);
    };
    auto ran = server_.InRequest(*m, [&]() -> Status {
      // Between chunks, whatever ends the request (the client gone, the
      // backend stalled, the deadline, the runtime stopping) stops the
      // prefill: an ordinary end, the state holding the chunks that ran
      // (serving.h Llm::Prefill).
      std::vector<float> last;
      if (auto r = l.PreparePrompt(tokens, stable_boundary, last, reused, go_on, &prefill); !r) {
        return r;
      }
      if (prefill.stopped || !exchange.Next(Phase::kDecode, 0)) {
        return {};  // the state holds what ran; the exchange answers for why
      }
      return l.Generate(last, options, generation);
    });
    if (!ran) {
      Fail(std::format("{}'s request: {}", m->name(), ran.error()));
      return std::unexpected(Failure(500, "the generation failed; the runtime is stopping"));
    }
    if (prefill.stopped) {
      // Token counts and times only (D-014); the exchange answers for why.
      Say(log_, std::format("{}'s prefill stopped after {} chunks: the state holds {} of the "
                            "prompt's {} tokens",
                            m->name(), prefill.chunks, l.history().size(), tokens.size()));
      return api::Completion{.completion_tokens = 0, .cached_tokens = reused, .stopped = false};
    }
    std::string rest;
    decoder.Finish(rest);
    if (!rest.empty()) {
      (void)(thinking ? exchange.Reasoning(rest) : exchange.Content(rest));
    }
    return api::Completion{
        .completion_tokens = static_cast<std::uint32_t>(generation.tokens.size()),
        .cached_tokens = reused,
        .stopped = generation.stopped};
  }

  void AfterResponse() override {
    if (!swapped_) {
      return;
    }
    swapped_ = false;
    if (auto r = server_.FinishSwap(parts_); !r) {
      Fail("finishing the swap: " + r.error());
      return;
    }
    Say(log_,
        std::format("swap {} -> {}: ready in {:.3f} s (evict {:.3f}, restore {:.3f}, "
                    "page-in {:.3f}, setup {:.3f}); backing released {:.3f} s later",
                    parts_.from.empty() ? "(nothing)" : parts_.from, parts_.to, parts_.total,
                    parts_.evict, parts_.restore, parts_.page_in, parts_.setup, parts_.release));
  }

  bool healthy() const override { return failure_.empty(); }
  std::string failure() const override { return failure_; }

 private:
  void Fail(std::string what) {
    Say(log_, what);
    if (failure_.empty()) {
      failure_ = std::move(what);
    }
  }

  Server& server_;
  const config::NodeConfig& config_;
  std::FILE* log_;
  SwapParts parts_;
  bool swapped_ = false;
  std::string failure_;  // a node failure: the service stops
};

// Descriptors the node needs besides the chat route's connections.
constexpr std::uint64_t kReservedFiles = 256;

// The reverse lookups' time at startup, all of them together: MagicDNS
// answers in milliseconds, and a resolver that does not answer costs no
// more than this (its names are then not accepted; the start log says
// which are).
constexpr auto kLookupBudget = std::chrono::seconds(5);

// `[client] bind` against the node's addresses (binding.h), its notes
// logged (its unauthenticated listeners are, once it listens).
std::expected<api::Listening, std::string> ResolveBind(const config::ClientConfig& client,
                                                       std::FILE* log) {
  auto addresses = platform::ReadInterfaceAddresses();
  if (!addresses) {
    return std::unexpected("the chat route cannot read the node's addresses: " + addresses.error());
  }
  const auto by = std::chrono::steady_clock::now() + kLookupBudget;
  const api::ReverseLookup reverse =
      [by](const platform::InterfaceAddress& a) -> std::optional<std::string> {
    const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(
        by - std::chrono::steady_clock::now());
    if (left.count() <= 0) {
      return std::nullopt;
    }
    return platform::ReverseName(a, left);
  };
  api::Listening listening =
      api::ResolveListening(client, *addresses, platform::HostName(), reverse);
  for (const std::string& note : listening.notes) {
    Say(log, note);
  }
  if (listening.endpoints.empty()) {
    return std::unexpected("[client] bind resolves to no address to listen on");
  }
  return listening;
}

std::string HostNames(const api::HostGuard& hosts) {
  std::string names;
  for (const std::string& name : hosts.names()) {
    names += (names.empty() ? "" : ", ") + name;
  }
  return names.empty() ? "(none)" : names;
}

}  // namespace

int RunService(const config::NodeConfig& config, const config::RuntimeRoles& roles,
               std::FILE* log) {
  const ServingOptions serving;  // speculative where there is a drafter; no image prompt
  int status = kExitOk;
  {
    Server server(config, roles, serving, log);
    auto started = server.Start(false);
    std::optional<NodeBackend> backend;
    std::optional<api::Server> http;
    if (started) {
      backend.emplace(server, config, log);
      api::ServerOptions options;
      std::vector<std::string> unauthenticated;
      if (auto listening = ResolveBind(config.client, log); !listening) {
        started = std::unexpected(listening.error());
      } else {
        options.bind = std::move(listening->endpoints);
        options.hosts = std::move(listening->hosts);
        unauthenticated = std::move(listening->unauthenticated);
      }
      const std::vector<config::ClientEndpoint> endpoints = options.bind;
      const std::string names = HostNames(options.hosts);
      options.max_queued = config.client.max_queued;
      options.max_connections = config.client.max_connections;
      options.stall = std::chrono::seconds(config.client.stall_seconds);
      options.deadline_cap = std::chrono::seconds(config.client.deadline_cap_seconds);
      // The server logs each change of the backend's health; the service
      // manager's status line says it too (`systemctl status`).
      const std::size_t served = backend->Models().size();
      options.on_health = [log, served](const Health& health) {
        const std::string status =
            health.healthy ? std::format("STATUS=serving {} models on the chat route", served)
                           : std::format(
                                 "STATUS=the model backend is not making progress ({}); refusing "
                                 "requests until it does",
                                 PhaseName(health.phase));
        if (auto notified = platform::NotifyServiceManager(status); !notified) {
          Say(log, notified.error());
        }
      };
      // Each connection is a descriptor; the rest (model files, the
      // node's own) fit in kReservedFiles.
      const std::uint64_t files =
          platform::RaiseOpenFileLimit(std::uint64_t{options.max_connections} + kReservedFiles);
      if (files < std::uint64_t{options.max_connections} + kReservedFiles) {
        const std::uint64_t fits = files > kReservedFiles + 16 ? files - kReservedFiles : 16;
        Say(log, std::format("the open-file limit is {}: the chat route keeps at most {} "
                             "connections, not the {} configured",
                             files, fits, options.max_connections));
        options.max_connections = static_cast<std::size_t>(fits);
      }
      options.log = log;
      if (started) {
        http.emplace(*backend, std::move(options));
        if (auto ports = http->Listen(); !ports) {
          started = std::unexpected(std::format("the chat route cannot listen: {}", ports.error()));
        } else {
          std::string where;
          for (std::size_t i = 0; i < endpoints.size() && i < ports->size(); ++i) {
            const config::ClientEndpoint& e = endpoints[i];
            where += std::format("{}{}{}{}:{}", where.empty() ? "" : ", ", e.ipv6 ? "[" : "",
                                 e.address, e.ipv6 ? "]" : "", (*ports)[i]);
          }
          Say(log, std::format("the chat route listens on {} (/v1/chat/completions, /v1/models); "
                               "Host names accepted besides loopback: {}",
                               where, names));
          for (const std::string& line : unauthenticated) {
            Say(log, line);
          }
        }
      }
    }
    if (!started) {
      Say(log, "refusing to serve: " + started.error());
      status = kExitFailure;
    } else if (!backend.has_value() || !http.has_value()) {
      Say(log, "refusing to serve: the chat route was not set up");  // not reached
      status = kExitFailure;
    } else {
      constexpr std::array kWatched = {SIGTERM, SIGINT, SIGHUP, SIGCHLD};
      const platform::SignalWatch signals = platform::SignalWatch::Open(kWatched);
      if (!signals.valid()) {
        Say(log, "cannot watch signals");
        status = kExitFailure;
      } else {
        if (auto notified = platform::NotifyServiceManager(std::format(
                "READY=1\nSTATUS=serving {} models on the chat route", backend->Models().size()));
            !notified) {
          Say(log, notified.error());
        }
        Say(log, "ready");
        const auto on_wake = [&]() {
          bool stop = false;
          for (int signo = signals.Take(); signo != 0; signo = signals.Take()) {
            if (signo == SIGCHLD) {
              (void)platform::ReapExited();
            } else if (signo == SIGHUP) {
              Say(log,
                  "the configuration is read only at startup; restart the runtime to apply a "
                  "change");
            } else {
              stop = true;
            }
          }
          return stop;
        };
        auto ran = http->Run(signals.descriptor(), on_wake);
        if (auto notified = platform::NotifyServiceManager("STOPPING=1"); !notified) {
          Say(log, notified.error());
        }
        if (!ran) {
          Say(log, "stopping after a failure: " + ran.error());
          status = kExitFailure;
        } else {
          Say(log, "stopping");
        }
      }
    }
    http.reset();
    if (auto stopped = server.TearDown(); !stopped) {
      Say(log, "stopping: " + stopped.error());
      status = kExitFailure;
    }
  }
  return status;
}

}  // namespace jitllm::runtime
