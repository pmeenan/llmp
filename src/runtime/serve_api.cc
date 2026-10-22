// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <expected>
#include <filesystem>
#include <format>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include "base/check.h"
#include "base/report.h"
#include "base/sha256.h"
#include "base/work_pulse.h"
#include "chat/chat.h"
#include "config/node_config.h"
#include "platform/event_loop.h"
#include "platform/host_probe.h"
#include "platform/interfaces.h"
#include "platform/job.h"
#include "platform/sd_notify.h"
#include "platform/sockets.h"
#include "runtime/api.h"
#include "runtime/api_server.h"
#include "runtime/binding.h"
#include "runtime/cohort_capacity.h"
#include "runtime/commands.h"
#include "runtime/completion_tokens.h"
#include "runtime/hang_ladder.h"
#include "runtime/intake_limits.h"
#include "runtime/prefill.h"
#include "runtime/runtime.h"
#include "runtime/service_backend.h"
#include "runtime/serving.h"
#include "runtime/watchdog.h"
#include "scheduler/cohort_schedule.h"
#include "tokenizer/tokenizer.h"

namespace llmp::runtime {
namespace {
using service_detail::Say;
// D-105 at the process's end. A hang's last resort writes the conversation
// records already queued for at most this long (best effort, bounded:
// recovery comes first). A graceful stop keeps conversations for at most
// kPersistLongest, giving up sooner only when nothing moves (Server::
// Persist), and asks the service manager each second for kStopExtension
// more.
constexpr auto kLastResortKeep = std::chrono::seconds(10);
// Past kLastResortKeep, the last resort exits however far it got.
constexpr auto kLastResortExitSlack = std::chrono::seconds(2);
constexpr auto kPersistLongest = std::chrono::hours(1);
constexpr std::chrono::microseconds kStopExtension = std::chrono::seconds(15);
// A start step's extension of the start timeout: llmp.service's
// TimeoutStartSec.
constexpr std::chrono::microseconds kStartExtension = std::chrono::minutes(5);
// The node's waits under the hang ladder (D-102): each wait counts as work
// under way while it lasts, feeds the ladder the node's progress as it
// polls, and cancels its request at rung 1 when it began before the hang
// was confirmed (a later wait is new work). Rung 3 is the ladder's last
// resort, run by whichever thread entered it: the wait itself never gives
// up.
class LadderPatience final : public engine::Patience {
 public:
  explicit LadderPatience(HangLadder& ladder) : ladder_(ladder) {}
  void Begin() override { ladder_.BeginWait(); }
  void End() override { ladder_.EndWait(); }
  engine::WaitVerdict Check(const engine::WaitState& wait, std::uint64_t progress) override {
    const auto now = Clock::now();
    ladder_.Activity(progress, now);
    (void)ladder_.Check(now);
    if (ladder_.rung() == HangLadder::Rung::kCancel && !wait.cancelled &&
        wait.began <= ladder_.cancelled_at()) {
      return engine::WaitVerdict::kCancel;
    }
    return engine::WaitVerdict::kWait;
  }

 private:
  HangLadder& ladder_;
};

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
using service_detail::NodeBackend;
using service_detail::Say;
int RunService(const config::NodeConfig& config, const config::RuntimeRoles& roles,
               ServingOptions serving, std::FILE* log) {
  // Speculative where there is a drafter; no image prompt; conversations
  // kept across a restart (D-105).
  serving.keep_conversations = true;
  if (auto configured = ApplyServiceDiagnosticEnvironment(serving); !configured) {
    Say(log, configured.error());
    return kExitFailure;
  }
  int status = kExitOk;
  // Hang recovery (D-102; hang_ladder.h): one ladder for the chat route's
  // watch and the node's waits, outliving both (the teardown's waits are
  // watched too). Its last resort exits for the supervisor; nothing is torn
  // down (the driver may be the thread that hangs).
  const config::ClientConfig& client_limits = config.client;
  const std::chrono::milliseconds hang =
      client_limits.hang_seconds
          ? std::chrono::milliseconds(std::chrono::seconds(*client_limits.hang_seconds))
          : api::DefaultHang(std::chrono::seconds(client_limits.stall_seconds));
  HangLadder ladder(hang, Clock::now());
  ladder.set_log([log](std::string_view line) { Say(log, std::string(line)); });
  // What can be kept is kept first, briefly (D-105): the conversation
  // records already queued are written (their spill files are whole on
  // disk); nothing new is spilled, since the device may be what hangs.
  std::atomic<Server*> running{nullptr};
  ladder.set_last_resort([log, &running](const std::string& why) {
    // The exit at the deadline, whatever blocks below (a drive that holds
    // a record's write, a log that cannot be written): the keeping is
    // bounded, never a wait on the hang itself.
    std::thread([] {
      std::this_thread::sleep_for(kLastResortKeep + kLastResortExitSlack);
      std::_Exit(kExitFailure);
    }).detach();
    Say(log, why);
    std::ignore = platform::NotifyServiceManager("STATUS=exiting after a confirmed hang");
    if (Server* server = running.load(); server != nullptr) {
      const bool kept = server->DrainKept(Clock::now() + kLastResortKeep);
      Say(log, kept ? "the conversation records queued were written"
                    : "some conversation records queued were not written in time");
    }
    Say(log, "exiting for the supervisor to restart the runtime (hang recovery's last resort)");
    std::_Exit(kExitFailure);
  });
  LadderPatience patience(ladder);
  // This thread is the driver: its long CPU work beats the ladder's pulse
  // and stops when rung 1 asks it to (base/work_pulse.h).
  base::SetThreadPulse(&ladder.pulse());
  // M3.6's speed gate's diagnostic (ServingOptions::diagnostic_skip_turn_
  // checkpoints; temporary, never in configuration).
  // NOLINTNEXTLINE(concurrency-mt-unsafe): read before any thread starts
  if (const char* skip = std::getenv("LLMP_DIAGNOSTIC_SKIP_TURN_CHECKPOINTS");
      skip != nullptr && std::string_view(skip) == "1") {
    serving.diagnostic_skip_turn_checkpoints = true;
  }
  {
    Server server(config, roles, serving, log);
    server.SetPatience(&patience);
    // Each step of the start that made progress extends the service
    // manager's start timeout (llmp.service's TimeoutStartSec).
    server.set_start_progress([] {
      std::ignore = platform::NotifyServiceManager(
          std::format("EXTEND_TIMEOUT_USEC={}", kStartExtension.count()));
    });
    running.store(&server);
    auto started = server.Start(false);
    // A cancellation drains only once nothing submitted before it is still
    // in flight (a read a drive holds does not end when cancelled).
    ladder.set_operations([&server] { return server.node().oldest_io(); });
    std::optional<NodeBackend> backend;
    std::optional<api::Server> http;
    if (started) {
      // The limits that follow memory (D-102), or what [client] sets: the
      // request memory the start reserved beside the guard's margin, a
      // body's most within it and the largest configured context's bytes.
      const IntakeLimits intake =
          DeriveIntakeLimits(server.request_memory(), server.request_capacity(),
                             server.context_body_bytes(), config.client);
      auto memory = std::make_shared<RequestMemory>(intake.request_floor, intake.request_capacity);
      // This thread is the driver: its charges grow the request memory at
      // once, through the reclaim order; the I/O and parse threads' wait
      // for Maintain to grow it (RequestMemory::Settle).
      memory->SetDriver(std::this_thread::get_id(),
                        [&server](std::uint64_t to) { return server.SetRequestMemory(to); });
      backend.emplace(server, config, memory, log, &ladder);
      api::ServerOptions options;
      options.intake = intake;
      options.memory = memory;
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
      const config::ClientConfig& client = config.client;
      const auto seconds = [](std::optional<std::uint32_t> s) {
        return s ? std::optional<std::chrono::milliseconds>(std::chrono::seconds(*s))
                 : std::nullopt;
      };
      options.max_queued = client.max_queued;
      options.queue_wait = seconds(client.queue_wait_seconds);
      options.stall = std::chrono::seconds(client.stall_seconds);
      options.stall_fails = client.stall_action == config::StallAction::kFail;
      options.deadline_cap = seconds(client.deadline_cap_seconds);
      options.idle_timeout = std::chrono::seconds(client.idle_seconds);
      options.request_inactivity = std::chrono::seconds(client.request_inactivity_seconds);
      options.write_inactivity = seconds(client.write_inactivity_seconds);
      options.model_turn = std::chrono::seconds(client.model_turn_seconds);
      // A genuine hang is recovered (D-102): the I/O thread feeds the ladder
      // the node's progress (any lane's completion, any wait of the
      // driver's ending) and escalates it; the node's waits cancel at its
      // rung 1, and its last resort restarts the process.
      options.ladder = &ladder;
      options.activity = [&server] { return server.progress(); };
      // The server logs each change of the backend's health; the service
      // manager's status line says it too (`systemctl status`).
      const std::size_t served = backend->Models().size();
      const bool fails = options.stall_fails;
      options.on_health = [log, served, fails](const Health& health) {
        std::string status = std::format("STATUS=serving {} models on the chat route", served);
        if (!health.healthy) {
          status = std::format("STATUS=the model backend is not making progress ({}); {}",
                               PhaseName(health.phase),
                               fails ? "refusing requests until it does" : "requests wait for it");
        }
        if (auto notified = platform::NotifyServiceManager(status); !notified) {
          Say(log, notified.error());
        }
      };
      // Each connection is a descriptor; the rest (model files, the
      // node's own) fit in kReservedFiles. Without max_connections the
      // route keeps what the open-file hard limit allows (D-102).
      const std::uint64_t want = client.max_connections
                                     ? std::uint64_t{*client.max_connections} + kReservedFiles
                                     : std::numeric_limits<std::uint64_t>::max();
      const std::uint64_t files = platform::RaiseOpenFileLimit(want);
      const std::uint64_t fits = ConnectionsFor(files);
      if (client.max_connections && fits < *client.max_connections) {
        Say(log, std::format("the open-file limit is {}: the chat route keeps at most {} "
                             "connections, not the {} configured",
                             files, fits, *client.max_connections));
      }
      options.max_connections = static_cast<std::size_t>(
          client.max_connections ? std::min<std::uint64_t>(*client.max_connections, fits) : fits);
      Say(log, std::format("the chat route's limits: request memory {} bytes set apart, growing "
                           "within the budget to {} bytes{} (bodies, parses, renderings, unread "
                           "output, responses); bodies up to {} bytes; a stream pauses past {} "
                           "unread bytes; {} connections; queue {}; stalls reported after {} s{}; "
                           "a hang confirmed after {} s without progress is recovered (cancel, "
                           "reset the model, restart)",
                           intake.request_floor, intake.request_capacity,
                           client.request_memory_bytes ? " ([client] request_memory_bytes)" : "",
                           intake.max_body, intake.stream_buffer, options.max_connections,
                           client.max_queued ? std::format("at most {}", *client.max_queued)
                                             : std::string("unbounded"),
                           client.stall_seconds, fails ? " (and fail requests)" : "",
                           std::chrono::duration_cast<std::chrono::seconds>(hang).count()));
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
          // A graceful stop keeps the conversations for the next start
          // (D-105): spilled and recorded, telling the service manager it is
          // still working (each extension a few seconds past the last), so
          // its stop timeout never cuts a stop that makes progress.
          const auto extend = [] {
            std::ignore = platform::NotifyServiceManager(
                std::format("EXTEND_TIMEOUT_USEC={}", kStopExtension.count()));
          };
          extend();
          server.Persist(Clock::now() + kPersistLongest, extend);
          extend();  // and the teardown that follows
        }
      }
    }
    http.reset();
    for (const auto& model : server.models())
      if (model->llm())
        Say(log, std::format("serving final {}: {}", model->name(), model->extra()));
    if (auto stopped = server.TearDown(); !stopped) {
      Say(log, "stopping: " + stopped.error());
      status = kExitFailure;
    }
    ladder.set_operations({});  // the server goes
    running.store(nullptr);
  }
  base::SetThreadPulse(nullptr);  // the ladder goes
  return status;
}

}  // namespace llmp::runtime
