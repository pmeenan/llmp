// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// M3's chat route as a server (D-097 as the owner amended it on 2026-09-28;
// docs/runtime-serving.md#the-chat-route): `GET /v1/models`,
// `GET /v1/models/{id}`, `POST /v1/chat/completions`, non-streaming
// `POST /v1/completions` (D-100) and, for loopback
// peers, `GET /jitllm/v1/ignored-fields`, on one listener per resolved
// endpoint (binding.h), over a Backend that runs the requests.
// Vendor-free, so the CPU tests drive it with a fake backend.
//
// Threads. An I/O thread runs an event loop (platform/event_loop.h, epoll
// on Linux) over the listeners and every connection, all non-blocking: it
// reads requests under http.h's bounds, answers the model list and every
// refusal itself, and queues valid chat requests, never waiting on the
// model. Connections persist (HTTP/1.1
// keep-alive, closed after kIdleTimeoutMs idle), up to max_connections
// open (idle ones closed oldest first to admit a new one); a request that
// arrives on a connection before the previous response has ended
// (pipelining) is not read: that connection closes after the response.
// The thread that calls Run is the backend's (the node's one driver): it
// takes queued requests in arrival order and runs each to its end, one at
// a time, watching the caller's wake descriptor (the runtime's signals)
// between requests and between generation steps. At most max_queued wait
// behind the running one, beyond which a 429 with Retry-After; a
// non-streaming request waits at most queue_wait (then the same 429), a
// stream, held by keepalives, as long as the backend makes progress.
//
// Progress (watchdog.h). Every call the backend makes on its Exchange is
// a beat; Next names the unit that follows (a swap, a prefill chunk,
// decode) and its size, which with the model's floors sets how long it
// may take: `stall` plus kWorkMargin times its expected time. The I/O
// thread watches the beats: past that allowance the running request ends
// (a 504 before the headers, an in-stream error after), its generation is
// cancelled as when a client leaves, what is queued gets a 503, and the
// backend is unhealthy (health(), the log, `on_health`): new requests get
// a 503 until the backend's next beat, which the unit it hung in makes
// when it returns. A hung unit that never returns is the node's to end
// (its per-step patience, engine/paged_node.h). A request has no fixed
// deadline: a stream runs until it is done, its client leaves or the
// watchdog fires; a non-streaming one also ends at its scaled deadline
// (watchdog.h ScaledDeadline, capped at `deadline_cap`).
//
// Output. The driver never touches a socket: a request's response is a
// Channel it appends to (whole events, under the server's lock) and the
// I/O thread writes it out as the socket takes it. A client that takes
// nothing for write_timeout, or lets max_unsent bytes of a stream pile up,
// is dropped, and its generation ends at the next step. A connection that
// closes ends its request's generation at the next step; the backend
// releases the request's lease as it returns (completion-aware: the
// channel outlives the connection until the driver lets go of it). A
// client that shuts only its sending side after a whole request gets its
// response, then the connection closes; that looks like a close until
// something is sent, so the server sends at once a probe that a closed
// peer answers with a reset: a stream's start (headers and role chunk) or
// a `: keepalive` comment; before a non-streaming response's head, on
// HTTP/1.1, an interim 102 (HTTP/1.0 runs to its end). A connection
// between requests keeps no large buffer (held_bytes).
// Completed literal bodies share response_budget by their allocated capacity;
// their reservation follows the bytes through the channel and socket buffer
// until that allocation is released. A full budget refuses another body (503).
//
// A request's end. Non-streaming: one JSON body once the outcome is known
// (a 504 past its deadline or on a stall, a 503 when the runtime stops),
// nothing before but that 102. Streaming: the headers and the role chunk
// once the backend admits it, or earlier, once it has waited `keepalive`
// in the queue or its client has half-closed; then a chunk per step's
// text, the finish chunk, the usage chunk if asked for, and
// `[DONE]`, with `: keepalive` comment lines whenever nothing was sent for
// `keepalive` (queued, swapping, prefilling). A failure after the headers
// is an `error` event and the stream ends without `[DONE]`. A stream is
// chunked on HTTP/1.1, so its connection persists. Nothing a request holds
// is logged: each logs an opaque ID, the model, the status and token
// counts; an unknown field's name is logged once, when first seen.

#ifndef JITLLM_RUNTIME_API_SERVER_H_
#define JITLLM_RUNTIME_API_SERVER_H_

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <expected>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <vector>

#include "config/node_config.h"
#include "platform/event_loop.h"
#include "runtime/api.h"
#include "runtime/binding.h"
#include "runtime/http.h"
#include "runtime/watchdog.h"

namespace jitllm::runtime::api {

using Clock = std::chrono::steady_clock;

// A connection between requests keeps at most this much of each buffer's
// allocation; a larger one is given back once its request is done, so idle
// connections hold no body's or response's bytes outside the budgets.
inline constexpr std::size_t kKeptBufferBytes = std::size_t{16} << 10U;

// One admitted request as the backend sees it. Each call is progress for
// the watchdog (watchdog.h) and returns whether to go on: false ends the
// generation (a stop string, the peer gone, the backend stalled, a
// non-streaming deadline, the runtime stopping).
class Exchange {
 public:
  Exchange() = default;
  Exchange(const Exchange&) = delete;
  Exchange& operator=(const Exchange&) = delete;
  Exchange(Exchange&&) = delete;
  Exchange& operator=(Exchange&&) = delete;
  virtual ~Exchange() = default;

  struct Admission {
    std::uint32_t prompt_tokens = 0;  // the whole conversation, rendered
    std::uint32_t max_tokens = 0;     // the completion's limit
    std::uint64_t swap_bytes = 0;     // what making the model resident pages in (0: it is)
    Floors floors;                    // the model's (config::ModelEntry)
  };
  // Once, after the model's own checks and before any model work: a
  // streaming response starts here unless it already has, and a
  // non-streaming one's deadline is set from the admission.
  virtual bool Admit(const Admission& admission) = 0;
  // Before a unit of work: a swap paging in `size` bytes, a prefill chunk
  // of `size` rows, decoding (`size` 0: each step is its own beat). The
  // watchdog allows it the stall time plus its expected time at the
  // floors times kWorkMargin.
  virtual bool Next(Phase phase, std::uint64_t size) = 0;
  // Generated text, in order. Reasoning with empty text: the reasoning
  // block ended having said nothing.
  virtual bool Reasoning(std::string_view text) = 0;
  virtual bool Content(std::string_view text) = 0;
  // Between steps that produced no text.
  virtual bool Continue() = 0;
};

struct Completion {
  std::uint32_t completion_tokens = 0;  // generated, a stop token included
  std::uint32_t cached_tokens = 0;      // the prompt's tokens the state already held
  bool stopped = false;                 // ended at the model's stop token
  LiteralResult literal{};              // empty for chat completions
};

class Backend {
 public:
  Backend() = default;
  Backend(const Backend&) = delete;
  Backend& operator=(const Backend&) = delete;
  Backend(Backend&&) = delete;
  Backend& operator=(Backend&&) = delete;
  virtual ~Backend() = default;

  // The models, once, when the server is made.
  virtual std::vector<ModelInfo> Models() const = 0;
  // Runs one request on Run's thread: an Error before Admit refuses it (a
  // 4xx, or a 5xx); after Admit, an Error is a failure mid-response.
  virtual std::expected<Completion, Error> Complete(const ChatRequest& request,
                                                    Exchange& exchange) = 0;
  virtual std::expected<Completion, Error> Complete(const CompletionRequest& request,
                                                    Exchange& exchange) {
    (void)request;
    (void)exchange;
    return std::unexpected(Error{.status = 501,
                                 .type = "server_error",
                                 .message = "literal completions are unavailable",
                                 .param = {},
                                 .code = {}});
  }
  // After the response is handed to the I/O thread: work off the
  // request's path.
  virtual void AfterResponse() {}
  // False once a failure left the backend unable to serve: Run returns.
  virtual bool healthy() const { return true; }
  virtual std::string failure() const { return {}; }
};

struct ServerOptions {
  std::vector<config::ClientEndpoint> bind;  // resolved (binding.h)
  HostGuard hosts;                           // the listening ports are added by Listen
  std::chrono::milliseconds head_timeout{kHeaderTimeoutMs};
  std::chrono::milliseconds body_timeout{kBodyTimeoutMs};
  std::chrono::milliseconds write_timeout{kWriteTimeoutMs};
  std::chrono::milliseconds idle_timeout{kIdleTimeoutMs};
  std::chrono::milliseconds keepalive{kKeepaliveMs};
  std::chrono::milliseconds queue_wait{kQueueWaitMs};  // a non-streaming request's
  // The watchdog's stall time, and a non-streaming deadline's cap
  // ([client] stall_seconds, deadline_cap_seconds).
  std::chrono::milliseconds stall{std::chrono::seconds(config::kDefaultStallSeconds)};
  std::chrono::milliseconds deadline_cap{std::chrono::seconds(config::kDefaultDeadlineCapSeconds)};
  std::size_t max_queued = kMaxQueued;
  std::size_t max_connections = kMaxConnections;
  std::size_t max_unsent = kMaxUnsentBytes;
  std::size_t body_budget = kBodyBudgetBytes;
  std::size_t response_budget = 2 * kMaxCompletionResponseBytes;
  std::FILE* log = nullptr;  // one line a request; nullptr: none
  // Called as the backend turns unhealthy or recovers, on the thread that
  // noticed (the I/O thread, the driver), one call at a time, with the
  // health as it is then (not under the server's lock).
  std::function<void(const Health&)> on_health;
};

class Server {
 public:
  Server(Backend& backend, ServerOptions options);
  Server(const Server&) = delete;
  Server& operator=(const Server&) = delete;
  Server(Server&&) = delete;
  Server& operator=(Server&&) = delete;
  ~Server();

  // Binds every listener; the ports they got, in `bind`'s order.
  std::expected<std::vector<std::uint16_t>, std::string> Listen();
  // Serves until `wake_fd` is readable and `on_wake` (which consumes what
  // made it so) returns true, or the backend fails; then refuses what is
  // still queued (503), lets the I/O thread write out what it holds (for
  // at most a second) and returns. The error: the backend's failure.
  std::expected<void, std::string> Run(int wake_fd, const std::function<bool()>& on_wake);

  // The unknown fields seen so far.
  const IgnoredFields& ignored_fields() const { return ignored_; }
  // The bytes the connections' own buffers hold allocated (read and write,
  // by capacity), as of the I/O thread's last pass (at least once a
  // second). Between requests a connection keeps at most 16 KiB of each.
  std::size_t held_bytes() const { return held_bytes_.load(std::memory_order_relaxed); }
  // Capacity of completed literal response allocations still retained by the
  // server, including a partially sent socket buffer.
  std::size_t response_bytes() const;
  // The backend's health as the watchdog sees it (for M5's management
  // listener; the log says each change).
  Health health() const;

  struct Channel;
  struct Connection;

 private:
  struct Pending {
    std::shared_ptr<Channel> channel;
    ChatRequest request;
    std::optional<CompletionRequest> literal;
  };
  class Stream;

  // I/O thread.
  void Loop();
  void AcceptAll(std::size_t index);
  void OnEvent(Connection& c, std::uint32_t events);
  void OnReadable(Connection& c);
  void OnRequest(Connection& c, http::Request request);
  // Answers with an error; `log`: a line with the status (never the
  // message, which may quote the request).
  void Refuse(Connection& c, const Error& error, std::vector<std::string> extra = {},
              bool log = true);
  void Flush(Connection& c);
  void Drop(Connection& c);
  void Linger(Connection& c);
  // The peer shut its sending side after a whole request: the response goes
  // on, then the connection closes; a probe tells a closed peer apart.
  void InputClosed(Connection& c);
  void Watch(Connection& c);
  Clock::time_point Sweep(Clock::time_point now);
  bool EvictIdle();

  // Driver thread.
  void Serve(Pending& pending, int wake_fd, const std::function<bool()>& on_wake);
  // Progress (with mutex_ held): the watchdog's beat, and a recovery's log
  // line; true when the backend recovered (on_health is then owed).
  bool BeatLocked(Phase next, double expected);

  // I/O thread: whether the backend stalled; if so the running request
  // ends and the queue is refused (connections to flush in `ended`).
  void WatchBackend(Clock::time_point now, std::vector<std::uint64_t>& ended);

  // Any thread.
  void Log(std::string_view line) const;
  void WakeIo() const;
  void HealthChanged() const;  // on_health, if set, with the health as it is

  Backend& backend_;
  ServerOptions options_;
  std::vector<ModelInfo> models_;
  std::int64_t created_ = 0;
  std::vector<http::Fd> listeners_;
  platform::EventLoop loop_;
  platform::Waker stop_;     // the I/O thread ends
  platform::Waker io_wake_;  // a channel has output
  platform::Waker ready_;    // something was queued
  std::jthread io_;
  IgnoredFields ignored_;

  // The I/O thread's own.
  std::unordered_map<std::uint64_t, std::unique_ptr<Connection>> connections_;  // by ID
  std::uint64_t next_id_ = 0;    // connections are numbered from 1
  std::uint64_t activity_ = 0;   // a clock of connection activity, for eviction
  std::size_t body_in_use_ = 0;  // bytes of bodies being received
  Clock::time_point accept_paused_until_;
  Clock::time_point out_of_files_logged_;
  bool draining_ = false;
  std::atomic<std::size_t> held_bytes_{0};  // written by the I/O thread, read by any

  // Under mutex_: the queue, every channel's shared state, stopping_,
  // the watchdog and the running request.
  mutable std::mutex mutex_;
  mutable std::mutex health_mutex_;  // on_health's calls, one at a time
  std::deque<Pending> queue_;
  std::vector<std::uint64_t> dirty_;  // connections whose channels have new output
  std::size_t response_in_use_ = 0;   // channel/socket literal-body allocations
  bool stopping_ = false;
  Watchdog watchdog_;
  std::shared_ptr<Channel> running_;  // the request the driver runs
};

}  // namespace jitllm::runtime::api

#endif  // JITLLM_RUNTIME_API_SERVER_H_
