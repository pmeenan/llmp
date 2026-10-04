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
// model; a parse thread parses bodies of kParseOffThreadBytes and more, so
// a large one holds up no other connection. Connections persist (HTTP/1.1
// keep-alive, closed after idle_timeout idle), up to max_connections
// open (idle ones closed oldest first to admit a new one); a request that
// arrives on a connection before the previous response has ended
// (pipelining) is not read: that connection closes after the response. A
// request whose head or body sends no byte for request_inactivity gets a
// 408; a slow one that keeps sending is never cut off (D-102).
// The thread that calls Run is the backend's (the node's one driver): it
// takes queued requests in arrival order. Backends without a cooperative
// capability run one to its end; a cooperative backend admits as many
// same-model requests as the model has request slots (at most
// kMaxActiveRequests) and advances declared units, draining before an
// incompatible FIFO head. The driver watches the caller's wake descriptor
// (the runtime's signals) between units. By default the queue has no count
// and no wait limit (D-102); with max_queued set, more get a 429 with
// Retry-After, and with queue_wait set, a non-streaming request that waits
// longer gets the same 429. A stream, held by keepalives, waits its turn.
//
// Progress (watchdog.h). On the serial path every backend Exchange call is
// a beat; Next names the unit that follows (a swap, a prefill chunk,
// decode) and its size, which with the model's floors sets how long it
// may take: `stall` plus kWorkMargin times its expected time.
// Cooperative execution declares and completes units explicitly:
// polling its channels or emitting text never renews an in-flight unit.
// The I/O thread watches the beats: past that allowance the backend is
// unhealthy (health(), the log, `on_health`) until its next beat, which the
// unit it hung in makes when it returns. Detection is not a limit (D-102):
// by default a stall only reports. With stall_fails the active requests
// also end (a 504 before the headers, an in-stream error after), their
// generation is cancelled as when a client leaves, what is queued gets a
// 503 and so do new requests until the next beat. A genuine hang is
// recovered (D-102, hang_ladder.h): the I/O thread feeds the beats and the
// node's progress to the hang ladder, which confirms a hang once nothing at
// all moves for ServerOptions::hang past the unit's allowance, and
// escalates: the driver's node wait cancels the stuck work (the backend
// then fails the requests that needed it and recovers its model), and
// when nothing less frees it the ladder's last resort restarts the
// process for its supervisor. A request has no fixed deadline: a stream runs
// until it is done or its client leaves; a non-streaming one too unless
// deadline_cap is set, when it ends at its scaled deadline (watchdog.h
// ScaledDeadline, capped there).
//
// Output. The driver never touches a socket: a request's response is a
// Channel it appends to (whole events, under the server's lock) and the
// I/O thread writes it out as the socket takes it. A stream whose client
// lets more than stream_buffer bytes pile up gets backpressure: its request
// pauses at its next completed step until the client has taken them (the
// serial path waits there; a cooperative backend leaves a paused request
// out of its units, Exchange::Paused, and the others go on), and the
// watchdog sees a pause, not a stall. A paused request that keeps queued
// requests waiting (the serial path's, or a cohort's that must drain for
// another model or is full) for yield_after yields its place: its
// generation ends at that completed step, its state kept as a finished
// turn's (spilled when memory needs it), and it goes to the back of the
// queue, continuing from there when taken up again (Exchange::Yielding).
// Only with write_inactivity set is a
// client that takes nothing for that long dropped. A connection that
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
// Memory (intake_limits.h): every request's host memory is charged to one
// pool by real size: a body as it arrives, its parse's working set, the
// parsed request while it lives, a stream's unread output, a non-streaming
// response's text and its completed body (which the charge follows to the
// socket). A charge that does not fit is refused before the work (413
// past the whole pool, else 503 with Retry-After).
//
// A request's end. Non-streaming: one JSON body once the outcome is known
// (a 504 past a configured deadline or on a stall with stall_fails, a 503
// when the runtime stops),
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
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <expected>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <vector>

#include "config/node_config.h"
#include "platform/event_loop.h"
#include "runtime/api.h"
#include "runtime/binding.h"
#include "runtime/hang_ladder.h"
#include "runtime/http.h"
#include "runtime/intake_limits.h"
#include "runtime/watchdog.h"

namespace jitllm::runtime::api {

using Clock = std::chrono::steady_clock;

// A connection between requests keeps at most this much of each buffer's
// allocation; a larger one is given back once its request is done, so idle
// connections hold no body's or response's bytes outside the budgets.
inline constexpr std::size_t kKeptBufferBytes = std::size_t{16} << 10U;

// One admitted request as the backend sees it. Each call on the serial
// path is watchdog progress; cooperative units declare progress separately.
// Every call returns whether to go on: false ends the
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
    Floors floors;                    // the model's (its settings, model_settings.h)
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
  // Whether the client has more of a stream's output unread than the
  // server buffers (backpressure, D-102): a cooperative backend leaves the
  // request out of its next units until it is not; on the serial path the
  // calls above wait instead. Any thread; cheap.
  virtual bool Paused() const { return false; }
  // Whether the request is to yield its place: paused for its client while
  // it keeps queued requests waiting (on the serial path, or a cohort that
  // must drain or is full), when one of the calls above returned false for
  // it. The backend then ends its generation at that completed step, its
  // state kept as any finished turn's, and returns a Completion with
  // `yielded` (what continues it); the server queues it again, and it
  // continues when it is next taken up (Backend::Complete with `resume`).
  virtual bool Yielding() const { return false; }
};

// What a backend continues a yielded request from (its own type).
class Yielded {
 public:
  Yielded() = default;
  Yielded(const Yielded&) = delete;
  Yielded& operator=(const Yielded&) = delete;
  Yielded(Yielded&&) = delete;
  Yielded& operator=(Yielded&&) = delete;
  virtual ~Yielded() = default;
};

struct Completion {
  std::uint32_t completion_tokens = 0;  // generated, a stop token included
  std::uint32_t cached_tokens = 0;      // the prompt's tokens the state already held
  bool stopped = false;                 // ended at the model's stop token
  LiteralResult literal{};              // empty for chat completions
  // Set when the request yielded its place (Exchange::Yielding): the
  // counts above are so far, and the response goes on when it continues.
  std::shared_ptr<Yielded> yielded = nullptr;
};

// Optional native cooperative execution on the node's one driver. Start
// creates host-owned request state; Advance performs one declared unit and
// returns only at its completed boundary. No API object may be borrowed by
// work after Retire proves its references retired, including error outcomes.
class CooperativeBackend {
 public:
  struct Request {
    const ChatRequest& options;
    const CompletionRequest* literal = nullptr;
    // A request that yielded its place before, to continue (Exchange::Yielding).
    const Yielded* resume = nullptr;
  };
  class Work {
   public:
    Work() = default;
    Work(const Work&) = delete;
    Work& operator=(const Work&) = delete;
    Work(Work&&) = delete;
    Work& operator=(Work&&) = delete;
    virtual ~Work() = default;
    virtual bool terminal() const = 0;
    // Marks only this request. Its outstanding effects still retire.
    virtual void Cancel() = 0;
    // Asks the request to yield its place at its completed step
    // (Exchange::Yielding): it turns terminal, and its Retire returns a
    // Completion with `yielded`. False if it cannot (it has no generation
    // to continue yet): it goes on.
    virtual bool Yield() { return false; }
    // Pause for one other model's turn, at this completed unit. Unlike
    // reader backpressure, a prompt or a capacity waiter may continue too.
    // True makes this work terminal; Retire must supply its continuation.
    // False has no effects and keeps the work running until it can pause.
    virtual bool YieldForSwitch() { return false; }
  };
  struct Unit {
    Phase phase = Phase::kStarting;
    double expected_seconds = 0;
  };
  struct Retirement {
    std::expected<Completion, Error> result;
    // False is an undrainable ownership failure: the server aborts before
    // freeing Work, its Exchange or another active request's frame.
    bool references_retired = false;
  };

  CooperativeBackend() = default;
  CooperativeBackend(const CooperativeBackend&) = delete;
  CooperativeBackend& operator=(const CooperativeBackend&) = delete;
  CooperativeBackend(CooperativeBackend&&) = delete;
  CooperativeBackend& operator=(CooperativeBackend&&) = delete;
  virtual ~CooperativeBackend() = default;
  // Fast host-only eligibility, called with the queue lock held. No waits
  // or callbacks. The driver batches only one model and at most
  // kMaxActiveRequests works; Start defers past the model's own slots.
  // This descriptor is borrowed only for this call.
  virtual bool Supports(const Request& request) const = 0;
  // Validation and admission/session creation only; native work belongs to
  // Advance, but for one exception: admission by memory may give memory
  // back through the node's reclaim order (spilling idle conversations
  // outside the cohort, dropping plans and graphs), completed jobs on the
  // driver between units, as a refused unit's reclaim does
  // (docs/runtime-serving.md#request-slots). A null work means capacity deferred: no Admit, output,
  // mutation or retained reference. With no active work it falls back to Complete. An Error
  // likewise retains no references and ends only this request. On success, Request itself, its
  // parsed members and Exchange have stable addresses until Retire proves their references retired.
  virtual std::expected<std::unique_ptr<Work>, Error> Start(const Request& request,
                                                            Exchange& exchange) = 0;
  // A unit of Phase::kPaused means every runnable work waits for its
  // client to read (Exchange::Paused); the server then waits for one to
  // read, leave or the runtime to stop. Any other nonterminal work without
  // a next unit is a backend fault, as is a kPaused unit while no work is
  // paused. Both methods run on the driver, without the queue/output lock.
  virtual std::expected<Unit, std::string> NextUnit(std::span<Work* const> work) = 0;
  virtual std::expected<void, std::string> Advance(std::span<Work* const> work) = 0;
  // Cancelled, terminal and failed work all pass through this completion
  // boundary. It may settle native effects; its allowance is the stall
  // interval. It must not return an unproven lifetime as an ordinary Error.
  virtual Retirement Retire(Work& work) = 0;
};

// The most works a cooperative backend runs at once: a model's most request
// slots (engine/request_cohort.h kMaxRequestSlots).
inline constexpr std::size_t kMaxActiveRequests = 16;

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
  // A request that yielded its place (Completion::yielded) goes on from
  // there, as Complete. Only a backend that yields implements it.
  virtual std::expected<Completion, Error> Resume(const ChatRequest& request, Exchange& exchange,
                                                  const Yielded& from) {
    (void)request;
    (void)exchange;
    (void)from;
    return std::unexpected(Error{.status = 500,
                                 .type = "server_error",
                                 .message = "the request cannot continue",
                                 .param = {},
                                 .code = {}});
  }
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
  virtual std::expected<Completion, Error> Resume(const CompletionRequest& request,
                                                  Exchange& exchange, const Yielded& from) {
    (void)request;
    (void)exchange;
    (void)from;
    return std::unexpected(Error{.status = 500,
                                 .type = "server_error",
                                 .message = "the literal request cannot continue",
                                 .param = {},
                                 .code = {}});
  }
  // Absent, every request retains the existing serial Complete path.
  virtual CooperativeBackend* cooperative() { return nullptr; }
  // After the response is handed to the I/O thread: work off the
  // request's path. With cooperative execution, this must preserve every
  // other active Work's resources, references and cohort ownership.
  virtual void AfterResponse() {}
  // The backend's own housekeeping, on Run's thread between completed
  // units and while idle (at least every kMaintenanceMs): with cooperative
  // execution, it must preserve every active Work's resources.
  virtual void Maintain() {}
  // False once a failure left the backend unable to serve: Run returns.
  virtual bool healthy() const { return true; }
  virtual std::string failure() const { return {}; }
};

// The server's limits, each from [client] (config/node_config.h): every
// default permissive (D-102). An absent optional limit does not apply.
struct ServerOptions {
  std::vector<config::ClientEndpoint> bind;  // resolved (binding.h)
  HostGuard hosts;                           // the listening ports are added by Listen
  // A request's head or body with no byte arriving for this long: a 408.
  std::chrono::milliseconds request_inactivity{
      std::chrono::seconds(config::kDefaultRequestInactivitySeconds)};
  // A client that takes no output for this long is dropped; none: never
  // (it gets backpressure only).
  std::optional<std::chrono::milliseconds> write_inactivity;
  std::chrono::milliseconds idle_timeout{std::chrono::seconds(config::kDefaultIdleSeconds)};
  std::chrono::milliseconds keepalive{kKeepaliveMs};
  // A non-streaming request's most time in the queue; none: no limit.
  std::optional<std::chrono::milliseconds> queue_wait;
  // The watchdog's stall time; whether a stall also fails the requests
  // ([client] stall_action "fail"); a non-streaming deadline's cap (none:
  // no deadline).
  std::chrono::milliseconds stall{std::chrono::seconds(config::kDefaultStallSeconds)};
  bool stall_fails = false;
  std::optional<std::chrono::milliseconds> deadline_cap;
  std::optional<std::size_t> max_queued;                                  // none: no count
  std::size_t max_connections = std::numeric_limits<std::size_t>::max();  // the process's limit
  // The request memory and what follows from it (intake_limits.h): the
  // pool every request's host memory is charged to, a body's most and a
  // stream's unread output before it pauses. `memory`, when given, is the
  // pool (shared with the backend, which charges renderings to it and
  // grows it); otherwise the server makes a fixed one of
  // intake.request_capacity bytes.
  IntakeLimits intake = NominalIntakeLimits();
  std::shared_ptr<RequestMemory> memory;
  // A paused stream that keeps queued requests waiting this long yields
  // its place (Exchange::Yielding).
  std::chrono::milliseconds yield_after{std::chrono::seconds(10)};
  // A pending other model may pause a cohort after this much resident
  // work. One substitute cohort runs to completion, then this cohort resumes
  // ahead of later arrivals. This is a scheduling quantum, not a request
  // deadline; no request is truncated when its turn ends.
  std::chrono::milliseconds model_turn{std::chrono::seconds(30)};
  // Hang recovery (D-102; hang_ladder.h). The I/O thread feeds `ladder` the
  // backend's activity (`activity`: the node's progress count, read each
  // sweep) and advances it; the server's beats tell it each unit and its
  // allowance. Without one the server makes its own, of `hang` (none:
  // DefaultHang) with `on_hang` its last resort (run once, on the I/O
  // thread, with why), as the tests do. Not owned: it outlives the server.
  HangLadder* ladder = nullptr;
  std::optional<std::chrono::milliseconds> hang;
  std::function<std::uint64_t()> activity;
  std::function<void(const std::string&)> on_hang;
  std::FILE* log = nullptr;  // one line a request; nullptr: none
  // Called as the backend turns unhealthy or recovers, on the thread that
  // noticed (the I/O thread, the driver), one call at a time, with the
  // health as it is then (not under the server's lock).
  std::function<void(const Health&)> on_health;
};

// The hang's default ([client] hang_seconds absent): ten minutes, or this
// many stall times if longer.
inline constexpr std::chrono::minutes kHangFloor{10};
inline constexpr int kHangStalls = 5;
std::chrono::milliseconds DefaultHang(std::chrono::milliseconds stall);

// A body at least this large is parsed on the server's parse thread, so the
// I/O thread goes on serving every other connection meanwhile.
inline constexpr std::size_t kParseOffThreadBytes = std::size_t{64} << 10U;

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
  // Response bytes still waiting for sockets, as of the same I/O pass.
  std::size_t output_bytes() const { return output_bytes_.load(std::memory_order_relaxed); }
  // The request memory pool (ServerOptions::memory), and what is charged
  // to it now (requests' bodies, parses, queued requests, unread output
  // and responses).
  RequestMemory& request_memory() const { return *memory_; }
  std::size_t response_bytes() const { return static_cast<std::size_t>(memory_->used()); }
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
    MemoryCharge charge;  // the parsed request's bytes
    // A request that yielded its place: what its backend continues it from
    // and its response's progress (Stream).
    std::shared_ptr<Yielded> resume;
    std::optional<OutputText> text;
    Usage usage{};
    std::string reasoning;  // a non-streaming response's text so far
    std::string content;
    MemoryCharge text_charge;
    std::uint64_t pauses = 0;
    std::optional<Clock::time_point> started;
  };
  struct Parse;
  class Stream;
  struct Active;

  // I/O thread.
  void Loop();
  void AcceptAll(std::size_t index);
  void OnEvent(Connection& c, std::uint32_t events);
  void OnReadable(Connection& c);
  void OnRequest(Connection& c, http::Request request);
  // A parsed body's request (inline, or from the parse thread): admitted to
  // the queue, or refused.
  void OnParsed(Connection& c, Parse& parse);
  // The parse thread: parses queued bodies, hands each result back.
  void ParseLoop(const std::stop_token& stop);
  void TakeParsed();
  // Grows a body's buffer by doubling (to hold `more` bytes more), charged
  // to the request memory as it grows: kWait when the memory cannot take it
  // now (the connection is not read until the driver grows it), kRefused
  // (answered) when it never could or the driver could not.
  enum class Growth : std::uint8_t { kOk, kWait, kRefused };
  Growth GrowBody(Connection& c, std::size_t more);
  // A body waiting for the request memory: read again once it fits.
  void RetryBody(Connection& c);
  // A whole body arrived: the request handed on (OnRequest).
  void BodyArrived(Connection& c);
  // Hang recovery's watch (ServerOptions::ladder), without mutex_ held.
  void WatchHang(Clock::time_point now);
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
  std::expected<void, std::string> ServeCooperative(Pending first, CooperativeBackend& cooperative,
                                                    int wake_fd,
                                                    const std::function<bool()>& on_wake,
                                                    bool substitute = false);
  void FinishResponse(Pending& pending, Stream& stream,
                      const std::expected<Completion, Error>& result, Clock::time_point started);
  // A request that yielded its place goes to the back of the queue, with
  // its response's progress (`stream`'s) and its backend's continuation.
  void Requeue(Pending& pending, Stream& stream, std::shared_ptr<Yielded> yielded);
  // With mutex_ held: whether requests wait in the queue that a paused
  // request keeps from running, and since when a channel paused would make
  // it yield (ServerOptions::yield_after).
  bool YieldDueLocked(const Channel& channel, Clock::time_point now) const;
  // With mutex_ held: a parked request whose client read is queued again.
  void UnparkLocked(const Channel& channel);
  // After a large request's memory is freed: the C library's free heap
  // back to the system.
  void TrimIfOwed();
  void Progress(Phase next, double expected);
  // Progress (with mutex_ held): the watchdog's beat, and a recovery's log
  // line; true when the backend recovered (on_health is then owed).
  bool BeatLocked(Phase next, double expected);
  // Backpressure: waits while `paused` (called with mutex_ held) holds and
  // the runtime is not stopping, the watchdog seeing a pause meanwhile,
  // then resumes the phase it saw before. False when the runtime stops.
  bool AwaitReaders(const std::function<bool()>& paused, int wake_fd,
                    const std::function<bool()>& on_wake);

  // I/O thread: whether the backend stalled; if so it is reported, and
  // with stall_fails the active requests end and the queue is refused
  // (connections to flush in `ended`; FailStalledLocked, mutex_ held).
  void WatchBackend(Clock::time_point now, std::vector<std::uint64_t>& ended);
  void FailStalledLocked(Clock::duration idle, std::vector<std::uint64_t>& ended);

  // Any thread.
  void Log(std::string_view line) const;
  void WakeIo() const;
  void HealthChanged() const;  // on_health, if set, with the health as it is

  Backend& backend_;
  ServerOptions options_;
  // Hang recovery's ladder: options_.ladder, or the server's own.
  std::unique_ptr<HangLadder> own_ladder_;
  HangLadder* ladder_ = nullptr;
  // The request memory (ServerOptions::memory): declared before everything
  // that holds a charge to it, so it outlives them.
  std::shared_ptr<RequestMemory> memory_;
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
  std::uint64_t next_id_ = 0;   // connections are numbered from 1
  std::uint64_t activity_ = 0;  // a clock of connection activity, for eviction
  Clock::time_point accept_paused_until_;
  Clock::time_point out_of_files_logged_;
  bool draining_ = false;
  std::atomic<std::size_t> held_bytes_{0};  // written by the I/O thread, read by any
  std::atomic<std::size_t> output_bytes_{0};

  // Under mutex_: the queue, every channel's shared state, stopping_,
  // the watchdog and the running request.
  mutable std::mutex mutex_;
  std::condition_variable drained_;  // a paused channel's output was taken, or it ended
  mutable std::mutex health_mutex_;  // on_health's calls, one at a time
  std::deque<Pending> queue_;
  std::uint64_t enqueued_ = 0;          // requests ever queued (a waiting cohort's wake)
  std::size_t suspended_requests_ = 0;  // waiting for a substitute, driver-owned under mutex_
  std::vector<std::uint64_t> dirty_;    // connections whose channels have new output
  bool stopping_ = false;
  // Requests ahead of the queue that the active ones keep waiting: a
  // cohort draining for another model, or full (the driver's, read by
  // Stream; YieldDueLocked).
  bool blocked_ = false;
  bool trim_owed_ = false;  // the driver's (TrimIfOwed)
  Watchdog watchdog_;
  std::vector<std::shared_ptr<Channel>> running_;  // bounded, driver-owned active requests
  // Requests that yielded their place while their clients do not read:
  // queued again once their clients take their output (Flush), so they are
  // not taken up (and their models swapped in) only to yield again.
  std::deque<Pending> parked_;

  // The parse thread's: bodies to parse and results to hand back (under
  // parse_mutex_), the I/O thread woken for each.
  std::mutex parse_mutex_;
  std::condition_variable_any parse_ready_;
  std::deque<std::unique_ptr<Parse>> parse_in_;
  std::vector<std::unique_ptr<Parse>> parse_out_;
  std::jthread parser_;
};

}  // namespace jitllm::runtime::api

#endif  // JITLLM_RUNTIME_API_SERVER_H_
