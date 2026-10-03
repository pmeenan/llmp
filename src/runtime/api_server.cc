// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "runtime/api_server.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <deque>
#include <expected>
#include <format>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <tuple>
#include <utility>
#include <vector>

#include "base/report.h"
#include "base/surface_versions.h"
#include "platform/crash_policy.h"
#include "platform/event_loop.h"
#include "platform/sockets.h"

namespace jitllm::runtime::api {
namespace {

constexpr std::string_view kJson = "application/json";
constexpr std::string_view kContinue = "HTTP/1.1 100 Continue\r\n\r\n";
// Event-loop tags beside connection IDs (which count up from 1).
constexpr std::uint64_t kListenerTag = std::uint64_t{1} << 56U;
constexpr std::uint64_t kStopTag = std::uint64_t{2} << 56U;
constexpr std::uint64_t kWakeTag = std::uint64_t{3} << 56U;
// A closing connection's last reads, so a response is not lost to a reset.
constexpr auto kLinger = std::chrono::milliseconds(200);
constexpr std::size_t kLingerBytes = std::size_t{1} << 20U;
// Accepting pauses this long when the process is out of descriptors.
constexpr auto kAcceptPause = std::chrono::milliseconds(100);
// How long a stopping server lets responses finish going out.
constexpr auto kDrain = std::chrono::seconds(1);
constexpr std::size_t kReadChunk = 65536;
// Sent to an HTTP/1.1 client whose input side closed while its
// non-streaming response is being made: a client that half-closed must
// accept an interim response (RFC 9110, section 15.2), though Python's
// http.client takes any 1xx but 100 as the final one (and undici fails on
// an unasked 100); one that closed entirely answers with a reset, which
// ends the request as a disconnect.
constexpr std::string_view kProcessing = "HTTP/1.1 102 Processing\r\n\r\n";

// Empties a buffer, giving back its allocation when that is large.
void Release(std::string& buffer) {
  if (buffer.capacity() > kKeptBufferBytes) {
    std::string().swap(buffer);
  } else {
    buffer.clear();
  }
}

Error Refusal(int status, std::string message, std::string code = {}) {
  std::string type = "invalid_request_error";
  if (status == 429) {
    type = "rate_limit_error";
  } else if (status == 403) {
    type = "permission_error";
  } else if (status >= 500) {
    type = "server_error";
  }
  return Error{.status = status,
               .type = std::move(type),
               .message = std::move(message),
               .param = {},
               .code = std::move(code)};
}

// While the backend is unhealthy (watchdog.h): what is queued, and what
// arrives, until its next progress.
Error Unresponsive() {
  return Refusal(503, "the model backend has stopped making progress; retry later",
                 "backend_unresponsive");
}

// The running request's end when its backend stalled, `idle` after its
// last progress.
Error Stalled(Clock::duration idle) {
  return Refusal(504,
                 std::format("the model backend made no progress for {} s",
                             std::chrono::duration_cast<std::chrono::seconds>(idle).count()),
                 "backend_stalled");
}

bool Readable(int fd) {
  pollfd p{.fd = fd, .events = POLLIN, .revents = 0};
  return ::poll(&p, 1, 0) > 0 && (p.revents & POLLIN) != 0;
}

std::string RequestId() {
  std::array<std::uint8_t, 12> bytes{};
  if (!platform::FillRandom(std::as_writable_bytes(std::span(bytes)))) {
    // Unique within the process is enough for an opaque ID.
    static std::atomic<std::uint64_t> counter{0};
    const std::uint64_t n = ++counter;
    for (std::size_t i = 0; i < 8; ++i) {
      bytes[i] = static_cast<std::uint8_t>(n >> (8 * i));
    }
  }
  std::string id = "chatcmpl-";
  for (const std::uint8_t b : bytes) {
    id += std::format("{:02x}", b);
  }
  return id;
}

double Seconds(Clock::duration d) { return std::chrono::duration<double>(d).count(); }

bool PeerIsLoopback(const sockaddr_storage& peer) {
  if (peer.ss_family == AF_INET) {
    sockaddr_in four{};
    std::memcpy(&four, &peer, sizeof four);
    return (ntohl(four.sin_addr.s_addr) >> 24U) == 127U;
  }
  if (peer.ss_family == AF_INET6) {
    sockaddr_in6 six{};
    std::memcpy(&six, &peer, sizeof six);
    return std::memcmp(&six.sin6_addr, &in6addr_loopback, sizeof six.sin6_addr) == 0;
  }
  return false;
}

}  // namespace

// ---------------------------------------------------------------- Channel

// One request's response, shared by the I/O thread (which writes it to the
// socket) and the driver (which produces it). The first fields are fixed
// when it is made; the rest are under Server::mutex_.
struct Server::Channel {
  std::uint64_t connection = 0;  // the Connection's ID
  std::string id;                // the request's opaque ID
  std::string model;
  std::int64_t created = 0;
  bool stream = false;  // SSE
  bool http10 = false;
  Clock::time_point queued_at;
  std::chrono::seconds idle_timeout{0};

  std::string out;           // bytes the I/O thread has not taken yet
  std::string literal_body;  // kept separately so transfer never copies a large body
  std::size_t response_reserved = 0;
  bool head_sent = false;   // the head is in `out` or gone out
  bool chunked = false;     // the body is framed in chunks
  bool keep_alive = false;  // the connection serves another request after this
  bool ended = false;       // the response is whole
  bool gone = false;        // the client left or stopped reading: the generation ends
  bool stalled = false;     // the watchdog answered it: the generation ends
  bool queued = false;      // waiting in Server::queue_
  bool dirty = false;       // on Server::dirty_
  Clock::time_point last_out;

  // Everything below: with Server::mutex_ held.
  void PutHead(int status, std::string_view type, std::optional<std::size_t> length,
               std::vector<std::string> extra = {}) {
    extra.push_back(std::format("jitllm-inference-version: {}", surface::kInferenceVersion));
    chunked = !length && !http10;
    if (!length && http10) {
      keep_alive = false;  // the body ends with the connection
    }
    out += http::Head({.status = status,
                       .content_type = type,
                       .length = length,
                       .chunked = chunked,
                       .keep_alive = keep_alive,
                       .idle_timeout = idle_timeout,
                       .extra = std::move(extra)});
    head_sent = true;
    last_out = Clock::now();
  }
  void PutBody(std::string_view data) {
    if (data.empty()) {
      return;
    }
    if (chunked) {
      out += http::Chunk(data);
    } else {
      out += data;
    }
    last_out = Clock::now();
  }
  void PutEvent(std::string_view json) {
    std::string event = "data: ";
    event += json;
    event += "\n\n";
    PutBody(event);
  }
  void PutStart() {
    PutHead(200, "text/event-stream", std::nullopt);
    PutEvent(ChunkJson(id, created, model, Delta::kRole, {}));
  }
  void PutEnd() {
    if (chunked) {
      out += http::Chunk({});
    }
    ended = true;
  }
  // An error: the whole response before the head, an in-stream error
  // event after it (the stream then ends without [DONE]).
  void PutError(const Error& error, std::vector<std::string> extra = {}) {
    if (!head_sent) {
      const std::string body = ErrorJson(error);
      if (error.status == 429 || error.status == 503) {
        extra.push_back(std::format("Retry-After: {}", kRetryAfterSeconds));
        extra.emplace_back("x-should-retry: true");
      }
      PutHead(error.status, kJson, body.size(), std::move(extra));
      PutBody(body);
    } else if (stream) {
      PutEvent(ErrorJson(error));
    }
    PutEnd();
  }
};

// ---------------------------------------------------------------- Connection

// A client connection: the I/O thread's alone.
struct Server::Connection {
  enum class State : std::uint8_t {
    kIdle,    // between requests (or none yet)
    kHead,    // reading a request's head
    kBody,    // reading its body
    kBusy,    // its response is being made or written
    kLinger,  // half-closed, discarding what still arrives
  };
  std::uint64_t id = 0;
  http::Fd fd;
  bool peer_loopback = false;
  State state = State::kIdle;
  std::string in;
  std::size_t head_end = 0;
  http::Request request;
  std::size_t reserved = 0;  // of the body budget
  Clock::time_point idle_by;
  Clock::time_point head_by;
  Clock::time_point body_by;
  Clock::time_point linger_by;
  Clock::time_point progress_at;  // the last write that went out, or when output became pending
  std::string wbuf;
  std::size_t woff = 0;
  std::size_t response_reserved = 0;  // follows a literal body's full allocation
  std::shared_ptr<Channel> channel;
  bool close_after = false;   // pipelined, refused mid-request, or input closed: no reuse
  bool input_closed = false;  // the peer shut its sending side after a whole request
  std::uint32_t interest = 0;
  std::uint64_t last_active = 0;
  std::size_t lingered = 0;
  bool dead = false;

  bool pending_output() const { return woff < wbuf.size(); }
};

// ---------------------------------------------------------------- Stream

// One admitted request's response as the driver makes it.
class Server::Stream final : public Exchange {
 public:
  // `started`: when the driver took the request up, which a non-streaming
  // deadline counts from.
  Stream(Server& server, Pending& pending, int wake_fd, const std::function<bool()>& on_wake,
         Clock::time_point started, bool cooperative = false)
      : server_(server),
        pending_(pending),
        channel_(*pending.channel),
        wake_fd_(wake_fd),
        on_wake_(on_wake),
        started_(started),
        text_(pending.request.stop),
        cooperative_(cooperative) {}

  bool Admit(const Admission& admission) override {
    usage_.prompt_tokens = admission.prompt_tokens;
    floors_ = admission.floors;
    if (!pending_.request.stream) {
      // A stream has no deadline; a non-streaming request's is its work's.
      deadline_ = started_ + ScaledDeadline(server_.options_.stall, server_.options_.deadline_cap,
                                            floors_, admission.swap_bytes, admission.prompt_tokens,
                                            admission.max_tokens);
    }
    if (pending_.request.stream) {
      const std::scoped_lock lock(server_.mutex_);
      if (!channel_.gone && !channel_.stalled && !channel_.head_sent) {
        channel_.PutStart();
        Dirty();
      }
    }
    server_.WakeIo();
    return Check(std::nullopt, 0);
  }
  bool Next(Phase phase, std::uint64_t size) override {
    return Check(phase, ExpectedSeconds(phase, size, floors_));
  }
  bool Reasoning(std::string_view text) override {
    Emit(text_.Reasoning(text));
    return Check(std::nullopt, 0);
  }
  bool Content(std::string_view text) override {
    Emit(text_.Content(text));
    if (text_.stopped()) {
      Progress(std::nullopt, 0);  // the answer is whole: nothing else to ask
      return false;
    }
    return Check(std::nullopt, 0);
  }
  bool Continue() override { return Check(std::nullopt, 0); }

  // Hands the rest of the response to the I/O thread for the backend's
  // result; the status for the log.
  int End(const std::expected<Completion, Error>& result) {
    {
      const std::scoped_lock lock(server_.mutex_);
      if (channel_.stalled || channel_.gone) {
        return channel_.stalled ? 504 : 499;
      }
    }
    // Large literal score arrays are driver-owned. Serialize without the
    // shared I/O lock so another connection can still disconnect or read.
    std::optional<std::expected<std::string, Error>> literal_body;
    if (pending_.literal && result) {
      EmitLocked(text_.Finish());  // non-streaming: only local strings
      usage_.completion_tokens = result->completion_tokens;
      usage_.cached_tokens = result->cached_tokens;
      const Finish finish = result->stopped || text_.stopped() ? Finish::kStop : Finish::kLength;
      literal_body = LiteralCompletionJson(channel_.id, channel_.created, *pending_.literal,
                                           content_, result->literal, finish, usage_);
      timed_out_ = timed_out_ || Clock::now() > deadline_;
    }
    int status = 200;
    {
      const std::scoped_lock lock(server_.mutex_);
      if (channel_.stalled || channel_.gone) {
        // The watchdog answered it (a 504), or the client closed the
        // request or stopped reading (499, logged only).
        return channel_.stalled ? 504 : 499;
      }
      if (!result) {
        channel_.PutError(result.error());
        status = result.error().status;
      } else {
        if (!pending_.literal) {
          EmitLocked(text_.Finish());
        }
        usage_.completion_tokens = result->completion_tokens;
        usage_.cached_tokens = result->cached_tokens;
        if (timed_out_ || stopping_) {
          const Error error =
              timed_out_
                  ? Refusal(504, std::format("the request passed its {} s deadline",
                                             std::chrono::duration_cast<std::chrono::seconds>(
                                                 deadline_ - started_)
                                                 .count()))
                  : Refusal(503, "the runtime is stopping");
          if (stopping_) {
            channel_.keep_alive = false;
          }
          channel_.PutError(error);
          status = error.status;
        } else {
          const Finish finish =
              result->stopped || text_.stopped() ? Finish::kStop : Finish::kLength;
          const std::string& model = pending_.request.model;
          if (pending_.request.stream) {
            if (!channel_.head_sent) {
              channel_.PutStart();
            }
            channel_.PutEvent(
                ChunkJson(channel_.id, channel_.created, model, Delta::kFinish, {}, finish));
            if (pending_.request.include_usage) {
              channel_.PutEvent(UsageChunkJson(channel_.id, channel_.created, model, usage_));
            }
            channel_.PutBody("data: [DONE]\n\n");
            channel_.PutEnd();
          } else {
            auto body = literal_body
                            ? std::move(*literal_body)
                            : std::expected<std::string, Error>(CompletionJson(
                                  channel_.id, channel_.created, model, content_,
                                  reasoning_.empty() ? std::nullopt
                                                     : std::optional<std::string>(reasoning_),
                                  finish, usage_));
            if (!body) {
              channel_.PutError(body.error());
              status = body.error().status;
            } else if (pending_.literal &&
                       (body->capacity() > server_.options_.response_budget ||
                        server_.response_in_use_ >
                            server_.options_.response_budget - body->capacity())) {
              channel_.PutError(Refusal(503, "completed response buffers are full; retry later",
                                        "response_budget_exceeded"));
              status = 503;
            } else {
              channel_.PutHead(200, kJson, body->size());
              if (pending_.literal) {
                channel_.response_reserved = body->capacity();
                server_.response_in_use_ += channel_.response_reserved;
                channel_.literal_body = std::move(*body);
              } else {
                channel_.PutBody(*body);
              }
              channel_.PutEnd();
            }
          }
        }
      }
      Dirty();
    }
    server_.WakeIo();
    return status;
  }

  const Usage& usage() const { return usage_; }
  bool slow() const { return slow_; }

 private:
  // With the lock held.
  void Dirty() {
    if (!channel_.dirty) {
      channel_.dirty = true;
      server_.dirty_.push_back(channel_.connection);
    }
  }

  void EmitLocked(const OutputText::Out& out) {
    if (!pending_.request.stream) {
      reasoning_ += out.reasoning;
      content_ += out.content;
      return;
    }
    if (channel_.gone || channel_.stalled || (out.reasoning.empty() && out.content.empty())) {
      return;
    }
    if (!channel_.head_sent) {
      channel_.PutStart();
    }
    if (!out.reasoning.empty()) {
      channel_.PutEvent(ChunkJson(channel_.id, channel_.created, pending_.request.model,
                                  Delta::kReasoning, out.reasoning));
    }
    if (!out.content.empty()) {
      channel_.PutEvent(ChunkJson(channel_.id, channel_.created, pending_.request.model,
                                  Delta::kContent, out.content));
    }
    if (channel_.out.size() > server_.options_.max_unsent) {
      // A reader this far behind is not reading: it is dropped rather than
      // let the buffer grow.
      channel_.gone = true;
      slow_ = true;
    }
    Dirty();
  }

  void Emit(const OutputText::Out& out) {
    if (!pending_.request.stream) {
      EmitLocked(out);  // local text only
      return;
    }
    {
      const std::scoped_lock lock(server_.mutex_);
      EmitLocked(out);
    }
    server_.WakeIo();
  }

  // Progress: the watchdog's beat, `next` the unit that follows (none:
  // the one under way goes on) and its expected seconds at the floors.
  void Progress(std::optional<Phase> next, double expected) {
    if (cooperative_) {
      return;  // the driver declares and completes units, not channel polling
    }
    bool recovered = false;
    {
      const std::scoped_lock lock(server_.mutex_);
      recovered = server_.BeatLocked(next.value_or(server_.watchdog_.health().phase), expected);
    }
    if (recovered) {
      server_.HealthChanged();
    }
  }

  // A beat (above), then whether the generation goes on.
  bool Check(std::optional<Phase> next, double expected) {
    Progress(next, expected);
    if (timed_out_ || stopping_) {
      return false;
    }
    {
      const std::scoped_lock lock(server_.mutex_);
      stopping_ = stopping_ || server_.stopping_;
      if (stopping_ || channel_.gone || channel_.stalled) {
        return false;
      }
    }
    if (Clock::now() > deadline_) {
      timed_out_ = true;
      return false;
    }
    if (Readable(wake_fd_) && on_wake_()) {
      stopping_ = true;
      const std::scoped_lock lock(server_.mutex_);
      server_.stopping_ = true;
      return false;
    }
    return true;
  }

  Server& server_;
  Pending& pending_;
  Channel& channel_;
  int wake_fd_;
  const std::function<bool()>& on_wake_;
  Clock::time_point started_;
  Clock::time_point deadline_ = Clock::time_point::max();  // a stream's: none
  Floors floors_;
  OutputText text_;
  Usage usage_;
  std::string reasoning_;
  std::string content_;
  bool timed_out_ = false;
  bool stopping_ = false;
  bool slow_ = false;
  bool cooperative_ = false;
};

// Stable addresses: Start may borrow the exchange and the parsed request
// until Retire proves every such reference has gone. Slots are never moved.
struct Server::Active {
  Pending pending;
  CooperativeBackend::Request request;
  Clock::time_point started;
  Stream exchange;
  std::unique_ptr<CooperativeBackend::Work> work;

  Active(Server& server, Pending request, int wake_fd, const std::function<bool()>& on_wake)
      : pending(std::move(request)),
        request{.options = pending.request,
                .literal = pending.literal ? &*pending.literal : nullptr},
        started(Clock::now()),
        exchange(server, pending, wake_fd, on_wake, started, true) {}
};

// ---------------------------------------------------------------- Server

Server::Server(Backend& backend, ServerOptions options)
    : backend_(backend),
      options_(std::move(options)),
      models_(backend.Models()),
      created_(static_cast<std::int64_t>(std::time(nullptr))),
      loop_(platform::EventLoop::Open()),
      stop_(platform::Waker::Open()),
      io_wake_(platform::Waker::Open()),
      ready_(platform::Waker::Open()),
      watchdog_(options_.stall, Clock::now()) {}

Server::~Server() {
  if (io_.joinable()) {
    stop_.Signal();
    io_.join();
  }
}

std::expected<std::vector<std::uint16_t>, std::string> Server::Listen() {
  if (!loop_.valid() || !stop_.valid() || !io_wake_.valid() || !ready_.valid()) {
    return std::unexpected("the event loop or a waker failed");
  }
  if (options_.bind.empty()) {
    return std::unexpected("there is nothing to listen on");
  }
  std::vector<std::uint16_t> ports;
  for (const config::ClientEndpoint& endpoint : options_.bind) {
    auto listener = http::Listen(endpoint);
    if (!listener) {
      listeners_.clear();
      return std::unexpected(listener.error());
    }
    listeners_.push_back(std::move(listener->fd));
    ports.push_back(listener->port);
    options_.hosts.AddPort(listener->port);
  }
  return ports;
}

void Server::Log(std::string_view line) const {
  if (options_.log == nullptr) {
    return;
  }
  const std::string text = std::format("jitllm-runtime: {}\n", line);
  (void)std::fwrite(text.data(), 1, text.size(), options_.log);
  (void)std::fflush(options_.log);
}

void Server::WakeIo() const { io_wake_.Signal(); }

Health Server::health() const {
  const std::scoped_lock lock(mutex_);
  return watchdog_.health();
}

std::size_t Server::response_bytes() const {
  const std::scoped_lock lock(mutex_);
  return response_in_use_;
}

void Server::HealthChanged() const {
  if (!options_.on_health) {
    return;
  }
  // Serialized, each with the health as it is then: the last report is
  // the current state whichever thread noticed the change last.
  const std::scoped_lock lock(health_mutex_);
  options_.on_health(health());
}

bool Server::BeatLocked(Phase next, double expected) {
  const auto now = Clock::now();
  const Health before = watchdog_.health();
  if (!watchdog_.Beat(next, expected, now)) {
    return false;
  }
  Log(
      std::format("the model backend made progress again ({}), {:.1f} s after its last; "
                  "requests are accepted again",
                  PhaseName(before.phase), Seconds(now - before.last_progress)));
  return true;
}

// ---------------------------------------------------------------- I/O thread

void Server::WatchBackend(Clock::time_point now, std::vector<std::uint64_t>& ended) {
  {
    const std::scoped_lock lock(mutex_);
    if (!watchdog_.Check(now)) {
      return;
    }
    const Health& health = watchdog_.health();
    const Clock::duration idle = now - health.last_progress;
    std::string which = "no request";
    for (const auto& running : running_) {
      if (running->ended || running->gone) {
        continue;
      }
      // Ended as a client that leaves ends it: the generation stops at the
      // backend's next step, and the lease is released as it returns.
      Channel& ch = *running;
      which = std::format("request {} ends with 504", ch.id);
      ch.PutError(Stalled(idle));
      ch.stalled = true;  // not `gone`: the connection stays to carry the error
      ended.push_back(ch.connection);
    }
    Log(
        std::format("the model backend made no progress for {:.1f} s ({}; stall {}): {}; it is "
                    "unhealthy, and requests get 503 until it makes progress",
                    Seconds(idle), PhaseName(health.phase), health.stalls, which));
    // Nothing queued would run before the backend moves again.
    for (Pending& p : queue_) {
      Channel& ch = *p.channel;
      ch.queued = false;
      Log(std::format("request {}: 503, the backend is not making progress", ch.id));
      ch.PutError(Unresponsive());
      ended.push_back(ch.connection);
    }
    queue_.clear();
  }
  HealthChanged();
}

void Server::Watch(Connection& c) {
  if (c.dead) {
    return;
  }
  // A closed input side reads as ready forever: it is watched no longer,
  // except while lingering, which reads to the end and then drops.
  std::uint32_t want = c.input_closed ? 0U : platform::kPeerClosed;
  if (c.state == Connection::State::kLinger ||
      (!c.input_closed && (c.state != Connection::State::kBusy || !c.close_after))) {
    want |= platform::kReadable;
  }
  if (c.pending_output()) {
    want |= platform::kWritable;
  }
  if (want != c.interest) {
    std::ignore = loop_.Change(c.fd.get(), c.id, want);
    c.interest = want;
  }
}

bool Server::EvictIdle() {
  Connection* oldest = nullptr;
  for (auto& [id, c] : connections_) {
    if (!c->dead && c->state == Connection::State::kIdle && c->in.empty() &&
        (oldest == nullptr || c->last_active < oldest->last_active)) {
      oldest = c.get();
    }
  }
  if (oldest == nullptr) {
    return false;
  }
  Drop(*oldest);
  return true;
}

void Server::AcceptAll(std::size_t index) {
  const int listener = listeners_[index].get();
  for (;;) {
    sockaddr_storage peer{};
    const int fd = platform::AcceptConnection(listener, peer);
    if (fd < 0) {
      if (errno == EINTR || errno == ECONNABORTED || errno == EPROTO) {
        continue;
      }
      if (errno == EMFILE || errno == ENFILE || errno == ENOBUFS || errno == ENOMEM) {
        // Out of descriptors: stop accepting for a moment rather than spin,
        // and say so at most once a minute (the pause repeats every 100 ms).
        if (out_of_files_logged_ == Clock::time_point{} ||
            Clock::now() - out_of_files_logged_ >= std::chrono::minutes(1)) {
          Log("the chat route is out of file descriptors; pausing new connections");
          out_of_files_logged_ = Clock::now();
        }
        accept_paused_until_ = Clock::now() + kAcceptPause;
        for (const http::Fd& l : listeners_) {
          loop_.Remove(l.get());
        }
      }
      return;
    }
    http::Fd accepted(fd);
    std::size_t live = 0;
    for (const auto& [id, c] : connections_) {
      live += c->dead ? 0 : 1;
    }
    if (live >= options_.max_connections && !EvictIdle()) {
      const std::string body = ErrorJson(Refusal(503, "too many connections; retry later"));
      const std::string response =
          http::Head({.status = 503,
                      .content_type = kJson,
                      .length = body.size(),
                      .chunked = false,
                      .keep_alive = false,
                      .idle_timeout = {},
                      .extra = {std::format("Retry-After: {}", kRetryAfterSeconds),
                                "x-should-retry: true"}}) +
          body;
      (void)platform::SendNoSignal(fd, response.data(), response.size(), false);
      continue;
    }
    const int on = 1;
    (void)::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &on, sizeof on);
    auto c = std::make_unique<Connection>();
    c->id = ++next_id_;
    c->fd = std::move(accepted);
    c->peer_loopback = PeerIsLoopback(peer);
    c->idle_by = Clock::now() + options_.idle_timeout;
    c->last_active = ++activity_;
    c->interest = platform::kReadable | platform::kPeerClosed;
    if (!loop_.Add(fd, c->id, c->interest)) {
      continue;
    }
    connections_.emplace(c->id, std::move(c));
  }
}

void Server::Drop(Connection& c) {
  if (c.dead) {
    return;
  }
  {
    const std::scoped_lock lock(mutex_);
    if (c.channel) {
      c.channel->gone = true;
      if (c.channel->queued) {
        c.channel->queued = false;
        std::erase_if(queue_, [&](const Pending& p) { return p.channel == c.channel; });
      }
      Release(c.channel->literal_body);
      response_in_use_ -= std::exchange(c.channel->response_reserved, 0);
      Release(c.channel->out);
    }
    Release(c.wbuf);
    response_in_use_ -= std::exchange(c.response_reserved, 0);
  }
  body_in_use_ -= c.reserved;
  c.reserved = 0;
  loop_.Remove(c.fd.get());
  c.fd = http::Fd();
  c.channel.reset();
  c.dead = true;
}

void Server::Linger(Connection& c) {
  (void)::shutdown(c.fd.get(), SHUT_WR);
  c.state = Connection::State::kLinger;
  c.linger_by = Clock::now() + kLinger;
  c.channel.reset();
  Watch(c);
}

void Server::Refuse(Connection& c, const Error& error, std::vector<std::string> extra, bool log) {
  if (log) {
    Log(std::format("refused a request: {}", error.status));
  }
  if (!c.channel) {
    // Before a request was whole: nobody knows where the next would begin.
    c.close_after = true;
    auto channel = std::make_shared<Channel>();
    channel->connection = c.id;
    channel->http10 = c.request.http10;
    channel->idle_timeout = std::chrono::duration_cast<std::chrono::seconds>(options_.idle_timeout);
    c.channel = std::move(channel);
    body_in_use_ -= c.reserved;
    c.reserved = 0;
  }
  c.state = Connection::State::kBusy;
  {
    const std::scoped_lock lock(mutex_);
    if (c.close_after) {
      c.channel->keep_alive = false;
    }
    c.channel->PutError(error, std::move(extra));
  }
  Flush(c);
}

void Server::Flush(Connection& c) {
  while (!c.dead) {
    if (!c.pending_output()) {
      Release(c.wbuf);
      c.woff = 0;
      bool ended = false;
      bool gone = false;
      bool keep_alive = false;
      {
        const std::scoped_lock lock(mutex_);
        response_in_use_ -= std::exchange(c.response_reserved, 0);
        if (c.channel) {
          if (!c.channel->out.empty()) {
            c.wbuf.swap(c.channel->out);
          } else if (!c.channel->literal_body.empty()) {
            c.wbuf.swap(c.channel->literal_body);
            c.response_reserved = std::exchange(c.channel->response_reserved, 0);
          }
          ended = c.channel->ended;
          gone = c.channel->gone;
          keep_alive = c.channel->keep_alive;
        }
      }
      if (gone) {
        Drop(c);
        return;
      }
      if (c.wbuf.empty()) {
        if (c.channel && ended && c.state == Connection::State::kBusy) {
          // The response is out.
          c.channel.reset();
          if (keep_alive && !c.close_after && !draining_) {
            c.state = Connection::State::kIdle;
            Release(c.in);
            c.head_end = 0;
            c.request = {};
            c.idle_by = Clock::now() + options_.idle_timeout;
          } else {
            Linger(c);
            return;
          }
        }
        Watch(c);
        return;
      }
      c.progress_at = Clock::now();
    }
    const ssize_t n =
        platform::SendNoSignal(c.fd.get(), c.wbuf.data() + c.woff, c.wbuf.size() - c.woff, false);
    if (n > 0) {
      c.woff += static_cast<std::size_t>(n);
      c.progress_at = Clock::now();
      c.last_active = ++activity_;
      continue;
    }
    if (n < 0 && errno == EINTR) {
      continue;
    }
    if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
      Watch(c);  // writable
      return;
    }
    Drop(c);  // the peer is gone
    return;
  }
}

void Server::OnReadable(Connection& c) {
  using State = Connection::State;
  const http::Limits limits{.max_header_bytes = kMaxHeaderBytes,
                            .max_headers = kMaxHeaders,
                            .max_target_bytes = kMaxTargetBytes,
                            .max_body_bytes = kMaxBodyBytes};
  std::array<char, kReadChunk> chunk{};
  while (!c.dead) {
    if (c.state == State::kLinger) {
      const ssize_t n = ::recv(c.fd.get(), chunk.data(), chunk.size(), 0);
      if (n > 0) {
        c.lingered += static_cast<std::size_t>(n);
        if (c.lingered < kLingerBytes) {
          continue;
        }
      } else if (n < 0 && errno == EINTR) {
        continue;
      } else if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
        return;
      }
      Drop(c);
      return;
    }
    if (c.state == State::kBusy) {
      // Bytes before the response has ended: a pipelined request, which is
      // not served; the connection closes after this response. Or the end.
      const ssize_t n = ::recv(c.fd.get(), chunk.data(), 1, 0);
      if (n == 0) {
        InputClosed(c);
        return;
      }
      if (n < 0 && errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR) {
        Drop(c);
        return;
      }
      if (n > 0) {
        c.close_after = true;
        if (c.channel) {
          const std::scoped_lock lock(mutex_);
          c.channel->keep_alive = false;
        }
        Watch(c);
      }
      return;
    }
    // kIdle, kHead, kBody: at most what the request may still hold.
    std::size_t most = kReadChunk;
    if (c.state == State::kBody) {
      most = std::min(most, c.head_end + c.request.body_bytes - c.in.size());
    } else {
      most = std::min(most,
                      limits.max_header_bytes + 4 - std::min(c.in.size(), limits.max_header_bytes));
    }
    const ssize_t n = ::recv(c.fd.get(), chunk.data(), most, 0);
    if (n < 0 && errno == EINTR) {
      continue;
    }
    if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
      return;
    }
    if (n <= 0) {
      Drop(c);  // the peer closed (mid-request: nobody to answer)
      return;
    }
    c.last_active = ++activity_;
    if (c.state == State::kIdle) {
      const auto now = Clock::now();
      c.state = State::kHead;
      c.head_by = now + options_.head_timeout;
      c.body_by = now + options_.body_timeout;
    }
    c.in.append(chunk.data(), static_cast<std::size_t>(n));
    if (c.state == State::kHead) {
      auto end = http::FindHeadEnd(c.in, limits);
      if (!end) {
        Refuse(c, Refusal(end.error().status, end.error().message));
        return;
      }
      if (!*end) {
        continue;
      }
      auto request = http::ParseHead(std::string_view(c.in).substr(0, **end), limits);
      if (!request) {
        Refuse(c, Refusal(request.error().status, request.error().message));
        return;
      }
      c.head_end = **end;
      c.request = std::move(*request);
      if (c.request.body_bytes > 0) {
        if (body_in_use_ + c.request.body_bytes > options_.body_budget) {
          Refuse(c, Refusal(503, "too many request bodies are arriving at once; retry later"));
          return;
        }
        body_in_use_ += c.request.body_bytes;
        c.reserved = c.request.body_bytes;
        // The buffer grows to what the budget was charged for, no further.
        c.in.reserve(c.head_end + c.request.body_bytes);
      }
      c.state = State::kBody;
      if (c.request.expect_continue && c.in.size() < c.head_end + c.request.body_bytes) {
        c.wbuf.append(kContinue);
        c.progress_at = Clock::now();
        Flush(c);
        if (c.dead) {
          return;
        }
      }
    }
    if (c.state == State::kBody && c.in.size() >= c.head_end + c.request.body_bytes) {
      http::Request request = std::move(c.request);
      request.body = c.in.substr(c.head_end, request.body_bytes);
      if (c.in.size() > c.head_end + request.body_bytes) {
        c.close_after = true;  // bytes after the body: a pipelined request
      }
      Release(c.in);  // the body's bytes are the request's now, not the connection's
      body_in_use_ -= c.reserved;
      c.reserved = 0;
      c.request = {};
      c.request.http10 = request.http10;
      OnRequest(c, std::move(request));
      return;
    }
  }
}

void Server::OnRequest(Connection& c, http::Request request) {
  auto channel = std::make_shared<Channel>();
  channel->connection = c.id;
  channel->id = RequestId();
  channel->created = static_cast<std::int64_t>(std::time(nullptr));
  channel->http10 = request.http10;
  channel->idle_timeout = std::chrono::duration_cast<std::chrono::seconds>(options_.idle_timeout);
  channel->keep_alive = request.KeepAlive() && !c.close_after;
  c.channel = channel;
  c.state = Connection::State::kBusy;

  // Browser guards (D-045, D-064): the Host names this node as it listens,
  // and nothing arrives from another site's page.
  const std::string* host = request.Find("host");
  if (host == nullptr || !options_.hosts.AllowsHost(*host)) {
    Refuse(c, Refusal(403, "the Host header must name this node as the chat route listens"));
    return;
  }
  const std::string* origin = request.Find("origin");
  const std::string* site = request.Find("sec-fetch-site");
  if ((origin != nullptr && !options_.hosts.AllowsOrigin(*origin)) ||
      (site != nullptr && *site != "none" && *site != "same-origin")) {
    Refuse(c, Refusal(403, "requests from other sites' web pages are refused on this route"));
    return;
  }
  const auto method_not_allowed = [&](std::string_view allow) {
    Refuse(c, Refusal(405, std::format("use {}", allow)), {std::format("Allow: {}", allow)});
  };
  const auto answer = [&](const std::string& body) {
    {
      const std::scoped_lock lock(mutex_);
      channel->PutHead(200, kJson, body.size());
      channel->PutBody(body);
      channel->PutEnd();
    }
    Flush(c);
  };
  if (request.path == "/jitllm/v1/ignored-fields" && c.peer_loopback) {
    if (request.method != "GET") {
      method_not_allowed("GET");
      return;
    }
    answer(ignored_.Json());
    return;
  }
  if (request.path == "/v1/models" || request.path.starts_with("/v1/models/")) {
    if (request.method != "GET") {
      method_not_allowed("GET");
      return;
    }
    if (request.path == "/v1/models") {
      answer(ModelsJson(models_, created_));
      return;
    }
    const std::string_view id = std::string_view(request.path).substr(11);
    const auto it = std::ranges::find(models_, id, &ModelInfo::name);
    if (it == models_.end()) {
      Refuse(c, Refusal(404, "The model does not exist", "model_not_found"));
      return;
    }
    answer(ModelJson(*it, created_));
    return;
  }
  const bool literal_route = request.path == "/v1/completions";
  if (request.path != "/v1/chat/completions" && !literal_route) {
    Refuse(c, Refusal(404,
                      "no such route; this runtime serves /v1/chat/completions, /v1/completions "
                      "and /v1/models"));
    return;
  }
  if (request.method != "POST") {
    method_not_allowed("POST");
    return;
  }
  const std::string* type = request.Find("content-type");
  std::string media;
  if (type != nullptr) {
    for (const char ch : std::string_view(*type).substr(0, type->find(';'))) {
      if (ch != ' ' && ch != '\t') {
        media += ch >= 'A' && ch <= 'Z' ? static_cast<char>(ch - 'A' + 'a') : ch;
      }
    }
  }
  if (media != kJson) {
    Refuse(c, Refusal(415, "the body must be Content-Type: application/json"));
    return;
  }
  std::optional<CompletionRequest> literal;
  std::expected<ChatRequest, Error> parsed;
  if (literal_route) {
    auto raw = ParseCompletionRequest(request.body);
    if (raw) {
      parsed = raw->options;
      literal = std::move(*raw);
    } else {
      parsed = std::unexpected(raw.error());
    }
  } else {
    parsed = ParseChatRequest(request.body);
  }
  request.body.clear();
  request.body.shrink_to_fit();
  if (!parsed) {
    Refuse(c, parsed.error());
    return;
  }
  for (const std::string& name :
       ignored_.Record(parsed->ignored, static_cast<std::int64_t>(std::time(nullptr)))) {
    Log(
        std::format("an unknown request field is ignored: {} (logged once; "
                    "GET /jitllm/v1/ignored-fields counts them)",
                    base::Printable(name)));
  }
  const auto model = std::ranges::find(models_, parsed->model, &ModelInfo::name);
  if (model == models_.end()) {
    Error error = Refusal(404, "The model does not exist or is not configured on this node",
                          "model_not_found");
    error.param = "model";
    Refuse(c, error);
    return;
  }
  if (!model->chat) {
    Error error =
        Refusal(400, "This is not a text model and thus not supported by a completions endpoint",
                "model_not_supported");
    error.param = "model";
    Refuse(c, error);
    return;
  }
  channel->model = parsed->model;
  if (literal_route) {
    channel->id.replace(0, 9, "cmpl-");
  }
  channel->stream = parsed->stream;
  channel->queued_at = Clock::now();
  bool stopping = false;
  bool unhealthy = false;
  {
    const std::scoped_lock lock(mutex_);
    stopping = stopping_;
    unhealthy = !watchdog_.health().healthy;
    if (!stopping && !unhealthy && queue_.size() < options_.max_queued) {
      channel->queued = true;
      queue_.push_back(
          {.channel = channel, .request = std::move(*parsed), .literal = std::move(literal)});
      ready_.Signal();
      return;
    }
  }
  if (unhealthy && !stopping) {
    // The backend stalled and has not made progress since: nothing queued
    // now would run until it does (watchdog.h).
    Log(std::format("request {}: 503, the backend is not making progress", channel->id));
    Refuse(c, Unresponsive(), {}, false);
    return;
  }
  if (stopping) {
    Log(std::format("request {}: 503, the runtime is stopping", channel->id));
    {
      const std::scoped_lock lock(mutex_);
      channel->keep_alive = false;
    }
    Refuse(c, Refusal(503, "the runtime is stopping"), {}, false);
    return;
  }
  Log(std::format("request {}: 429, {} requests queued", channel->id, options_.max_queued));
  Refuse(c,
         Refusal(429,
                 std::format("{} requests are already waiting; retry later", options_.max_queued)),
         {}, false);
}

void Server::OnEvent(Connection& c, std::uint32_t events) {
  if ((events & platform::kError) != 0) {
    Drop(c);
    return;
  }
  if ((events & platform::kReadable) != 0) {
    OnReadable(c);
  }
  if (!c.dead && (events & platform::kWritable) != 0) {
    Flush(c);
  }
  if (!c.dead && (events & (platform::kPeerClosed | platform::kHangUp)) != 0) {
    // The peer closed its sending side. Reading states find that out from
    // recv (after taking what it sent): a request not yet whole is a
    // disconnect. A whole request's response goes on (InputClosed); a
    // connection closed both ways (a reset) ends it.
    if ((events & platform::kHangUp) != 0) {
      Drop(c);
    } else if (c.state == Connection::State::kBusy) {
      InputClosed(c);
    } else if ((c.interest & platform::kReadable) != 0 && (events & platform::kReadable) == 0) {
      OnReadable(c);
    }
  }
}

void Server::InputClosed(Connection& c) {
  if (c.input_closed) {
    return;
  }
  c.input_closed = true;
  c.close_after = true;  // nothing more can arrive: the connection closes after this response
  if (c.channel) {
    {
      const std::scoped_lock lock(mutex_);
      Channel& ch = *c.channel;
      ch.keep_alive = false;
      // Whether the peer shut only its sending side or closed entirely looks
      // the same until something is sent: a closed socket answers with a
      // reset (an error event, a disconnect, which ends the generation). A
      // stream sends what any client parses: its start (as after `keepalive`
      // in the queue) or a comment. A non-streaming response has nothing to send
      // before its head but an interim response: a 102 on HTTP/1.1 (no 1xx
      // is safe for every client, docs/runtime-serving.md, but a client
      // that half-closes is rare). On HTTP/1.0 it runs to its end.
      if (!ch.ended && !ch.gone) {
        if (ch.stream && !ch.head_sent) {
          ch.PutStart();
        } else if (ch.stream) {
          ch.PutBody(": keepalive\n\n");
        } else if (!ch.head_sent && !ch.http10) {
          ch.out += kProcessing;
        }
      }
    }
    Flush(c);
    return;
  }
  Watch(c);
}

Clock::time_point Server::Sweep(Clock::time_point now) {
  using State = Connection::State;
  Clock::time_point next = now + std::chrono::seconds(1);
  const auto soon = [&](Clock::time_point t) { next = std::min(next, t); };
  // The backend: a stall ends the running request and refuses the queue.
  std::vector<std::uint64_t> expired;
  WatchBackend(now, expired);
  // The queue: a non-streaming request that waited too long gets a 429; a
  // stream, held by keepalives, waits as long as the backend makes
  // progress.
  {
    const std::scoped_lock lock(mutex_);
    if (const auto due = watchdog_.due()) {
      soon(*due);
    }
    std::erase_if(queue_, [&](const Pending& p) {
      Channel& ch = *p.channel;
      if (ch.stream) {
        return false;
      }
      if (now - ch.queued_at < options_.queue_wait) {
        soon(ch.queued_at + options_.queue_wait);
        return false;
      }
      ch.queued = false;
      Log(std::format("request {}: 429 after waiting {} s in the queue", ch.id,
                      options_.queue_wait.count() / 1000));
      ch.PutError(Refusal(429, "the request waited too long behind others; retry later"));
      expired.push_back(ch.connection);
      return true;
    });
  }
  for (const std::uint64_t id : expired) {
    if (auto it = connections_.find(id); it != connections_.end()) {
      Flush(*it->second);
    }
  }
  for (auto& [id, owned] : connections_) {
    Connection& c = *owned;
    if (c.dead) {
      continue;
    }
    switch (c.state) {
      case State::kIdle:
        if (now >= c.idle_by) {
          Drop(c);
        } else {
          soon(c.idle_by);
        }
        break;
      case State::kHead:
      case State::kBody: {
        const Clock::time_point by = c.state == State::kHead ? c.head_by : c.body_by;
        if (now >= by) {
          Refuse(c, Refusal(408, c.state == State::kHead ? "the request head took too long"
                                                         : "the request body took too long"));
        } else {
          soon(by);
        }
        break;
      }
      case State::kLinger:
        if (now >= c.linger_by) {
          Drop(c);
        } else {
          soon(c.linger_by);
        }
        break;
      case State::kBusy: {
        if (c.pending_output()) {
          if (now - c.progress_at >= options_.write_timeout) {
            Log("a client stopped reading its response; the connection is closed");
            Drop(c);
            break;
          }
          soon(c.progress_at + options_.write_timeout);
        }
        bool added = false;
        if (c.channel) {
          Channel& ch = *c.channel;
          const std::scoped_lock lock(mutex_);
          if (ch.stream && !ch.ended && !ch.gone) {
            if (!ch.head_sent && ch.queued) {
              // A stream that waits its turn starts now, so the client
              // sees it admitted; the wait goes on with keepalives.
              if (now >= ch.queued_at + options_.keepalive) {
                ch.PutStart();
                added = true;
              } else {
                soon(ch.queued_at + options_.keepalive);
              }
            } else if (ch.head_sent) {
              if (now >= ch.last_out + options_.keepalive) {
                ch.PutBody(": keepalive\n\n");
                added = true;
              }
              soon(ch.last_out + options_.keepalive);
            }
          }
        }
        if (added) {
          Flush(c);
        }
        break;
      }
    }
  }
  std::size_t held = 0;
  for (const auto& [id, c] : connections_) {
    if (!c->dead) {
      held += c->in.capacity() + c->wbuf.capacity();
    }
  }
  held_bytes_.store(held, std::memory_order_relaxed);
  return next;
}

void Server::Loop() {
  if (auto stack = platform::InstallThreadSignalStack(); !stack) {
    Log("the chat route's I/O thread has no signal stack: " + stack.error());
  }
  std::array<platform::ReadyEvent, 256> events{};
  Clock::time_point drain_by{};
  for (;;) {
    const auto now = Clock::now();
    if (accept_paused_until_ != Clock::time_point{} && now >= accept_paused_until_ && !draining_) {
      accept_paused_until_ = {};
      for (std::size_t i = 0; i < listeners_.size(); ++i) {
        std::ignore = loop_.Add(listeners_[i].get(), kListenerTag | i, platform::kReadable);
      }
    }
    Clock::time_point next = Sweep(now);
    if (accept_paused_until_ != Clock::time_point{}) {
      next = std::min(next, accept_paused_until_);
    }
    std::erase_if(connections_, [](const auto& entry) { return entry.second->dead; });
    if (draining_) {
      const bool busy = std::ranges::any_of(connections_, [](const auto& entry) {
        return entry.second->state == Connection::State::kBusy ||
               entry.second->state == Connection::State::kLinger;
      });
      if (!busy || now >= drain_by) {
        for (auto& [id, c] : connections_) {
          Drop(*c);
        }
        connections_.clear();
        return;
      }
      next = std::min(next, drain_by);
    }
    const auto wait = std::chrono::ceil<std::chrono::milliseconds>(next - Clock::now());
    const auto n = loop_.Wait(
        events, std::chrono::milliseconds(std::clamp<std::int64_t>(wait.count(), 0, 1000)));
    if (!n) {
      if (n.error() == std::errc::interrupted) {
        continue;
      }
      Log(std::format("the chat route's event loop failed: {}", n.error().value()));
      return;
    }
    for (std::size_t i = 0; i < *n; ++i) {
      const std::uint64_t tag = events[i].tag;
      const std::uint32_t flags = events[i].ready;
      if (tag == kStopTag) {
        stop_.Drain();
        if (!draining_) {
          draining_ = true;
          drain_by = Clock::now() + kDrain;
          for (const http::Fd& l : listeners_) {
            loop_.Remove(l.get());
          }
          listeners_.clear();
          for (auto& [id, c] : connections_) {
            if (c->state != Connection::State::kBusy) {
              Drop(*c);
            }
          }
        }
      } else if (tag == kWakeTag) {
        io_wake_.Drain();
        std::vector<std::uint64_t> dirty;
        {
          const std::scoped_lock lock(mutex_);
          dirty.swap(dirty_);
          for (const std::uint64_t id : dirty) {
            if (auto it = connections_.find(id); it != connections_.end() && it->second->channel) {
              it->second->channel->dirty = false;
            }
          }
        }
        for (const std::uint64_t id : dirty) {
          if (auto it = connections_.find(id); it != connections_.end()) {
            Flush(*it->second);
          }
        }
      } else if ((tag & kListenerTag) != 0 && tag < kStopTag) {
        if (!draining_) {
          AcceptAll(static_cast<std::size_t>(tag & ~kListenerTag));
        }
      } else if (auto it = connections_.find(tag); it != connections_.end()) {
        OnEvent(*it->second, flags);
      }
    }
  }
}

// ---------------------------------------------------------------- driver

void Server::Serve(Pending& pending, int wake_fd, const std::function<bool()>& on_wake) {
  const auto started = Clock::now();
  Channel& channel = *pending.channel;
  bool recovered = false;
  {
    const std::scoped_lock lock(mutex_);
    if (channel.gone) {
      Log(std::format("request {}: the client left while it was queued", channel.id));
      return;
    }
    running_.push_back(pending.channel);
    recovered = BeatLocked(Phase::kStarting, 0);
  }
  if (recovered) {
    HealthChanged();
  }
  Stream stream(*this, pending, wake_fd, on_wake, started);
  const std::expected<Completion, Error> result = pending.literal
                                                      ? backend_.Complete(*pending.literal, stream)
                                                      : backend_.Complete(pending.request, stream);
  FinishResponse(pending, stream, result, started);
  Progress(Phase::kIdle, 0);
}

void Server::Progress(Phase next, double expected) {
  bool recovered = false;
  {
    const std::scoped_lock lock(mutex_);
    recovered = BeatLocked(next, expected);
  }
  if (recovered) {
    HealthChanged();
  }
}

void Server::FinishResponse(Pending& pending, Stream& stream,
                            const std::expected<Completion, Error>& result,
                            Clock::time_point started) {
  Channel& channel = *pending.channel;
  const int status = stream.End(result);
  bool stalled = false;
  bool recovered = false;
  {
    const std::scoped_lock lock(mutex_);
    stalled = channel.stalled;
    std::erase(running_, pending.channel);
    // The backend returned: progress, and what follows the response.
    recovered = BeatLocked(Phase::kFinishing, 0);
  }
  if (recovered) {
    HealthChanged();
  }
  const Usage& u = stream.usage();
  std::string_view why;
  if (stalled) {
    why = " (the backend made no progress)";
  } else if (status == 499) {
    why = stream.slow() ? " (the client stopped reading)" : " (the client left)";
  }
  Log(std::format(
      "request {}: {} {}{}, {} prompt tokens ({} cached), {} generated, {:.3f} s "
      "({:.3f} s queued)",
      channel.id, pending.request.model, status, why, u.prompt_tokens, u.cached_tokens,
      u.completion_tokens, Seconds(Clock::now() - started), Seconds(started - channel.queued_at)));
  backend_.AfterResponse();
}

std::expected<void, std::string> Server::ServeCooperative(Pending first,
                                                          CooperativeBackend& cooperative,
                                                          int wake_fd,
                                                          const std::function<bool()>& on_wake) {
  std::vector<std::unique_ptr<Active>> active;
  active.reserve(kMaxActiveRequests);
  bool drain = false;
  std::string fatal;
  // False means deferred with no effects. The caller still owns the frame
  // and either restores it to the FIFO or uses the serial path alone.
  const auto start = [&](std::unique_ptr<Active>& frame) {
    {
      const std::scoped_lock lock(mutex_);
      if (frame->pending.channel->gone) {
        Log(std::format("request {}: the client left while it was queued",
                        frame->pending.channel->id));
        return true;
      }
      running_.push_back(frame->pending.channel);
    }
    Progress(Phase::kStarting, 0);
    auto created = cooperative.Start(frame->request, frame->exchange);
    if (!created) {
      FinishResponse(frame->pending, frame->exchange, std::unexpected(created.error()),
                     frame->started);
      return true;
    }
    if (!*created) {
      const std::scoped_lock lock(mutex_);
      std::erase(running_, frame->pending.channel);
      return false;
    }
    frame->work = std::move(*created);
    active.push_back(std::move(frame));
    return true;
  };

  auto initial = std::make_unique<Active>(*this, std::move(first), wake_fd, on_wake);
  if (!start(initial)) {
    // A request that cannot join an empty cohort must not wait forever.
    // Start's no-effects contract makes one serial fallback safe.
    Serve(initial->pending, wake_fd, on_wake);
    Progress(Phase::kIdle, 0);
    return {};
  }
  const auto retire = [&](bool stop) {
    std::erase_if(active, [&](const std::unique_ptr<Active>& frame) {
      const bool go_on = frame->exchange.Continue();
      if (!go_on || stop || !fatal.empty()) {
        frame->work->Cancel();
      } else if (!frame->work->terminal()) {
        return false;
      }
      Progress(Phase::kFinishing, 0);
      auto retired = cooperative.Retire(*frame->work);
      if (!retired.references_retired) {
        (void)std::fputs("cooperative backend did not retire its references: aborting\n", stderr);
        std::abort();
      }
      if (!fatal.empty()) {
        retired.result = std::unexpected(
            Refusal(500, "the model backend failed; the runtime is stopping", "backend_failed"));
      }
      FinishResponse(frame->pending, frame->exchange, retired.result, frame->started);
      return true;
    });
  };
  while (!active.empty()) {
    if (Readable(wake_fd) && on_wake()) {
      const std::scoped_lock lock(mutex_);
      stopping_ = true;
    }
    if (!backend_.healthy() && fatal.empty()) {
      fatal = backend_.failure();
      if (fatal.empty()) {
        fatal = "the cooperative backend failed";
      }
    }
    bool stop = false;
    {
      const std::scoped_lock lock(mutex_);
      stop = stopping_;
    }
    // Retire before admitting a replacement. The exchange and parsed
    // inputs survive even after their socket disappeared.
    retire(stop);
    if (!backend_.healthy() && fatal.empty()) {
      fatal = backend_.failure().empty() ? "the cooperative backend failed" : backend_.failure();
    }
    if (active.empty()) {
      break;
    }
    if (stop || !fatal.empty()) {
      continue;
    }

    // Failed or departed starters do not grow active.size(). Bound attempts
    // as well as live works so an endless invalid FIFO cannot starve decode.
    std::size_t attempts = 0;
    while (!drain && active.size() < kMaxActiveRequests && attempts < kMaxActiveRequests) {
      std::optional<Pending> pending;
      {
        const std::scoped_lock lock(mutex_);
        if (stopping_ || queue_.empty()) {
          break;
        }
        Pending& head = queue_.front();
        if (head.request.model != active.front()->pending.request.model ||
            !cooperative.Supports(
                {.options = head.request, .literal = head.literal ? &*head.literal : nullptr})) {
          drain = true;  // never refill past an incompatible FIFO head
          break;
        }
        pending.emplace(std::move(head));
        queue_.pop_front();
        pending->channel->queued = false;
      }
      ++attempts;
      auto frame = std::make_unique<Active>(*this, std::move(*pending), wake_fd, on_wake);
      if (!start(frame)) {
        const std::scoped_lock lock(mutex_);
        frame->pending.channel->queued = true;
        queue_.push_front(std::move(frame->pending));
        break;
      }
    }
    if (!backend_.healthy() && fatal.empty()) {
      fatal = backend_.failure().empty() ? "the cooperative backend failed" : backend_.failure();
    }
    {
      const std::scoped_lock lock(mutex_);
      stop = stopping_;
    }
    // A new work can already be terminal, and intake may have consumed a
    // shutdown signal. Recheck without starting another admission cycle.
    retire(stop);
    if (active.empty()) {
      break;
    }
    if (stop || !fatal.empty()) {
      continue;
    }

    // Between completed units: the backend's housekeeping (it leaves every
    // active Work's resources alone).
    backend_.Maintain();
    std::array<CooperativeBackend::Work*, kMaxActiveRequests> work{};
    for (std::size_t i = 0; i < active.size(); ++i) {
      work[i] = active[i]->work.get();
    }
    const std::span<CooperativeBackend::Work* const> wave(work.data(), active.size());
    auto next = cooperative.NextUnit(wave);
    if (!next) {
      fatal = next.error().empty() ? "no cooperative backend unit" : next.error();
      continue;
    }
    if (next->phase == Phase::kIdle || !std::isfinite(next->expected_seconds) ||
        next->expected_seconds < 0) {
      fatal = "invalid cooperative backend unit";
      continue;
    }
    Progress(next->phase, next->expected_seconds);
    auto advanced = cooperative.Advance(wave);
    // This is an actual unit's return, including failure. Exchange polling
    // and output from its members never renew a hung unit's allowance.
    Progress(Phase::kFinishing, 0);
    if (!advanced) {
      fatal = advanced.error().empty() ? "the cooperative backend unit failed" : advanced.error();
    }
  }
  Progress(Phase::kIdle, 0);
  return fatal.empty() ? std::expected<void, std::string>{} : std::unexpected(std::move(fatal));
}

std::expected<void, std::string> Server::Run(int wake_fd, const std::function<bool()>& on_wake) {
  if (listeners_.empty()) {
    return std::unexpected("the chat route is not listening");
  }
  std::expected<void, std::error_code> added =
      loop_.Add(stop_.descriptor(), kStopTag, platform::kReadable);
  added = added ? loop_.Add(io_wake_.descriptor(), kWakeTag, platform::kReadable) : added;
  for (std::size_t i = 0; i < listeners_.size(); ++i) {
    added = added ? loop_.Add(listeners_[i].get(), kListenerTag | i, platform::kReadable) : added;
  }
  if (!added) {
    return std::unexpected(std::format("the event loop: {}", added.error().value()));
  }
  io_ = std::jthread([this] { Loop(); });
  CooperativeBackend* const cooperative = backend_.cooperative();
  std::expected<void, std::string> result;
  for (;;) {
    {
      const std::scoped_lock lock(mutex_);
      if (stopping_) {
        break;
      }
    }
    // Idle, the backend's housekeeping runs between waits (Backend::Maintain:
    // retention, the spill budget, memory pressure from outside).
    backend_.Maintain();
    std::array<pollfd, 2> fds{{{.fd = wake_fd, .events = POLLIN, .revents = 0},
                               {.fd = ready_.descriptor(), .events = POLLIN, .revents = 0}}};
    const int n = ::poll(fds.data(), fds.size(), static_cast<int>(kMaintenanceMs));
    if (n < 0) {
      if (errno == EINTR) {
        continue;
      }
      result = std::unexpected(std::format("poll: {}", errno));
      break;
    }
    if ((fds[0].revents & POLLIN) != 0 && on_wake()) {
      break;
    }
    if ((fds[1].revents & POLLIN) == 0) {
      continue;
    }
    ready_.Drain();
    for (;;) {
      std::optional<Pending> next;
      {
        const std::scoped_lock lock(mutex_);
        if (stopping_ || queue_.empty()) {
          break;
        }
        next.emplace(std::move(queue_.front()));
        next->channel->queued = false;
        queue_.pop_front();
      }
      if (cooperative != nullptr &&
          cooperative->Supports(
              {.options = next->request, .literal = next->literal ? &*next->literal : nullptr})) {
        result = ServeCooperative(std::move(*next), *cooperative, wake_fd, on_wake);
      } else {
        Serve(*next, wake_fd, on_wake);
      }
      if (!result) {
        break;
      }
      if (!backend_.healthy()) {
        result = std::unexpected(backend_.failure());
        break;
      }
      if (Readable(wake_fd) && on_wake()) {
        const std::scoped_lock lock(mutex_);
        stopping_ = true;
      }
    }
    if (!result) {
      break;
    }
  }
  // Whatever still waits is refused; the I/O thread writes it out.
  {
    const std::scoped_lock lock(mutex_);
    stopping_ = true;
    for (Pending& p : queue_) {
      Channel& channel = *p.channel;
      channel.queued = false;
      channel.keep_alive = false;
      Log(std::format("request {}: 503, the runtime is stopping", channel.id));
      channel.PutError(Refusal(503, "the runtime is stopping"));
      if (!channel.dirty) {
        channel.dirty = true;
        dirty_.push_back(channel.connection);
      }
    }
    queue_.clear();
  }
  WakeIo();
  stop_.Signal();
  io_.join();
  return result;
}

}  // namespace jitllm::runtime::api
