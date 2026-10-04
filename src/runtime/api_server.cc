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
#include <stop_token>
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
#include "platform/memory_pressure.h"
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
// A body's buffer grows by doubling from this, charged as it grows.
constexpr std::size_t kFirstBodyBytes = std::size_t{64} << 10U;
// How often a body waiting for the request memory tries again (a check of
// the pool's epoch, not a parse), and how often the parse thread looks.
constexpr auto kMemoryRetry = std::chrono::milliseconds(50);
constexpr auto kParseRetryPoll = std::chrono::milliseconds(50);
// A request that held this much of the request memory gives the C
// library's free heap back to the system when it ends (D-102): a large
// parse leaves many small allocations freed.
constexpr std::uint64_t kTrimAfterBytes = std::uint64_t{16} << 20U;
// Consecutive pauses with no paused member before the cohort is a fault
// (a member's reader may resume between the backend's look and the
// server's: that pass only resumes).
constexpr int kMaxEmptyPauses = 16;

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

  std::string out;              // bytes the I/O thread has not taken yet
  std::string body;             // a non-streaming body, kept apart so it is moved, never copied
  MemoryCharge response;        // `body`'s allocation, which follows it to the socket
  MemoryCharge unread;          // `out`'s allocation, as a stream's output grows
  Clock::time_point paused_at;  // when `paused` was last set
  bool head_sent = false;       // the head is in `out` or gone out
  bool chunked = false;         // the body is framed in chunks
  bool keep_alive = false;      // the connection serves another request after this
  bool ended = false;           // the response is whole
  bool gone = false;            // the client left (or was dropped): the generation ends
  bool stalled = false;         // the watchdog answered it: the generation ends
  bool paused = false;          // more unread than stream_buffer: the request waits (backpressure)
  bool queued = false;          // waiting in Server::queue_
  bool parked = false;          // yielded and waiting for its client to read (Server::parked_)
  bool dirty = false;           // on Server::dirty_
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
      const std::string json = ErrorJson(error);
      if (error.status == 429 || error.status == 503) {
        extra.push_back(std::format("Retry-After: {}", kRetryAfterSeconds));
        extra.emplace_back("x-should-retry: true");
      }
      PutHead(error.status, kJson, json.size(), std::move(extra));
      PutBody(json);
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
  MemoryCharge body;  // `in`'s allocation while it holds a body (the request memory)
  // The body's buffer could not grow now: the connection is not read until
  // the driver grows the request memory, or refuses (its denials past
  // `wait_denials`).
  bool memory_wait = false;
  std::uint64_t wait_denials = 0;
  Clock::time_point idle_by;
  Clock::time_point inactive_by;  // a head or body: when no byte arriving gets a 408
  Clock::time_point linger_by;
  Clock::time_point progress_at;  // the last write that went out, or when output became pending
  std::string wbuf;
  std::size_t woff = 0;
  MemoryCharge response;  // a non-streaming body's allocation, in `wbuf`
  std::shared_ptr<Channel> channel;
  bool close_after = false;   // pipelined, refused mid-request, or input closed: no reuse
  bool input_closed = false;  // the peer shut its sending side after a whole request
  std::uint32_t interest = 0;
  std::uint64_t last_active = 0;
  std::size_t lingered = 0;
  bool dead = false;

  bool pending_output() const { return woff < wbuf.size(); }
};

// ---------------------------------------------------------------- Parse

// A body to parse, on the I/O thread or (kParseOffThreadBytes and longer)
// the parse thread, and what came of it.
struct Server::Parse {
  std::uint64_t connection = 0;
  std::shared_ptr<Channel> channel;
  bool literal_route = false;
  std::string body;
  MemoryCharge body_charge;  // the body's buffer
  std::expected<ChatRequest, Error> parsed;
  std::optional<CompletionRequest> literal;
  MemoryCharge request_charge;  // the parsed request's bytes
  std::uint64_t denials = 0;    // the request memory's when it began to wait
  bool large = false;           // its body was large (the heap is trimmed after it)

  // Parses, charging the parse's working set as it goes, then the parsed
  // request; frees the body. False, the body kept, when the request memory
  // cannot take the charges now but its driver may grow it (`may_wait`):
  // the caller waits for it and runs the parse again.
  bool Run(RequestMemory& memory, bool may_wait) {
    if (literal_route) {
      auto raw = ParseCompletionRequest(body, &memory);
      if (raw) {
        parsed = raw->options;
        literal = std::move(*raw);
      } else {
        parsed = std::unexpected(raw.error());
      }
    } else {
      parsed = ParseChatRequest(body, &memory);
    }
    const auto busy = [&] {
      return !parsed && parsed.error().code == "request_memory_busy" && may_wait && memory.grows();
    };
    if (parsed) {
      const std::uint64_t bytes = literal ? RequestBytes(*literal) : RequestBytes(*parsed);
      if (!request_charge.Add(memory, bytes)) {
        parsed = std::unexpected(MemoryRefusal(memory, bytes, "the parsed request"));
        literal.reset();
      }
    }
    if (busy()) {
      literal.reset();
      return false;
    }
    large = body.size() >= kTrimAfterBytes;
    std::string().swap(body);
    body_charge.Reset();
    if (large) {
      platform::ReleaseFreeHeap();  // the parse's many small allocations, freed
    }
    return true;
  }
};

// ---------------------------------------------------------------- Stream

// One admitted request's response as the driver makes it.
class Server::Stream final : public Exchange {
 public:
  // `started`: when the driver took the request up, which a non-streaming
  // deadline counts from (a request that yielded its place: when it was
  // first taken up, its response's progress carried on from `pending`).
  Stream(Server& server, Pending& pending, int wake_fd, const std::function<bool()>& on_wake,
         Clock::time_point started, bool cooperative = false)
      : server_(server),
        pending_(pending),
        channel_(*pending.channel),
        wake_fd_(wake_fd),
        on_wake_(on_wake),
        started_(pending.started.value_or(started)),
        text_(pending.text ? std::move(*pending.text) : OutputText(pending.request.stop)),
        usage_(pending.usage),
        reasoning_(std::move(pending.reasoning)),
        content_(std::move(pending.content)),
        text_charge_(std::move(pending.text_charge)),
        pauses_(pending.pauses),
        cooperative_(cooperative) {
    pending.text.reset();
    if (!pending_.request.stream && server_.options_.deadline_cap) {
      // Set before any work, so a configured deadline ends a long rendering
      // too; Admit then scales it to the request's work, within the cap.
      deadline_ = started_ + *server_.options_.deadline_cap;
    }
  }

  bool Admit(const Admission& admission) override {
    usage_.prompt_tokens = admission.prompt_tokens;
    floors_ = admission.floors;
    if (!pending_.request.stream && server_.options_.deadline_cap) {
      // A stream has no deadline, nor by default a non-streaming request
      // (D-102); with a cap configured, its work's, at most the cap.
      deadline_ = started_ + ScaledDeadline(server_.options_.stall, *server_.options_.deadline_cap,
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
    return Backpressure() && Check(std::nullopt, 0);
  }
  bool Content(std::string_view text) override {
    Emit(text_.Content(text));
    if (text_.stopped()) {
      Progress(std::nullopt, 0);  // the answer is whole: nothing else to ask
      return false;
    }
    return Backpressure() && Check(std::nullopt, 0);
  }
  bool Continue() override { return Backpressure() && Check(std::nullopt, 0); }
  bool Paused() const override {
    const std::scoped_lock lock(server_.mutex_);
    return PausedLocked();
  }
  bool Yielding() const override { return yielding_; }
  // A cooperative member that yields (Work::Yield).
  void SetYielding() { yielding_ = true; }

  // A yielded request's response so far, for its next Stream (Requeue, or
  // a deferral back to the queue); `taken_up`: it ran, so its next Stream
  // counts from when it was first taken up.
  void Carry(Pending& pending, bool taken_up) {
    pending.text = std::move(text_);
    pending.usage = usage_;
    pending.reasoning = std::move(reasoning_);
    pending.content = std::move(content_);
    pending.text_charge = std::move(text_charge_);
    pending.pauses = pauses_;
    if (taken_up) {
      pending.started = started_;
    }
  }

  // Hands the rest of the response to the I/O thread for the backend's
  // result; the status for the log.
  int End(const std::expected<Completion, Error>& result) {
    {
      const std::scoped_lock lock(server_.mutex_);
      if (channel_.stalled || channel_.gone) {
        return channel_.stalled ? 504 : 499;
      }
    }
    // A non-streaming body (large literal score arrays among them) is the
    // driver's: made without the shared I/O lock, so another connection can
    // still disconnect or read, and charged to the request memory by its
    // allocation, which follows it to the socket.
    std::optional<std::expected<std::string, Error>> body;
    MemoryCharge body_charge;
    if (!pending_.request.stream && result && !timed_out_ && !stopping_) {
      EmitLocked(text_.Finish());  // non-streaming: only local strings
      usage_.completion_tokens = result->completion_tokens;
      usage_.cached_tokens = result->cached_tokens;
      const Finish finish = result->stopped || text_.stopped() ? Finish::kStop : Finish::kLength;
      // The body's most: what the request memory could hold besides it.
      const std::uint64_t most = server_.memory_->capacity();
      body = pending_.literal
                 ? LiteralCompletionJson(channel_.id, channel_.created, *pending_.literal, content_,
                                         result->literal, finish, usage_,
                                         static_cast<std::size_t>(most))
                 : std::expected<std::string, Error>(CompletionJson(
                       channel_.id, channel_.created, pending_.request.model, content_,
                       reasoning_.empty() ? std::nullopt : std::optional<std::string>(reasoning_),
                       finish, usage_));
      std::string().swap(content_);
      std::string().swap(reasoning_);
      text_charge_.Reset();
      // Charged as it goes to the socket (on the driver, the request memory
      // grows for it); one that cannot be held beside what slow readers
      // keep is refused (503), so they cannot retain responses without
      // bound.
      if (*body && !body_charge.Add(*server_.memory_, (*body)->capacity())) {
        body = std::unexpected(
            MemoryRefusal(*server_.memory_, (*body)->capacity(), "the completed response"));
      }
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
        if (pending_.request.stream) {
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
          } else if (!body || !*body) {
            const Error error =
                body ? body->error() : Refusal(500, "the response was not made", "internal");
            channel_.PutError(error);
            status = error.status;
          } else {
            channel_.PutHead(200, kJson, (*body)->size());
            channel_.response = std::move(body_charge);
            channel_.body = std::move(**body);
            channel_.PutEnd();
          }
        }
      }
      Dirty();
    }
    server_.WakeIo();
    return status;
  }

  const Usage& usage() const { return usage_; }
  // How often the request paused for its client (backpressure).
  std::uint64_t pauses() const { return pauses_; }

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
      // The text so far, charged by its allocation as it grows (bounded by
      // the context's tokens). Already held: past the grant it is charged
      // anyway, for the driver to grow it, and later charges wait.
      const std::uint64_t held = reasoning_.capacity() + content_.capacity();
      if (held > text_charge_.bytes()) {
        text_charge_.Force(*server_.memory_, held - text_charge_.bytes());
      }
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
    // The unread output, charged as it grows (within the grant: this is
    // the middle of a unit, where the driver does not reclaim). What the
    // request memory cannot take now is still held, charged past the grant
    // for the driver to grow it, and pauses the request like a reader
    // behind.
    bool over = false;
    if (const std::uint64_t held = channel_.out.capacity(); held > channel_.unread.bytes()) {
      const std::uint64_t more = held - channel_.unread.bytes();
      if (!channel_.unread.Add(*server_.memory_, more, true)) {
        channel_.unread.Force(*server_.memory_, more);
        over = true;
      }
    }
    if (over || channel_.out.size() > server_.options_.intake.stream_buffer) {
      // A reader this far behind gets backpressure (D-102): the request
      // pauses at this completed step until the I/O thread has handed the
      // buffer to the socket (Server::Flush clears it), instead of the
      // buffer growing or the client being dropped.
      if (!channel_.paused) {
        channel_.paused = true;
        channel_.paused_at = Clock::now();
        ++pauses_;
      }
    }
    Dirty();
  }

  // With the lock held: paused, and not ended otherwise.
  bool PausedLocked() const { return channel_.paused && !channel_.gone && !channel_.stalled; }

  // The serial path waits here, at a completed step, while its client is
  // behind; a cooperative backend leaves the request out of its units
  // instead (Paused). False once the runtime stops, or when the request is
  // to yield its place: queued requests waited behind its pause for
  // ServerOptions::yield_after (Yielding).
  bool Backpressure() {
    if (yielding_) {
      return false;
    }
    if (cooperative_ || !pending_.request.stream) {
      return true;
    }
    {
      const std::scoped_lock lock(server_.mutex_);
      if (!PausedLocked()) {
        return true;
      }
    }
    const bool go_on = server_.AwaitReaders(
        [this] { return PausedLocked() && !server_.YieldDueLocked(channel_, Clock::now()); },
        wake_fd_, on_wake_);
    if (!go_on) {
      stopping_ = true;
      return false;
    }
    const std::scoped_lock lock(server_.mutex_);
    if (PausedLocked() && server_.YieldDueLocked(channel_, Clock::now())) {
      yielding_ = true;
      return false;
    }
    return true;
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
    if (!cooperative_) {
      // Between a serial request's steps the driver settles the request
      // memory (a cohort's settles between its units, Backend::Maintain),
      // so bodies and parses waiting for it do not wait for the request.
      server_.memory_->Settle();
    }
    if (timed_out_ || stopping_ || yielding_) {
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
  std::string reasoning_;  // a non-streaming response's text so far
  std::string content_;
  MemoryCharge text_charge_;  // theirs, in the request memory
  bool timed_out_ = false;
  bool stopping_ = false;
  bool yielding_ = false;
  std::uint64_t pauses_ = 0;  // under the server's lock
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
  bool model_paused = false;

  Active(Server& server, Pending request, int wake_fd, const std::function<bool()>& on_wake)
      : pending(std::move(request)),
        request{.options = pending.request,
                .literal = pending.literal ? &*pending.literal : nullptr,
                .resume = pending.resume.get()},
        started(pending.started.value_or(Clock::now())),
        exchange(server, pending, wake_fd, on_wake, started, true) {}
};

// ---------------------------------------------------------------- Server

Server::Server(Backend& backend, ServerOptions options)
    : backend_(backend),
      options_(std::move(options)),
      memory_(options_.memory != nullptr
                  ? options_.memory
                  : std::make_shared<RequestMemory>(options_.intake.request_capacity,
                                                    options_.intake.request_capacity)),
      models_(backend.Models()),
      created_(static_cast<std::int64_t>(std::time(nullptr))),
      loop_(platform::EventLoop::Open()),
      stop_(platform::Waker::Open()),
      io_wake_(platform::Waker::Open()),
      ready_(platform::Waker::Open()),
      watchdog_(options_.stall, Clock::now()) {
  if (options_.ladder != nullptr) {
    ladder_ = options_.ladder;
    return;
  }
  // A ladder of the server's own (the tests'): `hang`, or its default, and
  // `on_hang` its last resort.
  own_ladder_ = std::make_unique<HangLadder>(options_.hang.value_or(DefaultHang(options_.stall)),
                                             Clock::now());
  own_ladder_->set_log([this](std::string_view line) { Log(line); });
  if (options_.on_hang) {
    own_ladder_->set_last_resort(options_.on_hang);
  }
  ladder_ = own_ladder_.get();
}

std::chrono::milliseconds DefaultHang(std::chrono::milliseconds stall) {
  return std::max<std::chrono::milliseconds>(kHangFloor, kHangStalls * stall);
}

Server::~Server() {
  if (io_.joinable()) {
    stop_.Signal();
    io_.join();
  }
  if (parser_.joinable()) {
    parser_.request_stop();
    parser_.join();
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
  // The hang ladder (hang_ladder.h) sees the same beat: the unit that
  // follows and the allowance the watchdog gives it.
  ladder_->Unit(next != Phase::kIdle && next != Phase::kPaused,
                Allowance(watchdog_.stall(), expected), PhaseName(next), now);
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
    if (!options_.stall_fails) {
      // Detection is not a limit (D-102): reported, while the requests go
      // on (a hang that never returns is the node's to end).
      Log(std::format(
          "the model backend made no progress for {:.1f} s ({}; stall {}); it is reported "
          "unhealthy until it makes progress; its {} active and {} queued requests go on",
          Seconds(idle), PhaseName(health.phase), health.stalls, running_.size(), queue_.size()));
    } else {
      FailStalledLocked(idle, ended);
    }
  }
  HealthChanged();
}

void Server::WatchHang(Clock::time_point now) {
  // Progress of any kind: a unit's beat (the ladder sees each, BeatLocked)
  // or the backend's own activity moving (the node's progress count). The
  // ladder logs each rung it enters; entering rung 3 runs its last resort
  // here, on the I/O thread, which never waits on the model.
  if (options_.activity) {
    ladder_->Activity(options_.activity(), now);
  }
  (void)ladder_->Check(now);
}

void Server::FailStalledLocked(Clock::duration idle, std::vector<std::uint64_t>& ended) {
  const Health& health = watchdog_.health();
  {
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
    // Nothing queued (or parked) would run before the backend moves again.
    for (Pending& p : parked_) {
      p.channel->parked = false;
      queue_.push_back(std::move(p));
    }
    parked_.clear();
    for (Pending& p : queue_) {
      Channel& ch = *p.channel;
      ch.queued = false;
      Log(std::format("request {}: 503, the backend is not making progress", ch.id));
      ch.PutError(Unresponsive());
      ended.push_back(ch.connection);
    }
    queue_.clear();
  }
  drained_.notify_all();  // a paused request ends too
}

void Server::Watch(Connection& c) {
  if (c.dead) {
    return;
  }
  // A closed input side reads as ready forever: it is watched no longer,
  // except while lingering, which reads to the end and then drops. Nor
  // while a body waits, unread, for the request memory: the epoll watch is
  // level-triggered, so a peer that shut its sending side would wake the
  // loop on every pass until the body is read (a reset is still an error
  // event); what it sent is read, and its close seen, once the body fits.
  std::uint32_t want = (c.input_closed || c.memory_wait) ? 0U : platform::kPeerClosed;
  if (c.state == Connection::State::kLinger ||
      (!c.input_closed && !c.memory_wait &&
       (c.state != Connection::State::kBusy || !c.close_after))) {
    want |= platform::kReadable;  // not while its body waits for the request memory
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
      if (c.channel->paused) {
        drained_.notify_all();  // its request no longer waits for it
      }
      if (c.channel->queued) {
        c.channel->queued = false;
        std::erase_if(queue_, [&](const Pending& p) { return p.channel == c.channel; });
      }
      if (c.channel->parked) {
        c.channel->parked = false;
        std::erase_if(parked_, [&](const Pending& p) { return p.channel == c.channel; });
      }
      Release(c.channel->body);
      c.channel->response.Reset();
      std::string().swap(c.channel->out);
      c.channel->unread.Reset();
    }
    Release(c.wbuf);
    c.response.Reset();
  }
  Release(c.in);
  c.body.Reset();
  c.memory_wait = false;
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
    Release(c.in);
    c.body.Reset();
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
        c.response.Reset();
        if (c.channel) {
          if (!c.channel->out.empty()) {
            // Its allocation's charge follows it to the socket.
            c.wbuf.swap(c.channel->out);
            c.response = std::move(c.channel->unread);
            if (c.channel->paused) {
              // The client took what was sent before: its request goes on,
              // with up to stream_buffer more to fill before it waits again
              // (a request that yielded its place is queued again).
              c.channel->paused = false;
              drained_.notify_all();
              UnparkLocked(*c.channel);
            }
          } else if (!c.channel->body.empty()) {
            c.wbuf.swap(c.channel->body);
            c.response = std::move(c.channel->response);
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
            c.body.Reset();  // a refused request's body, or one no route read
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
                            .max_body_bytes = static_cast<std::size_t>(options_.intake.max_body)};
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
      // Room for it in the request memory first, or the connection waits,
      // unread, until the driver grows the memory (or refuses).
      if (const Growth grown = GrowBody(c, most); grown != Growth::kOk) {
        return;
      }
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
      c.state = State::kHead;
    }
    // An inactivity timeout (D-102): a request that keeps sending, however
    // slowly, is never cut off; one that sends nothing for this long is.
    c.inactive_by = Clock::now() + options_.request_inactivity;
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
      c.state = State::kBody;
      // The body is charged to the request memory as it arrives, not as
      // declared (D-102): a client that declares a large body and sends
      // little holds little. What arrived with the head is charged now; a
      // body the memory cannot take now waits (RetryBody), and is not
      // asked to continue until it can.
      if (c.request.body_bytes > 0) {
        if (const Growth grown = GrowBody(c, 0); grown != Growth::kOk) {
          return;
        }
      }
      if (c.request.expect_continue && c.in.size() < c.head_end + c.request.body_bytes) {
        c.request.expect_continue = false;  // answered once, not again after a memory wait
        c.wbuf.append(kContinue);
        c.progress_at = Clock::now();
        Flush(c);
        if (c.dead) {
          return;
        }
      }
    }
    if (c.state == State::kBody && c.in.size() >= c.head_end + c.request.body_bytes) {
      BodyArrived(c);
      return;
    }
  }
}

void Server::BodyArrived(Connection& c) {
  http::Request request = std::move(c.request);
  if (c.in.size() > c.head_end + request.body_bytes) {
    c.close_after = true;  // bytes after the body: a pipelined request
  }
  // The body's bytes are the request's now, not the connection's: moved,
  // not copied, since a body may be large (D-102).
  c.in.erase(0, c.head_end);
  c.in.resize(request.body_bytes);
  request.body = std::move(c.in);
  c.in = std::string();
  c.request = {};
  c.request.http10 = request.http10;
  OnRequest(c, std::move(request));
}

void Server::OnRequest(Connection& c, http::Request request) {
  // The body's charge is the request's now: handed to its parse, or
  // released with the body when another route answers or refuses it.
  MemoryCharge body_charge = std::move(c.body);
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
  auto parse = std::make_unique<Parse>();
  parse->connection = c.id;
  parse->channel = channel;
  parse->literal_route = literal_route;
  parse->body = std::move(request.body);
  parse->body_charge = std::move(body_charge);
  // A large body is parsed on the parse thread, so this thread goes on
  // serving every other connection (TakeParsed hands it back); so is one
  // the request memory cannot take now, which waits there for it.
  if (parse->body.size() < kParseOffThreadBytes && parse->Run(*memory_, true)) {
    OnParsed(c, *parse);
    return;
  }
  {
    const std::scoped_lock lock(parse_mutex_);
    parse_in_.push_back(std::move(parse));
  }
  parse_ready_.notify_one();
}

void Server::TakeParsed() {
  std::vector<std::unique_ptr<Parse>> done;
  {
    const std::scoped_lock lock(parse_mutex_);
    done.swap(parse_out_);
  }
  bool large = false;
  for (const std::unique_ptr<Parse>& parse : done) {
    large = large || parse->large;
    // The connection may have gone (or moved on) while its body parsed.
    if (auto it = connections_.find(parse->connection);
        it != connections_.end() && !it->second->dead && it->second->channel == parse->channel) {
      OnParsed(*it->second, *parse);
    }
  }
  done.clear();
  if (large) {
    // A large request refused here (an unknown model, a prompt that cannot
    // fit) leaves its many small allocations freed: back to the system.
    platform::ReleaseFreeHeap();
  }
}

void Server::ParseLoop(const std::stop_token& stop) {
  if (auto stack = platform::InstallThreadSignalStack(); !stack) {
    Log("the chat route's parse thread has no signal stack: " + stack.error());
  }
  // Parses the request memory cannot take now are set aside, so the bodies
  // behind them go on, and tried again only when what may let them fit
  // moves (RequestMemory::epoch: the grant grown or shrunk, a denial, room
  // now); one the driver could not grow the memory for is refused (503).
  std::deque<std::unique_ptr<Parse>> waiting;
  std::uint64_t seen = memory_->epoch();
  const auto hand_back = [this](std::unique_ptr<Parse> parse) {
    {
      const std::scoped_lock lock(parse_mutex_);
      parse_out_.push_back(std::move(parse));
    }
    WakeIo();
  };
  for (;;) {
    std::unique_ptr<Parse> parse;
    {
      std::unique_lock lock(parse_mutex_);
      const auto ready = [this] { return !parse_in_.empty(); };
      if (waiting.empty()) {
        if (!parse_ready_.wait(lock, stop, ready)) {
          return;  // stopping
        }
      } else {
        (void)parse_ready_.wait_for(lock, stop, kParseRetryPoll, ready);
        if (stop.stop_requested()) {
          return;
        }
      }
      if (!parse_in_.empty()) {
        parse = std::move(parse_in_.front());
        parse_in_.pop_front();
      }
    }
    if (parse != nullptr) {
      parse->denials = memory_->denials();
      if (parse->Run(*memory_, true)) {
        hand_back(std::move(parse));
      } else {
        waiting.push_back(std::move(parse));
      }
    }
    if (const std::uint64_t epoch = memory_->epoch(); epoch != seen && !waiting.empty()) {
      seen = epoch;
      const std::uint64_t denials = memory_->denials();
      for (auto it = waiting.begin(); it != waiting.end();) {
        if ((*it)->Run(*memory_, denials == (*it)->denials)) {
          hand_back(std::move(*it));
          it = waiting.erase(it);
        } else {
          ++it;
        }
      }
    } else if (waiting.empty()) {
      seen = memory_->epoch();
    }
  }
}

Server::Growth Server::GrowBody(Connection& c, std::size_t more) {
  const std::size_t whole = c.head_end + c.request.body_bytes;
  const std::size_t need = c.in.size() + more;
  std::size_t target = c.in.capacity();
  if (need > target) {
    target = std::min(whole, std::max({need, 2 * c.in.capacity(), kFirstBodyBytes}));
  }
  if (target > c.body.bytes() && !c.body.Add(*memory_, target - c.body.bytes())) {
    if (target > memory_->capacity() || !memory_->grows()) {
      c.memory_wait = false;  // refused: no longer waiting
      Refuse(c, MemoryRefusal(*memory_, target, "the request body"));
      return Growth::kRefused;
    }
    // Not now: the connection is not read until the driver grows the
    // request memory (RetryBody), its inactivity clock held meanwhile.
    if (!c.memory_wait) {
      c.memory_wait = true;
      c.wait_denials = memory_->denials();
      Watch(c);
    }
    return Growth::kWait;
  }
  if (target > c.in.capacity()) {
    c.in.reserve(target);
    // The library may round an allocation up: charged as it is.
    if (c.in.capacity() > c.body.bytes()) {
      c.body.Force(*memory_, c.in.capacity() - c.body.bytes());
    }
  }
  return Growth::kOk;
}

void Server::RetryBody(Connection& c) {
  if (c.dead || !c.memory_wait) {
    return;
  }
  c.inactive_by = Clock::now() + options_.request_inactivity;
  const std::size_t remaining = c.head_end + c.request.body_bytes - c.in.size();
  const bool denied = memory_->denials() != c.wait_denials;
  switch (GrowBody(c, std::min(kReadChunk, remaining))) {
    case Growth::kRefused:
      return;
    case Growth::kWait:
      if (denied) {
        // The driver could not grow the request memory for it: refused now
        // (503, to retry), not left waiting.
        c.memory_wait = false;
        Refuse(c, MemoryRefusal(*memory_, c.body.bytes() + std::min(kReadChunk, remaining),
                                "the request body"));
      }
      return;
    case Growth::kOk:
      break;
  }
  c.memory_wait = false;
  if (remaining == 0) {
    Watch(c);
    BodyArrived(c);  // it had all arrived with its head
    return;
  }
  if (c.request.expect_continue) {
    // Only when the head's wait kept it from being sent: a body that waited
    // mid-way was already told to continue.
    c.request.expect_continue = false;
    c.wbuf.append(kContinue);
    c.progress_at = Clock::now();
    Flush(c);
    if (c.dead) {
      return;
    }
  }
  Watch(c);
  OnReadable(c);
}

void Server::OnParsed(Connection& c, Parse& parse) {
  const std::shared_ptr<Channel>& channel = parse.channel;
  const bool literal_route = parse.literal_route;
  std::expected<ChatRequest, Error>& parsed = parse.parsed;
  std::optional<CompletionRequest>& literal = parse.literal;
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
  // A prompt that cannot fit the model's context is refused before it is
  // queued (D-102): text longer than any prompt that fits could render to
  // (the model checks the same before rendering), or more token IDs than
  // the context holds. It would otherwise hold the request memory while it
  // waits, only to be refused.
  std::uint64_t text = 0;
  for (const Message& m : parsed->messages) {
    text += m.content.size() + (m.reasoning ? m.reasoning->size() : 0);
  }
  if (literal && literal->prompt) {
    text += literal->prompt->size();
  }
  if ((model->render_bytes != 0 && text > model->render_bytes) ||
      (literal && model->context != 0 && literal->token_ids.size() > model->context)) {
    Error error = Refusal(
        400,
        std::format("This model's maximum context length is {} tokens. However, your {} resulted "
                    "in more than {} tokens. Please reduce the length of the {}.",
                    model->context, literal ? "prompt" : "messages", model->context,
                    literal ? "prompt" : "messages"),
        "context_length_exceeded");
    error.param = literal ? "prompt" : "messages";
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
  bool no_room = false;
  {
    const std::scoped_lock lock(mutex_);
    stopping = stopping_;
    // Unhealthy refuses only when the owner asked stalls to fail requests;
    // by default a stall is reported and requests still queue (D-102).
    unhealthy = options_.stall_fails && !watchdog_.health().healthy;
    // Queued requests hold at most all but a quarter of the request
    // memory's current grant (not what it could grow to: growth needs the
    // reclaim order to free room, and queued requests must not hold room
    // a running conversation's state may need), leaving the driver
    // headroom for the request it takes up (its rendering and tokenization
    // are charged then). The headroom is not a promise: growth beyond the
    // grant still needs the reclaim order.
    std::uint64_t queued = parse.request_charge.bytes();
    for (const Pending& p : queue_) {
      queued += p.charge.bytes();
    }
    const std::uint64_t grant = std::max(memory_->grant(), memory_->used());
    no_room = queued > grant - (grant / kQueueHeadroomShare);
    if (!stopping && !unhealthy && !no_room &&
        (!options_.max_queued || queue_.size() < *options_.max_queued)) {
      channel->queued = true;
      Pending pending;
      pending.channel = channel;
      pending.request = std::move(*parsed);
      pending.literal = std::move(literal);
      pending.charge = std::move(parse.request_charge);
      queue_.push_back(std::move(pending));
      ++enqueued_;
      drained_.notify_all();  // a cohort waiting for its readers may admit it
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
  if (no_room) {
    Log(std::format("request {}: 503, the queued requests hold the request memory", channel->id));
    Refuse(c,
           Refusal(503,
                   "the queued requests hold all the request memory but the driver's headroom "
                   "([client] request_memory_bytes); retry",
                   "request_memory_busy"),
           {}, false);
    return;
  }
  const std::size_t most = options_.max_queued.value_or(0);
  Log(std::format("request {}: 429, {} requests queued", channel->id, most));
  Refuse(c,
         Refusal(429, std::format("{} requests are already waiting ([client] max_queued); retry "
                                  "later",
                                  most)),
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
  WatchHang(now);
  // The queue: only with queue_wait set does a non-streaming request that
  // waited that long get a 429 (D-102); a stream, held by keepalives,
  // waits its turn.
  {
    const std::scoped_lock lock(mutex_);
    if (const auto due = watchdog_.due()) {
      soon(*due);
    }
    if (const auto wait = options_.queue_wait) {
      std::erase_if(queue_, [&](const Pending& p) {
        Channel& ch = *p.channel;
        if (ch.stream || p.started) {
          // A model-paused or reader-yielded request is already admitted.
          // Returning it to the FIFO cannot turn it into a new 429 refusal.
          return false;
        }
        if (now - ch.queued_at < *wait) {
          soon(ch.queued_at + *wait);
          return false;
        }
        ch.queued = false;
        Log(std::format("request {}: 429 after waiting {} s in the queue", ch.id,
                        std::chrono::duration_cast<std::chrono::seconds>(*wait).count()));
        ch.PutError(Refusal(429,
                            "the request waited longer than [client] queue_wait_seconds behind "
                            "others; retry later"));
        expired.push_back(ch.connection);
        return true;
      });
    }
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
        if (c.memory_wait) {
          // Waiting for the request memory, not for the client: tried again
          // as the driver grows it (or refuses), never timed out meanwhile.
          RetryBody(c);
          soon(now + kMemoryRetry);
          break;
        }
        if (now >= c.inactive_by) {
          Refuse(c, Refusal(408, std::format("no byte of the request's {} arrived for {} s "
                                             "([client] request_inactivity_seconds)",
                                             c.state == State::kHead ? "head" : "body",
                                             std::chrono::duration_cast<std::chrono::seconds>(
                                                 options_.request_inactivity)
                                                 .count())));
        } else {
          soon(c.inactive_by);
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
        if (c.pending_output() && options_.write_inactivity) {
          // Only when configured (D-102): otherwise a client that stops
          // reading holds its own buffer, and a stream's request waits.
          if (now - c.progress_at >= *options_.write_inactivity) {
            Log("a client took no output for [client] write_inactivity_seconds; the connection "
                "is closed");
            Drop(c);
            break;
          }
          soon(c.progress_at + *options_.write_inactivity);
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
              // Pending bytes already keep the stream alive as soon as the
              // client reads. A paused reader must not accumulate comments
              // forever outside the generation's bounded output buffer.
              const bool pending = c.pending_output() || !ch.out.empty();
              if (!pending && now >= ch.last_out + options_.keepalive) {
                ch.PutBody(": keepalive\n\n");
                added = true;
              }
              soon(pending ? now + options_.keepalive : ch.last_out + options_.keepalive);
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
  std::size_t output = 0;
  for (const auto& [id, c] : connections_) {
    if (!c->dead) {
      held += c->in.capacity() + c->wbuf.capacity();
      output += c->wbuf.size() - c->woff;
      const std::scoped_lock lock(mutex_);
      if (c->channel) {
        output += c->channel->out.size() + c->channel->body.size();
      }
    }
  }
  held_bytes_.store(held, std::memory_order_relaxed);
  output_bytes_.store(output, std::memory_order_relaxed);
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
        TakeParsed();
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
    // Whatever is queued now waits for this request alone: a paused stream
    // yields its place to it (YieldDueLocked).
    blocked_ = true;
    recovered = BeatLocked(Phase::kStarting, 0);
  }
  if (recovered) {
    HealthChanged();
  }
  Stream stream(*this, pending, wake_fd, on_wake, started);
  std::expected<Completion, Error> result;
  if (pending.literal) {
    if (const std::shared_ptr<Yielded> from = std::move(pending.resume); from != nullptr) {
      result = backend_.Resume(*pending.literal, stream, *from);
    } else {
      result = backend_.Complete(*pending.literal, stream);
    }
  } else if (const std::shared_ptr<Yielded> from = std::move(pending.resume); from != nullptr) {
    result = backend_.Resume(pending.request, stream, *from);
  } else {
    result = backend_.Complete(pending.request, stream);
  }
  {
    const std::scoped_lock lock(mutex_);
    blocked_ = false;
  }
  if (result && result->yielded != nullptr) {
    Requeue(pending, stream, std::move(result->yielded));
    Progress(Phase::kIdle, 0);
    return;
  }
  FinishResponse(pending, stream, result, started);
  Progress(Phase::kIdle, 0);
}

bool Server::YieldDueLocked(const Channel& channel, Clock::time_point now) const {
  return blocked_ && (!queue_.empty() || suspended_requests_ != 0) && channel.paused &&
         !channel.gone && !channel.stalled && now - channel.paused_at >= options_.yield_after;
}

void Server::Requeue(Pending& pending, Stream& stream, std::shared_ptr<Yielded> yielded) {
  stream.Carry(pending, true);
  pending.resume = std::move(yielded);
  bool recovered = false;
  {
    const std::scoped_lock lock(mutex_);
    std::erase(running_, pending.channel);
    recovered = BeatLocked(Phase::kFinishing, 0);
    Channel& channel = *pending.channel;
    if (channel.gone || channel.stalled) {
      Log(std::format("request {}: {} as it yielded its place", channel.id,
                      channel.gone ? "499, the client left" : "504, the backend stalled"));
    } else {
      // Positions and counts only (D-014).
      Log(
          std::format("request {}: its client has read nothing for {:.1f} s while {} requests "
                      "wait; it yields its place and continues once its client reads",
                      channel.id, Seconds(Clock::now() - channel.paused_at),
                      queue_.size() + suspended_requests_));
      if (channel.paused) {
        // Parked until its client reads (Flush, UnparkLocked): taken up
        // while it still could not send, it would only yield again (after
        // its model was swapped back in).
        channel.parked = true;
        parked_.push_back(std::move(pending));
      } else {
        channel.queued = true;
        queue_.push_back(std::move(pending));
        ++enqueued_;
        ready_.Signal();
      }
    }
  }
  if (recovered) {
    HealthChanged();
  }
}

void Server::UnparkLocked(const Channel& channel) {
  if (!channel.parked) {
    return;
  }
  const auto at =
      std::ranges::find_if(parked_, [&](const Pending& p) { return p.channel.get() == &channel; });
  if (at == parked_.end()) {
    return;
  }
  at->channel->parked = false;
  at->channel->queued = true;
  queue_.push_back(std::move(*at));
  parked_.erase(at);
  ++enqueued_;
  ready_.Signal();
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

bool Server::AwaitReaders(const std::function<bool()>& paused, int wake_fd,
                          const std::function<bool()>& on_wake) {
  Phase before = Phase::kDecode;
  {
    const std::scoped_lock lock(mutex_);
    before = watchdog_.health().phase;
  }
  // A pause is not a stall: the watchdog watches nothing until it ends.
  Progress(Phase::kPaused, 0);
  bool stopped = false;
  {
    std::unique_lock lock(mutex_);
    while (paused() && !stopping_) {
      // The runtime's signals are the driver's to read: woken every 100 ms
      // to look, or at once when the I/O thread hands output on or a client
      // leaves.
      (void)drained_.wait_for(lock, std::chrono::milliseconds(100));
      lock.unlock();
      const bool wake = Readable(wake_fd) && on_wake();
      // The driver waits here, so it settles the request memory here too:
      // a body or a parse waiting for it is not held up by a stuck reader.
      memory_->Settle();
      lock.lock();
      if (wake) {
        stopping_ = true;
      }
    }
    stopped = stopping_;
  }
  Progress(before == Phase::kPaused ? Phase::kDecode : before, 0);
  return !stopped;
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
    why = " (the client left)";
  }
  Log(std::format(
      "request {}: {} {}{}, {} prompt tokens ({} cached), {} generated, {:.3f} s "
      "({:.3f} s queued){}",
      channel.id, pending.request.model, status, why, u.prompt_tokens, u.cached_tokens,
      u.completion_tokens, Seconds(Clock::now() - started), Seconds(started - channel.queued_at),
      stream.pauses() == 0
          ? std::string()
          : std::format(", paused {} times for its client to read", stream.pauses())));
  // A large request's many small allocations, freed with it, go back to
  // the system once it is gone (TrimIfOwed).
  trim_owed_ = trim_owed_ || pending.charge.bytes() >= kTrimAfterBytes;
  backend_.AfterResponse();
}

void Server::TrimIfOwed() {
  if (trim_owed_) {
    trim_owed_ = false;
    platform::ReleaseFreeHeap();
  }
}

std::expected<void, std::string> Server::ServeCooperative(Pending first,
                                                          CooperativeBackend& cooperative,
                                                          int wake_fd,
                                                          const std::function<bool()>& on_wake,
                                                          bool substitute) {
  std::vector<std::unique_ptr<Active>> active;
  active.reserve(kMaxActiveRequests);
  std::deque<Pending> suspended;
  auto turn_since = Clock::now();
  bool switching = false;
  bool substitute_formed = false;
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
    // A yielded request's response so far.
    initial->exchange.Carry(initial->pending, initial->pending.resume != nullptr);
    Serve(initial->pending, wake_fd, on_wake);
    Progress(Phase::kIdle, 0);
    return {};
  }
  const auto retire = [&](bool stop) {
    std::erase_if(active, [&](const std::unique_ptr<Active>& frame) {
      // A member yielding its place (Work::Yield) ends at its completed
      // step; it is not cancelled unless the runtime stops or fails.
      const bool yielding = frame->exchange.Yielding();
      const bool go_on = frame->exchange.Continue();
      if ((!go_on && !yielding) || stop || !fatal.empty()) {
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
      if (retired.result && retired.result->yielded != nullptr) {
        if (frame->model_paused) {
          frame->exchange.Carry(frame->pending, true);
          frame->pending.resume = std::move(retired.result->yielded);
          {
            const std::scoped_lock lock(mutex_);
            std::erase(running_, frame->pending.channel);
            frame->pending.channel->queued = true;
            ++suspended_requests_;
          }
          suspended.push_back(std::move(frame->pending));
          return true;
        }
        Requeue(frame->pending, frame->exchange, std::move(retired.result->yielded));
        return true;
      }
      FinishResponse(frame->pending, frame->exchange, retired.result, frame->started);
      return true;
    });
    TrimIfOwed();
  };
  int empty_pauses = 0;  // consecutive kPaused units with no member paused (M1's race)
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
    bool deferred = false;  // the backend could not take the head now
    bool other_model = false;
    {
      const std::scoped_lock lock(mutex_);
      other_model =
          !queue_.empty() && queue_.front().request.model != active.front()->pending.request.model;
    }
    // Look even when every slot is occupied: an incompatible head must
    // neither refill the old cohort nor wait for every response to end.
    drain = drain || other_model || (substitute && substitute_formed);
    if (!substitute && other_model && Clock::now() - turn_since >= options_.model_turn) {
      switching = true;
    }
    if (switching) {
      for (const auto& frame : active) {
        if (!frame->exchange.Yielding() && frame->work->YieldForSwitch()) {
          frame->model_paused = true;
          frame->exchange.SetYielding();
        }
      }
      retire(false);
      if (active.empty()) {
        break;
      }
    }
    while (!drain && active.size() < kMaxActiveRequests && attempts < kMaxActiveRequests) {
      std::optional<Pending> pending;
      {
        const std::scoped_lock lock(mutex_);
        if (stopping_ || queue_.empty()) {
          break;
        }
        Pending& head = queue_.front();
        if (head.request.model != active.front()->pending.request.model ||
            !cooperative.Supports({.options = head.request,
                                   .literal = head.literal ? &*head.literal : nullptr,
                                   .resume = head.resume.get()})) {
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
        // Restored to the queue's head as it came: its carried response
        // (a yielded request's) goes back with it.
        frame->exchange.Carry(frame->pending, frame->pending.resume != nullptr);
        const std::scoped_lock lock(mutex_);
        frame->pending.channel->queued = true;
        queue_.push_front(std::move(frame->pending));
        deferred = true;
        break;
      }
    }
    // The substitute may batch the compatible requests already ready at
    // its first admission pass. Later arrivals wait until the suspended
    // cohort has resumed, so continuous refill cannot starve it.
    substitute_formed = true;
    // Backpressure that blocks others (D-102): the queue waits on this
    // cohort (it must drain for another model, or is full) while a member
    // waits for its client. Such a member yields its place once paused for
    // yield_after (Work::Yield); one that cannot yet goes on.
    std::vector<Active*> yielding;
    {
      const std::scoped_lock lock(mutex_);
      blocked_ =
          drain || deferred || active.size() >= kMaxActiveRequests || suspended_requests_ != 0;
      const auto now = Clock::now();
      for (const auto& frame : active) {
        if (!frame->exchange.Yielding() && YieldDueLocked(*frame->pending.channel, now)) {
          yielding.push_back(frame.get());
        }
      }
    }
    bool yielded = false;
    for (Active* frame : yielding) {
      if (frame->work->Yield()) {
        frame->exchange.SetYielding();
        yielded = true;
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
    if (stop || !fatal.empty() || yielded) {
      continue;  // a member yielding its place retires first
    }

    // Between completed units: the backend's housekeeping (it leaves every
    // active Work's resources alone).
    backend_.Maintain();
    std::array<CooperativeBackend::Work*, kMaxActiveRequests> work{};
    for (std::size_t i = 0; i < active.size(); ++i) {
      work[i] = active[i]->work.get();
    }
    const std::span<CooperativeBackend::Work* const> wave(work.data(), active.size());
    // The queue as the unit is chosen: a request that arrives after it ends
    // a pause's wait (below), even one that arrives while it is chosen.
    std::uint64_t seen = 0;
    {
      const std::scoped_lock lock(mutex_);
      seen = enqueued_;
    }
    auto next = cooperative.NextUnit(wave);
    if (!next) {
      fatal = next.error().empty() ? "no cooperative backend unit" : next.error();
      continue;
    }
    if (next->phase == Phase::kPaused) {
      // Every runnable request waits for its client to read (backpressure,
      // D-102): wait until one of them reads or leaves, a request arrives
      // (it may join), one is due to yield its place, or the runtime stops;
      // the next pass then retires, admits, yields or advances.
      std::vector<const Channel*> waiting;
      Clock::time_point yield_due =
          !substitute && other_model ? turn_since + options_.model_turn : Clock::time_point::max();
      {
        const std::scoped_lock lock(mutex_);
        for (const auto& frame : active) {
          const Channel& ch = *frame->pending.channel;
          if (ch.paused && !ch.gone && !ch.stalled) {
            waiting.push_back(&ch);
            if (blocked_ && (!queue_.empty() || suspended_requests_ != 0)) {
              yield_due = std::min(yield_due, ch.paused_at + options_.yield_after);
            }
          }
        }
      }
      if (waiting.empty()) {
        // A reader caught up (or left) between the backend's look and this
        // one: only resume. The same with nothing paused, pass after pass,
        // is the backend's fault.
        if (++empty_pauses > kMaxEmptyPauses) {
          fatal = "the cooperative backend keeps pausing while no request waits for its client";
        }
        continue;
      }
      empty_pauses = 0;
      std::ignore = AwaitReaders(
          [this, &waiting, seen, yield_due] {
            return enqueued_ == seen && Clock::now() < yield_due &&
                   std::ranges::all_of(
                       waiting,
                       [](const Channel* ch) { return ch->paused && !ch->gone && !ch->stalled; });
          },
          wake_fd, on_wake);
      continue;
    }
    empty_pauses = 0;
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
    if (next->phase == Phase::kSwap) {
      turn_since = Clock::now();  // residency cost is not useful work in the turn
    }
  }
  if (!suspended.empty()) {
    std::optional<Pending> replacement;
    {
      const std::scoped_lock lock(mutex_);
      // The different-model head may have left while native retirement
      // waited. A fresh same-model request must wait for the originals'
      // held slots, rather than falling back to a serial clear of one.
      if (!stopping_ && fatal.empty() && !queue_.empty() &&
          queue_.front().request.model != suspended.front().request.model) {
        replacement.emplace(std::move(queue_.front()));
        replacement->channel->queued = false;
        queue_.pop_front();
      }
    }
    if (replacement) {
      if (cooperative.Supports({.options = replacement->request,
                                .literal = replacement->literal ? &*replacement->literal : nullptr,
                                .resume = replacement->resume.get()})) {
        auto ran = ServeCooperative(std::move(*replacement), cooperative, wake_fd, on_wake, true);
        if (!ran) {
          fatal = ran.error();
        }
      } else {
        Serve(*replacement, wake_fd, on_wake);
      }
    }
    // Restore the whole cohort ahead of every later arrival, including
    // when the substitute failed or shutdown arrived. Run's stop path
    // then ends queued continuations normally. Reverse insertion retains
    // admission order and carried output/usage for each member.
    {
      const std::scoped_lock lock(mutex_);
      while (!suspended.empty()) {
        queue_.push_front(std::move(suspended.back()));
        suspended.pop_back();
        --suspended_requests_;
        ++enqueued_;
      }
      ready_.Signal();
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
  parser_ = std::jthread([this](const std::stop_token& stop) { ParseLoop(stop); });
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
          cooperative->Supports({.options = next->request,
                                 .literal = next->literal ? &*next->literal : nullptr,
                                 .resume = next->resume.get()})) {
        result = ServeCooperative(std::move(*next), *cooperative, wake_fd, on_wake);
      } else {
        Serve(*next, wake_fd, on_wake);
      }
      next.reset();
      TrimIfOwed();
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
    for (Pending& p : parked_) {
      p.channel->parked = false;
      queue_.push_back(std::move(p));
    }
    parked_.clear();
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
  parser_.request_stop();
  parser_.join();
  return result;
}

}  // namespace jitllm::runtime::api
