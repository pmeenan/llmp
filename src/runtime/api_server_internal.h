// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
// Private connection, channel and exchange state shared by the API I/O and
// driver units. Public API declarations remain in api_server.h.
#ifndef LLMP_RUNTIME_API_SERVER_INTERNAL_H_
#define LLMP_RUNTIME_API_SERVER_INTERNAL_H_
#include <sys/socket.h>

#include <format>
#include <utility>

#include "base/surface_versions.h"
#include "platform/memory_pressure.h"
#include "runtime/api_server.h"
#include "runtime/stream_encoder.h"
namespace llmp::runtime::api {
namespace server_detail {
inline constexpr std::string_view kJson = "application/json";
inline constexpr std::string_view kContinue = "HTTP/1.1 100 Continue\r\n\r\n";
// Event-loop tags beside connection IDs (which count up from 1).
inline constexpr std::uint64_t kListenerTag = std::uint64_t{1} << 56U;
inline constexpr std::uint64_t kStopTag = std::uint64_t{2} << 56U;
inline constexpr std::uint64_t kWakeTag = std::uint64_t{3} << 56U;
// A closing connection's last reads, so a response is not lost to a reset.
inline constexpr auto kLinger = std::chrono::milliseconds(200);
inline constexpr std::size_t kLingerBytes = std::size_t{1} << 20U;
// Accepting pauses this long when the process is out of descriptors.
inline constexpr auto kAcceptPause = std::chrono::milliseconds(100);
// How long a stopping server lets responses finish going out.
inline constexpr auto kDrain = std::chrono::seconds(1);
inline constexpr std::size_t kReadChunk = 65536;
// Sent to an HTTP/1.1 client whose input side closed while its
// non-streaming response is being made: a client that half-closed must
// accept an interim response (RFC 9110, section 15.2), though Python's
// http.client takes any 1xx but 100 as the final one (and undici fails on
// an unasked 100); one that closed entirely answers with a reset, which
// ends the request as a disconnect.
inline constexpr std::string_view kProcessing = "HTTP/1.1 102 Processing\r\n\r\n";
// A body's buffer grows by doubling from this, charged as it grows.
inline constexpr std::size_t kFirstBodyBytes = std::size_t{64} << 10U;
// How often a body waiting for the request memory tries again (a check of
// the pool's epoch, not a parse), and how often the parse thread looks.
inline constexpr auto kMemoryRetry = std::chrono::milliseconds(50);
inline constexpr auto kParseRetryPoll = std::chrono::milliseconds(50);
// A request that held this much of the request memory gives the C
// library's free heap back to the system when it ends (D-102): a large
// parse leaves many small allocations freed.
inline constexpr std::uint64_t kTrimAfterBytes = std::uint64_t{16} << 20U;
// Consecutive pauses with no paused member before the cohort is a fault
// (a member's reader may resume between the backend's look and the
// server's: that pass only resumes).
inline constexpr int kMaxEmptyPauses = 16;

void Release(std::string& buffer);
Error Refusal(int status, std::string message, std::string code = {});
Error Unresponsive();
Error Stalled(Clock::duration idle);
bool Readable(int fd);
std::string RequestId();
double Seconds(Clock::duration duration);
bool PeerIsLoopback(const sockaddr_storage& peer);
}  // namespace server_detail
// ---------------------------------------------------------------- Channel

// One request's response, shared by the I/O thread (which writes it to the
// socket) and the driver (which produces it). The first fields are fixed
// when it is made; the rest are under Server::mutex_.
struct Server::Channel {
  std::uint64_t connection = 0;  // the Connection's ID
  std::string id;                // the request's opaque ID
  std::string model;
  std::int64_t created = 0;
  bool stream = false;    // SSE
  StreamEncoder encoder;  // selected by the route before admission
  std::optional<Error> encoding_error;
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
    extra.push_back(std::format("llmp-inference-version: {}", surface::kInferenceVersion));
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
  StreamIdentity identity() const { return {.id = id, .model = model, .created = created}; }
  StreamEncoder::Sink sink() {
    return [this](std::string_view event) { PutBody(event); };
  }
  void Encoded(StreamEncoder::Status status) {
    if (!status) {
      encoding_error = server_detail::Refusal(500, std::string(status.error()), "internal");
      PutError(*encoding_error);
    }
  }
  void PutStart(std::optional<Usage> usage = {}) {
    if (!head_sent) {
      PutHead(200, "text/event-stream", std::nullopt);
    }
    Encoded(encoder.Start(identity(), usage, sink()));
  }
  void PutText(StreamText kind, std::string_view text) {
    if (!ended) {
      Encoded(encoder.Text(identity(), kind, text, sink()));
    }
  }
  void PutFinish(Finish finish, const Usage& usage, bool include_usage) {
    if (!ended) {
      Encoded(encoder.End(identity(),
                          finish == Finish::kStop ? StreamStop::kStop : StreamStop::kLength, usage,
                          include_usage, sink()));
    }
    if (!ended) {
      PutEnd();
    }
  }
  void PutKeepalive() { encoder.Keepalive(sink()); }
  void PutEnd() {
    if (chunked) {
      out += http::Chunk({});
    }
    ended = true;
  }
  // An error: the whole response before the head, an in-stream error
  // event after it (without a successful protocol finish).
  void PutError(const Error& error, std::vector<std::string> extra = {}) {
    if (ended) {
      return;
    }
    if (!head_sent) {
      const std::string json = encoder.ErrorBody(error);
      if (error.status == 429 || error.status == 503) {
        extra.push_back(std::format("Retry-After: {}", kRetryAfterSeconds));
        extra.emplace_back("x-should-retry: true");
      }
      PutHead(error.status, server_detail::kJson, json.size(), std::move(extra));
      PutBody(json);
    } else if (stream) {
      encoder.Fail(error, sink());
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
      if (channel) {
        channel->stream = parsed->stream;
      }
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
    large = body.size() >= server_detail::kTrimAfterBytes;
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
         Clock::time_point started, bool cooperative = false);

  bool Admit(const Admission& admission) override;
  bool Next(Phase phase, std::uint64_t size) override;
  bool Reasoning(std::string_view text) override;
  bool Content(std::string_view text) override;
  bool Continue() override;
  bool Paused() const override;
  bool timed_out() const override { return timed_out_; }
  bool Yielding() const override;
  // A cooperative member that yields (Work::Yield).
  void SetYielding();

  // A yielded request's response so far, for its next Stream (Requeue, or
  // a deferral back to the queue); `taken_up`: it ran, so its next Stream
  // counts from when it was first taken up.
  void Carry(Pending& pending, bool taken_up);

  // Hands the rest of the response to the I/O thread for the backend's
  // result; the status for the log.
  int End(const std::expected<Completion, Error>& result);

  const Usage& usage() const;
  // How often the request paused for its client (backpressure).
  std::uint64_t pauses() const;

 private:
  // With the lock held.
  void Dirty();

  void EmitLocked(const OutputText::Out& out);

  // With the lock held: paused, and not ended otherwise.
  bool PausedLocked() const;

  // The serial path waits here, at a completed step, while its client is
  // behind; a cooperative backend leaves the request out of its units
  // instead (Paused). False once the runtime stops, or when the request is
  // to yield its place: queued requests waited behind its pause for
  // ServerOptions::yield_after (Yielding).
  bool Backpressure();

  void Emit(const OutputText::Out& out);

  // Progress: the watchdog's beat, `next` the unit that follows (none:
  // the one under way goes on) and its expected seconds at the floors.
  void Progress(std::optional<Phase> next, double expected);

  // A beat (above), then whether the generation goes on.
  bool Check(std::optional<Phase> next, double expected);

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

}  // namespace llmp::runtime::api
#endif  // LLMP_RUNTIME_API_SERVER_INTERNAL_H_
