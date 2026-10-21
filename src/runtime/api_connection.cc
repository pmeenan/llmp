// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
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
#include "runtime/api_server.h"
#include "runtime/api_server_internal.h"

namespace llmp::runtime::api {
using server_detail::kAcceptPause;
using server_detail::kContinue;
using server_detail::kDrain;
using server_detail::kFirstBodyBytes;
using server_detail::kJson;
using server_detail::kLinger;
using server_detail::kLingerBytes;
using server_detail::kListenerTag;
using server_detail::kMemoryRetry;
using server_detail::kParseRetryPoll;
using server_detail::kProcessing;
using server_detail::kReadChunk;
using server_detail::kStopTag;
using server_detail::kWakeTag;
using server_detail::PeerIsLoopback;
using server_detail::Refusal;
using server_detail::Release;
using server_detail::RequestId;
using server_detail::Unresponsive;
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
      // What the body still owes: none when it all arrived (with pipelined
      // bytes after it, perhaps) while it waited for the request memory.
      const std::size_t whole = c.head_end + c.request.body_bytes;
      most = std::min(most, whole - std::min(whole, c.in.size()));
      // Room for it in the request memory first, or the connection waits,
      // unread, until the driver grows the memory (or refuses).
      if (const Growth grown = GrowBody(c, most); grown != Growth::kOk) {
        return;
      }
      if (most == 0) {
        BodyArrived(c);
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
      // Its head's checks come first: a request the head refuses has none
      // of its body read or charged, and is not asked to continue. With its
      // body still to come it closes the connection (where the next request
      // would begin is unknown); one whole already is refused as usual.
      if (std::optional<HeadRefusal> refused = Screen(c, c.request)) {
        if (c.in.size() < c.head_end + c.request.body_bytes) {
          Refuse(c, refused->error, std::move(refused->extra));
        } else {
          BodyArrived(c);
        }
        return;
      }
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

std::optional<Server::HeadRefusal> Server::Screen(const Connection& c,
                                                  const http::Request& request) const {
  const auto refuse = [](Error error, std::vector<std::string> extra = {}) {
    return std::optional<HeadRefusal>(
        HeadRefusal{.error = std::move(error), .extra = std::move(extra)});
  };
  // Browser guards (D-045, D-064): the Host names this node as it listens,
  // and nothing arrives from another site's page.
  const std::string* host = request.Find("host");
  if (host == nullptr || !options_.hosts.AllowsHost(*host)) {
    return refuse(Refusal(403, "the Host header must name this node as the chat route listens"));
  }
  const std::string* origin = request.Find("origin");
  const std::string* site = request.Find("sec-fetch-site");
  if ((origin != nullptr && !options_.hosts.AllowsOrigin(*origin)) ||
      (site != nullptr && *site != "none" && *site != "same-origin")) {
    return refuse(Refusal(403, "requests from other sites' web pages are refused on this route"));
  }
  const auto method_not_allowed = [&](std::string_view allow) {
    return refuse(Refusal(405, std::format("use {}", allow)), {std::format("Allow: {}", allow)});
  };
  if (request.path == "/llmp/v1/memory/release") {
    if (!c.peer_loopback) {
      return refuse(Refusal(403, "/llmp/v1/memory/release is restricted to loopback"));
    }
    if (request.method != "POST") {
      return method_not_allowed("POST");
    }
    return std::nullopt;
  }
  if ((request.path == "/llmp/v1/ignored-fields" && c.peer_loopback) ||
      request.path == "/v1/models" || request.path.starts_with("/v1/models/")) {
    if (request.method != "GET") {
      return method_not_allowed("GET");
    }
    return std::nullopt;
  }
  if (request.path != "/v1/chat/completions" && request.path != "/v1/completions") {
    return refuse(Refusal(404,
                          "no such route; this runtime serves /v1/chat/completions, "
                          "/v1/completions and /v1/models"));
  }
  if (request.method != "POST") {
    return method_not_allowed("POST");
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
    return refuse(Refusal(415, "the body must be Content-Type: application/json"));
  }
  return std::nullopt;
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

  // Its head's checks: here a request whose body arrived with its head (the
  // others were screened before their bodies were read).
  if (std::optional<HeadRefusal> refused = Screen(c, request)) {
    Refuse(c, refused->error, std::move(refused->extra));
    return;
  }
  const auto answer = [&](const std::string& body) {
    {
      const std::scoped_lock lock(mutex_);
      channel->PutHead(200, kJson, body.size());
      channel->PutBody(body);
      channel->PutEnd();
    }
    Flush(c);
  };
  if (request.path == "/llmp/v1/memory/release") {
    bool busy = false;
    {
      const std::scoped_lock lock(mutex_);
      busy = !running_.empty() || !queue_.empty() || !parked_.empty() || suspended_requests_ != 0 ||
             pending_release_.has_value();
      if (!busy) {
        pending_release_ = PendingRelease{
            .connection = c.id,
            .channel = channel,
        };
      }
    }
    if (busy) {
      Refuse(c, Refusal(409, "server is busy with active work", "busy"));
      return;
    }
    ready_.Signal();
    return;
  }
  if (request.path == "/llmp/v1/ignored-fields") {
    answer(ignored_.Json());
    return;
  }
  if (request.path == "/v1/models") {
    answer(ModelsJson(models_, created_));
    return;
  }
  if (request.path.starts_with("/v1/models/")) {
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
    if (!waiting.empty()) {
      // A departed client's parse waits no longer: handed back, it answers
      // no one (TakeParsed) and its body and charges are freed.
      std::vector<std::unique_ptr<Parse>> departed;
      {
        const std::scoped_lock lock(mutex_);
        for (auto it = waiting.begin(); it != waiting.end();) {
          if ((*it)->channel != nullptr && (*it)->channel->gone) {
            departed.push_back(std::move(*it));
            it = waiting.erase(it);
          } else {
            ++it;
          }
        }
      }
      for (auto& gone : departed) {
        hand_back(std::move(gone));
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
  // None when the whole body (and perhaps pipelined bytes after it)
  // arrived with its head.
  const std::size_t whole = c.head_end + c.request.body_bytes;
  const std::size_t remaining = whole - std::min(whole, c.in.size());
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
                    "GET /llmp/v1/ignored-fields counts them)",
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
  std::vector<std::string> lines;  // logged once mutex_ is released
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
        lines.push_back(
            std::format("request {}: 429 after waiting {} s in the queue", ch.id,
                        std::chrono::duration_cast<std::chrono::seconds>(*wait).count()));
        ch.PutError(Refusal(429,
                            "the request waited longer than [client] queue_wait_seconds behind "
                            "others; retry later"));
        expired.push_back(ch.connection);
        return true;
      });
    }
  }
  for (const std::string& line : lines) {
    Log(line);
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

}  // namespace llmp::runtime::api
