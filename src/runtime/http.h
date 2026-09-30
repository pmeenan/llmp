// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// A small, bounded HTTP/1.1 server's pieces for M3's chat route (D-097;
// docs/runtime-serving.md#the-chat-route): a request head parsed from bytes
// under fixed bounds, response heads for persistent connections (with a
// chunked body when its length is not known), and a non-blocking listener.
// No third-party code; the event loop is api_server.h's.
//
// What a request may be: a request line of a method, an origin-form target
// and HTTP/1.1 (or 1.0); header lines ending in CRLF, without folding or
// control characters; a body only by Content-Length (a Transfer-Encoding
// is refused), which `Expect: 100-continue` is answered for.

#ifndef JITLLM_RUNTIME_HTTP_H_
#define JITLLM_RUNTIME_HTTP_H_

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "config/node_config.h"

namespace jitllm::runtime::http {

using Clock = std::chrono::steady_clock;

// An owned file descriptor.
class Fd {
 public:
  Fd() = default;
  explicit Fd(int fd) : fd_(fd) {}
  Fd(const Fd&) = delete;
  Fd& operator=(const Fd&) = delete;
  Fd(Fd&& other) noexcept : fd_(std::exchange(other.fd_, -1)) {}
  Fd& operator=(Fd&& other) noexcept;
  ~Fd();
  int get() const { return fd_; }
  bool valid() const { return fd_ >= 0; }

 private:
  int fd_ = -1;
};

struct Limits {
  std::size_t max_header_bytes = 16384;  // the request line and headers
  std::size_t max_headers = 64;
  std::size_t max_target_bytes = 2048;
  std::size_t max_body_bytes = std::size_t{16} << 20U;
};

struct Header {
  std::string name;  // lower case
  std::string value;
};

struct Request {
  std::string method;
  std::string target;  // as sent
  std::string path;    // the target without its query
  std::vector<Header> headers;
  std::string body;
  bool http10 = false;           // HTTP/1.0
  std::size_t body_bytes = 0;    // Content-Length (0 when none)
  bool expect_continue = false;  // Expect: 100-continue

  // The value of the header named `lowercase` (the first), or nullptr.
  const std::string* Find(std::string_view lowercase) const;
  // Whether the client lets the connection persist after the response:
  // HTTP/1.1 unless `Connection: close`, HTTP/1.0 only with
  // `Connection: keep-alive`.
  bool KeepAlive() const;
};

// Why a request could not be read: the status to answer with.
struct Failure {
  int status = 400;
  std::string message;
};

// Where a request's head ends in `buffer`: the offset just past its blank
// line, none while it is incomplete, or a 413 once it cannot fit.
std::expected<std::optional<std::size_t>, Failure> FindHeadEnd(std::string_view buffer,
                                                               const Limits& limits);

// Parses a complete head, `full` through its blank line, checking every bound
// but the body's arrival: a Failure is the status to refuse it with.
std::expected<Request, Failure> ParseHead(std::string_view full, const Limits& limits);

// A status line's reason phrase.
std::string_view Reason(int status);

// A response head: the status line, Content-Type, Content-Length (when
// given) or else `Transfer-Encoding: chunked` (when `chunked`; otherwise
// the body ends with the connection), Cache-Control: no-store, Connection
// (keep-alive, with Keep-Alive's timeout, or close) and `extra` header
// lines (each "Name: value").
struct HeadOptions {
  int status = 200;
  std::string_view content_type;
  std::optional<std::size_t> length;
  bool chunked = false;
  bool keep_alive = false;
  std::chrono::seconds idle_timeout{0};
  std::vector<std::string> extra;
};
std::string Head(const HeadOptions& options);

// A chunk of a chunked body ("" is the last chunk: "0\r\n\r\n").
std::string Chunk(std::string_view data);

// Writes all of data to a blocking socket; false when the peer is gone.
// (The tests' clients; the server's own sockets never block.)
bool WriteAll(int fd, std::string_view data);

// A listening TCP socket, non-blocking, on an endpoint (port 0: the
// kernel's choice) and the port it got. An IPv6 socket is IPv6 only.
struct Listener {
  Fd fd;
  std::uint16_t port = 0;
};
std::expected<Listener, std::string> Listen(const config::ClientEndpoint& endpoint);

}  // namespace jitllm::runtime::http

#endif  // JITLLM_RUNTIME_HTTP_H_
