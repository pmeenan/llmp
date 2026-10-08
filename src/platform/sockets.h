// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The few socket calls whose spelling differs between systems
// (docs/portability.md), for the chat route's listeners and connections
// (runtime/http.h, runtime/api_server.h): sockets made non-blocking and
// close-on-exec atomically where the system can (Linux's SOCK_NONBLOCK,
// SOCK_CLOEXEC and accept4; macOS sets them with fcntl after the call),
// sends that never raise SIGPIPE (Linux's MSG_NOSIGNAL; macOS's
// SO_NOSIGPIPE, set once on the socket), and the system's cryptographic
// random bytes (getrandom; getentropy or arc4random_buf on macOS,
// BCryptGenRandom on Windows). The rest of the sockets API is POSIX's.

#ifndef LLMP_PLATFORM_SOCKETS_H_
#define LLMP_PLATFORM_SOCKETS_H_

#include <sys/socket.h>
#include <sys/types.h>

#include <cstddef>
#include <span>

namespace llmp::platform {

// A non-blocking, close-on-exec stream socket of `family`; -1 with errno
// as socket()'s.
int OpenStreamSocket(int family);

// The next connection waiting on `listener`, as a non-blocking,
// close-on-exec socket, its peer's address in `peer`; -1 with errno as
// accept()'s.
int AcceptConnection(int listener, sockaddr_storage& peer);

// send() that never raises SIGPIPE; with `wait` false it never blocks
// either (EAGAIN instead). As send()'s result.
ssize_t SendNoSignal(int fd, const void* data, std::size_t size, bool wait);

// Fills `out` with the system's cryptographic random bytes. False if it
// could not fill all of it.
bool FillRandom(std::span<std::byte> out);

}  // namespace llmp::platform

#endif  // LLMP_PLATFORM_SOCKETS_H_
