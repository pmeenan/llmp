// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "platform/sockets.h"

#include <sys/random.h>
#include <sys/socket.h>
#include <sys/types.h>

#include <cstddef>
#include <span>

namespace llmp::platform {

int OpenStreamSocket(int family) {
  return ::socket(family, SOCK_STREAM | SOCK_CLOEXEC | SOCK_NONBLOCK, 0);
}

int AcceptConnection(int listener, sockaddr_storage& peer) {
  socklen_t size = sizeof peer;
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast): the sockets API
  return ::accept4(listener, reinterpret_cast<sockaddr*>(&peer), &size,
                   SOCK_NONBLOCK | SOCK_CLOEXEC);
}

ssize_t SendNoSignal(int fd, const void* data, std::size_t size, bool wait) {
  int flags = MSG_NOSIGNAL;
  if (!wait) {
    flags |= MSG_DONTWAIT;
  }
  return ::send(fd, data, size, flags);
}

bool FillRandom(std::span<std::byte> out) {
  return ::getrandom(out.data(), out.size(), 0) == static_cast<ssize_t>(out.size());
}

}  // namespace llmp::platform
