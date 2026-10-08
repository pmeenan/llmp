// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "platform/sd_notify.h"

#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <cerrno>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <expected>
#include <string>
#include <string_view>

namespace llmp::platform {

std::expected<bool, std::string> NotifyServiceManager(std::string_view state) {
  const char* socket_path = std::getenv("NOTIFY_SOCKET");  // NOLINT(concurrency-mt-unsafe)
  if (socket_path == nullptr || *socket_path == '\0') {
    return false;
  }
  const std::string_view name(socket_path);
  sockaddr_un address{};
  address.sun_family = AF_UNIX;
  if ((name.front() != '/' && name.front() != '@') || name.size() >= sizeof(address.sun_path)) {
    return std::unexpected("NOTIFY_SOCKET is not a socket path");
  }
  (void)name.copy(address.sun_path, name.size());
  if (name.front() == '@') {
    address.sun_path[0] = '\0';  // an abstract socket
  }
  const auto length = static_cast<socklen_t>(offsetof(sockaddr_un, sun_path) + name.size());
  const int fd = ::socket(AF_UNIX, SOCK_DGRAM | SOCK_CLOEXEC, 0);
  if (fd < 0) {
    return std::unexpected(std::string("cannot create a socket: ") +
                           std::strerror(errno));  // NOLINT
  }
  const ssize_t sent = ::sendto(fd, state.data(), state.size(), MSG_NOSIGNAL,
                                reinterpret_cast<const sockaddr*>(&address), length);  // NOLINT
  const int error = errno;
  (void)::close(fd);
  if (sent < 0 || static_cast<std::size_t>(sent) != state.size()) {
    return std::unexpected(std::string("cannot notify the service manager: ") +
                           std::strerror(error));  // NOLINT
  }
  return true;
}

}  // namespace llmp::platform
