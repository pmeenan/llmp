// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "platform/interfaces.h"

#include <arpa/inet.h>
#include <ifaddrs.h>
#include <net/if.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/resource.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <expected>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace llmp::platform {
namespace {

std::string Errno(int error) { return std::strerror(error); }  // NOLINT(concurrency-mt-unsafe)

std::string Lower(std::string text) {
  for (char& c : text) {
    if (c >= 'A' && c <= 'Z') {
      c = static_cast<char>(c - 'A' + 'a');
    }
  }
  return text;
}

}  // namespace

std::string InterfaceAddress::Text() const {
  std::array<char, INET6_ADDRSTRLEN> text{};
  (void)::inet_ntop(ipv6 ? AF_INET6 : AF_INET, bytes.data(), text.data(), text.size());
  return text.data();
}

std::expected<std::vector<InterfaceAddress>, std::string> ReadInterfaceAddresses() {
  ifaddrs* list = nullptr;
  if (::getifaddrs(&list) != 0) {
    return std::unexpected("getifaddrs: " + Errno(errno));
  }
  std::vector<InterfaceAddress> out;
  for (const ifaddrs* a = list; a != nullptr; a = a->ifa_next) {
    if (a->ifa_addr == nullptr || a->ifa_name == nullptr) {
      continue;
    }
    InterfaceAddress address;
    address.interface = a->ifa_name;
    address.up = (a->ifa_flags & IFF_UP) != 0;
    address.loopback = (a->ifa_flags & IFF_LOOPBACK) != 0;
    if (a->ifa_addr->sa_family == AF_INET) {
      sockaddr_in four{};
      std::memcpy(&four, a->ifa_addr, sizeof four);
      std::memcpy(address.bytes.data(), &four.sin_addr, 4);
    } else if (a->ifa_addr->sa_family == AF_INET6) {
      sockaddr_in6 six{};
      std::memcpy(&six, a->ifa_addr, sizeof six);
      std::memcpy(address.bytes.data(), &six.sin6_addr, 16);
      address.ipv6 = true;
    } else {
      continue;
    }
    out.push_back(std::move(address));
  }
  ::freeifaddrs(list);
  return out;
}

namespace {

std::optional<std::string> LookUp(const InterfaceAddress& address) {
  sockaddr_storage storage{};
  socklen_t size = 0;
  if (address.ipv6) {
    sockaddr_in6 six{};
    six.sin6_family = AF_INET6;
    std::memcpy(&six.sin6_addr, address.bytes.data(), 16);
    std::memcpy(&storage, &six, sizeof six);
    size = sizeof six;
  } else {
    sockaddr_in four{};
    four.sin_family = AF_INET;
    std::memcpy(&four.sin_addr, address.bytes.data(), 4);
    std::memcpy(&storage, &four, sizeof four);
    size = sizeof four;
  }
  std::array<char, NI_MAXHOST> host{};
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast): the sockets API
  if (::getnameinfo(reinterpret_cast<const sockaddr*>(&storage), size, host.data(), host.size(),
                    nullptr, 0, NI_NAMEREQD) != 0) {
    return std::nullopt;
  }
  std::string name = Lower(host.data());
  while (name.ends_with('.')) {
    name.pop_back();
  }
  if (name.empty()) {
    return std::nullopt;
  }
  return name;
}

}  // namespace

std::optional<std::string> ReverseName(const InterfaceAddress& address,
                                       std::chrono::milliseconds limit) {
  // Shared with the lookup's thread, which may outlive this call.
  struct Lookup {
    std::mutex mutex;
    std::condition_variable done_cv;
    bool done = false;
    std::optional<std::string> name;
  };
  auto lookup = std::make_shared<Lookup>();
  std::thread([lookup, address] {
    std::optional<std::string> name = LookUp(address);
    const std::scoped_lock lock(lookup->mutex);
    lookup->name = std::move(name);
    lookup->done = true;
    lookup->done_cv.notify_all();
  }).detach();  // owns what it touches; a late answer is dropped with it
  std::unique_lock lock(lookup->mutex);
  if (!lookup->done_cv.wait_for(lock, limit, [&] { return lookup->done; })) {
    return std::nullopt;
  }
  return lookup->name;
}

std::string HostName() {
  std::array<char, 256> name{};
  if (::gethostname(name.data(), name.size() - 1) != 0) {
    return {};
  }
  return Lower(name.data());
}

std::uint64_t RaiseOpenFileLimit(std::uint64_t want) {
  rlimit limit{};
  if (::getrlimit(RLIMIT_NOFILE, &limit) != 0) {
    return 0;
  }
  if (limit.rlim_cur != RLIM_INFINITY && limit.rlim_cur < want) {
    const rlim_t raised =
        limit.rlim_max == RLIM_INFINITY ? want : std::min<rlim_t>(limit.rlim_max, want);
    if (raised > limit.rlim_cur) {
      rlimit next = limit;
      next.rlim_cur = raised;
      if (::setrlimit(RLIMIT_NOFILE, &next) == 0) {
        limit = next;
      }
    }
  }
  return limit.rlim_cur == RLIM_INFINITY ? std::numeric_limits<std::uint64_t>::max()
                                         : static_cast<std::uint64_t>(limit.rlim_cur);
}

}  // namespace llmp::platform
