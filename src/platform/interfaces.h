// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The node's network identity for its listeners (D-097): the addresses its
// interfaces hold (getifaddrs, which reads them over netlink), its host
// name, the name its resolver gives an address, and the open-file limit a
// server of many connections needs. No external command is run: in
// particular, the tailnet is found from its interface, not the tailscale
// CLI.

#ifndef LLMP_PLATFORM_INTERFACES_H_
#define LLMP_PLATFORM_INTERFACES_H_

#include <array>
#include <chrono>
#include <cstdint>
#include <expected>
#include <optional>
#include <string>
#include <vector>

namespace llmp::platform {

struct InterfaceAddress {
  std::string interface;  // "lo", "tailscale0", ...
  bool ipv6 = false;
  // Network byte order; IPv4 in the first four bytes.
  std::array<std::uint8_t, 16> bytes{};
  bool up = false;        // IFF_UP
  bool loopback = false;  // IFF_LOOPBACK

  // As inet_ntop writes it.
  std::string Text() const;
};

// Every IPv4 and IPv6 address on the node's interfaces.
std::expected<std::vector<InterfaceAddress>, std::string> ReadInterfaceAddresses();

// The name the node's resolver gives an address (a reverse lookup,
// getnameinfo with NI_NAMEREQD), lower case without a trailing dot, or
// none: none too once `limit` has passed. The resolver has no timeout of
// ours (glibc's is 5 s a try, two tries a server), so the lookup runs on a
// thread of its own, detached, whose late answer is dropped: a resolver
// that does not answer holds startup at most `limit`.
std::optional<std::string> ReverseName(const InterfaceAddress& address,
                                       std::chrono::milliseconds limit);

// gethostname's, lower case; empty if it fails.
std::string HostName();

// Raises the soft open-file limit (RLIMIT_NOFILE) toward `want`, as far as
// the hard limit allows; the soft limit afterwards.
std::uint64_t RaiseOpenFileLimit(std::uint64_t want);

}  // namespace llmp::platform

#endif  // LLMP_PLATFORM_INTERFACES_H_
