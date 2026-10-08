// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Where the chat route listens and which names reach it (D-097 as the
// owner amended it on 2026-09-28; docs/runtime-serving.md#the-chat-route):
// `[client] bind` resolved against the node's interface addresses, and the
// Host and Origin guard (the DNS-rebinding defense of D-045 and D-064) that
// accepts exactly the names and addresses the node listens under. Pure
// functions of their inputs, so the CPU tests give them a fake interface
// list.
//
// The tailnet. An interface is Tailscale's when its name begins with
// "tailscale" (tailscaled's default is tailscale0) or it holds an address in
// Tailscale's IPv6 range, fd7a:115c:a1e0::/48; its tailnet addresses are
// those in that range or in 100.64.0.0/10. A 100.64.0.0/10 address on any
// other interface is a carrier's shared address space (RFC 6598), not the
// tailnet. No binding needs authentication (D-014's 2026-09-28 owner
// note), as with other engines' defaults; an address that is neither
// loopback nor the tailnet is served without it, and the resolution says
// so for the start log.

#ifndef LLMP_RUNTIME_BINDING_H_
#define LLMP_RUNTIME_BINDING_H_

#include <array>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "config/node_config.h"
#include "platform/interfaces.h"

namespace llmp::runtime::api {

// Whether a Host header names a loopback address or localhost (any port).
bool IsLoopbackHost(std::string_view host);

// Whether an address lies in Tailscale's ranges (100.64.0.0/10,
// fd7a:115c:a1e0::/48), whatever its interface.
bool InTailnetRange(const platform::InterfaceAddress& address);

// The names and addresses a request may name in its Host header (and an
// Origin, with one of the listening ports): loopback always, and whatever
// the resolution adds. A name is compared without case and never with a
// trailing dot; an address as an address.
class HostGuard {
 public:
  // A host name ([a-z0-9.-], at most 253 bytes); anything else is ignored.
  void AddName(std::string_view name);
  void AddAddress(bool ipv6, const std::array<std::uint8_t, 16>& bytes);
  void AddPort(std::uint16_t port);

  // "name", "name:port", "1.2.3.4:port", "[::1]:port".
  bool AllowsHost(std::string_view host) const;
  // "http://host[:port]" or "https://..." naming an allowed host and one of
  // the listening ports (80 and 443 when none is written).
  bool AllowsOrigin(std::string_view origin) const;

  const std::vector<std::string>& names() const { return names_; }

 private:
  struct Address {
    bool ipv6 = false;
    std::array<std::uint8_t, 16> bytes{};
  };
  std::vector<std::string> names_;
  std::vector<Address> addresses_;
  std::vector<std::uint16_t> ports_;
};

// What the service listens on, and what to say about it at startup.
struct Listening {
  std::vector<config::ClientEndpoint> endpoints;  // no duplicates; a wildcard covers its family
  HostGuard hosts;
  std::vector<std::string> notes;            // what was found
  std::vector<std::string> unauthenticated;  // listeners beyond loopback and the tailnet, logged
};

// A reverse lookup (platform::ReverseName under a time budget in the
// service).
using ReverseLookup =
    std::function<std::optional<std::string>(const platform::InterfaceAddress& address)>;

inline constexpr std::size_t kMaxReverseLookups = 16;

// Resolves `[client]` against the node's addresses: "loopback" is
// 127.0.0.1 and, if the node has it, ::1; "tailscale" the tailnet
// addresses found (none: a note, and loopback alone); an address itself.
// The Host guard gets the node's host name, the tailnet addresses and
// their MagicDNS names (a reverse name under ts.net, and its first label),
// and each explicit address with its reverse name; a wildcard adds every
// address of its family but link-local ones.
Listening ResolveListening(const config::ClientConfig& client,
                           const std::vector<platform::InterfaceAddress>& addresses,
                           std::string_view hostname, const ReverseLookup& reverse);

}  // namespace llmp::runtime::api

#endif  // LLMP_RUNTIME_BINDING_H_
