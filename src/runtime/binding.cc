// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "runtime/binding.h"

#include <arpa/inet.h>
#include <netinet/in.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <format>
#include <iterator>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace llmp::runtime::api {
namespace {

using Bytes = std::array<std::uint8_t, 16>;

char Lower(char c) { return c >= 'A' && c <= 'Z' ? static_cast<char>(c - 'A' + 'a') : c; }

std::string Lowered(std::string_view text) {
  std::string out(text);
  std::ranges::transform(out, out.begin(), Lower);
  return out;
}

bool IsHostName(std::string_view name) {
  return !name.empty() && name.size() <= 253 && name.front() != '.' && name.back() != '.' &&
         name.front() != '-' && std::ranges::all_of(name, [](char c) {
           return (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '.' || c == '-';
         });
}

// The address in `text`, if it is one: IPv6 without brackets, or IPv4.
std::optional<std::pair<bool, Bytes>> ParseAddress(std::string_view text, bool ipv6) {
  const std::string owned(text);
  Bytes bytes{};
  if (::inet_pton(ipv6 ? AF_INET6 : AF_INET, owned.c_str(), bytes.data()) != 1) {
    return std::nullopt;
  }
  return std::pair{ipv6, bytes};
}

bool IsLoopbackAddress(bool ipv6, const Bytes& bytes) {
  if (!ipv6) {
    return bytes[0] == 127;
  }
  return std::memcmp(bytes.data(), &in6addr_loopback, 16) == 0;
}

bool IsWildcard(const config::ClientEndpoint& endpoint) {
  return endpoint.address == "0.0.0.0" || endpoint.address == "::";
}

bool IsLinkLocal(const platform::InterfaceAddress& a) {
  return a.ipv6 ? (a.bytes[0] == 0xFE && (a.bytes[1] & 0xC0U) == 0x80U)
                : (a.bytes[0] == 169 && a.bytes[1] == 254);
}

std::string EndpointText(const config::ClientEndpoint& e) {
  return e.ipv6 ? std::format("[{}]:{}", e.address, e.port)
                : std::format("{}:{}", e.address, e.port);
}

// A Host (or Origin) authority split into the host and its port text;
// none when it is malformed.
struct Authority {
  std::string_view host;  // an IPv6 literal without its brackets
  bool bracketed = false;
  std::optional<std::string_view> port;
};

std::optional<Authority> SplitAuthority(std::string_view text) {
  Authority out;
  std::string_view rest;
  if (text.starts_with('[')) {
    const std::size_t close = text.find(']');
    if (close == std::string_view::npos) {
      return std::nullopt;
    }
    out.host = text.substr(1, close - 1);
    out.bracketed = true;
    rest = text.substr(close + 1);
  } else {
    const std::size_t colon = text.find(':');
    out.host = text.substr(0, colon);
    rest = colon == std::string_view::npos ? std::string_view() : text.substr(colon);
  }
  if (!rest.empty()) {
    if (!rest.starts_with(':')) {
      return std::nullopt;
    }
    const std::string_view port = rest.substr(1);
    if (port.empty() || port.size() > 5 ||
        !std::ranges::all_of(port, [](char c) { return c >= '0' && c <= '9'; })) {
      return std::nullopt;
    }
    out.port = port;
  }
  if (out.host.empty()) {
    return std::nullopt;
  }
  return out;
}

}  // namespace

bool IsLoopbackHost(std::string_view host) {
  const auto authority = SplitAuthority(host);
  if (!authority) {
    return false;
  }
  if (authority->bracketed) {
    const auto address = ParseAddress(authority->host, true);
    return address && IsLoopbackAddress(true, address->second);
  }
  if (Lowered(authority->host) == "localhost") {
    return true;
  }
  const auto address = ParseAddress(authority->host, false);
  return address && IsLoopbackAddress(false, address->second);
}

bool InTailnetRange(const platform::InterfaceAddress& a) {
  if (!a.ipv6) {
    return a.bytes[0] == 100 && (a.bytes[1] & 0xC0U) == 64U;  // 100.64.0.0/10
  }
  static constexpr std::array<std::uint8_t, 6> kPrefix = {0xFD, 0x7A, 0x11, 0x5C, 0xA1, 0xE0};
  return std::ranges::equal(kPrefix, std::span(a.bytes).first<6>());  // fd7a:115c:a1e0::/48
}

void HostGuard::AddName(std::string_view name) {
  std::string lowered = Lowered(name);
  if (IsHostName(lowered) && std::ranges::find(names_, lowered) == names_.end()) {
    names_.push_back(std::move(lowered));
  }
}

void HostGuard::AddAddress(bool ipv6, const std::array<std::uint8_t, 16>& bytes) {
  Address address{.ipv6 = ipv6, .bytes = {}};
  std::ranges::copy_n(bytes.begin(), ipv6 ? 16 : 4, address.bytes.begin());
  if (std::ranges::none_of(addresses_, [&](const Address& a) {
        return a.ipv6 == address.ipv6 && a.bytes == address.bytes;
      })) {
    addresses_.push_back(address);
  }
}

void HostGuard::AddPort(std::uint16_t port) {
  if (std::ranges::find(ports_, port) == ports_.end()) {
    ports_.push_back(port);
  }
}

bool HostGuard::AllowsHost(std::string_view host) const {
  if (IsLoopbackHost(host)) {
    return true;
  }
  const auto authority = SplitAuthority(host);
  if (!authority) {
    return false;
  }
  if (const auto address = ParseAddress(authority->host, authority->bracketed)) {
    const bool ipv6 = address->first;
    Bytes bytes{};
    std::ranges::copy_n(address->second.begin(), ipv6 ? 16 : 4, bytes.begin());
    return std::ranges::any_of(
        addresses_, [&](const Address& a) { return a.ipv6 == ipv6 && a.bytes == bytes; });
  }
  if (authority->bracketed) {
    return false;
  }
  return std::ranges::find(names_, Lowered(authority->host)) != names_.end();
}

bool HostGuard::AllowsOrigin(std::string_view origin) const {
  std::uint16_t default_port = 0;
  std::string_view rest;
  const std::size_t separator = origin.find("://");
  if (separator == std::string_view::npos) {
    return false;  // "null", or a bare "http": substr past its end would abort the process
  }
  const std::string scheme = Lowered(origin.substr(0, separator));
  if (scheme == "http") {
    default_port = 80;
  } else if (scheme == "https") {
    default_port = 443;
  } else {
    return false;
  }
  rest = origin.substr(separator + 3);
  const auto authority = SplitAuthority(rest);
  if (!authority || !AllowsHost(rest)) {
    return false;
  }
  std::uint32_t port = default_port;
  if (authority->port) {
    port = 0;
    for (const char c : *authority->port) {
      port = (port * 10) + static_cast<std::uint32_t>(c - '0');
    }
  }
  return std::ranges::any_of(ports_, [&](std::uint16_t p) { return p == port; });
}

Listening ResolveListening(const config::ClientConfig& client,
                           const std::vector<platform::InterfaceAddress>& addresses,
                           std::string_view hostname, const ReverseLookup& reverse) {
  Listening out;
  using Kind = config::BindEntry::Kind;
  std::size_t lookups = 0;
  const auto lookup = [&](const platform::InterfaceAddress& a) -> std::optional<std::string> {
    if (!reverse || lookups >= kMaxReverseLookups) {
      return std::nullopt;
    }
    ++lookups;
    return reverse(a);
  };
  const auto add = [&](config::ClientEndpoint endpoint) {
    if (std::ranges::none_of(out.endpoints, [&](const config::ClientEndpoint& e) {
          return e.ipv6 == endpoint.ipv6 && e.address == endpoint.address &&
                 e.port == endpoint.port;
        })) {
      out.endpoints.push_back(std::move(endpoint));
    }
  };
  const auto usable = [](const platform::InterfaceAddress& a) { return a.up; };

  // The tailnet: its interfaces and their addresses in its ranges.
  std::vector<std::string> tailnet_interfaces;
  for (const platform::InterfaceAddress& a : addresses) {
    if ((a.interface.starts_with("tailscale") || (a.ipv6 && InTailnetRange(a))) &&
        std::ranges::find(tailnet_interfaces, a.interface) == tailnet_interfaces.end()) {
      tailnet_interfaces.push_back(a.interface);
    }
  }
  std::vector<platform::InterfaceAddress> tailnet;
  for (const platform::InterfaceAddress& a : addresses) {
    if (usable(a) && InTailnetRange(a) &&
        std::ranges::find(tailnet_interfaces, a.interface) != tailnet_interfaces.end()) {
      tailnet.push_back(a);
    }
  }
  const auto on_tailnet = [&](const config::ClientEndpoint& e) {
    return std::ranges::any_of(tailnet, [&](const platform::InterfaceAddress& a) {
      return a.ipv6 == e.ipv6 && a.Text() == e.address;
    });
  };

  // An address's reverse name. An address in Tailscale's ranges is named
  // only under ts.net (MagicDNS's answer), with its short name too, as
  // "tailscale" names it: another name for it is not trusted.
  const auto add_names = [&](const platform::InterfaceAddress& a) {
    auto name = lookup(a);
    if (!name) {
      return;
    }
    if (!InTailnetRange(a)) {
      out.hosts.AddName(*name);
    } else if (name->ends_with(".ts.net")) {
      out.hosts.AddName(*name);
      out.hosts.AddName(std::string_view(*name).substr(0, name->find('.')));
    }
  };

  if (!hostname.empty()) {
    out.hosts.AddName(hostname);
    out.hosts.AddName(hostname.substr(0, hostname.find('.')));
  }
  bool tailnet_named = false;
  for (const config::BindEntry& entry : client.bind) {
    switch (entry.kind) {
      case Kind::kLoopback: {
        add({.address = "127.0.0.1", .ipv6 = false, .port = client.port});
        const bool six = std::ranges::any_of(addresses, [&](const platform::InterfaceAddress& a) {
          return usable(a) && a.ipv6 && a.loopback && a.Text() == "::1";
        });
        if (six) {
          add({.address = "::1", .ipv6 = true, .port = client.port});
        }
        break;
      }
      case Kind::kTailscale: {
        if (tailnet.empty()) {
          out.notes.emplace_back(
              "Tailscale: no tailnet interface found, so the chat route serves loopback and any "
              "configured address only; restart the runtime once Tailscale is up to serve the "
              "tailnet");
          break;
        }
        std::vector<std::string> names;
        for (const platform::InterfaceAddress& a : tailnet) {
          add({.address = a.Text(), .ipv6 = a.ipv6, .port = client.port});
          out.hosts.AddAddress(a.ipv6, a.bytes);
          if (tailnet_named) {
            continue;
          }
          // MagicDNS answers the reverse lookup with the node's name under
          // ts.net; both its addresses give the same one.
          if (auto name = lookup(a); name && name->ends_with(".ts.net")) {
            out.hosts.AddName(*name);
            out.hosts.AddName(std::string_view(*name).substr(0, name->find('.')));
            names.push_back(*name);
            tailnet_named = true;
          }  // another name is not the tailnet's, and is not trusted as one
        }
        std::string found;
        for (const platform::InterfaceAddress& a : tailnet) {
          found += std::format("{}{} on {}", found.empty() ? "" : ", ", a.Text(), a.interface);
        }
        out.notes.push_back(std::format(
            "Tailscale: serving the tailnet at {}{}", found,
            names.empty() ? std::string("; its MagicDNS name is unknown (no reverse name under "
                                        "ts.net), so only its addresses pass the Host check")
                          : std::format(" as {}", names.front())));
        break;
      }
      case Kind::kAddress: {
        config::ClientEndpoint endpoint = entry.endpoint;
        if (endpoint.port == 0) {
          endpoint.port = client.port;
        }
        const auto bytes = ParseAddress(endpoint.address, endpoint.ipv6);
        if (IsWildcard(endpoint)) {
          for (const platform::InterfaceAddress& a : addresses) {
            if (a.ipv6 != endpoint.ipv6 || !usable(a) || IsLinkLocal(a)) {
              continue;
            }
            out.hosts.AddAddress(a.ipv6, a.bytes);
            if (!a.loopback) {
              add_names(a);
            }
          }
          out.unauthenticated.push_back(std::format(
              "the chat route is serving without authentication on {} (every {} interface): "
              "anyone who can reach this node there can use its models, as with other engines' "
              "defaults; an optional API key comes with M5's front door (D-014, D-097)",
              EndpointText(endpoint), endpoint.ipv6 ? "IPv6" : "IPv4"));
        } else if (bytes) {
          out.hosts.AddAddress(bytes->first, bytes->second);
          const bool loopback = IsLoopbackAddress(bytes->first, bytes->second);
          if (!loopback) {
            platform::InterfaceAddress a{.interface = {},
                                         .ipv6 = bytes->first,
                                         .bytes = bytes->second,
                                         .up = true,
                                         .loopback = false};
            add_names(a);
          }
          if (!loopback && !on_tailnet(endpoint)) {
            out.unauthenticated.push_back(std::format(
                "the chat route is serving without authentication on {}, which is neither "
                "loopback nor the tailnet: anyone who can reach that address can use this "
                "node's models; an optional API key comes with M5's front door (D-014, D-097)",
                EndpointText(endpoint)));
          }
        }
        add(std::move(endpoint));
        break;
      }
    }
  }
  // A wildcard covers its family's addresses on its port: binding both
  // would fail.
  std::vector<config::ClientEndpoint> wildcards;
  std::ranges::copy_if(out.endpoints, std::back_inserter(wildcards), IsWildcard);
  std::erase_if(out.endpoints, [&](const config::ClientEndpoint& e) {
    return !IsWildcard(e) && std::ranges::any_of(wildcards, [&](const config::ClientEndpoint& w) {
      return w.ipv6 == e.ipv6 && w.port == e.port;
    });
  });
  return out;
}

}  // namespace llmp::runtime::api
