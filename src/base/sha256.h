// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// SHA-256 (FIPS 180-4), for identities such as implementation and plan
// identities (D-053) and a v0 artifact's ID and document digests (D-056),
// whose bytes may be untrusted. Incremental: Update any number of times,
// then Finish once.

#ifndef LLMP_BASE_SHA256_H_
#define LLMP_BASE_SHA256_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>

namespace llmp::base {

using Sha256Digest = std::array<std::uint8_t, 32>;

class Sha256 {
 public:
  Sha256& Update(std::span<const std::byte> bytes);
  Sha256& Update(std::string_view text) { return Update(std::as_bytes(std::span(text))); }
  // The digest of everything updated so far; the object is then spent.
  Sha256Digest Finish();

 private:
  void Block(const std::uint8_t* block);

  std::array<std::uint32_t, 8> state_ = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                                         0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
  std::array<std::uint8_t, 64> buffer_ = {};
  std::size_t buffered_ = 0;
  std::uint64_t length_ = 0;  // bytes, all of them so far
  bool finished_ = false;
};

// Lower-case hexadecimal.
std::string ToHex(const Sha256Digest& digest);

}  // namespace llmp::base

#endif  // LLMP_BASE_SHA256_H_
