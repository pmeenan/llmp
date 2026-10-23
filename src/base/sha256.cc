// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "base/sha256.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <string>
#include <string_view>

#include "base/check.h"
#include "base/sha256_internal.h"

namespace llmp::base {
namespace {

std::uint32_t LoadBigEndian(const std::uint8_t* bytes) {
  return (std::uint32_t{bytes[0]} << 24) | (std::uint32_t{bytes[1]} << 16) |
         (std::uint32_t{bytes[2]} << 8) | std::uint32_t{bytes[3]};
}

}  // namespace

void sha256_internal::Scalar(State& state, const std::uint8_t* block, std::size_t count) {
  for (std::size_t index = 0; index < count; ++index, block += 64) {
    std::array<std::uint32_t, 64> w{};
    for (std::size_t i = 0; i < 16; ++i) {
      w[i] = LoadBigEndian(block + (4 * i));
    }
    for (std::size_t i = 16; i < 64; ++i) {
      const std::uint32_t s0 =
          std::rotr(w[i - 15], 7) ^ std::rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
      const std::uint32_t s1 = std::rotr(w[i - 2], 17) ^ std::rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
      w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    std::uint32_t a = state[0];
    std::uint32_t b = state[1];
    std::uint32_t c = state[2];
    std::uint32_t d = state[3];
    std::uint32_t e = state[4];
    std::uint32_t f = state[5];
    std::uint32_t g = state[6];
    std::uint32_t h = state[7];
    for (std::size_t i = 0; i < 64; ++i) {
      const std::uint32_t s1 = std::rotr(e, 6) ^ std::rotr(e, 11) ^ std::rotr(e, 25);
      const std::uint32_t choose = (e & f) ^ (~e & g);
      const std::uint32_t t1 = h + s1 + choose + kRound[i] + w[i];
      const std::uint32_t s0 = std::rotr(a, 2) ^ std::rotr(a, 13) ^ std::rotr(a, 22);
      const std::uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
      const std::uint32_t t2 = s0 + majority;
      h = g;
      g = f;
      f = e;
      e = d + t1;
      d = c;
      c = b;
      b = a;
      a = t1 + t2;
    }
    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
    state[4] += e;
    state[5] += f;
    state[6] += g;
    state[7] += h;
  }
}

sha256_internal::Compressor sha256_internal::Select(bool allow_hardware) {
  if (allow_hardware) {
    if (auto accelerated = ArmCompressor()) return accelerated;
  }
  return Scalar;
}

namespace {
sha256_internal::Compressor Compressor() {
  static const auto selected = sha256_internal::Select(true);
  return selected;
}
}  // namespace

void Sha256::Block(const std::uint8_t* block) { Compressor()(state_, block, 1); }

Sha256& Sha256::Update(std::span<const std::byte> bytes) {
  Check(!finished_, "SHA-256 updated after Finish");
  // An empty span may have no data pointer, which memcpy may not be given.
  if (bytes.empty()) {
    return *this;
  }
  const auto* data = reinterpret_cast<const std::uint8_t*>(bytes.data());
  std::size_t size = bytes.size();
  length_ += size;
  if (buffered_ > 0) {
    const std::size_t take = std::min(size, buffer_.size() - buffered_);
    std::memcpy(buffer_.data() + buffered_, data, take);
    buffered_ += take;
    data += take;
    size -= take;
    if (buffered_ < buffer_.size()) {
      return *this;
    }
    Block(buffer_.data());
    buffered_ = 0;
  }
  const auto blocks = size / buffer_.size();
  if (blocks != 0) {
    Compressor()(state_, data, blocks);
    const auto consumed = blocks * buffer_.size();
    data += consumed;
    size -= consumed;
  }
  if (size > 0) {
    std::memcpy(buffer_.data(), data, size);
    buffered_ = size;
  }
  return *this;
}

Sha256Digest Sha256::Finish() {
  Check(!finished_, "SHA-256 finished twice");
  // The message length in bits, as FIPS 180-4 appends it, taken before
  // the padding that Update would count.
  const std::uint64_t bits = length_ * 8;
  static constexpr std::array<std::uint8_t, 64> kPad = {0x80};
  const std::size_t pad = buffered_ < 56 ? 56 - buffered_ : 120 - buffered_;
  Update(std::as_bytes(std::span(kPad.data(), pad)));
  std::array<std::uint8_t, 8> length{};
  for (std::size_t i = 0; i < 8; ++i) {
    length[i] = static_cast<std::uint8_t>(bits >> (56 - (8 * i)));
  }
  Update(std::as_bytes(std::span(length)));
  finished_ = true;
  Sha256Digest digest{};
  for (std::size_t i = 0; i < 8; ++i) {
    digest[4 * i] = static_cast<std::uint8_t>(state_[i] >> 24);
    digest[(4 * i) + 1] = static_cast<std::uint8_t>(state_[i] >> 16);
    digest[(4 * i) + 2] = static_cast<std::uint8_t>(state_[i] >> 8);
    digest[(4 * i) + 3] = static_cast<std::uint8_t>(state_[i]);
  }
  return digest;
}

std::string ToHex(const Sha256Digest& digest) {
  static constexpr std::string_view kDigits = "0123456789abcdef";
  std::string hex;
  hex.reserve(2 * digest.size());
  for (const std::uint8_t byte : digest) {
    hex.push_back(kDigits[byte >> 4]);
    hex.push_back(kDigits[byte & 0xf]);
  }
  return hex;
}

}  // namespace llmp::base
