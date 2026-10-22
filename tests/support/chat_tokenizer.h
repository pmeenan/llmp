// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
#ifndef LLMP_TESTS_SUPPORT_CHAT_TOKENIZER_H_
#define LLMP_TESTS_SUPPORT_CHAT_TOKENIZER_H_
#include <cstdint>
#include <string>
#include <string_view>
namespace llmp::test_support::chat_tokenizer {
// An independent small ModelProto writer: one unknown, BOS/EOS, byte
// fallback and two normal pieces. Omitting scores supplies protobuf's 0.
inline std::string Varint(std::uint64_t value) {
  std::string bytes;
  do {
    const auto low = value & 127U;
    value >>= 7U;
    bytes.push_back(static_cast<char>(low | (value != 0 ? 128U : 0U)));
  } while (value != 0);
  return bytes;
}
inline std::string Scalar(unsigned field, std::uint64_t value) {
  return Varint(std::uint64_t{field} << 3U) + Varint(value);
}
inline std::string Bytes(unsigned field, std::string_view bytes) {
  return Varint((std::uint64_t{field} << 3U) | 2U) + Varint(bytes.size()) + std::string(bytes);
}
inline std::string Model(std::string_view normal = "a") {
  std::string model;
  const auto piece = [&](std::string_view text, unsigned kind) {
    model += Bytes(1, Bytes(1, text) + Scalar(3, kind));
  };
  piece("<unk>", 2);
  piece("<bos>", 3);
  piece("<eos>", 3);
  constexpr std::string_view hex = "0123456789ABCDEF";
  for (unsigned b = 0; b < 256; ++b) {
    std::string text = "<0x00>";
    text[3] = hex[b >> 4U];
    text[4] = hex[b & 15U];
    piece(text, 6);
  }
  piece("▁", 1);
  piece(normal, 1);
  model += Bytes(2, Scalar(3, 2) + Scalar(35, 1) + Scalar(40, 0) + Scalar(41, 1) + Scalar(42, 2));
  model += Bytes(3, Bytes(1, "identity") + Scalar(3, 0) + Scalar(4, 0));
  return model;
}

}  // namespace llmp::test_support::chat_tokenizer
#endif  // LLMP_TESTS_SUPPORT_CHAT_TOKENIZER_H_
