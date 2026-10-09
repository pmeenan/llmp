// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// A fuzzing entry for the importer's container readers (import/): the
// first byte picks GGUF, a safetensors shard under a fixed config.json, or
// a config.json, an hf_quant_config.json and a shard (each after a 2-byte
// length), and the rest is the input; a checkpoint the reader accepts is
// also turned into its mapping source. Built as a plain driver that runs
// each file named on its command line once; to fuzz, build this file with
// -DLLMP_LIBFUZZER and clang's -fsanitize=fuzzer, which brings its own
// main. Never run by CTest.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <fstream>
#include <iterator>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "import/checkpoint.h"
#include "import/error.h"
#include "import/readers/gguf.h"
#include "import/readers/safetensors.h"

namespace {

std::string_view Take(std::span<const std::byte>& rest) {
  if (rest.size() < 2) {
    rest = {};
    return {};
  }
  const std::size_t n =
      std::min<std::size_t>(rest.size() - 2, std::to_integer<std::size_t>(rest[0]) |
                                                 (std::to_integer<std::size_t>(rest[1]) << 8U));
  const std::string_view text(reinterpret_cast<const char*>(rest.data() + 2), n);
  rest = rest.subspan(2 + n);
  return text;
}

void Use(const std::expected<llmp::import::Checkpoint, llmp::import::Error>& c) {
  if (c) {
    const auto source = llmp::import::SourceOf(*c);
    const auto values = llmp::import::ValuesOf(*c);
    static_cast<void>(source.tensors.size() + values.size());
  } else {
    static_cast<void>(c.error().ToString());
  }
}

}  // namespace

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
  if (size == 0) {
    return 0;
  }
  std::span<const std::byte> rest = std::as_bytes(std::span(data + 1, size - 1));
  switch (data[0] % 3) {
    case 0:
      Use(llmp::import::ParseGguf(rest));
      break;
    case 1:
      Use(llmp::import::ParseSafetensors(rest, R"({"model_type":"fuzz"})"));
      break;
    default: {
      const std::string_view config = Take(rest);
      const std::string_view hf = Take(rest);
      Use(llmp::import::ParseSafetensors(rest, config, hf));
      break;
    }
  }
  return 0;
}

#ifndef LLMP_LIBFUZZER
int main(int argc, char** argv) {
  for (int i = 1; i < argc; ++i) {
    std::ifstream in(argv[i],
                     std::ios::binary);  // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    const std::vector<char> bytes{std::istreambuf_iterator<char>(in),
                                  std::istreambuf_iterator<char>()};
    LLVMFuzzerTestOneInput(reinterpret_cast<const std::uint8_t*>(bytes.data()), bytes.size());
  }
  return 0;
}
#endif
