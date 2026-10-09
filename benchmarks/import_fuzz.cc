// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// A fuzzing entry for the importer's container readers (import/) and the
// reference decoder of what they describe: the first byte picks GGUF, a
// safetensors shard under a fixed config.json, a config.json and an MLX or
// an EXL3 shard, a diffusers model_index.json, a config.json, an
// hf_quant_config.json and a shard (each document after a 2-byte length),
// or an affine tensor's parameters and part bytes; the rest is the input.
// A checkpoint a reader accepts must be consistent (every extent inside
// its file, of its format's part size, overlapping no other; at most one
// part role a tensor) or the entry aborts, and it is turned into its
// mapping source; its quantized tensors are decoded by the reference from
// the shard's bytes, every part found by its module. Built as a plain
// driver that runs each file named on its command line once; to fuzz,
// build this file with -DLLMP_LIBFUZZER and clang's -fsanitize=fuzzer,
// which brings its own main. Never run by CTest.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <expected>
#include <fstream>
#include <iterator>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include "format/encoding.h"
#include "format/tensor_format.h"
#include "import/checkpoint.h"
#include "import/error.h"
#include "import/readers/diffusers.h"
#include "import/readers/exl3.h"
#include "import/readers/gguf.h"
#include "import/readers/mlx.h"
#include "import/readers/safetensors.h"
#include "kernels/reference/decode.h"

namespace {

namespace format = llmp::format;
namespace ref = llmp::kernels::reference;
using llmp::import::Checkpoint;
using llmp::import::Error;
using llmp::import::Tensor;

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

void Check(bool ok) {
  if (!ok) {
    std::abort();  // an accepted checkpoint broke an invariant
  }
}

// Every extent inside its file, its part's size under its format, no two
// overlapping.
void ExpectConsistent(const Checkpoint& c) {
  std::vector<std::tuple<std::uint64_t, std::uint64_t, std::uint64_t>> spans;  // file, offset, end
  for (const Tensor& t : c.tensors) {
    Check(t.extent.file < c.files.size());
    const std::uint64_t size = c.files[t.extent.file].bytes;
    Check(t.extent.offset <= size && t.extent.bytes <= size - t.extent.offset);
    const auto parts = format::PartsOf(t.format);
    Check(parts.has_value());
    const auto part = parts->Find(t.part);
    Check(part.has_value() && part->bytes == t.extent.bytes);
    if (t.extent.bytes != 0) {
      spans.emplace_back(t.extent.file, t.extent.offset, t.extent.offset + t.extent.bytes);
    }
  }
  std::ranges::sort(spans);
  for (std::size_t i = 1; i < spans.size(); ++i) {
    const auto& [file, offset, end] = spans[i];
    const auto& [previous_file, previous_offset, previous_end] = spans[i - 1];
    Check(file != previous_file || offset >= previous_end);
  }
}

// Each quantized tensor of a one-shard checkpoint decoded from the shard:
// its parts are the tensors of its format with the same name up to the
// last '.', one per role.
void DecodeQuantized(const Checkpoint& c, std::span<const std::byte> shard) {
  std::map<std::string_view, std::vector<const Tensor*>> modules;
  for (const Tensor& t : c.tensors) {
    if (!format::Info(t.format.encoding).quantized) {
      continue;
    }
    const std::string_view name = t.name;
    modules[name.substr(0, name.rfind('.'))].push_back(&t);
  }
  // MLX's and EXL3's modules are named by their parts' suffixes alone
  // (ModelOpt's may share a tensor scale across modules).
  const bool by_suffix = c.container == "mlx" || c.container == "exl3";
  for (const auto& [module, tensors] : modules) {
    const format::TensorFormat& f = tensors.front()->format;
    std::vector<ref::PartView> views;
    bool same = true;
    for (const Tensor* t : tensors) {
      same = same && t->format == f;
      Check(!by_suffix || same);  // a module's parts share their format
      Check(std::ranges::none_of(views, [&](const ref::PartView& v) { return v.role == t->part; }));
      views.push_back({.role = t->part, .bytes = shard.subspan(t->extent.offset, t->extent.bytes)});
    }
    if (!same) {
      continue;
    }
    const auto parts = format::PartsOf(f);
    if (!parts || views.size() != parts->size() || f.shape.ne[1] > 64 || f.shape.ne[0] > 4096) {
      continue;  // incomplete here, or too large to decode in a fuzz step
    }
    std::vector<float> out(f.shape.ne[0] * f.shape.ne[1]);
    static_cast<void>(ref::DecodeRows(f, views, 0, 0, f.shape.ne[1], out));
  }
}

void Use(const std::expected<Checkpoint, Error>& c,
         std::optional<std::span<const std::byte>> shard = std::nullopt) {
  if (c) {
    ExpectConsistent(*c);
    if (shard) {
      DecodeQuantized(*c, *shard);
    }
    const auto source = llmp::import::SourceOf(*c);
    const auto values = llmp::import::ValuesOf(*c);
    static_cast<void>(source.tensors.size() + values.size());
  } else {
    static_cast<void>(c.error().ToString());
  }
}

std::uint64_t Byte(std::span<const std::byte>& rest) {
  if (rest.empty()) {
    return 0;
  }
  const auto b = std::to_integer<std::uint64_t>(rest[0]);
  rest = rest.subspan(1);
  return b;
}

// An affine tensor from fuzzed parameters, its parts cut from the rest of
// the input (or zeros past its end), decoded whole when valid.
void DecodeAffine(std::span<const std::byte> rest) {
  format::TensorFormat f{.encoding = format::Encoding::kAffine, .layout = format::Layout::kSplit};
  f.params.bits = static_cast<std::uint8_t>(Byte(rest) % 10);
  f.params.group = static_cast<std::uint32_t>(Byte(rest) * 8 % 264);
  f.params.scale = static_cast<format::AffineScale>(Byte(rest) % 5);
  f.params.zero_point = Byte(rest) % 2 == 1;
  const std::uint64_t k = (Byte(rest) + 1) * 8;
  const std::uint64_t n = (Byte(rest) % 8) + 1;
  const std::uint64_t e = (Byte(rest) % 3) + 1;
  f.shape = {.ne = {k, n, e, 1}, .rank = static_cast<std::uint8_t>(e > 1 ? 3 : 2)};
  const auto parts = format::PartsOf(f);
  if (!parts) {
    return;
  }
  std::vector<std::vector<std::byte>> bytes;
  std::vector<ref::PartView> views;
  for (const format::PartExtent& p : parts->parts()) {
    std::vector<std::byte>& b = bytes.emplace_back(p.bytes);
    const std::size_t take = std::min<std::size_t>(b.size(), rest.size());
    std::ranges::copy(rest.first(take), b.begin());
    rest = rest.subspan(take);
  }
  for (std::size_t i = 0; i < bytes.size(); ++i) {
    views.push_back({.role = parts->parts()[i].role, .bytes = bytes[i]});
  }
  std::vector<float> out(k * n);
  for (std::uint64_t m = 0; m < e; ++m) {
    Check(ref::DecodeRows(f, views, m, 0, n, out).has_value());
  }
}

}  // namespace

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
  if (size == 0) {
    return 0;
  }
  std::span<const std::byte> rest = std::as_bytes(std::span(data + 1, size - 1));
  switch (data[0] % 7) {
    case 0:
      Use(llmp::import::ParseGguf(rest));
      break;
    case 1:
      Use(llmp::import::ParseSafetensors(rest, R"({"model_type":"fuzz"})"), rest);
      break;
    case 2: {
      const std::string_view config = Take(rest);
      Use(llmp::import::ParseMlx(rest, config), rest);
      break;
    }
    case 3: {
      const std::string_view config = Take(rest);
      Use(llmp::import::ParseExl3(rest, config), rest);
      break;
    }
    case 4: {
      const auto index = llmp::import::ParseModelIndex(
          std::string_view(reinterpret_cast<const char*>(rest.data()), rest.size()));
      static_cast<void>(index ? index->roles.size() : index.error().ToString().size());
      break;
    }
    case 5:
      DecodeAffine(rest);
      break;
    default: {
      const std::string_view config = Take(rest);
      const std::string_view hf = Take(rest);
      Use(llmp::import::ParseSafetensors(rest, config, hf), rest);
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
