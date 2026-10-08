// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "plan_record.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <format>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace llmp::test_support {
namespace {

// A JSON string: quotes, backslashes and control characters escaped.
std::string Quoted(std::string_view text) {
  std::string out = "\"";
  for (const char c : text) {
    if (c == '"' || c == '\\') {
      out += '\\';
      out += c;
    } else if (static_cast<unsigned char>(c) < 0x20) {
      out += std::format("\\u{:04x}", static_cast<unsigned>(static_cast<unsigned char>(c)));
    } else {
      out += c;
    }
  }
  return out + "\"";
}

std::string Dims(const std::array<unsigned, 3>& dims) {
  return std::format("[{},{},{}]", dims[0], dims[1], dims[2]);
}

std::string Stream(const void* stream) { return Quoted(std::format("{}", stream)); }

}  // namespace

std::string NormalizedKernelName(std::string_view name) {
  std::string out(name);
  constexpr std::string_view kToken = "_INTERNAL_";
  for (std::size_t at = out.find(kToken); at != std::string::npos;
       at = out.find(kToken, at + kToken.size())) {
    std::size_t p = at + kToken.size();
    if (p + 8 > out.size()) {
      break;
    }
    out.replace(p, 8, "xxxxxxxx");
    p += 9;  // the hash and '_'
    std::size_t length = 0;
    while (p < out.size() && out[p] >= '0' && out[p] <= '9') {
      length = (length * 10) + static_cast<std::size_t>(out[p] - '0');
      ++p;
    }
    p += 1 + length + 1;  // '_', the file, '_'
    if (p + 8 > out.size()) {
      break;
    }
    out.replace(p, 8, "xxxxxxxx");
  }
  return out;
}

std::string HeaderLine(std::string_view source,
                       const std::vector<std::pair<std::string, std::string>>& loaded) {
  std::string libraries;
  for (const auto& [name, path] : loaded) {
    libraries += (libraries.empty() ? "" : ",") + Quoted(name) + ":" + Quoted(path);
  }
  return std::format(
      R"({{"type":"header","format":"llmp-plan-record/1","source":{},"loaded":{{{}}}}})"
      "\n",
      Quoted(source), libraries);
}

std::string ChunkLine(const Chunk& chunk) {
  return std::format(R"({{"type":"chunk","evaluation":{},"chunk":{},"rows":{},"n_past":{}}})"
                     "\n",
                     chunk.evaluation, chunk.chunk, chunk.rows, chunk.n_past);
}

std::string EndChunkLine() { return "{\"type\":\"end_chunk\"}\n"; }

std::string OpLine(std::string_view name, int layer) {
  return std::format(R"({{"type":"op","name":{},"layer":{}}})"
                     "\n",
                     Quoted(name), layer);
}

std::string EventLine(const Event& event) {
  switch (event.kind) {
    case EventKind::kKernel:
      return std::format(R"({{"type":"kernel","name":{},"api":{},"grid":{},"block":{},"shared":{},)"
                         R"("registers":{},"static_shared":{},"local":{},"stream":{}}})"
                         "\n",
                         Quoted(event.name), Quoted(event.api), Dims(event.grid), Dims(event.block),
                         event.shared, event.registers, event.static_shared, event.local,
                         Stream(event.stream));
    case EventKind::kCopy:
      return std::format(R"({{"type":"memcpy","kind":{},"bytes":{},"stream":{}}})"
                         "\n",
                         Quoted(event.api), event.bytes, Stream(event.stream));
    case EventKind::kMemset:
      return std::format(R"({{"type":"memset","bytes":{},"value":{},"stream":{}}})"
                         "\n",
                         event.bytes, event.value, Stream(event.stream));
    case EventKind::kCublas: {
      std::string line =
          std::format(R"({{"type":"cublas","function":{},"m":{},"n":{},"k":{})", Quoted(event.name),
                      event.shape[0], event.shape[1], event.shape[2]);
      if (event.shape[3] != 0) {
        line += std::format(R"(,"batch":{})", event.shape[3]);
      }
      return line + "}\n";
    }
  }
  return {};
}

}  // namespace llmp::test_support
