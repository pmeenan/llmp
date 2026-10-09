// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "import/error.h"

#include <format>
#include <string>
#include <string_view>

namespace llmp::import {

std::string_view RuleName(Rule rule) {
  switch (rule) {
    case Rule::kIo:
      return "io";
    case Rule::kEscape:
      return "escape";
    case Rule::kFileType:
      return "file-type";
    case Rule::kContainer:
      return "container";
    case Rule::kBounds:
      return "bounds";
    case Rule::kUnsupported:
      return "unsupported";
    case Rule::kShards:
      return "shards";
    case Rule::kConfig:
      return "config";
    case Rule::kQuantization:
      return "quantization";
  }
  return "unknown";
}

std::string Error::ToString() const {
  std::string where;
  if (component != kNoItem) {
    where = std::format("component {}", component);
  }
  if (file != kNoItem) {
    where += std::format("{}file {}", where.empty() ? "" : ", ", file);
  }
  if (kind != ItemKind::kNone && item != kNoItem) {
    std::string_view noun = "byte";
    switch (kind) {
      case ItemKind::kNone:
      case ItemKind::kByte:
        break;
      case ItemKind::kTensor:
        noun = "tensor";
        break;
      case ItemKind::kKey:
        noun = "key";
        break;
      case ItemKind::kShard:
        noun = "shard";
        break;
    }
    where += std::format("{}{} {}", where.empty() ? "" : ", ", noun, item);
  }
  if (where.empty()) {
    return std::format("{}: {}", RuleName(rule), reason);
  }
  return std::format("{}: {} ({})", RuleName(rule), reason, where);
}

}  // namespace llmp::import
