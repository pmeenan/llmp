// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "artifact/error.h"

#include <format>
#include <string>
#include <string_view>

namespace llmp::artifact {

std::string_view RuleName(Rule rule) {
  switch (rule) {
    case Rule::kIo:
      return "io";
    case Rule::kFormat:
      return "format";
    case Rule::kUnsupportedVersion:
      return "unsupported-version";
    case Rule::kUnsupportedProfile:
      return "unsupported-profile";
    case Rule::kJson:
      return "json";
    case Rule::kCanonical:
      return "canonical";
    case Rule::kSchema:
      return "schema";
    case Rule::kIdentity:
      return "identity";
    case Rule::kFileType:
      return "file-type";
    case Rule::kFileSet:
      return "file-set";
    case Rule::kPath:
      return "path";
    case Rule::kFileSize:
      return "file-size";
    case Rule::kHash:
      return "hash";
    case Rule::kBounds:
      return "bounds";
    case Rule::kAlignment:
      return "alignment";
    case Rule::kOverlap:
      return "overlap";
    case Rule::kRepr:
      return "repr";
    case Rule::kExpertArray:
      return "expert-array";
    case Rule::kContainer:
      return "container";
    case Rule::kUntrusted:
      return "untrusted";
    case Rule::kState:
      return "state";
  }
  return "unknown";
}

std::string Error::ToString() const {
  if (item == kNoItem) {
    return std::format("{}: {}", RuleName(rule), reason);
  }
  return std::format("{}: {} (item {})", RuleName(rule), reason, item);
}

}  // namespace llmp::artifact
