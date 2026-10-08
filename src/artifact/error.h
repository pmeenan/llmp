// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Why an artifact was refused (D-066): the violated rule, named as
// docs/artifact-format.md's "Verification and rejection" names it, a static
// reason, and which item (a shard, group, resource, file or byte offset,
// by index) it concerns. Nothing from the artifact's own text is copied into
// an error, so creating one never allocates and never carries untrusted
// strings.

#ifndef LLMP_ARTIFACT_ERROR_H_
#define LLMP_ARTIFACT_ERROR_H_

#include <cstdint>
#include <limits>
#include <string>
#include <string_view>

namespace llmp::artifact {

enum class Rule : std::uint8_t {
  kIo,                  // a system call failed
  kFormat,              // not a llmpalooza artifact at all
  kUnsupportedVersion,  // re-import
  kUnsupportedProfile,  // re-import
  kJson,                // not strict JSON, or over a parser cap
  kCanonical,           // not the one canonical encoding
  kSchema,              // unknown, missing or mistyped keys and values
  kIdentity,            // the name is not the manifest's digest
  kFileType,            // a link, a hard-linked or non-regular file
  kFileSet,             // files listed and present differ
  kPath,                // a listed path does not fit its role
  kFileSize,            // a size differs from the manifest, or a cap
  kHash,                // a digest differs
  kBounds,              // tiling, chunk numbering, ranges, caps
  kAlignment,           // a resource off its 256-byte alignment
  kOverlap,             // readable ranges overlap
  kRepr,                // a representation unknown, unsupported or inconsistent
  kExpertArray,         // expert groups and arrays disagree
  kContainer,           // a shard header differs from the one the index implies
  kUntrusted,           // the reader's owner policy, not a format rule
  kState,               // a read plan's caller named a chunk both missing and resident
};

// The rule's name as artifact-format.md and the prototype verifier spell it
// ("file-set").
std::string_view RuleName(Rule rule);

inline constexpr std::uint64_t kNoItem = std::numeric_limits<std::uint64_t>::max();

struct Error {
  Rule rule = Rule::kSchema;
  std::string_view reason;  // static text
  std::uint64_t item = kNoItem;

  // "bounds: group offset or size (item 7)".
  std::string ToString() const;
};

}  // namespace llmp::artifact

#endif  // LLMP_ARTIFACT_ERROR_H_
