// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// A kept conversation's record (D-105; docs/runtime-serving.md#state-kept-
// across-a-restart): what lets a new process adopt a request slot's
// conversation that the one before it spilled, instead of prefilling it
// again. A durable on-disk format, version 1, refused whole unless every
// field validates.
//
// One record a request slot, `slot-<N>.record` beside the slot's spill
// file `slot-<N>.state` and its turn checkpoints' files
// (`slot-<N>.turn-<K>.state`), in the model's private directory beneath
// the spill role (`conversations/<artifact>/`, 0700; every file 0600,
// D-014). It is strict JSON, written whole and atomically (a temporary
// file renamed over it), and ends with the SHA-256 of everything before
// it, so a record cut short or edited is refused:
//
//   {"format":"jitllm-kept-conversation","version":1,
//    "build":B,"artifact":A,"drafter":D,"layout":L,"slot":N,
//    "file":F,"device":N,"inode":N,"generation":N|null,"file_bytes":N,
//    "regions":[N,...],"extents":[[region,index,"sha256"],...],
//    "tokens":[N,...],"cursor":N,"decoding":["hex64",...],"used_unix_ms":N,
//    "checkpoints":[{"file":F,"device":N,"inode":N,"generation":N|null,
//      "file_bytes":N,"position":N,"created_unix_ms":N,
//      "ranges":[[region,offset,bytes],...],"footprint":[[...],...],
//      "cursor":N,"decoding":["hex64",...],"digests":["sha256",...]},...],
//    "digest":"sha256"}
//
// - Identity: the build that wrote it (its version, commit and the
//   executable file's identity, so another build or a reinstalled one
//   never reads it), the artifact and drafter, and the runner's state
//   layout (its architecture, context, regions and their bytes). A new
//   process adopts only what it would have written itself.
// - The state: the spill file (its device, inode and inode generation,
//   and size: the very file the record was made for) and the 2 MiB
//   extents of its regions the conversation used, each with its SHA-256 as
//   the file held it.
// - The conversation: every token the state has seen, the speculation
//   cursor and adaptive draft depth that go with it, and when it was last
//   used (wall-clock milliseconds, for retention across the restart).
// - Its turn checkpoints, each a file of whole state pages (engine/
//   checkpoint_file.h) with its own identity and digests: one that does not
//   validate is dropped alone.
//
// Vendor-free and host-only: encoding, decoding and the checks against the
// adopting process are pure functions the CPU tests drive.

#ifndef JITLLM_RUNTIME_KEPT_RECORD_H_
#define JITLLM_RUNTIME_KEPT_RECORD_H_

#include <cstddef>
#include <cstdint>
#include <expected>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "base/sha256.h"

namespace jitllm::runtime {
class RequestMemory;
class MemoryCharge;
}  // namespace jitllm::runtime

namespace jitllm::runtime::kept {

inline constexpr std::string_view kFormat = "jitllm-kept-conversation";
inline constexpr std::uint32_t kVersion = 1;
// Beneath the spill role.
inline constexpr std::string_view kDirectory = "conversations";
// The state's pages, as its spill file holds them.
inline constexpr std::uint64_t kExtentBytes = std::uint64_t{2} << 20U;
// A checkpoint page's place in its file is its bytes rounded up to this.
inline constexpr std::uint64_t kFileAlignment = 4096;
// The largest record read: a 1,048,576-token conversation's tokens, its
// extents' digests and its checkpoints fit many times over.
inline constexpr std::size_t kMostRecordBytes = std::size_t{64} << 20U;

using Digest = base::Sha256Digest;

struct FileId {
  std::uint64_t device = 0;
  std::uint64_t inode = 0;
  std::optional<std::uint64_t> generation;
  bool operator==(const FileId&) const = default;
};

struct Range {
  std::uint32_t region = 0;
  std::uint64_t offset = 0;
  std::uint64_t bytes = 0;
  bool operator==(const Range&) const = default;
};

struct Extent {
  std::uint32_t region = 0;
  std::uint32_t index = 0;  // the region's 2 MiB extent
  Digest digest{};
  bool operator==(const Extent&) const = default;
};

struct Checkpoint {
  std::string file;
  FileId id;
  std::uint64_t file_bytes = 0;
  std::uint32_t position = 0;  // the tokens of the conversation it holds
  std::int64_t created_unix_ms = 0;
  std::vector<Range> ranges;  // its pages, in file order
  std::vector<Range> footprint;
  std::uint32_t cursor = 0;
  std::vector<std::uint64_t> decoding;
  std::vector<Digest> digests;  // one a page, over its place in the file
  bool operator==(const Checkpoint&) const = default;
};

// What a record must match to be adopted.
struct Identity {
  std::string build;
  std::string artifact;
  std::string drafter;  // empty without one
  std::string layout;
  bool operator==(const Identity&) const = default;
};

struct Record {
  Identity identity;
  std::uint32_t slot = 0;
  std::string file;
  FileId id;
  std::uint64_t file_bytes = 0;
  std::vector<std::uint64_t> regions;  // each region's bytes in the file (whole extents)
  std::vector<Extent> extents;         // the used ones, in region and index order
  std::vector<std::int32_t> tokens;
  std::uint32_t cursor = 0;
  std::vector<std::uint64_t> decoding;
  std::int64_t used_unix_ms = 0;
  std::vector<Checkpoint> checkpoints;
  bool operator==(const Record&) const = default;
};

// File names beneath a model's directory.
std::string StateFileName(std::uint32_t slot);
std::string RecordFileName(std::uint32_t slot);
std::string CheckpointFileName(std::uint32_t slot, std::uint64_t serial);
// What a name is: a slot's state, record or checkpoint (its slot), or
// none of them (empty).
struct NameOf {
  enum class Kind : std::uint8_t { kState, kRecord, kCheckpoint };
  Kind kind = Kind::kState;
  std::uint32_t slot = 0;
  std::uint64_t serial = 0;  // a checkpoint's
};
std::optional<NameOf> ParseFileName(std::string_view name);
// Whether `id` may name a model's directory: lower-case hexadecimal, 1 to
// 128 characters (an artifact's ID).
bool ValidDirectoryName(std::string_view id);

// The record as written, digest included.
std::string Encode(const Record& record);
// A record read back: well-formed, every field present with its type and
// nothing else, and its digest its text's. Not yet checked against the
// adopting process (Check).
std::expected<Record, std::string> Decode(std::string_view text, RequestMemory* memory = nullptr,
                                          MemoryCharge* token_charge = nullptr);

// Where an extent lies in its spill file.
std::uint64_t ExtentOffset(std::span<const std::uint64_t> regions, const Extent& extent);
// A checkpoint page's place in its file: its offset and transfer bytes.
struct Place {
  std::uint64_t offset = 0;
  std::uint64_t bytes = 0;
};
std::vector<Place> CheckpointPlaces(const Checkpoint& checkpoint);
// What of the record's spill file its extents do not cover: each run of
// unlisted extents, region by region, in file order (adjacent runs
// joined). Adopting the record empties them (they read as zeros, as a
// fresh slot's do): a removed record can outlive a crash, its listed
// extents still matching while the others hold what the file took since.
std::vector<Place> UnlistedPlaces(const Record& record);

// What the adopting process holds a record against.
struct Expected {
  Identity identity;
  std::uint32_t slot = 0;
  std::vector<std::uint64_t> regions;  // the slot's regions' bytes in the file
  std::vector<std::uint64_t> layouts;  // each region's layout bytes (a range's bound)
  std::uint32_t context = 0;           // positions a conversation may hold
  std::uint32_t vocabulary = 0;        // token IDs below it
  std::int64_t now_unix_ms = 0;
  std::int64_t retention_ms = 0;  // used or created longer ago is expired
};
// Everything but the files' contents: the identity, the slot, the layout,
// the extents and checkpoint ranges inside it, the tokens, retention.
// A checkpoint that fails its own checks is removed from `record` (its
// file is then not adopted), with why in `dropped`; anything else refuses
// the record.
std::expected<void, std::string> Check(Record& record, const Expected& expected,
                                       std::vector<std::string>* dropped = nullptr);

}  // namespace jitllm::runtime::kept

#endif  // JITLLM_RUNTIME_KEPT_RECORD_H_
