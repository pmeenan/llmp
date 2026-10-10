// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// A kept conversation's record (D-105; docs/runtime-serving.md#state-kept-
// across-a-restart): what lets a new process adopt a request slot's
// conversation that the one before it spilled, instead of prefilling it
// again. A durable on-disk format in two versions, each refused whole
// unless every field validates:
//
// - version 1 (the M3 runners'): one spill file a slot, `slot-<N>.state`,
//   holding the slot's regions one after another;
// - version 2 (the shared runner's, D-112): one spill file a stage (the
//   target, then each drafter: a region each), `slot-<N>.state` for the
//   target and `slot-<N>.stage-<S>.state` for stage S.
//
// The reader takes both; a model expects exactly the version it writes
// (Expected::version), so a record of the other version is refused and
// discarded, never migrated (engine-components.md, "Execution").
//
// One record a request slot, `slot-<N>.record` beside the slot's spill
// files and its turn checkpoints' files (`slot-<N>.turn-<K>.state`), in
// the model's private directory beneath the spill role
// (`conversations/<artifact>/`, created 0700; every file created 0600,
// D-014; ones found are used, D-105 as amended). It is
// strict JSON, written whole and atomically (a temporary file renamed over
// it), and ends with the SHA-256 of everything before it, so a record cut
// short or edited is refused:
//
//   {"format":"llmp-kept-conversation","version":1,
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
//   {"format":"llmp-kept-conversation","version":2,
//    "build":B,"artifact":A,"drafter":D,"layout":L,"slot":N,
//    "stages":[{"file":F,"device":N,"inode":N,"generation":N|null,
//      "file_bytes":N},...],
//    "extents":[[stage,index,"sha256"],...],
//    "tokens":...,"cursor":...,"decoding":...,"used_unix_ms":...,
//    "checkpoints":[...],"digest":"sha256"}
//
//   (version 2's checkpoint ranges and footprints name the stage as their
//   region; each stage's file holds that stage's region alone, so its
//   region's bytes are its file's.)
//
// - Identity: the build that wrote it (its version, commit and the
//   executable file's identity, so another build or a reinstalled one
//   never reads it), the artifact and drafter, and the runner's state
//   layout (version 1: its architecture, context, regions and their
//   bytes; version 2: model::KeptLayoutId, the spec bound to its artifacts
//   and every stage's layout). A new process adopts only what it would
//   have written itself.
// - The state: each spill file (its device, inode and inode generation,
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

#ifndef LLMP_RUNTIME_KEPT_RECORD_H_
#define LLMP_RUNTIME_KEPT_RECORD_H_

#include <cstddef>
#include <cstdint>
#include <expected>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "base/sha256.h"
#include "config/storage_roles.h"

namespace llmp::runtime {
class RequestMemory;
class MemoryCharge;
}  // namespace llmp::runtime

namespace llmp::runtime::kept {

inline constexpr std::string_view kFormat = "llmp-kept-conversation";
// One spill file a slot (the M3 runners', until each family migrates).
inline constexpr std::uint32_t kOneFileVersion = 1;
// One spill file a stage (the shared runner's, D-112).
inline constexpr std::uint32_t kStageFilesVersion = 2;
// The newest version this build writes.
inline constexpr std::uint32_t kVersion = kStageFilesVersion;
// The most stages (spill files) a version 2 record names.
inline constexpr std::size_t kMostStages = 16;
// Beneath the spill role.
inline constexpr std::string_view kDirectory = config::kKeptConversationsDirectory;
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

// A spill file the record names: its name beneath the model's directory,
// its identity and its size.
struct StateFile {
  std::string name;
  FileId id;
  std::uint64_t bytes = 0;
  bool operator==(const StateFile&) const = default;
};

struct Record {
  std::uint32_t version = kOneFileVersion;
  Identity identity;
  std::uint32_t slot = 0;
  // Version 1: one, holding every region in order; version 2: one a
  // region (a stage), whose bytes are its region's.
  std::vector<StateFile> files;
  std::vector<std::uint64_t> regions;  // each region's bytes in its file (whole extents)
  std::vector<Extent> extents;         // the used ones, in region and index order
  std::vector<std::int32_t> tokens;
  std::uint32_t cursor = 0;
  std::vector<std::uint64_t> decoding;
  std::int64_t used_unix_ms = 0;
  std::vector<Checkpoint> checkpoints;
  bool operator==(const Record&) const = default;
};

// File names beneath a model's directory: a slot's spill file (stage 0's,
// or version 1's only one: `slot-<N>.state`; stage S's from 1:
// `slot-<N>.stage-<S>.state`), its record and its turn checkpoints.
std::string StateFileName(std::uint32_t slot, std::uint32_t stage = 0);
std::string RecordFileName(std::uint32_t slot);
std::string CheckpointFileName(std::uint32_t slot, std::uint64_t serial);
// What a name is: a slot's state (its stage), record or checkpoint (its
// slot), or none of them (empty).
struct NameOf {
  enum class Kind : std::uint8_t { kState, kRecord, kCheckpoint };
  Kind kind = Kind::kState;
  std::uint32_t slot = 0;
  std::uint32_t stage = 0;   // a state file's
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
// adopting process (Check). With `memory`, its token history is charged
// to `token_charge`; `memory_refused` is set when that charge refused it
// (the record may fit a later start).
std::expected<Record, std::string> Decode(std::string_view text, RequestMemory* memory = nullptr,
                                          MemoryCharge* token_charge = nullptr,
                                          bool* memory_refused = nullptr);

// Where an extent lies in a version 1 spill file (the regions one after
// another).
std::uint64_t ExtentOffset(std::span<const std::uint64_t> regions, const Extent& extent);
// Where an extent lies: the index of its file in Record::files, and its
// offset there (version 1: ExtentOffset; version 2: its region's file,
// from 0). For a record Decode returned.
struct ExtentAt {
  std::size_t file = 0;
  std::uint64_t offset = 0;
};
ExtentAt ExtentPlace(const Record& record, const Extent& extent);
// A checkpoint page's place in its file: its offset and transfer bytes.
struct Place {
  std::uint64_t offset = 0;
  std::uint64_t bytes = 0;
};
std::vector<Place> CheckpointPlaces(const Checkpoint& checkpoint);
// What of the record's spill file `file` (an index into Record::files) its
// extents do not cover: each run of unlisted extents, region by region, in
// file order (adjacent runs joined). Adopting the record empties them
// (they read as zeros, as a fresh slot's do): a removed record can outlive
// a crash, its listed extents still matching while the others hold what
// the file took since.
std::vector<Place> UnlistedPlaces(const Record& record, std::size_t file = 0);

// What the adopting process holds a record against.
struct Expected {
  // The version its model writes: the M3 runners' 1 by default, the
  // shared runner's 2 (Llm::KeptVersion).
  std::uint32_t version = kOneFileVersion;
  Identity identity;
  std::uint32_t slot = 0;
  std::vector<std::uint64_t> regions;  // the slot's regions' bytes in their files
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

}  // namespace llmp::runtime::kept

#endif  // LLMP_RUNTIME_KEPT_RECORD_H_
