// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#ifndef LLMP_ENGINE_CHECKPOINT_FILE_H_
#define LLMP_ENGINE_CHECKPOINT_FILE_H_

#include <cstdint>
#include <expected>
#include <filesystem>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "engine/live_state.h"
#include "platform/kept_files.h"

namespace llmp::engine {

// A mutable byte anywhere in a retained extent makes that whole extent a
// checkpoint range. Bytes beyond the last padded read may be overwritten by
// later turns, too. Used ranges come from LiveState::used_ranges().
std::expected<std::vector<LiveState::Range>, std::string> CheckpointPages(
    std::span<const LiveState::Range> used, std::span<const LiveState::Range> writes);

// The pinned staging one capture or restore uses at a time: one extent
// and room to align it for direct I/O. The runtime sets it apart at start
// (PagedNode::ReserveStaging) so a full budget never starves a checkpoint.
std::uint64_t CheckpointStagingBytes();

struct CheckpointFailure {
  std::string detail;
  // Capture: a device copy has uncertain effects. Restore: the old branch
  // may already have been trimmed or overwritten. Clear before reuse.
  bool invalid_state = false;
  bool cancelled = false;
};

// Two bounded runtime entries own these direct-I/O files: unnamed (a
// process-lifetime cache), or named owner-only files kept across a restart
// with the conversation's record (D-105), removed when the checkpoint is
// dropped in the process and kept when the process ends (Preserve). Each
// page's place in the file is its bytes rounded up to 4 KiB, in order. No
// host payload buffer stays resident between transfers; one cataloged
// extent of staging serves every page.
class CheckpointFile {
 public:
  using Range = LiveState::Range;
  using Copy = std::function<Status(void*, std::span<const Range>)>;
  using Prepare = std::function<Status()>;
  using Continue = std::function<bool()>;

  // Where its file lives: unnamed in `directory`, or with `dir` (an open
  // private directory) the named file `name` beneath it.
  struct Place {
    std::filesystem::path directory;
    int dir = -1;
    std::string name;
  };

  CheckpointFile() = default;
  CheckpointFile(const CheckpointFile&) = delete;
  CheckpointFile& operator=(const CheckpointFile&) = delete;
  CheckpointFile(CheckpointFile&& other) noexcept;
  CheckpointFile& operator=(CheckpointFile&& other) noexcept;
  ~CheckpointFile();

  static std::expected<CheckpointFile, CheckpointFailure> Capture(
      PagedNode& node, const std::filesystem::path& directory, std::span<const Range> ranges,
      const Copy& copy, const Continue& go_on = {});
  static std::expected<CheckpointFile, CheckpointFailure> Capture(PagedNode& node,
                                                                  const Place& place,
                                                                  std::span<const Range> ranges,
                                                                  const Copy& copy,
                                                                  const Continue& go_on = {});
  // A kept checkpoint adopted from the process before (D-105): its named
  // file beneath `dir`, which its record validated, holding `ranges`.
  static std::expected<CheckpointFile, std::string> Adopt(int dir, std::string name,
                                                          std::vector<Range> ranges);
  // The named file's name and identity (none while unnamed), and its size.
  const std::string& name() const { return name_; }
  const std::optional<platform::FileIdentity>& identity() const { return identity_; }
  std::uint64_t file_bytes() const;
  // Its named file stays when this goes: the process is ending and keeps
  // it for the next (a record may name it).
  void Preserve() { preserve_ = true; }
  // Staging allocation happens before prepare mutates the branch. Copy is
  // H2D only: the caller trims/restores the complete footprint once in prepare.
  std::expected<void, CheckpointFailure> Restore(PagedNode& node, const Prepare& prepare,
                                                 const Copy& copy,
                                                 const Continue& go_on = {}) const;
  std::uint64_t bytes() const { return bytes_; }
  const std::vector<Range>& ranges() const { return ranges_; }

 private:
  // Closes the file, and removes a named one unless preserved.
  void Close();

  int fd_ = -1;
  std::uint64_t bytes_ = 0;  // logical payload, excluding file alignment padding
  std::vector<Range> ranges_;
  int dir_ = -1;  // a named file's directory (not owned)
  std::string name_;
  std::optional<platform::FileIdentity> identity_;
  bool preserve_ = false;
};

}  // namespace llmp::engine

#endif  // LLMP_ENGINE_CHECKPOINT_FILE_H_
