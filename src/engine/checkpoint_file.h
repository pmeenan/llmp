// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#ifndef JITLLM_ENGINE_CHECKPOINT_FILE_H_
#define JITLLM_ENGINE_CHECKPOINT_FILE_H_

#include <cstdint>
#include <expected>
#include <filesystem>
#include <functional>
#include <span>
#include <string>
#include <vector>

#include "engine/live_state.h"

namespace jitllm::engine {

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

// Process-lifetime cache, not restart recovery. Two bounded runtime entries
// own these unnamed direct-I/O files. No host payload buffer stays resident
// between transfers; one cataloged extent of staging serves every page.
class CheckpointFile {
 public:
  using Range = LiveState::Range;
  using Copy = std::function<Status(void*, std::span<const Range>)>;
  using Prepare = std::function<Status()>;
  using Continue = std::function<bool()>;

  CheckpointFile() = default;
  CheckpointFile(const CheckpointFile&) = delete;
  CheckpointFile& operator=(const CheckpointFile&) = delete;
  CheckpointFile(CheckpointFile&& other) noexcept;
  CheckpointFile& operator=(CheckpointFile&& other) noexcept;
  ~CheckpointFile();

  static std::expected<CheckpointFile, CheckpointFailure> Capture(
      PagedNode& node, const std::filesystem::path& directory, std::span<const Range> ranges,
      const Copy& copy, const Continue& go_on = {});
  // Staging allocation happens before prepare mutates the branch. Copy is
  // H2D only: the caller trims/restores the complete footprint once in prepare.
  std::expected<void, CheckpointFailure> Restore(PagedNode& node, const Prepare& prepare,
                                                 const Copy& copy,
                                                 const Continue& go_on = {}) const;
  std::uint64_t bytes() const { return bytes_; }
  const std::vector<Range>& ranges() const { return ranges_; }

 private:
  int fd_ = -1;
  std::uint64_t bytes_ = 0;  // logical payload, excluding file alignment padding
  std::vector<Range> ranges_;
};

}  // namespace jitllm::engine

#endif  // JITLLM_ENGINE_CHECKPOINT_FILE_H_
