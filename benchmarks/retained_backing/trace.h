// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The cross-model swap trace (docs/experiments/retained-backing/README.md),
// as the retained-backing replay reads it, and its fragmentation-free
// reference (swap_trace.py's Reference, mirrored exactly).
//
// Load checks the trace's identity before replaying anything: the
// manifest's SHA-256 must be the recorded one for the chosen seed, and the
// file's the manifest's. The confirmation seed is refused unless asked for
// by name: the criteria reserve it for the winner (D-079).

#ifndef LLMP_BENCHMARKS_RETAINED_BACKING_TRACE_H_
#define LLMP_BENCHMARKS_RETAINED_BACKING_TRACE_H_

#include <array>
#include <cstdint>
#include <expected>
#include <map>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace llmp::rb {

using GroupId = std::uint32_t;

inline constexpr std::uint64_t kFileAlign = 4096;  // group alignment (D-056)
inline constexpr std::uint64_t kChunk = std::uint64_t{2} << 20U;
inline constexpr std::uint64_t kGiB = std::uint64_t{1} << 30U;

// The recorded identities (README.md#identity).
inline constexpr std::string_view kPrimaryManifest =
    "44f9f2b40bb9ed8bab7ac337c5736de08dfdf2e1821ae2755c7a78bb191b7fc9";
inline constexpr std::string_view kConfirmationManifest =
    "8aee64a56095a1ef0f3806197e173ae405c1173f007ae417b6bcbec6aaec3f6b";
// The budget files, in the criteria's session order (5/4, 3/2, 2).
inline constexpr std::array<std::string_view, 3> kTraceFiles = {
    "trace-r5-4.jsonl", "trace-r3-2.jsonl", "trace-r2-1.jsonl"};

struct Group {
  std::uint32_t model = 0;
  bool expert = false;
  std::uint64_t used = 0;
  std::uint64_t stored = 0;  // a multiple of kFileAlign
};

struct Model {
  std::string name;
  GroupId first = 0;
  GroupId end = 0;
};

enum class Ev : std::uint8_t { kRequest, kLease, kRelease, kUse, kShrink, kGrow, kEvict, kRestore };

struct Event {
  Ev ev = Ev::kRequest;
  std::uint32_t lease = 0;  // kLease, kRelease
  std::uint64_t bytes = 0;  // kShrink, kGrow
  // kLease, kUse, kEvict, kRestore: ids[first, first + count) of Trace::ids.
  std::uint32_t first = 0;
  std::uint32_t count = 0;
};

struct Trace {
  std::string file;  // the budget file's name
  std::string role;
  std::string sha256;  // the file's
  std::uint64_t budget = 0;
  std::uint64_t unique = 0;
  std::vector<Model> models;
  std::vector<Group> groups;
  std::vector<Event> events;
  std::vector<GroupId> ids;

  std::span<const GroupId> Ids(const Event& e) const {
    return std::span(ids).subspan(e.first, e.count);
  }
};

// Loads DIR/NAME after checking the manifest and the file against the
// recorded identities. `role` is "primary" or "confirmation".
std::expected<Trace, std::string> LoadTrace(const std::string& dir, std::string_view name,
                                            std::string_view role);

// Parses trace bytes whose identity the caller has checked (tests build
// small traces this way).
std::expected<Trace, std::string> ParseTrace(std::string_view bytes);

// swap_trace.py's Reference: LRU over whole unleased groups at their stored
// bytes. A group's recency is its latest lease, use or release; within one
// event the lower id (earlier in the event's list) is older.
class Reference {
 public:
  Reference(std::span<const Group> groups, std::uint64_t budget);

  struct Step {
    std::vector<GroupId> evicted;
    std::vector<GroupId> restored;
  };
  // Applies an access, shrink or grow, a release or a request; returns the
  // reference's evictions and restores before it. False if it does not fit.
  std::expected<Step, std::string> Apply(const Event& event, std::span<const GroupId> ids);

  // The recency stamp: larger is more recent.
  std::uint64_t Stamp(GroupId g) const { return stamp_[g]; }
  bool Resident(GroupId g) const { return resident_[g]; }
  std::uint64_t limit() const { return budget_ - shrunk_; }
  std::uint64_t used() const { return used_; }

 private:
  std::expected<Step, std::string> Need(std::span<const GroupId> ids);
  void Touch(std::span<const GroupId> ids);

  std::vector<std::uint64_t> size_;
  std::uint64_t budget_ = 0;
  std::uint64_t used_ = 0;
  std::uint64_t shrunk_ = 0;
  std::uint64_t clock_ = 0;
  std::vector<std::uint64_t> stamp_;
  std::vector<bool> resident_;
  std::set<std::pair<std::uint64_t, GroupId>> order_;  // resident, by stamp
  std::vector<std::uint32_t> protected_;
  std::vector<std::uint64_t> wanted_;  // the epoch of the access that wants it
  std::uint64_t epoch_ = 0;
  std::uint64_t protected_bytes_ = 0;
  std::map<std::uint32_t, std::vector<GroupId>> leases_;
};

}  // namespace llmp::rb

#endif  // LLMP_BENCHMARKS_RETAINED_BACKING_TRACE_H_
