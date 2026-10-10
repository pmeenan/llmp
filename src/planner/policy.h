// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// A planning policy (D-053, D-107; docs/engine-components.md#the-graph-ir):
// what the planner may do for one model alias, as data. The patterns it
// leaves off (a family's first migration reproduces its current fusion
// selection: Gemma 2 serves without the fused norm-rope, Gemma 3 with it),
// the decode-shaped product bound (D-108), whether every plan must be
// row-invariant (D-092), the lossy opt-in, and measured ranks per device
// class, which order the registry's candidates before the format ladder
// does.
//
// The text form is canonical, so a policy can live in a spec, an alias's
// settings or a test as bytes, and its digest is a plan key's input
// (step_shape.h PlanKey::policy). One directive a line, each line ending
// in '\n', in this order:
//   max_decode_rows <n>                 always (1 to 65,536)
//   row_invariant                       when set
//   allow_lossy                         when set
//   enable <pattern>                    an opt-in pattern of the table, names ascending
//   disable <pattern>                   a pattern of the table (pattern.h), names ascending
//   rank <class> <implementation> <n>   ascending by class, then rank, then name
// ParsePolicy takes only canonical text (ToText(ParsePolicy(t)) == t).

#ifndef LLMP_PLANNER_POLICY_H_
#define LLMP_PLANNER_POLICY_H_

#include <cstddef>
#include <cstdint>
#include <expected>
#include <string>
#include <string_view>
#include <vector>

#include "base/sha256.h"
#include "execution/registry.h"

namespace llmp::planner {

inline constexpr std::size_t kMaxPolicyBytes = std::size_t{1} << 20U;
inline constexpr std::size_t kMaxDisabled = 256;
inline constexpr std::size_t kMaxRanks = 4096;
inline constexpr std::uint32_t kMaxDecodeRowsBound = 65536;

// A measured rank of an implementation on a device class (lower first).
struct ClassRank {
  std::string device_class{};  // the DeviceFacts class it was measured on ("sm_121")
  std::string implementation{};
  std::uint32_t rank = 0;
  bool operator==(const ClassRank&) const = default;
};

struct Policy {
  // Opt-in patterns it enables (pattern.h Pattern::opt_in), ascending, each
  // once.
  std::vector<std::string> enabled{};
  std::vector<std::string> disabled{};  // pattern names, ascending, each once
  std::uint32_t max_decode_rows = 16;
  bool row_invariant = false;
  bool allow_lossy = false;
  std::vector<ClassRank> ranks{};  // ascending by (class, rank, implementation)
  bool operator==(const Policy&) const = default;
};

// Whether `pattern` is off under the policy: disabled, or opt-in and not
// enabled.
bool Disabled(const Policy& policy, std::string_view pattern);

// Sorts the lists ToText writes in order; it does not validate.
void Canonicalize(Policy& policy);

// The policy's own rules: names and classes of [a-z0-9_.] (classes
// without '.'), patterns of the table, each once (an enabled one opt-in,
// and not also disabled), an implementation ranked once a class, the
// bounds above.
std::expected<void, std::string> Validate(const Policy& policy);

std::string ToText(const Policy& policy);
std::expected<Policy, std::string> ParsePolicy(std::string_view text);

// SHA-256 of ToText.
base::Sha256Digest Digest(const Policy& policy);

// The ranks measured on `device_class`, as Candidates() takes them; they
// point into `policy`.
std::vector<execution::MeasuredRank> RanksFor(const Policy& policy, std::string_view device_class);

}  // namespace llmp::planner

#endif  // LLMP_PLANNER_POLICY_H_
