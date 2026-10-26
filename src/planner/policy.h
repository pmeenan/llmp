// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// A planning policy (D-053, D-107; docs/engine-components.md#the-graph-ir):
// what the planner may do for one model alias, as data. Structural and
// proved incumbent-equivalent patterns default on. Changed rounding or
// reduction requires authenticated numerical evidence, or an explicitly
// reported diagnostic candidate. Independent lossy and product choices,
// row invariance, step-specific bounds, and measured device ranks remain
// explicit policy inputs.
//
// The text form is canonical, so a policy can live in a spec, an alias's
// settings or a test as bytes, and its digest is a plan key's input
// (step_shape.h PlanKey::policy). One directive a line, each line ending
// in '\n', in this order:
//   max_decode_rows <n>                 always (1 to 65,536)
//   vector_rows <n>                     when set (1 to 65,536)
//   row_invariant                       when set
//   allow_lossy                         when set
//   compact_expert_min_rows <n>         when not 2048 (1 to 65,536)
//   product <choice>                    independent product policies, names ascending
//   candidate_numerics <change>         explicit diagnostic proposals, names ascending
//   enable <pattern>                    known patterns, names ascending
//   disable <pattern>                   a pattern of the table (pattern.h), names ascending
//   when <scope> <test> <n> <action>    step rules, ascending (StepRule's order)
//   rank <class> <implementation> <n>   ascending by class, then rank, then name
// A step rule's scope is any, solo (one owner's rows) or joined (several
// owners'); its test past <n> (more than n rows) or under <n> (fewer); its
// action disable <pattern>, max_decode_rows <m> or vector_rows <m>.
// ParsePolicy takes only canonical text (ToText(ParsePolicy(t)) == t).

#ifndef LLMP_PLANNER_POLICY_H_
#define LLMP_PLANNER_POLICY_H_

#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "base/sha256.h"
#include "execution/registry.h"
#include "ir/numerics.h"

namespace llmp::planner {

inline constexpr std::size_t kMaxPolicyBytes = std::size_t{1} << 20U;
inline constexpr std::size_t kMaxDisabled = 256;
inline constexpr std::size_t kMaxRanks = 4096;
inline constexpr std::uint32_t kMaxDecodeRowsBound = 65536;
inline constexpr std::size_t kMaxStepRules = 256;

// A measured rank of an implementation on a device class (lower first).
struct ClassRank {
  std::string device_class{};  // the DeviceFacts class it was measured on ("sm_121")
  std::string implementation{};
  std::uint32_t rank = 0;
  bool operator==(const ClassRank&) const = default;
};

// Which steps a step rule holds for: every step, a solo step (one
// segment: one owner's rows), or a joined one (several owners' rows); and
// by its rows, past (more than) or under (fewer than) a count.
enum class StepScope : std::uint8_t { kAny, kSolo, kJoined };
enum class RowTest : std::uint8_t { kPast, kUnder };
// What a step rule does: turns a pattern off, or tightens the decode-shaped
// bound or the vector forms' bound to at most its value.
enum class StepAction : std::uint8_t { kDisable, kMaxDecodeRows, kVectorRows };

struct StepRule {
  StepScope scope = StepScope::kAny;
  RowTest test = RowTest::kPast;
  std::uint32_t rows = 0;
  StepAction action = StepAction::kDisable;
  std::string pattern{};    // kDisable's pattern of the table
  std::uint32_t value = 0;  // a bound's (1 to 65,536)
  bool operator==(const StepRule&) const = default;
};

// Independent product policies. Each is default-off and still subject to
// backend shape/device/liveness checks. D2R changes arithmetic and requires
// its own explicit choice; pairing never opts into it.
struct ProductPolicy {
  bool compact_experts = false;
  // Measured crossover, configurable per alias. Omitted from canonical text
  // at the canonical default.
  std::uint32_t compact_expert_min_rows = 2048;
  bool d2r_experts = false;
  bool dense_pair = false;
  bool pair_experts = false;
  bool pair_glu = false;
  bool pair_glu_q8 = false;
  bool vector_floats = false;
  bool operator==(const ProductPolicy&) const = default;
};

// Component declarations and planner requirements share one neutral vocabulary.
using ir::NumericalChange;
using ir::NumericalChangeName;
struct NumericalQualification {
  NumericalChange change{};
  base::Sha256Digest evidence{}, source{}, semantic{};
  std::uint16_t stage = 0, component = 0;
  ir::OpKey anchor{};
  std::uint32_t arch = 0;
  // Bounds within this exact admitted source/spec envelope, not evidence
  // for unrelated smaller graphs. exact_shape restricts rows/K/N to one
  // measured geometry instead of an explicitly qualified envelope.
  std::uint64_t min_rows = 1, max_rows = 1, max_k = 0, max_n = 0;
  bool exact_shape = false;
  // Role-bound incumbent recipes keep their exact K/N over the accepted row
  // envelope. A measured one-shape candidate additionally fixes its rows.
  bool exact_dimensions = false;
  bool incumbent_recipe = false;
  std::uint16_t weight_role = UINT16_MAX;
  bool operator==(const NumericalQualification&) const = default;
};
class NumericalQualifications {
 public:
  std::span<const NumericalQualification> entries() const { return entries_; }
  std::uint64_t host_bytes() const;
  const base::Sha256Digest& digest() const { return digest_; }
  // Caller prepays the exact entries/control inventory before this copy.
  // Admission supplies independently authenticated active identities; zero
  // or different receipts/specs refuse before any retained allocation.
  static std::expected<std::shared_ptr<const NumericalQualifications>, std::string> Bind(
      std::span<const NumericalQualification> entries, const base::Sha256Digest& active_source,
      const base::Sha256Digest& active_semantic, std::shared_ptr<const void> paid_owner = {});
  static std::expected<std::shared_ptr<const NumericalQualifications>, std::string> Bind(
      std::vector<NumericalQualification>&& entries, const base::Sha256Digest& active_source,
      const base::Sha256Digest& active_semantic, std::shared_ptr<const void> paid_owner = {});

 private:
  // Payment outlives the retained entries on every failed/retired path.
  std::shared_ptr<const void> paid_owner_;
  std::vector<NumericalQualification> entries_;
  base::Sha256Digest digest_{};
};
std::expected<std::uint64_t, std::string> NumericalQualificationBytes(std::size_t count);

struct Policy {
  // Explicit enables, ascending and unique. Enabling an already-default
  // pattern is idempotent; an explicit disable may not name the same pattern.
  std::vector<std::string> enabled{};
  std::vector<std::string> disabled{};  // pattern names, ascending, each once
  std::uint32_t max_decode_rows = 16;
  bool row_invariant = false;
  bool allow_lossy = false;
  std::vector<ClassRank> ranks{};  // ascending by (class, rank, implementation)
  // The most rows the row-at-a-time forms take (execution::SelectionPolicy
  // vector_rows); zero: each one's own bound.
  std::uint32_t vector_rows = 0;
  // Rules for some steps only, in StepRule order, each once; ForStep
  // applies them.
  std::vector<StepRule> rules{};
  ProductPolicy products{};
  // Explicit diagnostic arithmetic proposals. They are never evidence and
  // never enabled by production mappings; applied plans report them as
  // UNQUALIFIED_CANDIDATE until a separate accepted measurement grants them.
  std::vector<NumericalChange> numerical_candidates{};
  // Authenticated immutable facts shared by all plans of an admitted source.
  // They are never parsed from alias policy text. Digest includes their seal.
  std::shared_ptr<const NumericalQualifications> numerical{};
  std::uint16_t active_stage = UINT16_MAX;
  bool operator==(const Policy&) const = default;
};

// Whether `pattern` is off under the policy: disabled, or opt-in and not
// enabled.
bool Disabled(const Policy& policy, std::string_view pattern);

// Sorts the lists ToText writes in order; it does not validate.
void Canonicalize(Policy& policy);

// The policy's own rules: names and classes of [a-z0-9_.] (classes
// without '.'), patterns of the table, each once (not both enabled and
// disabled), an implementation ranked once a class, the
// bounds above.
std::expected<void, std::string> Validate(const Policy& policy);

std::string ToText(const Policy& policy);
std::expected<Policy, std::string> ParsePolicy(std::string_view text);

// The policy a step of `rows` rows over `segments` segments plans with:
// each of the policy's step rules that holds for it applied (a disabled
// pattern off even where enabled; a bound the least of the policy's and
// the rules'), and no rules left, validated (Validate). Without rules, the
// policy itself.
std::expected<Policy, std::string> ForStep(const Policy& policy, std::uint64_t rows,
                                           std::size_t segments);

// SHA-256 of canonical text plus the authenticated numerical seal and active stage.
base::Sha256Digest Digest(const Policy& policy);

// The ranks measured on `device_class`, as Candidates() takes them; they
// point into `policy`.
std::vector<execution::MeasuredRank> RanksFor(const Policy& policy, std::string_view device_class);

}  // namespace llmp::planner

#endif  // LLMP_PLANNER_POLICY_H_
